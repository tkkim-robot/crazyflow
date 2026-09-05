"""Known-model waypoint detour witnesses for physically achievable test regimes.

This helper executes an explicit route using the same direct-wrench plant and motor allocator as
the experiment. It checks the realized trajectory after integration. A successful run is one
feasible numerical witness; a failed route does not establish that the task is impossible. No
BPTT, policy-library certificate, or online safety filter is used here.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.continuous_demo_scenarios import model_with_wind, scenario_true_wind
from crazyflow.safety.da_plcbf.direct_wrench import wrench_to_motor_forces
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig, waypoint_nominal_wrench
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from crazyflow.safety.da_plcbf.continuous_demo_scenarios import ContinuousDemoScenario
    from crazyflow.safety.da_plcbf.online_constant_wind import VersionAResources
    from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel


@dataclass(frozen=True, slots=True)
class FeasibilityReferenceConfig:
    """Explicit route-following settings and numerical acceptance tolerances."""

    acceleration_limit: float = 1.2
    position_gain: float = 2.0
    velocity_gain: float = 2.8
    waypoint_tolerance_m: float = 0.12
    waypoint_speed_tolerance_mps: float = 0.35
    goal_tolerance_m: float = 0.25
    goal_speed_tolerance_mps: float = 0.20
    goal_hold_steps: int = 10
    arena_clearance_m: float = 0.08
    motor_tolerance_N: float = 3e-6
    state_limit_tolerance: float = 2e-5

    def validate(self) -> None:
        positive = (
            self.acceleration_limit,
            self.position_gain,
            self.velocity_gain,
            self.waypoint_tolerance_m,
            self.waypoint_speed_tolerance_mps,
            self.goal_tolerance_m,
            self.goal_speed_tolerance_mps,
        )
        if not all(math.isfinite(x) and x > 0 for x in positive):
            raise ValueError("reference controller scales must be positive and finite")
        if (
            isinstance(self.goal_hold_steps, bool)
            or not isinstance(self.goal_hold_steps, int)
            or self.goal_hold_steps < 1
        ):
            raise ValueError("goal_hold_steps must be a positive integer")
        if not all(
            math.isfinite(x) and x >= 0
            for x in (self.arena_clearance_m, self.motor_tolerance_N, self.state_limit_tolerance)
        ):
            raise ValueError("reference clearances/tolerances must be nonnegative and finite")


@dataclass(frozen=True, slots=True)
class FeasibilityReferenceResult:
    """Complete witness trajectory; node arrays include the final integrated state."""

    time_seconds: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    motor_forces: np.ndarray
    waypoint_indices: np.ndarray
    model_parameters: dict[str, np.ndarray]
    ego_radii: np.ndarray
    summary: dict[str, Any]


def _collision_clearance(
    positions: np.ndarray, centers: np.ndarray, obstacle_radii: np.ndarray, ego_radii: np.ndarray
) -> tuple[float | None, int | None]:
    if not obstacle_radii.size:
        return None, None
    relative = positions[:, None, :] - centers
    node_clearance = np.linalg.norm(relative, axis=-1) - obstacle_radii - ego_radii[:, None]
    if len(positions) == 1:
        return float(np.min(node_clearance)), 0
    delta = np.diff(relative, axis=0)
    squared = np.sum(delta * delta, axis=-1)
    fraction = np.clip(
        -np.sum(relative[:-1] * delta, axis=-1) / np.maximum(squared, 1e-20), 0.0, 1.0
    )
    swept = np.linalg.norm(relative[:-1] + fraction[..., None] * delta, axis=-1)
    swept = swept - obstacle_radii - ego_radii[:-1, None]
    # Parameter/radius switches take effect at interval boundaries; check both neighboring nodes.
    per_interval = np.minimum(np.min(swept, axis=1), np.min(node_clearance[1:], axis=1))
    return min(float(np.min(node_clearance[0])), float(np.min(per_interval))), int(
        np.argmin(per_interval)
    )


def run_feasibility_reference(
    scenario: ContinuousDemoScenario,
    resources: VersionAResources,
    state: jax.Array,
    *,
    start_step: int,
    waypoints: Sequence[Sequence[float]] | np.ndarray | jax.Array,
    max_steps: int | None = None,
    config: FeasibilityReferenceConfig = FeasibilityReferenceConfig(),
    model_at_step: Callable[[int], VersionAModel] | None = None,
    ego_radius_at_step: Callable[[int], float] | None = None,
    device: jax.Device | None = None,
) -> FeasibilityReferenceResult:
    """Execute supplied detour waypoints, then the scenario goal, with known-model feedforward.

    Callbacks receive the absolute plant integration index. The exact model used by the plant is
    also supplied to the nominal controller; this is deliberately an oracle feasibility witness.
    The final scenario goal is appended when absent from the explicit waypoint list. The helper
    never hides a state/motor/clearance violation or converts route failure into impossibility.
    """
    config.validate()
    scenario.validate()
    if isinstance(start_step, bool) or not isinstance(start_step, int) or start_step < 0:
        raise ValueError("start_step must be a nonnegative integer")
    if max_steps is None:
        max_steps = scenario.steps - start_step
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    initial_state = np.asarray(state)
    if initial_state.shape != (13,) or not np.all(np.isfinite(initial_state)):
        raise ValueError("state must have 13 finite components")
    route = np.asarray(waypoints, dtype=initial_state.dtype)
    if route.size == 0:
        route = np.empty((0, 3), dtype=initial_state.dtype)
    if route.ndim != 2 or route.shape[1] != 3 or not np.all(np.isfinite(route)):
        raise ValueError("waypoints must be finite shape (N, 3)")
    goal = np.asarray(scenario.goal_position)
    if not len(route) or not np.allclose(route[-1], goal, atol=1e-7, rtol=0.0):
        route = np.concatenate((route, goal[None, :]), axis=0)
    selected_device = jax.devices()[0] if device is None else device
    current = jax.device_put(state, selected_device)
    actuator = jax.device_put(resources.actuator, selected_device)
    route_device = jax.device_put(jnp.asarray(route), selected_device)
    quad_config = QuadPolicyConfig(acceleration_limit=config.acceleration_limit)
    goal_velocity = jax.device_put(scenario.goal_velocity, selected_device)

    @jax.jit
    def advance(
        current_state: jax.Array,
        target: jax.Array,
        target_velocity: jax.Array,
        model: VersionAModel,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        command = waypoint_nominal_wrench(
            current_state,
            target,
            target_velocity,
            model,
            actuator,
            quad_config,
            position_gain=config.position_gain,
            velocity_gain=config.velocity_gain,
            model_compensation=True,
        )
        following = direct_wrench_symplectic_step(current_state, command.wrench, model, scenario.dt)
        motors = wrench_to_motor_forces(
            command.wrench,
            L=actuator.arm_length,
            thrust2torque=actuator.thrust_to_torque,
            mixing_matrix=actuator.mixing_matrix,
        )
        return following, command.wrench, motors, command.input_valid

    def point_model(index: int) -> VersionAModel:
        return (
            model_with_wind(resources.model, scenario_true_wind(scenario, index))
            if model_at_step is None
            else model_at_step(index)
        )

    def radius(index: int) -> float:
        value = (
            scenario.ego_radius if ego_radius_at_step is None else float(ego_radius_at_step(index))
        )
        if not math.isfinite(value) or value < 0:
            raise ValueError("every prescribed ego radius must be nonnegative and finite")
        return value

    recorded_states = [initial_state.copy()]
    actions, motor_forces, targets = [], [], []
    models: dict[str, list[np.ndarray]] = {name: [] for name in resources.model._fields}
    radii = [radius(start_step)]
    target_index = 0
    goal_hold = 0
    failure_at_step = None
    for offset in range(max_steps):
        index = start_step + offset
        position = np.asarray(current[:3])
        speed = float(np.linalg.norm(np.asarray(current[7:10])))
        while (
            target_index < len(route) - 1
            and np.linalg.norm(position - route[target_index]) <= config.waypoint_tolerance_m
            and speed <= config.waypoint_speed_tolerance_mps
        ):
            target_index += 1
        model = jax.device_put(point_model(index), selected_device)
        target_velocity = (
            goal_velocity if target_index == len(route) - 1 else jnp.zeros_like(goal_velocity)
        )
        following, command, motors, valid = advance(
            current, route_device[target_index], target_velocity, model
        )
        jax.block_until_ready((following, command, motors, valid))
        if (
            not bool(np.asarray(valid))
            or not np.all(np.isfinite(np.asarray(following)))
            or not np.all(np.isfinite(np.asarray(command)))
            or not np.all(np.isfinite(np.asarray(motors)))
        ):
            failure_at_step = index
            break
        recorded_states.append(np.asarray(following))
        actions.append(np.asarray(command))
        motor_forces.append(np.asarray(motors))
        targets.append(target_index)
        for name in models:
            models[name].append(np.asarray(getattr(model, name)))
        radii.append(radius(index + 1))
        current = following
        close = (
            target_index == len(route) - 1
            and np.linalg.norm(np.asarray(current[:3]) - goal) <= config.goal_tolerance_m
        )
        settled = (
            np.linalg.norm(np.asarray(current[7:10]) - np.asarray(goal_velocity))
            <= config.goal_speed_tolerance_mps
        )
        goal_hold = goal_hold + 1 if close and settled else 0
        if goal_hold >= config.goal_hold_steps:
            break
    states = np.asarray(recorded_states)
    action_array = np.asarray(actions, dtype=states.dtype).reshape(-1, 4)
    motors = np.asarray(motor_forces, dtype=states.dtype).reshape(-1, 4)
    time_seconds = (start_step + np.arange(len(states))) * scenario.dt
    ego_radii = np.asarray(radii)
    active = np.flatnonzero(np.asarray(scenario.obstacle_mask))
    centers = (
        np.asarray(scenario.obstacle_initial_centers)[None, active, :]
        + time_seconds[:, None, None] * np.asarray(scenario.obstacle_velocities)[None, active, :]
    )
    physical_clearance, closest_index = _collision_clearance(
        states[:, :3], centers, np.asarray(scenario.obstacle_radii)[active], ego_radii
    )
    shell_clearance = (
        None if physical_clearance is None else physical_clearance - scenario.obstacle_clearance
    )
    quaternion = states[:, 3:7]
    quaternion = quaternion / np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12)
    tilt = np.arccos(
        np.clip(1.0 - 2.0 * (quaternion[:, 0] ** 2 + quaternion[:, 1] ** 2), -1.0, 1.0)
    )
    speeds = np.linalg.norm(states[:, 7:10], axis=-1)
    angular_rates = np.linalg.norm(states[:, 10:13], axis=-1)
    arena_margin = float(
        min(
            np.min(states[:, :3] - np.asarray(scenario.arena_lower) - config.arena_clearance_m),
            np.min(np.asarray(scenario.arena_upper) - config.arena_clearance_m - states[:, :3]),
        )
    )
    lower = np.broadcast_to(np.asarray(actuator.thrust_min), (4,))
    upper = np.broadcast_to(np.asarray(actuator.thrust_max), (4,))
    motor_margin = (
        None if not len(motors) else float(min(np.min(motors - lower), np.min(upper - motors)))
    )
    final_goal_distance = float(np.linalg.norm(states[-1, :3] - goal))
    goal_reached = goal_hold >= config.goal_hold_steps
    tolerance = config.state_limit_tolerance
    checks = {
        "finite_execution": failure_at_step is None,
        "physical_collision_clear": physical_clearance is None or physical_clearance > 0.0,
        "inflated_shell_clear": shell_clearance is None or shell_clearance > 0.0,
        "motor_limits": motor_margin is not None and motor_margin >= -config.motor_tolerance_N,
        "arena_limits": arena_margin >= -tolerance,
        "speed_limit": float(np.max(speeds)) <= float(np.asarray(scenario.speed_max)) + tolerance,
        "tilt_limit": float(np.max(tilt))
        <= float(np.asarray(scenario.tilt_max_radians)) + tolerance,
        "angular_rate_limit": float(np.max(angular_rates))
        <= float(np.asarray(scenario.angular_rate_max)) + tolerance,
        "goal_reached_and_settled": goal_reached,
    }
    summary = {
        "purpose": "known-model physically achievable detour witness; not a PL-CBF certificate",
        "interpretation": (
            "success proves one sampled numerical route; failure does not prove impossibility"
        ),
        "device": str(selected_device),
        "config": asdict(config),
        "scenario": {
            name: np.asarray(value).tolist()
            if isinstance(value, (jax.Array, np.ndarray))
            else value
            for name, value in asdict(scenario).items()
        },
        "nominal_model_compensation": True,
        "start_step": start_step,
        "integration_dt_seconds": scenario.dt,
        "requested_max_steps": max_steps,
        "executed_steps": len(actions),
        "commanded_waypoints": route.tolist(),
        "final_waypoint_index": target_index,
        "failure_at_absolute_step": failure_at_step,
        "checks": checks,
        "feasible_witness_found": all(checks.values()),
        "minimum_physical_clearance_m": physical_clearance,
        "minimum_inflated_clearance_m": shell_clearance,
        "closest_obstacle_time_seconds": None
        if closest_index is None
        else float(time_seconds[closest_index]),
        "minimum_motor_margin_N": motor_margin,
        "minimum_arena_center_margin_m": arena_margin,
        "arena_scope": (
            "center position with the same explicit arena clearance as the safety filter"
        ),
        "maximum_speed_mps": float(np.max(speeds)),
        "maximum_tilt_radians": float(np.max(tilt)),
        "maximum_angular_rate_rps": float(np.max(angular_rates)),
        "final_goal_distance_m": final_goal_distance,
        "final_speed_mps": float(speeds[-1]),
        "physical_configuration": {
            "actuator": {
                name: np.asarray(value).tolist() for name, value in actuator._asdict().items()
            },
            "initial_model": {
                name: np.asarray(value).tolist()
                for name, value in point_model(start_step)._asdict().items()
            },
            "ego_radius_initial_m": float(ego_radii[0]),
            "ego_radius_final_m": float(ego_radii[-1]),
        },
    }
    model_arrays = {
        name: np.asarray(values)
        if values
        else np.empty((0, *np.asarray(getattr(resources.model, name)).shape))
        for name, values in models.items()
    }
    return FeasibilityReferenceResult(
        time_seconds,
        states,
        action_array,
        motors,
        np.asarray(targets, dtype=np.int32),
        model_arrays,
        ego_radii,
        summary,
    )


def save_feasibility_reference(
    result: FeasibilityReferenceResult, path_stem: str | Path
) -> tuple[Path, Path]:
    """Save all realized states/actions/model parameters, including unsuccessful witnesses."""
    stem = Path(path_stem)
    npz_path, json_path = Path(f"{stem}.npz"), Path(f"{stem}.json")
    if npz_path.exists() or json_path.exists():
        raise FileExistsError("refusing to overwrite an existing feasibility witness")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "time_seconds": result.time_seconds,
        "states": result.states,
        "actions": result.actions,
        "motor_forces": result.motor_forces,
        "waypoint_indices": result.waypoint_indices,
        "ego_radii": result.ego_radii,
        **{f"model_{name}": value for name, value in result.model_parameters.items()},
    }
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return npz_path, json_path


__all__ = [
    "FeasibilityReferenceConfig",
    "FeasibilityReferenceResult",
    "run_feasibility_reference",
    "save_feasibility_reference",
]
