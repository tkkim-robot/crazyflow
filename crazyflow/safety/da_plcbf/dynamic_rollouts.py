"""Finite-scenario moving-sphere rollouts for DA-PLCBF.

Dynamic obstacles are exogenous predicted trajectories.  Each prediction is kept as an explicit
``R`` axis; it is never averaged into one nominal path.  The shared fallback actor observes the
current predicted obstacle locations, while exact hard policy values take the minimum over every
enabled constraint, time node, swept relative-motion segment, and finite prediction scenario.

The result is robust only to the recorded finite prediction set.  It is not a distribution-free or
continuous uncertainty guarantee.  Runtime forecasts are predeclared exogenous oracle forecasts,
not sensor-derived predictions.  A dynamic slot is exposed only after it is active at the current
observation boundary, so an unreleased ballistic obstacle cannot reveal its future release schedule.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp

from crazyflow.safety.da_plcbf.direct_wrench import quaternion_to_rotation_matrix
from crazyflow.safety.da_plcbf.quad_policy import shared_quad_fallback_wrenches
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.actor import (
        SharedActorConfig,
        SharedActorParams,
        SharedActorSpec,
    )
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
    from crazyflow.safety.da_plcbf.scenarios import ScenarioTape
    from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig, VersionAModel
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


DYNAMIC_PREDICTION_CONTRACT = "predeclared-exogenous-oracle-forecast-observed-active-slots-only-v1"
"""Machine-readable availability and provenance contract for runtime obstacle forecasts."""


class DynamicSphereScenarioBatch(NamedTuple):
    """Time-varying sphere predictions with shapes ``[B,R,T,O,...]``."""

    obstacle_centers: Array
    obstacle_radii: Array
    obstacle_mask: Array
    arena_lower: Array
    arena_upper: Array
    speed_limit: Array
    angular_rate_max: Array
    tilt_max_radians: Array


class DynamicQuadRolloutBatch(NamedTuple):
    """Shared-policy rollout traces with leading axes ``[K,B,R,T]``."""

    states: Array
    wrenches: Array
    desired_accelerations: Array
    raw_motor_forces: Array
    bounded_motor_forces: Array
    policy_valid: Array


class DynamicSafetyValues(NamedTuple):
    """Node, swept, per-prediction, and robust hard/smooth margins."""

    node_values: Array
    node_enabled: Array
    segment_obstacle_values: Array
    segment_obstacle_enabled: Array
    input_valid: Array
    prediction_hard_margins: Array
    robust_hard_margins: Array
    robust_smooth_margins: Array


class DynamicLibraryEvaluation(NamedTuple):
    """Finite-prediction hard values and common first motor-force fallbacks."""

    rollouts: DynamicQuadRolloutBatch
    safety_values: DynamicSafetyValues
    hard_values: Array
    first_motor_forces: Array
    first_wrenches: Array
    first_action_consistent: Array
    policy_valid: Array


def validate_dynamic_sphere_batch(scenarios: DynamicSphereScenarioBatch) -> None:
    """Validate fixed shapes and host values before tracing an expensive rollout."""
    centers = np.asarray(scenarios.obstacle_centers)
    radii = np.asarray(scenarios.obstacle_radii)
    mask = np.asarray(scenarios.obstacle_mask)
    if centers.ndim != 5 or centers.shape[-1] != 3:
        raise ValueError("obstacle_centers must have shape (B, R, T, O, 3)")
    batch, predictions, nodes, obstacles, _ = centers.shape
    if min(batch, predictions, nodes) <= 0 or nodes < 2:
        raise ValueError("B and R must be positive and T must contain at least two nodes")
    if radii.shape != (batch, predictions, nodes, obstacles):
        raise ValueError("obstacle_radii must have shape (B, R, T, O)")
    if mask.shape != radii.shape or mask.dtype != np.bool_:
        raise ValueError("obstacle_mask must be boolean with shape (B, R, T, O)")
    expected = {
        "arena_lower": (batch, 3),
        "arena_upper": (batch, 3),
        "speed_limit": (batch,),
        "angular_rate_max": (batch,),
        "tilt_max_radians": (batch,),
    }
    for name, shape in expected.items():
        if np.asarray(getattr(scenarios, name)).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    real_centers = centers[mask]
    real_radii = radii[mask]
    if not np.all(np.isfinite(real_centers)):
        raise ValueError("enabled moving-sphere centers must be finite")
    if not np.all(np.isfinite(real_radii) & (real_radii > 0)):
        raise ValueError("enabled moving-sphere radii must be finite and positive")
    lower = np.asarray(scenarios.arena_lower)
    upper = np.asarray(scenarios.arena_upper)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper) & (upper > lower)):
        raise ValueError("arena bounds must be finite and strictly ordered")
    speed = np.asarray(scenarios.speed_limit)
    angular = np.asarray(scenarios.angular_rate_max)
    tilt = np.asarray(scenarios.tilt_max_radians)
    if not np.all(np.isfinite(speed) & (speed > 0)):
        raise ValueError("speed_limit must be finite and positive")
    if not np.all(np.isfinite(angular) & (angular > 0)):
        raise ValueError("angular_rate_max must be finite and positive")
    if not np.all(np.isfinite(tilt) & (tilt > 0) & (tilt <= 0.5 * np.pi)):
        raise ValueError("tilt_max_radians must lie in (0, pi/2]")


def _validate_dynamic_sphere_shapes(scenarios: DynamicSphereScenarioBatch) -> None:
    """Perform tracer-safe structural validation inside public numerical functions."""
    centers = scenarios.obstacle_centers
    if centers.ndim != 5 or centers.shape[-1] != 3:
        raise ValueError("obstacle_centers must have shape (B, R, T, O, 3)")
    batch, predictions, nodes, obstacles, _ = centers.shape
    if min(batch, predictions, nodes) <= 0 or nodes < 2:
        raise ValueError("B and R must be positive and T must contain at least two nodes")
    shape = (batch, predictions, nodes, obstacles)
    if scenarios.obstacle_radii.shape != shape:
        raise ValueError("obstacle_radii must have shape (B, R, T, O)")
    if scenarios.obstacle_mask.shape != shape or not jnp.issubdtype(
        scenarios.obstacle_mask.dtype, jnp.bool_
    ):
        raise ValueError("obstacle_mask must be boolean with shape (B, R, T, O)")
    expected = {
        "arena_lower": (batch, 3),
        "arena_upper": (batch, 3),
        "speed_limit": (batch,),
        "angular_rate_max": (batch,),
        "tilt_max_radians": (batch,),
    }
    for name, expected_shape in expected.items():
        if getattr(scenarios, name).shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")


def dynamic_sphere_window_from_tape(
    tape: ScenarioTape,
    *,
    start_index: int,
    horizon: int,
    speed_limit: float,
    angular_rate_max: float,
    tilt_max_radians: float,
    include_static: bool = True,
) -> DynamicSphereScenarioBatch:
    """Build one ``B=1`` prediction window from an immutable scenario tape.

    The window has ``horizon + 1`` nodes.  Requests near the end of a tape are rejected rather than
    padded with a fabricated stationary future.
    """
    from crazyflow.safety.da_plcbf.scenarios import ScenarioTape

    if not isinstance(tape, ScenarioTape):
        raise TypeError("tape must be a ScenarioTape")
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a nonnegative integer")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    stop = start_index + horizon + 1
    if stop > tape.steps:
        raise ValueError("requested prediction window extends past the scenario tape")
    if not math.isfinite(speed_limit) or speed_limit <= 0:
        raise ValueError("speed_limit must be finite and positive")
    if not math.isfinite(angular_rate_max) or angular_rate_max <= 0:
        raise ValueError("angular_rate_max must be finite and positive")
    if (
        not math.isfinite(tilt_max_radians)
        or tilt_max_radians <= 0
        or tilt_max_radians > 0.5 * math.pi
    ):
        raise ValueError("tilt_max_radians must lie in (0, pi/2]")
    if not isinstance(include_static, bool):
        raise TypeError("include_static must be boolean")

    predictions = tape.prediction_samples
    nodes = horizon + 1
    dynamic_centers = np.array(tape.prediction_positions[:, start_index:stop], copy=True)
    # Predictions are generated from release/initial time. At an online decision boundary the
    # present obstacle position is observed, so translate each hypothesis to that common current
    # observation while preserving its predicted relative motion. This prevents the actor from
    # choosing an R-specific current command based on mutually contradictory present worlds.
    current_offsets = (
        tape.dynamic_positions[start_index][None, :, :] - tape.prediction_positions[:, start_index]
    )
    dynamic_centers += current_offsets[:, None, :, :]
    dynamic_radii = np.broadcast_to(
        tape.dynamic_radii[None, None, :], (predictions, nodes, tape.dynamic_radii.size)
    )
    future_dynamic_mask = np.broadcast_to(
        tape.dynamic_time_mask[start_index:stop][None, :, :], dynamic_radii.shape
    ) & np.broadcast_to(tape.dynamic_slot_mask[None, None, :], dynamic_radii.shape)
    # A scenario tape contains the full predeclared oracle forecast for reproducibility, but the
    # controller may consume only slots observed active at this decision boundary.  In particular,
    # a ballistic slot that has not yet been released must not reveal its future activation time,
    # release point, or velocity support.  Once observed, the recorded finite forecast is supplied
    # exogenously for the rest of this horizon.
    observed_active = tape.dynamic_time_mask[start_index] & tape.dynamic_slot_mask
    dynamic_mask = future_dynamic_mask & np.broadcast_to(
        observed_active[None, None, :], dynamic_radii.shape
    )

    if include_static:
        static_centers = np.broadcast_to(
            tape.static_positions[None, None, :, :],
            (predictions, nodes, tape.static_positions.shape[0], 3),
        )
        static_radii = np.broadcast_to(
            tape.static_radii[None, None, :], (predictions, nodes, tape.static_radii.size)
        )
        static_mask = np.broadcast_to(tape.static_mask[None, None, :], static_radii.shape)
        centers = np.concatenate((static_centers, dynamic_centers), axis=2)
        radii = np.concatenate((static_radii, dynamic_radii), axis=2)
        mask = np.concatenate((static_mask, dynamic_mask), axis=2)
    else:
        centers, radii, mask = dynamic_centers, dynamic_radii, dynamic_mask

    result = DynamicSphereScenarioBatch(
        obstacle_centers=jnp.asarray(centers[None, ...], dtype=jnp.float32),
        obstacle_radii=jnp.asarray(radii[None, ...], dtype=jnp.float32),
        obstacle_mask=jnp.asarray(mask[None, ...]),
        arena_lower=jnp.asarray(tape.arena_lower[None, :], dtype=jnp.float32),
        arena_upper=jnp.asarray(tape.arena_upper[None, :], dtype=jnp.float32),
        speed_limit=jnp.asarray([speed_limit], dtype=jnp.float32),
        angular_rate_max=jnp.asarray([angular_rate_max], dtype=jnp.float32),
        tilt_max_radians=jnp.asarray([tilt_max_radians], dtype=jnp.float32),
    )
    validate_dynamic_sphere_batch(result)
    return result


def rollout_shared_quad_dynamic_library(
    params: SharedActorParams,
    spec: SharedActorSpec,
    initial_states: Array,
    scenarios: DynamicSphereScenarioBatch,
    model: VersionAModel,
    actuator: VersionAActuator,
    *,
    dt: float,
    policy_gain: float,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
) -> DynamicQuadRolloutBatch:
    """Roll out every policy under every finite moving-obstacle prediction scenario."""
    _validate_dynamic_sphere_shapes(scenarios)
    if initial_states.ndim != 2 or initial_states.shape != (
        scenarios.obstacle_centers.shape[0],
        13,
    ):
        raise ValueError("initial_states must have shape (B, 13)")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(policy_gain) or policy_gain <= 0:
        raise ValueError("policy_gain must be finite and positive")
    policy_count = spec.base_codes.shape[0]
    batch, predictions, nodes, _, _ = scenarios.obstacle_centers.shape
    horizon = nodes - 1
    current = jnp.broadcast_to(
        initial_states[None, :, None, :], (policy_count, batch, predictions, 13)
    )
    horizon_duration = horizon * dt

    def advance(state: Array, step_index: Array) -> tuple[Array, tuple[Array, ...]]:
        step_centers = scenarios.obstacle_centers[:, :, step_index]
        step_radii = scenarios.obstacle_radii[:, :, step_index]
        step_mask = scenarios.obstacle_mask[:, :, step_index]
        flat_scenarios = CircleScenarioBatch(
            obstacle_centers=step_centers.reshape(batch * predictions, *step_centers.shape[-2:]),
            obstacle_radii=step_radii.reshape(batch * predictions, step_radii.shape[-1]),
            obstacle_mask=step_mask.reshape(batch * predictions, step_mask.shape[-1]),
            arena_lower=jnp.repeat(scenarios.arena_lower, predictions, axis=0),
            arena_upper=jnp.repeat(scenarios.arena_upper, predictions, axis=0),
            speed_limit=jnp.repeat(scenarios.speed_limit, predictions, axis=0),
        )
        flat_state = state.reshape(policy_count, batch * predictions, 13)
        command = shared_quad_fallback_wrenches(
            params,
            spec,
            flat_state,
            flat_scenarios,
            model,
            actuator,
            elapsed=step_index * dt,
            horizon_duration=horizon_duration,
            policy_gain=policy_gain,
            actor_config=actor_config,
            quad_config=quad_config,
        )
        wrench = command.wrench.reshape(policy_count, batch, predictions, 4)
        following = direct_wrench_symplectic_step(state, wrench, model, dt)
        output_shape = (policy_count, batch, predictions)
        return following, (
            following,
            wrench,
            command.desired_acceleration.reshape(*output_shape, 3),
            command.raw_motor_forces.reshape(*output_shape, 4),
            command.bounded_motor_forces.reshape(*output_shape, 4),
            command.input_valid.reshape(output_shape),
        )

    _, output = jax.lax.scan(advance, current, jnp.arange(horizon, dtype=jnp.int32))
    future, wrench, acceleration, raw_motor, bounded_motor, valid = (
        jnp.moveaxis(value, 0, 3) for value in output
    )
    states = jnp.concatenate((current[:, :, :, None, :], future), axis=3)
    return DynamicQuadRolloutBatch(states, wrench, acceleration, raw_motor, bounded_motor, valid)


def dynamic_quad_safety_values(
    states: Array,
    scenarios: DynamicSphereScenarioBatch,
    barrier_config: VersionABarrierConfig,
    *,
    softmin_beta: float,
) -> DynamicSafetyValues:
    """Reduce hard physical values over time and finite predictions without averaging ``R``."""
    barrier_config.validate()
    _validate_dynamic_sphere_shapes(scenarios)
    expected = (*scenarios.obstacle_centers.shape[:3], 13)
    if states.ndim != 5 or states.shape[1:4] != expected[:3] or states.shape[-1] != 13:
        raise ValueError("states must have shape (K, B, R, T, 13) matching scenarios")
    if not math.isfinite(softmin_beta) or softmin_beta <= 0:
        raise ValueError("softmin_beta must be finite and positive")

    dtype = states.dtype
    mask = scenarios.obstacle_mask
    safe_centers = jnp.where(
        mask[..., None] & jnp.isfinite(scenarios.obstacle_centers), scenarios.obstacle_centers, 0.0
    )
    safe_radii = jnp.where(
        mask & jnp.isfinite(scenarios.obstacle_radii) & (scenarios.obstacle_radii > 0),
        scenarios.obstacle_radii,
        1.0,
    )
    effective_radii = safe_radii + barrier_config.obstacle_clearance
    position = states[..., :3]
    quaternion = states[..., 3:7]
    velocity = states[..., 7:10]
    angular_velocity = states[..., 10:13]
    relative = position[..., None, :] - safe_centers[None, ...]
    obstacle_values = (
        jnp.sum(relative**2, axis=-1) - effective_radii[None, ...] ** 2
    ) / effective_radii[None, ...] ** 2
    obstacle_values = jnp.where(mask[None, ...], obstacle_values, jnp.inf)

    span = scenarios.arena_upper - scenarios.arena_lower
    lower = (
        position - (scenarios.arena_lower + barrier_config.arena_clearance)[None, :, None, None, :]
    ) / span[None, :, None, None, :]
    upper = (
        (scenarios.arena_upper - barrier_config.arena_clearance)[None, :, None, None, :] - position
    ) / span[None, :, None, None, :]
    speed = 1.0 - jnp.sum(velocity**2, axis=-1) / scenarios.speed_limit[None, :, None, None] ** 2
    angular = (
        1.0
        - jnp.sum(angular_velocity**2, axis=-1)
        / scenarios.angular_rate_max[None, :, None, None] ** 2
    )
    rotation = quaternion_to_rotation_matrix(quaternion)
    cosine_limit = jnp.cos(scenarios.tilt_max_radians)[None, :, None, None]
    tilt = (rotation[..., 2, 2] - cosine_limit) / (1.0 - cosine_limit)
    node_values = jnp.concatenate(
        (obstacle_values, lower, upper, speed[..., None], angular[..., None], tilt[..., None]),
        axis=-1,
    )
    node_enabled = jnp.concatenate((mask, jnp.ones((*mask.shape[:-1], 9), dtype=bool)), axis=-1)
    node_enabled = jnp.broadcast_to(node_enabled[None, ...], node_values.shape)

    vehicle_start = position[..., :-1, :]
    vehicle_delta = position[..., 1:, :] - vehicle_start
    obstacle_start = safe_centers[:, :, :-1]
    obstacle_delta = safe_centers[:, :, 1:] - obstacle_start
    relative_start = vehicle_start[..., None, :] - obstacle_start[None, ...]
    relative_delta = vehicle_delta[..., None, :] - obstacle_delta[None, ...]
    denominator = jnp.sum(relative_delta**2, axis=-1)
    safe_denominator = jnp.where(denominator > 0, denominator, 1.0)
    fraction = jnp.clip(
        -jnp.sum(relative_start * relative_delta, axis=-1) / safe_denominator, 0.0, 1.0
    )
    closest = relative_start + fraction[..., None] * relative_delta
    segment_radius = jnp.maximum(effective_radii[:, :, :-1], effective_radii[:, :, 1:])
    segment_values = (
        jnp.sum(closest**2, axis=-1) - segment_radius[None, ...] ** 2
    ) / segment_radius[None, ...] ** 2
    segment_mask = mask[:, :, :-1] & mask[:, :, 1:]
    segment_enabled = jnp.broadcast_to(segment_mask[None, ...], segment_values.shape)

    finite_state = jnp.all(jnp.isfinite(states), axis=-1)
    quaternion_norm = jnp.linalg.norm(quaternion, axis=-1)
    quaternion_valid = (quaternion_norm > 32 * jnp.finfo(dtype).eps) & (
        jnp.abs(quaternion_norm - 1.0) <= barrier_config.quaternion_norm_tolerance
    )
    scenario_valid = (
        jnp.all(
            jnp.where(mask[..., None], jnp.isfinite(scenarios.obstacle_centers), True),
            axis=(-2, -1),
        )
        & jnp.all(
            jnp.where(
                mask, jnp.isfinite(scenarios.obstacle_radii) & (scenarios.obstacle_radii > 0), True
            ),
            axis=-1,
        )
        & jnp.all(
            jnp.isfinite(scenarios.arena_lower)
            & jnp.isfinite(scenarios.arena_upper)
            & (scenarios.arena_upper > scenarios.arena_lower),
            axis=-1,
        )[:, None, None]
        & (jnp.isfinite(scenarios.speed_limit) & (scenarios.speed_limit > 0))[:, None, None]
        & (jnp.isfinite(scenarios.angular_rate_max) & (scenarios.angular_rate_max > 0))[
            :, None, None
        ]
        & (
            jnp.isfinite(scenarios.tilt_max_radians)
            & (scenarios.tilt_max_radians > 0)
            & (scenarios.tilt_max_radians <= 0.5 * jnp.pi)
        )[:, None, None]
    )
    node_valid = finite_state & quaternion_valid & scenario_valid[None, ...]
    enabled_node_finite = jnp.all(
        jnp.where(node_enabled, jnp.isfinite(node_values), True), axis=(-2, -1)
    )
    enabled_segment_finite = jnp.all(
        jnp.where(segment_enabled, jnp.isfinite(segment_values), True), axis=(-2, -1)
    )
    input_valid = jnp.all(node_valid, axis=-1) & enabled_node_finite & enabled_segment_finite

    masked_node = jnp.where(node_enabled, node_values, jnp.inf).reshape(
        states.shape[0], states.shape[1], states.shape[2], -1
    )
    masked_segment = jnp.where(segment_enabled, segment_values, jnp.inf).reshape(
        states.shape[0], states.shape[1], states.shape[2], -1
    )
    flattened = jnp.concatenate((masked_node, masked_segment), axis=-1)
    prediction_hard = jnp.min(flattened, axis=-1)
    prediction_hard = jnp.where(input_valid, prediction_hard, -jnp.inf)
    robust_hard = jnp.min(prediction_hard, axis=2)
    robust_smooth = -logsumexp(-softmin_beta * flattened, axis=(-2, -1)) / softmin_beta
    robust_smooth = jnp.where(jnp.all(input_valid, axis=2), robust_smooth, -jnp.inf)
    return DynamicSafetyValues(
        node_values,
        node_enabled,
        segment_values,
        segment_enabled,
        input_valid,
        prediction_hard,
        robust_hard,
        robust_smooth,
    )


def evaluate_dynamic_quad_library(
    params: SharedActorParams,
    spec: SharedActorSpec,
    state: Array,
    scenarios: DynamicSphereScenarioBatch,
    model: VersionAModel,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    barrier_config: VersionABarrierConfig,
    *,
    dt: float,
    policy_gain: float,
    softmin_beta: float = 40.0,
    current_action_tolerance: float = 2e-5,
) -> DynamicLibraryEvaluation:
    """Evaluate one runtime library without hiding prediction disagreement at the current node."""
    if state.shape != (13,) or scenarios.obstacle_centers.shape[0] != 1:
        raise ValueError("state must be (13,) and dynamic scenario batch must have B=1")
    if not math.isfinite(current_action_tolerance) or current_action_tolerance < 0:
        raise ValueError("current_action_tolerance must be finite and nonnegative")
    rollouts = rollout_shared_quad_dynamic_library(
        params,
        spec,
        state[None, :],
        scenarios,
        model,
        actuator,
        dt=dt,
        policy_gain=policy_gain,
        actor_config=actor_config,
        quad_config=quad_config,
    )
    safety_values = dynamic_quad_safety_values(
        rollouts.states, scenarios, barrier_config, softmin_beta=softmin_beta
    )
    first_motor_all = rollouts.bounded_motor_forces[:, 0, :, 0]
    first_wrench_all = rollouts.wrenches[:, 0, :, 0]
    motor_spread = jnp.max(jnp.abs(first_motor_all - first_motor_all[:, :1]), axis=(-2, -1))
    wrench_spread = jnp.max(jnp.abs(first_wrench_all - first_wrench_all[:, :1]), axis=(-2, -1))
    first_consistent = (
        jnp.all(jnp.isfinite(first_motor_all), axis=(-2, -1))
        & jnp.all(jnp.isfinite(first_wrench_all), axis=(-2, -1))
        & (motor_spread <= current_action_tolerance)
        & (wrench_spread <= current_action_tolerance)
    )
    policy_valid = (
        jnp.all(safety_values.input_valid[:, 0], axis=-1)
        & jnp.all(rollouts.policy_valid[:, 0], axis=(-2, -1))
        & first_consistent
    )
    hard = jnp.where(policy_valid, safety_values.robust_hard_margins[:, 0], -jnp.inf)
    return DynamicLibraryEvaluation(
        rollouts,
        safety_values,
        hard,
        first_motor_all[:, 0],
        first_wrench_all[:, 0],
        first_consistent,
        policy_valid,
    )


__all__ = [
    "DYNAMIC_PREDICTION_CONTRACT",
    "DynamicQuadRolloutBatch",
    "DynamicLibraryEvaluation",
    "DynamicSafetyValues",
    "DynamicSphereScenarioBatch",
    "dynamic_quad_safety_values",
    "dynamic_sphere_window_from_tape",
    "evaluate_dynamic_quad_library",
    "rollout_shared_quad_dynamic_library",
    "validate_dynamic_sphere_batch",
]
