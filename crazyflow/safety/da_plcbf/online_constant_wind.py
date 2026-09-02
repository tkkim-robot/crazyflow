"""End-to-end corrected constant-wind DA-PLCBF mechanism demonstration.

This is intentionally a small numerical integration path.  It owns one telemetry-derived point
wind estimate, one fixed fallback library, and one persistently optimized fallback library.  The
two controllers are identical until the single wind change at four seconds.  Thereafter every
finite BPTT micro-step is published at the next control boundary; there is no candidate protocol,
admission gate, validation set, uncertainty particle, or rollback state in this module.

The simulator and learner use the airborne Version-A direct-wrench model.  Obstacle geometry is
passed only to the continuous PL-CBF controller.  The fallback actor receives state, skill-start
state, latent identity, phase, and the current point dynamics model only.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.continuous_demo_scenarios import (
    ContinuousDemoScenario,
    constant_wind_scenario,
    model_with_wind,
    scenario_obstacle_window,
    scenario_safety_limits,
    scenario_true_wind,
)
from crazyflow.safety.da_plcbf.continuous_version_a import (
    ContinuousVersionAConfig,
    PolicyRollouts,
    RuntimeObstacleTrajectories,
    continuous_version_a_step,
    rollout_waypoint_library,
    runtime_policy_values,
)
from crazyflow.safety.da_plcbf.direct_wrench import quaternion_to_rotation_matrix
from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
    ComparisonVideoTrace,
    MethodVideoTrace,
    ObstacleTrack,
)
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    SkillActorParams,
    SkillLibrarySpec,
    build_fibonacci_skill_spec,
    build_persistent_skill_learner,
    initialize_skill_actor,
    rollout_skill_library,
)
from crazyflow.safety.da_plcbf.point_wind_estimator import (
    PointWindEstimatorConfig,
    initialize_point_wind_estimator,
    model_with_point_wind,
    update_point_wind_estimator,
)
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.selector import SelectionConfig
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig, VersionAModel
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator, VersionAFilterConfig


@dataclass(frozen=True, slots=True)
class OnlineConstantWindConfig:
    """Static settings for the one-wind-step comparison."""

    policy_count: int = 8
    seed: int = 7
    nominal_acceleration_limit: float = 0.75
    waypoint_position_gain: float = 2.0
    waypoint_velocity_gain: float = 2.8
    fallback_acceleration_limit: float = 2.5
    learning_rate: float = 5.0e-4
    wind_detection_threshold: float = 0.08
    estimator_response_rate: float = 2.4
    wind_after: tuple[float, float, float] = (0.9, 0.55, 0.0)
    steps: int = 600

    def validate(self) -> None:
        """Reject settings that would change the claimed mechanism or trace shapes."""
        if self.policy_count != 8:
            raise ValueError("the corrected demonstration uses exactly K=8 fallback skills")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        positive = (
            self.nominal_acceleration_limit,
            self.waypoint_position_gain,
            self.waypoint_velocity_gain,
            self.fallback_acceleration_limit,
            self.learning_rate,
            self.wind_detection_threshold,
            self.estimator_response_rate,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("controller, learner, and estimator scales must be positive finite")
        if len(self.wind_after) != 3 or not all(math.isfinite(value) for value in self.wind_after):
            raise ValueError("wind_after must contain three finite components")
        if np.linalg.norm(np.asarray(self.wind_after)) <= 0.0:
            raise ValueError("wind_after must be nonzero")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps <= 200:
            raise ValueError("steps must leave recovery time after the fixed 4 s wind change")


class VersionAResources(NamedTuple):
    """Known fixed physical model and actuator parameters for cf21B."""

    model: VersionAModel
    actuator: VersionAActuator


@dataclass(frozen=True, slots=True)
class OnlineConstantWindResult:
    """Renderer-ready trace plus objective checks and simple device timings."""

    trace: ComparisonVideoTrace
    summary: dict[str, Any]


def build_cf21b_version_a_resources(*, dtype: jnp.dtype = jnp.float32) -> VersionAResources:
    """Load the repository's cf21B parameters into the direct-wrench Version-A structures."""
    raw: dict[str, Any] = load_params("cf21B_500")
    model = VersionAModel(
        mass=jnp.asarray(raw["mass"], dtype=dtype),
        gravity_vec=jnp.asarray(raw["gravity_vec"], dtype=dtype),
        inertia=jnp.asarray(raw["J"], dtype=dtype),
        inertia_inv=jnp.linalg.inv(jnp.asarray(raw["J"], dtype=dtype)),
        drag_matrix=jnp.asarray(raw["drag_matrix"], dtype=dtype),
        wind_velocity=jnp.zeros(3, dtype=dtype),
        external_force=jnp.zeros(3, dtype=dtype),
        external_torque=jnp.zeros(3, dtype=dtype),
    )
    actuator = VersionAActuator(
        arm_length=jnp.asarray(raw["L"], dtype=dtype),
        thrust_to_torque=jnp.asarray(raw["thrust2torque"], dtype=dtype),
        mixing_matrix=jnp.asarray(raw["mixing_matrix"], dtype=dtype),
        thrust_min=jnp.asarray(raw["thrust_min"], dtype=dtype),
        thrust_max=jnp.asarray(raw["thrust_max"], dtype=dtype),
    )
    return VersionAResources(model, actuator)


def _descriptor_metrics(
    descriptors: np.ndarray, targets: np.ndarray, scales: np.ndarray, epsilon: float
) -> tuple[float, float]:
    normalized = descriptors / scales
    normalized_targets = targets / scales
    target_loss = float(np.mean(np.square(normalized - normalized_targets)))
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / normalized.shape[0]
    sign, logdet = np.linalg.slogdet(covariance + epsilon * np.eye(covariance.shape[0]))
    diversity_loss = float(-logdet if sign > 0 else np.inf)
    return target_loss, diversity_loss


def _pairwise_descriptor_spread(descriptors: np.ndarray, scales: np.ndarray) -> float:
    normalized = descriptors / scales
    differences = normalized[:, None, :] - normalized[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    mask = ~np.eye(normalized.shape[0], dtype=bool)
    return float(np.mean(distances[mask]))


def _trajectory_descriptors(states: np.ndarray) -> np.ndarray:
    displacement = states[:, -1, :3] - states[:, 0, :3]
    mean_velocity = states[:, :, 7:10].mean(axis=1)
    terminal_velocity = states[:, -1, 7:10]
    return np.concatenate((displacement, mean_velocity, terminal_velocity), axis=-1)


def _world_intervention(state: np.ndarray, action: np.ndarray, nominal: np.ndarray) -> np.ndarray:
    rotation = np.asarray(quaternion_to_rotation_matrix(jnp.asarray(state[3:7])))
    return float(action[0] - nominal[0]) * rotation[:, 2]


def _empty_method_records() -> dict[str, list[np.ndarray | float | int]]:
    return {
        "position": [],
        "quaternion_xyzw": [],
        "nominal_rollout": [],
        "fallback_rollouts": [],
        "fallback_safe": [],
        "selected_policy": [],
        "selected_rollout": [],
        "intervention_world": [],
        "intervention_norm": [],
        "descriptors": [],
        "library_version": [],
        "cumulative_gradient_steps": [],
        "diversity_loss": [],
        "descriptor_target_loss": [],
        "gradient_norm": [],
        "parameter_update_norm": [],
        "minimum_library_value": [],
    }


def _append_method_record(
    records: dict[str, list[np.ndarray | float | int]],
    state: Array,
    decision: Any,
    *,
    library_version: int,
    cumulative_gradient_steps: int,
    diversity_loss: float,
    descriptor_target_loss: float,
    gradient_norm: float,
    parameter_update_norm: float,
) -> None:
    state_np = np.asarray(state)
    candidates = np.asarray(decision.candidates.states)
    values = np.asarray(decision.values.values)
    selected = int(np.asarray(decision.selected_index))
    records["position"].append(state_np[:3])
    records["quaternion_xyzw"].append(state_np[3:7])
    records["nominal_rollout"].append(candidates[0, :, :3])
    records["fallback_rollouts"].append(candidates[1:, :, :3])
    records["fallback_safe"].append(values[1:] >= 0.0)
    records["selected_policy"].append(-1 if selected == 0 else selected - 1)
    records["selected_rollout"].append(candidates[selected, :, :3])
    action = np.asarray(decision.action)
    nominal = np.asarray(decision.nominal_action)
    records["intervention_world"].append(_world_intervention(state_np, action, nominal))
    records["intervention_norm"].append(float(np.asarray(decision.qp_intervention_norm)))
    descriptors = _trajectory_descriptors(candidates[1:])
    records["descriptors"].append(descriptors)
    records["library_version"].append(library_version)
    records["cumulative_gradient_steps"].append(cumulative_gradient_steps)
    records["diversity_loss"].append(diversity_loss)
    records["descriptor_target_loss"].append(descriptor_target_loss)
    records["gradient_norm"].append(gradient_norm)
    records["parameter_update_norm"].append(parameter_update_norm)
    records["minimum_library_value"].append(float(np.min(values[1:])))


def _method_trace(records: dict[str, list[np.ndarray | float | int]]) -> MethodVideoTrace:
    integer_names = {"selected_policy", "library_version", "cumulative_gradient_steps"}
    boolean_names = {"fallback_safe"}
    arrays: dict[str, np.ndarray] = {}
    for name, values in records.items():
        dtype = np.int32 if name in integer_names else bool if name in boolean_names else np.float32
        arrays[name] = np.asarray(values, dtype=dtype)
    return MethodVideoTrace(**arrays)


def _make_controller(
    scenario: ContinuousDemoScenario,
    resources: VersionAResources,
    spec: SkillLibrarySpec,
    learner_config: PersistentSkillConfig,
    *,
    nominal_acceleration_limit: float,
    waypoint_position_gain: float,
    waypoint_velocity_gain: float,
    device: jax.Device,
) -> Any:
    actuator = resources.actuator
    nominal_config = QuadPolicyConfig(acceleration_limit=nominal_acceleration_limit)
    safety_limits = scenario_safety_limits(scenario)
    barrier_config = VersionABarrierConfig(
        obstacle_clearance=scenario.obstacle_clearance, arena_clearance=0.08
    )
    filter_config = VersionAFilterConfig()
    continuous_config = ContinuousVersionAConfig(
        dt=scenario.dt,
        horizon=scenario.horizon,
        obstacle_clearance=scenario.obstacle_clearance,
        prefer_nominal_when_safe=False,
    )
    # Candidate zero is the explicit task nominal.  With zero hysteresis the runtime selector uses
    # the largest admissible-set score at each boundary, so a fallback may define the certificate
    # near an obstacle and the nominal may be selected again afterward.  This is controller policy
    # selection, not policy-learning admission.
    selection_config = SelectionConfig(switch_score_margin=0.0, prefer_first_eligible=False)

    def controller(
        state: Array,
        params: SkillActorParams,
        point_model: VersionAModel,
        obstacles: RuntimeObstacleTrajectories,
        previous_policy_index: Array,
    ) -> Any:
        def nominal(candidate_state: Array, model: VersionAModel) -> PolicyRollouts:
            return rollout_waypoint_library(
                candidate_state,
                scenario.goal_position[None, :],
                scenario.goal_velocity[None, :],
                model,
                actuator,
                nominal_config,
                dt=scenario.dt,
                horizon=scenario.horizon,
                position_gain=waypoint_position_gain,
                velocity_gain=waypoint_velocity_gain,
            )

        def fallbacks(candidate_state: Array, model: VersionAModel) -> PolicyRollouts:
            rollout = rollout_skill_library(
                params, spec, candidate_state, model, actuator, learner_config
            )
            valid = jnp.all(rollout.policy_valid, axis=1) & jnp.all(
                jnp.isfinite(rollout.states), axis=(1, 2)
            )
            return PolicyRollouts(rollout.states, rollout.wrenches, valid)

        return continuous_version_a_step(
            state,
            nominal,
            fallbacks,
            obstacles,
            point_model,
            actuator,
            safety_limits,
            barrier_config,
            filter_config,
            continuous_config,
            previous_policy_index=previous_policy_index,
            selection_config=selection_config,
        )

    return jax.jit(controller, device=device)


def run_online_constant_wind_demo(
    config: OnlineConstantWindConfig = OnlineConstantWindConfig(),
    *,
    device: jax.Device | None = None,
) -> OnlineConstantWindResult:
    """Run the fixed-versus-adaptive comparison and return an actual-video-ready trace."""
    config.validate()
    if device is None:
        device = jax.devices()[0]
    scenario = replace(
        constant_wind_scenario(),
        goal_position=jnp.asarray([10.0, 0.0, 1.4], dtype=jnp.float32),
        obstacle_initial_centers=jnp.asarray(
            [[5.8, 0.0, 1.2], [7.3, 0.15, 2.65]], dtype=jnp.float32
        ),
        obstacle_velocities=jnp.zeros((2, 3), dtype=jnp.float32),
        obstacle_radii=jnp.asarray([0.55, 0.55], dtype=jnp.float32),
        obstacle_mask=jnp.asarray([True, True]),
        arena_upper=jnp.asarray([12.0, 4.0, 4.0], dtype=jnp.float32),
        wind_after=jnp.asarray(config.wind_after, dtype=jnp.float32),
        steps=config.steps,
    )
    scenario.validate()
    resources = build_cf21b_version_a_resources()
    learner_config = PersistentSkillConfig(
        dt=scenario.dt,
        horizon=scenario.horizon,
        acceleration_limit=config.fallback_acceleration_limit,
        learning_rate=config.learning_rate,
        target_weight=10.0,
        diversity_weight=0.001,
        pairwise_weight=0.005,
        trust_weight=1.0e-3,
    )
    spec = build_fibonacci_skill_spec(policy_count=config.policy_count)
    with jax.default_device(device):
        initial_params = initialize_skill_actor(jax.random.key(config.seed), spec, learner_config)
    learner = build_persistent_skill_learner(
        spec, resources.actuator, learner_config, device=device
    )
    fixed_learner_state = learner.initialize(initial_params, resources.model)
    adaptive_learner_state = learner.initialize(initial_params, resources.model)
    controller = _make_controller(
        scenario,
        resources,
        spec,
        learner_config,
        nominal_acceleration_limit=config.nominal_acceleration_limit,
        waypoint_position_gain=config.waypoint_position_gain,
        waypoint_velocity_gain=config.waypoint_velocity_gain,
        device=device,
    )
    estimator_config = PointWindEstimatorConfig(response_rate=config.estimator_response_rate)
    estimator = initialize_point_wind_estimator()

    fixed_state = jax.device_put(scenario.initial_state, device)
    adaptive_state = jax.device_put(scenario.initial_state, device)
    previous_fixed = jnp.asarray(-1, dtype=jnp.int32)
    previous_adaptive = jnp.asarray(-1, dtype=jnp.int32)
    fixed_records = _empty_method_records()
    adaptive_records = _empty_method_records()
    true_wind_records: list[np.ndarray] = []
    estimated_wind_records: list[np.ndarray] = []
    estimator_instantaneous_records: list[np.ndarray] = []
    finite_update_flags: list[bool] = []
    controller_times: list[float] = []
    learner_times: list[float] = []
    fixed_degraded: list[bool] = []
    adaptive_degraded: list[bool] = []
    fixed_used_fallback: list[bool] = []
    adaptive_used_fallback: list[bool] = []
    prewind_full_state_difference = 0.0
    shared_t4_probe_state: Array | None = None
    shared_t4_baseline_target: float | None = None
    shared_t4_baseline_spread: float | None = None
    shared_probe_fixed_targets: list[float] = []
    shared_probe_adaptive_targets: list[float] = []
    shared_probe_fixed_diversities: list[float] = []
    shared_probe_adaptive_diversities: list[float] = []
    shared_probe_fixed_spreads: list[float] = []
    shared_probe_adaptive_spreads: list[float] = []
    common_actual_fixed_safe_counts: list[int] = []
    common_actual_adaptive_safe_counts: list[int] = []
    preencounter_full_state_differences: list[float] = []
    first_detected_step: int | None = None
    last_adaptive_gradient_norm = 0.0
    last_adaptive_update_norm = 0.0
    scales = np.asarray(learner_config.descriptor_scales)
    targets = np.asarray(spec.target_descriptors)

    start_wall = time.perf_counter()
    for step_index in range(scenario.steps):
        estimated_model = model_with_point_wind(resources.model, estimator)
        detected = (
            step_index >= scenario.wind_change_step
            and float(np.linalg.norm(np.asarray(estimator.wind_velocity)))
            >= config.wind_detection_threshold
        )
        if detected:
            if first_detected_step is None:
                first_detected_step = step_index
            learning_start = time.perf_counter()
            adaptive_learner_state, update_metrics = learner.step(
                adaptive_learner_state, adaptive_state, estimated_model
            )
            jax.block_until_ready(adaptive_learner_state.params)
            learner_times.append(time.perf_counter() - learning_start)
            finite = bool(np.asarray(update_metrics.finite_update_applied))
            finite_update_flags.append(finite)
            last_adaptive_gradient_norm = float(np.asarray(update_metrics.gradient_norm))
            last_adaptive_update_norm = float(np.asarray(update_metrics.parameter_update_norm))

        obstacles = scenario_obstacle_window(scenario, step_index)
        controller_start = time.perf_counter()
        fixed_decision = controller(
            fixed_state, fixed_learner_state.params, estimated_model, obstacles, previous_fixed
        )
        adaptive_decision = controller(
            adaptive_state,
            adaptive_learner_state.params,
            estimated_model,
            obstacles,
            previous_adaptive,
        )
        jax.block_until_ready((fixed_decision.action, adaptive_decision.action))
        controller_times.append(0.5 * (time.perf_counter() - controller_start))
        fixed_degraded.append(bool(np.asarray(fixed_decision.degraded)))
        adaptive_degraded.append(bool(np.asarray(adaptive_decision.degraded)))
        fixed_used_fallback.append(bool(np.asarray(fixed_decision.used_fallback)))
        adaptive_used_fallback.append(bool(np.asarray(adaptive_decision.used_fallback)))
        if step_index < scenario.wind_change_step:
            prewind_full_state_difference = max(
                prewind_full_state_difference,
                float(np.max(np.abs(np.asarray(fixed_state) - np.asarray(adaptive_state)))),
            )

        if step_index == scenario.wind_change_step:
            shared_t4_probe_state = fixed_state
            baseline_rollout = learner.rollout(
                fixed_learner_state.params, shared_t4_probe_state, estimated_model
            )
            jax.block_until_ready(baseline_rollout.descriptors)
            baseline_descriptors = np.asarray(baseline_rollout.descriptors)
            shared_t4_baseline_target, _ = _descriptor_metrics(
                baseline_descriptors, targets, scales, learner_config.covariance_epsilon
            )
            shared_t4_baseline_spread = _pairwise_descriptor_spread(baseline_descriptors, scales)

        step_time = step_index * scenario.dt
        if 7.65 <= step_time <= 7.95:
            if shared_t4_probe_state is None:
                raise RuntimeError("the shared t=4 s probe was not initialized")
            fixed_probe = learner.rollout(
                fixed_learner_state.params, shared_t4_probe_state, estimated_model
            )
            adaptive_probe = learner.rollout(
                adaptive_learner_state.params, shared_t4_probe_state, estimated_model
            )
            # For safe coverage, both libraries are evaluated from the exact same current state.
            # The frozen method's state is the deterministic probe; neither policy sees geometry.
            fixed_common_actual = learner.rollout(
                fixed_learner_state.params, fixed_state, estimated_model
            )
            adaptive_common_actual = learner.rollout(
                adaptive_learner_state.params, fixed_state, estimated_model
            )
            jax.block_until_ready(
                (
                    fixed_probe.descriptors,
                    adaptive_probe.descriptors,
                    fixed_common_actual.states,
                    adaptive_common_actual.states,
                )
            )
            fixed_probe_descriptors = np.asarray(fixed_probe.descriptors)
            adaptive_probe_descriptors = np.asarray(adaptive_probe.descriptors)
            fixed_probe_target, fixed_probe_diversity = _descriptor_metrics(
                fixed_probe_descriptors, targets, scales, learner_config.covariance_epsilon
            )
            adaptive_probe_target, adaptive_probe_diversity = _descriptor_metrics(
                adaptive_probe_descriptors, targets, scales, learner_config.covariance_epsilon
            )
            shared_probe_fixed_targets.append(fixed_probe_target)
            shared_probe_adaptive_targets.append(adaptive_probe_target)
            shared_probe_fixed_diversities.append(fixed_probe_diversity)
            shared_probe_adaptive_diversities.append(adaptive_probe_diversity)
            shared_probe_fixed_spreads.append(
                _pairwise_descriptor_spread(fixed_probe_descriptors, scales)
            )
            shared_probe_adaptive_spreads.append(
                _pairwise_descriptor_spread(adaptive_probe_descriptors, scales)
            )
            fixed_common_values = runtime_policy_values(
                fixed_common_actual.states,
                obstacles,
                obstacle_clearance=scenario.obstacle_clearance,
            )
            adaptive_common_values = runtime_policy_values(
                adaptive_common_actual.states,
                obstacles,
                obstacle_clearance=scenario.obstacle_clearance,
            )
            common_actual_fixed_safe_counts.append(
                int(np.count_nonzero(np.asarray(fixed_common_values.values) >= 0.0))
            )
            common_actual_adaptive_safe_counts.append(
                int(np.count_nonzero(np.asarray(adaptive_common_values.values) >= 0.0))
            )
            preencounter_full_state_differences.append(
                float(np.max(np.abs(np.asarray(fixed_state) - np.asarray(adaptive_state))))
            )

        fixed_descriptors = _trajectory_descriptors(
            np.asarray(fixed_decision.candidates.states)[1:]
        )
        adaptive_descriptors = _trajectory_descriptors(
            np.asarray(adaptive_decision.candidates.states)[1:]
        )
        fixed_target, fixed_diversity = _descriptor_metrics(
            fixed_descriptors, targets, scales, learner_config.covariance_epsilon
        )
        adaptive_target, adaptive_diversity = _descriptor_metrics(
            adaptive_descriptors, targets, scales, learner_config.covariance_epsilon
        )
        _append_method_record(
            fixed_records,
            fixed_state,
            fixed_decision,
            library_version=0,
            cumulative_gradient_steps=0,
            diversity_loss=fixed_diversity,
            descriptor_target_loss=fixed_target,
            gradient_norm=0.0,
            parameter_update_norm=0.0,
        )
        _append_method_record(
            adaptive_records,
            adaptive_state,
            adaptive_decision,
            library_version=int(np.asarray(adaptive_learner_state.library_version)),
            cumulative_gradient_steps=int(
                np.asarray(adaptive_learner_state.cumulative_gradient_steps)
            ),
            diversity_loss=adaptive_diversity,
            descriptor_target_loss=adaptive_target,
            gradient_norm=last_adaptive_gradient_norm,
            parameter_update_norm=last_adaptive_update_norm,
        )
        true_wind = scenario_true_wind(scenario, step_index)
        true_wind_records.append(np.asarray(true_wind))
        estimated_wind_records.append(np.asarray(estimator.wind_velocity))
        true_model = model_with_wind(resources.model, true_wind)
        next_fixed = direct_wrench_symplectic_step(
            fixed_state, fixed_decision.action, true_model, scenario.dt
        )
        next_adaptive = direct_wrench_symplectic_step(
            adaptive_state, adaptive_decision.action, true_model, scenario.dt
        )
        estimator_update = update_point_wind_estimator(
            estimator,
            adaptive_state,
            next_adaptive,
            adaptive_decision.action,
            resources.model,
            dt=scenario.dt,
            config=estimator_config,
        )
        estimator = estimator_update.state
        estimator_instantaneous_records.append(np.asarray(estimator_update.instantaneous_wind))
        fixed_state, adaptive_state = next_fixed, next_adaptive
        previous_fixed = fixed_decision.selected_index
        previous_adaptive = adaptive_decision.selected_index

    jax.block_until_ready((fixed_state, adaptive_state, adaptive_learner_state.params))
    total_wall = time.perf_counter() - start_wall
    fixed_trace = _method_trace(fixed_records)
    adaptive_trace = _method_trace(adaptive_records)
    times = np.arange(scenario.steps, dtype=np.float64) * scenario.dt
    obstacle_centers = (
        np.asarray(scenario.obstacle_initial_centers)[None, ...]
        + times[:, None, None] * np.asarray(scenario.obstacle_velocities)[None, ...]
    )
    obstacles = tuple(
        ObstacleTrack(
            centers=obstacle_centers[:, obstacle_index],
            physical_radius=float(scenario.obstacle_radii[obstacle_index]),
            inflated_radius=float(
                scenario.obstacle_radii[obstacle_index] + scenario.obstacle_clearance
            ),
            label=f"blocking obstacle {obstacle_index + 1}",
        )
        for obstacle_index in range(scenario.obstacle_radii.size)
    )
    trace = ComparisonVideoTrace(
        time_seconds=times,
        goal_position=np.asarray(scenario.goal_position),
        obstacles=obstacles,
        true_wind=np.asarray(true_wind_records),
        estimated_wind=np.asarray(estimated_wind_records),
        wind_change_time=scenario.wind_change_step * scenario.dt,
        descriptor_targets=targets,
        fixed=fixed_trace,
        adaptive=adaptive_trace,
    )
    trace.validate()

    change_count = int(
        np.count_nonzero(np.linalg.norm(np.diff(trace.true_wind, axis=0), axis=1) > 1e-8)
    )
    prewind = times < trace.wind_change_time
    prewind_state_difference = np.max(
        np.linalg.norm(fixed_trace.position[prewind] - adaptive_trace.position[prewind], axis=1)
    )
    fixed_clearance = min(
        float(
            np.min(
                np.linalg.norm(fixed_trace.position - obstacle.centers, axis=1)
                - obstacle.physical_radius
            )
        )
        for obstacle in obstacles
    )
    adaptive_clearance = min(
        float(
            np.min(
                np.linalg.norm(adaptive_trace.position - obstacle.centers, axis=1)
                - obstacle.physical_radius
            )
        )
        for obstacle in obstacles
    )
    final_estimator_error = float(np.linalg.norm(trace.estimated_wind[-1] - trace.true_wind[-1]))
    parameter_delta = float(
        np.sqrt(
            sum(
                np.sum(np.square(np.asarray(updated) - np.asarray(initial)))
                for updated, initial in zip(
                    jax.tree.leaves(adaptive_learner_state.params),
                    jax.tree.leaves(initial_params),
                    strict=True,
                )
            )
        )
    )
    adaptive_steps = int(np.asarray(adaptive_learner_state.cumulative_gradient_steps))
    common_reference_state = scenario.initial_state.at[:3].set(scenario.goal_position)
    final_point_model = model_with_point_wind(resources.model, estimator)
    fixed_common_rollout = learner.rollout(
        fixed_learner_state.params, common_reference_state, final_point_model
    )
    adaptive_common_rollout = learner.rollout(
        adaptive_learner_state.params, common_reference_state, final_point_model
    )
    jax.block_until_ready((fixed_common_rollout.descriptors, adaptive_common_rollout.descriptors))
    fixed_common_target, fixed_common_diversity = _descriptor_metrics(
        np.asarray(fixed_common_rollout.descriptors),
        targets,
        scales,
        learner_config.covariance_epsilon,
    )
    adaptive_common_target, adaptive_common_diversity = _descriptor_metrics(
        np.asarray(adaptive_common_rollout.descriptors),
        targets,
        scales,
        learner_config.covariance_epsilon,
    )
    if (
        shared_t4_baseline_target is None
        or shared_t4_baseline_spread is None
        or not shared_probe_fixed_targets
        or not common_actual_fixed_safe_counts
    ):
        raise RuntimeError(
            "the configured run did not cover the 4 s and 7.65--7.95 s probe windows"
        )
    postwindow_fixed_target = float(np.mean(shared_probe_fixed_targets))
    postwindow_adaptive_target = float(np.mean(shared_probe_adaptive_targets))
    postwindow_fixed_diversity = float(np.mean(shared_probe_fixed_diversities))
    postwindow_adaptive_diversity = float(np.mean(shared_probe_adaptive_diversities))
    postwindow_fixed_spread = float(np.mean(shared_probe_fixed_spreads))
    postwindow_adaptive_spread = float(np.mean(shared_probe_adaptive_spreads))
    maximum_common_safe_advantage = int(
        np.max(
            np.asarray(common_actual_adaptive_safe_counts)
            - np.asarray(common_actual_fixed_safe_counts)
        )
    )
    encounter_window = (times >= 7.65) & (times <= 7.95)
    actual_fixed_safe_counts = np.count_nonzero(fixed_trace.fallback_safe[encounter_window], axis=1)
    actual_adaptive_safe_counts = np.count_nonzero(
        adaptive_trace.fallback_safe[encounter_window], axis=1
    )
    maximum_actual_safe_advantage = int(
        np.max(actual_adaptive_safe_counts - actual_fixed_safe_counts)
    )
    initial_physical_clearance = min(
        float(
            np.linalg.norm(fixed_trace.position[0] - obstacle.centers[0]) - obstacle.physical_radius
        )
        for obstacle in obstacles
    )
    fixed_closest_step = int(
        np.argmin(
            np.min(
                np.stack(
                    [
                        np.linalg.norm(fixed_trace.position - obstacle.centers, axis=1)
                        - obstacle.inflated_radius
                        for obstacle in obstacles
                    ],
                    axis=1,
                ),
                axis=1,
            )
        )
    )
    adaptive_closest_step = int(
        np.argmin(
            np.min(
                np.stack(
                    [
                        np.linalg.norm(adaptive_trace.position - obstacle.centers, axis=1)
                        - obstacle.inflated_radius
                        for obstacle in obstacles
                    ],
                    axis=1,
                ),
                axis=1,
            )
        )
    )
    publication_consistent = (
        int(np.asarray(adaptive_learner_state.library_version)) == adaptive_steps
    )
    prewind_identical = prewind_full_state_difference == 0.0
    single_wind_step = change_count == 1
    estimator_converged = final_estimator_error < 0.01
    finite_persistent_learning = bool(all(finite_update_flags)) and publication_consistent
    shared_probe_fixed_bias = postwindow_fixed_target >= 1.25 * shared_t4_baseline_target
    shared_probe_target_recovery = postwindow_adaptive_target <= 0.90 * postwindow_fixed_target
    shared_probe_spread_recovery = postwindow_adaptive_spread >= 1.15 * postwindow_fixed_spread
    useful_safe_coverage_recovery = (
        maximum_common_safe_advantage >= 2 or maximum_actual_safe_advantage >= 2
    )
    no_degraded_steps = not any(fixed_degraded) and not any(adaptive_degraded)
    adaptive_avoided_inflated_obstacle = adaptive_clearance > scenario.obstacle_clearance
    nonzero_qp_intervention = float(np.max(adaptive_trace.intervention_norm)) > 1e-5
    adaptive_reached_goal = (
        float(np.linalg.norm(adaptive_trace.position[-1] - trace.goal_position)) < 0.5
    )
    checks = {
        "exactly_one_constant_wind_step": single_wind_step,
        "prewind_methods_identical": prewind_identical,
        "every_finite_step_published_persistently": finite_persistent_learning,
        "frozen_library_has_postwind_descriptor_bias": shared_probe_fixed_bias,
        "adaptive_shared_probe_target_recovery": shared_probe_target_recovery,
        "adaptive_shared_probe_spread_recovery": shared_probe_spread_recovery,
        "adaptive_useful_safe_coverage_recovery": useful_safe_coverage_recovery,
        "point_estimator_converged_below_0p01_mps": estimator_converged,
        "adaptive_inflated_clearance_positive": adaptive_avoided_inflated_obstacle,
        "qp_intervention_above_1e-5": nonzero_qp_intervention,
        "zero_degraded_controller_steps": no_degraded_steps,
        "adaptive_final_goal_distance_below_0p5_m": adaptive_reached_goal,
    }
    summary: dict[str, Any] = {
        "device": str(device),
        "steps": scenario.steps,
        "dt_seconds": scenario.dt,
        "true_wind_change_count": change_count,
        "wind_change_time_seconds": trace.wind_change_time,
        "wind_detected_time_seconds": (
            None if first_detected_step is None else first_detected_step * scenario.dt
        ),
        "prewind_max_position_difference_m": float(prewind_state_difference),
        "prewind_max_full_state_component_difference": prewind_full_state_difference,
        "point_wind_final_error_mps": final_estimator_error,
        "point_wind_finite_updates": int(np.asarray(estimator.finite_update_count)),
        "adaptive_gradient_steps": adaptive_steps,
        "adaptive_library_version": int(np.asarray(adaptive_learner_state.library_version)),
        "library_version_equals_finite_gradient_steps": int(
            np.asarray(adaptive_learner_state.library_version)
        )
        == adaptive_steps,
        "adaptive_parameter_delta_norm": parameter_delta,
        "all_attempted_bptt_updates_finite": bool(all(finite_update_flags)),
        "fixed_library_version": 0,
        "fixed_minimum_physical_clearance_m": fixed_clearance,
        "adaptive_minimum_physical_clearance_m": adaptive_clearance,
        "fixed_minimum_inflated_clearance_m": min(
            float(
                np.min(
                    np.linalg.norm(fixed_trace.position - obstacle.centers, axis=1)
                    - obstacle.inflated_radius
                )
            )
            for obstacle in obstacles
        ),
        "adaptive_minimum_inflated_clearance_m": min(
            float(
                np.min(
                    np.linalg.norm(adaptive_trace.position - obstacle.centers, axis=1)
                    - obstacle.inflated_radius
                )
            )
            for obstacle in obstacles
        ),
        "fixed_final_goal_distance_m": float(
            np.linalg.norm(fixed_trace.position[-1] - trace.goal_position)
        ),
        "adaptive_final_goal_distance_m": float(
            np.linalg.norm(adaptive_trace.position[-1] - trace.goal_position)
        ),
        "fixed_max_qp_intervention_norm": float(np.max(fixed_trace.intervention_norm)),
        "adaptive_max_qp_intervention_norm": float(np.max(adaptive_trace.intervention_norm)),
        "fixed_final_safe_fallback_count": int(np.count_nonzero(fixed_trace.fallback_safe[-1])),
        "adaptive_final_safe_fallback_count": int(
            np.count_nonzero(adaptive_trace.fallback_safe[-1])
        ),
        "fixed_final_descriptor_target_loss": float(fixed_trace.descriptor_target_loss[-1]),
        "adaptive_final_descriptor_target_loss": float(adaptive_trace.descriptor_target_loss[-1]),
        "fixed_final_diversity_loss": float(fixed_trace.diversity_loss[-1]),
        "adaptive_final_diversity_loss": float(adaptive_trace.diversity_loss[-1]),
        "common_state_fixed_descriptor_target_loss": fixed_common_target,
        "common_state_adaptive_descriptor_target_loss": adaptive_common_target,
        "common_state_fixed_diversity_loss": fixed_common_diversity,
        "common_state_adaptive_diversity_loss": adaptive_common_diversity,
        "shared_t4_zero_wind_descriptor_target_loss": shared_t4_baseline_target,
        "shared_t4_zero_wind_pairwise_spread": shared_t4_baseline_spread,
        "shared_t4_probe_postwindow_fixed_target_loss": postwindow_fixed_target,
        "shared_t4_probe_postwindow_adaptive_target_loss": postwindow_adaptive_target,
        "shared_t4_probe_postwindow_fixed_diversity_loss": postwindow_fixed_diversity,
        "shared_t4_probe_postwindow_adaptive_diversity_loss": postwindow_adaptive_diversity,
        "shared_t4_probe_postwindow_fixed_pairwise_spread": postwindow_fixed_spread,
        "shared_t4_probe_postwindow_adaptive_pairwise_spread": postwindow_adaptive_spread,
        "common_actual_postwindow_fixed_safe_counts": common_actual_fixed_safe_counts,
        "common_actual_postwindow_adaptive_safe_counts": common_actual_adaptive_safe_counts,
        "common_actual_maximum_adaptive_safe_count_advantage": maximum_common_safe_advantage,
        "encounter_actual_fixed_safe_counts": actual_fixed_safe_counts.tolist(),
        "encounter_actual_adaptive_safe_counts": actual_adaptive_safe_counts.tolist(),
        "encounter_actual_maximum_adaptive_safe_count_advantage": (maximum_actual_safe_advantage),
        "common_actual_postwindow_fixed_mean_safe_fallback_count": float(
            np.mean(common_actual_fixed_safe_counts)
        ),
        "common_actual_postwindow_adaptive_mean_safe_fallback_count": float(
            np.mean(common_actual_adaptive_safe_counts)
        ),
        "common_probe_7p65_to_7p95_max_full_state_component_difference": float(
            np.max(preencounter_full_state_differences)
        ),
        "initial_physical_obstacle_clearance_m": initial_physical_clearance,
        "fixed_closest_obstacle_time_seconds": fixed_closest_step * scenario.dt,
        "adaptive_closest_obstacle_time_seconds": adaptive_closest_step * scenario.dt,
        "fixed_degraded_step_count": int(np.count_nonzero(fixed_degraded)),
        "adaptive_degraded_step_count": int(np.count_nonzero(adaptive_degraded)),
        "fixed_fallback_execution_count": int(np.count_nonzero(fixed_used_fallback)),
        "adaptive_fallback_execution_count": int(np.count_nonzero(adaptive_used_fallback)),
        "controller_median_seconds_per_method": float(np.median(controller_times)),
        "controller_p95_seconds_per_method": float(np.percentile(controller_times, 95)),
        "bptt_median_seconds": (None if not learner_times else float(np.median(learner_times))),
        "bptt_p95_seconds": (
            None if not learner_times else float(np.percentile(learner_times, 95))
        ),
        "total_wall_seconds_including_jit": total_wall,
        "instantaneous_estimator_final_mps": estimator_instantaneous_records[-1].tolist(),
        "acceptance_prewind_methods_identical": prewind_identical,
        "acceptance_exactly_one_constant_wind_step": single_wind_step,
        "acceptance_point_estimator_converged": estimator_converged,
        "acceptance_every_finite_step_published_persistently": finite_persistent_learning,
        "acceptance_frozen_library_has_postwind_descriptor_bias": shared_probe_fixed_bias,
        "acceptance_adaptive_shared_probe_target_recovery": shared_probe_target_recovery,
        "acceptance_adaptive_shared_probe_spread_recovery": shared_probe_spread_recovery,
        "acceptance_adaptive_useful_safe_coverage_recovery": (useful_safe_coverage_recovery),
        "acceptance_no_degraded_controller_steps": no_degraded_steps,
        "acceptance_adaptive_avoids_inflated_obstacle": adaptive_avoided_inflated_obstacle,
        "acceptance_nonzero_qp_intervention": nonzero_qp_intervention,
        "acceptance_adaptive_reaches_goal": adaptive_reached_goal,
        "checks": checks,
    }
    acceptance_flags = [
        value
        for name, value in summary.items()
        if name.startswith("acceptance_") and name != "acceptance_all_corrected_mechanism_gates"
    ]
    summary["acceptance_all_corrected_mechanism_gates"] = bool(all(acceptance_flags))
    summary["all_checks_passed"] = bool(all(checks.values()))
    return OnlineConstantWindResult(trace=trace, summary=summary)


def _trace_arrays(trace: ComparisonVideoTrace) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "time_seconds": np.asarray(trace.time_seconds),
        "goal_position": np.asarray(trace.goal_position),
        "true_wind": np.asarray(trace.true_wind),
        "estimated_wind": np.asarray(trace.estimated_wind),
        "descriptor_targets": np.asarray(trace.descriptor_targets),
        "obstacle_centers": np.stack([obstacle.centers for obstacle in trace.obstacles], axis=1),
        "obstacle_physical_radii": np.asarray(
            [obstacle.physical_radius for obstacle in trace.obstacles]
        ),
        "obstacle_inflated_radii": np.asarray(
            [obstacle.inflated_radius for obstacle in trace.obstacles]
        ),
    }
    for prefix, method in (("fixed", trace.fixed), ("adaptive", trace.adaptive)):
        for name in MethodVideoTrace.__dataclass_fields__:
            arrays[f"{prefix}_{name}"] = np.asarray(getattr(method, name))
    return arrays


def save_online_constant_wind_result(
    result: OnlineConstantWindResult,
    output_directory: str | Path,
    *,
    stem: str = "online_constant_wind",
) -> tuple[Path, Path]:
    """Save a compact numerical trace and human-readable objective/timing summary."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    trace_path = directory / f"{stem}.npz"
    summary_path = directory / f"{stem}.json"
    if trace_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite an existing corrected-demo result")
    np.savez_compressed(trace_path, **_trace_arrays(result.trace))
    metadata = {
        "title": result.trace.title,
        "wind_change_time": result.trace.wind_change_time,
        "obstacle_labels": [obstacle.label for obstacle in result.trace.obstacles],
        "summary": result.summary,
    }
    summary_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return trace_path, summary_path


def load_online_constant_wind_trace(
    trace_path: str | Path, summary_path: str | Path
) -> ComparisonVideoTrace:
    """Reconstruct renderer input without re-running simulation or online learning."""
    metadata = json.loads(Path(summary_path).read_text())
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    methods = {}
    for prefix in ("fixed", "adaptive"):
        methods[prefix] = MethodVideoTrace(
            **{name: arrays[f"{prefix}_{name}"] for name in MethodVideoTrace.__dataclass_fields__}
        )
    obstacles = tuple(
        ObstacleTrack(
            centers=arrays["obstacle_centers"][:, index],
            physical_radius=float(arrays["obstacle_physical_radii"][index]),
            inflated_radius=float(arrays["obstacle_inflated_radii"][index]),
            label=metadata["obstacle_labels"][index],
        )
        for index in range(arrays["obstacle_centers"].shape[1])
    )
    trace = ComparisonVideoTrace(
        time_seconds=arrays["time_seconds"],
        goal_position=arrays["goal_position"],
        obstacles=obstacles,
        true_wind=arrays["true_wind"],
        estimated_wind=arrays["estimated_wind"],
        wind_change_time=float(metadata["wind_change_time"]),
        descriptor_targets=arrays["descriptor_targets"],
        fixed=methods["fixed"],
        adaptive=methods["adaptive"],
        title=metadata["title"],
    )
    trace.validate()
    return trace


__all__ = [
    "OnlineConstantWindConfig",
    "OnlineConstantWindResult",
    "VersionAResources",
    "build_cf21b_version_a_resources",
    "load_online_constant_wind_trace",
    "run_online_constant_wind_demo",
    "save_online_constant_wind_result",
]
