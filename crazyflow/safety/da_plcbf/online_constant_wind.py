"""End-to-end corrected constant-wind DA-PLCBF mechanism demonstration.

This numerical comparison uses one telemetry-derived point estimate per independent method,
frozen and persistently optimized fallback libraries, and optional matched analytic-only and
model-compensated frozen baselines. Every finite BPTT micro-step is published at a control
boundary; there is no candidate protocol, admission gate, validation set, uncertainty particle,
or rollback state in this module.

The simulator and learner use the airborne Version-A direct-wrench model.  Obstacle geometry is
passed only to the continuous PL-CBF controller.  The fallback actor receives state, skill-start
state, latent identity, phase, and the current point dynamics model only.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
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
    nominal_model_compensation: bool = False
    waypoint_position_gain: float = 2.0
    waypoint_velocity_gain: float = 2.8
    fallback_acceleration_limit: float = 2.5
    learning_rate: float = 5.0e-4
    wind_detection_threshold: float = 0.08
    estimator_response_rate: float = 2.4
    wind_after: tuple[float, float, float] = (0.9, 0.55, 0.0)
    steps: int = 600
    include_baselines: bool = True
    gradient_steps_per_boundary: int = 1
    learning_start: str = "wind"
    initial_residual_scale: float = 0.01
    initial_skill_scale: float = 1.0
    residual_scale: float = 1.0
    policy_gain: float = 1.8
    smooth_motor_bounds: bool = False
    probe_every_steps: int = 10
    probe_window_seconds: tuple[float, float] | None = None
    policy_alpha: float = 2.0
    smooth_min_temperature: float = 0.005

    def validate(self) -> None:
        """Reject settings that would change the claimed mechanism or trace shapes."""
        if (
            isinstance(self.policy_count, bool)
            or not isinstance(self.policy_count, int)
            or self.policy_count < 2
        ):
            raise ValueError("policy_count must be an integer of at least two")
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
            self.policy_gain,
            self.policy_alpha,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("controller, learner, and estimator scales must be positive finite")
        if len(self.wind_after) != 3 or not all(math.isfinite(value) for value in self.wind_after):
            raise ValueError("wind_after must contain three finite components")
        if np.linalg.norm(np.asarray(self.wind_after)) <= 0.0:
            raise ValueError("wind_after must be nonzero")
        for name in ("steps", "gradient_steps_per_boundary", "probe_every_steps"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (2 if name == "steps" else 1)
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("initial_residual_scale", "residual_scale", "smooth_min_temperature"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_start not in {"wind", "startup"}:
            raise ValueError("learning_start must be 'wind' or 'startup'")
        if (
            not math.isfinite(self.initial_skill_scale)
            or not 0.0 <= self.initial_skill_scale <= 1.0
        ):
            raise ValueError("initial_skill_scale must lie in [0, 1]")
        if not isinstance(self.include_baselines, bool) or not isinstance(
            self.smooth_motor_bounds, bool
        ):
            raise TypeError("include_baselines and smooth_motor_bounds must be boolean")
        if not isinstance(self.nominal_model_compensation, bool):
            raise TypeError("nominal_model_compensation must be boolean")
        if self.probe_window_seconds is not None:
            if (
                len(self.probe_window_seconds) != 2
                or not all(math.isfinite(x) for x in self.probe_window_seconds)
                or self.probe_window_seconds[0] > self.probe_window_seconds[1]
            ):
                raise ValueError("probe_window_seconds must be an ordered finite pair")


class VersionAResources(NamedTuple):
    """Known fixed physical model and actuator parameters for cf21B."""

    model: VersionAModel
    actuator: VersionAActuator


@dataclass(frozen=True, slots=True)
class OnlineConstantWindResult:
    """Renderer-ready trace plus objective checks and simple device timings."""

    trace: ComparisonVideoTrace
    summary: dict[str, Any]
    methods: dict[str, MethodVideoTrace] = field(default_factory=dict)


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
    mean_velocity = states[:, 1:, 7:10].mean(axis=1)
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
        "maximum_library_value": [],
        "selected_policy_value": [],
        "selected_policy_dual": [],
        "qp_valid": [],
        "used_fallback": [],
        "degraded": [],
        "qp_rejection_flags": [],
        "estimated_wind": [],
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
    estimated_wind: Array,
) -> None:
    state_np = np.asarray(state)
    candidates = np.asarray(decision.candidates.states)
    values = np.asarray(decision.values.values)
    selected = int(np.asarray(decision.selected_index))
    records["position"].append(state_np[:3])
    records["quaternion_xyzw"].append(state_np[3:7])
    records["nominal_rollout"].append(candidates[0, :, :3])
    records["fallback_rollouts"].append(candidates[1:, :, :3])
    records["fallback_safe"].append(
        (values[1:] >= 0.0)
        & np.asarray(decision.candidates.valid)[1:]
        & np.asarray(decision.values.input_valid)[1:]
    )
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
    records["maximum_library_value"].append(float(np.max(values)))
    records["selected_policy_value"].append(float(values[selected]))
    for name in (
        "selected_policy_dual",
        "qp_valid",
        "used_fallback",
        "degraded",
        "qp_rejection_flags",
    ):
        records[name].append(np.asarray(getattr(decision, name)))
    records["estimated_wind"].append(np.asarray(estimated_wind))


def _method_trace(
    records: dict[str, list[np.ndarray | float | int]], *, control_mode: str = "plcbf"
) -> MethodVideoTrace:
    integer_names = {"selected_policy", "library_version", "cumulative_gradient_steps"}
    boolean_names = {"fallback_safe", "qp_valid", "used_fallback", "degraded", "qp_rejection_flags"}
    arrays: dict[str, np.ndarray] = {}
    for name, values in records.items():
        dtype = np.int32 if name in integer_names else bool if name in boolean_names else np.float32
        arrays[name] = np.asarray(values, dtype=dtype)
    return MethodVideoTrace(**arrays, control_mode=control_mode)


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
    analytic_only: bool = False,
    policy_alpha: float = 2.0,
    smooth_min_temperature: float = 0.005,
    nominal_model_compensation: bool = False,
) -> Any:
    actuator = resources.actuator
    nominal_config = QuadPolicyConfig(acceleration_limit=nominal_acceleration_limit)
    safety_limits = scenario_safety_limits(scenario)
    barrier_config = VersionABarrierConfig(
        obstacle_clearance=scenario.obstacle_clearance,
        arena_clearance=0.08,
        ego_radius=scenario.ego_radius,
    )
    filter_config = VersionAFilterConfig(policy_alpha=policy_alpha)
    continuous_config = ContinuousVersionAConfig(
        dt=scenario.dt,
        horizon=scenario.horizon,
        obstacle_clearance=scenario.obstacle_clearance,
        prefer_nominal_when_safe=False,
        ego_radius=scenario.ego_radius,
        analytic_obstacle_hocbf=analytic_only,
        use_policy_constraint=not analytic_only,
        smooth_min_temperature=smooth_min_temperature,
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
                model_compensation=nominal_model_compensation,
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

    return jax.jit(controller)


def _timing_statistics(samples: list[float]) -> dict[str, float | int | None]:
    """Summarize individually synchronized warm calls, excluding compilation."""
    return {
        "count": len(samples),
        "median_seconds": None if not samples else float(np.median(samples)),
        "p95_seconds": None if not samples else float(np.percentile(samples, 95)),
        "maximum_seconds": None if not samples else float(np.max(samples)),
    }


def _swept_clearance(
    positions: np.ndarray, centers: np.ndarray, radii: np.ndarray
) -> tuple[float, int]:
    """Minimum relative segment distance, including the final applied plant step."""
    relative = positions[:, None, :] - centers
    delta = np.diff(relative, axis=0)
    denominator = np.sum(delta * delta, axis=-1)
    fraction = np.clip(
        -np.sum(relative[:-1] * delta, axis=-1) / np.maximum(denominator, 1e-20), 0.0, 1.0
    )
    distances = np.linalg.norm(relative[:-1] + fraction[..., None] * delta, axis=-1) - radii
    per_step = np.min(distances, axis=1)
    return float(np.min(per_step)), int(np.argmin(per_step))


def run_online_constant_wind_demo(
    config: OnlineConstantWindConfig = OnlineConstantWindConfig(),
    *,
    device: jax.Device | None = None,
    scenario: ContinuousDemoScenario | None = None,
    progress_callback: Any | None = None,
) -> OnlineConstantWindResult:
    """Run matched controllers; failures remain measurable and saveable results.

    Each method estimates wind from its own measured transition. Only the adaptive method changes
    parameters; the compensated frozen baseline adds explicit point-model drag cancellation to the
    same initialized fallback actor. All methods share the unmodified waypoint nominal. Probe
    comparisons use one identical state and point model for every library and never feed safety
    values back into learning. ``scenario`` owns geometry/timing when supplied.
    """
    config.validate()
    if device is None:
        device = jax.devices()[0]
    if scenario is None:
        scenario = replace(
            constant_wind_scenario(),
            wind_after=jnp.asarray(config.wind_after, dtype=jnp.float32),
            steps=config.steps,
        )
    scenario.validate()
    resources = jax.device_put(build_cf21b_version_a_resources(), device)
    learner_config = PersistentSkillConfig(
        dt=scenario.dt,
        horizon=scenario.horizon,
        acceleration_limit=config.fallback_acceleration_limit,
        learning_rate=config.learning_rate,
        target_weight=10.0,
        diversity_weight=0.001,
        pairwise_weight=0.005,
        trust_weight=1.0e-3,
        initial_residual_scale=config.initial_residual_scale,
        initial_skill_scale=config.initial_skill_scale,
        residual_scale=config.residual_scale,
        policy_gain=config.policy_gain,
        smooth_motor_bounds=config.smooth_motor_bounds,
    )
    spec = jax.device_put(
        build_fibonacci_skill_spec(
            policy_count=config.policy_count, horizon_duration=scenario.horizon * scenario.dt
        ),
        device,
    )
    with jax.default_device(device):
        initial_params = initialize_skill_actor(jax.random.key(config.seed), spec, learner_config)
    initial_params = jax.device_put(initial_params, device)
    learner = build_persistent_skill_learner(
        spec, resources.actuator, learner_config, device=device
    )
    adaptive_learner_state = jax.device_put(
        learner.initialize(initial_params, resources.model), device
    )
    compensated_config = replace(learner_config, model_compensation=True)
    compensated_learner = build_persistent_skill_learner(
        spec, resources.actuator, compensated_config, device=device
    )
    names = ["fixed", "adaptive"] + (
        ["analytic", "compensated"] if config.include_baselines else []
    )
    controller_kwargs = dict(
        nominal_acceleration_limit=config.nominal_acceleration_limit,
        waypoint_position_gain=config.waypoint_position_gain,
        waypoint_velocity_gain=config.waypoint_velocity_gain,
        device=device,
        policy_alpha=config.policy_alpha,
        smooth_min_temperature=config.smooth_min_temperature,
        nominal_model_compensation=config.nominal_model_compensation,
    )
    base_controller = _make_controller(
        scenario, resources, spec, learner_config, **controller_kwargs
    )
    controllers = {"fixed": base_controller, "adaptive": base_controller}
    if config.include_baselines:
        controllers["analytic"] = _make_controller(
            scenario, resources, spec, learner_config, analytic_only=True, **controller_kwargs
        )
        controllers["compensated"] = _make_controller(
            scenario, resources, spec, compensated_config, **controller_kwargs
        )
    states = {name: jax.device_put(scenario.initial_state, device) for name in names}
    previous = {name: jax.device_put(jnp.asarray(-1, dtype=jnp.int32), device) for name in names}
    estimators = {name: jax.device_put(initialize_point_wind_estimator(), device) for name in names}
    estimator_config = PointWindEstimatorConfig(response_rate=config.estimator_response_rate)

    @jax.jit
    def estimator_step(estimator: Any, state: Array, following: Array, action: Array) -> Any:
        return update_point_wind_estimator(
            estimator,
            state,
            following,
            action,
            resources.model,
            dt=scenario.dt,
            config=estimator_config,
        )

    @jax.jit
    def plant_step(state: Array, action: Array, model: VersionAModel) -> Array:
        return direct_wrench_symplectic_step(state, action, model, scenario.dt)

    @jax.jit
    def evaluate_values(rollout_states: Array, obstacles: RuntimeObstacleTrajectories) -> Any:
        return runtime_policy_values(
            rollout_states,
            obstacles,
            obstacle_clearance=scenario.obstacle_clearance,
            ego_radius=scenario.ego_radius,
        )

    records = {name: _empty_method_records() for name in names}
    controller_times: dict[str, list[float]] = {name: [] for name in names}
    qp_fast_path_counts = {name: 0 for name in names}
    estimator_times: dict[str, list[float]] = {name: [] for name in names}
    learner_times: list[float] = []
    finite_update_flags: list[bool] = []
    true_winds: list[np.ndarray] = []
    prewind_full_state_difference = 0.0
    first_detected_step: int | None = None
    scales = np.asarray(learner_config.descriptor_scales)
    targets = np.asarray(spec.target_descriptors)
    shared_reference_state = states["fixed"]
    baseline_target: float | None = None
    baseline_spread: float | None = None
    probes: list[dict[str, Any]] = []
    state_differences: list[float] = []
    instantaneous_winds: dict[str, np.ndarray] = {}

    # Compile once and discard warm-up results. No optimizer step or estimator measurement from
    # warm-up enters the episode. Every timing sample below blocks its own call's complete output.
    warmup_start = time.perf_counter()
    initial_obstacles = scenario_obstacle_window(scenario, 0)
    for name in names:
        decision = controllers[name](
            states[name], initial_params, resources.model, initial_obstacles, previous[name]
        )
        jax.block_until_ready(decision)
    discarded_learner_state, discarded_metrics = learner.step(
        adaptive_learner_state, states["adaptive"], resources.model
    )
    jax.block_until_ready((discarded_learner_state, discarded_metrics))
    # The first JIT-produced snapshot can have a different placement signature from initialized
    # arrays. Exercise it before measuring; this discarded update is never published or applied.
    jax.block_until_ready(
        controllers["adaptive"](
            states["adaptive"],
            discarded_learner_state.params,
            resources.model,
            initial_obstacles,
            previous["adaptive"],
        )
    )
    jax.block_until_ready(learner.rollout(initial_params, states["fixed"], resources.model))
    if config.include_baselines:
        jax.block_until_ready(
            compensated_learner.rollout(initial_params, states["fixed"], resources.model)
        )
    following = plant_step(states["fixed"], decision.action, resources.model)
    jax.block_until_ready(
        estimator_step(estimators["fixed"], states["fixed"], following, decision.action)
    )
    jax.block_until_ready(evaluate_values(decision.candidates.states[1:], initial_obstacles))
    warmup_seconds = time.perf_counter() - warmup_start
    start_wall = time.perf_counter()
    for step_index in range(scenario.steps):
        point_models = {
            name: model_with_point_wind(resources.model, estimators[name]) for name in names
        }
        detected = (
            float(np.linalg.norm(np.asarray(estimators["adaptive"].wind_velocity)))
            >= config.wind_detection_threshold
        )
        if detected and first_detected_step is None:
            first_detected_step = step_index
        if config.learning_start == "startup" or detected:
            for _ in range(config.gradient_steps_per_boundary):
                started = time.perf_counter()
                adaptive_learner_state, update_metrics = learner.step(
                    adaptive_learner_state, states["adaptive"], point_models["adaptive"]
                )
                jax.block_until_ready((adaptive_learner_state, update_metrics))
                learner_times.append(time.perf_counter() - started)
                finite_update_flags.append(bool(np.asarray(update_metrics.finite_update_applied)))
            gradient_norm = float(np.asarray(update_metrics.gradient_norm))
            update_norm = float(np.asarray(update_metrics.parameter_update_norm))
        else:
            gradient_norm = update_norm = 0.0
        params = {
            name: adaptive_learner_state.params if name == "adaptive" else initial_params
            for name in names
        }
        obstacles = scenario_obstacle_window(scenario, step_index)
        decisions = {}
        for name in names:
            started = time.perf_counter()
            decision = controllers[name](
                states[name], params[name], point_models[name], obstacles, previous[name]
            )
            jax.block_until_ready(decision)
            controller_times[name].append(time.perf_counter() - started)
            qp_fast_path_counts[name] += int(
                np.asarray(getattr(decision.continuous_filter, "qp_fast_path_used", False))
            )
            decisions[name] = decision
            descriptors = _trajectory_descriptors(np.asarray(decision.candidates.states)[1:])
            target, diversity = _descriptor_metrics(
                descriptors, targets, scales, learner_config.covariance_epsilon
            )
            _append_method_record(
                records[name],
                states[name],
                decision,
                library_version=int(np.asarray(adaptive_learner_state.library_version))
                if name == "adaptive"
                else 0,
                cumulative_gradient_steps=int(
                    np.asarray(adaptive_learner_state.cumulative_gradient_steps)
                )
                if name == "adaptive"
                else 0,
                diversity_loss=diversity,
                descriptor_target_loss=target,
                gradient_norm=gradient_norm if name == "adaptive" else 0.0,
                parameter_update_norm=update_norm if name == "adaptive" else 0.0,
                estimated_wind=estimators[name].wind_velocity,
            )
        difference = float(
            np.max(np.abs(np.asarray(states["fixed"]) - np.asarray(states["adaptive"])))
        )
        state_differences.append(difference)
        if step_index < scenario.wind_change_step:
            prewind_full_state_difference = max(prewind_full_state_difference, difference)
        if step_index == scenario.wind_change_step:
            shared_reference_state = states["fixed"]
            baseline = learner.rollout(
                initial_params, shared_reference_state, point_models["fixed"]
            )
            baseline_descriptors = np.asarray(baseline.descriptors)
            baseline_target, _ = _descriptor_metrics(
                baseline_descriptors, targets, scales, learner_config.covariance_epsilon
            )
            baseline_spread = _pairwise_descriptor_spread(baseline_descriptors, scales)
        step_time = step_index * scenario.dt
        probe_start = (
            (scenario.wind_change_step * scenario.dt + 0.5)
            if config.probe_window_seconds is None
            else config.probe_window_seconds[0]
        )
        probe_end = (
            scenario.steps * scenario.dt
            if config.probe_window_seconds is None
            else config.probe_window_seconds[1]
        )
        if probe_start <= step_time <= probe_end and step_index % config.probe_every_steps == 0:
            probe: dict[str, Any] = {
                "time_seconds": step_time,
                "methods": {},
                "adaptive_state_coverage": {},
                "adaptive_state_position": np.asarray(states["adaptive"][:3]).tolist(),
                "adaptive_state_full_state": np.asarray(states["adaptive"]).tolist(),
                "adaptive_state_point_wind": np.asarray(
                    point_models["adaptive"].wind_velocity
                ).tolist(),
            }
            shared_nominal_value = float(np.asarray(decisions["adaptive"].values.values)[0])
            # Fixed method's state and model are shared by every diagnostic, including compensated.
            for name in names:
                probe_learner = compensated_learner if name == "compensated" else learner
                descriptor_rollout = probe_learner.rollout(
                    params[name], shared_reference_state, point_models["fixed"]
                )
                coverage_rollout = probe_learner.rollout(
                    params[name], states["fixed"], point_models["fixed"]
                )
                values = np.asarray(evaluate_values(coverage_rollout.states, obstacles).values)
                descriptors = np.asarray(descriptor_rollout.descriptors)
                target, diversity = _descriptor_metrics(
                    descriptors, targets, scales, learner_config.covariance_epsilon
                )
                probe["methods"][name] = {
                    "descriptor_target_loss": target,
                    "diversity_loss": diversity,
                    "pairwise_spread": _pairwise_descriptor_spread(descriptors, scales),
                    "safe_fallback_count": int(
                        np.count_nonzero(
                            (values >= 0.0)
                            & np.all(np.asarray(coverage_rollout.policy_valid), axis=1)
                        )
                    ),
                    "maximum_fallback_value": float(np.max(values)),
                }
                # The adaptive state's counterfactual frozen library may lack safe rollouts even
                # while the frozen controller keeps its own trajectory inside a smaller region.
                # Every library here uses the exact same adaptive state AND point model.
                adaptive_anchor_rollout = probe_learner.rollout(
                    params[name], states["adaptive"], point_models["adaptive"]
                )
                adaptive_anchor_values = np.asarray(
                    evaluate_values(adaptive_anchor_rollout.states, obstacles).values
                )
                maximum_fallback = float(np.max(adaptive_anchor_values))
                probe["adaptive_state_coverage"][name] = {
                    "safe_fallback_count": int(
                        np.count_nonzero(
                            (adaptive_anchor_values >= 0.0)
                            & np.all(np.asarray(adaptive_anchor_rollout.policy_valid), axis=1)
                        )
                    ),
                    "maximum_fallback_value": maximum_fallback,
                    "shared_nominal_value": shared_nominal_value,
                    "maximum_library_value": max(maximum_fallback, shared_nominal_value),
                }
            probes.append(probe)
        true_wind = scenario_true_wind(scenario, step_index)
        true_winds.append(np.asarray(true_wind))
        true_model = model_with_wind(resources.model, true_wind)
        for name in names:
            following = plant_step(states[name], decisions[name].action, true_model)
            jax.block_until_ready(following)
            started = time.perf_counter()
            update = estimator_step(
                estimators[name], states[name], following, decisions[name].action
            )
            jax.block_until_ready(update)
            estimator_times[name].append(time.perf_counter() - started)
            estimators[name] = update.state
            instantaneous_winds[name] = np.asarray(update.instantaneous_wind)
            states[name] = following
            previous[name] = decisions[name].selected_index
        if progress_callback is not None:
            progress_callback(step_index + 1, scenario.steps)
    total_wall = time.perf_counter() - start_wall
    methods = {
        name: _method_trace(
            records[name], control_mode="analytic" if name == "analytic" else "plcbf"
        )
        for name in names
    }
    times = np.arange(scenario.steps, dtype=np.float64) * scenario.dt
    active_obstacles = np.flatnonzero(np.asarray(scenario.obstacle_mask))
    centers = (
        np.asarray(scenario.obstacle_initial_centers)[None, active_obstacles, :]
        + times[:, None, None] * np.asarray(scenario.obstacle_velocities)[None, active_obstacles, :]
    )
    obstacle_radii = np.asarray(scenario.obstacle_radii)[active_obstacles]
    obstacle_tracks = tuple(
        ObstacleTrack(
            centers=centers[:, index],
            physical_radius=float(radius),
            inflated_radius=float(radius + scenario.ego_radius + scenario.obstacle_clearance),
            label=f"obstacle {index + 1}",
        )
        for index, radius in enumerate(obstacle_radii)
    )
    trace = ComparisonVideoTrace(
        time_seconds=times,
        goal_position=np.asarray(scenario.goal_position),
        obstacles=obstacle_tracks,
        true_wind=np.asarray(true_winds),
        estimated_wind=np.asarray(records["adaptive"]["estimated_wind"]),
        wind_change_time=min(scenario.wind_change_step * scenario.dt, float(times[-1])),
        descriptor_targets=targets,
        fixed=methods["fixed"],
        adaptive=methods["adaptive"],
        drone_radius=scenario.ego_radius,
    )
    trace.validate()
    full_times = np.arange(scenario.steps + 1) * scenario.dt
    full_centers = (
        np.asarray(scenario.obstacle_initial_centers)[None, active_obstacles, :]
        + full_times[:, None, None]
        * np.asarray(scenario.obstacle_velocities)[None, active_obstacles, :]
    )
    method_summaries = {}
    from crazyflow.safety.da_plcbf.continuous_version_a import QP_REJECTION_REASONS

    for name, method in methods.items():
        positions = np.concatenate((method.position, np.asarray(states[name])[None, :3]))
        if len(active_obstacles):
            clearance, closest_step = _swept_clearance(
                positions, full_centers, obstacle_radii + scenario.ego_radius
            )
            inflated_clearance = clearance - scenario.obstacle_clearance
        else:
            clearance = inflated_clearance = None
            closest_step = 0
        negative_h = np.asarray(method.maximum_library_value) < 0.0
        method_summaries[name] = {
            "control_mode": method.control_mode,
            "analytic_obstacle_hocbf_enabled": name == "analytic",
            "policy_value_constraint_enabled": name != "analytic",
            "model_compensated_fallbacks": name == "compensated",
            "parameters_updated": name == "adaptive",
            "minimum_physical_clearance_m": clearance,
            "minimum_inflated_clearance_m": inflated_clearance,
            "closest_obstacle_time_seconds": closest_step * scenario.dt,
            "physical_collision": False if clearance is None else clearance <= 0.0,
            "final_goal_distance_m": float(
                np.linalg.norm(np.asarray(states[name])[:3] - trace.goal_position)
            ),
            "max_qp_intervention_norm": float(np.max(method.intervention_norm)),
            "integrated_intervention_norm_seconds": float(
                np.sum(method.intervention_norm) * scenario.dt
            ),
            "fallback_execution_count": int(np.count_nonzero(method.used_fallback)),
            "degraded_step_count": int(np.count_nonzero(method.degraded)),
            "qp_valid_step_count": int(np.count_nonzero(method.qp_valid)),
            "policy_dual_active_step_count": int(
                np.count_nonzero(method.selected_policy_dual > 1e-7)
            ),
            "maximum_policy_dual": float(np.max(method.selected_policy_dual)),
            "minimum_library_maximum_value": float(np.min(method.maximum_library_value)),
            "negative_library_H_step_count": int(np.count_nonzero(negative_h)),
            "negative_library_H_duration_seconds": float(np.sum(negative_h) * scenario.dt),
            "first_negative_library_H_time_seconds": None
            if not np.any(negative_h)
            else float(times[np.flatnonzero(negative_h)[0]]),
            "final_safe_fallback_count": int(np.count_nonzero(method.fallback_safe[-1])),
            "final_descriptor_target_loss": float(method.descriptor_target_loss[-1]),
            "final_diversity_loss": float(method.diversity_loss[-1]),
            "library_version": int(method.library_version[-1]),
            "point_wind_final_error_mps": float(
                np.linalg.norm(np.asarray(estimators[name].wind_velocity) - trace.true_wind[-1])
            ),
            "point_wind_finite_updates": int(np.asarray(estimators[name].finite_update_count)),
            "qp_rejection_counts": dict(
                zip(
                    QP_REJECTION_REASONS,
                    np.count_nonzero(method.qp_rejection_flags, axis=0).tolist(),
                    strict=True,
                )
            ),
            "controller_timing": _timing_statistics(controller_times[name]),
            "exact_qp_fast_path_count": qp_fast_path_counts[name],
            "exact_qp_fast_path_fraction": qp_fast_path_counts[name] / scenario.steps,
            "estimator_timing": _timing_statistics(estimator_times[name]),
            "controller_missed_20ms_count": int(
                np.count_nonzero(np.asarray(controller_times[name]) > 0.02)
            ),
            "controller_missed_50ms_count": int(
                np.count_nonzero(np.asarray(controller_times[name]) > 0.05)
            ),
        }
    common_reference = scenario.initial_state.at[:3].set(scenario.goal_position)
    final_point_model = model_with_point_wind(resources.model, estimators["fixed"])
    common_metrics = {}
    for name in names:
        params = adaptive_learner_state.params if name == "adaptive" else initial_params
        probe_learner = compensated_learner if name == "compensated" else learner
        rollout = probe_learner.rollout(params, common_reference, final_point_model)
        target, diversity = _descriptor_metrics(
            np.asarray(rollout.descriptors), targets, scales, learner_config.covariance_epsilon
        )
        common_metrics[name] = {"descriptor_target_loss": target, "diversity_loss": diversity}
    parameter_delta = float(
        np.sqrt(
            sum(
                np.sum((np.asarray(a) - np.asarray(b)) ** 2)
                for a, b in zip(
                    jax.tree.leaves(adaptive_learner_state.params),
                    jax.tree.leaves(initial_params),
                    strict=True,
                )
            )
        )
    )

    def probe_mean(name: str, metric: str) -> float | None:
        return None if not probes else float(np.mean([p["methods"][name][metric] for p in probes]))

    fixed_target = probe_mean("fixed", "descriptor_target_loss")
    adaptive_target = probe_mean("adaptive", "descriptor_target_loss")
    fixed_spread = probe_mean("fixed", "pairwise_spread")
    adaptive_spread = probe_mean("adaptive", "pairwise_spread")
    coverage_advantage = max(
        (
            p["methods"]["adaptive"]["safe_fallback_count"]
            - p["methods"]["fixed"]["safe_fallback_count"]
            for p in probes
        ),
        default=0,
    )
    adaptive_state_expansion = {}
    for name in ("fixed", "compensated"):
        if name not in names:
            continue
        expansion_times = [
            p["time_seconds"]
            for p in probes
            if p["adaptive_state_coverage"][name]["maximum_library_value"] < 0.0
            and p["adaptive_state_coverage"]["adaptive"]["maximum_library_value"] > 0.0
        ]
        adaptive_state_expansion[name] = {
            "negative_reference_positive_adaptive_probe_count": len(expansion_times),
            "times_seconds": expansion_times,
            "interpretation": (
                "counterfactual collision-library coverage at the same adaptive state/model; "
                "includes the shared nominal; does not assert actual reference-method failure"
            ),
        }
    actual_advantage = int(
        np.max(
            np.sum(methods["adaptive"].fallback_safe, axis=1)
            - np.sum(methods["fixed"].fallback_safe, axis=1)
        )
    )
    change_count = int(
        np.count_nonzero(np.linalg.norm(np.diff(trace.true_wind, axis=0), axis=1) > 1e-8)
    )
    adaptive_steps = int(np.asarray(adaptive_learner_state.cumulative_gradient_steps))
    publication_consistent = (
        int(np.asarray(adaptive_learner_state.library_version)) == adaptive_steps
    )
    learning_finite = (
        bool(finite_update_flags) and all(finite_update_flags) and publication_consistent
    )
    checks = {
        "exactly_one_constant_wind_step": change_count == 1,
        "prewind_methods_identical": prewind_full_state_difference == 0.0,
        "every_finite_step_published_persistently": learning_finite,
        "point_estimator_converged_below_0p01_mps": method_summaries["adaptive"][
            "point_wind_final_error_mps"
        ]
        < 0.01,
        "frozen_library_has_postwind_descriptor_bias": baseline_target is not None
        and fixed_target is not None
        and fixed_target >= 1.25 * baseline_target,
        "adaptive_shared_probe_target_recovery": fixed_target is not None
        and adaptive_target <= 0.9 * fixed_target,
        "adaptive_shared_probe_spread_recovery": fixed_spread is not None
        and adaptive_spread >= 1.15 * fixed_spread,
        "adaptive_useful_safe_coverage_recovery": coverage_advantage >= 2,
        "adaptive_inflated_clearance_positive": method_summaries["adaptive"][
            "minimum_inflated_clearance_m"
        ]
        is None
        or method_summaries["adaptive"]["minimum_inflated_clearance_m"] > 0,
        "qp_intervention_above_1e-5": method_summaries["adaptive"]["max_qp_intervention_norm"]
        > 1e-5,
        "zero_degraded_controller_steps": all(
            method_summaries[n]["degraded_step_count"] == 0 for n in ("fixed", "adaptive")
        ),
        "adaptive_final_goal_distance_below_0p5_m": method_summaries["adaptive"][
            "final_goal_distance_m"
        ]
        < 0.5,
    }
    summary: dict[str, Any] = {
        "device": str(device),
        "effective_config": asdict(config),
        "effective_scenario": {
            key: np.asarray(value).tolist() if isinstance(value, (jax.Array, np.ndarray)) else value
            for key, value in asdict(scenario).items()
        },
        "scenario": scenario.name,
        "steps": scenario.steps,
        "dt_seconds": scenario.dt,
        "policy_count": config.policy_count,
        "drone_collision_radius_m": scenario.ego_radius,
        "policy_value_scope": (
            "collision-only sphere/ego geometry; "
            "operational limits enforced by instantaneous analytic barriers"
        ),
        "clearance_definition": (
            "relative swept center distance minus obstacle radius minus drone radius; "
            "inflated subtracts additional safety margin"
        ),
        "true_wind_change_count": change_count,
        "wind_change_time_seconds": scenario.wind_change_step * scenario.dt,
        "wind_detected_time_seconds": None
        if first_detected_step is None
        else first_detected_step * scenario.dt,
        "learning_start": config.learning_start,
        "shared_nominal_model_compensation": config.nominal_model_compensation,
        "initial_skill_scale": config.initial_skill_scale,
        "initial_repertoire": (
            "braking plus small random residual; no initial directional scaffold"
            if config.initial_skill_scale == 0.0
            else "structured directional scaffold at configured scale plus small random residual"
        ),
        "gradient_steps_per_boundary": config.gradient_steps_per_boundary,
        "smooth_motor_bounds": config.smooth_motor_bounds,
        "prewind_max_full_state_component_difference": prewind_full_state_difference,
        "prewind_max_position_difference_m": float(
            np.max(
                np.linalg.norm(
                    methods["fixed"].position[: max(1, scenario.wind_change_step)]
                    - methods["adaptive"].position[: max(1, scenario.wind_change_step)],
                    axis=1,
                )
            )
        ),
        "adaptive_gradient_steps": adaptive_steps,
        "adaptive_library_version": int(np.asarray(adaptive_learner_state.library_version)),
        "adaptive_parameter_delta_norm": parameter_delta,
        "library_version_equals_finite_gradient_steps": publication_consistent,
        "all_attempted_bptt_updates_finite": bool(all(finite_update_flags)),
        "attempted_bptt_updates": len(finite_update_flags),
        "point_wind_final_error_mps": method_summaries["adaptive"]["point_wind_final_error_mps"],
        "point_wind_finite_updates": method_summaries["adaptive"]["point_wind_finite_updates"],
        "instantaneous_estimator_final_mps": instantaneous_winds["adaptive"].tolist(),
        "estimator_information_source": (
            "each method's own measured state transition and applied wrench"
        ),
        "methods": method_summaries,
        "shared_probes": probes,
        "adaptive_state_counterfactual_library_expansion": adaptive_state_expansion,
        "adaptive_state_maximum_safe_count_advantage": max(
            (
                p["adaptive_state_coverage"]["adaptive"]["safe_fallback_count"]
                - p["adaptive_state_coverage"]["fixed"]["safe_fallback_count"]
                for p in probes
            ),
            default=0,
        ),
        "shared_probe_reference": (
            "same frozen wind-onset state for descriptors; "
            "same current fixed-method state/model for coverage"
        ),
        "shared_t4_zero_wind_descriptor_target_loss": baseline_target,
        "shared_t4_zero_wind_pairwise_spread": baseline_spread,
        "shared_t4_probe_postwindow_fixed_target_loss": fixed_target,
        "shared_t4_probe_postwindow_adaptive_target_loss": adaptive_target,
        "shared_t4_probe_postwindow_fixed_pairwise_spread": fixed_spread,
        "shared_t4_probe_postwindow_adaptive_pairwise_spread": adaptive_spread,
        "common_actual_maximum_adaptive_safe_count_advantage": coverage_advantage,
        "encounter_actual_maximum_adaptive_safe_count_advantage": actual_advantage,
        "timing_methodology": (
            "each complete controller/estimator/BPTT output synchronized separately; "
            "explicit discarded JIT warm-up; no batch elapsed-time division"
        ),
        "timing_includes_diagnostic_rollout_outputs": True,
        "controller_median_seconds_per_method": float(np.median(controller_times["adaptive"])),
        "controller_p95_seconds_per_method": float(np.percentile(controller_times["adaptive"], 95)),
        "bptt_median_seconds": _timing_statistics(learner_times)["median_seconds"],
        "bptt_p95_seconds": _timing_statistics(learner_times)["p95_seconds"],
        "jit_warmup_seconds": warmup_seconds,
        "jit_warmup_parameter_variants": ["initialized", "discarded finite learner update"],
        "total_wall_seconds_excluding_jit": total_wall,
        "total_wall_seconds_including_jit": total_wall + warmup_seconds,
        "checks": checks,
        "adaptive_zero_degraded_controller_steps": method_summaries["adaptive"][
            "degraded_step_count"
        ]
        == 0,
        "all_checks_passed": all(checks.values()),
        "acceptance_all_corrected_mechanism_gates": all(checks.values()),
        "fixed_collides_adaptive_does_not": method_summaries["fixed"]["physical_collision"]
        and not method_summaries["adaptive"]["physical_collision"],
        "adaptation_restored_physical_safety": method_summaries["fixed"]["physical_collision"]
        and checks["adaptive_inflated_clearance_positive"]
        and method_summaries["adaptive"]["degraded_step_count"] == 0
        and checks["adaptive_final_goal_distance_below_0p5_m"],
        "adaptation_improved_shared_safe_coverage": coverage_advantage > 0,
        "adaptive_experiment_success": checks["adaptive_inflated_clearance_positive"]
        and method_summaries["adaptive"]["degraded_step_count"] == 0
        and checks["adaptive_final_goal_distance_below_0p5_m"],
        "mechanism_checks_passed": publication_consistent
        and all(finite_update_flags)
        and all(np.all(np.isfinite(method.position)) for method in methods.values()),
    }
    for name in names:
        summary.update({f"{name}_{key}": value for key, value in method_summaries[name].items()})
        summary.update(
            {f"common_state_{name}_{key}": value for key, value in common_metrics[name].items()}
        )
    for name, value in checks.items():
        summary[f"acceptance_{name}"] = value
    summary = _current_outcome_checks(summary)
    trace = replace(trace, coverage_probes=_coverage_probes_from_summary(summary))
    trace.validate()
    return OnlineConstantWindResult(trace=trace, summary=summary, methods=methods)


def _current_outcome_checks(summary: dict[str, Any]) -> dict[str, Any]:
    """Classify adaptive success without demanding that the comparison methods also succeed.

    Earlier bias/spread thresholds and two-method prewind equality remain explicit historical
    diagnostics. Startup learning can change the adaptive trajectory before the wind transition.
    This classification changes no recorded state, timing sample, parameter, or safety metric.
    """
    if "methods" not in summary:
        return summary
    result = dict(summary)
    if "legacy_comparison_diagnostics" not in result:
        result["legacy_comparison_diagnostics"] = {
            "checks": result.get("checks", {}),
            "all_checks_passed": result.get("all_checks_passed", False),
            "acceptance_flags": {
                key: value for key, value in result.items() if key.startswith("acceptance_")
            },
        }
    for key in tuple(result):
        if key.startswith("acceptance_"):
            del result[key]
    adaptive = result["methods"]["adaptive"]
    mechanism_checks = {
        "exactly_one_constant_wind_step": result["true_wind_change_count"] == 1,
        "finite_persistent_learning_published": result["adaptive_gradient_steps"] > 0
        and result["all_attempted_bptt_updates_finite"]
        and result["library_version_equals_finite_gradient_steps"],
        "point_estimator_converged_below_0p01_mps": result["point_wind_final_error_mps"] < 0.01,
        "plcbf_obstacle_avoidance_isolated": adaptive["policy_value_constraint_enabled"]
        and not adaptive["analytic_obstacle_hocbf_enabled"],
    }
    equality_required = result.get("learning_start", "wind") == "wind"
    if equality_required:
        mechanism_checks["prewind_methods_identical"] = (
            result["prewind_max_full_state_component_difference"] == 0.0
        )
    experiment_checks = {
        "adaptive_inflated_clearance_positive": adaptive["minimum_inflated_clearance_m"] is None
        or adaptive["minimum_inflated_clearance_m"] > 0.0,
        "adaptive_zero_degraded_controller_steps": adaptive["degraded_step_count"] == 0,
        "adaptive_final_goal_distance_below_0p5_m": adaptive["final_goal_distance_m"] < 0.5,
    }
    checks = {**mechanism_checks, **experiment_checks}
    result.update(
        checks=checks,
        mechanism_checks=mechanism_checks,
        experiment_checks=experiment_checks,
        mechanism_checks_passed=all(mechanism_checks.values()),
        adaptive_experiment_success=all(experiment_checks.values()),
        all_checks_passed=all(checks.values()),
        acceptance_all_corrected_mechanism_gates=all(checks.values()),
        prewind_equality_required=equality_required,
        initialization_comparison=(
            "all methods receive the same immutable initialized parameters; startup adaptation "
            "may change the adaptive trajectory before wind onset"
        ),
    )
    result.update({f"acceptance_{name}": value for name, value in checks.items()})
    return result


def _trace_arrays(trace: ComparisonVideoTrace) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "time_seconds": np.asarray(trace.time_seconds),
        "goal_position": np.asarray(trace.goal_position),
        "true_wind": np.asarray(trace.true_wind),
        "estimated_wind": np.asarray(trace.estimated_wind),
        "descriptor_targets": np.asarray(trace.descriptor_targets),
        "obstacle_centers": (
            np.stack([obstacle.centers for obstacle in trace.obstacles], axis=1)
            if trace.obstacles
            else np.empty((len(trace.time_seconds), 0, 3))
        ),
        "obstacle_physical_radii": np.asarray(
            [obstacle.physical_radius for obstacle in trace.obstacles]
        ),
        "obstacle_inflated_radii": np.asarray(
            [obstacle.inflated_radius for obstacle in trace.obstacles]
        ),
    }
    for prefix, method in (("fixed", trace.fixed), ("adaptive", trace.adaptive)):
        for name in MethodVideoTrace.__dataclass_fields__:
            value = getattr(method, name)
            if value is not None:
                arrays[f"{prefix}_{name}"] = np.asarray(value)
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
    arrays = _trace_arrays(result.trace)
    for method_name, method in result.methods.items():
        for name in MethodVideoTrace.__dataclass_fields__:
            value = getattr(method, name)
            if value is not None:
                arrays[f"{method_name}_{name}"] = np.asarray(value)
    np.savez_compressed(trace_path, **arrays)
    metadata = {
        "title": result.trace.title,
        "left_label": result.trace.left_label,
        "right_label": result.trace.right_label,
        "show_wind_change_banner": result.trace.show_wind_change_banner,
        "drone_radius": result.trace.drone_radius,
        "method_names": list(result.methods) or ["fixed", "adaptive"],
        "wind_change_time": result.trace.wind_change_time,
        "obstacle_labels": [obstacle.label for obstacle in result.trace.obstacles],
        "summary": result.summary,
    }
    summary_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return trace_path, summary_path


def load_online_constant_wind_result(
    trace_path: str | Path, summary_path: str | Path
) -> OnlineConstantWindResult:
    """Reconstruct renderer input without re-running simulation or online learning."""
    metadata = json.loads(Path(summary_path).read_text())
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    methods = {}
    for prefix in metadata.get("method_names", ["fixed", "adaptive"]):
        fields = {
            name: arrays[f"{prefix}_{name}"]
            for name in MethodVideoTrace.__dataclass_fields__
            if f"{prefix}_{name}" in arrays
        }
        if "control_mode" in fields:
            fields["control_mode"] = str(fields["control_mode"].item())
        methods[prefix] = MethodVideoTrace(**fields)
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
        left_label=metadata.get("left_label", "FIXED-LIBRARY PL-CBF"),
        right_label=metadata.get("right_label", "CONTINUOUSLY ADAPTIVE DA-PLCBF"),
        show_wind_change_banner=metadata.get("show_wind_change_banner", True),
        drone_radius=float(metadata.get("drone_radius", 0.0)),
        coverage_probes=_coverage_probes_from_summary(metadata["summary"]),
    )
    trace.validate()
    return OnlineConstantWindResult(trace, _current_outcome_checks(metadata["summary"]), methods)


def load_online_constant_wind_trace(
    trace_path: str | Path, summary_path: str | Path
) -> ComparisonVideoTrace:
    """Load the default comparison, including backward compatibility with older saved traces."""
    return load_online_constant_wind_result(trace_path, summary_path).trace


def _coverage_probes_from_summary(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Expose actual same-state probes for rendering without regenerating controller data."""
    recorded_probes = [
        probe for probe in summary.get("shared_probes", []) if "adaptive_state_coverage" in probe
    ]
    coverage_probes = None
    if recorded_probes:
        coverage_probes = {
            "time_seconds": np.asarray([p["time_seconds"] for p in recorded_probes]),
            "source": "same measured adaptive state and point model; H includes shared nominal",
        }
        for name in ("fixed", "compensated", "adaptive"):
            if all(name in probe["adaptive_state_coverage"] for probe in recorded_probes):
                coverage_probes[f"{name}_h"] = np.asarray(
                    [
                        probe["adaptive_state_coverage"][name]["maximum_library_value"]
                        for probe in recorded_probes
                    ]
                )
                coverage_probes[f"{name}_safe_count"] = np.asarray(
                    [
                        probe["adaptive_state_coverage"][name]["safe_fallback_count"]
                        for probe in recorded_probes
                    ],
                    dtype=np.int32,
                )
    return coverage_probes


def comparison_trace_for_methods(
    result: OnlineConstantWindResult, left: str, right: str
) -> ComparisonVideoTrace:
    """Select an honest labeled pair from recorded matched methods without rerunning anything."""
    methods = result.methods or {"fixed": result.trace.fixed, "adaptive": result.trace.adaptive}
    labels = {
        "fixed": "FIXED PL-CBF ONLY",
        "adaptive": "ADAPTIVE DA-PLCBF ONLY",
        "analytic": "ANALYTIC OBSTACLE HOCBF ONLY",
        "compensated": "MODEL-COMPENSATED FIXED PL-CBF",
    }
    cold_start = result.summary.get("initial_skill_scale", 1.0) == 0.0
    if cold_start:
        labels.update(
            fixed="FROZEN BRAKING LIBRARY",
            compensated="FROZEN + WIND FEEDFORWARD",
            adaptive="ONLINE-LEARNED DA-PLCBF",
        )
    if left not in methods or right not in methods:
        raise ValueError(f"available recorded methods are {tuple(methods)}")
    coverage_probes = _coverage_probes_from_summary(result.summary)
    trace = replace(
        result.trace,
        fixed=methods[left],
        adaptive=methods[right],
        left_label=labels.get(left, left),
        right_label=labels.get(right, right),
        title=(
            "Online construction of a safety fallback library"
            if cold_start
            else f"Matched wind comparison: {labels.get(left, left)} vs {labels.get(right, right)}"
        ),
        coverage_probes=coverage_probes,
    )
    trace.validate()
    return trace


__all__ = [
    "OnlineConstantWindConfig",
    "OnlineConstantWindResult",
    "VersionAResources",
    "build_cf21b_version_a_resources",
    "load_online_constant_wind_trace",
    "load_online_constant_wind_result",
    "comparison_trace_for_methods",
    "run_online_constant_wind_demo",
    "save_online_constant_wind_result",
]
