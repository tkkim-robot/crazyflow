"""Matched competent-checkpoint experiments with measured learner availability.

A single obstacle-free nominal-model warmup produces a complete persistent learner checkpoint.
One shared physical prefix ends at the disturbance. Frozen, compensated, and adaptive branches
then start from the identical physical state and parameter/optimizer checkpoint. Probes are
computed after the episode and cannot change learning, selection, or the execution timeline.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.continuous_demo_scenarios import (
    constant_wind_scenario,
    model_with_wind,
    scenario_obstacle_window,
)
from crazyflow.safety.da_plcbf.continuous_version_a import runtime_policy_values
from crazyflow.safety.da_plcbf.deadline_schedule import BoundarySnapshotScheduler, CompletedSnapshot
from crazyflow.safety.da_plcbf.feasibility_reference import (
    FeasibilityReferenceConfig,
    run_feasibility_reference,
    save_feasibility_reference,
)
from crazyflow.safety.da_plcbf.learner_checkpoint import (
    load_learner_checkpoint,
    save_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
    ComparisonVideoTrace,
    MethodVideoTrace,
    ObstacleTrack,
)
from crazyflow.safety.da_plcbf.online_constant_wind import (
    OnlineConstantWindResult,
    VersionAResources,
    _append_method_record,
    _coverage_probes_from_summary,
    _descriptor_metrics,
    _empty_method_records,
    _make_controller,
    _method_trace,
    _timing_statistics,
    _trajectory_descriptors,
    build_cf21b_version_a_resources,
    save_online_constant_wind_result,
)
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    build_fibonacci_skill_spec,
    build_persistent_skill_learner,
    initialize_skill_actor,
    skill_library_competency,
)
from crazyflow.safety.da_plcbf.point_wind_estimator import (
    PointWindEstimatorConfig,
    initialize_point_wind_estimator,
    model_with_point_wind,
    update_point_wind_estimator,
)
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.rigid_payload import CenteredRigidPayload, hover_authority

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class CompetentExperimentConfig:
    """Predeclared physical, checkpoint, and runtime settings for a small matched experiment."""

    policy_count: int = 16
    seed: int = 7
    warmup_steps: int = 500
    learning_rate: float = 0.001
    dt: float = 0.02
    horizon: int = 60
    control_interval_steps: int = 2
    duration_seconds: float = 20.0
    event_time_seconds: float = 4.0
    wind_after: tuple[float, float, float] = (2.0, 0.8, 0.0)
    disturbance: str = "wind"
    model_mode: str = "oracle"
    adaptive_model_compensation: bool = True
    schedule: str = "budgeted"
    probe_every_controls: int = 10
    nominal_acceleration_limit: float = 1.2
    fallback_acceleration_limit: float = 2.5
    maximum_skill_speed: float = 0.9
    maximum_skill_duration: float = 0.7
    terminal_braking_weight: float = 2.0
    payload_mass_fraction: float = 0.25
    controller_reserve_seconds: float = 0.003
    update_safety_factor: float = 1.25

    @property
    def control_period(self) -> float:
        return self.dt * self.control_interval_steps

    def validate(self) -> None:
        for name in (
            "policy_count",
            "warmup_steps",
            "horizon",
            "control_interval_steps",
            "probe_every_controls",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.policy_count < 4 or self.control_interval_steps > self.horizon:
            raise ValueError("use at least four skills and a holding period within the horizon")
        for name in (
            "dt",
            "learning_rate",
            "duration_seconds",
            "event_time_seconds",
            "nominal_acceleration_limit",
            "fallback_acceleration_limit",
            "maximum_skill_speed",
            "maximum_skill_duration",
            "terminal_braking_weight",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive finite")
        if self.event_time_seconds >= self.duration_seconds:
            raise ValueError("disturbance must precede the episode end")
        for duration in (self.event_time_seconds, self.duration_seconds):
            if not np.isclose(
                duration / self.control_period, round(duration / self.control_period)
            ):
                raise ValueError("event and duration must align with control boundaries")
        if self.disturbance not in {"wind", "unchanged", "payload", "crossing"}:
            raise ValueError("unknown disturbance")
        if self.model_mode not in {"oracle", "estimated"} or self.schedule not in {
            "budgeted",
            "unlimited",
        }:
            raise ValueError("model_mode must be oracle/estimated; schedule budgeted/unlimited")
        if self.disturbance == "payload" and self.model_mode != "oracle":
            raise ValueError(
                "the payload experiment supplies known parameters; "
                "wind estimation is not a mass estimator"
            )
        if len(self.wind_after) != 3 or not np.all(np.isfinite(self.wind_after)):
            raise ValueError("wind must contain three finite components")
        if not isinstance(self.adaptive_model_compensation, bool):
            raise TypeError("adaptive_model_compensation must be boolean")
        if not math.isfinite(self.payload_mass_fraction) or self.payload_mass_fraction < 0:
            raise ValueError("payload_mass_fraction must be nonnegative finite")
        if (
            not math.isfinite(self.controller_reserve_seconds)
            or self.controller_reserve_seconds < 0
        ):
            raise ValueError("controller reserve must be nonnegative finite")
        if not math.isfinite(self.update_safety_factor) or self.update_safety_factor < 1:
            raise ValueError("update safety factor must be finite and at least one")


def _tree_digest(tree: Any) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        array = np.asarray(leaf)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


_PREFIX_CONFIG_FIELDS = (
    "policy_count",
    "seed",
    "warmup_steps",
    "learning_rate",
    "dt",
    "horizon",
    "control_interval_steps",
    "event_time_seconds",
    "nominal_acceleration_limit",
    "fallback_acceleration_limit",
    "maximum_skill_speed",
    "maximum_skill_duration",
    "terminal_braking_weight",
)


def _prefix_provenance(config: CompetentExperimentConfig, scenario: Any) -> dict[str, Any]:
    """Record actual prefix geometry and controls, independent of post-event experiment options."""
    # Keep original numeric values instead of regenerating the old scenario with newer code.
    # Episode duration/name and the post-event wind do not affect the executed calm prefix.
    scenario_values = {
        name: np.asarray(value).tolist()
        for name, value in asdict(scenario).items()
        if name not in {"name", "steps", "wind_after"}
    }
    return {
        "version": 1,
        "scenario": scenario_values,
        "prefix_config": {name: getattr(config, name) for name in _PREFIX_CONFIG_FIELDS},
        "controller_arguments": {
            "nominal_acceleration_limit": config.nominal_acceleration_limit,
            "waypoint_position_gain": 2.0,
            "waypoint_velocity_gain": 2.8,
            "nominal_model_compensation": True,
            "control_interval_steps": config.control_interval_steps,
        },
        # The factory contains internal filter/selection defaults that are not public experiment
        # options. A changed implementation must regenerate its physical prefix as well.
        "controller_factory_sha256": hashlib.sha256(
            (inspect.getsource(_controller) + inspect.getsource(_make_controller)).encode()
        ).hexdigest(),
    }


def validate_checkpoint_compatibility(
    config: CompetentExperimentConfig, metadata: dict[str, Any]
) -> None:
    """Reject overrides that silently change an already executed shared physical prefix."""
    stored = metadata.get("prefix_provenance")
    if not isinstance(stored, dict) or stored.get("version") != 1:
        raise ValueError(
            "checkpoint has no supported effective prefix provenance; regenerate it with "
            "prepare_competent_checkpoint before starting a new experiment"
        )
    original = CompetentExperimentConfig(**metadata["experiment_config"])
    changed = [
        name for name in _PREFIX_CONFIG_FIELDS if getattr(config, name) != getattr(original, name)
    ]
    if changed:
        raise ValueError(f"checkpoint and effective experiment differ in {changed}")
    effective = _prefix_provenance(config, _scenario(config))
    for section, expected in effective.items():
        if stored.get(section) != expected:
            raise ValueError(
                f"checkpoint effective prefix {section} differs; regenerate the shared checkpoint"
            )


def _scenario(config: CompetentExperimentConfig) -> Any:
    scenario = replace(
        constant_wind_scenario(),
        dt=config.dt,
        horizon=config.horizon,
        steps=round(config.duration_seconds / config.dt),
        wind_change_step=round(config.event_time_seconds / config.dt),
        wind_after=jnp.asarray(
            config.wind_after if config.disturbance == "wind" else (0.0, 0.0, 0.0),
            dtype=jnp.float32,
        ),
        name=f"competent_{config.disturbance}_{config.model_mode}",
    )
    if config.disturbance == "crossing":
        # One prescribed crossing trajectory is shared by every method. Its absolute-time
        # prediction and derivative enter only the runtime filter, never the skill learner.
        scenario = replace(
            scenario,
            obstacle_initial_centers=jnp.asarray(
                [[5.8, -4.2, 1.4], [7.5, 0.5, 2.7]], dtype=jnp.float32
            ),
            obstacle_velocities=jnp.asarray([[0.0, 0.7, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32),
        )
    scenario.validate()
    return scenario


def _controller(
    scenario: Any,
    resources: VersionAResources,
    spec: Any,
    learner_config: PersistentSkillConfig,
    config: CompetentExperimentConfig,
    device: Any,
) -> Any:
    return _make_controller(
        scenario,
        resources,
        spec,
        learner_config,
        nominal_acceleration_limit=config.nominal_acceleration_limit,
        waypoint_position_gain=2.0,
        waypoint_velocity_gain=2.8,
        device=device,
        nominal_model_compensation=True,
        control_interval_steps=config.control_interval_steps,
    )


def _record(
    records: Any,
    state: Any,
    decision: Any,
    learner_state: Any,
    spec: Any,
    learner_config: Any,
    model: Any,
    gradient_norm: float = 0.0,
    parameter_update_norm: float = 0.0,
    **timing: Any,
) -> None:
    descriptors = _trajectory_descriptors(np.asarray(decision.candidates.states)[1:])
    target, diversity = _descriptor_metrics(
        descriptors,
        np.asarray(spec.target_descriptors),
        np.asarray(learner_config.descriptor_scales),
        learner_config.covariance_epsilon,
    )
    _append_method_record(
        records,
        state,
        decision,
        library_version=int(np.asarray(learner_state.library_version)),
        cumulative_gradient_steps=int(np.asarray(learner_state.cumulative_gradient_steps)),
        diversity_loss=diversity,
        descriptor_target_loss=target,
        gradient_norm=gradient_norm,
        parameter_update_norm=parameter_update_norm,
        estimated_wind=model.wind_velocity,
        **timing,
    )


def prepare_competent_checkpoint(
    config: CompetentExperimentConfig, directory: Path, device: Any
) -> Path:
    """Learn attainable obstacle-free motion targets, then execute one shared calm prefix."""
    config.validate()
    directory.mkdir(parents=True, exist_ok=False)
    resources = jax.device_put(build_cf21b_version_a_resources(), device)
    scenario = _scenario(config)
    learner_config = PersistentSkillConfig(
        dt=config.dt,
        horizon=config.horizon,
        learning_rate=config.learning_rate,
        acceleration_limit=config.fallback_acceleration_limit,
        target_weight=10.0,
        diversity_weight=0.001,
        pairwise_weight=0.005,
        trust_weight=0.001,
        terminal_braking_weight=config.terminal_braking_weight,
        smooth_motor_bounds=False,
        initial_skill_scale=1.0,
    )
    spec = jax.device_put(
        build_fibonacci_skill_spec(
            policy_count=config.policy_count,
            maximum_speed=config.maximum_skill_speed,
            maximum_duration=config.maximum_skill_duration,
            horizon_duration=config.dt * config.horizon,
        ),
        device,
    )
    with jax.default_device(device):
        params = initialize_skill_actor(jax.random.key(config.seed), spec, learner_config)
    learner = build_persistent_skill_learner(
        spec, resources.actuator, learner_config, device=device
    )
    persistent = learner.initialize(params, resources.model)
    initial_state = jax.device_put(scenario.initial_state, device)
    initial_rollout = learner.rollout(params, initial_state, resources.model)
    jax.block_until_ready(initial_rollout)
    warmup_start = time.perf_counter()
    warmup_rows = []
    for index in range(config.warmup_steps):
        # Proprioceptive warmup samples are specified independently of all obstacle geometry.
        # Rest samples establish directional behavior; modest forward motion checks reuse.
        sample = initial_state.at[7].set(0.5 if index % 5 == 4 else 0.0)
        started = time.perf_counter()
        persistent, metrics = learner.step(persistent, sample, resources.model)
        jax.block_until_ready((persistent, metrics))
        warmup_rows.append(
            {
                "step": index + 1,
                "seconds": time.perf_counter() - started,
                "finite": bool(np.asarray(metrics.finite_update_applied)),
                "spatial_target_loss": float(np.asarray(metrics.loss.descriptor_target)),
                "terminal_braking_loss": float(np.asarray(metrics.loss.terminal_braking)),
            }
        )
    measured = learner.rollout(persistent.params, initial_state, resources.model)
    jax.block_until_ready(measured)
    competency = skill_library_competency(measured, spec, learner_config)
    controller = _controller(scenario, resources, spec, learner_config, config, device)
    previous = jnp.asarray(-1, dtype=jnp.int32)
    jax.block_until_ready(
        controller(
            initial_state,
            persistent.params,
            resources.model,
            scenario_obstacle_window(scenario, 0),
            previous,
        )
    )
    plant = jax.jit(
        lambda state, action: direct_wrench_symplectic_step(
            state, action, resources.model, config.dt
        )
    )
    records = _empty_method_records()
    physical = initial_state
    physical_states = [np.asarray(physical)]
    for step in range(0, scenario.wind_change_step, config.control_interval_steps):
        decision = controller(
            physical,
            persistent.params,
            resources.model,
            scenario_obstacle_window(scenario, step),
            previous,
        )
        jax.block_until_ready(decision)
        _record(records, physical, decision, persistent, spec, learner_config, resources.model)
        for _ in range(config.control_interval_steps):
            physical = plant(physical, decision.action)
            physical_states.append(np.asarray(physical))
        previous = decision.selected_index
    prefix = _method_trace(records)
    arrays = {
        name: np.asarray(getattr(prefix, name))
        for name in MethodVideoTrace.__dataclass_fields__
        if getattr(prefix, name) is not None
    }
    arrays["physical_states"] = np.asarray(physical_states)
    np.savez_compressed(directory / "shared_prefix.npz", **arrays)
    np.savez_compressed(
        directory / "nominal_repertoire.npz",
        initial_states=np.asarray(initial_rollout.states),
        learned_states=np.asarray(measured.states),
        targets=np.asarray(spec.target_descriptors),
    )
    event_rollout = learner.rollout(persistent.params, physical, resources.model)
    jax.block_until_ready(event_rollout)
    metadata = {
        "experiment_config": asdict(config),
        "prefix_file": "shared_prefix.npz",
        "prefix_npz_sha256": hashlib.sha256(
            (directory / "shared_prefix.npz").read_bytes()
        ).hexdigest(),
        "prefix_provenance": _prefix_provenance(config, scenario),
        "event_time_seconds": config.event_time_seconds,
        "warmup_kind": "shared obstacle-free point-model warmup",
        "warmup_wall_seconds": time.perf_counter() - warmup_start,
        "warmup_updates": warmup_rows,
        "competency": competency,
        "event_state_competency": skill_library_competency(event_rollout, spec, learner_config),
        "parameter_sha256": _tree_digest(persistent.params),
        "optimizer_sha256": _tree_digest(persistent.optimizer_state),
        "physical_model": "cf21B_500",
        "previous_policy_index": int(np.asarray(previous)),
    }
    stem = directory / "competent_checkpoint"
    save_learner_checkpoint(
        persistent, spec, learner_config, resources.actuator, physical, stem, metadata=metadata
    )
    print(json.dumps({"checkpoint": str(stem), "competency": competency}), flush=True)
    return stem


def _load_prefix(
    stem: Path, *, metadata: dict[str, Any], expected_final_state: Any
) -> tuple[dict[str, list[Any]], np.ndarray]:
    """Read the authenticated recorded prefix and verify its checkpoint boundary state."""
    expected_hash = metadata.get("prefix_npz_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("checkpoint has no prefix checksum; regenerate the shared checkpoint")
    filename = metadata.get("prefix_file")
    if filename != "shared_prefix.npz":
        raise ValueError("checkpoint prefix file must be shared_prefix.npz")
    prefix_bytes = (stem.parent / filename).read_bytes()
    if hashlib.sha256(prefix_bytes).hexdigest() != expected_hash:
        raise ValueError("checkpoint prefix checksum mismatch; recorded prefix was changed")
    original = CompetentExperimentConfig(**metadata["experiment_config"])
    control_count = round(original.event_time_seconds / original.control_period)
    physical_count = round(original.event_time_seconds / original.dt) + 1
    with np.load(io.BytesIO(prefix_bytes), allow_pickle=False) as data:
        missing = set(_empty_method_records()) - set(data.files)
        if missing:
            raise ValueError(f"checkpoint prefix is missing telemetry fields: {sorted(missing)}")
        records = {name: list(data[name]) for name in _empty_method_records()}
        physical_states = data["physical_states"]
    if physical_states.shape != (physical_count, 13) or not np.all(np.isfinite(physical_states)):
        raise ValueError("checkpoint prefix has invalid physical state samples")
    if any(len(values) != control_count for values in records.values()):
        raise ValueError("checkpoint prefix telemetry does not match its control interval")
    if not np.array_equal(physical_states[-1], np.asarray(expected_final_state)):
        raise ValueError(
            "checkpoint prefix final physical state differs from the learner checkpoint"
        )
    if not np.array_equal(
        np.asarray(records["full_state"]), physical_states[: -1 : original.control_interval_steps]
    ):
        raise ValueError("checkpoint prefix control states differ from its plant state samples")
    return records, physical_states


def _first_time(mask: np.ndarray, times: np.ndarray) -> float | None:
    indices = np.flatnonzero(mask)
    return float(times[indices[0]]) if indices.size else None


def _summarize(
    method: MethodVideoTrace,
    full_states: np.ndarray,
    scenario: Any,
    times: np.ndarray,
    timing: dict[str, Any],
) -> dict[str, Any]:
    physical_times = np.arange(len(full_states)) * scenario.dt
    centers = (
        np.asarray(scenario.obstacle_initial_centers)[None]
        + physical_times[:, None, None] * np.asarray(scenario.obstacle_velocities)[None]
    )
    relative = full_states[:, None, :3] - centers
    delta = np.diff(relative, axis=0)
    denominator = np.sum(delta * delta, axis=-1)
    fraction = np.clip(
        -np.sum(relative[:-1] * delta, axis=-1) / np.maximum(denominator, 1e-20), 0.0, 1.0
    )
    clearance = (
        np.linalg.norm(relative[:-1] + fraction[..., None] * delta, axis=-1)
        - np.asarray(scenario.obstacle_radii)[None]
        - scenario.ego_radius
    )
    per_interval = np.min(clearance[:, np.asarray(scenario.obstacle_mask)], axis=1)
    shell = per_interval - scenario.obstacle_clearance
    modes = np.asarray(method.execution_mode)
    hard = np.asarray(method.maximum_library_value)
    first_h = _first_time(hard < 0, times)
    first_shell = _first_time(shell < 0, physical_times[:-1])
    first_collision = _first_time(per_interval <= 0, physical_times[:-1])
    result = {
        "minimum_physical_clearance_m": float(per_interval.min()),
        "minimum_inflated_clearance_m": float(shell.min()),
        "physical_collision": bool(np.any(per_interval <= 0)),
        "final_goal_distance_m": float(
            np.linalg.norm(full_states[-1, :3] - np.asarray(scenario.goal_position))
        ),
        "negative_hard_H_intervals": int(np.sum(hard < 0)),
        "zero_selectable_intervals": int(np.sum(np.asarray(method.eligible_candidate_count) == 0)),
        "qp_execution_count": int(np.sum(modes == 0)),
        "fallback_execution_count": int(np.sum(modes == 1)),
        "emergency_execution_count": int(np.sum(modes == 2)),
        "midpoint_execution_count": int(np.sum(modes == 3)),
        "degraded_step_count": int(np.sum(method.degraded)),
        "positive_dual_on_executed_QP_count": int(
            np.sum(np.asarray(method.executed_policy_dual) > 1e-7)
        ),
        "minimum_motor_margin_N": float(np.min(method.actuator_margins)),
        "minimum_operational_residual": float(np.min(method.operational_residuals)),
        "first_negative_hard_H_time": first_h,
        "first_zero_selectable_time": _first_time(
            np.asarray(method.eligible_candidate_count) == 0, times
        ),
        "first_emergency_time": _first_time(modes == 2, times),
        "first_shell_violation_time": first_shell,
        "first_physical_collision_time": first_collision,
        "failure_sequence": (
            "Recorded events are chronology, not proof that every safe control was impossible."
        ),
        **timing,
    }
    result["safe_goal_success"] = (
        result["minimum_inflated_clearance_m"] > 0 and result["final_goal_distance_m"] < 0.5
    )
    return result


def run_competent_experiment(
    config: CompetentExperimentConfig,
    directory: Path,
    *,
    checkpoint_stem: Path | None = None,
    device: Any = None,
    progress_callback: Any = None,
) -> OnlineConstantWindResult:
    """Run each branch separately on the same GPU, then evaluate symmetric counterfactuals.

    Budgeted mode paces each method's control boundaries against a monotonic clock. The controller
    is serviced first, one serialized learner call may use measured slack, and its complete result
    is published only on the next boundary. This is a sampled-simulation deadline experiment;
    sensor/actuation transport delays and hard operating-system real-time guarantees are excluded.
    """
    config.validate()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    device = device or jax.devices()[0]
    if checkpoint_stem is None:
        checkpoint_stem = prepare_competent_checkpoint(config, directory / "checkpoint", device)
    bundle = load_learner_checkpoint(checkpoint_stem, device=device)
    validate_checkpoint_compatibility(config, bundle.metadata)
    scenario = _scenario(config)
    if not np.isclose(bundle.metadata["event_time_seconds"], config.event_time_seconds):
        raise ValueError("checkpoint branch time differs from the experiment")
    if bundle.config.dt != config.dt or bundle.config.horizon != config.horizon:
        raise ValueError("checkpoint prediction duration/discretization must be preserved")
    resources = VersionAResources(bundle.point_model, bundle.actuator)
    payload = CenteredRigidPayload(
        float(np.asarray(resources.model.mass)) * config.payload_mass_fraction
    )
    event_model = (
        payload.apply(resources.model) if config.disturbance == "payload" else resources.model
    )
    if config.disturbance == "payload":
        # This centered payload fits inside the existing collision enclosure. A larger
        # payload needs time-varying geometry, including in the stored shared prefix.
        if payload.enclosing_radius(scenario.ego_radius) > scenario.ego_radius:
            raise ValueError("payload extends beyond the experiment's fixed collision enclosure")
    event_model = model_with_wind(event_model, scenario.wind_after)
    fixed_learner = build_persistent_skill_learner(
        bundle.spec, bundle.actuator, bundle.config, device=device
    )
    comp_config = replace(bundle.config, model_compensation=True)
    comp_learner = build_persistent_skill_learner(
        bundle.spec, bundle.actuator, comp_config, device=device
    )
    adaptive_config = comp_config if config.adaptive_model_compensation else bundle.config
    learner = comp_learner if config.adaptive_model_compensation else fixed_learner
    controllers = {
        "fixed": _controller(scenario, resources, bundle.spec, bundle.config, config, device),
        "compensated": _controller(scenario, resources, bundle.spec, comp_config, config, device),
    }
    controllers["adaptive"] = (
        controllers["compensated"] if config.adaptive_model_compensation else controllers["fixed"]
    )
    estimator_config = PointWindEstimatorConfig(response_rate=2.4)
    plant = jax.jit(
        lambda state, action, model: direct_wrench_symplectic_step(state, action, model, config.dt)
    )
    estimate = jax.jit(
        lambda estimator, state, following, action: update_point_wind_estimator(
            estimator,
            state,
            following,
            action,
            resources.model,
            dt=config.dt,
            config=estimator_config,
        )
    )
    event_step = scenario.wind_change_step
    event_obstacles = scenario_obstacle_window(scenario, event_step)
    previous_initial = jnp.asarray(bundle.metadata["previous_policy_index"], dtype=jnp.int32)
    discarded, metric = learner.step(bundle.state, bundle.physical_state, event_model)
    jax.block_until_ready((discarded, metric))
    for controller in controllers.values():
        jax.block_until_ready(
            controller(
                bundle.physical_state,
                bundle.state.params,
                event_model,
                event_obstacles,
                previous_initial,
            )
        )
        # XLA placement signatures can differ between deserialized arrays and outputs of the
        # plant/selector. Warm the actual closed-loop path as well as the initial checkpoint.
        warm_state, warm_previous = bundle.physical_state, previous_initial
        for warm_index in range(3):
            warm_decision = controller(
                warm_state,
                discarded.params if warm_index == 2 else bundle.state.params,
                event_model,
                event_obstacles,
                warm_previous,
            )
            jax.block_until_ready(warm_decision)
            for _ in range(config.control_interval_steps):
                warm_state = plant(warm_state, warm_decision.action, event_model)
            jax.block_until_ready(warm_state)
            warm_previous = warm_decision.selected_index
        jax.block_until_ready(
            controller(
                bundle.physical_state,
                discarded.params,
                event_model,
                event_obstacles,
                previous_initial,
            )
        )
    jax.block_until_ready(
        comp_learner.rollout(bundle.state.params, bundle.physical_state, event_model)
    )
    following = plant(
        bundle.physical_state,
        jnp.asarray([float(np.asarray(event_model.mass)) * 9.81, 0.0, 0.0, 0.0]),
        event_model,
    )
    jax.block_until_ready(
        estimate(
            initialize_point_wind_estimator(),
            bundle.physical_state,
            following,
            jnp.asarray([0.35, 0.0, 0.0, 0.0]),
        )
    )
    if config.model_mode == "estimated":
        # Estimator-produced arrays can have different JAX placement signatures from a
        # checkpoint or prescribed oracle model. Warm the exact live dataflow, including
        # the next optimizer snapshot, before starting any measured deadline clock.
        for controller in (controllers["fixed"], controllers["compensated"]):
            warm_state, warm_previous = bundle.physical_state, previous_initial
            warm_estimator = initialize_point_wind_estimator()
            warm_learning = bundle.state
            for _ in range(3):
                warm_model = model_with_point_wind(resources.model, warm_estimator)
                warm_decision = controller(
                    warm_state, warm_learning.params, warm_model, event_obstacles, warm_previous
                )
                warm_learning, warm_metric = learner.step(warm_learning, warm_state, warm_model)
                jax.block_until_ready((warm_decision, warm_learning, warm_metric))
                for _ in range(config.control_interval_steps):
                    next_warm_state = plant(warm_state, warm_decision.action, event_model)
                    warm_estimator = estimate(
                        warm_estimator, warm_state, next_warm_state, warm_decision.action
                    ).state
                    warm_state = next_warm_state
                jax.block_until_ready((warm_state, warm_estimator))
                warm_previous = warm_decision.selected_index
    warm_durations = []
    for _ in range(3):
        started = time.perf_counter()
        jax.block_until_ready(learner.step(bundle.state, bundle.physical_state, event_model))
        warm_durations.append(time.perf_counter() - started)
    methods, summaries, snapshots, dense_states = {}, {}, {}, {}
    all_times = np.arange(0, scenario.steps, config.control_interval_steps) * config.dt
    prefix_count = round(config.event_time_seconds / config.control_period)
    names = ("fixed", "compensated", "adaptive")
    event_hashes = {}
    for name in names:
        records, prefix_states = _load_prefix(
            checkpoint_stem, metadata=bundle.metadata, expected_final_state=bundle.physical_state
        )
        full_states = list(prefix_states)
        physical, previous = bundle.physical_state, previous_initial
        estimator = initialize_point_wind_estimator()
        scheduler = BoundarySnapshotScheduler(
            bundle.state,
            int(np.asarray(bundle.state.library_version)),
            max(warm_durations),
            config.controller_reserve_seconds,
            config.update_safety_factor,
        )
        published = bundle.state
        branch_snapshots = []
        timings, update_timings, services, finite_updates = [], [], [], []
        event_hashes[name] = {
            "physical_state": _tree_digest(physical),
            "parameters": _tree_digest(published.params),
            "optimizer": _tree_digest(published.optimizer_state),
        }
        epoch = time.perf_counter()
        for local_index, step in enumerate(
            range(event_step, scenario.steps, config.control_interval_steps)
        ):
            simulation_time = step * config.dt
            scheduled_boundary = epoch + local_index * config.control_period
            if config.schedule == "budgeted":
                delay = scheduled_boundary - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            boundary = time.perf_counter()
            deadline = (
                scheduled_boundary + config.control_period
                if config.schedule == "budgeted"
                else math.inf
            )
            if name == "adaptive":
                published_snapshot = scheduler.publish(boundary, simulation_time)
                published = published_snapshot.state
            else:
                published_snapshot = scheduler.published
            point_model = (
                event_model
                if config.model_mode == "oracle"
                else model_with_point_wind(resources.model, estimator)
            )
            started = time.perf_counter()
            decision = controllers[name](
                physical,
                published.params,
                point_model,
                scenario_obstacle_window(scenario, step),
                previous,
            )
            jax.block_until_ready(decision)
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            missed = time.perf_counter() > deadline
            _record(
                records,
                physical,
                decision,
                published,
                bundle.spec,
                adaptive_config
                if name == "adaptive"
                else comp_config
                if name == "compensated"
                else bundle.config,
                point_model,
                snapshot_age_seconds=simulation_time - published_snapshot.training_simulation_time,
                controller_seconds=elapsed,
                missed_deadline=missed,
                gradient_norm=published_snapshot.gradient_norm,
                parameter_update_norm=published_snapshot.parameter_update_norm,
            )
            if local_index % config.probe_every_controls == 0:
                branch_snapshots.append((simulation_time, physical, point_model, published.params))
            training_state = physical
            for _ in range(config.control_interval_steps):
                following = plant(physical, decision.action, event_model)
                if config.model_mode == "estimated":
                    estimator = estimate(estimator, physical, following, decision.action).state
                physical = following
                full_states.append(np.asarray(physical))
            # Estimator completion is part of control-period service even when no learner
            # is launched; synchronizing only the plant state would undercount its work.
            jax.block_until_ready((physical, estimator))
            previous = decision.selected_index
            update_seconds = 0.0
            if name == "adaptive" and scheduler.can_start(time.perf_counter(), deadline):
                update_start = time.perf_counter()
                changed, metrics = learner.step(published, training_state, point_model)
                jax.block_until_ready((changed, metrics))
                update_finish = time.perf_counter()
                update_seconds = update_finish - update_start
                finite_updates.append(bool(np.asarray(metrics.finite_update_applied)))
                scheduler.complete(
                    CompletedSnapshot(
                        changed,
                        int(np.asarray(changed.library_version)),
                        simulation_time,
                        update_start,
                        update_finish,
                        float(np.asarray(metrics.gradient_norm)),
                        float(np.asarray(metrics.parameter_update_norm)),
                    )
                )
                update_timings.append(update_seconds)
            end = time.perf_counter()
            records["learner_seconds"][-1] = update_seconds
            records["missed_deadline"][-1] = missed or end > deadline
            services.append(
                {
                    "simulation_time": simulation_time,
                    "scheduled_wall_time": scheduled_boundary - epoch,
                    "started_wall_time": boundary - epoch,
                    "completed_wall_time": end - epoch,
                    "controller_seconds": elapsed,
                    "learner_seconds": update_seconds,
                    "missed_deadline": bool(end > deadline),
                    "snapshot_version": int(np.asarray(published.library_version)),
                }
            )
            if progress_callback and local_index % 50 == 0:
                progress_callback(
                    name,
                    local_index,
                    round(
                        (config.duration_seconds - config.event_time_seconds)
                        / config.control_period
                    ),
                )
        if config.schedule == "budgeted":
            terminal_boundary = epoch + config.duration_seconds - config.event_time_seconds
            if (remaining := terminal_boundary - time.perf_counter()) > 0:
                time.sleep(remaining)
        scheduler.publish(time.perf_counter(), config.duration_seconds)
        method = _method_trace(records)
        methods[name] = method
        dense_states[name] = np.asarray(full_states)
        # Include genuinely recorded pre-disturbance repertoire observations in the same grid.
        prefix_probes = []
        for prefix_index in range(0, prefix_count, config.probe_every_controls):
            prefix_probes.append(
                (
                    float(all_times[prefix_index]),
                    jnp.asarray(prefix_states[prefix_index * config.control_interval_steps]),
                    resources.model,
                    bundle.state.params,
                )
            )
        snapshots[name] = prefix_probes + branch_snapshots
        timing = {
            "controller_timing": _timing_statistics(timings),
            "learner_timing": _timing_statistics(update_timings),
            "attempted_updates": len(finite_updates),
            "finite_updates": sum(finite_updates),
            "initial_library_version": int(np.asarray(bundle.state.library_version)),
            "final_library_version": int(np.asarray(scheduler.published.version))
            if name == "adaptive"
            else int(np.asarray(bundle.state.library_version)),
            "last_controller_used_library_version": int(method.library_version[-1]),
            "deadline_misses": sum(x["missed_deadline"] for x in services),
            "runtime_services": services,
            "snapshot_publications": [
                {
                    **p,
                    "completed_wall_time": p["completed_wall_time"] - epoch,
                    "published_wall_time": p["published_wall_time"] - epoch,
                }
                for p in scheduler.publications
            ],
            "maximum_snapshot_age_seconds": float(
                np.max(method.snapshot_age_seconds[prefix_count:])
            ),
        }
        summaries[name] = _summarize(method, dense_states[name], scenario, all_times, timing)
        if name == "adaptive":
            save_learner_checkpoint(
                scheduler.published.state,
                bundle.spec,
                adaptive_config,
                bundle.actuator,
                physical,
                directory / "final_adaptive_checkpoint",
                metadata={"experiment_config": asdict(config)},
            )
        print(
            json.dumps(
                {
                    "method": name,
                    "result": {
                        k: v
                        for k, v in summaries[name].items()
                        if k not in ("runtime_services", "snapshot_publications")
                    },
                }
            ),
            flush=True,
        )
    # A supplied detour is a numerical witness of remaining control authority. It is measured
    # after timed branches and never becomes a learner target or a policy admission condition.
    detour = np.asarray([[4.0, 1.8, 2.2], [8.5, 1.8, 2.2]], dtype=np.float32)
    witness = run_feasibility_reference(
        scenario,
        resources,
        bundle.physical_state,
        start_step=event_step,
        waypoints=detour,
        config=FeasibilityReferenceConfig(acceleration_limit=config.nominal_acceleration_limit),
        model_at_step=lambda _step: event_model,
        device=device,
    )
    save_feasibility_reference(witness, directory / "feasibility_reference")
    # Symmetric probes of every library at every method's state/model, plus a fixed neutral
    # reference. All snapshots below were already published at their recorded sample time.
    probe_arrays: dict[str, list[Any]] = {"time_seconds": [], "reference_position": []}
    for name in names:
        probe_arrays[f"{name}_rollouts"] = []
        probe_arrays[f"{name}_safe"] = []
    symmetric_arrays: dict[str, list[Any]] = {}
    probes = []
    neutral = jnp.asarray(scenario.initial_state)
    neutral_values = jax.jit(
        lambda states, obstacles: runtime_policy_values(
            states,
            obstacles,
            obstacle_clearance=scenario.obstacle_clearance,
            ego_radius=scenario.ego_radius,
        )
    )
    for probe_index in range(len(snapshots["adaptive"])):
        when = snapshots["adaptive"][probe_index][0]
        obstacles = scenario_obstacle_window(scenario, round(when / config.dt))
        row: dict[str, Any] = {"time_seconds": when, "anchors": {}}
        probe_arrays["time_seconds"].append(when)
        probe_arrays["reference_position"].append(np.asarray(neutral[:3]))
        for anchor in names:
            _, anchor_state, anchor_model, _ = snapshots[anchor][probe_index]
            values = {}
            for candidate in names:
                params = snapshots[candidate][probe_index][3]
                evaluator_controller = (
                    controllers["fixed"]
                    if when < config.event_time_seconds
                    else controllers[candidate]
                )
                decision = evaluator_controller(
                    anchor_state, params, anchor_model, obstacles, previous_initial
                )
                jax.block_until_ready(decision)
                hard = np.asarray(decision.values.values)
                values[candidate] = {
                    "maximum_library_value": float(hard.max()),
                    "maximum_fallback_value": float(hard[1:].max()),
                    "safe_fallback_count": int(
                        np.sum(
                            (hard[1:] >= 0)
                            & np.asarray(decision.candidates.valid)[1:]
                            & np.asarray(decision.values.input_valid)[1:]
                        )
                    ),
                    "selectable_count": int(np.asarray(decision.eligible_candidate_count)),
                    "qp_accepted": bool(np.asarray(decision.qp_valid)),
                    "selected_smooth_value": float(np.asarray(decision.selected_smooth_value)),
                }
                key = f"{anchor}__{candidate}"
                symmetric_arrays.setdefault(f"{key}_states", []).append(
                    np.asarray(decision.candidates.states)
                )
                symmetric_arrays.setdefault(f"{key}_values", []).append(hard)
            row["anchors"][anchor] = {
                "full_state": np.asarray(anchor_state).tolist(),
                "point_wind": np.asarray(anchor_model.wind_velocity).tolist(),
                "libraries": values,
            }
        row["adaptive_state_coverage"] = row["anchors"]["adaptive"]["libraries"]
        row["adaptive_state_full_state"] = row["anchors"]["adaptive"]["full_state"]
        row["adaptive_state_point_wind"] = row["anchors"]["adaptive"]["point_wind"]
        row["neutral_reference_competency"] = {}
        reference_model = snapshots["adaptive"][probe_index][2]
        for name in names:
            evaluator = {"fixed": fixed_learner, "compensated": comp_learner, "adaptive": learner}[
                name
            ]
            if when < config.event_time_seconds:
                evaluator = fixed_learner
            rollout = evaluator.rollout(snapshots[name][probe_index][3], neutral, reference_model)
            jax.block_until_ready(rollout)
            probe_arrays[f"{name}_rollouts"].append(np.asarray(rollout.states)[..., :3])
            # This unobstructed fixed reference measures repertoire geometry, not encounter safety.
            clear = neutral_values(rollout.states, obstacles)
            probe_arrays[f"{name}_safe"].append(
                (np.asarray(clear.values) >= 0)
                & np.asarray(clear.input_valid)
                & np.all(np.asarray(rollout.policy_valid), axis=1)
            )
            row["neutral_reference_competency"][name] = skill_library_competency(
                rollout, bundle.spec, bundle.config
            )
        probes.append(row)
    repertoire = {key: np.asarray(value) for key, value in probe_arrays.items()}
    repertoire.update(
        source=(
            "Recorded fixed neutral state / shared point model; endpoint geometry only, "
            "encounter safety is in symmetric probes"
        ),
        left_rollouts=repertoire["fixed_rollouts"],
        right_rollouts=repertoire["adaptive_rollouts"],
        left_safe=repertoire["fixed_safe"],
        right_safe=repertoire["adaptive_safe"],
    )
    np.savez_compressed(
        directory / "symmetric_probe_trajectories.npz",
        time_seconds=np.asarray(probe_arrays["time_seconds"]),
        **{key: np.asarray(value) for key, value in symmetric_arrays.items()},
    )
    np.savez_compressed(directory / "dense_plant_states.npz", **dense_states)
    changed_wind = np.asarray(scenario.wind_after)
    winds = np.where(
        (all_times >= config.event_time_seconds)[:, None], changed_wind[None], np.zeros((1, 3))
    )
    obstacle_tracks = tuple(
        ObstacleTrack(
            np.asarray(scenario.obstacle_initial_centers)[i][None]
            + all_times[:, None] * np.asarray(scenario.obstacle_velocities)[i][None],
            float(scenario.obstacle_radii[i]),
            float(scenario.obstacle_radii[i] + scenario.ego_radius + scenario.obstacle_clearance),
            f"obstacle {i + 1}",
        )
        for i in range(len(scenario.obstacle_radii))
        if bool(scenario.obstacle_mask[i])
    )
    summary = {
        "experiment": "competent_checkpoint",
        "config": asdict(config),
        "checkpoint": str(checkpoint_stem),
        "checkpoint_npz_sha256": bundle.sha256,
        "initial_checkpoint_hashes": event_hashes,
        "same_event_state_and_parameters": len(
            {json.dumps(v, sort_keys=True) for v in event_hashes.values()}
        )
        == 1,
        "checkpoint_competency": bundle.metadata["competency"],
        "event_state_competency": bundle.metadata["event_state_competency"],
        "methods": summaries,
        "shared_probes": probes,
        "physical_model": "cf21B_500",
        "point_model_information": config.model_mode,
        "post_event_compensation": {
            "fixed": False,
            "compensated": True,
            "adaptive": config.adaptive_model_compensation,
            "scope": (
                "Shared nominal pre-event actor; the compensated branches enable identical "
                "point-model force feedforward at the event. Compare compensated versus "
                "adaptive to isolate subsequent BPTT updates."
            ),
        },
        "hover_authority_after_change": hover_authority(event_model, resources.actuator),
        "feasibility_reference": witness.summary,
        "control_period_seconds": config.control_period,
        "prediction_duration_seconds": config.dt * config.horizon,
        "learner_warm_service_seconds": warm_durations,
        "runtime_scope": (
            "Measured sequential GPU service, paced control boundaries, "
            "next-boundary completed-snapshot publication; sensor/actuator "
            "transport latency excluded. Diagnostic probes run afterward."
        ),
        "emergency_rule": (
            "Same obstacle-agnostic model-aware velocity braking and attitude "
            "stabilization; explicitly uncertified, midpoint only invalid resources."
        ),
        "payload_scope": (
            "Centered rigid box, supplied mass/inertia switch at fixed COM; "
            "no contact-resolved pickup, off-center load or tether."
        )
        if config.disturbance == "payload"
        else None,
    }
    summary["checks"] = {
        "same_event_checkpoint": summary["same_event_state_and_parameters"],
        "nominal_repertoire_competent": bool(
            bundle.metadata["competency"]["competent_under_declared_criteria"]
        ),
        "hover_remains_feasible": bool(summary["hover_authority_after_change"]["hover_feasible"]),
        "no_midpoint_execution": all(
            x["midpoint_execution_count"] == 0 for x in summaries.values()
        ),
        "adaptive_safe_goal": bool(summaries["adaptive"]["safe_goal_success"]),
        "adaptive_zero_degraded": summaries["adaptive"]["degraded_step_count"] == 0,
        "adaptive_updates_finite": summaries["adaptive"]["attempted_updates"] > 0
        and summaries["adaptive"]["attempted_updates"] == summaries["adaptive"]["finite_updates"],
        "adaptive_budget_met": config.schedule == "budgeted"
        and summaries["adaptive"]["deadline_misses"] == 0,
    }
    summary["all_checks_passed"] = all(summary["checks"].values())
    trace = ComparisonVideoTrace(
        all_times,
        np.asarray(scenario.goal_position),
        obstacle_tracks,
        winds,
        np.asarray(methods["adaptive"].estimated_wind),
        config.event_time_seconds,
        np.asarray(bundle.spec.target_descriptors),
        methods["fixed"],
        methods["adaptive"],
        title="A shared repertoire adapts to changed dynamics",
        left_label="FROZEN SKILLS",
        right_label="ADAPTIVE SKILLS",
        drone_radius=scenario.ego_radius,
        drone_model="cf21B_500",
        physical_model_name="cf21B_500",
        show_wind_change_banner=config.disturbance == "wind",
        payload_attachment_time_seconds=(
            config.event_time_seconds if config.disturbance == "payload" else None
        ),
        payload_half_extents=(
            np.asarray(payload.half_extents) if config.disturbance == "payload" else None
        ),
        payload_mass_delta_kg=payload.mass if config.disturbance == "payload" else None,
        payload_base_mass_kg=(
            float(np.asarray(resources.model.mass)) if config.disturbance == "payload" else None
        ),
        repertoire_probes=repertoire,
        coverage_probes=_coverage_probes_from_summary(summary),
    )
    trace.validate()
    result = OnlineConstantWindResult(trace, summary, methods)
    save_online_constant_wind_result(result, directory, stem="competent_comparison")
    return result
