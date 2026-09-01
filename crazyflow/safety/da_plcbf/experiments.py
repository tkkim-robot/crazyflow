"""Deterministic DA-PLCBF trial and paired-campaign execution.

This module is the numeric orchestration layer.  It creates one immutable scenario tape per
condition/fold, gives byte-identical tapes to paired methods, keeps controller and true-plant
dynamics separate, and writes :class:`~crazyflow.safety.da_plcbf.artifacts.ImmutableTrace` data
before any rendering.  The plant is the airborne direct-wrench model: no floor clamp, contact
response, or hidden maneuver state machine is used.

The full method dispatch combines the Cartesian dynamics/obstacle robust discrete filter, online
bounded parameter estimation, deterministic covariance particles, immutable active/candidate
snapshots, fixed-budget quadrotor BPTT, hard non-regression admission, and atomic publication.
Compilation is performed and timed before the recorded warm control loop.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.actor import (
    SharedActorConfig,
    SharedActorParams,
    SharedActorSpec,
    initialize_shared_actor,
)
from crazyflow.safety.da_plcbf.adaptation_evidence import (
    ADMISSION_PUBLICATION_ACCOUNTING,
    ADMISSION_RUNTIME_SCOPE,
    BPTT_EXECUTION_CONTRACT,
    AdaptationDecisionProof,
    AdaptationEvidence,
    CandidateValidationMaterial,
    save_adaptation_evidence,
    validate_adaptation_evidence_binding,
)
from crazyflow.safety.da_plcbf.artifacts import (
    TRACE_SCHEMA_VERSION,
    ArtifactEvent,
    ImmutableTrace,
    load_events,
    load_trace,
    save_trace,
    write_events,
    write_metrics,
    write_timing,
)
from crazyflow.safety.da_plcbf.baselines import MethodID, method_spec
from crazyflow.safety.da_plcbf.config import LibraryLossConfig
from crazyflow.safety.da_plcbf.dashboard_evidence import (
    DASHBOARD_EVIDENCE_SCHEMA_VERSION,
    DashboardEvidence,
    _admission_evidence_from_events,
    save_dashboard_evidence,
    validate_dashboard_evidence_binding,
)
from crazyflow.safety.da_plcbf.descriptors import trajectory_descriptors
from crazyflow.safety.da_plcbf.direct_wrench import (
    motor_forces_to_wrench,
    quaternion_to_rotation_matrix,
    wrench_to_motor_forces,
)
from crazyflow.safety.da_plcbf.dynamic_filter import (
    DynamicFilterConfig,
    dynamic_discrete_runtime_step,
)
from crazyflow.safety.da_plcbf.dynamic_rollouts import (
    DYNAMIC_PREDICTION_CONTRACT,
    DynamicSphereScenarioBatch,
    dynamic_sphere_window_from_tape,
)
from crazyflow.safety.da_plcbf.estimator import (
    EstimatorConfig,
    EstimatorState,
    RotorEfficiencyObservations,
    TranslationalObservations,
    deterministic_parameter_samples,
    initialize_estimator,
    physical_parameters,
    update_rotor_efficiency,
    update_translational_estimate,
)
from crazyflow.safety.da_plcbf.library import (
    build_shared_quad_library_spec,
    descriptor_targets_from_spec,
    slice_shared_actor_policy,
)
from crazyflow.safety.da_plcbf.quad_actor_bptt import build_dynamic_model_quad_actor_bptt_functions
from crazyflow.safety.da_plcbf.quad_actor_losses import (
    QuadLearningConfig,
    rigid_body_safety_batch_from_circles,
)
from crazyflow.safety.da_plcbf.quad_generic_diversity_bptt import (
    build_quad_generic_diversity_bptt_functions,
)
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig, waypoint_nominal_wrench
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.quad_uncertainty import (
    VersionAModelSamples,
    version_a_model_samples_from_estimator,
)
from crazyflow.safety.da_plcbf.runtime import AdaptationStatus, AdaptationWorker
from crazyflow.safety.da_plcbf.scenarios import (
    ScenarioTape,
    ScenarioTapeConfig,
    generate_scenario_tape,
    hard_contact_labels,
    load_scenario_tape,
    save_scenario_tape,
)
from crazyflow.safety.da_plcbf.scientific_evaluation import (
    MINIMUM_FINAL_PAIRED_TRIALS,
    AnalysisRole,
    MetricDirection,
    PairedComparison,
    PairedInferenceConfig,
    PairedTrialDataset,
    PairedTrialSchedule,
    ScientificTrialMetrics,
    ScientificTrialRecord,
    TrialAssignment,
    TrialStatus,
    compare_paired_metric,
    confirmatory_bootstrap_replicates,
    derive_scientific_metrics,
    make_paired_trial_schedule,
)
from crazyflow.safety.da_plcbf.snapshots import (
    ActiveSnapshotStore,
    PolicySnapshot,
    create_active_snapshot,
    create_candidate_snapshot,
)
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.uncertain_dynamic_filter import (
    CartesianUncertaintyConfig,
    evaluate_uncertain_dynamic_quad_library,
    uncertain_dynamic_discrete_runtime_step,
)
from crazyflow.safety.da_plcbf.validation import (
    HardValidationEvidence,
    HardValidationThresholds,
    ValidationReport,
    hard_validate_candidate,
)
from crazyflow.safety.da_plcbf.version_a_analytic_runtime import version_a_analytic_runtime_step
from crazyflow.safety.da_plcbf.version_a_barriers import (
    RigidBodySafetySet,
    VersionABarrierConfig,
    VersionAModel,
)
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator, VersionAFilterConfig
from crazyflow.safety.da_plcbf.version_a_runtime import version_a_runtime_step

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class ConditionID(StrEnum):
    """Required finite-horizon evaluation conditions."""

    STATIC = "static"
    DYNAMICS_CHANGE = "dynamics_change"
    BALLISTIC_BALL = "ballistic_ball"
    INTERCEPTOR_DRONE = "interceptor_drone"
    FALSIFICATION_COMBINED = "falsification_combined"


REQUIRED_CONDITIONS = (
    ConditionID.STATIC.value,
    ConditionID.DYNAMICS_CHANGE.value,
    ConditionID.BALLISTIC_BALL.value,
    ConditionID.INTERCEPTOR_DRONE.value,
)


class AdaptationExecutionMode(StrEnum):
    """How post-startup candidate work is scheduled relative to simulated time."""

    LOGICAL_SIMULATION = "logical_simulation"
    REALTIME_PROBE = "realtime_probe"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """One fixed-shape trial configuration.

    ``final_defaults`` instantiates the plan's K=64/H=50/R=4 shape.  Smaller values are useful for
    deterministic unit and GPU smoke tests but are never silently relabeled as final evidence.
    """

    control_steps: int = 151
    dt: float = 0.02
    certificate_horizon: int = 50
    policy_count: int = 64
    prediction_samples: int = 4
    training_scenario_count: int = 64
    validation_scenarios_per_fold: int = 2
    bptt_burst_steps: int = 10
    adaptation_interval_steps: int = 25
    estimator_interval_steps: int = 10
    estimator_window_steps: int = 12
    policy_gain: float = 1.8
    speed_limit: float = 3.0
    angular_rate_max: float = 8.0
    tilt_max_radians: float = 1.1
    static_capacity: int = 4
    static_count: int = 3
    dynamic_capacity: int = 4
    obstacle_clearance: float = 0.03
    random_seed: int = 0
    uncertainty_sample_count: int = 4
    controller_deadline_seconds: float = 0.02
    estimator_deadline_seconds: float = 0.02
    logging_deadline_seconds: float = 0.01
    validation_runtime_budget_seconds: float = 120.0
    validation_minimum_coverage: float = 1.0
    validation_minimum_redundancy: int = 2
    validation_minimum_diversity: float = 1e-3
    validation_retention_tolerance: float = 0.05
    adaptation_execution_mode: str = AdaptationExecutionMode.LOGICAL_SIMULATION.value
    realtime_pacing: bool = False

    def validate(self) -> None:
        """Validate fixed shapes, physical ranges, and scheduling intervals."""
        integer_values = {
            "control_steps": self.control_steps,
            "certificate_horizon": self.certificate_horizon,
            "policy_count": self.policy_count,
            "prediction_samples": self.prediction_samples,
            "training_scenario_count": self.training_scenario_count,
            "validation_scenarios_per_fold": self.validation_scenarios_per_fold,
            "bptt_burst_steps": self.bptt_burst_steps,
            "adaptation_interval_steps": self.adaptation_interval_steps,
            "estimator_interval_steps": self.estimator_interval_steps,
            "estimator_window_steps": self.estimator_window_steps,
            "static_capacity": self.static_capacity,
            "static_count": self.static_count,
            "dynamic_capacity": self.dynamic_capacity,
            "validation_minimum_redundancy": self.validation_minimum_redundancy,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.control_steps < 2 or self.policy_count < 8:
            raise ValueError("control_steps must be >=2 and policy_count must be >=8")
        if self.static_count > self.static_capacity:
            raise ValueError("static_count must not exceed static_capacity")
        if self.uncertainty_sample_count not in (4, 8):
            raise ValueError("uncertainty_sample_count must be exactly 4 or 8")
        numeric = (
            self.dt,
            self.policy_gain,
            self.speed_limit,
            self.angular_rate_max,
            self.tilt_max_radians,
            self.controller_deadline_seconds,
            self.estimator_deadline_seconds,
            self.logging_deadline_seconds,
            self.validation_runtime_budget_seconds,
        )
        if not all(math.isfinite(value) and value > 0 for value in numeric):
            raise ValueError("trial rates, limits, and deadlines must be finite and positive")
        if not 0 < self.tilt_max_radians <= 0.5 * math.pi:
            raise ValueError("tilt_max_radians must lie in (0,pi/2]")
        if not math.isfinite(self.obstacle_clearance) or self.obstacle_clearance < 0:
            raise ValueError("obstacle_clearance must be finite and nonnegative")
        if not 0.0 < self.validation_minimum_coverage <= 1.0:
            raise ValueError("validation_minimum_coverage must lie in (0,1]")
        if self.validation_minimum_redundancy > self.policy_count:
            raise ValueError("validation_minimum_redundancy must not exceed policy_count")
        if (
            not math.isfinite(self.validation_minimum_diversity)
            or self.validation_minimum_diversity <= 0
        ):
            raise ValueError("validation_minimum_diversity must be finite and positive")
        if (
            not math.isfinite(self.validation_retention_tolerance)
            or self.validation_retention_tolerance < 0
        ):
            raise ValueError("validation_retention_tolerance must be finite and nonnegative")
        if not 0 <= self.random_seed <= np.iinfo(np.uint32).max:
            raise ValueError("random_seed must fit uint32")
        if not isinstance(self.realtime_pacing, bool):
            raise TypeError("realtime_pacing must be boolean")
        try:
            AdaptationExecutionMode(self.adaptation_execution_mode)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "adaptation_execution_mode must be logical_simulation or realtime_probe"
            ) from error

    @classmethod
    def final_defaults(cls, *, random_seed: int = 0) -> ExperimentConfig:
        """Return the predeclared K=64/H=50/R=4 final-shape configuration."""
        return cls(
            random_seed=random_seed,
            adaptation_execution_mode=AdaptationExecutionMode.REALTIME_PROBE.value,
            realtime_pacing=True,
        )


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    """Complete paired schedule plus one fixed-shape trial configuration."""

    trial: ExperimentConfig
    methods: tuple[str, ...]
    conditions: tuple[str, ...] = REQUIRED_CONDITIONS
    trials_per_condition: int = 1
    root_seed: int = 0
    fold_start: int = 0
    intended_for_final_claim: bool = False

    def schedule(self) -> PairedTrialSchedule:
        """Build the deterministic complete-factorial paired assignment schedule."""
        self.trial.validate()
        return make_paired_trial_schedule(
            root_seed=self.root_seed,
            methods=self.methods,
            conditions=self.conditions,
            trials_per_condition=self.trials_per_condition,
            fold_start=self.fold_start,
            intended_for_final_claim=self.intended_for_final_claim,
        )

    def final_contract_blockers(self) -> tuple[str, ...]:
        """Return every departure from the exact predeclared final scientific matrix."""
        blockers: list[str] = []
        core_methods = tuple(item.value for item in MethodID)
        if self.methods != core_methods:
            blockers.append("methods are not exactly the seven ordered core method IDs")
        if self.conditions != REQUIRED_CONDITIONS:
            blockers.append("conditions are not exactly the four required condition IDs")
        if self.trials_per_condition < MINIMUM_FINAL_PAIRED_TRIALS:
            blockers.append("fewer than 100 paired trials per condition")
        if not self.intended_for_final_claim:
            blockers.append("schedule was not predeclared for a final claim")
        reference = ExperimentConfig.final_defaults(random_seed=self.trial.random_seed)
        differing = tuple(
            name
            for name in self.trial.__dataclass_fields__
            if name != "random_seed" and getattr(self.trial, name) != getattr(reference, name)
        )
        if differing:
            blockers.append(
                f"trial configuration differs from final defaults: {','.join(differing)}"
            )
        if self.trial.prediction_samples != 4 or self.trial.uncertainty_sample_count != 4:
            blockers.append("final Cartesian uncertainty shape must be R_o=4 and R_m=4")
        return tuple(blockers)

    @classmethod
    def final_core(cls, *, root_seed: int = 0) -> CampaignConfig:
        """Return the required seven-method, four-condition, 100-pair final schedule."""
        return cls(
            trial=ExperimentConfig.final_defaults(random_seed=root_seed),
            methods=tuple(item.value for item in MethodID),
            trials_per_condition=MINIMUM_FINAL_PAIRED_TRIALS,
            root_seed=root_seed,
            intended_for_final_claim=True,
        )


@dataclass(frozen=True, slots=True)
class ExperimentResources:
    """Common paired controller initialization and physical parameters."""

    model: VersionAModel
    actuator: VersionAActuator
    spec: SharedActorSpec
    initial_params: SharedActorParams
    actor_config: SharedActorConfig
    quad_config: QuadPolicyConfig
    barrier_config: VersionABarrierConfig
    version_a_filter_config: VersionAFilterConfig
    dynamic_filter_config: DynamicFilterConfig
    estimator_config: EstimatorConfig
    uncertainty_config: CartesianUncertaintyConfig
    loss_config: LibraryLossConfig


@dataclass(frozen=True, slots=True)
class TrialRun:
    """One completed real trace and its replay/scientific auxiliary evidence."""

    assignment: TrialAssignment
    tape: ScenarioTape
    trace: ImmutableTrace
    events: tuple[ArtifactEvent, ...]
    compile_seconds: Mapping[str, float]
    compile_cache_hits: Mapping[str, bool]
    deadlines_seconds: Mapping[str, float]
    hard_certified_policy: np.ndarray
    estimation_error: np.ndarray
    estimation_scale: np.ndarray
    scientific_metrics: ScientificTrialMetrics
    dashboard_evidence: DashboardEvidence
    adaptation_evidence: AdaptationEvidence | None
    method_claim_eligible: bool
    claim_blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CampaignRun:
    """Every retained paired outcome; execution failures are never dropped."""

    schedule: PairedTrialSchedule
    trial_runs: tuple[TrialRun, ...]
    records: tuple[ScientificTrialRecord, ...]
    paired_comparisons: tuple[PairedComparison, ...]
    inference_config: PairedInferenceConfig | None
    exploratory_inference_config: PairedInferenceConfig | None
    execution_complete: bool
    scientific_claim_eligible: bool
    global_confirmatory_superiority_supported: bool
    claim_blockers: tuple[str, ...]


class TrialExecutionError(RuntimeError):
    """A deterministic trial failed before a valid immutable trace could be produced."""


class _BPTTExecutablePool:
    """Campaign-local BPTT function and device executable cache for one exact static shape."""

    def __init__(self, signature: str, bptt: Any, device: jax.Device) -> None:
        self.signature = signature
        self.bptt = bptt
        self.device = device
        self.device_key = (str(device.platform), int(device.id))
        self.compiled_bursts: dict[str, Any] = {}
        self.compile_timings: dict[str, tuple[float, float]] = {}
        self.compiled_evidence: dict[str, Any] = {}
        self.evidence_compile_timings: dict[str, tuple[float, float]] = {}
        self.lock = threading.RLock()


class _CampaignExecutableCache:
    """Campaign-local compiled executables; numerical per-trial values remain dynamic inputs."""

    def __init__(self) -> None:
        self.controllers: dict[str, Any] = {}
        self.plants: dict[str, Any] = {}
        self.estimators: dict[str, Any] = {}
        self.bptt_pools: dict[str, _BPTTExecutablePool] = {}


class _ControlDiagnostics(NamedTuple):
    motor_command: Array
    nominal_motor: Array
    policy_values: Array
    training_values: Array
    selected_policy: Array
    degraded: Array
    kkt_residual: Array
    postcheck_residual: Array
    clipped: Array
    saturated: Array
    nominal_rollout_positions: Array
    fallback_rollout_positions: Array
    fallback_rollout_available: Array
    selected_rollout_positions: Array
    selected_rollout_available: Array
    normalized_descriptors: Array
    descriptor_available: Array
    ghost_rollout_positions: Array
    ghost_rollout_available: Array


def _condition(value: str | ConditionID) -> ConditionID:
    try:
        return value if isinstance(value, ConditionID) else ConditionID(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown DA-PLCBF condition {value!r}") from error


def scenario_config_for_condition(
    condition: str | ConditionID, config: ExperimentConfig
) -> ScenarioTapeConfig:
    """Return an isolated, fixed-capacity tape configuration for one named condition."""
    config.validate()
    selected = _condition(condition)
    counts = {
        "ballistic_count": 2 if selected is ConditionID.BALLISTIC_BALL else 0,
        "crossing_count": 1 if selected is ConditionID.FALSIFICATION_COMBINED else 0,
        "pursuit_count": 0,
        "interceptor_count": 2 if selected is ConditionID.INTERCEPTOR_DRONE else 0,
        "random_attacker_count": 0,
    }
    tape_steps = config.control_steps + config.certificate_horizon + 1
    # ScenarioTape expresses schedule changes as fractions of its complete lookahead-padded time
    # axis.  Re-map the predeclared 20/35/50/65/80% challenge nodes to the *executed* control
    # horizon so all changes occur during evaluation and retain recovery time afterward.
    schedule_targets = np.floor(
        np.asarray((0.20, 0.35, 0.50, 0.65, 0.80)) * (config.control_steps - 1) + 0.5
    ).astype(np.int64)
    schedule_fractions = schedule_targets / (tape_steps - 1)
    return ScenarioTapeConfig(
        steps=tape_steps,
        dt=config.dt,
        prediction_samples=config.prediction_samples,
        static_capacity=config.static_capacity,
        static_count=config.static_count,
        dynamic_capacity=config.dynamic_capacity,
        wind_change_fraction=float(schedule_fractions[0]),
        mass_change_fraction=float(schedule_fractions[1]),
        drag_change_fraction=float(schedule_fractions[2]),
        rotor_symmetric_change_fraction=float(schedule_fractions[3]),
        rotor_single_change_fraction=float(schedule_fractions[4]),
        **counts,
    )


def generate_condition_tape(
    condition: str | ConditionID, config: ExperimentConfig, *, seed: int, fold: int
) -> ScenarioTape:
    """Generate one immutable condition/fold tape shared by all paired methods."""
    return generate_scenario_tape(seed, scenario_config_for_condition(condition, config), fold=fold)


def build_experiment_resources(
    config: ExperimentConfig, *, obstacle_count: int, initialization_seed: int
) -> ExperimentResources:
    """Build common physical/controller resources without method-specific randomness."""
    config.validate()
    raw: dict[str, Any] = load_params("cf21B_500")
    model = VersionAModel(
        mass=jnp.asarray(raw["mass"]),
        gravity_vec=jnp.asarray(raw["gravity_vec"]),
        inertia=jnp.asarray(raw["J"]),
        inertia_inv=jnp.linalg.inv(jnp.asarray(raw["J"])),
        drag_matrix=jnp.asarray(raw["drag_matrix"]),
        wind_velocity=jnp.zeros(3),
        external_force=jnp.zeros(3),
        external_torque=jnp.zeros(3),
    )
    actuator = VersionAActuator(
        arm_length=jnp.asarray(raw["L"]),
        thrust_to_torque=jnp.asarray(raw["thrust2torque"]),
        mixing_matrix=jnp.asarray(raw["mixing_matrix"]),
        thrust_min=jnp.asarray(raw["thrust_min"]),
        thrust_max=jnp.asarray(raw["thrust_max"]),
    )
    actor_config = SharedActorConfig(hidden_width=32)
    # Policy initialization is claim-bearing evidence.  Canonicalize its random transform on CPU
    # so a GPU-produced run reconstructs byte-exactly in a fresh CPU-only validator.
    actor_initialization_device = jax.devices("cpu")[0]
    with jax.default_device(actor_initialization_device):
        spec = build_shared_quad_library_spec(policy_count=config.policy_count)
        params = initialize_shared_actor(
            jax.random.key(initialization_seed),
            spec,
            dimension=3,
            n_obstacles=obstacle_count,
            config=actor_config,
        )
    return ExperimentResources(
        model=model,
        actuator=actuator,
        spec=spec,
        initial_params=params,
        actor_config=actor_config,
        quad_config=QuadPolicyConfig(),
        barrier_config=VersionABarrierConfig(obstacle_clearance=config.obstacle_clearance),
        version_a_filter_config=VersionAFilterConfig(),
        dynamic_filter_config=DynamicFilterConfig(),
        estimator_config=EstimatorConfig(),
        uncertainty_config=CartesianUncertaintyConfig(),
        loss_config=LibraryLossConfig(),
    )


def _resources_for_tape(
    resources: ExperimentResources, tape: ScenarioTape, config: ExperimentConfig
) -> ExperimentResources:
    """Bind controller geometry to the same spherical footprint as authoritative contact labels."""
    footprint = float(tape.vehicle_radius)
    barrier = replace(
        resources.barrier_config,
        obstacle_clearance=footprint + config.obstacle_clearance,
        arena_clearance=footprint,
    )
    barrier.validate()
    return replace(resources, barrier_config=barrier)


def _initial_state(tape: ScenarioTape) -> Array:
    return jnp.asarray(
        np.concatenate(
            (
                tape.vehicle_initial_position,
                np.asarray([0.0, 0.0, 0.0, 1.0]),
                tape.vehicle_initial_velocity,
                np.zeros(3),
            )
        ),
        dtype=jnp.float32,
    )


def _circle_scenario_at(tape: ScenarioTape, index: int, speed_limit: float) -> CircleScenarioBatch:
    centers = np.concatenate((tape.static_positions, tape.dynamic_positions[index]), axis=0)
    radii = np.concatenate((tape.static_radii, tape.dynamic_radii), axis=0)
    mask = np.concatenate((tape.static_mask, tape.dynamic_time_mask[index]), axis=0)
    return CircleScenarioBatch(
        obstacle_centers=jnp.asarray(centers[None], dtype=jnp.float32),
        obstacle_radii=jnp.asarray(radii[None], dtype=jnp.float32),
        obstacle_mask=jnp.asarray(mask[None]),
        arena_lower=jnp.asarray(tape.arena_lower[None], dtype=jnp.float32),
        arena_upper=jnp.asarray(tape.arena_upper[None], dtype=jnp.float32),
        speed_limit=jnp.asarray([speed_limit], dtype=jnp.float32),
    )


def _safety_from_circles(
    circles: CircleScenarioBatch, config: ExperimentConfig
) -> RigidBodySafetySet:
    return rigid_body_safety_batch_from_circles(
        circles, angular_rate_max=config.angular_rate_max, tilt_max_radians=config.tilt_max_radians
    )


def _true_model(
    base: VersionAModel, tape: ScenarioTape, condition: ConditionID, index: int
) -> tuple[VersionAModel, Array]:
    scheduled = condition in {ConditionID.DYNAMICS_CHANGE, ConditionID.FALSIFICATION_COMBINED}
    mass_scale = float(tape.mass_scale[index]) if scheduled else 1.0
    drag_scale = jnp.asarray(tape.drag_scale[index] if scheduled else np.ones(3))
    wind = jnp.asarray(tape.wind_velocity[index] if scheduled else np.zeros(3))
    efficiency = jnp.asarray(tape.rotor_efficiency[index] if scheduled else np.ones(4))
    model = base._replace(
        mass=base.mass * mass_scale,
        drag_matrix=base.drag_matrix * drag_scale[None, :],
        wind_velocity=wind,
    )
    return model, efficiency


def _controller_model(base: VersionAModel, estimator: EstimatorState) -> VersionAModel:
    estimated = physical_parameters(estimator)
    return base._replace(
        mass=estimated.mass,
        drag_matrix=estimated.drag_matrix,
        wind_velocity=estimated.wind_velocity,
    )


def _model_samples(
    estimator: EstimatorState, resources: ExperimentResources, count: int
) -> VersionAModelSamples:
    particles = deterministic_parameter_samples(
        estimator, sample_count=count, config=resources.estimator_config
    )
    return version_a_model_samples_from_estimator(
        particles, _controller_model(resources.model, estimator), resources.estimator_config
    )


def _initialize_estimator(resources: ExperimentResources) -> EstimatorState:
    drag_coefficients = -jnp.diag(resources.model.drag_matrix)
    return initialize_estimator(
        resources.estimator_config,
        mass=float(resources.model.mass),
        drag_force_coefficients=drag_coefficients,
        wind_velocity=resources.model.wind_velocity,
        rotor_efficiency=1.0,
    )


_DYNAMICS_PARAMETER_NAMES = (
    "mass_kg",
    "drag_acceleration_x",
    "drag_acceleration_y",
    "drag_acceleration_z",
    "wind_x",
    "wind_y",
    "wind_z",
    "rotor_efficiency_0",
    "rotor_efficiency_1",
    "rotor_efficiency_2",
    "rotor_efficiency_3",
)


def _dynamics_parameter_vector(model: VersionAModel, rotor_efficiency: Array) -> Array:
    """Return the exact physical parameter vector used by dashboard evidence."""
    drag_acceleration = -jnp.diag(model.drag_matrix) / model.mass
    return jnp.concatenate(
        (
            jnp.reshape(model.mass, (1,)),
            drag_acceleration,
            model.wind_velocity,
            jnp.asarray(rotor_efficiency),
        )
    )


def _sampled_dynamics_parameter_vectors(samples: VersionAModelSamples) -> Array:
    """Return one physical dashboard vector for every explicit dynamics particle."""
    models = samples.models
    drag_acceleration = -jnp.diagonal(models.drag_matrix, axis1=-2, axis2=-1) / models.mass[:, None]
    return jnp.concatenate(
        (models.mass[:, None], drag_acceleration, models.wind_velocity, samples.rotor_efficiency),
        axis=1,
    )


def _replay_dashboard_dynamics_and_contexts(
    trace: ImmutableTrace,
    tape: ScenarioTape,
    condition: ConditionID | str,
    method: MethodID | str,
    config: ExperimentConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[VersionAModel, ...],
    tuple[VersionAModelSamples, ...],
]:
    """Independently reconstruct truth, estimator state, and uncertainty sidecar arrays.

    The replay consumes only the immutable trace, predeclared tape, method/configuration, and the
    packaged vehicle parameters.  In particular it does not trust any dynamics array stored in the
    dashboard sidecar.
    """
    config.validate()
    trace.validate()
    tape.validate()
    selected_condition = ConditionID(condition)
    selected_method = MethodID(method)
    if trace.steps > tape.steps:
        raise ValueError("dynamics replay trace exceeds its scenario tape")
    device = _authoritative_estimator_device()
    raw: dict[str, Any] = load_params("cf21B_500")
    with jax.default_device(device):
        base_model = VersionAModel(
            mass=jnp.asarray(raw["mass"]),
            gravity_vec=jnp.asarray(raw["gravity_vec"]),
            inertia=jnp.asarray(raw["J"]),
            inertia_inv=jnp.linalg.inv(jnp.asarray(raw["J"])),
            drag_matrix=jnp.asarray(raw["drag_matrix"]),
            wind_velocity=jnp.zeros(3),
            external_force=jnp.zeros(3),
            external_torque=jnp.zeros(3),
        )
        actuator = VersionAActuator(
            arm_length=jnp.asarray(raw["L"]),
            thrust_to_torque=jnp.asarray(raw["thrust2torque"]),
            mixing_matrix=jnp.asarray(raw["mixing_matrix"]),
            thrust_min=jnp.asarray(raw["thrust_min"]),
            thrust_max=jnp.asarray(raw["thrust_max"]),
        )
    estimator_config = EstimatorConfig()
    estimator = _initialize_authoritative_estimator(base_model, estimator_config, device=device)
    uses_uncertainty = selected_method in {
        MethodID.DA_PLCBF_NO_ONLINE_MODEL_ADAPTATION,
        MethodID.DA_PLCBF_FULL,
    }
    updates_estimator = selected_method is MethodID.DA_PLCBF_FULL
    steps = trace.steps
    truth = np.zeros((steps, len(_DYNAMICS_PARAMETER_NAMES)), dtype=np.float64)
    estimated = np.zeros_like(truth)
    uncertainty = np.zeros(
        (steps, config.uncertainty_sample_count, len(_DYNAMICS_PARAMETER_NAMES)), dtype=np.float64
    )
    uncertainty_available = np.zeros((steps, config.uncertainty_sample_count), dtype=np.bool_)
    controller_models: list[VersionAModel] = []
    model_sample_contexts: list[VersionAModelSamples] = []
    history: list[tuple[np.ndarray, ...]] = []
    compiled_estimator = None
    if updates_estimator:
        empty = _authoritative_estimator_observations(
            history, config.estimator_window_steps, device=device
        )
        estimator_arguments = _authoritative_estimator_arguments(estimator, empty, 0, device=device)
        compiled_estimator = (
            _authoritative_estimator_function(estimator_config)
            .lower(*estimator_arguments)
            .compile()
        )
        _block(compiled_estimator(*estimator_arguments))
    for index in range(steps):
        with jax.default_device(device):
            true_model, efficiency = _true_model(base_model, tape, selected_condition, index)
        truth[index] = np.asarray(
            _dynamics_parameter_vector(true_model, efficiency), dtype=np.float64
        )
        controller_model, samples = _authoritative_model_samples(
            base_model, estimator, estimator_config, config.uncertainty_sample_count, device=device
        )
        controller_models.append(controller_model)
        estimated[index] = np.asarray(
            _dynamics_parameter_vector(
                controller_model, physical_parameters(estimator).rotor_efficiency
            ),
            dtype=np.float64,
        )
        model_sample_contexts.append(samples)
        if uses_uncertainty:
            uncertainty[index] = np.asarray(
                _sampled_dynamics_parameter_vectors(samples), dtype=np.float64
            )
            uncertainty_available[index] = np.asarray(samples.sample_valid, dtype=np.bool_)
        if int(np.asarray(estimator.model_version)) != int(trace.model_version[index]):
            raise ValueError("dashboard estimator model-version history does not replay")
        if index == steps - 1:
            continue
        with jax.default_device(device):
            state = jnp.asarray(trace.true_state[index], dtype=jnp.float32)
            next_state = jnp.asarray(trace.true_state[index + 1], dtype=jnp.float32)
            commanded = jnp.asarray(trace.filtered_control[index], dtype=jnp.float32)
            realized = jnp.asarray(trace.applied_control[index], dtype=jnp.float32)
        history.append(
            _estimator_history_entry(
                state, next_state, commanded, realized, true_model, actuator, tape, index, config.dt
            )
        )
        if updates_estimator and (index + 1) % config.estimator_interval_steps == 0:
            if compiled_estimator is None:
                raise RuntimeError("authoritative estimator executable was not prepared")
            observations = _authoritative_estimator_observations(
                history, config.estimator_window_steps, device=device
            )
            arguments = _authoritative_estimator_arguments(
                estimator, observations, index, device=device
            )
            translation_update, rotor_update = compiled_estimator(*arguments)
            _block((translation_update, rotor_update))
            estimator = translation_update.state
    return (
        truth,
        estimated,
        uncertainty,
        uncertainty_available,
        tuple(controller_models),
        tuple(model_sample_contexts),
    )


def replay_dashboard_dynamics_evidence(
    trace: ImmutableTrace,
    tape: ScenarioTape,
    condition: ConditionID | str,
    method: MethodID | str,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independently reconstruct dynamics arrays without trusting the dashboard sidecar."""
    replay = _replay_dashboard_dynamics_and_contexts(trace, tape, condition, method, config)
    return replay[:4]


def _tree_device(params: Any) -> Any:
    return jax.tree.map(jnp.asarray, params)


def _tree_to_device(tree: Any, device: jax.Device) -> Any:
    """Place a pytree on the controller device without rebuilding unchanged leaves."""
    return jax.device_put(tree, device)


def _block(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def _finite_policy_evidence(
    policy_values: Any, training_values: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw policy evidence or fail the trial without manufacturing finite margins.

    Nonfinite certificate outputs are scientific evidence of an invalid controller evaluation,
    not large positive/negative values that may be clipped into an apparently safe trace.
    """
    policy = np.asarray(policy_values, dtype=np.float64)
    training = np.asarray(training_values, dtype=np.float64)
    if not np.all(np.isfinite(policy)) or not np.all(np.isfinite(training)):
        raise TrialExecutionError("controller returned non-finite policy or training evidence")
    return policy, training


def _polytope_kkt(qp: Any) -> Array:
    values = jnp.asarray(
        [
            qp.primal_residual,
            qp.dual_residual,
            qp.stationarity_residual,
            qp.complementarity_residual,
        ]
    )
    return jnp.where(jnp.all(jnp.isfinite(values)), jnp.max(values), 1e6)


def _discrete_kkt(result: Any, lower: Array, upper: Array) -> Array:
    filtered = result.filter
    weight = 1.0 / (upper - lower) ** 2
    reconstructed = jnp.clip(
        filtered.linearization_action
        + filtered.qp_multiplier * filtered.residual_gradient / weight,
        filtered.trust_lower,
        filtered.trust_upper,
    )
    # ``filtered.action`` is the checked fallback whenever the nonlinear proposal is rejected.
    # KKT evidence belongs to the QP proposal itself, irrespective of that postcheck decision.
    stationarity = jnp.max(jnp.abs(reconstructed - filtered.qp_action))
    primal = jnp.maximum(-filtered.qp_constraint_residual, 0.0)
    complementarity = jnp.abs(
        filtered.qp_multiplier * jnp.maximum(filtered.qp_constraint_residual, 0.0)
    )
    values = jnp.asarray((stationarity, primal, complementarity))
    return jnp.where(jnp.all(jnp.isfinite(values)), jnp.max(values), 1e6)


def _held_nominal_preview(
    state: Array,
    motor_forces: Array,
    model: VersionAModel,
    actuator: VersionAActuator,
    *,
    dt: float,
    horizon: int,
) -> Array:
    """Roll out the recorded nominal motor command held over the certificate horizon."""
    wrench = motor_forces_to_wrench(
        motor_forces,
        L=actuator.arm_length,
        thrust2torque=actuator.thrust_to_torque,
        mixing_matrix=actuator.mixing_matrix,
    )

    def advance(candidate_state: Array, _: None) -> tuple[Array, Array]:
        next_state = direct_wrench_symplectic_step(candidate_state, wrench, model, dt)
        return next_state, next_state[:3]

    _, future_positions = jax.lax.scan(advance, state, None, length=horizon)
    return jnp.concatenate((state[None, :3], future_positions), axis=0)


def _normalized_rollout_descriptors(states: Array) -> Array:
    """Return the same dimensionless 9-D descriptors used by proposal training."""
    translation = jnp.concatenate((states[..., :3], states[..., 7:10]), axis=-1)
    raw = trajectory_descriptors(translation[:, None])[:, 0]
    scales = jnp.asarray((2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0), dtype=states.dtype)
    return raw / scales


def _empty_library_visual_evidence(
    state: Array,
    nominal_motor: Array,
    model: VersionAModel,
    actuator: VersionAActuator,
    *,
    dt: float,
    horizon: int,
    policy_count: int = 1,
) -> tuple[Array, ...]:
    nodes = horizon + 1
    nominal = _held_nominal_preview(state, nominal_motor, model, actuator, dt=dt, horizon=horizon)
    return (
        nominal,
        jnp.zeros((policy_count, nodes, 3), dtype=state.dtype),
        jnp.zeros((policy_count,), dtype=bool),
        jnp.zeros((nodes, 3), dtype=state.dtype),
        jnp.asarray(False),
        jnp.zeros((policy_count, 9), dtype=state.dtype),
        jnp.zeros((policy_count,), dtype=bool),
        jnp.zeros((2, nodes, 3), dtype=state.dtype),
        jnp.zeros((2,), dtype=bool),
    )


def _version_a_diagnostics(
    result: Any,
    actuator: VersionAActuator,
    state: Array,
    model: VersionAModel,
    *,
    dt: float,
    horizon: int,
) -> _ControlDiagnostics:
    motor = wrench_to_motor_forces(
        result.action,
        L=actuator.arm_length,
        thrust2torque=actuator.thrust_to_torque,
        mixing_matrix=actuator.mixing_matrix,
    )
    discrete_postcheck = jnp.where(
        result.applied_discrete_applicable,
        result.applied_discrete_residual,
        jnp.asarray(-1.0, dtype=result.action.dtype),
    )
    post = jnp.min(
        jnp.asarray(
            (
                result.applied_interval_margin,
                discrete_postcheck,
                result.applied_continuous_postcheck.policy_barrier_residual,
                result.applied_continuous_postcheck.minimum_analytic_barrier_residual,
            )
        )
    )
    nominal = result.nominal
    rollout_states = result.certificates.rollout_states
    fallback_positions = rollout_states[..., :3]
    fallback_available = jnp.all(jnp.isfinite(rollout_states), axis=(-2, -1))
    selected_index = result.continuous_filter.selected_index
    selected_positions = fallback_positions[selected_index]
    selected_available = fallback_available[selected_index]
    descriptors = _normalized_rollout_descriptors(rollout_states)
    visual = (
        _held_nominal_preview(
            state, nominal.bounded_motor_forces, model, actuator, dt=dt, horizon=horizon
        ),
        fallback_positions,
        fallback_available,
        selected_positions,
        selected_available,
        descriptors,
        fallback_available & jnp.all(jnp.isfinite(descriptors), axis=-1),
        jnp.zeros((2, horizon + 1, 3), dtype=state.dtype),
        jnp.zeros((2,), dtype=bool),
    )
    return _ControlDiagnostics(
        motor,
        nominal.bounded_motor_forces,
        result.certificates.certificates.values,
        result.certificates.certificates.values,
        result.continuous_filter.selected_index,
        result.degraded,
        _polytope_kkt(result.continuous_filter.qp),
        post,
        jnp.any(jnp.abs(nominal.raw_motor_forces - nominal.bounded_motor_forces) > 1e-7),
        jnp.any(
            (motor <= jnp.asarray(actuator.thrust_min) + 1e-6)
            | (motor >= jnp.asarray(actuator.thrust_max) - 1e-6)
        ),
        *visual,
    )


def _dynamic_diagnostics(
    result: Any,
    actuator: VersionAActuator,
    state: Array,
    model: VersionAModel,
    *,
    dt: float,
    horizon: int,
) -> _ControlDiagnostics:
    filtered = result.filter
    post = jnp.where(
        filtered.proposal_accepted,
        jnp.minimum(filtered.proposal_exact_residual, filtered.proposal_interval_margin),
        jnp.minimum(filtered.fallback_exact_residual, filtered.fallback_interval_margin),
    )
    lower = jnp.broadcast_to(jnp.asarray(actuator.thrust_min), (4,))
    upper = jnp.broadcast_to(jnp.asarray(actuator.thrust_max), (4,))
    rollout_states = result.library.rollouts.states
    selected_index = result.selection.selected_index
    if rollout_states.ndim == 5:
        # [K,B,R_o,T,13]: show sample zero for the library and the worst recorded
        # prediction sample for the selected policy.
        representative = rollout_states[:, 0, 0]
        selected_samples = rollout_states[selected_index, 0]
    elif rollout_states.ndim == 6:
        # [K,B,R_o,R_m,T,13]: flatten the explicit Cartesian sample axes only for
        # choosing and displaying a recorded member; the hard value itself remains robust.
        policy_count, batch, obstacle_samples, model_samples, nodes, dimensions = (
            rollout_states.shape
        )
        combined = rollout_states.reshape(
            policy_count, batch, obstacle_samples * model_samples, nodes, dimensions
        )
        representative = combined[:, 0, 0]
        selected_samples = combined[selected_index, 0]
    else:  # pragma: no cover - upstream result types make this structurally unreachable.
        raise ValueError("dynamic rollout evidence has an unsupported rank")
    selected_margins = result.library.safety_values.prediction_hard_margins[selected_index, 0]
    worst_sample = jnp.argmin(selected_margins)
    selected_states = selected_samples[worst_sample]
    fallback_positions = representative[..., :3]
    fallback_available = jnp.all(jnp.isfinite(representative), axis=(-2, -1))
    descriptors = _normalized_rollout_descriptors(representative)
    ghost_states = jnp.stack((selected_samples[0], selected_samples[-1]))
    ghost_available = jnp.all(jnp.isfinite(ghost_states), axis=(-2, -1))
    visual = (
        _held_nominal_preview(
            state, result.nominal.bounded_motor_forces, model, actuator, dt=dt, horizon=horizon
        ),
        fallback_positions,
        fallback_available,
        selected_states[..., :3],
        jnp.all(jnp.isfinite(selected_states)),
        descriptors,
        fallback_available & jnp.all(jnp.isfinite(descriptors), axis=-1),
        ghost_states[..., :3],
        ghost_available,
    )
    return _ControlDiagnostics(
        result.motor_forces,
        result.nominal.bounded_motor_forces,
        result.library.hard_values[:, 0]
        if result.library.hard_values.ndim == 2
        else result.library.hard_values,
        result.library.smooth_values[:, 0]
        if hasattr(result.library, "smooth_values")
        else result.library.safety_values.robust_smooth_margins[:, 0],
        result.selection.selected_index,
        result.degraded,
        _discrete_kkt(result, lower, upper),
        post,
        jnp.any(
            jnp.abs(result.nominal.raw_motor_forces - result.nominal.bounded_motor_forces) > 1e-7
        )
        | filtered.fallback_substituted,
        jnp.any((result.motor_forces <= lower + 1e-6) | (result.motor_forces >= upper - 1e-6)),
        *visual,
    )


def _analytic_diagnostics(
    result: Any,
    nominal: Any,
    actuator: VersionAActuator,
    state: Array,
    model: VersionAModel,
    *,
    dt: float,
    horizon: int,
) -> _ControlDiagnostics:
    continuous = result.continuous_filter
    motor = wrench_to_motor_forces(
        result.action,
        L=actuator.arm_length,
        thrust2torque=actuator.thrust_to_torque,
        mixing_matrix=actuator.mixing_matrix,
    )
    lower = jnp.broadcast_to(jnp.asarray(actuator.thrust_min), (4,))
    upper = jnp.broadcast_to(jnp.asarray(actuator.thrust_max), (4,))
    visual = _empty_library_visual_evidence(
        state, nominal.bounded_motor_forces, model, actuator, dt=dt, horizon=horizon
    )
    return _ControlDiagnostics(
        motor,
        nominal.bounded_motor_forces,
        jnp.zeros((1,), dtype=motor.dtype),
        jnp.zeros((1,), dtype=motor.dtype),
        jnp.asarray(-1, dtype=jnp.int32),
        result.degraded,
        _polytope_kkt(continuous.qp),
        jnp.minimum(
            result.applied_interval_margin, continuous.applied_postcheck.minimum_analytic_residual
        ),
        jnp.any(jnp.abs(nominal.raw_motor_forces - nominal.bounded_motor_forces) > 1e-7),
        jnp.any((motor <= lower + 1e-6) | (motor >= upper - 1e-6)),
        *visual,
    )


def _nominal_diagnostics(
    nominal: Any,
    actuator: VersionAActuator,
    state: Array,
    model: VersionAModel,
    *,
    dt: float,
    horizon: int,
) -> _ControlDiagnostics:
    lower = jnp.broadcast_to(jnp.asarray(actuator.thrust_min), (4,))
    upper = jnp.broadcast_to(jnp.asarray(actuator.thrust_max), (4,))
    motor = nominal.bounded_motor_forces
    visual = _empty_library_visual_evidence(state, motor, model, actuator, dt=dt, horizon=horizon)
    return _ControlDiagnostics(
        motor,
        motor,
        jnp.zeros((1,), dtype=motor.dtype),
        jnp.zeros((1,), dtype=motor.dtype),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(True),
        jnp.asarray(0.0, dtype=motor.dtype),
        jnp.asarray(0.0, dtype=motor.dtype),
        jnp.any(jnp.abs(nominal.raw_motor_forces - motor) > 1e-7),
        jnp.any((motor <= lower + 1e-6) | (motor >= upper - 1e-6)),
        *visual,
    )


def _barrier_trace(
    states: np.ndarray, tape: ScenarioTape, config: ExperimentConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute finite node and conservative continuous-segment hard evidence."""
    steps = states.shape[0]
    positions = states[:, :3]
    speed = np.linalg.norm(states[:, 7:10], axis=-1)
    angular = np.linalg.norm(states[:, 10:13], axis=-1)
    # Persisted physical evidence must reconstruct independently of the controller/plant backend.
    # In particular, float32 quaternion normalization differs by a few ulps between CPU and GPU
    # XLA lowerings.  Keep this post-run calculation on the host in float64, like the other hard
    # barrier terms, so a GPU-produced trace replays exactly in a fresh CPU-only process.
    rotations = quaternion_to_rotation_matrix(np.asarray(states[:, 3:7], dtype=np.float64))
    cosine_tilt = rotations[:, 2, 2]
    span = tape.arena_upper - tape.arena_lower
    arena = np.min(
        np.concatenate(
            (
                (positions - tape.arena_lower - float(tape.vehicle_radius)) / span,
                (tape.arena_upper - positions - float(tape.vehicle_radius)) / span,
            ),
            axis=-1,
        ),
        axis=-1,
    )
    speed_margin = 1.0 - (speed / config.speed_limit) ** 2
    angular_margin = 1.0 - (angular / config.angular_rate_max) ** 2
    tilt_margin = (cosine_tilt - math.cos(config.tilt_max_radians)) / (
        1.0 - math.cos(config.tilt_max_radians)
    )

    def node_margin(centers: np.ndarray, radii: np.ndarray, mask: np.ndarray) -> np.ndarray:
        effective = radii + float(tape.vehicle_radius) + config.obstacle_clearance
        distance_squared = np.sum((positions[:, None] - centers) ** 2, axis=-1)
        values = (distance_squared - effective**2) / np.maximum(effective**2, 1e-12)
        sentinel = np.full(values.shape, 2.0)
        return np.min(np.where(mask, values, sentinel), axis=-1)

    static_centers = np.broadcast_to(
        tape.static_positions[None], (steps, *tape.static_positions.shape)
    )
    static_mask = np.broadcast_to(tape.static_mask[None], (steps, tape.static_mask.size))
    static_node = node_margin(static_centers, tape.static_radii[None], static_mask)
    dynamic_node = node_margin(
        tape.dynamic_positions[:steps], tape.dynamic_radii[None], tape.dynamic_time_mask[:steps]
    )

    def swept_margin(dynamic: bool, *, extra_clearance: float) -> np.ndarray:
        output = np.empty((steps,), dtype=np.float64)
        output[0] = dynamic_node[0] if dynamic else static_node[0]
        for index in range(1, steps):
            if dynamic:
                centers_start = tape.dynamic_positions[index - 1]
                centers_end = tape.dynamic_positions[index]
                radii = tape.dynamic_radii
                mask = tape.dynamic_time_mask[index - 1] | tape.dynamic_time_mask[index]
            else:
                centers_start = tape.static_positions
                centers_end = tape.static_positions
                radii = tape.static_radii
                mask = tape.static_mask
            relative_start = positions[index - 1, None] - centers_start
            relative_delta = (positions[index] - positions[index - 1])[None] - (
                centers_end - centers_start
            )
            denominator = np.sum(relative_delta**2, axis=-1)
            fraction = np.where(
                denominator > 0,
                -np.sum(relative_start * relative_delta, axis=-1) / np.maximum(denominator, 1e-30),
                0.0,
            )
            closest = relative_start + np.clip(fraction, 0.0, 1.0)[:, None] * relative_delta
            effective = radii + float(tape.vehicle_radius) + extra_clearance
            values = (np.sum(closest**2, axis=-1) - effective**2) / np.maximum(effective**2, 1e-12)
            output[index] = np.min(np.where(mask, values, 2.0))
        return output

    static_swept = swept_margin(False, extra_clearance=config.obstacle_clearance)
    dynamic_swept = swept_margin(True, extra_clearance=config.obstacle_clearance)
    barriers = np.stack(
        (
            static_node,
            dynamic_node,
            arena,
            speed_margin,
            angular_margin,
            tilt_margin,
            static_swept,
            dynamic_swept,
        ),
        axis=-1,
    )
    padded_positions = np.concatenate(
        (positions, np.broadcast_to(positions[-1], (tape.steps - steps, 3))), axis=0
    )
    node_contact = hard_contact_labels(padded_positions, tape).any_contact[:steps]
    physical_static_swept = swept_margin(False, extra_clearance=0.0)
    physical_dynamic_swept = swept_margin(True, extra_clearance=0.0)
    swept_contact = (physical_static_swept <= 0.0) | (physical_dynamic_swept <= 0.0)
    swept_contact[0] = node_contact[0]
    contact = node_contact | swept_contact
    failure = contact | np.any(barriers < 0.0, axis=-1)
    return barriers, contact, failure


def _scenario_digest(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode()
    return hashlib.sha256(b"crazyflow.da_plcbf.experiment.v1\0" + payload).hexdigest()


def _numeric_digest(label: str, *arrays: Any) -> str:
    """Hash exact numeric scenario content, including dtype and shape boundaries."""
    digest = hashlib.sha256(b"crazyflow.da_plcbf.numeric-fold.v1\0" + label.encode())
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _training_batch(
    tape: ScenarioTape, state: Array, start_index: int, config: ExperimentConfig
) -> tuple[Array, CircleScenarioBatch, RigidBodySafetySet]:
    """Build the held-in proposal-training batch; hard validation never reuses these rows."""
    batch = config.training_scenario_count
    indices = _causal_history_indices(start_index, batch)
    prediction_indices = np.arange(batch) % tape.prediction_samples
    centers: list[np.ndarray] = []
    radii: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for index, prediction in zip(indices, prediction_indices, strict=True):
        centers.append(
            np.concatenate(
                (tape.static_positions, tape.prediction_positions[prediction, index]), axis=0
            )
        )
        radii.append(np.concatenate((tape.static_radii, tape.dynamic_radii), axis=0))
        masks.append(np.concatenate((tape.static_mask, tape.dynamic_time_mask[index]), axis=0))
    circles = CircleScenarioBatch(
        obstacle_centers=jnp.asarray(np.stack(centers), dtype=jnp.float32),
        obstacle_radii=jnp.asarray(np.stack(radii), dtype=jnp.float32),
        obstacle_mask=jnp.asarray(np.stack(masks)),
        arena_lower=jnp.asarray(np.broadcast_to(tape.arena_lower, (batch, 3)), dtype=jnp.float32),
        arena_upper=jnp.asarray(np.broadcast_to(tape.arena_upper, (batch, 3)), dtype=jnp.float32),
        speed_limit=jnp.full((batch,), config.speed_limit, dtype=jnp.float32),
    )
    # Deterministic small local perturbations generate reachable-neighbour validation states.  They
    # are scenario construction only and never choose a runtime maneuver.
    phase = 2.0 * np.pi * (np.arange(batch) + 0.5) / batch
    offsets = np.stack((np.cos(phase), np.sin(phase), 0.25 * np.sin(2.0 * phase)), axis=-1)
    initial = jnp.broadcast_to(state[None], (batch, 13))
    initial = initial.at[:, :3].add(jnp.asarray(0.03 * offsets, dtype=state.dtype))
    initial = initial.at[:, 7:10].add(jnp.asarray(0.05 * offsets, dtype=state.dtype))
    safety = _safety_from_circles(circles, config)
    return initial, circles, safety


def _causal_history_indices(start_index: int, count: int) -> np.ndarray:
    """Return current/past-only replay indices; no evaluation-future row is reachable."""
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a nonnegative integer")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    return np.maximum(start_index - 7 * np.arange(count, dtype=np.int64), 0)


def _offline_training_batch(
    tape: ScenarioTape, config: ExperimentConfig
) -> tuple[Array, CircleScenarioBatch, np.ndarray]:
    """Build the predeclared offline batch from the *whole* auxiliary tape.

    Causality constrains online adaptation on an evaluation trajectory; it must not collapse a
    condition-level offline training set to repeated copies of time zero.  The frozen comparator
    therefore uses deterministic, evenly spaced indices over every available auxiliary-tape node.
    Those exact indices are returned for content-addressed provenance.
    """
    batch = config.training_scenario_count
    indices = np.rint(np.linspace(0, tape.steps - 1, num=batch)).astype(np.int64)
    prediction_indices = np.arange(batch, dtype=np.int64) % tape.prediction_samples
    centers = np.stack(
        [
            np.concatenate(
                (tape.static_positions, tape.prediction_positions[prediction, index]), axis=0
            )
            for index, prediction in zip(indices, prediction_indices, strict=True)
        ]
    )
    radii = np.broadcast_to(
        np.concatenate((tape.static_radii, tape.dynamic_radii), axis=0),
        (batch, tape.static_radii.size + tape.dynamic_radii.size),
    )
    masks = np.stack(
        [
            np.concatenate((tape.static_mask, tape.dynamic_time_mask[index]), axis=0)
            for index in indices
        ]
    )
    circles = CircleScenarioBatch(
        obstacle_centers=jnp.asarray(centers, dtype=jnp.float32),
        obstacle_radii=jnp.asarray(radii, dtype=jnp.float32),
        obstacle_mask=jnp.asarray(masks),
        arena_lower=jnp.asarray(np.broadcast_to(tape.arena_lower, (batch, 3)), dtype=jnp.float32),
        arena_upper=jnp.asarray(np.broadcast_to(tape.arena_upper, (batch, 3)), dtype=jnp.float32),
        speed_limit=jnp.full((batch,), config.speed_limit, dtype=jnp.float32),
    )
    phase = 2.0 * np.pi * (np.arange(batch) + 0.5) / batch
    offsets = np.stack((np.cos(phase), np.sin(phase), 0.25 * np.sin(2.0 * phase)), axis=-1)
    states = np.zeros((batch, 13), dtype=np.float32)
    states[:, :3] = tape.defender_reference_position[indices] + 0.03 * offsets
    states[:, 6] = 1.0
    states[:, 7:10] = tape.defender_reference_velocity[indices] + 0.05 * offsets
    return jnp.asarray(states), circles, indices


def _predeclared_auxiliary_tape(
    condition: ConditionID, config: ExperimentConfig, *, purpose: str
) -> ScenarioTape:
    """Generate a condition-level tape independent of every evaluation fold and tape digest."""
    digest = hashlib.sha256(
        b"crazyflow.da_plcbf.auxiliary-tape.v1\0"
        + str(config.random_seed).encode()
        + b"\0"
        + condition.value.encode()
        + b"\0"
        + purpose.encode()
    ).digest()
    seed = int.from_bytes(digest[:4], "little")
    fold_offset = 1_000_003 if purpose == "proposal-training" else 2_000_003
    fold = (config.random_seed + fold_offset) % (np.iinfo(np.uint32).max + 1)
    return generate_condition_tape(condition, config, seed=seed, fold=fold)


def _auxiliary_tape(
    evaluation_tape: ScenarioTape, condition: ConditionID, config: ExperimentConfig, *, purpose: str
) -> ScenarioTape:
    """Return the predeclared auxiliary tape and prove it differs from this evaluation tape."""
    tape = _predeclared_auxiliary_tape(condition, config, purpose=purpose)
    if tape.sha256 == evaluation_tape.sha256:
        raise RuntimeError("auxiliary and evaluation scenario tapes are not content-disjoint")
    return tape


_VALIDATION_FOLD_NAMES = ("current", "perturbed", "replay", "reachable", "dynamics", "obstacle")


class _HardValidationBatch(NamedTuple):
    """Predeclared disjoint hard-validation folds and their exact content digest."""

    initial_states: Array
    scenarios: DynamicSphereScenarioBatch
    fold_index: np.ndarray
    fold_names: tuple[str, ...]
    digest: str


def _concatenate_dynamic_windows(
    windows: Sequence[DynamicSphereScenarioBatch],
) -> DynamicSphereScenarioBatch:
    return DynamicSphereScenarioBatch(
        obstacle_centers=jnp.concatenate(tuple(item.obstacle_centers for item in windows), axis=0),
        obstacle_radii=jnp.concatenate(tuple(item.obstacle_radii for item in windows), axis=0),
        obstacle_mask=jnp.concatenate(tuple(item.obstacle_mask for item in windows), axis=0),
        arena_lower=jnp.concatenate(tuple(item.arena_lower for item in windows), axis=0),
        arena_upper=jnp.concatenate(tuple(item.arena_upper for item in windows), axis=0),
        speed_limit=jnp.concatenate(tuple(item.speed_limit for item in windows), axis=0),
        angular_rate_max=jnp.concatenate(tuple(item.angular_rate_max for item in windows), axis=0),
        tilt_max_radians=jnp.concatenate(tuple(item.tilt_max_radians for item in windows), axis=0),
    )


def _hard_validation_batch(
    tape: ScenarioTape,
    heldout_tape: ScenarioTape,
    state: Array,
    start_index: int,
    controller_model: VersionAModel,
    resources: ExperimentResources,
    config: ExperimentConfig,
) -> _HardValidationBatch:
    """Construct six deterministic, held-out folds with Cartesian obstacle/model evaluation.

    The rows are generated independently from :func:`_training_batch`: they use different state
    constructions and full time-varying prediction windows.  ``replay`` reuses exogenous reference
    states from the immutable tape, while ``reachable`` is obtained by actual nominal plant
    transitions rather than an arbitrary state perturbation.
    """
    per_fold = config.validation_scenarios_per_fold
    dtype = state.dtype
    states: list[Array] = []
    windows: list[DynamicSphereScenarioBatch] = []
    fold_indices: list[int] = []
    maximum_start = config.control_steps
    base_phase = 2.0 * np.pi * (np.arange(per_fold) + 0.5) / per_fold
    offsets = np.stack(
        (np.cos(base_phase), np.sin(base_phase), 0.35 * np.sin(2.0 * base_phase)), axis=-1
    )

    def append(
        fold: int, candidate: Array, scenario_index: int, source_tape: ScenarioTape = tape
    ) -> None:
        states.append(jnp.asarray(candidate, dtype=dtype))
        windows.append(
            dynamic_sphere_window_from_tape(
                source_tape,
                start_index=int(scenario_index % (maximum_start + 1)),
                horizon=config.certificate_horizon,
                speed_limit=config.speed_limit,
                angular_rate_max=config.angular_rate_max,
                tilt_max_radians=config.tilt_max_radians,
            )
        )
        fold_indices.append(fold)

    # Exact current state first, followed by tiny distinct measurement-neighbour rows.
    for slot in range(per_fold):
        candidate = (
            state
            if slot == 0
            else state.at[:3].add(jnp.asarray(0.005 * offsets[slot], dtype=dtype))
        )
        append(0, candidate, start_index)

    # Larger local state perturbations are held out from the smaller training ring.
    for slot in range(per_fold):
        candidate = state.at[:3].add(jnp.asarray(0.07 * offsets[slot], dtype=dtype))
        candidate = candidate.at[7:10].add(jnp.asarray(0.12 * offsets[::-1][slot], dtype=dtype))
        append(1, candidate, start_index)

    # Immutable exogenous tape replay: reference states at disjoint deterministic tape indices.
    for slot in range(per_fold):
        use_heldout = start_index == 0
        source = heldout_tape if use_heldout else tape
        replay_index = (
            (3 + 5 * slot) % (maximum_start + 1)
            if use_heldout
            else max(start_index - 1 - 5 * slot, 0)
        )
        candidate = state.at[:3].set(
            jnp.asarray(source.defender_reference_position[replay_index], dtype=dtype)
        )
        candidate = candidate.at[7:10].set(
            jnp.asarray(source.defender_reference_velocity[replay_index], dtype=dtype)
        )
        append(2, candidate, replay_index, source)

    # Reachable neighbours come from one or more real nominal held-step transitions.
    reachable = state
    for slot in range(per_fold):
        target_index = min(start_index + slot, config.control_steps - 1)
        nominal = waypoint_nominal_wrench(
            reachable,
            jnp.asarray(tape.defender_reference_position[target_index], dtype=dtype),
            jnp.asarray(tape.defender_reference_velocity[target_index], dtype=dtype),
            controller_model,
            resources.actuator,
            resources.quad_config,
        )
        reachable = direct_wrench_symplectic_step(
            reachable, nominal.wrench, controller_model, config.dt
        )
        append(3, reachable, start_index)

    # Dynamics-stress states vary translational and angular rates; every row is subsequently
    # rolled through the complete estimator-derived R_m set.
    for slot in range(per_fold):
        candidate = state.at[7:10].add(jnp.asarray(0.3 * offsets[slot], dtype=dtype))
        candidate = candidate.at[10:13].add(jnp.asarray(0.2 * offsets[::-1][slot], dtype=dtype))
        append(4, candidate, start_index)

    # Obstacle-stress rows keep the measured state local but use separated future tape windows.
    for slot in range(per_fold):
        obstacle_index = (7 + 11 * slot) % (maximum_start + 1)
        candidate = state.at[:3].add(jnp.asarray(0.04 * offsets[::-1][slot], dtype=dtype))
        append(5, candidate, obstacle_index, heldout_tape)

    initial_states = jnp.stack(states)
    scenarios = _concatenate_dynamic_windows(windows)
    fold_index = np.asarray(fold_indices, dtype=np.int16)
    digest = _numeric_digest(
        "scenario-content",
        initial_states,
        scenarios.obstacle_centers,
        scenarios.obstacle_radii,
        scenarios.obstacle_mask,
        scenarios.arena_lower,
        scenarios.arena_upper,
        scenarios.speed_limit,
        scenarios.angular_rate_max,
        scenarios.tilt_max_radians,
        fold_index,
    )
    return _HardValidationBatch(
        initial_states, scenarios, fold_index, _VALIDATION_FOLD_NAMES, digest
    )


def _candidate_evidence_device(
    params: SharedActorParams,
    active_params: SharedActorParams,
    spec: SharedActorSpec,
    validation_initial_states: Array,
    validation_scenarios: DynamicSphereScenarioBatch,
    current_state: Array,
    current_window: Any,
    controller_model: VersionAModel,
    model_samples: VersionAModelSamples,
    resources: ExperimentResources,
    config: ExperimentConfig,
) -> tuple[Array, Array, Array, Array, Array]:
    """Return device-resident hard-validation evidence for one fixed-shape candidate."""

    def evaluate(candidate: SharedActorParams) -> tuple[Any, Any]:
        evaluation = evaluate_uncertain_dynamic_quad_library(
            candidate,
            spec,
            validation_initial_states,
            validation_scenarios,
            controller_model,
            model_samples,
            resources.actuator,
            resources.actor_config,
            resources.quad_config,
            resources.barrier_config,
            dt=config.dt,
            policy_gain=config.policy_gain,
            softmin_beta=40.0,
            uncertainty_config=resources.uncertainty_config,
        )
        return evaluation.rollouts, evaluation

    candidate_rollout, candidate_values = evaluate(params)
    _, active_values = evaluate(active_params)
    current = evaluate_uncertain_dynamic_quad_library(
        params,
        spec,
        current_state[None],
        current_window._replace(
            obstacle_centers=current_window.obstacle_centers[:, :, :-1],
            obstacle_radii=current_window.obstacle_radii[:, :, :-1],
            obstacle_mask=current_window.obstacle_mask[:, :, :-1],
        ),
        controller_model,
        model_samples,
        resources.actuator,
        resources.actor_config,
        resources.quad_config,
        resources.barrier_config,
        dt=config.dt,
        policy_gain=config.policy_gain,
        softmin_beta=40.0,
        uncertainty_config=resources.uncertainty_config,
    )
    translation = jnp.concatenate(
        (candidate_rollout.states[..., :3], candidate_rollout.states[..., 7:10]), axis=-1
    )
    policy_count, batch, obstacle_count, model_count, nodes, _ = translation.shape
    flattened_translation = translation.reshape(
        (policy_count, batch * obstacle_count * model_count, nodes, 6)
    )
    descriptors = jnp.mean(
        trajectory_descriptors(flattened_translation).reshape(
            (policy_count, batch, obstacle_count, model_count, 9)
        ),
        axis=(1, 2, 3),
    )
    lower = jnp.broadcast_to(jnp.asarray(resources.actuator.thrust_min), (4,))
    upper = jnp.broadcast_to(jnp.asarray(resources.actuator.thrust_max), (4,))
    motors = candidate_rollout.commanded_motor_forces
    feasibility = jnp.minimum(motors - lower, upper - motors)
    return (
        current.hard_values[:, 0],
        candidate_values.hard_values,
        active_values.hard_values,
        descriptors,
        feasibility,
    )


def _candidate_evidence(
    params: SharedActorParams,
    active_params: SharedActorParams,
    spec: SharedActorSpec,
    validation: _HardValidationBatch,
    current_state: Array,
    current_window: Any,
    controller_model: VersionAModel,
    model_samples: VersionAModelSamples,
    resources: ExperimentResources,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate and synchronize held-out Cartesian hard evidence on the current backend."""
    outputs = _candidate_evidence_device(
        params,
        active_params,
        spec,
        validation.initial_states,
        validation.scenarios,
        current_state,
        current_window,
        controller_model,
        model_samples,
        resources,
        config,
    )
    _block(outputs)
    return tuple(np.asarray(value) for value in outputs)  # type: ignore[return-value]


def _online_bptt_device() -> jax.Device:
    """Prefer the first CUDA device for online JIT BPTT, with a CPU-only fallback.

    Production GPU runs keep differentiable rollout, reverse-mode differentiation, and the
    fixed-budget optimizer burst on the accelerator.  The fallback keeps CPU-only development and
    unit-test environments usable; every event records which backend was actually selected.
    """
    try:
        gpu_devices = jax.devices("gpu")
    except (RuntimeError, ValueError):
        gpu_devices = []
    if gpu_devices:
        return gpu_devices[0]
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        raise RuntimeError("online BPTT requires an available JAX GPU or CPU device")
    return cpu_devices[0]


def _authoritative_estimator_device() -> jax.Device:
    """Return the CPU device shared by estimator production and evidence replay."""
    devices = jax.devices("cpu")
    if not devices:
        raise RuntimeError("the causal estimator requires an available JAX CPU device")
    return devices[0]


def _authoritative_model(
    base_model: VersionAModel, *, device: jax.Device | None = None
) -> VersionAModel:
    """Canonicalize static model fields whose construction may depend on ambient XLA."""
    device = _authoritative_estimator_device() if device is None else device
    if str(device.platform) != "cpu":
        raise ValueError("authoritative model canonicalization requires a CPU device")
    with jax.default_device(device):
        model = jax.device_put(base_model, device)
        model = model._replace(inertia_inv=jnp.linalg.inv(model.inertia))
    return jax.device_put(model, device)


def _authoritative_resources(
    resources: ExperimentResources, *, device: jax.Device | None = None
) -> ExperimentResources:
    """Place all numerical proof resources on CPU and canonicalize derived model fields."""
    device = _authoritative_estimator_device() if device is None else device
    if str(device.platform) != "cpu":
        raise ValueError("authoritative proof resources require a CPU device")
    return replace(
        resources,
        model=_authoritative_model(resources.model, device=device),
        actuator=jax.device_put(resources.actuator, device),
        spec=jax.device_put(resources.spec, device),
        initial_params=jax.device_put(resources.initial_params, device),
    )


def _resources_on_device(resources: ExperimentResources, device: jax.Device) -> ExperimentResources:
    """Canonicalize model constants once on CPU, then place numerical resources on ``device``."""
    canonical = _authoritative_resources(resources)
    with jax.default_device(device):
        return replace(
            canonical,
            model=jax.device_put(canonical.model, device),
            actuator=jax.device_put(canonical.actuator, device),
            spec=jax.device_put(canonical.spec, device),
            initial_params=jax.device_put(canonical.initial_params, device),
        )


def _initialize_authoritative_estimator(
    base_model: VersionAModel,
    estimator_config: EstimatorConfig,
    *,
    device: jax.Device | None = None,
) -> EstimatorState:
    """Initialize the causal estimator from canonical CPU-resident model values."""
    device = _authoritative_estimator_device() if device is None else device
    if str(device.platform) != "cpu":
        raise ValueError("authoritative estimator initialization requires a CPU device")
    with jax.default_device(device):
        model = _authoritative_model(base_model, device=device)
        estimator = initialize_estimator(
            estimator_config,
            mass=float(model.mass),
            drag_force_coefficients=-jnp.diag(model.drag_matrix),
            wind_velocity=model.wind_velocity,
            rotor_efficiency=1.0,
        )
    return jax.device_put(estimator, device)


def _authoritative_model_samples(
    base_model: VersionAModel,
    estimator: EstimatorState,
    estimator_config: EstimatorConfig,
    sample_count: int,
    *,
    device: jax.Device | None = None,
) -> tuple[VersionAModel, VersionAModelSamples]:
    """Derive every estimator-dependent BPTT context value on authoritative CPU."""
    device = _authoritative_estimator_device() if device is None else device
    if str(device.platform) != "cpu":
        raise ValueError("authoritative adaptation context requires a CPU device")
    with jax.default_device(device):
        model = _authoritative_model(base_model, device=device)
        estimator = jax.device_put(estimator, device)
        controller_model = _controller_model(model, estimator)
        particles = deterministic_parameter_samples(
            estimator, sample_count=sample_count, config=estimator_config
        )
        samples = version_a_model_samples_from_estimator(
            particles, controller_model, estimator_config
        )
    return jax.device_put(controller_model, device), jax.device_put(samples, device)


def _authoritative_estimator_function(estimator_config: EstimatorConfig) -> Any:
    """Build the one compiled estimator body used by production and fresh replay."""

    def estimate(
        current: EstimatorState,
        translational: TranslationalObservations,
        rotor: RotorEfficiencyObservations,
        sequence: Array,
    ) -> tuple[Any, Any]:
        rotor_update = update_rotor_efficiency(
            current, rotor, sequence=sequence, mode="per_rotor", config=estimator_config
        )
        translation_update = update_translational_estimate(
            rotor_update.state, translational, sequence=sequence, config=estimator_config
        )
        return translation_update, rotor_update

    return jax.jit(estimate)


def _authoritative_estimator_arguments(
    estimator: EstimatorState,
    observations: tuple[TranslationalObservations, RotorEfficiencyObservations],
    sequence: int | Array,
    *,
    device: jax.Device | None = None,
) -> tuple[Any, ...]:
    """Place one fixed-shape estimator invocation entirely on authoritative CPU."""
    device = _authoritative_estimator_device() if device is None else device
    if str(device.platform) != "cpu":
        raise ValueError("authoritative estimator execution requires a CPU device")
    with jax.default_device(device):
        arguments = (
            jax.device_put(estimator, device),
            jax.device_put(observations[0], device),
            jax.device_put(observations[1], device),
            jax.device_put(jnp.asarray(sequence, dtype=jnp.int32), device),
        )
    return arguments


def _bptt_runtime_input_digest(arguments: tuple[Any, ...]) -> str:
    """Hash every exact runtime leaf supplied to one online BPTT burst."""
    return _numeric_digest("online-bptt-runtime-inputs-v1", *jax.tree.leaves(arguments))


def _bptt_executable_signature(
    resources: ExperimentResources, config: ExperimentConfig, *, device: jax.Device | None = None
) -> str:
    device = _online_bptt_device() if device is None else device
    canonical_resources = _authoritative_resources(resources)
    numeric = _numeric_digest(
        "bptt-static-resources",
        *jax.tree.leaves(canonical_resources.spec),
        *jax.tree.leaves(canonical_resources.actuator),
    )
    return _scenario_digest(
        "dynamic-model-online-bptt-executable-v4",
        BPTT_EXECUTION_CONTRACT,
        f"{device.platform}:{device.id}",
        config,
        resources.actor_config,
        resources.quad_config,
        resources.barrier_config,
        resources.loss_config,
        numeric,
    )


def _build_bptt_executable_pool(
    resources: ExperimentResources, config: ExperimentConfig, *, device: jax.Device | None = None
) -> _BPTTExecutablePool:
    device = _online_bptt_device() if device is None else device
    learning = QuadLearningConfig(
        dt=config.dt, horizon=config.certificate_horizon, policy_gain=config.policy_gain
    )
    with jax.default_device(device):
        runtime_resources = _resources_on_device(resources, device)
        spec = runtime_resources.spec
        actuator = runtime_resources.actuator
        bptt = build_dynamic_model_quad_actor_bptt_functions(
            spec,
            actuator,
            resources.actor_config,
            resources.quad_config,
            resources.barrier_config,
            learning,
            resources.loss_config,
            burst_steps=config.bptt_burst_steps,
            device=device,
        )
    return _BPTTExecutablePool(
        _bptt_executable_signature(resources, config, device=device), bptt, device
    )


class _CandidateJob:
    """Thread-safe BPTT + hard-validation job used by :class:`AdaptationWorker`."""

    def __init__(
        self,
        tape: ScenarioTape,
        condition: ConditionID,
        resources: ExperimentResources,
        config: ExperimentConfig,
        executable_pool: _BPTTExecutablePool | None = None,
    ) -> None:
        self._tape = tape
        self._heldout_tape = _auxiliary_tape(tape, condition, config, purpose="hard-validation")
        self._resources = resources
        self._config = config
        self._lock = threading.RLock()
        self._contexts: dict[
            int, list[tuple[int, Array, VersionAModel, VersionAModelSamples, Any]]
        ] = {}
        pool = executable_pool or _build_bptt_executable_pool(resources, config)
        if pool.signature != _bptt_executable_signature(resources, config, device=pool.device):
            raise ValueError("BPTT executable pool is incompatible with this trial")
        # Model parameters are runtime values.  Estimator updates and paired folds therefore reuse
        # one compiled executable instead of manufacturing value-specialized closures.
        self._adaptation_device = pool.device
        self._bptt = pool.bptt
        self._bptt_device_key = pool.device_key
        self._compiled_bursts = pool.compiled_bursts
        self._compile_timings = pool.compile_timings
        self._compiled_evidence = pool.compiled_evidence
        self._evidence_compile_timings = pool.evidence_compile_timings
        self._compile_lock = pool.lock
        self.diagnostics: dict[str, dict[str, Any]] = {}
        self.validation_material: dict[str, CandidateValidationMaterial] = {}

    @property
    def adaptation_device(self) -> jax.Device:
        """Device used by the compiled online BPTT and hard-admission graphs."""
        return self._adaptation_device

    @staticmethod
    def _on_device(tree: Any, device: jax.Device | None) -> Any:
        return tree if device is None else jax.device_put(tree, device)

    def _resources_on_device(self, device: jax.Device | None) -> ExperimentResources:
        if device is None:
            return self._resources
        return _resources_on_device(self._resources, device)

    def _bptt_arguments(
        self,
        active_params: SharedActorParams,
        state: Array,
        start_index: int,
        controller_model: VersionAModel,
        device: jax.Device | None,
    ) -> tuple[tuple[Any, ...], str, Array, str]:
        with nullcontext() if device is None else jax.default_device(device):
            initial_states, circles, safety = _training_batch(
                self._tape, self._on_device(state, device), start_index, self._config
            )
            training_digest = _numeric_digest(
                "scenario-content",
                initial_states,
                circles.obstacle_centers,
                circles.obstacle_radii,
                circles.obstacle_mask,
            )
            params = self._on_device(_tree_device(active_params), device)
            optimizer_state = self._on_device(self._bptt.initialize(params), device)
            proof_spec = self._resources_on_device(device).spec
            targets = self._on_device(descriptor_targets_from_spec(proof_spec), device)
            descriptor_scales = self._on_device(
                jnp.asarray([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=state.dtype),
                device,
            )
            arguments = (
                optimizer_state,
                self._on_device(initial_states, device),
                self._on_device(circles, device),
                self._on_device(safety, device),
                targets,
                params,
                descriptor_scales,
                self._on_device(controller_model, device),
            )
        return (
            arguments,
            training_digest,
            descriptor_scales,
            _bptt_runtime_input_digest(arguments),
        )

    def _ensure_compiled(
        self, cache_key: str, arguments: tuple[Any, ...]
    ) -> tuple[Any, bool, float, float]:
        with self._compile_lock:
            cached = self._compiled_bursts.get(cache_key)
            cached_timing = self._compile_timings.get(cache_key)
        if cached is not None and cached_timing is not None:
            return cached, True, 0.0, 0.0

        compile_start = time.perf_counter()
        compiled = self._bptt.burst.lower(*arguments).compile()
        compile_seconds = time.perf_counter() - compile_start
        warmup_start = time.perf_counter()
        _block(compiled(*arguments))
        warmup_seconds = time.perf_counter() - warmup_start
        with self._compile_lock:
            # The worker is single-flight, but keep publication deterministic if a future caller
            # explicitly prewarms at the same instant.
            existing = self._compiled_bursts.setdefault(cache_key, compiled)
            timing = self._compile_timings.setdefault(cache_key, (compile_seconds, warmup_seconds))
        return existing, False, timing[0], timing[1]

    def _evidence_arguments(
        self,
        candidate_params: SharedActorParams,
        active_params: SharedActorParams,
        validation: _HardValidationBatch,
        state: Array,
        window: Any,
        controller_model: VersionAModel,
        model_samples: VersionAModelSamples,
        device: jax.Device | None,
    ) -> tuple[tuple[Any, ...], ExperimentResources]:
        resources = self._resources_on_device(device)
        arguments = (
            self._on_device(candidate_params, device),
            self._on_device(active_params, device),
            self._on_device(validation.initial_states, device),
            self._on_device(validation.scenarios, device),
            self._on_device(state, device),
            self._on_device(window, device),
            self._on_device(controller_model, device),
            self._on_device(model_samples, device),
        )
        return arguments, resources

    def _ensure_evidence_compiled(
        self, cache_key: str, arguments: tuple[Any, ...], resources: ExperimentResources
    ) -> tuple[Any, bool, float, float]:
        """Compile the full hard-evidence graph once per backend and fixed shape."""
        with self._compile_lock:
            cached = self._compiled_evidence.get(cache_key)
            cached_timing = self._evidence_compile_timings.get(cache_key)
        if cached is not None and cached_timing is not None:
            return cached, True, 0.0, 0.0

        def evidence(
            candidate_params: SharedActorParams,
            active_params: SharedActorParams,
            validation_initial_states: Array,
            validation_scenarios: DynamicSphereScenarioBatch,
            state: Array,
            window: Any,
            controller_model: VersionAModel,
            model_samples: VersionAModelSamples,
        ) -> tuple[Array, Array, Array, Array, Array]:
            return _candidate_evidence_device(
                candidate_params,
                active_params,
                resources.spec,
                validation_initial_states,
                validation_scenarios,
                state,
                window,
                controller_model,
                model_samples,
                resources,
                self._config,
            )

        function = jax.jit(evidence)
        compile_start = time.perf_counter()
        compiled = function.lower(*arguments).compile()
        compile_seconds = time.perf_counter() - compile_start
        warmup_start = time.perf_counter()
        _block(compiled(*arguments))
        warmup_seconds = time.perf_counter() - warmup_start
        with self._compile_lock:
            existing = self._compiled_evidence.setdefault(cache_key, compiled)
            timing = self._evidence_compile_timings.setdefault(
                cache_key, (compile_seconds, warmup_seconds)
            )
        return existing, False, timing[0], timing[1]

    def precompile_online(
        self,
        active: PolicySnapshot,
        state: Array,
        controller_model: VersionAModel,
        model_samples: VersionAModelSamples,
        window: Any,
        *,
        start_index: int,
    ) -> dict[str, Any]:
        """Compile and warm accelerator BPTT and hard validation before control begins."""
        arguments, training_digest, _, bptt_input_digest = self._bptt_arguments(
            active.params, state, start_index, controller_model, self._adaptation_device
        )
        cache_key = f"online:{self._adaptation_device.platform}:{self._adaptation_device.id}"
        _, hit, compile_seconds, warmup_seconds = self._ensure_compiled(cache_key, arguments)
        with jax.default_device(self._adaptation_device):
            resources = self._resources_on_device(self._adaptation_device)
            validation = _hard_validation_batch(
                self._tape,
                self._heldout_tape,
                self._on_device(state, self._adaptation_device),
                start_index,
                self._on_device(controller_model, self._adaptation_device),
                resources,
                self._config,
            )
            evidence_arguments, evidence_resources = self._evidence_arguments(
                active.params,
                active.params,
                validation,
                state,
                window,
                controller_model,
                model_samples,
                self._adaptation_device,
            )
            evidence_key = (
                f"evidence:online:{self._adaptation_device.platform}:{self._adaptation_device.id}"
            )
            _, evidence_hit, evidence_compile, evidence_warmup = self._ensure_evidence_compiled(
                evidence_key, evidence_arguments, evidence_resources
            )
        return {
            "cache_key": cache_key,
            "cache_hit": hit,
            "compile_seconds": compile_seconds,
            "warmup_seconds": warmup_seconds,
            "evidence_cache_key": evidence_key,
            "evidence_cache_hit": evidence_hit,
            "evidence_compile_seconds": evidence_compile,
            "evidence_warmup_seconds": evidence_warmup,
            "training_batch_digest": training_digest,
            "bptt_input_digest": bptt_input_digest,
            "compilation_excluded_from_execution_timing": True,
        }

    def set_context(
        self,
        model_version: int,
        state: Array,
        controller_model: VersionAModel,
        model_samples: VersionAModelSamples,
        window: Any,
        *,
        start_index: int,
    ) -> None:
        """Publish immutable numerical context before submitting a candidate job."""
        with self._lock:
            self._contexts.setdefault(model_version, []).append(
                (
                    start_index,
                    jnp.asarray(state),
                    jax.tree.map(jnp.asarray, controller_model),
                    jax.tree.map(jnp.asarray, model_samples),
                    window,
                )
            )

    def __call__(
        self, active: PolicySnapshot, model_version: int
    ) -> tuple[PolicySnapshot, ValidationReport]:
        with self._lock:
            try:
                contexts = self._contexts[model_version]
                start_index, state, controller_model, model_samples, window = contexts.pop(0)
                if not contexts:
                    del self._contexts[model_version]
            except (KeyError, IndexError) as error:
                raise RuntimeError(
                    "candidate context missing for captured model version"
                ) from error
        admission_start = time.perf_counter()
        device = self._adaptation_device
        cache_key = f"online:{device.platform}:{device.id}"
        job_resources = self._resources_on_device(device)
        setup_start = time.perf_counter()
        with nullcontext() if device is None else jax.default_device(device):
            state = self._on_device(state, device)
            controller_model = self._on_device(controller_model, device)
            model_samples = self._on_device(model_samples, device)
            window = self._on_device(window, device)
            arguments, training_digest, descriptor_scales, bptt_input_digest = self._bptt_arguments(
                active.params, state, start_index, controller_model, device
            )
            active_params = arguments[5]
            validation = _hard_validation_batch(
                self._tape,
                self._heldout_tape,
                state,
                start_index,
                controller_model,
                job_resources,
                self._config,
            )
            if device is not None:
                validation = validation._replace(
                    initial_states=self._on_device(validation.initial_states, device),
                    scenarios=self._on_device(validation.scenarios, device),
                )
        setup_seconds = time.perf_counter() - setup_start
        if training_digest == validation.digest:
            raise RuntimeError("proposal-training and hard-validation folds are not disjoint")
        compiled_burst, cache_hit, compile_seconds, warmup_seconds = self._ensure_compiled(
            cache_key, arguments
        )
        execution_start = time.perf_counter()
        trained, metrics = compiled_burst(*arguments)
        _block((trained, metrics))
        execution_seconds = time.perf_counter() - execution_start
        accepted_updates = np.asarray(metrics.update_accepted, dtype=np.bool_)
        parameter_deltas = np.asarray(metrics.parameter_delta_norm, dtype=np.float64)
        gradient_norms = np.asarray(metrics.gradient_norm, dtype=np.float64)
        if (
            not np.all(accepted_updates)
            or not np.all(np.isfinite(parameter_deltas))
            or not np.all(np.isfinite(gradient_norms))
        ):
            raise TrialExecutionError("online BPTT rejected or produced non-finite update evidence")
        if not np.any(parameter_deltas > 0.0):
            raise TrialExecutionError("online BPTT produced no parameter change")
        trained_leaves = jax.tree.leaves(trained.params)
        if not trained_leaves or not hasattr(trained_leaves[0], "device"):
            raise TrialExecutionError("online BPTT did not expose its execution device")
        execution_device = trained_leaves[0].device
        execution_backend = str(execution_device.platform)
        execution_device_id = int(execution_device.id)
        execution_device_name = str(execution_device)
        candidate = create_candidate_snapshot(
            trained.params,
            version=active.version + 1,
            base_active=active,
            model_version=model_version,
            structural_core=self._resources.spec,
            metadata={
                "algorithm": "fixed_budget_truncated_bptt",
                "burst_steps": self._config.bptt_burst_steps,
                "bptt_execution_contract": BPTT_EXECUTION_CONTRACT,
                "objective": "plcbf_aligned_coverage_diversity",
                "proposal_training_digest": training_digest,
                "bptt_input_digest": bptt_input_digest,
                "hard_validation_digest": validation.digest,
                "bptt_cache_key": cache_key,
                "bptt_execution_backend": execution_backend,
                "bptt_execution_device_id": execution_device_id,
                "bptt_execution_device": execution_device_name,
                "bptt_compilation_excluded_from_execution_timing": True,
                "bptt_execution_scope": "compiled_burst_only",
            },
        )
        evidence_arguments, evidence_resources = self._evidence_arguments(
            trained.params,
            active_params,
            validation,
            state,
            window,
            controller_model,
            model_samples,
            device,
        )
        evidence_key = f"evidence:online:{device.platform}:{device.id}"
        (
            compiled_evidence,
            evidence_cache_hit,
            evidence_compile_seconds,
            evidence_warmup_seconds,
        ) = self._ensure_evidence_compiled(evidence_key, evidence_arguments, evidence_resources)
        validation_start = time.perf_counter()
        with nullcontext() if device is None else jax.default_device(device):
            device_evidence = compiled_evidence(*evidence_arguments)
            _block(device_evidence)
        current, candidate_local, active_local, descriptors, feasibility = (
            np.asarray(value) for value in device_evidence
        )
        validation_seconds = time.perf_counter() - validation_start
        validation_digest = _scenario_digest(
            self._tape.sha256, model_version, validation.digest, *_VALIDATION_FOLD_NAMES
        )
        thresholds = HardValidationThresholds(
            minimum_current_margin=0.0,
            safe_policy_margin=0.0,
            local_non_regression_tolerance=self._config.validation_retention_tolerance,
            minimum_coverage=self._config.validation_minimum_coverage,
            minimum_redundancy=self._config.validation_minimum_redundancy,
            minimum_diversity=self._config.validation_minimum_diversity,
            minimum_feasible_fraction=1.0,
            maximum_runtime_seconds=self._config.validation_runtime_budget_seconds,
        )
        provisional_evidence = HardValidationEvidence(
            current_policy_margins=current,
            candidate_local_policy_margins=candidate_local,
            active_local_policy_margins=active_local,
            candidate_descriptors=descriptors,
            descriptor_scales=np.asarray(descriptor_scales),
            feasibility_margins=feasibility,
            runtime_seconds=np.asarray([0.0]),
            validation_set_digest=validation_digest,
        )
        # Measure one complete host gate/report/digest pass before binding the measured warm
        # runtime into the final report.  The second deterministic pass creates the authoritative
        # runtime gate; the equivalent provisional pass ensures report construction itself is not
        # silently omitted from the budget.
        validation_report_start = time.perf_counter()
        hard_validate_candidate(
            active, candidate, provisional_evidence, thresholds, current_model_version=model_version
        )
        validation_report_seconds = time.perf_counter() - validation_report_start
        excluded_compile_warmup_seconds = (
            compile_seconds + warmup_seconds + evidence_compile_seconds + evidence_warmup_seconds
        )
        admission_runtime_seconds = max(
            time.perf_counter() - admission_start - excluded_compile_warmup_seconds, 0.0
        )
        evidence = replace(
            provisional_evidence, runtime_seconds=np.asarray([admission_runtime_seconds])
        )
        report = hard_validate_candidate(
            active, candidate, evidence, thresholds, current_model_version=model_version
        )
        local_retention = np.asarray(report.candidate_local_best) - np.asarray(
            report.active_local_best
        )
        admission_margin = float(
            np.min(local_retention + self._config.validation_retention_tolerance)
        )
        with self._lock:
            self.diagnostics[candidate.digest] = {
                # Compatibility alias is execution-only; compilation and warmup are separate.
                "bptt_seconds": execution_seconds,
                "bptt_execution_seconds": execution_seconds,
                "bptt_setup_seconds": setup_seconds,
                "bptt_compile_seconds": compile_seconds,
                "bptt_warmup_seconds": warmup_seconds,
                "bptt_compiled_cache_hit": cache_hit,
                "bptt_cache_key": cache_key,
                "bptt_execution_backend": execution_backend,
                "bptt_execution_device_id": execution_device_id,
                "bptt_execution_device": execution_device_name,
                "bptt_compilation_excluded_from_execution_timing": True,
                "bptt_execution_scope": "compiled_burst_only",
                "bptt_execution_contract": BPTT_EXECUTION_CONTRACT,
                "bptt_input_digest": bptt_input_digest,
                "validation_seconds": validation_seconds,
                "validation_report_seconds": validation_report_seconds,
                "admission_runtime_seconds": admission_runtime_seconds,
                "admission_runtime_scope": ADMISSION_RUNTIME_SCOPE,
                "admission_publication_included": False,
                "admission_publication_accounting": ADMISSION_PUBLICATION_ACCOUNTING,
                "admission_excluded_compile_warmup_seconds": excluded_compile_warmup_seconds,
                "validation_compile_seconds": evidence_compile_seconds,
                "validation_warmup_seconds": evidence_warmup_seconds,
                "validation_compiled_cache_hit": evidence_cache_hit,
                "validation_cache_key": evidence_key,
                "validation_execution_device": execution_device_name,
                "validation_compilation_excluded_from_execution_timing": True,
                "validation_execution_synchronized": True,
                "gradient_norm": float(np.asarray(metrics.gradient_norm[-1])),
                "loss": float(np.asarray(metrics.loss.total[-1])),
                "update_accepted": bool(np.asarray(metrics.update_accepted[-1])),
                "report_passed": report.passed,
                "admission_margin": admission_margin,
                "failed_gates": list(report.failed_gate_names),
                "training_validation_disjoint": training_digest != validation.digest,
                "validation_fold_count": float(len(_VALIDATION_FOLD_NAMES)),
                "validation_scenario_count": float(validation.initial_states.shape[0]),
                "minimum_coverage_threshold": self._config.validation_minimum_coverage,
                "minimum_redundancy_threshold": self._config.validation_minimum_redundancy,
                "minimum_diversity_threshold": self._config.validation_minimum_diversity,
                "retention_tolerance": self._config.validation_retention_tolerance,
                "execution_device_is_cpu": execution_backend == "cpu",
                "execution_device_is_gpu": execution_backend == "gpu",
            }
            self.validation_material[candidate.digest] = CandidateValidationMaterial(
                proposal_active=active,
                context_step=start_index,
                candidate=candidate,
                evidence=evidence,
                thresholds=thresholds,
                report=report,
            )
        return candidate, report


def _offline_generic_checkpoint(
    condition: ConditionID, resources: ExperimentResources, config: ExperimentConfig
) -> tuple[PolicySnapshot, dict[str, Any]]:
    """Build one condition-level frozen checkpoint, independent of all evaluation folds."""
    training_tape = _predeclared_auxiliary_tape(condition, config, purpose="proposal-training")
    initialization_digest = hashlib.sha256(
        b"crazyflow.da_plcbf.offline-initialization.v1\0"
        + str(config.random_seed).encode()
        + b"\0"
        + condition.value.encode()
    ).digest()
    initialization_seed = int.from_bytes(initialization_digest[:4], "little")
    training_params = initialize_shared_actor(
        jax.random.key(initialization_seed),
        resources.spec,
        dimension=3,
        n_obstacles=(
            training_tape.static_positions.shape[0] + training_tape.dynamic_positions.shape[1]
        ),
        config=resources.actor_config,
    )
    initial_active = create_active_snapshot(
        training_params,
        version=0,
        model_version=0,
        structural_core=resources.spec,
        metadata={
            "initialization": "predeclared_condition_level_offline_checkpoint",
            "initialization_seed": initialization_seed,
        },
    )
    training_state = _initial_state(training_tape)
    initial_states, circles, training_indices = _offline_training_batch(training_tape, config)
    training_digest = _numeric_digest(
        "scenario-content",
        training_indices,
        initial_states,
        circles.obstacle_centers,
        circles.obstacle_radii,
        circles.obstacle_mask,
        training_tape.mass_scale[training_indices],
        training_tape.drag_scale[training_indices],
        training_tape.wind_velocity[training_indices],
        training_tape.rotor_efficiency[training_indices],
    )
    model = _controller_model(resources.model, _initialize_estimator(resources))
    functions = build_quad_generic_diversity_bptt_functions(
        resources.spec,
        model,
        resources.actuator,
        resources.actor_config,
        resources.quad_config,
        dt=config.dt,
        horizon=config.certificate_horizon,
        policy_gain=config.policy_gain,
        burst_steps=config.bptt_burst_steps,
    )
    targets = descriptor_targets_from_spec(resources.spec)
    scales = jnp.asarray([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=training_state.dtype)
    optimizer = functions.initialize(_tree_device(initial_active.params))
    start = time.perf_counter()
    trained, metrics = functions.burst(optimizer, initial_states, circles, targets, scales)
    _block((trained, metrics))
    elapsed = time.perf_counter() - start
    accepted = bool(np.all(np.asarray(metrics.update_accepted)))
    if not accepted:
        raise TrialExecutionError("offline generic-diversity BPTT rejected a non-finite update")
    trained_active = create_active_snapshot(
        trained.params,
        version=1,
        model_version=0,
        structural_core=resources.spec,
        metadata={
            "algorithm": "fixed_budget_truncated_bptt",
            "objective": "generic_diversity",
            "burst_steps": config.bptt_burst_steps,
            "training_tape_sha256": training_tape.sha256,
            "training_batch_digest": training_digest,
            "training_indices": training_indices.tolist(),
            "frozen_during_evaluation": True,
            "validation_gate_used": False,
            "condition": condition.value,
            "root_seed": config.random_seed,
        },
    )
    if trained_active.params_digest == initial_active.params_digest:
        raise TrialExecutionError("offline generic-diversity BPTT produced no parameter change")
    return trained_active, {
        "bptt_seconds": elapsed,
        "gradient_norm": float(np.asarray(metrics.gradient_norm[-1])),
        "loss": float(np.asarray(metrics.loss.total[-1])),
        "update_accepted": accepted,
        "training_tape_sha256": training_tape.sha256,
        "training_batch_digest": training_digest,
        "training_indices": training_indices.tolist(),
        "objective": "generic_diversity",
        "validation_gate_used": False,
    }


def _build_dashboard_evidence(
    trace: ImmutableTrace,
    tape: ScenarioTape,
    config: ExperimentConfig,
    events: Sequence[ArtifactEvent],
    nominal_rollout_positions: np.ndarray,
    nominal_rollout_available: np.ndarray,
    fallback_rollout_positions: np.ndarray,
    fallback_rollout_available: np.ndarray,
    selected_rollout_positions: np.ndarray,
    selected_rollout_available: np.ndarray,
    ghost_rollout_positions: np.ndarray,
    ghost_rollout_available: np.ndarray,
    normalized_descriptors: np.ndarray,
    descriptor_available: np.ndarray,
    dynamics_true: np.ndarray,
    dynamics_estimated: np.ndarray,
    dynamics_uncertainty: np.ndarray,
    dynamics_uncertainty_available: np.ndarray,
) -> DashboardEvidence:
    """Build a strict sidecar using only numerical evidence recorded by this trial.

    The runner records the held nominal preview, one explicit finite-scenario trajectory for every
    fallback, the worst-margin recorded sample for the selected fallback, endpoint sample ghosts,
    normalized training descriptors, prediction ensembles, physical truth/estimate/particles,
    admission decisions, and decomposed BPTT timings.  These are review evidence only; the trace's
    hard values and executed state remain authoritative.
    """
    steps = trace.steps
    prediction_nodes = config.certificate_horizon + 1
    prediction_time = np.arange(prediction_nodes, dtype=np.float64) * config.dt
    obstacle_samples = tape.prediction_samples
    obstacle_count = tape.dynamic_positions.shape[1]
    predictions = np.zeros(
        (steps, obstacle_samples, obstacle_count, prediction_nodes, 3), dtype=np.float64
    )
    prediction_available = np.zeros(
        (steps, obstacle_samples, obstacle_count, prediction_nodes), dtype=np.bool_
    )
    for step in range(steps):
        stop = step + prediction_nodes
        positions = np.transpose(tape.prediction_positions[:, step:stop], (0, 2, 1, 3))
        mask = np.transpose(tape.dynamic_time_mask[step:stop], (1, 0))
        observed_active = tape.dynamic_time_mask[step] & tape.dynamic_slot_mask
        available = np.broadcast_to(mask[None], positions.shape[:-1]) & np.broadcast_to(
            observed_active[None, :, None], positions.shape[:-1]
        )
        prediction_available[step] = available
        predictions[step] = np.where(available[..., None], positions, 0.0)

    (
        admission_recorded,
        candidate_present,
        candidate_admitted,
        candidate_rejected,
        admission_margin,
        reason_names,
        reason_index,
        timing_names,
        bptt_timing,
        bptt_available,
    ) = _admission_evidence_from_events(events, steps=steps)

    return DashboardEvidence(
        schema_version=np.asarray(DASHBOARD_EVIDENCE_SCHEMA_VERSION, dtype=np.uint16),
        trace_content_sha256=np.asarray(trace.content_sha256),
        scenario_tape_sha256=np.asarray(tape.sha256),
        policy_names=np.asarray(trace.policy_names),
        rollout_time=np.arange(config.certificate_horizon + 1, dtype=np.float64) * config.dt,
        nominal_rollout_positions=np.asarray(nominal_rollout_positions, dtype=np.float64),
        nominal_rollout_available=np.asarray(nominal_rollout_available, dtype=np.bool_),
        fallback_rollout_positions=np.asarray(fallback_rollout_positions, dtype=np.float64),
        fallback_rollout_available=np.asarray(fallback_rollout_available, dtype=np.bool_),
        selected_rollout_positions=np.asarray(selected_rollout_positions, dtype=np.float64),
        selected_rollout_available=np.asarray(selected_rollout_available, dtype=np.bool_),
        ghost_rollout_names=np.asarray(("selected_sample_0", "selected_sample_last")),
        ghost_rollout_positions=np.asarray(ghost_rollout_positions, dtype=np.float64),
        ghost_rollout_available=np.asarray(ghost_rollout_available, dtype=np.bool_),
        prediction_time=prediction_time,
        prediction_positions=predictions,
        prediction_available=prediction_available,
        descriptor_names=np.asarray(
            (
                "dx / 2m",
                "dy / 2m",
                "dz / 2m",
                "mean vx / 2m/s",
                "mean vy / 2m/s",
                "mean vz / 2m/s",
                "final vx / 3m/s",
                "final vy / 3m/s",
                "final vz / 3m/s",
            )
        ),
        normalized_descriptors=np.asarray(normalized_descriptors, dtype=np.float64),
        descriptor_available=np.asarray(descriptor_available, dtype=np.bool_),
        dynamics_parameter_names=np.asarray(_DYNAMICS_PARAMETER_NAMES),
        dynamics_true=np.asarray(dynamics_true, dtype=np.float64),
        dynamics_true_available=np.ones_like(dynamics_true, dtype=np.bool_),
        dynamics_estimated=np.asarray(dynamics_estimated, dtype=np.float64),
        dynamics_estimated_available=np.ones_like(dynamics_estimated, dtype=np.bool_),
        dynamics_uncertainty_samples=np.asarray(dynamics_uncertainty, dtype=np.float64),
        dynamics_uncertainty_available=np.asarray(dynamics_uncertainty_available, dtype=np.bool_),
        admission_recorded=admission_recorded,
        candidate_present=candidate_present,
        candidate_admitted=candidate_admitted,
        candidate_rejected=candidate_rejected,
        admission_margin=admission_margin,
        admission_reason_names=reason_names,
        admission_reason_index=reason_index,
        bptt_timing_names=timing_names,
        bptt_timing_seconds=bptt_timing,
        bptt_timing_available=bptt_available,
    )


def _used_online_adaptation_versions(
    events: Sequence[ArtifactEvent], trace: ImmutableTrace
) -> tuple[int, ...]:
    """Return admitted online snapshot versions proven to drive an executed control node."""
    used: list[int] = []
    for event in events:
        if event.category != "adaptation" or event.name != "candidate_admitted":
            continue
        version = event.details.get("published_snapshot_version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            continue
        indices = np.flatnonzero(
            trace.executed_control
            & (np.arange(trace.steps, dtype=np.int64) >= event.step)
            & (trace.snapshot_version == version)
        )
        if indices.size:
            used.append(version)
    return tuple(sorted(set(used)))


def _online_adaptation_lifecycle_blockers(events: Sequence[ArtifactEvent]) -> tuple[str, ...]:
    """Audit one complete post-startup proposal lifecycle without requiring admission.

    An adversarial or initially infeasible fold can correctly reject every candidate.  Requiring
    one admitted snapshot in *each* such fold would reward weakening the hard validation gate.
    Per-trial eligibility therefore requires an explicitly recorded scheduler, a post-startup
    candidate submission, and a non-exception hard admission decision.  Logical-simulation work is
    boundary-synchronized so host load cannot move the decision to a different simulated step;
    realtime-probe work remains isolated and asynchronous.  Actual use of an admitted snapshot is
    proved separately across the complete paired campaign.
    """
    scheduler_recorded = any(
        event.category == "adaptation"
        and event.name in {"online_execution_isolated", "logical_simulation_scheduler"}
        for event in events
    )
    submitted = any(
        event.category == "adaptation"
        and event.name == "candidate_submitted"
        and event.details.get("reason") == "submitted"
        for event in events
    )
    resolved = any(
        event.category == "adaptation"
        and event.name in {"candidate_admitted", "candidate_rejected"}
        and isinstance(event.details.get("job_id"), int)
        and not isinstance(event.details.get("job_id"), bool)
        and int(event.details["job_id"]) >= 0
        for event in events
    )
    failed = any(
        event.category == "adaptation" and event.name == "candidate_failed" for event in events
    )
    decisions = tuple(
        event
        for event in events
        if event.category in {"cold_start", "adaptation"}
        and event.name in {"candidate_admitted", "candidate_rejected", "candidate_expired"}
    )
    execution_bound = bool(decisions) and all(
        event.details.get("bptt_execution_backend") in {"gpu", "cpu"}
        and event.details.get("execution_device_is_gpu")
        is (event.details.get("bptt_execution_backend") == "gpu")
        and event.details.get("execution_device_is_cpu")
        is (event.details.get("bptt_execution_backend") == "cpu")
        and str(event.details.get("bptt_cache_key", "")).startswith("online:")
        and event.details.get("bptt_execution_contract") == BPTT_EXECUTION_CONTRACT
        for event in decisions
    )
    try:
        gpu_available = bool(jax.devices("gpu"))
    except (RuntimeError, ValueError):
        gpu_available = False
    gpu_used_when_available = not gpu_available or all(
        event.details.get("bptt_execution_backend") == "gpu" for event in decisions
    )
    blockers: list[str] = []
    if not scheduler_recorded:
        blockers.append("online adaptation execution mode was not recorded")
    if not submitted:
        blockers.append("no post-startup online candidate job was submitted")
    if not resolved:
        blockers.append("no post-startup online candidate reached a hard admission decision")
    if failed:
        blockers.append("an online candidate job ended in an execution failure")
    if not execution_bound:
        blockers.append("online adaptation execution device was not evidence-bound")
    if not gpu_used_when_available:
        blockers.append("online BPTT did not use the available GPU backend")
    return tuple(blockers)


def _campaign_online_snapshot_use(
    methods: Sequence[str], runs: Sequence[TrialRun], output_directory: str | Path | None
) -> frozenset[str]:
    """Return online methods with trace-bound proof that an admitted snapshot executed.

    Artifact-backed campaigns may be resumed with no newly executed ``TrialRun`` objects, so the
    persisted traces and events are also audited.  This keeps resume results byte-for-byte stable
    while retaining a stronger campaign-level proof than merely observing an admission event.
    """
    online = {
        MethodID.DA_PLCBF_NO_ONLINE_MODEL_ADAPTATION.value,
        MethodID.DA_PLCBF_FULL.value,
    }.intersection(methods)
    proven = {
        run.assignment.method
        for run in runs
        if run.assignment.method in online
        and _used_online_adaptation_versions(run.events, run.trace)
    }
    if output_directory is not None:
        root = Path(output_directory)
        for method in sorted(online - proven):
            for trace_path in sorted((root / "methods" / method).glob("*/*/trace.npz")):
                trace = load_trace(trace_path)
                events = load_events(trace_path.with_name("events.jsonl"), trace=trace)
                if _used_online_adaptation_versions(events, trace):
                    proven.add(method)
                    break
    return frozenset(proven)


def _plant_step(
    state: Array,
    command_motor: Array,
    model: VersionAModel,
    efficiency: Array,
    actuator: VersionAActuator,
    dt: float,
) -> tuple[Array, Array]:
    realized_motor = command_motor * efficiency
    wrench = motor_forces_to_wrench(
        realized_motor,
        L=actuator.arm_length,
        thrust2torque=actuator.thrust_to_torque,
        mixing_matrix=actuator.mixing_matrix,
    )
    return direct_wrench_symplectic_step(state, wrench, model, dt), realized_motor


def _controller_executable_key(
    method: MethodID,
    condition: ConditionID,
    config: ExperimentConfig,
    resources: ExperimentResources,
    policy_spec: SharedActorSpec,
) -> str:
    numeric = _numeric_digest(
        "controller-static-resources",
        *jax.tree.leaves(policy_spec),
        *jax.tree.leaves(resources.actuator),
    )
    return _scenario_digest(
        "controller-executable-v1",
        method.value,
        condition.value,
        config,
        resources.actor_config,
        resources.quad_config,
        resources.barrier_config,
        resources.version_a_filter_config,
        resources.dynamic_filter_config,
        resources.uncertainty_config,
        numeric,
    )


def _plant_executable_key(config: ExperimentConfig, resources: ExperimentResources) -> str:
    return _scenario_digest(
        "plant-executable-v1",
        config.dt,
        _numeric_digest("plant-actuator", *jax.tree.leaves(resources.actuator)),
    )


def _estimator_executable_key(config: ExperimentConfig, resources: ExperimentResources) -> str:
    device = _authoritative_estimator_device()
    return _scenario_digest(
        "authoritative-cpu-estimator-executable-v2",
        "authoritative-cpu-estimator-v1",
        f"{device.platform}:{device.id}",
        config.estimator_window_steps,
        resources.estimator_config,
    )


def _estimator_observations(
    history: Sequence[tuple[np.ndarray, ...]], window: int
) -> tuple[TranslationalObservations, RotorEfficiencyObservations]:
    entries = list(history[-window:])
    count = len(entries)
    rotation = np.zeros((window, 3, 3), dtype=np.float32)
    velocity = np.zeros((window, 3), dtype=np.float32)
    acceleration = np.zeros((window, 3), dtype=np.float32)
    collective = np.zeros((window,), dtype=np.float32)
    gravity = np.zeros((window, 3), dtype=np.float32)
    commanded = np.zeros((window, 4), dtype=np.float32)
    realized = np.zeros((window, 4), dtype=np.float32)
    mask = np.zeros((window,), dtype=bool)
    if count:
        values = tuple(np.stack([entry[index] for entry in entries]) for index in range(7))
        rotation[:count], velocity[:count], acceleration[:count], collective[:count] = values[:4]
        gravity[:count], commanded[:count], realized[:count] = values[4:]
        mask[:count] = True
    return (
        TranslationalObservations(
            jnp.asarray(rotation),
            jnp.asarray(velocity),
            jnp.asarray(acceleration),
            jnp.asarray(collective),
            jnp.asarray(gravity),
            jnp.asarray(mask),
        ),
        RotorEfficiencyObservations(
            jnp.asarray(commanded),
            jnp.asarray(realized),
            jnp.asarray(np.broadcast_to(mask[:, None], (window, 4))),
        ),
    )


def _authoritative_estimator_observations(
    history: Sequence[tuple[np.ndarray, ...]], window: int, *, device: jax.Device | None = None
) -> tuple[TranslationalObservations, RotorEfficiencyObservations]:
    """Materialize a fixed estimator window directly on authoritative CPU."""
    device = _authoritative_estimator_device() if device is None else device
    if str(device.platform) != "cpu":
        raise ValueError("authoritative estimator observations require a CPU device")
    with jax.default_device(device):
        observations = _estimator_observations(history, window)
    return jax.device_put(observations, device)


def _estimator_history_entry(
    state: Array,
    next_state: Array,
    commanded_motor: Array,
    realized_motor: Array,
    true_model: VersionAModel,
    actuator: VersionAActuator,
    tape: ScenarioTape,
    index: int,
    dt: float,
) -> tuple[np.ndarray, ...]:
    """Construct one estimator observation using only the tape's predeclared sensor noise.

    The noisy values affect the estimator, never the true plant or authoritative applied-control
    trace.  This makes estimator error a paired exogenous condition instead of an unlogged random
    draw.  The force sensor perturbation is applied before reconstructing collective force so the
    translational and per-rotor estimators consume one physically consistent measured vector.
    """
    device = _authoritative_estimator_device()
    state_host = np.asarray(state)
    next_state_host = np.asarray(next_state)
    commanded_host = np.asarray(commanded_motor)
    realized_host = np.asarray(realized_motor)
    measured_acceleration = (next_state_host[7:10] - state_host[7:10]) / dt + np.asarray(
        tape.estimator_acceleration_noise[index]
    )
    measured_motor = realized_host + np.asarray(tape.estimator_motor_force_noise[index])
    with jax.default_device(device):
        state_cpu = jax.device_put(jnp.asarray(state_host), device)
        actuator_cpu = jax.device_put(actuator, device)
        measured_wrench = motor_forces_to_wrench(
            jnp.asarray(measured_motor, dtype=state_cpu.dtype),
            L=actuator_cpu.arm_length,
            thrust2torque=actuator_cpu.thrust_to_torque,
            mixing_matrix=actuator_cpu.mixing_matrix,
        )
        rotation = quaternion_to_rotation_matrix(state_cpu[3:7])
    _block((measured_wrench, rotation))
    return (
        np.asarray(rotation),
        state_host[7:10],
        measured_acceleration,
        np.asarray(measured_wrench[0]),
        np.asarray(true_model.gravity_vec),
        commanded_host,
        measured_motor,
    )


class _RuntimeTapeInputs(NamedTuple):
    """Immutable tape-derived inputs materialized before the warm control loop.

    This is harness preparation, not controller work.  Every window still contains only the
    prediction horizon that would be supplied at its corresponding decision boundary; no value
    influences a maneuver before it is selected by that boundary's controller call.
    """

    target_positions: Array
    target_velocities: Array
    windows: tuple[DynamicSphereScenarioBatch, ...]
    circles: tuple[CircleScenarioBatch, ...]
    safety_sets: tuple[RigidBodySafetySet, ...]
    true_models: tuple[VersionAModel, ...]
    true_efficiencies: tuple[Array, ...]
    true_parameter_vectors: np.ndarray


def _prepare_runtime_tape_inputs(
    tape: ScenarioTape,
    condition: ConditionID,
    config: ExperimentConfig,
    resources: ExperimentResources,
    state: Array,
    *,
    needs_windows: bool,
    needs_safety_sets: bool,
    initial_window: DynamicSphereScenarioBatch,
    initial_circles: CircleScenarioBatch,
    initial_safety: RigidBodySafetySet,
) -> _RuntimeTapeInputs:
    """Materialize immutable scenario inputs once and place them on the controller device."""
    steps = config.control_steps
    device = state.device
    dtype = np.dtype(state.dtype)
    target_positions = _tree_to_device(
        np.asarray(tape.defender_reference_position[:steps], dtype=dtype), device
    )
    target_velocities = _tree_to_device(
        np.asarray(tape.defender_reference_velocity[:steps], dtype=dtype), device
    )

    if needs_windows:
        placed_window = _tree_to_device(initial_window, device)
        windows = (placed_window,) + tuple(
            _tree_to_device(
                dynamic_sphere_window_from_tape(
                    tape,
                    start_index=index,
                    horizon=config.certificate_horizon + 1,
                    speed_limit=config.speed_limit,
                    angular_rate_max=config.angular_rate_max,
                    tilt_max_radians=config.tilt_max_radians,
                ),
                device,
            )
            for index in range(1, steps)
        )
    else:
        placed_window = _tree_to_device(initial_window, device)
        windows = (placed_window,) * steps

    if needs_safety_sets:
        placed_circles = _tree_to_device(initial_circles, device)
        circle_values = (placed_circles,) + tuple(
            _tree_to_device(_circle_scenario_at(tape, index, config.speed_limit), device)
            for index in range(1, steps)
        )
        placed_safety = _tree_to_device(initial_safety, device)
        safety_sets = (placed_safety,) + tuple(
            _tree_to_device(_safety_from_circles(circles, config), device)
            for circles in circle_values[1:]
        )
    else:
        placed_circles = _tree_to_device(initial_circles, device)
        placed_safety = _tree_to_device(initial_safety, device)
        circle_values = (placed_circles,) * steps
        safety_sets = (placed_safety,) * steps

    true_pairs = tuple(
        _true_model(resources.model, tape, condition, index) for index in range(steps)
    )
    true_models = tuple(_tree_to_device(model, device) for model, _ in true_pairs)
    true_efficiencies = tuple(_tree_to_device(efficiency, device) for _, efficiency in true_pairs)
    true_parameter_vectors = np.stack(
        [
            np.asarray(_dynamics_parameter_vector(model, efficiency), dtype=np.float64)
            for model, efficiency in zip(true_models, true_efficiencies, strict=True)
        ]
    )
    return _RuntimeTapeInputs(
        target_positions=target_positions,
        target_velocities=target_velocities,
        windows=windows,
        circles=circle_values,
        safety_sets=safety_sets,
        true_models=true_models,
        true_efficiencies=true_efficiencies,
        true_parameter_vectors=true_parameter_vectors,
    )


def run_trial(
    assignment: TrialAssignment,
    tape: ScenarioTape,
    config: ExperimentConfig,
    resources: ExperimentResources | None = None,
    offline_checkpoint: tuple[PolicySnapshot, Mapping[str, Any]] | None = None,
    executable_cache: _CampaignExecutableCache | None = None,
) -> TrialRun:
    """Execute one real finite-horizon controller/plant trace.

    The supplied assignment determines only pairing/provenance; every exogenous trajectory and
    dynamics change comes from ``tape``.  Runtime method randomness is currently unnecessary, so
    its declared method-specific seed is recorded by the campaign but never allowed to perturb the
    shared tape.
    """
    config.validate()
    method = MethodID(assignment.method)
    condition = _condition(assignment.condition)
    if (
        str(tape.root_seed) != str(assignment.scenario_root_seed)
        or int(tape.generation_fold) != assignment.scenario_fold
    ):
        raise ValueError("scenario tape seed/fold does not match the paired assignment")
    obstacle_count = tape.static_positions.shape[0] + tape.dynamic_positions.shape[1]
    scheduled_resources = build_experiment_resources(
        config,
        obstacle_count=obstacle_count,
        initialization_seed=int(assignment.shared_stochastic_seed & 0xFFFFFFFF),
    )
    if resources is not None:
        scheduled_root = create_active_snapshot(
            scheduled_resources.initial_params,
            version=0,
            model_version=0,
            structural_core=scheduled_resources.spec,
            metadata={"initialization": "deterministic_structured_zero_residual"},
        )
        supplied_root = create_active_snapshot(
            resources.initial_params,
            version=0,
            model_version=0,
            structural_core=resources.spec,
            metadata={"initialization": "deterministic_structured_zero_residual"},
        )
        if supplied_root.digest != scheduled_root.digest:
            raise ValueError(
                "caller-supplied resources do not match the CPU-canonical scheduled policy root"
            )
    resolved = scheduled_resources if resources is None else resources
    resolved = _resources_for_tape(resolved, tape, config)
    if resolved.spec.base_codes.shape[0] != config.policy_count:
        raise ValueError("resource policy count does not match the trial configuration")

    state = _initial_state(tape)
    estimator_device = _authoritative_estimator_device()
    estimator = _initialize_authoritative_estimator(
        resolved.model, resolved.estimator_config, device=estimator_device
    )
    initial_active = create_active_snapshot(
        resolved.initial_params,
        version=0,
        model_version=0,
        structural_core=resolved.spec,
        metadata={"initialization": "deterministic_structured_zero_residual"},
    )
    online_library = method in {
        MethodID.DA_PLCBF_NO_ONLINE_MODEL_ADAPTATION,
        MethodID.DA_PLCBF_FULL,
    }
    adaptation_mode = AdaptationExecutionMode(config.adaptation_execution_mode)
    bptt_pool = None
    if online_library and executable_cache is not None:
        bptt_signature = _bptt_executable_signature(resolved, config)
        bptt_pool = executable_cache.bptt_pools.get(bptt_signature)
        if bptt_pool is None:
            bptt_pool = _build_bptt_executable_pool(resolved, config)
            executable_cache.bptt_pools[bptt_signature] = bptt_pool
    candidate_job = (
        _CandidateJob(tape, condition, resolved, config, bptt_pool) if online_library else None
    )
    cold_start_diagnostics: dict[str, Any] = {}
    offline_pretrain_succeeded: bool | None = None
    event_payloads: list[tuple[int, str, str, dict[str, Any]]] = []
    adaptation_decisions: list[AdaptationDecisionProof] = []

    initial_model, initial_samples = _authoritative_model_samples(
        resolved.model,
        estimator,
        resolved.estimator_config,
        config.uncertainty_sample_count,
        device=estimator_device,
    )
    initial_window = dynamic_sphere_window_from_tape(
        tape,
        start_index=0,
        horizon=config.certificate_horizon + 1,
        speed_limit=config.speed_limit,
        angular_rate_max=config.angular_rate_max,
        tilt_max_radians=config.tilt_max_radians,
    )
    active_for_store = initial_active
    if method is MethodID.OFFLINE_FROZEN_SDCBF_STYLE:
        checkpoint, raw_diagnostics = (
            _offline_generic_checkpoint(condition, resolved, config)
            if offline_checkpoint is None
            else offline_checkpoint
        )
        if checkpoint.structural_core_digest != initial_active.structural_core_digest:
            raise TrialExecutionError("offline checkpoint structural core is incompatible")
        if checkpoint.model_version != 0 or checkpoint.kind != "active":
            raise TrialExecutionError(
                "offline checkpoint must be an active model-version-0 snapshot"
            )
        training_tape_digest = str(checkpoint.metadata.get("training_tape_sha256", ""))
        if training_tape_digest == tape.sha256:
            raise TrialExecutionError("offline checkpoint training tape equals evaluation tape")
        active_for_store = checkpoint
        offline_diagnostics = dict(raw_diagnostics)
        offline_diagnostics.update(
            {
                "checkpoint_digest": checkpoint.digest,
                "evaluation_tape_sha256": tape.sha256,
                "training_evaluation_disjoint": training_tape_digest != tape.sha256,
                "reused_common_checkpoint": offline_checkpoint is not None,
            }
        )
        offline_pretrain_succeeded = True
        event_payloads.append(
            (0, "offline_pretraining", "generic_diversity_training_completed", offline_diagnostics)
        )
    store = ActiveSnapshotStore(active_for_store)
    worker: AdaptationWorker | None = None

    # Adaptive methods begin only after a synchronous proposal and hard admission.  This startup
    # work is outside the warm control loop and cannot race the first plant transition.
    if online_library:
        assert candidate_job is not None
        candidate_job.set_context(
            0, state, initial_model, initial_samples, initial_window, start_index=0
        )
        candidate, report = candidate_job(initial_active, 0)
        decision_active = store.active
        decision_model_version = store.model_version
        publication = store.admit(candidate, report)
        material = candidate_job.validation_material[candidate.digest]
        adaptation_decisions.append(
            AdaptationDecisionProof(
                phase="cold_start",
                job_id=-1,
                context_step=material.context_step,
                boundary_step=0,
                status="admitted" if publication.accepted else "rejected",
                decision_model_version=decision_model_version,
                publication_reason=publication.reason,
                used_by_executed_control=False,
                proposal_active=material.proposal_active,
                decision_active=decision_active,
                candidate=material.candidate,
                publication_active=publication.active,
                evidence=material.evidence,
                thresholds=material.thresholds,
                report=material.report,
            )
        )
        cold_start_diagnostics = dict(candidate_job.diagnostics.get(candidate.digest, {}))
        event_payloads.append(
            (
                0,
                "cold_start",
                "candidate_admitted" if publication.accepted else "candidate_rejected",
                {
                    "candidate_digest": candidate.digest,
                    "report_digest": report.digest,
                    "reason": publication.reason,
                    "published_snapshot_version": (
                        publication.active.version if publication.accepted else None
                    ),
                    "training_model_version": candidate.model_version,
                    "validation_model_version": report.model_version,
                    "decision_model_version": decision_model_version,
                    "failed_gates": list(report.failed_gate_names),
                    **cold_start_diagnostics,
                },
            )
        )
        # Rejection is a controller outcome, not a software execution failure.  The immutable
        # version-0 structural library remains active, and every runtime command still has to
        # pass the exact hard selector/filter/postcheck.  This is essential for intentionally
        # infeasible adversarial folds: they must produce an explicit degraded/failure trace
        # instead of disappearing from paired statistics.  No rejected candidate is executed.
        adaptation_device = candidate_job.adaptation_device
        controller_device = state.device
        shared_gpu_queue = (
            str(adaptation_device.platform) == "gpu"
            and str(controller_device.platform) == "gpu"
            and int(adaptation_device.id) == int(controller_device.id)
        )
        if adaptation_mode is AdaptationExecutionMode.REALTIME_PROBE:
            worker = AdaptationWorker(store, candidate_job)
            try:
                online_precompile = worker.prewarm(
                    lambda: candidate_job.precompile_online(
                        store.active,
                        state,
                        initial_model,
                        initial_samples,
                        initial_window,
                        start_index=0,
                    )
                )
            except Exception:
                worker.close(wait=True)
                raise
            event_payloads.append(
                (
                    0,
                    "adaptation",
                    "online_execution_isolated",
                    {
                        "execution_mode": adaptation_mode.value,
                        "compiled_execution_device": str(adaptation_device),
                        "compiled_execution_backend": str(adaptation_device.platform),
                        "gpu_jit_bptt": str(adaptation_device.platform) == "gpu",
                        "controller_thread_nonblocking": True,
                        "publication": "controller_boundary_only",
                        "controller_gpu_queue_shared": shared_gpu_queue,
                        "host_to_device_context_setup_may_synchronize": True,
                        "complete_cuda_queue_isolation_proven": False,
                        **online_precompile,
                    },
                )
            )
        else:
            online_precompile = candidate_job.precompile_online(
                store.active, state, initial_model, initial_samples, initial_window, start_index=0
            )
            event_payloads.append(
                (
                    0,
                    "adaptation",
                    "logical_simulation_scheduler",
                    {
                        "execution_mode": adaptation_mode.value,
                        "causal_context": "post_transition_state_and_history",
                        "publication": "next_control_boundary_only",
                        "host_load_can_change_event_step": False,
                        "real_time_claim_eligible": False,
                        "compiled_execution_device": str(adaptation_device),
                        "compiled_execution_backend": str(adaptation_device.platform),
                        "gpu_jit_bptt": str(adaptation_device.platform) == "gpu",
                        "controller_thread_nonblocking": False,
                        "controller_gpu_queue_shared": shared_gpu_queue,
                        "host_to_device_context_setup_may_synchronize": True,
                        "complete_cuda_queue_isolation_proven": False,
                        "gpu_adaptation_included_in_wall_step": (
                            str(adaptation_device.platform) == "gpu"
                        ),
                        "cpu_adaptation_included_in_wall_step": (
                            str(adaptation_device.platform) == "cpu"
                        ),
                        **online_precompile,
                    },
                )
            )
    # PCBF uses exactly one immutable fallback.  All other policy-library methods keep their
    # declared K slots.  The active snapshot remains the source for learned methods.
    if method is MethodID.FIXED_FALLBACK_PCBF:
        fixed_params, fixed_spec = slice_shared_actor_policy(
            resolved.initial_params, resolved.spec, jnp.asarray(0)
        )
    else:
        fixed_params, fixed_spec = resolved.initial_params, resolved.spec
    uses_active = method in {
        MethodID.OFFLINE_FROZEN_SDCBF_STYLE,
        MethodID.DA_PLCBF_NO_ONLINE_MODEL_ADAPTATION,
        MethodID.DA_PLCBF_FULL,
    }
    uses_uncertainty = method in {
        MethodID.DA_PLCBF_NO_ONLINE_MODEL_ADAPTATION,
        MethodID.DA_PLCBF_FULL,
    }
    uses_dynamic_filter = condition in {
        ConditionID.BALLISTIC_BALL,
        ConditionID.INTERCEPTOR_DRONE,
        ConditionID.FALSIFICATION_COMBINED,
    }

    policy_spec = resolved.spec if uses_active else fixed_spec
    policy_count = policy_spec.base_codes.shape[0] if method_spec(method).uses_policy_library else 1

    initial_circles = _circle_scenario_at(tape, 0, config.speed_limit)
    initial_safety = _safety_from_circles(initial_circles, config)
    previous_policy = jnp.asarray(-1, dtype=jnp.int32)
    wrench_weight = jnp.ones((4,), dtype=state.dtype)
    controller_device = state.device
    adaptation_model_current = initial_model
    adaptation_samples_current = initial_samples

    # Candidate snapshots may originate on the isolated CPU adaptation path.  Place each immutable
    # active snapshot on the controller device once per digest instead of relying on an implicit
    # transfer at every compiled-controller invocation.
    active_params_digest = store.active.digest if uses_active else None
    controller_params = _tree_to_device(
        store.active.params if uses_active else fixed_params, controller_device
    )
    fixed_params = _tree_to_device(fixed_params, controller_device)
    initial_model = _tree_to_device(initial_model, controller_device)
    initial_samples = _tree_to_device(initial_samples, controller_device)
    initial_window = _tree_to_device(initial_window, controller_device)
    initial_circles = _tree_to_device(initial_circles, controller_device)
    initial_safety = _tree_to_device(initial_safety, controller_device)
    previous_policy = _tree_to_device(previous_policy, controller_device)

    if method is MethodID.NOMINAL_ONLY:

        def controller(
            candidate_state: Array,
            target_position: Array,
            target_velocity: Array,
            params: SharedActorParams,
            model: VersionAModel,
            samples: VersionAModelSamples,
            window: Any,
            circles: CircleScenarioBatch,
            safety: RigidBodySafetySet,
            previous: Array,
        ) -> _ControlDiagnostics:
            del params, samples, window, circles, safety, previous
            nominal = waypoint_nominal_wrench(
                candidate_state,
                target_position,
                target_velocity,
                model,
                resolved.actuator,
                resolved.quad_config,
            )
            return _nominal_diagnostics(
                nominal,
                resolved.actuator,
                candidate_state,
                model,
                dt=config.dt,
                horizon=config.certificate_horizon,
            )

    elif method is MethodID.ANALYTIC_CBF_HOCBF:

        def controller(
            candidate_state: Array,
            target_position: Array,
            target_velocity: Array,
            params: SharedActorParams,
            model: VersionAModel,
            samples: VersionAModelSamples,
            window: Any,
            circles: CircleScenarioBatch,
            safety: RigidBodySafetySet,
            previous: Array,
        ) -> _ControlDiagnostics:
            del params, samples, circles, previous
            nominal = waypoint_nominal_wrench(
                candidate_state,
                target_position,
                target_velocity,
                model,
                resolved.actuator,
                resolved.quad_config,
            )
            result = version_a_analytic_runtime_step(
                candidate_state,
                nominal.wrench,
                wrench_weight,
                model,
                resolved.actuator,
                safety,
                resolved.barrier_config,
                resolved.version_a_filter_config,
                dt=config.dt,
                prediction_window=window if uses_dynamic_filter else None,
            )
            return _analytic_diagnostics(
                result,
                nominal,
                resolved.actuator,
                candidate_state,
                model,
                dt=config.dt,
                horizon=config.certificate_horizon,
            )

    elif uses_uncertainty:

        def controller(
            candidate_state: Array,
            target_position: Array,
            target_velocity: Array,
            params: SharedActorParams,
            model: VersionAModel,
            samples: VersionAModelSamples,
            window: Any,
            circles: CircleScenarioBatch,
            safety: RigidBodySafetySet,
            previous: Array,
        ) -> _ControlDiagnostics:
            del circles, safety
            result = uncertain_dynamic_discrete_runtime_step(
                candidate_state,
                target_position,
                target_velocity,
                previous,
                params,
                policy_spec,
                window,
                model,
                samples,
                resolved.actuator,
                resolved.actor_config,
                resolved.quad_config,
                resolved.barrier_config,
                resolved.dynamic_filter_config,
                dt=config.dt,
                policy_gain=config.policy_gain,
                uncertainty_config=resolved.uncertainty_config,
            )
            return _dynamic_diagnostics(
                result,
                resolved.actuator,
                candidate_state,
                model,
                dt=config.dt,
                horizon=config.certificate_horizon,
            )

    elif uses_dynamic_filter:

        def controller(
            candidate_state: Array,
            target_position: Array,
            target_velocity: Array,
            params: SharedActorParams,
            model: VersionAModel,
            samples: VersionAModelSamples,
            window: Any,
            circles: CircleScenarioBatch,
            safety: RigidBodySafetySet,
            previous: Array,
        ) -> _ControlDiagnostics:
            del samples, circles, safety
            result = dynamic_discrete_runtime_step(
                candidate_state,
                target_position,
                target_velocity,
                previous,
                params,
                policy_spec,
                window,
                model,
                resolved.actuator,
                resolved.actor_config,
                resolved.quad_config,
                resolved.barrier_config,
                resolved.dynamic_filter_config,
                dt=config.dt,
                policy_gain=config.policy_gain,
            )
            return _dynamic_diagnostics(
                result,
                resolved.actuator,
                candidate_state,
                model,
                dt=config.dt,
                horizon=config.certificate_horizon,
            )

    else:

        def controller(
            candidate_state: Array,
            target_position: Array,
            target_velocity: Array,
            params: SharedActorParams,
            model: VersionAModel,
            samples: VersionAModelSamples,
            window: Any,
            circles: CircleScenarioBatch,
            safety: RigidBodySafetySet,
            previous: Array,
        ) -> _ControlDiagnostics:
            del samples, window
            result = version_a_runtime_step(
                candidate_state,
                target_position,
                target_velocity,
                params,
                policy_spec,
                circles,
                safety,
                model,
                resolved.actuator,
                resolved.actor_config,
                resolved.quad_config,
                resolved.barrier_config,
                resolved.version_a_filter_config,
                dt=config.dt,
                certificate_horizon=config.certificate_horizon,
                policy_gain=config.policy_gain,
                previous_policy_index=previous,
            )
            return _version_a_diagnostics(
                result,
                resolved.actuator,
                candidate_state,
                model,
                dt=config.dt,
                horizon=config.certificate_horizon,
            )

    jitted_controller = jax.jit(controller)
    target_position_0 = jnp.asarray(tape.defender_reference_position[0], dtype=state.dtype)
    target_velocity_0 = jnp.asarray(tape.defender_reference_velocity[0], dtype=state.dtype)
    controller_args = (
        state,
        target_position_0,
        target_velocity_0,
        controller_params,
        initial_model,
        initial_samples,
        initial_window,
        initial_circles,
        initial_safety,
        previous_policy,
    )
    controller_key = _controller_executable_key(method, condition, config, resolved, policy_spec)
    compiled_controller = (
        executable_cache.controllers.get(controller_key) if executable_cache is not None else None
    )
    controller_cache_hit = compiled_controller is not None
    controller_compile = 0.0
    if compiled_controller is None:
        compile_start = time.perf_counter()
        compiled_controller = jitted_controller.lower(*controller_args).compile()
        controller_compile = time.perf_counter() - compile_start
        _block(compiled_controller(*controller_args))
        if executable_cache is not None:
            executable_cache.controllers[controller_key] = compiled_controller

    plant_function = jax.jit(
        lambda candidate_state, command, model, efficiency: _plant_step(
            candidate_state, command, model, efficiency, resolved.actuator, config.dt
        )
    )
    plant_args = (
        state,
        jnp.broadcast_to(jnp.asarray(resolved.actuator.thrust_min), (4,)),
        resolved.model,
        jnp.ones(4),
    )
    plant_key = _plant_executable_key(config, resolved)
    compiled_plant = (
        executable_cache.plants.get(plant_key) if executable_cache is not None else None
    )
    plant_cache_hit = compiled_plant is not None
    plant_compile = 0.0
    if compiled_plant is None:
        compile_start = time.perf_counter()
        compiled_plant = plant_function.lower(*plant_args).compile()
        plant_compile = time.perf_counter() - compile_start
        _block(compiled_plant(*plant_args))
        if executable_cache is not None:
            executable_cache.plants[plant_key] = compiled_plant

    estimator_history: list[tuple[np.ndarray, ...]] = []
    empty_observations = _authoritative_estimator_observations(
        estimator_history, config.estimator_window_steps, device=estimator_device
    )
    estimator_compile = 0.0
    compiled_estimator = None
    estimator_cache_hit = False
    if method is MethodID.DA_PLCBF_FULL:
        estimator_function = _authoritative_estimator_function(resolved.estimator_config)
        estimator_args = _authoritative_estimator_arguments(
            estimator, empty_observations, 0, device=estimator_device
        )
        estimator_key = _estimator_executable_key(config, resolved)
        compiled_estimator = (
            executable_cache.estimators.get(estimator_key) if executable_cache is not None else None
        )
        estimator_cache_hit = compiled_estimator is not None
        if compiled_estimator is None:
            compile_start = time.perf_counter()
            compiled_estimator = estimator_function.lower(*estimator_args).compile()
            estimator_compile = time.perf_counter() - compile_start
            _block(compiled_estimator(*estimator_args))
            if executable_cache is not None:
                executable_cache.estimators[estimator_key] = compiled_estimator

    steps = config.control_steps
    needs_runtime_windows = uses_uncertainty or uses_dynamic_filter
    needs_runtime_safety_sets = method is MethodID.ANALYTIC_CBF_HOCBF or (
        method is not MethodID.NOMINAL_ONLY and not uses_uncertainty and not uses_dynamic_filter
    )
    runtime_preparation_start = time.perf_counter()
    runtime_inputs = _prepare_runtime_tape_inputs(
        tape,
        condition,
        config,
        resolved,
        state,
        needs_windows=needs_runtime_windows,
        needs_safety_sets=needs_runtime_safety_sets,
        initial_window=initial_window,
        initial_circles=initial_circles,
        initial_safety=initial_safety,
    )
    _block(runtime_inputs)
    runtime_preparation_seconds = time.perf_counter() - runtime_preparation_start

    # The estimator is immutable between scheduled updates.  Its physical model and deterministic
    # particles are consequently identical between those boundaries, so construct and transfer
    # them only when the estimator changes.
    controller_model_current = initial_model
    model_samples_current = initial_samples
    estimated_physical_current = physical_parameters(estimator)
    estimated_parameter_vector_current = np.asarray(
        _dynamics_parameter_vector(
            adaptation_model_current, estimated_physical_current.rotor_efficiency
        ),
        dtype=np.float64,
    )
    uncertainty_vectors_current = np.asarray(
        _sampled_dynamics_parameter_vectors(adaptation_samples_current), dtype=np.float64
    )
    uncertainty_valid_current = np.asarray(adaptation_samples_current.sample_valid, dtype=np.bool_)

    event_payloads.append(
        (
            0,
            "runtime",
            "compile_cache_accounting",
            {
                "controller_cache_hit": controller_cache_hit,
                "plant_cache_hit": plant_cache_hit,
                "estimator_cache_hit": estimator_cache_hit,
                "bptt_startup_cache_hit": bool(
                    online_library and cold_start_diagnostics.get("bptt_compiled_cache_hit", False)
                ),
                "validation_startup_cache_hit": bool(
                    online_library
                    and cold_start_diagnostics.get("validation_compiled_cache_hit", False)
                ),
                "new_controller_compile_seconds": controller_compile,
                "new_plant_compile_seconds": plant_compile,
                "new_estimator_compile_seconds": estimator_compile,
                "warm_execution_excludes_compilation": True,
            },
        )
    )
    event_payloads.append(
        (
            0,
            "runtime",
            "runtime_inputs_precomputed",
            {
                "execution_seconds": runtime_preparation_seconds,
                "decision_boundaries": steps,
                "prediction_windows_materialized": (steps if needs_runtime_windows else 1),
                "safety_sets_materialized": steps if needs_runtime_safety_sets else 1,
                "target_trajectory_materialized": True,
                "true_dynamics_schedule_materialized": True,
                "controller_device": str(controller_device),
                "estimator_device": str(estimator_device),
                "excluded_from_warm_step_latency": True,
                "prediction_horizon_only_at_each_boundary": True,
                "dynamic_prediction_contract": DYNAMIC_PREDICTION_CONTRACT,
                "forecast_source": "predeclared_exogenous_oracle",
                "unobserved_dynamic_slots_masked_for_entire_horizon": True,
            },
        )
    )

    states = np.zeros((steps, 13), dtype=np.float64)
    nominal_controls = np.zeros((steps, 4), dtype=np.float64)
    filtered_controls = np.zeros((steps, 4), dtype=np.float64)
    applied_controls = np.zeros((steps, 4), dtype=np.float64)
    policy_values = np.zeros((steps, policy_count if policy_count > 0 else 1), dtype=np.float64)
    training_values = np.zeros_like(policy_values)
    selected_policies = np.full((steps,), -1, dtype=np.int32)
    snapshot_versions = np.zeros((steps,), dtype=np.int32)
    model_versions = np.zeros((steps,), dtype=np.int32)
    kkt = np.zeros((steps,), dtype=np.float64)
    postcheck = np.zeros((steps,), dtype=np.float64)
    clipped = np.zeros((steps,), dtype=bool)
    saturated = np.zeros((steps,), dtype=bool)
    degraded = np.zeros((steps,), dtype=bool)
    rollout_nodes = config.certificate_horizon + 1
    nominal_rollout_positions = np.zeros((steps, rollout_nodes, 3), dtype=np.float64)
    nominal_rollout_available = np.zeros((steps,), dtype=np.bool_)
    fallback_rollout_positions = np.zeros((steps, policy_count, rollout_nodes, 3), dtype=np.float64)
    fallback_rollout_available = np.zeros((steps, policy_count), dtype=np.bool_)
    selected_rollout_positions = np.zeros((steps, rollout_nodes, 3), dtype=np.float64)
    selected_rollout_available = np.zeros((steps,), dtype=np.bool_)
    normalized_descriptors = np.zeros((steps, policy_count, 9), dtype=np.float64)
    descriptor_available = np.zeros((steps, policy_count), dtype=np.bool_)
    ghost_rollout_positions = np.zeros((steps, 2, rollout_nodes, 3), dtype=np.float64)
    ghost_rollout_available = np.zeros((steps, 2), dtype=np.bool_)
    losses = np.zeros((steps, 6), dtype=np.float64)
    gradient_norms = np.zeros((steps,), dtype=np.float64)
    # command_preparation and postprocessing are disjoint host/orchestration intervals. wall_step
    # overlaps every warm component and ends before optional real-time pacing sleep.
    latencies = np.zeros((steps, 7), dtype=np.float64)
    pacing_sleep_seconds = np.zeros((steps,), dtype=np.float64)
    release_lateness_seconds = np.zeros((steps,), dtype=np.float64)
    estimation_error = np.zeros((steps, 11), dtype=np.float64)
    dynamics_true_parameters = np.zeros((steps, len(_DYNAMICS_PARAMETER_NAMES)), dtype=np.float64)
    dynamics_estimated_parameters = np.zeros_like(dynamics_true_parameters)
    dynamics_uncertainty = np.zeros(
        (steps, config.uncertainty_sample_count, len(_DYNAMICS_PARAMETER_NAMES)), dtype=np.float64
    )
    dynamics_uncertainty_available = np.zeros(
        (steps, config.uncertainty_sample_count), dtype=np.bool_
    )
    certified = np.zeros_like(policy_values, dtype=bool)
    last_outcome_job = -1
    next_logical_job_id = 0
    last_candidate_loss = 0.0
    last_gradient = 0.0

    try:
        # There are ``steps`` saved state nodes and exactly ``steps-1`` executed controls.  The
        # final row is an explicit terminal observation with zero no-command sentinels, so every
        # executed transition and its swept interval is present in the immutable trace.
        for index in range(steps - 1):
            wall_step_start = time.perf_counter()
            command_preparation_start = wall_step_start

            if worker is not None:
                outcome = worker.publish_at_boundary()
                if outcome is not None and outcome.job_id != last_outcome_job:
                    last_outcome_job = outcome.job_id
                    assert candidate_job is not None
                    candidate_diagnostics = dict(
                        candidate_job.diagnostics.get(outcome.candidate_digest, {})
                    )
                    model_lineage_details: dict[str, int] = {}
                    if outcome.status in {AdaptationStatus.ADMITTED, AdaptationStatus.REJECTED}:
                        if outcome.publication is None:
                            raise TrialExecutionError(
                                "completed adaptation outcome omitted its publication result"
                            )
                        material = candidate_job.validation_material.get(outcome.candidate_digest)
                        if material is None:
                            raise TrialExecutionError(
                                "completed adaptation outcome omitted validation material"
                            )
                        decision_active = (
                            material.proposal_active
                            if outcome.publication.accepted
                            else outcome.publication.active
                        )
                        decision_model_version = store.model_version
                        adaptation_decisions.append(
                            AdaptationDecisionProof(
                                phase="online",
                                job_id=outcome.job_id,
                                context_step=material.context_step,
                                boundary_step=index,
                                status=(
                                    "admitted"
                                    if outcome.status is AdaptationStatus.ADMITTED
                                    else "rejected"
                                ),
                                decision_model_version=decision_model_version,
                                publication_reason=outcome.publication.reason,
                                used_by_executed_control=False,
                                proposal_active=material.proposal_active,
                                decision_active=decision_active,
                                candidate=material.candidate,
                                publication_active=outcome.publication.active,
                                evidence=material.evidence,
                                thresholds=material.thresholds,
                                report=material.report,
                            )
                        )
                        model_lineage_details = {
                            "training_model_version": material.candidate.model_version,
                            "validation_model_version": material.report.model_version,
                            "decision_model_version": decision_model_version,
                        }
                    if candidate_diagnostics:
                        last_candidate_loss = float(candidate_diagnostics["loss"])
                        last_gradient = float(candidate_diagnostics["gradient_norm"])
                    event_payloads.append(
                        (
                            index,
                            "adaptation",
                            f"candidate_{outcome.status.value}",
                            {
                                "job_id": outcome.job_id,
                                "candidate_digest": outcome.candidate_digest,
                                "report_digest": outcome.report_digest,
                                "reason": (
                                    outcome.publication.reason
                                    if outcome.publication is not None
                                    else outcome.error_type
                                ),
                                "published_snapshot_version": (
                                    outcome.publication.active.version
                                    if outcome.publication is not None
                                    and outcome.publication.accepted
                                    else None
                                ),
                                "publication_boundary": index,
                                **model_lineage_details,
                                **candidate_diagnostics,
                            },
                        )
                    )

            active_snapshot = store.active if uses_active else None
            if active_snapshot is not None and active_snapshot.digest != active_params_digest:
                controller_params = _tree_to_device(active_snapshot.params, controller_device)
                _block(controller_params)
                active_params_digest = active_snapshot.digest
            params = controller_params if active_snapshot is not None else fixed_params
            snapshot_version = active_snapshot.version if active_snapshot is not None else 0
            current_model = controller_model_current
            samples = model_samples_current
            window = runtime_inputs.windows[index]
            circles = runtime_inputs.circles[index]
            safety = runtime_inputs.safety_sets[index]

            model_version = int(np.asarray(estimator.model_version))
            states[index] = np.asarray(state)
            snapshot_versions[index] = snapshot_version
            model_versions[index] = model_version

            true_model = runtime_inputs.true_models[index]
            true_efficiency = runtime_inputs.true_efficiencies[index]
            dynamics_true_parameters[index] = runtime_inputs.true_parameter_vectors[index]
            dynamics_estimated_parameters[index] = estimated_parameter_vector_current
            if uses_uncertainty:
                dynamics_uncertainty[index] = uncertainty_vectors_current
                dynamics_uncertainty_available[index] = uncertainty_valid_current
            estimation_error[index] = (
                estimated_parameter_vector_current - runtime_inputs.true_parameter_vectors[index]
            )
            losses[index, :4] = (
                abs(estimation_error[index, 0]),
                np.linalg.norm(estimation_error[index, 1:4]),
                np.linalg.norm(estimation_error[index, 4:7]),
                np.linalg.norm(estimation_error[index, 7:11]),
            )
            losses[index, 4] = last_candidate_loss
            gradient_norms[index] = last_gradient
            target_position = runtime_inputs.target_positions[index]
            target_velocity = runtime_inputs.target_velocities[index]
            latencies[index, 3] = time.perf_counter() - command_preparation_start

            control_start = time.perf_counter()
            diagnostics = compiled_controller(
                state,
                target_position,
                target_velocity,
                params,
                current_model,
                samples,
                window,
                circles,
                safety,
                previous_policy,
            )
            _block(diagnostics)
            latencies[index, 0] = time.perf_counter() - control_start
            command_motor = diagnostics.motor_command
            if not np.all(np.isfinite(np.asarray(command_motor))):
                raise TrialExecutionError("controller returned a non-finite motor command")
            # A command is ready for the plant only after synchronized controller execution and
            # the mandatory host-side finite-value guard both pass.  This contiguous interval also
            # captures orchestration gaps that a sum of separately timed components would omit.
            latencies[index, 6] = time.perf_counter() - command_preparation_start

            plant_start = time.perf_counter()
            next_state, realized_motor = compiled_plant(
                state, command_motor, true_model, true_efficiency
            )
            _block((next_state, realized_motor))
            latencies[index, 1] = time.perf_counter() - plant_start
            if not np.all(np.isfinite(np.asarray(next_state))):
                raise TrialExecutionError("true airborne plant produced a non-finite state")

            nominal_controls[index] = np.asarray(diagnostics.nominal_motor)
            filtered_controls[index] = np.asarray(command_motor)
            applied_controls[index] = np.asarray(realized_motor)
            raw_policy, raw_training = _finite_policy_evidence(
                diagnostics.policy_values, diagnostics.training_values
            )
            policy_values[index] = raw_policy
            training_values[index] = raw_training
            if method_spec(method).uses_policy_library:
                certified[index] = np.isfinite(raw_policy) & (raw_policy >= 0.0)
            selected_policies[index] = int(np.asarray(diagnostics.selected_policy))
            previous_policy = diagnostics.selected_policy
            kkt[index] = float(np.asarray(diagnostics.kkt_residual))
            postcheck[index] = float(np.asarray(diagnostics.postcheck_residual))
            clipped[index] = bool(np.asarray(diagnostics.clipped))
            saturated[index] = bool(np.asarray(diagnostics.saturated))
            degraded[index] = bool(np.asarray(diagnostics.degraded))
            losses[index, 5] = float(np.count_nonzero(certified[index]))

            nominal_preview = np.asarray(diagnostics.nominal_rollout_positions, dtype=np.float64)
            nominal_preview_valid = bool(np.all(np.isfinite(nominal_preview)))
            nominal_rollout_available[index] = nominal_preview_valid
            if nominal_preview_valid:
                nominal_rollout_positions[index] = nominal_preview
            fallback_mask = np.asarray(diagnostics.fallback_rollout_available, dtype=np.bool_)
            fallback_values = np.asarray(diagnostics.fallback_rollout_positions, dtype=np.float64)
            fallback_rollout_available[index] = fallback_mask
            fallback_rollout_positions[index] = np.where(
                fallback_mask[:, None, None], fallback_values, 0.0
            )
            selected_available = bool(np.asarray(diagnostics.selected_rollout_available))
            selected_values = np.asarray(diagnostics.selected_rollout_positions, dtype=np.float64)
            selected_rollout_available[index] = selected_available
            if selected_available:
                selected_rollout_positions[index] = selected_values
            descriptor_mask = np.asarray(diagnostics.descriptor_available, dtype=np.bool_)
            descriptor_values = np.asarray(diagnostics.normalized_descriptors, dtype=np.float64)
            descriptor_available[index] = descriptor_mask
            normalized_descriptors[index] = np.where(
                descriptor_mask[:, None], descriptor_values, 0.0
            )
            ghost_mask = np.asarray(diagnostics.ghost_rollout_available, dtype=np.bool_)
            ghost_values = np.asarray(diagnostics.ghost_rollout_positions, dtype=np.float64)
            ghost_rollout_available[index] = ghost_mask
            ghost_rollout_positions[index] = np.where(ghost_mask[:, None, None], ghost_values, 0.0)

            estimator_history.append(
                _estimator_history_entry(
                    state,
                    next_state,
                    command_motor,
                    realized_motor,
                    true_model,
                    resolved.actuator,
                    tape,
                    index,
                    config.dt,
                )
            )

            if (
                compiled_estimator is not None
                and (index + 1) % config.estimator_interval_steps == 0
            ):
                observations = _authoritative_estimator_observations(
                    estimator_history, config.estimator_window_steps, device=estimator_device
                )
                estimator_arguments = _authoritative_estimator_arguments(
                    estimator, observations, index, device=estimator_device
                )
                estimator_start = time.perf_counter()
                translation_update, rotor_update = compiled_estimator(*estimator_arguments)
                _block((translation_update, rotor_update))
                latencies[index, 2] = time.perf_counter() - estimator_start
                previous_version = int(np.asarray(estimator.model_version))
                estimator = translation_update.state
                next_version = int(np.asarray(estimator.model_version))
                event_payloads.append(
                    (
                        index,
                        "estimator",
                        "estimator_update_executed",
                        {
                            "execution_seconds": latencies[index, 2],
                            "translation_status": int(np.asarray(translation_update.status)),
                            "rotor_status": int(np.asarray(rotor_update.status)),
                            "previous_version": previous_version,
                            "next_version": next_version,
                            "event_only_latency_sample": True,
                        },
                    )
                )
                if next_version > previous_version:
                    store.advance_model_version(next_version)
                    event_payloads.append(
                        (
                            index,
                            "estimator",
                            "model_version_advanced",
                            {
                                "previous_version": previous_version,
                                "next_version": next_version,
                                "translation_status": int(np.asarray(translation_update.status)),
                                "rotor_status": int(np.asarray(rotor_update.status)),
                            },
                        )
                    )
                adaptation_model_current, adaptation_samples_current = _authoritative_model_samples(
                    resolved.model,
                    estimator,
                    resolved.estimator_config,
                    config.uncertainty_sample_count,
                    device=estimator_device,
                )
                controller_model_current = _tree_to_device(
                    adaptation_model_current, controller_device
                )
                model_samples_current = _tree_to_device(
                    adaptation_samples_current, controller_device
                )
                _block((controller_model_current, model_samples_current))
                estimated_physical_current = physical_parameters(estimator)
                estimated_parameter_vector_current = np.asarray(
                    _dynamics_parameter_vector(
                        adaptation_model_current, estimated_physical_current.rotor_efficiency
                    ),
                    dtype=np.float64,
                )
                uncertainty_vectors_current = np.asarray(
                    _sampled_dynamics_parameter_vectors(adaptation_samples_current),
                    dtype=np.float64,
                )
                uncertainty_valid_current = np.asarray(
                    adaptation_samples_current.sample_valid, dtype=np.bool_
                )

            if online_library and index % config.adaptation_interval_steps == 0:
                assert candidate_job is not None
                publication_boundary = index + 1
                # A candidate may only be submitted if at least one later control can execute it.
                # The terminal observation is deliberately not treated as a publication boundary.
                if publication_boundary >= steps - 1:
                    event_payloads.append(
                        (
                            publication_boundary,
                            "adaptation",
                            "candidate_not_submitted",
                            {
                                "job_id": -1,
                                "reason": "no_future_executed_control",
                                "execution_mode": adaptation_mode.value,
                            },
                        )
                    )
                elif adaptation_mode is AdaptationExecutionMode.LOGICAL_SIMULATION:
                    current_version = store.model_version
                    candidate_job.set_context(
                        current_version,
                        next_state,
                        adaptation_model_current,
                        adaptation_samples_current,
                        runtime_inputs.windows[publication_boundary],
                        start_index=publication_boundary,
                    )
                    job_id = next_logical_job_id
                    next_logical_job_id += 1
                    event_payloads.append(
                        (
                            publication_boundary,
                            "adaptation",
                            "candidate_submitted",
                            {
                                "job_id": job_id,
                                "reason": "submitted",
                                "execution_mode": adaptation_mode.value,
                                "causal_state_step": publication_boundary,
                            },
                        )
                    )
                    try:
                        decision_active = store.active
                        decision_model_version = store.model_version
                        candidate, report = candidate_job(decision_active, current_version)
                        publication = store.admit(candidate, report)
                    except Exception as error:
                        event_payloads.append(
                            (
                                publication_boundary,
                                "adaptation",
                                "candidate_failed",
                                {
                                    "job_id": job_id,
                                    "reason": type(error).__name__,
                                    "error_message": str(error),
                                    "execution_mode": adaptation_mode.value,
                                    "publication_boundary": publication_boundary,
                                },
                            )
                        )
                    else:
                        material = candidate_job.validation_material[candidate.digest]
                        adaptation_decisions.append(
                            AdaptationDecisionProof(
                                phase="online",
                                job_id=job_id,
                                context_step=material.context_step,
                                boundary_step=publication_boundary,
                                status="admitted" if publication.accepted else "rejected",
                                decision_model_version=decision_model_version,
                                publication_reason=publication.reason,
                                used_by_executed_control=False,
                                proposal_active=material.proposal_active,
                                decision_active=decision_active,
                                candidate=material.candidate,
                                publication_active=publication.active,
                                evidence=material.evidence,
                                thresholds=material.thresholds,
                                report=material.report,
                            )
                        )
                        candidate_diagnostics = dict(
                            candidate_job.diagnostics.get(candidate.digest, {})
                        )
                        if candidate_diagnostics:
                            last_candidate_loss = float(candidate_diagnostics["loss"])
                            last_gradient = float(candidate_diagnostics["gradient_norm"])
                        event_payloads.append(
                            (
                                publication_boundary,
                                "adaptation",
                                "candidate_admitted"
                                if publication.accepted
                                else "candidate_rejected",
                                {
                                    "job_id": job_id,
                                    "candidate_digest": candidate.digest,
                                    "report_digest": report.digest,
                                    "reason": publication.reason,
                                    "published_snapshot_version": (
                                        publication.active.version if publication.accepted else None
                                    ),
                                    "publication_boundary": publication_boundary,
                                    "execution_mode": adaptation_mode.value,
                                    "training_model_version": current_version,
                                    "validation_model_version": report.model_version,
                                    "decision_model_version": decision_model_version,
                                    **candidate_diagnostics,
                                },
                            )
                        )
                else:
                    assert worker is not None
                    if not worker.in_flight and not worker.candidate_ready:
                        current_version = store.model_version
                        candidate_job.set_context(
                            current_version,
                            next_state,
                            adaptation_model_current,
                            adaptation_samples_current,
                            runtime_inputs.windows[publication_boundary],
                            start_index=publication_boundary,
                        )
                    submission = worker.submit()
                    event_payloads.append(
                        (
                            publication_boundary,
                            "adaptation",
                            "candidate_submitted"
                            if submission.submitted
                            else "candidate_not_submitted",
                            {
                                "job_id": submission.job_id,
                                "reason": submission.reason,
                                "execution_mode": adaptation_mode.value,
                            },
                        )
                    )
            state = next_state
            wall_step_seconds = time.perf_counter() - wall_step_start
            timed_work_seconds = float(np.sum(latencies[index, :4]))
            latencies[index, 4] = max(wall_step_seconds - timed_work_seconds, 0.0)
            latencies[index, 5] = wall_step_seconds
            if config.realtime_pacing:
                remaining = config.dt - wall_step_seconds
                if remaining > 0.0:
                    sleep_start = time.perf_counter()
                    time.sleep(remaining)
                    pacing_sleep_seconds[index] = time.perf_counter() - sleep_start
                release_lateness_seconds[index] = max(
                    time.perf_counter() - wall_step_start - config.dt, 0.0
                )

        event_payloads.append(
            (
                steps - 1,
                "runtime",
                "warm_step_timing_semantics",
                {
                    "wall_step_excludes_compilation": True,
                    "wall_step_excludes_pacing_sleep": True,
                    "wall_step_includes_command_preparation": True,
                    "wall_step_includes_controller": True,
                    "wall_step_includes_plant": True,
                    "wall_step_includes_estimator_updates": True,
                    "wall_step_includes_postprocessing_and_adaptation_submission": True,
                    "command_ready_includes_preparation_controller_and_finite_guard": True,
                    "command_ready_recorded_as_contiguous_per_step_interval": True,
                    "estimator_tick_work_zeros_mean_no_scheduled_update": True,
                    "estimator_event_only_samples_emitted": True,
                    "realtime_pacing_enabled": config.realtime_pacing,
                    "pacing_sleep_median_seconds": float(
                        np.median(pacing_sleep_seconds[: steps - 1])
                    ),
                    "pacing_sleep_worst_seconds": float(np.max(pacing_sleep_seconds[: steps - 1])),
                    "release_lateness_worst_seconds": float(
                        np.max(release_lateness_seconds[: steps - 1])
                    ),
                    "release_lateness_steps": int(
                        np.count_nonzero(release_lateness_seconds[: steps - 1] > 0.0)
                    ),
                },
            )
        )

        # Cleanup may wait outside the measured trace so one trial's CPU job cannot overlap the
        # next paired trial.  The terminal observation has no future control, however, so a staged
        # candidate is expired rather than published into a snapshot that never drove the plant.
        if worker is not None:
            final_outcome = worker.expire_at_terminal(
                timeout=config.validation_runtime_budget_seconds * 2.0
            )
            if final_outcome is not None and final_outcome.job_id != last_outcome_job:
                last_outcome_job = final_outcome.job_id
                assert candidate_job is not None
                candidate_diagnostics = dict(
                    candidate_job.diagnostics.get(final_outcome.candidate_digest, {})
                )
                model_lineage_details = {}
                if final_outcome.status is AdaptationStatus.EXPIRED:
                    material = candidate_job.validation_material.get(final_outcome.candidate_digest)
                    if material is None:
                        raise TrialExecutionError(
                            "expired adaptation outcome omitted validation material"
                        )
                    terminal_reason = (
                        final_outcome.error_type
                        or final_outcome.error_message
                        or "terminal_boundary_has_no_future_control"
                    )
                    decision_model_version = store.model_version
                    adaptation_decisions.append(
                        AdaptationDecisionProof(
                            phase="online",
                            job_id=final_outcome.job_id,
                            context_step=material.context_step,
                            boundary_step=steps - 1,
                            status="expired",
                            decision_model_version=decision_model_version,
                            publication_reason=terminal_reason,
                            used_by_executed_control=False,
                            proposal_active=material.proposal_active,
                            decision_active=store.active,
                            candidate=material.candidate,
                            publication_active=store.active,
                            evidence=material.evidence,
                            thresholds=material.thresholds,
                            report=material.report,
                        )
                    )
                    model_lineage_details = {
                        "training_model_version": material.candidate.model_version,
                        "validation_model_version": material.report.model_version,
                        "decision_model_version": decision_model_version,
                    }
                event_payloads.append(
                    (
                        steps - 1,
                        "adaptation",
                        f"candidate_{final_outcome.status.value}",
                        {
                            "job_id": final_outcome.job_id,
                            "candidate_digest": final_outcome.candidate_digest,
                            "report_digest": final_outcome.report_digest,
                            "reason": (
                                final_outcome.publication.reason
                                if final_outcome.publication is not None
                                else final_outcome.error_type or final_outcome.error_message
                            ),
                            "published_snapshot_version": (
                                final_outcome.publication.active.version
                                if final_outcome.publication is not None
                                and final_outcome.publication.accepted
                                else None
                            ),
                            "publication_boundary": steps - 1,
                            "terminal_only_publication": False,
                            "terminal_result_expired_without_publication": (
                                final_outcome.status is AdaptationStatus.EXPIRED
                            ),
                            "used_by_executed_control": False,
                            **model_lineage_details,
                            **candidate_diagnostics,
                        },
                    )
                )

        terminal_active = store.active if uses_active else None
        states[-1] = np.asarray(state)
        snapshot_versions[-1] = terminal_active.version if terminal_active is not None else 0
        model_versions[-1] = int(np.asarray(estimator.model_version))
        dynamics_true_parameters[-1] = runtime_inputs.true_parameter_vectors[-1]
        dynamics_estimated_parameters[-1] = estimated_parameter_vector_current
        if uses_uncertainty:
            dynamics_uncertainty[-1] = uncertainty_vectors_current
            dynamics_uncertainty_available[-1] = uncertainty_valid_current
        estimation_error[-1] = (
            estimated_parameter_vector_current - runtime_inputs.true_parameter_vectors[-1]
        )
        losses[-1, :4] = (
            abs(estimation_error[-1, 0]),
            np.linalg.norm(estimation_error[-1, 1:4]),
            np.linalg.norm(estimation_error[-1, 4:7]),
            np.linalg.norm(estimation_error[-1, 7:11]),
        )
        # The terminal row has no command.  Candidate diagnostics live in the adaptation event;
        # retaining them here would violate the terminal no-control sentinel contract.
        event_payloads.append(
            (
                steps - 1,
                "runtime",
                "terminal_observation",
                {
                    "executed_control_count": steps - 1,
                    "terminal_control_row": "zero_no_command_sentinel",
                },
            )
        )
    finally:
        if worker is not None:
            worker.close(wait=True)

    barriers, contact, failure = _barrier_trace(states, tape, config)
    degraded |= failure
    time_nodes = np.asarray(tape.time[:steps], dtype=np.float64)
    state_names = np.asarray(
        (
            "position_x",
            "position_y",
            "position_z",
            "quaternion_x",
            "quaternion_y",
            "quaternion_z",
            "quaternion_w",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "angular_velocity_x",
            "angular_velocity_y",
            "angular_velocity_z",
        )
    )
    policy_names = np.asarray(tuple(f"policy_{index:02d}" for index in range(policy_count)))
    trace = ImmutableTrace(
        schema_version=np.asarray(TRACE_SCHEMA_VERSION, dtype=np.uint16),
        scenario_tape_sha256=np.asarray(tape.sha256),
        time=time_nodes,
        state_names=state_names,
        control_names=np.asarray(
            ("motor_force_0", "motor_force_1", "motor_force_2", "motor_force_3")
        ),
        barrier_names=np.asarray(
            (
                "static_node",
                "dynamic_node",
                "arena",
                "speed",
                "angular_rate",
                "tilt",
                "static_swept",
                "dynamic_swept",
            )
        ),
        policy_names=policy_names,
        loss_term_names=np.asarray(
            (
                "mass_error",
                "drag_error",
                "wind_error",
                "rotor_error",
                "candidate_bptt_loss",
                "hard_safe_policy_count",
            )
        ),
        latency_names=np.asarray(
            (
                "controller",
                "plant",
                "estimator_tick_work",
                "command_preparation",
                "postprocessing",
                "wall_step",
                "command_ready",
            )
        ),
        true_state=states,
        estimated_state=states.copy(),
        nominal_control=nominal_controls,
        filtered_control=filtered_controls,
        applied_control=applied_controls,
        executed_control=np.arange(steps) < steps - 1,
        hard_barriers=barriers,
        training_values=training_values,
        policy_values=policy_values,
        selected_policy=selected_policies,
        snapshot_version=snapshot_versions,
        model_version=model_versions,
        solver_kkt_residual=np.maximum(kkt, 0.0),
        postcheck_residual=postcheck,
        clipped=clipped,
        saturated=saturated,
        degraded=degraded,
        contact=contact,
        failure=failure,
        loss_terms=losses,
        gradient_norm=gradient_norms,
        component_latency_seconds=latencies,
    )

    events: list[ArtifactEvent] = [
        ArtifactEvent(
            sequence=0,
            step=0,
            time_seconds=float(trace.time[0]),
            category="runtime",
            name="trial_started",
            severity="info",
            snapshot_version=int(trace.snapshot_version[0]),
            model_version=int(trace.model_version[0]),
            details={"method": method.value, "condition": condition.value},
        )
    ]
    for step, category, name, details in event_payloads:
        events.append(
            ArtifactEvent(
                sequence=len(events),
                step=step,
                time_seconds=float(trace.time[step]),
                category=category,
                name=name,
                severity="warning" if "rejected" in name or "not_submitted" in name else "info",
                snapshot_version=int(trace.snapshot_version[step]),
                model_version=int(trace.model_version[step]),
                details=details,
            )
        )
    for step in np.flatnonzero(failure):
        events.append(
            ArtifactEvent(
                sequence=len(events),
                step=int(step),
                time_seconds=float(trace.time[step]),
                category="safety",
                name="hard_failure",
                severity="failure",
                snapshot_version=int(trace.snapshot_version[step]),
                model_version=int(trace.model_version[step]),
                details={
                    "minimum_margin": float(np.min(trace.hard_barriers[step])),
                    "contact": bool(trace.contact[step]),
                },
            )
        )
    ordered_events = [events[0], *sorted(events[1:], key=lambda event: event.step)]
    events = [
        ArtifactEvent(
            sequence=sequence,
            step=event.step,
            time_seconds=event.time_seconds,
            category=event.category,
            name=event.name,
            severity=event.severity,
            snapshot_version=event.snapshot_version,
            model_version=event.model_version,
            details=event.details,
        )
        for sequence, event in enumerate(ordered_events)
    ]

    adaptation_evidence: AdaptationEvidence | None = None
    if online_library:
        bound_decisions = tuple(
            replace(
                decision,
                used_by_executed_control=bool(
                    decision.status == "admitted"
                    and np.any(
                        trace.executed_control
                        & (np.arange(trace.steps, dtype=np.int64) >= decision.boundary_step)
                        & (trace.snapshot_version == decision.publication_active.version)
                    )
                ),
            )
            for decision in adaptation_decisions
        )
        adaptation_evidence = AdaptationEvidence(
            trace_content_sha256=trace.content_sha256, decisions=bound_decisions
        )
        validate_adaptation_evidence_binding(
            adaptation_evidence,
            trace,
            tuple(events),
            shared_stochastic_seed=assignment.shared_stochastic_seed,
        )

    compile_seconds = {
        "controller": controller_compile,
        "plant": plant_compile,
        "estimator_tick_work": estimator_compile,
        "command_preparation": 0.0,
        "postprocessing": 0.0,
        "wall_step": 0.0,
        "command_ready": 0.0,
    }
    compile_cache_hits = {
        "controller": controller_cache_hit,
        "plant": plant_cache_hit,
        "estimator": estimator_cache_hit,
        "bptt_startup": bool(
            online_library and cold_start_diagnostics.get("bptt_compiled_cache_hit", False)
        ),
        "bptt_online": bool(
            online_library
            and (
                any(
                    bool(details.get("cache_hit", False))
                    for _, _, name, details in event_payloads
                    if name == "online_execution_isolated"
                )
                or any(
                    bool(details.get("bptt_compiled_cache_hit", False))
                    for _, category, name, details in event_payloads
                    if category == "adaptation"
                    and name in {"candidate_admitted", "candidate_rejected"}
                )
            )
        ),
        "validation_startup": bool(
            online_library and cold_start_diagnostics.get("validation_compiled_cache_hit", False)
        ),
        "validation_online": bool(
            online_library
            and (
                any(
                    bool(details.get("evidence_cache_hit", False))
                    for _, _, name, details in event_payloads
                    if name == "online_execution_isolated"
                )
                or any(
                    bool(details.get("validation_compiled_cache_hit", False))
                    for _, category, name, details in event_payloads
                    if category == "adaptation"
                    and name in {"candidate_admitted", "candidate_rejected"}
                )
            )
        ),
    }
    deadlines = {
        "controller": config.controller_deadline_seconds,
        "plant": config.controller_deadline_seconds,
        "estimator_tick_work": config.estimator_deadline_seconds,
        "command_preparation": config.logging_deadline_seconds,
        "postprocessing": config.logging_deadline_seconds,
        "wall_step": config.dt,
        "command_ready": config.dt,
    }
    estimation_scale = np.asarray(
        [float(resolved.model.mass), 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0]
    )
    change_indices: tuple[int, ...] = ()
    if condition in {ConditionID.DYNAMICS_CHANGE, ConditionID.FALSIFICATION_COMBINED}:
        change_indices = tuple(
            sorted(
                {
                    int(index)
                    for index in tape.schedule_change_indices
                    if 0 <= int(index) < steps - 1
                }
            )
        )
    scientific = derive_scientific_metrics(
        trace,
        hard_certified_policy=certified,
        estimation_error=estimation_error,
        estimation_scale=estimation_scale,
        change_indices=change_indices,
        latency_deadlines_seconds=deadlines,
        interval_safety_evidence=True,
        warm_execution_excludes_compilation=True,
    )
    dashboard_evidence = _build_dashboard_evidence(
        trace,
        tape,
        config,
        events,
        nominal_rollout_positions,
        nominal_rollout_available,
        fallback_rollout_positions,
        fallback_rollout_available,
        selected_rollout_positions,
        selected_rollout_available,
        ghost_rollout_positions,
        ghost_rollout_available,
        normalized_descriptors,
        descriptor_available,
        dynamics_true_parameters,
        dynamics_estimated_parameters,
        dynamics_uncertainty,
        dynamics_uncertainty_available,
    )
    events.append(
        ArtifactEvent(
            sequence=len(events),
            step=steps - 1,
            time_seconds=float(trace.time[-1]),
            category="runtime",
            name="dashboard_evidence_committed",
            severity="info",
            snapshot_version=int(trace.snapshot_version[-1]),
            model_version=int(trace.model_version[-1]),
            details={
                "dashboard_evidence_sha256": dashboard_evidence.content_sha256,
                "trace_content_sha256": trace.content_sha256,
                "schema_version": DASHBOARD_EVIDENCE_SCHEMA_VERSION,
            },
        )
    )
    validate_dashboard_evidence_binding(
        dashboard_evidence,
        trace,
        tape,
        events=tuple(events),
        expected_dynamics=(
            dynamics_true_parameters,
            dynamics_estimated_parameters,
            dynamics_uncertainty,
            dynamics_uncertainty_available,
        ),
    )

    blockers: list[str] = []
    if config.policy_count != 64 and method_spec(method).uses_policy_library:
        blockers.append("development policy count is not K=64")
    if config.certificate_horizon != 50 and method_spec(method).uses_policy_library:
        blockers.append("development certificate horizon is not H=50")
    if method is MethodID.OFFLINE_FROZEN_SDCBF_STYLE and not offline_pretrain_succeeded:
        blockers.append("offline generic-diversity BPTT did not produce a frozen learned library")
    if online_library:
        blockers.extend(_online_adaptation_lifecycle_blockers(events))
        if adaptation_mode is AdaptationExecutionMode.REALTIME_PROBE:
            blockers.append(
                "realtime_probe is hardware-feasibility evidence, not load-invariant "
                "safety evidence"
            )
    if (
        method is MethodID.DA_PLCBF_FULL
        and condition in {ConditionID.DYNAMICS_CHANGE, ConditionID.FALSIFICATION_COMBINED}
        and not np.any(model_versions > 0)
    ):
        blockers.append("online estimator produced no accepted model update")
    return TrialRun(
        assignment=assignment,
        tape=tape,
        trace=trace,
        events=tuple(events),
        compile_seconds=compile_seconds,
        compile_cache_hits=compile_cache_hits,
        deadlines_seconds=deadlines,
        hard_certified_policy=certified,
        estimation_error=estimation_error,
        estimation_scale=estimation_scale,
        scientific_metrics=scientific,
        dashboard_evidence=dashboard_evidence,
        adaptation_evidence=adaptation_evidence,
        method_claim_eligible=not blockers,
        claim_blockers=tuple(blockers),
    )


def save_trial_run(run: TrialRun, run_directory: str | Path) -> Path:
    """Write one tape, trace, sidecar, events, metrics, and timing files."""
    root = Path(run_directory)
    tape_path = root / "scenario_tapes" / run.assignment.condition / f"{run.assignment.fold}.npz"
    tape_path.parent.mkdir(parents=True, exist_ok=True)
    if tape_path.exists():
        if load_scenario_tape(tape_path).sha256 != run.tape.sha256:
            raise ValueError("existing paired scenario tape has a different semantic digest")
    else:
        save_scenario_tape(run.tape, tape_path)
    method_directory = (
        root
        / "methods"
        / run.assignment.method
        / run.assignment.condition
        / str(run.assignment.fold)
    )
    method_directory.mkdir(parents=True, exist_ok=False)
    save_trace(run.trace, method_directory / "trace.npz")
    save_dashboard_evidence(run.dashboard_evidence, method_directory / "dashboard_evidence.npz")
    if run.adaptation_evidence is not None:
        save_adaptation_evidence(
            run.adaptation_evidence, method_directory / "adaptation_evidence.npz"
        )
    write_events(run.events, method_directory / "events.jsonl", trace=run.trace)
    write_metrics(run.trace, method_directory / "metrics.json")
    write_timing(
        run.trace,
        method_directory / "timing.json",
        compile_seconds=run.compile_seconds,
        deadline_seconds=run.deadlines_seconds,
    )
    return method_directory


_CONFIRMATORY_SAFETY_METRICS: tuple[tuple[str, MetricDirection], ...] = (
    ("operational_failure", MetricDirection.LOWER_IS_BETTER),
    ("any_failure", MetricDirection.LOWER_IS_BETTER),
    ("minimum_hard_margin", MetricDirection.HIGHER_IS_BETTER),
)

_EXPLORATORY_COMPARISON_METRICS: tuple[tuple[str, MetricDirection], ...] = (
    ("certified_time_fraction", MetricDirection.HIGHER_IS_BETTER),
    ("degraded_duration_seconds", MetricDirection.LOWER_IS_BETTER),
    ("intervention_integral", MetricDirection.LOWER_IS_BETTER),
    ("controller_p99_seconds", MetricDirection.LOWER_IS_BETTER),
    ("command_ready_p99_seconds", MetricDirection.LOWER_IS_BETTER),
    ("wall_step_p99_seconds", MetricDirection.LOWER_IS_BETTER),
)


def _controller_p99_seconds(metrics: ScientificTrialMetrics) -> float:
    return metrics.latency_scalar("controller", "p99_seconds")


def _command_ready_p99_seconds(metrics: ScientificTrialMetrics) -> float:
    return metrics.latency_scalar("command_ready", "p99_seconds")


def _wall_step_p99_seconds(metrics: ScientificTrialMetrics) -> float:
    return metrics.latency_scalar("wall_step", "p99_seconds")


def _campaign_paired_comparisons(
    dataset: PairedTrialDataset,
) -> tuple[
    PairedInferenceConfig | None, PairedInferenceConfig | None, tuple[PairedComparison, ...]
]:
    """Run the predeclared paired protocol without converting eligibility into a claim."""
    candidate = MethodID.DA_PLCBF_FULL.value
    if candidate not in dataset.schedule.methods:
        return None, None, ()
    baselines = tuple(method for method in dataset.schedule.methods if method != candidate)
    if not baselines:
        return None, None, ()
    confirmatory_count = (
        len(dataset.schedule.conditions) * len(baselines) * len(_CONFIRMATORY_SAFETY_METRICS)
    )
    inference = PairedInferenceConfig(
        analysis_role=AnalysisRole.CONFIRMATORY,
        bootstrap_replicates=confirmatory_bootstrap_replicates(confirmatory_count),
        familywise_comparisons=confirmatory_count,
    )
    exploratory_inference = PairedInferenceConfig(
        analysis_role=AnalysisRole.EXPLORATORY, familywise_comparisons=1
    )
    comparisons: list[PairedComparison] = []
    protocols = (
        (inference, _CONFIRMATORY_SAFETY_METRICS),
        (exploratory_inference, _EXPLORATORY_COMPARISON_METRICS),
    )
    for protocol, metrics in protocols:
        for condition in dataset.schedule.conditions:
            for baseline in baselines:
                for metric_name, direction in metrics:
                    getter = None
                    if metric_name == "controller_p99_seconds":
                        getter = _controller_p99_seconds
                    elif metric_name == "command_ready_p99_seconds":
                        getter = _command_ready_p99_seconds
                    elif metric_name == "wall_step_p99_seconds":
                        getter = _wall_step_p99_seconds
                    comparisons.append(
                        compare_paired_metric(
                            dataset,
                            condition=condition,
                            candidate_method=candidate,
                            baseline_method=baseline,
                            metric_name=metric_name,
                            direction=direction,
                            inference=protocol,
                            metric_getter=getter,
                        )
                    )
    return inference, exploratory_inference, tuple(comparisons)


def _global_confirmatory_superiority_supported(comparisons: tuple[PairedComparison, ...]) -> bool:
    """Require every member of a nonempty confirmatory family to support superiority."""
    confirmatory = tuple(
        item for item in comparisons if item.analysis_role is AnalysisRole.CONFIRMATORY
    )
    return bool(confirmatory) and all(item.superiority_supported for item in confirmatory)


def run_campaign(
    campaign: CampaignConfig,
    *,
    output_directory: str | Path | None = None,
    resume: bool = False,
    repository: str | Path | None = None,
) -> CampaignRun:
    """Run or resume the complete paired matrix, retaining every scheduled outcome.

    When ``output_directory`` is supplied, configuration, provenance, all scenario tapes, every
    success/failure outcome, numerical artifacts, and paired inference are committed through the
    crash-auditable :class:`CampaignArtifactStore`.  A resume never silently retries a recorded
    failure or skips a success whose trace/sidecar/events/metrics/timing files fail validation.
    """
    if resume and output_directory is None:
        raise ValueError("resume requires output_directory")
    schedule = campaign.schedule()
    tape_by_pair: dict[tuple[str, int], ScenarioTape] = {}
    resources_by_pair: dict[tuple[str, int], ExperimentResources] = {}
    assignment_by_pair: dict[tuple[str, int], TrialAssignment] = {}
    for assignment in schedule.assignments:
        assignment_by_pair.setdefault(assignment.pair_key, assignment)
    for pair, assignment in assignment_by_pair.items():
        tape_by_pair[pair] = generate_condition_tape(
            assignment.condition,
            campaign.trial,
            seed=assignment.scenario_root_seed,
            fold=assignment.scenario_fold,
        )

    artifact_store = None
    if output_directory is not None:
        from crazyflow.safety.da_plcbf.campaign_artifacts import CampaignArtifactStore

        repository_root = (
            Path(repository).resolve()
            if repository is not None
            else Path(__file__).resolve().parents[3]
        )
        artifact_store = CampaignArtifactStore(
            output_directory,
            campaign,
            schedule,
            tape_by_pair,
            repository=repository_root,
            resume=resume,
        )

    offline_checkpoints: dict[str, tuple[PolicySnapshot, Mapping[str, Any]]] = {}
    executable_cache = _CampaignExecutableCache()
    runs: list[TrialRun] = []
    records: list[ScientificTrialRecord] = (
        [] if artifact_store is None else list(artifact_store.records())
    )
    completed_keys = frozenset() if artifact_store is None else artifact_store.completed_keys()
    for assignment in schedule.assignments:
        if assignment.key in completed_keys:
            continue
        pair = assignment.pair_key
        tape = tape_by_pair[pair]
        if pair not in resources_by_pair:
            resources_by_pair[pair] = build_experiment_resources(
                campaign.trial,
                obstacle_count=tape.static_positions.shape[0] + tape.dynamic_positions.shape[1],
                initialization_seed=int(assignment.shared_stochastic_seed & 0xFFFFFFFF),
            )
        try:
            offline_checkpoint = None
            if assignment.method == MethodID.OFFLINE_FROZEN_SDCBF_STYLE.value:
                if assignment.condition not in offline_checkpoints:
                    offline_checkpoints[assignment.condition] = _offline_generic_checkpoint(
                        _condition(assignment.condition), resources_by_pair[pair], campaign.trial
                    )
                offline_checkpoint = offline_checkpoints[assignment.condition]
            run = run_trial(
                assignment,
                tape,
                campaign.trial,
                resources=resources_by_pair[pair],
                offline_checkpoint=offline_checkpoint,
                executable_cache=executable_cache,
            )
        except Exception as error:
            failure_code = f"{type(error).__name__.lower()}"
            failure_message = " ".join(str(error).split())[:1000] or failure_code
            record = ScientificTrialRecord(
                method=assignment.method,
                condition=assignment.condition,
                fold=assignment.fold,
                pairing_id=assignment.pairing_id,
                scenario_tape_sha256=tape.sha256,
                status=TrialStatus.EXECUTION_FAILURE,
                metrics=None,
                failure_code=failure_code,
                failure_message=failure_message,
            )
            records.append(record)
            if artifact_store is not None:
                artifact_store.record(assignment, record, None)
            continue
        runs.append(run)
        record = ScientificTrialRecord(
            method=assignment.method,
            condition=assignment.condition,
            fold=assignment.fold,
            pairing_id=assignment.pairing_id,
            scenario_tape_sha256=tape.sha256,
            status=TrialStatus.COMPLETE,
            metrics=run.scientific_metrics,
        )
        records.append(record)
        if output_directory is not None:
            save_trial_run(run, output_directory)
            assert artifact_store is not None
            artifact_store.record(assignment, record, run)

    if artifact_store is not None:
        records = list(artifact_store.records())
    dataset = PairedTrialDataset(schedule=schedule, records=tuple(records))
    dataset.validate()
    inference_config, exploratory_inference_config, comparisons = _campaign_paired_comparisons(
        dataset
    )

    blockers: list[str] = []
    failed = [record.key for record in records if record.status is TrialStatus.EXECUTION_FAILURE]
    if failed:
        blockers.append(f"{len(failed)} scheduled executions failed")
    ineligible = (
        [run.assignment.key for run in runs if not run.method_claim_eligible]
        if artifact_store is None
        else list(artifact_store.ineligible_success_keys())
    )
    if ineligible:
        blockers.append(f"{len(ineligible)} completed runs failed method claim gates")
    online_methods = {
        MethodID.DA_PLCBF_NO_ONLINE_MODEL_ADAPTATION.value,
        MethodID.DA_PLCBF_FULL.value,
    }.intersection(campaign.methods)
    proven_online_methods = _campaign_online_snapshot_use(campaign.methods, runs, output_directory)
    for method in sorted(online_methods - proven_online_methods):
        blockers.append(
            f"online method {method} never proved an admitted snapshot drove executed control"
        )
    blockers.extend(campaign.final_contract_blockers())
    if not schedule.final_claim_eligible:
        blockers.append("schedule is not a predeclared >=100-pair final-claim schedule")
    result = CampaignRun(
        schedule=schedule,
        trial_runs=tuple(runs),
        records=tuple(records),
        paired_comparisons=comparisons,
        inference_config=inference_config,
        exploratory_inference_config=exploratory_inference_config,
        execution_complete=not failed,
        scientific_claim_eligible=not blockers,
        # This aggregate is deliberately a conjunction across the entire predeclared
        # confirmatory family.  Individual supported endpoints remain available in
        # ``paired_comparisons``; one favorable metric must never be summarized as global
        # superiority over every baseline and condition.
        global_confirmatory_superiority_supported=(
            _global_confirmatory_superiority_supported(comparisons)
        ),
        claim_blockers=tuple(blockers),
    )
    if artifact_store is not None:
        artifact_store.finalize_numeric(result)
    return result


__all__ = [
    "AdaptationExecutionMode",
    "CampaignConfig",
    "CampaignRun",
    "ConditionID",
    "ExperimentConfig",
    "ExperimentResources",
    "REQUIRED_CONDITIONS",
    "TrialExecutionError",
    "TrialRun",
    "build_experiment_resources",
    "generate_condition_tape",
    "run_campaign",
    "run_trial",
    "save_trial_run",
    "scenario_config_for_condition",
]
