"""Matched closed-loop evidence for dynamics knowledge and finite uncertainty.

This campaign is intentionally separate from the seven-method safety campaign and from the
candidate-proposal ablations.  It holds the DA-PLCBF policy library, point-model BPTT objective,
hard candidate admission, moving-obstacle prediction tape, nonlinear discrete filter, and true
plant fixed while changing only the dynamics information supplied to the runtime filter:

* ``oracle_point`` reads the true dynamics at the current control boundary.  It is a privileged,
  nondeployable upper-bound knowledge reference, not a baseline available to a real controller;
* ``estimated_r0`` uses the causal estimator mean and no dynamics particles;
* ``estimated_cartesian_r4`` and ``estimated_cartesian_r8`` use the same causal estimator mean
  with four/eight deterministic covariance particles in the Cartesian robust filter.

Every variant uses :func:`discrete_nonlinear_plcbf_filter` through either the point-model dynamic
runtime or its Cartesian finite-uncertainty counterpart.  R=0 therefore never aliases the
continuous Version-A filter.  Online BPTT remains a *point-model* objective for all variants; the
R=4 hard admission set and R=4/R=8 runtime sets are not described as uncertainty-aware training.

The declared dynamics-change condition changes mass, drag, and wind while holding true rotor
efficiency exactly one.  This makes the point-model oracle mathematically representable by the
same direct-wrench plant/filter equations; estimated particle sets may still express rotor-force
uncertainty from their covariance.  All exogenous values live in one immutable tape per fold.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
import time
import zipfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from crazyflow.safety.da_plcbf import experiments as experiment_core
from crazyflow.safety.da_plcbf.direct_wrench import motor_forces_to_wrench
from crazyflow.safety.da_plcbf.dynamic_filter import dynamic_discrete_runtime_step
from crazyflow.safety.da_plcbf.dynamic_rollouts import dynamic_sphere_window_from_tape
from crazyflow.safety.da_plcbf.estimator import (
    EstimatorState,
    RotorEfficiencyObservations,
    TranslationalObservations,
    initialize_estimator,
    physical_parameters,
    update_rotor_efficiency,
    update_translational_estimate,
)
from crazyflow.safety.da_plcbf.experiments import ConditionID, ExperimentConfig, ExperimentResources
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.scenarios import (
    ScenarioTape,
    generate_scenario_tape,
    load_scenario_tape,
    save_scenario_tape,
)
from crazyflow.safety.da_plcbf.snapshots import (
    ActiveSnapshotStore,
    PolicySnapshot,
    create_active_snapshot,
)
from crazyflow.safety.da_plcbf.uncertain_dynamic_filter import (
    uncertain_dynamic_discrete_runtime_step,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crazyflow.safety.da_plcbf.actor import SharedActorParams
    from crazyflow.safety.da_plcbf.quad_uncertainty import VersionAModelSamples
    from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel


CAMPAIGN_SCHEMA_VERSION = 3
OUTCOME_SCHEMA_VERSION = 3
TRACE_SCHEMA_VERSION = 2
MANIFEST_SCHEMA_VERSION = 3
AGGREGATE_SCHEMA_VERSION = 3
COMPLETION_MARKER_SCHEMA_VERSION = 3
STARTUP_SCHEMA_VERSION = 2
STARTUP_BUNDLE_SCHEMA_VERSION = 1
RUNTIME_PROVENANCE_SCHEMA_VERSION = 2
CONDITION_ID = ConditionID.DYNAMICS_CHANGE.value
FILTER_IMPLEMENTATION_ID = "discrete_nonlinear_plcbf_filter"
TRAINING_OBJECTIVE_ID = "point_model_plcbf_aligned_coverage_diversity"
TRAINING_CLAIM = (
    "BPTT uses one estimator/oracle point model. Runtime R=4/R=8 particles and the common R=4 "
    "hard admission set are not uncertainty-aware BPTT training."
)
ORACLE_LABEL = (
    "Privileged upper-bound dynamics knowledge: reads only true current-boundary dynamics; "
    "not deployable and not eligible for blanket superiority claims."
)
CLAIM_BOUNDARY = (
    "Closed-loop finite-tape dynamics-knowledge evidence only. Results are paired differences "
    "for the declared condition, not a blanket safety-superiority claim."
)
OPERATIONAL_FAILURE_DEFINITION = (
    "A scheduled outcome fails operationally on trial execution failure, physical/constraint "
    "failure, any degraded executed control interval, or startup/periodic adaptation execution "
    "failure. Terminal no-command rows do not create degraded-control failures."
)
AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION = (
    "Mean coordinatewise fraction of the 11 true parameter components between the minimum and "
    "maximum of the finite runtime samples at boundaries containing samples. This axis-aligned "
    "sample-range statistic is not convex-hull, posterior, probability, continuum, or safety "
    "coverage."
)
ADAPTATION_VALIDATION_SAMPLES = 4
DYNAMICS_PARAMETER_COUNT = 11
MAX_RUNTIME_SAMPLES = 8
PLANT_REPLAY_RTOL = 5e-6
PLANT_REPLAY_ATOL = 1e-6
_RUNTIME_TIMING_KEYS = frozenset(
    {
        "point_controller",
        "cartesian_r4_controller",
        "cartesian_r8_controller",
        "true_plant",
        "causal_estimator",
    }
)

VariantID = Literal[
    "oracle_point", "estimated_r0", "estimated_cartesian_r4", "estimated_cartesian_r8"
]


@dataclass(frozen=True, slots=True)
class DynamicsKnowledgeVariant:
    """One exact controller-knowledge intervention."""

    variant_id: VariantID
    knowledge_source: str
    runtime_dynamics_samples: int
    cartesian_runtime_filter: bool
    causal_estimator_history_only: bool
    privileged_oracle_upper_bound: bool
    deployable_interpretation: bool
    filter_implementation: str = FILTER_IMPLEMENTATION_ID
    training_objective: str = TRAINING_OBJECTIVE_ID
    uncertainty_aware_bptt_training: bool = False
    adaptation_validation_samples: int = ADAPTATION_VALIDATION_SAMPLES


VARIANTS = (
    DynamicsKnowledgeVariant(
        "oracle_point", "true_current_boundary_dynamics", 0, False, False, True, False
    ),
    DynamicsKnowledgeVariant("estimated_r0", "causal_estimator_mean", 0, False, True, False, True),
    DynamicsKnowledgeVariant(
        "estimated_cartesian_r4",
        "causal_estimator_mean_and_covariance_particles",
        4,
        True,
        True,
        False,
        True,
    ),
    DynamicsKnowledgeVariant(
        "estimated_cartesian_r8",
        "causal_estimator_mean_and_covariance_particles",
        8,
        True,
        True,
        False,
        True,
    ),
)


def _validate_variants() -> None:
    if tuple(item.variant_id for item in VARIANTS) != (
        "oracle_point",
        "estimated_r0",
        "estimated_cartesian_r4",
        "estimated_cartesian_r8",
    ):
        raise ValueError("dynamics-knowledge variants changed order")
    for variant in VARIANTS:
        if variant.runtime_dynamics_samples not in (0, 4, 8):
            raise ValueError("runtime dynamics samples must be 0, 4, or 8")
        if variant.cartesian_runtime_filter != (variant.runtime_dynamics_samples > 0):
            raise ValueError("Cartesian runtime label and dynamics sample count differ")
        if variant.filter_implementation != FILTER_IMPLEMENTATION_ID:
            raise ValueError("every variant must use the discrete nonlinear filter")
        if variant.training_objective != TRAINING_OBJECTIVE_ID:
            raise ValueError("every variant must use the same point-model training objective")
        if variant.uncertainty_aware_bptt_training:
            raise ValueError("no dispatched variant has uncertainty-aware BPTT training")
        if variant.privileged_oracle_upper_bound != (variant.variant_id == "oracle_point"):
            raise ValueError("only oracle_point may be marked as the privileged upper bound")


@dataclass(frozen=True, slots=True)
class DynamicsKnowledgeProfile:
    """Predeclared closed-loop shape and matched-trial count."""

    name: str
    trials: int
    trial: ExperimentConfig
    intended_for_confirmatory_differences: bool

    def validate(self) -> None:
        _validate_variants()
        if self.name not in {"smoke", "development", "final"}:
            raise ValueError("unknown dynamics-knowledge profile")
        if self.trials <= 0:
            raise ValueError("profile trials must be positive")
        self.trial.validate()
        if self.trial.policy_count < 16:
            raise ValueError("dynamics-knowledge profiles require K >= 16")
        if self.trial.prediction_samples != 4:
            raise ValueError("all profiles require the common R_o=4 obstacle prediction axis")
        if self.trial.uncertainty_sample_count != ADAPTATION_VALIDATION_SAMPLES:
            raise ValueError("common candidate admission must use R_m=4")
        if self.intended_for_confirmatory_differences:
            if self.name != "final" or self.trials < 100:
                raise ValueError("confirmatory differences require the final >=100-trial profile")
            exact = (
                self.trial.policy_count,
                self.trial.certificate_horizon,
                self.trial.training_scenario_count,
            )
            if exact != (64, 50, 64):
                raise ValueError("final confirmatory shape must be exactly K64/H50/B64")


def dynamics_knowledge_profile(name: str, *, random_seed: int = 0) -> DynamicsKnowledgeProfile:
    """Return the exact smoke, development, or final protocol."""
    if name == "smoke":
        trial = ExperimentConfig(
            control_steps=6,
            certificate_horizon=2,
            policy_count=16,
            prediction_samples=4,
            training_scenario_count=2,
            validation_scenarios_per_fold=1,
            bptt_burst_steps=1,
            adaptation_interval_steps=3,
            estimator_interval_steps=1,
            estimator_window_steps=3,
            static_capacity=2,
            static_count=1,
            dynamic_capacity=1,
            random_seed=random_seed,
        )
        profile = DynamicsKnowledgeProfile("smoke", 1, trial, False)
    elif name == "development":
        trial = ExperimentConfig(
            control_steps=76,
            certificate_horizon=25,
            policy_count=32,
            prediction_samples=4,
            training_scenario_count=32,
            validation_scenarios_per_fold=2,
            bptt_burst_steps=5,
            adaptation_interval_steps=15,
            estimator_interval_steps=5,
            estimator_window_steps=12,
            random_seed=random_seed,
        )
        profile = DynamicsKnowledgeProfile("development", 10, trial, False)
    elif name == "final":
        trial = ExperimentConfig(
            control_steps=151,
            certificate_horizon=50,
            policy_count=64,
            prediction_samples=4,
            training_scenario_count=64,
            validation_scenarios_per_fold=2,
            bptt_burst_steps=10,
            adaptation_interval_steps=25,
            estimator_interval_steps=10,
            estimator_window_steps=12,
            random_seed=random_seed,
        )
        profile = DynamicsKnowledgeProfile("final", 100, trial, True)
    else:
        raise ValueError(f"unknown dynamics-knowledge profile {name!r}")
    profile.validate()
    return profile


@dataclass(frozen=True, slots=True)
class DynamicsKnowledgeCampaignConfig:
    """Resolved profile and deterministic matched-fold schedule."""

    profile: str = "smoke"
    root_seed: int = 260831
    fold_start: int = 0
    trials: int | None = None

    def resolved_profile(self) -> DynamicsKnowledgeProfile:
        selected = dynamics_knowledge_profile(self.profile, random_seed=self.root_seed)
        if self.trials is not None:
            if self.trials <= 0:
                raise ValueError("trials override must be positive")
            selected = replace(
                selected,
                trials=self.trials,
                intended_for_confirmatory_differences=(
                    selected.name == "final" and self.trials >= 100
                ),
            )
        selected.validate()
        return selected

    def validate(self) -> None:
        self.resolved_profile()
        if not 0 <= self.root_seed <= np.iinfo(np.uint32).max:
            raise ValueError("root_seed must fit uint32")
        if self.fold_start < 0:
            raise ValueError("fold_start must be nonnegative")
        profile = self.resolved_profile()
        if self.fold_start + profile.trials - 1 > np.iinfo(np.uint32).max:
            raise ValueError("scheduled fold indices must fit uint32")


@dataclass(frozen=True, slots=True)
class DynamicsKnowledgeCampaignRun:
    """Campaign execution summary."""

    root: Path
    expected_outcomes: int
    completed_outcomes: int
    failed_outcomes: int
    execution_complete: bool
    manifest_sha256: str
    operational_failures: int = 0


@dataclass(frozen=True, slots=True)
class DynamicsKnowledgeVerification:
    """Strict artifact verification result."""

    valid: bool
    errors: tuple[str, ...]
    expected_outcomes: int
    retained_outcomes: int
    completed_outcomes: int
    failed_outcomes: int
    operational_failures: int = 0


def _stable_seed(root_seed: int, fold: int, label: str) -> int:
    digest = hashlib.sha256(
        b"crazyflow.da_plcbf.dynamics-knowledge-seed.v1\0"
        + str(root_seed).encode("ascii")
        + b"\0"
        + str(fold).encode("ascii")
        + b"\0"
        + label.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little")


def generate_matched_dynamics_tape(
    profile: DynamicsKnowledgeProfile, *, root_seed: int, fold: int
) -> ScenarioTape:
    """Generate the common mass/drag/wind-change tape with true rotor efficiency fixed at one."""
    profile.validate()
    base = experiment_core.scenario_config_for_condition(CONDITION_ID, profile.trial)
    tape_config = replace(
        base,
        rotor_efficiency_bounds=(1.0, 1.0),
        rotor_single_efficiency_lower=1.0,
        estimator_motor_force_noise_std=0.0,
    )
    tape = generate_scenario_tape(
        _stable_seed(root_seed, fold, "scenario-tape"), tape_config, fold=fold
    )
    if not np.array_equal(tape.rotor_efficiency, np.ones_like(tape.rotor_efficiency)):
        raise RuntimeError("declared knowledge campaign must hold true rotor efficiency at one")
    changed = (
        np.any(tape.mass_scale[: profile.trial.control_steps] != 1.0)
        and np.any(tape.drag_scale[: profile.trial.control_steps] != 1.0)
        and np.any(tape.wind_velocity[: profile.trial.control_steps] != 0.0)
    )
    if not changed:
        raise RuntimeError("declared dynamics-change tape did not change mass, drag, and wind")
    return tape


def build_matched_resources(
    profile: DynamicsKnowledgeProfile, tape: ScenarioTape, *, root_seed: int, fold: int
) -> ExperimentResources:
    """Build the byte-identical library/controller resources shared by all four variants."""
    obstacle_count = tape.static_positions.shape[0] + tape.dynamic_positions.shape[1]
    resources = experiment_core.build_experiment_resources(
        profile.trial,
        obstacle_count=obstacle_count,
        initialization_seed=_stable_seed(root_seed, fold, "shared-library-initialization"),
    )
    return experiment_core._resources_for_tape(resources, tape, profile.trial)


class _ControlEvidence(NamedTuple):
    motor_forces: Array
    nominal_motor_forces: Array
    policy_hard_values: Array
    selected_policy: Array
    selected_hard_value: Array
    degraded: Array
    proposal_accepted: Array
    fallback_accepted: Array
    used_fallback: Array
    interval_margin: Array
    next_value: Array
    exact_residual: Array


class _CompiledExecutables:
    """One fixed-shape controller/plant/estimator executable set reused across folds."""

    def __init__(self, resources: ExperimentResources, profile: DynamicsKnowledgeProfile) -> None:
        self.resources = resources
        self.profile = profile
        self.point: Any | None = None
        self.cartesian: dict[int, Any] = {}
        self.plant: Any | None = None
        self.estimator: Any | None = None
        self.compile_seconds: dict[str, float] = {}
        self.warmup_seconds: dict[str, float] = {}


def _block(value: Any) -> Any:
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, value
    )


def _point_control_function(
    resources: ExperimentResources, profile: DynamicsKnowledgeProfile
) -> Any:
    config = profile.trial

    def control(
        state: Array,
        target_position: Array,
        target_velocity: Array,
        previous_policy: Array,
        params: SharedActorParams,
        window: Any,
        controller_model: VersionAModel,
    ) -> _ControlEvidence:
        result = dynamic_discrete_runtime_step(
            state,
            target_position,
            target_velocity,
            previous_policy,
            params,
            resources.spec,
            window,
            controller_model,
            resources.actuator,
            resources.actor_config,
            resources.quad_config,
            resources.barrier_config,
            resources.dynamic_filter_config,
            dt=config.dt,
            policy_gain=config.policy_gain,
        )
        selected_residual = jnp.where(
            result.filter.proposal_accepted,
            result.filter.proposal_exact_residual,
            result.filter.fallback_exact_residual,
        )
        return _ControlEvidence(
            result.motor_forces,
            result.nominal.bounded_motor_forces,
            result.library.hard_values,
            result.selection.selected_index,
            result.selection.selected_hard_value,
            result.degraded,
            result.filter.proposal_accepted,
            result.filter.fallback_accepted,
            result.filter.used_fallback,
            result.applied_interval_margin,
            result.applied_next_value,
            selected_residual,
        )

    return jax.jit(control)


def _cartesian_control_function(
    resources: ExperimentResources, profile: DynamicsKnowledgeProfile
) -> Any:
    config = profile.trial

    def control(
        state: Array,
        target_position: Array,
        target_velocity: Array,
        previous_policy: Array,
        params: SharedActorParams,
        window: Any,
        controller_model: VersionAModel,
        model_samples: VersionAModelSamples,
    ) -> _ControlEvidence:
        result = uncertain_dynamic_discrete_runtime_step(
            state,
            target_position,
            target_velocity,
            previous_policy,
            params,
            resources.spec,
            window,
            controller_model,
            model_samples,
            resources.actuator,
            resources.actor_config,
            resources.quad_config,
            resources.barrier_config,
            resources.dynamic_filter_config,
            dt=config.dt,
            policy_gain=config.policy_gain,
            uncertainty_config=resources.uncertainty_config,
        )
        selected_residual = jnp.where(
            result.filter.proposal_accepted,
            result.filter.proposal_exact_residual,
            result.filter.fallback_exact_residual,
        )
        return _ControlEvidence(
            result.motor_forces,
            result.nominal.bounded_motor_forces,
            result.library.hard_values[:, 0],
            result.selection.selected_index,
            result.selection.selected_hard_value,
            result.degraded,
            result.filter.proposal_accepted,
            result.filter.fallback_accepted,
            result.filter.used_fallback,
            result.applied_interval_margin,
            result.applied_next_value,
            selected_residual,
        )

    return jax.jit(control)


def _plant_function(resources: ExperimentResources, profile: DynamicsKnowledgeProfile) -> Any:
    def plant(
        state: Array, motor_forces: Array, true_model: VersionAModel, true_efficiency: Array
    ) -> tuple[Array, Array]:
        realized_motor = motor_forces * true_efficiency
        realized_wrench = motor_forces_to_wrench(
            realized_motor,
            L=resources.actuator.arm_length,
            thrust2torque=resources.actuator.thrust_to_torque,
            mixing_matrix=resources.actuator.mixing_matrix,
        )
        return (
            direct_wrench_symplectic_step(state, realized_wrench, true_model, profile.trial.dt),
            realized_motor,
        )

    return jax.jit(plant)


def _plant_replay_function(
    resources: ExperimentResources, profile: DynamicsKnowledgeProfile
) -> Any:
    """Return a batched verifier for every recorded true-plant transition."""
    base = resources.model

    def transition(
        state: Array,
        motor_forces: Array,
        mass_scale: Array,
        drag_scale: Array,
        wind_velocity: Array,
        rotor_efficiency: Array,
    ) -> tuple[Array, Array]:
        model = base._replace(
            mass=base.mass * mass_scale,
            drag_matrix=base.drag_matrix * drag_scale[None, :],
            wind_velocity=wind_velocity,
        )
        realized_motor = motor_forces * rotor_efficiency
        realized_wrench = motor_forces_to_wrench(
            realized_motor,
            L=resources.actuator.arm_length,
            thrust2torque=resources.actuator.thrust_to_torque,
            mixing_matrix=resources.actuator.mixing_matrix,
        )
        return (
            direct_wrench_symplectic_step(state, realized_wrench, model, profile.trial.dt),
            realized_motor,
        )

    return jax.jit(jax.vmap(transition))


def _estimator_function(resources: ExperimentResources) -> Any:
    def estimate(
        current: EstimatorState,
        translational: TranslationalObservations,
        rotor: RotorEfficiencyObservations,
        sequence: Array,
    ) -> tuple[Any, Any]:
        rotor_update = update_rotor_efficiency(
            current, rotor, sequence=sequence, mode="per_rotor", config=resources.estimator_config
        )
        translation_update = update_translational_estimate(
            rotor_update.state, translational, sequence=sequence, config=resources.estimator_config
        )
        return translation_update, rotor_update

    return jax.jit(estimate)


def _causal_estimator(resources: ExperimentResources) -> EstimatorState:
    return experiment_core._initialize_estimator(resources)


def _controller_model(resources: ExperimentResources, estimator: EstimatorState) -> VersionAModel:
    return experiment_core._controller_model(resources.model, estimator)


def _model_samples(
    resources: ExperimentResources, estimator: EstimatorState, count: int
) -> VersionAModelSamples:
    return experiment_core._model_samples(estimator, resources, count)


def _require_valid_runtime_samples(samples: VersionAModelSamples, count: int) -> None:
    valid = np.asarray(samples.sample_valid, dtype=bool)
    if valid.shape != (count,) or not np.all(valid):
        raise RuntimeError(f"configured Cartesian runtime set R={count} is not fully valid")


def _oracle_estimator_state(
    resources: ExperimentResources, true_model: VersionAModel, *, model_version: int
) -> EstimatorState:
    """Center the common admission covariance on the oracle's true current point model."""
    nominal = _causal_estimator(resources)
    return initialize_estimator(
        resources.estimator_config,
        mass=float(np.asarray(true_model.mass)),
        drag_force_coefficients=-jnp.diag(true_model.drag_matrix),
        wind_velocity=true_model.wind_velocity,
        rotor_efficiency=1.0,
        covariance=nominal.covariance,
        model_version=model_version,
    )


def _compile_one(name: str, function: Any, arguments: tuple[Any, ...]) -> tuple[Any, float, float]:
    compile_start = time.perf_counter()
    executable = function.lower(*arguments).compile()
    compile_seconds = time.perf_counter() - compile_start
    warm_start = time.perf_counter()
    _block(executable(*arguments))
    warm_seconds = time.perf_counter() - warm_start
    if not all(math.isfinite(value) and value >= 0.0 for value in (compile_seconds, warm_seconds)):
        raise RuntimeError(f"{name} compilation timing is invalid")
    return executable, compile_seconds, warm_seconds


def compile_knowledge_executables(
    bundle: _CompiledExecutables, tape: ScenarioTape
) -> _CompiledExecutables:
    """Compile and warm point, Cartesian R4/R8, true-plant, and causal-estimator graphs."""
    if bundle.point is not None:
        return bundle
    resources = bundle.resources
    profile = bundle.profile
    config = profile.trial
    state = experiment_core._initial_state(tape)
    target_position = jnp.asarray(tape.defender_reference_position[0], dtype=state.dtype)
    target_velocity = jnp.asarray(tape.defender_reference_velocity[0], dtype=state.dtype)
    previous = jnp.asarray(-1, dtype=jnp.int32)
    window = dynamic_sphere_window_from_tape(
        tape,
        start_index=0,
        horizon=config.certificate_horizon + 1,
        speed_limit=config.speed_limit,
        angular_rate_max=config.angular_rate_max,
        tilt_max_radians=config.tilt_max_radians,
    )
    estimator = _causal_estimator(resources)
    model = _controller_model(resources, estimator)
    true_model, true_efficiency = experiment_core._true_model(
        resources.model, tape, ConditionID.DYNAMICS_CHANGE, 0
    )
    point_function = _point_control_function(resources, profile)
    point_args = (
        state,
        target_position,
        target_velocity,
        previous,
        resources.initial_params,
        window,
        model,
    )
    bundle.point, seconds, warm = _compile_one("point-controller", point_function, point_args)
    bundle.compile_seconds["point_controller"] = seconds
    bundle.warmup_seconds["point_controller"] = warm

    cartesian_function = _cartesian_control_function(resources, profile)
    for count in (4, 8):
        samples = _model_samples(resources, estimator, count)
        arguments = (*point_args, samples)
        executable, seconds, warm = _compile_one(
            f"cartesian-r{count}-controller", cartesian_function, arguments
        )
        bundle.cartesian[count] = executable
        bundle.compile_seconds[f"cartesian_r{count}_controller"] = seconds
        bundle.warmup_seconds[f"cartesian_r{count}_controller"] = warm

    plant_function = _plant_function(resources, profile)
    plant_args = (state, jnp.zeros((4,), dtype=state.dtype), true_model, true_efficiency)
    bundle.plant, seconds, warm = _compile_one("true-plant", plant_function, plant_args)
    bundle.compile_seconds["true_plant"] = seconds
    bundle.warmup_seconds["true_plant"] = warm

    observations = experiment_core._estimator_observations([], config.estimator_window_steps)
    estimator_function = _estimator_function(resources)
    estimator_args = (estimator, observations[0], observations[1], jnp.asarray(0, dtype=jnp.int32))
    bundle.estimator, seconds, warm = _compile_one(
        "causal-estimator", estimator_function, estimator_args
    )
    bundle.compile_seconds["causal_estimator"] = seconds
    bundle.warmup_seconds["causal_estimator"] = warm
    return bundle


@dataclass(frozen=True, slots=True)
class _StartupPreparation:
    active: PolicySnapshot
    record: dict[str, Any]


def _initial_active_snapshot(resources: ExperimentResources) -> PolicySnapshot:
    return create_active_snapshot(
        resources.initial_params,
        version=0,
        model_version=0,
        structural_core=resources.spec,
        metadata={
            "campaign": "dynamics_knowledge",
            "initialization": "shared_deterministic_structured_zero_residual",
        },
    )


def prepare_common_startup_adaptation(
    profile: DynamicsKnowledgeProfile,
    tape: ScenarioTape,
    resources: ExperimentResources,
    executable_pool: Any,
) -> _StartupPreparation:
    """Run one common point-model BPTT/admission job before splitting into variants."""
    initial = _initial_active_snapshot(resources)
    store = ActiveSnapshotStore(initial)
    state = experiment_core._initial_state(tape)
    estimator = _causal_estimator(resources)
    model = _controller_model(resources, estimator)
    samples = _model_samples(resources, estimator, ADAPTATION_VALIDATION_SAMPLES)
    window = dynamic_sphere_window_from_tape(
        tape,
        start_index=0,
        horizon=profile.trial.certificate_horizon + 1,
        speed_limit=profile.trial.speed_limit,
        angular_rate_max=profile.trial.angular_rate_max,
        tilt_max_radians=profile.trial.tilt_max_radians,
    )
    job = experiment_core._CandidateJob(
        tape, ConditionID.DYNAMICS_CHANGE, resources, profile.trial, executable_pool
    )
    started = time.perf_counter()
    try:
        job.set_context(0, state, model, samples, window, start_index=0)
        candidate, report = job(initial, 0)
        publication = store.admit(candidate, report)
        diagnostics = dict(job.diagnostics.get(candidate.digest, {}))
        record = {
            "status": "complete",
            "candidate_digest": candidate.digest,
            "candidate_params_digest": candidate.params_digest,
            "report_digest": report.digest,
            "report_integrity_verified": report.verify_integrity(),
            "report_passed": report.passed,
            "failed_gate_names": list(report.failed_gate_names),
            "publication_accepted": publication.accepted,
            "publication_reason": publication.reason,
            "active_digest": publication.active.digest,
            "active_params_digest": publication.active.params_digest,
            "active_version": publication.active.version,
            "model_version": publication.active.model_version,
            "training_objective": TRAINING_OBJECTIVE_ID,
            "uncertainty_aware_bptt_training": False,
            "hard_admission_dynamics_samples": ADAPTATION_VALIDATION_SAMPLES,
            "diagnostics": diagnostics,
            "execution_seconds": time.perf_counter() - started,
        }
    except Exception as error:  # noqa: BLE001 - retained common startup outcome.
        record = {
            "status": "failed",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "publication_accepted": False,
            "active_digest": initial.digest,
            "active_params_digest": initial.params_digest,
            "active_version": initial.version,
            "model_version": initial.model_version,
            "training_objective": TRAINING_OBJECTIVE_ID,
            "uncertainty_aware_bptt_training": False,
            "hard_admission_dynamics_samples": ADAPTATION_VALIDATION_SAMPLES,
            "execution_seconds": time.perf_counter() - started,
        }
    return _StartupPreparation(store.active, record)


def _periodic_adaptation(
    *,
    boundary: int,
    state: Array,
    controller_model: VersionAModel,
    admission_samples: VersionAModelSamples,
    store: ActiveSnapshotStore,
    job: Any,
    tape: ScenarioTape,
    profile: DynamicsKnowledgeProfile,
) -> dict[str, Any]:
    """Execute one synchronous causal point-model adaptation and retain rejection/failure."""
    window = dynamic_sphere_window_from_tape(
        tape,
        start_index=boundary,
        horizon=profile.trial.certificate_horizon + 1,
        speed_limit=profile.trial.speed_limit,
        angular_rate_max=profile.trial.angular_rate_max,
        tilt_max_radians=profile.trial.tilt_max_radians,
    )
    started = time.perf_counter()
    try:
        captured_version = store.model_version
        active_before = store.active
        job.set_context(
            captured_version,
            state,
            controller_model,
            admission_samples,
            window,
            start_index=boundary,
        )
        candidate, report = job(active_before, captured_version)
        publication = store.admit(candidate, report)
        return {
            "boundary": boundary,
            "status": "complete",
            "captured_model_version": captured_version,
            "active_before_digest": active_before.digest,
            "candidate_digest": candidate.digest,
            "candidate_params_digest": candidate.params_digest,
            "report_digest": report.digest,
            "report_integrity_verified": report.verify_integrity(),
            "report_passed": report.passed,
            "failed_gate_names": list(report.failed_gate_names),
            "publication_accepted": publication.accepted,
            "publication_reason": publication.reason,
            "active_after_digest": publication.active.digest,
            "active_after_params_digest": publication.active.params_digest,
            "active_after_version": publication.active.version,
            "training_objective": TRAINING_OBJECTIVE_ID,
            "uncertainty_aware_bptt_training": False,
            "hard_admission_dynamics_samples": ADAPTATION_VALIDATION_SAMPLES,
            "diagnostics": dict(job.diagnostics.get(candidate.digest, {})),
            "execution_seconds": time.perf_counter() - started,
        }
    except Exception as error:  # noqa: BLE001 - adaptation failure is retained, library stays active.
        return {
            "boundary": boundary,
            "status": "failed",
            "captured_model_version": store.model_version,
            "active_before_digest": store.active.digest,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "publication_accepted": False,
            "active_after_digest": store.active.digest,
            "active_after_params_digest": store.active.params_digest,
            "active_after_version": store.active.version,
            "training_objective": TRAINING_OBJECTIVE_ID,
            "uncertainty_aware_bptt_training": False,
            "hard_admission_dynamics_samples": ADAPTATION_VALIDATION_SAMPLES,
            "execution_seconds": time.perf_counter() - started,
        }


@dataclass(frozen=True, slots=True)
class _TrialExecution:
    fold: int
    variant: DynamicsKnowledgeVariant
    tape_digest: str
    arrays: dict[str, np.ndarray]
    metrics: dict[str, Any]
    adaptation_events: tuple[dict[str, Any], ...]
    final_params: SharedActorParams
    execution_seconds: float


def _parameter_vector(model: VersionAModel, rotor_efficiency: Array) -> np.ndarray:
    return np.asarray(
        experiment_core._dynamics_parameter_vector(model, rotor_efficiency), dtype=np.float64
    )


def _sample_parameter_vectors(samples: VersionAModelSamples) -> np.ndarray:
    return np.asarray(
        experiment_core._sampled_dynamics_parameter_vectors(samples), dtype=np.float64
    )


def _knowledge_metrics(
    arrays: Mapping[str, np.ndarray],
    tape: ScenarioTape,
    profile: DynamicsKnowledgeProfile,
    *,
    startup_adaptation_status: str = "complete",
    adaptation_events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if startup_adaptation_status not in {"complete", "failed"}:
        raise ValueError("startup adaptation status must be complete or failed")
    states = arrays["states"]
    barriers = arrays["barrier_margins"]
    contact = arrays["contact"]
    failure = arrays["failure"]
    executed = arrays["command_valid"]
    tracking = states[:, :3] - tape.defender_reference_position[: states.shape[0]]
    true_parameters = arrays["true_parameters"]
    estimated_parameters = arrays["estimated_parameters"]
    scale_floor = np.asarray([0.03, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    scales = np.maximum(np.abs(true_parameters), scale_floor[None, :])
    normalized_error = (estimated_parameters - true_parameters) / scales
    change_index = int(np.min(np.asarray(tape.schedule_change_indices[:3], dtype=np.int64)))
    change_index = min(change_index, states.shape[0] - 1)
    runtime_samples = arrays["runtime_sample_valid"]
    executed_degraded = arrays["degraded"] & executed
    adaptation_execution_failed = any(
        isinstance(event, dict) and event.get("status") == "failed" for event in adaptation_events
    )
    operational_failure = bool(
        np.any(failure)
        or np.any(executed_degraded)
        or startup_adaptation_status == "failed"
        or adaptation_execution_failed
    )
    axis_aligned_enclosure: float | None = None
    if np.any(runtime_samples):
        enclosed: list[np.ndarray] = []
        values = arrays["runtime_sample_parameters"]
        for index in range(states.shape[0]):
            valid = runtime_samples[index]
            if np.any(valid):
                selected = values[index, valid]
                truth = true_parameters[index]
                enclosed.append(
                    (truth >= np.min(selected, axis=0)) & (truth <= np.max(selected, axis=0))
                )
        axis_aligned_enclosure = float(np.mean(np.stack(enclosed))) if enclosed else None
    valid_control_count = max(int(np.sum(executed)), 1)
    return {
        "operational_failure": operational_failure,
        "executed_degraded_any": bool(np.any(executed_degraded)),
        "failure_any": bool(np.any(failure)),
        "contact_any": bool(np.any(contact)),
        "failure_step_fraction": float(np.mean(failure)),
        "minimum_barrier_margin": float(np.min(barriers)),
        "post_change_minimum_barrier_margin": float(np.min(barriers[change_index:])),
        "post_change_failure_fraction": float(np.mean(failure[change_index:])),
        "tracking_position_rmse": float(np.sqrt(np.mean(tracking**2))),
        "normalized_estimation_rmse": float(np.sqrt(np.mean(normalized_error**2))),
        "degraded_fraction": float(np.sum(executed_degraded) / valid_control_count),
        "proposal_acceptance_fraction": float(
            np.sum(arrays["proposal_accepted"] & executed) / valid_control_count
        ),
        "fallback_use_fraction": float(
            np.sum(arrays["used_fallback"] & executed) / valid_control_count
        ),
        "fallback_acceptance_fraction": float(
            np.sum(arrays["fallback_accepted"] & executed) / valid_control_count
        ),
        "axis_aligned_sample_range_enclosure_fraction": axis_aligned_enclosure,
        "accepted_estimator_updates": int(np.sum(arrays["translation_update_status"] == 0)),
        "accepted_adaptations": int(np.sum(arrays["adaptation_publication_accepted"])),
        "mean_controller_seconds": float(
            np.sum(arrays["controller_seconds"]) / valid_control_count
        ),
        "maximum_controller_seconds": float(np.max(arrays["controller_seconds"])),
        "mean_estimator_seconds_per_executed_control": float(
            np.sum(arrays["estimator_seconds"]) / valid_control_count
        ),
        "mean_adaptation_seconds_per_executed_control": float(
            np.sum(arrays["adaptation_seconds"]) / valid_control_count
        ),
    }


def execute_dynamics_knowledge_trial(
    *,
    fold: int,
    variant: DynamicsKnowledgeVariant,
    profile: DynamicsKnowledgeProfile,
    tape: ScenarioTape,
    resources: ExperimentResources,
    startup: _StartupPreparation,
    executables: _CompiledExecutables,
    adaptation_pool: Any,
) -> _TrialExecution:
    """Execute one real plant trace with causal boundary ordering and synchronous adaptation."""
    profile.validate()
    config = profile.trial
    if executables.point is None or executables.plant is None or executables.estimator is None:
        raise RuntimeError("knowledge executables must be compiled before trial execution")
    started = time.perf_counter()
    steps = config.control_steps
    policy_count = config.policy_count
    state = experiment_core._initial_state(tape)
    previous_policy = jnp.asarray(-1, dtype=jnp.int32)
    store = ActiveSnapshotStore(startup.active)
    estimator = _causal_estimator(resources)
    estimator_history: list[tuple[np.ndarray, ...]] = []
    model_last_observation = -1
    controller_model = _controller_model(resources, estimator)
    runtime_samples = (
        _model_samples(resources, estimator, variant.runtime_dynamics_samples)
        if variant.runtime_dynamics_samples
        else None
    )
    if runtime_samples is not None:
        _require_valid_runtime_samples(runtime_samples, variant.runtime_dynamics_samples)
    admission_samples = _model_samples(resources, estimator, ADAPTATION_VALIDATION_SAMPLES)
    true_pairs = tuple(
        experiment_core._true_model(resources.model, tape, ConditionID.DYNAMICS_CHANGE, index)
        for index in range(steps)
    )
    true_vectors = np.stack(
        [_parameter_vector(model, efficiency) for model, efficiency in true_pairs]
    )
    oracle_regime = 0
    oracle_vector = true_vectors[0].copy()
    if variant.privileged_oracle_upper_bound:
        controller_model = true_pairs[0][0]
        oracle_state = _oracle_estimator_state(resources, controller_model, model_version=0)
        admission_samples = _model_samples(resources, oracle_state, ADAPTATION_VALIDATION_SAMPLES)

    job = experiment_core._CandidateJob(
        tape, ConditionID.DYNAMICS_CHANGE, resources, config, adaptation_pool
    )
    states = np.zeros((steps, 13), dtype=np.float64)
    state_valid = np.zeros((steps,), dtype=bool)
    command_valid = np.zeros((steps,), dtype=bool)
    commanded_motor = np.zeros((steps, 4), dtype=np.float64)
    realized_motor = np.zeros((steps, 4), dtype=np.float64)
    nominal_motor = np.zeros((steps, 4), dtype=np.float64)
    policy_hard = np.zeros((steps, policy_count), dtype=np.float64)
    selected_policy = np.full((steps,), -1, dtype=np.int32)
    selected_hard = np.zeros((steps,), dtype=np.float64)
    degraded = np.zeros((steps,), dtype=bool)
    proposal_accepted = np.zeros((steps,), dtype=bool)
    fallback_accepted = np.zeros((steps,), dtype=bool)
    used_fallback = np.zeros((steps,), dtype=bool)
    interval_margin = np.zeros((steps,), dtype=np.float64)
    next_value = np.zeros((steps,), dtype=np.float64)
    exact_residual = np.zeros((steps,), dtype=np.float64)
    controller_seconds = np.zeros((steps,), dtype=np.float64)
    plant_seconds = np.zeros((steps,), dtype=np.float64)
    estimator_seconds = np.zeros((steps,), dtype=np.float64)
    adaptation_seconds = np.zeros((steps,), dtype=np.float64)
    translation_status = np.full((steps,), -1, dtype=np.int16)
    rotor_status = np.full((steps,), -1, dtype=np.int16)
    estimator_model_version = np.zeros((steps,), dtype=np.int32)
    snapshot_version = np.zeros((steps,), dtype=np.int32)
    model_last_observation_trace = np.full((steps,), -1, dtype=np.int32)
    history_count = np.zeros((steps,), dtype=np.int32)
    estimated_parameters = np.zeros((steps, DYNAMICS_PARAMETER_COUNT), dtype=np.float64)
    sample_parameters = np.zeros(
        (steps, MAX_RUNTIME_SAMPLES, DYNAMICS_PARAMETER_COUNT), dtype=np.float64
    )
    sample_valid = np.zeros((steps, MAX_RUNTIME_SAMPLES), dtype=bool)
    adaptation_accepted = np.zeros((steps,), dtype=bool)
    adaptation_events: list[dict[str, Any]] = []

    for index in range(steps - 1):
        true_model, true_efficiency = true_pairs[index]
        if variant.privileged_oracle_upper_bound:
            controller_model = true_model
            estimated_parameters[index] = true_vectors[index]
            model_last_observation_trace[index] = index
            estimator_model_version[index] = oracle_regime
        else:
            physical = physical_parameters(estimator)
            estimated_parameters[index] = _parameter_vector(
                controller_model, physical.rotor_efficiency
            )
            model_last_observation_trace[index] = model_last_observation
            estimator_model_version[index] = int(np.asarray(estimator.model_version))
            if model_last_observation > index - 1:
                raise RuntimeError("causal estimator used an observation from the future")
        if runtime_samples is not None:
            count = variant.runtime_dynamics_samples
            sample_parameters[index, :count] = _sample_parameter_vectors(runtime_samples)
            sample_valid[index, :count] = np.asarray(runtime_samples.sample_valid, dtype=bool)
        states[index] = np.asarray(state)
        state_valid[index] = True
        history_count[index] = len(estimator_history)
        snapshot_version[index] = store.active.version
        target_position = jnp.asarray(tape.defender_reference_position[index], dtype=state.dtype)
        target_velocity = jnp.asarray(tape.defender_reference_velocity[index], dtype=state.dtype)
        window = dynamic_sphere_window_from_tape(
            tape,
            start_index=index,
            horizon=config.certificate_horizon + 1,
            speed_limit=config.speed_limit,
            angular_rate_max=config.angular_rate_max,
            tilt_max_radians=config.tilt_max_radians,
        )
        params = jax.tree.map(jnp.asarray, store.active.params)
        control_start = time.perf_counter()
        if variant.runtime_dynamics_samples:
            assert runtime_samples is not None
            output = executables.cartesian[variant.runtime_dynamics_samples](
                state,
                target_position,
                target_velocity,
                previous_policy,
                params,
                window,
                controller_model,
                runtime_samples,
            )
        else:
            output = executables.point(
                state,
                target_position,
                target_velocity,
                previous_policy,
                params,
                window,
                controller_model,
            )
        _block(output)
        controller_seconds[index] = time.perf_counter() - control_start
        if not np.all(np.isfinite(np.asarray(output.motor_forces))):
            raise RuntimeError("discrete DA-PLCBF controller returned nonfinite motor forces")
        plant_start = time.perf_counter()
        next_state_device, realized_device = executables.plant(
            state, output.motor_forces, true_model, true_efficiency
        )
        _block((next_state_device, realized_device))
        plant_seconds[index] = time.perf_counter() - plant_start
        if not np.all(np.isfinite(np.asarray(next_state_device))):
            raise RuntimeError("true direct-wrench plant produced a nonfinite state")

        command_valid[index] = True
        commanded_motor[index] = np.asarray(output.motor_forces)
        realized_motor[index] = np.asarray(realized_device)
        nominal_motor[index] = np.asarray(output.nominal_motor_forces)
        policy_hard[index] = np.asarray(output.policy_hard_values)
        selected_policy[index] = int(np.asarray(output.selected_policy))
        selected_hard[index] = float(np.asarray(output.selected_hard_value))
        degraded[index] = bool(np.asarray(output.degraded))
        proposal_accepted[index] = bool(np.asarray(output.proposal_accepted))
        fallback_accepted[index] = bool(np.asarray(output.fallback_accepted))
        used_fallback[index] = bool(np.asarray(output.used_fallback))
        interval_margin[index] = float(np.asarray(output.interval_margin))
        next_value[index] = float(np.asarray(output.next_value))
        exact_residual[index] = float(np.asarray(output.exact_residual))
        previous_policy = output.selected_policy

        estimator_history.append(
            experiment_core._estimator_history_entry(
                state,
                next_state_device,
                output.motor_forces,
                realized_device,
                true_model,
                resources.actuator,
                tape,
                index,
                config.dt,
            )
        )
        state = next_state_device
        boundary = index + 1
        if (
            variant.causal_estimator_history_only
            and boundary % config.estimator_interval_steps == 0
        ):
            observations = experiment_core._estimator_observations(
                estimator_history, config.estimator_window_steps
            )
            estimator_start = time.perf_counter()
            translation_update, rotor_update = executables.estimator(
                estimator, observations[0], observations[1], jnp.asarray(index, dtype=jnp.int32)
            )
            _block((translation_update, rotor_update))
            estimator_seconds[index] = time.perf_counter() - estimator_start
            translation_status[index] = int(np.asarray(translation_update.status))
            rotor_status[index] = int(np.asarray(rotor_update.status))
            previous_version = int(np.asarray(estimator.model_version))
            estimator = translation_update.state
            next_model_version = int(np.asarray(estimator.model_version))
            if next_model_version > previous_version:
                model_last_observation = index
                if next_model_version > store.model_version:
                    store.advance_model_version(next_model_version)
            controller_model = _controller_model(resources, estimator)
            admission_samples = _model_samples(resources, estimator, ADAPTATION_VALIDATION_SAMPLES)
            runtime_samples = (
                _model_samples(resources, estimator, variant.runtime_dynamics_samples)
                if variant.runtime_dynamics_samples
                else None
            )
            if runtime_samples is not None:
                _require_valid_runtime_samples(runtime_samples, variant.runtime_dynamics_samples)
        elif variant.privileged_oracle_upper_bound:
            next_true_model, _ = true_pairs[boundary]
            next_vector = true_vectors[boundary]
            if not np.array_equal(next_vector, oracle_vector):
                oracle_regime += 1
                oracle_vector = next_vector.copy()
                store.advance_model_version(oracle_regime)
            controller_model = next_true_model
            oracle_state = _oracle_estimator_state(
                resources, next_true_model, model_version=oracle_regime
            )
            admission_samples = _model_samples(
                resources, oracle_state, ADAPTATION_VALIDATION_SAMPLES
            )

        if boundary % config.adaptation_interval_steps == 0 and boundary < steps - 1:
            adaptation = _periodic_adaptation(
                boundary=boundary,
                state=state,
                controller_model=controller_model,
                admission_samples=admission_samples,
                store=store,
                job=job,
                tape=tape,
                profile=profile,
            )
            adaptation_events.append(adaptation)
            adaptation_seconds[index] = float(adaptation["execution_seconds"])
            adaptation_accepted[index] = bool(adaptation["publication_accepted"])

    terminal = steps - 1
    states[terminal] = np.asarray(state)
    state_valid[terminal] = True
    history_count[terminal] = len(estimator_history)
    snapshot_version[terminal] = store.active.version
    if variant.privileged_oracle_upper_bound:
        estimated_parameters[terminal] = true_vectors[terminal]
        model_last_observation_trace[terminal] = terminal
        estimator_model_version[terminal] = oracle_regime
    else:
        physical = physical_parameters(estimator)
        estimated_parameters[terminal] = _parameter_vector(
            controller_model, physical.rotor_efficiency
        )
        model_last_observation_trace[terminal] = model_last_observation
        estimator_model_version[terminal] = int(np.asarray(estimator.model_version))
        if runtime_samples is not None:
            count = variant.runtime_dynamics_samples
            sample_parameters[terminal, :count] = _sample_parameter_vectors(runtime_samples)
            sample_valid[terminal, :count] = np.asarray(runtime_samples.sample_valid, dtype=bool)
    barriers, contact, failure = experiment_core._barrier_trace(states, tape, config)
    arrays = {
        "states": states,
        "state_valid": state_valid,
        "command_valid": command_valid,
        "commanded_motor_forces": commanded_motor,
        "realized_motor_forces": realized_motor,
        "nominal_motor_forces": nominal_motor,
        "policy_hard_values": policy_hard,
        "selected_policy": selected_policy,
        "selected_hard_value": selected_hard,
        "degraded": degraded,
        "proposal_accepted": proposal_accepted,
        "fallback_accepted": fallback_accepted,
        "used_fallback": used_fallback,
        "applied_interval_margin": interval_margin,
        "applied_next_value": next_value,
        "applied_exact_residual": exact_residual,
        "controller_seconds": controller_seconds,
        "plant_seconds": plant_seconds,
        "estimator_seconds": estimator_seconds,
        "adaptation_seconds": adaptation_seconds,
        "translation_update_status": translation_status,
        "rotor_update_status": rotor_status,
        "estimator_model_version": estimator_model_version,
        "snapshot_version": snapshot_version,
        "model_last_observation_transition": model_last_observation_trace,
        "estimator_history_count": history_count,
        "true_parameters": true_vectors,
        "estimated_parameters": estimated_parameters,
        "runtime_sample_parameters": sample_parameters,
        "runtime_sample_valid": sample_valid,
        "adaptation_publication_accepted": adaptation_accepted,
        "barrier_margins": barriers,
        "contact": contact,
        "failure": failure,
    }
    metrics = _knowledge_metrics(
        arrays,
        tape,
        profile,
        startup_adaptation_status=str(startup.record["status"]),
        adaptation_events=tuple(adaptation_events),
    )
    execution_seconds = time.perf_counter() - started
    return _TrialExecution(
        fold,
        variant,
        tape.sha256,
        arrays,
        metrics,
        tuple(adaptation_events),
        store.active.params,
        execution_seconds,
    )


def run_dynamics_knowledge_campaign(
    config: DynamicsKnowledgeCampaignConfig,
    output: str | os.PathLike[str],
    *,
    repository: str | os.PathLike[str] | None = None,
    resume: bool = True,
) -> DynamicsKnowledgeCampaignRun:
    """Run or resume the exact four-way matched closed-loop campaign.

    A scheduled execution exception is appended as a typed failure outcome.  It is never retried
    implicitly on resume and never dropped from aggregate completeness accounting.
    """
    config.validate()
    profile = config.resolved_profile()
    root = Path(output).resolve()
    repository_path = _repository_root(repository)
    root.mkdir(parents=True, exist_ok=True)
    configuration = _configuration_mapping(config, profile, repository_path)
    config_path = root / "config.json"
    manifest_path = root / "manifest.json"
    marker_path = root / "complete.marker"
    if marker_path.exists():
        if not manifest_path.is_file():
            raise ValueError("completed campaign marker exists without its bound manifest")
        if not resume:
            raise FileExistsError("finalized dynamics-knowledge campaign exists and resume=False")
        if not config_path.is_file() or _read_object(config_path) != configuration:
            raise ValueError("finalized dynamics-knowledge campaign config does not match")
        verification = verify_dynamics_knowledge_campaign(
            root, repository=repository_path, require_current_source=True
        )
        if not verification.valid:
            details = "; ".join(verification.errors[:4])
            raise ValueError(
                f"finalized dynamics-knowledge campaign failed verification: {details}"
            )
        return DynamicsKnowledgeCampaignRun(
            root=root,
            expected_outcomes=verification.expected_outcomes,
            completed_outcomes=verification.completed_outcomes,
            failed_outcomes=verification.failed_outcomes,
            execution_complete=(verification.retained_outcomes == verification.expected_outcomes),
            manifest_sha256=_file_sha256(manifest_path),
            operational_failures=verification.operational_failures,
        )
    if config_path.exists():
        if _read_object(config_path) != configuration:
            raise ValueError("existing dynamics-knowledge campaign config does not match")
        if not resume:
            raise FileExistsError("dynamics-knowledge campaign exists and resume=False")
    else:
        _atomic_json(config_path, configuration)

    records = _read_outcomes(root / "outcomes.jsonl")
    known = {_outcome_key(record): record for record in records}
    provenance_path = root / "provenance.json"
    stored_provenance: dict[str, Any] | None = None
    if provenance_path.exists():
        stored_provenance = _read_object(provenance_path)
        _validate_runtime_provenance(stored_provenance, configuration)
    elif records:
        raise ValueError("existing outcomes lack the immutable pre-execution runtime provenance")
    compiled: _CompiledExecutables | None = None
    adaptation_pool: Any | None = None
    resume_plant_replay: Any | None = None
    for fold in range(config.fold_start, config.fold_start + profile.trials):
        tape_path = root / "tapes" / f"fold-{fold:04d}.npz"
        expected_tape = generate_matched_dynamics_tape(
            profile, root_seed=config.root_seed, fold=fold
        )
        if tape_path.exists():
            tape = load_scenario_tape(tape_path)
            if tape.sha256 != expected_tape.sha256:
                raise ValueError(f"existing tape differs from deterministic schedule: fold {fold}")
        else:
            tape_path.parent.mkdir(parents=True, exist_ok=True)
            save_scenario_tape(expected_tape, tape_path)
            tape = expected_tape

        resources = build_matched_resources(profile, tape, root_seed=config.root_seed, fold=fold)
        if any((fold, variant.variant_id) in known for variant in VARIANTS):
            if resume_plant_replay is None:
                resume_plant_replay = _plant_replay_function(resources, profile)
        pending = tuple(variant for variant in VARIANTS if (fold, variant.variant_id) not in known)
        if not pending:
            _startup, startup_mapping = _load_common_startup(
                root, fold, tape.sha256, resources, recover_missing_sidecar=True
            )
            for variant in VARIANTS:
                _verify_resumable_record(
                    root,
                    known[(fold, variant.variant_id)],
                    profile=profile,
                    tape=tape,
                    startup=startup_mapping,
                    variant=variant,
                    resources=resources,
                    plant_replay=resume_plant_replay,
                )
            continue

        if compiled is None:
            compiled = compile_knowledge_executables(_CompiledExecutables(resources, profile), tape)
            if stored_provenance is None:
                generated_provenance = _runtime_provenance(compiled, configuration)
                _write_once_json(provenance_path, generated_provenance)
                stored_provenance = _read_object(provenance_path)
                _validate_runtime_provenance(stored_provenance, configuration)
        if adaptation_pool is None:
            adaptation_pool = experiment_core._build_bptt_executable_pool(resources, profile.trial)
        startup_json, startup_bundle = _startup_paths(root, fold)
        if startup_json.exists() or startup_bundle.exists():
            startup, startup_mapping = _load_common_startup(
                root, fold, tape.sha256, resources, recover_missing_sidecar=True
            )
        else:
            prepared = prepare_common_startup_adaptation(profile, tape, resources, adaptation_pool)
            _save_or_verify_startup(root, fold, tape.sha256, prepared, resources)
            startup, startup_mapping = _load_common_startup(
                root, fold, tape.sha256, resources, recover_missing_sidecar=False
            )

        for variant in VARIANTS:
            key = (fold, variant.variant_id)
            if key in known:
                _verify_resumable_record(
                    root,
                    known[key],
                    profile=profile,
                    tape=tape,
                    startup=startup_mapping,
                    variant=variant,
                    resources=resources,
                    plant_replay=resume_plant_replay,
                )
                continue
            try:
                execution = execute_dynamics_knowledge_trial(
                    fold=fold,
                    variant=variant,
                    profile=profile,
                    tape=tape,
                    resources=resources,
                    startup=startup,
                    executables=compiled,
                    adaptation_pool=adaptation_pool,
                )
                record = _successful_outcome(root, profile, startup, execution)
            except Exception as error:  # noqa: BLE001 - retained scientific failure.
                record = _failed_outcome(fold, variant, tape.sha256, startup, error)
            known[key] = record
            _flush_outcomes(root / "outcomes.jsonl", known)

    outcomes = _ordered_outcomes(known)
    if stored_provenance is None:
        raise RuntimeError("campaign produced no immutable runtime provenance")
    _validate_runtime_provenance(stored_provenance, configuration)
    aggregates = aggregate_dynamics_knowledge_outcomes(config, profile, outcomes)
    _atomic_json(root / "aggregates.json", aggregates)
    expected = profile.trials * len(VARIANTS)
    completed = sum(record.get("status") == "complete" for record in outcomes)
    failed = sum(record.get("status") == "failed" for record in outcomes)
    operational_failures = sum(bool(record.get("operational_failure")) for record in outcomes)
    startup_failures, periodic_adaptation_failures = _adaptation_failure_counts(outcomes)
    execution_complete = len(outcomes) == expected
    manifest = _manifest_mapping(
        root,
        configuration,
        expected=expected,
        completed=completed,
        failed=failed,
        operational_failures=operational_failures,
        adaptation_execution_failures=startup_failures + periodic_adaptation_failures,
        execution_complete=execution_complete,
        profile=profile,
    )
    _atomic_json(root / "manifest.json", manifest)
    manifest_hash = _file_sha256(root / "manifest.json")
    precommit = verify_dynamics_knowledge_campaign(
        root,
        repository=repository_path,
        require_current_source=True,
        require_completion_marker=False,
    )
    if not precommit.valid:
        details = "; ".join(precommit.errors[:4])
        raise RuntimeError(f"dynamics-knowledge precommit verification failed: {details}")
    _atomic_json(
        root / "complete.marker",
        {
            "schema_version": COMPLETION_MARKER_SCHEMA_VERSION,
            "manifest_sha256": manifest_hash,
            "execution_complete": execution_complete,
            "retained_failures": failed,
            "operational_failures": operational_failures,
            "adaptation_execution_failures": (startup_failures + periodic_adaptation_failures),
            "blanket_safety_superiority_supported": False,
            "operational_failure_definition": OPERATIONAL_FAILURE_DEFINITION,
            "axis_aligned_sample_range_metric_definition": (
                AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION
            ),
        },
    )
    return DynamicsKnowledgeCampaignRun(
        root, expected, completed, failed, execution_complete, manifest_hash, operational_failures
    )


def _successful_outcome(
    root: Path,
    profile: DynamicsKnowledgeProfile,
    startup: _StartupPreparation,
    execution: _TrialExecution,
) -> dict[str, Any]:
    trace_path, trace_sha256, trace_digest, final_leaf_digest = _save_trial_trace(
        root, profile, startup, execution
    )
    return {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "scope": "closed_loop_dynamics_knowledge",
        "claim_boundary": CLAIM_BOUNDARY,
        "fold": execution.fold,
        "variant": asdict(execution.variant),
        "status": "complete",
        "tape_digest": execution.tape_digest,
        "startup_semantic_digest": _startup_semantic_digest(startup.record),
        "startup_active_digest": startup.active.digest,
        "startup_active_params_digest": startup.active.params_digest,
        "startup_adaptation_status": startup.record["status"],
        "trace_artifact": trace_path.relative_to(root).as_posix(),
        "trace_artifact_sha256": trace_sha256,
        "trace_content_digest": trace_digest,
        "final_parameter_leaf_digest": final_leaf_digest,
        "metrics": execution.metrics,
        "operational_failure": bool(execution.metrics["operational_failure"]),
        "adaptation_events": list(execution.adaptation_events),
        "execution_seconds": execution.execution_seconds,
        "filter_implementation": FILTER_IMPLEMENTATION_ID,
        "training_objective": TRAINING_OBJECTIVE_ID,
        "training_claim": TRAINING_CLAIM,
        "uncertainty_aware_bptt_training": False,
        "axis_aligned_sample_range_metric_definition": (
            AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION
        ),
        "oracle_label": ORACLE_LABEL if execution.variant.variant_id == "oracle_point" else None,
        "blanket_safety_superiority_supported": False,
    }


def _failed_outcome(
    fold: int,
    variant: DynamicsKnowledgeVariant,
    tape_digest: str,
    startup: _StartupPreparation,
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "scope": "closed_loop_dynamics_knowledge",
        "claim_boundary": CLAIM_BOUNDARY,
        "fold": fold,
        "variant": asdict(variant),
        "status": "failed",
        "tape_digest": tape_digest,
        "startup_semantic_digest": _startup_semantic_digest(startup.record),
        "startup_active_digest": startup.active.digest,
        "startup_active_params_digest": startup.active.params_digest,
        "startup_adaptation_status": startup.record["status"],
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "trace_artifact": None,
        "trace_artifact_sha256": None,
        "trace_content_digest": None,
        "final_parameter_leaf_digest": None,
        "metrics": None,
        "operational_failure": True,
        "adaptation_events": None,
        "execution_seconds": None,
        "filter_implementation": FILTER_IMPLEMENTATION_ID,
        "training_objective": TRAINING_OBJECTIVE_ID,
        "training_claim": TRAINING_CLAIM,
        "uncertainty_aware_bptt_training": False,
        "axis_aligned_sample_range_metric_definition": (
            AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION
        ),
        "oracle_label": ORACLE_LABEL if variant.variant_id == "oracle_point" else None,
        "blanket_safety_superiority_supported": False,
    }


def _save_trial_trace(
    root: Path,
    profile: DynamicsKnowledgeProfile,
    startup: _StartupPreparation,
    execution: _TrialExecution,
) -> tuple[Path, str, str, str]:
    directory = root / "traces" / f"fold-{execution.fold:04d}"
    path = directory / f"{execution.variant.variant_id}.npz"
    metadata = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "scope": "closed_loop_dynamics_knowledge",
        "claim_boundary": CLAIM_BOUNDARY,
        "fold": execution.fold,
        "variant": asdict(execution.variant),
        "profile": profile.name,
        "trial_shape": {
            "control_steps": profile.trial.control_steps,
            "policy_count": profile.trial.policy_count,
            "certificate_horizon": profile.trial.certificate_horizon,
            "training_scenario_count": profile.trial.training_scenario_count,
        },
        "tape_digest": execution.tape_digest,
        "startup_semantic_digest": _startup_semantic_digest(startup.record),
        "startup_active_digest": startup.active.digest,
        "startup_active_params_digest": startup.active.params_digest,
        "filter_implementation": FILTER_IMPLEMENTATION_ID,
        "training_objective": TRAINING_OBJECTIVE_ID,
        "uncertainty_aware_bptt_training": False,
        "axis_aligned_sample_range_metric_definition": (
            AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION
        ),
    }
    metadata_bytes = _canonical_json(metadata)
    arrays = {name: np.asarray(value) for name, value in execution.arrays.items()}
    arrays["metadata_json_utf8"] = np.frombuffer(metadata_bytes, dtype=np.uint8).copy()
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(execution.final_params)]
    for index, leaf in enumerate(leaves):
        arrays[f"final_param_leaf_{index:03d}"] = leaf
    final_leaf_digest = _canonical_array_digest(
        "dynamics-knowledge-final-parameter-leaves-v1",
        {name: value for name, value in arrays.items() if name.startswith("final_param_leaf_")},
    )
    content_digest = _canonical_array_digest("dynamics-knowledge-trace-v2", arrays)
    arrays["content_digest"] = np.asarray(content_digest)
    arrays["final_parameter_leaf_digest"] = np.asarray(final_leaf_digest)
    _atomic_npz(path, arrays)
    return path, _file_sha256(path), content_digest, final_leaf_digest


def _load_trace_payload(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], str, str]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if len(loaded.files) != len(set(loaded.files)):
                raise ValueError("trace has duplicate members")
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    except (OSError, ValueError) as error:
        raise ValueError("trace is not a valid non-object NPZ archive") from error
    allowed = {
        *_TRACE_SCIENTIFIC_ARRAY_NAMES,
        "metadata_json_utf8",
        "content_digest",
        "final_parameter_leaf_digest",
    }
    unexpected = {
        name for name in arrays if name not in allowed and not name.startswith("final_param_leaf_")
    }
    if unexpected:
        raise ValueError(f"trace contains unexpected members: {sorted(unexpected)}")
    try:
        stored_content = str(np.asarray(arrays.pop("content_digest")).item())
        stored_leaves = str(np.asarray(arrays.pop("final_parameter_leaf_digest")).item())
        metadata_array = arrays["metadata_json_utf8"]
        if metadata_array.ndim != 1 or metadata_array.dtype != np.dtype(np.uint8):
            raise ValueError("metadata_json_utf8 must be a one-dimensional uint8 array")
        metadata_bytes = metadata_array.tobytes()
        metadata = _json_loads_object(metadata_bytes)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("trace metadata or stored digest is malformed") from error
    content = _canonical_array_digest("dynamics-knowledge-trace-v2", arrays)
    if content != stored_content:
        raise ValueError("trace semantic content digest mismatch")
    leaf_arrays = {
        name: value for name, value in arrays.items() if name.startswith("final_param_leaf_")
    }
    if not leaf_arrays:
        raise ValueError("trace lacks final parameter leaves")
    leaf_digest = _canonical_array_digest(
        "dynamics-knowledge-final-parameter-leaves-v1", leaf_arrays
    )
    if leaf_digest != stored_leaves:
        raise ValueError("trace final parameter-leaf digest mismatch")
    return arrays, metadata, content, leaf_digest


def _startup_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "status",
        "candidate_digest",
        "candidate_params_digest",
        "report_digest",
        "report_integrity_verified",
        "report_passed",
        "failed_gate_names",
        "publication_accepted",
        "publication_reason",
        "active_digest",
        "active_params_digest",
        "active_version",
        "model_version",
        "failure_type",
        "failure_message",
        "training_objective",
        "uncertainty_aware_bptt_training",
        "hard_admission_dynamics_samples",
    )
    return {name: record.get(name) for name in fields}


def _startup_semantic_digest(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        b"crazyflow.da_plcbf.dynamics-knowledge-startup.v1\0"
        + _canonical_json(_startup_identity(record))
    ).hexdigest()


def _startup_paths(root: Path, fold: int) -> tuple[Path, Path]:
    common = root / "common"
    stem = f"fold-{fold:04d}-startup"
    return common / f"{stem}.json", common / f"{stem}-active.npz"


def _startup_snapshot_header(active: PolicySnapshot) -> dict[str, Any]:
    return {
        "kind": active.kind,
        "version": active.version,
        "base_active_version": active.base_active_version,
        "base_active_digest": active.base_active_digest,
        "model_version": active.model_version,
        "digest": active.digest,
        "params_digest": active.params_digest,
        "params_schema_digest": active.params_schema_digest,
        "structural_core_digest": active.structural_core_digest,
        "metadata": _json_safe(active.metadata),
    }


def _startup_bundle_arrays(
    fold: int, tape_digest: str, startup: _StartupPreparation
) -> dict[str, np.ndarray]:
    active = startup.active
    if active.kind != "active" or not active.verify_integrity() or not active.all_finite():
        raise ValueError(f"common startup active snapshot is invalid: fold {fold}")
    params, _params_treedef = jax.tree_util.tree_flatten(active.params)
    structural_core, _core_treedef = jax.tree_util.tree_flatten(active.structural_core)
    record = _json_safe(startup.record)
    if not isinstance(record, dict):
        raise TypeError("common startup record must be an object")
    metadata = {
        "schema_version": STARTUP_BUNDLE_SCHEMA_VERSION,
        "fold": fold,
        "tape_digest": tape_digest,
        "record": record,
        "record_digest": _record_digest(
            "crazyflow.da_plcbf.dynamics-knowledge-startup-record.v1", record
        ),
        "semantic_identity": _startup_identity(record),
        "semantic_digest": _startup_semantic_digest(record),
        "active_snapshot": _startup_snapshot_header(active),
        "parameter_leaf_count": len(params),
        "structural_core_leaf_count": len(structural_core),
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json_utf8": np.frombuffer(_canonical_json(metadata), dtype=np.uint8).copy()
    }
    arrays.update(
        {f"param_leaf_{index:03d}": np.asarray(leaf) for index, leaf in enumerate(params)}
    )
    arrays.update(
        {
            f"structural_core_leaf_{index:03d}": np.asarray(leaf)
            for index, leaf in enumerate(structural_core)
        }
    )
    content_digest = _canonical_array_digest(
        "crazyflow.da_plcbf.dynamics-knowledge-startup-active.v1", arrays
    )
    arrays["content_digest"] = np.asarray(content_digest)
    return arrays


def _startup_sidecar(
    root: Path, bundle_path: Path, metadata: Mapping[str, Any], content_digest: str
) -> dict[str, Any]:
    return {
        "schema_version": STARTUP_SCHEMA_VERSION,
        "fold": metadata["fold"],
        "tape_digest": metadata["tape_digest"],
        "record": metadata["record"],
        "record_digest": metadata["record_digest"],
        "semantic_identity": metadata["semantic_identity"],
        "semantic_digest": metadata["semantic_digest"],
        "active_snapshot": metadata["active_snapshot"],
        "active_artifact": bundle_path.relative_to(root).as_posix(),
        "active_artifact_sha256": _file_sha256(bundle_path),
        "active_content_digest": content_digest,
    }


def _load_startup_bundle(
    path: Path, *, fold: int, tape_digest: str, resources: ExperimentResources
) -> tuple[_StartupPreparation, dict[str, Any], str]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("startup bundle has duplicate members")
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
        stored_content = str(np.asarray(arrays.pop("content_digest")).item())
        metadata_array = arrays["metadata_json_utf8"]
        if metadata_array.ndim != 1 or metadata_array.dtype != np.dtype(np.uint8):
            raise ValueError("metadata_json_utf8 must be a one-dimensional uint8 array")
        metadata = _json_loads_object(metadata_array.tobytes())
    except (EOFError, KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"common startup active bundle is malformed: fold {fold}") from error
    computed_content = _canonical_array_digest(
        "crazyflow.da_plcbf.dynamics-knowledge-startup-active.v1", arrays
    )
    if stored_content != computed_content:
        raise ValueError(f"common startup active content digest mismatch: fold {fold}")
    expected_metadata_keys = {
        "schema_version",
        "fold",
        "tape_digest",
        "record",
        "record_digest",
        "semantic_identity",
        "semantic_digest",
        "active_snapshot",
        "parameter_leaf_count",
        "structural_core_leaf_count",
    }
    if set(metadata) != expected_metadata_keys:
        raise ValueError(f"common startup active metadata fields changed: fold {fold}")
    if (
        metadata["schema_version"] != STARTUP_BUNDLE_SCHEMA_VERSION
        or metadata["fold"] != fold
        or metadata["tape_digest"] != tape_digest
    ):
        raise ValueError(f"common startup active header mismatch: fold {fold}")
    record = metadata["record"]
    identity = metadata["semantic_identity"]
    header = metadata["active_snapshot"]
    if (
        not isinstance(record, dict)
        or not isinstance(identity, dict)
        or not isinstance(header, dict)
    ):
        raise ValueError(f"common startup active metadata objects are malformed: fold {fold}")
    if metadata["record_digest"] != _record_digest(
        "crazyflow.da_plcbf.dynamics-knowledge-startup-record.v1", record
    ):
        raise ValueError(f"common startup record digest mismatch: fold {fold}")
    if identity != _startup_identity(record):
        raise ValueError(f"common startup semantic identity mismatch: fold {fold}")
    if metadata["semantic_digest"] != _startup_semantic_digest(record):
        raise ValueError(f"common startup semantic digest mismatch: fold {fold}")
    if record.get("status") not in {"complete", "failed"}:
        raise ValueError(f"common startup status is invalid: fold {fold}")
    if record.get("training_objective") != TRAINING_OBJECTIVE_ID:
        raise ValueError(f"common startup training objective changed: fold {fold}")
    if record.get("uncertainty_aware_bptt_training") is not False:
        raise ValueError(f"common startup incorrectly claims robust BPTT: fold {fold}")
    if record.get("hard_admission_dynamics_samples") != ADAPTATION_VALIDATION_SAMPLES:
        raise ValueError(f"common startup admission sample count changed: fold {fold}")
    if not isinstance(record.get("publication_accepted"), bool):
        raise ValueError(f"common startup publication result is malformed: fold {fold}")
    execution_seconds = record.get("execution_seconds")
    if (
        isinstance(execution_seconds, bool)
        or not isinstance(execution_seconds, (int, float))
        or not math.isfinite(float(execution_seconds))
        or float(execution_seconds) < 0.0
    ):
        raise ValueError(f"common startup execution timing is invalid: fold {fold}")
    if record["status"] == "failed" and (
        not isinstance(record.get("failure_type"), str)
        or not isinstance(record.get("failure_message"), str)
    ):
        raise ValueError(f"common startup failure is untyped: fold {fold}")
    expected_header_keys = {
        "kind",
        "version",
        "base_active_version",
        "base_active_digest",
        "model_version",
        "digest",
        "params_digest",
        "params_schema_digest",
        "structural_core_digest",
        "metadata",
    }
    if set(header) != expected_header_keys or header["kind"] != "active":
        raise ValueError(f"common startup active snapshot header changed: fold {fold}")
    integer_fields = ("version", "base_active_version", "model_version")
    if any(
        not isinstance(header[name], int) or isinstance(header[name], bool)
        for name in integer_fields
    ) or not isinstance(header["metadata"], dict):
        raise ValueError(f"common startup active snapshot header is malformed: fold {fold}")
    parameter_count = metadata["parameter_leaf_count"]
    core_count = metadata["structural_core_leaf_count"]
    if (
        not isinstance(parameter_count, int)
        or isinstance(parameter_count, bool)
        or parameter_count <= 0
        or not isinstance(core_count, int)
        or isinstance(core_count, bool)
        or core_count <= 0
    ):
        raise ValueError(f"common startup active leaf counts are malformed: fold {fold}")
    parameter_names = [f"param_leaf_{index:03d}" for index in range(parameter_count)]
    core_names = [f"structural_core_leaf_{index:03d}" for index in range(core_count)]
    if set(arrays) != {"metadata_json_utf8", *parameter_names, *core_names}:
        raise ValueError(f"common startup active leaf inventory changed: fold {fold}")
    expected_params, params_treedef = jax.tree_util.tree_flatten(resources.initial_params)
    expected_core, core_treedef = jax.tree_util.tree_flatten(resources.spec)
    if len(expected_params) != parameter_count or len(expected_core) != core_count:
        raise ValueError(f"common startup active tree structure changed: fold {fold}")

    def checked_leaves(names: list[str], templates: list[Any], label: str) -> list[np.ndarray]:
        leaves: list[np.ndarray] = []
        for name, template in zip(names, templates, strict=True):
            value = arrays[name]
            expected = np.asarray(template)
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"common startup {label} schema changed: fold {fold}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"common startup {label} contains nonfinite data: fold {fold}")
            leaves.append(value)
        return leaves

    params = params_treedef.unflatten(checked_leaves(parameter_names, expected_params, "parameter"))
    structural_core = core_treedef.unflatten(
        checked_leaves(core_names, expected_core, "structural-core")
    )
    try:
        active = create_active_snapshot(
            params,
            version=header["version"],
            model_version=header["model_version"],
            structural_core=structural_core,
            metadata=header["metadata"],
            base_active_version=header["base_active_version"],
            base_active_digest=header["base_active_digest"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"common startup active snapshot reconstruction failed: fold {fold}"
        ) from error
    initial = _initial_active_snapshot(resources)
    expected_header = _startup_snapshot_header(active)
    if header != expected_header or not active.verify_integrity() or not active.all_finite():
        raise ValueError(f"common startup active snapshot digest mismatch: fold {fold}")
    if (
        active.params_schema_digest != initial.params_schema_digest
        or active.structural_core_digest != initial.structural_core_digest
    ):
        raise ValueError(f"common startup active snapshot schema/core mismatch: fold {fold}")
    expected_record_fields = {
        "active_digest": active.digest,
        "active_params_digest": active.params_digest,
        "active_version": active.version,
        "model_version": active.model_version,
    }
    if any(record.get(name) != value for name, value in expected_record_fields.items()):
        raise ValueError(f"common startup record does not bind active snapshot: fold {fold}")
    return _StartupPreparation(active, record), metadata, stored_content


def _load_common_startup(
    root: Path,
    fold: int,
    tape_digest: str,
    resources: ExperimentResources,
    *,
    recover_missing_sidecar: bool,
) -> tuple[_StartupPreparation, dict[str, Any]]:
    sidecar_path, bundle_path = _startup_paths(root, fold)
    if not bundle_path.is_file():
        raise ValueError(f"common startup active bundle is missing: fold {fold}")
    startup, metadata, content_digest = _load_startup_bundle(
        bundle_path, fold=fold, tape_digest=tape_digest, resources=resources
    )
    expected_sidecar = _startup_sidecar(root, bundle_path, metadata, content_digest)
    if sidecar_path.exists():
        sidecar = _read_object(sidecar_path)
        if sidecar != expected_sidecar:
            raise ValueError(f"common startup sidecar does not bind active bundle: fold {fold}")
    elif recover_missing_sidecar:
        _write_once_json(sidecar_path, expected_sidecar)
        sidecar = _read_object(sidecar_path)
        if sidecar != expected_sidecar:
            raise ValueError(f"recovered common startup sidecar differs: fold {fold}")
    else:
        raise ValueError(f"common startup sidecar is missing: fold {fold}")
    return startup, sidecar


def _save_or_verify_startup(
    root: Path,
    fold: int,
    tape_digest: str,
    startup: _StartupPreparation,
    resources: ExperimentResources,
) -> None:
    sidecar_path, bundle_path = _startup_paths(root, fold)
    if sidecar_path.exists() or bundle_path.exists():
        existing, _sidecar = _load_common_startup(
            root, fold, tape_digest, resources, recover_missing_sidecar=True
        )
        if existing.active.digest != startup.active.digest or _canonical_json(
            existing.record
        ) != _canonical_json(startup.record):
            raise ValueError(f"existing common startup differs from prepared startup: fold {fold}")
        return
    arrays = _startup_bundle_arrays(fold, tape_digest, startup)
    _write_once_npz(bundle_path, arrays)
    persisted, _sidecar = _load_common_startup(
        root, fold, tape_digest, resources, recover_missing_sidecar=True
    )
    if persisted.active.digest != startup.active.digest or _canonical_json(
        persisted.record
    ) != _canonical_json(startup.record):
        raise ValueError(f"persisted common startup differs from prepared startup: fold {fold}")


def _configuration_mapping(
    config: DynamicsKnowledgeCampaignConfig, profile: DynamicsKnowledgeProfile, repository: Path
) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "experiment_id": "da-plcbf-dynamics-knowledge-closed-loop-v3",
        "scope": "closed_loop_dynamics_knowledge",
        "claim_boundary": CLAIM_BOUNDARY,
        "profile": profile.name,
        "trials": profile.trials,
        "root_seed": config.root_seed,
        "fold_start": config.fold_start,
        "intended_for_confirmatory_differences": (profile.intended_for_confirmatory_differences),
        "trial": asdict(profile.trial),
        "variants": [asdict(variant) for variant in VARIANTS],
        "filter_implementation": FILTER_IMPLEMENTATION_ID,
        "training_objective": TRAINING_OBJECTIVE_ID,
        "training_claim": TRAINING_CLAIM,
        "uncertainty_aware_bptt_training": False,
        "common_hard_admission_dynamics_samples": ADAPTATION_VALIDATION_SAMPLES,
        "oracle_label": ORACLE_LABEL,
        "oracle_uses_future_truth": False,
        "plant_replay_tolerance": {
            "relative": PLANT_REPLAY_RTOL,
            "absolute": PLANT_REPLAY_ATOL,
            "reason": "float32 scalar execution versus batched verifier replay",
        },
        "blanket_safety_superiority_supported": False,
        "operational_failure_definition": OPERATIONAL_FAILURE_DEFINITION,
        "axis_aligned_sample_range_metric_definition": (
            AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION
        ),
        "source_tree_sha256": source_tree_digest(repository),
    }


def _runtime_provenance(
    compiled: _CompiledExecutables, configuration: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        set(compiled.compile_seconds) != _RUNTIME_TIMING_KEYS
        or set(compiled.warmup_seconds) != _RUNTIME_TIMING_KEYS
    ):
        raise ValueError("compiled executable timings are incomplete")
    devices = []
    for device in jax.devices():
        devices.append(
            {
                "platform": device.platform,
                "device_kind": getattr(device, "device_kind", "unknown"),
                "id": int(device.id),
            }
        )
    return {
        "schema_version": RUNTIME_PROVENANCE_SCHEMA_VERSION,
        "source_tree_sha256": configuration["source_tree_sha256"],
        "configuration_sha256": _record_digest(
            "crazyflow.da_plcbf.dynamics-knowledge-configuration.v1", configuration
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "devices": devices,
        "compile_seconds": dict(compiled.compile_seconds),
        "warmup_seconds": dict(compiled.warmup_seconds),
        "timings_are_descriptive_not_hard_realtime_proofs": True,
    }


def _validate_runtime_provenance(
    provenance: Mapping[str, Any], configuration: Mapping[str, Any]
) -> None:
    expected_keys = {
        "schema_version",
        "source_tree_sha256",
        "configuration_sha256",
        "python",
        "platform",
        "jax_version",
        "numpy_version",
        "jax_enable_x64",
        "devices",
        "compile_seconds",
        "warmup_seconds",
        "timings_are_descriptive_not_hard_realtime_proofs",
    }
    if set(provenance) != expected_keys:
        raise ValueError("runtime provenance fields changed")
    if provenance["schema_version"] != RUNTIME_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("runtime provenance schema changed")
    if provenance["source_tree_sha256"] != configuration.get("source_tree_sha256"):
        raise ValueError("runtime provenance source-tree digest differs from configuration")
    expected_configuration_digest = _record_digest(
        "crazyflow.da_plcbf.dynamics-knowledge-configuration.v1", configuration
    )
    if provenance["configuration_sha256"] != expected_configuration_digest:
        raise ValueError("runtime provenance configuration digest mismatch")
    for name in ("python", "platform", "jax_version", "numpy_version"):
        if not isinstance(provenance[name], str) or not provenance[name]:
            raise ValueError(f"runtime provenance {name} is malformed")
    if not isinstance(provenance["jax_enable_x64"], bool):
        raise ValueError("runtime provenance x64 flag is malformed")
    devices = provenance["devices"]
    if not isinstance(devices, list) or not devices:
        raise ValueError("runtime provenance device inventory is empty or malformed")
    for device in devices:
        if not isinstance(device, dict) or set(device) != {"platform", "device_kind", "id"}:
            raise ValueError("runtime provenance device record is malformed")
        if not isinstance(device["platform"], str) or not device["platform"]:
            raise ValueError("runtime provenance device platform is malformed")
        if not isinstance(device["device_kind"], str) or not device["device_kind"]:
            raise ValueError("runtime provenance device kind is malformed")
        if not isinstance(device["id"], int) or isinstance(device["id"], bool):
            raise ValueError("runtime provenance device id is malformed")
    for timing_kind in ("compile_seconds", "warmup_seconds"):
        timings = provenance[timing_kind]
        if not isinstance(timings, dict) or set(timings) != _RUNTIME_TIMING_KEYS:
            raise ValueError(f"runtime provenance {timing_kind} is incomplete")
        for name, value in timings.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"runtime provenance {timing_kind}.{name} is malformed")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"runtime provenance {timing_kind}.{name} is invalid")
    if provenance["timings_are_descriptive_not_hard_realtime_proofs"] is not True:
        raise ValueError("runtime provenance timing-claim boundary changed")


_AGGREGATE_METRICS: tuple[tuple[str, str], ...] = (
    ("operational_failure", "lower_is_better"),
    ("failure_any", "lower_is_better"),
    ("contact_any", "lower_is_better"),
    ("minimum_barrier_margin", "higher_is_better"),
    ("post_change_minimum_barrier_margin", "higher_is_better"),
    ("degraded_fraction", "lower_is_better"),
    ("tracking_position_rmse", "lower_is_better"),
    ("normalized_estimation_rmse", "lower_is_better"),
    ("fallback_use_fraction", "lower_is_better"),
)
_COMPARISON_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("estimated_r0_oracle_gap", "estimated_r0", "oracle_point", "descriptive"),
    ("estimated_cartesian_r4_oracle_gap", "estimated_cartesian_r4", "oracle_point", "descriptive"),
    ("estimated_cartesian_r8_oracle_gap", "estimated_cartesian_r8", "oracle_point", "descriptive"),
    (
        "estimated_cartesian_r4_vs_r0",
        "estimated_cartesian_r4",
        "estimated_r0",
        "predeclared_primary",
    ),
    (
        "estimated_cartesian_r8_vs_r0",
        "estimated_cartesian_r8",
        "estimated_r0",
        "predeclared_primary",
    ),
    (
        "estimated_cartesian_r8_vs_r4",
        "estimated_cartesian_r8",
        "estimated_cartesian_r4",
        "exploratory",
    ),
)
_PRIMARY_METRICS = frozenset({"operational_failure", "minimum_barrier_margin"})
_SIGN_FLIP_RANDOMIZATIONS = 10_000


def _adaptation_failure_counts(outcomes: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    startup_failed_folds = {
        int(item["fold"]) for item in outcomes if item.get("startup_adaptation_status") == "failed"
    }
    periodic_failures = 0
    for item in outcomes:
        events = item.get("adaptation_events")
        if isinstance(events, list):
            periodic_failures += sum(
                isinstance(event, dict) and event.get("status") == "failed" for event in events
            )
    return len(startup_failed_folds), periodic_failures


def _outcome_metric_value(outcome: Mapping[str, Any], metric: str) -> float:
    if metric == "operational_failure":
        value = outcome.get("operational_failure")
        if not isinstance(value, bool):
            raise TypeError("every outcome must persist a boolean operational_failure")
        return float(value)
    metrics = outcome.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError(f"completed outcome lacks metrics for {metric}")
    return float(metrics[metric])


def aggregate_dynamics_knowledge_outcomes(
    config: DynamicsKnowledgeCampaignConfig,
    profile: DynamicsKnowledgeProfile,
    outcomes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build paired metric-level evidence without emitting a blanket safety claim."""
    folds = range(config.fold_start, config.fold_start + profile.trials)
    expected_keys = {(fold, variant.variant_id) for fold in folds for variant in VARIANTS}
    keys = [_outcome_key(item) for item in outcomes]
    exact_unique_schedule = len(keys) == len(set(keys)) and set(keys) == expected_keys
    registry = {variant.variant_id: asdict(variant) for variant in VARIANTS}
    registry_exact = all(
        key in expected_keys and item.get("variant") == registry[key[1]]
        for key, item in zip(keys, outcomes, strict=True)
    )
    complete = [item for item in outcomes if item.get("status") == "complete"]
    indexed_all = {_outcome_key(item): item for item in outcomes}
    indexed = {_outcome_key(item): item for item in complete}
    expected = profile.trials * len(VARIANTS)
    failed_count = sum(item.get("status") == "failed" for item in outcomes)
    startup_failure_count, periodic_adaptation_failure_count = _adaptation_failure_counts(outcomes)
    adaptation_execution_failures = startup_failure_count + periodic_adaptation_failure_count
    schedule_complete = exact_unique_schedule and registry_exact and len(outcomes) == expected
    confirmatory_eligible = (
        profile.intended_for_confirmatory_differences
        and schedule_complete
        and failed_count == 0
        and adaptation_execution_failures == 0
        and len(complete) == expected
    )
    comparisons: list[dict[str, Any]] = []
    for pair_label, candidate_id, reference_id, family_role in _COMPARISON_PAIRS:
        for metric, direction in _AGGREGATE_METRICS:
            candidate_values: list[float] = []
            reference_values: list[float] = []
            retained_folds: list[int] = []
            for fold in folds:
                candidate = indexed.get((fold, candidate_id))
                reference = indexed.get((fold, reference_id))
                if candidate is None or reference is None:
                    continue
                left = _outcome_metric_value(candidate, metric)
                right = _outcome_metric_value(reference, metric)
                if math.isfinite(left) and math.isfinite(right):
                    candidate_values.append(left)
                    reference_values.append(right)
                    retained_folds.append(fold)
            candidate_array = np.asarray(candidate_values, dtype=np.float64)
            reference_array = np.asarray(reference_values, dtype=np.float64)
            raw_delta = candidate_array - reference_array
            oriented = raw_delta if direction == "higher_is_better" else -raw_delta
            seed = _aggregate_seed(config.root_seed, pair_label, candidate_id, reference_id, metric)
            interval = _bootstrap_interval(oriented, seed=seed)
            pvalue = _paired_sign_flip_pvalue(oriented, seed=seed ^ 0x9E3779B97F4A7C15)
            is_primary = family_role == "predeclared_primary" and metric in _PRIMARY_METRICS
            analysis_role = (
                "confirmatory"
                if is_primary
                else "exploratory"
                if family_role == "predeclared_primary"
                else family_role
            )
            comparisons.append(
                {
                    "pair": pair_label,
                    "candidate_variant_id": candidate_id,
                    "reference_variant_id": reference_id,
                    "reference_is_privileged_oracle": reference_id == "oracle_point",
                    "analysis_role": analysis_role,
                    "metric": metric,
                    "direction": direction,
                    "paired_count": int(oriented.size),
                    "missing_scheduled_pairs": profile.trials - int(oriented.size),
                    "folds": retained_folds,
                    "candidate_mean": (
                        float(np.mean(candidate_array)) if candidate_array.size else None
                    ),
                    "reference_mean": (
                        float(np.mean(reference_array)) if reference_array.size else None
                    ),
                    "raw_paired_deltas_candidate_minus_reference": raw_delta.tolist(),
                    "oriented_improvements": oriented.tolist(),
                    "mean_oriented_improvement": (
                        float(np.mean(oriented)) if oriented.size else None
                    ),
                    "bootstrap_95_interval": list(interval) if interval is not None else None,
                    "one_sided_sign_flip_pvalue": pvalue,
                    "sign_flip_randomizations": _SIGN_FLIP_RANDOMIZATIONS,
                    "holm_adjusted_pvalue": None,
                    "confirmatory_eligible": bool(
                        confirmatory_eligible and is_primary and oriented.size == profile.trials
                    ),
                    "metric_level_improvement_supported": False,
                    "blanket_safety_superiority_interpretation_permitted": False,
                }
            )

    primary_indices = [
        index for index, item in enumerate(comparisons) if item["analysis_role"] == "confirmatory"
    ]
    adjusted = _holm_adjust(
        [comparisons[index]["one_sided_sign_flip_pvalue"] for index in primary_indices]
    )
    for index, adjusted_pvalue in zip(primary_indices, adjusted, strict=True):
        comparison = comparisons[index]
        comparison["holm_adjusted_pvalue"] = adjusted_pvalue
        interval = comparison["bootstrap_95_interval"]
        comparison["metric_level_improvement_supported"] = bool(
            comparison["confirmatory_eligible"]
            and adjusted_pvalue is not None
            and adjusted_pvalue <= 0.05
            and interval is not None
            and float(interval[0]) > 0.0
        )

    variant_summaries = []
    for variant in VARIANTS:
        complete_rows = [
            indexed[(fold, variant.variant_id)]
            for fold in folds
            if (fold, variant.variant_id) in indexed
        ]
        scheduled_rows = [
            indexed_all[(fold, variant.variant_id)]
            for fold in folds
            if (fold, variant.variant_id) in indexed_all
        ]
        metric_summary = {}
        for metric, _ in _AGGREGATE_METRICS:
            rows = scheduled_rows if metric == "operational_failure" else complete_rows
            values = np.asarray(
                [_outcome_metric_value(row, metric) for row in rows], dtype=np.float64
            )
            metric_summary[metric] = {
                "count": int(values.size),
                "mean": float(np.mean(values)) if values.size else None,
                "minimum": float(np.min(values)) if values.size else None,
                "maximum": float(np.max(values)) if values.size else None,
            }
        variant_summaries.append(
            {
                "variant": asdict(variant),
                "scheduled_outcome_count": len(scheduled_rows),
                "completed_count": len(complete_rows),
                "metrics": metric_summary,
            }
        )
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "scope": "closed_loop_dynamics_knowledge",
        "claim_boundary": CLAIM_BOUNDARY,
        "profile": profile.name,
        "scheduled_trials": profile.trials,
        "scheduled_outcomes": expected,
        "retained_outcomes": len(outcomes),
        "completed_outcomes": len(complete),
        "failed_outcomes": failed_count,
        "operational_failures": sum(bool(item.get("operational_failure")) for item in outcomes),
        "startup_adaptation_failure_folds": startup_failure_count,
        "periodic_adaptation_execution_failures": periodic_adaptation_failure_count,
        "adaptation_execution_failures": adaptation_execution_failures,
        "schedule_complete": schedule_complete,
        "confirmatory_metric_family_eligible": confirmatory_eligible,
        "confirmatory_metric_family_size": len(primary_indices),
        "confirmatory_metric_family": [
            {"pair": comparisons[index]["pair"], "metric": comparisons[index]["metric"]}
            for index in primary_indices
        ],
        "sign_flip_randomizations": _SIGN_FLIP_RANDOMIZATIONS,
        "holm_familywise_alpha": 0.05,
        "variant_summaries": variant_summaries,
        "comparisons": comparisons,
        "oracle_label": ORACLE_LABEL,
        "training_claim": TRAINING_CLAIM,
        "blanket_safety_superiority_supported": False,
        "operational_failure_definition": OPERATIONAL_FAILURE_DEFINITION,
        "axis_aligned_sample_range_metric_definition": (
            AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION
        ),
    }


def _aggregate_seed(root_seed: int, *parts: str) -> int:
    digest = hashlib.sha256(
        b"crazyflow.da_plcbf.dynamics-knowledge-aggregate.v2\0" + str(root_seed).encode("ascii")
    )
    for part in parts:
        digest.update(b"\0")
        digest.update(part.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little")


def _bootstrap_interval(values: np.ndarray, *, seed: int) -> tuple[float, float] | None:
    if values.size == 0:
        return None
    if values.size == 1 or np.all(values == values[0]):
        value = float(values[0])
        return value, value
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(_SIGN_FLIP_RANDOMIZATIONS, dtype=np.float64)
    batch = max(1, min(_SIGN_FLIP_RANDOMIZATIONS, 2_000_000 // values.size))
    for start in range(0, _SIGN_FLIP_RANDOMIZATIONS, batch):
        stop = min(start + batch, _SIGN_FLIP_RANDOMIZATIONS)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975), method="linear")
    return float(lower), float(upper)


def _paired_sign_flip_pvalue(values: np.ndarray, *, seed: int) -> float | None:
    if values.size == 0:
        return None
    observed = float(np.mean(values))
    rng = np.random.Generator(np.random.PCG64(seed))
    extreme = 0
    batch = max(1, min(_SIGN_FLIP_RANDOMIZATIONS, 2_000_000 // values.size))
    for start in range(0, _SIGN_FLIP_RANDOMIZATIONS, batch):
        stop = min(start + batch, _SIGN_FLIP_RANDOMIZATIONS)
        signs = rng.integers(0, 2, size=(stop - start, values.size), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        null_means = np.mean(signs * values[None, :], axis=1)
        extreme += int(np.sum(null_means >= observed - 1e-15))
    return float((extreme + 1) / (_SIGN_FLIP_RANDOMIZATIONS + 1))


def _holm_adjust(pvalues: Sequence[float | None]) -> list[float | None]:
    valid = [(index, float(value)) for index, value in enumerate(pvalues) if value is not None]
    adjusted: list[float | None] = [None] * len(pvalues)
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(sorted(valid, key=lambda item: item[1])):
        running = max(running, min(1.0, value * (count - rank)))
        adjusted[index] = running
    return adjusted


_TRACE_SCIENTIFIC_ARRAY_NAMES = frozenset(
    {
        "states",
        "state_valid",
        "command_valid",
        "commanded_motor_forces",
        "realized_motor_forces",
        "nominal_motor_forces",
        "policy_hard_values",
        "selected_policy",
        "selected_hard_value",
        "degraded",
        "proposal_accepted",
        "fallback_accepted",
        "used_fallback",
        "applied_interval_margin",
        "applied_next_value",
        "applied_exact_residual",
        "controller_seconds",
        "plant_seconds",
        "estimator_seconds",
        "adaptation_seconds",
        "translation_update_status",
        "rotor_update_status",
        "estimator_model_version",
        "snapshot_version",
        "model_last_observation_transition",
        "estimator_history_count",
        "true_parameters",
        "estimated_parameters",
        "runtime_sample_parameters",
        "runtime_sample_valid",
        "adaptation_publication_accepted",
        "barrier_margins",
        "contact",
        "failure",
    }
)


def verify_dynamics_knowledge_campaign(
    root: str | os.PathLike[str],
    *,
    repository: str | os.PathLike[str] | None = None,
    require_current_source: bool = True,
    require_completion_marker: bool = True,
) -> DynamicsKnowledgeVerification:
    """Strictly verify schedule, hashes, trace semantics, causality, and paired statistics.

    ``require_completion_marker=False`` is reserved for the runner's read-only precommit audit.  It
    proves every manifest-bound artifact before the atomic completion marker is created.
    """
    campaign = Path(root).resolve()
    errors: list[str] = []
    required_names = [
        "config.json",
        "outcomes.jsonl",
        "aggregates.json",
        "provenance.json",
        "manifest.json",
    ]
    if require_completion_marker:
        required_names.append("complete.marker")
    required = tuple(campaign / name for name in required_names)
    for path in required:
        if not path.is_file():
            errors.append(f"missing required artifact: {path.name}")
    if errors:
        return DynamicsKnowledgeVerification(False, tuple(errors), 0, 0, 0, 0)
    try:
        configuration = _read_object(campaign / "config.json")
        aggregates = _read_object(campaign / "aggregates.json")
        provenance = _read_object(campaign / "provenance.json")
        manifest = _read_object(campaign / "manifest.json")
        marker = _read_object(campaign / "complete.marker") if require_completion_marker else None
        outcomes = _read_outcomes(campaign / "outcomes.jsonl")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return DynamicsKnowledgeVerification(
            False, (f"artifact parse failed: {error}",), 0, 0, 0, 0
        )

    if configuration.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        errors.append("unsupported dynamics-knowledge campaign schema")
    if configuration.get("experiment_id") != "da-plcbf-dynamics-knowledge-closed-loop-v3":
        errors.append("unexpected dynamics-knowledge experiment identifier")
    if configuration.get("scope") != "closed_loop_dynamics_knowledge":
        errors.append("campaign scope changed")
    if configuration.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("campaign claim boundary changed")
    if configuration.get("blanket_safety_superiority_supported") is not False:
        errors.append("campaign must not claim blanket safety superiority")
    if configuration.get("operational_failure_definition") != OPERATIONAL_FAILURE_DEFINITION:
        errors.append("campaign operational-failure definition changed")
    if configuration.get("oracle_label") != ORACLE_LABEL:
        errors.append("privileged oracle label changed")
    if configuration.get("oracle_uses_future_truth") is not False:
        errors.append("oracle must be recorded as using no future truth")
    if configuration.get("training_objective") != TRAINING_OBJECTIVE_ID:
        errors.append("campaign training objective changed")
    if configuration.get("training_claim") != TRAINING_CLAIM:
        errors.append("campaign point-model training claim changed")
    if configuration.get("uncertainty_aware_bptt_training") is not False:
        errors.append("campaign incorrectly claims uncertainty-aware BPTT")
    if configuration.get("filter_implementation") != FILTER_IMPLEMENTATION_ID:
        errors.append("campaign does not name the nonlinear discrete filter")
    if configuration.get("plant_replay_tolerance") != {
        "relative": PLANT_REPLAY_RTOL,
        "absolute": PLANT_REPLAY_ATOL,
        "reason": "float32 scalar execution versus batched verifier replay",
    }:
        errors.append("campaign plant-replay tolerance changed")
    if configuration.get("common_hard_admission_dynamics_samples") != ADAPTATION_VALIDATION_SAMPLES:
        errors.append("campaign common hard-admission sample count changed")
    if configuration.get("axis_aligned_sample_range_metric_definition") != (
        AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION
    ):
        errors.append("campaign axis-aligned sample-range metric definition changed")

    try:
        profile_name = str(configuration["profile"])
        trials = int(configuration["trials"])
        root_seed = int(configuration["root_seed"])
        fold_start = int(configuration["fold_start"])
        campaign_config = DynamicsKnowledgeCampaignConfig(
            profile=profile_name, root_seed=root_seed, fold_start=fold_start, trials=trials
        )
        profile = campaign_config.resolved_profile()
        if configuration.get("trial") != asdict(profile.trial):
            errors.append("configured trial differs from the predeclared profile")
        if configuration.get("variants") != [asdict(variant) for variant in VARIANTS]:
            errors.append("configured variants differ from the exact four-way registry")
        if configuration.get("intended_for_confirmatory_differences") != (
            profile.intended_for_confirmatory_differences
        ):
            errors.append("configured confirmatory intent differs from the profile")
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid profile configuration: {error}")
        campaign_config = DynamicsKnowledgeCampaignConfig()
        profile = campaign_config.resolved_profile()
        trials = 0
        root_seed = 0
        fold_start = 0
    expected = trials * len(VARIANTS)
    scheduled_folds = set(range(fold_start, fold_start + trials))
    expected_variants = {variant.variant_id: variant for variant in VARIANTS}

    source_digest = configuration.get("source_tree_sha256")
    if manifest.get("source_tree_sha256") != source_digest:
        errors.append("manifest/config source-tree digests differ")
    if require_current_source:
        try:
            current = source_tree_digest(_repository_root(repository))
            if current != source_digest:
                errors.append("current source tree differs from campaign source digest")
        except (OSError, ValueError) as error:
            errors.append(f"current source-tree digest failed: {error}")

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append("unsupported dynamics-knowledge manifest schema")
    if aggregates.get("schema_version") != AGGREGATE_SCHEMA_VERSION:
        errors.append("unsupported dynamics-knowledge aggregate schema")
    if aggregates.get("operational_failure_definition") != OPERATIONAL_FAILURE_DEFINITION:
        errors.append("aggregate operational-failure definition changed")
    if aggregates.get("axis_aligned_sample_range_metric_definition") != (
        AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION
    ):
        errors.append("aggregate axis-aligned sample-range metric definition changed")
    if manifest.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("manifest claim boundary changed")
    if manifest.get("blanket_safety_superiority_supported") is not False:
        errors.append("manifest must not claim blanket safety superiority")
    if manifest.get("operational_failure_definition") != OPERATIONAL_FAILURE_DEFINITION:
        errors.append("manifest operational-failure definition changed")
    if manifest.get("axis_aligned_sample_range_metric_definition") != (
        AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION
    ):
        errors.append("manifest axis-aligned sample-range metric definition changed")
    if marker is not None:
        if marker.get("manifest_sha256") != _file_sha256(campaign / "manifest.json"):
            errors.append("completion marker does not bind the manifest")
        if marker.get("schema_version") != COMPLETION_MARKER_SCHEMA_VERSION:
            errors.append("unsupported dynamics-knowledge completion marker schema")
        if marker.get("blanket_safety_superiority_supported") is not False:
            errors.append("completion marker must not claim blanket safety superiority")
        if marker.get("operational_failure_definition") != OPERATIONAL_FAILURE_DEFINITION:
            errors.append("completion marker operational-failure definition changed")
        if marker.get("axis_aligned_sample_range_metric_definition") != (
            AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION
        ):
            errors.append("completion marker axis-aligned sample-range definition changed")
    try:
        _validate_runtime_provenance(provenance, configuration)
    except (TypeError, ValueError) as error:
        errors.append(f"runtime provenance is malformed: {error}")
    _verify_manifest_files(campaign, manifest, errors)

    tape_by_fold: dict[int, ScenarioTape] = {}
    resources_by_fold: dict[int, ExperimentResources] = {}
    true_parameters_by_fold: dict[int, np.ndarray] = {}
    plant_replay: Any | None = None
    startup_by_fold: dict[int, dict[str, Any]] = {}
    for fold in scheduled_folds:
        tape_path = campaign / "tapes" / f"fold-{fold:04d}.npz"
        try:
            tape = load_scenario_tape(tape_path)
            expected_tape = generate_matched_dynamics_tape(profile, root_seed=root_seed, fold=fold)
            if tape.sha256 != expected_tape.sha256:
                errors.append(f"tape differs from deterministic schedule: fold {fold}")
            tape_by_fold[fold] = tape
            resources = build_matched_resources(profile, tape, root_seed=root_seed, fold=fold)
            resources_by_fold[fold] = resources
            if plant_replay is None:
                plant_replay = _plant_replay_function(resources, profile)
            true_parameters_by_fold[fold] = np.stack(
                [
                    _parameter_vector(
                        *experiment_core._true_model(
                            resources.model, tape, ConditionID.DYNAMICS_CHANGE, boundary
                        )
                    )
                    for boundary in range(profile.trial.control_steps)
                ]
            )
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"invalid tape for fold {fold}: {error}")
        try:
            tape = tape_by_fold.get(fold)
            resources = resources_by_fold.get(fold)
            if tape is None or resources is None:
                raise ValueError("validated tape/resources are unavailable")
            _startup_preparation, startup = _load_common_startup(
                campaign, fold, tape.sha256, resources, recover_missing_sidecar=False
            )
            record = startup["record"]
            if record.get("training_objective") != TRAINING_OBJECTIVE_ID:
                errors.append(f"common startup training objective changed: fold {fold}")
            if record.get("uncertainty_aware_bptt_training") is not False:
                errors.append(f"common startup incorrectly claims robust BPTT: fold {fold}")
            if record.get("hard_admission_dynamics_samples") != ADAPTATION_VALIDATION_SAMPLES:
                errors.append(f"common startup admission sample count changed: fold {fold}")
            startup_by_fold[fold] = startup
        except (KeyError, OSError, TypeError, ValueError) as error:
            errors.append(f"invalid common startup for fold {fold}: {error}")

    seen: set[tuple[int, str]] = set()
    completed = 0
    failed = 0
    truth_digest_by_fold: dict[int, set[str]] = defaultdict(set)
    startup_digest_by_fold: dict[int, set[str]] = defaultdict(set)
    for record in outcomes:
        try:
            key = _outcome_key(record)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"malformed outcome key: {error}")
            continue
        if key in seen:
            errors.append(f"duplicate outcome: {key}")
            continue
        seen.add(key)
        fold, variant_id = key
        if fold not in scheduled_folds or variant_id not in expected_variants:
            errors.append(f"outcome is outside the scheduled matrix: {key}")
            continue
        variant = expected_variants[variant_id]
        startup = startup_by_fold.get(fold)
        tape = tape_by_fold.get(fold)
        if record.get("schema_version") != OUTCOME_SCHEMA_VERSION:
            errors.append(f"outcome schema changed: {key}")
        if record.get("scope") != "closed_loop_dynamics_knowledge":
            errors.append(f"outcome scope changed: {key}")
        if record.get("claim_boundary") != CLAIM_BOUNDARY:
            errors.append(f"outcome claim boundary changed: {key}")
        if record.get("variant") != asdict(variant):
            errors.append(f"outcome variant differs from registry: {key}")
        if record.get("filter_implementation") != FILTER_IMPLEMENTATION_ID:
            errors.append(f"outcome filter implementation changed: {key}")
        if record.get("training_objective") != TRAINING_OBJECTIVE_ID:
            errors.append(f"outcome training objective changed: {key}")
        if record.get("training_claim") != TRAINING_CLAIM:
            errors.append(f"outcome training claim changed: {key}")
        if record.get("uncertainty_aware_bptt_training") is not False:
            errors.append(f"outcome incorrectly claims robust BPTT: {key}")
        if record.get("axis_aligned_sample_range_metric_definition") != (
            AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION
        ):
            errors.append(f"outcome axis-aligned sample-range definition changed: {key}")
        if record.get("blanket_safety_superiority_supported") is not False:
            errors.append(f"outcome permits a blanket safety claim: {key}")
        expected_oracle_label = ORACLE_LABEL if variant_id == "oracle_point" else None
        if record.get("oracle_label") != expected_oracle_label:
            errors.append(f"outcome oracle label is wrong: {key}")
        if tape is not None and record.get("tape_digest") != tape.sha256:
            errors.append(f"outcome/tape semantic digest mismatch: {key}")
        if startup is not None:
            startup_digest = str(startup.get("semantic_digest"))
            if record.get("startup_semantic_digest") != startup_digest:
                errors.append(f"outcome/common-startup digest mismatch: {key}")
            startup_digest_by_fold[fold].add(str(record.get("startup_semantic_digest")))
        status = record.get("status")
        if status == "failed":
            failed += 1
            if record.get("operational_failure") is not True:
                errors.append(f"failed outcome is not an operational failure: {key}")
            if not isinstance(record.get("failure_type"), str) or not isinstance(
                record.get("failure_message"), str
            ):
                errors.append(f"failed outcome lacks typed retained failure: {key}")
            if record.get("trace_artifact") is not None or record.get("metrics") is not None:
                errors.append(f"failed outcome incorrectly contains successful evidence: {key}")
            continue
        if status != "complete":
            errors.append(f"unknown outcome status: {key}")
            continue
        completed += 1
        if not isinstance(record.get("operational_failure"), bool):
            errors.append(f"complete outcome lacks boolean operational failure: {key}")
        if tape is None or startup is None:
            continue
        try:
            trace_path = campaign / str(record["trace_artifact"])
            if _file_sha256(trace_path) != record.get("trace_artifact_sha256"):
                errors.append(f"trace file hash mismatch: {key}")
            arrays, metadata, trace_digest, leaf_digest = _load_trace_payload(trace_path)
            if trace_digest != record.get("trace_content_digest"):
                errors.append(f"outcome/trace semantic digest mismatch: {key}")
            if leaf_digest != record.get("final_parameter_leaf_digest"):
                errors.append(f"outcome/final-parameter digest mismatch: {key}")
            expected_metadata = _expected_trace_metadata(
                profile, variant, fold, tape.sha256, startup
            )
            if metadata != expected_metadata:
                errors.append(f"trace metadata differs from protocol: {key}")
            scientific = {name: arrays[name] for name in _TRACE_SCIENTIFIC_ARRAY_NAMES}
            errors.extend(
                f"{message}: {key}"
                for message in _trace_semantic_errors(scientific, variant, profile, tape)
            )
            expected_truth = true_parameters_by_fold.get(fold)
            if expected_truth is None or not np.array_equal(
                scientific["true_parameters"], expected_truth
            ):
                errors.append(f"trace true dynamics do not match the immutable tape: {key}")
            replay_barriers, replay_contact, replay_failure = experiment_core._barrier_trace(
                scientific["states"], tape, profile.trial
            )
            if not np.array_equal(scientific["barrier_margins"], replay_barriers):
                errors.append(f"trace barrier audit does not replay from states/tape: {key}")
            if not np.array_equal(scientific["contact"], replay_contact):
                errors.append(f"trace contact audit does not replay from states/tape: {key}")
            if not np.array_equal(scientific["failure"], replay_failure):
                errors.append(f"trace failure audit does not replay from states/tape: {key}")
            resources = resources_by_fold.get(fold)
            if resources is not None:
                errors.extend(
                    f"{message}: {key}"
                    for message in _final_parameter_leaf_errors(arrays, resources)
                )
            if plant_replay is not None:
                replay_next, replay_realized = plant_replay(
                    jnp.asarray(scientific["states"][:-1]),
                    jnp.asarray(scientific["commanded_motor_forces"][:-1]),
                    jnp.asarray(tape.mass_scale[: profile.trial.control_steps - 1]),
                    jnp.asarray(tape.drag_scale[: profile.trial.control_steps - 1]),
                    jnp.asarray(tape.wind_velocity[: profile.trial.control_steps - 1]),
                    jnp.asarray(tape.rotor_efficiency[: profile.trial.control_steps - 1]),
                )
                _block((replay_next, replay_realized))
                if not np.allclose(
                    scientific["states"][1:],
                    np.asarray(replay_next),
                    rtol=PLANT_REPLAY_RTOL,
                    atol=PLANT_REPLAY_ATOL,
                ):
                    errors.append(f"trace states do not replay through the true plant: {key}")
                if not np.array_equal(
                    scientific["realized_motor_forces"][:-1], np.asarray(replay_realized)
                ):
                    errors.append(
                        f"trace realized motor forces do not replay from commands/tape: {key}"
                    )
            adaptation_events = record.get("adaptation_events")
            recomputed_metrics = _knowledge_metrics(
                scientific,
                tape,
                profile,
                startup_adaptation_status=str(startup["record"]["status"]),
                adaptation_events=(
                    adaptation_events if isinstance(adaptation_events, list) else ()
                ),
            )
            if _canonical_json(record.get("metrics")) != _canonical_json(recomputed_metrics):
                errors.append(f"outcome metrics do not match trace: {key}")
            if record.get("operational_failure") is not recomputed_metrics["operational_failure"]:
                errors.append(f"outcome operational failure does not match evidence: {key}")
            truth_digest_by_fold[fold].add(
                _canonical_array_digest(
                    "dynamics-knowledge-true-parameters-v1",
                    {"true_parameters": scientific["true_parameters"]},
                )
            )
            if not all(
                np.all(np.isfinite(value))
                for name, value in arrays.items()
                if name.startswith("final_param_leaf_")
            ):
                errors.append(f"trace final parameters are nonfinite: {key}")
            errors.extend(
                f"{message}: {key}"
                for message in _adaptation_event_errors(record.get("adaptation_events"), profile)
            )
            execution_seconds = float(record.get("execution_seconds"))
            if not math.isfinite(execution_seconds) or execution_seconds < 0.0:
                errors.append(f"outcome execution timing is invalid: {key}")
        except (KeyError, OSError, TypeError, ValueError) as error:
            errors.append(f"trace verification failed for {key}: {error}")

    for fold in scheduled_folds:
        if len(truth_digest_by_fold[fold]) > 1:
            errors.append(f"paired variants used different true dynamics: fold {fold}")
        if len(startup_digest_by_fold[fold]) > 1:
            errors.append(f"paired variants used different startup adaptation: fold {fold}")
    if len(outcomes) != expected:
        errors.append(f"retained {len(outcomes)} outcomes but expected {expected}")
    execution_complete = len(outcomes) == expected
    startup_failures, periodic_adaptation_failures = _adaptation_failure_counts(outcomes)
    adaptation_execution_failures = startup_failures + periodic_adaptation_failures
    operational_failures = sum(bool(record.get("operational_failure")) for record in outcomes)
    if manifest.get("expected_outcomes") != expected:
        errors.append("manifest expected outcome count is wrong")
    if manifest.get("completed_outcomes") != completed:
        errors.append("manifest completed outcome count is wrong")
    if manifest.get("failed_outcomes") != failed:
        errors.append("manifest failed outcome count is wrong")
    if manifest.get("operational_failures") != operational_failures:
        errors.append("manifest operational-failure count is wrong")
    if manifest.get("adaptation_execution_failures") != adaptation_execution_failures:
        errors.append("manifest adaptation execution-failure count is wrong")
    if manifest.get("execution_complete") != execution_complete:
        errors.append("manifest completion status disagrees with retained outcomes")
    if marker is not None:
        if marker.get("execution_complete") != execution_complete:
            errors.append("completion marker disagrees with retained outcomes")
        if marker.get("retained_failures") != failed:
            errors.append("completion marker failure count is wrong")
        if marker.get("operational_failures") != operational_failures:
            errors.append("completion marker operational-failure count is wrong")
        if marker.get("adaptation_execution_failures") != adaptation_execution_failures:
            errors.append("completion marker adaptation failure count is wrong")
    try:
        recomputed = aggregate_dynamics_knowledge_outcomes(campaign_config, profile, outcomes)
        if _canonical_json(aggregates) != _canonical_json(recomputed):
            errors.append("aggregates do not match retained paired outcomes")
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"aggregate recomputation failed: {error}")
    return DynamicsKnowledgeVerification(
        not errors, tuple(errors), expected, len(outcomes), completed, failed, operational_failures
    )


def _expected_trace_metadata(
    profile: DynamicsKnowledgeProfile,
    variant: DynamicsKnowledgeVariant,
    fold: int,
    tape_digest: str,
    startup: Mapping[str, Any],
) -> dict[str, Any]:
    record = startup["record"]
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "scope": "closed_loop_dynamics_knowledge",
        "claim_boundary": CLAIM_BOUNDARY,
        "fold": fold,
        "variant": asdict(variant),
        "profile": profile.name,
        "trial_shape": {
            "control_steps": profile.trial.control_steps,
            "policy_count": profile.trial.policy_count,
            "certificate_horizon": profile.trial.certificate_horizon,
            "training_scenario_count": profile.trial.training_scenario_count,
        },
        "tape_digest": tape_digest,
        "startup_semantic_digest": startup["semantic_digest"],
        "startup_active_digest": record["active_digest"],
        "startup_active_params_digest": record["active_params_digest"],
        "filter_implementation": FILTER_IMPLEMENTATION_ID,
        "training_objective": TRAINING_OBJECTIVE_ID,
        "uncertainty_aware_bptt_training": False,
        "axis_aligned_sample_range_metric_definition": (
            AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION
        ),
    }


def _trace_semantic_errors(
    arrays: Mapping[str, np.ndarray],
    variant: DynamicsKnowledgeVariant,
    profile: DynamicsKnowledgeProfile,
    tape: ScenarioTape,
) -> tuple[str, ...]:
    errors: list[str] = []
    expected_names = _TRACE_SCIENTIFIC_ARRAY_NAMES
    if set(arrays) != expected_names:
        return ("trace has missing or unexpected scientific arrays",)
    steps = profile.trial.control_steps
    policy_count = profile.trial.policy_count
    expected_shapes = {
        "states": (steps, 13),
        "state_valid": (steps,),
        "command_valid": (steps,),
        "commanded_motor_forces": (steps, 4),
        "realized_motor_forces": (steps, 4),
        "nominal_motor_forces": (steps, 4),
        "policy_hard_values": (steps, policy_count),
        "selected_policy": (steps,),
        "selected_hard_value": (steps,),
        "degraded": (steps,),
        "proposal_accepted": (steps,),
        "fallback_accepted": (steps,),
        "used_fallback": (steps,),
        "applied_interval_margin": (steps,),
        "applied_next_value": (steps,),
        "applied_exact_residual": (steps,),
        "controller_seconds": (steps,),
        "plant_seconds": (steps,),
        "estimator_seconds": (steps,),
        "adaptation_seconds": (steps,),
        "translation_update_status": (steps,),
        "rotor_update_status": (steps,),
        "estimator_model_version": (steps,),
        "snapshot_version": (steps,),
        "model_last_observation_transition": (steps,),
        "estimator_history_count": (steps,),
        "true_parameters": (steps, DYNAMICS_PARAMETER_COUNT),
        "estimated_parameters": (steps, DYNAMICS_PARAMETER_COUNT),
        "runtime_sample_parameters": (steps, MAX_RUNTIME_SAMPLES, DYNAMICS_PARAMETER_COUNT),
        "runtime_sample_valid": (steps, MAX_RUNTIME_SAMPLES),
        "adaptation_publication_accepted": (steps,),
        "contact": (steps,),
        "failure": (steps,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            errors.append(f"trace array {name} has shape {arrays[name].shape}, expected {shape}")
    if errors:
        return tuple(errors)
    if arrays["barrier_margins"].ndim != 2 or arrays["barrier_margins"].shape[0] != steps:
        errors.append("barrier_margins must have shape (control_steps, barrier_components)")
    elif arrays["barrier_margins"].shape[1] < 1:
        errors.append("barrier_margins must contain at least one barrier component")
    boolean_names = (
        "state_valid",
        "command_valid",
        "degraded",
        "proposal_accepted",
        "fallback_accepted",
        "used_fallback",
        "runtime_sample_valid",
        "adaptation_publication_accepted",
        "contact",
        "failure",
    )
    for name in boolean_names:
        if arrays[name].dtype.kind != "b":
            errors.append(f"trace array {name} must have boolean dtype")
    expected_command = np.ones((steps,), dtype=bool)
    expected_command[-1] = False
    if not np.array_equal(arrays["state_valid"], np.ones((steps,), dtype=bool)):
        errors.append("state validity must include every control boundary")
    if not np.array_equal(arrays["command_valid"], expected_command):
        errors.append("command validity must exclude only the terminal boundary")
    if arrays["selected_policy"][-1] != -1:
        errors.append("terminal no-command row must use selected_policy=-1")
    selected = arrays["selected_policy"][:-1]
    if np.any((selected < 0) | (selected >= policy_count)):
        errors.append("executed selected_policy index is outside the library")
    finite_names = (
        "states",
        "commanded_motor_forces",
        "realized_motor_forces",
        "nominal_motor_forces",
        "selected_hard_value",
        "applied_interval_margin",
        "applied_next_value",
        "applied_exact_residual",
        "controller_seconds",
        "plant_seconds",
        "estimator_seconds",
        "adaptation_seconds",
        "true_parameters",
        "estimated_parameters",
        "runtime_sample_parameters",
        "barrier_margins",
    )
    for name in finite_names:
        if not np.all(np.isfinite(arrays[name])):
            errors.append(f"trace array {name} contains a nonfinite value")
    for name in ("controller_seconds", "plant_seconds", "estimator_seconds", "adaptation_seconds"):
        if np.any(arrays[name] < 0.0):
            errors.append(f"trace timing array {name} contains a negative value")
    if not np.all(arrays["controller_seconds"][:-1] > 0.0):
        errors.append("every executed control boundary must have positive controller timing")
    if not np.all(arrays["plant_seconds"][:-1] > 0.0):
        errors.append("every executed transition must have positive plant timing")
    if np.any(arrays["controller_seconds"][-1:] != 0.0) or np.any(
        arrays["plant_seconds"][-1:] != 0.0
    ):
        errors.append("terminal no-command timing must be exactly zero")

    last_observation = arrays["model_last_observation_transition"]
    if variant.privileged_oracle_upper_bound:
        if not np.array_equal(last_observation, np.arange(steps, dtype=last_observation.dtype)):
            errors.append("oracle trace must read only exact current-boundary truth")
        if not np.array_equal(arrays["estimated_parameters"], arrays["true_parameters"]):
            errors.append("oracle point model differs from true current-boundary parameters")
    else:
        latest_permitted = np.arange(steps, dtype=np.int64) - 1
        if np.any(last_observation.astype(np.int64) > latest_permitted):
            errors.append("estimated variant consumed future or same-transition observations")
    valid_samples = arrays["runtime_sample_valid"]
    count = variant.runtime_dynamics_samples
    if count == 0:
        if np.any(valid_samples):
            errors.append("point-model variant unexpectedly contains runtime particles")
    else:
        if not np.all(valid_samples[:, :count]):
            errors.append(f"configured R={count} runtime particle set contains invalid members")
        if np.any(valid_samples[:, count:]):
            errors.append(f"runtime sample mask exceeds configured R={count}")
    if not np.array_equal(arrays["true_parameters"][:, 7:], np.ones((steps, 4), dtype=np.float64)):
        errors.append("true rotor efficiency changed in the declared mass/drag/wind condition")
    if tape.sha256 == "":
        errors.append("scenario tape digest is empty")
    if np.any(arrays["estimator_history_count"] < 0):
        errors.append("estimator history count is negative")
    if np.any(np.diff(arrays["estimator_history_count"]) < 0):
        errors.append("estimator history count is not monotone")
    if arrays["estimator_history_count"][-1] != steps - 1:
        errors.append("terminal estimator history does not contain every executed transition")
    return tuple(errors)


def _adaptation_event_errors(value: Any, profile: DynamicsKnowledgeProfile) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ("adaptation events must be a list",)
    errors: list[str] = []
    expected_boundaries = set(
        range(
            profile.trial.adaptation_interval_steps,
            profile.trial.control_steps - 1,
            profile.trial.adaptation_interval_steps,
        )
    )
    seen: set[int] = set()
    for event in value:
        if not isinstance(event, dict):
            errors.append("adaptation event must be an object")
            continue
        try:
            boundary = int(event["boundary"])
            if boundary in seen:
                errors.append(f"duplicate adaptation boundary {boundary}")
            seen.add(boundary)
            if boundary not in expected_boundaries:
                errors.append(f"adaptation boundary {boundary} is outside the schedule")
            if event.get("training_objective") != TRAINING_OBJECTIVE_ID:
                errors.append(f"adaptation training objective changed at boundary {boundary}")
            if event.get("uncertainty_aware_bptt_training") is not False:
                errors.append(f"adaptation incorrectly claims robust BPTT at boundary {boundary}")
            if event.get("hard_admission_dynamics_samples") != ADAPTATION_VALIDATION_SAMPLES:
                errors.append(f"adaptation admission sample count changed at boundary {boundary}")
            seconds = float(event["execution_seconds"])
            if not math.isfinite(seconds) or seconds < 0.0:
                errors.append(f"adaptation timing is invalid at boundary {boundary}")
            if event.get("status") not in {"complete", "failed"}:
                errors.append(f"adaptation status is invalid at boundary {boundary}")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"malformed adaptation event: {error}")
    if seen != expected_boundaries:
        errors.append("adaptation event boundaries do not match the synchronous schedule")
    return tuple(errors)


def _final_parameter_leaf_errors(
    arrays: Mapping[str, np.ndarray], resources: ExperimentResources
) -> tuple[str, ...]:
    names = sorted(name for name in arrays if name.startswith("final_param_leaf_"))
    expected_leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(resources.initial_params)]
    expected_names = [f"final_param_leaf_{index:03d}" for index in range(len(expected_leaves))]
    if names != expected_names:
        return ("final parameter leaves do not match the shared-actor tree structure",)
    errors: list[str] = []
    for name, expected in zip(names, expected_leaves, strict=True):
        value = arrays[name]
        if value.shape != expected.shape:
            errors.append(f"{name} shape differs from the shared-actor parameter tree")
        if value.dtype != expected.dtype:
            errors.append(f"{name} dtype differs from the shared-actor parameter tree")
        if not np.all(np.isfinite(value)):
            errors.append(f"{name} contains a nonfinite parameter")
    return tuple(errors)


def _verify_manifest_files(root: Path, manifest: Mapping[str, Any], errors: list[str]) -> None:
    files = manifest.get("files")
    if not isinstance(files, list):
        errors.append("manifest files must be a list")
        return
    recorded: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            errors.append("malformed manifest file record")
            continue
        relative = str(item["path"])
        if relative in recorded:
            errors.append(f"duplicate manifest file record: {relative}")
            continue
        recorded.add(relative)
        path = root / relative
        if not path.is_file():
            errors.append(f"manifest file missing: {relative}")
        elif _file_sha256(path) != item["sha256"]:
            errors.append(f"manifest file hash mismatch: {relative}")
        elif path.stat().st_size != item["bytes"]:
            errors.append(f"manifest file size mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in {"manifest.json", "complete.marker"}
    }
    if recorded != actual:
        errors.append("manifest file inventory differs from the artifact tree")


def _manifest_mapping(
    root: Path,
    configuration: Mapping[str, Any],
    *,
    expected: int,
    completed: int,
    failed: int,
    operational_failures: int,
    adaptation_execution_failures: int,
    execution_complete: bool,
    profile: DynamicsKnowledgeProfile,
) -> dict[str, Any]:
    excluded = {"manifest.json", "complete.marker"}
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        files.append({"path": relative, "sha256": _file_sha256(path), "bytes": path.stat().st_size})
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": configuration["experiment_id"],
        "scope": "closed_loop_dynamics_knowledge",
        "claim_boundary": CLAIM_BOUNDARY,
        "profile": profile.name,
        "source_tree_sha256": configuration["source_tree_sha256"],
        "expected_outcomes": expected,
        "completed_outcomes": completed,
        "failed_outcomes": failed,
        "operational_failures": operational_failures,
        "adaptation_execution_failures": adaptation_execution_failures,
        "execution_complete": execution_complete,
        "confirmatory_metric_family_eligible": (
            profile.intended_for_confirmatory_differences
            and execution_complete
            and failed == 0
            and adaptation_execution_failures == 0
            and completed == expected
        ),
        "oracle_label": ORACLE_LABEL,
        "training_claim": TRAINING_CLAIM,
        "operational_failure_definition": OPERATIONAL_FAILURE_DEFINITION,
        "axis_aligned_sample_range_metric_definition": (
            AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION
        ),
        "blanket_safety_superiority_supported": False,
        "files": files,
    }


def source_tree_digest(repository: Path) -> str:
    """Hash implementation and runtime assets, excluding docs/tests/generated evidence."""
    digest = hashlib.sha256(b"crazyflow.da_plcbf.dynamics-knowledge-source-tree.v3\0")
    package = repository / "crazyflow"
    runtime_suffixes = frozenset({".py", ".toml", ".xml", ".stl"})
    paths = (
        [
            path
            for path in package.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() in runtime_suffixes
        ]
        if package.is_dir()
        else []
    )
    for root in (repository / "examples" / "da_plcbf", repository / "benchmark"):
        if root.is_dir():
            paths.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    paths.extend(
        path for name in ("pyproject.toml", "pixi.lock") if (path := repository / name).is_file()
    )
    for path in sorted(set(paths)):
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _repository_root(value: str | os.PathLike[str] | None) -> Path:
    if value is not None:
        root = Path(value).resolve()
    else:
        root = Path.cwd().resolve()
        while root.parent != root and not (root / "pyproject.toml").is_file():
            root = root.parent
    if not (root / "pyproject.toml").is_file():
        raise ValueError("repository root must contain pyproject.toml")
    return root


def _canonical_array_digest(domain: str, arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(domain.encode("utf-8") + b"\0")
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        if array.dtype.hasobject:
            raise TypeError(f"canonical array {name} must not have object dtype")
        contiguous = np.ascontiguousarray(array)
        encoded_name = name.encode("utf-8")
        encoded_dtype = contiguous.dtype.str.encode("ascii")
        payload = contiguous.tobytes(order="C")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(8, "little"))
        digest.update(encoded_dtype)
        digest.update(contiguous.ndim.to_bytes(8, "little"))
        for dimension in contiguous.shape:
            digest.update(int(dimension).to_bytes(8, "little"))
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _outcome_key(record: Mapping[str, Any]) -> tuple[int, str]:
    variant = record["variant"]
    if not isinstance(variant, dict):
        raise TypeError("outcome variant must be an object")
    return int(record["fold"]), str(variant["variant_id"])


def _read_outcomes(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        return ()
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank outcome line {line_number}")
        record = _json_loads_object(line)
        key = _outcome_key(record)
        if key in seen:
            raise ValueError(f"duplicate outcome key {key}")
        seen.add(key)
        records.append(record)
    return tuple(records)


def _ordered_outcomes(
    records: Mapping[tuple[int, str], dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return the unique retained ledger in deterministic fold/registry order."""
    variant_order = {variant.variant_id: index for index, variant in enumerate(VARIANTS)}
    for key, record in records.items():
        if key != _outcome_key(record):
            raise ValueError(f"outcome map key does not match its record: {key}")
        if key[1] not in variant_order:
            raise ValueError(f"outcome map contains an unknown variant: {key}")
    return tuple(
        records[key] for key in sorted(records, key=lambda item: (item[0], variant_order[item[1]]))
    )


def _flush_outcomes(path: Path, records: Mapping[tuple[int, str], dict[str, Any]]) -> None:
    """Atomically commit the complete ledger; a torn append can never block resume."""
    payload = b"".join(_canonical_json(record) for record in _ordered_outcomes(records))
    _atomic_bytes(path, payload)


def _verify_resumable_record(
    root: Path,
    record: Mapping[str, Any],
    *,
    profile: DynamicsKnowledgeProfile,
    tape: ScenarioTape,
    startup: Mapping[str, Any],
    variant: DynamicsKnowledgeVariant,
    resources: ExperimentResources,
    plant_replay: Any,
) -> None:
    key = (int(record.get("fold", -1)), variant.variant_id)
    startup_record = startup.get("record")
    if not isinstance(startup_record, dict):
        raise ValueError(f"existing common startup record is malformed: {key[0]}")
    if startup.get("record_digest") != _record_digest(
        "crazyflow.da_plcbf.dynamics-knowledge-startup-record.v1", startup_record
    ):
        raise ValueError(f"existing common startup record digest changed: {key[0]}")
    if startup.get("semantic_identity") != _startup_identity(startup_record):
        raise ValueError(f"existing common startup semantic identity changed: {key[0]}")
    if startup.get("semantic_digest") != _startup_semantic_digest(startup_record):
        raise ValueError(f"existing common startup semantic digest changed: {key[0]}")
    expected_startup_digest = startup.get("semantic_digest")
    expected = {
        "scope": "closed_loop_dynamics_knowledge",
        "claim_boundary": CLAIM_BOUNDARY,
        "variant": asdict(variant),
        "tape_digest": tape.sha256,
        "startup_semantic_digest": expected_startup_digest,
        "startup_active_digest": startup_record.get("active_digest"),
        "startup_active_params_digest": startup_record.get("active_params_digest"),
        "filter_implementation": FILTER_IMPLEMENTATION_ID,
        "training_objective": TRAINING_OBJECTIVE_ID,
        "training_claim": TRAINING_CLAIM,
        "uncertainty_aware_bptt_training": False,
        "axis_aligned_sample_range_metric_definition": (
            AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION
        ),
        "blanket_safety_superiority_supported": False,
    }
    if record.get("schema_version") != OUTCOME_SCHEMA_VERSION:
        raise ValueError(f"existing outcome schema changed: {key}")
    for name, value in expected.items():
        if record.get(name) != value:
            raise ValueError(f"existing outcome field {name} changed: {key}")
    expected_oracle_label = ORACLE_LABEL if variant.variant_id == "oracle_point" else None
    if record.get("oracle_label") != expected_oracle_label:
        raise ValueError(f"existing outcome oracle label changed: {key}")
    if record.get("status") == "failed":
        if record.get("operational_failure") is not True:
            raise ValueError(f"existing failed outcome is not an operational failure: {key}")
        if not isinstance(record.get("failure_type"), str) or not isinstance(
            record.get("failure_message"), str
        ):
            raise ValueError(f"existing failed outcome lacks a typed failure: {key}")
        if record.get("trace_artifact") is not None or record.get("metrics") is not None:
            raise ValueError(f"existing failed outcome contains success evidence: {key}")
        return
    if record.get("status") != "complete":
        raise ValueError("existing dynamics-knowledge outcome has an unknown status")
    if not isinstance(record.get("operational_failure"), bool):
        raise ValueError(f"existing complete outcome lacks operational failure: {key}")
    path = (root / str(record["trace_artifact"])).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"existing trace path escapes the campaign root: {key}")
    if _file_sha256(path) != record.get("trace_artifact_sha256"):
        raise ValueError("existing dynamics-knowledge trace file hash mismatch")
    arrays, metadata, content_digest, leaf_digest = _load_trace_payload(path)
    if content_digest != record.get("trace_content_digest"):
        raise ValueError("existing dynamics-knowledge trace semantic digest mismatch")
    if leaf_digest != record.get("final_parameter_leaf_digest"):
        raise ValueError("existing dynamics-knowledge final parameter digest mismatch")
    expected_metadata = _expected_trace_metadata(
        profile, variant, int(record["fold"]), tape.sha256, startup
    )
    if metadata != expected_metadata:
        raise ValueError(f"existing dynamics-knowledge trace metadata changed: {key}")
    scientific = {name: arrays[name] for name in _TRACE_SCIENTIFIC_ARRAY_NAMES}
    semantic_errors = _trace_semantic_errors(scientific, variant, profile, tape)
    if semantic_errors:
        raise ValueError(f"existing trace semantics changed for {key}: {semantic_errors[0]}")
    expected_truth = np.stack(
        [
            _parameter_vector(
                *experiment_core._true_model(
                    resources.model, tape, ConditionID.DYNAMICS_CHANGE, boundary
                )
            )
            for boundary in range(profile.trial.control_steps)
        ]
    )
    if not np.array_equal(scientific["true_parameters"], expected_truth):
        raise ValueError(f"existing trace true dynamics do not match tape: {key}")
    barriers, contact, failure = experiment_core._barrier_trace(
        scientific["states"], tape, profile.trial
    )
    if not np.array_equal(scientific["barrier_margins"], barriers):
        raise ValueError(f"existing trace barrier audit does not replay: {key}")
    if not np.array_equal(scientific["contact"], contact):
        raise ValueError(f"existing trace contact audit does not replay: {key}")
    if not np.array_equal(scientific["failure"], failure):
        raise ValueError(f"existing trace failure audit does not replay: {key}")
    replay_next, replay_realized = plant_replay(
        jnp.asarray(scientific["states"][:-1]),
        jnp.asarray(scientific["commanded_motor_forces"][:-1]),
        jnp.asarray(tape.mass_scale[: profile.trial.control_steps - 1]),
        jnp.asarray(tape.drag_scale[: profile.trial.control_steps - 1]),
        jnp.asarray(tape.wind_velocity[: profile.trial.control_steps - 1]),
        jnp.asarray(tape.rotor_efficiency[: profile.trial.control_steps - 1]),
    )
    _block((replay_next, replay_realized))
    if not np.allclose(
        scientific["states"][1:],
        np.asarray(replay_next),
        rtol=PLANT_REPLAY_RTOL,
        atol=PLANT_REPLAY_ATOL,
    ):
        raise ValueError(f"existing trace states do not replay through the true plant: {key}")
    if not np.array_equal(scientific["realized_motor_forces"][:-1], np.asarray(replay_realized)):
        raise ValueError(f"existing trace realized motor forces do not replay: {key}")
    leaf_errors = _final_parameter_leaf_errors(arrays, resources)
    if leaf_errors:
        raise ValueError(f"existing final parameter tree changed for {key}: {leaf_errors[0]}")
    adaptation_events = record.get("adaptation_events")
    metrics = _knowledge_metrics(
        scientific,
        tape,
        profile,
        startup_adaptation_status=str(startup_record["status"]),
        adaptation_events=(adaptation_events if isinstance(adaptation_events, list) else ()),
    )
    if _canonical_json(record.get("metrics")) != _canonical_json(metrics):
        raise ValueError(f"existing outcome metrics do not match trace: {key}")
    if record.get("operational_failure") is not metrics["operational_failure"]:
        raise ValueError(f"existing outcome operational failure does not match evidence: {key}")
    execution_seconds = float(record.get("execution_seconds"))
    if not math.isfinite(execution_seconds) or execution_seconds < 0.0:
        raise ValueError(f"existing outcome execution timing is invalid: {key}")
    adaptation_errors = _adaptation_event_errors(record.get("adaptation_events"), profile)
    if adaptation_errors:
        raise ValueError(f"existing adaptation ledger changed for {key}: {adaptation_errors[0]}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Array):
        return _json_safe(np.asarray(value))
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_safe(value.item())
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value == math.inf:
            return "Infinity"
        if value == -math.inf:
            return "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _record_digest(domain: str, value: Any) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + _canonical_json(value)).hexdigest()


def _json_loads_object(payload: str | bytes) -> dict[str, Any]:
    def pairs(pairs_value: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs_value]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate JSON key")
        return dict(pairs_value)

    value = json.loads(payload, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise TypeError("JSON root must be an object")
    return value


def _read_object(path: Path) -> dict[str, Any]:
    return _json_loads_object(path.read_bytes())


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _canonical_json(value))


def _write_once_json(path: Path, value: Any) -> None:
    _write_once_bytes(path, _canonical_json(value))


def _write_once_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Atomically publish one immutable NPZ without ever replacing an existing inode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_once_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish immutable bytes, failing if the destination already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ADAPTATION_VALIDATION_SAMPLES",
    "AGGREGATE_SCHEMA_VERSION",
    "AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION",
    "CAMPAIGN_SCHEMA_VERSION",
    "CLAIM_BOUNDARY",
    "CONDITION_ID",
    "DynamicsKnowledgeCampaignConfig",
    "DynamicsKnowledgeCampaignRun",
    "DynamicsKnowledgeProfile",
    "DynamicsKnowledgeVariant",
    "DynamicsKnowledgeVerification",
    "FILTER_IMPLEMENTATION_ID",
    "MANIFEST_SCHEMA_VERSION",
    "ORACLE_LABEL",
    "OPERATIONAL_FAILURE_DEFINITION",
    "OUTCOME_SCHEMA_VERSION",
    "RUNTIME_PROVENANCE_SCHEMA_VERSION",
    "STARTUP_BUNDLE_SCHEMA_VERSION",
    "STARTUP_SCHEMA_VERSION",
    "TRACE_SCHEMA_VERSION",
    "TRAINING_CLAIM",
    "TRAINING_OBJECTIVE_ID",
    "VARIANTS",
    "aggregate_dynamics_knowledge_outcomes",
    "build_matched_resources",
    "compile_knowledge_executables",
    "dynamics_knowledge_profile",
    "execute_dynamics_knowledge_trial",
    "generate_matched_dynamics_tape",
    "prepare_common_startup_adaptation",
    "run_dynamics_knowledge_campaign",
    "source_tree_digest",
    "verify_dynamics_knowledge_campaign",
]
