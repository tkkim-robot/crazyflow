"""Matched candidate-proposal protocols for the DA-PLCBF ablation study.

This module is deliberately separate from :mod:`experiments`.  It compares *candidate proposal
quality* under fixed objective-evaluation budgets and a common hard finite-scenario scorer.  It
does not execute a closed-loop controller, admit a snapshot into the runtime store, or establish
that one controller is safer than another.

The proposal comparison uses the safety-agnostic generic-diversity objective so sampling-only,
BPTT-only, and hybrid sampling+BPTT differ only in how they spend a charged budget.  The smaller
architecture comparison uses the same PL-CBF-aligned objective for the shared and genuinely
independent actors.  Every returned candidate is then scored by exact hard Version-A values on a
held-out finite dynamics set.  The hard scorer is outside the charged proposal objective and is
identical within each paired comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.actor import (
    SharedActorConfig,
    SharedActorParams,
    SharedActorSpec,
    initialize_shared_actor,
)
from crazyflow.safety.da_plcbf.bptt import (
    BPTTFunctions,
    BPTTState,
    BPTTStepMetrics,
    tree_all_finite,
)
from crazyflow.safety.da_plcbf.config import LibraryLossConfig
from crazyflow.safety.da_plcbf.descriptors import trajectory_descriptors
from crazyflow.safety.da_plcbf.direct_wrench import motor_forces_to_wrench
from crazyflow.safety.da_plcbf.independent_actor import (
    IndependentActorParams,
    independent_quad_actor_library_loss,
    independent_quad_fallback_wrenches,
    initialize_independent_actor,
)
from crazyflow.safety.da_plcbf.library import (
    build_shared_quad_library_spec,
    descriptor_targets_from_spec,
)
from crazyflow.safety.da_plcbf.proposal_ablations import (
    HybridProposalConfig,
    ProposalBudget,
    ProposalResult,
    SamplingProposalConfig,
    require_matched_objective_budget,
    run_bptt_only_proposal,
    run_hybrid_proposal_bptt,
    run_sampling_only_proposal,
)
from crazyflow.safety.da_plcbf.quad_actor_losses import (
    QuadLearningConfig,
    quad_actor_library_loss,
    rigid_body_safety_batch_from_circles,
)
from crazyflow.safety.da_plcbf.quad_generic_diversity_bptt import (
    GenericDiversityConfig,
    generic_diversity_loss,
)
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.quad_uncertainty import (
    UncertainQuadRolloutBatch,
    VersionAModelSamples,
    duplicate_circle_scenarios_for_samples,
    rollout_shared_quad_library_under_uncertainty,
    uncertain_quad_safety_values,
)
from crazyflow.safety.da_plcbf.snapshots import (
    create_active_snapshot,
    create_candidate_snapshot,
    tree_content_digest,
)
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.validation import (
    GateResult,
    HardValidationEvidence,
    HardValidationThresholds,
    ValidationReport,
    hard_validate_candidate,
)
from crazyflow.safety.da_plcbf.version_a_barriers import (
    VersionABarrierConfig,
    VersionAModel,
    safety_constraint_names,
)
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator

CANDIDATE_ONLY_CLAIM_BOUNDARY = (
    "Candidate-quality ablation only: these open-loop proposal and held-out finite-scenario "
    "scores are not closed-loop safety outcomes and cannot support a safer-than-baseline claim."
)
STRUCTURAL_POLICY_COUNT = 8
DESCRIPTOR_SCALES = (2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0)

ProposalMethod = Literal["bptt", "sampling", "hybrid"]
ActorArchitecture = Literal["shared", "independent"]
StudyFamily = Literal["proposal", "architecture", "component", "scale"]
LossAblation = Literal["full", "no_redundancy", "no_diversity", "no_trust"]


@dataclass(frozen=True, slots=True)
class ProposalPoint:
    """One fixed-shape proposal comparison point."""

    policy_count: int
    horizon: int
    batch_size: int
    uncertainty_samples: int
    objective_evaluations: int
    hybrid_gradient_updates: int


@dataclass(frozen=True, slots=True)
class ArchitecturePoint:
    """One fixed-shape shared-versus-independent architecture point."""

    policy_count: int
    horizon: int
    batch_size: int
    uncertainty_samples: int
    gradient_updates: int


@dataclass(frozen=True, slots=True)
class CandidateStudyProfile:
    """Predeclared fold count, shapes, and budgets for a candidate-only study."""

    name: str
    folds: int
    proposal_points: tuple[ProposalPoint, ...]
    architecture_point: ArchitecturePoint
    predeclared_confirmatory_schedule: bool

    def validate(self) -> None:
        if self.name not in {"smoke", "development", "confirmatory"}:
            raise ValueError("unknown candidate-study profile")
        if self.folds <= 0:
            raise ValueError("folds must be positive")
        for point in self.proposal_points:
            _validate_shape(point.policy_count, point.horizon, point.batch_size)
            if point.uncertainty_samples not in (4, 8):
                raise ValueError("uncertainty_samples must be 4 or 8")
            HybridProposalConfig(
                ProposalBudget(
                    point.objective_evaluations, gradient_updates=point.hybrid_gradient_updates
                )
            ).validate()
        architecture = self.architecture_point
        _validate_shape(architecture.policy_count, architecture.horizon, architecture.batch_size)
        if architecture.uncertainty_samples not in (4, 8):
            raise ValueError("uncertainty_samples must be 4 or 8")
        if architecture.gradient_updates <= 0:
            raise ValueError("architecture gradient_updates must be positive")
        if self.predeclared_confirmatory_schedule and (
            self.name != "confirmatory" or self.folds != 100
        ):
            raise ValueError(
                "the predeclared confirmatory schedule requires the full 100-fold profile"
            )


def candidate_study_profile(name: str) -> CandidateStudyProfile:
    """Return an exact smoke, development, or confirmatory candidate-study contract."""
    if name == "smoke":
        profile = CandidateStudyProfile(
            name="smoke",
            folds=1,
            proposal_points=(ProposalPoint(16, 2, 2, 4, 4, 2),),
            architecture_point=ArchitecturePoint(16, 2, 2, 4, 4),
            predeclared_confirmatory_schedule=False,
        )
    elif name in {"development", "confirmatory"}:
        profile = CandidateStudyProfile(
            name=name,
            folds=20 if name == "development" else 100,
            proposal_points=(
                ProposalPoint(16, 25, 16, 8, 10, 5),
                ProposalPoint(32, 25, 16, 8, 10, 5),
            ),
            architecture_point=ArchitecturePoint(16, 25, 16, 8, 4),
            predeclared_confirmatory_schedule=name == "confirmatory",
        )
    else:
        raise ValueError(f"unknown candidate-study profile {name!r}")
    profile.validate()
    return profile


@dataclass(frozen=True, slots=True)
class CandidateVariant:
    """Fully resolved variant identity; no field is inferred during execution."""

    variant_id: str
    family: StudyFamily
    proposal_method: ProposalMethod
    architecture: ActorArchitecture
    objective_id: str
    policy_count: int
    horizon: int
    batch_size: int
    score_horizon: int
    score_batch_size: int
    uncertainty_samples: int
    objective_evaluations: int
    gradient_updates: int
    loss_ablation: LossAblation = "full"
    validation_gate_enabled: bool = True
    train_skill_codes: bool = False
    train_durations: bool = False


def variants_for_profile(profile: CandidateStudyProfile) -> tuple[CandidateVariant, ...]:
    """Expand a profile into deterministic proposal and architecture variants."""
    profile.validate()
    variants: list[CandidateVariant] = []
    for point in profile.proposal_points:
        suffix = (
            f"k{point.policy_count}-h{point.horizon}-b{point.batch_size}"
            f"-r{point.uncertainty_samples}"
        )
        variants.extend(
            (
                CandidateVariant(
                    f"proposal-bptt-{suffix}",
                    "proposal",
                    "bptt",
                    "shared",
                    "objective_generic_diversity",
                    point.policy_count,
                    point.horizon,
                    point.batch_size,
                    point.horizon,
                    point.batch_size,
                    point.uncertainty_samples,
                    point.objective_evaluations,
                    point.objective_evaluations,
                ),
                CandidateVariant(
                    f"proposal-sampling-{suffix}",
                    "proposal",
                    "sampling",
                    "shared",
                    "objective_generic_diversity",
                    point.policy_count,
                    point.horizon,
                    point.batch_size,
                    point.horizon,
                    point.batch_size,
                    point.uncertainty_samples,
                    point.objective_evaluations,
                    0,
                ),
                CandidateVariant(
                    f"proposal-hybrid-{suffix}",
                    "proposal",
                    "hybrid",
                    "shared",
                    "objective_generic_diversity",
                    point.policy_count,
                    point.horizon,
                    point.batch_size,
                    point.horizon,
                    point.batch_size,
                    point.uncertainty_samples,
                    point.objective_evaluations,
                    point.hybrid_gradient_updates,
                ),
            )
        )
    point = profile.architecture_point
    suffix = (
        f"k{point.policy_count}-h{point.horizon}-b{point.batch_size}"
        f"-r{point.uncertainty_samples}-a{point.gradient_updates}"
    )
    for architecture in ("shared", "independent"):
        variants.append(
            CandidateVariant(
                f"architecture-{architecture}-small-{suffix}",
                "architecture",
                "bptt",
                architecture,
                "objective_plcbf_aligned",
                point.policy_count,
                point.horizon,
                point.batch_size,
                point.horizon,
                point.batch_size,
                point.uncertainty_samples,
                point.gradient_updates,
                point.gradient_updates,
            )
        )
    if profile.name != "smoke":
        common = {
            "architecture": "shared",
            "policy_count": 16,
            "horizon": 25,
            "batch_size": 16,
            "score_horizon": 25,
            "score_batch_size": 16,
            "uncertainty_samples": 8,
            "objective_evaluations": 4,
            "gradient_updates": 4,
        }
        variants.extend(
            (
                CandidateVariant(
                    "component-reference-plcbf-full-fixed-gated",
                    "component",
                    "bptt",
                    objective_id="objective_plcbf_aligned",
                    **common,
                ),
                CandidateVariant(
                    "component-objective-generic",
                    "component",
                    "bptt",
                    objective_id="objective_generic_diversity",
                    **common,
                ),
                CandidateVariant(
                    "component-loss-no-redundancy",
                    "component",
                    "bptt",
                    objective_id="objective_plcbf_aligned",
                    loss_ablation="no_redundancy",
                    **common,
                ),
                CandidateVariant(
                    "component-loss-no-diversity",
                    "component",
                    "bptt",
                    objective_id="objective_plcbf_aligned",
                    loss_ablation="no_diversity",
                    **common,
                ),
                CandidateVariant(
                    "component-loss-no-trust",
                    "component",
                    "bptt",
                    objective_id="objective_plcbf_aligned",
                    loss_ablation="no_trust",
                    **common,
                ),
                CandidateVariant(
                    "component-validation-gate-off",
                    "component",
                    "bptt",
                    objective_id="objective_plcbf_aligned",
                    validation_gate_enabled=False,
                    **common,
                ),
                CandidateVariant(
                    "component-train-skill-codes",
                    "component",
                    "bptt",
                    objective_id="objective_plcbf_aligned",
                    train_skill_codes=True,
                    **common,
                ),
                CandidateVariant(
                    "component-train-durations",
                    "component",
                    "bptt",
                    objective_id="objective_plcbf_aligned",
                    train_durations=True,
                    **common,
                ),
            )
        )
        scale_common = {
            "architecture": "shared",
            "proposal_method": "bptt",
            "objective_id": "objective_plcbf_aligned",
            "score_horizon": 50,
            "score_batch_size": 16,
            "uncertainty_samples": 8,
        }
        variants.extend(
            (
                CandidateVariant(
                    "scale-reference-k16-h25-b16-a4",
                    "scale",
                    policy_count=16,
                    horizon=25,
                    batch_size=16,
                    objective_evaluations=4,
                    gradient_updates=4,
                    **scale_common,
                ),
                CandidateVariant(
                    "scale-policy-count-k32",
                    "scale",
                    policy_count=32,
                    horizon=25,
                    batch_size=16,
                    objective_evaluations=4,
                    gradient_updates=4,
                    **scale_common,
                ),
                CandidateVariant(
                    "scale-horizon-h50",
                    "scale",
                    policy_count=16,
                    horizon=50,
                    batch_size=16,
                    objective_evaluations=4,
                    gradient_updates=4,
                    **scale_common,
                ),
                CandidateVariant(
                    "scale-scenario-batch-b64",
                    "scale",
                    policy_count=16,
                    horizon=25,
                    batch_size=64,
                    objective_evaluations=4,
                    gradient_updates=4,
                    **scale_common,
                ),
                CandidateVariant(
                    "scale-adaptation-budget-a10",
                    "scale",
                    policy_count=16,
                    horizon=25,
                    batch_size=16,
                    objective_evaluations=10,
                    gradient_updates=10,
                    **scale_common,
                ),
            )
        )
    resolved = tuple(variants)
    for variant in resolved:
        _validate_variant(variant)
    if len({variant.variant_id for variant in resolved}) != len(resolved):
        raise ValueError("candidate variant identifiers must be unique")
    return resolved


def _validate_shape(policy_count: int, horizon: int, batch_size: int) -> None:
    if policy_count < 16:
        raise ValueError("candidate ablations require K >= 16")
    if min(horizon, batch_size) <= 0:
        raise ValueError("horizon and batch_size must be positive")


def _validate_variant(variant: CandidateVariant) -> None:
    _validate_shape(variant.policy_count, variant.horizon, variant.batch_size)
    if min(variant.score_horizon, variant.score_batch_size) <= 0:
        raise ValueError("score_horizon and score_batch_size must be positive")
    if variant.uncertainty_samples not in (4, 8):
        raise ValueError("held-out uncertainty_samples must be 4 or 8")
    ProposalBudget(variant.objective_evaluations, variant.gradient_updates).validate()
    if variant.proposal_method == "bptt" and (
        variant.objective_evaluations != variant.gradient_updates
    ):
        raise ValueError("BPTT variants require one objective evaluation per gradient")
    if variant.objective_id == "objective_generic_diversity" and (variant.loss_ablation != "full"):
        raise ValueError("PL-CBF loss-term ablations do not apply to the generic objective")
    if variant.architecture == "independent" and (
        variant.train_skill_codes or variant.train_durations
    ):
        raise ValueError("independent skill training is not part of the matched architecture axis")


@dataclass(frozen=True, slots=True)
class CandidateProtocolResources:
    """Physical and numerical resources shared by all folds and variants."""

    model: VersionAModel
    actuator: VersionAActuator
    actor_config: SharedActorConfig
    quad_config: QuadPolicyConfig
    barrier_config: VersionABarrierConfig
    generic_config: GenericDiversityConfig
    plcbf_loss_config: LibraryLossConfig


def build_candidate_protocol_resources() -> CandidateProtocolResources:
    """Build the documented Crazyflie Version-A resources once per campaign."""
    raw: dict[str, Any] = load_params("cf21B_500")
    model = VersionAModel(
        mass=jnp.asarray(raw["mass"], dtype=jnp.float32),
        gravity_vec=jnp.asarray(raw["gravity_vec"], dtype=jnp.float32),
        inertia=jnp.asarray(raw["J"], dtype=jnp.float32),
        inertia_inv=jnp.linalg.inv(jnp.asarray(raw["J"], dtype=jnp.float32)),
        drag_matrix=jnp.asarray(raw["drag_matrix"], dtype=jnp.float32),
        wind_velocity=jnp.zeros(3, dtype=jnp.float32),
        external_force=jnp.zeros(3, dtype=jnp.float32),
        external_torque=jnp.zeros(3, dtype=jnp.float32),
    )
    actuator = VersionAActuator(
        arm_length=jnp.asarray(raw["L"], dtype=jnp.float32),
        thrust_to_torque=jnp.asarray(raw["thrust2torque"], dtype=jnp.float32),
        mixing_matrix=jnp.asarray(raw["mixing_matrix"], dtype=jnp.float32),
        thrust_min=jnp.asarray(raw["thrust_min"], dtype=jnp.float32),
        thrust_max=jnp.asarray(raw["thrust_max"], dtype=jnp.float32),
    )
    return CandidateProtocolResources(
        model=model,
        actuator=actuator,
        actor_config=SharedActorConfig(hidden_width=16),
        quad_config=QuadPolicyConfig(),
        barrier_config=VersionABarrierConfig(obstacle_clearance=0.12, arena_clearance=0.08),
        generic_config=GenericDiversityConfig(),
        plcbf_loss_config=LibraryLossConfig(),
    )


@dataclass(frozen=True, slots=True)
class CommonFoldInputs:
    """Maximal fold inputs from which every K/B comparison takes a nested prefix."""

    fold: int
    root_seed: int
    spec: SharedActorSpec
    shared_params: SharedActorParams
    independent_params: IndependentActorParams
    initial_states: Array
    scenarios: CircleScenarioBatch
    validation_initial_states: Array
    validation_scenarios: CircleScenarioBatch
    model_samples: VersionAModelSamples
    content_digest: str


def generate_common_fold_inputs(
    profile: CandidateStudyProfile,
    *,
    fold: int,
    root_seed: int,
    resources: CandidateProtocolResources,
) -> CommonFoldInputs:
    """Generate maximal deterministic arrays once; variants consume only nested prefixes."""
    profile.validate()
    if fold < 0 or root_seed < 0:
        raise ValueError("fold and root_seed must be nonnegative")
    variants = variants_for_profile(profile)
    # Generate the predeclared maximum tensors once.  Every reported K/B point is a literal prefix
    # of these K=64/B=64 values, even when a profile only dispatches smaller study points.
    max_k = max(64, *(item.policy_count for item in variants))
    max_b = max(64, *(item.batch_size for item in variants))
    spec = build_shared_quad_library_spec(
        policy_count=max_k, structural_policy_count=STRUCTURAL_POLICY_COUNT
    )
    seed_sequence = np.random.SeedSequence([root_seed, fold, 0xDA, 0xCBF])
    state_seed, shared_seed, independent_seed = seed_sequence.generate_state(3, dtype=np.uint32)
    rng = np.random.default_rng(int(state_seed))

    def sample_batch() -> tuple[Array, CircleScenarioBatch]:
        position = np.column_stack(
            (
                rng.uniform(-0.25, 0.25, max_b),
                rng.uniform(-0.25, 0.25, max_b),
                rng.uniform(0.85, 1.15, max_b),
            )
        )
        velocity = rng.uniform(-0.12, 0.12, (max_b, 3))
        quaternion = np.broadcast_to(np.asarray([0.0, 0.0, 0.0, 1.0]), (max_b, 4))
        angular = rng.uniform(-0.03, 0.03, (max_b, 3))
        states = jnp.asarray(
            np.concatenate((position, quaternion, velocity, angular), axis=1), dtype=jnp.float32
        )

        centers = np.empty((max_b, 2, 3), dtype=np.float32)
        centers[:, 0] = np.column_stack(
            (
                rng.uniform(0.65, 1.05, max_b),
                rng.uniform(-0.55, 0.55, max_b),
                rng.uniform(0.75, 1.25, max_b),
            )
        )
        centers[:, 1] = np.column_stack(
            (
                rng.uniform(-1.05, -0.65, max_b),
                rng.uniform(-0.55, 0.55, max_b),
                rng.uniform(0.75, 1.25, max_b),
            )
        )
        batch = CircleScenarioBatch(
            obstacle_centers=jnp.asarray(centers),
            obstacle_radii=jnp.asarray(rng.uniform(0.14, 0.22, (max_b, 2)), dtype=jnp.float32),
            obstacle_mask=jnp.ones((max_b, 2), dtype=bool),
            arena_lower=jnp.broadcast_to(jnp.asarray([-2.0, -2.0, 0.2]), (max_b, 3)),
            arena_upper=jnp.broadcast_to(jnp.asarray([2.0, 2.0, 2.0]), (max_b, 3)),
            speed_limit=jnp.full((max_b,), 3.0, dtype=jnp.float32),
        )
        return states, batch

    initial_states, scenarios = sample_batch()
    validation_initial_states, validation_scenarios = sample_batch()
    shared = initialize_shared_actor(
        jax.random.key(int(shared_seed)),
        spec,
        dimension=3,
        n_obstacles=2,
        config=resources.actor_config,
    )
    independent = initialize_independent_actor(
        jax.random.key(int(independent_seed)),
        spec,
        dimension=3,
        n_obstacles=2,
        config=resources.actor_config,
    )
    samples = deterministic_model_samples(resources.model, count=8)
    digest = numeric_digest(
        "candidate-common-fold-v1",
        np.asarray(spec.base_codes),
        np.asarray(spec.base_desired_velocities),
        np.asarray(spec.base_durations),
        np.asarray(spec.adaptive_mask),
        *[np.asarray(leaf) for leaf in jax.tree.leaves(shared)],
        *[np.asarray(leaf) for leaf in jax.tree.leaves(independent)],
        np.asarray(initial_states),
        np.asarray(scenarios.obstacle_centers),
        np.asarray(scenarios.obstacle_radii),
        np.asarray(scenarios.obstacle_mask),
        np.asarray(scenarios.arena_lower),
        np.asarray(scenarios.arena_upper),
        np.asarray(scenarios.speed_limit),
        np.asarray(validation_initial_states),
        np.asarray(validation_scenarios.obstacle_centers),
        np.asarray(validation_scenarios.obstacle_radii),
        np.asarray(validation_scenarios.obstacle_mask),
        np.asarray(validation_scenarios.arena_lower),
        np.asarray(validation_scenarios.arena_upper),
        np.asarray(validation_scenarios.speed_limit),
        *[np.asarray(leaf) for leaf in jax.tree.leaves(samples)],
    )
    return CommonFoldInputs(
        fold,
        root_seed,
        spec,
        shared,
        independent,
        initial_states,
        scenarios,
        validation_initial_states,
        validation_scenarios,
        samples,
        digest,
    )


def deterministic_model_samples(model: VersionAModel, *, count: int) -> VersionAModelSamples:
    """Return a fixed, bounded R=4/R=8 dynamics set with a common controller model."""
    if count not in (4, 8):
        raise ValueError("model sample count must be 4 or 8")
    signs = jnp.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )[:count]
    mass = jnp.asarray(model.mass) * (1.0 + 0.08 * signs[:, 0])
    drag_scale = 1.0 + 0.12 * signs[:, 1]
    drag = jnp.asarray(model.drag_matrix)[None, :, :] * drag_scale[:, None, None]
    wind = 0.12 * jnp.stack((signs[:, 1], signs[:, 2], 0.5 * signs[:, 0]), axis=1)
    efficiency_pattern = jnp.asarray(
        [
            [1.0, 0.92, 1.0, 0.96],
            [0.94, 1.0, 0.96, 1.0],
            [1.0, 0.96, 0.92, 1.0],
            [0.96, 1.0, 1.0, 0.92],
            [0.93, 0.97, 1.0, 1.0],
            [1.0, 0.93, 0.97, 1.0],
            [1.0, 1.0, 0.93, 0.97],
            [0.97, 1.0, 1.0, 0.93],
        ],
        dtype=jnp.float32,
    )[:count]

    def repeat(value: Array) -> Array:
        array = jnp.asarray(value, dtype=jnp.float32)
        return jnp.broadcast_to(array, (count, *array.shape))

    return VersionAModelSamples(
        models=VersionAModel(
            mass=mass,
            gravity_vec=repeat(model.gravity_vec),
            inertia=repeat(model.inertia),
            inertia_inv=repeat(model.inertia_inv),
            drag_matrix=drag,
            wind_velocity=wind,
            external_force=repeat(model.external_force),
            external_torque=repeat(model.external_torque),
        ),
        rotor_efficiency=efficiency_pattern,
        weights=jnp.full((count,), 1.0 / count, dtype=jnp.float32),
        sample_valid=jnp.ones((count,), dtype=bool),
        retained_variance_fraction=jnp.asarray(1.0, dtype=jnp.float32),
        model_version=jnp.asarray(0, dtype=jnp.int32),
    )


def _slice_spec(spec: SharedActorSpec, count: int) -> SharedActorSpec:
    return spec.replace(
        base_codes=spec.base_codes[:count],
        base_desired_velocities=spec.base_desired_velocities[:count],
        base_durations=spec.base_durations[:count],
        adaptive_mask=spec.adaptive_mask[:count],
    )


def _slice_shared(params: SharedActorParams, count: int) -> SharedActorParams:
    return params.replace(
        code_offsets=params.code_offsets[:count],
        velocity_offsets=params.velocity_offsets[:count],
        duration_offsets=params.duration_offsets[:count],
    )


def _slice_independent(params: IndependentActorParams, count: int) -> IndependentActorParams:
    return jax.tree.map(lambda value: value[:count], params)


def _slice_scenarios(scenarios: CircleScenarioBatch, count: int) -> CircleScenarioBatch:
    return jax.tree.map(lambda value: value[:count], scenarios)


def _slice_model_samples(samples: VersionAModelSamples, count: int) -> VersionAModelSamples:
    if count not in (4, 8):
        raise ValueError("model sample count must be 4 or 8")
    return VersionAModelSamples(
        models=jax.tree.map(lambda value: value[:count], samples.models),
        rotor_efficiency=samples.rotor_efficiency[:count],
        weights=jnp.full((count,), 1.0 / count, dtype=samples.weights.dtype),
        sample_valid=samples.sample_valid[:count],
        retained_variance_fraction=samples.retained_variance_fraction,
        model_version=samples.model_version,
    )


@dataclass(frozen=True, slots=True)
class VariantInputs:
    """Exact nested prefix consumed by one variant."""

    spec: SharedActorSpec
    params: SharedActorParams | IndependentActorParams
    initial_states: Array
    scenarios: CircleScenarioBatch
    validation_initial_states: Array
    validation_scenarios: CircleScenarioBatch
    model_samples: VersionAModelSamples
    safety: Any
    validation_safety: Any
    initial_params_digest: str
    content_digest: str


def inputs_for_variant(
    common: CommonFoldInputs, variant: CandidateVariant, resources: CandidateProtocolResources
) -> VariantInputs:
    """Slice K/B/R prefixes and bind their digest to the variant output."""
    _validate_shape(variant.policy_count, variant.horizon, variant.batch_size)
    spec = _slice_spec(common.spec, variant.policy_count)
    params: SharedActorParams | IndependentActorParams
    if variant.architecture == "shared":
        params = _slice_shared(common.shared_params, variant.policy_count)
    else:
        params = _slice_independent(common.independent_params, variant.policy_count)
    initial = common.initial_states[: variant.batch_size]
    scenarios = _slice_scenarios(common.scenarios, variant.batch_size)
    validation_initial = common.validation_initial_states[: variant.score_batch_size]
    validation_scenarios = _slice_scenarios(common.validation_scenarios, variant.score_batch_size)
    samples = _slice_model_samples(common.model_samples, variant.uncertainty_samples)
    safety = rigid_body_safety_batch_from_circles(
        scenarios, angular_rate_max=8.0, tilt_max_radians=1.1
    )
    validation_safety = rigid_body_safety_batch_from_circles(
        validation_scenarios, angular_rate_max=8.0, tilt_max_radians=1.1
    )
    digest = numeric_digest(
        "candidate-variant-prefix-v1",
        np.asarray(spec.base_codes),
        np.asarray(spec.base_desired_velocities),
        np.asarray(spec.base_durations),
        np.asarray(spec.adaptive_mask),
        np.asarray(initial),
        np.asarray(scenarios.obstacle_centers),
        np.asarray(scenarios.obstacle_radii),
        np.asarray(scenarios.obstacle_mask),
        np.asarray(scenarios.arena_lower),
        np.asarray(scenarios.arena_upper),
        np.asarray(scenarios.speed_limit),
        np.asarray(validation_initial),
        np.asarray(validation_scenarios.obstacle_centers),
        np.asarray(validation_scenarios.obstacle_radii),
        np.asarray(validation_scenarios.obstacle_mask),
        np.asarray(validation_scenarios.arena_lower),
        np.asarray(validation_scenarios.arena_upper),
        np.asarray(validation_scenarios.speed_limit),
        *[np.asarray(leaf) for leaf in jax.tree.leaves(samples)],
    )
    return VariantInputs(
        spec=spec,
        params=params,
        initial_states=initial,
        scenarios=scenarios,
        validation_initial_states=validation_initial,
        validation_scenarios=validation_scenarios,
        model_samples=samples,
        safety=safety,
        validation_safety=validation_safety,
        initial_params_digest=tree_content_digest(params),
        content_digest=digest,
    )


def project_shared_candidate(
    candidate: SharedActorParams,
    reference: SharedActorParams,
    adaptive_mask: Array,
    *,
    train_skill_codes: bool = False,
    train_durations: bool = False,
) -> SharedActorParams:
    """Freeze structural rows and any skill leaf disabled by the ablation contract."""
    mask = jnp.asarray(adaptive_mask, dtype=bool)
    return candidate.replace(
        code_offsets=(
            jnp.where(mask[:, None], candidate.code_offsets, reference.code_offsets)
            if train_skill_codes
            else reference.code_offsets
        ),
        velocity_offsets=jnp.where(
            mask[:, None], candidate.velocity_offsets, reference.velocity_offsets
        ),
        duration_offsets=(
            jnp.where(mask, candidate.duration_offsets, reference.duration_offsets)
            if train_durations
            else reference.duration_offsets
        ),
    )


def project_independent_candidate(
    candidate: IndependentActorParams,
    reference: IndependentActorParams,
    adaptive_mask: Array,
    *,
    train_skill_codes: bool = False,
    train_durations: bool = False,
) -> IndependentActorParams:
    """Freeze codes/durations and every tensor row owned by a structural policy."""
    mask = jnp.asarray(adaptive_mask, dtype=bool)

    def rows(proposed: Array, active: Array) -> Array:
        broadcast = mask.reshape((mask.shape[0], *(1 for _ in proposed.shape[1:])))
        return jnp.where(broadcast, proposed, active)

    return candidate.replace(
        code_offsets=(
            rows(candidate.code_offsets, reference.code_offsets)
            if train_skill_codes
            else reference.code_offsets
        ),
        velocity_offsets=rows(candidate.velocity_offsets, reference.velocity_offsets),
        duration_offsets=(
            rows(candidate.duration_offsets, reference.duration_offsets)
            if train_durations
            else reference.duration_offsets
        ),
        input_kernel=rows(candidate.input_kernel, reference.input_kernel),
        input_bias=rows(candidate.input_bias, reference.input_bias),
        hidden_kernel=rows(candidate.hidden_kernel, reference.hidden_kernel),
        hidden_bias=rows(candidate.hidden_bias, reference.hidden_bias),
        output_kernel=rows(candidate.output_kernel, reference.output_kernel),
        output_bias=rows(candidate.output_bias, reference.output_bias),
    )


class _ScalarLoss(NamedTuple):
    total: Array


def build_projected_scalar_bptt(
    objective: Any, project: Any, *, learning_rate: float, burst_steps: int
) -> BPTTFunctions:
    """Build BPTT whose projected leaves have zero gradient and zero optimizer influence."""
    if not callable(objective) or not callable(project):
        raise TypeError("objective and project must be callable")
    if not math.isfinite(learning_rate) or learning_rate <= 0 or burst_steps <= 0:
        raise ValueError("learning_rate and burst_steps must be positive")
    optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(learning_rate))

    def initialize(params: Any) -> BPTTState:
        projected = project(params)
        return BPTTState(projected, optimizer.init(projected), jnp.zeros((), dtype=jnp.int32))

    def update(state: BPTTState) -> tuple[BPTTState, BPTTStepMetrics]:
        projected = project(state.params)

        def charged(candidate: Any) -> Array:
            return objective(project(candidate))

        loss, gradients = jax.value_and_grad(charged)(projected)
        updates, proposed_optimizer = optimizer.update(
            gradients, state.optimizer_state, params=projected
        )
        proposed_params = project(optax.apply_updates(projected, updates))
        accepted = (
            jnp.isfinite(loss)
            & tree_all_finite(gradients)
            & tree_all_finite(proposed_params)
            & tree_all_finite(proposed_optimizer)
        )
        params = jax.tree.map(
            lambda proposed, current: jnp.where(accepted, proposed, current),
            proposed_params,
            projected,
        )
        optimizer_state = jax.tree.map(
            lambda proposed, current: jnp.where(accepted, proposed, current),
            proposed_optimizer,
            state.optimizer_state,
        )
        metrics = BPTTStepMetrics(
            loss=_ScalarLoss(loss),
            gradient_norm=optax.tree.norm(gradients),
            parameter_delta_norm=optax.tree.norm(
                jax.tree.map(lambda new, old: new - old, params, projected)
            ),
            update_accepted=accepted,
        )
        return BPTTState(params, optimizer_state, state.steps + 1), metrics

    step = jax.jit(update)

    @jax.jit
    def burst(state: BPTTState) -> tuple[BPTTState, BPTTStepMetrics]:
        return jax.lax.scan(lambda current, _: update(current), state, None, length=burst_steps)

    return BPTTFunctions(initialize, step, burst)


def _learning_config(variant: CandidateVariant) -> QuadLearningConfig:
    return QuadLearningConfig(dt=0.02, horizon=variant.horizon, policy_gain=1.8)


def _shared_objective(
    variant: CandidateVariant, inputs: VariantInputs, resources: CandidateProtocolResources
) -> Any:
    params = inputs.params
    if not isinstance(params, SharedActorParams):
        raise TypeError("shared objective requires SharedActorParams")
    targets = descriptor_targets_from_spec(inputs.spec)
    scales = jnp.asarray(DESCRIPTOR_SCALES, dtype=inputs.initial_states.dtype)
    learning = _learning_config(variant)
    if variant.objective_id == "objective_generic_diversity":
        return lambda candidate: generic_diversity_loss(
            candidate,
            inputs.spec,
            inputs.initial_states,
            inputs.scenarios,
            targets,
            scales,
            resources.model,
            resources.actuator,
            resources.actor_config,
            resources.quad_config,
            dt=learning.dt,
            horizon=learning.horizon,
            policy_gain=learning.policy_gain,
            config=resources.generic_config,
        )[0]
    if variant.objective_id == "objective_plcbf_aligned":
        return lambda candidate: quad_actor_library_loss(
            candidate,
            inputs.spec,
            inputs.initial_states,
            inputs.scenarios,
            inputs.safety,
            targets,
            params,
            scales,
            resources.model,
            resources.actuator,
            resources.actor_config,
            resources.quad_config,
            resources.barrier_config,
            learning,
            _loss_config_for_variant(variant, resources.plcbf_loss_config),
        )[0]
    raise ValueError(f"unsupported objective {variant.objective_id!r}")


def _loss_config_for_variant(
    variant: CandidateVariant, base: LibraryLossConfig
) -> LibraryLossConfig:
    """Apply exactly one declared PL-CBF loss-term removal."""
    if variant.loss_ablation == "full":
        return base
    field = {
        "no_redundancy": "redundancy_weight",
        "no_diversity": "diversity_weight",
        "no_trust": "trust_weight",
    }.get(variant.loss_ablation)
    if field is None:
        raise ValueError(f"unknown loss ablation {variant.loss_ablation!r}")
    selected = replace(base, **{field: 0.0})
    selected.validate()
    return selected


def _independent_objective(
    variant: CandidateVariant, inputs: VariantInputs, resources: CandidateProtocolResources
) -> Any:
    params = inputs.params
    if not isinstance(params, IndependentActorParams):
        raise TypeError("independent objective requires IndependentActorParams")
    if not bool(jax.device_get(jnp.any(inputs.spec.adaptive_mask))):
        raise ValueError("independent actor BPTT requires at least one adaptive policy slot")
    if variant.objective_id != "objective_plcbf_aligned":
        raise ValueError("independent architecture is only paired on the PL-CBF objective")
    targets = descriptor_targets_from_spec(inputs.spec)
    scales = jnp.asarray(DESCRIPTOR_SCALES, dtype=inputs.initial_states.dtype)
    learning = _learning_config(variant)
    return lambda candidate: independent_quad_actor_library_loss(
        candidate,
        inputs.spec,
        inputs.initial_states,
        inputs.scenarios,
        inputs.safety,
        targets,
        params,
        scales,
        resources.model,
        resources.actuator,
        resources.actor_config,
        resources.quad_config,
        resources.barrier_config,
        learning,
        _loss_config_for_variant(variant, resources.plcbf_loss_config),
    )[0]


def execute_proposal(
    variant: CandidateVariant,
    inputs: VariantInputs,
    resources: CandidateProtocolResources,
    *,
    seed: int,
) -> ProposalResult:
    """Execute exactly one charged proposal protocol and return its complete ledger."""
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    params = inputs.params
    if variant.architecture == "shared":
        if not isinstance(params, SharedActorParams):
            raise TypeError("shared variant received independent parameters")
        objective = _shared_objective(variant, inputs, resources)
        project = lambda candidate: project_shared_candidate(  # noqa: E731
            candidate,
            params,
            inputs.spec.adaptive_mask,
            train_skill_codes=variant.train_skill_codes,
            train_durations=variant.train_durations,
        )
    else:
        if not isinstance(params, IndependentActorParams):
            raise TypeError("independent variant received shared parameters")
        objective = _independent_objective(variant, inputs, resources)
        project = lambda candidate: project_independent_candidate(  # noqa: E731
            candidate,
            params,
            inputs.spec.adaptive_mask,
            train_skill_codes=variant.train_skill_codes,
            train_durations=variant.train_durations,
        )

    bptt = build_projected_scalar_bptt(
        objective, project, learning_rate=1e-3, burst_steps=max(variant.gradient_updates, 1)
    )
    budget = ProposalBudget(variant.objective_evaluations, variant.gradient_updates)
    if variant.proposal_method == "bptt":
        return run_bptt_only_proposal(params, bptt, (), budget, project_params=project)
    if variant.proposal_method == "sampling":
        return run_sampling_only_proposal(
            params,
            objective,
            SamplingProposalConfig(
                ProposalBudget(variant.objective_evaluations),
                seed=seed,
                relative_stddev=0.05,
                absolute_stddev=0.01,
            ),
            project_params=project,
        )
    if variant.proposal_method == "hybrid":
        return run_hybrid_proposal_bptt(
            params,
            objective,
            bptt,
            (),
            HybridProposalConfig(budget, seed=seed, relative_stddev=0.05, absolute_stddev=0.01),
            project_params=project,
        )
    raise ValueError(f"unknown proposal method {variant.proposal_method!r}")


def assert_proposal_budget_match(results: tuple[ProposalResult, ...]) -> None:
    """Require exact charged budgets across a complete proposal trio."""
    require_matched_objective_budget(*results)


def _independent_rollout_under_uncertainty(
    params: IndependentActorParams,
    inputs: VariantInputs,
    resources: CandidateProtocolResources,
    variant: CandidateVariant,
) -> UncertainQuadRolloutBatch:
    """Roll out independent actors without exposing the sampled plant to their controller."""
    models = inputs.model_samples
    sample_count = int(models.sample_valid.shape[0])
    policy_count = variant.policy_count
    batch_size = variant.score_batch_size
    scenarios = duplicate_circle_scenarios_for_samples(inputs.validation_scenarios, sample_count)
    current = jnp.broadcast_to(
        inputs.validation_initial_states[None, :, None, :],
        (policy_count, batch_size, sample_count, 13),
    )
    horizon_duration = variant.score_horizon * 0.02

    def advance(state: Array, step_index: Array) -> tuple[Array, tuple[Array, ...]]:
        flattened = state.reshape((policy_count, batch_size * sample_count, 13))
        command = independent_quad_fallback_wrenches(
            params,
            inputs.spec,
            flattened,
            scenarios,
            resources.model,
            resources.actuator,
            elapsed=step_index * 0.02,
            horizon_duration=horizon_duration,
            policy_gain=1.8,
            actor_config=resources.actor_config,
            quad_config=resources.quad_config,
        )

        def unflatten(value: Array) -> Array:
            return value.reshape((policy_count, batch_size, sample_count, *value.shape[2:]))

        commanded_wrench = unflatten(command.wrench)
        acceleration = unflatten(command.desired_acceleration)
        raw_motor = unflatten(command.raw_motor_forces)
        commanded_motor = unflatten(command.bounded_motor_forces)
        command_valid = command.input_valid.reshape((policy_count, batch_size, sample_count))
        realized_motor = commanded_motor * models.rotor_efficiency[None, None, :, :]
        realized_wrench = motor_forces_to_wrench(
            realized_motor,
            L=resources.actuator.arm_length,
            thrust2torque=resources.actuator.thrust_to_torque,
            mixing_matrix=resources.actuator.mixing_matrix,
        )

        def step_one(sample_state: Array, sample_wrench: Array, model: VersionAModel) -> Array:
            return direct_wrench_symplectic_step(sample_state, sample_wrench, model, 0.02)

        following = jax.vmap(step_one, in_axes=(2, 2, 0), out_axes=2)(
            state, realized_wrench, models.models
        )
        valid = (
            command_valid
            & models.sample_valid[None, None, :]
            & jnp.all(jnp.isfinite(following), axis=-1)
            & jnp.all(jnp.isfinite(realized_wrench), axis=-1)
        )
        following = jnp.where(valid[..., None], following, jnp.nan)
        return following, (
            following,
            commanded_wrench,
            realized_wrench,
            acceleration,
            raw_motor,
            commanded_motor,
            realized_motor,
            valid,
        )

    _, outputs = jax.lax.scan(
        advance, current, jnp.arange(variant.score_horizon, dtype=current.dtype)
    )
    moved = tuple(jnp.moveaxis(value, 0, 3) for value in outputs)
    future, commanded, realized, acceleration, raw, bounded, realized_motor, valid = moved
    states = jnp.concatenate((current[:, :, :, None, :], future), axis=3)
    return UncertainQuadRolloutBatch(
        states,
        commanded,
        realized,
        acceleration,
        raw,
        bounded,
        realized_motor,
        valid,
        models.sample_valid,
        models.weights,
        models.retained_variance_fraction,
        models.model_version,
    )


@dataclass(frozen=True, slots=True)
class CandidateHardScore:
    """Common held-out hard score; all fields are candidate-quality diagnostics."""

    minimum_library_hard_margin: float
    per_barrier_hard_margins: dict[str, float]
    safe_policy_count_minimum: int
    safe_policy_count_mean: float
    safe_policy_fraction: float
    scenario_coverage_fraction: float
    descriptor_covariance_logdet: float
    minimum_descriptor_distance: float
    feasible_fraction: float
    candidate_feasible: bool
    rollout_valid_fraction: float
    structural_exact_retention_fraction: float
    configured_frozen_exact_retention_fraction: float
    fixed_code_duration_exact: bool
    skill_code_changed_fraction: float
    duration_changed_fraction: float
    adaptive_local_non_regression_fraction: float
    adaptive_parameter_changed_fraction: float
    score_seconds: float
    hard_policy_margins: np.ndarray
    descriptors: np.ndarray
    feasibility_margins: np.ndarray


def _retention_metrics(
    candidate: SharedActorParams | IndependentActorParams,
    active: SharedActorParams | IndependentActorParams,
    spec: SharedActorSpec,
    variant: CandidateVariant,
) -> tuple[float, float, bool, float, float, float]:
    mask = np.asarray(spec.adaptive_mask, dtype=bool)
    structural_pairs: list[tuple[np.ndarray, np.ndarray]] = [
        (np.asarray(candidate.code_offsets)[~mask], np.asarray(active.code_offsets)[~mask]),
        (np.asarray(candidate.velocity_offsets)[~mask], np.asarray(active.velocity_offsets)[~mask]),
        (np.asarray(candidate.duration_offsets)[~mask], np.asarray(active.duration_offsets)[~mask]),
    ]
    configured_pairs = list(structural_pairs)
    if not variant.train_skill_codes:
        configured_pairs.append(
            (np.asarray(candidate.code_offsets)[mask], np.asarray(active.code_offsets)[mask])
        )
    if not variant.train_durations:
        configured_pairs.append(
            (
                np.asarray(candidate.duration_offsets)[mask],
                np.asarray(active.duration_offsets)[mask],
            )
        )
    if isinstance(candidate, SharedActorParams) and isinstance(active, SharedActorParams):
        shared_adaptive_pairs = [
            (
                np.asarray(candidate.velocity_offsets)[mask],
                np.asarray(active.velocity_offsets)[mask],
            ),
            (np.asarray(candidate.input_kernel), np.asarray(active.input_kernel)),
            (np.asarray(candidate.input_bias), np.asarray(active.input_bias)),
            (np.asarray(candidate.hidden_kernel), np.asarray(active.hidden_kernel)),
            (np.asarray(candidate.hidden_bias), np.asarray(active.hidden_bias)),
            (np.asarray(candidate.output_kernel), np.asarray(active.output_kernel)),
            (np.asarray(candidate.output_bias), np.asarray(active.output_bias)),
        ]
        if variant.train_skill_codes:
            shared_adaptive_pairs.append(
                (np.asarray(candidate.code_offsets)[mask], np.asarray(active.code_offsets)[mask])
            )
        if variant.train_durations:
            shared_adaptive_pairs.append(
                (
                    np.asarray(candidate.duration_offsets)[mask],
                    np.asarray(active.duration_offsets)[mask],
                )
            )
        adaptive_pairs = tuple(shared_adaptive_pairs)
    elif isinstance(candidate, IndependentActorParams) and isinstance(
        active, IndependentActorParams
    ):
        adaptive_pairs_list: list[tuple[np.ndarray, np.ndarray]] = []
        for name in (
            "velocity_offsets",
            "input_kernel",
            "input_bias",
            "hidden_kernel",
            "hidden_bias",
            "output_kernel",
            "output_bias",
        ):
            proposed = np.asarray(getattr(candidate, name))
            reference = np.asarray(getattr(active, name))
            structural_pairs.append((proposed[~mask], reference[~mask]))
            configured_pairs.append((proposed[~mask], reference[~mask]))
            adaptive_pairs_list.append((proposed[mask], reference[mask]))
        if variant.train_skill_codes:
            adaptive_pairs_list.append(
                (np.asarray(candidate.code_offsets)[mask], np.asarray(active.code_offsets)[mask])
            )
        if variant.train_durations:
            adaptive_pairs_list.append(
                (
                    np.asarray(candidate.duration_offsets)[mask],
                    np.asarray(active.duration_offsets)[mask],
                )
            )
        adaptive_pairs = tuple(adaptive_pairs_list)
    else:
        raise TypeError("candidate and active parameter architectures must match")

    structural_vector = np.concatenate(
        [np.equal(proposed, reference).reshape(-1) for proposed, reference in structural_pairs]
    )
    configured_vector = np.concatenate(
        [np.equal(proposed, reference).reshape(-1) for proposed, reference in configured_pairs]
    )
    adaptive_changed = np.concatenate(
        [np.not_equal(proposed, reference).reshape(-1) for proposed, reference in adaptive_pairs]
    )
    fixed_exact = bool(
        np.array_equal(np.asarray(candidate.code_offsets), np.asarray(active.code_offsets))
        and np.array_equal(
            np.asarray(candidate.duration_offsets), np.asarray(active.duration_offsets)
        )
    )
    code_changed = np.not_equal(
        np.asarray(candidate.code_offsets)[mask], np.asarray(active.code_offsets)[mask]
    )
    duration_changed = np.not_equal(
        np.asarray(candidate.duration_offsets)[mask], np.asarray(active.duration_offsets)[mask]
    )
    return (
        float(np.mean(structural_vector)),
        float(np.mean(configured_vector)),
        fixed_exact,
        float(np.mean(code_changed)),
        float(np.mean(duration_changed)),
        float(np.mean(adaptive_changed)),
    )


def score_candidate(
    variant: CandidateVariant,
    inputs: VariantInputs,
    candidate: SharedActorParams | IndependentActorParams,
    resources: CandidateProtocolResources,
) -> CandidateHardScore:
    """Apply the common exact R-sample hard scorer to one returned candidate."""
    start = time.perf_counter()
    if variant.architecture == "shared":
        if not isinstance(candidate, SharedActorParams):
            raise TypeError("shared score requires SharedActorParams")
        rollouts = rollout_shared_quad_library_under_uncertainty(
            candidate,
            inputs.spec,
            inputs.validation_initial_states,
            inputs.validation_scenarios,
            resources.model,
            inputs.model_samples,
            resources.actuator,
            dt=0.02,
            horizon=variant.score_horizon,
            policy_gain=1.8,
            actor_config=resources.actor_config,
            quad_config=resources.quad_config,
        )
    else:
        if not isinstance(candidate, IndependentActorParams):
            raise TypeError("independent score requires IndependentActorParams")
        rollouts = _independent_rollout_under_uncertainty(candidate, inputs, resources, variant)
    values = uncertain_quad_safety_values(
        rollouts, inputs.validation_safety, resources.barrier_config, softmin_beta=40.0
    )
    jax.block_until_ready((rollouts.states, values.robust_hard_policy_margins))

    hard = np.asarray(values.robust_hard_policy_margins, dtype=np.float64)
    library = np.max(hard, axis=0)
    safe_counts = np.sum(hard >= 0.0, axis=0)
    states = np.asarray(rollouts.states)
    translation = np.concatenate((states[..., :3], states[..., 7:10]), axis=-1)
    policy_count, batch, samples, nodes, _ = translation.shape
    descriptor_array = np.asarray(
        trajectory_descriptors(
            jnp.asarray(translation.reshape(policy_count, batch * samples, nodes, 6))
        )
    ).reshape(policy_count, batch, samples, 9)
    descriptors = np.mean(descriptor_array, axis=(1, 2))
    normalized = descriptors / np.asarray(DESCRIPTOR_SCALES)
    centered = normalized - np.mean(normalized, axis=0, keepdims=True)
    covariance = centered.T @ centered / policy_count + 1e-4 * np.eye(9)
    sign, logdet = np.linalg.slogdet(covariance)
    distances = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)

    lower = np.broadcast_to(np.asarray(resources.actuator.thrust_min), (4,))
    upper = np.broadcast_to(np.asarray(resources.actuator.thrust_max), (4,))
    motors = np.asarray(rollouts.commanded_motor_forces)
    feasibility = np.minimum(motors - lower, upper - motors)
    valid = np.asarray(rollouts.policy_valid, dtype=bool)
    feasible = np.all(feasibility >= -1e-7, axis=-1) & valid
    (
        structural,
        configured_frozen,
        fixed_exact,
        code_changed,
        duration_changed,
        adaptive_changed,
    ) = _retention_metrics(candidate, inputs.params, inputs.spec, variant)
    active_score = _score_active_hard_margins(variant, inputs, resources)
    adaptive = np.asarray(inputs.spec.adaptive_mask, dtype=bool)
    local_retained = hard[adaptive] >= active_score[adaptive] - 1e-7

    node = np.asarray(values.node_values, dtype=np.float64)
    node_enabled = np.asarray(values.node_enabled, dtype=bool)
    masked = np.where(node_enabled, node, np.inf)
    policy_barrier = np.min(masked, axis=3)
    library_barrier = np.max(policy_barrier, axis=0)
    per_barrier = np.min(library_barrier, axis=(0, 1))
    names = safety_constraint_names(inputs.validation_scenarios.obstacle_centers.shape[1])
    segment = np.asarray(values.segment_obstacle_values, dtype=np.float64)
    segment_enabled = np.asarray(values.segment_obstacle_enabled, dtype=bool)
    masked_segment = np.where(segment_enabled, segment, np.inf)
    policy_segment = np.min(masked_segment, axis=3)
    library_segment = np.max(policy_segment, axis=0)
    segment_margin = np.min(library_segment, axis=(0, 1))
    margin_mapping = dict(zip(names, per_barrier.tolist(), strict=True))
    margin_mapping.update(
        {f"swept_obstacle_{index}": float(value) for index, value in enumerate(segment_margin)}
    )
    score_seconds = time.perf_counter() - start
    return CandidateHardScore(
        minimum_library_hard_margin=float(np.min(library)),
        per_barrier_hard_margins=margin_mapping,
        safe_policy_count_minimum=int(np.min(safe_counts)),
        safe_policy_count_mean=float(np.mean(safe_counts)),
        safe_policy_fraction=float(np.mean(hard >= 0.0)),
        scenario_coverage_fraction=float(np.mean(library >= 0.0)),
        descriptor_covariance_logdet=float(logdet if sign > 0 else -np.inf),
        minimum_descriptor_distance=float(np.min(distances)),
        feasible_fraction=float(np.mean(feasible)),
        candidate_feasible=bool(np.all(feasible)),
        rollout_valid_fraction=float(np.mean(valid)),
        structural_exact_retention_fraction=structural,
        configured_frozen_exact_retention_fraction=configured_frozen,
        fixed_code_duration_exact=fixed_exact,
        skill_code_changed_fraction=code_changed,
        duration_changed_fraction=duration_changed,
        adaptive_local_non_regression_fraction=float(np.mean(local_retained)),
        adaptive_parameter_changed_fraction=adaptive_changed,
        score_seconds=score_seconds,
        hard_policy_margins=hard,
        descriptors=descriptors,
        feasibility_margins=feasibility,
    )


def _score_active_hard_margins(
    variant: CandidateVariant, inputs: VariantInputs, resources: CandidateProtocolResources
) -> np.ndarray:
    """Score the incumbent through the same hard path without recursively deriving metrics."""
    if variant.architecture == "shared":
        if not isinstance(inputs.params, SharedActorParams):
            raise TypeError("shared variant requires SharedActorParams")
        rollout = rollout_shared_quad_library_under_uncertainty(
            inputs.params,
            inputs.spec,
            inputs.validation_initial_states,
            inputs.validation_scenarios,
            resources.model,
            inputs.model_samples,
            resources.actuator,
            dt=0.02,
            horizon=variant.score_horizon,
            policy_gain=1.8,
            actor_config=resources.actor_config,
            quad_config=resources.quad_config,
        )
    else:
        if not isinstance(inputs.params, IndependentActorParams):
            raise TypeError("independent variant requires IndependentActorParams")
        rollout = _independent_rollout_under_uncertainty(inputs.params, inputs, resources, variant)
    values = uncertain_quad_safety_values(
        rollout, inputs.validation_safety, resources.barrier_config, softmin_beta=40.0
    )
    return np.asarray(values.robust_hard_policy_margins, dtype=np.float64)


def validation_report_for_score(
    variant: CandidateVariant,
    inputs: VariantInputs,
    candidate: SharedActorParams | IndependentActorParams,
    score: CandidateHardScore,
    resources: CandidateProtocolResources,
) -> ValidationReport:
    """Create the real hard-gate report, retaining rejection rather than forcing admission."""
    active_hard = _score_active_hard_margins(variant, inputs, resources)
    core = {
        "base_codes": np.asarray(inputs.spec.base_codes[:STRUCTURAL_POLICY_COUNT]),
        "base_desired_velocities": np.asarray(
            inputs.spec.base_desired_velocities[:STRUCTURAL_POLICY_COUNT]
        ),
        "base_durations": np.asarray(inputs.spec.base_durations[:STRUCTURAL_POLICY_COUNT]),
    }
    active = create_active_snapshot(
        inputs.params,
        version=0,
        model_version=0,
        structural_core=core,
        metadata={"scope": "candidate-quality-ablation"},
    )
    candidate_snapshot = create_candidate_snapshot(
        candidate, version=1, base_active=active, metadata={"variant_id": variant.variant_id}
    )
    evidence = HardValidationEvidence(
        current_policy_margins=score.hard_policy_margins[:, 0],
        candidate_local_policy_margins=score.hard_policy_margins,
        active_local_policy_margins=active_hard,
        candidate_descriptors=score.descriptors,
        descriptor_scales=np.asarray(DESCRIPTOR_SCALES),
        feasibility_margins=score.feasibility_margins,
        runtime_seconds=np.asarray([score.score_seconds]),
        validation_set_digest=inputs.content_digest,
    )
    return hard_validate_candidate(
        active,
        candidate_snapshot,
        evidence,
        HardValidationThresholds(
            minimum_current_margin=0.0,
            safe_policy_margin=0.0,
            local_non_regression_tolerance=0.0,
            minimum_coverage=1.0,
            minimum_redundancy=1,
            minimum_diversity=0.0,
            minimum_feasible_fraction=1.0,
            maximum_runtime_seconds=600.0,
        ),
        current_model_version=0,
    )


@dataclass(frozen=True, slots=True)
class VariantExecution:
    """One proposal, common hard score, and exact hard-gate report."""

    variant: CandidateVariant
    fold: int
    common_fold_digest: str
    input_prefix_digest: str
    initial_params_digest: str
    candidate_params: SharedActorParams | IndependentActorParams
    candidate_params_digest: str
    proposal: ProposalResult
    hard_score: CandidateHardScore
    validation_report: ValidationReport
    admission_mode: str
    protocol_admission_accepted: bool


def execute_variant(
    variant: CandidateVariant,
    common: CommonFoldInputs,
    resources: CandidateProtocolResources,
    *,
    seed: int,
) -> VariantExecution:
    """Run the complete candidate-only vertical slice for one fold/variant."""
    inputs = inputs_for_variant(common, variant, resources)
    proposal = execute_proposal(variant, inputs, resources, seed=seed)
    score = score_candidate(variant, inputs, proposal.params, resources)
    report = validation_report_for_score(variant, inputs, proposal.params, score, resources)
    if variant.validation_gate_enabled:
        admission_mode = "hard_validation_gate"
        admitted = report.passed
    else:
        admission_mode = "hard_validation_gate_bypassed_for_ablation"
        admitted = proposal.input_valid and all(
            bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in jax.tree.leaves(proposal.params)
        )
    return VariantExecution(
        variant=variant,
        fold=common.fold,
        common_fold_digest=common.content_digest,
        input_prefix_digest=inputs.content_digest,
        initial_params_digest=inputs.initial_params_digest,
        candidate_params=proposal.params,
        candidate_params_digest=tree_content_digest(proposal.params),
        proposal=proposal,
        hard_score=score,
        validation_report=report,
        admission_mode=admission_mode,
        protocol_admission_accepted=admitted,
    )


def proposal_result_mapping(result: ProposalResult) -> dict[str, Any]:
    """Convert a proposal result into a JSON-compatible exact evaluation ledger."""
    if result.accounting.sampling_evaluations > 0 and result.accounting.final_evaluations == 0:
        selected_semantics = "best_finite_charged_sampling_loss"
    elif result.accounting.final_evaluations > 0:
        selected_semantics = "best_of_charged_sampling_seed_and_charged_final_bptt_loss"
    else:
        selected_semantics = (
            "last_charged_pre_update_bptt_loss; returned post_update_params_not_evaluated"
        )
    return {
        "selected_loss": result.selected_loss,
        "selected_loss_semantics": selected_semantics,
        "incumbent_loss_first_charged": result.incumbent_loss,
        "selected_index": result.selected_index,
        "improved_on_charged_losses": result.improved,
        "input_valid": result.input_valid,
        "seed": result.seed,
        "evaluation_ledger": asdict(result.accounting),
        "raw_timing_seconds": asdict(result.timing),
        "candidate_losses": np.asarray(result.candidate_losses).tolist(),
        "gradient_losses": np.asarray(result.gradient_losses).tolist(),
        "gradient_update_accepted": np.asarray(
            result.gradient_update_accepted, dtype=bool
        ).tolist(),
        "post_update_objective_evaluated": result.accounting.final_evaluations > 0,
    }


def hard_score_mapping(score: CandidateHardScore) -> dict[str, Any]:
    """Return scalar hard metrics; dense arrays are saved in the variant NPZ artifact."""
    return {
        field: value
        for field, value in asdict(score).items()
        if field not in {"hard_policy_margins", "descriptors", "feasibility_margins"}
    }


def validation_report_mapping(report: ValidationReport) -> dict[str, Any]:
    """Serialize every digest-bound report field for independent offline verification."""
    return {
        "schema_version": 1,
        "active_digest": report.active_digest,
        "active_version": report.active_version,
        "candidate_digest": report.candidate_digest,
        "candidate_version": report.candidate_version,
        "model_version": report.model_version,
        "validation_set_digest": report.validation_set_digest,
        "passed": report.passed,
        "failed_gate_names": list(report.failed_gate_names),
        "digest": report.digest,
        "integrity_verified": report.verify_integrity(),
        "gates": list(report.as_log_records()),
        "candidate_local_best": list(report.candidate_local_best),
        "active_local_best": list(report.active_local_best),
        "local_non_regression_passes": list(report.local_non_regression_passes),
    }


def validation_report_from_mapping(value: dict[str, Any]) -> ValidationReport:
    """Reconstruct and verify a canonical validation report from persisted JSON evidence."""
    expected_fields = {
        "schema_version",
        "active_digest",
        "active_version",
        "candidate_digest",
        "candidate_version",
        "model_version",
        "validation_set_digest",
        "passed",
        "failed_gate_names",
        "digest",
        "integrity_verified",
        "gates",
        "candidate_local_best",
        "active_local_best",
        "local_non_regression_passes",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("validation report mapping has the wrong schema")
    if value["schema_version"] != 1:
        raise ValueError("unsupported validation report mapping schema")

    def integer(name: str) -> int:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"validation report {name} must be a nonnegative integer")
        return item

    def text_field(name: str) -> str:
        item = value[name]
        if not isinstance(item, str) or not item:
            raise ValueError(f"validation report {name} must be a nonempty string")
        return item

    raw_gates = value["gates"]
    if not isinstance(raw_gates, list):
        raise ValueError("validation report gates must be a list")
    gates: list[GateResult] = []
    gate_fields = {"name", "passed", "observed", "requirement", "detail"}
    for raw_gate in raw_gates:
        if not isinstance(raw_gate, dict) or set(raw_gate) != gate_fields:
            raise ValueError("validation report gate has the wrong schema")
        if not isinstance(raw_gate["passed"], bool):
            raise ValueError("validation report gate passed flag must be boolean")
        if not all(isinstance(raw_gate[name], str) for name in gate_fields - {"passed"}):
            raise ValueError("validation report gate text fields must be strings")
        gates.append(
            GateResult(
                name=raw_gate["name"],
                passed=raw_gate["passed"],
                observed=raw_gate["observed"],
                requirement=raw_gate["requirement"],
                detail=raw_gate["detail"],
            )
        )

    def finite_float_list(name: str) -> tuple[float, ...]:
        items = value[name]
        if not isinstance(items, list):
            raise ValueError(f"validation report {name} must be a list")
        converted: list[float] = []
        for item in items:
            if isinstance(item, bool):
                raise ValueError(f"validation report {name} must contain numbers")
            try:
                converted_item = float(item)
            except (TypeError, ValueError) as error:
                raise ValueError(f"validation report {name} must contain numbers") from error
            if not math.isfinite(converted_item):
                raise ValueError(f"validation report {name} must contain finite numbers")
            converted.append(converted_item)
        return tuple(converted)

    raw_passes = value["local_non_regression_passes"]
    if not isinstance(raw_passes, list) or any(not isinstance(item, bool) for item in raw_passes):
        raise ValueError("validation report local passes must be a boolean list")
    if not isinstance(value["passed"], bool) or not isinstance(value["integrity_verified"], bool):
        raise ValueError("validation report integrity/pass flags must be boolean")
    failed = value["failed_gate_names"]
    if not isinstance(failed, list) or any(not isinstance(item, str) for item in failed):
        raise ValueError("validation report failed gates must be a string list")

    report = ValidationReport(
        active_digest=text_field("active_digest"),
        active_version=integer("active_version"),
        candidate_digest=text_field("candidate_digest"),
        candidate_version=integer("candidate_version"),
        model_version=integer("model_version"),
        validation_set_digest=text_field("validation_set_digest"),
        gates=tuple(gates),
        candidate_local_best=finite_float_list("candidate_local_best"),
        active_local_best=finite_float_list("active_local_best"),
        local_non_regression_passes=tuple(raw_passes),
        digest=text_field("digest"),
    )
    if not report.verify_integrity() or value["integrity_verified"] is not True:
        raise ValueError("validation report failed digest integrity verification")
    if value["passed"] != report.passed:
        raise ValueError("validation report passed flag disagrees with its gates")
    if tuple(failed) != report.failed_gate_names:
        raise ValueError("validation report failed-gate names disagree with its gates")
    return report


def numeric_digest(domain: str, *arrays: np.ndarray) -> str:
    """Hash exact array names, dtypes, shapes, and bytes under a domain separator."""
    digest = hashlib.sha256(domain.encode("utf-8") + b"\0")
    for index, value in enumerate(arrays):
        array = np.ascontiguousarray(np.asarray(value))
        header = json.dumps(
            {"index": index, "dtype": array.dtype.str, "shape": array.shape},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


__all__ = [
    "ArchitecturePoint",
    "CANDIDATE_ONLY_CLAIM_BOUNDARY",
    "CandidateHardScore",
    "CandidateProtocolResources",
    "CandidateStudyProfile",
    "CandidateVariant",
    "CommonFoldInputs",
    "ProposalPoint",
    "VariantExecution",
    "VariantInputs",
    "assert_proposal_budget_match",
    "build_candidate_protocol_resources",
    "build_projected_scalar_bptt",
    "candidate_study_profile",
    "deterministic_model_samples",
    "execute_proposal",
    "execute_variant",
    "generate_common_fold_inputs",
    "hard_score_mapping",
    "inputs_for_variant",
    "numeric_digest",
    "project_independent_candidate",
    "project_shared_candidate",
    "proposal_result_mapping",
    "score_candidate",
    "validation_report_for_score",
    "validation_report_from_mapping",
    "validation_report_mapping",
    "variants_for_profile",
]
