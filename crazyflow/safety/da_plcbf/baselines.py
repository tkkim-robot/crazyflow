"""Stable Phase-7 baseline and ablation contracts for DA-PLCBF experiments.

This module is intentionally a registry and validation boundary, not an experiment runner.  It
prevents two easy scientific errors: silently changing more than one factor in an ablation and
giving an unavailable or only "style-like" comparator a stronger algorithm name.  Runtime code may
serialize :func:`method_config_metadata` directly into an artifact manifest.

The action helpers at the bottom are deliberately small.  They expose nominal pass-through and
hard-certificate fallback selection using the audited selector already implemented in this
package.  Analytic CBF/HOCBF projection and the Version-A/Version-B filters remain in their
respective modules; this registry does not invent substitute control laws for them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.selector import PolicySelection, SelectionConfig, select_hard_policy


class MethodID(str, Enum):
    """Stable, filesystem-safe identifiers for the seven Phase-7 core methods."""

    NOMINAL_ONLY = "nominal_only"
    ANALYTIC_CBF_HOCBF = "analytic_cbf_hocbf"
    FIXED_FALLBACK_PCBF = "fixed_fallback_pcbf"
    HANDCRAFTED_FIXED_LIBRARY_PLCBF = "handcrafted_fixed_library_plcbf"
    OFFLINE_FROZEN_SDCBF_STYLE = "offline_frozen_sdcbf_style"
    DA_PLCBF_NO_ONLINE_MODEL_ADAPTATION = "da_plcbf_no_online_model_adaptation"
    DA_PLCBF_FULL = "da_plcbf_full"


class TrainingStrategy(str, Enum):
    """Candidate-library training strategy; SHAC is never implied by BPTT."""

    NONE = "none"
    BPTT = "bptt"
    SAMPLING_ONLY = "sampling_only"
    HYBRID_PROPOSAL_BPTT = "hybrid_proposal_bptt"
    SHAC = "shac"


class LibraryObjective(str, Enum):
    """Library objective family used for the Phase-7 objective ablation."""

    NONE = "none"
    GENERIC_DIVERSITY = "generic_diversity"
    PLCBF_ALIGNED = "plcbf_aligned_coverage_diversity"


class PolicyArchitecture(str, Enum):
    """Policy-library parameterization used during training and evaluation."""

    NONE = "none"
    SINGLE_FIXED = "single_fixed_policy"
    HANDCRAFTED_FIXED = "handcrafted_fixed_library"
    SHARED_ACTOR = "shared_actor"
    INDEPENDENT_ACTORS = "independent_actors"


class SourceFidelity(str, Enum):
    """Maximum source-fidelity claim encoded by a registered comparator."""

    NOT_APPLICABLE = "not_applicable"
    CLEAN_ROOM_EQUATIONS = "clean_room_equations"
    STYLE_ONLY = "style_only"
    PROJECT_IMPLEMENTATION = "project_implementation"


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """Immutable capability and claim boundary for one stable core method."""

    method_id: MethodID
    display_name: str
    filter_family: str
    uses_policy_library: bool
    learned_library: bool
    online_library_updates: bool
    online_model_updates: bool
    runtime_hard_postcheck_required: bool
    source_fidelity: SourceFidelity
    source_label: str
    claim_boundaries: tuple[str, ...]

    def metadata(self) -> dict[str, object]:
        """Return a JSON-compatible description with stable field names."""
        return {
            "method_id": self.method_id.value,
            "display_name": self.display_name,
            "filter_family": self.filter_family,
            "uses_policy_library": self.uses_policy_library,
            "learned_library": self.learned_library,
            "online_library_updates": self.online_library_updates,
            "online_model_updates": self.online_model_updates,
            "runtime_hard_postcheck_required": self.runtime_hard_postcheck_required,
            "source_fidelity": self.source_fidelity.value,
            "source_label": self.source_label,
            "claim_boundaries": list(self.claim_boundaries),
        }


_METHOD_SPECS_IN_ORDER = (
    MethodSpec(
        MethodID.NOMINAL_ONLY,
        "Nominal only",
        "none",
        False,
        False,
        False,
        False,
        False,
        SourceFidelity.NOT_APPLICABLE,
        "Crazyflow nominal controller without a safety filter",
        (
            "Tracking comparator only; it carries no safety certificate.",
            "A safe outcome must not be reported as evidence of a safety-filter guarantee.",
        ),
    ),
    MethodSpec(
        MethodID.ANALYTIC_CBF_HOCBF,
        "Analytic distance CBF/HOCBF",
        "analytic_cbf_hocbf",
        False,
        False,
        False,
        False,
        True,
        SourceFidelity.CLEAN_ROOM_EQUATIONS,
        "Project analytic distance/speed/rate/tilt CBF and HOCBF equations",
        (
            "Covers only the explicitly configured analytic barriers.",
            "It is not a finite-horizon policy-library or learned-filter method.",
        ),
    ),
    MethodSpec(
        MethodID.FIXED_FALLBACK_PCBF,
        "One fixed-fallback PCBF",
        "policy_cbf",
        True,
        False,
        False,
        False,
        True,
        SourceFidelity.CLEAN_ROOM_EQUATIONS,
        "Clean-room finite-horizon PCBF equations with exactly one fixed fallback",
        (
            "Exactly one fixed fallback policy is permitted.",
            "No library coverage, learned-library, or online-adaptation claim is permitted.",
        ),
    ),
    MethodSpec(
        MethodID.HANDCRAFTED_FIXED_LIBRARY_PLCBF,
        "Handcrafted fixed-library PL-CBF",
        "policy_library_cbf",
        True,
        False,
        False,
        False,
        True,
        SourceFidelity.CLEAN_ROOM_EQUATIONS,
        "Clean-room PL-CBF equations with a fixed handcrafted policy library",
        (
            "The library is fixed and handcrafted for the complete evaluation.",
            "No learned-library or online-adaptation claim is permitted.",
        ),
    ),
    MethodSpec(
        MethodID.OFFLINE_FROZEN_SDCBF_STYLE,
        "Offline/frozen learned library (SDCBF-style comparison only)",
        "frozen_learned_policy_library_cbf",
        True,
        True,
        False,
        False,
        True,
        SourceFidelity.STYLE_ONLY,
        "Phase-7 SDCBF-style offline/frozen protocol; no SDCBF source implementation integrated",
        (
            "This is a clean-room offline/frozen learned-library comparator only.",
            "It must not be labeled an SDCBF reproduction or attributed SDCBF-specific claims.",
            "Its policy parameters and dynamics model remain frozen during evaluation.",
        ),
    ),
    MethodSpec(
        MethodID.DA_PLCBF_NO_ONLINE_MODEL_ADAPTATION,
        "DA-PLCBF without online model adaptation",
        "adaptive_policy_library_cbf",
        True,
        True,
        True,
        False,
        True,
        SourceFidelity.PROJECT_IMPLEMENTATION,
        "Project DA-PLCBF implementation with a fixed evaluation dynamics model",
        (
            "Candidate policy-library updates may run, but estimator-driven model updates may not.",
            "Results cannot be used to claim benefit from online dynamics-model adaptation.",
            "Certificates remain finite-horizon and conditional on sampled validation scenarios.",
        ),
    ),
    MethodSpec(
        MethodID.DA_PLCBF_FULL,
        "Full DA-PLCBF",
        "adaptive_policy_library_cbf",
        True,
        True,
        True,
        True,
        True,
        SourceFidelity.PROJECT_IMPLEMENTATION,
        "Project DA-PLCBF implementation",
        (
            "Claims are limited to the configured finite horizon, plant model, and sampled trials.",
            "A sampled simulation certificate is not a universal continuous-time or "
            "real-world guarantee.",
            "Ablated configurations must be reported as ablations, not as the full method.",
        ),
    ),
)

METHOD_SPECS = MappingProxyType({item.method_id: item for item in _METHOD_SPECS_IN_ORDER})
"""Read-only mapping from stable method ID to immutable method specification."""


@dataclass(frozen=True, slots=True)
class AblationConfig:
    """All Phase-7 train/objective/component/scale ablation axes.

    Counts refer to fixed-shape policy rollouts: ``policy_count=K``,
    ``training_scenario_count=B``, and ``uncertainty_sample_count=R``.  The adaptation budget is
    the number of optimizer updates allowed for one online candidate burst; it is zero for methods
    whose library is frozen during evaluation.
    """

    training_strategy: TrainingStrategy
    objective: LibraryObjective
    use_redundancy: bool
    use_diversity: bool
    use_trust: bool
    use_validation_gate: bool
    use_uncertainty_sampling: bool
    trainable_skill_codes: bool
    trainable_durations: bool
    policy_count: int
    horizon: int
    training_scenario_count: int
    uncertainty_sample_count: int
    adaptation_budget: int
    architecture: PolicyArchitecture

    def metadata(self) -> dict[str, object]:
        """Return primitive-valued metadata suitable for canonical JSON."""
        return {
            "training_strategy": self.training_strategy.value,
            "objective": self.objective.value,
            "use_redundancy": self.use_redundancy,
            "use_diversity": self.use_diversity,
            "use_trust": self.use_trust,
            "use_validation_gate": self.use_validation_gate,
            "use_uncertainty_sampling": self.use_uncertainty_sampling,
            "trainable_skill_codes": self.trainable_skill_codes,
            "trainable_durations": self.trainable_durations,
            "policy_count": self.policy_count,
            "horizon": self.horizon,
            "training_scenario_count": self.training_scenario_count,
            "uncertainty_sample_count": self.uncertainty_sample_count,
            "adaptation_budget": self.adaptation_budget,
            "architecture": self.architecture.value,
        }


@dataclass(frozen=True, slots=True)
class MethodConfig:
    """One core method plus an explicit set of ablation-axis values."""

    method_id: MethodID
    ablation: AblationConfig


@dataclass(frozen=True, slots=True)
class ImplementationAvailability:
    """Faithful implementations that the experiment runner can actually dispatch.

    Optional comparison algorithms default to unavailable.  A runner that actually binds the
    project sampling, hybrid, and independent-actor implementations may opt in through
    ``proposal_ablations.implemented_proposal_ablation_availability``.  SHAC remains unavailable:
    BPTT is not relabeled as SHAC.  Changing a field is a provenance assertion by the runner.
    """

    bptt_training: bool = True
    faithful_sampling_only_training: bool = False
    faithful_hybrid_proposal_bptt_training: bool = False
    faithful_shac_training: bool = False
    shared_actor: bool = True
    faithful_independent_actors: bool = False
    candidate_validation: bool = True
    uncertainty_rollouts: bool = True
    online_library_updates: bool = True
    online_model_updates: bool = True
    exact_runtime_postcheck: bool = True

    def metadata(self) -> dict[str, bool]:
        """Return stable primitive-valued capability metadata."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


def _static_ablation(policy_count: int, architecture: PolicyArchitecture) -> AblationConfig:
    horizon = 0 if policy_count == 0 else 50
    return AblationConfig(
        training_strategy=TrainingStrategy.NONE,
        objective=LibraryObjective.NONE,
        use_redundancy=False,
        use_diversity=False,
        use_trust=False,
        use_validation_gate=False,
        use_uncertainty_sampling=False,
        trainable_skill_codes=False,
        trainable_durations=False,
        policy_count=policy_count,
        horizon=horizon,
        training_scenario_count=0,
        uncertainty_sample_count=0,
        adaptation_budget=0,
        architecture=architecture,
    )


_CANONICAL_CONFIGS = MappingProxyType(
    {
        MethodID.NOMINAL_ONLY: MethodConfig(
            MethodID.NOMINAL_ONLY, _static_ablation(0, PolicyArchitecture.NONE)
        ),
        MethodID.ANALYTIC_CBF_HOCBF: MethodConfig(
            MethodID.ANALYTIC_CBF_HOCBF, _static_ablation(0, PolicyArchitecture.NONE)
        ),
        MethodID.FIXED_FALLBACK_PCBF: MethodConfig(
            MethodID.FIXED_FALLBACK_PCBF, _static_ablation(1, PolicyArchitecture.SINGLE_FIXED)
        ),
        MethodID.HANDCRAFTED_FIXED_LIBRARY_PLCBF: MethodConfig(
            MethodID.HANDCRAFTED_FIXED_LIBRARY_PLCBF,
            _static_ablation(64, PolicyArchitecture.HANDCRAFTED_FIXED),
        ),
        MethodID.OFFLINE_FROZEN_SDCBF_STYLE: MethodConfig(
            MethodID.OFFLINE_FROZEN_SDCBF_STYLE,
            AblationConfig(
                training_strategy=TrainingStrategy.BPTT,
                objective=LibraryObjective.GENERIC_DIVERSITY,
                use_redundancy=False,
                use_diversity=True,
                use_trust=False,
                use_validation_gate=False,
                use_uncertainty_sampling=False,
                trainable_skill_codes=True,
                trainable_durations=True,
                policy_count=64,
                horizon=50,
                training_scenario_count=64,
                uncertainty_sample_count=0,
                adaptation_budget=0,
                architecture=PolicyArchitecture.SHARED_ACTOR,
            ),
        ),
        MethodID.DA_PLCBF_NO_ONLINE_MODEL_ADAPTATION: MethodConfig(
            MethodID.DA_PLCBF_NO_ONLINE_MODEL_ADAPTATION,
            AblationConfig(
                training_strategy=TrainingStrategy.BPTT,
                objective=LibraryObjective.PLCBF_ALIGNED,
                use_redundancy=True,
                use_diversity=True,
                use_trust=True,
                use_validation_gate=True,
                use_uncertainty_sampling=True,
                trainable_skill_codes=True,
                trainable_durations=True,
                policy_count=64,
                horizon=50,
                training_scenario_count=64,
                uncertainty_sample_count=4,
                adaptation_budget=10,
                architecture=PolicyArchitecture.SHARED_ACTOR,
            ),
        ),
        MethodID.DA_PLCBF_FULL: MethodConfig(
            MethodID.DA_PLCBF_FULL,
            AblationConfig(
                training_strategy=TrainingStrategy.BPTT,
                objective=LibraryObjective.PLCBF_ALIGNED,
                use_redundancy=True,
                use_diversity=True,
                use_trust=True,
                use_validation_gate=True,
                use_uncertainty_sampling=True,
                trainable_skill_codes=True,
                trainable_durations=True,
                policy_count=64,
                horizon=50,
                training_scenario_count=64,
                uncertainty_sample_count=4,
                adaptation_budget=10,
                architecture=PolicyArchitecture.SHARED_ACTOR,
            ),
        ),
    }
)


def method_spec(method_id: MethodID | str) -> MethodSpec:
    """Return a registered specification, rejecting unknown or unstable IDs."""
    try:
        normalized = method_id if isinstance(method_id, MethodID) else MethodID(method_id)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown DA-PLCBF method ID: {method_id!r}") from error
    return METHOD_SPECS[normalized]


def core_method_ids() -> tuple[str, ...]:
    """Return all seven stable method IDs in the Phase-7 reporting order."""
    return tuple(item.method_id.value for item in _METHOD_SPECS_IN_ORDER)


def canonical_method_config(method_id: MethodID | str) -> MethodConfig:
    """Return the immutable, non-ablated configuration for a core registry method."""
    return _CANONICAL_CONFIGS[method_spec(method_id).method_id]


def _require_bool_fields(instance: object, names: tuple[str, ...]) -> None:
    for name in names:
        if not isinstance(getattr(instance, name), bool):
            raise TypeError(f"{name} must be boolean")


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _validate_availability(availability: ImplementationAvailability) -> None:
    if not isinstance(availability, ImplementationAvailability):
        raise TypeError("availability must be an ImplementationAvailability")
    _require_bool_fields(availability, tuple(item.name for item in fields(availability)))


def validate_method_config(
    config: MethodConfig, availability: ImplementationAvailability | None = None
) -> None:
    """Reject semantic conflicts, unsupported algorithms, and mislabeled core methods.

    Disabling a full-method component such as validation or uncertainty is valid because Phase 7
    explicitly requires those ablations.  Such a configuration is tagged by
    :func:`ablation_tags` and is no longer a claim-eligible full-method configuration.
    """
    if not isinstance(config, MethodConfig):
        raise TypeError("config must be a MethodConfig")
    if not isinstance(config.method_id, MethodID):
        raise TypeError("config.method_id must be a MethodID")
    if not isinstance(config.ablation, AblationConfig):
        raise TypeError("config.ablation must be an AblationConfig")
    selected_availability = availability or ImplementationAvailability()
    _validate_availability(selected_availability)
    spec = method_spec(config.method_id)
    item = config.ablation

    enum_values = (
        (item.training_strategy, TrainingStrategy, "training_strategy"),
        (item.objective, LibraryObjective, "objective"),
        (item.architecture, PolicyArchitecture, "architecture"),
    )
    for value, enum_type, name in enum_values:
        if not isinstance(value, enum_type):
            raise TypeError(f"{name} must be a {enum_type.__name__}")
    boolean_names = (
        "use_redundancy",
        "use_diversity",
        "use_trust",
        "use_validation_gate",
        "use_uncertainty_sampling",
        "trainable_skill_codes",
        "trainable_durations",
    )
    _require_bool_fields(item, boolean_names)
    counts = {
        name: _require_nonnegative_int(getattr(item, name), name)
        for name in (
            "policy_count",
            "horizon",
            "training_scenario_count",
            "uncertainty_sample_count",
            "adaptation_budget",
        )
    }

    if spec.uses_policy_library:
        if counts["policy_count"] < 1 or counts["horizon"] < 1:
            raise ValueError("policy-library methods require positive policy_count and horizon")
    elif counts["policy_count"] != 0 or counts["horizon"] != 0:
        raise ValueError("methods without a policy library require policy_count=horizon=0")

    if config.method_id is MethodID.FIXED_FALLBACK_PCBF and counts["policy_count"] != 1:
        raise ValueError("fixed_fallback_pcbf requires exactly one policy")
    if config.method_id is MethodID.HANDCRAFTED_FIXED_LIBRARY_PLCBF and counts["policy_count"] < 2:
        raise ValueError("a handcrafted policy library requires at least two policies")

    if spec.learned_library:
        if item.training_strategy is TrainingStrategy.NONE:
            raise ValueError("learned-library methods require an explicit training strategy")
        if item.objective is LibraryObjective.NONE:
            raise ValueError("learned-library methods require an explicit library objective")
        if item.architecture not in (
            PolicyArchitecture.SHARED_ACTOR,
            PolicyArchitecture.INDEPENDENT_ACTORS,
        ):
            raise ValueError("learned libraries require shared or independent actor architecture")
        if counts["training_scenario_count"] < 1:
            raise ValueError("learned libraries require at least one training scenario")
    else:
        forbidden_static = (
            item.training_strategy is not TrainingStrategy.NONE
            or item.objective is not LibraryObjective.NONE
            or item.use_redundancy
            or item.use_diversity
            or item.use_trust
            or item.use_validation_gate
            or item.use_uncertainty_sampling
            or item.trainable_skill_codes
            or item.trainable_durations
            or counts["training_scenario_count"] != 0
            or counts["uncertainty_sample_count"] != 0
            or counts["adaptation_budget"] != 0
        )
        if forbidden_static:
            raise ValueError("fixed/non-library baselines cannot enable learned-library ablations")

    if item.objective is LibraryObjective.GENERIC_DIVERSITY and item.use_redundancy:
        raise ValueError(
            "the PL-CBF redundancy term is incompatible with generic-diversity objective"
        )
    if item.objective is LibraryObjective.GENERIC_DIVERSITY and not item.use_diversity:
        raise ValueError("generic-diversity objective requires use_diversity=True")

    if spec.online_library_updates:
        if counts["adaptation_budget"] < 1:
            raise ValueError("online-library methods require a positive adaptation_budget")
    elif counts["adaptation_budget"] != 0:
        raise ValueError("frozen/static methods require adaptation_budget=0")

    if item.use_validation_gate and not spec.online_library_updates:
        raise ValueError("candidate validation gate is meaningful only for online library updates")
    if item.use_trust and not spec.online_library_updates:
        raise ValueError("active/candidate trust is meaningful only for online library updates")
    if item.use_uncertainty_sampling:
        if counts["uncertainty_sample_count"] < 1:
            raise ValueError("uncertainty sampling requires uncertainty_sample_count >= 1")
    elif counts["uncertainty_sample_count"] != 0:
        raise ValueError("uncertainty_sample_count must be zero when sampling is disabled")

    strategy_capability = {
        TrainingStrategy.NONE: True,
        TrainingStrategy.BPTT: selected_availability.bptt_training,
        TrainingStrategy.SAMPLING_ONLY: selected_availability.faithful_sampling_only_training,
        TrainingStrategy.HYBRID_PROPOSAL_BPTT: (
            selected_availability.faithful_hybrid_proposal_bptt_training
        ),
        TrainingStrategy.SHAC: selected_availability.faithful_shac_training,
    }[item.training_strategy]
    if not strategy_capability:
        strategy = item.training_strategy.value
        raise ValueError(f"training strategy {strategy!r} has no declared faithful implementation")
    if (
        item.architecture is PolicyArchitecture.SHARED_ACTOR
        and not selected_availability.shared_actor
    ):
        raise ValueError("shared-actor implementation is unavailable")
    if (
        item.architecture is PolicyArchitecture.INDEPENDENT_ACTORS
        and not selected_availability.faithful_independent_actors
    ):
        raise ValueError("independent-actor comparator has no declared faithful implementation")
    if item.use_validation_gate and not selected_availability.candidate_validation:
        raise ValueError("candidate-validation implementation is unavailable")
    if item.use_uncertainty_sampling and not selected_availability.uncertainty_rollouts:
        raise ValueError("uncertainty-rollout implementation is unavailable")
    if spec.online_library_updates and not selected_availability.online_library_updates:
        raise ValueError("online library-update implementation is unavailable")
    if spec.online_model_updates and not selected_availability.online_model_updates:
        raise ValueError("online model-update implementation is unavailable")
    if spec.runtime_hard_postcheck_required and not selected_availability.exact_runtime_postcheck:
        raise ValueError("this method requires an exact runtime post-check implementation")


def ablation_tags(config: MethodConfig) -> tuple[str, ...]:
    """Return stable tags for every departure from the registered core configuration."""
    if not isinstance(config, MethodConfig) or not isinstance(config.method_id, MethodID):
        raise TypeError("config must contain a stable MethodID")
    canonical = canonical_method_config(config.method_id).ablation
    tags: list[str] = []
    for item in fields(AblationConfig):
        current = getattr(config.ablation, item.name)
        reference = getattr(canonical, item.name)
        if current == reference:
            continue
        if isinstance(current, Enum):
            rendered = current.value
        elif isinstance(current, bool):
            rendered = "on" if current else "off"
        else:
            rendered = str(current)
        tags.append(f"{item.name}={rendered}")
    return tuple(tags)


def is_claim_eligible_core(config: MethodConfig) -> bool:
    """Return whether configuration exactly matches its registered core method."""
    return not ablation_tags(config)


def method_config_metadata(
    config: MethodConfig, availability: ImplementationAvailability | None = None
) -> dict[str, object]:
    """Validate and serialize one method configuration for a run manifest."""
    selected_availability = availability or ImplementationAvailability()
    validate_method_config(config, selected_availability)
    spec = method_spec(config.method_id)
    tags = ablation_tags(config)
    return {
        "schema_version": 1,
        "method": spec.metadata(),
        "ablation": config.ablation.metadata(),
        "ablation_tags": list(tags),
        "claim_eligible_core_configuration": not tags,
        "implementation_availability": selected_availability.metadata(),
    }


def canonical_method_config_json(
    config: MethodConfig, availability: ImplementationAvailability | None = None
) -> str:
    """Return deterministic compact JSON for hashing and artifact comparison."""
    return json.dumps(
        method_config_metadata(config, availability),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def method_config_digest(
    config: MethodConfig, availability: ImplementationAvailability | None = None
) -> str:
    """Return the SHA-256 digest of canonical method metadata."""
    payload = canonical_method_config_json(config, availability).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def differing_ablation_fields(reference: MethodConfig, variant: MethodConfig) -> tuple[str, ...]:
    """Return Phase-7 axes that differ between same-method configurations."""
    if reference.method_id is not variant.method_id:
        raise ValueError("matched ablations must use the same core method ID")
    return tuple(
        item.name
        for item in fields(AblationConfig)
        if getattr(reference.ablation, item.name) != getattr(variant.ablation, item.name)
    )


def validate_matched_ablation_pair(
    reference: MethodConfig,
    variant: MethodConfig,
    expected_differences: tuple[str, ...],
    availability: ImplementationAvailability | None = None,
) -> None:
    """Require a valid pair to differ on exactly the declared ablation axes."""
    validate_method_config(reference, availability)
    validate_method_config(variant, availability)
    known_fields = {item.name for item in fields(AblationConfig)}
    if not expected_differences or len(set(expected_differences)) != len(expected_differences):
        raise ValueError("expected_differences must be a nonempty tuple without duplicates")
    unknown = set(expected_differences) - known_fields
    if unknown:
        raise ValueError(f"unknown ablation fields: {sorted(unknown)}")
    actual = differing_ablation_fields(reference, variant)
    if set(actual) != set(expected_differences):
        raise ValueError(
            f"unmatched ablation pair: expected {expected_differences}, observed {actual}"
        )


class NominalActionAudit(NamedTuple):
    """Unmodified nominal command with explicit absence of a safety certificate."""

    action: Array
    input_finite: Array
    safety_filtered: Array
    has_certificate: Array


def nominal_only_action(nominal_action: Array) -> NominalActionAudit:
    """Return the nominal command unchanged and mark it as uncertified."""
    nominal_action = jnp.asarray(nominal_action)
    if nominal_action.ndim != 1 or nominal_action.size == 0:
        raise ValueError("nominal_action must be a nonempty 1-D command")
    return NominalActionAudit(
        action=nominal_action,
        input_finite=jnp.all(jnp.isfinite(nominal_action)),
        safety_filtered=jnp.asarray(False),
        has_certificate=jnp.asarray(False),
    )


class LibraryActionAudit(NamedTuple):
    """Selected fallback action and hard-selection evidence; not a complete action filter."""

    action: Array
    action_finite: Array
    has_action_certificate: Array
    uncertified_best_effort: Array
    selection: PolicySelection


def select_library_fallback_action(
    config: MethodConfig,
    first_actions: Array,
    hard_values: Array,
    admissible_scores: Array,
    previous_index: Array,
    selection_config: SelectionConfig,
    availability: ImplementationAvailability | None = None,
) -> LibraryActionAudit:
    """Select a precomputed library fallback through the audited hard policy selector.

    This helper does not project a nominal command and therefore must not be presented as the full
    PCBF/PL-CBF filter.  Nonfinite policy actions are made ineligible before selection.  When no
    valid certificate remains, the selector's documented best-effort index is returned without
    clipping or replacement, and ``uncertified_best_effort`` exposes that condition.
    """
    validate_method_config(config, availability)
    spec = method_spec(config.method_id)
    if not spec.uses_policy_library:
        raise ValueError(f"method {config.method_id.value!r} has no fallback policy library")
    first_actions = jnp.asarray(first_actions)
    hard_values = jnp.asarray(hard_values)
    admissible_scores = jnp.asarray(admissible_scores)
    if first_actions.ndim != 2 or first_actions.shape[0] != config.ablation.policy_count:
        raise ValueError(
            "first_actions must have shape (configured policy_count, action_dimension)"
        )
    if first_actions.shape[1] < 1:
        raise ValueError("action_dimension must be positive")
    if hard_values.shape != (first_actions.shape[0],):
        raise ValueError("hard_values must have shape (policy_count,)")
    if admissible_scores.shape != hard_values.shape:
        raise ValueError("admissible_scores must have shape (policy_count,)")

    finite_actions = jnp.all(jnp.isfinite(first_actions), axis=-1)
    eligible_hard_values = jnp.where(finite_actions, hard_values, jnp.nan)
    eligible_scores = jnp.where(finite_actions, admissible_scores, jnp.nan)
    selection = select_hard_policy(
        eligible_hard_values, eligible_scores, previous_index, selection_config
    )
    selected_action = first_actions[selection.selected_index]
    action_finite = finite_actions[selection.selected_index]
    has_action_certificate = selection.has_certificate & action_finite
    return LibraryActionAudit(
        action=selected_action,
        action_finite=action_finite,
        has_action_certificate=has_action_certificate,
        uncertified_best_effort=~has_action_certificate,
        selection=selection,
    )


__all__ = [
    "METHOD_SPECS",
    "AblationConfig",
    "ImplementationAvailability",
    "LibraryActionAudit",
    "LibraryObjective",
    "MethodConfig",
    "MethodID",
    "MethodSpec",
    "NominalActionAudit",
    "PolicyArchitecture",
    "SourceFidelity",
    "TrainingStrategy",
    "ablation_tags",
    "canonical_method_config",
    "canonical_method_config_json",
    "core_method_ids",
    "differing_ablation_fields",
    "is_claim_eligible_core",
    "method_config_digest",
    "method_config_metadata",
    "method_spec",
    "nominal_only_action",
    "select_library_fallback_action",
    "validate_matched_ablation_pair",
    "validate_method_config",
]
