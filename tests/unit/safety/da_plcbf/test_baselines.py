from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.baselines import (
    METHOD_SPECS,
    ImplementationAvailability,
    LibraryObjective,
    MethodConfig,
    MethodID,
    PolicyArchitecture,
    SourceFidelity,
    TrainingStrategy,
    ablation_tags,
    canonical_method_config,
    canonical_method_config_json,
    core_method_ids,
    differing_ablation_fields,
    is_claim_eligible_core,
    method_config_digest,
    method_config_metadata,
    method_spec,
    nominal_only_action,
    select_library_fallback_action,
    validate_matched_ablation_pair,
    validate_method_config,
)
from crazyflow.safety.da_plcbf.proposal_ablations import implemented_proposal_ablation_availability
from crazyflow.safety.da_plcbf.selector import SelectionConfig


def _with_ablation(config: MethodConfig, **changes: object) -> MethodConfig:
    return replace(config, ablation=replace(config.ablation, **changes))


def test_registry_has_exact_stable_phase_seven_order_and_immutable_specs() -> None:
    assert core_method_ids() == (
        "nominal_only",
        "analytic_cbf_hocbf",
        "fixed_fallback_pcbf",
        "handcrafted_fixed_library_plcbf",
        "offline_frozen_sdcbf_style",
        "da_plcbf_no_online_model_adaptation",
        "da_plcbf_full",
    )
    assert len(METHOD_SPECS) == 7
    assert len(set(core_method_ids())) == 7
    with pytest.raises(TypeError):
        METHOD_SPECS[MethodID.NOMINAL_ONLY] = method_spec(MethodID.NOMINAL_ONLY)
    with pytest.raises(FrozenInstanceError):
        method_spec(MethodID.NOMINAL_ONLY).display_name = "unsafe rename"


def test_all_canonical_core_configs_validate_and_have_no_ablation_tags() -> None:
    for stable_id in core_method_ids():
        config = canonical_method_config(stable_id)
        validate_method_config(config)
        assert config.method_id.value == stable_id
        assert ablation_tags(config) == ()
        assert is_claim_eligible_core(config)


def test_sdcbf_comparator_is_mechanically_limited_to_style_only_claim() -> None:
    spec = method_spec(MethodID.OFFLINE_FROZEN_SDCBF_STYLE)
    metadata = method_config_metadata(canonical_method_config(spec.method_id))

    assert spec.source_fidelity is SourceFidelity.STYLE_ONLY
    assert "no SDCBF source implementation integrated" in spec.source_label
    assert any(
        "must not be labeled an SDCBF reproduction" in item for item in spec.claim_boundaries
    )
    assert metadata["method"]["source_fidelity"] == "style_only"
    assert not spec.online_library_updates
    assert not spec.online_model_updates


@pytest.mark.parametrize(
    ("method_id", "changes", "message"),
    [
        (MethodID.FIXED_FALLBACK_PCBF, {"policy_count": 2}, "exactly one"),
        (MethodID.HANDCRAFTED_FIXED_LIBRARY_PLCBF, {"policy_count": 1}, "at least two"),
        (
            MethodID.NOMINAL_ONLY,
            {"training_strategy": TrainingStrategy.BPTT},
            "learned-library ablations",
        ),
        (MethodID.OFFLINE_FROZEN_SDCBF_STYLE, {"adaptation_budget": 1}, "adaptation_budget=0"),
        (MethodID.DA_PLCBF_FULL, {"adaptation_budget": 0}, "positive adaptation_budget"),
        (MethodID.DA_PLCBF_FULL, {"use_uncertainty_sampling": False}, "must be zero"),
        (
            MethodID.DA_PLCBF_FULL,
            {"objective": LibraryObjective.GENERIC_DIVERSITY},
            "redundancy term",
        ),
    ],
)
def test_registry_rejects_semantically_mislabeled_combinations(
    method_id: MethodID, changes: dict[str, object], message: str
) -> None:
    config = _with_ablation(canonical_method_config(method_id), **changes)
    with pytest.raises(ValueError, match=message):
        validate_method_config(config)


@pytest.mark.parametrize(
    ("strategy", "capability_name"),
    [
        (TrainingStrategy.SAMPLING_ONLY, "faithful_sampling_only_training"),
        (TrainingStrategy.HYBRID_PROPOSAL_BPTT, "faithful_hybrid_proposal_bptt_training"),
        (TrainingStrategy.SHAC, "faithful_shac_training"),
    ],
)
def test_optional_training_comparators_require_explicit_faithful_availability(
    strategy: TrainingStrategy, capability_name: str
) -> None:
    reference = canonical_method_config(MethodID.DA_PLCBF_FULL)
    variant = _with_ablation(reference, training_strategy=strategy)

    with pytest.raises(ValueError, match="no declared faithful implementation"):
        validate_method_config(variant)
    availability = replace(ImplementationAvailability(), **{capability_name: True})
    validate_method_config(variant, availability)
    validate_matched_ablation_pair(
        reference, variant, ("training_strategy",), availability=availability
    )


@pytest.mark.parametrize(
    "strategy", [TrainingStrategy.SAMPLING_ONLY, TrainingStrategy.HYBRID_PROPOSAL_BPTT]
)
def test_runner_can_opt_into_project_comparators_without_claiming_shac(
    strategy: TrainingStrategy,
) -> None:
    reference = canonical_method_config(MethodID.DA_PLCBF_FULL)
    variant = _with_ablation(reference, training_strategy=strategy)
    availability = implemented_proposal_ablation_availability()

    validate_method_config(variant, availability)
    metadata = method_config_metadata(variant, availability)
    assert metadata["implementation_availability"]["faithful_sampling_only_training"]
    assert metadata["implementation_availability"]["faithful_hybrid_proposal_bptt_training"]
    assert not metadata["implementation_availability"]["faithful_shac_training"]


def test_shac_flag_never_changes_runtime_filter_or_method_identity() -> None:
    reference = canonical_method_config(MethodID.DA_PLCBF_FULL)
    shac = _with_ablation(reference, training_strategy=TrainingStrategy.SHAC)
    availability = replace(ImplementationAvailability(), faithful_shac_training=True)
    metadata = method_config_metadata(shac, availability)

    assert shac.method_id is MethodID.DA_PLCBF_FULL
    assert metadata["method"]["filter_family"] == "adaptive_policy_library_cbf"
    assert metadata["ablation"]["training_strategy"] == "shac"
    assert metadata["ablation_tags"] == ["training_strategy=shac"]
    assert not metadata["claim_eligible_core_configuration"]


def test_independent_policy_ablation_requires_faithful_implementation_and_exact_matching() -> None:
    reference = _with_ablation(
        canonical_method_config(MethodID.DA_PLCBF_FULL),
        policy_count=8,
        training_scenario_count=8,
        uncertainty_sample_count=2,
        adaptation_budget=4,
    )
    variant = _with_ablation(reference, architecture=PolicyArchitecture.INDEPENDENT_ACTORS)
    with pytest.raises(ValueError, match="independent-actor comparator"):
        validate_method_config(variant)

    availability = implemented_proposal_ablation_availability()
    validate_matched_ablation_pair(reference, variant, ("architecture",), availability)
    assert differing_ablation_fields(reference, variant) == ("architecture",)

    confounded = _with_ablation(variant, policy_count=9)
    with pytest.raises(ValueError, match="unmatched ablation pair"):
        validate_matched_ablation_pair(reference, confounded, ("architecture",), availability)


def test_required_component_and_scaling_ablation_axes_are_stably_tagged() -> None:
    reference = canonical_method_config(MethodID.DA_PLCBF_FULL)
    variant = _with_ablation(
        reference,
        use_redundancy=False,
        use_diversity=False,
        use_trust=False,
        use_validation_gate=False,
        use_uncertainty_sampling=False,
        uncertainty_sample_count=0,
        trainable_skill_codes=False,
        trainable_durations=False,
        policy_count=32,
        horizon=25,
        training_scenario_count=16,
        adaptation_budget=4,
    )
    validate_method_config(variant)

    assert ablation_tags(variant) == (
        "use_redundancy=off",
        "use_diversity=off",
        "use_trust=off",
        "use_validation_gate=off",
        "use_uncertainty_sampling=off",
        "trainable_skill_codes=off",
        "trainable_durations=off",
        "policy_count=32",
        "horizon=25",
        "training_scenario_count=16",
        "uncertainty_sample_count=0",
        "adaptation_budget=4",
    )
    assert not is_claim_eligible_core(variant)


def test_metadata_json_and_digest_are_deterministic_and_primitive_only() -> None:
    config = canonical_method_config(MethodID.DA_PLCBF_FULL)
    payload_a = canonical_method_config_json(config)
    payload_b = canonical_method_config_json(config)
    decoded = json.loads(payload_a)

    assert payload_a == payload_b
    assert len(method_config_digest(config)) == 64
    assert method_config_digest(config) == method_config_digest(config)
    assert decoded["schema_version"] == 1
    assert decoded["method"]["method_id"] == "da_plcbf_full"
    assert decoded["ablation"]["policy_count"] == 64
    assert decoded["implementation_availability"]["faithful_shac_training"] is False


def test_nominal_adapter_returns_command_byte_for_byte_and_never_claims_certificate() -> None:
    command = jnp.array([1.25, -2.5, 0.0], dtype=jnp.float32)
    result = jax.jit(nominal_only_action)(command)

    np.testing.assert_array_equal(result.action, command)
    assert bool(result.input_finite)
    assert not bool(result.safety_filtered)
    assert not bool(result.has_certificate)

    invalid = nominal_only_action(jnp.array([jnp.nan]))
    assert not bool(invalid.input_finite)
    assert bool(jnp.isnan(invalid.action[0]))


def test_library_adapter_uses_hard_selector_and_excludes_nonfinite_actions() -> None:
    config = _with_ablation(
        canonical_method_config(MethodID.HANDCRAFTED_FIXED_LIBRARY_PLCBF), policy_count=3
    )
    actions = jnp.array([[1.0, 0.0], [jnp.nan, 4.0], [3.0, 0.0]])
    result = select_library_fallback_action(
        config,
        actions,
        hard_values=jnp.array([0.2, 0.9, 0.4]),
        admissible_scores=jnp.array([0.3, 1.0, 0.8]),
        previous_index=jnp.array(-1),
        selection_config=SelectionConfig(),
    )

    assert int(result.selection.selected_index) == 2
    np.testing.assert_array_equal(result.action, np.array([3.0, 0.0]))
    assert bool(result.action_finite)
    assert bool(result.has_action_certificate)
    assert not bool(result.uncertified_best_effort)


def test_library_adapter_reports_uncertified_best_effort_without_replacement() -> None:
    config = canonical_method_config(MethodID.FIXED_FALLBACK_PCBF)
    result = select_library_fallback_action(
        config,
        first_actions=jnp.array([[7.0, -1.0]]),
        hard_values=jnp.array([-0.2]),
        admissible_scores=jnp.array([0.9]),
        previous_index=jnp.array(0),
        selection_config=SelectionConfig(),
    )

    np.testing.assert_array_equal(result.action, np.array([7.0, -1.0]))
    assert not bool(result.has_action_certificate)
    assert bool(result.uncertified_best_effort)


def test_library_adapter_rejects_nonlibrary_method_and_mismatched_policy_axis() -> None:
    nominal = canonical_method_config(MethodID.NOMINAL_ONLY)
    with pytest.raises(ValueError, match="no fallback policy library"):
        select_library_fallback_action(
            nominal, jnp.ones((1, 2)), jnp.ones(1), jnp.ones(1), jnp.array(-1), SelectionConfig()
        )

    fixed = canonical_method_config(MethodID.FIXED_FALLBACK_PCBF)
    with pytest.raises(ValueError, match="configured policy_count"):
        select_library_fallback_action(
            fixed, jnp.ones((2, 2)), jnp.ones(2), jnp.ones(2), jnp.array(-1), SelectionConfig()
        )


def test_safety_methods_require_exact_postcheck_but_nominal_comparator_does_not() -> None:
    no_postcheck = replace(ImplementationAvailability(), exact_runtime_postcheck=False)
    validate_method_config(canonical_method_config(MethodID.NOMINAL_ONLY), no_postcheck)
    with pytest.raises(ValueError, match="exact runtime post-check"):
        validate_method_config(canonical_method_config(MethodID.ANALYTIC_CBF_HOCBF), no_postcheck)


def test_unknown_method_and_unmatched_pair_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown DA-PLCBF method ID"):
        method_spec("plcbf_magic")
    with pytest.raises(ValueError, match="same core method ID"):
        differing_ablation_fields(
            canonical_method_config(MethodID.DA_PLCBF_FULL),
            canonical_method_config(MethodID.DA_PLCBF_NO_ONLINE_MODEL_ADAPTATION),
        )
    config = canonical_method_config(MethodID.DA_PLCBF_FULL)
    with pytest.raises(ValueError, match="without duplicates"):
        validate_matched_ablation_pair(config, config, ("horizon", "horizon"))
