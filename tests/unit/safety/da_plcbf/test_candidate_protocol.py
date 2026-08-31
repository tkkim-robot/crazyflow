from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf import candidate_protocol as protocol
from crazyflow.safety.da_plcbf.actor import SharedActorParams
from crazyflow.safety.da_plcbf.candidate_protocol import (
    CANDIDATE_ONLY_CLAIM_BOUNDARY,
    build_candidate_protocol_resources,
    build_projected_scalar_bptt,
    candidate_study_profile,
    execute_variant,
    generate_common_fold_inputs,
    inputs_for_variant,
    project_independent_candidate,
    project_shared_candidate,
    validation_report_from_mapping,
    validation_report_mapping,
    variants_for_profile,
)
from crazyflow.safety.da_plcbf.config import LibraryLossConfig
from crazyflow.safety.da_plcbf.independent_actor import (
    IndependentActorParams,
    build_independent_quad_actor_bptt_functions,
)
from crazyflow.safety.da_plcbf.proposal_ablations import ProposalBudget, run_bptt_only_proposal
from crazyflow.safety.da_plcbf.quad_actor_losses import QuadLearningConfig
from crazyflow.safety.da_plcbf.snapshots import create_active_snapshot, create_candidate_snapshot
from crazyflow.safety.da_plcbf.validation import (
    HardValidationEvidence,
    HardValidationThresholds,
    hard_validate_candidate,
)


def test_profiles_encode_the_predeclared_smoke_development_and_confirmatory_contracts() -> None:
    smoke = candidate_study_profile("smoke")
    assert smoke.folds == 1
    assert smoke.proposal_points[0].policy_count == 16
    assert smoke.proposal_points[0].horizon == 2
    assert smoke.proposal_points[0].batch_size == 2
    assert smoke.proposal_points[0].uncertainty_samples == 4
    assert smoke.proposal_points[0].objective_evaluations == 4
    assert smoke.architecture_point.gradient_updates == 4

    development = candidate_study_profile("development")
    assert development.folds == 20
    assert tuple(point.policy_count for point in development.proposal_points) == (16, 32)
    assert all(point.horizon == 25 for point in development.proposal_points)
    assert all(point.batch_size == 16 for point in development.proposal_points)
    assert all(point.uncertainty_samples == 8 for point in development.proposal_points)
    assert all(point.objective_evaluations == 10 for point in development.proposal_points)
    assert all(point.hybrid_gradient_updates == 5 for point in development.proposal_points)
    assert development.architecture_point.uncertainty_samples == 8

    confirmatory = candidate_study_profile("confirmatory")
    assert confirmatory.folds == 100
    assert confirmatory.predeclared_confirmatory_schedule
    assert "not closed-loop safety outcomes" in CANDIDATE_ONLY_CLAIM_BOUNDARY


def test_variant_registry_keeps_shac_absent_and_exact_matched_proposal_ledgers() -> None:
    variants = variants_for_profile(candidate_study_profile("development"))
    assert len(variants) == 21
    assert {item.uncertainty_samples for item in variants} == {8}
    assert not any("shac" in item.variant_id for item in variants)
    proposal = [item for item in variants if item.family == "proposal"]
    for policy_count in (16, 32):
        trio = [item for item in proposal if item.policy_count == policy_count]
        assert {item.proposal_method for item in trio} == {"bptt", "sampling", "hybrid"}
        assert {item.objective_evaluations for item in trio} == {10}
        by_method = {item.proposal_method: item for item in trio}
        assert by_method["bptt"].gradient_updates == 10
        assert by_method["sampling"].gradient_updates == 0
        assert by_method["hybrid"].gradient_updates == 5

    by_id = {item.variant_id: item for item in variants}
    assert {item.uncertainty_samples for item in variants if item.family == "architecture"} == {8}
    reference = by_id["component-reference-plcbf-full-fixed-gated"]
    assert reference.objective_id == "objective_plcbf_aligned"
    assert reference.uncertainty_samples == 8
    assert reference.score_horizon == 25
    assert reference.score_batch_size == 16
    assert not reference.train_skill_codes
    assert not reference.train_durations
    assert by_id["component-objective-generic"].objective_id == ("objective_generic_diversity")
    assert by_id["component-loss-no-redundancy"].loss_ablation == "no_redundancy"
    assert by_id["component-loss-no-diversity"].loss_ablation == "no_diversity"
    assert by_id["component-loss-no-trust"].loss_ablation == "no_trust"
    assert not by_id["component-validation-gate-off"].validation_gate_enabled
    assert by_id["component-train-skill-codes"].train_skill_codes
    assert by_id["component-train-durations"].train_durations

    scale = [item for item in variants if item.family == "scale"]
    assert len(scale) == 5
    assert {
        (item.score_horizon, item.score_batch_size, item.uncertainty_samples) for item in scale
    } == {(50, 16, 8)}
    assert by_id["scale-policy-count-k32"].policy_count == 32
    assert by_id["scale-horizon-h50"].horizon == 50
    assert by_id["scale-scenario-batch-b64"].batch_size == 64
    assert by_id["scale-adaptation-budget-a10"].gradient_updates == 10


def test_common_fold_uses_true_nested_k_and_b_prefixes() -> None:
    profile = candidate_study_profile("development")
    resources = build_candidate_protocol_resources()
    common = generate_common_fold_inputs(profile, fold=3, root_seed=77, resources=resources)
    variants = variants_for_profile(profile)
    small_variant = next(
        item
        for item in variants
        if item.family == "proposal" and item.policy_count == 16 and item.proposal_method == "bptt"
    )
    large_variant = next(
        item
        for item in variants
        if item.family == "proposal" and item.policy_count == 32 and item.proposal_method == "bptt"
    )
    small = inputs_for_variant(common, small_variant, resources)
    large = inputs_for_variant(common, large_variant, resources)

    assert common.spec.base_codes.shape[0] == 64
    assert common.initial_states.shape[0] == 64
    np.testing.assert_array_equal(small.spec.base_codes, large.spec.base_codes[:16])
    np.testing.assert_array_equal(small.initial_states, common.initial_states[:16])
    np.testing.assert_array_equal(
        small.scenarios.obstacle_centers, large.scenarios.obstacle_centers[:16]
    )
    assert np.all(~np.asarray(small.spec.adaptive_mask[:8]))
    assert np.all(np.asarray(small.spec.adaptive_mask[8:]))


def test_shared_and_independent_projection_freeze_structural_rows_codes_and_durations() -> None:
    profile = candidate_study_profile("smoke")
    resources = build_candidate_protocol_resources()
    common = generate_common_fold_inputs(profile, fold=0, root_seed=8, resources=resources)
    mask = common.spec.adaptive_mask

    shared_changed = jax.tree.map(lambda value: value + 1.0, common.shared_params)
    shared = project_shared_candidate(shared_changed, common.shared_params, mask)
    np.testing.assert_array_equal(shared.code_offsets, common.shared_params.code_offsets)
    np.testing.assert_array_equal(shared.duration_offsets, common.shared_params.duration_offsets)
    np.testing.assert_array_equal(
        shared.velocity_offsets[:8], common.shared_params.velocity_offsets[:8]
    )
    assert np.all(np.asarray(shared.velocity_offsets[8:]) == 1.0)

    independent_changed = jax.tree.map(lambda value: value + 1.0, common.independent_params)
    independent = project_independent_candidate(
        independent_changed, common.independent_params, mask
    )
    np.testing.assert_array_equal(independent.code_offsets, common.independent_params.code_offsets)
    np.testing.assert_array_equal(
        independent.duration_offsets, common.independent_params.duration_offsets
    )
    for name in (
        "velocity_offsets",
        "input_kernel",
        "input_bias",
        "hidden_kernel",
        "hidden_bias",
        "output_kernel",
        "output_bias",
    ):
        np.testing.assert_array_equal(
            getattr(independent, name)[:8], getattr(common.independent_params, name)[:8]
        )
        assert np.all(
            np.asarray(getattr(independent, name)[8:])
            == np.asarray(getattr(common.independent_params, name)[8:]) + 1.0
        )


def test_trainable_skill_projection_changes_only_adaptive_code_and_duration_rows() -> None:
    profile = candidate_study_profile("smoke")
    resources = build_candidate_protocol_resources()
    common = generate_common_fold_inputs(profile, fold=6, root_seed=12, resources=resources)
    changed = jax.tree.map(lambda value: value + 1.0, common.shared_params)
    projected = project_shared_candidate(
        changed,
        common.shared_params,
        common.spec.adaptive_mask,
        train_skill_codes=True,
        train_durations=True,
    )

    np.testing.assert_array_equal(projected.code_offsets[:8], common.shared_params.code_offsets[:8])
    np.testing.assert_array_equal(
        projected.duration_offsets[:8], common.shared_params.duration_offsets[:8]
    )
    assert np.all(np.asarray(projected.code_offsets[8:]) == 1.0)
    assert np.all(np.asarray(projected.duration_offsets[8:]) == 1.0)


def test_loss_term_variants_zero_only_the_declared_weight() -> None:
    variants = {
        item.variant_id: item
        for item in variants_for_profile(candidate_study_profile("development"))
    }
    base = LibraryLossConfig()
    redundancy = protocol._loss_config_for_variant(variants["component-loss-no-redundancy"], base)
    diversity = protocol._loss_config_for_variant(variants["component-loss-no-diversity"], base)
    trust = protocol._loss_config_for_variant(variants["component-loss-no-trust"], base)

    assert redundancy.redundancy_weight == 0.0
    assert redundancy.diversity_weight == base.diversity_weight
    assert redundancy.trust_weight == base.trust_weight
    assert diversity.diversity_weight == 0.0
    assert diversity.redundancy_weight == base.redundancy_weight
    assert trust.trust_weight == 0.0
    assert trust.redundancy_weight == base.redundancy_weight


def test_projected_bptt_has_no_optimizer_influence_on_frozen_leaves() -> None:
    profile = candidate_study_profile("smoke")
    resources = build_candidate_protocol_resources()
    common = generate_common_fold_inputs(profile, fold=1, root_seed=9, resources=resources)
    reference = common.shared_params
    project = lambda candidate: project_shared_candidate(  # noqa: E731
        candidate, reference, common.spec.adaptive_mask
    )

    def objective(candidate: SharedActorParams) -> jax.Array:
        leaves = jax.tree.leaves(candidate)
        return sum(jnp.mean((leaf - 0.25) ** 2) for leaf in leaves)

    functions = build_projected_scalar_bptt(objective, project, learning_rate=1e-2, burst_steps=2)
    result = run_bptt_only_proposal(
        reference, functions, (), ProposalBudget(2, gradient_updates=2), project_params=project
    )
    candidate = result.params
    np.testing.assert_array_equal(candidate.code_offsets, reference.code_offsets)
    np.testing.assert_array_equal(candidate.duration_offsets, reference.duration_offsets)
    np.testing.assert_array_equal(candidate.velocity_offsets[:8], reference.velocity_offsets[:8])
    assert not np.array_equal(
        np.asarray(candidate.velocity_offsets[8:]), np.asarray(reference.velocity_offsets[8:])
    )


def test_independent_bptt_rejects_a_library_with_no_adaptive_slots() -> None:
    profile = candidate_study_profile("smoke")
    resources = build_candidate_protocol_resources()
    common = generate_common_fold_inputs(profile, fold=2, root_seed=10, resources=resources)
    spec = common.spec.replace(adaptive_mask=jnp.zeros_like(common.spec.adaptive_mask))
    with pytest.raises(ValueError, match="at least one adaptive"):
        build_independent_quad_actor_bptt_functions(
            spec,
            resources.model,
            resources.actuator,
            resources.actor_config,
            resources.quad_config,
            resources.barrier_config,
            QuadLearningConfig(horizon=2),
            LibraryLossConfig(),
        )


def test_projection_preserves_parameter_dataclass_types() -> None:
    profile = candidate_study_profile("smoke")
    resources = build_candidate_protocol_resources()
    common = generate_common_fold_inputs(profile, fold=4, root_seed=11, resources=resources)
    shared = project_shared_candidate(
        common.shared_params, common.shared_params, common.spec.adaptive_mask
    )
    independent = project_independent_candidate(
        common.independent_params, common.independent_params, common.spec.adaptive_mask
    )
    assert isinstance(shared, SharedActorParams)
    assert isinstance(independent, IndependentActorParams)


def test_validation_gate_ablation_changes_only_the_admission_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = candidate_study_profile("development")
    resources = build_candidate_protocol_resources()
    common = generate_common_fold_inputs(profile, fold=7, root_seed=13, resources=resources)
    by_id = {item.variant_id: item for item in variants_for_profile(profile)}
    gated = by_id["component-reference-plcbf-full-fixed-gated"]
    bypassed = by_id["component-validation-gate-off"]
    params = inputs_for_variant(common, gated, resources).params
    proposal = SimpleNamespace(params=params, input_valid=True)
    score = SimpleNamespace(marker="same-hard-evidence")
    report = SimpleNamespace(passed=False)
    monkeypatch.setattr(protocol, "execute_proposal", lambda *args, **kwargs: proposal)
    monkeypatch.setattr(protocol, "score_candidate", lambda *args, **kwargs: score)
    monkeypatch.setattr(protocol, "validation_report_for_score", lambda *args, **kwargs: report)

    gated_execution = execute_variant(gated, common, resources, seed=1)
    bypassed_execution = execute_variant(bypassed, common, resources, seed=1)

    assert gated_execution.candidate_params_digest == bypassed_execution.candidate_params_digest
    assert gated_execution.initial_params_digest == bypassed_execution.initial_params_digest
    assert gated_execution.input_prefix_digest == bypassed_execution.input_prefix_digest
    assert gated_execution.admission_mode == "hard_validation_gate"
    assert not gated_execution.protocol_admission_accepted
    assert bypassed_execution.admission_mode == "hard_validation_gate_bypassed_for_ablation"
    assert bypassed_execution.protocol_admission_accepted


def test_validation_report_mapping_reconstructs_digest_and_rejects_tampering() -> None:
    active = create_active_snapshot(
        {"weights": np.zeros((2, 1), dtype=np.float32)},
        version=0,
        model_version=0,
        structural_core={"fixed": np.ones((1,), dtype=np.float32)},
    )
    candidate = create_candidate_snapshot(
        {"weights": np.full((2, 1), 0.1, dtype=np.float32)}, version=1, base_active=active
    )
    evidence = HardValidationEvidence(
        current_policy_margins=np.asarray([0.2, 0.3]),
        candidate_local_policy_margins=np.asarray([[0.2], [0.3]]),
        active_local_policy_margins=np.asarray([[0.1], [0.2]]),
        candidate_descriptors=np.asarray([[0.0], [1.0]]),
        descriptor_scales=np.asarray([1.0]),
        feasibility_margins=np.asarray([0.1]),
        runtime_seconds=np.asarray([0.01]),
        validation_set_digest="heldout-prefix-digest",
    )
    report = hard_validate_candidate(
        active,
        candidate,
        evidence,
        HardValidationThresholds(maximum_runtime_seconds=1.0),
        current_model_version=0,
    )
    mapping = validation_report_mapping(report)

    assert validation_report_from_mapping(mapping) == report
    tampered = deepcopy(mapping)
    tampered["gates"][0]["observed"] = "tampered"
    with pytest.raises(ValueError, match="integrity"):
        validation_report_from_mapping(tampered)
    tampered = deepcopy(mapping)
    tampered["passed"] = not mapping["passed"]
    with pytest.raises(ValueError, match="passed flag"):
        validation_report_from_mapping(tampered)
