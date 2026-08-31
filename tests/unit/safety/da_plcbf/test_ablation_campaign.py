from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from crazyflow.safety.da_plcbf import ablation_campaign as campaign
from crazyflow.safety.da_plcbf.candidate_protocol import (
    build_candidate_protocol_resources,
    candidate_study_profile,
    generate_common_fold_inputs,
    variants_for_profile,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_campaign_config_demotes_a_shortened_confirmatory_schedule() -> None:
    full = campaign.CandidateCampaignConfig(profile="confirmatory").resolved_profile()
    short = campaign.CandidateCampaignConfig(profile="confirmatory", folds=3).resolved_profile()

    assert full.folds == 100
    assert full.predeclared_confirmatory_schedule
    assert short.folds == 3
    assert not short.predeclared_confirmatory_schedule
    campaign.CandidateCampaignConfig(profile="confirmatory", folds=3).validate()


def _proposal_mapping(variant: object) -> dict[str, object]:
    resolved = asdict(variant)
    evaluations = int(resolved["objective_evaluations"])
    gradients = int(resolved["gradient_updates"])
    method = str(resolved["proposal_method"])
    if method == "bptt":
        sampling = 0
        final = 0
        candidate_count = 0
        selected_index = -1
        semantics = "last_charged_pre_update_bptt_loss; returned post_update_params_not_evaluated"
    elif method == "sampling":
        sampling = evaluations
        final = 0
        candidate_count = evaluations
        selected_index = 0
        semantics = "best_finite_charged_sampling_loss"
    else:
        sampling = evaluations - gradients - 1
        final = 1
        candidate_count = sampling
        selected_index = -1
        semantics = "best_of_charged_sampling_seed_and_charged_final_bptt_loss"
    accepted = [True] * gradients
    return {
        "selected_loss": 0.0,
        "selected_loss_semantics": semantics,
        "incumbent_loss_first_charged": 0.0,
        "selected_index": selected_index,
        "improved_on_charged_losses": False,
        "input_valid": True,
        "seed": 17,
        "evaluation_ledger": {
            "requested_objective_evaluations": evaluations,
            "actual_objective_evaluations": evaluations,
            "gradient_evaluations": gradients,
            "attempted_gradient_updates": gradients,
            "accepted_gradient_updates": gradients,
            "sampling_evaluations": sampling,
            "final_evaluations": final,
        },
        "raw_timing_seconds": {
            "compile_seconds": 0.1,
            "sampling_seconds": 0.1 if sampling else 0.0,
            "gradient_seconds": 0.1 if gradients else 0.0,
            "final_evaluation_seconds": 0.1 if final else 0.0,
            "total_seconds": 0.4,
        },
        "candidate_losses": [0.0] * candidate_count,
        "gradient_losses": [0.0] * gradients,
        "gradient_update_accepted": accepted,
        "post_update_objective_evaluated": bool(final),
    }


def test_strict_proposal_schema_checks_every_method_specific_budget_component() -> None:
    variants = variants_for_profile(candidate_study_profile("smoke"))
    for variant in variants:
        proposal = _proposal_mapping(variant)
        assert campaign._proposal_record_errors(proposal, asdict(variant)) == ()

        tampered = {**proposal, "evaluation_ledger": dict(proposal["evaluation_ledger"])}
        tampered["evaluation_ledger"]["actual_objective_evaluations"] += 1
        assert any(
            "actual_objective_evaluations" in message
            for message in campaign._proposal_record_errors(tampered, asdict(variant))
        )


def test_candidate_artifact_shape_schema_binds_k_b_r_and_h(tmp_path: Path) -> None:
    variant = variants_for_profile(candidate_study_profile("smoke"))[0]
    path = tmp_path / "candidate.npz"
    arrays = {
        "hard_policy_margins": np.zeros((16, 2), dtype=np.float32),
        "descriptors": np.zeros((16, 9), dtype=np.float32),
        "feasibility_margins": np.zeros((16, 2, 4, 2, 4), dtype=np.float32),
    }
    leaves = (
        np.zeros((16, 4), dtype=np.float32),
        np.zeros((16, 3), dtype=np.float32),
        np.zeros((16,), dtype=np.float32),
        np.zeros((10, 16), dtype=np.float32),
        np.zeros((16,), dtype=np.float32),
        np.zeros((16, 16), dtype=np.float32),
        np.zeros((16,), dtype=np.float32),
        np.zeros((16, 3), dtype=np.float32),
        np.zeros((3,), dtype=np.float32),
    )
    arrays.update({f"candidate_leaf_{index:03d}": leaf for index, leaf in enumerate(leaves)})
    arrays.update(
        {
            "metadata_json": np.asarray("{}"),
            "content_digest": np.asarray("content"),
            "candidate_leaf_digest": np.asarray("leaves"),
            "hard_evidence_digest": np.asarray("evidence"),
        }
    )
    np.savez_compressed(path, **arrays)
    assert campaign._candidate_artifact_shape_and_finiteness(path, asdict(variant)) == (True, True)

    arrays["hard_policy_margins"][0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    assert campaign._candidate_artifact_shape_and_finiteness(path, asdict(variant)) == (True, False)


def test_configuration_marks_uncertainty_training_axis_truthfully_unavailable() -> None:
    profile = candidate_study_profile("development")
    config = campaign.CandidateCampaignConfig(profile="development")
    mapping = campaign._configuration_mapping(config, profile, campaign._repository_root(None))

    assert mapping["schema_version"] == 3
    assert mapping["uncertainty_training_ablation"] == campaign.UNCERTAINTY_TRAINING_BLOCKER
    assert mapping["uncertainty_training_ablation"]["available"] is False
    assert "nominal model" in mapping["uncertainty_training_ablation"]["reason"]


def test_source_digest_binds_runtime_drone_parameter_toml(tmp_path: Path) -> None:
    (tmp_path / "crazyflow" / "drones").mkdir(parents=True)
    (tmp_path / "examples" / "da_plcbf").mkdir(parents=True)
    (tmp_path / "benchmark").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='digest-test'\n", encoding="utf-8")
    (tmp_path / "pixi.lock").write_text("lock-v1\n", encoding="utf-8")
    params = tmp_path / "crazyflow" / "drones" / "params.toml"
    params.write_text("[cf21B_500]\nmass=0.032\n", encoding="utf-8")
    before = campaign.source_tree_digest(tmp_path)

    params.write_text("[cf21B_500]\nmass=0.040\n", encoding="utf-8")
    after = campaign.source_tree_digest(tmp_path)

    assert before != after


def test_source_drift_aborts_before_manifest_and_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    output = tmp_path / "campaign"
    digests = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(campaign, "source_tree_digest", lambda _repository: next(digests))
    monkeypatch.setattr(campaign, "build_candidate_protocol_resources", object)
    monkeypatch.setattr(
        campaign,
        "generate_common_fold_inputs",
        lambda *_args, **_kwargs: SimpleNamespace(content_digest="common-fold"),
    )
    monkeypatch.setattr(campaign, "_save_common_fold", lambda *_args: None)

    def fail_execution(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("retained scheduled failure")

    monkeypatch.setattr(campaign, "execute_variant", fail_execution)

    with pytest.raises(RuntimeError, match="source tree changed"):
        campaign.run_candidate_ablation_campaign(
            campaign.CandidateCampaignConfig(profile="smoke"), output, repository=repository
        )

    assert not (output / "manifest.json").exists()
    assert not (output / "complete.marker").exists()


def test_common_fold_artifact_has_a_semantic_digest_and_rejects_tampering(tmp_path: Path) -> None:
    profile = candidate_study_profile("smoke")
    resources = build_candidate_protocol_resources()
    common = generate_common_fold_inputs(profile, fold=0, root_seed=71, resources=resources)
    campaign._save_common_fold(tmp_path, common)
    path = tmp_path / "inputs" / "fold-0000.npz"

    assert campaign._common_artifact_digest(path) == common.content_digest
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    arrays["initial_states"] = arrays["initial_states"].copy()
    arrays["initial_states"][0, 0] += 0.25
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="semantic digest mismatch"):
        campaign._common_artifact_digest(path)


def test_candidate_aggregate_is_paired_and_never_safety_claim_eligible() -> None:
    profile = candidate_study_profile("smoke")
    variants = variants_for_profile(profile)
    outcomes = []
    for index, variant in enumerate(variants):
        outcomes.append(
            {
                "fold": 0,
                "status": "complete",
                "variant": {"variant_id": variant.variant_id},
                "hard_score": {
                    "minimum_library_hard_margin": 0.1 + 0.01 * index,
                    "scenario_coverage_fraction": 1.0,
                    "safe_policy_fraction": 0.8 + 0.01 * index,
                    "minimum_descriptor_distance": 0.02 + 0.001 * index,
                    "feasible_fraction": 1.0,
                    "adaptive_local_non_regression_fraction": 0.7 + 0.01 * index,
                },
                "proposal": {"raw_timing_seconds": {"total_seconds": 0.1 + index}},
            }
        )
    aggregate = campaign._aggregate_mapping(
        campaign.CandidateCampaignConfig(profile="smoke"), profile, tuple(outcomes)
    )

    assert aggregate["scope"] == "candidate_quality_only"
    assert not aggregate["safety_superiority_eligible"]
    assert aggregate["comparisons"]
    assert all(item["paired_count"] == 1 for item in aggregate["comparisons"])
    assert all(
        item["safety_superiority_interpretation_permitted"] is False
        for item in aggregate["comparisons"]
    )


def test_development_aggregate_contains_only_predeclared_one_factor_comparisons() -> None:
    profile = candidate_study_profile("development")
    variants = variants_for_profile(profile)
    outcomes = []
    for index, variant in enumerate(variants):
        outcomes.append(
            {
                "fold": 0,
                "status": "complete",
                "variant": {"variant_id": variant.variant_id},
                "hard_score": {
                    "minimum_library_hard_margin": 0.1 + index * 1e-3,
                    "scenario_coverage_fraction": 1.0,
                    "safe_policy_fraction": 0.8,
                    "minimum_descriptor_distance": 0.02,
                    "feasible_fraction": 1.0,
                    "adaptive_local_non_regression_fraction": 0.75,
                    "skill_code_changed_fraction": float(variant.train_skill_codes),
                    "duration_changed_fraction": float(variant.train_durations),
                },
                "proposal": {"raw_timing_seconds": {"total_seconds": 0.2 + index * 1e-3}},
                "protocol_admission_accepted": variant.validation_gate_enabled,
            }
        )
    aggregate = campaign._aggregate_mapping(
        campaign.CandidateCampaignConfig(profile="development", folds=1), profile, tuple(outcomes)
    )

    assert len(aggregate["comparisons"]) == 115
    identities = {
        (item["family"], item["comparator_variant_id"], item["reference_variant_id"])
        for item in aggregate["comparisons"]
    }
    assert (
        "objective",
        "component-objective-generic",
        "component-reference-plcbf-full-fixed-gated",
    ) in identities
    assert (
        "validation",
        "component-validation-gate-off",
        "component-reference-plcbf-full-fixed-gated",
    ) in identities
    assert ("scale", "scale-adaptation-budget-a10", "scale-reference-k16-h25-b16-a4") in identities
    assert not any(item["family"] == "uncertainty_training" for item in aggregate["comparisons"])
    assert aggregate["uncertainty_training_ablation"] == campaign.UNCERTAINTY_TRAINING_BLOCKER


def test_nonfinite_hard_endpoints_are_rejected_instead_of_silently_dropped() -> None:
    variant = asdict(variants_for_profile(candidate_study_profile("smoke"))[0])
    hard = {
        "minimum_library_hard_margin": 0.1,
        "per_barrier_hard_margins": {"obstacle_0": 0.1},
        "safe_policy_count_minimum": 1,
        "safe_policy_count_mean": 2.0,
        "safe_policy_fraction": 0.5,
        "scenario_coverage_fraction": 1.0,
        "descriptor_covariance_logdet": -2.0,
        "minimum_descriptor_distance": 0.1,
        "feasible_fraction": 1.0,
        "candidate_feasible": True,
        "rollout_valid_fraction": 1.0,
        "structural_exact_retention_fraction": 1.0,
        "configured_frozen_exact_retention_fraction": 1.0,
        "fixed_code_duration_exact": True,
        "skill_code_changed_fraction": 0.0,
        "duration_changed_fraction": 0.0,
        "adaptive_local_non_regression_fraction": 1.0,
        "adaptive_parameter_changed_fraction": 1.0,
        "score_seconds": 0.1,
    }
    assert campaign._hard_score_record_errors(hard, variant) == ()

    hard["minimum_library_hard_margin"] = "NaN"
    hard["scenario_coverage_fraction"] = "Infinity"
    hard["per_barrier_hard_margins"] = {"obstacle_0": "-Infinity"}
    errors = campaign._hard_score_record_errors(hard, variant)
    assert any("minimum_library_hard_margin" in error for error in errors)
    assert any("scenario_coverage_fraction" in error for error in errors)
    assert any("per-barrier" in error for error in errors)


def _candidate_npz_payload() -> tuple[dict[str, np.ndarray], dict[str, object], str, str, str]:
    metadata: dict[str, object] = {
        "schema_version": campaign.CANDIDATE_ARTIFACT_SCHEMA_VERSION,
        "variant": {"variant_id": "test"},
        "fold": 0,
        "initial_params_digest": "initial",
        "candidate_params_digest": "candidate",
        "input_prefix_digest": "prefix",
    }
    leaves = [np.full((2,), index, dtype=np.float32) for index in range(9)]
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        "hard_policy_margins": np.ones((2, 1), dtype=np.float32),
        "descriptors": np.ones((2, 9), dtype=np.float32),
        "feasibility_margins": np.ones((2, 1, 4, 2, 4), dtype=np.float32),
    }
    arrays.update({f"candidate_leaf_{index:03d}": leaf for index, leaf in enumerate(leaves)})
    content = campaign.numeric_digest(
        "candidate-output-artifact-v1",
        np.frombuffer(arrays["metadata_json"].item().encode("utf-8"), dtype=np.uint8),
        *leaves,
        arrays["hard_policy_margins"],
        arrays["descriptors"],
        arrays["feasibility_margins"],
    )
    leaf_digest = campaign.numeric_digest("candidate-parameter-leaves-v1", *leaves)
    evidence = campaign.numeric_digest(
        "candidate-hard-evidence-v1",
        arrays["hard_policy_margins"],
        arrays["descriptors"],
        arrays["feasibility_margins"],
    )
    arrays["content_digest"] = np.asarray(content)
    arrays["candidate_leaf_digest"] = np.asarray(leaf_digest)
    arrays["hard_evidence_digest"] = np.asarray(evidence)
    return arrays, metadata, content, leaf_digest, evidence


def test_candidate_artifact_publication_reuses_matching_orphan_and_refuses_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.npz"
    arrays, metadata, content, leaf_digest, evidence = _candidate_npz_payload()
    arguments = {
        "expected_metadata": metadata,
        "expected_content_digest": content,
        "expected_leaf_digest": leaf_digest,
        "expected_evidence_digest": evidence,
    }
    campaign._publish_candidate_artifact(path, arrays, **arguments)
    original = path.read_bytes()
    campaign._publish_candidate_artifact(path, arrays, **arguments)
    assert path.read_bytes() == original

    changed, metadata, changed_content, changed_leaf, changed_evidence = _candidate_npz_payload()
    changed["hard_policy_margins"] = np.full((2, 1), 2.0, dtype=np.float32)
    changed_content = campaign.numeric_digest(
        "candidate-output-artifact-v1",
        np.frombuffer(changed["metadata_json"].item().encode("utf-8"), dtype=np.uint8),
        *[changed[f"candidate_leaf_{index:03d}"] for index in range(9)],
        changed["hard_policy_margins"],
        changed["descriptors"],
        changed["feasibility_margins"],
    )
    changed_evidence = campaign.numeric_digest(
        "candidate-hard-evidence-v1",
        changed["hard_policy_margins"],
        changed["descriptors"],
        changed["feasibility_margins"],
    )
    with pytest.raises(ValueError, match="differs from recomputed evidence"):
        campaign._publish_candidate_artifact(
            path,
            changed,
            expected_metadata=metadata,
            expected_content_digest=changed_content,
            expected_leaf_digest=changed_leaf,
            expected_evidence_digest=changed_evidence,
        )
    assert path.read_bytes() == original


def test_outcome_journal_repairs_only_an_interrupted_trailing_record(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.jsonl"
    record = {"fold": 0, "variant": {"variant_id": "kept"}}
    campaign._append_jsonl(path, record)
    with path.open("ab") as stream:
        stream.write(b'{"fold":1,"variant":')
        stream.flush()

    with pytest.raises(ValueError, match="interrupted trailing record"):
        campaign._read_outcomes(path)
    assert campaign._read_outcomes(path, repair_trailing_partial=True) == (record,)
    assert path.read_bytes().endswith(b"\n")


def test_complete_confirmatory_pairing_is_required_but_all_intervals_remain_descriptive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "_bootstrap_interval", lambda values, seed: (None, None))
    profile = candidate_study_profile("confirmatory")
    outcomes = []
    for fold in range(100):
        for variant in variants_for_profile(profile):
            outcomes.append(
                {
                    "fold": fold,
                    "status": "complete",
                    "variant": {"variant_id": variant.variant_id},
                    "hard_score": {
                        "minimum_library_hard_margin": 0.1,
                        "scenario_coverage_fraction": 1.0,
                        "safe_policy_fraction": 0.8,
                        "minimum_descriptor_distance": 0.02,
                        "feasible_fraction": 1.0,
                        "adaptive_local_non_regression_fraction": 0.75,
                        "skill_code_changed_fraction": float(variant.train_skill_codes),
                        "duration_changed_fraction": float(variant.train_durations),
                    },
                    "proposal": {"raw_timing_seconds": {"total_seconds": 0.2}},
                    "protocol_admission_accepted": variant.validation_gate_enabled,
                }
            )
    aggregate = campaign._aggregate_mapping(
        campaign.CandidateCampaignConfig(profile="confirmatory"), profile, tuple(outcomes)
    )
    assert aggregate["complete_valid_confirmatory_pairs"]
    assert not aggregate["candidate_quality_superiority_eligible"]
    assert all(item["paired_count"] == 100 for item in aggregate["comparisons"])
    assert all(
        item["inference_role"] == "exploratory_descriptive"
        and not item["candidate_quality_superiority_interpretation_permitted"]
        for item in aggregate["comparisons"]
    )

    incomplete = campaign._aggregate_mapping(
        campaign.CandidateCampaignConfig(profile="confirmatory"), profile, tuple(outcomes[1:])
    )
    assert not incomplete["complete_valid_confirmatory_pairs"]
    assert any(item["excluded_folds"] for item in incomplete["comparisons"])
