"""Crash-safe artifacts for the separate DA-PLCBF candidate-quality ablation campaign.

The seven-method experiment campaign remains the only source of closed-loop safety comparisons.
This module persists open-loop candidate proposal evidence, exact evaluation ledgers, common hard
scores, and hard validation reports.  Its manifest sets ``safety_superiority_eligible`` to false
unconditionally so downstream reporting cannot silently merge these outcomes into safety claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import jax
import numpy as np

from crazyflow.safety.da_plcbf.actor import SharedActorParams
from crazyflow.safety.da_plcbf.candidate_protocol import (
    CANDIDATE_ONLY_CLAIM_BOUNDARY,
    CandidateStudyProfile,
    CommonFoldInputs,
    VariantExecution,
    build_candidate_protocol_resources,
    candidate_study_profile,
    execute_variant,
    generate_common_fold_inputs,
    hard_score_mapping,
    numeric_digest,
    proposal_result_mapping,
    validation_report_from_mapping,
    validation_report_mapping,
    variants_for_profile,
)
from crazyflow.safety.da_plcbf.independent_actor import IndependentActorParams
from crazyflow.safety.da_plcbf.snapshots import (
    create_active_snapshot,
    create_candidate_snapshot,
    tree_content_digest,
)

CAMPAIGN_SCHEMA_VERSION = 3
OUTCOME_SCHEMA_VERSION = 3
MANIFEST_SCHEMA_VERSION = 3
CANDIDATE_ARTIFACT_SCHEMA_VERSION = 3
CANDIDATE_INFERENCE_BOUNDARY = (
    "All candidate-quality endpoints, paired deltas, and bootstrap intervals are descriptive and "
    "exploratory. They do not permit candidate-quality or safety-superiority inference; the "
    "confirmatory profile name denotes only a predeclared complete 100-fold schedule."
)
_PARAMETER_FIELDS = (
    "code_offsets",
    "velocity_offsets",
    "duration_offsets",
    "input_kernel",
    "input_bias",
    "hidden_kernel",
    "hidden_bias",
    "output_kernel",
    "output_bias",
)
UNCERTAINTY_TRAINING_BLOCKER = {
    "available": False,
    "axis": "uncertainty_aware_training_r0_r4_r8",
    "reason": (
        "The dispatched generic and PL-CBF candidate objectives roll out one nominal model. "
        "R=4/R=8 are held-out hard-scoring shapes only; changing that scorer is not a training "
        "ablation. A nominal/R4/R8 training comparison remains unavailable until a genuine "
        "uncertainty-aware differentiable objective is implemented."
    ),
}


@dataclass(frozen=True, slots=True)
class CandidateCampaignConfig:
    """Resolved profile and deterministic fold schedule."""

    profile: str = "smoke"
    root_seed: int = 260831
    fold_start: int = 0
    folds: int | None = None

    def resolved_profile(self) -> CandidateStudyProfile:
        selected = candidate_study_profile(self.profile)
        if self.folds is not None:
            if self.folds <= 0:
                raise ValueError("folds override must be positive")
            # A shortened confirmatory schedule is explicitly demoted to development evidence.
            selected = replace(
                selected,
                folds=self.folds,
                predeclared_confirmatory_schedule=(
                    selected.predeclared_confirmatory_schedule and self.folds == 100
                ),
            )
        return selected

    def validate(self) -> None:
        self.resolved_profile().validate()
        if self.root_seed < 0 or self.fold_start < 0:
            raise ValueError("root_seed and fold_start must be nonnegative")


@dataclass(frozen=True, slots=True)
class CandidateCampaignRun:
    """Summary returned after executing or resuming a campaign."""

    root: Path
    expected_outcomes: int
    completed_outcomes: int
    failed_outcomes: int
    execution_complete: bool
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class CampaignVerification:
    """Strict artifact verification result."""

    valid: bool
    errors: tuple[str, ...]
    expected_outcomes: int
    retained_outcomes: int
    completed_outcomes: int
    failed_outcomes: int


def run_candidate_ablation_campaign(
    config: CandidateCampaignConfig,
    output: str | os.PathLike[str],
    *,
    repository: str | os.PathLike[str] | None = None,
    resume: bool = True,
) -> CandidateCampaignRun:
    """Execute every scheduled candidate variant and write durable evidence after each one."""
    config.validate()
    profile = config.resolved_profile()
    root = Path(output).resolve()
    repository_path = _repository_root(repository)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    configuration = _configuration_mapping(config, profile, repository_path)
    source_before = str(configuration["source_tree_sha256"])
    if config_path.exists():
        existing = _read_object(config_path)
        if existing != configuration:
            raise ValueError("existing candidate campaign config does not match this run")
        if not resume:
            raise FileExistsError("candidate campaign exists and resume=False")
    else:
        _atomic_json(config_path, configuration)

    records = _read_outcomes(root / "outcomes.jsonl", repair_trailing_partial=True)
    known = {_outcome_key(record): record for record in records}
    variants = variants_for_profile(profile)
    resources = build_candidate_protocol_resources()
    for fold in range(config.fold_start, config.fold_start + profile.folds):
        common = generate_common_fold_inputs(
            profile, fold=fold, root_seed=config.root_seed, resources=resources
        )
        _save_common_fold(root, common)
        for variant in variants:
            key = (fold, variant.variant_id)
            if key in known:
                _verify_resumable_record(root, known[key])
                continue
            # One seed belongs to the paired family/shape, not to a method.  Sampling-only and
            # hybrid consequently consume nested prefixes of identical indexed perturbations.
            family_code = 1 if variant.family == "proposal" else 2
            seed = int(
                np.random.SeedSequence(
                    [
                        config.root_seed,
                        fold,
                        family_code,
                        variant.policy_count,
                        variant.horizon,
                        variant.batch_size,
                        variant.uncertainty_samples,
                    ]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            try:
                execution = execute_variant(variant, common, resources, seed=seed)
                output_record = _successful_outcome(root, execution)
            except Exception as error:  # noqa: BLE001 - failures are scientific outcomes.
                output_record = {
                    "schema_version": OUTCOME_SCHEMA_VERSION,
                    "scope": "candidate_quality_only",
                    "claim_boundary": CANDIDATE_ONLY_CLAIM_BOUNDARY,
                    "inference_boundary": CANDIDATE_INFERENCE_BOUNDARY,
                    "fold": fold,
                    "variant": asdict(variant),
                    "status": "failed",
                    "common_fold_digest": common.content_digest,
                    "input_prefix_digest": None,
                    "initial_params_digest": None,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "candidate_artifact": None,
                    "candidate_artifact_sha256": None,
                    "candidate_content_digest": None,
                    "candidate_params_digest": None,
                    "candidate_leaf_digest": None,
                    "hard_evidence_digest": None,
                    "proposal": None,
                    "hard_score": None,
                    "validation_report": None,
                    "admission_mode": None,
                    "protocol_admission_accepted": None,
                    "candidate_quality_superiority_eligible": False,
                    "safety_superiority_eligible": False,
                }
            _append_jsonl(root / "outcomes.jsonl", output_record)
            known[key] = output_record

    outcomes = tuple(known.values())
    aggregates = _aggregate_mapping(config, profile, outcomes)
    _atomic_json(root / "aggregates.json", aggregates)
    expected = profile.folds * len(variants)
    completed = sum(record["status"] == "complete" for record in outcomes)
    failed = sum(record["status"] == "failed" for record in outcomes)
    execution_complete = len(outcomes) == expected
    if source_tree_digest(repository_path) != source_before:
        raise RuntimeError("source tree changed while candidate-ablation evidence was executing")
    manifest = _manifest_mapping(
        root,
        configuration,
        expected=expected,
        completed=completed,
        failed=failed,
        execution_complete=execution_complete,
        profile=profile,
        aggregates=aggregates,
    )
    _atomic_json(root / "manifest.json", manifest)
    manifest_hash = _file_sha256(root / "manifest.json")
    precommit = verify_candidate_ablation_campaign(
        root,
        repository=repository_path,
        require_current_source=True,
        require_completion_marker=False,
    )
    if not precommit.valid:
        details = "; ".join(precommit.errors[:4])
        raise RuntimeError(f"candidate-ablation precommit verification failed: {details}")
    _atomic_json(
        root / "complete.marker",
        {
            "manifest_sha256": manifest_hash,
            "execution_complete": execution_complete,
            "retained_failures": failed,
            "complete_valid_confirmatory_pairs": aggregates["complete_valid_confirmatory_pairs"],
        },
    )
    return CandidateCampaignRun(
        root, expected, completed, failed, execution_complete, manifest_hash
    )


def verify_candidate_ablation_campaign(
    root: str | os.PathLike[str],
    *,
    repository: str | os.PathLike[str] | None = None,
    require_current_source: bool = True,
    require_completion_marker: bool = True,
) -> CampaignVerification:
    """Verify hashes, ledgers, pairing, frozen leaves, and the candidate-only claim boundary."""
    campaign = Path(root).resolve()
    errors: list[str] = []
    required = [
        campaign / "config.json",
        campaign / "outcomes.jsonl",
        campaign / "aggregates.json",
        campaign / "manifest.json",
    ]
    if require_completion_marker:
        required.append(campaign / "complete.marker")
    for path in required:
        if not path.is_file():
            errors.append(f"missing required artifact: {path.name}")
    if errors:
        return CampaignVerification(False, tuple(errors), 0, 0, 0, 0)

    try:
        configuration = _read_object(campaign / "config.json")
        aggregates = _read_object(campaign / "aggregates.json")
        manifest = _read_object(campaign / "manifest.json")
        marker = _read_object(campaign / "complete.marker") if require_completion_marker else None
        outcomes = _read_outcomes(campaign / "outcomes.jsonl")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return CampaignVerification(False, (f"artifact parse failed: {error}",), 0, 0, 0, 0)

    if configuration.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        errors.append("unsupported campaign schema version")
    if configuration.get("experiment_id") != "da-plcbf-candidate-quality-ablation-v3":
        errors.append("unexpected candidate campaign experiment identifier")
    if configuration.get("scope") != "candidate_quality_only":
        errors.append("campaign scope changed")
    if configuration.get("claim_boundary") != CANDIDATE_ONLY_CLAIM_BOUNDARY:
        errors.append("campaign candidate-only claim boundary changed")
    if configuration.get("inference_boundary") != CANDIDATE_INFERENCE_BOUNDARY:
        errors.append("campaign descriptive-inference boundary changed")
    if configuration.get("safety_superiority_eligible") is not False:
        errors.append("campaign config must prohibit safety superiority claims")
    if configuration.get("candidate_quality_superiority_eligible") is not False:
        errors.append("campaign config must prohibit candidate-quality superiority claims")
    if configuration.get("uncertainty_training_ablation") != UNCERTAINTY_TRAINING_BLOCKER:
        errors.append("campaign uncertainty-training blocker is missing or changed")
    configured_shac = configuration.get("shac")
    if not isinstance(configured_shac, dict) or configured_shac.get("available") is not False:
        errors.append("campaign config must keep SHAC explicitly unavailable")
    profile_name = configuration.get("profile")
    try:
        base_profile = candidate_study_profile(str(profile_name))
        folds = int(configuration["folds"])
        profile = replace(
            base_profile,
            folds=folds,
            predeclared_confirmatory_schedule=(
                base_profile.predeclared_confirmatory_schedule and folds == 100
            ),
        )
        profile.validate()
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid profile configuration: {error}")
        profile = candidate_study_profile("smoke")
        folds = 0
    variants = variants_for_profile(profile)
    expected = folds * len(variants)
    expected_variants = {item.variant_id: asdict(item) for item in variants}
    if configuration.get("predeclared_confirmatory_schedule") is not (
        profile.predeclared_confirmatory_schedule
    ):
        errors.append("configured confirmatory-schedule intent differs from the profile")
    if configuration.get("variants") != list(expected_variants.values()):
        errors.append("configured variants do not match the predeclared profile")
    expected_pairing = {
        "nested_policy_and_batch_prefixes": True,
        "proposal_common_hard_scoring_samples": 8 if profile.name != "smoke" else 4,
        "architecture_common_hard_scoring_samples": (
            profile.architecture_point.uncertainty_samples
        ),
    }
    if configuration.get("pairing") != expected_pairing:
        errors.append("configured held-out pairing contract differs from the profile")
    try:
        raw_fold_start = configuration["fold_start"]
        raw_root_seed = configuration["root_seed"]
        if isinstance(raw_fold_start, bool) or not isinstance(raw_fold_start, int):
            raise TypeError("fold_start must be an integer")
        if isinstance(raw_root_seed, bool) or not isinstance(raw_root_seed, int):
            raise TypeError("root_seed must be an integer")
        fold_start = raw_fold_start
        root_seed = raw_root_seed
        if min(fold_start, root_seed) < 0:
            raise ValueError("fold_start and root_seed must be nonnegative")
    except (KeyError, TypeError, ValueError):
        fold_start = 0
        root_seed = 0
        errors.append("invalid fold_start or root_seed")
    scheduled_folds = set(range(fold_start, fold_start + folds))

    source_digest = configuration.get("source_tree_sha256")
    if not isinstance(source_digest, str) or len(source_digest) != 64:
        errors.append("configured source-tree digest is malformed")
    else:
        try:
            expected_configuration = _configuration_mapping(
                CandidateCampaignConfig(
                    profile=profile.name, root_seed=root_seed, fold_start=fold_start, folds=folds
                ),
                profile,
                _repository_root(repository),
            )
            expected_configuration["source_tree_sha256"] = source_digest
            if _canonical_json(configuration) != _canonical_json(expected_configuration):
                errors.append("campaign configuration differs from the predeclared schema")
        except (TypeError, ValueError) as error:
            errors.append(f"campaign configuration recomputation failed: {error}")

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append("unsupported manifest schema version")
    if manifest.get("safety_superiority_eligible") is not False:
        errors.append("candidate manifest must prohibit safety superiority claims")
    if manifest.get("candidate_quality_superiority_eligible") is not False:
        errors.append("candidate manifest must prohibit candidate-quality superiority claims")
    if manifest.get("claim_boundary") != CANDIDATE_ONLY_CLAIM_BOUNDARY:
        errors.append("candidate-only claim boundary changed")
    if manifest.get("inference_boundary") != CANDIDATE_INFERENCE_BOUNDARY:
        errors.append("manifest descriptive-inference boundary changed")
    shac = manifest.get("shac")
    if not isinstance(shac, dict) or shac.get("available") is not False:
        errors.append("unimplemented SHAC must remain explicitly unavailable")
    if manifest.get("uncertainty_training_ablation") != UNCERTAINTY_TRAINING_BLOCKER:
        errors.append("uncertainty-training blocker is missing or changed")
    if marker is not None and marker.get("manifest_sha256") != _file_sha256(
        campaign / "manifest.json"
    ):
        errors.append("complete marker does not bind the manifest")
    if aggregates.get("schema_version") != 3:
        errors.append("unsupported aggregate schema version")
    if aggregates.get("scope") != "candidate_quality_only":
        errors.append("aggregate scope changed")
    if aggregates.get("claim_boundary") != CANDIDATE_ONLY_CLAIM_BOUNDARY:
        errors.append("aggregate candidate-only claim boundary changed")
    if aggregates.get("inference_boundary") != CANDIDATE_INFERENCE_BOUNDARY:
        errors.append("aggregate descriptive-inference boundary changed")
    if aggregates.get("safety_superiority_eligible") is not False:
        errors.append("aggregates must prohibit safety superiority claims")
    if aggregates.get("candidate_quality_superiority_eligible") is not False:
        errors.append("aggregates must prohibit candidate-quality superiority claims")
    if aggregates.get("uncertainty_training_ablation") != UNCERTAINTY_TRAINING_BLOCKER:
        errors.append("aggregate uncertainty-training blocker is missing or changed")

    if manifest.get("source_tree_sha256") != source_digest:
        errors.append("manifest/config source digests differ")
    if require_current_source:
        current_source = source_tree_digest(_repository_root(repository))
        if current_source != source_digest:
            errors.append("current source tree differs from the campaign source digest")

    files = manifest.get("files")
    common_digests: dict[int, str] = {}
    if not isinstance(files, list):
        errors.append("manifest files must be a list")
    else:
        for item in files:
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
                errors.append("malformed manifest file record")
                continue
            path = campaign / str(item["path"])
            if not path.is_file():
                errors.append(f"manifest file missing: {item['path']}")
            elif _file_sha256(path) != item["sha256"]:
                errors.append(f"manifest hash mismatch: {item['path']}")
            elif path.stat().st_size != item["bytes"]:
                errors.append(f"manifest size mismatch: {item['path']}")
            elif str(item["path"]).startswith("inputs/fold-"):
                try:
                    fold = int(Path(str(item["path"])).stem.removeprefix("fold-"))
                    common_digests[fold] = _common_artifact_digest(
                        path, expected_fold=fold, expected_root_seed=root_seed
                    )
                except (OSError, TypeError, ValueError) as error:
                    errors.append(f"common fold artifact invalid: {item['path']}: {error}")

    seen: set[tuple[int, str]] = set()
    completed = 0
    failed = 0
    input_pairing: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    initial_pairing: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    proposal_trios: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    outcome_fields = {
        "schema_version",
        "scope",
        "claim_boundary",
        "inference_boundary",
        "fold",
        "variant",
        "status",
        "common_fold_digest",
        "input_prefix_digest",
        "initial_params_digest",
        "failure_type",
        "failure_message",
        "candidate_artifact",
        "candidate_artifact_sha256",
        "candidate_content_digest",
        "candidate_params_digest",
        "candidate_leaf_digest",
        "hard_evidence_digest",
        "proposal",
        "hard_score",
        "validation_report",
        "admission_mode",
        "protocol_admission_accepted",
        "candidate_quality_superiority_eligible",
        "safety_superiority_eligible",
    }
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
        if set(record) != outcome_fields:
            errors.append(f"outcome record has the wrong schema: {key}")
        if key[0] not in scheduled_folds or key[1] not in expected_variants:
            errors.append(f"outcome is outside the scheduled profile: {key}")
            continue
        variant = expected_variants[key[1]]
        if record.get("schema_version") != OUTCOME_SCHEMA_VERSION:
            errors.append(f"unsupported outcome schema version: {key}")
        if record.get("variant") != variant:
            errors.append(f"outcome variant fields differ from the schedule: {key}")
        if record.get("scope") != "candidate_quality_only":
            errors.append(f"outcome scope changed: {key}")
        if record.get("claim_boundary") != CANDIDATE_ONLY_CLAIM_BOUNDARY:
            errors.append(f"outcome candidate-only claim boundary changed: {key}")
        if record.get("inference_boundary") != CANDIDATE_INFERENCE_BOUNDARY:
            errors.append(f"outcome descriptive-inference boundary changed: {key}")
        if record.get("safety_superiority_eligible") is not False:
            errors.append(f"outcome permits an unsupported safety claim: {key}")
        if record.get("candidate_quality_superiority_eligible") is not False:
            errors.append(f"outcome permits an unsupported candidate-quality claim: {key}")
        status = record.get("status")
        if record.get("common_fold_digest") != common_digests.get(key[0]):
            errors.append(f"outcome/common-fold digest mismatch: {key}")
        if status == "failed":
            failed += 1
            if (
                not isinstance(record.get("failure_type"), str)
                or not record.get("failure_type")
                or not isinstance(record.get("failure_message"), str)
            ):
                errors.append(f"failed outcome lacks a typed failure record: {key}")
            null_fields = (
                "input_prefix_digest",
                "initial_params_digest",
                "candidate_artifact",
                "candidate_artifact_sha256",
                "candidate_content_digest",
                "candidate_params_digest",
                "candidate_leaf_digest",
                "hard_evidence_digest",
                "proposal",
                "hard_score",
                "validation_report",
                "admission_mode",
                "protocol_admission_accepted",
            )
            if any(record.get(name) is not None for name in null_fields):
                errors.append(f"failed outcome contains partial success evidence: {key}")
            continue
        if status != "complete":
            errors.append(f"unknown outcome status: {key}")
            continue
        completed += 1
        if record.get("failure_type") is not None or record.get("failure_message") is not None:
            errors.append(f"complete outcome contains a failure payload: {key}")
        proposal = record.get("proposal")
        hard = record.get("hard_score")
        if not isinstance(proposal, dict) or not isinstance(hard, dict):
            errors.append(f"complete outcome lacks evidence: {key}")
            continue
        errors.extend(f"{message}: {key}" for message in _proposal_record_errors(proposal, variant))
        errors.extend(f"{message}: {key}" for message in _hard_score_record_errors(hard, variant))
        if hard.get("structural_exact_retention_fraction") != 1.0:
            errors.append(f"structural parameters changed: {key}")
        if hard.get("configured_frozen_exact_retention_fraction") != 1.0:
            errors.append(f"a configured frozen parameter changed: {key}")
        if (
            not variant["train_skill_codes"]
            and not variant["train_durations"]
            and hard.get("fixed_code_duration_exact") is not True
        ):
            errors.append(f"fixed code/duration changed: {key}")
        report = record.get("validation_report")
        expected_mode = (
            "hard_validation_gate"
            if variant["validation_gate_enabled"]
            else "hard_validation_gate_bypassed_for_ablation"
        )
        if record.get("admission_mode") != expected_mode:
            errors.append(f"admission mode differs from the variant contract: {key}")
        report_object = None
        try:
            if not isinstance(report, dict):
                raise ValueError("validation report must be an object")
            report_object = validation_report_from_mapping(report)
            if report_object.validation_set_digest != record.get("input_prefix_digest"):
                errors.append(f"validation report/input-prefix digest mismatch: {key}")
            if variant["validation_gate_enabled"] and (
                record.get("protocol_admission_accepted") != report_object.passed
            ):
                errors.append(f"gated admission result differs from the hard report: {key}")
        except (TypeError, ValueError) as error:
            errors.append(f"validation report integrity failed for {key}: {error}")
        try:
            relative_artifact = Path(str(record["candidate_artifact"]))
            if relative_artifact.is_absolute() or ".." in relative_artifact.parts:
                raise ValueError("candidate artifact path escapes the campaign")
            artifact = (campaign / relative_artifact).resolve()
            if not artifact.is_relative_to(campaign):
                raise ValueError("candidate artifact path escapes the campaign")
            if not artifact.is_file():
                raise ValueError("candidate artifact is missing")
            if _file_sha256(artifact) != record["candidate_artifact_sha256"]:
                errors.append(f"candidate artifact hash mismatch: {key}")
            if _candidate_artifact_digest(artifact) != record["candidate_content_digest"]:
                errors.append(f"candidate semantic digest mismatch: {key}")
            if _candidate_leaf_digest(artifact) != record["candidate_leaf_digest"]:
                errors.append(f"candidate parameter-leaf digest mismatch: {key}")
            if _candidate_evidence_digest(artifact) != record["hard_evidence_digest"]:
                errors.append(f"candidate hard-evidence digest mismatch: {key}")
            expected_metadata = {
                "schema_version": CANDIDATE_ARTIFACT_SCHEMA_VERSION,
                "variant": variant,
                "fold": key[0],
                "initial_params_digest": record["initial_params_digest"],
                "candidate_params_digest": record["candidate_params_digest"],
                "input_prefix_digest": record["input_prefix_digest"],
            }
            if _candidate_artifact_metadata(artifact) != expected_metadata:
                errors.append(f"candidate artifact metadata mismatch: {key}")
            shapes_match, artifact_finite = _candidate_artifact_shape_and_finiteness(
                artifact, variant
            )
            if not shapes_match:
                errors.append(
                    f"candidate parameter/evidence shapes differ from the protocol: {key}"
                )
            if not artifact_finite:
                errors.append(f"candidate parameter/evidence artifact is nonfinite: {key}")
            bindings = _validation_snapshot_bindings(
                campaign / "inputs" / f"fold-{key[0]:04d}.npz", artifact, variant
            )
            if bindings["initial_params_digest"] != record.get("initial_params_digest"):
                errors.append(f"initial parameter digest mismatch: {key}")
            if bindings["candidate_params_digest"] != record.get("candidate_params_digest"):
                errors.append(f"candidate parameter-tree digest mismatch: {key}")
            if report_object is not None:
                if report_object.active_digest != bindings["active_snapshot_digest"]:
                    errors.append(f"validation report/active snapshot digest mismatch: {key}")
                if report_object.candidate_digest != bindings["candidate_snapshot_digest"]:
                    errors.append(f"validation report/candidate snapshot digest mismatch: {key}")
                if (
                    report_object.active_version != 0
                    or report_object.candidate_version != 1
                    or report_object.model_version != 0
                ):
                    errors.append(
                        f"validation report snapshot versions differ from protocol: {key}"
                    )
            if not variant["validation_gate_enabled"]:
                expected_admission = bool(proposal.get("input_valid")) and artifact_finite
                if record.get("protocol_admission_accepted") != expected_admission:
                    errors.append(f"bypassed admission result is not finite/input-valid: {key}")
            input_pair_key = (
                key[0],
                variant["family"],
                variant["policy_count"],
                variant["batch_size"],
                variant["score_batch_size"],
                variant["uncertainty_samples"],
            )
            input_pairing[input_pair_key].add(str(record["input_prefix_digest"]))
            initial_pair_key = (
                key[0],
                variant["family"],
                variant["architecture"],
                variant["policy_count"],
            )
            initial_pairing[initial_pair_key].add(str(record["initial_params_digest"]))
            if variant["family"] == "proposal":
                trio_key = (
                    key[0],
                    variant["policy_count"],
                    variant["horizon"],
                    variant["batch_size"],
                    variant["uncertainty_samples"],
                )
                proposal_trios[trio_key][str(variant["proposal_method"])] = record
        except (KeyError, OSError, TypeError, ValueError) as error:
            errors.append(f"candidate artifact invalid for {key}: {error}")
    for pair_key, digests in input_pairing.items():
        if len(digests) != 1:
            errors.append(f"paired variants did not use one data prefix: {pair_key}")
    for pair_key, digests in initial_pairing.items():
        if len(digests) != 1:
            errors.append(f"paired variants did not use one parameter initialization: {pair_key}")
    for trio_key, trio in proposal_trios.items():
        if set(trio) != {"bptt", "sampling", "hybrid"}:
            continue
        sampling = trio["sampling"]["proposal"]
        hybrid = trio["hybrid"]["proposal"]
        hybrid_losses = hybrid["candidate_losses"]
        if sampling.get("seed") != hybrid.get("seed"):
            errors.append(f"sampling/hybrid proposal seeds differ: {trio_key}")
        if sampling["candidate_losses"][: len(hybrid_losses)] != hybrid_losses:
            errors.append(f"hybrid sampling losses are not a sampling-only prefix: {trio_key}")
    if profile.name != "smoke":
        complete_by_id = {
            (int(record["fold"]), str(record["variant"]["variant_id"])): record
            for record in outcomes
            if record.get("status") == "complete"
        }
        for fold in scheduled_folds:
            gated = complete_by_id.get((fold, "component-reference-plcbf-full-fixed-gated"))
            bypassed = complete_by_id.get((fold, "component-validation-gate-off"))
            if gated is None or bypassed is None:
                continue
            if gated.get("candidate_params_digest") != bypassed.get("candidate_params_digest"):
                errors.append(
                    f"validation on/off variants proposed different candidates: fold {fold}"
                )
            if gated.get("candidate_leaf_digest") != bypassed.get("candidate_leaf_digest"):
                errors.append(f"validation on/off candidate parameter arrays differ: fold {fold}")
            if gated.get("initial_params_digest") != bypassed.get("initial_params_digest"):
                errors.append(
                    f"validation on/off variants used different initial parameters: fold {fold}"
                )
            if gated.get("input_prefix_digest") != bypassed.get("input_prefix_digest"):
                errors.append(f"validation on/off variants used different inputs: fold {fold}")
            if gated.get("hard_evidence_digest") != bypassed.get("hard_evidence_digest"):
                errors.append(
                    f"validation on/off variants used different dense hard evidence: fold {fold}"
                )
            gated_hard = dict(gated["hard_score"])
            bypassed_hard = dict(bypassed["hard_score"])
            gated_hard.pop("score_seconds", None)
            bypassed_hard.pop("score_seconds", None)
            if gated_hard != bypassed_hard:
                errors.append(
                    f"validation on/off variants used different hard evidence: fold {fold}"
                )
    if len(outcomes) != expected:
        errors.append(f"retained {len(outcomes)} outcomes but expected {expected}")
    if manifest.get("expected_outcomes") != expected:
        errors.append("manifest expected outcome count is wrong")
    execution_complete = len(outcomes) == expected
    if manifest.get("completed_outcomes") != completed:
        errors.append("manifest completed outcome count is wrong")
    if manifest.get("failed_outcomes") != failed:
        errors.append("manifest failed outcome count is wrong")
    if manifest.get("execution_complete") != execution_complete:
        errors.append("manifest completion status disagrees with the retained schedule")
    if marker is not None and marker.get("execution_complete") != execution_complete:
        errors.append("completion marker disagrees with retained schedule")
    if marker is not None and marker.get("retained_failures") != failed:
        errors.append("completion marker failure count is wrong")
    recomputed_aggregates: dict[str, Any] | None = None
    try:
        recomputed_aggregates = _aggregate_mapping(
            CandidateCampaignConfig(
                profile=profile.name, root_seed=root_seed, fold_start=fold_start, folds=folds
            ),
            profile,
            outcomes,
        )
        if _canonical_json(aggregates) != _canonical_json(recomputed_aggregates):
            errors.append("aggregates do not match the retained paired outcomes")
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"aggregate recomputation failed: {error}")
    if recomputed_aggregates is not None:
        recomputed_manifest = _manifest_mapping(
            campaign,
            configuration,
            expected=expected,
            completed=completed,
            failed=failed,
            execution_complete=execution_complete,
            profile=profile,
            aggregates=recomputed_aggregates,
        )
        if _canonical_json(manifest) != _canonical_json(recomputed_manifest):
            errors.append("manifest does not match the retained campaign files and outcomes")
        expected_marker = {
            "manifest_sha256": _file_sha256(campaign / "manifest.json"),
            "execution_complete": execution_complete,
            "retained_failures": failed,
            "complete_valid_confirmatory_pairs": recomputed_aggregates[
                "complete_valid_confirmatory_pairs"
            ],
        }
        if marker is not None and _canonical_json(marker) != _canonical_json(expected_marker):
            errors.append("completion marker does not match the retained campaign state")
    return CampaignVerification(
        not errors, tuple(errors), expected, len(outcomes), completed, failed
    )


def _proposal_record_errors(proposal: dict[str, Any], variant: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    ledger = proposal.get("evaluation_ledger")
    ledger_fields = {
        "requested_objective_evaluations",
        "actual_objective_evaluations",
        "gradient_evaluations",
        "attempted_gradient_updates",
        "accepted_gradient_updates",
        "sampling_evaluations",
        "final_evaluations",
    }
    if not isinstance(ledger, dict) or set(ledger) != ledger_fields:
        return ("proposal evaluation ledger has the wrong schema",)
    try:
        values = {name: int(ledger[name]) for name in ledger_fields}
        if any(isinstance(ledger[name], bool) for name in ledger_fields):
            raise TypeError("boolean ledger count")
        evaluations = int(variant["objective_evaluations"])
        gradients = int(variant["gradient_updates"])
        method = str(variant["proposal_method"])
    except (KeyError, TypeError, ValueError):
        return ("proposal evaluation ledger contains a non-integer count",)
    if method == "bptt":
        expected = {
            "requested_objective_evaluations": evaluations,
            "actual_objective_evaluations": evaluations,
            "gradient_evaluations": gradients,
            "attempted_gradient_updates": gradients,
            "sampling_evaluations": 0,
            "final_evaluations": 0,
        }
        candidate_count = 0
        selected_semantics = (
            "last_charged_pre_update_bptt_loss; returned post_update_params_not_evaluated"
        )
    elif method == "sampling":
        expected = {
            "requested_objective_evaluations": evaluations,
            "actual_objective_evaluations": evaluations,
            "gradient_evaluations": 0,
            "attempted_gradient_updates": 0,
            "sampling_evaluations": evaluations,
            "final_evaluations": 0,
        }
        candidate_count = evaluations
        selected_semantics = "best_finite_charged_sampling_loss"
    elif method == "hybrid":
        sampling = evaluations - gradients - 1
        expected = {
            "requested_objective_evaluations": evaluations,
            "actual_objective_evaluations": evaluations,
            "gradient_evaluations": gradients,
            "attempted_gradient_updates": gradients,
            "sampling_evaluations": sampling,
            "final_evaluations": 1,
        }
        candidate_count = sampling
        selected_semantics = "best_of_charged_sampling_seed_and_charged_final_bptt_loss"
    else:
        return ("proposal method is unknown",)
    for name, expected_value in expected.items():
        if values[name] != expected_value:
            errors.append(f"proposal ledger {name} differs from the variant budget")
    accepted = values["accepted_gradient_updates"]
    if accepted < 0 or accepted > values["attempted_gradient_updates"]:
        errors.append("proposal accepted-gradient count is outside its attempted count")
    candidate_losses = proposal.get("candidate_losses")
    gradient_losses = proposal.get("gradient_losses")
    accepted_flags = proposal.get("gradient_update_accepted")
    if not isinstance(candidate_losses, list) or len(candidate_losses) != candidate_count:
        errors.append("proposal candidate-loss ledger has the wrong length")
    if not isinstance(gradient_losses, list) or len(gradient_losses) != gradients:
        errors.append("proposal gradient-loss ledger has the wrong length")
    if (
        not isinstance(accepted_flags, list)
        or len(accepted_flags) != gradients
        or any(not isinstance(item, bool) for item in accepted_flags)
    ):
        errors.append("proposal accepted-gradient flags have the wrong schema")
    elif sum(accepted_flags) != accepted:
        errors.append("proposal accepted-gradient flags disagree with the ledger count")
    if proposal.get("post_update_objective_evaluated") != (expected["final_evaluations"] > 0):
        errors.append("proposal post-update evaluation flag disagrees with the ledger")
    if proposal.get("selected_loss_semantics") != selected_semantics:
        errors.append("proposal selected-loss semantics are mislabeled")
    selected_index = proposal.get("selected_index")
    if isinstance(selected_index, bool) or not isinstance(selected_index, int):
        errors.append("proposal selected index must be an integer")
    elif method == "bptt" and selected_index != -1:
        errors.append("BPTT-only selected index must denote its unevaluated post-update output")
    elif method == "sampling" and not 0 <= selected_index < candidate_count:
        errors.append("sampling selected index is outside its candidate ledger")
    elif method == "hybrid" and not (selected_index == -1 or 0 <= selected_index < candidate_count):
        errors.append("hybrid selected index is outside its charged candidates")
    timing = proposal.get("raw_timing_seconds")
    timing_fields = {
        "compile_seconds",
        "sampling_seconds",
        "gradient_seconds",
        "final_evaluation_seconds",
        "total_seconds",
    }
    if not isinstance(timing, dict) or set(timing) != timing_fields:
        errors.append("proposal timing ledger has the wrong schema")
    else:
        try:
            measured = {name: float(timing[name]) for name in timing_fields}
            if not all(math.isfinite(value) and value >= 0.0 for value in measured.values()):
                raise ValueError("nonfinite or negative timing")
            components = sum(
                measured[name]
                for name in (
                    "compile_seconds",
                    "sampling_seconds",
                    "gradient_seconds",
                    "final_evaluation_seconds",
                )
            )
            if measured["total_seconds"] + 1e-6 < components:
                errors.append("proposal total timing is shorter than its measured components")
        except (TypeError, ValueError):
            errors.append("proposal timing ledger contains a nonfinite or negative value")
    return tuple(errors)


def _hard_score_record_errors(hard: dict[str, Any], variant: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    expected_fields = {
        "minimum_library_hard_margin",
        "per_barrier_hard_margins",
        "safe_policy_count_minimum",
        "safe_policy_count_mean",
        "safe_policy_fraction",
        "scenario_coverage_fraction",
        "descriptor_covariance_logdet",
        "minimum_descriptor_distance",
        "feasible_fraction",
        "candidate_feasible",
        "rollout_valid_fraction",
        "structural_exact_retention_fraction",
        "configured_frozen_exact_retention_fraction",
        "fixed_code_duration_exact",
        "skill_code_changed_fraction",
        "duration_changed_fraction",
        "adaptive_local_non_regression_fraction",
        "adaptive_parameter_changed_fraction",
        "score_seconds",
    }
    if set(hard) != expected_fields:
        errors.append("hard-score record has the wrong schema")
    fraction_fields = (
        "safe_policy_fraction",
        "scenario_coverage_fraction",
        "feasible_fraction",
        "rollout_valid_fraction",
        "structural_exact_retention_fraction",
        "configured_frozen_exact_retention_fraction",
        "skill_code_changed_fraction",
        "duration_changed_fraction",
        "adaptive_local_non_regression_fraction",
        "adaptive_parameter_changed_fraction",
    )
    for name in fraction_fields:
        try:
            value = float(hard[name])
            if not math.isfinite(value):
                errors.append(f"hard-score fraction {name} is nonfinite")
            elif not 0.0 <= value <= 1.0:
                errors.append(f"hard-score fraction {name} is outside [0, 1]")
        except (KeyError, TypeError, ValueError):
            errors.append(f"hard-score fraction {name} is missing or nonnumeric")
    try:
        score_seconds = float(hard["score_seconds"])
        if not math.isfinite(score_seconds) or score_seconds < 0.0:
            raise ValueError("invalid scorer time")
    except (KeyError, TypeError, ValueError):
        errors.append("hard-score timing is missing, nonfinite, or negative")
    try:
        safe_minimum = hard["safe_policy_count_minimum"]
        if isinstance(safe_minimum, bool) or not isinstance(safe_minimum, int):
            raise TypeError("safe minimum is not an integer")
        if not 0 <= safe_minimum <= int(variant["policy_count"]):
            raise ValueError("safe minimum outside policy count")
    except (KeyError, TypeError, ValueError):
        errors.append("hard-score minimum safe-policy count is invalid")
    for name in ("minimum_library_hard_margin", "descriptor_covariance_logdet"):
        try:
            value = float(hard[name])
            if not math.isfinite(value):
                raise ValueError("nonfinite scalar")
        except (KeyError, TypeError, ValueError):
            errors.append(f"hard-score scalar {name} is missing, nonnumeric, or nonfinite")
    for name in ("safe_policy_count_mean", "minimum_descriptor_distance"):
        try:
            value = float(hard[name])
            upper = float(variant["policy_count"]) if name == "safe_policy_count_mean" else math.inf
            if not math.isfinite(value) or not 0.0 <= value <= upper:
                raise ValueError("invalid nonnegative scalar")
        except (KeyError, TypeError, ValueError):
            errors.append(f"hard-score scalar {name} is missing, nonfinite, or outside its range")
    if not isinstance(hard.get("candidate_feasible"), bool):
        errors.append("hard-score candidate_feasible must be boolean")
    if not isinstance(hard.get("fixed_code_duration_exact"), bool):
        errors.append("hard-score fixed-code/duration flag must be boolean")
    barrier_margins = hard.get("per_barrier_hard_margins")
    if not isinstance(barrier_margins, dict) or not barrier_margins:
        errors.append("hard-score per-barrier margin mapping is empty or missing")
    else:
        try:
            for value in barrier_margins.values():
                if not math.isfinite(float(value)):
                    raise ValueError("nonfinite barrier margin")
        except (TypeError, ValueError):
            errors.append("hard-score per-barrier margins contain a nonnumeric or nonfinite value")
    return tuple(errors)


def _parameter_tree_from_leaves(
    leaves: list[np.ndarray], architecture: str
) -> SharedActorParams | IndependentActorParams:
    if len(leaves) != len(_PARAMETER_FIELDS):
        raise ValueError("candidate parameter artifact must contain exactly nine ordered leaves")
    values = dict(zip(_PARAMETER_FIELDS, leaves, strict=True))
    if architecture == "shared":
        return SharedActorParams(**values)
    if architecture == "independent":
        return IndependentActorParams(**values)
    raise ValueError("candidate architecture is unknown")


def _parameter_shape_errors(
    params: SharedActorParams | IndependentActorParams, *, policy_count: int
) -> tuple[str, ...]:
    errors: list[str] = []
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(params)]
    if any(leaf.dtype.kind != "f" for leaf in leaves):
        errors.append("parameter leaves must use real floating dtypes")
    if len({leaf.dtype.str for leaf in leaves}) != 1:
        errors.append("parameter leaves must use one common dtype")
    code = np.asarray(params.code_offsets)
    velocity = np.asarray(params.velocity_offsets)
    duration = np.asarray(params.duration_offsets)
    if code.ndim != 2 or code.shape[0] != policy_count or code.shape[1] <= 0:
        errors.append("code offsets have the wrong shape")
    if velocity.shape != (policy_count, 3):
        errors.append("velocity offsets have the wrong shape")
    if duration.shape != (policy_count,):
        errors.append("duration offsets have the wrong shape")
    input_kernel = np.asarray(params.input_kernel)
    input_bias = np.asarray(params.input_bias)
    hidden_kernel = np.asarray(params.hidden_kernel)
    hidden_bias = np.asarray(params.hidden_bias)
    output_kernel = np.asarray(params.output_kernel)
    output_bias = np.asarray(params.output_bias)
    if isinstance(params, SharedActorParams):
        if input_bias.ndim != 1 or input_bias.shape[0] <= 0:
            errors.append("shared input bias has the wrong shape")
            width = -1
        else:
            width = input_bias.shape[0]
        expected = (
            input_kernel.ndim == 2
            and input_kernel.shape[0] > 0
            and input_kernel.shape[1] == width
            and hidden_kernel.shape == (width, width)
            and hidden_bias.shape == (width,)
            and output_kernel.shape == (width, 3)
            and output_bias.shape == (3,)
        )
    else:
        if input_bias.ndim != 2 or input_bias.shape[0] != policy_count or input_bias.shape[1] <= 0:
            errors.append("independent input bias has the wrong shape")
            width = -1
        else:
            width = input_bias.shape[1]
        expected = (
            input_kernel.ndim == 3
            and input_kernel.shape[0] == policy_count
            and input_kernel.shape[1] > 0
            and input_kernel.shape[2] == width
            and hidden_kernel.shape == (policy_count, width, width)
            and hidden_bias.shape == (policy_count, width)
            and output_kernel.shape == (policy_count, width, 3)
            and output_bias.shape == (policy_count, 3)
        )
    if not expected:
        errors.append("actor network leaves have inconsistent architecture shapes")
    return tuple(errors)


def _ordered_artifact_leaves(arrays: Any, prefix: str) -> list[np.ndarray]:
    token = f"{prefix}_leaf_"
    leaf_like = {name for name in arrays if name.startswith(token)}
    allowed_nonleaf = {"candidate_leaf_digest"} if prefix == "candidate" else set()
    names = sorted(name for name in leaf_like if name.removeprefix(token).isdigit())
    unexpected = leaf_like - set(names) - allowed_nonleaf
    expected = [f"{prefix}_leaf_{index:03d}" for index in range(len(_PARAMETER_FIELDS))]
    if names != expected or unexpected:
        raise ValueError(f"{prefix} parameter leaves are missing, extra, or out of sequence")
    return [np.asarray(arrays[name]) for name in names]


def _candidate_params_from_artifact(
    path: Path, variant: dict[str, Any]
) -> SharedActorParams | IndependentActorParams:
    with np.load(path, allow_pickle=False) as loaded:
        leaves = _ordered_artifact_leaves(loaded, "candidate")
    params = _parameter_tree_from_leaves(leaves, str(variant["architecture"]))
    errors = _parameter_shape_errors(params, policy_count=int(variant["policy_count"]))
    if errors:
        raise ValueError("; ".join(errors))
    return params


def _active_params_from_common_artifact(
    path: Path, variant: dict[str, Any]
) -> SharedActorParams | IndependentActorParams:
    architecture = str(variant["architecture"])
    with np.load(path, allow_pickle=False) as loaded:
        leaves = _ordered_artifact_leaves(loaded, architecture)
    params = _parameter_tree_from_leaves(leaves, architecture)
    policy_count = int(variant["policy_count"])
    if isinstance(params, SharedActorParams):
        params = params.replace(
            code_offsets=params.code_offsets[:policy_count],
            velocity_offsets=params.velocity_offsets[:policy_count],
            duration_offsets=params.duration_offsets[:policy_count],
        )
    else:
        params = jax.tree.map(lambda value: value[:policy_count], params)
    errors = _parameter_shape_errors(params, policy_count=policy_count)
    if errors:
        raise ValueError("; ".join(errors))
    return params


def _validation_snapshot_bindings(
    common_path: Path, candidate_path: Path, variant: dict[str, Any]
) -> dict[str, str]:
    active_params = _active_params_from_common_artifact(common_path, variant)
    candidate_params = _candidate_params_from_artifact(candidate_path, variant)
    active_leaves, active_tree = jax.tree.flatten(active_params)
    candidate_leaves, candidate_tree = jax.tree.flatten(candidate_params)
    active_schema = tuple(
        (np.asarray(leaf).dtype.str, np.asarray(leaf).shape) for leaf in active_leaves
    )
    candidate_schema = tuple(
        (np.asarray(leaf).dtype.str, np.asarray(leaf).shape) for leaf in candidate_leaves
    )
    if active_tree != candidate_tree or active_schema != candidate_schema:
        raise ValueError("candidate parameter tree does not match the active parameter schema")
    with np.load(common_path, allow_pickle=False) as loaded:
        core = {
            "base_codes": np.asarray(loaded["base_codes"])[:8],
            "base_desired_velocities": np.asarray(loaded["base_desired_velocities"])[:8],
            "base_durations": np.asarray(loaded["base_durations"])[:8],
        }
    active = create_active_snapshot(
        active_params,
        version=0,
        model_version=0,
        structural_core=core,
        metadata={"scope": "candidate-quality-ablation"},
    )
    candidate = create_candidate_snapshot(
        candidate_params,
        version=1,
        base_active=active,
        metadata={"variant_id": str(variant["variant_id"])},
    )
    return {
        "initial_params_digest": tree_content_digest(active_params),
        "candidate_params_digest": tree_content_digest(candidate_params),
        "active_snapshot_digest": active.digest,
        "candidate_snapshot_digest": candidate.digest,
    }


def _candidate_artifact_shape_and_finiteness(
    path: Path, variant: dict[str, Any]
) -> tuple[bool, bool]:
    with np.load(path, allow_pickle=False) as loaded:
        expected_files = {
            "metadata_json",
            "hard_policy_margins",
            "descriptors",
            "feasibility_margins",
            "content_digest",
            "candidate_leaf_digest",
            "hard_evidence_digest",
            *{f"candidate_leaf_{index:03d}" for index in range(len(_PARAMETER_FIELDS))},
        }
        file_schema_matches = set(loaded.files) == expected_files
        hard = np.asarray(loaded["hard_policy_margins"])
        descriptors = np.asarray(loaded["descriptors"])
        feasibility = np.asarray(loaded["feasibility_margins"])
        expected_shapes = (
            (int(variant["policy_count"]), int(variant["score_batch_size"])),
            (int(variant["policy_count"]), 9),
            (
                int(variant["policy_count"]),
                int(variant["score_batch_size"]),
                int(variant["uncertainty_samples"]),
                int(variant["score_horizon"]),
                4,
            ),
        )
        shapes_match = (
            file_schema_matches
            and (hard.shape, descriptors.shape, feasibility.shape) == expected_shapes
        )
        evidence_finite = all(
            value.dtype.kind in "fiu" and bool(np.all(np.isfinite(value)))
            for value in (hard, descriptors, feasibility)
        )
        try:
            leaves = _ordered_artifact_leaves(loaded, "candidate")
            params = _parameter_tree_from_leaves(leaves, str(variant["architecture"]))
            shapes_match = shapes_match and not _parameter_shape_errors(
                params, policy_count=int(variant["policy_count"])
            )
            params_finite = all(
                leaf.dtype.kind == "f" and bool(np.all(np.isfinite(leaf))) for leaf in leaves
            )
        except (KeyError, TypeError, ValueError):
            shapes_match = False
            params_finite = False
    return shapes_match, params_finite and evidence_finite


def _configuration_mapping(
    config: CandidateCampaignConfig, profile: CandidateStudyProfile, repository: Path
) -> dict[str, Any]:
    variants = variants_for_profile(profile)
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "experiment_id": "da-plcbf-candidate-quality-ablation-v3",
        "scope": "candidate_quality_only",
        "claim_boundary": CANDIDATE_ONLY_CLAIM_BOUNDARY,
        "inference_boundary": CANDIDATE_INFERENCE_BOUNDARY,
        "profile": profile.name,
        "fold_start": config.fold_start,
        "folds": profile.folds,
        "root_seed": config.root_seed,
        "predeclared_confirmatory_schedule": profile.predeclared_confirmatory_schedule,
        "candidate_quality_superiority_eligible": False,
        "safety_superiority_eligible": False,
        "source_tree_sha256": source_tree_digest(repository),
        "variants": [asdict(item) for item in variants],
        "pairing": {
            "nested_policy_and_batch_prefixes": True,
            "proposal_common_hard_scoring_samples": 8 if profile.name != "smoke" else 4,
            "architecture_common_hard_scoring_samples": (
                profile.architecture_point.uncertainty_samples
            ),
        },
        "shac": {
            "available": False,
            "reason": "No faithful SHAC training-only implementation is dispatched.",
        },
        "uncertainty_training_ablation": UNCERTAINTY_TRAINING_BLOCKER,
    }


def _execution_evidence_errors(execution: VariantExecution) -> tuple[str, ...]:
    """Reject nonfinite or internally inconsistent evidence before it can look complete."""
    variant = asdict(execution.variant)
    errors = list(_proposal_record_errors(proposal_result_mapping(execution.proposal), variant))
    errors.extend(_hard_score_record_errors(hard_score_mapping(execution.hard_score), variant))
    dense = {
        "hard_policy_margins": np.asarray(execution.hard_score.hard_policy_margins),
        "descriptors": np.asarray(execution.hard_score.descriptors),
        "feasibility_margins": np.asarray(execution.hard_score.feasibility_margins),
    }
    expected_shapes = {
        "hard_policy_margins": (execution.variant.policy_count, execution.variant.score_batch_size),
        "descriptors": (execution.variant.policy_count, 9),
        "feasibility_margins": (
            execution.variant.policy_count,
            execution.variant.score_batch_size,
            execution.variant.uncertainty_samples,
            execution.variant.score_horizon,
            4,
        ),
    }
    for name, array in dense.items():
        if array.shape != expected_shapes[name]:
            errors.append(f"dense {name} has the wrong shape")
        if array.dtype.kind not in "fiu" or not bool(np.all(np.isfinite(array))):
            errors.append(f"dense {name} is nonnumeric or nonfinite")
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(execution.candidate_params)]
    if not leaves or any(
        leaf.dtype.kind not in "fiu" or not bool(np.all(np.isfinite(leaf))) for leaf in leaves
    ):
        errors.append("candidate parameter tree is empty, nonnumeric, or nonfinite")
    try:
        validation_report_from_mapping(validation_report_mapping(execution.validation_report))
    except (TypeError, ValueError) as error:
        errors.append(f"validation report is invalid: {error}")
    return tuple(errors)


def _successful_outcome(root: Path, execution: VariantExecution) -> dict[str, Any]:
    evidence_errors = _execution_evidence_errors(execution)
    if evidence_errors:
        raise ValueError("invalid candidate evidence: " + "; ".join(evidence_errors))
    path, file_hash, content_digest, leaf_digest, evidence_digest = _save_candidate_execution(
        root, execution
    )
    return {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "scope": "candidate_quality_only",
        "claim_boundary": CANDIDATE_ONLY_CLAIM_BOUNDARY,
        "inference_boundary": CANDIDATE_INFERENCE_BOUNDARY,
        "fold": execution.fold,
        "variant": asdict(execution.variant),
        "status": "complete",
        "common_fold_digest": execution.common_fold_digest,
        "input_prefix_digest": execution.input_prefix_digest,
        "initial_params_digest": execution.initial_params_digest,
        "failure_type": None,
        "failure_message": None,
        "candidate_artifact": path.relative_to(root).as_posix(),
        "candidate_artifact_sha256": file_hash,
        "candidate_content_digest": content_digest,
        "candidate_params_digest": execution.candidate_params_digest,
        "candidate_leaf_digest": leaf_digest,
        "hard_evidence_digest": evidence_digest,
        "proposal": proposal_result_mapping(execution.proposal),
        "hard_score": hard_score_mapping(execution.hard_score),
        "validation_report": validation_report_mapping(execution.validation_report),
        "admission_mode": execution.admission_mode,
        "protocol_admission_accepted": execution.protocol_admission_accepted,
        "candidate_quality_superiority_eligible": False,
        "safety_superiority_eligible": False,
    }


def _save_common_fold(root: Path, common: CommonFoldInputs) -> None:
    path = root / "inputs" / f"fold-{common.fold:04d}.npz"
    if path.exists():
        if (
            _common_artifact_digest(
                path, expected_fold=common.fold, expected_root_seed=common.root_seed
            )
            != common.content_digest
        ):
            raise ValueError(f"common fold artifact digest mismatch: {common.fold}")
        return
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "content_digest": np.asarray(common.content_digest),
        "fold": np.asarray(common.fold, dtype=np.int64),
        "root_seed": np.asarray(common.root_seed, dtype=np.int64),
        "initial_states": np.asarray(common.initial_states),
        "obstacle_centers": np.asarray(common.scenarios.obstacle_centers),
        "obstacle_radii": np.asarray(common.scenarios.obstacle_radii),
        "obstacle_mask": np.asarray(common.scenarios.obstacle_mask),
        "arena_lower": np.asarray(common.scenarios.arena_lower),
        "arena_upper": np.asarray(common.scenarios.arena_upper),
        "speed_limit": np.asarray(common.scenarios.speed_limit),
        "validation_initial_states": np.asarray(common.validation_initial_states),
        "validation_obstacle_centers": np.asarray(common.validation_scenarios.obstacle_centers),
        "validation_obstacle_radii": np.asarray(common.validation_scenarios.obstacle_radii),
        "validation_obstacle_mask": np.asarray(common.validation_scenarios.obstacle_mask),
        "validation_arena_lower": np.asarray(common.validation_scenarios.arena_lower),
        "validation_arena_upper": np.asarray(common.validation_scenarios.arena_upper),
        "validation_speed_limit": np.asarray(common.validation_scenarios.speed_limit),
        "base_codes": np.asarray(common.spec.base_codes),
        "base_desired_velocities": np.asarray(common.spec.base_desired_velocities),
        "base_durations": np.asarray(common.spec.base_durations),
        "adaptive_mask": np.asarray(common.spec.adaptive_mask),
    }
    for prefix, tree in (
        ("shared", common.shared_params),
        ("independent", common.independent_params),
        ("model_samples", common.model_samples),
    ):
        for index, leaf in enumerate(jax.tree.leaves(tree)):
            arrays[f"{prefix}_leaf_{index:03d}"] = np.asarray(leaf)
    if _common_payload_digest(arrays) != common.content_digest:
        raise ValueError("common fold payload does not match its generated content digest")
    _atomic_npz(path, arrays)


def _common_payload_digest(arrays: Any) -> str:
    def leaves(prefix: str) -> list[np.ndarray]:
        names = sorted(name for name in arrays if name.startswith(f"{prefix}_leaf_"))
        return [np.asarray(arrays[name]) for name in names]

    return numeric_digest(
        "candidate-common-fold-v1",
        np.asarray(arrays["base_codes"]),
        np.asarray(arrays["base_desired_velocities"]),
        np.asarray(arrays["base_durations"]),
        np.asarray(arrays["adaptive_mask"]),
        *leaves("shared"),
        *leaves("independent"),
        np.asarray(arrays["initial_states"]),
        np.asarray(arrays["obstacle_centers"]),
        np.asarray(arrays["obstacle_radii"]),
        np.asarray(arrays["obstacle_mask"]),
        np.asarray(arrays["arena_lower"]),
        np.asarray(arrays["arena_upper"]),
        np.asarray(arrays["speed_limit"]),
        np.asarray(arrays["validation_initial_states"]),
        np.asarray(arrays["validation_obstacle_centers"]),
        np.asarray(arrays["validation_obstacle_radii"]),
        np.asarray(arrays["validation_obstacle_mask"]),
        np.asarray(arrays["validation_arena_lower"]),
        np.asarray(arrays["validation_arena_upper"]),
        np.asarray(arrays["validation_speed_limit"]),
        *leaves("model_samples"),
    )


def _common_artifact_digest(
    path: Path, *, expected_fold: int | None = None, expected_root_seed: int | None = None
) -> str:
    with np.load(path, allow_pickle=False) as loaded:
        fixed = {
            "schema_version",
            "content_digest",
            "fold",
            "root_seed",
            "initial_states",
            "obstacle_centers",
            "obstacle_radii",
            "obstacle_mask",
            "arena_lower",
            "arena_upper",
            "speed_limit",
            "validation_initial_states",
            "validation_obstacle_centers",
            "validation_obstacle_radii",
            "validation_obstacle_mask",
            "validation_arena_lower",
            "validation_arena_upper",
            "validation_speed_limit",
            "base_codes",
            "base_desired_velocities",
            "base_durations",
            "adaptive_mask",
        }
        leaf_names = {
            *(f"shared_leaf_{index:03d}" for index in range(9)),
            *(f"independent_leaf_{index:03d}" for index in range(9)),
            *(f"model_samples_leaf_{index:03d}" for index in range(13)),
        }
        if set(loaded.files) != fixed | leaf_names:
            raise ValueError("common fold artifact has the wrong array schema")
        if int(np.asarray(loaded["schema_version"]).item()) != 1:
            raise ValueError("common fold artifact schema version changed")
        stored_fold = int(np.asarray(loaded["fold"]).item())
        stored_seed = int(np.asarray(loaded["root_seed"]).item())
        if expected_fold is not None and stored_fold != expected_fold:
            raise ValueError("common fold artifact stores the wrong fold")
        if expected_root_seed is not None and stored_seed != expected_root_seed:
            raise ValueError("common fold artifact stores the wrong root seed")
        for name in set(loaded.files) - {"content_digest"}:
            value = np.asarray(loaded[name])
            if value.dtype.kind not in "biuf" or not bool(np.all(np.isfinite(value))):
                raise ValueError(f"common fold artifact array is nonnumeric or nonfinite: {name}")
        digest = _common_payload_digest(loaded)
        stored = str(np.asarray(loaded["content_digest"]).item())
    if digest != stored:
        raise ValueError("common fold artifact semantic digest mismatch")
    return digest


def _save_candidate_execution(
    root: Path, execution: VariantExecution
) -> tuple[Path, str, str, str, str]:
    directory = root / "candidates" / f"fold-{execution.fold:04d}"
    path = directory / f"{execution.variant.variant_id}.npz"
    metadata = {
        "schema_version": CANDIDATE_ARTIFACT_SCHEMA_VERSION,
        "variant": asdict(execution.variant),
        "fold": execution.fold,
        "initial_params_digest": execution.initial_params_digest,
        "candidate_params_digest": execution.candidate_params_digest,
        "input_prefix_digest": execution.input_prefix_digest,
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        "hard_policy_margins": execution.hard_score.hard_policy_margins,
        "descriptors": execution.hard_score.descriptors,
        "feasibility_margins": execution.hard_score.feasibility_margins,
    }
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(execution.candidate_params)]
    for index, leaf in enumerate(leaves):
        arrays[f"candidate_leaf_{index:03d}"] = leaf
    content_digest = numeric_digest(
        "candidate-output-artifact-v1",
        np.frombuffer(arrays["metadata_json"].item().encode("utf-8"), dtype=np.uint8),
        *leaves,
        arrays["hard_policy_margins"],
        arrays["descriptors"],
        arrays["feasibility_margins"],
    )
    leaf_digest = numeric_digest("candidate-parameter-leaves-v1", *leaves)
    evidence_digest = numeric_digest(
        "candidate-hard-evidence-v1",
        arrays["hard_policy_margins"],
        arrays["descriptors"],
        arrays["feasibility_margins"],
    )
    arrays["content_digest"] = np.asarray(content_digest)
    arrays["candidate_leaf_digest"] = np.asarray(leaf_digest)
    arrays["hard_evidence_digest"] = np.asarray(evidence_digest)
    _publish_candidate_artifact(
        path,
        arrays,
        expected_metadata=metadata,
        expected_content_digest=content_digest,
        expected_leaf_digest=leaf_digest,
        expected_evidence_digest=evidence_digest,
    )
    return path, _file_sha256(path), content_digest, leaf_digest, evidence_digest


def _publish_candidate_artifact(
    path: Path,
    arrays: dict[str, np.ndarray],
    *,
    expected_metadata: dict[str, Any],
    expected_content_digest: str,
    expected_leaf_digest: str,
    expected_evidence_digest: str,
) -> None:
    """Publish once, reusing a matching crash orphan and refusing any differing occupant."""

    def verify_existing() -> None:
        try:
            actual = (
                _candidate_artifact_metadata(path),
                _candidate_artifact_digest(path),
                _candidate_leaf_digest(path),
                _candidate_evidence_digest(path),
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError(f"existing candidate artifact is invalid: {path}") from error
        expected = (
            expected_metadata,
            expected_content_digest,
            expected_leaf_digest,
            expected_evidence_digest,
        )
        if actual != expected:
            raise ValueError(
                f"existing candidate artifact differs from recomputed evidence: {path}"
            )

    if path.is_symlink():
        raise ValueError(f"candidate artifact path must not be a symlink: {path}")
    if path.exists():
        verify_existing()
        return
    if not _write_once_npz(path, arrays):
        verify_existing()


def _candidate_artifact_digest(path: Path) -> str:
    with np.load(path, allow_pickle=False) as loaded:
        leaves = _ordered_artifact_leaves(loaded, "candidate")
        metadata = str(np.asarray(loaded["metadata_json"]).item()).encode("utf-8")
        digest = numeric_digest(
            "candidate-output-artifact-v1",
            np.frombuffer(metadata, dtype=np.uint8),
            *leaves,
            np.asarray(loaded["hard_policy_margins"]),
            np.asarray(loaded["descriptors"]),
            np.asarray(loaded["feasibility_margins"]),
        )
        stored = str(np.asarray(loaded["content_digest"]).item())
    if digest != stored:
        raise ValueError("candidate artifact semantic digest mismatch")
    return digest


def _candidate_leaf_digest(path: Path) -> str:
    with np.load(path, allow_pickle=False) as loaded:
        leaves = _ordered_artifact_leaves(loaded, "candidate")
        digest = numeric_digest("candidate-parameter-leaves-v1", *leaves)
        stored = str(np.asarray(loaded["candidate_leaf_digest"]).item())
    if digest != stored:
        raise ValueError("candidate parameter-leaf digest mismatch")
    return digest


def _candidate_evidence_digest(path: Path) -> str:
    with np.load(path, allow_pickle=False) as loaded:
        digest = numeric_digest(
            "candidate-hard-evidence-v1",
            np.asarray(loaded["hard_policy_margins"]),
            np.asarray(loaded["descriptors"]),
            np.asarray(loaded["feasibility_margins"]),
        )
        stored = str(np.asarray(loaded["hard_evidence_digest"]).item())
    if digest != stored:
        raise ValueError("candidate hard-evidence digest mismatch")
    return digest


def _candidate_artifact_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as loaded:
        return _json_loads_object(str(np.asarray(loaded["metadata_json"]).item()))


def _aggregate_mapping(
    config: CandidateCampaignConfig,
    profile: CandidateStudyProfile,
    outcomes: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    indexed = {(int(item["fold"]), str(item["variant"]["variant_id"])): item for item in outcomes}
    comparisons: list[dict[str, Any]] = []
    base_metrics = (
        "minimum_library_hard_margin",
        "scenario_coverage_fraction",
        "safe_policy_fraction",
        "minimum_descriptor_distance",
        "feasible_fraction",
        "adaptive_local_non_regression_fraction",
        "proposal_total_seconds",
    )
    variants = variants_for_profile(profile)
    by_id = {variant.variant_id: variant for variant in variants}
    pairs: list[tuple[str, str, str, str, tuple[str, ...]]] = []
    proposal_groups: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(dict)
    for variant in variants:
        if variant.family == "proposal":
            shape = (
                variant.policy_count,
                variant.horizon,
                variant.batch_size,
                variant.score_horizon,
                variant.score_batch_size,
                variant.uncertainty_samples,
            )
            proposal_groups[shape][variant.proposal_method] = variant
    for members in proposal_groups.values():
        reference = members["bptt"]
        for label in ("sampling", "hybrid"):
            comparator = members[label]
            pairs.append(
                ("proposal", label, comparator.variant_id, reference.variant_id, base_metrics)
            )

    architecture = {
        variant.architecture: variant for variant in variants if variant.family == "architecture"
    }
    pairs.append(
        (
            "architecture",
            "independent",
            architecture["independent"].variant_id,
            architecture["shared"].variant_id,
            base_metrics,
        )
    )
    if profile.name != "smoke":
        reference = "component-reference-plcbf-full-fixed-gated"
        for family, label, comparator, extra_metrics in (
            ("objective", "generic", "component-objective-generic", ()),
            ("loss_term", "no_redundancy", "component-loss-no-redundancy", ()),
            ("loss_term", "no_diversity", "component-loss-no-diversity", ()),
            ("loss_term", "no_trust", "component-loss-no-trust", ()),
            (
                "validation",
                "gate_off",
                "component-validation-gate-off",
                ("protocol_admission_accepted",),
            ),
            (
                "skill_parameter",
                "train_codes",
                "component-train-skill-codes",
                ("skill_code_changed_fraction",),
            ),
            (
                "skill_parameter",
                "train_durations",
                "component-train-durations",
                ("duration_changed_fraction",),
            ),
        ):
            pairs.append((family, label, comparator, reference, (*base_metrics, *extra_metrics)))
        scale_reference = "scale-reference-k16-h25-b16-a4"
        for label, comparator in (
            ("policy_count_k32", "scale-policy-count-k32"),
            ("horizon_h50", "scale-horizon-h50"),
            ("scenario_batch_b64", "scale-scenario-batch-b64"),
            ("adaptation_budget_a10", "scale-adaptation-budget-a10"),
        ):
            pairs.append(("scale", label, comparator, scale_reference, base_metrics))

    folds = range(config.fold_start, config.fold_start + profile.folds)
    for family, label, comparator_id, reference_id, metrics in pairs:
        comparator = by_id[comparator_id]
        reference = by_id[reference_id]
        if (
            comparator.score_horizon,
            comparator.score_batch_size,
            comparator.uncertainty_samples,
        ) != (reference.score_horizon, reference.score_batch_size, reference.uncertainty_samples):
            raise ValueError("paired candidate variants must share one held-out scoring shape")
        for metric in metrics:
            deltas: list[float] = []
            retained_folds: list[int] = []
            excluded_folds: list[dict[str, Any]] = []
            for fold in folds:
                left = indexed.get((fold, comparator_id))
                right = indexed.get((fold, reference_id))
                if left is None:
                    excluded_folds.append({"fold": fold, "reason": "comparator_missing"})
                    continue
                if right is None:
                    excluded_folds.append({"fold": fold, "reason": "reference_missing"})
                    continue
                if left.get("status") != "complete":
                    excluded_folds.append({"fold": fold, "reason": "comparator_failed"})
                    continue
                if right.get("status") != "complete":
                    excluded_folds.append({"fold": fold, "reason": "reference_failed"})
                    continue
                try:
                    delta = _aggregate_metric(left, metric) - _aggregate_metric(right, metric)
                except (KeyError, TypeError, ValueError):
                    excluded_folds.append({"fold": fold, "reason": "metric_missing_or_invalid"})
                    continue
                if math.isfinite(delta):
                    deltas.append(delta)
                    retained_folds.append(fold)
                else:
                    excluded_folds.append({"fold": fold, "reason": "metric_nonfinite"})
            lower, upper = _bootstrap_interval(
                np.asarray(deltas),
                seed=_stable_seed(config.root_seed, family, label, reference_id, metric),
            )
            comparisons.append(
                {
                    "scope": "candidate_quality_only",
                    "family": family,
                    "comparator": label,
                    "comparator_variant_id": comparator_id,
                    "reference_variant_id": reference_id,
                    "comparator_training_shape": _variant_shape_mapping(comparator),
                    "reference_training_shape": _variant_shape_mapping(reference),
                    "common_heldout_shape": {
                        "horizon": comparator.score_horizon,
                        "batch_size": comparator.score_batch_size,
                        "uncertainty_samples": comparator.uncertainty_samples,
                    },
                    "metric": metric,
                    "paired_count": len(deltas),
                    "folds": retained_folds,
                    "excluded_folds": excluded_folds,
                    "raw_paired_deltas_comparator_minus_reference": deltas,
                    "mean_delta": float(np.mean(deltas)) if deltas else None,
                    "descriptive_bootstrap_95_interval": [lower, upper],
                    "inference_role": "exploratory_descriptive",
                    "candidate_quality_superiority_interpretation_permitted": False,
                    "safety_superiority_interpretation_permitted": False,
                    "timing_interpretation": (
                        "raw_order_dependent_total_including_compile"
                        if metric == "proposal_total_seconds"
                        else None
                    ),
                }
            )
    complete_pairs = bool(comparisons) and all(
        item["paired_count"] == profile.folds and not item["excluded_folds"] for item in comparisons
    )
    complete_valid_confirmatory_pairs = bool(
        profile.predeclared_confirmatory_schedule and complete_pairs
    )
    return {
        "schema_version": 3,
        "scope": "candidate_quality_only",
        "claim_boundary": CANDIDATE_ONLY_CLAIM_BOUNDARY,
        "inference_boundary": CANDIDATE_INFERENCE_BOUNDARY,
        "profile": profile.name,
        "predeclared_confirmatory_schedule": profile.predeclared_confirmatory_schedule,
        "complete_valid_confirmatory_pairs": complete_valid_confirmatory_pairs,
        "candidate_quality_superiority_eligible": False,
        "safety_superiority_eligible": False,
        "multiple_comparison_policy": {
            "confirmatory_tests": 0,
            "correction": "not_applicable_all_intervals_descriptive",
            "statement": (
                "No interval or sign is a candidate-quality superiority test; all displayed "
                "intervals are unadjusted descriptive summaries."
            ),
        },
        "uncertainty_training_ablation": UNCERTAINTY_TRAINING_BLOCKER,
        "comparisons": comparisons,
    }


def _aggregate_metric(record: dict[str, Any], metric: str) -> float:
    if metric == "proposal_total_seconds":
        return float(record["proposal"]["raw_timing_seconds"]["total_seconds"])
    if metric == "protocol_admission_accepted":
        return float(bool(record["protocol_admission_accepted"]))
    return float(record["hard_score"][metric])


def _variant_shape_mapping(variant: Any) -> dict[str, int]:
    return {
        "policy_count": variant.policy_count,
        "horizon": variant.horizon,
        "batch_size": variant.batch_size,
        "objective_evaluations": variant.objective_evaluations,
    }


def _bootstrap_interval(values: np.ndarray, *, seed: int) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(10_000, values.size))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _stable_seed(root_seed: int, *parts: str) -> int:
    digest = hashlib.sha256(str(root_seed).encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little")


def _manifest_mapping(
    root: Path,
    configuration: dict[str, Any],
    *,
    expected: int,
    completed: int,
    failed: int,
    execution_complete: bool,
    profile: CandidateStudyProfile,
    aggregates: dict[str, Any],
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
        "scope": "candidate_quality_only",
        "claim_boundary": CANDIDATE_ONLY_CLAIM_BOUNDARY,
        "inference_boundary": CANDIDATE_INFERENCE_BOUNDARY,
        "profile": profile.name,
        "source_tree_sha256": configuration["source_tree_sha256"],
        "expected_outcomes": expected,
        "completed_outcomes": completed,
        "failed_outcomes": failed,
        "execution_complete": execution_complete,
        "predeclared_confirmatory_schedule": profile.predeclared_confirmatory_schedule,
        "complete_valid_confirmatory_pairs": aggregates["complete_valid_confirmatory_pairs"],
        "candidate_quality_superiority_eligible": False,
        "safety_superiority_eligible": False,
        "uncertainty_training_ablation": UNCERTAINTY_TRAINING_BLOCKER,
        "shac": configuration["shac"],
        "files": files,
    }


def source_tree_digest(repository: Path) -> str:
    """Hash implementation/config sources while excluding generated campaign artifacts."""
    digest = hashlib.sha256(b"crazyflow.da_plcbf.candidate-source-tree.v2\0")
    roots = (
        repository / "crazyflow",
        repository / "examples" / "da_plcbf",
        repository / "benchmark",
    )
    paths = sorted(
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".toml", ".xml"}
    )
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


def _outcome_key(record: dict[str, Any]) -> tuple[int, str]:
    variant = record["variant"]
    if not isinstance(variant, dict):
        raise TypeError("outcome variant must be an object")
    return int(record["fold"]), str(variant["variant_id"])


def _read_outcomes(
    path: Path, *, repair_trailing_partial: bool = False
) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        return ()
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        try:
            _json_loads_object(lines[-1])
        except (TypeError, ValueError, json.JSONDecodeError):
            if not repair_trailing_partial:
                raise ValueError("outcome journal has an interrupted trailing record") from None
            payload = b"".join(lines[:-1])
            _atomic_bytes(path, payload)
            lines = lines[:-1]
        else:
            if not repair_trailing_partial:
                raise ValueError("outcome journal final record is not newline-terminated")
            payload += b"\n"
            _atomic_bytes(path, payload)
            lines[-1] += b"\n"
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.decode("utf-8").removesuffix("\n")
        if not line.strip():
            raise ValueError(f"blank outcome line {line_number}")
        record = _json_loads_object(line)
        key = _outcome_key(record)
        if key in seen:
            raise ValueError(f"duplicate outcome key {key}")
        seen.add(key)
        records.append(record)
    return tuple(records)


def _verify_resumable_record(root: Path, record: dict[str, Any]) -> None:
    if record.get("status") == "failed":
        return
    if record.get("status") != "complete":
        raise ValueError("existing outcome has an unknown status")
    artifact = root / str(record["candidate_artifact"])
    if _file_sha256(artifact) != record["candidate_artifact_sha256"]:
        raise ValueError("existing candidate artifact hash mismatch")
    if _candidate_artifact_digest(artifact) != record["candidate_content_digest"]:
        raise ValueError("existing candidate artifact semantic digest mismatch")
    if _candidate_leaf_digest(artifact) != record["candidate_leaf_digest"]:
        raise ValueError("existing candidate parameter-leaf digest mismatch")
    if _candidate_evidence_digest(artifact) != record["hard_evidence_digest"]:
        raise ValueError("existing candidate hard-evidence digest mismatch")
    variant = record.get("variant")
    report_mapping = record.get("validation_report")
    if not isinstance(variant, dict) or not isinstance(report_mapping, dict):
        raise ValueError("existing candidate record lacks variant/report evidence")
    shapes_match, artifact_finite = _candidate_artifact_shape_and_finiteness(artifact, variant)
    if not shapes_match or not artifact_finite:
        raise ValueError("existing candidate parameter/evidence artifact is invalid")
    bindings = _validation_snapshot_bindings(
        root / "inputs" / f"fold-{int(record['fold']):04d}.npz", artifact, variant
    )
    if bindings["initial_params_digest"] != record.get("initial_params_digest"):
        raise ValueError("existing initial parameter digest mismatch")
    if bindings["candidate_params_digest"] != record.get("candidate_params_digest"):
        raise ValueError("existing candidate parameter-tree digest mismatch")
    report = validation_report_from_mapping(report_mapping)
    if report.validation_set_digest != record.get("input_prefix_digest"):
        raise ValueError("existing validation/input-prefix digest mismatch")
    if report.active_digest != bindings["active_snapshot_digest"]:
        raise ValueError("existing validation/active snapshot digest mismatch")
    if report.candidate_digest != bindings["candidate_snapshot_digest"]:
        raise ValueError("existing validation/candidate snapshot digest mismatch")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value == math.inf:
            return "Infinity"
        if value == -math.inf:
            return "-Infinity"
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


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


def _append_jsonl(path: Path, value: Any) -> None:
    payload = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
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


def _write_once_npz(path: Path, arrays: dict[str, np.ndarray]) -> bool:
    """Atomically link a complete NPZ into place without ever replacing an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        _fsync_directory(path.parent)
        return True
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
    "CAMPAIGN_SCHEMA_VERSION",
    "CANDIDATE_ARTIFACT_SCHEMA_VERSION",
    "CANDIDATE_INFERENCE_BOUNDARY",
    "CampaignVerification",
    "CandidateCampaignConfig",
    "CandidateCampaignRun",
    "MANIFEST_SCHEMA_VERSION",
    "OUTCOME_SCHEMA_VERSION",
    "UNCERTAINTY_TRAINING_BLOCKER",
    "run_candidate_ablation_campaign",
    "source_tree_digest",
    "verify_candidate_ablation_campaign",
]
