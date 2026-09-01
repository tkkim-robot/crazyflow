from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.actor import (
    SharedActorConfig,
    SharedActorParams,
    SharedActorSpec,
    initialize_shared_actor,
)
from crazyflow.safety.da_plcbf.adaptation_evidence import (
    _PREFIX,
    ADMISSION_PUBLICATION_ACCOUNTING,
    ADMISSION_RUNTIME_SCOPE,
    BPTT_EXECUTION_CONTRACT,
    AdaptationDecisionProof,
    AdaptationEvidence,
    _candidate_leaf_matches_replay,
    load_adaptation_evidence,
    save_adaptation_evidence,
    validate_adaptation_evidence_binding,
)
from crazyflow.safety.da_plcbf.artifact_smoke import synthetic_trace
from crazyflow.safety.da_plcbf.artifacts import (
    ArtifactEvent,
    _canonical_array_digest,
    _deterministic_npz_bytes,
)
from crazyflow.safety.da_plcbf.snapshots import (
    ActiveSnapshotStore,
    create_active_snapshot,
    create_candidate_snapshot,
)
from crazyflow.safety.da_plcbf.validation import (
    HardValidationEvidence,
    HardValidationThresholds,
    hard_validate_candidate,
)

if TYPE_CHECKING:
    from pathlib import Path

_SHARED_STOCHASTIC_SEED = 0x123456789ABCDEF0


def _bptt_metadata() -> dict[str, Any]:
    return {
        "bptt_execution_contract": BPTT_EXECUTION_CONTRACT,
        "bptt_input_digest": "b" * 64,
        "bptt_cache_key": "online:cpu:0",
        "bptt_execution_backend": "cpu",
        "bptt_execution_device_id": 0,
        "bptt_execution_device": str(jax.devices("cpu")[0]),
    }


def _decision_diagnostics(
    evidence: HardValidationEvidence, thresholds: HardValidationThresholds, report: Any
) -> dict[str, Any]:
    local_retention = np.asarray(report.candidate_local_best) - np.asarray(report.active_local_best)
    return {
        **_bptt_metadata(),
        "execution_device_is_cpu": True,
        "execution_device_is_gpu": False,
        "admission_runtime_seconds": float(np.asarray(evidence.runtime_seconds).item()),
        "admission_runtime_scope": ADMISSION_RUNTIME_SCOPE,
        "admission_publication_included": False,
        "admission_publication_accounting": ADMISSION_PUBLICATION_ACCOUNTING,
        "admission_excluded_compile_warmup_seconds": 0.0,
        "bptt_compile_seconds": 0.0,
        "bptt_warmup_seconds": 0.0,
        "validation_compile_seconds": 0.0,
        "validation_warmup_seconds": 0.0,
        "bptt_compilation_excluded_from_execution_timing": True,
        "validation_compilation_excluded_from_execution_timing": True,
        "report_passed": report.passed,
        "failed_gates": list(report.failed_gate_names),
        "admission_margin": float(
            np.min(local_retention + thresholds.local_non_regression_tolerance)
        ),
        "minimum_coverage_threshold": thresholds.minimum_coverage,
        "minimum_redundancy_threshold": thresholds.minimum_redundancy,
        "minimum_diversity_threshold": thresholds.minimum_diversity,
        "retention_tolerance": thresholds.local_non_regression_tolerance,
    }


def _material(
    *, changed: bool = True, shared_stochastic_seed: int = _SHARED_STOCHASTIC_SEED
) -> tuple[Any, ...]:
    spec = SharedActorSpec(
        base_codes=np.asarray([[-1.0], [1.0]], dtype=np.float32),
        base_desired_velocities=np.zeros((2, 3), dtype=np.float32),
        base_durations=np.ones(2, dtype=np.float32),
        adaptive_mask=np.asarray([False, True]),
    )
    with jax.default_device(jax.devices("cpu")[0]):
        params = initialize_shared_actor(
            jax.random.key(shared_stochastic_seed & 0xFFFFFFFF),
            jax.device_put(spec, jax.devices("cpu")[0]),
            dimension=3,
            n_obstacles=0,
            config=SharedActorConfig(hidden_width=2),
        )
    active = create_active_snapshot(
        params,
        structural_core=spec,
        metadata={"initialization": "deterministic_structured_zero_residual"},
    )
    candidate_params = replace(
        active.params,
        code_offsets=(
            np.asarray([[0.11], [0.21]], dtype=np.float32)
            if changed
            else active.params.code_offsets
        ),
    )
    candidate = create_candidate_snapshot(
        candidate_params, version=1, base_active=active, model_version=0, metadata=_bptt_metadata()
    )
    raw = HardValidationEvidence(
        current_policy_margins=np.asarray([0.2, 0.1]),
        candidate_local_policy_margins=np.asarray([[0.3, 0.2], [0.1, 0.1]]),
        active_local_policy_margins=np.asarray([[0.2, 0.1], [0.0, 0.0]]),
        candidate_descriptors=np.asarray([[0.0], [1.0]]),
        descriptor_scales=np.ones(1),
        feasibility_margins=np.ones(2),
        runtime_seconds=np.asarray([0.001]),
        validation_set_digest="adaptation-proof-test",
    )
    thresholds = HardValidationThresholds(
        minimum_redundancy=1, minimum_diversity=0.5, maximum_runtime_seconds=0.01
    )
    report = hard_validate_candidate(active, candidate, raw, thresholds, current_model_version=0)
    publication = ActiveSnapshotStore(active).admit(candidate, report)
    assert publication.accepted
    return active, candidate, raw, thresholds, report, publication.active


def _bound_evidence(
    *, changed: bool = True, shared_stochastic_seed: int = _SHARED_STOCHASTIC_SEED
) -> tuple[Any, ...]:
    active, candidate, raw, thresholds, report, published = _material(
        changed=changed, shared_stochastic_seed=shared_stochastic_seed
    )
    trace = replace(
        synthetic_trace("a" * 64, steps=12, dt=0.05), snapshot_version=np.ones(12, dtype=np.int32)
    )
    proof = AdaptationDecisionProof(
        phase="cold_start",
        job_id=-1,
        context_step=0,
        boundary_step=0,
        status="admitted",
        decision_model_version=0,
        publication_reason="admitted",
        used_by_executed_control=True,
        proposal_active=active,
        decision_active=active,
        candidate=candidate,
        publication_active=published,
        evidence=raw,
        thresholds=thresholds,
        report=report,
    )
    event = ArtifactEvent(
        sequence=0,
        step=0,
        time_seconds=0.0,
        category="cold_start",
        name="candidate_admitted",
        severity="info",
        snapshot_version=1,
        model_version=0,
        details={
            "candidate_digest": candidate.digest,
            "report_digest": report.digest,
            "reason": "admitted",
            "published_snapshot_version": 1,
            "training_model_version": 0,
            "validation_model_version": 0,
            "decision_model_version": 0,
            **_decision_diagnostics(raw, thresholds, report),
        },
    )
    evidence = AdaptationEvidence(trace.content_sha256, (proof,))
    return evidence, trace, (event,)


def _online_admission(
    active: Any, *, boundary_step: int = 6, job_id: int = 0, model_version: int = 0
) -> tuple[Any, tuple[ArtifactEvent, ArtifactEvent]]:
    candidate_params = replace(
        active.params, code_offsets=np.asarray(active.params.code_offsets) + np.float32(0.015)
    )
    candidate = create_candidate_snapshot(
        candidate_params,
        version=active.version + 1,
        base_active=active,
        model_version=model_version,
        metadata=_bptt_metadata(),
    )
    raw = HardValidationEvidence(
        current_policy_margins=np.asarray([0.2, 0.1]),
        candidate_local_policy_margins=np.asarray([[0.3, 0.2], [0.1, 0.1]]),
        active_local_policy_margins=np.asarray([[0.2, 0.1], [0.0, 0.0]]),
        candidate_descriptors=np.asarray([[0.0], [1.0]]),
        descriptor_scales=np.ones(1),
        feasibility_margins=np.ones(2),
        runtime_seconds=np.asarray([0.001]),
        validation_set_digest=f"adaptation-proof-online-{job_id}",
    )
    thresholds = HardValidationThresholds(
        minimum_redundancy=1, minimum_diversity=0.5, maximum_runtime_seconds=0.01
    )
    report = hard_validate_candidate(
        active, candidate, raw, thresholds, current_model_version=model_version
    )
    store = ActiveSnapshotStore(active)
    if model_version > store.model_version:
        store.advance_model_version(model_version)
    publication = store.admit(candidate, report)
    assert publication.accepted
    proof = AdaptationDecisionProof(
        phase="online",
        job_id=job_id,
        context_step=boundary_step,
        boundary_step=boundary_step,
        status="admitted",
        decision_model_version=model_version,
        publication_reason="admitted",
        used_by_executed_control=True,
        proposal_active=active,
        decision_active=active,
        candidate=candidate,
        publication_active=publication.active,
        evidence=raw,
        thresholds=thresholds,
        report=report,
    )
    submission = ArtifactEvent(
        sequence=1,
        step=boundary_step,
        time_seconds=boundary_step * 0.05,
        category="adaptation",
        name="candidate_submitted",
        severity="info",
        snapshot_version=active.version,
        model_version=model_version,
        details={"job_id": job_id, "reason": "submitted"},
    )
    decision = ArtifactEvent(
        sequence=2,
        step=boundary_step,
        time_seconds=boundary_step * 0.05,
        category="adaptation",
        name="candidate_admitted",
        severity="info",
        snapshot_version=publication.active.version,
        model_version=model_version,
        details={
            "job_id": job_id,
            "candidate_digest": candidate.digest,
            "report_digest": report.digest,
            "reason": "admitted",
            "published_snapshot_version": publication.active.version,
            "publication_boundary": boundary_step,
            "training_model_version": model_version,
            "validation_model_version": model_version,
            "decision_model_version": model_version,
            **_decision_diagnostics(raw, thresholds, report),
        },
    )
    return proof, (submission, decision)


def test_adaptation_evidence_round_trips_and_replays_report_and_publication(tmp_path: Path) -> None:
    evidence, trace, events = _bound_evidence()
    validate_adaptation_evidence_binding(
        evidence, trace, events, shared_stochastic_seed=_SHARED_STOCHASTIC_SEED
    )
    path = tmp_path / "adaptation_evidence.npz"
    save_adaptation_evidence(evidence, path)
    restored = load_adaptation_evidence(path)
    assert restored.content_sha256 == evidence.content_sha256
    for snapshot in (
        restored.decisions[0].proposal_active,
        restored.decisions[0].decision_active,
        restored.decisions[0].candidate,
        restored.decisions[0].publication_active,
    ):
        assert type(snapshot.params) is SharedActorParams
        assert type(snapshot.structural_core) is SharedActorSpec
    validate_adaptation_evidence_binding(
        restored, trace, events, shared_stochastic_seed=_SHARED_STOCHASTIC_SEED
    )


def test_adaptation_evidence_rejects_self_consistent_root_seed_substitution() -> None:
    substituted_seed = _SHARED_STOCHASTIC_SEED + 1
    evidence, trace, events = _bound_evidence(shared_stochastic_seed=substituted_seed)
    validate_adaptation_evidence_binding(
        evidence, trace, events, shared_stochastic_seed=substituted_seed
    )
    with pytest.raises(ValueError, match="lineage root.*scheduled initial"):
        validate_adaptation_evidence_binding(
            evidence, trace, events, shared_stochastic_seed=_SHARED_STOCHASTIC_SEED
        )


def test_adaptation_evidence_rejects_shifted_cold_start_boundary() -> None:
    evidence, trace, events = _bound_evidence()
    shifted_proof = replace(evidence.decisions[0], boundary_step=1)
    shifted_event = replace(events[0], step=1, time_seconds=float(trace.time[1]))
    with pytest.raises(ValueError, match="context and boundary step zero"):
        validate_adaptation_evidence_binding(
            replace(evidence, decisions=(shifted_proof,)),
            trace,
            (shifted_event,),
            shared_stochastic_seed=_SHARED_STOCHASTIC_SEED,
        )


def test_adaptation_evidence_requires_complete_event_model_lineage() -> None:
    evidence, trace, events = _bound_evidence()
    incomplete_details = dict(events[0].details)
    incomplete_details.pop("decision_model_version")
    with pytest.raises(ValueError, match="event model-version lineage"):
        validate_adaptation_evidence_binding(
            evidence,
            trace,
            (replace(events[0], details=incomplete_details),),
            shared_stochastic_seed=_SHARED_STOCHASTIC_SEED,
        )


def test_adaptation_evidence_rejects_self_consistent_intermediate_chain_fork() -> None:
    first_evidence, _, first_events = _bound_evidence()
    first = first_evidence.decisions[0]
    second, second_events = _online_admission(first.publication_active)
    snapshot_versions = np.concatenate(
        (np.full(6, 1, dtype=np.int32), np.full(6, 2, dtype=np.int32))
    )
    trace = replace(
        synthetic_trace("a" * 64, steps=12, dt=0.05), snapshot_version=snapshot_versions
    )
    evidence = AdaptationEvidence(trace.content_sha256, (first, second))
    events = (first_events[0], *second_events)
    validate_adaptation_evidence_binding(
        evidence, trace, events, shared_stochastic_seed=_SHARED_STOCHASTIC_SEED
    )

    fork_params = replace(
        first.publication_active.params,
        input_kernel=np.asarray(first.publication_active.params.input_kernel) + np.float32(0.02),
    )
    fork = create_active_snapshot(
        fork_params,
        version=first.publication_active.version,
        model_version=first.publication_active.model_version,
        structural_core=first.publication_active.structural_core,
        metadata={"tamper": "self-consistent-intermediate-fork"},
    )
    forked_second, forked_events = _online_admission(fork)
    forked = AdaptationEvidence(trace.content_sha256, (first, forked_second))
    with pytest.raises(ValueError, match="lineage forks.*previous publication"):
        validate_adaptation_evidence_binding(
            forked,
            trace,
            (first_events[0], *forked_events),
            shared_stochastic_seed=_SHARED_STOCHASTIC_SEED,
        )


def test_adaptation_evidence_rejects_nonboundary_snapshot_version_substitution() -> None:
    evidence, trace, events = _bound_evidence()
    snapshot_versions = np.array(trace.snapshot_version, copy=True)
    snapshot_versions[5] = 99
    tampered_trace = replace(trace, snapshot_version=snapshot_versions)
    tampered_evidence = replace(evidence, trace_content_sha256=tampered_trace.content_sha256)
    with pytest.raises(ValueError, match="snapshot-version segment"):
        validate_adaptation_evidence_binding(
            tampered_evidence,
            tampered_trace,
            events,
            shared_stochastic_seed=_SHARED_STOCHASTIC_SEED,
        )


def test_adaptation_evidence_rejects_duplicate_online_job_resolution() -> None:
    first_evidence, _, first_events = _bound_evidence()
    first = first_evidence.decisions[0]
    second, second_events = _online_admission(first.publication_active, boundary_step=4, job_id=0)
    third, third_events = _online_admission(second.publication_active, boundary_step=8, job_id=1)
    duplicate_third = replace(third, job_id=0, context_step=4)
    duplicate_decision = replace(third_events[1], details={**third_events[1].details, "job_id": 0})
    snapshot_versions = np.concatenate(
        (
            np.full(4, 1, dtype=np.int32),
            np.full(4, 2, dtype=np.int32),
            np.full(4, 3, dtype=np.int32),
        )
    )
    trace = replace(
        synthetic_trace("a" * 64, steps=12, dt=0.05), snapshot_version=snapshot_versions
    )
    evidence = AdaptationEvidence(trace.content_sha256, (first, second, duplicate_third))
    events = (first_events[0], *second_events, duplicate_decision)
    with pytest.raises(ValueError, match="reuse a scheduler job id"):
        validate_adaptation_evidence_binding(
            evidence, trace, events, shared_stochastic_seed=_SHARED_STOCHASTIC_SEED
        )


def test_adaptation_lineage_allows_model_counter_advance_without_policy_root_fork() -> None:
    first_evidence, _, first_events = _bound_evidence()
    first = first_evidence.decisions[0]
    second, second_events = _online_admission(first.publication_active, model_version=1)
    assert second.decision_model_version == 1
    assert second.decision_active.model_version == 0
    assert second.proposal_active.digest == first.publication_active.digest
    snapshot_versions = np.concatenate(
        (np.full(6, 1, dtype=np.int32), np.full(6, 2, dtype=np.int32))
    )
    model_versions = np.concatenate((np.zeros(6, dtype=np.int32), np.ones(6, dtype=np.int32)))
    trace = replace(
        synthetic_trace("a" * 64, steps=12, dt=0.05),
        snapshot_version=snapshot_versions,
        model_version=model_versions,
    )
    validate_adaptation_evidence_binding(
        AdaptationEvidence(trace.content_sha256, (first, second)),
        trace,
        (first_events[0], *second_events),
        shared_stochastic_seed=_SHARED_STOCHASTIC_SEED,
    )


def test_adaptation_evidence_rejects_async_decision_model_staleness_tamper() -> None:
    first_evidence, _, first_events = _bound_evidence()
    first = first_evidence.decisions[0]
    active = first.publication_active
    candidate = create_candidate_snapshot(
        replace(
            active.params, code_offsets=np.asarray(active.params.code_offsets) + np.float32(0.025)
        ),
        version=active.version + 1,
        base_active=active,
        model_version=0,
        metadata=_bptt_metadata(),
    )
    raw = HardValidationEvidence(
        current_policy_margins=np.asarray([0.2, 0.1]),
        candidate_local_policy_margins=np.asarray([[0.3, 0.2], [0.1, 0.1]]),
        active_local_policy_margins=np.asarray([[0.2, 0.1], [0.0, 0.0]]),
        candidate_descriptors=np.asarray([[0.0], [1.0]]),
        descriptor_scales=np.ones(1),
        feasibility_margins=np.asarray([-1.0, -1.0]),
        runtime_seconds=np.asarray([0.001]),
        validation_set_digest="adaptation-proof-async-stale",
    )
    thresholds = HardValidationThresholds(
        minimum_redundancy=1, minimum_diversity=0.5, maximum_runtime_seconds=0.01
    )
    report = hard_validate_candidate(active, candidate, raw, thresholds, current_model_version=0)
    assert not report.passed
    store = ActiveSnapshotStore(active)
    store.advance_model_version(1)
    publication = store.admit(candidate, report)
    assert not publication.accepted
    assert publication.reason == "stale_model_version"
    proof = AdaptationDecisionProof(
        phase="online",
        job_id=0,
        context_step=4,
        boundary_step=6,
        status="rejected",
        decision_model_version=1,
        publication_reason=publication.reason,
        used_by_executed_control=False,
        proposal_active=active,
        decision_active=active,
        candidate=candidate,
        publication_active=publication.active,
        evidence=raw,
        thresholds=thresholds,
        report=report,
    )
    snapshot_versions = np.ones(12, dtype=np.int32)
    model_versions = np.concatenate((np.zeros(6, dtype=np.int32), np.ones(6, dtype=np.int32)))
    trace = replace(
        synthetic_trace("a" * 64, steps=12, dt=0.05),
        snapshot_version=snapshot_versions,
        model_version=model_versions,
    )
    submission = ArtifactEvent(
        sequence=1,
        step=4,
        time_seconds=float(trace.time[4]),
        category="adaptation",
        name="candidate_submitted",
        severity="info",
        snapshot_version=1,
        model_version=0,
        details={"job_id": 0, "reason": "submitted"},
    )
    decision = ArtifactEvent(
        sequence=2,
        step=6,
        time_seconds=float(trace.time[6]),
        category="adaptation",
        name="candidate_rejected",
        severity="warning",
        snapshot_version=1,
        model_version=1,
        details={
            "job_id": 0,
            "candidate_digest": candidate.digest,
            "report_digest": report.digest,
            "reason": "stale_model_version",
            "published_snapshot_version": None,
            "publication_boundary": 6,
            "training_model_version": 0,
            "validation_model_version": 0,
            "decision_model_version": 1,
            **_decision_diagnostics(raw, thresholds, report),
        },
    )
    valid = AdaptationEvidence(trace.content_sha256, (first, proof))
    validate_adaptation_evidence_binding(
        valid,
        trace,
        (first_events[0], submission, decision),
        shared_stochastic_seed=_SHARED_STOCHASTIC_SEED,
    )

    tampered_proof = replace(
        proof, decision_model_version=0, publication_reason="hard_validation_failed"
    )
    tampered_decision = replace(
        decision,
        model_version=0,
        details={
            **decision.details,
            "reason": "hard_validation_failed",
            "decision_model_version": 0,
        },
    )
    with pytest.raises(ValueError, match="decision model version.*boundary trace"):
        validate_adaptation_evidence_binding(
            replace(valid, decisions=(first, tampered_proof)),
            trace,
            (first_events[0], submission, tampered_decision),
            shared_stochastic_seed=_SHARED_STOCHASTIC_SEED,
        )


def test_adaptation_evidence_is_byte_deterministic(tmp_path: Path) -> None:
    evidence, _, _ = _bound_evidence()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    save_adaptation_evidence(evidence, first)
    save_adaptation_evidence(evidence, second)
    assert first.read_bytes() == second.read_bytes()


def test_adaptation_evidence_rejects_unknown_codec_after_outer_rehash(tmp_path: Path) -> None:
    evidence, _, _ = _bound_evidence()
    path = tmp_path / "adaptation_evidence.npz"
    save_adaptation_evidence(evidence, path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    schema = json.loads(str(arrays["snapshot_schema_json"]))
    schema["params_tree"]["codec"] = "untrusted.module.CustomParams"
    arrays["snapshot_schema_json"] = np.asarray(
        json.dumps(schema, sort_keys=True, separators=(",", ":"))
    )
    semantic = {name: value for name, value in arrays.items() if name != "content_sha256"}
    arrays["content_sha256"] = np.asarray(_canonical_array_digest(_PREFIX, semantic), dtype="<U64")
    path.write_bytes(_deterministic_npz_bytes(arrays))
    with pytest.raises(ValueError, match="tree descriptor"):
        load_adaptation_evidence(path)


def test_adaptation_evidence_rejects_legacy_schema_before_digest_check(tmp_path: Path) -> None:
    evidence, _, _ = _bound_evidence()
    path = tmp_path / "adaptation_evidence.npz"
    save_adaptation_evidence(evidence, path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["schema_version"] = np.asarray(5, dtype=np.uint16)
    path.write_bytes(_deterministic_npz_bytes(arrays))
    with pytest.raises(ValueError, match="unsupported; regenerate"):
        load_adaptation_evidence(path)


def test_optional_candidate_replay_uses_floating_point_tolerance() -> None:
    baseline = np.asarray([0.001, -0.001, 0.0], dtype=np.float32)
    within = baseline + np.asarray([1e-8, -1e-8, 1e-8], dtype=np.float32)
    outside = baseline + np.asarray([1e-3, -1e-3, 1e-3], dtype=np.float32)

    assert _candidate_leaf_matches_replay(baseline.copy(), baseline, backend="cpu")
    assert _candidate_leaf_matches_replay(baseline.copy(), baseline, backend="gpu")
    assert _candidate_leaf_matches_replay(within, baseline, backend="gpu")
    assert not _candidate_leaf_matches_replay(outside, baseline, backend="gpu")
    assert _candidate_leaf_matches_replay(within, baseline, backend="cpu")
    assert not _candidate_leaf_matches_replay(within.astype(np.float64), baseline, backend="cpu")
    assert not _candidate_leaf_matches_replay(within[:, None], baseline, backend="cpu")
    nonfinite = np.asarray([np.inf, -np.inf, np.nan], dtype=np.float32)
    assert not _candidate_leaf_matches_replay(nonfinite, nonfinite, backend="gpu")
    assert not _candidate_leaf_matches_replay(nonfinite, nonfinite, backend="cpu")
    positive_zero = np.asarray([0.0], dtype=np.float32)
    negative_zero = np.asarray([-0.0], dtype=np.float32)
    assert _candidate_leaf_matches_replay(positive_zero, negative_zero, backend="cpu")
    assert not _candidate_leaf_matches_replay(positive_zero, negative_zero, backend="tpu")


def test_adaptation_evidence_rejects_no_parameter_change_and_forged_event() -> None:
    unchanged, trace, events = _bound_evidence(changed=False)
    with pytest.raises(ValueError, match="no BPTT parameter change"):
        validate_adaptation_evidence_binding(
            unchanged, trace, events, shared_stochastic_seed=_SHARED_STOCHASTIC_SEED
        )

    evidence, trace, events = _bound_evidence()
    forged = replace(events[0], details={**events[0].details, "candidate_digest": "f" * 64})
    with pytest.raises(ValueError, match="candidate digest"):
        validate_adaptation_evidence_binding(
            evidence, trace, (forged,), shared_stochastic_seed=_SHARED_STOCHASTIC_SEED
        )


def test_adaptation_evidence_rejects_raw_validation_tampering() -> None:
    evidence, trace, events = _bound_evidence()
    decision = evidence.decisions[0]
    tampered_raw = replace(decision.evidence, current_policy_margins=np.asarray([-9.0, -8.0]))
    tampered = replace(evidence, decisions=(replace(decision, evidence=tampered_raw),))
    with pytest.raises(ValueError, match="does not recompute"):
        validate_adaptation_evidence_binding(
            tampered, trace, events, shared_stochastic_seed=_SHARED_STOCHASTIC_SEED
        )
