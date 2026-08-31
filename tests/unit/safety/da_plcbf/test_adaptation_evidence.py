from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from crazyflow.safety.da_plcbf.adaptation_evidence import (
    AdaptationDecisionProof,
    AdaptationEvidence,
    load_adaptation_evidence,
    save_adaptation_evidence,
    validate_adaptation_evidence_binding,
)
from crazyflow.safety.da_plcbf.artifact_smoke import synthetic_trace
from crazyflow.safety.da_plcbf.artifacts import ArtifactEvent
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


def _material(*, changed: bool = True) -> tuple[Any, ...]:
    active = create_active_snapshot(
        {"adaptive": np.asarray([[0.1], [0.2]], dtype=np.float32)},
        structural_core={"codes": np.asarray([[-1.0], [1.0]], dtype=np.float32)},
    )
    params = (
        np.asarray([[0.11], [0.21]], dtype=np.float32) if changed else active.params["adaptive"]
    )
    candidate = create_candidate_snapshot(
        {"adaptive": params}, version=1, base_active=active, model_version=0
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


def _bound_evidence(*, changed: bool = True) -> tuple[Any, ...]:
    active, candidate, raw, thresholds, report, published = _material(changed=changed)
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
        },
    )
    evidence = AdaptationEvidence(trace.content_sha256, (proof,))
    return evidence, trace, (event,)


def test_adaptation_evidence_round_trips_and_replays_report_and_publication(tmp_path: Path) -> None:
    evidence, trace, events = _bound_evidence()
    validate_adaptation_evidence_binding(evidence, trace, events)
    path = tmp_path / "adaptation_evidence.npz"
    save_adaptation_evidence(evidence, path)
    restored = load_adaptation_evidence(path)
    assert restored.content_sha256 == evidence.content_sha256
    validate_adaptation_evidence_binding(restored, trace, events)


def test_adaptation_evidence_rejects_no_parameter_change_and_forged_event() -> None:
    unchanged, trace, events = _bound_evidence(changed=False)
    with pytest.raises(ValueError, match="no BPTT parameter change"):
        validate_adaptation_evidence_binding(unchanged, trace, events)

    evidence, trace, events = _bound_evidence()
    forged = replace(events[0], details={**events[0].details, "candidate_digest": "f" * 64})
    with pytest.raises(ValueError, match="candidate digest"):
        validate_adaptation_evidence_binding(evidence, trace, (forged,))


def test_adaptation_evidence_rejects_raw_validation_tampering() -> None:
    evidence, trace, events = _bound_evidence()
    decision = evidence.decisions[0]
    tampered_raw = replace(decision.evidence, current_policy_margins=np.asarray([-9.0, -8.0]))
    tampered = replace(evidence, decisions=(replace(decision, evidence=tampered_raw),))
    with pytest.raises(ValueError, match="does not recompute"):
        validate_adaptation_evidence_binding(tampered, trace, events)
