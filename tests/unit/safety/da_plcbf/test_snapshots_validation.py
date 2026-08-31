from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.snapshots import (
    ActiveSnapshotStore,
    create_active_snapshot,
    create_candidate_snapshot,
    tree_content_digest,
)
from crazyflow.safety.da_plcbf.validation import (
    GATE_NAMES,
    HardValidationEvidence,
    HardValidationThresholds,
    hard_validate_candidate,
)


def _active(*, model_version: int = 3) -> Any:
    return create_active_snapshot(
        {"adaptive": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)},
        version=4,
        model_version=model_version,
        structural_core={"codes": np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)},
        metadata={"name": "validated seed", "nested": {"fold": 2}},
    )


def _candidate(active: Any, *, model_version: int | None = None) -> Any:
    return create_candidate_snapshot(
        {"adaptive": np.array([[0.15, 0.25], [0.35, 0.45]], dtype=np.float32)},
        version=active.version + 1,
        base_active=active,
        model_version=model_version,
        metadata={"optimizer_steps": 10},
    )


def _evidence(**replacements: Any) -> HardValidationEvidence:
    values: dict[str, Any] = {
        "current_policy_margins": np.array([0.2, 0.1, -0.3]),
        "candidate_local_policy_margins": np.array(
            [[0.30, 0.20, 0.25], [0.10, 0.05, 0.15], [-0.4, -0.3, -0.2]]
        ),
        "active_local_policy_margins": np.array(
            [[0.25, 0.18, 0.20], [0.05, 0.02, 0.10], [-0.5, -0.4, -0.3]]
        ),
        "candidate_descriptors": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "descriptor_scales": np.ones(2),
        "feasibility_margins": np.array([0.1, 0.2, 0.05]),
        "runtime_seconds": np.array([0.002, 0.003, 0.004]),
        "validation_set_digest": "scenario-tape-sha256",
    }
    values.update(replacements)
    return HardValidationEvidence(**values)


def _thresholds(**replacements: Any) -> HardValidationThresholds:
    values: dict[str, Any] = {
        "minimum_current_margin": 0.0,
        "safe_policy_margin": 0.0,
        "local_non_regression_tolerance": 0.0,
        "minimum_coverage": 1.0,
        "minimum_redundancy": 2,
        "minimum_diversity": 0.5,
        "minimum_feasible_fraction": 1.0,
        "maximum_runtime_seconds": 0.005,
    }
    values.update(replacements)
    return HardValidationThresholds(**values)


def _report(active: Any, candidate: Any, **evidence_replacements: Any) -> Any:
    return hard_validate_candidate(
        active,
        candidate,
        _evidence(**evidence_replacements),
        _thresholds(),
        current_model_version=candidate.model_version,
    )


def test_snapshot_hash_is_deterministic_and_covers_all_payload_components() -> None:
    first = create_active_snapshot(
        {"b": np.array([3], dtype=np.int32), "a": np.array([1.0, 2.0], dtype=np.float32)},
        version=0,
        model_version=0,
        structural_core=(np.array([5.0], dtype=np.float32),),
        metadata={"z": 1, "a": {"right": True}},
    )
    reordered = create_active_snapshot(
        {"a": jnp.array([1.0, 2.0], dtype=jnp.float32), "b": jnp.array([3], dtype=jnp.int32)},
        version=0,
        model_version=0,
        structural_core=(jnp.array([5.0], dtype=jnp.float32),),
        metadata={"a": {"right": True}, "z": 1},
    )

    assert first.digest == reordered.digest
    assert first.params_digest == reordered.params_digest
    assert first.verify_integrity()
    assert tree_content_digest(first.params) == first.params_digest

    variants = (
        create_active_snapshot(
            {"a": np.array([1.0, 2.0], dtype=np.float64), "b": np.array([3], dtype=np.int32)},
            structural_core=(np.array([5.0]),),
            metadata={"a": {"right": True}, "z": 1},
        ),
        create_active_snapshot(
            {"a": np.array([[1.0, 2.0]], dtype=np.float32), "b": np.array([3], dtype=np.int32)},
            structural_core=(np.array([5.0]),),
            metadata={"a": {"right": True}, "z": 1},
        ),
        create_active_snapshot(
            {"a": np.array([1.0, 2.1], dtype=np.float32), "b": np.array([3], dtype=np.int32)},
            structural_core=(np.array([5.0]),),
            metadata={"a": {"right": True}, "z": 1},
        ),
        create_active_snapshot(
            (np.array([1.0, 2.0], dtype=np.float32), np.array([3], dtype=np.int32)),
            structural_core=(np.array([5.0]),),
            metadata={"a": {"right": True}, "z": 1},
        ),
        create_active_snapshot(
            {"a": np.array([1.0, 2.0], dtype=np.float32), "b": np.array([3], dtype=np.int32)},
            structural_core=(np.array([6.0]),),
            metadata={"a": {"right": True}, "z": 1},
        ),
        create_active_snapshot(
            {"a": np.array([1.0, 2.0], dtype=np.float32), "b": np.array([3], dtype=np.int32)},
            structural_core=(np.array([5.0]),),
            metadata={"a": {"right": False}, "z": 1},
        ),
    )
    assert len({first.digest, *(snapshot.digest for snapshot in variants)}) == len(variants) + 1


def test_snapshot_defensively_freezes_inputs_outputs_and_metadata() -> None:
    source = np.array([1.0, 2.0], dtype=np.float32)
    params = {"weights": source}
    snapshot = create_active_snapshot(
        params, structural_core={"fixed": source}, metadata={"nested": {"values": [1, 2]}}
    )
    original_digest = snapshot.digest

    source[:] = 99.0
    params["extra"] = np.array([4.0])
    np.testing.assert_array_equal(snapshot.params["weights"], [1.0, 2.0])
    np.testing.assert_array_equal(snapshot.structural_core["fixed"], [1.0, 2.0])

    exposed = snapshot.params
    with pytest.raises(ValueError):
        exposed["weights"][0] = -4.0
    with pytest.raises(ValueError):
        exposed["weights"].setflags(write=True)
    exposed["new_container_key"] = np.array([10.0])
    assert "new_container_key" not in snapshot.params

    with pytest.raises(TypeError):
        snapshot.metadata["new"] = 3
    with pytest.raises(TypeError):
        snapshot.metadata["nested"]["other"] = 4
    assert snapshot.digest == original_digest
    assert snapshot.verify_integrity()

    copied_metadata = create_active_snapshot(np.array([1.0]), metadata=snapshot.metadata).metadata
    assert copied_metadata["nested"]["values"] == (1, 2)


def test_passing_report_logs_every_named_gate_and_is_content_addressed() -> None:
    active = _active()
    candidate = _candidate(active)
    report = _report(active, candidate)

    assert report.passed
    assert report.failed_gate_names == ()
    assert tuple(record["name"] for record in report.as_log_records()) == GATE_NAMES
    assert len(report.as_log_records()) == len(GATE_NAMES)
    assert report.local_non_regression_passes == (True, True, True)
    assert report.verify_integrity()
    current_gate = next(gate for gate in report.gates if gate.name == "current_margin")
    assert current_gate.observed == "0.20000000000000001"
    assert current_gate.requirement == ">=0"

    tampered = replace(report, validation_set_digest="another-tape")
    assert not tampered.verify_integrity()


@pytest.mark.parametrize(
    ("evidence_replacements", "failed_gate"),
    [
        ({"current_policy_margins": np.array([-0.2, -0.1])}, "current_margin"),
        (
            {
                "candidate_local_policy_margins": np.array(
                    [[0.30, 0.10, 0.25], [0.10, -0.05, 0.15], [-0.4, -0.3, -0.2]]
                )
            },
            "local_non_regression",
        ),
        (
            {
                "candidate_local_policy_margins": np.array(
                    [[0.30, -0.10, 0.25], [0.10, -0.05, 0.15], [-0.4, -0.3, -0.2]]
                )
            },
            "coverage",
        ),
        (
            {
                "candidate_local_policy_margins": np.array(
                    [[0.30, 0.20, 0.25], [-0.10, -0.05, -0.15], [-0.4, -0.3, -0.2]]
                )
            },
            "redundancy",
        ),
        ({"feasibility_margins": np.array([0.1, -1e-6, 0.2])}, "feasibility"),
    ],
)
def test_quantitative_bad_candidates_fail_the_corresponding_hard_gate(
    evidence_replacements: dict[str, Any], failed_gate: str
) -> None:
    active = _active()
    candidate = _candidate(active)
    report = _report(active, candidate, **evidence_replacements)

    assert not report.passed
    assert failed_gate in report.failed_gate_names
    assert report.verify_integrity()


def test_nonfinite_candidate_and_evidence_fail_closed() -> None:
    active = _active()
    candidate = create_candidate_snapshot(
        {"adaptive": np.array([[np.nan, 0.0]])}, version=active.version + 1, base_active=active
    )
    report = _report(active, candidate, runtime_seconds=np.array([np.inf]))

    assert not report.passed
    assert "finite_values" in report.failed_gate_names
    assert "runtime_budget" in report.failed_gate_names
    store = ActiveSnapshotStore(active)
    result = store.admit(candidate, report)
    assert not result.accepted
    assert result.reason == "candidate_nonfinite"
    assert store.active is active


def test_collapsed_library_and_slower_candidate_are_rejected() -> None:
    active = _active()
    candidate = _candidate(active)
    collapsed = _report(active, candidate, candidate_descriptors=np.zeros((3, 2)))
    slower = _report(active, candidate, runtime_seconds=np.array([0.004, 0.005001]))

    assert collapsed.failed_gate_names == ("diversity",)
    assert slower.failed_gate_names == ("runtime_budget",)
    store = ActiveSnapshotStore(active)
    assert store.admit(candidate, collapsed).reason == "hard_validation_failed"
    assert store.admit(candidate, slower).reason == "hard_validation_failed"
    assert store.active is active


def test_structural_core_change_is_detected_from_snapshot_content() -> None:
    active = _active()
    candidate = create_candidate_snapshot(
        {"adaptive": np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)},
        version=active.version + 1,
        base_active=active,
        structural_core={"codes": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)},
    )
    report = _report(active, candidate)

    assert report.failed_gate_names == ("structural_core_preservation",)


def test_parameter_schema_change_cannot_be_admitted() -> None:
    active = _active()
    candidate = create_candidate_snapshot(
        {"adaptive": np.zeros((3, 2), dtype=np.float32)},
        version=active.version + 1,
        base_active=active,
    )
    report = _report(active, candidate)

    assert report.failed_gate_names == ("parameter_schema_compatibility",)
    store = ActiveSnapshotStore(active)
    assert store.admit(candidate, report).reason == "hard_validation_failed"


def test_model_change_rejects_inflight_candidate_and_old_report() -> None:
    active = _active(model_version=3)
    candidate = _candidate(active)
    old_report = _report(active, candidate)
    store = ActiveSnapshotStore(active)

    store.advance_model_version(4)
    result = store.admit(candidate, old_report)
    assert not result.accepted
    assert result.reason == "stale_model_version"
    assert store.active is active

    fresh_candidate = _candidate(active, model_version=4)
    stale_validation = hard_validate_candidate(
        active, fresh_candidate, _evidence(), _thresholds(), current_model_version=3
    )
    assert stale_validation.failed_gate_names == ("model_version_freshness",)

    fresh_report = hard_validate_candidate(
        active, fresh_candidate, _evidence(), _thresholds(), current_model_version=4
    )
    assert fresh_report.passed
    assert store.admit(fresh_candidate, fresh_report).accepted


def test_atomic_concurrent_admission_accepts_exactly_one_candidate() -> None:
    active = _active()
    candidate = _candidate(active)
    report = _report(active, candidate)
    store = ActiveSnapshotStore(active)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: store.admit(candidate, report), range(32)))

    assert sum(result.accepted for result in results) == 1
    assert store.active.version == active.version + 1
    assert store.previous is active
    assert all(result.reason in {"admitted", "stale_base_active_version"} for result in results)


def test_stale_base_and_report_binding_cannot_publish() -> None:
    active = _active()
    first_candidate = _candidate(active)
    first_report = _report(active, first_candidate)
    store = ActiveSnapshotStore(active)
    assert store.admit(first_candidate, first_report).accepted
    first_publication = store.active

    stale = store.admit(first_candidate, first_report)
    assert not stale.accepted
    assert stale.reason == "stale_base_active_version"
    assert store.active is first_publication

    fresh_candidate = _candidate(first_publication)
    mismatched = store.admit(fresh_candidate, first_report)
    assert not mismatched.accepted
    assert mismatched.reason == "report_active_mismatch"
    assert store.active is first_publication


def test_tampered_snapshot_and_report_integrity_are_rejected() -> None:
    active = _active()
    candidate = _candidate(active)
    report = _report(active, candidate)
    store = ActiveSnapshotStore(active)

    bad_candidate = replace(candidate, digest="0" * 64)
    assert not bad_candidate.verify_integrity()
    assert store.admit(bad_candidate, report).reason == "snapshot_integrity_failed"

    bad_report = replace(report, digest="f" * 64)
    assert not bad_report.verify_integrity()
    assert store.admit(candidate, bad_report).reason == "report_integrity_failed"
    assert store.active is active


def test_rollback_republishes_previous_payload_without_mutating_history() -> None:
    active = _active()
    active_digest = active.digest
    candidate = _candidate(active)
    candidate_digest = candidate.digest
    report = _report(active, candidate)
    store = ActiveSnapshotStore(active)
    admitted = store.admit(candidate, report)
    assert admitted.accepted
    admitted_snapshot = admitted.active
    admitted_digest = admitted_snapshot.digest

    rolled_back = store.publish_rollback(expected_active_version=admitted_snapshot.version)

    assert rolled_back.accepted
    assert rolled_back.reason == "rollback_published"
    assert rolled_back.active.version == admitted_snapshot.version + 1
    assert rolled_back.active.digest not in {active_digest, admitted_digest, candidate_digest}
    assert rolled_back.active.params_digest == active.params_digest
    assert rolled_back.active.structural_core_digest == active.structural_core_digest
    assert rolled_back.active.metadata["publication"]["type"] == "rollback"
    assert store.previous is admitted_snapshot
    assert active.digest == active_digest
    assert candidate.digest == candidate_digest
    assert admitted_snapshot.digest == admitted_digest
    assert active.version == 4
    assert admitted_snapshot.version == 5


def test_rollback_rejects_wrong_active_or_model_version() -> None:
    active = _active()
    store = ActiveSnapshotStore(active)
    assert store.publish_rollback().reason == "no_previous_snapshot"
    candidate = _candidate(active)
    assert store.admit(candidate, _report(active, candidate)).accepted

    assert store.publish_rollback(expected_active_version=999).reason == "active_version_mismatch"
    store.advance_model_version(4)
    assert store.publish_rollback().reason == "previous_model_version_stale"
