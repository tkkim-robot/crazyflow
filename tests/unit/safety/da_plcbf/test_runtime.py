from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from crazyflow.safety.da_plcbf.runtime import AdaptationStatus, AdaptationWorker
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


def _store() -> ActiveSnapshotStore:
    active = create_active_snapshot(
        {"adaptive": np.array([[0.1], [0.2]], dtype=np.float32)},
        structural_core={"fixed": np.array([[-1.0], [1.0]], dtype=np.float32)},
    )
    return ActiveSnapshotStore(active)


def _passing_job(active: Any, model_version: int) -> tuple[Any, Any]:
    candidate = create_candidate_snapshot(
        {"adaptive": np.array([[0.11], [0.21]], dtype=np.float32)},
        version=active.version + 1,
        base_active=active,
        model_version=model_version,
    )
    evidence = HardValidationEvidence(
        current_policy_margins=np.array([0.2, 0.1]),
        candidate_local_policy_margins=np.array([[0.3, 0.2], [0.1, 0.1]]),
        active_local_policy_margins=np.array([[0.2, 0.1], [0.0, 0.0]]),
        candidate_descriptors=np.array([[0.0], [1.0]]),
        descriptor_scales=np.ones(1),
        feasibility_margins=np.ones(2),
        runtime_seconds=np.array([0.001]),
        validation_set_digest="runtime-test-tape",
    )
    thresholds = HardValidationThresholds(
        minimum_redundancy=1, minimum_diversity=0.5, maximum_runtime_seconds=0.01
    )
    report = hard_validate_candidate(
        active, candidate, evidence, thresholds, current_model_version=model_version
    )
    assert report.passed
    return candidate, report


def test_active_snapshot_remains_byte_identical_while_candidate_job_is_blocked() -> None:
    store = _store()
    entered = threading.Event()
    release = threading.Event()

    def blocked_job(active: Any, model_version: int) -> tuple[Any, Any]:
        entered.set()
        assert release.wait(timeout=5)
        return _passing_job(active, model_version)

    with AdaptationWorker(store, blocked_job) as worker:
        original = worker.active
        submission = worker.submit()
        assert submission.submitted
        assert entered.wait(timeout=5)
        assert worker.in_flight
        for _ in range(100):
            assert worker.active is original
            assert worker.active.digest == original.digest
            np.testing.assert_array_equal(
                worker.active.params["adaptive"], original.params["adaptive"]
            )
        busy = worker.submit()
        assert not busy.submitted
        assert busy.reason == "job_in_flight"
        release.set()
        outcome = worker.wait(timeout=5)

    assert outcome is not None
    assert outcome.status == AdaptationStatus.ADMITTED
    assert outcome.publication is not None and outcome.publication.accepted
    assert store.active.version == original.version + 1
    np.testing.assert_array_equal(
        original.params["adaptive"], np.array([[0.1], [0.2]], dtype=np.float32)
    )


def test_completed_background_candidate_cannot_publish_before_explicit_control_boundary() -> None:
    store = _store()
    original = store.active
    job_thread = -1

    def recording_job(active: Any, model_version: int) -> tuple[Any, Any]:
        nonlocal job_thread
        job_thread = threading.get_ident()
        return _passing_job(active, model_version)

    with AdaptationWorker(store, recording_job) as worker:
        assert worker.submit().submitted
        deadline = time.monotonic() + 5.0
        while not worker.candidate_ready and time.monotonic() < deadline:
            time.sleep(0.001)

        assert worker.candidate_ready
        assert not worker.in_flight
        assert store.active is original
        assert store.active.version == 0
        blocked = worker.submit()
        assert not blocked.submitted
        assert blocked.reason == "candidate_awaiting_boundary"

        controller_thread = threading.get_ident()
        outcome = worker.publish_at_boundary()

    assert outcome is not None and outcome.status == AdaptationStatus.ADMITTED
    assert job_thread != controller_thread
    assert store.active.version == 1


def test_terminal_cleanup_expires_candidate_without_publishing() -> None:
    store = _store()
    original = store.active
    with AdaptationWorker(store, _passing_job) as worker:
        assert worker.submit().submitted
        outcome = worker.expire_at_terminal(timeout=5)

    assert outcome is not None
    assert outcome.status is AdaptationStatus.EXPIRED
    assert outcome.publication is None
    assert outcome.candidate_digest
    assert outcome.report_digest
    assert outcome.error_message == "terminal_boundary_has_no_future_control"
    assert store.active is original
    assert store.active.version == 0


def test_startup_prewarm_runs_on_the_persistent_candidate_thread() -> None:
    store = _store()
    warm_thread = -1
    job_thread = -1

    def warm() -> str:
        nonlocal warm_thread
        warm_thread = threading.get_ident()
        return "ready"

    def recording_job(active: Any, model_version: int) -> tuple[Any, Any]:
        nonlocal job_thread
        job_thread = threading.get_ident()
        return _passing_job(active, model_version)

    with AdaptationWorker(store, recording_job) as worker:
        assert worker.prewarm(warm) == "ready"
        assert worker.submit().submitted
        outcome = worker.wait(timeout=5)

    assert outcome is not None and outcome.status == AdaptationStatus.ADMITTED
    assert warm_thread == job_thread
    assert warm_thread != threading.get_ident()


def test_model_change_makes_inflight_candidate_stale_instead_of_publishing() -> None:
    store = _store()
    entered = threading.Event()
    release = threading.Event()

    def blocked_job(active: Any, model_version: int) -> tuple[Any, Any]:
        entered.set()
        assert release.wait(timeout=5)
        return _passing_job(active, model_version)

    with AdaptationWorker(store, blocked_job) as worker:
        assert worker.submit().submitted
        assert entered.wait(timeout=5)
        store.advance_model_version(1)
        release.set()
        outcome = worker.wait(timeout=5)

    assert outcome is not None
    assert outcome.status == AdaptationStatus.REJECTED
    assert outcome.publication is not None
    assert outcome.publication.reason == "stale_model_version"
    assert store.active.version == 0


def test_candidate_exception_is_contained_and_filter_active_remains_available() -> None:
    store = _store()

    def failing_job(_active: Any, _model_version: int) -> tuple[Any, Any]:
        raise FloatingPointError("nonfinite candidate loss")

    worker = AdaptationWorker(store, failing_job)
    assert worker.submit().submitted
    outcome = worker.wait(timeout=5)

    assert outcome is not None
    assert outcome.status == AdaptationStatus.FAILED
    assert outcome.error_type == "FloatingPointError"
    assert "nonfinite" in outcome.error_message
    assert worker.active is store.active
    assert store.active.version == 0
    worker.close()
    closed = worker.submit()
    assert not closed.submitted
    assert closed.reason == "worker_closed"


def test_sequential_completed_jobs_capture_the_new_active_identity() -> None:
    store = _store()
    with AdaptationWorker(store, _passing_job) as worker:
        first = worker.submit()
        first_outcome = worker.wait(timeout=5)
        second = worker.submit()
        second_outcome = worker.wait(timeout=5)

    assert first_outcome is not None and second_outcome is not None
    assert first.job_id == 0 and second.job_id == 1
    assert first_outcome.status == second_outcome.status == AdaptationStatus.ADMITTED
    assert second_outcome.base_active_version == first_outcome.base_active_version + 1
    assert second_outcome.base_active_digest == first_outcome.publication.active.digest
    assert store.active.version == 2
