"""Nonblocking active/candidate adaptation orchestration.

The runtime filter reads only :class:`ActiveSnapshotStore.active`. Candidate construction, fixed-
step BPTT, and hard validation run in a single background worker against an immutable captured
active snapshot.  The worker only *stages* a validated candidate.  The controller thread calls
:meth:`AdaptationWorker.publish_at_boundary` at an explicit control boundary, where publication is
an atomic compare-and-swap in ``ActiveSnapshotStore``.  A model update or another admission makes
the staged result stale rather than mutating live data between control-boundary reads.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from crazyflow.safety.da_plcbf.snapshots import (
    ActiveSnapshotStore,
    PolicySnapshot,
    PublicationResult,
)
from crazyflow.safety.da_plcbf.validation import ValidationReport


class AdaptationStatus(StrEnum):
    """Terminal status of one candidate build/validation/publication job."""

    ADMITTED = "admitted"
    REJECTED = "rejected"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class AdaptationSubmission:
    """Immediate, nonblocking response to an adaptation request."""

    submitted: bool
    job_id: int
    reason: str


@dataclass(frozen=True, slots=True)
class AdaptationOutcome:
    """Terminal auditable outcome of a background adaptation job."""

    job_id: int
    status: AdaptationStatus
    base_active_version: int
    base_active_digest: str
    model_version: int
    candidate_digest: str
    report_digest: str
    publication: PublicationResult | None
    error_type: str
    error_message: str


CandidateJob = Callable[[PolicySnapshot, int], tuple[PolicySnapshot, ValidationReport]]


@dataclass(frozen=True, slots=True)
class _PreparedAdaptation:
    """Background-only result retained until the controller declares a publication boundary."""

    job_id: int
    active: PolicySnapshot
    model_version: int
    candidate: PolicySnapshot | None
    report: ValidationReport | None
    error_type: str
    error_message: str


class AdaptationWorker:
    """Single-flight candidate worker that never blocks the active filter hot path."""

    def __init__(self, store: ActiveSnapshotStore, candidate_job: CandidateJob) -> None:
        if not callable(candidate_job):
            raise TypeError("candidate_job must be callable")
        self._store = store
        self._candidate_job = candidate_job
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="da-plcbf-adapt")
        self._lock = threading.RLock()
        self._future: Future[_PreparedAdaptation] | None = None
        self._prepared: _PreparedAdaptation | None = None
        self._last_outcome: AdaptationOutcome | None = None
        self._next_job_id = 0
        self._closed = False

    @property
    def active(self) -> PolicySnapshot:
        """Return the immutable active snapshot without waiting for candidate work."""
        return self._store.active

    @property
    def in_flight(self) -> bool:
        """Return whether candidate construction/validation is currently running."""
        with self._lock:
            self._collect_done_locked()
            return self._future is not None

    @property
    def candidate_ready(self) -> bool:
        """Return whether a background result awaits an explicit publication boundary."""
        with self._lock:
            self._collect_done_locked()
            return self._prepared is not None

    def submit(self) -> AdaptationSubmission:
        """Capture the active/model versions and start one job, or report single-flight busy."""
        with self._lock:
            if self._closed:
                return AdaptationSubmission(False, -1, "worker_closed")
            self._collect_done_locked()
            if self._future is not None:
                return AdaptationSubmission(False, -1, "job_in_flight")
            if self._prepared is not None:
                return AdaptationSubmission(False, -1, "candidate_awaiting_boundary")
            active = self._store.active
            model_version = self._store.model_version
            job_id = self._next_job_id
            self._next_job_id += 1
            self._future = self._executor.submit(self._run_job, job_id, active, model_version)
            return AdaptationSubmission(True, job_id, "submitted")

    def prewarm(self, callback: Callable[[], Any]) -> Any:
        """Run startup-only compilation on the persistent adaptation thread.

        This blocking method is valid only before the first submission.  Warming thread-local
        JAX CPU dispatch on the same executor thread prevents compilation from reappearing inside
        the nonblocking candidate job; it is never called from the controller hot path.
        """
        if not callable(callback):
            raise TypeError("prewarm callback must be callable")
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot prewarm a closed adaptation worker")
            self._collect_done_locked()
            if self._future is not None or self._prepared is not None or self._next_job_id != 0:
                raise RuntimeError("prewarm must run before the first adaptation submission")
            future = self._executor.submit(callback)
        return future.result()

    def poll(self) -> AdaptationOutcome | None:
        """Publish a ready candidate at this caller-declared control boundary without waiting.

        This compatibility name is equivalent to :meth:`publish_at_boundary`.  It never causes a
        background executor thread to mutate the active store.
        """
        return self.publish_at_boundary()

    def publish_at_boundary(self) -> AdaptationOutcome | None:
        """Atomically admit/reject one staged result on the calling controller thread."""
        with self._lock:
            self._collect_done_locked()
            prepared = self._prepared
            if prepared is None:
                return self._last_outcome
            self._prepared = None
            if prepared.candidate is None or prepared.report is None:
                outcome = AdaptationOutcome(
                    job_id=prepared.job_id,
                    status=AdaptationStatus.FAILED,
                    base_active_version=prepared.active.version,
                    base_active_digest=prepared.active.digest,
                    model_version=prepared.model_version,
                    candidate_digest="",
                    report_digest="",
                    publication=None,
                    error_type=prepared.error_type,
                    error_message=prepared.error_message,
                )
            else:
                publication = self._store.admit(prepared.candidate, prepared.report)
                status = (
                    AdaptationStatus.ADMITTED if publication.accepted else AdaptationStatus.REJECTED
                )
                outcome = AdaptationOutcome(
                    job_id=prepared.job_id,
                    status=status,
                    base_active_version=prepared.active.version,
                    base_active_digest=prepared.active.digest,
                    model_version=prepared.model_version,
                    candidate_digest=prepared.candidate.digest,
                    report_digest=prepared.report.digest,
                    publication=publication,
                    error_type="",
                    error_message="",
                )
            self._last_outcome = outcome
            return self._last_outcome

    def wait(self, timeout: float | None = None) -> AdaptationOutcome | None:
        """Wait for tests/offline orchestration; runtime filter code should use :meth:`poll`."""
        with self._lock:
            future = self._future
        if future is None:
            return self.poll()
        future.result(timeout=timeout)
        return self.publish_at_boundary()

    def expire_at_terminal(self, timeout: float | None = None) -> AdaptationOutcome | None:
        """Finish cleanup without publishing work that cannot drive another control.

        A terminal observation is not a control boundary with a future plant transition.  Waiting
        for a candidate there and then admitting it would manufacture an apparent online update
        that never controlled the system.  Long-running experiment harnesses may still wait for
        the single-flight job so its resources do not overlap the next paired trial, but this
        method consumes the staged result without calling :meth:`ActiveSnapshotStore.admit`.
        """
        with self._lock:
            future = self._future
        if future is not None:
            future.result(timeout=timeout)
        with self._lock:
            self._collect_done_locked()
            prepared = self._prepared
            if prepared is None:
                return self._last_outcome
            self._prepared = None
            if prepared.candidate is None or prepared.report is None:
                outcome = AdaptationOutcome(
                    job_id=prepared.job_id,
                    status=AdaptationStatus.FAILED,
                    base_active_version=prepared.active.version,
                    base_active_digest=prepared.active.digest,
                    model_version=prepared.model_version,
                    candidate_digest="",
                    report_digest="",
                    publication=None,
                    error_type=prepared.error_type,
                    error_message=prepared.error_message,
                )
            else:
                outcome = AdaptationOutcome(
                    job_id=prepared.job_id,
                    status=AdaptationStatus.EXPIRED,
                    base_active_version=prepared.active.version,
                    base_active_digest=prepared.active.digest,
                    model_version=prepared.model_version,
                    candidate_digest=prepared.candidate.digest,
                    report_digest=prepared.report.digest,
                    publication=None,
                    error_type="",
                    error_message="terminal_boundary_has_no_future_control",
                )
            self._last_outcome = outcome
            return outcome

    def close(self, *, wait: bool = True) -> None:
        """Stop accepting work and release the executor after optional in-flight completion."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        with self._lock:
            self._collect_done_locked()

    def __enter__(self) -> AdaptationWorker:
        """Return this worker as a managed context."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the worker on context exit."""
        self.close()

    def _run_job(
        self, job_id: int, active: PolicySnapshot, model_version: int
    ) -> _PreparedAdaptation:
        try:
            candidate, report = self._candidate_job(active, model_version)
            return _PreparedAdaptation(
                job_id=job_id,
                active=active,
                model_version=model_version,
                candidate=candidate,
                report=report,
                error_type="",
                error_message="",
            )
        except Exception as error:  # the filter must survive candidate/validation failures
            return _PreparedAdaptation(
                job_id=job_id,
                active=active,
                model_version=model_version,
                candidate=None,
                report=None,
                error_type=type(error).__name__,
                error_message=str(error),
            )

    def _collect_done_locked(self) -> None:
        future = self._future
        if future is not None and future.done():
            self._prepared = future.result()
            self._future = None


__all__ = [
    "AdaptationOutcome",
    "AdaptationStatus",
    "AdaptationSubmission",
    "AdaptationWorker",
    "CandidateJob",
]
