"""One real background learner with immutable inputs and bounded publication backlog.

The worker owns the persistent learner/optimizer continuation. A caller can submit
one complete observation/model job, and must consume its completed result before
submitting another. Completion means the returned device work has synchronized,
the result has been copied into an immutable snapshot, and that snapshot is visible
under the worker lock. There is no speculative optimizer branch, update cancellation,
quality gate, fake service duration, or assumption of concurrent GPU kernels.
"""

from __future__ import annotations

import copy
import math
import threading
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


def immutable_learner_copy(value: Any) -> Any:
    """Detach mutable host leaves; JAX arrays are shared only as immutable values.

    Learner functions must not donate input buffers. The current actuator learner
    does not donate them. A copied pytree also separates mutable container objects
    from the controller/job caller, while NumPy leaves reject in-place mutation.
    """

    def copy_leaf(leaf: Any) -> Any:
        if isinstance(leaf, np.ndarray):
            copied = leaf.copy()
            copied.setflags(write=False)
            return copied
        if isinstance(leaf, jax.Array):
            return leaf
        return copy.deepcopy(leaf)

    return jax.tree_util.tree_map(copy_leaf, value)


@dataclass(frozen=True, slots=True)
class AsyncLearnerJob:
    """One immutable sensed state/model and its provenance at a real submission."""

    job_id: int
    observation: Any
    model: Any
    training_simulation_time: float
    model_observation_simulation_time: float
    submitted_wall_time: float
    parent_version: int
    sensed_state_sha256: str
    estimated_model_sha256: str
    control_index: int


@dataclass(frozen=True, slots=True)
class AsyncLearnerCompletion:
    """A synchronized finite/nonfinite result or an explicitly captured worker error."""

    job: AsyncLearnerJob
    state: Any
    metrics: Any
    started_wall_time: float
    completed_wall_time: float
    error: dict[str, str] | None = None

    @property
    def service_seconds(self) -> float:
        """Actual job service through visible completion, including host snapshot work."""
        return self.completed_wall_time - self.started_wall_time


class AsyncLearnerWorker:
    """Execute sequential updates on a dedicated thread with at most one held result."""

    def __init__(
        self,
        initial_state: Any,
        step: Callable[[Any, Any, Any], Any],
        *,
        synchronize: Callable[[Any], Any] = jax.block_until_ready,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Start an idle worker; construction neither trains nor publishes a snapshot."""
        self._state = immutable_learner_copy(initial_state)
        self._step = step
        self._synchronize = synchronize
        self._clock = clock
        self._condition = threading.Condition()
        self._job: AsyncLearnerJob | None = None
        self._running = False
        self._completion: AsyncLearnerCompletion | None = None
        self._next_job_id = 0
        self._closed = False
        self._failed = False
        self._thread = threading.Thread(
            target=self._run, name="actuator-persistent-learner", daemon=True
        )
        self._thread.start()

    @property
    def busy(self) -> bool:
        """Reject another job while queued/running or while a result awaits consumption."""
        with self._condition:
            return self._job is not None or self._running or self._completion is not None

    @property
    def active(self) -> bool:
        """Report submitted work that has not yet reached visible synchronized completion."""
        with self._condition:
            return self._job is not None or self._running

    @property
    def continuation_state(self) -> Any:
        """Return the immutable completed continuation, never a partly computed state."""
        with self._condition:
            return self._state

    def submit(
        self,
        observation: Any,
        model: Any,
        *,
        training_simulation_time: float,
        model_observation_simulation_time: float,
        sensed_state_sha256: str,
        estimated_model_sha256: str,
        control_index: int,
    ) -> AsyncLearnerJob:
        """Submit one complete job without waiting for its learner computation.

        Input snapshot-copy cost is paid here on the caller and must be measured by
        the runner. The worker's current continuation determines the parent state;
        a caller cannot supply a competing/stale optimizer branch.
        """
        if (
            not math.isfinite(training_simulation_time)
            or training_simulation_time < 0
            or not math.isfinite(model_observation_simulation_time)
            or not 0 <= model_observation_simulation_time <= training_simulation_time
        ):
            raise ValueError("job state/model observation times must be finite and causal")
        with self._condition:
            if self._closed or self._failed:
                raise RuntimeError("asynchronous learner is closed or failed")
            if self._job is not None or self._running or self._completion is not None:
                raise RuntimeError("consume the preceding result before another learner job")
            job = AsyncLearnerJob(
                self._next_job_id,
                immutable_learner_copy(observation),
                immutable_learner_copy(model),
                training_simulation_time,
                model_observation_simulation_time,
                self._clock(),
                int(self._state.library_version),
                sensed_state_sha256,
                estimated_model_sha256,
                control_index,
            )
            self._next_job_id += 1
            self._job = job
            self._condition.notify()
            return job

    def take_completed(self, through_wall_time: float) -> AsyncLearnerCompletion | None:
        """Consume only a result actually visible by the caller's boundary cutoff."""
        with self._condition:
            result = self._completion
            if result is None or result.completed_wall_time > through_wall_time:
                return None
            self._completion = None
            return result

    def shutdown(self) -> None:
        """Stop accepting work, finish the one outstanding job, and join the worker.

        Waiting after an episode does not grant flight credit or publication; the
        runner must compare the retained completion timestamp with its terminal
        physical-clock cutoff and report cleanup time separately.
        """
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._job is None and not self._closed:
                    self._condition.wait()
                if self._job is None:
                    return
                job = self._job
                self._job = None
                self._running = True
                state = self._state
            started = self._clock()
            following, metrics, error = state, None, None
            try:
                following, metrics = self._synchronize(
                    self._step(state, job.observation, job.model)
                )
                following = immutable_learner_copy(following)
                metrics = immutable_learner_copy(metrics)
            except BaseException as exc:
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            with self._condition:
                # Visibility is part of completion: timestamp and pointer are
                # installed together, after synchronization, under this lock.
                completed = self._clock()
                self._running = False
                if error is None:
                    self._state = following
                else:
                    self._failed = True
                self._completion = AsyncLearnerCompletion(
                    job, following, metrics, started, completed, error
                )
                if self._closed:
                    return
