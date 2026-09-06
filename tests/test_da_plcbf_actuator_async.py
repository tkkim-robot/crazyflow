"""Real CPU thread tests for immutable, sequential asynchronous learner handoffs."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pytest

from crazyflow.safety.da_plcbf.actuator_async import (
    AsyncLearnerCompletion,
    AsyncLearnerJob,
    AsyncLearnerWorker,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_WAIT_SECONDS = 2.0


class _State(NamedTuple):
    """Small NumPy pytree containing persistent policy and optimizer state."""

    params: np.ndarray
    optimizer_state: np.ndarray
    library_version: np.ndarray
    latest_dynamics_estimate: dict[str, Any]


def _initial_state() -> _State:
    return _State(
        np.asarray([2.0]), np.asarray([11.0]), np.asarray(7), {"effectiveness": [np.asarray([0.8])]}
    )


def _advance(state: _State, observation: Any, model: Any) -> tuple[_State, dict[str, Any]]:
    return state._replace(
        params=state.params + state.optimizer_state,
        optimizer_state=2 * state.optimizer_state + observation[0],
        library_version=state.library_version + 1,
        latest_dynamics_estimate=model,
    ), {"finite_update_applied": True, "loss": np.asarray([1.0])}


@contextmanager
def _worker(
    step: Callable[..., Any] = _advance,
    *,
    initial: _State | None = None,
    synchronize: Callable[[Any], Any] = lambda value: value,
    unblock: tuple[threading.Event, ...] = (),
) -> Iterator[AsyncLearnerWorker]:
    worker = AsyncLearnerWorker(
        _initial_state() if initial is None else initial, step, synchronize=synchronize
    )
    try:
        yield worker
    finally:
        for event in unblock:
            event.set()
        worker.shutdown()


def _submit(
    worker: AsyncLearnerWorker,
    observation: Any = None,
    model: Any = None,
    *,
    control_index: int = 4,
) -> AsyncLearnerJob:
    return worker.submit(
        np.asarray([1.0]) if observation is None else observation,
        {"effectiveness": [np.asarray([0.9])]} if model is None else model,
        training_simulation_time=1.0,
        model_observation_simulation_time=0.96,
        sensed_state_sha256="sensed-state-sha256",
        estimated_model_sha256="model-sha256",
        control_index=control_index,
    )


def _wait_idle(worker: AsyncLearnerWorker) -> None:
    deadline = time.perf_counter() + _WAIT_SECONDS
    pause = threading.Event()
    while worker.active and time.perf_counter() < deadline:
        pause.wait(0.001)
    assert not worker.active, "learner did not reach visible completion"


def _completed(worker: AsyncLearnerWorker) -> AsyncLearnerCompletion:
    _wait_idle(worker)
    result = worker.take_completed(time.perf_counter())
    assert result is not None
    return result


def test_submit_returns_while_learner_is_blocked() -> None:
    entered = threading.Event()
    release = threading.Event()
    submitted = threading.Event()
    calls: list[int] = []
    jobs: list[AsyncLearnerJob] = []
    errors: list[BaseException] = []

    def step(state: _State, observation: Any, model: Any) -> Any:
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(_WAIT_SECONDS)
        return _advance(state, observation, model)

    with _worker(step, unblock=(release,)) as worker:

        def submit() -> None:
            try:
                jobs.append(_submit(worker))
            except BaseException as exc:
                errors.append(exc)
            finally:
                submitted.set()

        caller = threading.Thread(target=submit)
        caller.start()
        try:
            assert submitted.wait(_WAIT_SECONDS), "submit waited for learner computation"
            assert not errors
            assert entered.wait(_WAIT_SECONDS)
            assert calls == [worker._thread.ident]
            assert calls[0] != caller.ident
            assert not release.is_set()
            assert worker.active and worker.busy
            assert worker.take_completed(time.perf_counter()) is None
        finally:
            release.set()
            caller.join(_WAIT_SECONDS)
        assert not caller.is_alive()
        result = _completed(worker)
        assert result.error is None
        assert result.job is jobs[0]
        assert result.job.submitted_wall_time <= result.started_wall_time
        assert result.started_wall_time <= result.completed_wall_time <= time.perf_counter()
        assert result.service_seconds >= 0.0


def test_busy_rejects_jobs_until_completed_result_is_consumed() -> None:
    entered = threading.Event()
    release = threading.Event()

    def step(state: _State, observation: Any, model: Any) -> Any:
        entered.set()
        assert release.wait(_WAIT_SECONDS)
        return _advance(state, observation, model)

    with _worker(step, unblock=(release,)) as worker:
        first = _submit(worker)
        assert entered.wait(_WAIT_SECONDS)
        with pytest.raises(RuntimeError, match="consume the preceding result"):
            _submit(worker)
        release.set()
        _wait_idle(worker)
        assert worker.busy and not worker.active
        with pytest.raises(RuntimeError, match="consume the preceding result"):
            _submit(worker)
        assert _completed(worker).job is first
        assert not worker.busy
        assert worker.take_completed(time.perf_counter()) is None
        second = _submit(worker, control_index=5)
        assert second.job_id == first.job_id + 1
        assert _completed(worker).job is second


def test_completion_waits_for_synchronization_and_respects_boundary_cutoff() -> None:
    synchronizing = threading.Event()
    release = threading.Event()

    def synchronize(value: Any) -> Any:
        synchronizing.set()
        assert release.wait(_WAIT_SECONDS)
        return value

    with _worker(synchronize=synchronize, unblock=(release,)) as worker:
        initial = worker.continuation_state
        job = _submit(worker)
        assert synchronizing.wait(_WAIT_SECONDS)
        assert worker.active and worker.busy
        assert worker.continuation_state is initial
        assert worker.take_completed(float("inf")) is None
        boundary = time.perf_counter()
        release.set()
        _wait_idle(worker)
        assert worker.take_completed(boundary) is None
        assert worker.busy
        result = _completed(worker)
        assert result.error is None
        assert result.completed_wall_time > boundary
        assert result.job is job
        assert worker.continuation_state is result.state
        assert int(result.state.library_version) == 8


def test_job_and_completed_state_detach_mutable_numpy_inputs_and_outputs() -> None:
    entered = threading.Event()
    release = threading.Event()
    initial = _initial_state()
    observation = np.asarray([3.0, 4.0])
    model = {"effectiveness": [np.asarray([0.7])]}
    outputs: list[Any] = []

    def step(state: _State, observed: Any, point: Any) -> Any:
        entered.set()
        assert release.wait(_WAIT_SECONDS)
        np.testing.assert_array_equal(state.params, [2.0])
        np.testing.assert_array_equal(state.optimizer_state, [11.0])
        np.testing.assert_array_equal(state.latest_dynamics_estimate["effectiveness"][0], [0.8])
        np.testing.assert_array_equal(observed, [3.0, 4.0])
        np.testing.assert_array_equal(point["effectiveness"][0], [0.7])
        value = _advance(state, observed, point)
        outputs.append(value)
        return value

    with _worker(step, initial=initial, unblock=(release,)) as worker:
        job = _submit(worker, observation, model)
        assert entered.wait(_WAIT_SECONDS)
        initial.params[:] = 100
        initial.optimizer_state[:] = 200
        initial.latest_dynamics_estimate["effectiveness"][0][:] = 300
        observation[:] = 400
        model["effectiveness"][0][:] = 500
        model["effectiveness"].append(np.asarray([600.0]))
        release.set()
        result = _completed(worker)
        assert result.error is None
        np.testing.assert_array_equal(job.observation, [3.0, 4.0])
        assert len(job.model["effectiveness"]) == 1
        np.testing.assert_array_equal(job.model["effectiveness"][0], [0.7])
        assert job.training_simulation_time == 1.0
        assert job.model_observation_simulation_time == 0.96
        assert job.sensed_state_sha256 == "sensed-state-sha256"
        assert job.estimated_model_sha256 == "model-sha256"
        assert job.control_index == 4
        outputs[0][0].params[:] = 700
        outputs[0][0].optimizer_state[:] = 800
        outputs[0][1]["loss"][:] = 900
        np.testing.assert_array_equal(result.state.params, [13.0])
        np.testing.assert_array_equal(result.state.optimizer_state, [25.0])
        np.testing.assert_array_equal(result.metrics["loss"], [1.0])
        for array in (
            job.observation,
            job.model["effectiveness"][0],
            result.state.params,
            result.state.optimizer_state,
            result.metrics["loss"],
        ):
            with pytest.raises(ValueError, match="read-only"):
                array.flat[0] = -1


def test_each_job_continues_the_completed_optimizer_without_stale_branches() -> None:
    seen: list[_State] = []

    def step(state: _State, observation: Any, model: Any) -> Any:
        seen.append(state)
        return _advance(state, observation, model)

    with _worker(step) as worker:
        first_job = _submit(worker, np.asarray([1.0]))
        first = _completed(worker)
        second_job = _submit(worker, np.asarray([3.0]))
        second = _completed(worker)
        assert first.error is second.error is None
        assert (first_job.parent_version, second_job.parent_version) == (7, 8)
        assert (first_job.job_id, second_job.job_id) == (0, 1)
        assert seen[1] is first.state
        np.testing.assert_array_equal(seen[0].optimizer_state, [11.0])
        np.testing.assert_array_equal(seen[1].optimizer_state, [23.0])
        np.testing.assert_array_equal(second.state.optimizer_state, [49.0])
        np.testing.assert_array_equal(second.state.params, [36.0])
        np.testing.assert_array_equal(first.state.params, [13.0])
        assert int(second.state.library_version) == 9
        assert worker.continuation_state is second.state


def test_nonfinite_update_diagnostics_allow_recovery_from_retained_optimizer() -> None:
    seen: list[_State] = []

    def step(state: _State, observation: Any, model: Any) -> Any:
        seen.append(state)
        if len(seen) == 1:
            # The learner rejects its proposal but retains the observed model.
            return state._replace(latest_dynamics_estimate=model), {
                "finite_update_applied": False,
                "loss": np.asarray([float("nan")]),
            }
        return _advance(state, observation, model)

    with _worker(step) as worker:
        _submit(worker)
        rejected = _completed(worker)
        assert rejected.error is None
        assert not rejected.metrics["finite_update_applied"]
        assert np.isnan(rejected.metrics["loss"]).all()
        np.testing.assert_array_equal(rejected.state.optimizer_state, [11.0])
        assert int(rejected.state.library_version) == 7
        job = _submit(worker)
        recovered = _completed(worker)
        assert recovered.error is None
        assert seen[1] is rejected.state
        assert job.parent_version == 7
        np.testing.assert_array_equal(seen[1].latest_dynamics_estimate["effectiveness"][0], [0.9])
        np.testing.assert_array_equal(recovered.state.optimizer_state, [23.0])
        assert int(recovered.state.library_version) == 8


@pytest.mark.parametrize("failure_site", ["step", "synchronize"])
def test_exception_is_captured_and_stops_further_submissions(failure_site: str) -> None:
    def step(state: _State, observation: Any, model: Any) -> Any:
        if failure_site == "step":
            raise ValueError("learner failed deliberately")
        return _advance(state, observation, model)

    def synchronize(value: Any) -> Any:
        if failure_site == "synchronize":
            raise ValueError("learner failed deliberately")
        return value

    with _worker(step, synchronize=synchronize) as worker:
        initial = worker.continuation_state
        job = _submit(worker)
        result = _completed(worker)
        assert result.job is job
        assert result.error is not None
        assert result.error["type"] == "ValueError"
        assert result.error["message"] == "learner failed deliberately"
        assert "ValueError: learner failed deliberately" in result.error["traceback"]
        assert worker.continuation_state is initial
        assert not worker.busy and not worker.active
        with pytest.raises(RuntimeError, match="closed or failed"):
            _submit(worker)


def test_shutdown_finishes_outstanding_job_and_rejects_new_submissions() -> None:
    entered = threading.Event()
    release = threading.Event()
    closing = threading.Event()
    closed = threading.Event()

    def step(state: _State, observation: Any, model: Any) -> Any:
        entered.set()
        assert release.wait(_WAIT_SECONDS)
        return _advance(state, observation, model)

    with _worker(step, unblock=(release,)) as worker:
        job = _submit(worker)
        assert entered.wait(_WAIT_SECONDS)

        def shutdown() -> None:
            closing.set()
            worker.shutdown()
            closed.set()

        closer = threading.Thread(target=shutdown)
        closer.start()
        try:
            assert closing.wait(_WAIT_SECONDS)
            assert not closed.wait(0.02)
            assert worker.active
        finally:
            release.set()
            closer.join(_WAIT_SECONDS)
        assert closed.is_set()
        assert not closer.is_alive()
        assert not worker._thread.is_alive()
        result = _completed(worker)
        assert result.job is job and result.error is None
        assert int(result.state.library_version) == 8
        assert worker.continuation_state is result.state
        with pytest.raises(RuntimeError, match="closed or failed"):
            _submit(worker)
