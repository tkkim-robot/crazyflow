from dataclasses import replace

import numpy as np
import pytest

from crazyflow.safety.da_plcbf.actuator_hover import HoverReturnTask, evaluate_hover_return


def trace() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(101) * 0.1
    state = np.zeros((len(t), 17))
    state[:, 6] = 1
    return t, state


def test_return_requires_speed_and_continuous_dwell():
    t, state = trace()
    task = HoverReturnTask(return_windows=((2, 4), (6, 8)))
    state[(t >= 2) & (t < 3), 7] = 1
    state[(t >= 6) & (t < 7), 0] = 1
    result = evaluate_hover_return(t, state, np.zeros(3), 10, task)
    assert result["task_completed"]
    assert result["return_windows"][0]["dwell_completed_at_seconds"] == pytest.approx(3.8)
    assert result["return_windows"][1]["dwell_completed_at_seconds"] == pytest.approx(7.8)
    state[(t >= 7.4) & (t < 7.8), 0] = 1
    assert not evaluate_hover_return(t, state, np.zeros(3), 10, task)["task_completed"]


def test_safe_early_return_cannot_credit_an_unobserved_suffix_or_final_departure():
    t, state = trace()
    task = HoverReturnTask(return_windows=((2, 4),))
    assert not evaluate_hover_return(t[:51], state[:51], np.zeros(3), 10, task)["task_completed"]
    state[t > 9.5, 0] = 2
    result = evaluate_hover_return(t, state, np.zeros(3), 10, task)
    assert result["return_windows"][0]["returned"]
    assert not result["task_completed"]
    assert not result["final_home_dwell_passed"]


def test_task_rejects_overlapping_or_unobservable_windows():
    for windows in (((2, 4), (3, 5)), ((2, 2.2),), ((9, 11),)):
        with pytest.raises(ValueError):
            replace(HoverReturnTask(), return_windows=windows).validate(10)
