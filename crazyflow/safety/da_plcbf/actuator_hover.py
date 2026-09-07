"""Station-keeping task metrics, separate from the obstacle-agnostic fallback learner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class HoverReturnTask:
    """Prespecified home tolerances and windows in which return should be possible."""

    position_tolerance_m: float = 0.4
    speed_tolerance_mps: float = 0.35
    return_dwell_seconds: float = 0.8
    final_dwell_seconds: float = 1.0
    final_position_tolerance_m: float = 0.15
    final_speed_tolerance_mps: float = 0.15
    return_windows: tuple[tuple[float, float], ...] = ()

    def validate(self, duration: float) -> None:
        for name in (
            "position_tolerance_m",
            "speed_tolerance_mps",
            "return_dwell_seconds",
            "final_dwell_seconds",
            "final_position_tolerance_m",
            "final_speed_tolerance_mps",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite")
        previous = 0.0
        for start, end in self.return_windows:
            if not (
                math.isfinite(start)
                and math.isfinite(end)
                and previous <= start < end <= duration
                and end - start >= self.return_dwell_seconds
            ):
                raise ValueError("return windows must be ordered, disjoint and long enough")
            previous = end
        if self.final_dwell_seconds > duration:
            raise ValueError("final dwell exceeds episode duration")


def _first_dwell(
    times: np.ndarray, near: np.ndarray, start: float, end: float, dwell: float
) -> float | None:
    run_start = None
    for i in range(len(times) - 1):
        left, right = max(start, times[i]), min(end, times[i + 1])
        if right <= left:
            continue
        if near[i] and near[i + 1]:
            run_start = left if run_start is None else run_start
            if right - run_start >= dwell - 1e-9:
                return float(run_start + dwell)
        else:
            run_start = None
    return None


def evaluate_hover_return(
    times: np.ndarray, states: np.ndarray, home: np.ndarray, duration: float, task: HoverReturnTask
) -> dict[str, Any]:
    """Use recorded physical exposure; an unobserved suffix never earns return credit."""
    task.validate(duration)
    times, states, home = np.asarray(times), np.asarray(states), np.asarray(home)
    if (
        times.ndim != 1
        or not len(times)
        or states.shape != (len(times), 17)
        or home.shape != (3,)
        or np.any(np.diff(times) < 0)
        or not np.all(np.isfinite(times))
        or not np.all(np.isfinite(states))
    ):
        raise ValueError("hover metrics require finite ordered time and matching 17-state samples")
    distance = np.linalg.norm(states[:, :3] - home, axis=1)
    speed = np.linalg.norm(states[:, 7:10], axis=1)
    near = (distance <= task.position_tolerance_m) & (speed <= task.speed_tolerance_mps)
    exposure = float(times[-1] - times[0])
    near_seconds = float(np.sum(np.diff(times) * (near[:-1] & near[1:])))
    returns = []
    for start, end in task.return_windows:
        completed = _first_dwell(times, near, start, end, task.return_dwell_seconds)
        returns.append(
            {
                "window_seconds": [start, end],
                "dwell_completed_at_seconds": completed,
                "return_delay_seconds": None
                if completed is None
                else completed - task.return_dwell_seconds - start,
                "window_fully_observed": bool(times[-1] >= end - 1e-9),
                "returned": completed is not None,
            }
        )
    final_start = duration - task.final_dwell_seconds
    final_near = (distance <= task.final_position_tolerance_m) & (
        speed <= task.final_speed_tolerance_mps
    )
    final = _first_dwell(times, final_near, final_start, duration, task.final_dwell_seconds)
    reached = bool(times[-1] >= duration - 1e-9)
    completed = reached and final is not None and all(r["returned"] for r in returns)
    return {
        "task_kind": "hover_avoid_return",
        "home_position_m": home.tolist(),
        "physical_exposure_seconds": exposure,
        "time_near_home_seconds": near_seconds,
        "fraction_near_home": near_seconds / exposure if exposure else 0.0,
        "maximum_displacement_m": float(np.max(distance)),
        "final_home_distance_m": float(distance[-1]),
        "final_speed_mps": float(speed[-1]),
        "return_windows": returns,
        "final_home_dwell_passed": final is not None,
        "task_completed": completed,
        "task_completion_time_seconds": duration if completed else None,
        "measurement": "Both recorded endpoints must satisfy position and speed tolerances; "
        "5 ms physical intervals in aligned experiments; no credit for unobserved suffixes.",
    }
