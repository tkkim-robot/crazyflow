"""Auditable update opportunities and next-boundary publication for mechanism experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DeterministicUpdateSchedule:
    """An exogenous launch mask, independent of safety, parameter values, and machine load.

    Mechanism replay synchronizes each launched update and publishes its complete result at
    the next simulated boundary. It does not claim the service fits the simulated period.
    Measured paced execution must use BoundarySnapshotScheduler instead.
    """

    opportunities: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Reject ambiguous masks before any experiment begins."""
        if not self.opportunities or any(type(value) is not bool for value in self.opportunities):
            raise ValueError("opportunities must be a nonempty sequence of booleans")

    @classmethod
    def periodic(cls, count: int, *, first: int = 0, every: int = 1) -> DeterministicUpdateSchedule:
        for name, value in (("count", count), ("first", first), ("every", every)):
            if type(value) is not int or value < (0 if name == "first" else 1):
                raise ValueError("count/every must be positive integers; first nonnegative")
        return cls(tuple(index >= first and (index - first) % every == 0 for index in range(count)))

    @classmethod
    def from_service_records(cls, rows: list[dict[str, object]]) -> DeterministicUpdateSchedule:
        times = np.asarray([row["simulation_time"] for row in rows], dtype=float)
        durations = np.asarray([row["learner_seconds"] for row in rows], dtype=float)
        if (
            not np.all(np.isfinite(times))
            or not np.all(np.diff(times) > 0)
            or not np.all(np.isfinite(durations))
            or np.any(durations < 0)
        ):
            raise ValueError("service times must increase; durations must be nonnegative finite")
        return cls(tuple(bool(value > 0) for value in durations))

    def metadata(self) -> dict[str, object]:
        return {
            "mode": "deterministic",
            "opportunities": list(self.opportunities),
            "publication": "complete synchronized update at the next simulated boundary",
            "deployment_budget_claim": False,
        }
