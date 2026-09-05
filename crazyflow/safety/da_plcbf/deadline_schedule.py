"""Measured, serialized learner service with publication at control boundaries.

This scheduler makes no GPU-overlap assumption. A learner call is launched only when its recent
measured service cost fits after the controller, with reserve. Completed immutable snapshots are
published at a later control boundary. An overrun remains an explicit experiment failure; it is
never hidden by changing the simulated clock or reporting only the fast controller calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CompletedSnapshot:
    """A complete learner result and the timestamps required to audit its availability."""

    state: Any
    version: int
    training_simulation_time: float
    started_wall_time: float
    completed_wall_time: float
    gradient_norm: float = 0.0
    parameter_update_norm: float = 0.0


@dataclass(slots=True)
class BoundarySnapshotScheduler:
    """Nonpreemptive service-cost budget; no partial state or uncompleted update is published."""

    initial_state: Any
    initial_version: int
    measured_update_seconds: float
    reserve_seconds: float = 0.003
    safety_factor: float = 1.25
    published: CompletedSnapshot = field(init=False)
    pending: CompletedSnapshot | None = field(default=None, init=False)
    durations: list[float] = field(default_factory=list, init=False)
    publications: list[dict[str, float | int]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Initialize service history and the immutable pre-experiment snapshot."""
        if not math.isfinite(self.measured_update_seconds) or self.measured_update_seconds <= 0:
            raise ValueError("initial measured update duration must be positive finite")
        if self.reserve_seconds < 0 or self.safety_factor < 1:
            raise ValueError("reserve must be nonnegative and safety factor at least one")
        self.durations.append(self.measured_update_seconds)
        self.published = CompletedSnapshot(self.initial_state, self.initial_version, 0.0, 0.0, 0.0)

    @property
    def estimated_service_seconds(self) -> float:
        """Conservative rolling service estimate, including slower recent learner calls."""
        return self.safety_factor * float(np.percentile(self.durations[-32:], 95))

    def can_start(self, now: float, deadline: float) -> bool:
        """Check available serialized GPU time after controller and telemetry work."""
        return (
            self.pending is None
            and now + self.estimated_service_seconds + self.reserve_seconds <= deadline
        )

    def complete(self, snapshot: CompletedSnapshot) -> None:
        """Queue a synchronized whole result; parameter quality is not an admission condition."""
        if self.pending is not None:
            raise RuntimeError("publish the preceding completed snapshot before starting another")
        duration = snapshot.completed_wall_time - snapshot.started_wall_time
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("completion must follow start by a finite positive duration")
        if snapshot.version < self.published.version:
            raise ValueError("library versions cannot go backward")
        self.durations.append(duration)
        self.pending = snapshot

    def publish(self, boundary_wall_time: float, simulation_time: float) -> CompletedSnapshot:
        """Expose a completed snapshot only at an explicitly supplied control boundary."""
        if self.pending is not None and self.pending.completed_wall_time <= boundary_wall_time:
            self.published = self.pending
            self.pending = None
            self.publications.append(
                {
                    "version": self.published.version,
                    "training_simulation_time": self.published.training_simulation_time,
                    "completed_wall_time": self.published.completed_wall_time,
                    "published_wall_time": boundary_wall_time,
                    "published_simulation_time": simulation_time,
                    "snapshot_age_seconds": simulation_time
                    - self.published.training_simulation_time,
                }
            )
        return self.published
