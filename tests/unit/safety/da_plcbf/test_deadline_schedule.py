"""Availability and nonpreemptive budgeting, independent of machine timing noise."""

from crazyflow.safety.da_plcbf.deadline_schedule import BoundarySnapshotScheduler, CompletedSnapshot


def test_snapshot_waits_for_completion_and_control_boundary() -> None:
    scheduler = BoundarySnapshotScheduler("old", 12, 0.012)
    assert scheduler.can_start(0.015, 0.040)
    scheduler.complete(CompletedSnapshot("whole new state", 13, 0.0, 0.015, 0.029))
    assert not scheduler.can_start(0.020, 0.080)
    assert scheduler.publish(0.028, 0.0).state == "old"
    assert scheduler.publish(0.040, 0.040).state == "whole new state"
    assert scheduler.publications[-1]["snapshot_age_seconds"] == 0.040


def test_slow_calls_reduce_future_update_budget_without_rejecting_updates() -> None:
    scheduler = BoundarySnapshotScheduler("old", 0, 0.012)
    assert not scheduler.can_start(0.015, 0.020)
    scheduler.complete(CompletedSnapshot("finite", 1, 0.0, 0.010, 0.035))
    scheduler.publish(0.040, 0.040)
    assert scheduler.published.version == 1
    assert not scheduler.can_start(0.055, 0.080)
    assert scheduler.can_start(0.055, 0.100)
