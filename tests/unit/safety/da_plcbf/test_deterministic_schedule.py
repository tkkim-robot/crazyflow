"""Update opportunities replay independently of measured machine load."""

import pytest

from crazyflow.safety.da_plcbf.deterministic_schedule import DeterministicUpdateSchedule


def test_recorded_opportunities_preserve_skips() -> None:
    rows = [
        {"simulation_time": index * 0.04, "learner_seconds": value}
        for index, value in enumerate((0.012, 0.0, 0.011, 0.0))
    ]
    schedule = DeterministicUpdateSchedule.from_service_records(rows)
    assert schedule.opportunities == (True, False, True, False)
    changed = [{**row, "learner_seconds": float(row["learner_seconds"]) * 100} for row in rows]
    assert DeterministicUpdateSchedule.from_service_records(changed) == schedule
    assert schedule.metadata()["deployment_budget_claim"] is False


def test_periodic_schedule_is_exogenous_and_starts_at_declared_boundary() -> None:
    schedule = DeterministicUpdateSchedule.periodic(7, first=2, every=2)
    assert schedule.opportunities == (False, False, True, False, True, False, True)


@pytest.mark.parametrize("duration", [-1.0, float("nan"), float("inf")])
def test_replay_rejects_invalid_service_duration(duration: float) -> None:
    with pytest.raises(ValueError):
        DeterministicUpdateSchedule.from_service_records(
            [{"simulation_time": 0.0, "learner_seconds": duration}]
        )


def test_replay_rejects_unordered_publication_boundaries() -> None:
    with pytest.raises(ValueError):
        DeterministicUpdateSchedule.from_service_records(
            [
                {"simulation_time": 0.04, "learner_seconds": 0.01},
                {"simulation_time": 0.0, "learner_seconds": 0.01},
            ]
        )
