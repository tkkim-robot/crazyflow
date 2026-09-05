from __future__ import annotations

from copy import deepcopy

from benchmark.da_plcbf_case_episode import snapshot_availability_at_arrival


def test_arrival_audit_counts_published_control_inputs_not_unpublished_service() -> None:
    # A service completes before the4.03s encounter, but the result is first published at
    # 4.04s. It must not be credited to the last control actually used before the encounter.
    method = {
        "termination_time_seconds": 12.0,
        "publications_and_inputs": [
            {
                "time": 4.0,
                "version_used": 100,
                "finite": True,
                "learner_completed_wall_seconds": 4.02,
                "missed_deadline": False,
            },
            {
                "time": 4.04,
                "version_used": 101,
                "finite": False,
                "learner_completed_wall_seconds": None,
                "missed_deadline": False,
            },
        ],
        "snapshot_publications": [
            {
                "version": 101,
                "published_simulation_time": 4.04,
                "completed_wall_time": 4.02,
                "published_wall_time": 4.0401,
            }
        ],
    }
    summary = {
        "initial_library_version": 100,
        "execution_mode": "budgeted",
        "methods": {"adaptive": method},
    }
    before = snapshot_availability_at_arrival(summary, 4.03)["adaptive"]
    assert before["completed_updates_actually_used_before_arrival"] == 0
    assert before["finite_services_completed_before_arrival_wall_clock"] == 1
    assert before["actual_snapshot_publications_before_arrival"] == []
    assert before["every_publication_follows_service_completion"]
    at_boundary = snapshot_availability_at_arrival(summary, 4.04)["adaptive"]
    assert at_boundary["completed_updates_actually_used_before_arrival"] == 1
    assert at_boundary["version_used_at_last_boundary_before_arrival"] == 101
    assert len(at_boundary["actual_snapshot_publications_before_arrival"]) == 1
    deterministic = deepcopy(summary)
    deterministic["execution_mode"] = "deterministic"
    result = snapshot_availability_at_arrival(deterministic, 4.03)["adaptive"]
    assert result["finite_services_completed_before_arrival_wall_clock"] is None
    assert result["actual_snapshot_publications_before_arrival"] is None
    assert result["deadline_misses_before_arrival"] is None
