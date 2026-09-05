from __future__ import annotations

from typing import Any

import pytest

from crazyflow.safety.da_plcbf.runtime_feasibility import assess_navigation_runtime_feasibility


def _summary() -> dict[str, Any]:
    rows = [
        {
            "time": index * 0.04,
            "scheduled_wall_seconds": index * 0.04,
            "started_wall_seconds": index * 0.04 + 0.0001,
            "completed_wall_seconds": index * 0.04 + 0.03,
            "version_used": 10 if index == 0 else 11,
            "update_launched": index == 0,
            "finite": index == 0,
            "completed_version": 11 if index == 0 else None,
            "learner_started_wall_seconds": 0.015 if index == 0 else None,
            "learner_completed_wall_seconds": 0.025 if index == 0 else None,
            "missed_deadline": False,
        }
        for index in range(3)
    ]
    task = {
        "termination": "completed",
        "physical_collision": False,
        "minimum_inflated_clearance_m": 0.1,
        "degraded_controls": 0,
        "execution_audit": {
            "all_actual_physical_nodes_pass": True,
            "applied_motor_limit_violating_controls": 0,
        },
        "service_exceeds_nominal_period_count": 0,
    }
    return {
        "config": {"enable_learning": True},
        "world": {"config": {"dt": 0.02, "control_interval_steps": 2}},
        "execution_mode": "budgeted",
        "schedule": {"opportunities": [True, False, False]},
        "methods": {
            "fixed": task.copy(),
            "adaptive": {
                **task,
                "active_controls": 3,
                "finite_updates": 1,
                "learner_service": {"count": 1},
                "publications_and_inputs": rows,
                "snapshot_publications": [
                    {
                        "version": 11,
                        "completed_wall_time": 0.025,
                        "published_wall_time": 0.0401,
                        "published_simulation_time": 0.04,
                    }
                ],
            },
        },
    }


def test_safe_task_with_zero_updates_is_not_online_runtime_success() -> None:
    summary = _summary()
    adaptive = summary["methods"]["adaptive"]
    adaptive.update(finite_updates=0, learner_service={"count": 0}, snapshot_publications=[])
    for row in adaptive["publications_and_inputs"]:
        row.update(version_used=10, update_launched=False, finite=False, completed_version=None)
    result = assess_navigation_runtime_feasibility(summary)
    assert result["task_success"]["adaptive"]
    assert not result["adaptive_online_runtime_feasible"]
    assert not result["adaptive_task_and_runtime_success"]
    assert "at_least_one_finite_completed_update" in result["failed_checks"]


def test_completed_published_and_used_finite_advance_can_pass_observed_runtime() -> None:
    result = assess_navigation_runtime_feasibility(_summary())
    assert result["adaptive_online_runtime_feasible"]
    assert result["paired_zero_miss_online_runtime_feasible"]
    assert result["verified_advanced_versions_used"] == [11]


def test_terminal_publication_without_controller_use_does_not_count() -> None:
    summary = _summary()
    adaptive = summary["methods"]["adaptive"]
    for row in adaptive["publications_and_inputs"]:
        row["version_used"] = 10
    result = assess_navigation_runtime_feasibility(summary)
    assert result["advanced_publications"] == 1
    assert not result["adaptive_online_runtime_feasible"]


@pytest.mark.parametrize(
    "failure", ["miss", "not_paced", "forbidden_mask", "early_publication", "unpublished_use"]
)
def test_missing_or_inconsistent_runtime_evidence_fails(failure: str) -> None:
    summary = _summary()
    adaptive = summary["methods"]["adaptive"]
    if failure == "miss":
        adaptive["service_exceeds_nominal_period_count"] = 1
        adaptive["publications_and_inputs"][2]["missed_deadline"] = True
    elif failure == "not_paced":
        summary["execution_mode"] = "deterministic"
    elif failure == "forbidden_mask":
        summary["schedule"]["opportunities"][0] = False
    elif failure == "early_publication":
        adaptive["snapshot_publications"][0]["published_wall_time"] = 0.01
    else:
        adaptive["publications_and_inputs"][2]["version_used"] = 12
    assert not assess_navigation_runtime_feasibility(summary)["adaptive_online_runtime_feasible"]


def test_task_failure_and_fixed_deadline_miss_remain_separate() -> None:
    summary = _summary()
    summary["methods"]["adaptive"]["termination"] = "timeout"
    summary["methods"]["fixed"]["service_exceeds_nominal_period_count"] = 1
    result = assess_navigation_runtime_feasibility(summary)
    assert result["adaptive_online_runtime_feasible"]
    assert not result["paired_zero_miss_online_runtime_feasible"]
    assert not result["adaptive_task_and_runtime_success"]


@pytest.mark.parametrize(
    "tamper", ["hidden_miss", "missing_completion", "backward_version", "missing_period"]
)
def test_missing_or_tampered_chronology_cannot_pass(tamper: str) -> None:
    summary = _summary()
    rows = summary["methods"]["adaptive"]["publications_and_inputs"]
    if tamper == "hidden_miss":
        rows[2]["completed_wall_seconds"] = 0.13  # Deadline .12; flag/count left falsely zero.
    elif tamper == "missing_completion":
        rows[2].pop("completed_wall_seconds")
    elif tamper == "backward_version":
        rows[2]["version_used"] = 10
    else:
        summary.pop("world")
    result = assess_navigation_runtime_feasibility(summary)
    assert not result["adaptive_online_runtime_feasible"]
    assert not result["checks"]["complete_consistent_accounting"]
