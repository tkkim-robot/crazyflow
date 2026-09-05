"""Separate task outcomes from observed paced online-learning service feasibility."""

from __future__ import annotations

import math
from typing import Any


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def assess_navigation_runtime_feasibility(summary: dict[str, Any]) -> dict[str, Any]:
    """Audit completed/available/used versions, exogenous opportunities and all active misses.

    No successful task trajectory can substitute for a finite online update actually used by a
    controller. This minimum observed-service criterion promises no update rate or real-time OS.
    Missing required accounting evidence fails conservatively rather than being inferred as success.
    """
    config = summary["config"]
    methods = summary["methods"]
    adaptive = methods["adaptive"]
    rows = [
        {
            **row,
            **{
                name: _number(row.get(name))
                for name in (
                    "time",
                    "scheduled_wall_seconds",
                    "started_wall_seconds",
                    "completed_wall_seconds",
                )
            },
        }
        for row in adaptive.get("publications_and_inputs", [])
    ]
    publications = [
        {
            **row,
            **{
                name: _number(row.get(name))
                for name in (
                    "completed_wall_time",
                    "published_wall_time",
                    "published_simulation_time",
                )
            },
        }
        for row in adaptive.get("snapshot_publications", [])
    ]
    opportunities = summary.get("schedule", {}).get("opportunities", [])
    initial_version = summary.get("initial_library_version")
    if initial_version is None and rows:
        initial_version = int(rows[0]["version_used"])
    launches = [row for row in rows if row.get("update_launched", False)]
    finite = [row for row in launches if row.get("finite", False)]
    allowed_count = sum(bool(value) for value in opportunities[: len(rows)])
    mask_valid = len(opportunities) >= len(rows) and all(
        not row.get("update_launched", False) or opportunities[index]
        for index, row in enumerate(rows)
    )
    advanced_publications = [
        publication
        for publication in publications
        if initial_version is not None and publication["version"] > initial_version
    ]
    verified_used_versions, invalid_publications = set(), 0
    valid_publications = []
    for publication in advanced_publications:
        version = publication["version"]
        completed = publication["completed_wall_time"]
        published = publication["published_wall_time"]
        ordering_valid = (
            math.isfinite(completed) and math.isfinite(published) and completed <= published + 1e-8
        )
        source_rows = [row for row in finite if row.get("completed_version") == version]
        source_valid = any(
            row.get("learner_completed_wall_seconds") is not None
            and abs(row["learner_completed_wall_seconds"] - completed) <= 1e-6
            for row in source_rows
        )
        uses = [row for row in rows if row["version_used"] == version]
        used_after_publication = any(
            row["time"] >= publication["published_simulation_time"] - 1e-8
            and row["started_wall_seconds"] >= published - 1e-8
            for row in uses
        )
        if not ordering_valid or not source_valid:
            invalid_publications += 1
        else:
            valid_publications.append(publication)
            if used_after_publication:
                verified_used_versions.add(int(version))
    invalid_uses = (
        sum(
            row["version_used"] < initial_version
            or (
                row["version_used"] > initial_version
                and not any(
                    publication["version"] == row["version_used"]
                    and row["time"] >= publication["published_simulation_time"] - 1e-8
                    and row["started_wall_seconds"] >= publication["published_wall_time"] - 1e-8
                    for publication in valid_publications
                )
            )
            for row in rows
        )
        if initial_version is not None
        else 0
    )
    reported_misses = int(adaptive.get("service_exceeds_nominal_period_count", -1))
    recorded_misses = sum(bool(row.get("missed_deadline", False)) for row in rows)
    world = summary.get("world", {}).get("config", {})
    period = _number(world.get("dt")) * _number(world.get("control_interval_steps"))
    chronology = (
        bool(rows)
        and math.isfinite(period)
        and period > 0
        and all(
            all(
                math.isfinite(row[name])
                for name in (
                    "time",
                    "scheduled_wall_seconds",
                    "started_wall_seconds",
                    "completed_wall_seconds",
                )
            )
            and row["scheduled_wall_seconds"] <= row["started_wall_seconds"] + 1e-8
            and row["started_wall_seconds"] <= row["completed_wall_seconds"]
            for row in rows
        )
    )
    chronology = chronology and all(
        left["time"] < right["time"]
        and left["scheduled_wall_seconds"] < right["scheduled_wall_seconds"]
        and left["started_wall_seconds"] <= right["started_wall_seconds"]
        for left, right in zip(rows[:-1], rows[1:], strict=True)
    )
    recomputed_flags = (
        [row["completed_wall_seconds"] > row["scheduled_wall_seconds"] + period for row in rows]
        if chronology
        else []
    )
    flags_match_wall = chronology and all(
        bool(row.get("missed_deadline", False)) == missed
        for row, missed in zip(rows, recomputed_flags, strict=True)
    )
    versions_monotonic = all(
        left["version_used"] <= right["version_used"]
        for left, right in zip(rows[:-1], rows[1:], strict=True)
    )
    accounting = (
        bool(rows)
        and len(rows) == adaptive.get("active_controls")
        and len(finite) == adaptive.get("finite_updates")
        and len(launches) == adaptive.get("learner_service", {}).get("count")
        and reported_misses == recorded_misses
        and invalid_publications == 0
        and invalid_uses == 0
        and mask_valid
        and flags_match_wall
        and versions_monotonic
    )
    checks = {
        "measured_paced_execution": summary.get("execution_mode") == "budgeted",
        "learning_enabled": bool(config.get("enable_learning", False)),
        "allowed_opportunities_while_active": allowed_count > 0,
        "at_least_one_finite_completed_update": len(finite) > 0,
        "advanced_snapshot_published": len(advanced_publications) > 0,
        "published_advanced_snapshot_used_by_controller": bool(verified_used_versions),
        "all_launched_updates_finite": len(launches) == len(finite),
        "zero_adaptive_deadline_misses": (
            reported_misses == 0
            and recorded_misses == 0
            and chronology
            and not any(recomputed_flags)
        ),
        "complete_consistent_accounting": accounting,
    }
    task_checks = {}
    for method, record in methods.items():
        audit = record.get("execution_audit", {})
        clearance = record.get("minimum_inflated_clearance_m")
        task_checks[method] = {
            "all_waypoints_completed": record.get("termination") == "completed",
            "no_physical_collision": record.get("physical_collision") is False,
            "positive_shell_clearance": clearance is not None
            and math.isfinite(clearance)
            and clearance > 0,
            "zero_degraded_controls": record.get("degraded_controls") == 0,
            "physical_operational_nodes_pass": audit.get("all_actual_physical_nodes_pass") is True,
            "actuator_limits_pass": audit.get("applied_motor_limit_violating_controls") == 0,
        }
    runtime = all(checks.values())
    tasks = {method: all(values.values()) for method, values in task_checks.items()}
    paired_misses = {
        method: record.get("service_exceeds_nominal_period_count")
        for method, record in methods.items()
    }
    return {
        "scope": (
            "minimum observed paced serialized online-learning feasibility; requires a finite "
            "published advance actually used and zero active adaptive misses; "
            "no guaranteed update frequency, continuous-time safety or OS real-time guarantee"
        ),
        "adaptive_online_runtime_feasible": runtime,
        "paired_zero_miss_online_runtime_feasible": runtime
        and all(value == 0 for value in paired_misses.values()),
        "task_success": tasks,
        "adaptive_task_and_runtime_success": runtime and tasks.get("adaptive", False),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "task_checks": task_checks,
        "initial_used_library_version": initial_version,
        "initial_version_source": (
            "checkpoint_summary"
            if "initial_library_version" in summary
            else "first_recorded_control"
        ),
        "allowed_opportunities_during_active_controls": allowed_count,
        "launched_updates": len(launches),
        "finite_completed_updates": len(finite),
        "advanced_publications": len(advanced_publications),
        "verified_advanced_versions_used": sorted(verified_used_versions),
        "invalid_publication_records": invalid_publications,
        "controls_using_unavailable_advanced_versions": invalid_uses,
        "finite_updates_per_allowed_opportunity": len(finite) / allowed_count
        if allowed_count
        else None,
        "deadline_misses_by_method": paired_misses,
        "adaptive_deadline_misses_recomputed_from_wall": sum(recomputed_flags)
        if chronology
        else None,
        "wall_chronology_valid": chronology,
        "used_versions_nondecreasing": versions_monotonic,
        "zero_updates_are_runtime_success": False,
    }
