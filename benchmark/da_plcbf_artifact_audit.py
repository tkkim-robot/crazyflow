"""Audit saved competent-experiment execution with NumPy, without importing JAX.

Reads the original comparison JSON/NPZ, dense plant states, and the saved reference's scenario
and arena-clearance configuration. Writes a separate execution_audit.json; original evidence is
never modified. Physical node limits, derivative residuals, collision/goal outcomes, and measured
service chronology are reported separately. No controller, learner, or simulator is rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _first(mask: np.ndarray, times: np.ndarray) -> float | None:
    indices = np.flatnonzero(mask)
    return float(times[indices[0]]) if indices.size else None


def _node_limits(
    states: np.ndarray, times: np.ndarray, scenario: dict[str, Any], arena_clearance: float
) -> dict[str, Any]:
    """Native physical-unit margins; strict negatives are retained without tolerance clipping."""
    quaternion = states[:, 3:7]
    norms = np.linalg.norm(quaternion, axis=1)
    normalized = quaternion / np.where(norms > 0, norms, np.nan)[:, None]
    body_z_vertical = 1 - 2 * (normalized[:, 0] ** 2 + normalized[:, 1] ** 2)
    tilt = np.arccos(np.clip(body_z_vertical, -1, 1))
    speed = np.linalg.norm(states[:, 7:10], axis=1)
    rate = np.linalg.norm(states[:, 10:13], axis=1)
    lower = np.asarray(scenario["arena_lower"]) + arena_clearance
    upper = np.asarray(scenario["arena_upper"]) - arena_clearance
    margins = {
        "speed_mps": float(scenario["speed_max"]) - speed,
        "angular_rate_rps": float(scenario["angular_rate_max"]) - rate,
        "tilt_radians": float(scenario["tilt_max_radians"]) - tilt,
        "arena_center_m": np.minimum(states[:, :3] - lower, upper - states[:, :3]).min(axis=1),
    }
    invalid = ~np.all(np.isfinite(states), axis=1) | ~np.isfinite(norms) | (norms == 0)
    violations = np.stack([~np.isfinite(value) | (value < 0) for value in margins.values()])
    any_violation = np.any(violations, axis=0) | invalid
    first_index = np.flatnonzero(any_violation)
    return {
        "node_count": len(states),
        "first_time_seconds": float(times[0]),
        "last_time_seconds": float(times[-1]),
        "all_physical_node_margins_nonnegative": not bool(np.any(any_violation)),
        "first_physical_node_violation_time_seconds": _first(any_violation, times),
        "first_violation_margin_values": None
        if not first_index.size
        else {name: values[first_index[0]] for name, values in margins.items()},
        "first_invalid_state_time_seconds": _first(invalid, times),
        "maximum_quaternion_norm_error": np.max(np.abs(norms - 1)),
        "maximum_speed_mps": np.max(speed),
        "maximum_angular_rate_rps": np.max(rate),
        "maximum_tilt_radians": np.max(tilt),
        "limits": {
            name: {
                "minimum_margin": np.min(values),
                "negative_node_count": int(np.count_nonzero(values < 0)),
                "first_negative_node_time_seconds": _first(values < 0, times),
                "worst_node_time_seconds": float(times[int(np.argmin(values))]),
            }
            for name, values in margins.items()
        },
        "scope": (
            "physical states at saved integration nodes, including terminal state; "
            "arena margin is center position against bounds with explicit arena clearance; "
            "strict negative margins are reported without a retrospective tolerance"
        ),
    }


def _collision(states: np.ndarray, times: np.ndarray, scenario: dict[str, Any]) -> dict[str, Any]:
    active = np.asarray(scenario["obstacle_mask"], dtype=bool)
    if not np.any(active):
        return {
            "collision_constraint_active": False,
            "minimum_physical_clearance_m": None,
            "minimum_inflated_clearance_m": None,
            "physical_collision": False,
            "shell_clear": True,
        }
    centers = (
        np.asarray(scenario["obstacle_initial_centers"])[None, active]
        + times[:, None, None] * np.asarray(scenario["obstacle_velocities"])[None, active]
    )
    relative = states[:, None, :3] - centers
    delta = np.diff(relative, axis=0)
    denominator = np.sum(delta * delta, axis=-1)
    fraction = np.clip(
        -np.sum(relative[:-1] * delta, axis=-1) / np.where(denominator > 0, denominator, 1), 0, 1
    )
    clearance = (
        np.linalg.norm(relative[:-1] + fraction[..., None] * delta, axis=-1)
        - np.asarray(scenario["obstacle_radii"])[None, active]
        - float(scenario["ego_radius"])
    ).min(axis=1)
    shell = clearance - float(scenario["obstacle_clearance"])
    return {
        "collision_constraint_active": True,
        "minimum_physical_clearance_m": np.min(clearance),
        "minimum_inflated_clearance_m": np.min(shell),
        "physical_collision": bool(np.any(clearance <= 0)),
        "shell_clear": bool(np.all(shell > 0)),
        "first_colliding_interval_start_seconds": _first(clearance <= 0, times[:-1]),
        "first_shell_violating_interval_start_seconds": _first(shell < 0, times[:-1]),
        "scope": "exact relative linear-segment sweep of saved integration nodes",
    }


def _chronology(method: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Cross-check recorded completion/release/version claims; do not infer unlogged GPU work."""
    services = method["runtime_services"]
    publications = method["snapshot_publications"]
    period = config["dt"] * config["control_interval_steps"]
    event = config["event_time_seconds"]
    expected_count = round((config["duration_seconds"] - event) / period)
    issues = []
    computed_misses = 0
    service_seconds = []
    for index, service in enumerate(services):
        scheduled = float(service["scheduled_wall_time"])
        started = float(service["started_wall_time"])
        completed = float(service["completed_wall_time"])
        when = float(service["simulation_time"])
        finite = np.all(np.isfinite([scheduled, started, completed, when]))
        if (
            not finite
            or completed < started
            or (config["schedule"] == "budgeted" and started + 1e-8 < scheduled)
        ):
            issues.append(f"service {index}: invalid release/start/completion ordering")
        if not np.isclose(when, event + index * period, rtol=0, atol=1e-7):
            issues.append(f"service {index}: simulation release differs from configured grid")
        if config["schedule"] == "budgeted":
            missed = completed > scheduled + period
            computed_misses += int(missed)
            if missed != bool(service["missed_deadline"]):
                issues.append(f"service {index}: deadline flag differs from recorded completion")
        service_seconds.append(completed - started)
        eligible_publications = [
            publication
            for publication in publications
            if publication["published_wall_time"] <= started + 1e-8
            and publication["published_simulation_time"] <= when + 1e-8
        ]
        available_version = (
            eligible_publications[-1]["version"]
            if eligible_publications
            else method["initial_library_version"]
        )
        if service["snapshot_version"] != available_version:
            issues.append(f"service {index}: controller version was not the published snapshot")
    previous_version = method["initial_library_version"]
    previous_publication_wall = -np.inf
    for index, publication in enumerate(publications):
        completed = publication["completed_wall_time"]
        published = publication["published_wall_time"]
        training_time = publication["training_simulation_time"]
        publication_time = publication["published_simulation_time"]
        if not np.all(np.isfinite([completed, published, training_time, publication_time])):
            issues.append(f"publication {index}: nonfinite timestamps")
            continue
        if completed > published + 1e-8:
            issues.append(f"publication {index}: snapshot exposed before recorded completion")
        if publication_time <= training_time:
            issues.append(f"publication {index}: snapshot published before a later boundary")
        if publication["version"] < previous_version or published < previous_publication_wall:
            issues.append(f"publication {index}: version/publication chronology moved backward")
        if config["schedule"] == "budgeted" and published + 1e-8 < publication_time - event:
            issues.append(f"publication {index}: actual wall clock preceded its release boundary")
        previous_version = publication["version"]
        previous_publication_wall = published
    if len(services) != expected_count:
        issues.append("service count differs from configured post-event control count")
    if config["schedule"] == "budgeted" and computed_misses != method["deadline_misses"]:
        issues.append("summary deadline count differs from recorded service completions")
    version_increment = method["final_library_version"] - method["initial_library_version"]
    if version_increment != method["finite_updates"]:
        issues.append("final published version increment differs from finite update count")
    return {
        "recorded_chronology_consistent": not issues,
        "issues": issues,
        "service_count": len(services),
        "publication_count": len(publications),
        "recorded_completion_deadline_misses": computed_misses
        if config["schedule"] == "budgeted"
        else None,
        "maximum_logged_service_seconds": max(service_seconds, default=None),
        "median_logged_service_seconds": np.median(service_seconds) if service_seconds else None,
        "scope": (
            "consistency of saved service/publication timestamps only; cannot certify completion "
            "of unlogged asynchronous kernels, latency-free actuation, or hard real-time behavior"
        ),
    }


def audit_directory(directory: Path) -> Path:
    """Write one append-only audit alongside a complete saved competent experiment."""
    directory = directory.resolve()
    output = directory / "execution_audit.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    paths = {
        "comparison": directory / "competent_comparison.json",
        "dense_states": directory / "dense_plant_states.npz",
        "reference": directory / "feasibility_reference.json",
        "trace": directory / "competent_comparison.npz",
    }
    metadata = json.loads(paths["comparison"].read_text())
    summary = metadata["summary"]
    config = summary["config"]
    reference = json.loads(paths["reference"].read_text())
    scenario = reference["scenario"]
    arena_clearance = reference["config"]["arena_clearance_m"]
    if not np.isclose(config["dt"], scenario["dt"], rtol=0, atol=1e-12):
        raise ValueError("saved reference and comparison have different integration steps")
    expected_nodes = round(config["duration_seconds"] / config["dt"]) + 1
    times = np.arange(expected_nodes) * config["dt"]
    event = config["event_time_seconds"]
    report: dict[str, Any] = {
        "audit": "saved physical nodes, execution scope, and recorded service chronology",
        "sources": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in paths.items()
        },
        "physical_limits_source": "saved feasibility_reference scenario and arena-clearance config",
        "summary_scope": (
            "original safe_goal_success means positive inflated obstacle clearance and final "
            "goal distance <0.5 m; renamed collision_clear_goal here; it does not include "
            "physical operational limits, zero degraded execution, or timing success"
        ),
        "loss_scope": (
            "finite/low learner losses and repertoire competency are not safety certificates; "
            "this audit never uses learning loss as an execution acceptance criterion"
        ),
        "event_time_seconds": event,
        "methods": {},
    }
    with (
        np.load(paths["dense_states"], allow_pickle=False) as dense,
        np.load(paths["trace"], allow_pickle=False) as trace,
    ):
        control_times = trace["time_seconds"]
        for name, method in summary["methods"].items():
            states = np.asarray(dense[name], dtype=np.float64)
            if states.shape != (expected_nodes, 13):
                raise ValueError(f"{name}: dense plant shape does not match configured duration")
            physical = _node_limits(states, times, scenario, arena_clearance)
            collision = _collision(states, times, scenario)
            goal_distance = float(
                np.linalg.norm(states[-1, :3] - np.asarray(scenario["goal_position"]))
            )
            collision_clear_goal = collision["shell_clear"] and goal_distance < 0.5
            modes = trace[f"{name}_execution_mode"]
            degraded = trace[f"{name}_degraded"].astype(bool)
            residuals = trace[f"{name}_operational_residuals"]
            minimum_residual = np.min(residuals, axis=1)
            method_report = {
                "physical_node_limits": physical,
                "shared_prefix_physical_node_limits": _node_limits(
                    states[times <= event], times[times <= event], scenario, arena_clearance
                ),
                "post_event_physical_node_limits": _node_limits(
                    states[times >= event], times[times >= event], scenario, arena_clearance
                ),
                "collision": collision,
                "final_goal_distance_m": goal_distance,
                "collision_clear_goal": collision_clear_goal,
                "original_safe_goal_success_matches_collision_clear_goal": bool(
                    method["safe_goal_success"]
                )
                == collision_clear_goal,
                "physical_node_limits_and_collision_clear_goal": (
                    physical["all_physical_node_margins_nonnegative"] and collision_clear_goal
                ),
                "recorded_initial_analytic_residuals": {
                    "minimum": np.min(minimum_residual),
                    "first_negative_boundary_seconds": _first(minimum_residual < 0, control_times),
                    "negative_boundary_count": int(np.sum(minimum_residual < 0)),
                    "scope": (
                        "initial-boundary applied-action analytic residuals; not the minimum "
                        "across all held substeps; a negative derivative residual alone is not "
                        "a physical state-limit violation"
                    ),
                },
                "execution": {
                    "degraded_count": int(np.sum(degraded)),
                    "first_degraded_boundary_seconds": _first(degraded, control_times),
                    "first_emergency_boundary_seconds": _first(modes == 2, control_times),
                    "shared_prefix_emergency_count": int(
                        np.sum((modes == 2) & (control_times < event))
                    ),
                    "post_event_emergency_count": int(
                        np.sum((modes == 2) & (control_times >= event))
                    ),
                    "midpoint_count": int(np.sum(modes == 3)),
                },
                "chronology": _chronology(method, config),
            }
            report["methods"][name] = method_report
    with output.open("x") as stream:
        json.dump(_jsonable(report), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args()
    for directory in args.directories:
        output = audit_directory(directory)
        print(output, flush=True)


if __name__ == "__main__":
    main()
