"""Audit isolated navigation events against matching unchanged worlds without replaying control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _segment_clearances(states: np.ndarray, world: dict) -> np.ndarray:
    """Use the experiment's relative segment convention at each actual integration interval."""
    config = world["config"]
    if any(
        np.linalg.norm(event["half_extents"]) > config["ego_radius"]
        for event in config["payload_events"]
    ):
        raise ValueError("this audit requires every centered payload to fit the bare ego radius")
    times = np.arange(len(states)) * config["dt"]
    mean = np.asarray(world["obstacle_mean_centers"])
    amplitudes = np.asarray(world["obstacle_amplitudes"])
    rates = np.asarray(world["obstacle_angular_frequencies"])
    phases = np.asarray(world["obstacle_phases"])
    centers = mean + amplitudes * np.sin(
        times[:, None, None] * rates[None, :, None] + phases[None, :, None]
    )
    relative = states[:, None, :3] - centers
    start, delta = relative[:-1], np.diff(relative, axis=0)
    fractions = np.clip(
        -np.sum(start * delta, axis=-1) / np.maximum(np.sum(delta**2, axis=-1), 1e-30), 0, 1
    )
    radius = np.asarray(world["obstacle_radii"]) + config["ego_radius"]
    return np.min(np.linalg.norm(start + fractions[..., None] * delta, axis=-1) - radius, axis=1)


def _compare_arrays(left: np.ndarray, right: np.ndarray) -> dict:
    equal = (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )
    delta = None
    if left.shape == right.shape:
        finite = np.isfinite(left) & np.isfinite(right)
        if np.any(finite):
            delta = float(np.max(np.abs(left[finite].astype(float) - right[finite].astype(float))))
    return {"bitwise_equal": bool(equal), "max_absolute_finite_difference": delta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite composition audit evidence")
    protocol_path = args.root / "ISOLATED_DISTURBANCE_EXECUTION_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    original_protocol_path = args.root / "CAMPAIGN_PROTOCOL.json"
    original_protocol = json.loads(original_protocol_path.read_text())
    runner_name = "crazyflow/safety/da_plcbf/navigation_experiment.py"
    if (
        hashlib.sha256(original_protocol_path.read_bytes()).hexdigest()
        != protocol["original_campaign_protocol_sha256"]
    ):
        raise ValueError("original campaign declaration changed")
    results = []
    for condition in protocol["conditions"]:
        event_spec = protocol["condition_events"][condition]
        event_times = sorted(
            event["time_seconds"]
            for key in ("wind_events", "payload_events")
            for event in event_spec[key]
        )
        first_event = event_times[0]
        for seed in protocol["heldout_world_seeds"]:
            directory = args.root / f"isolated-{condition}" / f"{condition}-seed{seed}"
            candidates = list(
                args.root.glob(f"heldout-*/unchanged-seed{seed}/navigation_comparison.json")
            )
            if len(candidates) != 1:
                raise ValueError(f"need one unchanged baseline for seed{seed}")
            baseline = candidates[0].parent
            meta = json.loads((directory / "navigation_comparison.json").read_text())
            summary = meta["summary"]
            original = json.loads(candidates[0].read_text())["summary"]
            original_source = json.loads((baseline / "SOURCE_SHA256.json").read_text())
            if (
                original["checkpoint_sha256"] != original_protocol["checkpoint_sha256"]
                or original["config"] != protocol["experiment_config"]
                or original_source.get(runner_name) != original_protocol["files"][runner_name]
                or any(
                    original_protocol["files"].get(name) != digest
                    for name, digest in original_source.items()
                )
            ):
                raise ValueError(f"unchanged baseline config/candidate/source mismatch: {baseline}")
            if (
                summary["checkpoint_sha256"] != protocol["checkpoint_sha256"]
                or summary["config"] != protocol["experiment_config"]
            ):
                raise ValueError(f"candidate or config changed: {directory}")
            actual_world, reference_world = summary["world"], original["world"]
            if (
                reference_world["config"]["seed"] != seed
                or reference_world["config"]["wind_events"]
                or reference_world["config"]["payload_events"]
            ):
                raise ValueError(f"unchanged baseline has unexpected seed or events: {baseline}")
            if {k: v for k, v in actual_world.items() if k != "config"} != {
                k: v for k, v in reference_world.items() if k != "config"
            }:
                raise ValueError(f"geometry changed across compositional conditions: {directory}")
            expected_config = {**reference_world["config"], **event_spec}
            if actual_world["config"] != expected_config:
                raise ValueError(f"events or world settings changed: {directory}")
            source = json.loads((directory / "SOURCE_SHA256.json").read_text())
            if source.get(runner_name) != protocol["files"][runner_name] or any(
                protocol["files"].get(name) != digest for name, digest in source.items()
            ):
                raise ValueError(
                    f"recorded source differs from locked execution protocol: {directory}"
                )
            row = {
                "condition": condition,
                "seed": seed,
                "directory": str(directory),
                "unchanged_baseline": str(baseline),
                "first_event_seconds": first_event,
                "geometry_and_settings_exact": True,
                "methods": {},
            }
            with (
                np.load(directory / "navigation_comparison.npz", allow_pickle=False) as trace,
                np.load(baseline / "navigation_comparison.npz", allow_pickle=False) as reference,
                np.load(directory / "dense_plant_states.npz", allow_pickle=False) as dense,
                np.load(baseline / "dense_plant_states.npz", allow_pickle=False) as reference_dense,
                np.load(directory / "raw_diagnostics.npz", allow_pickle=False) as raw,
            ):
                time = trace["time_seconds"]
                if not np.array_equal(time, reference["time_seconds"]):
                    raise ValueError("recording cadence changed")
                before = time < first_event - 1e-9
                for method in ("fixed", "adaptive"):
                    comparisons = {
                        field: _compare_arrays(
                            trace[f"{method}_{field}"][before],
                            reference[f"{method}_{field}"][before],
                        )
                        for field in (
                            "full_state",
                            "applied_wrench",
                            "nominal_wrench",
                            "goal_position",
                            "selected_policy",
                            "library_version",
                            "cumulative_gradient_steps",
                            "parameter_update_norm",
                            "descriptors",
                            "estimated_wind",
                            "recorded_control_valid",
                        )
                    }
                    active = trace[f"{method}_recorded_control_valid"]
                    degraded = trace[f"{method}_degraded"] & active
                    actual_wind = raw[f"{method}_actual_wind"]
                    expected_wind = np.zeros_like(actual_wind)
                    for event in event_spec["wind_events"]:
                        expected_wind[time >= event["time_seconds"] - 1e-9] = event["velocity"]
                    actual_mass = raw[f"{method}_actual_mass"]
                    expected_mass = np.full_like(actual_mass, actual_mass[0])
                    actual_inertia = raw[f"{method}_actual_inertia"]
                    expected_inertia = np.broadcast_to(
                        actual_inertia[0], actual_inertia.shape
                    ).copy()
                    for event in event_spec["payload_events"]:
                        after = time >= event["time_seconds"] - 1e-9
                        added_mass = actual_mass[0] * event["mass_fraction"]
                        expected_mass[after] += added_mass
                        side_squared = (2 * np.asarray(event["half_extents"])) ** 2
                        inertia_delta = added_mass * (np.sum(side_squared) - side_squared) / 12
                        expected_inertia[after] += np.diag(inertia_delta)
                    plant_events_match = bool(
                        np.array_equal(actual_wind[active], expected_wind[active])
                        and np.allclose(
                            actual_mass[active], expected_mass[active], rtol=2e-7, atol=0
                        )
                        and np.allclose(
                            actual_inertia[active], expected_inertia[active], rtol=2e-7, atol=0
                        )
                    )
                    if not plant_events_match:
                        raise ValueError(f"recorded plant events differ: {directory}/{method}")
                    states = dense[method]
                    first_event_node = round(first_event / actual_world["config"]["dt"])
                    comparisons["dense_plant_states_through_event_boundary"] = _compare_arrays(
                        states[: first_event_node + 1],
                        reference_dense[method][: first_event_node + 1],
                    )
                    physical = _segment_clearances(states, actual_world)
                    shell = physical - actual_world["config"]["obstacle_clearance"]
                    interval_starts = np.arange(len(physical)) * actual_world["config"]["dt"]
                    intervals = []
                    boundaries = [0.0, *event_times, actual_world["config"]["duration_seconds"]]
                    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
                        controls = active & (time >= start - 1e-9) & (time < stop - 1e-9)
                        segments = (interval_starts >= start - 1e-9) & (
                            interval_starts < stop - 1e-9
                        )
                        intervals.append(
                            {
                                "start_seconds": start,
                                "stop_seconds": stop,
                                "executed_controls": int(np.sum(controls)),
                                "degraded_controls": int(np.sum(degraded & controls)),
                                "minimum_physical_clearance_m": float(np.min(physical[segments]))
                                if np.any(segments)
                                else None,
                                "minimum_shell_clearance_m": float(np.min(shell[segments]))
                                if np.any(segments)
                                else None,
                            }
                        )
                    bad_shell = np.flatnonzero(shell <= 0)
                    bad_control = np.flatnonzero(degraded)
                    measured = summary["methods"][method]
                    if not np.isclose(
                        np.min(physical),
                        measured["minimum_physical_clearance_m"],
                        atol=1e-10,
                        rtol=0,
                    ):
                        raise ValueError(
                            f"clearance audit disagrees with saved result: {directory}/{method}"
                        )
                    row["methods"][method] = {
                        "pre_event_bitwise_equal": all(
                            value["bitwise_equal"] for value in comparisons.values()
                        ),
                        "pre_event_arrays": comparisons,
                        "recorded_plant_events_match": plant_events_match,
                        "plant_event_check_scope": (
                            "Actual active rows: exact prescribed wind; centered rigid payload "
                            "mass and inertia formula with 2e-7 relative float32 bookkeeping "
                            "tolerance. Payload remains inside the bare ego radius. "
                            "This does not change any controller safety tolerance."
                        ),
                        "event_intervals": intervals,
                        "first_degraded_control_time_seconds": float(time[bad_control[0]])
                        if len(bad_control)
                        else None,
                        "first_nonpositive_shell_interval_start_seconds": float(
                            interval_starts[bad_shell[0]]
                        )
                        if len(bad_shell)
                        else None,
                        "strict_success": bool(
                            not measured["physical_collision"]
                            and measured["minimum_inflated_clearance_m"] > 0
                            and measured["termination"] == "completed"
                            and measured["degraded_controls"] == 0
                            and measured["execution_audit"]["all_actual_physical_nodes_pass"]
                            and measured["execution_audit"][
                                "applied_motor_limit_violating_controls"
                            ]
                            == 0
                            and measured["execution_audit"][
                                "applied_predicted_derivative_violating_controls"
                            ]
                            == 0
                        ),
                    }
            results.append(row)
    output = {
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "Exact recorded prefix comparison before each first event; physical clearance uses "
            "the experiment's relative linear segment convention at 20 ms integration nodes. "
            "Event intervals are scheduled windows with actual observed control counts; "
            "a window may extend past route completion, and terminal padding is excluded. "
            "No continuous-time certificate and no deployment timing claim."
        ),
        "all_geometry_and_settings_exact": True,
        "all_pre_event_prefixes_bitwise_equal": all(
            method["pre_event_bitwise_equal"]
            for row in results
            for method in row["methods"].values()
        ),
        "strict_success_counts": {
            condition: {
                method: sum(
                    row["methods"][method]["strict_success"]
                    for row in results
                    if row["condition"] == condition
                )
                for method in ("fixed", "adaptive")
            }
            for condition in protocol["conditions"]
        },
        "runs": results,
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
