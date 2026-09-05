"""Derive branch clearances from immutable replay states; no controller or simulation runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clearance(
    states: np.ndarray, centers: np.ndarray, radii: np.ndarray, times: np.ndarray
) -> dict[str, Any]:
    relative = states[:, None, :3] - centers
    delta = np.diff(relative, axis=0)
    squared_length = np.sum(delta * delta, axis=-1)
    fraction = np.clip(
        -np.sum(relative[:-1] * delta, axis=-1) / np.maximum(squared_length, 1e-30), 0, 1
    )
    distances = np.linalg.norm(relative[:-1] + fraction[..., None] * delta, axis=-1) - radii
    interval, obstacle = np.unravel_index(np.argmin(distances), distances.shape)
    violating = np.flatnonzero(np.min(distances, axis=1) <= 0)
    first = int(violating[0]) if len(violating) else None
    return {
        "minimum_clearance_m": float(distances[interval, obstacle]),
        "minimum_clearance_obstacle_index": int(obstacle),
        "minimum_clearance_chord_time_seconds": float(
            times[interval] + fraction[interval, obstacle] * (times[interval + 1] - times[interval])
        ),
        "first_violating_interval_index": first,
        "first_violating_interval_start_seconds": float(times[first])
        if first is not None
        else None,
        "first_violating_interval_end_seconds": float(times[first + 1])
        if first is not None
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision_directory", type=Path)
    args = parser.parse_args()
    output = args.revision_directory / "branch_clearance_audit.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    reports = []
    for name in ("legacy-estimated-replay", "legacy-oracle-replay-2"):
        directory = args.revision_directory / name
        replay_path = directory / "replay.json"
        replay = json.loads(replay_path.read_text())
        source = Path(replay["source_directory"])
        source_trace = source / "competent_comparison.npz"
        if _digest(source_trace) != replay["source_trace_sha256"]:
            raise ValueError("archived source trace differs from replay provenance")
        scenario_path = source / "feasibility_reference.json"
        scenario = json.loads(scenario_path.read_text())["scenario"]
        scenario_config = json.loads((source / "competent_comparison.json").read_text())["summary"][
            "config"
        ]
        hold = int(scenario_config["control_interval_steps"])
        dt = float(scenario["dt"])
        mask = np.asarray(scenario["obstacle_mask"], dtype=bool)
        original_indices = np.flatnonzero(mask)
        mean = np.asarray(scenario["obstacle_initial_centers"])[mask]
        velocity = np.asarray(scenario["obstacle_velocities"])[mask]
        radius = np.asarray(scenario["obstacle_radii"])[mask] + float(scenario["ego_radius"])
        rows = []
        with np.load(directory / "replay.npz", allow_pickle=False) as archive:
            for branch, metadata in replay["pre_failure_closed_loop_branches"].items():
                states = np.asarray(archive[f"branch_{branch}_states"], dtype=float)
                modes = archive[f"branch_{branch}_modes"]
                if len(states) != len(modes) * hold + 1 or not np.all(np.isfinite(states)):
                    raise ValueError("replay nodes must exactly span all recorded controls")
                start = float(metadata["start_time"])
                times = start + np.arange(len(states)) * dt
                centers = mean[None] + times[:, None, None] * velocity[None]
                physical = _clearance(states, centers, radius, times)
                shell = _clearance(
                    states, centers, radius + float(scenario["obstacle_clearance"]), times
                )
                for result in (physical, shell):
                    result["minimum_clearance_obstacle_index"] = int(
                        original_indices[result["minimum_clearance_obstacle_index"]]
                    )
                first = physical["first_violating_interval_index"]
                controls = len(modes) if first is None else first // hold + 1
                emergency = int(np.sum(modes == 2))
                if emergency != metadata["emergency_controls"]:
                    raise ValueError("saved emergency summary does not match raw execution modes")
                rows.append(
                    {
                        "branch": branch,
                        "start_time_seconds": start,
                        "end_time_seconds": float(times[-1]),
                        "start_library_version": metadata["start_version"],
                        "integration_dt_seconds": dt,
                        "control_interval_steps": hold,
                        "raw_control_count": len(modes),
                        "raw_dense_node_count": len(states),
                        "raw_emergency_controls": emergency,
                        "controls_through_first_physical_collision_interval": controls,
                        "emergency_controls_through_first_physical_collision_interval": int(
                            np.sum(modes[:controls] == 2)
                        ),
                        "post_collision_control_continuation_count": len(modes) - controls,
                        "physical": physical,
                        "inflated_shell": shell,
                    }
                )
        reports.append(
            {
                "replay_directory": str(directory),
                "source_directory": str(source),
                "branches": rows,
                "replay_npz_sha256": _digest(directory / "replay.npz"),
                "replay_json_sha256": _digest(replay_path),
                "source_trace_sha256": _digest(source_trace),
                "source_scenario_sha256": _digest(scenario_path),
            }
        )
    result = {
        "scope": (
            "Read-only derivation from saved dense plant nodes and archived absolute-time "
            "constant-velocity obstacle scenarios. Relative swept chords, not a continuous-time "
            "flight guarantee. Raw minima and raw emergency counts include explicitly disclosed "
            "diagnostic continuation after collision; censored counts are separate."
        ),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _digest(Path(__file__)),
        "replays": reports,
    }
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for report in reports:
        for row in report["branches"]:
            print(
                report["replay_directory"],
                row["branch"],
                row["physical"]["minimum_clearance_m"],
                row["inflated_shell"]["minimum_clearance_m"],
                row["raw_emergency_controls"],
                row["emergency_controls_through_first_physical_collision_interval"],
            )


if __name__ == "__main__":
    main()
