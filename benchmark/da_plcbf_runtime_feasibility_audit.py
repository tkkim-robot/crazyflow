"""Read-only retrospective task/runtime audit of saved paced navigation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from crazyflow.safety.da_plcbf.runtime_feasibility import assess_navigation_runtime_feasibility


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for directory in args.directories:
        summary_path = directory / "navigation_comparison.json"
        trace_path = directory / "navigation_comparison.npz"
        raw_path = directory / "raw_diagnostics.npz"
        summary = json.loads(summary_path.read_text())["summary"]
        feasibility = assess_navigation_runtime_feasibility(summary)
        controls = {}
        world = summary["world"]["config"]
        period = world["dt"] * world["control_interval_steps"]
        with (
            np.load(trace_path, allow_pickle=False) as trace,
            np.load(raw_path, allow_pickle=False) as raw,
        ):
            for method in ("fixed", "adaptive"):
                record = summary["methods"][method]
                active = trace[f"{method}_recorded_control_valid"].astype(bool)
                versions = trace[f"{method}_library_version"][active]
                missed = trace[f"{method}_missed_deadline"][active]
                rows = record["publications_and_inputs"]
                np.testing.assert_array_equal(versions, [row["version_used"] for row in rows])
                assert np.all(np.diff(versions) >= 0), "used versions went backward"
                np.testing.assert_array_equal(missed, [row["missed_deadline"] for row in rows])
                assert int(np.sum(active)) == record["active_controls"]
                assert int(np.sum(missed)) == record["service_exceeds_nominal_period_count"]
                finite = int(np.sum(raw[f"{method}_finite_update"][active]))
                assert finite == record["finite_updates"]
                recomputed = [
                    row["completed_wall_seconds"] > row["scheduled_wall_seconds"] + period
                    for row in rows
                ]
                np.testing.assert_array_equal(missed, recomputed)
                controls[method] = {
                    "active_controls": int(np.sum(active)),
                    "finite_updates": finite,
                    "deadline_misses_from_raw_trace_and_wall_times": int(np.sum(missed)),
                    "used_advanced_version_controls": int(np.sum(versions > versions[0])),
                    "publication_records": len(record["snapshot_publications"]),
                    "controller_service": record["controller_service"],
                    "learner_service": record["learner_service"],
                }
        adaptive = summary["methods"]["adaptive"]
        warm = adaptive.get("warmup", {})
        controller_median = adaptive["controller_service"]["median_seconds"]
        reserved = warm.get("initial_reserved_update_seconds")
        reserve = summary["config"]["controller_reserve_seconds"]
        paths = [summary_path, trace_path, raw_path]
        paths.extend(
            path
            for path in (directory / "adaptive_warmup.json", directory / "SOURCE_SHA256.json")
            if path.exists()
        )
        results.append(
            {
                "directory": str(directory),
                "runtime_feasibility": feasibility,
                "actual_counts": controls,
                "file_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
                },
                "initial_budget": {
                    "control_period_seconds": period,
                    "warmup": warm or None,
                    "controller_median_seconds": controller_median,
                    "reserved_update_seconds": reserved,
                    "additional_reserve_seconds": reserve,
                    "controller_median_plus_reserved_update_and_reserve_seconds": (
                        None if reserved is None else controller_median + reserved + reserve
                    ),
                    "scope": (
                        "illustrative serialized cost sum before plant/telemetry; "
                        "not an inferred update launch"
                    ),
                },
            }
        )
    output = {
        "scope": (
            "read-only retrospective audit; task completion does not establish "
            "online-learning runtime feasibility"
        ),
        "source_files_unchanged": True,
        "audit_helper_sha256": hashlib.sha256(
            Path("crazyflow/safety/da_plcbf/runtime_feasibility.py").read_bytes()
        ).hexdigest(),
        "runs": results,
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "feasible": [
                    row["runtime_feasibility"]["adaptive_online_runtime_feasible"]
                    for row in results
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
