"""Read-only audit of every completed development attempt in the recovery iteration."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/da_plcbf/recovery-interaction-20260907/v1"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    output = BASE / "analysis-v2"
    output.mkdir(exist_ok=False)
    rows = []
    for stage in ("guard-fixed-v1", "guard-fixed-v2", "wind-demo-v1"):
        for directory in sorted((BASE / stage).glob("*/attempt-00")):
            summary = json.loads((directory / "summary.json").read_text())
            binding = json.loads((directory / "binding.json").read_text())
            with np.load(directory / "controls.npz", allow_pickle=False) as c:
                has_plan = c["backup_has_checked_plan"]
                executed = c["backup_executed_backup"]
                times = c["time"]
                row = {
                    "stage": stage,
                    "case_method": directory.parent.name,
                    "method": summary["method"],
                    "safe_task": summary["successful_full_episode"],
                    "termination": summary["termination"],
                    "end_time": summary["physical_time_seconds"],
                    "waypoints": summary["waypoints_completed"],
                    "waypoints_total": summary["waypoints_total"],
                    "operational_pass": summary["actual_operational_all_nodes_pass"],
                    "finite_credited_updates": summary["finite_credited_updates"],
                    "all_controls_have_checked_backup": bool(np.all(has_plan)),
                    "unchecked_controls": int(np.count_nonzero(~has_plan)),
                    "executed_backup_controls": int(np.count_nonzero(executed)),
                    "first_backup_execution": float(times[executed][0])
                    if np.any(executed)
                    else None,
                    "first_unchecked_control": float(times[~has_plan][0])
                    if np.any(~has_plan)
                    else None,
                    "stored_tail_controls": int(np.count_nonzero(c["backup_used_stored_tail"]))
                    if "backup_used_stored_tail" in c
                    else None,
                    "held_timing_covered": summary["online_held_check_timing_all_covered"],
                    "directory": str(directory.relative_to(ROOT)),
                    "controls_sha256": sha(directory / "controls.npz"),
                    "binding_sha256": sha(directory / "binding.json"),
                }
                if stage == "wind-demo-v1":
                    schedule = binding["scene"]["world"]["config"]["wind_events"]

                    def wind_at(when: float) -> np.ndarray:
                        wind = np.zeros(3)
                        for event in schedule:
                            if when >= event["time_seconds"] - 1e-10:
                                wind = np.asarray(event["velocity"])
                        return wind

                    expected = np.asarray([wind_at(t) for t in times], dtype=np.float32)
                    np.testing.assert_array_equal(c["actual_wind_velocity"], expected)
                    np.testing.assert_array_equal(c["estimated_wind_velocity"], expected)
                    with np.load(directory / "dense.npz", allow_pickle=False) as d:
                        np.testing.assert_array_equal(
                            d["actual_wind_velocity"], [wind_at(t) for t in d["time"]]
                        )
                    row["actual_and_estimated_wind_match_recorded_schedule"] = True
                    row["preflight_updates"] = (
                        int(np.max(c["control_library_version"][times < 19]))
                        - summary["initial_library_version"]
                    )
            rows.append(row)
    assert sum(r["stage"] == "guard-fixed-v2" for r in rows) == 12
    assert sum(r["stage"] == "wind-demo-v1" for r in rows) == 3
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (output / "flight_source.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {"rows": rows, "scope": "Development evidence, not fresh validation."}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
