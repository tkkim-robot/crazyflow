"""Read actual filter decisions at each authenticated first applied-command difference.

The arrays here are retained decisions, not reconstructed controller executions.
Nominal policy-row residuals are explicitly host-float64 reconstructions from
logged rows; the actual GPU acceptance flags are retained separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from benchmark.da_plcbf_actuator_diagnostic_analysis import arrays_digest


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=128)
def controls(record_path: str, expected: str) -> tuple[dict, dict]:
    path = Path(record_path)
    if sha(path) != expected:
        raise ValueError("authenticated original runner record changed")
    record = json.loads(path.read_text())
    archive = Path(record["episode_directory"]) / "controls.npz"
    digest = sha(archive)
    if digest != record["evidence_sha256"][str(archive)]:
        raise ValueError("retained control decisions changed")
    # Load only the diagnostic fields, never optional full predicted trajectories.
    keys = (
        "time",
        "controller_input_state",
        "observed_state",
        "estimated_model_sha256",
        "previous_command",
        "goal",
        "nominal_command",
        "selected_index",
        "selected_row",
        "selected_bound",
        "mode",
        "qp_valid",
        "qp_rejection_flags",
        "selected_policy_dual",
        "control_params_sha256",
        "control_library_version",
        "applied_passed",
    )
    with np.load(archive, allow_pickle=False) as source:
        values = {key: source[key] for key in keys}
    return values, {str(path): expected, str(archive): digest}


def finite(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def extract(analyses: list[Path], output: Path) -> dict:
    """Retain every declared pair and require shared recorded inputs before attribution."""
    output.mkdir(parents=True, exist_ok=False)
    rows, sources = [], {}
    for analysis in analyses:
        report_path = analysis / "report.json"
        report = json.loads(report_path.read_text())
        sources[str(report_path)] = sha(report_path)
        for pair in report["rows"]:
            row = {
                key: pair[key]
                for key in (
                    "pair_id",
                    "world_key",
                    "cell_id",
                    "runtime_method",
                    "realization",
                    "task_outcome",
                    "collision_free_outcome",
                    "admissible_pair",
                )
            }
            row["stage"] = report["stage"]
            first = pair.get("actual_command_comparison", {}).get("first_divergence")
            row["first_applied_difference"] = first
            if not pair["admissible_pair"] or first is None:
                row["status"] = "no_authenticated_first_difference"
                rows.append(row)
                continue
            command_path = analysis / pair["pair_id"] / "actual_commands.npz"
            with np.load(command_path, allow_pickle=False) as archive:
                command = {key: archive[key] for key in archive.files}
                if (
                    arrays_digest(command)
                    != pair["actual_command_comparison"]["extract_arrays_sha256"]
                ):
                    raise ValueError("authenticated applied-command extract changed")
                location = np.flatnonzero(command["time"] == first["time_seconds"])
                if len(location) != 1:
                    raise ValueError("first actual application time is not unique")
                position = int(location[0])
                indices = {
                    branch: int(command[f"{branch}.control_index"][position])
                    for branch in ("frozen", "adaptive")
                }
            sources[str(command_path)] = sha(command_path)
            values = {}
            for branch in ("frozen", "adaptive"):
                resolution = pair["record_resolution"][branch]
                data, provenance = controls(
                    resolution["original_record_file"], resolution["original_record_sha256"]
                )
                sources.update(provenance)
                values[branch] = {key: value[indices[branch]] for key, value in data.items()}
                if not np.isclose(values[branch]["time"], first["time_seconds"], atol=1e-9, rtol=0):
                    raise ValueError("application refers to a different controller time")
            row["recorded_input_equalities"] = {
                key: bool(np.array_equal(values["frozen"][key], values["adaptive"][key]))
                for key in (
                    "time",
                    "controller_input_state",
                    "observed_state",
                    "estimated_model_sha256",
                    "previous_command",
                    "goal",
                    "nominal_command",
                )
            }
            row["shared_recorded_input"] = all(row["recorded_input_equalities"].values())
            for branch, value in values.items():
                row[branch] = {
                    "selected_policy": int(value["selected_index"]),
                    "mode": str(value["mode"]),
                    "qp_valid": bool(value["qp_valid"]),
                    "applied_passed": bool(value["applied_passed"]),
                    "qp_rejection_flags": value["qp_rejection_flags"].tolist(),
                    "selected_policy_dual": finite(value["selected_policy_dual"]),
                    "nominal_row_residual_host_float64": finite(
                        np.float64(value["selected_bound"])
                        - np.asarray(value["selected_row"], dtype=np.float64)
                        @ np.asarray(value["nominal_command"], dtype=np.float64)
                    ),
                    "control_params_sha256": str(value["control_params_sha256"]),
                    "control_library_version": int(value["control_library_version"]),
                }
            row["selected_policy_equal"] = (
                row["frozen"]["selected_policy"] == row["adaptive"]["selected_policy"]
            )
            row["selected_row_equal"] = bool(
                np.array_equal(values["frozen"]["selected_row"], values["adaptive"]["selected_row"])
                and np.array_equal(
                    values["frozen"]["selected_bound"], values["adaptive"]["selected_bound"]
                )
            )
            row["status"] = "authenticated_decision_extract"
            rows.append(row)
    result = {
        "status": "completed",
        "pair_count": len(rows),
        "rows": rows,
        "source_files_sha256": sources,
        "extractor_sha256": sha(Path(__file__)),
        "qualification": (
            "Actual saved decisions at the first applied difference. Shared input means exact "
            "equality of the listed recorded fields; unlogged obstacle/safety arrays have common "
            "scene provenance. Host float64 row residuals do not reproduce the original GPU "
            "predicate. Flags/modes are the actual recorded predicate results. No optimizer or "
            "controller was rerun and no missing parameter checkpoint was reconstructed."
        ),
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    flat = []
    for row in rows:
        entry = {key: value for key, value in row.items() if not isinstance(value, dict)}
        for key in ("first_applied_difference", "frozen", "adaptive", "recorded_input_equalities"):
            entry.update({f"{key}.{name}": value for name, value in (row.get(key) or {}).items()})
        flat.append(entry)
    columns = sorted(set().union(*(row.keys() for row in flat)))
    with (output / "boundaries.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.analysis, args.output)
    print(json.dumps({"status": result["status"], "pair_count": result["pair_count"]}))


if __name__ == "__main__":
    main()
