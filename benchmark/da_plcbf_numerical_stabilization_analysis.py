"""Read-only causal and command-precision audit of the closed development set."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/da_plcbf/numerical-stabilization-20260906/v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from benchmark.da_plcbf_actuator_confirmation import (
        authenticate_boundary_snapshot,
        load_saved_run,
    )
    from benchmark.da_plcbf_actuator_diagnostic_analysis import (
        _snapshot,
        command_comparison,
        compare_prefix_arrays,
        extract_prefix,
        prefix_arrays,
    )

    protocol_path = BASE / "sealed-v4/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    campaign = json.loads((BASE / "development-36-v3/results.json").read_text())
    records = {r["trial"]["id"]: r for r in campaign["records"]}
    assert len(records) == 36
    assert set(records) == {t["id"] for t in protocol["trials"]}
    output = BASE / "analysis-v1"
    output.mkdir(exist_ok=False)
    rows = []
    evidence = {str(protocol_path): sha(protocol_path)}
    for record in records.values():
        assert record["protocol_sha256"] == sha(protocol_path)
        for name, expected in record["evidence_sha256"].items():
            assert sha(Path(name)) == expected, name
            evidence[name] = expected
        t, s = record["trial"], record["summary"]
        assert s["status"] == "completed"
        assert s["duration_seconds"] == (14.0 if t["case"] == "harm" else 24.0)
        with np.load(Path(record["episode_directory"]) / "controls.npz") as c:
            flags = c["qp_rejection_flags"]
            raw = c["qp_numerics_executed_residuals"][:, -1]
            solver = c["qp_numerics_solver_feasible"]
            qp_counts = {
                "controller_decisions": len(flags),
                "qp_accepted": int(np.sum(c["qp_valid"])),
                "raw_policy_rejections": int(np.sum(flags[:, 5])),
                "solver_feasible_but_raw_row_fails": int(np.sum(solver & (raw < -2e-6))),
                "executed_raw_row_min_for_solver_feasible": float(np.min(raw[solver]))
                if np.any(solver)
                else None,
                "held_rejections": int(np.sum(flags[:, 6] | flags[:, 7])),
            }
        rows.append(
            {k: t[k] for k in ["id", "case", "method", "learner_numerics", "qp_numerics"]}
            | {
                "outcome": "safe_task"
                if s["successful_full_episode"]
                else "collision"
                if s["modeled_collider_collision"]
                else "operational_failure"
                if not s["actual_operational_all_nodes_pass"]
                else "timeout",
                "waypoints": s["waypoints_completed"],
                "physical_seconds": s["physical_time_seconds"],
                "online_updates": s["finite_credited_updates"],
                "hold_timing_covered": s["online_held_check_timing_all_covered"],
                **qp_counts,
                "source_summary": str(Path(record["episode_directory"]) / "summary.json"),
            }
        )
    with (output / "flight_source.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    pairs = []
    for case in ["harm", "gain", "no_change_regression"]:
        for qp in ["legacy", "inward"]:
            for learner in ["legacy", "matched"]:
                aid = f"{case}__A_BAL__{learner}__{qp}"
                fid = f"{case}__ONSET_FREEZE__{learner}__{qp}"
                ar, fr = records[aid], records[fid]
                a, f = (
                    load_saved_run(ar["episode_directory"]),
                    load_saved_run(fr["episode_directory"]),
                )
                assert a.binding["scene"] == f.binding["scene"]
                assert a.binding["initial_learner_sha256"] == f.binding["initial_learner_sha256"]
                pa, _ = extract_prefix(
                    a, 2.0, authenticate_boundary_snapshot(a, _snapshot(a, 2.0), 2.0)
                )
                pf, _ = extract_prefix(
                    f, 2.0, authenticate_boundary_snapshot(f, _snapshot(f, 2.0), 2.0)
                )
                arrays = compare_prefix_arrays(prefix_arrays(a, 2.0), prefix_arrays(f, 2.0))
                assert pa == pf and arrays["exact_equal"], aid
                commands, _ = command_comparison(f, a)
                boundary = None
                if commands["first_divergence"] is not None:
                    when = commands["first_divergence"]["time_seconds"]
                    indices = [
                        int(np.flatnonzero(run.controls["time"] == when)[0]) for run in (f, a)
                    ]
                    shared_fields = (
                        "controller_input_state_sha256",
                        "estimated_model_sha256",
                        "previous_command",
                        "goal",
                        "nominal_command",
                    )
                    boundary = {
                        "time": when,
                        "shared_inputs": {
                            name: bool(
                                np.array_equal(
                                    f.controls[name][indices[0]], a.controls[name][indices[1]]
                                )
                            )
                            for name in shared_fields
                        },
                        "decisions": {},
                    }
                    for label, run, index in zip(
                        ("frozen", "adaptive"), (f, a), indices, strict=True
                    ):
                        names = (
                            "selected_index",
                            "selected_row",
                            "selected_bound",
                            "selected_gradient",
                            "selected_policy_dual",
                            "qp_rejection_flags",
                            "qp_proposed_command",
                            "planned_command",
                            "nominal_command",
                            "qp_numerics_executed_residuals",
                        )
                        boundary["decisions"][label] = {
                            name: run.controls[name][index].tolist() for name in names
                        }
                        boundary["decisions"][label]["nominal_raw_row_residual_host_float64"] = (
                            float(
                                np.float64(run.controls["selected_bound"][index])
                                - np.asarray(run.controls["selected_row"][index], dtype=np.float64)
                                @ np.asarray(
                                    run.controls["nominal_command"][index], dtype=np.float64
                                )
                            )
                        )
                success_a, success_f = (
                    ar["summary"]["successful_full_episode"],
                    fr["summary"]["successful_full_episode"],
                )
                pairs.append(
                    {
                        "case": case,
                        "qp_numerics": qp,
                        "learner_numerics": learner,
                        "full_prefix_authenticated": True,
                        "prefix_hashes": pa,
                        "outcome": "both"
                        if success_a and success_f
                        else "adaptation_only"
                        if success_a
                        else "freeze_only"
                        if success_f
                        else "neither",
                        "commands": commands,
                        "first_boundary": boundary,
                    }
                )
    report = {
        "completed_flights": 36,
        "authenticated_pairs": len(pairs),
        "pairs": pairs,
        "rows": rows,
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "evidence_sha256": evidence,
        "analysis_source_sha256": sha(Path(__file__)),
        "qualification": (
            "All three cases were previously observed; these are development "
            "re-evaluations, not new validation worlds."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed": 36, "pairs": len(pairs), "outcomes": report["outcomes"]}))


if __name__ == "__main__":
    main()
