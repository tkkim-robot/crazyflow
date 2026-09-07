"""Retain complete paired development outcomes, including failures and confirmation cases."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from benchmark.da_plcbf_hover_return import BASE, METHODS
from benchmark.da_plcbf_recovery_diagnosis import sha, write

PHASES = {
    "calibration-v1": 3,
    "calibration-v2": 3,
    "pilot-v1": 3,
    "search-v1": 16,
    "dense-search-v1": 8,
    "recovery-window-v1": 3,
    "period-refinement-v1": 4,
    "direction-refinement-v1": 8,
    "confirmation-v1": 16,
}


def main() -> None:
    output = BASE / "inventory-v1"
    output.mkdir(exist_ok=False)
    cases, rows, phases = [], [], []
    for phase, expected in PHASES.items():
        directory = BASE / phase
        summaries = sorted(directory.glob("case-*/*/attempt-00/summary.json"))
        if not summaries:
            summaries = sorted(directory.glob("*/*/attempt-00/summary.json"))
        grouped = defaultdict(dict)
        for path in summaries:
            summary = json.loads(path.read_text())
            case = path.parents[2].name
            grouped[case][summary["method"]] = (path, summary)
        assert len(grouped) == expected, (phase, len(grouped), expected)
        phase_rows = []
        for name, methods in sorted(grouped.items()):
            assert set(methods) == set(METHODS), (phase, name, set(methods))
            worlds = {s["physical_world_id"] for _, s in methods.values()}
            assert len(worlds) == 1
            outcome = {
                m: {
                    k: s[k]
                    for k in (
                        "physical_world_id",
                        "termination",
                        "physical_time_seconds",
                        "duration_seconds",
                        "successful_full_episode",
                        "modeled_collider_collision",
                        "actual_operational_all_nodes_pass",
                        "hover_return",
                        "control_count",
                        "initial_library_version",
                        "final_published_library_version",
                        "finite_credited_updates",
                        "freeze_learning_actual_boundary",
                    )
                }
                for m, (_, s) in methods.items()
            }
            case = dict(
                phase=phase,
                case=name,
                physical_world_id=worlds.pop(),
                outcomes=outcome,
                adaptive_only=outcome["A_BAL"]["successful_full_episode"]
                and all(outcome[m]["modeled_collider_collision"] is True for m in ("PD_F", "F2")),
                summary_sha256={m: sha(p) for m, (p, _) in methods.items()},
            )
            cases.append(case)
            phase_rows.append(case)
            for m, s in outcome.items():
                h = s["hover_return"]
                rows.append(
                    dict(
                        phase=phase,
                        case=name,
                        method=m,
                        physical_world_id=s["physical_world_id"],
                        termination=s["termination"],
                        physical_seconds=s["physical_time_seconds"],
                        collision=s["modeled_collider_collision"],
                        safe_return=s["successful_full_episode"],
                        operational_pass=s["actual_operational_all_nodes_pass"],
                        time_near_home_seconds=h["time_near_home_seconds"],
                        max_home_distance_m=h["maximum_displacement_m"],
                        final_home_distance_m=h["final_home_distance_m"],
                        final_speed_mps=h["final_speed_mps"],
                        adaptive_only_triplet=case["adaptive_only"],
                    )
                )
        phases.append(
            dict(
                phase=phase,
                paired_cases=len(phase_rows),
                adaptive_only_cases=sum(r["adaptive_only"] for r in phase_rows),
                methods={
                    m: dict(
                        safe_return=sum(
                            r["outcomes"][m]["successful_full_episode"] for r in phase_rows
                        ),
                        collision=sum(
                            r["outcomes"][m]["modeled_collider_collision"] is True
                            for r in phase_rows
                        ),
                    )
                    for m in METHODS
                },
            )
        )
    groups = []
    confirmation = [r for r in cases if r["phase"] == "confirmation-v1"]
    for group in ["nearby", "fresh"]:
        members = []
        for r in confirmation:
            plan = json.loads((BASE / r["phase"] / r["case"] / "scene-plan.json").read_text())
            if plan["trial"]["group"] == group:
                members.append(r)
        assert len(members) == 8
        groups.append(
            dict(
                group=group,
                paired_cases=len(members),
                adaptive_only_cases=sum(r["adaptive_only"] for r in members),
                methods={
                    m: dict(
                        safe_return=sum(
                            r["outcomes"][m]["successful_full_episode"] for r in members
                        ),
                        collision=sum(
                            r["outcomes"][m]["modeled_collider_collision"] is True for r in members
                        ),
                    )
                    for m in METHODS
                },
            )
        )
    write(
        output / "report.json",
        dict(
            complete=True,
            phases=phases,
            confirmation_groups=groups,
            cases=cases,
            scope="Every prespecified development triplet is included. Stages were adaptively "
            "selected from earlier development outcomes. Confirmation plans were fixed before "
            "their outcomes; fresh configurations remain within the selected development family.",
        ),
    )
    with (output / "outcomes.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(phases=phases, confirmation=groups), indent=2), flush=True)


if __name__ == "__main__":
    main()
