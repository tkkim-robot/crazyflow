"""Bounded second-wind directions with an unchanged adaptive-only first encounter."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from benchmark.da_plcbf_hover_followup import restore_scene, run_scene_case
from benchmark.da_plcbf_hover_return import BASE, resources
from benchmark.da_plcbf_recovery_diagnosis import begin, sha, write
from crazyflow.safety.da_plcbf.actuator_hover import HoverReturnTask
from crazyflow.safety.da_plcbf.navigation_world import WindEvent


def main() -> None:
    output = BASE / "direction-refinement-v1"
    source = BASE / "dense-search-v1/case-005/scene-plan.json"
    original = restore_scene(json.loads(source.read_text())["scene"])
    first = original.world.config.wind_events[0]
    trials = [
        {"index": index, "second_wind_angle": float(index * np.pi / 4), "second_wind_speed": 1.6}
        for index in range(8)
    ]
    begin(output, "Fixed second-wind direction grid; original initial crossing geometry unchanged.")
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    write(
        output / "plan.json",
        {
            "trials": trials,
            "source_scene_sha256": sha(source),
            "methods": ["PD_F", "F2", "A_BAL"],
            "scope": "Development direction sensitivity after the initial adaptive-only escape. "
            "The first wind and all six obstacle trajectories are unchanged. Every method faces "
            "the same prescribed calm return interval and the same second wind "
            "in each complete run.",
            "timing": "Wind 2–9 s, calm 9–16 s, new wind 16–25.2 s, final calm to 32 s.",
            "selection": "All eight complete triplets retained; "
            "full return and operational validity "
            "remain necessary in addition to avoiding collision.",
        },
    )
    cache, records, candidates = resources(), [], []
    for trial in trials:
        angle, speed = trial["second_wind_angle"], trial["second_wind_speed"]
        events = (
            first,
            WindEvent(9.0, (0.0, 0.0, 0.0)),
            WindEvent(16.0, (speed * np.cos(angle), speed * np.sin(angle), 0.0)),
            WindEvent(25.2, (0.0, 0.0, 0.0)),
        )
        scene = replace(
            original,
            world=replace(
                original.world, config=replace(original.world.config, wind_events=events)
            ),
        )
        task = HoverReturnTask(
            position_tolerance_m=0.7, return_windows=((10.0, 15.6), (26.0, 32.0))
        )
        capture = (
            1.96,
            2.0,
            2.04,
            3.0,
            4.0,
            5.4,
            5.6,
            6.0,
            8.0,
            9.0,
            10.0,
            15.96,
            16.0,
            17.0,
            18.0,
            20.0,
            22.0,
            24.0,
            25.2,
            31.96,
        )
        rows = run_scene_case(
            output / f"case-{trial['index']:03d}",
            scene,
            task,
            cache,
            plan={"trial": trial, "source_scene_plan": str(source)},
            capture_times=capture,
        )
        methods = {r["method"]: r["summary"] for r in rows}
        candidate = methods["A_BAL"]["successful_full_episode"] and all(
            methods[m]["modeled_collider_collision"] is True for m in ("PD_F", "F2")
        )
        if candidate:
            candidates.append(trial["index"])
        records.append(
            {
                "case": trial["index"],
                "trial": trial,
                "candidate": candidate,
                "outcomes": {
                    m: {
                        k: s[k]
                        for k in (
                            "termination",
                            "physical_time_seconds",
                            "successful_full_episode",
                            "modeled_collider_collision",
                            "actual_operational_all_nodes_pass",
                            "hover_return",
                        )
                    }
                    for m, s in methods.items()
                },
            }
        )
        write(
            output / "report.json",
            {
                "records": records,
                "candidates": candidates,
                "planned": len(trials),
                "complete": len(records) == len(trials),
            },
        )
        print(json.dumps({"case_complete": trial["index"], "candidates": candidates}), flush=True)


if __name__ == "__main__":
    main()
