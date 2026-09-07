"""Prescribed calm return opportunities for the adaptive-only initial-encounter case."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from benchmark.da_plcbf_hover_followup import restore_scene, run_scene_case
from benchmark.da_plcbf_hover_return import BASE, resources
from benchmark.da_plcbf_recovery_diagnosis import begin, sha, write
from crazyflow.safety.da_plcbf.actuator_hover import HoverReturnTask
from crazyflow.safety.da_plcbf.navigation_world import WindEvent


def main() -> None:
    output = BASE / "recovery-window-v1"
    source = BASE / "dense-search-v1/case-005/scene-plan.json"
    original = json.loads(source.read_text())
    original_scene = restore_scene(original["scene"])
    begin(
        output, "Prescribed recovery-window variants after a recorded adaptive-only initial escape."
    )
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    trials = [
        {"index": 0, "calm_start": 9.0, "reverse_time": 16.0, "second_wind_scale": 1.0},
        {"index": 1, "calm_start": 8.4, "reverse_time": 16.0, "second_wind_scale": 1.0},
        {"index": 2, "calm_start": 9.0, "reverse_time": 16.0, "second_wind_scale": 0.8},
    ]
    write(
        output / "plan.json",
        {
            "source_scene_sha256": sha(source),
            "trials": trials,
            "methods": ["PD_F", "F2", "A_BAL"],
            "development_basis": "The source case has frozen collisions near 5.6 s and adaptive "
            "obstacle clearance, followed by failed station-keeping. These variants introduce "
            "a calm return opportunity after the first moving-obstacle pass, "
            "as proposed in the task.",
            "invariants": "All obstacle trajectories, first wind and controller "
            "settings are unchanged. "
            "Each variant is prescribed identically for all methods. Original failure retained.",
            "selection": "All three full triplets retained. Promotion still requires all return "
            "criteria and operational limits, not survival alone.",
        },
    )
    cache, records, candidates = resources(), [], []
    first, reverse, calm = original_scene.world.config.wind_events
    for trial in trials:
        events = (
            first,
            WindEvent(trial["calm_start"], (0.0, 0.0, 0.0)),
            WindEvent(
                trial["reverse_time"],
                tuple(trial["second_wind_scale"] * x for x in reverse.velocity),
            ),
            calm,
        )
        scene = replace(
            original_scene,
            world=replace(
                original_scene.world,
                config=replace(original_scene.world.config, wind_events=events),
            ),
        )
        scene.world.config.validate()
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
            trial["calm_start"],
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
            plan={
                "trial": trial,
                "source_scene_plan": str(source),
                "source_scene_plan_sha256": sha(source),
            },
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
