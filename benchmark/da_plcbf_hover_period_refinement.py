"""Allow measured re-adaptation and a calm return between repeated crossing waves."""

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
    output = BASE / "period-refinement-v1"
    source = BASE / "dense-search-v1/case-005/scene-plan.json"
    original = restore_scene(json.loads(source.read_text())["scene"])
    first, reverse, _ = original.world.config.wind_events
    trials = [
        {"index": 0, "half_period": 20.0, "second_wind_scale": 0.8, "direction": "original"},
        {"index": 1, "half_period": 20.0, "second_wind_scale": 0.8, "direction": "opposite_first"},
        {"index": 2, "half_period": 20.0, "second_wind_scale": 1.0, "direction": "original"},
        {"index": 3, "half_period": 22.0, "second_wind_scale": 0.8, "direction": "original"},
    ]
    begin(
        output, "Prespecified repeated-wave spacing refinement after initial adaptive-only escape."
    )
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    write(
        output / "plan.json",
        {
            "trials": trials,
            "source_scene_sha256": sha(source),
            "methods": ["PD_F", "F2", "A_BAL"],
            "geometry_rule": "Preserve first crossing times, centers, radii "
            "and crossing velocities. "
            "Scale sinusoid period and amplitude together to delay the repeated wave; this also "
            "slightly changes first-wave path curvature, "
            "which is not treated as an identical prefix.",
            "timing_rule": "Wind at 2 s, calm recovery at 9 s, new wind at 16 s, then repeated "
            "encounters 10 or 12 seconds after the direction change. Final calm return lasts 9 s.",
            "selection": "Complete the fixed four-triplet batch and retain all results.",
        },
    )
    cache, records, candidates = resources(), [], []
    for trial in trials:
        half = trial["half_period"]
        scale = half / 16
        second = (
            np.asarray(
                reverse.velocity
                if trial["direction"] == "original"
                else -np.asarray(first.velocity)
            )
            * trial["second_wind_scale"]
        )
        duration = half + 18
        events = (
            first,
            WindEvent(9.0, (0.0, 0.0, 0.0)),
            WindEvent(16.0, tuple(second)),
            WindEvent(half + 9, (0.0, 0.0, 0.0)),
        )
        config = replace(original.world.config, duration_seconds=duration, wind_events=events)
        config.validate()
        world = replace(
            original.world,
            config=config,
            obstacle_amplitudes=original.world.obstacle_amplitudes * scale,
            obstacle_angular_frequencies=original.world.obstacle_angular_frequencies / scale,
            obstacle_phases=original.world.obstacle_phases / scale,
        )
        scene = replace(original, world=world)
        task = HoverReturnTask(
            position_tolerance_m=0.7, return_windows=((10.0, 15.6), (half + 10, duration))
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
            half + 6,
            half + 8,
            half + 9,
            duration - 0.04,
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
