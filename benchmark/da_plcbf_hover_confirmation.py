"""Prespecified nearby and fresh perturbations of the selected hover demonstration."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from benchmark.da_plcbf_hover_followup import restore_scene, run_scene_case
from benchmark.da_plcbf_hover_return import resources
from benchmark.da_plcbf_recovery_diagnosis import begin, sha, write
from crazyflow.safety.da_plcbf.actuator_hover import HoverReturnTask
from crazyflow.safety.da_plcbf.navigation_world import WindEvent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.case / "scene-plan.json"
    plan = json.loads(source.read_text())
    original = restore_scene(plan["scene"])
    task = HoverReturnTask(
        **{**plan["task"], "return_windows": tuple(map(tuple, plan["task"]["return_windows"]))}
    )
    defaults = dict(
        wind_scale=1.0,
        wind_rotation=0.0,
        second_rotation=0.0,
        offset_x=0.0,
        radius_scale=1.0,
        encounter_shift=0.0,
        speed_scale=1.0,
    )
    trials = []
    for field, delta in [
        ("wind_scale", 0.025),
        ("wind_rotation", 0.035),
        ("radius_scale", 0.02),
        ("encounter_shift", 0.08),
    ]:
        for sign in [-1, 1]:
            trials.append(dict(defaults, **{field: defaults[field] + sign * delta}, group="nearby"))
    rng = np.random.default_rng(71931)
    for _ in range(8):
        trials.append(
            dict(
                group="fresh",
                wind_scale=float(rng.uniform(0.94, 1.06)),
                wind_rotation=float(rng.uniform(-0.08, 0.08)),
                second_rotation=float(rng.uniform(-0.12, 0.12)),
                offset_x=float(rng.uniform(-0.04, 0.04)),
                radius_scale=float(rng.uniform(0.96, 1.04)),
                encounter_shift=float(rng.uniform(-0.18, 0.18)),
                speed_scale=float(rng.uniform(0.985, 1.015)),
            )
        )
    begin(
        args.output,
        "Sealed 8 one-factor nearby and 8 independent multi-factor fresh scenes; "
        "no retuning from outcomes.",
    )
    (args.output / "driver.py").write_bytes(Path(__file__).read_bytes())
    write(
        args.output / "plan.json",
        dict(
            source_sha256=sha(source),
            trials=trials,
            scope="Fresh means new physical configurations in this selected development family, "
            "not a held-out population guarantee.",
            selection="All 16 matched triplets retained, including negative cases. "
            "Same home tolerances and full duration.",
        ),
    )
    cache, records, ids = resources(), [], set()
    for index, trial in enumerate(trials):
        events = []
        for i, event in enumerate(original.world.config.wind_events):
            theta = trial["wind_rotation"] + (trial["second_rotation"] if i == 2 else 0.0)
            c, s = np.cos(theta), np.sin(theta)
            vector = np.asarray(event.velocity)
            rotated = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ vector
            events.append(WindEvent(event.time_seconds, tuple(rotated * trial["wind_scale"])))
        world = replace(
            original.world,
            config=replace(original.world.config, seed=71931 + index, wind_events=tuple(events)),
            obstacle_mean_centers=original.world.obstacle_mean_centers
            + np.array([trial["offset_x"], 0.0, 0.0]),
            obstacle_radii=original.world.obstacle_radii * trial["radius_scale"],
            obstacle_angular_frequencies=original.world.obstacle_angular_frequencies
            * trial["speed_scale"],
            obstacle_phases=original.world.obstacle_phases
            - original.world.obstacle_angular_frequencies * trial["encounter_shift"],
        )
        scene = replace(original, world=world, scene_seed=71931 + index)
        identity = scene.metadata()["physical_world_id"]
        assert identity not in ids
        ids.add(identity)
        rows = run_scene_case(
            args.output / f"case-{index:03d}",
            scene,
            task,
            cache,
            plan={"trial": trial, "source_scene_plan": str(source)},
            capture_times=(2.0, 4.0, 5.4, 16.0, 20.0),
        )
        summaries = {r["method"]: r["summary"] for r in rows}
        records.append(
            dict(
                index=index,
                trial=trial,
                physical_world_id=identity,
                outcomes=summaries,
                adaptive_only=summaries["A_BAL"]["successful_full_episode"]
                and all(summaries[m]["modeled_collider_collision"] for m in ("PD_F", "F2")),
            )
        )
        write(
            args.output / "report.json",
            dict(records=records, planned=len(trials), complete=len(records) == len(trials)),
        )
        print(
            json.dumps({"case_complete": index, "adaptive_only": records[-1]["adaptive_only"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
