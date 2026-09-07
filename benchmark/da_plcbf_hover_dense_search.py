"""Prescribed six-mover family that also tests vertical escape and return routes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from benchmark.da_plcbf_hover_return import (
    BASE,
    METHODS,
    HoverSceneSpec,
    make_scene,
    resources,
    task_for,
)
from benchmark.da_plcbf_recovery_diagnosis import begin, write
from crazyflow.safety.da_plcbf.actuator_experiment import (
    ActuatorEpisodeConfig,
    run_actuator_episode,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig


def plans(seed: int, count: int) -> list[dict]:
    """Fix all offsets and absolute-time trajectories before observing this batch."""
    rng = np.random.default_rng(seed)
    result = []
    for index in range(count):
        spec = HoverSceneSpec(
            seed=seed + index,
            obstacle_count=6,
            wind_speed=(1.6, 1.8, 2.0)[index % 3],
            first_encounter=(3.2, 4.8, 6.0)[index % 3],
            stagger=float(rng.choice((0.32, 0.4, 0.48))),
            half_period=16.0,
            amplitude=float(rng.choice((0.8, 1.0, 1.2))) * 16 / np.pi,
            radius=float(rng.choice((0.5, 0.6, 0.7))),
            path_rotation=float(rng.uniform(-np.pi, np.pi)),
            lateral_offset=float(rng.choice((-0.2, 0.0, 0.2))),
        )
        heights = (0.0, 0.4, 0.0, -0.4, 0.0, 0.4 if index % 2 == 0 else -0.4)
        result.append({"index": index, "base": asdict(spec), "height_offsets_m": heights})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "dense-search-v1")
    parser.add_argument("--seed", type=int, default=71500)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.count <= 24:
        raise ValueError("bounded dense batch requires 1–24 complete triplets")
    begin(
        args.output, "Six prescribed movers after three-mover pilot and same-state recovery probes."
    )
    (args.output / "driver.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "base_driver.py").write_bytes(
        Path("benchmark/da_plcbf_hover_return.py").read_bytes()
    )
    trials = plans(args.seed, args.count)
    write(
        args.output / "plan.json",
        {
            "trials": trials,
            "methods": METHODS,
            "motivation": "Three primary home-height crossings plus movers "
            "above and below the home "
            "region test whether vertical escape leaves the frozen repertoire sufficient.",
            "controller_contract": "Same committed governor, wind information, motor limits, "
            "feedback and feedforward setting in all methods. No obstacle tracks a controller.",
            "selection": "Full batch retained; adaptive-only collision-free "
            "operationally valid return "
            "requires observed modeled-collider collisions in both frozen comparators.",
        },
    )
    cache, records, candidates = resources(), [], []
    for trial in trials:
        spec = HoverSceneSpec(**trial["base"])
        scene = make_scene(spec)
        means = scene.world.obstacle_mean_centers.copy()
        means[:, 2] += np.asarray(trial["height_offsets_m"])
        scene = replace(
            scene, family="hover_return_3d", world=replace(scene.world, obstacle_mean_centers=means)
        )
        assert np.min(means[:, 2] - scene.world.obstacle_radii) >= 0
        task = task_for(spec)
        directory = args.output / f"case-{trial['index']:03d}"
        directory.mkdir()
        write(
            directory / "scene-plan.json",
            {"trial": trial, "spec": asdict(spec), "scene": scene.metadata(), "task": asdict(task)},
        )
        rows = []
        for method in METHODS:
            bundle, controller, learner, metadata = cache.resolve(method, ActuatorFilterConfig())
            capture = tuple(
                sorted(
                    set(
                        (
                            1.96,
                            2.0,
                            2.04,
                            3.0,
                            4.0,
                            spec.first_encounter,
                            14.0,
                            15.0,
                            16.0,
                            spec.first_encounter + 16,
                            spec.first_encounter + 16 + 5 * spec.stagger,
                            25.2,
                            31.96,
                        )
                    )
                )
            )
            config = ActuatorEpisodeConfig(
                method=method,
                command_governor="committed_backup",
                allow_wind=True,
                hover_task=task,
                capture_times=capture,
                save_checkpoints=True,
            )
            print(json.dumps({"starting": trial["index"], "method": method}), flush=True)
            result = run_actuator_episode(
                scene,
                bundle,
                config,
                directory / method / "attempt-00",
                controllerfunctions=controller,
                learner_functions=learner,
            )
            rows.append({"method": method, "summary": result.summary, "metadata": metadata})
            write(directory / "report.json", {"records": rows, "planned": len(METHODS)})
            print(
                json.dumps(
                    {
                        "finished": trial["index"],
                        "method": method,
                        "termination": result.summary["termination"],
                        "safe_return": result.summary["successful_full_episode"],
                    }
                ),
                flush=True,
            )
        by_method = {r["method"]: r["summary"] for r in rows}
        candidate = by_method["A_BAL"]["successful_full_episode"] and all(
            by_method[m]["modeled_collider_collision"] is True for m in ("PD_F", "F2")
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
                    for m, s in by_method.items()
                },
            }
        )
        write(
            args.output / "report.json",
            {
                "records": records,
                "candidates": candidates,
                "planned": args.count,
                "complete": len(records) == args.count,
            },
        )
        print(json.dumps({"case_complete": trial["index"], "candidates": candidates}), flush=True)


if __name__ == "__main__":
    main()
