"""Predeclared paired worlds; development and held-out seed sets must be separate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["unchanged", "combined"],
        choices=("unchanged", "wind", "payload", "combined", "static"),
    )
    parser.add_argument("--obstacles", type=int, default=8)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--model", choices=("oracle", "estimated"), default="oracle")
    parser.add_argument(
        "--execution", choices=("deterministic", "budgeted"), default="deterministic"
    )
    parser.add_argument("--learner-kind", choices=("reference", "original"), default="reference")
    args = parser.parse_args()
    if args.source_tree:
        sys.path.insert(0, str(args.source_tree.resolve()))
    import jax

    from crazyflow.safety.da_plcbf.navigation_experiment import (
        NavigationExperimentConfig,
        run_navigation_experiment,
    )
    from crazyflow.safety.da_plcbf.navigation_world import (
        NavigationWorldConfig,
        PayloadEvent,
        WindEvent,
        build_navigation_world,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    config = NavigationExperimentConfig(
        model_information=args.model, execution_mode=args.execution, learner_kind=args.learner_kind
    )
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "source_tree": str(args.source_tree),
                "world_seeds": args.seeds,
                "conditions": args.conditions,
                "obstacles": args.obstacles,
                "duration": args.duration,
                "experiment_config": asdict(config),
                "promotion": (
                    "Fixed before these world draws; nominal parameters and Adam shared "
                    "by both methods; no per-world tuning"
                ),
                "distribution": (
                    "world geometry and absolute-time obstacle phases; learner checkpoint fixed"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    rows = []
    for seed in args.seeds:
        for condition in args.conditions:
            world = build_navigation_world(
                NavigationWorldConfig(
                    seed=seed,
                    obstacle_count=args.obstacles,
                    duration_seconds=args.duration,
                    moving_obstacles=condition != "static",
                    wind_events=(WindEvent(8.0, (2.0, 0.8, 0.0)), WindEvent(24.0, (-1.6, 1.0, 0.0)))
                    if condition in {"wind", "combined"}
                    else (),
                    payload_events=(PayloadEvent(16.0, 0.25),)
                    if condition in {"payload", "combined"}
                    else (),
                )
            )
            target = args.output / f"{condition}-seed{seed}"
            result = run_navigation_experiment(
                world,
                config,
                args.checkpoint,
                target,
                device=jax.devices("gpu")[0],
                progress_callback=lambda method, index, count, reached: print(
                    json.dumps(
                        {
                            "seed": seed,
                            "condition": condition,
                            "method": method,
                            "index": index,
                            "count": count,
                            "waypoints": reached,
                        }
                    ),
                    flush=True,
                ),
            )
            row = {
                "seed": seed,
                "condition": condition,
                "directory": str(target),
                "methods": {
                    name: {
                        key: value
                        for key, value in metrics.items()
                        if key
                        not in {"encounters", "publications_and_inputs", "snapshot_publications"}
                    }
                    for name, metrics in result.summary["methods"].items()
                },
            }
            rows.append(row)
            # A new progress file records completed runs without rewriting a run's evidence.
            (args.output / f"completed-{len(rows):03d}.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )
            print(json.dumps({"completed": len(rows), **row}), flush=True)
    (args.output / "campaign.json").write_text(json.dumps({"runs": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
