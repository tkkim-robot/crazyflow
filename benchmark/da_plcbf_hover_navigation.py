"""Reproduce the hover-first wind, recovery, centered-load and navigation explanation.

The uncompensated experiment uses the SAME uncompensated fallback mapping in both panes.
The independent nominal hover/navigation controller remains model-aware in both. This is
a deterministic mechanism experiment, not evidence of a real-time learner deadline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compensated-control", action="store_true")
    parser.add_argument("--no-payload", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--navigation-seconds", type=float, default=40.0)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    payload = () if args.no_payload else (PayloadEvent(19.0, mass_fraction=0.25),)
    navigation_start = 19.0 if args.no_payload else 23.0
    world = build_navigation_world(
        NavigationWorldConfig(
            seed=args.seed,
            duration_seconds=navigation_start + args.navigation_seconds,
            wind_events=(WindEvent(3.0, (1.6, 0.8, 0.0)), WindEvent(11.0, (0.0, 0.0, 0.0))),
            payload_events=payload,
        )
    )
    config = NavigationExperimentConfig(
        navigation_start_seconds=navigation_start,
        learning_start_seconds=0.0,
        update_every_controls=1,
        probe_every_controls=5,
        fallback_mapping=("compensated" if args.compensated_control else "matched_uncompensated"),
    )
    result = run_navigation_experiment(
        world,
        config,
        args.checkpoint,
        args.output_dir,
        device=jax.devices(args.device)[0],
        progress_callback=lambda method, index, count, reached: print(
            f"{method}: {index}/{count} controls; {reached} waypoints reached", flush=True
        ),
    )
    for name, data in (
        ("config.json", asdict(config)),
        ("world_config.json", asdict(world.config)),
    ):
        (args.output_dir / name).write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            {
                "directory": str(args.output_dir),
                "methods": {
                    name: {
                        key: method[key]
                        for key in (
                            "waypoints_completed",
                            "termination",
                            "minimum_inflated_clearance_m",
                            "degraded_controls",
                            "finite_updates",
                        )
                    }
                    for name, method in result.summary["methods"].items()
                },
                "protocol": result.summary["compensation_protocol"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
