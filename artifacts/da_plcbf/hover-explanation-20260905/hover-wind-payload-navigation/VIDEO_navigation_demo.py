"""Generate an exogenous world, run a matched checkpoint, or render saved navigation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from crazyflow.safety.da_plcbf.deterministic_schedule import DeterministicUpdateSchedule
from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
    ComparisonRenderConfig,
    render_comparison_video,
)
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
from crazyflow.safety.da_plcbf.online_constant_wind import load_online_constant_wind_result


def load_world_config(path: Path | None) -> NavigationWorldConfig:
    """Accept a config object or a previously generated world.json without losing event types."""
    values = json.loads(path.read_text()) if path else {}
    values = values.get("config", values)
    return NavigationWorldConfig(
        **{
            **values,
            "wind_events": tuple(
                WindEvent(float(event["time_seconds"]), tuple(event["velocity"]))
                for event in values.get("wind_events", ())
            ),
            "payload_events": tuple(
                PayloadEvent(
                    float(event["time_seconds"]),
                    event.get("mass_fraction", 0.25),
                    tuple(event.get("half_extents", (0.025, 0.025, 0.025))),
                )
                for event in values.get("payload_events", ())
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("world", "run", "render"))
    parser.add_argument("--world-config", type=Path, help="World config or saved world.json")
    parser.add_argument("--config", type=Path, help="NavigationExperimentConfig JSON")
    parser.add_argument(
        "--checkpoint", type=Path, help="Complete learner checkpoint stem; required for run"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-dir", type=Path, help="Saved numerical run for render; defaults to output-dir"
    )
    parser.add_argument(
        "--schedule", type=Path, help="Saved deterministic opportunity schedule JSON"
    )
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--mode", choices=("demo", "diagnostic"), default="demo")
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--comparison-note", default="", help="Persistent explanation of the saved comparison"
    )
    parser.add_argument(
        "--camera-distance", type=float, help="Navigation camera distance in metres"
    )
    parser.add_argument(
        "--hover-camera-distance",
        type=float,
        help="Equal camera distance during the recorded hover phase",
    )
    parser.add_argument("--camera-transition-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.command == "render":
        source = args.input_dir or args.output_dir
        result = load_online_constant_wind_result(
            source / "navigation_comparison.npz", source / "navigation_comparison.json"
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        world = build_navigation_world(load_world_config(args.world_config))
        if args.command == "world":
            args.output_dir.mkdir(parents=True, exist_ok=False)
            (args.output_dir / "world.json").write_text(
                json.dumps(world.metadata(), indent=2) + "\n"
            )
            print(json.dumps({"world": str(args.output_dir / "world.json")}, indent=2))
            return
        if args.checkpoint is None:
            parser.error(
                "run requires --checkpoint; prepare the nominal reference checkpoint first"
            )
        config = NavigationExperimentConfig(
            **(json.loads(args.config.read_text()) if args.config else {})
        )
        schedule = (
            DeterministicUpdateSchedule(
                tuple(json.loads(args.schedule.read_text())["opportunities"])
            )
            if args.schedule
            else None
        )
        result = run_navigation_experiment(
            world,
            config,
            args.checkpoint,
            args.output_dir,
            device=jax.devices(args.device)[0],
            opportunity_schedule=schedule,
            progress_callback=lambda method, index, count, reached: print(
                f"{method}: {index}/{count} controls; {reached} waypoints reached", flush=True
            ),
        )
        print(
            json.dumps(
                {
                    "directory": str(args.output_dir),
                    "methods": {
                        name: {
                            key: summary[key]
                            for key in (
                                "waypoints_completed",
                                "termination",
                                "physical_collision",
                                "nominal_blocked_fraction",
                            )
                        }
                        for name, summary in result.summary["methods"].items()
                    },
                },
                indent=2,
            )
        )
    if not args.no_render:
        video = render_comparison_video(
            result.trace,
            args.output_dir / f"navigation_comparison_{args.mode}.mp4",
            ComparisonRenderConfig(
                mode=args.mode,
                fps=args.fps,
                width=args.width,
                height=args.height,
                comparison_note=args.comparison_note,
                camera_distance=args.camera_distance,
                hover_camera_distance=args.hover_camera_distance,
                camera_transition_seconds=args.camera_transition_seconds,
            ),
        )
        print(json.dumps({"video": str(video.path), "frames": video.frame_count}, indent=2))


if __name__ == "__main__":
    main()
