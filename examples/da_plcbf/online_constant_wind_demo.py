"""Run or render the corrected constant-wind DA-PLCBF mechanism demo."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path

import jax
import jax.numpy as jnp

from crazyflow.safety.da_plcbf.continuous_demo_scenarios import constant_wind_scenario
from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
    ComparisonRenderConfig,
    render_comparison_video,
)
from crazyflow.safety.da_plcbf.online_constant_wind import (
    OnlineConstantWindConfig,
    comparison_trace_for_methods,
    load_online_constant_wind_result,
    run_online_constant_wind_demo,
    save_online_constant_wind_result,
)


def _device(platform: str) -> jax.Device:
    devices = jax.devices(platform)
    if not devices:
        raise RuntimeError(f"no JAX {platform!r} device is available")
    return devices[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="simulate, learn online, save the trace, and render")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    run.add_argument("--no-render", action="store_true")
    run.add_argument("--fps", type=float, default=20.0)
    run.add_argument("--width", type=int, default=1600)
    run.add_argument("--height", type=int, default=900)
    run.add_argument("--config", type=Path, help="JSON overrides for OnlineConstantWindConfig")
    run.add_argument("--scenario", type=Path, help="JSON overrides for the wind scenario")
    run.add_argument("--policy-count", type=int)
    run.add_argument("--wind", type=float, nargs=3, metavar=("WX", "WY", "WZ"))
    run.add_argument("--learning-start", choices=("wind", "startup"))
    run.add_argument("--left-method", default="fixed")
    run.add_argument("--right-method", default="adaptive")
    run.add_argument("--probe-pause-time", type=float)
    run.add_argument("--probe-pause-seconds", type=float, default=0.0)

    render = subparsers.add_parser("render", help="render an already recorded trace")
    render.add_argument("--trace", type=Path, required=True)
    render.add_argument("--summary", type=Path, required=True)
    render.add_argument("--video", type=Path, required=True)
    render.add_argument("--fps", type=float, default=20.0)
    render.add_argument("--width", type=int, default=1600)
    render.add_argument("--height", type=int, default=900)
    render.add_argument("--left-method", default="fixed")
    render.add_argument("--right-method", default="adaptive")
    render.add_argument("--probe-pause-time", type=float)
    render.add_argument("--probe-pause-seconds", type=float, default=0.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    render_config = ComparisonRenderConfig(
        fps=args.fps,
        width=args.width,
        height=args.height,
        probe_pause_time=args.probe_pause_time,
        probe_pause_seconds=args.probe_pause_seconds,
    )
    if args.command == "render":
        result = load_online_constant_wind_result(args.trace, args.summary)
        trace = comparison_trace_for_methods(result, args.left_method, args.right_method)
        video = render_comparison_video(trace, args.video, render_config)
        print(json.dumps({"video": str(video.path), "frames": video.frame_count}, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = json.loads(args.config.read_text()) if args.config else {}
    if args.policy_count is not None:
        options["policy_count"] = args.policy_count
    if args.wind is not None:
        options["wind_after"] = tuple(args.wind)
    if args.learning_start is not None:
        options["learning_start"] = args.learning_start
    config = OnlineConstantWindConfig(**options)
    scenario = None
    if args.scenario:
        scenario = constant_wind_scenario()
        overrides = json.loads(args.scenario.read_text())
        allowed = {field.name for field in fields(scenario)}
        if unknown := set(overrides) - allowed:
            raise ValueError(f"unknown scenario fields: {sorted(unknown)}")
        for key, value in overrides.items():
            current = getattr(scenario, key)
            if hasattr(current, "dtype"):
                overrides[key] = jnp.asarray(value, dtype=current.dtype)
        scenario = replace(scenario, **overrides)
        scenario.validate()
    result = run_online_constant_wind_demo(config, device=_device(args.device), scenario=scenario)
    trace_path, summary_path = save_online_constant_wind_result(result, args.output_dir)
    output = {
        "trace": str(trace_path),
        "summary": str(summary_path),
        "objective_checks": result.summary,
    }
    if not args.no_render:
        video_path = args.output_dir / "online_constant_wind.mp4"
        trace = comparison_trace_for_methods(result, args.left_method, args.right_method)
        video = render_comparison_video(trace, video_path, render_config)
        output["video"] = str(video.path)
        output["video_frames"] = video.frame_count
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
