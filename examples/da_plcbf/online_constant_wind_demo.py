"""Run or render the corrected constant-wind DA-PLCBF mechanism demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
    ComparisonRenderConfig,
    render_comparison_video,
)
from crazyflow.safety.da_plcbf.online_constant_wind import (
    OnlineConstantWindConfig,
    load_online_constant_wind_trace,
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

    render = subparsers.add_parser("render", help="render an already recorded trace")
    render.add_argument("--trace", type=Path, required=True)
    render.add_argument("--summary", type=Path, required=True)
    render.add_argument("--video", type=Path, required=True)
    render.add_argument("--fps", type=float, default=20.0)
    render.add_argument("--width", type=int, default=1600)
    render.add_argument("--height", type=int, default=900)
    return parser


def main() -> None:
    args = _parser().parse_args()
    render_config = ComparisonRenderConfig(fps=args.fps, width=args.width, height=args.height)
    if args.command == "render":
        trace = load_online_constant_wind_trace(args.trace, args.summary)
        video = render_comparison_video(trace, args.video, render_config)
        print(json.dumps({"video": str(video.path), "frames": video.frame_count}, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_online_constant_wind_demo(OnlineConstantWindConfig(), device=_device(args.device))
    trace_path, summary_path = save_online_constant_wind_result(result, args.output_dir)
    output = {
        "trace": str(trace_path),
        "summary": str(summary_path),
        "objective_checks": result.summary,
    }
    if not args.no_render:
        video_path = args.output_dir / "online_constant_wind.mp4"
        video = render_comparison_video(result.trace, video_path, render_config)
        output["video"] = str(video.path)
        output["video_frames"] = video.frame_count
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
