"""Prepare or evaluate a shared competent library and render its recorded comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from crazyflow.safety.da_plcbf.competent_library_experiment import (
    CompetentExperimentConfig,
    prepare_competent_checkpoint,
    run_competent_experiment,
)
from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
    ComparisonRenderConfig,
    render_comparison_video,
)
from crazyflow.safety.da_plcbf.online_constant_wind import comparison_trace_for_methods
from examples.da_plcbf.plot_mechanism_comparison import plot_mechanism_comparison


def main() -> None:
    """Keep preparation, matched numerical execution, and rendering explicitly separable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--left-method", choices=("fixed", "compensated"), default="fixed")
    parser.add_argument("--mode", choices=("demo", "diagnostic"), default="demo")
    parser.add_argument("--probe-pause-time", type=float)
    args = parser.parse_args()
    config = CompetentExperimentConfig(
        **(json.loads(args.config.read_text()) if args.config else {})
    )
    device = jax.devices(args.device)[0]
    if args.command == "prepare":
        prepare_competent_checkpoint(config, args.output_dir, device)
        return
    result = run_competent_experiment(
        config,
        args.output_dir,
        checkpoint_stem=args.checkpoint,
        device=device,
        progress_callback=lambda method, current, total: print(
            f"{method}: {current}/{total}", flush=True
        ),
    )
    plot_mechanism_comparison(result, args.output_dir / "comparison.png")
    if not args.no_render:
        trace = comparison_trace_for_methods(result, args.left_method, "adaptive")
        render_comparison_video(
            trace,
            args.output_dir / "competent_comparison.mp4",
            ComparisonRenderConfig(
                mode=args.mode,
                probe_pause_time=args.probe_pause_time,
                probe_pause_seconds=2.0 if args.probe_pause_time is not None else 0.0,
            ),
        )
    print(
        json.dumps(
            {"directory": str(args.output_dir), "checks": result.summary["checks"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
