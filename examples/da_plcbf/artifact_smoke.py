"""Create or validate a non-scientific DA-PLCBF artifact/replay smoke tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from crazyflow.safety.da_plcbf.artifact_smoke import create_synthetic_smoke_run
from crazyflow.safety.da_plcbf.artifacts import load_trace, validate_run_artifacts
from crazyflow.safety.da_plcbf.dashboard import render_dashboard

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] = ()) -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("output_parent", type=Path)
    create.add_argument("--run-id", default="synthetic-artifact-smoke-v1")
    validate = commands.add_parser("validate")
    validate.add_argument("run_directory", type=Path)
    validate.add_argument("--verify-replay", action="store_true")
    replay = commands.add_parser("replay")
    replay.add_argument("trace", type=Path)
    replay.add_argument("output", type=Path)
    replay.add_argument("--fps", type=float, default=10.0)
    if not argv:
        parser.print_help()
        return
    args = parser.parse_args(argv)

    if args.command == "create":
        run, result = create_synthetic_smoke_run(args.output_parent, run_id=args.run_id)
        print(run)
        print(result)
    elif args.command == "validate":
        print(validate_run_artifacts(args.run_directory, verify_replay=args.verify_replay))
    else:
        trace = load_trace(args.trace)
        print(render_dashboard(trace, args.output, fps=args.fps))


if __name__ == "__main__":
    main(sys.argv[1:])
