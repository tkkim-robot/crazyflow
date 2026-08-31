"""Run the bounded matched Version-A/Version-B full-stack evidence protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from crazyflow.safety.da_plcbf.version_b_evidence import (
    comparison_profile,
    load_version_comparison_artifact,
    run_matched_version_comparison,
    save_version_comparison_artifact,
    save_version_comparison_report,
)

REPOSITORY = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute matched, descriptive Version-A/Version-B evidence. This is separate from "
            "the seven-method scientific campaign and does not transfer guarantees."
        )
    )
    parser.add_argument("--profile", choices=("smoke", "final"), default="smoke")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--root-seed", type=int, default=260601478)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/da_plcbf/version-b-evidence-smoke.json")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Markdown path; defaults to the JSON output with a .md suffix",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--verify-artifact",
        type=Path,
        help="strictly verify a saved JSON artifact against the current source, then exit",
    )
    parser.add_argument(
        "--require-clean-source",
        action="store_true",
        help="reject execution/verification unless both recorded and current source are clean",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    if argv is None:
        parser.print_help()
        return
    args = parser.parse_args(argv)
    if args.verify_artifact is not None:
        source = args.verify_artifact.resolve()
        artifact = load_version_comparison_artifact(
            source,
            repository=REPOSITORY,
            require_current_source=True,
            require_clean_source=args.require_clean_source,
        )
        print(
            json.dumps(
                {
                    "artifact": str(source),
                    "content_sha256": artifact["content_sha256"],
                    "profile": artifact["protocol"]["profile"],
                    "scheduled_cases": artifact["summary"]["scheduled_cases"],
                    "current_source_verified": True,
                    "clean_source_required": args.require_clean_source,
                },
                sort_keys=True,
            )
        )
        return
    profile = comparison_profile(args.profile, root_seed=args.root_seed)
    artifact = run_matched_version_comparison(
        profile, device=args.device, require_clean_source=args.require_clean_source
    )
    output = args.output.resolve()
    report = (args.report or output.with_suffix(".md")).resolve()
    digest = save_version_comparison_artifact(
        artifact,
        output,
        overwrite=args.overwrite,
        repository=REPOSITORY,
        require_clean_source=args.require_clean_source,
    )
    save_version_comparison_report(
        artifact,
        report,
        overwrite=args.overwrite,
        repository=REPOSITORY,
        require_clean_source=args.require_clean_source,
    )
    print(
        json.dumps(
            {
                "artifact": str(output),
                "report": str(report),
                "content_sha256": digest,
                "summary": artifact["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
