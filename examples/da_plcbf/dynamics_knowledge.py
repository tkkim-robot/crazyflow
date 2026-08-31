"""Run or strictly verify the matched DA-PLCBF dynamics-knowledge campaign.

Examples:
    pixi run -e gpu-tests python examples/da_plcbf/dynamics_knowledge.py run \
        --profile smoke --output artifacts/da_plcbf/dynamics-knowledge-smoke

    pixi run -e gpu-tests python examples/da_plcbf/dynamics_knowledge.py verify \
        --output artifacts/da_plcbf/dynamics-knowledge-smoke

The oracle is a privileged, nondeployable current-boundary-truth reference.  It never reads future
truth.  R=4/R=8 denote runtime Cartesian particle sets and common hard admission; BPTT itself
remains the same point-model objective for every variant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from crazyflow.safety.da_plcbf.dynamics_knowledge_campaign import (
    DynamicsKnowledgeCampaignConfig,
    run_dynamics_knowledge_campaign,
    verify_dynamics_knowledge_campaign,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run or resume a matched closed-loop campaign")
    run.add_argument("--profile", choices=("smoke", "development", "final"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--root-seed", type=int, default=260831)
    run.add_argument("--fold-start", type=int, default=0)
    run.add_argument(
        "--trials",
        type=int,
        default=None,
        help="optional shortened schedule; shortened final runs are demoted",
    )
    run.add_argument("--no-resume", action="store_true")

    verify = commands.add_parser("verify", help="strictly verify an existing campaign")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument(
        "--allow-source-drift",
        action="store_true",
        help="verify stored evidence without requiring the current source tree to match",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    if argv is None:
        parser.print_help()
        return 0
    arguments = parser.parse_args(argv)
    payload: dict[str, Any]
    if arguments.command == "run":
        run = run_dynamics_knowledge_campaign(
            DynamicsKnowledgeCampaignConfig(
                profile=arguments.profile,
                root_seed=arguments.root_seed,
                fold_start=arguments.fold_start,
                trials=arguments.trials,
            ),
            arguments.output,
            resume=not arguments.no_resume,
        )
        payload = {
            "root": str(run.root),
            "expected_outcomes": run.expected_outcomes,
            "completed_outcomes": run.completed_outcomes,
            "failed_outcomes": run.failed_outcomes,
            "operational_failures": run.operational_failures,
            "execution_complete": run.execution_complete,
            "manifest_sha256": run.manifest_sha256,
            "scope": "closed_loop_dynamics_knowledge",
            "blanket_safety_superiority_supported": False,
        }
        status = 0 if run.execution_complete and run.failed_outcomes == 0 else 2
    else:
        verification = verify_dynamics_knowledge_campaign(
            arguments.output, require_current_source=not arguments.allow_source_drift
        )
        payload = {
            "valid": verification.valid,
            "errors": verification.errors,
            "expected_outcomes": verification.expected_outcomes,
            "retained_outcomes": verification.retained_outcomes,
            "completed_outcomes": verification.completed_outcomes,
            "failed_outcomes": verification.failed_outcomes,
            "operational_failures": verification.operational_failures,
            "scope": "closed_loop_dynamics_knowledge",
            "blanket_safety_superiority_supported": False,
        }
        status = 0 if verification.valid else 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
