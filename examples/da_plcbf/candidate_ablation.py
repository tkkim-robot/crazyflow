"""Run or verify the separate DA-PLCBF candidate-quality ablation campaign.

Examples:
    pixi run -e gpu-tests python examples/da_plcbf/candidate_ablation.py run \
        --profile smoke --output artifacts/da_plcbf/candidate-ablation-smoke

    pixi run -e gpu-tests python examples/da_plcbf/candidate_ablation.py verify \
        --output artifacts/da_plcbf/candidate-ablation-smoke

The generated artifacts are candidate-proposal evidence only.  They are intentionally not merged
with the seven-method closed-loop safety campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from crazyflow.safety.da_plcbf.ablation_campaign import (
    CANDIDATE_INFERENCE_BOUNDARY,
    CandidateCampaignConfig,
    run_candidate_ablation_campaign,
    verify_candidate_ablation_campaign,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run or resume a candidate-quality campaign")
    run.add_argument("--profile", choices=("smoke", "development", "confirmatory"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--root-seed", type=int, default=260831)
    run.add_argument("--fold-start", type=int, default=0)
    run.add_argument(
        "--folds",
        type=int,
        default=None,
        help="optional shortened schedule; shortened confirmatory runs are demoted",
    )
    run.add_argument("--no-resume", action="store_true")

    verify = commands.add_parser("verify", help="strictly verify an existing campaign")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument(
        "--allow-source-drift",
        action="store_true",
        help="verify stored bytes without requiring the current source tree to match",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    if argv is None:
        # Repository integration tests import examples and call ``main()`` directly.  Keep that
        # check fast and side-effect free while the real CLI remains strict about its subcommand.
        parser.print_help()
        return 0
    arguments = parser.parse_args(argv)
    payload: dict[str, Any]
    if arguments.command == "run":
        run = run_candidate_ablation_campaign(
            CandidateCampaignConfig(
                profile=arguments.profile,
                root_seed=arguments.root_seed,
                fold_start=arguments.fold_start,
                folds=arguments.folds,
            ),
            arguments.output,
            resume=not arguments.no_resume,
        )
        verification = verify_candidate_ablation_campaign(
            arguments.output, require_current_source=True
        )
        if not verification.valid:
            details = "; ".join(verification.errors[:4])
            raise RuntimeError(f"candidate-ablation post-run verification failed: {details}")
        payload = {
            "root": str(run.root),
            "expected_outcomes": run.expected_outcomes,
            "completed_outcomes": run.completed_outcomes,
            "failed_outcomes": run.failed_outcomes,
            "execution_complete": run.execution_complete,
            "manifest_sha256": run.manifest_sha256,
            "artifact_verification_valid": True,
            "scope": "candidate_quality_only",
            "safety_superiority_eligible": False,
            "candidate_quality_superiority_eligible": False,
            "inference_boundary": CANDIDATE_INFERENCE_BOUNDARY,
        }
        status = 0 if run.execution_complete and run.failed_outcomes == 0 else 2
    else:
        verification = verify_candidate_ablation_campaign(
            arguments.output, require_current_source=not arguments.allow_source_drift
        )
        payload = {
            "valid": verification.valid,
            "errors": verification.errors,
            "expected_outcomes": verification.expected_outcomes,
            "retained_outcomes": verification.retained_outcomes,
            "completed_outcomes": verification.completed_outcomes,
            "failed_outcomes": verification.failed_outcomes,
            "scope": "candidate_quality_only",
            "safety_superiority_eligible": False,
            "candidate_quality_superiority_eligible": False,
            "inference_boundary": CANDIDATE_INFERENCE_BOUNDARY,
        }
        status = 0 if verification.valid else 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
