"""Run fixed-profile empirical DA-PLCBF falsification and one-candidate GPU probes.

Examples::

    pixi run -e gpu-tests python examples/da_plcbf/falsify.py probe \
        --profile smoke --run-dir artifacts/da_plcbf/falsification-probe

    pixi run -e gpu-tests python examples/da_plcbf/falsify.py run \
        --profile smoke --run-dir artifacts/da_plcbf/falsification-smoke

The search is empirical and finite.  It automatically replays a bounded number of worst retained
tapes across all seven registered methods.  Counterexamples falsify the evaluated finite-margin
claim; an empty search does not prove safety, and matched descriptive replays are not a superiority
test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from crazyflow.safety.da_plcbf.falsification_experiments import (
    FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
    probe_falsification_candidate,
    run_da_plcbf_falsification,
    seven_method_replay_registry,
    verify_da_plcbf_falsification,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _run(arguments: argparse.Namespace) -> None:
    summary = run_da_plcbf_falsification(
        arguments.profile,
        arguments.run_dir.resolve(),
        root_seed=arguments.root_seed,
        repository=arguments.repository,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _probe(arguments: argparse.Namespace) -> None:
    root = arguments.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary = probe_falsification_candidate(
        arguments.profile,
        root / "cache",
        candidate_index=arguments.candidate_index,
        root_seed=arguments.root_seed,
    )
    destination = root / f"probe-{arguments.profile}-{arguments.candidate_index:04d}.json"
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != payload:
        raise FileExistsError(f"existing probe artifact differs: {destination}")
    destination.write_text(payload, encoding="utf-8")
    print(payload, end="")


def _registry(_arguments: argparse.Namespace) -> None:
    print(
        json.dumps(
            {
                "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
                "methods": list(seven_method_replay_registry()),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _verify(arguments: argparse.Namespace) -> None:
    summary = verify_da_plcbf_falsification(
        arguments.run_dir.resolve(),
        repository=arguments.repository,
        require_current_source=not arguments.allow_source_drift,
        require_complete=True,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run one complete fixed-budget profile")
    run.add_argument("--profile", choices=("smoke", "development", "final"), required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--root-seed", type=int, default=20260831)
    run.add_argument("--repository", type=Path)
    run.set_defaults(function=_run)

    probe = commands.add_parser("probe", help="run one real full-method randomized candidate")
    probe.add_argument("--profile", choices=("smoke", "development", "final"), default="smoke")
    probe.add_argument("--run-dir", type=Path, required=True)
    probe.add_argument("--candidate-index", type=int, default=0)
    probe.add_argument("--root-seed", type=int, default=20260831)
    probe.set_defaults(function=_probe)

    registry = commands.add_parser("registry", help="print the descriptive seven-method registry")
    registry.set_defaults(function=_registry)

    verify = commands.add_parser("verify", help="strictly reconstruct a completed campaign")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--repository", type=Path)
    verify.add_argument(
        "--allow-source-drift",
        action="store_true",
        help="inspect historical evidence without current-source promotion eligibility",
    )
    verify.set_defaults(function=_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    if argv is None:
        parser.print_help()
        return
    arguments = parser.parse_args(argv)
    arguments.function(arguments)


if __name__ == "__main__":
    main(sys.argv[1:])
