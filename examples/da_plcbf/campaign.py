"""Run, render, finalize, and validate reproducible DA-PLCBF campaigns.

Numerical execution and visualization are deliberately separate commands.  ``run`` writes only
immutable traces and progress artifacts; ``render`` replays selected completed traces offline.
Finalization is irreversible for a run directory because it writes a complete hash inventory.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from crazyflow.safety.da_plcbf.artifacts import (
    load_events,
    load_trace,
    review_contact_sheet_title,
    validate_campaign_visual_reviews,
    validate_run_artifacts,
    write_manifest,
    write_sha256sums,
)
from crazyflow.safety.da_plcbf.campaign_artifacts import (
    validate_current_source_tree,
    validate_persisted_campaign_evidence,
)
from crazyflow.safety.da_plcbf.dashboard_evidence import load_dashboard_evidence
from crazyflow.safety.da_plcbf.experiments import (
    REQUIRED_CONDITIONS,
    AdaptationExecutionMode,
    CampaignConfig,
    ExperimentConfig,
    run_campaign,
)
from crazyflow.safety.da_plcbf.scenarios import load_scenario_tape
from crazyflow.safety.da_plcbf.scientific_dashboard import (
    extract_keyframes,
    render_contact_sheet,
    render_scientific_dashboard,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

REPOSITORY = Path(__file__).resolve().parents[2]
CORE_METHODS = (
    "nominal_only",
    "analytic_cbf_hocbf",
    "fixed_fallback_pcbf",
    "handcrafted_fixed_library_plcbf",
    "offline_frozen_sdcbf_style",
    "da_plcbf_no_online_model_adaptation",
    "da_plcbf_full",
)


def _comma_tuple(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("comma-separated values must be nonempty and unique")
    return result


def _development_config(arguments: argparse.Namespace) -> ExperimentConfig:
    base = ExperimentConfig.final_defaults(random_seed=arguments.root_seed)
    if arguments.profile == "smoke":
        base = replace(
            base,
            control_steps=arguments.control_steps or 8,
            certificate_horizon=arguments.horizon or 2,
            # The first eight slots are immutable structural fallbacks.  K=16 is the
            # smallest useful smoke shape because it also contains adaptive slots and
            # therefore exercises a real, nonzero BPTT update.
            policy_count=arguments.policy_count or 16,
            training_scenario_count=arguments.training_scenarios or 2,
            bptt_burst_steps=arguments.bptt_steps or 1,
            adaptation_interval_steps=2,
            estimator_interval_steps=6,
            estimator_window_steps=3,
            validation_runtime_budget_seconds=120.0,
            validation_minimum_diversity=1e-8,
            realtime_pacing=True,
        )
    else:
        changes: dict[str, Any] = {}
        for argument_name, field_name in (
            ("control_steps", "control_steps"),
            ("horizon", "certificate_horizon"),
            ("policy_count", "policy_count"),
            ("training_scenarios", "training_scenario_count"),
            ("bptt_steps", "bptt_burst_steps"),
        ):
            value = getattr(arguments, argument_name)
            if value is not None:
                changes[field_name] = value
        base = replace(base, **changes)
    if arguments.adaptation_mode is not None:
        base = replace(base, adaptation_execution_mode=arguments.adaptation_mode)
    base.validate()
    return base


def _run(arguments: argparse.Namespace) -> None:
    destination = arguments.run_dir.resolve()
    if arguments.profile == "final":
        forbidden = (
            arguments.methods is not None,
            arguments.conditions is not None,
            arguments.trials is not None,
            arguments.control_steps is not None,
            arguments.horizon is not None,
            arguments.policy_count is not None,
            arguments.training_scenarios is not None,
            arguments.bptt_steps is not None,
            arguments.adaptation_mode is not None,
        )
        if any(forbidden):
            raise ValueError("the final profile cannot override its predeclared matrix or shapes")
        campaign = CampaignConfig.final_core(root_seed=arguments.root_seed)
    else:
        methods = arguments.methods or CORE_METHODS
        conditions = arguments.conditions or REQUIRED_CONDITIONS
        campaign = CampaignConfig(
            trial=_development_config(arguments),
            methods=methods,
            conditions=conditions,
            trials_per_condition=arguments.trials or 1,
            root_seed=arguments.root_seed,
        )
    result = run_campaign(
        campaign, output_directory=destination, resume=arguments.resume, repository=REPOSITORY
    )
    summary = {
        "run_directory": str(destination),
        "scheduled": len(result.records),
        "executed_this_invocation": len(result.trial_runs),
        "execution_complete": result.execution_complete,
        "scientific_claim_eligible": result.scientific_claim_eligible,
        "global_confirmatory_superiority_supported": (
            result.global_confirmatory_superiority_supported
        ),
        "claim_blockers": list(result.claim_blockers),
    }
    print(json.dumps(summary, sort_keys=True, indent=2))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _outcomes(root: Path) -> tuple[dict[str, Any], ...]:
    values = []
    for line in (root / "aggregate" / "outcomes.jsonl").read_text().splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("campaign outcome must be a JSON object")
        values.append(value)
    return tuple(values)


def _representative_outcomes(
    root: Path, methods: tuple[str, ...], conditions: tuple[str, ...], per_condition: int
) -> tuple[dict[str, Any], ...]:
    completed = [item for item in _outcomes(root) if item.get("status") == "complete"]
    selected: list[dict[str, Any]] = []
    for condition in conditions:
        for method in methods:
            candidates = [
                item
                for item in completed
                if item["assignment"]["condition"] == condition
                and item["assignment"]["method"] == method
            ]
            candidates.sort(
                key=lambda item: (
                    item["scientific_metrics"]["minimum_hard_margin"],
                    item["assignment"]["fold"],
                )
            )
            selected.extend(candidates[:per_condition])
    if not selected:
        raise ValueError("no completed outcomes match the requested video selection")
    return tuple(selected)


def _render(arguments: argparse.Namespace) -> None:
    root = arguments.run_dir.resolve()
    if (root / "manifest.json").exists() or (root / "SHA256SUMS").exists():
        raise ValueError("a finalized campaign cannot be modified")
    config = _read_json(root / "config.json")
    parameters = config.get("parameters")
    final_intended = (
        isinstance(parameters, dict) and parameters.get("intended_for_final_claim") is True
    )
    if final_intended:
        validate_current_source_tree(root, repository=REPOSITORY)
        if arguments.methods not in (None, ("da_plcbf_full",)):
            raise ValueError("a final campaign must render only da_plcbf_full")
        if arguments.conditions not in (None, REQUIRED_CONDITIONS):
            raise ValueError("a final campaign must render exactly the four required conditions")
        if arguments.videos_per_condition != 1:
            raise ValueError("a final campaign requires exactly one video per required condition")
        methods = ("da_plcbf_full",)
        conditions = REQUIRED_CONDITIONS
    else:
        methods = arguments.methods or (
            ("da_plcbf_full",) if "da_plcbf_full" in config["methods"] else (config["methods"][-1],)
        )
        conditions = arguments.conditions or tuple(config["conditions"])
    selected = _representative_outcomes(root, methods, conditions, arguments.videos_per_condition)
    videos = root / "videos"
    keyframes_root = root / "keyframes"
    sheets = root / "contact_sheets"
    for directory in (videos, keyframes_root, sheets):
        directory.mkdir(exist_ok=True)

    records = []
    for item in selected:
        assignment = item["assignment"]
        method = assignment["method"]
        condition = assignment["condition"]
        fold = int(assignment["fold"])
        stem = f"{method}--{condition}--fold-{fold:04d}"
        method_directory = root / "methods" / method / condition / str(fold)
        source_trace = method_directory / "trace.npz"
        trace = load_trace(source_trace)
        tape = load_scenario_tape(root / "scenario_tapes" / condition / f"{fold}.npz")
        sidecar = load_dashboard_evidence(method_directory / "dashboard_evidence.npz")
        events = load_events(method_directory / "events.jsonl", trace=trace)
        video_path = videos / f"{stem}.mp4"
        rendered = render_scientific_dashboard(
            trace,
            video_path,
            tape=tape,
            sidecar=sidecar,
            events=events,
            fps=arguments.fps,
            size=(arguments.width, arguments.height),
        )
        keyframes = extract_keyframes(
            video_path,
            trace,
            keyframes_root / stem,
            tape=tape,
            sidecar=sidecar,
            count=min(arguments.keyframes, trace.steps),
        )
        render_contact_sheet(
            keyframes,
            sheets / f"{stem}.png",
            title=review_contact_sheet_title(method, condition, fold),
        )
        validation = rendered.validation
        records.append(
            {
                "renderer": "scientific-dashboard-v1",
                "path": video_path.relative_to(root).as_posix(),
                "source_trace_path": source_trace.relative_to(root).as_posix(),
                "sha256": validation.file_sha256,
                "codec": validation.codec,
                "width": validation.width,
                "height": validation.height,
                "fps": validation.fps,
                "frame_count": validation.frame_count,
                "duration_seconds": validation.duration_seconds,
                "decoded_frames_sha256": validation.decoded_frames_sha256,
            }
        )
    index = root / "aggregate" / "video_records.json"
    if index.exists():
        raise FileExistsError(index)
    index.write_text(
        json.dumps({"schema_version": 1, "videos": records}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rendered": len(records), "video_records": str(index)}, indent=2))


def _finalize(arguments: argparse.Namespace) -> None:
    root = arguments.run_dir.resolve()
    validate_current_source_tree(root, repository=REPOSITORY)
    reconstructed = validate_persisted_campaign_evidence(root)
    index = _read_json(root / "aggregate" / "video_records.json")
    video_records = index["videos"]
    if not isinstance(video_records, list) or not video_records:
        raise ValueError("video record index must contain at least one video")
    evidence = reconstructed.scientific_claim_eligible
    validate_campaign_visual_reviews(
        root, video_records, require_all=True, require_final_core=evidence
    )
    status = "complete" if evidence else "incomplete"
    run_id = root.name
    replay = (
        "pixi run -e gpu-tests python examples/da_plcbf/campaign.py validate "
        f"--run-dir {root} --verify-replay"
    )
    write_manifest(
        root,
        run_id=run_id,
        status=status,
        scientific_evidence=evidence,
        replay_command=replay,
        video_records=video_records,
    )
    write_sha256sums(root)
    result = validate_run_artifacts(
        root, verify_replay=arguments.verify_replay, _validated_campaign=reconstructed
    )
    print(json.dumps(result, sort_keys=True, indent=2))


def _validate(arguments: argparse.Namespace) -> None:
    result = validate_run_artifacts(
        arguments.run_dir.resolve(), verify_replay=arguments.verify_replay
    )
    print(json.dumps(result, sort_keys=True, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run or resume numerical trials")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--profile", choices=("smoke", "development", "final"), default="smoke")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--root-seed", type=int, default=20260830)
    run.add_argument("--methods", type=_comma_tuple)
    run.add_argument("--conditions", type=_comma_tuple)
    run.add_argument("--trials", type=int)
    run.add_argument("--control-steps", type=int)
    run.add_argument("--horizon", type=int)
    run.add_argument("--policy-count", type=int)
    run.add_argument("--training-scenarios", type=int)
    run.add_argument("--bptt-steps", type=int)
    run.add_argument(
        "--adaptation-mode",
        choices=tuple(item.value for item in AdaptationExecutionMode),
        help=(
            "logical_simulation gives load-invariant safety traces; realtime_probe is "
            "hardware-feasibility evidence only"
        ),
    )
    run.set_defaults(function=_run)

    render = commands.add_parser("render", help="render representative immutable traces")
    render.add_argument("--run-dir", type=Path, required=True)
    render.add_argument("--methods", type=_comma_tuple)
    render.add_argument("--conditions", type=_comma_tuple)
    render.add_argument("--videos-per-condition", type=int, default=1)
    render.add_argument("--fps", type=float, default=15.0)
    render.add_argument("--width", type=int, default=1600)
    render.add_argument("--height", type=int, default=900)
    render.add_argument("--keyframes", type=int, default=8)
    render.set_defaults(function=_render)

    finalize = commands.add_parser("finalize", help="hash a visually reviewed campaign")
    finalize.add_argument("--run-dir", type=Path, required=True)
    finalize.add_argument("--verify-replay", action="store_true")
    finalize.set_defaults(function=_finalize)

    validate = commands.add_parser("validate", help="validate a finalized campaign")
    validate.add_argument("--run-dir", type=Path, required=True)
    validate.add_argument("--verify-replay", action="store_true")
    validate.set_defaults(function=_validate)
    return parser


def main(argv: Sequence[str] = ()) -> None:
    parser = _parser()
    if not argv:
        parser.print_help()
        return
    arguments = parser.parse_args(argv)
    arguments.function(arguments)


if __name__ == "__main__":
    main(sys.argv[1:])
