"""Audit every sealed freeze/continue pair, retaining missing and rejected evidence.

The default CLI is read-only CPU analysis. ``--common-states`` also evaluates
saved boundary repertoires; it runs no flight, learning, or publication.
Output directories are exclusive so a partial analysis is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.da_plcbf_actuator_diagnostic_analysis import (  # noqa: E402
    _plain,
    common_state_evaluation,
    extract_pair,
)
from benchmark.da_plcbf_actuator_diagnostic_protocol import (  # noqa: E402
    OUTCOMES,
    analyze_pair,
    digest,
    file_digest,
    load_sealed,
)
from crazyflow.safety.da_plcbf.actuator_protocol import content_sha256  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_new(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(_plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def planned_pairs(proposal: dict, stage: str) -> list[dict]:
    """Form pairs from the sealed design, never from the observed successful runs."""
    selected = set(proposal["stage_trial_ids"][stage])
    groups: dict[str, dict] = {}
    for trial in proposal["trials"]:
        if trial["trial_id"] not in selected:
            continue
        arm = trial["arm"]
        identity = {key: value for key, value in trial.items() if key not in {"arm", "trial_id"}}
        identity["runtime_method"] = arm["runtime_method"]
        group = groups.setdefault(digest(identity), {"identity": identity, "arms": {}})
        role = "frozen" if arm["freeze_learning_at"] is not None else "adaptive"
        require(role not in group["arms"], "sealed stage contains duplicate pair roles")
        group["arms"][role] = trial
    require(bool(groups), "sealed stage has no trials")
    cells = {cell["cell_id"]: cell for cell in proposal["factorial_cells"]}
    pairs = []
    for key, group in groups.items():
        require(set(group["arms"]) == {"frozen", "adaptive"}, "sealed stage is not fully paired")
        frozen, adaptive = (group["arms"][role] for role in ("frozen", "adaptive"))
        require(
            frozen["arm"]["arm"] == f"{adaptive['arm']['arm']}_FREEZE_AT_FAULT",
            "sealed frozen arm does not match its adaptive parent",
        )
        event = cells[group["identity"]["cell_id"]]["event_time_seconds"]
        require(frozen["arm"]["freeze_learning_at"] == event, "sealed freeze/event mismatch")
        pairs.append({"pair_id": key, **group, "event_time_seconds": event})
    return sorted(
        pairs,
        key=lambda row: (
            row["identity"]["world_key"],
            row["identity"]["cell_id"],
            row["identity"]["runtime_method"],
            row["identity"]["realization"],
        ),
    )


def authenticate_campaign(directory: Path, protocol: Path, stage: str, ids: set[str]) -> dict:
    path = directory / "diagnostic_runtime_binding.json"
    binding = json.loads(path.read_text())
    require(
        content_sha256({k: v for k, v in binding.items() if k != "sha256"}) == binding["sha256"],
        "campaign checksum mismatch",
    )
    require(binding["protocol_file_sha256"] == file_digest(protocol), "campaign protocol differs")
    require(binding["stage"] == stage, "campaign belongs to a different sealed stage")
    require(set(binding["resolved_trial_ids"]).issubset(ids), "campaign declares unplanned trials")
    require(
        content_sha256(
            {"source": binding["source_sha256"], "checkpoints": binding["checkpoint_files_sha256"]}
        )
        == binding["scientific_runtime_sha256"],
        "campaign scientific binding checksum mismatch",
    )
    for name, expected in binding["source_sha256"].items():
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts, "source path escapes copy"
        )
        require(
            file_digest(directory / "source" / relative) == expected, "campaign source copy changed"
        )
    return {
        "directory": str(directory),
        "binding": binding,
        "binding_file_sha256": file_digest(path),
    }


def resolve_record(trial: dict, campaigns: list[dict]) -> tuple[dict, dict]:
    """A reused record must equal the record at the original actual episode parent."""
    candidates = []
    for campaign in campaigns:
        if trial["trial_id"] not in campaign["binding"]["resolved_trial_ids"]:
            continue
        path = Path(campaign["directory"]) / trial["trial_id"] / "record.json"
        if path.is_file():
            record = json.loads(path.read_text())
            require(record["trial"] == trial, "record is not the exact sealed trial")
            require(
                record["runtime_binding_sha256"]
                == campaign["binding"]["scientific_runtime_sha256"],
                "copied record scientific binding differs from receiving campaign",
            )
            candidates.append((path, record))
    if not candidates:
        raise FileNotFoundError(f"no retained record for declared trial {trial['trial_id']}")
    reference = candidates[0][1]
    require(
        all(record == reference for _, record in candidates), "conflicting retained trial records"
    )
    episode = Path(reference["episode_directory"]).resolve()
    original_path = episode.parent / "record.json"
    original = json.loads(original_path.read_text())
    require(original == reference, "copied record differs from its original episode-parent record")
    return reference, {
        "supplied_record_files_sha256": {str(path): file_digest(path) for path, _ in candidates},
        "original_record_file": str(original_path),
        "original_record_sha256": file_digest(original_path),
        "actual_episode_directory": str(episode),
        "original_source_campaign_directory": str(episode.parent.parent),
    }


def summarize(rows: list[dict]) -> dict:
    """Count every planned slot; unauthenticated/inadmissible rows receive no outcome credit."""
    accepted = [row for row in rows if row.get("admissible_pair") is True]
    return {
        "planned_pair_count": len(rows),
        "authenticated_matched_pair_count": len(accepted),
        "unresolved_or_inadmissible_pair_count": len(rows) - len(accepted),
        "pair_status_counts": dict(Counter(row["status"] for row in rows)),
        "task_outcomes": {
            name: sum(row.get("task_outcome") == name for row in accepted) for name in OUTCOMES
        },
        "collision_free_outcomes": {
            name: sum(row.get("collision_free_outcome") == name for row in accepted)
            for name in OUTCOMES
        },
        "completed_common_state_pair_count": sum(
            row.get("common_state_completed", False) for row in rows
        ),
    }


def run_batch(
    protocol: Path,
    stage: str,
    campaign_directories: list[Path],
    output: Path,
    *,
    common_states: bool = False,
) -> dict:
    proposal = load_sealed(protocol)
    pairs = planned_pairs(proposal, stage)
    ids = set(proposal["stage_trial_ids"][stage])
    output.mkdir(parents=True, exist_ok=False)
    campaigns, campaign_failures = [], []
    for directory in campaign_directories:
        try:
            campaigns.append(authenticate_campaign(directory.resolve(), protocol, stage, ids))
        except Exception as exc:
            campaign_failures.append(
                {"directory": str(directory), "error_type": type(exc).__name__, "error": str(exc)}
            )
    rows = []
    for pair in pairs:
        row = {
            "pair_id": pair["pair_id"],
            **pair["identity"],
            "event_time_seconds": pair["event_time_seconds"],
            "status": "pending",
            "admissible_pair": False,
        }
        pair_output = output / pair["pair_id"]
        pair_output.mkdir()
        try:
            frozen, frozen_auth = resolve_record(pair["arms"]["frozen"], campaigns)
            adaptive, adaptive_auth = resolve_record(pair["arms"]["adaptive"], campaigns)
            row["record_resolution"] = {"frozen": frozen_auth, "adaptive": adaptive_auth}
            row["retained_summaries"] = {
                "frozen": frozen["summary"],
                "adaptive": adaptive["summary"],
            }
            require(
                frozen["runtime_binding_sha256"] == adaptive["runtime_binding_sha256"],
                "paired scientific bindings differ",
            )
            report, commands = extract_pair(
                Path(frozen["episode_directory"]),
                Path(adaptive["episode_directory"]),
                event_time=pair["event_time_seconds"],
                world_key=row["world_key"],
                library_seed=row["library_seed"],
                realization=row["realization"],
                runtime_binding_sha256=frozen["runtime_binding_sha256"],
            )
            # A completed outcome still needs the actual zero-publication frozen contract.
            require(
                report["frozen_record"]["postfault_publication_count"] == 0,
                "frozen arm published a postfault update",
            )
            np.savez_compressed(pair_output / "actual_commands.npz", **commands)
            write_new(pair_output / "outcome_report.json", report)
            row.update(
                {
                    key: report[key]
                    for key in (
                        "status",
                        "admissible_pair",
                        "task_outcome",
                        "collision_free_outcome",
                        "actual_command_comparison",
                    )
                }
            )
            row["outcome_report_sha256"] = file_digest(pair_output / "outcome_report.json")
            if common_states and report["admissible_pair"]:
                try:
                    event = pair["event_time_seconds"]
                    common, values = common_state_evaluation(
                        Path(frozen["episode_directory"]),
                        Path(adaptive["episode_directory"]),
                        event_time=event,
                        times=(event, event + 0.4, event + 0.8),
                    )
                    common["actual_command_comparison_sha256"] = report[
                        "actual_command_comparison"
                    ]["extract_arrays_sha256"]
                    report["adaptive_record"]["common_state_evaluation"] = common
                    mechanism = analyze_pair(report["frozen_record"], report["adaptive_record"])
                    np.savez_compressed(pair_output / "common_states.npz", **values)
                    write_new(pair_output / "common_state_report.json", mechanism)
                    row["common_state_completed"] = True
                    row["common_state_report_sha256"] = file_digest(
                        pair_output / "common_state_report.json"
                    )
                    row["status"] = "completed"
                except Exception as exc:
                    row["status"] = "authenticated_outcomes_common_state_failed"
                    row["common_state_error"] = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
        except Exception as exc:
            row["status"] = (
                "pending_missing_record"
                if isinstance(exc, FileNotFoundError)
                else "rejected_evidence_or_runtime_outcome"
            )
            row["admissible_pair"] = False
            row["error"] = {"error_type": type(exc).__name__, "error": str(exc)}
        write_new(pair_output / "pair.json", row)
        rows.append(row)
        print(
            json.dumps(
                {
                    key: row.get(key)
                    for key in (
                        "world_key",
                        "cell_id",
                        "runtime_method",
                        "realization",
                        "status",
                        "task_outcome",
                    )
                }
            ),
            flush=True,
        )
    report = {
        "schema": "actuator_diagnostic_batch_analysis_v1",
        "stage": stage,
        "protocol_file": str(protocol.resolve()),
        "protocol_file_sha256": file_digest(protocol),
        "analysis_source_sha256": {
            str(path): file_digest(path)
            for path in (
                Path(__file__),
                ROOT / "benchmark/da_plcbf_actuator_diagnostic_analysis.py",
                ROOT / "benchmark/da_plcbf_actuator_confirmation.py",
            )
        },
        "common_states_requested": common_states,
        "common_state_times": (
            "actual event, event+0.4, event+0.8; no substituted or interpolated boundary"
        ),
        "campaigns": campaigns,
        "campaign_failures": campaign_failures,
        "summary": summarize(rows),
        "rows": rows,
        "qualification": (
            "All sealed pair slots retained. Only authenticated exact-prefix pairs receive "
            "four-way outcome counts. CPU-only reports keep common-state mechanism pending. "
            "No population or recursive safety inference."
        ),
    }
    write_new(output / "report.json", report)
    print(json.dumps(report["summary"]), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("matched_factorial", "fresh_validation", "numerical_robustness"),
        required=True,
    )
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--common-states", action="store_true")
    args = parser.parse_args()
    run_batch(
        args.protocol, args.stage, args.campaign, args.output, common_states=args.common_states
    )


if __name__ == "__main__":
    main()
