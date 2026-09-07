"""Authenticate secondary UNION refresh histories and all thirty-two fallback policies."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.da_plcbf_actuator_diagnostic_analysis import (  # noqa: E402
    _plain,
    arrays_digest,
    extract_pair,
)
from benchmark.da_plcbf_actuator_diagnostic_batch import (  # noqa: E402
    authenticate_campaign,
    planned_pairs,
    resolve_record,
    summarize,
)
from benchmark.da_plcbf_actuator_diagnostic_protocol import (  # noqa: E402
    analyze_pair,
    digest,
    file_digest,
)
from benchmark.da_plcbf_union_refresh_secondary import (  # noqa: E402
    PROTOCOL_ID,
    REALIZATIONS,
    audit_calls,
    load_secondary,
    require,
)

SERVICE_FIELDS = frozenset(
    {
        "fingerprint_and_dispatch_seconds",
        "delegated_controller_seconds",
        "wrapper_through_delegate_seconds",
    }
)
INDEXED_FIELDS = frozenset({"control_index", "time_seconds", "command_applied"})


def write_new(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(_plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def wrapper_prefix(calls: list[dict], event: float) -> dict:
    """Include all warmup/history state, excluding only noncausal service measurements."""
    prefix = [
        {key: value for key, value in row.items() if key not in SERVICE_FIELDS}
        for row in calls
        if row["warmup"] or row["time_seconds"] < event - 1e-9
    ]
    boundary = [
        row for row in calls if not row["warmup"] and abs(row["time_seconds"] - event) < 1e-9
    ]
    require(len(boundary) == 1, "wrapper needs one exact actual event-boundary call")
    return {
        "calls": prefix,
        "boundary_preceding_parameter_sha256": boundary[0]["preceding_call_parameter_sha256"],
        "boundary_requested_previous_index": boundary[0]["requested_previous_index"],
    }


def verify_previous_indices(calls: list[dict], selected: np.ndarray, applied: np.ndarray) -> None:
    """The raw request must equal the actual previously applied selection, starting at zero."""
    physical = [row for row in calls if not row["warmup"]]
    require(len(physical) == len(selected) == len(applied), "selection-history lengths differ")
    expected = 0
    for index, row in enumerate(physical):
        require(
            row["requested_previous_index"] == expected,
            "wrapper request differs from applied selection memory",
        )
        require(
            row["selected_index"] == int(selected[index]),
            "wrapper output index differs from controls",
        )
        if applied[index]:
            expected = int(selected[index])


def authenticate_wrapper(record: dict, proposal: dict, event: float) -> dict:
    episode = Path(record["episode_directory"])
    raw_path, indexed_path = (
        episode.parent / name for name in ("incumbent_audit.json", "incumbent_audit_indexed.json")
    )
    require(
        record["controller_override"] == proposal["controller_override"],
        "wrapper differs from sealed override",
    )
    require(
        record["actual_input_and_command_verification"] is True
        and record["wrapper_audit_error"] is None,
        "runtime wrapper audit was not successful",
    )
    for path in (raw_path, indexed_path):
        require(
            file_digest(path) == record["evidence_sha256"][str(path.resolve())],
            "wrapper evidence changed",
        )
    raw, indexed = (json.loads(path.read_text()) for path in (raw_path, indexed_path))
    require(
        raw
        == [
            {key: value for key, value in row.items() if key not in INDEXED_FIELDS}
            for row in indexed
        ],
        "indexed wrapper history differs from original raw capture",
    )
    with np.load(episode / "controls.npz", allow_pickle=False) as source:
        controls = {name: source[name] for name in source.files}
    checked = deepcopy(indexed)
    audit_calls(checked, controls)
    require(checked == indexed, "indexed wrapper capture does not match authenticated controls")
    verify_previous_indices(indexed, controls["selected_index"], controls["command_applied"])
    config = json.loads((episode / "binding.json").read_text())["config"]
    warmup = config["warmup_calls"]
    require(
        [row["warmup"] for row in indexed] == [True] * warmup + [False] * (len(indexed) - warmup),
        "wrapper warmup/physical boundary differs",
    )
    claim_path = Path(record["exclusive_flight_claim"])
    require(
        claim_path.resolve()
        == (
            Path(proposal["execution"]["claim_directory"]) / f"{record['trial']['trial_id']}.json"
        ).resolve(),
        "flight claim is outside sealed ledger",
    )
    require(
        file_digest(claim_path) == record["exclusive_flight_claim_sha256"], "flight claim changed"
    )
    claim = json.loads(claim_path.read_text())
    require(
        claim["trial_id"] == record["trial"]["trial_id"]
        and Path(claim["episode_directory"]).resolve() == episode.resolve(),
        "claim belongs to a different physical attempt",
    )
    prefix = wrapper_prefix(indexed, event)
    return {
        "authenticated": True,
        "prefix_sha256": digest(prefix),
        "prefix": prefix,
        "total_calls": len(indexed),
        "physical_calls": len(controls["time"]),
        "source_files_sha256": {
            str(path): file_digest(path) for path in (raw_path, indexed_path, claim_path)
        },
    }


def common_state_union(
    frozen_record: dict, adaptive_record: dict, *, times: tuple[float, ...] = (2.0, 2.4, 2.8)
) -> tuple[dict, dict]:
    """Both compared libraries include the identical immutable sixteen-policy F2 core."""
    import jax

    from benchmark.da_plcbf_actuator_confirmation import (
        _hash_tree,
        _recorded_inputs,
        authenticate_boundary_snapshot,
        load_saved_run,
    )
    from crazyflow.safety.da_plcbf.actuator_learning import load_actuator_learner_checkpoint
    from crazyflow.safety.da_plcbf.actuator_study import build_actuator_controller
    from crazyflow.safety.da_plcbf.continuous_version_a import (
        conservative_smooth_policy_values,
        runtime_policy_values,
    )

    frozen, adaptive = (
        load_saved_run(record["episode_directory"]) for record in (frozen_record, adaptive_record)
    )
    core_meta = adaptive_record["method_metadata"]["immutable_core"]
    require(
        core_meta == frozen_record["method_metadata"]["immutable_core"],
        "paired immutable core metadata differs",
    )
    core = load_actuator_learner_checkpoint(Path(core_meta["checkpoint"]))
    require(
        file_digest(core.npz_path) == core_meta["npz_sha256"]
        and file_digest(core.json_path) == core_meta["json_sha256"],
        "immutable F2 core checkpoint changed",
    )
    require(
        int(core.contract.spec.latent_codes.shape[0]) == 16, "immutable core has wrong cardinality"
    )
    fixed_stem = frozen.directory / "snapshots/critical-2.000000000"
    fixed = authenticate_boundary_snapshot(frozen, fixed_stem, 2.0).checkpoint
    rows, arrays, sources, functions = [], {}, {}, None
    for path in (
        core.npz_path,
        core.json_path,
        Path(f"{fixed_stem}.npz"),
        Path(f"{fixed_stem}.json"),
    ):
        sources[str(path)] = file_digest(path)
    for when in times:
        current_stem = adaptive.directory / f"snapshots/critical-{when:.9f}"
        current = authenticate_boundary_snapshot(adaptive, current_stem, when).checkpoint
        require(
            fixed.config == current.config
            and _hash_tree(fixed.contract.spec) == _hash_tree(current.contract.spec),
            "paired adaptive component specifications differ",
        )
        for path in (Path(f"{current_stem}.npz"), Path(f"{current_stem}.json")):
            sources[str(path)] = file_digest(path)
        for branch, run in (("frozen", frozen), ("adaptive", adaptive)):
            index = run.boundary_index(when)
            state, model, obstacles, _safety, _previous, goal, config = _recorded_inputs(
                run, index, when
            )
            if functions is None:
                functions = build_actuator_controller(
                    fixed.contract.spec,
                    fixed.config,
                    config,
                    frozen_library=(core.state.params, core.contract.spec, core.config),
                )
            key = f"t{when:.9f}.{branch}"
            shared = {
                "state": np.asarray(state),
                "goal": np.asarray(goal),
                "obstacle_centers": np.asarray(obstacles.centers),
                "obstacle_radii": np.asarray(obstacles.radii),
                "obstacle_mask": np.asarray(obstacles.mask),
            }
            shared.update(
                {
                    f"complete_model_leaf_{i}": np.asarray(value)
                    for i, value in enumerate(jax.tree.leaves(model))
                }
            )
            arrays.update({f"{key}.inputs.{name}": value for name, value in shared.items()})
            maxima, core_values = {}, {}
            for label, checkpoint in (("fault_frozen", fixed), ("adapted_current", current)):
                rollout = functions.candidates(state, checkpoint.state.params, goal, model)
                hard = runtime_policy_values(
                    rollout.states[..., :13],
                    obstacles,
                    obstacle_clearance=config.obstacle_clearance,
                    ego_radius=config.ego_radius,
                    envelope_derivative=True,
                )
                smooth = conservative_smooth_policy_values(
                    hard,
                    temperature=config.smooth_temperature,
                    max_gap_budget=config.smooth_gap_budget,
                )
                predicted, h, s = jax.device_get((rollout, hard.values, smooth))
                valid = np.asarray(predicted.valid)
                require(
                    len(valid) == 33,
                    "secondary common-state evaluation omitted core or adaptive policies",
                )
                prefix = f"{key}.{label}"
                named = {
                    "hard_values": np.asarray(h),
                    "smooth_values": np.asarray(s),
                    "rollout_valid": valid,
                    "prefix_displacement_m": np.asarray(
                        predicted.states[:, 20, :3] - predicted.states[:, 0, :3]
                    ),
                    "terminal_speed_mps": np.linalg.norm(
                        np.asarray(predicted.states[:, -1, 7:10]), axis=-1
                    ),
                    "first_command": np.asarray(predicted.commands[:, 0]),
                }
                arrays.update({f"{prefix}.{name}": value for name, value in named.items()})
                maxima[label] = float(np.max(np.where(valid[1:], np.asarray(h)[1:], -np.inf)))
                core_values[label] = arrays_digest(
                    {name: value[1:17] for name, value in named.items()}
                )
            require(
                core_values["fault_frozen"] == core_values["adapted_current"],
                "immutable core values differ on identical inputs",
            )
            union_h = max(maxima.values())
            rows.append(
                {
                    "time_seconds": when,
                    "physical_branch": branch,
                    "shared_inputs_sha256": arrays_digest(shared),
                    "frozen_max_fallback_hard_value": maxima["fault_frozen"],
                    "adaptive_max_fallback_hard_value": maxima["adapted_current"],
                    "diagnostic_union_max_fallback_hard_value": union_h,
                    "diagnostic_union_minus_fault_frozen": union_h - maxima["fault_frozen"],
                    "fallback_count_each": 32,
                    "immutable_core_values_exact_equal": True,
                    "actual_applied_command": run.controls["planned_command"][index].tolist()
                    if run.controls["command_applied"][index]
                    else None,
                    "actual_selected_policy": int(run.controls["selected_index"][index]),
                }
            )
    return {
        "status": "completed",
        "rows": rows,
        "shared_inputs_sha256": digest([row["shared_inputs_sha256"] for row in rows]),
        "per_skill_values_sha256": arrays_digest(arrays),
        "minimum_union_minus_frozen_hard_value": min(
            row["diagnostic_union_minus_fault_frozen"] for row in rows
        ),
        "source_files_sha256": sources,
        "fallback_count_each": 32,
        "index_semantics": "nominal0, immutableF2core1..16, adaptivecomponent17..32",
        "qualification": (
            "Hard values describe predicted spherical collision geometry and rollout validity; "
            "they do not imply QP or complete operational eligibility. Both libraries include "
            "the same immutable core. Actual commands remain separately authenticated."
        ),
    }, arrays


def analyze_stage(
    protocol: Path,
    proposal: dict,
    stage: str,
    directories: list[Path],
    output: Path,
    *,
    common_states: bool,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    ids = set(proposal["stage_trial_ids"][stage])
    campaigns, failures = [], []
    for directory in directories:
        try:
            campaigns.append(authenticate_campaign(directory.resolve(), protocol, stage, ids))
        except Exception as exc:
            failures.append(
                {"directory": str(directory), "type": type(exc).__name__, "error": str(exc)}
            )
    rows = []
    for pair in planned_pairs(proposal, stage):
        row = {
            "pair_id": pair["pair_id"],
            **pair["identity"],
            "declared_controller": "UNION_REFRESH",
            "status": "pending",
            "admissible_pair": False,
        }
        pair_output = output / pair["pair_id"]
        pair_output.mkdir()
        try:
            frozen, frozen_resolution = resolve_record(pair["arms"]["frozen"], campaigns)
            adaptive, adaptive_resolution = resolve_record(pair["arms"]["adaptive"], campaigns)
            row["record_resolution"] = {
                "frozen": frozen_resolution,
                "adaptive": adaptive_resolution,
            }
            row["retained_summaries"] = {
                "frozen": frozen["summary"],
                "adaptive": adaptive["summary"],
            }
            report, commands = extract_pair(
                Path(frozen["episode_directory"]),
                Path(adaptive["episode_directory"]),
                event_time=2.0,
                world_key=row["world_key"],
                library_seed=row["library_seed"],
                realization=row["realization"],
                runtime_binding_sha256=frozen["runtime_binding_sha256"],
            )
            auth_frozen, auth_adaptive = (
                authenticate_wrapper(record, proposal, 2.0) for record in (frozen, adaptive)
            )
            same = auth_frozen["prefix_sha256"] == auth_adaptive["prefix_sha256"]
            report["wrapper_authentication"] = {
                "frozen": auth_frozen,
                "adaptive": auth_adaptive,
                "exact_prefix_equal": same,
            }
            report["admissible_pair"] = report["admissible_pair"] and same
            require(
                report["frozen_record"]["postfault_publication_count"] == 0,
                "frozen arm published a post-event update",
            )
            if not report["admissible_pair"]:
                report["task_outcome"] = report["collision_free_outcome"] = None
                report["status"] = "inadmissible_original_or_wrapper_prefix"
            else:
                report["status"] = "secondary_outcomes_and_wrapper_authenticated_mechanism_pending"
            report["declared_controller"] = "UNION_REFRESH"
            report["unwrapped_runtime_method"] = "UNION"
            report["frozen_record"]["arm"] = "UNION_REFRESH_FREEZE_AT_FAULT"
            report["adaptive_record"]["arm"] = "UNION_REFRESH"
            write_new(pair_output / "outcome_report.json", report)
            np.savez_compressed(pair_output / "actual_commands.npz", **commands)
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
            row["wrapper_prefix_exact_equal"] = same
            row["outcome_report_sha256"] = file_digest(pair_output / "outcome_report.json")
            if common_states and report["admissible_pair"]:
                try:
                    common, values = common_state_union(frozen, adaptive)
                    common["actual_command_comparison_sha256"] = report[
                        "actual_command_comparison"
                    ]["extract_arrays_sha256"]
                    report["adaptive_record"]["common_state_evaluation"] = common
                    mechanism = analyze_pair(report["frozen_record"], report["adaptive_record"])
                    write_new(pair_output / "common_state_report.json", mechanism)
                    np.savez_compressed(pair_output / "common_states.npz", **values)
                    row["common_state_completed"] = True
                    row["status"] = "completed"
                except Exception as exc:
                    row["status"] = "authenticated_outcomes_common_state_failed"
                    row["common_state_error"] = {"type": type(exc).__name__, "error": str(exc)}
        except Exception as exc:
            row["status"] = (
                "unresolved_missing_record"
                if isinstance(exc, FileNotFoundError)
                else "rejected_evidence_or_runtime_outcome"
            )
            row["admissible_pair"] = False
            row["error"] = {"type": type(exc).__name__, "error": str(exc)}
        rows.append(row)
        write_new(pair_output / "pair.json", row)
        print(
            json.dumps(
                {
                    key: row.get(key)
                    for key in ("world_key", "cell_id", "realization", "status", "task_outcome")
                }
            ),
            flush=True,
        )
    report = {
        "schema": "secondary_union_refresh_analysis_v1",
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "protocol_file_sha256": file_digest(protocol),
        "summary": summarize(rows),
        "campaign_failures": failures,
        "rows": rows,
        "source_sha256": {
            str(Path(__file__)): file_digest(Path(__file__)),
            str(ROOT / "benchmark/da_plcbf_actuator_diagnostic_analysis.py"): file_digest(
                ROOT / "benchmark/da_plcbf_actuator_diagnostic_analysis.py"
            ),
        },
        "qualification": (
            "Separate secondary40 denominator. Matched histories include complete wrapper "
            "state. Freeze/continue changes learning and its parameter-version selection "
            "consequences; no attribution exclusively to improved individual policies."
        ),
    }
    write_new(output / "report.json", report)
    print(json.dumps(report["summary"]), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--common-states", action="store_true")
    args = parser.parse_args()
    proposal = load_secondary(args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    all_reports = {}
    for stage in ("fresh_validation", "numerical_robustness"):
        directories = (
            [args.results / stage]
            if stage == "fresh_validation"
            else [args.results / stage / realization for realization in REALIZATIONS]
        )
        all_reports[stage] = analyze_stage(
            args.protocol,
            proposal,
            stage,
            directories,
            args.output / stage,
            common_states=args.common_states,
        )
    write_new(
        args.output / "report.json",
        {
            "schema": "secondary_union_refresh_analysis_v1",
            "protocol_id": PROTOCOL_ID,
            "stages": {key: value["summary"] for key, value in all_reports.items()},
        },
    )


if __name__ == "__main__":
    main()
