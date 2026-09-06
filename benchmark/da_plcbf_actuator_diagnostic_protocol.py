"""Seal the next diagnostic design and analyze authenticated matched arm outcomes.

This module deliberately does not execute flight episodes. ``seal`` resolves geometry
on CPU, records the predeclared splits and hashes inputs, then creates an exclusive
JSON envelope. Implementation/checkpoint bindings for the eventual runner are a
separate requirement: a sealed design is not evidence that any episode ran.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/da_plcbf/actuator-study-20260906/v1"
SCHEMA = "actuator_matched_diagnostic_proposal_v1"
DEVELOPMENT = (("structured", 30101), ("structured", 61001))
VALIDATION = tuple(("structured", 62001 + i) for i in range(4)) + tuple(
    ("navigation", 62101 + i) for i in range(4)
)
LIBRARY_SEED = 11
METHODS = {
    "F2": ("offline learned", "startup frozen", 16),
    "A": ("offline learned", "legacy persistent objective", 16),
    "A_BAL": ("offline learned", "balanced persistent objective", 16),
    "PD_F": ("handcrafted PD", "startup frozen", 16),
    "PD_A": ("handcrafted PD", "bounded maneuver parameter adaptation", 16),
    "DR": ("domain randomized learned", "startup frozen", 16),
    "UNION": ("immutable F2 plus balanced adaptive", "publish every finite update", 32),
    "F2_2K": ("immutable F2 plus immutable DR", "two distinct frozen libraries", 32),
}
OUTCOMES = ("both_succeed", "adaptation_alone_succeeds", "frozen_alone_succeeds", "both_fail")
PREFIX_FIELDS = (
    "physical_state_and_motor_state_sha256",
    "actor_params_sha256",
    "optimizer_moments_and_step_sha256",
    "teacher_retention_contract_sha256",
    "observations_and_current_model_sha256",
    "nominal_controller_and_filter_memory_sha256",
    "scheduler_rng_and_publication_history_sha256",
    "controls_dense_and_update_prefix_sha256",
)


def canonical(value: Any) -> bytes:
    """Return deterministic JSON bytes, refusing nonfinite numbers."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _resolve_geometry(family: str, seed: int) -> dict[str, Any]:
    # Lazy import keeps analysis and verification independent of JAX/GPU startup.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from crazyflow.safety.da_plcbf.actuator_study import make_actuator_scene

    scene = make_actuator_scene(seed, family, "effectiveness")
    physical = scene.physical_spec()
    physical.pop("actuator_events")
    return json.loads(json.dumps(physical))


def identify_old_harm(study: Path = STUDY) -> dict[str, Any]:
    """Select the first historical harmful trial; mark its outcome-dependent use."""
    paths = {
        "campaign": study / "main384-campaign-sealed-v1.json",
        "episodes": study / "main384-sealed-analysis-v1/episodes.csv",
        "results": study / "main384-sealed-results-v1/campaign_result.json",
    }
    campaign = json.loads(paths["campaign"].read_text())
    records = json.loads(paths["results"].read_text())["records"]
    with paths["episodes"].open() as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(r["world"], int(r["library_seed"]), r["method"]): r for r in rows}
    candidates = []
    for record in records:
        if record["method"] != "A":
            continue
        world = campaign["worlds"][record["world_index"]]
        if world["cell"] != "structured_effectiveness":
            continue
        key = (record["summary"]["physical_world_id"], record["library_seed"])
        adaptive, frozen = lookup[(*key, "A")], lookup[(*key, "F2")]
        if adaptive["actual_collision"] == "1" and frozen["actual_collision"] == "0":
            candidates.append((record["world_index"], record["library_seed"], record))
    if not candidates:
        raise ValueError("historical structured-effectiveness harmful pair not found")
    index, seed, selected = min(candidates, key=lambda item: item[:2])
    world = campaign["worlds"][index]
    selected_records = {
        r["method"]: r
        for r in records
        if r["world_index"] == index and r["library_seed"] == seed and r["method"] in {"A", "F2"}
    }
    artifacts = {}
    for method, record in selected_records.items():
        artifacts[method] = {}
        for role in ("binding", "summary", "controls", "applications", "dense", "updates"):
            path = Path(record["summary"]["artifacts"][role])
            artifacts[method][role] = {"path": str(path), "sha256": file_digest(path)}
    binding = json.loads(Path(artifacts["A"]["binding"]["path"]).read_text())
    checkpoints = {}
    for method in ("A", "F2", "DR"):
        stem = ROOT / campaign["checkpoints"][method][str(seed)]
        checkpoints[method] = {
            suffix: {
                "path": str(stem.with_suffix(suffix)),
                "sha256": file_digest(stem.with_suffix(suffix)),
            }
            for suffix in (".json", ".npz")
        }
    return {
        "selection_rule": (
            "minimum (old world_index, library_seed) among A-collision/F2-clear "
            "structured-effectiveness trials"
        ),
        "selection_uses_old_outcomes": True,
        "role": "development diagnosis only; not a held-out confirmation",
        "historical_harmful_trial_count": len(candidates),
        "world_index": index,
        "scene_seed": world["scene_seed"],
        "library_seed": seed,
        "physical_world_id": selected["summary"]["physical_world_id"],
        "scene": binding["scene"],
        "episode_config": campaign["episode_config"],
        "original_checkpoint_files": checkpoints,
        "outcomes": {
            method: {
                "collision": record["summary"]["collision"] is not None,
                "waypoints_completed": record["summary"]["waypoints_completed"],
                "task_completion_time_seconds": record["summary"]["task_completion_time_seconds"],
                "collision_time_seconds": (
                    record["summary"]["collision"]["first_intersection_time_seconds"]
                    if record["summary"]["collision"]
                    else None
                ),
            }
            for method, record in selected_records.items()
        },
        "read_only_artifact_files": artifacts,
        "source_files": {
            key: {"path": str(path), "sha256": file_digest(path)} for key, path in paths.items()
        },
    }


def factorial_cells() -> list[dict[str, Any]]:
    """Keep no-change once and cross the remaining declared physical factors."""
    cells = []
    for eta, lag, extra_lead in product((1.0, 0.85, 0.7), (1.0, 2.0), (0.0, 1.6)):
        if eta == lag == 1.0 and extra_lead:
            continue
        cells.append(
            {
                "cell_id": f"eta{eta:g}_lag{lag:g}_extra{extra_lead:g}",
                "effectiveness_after": [eta, eta, 1.0, 1.0],
                "lag_multipliers_after": [lag, lag, 1.0, 1.0],
                "event_time_seconds": round(2.0 - extra_lead, 8),
                "extra_lead_seconds": extra_lead,
                "no_change_control": eta == lag == 1.0,
            }
        )
    return cells


def _arm(label: str, event_time: float) -> dict[str, Any]:
    suffix = "_FREEZE_AT_FAULT"
    frozen_at_fault = label.endswith(suffix)
    return {
        "arm": label,
        "runtime_method": label.removesuffix(suffix),
        "freeze_learning_at": event_time if frozen_at_fault else None,
        "checkpoint_phase": "identical startup; complete common pre-fault history",
    }


def build_proposal(
    historical: dict[str, Any],
    *,
    resolver: Callable[[str, int], dict[str, Any]] = _resolve_geometry,
) -> dict[str, Any]:
    """Resolve fixed geometry and list every planned arm without observing new outcomes."""
    if historical["scene_seed"] != DEVELOPMENT[0][1] or historical["library_seed"] != LIBRARY_SEED:
        raise ValueError("historical selection differs from the fixed diagnostic anchor")
    worlds = {}
    for split, seeds in (("development", DEVELOPMENT), ("validation", VALIDATION)):
        for family, seed in seeds:
            geometry = resolver(family, seed)
            if "actuator_events" in geometry:
                raise ValueError("geometry must omit the separately varied fault schedule")
            key = f"{family}_{seed}"
            worlds[key] = {
                "world_key": key,
                "family": family,
                "scene_seed": seed,
                "split": split,
                "geometry": geometry,
                "geometry_sha256": digest(geometry),
                "selection": "historical harmful development anchor"
                if seed == 30101
                else "fixed seed before any new outcome",
            }
    if "scene" in historical:
        archived_geometry = deepcopy(historical["scene"]["physical_spec"])
        archived_geometry.pop("actuator_events")
        if archived_geometry != worlds["structured_30101"]["geometry"]:
            raise ValueError("resolved harmful geometry differs from the archived physical inputs")
    if len({world["geometry_sha256"] for world in worlds.values()}) != len(worlds):
        raise ValueError("development and validation must have distinct physical geometries")
    cells = factorial_cells()
    anchor_cells = [c for c in cells if c["cell_id"] in {"eta1_lag1_extra0", "eta0.7_lag1_extra0"}]
    trials = {}
    stage_members = {}

    def add(
        stage: str,
        world: dict[str, Any],
        cell: dict[str, Any],
        label: str,
        realization: str = "primary",
    ) -> None:
        arm = _arm(label, cell["event_time_seconds"])
        identity = {
            "world_key": world["world_key"],
            "cell_id": cell["cell_id"],
            "library_seed": LIBRARY_SEED,
            "arm": arm,
            "realization": realization,
        }
        trial_id = digest(identity)
        delta = {"perturb_plus": 1e-5, "perturb_minus": -1e-5}.get(realization, 0.0)
        physical = deepcopy(world["geometry"])
        physical["initial_state"][0] += delta
        physical["initial_state"][7] += delta
        physical["actuator_events"] = (
            []
            if cell["no_change_control"]
            else [
                {
                    "time_seconds": cell["event_time_seconds"],
                    "effectiveness": cell["effectiveness_after"],
                    "lag_multipliers": cell["lag_multipliers_after"],
                    "recovery_time_seconds": None,
                }
            ]
        )
        stage_members.setdefault(stage, []).append(trial_id)
        trials.setdefault(
            trial_id,
            {
                "trial_id": trial_id,
                **identity,
                "split": world["split"],
                "geometry_sha256": world["geometry_sha256"],
                "physical_spec_sha256": digest(physical),
                "initial_position_x_delta_m": delta,
                "initial_velocity_x_delta_mps": delta,
            },
        )

    for world in worlds.values():
        if world["split"] == "development":
            for cell, method in product(anchor_cells, METHODS):
                add("library_comparison", world, cell, method)
            for cell, arm in product(cells, ("A_BAL_FREEZE_AT_FAULT", "A_BAL")):
                add("matched_factorial", world, cell, arm)
        else:
            for cell, arm in product(
                anchor_cells, ("A_BAL_FREEZE_AT_FAULT", "A_BAL", "PD_A_FREEZE_AT_FAULT", "PD_A")
            ):
                add("fresh_validation", world, cell, arm)
    anchor = worlds["structured_30101"]
    fault = next(c for c in cells if c["cell_id"] == "eta0.7_lag1_extra0")
    for realization, arm in product(
        ("fresh_build_1", "fresh_build_2", "perturb_plus", "perturb_minus"),
        ("A_BAL_FREEZE_AT_FAULT", "A_BAL"),
    ):
        add("numerical_robustness", anchor, fault, arm, realization)
    return {
        "schema": SCHEMA,
        "status": "sealed_design_only_no_new_outcomes",
        "protocol_id": "quadrotor-harm-diagnosis-matched-v1",
        "prior_study_policy": (
            "read only; retain its splits, outcomes, archive, report and conclusions unchanged"
        ),
        "historical_development_anchor": historical,
        "selection": {
            "new_outcomes_consulted": False,
            "development": [list(world) for world in DEVELOPMENT],
            "validation": [list(world) for world in VALIDATION],
            "library_seeds": [LIBRARY_SEED],
            "validation_opening": (
                "after nominal PD validation and repair choice/source/checkpoints are bound;"
                " no tuning after validation outcomes"
            ),
            "validation_changes": (
                "retain all attempts; an amendment needs a new envelope and a genuinely new "
                "validation set"
            ),
            "inference": (
                "two prespecified within-world contrasts, one library seed; descriptive "
                "world-level evidence, not broad population superiority"
            ),
        },
        "methods": {
            key: {
                "initial_library": value[0],
                "update_rule": value[1],
                "policy_count": value[2],
                "adapter": "F2",
                "current_model_information": "same exact current point model",
            }
            for key, value in METHODS.items()
        },
        "shared_control": historical["episode_config"],
        "worlds": worlds,
        "factorial_cells": cells,
        "lead_time_definition": {
            "extra_lead_seconds": [0.0, 1.6],
            "fault_times_seconds": [2.0, 0.4],
            "command_period_seconds": 0.04,
            "rule": (
                "advance only the fault onset; preserve absolute obstacle paths, initial "
                "conditions, task, nominal controller and physical bounds"
            ),
            "qualification": (
                "physical pre-fault histories differ across lead cells; exact common history"
                " is required only within each freeze/continue pair"
            ),
            "also_record": (
                "first prescribed incoming-obstacle zero-crossing time minus fault onset; "
                "pre-threat published-update count"
            ),
        },
        "matched_history_contract": {
            "freeze_rule": (
                "same adaptive runtime method and full startup checkpoint in both arms; "
                "freeze_learning_at equals fault time only in frozen arm"
            ),
            "required_prefix_sha256_fields": list(PREFIX_FIELDS),
            "comparison_cut": (
                "immediately before the first control/learning operation at fault time; "
                "event and observation ordering identical"
            ),
            "prefix_reexecution": (
                "allowed only if every saved numerical prefix array and complete fault "
                "checkpoint match; otherwise pair is inadmissible"
            ),
            "all_finite_updates_published": True,
            "no_update_rejection_or_rollback": True,
        },
        "diagnostics": {
            "obstacle_free": (
                "use competent identical states first under no change and "
                "effectiveness-only; development/validation states remain separate"
            ),
            "learner": [
                "per-component and per-skill gradients",
                "Adam momentum inherited versus reset diagnostic",
                "last update versus accumulated updates",
                "prefix position/velocity error",
                "terminal braking speed",
                "auxiliary-gradient conflict and motor effort",
            ],
            "repertoire": [
                "all skills at common saved physical state, model and obstacle prediction",
                "frozen/adapted/union hard and smooth policy values",
                "early evasive displacement and braking",
                "coupled allocation and feasible maneuver witnesses",
            ],
            "executed_commands": [
                "authenticate actual applied motor commands against separate application log",
                "first exact matched-state command divergence",
                "selected policy, eligible set, QP row/bound and held-check outcome",
                "motor state and current-effectiveness actual force",
                "never substitute a candidate rollout for the executed command",
            ],
            "feasibility": (
                "failed hover trim or missing local witness does not prove every flight "
                "maneuver impossible"
            ),
            "union": (
                "pointwise max over immutable frozen plus adapted skills cannot decrease "
                "frozen coverage at identical inputs; no closed-loop guarantee"
            ),
            "equal_size_control": (
                "F2_2K uses distinct immutable F2 and DR policies; duplicated frozen rows "
                "are not a repertoire expansion"
            ),
            "compute": [
                "controller and learner service p50/p95/max",
                "published updates before each threat",
                "deadline misses and snapshot age",
                "union versus equal-total frozen cost",
                "paced contention and immutable publication test; reserve unchanged",
            ],
        },
        "primary_comparisons": [
            {"adaptive": "A_BAL", "frozen": "A_BAL_FREEZE_AT_FAULT"},
            {"adaptive": "PD_A", "frozen": "PD_A_FREEZE_AT_FAULT"},
        ],
        "outcome_definition": {
            "success": (
                "safe_task_completion: all waypoints, no collider contact, no physical "
                "operational violation, full common-duration exposure"
            ),
            "categories": list(OUTCOMES),
            "separate_metrics": [
                "actual collision",
                "operational violation",
                "timeout",
                "clearance over executed prefix",
                "actual published updates",
            ],
            "missing": (
                "unresolved, error, interrupted, unmatched prefix, or missing command "
                "evidence stays incomplete; never count as success or failure"
            ),
        },
        "numerical_robustness": {
            "selection": "fixed historical harmful geometry; no best-case or cache selection",
            "realizations": {
                "fresh_build_1": {
                    "compilation_cache": "new empty directory in a fresh process",
                    "initial_position_x_delta_m": 0.0,
                    "initial_velocity_x_delta_mps": 0.0,
                },
                "fresh_build_2": {
                    "compilation_cache": "another new empty directory in a fresh process",
                    "initial_position_x_delta_m": 0.0,
                    "initial_velocity_x_delta_mps": 0.0,
                },
                "perturb_plus": {
                    "compilation_cache": "fresh_build_1",
                    "initial_position_x_delta_m": 1e-5,
                    "initial_velocity_x_delta_mps": 1e-5,
                },
                "perturb_minus": {
                    "compilation_cache": "fresh_build_1",
                    "initial_position_x_delta_m": -1e-5,
                    "initial_velocity_x_delta_mps": -1e-5,
                },
            },
            "report": (
                "retain every physical outcome, command divergence and margin; any category "
                "change flags numerical fragility"
            ),
            "promoted_result": (
                "must additionally pass the same fixed subprotocol on its validation world; "
                "this requires an explicit bounded amendment if not already the anchor"
            ),
        },
        "execution_gate": {
            "runtime_binding_required": [
                "current source file hashes",
                "complete checkpoint and teacher hashes",
                "PD nominal competence report",
                "repair decision fixed before validation",
                "environment and compilation-cache identity",
            ],
            "runner_must_authenticate_geometry": True,
            "this_file_runs_gpu_episodes": False,
            "phase_order": [
                "harm/no-change reproduction and obstacle-free diagnosis",
                "nominal PD validation and fixed development comparison",
                "compute availability",
                "matched factorial",
                "fresh validation",
                "numerical robustness",
            ],
        },
        "budget": {
            "stage_episode_upper_bounds": {
                stage: len(members) for stage, members in stage_members.items()
            },
            "unique_episodes": len(trials),
            "uncollapsed_episode_upper_bound": sum(map(len, stage_members.values())),
            "rationale": (
                "two development geometries, eight fresh validation worlds and one fixed "
                "library seed; reuse exact duplicate trials; 148 maximum rather than another"
                " 384-episode benchmark"
            ),
        },
        "stage_trial_ids": stage_members,
        "trials": list(trials.values()),
    }


def seal(path: Path, proposal: dict[str, Any]) -> dict[str, Any]:
    """Bind the proposal and planner bytes without overwriting an existing envelope."""
    frozen = deepcopy(proposal)
    frozen["planner_files"] = {
        str(file): file_digest(file)
        for file in (
            Path(__file__).resolve(),
            ROOT / "tests/test_da_plcbf_actuator_diagnostic_protocol.py",
        )
    }
    envelope = {"schema": SCHEMA, "proposal": frozen, "sha256": digest(frozen)}
    _write_exclusive(path, envelope)
    return envelope


def load_sealed(path: Path, *, verify_files: bool = True) -> dict[str, Any]:
    """Reject changed proposal or bound planner bytes before downstream use."""
    envelope = json.loads(path.read_text())
    proposal = envelope["proposal"]
    if envelope.get("schema") != SCHEMA or envelope.get("sha256") != digest(proposal):
        raise ValueError("sealed proposal digest/schema mismatch")
    if verify_files:
        for name, expected in proposal["planner_files"].items():
            if file_digest(Path(name)) != expected:
                raise ValueError(f"planner file changed: {name}")
    return proposal


def four_outcomes(frozen_success: bool, adaptive_success: bool) -> str:
    """Classify actual declared success, never substituting collider-free timeout."""
    if type(frozen_success) is not bool or type(adaptive_success) is not bool:
        raise ValueError("success indicators must be explicit booleans")
    return OUTCOMES[(0 if frozen_success else 1) + (0 if adaptive_success else 2)]


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _success(record: dict[str, Any]) -> bool:
    if record.get("status") != "completed":
        raise ValueError("noncompleted attempt remains incomplete")
    fields = (
        "actual_collision",
        "operational_violation",
        "safe_task_completion",
        "reached_full_duration",
    )
    if any(type(record.get(key)) is not bool for key in fields):
        raise ValueError("completed outcomes require explicit boolean metrics")
    if record["safe_task_completion"] and (
        record["actual_collision"]
        or record["operational_violation"]
        or not record["reached_full_duration"]
    ):
        raise ValueError("safe task completion contradicts physical/full-exposure outcomes")
    if not record["reached_full_duration"] and not record["actual_collision"]:
        raise ValueError("noncollision partial exposure remains incomplete")
    count = record.get("postfault_publication_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("actual postfault publication count is required")
    if not _sha256(record.get("applied_commands_sha256")):
        raise ValueError("authenticated actual command evidence is required")
    return record["safe_task_completion"]


def analyze_pair(frozen: dict[str, Any], adaptive: dict[str, Any]) -> dict[str, Any]:
    """Require the same physical inputs and complete history before four-way mapping."""
    for key in (
        "world_key",
        "physical_spec_sha256",
        "library_seed",
        "realization",
        "runtime_binding_sha256",
    ):
        if key not in frozen or frozen[key] != adaptive.get(key):
            raise ValueError(f"unmatched paired field: {key}")
    for record in (frozen, adaptive):
        if not _sha256(record["physical_spec_sha256"]) or not _sha256(
            record["runtime_binding_sha256"]
        ):
            raise ValueError("physical and runtime bindings require SHA256 digests")
    if frozen.get("arm") != f"{adaptive.get('arm')}_FREEZE_AT_FAULT":
        raise ValueError("frozen arm must retain the same adaptive parent")
    for key in PREFIX_FIELDS:
        left = frozen.get("prefix", {}).get(key)
        right = adaptive.get("prefix", {}).get(key)
        if not _sha256(left) or left != right:
            raise ValueError(f"missing or unequal complete pre-fault history: {key}")
    frozen_success, adaptive_success = _success(frozen), _success(adaptive)
    if frozen["postfault_publication_count"] != 0:
        raise ValueError("freeze-at-fault arm published a postfault update")
    common_state = adaptive.get("common_state_evaluation", {})
    for key in (
        "shared_inputs_sha256",
        "per_skill_values_sha256",
        "actual_command_comparison_sha256",
    ):
        if not _sha256(common_state.get(key)):
            raise ValueError(f"missing common-state/actual-command evidence: {key}")
    gap = common_state.get("minimum_union_minus_frozen_hard_value")
    if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not math.isfinite(gap):
        raise ValueError("finite common-state union coverage difference is required")
    return {
        "world_key": frozen["world_key"],
        "library_seed": frozen["library_seed"],
        "realization": frozen["realization"],
        "frozen_arm": frozen["arm"],
        "adaptive_arm": adaptive["arm"],
        "task_outcome": four_outcomes(frozen_success, adaptive_success),
        "collision_free_outcome": four_outcomes(
            not frozen["actual_collision"], not adaptive["actual_collision"]
        ),
        "adaptive_postfault_publication_count": adaptive["postfault_publication_count"],
        "postfault_adaptation_available": adaptive["postfault_publication_count"] > 0,
        "complete_prefix_sha256": digest(frozen["prefix"]),
        "common_state_evaluation": common_state,
        "claims": (
            "pointwise repertoire and paired finite-horizon outcomes; no recursive or "
            "population safety guarantee"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("seal")
    prepare.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--protocol", type=Path, required=True)
    pair = commands.add_parser("analyze-pair")
    pair.add_argument("--frozen", type=Path, required=True)
    pair.add_argument("--adaptive", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        envelope = seal(args.output, build_proposal(identify_old_harm()))
        print(
            json.dumps(
                {
                    "path": str(args.output),
                    "sha256": envelope["sha256"],
                    "budget": envelope["proposal"]["budget"],
                }
            )
        )
    elif args.command == "verify":
        proposal = load_sealed(args.protocol)
        print(json.dumps({"verified": True, "budget": proposal["budget"]}))
    else:
        result = analyze_pair(
            json.loads(args.frozen.read_text()), json.loads(args.adaptive.read_text())
        )
        _write_exclusive(args.output, result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
