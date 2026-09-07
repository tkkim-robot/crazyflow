"""Audit full matched histories, actual commands, and common-state repertoires.

The default command only reads saved evidence. ``--common-states`` additionally
evaluates fixed saved snapshots, with no flight, optimizer step or publication.
Wall-service measurements are explicitly excluded from deterministic physical
prefix identity; they remain in the bound source archives and timing summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from benchmark.da_plcbf_actuator_confirmation import BoundaryAuthentication, SavedActuatorRun

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.da_plcbf_actuator_diagnostic_protocol import (  # noqa: E402
    PREFIX_FIELDS,
    analyze_pair,
    digest,
    file_digest,
    four_outcomes,
)

NONCAUSAL_CONTROL_FIELDS = frozenset(
    {
        "controller_available_at",
        "controller_compute_seconds",
        "controller_seconds",
        "controller_deadline_met",
        "device_transfer_seconds",
        "learner_budget_seconds",
        "learner_deadline_met",
        "learner_estimated_seconds",
        "learner_seconds",
        "mandatory_host_seconds",
        "observation_seconds",
        "simulator_and_audit_seconds",
    }
)
NONCAUSAL_HISTORY_FIELDS = frozenset(
    {"started_wall_time", "completed_wall_time", "published_wall_time", "service_seconds"}
)
OPTIONAL_REPERTOIRE_LOGGING_FIELDS = frozenset({"candidate_states", "candidate_commands"})
BOUNDARY_FIELDS = (
    "controller_input_state",
    "controller_input_state_sha256",
    "observed_state",
    "observed_state_sha256",
    "estimated_effectiveness",
    "estimated_time_constants",
    "estimated_mass",
    "estimated_inertia",
    "estimated_model_sha256",
    "actual_effectiveness",
    "actual_time_constants",
    "actual_model_sha256",
)
DECISION_FIELDS = (
    "planned_command",
    "selected_index",
    "selected_row",
    "selected_bound",
    "hard",
    "smooth",
    "eligible",
    "qp_valid",
    "applied_passed",
    "applied_collision_margin",
    "applied_operational_margins",
    "control_params_sha256",
    "control_library_version",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def arrays_digest(values: dict[str, np.ndarray]) -> str:
    """Hash each named array's original dtype, shape and bytes, including empty arrays."""
    return digest(
        {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            }
            for name, value in sorted(values.items())
        }
    )


def _canonical_prefix_array(value: np.ndarray) -> np.ndarray:
    """Remove only future-dependent string padding; preserve every numeric dtype/bit."""
    if value.dtype.kind in "US":
        return np.asarray(value.tolist(), dtype=value.dtype.kind).reshape(value.shape)
    return value


def _history_without_service(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _history_without_service(item)
            for key, item in value.items()
            if key not in NONCAUSAL_HISTORY_FIELDS
        }
    if isinstance(value, list):
        return [_history_without_service(item) for item in value]
    return value


def checkpoint_groups(stem: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Authenticate numerical checkpoint groups without constructing a JAX executable."""
    metadata = json.loads(Path(f"{stem}.json").read_text())
    archive = Path(f"{stem}.npz")
    _require(file_digest(archive) == metadata["npz_sha256"], "checkpoint archive digest mismatch")
    with np.load(archive, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    _require(set(arrays) == set(metadata["arrays"]), "checkpoint array schema changed")
    for name, array in arrays.items():
        expected = metadata["arrays"][name]
        _require(
            array.dtype.str == expected["dtype"] and list(array.shape) == expected["shape"],
            f"checkpoint array dtype/shape changed: {name}",
        )

    def unpack(node: dict[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
        if node["kind"] == "array":
            return {prefix: arrays[node["key"]]}
        _require(node["kind"] == "dict", "unsupported numerical checkpoint node")
        result = {}
        for name, child in node["items"].items():
            result.update(unpack(child, f"{prefix}.{name}" if prefix else name))
        return result

    named = unpack(metadata["structure"])
    groups = {
        "complete_numerical_checkpoint": arrays_digest(named),
        "physical": arrays_digest({k: v for k, v in named.items() if k == "physical_state"}),
        "actor": arrays_digest(
            {
                k: v
                for k, v in named.items()
                if k.startswith(("state.params.", "state.previous_params."))
            }
        ),
        "optimizer_and_counters": arrays_digest(
            {
                k: v
                for k, v in named.items()
                if k.startswith("state.optimizer_state.")
                or k in {"state.library_version", "state.cumulative_gradient_steps"}
            }
        ),
        "learner_model": arrays_digest(
            {k: v for k, v in named.items() if k.startswith("state.latest_dynamics_estimate.")}
        ),
        "reference": digest(
            {
                "arrays_sha256": arrays_digest(
                    {k: v for k, v in named.items() if k.startswith("reference.")}
                ),
                "reference_actor_config": metadata["reference_actor_config"],
                "reference_learning_config": metadata["reference_learning_config"],
                "reference_sha256": metadata["reference_sha256"],
                "actor_config": metadata["config"],
            }
        ),
    }
    _require(any(k.startswith("state.optimizer_state.") for k in named), "missing Adam state")
    return metadata, groups


def prefix_arrays(run: SavedActuatorRun, when: float) -> dict[str, np.ndarray]:
    """Retain every recorded deterministic numerical field before the intervention."""
    result = {}
    for name, values in run.controls.items():
        if name not in NONCAUSAL_CONTROL_FIELDS | OPTIONAL_REPERTOIRE_LOGGING_FIELDS:
            result[f"controls.{name}"] = values[run.controls["time"] < when - 1e-9]
    for name, values in run.dense.items():
        mask = run.dense["time"] < when - 1e-9
        if name in {"time", "state", "native_state", "quaternion_norm"}:
            mask = run.dense["time"] <= when + 1e-9
        result[f"dense.{name}"] = values[mask]
    for name, values in run.applications.items():
        result[f"applications.{name}"] = values[run.applications["time"] < when - 1e-9]
    return {name: _canonical_prefix_array(value) for name, value in result.items()}


def _snapshot(run: SavedActuatorRun, when: float) -> Path:
    label = f"critical-{when:.9f}"
    path = run.summary.get("artifacts", {}).get(f"snapshot_{label}_npz")
    stem = Path(path).with_suffix("") if path else run.directory / "snapshots" / label
    _require(Path(f"{stem}.npz").is_file(), f"missing saved boundary checkpoint {stem}")
    return stem


def extract_prefix(
    run: SavedActuatorRun, when: float, authenticated: BoundaryAuthentication
) -> tuple[dict[str, str], dict[str, Any]]:
    """Bind physical, learner, teacher, controller, observations and update history."""
    metadata, checkpoint = checkpoint_groups(_snapshot(run, when))
    update_path = run.directory / "updates.json"
    updates = json.loads(update_path.read_text())
    before_updates = [row for row in updates if row["training_simulation_time"] < when - 1e-9]
    publications = [
        row
        for row in run.summary["snapshot_publications"]
        if row["published_simulation_time"] <= when + 1e-9
    ]
    boundary = authenticated.index
    observations = {key: run.controls[key][: boundary + 1] for key in BOUNDARY_FIELDS}
    controls = {
        key: run.controls[key][:boundary]
        for key in (
            "goal",
            "previous_command",
            "selected_index",
            "command_applied",
            "control_params_sha256",
        )
    }
    context = metadata["metadata"]
    captured = {
        key: context[key]
        for key in (
            "published_version",
            "snapshot_training_time_seconds",
            "pending_version",
            "control_params_reverted",
            "control_library_version",
            "control_params_sha256",
        )
    }
    numerical = prefix_arrays(run, when)
    history = {
        "updates": _history_without_service(before_updates),
        "publications": _history_without_service(publications),
        "snapshot_context": captured,
        "observation_tape_seed": run.binding["scene"]["scene_seed"],
        "observation_config": run.binding["config"]["observation_config"],
        "execution_mode": run.binding["config"]["execution_mode"],
    }
    fields = {
        "physical_state_and_motor_state_sha256": checkpoint["physical"],
        "actor_params_sha256": checkpoint["actor"],
        "optimizer_moments_and_step_sha256": checkpoint["optimizer_and_counters"],
        "teacher_retention_contract_sha256": checkpoint["reference"],
        "observations_and_current_model_sha256": digest(
            {
                "recorded_observations_sha256": arrays_digest(observations),
                "learner_model_sha256": checkpoint["learner_model"],
                "scene_geometry": run.binding["scene"]["world"],
            }
        ),
        "nominal_controller_and_filter_memory_sha256": digest(
            {
                "history_sha256": arrays_digest(controls),
                "filter_config": run.binding["config"]["filter_config"],
                "boundary_goal": run.controls["goal"][boundary].tolist(),
            }
        ),
        "scheduler_rng_and_publication_history_sha256": digest(history),
        "controls_dense_and_update_prefix_sha256": digest(
            {
                "arrays_sha256": arrays_digest(numerical),
                "updates": _history_without_service(before_updates),
            }
        ),
    }
    _require(set(fields) == set(PREFIX_FIELDS), "prefix hash contract drift")
    return fields, {
        "boundary_authentication": authenticated.report,
        "checkpoint_groups": checkpoint,
        "numerical_array_names": sorted(numerical),
        "source_control_string_storage_dtypes": {
            name: str(value.dtype)
            for name, value in run.controls.items()
            if value.dtype.kind in "US"
        },
        "prefix_string_rule": (
            "compare exact logical string values; canonicalize only Unicode/byte string "
            "padding determined by later episode records; preserve numeric dtypes exactly"
        ),
        "excluded_noncausal_control_fields": sorted(NONCAUSAL_CONTROL_FIELDS),
        "excluded_service_history_fields": sorted(NONCAUSAL_HISTORY_FIELDS),
        "optional_repertoire_logging_fields": sorted(OPTIONAL_REPERTOIRE_LOGGING_FIELDS),
        "qualification": (
            "wall-service values remain in the hashed archives; deterministic execution "
            "does not use them to gate updates"
        ),
        "update_prefix_count": len(before_updates),
        "publication_prefix_count": len(publications),
        "source_files": {str(update_path): file_digest(update_path)},
    }


def compare_prefix_arrays(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict:
    """Report the first differences as evidence instead of silently accepting tolerance."""
    checks = {}
    for name in sorted(left.keys() | right.keys()):
        if name not in left or name not in right:
            checks[name] = {"exact_equal": False, "reason": "field missing in one arm"}
            continue
        stored_a, stored_b = left[name], right[name]
        a, b = _canonical_prefix_array(stored_a), _canonical_prefix_array(stored_b)
        same = a.shape == b.shape and a.dtype == b.dtype
        equal = same and (
            np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)
        )
        difference = None
        if same and a.size and a.dtype.kind in "biufc":
            finite = np.isfinite(a) & np.isfinite(b)
            if np.any(finite):
                difference = float(
                    np.max(np.abs(a[finite].astype(float) - b[finite].astype(float)))
                )
        checks[name] = {
            "exact_equal": bool(equal),
            "maximum_finite_absolute_difference": difference,
            "left_storage_dtype": str(stored_a.dtype),
            "right_storage_dtype": str(stored_b.dtype),
            "string_padding_canonicalized": stored_a.dtype.kind in "US"
            or stored_b.dtype.kind in "US",
        }
    return {"exact_equal": all(row["exact_equal"] for row in checks.values()), "checks": checks}


def command_comparison(
    frozen: SavedActuatorRun, adaptive: SavedActuatorRun
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Use separately authenticated applications; compare only actually applied controls."""
    left = np.flatnonzero(frozen.applications["control_index"] >= 0)
    right = np.flatnonzero(adaptive.applications["control_index"] >= 0)
    times, ia, ib = np.intersect1d(
        frozen.applications["time"][left], adaptive.applications["time"][right], return_indices=True
    )
    left, right = left[ia], right[ib]
    _require(len(times) > 0, "no common actual command application times")
    extract = {"time": times}
    for prefix, run, indices in (("frozen", frozen, left), ("adaptive", adaptive, right)):
        control_indices = run.applications["control_index"][indices].astype(int)
        for key in ("command", "state", "control_index"):
            extract[f"{prefix}.{key}"] = run.applications[key][indices]
        for key in DECISION_FIELDS:
            extract[f"{prefix}.decision.{key}"] = run.controls[key][control_indices]
    delta = np.max(np.abs(extract["frozen.command"] - extract["adaptive.command"]), axis=1)
    different = np.flatnonzero(delta != 0)
    first = int(different[0]) if len(different) else None
    record = None
    if first is not None:
        record = {
            "time_seconds": float(times[first]),
            "maximum_motor_command_difference_n": float(delta[first]),
            "identical_physical_state": bool(
                np.array_equal(extract["frozen.state"][first], extract["adaptive.state"][first])
            ),
            "frozen_command": extract["frozen.command"][first].tolist(),
            "adaptive_command": extract["adaptive.command"][first].tolist(),
            "frozen_selected_policy": int(extract["frozen.decision.selected_index"][first]),
            "adaptive_selected_policy": int(extract["adaptive.decision.selected_index"][first]),
        }
    return {
        "common_actual_application_count": len(times),
        "equal_commands_before_first_divergence": first if first is not None else len(times),
        "first_divergence": record,
        "source": (
            "actual applications authenticated against planned control and dense physical state"
        ),
        "extract_arrays_sha256": arrays_digest(extract),
    }, extract


def _outcome_record(
    run: SavedActuatorRun, arm: str, when: float, context: dict, prefix: dict
) -> dict:
    summary = run.summary
    publications = [
        row
        for row in summary["snapshot_publications"]
        if row["training_simulation_time"] >= when - 1e-9
        and row["published_simulation_time"] >= when - 1e-9
    ]
    _require(type(summary["modeled_collider_collision"]) is bool, "collision outcome is unresolved")
    collision = summary["modeled_collider_collision"]
    full = bool(summary["reached_full_duration"])
    operational = not bool(summary["actual_operational_all_nodes_pass"])
    completed_task = summary["waypoints_completed"] == summary["waypoints_total"]
    return {
        **context,
        "arm": arm,
        "prefix": prefix,
        "status": summary["status"],
        "actual_collision": collision,
        "operational_violation": operational,
        "reached_full_duration": full,
        "safe_task_completion": completed_task and full and not collision and not operational,
        "postfault_publication_count": len(publications),
        "applied_commands_sha256": arrays_digest(run.applications),
    }


def authenticate_diagnostic_record(
    run: SavedActuatorRun, context: dict[str, Any]
) -> dict[str, Any]:
    """Verify runner evidence and immutable source copies against the enclosing binding."""
    from crazyflow.safety.da_plcbf.actuator_protocol import content_sha256

    record_path = run.directory.parent / "record.json"
    record = json.loads(record_path.read_text())
    _require(
        Path(record["episode_directory"]).resolve() == run.directory,
        "runner record points to another episode",
    )
    _require(record["summary"] == run.summary, "runner and retained episode summaries disagree")
    _require(
        record["runtime_binding_sha256"] == context["runtime_binding_sha256"],
        "supplied scientific runtime binding is not the recorded one",
    )
    trial = record["trial"]
    for key in ("world_key", "library_seed", "realization", "physical_spec_sha256"):
        _require(trial[key] == context[key], f"supplied context differs from sealed trial: {key}")
    _require(
        trial["arm"]["runtime_method"] == run.binding["config"]["method"],
        "runtime method differs from sealed arm",
    )
    _require(
        trial["arm"]["freeze_learning_at"] == run.binding["config"]["freeze_learning_at"],
        "runtime freeze differs from sealed arm",
    )
    binding_path = run.directory.parent.parent / "diagnostic_runtime_binding.json"
    binding = json.loads(binding_path.read_text())
    _require(
        content_sha256({k: v for k, v in binding.items() if k != "sha256"}) == binding["sha256"],
        "diagnostic runtime binding checksum mismatch",
    )
    _require(
        binding["sha256"] == record["campaign_binding_sha256"],
        "runner record refers to a different enclosing runtime binding",
    )
    scientific = content_sha256(
        {"source": binding["source_sha256"], "checkpoints": binding["checkpoint_files_sha256"]}
    )
    _require(
        scientific == binding["scientific_runtime_sha256"] == context["runtime_binding_sha256"],
        "scientific runtime source/checkpoint identity mismatch",
    )
    for name, expected in record["evidence_sha256"].items():
        _require(file_digest(Path(name)) == expected, f"saved episode evidence changed: {name}")
    _require(
        set(run.source_sha256).issubset(record["evidence_sha256"]),
        "runner evidence envelope omits consumed episode files",
    )
    for name, expected in binding["source_sha256"].items():
        relative = Path(name)
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            "runtime source path escapes its immutable copy",
        )
        copied = binding_path.parent / "source" / relative
        _require(file_digest(copied) == expected, f"archived runtime source changed: {name}")
    return {
        "record_sha256": file_digest(record_path),
        "binding_sha256": file_digest(binding_path),
        "scientific_runtime_sha256": scientific,
        "authenticated_episode_file_count": len(record["evidence_sha256"]),
        "authenticated_source_file_count": len(binding["source_sha256"]),
    }


def extract_pair(
    frozen_directory: Path,
    adaptive_directory: Path,
    *,
    event_time: float,
    world_key: str,
    library_seed: int,
    realization: str,
    runtime_binding_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Authenticate one declared freeze/continue pair; keep numerical mismatches explicit."""
    from benchmark.da_plcbf_actuator_confirmation import (
        authenticate_boundary_snapshot,
        load_saved_run,
    )

    frozen, adaptive = load_saved_run(frozen_directory), load_saved_run(adaptive_directory)
    a, b = frozen.binding["config"], adaptive.binding["config"]
    _require(
        a["execution_mode"] == b["execution_mode"] == "deterministic",
        "exact-prefix protocol requires deterministic execution; "
        "paced/asynchronous comparisons need their own histories",
    )
    _require(
        a["freeze_learning_at"] == event_time and b["freeze_learning_at"] is None,
        "arms do not implement freeze exactly at the actual fault boundary",
    )
    _require(
        frozen.summary.get("freeze_learning_actual_boundary") == event_time,
        "requested freeze was not applied at the actual event boundary",
    )
    _require(
        {k: v for k, v in a.items() if k not in {"freeze_learning_at", "retain_rollouts"}}
        == {k: v for k, v in b.items() if k not in {"freeze_learning_at", "retain_rollouts"}},
        "paired runtime configuration differs",
    )
    _require(
        frozen.binding["source_sha256"] == adaptive.binding["source_sha256"],
        "paired runtime source differs",
    )
    _require(
        frozen.binding["initial_learner_sha256"] == adaptive.binding["initial_learner_sha256"],
        "paired complete startup learners differ",
    )
    _require(frozen.binding["scene"] == adaptive.binding["scene"], "paired physical scenes differ")
    lhs = authenticate_boundary_snapshot(frozen, _snapshot(frozen, event_time), event_time)
    rhs = authenticate_boundary_snapshot(adaptive, _snapshot(adaptive, event_time), event_time)
    prefix_frozen, auth_frozen = extract_prefix(frozen, event_time, lhs)
    prefix_adaptive, auth_adaptive = extract_prefix(adaptive, event_time, rhs)
    physical = compare_prefix_arrays(
        prefix_arrays(frozen, event_time), prefix_arrays(adaptive, event_time)
    )
    field_matches = {key: prefix_frozen[key] == prefix_adaptive[key] for key in PREFIX_FIELDS}
    commands, extract = command_comparison(frozen, adaptive)
    context = {
        "world_key": world_key,
        "library_seed": library_seed,
        "realization": realization,
        "physical_spec_sha256": digest(frozen.binding["scene"]["physical_spec"]),
        "runtime_binding_sha256": runtime_binding_sha256,
    }
    runner_frozen = authenticate_diagnostic_record(frozen, context)
    runner_adaptive = authenticate_diagnostic_record(adaptive, context)
    frozen_record = _outcome_record(
        frozen, f"{a['method']}_FREEZE_AT_FAULT", event_time, context, prefix_frozen
    )
    adaptive_record = _outcome_record(adaptive, b["method"], event_time, context, prefix_adaptive)
    admissible = physical["exact_equal"] and all(field_matches.values())
    return {
        "status": "paired_outcomes_authenticated_mechanism_pending"
        if admissible
        else "inadmissible_unmatched_prefault_history",
        "admissible_pair": admissible,
        "task_outcome": four_outcomes(
            frozen_record["safe_task_completion"], adaptive_record["safe_task_completion"]
        )
        if admissible
        else None,
        "collision_free_outcome": four_outcomes(
            not frozen_record["actual_collision"], not adaptive_record["actual_collision"]
        )
        if admissible
        else None,
        "event_time_seconds": event_time,
        "prefix_field_matches": field_matches,
        "prefix_array_comparison": physical,
        "frozen_prefix_authentication": auth_frozen,
        "adaptive_prefix_authentication": auth_adaptive,
        "actual_command_comparison": commands,
        "runner_evidence_authentication": {"frozen": runner_frozen, "adaptive": runner_adaptive},
        "frozen_record": frozen_record,
        "adaptive_record": adaptive_record,
        "source_files_sha256": {
            **frozen.source_sha256,
            **adaptive.source_sha256,
            **auth_frozen["source_files"],
            **auth_adaptive["source_files"],
        },
    }, extract


def common_state_evaluation(
    frozen_directory: Path, adaptive_directory: Path, *, event_time: float, times: tuple[float, ...]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate fixed learned snapshots at both recorded branches with identical inputs."""
    import jax

    from benchmark.da_plcbf_actuator_confirmation import (
        _recorded_inputs,
        authenticate_boundary_snapshot,
        load_saved_run,
    )
    from crazyflow.safety.da_plcbf.actuator_study import build_actuator_controller
    from crazyflow.safety.da_plcbf.continuous_version_a import (
        conservative_smooth_policy_values,
        runtime_policy_values,
    )

    frozen, adaptive = load_saved_run(frozen_directory), load_saved_run(adaptive_directory)
    fixed = authenticate_boundary_snapshot(
        frozen, _snapshot(frozen, event_time), event_time
    ).checkpoint
    arrays, rows, source_files = {}, [], {}
    functions = None
    for when in times:
        current = authenticate_boundary_snapshot(
            adaptive, _snapshot(adaptive, when), when
        ).checkpoint
        _require(fixed.config == current.config, "common-state actor configurations differ")
        for branch, run in (("frozen", frozen), ("adaptive", adaptive)):
            index = run.boundary_index(when)
            state, model, obstacles, _safety, _previous, goal, config = _recorded_inputs(
                run, index, when
            )
            if functions is None:
                functions = build_actuator_controller(fixed.contract.spec, fixed.config, config)
            key = f"t{when:.9f}.{branch}"
            shared = {
                "state": np.asarray(state),
                "goal": np.asarray(goal),
                "obstacle_centers": np.asarray(obstacles.centers),
                "obstacle_radii": np.asarray(obstacles.radii),
                "obstacle_mask": np.asarray(obstacles.mask),
                "effectiveness": np.asarray(model.effectiveness),
                "time_constants": np.asarray(model.time_constants),
            }
            arrays.update({f"{key}.inputs.{name}": value for name, value in shared.items()})
            maxima = {}
            for label, checkpoint in (("fault_frozen", fixed), ("adapted_current", current)):
                batch = functions.candidates(state, checkpoint.state.params, goal, model)
                hard = runtime_policy_values(
                    batch.states[..., :13],
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
                values = jax.device_get((batch, hard.values, smooth))
                trajectories, h, smooth_values = values
                suffix = f"{key}.{label}"
                arrays[f"{suffix}.hard_values"] = np.asarray(h)
                arrays[f"{suffix}.smooth_values"] = np.asarray(smooth_values)
                arrays[f"{suffix}.rollout_valid"] = np.asarray(trajectories.valid)
                arrays[f"{suffix}.prefix_displacement_m"] = np.asarray(
                    trajectories.states[:, min(20, config.horizon), :3]
                    - trajectories.states[:, 0, :3]
                )
                arrays[f"{suffix}.terminal_speed_mps"] = np.linalg.norm(
                    np.asarray(trajectories.states[:, -1, 7:10]), axis=-1
                )
                arrays[f"{suffix}.first_command"] = np.asarray(trajectories.commands[:, 0])
                valid = np.asarray(trajectories.valid)[1:]
                maxima[label] = float(np.max(np.where(valid, np.asarray(h)[1:], -np.inf)))
            union = max(maxima.values())
            rows.append(
                {
                    "time_seconds": when,
                    "physical_branch": branch,
                    "shared_inputs_sha256": arrays_digest(shared),
                    "frozen_max_fallback_hard_value": maxima["fault_frozen"],
                    "adaptive_max_fallback_hard_value": maxima["adapted_current"],
                    "diagnostic_union_max_fallback_hard_value": union,
                    "diagnostic_union_minus_fault_frozen": union - maxima["fault_frozen"],
                    "actual_applied_command": run.controls["planned_command"][index].tolist()
                    if run.controls["command_applied"][index]
                    else None,
                    "actual_selected_policy": int(run.controls["selected_index"][index]),
                    "candidate_zero_is_nominal": True,
                }
            )
        for checkpoint in (fixed, current):
            for path in (checkpoint.npz_path, checkpoint.json_path):
                source_files[str(path)] = file_digest(path)
    _require(bool(rows), "at least one prespecified common-state time is required")
    report = {
        "status": "completed",
        "rows": rows,
        "shared_inputs_sha256": digest([row["shared_inputs_sha256"] for row in rows]),
        "per_skill_values_sha256": arrays_digest(arrays),
        "minimum_union_minus_frozen_hard_value": min(
            row["diagnostic_union_minus_fault_frozen"] for row in rows
        ),
        "union_definition": (
            "offline fault-frozen plus current-adapted values; "
            "runtime UNION instead preserves startup F2"
        ),
        "prefix_displacement_horizon_seconds": min(20, config.horizon) * config.dt,
        "qualification": (
            "candidate rollouts are common-state behavior predictions; actual commands "
            "come separately from authenticated application records"
        ),
        "source_files_sha256": source_files,
        "filter_config": asdict(config),
    }
    return report, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--event-time", type=float, required=True)
    parser.add_argument("--world-key", required=True)
    parser.add_argument("--library-seed", type=int, default=11)
    parser.add_argument("--realization", default="primary")
    parser.add_argument("--runtime-binding-sha256", required=True)
    parser.add_argument("--common-states", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report, commands = extract_pair(
        args.frozen,
        args.adaptive,
        event_time=args.event_time,
        world_key=args.world_key,
        library_seed=args.library_seed,
        realization=args.realization,
        runtime_binding_sha256=args.runtime_binding_sha256,
    )
    np.savez_compressed(args.output / "actual_commands.npz", **commands)
    if args.common_states and report["admissible_pair"]:
        common, arrays = common_state_evaluation(
            args.frozen,
            args.adaptive,
            event_time=args.event_time,
            times=(args.event_time, args.event_time + 0.4, args.event_time + 0.8),
        )
        common["actual_command_comparison_sha256"] = report["actual_command_comparison"][
            "extract_arrays_sha256"
        ]
        report["common_state_evaluation"] = common
        report["adaptive_record"]["common_state_evaluation"] = common
        report["protocol_pair_analysis"] = analyze_pair(
            report["frozen_record"], report["adaptive_record"]
        )
        report["status"] = "completed"
        np.savez_compressed(args.output / "common_states.npz", **arrays)
    report["analysis_source_sha256"] = file_digest(Path(__file__))
    (args.output / "report.json").write_text(
        json.dumps(_plain(report), sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "task_outcome": report["task_outcome"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
