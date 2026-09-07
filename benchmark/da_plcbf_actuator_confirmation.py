"""Authenticate explicit actuator-study causal comparisons and shared-state audits.

This tool consumes complete saved episodes and authenticated sensing-boundary
checkpoints. It never selects cases, runs a new episode, publishes a learner,
changes a controller, or promotes an outcome. Counterfactual candidate audits
retain the complete rollout batch and original nominal objective; only the
other candidates' validity mask changes during offline branch enumeration.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crazyflow.safety.da_plcbf.actuator_experiment import _filter_record, _hash_tree, _jsonable
from crazyflow.safety.da_plcbf.actuator_learning import (
    actuator_reference_fingerprint,
    load_actuator_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig, actuator_plcbf_step
from crazyflow.safety.da_plcbf.actuator_protocol import content_sha256, physical_world_id
from crazyflow.safety.da_plcbf.actuator_study import (
    build_actuator_controller,
    nominal_actuator_model,
)
from crazyflow.safety.da_plcbf.navigation_world import NavigationWorld

KINDS = ("onset_freeze", "no_fault_control", "early_freeze_vs_revert", "late_reversion")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(_jsonable(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _same(left: Any, right: Any) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return (
        bool(np.array_equal(a, b, equal_nan=True))
        if a.dtype.kind in "fc"
        else bool(np.array_equal(a, b))
    )


def _losslessly_equal(left: Any, right: Any) -> bool:
    """Compare numerical values exactly after widening either storage dtype."""
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.dtype.kind not in "fc" or b.dtype.kind not in "fc":
        return _same(a, b)
    dtype = np.result_type(a.dtype, b.dtype)
    return _same(a.astype(dtype), b.astype(dtype))


def _authenticated_physical_dtype(array: Any, expected_hash: str) -> np.ndarray:
    """Recover a hash-authenticated original dtype after lossless archive stacking.

    P0 starts with a float64 scene vector and subsequently produces float32
    states. Stacking preserves all numerical values but widens later rows.
    A narrower representation is accepted only if it loses no values and its
    dtype/shape/bytes exactly reproduce the independently recorded original hash.
    """
    value = np.asarray(array)
    for dtype in (value.dtype, np.dtype("float32"), np.dtype("float64")):
        candidate = value.astype(dtype, copy=False)
        if _losslessly_equal(value, candidate) and _hash_tree(candidate) == expected_hash:
            return candidate
    raise ValueError(
        "physical state cannot reproduce its recorded hash with lossless dtype recovery"
    )


def _dense_node_index(times: np.ndarray, when: float) -> int:
    """Prefer the exact recorded timestamp over adjacent floating-point grid remnants."""
    matches = np.flatnonzero(times == when)
    if not len(matches):
        matches = np.flatnonzero(np.abs(times - when) <= 1e-9)
    _require(len(matches) == 1, "physical boundary is not one exact recorded dense node")
    return int(matches[0])


def _comparison(left: Any, right: Any) -> dict[str, Any]:
    a, b = np.asarray(left), np.asarray(right)
    same_shape = a.shape == b.shape
    numeric = a.dtype.kind in "biufc" and b.dtype.kind in "biufc"
    maximum = None
    if same_shape and numeric and a.size:
        finite = np.isfinite(a) & np.isfinite(b)
        if np.any(finite):
            maximum = float(np.max(np.abs(a[finite].astype(float) - b[finite].astype(float))))
    return {
        "exact_equal": _same(a, b),
        "left_shape": list(a.shape),
        "right_shape": list(b.shape),
        "left_dtype": str(a.dtype),
        "right_dtype": str(b.dtype),
        "maximum_finite_absolute_difference": maximum,
        "left_sha256": _hash_tree(a),
        "right_sha256": _hash_tree(b),
    }


def _tree_comparison(left: Any, right: Any) -> dict[str, Any]:
    a, shape_a = jax.tree_util.tree_flatten(left)
    b, shape_b = jax.tree_util.tree_flatten(right)
    equal_structure = shape_a == shape_b
    leaves = [_comparison(x, y) for x, y in zip(a, b, strict=True)] if equal_structure else []
    return {
        "exact_equal": equal_structure and all(row["exact_equal"] for row in leaves),
        "same_structure": equal_structure,
        "leaf_comparisons": leaves,
        "left_sha256": _hash_tree(left),
        "right_sha256": _hash_tree(right),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class SavedActuatorRun:
    """Read-only complete episode evidence, with hashes of every consumed file."""

    directory: Path
    binding: dict[str, Any]
    summary: dict[str, Any]
    dense: dict[str, np.ndarray]
    controls: dict[str, np.ndarray]
    applications: dict[str, np.ndarray]
    source_sha256: dict[str, str]
    campaign_envelope: dict[str, Any] | None = None
    storage_authentication: dict[str, Any] | None = None

    def boundary_index(self, when: float) -> int:
        """Require an actual recorded sensing boundary, not an interpolated grid tick."""
        matches = np.flatnonzero(np.abs(self.controls["time"] - when) <= 1e-9)
        _require(
            len(matches) == 1,
            f"{when} is not one unique actual sensing boundary in {self.directory}",
        )
        return int(matches[0])


@dataclass(frozen=True, slots=True)
class BoundaryAuthentication:
    """A full checkpoint proven to match its actual recorded control boundary."""

    checkpoint: Any
    index: int
    report: dict[str, Any]


def load_saved_run(directory: str | Path) -> SavedActuatorRun:
    """Load numerical evidence without regenerating a world or executing a policy."""
    directory = Path(directory).resolve()
    names = ("binding.json", "summary.json", "dense.npz", "controls.npz", "applications.npz")
    paths = [directory / name for name in names]
    for path in paths:
        _require(path.is_file(), f"missing episode artifact: {path}")
    binding, summary = (json.loads(path.read_text()) for path in paths[:2])
    _require(
        summary.get("physical_world_id")
        == binding["scene"].get("physical_world_id")
        == physical_world_id(binding["scene"]["physical_spec"]),
        "episode physical-world identity differs from its bound physical specification",
    )
    _require(
        summary.get("status") == "completed",
        "causal confirmation requires a completed saved run, including a recorded collision ending",
    )
    _require(
        summary.get("termination") in {"duration_complete", "physical_collision"},
        "incomplete numerical endings cannot authenticate a complete run",
    )
    arrays = []
    for path in paths[2:]:
        with np.load(path, allow_pickle=False) as archive:
            values = {name: archive[name] for name in archive.files}
        for value in values.values():
            value.flags.writeable = False
        arrays.append(values)
    dense, controls, applications = arrays
    _require(
        dense["state"].shape == (len(dense["time"]), 17),
        "dense states must include all four motor components",
    )
    _require(
        dense["time"][0] == 0 and np.all(np.diff(dense["time"]) > 0),
        "complete dense evidence must begin at time zero and increase strictly",
    )
    _require(np.all(np.isfinite(dense["state"])), "dense physical states must be finite")
    _require(
        math.isclose(
            float(dense["time"][-1]), float(summary["physical_time_seconds"]), abs_tol=1e-9
        ),
        "summary and physical trace ending disagree",
    )
    duration = float(binding["scene"]["world"]["config"]["duration_seconds"])
    _require(
        float(dense["time"][-1]) <= duration + 1e-9,
        "physical trace extends beyond the declared complete task",
    )
    if summary["termination"] == "duration_complete":
        _require(
            math.isclose(float(dense["time"][-1]), duration, abs_tol=1e-9),
            "duration_complete must include the entire declared task duration",
        )
    else:
        _require(
            isinstance(summary.get("collision"), dict),
            "a collision ending requires retained physical contact evidence",
        )
    initial_physical = _authenticated_physical_dtype(
        dense["state"][0], binding["initial_state_sha256"]
    )
    final_physical = _authenticated_physical_dtype(
        dense["state"][-1], summary["final_state_sha256"]
    )
    _require(
        np.all(np.diff(controls["time"]) > 0), "actual sensing timestamps must increase strictly"
    )
    _require(
        controls["actual_state"].shape == (len(controls["time"]), 17),
        "control records must contain complete physical states",
    )
    _require(
        np.all(np.diff(applications["time"]) >= 0),
        "actual command applications must have ordered physical times",
    )
    applied_indices = np.flatnonzero(controls["command_applied"])
    initialization = np.flatnonzero(applications["control_index"] < 0)
    if len(initialization):
        _require(
            np.array_equal(initialization, np.asarray([0]))
            and int(applications["control_index"][0]) == -1
            and float(applications["time"][0]) == 0
            and _losslessly_equal(applications["state"][0], initial_physical)
            and _losslessly_equal(applications["command"][0], initial_physical[13:]),
            "initial hover application must be the unique control_index=-1 record at time zero",
        )
    control_applications = np.flatnonzero(applications["control_index"] >= 0)
    _require(
        np.array_equal(applications["control_index"][control_applications], applied_indices),
        "command-applied flags do not exactly match actual command application records",
    )
    for application, control in zip(control_applications, applied_indices, strict=True):
        when = float(applications["time"][application])
        _require(
            when >= float(controls["time"][control]) - 1e-9
            and math.isclose(when, float(controls["command_applied_at"][control]), abs_tol=1e-9)
            and _losslessly_equal(
                applications["command"][application], controls["planned_command"][control]
            ),
            "actual application time/command differs from its planned control row",
        )
        physical_node = _dense_node_index(dense["time"], when)
        _require(
            _losslessly_equal(applications["state"][application], dense["state"][physical_node]),
            "command application is not bound to its exact physical dense state",
        )
    campaign = _campaign_envelope(directory, binding, summary)
    return SavedActuatorRun(
        directory,
        binding,
        summary,
        dense,
        controls,
        applications,
        {
            **{str(path): _sha(path) for path in paths},
            **({} if campaign is None else campaign["source_sha256"]),
        },
        campaign,
        {
            "dense_state_storage_dtype": str(dense["state"].dtype),
            "initial_physical_authenticated_dtype": str(initial_physical.dtype),
            "final_physical_authenticated_dtype": str(final_physical.dtype),
            "application_command_storage_dtype": str(applications["command"].dtype),
            "planned_command_storage_dtype": str(controls["planned_command"].dtype),
            "initial_hover_application_count": len(initialization),
            "command_application_comparison": (
                "exact numerical equality after lossless dtype promotion; "
                "storage dtypes are reported separately"
            ),
        },
    )


def _campaign_envelope(
    directory: Path, episode: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any] | None:
    """Authenticate an enclosing saved campaign when present, without selecting any case."""
    path = next(
        (
            parent / "campaign_binding.json"
            for parent in directory.parents
            if (parent / "campaign_binding.json").is_file()
        ),
        None,
    )
    if path is None:
        return None
    value = json.loads(path.read_text())
    _require(
        content_sha256({key: item for key, item in value.items() if key != "sha256"})
        == value.get("sha256"),
        "enclosing campaign binding checksum is invalid",
    )
    sources = {str(path): _sha(path)}
    current_matches = {}
    for relative, digest in value["source_sha256"].items():
        relative_path = Path(relative)
        _require(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            "campaign source keys must remain inside their source snapshot",
        )
        copied = path.parent / "source" / relative_path
        _require(
            copied.is_file() and _sha(copied) == digest,
            f"campaign source snapshot is missing or changed: {relative}",
        )
        sources[str(copied)] = digest
        current = ROOT / relative_path
        current_matches[relative] = current.is_file() and _sha(current) == digest
    for name, digest in episode["source_sha256"].items():
        relative = f"crazyflow/safety/da_plcbf/{name}"
        _require(
            value["source_sha256"].get(relative) == digest,
            f"episode source differs from its enclosing campaign: {name}",
        )
    _require(
        episode["checkpoint"]["checkpoint_sha256"] in value["checkpoint_sha256"].values(),
        "episode initial checkpoint is absent from the campaign binding",
    )
    record_path = path.parent / "campaign_result.json"
    if not record_path.is_file():
        record_path = path.parent / "campaign_progress.json"
    _require(record_path.is_file(), "enclosing campaign has no saved completion/progress records")
    saved = json.loads(record_path.read_text())
    expected = str((directory / "summary.json").relative_to(path.parent))
    matches = [row for row in saved["records"] if row["summary_path"] == expected]
    _require(
        len(matches) == 1 and matches[0]["summary"] == summary,
        "episode summary is not one exact recorded attempt in the enclosing campaign",
    )
    sources[str(record_path)] = _sha(record_path)
    return {
        "authenticated": True,
        "campaign_binding": str(path),
        "campaign_sha256": value["sha256"],
        "episode_summary_path": expected,
        "source_snapshot_file_count": len(value["source_sha256"]),
        "current_source_matches": current_matches,
        "all_current_campaign_sources_match": all(current_matches.values()),
        "source_sha256": sources,
    }


def authenticate_boundary_snapshot(
    run: SavedActuatorRun, checkpoint_path: str | Path, when: float
) -> BoundaryAuthentication:
    """Authenticate exact physical/learner/Adam state and the parameters actually used."""
    index = run.boundary_index(when)
    checkpoint = load_actuator_learner_checkpoint(checkpoint_path)
    meta = checkpoint.metadata
    _require(
        meta.get("capture_stage") == "sensing_boundary_after_publication"
        and meta.get("used_at_sensing_boundary") is True,
        "checkpoint is not an actual sensing-boundary snapshot after publication",
    )
    _require(
        math.isclose(float(meta.get("sensing_boundary_time_seconds", -1)), when, abs_tol=1e-9),
        "checkpoint sensing time differs from requested boundary",
    )
    _require(
        math.isclose(float(meta.get("physical_time_seconds", -1)), when, abs_tol=1e-9),
        "checkpoint physical time differs from requested boundary",
    )
    physical = np.asarray(checkpoint.physical_state)
    _require(
        _losslessly_equal(physical, run.controls["actual_state"][index]),
        "checkpoint does not preserve the exact recorded physical 17-state values",
    )
    _require(
        _hash_tree(physical)
        == meta.get("physical_state_sha256")
        == str(run.controls["actual_state_sha256"][index]),
        "checkpoint physical-state hash is not bound to the control record",
    )
    _require(
        _hash_tree(checkpoint.state) == meta.get("published_learner_sha256"),
        "checkpoint params/Adam/learner hash differs from the captured publication",
    )
    _require(
        int(checkpoint.state.library_version)
        == meta.get("published_version")
        == int(run.controls["library_version"][index]),
        "checkpoint published version differs from the version used at sensing",
    )
    _require(
        int(checkpoint.state.cumulative_gradient_steps)
        == int(run.controls["cumulative_gradient_steps"][index]),
        "checkpoint Adam counter differs from the recorded published counter",
    )
    _require(
        meta.get("control_library_version") == int(run.controls["control_library_version"][index]),
        "checkpoint control-library version differs from the actual control row",
    )
    control_hash = str(run.controls["control_params_sha256"][index])
    _require(
        meta.get("control_params_sha256") == control_hash,
        "checkpoint parameter-use hash differs from the actual control row",
    )
    reverted = bool(run.controls["control_params_reverted"][index])
    _require(
        meta.get("control_params_reverted") is reverted,
        "checkpoint parameter-reversion flag differs from the control row",
    )
    expected_params = (
        run.binding["initial_params_sha256"] if reverted else _hash_tree(checkpoint.state.params)
    )
    _require(
        control_hash == expected_params,
        "the control row does not use the authenticated original/available parameter snapshot",
    )
    _require(
        meta.get("checkpoint_sha256") == run.binding["checkpoint"]["checkpoint_sha256"],
        "boundary snapshot refers to a different initial checkpoint",
    )
    dense_index = _dense_node_index(run.dense["time"], float(run.controls["time"][index]))
    _require(
        _losslessly_equal(physical, run.dense["state"][dense_index]),
        "boundary snapshot is not bound to one exact dense physical node",
    )
    source = {
        str(checkpoint.npz_path.resolve()): _sha(checkpoint.npz_path),
        str(checkpoint.json_path.resolve()): _sha(checkpoint.json_path),
    }
    report = {
        "authenticated": True,
        "time_seconds": when,
        "actual_control_index": index,
        "capture_stage": meta["capture_stage"],
        "physical_state_sha256": _hash_tree(physical),
        "physical_state_dtype": str(physical.dtype),
        "control_state_storage_dtype": str(run.controls["actual_state"].dtype),
        "dense_state_storage_dtype": str(run.dense["state"].dtype),
        "physical_values_exact_after_lossless_storage_promotion": True,
        "published_learner_sha256": _hash_tree(checkpoint.state),
        "published_params_sha256": _hash_tree(checkpoint.state.params),
        "optimizer_state_sha256": _hash_tree(checkpoint.state.optimizer_state),
        "published_library_version": int(checkpoint.state.library_version),
        "control_library_version": int(run.controls["control_library_version"][index]),
        "control_params_sha256": control_hash,
        "control_params_reverted": reverted,
        "source_sha256": source,
    }
    return BoundaryAuthentication(checkpoint, index, report)


def _world_identity(run: SavedActuatorRun) -> dict[str, Any]:
    world = run.binding["scene"]["world"]
    fields = (
        "initial_state",
        "waypoint_positions",
        "obstacle_mean_centers",
        "obstacle_amplitudes",
        "obstacle_angular_frequencies",
        "obstacle_phases",
        "obstacle_radii",
    )
    return {
        **{name: world[name] for name in fields},
        "world_config": {name: value for name, value in world["config"].items() if name != "seed"},
        "navigation_start": run.binding["scene"]["navigation_start"],
    }


def _has_fault(run: SavedActuatorRun) -> bool:
    scene = run.binding["scene"]
    return not (
        np.all(np.asarray(scene["effectiveness_after"]) == 1)
        and np.all(np.asarray(scene["lag_multipliers_after"]) == 1)
    )


def _intervention_contract(
    left: SavedActuatorRun,
    right: SavedActuatorRun,
    kind: str,
    through: float,
    reference_time: float | None,
) -> dict[str, Any]:
    _require(kind in KINDS, "unknown causal comparison kind")
    _require(
        _world_identity(left) == _world_identity(right),
        "the pair changes geometry, task, initial body or operational constraints",
    )
    _require(
        left.binding["checkpoint"]["checkpoint_sha256"]
        == right.binding["checkpoint"]["checkpoint_sha256"],
        "the pair must start from the same initial learner checkpoint",
    )
    _require(
        left.binding["initial_learner_sha256"] == right.binding["initial_learner_sha256"],
        "initial params/Adam states differ",
    )
    _require(
        left.binding["config"].keys() == right.binding["config"].keys(),
        "paired configuration schemas differ",
    )
    for name in left.binding["config"].keys() - {"freeze_learning_at", "revert_control_params_at"}:
        _require(
            left.binding["config"][name] == right.binding["config"][name], f"paired {name} differs"
        )
    _require(
        left.binding.get("source_sha256") == right.binding.get("source_sha256"),
        "paired episode source hashes differ",
    )
    a, b = left.binding["config"], right.binding["config"]
    if kind != "no_fault_control":
        _require(a["method"] in {"A", "A1"}, "learning interventions require an online learner")
    if kind == "onset_freeze":
        _require(
            left.summary["physical_world_id"] == right.summary["physical_world_id"],
            "onset freeze must keep the same physical fault schedule",
        )
        _require(
            math.isclose(through, float(right.binding["scene"]["event_time"]), abs_tol=1e-9),
            "onset freeze time must equal the actual prescribed fault onset",
        )
        _require(
            a["freeze_learning_at"] is None and b["freeze_learning_at"] == through,
            "onset freeze requires continuing learning on the left "
            "and freeze-at-onset on the right",
        )
        _require(
            a["revert_control_params_at"] is None and b["revert_control_params_at"] is None,
            "onset freeze must not revert parameters",
        )
        _require(
            right.summary.get("freeze_learning_actual_boundary") is not None
            and math.isclose(
                right.summary["freeze_learning_actual_boundary"], through, abs_tol=1e-9
            ),
            "learning freeze did not occur at the prescribed actual onset boundary",
        )
    elif kind == "no_fault_control":
        _require(
            _has_fault(left) and not _has_fault(right),
            "no-fault control requires the faulted run on the left "
            "and nominal actuator schedule on the right",
        )
        _require(
            math.isclose(through, float(left.binding["scene"]["event_time"]), abs_tol=1e-9),
            "the common prefix ends at the fault onset",
        )
        _require(
            a["freeze_learning_at"] == b["freeze_learning_at"]
            and a["revert_control_params_at"] == b["revert_control_params_at"],
            "no-fault control cannot change learning interventions",
        )
        for name in ("event_time", "recovery_time"):
            _require(
                left.binding["scene"].get(name) == right.binding["scene"].get(name),
                f"no-fault control must retain the prescribed {name}",
            )
    else:
        _require(
            left.summary["physical_world_id"] == right.summary["physical_world_id"],
            "parameter reversion must retain the exact physical schedule",
        )
        _require(
            a["freeze_learning_at"] == b["freeze_learning_at"]
            and a["freeze_learning_at"] is not None
            and a["freeze_learning_at"] <= through,
            "freeze and reversion must share a learning freeze at or before reversion",
        )
        _require(
            left.summary.get("freeze_learning_actual_boundary")
            == right.summary.get("freeze_learning_actual_boundary")
            and left.summary.get("freeze_learning_actual_boundary") is not None
            and left.summary["freeze_learning_actual_boundary"] <= through + 1e-9,
            "freeze/revert pair did not execute the same actual learning freeze before reversion",
        )
        _require(
            a["revert_control_params_at"] is None and b["revert_control_params_at"] == through,
            "the left must retain learned parameters "
            "and the right must revert at the stated boundary",
        )
        _require(
            right.summary.get("revert_control_params_actual_boundary") is not None
            and math.isclose(
                right.summary["revert_control_params_actual_boundary"], through, abs_tol=1e-9
            ),
            "reversion did not occur at the stated actual sensing boundary",
        )
        if reference_time is not None:
            _require(
                through < reference_time
                if kind == "early_freeze_vs_revert"
                else through > reference_time,
                "early/late intervention time disagrees "
                "with the explicitly supplied reference time",
            )
    return {
        "kind": kind,
        "through_seconds": through,
        "reference_time_seconds": reference_time,
        "timing_label_source": (
            "explicit comparison kind; no risk time or case selected by this tool"
        ),
        "left_intervention": {
            key: a[key] for key in ("freeze_learning_at", "revert_control_params_at")
        },
        "right_intervention": {
            key: b[key] for key in ("freeze_learning_at", "revert_control_params_at")
        },
    }


def audit_common_prefix(
    left: SavedActuatorRun,
    right: SavedActuatorRun,
    through: float,
    *,
    kind: str,
    left_snapshot: str | Path,
    right_snapshot: str | Path,
    reference_time: float | None = None,
) -> dict[str, Any]:
    """Compare exact prefix samples, leaving interventions at the cut out of the prefix."""
    _require(math.isfinite(through) and through >= 0, "prefix time must be finite and nonnegative")
    contract = _intervention_contract(left, right, kind, through, reference_time)
    lhs = authenticate_boundary_snapshot(left, left_snapshot, through)
    rhs = authenticate_boundary_snapshot(right, right_snapshot, through)
    checks: dict[str, Any] = {}
    for name in ("time", "state", "native_state"):
        mask_a, mask_b = left.dense["time"] <= through + 1e-9, right.dense["time"] <= through + 1e-9
        checks[f"dense_{name}_through_boundary"] = _comparison(
            left.dense[name][mask_a], right.dense[name][mask_b]
        )
    for name in (
        "command",
        "actual_forces",
        "applied_wrenches",
        "actual_effectiveness",
        "actual_time_constants",
    ):
        mask_a, mask_b = left.dense["time"] < through - 1e-9, right.dense["time"] < through - 1e-9
        checks[f"dense_{name}_before_boundary"] = _comparison(
            left.dense[name][mask_a], right.dense[name][mask_b]
        )
    for name in ("time", "command", "state", "control_index"):
        mask_a, mask_b = (
            left.applications["time"] < through - 1e-9,
            right.applications["time"] < through - 1e-9,
        )
        checks[f"application_{name}_before_boundary"] = _comparison(
            left.applications[name][mask_a], right.applications[name][mask_b]
        )
    for name in (
        "time",
        "actual_state",
        "controller_input_state",
        "planned_command",
        "command_applied",
        "command_applied_at",
        "library_version",
        "control_library_version",
        "control_params_sha256",
        "cumulative_gradient_steps",
        "estimated_model_sha256",
    ):
        mask_a, mask_b = (
            left.controls["time"] < through - 1e-9,
            right.controls["time"] < through - 1e-9,
        )
        checks[f"control_{name}_before_boundary"] = _comparison(
            left.controls[name][mask_a], right.controls[name][mask_b]
        )
    checkpoints = {
        "published_params": _tree_comparison(
            lhs.checkpoint.state.params, rhs.checkpoint.state.params
        ),
        "optimizer_state": _tree_comparison(
            lhs.checkpoint.state.optimizer_state, rhs.checkpoint.state.optimizer_state
        ),
        "full_published_learner": _tree_comparison(lhs.checkpoint.state, rhs.checkpoint.state),
    }
    used_at_cut = _comparison(
        left.controls["control_params_sha256"][lhs.index],
        right.controls["control_params_sha256"][rhs.index],
    )
    passed = all(check["exact_equal"] for check in checks.values()) and all(
        check["exact_equal"] for check in checkpoints.values()
    )
    if kind in {"onset_freeze", "no_fault_control"}:
        passed = passed and used_at_cut["exact_equal"]
    return {
        "status": "completed",
        "authentication_passed": passed,
        "contract": contract,
        "prefix_scope": (
            "body/motor states include the cut; right-continuous commands/forces, control "
            "decisions and used versions exclude the intervention boundary; "
            "exact sample comparison without interpolation"
        ),
        "physical_and_control_checks": checks,
        "boundary_checkpoint_checks": checkpoints,
        "left_boundary": lhs.report,
        "right_boundary": rhs.report,
        "left_campaign_envelope": left.campaign_envelope,
        "right_campaign_envelope": right.campaign_envelope,
        "left_storage_authentication": left.storage_authentication,
        "right_storage_authentication": right.storage_authentication,
        "actual_control_params_at_intervention_boundary": used_at_cut,
        "nontrivial_parameter_reversion": kind in {"early_freeze_vs_revert", "late_reversion"}
        and not used_at_cut["exact_equal"],
        "source_sha256": {
            **left.source_sha256,
            **right.source_sha256,
            **lhs.report["source_sha256"],
            **rhs.report["source_sha256"],
        },
        "outcome_difference_is_not_inferred_from_prefix_authentication": True,
    }


@dataclass(frozen=True, slots=True)
class CandidateBranchAudit:
    """Normal decision and complete full-QP/hold branches for each eligible candidate."""

    normal: Any
    forced: dict[int, Any]
    report: dict[str, Any]


def _decision_comparison(normal: Any, forced: Any) -> dict[str, Any]:
    # Validity masking intentionally changes these three diagnostic masks only.
    checks = {
        name: _tree_comparison(getattr(normal, name), getattr(forced, name))
        for name in normal._fields
        if name != "certificates"
    }
    a, b = normal.certificates, forced.certificates
    for name in a._fields:
        if name not in {"rollouts", "gradient_valid", "eligible"}:
            checks[f"certificate_{name}"] = _tree_comparison(getattr(a, name), getattr(b, name))
    for name in ("states", "commands"):
        checks[f"rollout_{name}"] = _comparison(
            getattr(a.rollouts, name), getattr(b.rollouts, name)
        )
    return {
        "exact_equal": all(check["exact_equal"] for check in checks.values()),
        "checks": checks,
        "intentionally_changed": ["rollouts.valid", "gradient_valid", "eligible"],
    }


def _branch_record(step: Any, index: int, normally_eligible: bool) -> dict[str, Any]:
    record = _filter_record(step, retain_rollouts=False)
    record.update(
        candidate_index=index,
        normally_eligible=normally_eligible,
        command=np.asarray(step.action),
        qp_feasible=bool(step.qp.feasible),
        qp_action=np.asarray(step.qp.action),
        qp_multipliers=np.asarray(step.qp.multipliers),
        qp_stationarity_residual=float(step.qp.stationarity_residual),
        qp_complementarity_residual=float(step.qp.complementarity_residual),
        offline_branch_only=True,
    )
    return record


def evaluate_candidate_branches(
    state: Any,
    model: Any,
    obstacles: Any,
    safety: Any,
    rollouts: Any,
    emergency: Any,
    previous_index: Any,
    config: ActuatorFilterConfig,
    *,
    parameters: Any = (),
) -> CandidateBranchAudit:
    """Run every eligible candidate through the unchanged full decision and held checks.

    ``rollouts(y, model, parameters)`` must return the original complete batch,
    including nominal candidate zero. Only ``valid`` is masked in the offline
    forced calls, so command bounds, differentiation, fallback and QP objective
    retain their original definitions. Parameters remain dynamic JIT inputs.
    """

    def solve(y: Any, p: Any, m: Any, o: Any, s: Any, e: Any, previous: Any) -> Any:
        return actuator_plcbf_step(
            y, lambda initial, point: rollouts(initial, point, p), m, o, s, e, previous, config
        )

    def forced_solve(
        y: Any, p: Any, m: Any, o: Any, s: Any, e: Any, previous: Any, target: Any
    ) -> Any:
        def masked(initial: Any, point: Any) -> Any:
            full = rollouts(initial, point, p)
            return full._replace(valid=full.valid & (jnp.arange(full.valid.shape[0]) == target))

        return actuator_plcbf_step(y, masked, m, o, s, e, previous, config)

    inputs = (state, parameters, model, obstacles, safety, emergency, previous_index)
    normal = jax.block_until_ready(jax.jit(solve)(*inputs))
    normal_index = int(normal.selected_index)
    eligible = np.flatnonzero(np.asarray(normal.certificates.eligible)).tolist()
    indices = sorted(set(eligible) | {normal_index})
    force = jax.jit(forced_solve)
    forced = {
        index: jax.block_until_ready(force(*inputs, jnp.asarray(index, jnp.int32)))
        for index in indices
    }
    equivalence = _decision_comparison(normal, forced[normal_index])
    selected_as_requested = all(int(step.selected_index) == index for index, step in forced.items())
    return CandidateBranchAudit(
        normal,
        forced,
        {
            "normal": _branch_record(normal, normal_index, normal_index in eligible),
            "eligible_candidate_indices": eligible,
            "audited_candidate_indices": indices,
            "eligible_candidate_count": len(eligible),
            "eligible_candidates_with_full_qp_and_hold_audit": len(eligible),
            "eligible_full_qp_pass_count": sum(bool(forced[index].qp_valid) for index in eligible),
            "eligible_fallback_pass_count": sum(
                bool(forced[index].fallback_valid) for index in eligible
            ),
            "forced_candidates_selected_as_requested": selected_as_requested,
            "original_selection_equivalence": equivalence,
            "branches": {
                str(index): _branch_record(step, index, index in eligible)
                for index, step in forced.items()
            },
            "enumeration": (
                "complete original candidate batch; mask only other validity entries; "
                "nominal candidate zero remains the QP objective"
            ),
            "extra_ineligible_branch_reason": None
            if normal_index in eligible
            else "original selection retained solely to check normal-decision equivalence",
        },
    )


def _world_from_binding(run: SavedActuatorRun) -> NavigationWorld:
    value = run.binding["scene"]["world"]
    return NavigationWorld(
        config=SimpleNamespace(**value["config"]),
        **{
            name: np.asarray(value[name], dtype=float)
            for name in (
                "initial_state",
                "waypoint_positions",
                "obstacle_mean_centers",
                "obstacle_amplitudes",
                "obstacle_angular_frequencies",
                "obstacle_phases",
                "obstacle_radii",
            )
        },
    )


def _recorded_inputs(run: SavedActuatorRun, index: int, when: float) -> tuple[Any, ...]:
    """Use the observed controller state and parameter estimates recorded at this boundary."""
    controls, config = run.controls, ActuatorFilterConfig(**run.binding["config"]["filter_config"])
    state = jnp.asarray(controls["controller_input_state"][index])
    _require(state.shape == (17,), "recorded controller input must include motor state")
    _require(
        _hash_tree(state) == str(controls["controller_input_state_sha256"][index]),
        "recorded controller input hash or dtype changed during loading",
    )
    model = nominal_actuator_model(dtype=state.dtype)
    model = model._replace(
        effectiveness=jnp.asarray(controls["estimated_effectiveness"][index]),
        time_constants=jnp.asarray(controls["estimated_time_constants"][index]),
    )
    _require(
        _same(model.body.mass, controls["estimated_mass"][index])
        and _same(model.body.inertia, controls["estimated_inertia"][index]),
        "recorded observed model changes the declared unchanged mass/inertia",
    )
    _require(
        _hash_tree(model) == str(controls["estimated_model_sha256"][index]),
        "reconstructed observed model does not exactly match its recorded full-model hash",
    )
    world = _world_from_binding(run)
    obstacles = world.obstacle_prediction(when, dt=config.dt, horizon=config.horizon)
    safety = world.safety_limits(when)
    bias = jnp.asarray(run.binding["config"]["observation_config"]["obstacle_position_bias_m"])
    obstacles = obstacles._replace(centers=obstacles.centers + bias)
    safety = safety._replace(obstacle_centers=safety.obstacle_centers + bias)
    applied = np.flatnonzero(controls["command_applied"][:index])
    previous = int(controls["selected_index"][applied[-1]]) if len(applied) else 0
    return (
        state,
        model,
        obstacles,
        safety,
        jnp.asarray(previous, jnp.int32),
        jnp.asarray(controls["goal"][index]),
        config,
    )


def _replay_comparison(actual: Any, recorded: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    a, b = np.asarray(actual), np.asarray(recorded)
    result = _comparison(a, b)
    if a.shape != b.shape:
        close = False
    elif a.dtype.kind in "fc" and b.dtype.kind in "fc":
        close = bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
    else:
        close = bool(np.array_equal(a, b))
    return {**result, "within_replay_tolerance": close}


def _recorded_reproduction(
    step: Any, controls: dict[str, np.ndarray], index: int, *, atol: float, rtol: float
) -> dict[str, Any]:
    generated = _filter_record(step, retain_rollouts=False)
    generated["planned_command"] = np.asarray(step.action)
    checks = {
        name: _replay_comparison(value, controls[name][index], atol=atol, rtol=rtol)
        for name, value in generated.items()
        if name in controls
    }
    return {
        "reproduced": all(row["within_replay_tolerance"] for row in checks.values()),
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "checks": checks,
        "unavailable_record_fields": sorted(generated.keys() - controls.keys()),
        "command_was_physically_applied": bool(controls["command_applied"][index]),
        "command_application_time_seconds": float(controls["command_applied_at"][index]),
    }


def _numeric_tree(value: Any, prefix: str, destination: dict[str, np.ndarray]) -> None:
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        for name in value._fields:
            _numeric_tree(getattr(value, name), f"{prefix}.{name}", destination)
    elif isinstance(value, dict):
        for name, item in value.items():
            _numeric_tree(item, f"{prefix}.{name}", destination)
    elif value is not None:
        destination[prefix] = np.asarray(value)


def _authenticate_initial_reference(
    run: SavedActuatorRun,
    original_checkpoint: str | Path,
    boundaries: tuple[BoundaryAuthentication, ...],
    *,
    replay_atol: float,
    replay_rtol: float,
) -> Any:
    """Keep deployment identity independent of the parameter set being compared."""
    _require(
        run.binding["config"]["method"] in {"A", "A1", "F2"},
        "shared-state original-versus-available audit requires the shared F2 controller adapter",
    )
    _require(
        all(math.isfinite(v) and v >= 0 for v in (replay_atol, replay_rtol)),
        "replay tolerances must be finite and nonnegative",
    )
    original = load_actuator_learner_checkpoint(original_checkpoint)
    _require(
        original.sha256 == run.binding["checkpoint"]["checkpoint_sha256"],
        "original deployment checkpoint is not the run's initial checkpoint",
    )
    _require(
        _hash_tree(original.state.params) == run.binding["initial_params_sha256"]
        and _hash_tree(original.state) == run.binding["initial_learner_sha256"],
        "original learner/params differ from the run's initial snapshot",
    )
    for boundary in boundaries:
        available = boundary.checkpoint
        _require(
            original.config == available.config
            and actuator_reference_fingerprint(original.contract)
            == actuator_reference_fingerprint(available.contract),
            "original and available checkpoints do not share the immutable reference contract",
        )
    return original


def audit_shared_state(
    run: SavedActuatorRun,
    when: float,
    *,
    boundary_snapshot: str | Path,
    original_checkpoint: str | Path,
    replay_atol: float = 2e-6,
    replay_rtol: float = 2e-5,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compare original/available parameters at one authenticated recorded physical boundary."""
    boundary = authenticate_boundary_snapshot(run, boundary_snapshot, when)
    original = _authenticate_initial_reference(
        run, original_checkpoint, (boundary,), replay_atol=replay_atol, replay_rtol=replay_rtol
    )
    return _audit_parameter_pair(
        run,
        when,
        boundary=boundary,
        original=original,
        reference=original,
        reference_name="original",
        replay_atol=replay_atol,
        replay_rtol=replay_rtol,
    )


def audit_prior_available(
    run: SavedActuatorRun,
    when: float,
    *,
    boundary_snapshot: str | Path,
    prior_time: float,
    prior_snapshot: str | Path,
    original_checkpoint: str | Path,
    replay_atol: float = 2e-6,
    replay_rtol: float = 2e-5,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compare adjacent published parameters using only the later recorded inputs.

    Each checkpoint is authenticated against its own actual physical/control
    boundary. The previous state is used only to reproduce its recorded decision;
    the parameter counterfactual always uses the current state and obstacle clock.
    """
    _require(
        run.binding["config"]["method"] in {"A", "A1"},
        "prior-available comparison requires an online adaptive run",
    )
    prior = authenticate_boundary_snapshot(run, prior_snapshot, prior_time)
    boundary = authenticate_boundary_snapshot(run, boundary_snapshot, when)
    original = _authenticate_initial_reference(
        run,
        original_checkpoint,
        (prior, boundary),
        replay_atol=replay_atol,
        replay_rtol=replay_rtol,
    )
    _require(
        prior_time < when and boundary.index == prior.index + 1,
        "prior/current snapshots must be adjacent actual sensing boundaries",
    )
    _require(
        not prior.report["control_params_reverted"]
        and not boundary.report["control_params_reverted"],
        "prior-available comparison requires both boundaries to use available parameters",
    )
    gradient_delta = int(boundary.checkpoint.state.cumulative_gradient_steps) - int(
        prior.checkpoint.state.cumulative_gradient_steps
    )
    version_delta = int(boundary.checkpoint.state.library_version) - int(
        prior.checkpoint.state.library_version
    )
    _require(
        gradient_delta > 0 and version_delta > 0,
        "prior/current boundaries must contain a newly published learner update",
    )
    result, arrays = _audit_parameter_pair(
        run,
        when,
        boundary=boundary,
        original=original,
        reference=prior.checkpoint,
        reference_name="prior_available",
        replay_atol=replay_atol,
        replay_rtol=replay_rtol,
    )
    y, model, obstacles, safety, previous, goal, config = _recorded_inputs(
        run, prior.index, prior_time
    )
    functions = build_actuator_controller(original.contract.spec, original.config, config)
    prior_decision = jax.block_until_ready(
        functions.controller(
            y, prior.checkpoint.state.params, model, obstacles, safety, previous, goal
        )
    )
    prior_reproduction = _recorded_reproduction(
        prior_decision, run.controls, prior.index, atol=replay_atol, rtol=replay_rtol
    )
    result["confirmation_passed"] = (
        result["confirmation_passed"] and prior_reproduction["reproduced"]
    )
    result["prior_boundary"] = prior.report
    result["prior_recorded_decision_reproduction"] = prior_reproduction
    result["publication_comparison"] = {
        "adjacent_actual_sensing_boundaries": True,
        "prior_time_seconds": prior_time,
        "current_time_seconds": when,
        "published_gradient_step_delta": gradient_delta,
        "published_library_version_delta": version_delta,
        "single_gradient_update": gradient_delta == 1,
        "parameters_changed": _hash_tree(prior.checkpoint.state.params)
        != _hash_tree(boundary.checkpoint.state.params),
        "prior_full_learner_sha256": _hash_tree(prior.checkpoint.state),
        "current_full_learner_sha256": _hash_tree(boundary.checkpoint.state),
        "counterfactual_scope": (
            "prior available parameters evaluated at the current recorded controller state, "
            "observed model, goal, previous selection and absolute obstacle clock; "
            "the prior physical state is used only for prior-decision authentication"
        ),
    }
    result["source_sha256"].update(prior.report["source_sha256"])
    arrays["prior_boundary.physical_state"] = np.asarray(prior.checkpoint.physical_state)
    arrays["prior_boundary.controller_input_state"] = np.asarray(y)
    arrays["prior_boundary.absolute_obstacle_clock_seconds"] = np.asarray(prior_time)
    _numeric_tree(prior_decision, "prior_boundary.recorded_decision_replay", arrays)
    return result, arrays


def _audit_parameter_pair(
    run: SavedActuatorRun,
    when: float,
    *,
    boundary: BoundaryAuthentication,
    original: Any,
    reference: Any,
    reference_name: str,
    replay_atol: float,
    replay_rtol: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Enumerate complete branches only after callers authenticate parameter provenance."""
    available = boundary.checkpoint
    state, model, obstacles, safety, previous, goal, config = _recorded_inputs(
        run, boundary.index, when
    )
    functions = build_actuator_controller(original.contract.spec, original.config, config)
    emergency = functions.emergency(state, model)

    def rollouts(y: Any, point: Any, parameters: Any) -> Any:
        return functions.candidates(y, parameters, goal, point)

    audits = {
        name: evaluate_candidate_branches(
            state,
            model,
            obstacles,
            safety,
            rollouts,
            emergency,
            previous,
            config,
            parameters=checkpoint.state.params,
        )
        for name, checkpoint in ((reference_name, reference), ("available", available))
    }
    index = boundary.index
    control_name = (
        reference_name if bool(run.controls["control_params_reverted"][index]) else "available"
    )
    reproduction = _recorded_reproduction(
        audits[control_name].normal, run.controls, index, atol=replay_atol, rtol=replay_rtol
    )
    a, b = audits[reference_name].normal, audits["available"].normal
    changed_action = not bool(
        np.allclose(a.action, b.action, atol=replay_atol, rtol=replay_rtol, equal_nan=True)
    )
    learned_selected = int(b.selected_index) > 0
    learned_row_active = learned_selected and bool(b.qp_valid) and float(b.executed_policy_dual) > 0
    applied = bool(run.controls["command_applied"][index])
    actual_learned_execution = (
        applied and control_name == "available" and reproduction["reproduced"]
    )
    current_hashes = {
        path.name: _sha(path) for path in (ROOT / "crazyflow/safety/da_plcbf").glob("actuator_*.py")
    }
    source_matches = {
        name: current_hashes.get(name) == digest
        for name, digest in run.binding.get("source_sha256", {}).items()
    }
    required_sources = (
        "actuator_plcbf.py",
        "actuator_dynamics.py",
        "actuator_study.py",
        "actuator_learning.py",
    )
    exact_controller_sources = all(source_matches.get(name, False) for name in required_sources)
    if run.campaign_envelope is not None:
        exact_controller_sources = (
            exact_controller_sources and run.campaign_envelope["all_current_campaign_sources_match"]
        )
    enumeration_valid = all(
        audit.report["original_selection_equivalence"]["exact_equal"]
        and audit.report["forced_candidates_selected_as_requested"]
        for audit in audits.values()
    )
    result = {
        "status": "completed",
        "comparison_kind": f"{reference_name}_vs_available",
        "confirmation_passed": enumeration_valid
        and reproduction["reproduced"]
        and exact_controller_sources,
        "boundary": boundary.report,
        "campaign_envelope": run.campaign_envelope,
        "storage_authentication": run.storage_authentication,
        "time_seconds": when,
        "fixed_inputs": {
            "physical_state": np.asarray(available.physical_state),
            "recorded_controller_input_state": np.asarray(state),
            "physical_and_controller_state_are_exactly_equal": _same(
                available.physical_state, state
            ),
            "observation_semantics": (
                "use the saved controller input including original observation noise "
                "and dtype conversion; all 17 physical components retained separately"
            ),
            "model_source": (
                "exact full-model hash authenticated from this control record's observed "
                "parameter estimates, not a learner training estimate"
            ),
            "model_sha256": _hash_tree(model),
            "estimated_effectiveness": np.asarray(model.effectiveness),
            "estimated_time_constants": np.asarray(model.time_constants),
            "actual_effectiveness": run.controls["actual_effectiveness"][index],
            "actual_time_constants": run.controls["actual_time_constants"][index],
            "absolute_obstacle_clock_seconds": when,
            "obstacle_prediction_sha256": _hash_tree(obstacles),
            "goal": np.asarray(goal),
            "previous_selected_index": int(previous),
        },
        "deployment_checkpoint_sha256": original.sha256,
        f"{reference_name}_parameters_sha256": _hash_tree(reference.state.params),
        "available_parameters_sha256": _hash_tree(available.state.params),
        "actual_control_parameter_set": control_name,
        reference_name: audits[reference_name].report,
        "available": audits["available"].report,
        "recorded_decision_reproduction": reproduction,
        "control_effect_evidence": {
            f"{reference_name}_vs_available_action": _comparison(a.action, b.action),
            "action_difference_exceeds_replay_tolerance": changed_action,
            "available_selects_learned_candidate": learned_selected,
            "available_learned_row_has_executed_positive_qp_dual": learned_row_active,
            "available_learned_row_changes_offline_action": learned_row_active and changed_action,
            "recorded_applied_learned_row_changes_action": actual_learned_execution
            and learned_row_active
            and changed_action,
            "recorded_applied_learned_fallback": actual_learned_execution
            and learned_selected
            and int(b.execution_mode) == 1,
            f"{reference_name}_executed_policy_dual": float(a.executed_policy_dual),
            "available_executed_policy_dual": float(b.executed_policy_dual),
            "reference_eligible_current_ineligible_candidate_indices": np.flatnonzero(
                np.asarray(a.certificates.eligible) & ~np.asarray(b.certificates.eligible)
            ),
            "reference_ineligible_current_eligible_candidate_indices": np.flatnonzero(
                ~np.asarray(a.certificates.eligible) & np.asarray(b.certificates.eligible)
            ),
            "reference_qp_pass_available_qp_fail": bool(a.qp_valid) and not bool(b.qp_valid),
            "reference_full_qp_pass_count": audits[reference_name].report[
                "eligible_full_qp_pass_count"
            ],
            "available_full_qp_pass_count": audits["available"].report[
                "eligible_full_qp_pass_count"
            ],
            "scope": (
                "same-state offline action comparison and authenticated application only; "
                "unused predictions or selected duals from rejected QPs are not executed "
                "control effects; no episode outcome claim"
            ),
        },
        "source_integrity": {
            "recorded_runtime_source_matches_current": source_matches,
            "controller_sources_match": exact_controller_sources,
            "current_runtime_source_sha256": current_hashes,
        },
        "source_sha256": {
            **run.source_sha256,
            **boundary.report["source_sha256"],
            str(original.npz_path.resolve()): _sha(original.npz_path),
            str(original.json_path.resolve()): _sha(original.json_path),
        },
    }
    arrays = {
        "physical_state": np.asarray(available.physical_state),
        "controller_input_state": np.asarray(state),
        "goal": np.asarray(goal),
        "absolute_obstacle_clock_seconds": np.asarray(when),
        "previous_selected_index": np.asarray(previous),
    }
    _numeric_tree(model, "observed_model", arrays)
    _numeric_tree(obstacles, "obstacles", arrays)
    _numeric_tree(safety, "safety", arrays)
    for name, audit in audits.items():
        _numeric_tree(audit.normal, f"{name}.normal", arrays)
        for candidate, step in audit.forced.items():
            _numeric_tree(step, f"{name}.candidate_{candidate}", arrays)
    return result, arrays


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="operation", required=True)
    prefix = modes.add_parser("prefix", help="authenticate one explicitly specified causal pair")
    prefix.add_argument("--left", type=Path, required=True)
    prefix.add_argument("--right", type=Path, required=True)
    prefix.add_argument("--left-snapshot", type=Path, required=True)
    prefix.add_argument("--right-snapshot", type=Path, required=True)
    prefix.add_argument("--kind", choices=KINDS, required=True)
    prefix.add_argument("--through", type=float, required=True)
    prefix.add_argument("--reference-time", type=float)
    shared = modes.add_parser(
        "shared", help="audit original/available branches at one saved boundary"
    )
    prior = modes.add_parser(
        "shared-prior", help="audit adjacent prior/current publications at the current boundary"
    )
    prior.add_argument("--prior-snapshot", type=Path, required=True)
    prior.add_argument("--prior-time", type=float, required=True)
    for mode in (shared, prior):
        mode.add_argument("--episode", type=Path, required=True)
        mode.add_argument("--boundary-snapshot", type=Path, required=True)
        mode.add_argument("--original-checkpoint", type=Path, required=True)
        mode.add_argument("--time", type=float, required=True)
        mode.add_argument("--replay-atol", type=float, default=2e-6)
        mode.add_argument("--replay-rtol", type=float, default=2e-5)
    for mode in (prefix, shared, prior):
        mode.add_argument("--output", type=Path, required=True)
        mode.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    source_dir = output / "source"
    source_dir.mkdir()
    source_paths = [Path(__file__), *(ROOT / "crazyflow/safety/da_plcbf").glob("*.py")]
    copied_hashes = {}
    for path in source_paths:
        copy = source_dir / path.name
        with copy.open("xb") as stream:
            stream.write(path.read_bytes())
        copied_hashes[str(copy)] = _sha(copy)
    input_hashes = {}
    for name in ("left", "right", "episode"):
        directory = getattr(args, name, None)
        if directory is not None:
            for filename in (
                "binding.json",
                "summary.json",
                "dense.npz",
                "controls.npz",
                "applications.npz",
            ):
                path = directory.resolve() / filename
                if path.is_file():
                    input_hashes[str(path)] = _sha(path)
    for name in (
        "left_snapshot",
        "right_snapshot",
        "boundary_snapshot",
        "original_checkpoint",
        "prior_snapshot",
    ):
        stem = getattr(args, name, None)
        if stem is not None:
            stem = stem.resolve()
            if stem.suffix in {".npz", ".json"}:
                stem = stem.with_suffix("")
            for suffix in (".npz", ".json"):
                path = Path(f"{stem}{suffix}")
                if path.is_file():
                    input_hashes[str(path)] = _sha(path)
    _write_json(
        output / "request.json",
        {
            "arguments": vars(args),
            "started_unix_seconds": started,
            "source_sha256": copied_hashes,
            "input_sha256": input_hashes,
        },
    )
    try:
        # CPU is the default and does not initialize a GPU backend. GPU use is explicit.
        if args.platform == "cpu":
            os.environ["JAX_PLATFORMS"] = "cpu"
            jax.config.update("jax_platforms", "cpu")
        device = jax.devices(args.platform)[0]
        with jax.default_device(device):
            if args.operation == "prefix":
                result = audit_common_prefix(
                    load_saved_run(args.left),
                    load_saved_run(args.right),
                    args.through,
                    kind=args.kind,
                    left_snapshot=args.left_snapshot,
                    right_snapshot=args.right_snapshot,
                    reference_time=args.reference_time,
                )
                passed = result["authentication_passed"]
            else:
                comparison = (
                    audit_prior_available
                    if args.operation == "shared-prior"
                    else audit_shared_state
                )
                extra = (
                    {"prior_time": args.prior_time, "prior_snapshot": args.prior_snapshot}
                    if args.operation == "shared-prior"
                    else {}
                )
                result, arrays = comparison(
                    load_saved_run(args.episode),
                    args.time,
                    boundary_snapshot=args.boundary_snapshot,
                    original_checkpoint=args.original_checkpoint,
                    replay_atol=args.replay_atol,
                    replay_rtol=args.replay_rtol,
                    **extra,
                )
                with (output / "numerical.npz").open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                result["numerical_artifact_sha256"] = _sha(output / "numerical.npz")
                passed = result["confirmation_passed"]
        result["elapsed_wall_seconds"] = time.time() - started
        result["platform"] = str(device)
        _write_json(output / "results.json", result)
        print(json.dumps({"output": str(output), "status": "completed", "passed": passed}))
        return 0 if passed else 2
    except Exception as error:
        _write_json(
            output / "error.json",
            {
                "status": "incomplete",
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "elapsed_wall_seconds": time.time() - started,
            },
        )
        print(json.dumps({"output": str(output), "status": "incomplete", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
