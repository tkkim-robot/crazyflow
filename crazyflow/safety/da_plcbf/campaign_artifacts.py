"""Strict progress, resume, and paired-inference artifacts for real DA-PLCBF campaigns.

The legacy artifact helpers describe a fully rendered immutable run.  A long GPU campaign also
needs a crash-safe progress layer before videos exist.  This module records every scheduled
outcome (including execution failures), validates successful numerical artifacts before they may
be skipped on resume, and regenerates paired inference from the complete schedule.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from enum import Enum
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.adaptation_evidence import (
    load_adaptation_evidence,
    validate_adaptation_evidence_binding,
)
from crazyflow.safety.da_plcbf.artifacts import (
    RUN_CONFIG_SCHEMA_VERSION,
    SEEDS_SCHEMA_VERSION,
    _validate_runtime_device_roles,
    aggregate_row,
    collect_provenance,
    load_events,
    load_metrics,
    load_timing,
    load_trace,
    validate_provenance,
    validate_run_config,
    validate_seeds,
    validate_trace_scenario_binding,
    write_aggregate_report,
    write_confidence_intervals,
    write_paired_metrics_csv,
    write_provenance,
    write_run_config,
    write_seeds,
)
from crazyflow.safety.da_plcbf.dashboard_evidence import (
    load_dashboard_evidence,
    validate_dashboard_evidence_binding,
)
from crazyflow.safety.da_plcbf.direct_wrench import motor_forces_to_wrench
from crazyflow.safety.da_plcbf.dynamic_rollouts import DYNAMIC_PREDICTION_CONTRACT
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.scenarios import load_scenario_tape, save_scenario_tape
from crazyflow.safety.da_plcbf.scientific_evaluation import (
    AnalysisRole,
    LatencyMetrics,
    PairedTrialDataset,
    RecoveryMetrics,
    ScientificTrialMetrics,
    ScientificTrialRecord,
    TrialStatus,
)
from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel

if TYPE_CHECKING:
    from collections.abc import Mapping

    from crazyflow.safety.da_plcbf.experiments import CampaignConfig, CampaignRun, TrialRun
    from crazyflow.safety.da_plcbf.scenarios import ScenarioTape
    from crazyflow.safety.da_plcbf.scientific_evaluation import PairedTrialSchedule, TrialAssignment


CAMPAIGN_OUTCOME_SCHEMA_VERSION = 1
CAMPAIGN_COMPARISON_SCHEMA_VERSION = 3
_MAX_JSONL_LINE_BYTES = 4 * 1024 * 1024


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate JSON key in {path}")
        return dict(pairs)

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs_hook)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _source_tree_sha256(repository: Path) -> str:
    """Hash numerical sources and package runtime assets, excluding outputs and documentation."""
    digest = hashlib.sha256(b"crazyflow.da_plcbf.source-tree.v2\0")
    roots_and_suffixes = (
        (repository / "crazyflow", frozenset({".py", ".toml", ".xml", ".stl"})),
        (repository / "examples" / "da_plcbf", frozenset({".py"})),
        (repository / "benchmark", frozenset({".py"})),
    )
    paths = sorted(
        path
        for root, suffixes in roots_and_suffixes
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    paths.extend(
        path for name in ("pyproject.toml", "pixi.lock") if (path := repository / name).is_file()
    )
    paths = sorted(set(paths))
    for path in paths:
        relative = path.relative_to(repository).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _schedule_sha256(schedule: PairedTrialSchedule) -> str:
    payload = {
        "root_seed": schedule.root_seed,
        "methods": schedule.methods,
        "conditions": schedule.conditions,
        "trials_per_condition": schedule.trials_per_condition,
        "fold_start": schedule.fold_start,
        "intended_for_final_claim": schedule.intended_for_final_claim,
        "assignments": [asdict(assignment) for assignment in schedule.assignments],
    }
    return hashlib.sha256(
        b"crazyflow.da_plcbf.paired-schedule.v1\0" + _canonical_json_bytes(payload)
    ).hexdigest()


def _campaign_config_mapping(
    campaign: CampaignConfig, schedule: PairedTrialSchedule, repository: Path
) -> dict[str, Any]:
    return {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "experiment_id": "da-plcbf-scientific-campaign-v1",
        "description": "Real finite-horizon paired DA-PLCBF controller campaign",
        "control_dt_seconds": campaign.trial.dt,
        "horizon_steps": campaign.trial.certificate_horizon,
        "paired_trials": True,
        "trials_per_condition": campaign.trials_per_condition,
        "methods": list(campaign.methods),
        "conditions": list(campaign.conditions),
        "parameters": {
            "campaign_schema_version": CAMPAIGN_OUTCOME_SCHEMA_VERSION,
            "dynamic_prediction_contract": DYNAMIC_PREDICTION_CONTRACT,
            "trial": asdict(campaign.trial),
            "root_seed": campaign.root_seed,
            "fold_start": campaign.fold_start,
            "intended_for_final_claim": campaign.intended_for_final_claim,
            "schedule_sha256": _schedule_sha256(schedule),
            "source_tree_sha256": _source_tree_sha256(repository),
        },
    }


def _seed_mapping(
    schedule: PairedTrialSchedule, tapes: Mapping[tuple[str, int], ScenarioTape]
) -> dict[str, Any]:
    records = []
    for (condition, fold), tape in sorted(tapes.items()):
        records.append(
            {
                "condition": condition,
                "fold": fold,
                "path": f"scenario_tapes/{condition}/{fold}.npz",
                "content_sha256": tape.sha256,
            }
        )
    return {
        "schema_version": SEEDS_SCHEMA_VERSION,
        "root_seed": schedule.root_seed,
        "folds": list(
            range(schedule.fold_start, schedule.fold_start + schedule.trials_per_condition)
        ),
        "named_streams": dict(schedule.named_stream_ids),
        "scenario_tapes": records,
        "pairing_id": _schedule_sha256(schedule),
    }


_RESUME_NUMERICAL_PACKAGES = ("crazyflow", "numpy", "scipy", "jax", "jaxlib", "flax", "optax")
_RESUME_ANALYSIS_PACKAGES = ("numpy", "scipy")


def _resume_execution_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Select immutable controller-execution identity, excluding descriptive media metadata."""
    validated = validate_provenance(provenance)
    return {
        "git": validated["git"],
        "runtime": validated["runtime"],
        "hardware": validated["hardware"],
        "jax": validated["jax"],
        "numerical_packages": {
            name: validated["packages"].get(name, "unavailable")
            for name in _RESUME_NUMERICAL_PACKAGES
        },
    }


def _validate_resume_execution_identity(
    stored: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Reject pending execution under a different immutable runtime or device identity."""
    stored_identity = _resume_execution_identity(stored)
    current_identity = _resume_execution_identity(current)
    for field in stored_identity:
        if stored_identity[field] != current_identity[field]:
            raise ValueError(
                f"resume execution identity {field} differs from the stored campaign session"
            )


def _validate_resume_analysis_identity(
    stored: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Bind every resumable aggregate/finalization pass to its analysis semantics."""
    stored_validated = validate_provenance(stored)
    current_validated = validate_provenance(current)
    for field in ("python", "implementation"):
        if stored_validated["runtime"][field] != current_validated["runtime"][field]:
            raise ValueError(
                f"resume analysis runtime {field} differs from the stored campaign session"
            )
    stored_packages = stored_validated["packages"]
    current_packages = current_validated["packages"]
    for package in _RESUME_ANALYSIS_PACKAGES:
        if stored_packages.get(package, "unavailable") != current_packages.get(
            package, "unavailable"
        ):
            raise ValueError(
                f"resume analysis package {package} differs from the stored campaign session"
            )


def _metrics_mapping(metrics: ScientificTrialMetrics | None) -> dict[str, Any] | None:
    return None if metrics is None else _json_value(asdict(metrics))


def _metrics_from_mapping(value: Any) -> ScientificTrialMetrics | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("scientific_metrics must be an object or null")
    required = set(ScientificTrialMetrics.__dataclass_fields__)
    if set(value) != required:
        raise ValueError("scientific_metrics fields do not match schema")
    payload = dict(value)
    recoveries = payload.pop("recoveries")
    latencies = payload.pop("latencies")
    if not isinstance(recoveries, list) or not isinstance(latencies, list):
        raise ValueError("scientific metric nested records must be lists")
    metrics = ScientificTrialMetrics(
        **payload,
        recoveries=tuple(RecoveryMetrics(**item) for item in recoveries),
        latencies=tuple(LatencyMetrics(**item) for item in latencies),
    )
    metrics.validate()
    return metrics


def _assignment_mapping(assignment: TrialAssignment) -> dict[str, Any]:
    return _json_value(asdict(assignment))


def outcome_mapping(
    assignment: TrialAssignment, record: ScientificTrialRecord, run: TrialRun | None
) -> dict[str, Any]:
    """Build one exact scheduled outcome, retaining failure diagnostics and trace bindings."""
    record.validate()
    if record.key != assignment.key:
        raise ValueError("outcome record does not match assignment")
    mapping: dict[str, Any] = {
        "schema_version": CAMPAIGN_OUTCOME_SCHEMA_VERSION,
        "assignment": _assignment_mapping(assignment),
        "status": record.status.value,
        "scenario_tape_sha256": record.scenario_tape_sha256,
        "scientific_metrics": _metrics_mapping(record.metrics),
        "failure_code": record.failure_code,
        "failure_message": record.failure_message,
        "trace_content_sha256": None,
        "method_claim_eligible": None,
        "claim_blockers": None,
        "compile_seconds": None,
        "compile_cache_hits": None,
        "deadlines_seconds": None,
        "estimation_scale": None,
    }
    if record.status is TrialStatus.COMPLETE:
        if run is None or run.assignment.key != assignment.key:
            raise ValueError("complete outcome requires its TrialRun")
        mapping.update(
            {
                "trace_content_sha256": run.trace.content_sha256,
                "method_claim_eligible": run.method_claim_eligible,
                "claim_blockers": list(run.claim_blockers),
                "compile_seconds": dict(run.compile_seconds),
                "compile_cache_hits": dict(run.compile_cache_hits),
                "deadlines_seconds": dict(run.deadlines_seconds),
                "estimation_scale": np.asarray(run.estimation_scale).tolist(),
            }
        )
    elif run is not None:
        raise ValueError("failed outcome cannot contain a TrialRun")
    return mapping


def _record_from_outcome(
    mapping: Mapping[str, Any], assignment: TrialAssignment
) -> ScientificTrialRecord:
    expected = {
        "schema_version",
        "assignment",
        "status",
        "scenario_tape_sha256",
        "scientific_metrics",
        "failure_code",
        "failure_message",
        "trace_content_sha256",
        "method_claim_eligible",
        "claim_blockers",
        "compile_seconds",
        "compile_cache_hits",
        "deadlines_seconds",
        "estimation_scale",
    }
    if set(mapping) != expected or mapping["schema_version"] != CAMPAIGN_OUTCOME_SCHEMA_VERSION:
        raise ValueError("campaign outcome fields/schema do not match")
    if _canonical_json_bytes(mapping["assignment"]) != _canonical_json_bytes(
        _assignment_mapping(assignment)
    ):
        raise ValueError("campaign outcome assignment does not match deterministic schedule")
    try:
        status = TrialStatus(mapping["status"])
    except (TypeError, ValueError) as error:
        raise ValueError("campaign outcome status is invalid") from error
    record = ScientificTrialRecord(
        method=assignment.method,
        condition=assignment.condition,
        fold=assignment.fold,
        pairing_id=assignment.pairing_id,
        scenario_tape_sha256=mapping["scenario_tape_sha256"],
        status=status,
        metrics=_metrics_from_mapping(mapping["scientific_metrics"]),
        failure_code=mapping["failure_code"],
        failure_message=mapping["failure_message"],
    )
    record.validate()
    nullable = (
        "trace_content_sha256",
        "method_claim_eligible",
        "claim_blockers",
        "compile_seconds",
        "compile_cache_hits",
        "deadlines_seconds",
        "estimation_scale",
    )
    if status is TrialStatus.EXECUTION_FAILURE and any(
        mapping[name] is not None for name in nullable
    ):
        raise ValueError("execution failure outcome contains success-only fields")
    if status is TrialStatus.COMPLETE and any(mapping[name] is None for name in nullable):
        raise ValueError("complete outcome is missing success-only fields")
    return record


def _comparison_payload(run: CampaignRun) -> dict[str, Any]:
    inference = None if run.inference_config is None else _json_value(asdict(run.inference_config))
    exploratory_inference = (
        None
        if run.exploratory_inference_config is None
        else _json_value(asdict(run.exploratory_inference_config))
    )
    return {
        "schema_version": CAMPAIGN_COMPARISON_SCHEMA_VERSION,
        "schedule_sha256": _schedule_sha256(run.schedule),
        "inference_config": inference,
        "exploratory_inference_config": exploratory_inference,
        "comparisons": [_json_value(asdict(item)) for item in run.paired_comparisons],
        "execution_complete": run.execution_complete,
        "scientific_claim_eligible": run.scientific_claim_eligible,
        "global_confirmatory_superiority_supported": (
            run.global_confirmatory_superiority_supported
        ),
        "claim_blockers": list(run.claim_blockers),
    }


def _report_text(run: CampaignRun) -> str:
    completed = sum(record.status is TrialStatus.COMPLETE for record in run.records)
    failures = len(run.records) - completed
    confirmatory = [
        item for item in run.paired_comparisons if item.analysis_role is AnalysisRole.CONFIRMATORY
    ]
    exploratory = [
        item for item in run.paired_comparisons if item.analysis_role is AnalysisRole.EXPLORATORY
    ]
    supported = [item for item in confirmatory if item.superiority_supported]
    lines = [
        "# DA-PLCBF paired scientific report",
        "",
        (
            f"Scheduled outcomes: {len(run.records)}; complete: {completed}; "
            f"execution failures: {failures}."
        ),
        "",
        (
            "Execution completeness, final-claim eligibility, and metric-level superiority "
            "are separate gates."
        ),
        "",
        f"- Execution complete: `{str(run.execution_complete).lower()}`",
        f"- Final-claim eligible: `{str(run.scientific_claim_eligible).lower()}`",
        (
            "- Global confirmatory superiority across every predeclared condition, baseline, "
            f"and endpoint: `{str(run.global_confirmatory_superiority_supported).lower()}`"
        ),
        f"- Supported confirmatory comparisons: `{len(supported)}` of `{len(confirmatory)}`",
        f"- Exploratory comparisons (never claim-eligible): `{len(exploratory)}`",
        "",
    ]
    if run.inference_config is not None:
        lines.extend(
            (
                "## Predeclared analysis roles",
                "",
                (
                    "Confirmatory endpoints are `operational_failure`, trial-level "
                    "`any_failure`, and `minimum_hard_margin`. Their Bonferroni family spans "
                    "every condition and full-method-versus-baseline pairing."
                ),
                "",
                (
                    "For a declared safety-controller method, `operational_failure` includes "
                    "execution failure, physical trace failure, or any explicit degraded "
                    "interval. `nominal_only` has no safety controller, so its intentional "
                    "certificate-unavailable marker is excluded while physical failures remain "
                    "counted."
                ),
                "",
                (
                    f"The confirmatory percentile bootstrap uses "
                    f"`{run.inference_config.bootstrap_replicates}` replicates across "
                    f"`{run.inference_config.familywise_comparisons}` comparisons, yielding "
                    f"`{run.inference_config.expected_bootstrap_draws_per_tail:.3f}` expected "
                    "draws beyond each adjusted endpoint."
                ),
                "",
                (
                    "Certification coverage, degraded duration, intervention, controller latency, "
                    "command-ready latency, and wall-step latency are exploratory diagnostics "
                    "with unadjusted intervals; they cannot support a superiority statement."
                ),
                "",
            )
        )
    if run.claim_blockers:
        lines.extend(("## Claim blockers", ""))
        lines.extend(f"- {blocker}" for blocker in run.claim_blockers)
        lines.append("")
    lines.extend(("## Paired inference", ""))
    if not run.paired_comparisons:
        lines.append("No full-method-versus-baseline comparison was scheduled.")
    else:
        lines.extend(
            (
                "| Role | Condition | Baseline | Metric | Pairs | Missing | Superiority | "
                "Conclusion |",
                "|---|---|---|---|---:|---:|---|---|",
            )
        )
        for item in run.paired_comparisons:
            conclusion = item.conclusion.replace("|", "\\|")
            lines.append(
                f"| {item.analysis_role.value} | {item.condition} | {item.baseline_method} | "
                f"{item.metric_name} | "
                f"{item.pair_count} | {item.missing_metric_pairs} | "
                f"{str(item.superiority_supported).lower()} | {conclusion} |"
            )
    lines.extend(
        (
            "",
            (
                "> Conclusions apply only to the predeclared finite simulation horizon and "
                "recorded matched scenario tapes; they are not hardware or real-world safety "
                "guarantees."
            ),
            "",
        )
    )
    return "\n".join(lines)


_TRACE_STATE_NAMES = (
    "position_x",
    "position_y",
    "position_z",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "quaternion_w",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
)
_TRACE_CONTROL_NAMES = ("motor_force_0", "motor_force_1", "motor_force_2", "motor_force_3")
_TRACE_BARRIER_NAMES = (
    "static_node",
    "dynamic_node",
    "arena",
    "speed",
    "angular_rate",
    "tilt",
    "static_swept",
    "dynamic_swept",
)
_DYNAMICS_PARAMETER_NAMES = (
    "mass_kg",
    "drag_acceleration_x",
    "drag_acceleration_y",
    "drag_acceleration_z",
    "wind_x",
    "wind_y",
    "wind_z",
    "rotor_efficiency_0",
    "rotor_efficiency_1",
    "rotor_efficiency_2",
    "rotor_efficiency_3",
)
_COMPILE_CACHE_KEYS = {
    "controller",
    "plant",
    "estimator",
    "bptt_startup",
    "bptt_online",
    "validation_startup",
    "validation_online",
}


@lru_cache(maxsize=1)
def _base_airborne_plant() -> tuple[VersionAModel, tuple[Any, Any, Any]]:
    """Load the exact tracked physical plant/actuator parameters used by campaign execution."""
    raw = load_params("cf21B_500")
    model = VersionAModel(
        mass=jnp.asarray(raw["mass"]),
        gravity_vec=jnp.asarray(raw["gravity_vec"]),
        inertia=jnp.asarray(raw["J"]),
        inertia_inv=jnp.linalg.inv(jnp.asarray(raw["J"])),
        drag_matrix=jnp.asarray(raw["drag_matrix"]),
        wind_velocity=jnp.zeros(3),
        external_force=jnp.zeros(3),
        external_torque=jnp.zeros(3),
    )
    actuator = (
        jnp.asarray(raw["L"]),
        jnp.asarray(raw["thrust2torque"]),
        jnp.asarray(raw["mixing_matrix"]),
    )
    return model, actuator


@lru_cache(maxsize=1)
def _base_airborne_motor_limits() -> tuple[np.ndarray, np.ndarray]:
    """Load the tracked per-rotor command bounds used by controller execution."""
    raw = load_params("cf21B_500")
    lower = np.broadcast_to(np.asarray(raw["thrust_min"], dtype=np.float32), (4,)).copy()
    upper = np.broadcast_to(np.asarray(raw["thrust_max"], dtype=np.float32), (4,)).copy()
    return lower, upper


@partial(jax.jit, static_argnames=("dt",))
def _replay_airborne_transitions(
    states: Any,
    filtered_motor_forces: Any,
    rotor_efficiency: Any,
    mass_scale: Any,
    drag_scale: Any,
    wind_velocity: Any,
    base_model: VersionAModel,
    arm_length: Any,
    thrust_to_torque: Any,
    mixing_matrix: Any,
    dt: Any,
) -> Any:
    """Replay every controller interval with the scalar execution plant's exact geometry.

    Production lowers a scalar ``state[13]``/``motor[4]`` plant step and invokes it once per
    control interval.  ``vmap`` is deliberately not used here: on GPU it lowers the inertia
    products to different batched kernels, and torque cancellation can then amplify harmless
    rounding differences by hundreds of ULPs.  ``lax.map`` retains scalar body shapes while still
    executing the complete evidence replay in one compiled call.
    """

    def replay_one(
        state: Any, command_motor: Any, efficiency: Any, mass: Any, drag: Any, wind: Any
    ) -> tuple[Any, Any]:
        model = base_model._replace(
            mass=base_model.mass * mass,
            drag_matrix=base_model.drag_matrix * drag[None, :],
            wind_velocity=wind,
        )
        realized_motor = command_motor * efficiency
        wrench = motor_forces_to_wrench(
            realized_motor,
            L=arm_length,
            thrust2torque=thrust_to_torque,
            mixing_matrix=mixing_matrix,
        )
        return direct_wrench_symplectic_step(state, wrench, model, dt), realized_motor

    return jax.lax.map(
        lambda arguments: replay_one(*arguments),
        (states, filtered_motor_forces, rotor_efficiency, mass_scale, drag_scale, wind_velocity),
    )


def validate_current_source_tree(run_directory: str | Path, *, repository: str | Path) -> str:
    """Require finalization/rendering code to equal the numerical run's recorded source tree."""
    root = Path(run_directory).resolve()
    config = validate_run_config(_read_json(root / "config.json"))
    parameters = config["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("campaign config parameters must be an object")
    recorded = parameters.get("source_tree_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("campaign config is missing a source-tree SHA-256 digest")
    current = _source_tree_sha256(Path(repository).resolve())
    if current != recorded:
        raise ValueError("current source tree differs from the numerical campaign source tree")
    return current


def _campaign_from_config(config: Mapping[str, Any]) -> CampaignConfig:
    from crazyflow.safety.da_plcbf.experiments import CampaignConfig, ExperimentConfig

    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("campaign config parameters must be an object")
    required = {
        "campaign_schema_version",
        "dynamic_prediction_contract",
        "trial",
        "root_seed",
        "fold_start",
        "intended_for_final_claim",
        "schedule_sha256",
        "source_tree_sha256",
    }
    if set(parameters) != required:
        raise ValueError("campaign config parameters do not match the canonical schema")
    raw_trial = parameters["trial"]
    if not isinstance(raw_trial, dict) or set(raw_trial) != set(
        ExperimentConfig.__dataclass_fields__
    ):
        raise ValueError("campaign trial configuration does not match ExperimentConfig")
    try:
        trial = ExperimentConfig(**raw_trial)
        campaign = CampaignConfig(
            trial=trial,
            methods=tuple(config["methods"]),
            conditions=tuple(config["conditions"]),
            trials_per_condition=config["trials_per_condition"],
            root_seed=parameters["root_seed"],
            fold_start=parameters["fold_start"],
            intended_for_final_claim=parameters["intended_for_final_claim"],
        )
        campaign.schedule()
    except (TypeError, ValueError) as error:
        raise ValueError(
            "campaign configuration cannot reconstruct its deterministic schedule"
        ) from error
    if config["experiment_id"] != "da-plcbf-scientific-campaign-v1":
        raise ValueError("campaign experiment_id is not the scientific campaign schema")
    if config["control_dt_seconds"] != trial.dt:
        raise ValueError("campaign control dt disagrees with its trial configuration")
    if config["horizon_steps"] != trial.certificate_horizon:
        raise ValueError("campaign horizon disagrees with its trial configuration")
    return campaign


def _validate_trace_physical_evidence(
    trace: Any, tape: ScenarioTape, trial: Any, *, condition: str
) -> None:
    """Replay the true plant and recompute physical margins/contact/failure from the tape."""
    from crazyflow.safety.da_plcbf.experiments import ConditionID, _barrier_trace

    if tuple(str(name) for name in trace.state_names) != _TRACE_STATE_NAMES:
        raise ValueError("campaign trace state names do not match the physical-state schema")
    if tuple(str(name) for name in trace.control_names) != _TRACE_CONTROL_NAMES:
        raise ValueError("campaign trace control names do not match the four-motor schema")
    if tuple(str(name) for name in trace.barrier_names) != _TRACE_BARRIER_NAMES:
        raise ValueError("campaign trace barrier names do not match the hard-evidence schema")
    if not np.array_equal(trace.time, tape.time[: trace.steps]):
        raise ValueError("campaign trace time grid does not match the scenario tape")
    expected_executed = np.arange(trace.steps) < trace.steps - 1
    if not np.array_equal(trace.executed_control, expected_executed):
        raise ValueError("campaign trace must contain one terminal no-control observation")

    expected_initial = np.concatenate(
        (
            tape.vehicle_initial_position,
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            tape.vehicle_initial_velocity,
            np.zeros(3),
        )
    ).astype(np.float32)
    if not np.array_equal(trace.true_state[0], expected_initial.astype(np.float64)):
        raise ValueError("campaign true-state initial condition does not match the scenario tape")

    selected_condition = ConditionID(condition)
    scheduled_dynamics = selected_condition in {
        ConditionID.DYNAMICS_CHANGE,
        ConditionID.FALSIFICATION_COMBINED,
    }
    intervals = trace.steps - 1
    executed_filtered = np.asarray(trace.filtered_control[:intervals], dtype=np.float64)
    if executed_filtered.shape != (intervals, 4) or not np.all(np.isfinite(executed_filtered)):
        raise ValueError("campaign executed filtered control is not finite four-rotor evidence")
    thrust_min, thrust_max = _base_airborne_motor_limits()
    lower = np.asarray(thrust_min, dtype=np.float64)
    upper = np.asarray(thrust_max, dtype=np.float64)
    epsilon = float(np.finfo(np.float32).eps)
    bound_tolerance = 8.0 * epsilon * (1.0 + np.maximum(np.abs(lower), np.abs(upper)))
    lower_excess = lower[None, :] - executed_filtered
    upper_excess = executed_filtered - upper[None, :]
    excess = np.maximum(lower_excess, upper_excess)
    if np.any(excess > bound_tolerance[None, :]):
        step, motor_index = np.unravel_index(
            int(np.argmax(excess - bound_tolerance[None, :])), excess.shape
        )
        raise ValueError(
            "campaign executed filtered control is outside tracked per-rotor thrust bounds "
            f"at step {step} ({_TRACE_CONTROL_NAMES[motor_index]}, "
            f"value={executed_filtered[step, motor_index]:.9g}, "
            f"lower={lower[motor_index]:.9g}, upper={upper[motor_index]:.9g}, "
            f"tolerance={bound_tolerance[motor_index]:.9g})"
        )
    mass_scale = (
        tape.mass_scale[:intervals] if scheduled_dynamics else np.ones(intervals, dtype=np.float64)
    )
    drag_scale = (
        tape.drag_scale[:intervals]
        if scheduled_dynamics
        else np.ones((intervals, 3), dtype=np.float64)
    )
    wind_velocity = (
        tape.wind_velocity[:intervals]
        if scheduled_dynamics
        else np.zeros((intervals, 3), dtype=np.float64)
    )
    rotor_efficiency = (
        tape.rotor_efficiency[:intervals]
        if scheduled_dynamics
        else np.ones((intervals, 4), dtype=np.float64)
    )
    base_model, actuator = _base_airborne_plant()
    replayed_raw, realized_motor_raw = _replay_airborne_transitions(
        jnp.asarray(trace.true_state[:-1], dtype=jnp.float32),
        jnp.asarray(trace.filtered_control[:-1], dtype=jnp.float32),
        jnp.asarray(rotor_efficiency, dtype=jnp.float32),
        jnp.asarray(mass_scale, dtype=jnp.float32),
        jnp.asarray(drag_scale, dtype=jnp.float32),
        jnp.asarray(wind_velocity, dtype=jnp.float32),
        base_model,
        *actuator,
        float(trial.dt),
    )
    replayed = np.asarray(replayed_raw, dtype=np.float64)
    realized_motor = np.asarray(realized_motor_raw, dtype=np.float64)
    actual_motor = np.asarray(trace.applied_control[:-1], dtype=np.float64)
    if realized_motor.shape != actual_motor.shape or not np.all(np.isfinite(realized_motor)):
        raise ValueError("campaign realized-motor replay returned invalid evidence")
    if not np.array_equal(realized_motor, actual_motor):
        motor_error = np.abs(realized_motor - actual_motor)
        step, motor_index = np.unravel_index(int(np.argmax(motor_error)), motor_error.shape)
        control_name = str(trace.control_names[motor_index])
        raise ValueError(
            "campaign applied control does not match filtered control and scheduled rotor "
            f"efficiency at step {step} ({control_name}, "
            f"abs_error={motor_error[step, motor_index]:.9g})"
        )
    if replayed.shape != trace.true_state[1:].shape or not np.all(np.isfinite(replayed)):
        raise ValueError("campaign true-state plant replay returned invalid evidence")
    actual_next = np.asarray(trace.true_state[1:], dtype=np.float64)
    tolerance = 8.0 * epsilon * (1.0 + np.maximum(np.abs(replayed), np.abs(actual_next)))
    error = np.abs(replayed - actual_next)
    if np.any(error > tolerance):
        step, state_index = np.unravel_index(int(np.argmax(error - tolerance)), error.shape)
        state_name = str(trace.state_names[state_index])
        raise ValueError(
            "campaign true-state transition does not replay from filtered control and scheduled "
            f"plant at step {step} ({state_name}, abs_error={error[step, state_index]:.9g}, "
            f"tolerance={tolerance[step, state_index]:.9g})"
        )

    barriers, contact, failure = _barrier_trace(trace.true_state, tape, trial)
    if not np.allclose(trace.hard_barriers, barriers, rtol=1e-12, atol=1e-12):
        raise ValueError("campaign hard barriers do not recompute from true state and tape")
    if not np.array_equal(trace.contact, contact):
        raise ValueError("campaign contact labels do not recompute from true state and tape")
    if not np.array_equal(trace.failure, failure):
        raise ValueError("campaign failure labels do not recompute from true state and tape")
    if np.any(failure & ~trace.degraded):
        raise ValueError("campaign physical failures must be explicit degraded intervals")


def _strict_numeric_mapping(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a nonempty object")
    result: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{name} must map strings to finite numbers")
        converted = float(raw)
        if not np.isfinite(converted) or converted < 0.0:
            raise ValueError(f"{name} values must be finite and nonnegative")
        result[key] = converted
    return result


def _method_claim_blockers(
    *, method: str, condition: str, trial: Any, trace: Any, events: tuple[Any, ...]
) -> tuple[str, ...]:
    from crazyflow.safety.da_plcbf.baselines import MethodID, method_spec
    from crazyflow.safety.da_plcbf.experiments import (
        AdaptationExecutionMode,
        ConditionID,
        _online_adaptation_lifecycle_blockers,
    )

    method_id = MethodID(method)
    blockers: list[str] = []
    if trial.policy_count != 64 and method_spec(method_id).uses_policy_library:
        blockers.append("development policy count is not K=64")
    if trial.certificate_horizon != 50 and method_spec(method_id).uses_policy_library:
        blockers.append("development certificate horizon is not H=50")
    if method_id is MethodID.OFFLINE_FROZEN_SDCBF_STYLE and not any(
        event.category == "offline_pretraining"
        and event.name == "generic_diversity_training_completed"
        for event in events
    ):
        blockers.append("offline generic-diversity BPTT did not produce a frozen learned library")
    online = method_id in {MethodID.DA_PLCBF_NO_ONLINE_MODEL_ADAPTATION, MethodID.DA_PLCBF_FULL}
    if online:
        blockers.extend(_online_adaptation_lifecycle_blockers(events))
        if (
            AdaptationExecutionMode(trial.adaptation_execution_mode)
            is AdaptationExecutionMode.REALTIME_PROBE
        ):
            blockers.append(
                "realtime_probe is hardware-feasibility evidence, not load-invariant "
                "safety evidence"
            )
    if (
        method_id is MethodID.DA_PLCBF_FULL
        and ConditionID(condition)
        in {ConditionID.DYNAMICS_CHANGE, ConditionID.FALSIFICATION_COMBINED}
        and not np.any(trace.model_version > 0)
    ):
        blockers.append("online estimator produced no accepted model update")
    return tuple(blockers)


def _validate_success_outcome(
    root: Path,
    mapping: Mapping[str, Any],
    assignment: TrialAssignment,
    tape: ScenarioTape,
    campaign: CampaignConfig,
    provenance: Mapping[str, Any],
) -> ScientificTrialRecord:
    from crazyflow.safety.da_plcbf.baselines import method_spec
    from crazyflow.safety.da_plcbf.experiments import replay_dashboard_dynamics_evidence
    from crazyflow.safety.da_plcbf.scientific_evaluation import derive_scientific_metrics

    record = _record_from_outcome(mapping, assignment)
    directory = root / "methods" / assignment.method / assignment.condition / str(assignment.fold)
    expected_names = {
        "trace.npz",
        "dashboard_evidence.npz",
        "events.jsonl",
        "metrics.json",
        "timing.json",
    }
    online_library = method_spec(assignment.method).online_library_updates
    if online_library:
        expected_names.add("adaptation_evidence.npz")
    if (
        not directory.is_dir()
        or {path.name for path in directory.iterdir() if path.is_file()} != expected_names
    ):
        raise ValueError("successful campaign outcome has an invalid method artifact set")
    trace = load_trace(directory / "trace.npz")
    if trace.content_sha256 != mapping["trace_content_sha256"]:
        raise ValueError("successful campaign outcome trace digest mismatch")
    if str(trace.scenario_tape_sha256) != tape.sha256:
        raise ValueError("successful campaign trace does not bind its scheduled tape")
    _validate_trace_physical_evidence(trace, tape, campaign.trial, condition=assignment.condition)
    events = load_events(directory / "events.jsonl", trace=trace)
    _validate_runtime_device_roles(events, provenance)
    if online_library:
        adaptation_evidence = load_adaptation_evidence(directory / "adaptation_evidence.npz")
        validate_adaptation_evidence_binding(
            adaptation_evidence,
            trace,
            events,
            shared_stochastic_seed=assignment.shared_stochastic_seed,
            tape=tape,
            condition=assignment.condition,
            method=assignment.method,
            config=campaign.trial,
            provenance=provenance,
        )
    load_metrics(directory / "metrics.json", trace=trace)
    timing = load_timing(directory / "timing.json", trace=trace)
    sidecar = load_dashboard_evidence(directory / "dashboard_evidence.npz")
    expected_dynamics = replay_dashboard_dynamics_evidence(
        trace, tape, assignment.condition, assignment.method, campaign.trial
    )
    validate_dashboard_evidence_binding(
        sidecar, trace, tape, events=events, expected_dynamics=expected_dynamics
    )

    compile_seconds = _strict_numeric_mapping(mapping["compile_seconds"], "compile_seconds")
    deadlines = _strict_numeric_mapping(mapping["deadlines_seconds"], "deadlines_seconds")
    timing_compile = {
        name: float(component["compile_seconds"])
        for name, component in timing["components"].items()
    }
    timing_deadlines = {
        name: float(component["deadline_seconds"])
        for name, component in timing["components"].items()
    }
    expected_deadlines = {
        "controller": campaign.trial.controller_deadline_seconds,
        "plant": campaign.trial.controller_deadline_seconds,
        "estimator_tick_work": campaign.trial.estimator_deadline_seconds,
        "command_preparation": campaign.trial.logging_deadline_seconds,
        "postprocessing": campaign.trial.logging_deadline_seconds,
        "wall_step": campaign.trial.dt,
        "command_ready": campaign.trial.dt,
    }
    if compile_seconds != timing_compile:
        raise ValueError("campaign outcome compile times disagree with timing evidence")
    if deadlines != timing_deadlines or deadlines != expected_deadlines:
        raise ValueError("campaign outcome deadlines disagree with timing/trial evidence")
    cache_hits = mapping["compile_cache_hits"]
    if (
        not isinstance(cache_hits, dict)
        or set(cache_hits) != _COMPILE_CACHE_KEYS
        or any(not isinstance(value, bool) for value in cache_hits.values())
    ):
        raise ValueError("campaign outcome compile-cache fields do not match the schema")

    if tuple(str(name) for name in sidecar.dynamics_parameter_names) != _DYNAMICS_PARAMETER_NAMES:
        raise ValueError("campaign dynamics evidence parameter names do not match the schema")
    if not np.all(sidecar.dynamics_true_available) or not np.all(
        sidecar.dynamics_estimated_available
    ):
        raise ValueError(
            "campaign scientific metrics require complete dynamics truth/estimate evidence"
        )
    raw_scale = mapping["estimation_scale"]
    if not isinstance(raw_scale, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_scale
    ):
        raise ValueError("campaign estimation_scale must be a numeric list")
    scale = np.asarray(raw_scale, dtype=np.float64)
    expected_scale = np.asarray(
        [float(sidecar.dynamics_true[0, 0]), 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )
    if not np.array_equal(scale, expected_scale):
        raise ValueError("campaign estimation_scale does not match predeclared physical scales")
    hard_certified = np.zeros_like(trace.policy_values, dtype=np.bool_)
    if method_spec(assignment.method).uses_policy_library:
        hard_certified = trace.executed_control[:, None] & (trace.policy_values >= 0.0)
    change_indices: tuple[int, ...] = ()
    if assignment.condition in {"dynamics_change", "falsification_combined"}:
        change_indices = tuple(
            sorted(
                {
                    int(index)
                    for index in tape.schedule_change_indices
                    if 0 <= int(index) < trace.steps - 1
                }
            )
        )
    derived = derive_scientific_metrics(
        trace,
        hard_certified_policy=hard_certified,
        estimation_error=sidecar.dynamics_estimated - sidecar.dynamics_true,
        estimation_scale=scale,
        change_indices=change_indices,
        latency_deadlines_seconds=deadlines,
        interval_safety_evidence=True,
        warm_execution_excludes_compilation=True,
    )
    if _metrics_mapping(derived) != mapping["scientific_metrics"]:
        raise ValueError("campaign outcome scientific metrics do not recompute from raw evidence")
    blockers = _method_claim_blockers(
        method=assignment.method,
        condition=assignment.condition,
        trial=campaign.trial,
        trace=trace,
        events=events,
    )
    if mapping["method_claim_eligible"] is not (not blockers):
        raise ValueError("campaign outcome method eligibility does not recompute from raw evidence")
    if mapping["claim_blockers"] != list(blockers):
        raise ValueError("campaign outcome claim blockers do not recompute from raw evidence")
    return record


def _load_strict_outcomes(
    root: Path, assignments: Mapping[tuple[str, str, int], TrialAssignment]
) -> dict[tuple[str, str, int], dict[str, Any]]:
    path = root / "aggregate" / "outcomes.jsonl"
    raw_file = path.read_bytes()
    outcomes: dict[tuple[str, str, int], dict[str, Any]] = {}

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate JSON key in campaign outcome")
        return dict(pairs)

    for line_number, raw in enumerate(raw_file.splitlines(), 1):
        if not raw or len(raw) > _MAX_JSONL_LINE_BYTES:
            raise ValueError(f"invalid campaign outcome line {line_number}")
        try:
            mapping = json.loads(raw, object_pairs_hook=pairs_hook)
            raw_assignment = mapping["assignment"]
            key = (raw_assignment["method"], raw_assignment["condition"], raw_assignment["fold"])
            assignment = assignments[key]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid campaign outcome line {line_number}") from error
        if not isinstance(mapping, dict) or key in outcomes:
            raise ValueError("campaign outcomes are malformed or duplicated")
        _record_from_outcome(mapping, assignment)
        outcomes[key] = mapping
    if set(outcomes) != set(assignments):
        raise ValueError("campaign outcomes do not cover the deterministic schedule")
    canonical = b"".join(_canonical_json_bytes(outcomes[item.key]) for item in assignments.values())
    if raw_file != canonical:
        raise ValueError("campaign outcomes JSONL is not in canonical schedule order")
    return outcomes


def validate_persisted_campaign_evidence(run_directory: str | Path) -> CampaignRun:
    """Canonically reconstruct numeric campaign claims from persisted raw evidence.

    This is the sole authority for promotion: stored eligibility booleans, comparison objects, and
    report prose are treated as assertions and must exactly equal a fresh reconstruction.
    """
    from crazyflow.safety.da_plcbf.baselines import MethodID
    from crazyflow.safety.da_plcbf.experiments import (
        CampaignRun,
        _campaign_online_snapshot_use,
        _campaign_paired_comparisons,
        _global_confirmatory_superiority_supported,
        generate_condition_tape,
    )

    root = Path(run_directory).resolve()
    config = validate_run_config(_read_json(root / "config.json"))
    seeds = validate_seeds(_read_json(root / "seeds.json"))
    campaign = _campaign_from_config(config)
    provenance = validate_provenance(_read_json(root / "provenance.json"))
    if campaign.intended_for_final_claim and provenance["git"]["dirty"]:
        raise ValueError("final-claim campaign provenance requires a clean committed source tree")
    schedule = campaign.schedule()
    parameters = config["parameters"]
    schedule_digest = _schedule_sha256(schedule)
    if parameters["schedule_sha256"] != schedule_digest:
        raise ValueError("campaign config schedule digest does not match reconstruction")
    if seeds["pairing_id"] != schedule_digest or seeds["root_seed"] != schedule.root_seed:
        raise ValueError("campaign seeds do not bind the reconstructed paired schedule")

    pair_assignments: dict[tuple[str, int], TrialAssignment] = {}
    for assignment in schedule.assignments:
        pair_assignments.setdefault((assignment.condition, assignment.fold), assignment)
    tapes: dict[tuple[str, int], ScenarioTape] = {}
    for pair, assignment in pair_assignments.items():
        condition, fold = pair
        path = root / "scenario_tapes" / condition / f"{fold}.npz"
        tape = load_scenario_tape(path)
        if (
            int(tape.root_seed) != assignment.scenario_root_seed
            or int(tape.generation_fold) != assignment.scenario_fold
        ):
            raise ValueError("campaign scenario tape seed/fold disagrees with schedule")
        regenerated = generate_condition_tape(
            condition,
            campaign.trial,
            seed=assignment.scenario_root_seed,
            fold=assignment.scenario_fold,
        )
        if tape.sha256 != regenerated.sha256:
            raise ValueError("campaign scenario tape does not regenerate from config/schedule")
        tapes[pair] = tape
    expected_seeds = validate_seeds(_seed_mapping(schedule, tapes))
    if seeds != expected_seeds:
        raise ValueError("campaign seed/tape mapping does not match deterministic reconstruction")

    expected_config = _campaign_config_mapping(
        campaign, schedule, Path(__file__).resolve().parents[3]
    )
    expected_config["parameters"]["source_tree_sha256"] = parameters["source_tree_sha256"]
    if config != validate_run_config(expected_config):
        raise ValueError("campaign config does not match canonical reconstruction")

    assignments = {assignment.key: assignment for assignment in schedule.assignments}
    outcomes = _load_strict_outcomes(root, assignments)
    records: list[ScientificTrialRecord] = []
    ineligible: list[tuple[str, str, int]] = []
    for assignment in schedule.assignments:
        mapping = outcomes[assignment.key]
        tape = tapes[(assignment.condition, assignment.fold)]
        if mapping["scenario_tape_sha256"] != tape.sha256:
            raise ValueError("campaign outcome scenario digest disagrees with scheduled tape")
        record = _record_from_outcome(mapping, assignment)
        if record.status is TrialStatus.COMPLETE:
            record = _validate_success_outcome(
                root, mapping, assignment, tape, campaign, provenance
            )
            if mapping["method_claim_eligible"] is not True:
                ineligible.append(assignment.key)
        records.append(record)
    trace_keys = {
        (path.parts[-4], path.parts[-3], int(path.parts[-2]))
        for path in root.glob("methods/*/*/*/trace.npz")
    }
    success_keys = {record.key for record in records if record.status is TrialStatus.COMPLETE}
    if trace_keys != success_keys:
        raise ValueError("campaign success outcomes and trace directories disagree")

    dataset = PairedTrialDataset(schedule=schedule, records=tuple(records))
    dataset.validate()
    inference, exploratory_inference, comparisons = _campaign_paired_comparisons(dataset)
    blockers: list[str] = []
    failed = [record.key for record in records if record.status is TrialStatus.EXECUTION_FAILURE]
    if failed:
        blockers.append(f"{len(failed)} scheduled executions failed")
    if ineligible:
        blockers.append(f"{len(ineligible)} completed runs failed method claim gates")
    online_methods = {
        MethodID.DA_PLCBF_NO_ONLINE_MODEL_ADAPTATION.value,
        MethodID.DA_PLCBF_FULL.value,
    }.intersection(campaign.methods)
    proven = _campaign_online_snapshot_use(campaign.methods, (), root)
    for method in sorted(online_methods - proven):
        blockers.append(
            f"online method {method} never proved an admitted snapshot drove executed control"
        )
    blockers.extend(campaign.final_contract_blockers())
    if not schedule.final_claim_eligible:
        blockers.append("schedule is not a predeclared >=100-pair final-claim schedule")
    run = CampaignRun(
        schedule=schedule,
        trial_runs=(),
        records=tuple(records),
        paired_comparisons=comparisons,
        inference_config=inference,
        exploratory_inference_config=exploratory_inference,
        execution_complete=not failed,
        scientific_claim_eligible=not blockers,
        global_confirmatory_superiority_supported=(
            _global_confirmatory_superiority_supported(comparisons)
        ),
        claim_blockers=tuple(blockers),
    )
    comparison_path = root / "aggregate" / "paired_comparisons.json"
    if comparison_path.read_bytes() != _canonical_json_bytes(_comparison_payload(run)):
        raise ValueError("paired comparisons do not match canonical evidence reconstruction")
    report_path = root / "aggregate" / "scientific_report.md"
    if report_path.read_bytes() != _report_text(run).encode():
        raise ValueError("scientific report does not match canonical evidence reconstruction")
    return run


class CampaignArtifactStore:
    """Crash-safe campaign progress store with strict compatibility checks on resume."""

    def __init__(
        self,
        root: str | Path,
        campaign: CampaignConfig,
        schedule: PairedTrialSchedule,
        tapes: Mapping[tuple[str, int], ScenarioTape],
        *,
        repository: str | Path,
        resume: bool,
    ) -> None:
        self.root = Path(root).resolve()
        self.campaign = campaign
        self.schedule = schedule
        self.tapes = dict(tapes)
        self.repository = Path(repository).resolve()
        self.config = validate_run_config(
            _campaign_config_mapping(campaign, schedule, self.repository)
        )
        self.seeds = validate_seeds(_seed_mapping(schedule, self.tapes))
        self._assignments = {assignment.key: assignment for assignment in schedule.assignments}
        self._outcomes: dict[tuple[str, str, int], dict[str, Any]] = {}
        if resume:
            self._open_existing()
        else:
            self._create_new()

    @property
    def outcomes_path(self) -> Path:
        return self.root / "aggregate" / "outcomes.jsonl"

    def _create_new(self) -> None:
        provenance = collect_provenance(self.repository)
        if self.campaign.intended_for_final_claim and provenance["git"]["dirty"]:
            raise ValueError("final-claim campaign creation requires a clean committed source tree")
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "aggregate").mkdir()
        (self.root / "methods").mkdir()
        (self.root / "scenario_tapes").mkdir()
        write_run_config(self.config, self.root / "config.json")
        write_provenance(provenance, self.root / "provenance.json")
        for (condition, fold), tape in sorted(self.tapes.items()):
            destination = self.root / "scenario_tapes" / condition / f"{fold}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_scenario_tape(tape, destination)
        write_seeds(self.seeds, self.root / "seeds.json")
        _atomic_replace(self.outcomes_path, b"")

    def _open_existing(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError("resume directory does not exist")
        if (self.root / "manifest.json").exists() or (self.root / "SHA256SUMS").exists():
            raise ValueError("a finalized immutable campaign cannot be resumed")
        actual_config = validate_run_config(_read_json(self.root / "config.json"))
        if actual_config != self.config:
            raise ValueError("resume config/schedule/source digest is incompatible")
        stored_provenance = validate_provenance(_read_json(self.root / "provenance.json"))
        if self.campaign.intended_for_final_claim and stored_provenance["git"]["dirty"]:
            raise ValueError("final-claim campaign resume requires clean committed provenance")
        actual_seeds = validate_seeds(_read_json(self.root / "seeds.json"))
        if actual_seeds != self.seeds:
            raise ValueError("resume seed/tape mapping is incompatible")
        for (condition, fold), expected in sorted(self.tapes.items()):
            loaded = load_scenario_tape(self.root / "scenario_tapes" / condition / f"{fold}.npz")
            if loaded.sha256 != expected.sha256:
                raise ValueError("resume scenario tape digest mismatch")
        self._load_outcomes()
        pending = set(self._assignments).difference(self._outcomes)
        current_provenance = collect_provenance(self.repository)
        _validate_resume_analysis_identity(stored_provenance, current_provenance)
        if pending:
            _validate_resume_execution_identity(stored_provenance, current_provenance)
        self.validate_recorded_successes()

    def _load_outcomes(self) -> None:
        if not self.outcomes_path.is_file():
            raise FileNotFoundError(self.outcomes_path)
        for line_number, raw in enumerate(self.outcomes_path.read_bytes().splitlines(), 1):
            if len(raw) > _MAX_JSONL_LINE_BYTES:
                raise ValueError(f"outcome line {line_number} exceeds size limit")
            try:
                mapping = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid outcome JSON line {line_number}") from error
            if not isinstance(mapping, dict) or not isinstance(mapping.get("assignment"), dict):
                raise ValueError("outcome line must contain an assignment object")
            raw_assignment = mapping["assignment"]
            try:
                key = (
                    raw_assignment["method"],
                    raw_assignment["condition"],
                    raw_assignment["fold"],
                )
                assignment = self._assignments[key]
            except (KeyError, TypeError) as error:
                raise ValueError("outcome assignment is absent from schedule") from error
            _record_from_outcome(mapping, assignment)
            if key in self._outcomes:
                raise ValueError("duplicate campaign outcome")
            self._outcomes[key] = mapping

    def _flush_outcomes(self) -> None:
        lines = []
        for assignment in self.schedule.assignments:
            mapping = self._outcomes.get(assignment.key)
            if mapping is not None:
                lines.append(_canonical_json_bytes(mapping))
        _atomic_replace(self.outcomes_path, b"".join(lines))

    def completed_keys(self) -> frozenset[tuple[str, str, int]]:
        """Return exact scheduled outcomes already committed to the progress log."""
        return frozenset(self._outcomes)

    def ineligible_success_keys(self) -> tuple[tuple[str, str, int], ...]:
        """Return completed controller runs that failed one or more method-level claim gates."""
        return tuple(
            assignment.key
            for assignment in self.schedule.assignments
            if (mapping := self._outcomes.get(assignment.key)) is not None
            and mapping["status"] == TrialStatus.COMPLETE.value
            and mapping["method_claim_eligible"] is not True
        )

    def record(
        self, assignment: TrialAssignment, record: ScientificTrialRecord, run: TrialRun | None
    ) -> None:
        """Atomically append one never-overwritten scheduled outcome."""
        if assignment.key in self._outcomes:
            raise FileExistsError(f"outcome already recorded: {assignment.key}")
        self._outcomes[assignment.key] = outcome_mapping(assignment, record, run)
        self._flush_outcomes()

    def records(self) -> tuple[ScientificTrialRecord, ...]:
        values = []
        for assignment in self.schedule.assignments:
            mapping = self._outcomes.get(assignment.key)
            if mapping is not None:
                values.append(_record_from_outcome(mapping, assignment))
        return tuple(values)

    def validate_recorded_successes(self) -> None:
        """Validate every trace, sidecar, event, metric, and timing file before resume skips it."""
        provenance = validate_provenance(_read_json(self.root / "provenance.json"))
        success_keys = {
            key
            for key, mapping in self._outcomes.items()
            if mapping["status"] == TrialStatus.COMPLETE.value
        }
        trace_paths = sorted(self.root.glob("methods/*/*/*/trace.npz"))
        actual_keys = {
            (path.parts[-4], path.parts[-3], int(path.parts[-2])) for path in trace_paths
        }
        if actual_keys != success_keys:
            raise ValueError("recorded success outcomes and method artifact directories disagree")
        for key in sorted(success_keys):
            method, condition, fold = key
            mapping = self._outcomes[key]
            directory = self.root / "methods" / method / condition / str(fold)
            expected_names = {
                "trace.npz",
                "dashboard_evidence.npz",
                "events.jsonl",
                "metrics.json",
                "timing.json",
            }
            from crazyflow.safety.da_plcbf.baselines import method_spec

            online_library = method_spec(method).online_library_updates
            if online_library:
                expected_names.add("adaptation_evidence.npz")
            if {path.name for path in directory.iterdir() if path.is_file()} != expected_names:
                raise ValueError("resume method artifact member set is invalid")
            trace = load_trace(directory / "trace.npz")
            if trace.content_sha256 != mapping["trace_content_sha256"]:
                raise ValueError("resume trace content digest mismatch")
            validate_trace_scenario_binding(trace, condition=condition, fold=fold, seeds=self.seeds)
            events = load_events(directory / "events.jsonl", trace=trace)
            _validate_runtime_device_roles(events, provenance)
            if online_library:
                adaptation_evidence = load_adaptation_evidence(
                    directory / "adaptation_evidence.npz"
                )
                validate_adaptation_evidence_binding(
                    adaptation_evidence,
                    trace,
                    events,
                    shared_stochastic_seed=self._assignments[key].shared_stochastic_seed,
                    tape=self.tapes[(condition, fold)],
                    condition=condition,
                    method=method,
                    config=self.campaign.trial,
                    provenance=provenance,
                )
            load_metrics(directory / "metrics.json", trace=trace)
            load_timing(directory / "timing.json", trace=trace)
            evidence = load_dashboard_evidence(directory / "dashboard_evidence.npz")
            from crazyflow.safety.da_plcbf.experiments import replay_dashboard_dynamics_evidence

            validate_dashboard_evidence_binding(
                evidence,
                trace,
                self.tapes[(condition, fold)],
                events=events,
                expected_dynamics=replay_dashboard_dynamics_evidence(
                    trace, self.tapes[(condition, fold)], condition, method, self.campaign.trial
                ),
            )

    def finalize_numeric(self, run: CampaignRun) -> None:
        """Write paired comparisons/report only after the full scheduled matrix is retained."""
        provenance = validate_provenance(_read_json(self.root / "provenance.json"))
        if self.campaign.intended_for_final_claim and provenance["git"]["dirty"]:
            raise ValueError(
                "final-claim campaign finalization requires clean committed provenance"
            )
        if set(self._outcomes) != set(self._assignments):
            raise ValueError("cannot finalize an incomplete scheduled outcome matrix")
        if self.records() != run.records:
            raise ValueError("campaign result disagrees with persisted outcomes")
        dataset = PairedTrialDataset(run.schedule, run.records)
        dataset.validate()
        aggregate = self.root / "aggregate"
        rows = []
        for record in run.records:
            if record.status is not TrialStatus.COMPLETE:
                continue
            directory = self.root / "methods" / record.method / record.condition / str(record.fold)
            rows.append(
                aggregate_row(
                    record.method,
                    record.condition,
                    record.fold,
                    load_metrics(directory / "metrics.json"),
                )
            )
        if not rows:
            raise ValueError("cannot finalize a campaign without any successful numerical trace")
        with tempfile.TemporaryDirectory(prefix=".numeric-finalize-", dir=aggregate) as directory:
            staging = Path(directory)
            write_paired_metrics_csv(rows, staging / "paired_metrics.csv")
            write_confidence_intervals(rows, staging / "confidence_intervals.json")
            write_aggregate_report(
                rows, staging / "report.md", scientific_evidence=run.scientific_claim_eligible
            )
            for name in ("paired_metrics.csv", "confidence_intervals.json", "report.md"):
                os.replace(staging / name, aggregate / name)
        _atomic_replace(
            aggregate / "paired_comparisons.json", _canonical_json_bytes(_comparison_payload(run))
        )
        _atomic_replace(aggregate / "scientific_report.md", _report_text(run).encode())


__all__ = [
    "CAMPAIGN_COMPARISON_SCHEMA_VERSION",
    "CAMPAIGN_OUTCOME_SCHEMA_VERSION",
    "CampaignArtifactStore",
    "outcome_mapping",
    "validate_current_source_tree",
    "validate_persisted_campaign_evidence",
]
