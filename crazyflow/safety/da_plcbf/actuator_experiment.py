"""Complete actuator-study episodes with causal command and publication clocks.

Deterministic execution forces completed finite updates at exogenous opportunities
and makes no real-time claim. Paced execution measures serialized service against
wall-clock boundaries, while retaining the nondelayed physical command model.
Delayed execution advances the old physical command during measured controller
service, then the new command during learner service and the remaining command
period. Simulator and collision-audit overhead is recorded separately and is
never added a second time as sensing-to-actuation latency.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from crazyflow.safety.da_plcbf.actuator_inputs import CausalObservationInputCache
from crazyflow.safety.da_plcbf.actuator_learning import (
    build_actuator_skill_learner,
    save_actuator_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_study import (
    ALL_METHODS,
    ActuatorObservationConfig,
    build_actuator_controller,
    initial_augmented_state,
    make_actuator_plant,
    nominal_actuator_model,
    observe_actuator_state,
    observed_actuator_model,
)
from crazyflow.safety.da_plcbf.case_study_world import (
    CF21B_XML_COLLIDER_OFFSET_BODY_M,
    CF21B_XML_COLLIDER_RADIUS_M,
)
from crazyflow.safety.da_plcbf.deadline_schedule import BoundarySnapshotScheduler, CompletedSnapshot
from crazyflow.safety.da_plcbf.navigation_experiment import (
    evaluate_collision_termination,
    summarize_collision_observation,
)
from crazyflow.safety.da_plcbf.navigation_world import nominal_encounter_metrics

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.actuator_learning import (
        ActuatorLearnerCheckpoint,
        ActuatorLearnerState,
    )
    from crazyflow.safety.da_plcbf.actuator_study import ActuatorScene


@dataclass(frozen=True, slots=True)
class ActuatorEpisodeConfig:
    """One resolved method/timing configuration; every method shares physical bounds."""

    method: str = "A"
    execution_mode: str = "deterministic"
    plant_level: str = "P0"
    plant_step_seconds: float = 0.005
    filter_config: ActuatorFilterConfig = field(default_factory=ActuatorFilterConfig)
    observation_config: ActuatorObservationConfig = field(default_factory=ActuatorObservationConfig)
    cache_observation_inputs: bool = True
    learning_start_seconds: float = 0.0
    update_every_controls: int = 1
    controller_reserve_seconds: float = 0.003
    update_safety_factor: float = 1.25
    freeze_learning_at: float | None = None
    revert_control_params_at: float | None = None
    capture_times: tuple[float, ...] = ()
    retain_rollouts: bool = False
    save_checkpoints: bool = False
    warmup_calls: int = 3

    def validate(self, scene: ActuatorScene, bundle: ActuatorLearnerCheckpoint) -> None:
        """Reject mismatched clocks and mislabeled method or causal intervention settings."""
        self.filter_config.validate()
        self.observation_config.validate()
        if type(self.cache_observation_inputs) is not bool:
            raise ValueError("cache_observation_inputs must be a boolean")
        bundle.config.validate()
        if self.method not in ALL_METHODS:
            raise ValueError("unknown actuator study method")
        if self.execution_mode not in {"deterministic", "paced", "delayed"}:
            raise ValueError("execution_mode must be deterministic, paced, or delayed")
        if self.plant_level not in {"P0", "P1", "P2"}:
            raise ValueError("plant_level must be P0, P1, or P2")
        for name in ("plant_step_seconds", "update_safety_factor"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.update_safety_factor < 1:
            raise ValueError("update_safety_factor must be at least one")
        for name in ("learning_start_seconds", "controller_reserve_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.update_every_controls) is not int or self.update_every_controls < 1:
            raise ValueError("update_every_controls must be a positive integer")
        if type(self.warmup_calls) is not int or self.warmup_calls < 0:
            raise ValueError("warmup_calls must be a nonnegative integer")
        if self.execution_mode != "deterministic" and self.warmup_calls < 2:
            raise ValueError("measured scheduling requires at least two disposable warmup calls")
        period = self.filter_config.command_period
        if self.plant_step_seconds > period:
            raise ValueError("plant step must not exceed the command period")
        if (
            self.filter_config.dt != bundle.config.dt
            or self.filter_config.horizon != bundle.config.horizon
            or self.filter_config.command_hold_steps != bundle.config.control_interval_steps
            or not math.isclose(scene.world.config.control_period, period, abs_tol=1e-12)
            or not math.isclose(scene.world.config.dt, bundle.config.dt, abs_tol=1e-12)
        ):
            raise ValueError("checkpoint, prediction, world, and command-hold clocks must match")
        if self.method == "A1" and int(bundle.contract.spec.latent_codes.shape[0]) != 1:
            raise ValueError("A1 requires a separately prepared single-recovery checkpoint")
        if self.method not in {"F0", "F1"} and bundle.config.adapter_mode != "F2":
            raise ValueError("primary methods and A1 must preserve the shared F2 adapter")
        for name in ("freeze_learning_at", "revert_control_params_at"):
            when = getattr(self, name)
            if when is not None and (
                not math.isfinite(when)
                or when < 0
                or not math.isclose(when / period, round(when / period), abs_tol=1e-8)
            ):
                raise ValueError(f"{name} must lie on a nonnegative control boundary")
        if self.revert_control_params_at is not None and (
            self.freeze_learning_at is None
            or self.freeze_learning_at > self.revert_control_params_at
        ):
            raise ValueError(
                "parameter reversion requires learning frozen at or before that boundary"
            )
        if any(
            not math.isfinite(t) or t < 0 or t > scene.world.config.duration_seconds
            for t in self.capture_times
        ):
            raise ValueError("capture_times must lie in the physical episode")
        if tuple(sorted(set(self.capture_times))) != tuple(self.capture_times):
            raise ValueError("capture_times must increase strictly")
        if scene.world.config.payload_events or scene.world.config.wind_events:
            # HoverEncounterWorld historically stores an explicitly zero wind event.
            if scene.world.config.payload_events or any(
                np.any(np.asarray(event.velocity) != 0) for event in scene.world.config.wind_events
            ):
                raise ValueError("initial actuator study requires unchanged body and zero wind")


@dataclass(frozen=True, slots=True)
class ActuatorEpisodeResult:
    """Completed or explicitly incomplete episode evidence and continuation state."""

    summary: dict[str, Any]
    control_traces: dict[str, np.ndarray]
    dense_traces: dict[str, np.ndarray]
    final_state: np.ndarray
    final_learner_state: ActuatorLearnerState
    artifacts: dict[str, Path]


def _synchronize(value: Any) -> Any:
    return jax.block_until_ready(value)


def _jsonable(value: Any) -> Any:
    """Keep artifacts strict JSON, preserving nonfinite evidence as explicit nulls."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "_asdict"):
        return _jsonable(value._asdict())
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.ndarray, jax.Array)):
        return _jsonable(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _hash_tree(value: Any) -> str:
    """Hash tree structure, dtype, shape and every component, including motor state."""
    leaves, structure = jax.tree_util.tree_flatten(value)
    digest = hashlib.sha256(str(structure).encode())
    for leaf in leaves:
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _rows_to_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Numeric/text rows only, never object arrays or pickle-dependent artifacts."""
    if not rows:
        return {}
    result = {}
    for key in sorted(set.intersection(*(set(row) for row in rows))):
        if any(isinstance(row[key], dict) or row[key] is None for row in rows):
            continue
        array = np.asarray([row[key] for row in rows])
        if array.dtype != object:
            result[key] = array
    return result


def _audit_interval(world: Any, times: np.ndarray, states: np.ndarray) -> dict[str, Any]:
    return evaluate_collision_termination(
        world, times, states[:, :13], termination_geometry="modeled_collider"
    )


def _strictly_separated(audited: dict[str, Any]) -> bool:
    """Skip per-node audits only for a strictly separated complete recorded hold.

    This uses the unchanged 1 ms swept-geometry calculation and its lower bounds.
    A small positive roundoff guard sends near-contact, unresolved or incomplete
    evidence through the original chronological adjacent-node audit instead.
    """
    if audited["terminate"]:
        return False
    for name in ("actual_xml_sphere_geometry", "actual_xml_ground_geometry"):
        geometry = audited.get("audit", {}).get(name, {})
        lower = geometry.get("minimum_clearance_lower_bound_m")
        if lower is None:
            if geometry.get("intersection_classification") == "no_obstacles":
                continue
            return False
        if not math.isfinite(lower) or lower <= 1e-9:
            return False
    return True


class _RecordedActuatorModels:
    """Cache exact host fields for each active phase solely for plant telemetry.

    The controller and learner still receive their own current/past observations.
    No model is evaluated here until that phase is physically queried, and every
    returned array keeps the original model's dtype and bit pattern.
    """

    def __init__(self, scene: ActuatorScene, nominal: Any) -> None:
        self.scene = scene
        self.nominal = nominal
        self.cache: dict[bool, tuple[np.ndarray, ...]] = {}

    def at(self, when: float) -> tuple[np.ndarray, ...]:
        changed = when >= self.scene.event_time - 1e-10
        if self.scene.recovery_time is not None and when >= self.scene.recovery_time - 1e-10:
            changed = False
        if changed not in self.cache:
            model = self.scene.model_at(when, self.nominal)
            values = tuple(
                np.asarray(getattr(model, name)).copy()
                for name in ("command_lower", "command_upper", "effectiveness", "time_constants")
            )
            for array in values:
                array.flags.writeable = False
            self.cache[changed] = values
        return self.cache[changed]


def _initial_collision(world: Any, state: np.ndarray) -> dict[str, Any] | None:
    """Exact time-zero geometry needs no invented second trajectory sample."""
    center = state[:3] + Rotation.from_quat(state[3:7]).apply(CF21B_XML_COLLIDER_OFFSET_BODY_M)
    obstacles, _ = world.obstacle_kinematics(np.asarray([0.0]))
    clearance = np.linalg.norm(center - obstacles[0], axis=-1) - (
        np.asarray(world.obstacle_radii) + CF21B_XML_COLLIDER_RADIUS_M
    )
    kind = (
        "modeled_collider_floor"
        if center[2] - CF21B_XML_COLLIDER_RADIUS_M < 0
        else "modeled_collider_obstacle"
        if np.any(clearance < 0)
        else None
    )
    return (
        {
            "terminate": True,
            "termination_geometry": "modeled_collider",
            "collision_kind": kind,
            "first_intersection_time_seconds": 0.0,
            "initial_exact_geometry": True,
        }
        if kind is not None
        else None
    )


def _operational_margins(world: Any, state: np.ndarray, clearance: float = 0.08) -> np.ndarray:
    cfg = world.config
    lower, upper = np.asarray(cfg.arena_lower), np.asarray(cfg.arena_upper)
    q = state[3:7] / np.linalg.norm(state[3:7])
    cosine = np.cos(cfg.tilt_max_radians)
    return np.concatenate(
        (
            (state[:3] - lower - clearance) / (upper - lower),
            (upper - clearance - state[:3]) / (upper - lower),
            [1 - np.sum(state[7:10] ** 2) / cfg.speed_max**2],
            [1 - np.sum(state[10:13] ** 2) / cfg.angular_rate_max**2],
            [(1 - 2 * (q[0] ** 2 + q[1] ** 2) - cosine) / (1 - cosine)],
        )
    )


def _held_record(check: Any, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": np.asarray(getattr(check, key))
        for key in (
            "collision_margin",
            "operational_margins",
            "command_motor_margin",
            "finite",
            "command_motor_passed",
            "collision_passed",
            "operational_passed",
            "passed",
        )
    }


def _filter_record(step: Any, retain_rollouts: bool) -> dict[str, Any]:
    cert = step.certificates
    selected = int(step.selected_index)
    safe_index = max(0, selected)
    modes = ("qp", "fallback", "emergency", "degraded", "invalid_input")
    record = {
        "mode": modes[int(step.execution_mode)],
        "input_valid": bool(cert.input_valid),
        "hard": np.asarray(cert.hard.values),
        "smooth": np.asarray(cert.smooth_values),
        "candidate_valid": np.asarray(cert.rollouts.valid),
        "gradient_valid": np.asarray(cert.gradient_valid),
        "eligible": np.asarray(cert.eligible),
        "eligible_policy_count": int(np.sum(cert.eligible)),
        "qp_solved_policy_count": int(selected >= 0),
        "qp_feasible_selected_policy_count": int(bool(step.qp_valid)),
        "command_volume_fractions": np.asarray(cert.command_volume_fractions),
        "selected_index": selected,
        "selected_row": np.asarray(cert.rows[safe_index]),
        "selected_bound": np.asarray(cert.bounds[safe_index]),
        "selected_gradient": np.asarray(cert.gradients[safe_index]),
        "gradient_components_computed": np.asarray(cert.gradient_components_computed),
        "selected_drift_derivative": np.asarray(cert.drift_derivatives[safe_index]),
        "selected_time_derivative": np.asarray(cert.time_derivatives[safe_index]),
        "selected_policy_dual": np.asarray(step.selected_policy_dual),
        "executed_policy_dual": np.asarray(step.executed_policy_dual),
        "policy_residual": np.asarray(step.policy_residual),
        "qp_rejection_flags": np.asarray(step.qp_rejection_flags),
        "qp_valid": bool(step.qp_valid),
        "fallback_valid": bool(step.fallback_valid),
        "emergency_valid": bool(step.emergency_valid),
        "degraded": bool(step.degraded),
        "sqp_iterations": int(step.sqp_iterations),
        "intervention_norm": float(step.intervention_norm),
        "nominal_command": np.asarray(step.nominal_action),
        "next_estimated_state": np.asarray(step.next_estimated_state),
        "effective_temperature": np.asarray(cert.effective_temperature),
        "smooth_gap_bound": np.asarray(cert.smooth_gap_bound),
    }
    for name in ("applied", "qp_check", "fallback_check", "emergency_check"):
        record.update(_held_record(getattr(step, name), name))
    if retain_rollouts:
        record["candidate_states"] = np.asarray(cert.rollouts.states)
        record["candidate_commands"] = np.asarray(cert.rollouts.commands)
    return record


def _model_record(model: Any, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_effectiveness": np.asarray(model.effectiveness),
        f"{prefix}_time_constants": np.asarray(model.time_constants),
        f"{prefix}_mass": np.asarray(model.body.mass),
        f"{prefix}_inertia": np.asarray(model.body.inertia),
        f"{prefix}_model_sha256": _hash_tree(model),
    }


def _next_grid(when: float, period: float, minimum_tick: int) -> tuple[int, float]:
    tick = max(minimum_tick, int(math.ceil((when - 1e-10) / period)))
    return tick, tick * period


def _checkpoint_provenance(bundle: Any, method: str) -> dict[str, Any]:
    metadata = bundle.metadata
    mode = metadata.get("mode")
    if method == "DR" and mode != "dr":
        raise ValueError("DR requires a separately trained checkpoint with metadata mode=dr")
    if method == "A1" and mode != "single":
        raise ValueError("A1 requires a separately trained checkpoint with metadata mode=single")
    if method in {"F0", "F1", "F2", "A", "OPT"} and mode not in {None, "nominal"}:
        raise ValueError("this method must use the common nominal-library checkpoint")
    reports = {}
    for filename in ("final_training.json", "final_validation.json", "summary.json"):
        path = Path(bundle.json_path).parent / filename
        if path.is_file():
            reports[filename] = {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "report": json.loads(path.read_text()),
            }
    return {
        "checkpoint_npz": str(Path(bundle.npz_path).resolve()),
        "checkpoint_json": str(Path(bundle.json_path).resolve()),
        "checkpoint_sha256": bundle.sha256,
        "metadata": metadata,
        "competence_reports": reports,
        "competence_checked_by_runtime": False,
        "runtime_finite_update_quality_gate": False,
    }


def run_actuator_episode(
    scene: ActuatorScene,
    bundle: ActuatorLearnerCheckpoint,
    config: ActuatorEpisodeConfig,
    outdir: str | Path,
    *,
    controllerfunctions: Any = None,
    learner_functions: Any = None,
    opt_controller: Any = None,
    progress_callback: Any = None,
    snapshot_callback: Any = None,
) -> ActuatorEpisodeResult:
    """Run one complete episode, retaining evidence even when numerical service fails.

    The output directory is exclusive. Callbacks receive a dictionary containing an
    absolute physical time and a complete augmented state; their service is reported
    separately. Checkpoint serialization and final reporting occur after execution.
    A returned incomplete result is never a completed or collision-free episode.
    """
    config.validate(scene, bundle)
    provenance = _checkpoint_provenance(bundle, config.method)
    directory = Path(outdir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    events_path = directory / "events.jsonl"
    artifacts: dict[str, Path] = {"events": events_path}
    events_stream = events_path.open("x")
    controls: list[dict[str, Any]] = []
    dense: list[dict[str, Any]] = []
    applications: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    captures: list[tuple[str, Any, np.ndarray, dict[str, Any]]] = []
    capture_records: list[dict[str, Any]] = []
    deferred_captures: dict[float, tuple[Any, np.ndarray, dict[str, Any]]] = {}
    callback_seconds = 0.0
    simulator_seconds = 0.0
    warmup: dict[str, Any] = {"controller_seconds": [], "learner_seconds": []}
    period = config.filter_config.command_period
    duration = float(scene.world.config.duration_seconds)
    world = scene.world
    learns = config.method in {"A", "A1"}
    nominal = nominal_actuator_model()
    input_cache = (
        CausalObservationInputCache(scene, nominal, config.observation_config, config.filter_config)
        if config.cache_observation_inputs
        else None
    )
    recorded_models = _RecordedActuatorModels(scene, nominal)
    initial = initial_augmented_state(world.initial_state, nominal)
    final_state = initial.copy()
    learner_state = bundle.state
    initial_params = bundle.state.params
    initial_version = int(bundle.state.library_version)
    scheduler: BoundarySnapshotScheduler | None = None
    plant = None
    termination: str | None = None
    collision: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    interrupt: BaseException | None = None
    arrivals: list[float] = []
    freeze_boundary: float | None = None
    revert_boundary: float | None = None
    epoch = time.perf_counter()
    current_command = initial[13:].copy()
    previous_index = 0
    previous_plan = None
    nominal_command_for_control = None
    captured_times: set[float] = set()

    def event(kind: str, **payload: Any) -> None:
        events_stream.write(
            json.dumps(_jsonable({"event": kind, **payload}), sort_keys=True, allow_nan=False)
            + "\n"
        )
        events_stream.flush()

    def credit_waypoint(when: float, actual: np.ndarray) -> None:
        """Credit the current goal only after the preceding physical interval was audited."""
        if (
            collision is None
            and when >= scene.navigation_start - 1e-10
            and len(arrivals) < len(world.waypoint_positions)
        ):
            goal_now = world.waypoint_positions[len(arrivals)]
            if np.linalg.norm(actual[:3] - goal_now) <= world.config.reach_radius:
                arrivals.append(when)
                event("waypoint_arrival", time=when, waypoint_index=len(arrivals) - 1, state=actual)

    def callback(function: Any, payload: dict[str, Any]) -> None:
        nonlocal callback_seconds
        if function is not None:
            started = time.perf_counter()
            function(payload)
            callback_seconds += time.perf_counter() - started

    def capture_context() -> tuple[Any, dict[str, Any]]:
        published = bundle.state if scheduler is None else scheduler.published.state
        selected_params = initial_params if revert_boundary is not None else published.params
        return published, {
            "published_version": int(published.library_version),
            "snapshot_training_time_seconds": scheduler.published.training_simulation_time
            if scheduler is not None
            else 0.0,
            "pending_version": (
                scheduler.pending.version
                if scheduler is not None and scheduler.pending is not None
                else None
            ),
            "control_params_reverted": revert_boundary is not None,
            "control_library_version": initial_version
            if revert_boundary is not None
            else int(published.library_version),
            "control_params_sha256": _hash_tree(selected_params),
            "method": config.method,
        }

    def capture(
        label: str,
        when: float,
        state: np.ndarray,
        *,
        stage: str,
        saved_context: tuple[Any, dict[str, Any]] | None = None,
    ) -> None:
        published, context = capture_context() if saved_context is None else saved_context
        metadata = {
            **context,
            "label": label,
            "physical_time_seconds": float(when),
            "physical_state_sha256": _hash_tree(state),
            "published_learner_sha256": _hash_tree(published),
            "capture_stage": stage,
            "used_at_sensing_boundary": stage == "sensing_boundary_after_publication",
            "sensing_boundary_time_seconds": float(when)
            if stage == "sensing_boundary_after_publication"
            else None,
        }
        capture_records.append(metadata)
        if config.save_checkpoints:
            captures.append((label, published, state.copy(), metadata))
        callback(snapshot_callback, {**metadata, "state": state.copy(), "learner_state": published})

    def flush_deferred_captures(before: float | None = None) -> None:
        for requested in sorted(tuple(deferred_captures)):
            if before is not None and requested >= before - 1e-9:
                continue
            saved, physical, context = deferred_captures.pop(requested)
            capture(
                f"critical-{requested:.9f}",
                requested,
                physical,
                stage="physical_grid_without_sensing_boundary",
                saved_context=(saved, context),
            )
            captured_times.add(requested)

    def append_dense(
        when: float,
        state: np.ndarray,
        native: np.ndarray,
        forces: np.ndarray,
        wrench: np.ndarray,
        residual: float,
        *,
        overwrite: bool = False,
    ) -> None:
        command_lower, command_upper, effectiveness, time_constants = recorded_models.at(when)
        row = {
            "time": float(when),
            "state": np.asarray(state).copy(),
            "native_state": np.asarray(native).copy(),
            "command": current_command.copy(),
            "actual_forces": np.asarray(forces).copy(),
            "applied_wrenches": np.asarray(wrench).copy(),
            "conversion_residual": float(residual),
            "actual_effectiveness": effectiveness,
            "actual_time_constants": time_constants,
            "operational_margins": _operational_margins(
                world, np.asarray(state), config.filter_config.arena_clearance
            ),
            "motor_state_margins": np.concatenate(
                (np.asarray(state)[13:], command_upper - np.asarray(state)[13:])
            ),
            "command_margins": np.concatenate(
                (current_command - command_lower, command_upper - current_command)
            ),
            "quaternion_norm": float(np.linalg.norm(np.asarray(state)[3:7])),
        }
        if overwrite:
            dense[-1] = row
        else:
            dense.append(row)

    def advance_until(target: float, phase: str) -> bool:
        """Stop at the first audited physical substep; never replay later commands."""
        nonlocal simulator_seconds, final_state, collision, termination
        started = time.perf_counter()
        try:
            target = min(target, duration)
            while plant.time < target - 1e-10 and collision is None:
                remaining_captures = [
                    t for t in config.capture_times if plant.time + 1e-10 < t <= target + 1e-10
                ]
                end = min(
                    target,
                    target
                    if hasattr(plant, "advance_audited")
                    else plant.time + config.plant_step_seconds,
                    remaining_captures[0] if remaining_captures else target,
                )
                # Continuing physical service beyond a deferred grid tick proves that
                # no control was sensed there; retain its original published snapshot.
                flush_deferred_captures(before=end)
                audited = None
                if hasattr(plant, "advance_audited"):

                    def audit_nodes(times: np.ndarray, states: np.ndarray) -> bool:
                        nonlocal audited
                        audited = _audit_interval(world, times, states)
                        return bool(audited["terminate"])

                    def separation_precheck(times: np.ndarray, states: np.ndarray) -> bool:
                        nonlocal audited
                        audited = _audit_interval(world, times, states)
                        return _strictly_separated(audited)

                    trace = plant.advance_audited(
                        current_command,
                        end - plant.time,
                        audit_nodes,
                        separation_precheck=separation_precheck,
                    )
                else:
                    trace = plant.advance(current_command, end - plant.time)
                    audited = _audit_interval(world, trace.times, trace.states)
                for i, when in enumerate(trace.times):
                    append_dense(
                        float(when),
                        trace.states[i],
                        trace.native_states[i],
                        trace.actual_forces[i],
                        trace.wrenches[i],
                        float(np.max(np.asarray(trace.conversion_residual))),
                        overwrite=i == 0,
                    )
                final_state = np.asarray(trace.states[-1]).copy()
                if audited is not None and audited["terminate"]:
                    collision = {**audited, "detected_at": float(plant.time), "phase": phase}
                    termination = "physical_collision"
                    event("collision", **collision, state=final_state)
                    break
                for when in config.capture_times:
                    if abs(when - plant.time) <= 1e-9 and when not in captured_times:
                        if math.isclose(when / period, round(when / period), abs_tol=1e-8):
                            # Publication belongs to the next actual sensing boundary. A
                            # delayed service may instead cross this grid tick without sensing.
                            saved, context = capture_context()
                            deferred_captures[when] = (saved, final_state.copy(), context)
                        else:
                            captured_times.add(when)
                            capture(
                                f"critical-{when:.9f}", when, final_state, stage="physical_off_grid"
                            )
            return collision is None and plant.time >= target - 1e-9
        finally:
            simulator_seconds += time.perf_counter() - started

    def prediction_inputs(when: float) -> tuple[Any, Any]:
        if input_cache is not None:
            return input_cache.prediction_inputs(when)
        prediction = world.obstacle_prediction(
            when, dt=config.filter_config.dt, horizon=config.filter_config.horizon
        )
        safety = world.safety_limits(when)
        bias = jnp.asarray(config.observation_config.obstacle_position_bias_m)
        prediction = prediction._replace(centers=prediction.centers + bias)
        safety = safety._replace(obstacle_centers=safety.obstacle_centers + bias)
        return prediction, safety

    def control_call(
        observed: Any, params: Any, model: Any, prediction: Any, safety: Any, goal: Any, when: float
    ) -> tuple[Any, Any]:
        nonlocal nominal_command_for_control
        if config.method != "OPT":
            step = _synchronize(
                controllerfunctions.controller(
                    observed, params, model, prediction, safety, jnp.asarray(previous_index), goal
                )
            )
            return step, step.certificates.rollouts.states[0, :, :3]
        mission, emergency = _synchronize(
            (
                controllerfunctions.nominal(observed, goal, model),
                controllerfunctions.emergency(observed, model),
            )
        )
        nominal_command_for_control = np.asarray(mission.commands[0, 0])
        step = opt_controller.solve(
            observed,
            model,
            prediction,
            safety,
            mission.commands[0],
            emergency,
            previous_plan=previous_plan,
            previous_command=current_command,
            observation_time=when,
        )
        return step, mission.states[0, :, :3]

    try:
        binding = {
            "config": config,
            "scene": scene.metadata(),
            "checkpoint": provenance,
            "initial_state": initial,
            "initial_state_sha256": _hash_tree(initial),
            "initial_learner_sha256": _hash_tree(bundle.state),
            "initial_params_sha256": _hash_tree(initial_params),
            "initial_optimizer_sha256": _hash_tree(bundle.state.optimizer_state),
            "initial_library_version": initial_version,
            "physical_initialization": "scene body plus nominal hover motor effort",
            "controller_model_observation": (
                "oracle current active parameters"
                if config.observation_config.parameter_delay_seconds == 0
                and config.observation_config.effectiveness_bias == 0
                and config.observation_config.lag_scale == 1
                else "controlled parameter delay or bias"
            ),
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in Path(__file__).parent.glob("actuator_*.py")
            },
        }
        artifacts["binding"] = directory / "binding.json"
        _write_json(artifacts["binding"], binding)
        plant = make_actuator_plant(
            scene,
            nominal,
            level=config.plant_level,
            max_step=config.plant_step_seconds,
            initial_state=initial,
        )
        initial_forces = np.asarray(plant.model_snapshot().effectiveness) * initial[13:]
        append_dense(
            0.0,
            initial,
            plant.native_state,
            initial_forces,
            np.asarray(nominal.force_to_wrench) @ initial_forces,
            0.0,
        )
        capture("initial", 0.0, initial, stage="initial_before_control_service")
        applications.append(
            {
                "time": 0.0,
                "command": current_command.copy(),
                "state": initial.copy(),
                "control_index": -1,
            }
        )
        event("initial_command", **applications[-1])
        collision = _initial_collision(world, initial)
        if collision is not None:
            termination = "physical_collision"
            event("collision", **collision, state=initial)
        else:
            actor_config = (
                replace(bundle.config, adapter_mode=config.method)
                if config.method in {"F0", "F1"}
                else bundle.config
            )
            if controllerfunctions is None:
                controllerfunctions = build_actuator_controller(
                    bundle.contract.spec, actor_config, config.filter_config
                )
            if config.method == "OPT" and opt_controller is None:
                from crazyflow.safety.da_plcbf.actuator_opt import (
                    ActuatorOPTConfig,
                    build_actuator_opt,
                )

                opt_controller = build_actuator_opt(
                    config.filter_config, actor_config, ActuatorOPTConfig(wall_time_budget=period)
                )
            learner = (
                learner_functions
                if learner_functions is not None
                else build_actuator_skill_learner(bundle.contract, bundle.config)
                if learns
                else None
            )
            warm_observed = jnp.asarray(
                observe_actuator_state(initial, 0.0, scene.scene_seed, config.observation_config)
            )
            warm_model = (
                input_cache.model_at(0.0)
                if input_cache is not None
                else observed_actuator_model(scene, 0.0, nominal, config.observation_config)
            )
            warm_prediction, warm_safety = prediction_inputs(0.0)
            warm_goal = jnp.asarray(
                world.initial_state[:3]
                if scene.navigation_start > 0
                else world.waypoint_positions[0]
            )
            if config.method == "OPT" and config.warmup_calls:
                mission, emergency = _synchronize(
                    (
                        controllerfunctions.nominal(warm_observed, warm_goal, warm_model),
                        controllerfunctions.emergency(warm_observed, warm_model),
                    )
                )
                warmup["opt_compile_seconds"] = opt_controller.warmup(
                    warm_observed,
                    warm_model,
                    warm_prediction,
                    warm_safety,
                    mission.commands[0],
                    emergency,
                )
            disposable = bundle.state
            for _ in range(config.warmup_calls):
                started = time.perf_counter()
                step, nominal_positions = control_call(
                    warm_observed,
                    initial_params,
                    warm_model,
                    warm_prediction,
                    warm_safety,
                    warm_goal,
                    0.0,
                )
                if config.method != "OPT":
                    _filter_record(step, config.retain_rollouts)
                np.asarray(nominal_positions)
                warmup["controller_seconds"].append(time.perf_counter() - started)
                if learns:
                    started = time.perf_counter()
                    disposable, _metrics = _synchronize(
                        learner.step(disposable, warm_observed, warm_model)
                    )
                    warmup["learner_seconds"].append(time.perf_counter() - started)
            if config.warmup_calls and hasattr(plant, "warmup"):
                # Cover all short latency/event fragments without querying future model values.
                steady_service = max(warmup["controller_seconds"][-2:], default=0.0)
                steps = math.ceil(max(period, steady_service) / config.plant_step_seconds) + 1
                warmup["plant_compile_seconds"] = plant.warmup(
                    current_command, durations=(period,), substep_counts=tuple(range(1, steps + 1))
                )
            # Every compiled warm update is disposable. Preserve params, Adam and both counters.
            learner_state = bundle.state
            durations = warmup["learner_seconds"]
            estimate = max(float(np.percentile(durations[-2:], 95)), 1e-9) if durations else 1e-6
            scheduler = BoundarySnapshotScheduler(
                bundle.state,
                initial_version,
                estimate,
                reserve_seconds=config.controller_reserve_seconds,
                safety_factor=config.update_safety_factor,
            )
            event("warmup", **warmup, initial_version_preserved=int(learner_state.library_version))
            epoch = time.perf_counter()
            tick = 0
            while plant.time < duration - 1e-10 and collision is None:
                when = float(plant.time)
                if config.execution_mode == "paced":
                    remaining = epoch + tick * period - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                boundary_wall = time.perf_counter()
                published = scheduler.publish(boundary_wall, when)
                if (
                    config.freeze_learning_at is not None
                    and when >= config.freeze_learning_at - 1e-10
                ):
                    if freeze_boundary is None:
                        freeze_boundary = when
                        event(
                            "freeze_learning",
                            requested_time=config.freeze_learning_at,
                            actual_boundary=when,
                        )
                if (
                    config.revert_control_params_at is not None
                    and when >= config.revert_control_params_at - 1e-10
                ):
                    if revert_boundary is None:
                        revert_boundary = when
                        event(
                            "revert_control_params",
                            requested_time=config.revert_control_params_at,
                            actual_boundary=when,
                        )
                flush_deferred_captures(before=when)
                for requested in config.capture_times:
                    if abs(requested - when) <= 1e-9 and requested not in captured_times:
                        deferred_captures.pop(requested, None)
                        capture(
                            f"critical-{requested:.9f}",
                            when,
                            np.asarray(plant.observe()).copy(),
                            stage="sensing_boundary_after_publication",
                        )
                        captured_times.add(requested)
                service_started = time.perf_counter()
                actual = np.asarray(plant.observe()).copy()
                # The entire preceding physical interval was audited before any task credit.
                credit_waypoint(when, actual)
                goal = np.asarray(
                    world.initial_state[:3]
                    if when < scene.navigation_start - 1e-10
                    else world.waypoint_positions[
                        min(len(arrivals), len(world.waypoint_positions) - 1)
                    ]
                )
                control_params = (
                    initial_params if revert_boundary is not None else published.state.params
                )
                control_version = (
                    initial_version if revert_boundary is not None else published.version
                )
                observed = observe_actuator_state(
                    actual, when, scene.scene_seed, config.observation_config
                )
                model = (
                    input_cache.model_at(when)
                    if input_cache is not None
                    else observed_actuator_model(scene, when, nominal, config.observation_config)
                )
                prediction, safety = prediction_inputs(when)
                observation_done = time.perf_counter()
                device_observed, device_goal = _synchronize(
                    (jnp.asarray(observed), jnp.asarray(goal))
                )
                transfer_done = time.perf_counter()
                step, nominal_positions = control_call(
                    device_observed, control_params, model, prediction, safety, device_goal, when
                )
                solve_done = time.perf_counter()
                command = np.asarray(step.action).copy()
                if config.method == "OPT":
                    record = {
                        "mode": step.mode,
                        "input_valid": bool(step.input_valid),
                        "opt_feasible": bool(step.feasible),
                        "opt_deadline_met": bool(step.deadline_met),
                        "opt_internal_service_seconds": step.service_seconds,
                        "opt_available_at": step.available_at,
                        "opt_available_at_deadline": step.available_action is not None,
                        "opt_evaluations": step.evaluations,
                        "opt_compilation_in_solve": step.compilation_in_solve,
                        "opt_budget_exhausted": step.budget_exhausted,
                        "opt_warm_start_revalidated": step.warm_start_revalidated,
                        "opt_warm_start_feasible": step.warm_start_feasible,
                        "nominal_command": nominal_command_for_control,
                        **_held_record(step.held_check, "applied"),
                    }
                else:
                    record = _filter_record(step, config.retain_rollouts)
                encounter = nominal_encounter_metrics(
                    np.asarray(nominal_positions),
                    prediction,
                    dt=config.filter_config.dt,
                    ego_radius=config.filter_config.ego_radius,
                    obstacle_clearance=config.filter_config.obstacle_clearance,
                )
                record.update(
                    control_index=len(controls),
                    time=when,
                    sensing_tick=tick,
                    actual_state=actual,
                    observed_state=np.asarray(observed).copy(),
                    observed_state_sha256=_hash_tree(observed),
                    controller_input_state=np.asarray(device_observed).copy(),
                    controller_input_state_sha256=_hash_tree(device_observed),
                    actual_state_sha256=_hash_tree(actual),
                    goal=goal.copy(),
                    previous_command=current_command.copy(),
                    planned_command=command.copy(),
                    planned_at=when,
                    command_applied=False,
                    command_applied_at=np.nan,
                    library_version=published.version,
                    control_library_version=control_version,
                    control_params_sha256=_hash_tree(control_params),
                    control_params_reverted=revert_boundary is not None,
                    cumulative_gradient_steps=int(published.state.cumulative_gradient_steps),
                    snapshot_training_time=published.training_simulation_time,
                    snapshot_age_seconds=when - published.training_simulation_time,
                    boundary_lateness_seconds=max(0.0, boundary_wall - epoch - tick * period)
                    if config.execution_mode == "paced"
                    else 0.0,
                    **_model_record(plant.model_snapshot(), "actual"),
                    **_model_record(model, "estimated"),
                    **{f"nominal_{key}": value for key, value in encounter.items()},
                )
                controls.append(record)
                event("control_ready", **record)
                ready_wall = time.perf_counter()
                controller_seconds = ready_wall - service_started
                record.update(
                    observation_seconds=observation_done - service_started,
                    device_transfer_seconds=transfer_done - observation_done,
                    controller_compute_seconds=solve_done - transfer_done,
                    mandatory_host_seconds=ready_wall - solve_done,
                    controller_seconds=controller_seconds,
                    controller_available_at=when + controller_seconds,
                    controller_deadline_met=controller_seconds <= period,
                    learner_started=False,
                    learner_seconds=0.0,
                    learner_finite_update_applied=False,
                    learner_completion_credited=False,
                    learner_deadline_met=True,
                    learner_estimated_seconds=scheduler.estimated_service_seconds,
                    learner_budget_seconds=0.0,
                    simulator_and_audit_seconds=0.0,
                    resolved_physical_time=when,
                    skipped_sensing_ticks=0,
                )
                simulator_before = simulator_seconds
                if config.execution_mode == "delayed":
                    advance_until(when + controller_seconds, "controller_service_old_command")
                if collision is not None or plant.time >= duration - 1e-10:
                    event(
                        "control_discarded",
                        time=when,
                        reason=termination or "episode_ended_before_availability",
                    )
                    record["simulator_and_audit_seconds"] = simulator_seconds - simulator_before
                    event("control_resolved", **record)
                    break
                if not record["input_valid"]:
                    termination = "invalid_observation"
                    event("control_discarded", time=when, reason=termination)
                    event("control_resolved", **record)
                    break
                if (
                    command.shape != (4,)
                    or not np.all(np.isfinite(command))
                    or np.any(command < np.asarray(model.command_lower) - 1e-7)
                    or np.any(command > np.asarray(model.command_upper) + 1e-7)
                ):
                    termination = "invalid_command"
                    event("control_discarded", time=when, reason=termination)
                    event("control_resolved", **record)
                    break
                current_command = command
                record["command_applied"] = True
                record["command_applied_at"] = float(plant.time)
                application = {
                    "time": float(plant.time),
                    "command": current_command.copy(),
                    "state": np.asarray(plant.observe()).copy(),
                    "control_index": record["control_index"],
                }
                applications.append(application)
                event("command_application", **application)
                if config.method == "OPT":
                    previous_plan = step.plan
                else:
                    previous_index = int(step.selected_index)
                next_tick, next_boundary = _next_grid(float(plant.time), period, tick + 1)
                if config.execution_mode != "delayed":
                    advance_until(next_boundary, "command_hold")
                eligible_update = (
                    learns
                    and collision is None
                    and plant.time < duration - 1e-10
                    and when >= config.learning_start_seconds - 1e-10
                    and record["control_index"] % config.update_every_controls == 0
                    and freeze_boundary is None
                    and not (
                        config.freeze_learning_at is not None
                        and when >= config.freeze_learning_at - 1e-10
                    )
                )
                learner_now = time.perf_counter()
                learner_deadline = (
                    learner_now + max(0.0, next_boundary - plant.time)
                    if config.execution_mode == "delayed"
                    else epoch + next_tick * period
                )
                can_start = eligible_update and (
                    config.execution_mode == "deterministic"
                    or scheduler.can_start(learner_now, learner_deadline)
                )
                record["learner_estimated_seconds"] = scheduler.estimated_service_seconds
                record["learner_budget_seconds"] = max(0.0, learner_deadline - learner_now)
                if can_start:
                    record["learner_started"] = True
                    update_started = time.perf_counter()
                    following, metrics = _synchronize(
                        learner.step(learner_state, device_observed, model)
                    )
                    finite = bool(metrics.finite_update_applied)
                    update_record = {
                        "training_simulation_time": when,
                        "sensed_state_sha256": record["controller_input_state_sha256"],
                        "estimated_model_sha256": record["estimated_model_sha256"],
                        "started_wall_time": update_started,
                        "finite_update_applied": finite,
                        "computed_version": int(following.library_version),
                        "gradient_norm": float(metrics.gradient_norm),
                        "parameter_update_norm": float(metrics.parameter_update_norm),
                        "loss": _jsonable(metrics.loss),
                        "completion_credited": False,
                    }
                    event("learner_computed", **update_record)
                    completed_wall = time.perf_counter()
                    update_seconds = max(completed_wall - update_started, 1e-9)
                    update_record.update(
                        completed_wall_time=completed_wall, service_seconds=update_seconds
                    )
                    record["learner_seconds"] = update_seconds
                    record["learner_finite_update_applied"] = finite
                    physically_completed = True
                    if config.execution_mode == "delayed":
                        expected_finish = plant.time + update_seconds
                        physically_completed = (
                            advance_until(expected_finish, "learner_service_new_command")
                            and expected_finish <= duration + 1e-10
                        )
                        next_tick, next_boundary = _next_grid(float(plant.time), period, tick + 1)
                    if physically_completed and collision is None:
                        update_record["completion_credited"] = True
                        record["learner_completion_credited"] = True
                        learner_state = following
                        if finite:
                            scheduler.complete(
                                CompletedSnapshot(
                                    following,
                                    int(following.library_version),
                                    when,
                                    update_started,
                                    completed_wall,
                                    float(metrics.gradient_norm),
                                    float(metrics.parameter_update_norm),
                                )
                            )
                        else:
                            scheduler.durations.append(update_seconds)
                    else:
                        update_record["discard_reason"] = "episode_ended_before_learner_completion"
                    updates.append(update_record)
                    record["learner_deadline_met"] = (
                        completed_wall <= learner_deadline
                        if config.execution_mode == "paced"
                        else update_seconds <= record["learner_budget_seconds"]
                        if config.execution_mode == "delayed"
                        else True
                    )
                    event("learner_resolved", **update_record)
                if config.execution_mode == "delayed" and collision is None:
                    advance_until(next_boundary, "command_hold")
                record["skipped_sensing_ticks"] = max(0, next_tick - tick - 1)
                record["simulator_and_audit_seconds"] = simulator_seconds - simulator_before
                record["resolved_physical_time"] = float(plant.time)
                event("control_resolved", **record)
                callback(
                    progress_callback,
                    {
                        "time": float(plant.time),
                        "duration": duration,
                        "control_index": record["control_index"],
                        "state": np.asarray(plant.observe()).copy(),
                        "published_version": scheduler.published.version,
                        "termination": termination,
                    },
                )
                tick = next_tick
            if termination is None:
                # The duration endpoint is an accounting boundary, without another
                # controller call, learner update, or snapshot publication.
                credit_waypoint(float(plant.time), np.asarray(plant.observe()))
                termination = "duration_complete"
    except BaseException as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        termination = "incomplete_error"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            interrupt = exc
        event("error", **error)
    finally:
        events_stream.close()

    execution_wall_seconds = time.perf_counter() - epoch
    if plant is not None:
        final_state = np.asarray(plant.observe()).copy()
    physical_time = float(plant.time) if plant is not None else 0.0
    control_arrays = _rows_to_arrays(controls)
    dense_arrays = _rows_to_arrays(dense)
    application_arrays = _rows_to_arrays(applications)
    full_audit = None
    collision_observation = {
        "modeled_collider_collision": True if collision is not None else None,
        "modeled_collision_observation": "initial_exact_geometric_intersection"
        if collision is not None
        else "insufficient_trajectory",
        "measured_mujoco_contact_event": None,
    }
    if len(dense) >= 2:
        try:
            full_audit = _audit_interval(world, dense_arrays["time"], dense_arrays["state"])
            collision_observation = summarize_collision_observation(
                full_audit["audit"],
                termination_geometry="modeled_collider",
                termination=termination,
            )
        except Exception as exc:
            error = error or {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "stage": "final_collision_audit",
            }
            termination = "incomplete_error"
    try:
        flush_deferred_captures()
        capture("final", physical_time, final_state, stage="episode_end")
        if config.save_checkpoints and scheduler is not None and scheduler.pending is not None:
            captures.append(
                (
                    "final-completed-unpublished",
                    learner_state,
                    final_state.copy(),
                    {
                        "physical_time_seconds": physical_time,
                        "published": False,
                        "published_version": scheduler.published.version,
                    },
                )
            )
        for label, state, physical, metadata in captures:
            stem = directory / "snapshots" / label
            stem.parent.mkdir(exist_ok=True)
            paths = save_actuator_learner_checkpoint(
                state,
                bundle.contract,
                np.asarray(physical),
                stem,
                config=bundle.config,
                metadata={**provenance, **metadata},
            )
            artifacts[f"snapshot_{label}_npz"] = paths[0]
            artifacts[f"snapshot_{label}_json"] = paths[1]
    except Exception as exc:
        error = error or {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "stage": "final_snapshot",
        }
        termination = "incomplete_error"

    for label, values in (
        ("controls", control_arrays),
        ("dense", dense_arrays),
        ("applications", application_arrays),
    ):
        path = directory / f"{label}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **values)
        artifacts[label] = path
    artifacts["updates"] = directory / "updates.json"
    _write_json(artifacts["updates"], updates)
    artifacts["warmup"] = directory / "warmup.json"
    _write_json(artifacts["warmup"], warmup)
    if full_audit is not None:
        artifacts["collision_audit"] = directory / "collision_audit.json"
        _write_json(artifacts["collision_audit"], full_audit)
    if error is not None:
        artifacts["error"] = directory / "error.json"
        _write_json(artifacts["error"], error)

    def statistics(key: str) -> dict[str, float | int | None]:
        values = np.asarray([row[key] for row in controls if key in row], dtype=float)
        values = values[np.isfinite(values)]
        return {
            "count": len(values),
            "mean": float(np.mean(values)) if len(values) else None,
            "p95": float(np.percentile(values, 95)) if len(values) else None,
            "maximum": float(np.max(values)) if len(values) else None,
        }

    operational = dense_arrays.get("operational_margins", np.empty((0, 9)))
    motor_margins = dense_arrays.get("motor_state_margins", np.empty((0, 8)))
    command_margins = dense_arrays.get("command_margins", np.empty((0, 8)))
    valid_outcome = termination in {"duration_complete", "physical_collision"} and error is None
    reached_duration = physical_time >= duration - 1e-9
    summary = {
        "status": "completed" if valid_outcome else "incomplete",
        "termination": termination,
        "method": config.method,
        "execution_mode": config.execution_mode,
        "plant_level": config.plant_level,
        "physical_world_id": scene.metadata().get("physical_world_id"),
        "physical_time_seconds": physical_time,
        "duration_seconds": duration,
        "reached_full_duration": reached_duration,
        "full_episode_completed": valid_outcome and reached_duration,
        "collision": collision,
        **collision_observation,
        "collision_observation_scope": (
            "recorded physical prefix; no assertion about an unexecuted suffix"
        ),
        "collision_detection_boundary_seconds": collision.get("detected_at", physical_time)
        if collision
        else None,
        "waypoints_completed": len(arrivals),
        "waypoints_total": len(world.waypoint_positions),
        "waypoint_arrival_times_seconds": arrivals,
        "task_completed": len(arrivals) == len(world.waypoint_positions),
        "task_completion_time_seconds": arrivals[-1]
        if len(arrivals) == len(world.waypoint_positions)
        else None,
        "held_final_goal_after_completion": True,
        "successful_full_episode": valid_outcome
        and reached_duration
        and len(arrivals) == len(world.waypoint_positions)
        and collision_observation["modeled_collider_collision"] is False
        and bool(len(operational))
        and bool(np.all(operational >= -1e-7)),
        "control_count": len(controls),
        "applied_control_count": sum(bool(row.get("command_applied", False)) for row in controls),
        "actual_command_application_count": len(applications),
        "actual_node_count": len(dense),
        "initial_library_version": initial_version,
        "final_completed_library_version": int(learner_state.library_version),
        "final_published_library_version": scheduler.published.version
        if scheduler
        else initial_version,
        "final_pending_library_version": scheduler.pending.version
        if scheduler is not None and scheduler.pending is not None
        else None,
        "initial_cumulative_gradient_steps": int(bundle.state.cumulative_gradient_steps),
        "final_cumulative_gradient_steps": int(learner_state.cumulative_gradient_steps),
        "learner_calls": len(updates),
        "finite_credited_updates": sum(
            bool(row["finite_update_applied"] and row["completion_credited"]) for row in updates
        ),
        "finite_uncredited_updates": sum(
            bool(row["finite_update_applied"] and not row["completion_credited"]) for row in updates
        ),
        "snapshot_publications": scheduler.publications if scheduler else [],
        "snapshot_captures": capture_records,
        "freeze_learning_actual_boundary": freeze_boundary,
        "revert_control_params_actual_boundary": revert_boundary,
        "initial_state_sha256": _hash_tree(initial),
        "final_state_sha256": _hash_tree(final_state),
        "initial_learner_sha256": _hash_tree(bundle.state),
        "final_learner_sha256": _hash_tree(learner_state),
        "initial_optimizer_sha256": _hash_tree(bundle.state.optimizer_state),
        "final_optimizer_sha256": _hash_tree(learner_state.optimizer_state),
        "state_dimension": 17,
        "native_motor_state_units": "rpm" if config.plant_level == "P2" else "nominal_effort_N",
        "execution_wall_seconds": execution_wall_seconds,
        "simulator_and_audit_seconds": simulator_seconds,
        "diagnostic_callback_seconds": callback_seconds,
        "controller_seconds": statistics("controller_seconds"),
        "observation_seconds": statistics("observation_seconds"),
        "cache_observation_inputs_enabled": config.cache_observation_inputs,
        "observation_model_materializations": (
            input_cache.model_materializations if input_cache is not None else []
        ),
        "device_transfer_seconds": statistics("device_transfer_seconds"),
        "controller_compute_seconds": statistics("controller_compute_seconds"),
        "mandatory_host_seconds": statistics("mandatory_host_seconds"),
        "learner_seconds": statistics("learner_seconds"),
        "controller_deadline_misses": sum(
            not row.get("controller_deadline_met", False) for row in controls
        ),
        "learner_deadline_misses": sum(
            row.get("learner_started", False) and not row.get("learner_deadline_met", True)
            for row in controls
        ),
        "skipped_sensing_ticks": sum(row.get("skipped_sensing_ticks", 0) for row in controls),
        "timing_contract": {
            "deterministic": (
                "forced finite updates; publication at the following boundary; no real-time claim"
            ),
            "paced": (
                "measured serialized wall-clock scheduling with nondelayed fixed physical holds"
            ),
            "delayed": (
                "old command during measured complete controller service; new command during "
                "learner service; missed sensing ticks skipped; simulator and audit cost "
                "excluded from physical latency mapping"
            ),
        }[config.execution_mode],
        "qp_feasibility_scope": (
            "only the selected policy QP is solved; eligible-policy count is not "
            "full-library QP feasibility"
        ),
        "actual_operational_minimum_by_constraint": np.min(operational, axis=0)
        if len(operational)
        else None,
        "actual_operational_violating_nodes_by_constraint": np.sum(operational < -1e-7, axis=0)
        if len(operational)
        else None,
        "actual_operational_all_nodes_pass": bool(np.all(operational >= -1e-7))
        if len(operational)
        else None,
        "minimum_actual_motor_state_margin_N": float(np.min(motor_margins))
        if len(motor_margins)
        else None,
        "minimum_applied_command_margin_N": float(np.min(command_margins))
        if len(command_margins)
        else None,
        "error": error,
        "artifacts": artifacts,
    }
    artifacts["summary"] = directory / "summary.json"
    _write_json(artifacts["summary"], summary)
    result = ActuatorEpisodeResult(
        summary, control_arrays, dense_arrays, final_state, learner_state, artifacts
    )
    if interrupt is not None:
        raise interrupt
    return result
