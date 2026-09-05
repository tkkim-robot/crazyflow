"""Matched navigation with exogenous worlds and obstacle-free persistent learning.

The actor/learner receives proprioception and the same declared point model as the frozen
controller. Waypoints and predicted obstacle paths remain inside task control and safety.
Deterministic runs record an exogenous update schedule and actual synchronized service costs;
they are mechanism experiments, never claims of real-time availability.
"""

from __future__ import annotations

import hashlib
import json
import time
from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.case_study_world import audit_recorded_collider_clearance
from crazyflow.safety.da_plcbf.continuous_version_a import (
    ContinuousVersionAConfig,
    PolicyRollouts,
    continuous_version_a_step,
    rollout_waypoint_library,
)
from crazyflow.safety.da_plcbf.deadline_schedule import BoundarySnapshotScheduler, CompletedSnapshot
from crazyflow.safety.da_plcbf.deterministic_schedule import DeterministicUpdateSchedule
from crazyflow.safety.da_plcbf.learner_checkpoint import (
    load_learner_checkpoint,
    save_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.mujoco_comparison_video import ComparisonVideoTrace, ObstacleTrack
from crazyflow.safety.da_plcbf.navigation_audit import audit_navigation_execution
from crazyflow.safety.da_plcbf.navigation_world import (
    CF21B_BODY_ORIGIN_ENCLOSURE_M,
    NavigationWorld,
    WaypointProgress,
    advance_waypoints,
    nominal_encounter_metrics,
)
from crazyflow.safety.da_plcbf.online_constant_wind import (
    OnlineConstantWindResult,
    _append_method_record,
    _empty_method_records,
    _method_trace,
    _timing_statistics,
    save_online_constant_wind_result,
)
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    build_persistent_skill_learner,
    rollout_skill_library,
)
from crazyflow.safety.da_plcbf.point_wind_estimator import (
    PointWindEstimatorConfig,
    initialize_point_wind_estimator,
    model_with_point_wind,
    update_point_wind_estimator,
)
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.selector import SelectionConfig
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig
from crazyflow.safety.da_plcbf.version_a_filter import VersionAFilterConfig


@dataclass(frozen=True, slots=True)
class NavigationExperimentConfig:
    execution_mode: str = "deterministic"
    model_information: str = "oracle"
    estimator_response_rate: float = 2.4
    learning_start_seconds: float = 4.0
    update_every_controls: int = 2
    enable_learning: bool = True
    learner_kind: str = "reference"
    nominal_acceleration_limit: float = 1.2
    probe_every_controls: int = 25
    save_periodic_snapshots_controls: int = 100
    controller_reserve_seconds: float = 0.003
    update_safety_factor: float = 1.25
    navigation_start_seconds: float = 0.0
    fallback_mapping: str = "compensated"
    allow_legacy_point_enclosure: bool = False
    termination_geometry: str = "body_origin_enclosure"

    def validate(self, world: NavigationWorld) -> None:
        if self.execution_mode not in {"deterministic", "budgeted"}:
            raise ValueError("execution mode must be deterministic or budgeted")
        if (
            not np.isfinite(self.controller_reserve_seconds)
            or self.controller_reserve_seconds < 0
            or not np.isfinite(self.update_safety_factor)
            or self.update_safety_factor < 1
        ):
            raise ValueError("reserve must be nonnegative finite and safety factor at least one")
        if self.model_information not in {"oracle", "estimated"}:
            raise ValueError("model information must be oracle or estimated wind")
        if self.learner_kind not in {"reference", "original"}:
            raise ValueError("unknown learner kind")
        if self.fallback_mapping not in {"compensated", "matched_uncompensated"}:
            raise ValueError("unknown matched fallback mapping")
        if type(self.allow_legacy_point_enclosure) is not bool:
            raise ValueError("allow_legacy_point_enclosure must be boolean")
        if self.termination_geometry not in {"body_origin_enclosure", "modeled_collider"}:
            raise ValueError("unknown simulation termination geometry")
        if self.termination_geometry == "modeled_collider" and world.config.payload_events:
            raise ValueError("modeled-collider termination currently audits the unladen cf21B")
        if (
            world.config.ego_radius < CF21B_BODY_ORIGIN_ENCLOSURE_M - 1e-9
            and not self.allow_legacy_point_enclosure
        ):
            raise ValueError(
                "new navigation runs must enclose the cf21B asset collider (radius .106 m); "
                "legacy point-envelope reproduction requires an explicit opt-in"
            )
        if (
            not np.isfinite(self.navigation_start_seconds)
            or not 0 <= self.navigation_start_seconds < world.config.duration_seconds
            or not np.isclose(
                self.navigation_start_seconds / world.config.control_period,
                round(self.navigation_start_seconds / world.config.control_period),
            )
        ):
            raise ValueError("navigation start must be a control boundary within the episode")
        for name in (
            "update_every_controls",
            "probe_every_controls",
            "save_periodic_snapshots_controls",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not np.isfinite(self.learning_start_seconds)
            or self.learning_start_seconds < 0
            or not np.isclose(
                self.learning_start_seconds / world.config.control_period,
                round(self.learning_start_seconds / world.config.control_period),
            )
        ):
            raise ValueError("learning start must be a nonnegative control boundary")
        if not all(
            np.isfinite(value) and value > 0
            for value in (self.estimator_response_rate, self.nominal_acceleration_limit)
        ):
            raise ValueError("estimator rate and nominal authority must be positive finite")
        if type(self.enable_learning) is not bool:
            raise ValueError("enable_learning must be boolean")
        # The current video model represents one exact centered box and a constant enclosure.
        if len(world.config.payload_events) > 1:
            raise ValueError(
                "the recorded navigation renderer currently supports one payload attachment"
            )
        if any(
            np.linalg.norm(event.half_extents) > world.config.ego_radius
            for event in world.config.payload_events
        ):
            raise ValueError(
                "this experiment requires payload geometry inside the original enclosure"
            )


def build_navigation_controller(
    world: NavigationWorld,
    bundle: Any,
    config: NavigationExperimentConfig,
    *,
    selection_config: SelectionConfig = SelectionConfig(switch_score_margin=0.0),
    frozen_replacement: tuple[Any, int] | None = None,
) -> Any:
    """Combine tasks with safety geometry, optionally evaluating an explicit offline ablation.

    Normal calls retain the original selector and whole learned library. ``frozen_replacement``
    is an offline diagnostic ``(original_parameters, zero_based_fallback_skill_index)``: only
    that skill's states, wrenches, and validity come from its original evaluator. Both evaluators
    use the same candidate state/model inside differentiation, so the replaced certificate has
    its matching original gradient. The augmented nominal stays at index zero and library size
    is unchanged. ``selection_config`` supports forced-incumbent offline complete-QP audits.
    """
    selection_config.validate()
    policy_config = bundle.config
    if policy_config.control_interval_steps != world.config.control_interval_steps:
        raise ValueError("checkpoint command cadence must equal the executed command hold")
    if policy_config.dt != world.config.dt:
        raise ValueError("checkpoint and world integration steps differ")
    if frozen_replacement is not None:
        if not isinstance(frozen_replacement, tuple) or len(frozen_replacement) != 2:
            raise ValueError("frozen replacement must be (parameters, fallback skill index)")
        replacement_index = frozen_replacement[1]
        if (
            isinstance(replacement_index, bool)
            or not isinstance(replacement_index, (int, np.integer))
            or not 0 <= replacement_index < bundle.spec.latent_codes.shape[0]
        ):
            raise ValueError("replacement skill index must be inside the fallback library")
    safety = world.safety_limits()
    barrier = VersionABarrierConfig(
        obstacle_clearance=world.config.obstacle_clearance,
        arena_clearance=0.08,
        ego_radius=world.config.ego_radius,
    )
    runtime = ContinuousVersionAConfig(
        dt=policy_config.dt,
        horizon=policy_config.horizon,
        obstacle_clearance=world.config.obstacle_clearance,
        ego_radius=world.config.ego_radius,
        control_interval_steps=world.config.control_interval_steps,
        prefer_nominal_when_safe=False,
    )
    nominal_config = QuadPolicyConfig(acceleration_limit=config.nominal_acceleration_limit)

    @jax.jit
    def controller(
        state: Any, params: Any, model: Any, obstacles: Any, previous: Any, goal: Any
    ) -> Any:
        def nominal(candidate: Any, point: Any) -> PolicyRollouts:
            return rollout_waypoint_library(
                candidate,
                goal[None],
                jnp.zeros((1, 3)),
                point,
                bundle.actuator,
                nominal_config,
                dt=policy_config.dt,
                horizon=policy_config.horizon,
                position_gain=2.0,
                velocity_gain=2.8,
                model_compensation=True,
                command_hold_steps=world.config.control_interval_steps,
            )

        def fallback(candidate: Any, point: Any) -> PolicyRollouts:
            result = rollout_skill_library(
                params, bundle.spec, candidate, point, bundle.actuator, policy_config
            )
            policies = PolicyRollouts(
                result.states,
                result.wrenches,
                jnp.all(result.policy_valid, axis=1)
                & jnp.all(jnp.isfinite(result.states), axis=(1, 2)),
            )
            if frozen_replacement is not None:
                original_params, skill = frozen_replacement
                original = rollout_skill_library(
                    original_params, bundle.spec, candidate, point, bundle.actuator, policy_config
                )
                original_valid = jnp.all(original.policy_valid[skill]) & jnp.all(
                    jnp.isfinite(original.states[skill])
                )
                policies = PolicyRollouts(
                    policies.states.at[skill].set(original.states[skill]),
                    policies.wrenches.at[skill].set(original.wrenches[skill]),
                    policies.valid.at[skill].set(original_valid),
                )
            return policies

        return continuous_version_a_step(
            state,
            nominal,
            fallback,
            obstacles,
            model,
            bundle.actuator,
            safety,
            barrier,
            VersionAFilterConfig(),
            runtime,
            previous_policy_index=previous,
            selection_config=selection_config,
        )

    return controller


def _task_goal(
    world: NavigationWorld, progress: WaypointProgress, when: float, start: float
) -> np.ndarray:
    """Hover holds the initial position; no navigation waypoint is active before release."""
    return world.initial_state[:3] if when < start - 1e-10 else progress.active_goal(world)


def _phase_caption(world: NavigationWorld, when: float, navigation_start: float) -> str:
    if when >= navigation_start - 1e-10:
        return "NAVIGATION · follow the waypoint queue"
    if world.config.payload_events and when >= world.config.payload_events[0].time_seconds:
        if np.linalg.norm(world.wind_at(when)) > 1e-9:
            return "HOVER · wind on and centered payload attached; center of mass unchanged"
        return "HOVER · centered payload attached; center of mass unchanged"
    if np.linalg.norm(world.wind_at(when)) > 1e-9:
        return "HOVER · wind on; watch the reusable maneuvers"
    if any(event.time_seconds <= when for event in world.config.wind_events):
        return "HOVER · wind off; observe recovery toward the original maneuvers"
    return "HOVER · no navigation goal; learn the original maneuver colors"


def _minimum_segment_clearance(states: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> float:
    if len(radii) == 0:
        return float("inf")
    relative = states[:, None, :3] - centers
    delta = np.diff(relative, axis=0)
    fraction = np.clip(
        -np.sum(relative[:-1] * delta, axis=-1) / np.maximum(np.sum(delta * delta, axis=-1), 1e-30),
        0,
        1,
    )
    return float(
        np.min(np.linalg.norm(relative[:-1] + fraction[..., None] * delta, axis=-1) - radii)
    )


def _finite_probe_value(value: Any) -> float | None:
    """Preserve nonfinite certificate values in NPZ; use explicit null in strict JSON."""
    number = float(value)
    return number if np.isfinite(number) else None


def evaluate_collision_termination(
    world: NavigationWorld,
    times: np.ndarray,
    states: np.ndarray,
    *,
    termination_geometry: str = "modeled_collider",
) -> dict[str, Any]:
    """Audit an executed hold without changing its command or the safety constraints.

    The simulation-only collider mode stops only for a strictly negative upper clearance
    bound of the rotated cf21B XML sphere against an obstacle or the floor. A bound
    straddling zero is unresolved and does not count as a collision. The caller stops at
    the next control boundary; the interpolated geometric event time remains separate.
    Legacy enclosure mode deliberately keeps its original uncorrected chord criterion.
    """
    if termination_geometry not in {"body_origin_enclosure", "modeled_collider"}:
        raise ValueError("unknown simulation termination geometry")
    if termination_geometry == "modeled_collider" and world.config.payload_events:
        raise ValueError("modeled-collider termination currently audits the unladen cf21B")
    audit = audit_recorded_collider_clearance(world, times, states)
    sphere = audit["actual_xml_sphere_geometry"]
    floor = audit["actual_xml_ground_geometry"]
    candidates = [
        (name, geometry["first_chord_intersection_time_seconds"])
        for name, geometry in (
            ("modeled_collider_obstacle", sphere),
            ("modeled_collider_floor", floor),
        )
        if geometry["minimum_clearance_upper_bound_m"] is not None
        and geometry["minimum_clearance_upper_bound_m"] < 0
    ]
    kind, event_time = min(candidates, key=lambda row: row[1]) if candidates else (None, None)
    centers, _ = world.obstacle_kinematics(np.asarray(times))
    enclosure_clearance = _minimum_segment_clearance(
        np.asarray(states), centers, world.obstacle_radii + world.config.ego_radius
    )
    enclosure_breach = enclosure_clearance <= 0
    if termination_geometry == "body_origin_enclosure":
        kind = "body_origin_enclosure" if enclosure_breach else None
        event_time = (
            audit["body_origin_envelope"]["first_chord_intersection_time_seconds"]
            if enclosure_breach
            else None
        )
    return {
        "termination_geometry": termination_geometry,
        "terminate": kind is not None,
        "collision_kind": kind,
        "first_intersection_time_seconds": event_time,
        "body_origin_enclosure_breach": bool(enclosure_breach),
        "requested_shell_breach": bool(enclosure_clearance <= world.config.obstacle_clearance),
        "audit": audit,
    }


def summarize_collision_observation(
    audit: dict[str, Any], *, termination_geometry: str, termination: str | None
) -> dict[str, Any]:
    """Keep censored or numerically unresolved true-collider outcomes distinct from safety."""
    geometries = [audit["actual_xml_sphere_geometry"], audit["actual_xml_ground_geometry"]]
    intersecting = any(
        item["minimum_clearance_upper_bound_m"] is not None
        and item["minimum_clearance_upper_bound_m"] < 0
        for item in geometries
    )
    unresolved = any(
        item["minimum_clearance_lower_bound_m"] is not None
        and item["minimum_clearance_lower_bound_m"] <= 0
        and item["minimum_clearance_upper_bound_m"] >= 0
        for item in geometries
    )
    censored = (
        termination_geometry == "body_origin_enclosure"
        and termination == "physical_collision"
        and not intersecting
    )
    return {
        "modeled_collider_collision": (
            True if intersecting else None if censored or unresolved else False
        ),
        "modeled_collision_observation": (
            "observed_geometric_intersection"
            if intersecting
            else "censored_by_enclosure_termination"
            if censored
            else "unresolved_at_interpolation_error_bound"
            if unresolved
            else "separated_on_executed_trace"
        ),
        "enclosure_termination_censors_later_collider_outcome": bool(censored),
        "interpolation_error_straddles_zero": bool(unresolved),
        "measured_mujoco_contact_event": None,
    }


def _append_terminal_record(records: dict[str, list[Any]], state: np.ndarray) -> None:
    """Retain the final executed state while carrying the last real prediction as stale data.

    A separate recorded_control_valid mask labels this row as playback/terminal accounting;
    it is never counted as a new command, learner update, timing sample, or task arrival.
    """
    if not records["position"]:
        raise ValueError("terminal padding requires at least one executed control record")
    for values in records.values():
        values.append(values[-1])
    records["position"][-1] = np.asarray(state[:3]).copy()
    records["quaternion_xyzw"][-1] = np.asarray(state[3:7]).copy()
    records["full_state"][-1] = np.asarray(state).copy()
    for name in ("controller_seconds", "learner_seconds"):
        records[name][-1] = 0.0
    records["missed_deadline"][-1] = False


def run_navigation_experiment(
    world: NavigationWorld,
    config: NavigationExperimentConfig,
    checkpoint: Path,
    directory: Path,
    *,
    device: Any = None,
    opportunity_schedule: DeterministicUpdateSchedule | None = None,
    progress_callback: Any = None,
) -> OnlineConstantWindResult:
    """Run a paired world, censor progress at collision, and retain reproducible learner inputs."""
    config.validate(world)
    directory.mkdir(parents=True, exist_ok=False)
    device = device or jax.devices()[0]
    bundle = load_learner_checkpoint(checkpoint, device=device)
    expected_compensation = config.fallback_mapping == "compensated"
    if bundle.config.model_compensation != expected_compensation:
        raise ValueError("checkpoint mapping must match the declared mapping in both panes")
    reference_metadata = {}
    if config.learner_kind == "reference":
        from crazyflow.safety.da_plcbf.state_conditioned_learning import (
            build_reference_skill_learner_from_checkpoint,
            reference_contract_checkpoint_metadata,
            save_reference_contract,
        )

        bundle, contract, learner = build_reference_skill_learner_from_checkpoint(
            checkpoint, device=device
        )
        save_reference_contract(contract, directory / "nominal_reference")
        save_reference_contract(contract, directory / "snapshots/nominal_reference")
        reference_metadata = reference_contract_checkpoint_metadata(directory / "nominal_reference")
    else:
        contract = None
        learner = build_persistent_skill_learner(
            bundle.spec, bundle.actuator, bundle.config, device=device
        )
    count = round(world.config.duration_seconds / world.config.control_period)
    schedule = opportunity_schedule or DeterministicUpdateSchedule.periodic(
        count,
        first=round(config.learning_start_seconds / world.config.control_period),
        every=config.update_every_controls,
    )
    if len(schedule.opportunities) != count:
        raise ValueError("opportunity schedule must span exactly the experiment")
    # The endpoint at duration is a recorded state, not an additional control opportunity.
    times = np.arange(count + 1) * world.config.control_period
    (directory / "world.json").write_text(json.dumps(world.metadata(), indent=2) + "\n")
    (directory / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (directory / "schedule.json").write_text(
        json.dumps(
            {
                **schedule.metadata(),
                "execution_mode": config.execution_mode,
                "mask_role": "allowed exogenous update opportunities",
                "actual_publication": (
                    "completed snapshot at actual paced boundary"
                    if config.execution_mode == "budgeted"
                    else "completed snapshot at next simulated boundary"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    base = bundle.point_model
    controller = build_navigation_controller(world, bundle, config)
    plant = jax.jit(lambda x, u, m: direct_wrench_symplectic_step(x, u, m, world.config.dt))
    estimator_config = PointWindEstimatorConfig(response_rate=config.estimator_response_rate)
    estimate = jax.jit(
        lambda e, x, y, u, known: (
            update_point_wind_estimator(
                e, x, y, u, known, dt=world.config.dt, config=estimator_config
            ).state
        )
    )
    # These exogenous piecewise-constant models are known experiment inputs. Construct each
    # event model before service measurement; only the current event's model enters a call.
    # This avoids first-use payload arithmetic compilation at the attachment boundary.
    event_times = sorted(
        {0.0}
        | {event.time_seconds for event in world.config.wind_events}
        | {event.time_seconds for event in world.config.payload_events}
    )
    event_dynamics = [world.dynamics_at(when, base) for when in event_times]
    event_models = [jax.device_put(item.model, device) for item in event_dynamics]
    jax.block_until_ready(event_models)
    methods, summaries, raw, dense, probe_snapshots = {}, {}, {}, {}, {}
    for method in ("fixed", "adaptive"):
        records = _empty_method_records()
        persistent = bundle.state
        state = jax.device_put(jnp.asarray(world.initial_state, dtype=jnp.float32), device)
        previous = jax.device_put(jnp.asarray(-1, dtype=jnp.int32), device)
        estimator = initialize_point_wind_estimator()
        progress = WaypointProgress()
        states, goals, waypoint_indices, active, boundaries = [np.asarray(state)], [], [], [], []
        local_raw = {
            key: []
            for key in (
                "hard",
                "smooth",
                "candidate_valid",
                "point_wind",
                "actual_wind",
                "actual_mass",
                "actual_inertia",
                "candidate_wrenches",
                "selected_index",
                "applied_operational_margin",
                "applied_operational_residual",
                "finite_update",
            )
        }
        encounters, snapshots, controller_seconds, learner_seconds = [], [], [], []
        checkpoint_buffers = []
        first_diagnosis_saved = False
        collision_event = None
        last_update_metrics = None
        pending = None
        last_training_time = 0.0
        # Match the committed placement of every live model leaf. A mixed committed/uncommitted
        # wind leaf creates another JIT executable at the first real control or learner call.
        warm_model = event_models[0]
        warm_goal = jax.device_put(
            jnp.asarray(
                _task_goal(world, progress, 0.0, config.navigation_start_seconds), dtype=jnp.float32
            ),
            device,
        )
        warm_prediction = world.obstacle_prediction(0, horizon=bundle.config.horizon)
        # Compile the actual controller/plant/learner dataflow before recording service samples.
        warm_state, warm_previous = state, previous
        warm_learning = persistent
        warm_estimator = initialize_point_wind_estimator()
        warm_durations = []
        for _ in range(5):
            warm_point = (
                warm_model
                if config.model_information == "oracle"
                else model_with_point_wind(warm_model, warm_estimator)
            )
            decision = jax.block_until_ready(
                controller(
                    warm_state,
                    warm_learning.params,
                    warm_point,
                    warm_prediction,
                    warm_previous,
                    warm_goal,
                )
            )
            for _ in range(world.config.control_interval_steps):
                following = plant(warm_state, decision.action, warm_model)
                if config.model_information == "estimated":
                    warm_estimator = estimate(
                        warm_estimator, warm_state, following, decision.action, warm_model
                    )
                warm_state = following
            if method == "adaptive" and config.enable_learning:
                # Match the live stopwatch: learner inputs are ready before timing begins.
                jax.block_until_ready((warm_state, warm_estimator))
                started = time.perf_counter()
                warm_learning, _ = jax.block_until_ready(
                    learner.step(warm_learning, warm_state, warm_point)
                )
                warm_durations.append(time.perf_counter() - started)
            warm_previous = decision.selected_index
        jax.block_until_ready((warm_state, warm_estimator))
        scheduler = BoundarySnapshotScheduler(
            persistent,
            int(persistent.library_version),
            max(warm_durations[-2:]) if warm_durations else 1e-3,
            config.controller_reserve_seconds,
            config.update_safety_factor,
        )
        warmup_metadata = {
            "learner_call_seconds": warm_durations,
            "initial_measured_update_seconds": scheduler.measured_update_seconds,
            "initial_reserved_update_seconds": scheduler.estimated_service_seconds,
            "event_model_preconstruction": True,
        }
        (directory / f"{method}_warmup.json").write_text(
            json.dumps(warmup_metadata, indent=2) + "\n"
        )
        epoch = time.perf_counter()
        for index, when in enumerate(times):
            scheduled_boundary = epoch + float(when)
            if config.execution_mode == "budgeted" and progress.termination is None:
                remaining = scheduled_boundary - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
            boundary_wall = time.perf_counter()
            deadline = scheduled_boundary + world.config.control_period
            prior_snapshot = persistent
            if config.execution_mode == "budgeted" and progress.termination is None:
                published = scheduler.publish(boundary_wall, float(when))
                persistent = published.state
                last_training_time = published.training_simulation_time
                if pending is not None and pending[0] is persistent:
                    last_update_metrics = pending[1]
                    pending = None
            elif pending is not None:
                persistent, last_update_metrics, last_training_time = pending
                pending = None
            if progress.termination is not None:
                # Padding serves synchronized playback only. No controls, learning, arrivals,
                # or success credit are generated after the declared terminal boundary.
                _append_terminal_record(records, np.asarray(state))
                for values in local_raw.values():
                    values.append(values[-1])
                local_raw["finite_update"][-1] = False
                goals.append(goals[-1])
                waypoint_indices.append(min(progress.completed, len(world.waypoint_positions) - 1))
                active.append(False)
                continue
            active.append(True)
            event_index = bisect_right(event_times, float(when) + 1e-10) - 1
            dynamics = event_dynamics[event_index]
            actual_model = event_models[event_index]
            model = (
                actual_model
                if config.model_information == "oracle"
                else model_with_point_wind(actual_model, estimator)
            )
            prediction = world.obstacle_prediction(float(when), horizon=bundle.config.horizon)
            goal = jax.device_put(
                jnp.asarray(
                    _task_goal(world, progress, float(when), config.navigation_start_seconds),
                    dtype=jnp.float32,
                ),
                device,
            )
            goals.append(np.asarray(goal))
            waypoint_indices.append(progress.completed)
            started = time.perf_counter()
            decision = jax.block_until_ready(
                controller(state, persistent.params, model, prediction, previous, goal)
            )
            service = time.perf_counter() - started
            controller_seconds.append(service)
            gradient = (
                float(last_update_metrics.gradient_norm) if last_update_metrics is not None else 0.0
            )
            update_norm = (
                float(last_update_metrics.parameter_update_norm)
                if last_update_metrics is not None
                else 0.0
            )
            _append_method_record(
                records,
                state,
                decision,
                library_version=int(persistent.library_version),
                cumulative_gradient_steps=int(persistent.cumulative_gradient_steps),
                diversity_loss=0.0,
                descriptor_target_loss=0.0,
                gradient_norm=gradient,
                parameter_update_norm=update_norm,
                estimated_wind=model.wind_velocity,
                snapshot_age_seconds=float(when - last_training_time),
                controller_seconds=service,
            )
            encounter = nominal_encounter_metrics(
                np.asarray(decision.candidates.states)[0, :, :3],
                prediction,
                dt=world.config.dt,
                ego_radius=dynamics.ego_radius,
                obstacle_clearance=world.config.obstacle_clearance,
            )
            encounters.append({"time": float(when), **encounter})
            if index % config.probe_every_controls == 0:
                snapshots.append((index, state, model, persistent.params, goal, previous))
            for key, value in {
                "hard": decision.values.values,
                "smooth": decision.smooth_values,
                "candidate_valid": decision.candidates.valid,
                "point_wind": model.wind_velocity,
                "actual_wind": actual_model.wind_velocity,
                "actual_mass": actual_model.mass,
                "actual_inertia": actual_model.inertia,
                "candidate_wrenches": decision.candidates.wrenches,
                "selected_index": decision.selected_index,
                "applied_operational_margin": decision.applied_held_operational_margin,
                "applied_operational_residual": decision.applied_held_operational_residual,
            }.items():
                local_raw[key].append(np.asarray(value))
            diagnosis = not first_diagnosis_saved and (
                float(np.max(decision.values.values)) < 0 or bool(decision.degraded)
            )
            if method == "adaptive" and (
                index % config.save_periodic_snapshots_controls == 0 or diagnosis
            ):
                for label, snapshot in (
                    (("published", persistent), ("previous", prior_snapshot))
                    if diagnosis
                    else (("published", persistent),)
                ):
                    checkpoint_buffers.append(
                        (
                            snapshot,
                            state,
                            directory / f"snapshots/{index:04d}-{label}",
                            {
                                **reference_metadata,
                                "simulation_time": float(when),
                                "previous_policy_index": int(previous),
                                "goal_for_external_replay_only": np.asarray(goal).tolist(),
                                "point_wind_for_external_replay": np.asarray(
                                    model.wind_velocity
                                ).tolist(),
                                "reason": "first_degraded_or_negative" if diagnosis else "periodic",
                            },
                        )
                    )
                first_diagnosis_saved |= diagnosis
            training_state = state
            interval_states = [np.asarray(state)]
            for _ in range(world.config.control_interval_steps):
                following = plant(state, decision.action, actual_model)
                if config.model_information == "estimated":
                    estimator = estimate(estimator, state, following, decision.action, actual_model)
                state = following
                interval_states.append(np.asarray(state))
                states.append(np.asarray(state))
            jax.block_until_ready((state, estimator))
            previous = decision.selected_index
            physical_clearance = _minimum_segment_clearance(
                np.asarray(interval_states),
                np.asarray(prediction.centers)[: world.config.control_interval_steps + 1],
                world.obstacle_radii + dynamics.ego_radius,
            )
            interval_collision = None
            terminate_for_collision = physical_clearance <= 0
            if config.termination_geometry == "modeled_collider":
                interval_collision = evaluate_collision_termination(
                    world,
                    float(when) + np.arange(len(interval_states)) * world.config.dt,
                    np.asarray(interval_states),
                    termination_geometry=config.termination_geometry,
                )
                terminate_for_collision = interval_collision["terminate"]
                if terminate_for_collision and collision_event is None:
                    collision_event = {
                        key: interval_collision[key]
                        for key in ("collision_kind", "first_intersection_time_seconds")
                    }
            progress = advance_waypoints(
                world,
                progress,
                np.asarray(state)[:3],
                float(when + world.config.control_period),
                physical_collision=terminate_for_collision,
                navigation_enabled=float(when) >= config.navigation_start_seconds - 1e-10,
            )
            update_seconds, finite = 0.0, False
            learner_started_wall = None
            learner_completed_wall = None
            if (
                method == "adaptive"
                and config.enable_learning
                and schedule.opportunities[index]
                and progress.termination is None
                and (
                    config.execution_mode == "deterministic"
                    or scheduler.can_start(time.perf_counter(), deadline)
                )
            ):
                started = time.perf_counter()
                changed, metrics = jax.block_until_ready(
                    learner.step(persistent, training_state, model)
                )
                learner_started_wall = started - epoch
                update_seconds = time.perf_counter() - started
                learner_completed_wall = started + update_seconds - epoch
                learner_seconds.append(update_seconds)
                finite = bool(metrics.finite_update_applied)
                pending = changed, metrics, float(when)
                if config.execution_mode == "budgeted":
                    scheduler.complete(
                        CompletedSnapshot(
                            changed,
                            int(changed.library_version),
                            float(when),
                            started,
                            started + update_seconds,
                            float(metrics.gradient_norm),
                            float(metrics.parameter_update_norm),
                        )
                    )
                records["diversity_loss"][-1] = float(metrics.loss.diversity)
                records["descriptor_target_loss"][-1] = float(metrics.loss.descriptor_target)
            local_raw["finite_update"].append(finite)
            records["learner_seconds"][-1] = update_seconds
            finished_wall = time.perf_counter()
            records["missed_deadline"][-1] = (
                finished_wall > deadline
                if config.execution_mode == "budgeted"
                else service + update_seconds > world.config.control_period
            )
            boundaries.append(
                {
                    "time": float(when),
                    "scheduled_wall_seconds": scheduled_boundary - epoch,
                    "started_wall_seconds": boundary_wall - epoch,
                    "completed_wall_seconds": finished_wall - epoch,
                    "missed_deadline": bool(records["missed_deadline"][-1]),
                    "version_used": int(persistent.library_version),
                    "update_launched": update_seconds > 0,
                    "finite": finite,
                    "completed_version": int(pending[0].library_version) if pending else None,
                    "controller_seconds": service,
                    "learner_seconds": update_seconds,
                    "learner_started_wall_seconds": learner_started_wall,
                    "learner_completed_wall_seconds": learner_completed_wall,
                    "physical_clearance_m": physical_clearance
                    if np.isfinite(physical_clearance)
                    else None,
                    "body_origin_enclosure_breach": bool(physical_clearance <= 0),
                    "requested_shell_breach": bool(
                        physical_clearance <= world.config.obstacle_clearance
                    ),
                    "termination_geometry": config.termination_geometry,
                    "modeled_collider_clearance_bounds_m": (
                        {
                            key: [
                                interval_collision["audit"][key][f"minimum_clearance_{bound}_bound_m"]
                                for bound in ("lower", "upper")
                            ]
                            for key in ("actual_xml_sphere_geometry", "actual_xml_ground_geometry")
                        }
                        if interval_collision is not None
                        else None
                    ),
                }
            )
            if progress_callback and index % 100 == 0:
                progress_callback(method, index, count, progress.completed)
        method_trace = replace(
            _method_trace(records),
            goal_position=np.asarray(goals),
            waypoint_index=np.asarray(waypoint_indices),
            recorded_control_valid=np.asarray(active),
            physical_collision_recorded=(
                ~np.asarray(active) & (progress.termination == "physical_collision")
            ),
        )
        methods[method] = method_trace
        probe_snapshots[method] = snapshots
        dense[method] = np.asarray(states)
        execution_audit, actual_operational_margins = audit_navigation_execution(
            dense[method], method_trace, world
        )
        raw[f"{method}_actual_operational_margins"] = actual_operational_margins
        for key, values in local_raw.items():
            raw[f"{method}_{key}"] = np.asarray(values)
        raw[f"{method}_active_control"] = np.asarray(active)
        all_centers, _ = world.obstacle_kinematics(np.arange(len(states)) * world.config.dt)
        clearance = _minimum_segment_clearance(
            np.asarray(states), all_centers, world.obstacle_radii + world.config.ego_radius
        )
        mask = np.asarray(active)
        blocked = np.asarray([row["nominal_blocked"] for row in encounters], dtype=bool)
        modeled_audit = (
            audit_recorded_collider_clearance(
                world, np.arange(len(states)) * world.config.dt, dense[method]
            )
            if config.termination_geometry == "modeled_collider"
            else None
        )
        summaries[method] = {
            "waypoints_completed": progress.completed,
            "waypoints_total": len(world.waypoint_positions),
            "termination": progress.termination,
            "termination_time_seconds": progress.termination_time_seconds,
            "arrival_times_seconds": progress.arrival_times_seconds,
            "physical_collision": progress.termination == "physical_collision",
            "termination_geometry": config.termination_geometry,
            "collision_event": collision_event,
            "termination_time_scope": "following control boundary after the executed hold",
            "body_origin_enclosure_breach_recorded": bool(clearance <= 0),
            "requested_shell_breach_recorded": bool(clearance <= world.config.obstacle_clearance),
            "modeled_collider_audit": modeled_audit,
            "collision_observation": (
                summarize_collision_observation(
                    modeled_audit,
                    termination_geometry=config.termination_geometry,
                    termination=progress.termination,
                )
                if modeled_audit is not None
                else {
                    "modeled_collision_observation": "not_audited_by_legacy_runner",
                    "enclosure_termination_censors_later_collider_outcome": (
                        progress.termination == "physical_collision"
                    ),
                }
            ),
            "minimum_physical_clearance_m": clearance if np.isfinite(clearance) else None,
            "minimum_inflated_clearance_m": clearance - world.config.obstacle_clearance
            if np.isfinite(clearance)
            else None,
            "active_controls": int(np.sum(mask)),
            "degraded_controls": int(np.sum(method_trace.degraded[mask])),
            "accepted_qp_controls": int(np.sum(method_trace.qp_valid[mask])),
            "executed_positive_policy_dual_controls": int(
                np.sum(method_trace.executed_policy_dual[mask] > 1e-7)
            ),
            "emergency_controls": int(np.sum(method_trace.used_emergency[mask])),
            "finite_updates": int(np.sum(np.asarray(local_raw["finite_update"])[mask])),
            "controller_service": _timing_statistics(controller_seconds),
            "learner_service": _timing_statistics(learner_seconds),
            "warmup": warmup_metadata,
            "service_exceeds_nominal_period_count": int(np.sum(method_trace.missed_deadline[mask])),
            "nominal_blocked_fraction": float(np.mean(blocked)),
            "separate_nominal_encounter_episodes": int(
                np.sum(blocked & ~np.r_[False, blocked[:-1]])
            ),
            "encounters": encounters,
            "publications_and_inputs": boundaries,
            "snapshot_publications": [
                {
                    key: value - epoch if key.endswith("wall_time") else value
                    for key, value in publication.items()
                }
                for publication in scheduler.publications
            ]
            if config.execution_mode == "budgeted"
            else [],
            "execution_audit": execution_audit,
        }
        final_published = pending[0] if pending is not None else persistent
        if config.execution_mode == "budgeted":
            terminal_wall = epoch + float(progress.termination_time_seconds)
            if terminal_wall > time.perf_counter():
                time.sleep(terminal_wall - time.perf_counter())
            final_published = scheduler.publish(
                time.perf_counter(), float(progress.termination_time_seconds)
            ).state
        for snapshot, snapshot_state, stem, snapshot_metadata in checkpoint_buffers:
            save_learner_checkpoint(
                snapshot,
                bundle.spec,
                bundle.config,
                bundle.actuator,
                snapshot_state,
                stem,
                metadata=snapshot_metadata,
            )
        if method == "adaptive":
            save_learner_checkpoint(
                final_published,
                bundle.spec,
                bundle.config,
                bundle.actuator,
                state,
                directory / "final_adaptive_checkpoint",
                metadata={
                    **reference_metadata,
                    "terminal_publication": True,
                    "learning_is_censored_at_termination": True,
                },
            )
    # Every pair below shares a state, model, goal, and absolute-time obstacle prediction.
    # Evaluation occurs after execution and never changes the published learner sequence.
    probes, probe_arrays = [], {}
    fixed_params = bundle.state.params
    adaptive_at = {item[0]: item[3] for item in probe_snapshots["adaptive"]}
    for anchor in ("fixed", "adaptive"):
        for index, state, model, _, goal, previous in probe_snapshots[anchor]:
            if index not in adaptive_at:
                # Only compare actually published snapshots before either branch terminates.
                continue
            prediction = world.obstacle_prediction(
                float(times[index]), horizon=bundle.config.horizon
            )
            for label, params in (("fixed", fixed_params), ("adaptive", adaptive_at[index])):
                decision = jax.block_until_ready(
                    controller(state, params, model, prediction, previous, goal)
                )
                probes.append(
                    {
                        "time": float(times[index]),
                        "anchor": anchor,
                        "library": label,
                        "fallback_max_hard": _finite_probe_value(
                            np.max(decision.values.values[1:])
                        ),
                        "augmented_max_hard": _finite_probe_value(np.max(decision.values.values)),
                        "collision_constraint_active": bool(decision.collision_constraint_active),
                        "candidate_input_valid": np.asarray(decision.values.input_valid).tolist(),
                        "eligible_count": int(decision.eligible_candidate_count),
                        "accepted_qp": bool(decision.qp_valid),
                    }
                )
                probe_arrays[f"{index}_{anchor}_{label}_full_states"] = np.asarray(
                    decision.candidates.states
                )
    np.savez_compressed(directory / "raw_diagnostics.npz", **raw)
    np.savez_compressed(directory / "dense_plant_states.npz", **dense)
    np.savez_compressed(directory / "same_state_probe_trajectories.npz", **probe_arrays)
    centers, _ = world.obstacle_kinematics(times)
    wind_events = tuple(event.time_seconds for event in world.config.wind_events)
    payload = world.config.payload_events[0] if world.config.payload_events else None
    trace = ComparisonVideoTrace(
        time_seconds=times,
        goal_position=world.waypoint_positions[0],
        obstacles=tuple(
            ObstacleTrack(
                centers[:, k],
                float(radius),
                float(radius + world.config.ego_radius + world.config.obstacle_clearance),
                f"moving-{k}",
            )
            for k, radius in enumerate(world.obstacle_radii)
        ),
        true_wind=np.asarray([world.wind_at(t) for t in times]),
        estimated_wind=methods["adaptive"].estimated_wind,
        wind_change_time=wind_events[0] if wind_events else float(times[-1]),
        wind_event_times_seconds=np.asarray(wind_events),
        descriptor_targets=np.asarray(bundle.spec.target_descriptors),
        fixed=methods["fixed"],
        adaptive=methods["adaptive"],
        title=(
            (
                "Hover → navigation · learning wind correction in fallback maneuvers"
                if config.fallback_mapping == "matched_uncompensated"
                else "Hover control · fallback maneuvers already correct for known wind"
            )
            if config.navigation_start_seconds > 0
            else (
                f"3D waypoint navigation · {world.config.obstacle_count} moving obstacles · "
                f"{config.model_information}"
            )
        ),
        left_label="FIXED MANEUVERS · SAME MODEL INFORMATION",
        right_label="LEARNING MANEUVERS · SAME MODEL INFORMATION",
        show_wind_change_banner=bool(wind_events),
        drone_radius=world.config.ego_radius,
        drone_model="cf21B_500",
        physical_model_name="cf21B_500",
        payload_attachment_time_seconds=payload.time_seconds if payload else None,
        payload_half_extents=np.asarray(payload.half_extents) if payload else None,
        payload_mass_delta_kg=payload.mass_fraction * float(base.mass) if payload else None,
        payload_base_mass_kg=float(base.mass) if payload else None,
        task_phase=np.where(times < config.navigation_start_seconds - 1e-10, "hover", "navigation"),
        phase_caption=np.asarray(
            [_phase_caption(world, float(when), config.navigation_start_seconds) for when in times]
        ),
    )
    trace.validate()
    summary = {
        "experiment": "navigation",
        "config": asdict(config),
        "world": world.metadata(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": bundle.sha256,
        "initial_library_version": int(bundle.state.library_version),
        "compensation_protocol": {
            "checkpoint": expected_compensation,
            "prefix": expected_compensation,
            "post_event": expected_compensation,
            "same_fallback_mapping_both_methods": True,
            "nominal_task_controller_compensation": True,
            "fallback_mapping": config.fallback_mapping,
        },
        "task_phases": {
            "hover_until_seconds": config.navigation_start_seconds,
            "hover_position": world.initial_state[:3].tolist(),
            "no_waypoint_progress_during_hover": True,
        },
        "schedule": {
            **schedule.metadata(),
            "mode": "exogenous_deterministic_opportunity_mask",
            "actual_execution_mode": config.execution_mode,
            "publication": (
                "completed snapshot at actual paced boundary"
                if config.execution_mode == "budgeted"
                else "completed snapshot at next simulated boundary"
            ),
        },
        "execution_mode": config.execution_mode,
        "deadline_miss_semantics": (
            "reported failures do not automatically lengthen simulated command holds"
        ),
        "methods": summaries,
        "same_state_probes": probes,
        "reference_contract": contract is not None,
        "execution_scope": (
            "paced serialized controller and budgeted learner; complete synchronized snapshots "
            "published at actual boundaries; sampled simulation, no OS hard-real-time guarantee"
            if config.execution_mode == "budgeted"
            else "deterministic synchronized mechanism replay; measured service reported, "
            "no real-time deadline claim"
        ),
        "collision_scope": (
            "simulation termination at definite rotated cf21B XML-sphere obstacle or floor "
            "intersection; enclosure/shell breaches remain logged without stopping control; "
            "recorded-state interpolation bounds do not establish measured MuJoCo contact"
            if config.termination_geometry == "modeled_collider"
            else "swept relative chords of the configured spherical body-origin enclosure at "
            "integration nodes; enclosure violations are not measured MuJoCo contacts, and "
            "accelerated obstacle arcs are not continuously certified; enclosure termination "
            "censors subsequent actual-collider outcomes"
        ),
    }
    from crazyflow.safety.da_plcbf.runtime_feasibility import assess_navigation_runtime_feasibility

    summary["runtime_feasibility"] = assess_navigation_runtime_feasibility(summary)
    result = OnlineConstantWindResult(trace, summary, methods)
    save_online_constant_wind_result(result, directory, stem="navigation_comparison")
    (directory / "SOURCE_SHA256.json").write_text(
        json.dumps(
            {
                "crazyflow/safety/da_plcbf/navigation_experiment.py": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest()
            },
            indent=2,
        )
        + "\n"
    )
    return result
