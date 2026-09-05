"""Common-state closed-loop branches for selected persistent-wind encounters.

The production controller keeps all nominal, QP, fallback and emergency behavior. Envelope
breach is recorded but never causes motor cut; original commands continue until the oriented
asset geometry intersects. That geometric event is distinct from measured MuJoCo contact.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.da_plcbf_case_discovery import (
    CASE_CHECKPOINTS,
    validate_atlas_branch_snapshot,
    write_json,
)
from benchmark.da_plcbf_case_runtime import _tree_digest
from crazyflow.safety.da_plcbf.case_study_world import (
    HoverEncounterConfig,
    IncomingSphere,
    audit_recorded_collider_clearance,
    build_hover_encounter_world,
)
from crazyflow.safety.da_plcbf.learner_checkpoint import (
    load_learner_checkpoint,
    save_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.mujoco_comparison_video import ComparisonVideoTrace, ObstacleTrack
from crazyflow.safety.da_plcbf.navigation_experiment import (
    NavigationExperimentConfig,
    _append_terminal_record,
    build_navigation_controller,
)
from crazyflow.safety.da_plcbf.online_constant_wind import (
    OnlineConstantWindResult,
    _append_method_record,
    _empty_method_records,
    _method_trace,
    save_online_constant_wind_result,
)
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.state_conditioned_learning import (
    build_reference_skill_learner_from_checkpoint,
    reference_contract_checkpoint_metadata,
    save_reference_contract,
)


def persistent_provenance(persistent: Any) -> dict[str, Any]:
    """Hash the exact parameters, previous parameters, Adam history and complete learner state."""
    return {
        "library_version": int(persistent.library_version),
        "cumulative_gradient_steps": int(persistent.cumulative_gradient_steps),
        "persistent_state_sha256": _tree_digest(persistent),
        "parameters_sha256": _tree_digest(persistent.params),
        "previous_parameters_sha256": _tree_digest(persistent.previous_params),
        "optimizer_state_sha256": _tree_digest(persistent.optimizer_state),
    }


def encounter_from_row(row: dict) -> HoverEncounterConfig:
    start = row["time_seconds"]
    return HoverEncounterConfig(
        incoming=IncomingSphere(
            arrival_time_seconds=start + row["arrival_delay"],
            direction=tuple(row["direction"]),
            speed_m_s=row["speed"],
            radius_m=row["radius"],
            crossing_offset=tuple(row["crossing_offset"]),
            amplitude_m=30,
        ),
        duration_seconds=start + 8,
        navigation_start_seconds=start + 3,
    )


def run_branch(
    world: Any,
    bundle: Any,
    persistent: Any,
    controller: Any,
    *,
    end: float,
    learner: Any = None,
    snapshot_available: float,
    record_video: bool = False,
    plant_substeps: int = 1,
    record_provenance: bool = False,
    provenance_directory: Path | None = None,
    reference_contract: Any = None,
) -> tuple[dict, list, np.ndarray, np.ndarray, Any]:
    """Run one immutable-start branch on the world's absolute clock, no crash fabrication.

    Optional provenance hashes every observed/published learner state and completed update.
    A provenance directory additionally retains initial/final full Adam checkpoints bound to
    the supplied immutable nominal reference. Simulation publication boundaries are deterministic
    opportunities; separately measured service clocks do not imply paced online availability.
    """
    start = world.initial_state_time_seconds
    if not np.isfinite(snapshot_available) or snapshot_available > start:
        raise ValueError("branch cannot use a future learner snapshot")
    if type(plant_substeps) is not int or plant_substeps < 1:
        raise ValueError("plant_substeps must be a positive integer")
    if provenance_directory is not None and reference_contract is None:
        raise ValueError("provenance checkpoints require the bound nominal reference contract")
    record_provenance = record_provenance or provenance_directory is not None
    if (
        bundle.config.dt != world.config.dt
        or bundle.config.control_interval_steps != world.config.control_interval_steps
    ):
        raise ValueError("branch must preserve checkpoint prediction and command cadence")
    if end <= start or not np.isclose(
        (end - start) / world.config.control_period,
        round((end - start) / world.config.control_period),
    ):
        raise ValueError("branch end must be a later control boundary")
    period, dt = world.config.control_period, world.config.dt / plant_substeps
    count = round((end - start) / period)
    x = jnp.asarray(world.initial_state, dtype=jnp.float32)
    initial_provenance = persistent_provenance(persistent) if record_provenance else None
    checkpoint_binding = {}
    if provenance_directory is not None:
        provenance_directory.mkdir(parents=True, exist_ok=False)
        save_reference_contract(reference_contract, provenance_directory / "nominal_reference")
        checkpoint_binding = reference_contract_checkpoint_metadata(
            provenance_directory / "nominal_reference"
        )
        save_learner_checkpoint(
            persistent,
            bundle.spec,
            bundle.config,
            bundle.actuator,
            x,
            provenance_directory / "initial_checkpoint",
            metadata={
                **checkpoint_binding,
                "available_time_seconds": snapshot_available,
                "physical_state_time_seconds": start,
                "continuation_provenance": initial_provenance,
            },
        )
    goal = jnp.asarray(world.case_study_config.hover_position, dtype=jnp.float32)
    previous = jnp.asarray(-1, jnp.int32)
    base = bundle.point_model
    plant = jax.jit(lambda state, u, m: direct_wrench_symplectic_step(state, u, m, dt))
    records = _empty_method_records() if record_video else None
    states, control, dense_times, rows = [np.asarray(x)], [], [start], []
    terminated = None
    valid = []
    times = start + np.arange(count + 1) * period
    waypoint = 0
    for index, when in enumerate(times):
        if terminated is not None:
            if record_video:
                _append_terminal_record(records, np.asarray(x))
                valid.append(False)
            continue
        model = world.dynamics_at(float(when), base).model
        if when >= world.case_study_config.navigation_start_seconds:
            goal = jnp.asarray(world.waypoint_positions[waypoint], dtype=jnp.float32)
            if np.linalg.norm(np.asarray(x[:3] - goal)) <= world.config.reach_radius:
                waypoint += 1
                if waypoint == len(world.waypoint_positions):
                    terminated = "task_complete"
                    waypoint -= 1
                goal = jnp.asarray(world.waypoint_positions[waypoint], dtype=jnp.float32)
        prediction = world.obstacle_prediction(float(when), horizon=bundle.config.horizon)
        jax.block_until_ready((x, model, prediction, goal, persistent.params))
        started = time.perf_counter_ns()
        decision = jax.block_until_ready(
            controller(x, persistent.params, model, prediction, previous, goal)
        )
        completed = time.perf_counter_ns()
        service = (completed - started) * 1e-9
        # Terminal boundary is diagnostic only; never counts as an executed command.
        execute = index < count and terminated is None
        valid.append(execute)
        if record_video:
            _append_method_record(
                records,
                x,
                decision,
                library_version=int(persistent.library_version),
                cumulative_gradient_steps=int(persistent.cumulative_gradient_steps),
                diversity_loss=0,
                descriptor_target_loss=0,
                gradient_norm=0,
                parameter_update_norm=0,
                estimated_wind=model.wind_velocity,
                snapshot_age_seconds=float(when - snapshot_available),
                controller_seconds=service,
            )
        row = {
            "time": float(when),
            "executed": execute,
            "version": int(persistent.library_version),
            "snapshot_available_time": snapshot_available,
            "state": np.asarray(x).tolist(),
            "goal": np.asarray(goal).tolist(),
            "selected": int(decision.selected_index),
            "mode": int(decision.execution_mode),
            "qp": bool(decision.qp_valid),
            "degraded": bool(decision.degraded),
            "hard": np.asarray(decision.values.values).tolist(),
            "smooth": np.asarray(decision.smooth_values).tolist(),
            "eligible": np.asarray(decision.continuous_filter.policy_eligible).tolist(),
            "dual": float(decision.executed_policy_dual),
            "action": np.asarray(decision.action).tolist(),
            "controller_seconds": service,
            "rejections": np.asarray(decision.qp_rejection_flags).tolist(),
        }
        if record_provenance:
            row["published_learner_state"] = persistent_provenance(persistent)
            row["controller_started_perf_counter_ns"] = started
            row["controller_completed_perf_counter_ns"] = completed
        rows.append(row)
        if not execute:
            continue
        next_persistent = persistent
        if learner is not None:
            update_started = time.perf_counter_ns()
            next_persistent, metrics = jax.block_until_ready(learner.step(persistent, x, model))
            update_completed = time.perf_counter_ns()
            row["finite_update"] = bool(metrics.finite_update_applied)
            if record_provenance:
                following = persistent_provenance(next_persistent)
                before = row["published_learner_state"]
                increment = int(row["finite_update"])
                for counter in ("library_version", "cumulative_gradient_steps"):
                    if following[counter] != before[counter] + increment:
                        raise AssertionError("completed learner counters disagree with publication")
                row["completed_update"] = {
                    "finite_update_applied": row["finite_update"],
                    "training_time_seconds": float(when),
                    "publication_time_seconds": float(when + period) if increment else None,
                    "training_state_sha256": _tree_digest(x),
                    "point_model_sha256": _tree_digest(model),
                    "started_perf_counter_ns": update_started,
                    "completed_perf_counter_ns": update_completed,
                    "synchronized_service_seconds": (update_completed - update_started) * 1e-9,
                    "before": before,
                    "after": following,
                }
        interval = [np.asarray(x)]
        control.append(np.asarray(decision.action))
        for substep in range(world.config.control_interval_steps * plant_substeps):
            x = plant(x, decision.action, model)
            interval.append(np.asarray(x))
            states.append(np.asarray(x))
            dense_times.append(float(when + (substep + 1) * dt))
        audit = audit_recorded_collider_clearance(
            world, when + np.arange(len(interval)) * dt, np.asarray(interval)
        )
        if audit["actual_xml_sphere_geometry"]["minimum_clearance_upper_bound_m"] < 0:
            terminated = "asset_geometry_intersection"
        elif audit["actual_xml_ground_geometry"]["minimum_clearance_upper_bound_m"] < 0:
            terminated = "asset_ground_intersection"
        previous = decision.selected_index
        persistent = next_persistent
        if learner is not None and row["finite_update"]:
            snapshot_available = float(when + period)
    audit = audit_recorded_collider_clearance(world, np.asarray(dense_times), np.asarray(states))
    executed = [r for r in rows if r["executed"]]

    def first(predicate: Any) -> float | None:
        return next((r["time"] for r in executed if predicate(r)), None)

    summary = {
        "termination": terminated or "window_end",
        "executed_controls": len(executed),
        "degraded_controls": sum(r["degraded"] for r in executed),
        "qp_controls": sum(r["qp"] for r in executed),
        "fallback_controls": sum(r["mode"] == 1 for r in executed),
        "emergency_controls": sum(r["mode"] == 2 for r in executed),
        "learned_constraint_positive_dual_controls": sum(
            r["qp"] and r["selected"] > 0 and r["dual"] > 1e-7 for r in executed
        ),
        "first_no_certificate": first(lambda r: max(r["hard"]) < 0),
        "first_no_certificate_definition": "legacy field: no nonnegative hard collision path",
        "first_no_hard_collision_path": first(lambda r: max(r["hard"]) < 0),
        "first_no_eligible_certificate": first(lambda r: not any(r["eligible"])),
        "first_qp_rejection": first(lambda r: not r["qp"]),
        "first_degraded": first(lambda r: r["degraded"]),
        "finite_updates": sum(r.get("finite_update", False) for r in executed),
        "first_version": rows[0]["version"],
        "last_used_version": executed[-1]["version"],
        "waypoints_completed": waypoint + (terminated == "task_complete"),
        "geometry_audit": audit,
    }
    if record_provenance:
        summary["continuation_provenance"] = {
            "scope": "deterministic publication boundaries; measured service clocks are separate",
            "initial": initial_provenance,
            "final": persistent_provenance(persistent),
            "final_snapshot_available_time_seconds": snapshot_available,
            "last_executed_control_version": summary["last_used_version"],
            "checkpoint_directory": None
            if provenance_directory is None
            else str(provenance_directory),
        }
    if provenance_directory is not None:
        save_learner_checkpoint(
            persistent,
            bundle.spec,
            bundle.config,
            bundle.actuator,
            x,
            provenance_directory / "final_checkpoint",
            metadata={
                **checkpoint_binding,
                "available_time_seconds": snapshot_available,
                "physical_state_time_seconds": dense_times[-1],
                "last_executed_control_version": summary["last_used_version"],
                "continuation_provenance": summary["continuation_provenance"]["final"],
            },
        )
        write_json(
            provenance_directory / "publication_ledger.json",
            {
                "summary": summary["continuation_provenance"],
                "rows": [
                    {
                        key: row[key]
                        for key in (
                            "time",
                            "executed",
                            "version",
                            "snapshot_available_time",
                            "published_learner_state",
                            "controller_started_perf_counter_ns",
                            "controller_completed_perf_counter_ns",
                        )
                    }
                    | (
                        {"completed_update": row["completed_update"]}
                        if "completed_update" in row
                        else {}
                    )
                    for row in rows
                ],
            },
        )
    trace = None
    if record_video:
        trace = replace(
            _method_trace(records),
            recorded_control_valid=np.asarray(valid),
            physical_collision_recorded=np.asarray(
                [
                    terminated in {"asset_geometry_intersection", "asset_ground_intersection"}
                    and t >= dense_times[-1]
                    for t in times
                ]
            ),
        )
    return summary, rows, np.asarray(states), np.asarray(dense_times), trace


def screen_cases(
    atlas: Path,
    selected: Path,
    output: Path,
    *,
    limit: int,
    device: Any,
    record_video: bool = False,
):
    output.mkdir(parents=True, exist_ok=False)
    selections = json.loads(selected.read_text())
    ledger = []
    for case in sorted({r["case"] for r in selections}):
        source = load_learner_checkpoint(CASE_CHECKPOINTS[case], device=device)
        controller = None
        for rank, row in enumerate([r for r in selections if r["case"] == case][:limit]):
            bundle, contract, learner = build_reference_skill_learner_from_checkpoint(
                atlas / case / row["anchor"], device=device
            )
            available = validate_atlas_branch_snapshot(bundle, row["time_seconds"])
            cfg = encounter_from_row(row)
            world = build_hover_encounter_world(
                cfg,
                initial_state=np.asarray(bundle.physical_state),
                initial_time_seconds=row["time_seconds"],
            )
            config = NavigationExperimentConfig(
                navigation_start_seconds=cfg.navigation_start_seconds,
                fallback_mapping="compensated"
                if source.config.model_compensation
                else "matched_uncompensated",
            )
            # All rows of a family share controller constants/obstacle count. Current predicted
            # centers/radii replace the static template inside continuous_version_a_step.
            if controller is None:
                controller = build_navigation_controller(world, bundle, config)
            directory = output / f"{case}-{rank:03d}-{row['anchor']}-{row['index']}"
            directory.mkdir()
            write_json(directory / "selection.json", row)
            write_json(directory / "world.json", world.metadata())
            write_json(directory / "encounter.json", asdict(cfg))
            results, video = {}, {}
            raw = {}
            for method, persistent in (("fixed", source.state), ("adaptive_held", bundle.state)):
                summary, rows, dense, dense_times, trace = run_branch(
                    world,
                    source,
                    persistent,
                    controller,
                    end=row["time_seconds"] + 3,
                    snapshot_available=available if method != "fixed" else 0,
                    record_video=record_video,
                )
                results[method] = summary
                write_json(directory / f"{method}_controls.json", rows)
                raw[f"{method}_states"], raw[f"{method}_times"] = dense, dense_times
                if trace is not None:
                    video[method] = trace
                print(
                    directory.name,
                    method,
                    summary["termination"],
                    "shell",
                    round(summary["geometry_audit"]["safety_shell"]["minimum_clearance_m"], 5),
                    "degraded",
                    summary["degraded_controls"],
                    flush=True,
                )
            write_json(directory / "summary.json", results)
            np.savez_compressed(directory / "dense_states.npz", **raw)
            if record_video:
                times = row["time_seconds"] + np.arange(len(video["fixed"].position)) * 0.04
                centers, _ = world.obstacle_kinematics(times)
                winds = np.stack([world.wind_at(t) for t in times])
                trace = ComparisonVideoTrace(
                    time_seconds=times,
                    goal_position=np.asarray(cfg.hover_position),
                    obstacles=tuple(
                        ObstacleTrack(centers[:, i], r, r + 0.15, f"Incoming {i}")
                        for i, r in enumerate(world.obstacle_radii)
                    ),
                    true_wind=winds,
                    estimated_wind=winds,
                    wind_change_time=float(times[0]),
                    descriptor_targets=np.asarray(source.spec.target_descriptors),
                    fixed=video["fixed"],
                    adaptive=video["adaptive_held"],
                    title="Persistent wind encounter: controlled same-state branches",
                    right_label="ADAPTED SNAPSHOT HELD FIXED",
                    show_wind_change_banner=False,
                    drone_radius=0.106,
                    drone_model="cf21B_500",
                    physical_model_name="Shared point model; held motors; geometric contact audit",
                    task_phase=np.full(len(times), "hover"),
                    phase_caption=np.full(
                        len(times),
                        "Wind remains active; same starting state and full safety filter",
                    ),
                )
                save_online_constant_wind_result(
                    OnlineConstantWindResult(trace, results), directory, stem="comparison"
                )
            ledger.append({"directory": str(directory), "selection": row, "methods": results})
            write_json(output / "closed_loop_ledger.json", ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    screen_cases(
        args.atlas,
        args.selected,
        args.output_dir,
        limit=args.limit,
        device=jax.devices(args.device)[0],
        record_video=args.record_video,
    )


if __name__ == "__main__":
    main()
