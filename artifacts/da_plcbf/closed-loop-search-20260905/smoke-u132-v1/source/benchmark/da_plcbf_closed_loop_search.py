"""Outcome-driven full-episode discovery with the complete matched PLCBF controllers.

Geometry is development data, never a learner input. Every evaluation starts at time zero,
updates on its own observed state, and retains ordinary QP, fallback, and emergency paths.
Discovery skips rendering and checkpoint serialization; selected replays retain full provenance.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.da_plcbf_case_discovery import CASE_CHECKPOINTS
from crazyflow.safety.da_plcbf.case_study_world import (
    GuardSphere,
    HoverEncounterConfig,
    IncomingSphere,
    audit_recorded_collider_clearance,
    build_hover_encounter_world,
)
from crazyflow.safety.da_plcbf.continuous_version_a import RuntimeObstacleTrajectories
from crazyflow.safety.da_plcbf.navigation_experiment import (
    NavigationExperimentConfig,
    build_navigation_controller,
    evaluate_collision_termination,
)
from crazyflow.safety.da_plcbf.navigation_world import WaypointProgress, advance_waypoints
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.state_conditioned_learning import (
    build_reference_skill_learner_from_checkpoint,
)

ROOT = Path("artifacts/da_plcbf/closed-loop-search-20260905")
OLD = Path("artifacts/da_plcbf/case-study-20260905")
BUFFERS = (0.15, 0.05, 0.02, 0.0)
FAMILIES = ("single", "guards", "staggered", "moving")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checksum(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def physical_class(method: dict) -> str:
    """Do not turn an enclosure stop, timeout, or uncertain interpolation into survival."""
    if method.get("censored", False):
        return "censored"
    lower, upper = method["collider_lower_m"], method["collider_upper_m"]
    if upper < 0 or method["ground_upper_m"] < 0:
        return "collision"
    if lower <= 0 or method["ground_lower_m"] <= 0:
        return "unresolved"
    return "separated"


def classify_pair(methods: dict) -> dict:
    fixed, adaptive = methods["fixed"], methods["adaptive"]
    f, a = physical_class(fixed), physical_class(adaptive)
    if f in {"censored", "unresolved"} or a in {"censored", "unresolved"}:
        outcome = "censored_or_unresolved"
    else:
        outcome = {
            (False, False): "both_separated",
            (True, False): "fixed_only_collision",
            (False, True): "adaptive_only_collision",
            (True, True): "both_collision",
        }[(f == "collision", a == "collision")]
    # Frozen tolerances apply to discovery only. Finer plant validation follows promotion.
    candidate = (
        fixed["collider_upper_m"] < -0.002
        and adaptive["collider_lower_m"] > 0.01
        and adaptive["ground_lower_m"] > 0.01
        and adaptive["all_operational_nodes_pass"]
        and adaptive["encounter_completed"]
        and adaptive["termination"] == "completed"
    )
    penalty = 8 * max(0.01 - adaptive["collider_lower_m"], 0)
    penalty += 8 * max(0.01 - adaptive["ground_lower_m"], 0)
    penalty += 2 * (not adaptive["all_operational_nodes_pass"])
    penalty += 1 * (not adaptive["encounter_completed"])
    penalty += 0.2 * (adaptive["termination"] != "completed")
    return {
        "outcome_class": outcome,
        "fixed_physical_class": f,
        "adaptive_physical_class": a,
        "promotion_candidate": bool(candidate),
        "objective": float(fixed["collider_upper_m"] + penalty),
        "clearance_advantage_m": float(adaptive["collider_lower_m"] - fixed["collider_upper_m"]),
    }


def operational_audit(states: np.ndarray, world: Any, motor_minimum: float) -> dict:
    cfg = world.config
    q = states[:, 3:7] / np.linalg.norm(states[:, 3:7], axis=1, keepdims=True)
    tilt = np.arccos(np.clip(1 - 2 * (q[:, 0] ** 2 + q[:, 1] ** 2), -1, 1))
    margins = {
        "arena_m": float(
            min(
                np.min(states[:, :3] - cfg.arena_lower - np.asarray(0.08)),
                np.min(np.asarray(cfg.arena_upper) - 0.08 - states[:, :3]),
            )
        ),
        "speed_m_s": float(cfg.speed_max - np.linalg.norm(states[:, 7:10], axis=1).max()),
        "angular_rate_rad_s": float(
            cfg.angular_rate_max - np.linalg.norm(states[:, 10:13], axis=1).max()
        ),
        "tilt_rad": float(cfg.tilt_max_radians - tilt.max()),
        "motor_N": motor_minimum,
    }
    return {
        "operational_minima": margins,
        "all_operational_nodes_pass": bool(all(v >= -1e-7 for v in margins.values())),
    }


class EpisodeEvaluator:
    """Reuse compiled controller shapes and one immutable reference contract per mapping."""

    def __init__(self, device: Any, *, plant_substeps: int = 1):
        """Bind one device and a fixed plant refinement for a discovery campaign."""
        if type(plant_substeps) is not int or plant_substeps < 1:
            raise ValueError("plant_substeps must be a positive integer")
        self.device = device
        self.plant_substeps = plant_substeps
        self.bundles: dict = {}
        self.controllers: dict = {}
        self.plants: dict = {}

    def resources(self, mapping: str, world: Any) -> tuple:
        """Cache functions only across matching physics, geometry conventions, and limits."""
        if mapping not in self.bundles:
            self.bundles[mapping] = build_reference_skill_learner_from_checkpoint(
                CASE_CHECKPOINTS[mapping], device=self.device
            )
        bundle, contract, learner = self.bundles[mapping]
        # Scene trajectories, initial conditions, goals, and wind are dynamic arguments.
        key = (
            mapping,
            world.config.obstacle_clearance,
            world.config.ego_radius,
            tuple(world.config.arena_lower),
            tuple(world.config.arena_upper),
            world.config.speed_max,
            world.config.angular_rate_max,
            world.config.tilt_max_radians,
        )
        if key not in self.controllers:
            cfg = NavigationExperimentConfig(
                learning_start_seconds=0,
                update_every_controls=1,
                navigation_start_seconds=world.case_study_config.navigation_start_seconds,
                fallback_mapping="compensated"
                if bundle.config.model_compensation
                else "matched_uncompensated",
                termination_geometry="modeled_collider",
            )
            cfg.validate(world)
            full = build_navigation_controller(world, bundle, cfg)

            @jax.jit
            def compact(
                x: Any, params: Any, model: Any, obstacles: Any, previous: Any, goal: Any
            ) -> dict:
                d = full(x, params, model, obstacles, previous, goal)
                return {
                    "action": d.action,
                    "selected": d.selected_index,
                    "mode": d.execution_mode,
                    "qp": d.qp_valid,
                    "degraded": d.degraded,
                    "emergency": d.used_emergency,
                    "fallback": d.used_fallback,
                    "dual": d.executed_policy_dual,
                    "hard": d.values.values,
                    "smooth": d.smooth_values,
                    "eligible": d.continuous_filter.policy_eligible,
                    "motor_minimum": d.applied_postcheck.minimum_motor_margin,
                    "held_operational_pass": d.applied_held_operational_passed,
                }

            self.controllers[key] = compact
        dt = world.config.dt / self.plant_substeps
        if dt not in self.plants:
            self.plants[dt] = jax.jit(
                lambda x, u, model: direct_wrench_symplectic_step(x, u, model, dt)
            )
        return bundle, contract, learner, self.controllers[key], self.plants[dt]

    def run_method(
        self,
        scene: HoverEncounterConfig,
        mapping: str,
        method: str,
        *,
        freeze_at: float | None = None,
    ) -> tuple[dict, dict]:
        """Execute the ordinary controller from zero, stopping after first actual contact."""
        if method not in {"fixed", "adaptive"}:
            raise ValueError("unknown method")
        world = build_hover_encounter_world(scene, initial_time_seconds=0.0)
        bundle, _contract, learner, controller, plant = self.resources(mapping, world)
        period = world.config.control_period
        count = round(world.config.duration_seconds / period)
        dt = world.config.dt / self.plant_substeps
        hold = world.config.control_interval_steps * self.plant_substeps
        times = np.arange(count) * period
        future = times[:, None] + np.arange(bundle.config.horizon + 1)[None] * world.config.dt
        centers, velocities = world.obstacle_kinematics(future)
        centers, velocities = jax.device_put(
            (centers.astype(np.float32), velocities.astype(np.float32)), self.device
        )
        radii = jax.device_put(world.obstacle_radii.astype(np.float32), self.device)
        mask = jax.device_put(np.ones(centers.shape[1:-1], dtype=bool), self.device)
        models = [
            jax.device_put(world.dynamics_at(t, bundle.point_model).model, self.device)
            for t in (0, scene.wind_onset_seconds)
        ]
        state = jax.device_put(world.initial_state.astype(np.float32), self.device)
        persistent = bundle.state
        previous = jax.device_put(jnp.asarray(-1, jnp.int32), self.device)
        progress = WaypointProgress()
        dense = [np.asarray(state)]
        rows: dict[str, list] = {
            k: []
            for k in (
                "time",
                "state",
                "action",
                "selected",
                "mode",
                "qp",
                "degraded",
                "emergency",
                "fallback",
                "dual",
                "hard",
                "smooth",
                "eligible",
                "motor_minimum",
                "held_operational_pass",
                "version_used",
                "completed_version",
                "finite_update",
                "publication_time",
                "controller_seconds",
                "learner_seconds",
            )
        }
        collision_kind = None
        onset_version = None
        started_episode = time.perf_counter()
        for index, when in enumerate(times):
            if progress.termination is not None:
                break
            model = models[int(when >= scene.wind_onset_seconds - 1e-10)]
            if onset_version is None and when >= scene.wind_onset_seconds - 1e-10:
                onset_version = int(persistent.library_version)
            goal = (
                world.initial_state[:3]
                if when < scene.navigation_start_seconds - 1e-10
                else progress.active_goal(world)
            )
            prediction = RuntimeObstacleTrajectories(centers[index], radii, mask, velocities[index])
            goal_device = jax.device_put(np.asarray(goal, np.float32), self.device)
            jax.block_until_ready((state, model, prediction, persistent.params, goal_device))
            started = time.perf_counter()
            decision = jax.block_until_ready(
                controller(state, persistent.params, model, prediction, previous, goal_device)
            )
            controller_seconds = time.perf_counter() - started
            rows["time"].append(float(when))
            rows["state"].append(np.asarray(state))
            rows["version_used"].append(int(persistent.library_version))
            for name, value in decision.items():
                rows[name].append(np.asarray(value))
            training_state = state
            interval = [np.asarray(state)]
            for _ in range(hold):
                state = plant(state, decision["action"], model)
                interval.append(np.asarray(state))
                dense.append(np.asarray(state))
            observed = evaluate_collision_termination(
                world,
                when + np.arange(hold + 1) * dt,
                np.asarray(interval),
                termination_geometry="modeled_collider",
            )
            if observed["terminate"]:
                collision_kind = observed["collision_kind"]
            progress = advance_waypoints(
                world,
                progress,
                np.asarray(state)[:3],
                float(when + period),
                physical_collision=observed["terminate"],
                navigation_enabled=when >= scene.navigation_start_seconds - 1e-10,
            )
            finite, elapsed = False, 0.0
            if (
                method == "adaptive"
                and progress.termination is None
                and (freeze_at is None or when < freeze_at - 1e-10)
            ):
                started = time.perf_counter()
                updated, metrics = jax.block_until_ready(
                    learner.step(persistent, training_state, model)
                )
                elapsed = time.perf_counter() - started
                finite = bool(metrics.finite_update_applied)
                # Publish every finite completed update at the next simulation boundary.
                persistent = updated
            rows["completed_version"].append(int(persistent.library_version))
            rows["finite_update"].append(finite)
            rows["publication_time"].append(float(when + period) if finite else -1.0)
            rows["controller_seconds"].append(controller_seconds)
            rows["learner_seconds"].append(elapsed)
            previous = decision["selected"]
        dense_array = np.asarray(dense)
        dense_times = np.arange(len(dense_array)) * dt
        audit = audit_recorded_collider_clearance(world, dense_times, dense_array)
        collider, ground = audit["actual_xml_sphere_geometry"], audit["actual_xml_ground_geometry"]
        arrays = {name: np.asarray(values) for name, values in rows.items()}
        arrays.update(dense_states=dense_array, dense_times=dense_times)
        arrivals = [scene.incoming.arrival_time_seconds] + [
            s.arrival_time_seconds for s in scene.additional_incoming
        ]
        summary = {
            "termination": progress.termination or "timeout",
            "collision_kind": collision_kind,
            "termination_time_seconds": progress.termination_time_seconds,
            "censored": False,
            "start_time_seconds": 0.0,
            "collider_lower_m": collider["minimum_clearance_lower_bound_m"],
            "collider_upper_m": collider["minimum_clearance_upper_bound_m"],
            "collider_clearance_m": collider["minimum_clearance_m"],
            "ground_lower_m": ground["minimum_clearance_lower_bound_m"],
            "ground_upper_m": ground["minimum_clearance_upper_bound_m"],
            "enclosure_clearance_m": audit["body_origin_envelope"]["minimum_clearance_m"],
            "shell_clearance_m": audit["safety_shell"]["minimum_clearance_m"],
            "first_collider_intersection_seconds": collider[
                "first_chord_intersection_time_seconds"
            ],
            "controls": len(rows["time"]),
            "qp_controls": int(arrays["qp"].sum()),
            "fallback_controls": int(arrays["fallback"].sum()),
            "emergency_controls": int(arrays["emergency"].sum()),
            "degraded_controls": int(arrays["degraded"].sum()),
            "executed_learned_positive_dual_controls": int(
                np.sum(arrays["qp"] & (arrays["selected"] > 0) & (arrays["dual"] > 1e-8))
            ),
            "finite_updates": int(arrays["finite_update"].sum()),
            "initial_version": int(bundle.state.library_version),
            "final_version": int(persistent.library_version),
            "last_executed_version": int(arrays["version_used"][-1]),
            "wind_onset_version": onset_version,
            "encounter_completed": bool(
                dense_times[-1] > max(arrivals) + 0.8 and collision_kind is None
            ),
            "waypoints_completed": progress.completed,
            "waypoints_total": len(world.waypoint_positions),
            "freeze_at_seconds": freeze_at,
            "execution_mode": "deterministic_next_boundary",
            "wall_seconds": time.perf_counter() - started_episode,
            "controller_median_seconds": float(np.median(arrays["controller_seconds"])),
            "learner_median_seconds": float(
                np.median(arrays["learner_seconds"][arrays["finite_update"]])
            )
            if arrays["finite_update"].any()
            else None,
            **operational_audit(dense_array, world, float(arrays["motor_minimum"].min())),
        }
        return summary, arrays

    def evaluate(self, scene: HoverEncounterConfig, mapping: str, directory: Path) -> dict:
        """Run both matched methods and retain compact controls and dense physical states."""
        directory.mkdir(parents=True, exist_ok=False)
        methods, traces = {}, {}
        for method in ("fixed", "adaptive"):
            methods[method], arrays = self.run_method(scene, mapping, method)
            traces.update({f"{method}_{k}": v for k, v in arrays.items()})
        np.savez_compressed(directory / "traces.npz", **traces)
        record = {
            "scene": asdict(scene),
            "scene_sha256": checksum(asdict(scene)),
            "mapping": mapping,
            "checkpoint": str(CASE_CHECKPOINTS[mapping]),
            "checkpoint_sha256": self.bundles[mapping][0].sha256,
            "methods": methods,
            **classify_pair(methods),
        }
        write_json(directory / "result.json", record)
        return record


def existing_proposals(seed: int, count: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    with gzip.open(OLD / "publication/all_candidates.jsonl.gz", "rt") as source:
        rows = [json.loads(line) for line in source]
    proposals = []
    # Half of the proposals are stratified rejected rows; no hard-H promotion gate.
    for index in range(count):
        mapping = ("uncompensated", "compensated")[index % 2]
        accepted = (index // 2) % 2 == 0
        family = ("geometry-v1", "geometry-v2", "geometry-v3")[(index // 4) % 3]
        candidates = [
            r
            for r in rows
            if r["case"] == mapping
            and r["family"] == family
            and bool(r["stage_b_accepted"]) == accepted
        ]
        if not candidates:
            candidates = [
                r for r in rows if r["case"] == mapping and bool(r["stage_b_accepted"]) == accepted
            ]
        row = candidates[int(rng.integers(len(candidates)))]
        arrival = row["time_seconds"] + row["arrival_delay"]
        nav = np.ceil((arrival + 1.2) / 0.04) * 0.04
        scene = HoverEncounterConfig(
            incoming=IncomingSphere(
                arrival,
                tuple(row["direction"]),
                row["speed"],
                row["radius"],
                tuple(row["crossing_offset"]),
                30,
            ),
            seed=seed + index,
            navigation_start_seconds=float(nav),
            duration_seconds=float(np.ceil((nav + 5) / 0.04) * 0.04),
            obstacle_clearance=BUFFERS[(index // 8) % len(BUFFERS)],
        )
        proposals.append(
            {
                "scene": asdict(scene),
                "mapping": mapping,
                "family": "single",
                "proposal": "cached_accepted" if accepted else "ungated_cached_rejected",
                "parent": {k: row[k] for k in ("family", "case", "anchor", "index", "rejection")},
            }
        )
    return proposals


def random_scene(
    rng: np.random.Generator, family: str, seed: int, buffer: float
) -> HoverEncounterConfig:
    direction = rng.normal(size=3)
    direction[2] *= 0.7
    direction /= np.linalg.norm(direction)
    arrival = float(rng.uniform(4.2, 7.0))
    offset = rng.uniform(-0.55, 0.55, 3)
    offset[2] *= 0.65
    sphere = IncomingSphere(
        arrival,
        tuple(direction),
        float(rng.uniform(1.2, 3.6)),
        float(rng.uniform(0.3, 0.85)),
        tuple(offset),
        30,
    )
    wind = rng.normal(size=3)
    wind[2] *= 0.2
    wind *= rng.uniform(1.4, 3.2) / np.linalg.norm(wind)
    onset = float(round(rng.uniform(1.8, 3.2) / 0.04) * 0.04)
    scene = HoverEncounterConfig(
        incoming=sphere,
        seed=seed,
        wind_onset_seconds=onset,
        wind_velocity=tuple(wind),
        obstacle_clearance=buffer,
        navigation_start_seconds=float(np.ceil((arrival + 1.1) / 0.04) * 0.04),
        duration_seconds=14.0,
    )
    if family == "guards":
        guards = []
        for _ in range(2):
            radial = rng.normal(size=3)
            radial[2] = rng.uniform(-0.2, 1.2)
            radial /= np.linalg.norm(radial)
            radius = float(rng.uniform(0.22, 0.5))
            distance = radius + scene.ego_radius + buffer + rng.uniform(0.10, 0.65)
            guards.append(GuardSphere(tuple(radial * distance), radius))
        scene = replace(scene, guards=tuple(guards))
    elif family == "staggered":
        second_direction = rng.normal(size=3)
        second_direction /= np.linalg.norm(second_direction)
        second = IncomingSphere(
            arrival + float(rng.uniform(0.2, 0.9)),
            tuple(second_direction),
            float(rng.uniform(1.2, 3.6)),
            float(rng.uniform(0.3, 0.75)),
            tuple(rng.uniform(-0.55, 0.55, 3)),
            30,
        )
        scene = replace(
            scene,
            additional_incoming=(second,),
            navigation_start_seconds=float(
                np.ceil((second.arrival_time_seconds + 1.1) / 0.04) * 0.04
            ),
        )
    elif family == "moving":
        heading = rng.uniform(-np.pi, np.pi)
        velocity = np.asarray(
            (np.cos(heading), np.sin(heading), rng.uniform(-0.2, 0.3))
        ) * rng.uniform(0.3, 1.1)
        target = np.asarray((np.cos(heading), np.sin(heading), rng.uniform(-0.1, 0.5))) * 1.8
        crossing = target * rng.uniform(0.5, 1.0) + offset * 0.5
        # Navigation is active before the crossing, with a second leg extending beyond it.
        scene = replace(
            scene,
            initial_velocity=tuple(velocity),
            navigation_start_seconds=2.0,
            incoming=replace(sphere, crossing_offset=tuple(crossing)),
            waypoint_offsets=(tuple(target), tuple(-target * 1.4 + np.asarray((0, 0, 0.3)))),
        )
    return scene


def mutate_scene(
    scene: HoverEncounterConfig, rng: np.random.Generator, *, scale: float = 1.0
) -> HoverEncounterConfig:
    sphere = scene.incoming
    direction = np.asarray(sphere.direction) + rng.normal(0, 0.12 * scale, 3)
    direction /= np.linalg.norm(direction)
    sphere = replace(
        sphere,
        direction=tuple(direction),
        crossing_offset=tuple(np.asarray(sphere.crossing_offset) + rng.normal(0, 0.13 * scale, 3)),
        arrival_time_seconds=float(
            np.clip(sphere.arrival_time_seconds + rng.normal(0, 0.25 * scale), 3.5, 9.7)
        ),
        radius_m=float(np.clip(sphere.radius_m + rng.normal(0, 0.07 * scale), 0.15, 1.1)),
        speed_m_s=float(np.clip(sphere.speed_m_s + rng.normal(0, 0.35 * scale), 0.8, 4.5)),
    )
    wind = np.asarray(scene.wind_velocity) + rng.normal(0, 0.25 * scale, 3)
    wind[2] = np.clip(wind[2], -0.8, 0.8)
    onset = (
        round(np.clip(scene.wind_onset_seconds + rng.normal(0, 0.2 * scale), 1.2, 3.5) / 0.04)
        * 0.04
    )
    guards = tuple(
        replace(g, offset=tuple(np.asarray(g.offset) + rng.normal(0, 0.06 * scale, 3)))
        for g in scene.guards
    )
    return replace(
        scene,
        incoming=sphere,
        guards=guards,
        wind_velocity=tuple(wind),
        wind_onset_seconds=float(onset),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, default="single")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=37103)
    parser.add_argument("--parents", type=Path)
    parser.add_argument("--device", default="gpu", choices=("cpu", "gpu"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scene", type=Path)
    parser.add_argument(
        "--mapping", choices=("uncompensated", "compensated"), default="uncompensated"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    rng = np.random.default_rng(args.seed)
    plans = args.output / "proposals.json"
    if plans.exists():
        proposals = json.loads(plans.read_text())
    else:
        proposals = []
        if args.scene:
            scene = HoverEncounterConfig.from_dict(json.loads(args.scene.read_text()))
            proposals = [
                {
                    "scene": asdict(scene),
                    "mapping": args.mapping,
                    "family": args.family,
                    "proposal": "explicit_scene",
                    "parent": str(args.scene),
                }
            ]
        elif args.parents:
            parents = [json.loads(line) for line in args.parents.read_text().splitlines()]
            parents = [r for r in parents if "objective" in r]
            by_mapping = {
                m: sorted((r for r in parents if r["mapping"] == m), key=lambda r: r["objective"])[
                    :12
                ]
                for m in ("uncompensated", "compensated")
            }
            for i in range(args.count):
                mapping = ("uncompensated", "compensated")[i % 2]
                if i % 5 == 0 or not by_mapping[mapping]:
                    scene = random_scene(rng, args.family, args.seed + i, BUFFERS[(i // 2) % 4])
                    parent, label = None, "ungated_refinement_exploration"
                else:
                    p = by_mapping[mapping][int(rng.integers(len(by_mapping[mapping])))]
                    scene = mutate_scene(HoverEncounterConfig.from_dict(p["scene"]), rng)
                    parent, label = p["trial_id"], "outcome_guided_mutation"
                proposals.append(
                    {
                        "scene": asdict(scene),
                        "mapping": mapping,
                        "family": args.family,
                        "parent": parent,
                        "proposal": label,
                    }
                )
        elif args.family == "single":
            proposals = existing_proposals(args.seed, args.count)
        else:
            for i in range(args.count):
                scene = random_scene(rng, args.family, args.seed + i, BUFFERS[(i // 2) % 4])
                proposals.append(
                    {
                        "scene": asdict(scene),
                        "mapping": ("uncompensated", "compensated")[i % 2],
                        "family": args.family,
                        "parent": None,
                        "proposal": "ungated_stratified_scene",
                    }
                )
        write_json(plans, proposals)
    ledger = args.output / "trials.jsonl"
    prior = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    done = {r["trial_id"] for r in prior}
    write_json(
        args.output / "protocol.json",
        {
            "base_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_hashes": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    Path("crazyflow/safety/da_plcbf/navigation_experiment.py"),
                    Path("crazyflow/safety/da_plcbf/case_study_world.py"),
                )
            },
            "seed": args.seed,
            "planned_full_pairs": len(proposals),
            "buffer_sweep_m": BUFFERS,
            "fixed_penetration_threshold_m": -0.002,
            "adaptive_survival_threshold_m": 0.01,
            "same_controller_and_physical_limits": True,
            "analytic_obstacle_hocbf": False,
            "publication": "every finite update becomes available at next control boundary",
            "start": "both full physical episodes from time zero with all obstacles visible",
            "no_initial_H_gate": True,
            "discovery_only": True,
            "geometry_scope": "recorded-state interpolation; fine-plant confirmation required",
        },
    )
    engine = EpisodeEvaluator(jax.devices(args.device)[0])
    outcomes = Counter(r.get("outcome_class", "invalid_scene") for r in prior)
    for index, proposal in enumerate(proposals):
        trial_id = f"{args.output.name}-{index:04d}"
        if trial_id in done:
            continue
        started = time.perf_counter()
        scene = HoverEncounterConfig.from_dict(proposal["scene"])
        record = {**proposal, "trial_id": trial_id, "seed": args.seed, "index": index}
        try:
            build_hover_encounter_world(scene)
        except ValueError as error:
            record.update(outcome_class="invalid_scene", invalid_reason=str(error))
        else:
            result = engine.evaluate(scene, proposal["mapping"], args.output / trial_id)
            record.update(result)
        record["trial_wall_seconds"] = time.perf_counter() - started
        with ledger.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
        outcomes[record["outcome_class"]] += 1
        print(
            json.dumps(
                {
                    "trial": trial_id,
                    "outcome": record["outcome_class"],
                    "objective": record.get("objective"),
                    "promote": record.get("promotion_candidate"),
                    "wall_seconds": record["trial_wall_seconds"],
                    "counts": dict(outcomes),
                }
            ),
            flush=True,
        )
    write_json(
        args.output / "COUNTS.json",
        {
            "outcomes": dict(outcomes),
            "proposals": len(proposals),
            "executed_full_pairs": sum(outcomes.values()) - outcomes["invalid_scene"],
        },
    )


if __name__ == "__main__":
    main()
