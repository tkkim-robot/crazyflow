"""Propose prescribed scenes from actual paired escapes, then require full reruns.

The saved paths are development data for obstacle placement. They are never actor inputs,
and auditing a new obstacle against those old paths is only a proposal diagnostic: adding
that obstacle can change both controllers from time zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, replace
from itertools import zip_longest
from pathlib import Path

import numpy as np

from benchmark.da_plcbf_closed_loop_search import mutate_scene
from crazyflow.safety.da_plcbf.case_study_world import (
    CF21B_XML_COLLIDER_OFFSET_BODY_M,
    CF21B_XML_COLLIDER_RADIUS_M,
    GuardSphere,
    HoverEncounterConfig,
    IncomingSphere,
    audit_recorded_collider_clearance,
    build_hover_encounter_world,
)


def _scene_identity(scene: dict, mapping: str) -> tuple[str, str]:
    physical = {key: value for key, value in scene.items() if key != "seed"}
    return mapping, json.dumps(physical, sort_keys=True)


def _path(traces: dict[str, np.ndarray], method: str) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(traces[f"{method}_dense_times"], dtype=float)
    states = np.asarray(traces[f"{method}_dense_states"], dtype=float)
    if times.ndim != 1 or len(times) < 2 or states.shape != (len(times), 13):
        raise ValueError("refinement requires dense time and 13-state arrays for both methods")
    if not np.isfinite(times).all() or not np.isfinite(states).all() or np.any(np.diff(times) <= 0):
        raise ValueError("refinement paths must be finite with strictly increasing times")
    if abs(times[0]) > 1e-9:
        raise ValueError("refinement parents must be full episodes starting at time zero")
    return times, states


def _collider_centers(states: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    if np.any(np.abs(np.linalg.norm(states[:, 3:7], axis=1) - 1) > 2e-4):
        raise ValueError("refinement path quaternions must be normalized")
    return states[:, :3] + Rotation.from_quat(states[:, 3:7]).apply(
        CF21B_XML_COLLIDER_OFFSET_BODY_M
    )


def _encounter_indices(scene: HoverEncounterConfig, times: np.ndarray) -> np.ndarray:
    last = max(s.arrival_time_seconds for s in (scene.incoming, *scene.additional_incoming))
    return np.flatnonzero((times >= max(0, scene.wind_onset_seconds - 0.2)) & (times <= last + 0.8))


def escape_diagnostics(scene: HoverEncounterConfig, traces: dict[str, np.ndarray]) -> dict:
    """Classify the recorded full controller's movement and rescue modes near the encounter."""
    times, states = _path(traces, "fixed")
    indices = _encounter_indices(scene, times)
    if not len(indices):
        raise ValueError("parent trace ends before the encounter preparation window")
    displacement = states[indices, :3] - states[0, :3]
    speed = np.linalg.norm(states[indices, 7:10], axis=1)
    horizontal = np.linalg.norm(displacement[:, :2], axis=1)
    vertical = np.abs(displacement[:, 2])
    moved = (np.linalg.norm(displacement, axis=1) > 0.08) | (speed > 0.15)
    first_response = float(times[indices[np.flatnonzero(moved)[0]]]) if moved.any() else None
    first_arrival = min(
        s.arrival_time_seconds for s in (scene.incoming, *scene.additional_incoming)
    )
    labels = []
    if first_response is not None and first_response < first_arrival - 0.5:
        labels.append("early_recorded_motion")
    if horizontal.max() > 0.12:
        labels.append("lateral_escape_or_task_motion")
    if vertical.max() > 0.12:
        labels.append("vertical_escape_or_task_motion")
    control_times = np.asarray(traces.get("fixed_time", ()), dtype=float)
    active = (control_times >= times[indices[0]]) & (control_times <= times[indices[-1]])
    counts = {
        name: int(
            np.asarray(traces.get(f"fixed_{name}", np.zeros(len(control_times))))[active].sum()
        )
        for name in ("emergency", "fallback", "qp")
    }
    if counts["emergency"]:
        labels.append("emergency_control_executed")
    if counts["fallback"]:
        labels.append("fallback_prefix_executed")
    if scene.navigation_start_seconds < first_arrival:
        labels.append("waypoint_leg_active_before_encounter")
    return {
        "scope": "descriptive recorded motion; task motion alone is not attributed to avoidance",
        "labels": labels,
        "window_seconds": [float(times[indices[0]]), float(times[indices[-1]])],
        "first_recorded_response_seconds": first_response,
        "maximum_horizontal_displacement_m": float(horizontal.max()),
        "maximum_vertical_displacement_m": float(vertical.max()),
        "maximum_speed_m_s": float(speed.max()),
        "recorded_control_counts": counts,
    }


def _candidate_points(
    scene: HoverEncounterConfig, traces: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times, states = _path(traces, "fixed")
    adaptive_times, adaptive_states = _path(traces, "adaptive")
    fixed_centers, adaptive_centers = _collider_centers(states), _collider_centers(adaptive_states)
    indices = _encounter_indices(scene, times)
    indices = indices[
        (times[indices] > 0.2)
        & (times[indices] < scene.duration_seconds - 0.2)
        & (times[indices] <= adaptive_times[-1])
    ]
    if not len(indices):
        raise ValueError("no recorded encounter states can support a prescribed passage")
    # Keep time diversity; farthest old-adaptive-path distances guide static guard placement.
    sampled = indices[
        np.unique(np.linspace(0, len(indices) - 1, min(36, len(indices))).astype(int))
    ]
    return times, fixed_centers, adaptive_times, adaptive_centers, sampled


def propose_escape_scene(
    scene: HoverEncounterConfig,
    traces: dict[str, np.ndarray],
    strategy: str,
    rng: np.random.Generator,
) -> tuple[HoverEncounterConfig, dict]:
    """Fit one blocker or perturbation; old-path separation is never a rerun outcome."""
    if strategy not in {"guard", "mover", "local"}:
        raise ValueError("strategy must be guard, mover, or local")
    times, fixed, adaptive_times, adaptive, indices = _candidate_points(scene, traces)
    hover = np.asarray(scene.hover_position)
    candidates = []
    for index in indices:
        point, when = fixed[index], float(times[index])
        outward = point - fixed[0]
        outward /= max(np.linalg.norm(outward), 1e-12)
        if strategy == "guard":
            for shift in (0.0, 0.12, 0.24):
                center = point + shift * outward
                adaptive_distance = float(np.linalg.norm(adaptive - center, axis=1).min())
                initial_distance = float(np.linalg.norm(hover - center))
                radius = min(
                    float(rng.uniform(0.12, 0.30)),
                    adaptive_distance - CF21B_XML_COLLIDER_RADIUS_M - 0.02,
                    initial_distance - scene.ego_radius - scene.obstacle_clearance - 0.02,
                )
                if radius < 0.04:
                    continue
                proposed = replace(
                    scene, guards=(*scene.guards, GuardSphere(tuple(center - hover), radius))
                )
                fixed_margin = float(np.linalg.norm(fixed - center, axis=1).min()) - radius - 0.086
                adaptive_margin = adaptive_distance - radius - 0.086
                candidates.append((fixed_margin, -adaptive_margin, when, proposed))
        elif strategy == "mover":
            tangent = fixed[min(index + 1, len(fixed) - 1)] - fixed[max(0, index - 1)]
            for direction in (tangent, np.cross(outward, (0, 0, 1)), rng.normal(size=3)):
                norm = np.linalg.norm(direction)
                if norm < 1e-8:
                    continue
                direction = direction / norm
                speed = float(rng.uniform(1.2, 3.6))
                amplitude = 30.0
                centers = point + direction * amplitude * np.sin(
                    (adaptive_times - when)[:, None] * speed / amplitude
                )
                adaptive_distance = float(np.linalg.norm(adaptive - centers, axis=1).min())
                initial_center = point + direction * amplitude * np.sin(-when * speed / amplitude)
                radius = min(
                    float(rng.uniform(0.15, 0.45)),
                    adaptive_distance - CF21B_XML_COLLIDER_RADIUS_M - 0.02,
                    np.linalg.norm(hover - initial_center)
                    - scene.ego_radius
                    - scene.obstacle_clearance
                    - 0.02,
                )
                if radius < 0.04:
                    continue
                sphere = IncomingSphere(
                    when, tuple(direction), speed, float(radius), tuple(point - hover), amplitude
                )
                proposed = replace(scene, additional_incoming=(*scene.additional_incoming, sphere))
                candidates.append(
                    (-radius - 0.086, -(adaptive_distance - radius - 0.086), when, proposed)
                )
        else:
            # The closest recorded physical approach guides offsets, without an H-value gate.
            world = build_hover_encounter_world(scene)
            obstacle = world.obstacle_kinematics(when)[0][0]
            residual = point - obstacle
            proposed = mutate_scene(scene, rng, scale=0.65)
            sphere = replace(
                proposed.incoming,
                crossing_offset=tuple(
                    np.asarray(proposed.incoming.crossing_offset) + 0.35 * residual
                ),
            )
            candidates.append(
                (float(np.linalg.norm(residual)), 0.0, when, replace(proposed, incoming=sphere))
            )
    selected = None
    for _, _, when, proposed in sorted(candidates, key=lambda item: item[:2]):
        try:
            build_hover_encounter_world(proposed)
        except ValueError:
            continue
        selected = proposed, when
        break
    if selected is None:
        # An unseparable old route is useful evidence; retain a valid local proposal instead
        # of inventing a guard that starts in collision or silently dropping this parent.
        for _ in range(32):
            proposed = mutate_scene(scene, rng, scale=0.5)
            try:
                build_hover_encounter_world(proposed)
            except ValueError:
                continue
            selected = proposed, None
            break
    if selected is None:
        raise ValueError("no valid prescribed proposal found for this parent")
    proposed, target_time = selected
    world = build_hover_encounter_world(proposed)
    audits = {}
    for method in ("fixed", "adaptive"):
        path_times, states = _path(traces, method)
        audit = audit_recorded_collider_clearance(world, path_times, states)
        audits[method] = {
            "path_time_support_seconds": [float(path_times[0]), float(path_times[-1])],
            "source_node_count": len(path_times),
            **{
                key: audit[key]
                for key in ("actual_xml_sphere_geometry", "body_origin_envelope", "safety_shell")
            },
        }
    return proposed, {
        "scope": (
            "proposal only: new geometry evaluated on old recorded paths; not an executed "
            "closed-loop outcome, and both controllers must rerun from time zero"
        ),
        "requested_strategy": strategy,
        "used_local_fallback": target_time is None,
        "target_recorded_time_seconds": target_time,
        "escape_diagnostics": escape_diagnostics(scene, traces),
        "old_recorded_path_audits": audits,
    }


def select_parents(records: list[dict], limit: int) -> list[dict]:
    """Round-robin mapping/buffer/approach strata, ordered by actual paired outcome score."""
    if limit < 1:
        raise ValueError("parent limit must be positive")
    buckets = defaultdict(list)
    for record in records:
        if "objective" not in record or "scene" not in record:
            continue
        scene = record["scene"]
        direction = scene["incoming"]["direction"]
        key = (
            record["mapping"],
            scene["obstacle_clearance"],
            tuple(float(v) >= 0 for v in direction[:2]),
        )
        buckets[key].append(record)
    for rows in buckets.values():
        rows.sort(key=lambda row: (row["objective"], row["trial_id"]))
    # Priority within each round retains small true-collider margins without erasing strata.
    result, seen = [], set()
    while buckets and len(result) < limit:
        round_rows = defaultdict(list)
        for key in sorted(buckets, key=lambda k: buckets[k][0]["objective"]):
            round_rows[key[0]].append(buckets[key].pop(0))
        ordered = (
            row
            for group in zip_longest(*(round_rows[key] for key in sorted(round_rows)))
            for row in group
            if row is not None
        )
        for row in ordered:
            identity = _scene_identity(row["scene"], row["mapping"])
            if identity not in seen:
                seen.add(identity)
                result.append(row)
            if len(result) == limit:
                break
        buckets = {key: rows for key, rows in buckets.items() if rows}
    return result


def main() -> None:
    """Create proposals.json compatible with the full-episode driver's --resume mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--parent-limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=57301)
    parser.add_argument("--strategy", choices=("mixed", "guard", "mover", "local"), default="mixed")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    records = []
    for ledger in args.parents:
        for line in ledger.read_text().splitlines():
            record = json.loads(line)
            record["source_ledger"] = str(ledger)
            records.append(record)
    parents = select_parents(records, args.parent_limit)
    if not parents:
        parser.error("no completed controller outcomes are available")
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    proposals, diagnostics, identities = [], [], set()
    for index in range(args.count):
        parent = parents[index % len(parents)]
        directory = Path(parent["source_ledger"]).parent / parent["trial_id"]
        trace_path = directory / "traces.npz"
        with np.load(trace_path) as source:
            traces = dict(source)
        strategy = (
            ("guard", "mover", "local")[index % 3] if args.strategy == "mixed" else args.strategy
        )
        scene = HoverEncounterConfig.from_dict(parent["scene"])
        for _ in range(8):
            proposed, diagnostic = propose_escape_scene(scene, traces, strategy, rng)
            identity = _scene_identity(asdict(proposed), parent["mapping"])
            if identity not in identities:
                break
        if identity in identities:
            raise ValueError(
                "could not construct the requested number of distinct scene/mapping pairs"
            )
        identities.add(identity)
        lineage = {
            "trial_id": parent["trial_id"],
            "ledger": parent["source_ledger"],
            "trace_path": str(trace_path),
            "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "parent_scene_sha256": hashlib.sha256(
                json.dumps(parent["scene"], sort_keys=True).encode()
            ).hexdigest(),
        }
        proposals.append(
            {
                "scene": asdict(proposed),
                "mapping": parent["mapping"],
                "family": parent.get("family", "unknown"),
                "proposal": f"executed_escape_{strategy}",
                "parent": parent["trial_id"],
                "lineage": lineage,
            }
        )
        diagnostics.append({"proposal_index": index, "lineage": lineage, **diagnostic})
    protocol = {
        "seed": args.seed,
        "planned_distinct_scene_mapping_pairs": len(identities),
        "completed_parent_pairs": sum("objective" in row for row in records),
        "selected_parents": [row["trial_id"] for row in parents],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "local_mutation_source_sha256": hashlib.sha256(
            Path("benchmark/da_plcbf_closed_loop_search.py").read_bytes()
        ).hexdigest(),
        "scope": "development proposals only; no new full episode has been executed by this helper",
    }
    for name, value in (
        ("proposals.json", proposals),
        ("REFINEMENT_PROTOCOL.json", protocol),
        ("ESCAPE_PROPOSAL_DIAGNOSTICS.json", diagnostics),
    ):
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(protocol), flush=True)


if __name__ == "__main__":
    main()
