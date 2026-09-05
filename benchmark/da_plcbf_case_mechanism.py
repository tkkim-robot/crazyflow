"""Diagnose why a cached coverage contrast does or does not change closed-loop survival.

This reads immutable atlas rollouts and completed branch logs. The initial skill's saved
rollout advances its phase throughout the horizon; deployed fallback decisions restart their
rollout at every control call and may switch to QP. Neither trajectory is assumed to equal the
other beyond the first held command. Geometry intersection remains distinct from MuJoCo contact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from crazyflow.safety.da_plcbf.case_study_world import (
    HoverEncounterConfig,
    audit_recorded_collider_clearance,
    build_hover_encounter_world,
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact_geometry(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        kind: {
            field: audit[kind][field]
            for field in (
                "minimum_clearance_m",
                "minimum_clearance_lower_bound_m",
                "minimum_clearance_upper_bound_m",
                "first_chord_intersection_time_seconds",
            )
        }
        for kind in ("safety_shell", "body_origin_envelope", "actual_xml_sphere_geometry")
    }


def analyze_case(directory: Path, atlas: Path) -> dict[str, Any]:
    """Compare first committed candidates, shared brakes and actual commanded prefixes."""
    selection = _read(directory / "selection.json")
    configuration = HoverEncounterConfig.from_dict(_read(directory / "encounter.json"))
    source = atlas / selection["case"] / "atlas.npz"
    with np.load(source) as data:
        arrays = {name: data[name] for name in data.files}
    with np.load(directory / "dense_states.npz") as data:
        dense = {name: data[name] for name in data.files}
    anchor = selection["anchor"]
    state = arrays[f"{anchor}_state"]
    world = build_hover_encounter_world(
        configuration, initial_state=state, initial_time_seconds=selection["time_seconds"]
    )
    horizon = arrays[f"{anchor}_nominal"].shape[1] - 1
    dt = world.config.dt
    times = selection["time_seconds"] + dt * np.arange(horizon + 1)
    result: dict[str, Any] = {
        "selection": selection,
        "source_hashes": {
            str(path): _hash(path)
            for path in (
                source,
                directory / "selection.json",
                directory / "encounter.json",
                directory / "dense_states.npz",
                directory / "fixed_controls.json",
                directory / "adaptive_held_controls.json",
            )
        },
        "shared_committed_behaviors": {},
        "methods": {},
    }
    for name in ("nominal", "emergency", "stationary"):
        states = (
            np.broadcast_to(state, (horizon + 1, 13))
            if name == "stationary"
            else arrays[f"{anchor}_{name}"][0]
        )
        result["shared_committed_behaviors"][name] = _compact_geometry(
            audit_recorded_collider_clearance(world, times, states)
        )
    summary = _read(directory / "summary.json")
    for method, atlas_method in (("fixed", "fixed"), ("adaptive_held", "adaptive")):
        rows = [row for row in _read(directory / f"{method}_controls.json") if row["executed"]]
        np.testing.assert_array_equal(rows[0]["state"], state)
        policy = rows[0]["selected"]
        states = (
            arrays[f"{anchor}_nominal"][0]
            if policy == 0
            else arrays[f"{anchor}_{atlas_method}"][policy - 1]
        )
        committed = _compact_geometry(audit_recorded_collider_clearance(world, times, states))
        actual_times = dense[f"{method}_times"]
        actual_states = dense[f"{method}_states"]
        overlap = actual_times <= times[-1] + 1e-9
        actual_prefix = _compact_geometry(
            audit_recorded_collider_clearance(world, actual_times[overlap], actual_states[overlap])
        )
        first_qp = next((row["time"] for row in rows if row["qp"]), None)
        first_certificate = next((row["time"] for row in rows if any(row["eligible"])), None)
        first_positive_dual = next(
            (
                row["time"]
                for row in rows
                if row["qp"] and row["selected"] > 0 and row["dual"] > 1e-7
            ),
            None,
        )
        prefix = []
        wrenches = None if policy == 0 else arrays[f"{anchor}_{atlas_method}_wrenches"][policy - 1]
        first_action_difference = None
        for row in rows:
            relative = row["time"] - selection["time_seconds"]
            if relative > min(0.6, horizon * dt) + 1e-9:
                break
            step = round(relative / dt)
            delta = None
            if wrenches is not None and step < len(wrenches):
                delta = np.asarray(row["action"]) - wrenches[step]
                if first_action_difference is None and np.max(np.abs(delta)) > 2e-6:
                    first_action_difference = row["time"]
            prefix.append(
                {
                    "time_seconds": row["time"],
                    "selected_policy": row["selected"],
                    "execution_mode": ("qp", "fallback", "emergency", "midpoint")[row["mode"]],
                    "degraded": row["degraded"],
                    "max_hard_m2": max(row["hard"]),
                    "executed_dual": row["dual"],
                    "actual_minus_committed_initial_skill_wrench": None
                    if delta is None
                    else delta.tolist(),
                }
            )
        policy_geometry = []
        for index, trajectory in enumerate(arrays[f"{anchor}_{atlas_method}"]):
            geometry = _compact_geometry(
                audit_recorded_collider_clearance(world, times, trajectory)
            )
            policy_geometry.append({"fallback_skill_index": index, **geometry})
        result["methods"][method] = {
            "initial_selected_policy": policy,
            "initial_max_hard_m2": max(rows[0]["hard"]),
            "initial_execution_mode": ("qp", "fallback", "emergency", "midpoint")[rows[0]["mode"]],
            "first_eligible_certificate_time_seconds": first_certificate,
            "first_accepted_qp_time_seconds": first_qp,
            "first_learned_positive_dual_time_seconds": first_positive_dual,
            "first_command_differs_from_committed_initial_skill_time_seconds": (
                first_action_difference
            ),
            "first_held_interval_position_max_abs_difference_m": float(
                np.max(
                    np.abs(
                        actual_states[: world.config.control_interval_steps + 1, :3]
                        - states[: world.config.control_interval_steps + 1, :3]
                    )
                )
            ),
            "first_held_interval_state_max_abs_difference": float(
                np.max(
                    np.abs(
                        actual_states[: world.config.control_interval_steps + 1]
                        - states[: world.config.control_interval_steps + 1]
                    )
                )
            ),
            "actual_prefix_geometry": actual_prefix,
            "committed_initial_skill_geometry": committed,
            "committed_fallback_geometry": policy_geometry,
            "committed_fallback_xml_clear_count": sum(
                g["actual_xml_sphere_geometry"]["minimum_clearance_lower_bound_m"] > 0
                for g in policy_geometry
            ),
            "summary": {
                field: summary[method][field]
                for field in (
                    "termination",
                    "degraded_controls",
                    "fallback_controls",
                    "emergency_controls",
                    "qp_controls",
                    "learned_constraint_positive_dual_controls",
                )
            },
            "initial_control_chronology": prefix,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed-loop", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "schema": "recorded_case_mechanism_audit_v1",
                "source_sha256": _hash(Path(__file__)),
                "closed_loop_directory": str(args.closed_loop),
                "atlas_directory": str(args.atlas),
                "scope": "post hoc mechanism diagnosis of selected development cases",
                "committed_trajectory": (
                    "saved obstacle-independent initial skill rollout; phase advances through "
                    "the horizon, whereas runtime may restart or switch at later boundaries"
                ),
                "contact_scope": "oriented XML sphere geometry; no measured MuJoCo contact claim",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    cases = []
    for directory in sorted(args.closed_loop.iterdir()):
        if not directory.is_dir() or not (directory / "summary.json").exists():
            continue
        result = analyze_case(directory, args.atlas)
        (args.output_dir / f"{directory.name}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        fixed, adaptive = result["methods"]["fixed"], result["methods"]["adaptive_held"]
        compact = {
            "case": directory.name,
            "fixed_degraded": fixed["summary"]["degraded_controls"],
            "adaptive_degraded": adaptive["summary"]["degraded_controls"],
            "fixed_committed_initial_shell_clearance": fixed["committed_initial_skill_geometry"][
                "safety_shell"
            ]["minimum_clearance_m"],
            "fixed_committed_initial_xml_clearance": fixed["committed_initial_skill_geometry"][
                "actual_xml_sphere_geometry"
            ]["minimum_clearance_m"],
            "fixed_committed_xml_clear_count": fixed["committed_fallback_xml_clear_count"],
            "stationary_xml_clearance": result["shared_committed_behaviors"]["stationary"][
                "actual_xml_sphere_geometry"
            ]["minimum_clearance_m"],
            "emergency_xml_clearance": result["shared_committed_behaviors"]["emergency"][
                "actual_xml_sphere_geometry"
            ]["minimum_clearance_m"],
            "fixed_first_qp": fixed["first_accepted_qp_time_seconds"],
            "fixed_first_command_divergence": fixed[
                "first_command_differs_from_committed_initial_skill_time_seconds"
            ],
            "fixed_first_hold_state_error": fixed["first_held_interval_state_max_abs_difference"],
        }
        cases.append(compact)
        print(json.dumps(compact), flush=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(cases, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
