"""Run a continuous persistent-wind encounter through the production paired scheduler.

The complete episode begins at calm hover with obstacles already present on their prescribed
absolute-time trajectories. Saved summaries distinguish completed learner service, snapshots
actually used before arrival, conservative envelope breaches, and rotated XML geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import numpy as np

from crazyflow.safety.da_plcbf.case_study_world import (
    HoverEncounterConfig,
    audit_recorded_collider_clearance,
    build_hover_encounter_world,
)
from crazyflow.safety.da_plcbf.learner_checkpoint import load_learner_checkpoint
from crazyflow.safety.da_plcbf.navigation_experiment import (
    NavigationExperimentConfig,
    run_navigation_experiment,
    summarize_collision_observation,
)


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def snapshot_availability_at_arrival(
    summary: dict[str, Any], arrival_seconds: float
) -> dict[str, Any]:
    """Count snapshots used by real control inputs separately from completed learner calls."""
    initial_version = summary["initial_library_version"]
    output = {}
    paced = summary["execution_mode"] == "budgeted"
    for name, method in summary["methods"].items():
        inputs = method["publications_and_inputs"]
        preceding = [row for row in inputs if row["time"] <= arrival_seconds + 1e-10]
        last = preceding[-1] if preceding else None
        version_used = last["version_used"] if last else initial_version
        publications = [
            row
            for row in method["snapshot_publications"]
            if row["published_simulation_time"] <= arrival_seconds + 1e-10
        ]
        termination = method["termination_time_seconds"]
        output[name] = {
            "arrival_time_seconds": arrival_seconds,
            "initial_library_version": initial_version,
            "last_control_boundary_before_arrival_seconds": last["time"] if last else None,
            "version_used_at_last_boundary_before_arrival": version_used,
            "completed_updates_actually_used_before_arrival": version_used - initial_version,
            "versions_used_before_arrival": sorted({row["version_used"] for row in preceding}),
            "finite_update_calls_launched_before_arrival": sum(
                bool(row["finite"]) for row in inputs if row["time"] < arrival_seconds
            ),
            "finite_services_completed_before_arrival_wall_clock": (
                sum(
                    bool(row["finite"])
                    and row["learner_completed_wall_seconds"] is not None
                    and row["learner_completed_wall_seconds"] <= arrival_seconds
                    for row in inputs
                )
                if paced
                else None
            ),
            "actual_snapshot_publications_before_arrival": publications if paced else None,
            "every_publication_follows_service_completion": (
                all(
                    row["completed_wall_time"] <= row["published_wall_time"]
                    for row in method["snapshot_publications"]
                )
                if paced
                else None
            ),
            "episode_still_active_at_arrival": (
                termination is None or termination > arrival_seconds + 1e-10
            ),
            "deadline_misses_before_arrival": (
                sum(bool(row["missed_deadline"]) for row in preceding) if paced else None
            ),
            "timing_semantics": (
                "actual completed service and publication timestamps relative to method epoch"
                if paced
                else "deterministic completed snapshots appear at the next simulated boundary"
            ),
        }
    return output


def run_case_episode(
    encounter: HoverEncounterConfig,
    checkpoint: Path,
    output: Path,
    *,
    execution_mode: str = "deterministic",
    enable_learning: bool = True,
    termination_geometry: str = "modeled_collider",
    device: jax.Device,
) -> dict[str, Any]:
    """Run the unchanged paired experiment from time zero and audit only executed dense states."""
    world = build_hover_encounter_world(encounter, initial_time_seconds=0.0)
    bundle = load_learner_checkpoint(checkpoint, device=device)
    config = NavigationExperimentConfig(
        execution_mode=execution_mode,
        learning_start_seconds=0.0,
        update_every_controls=1,
        enable_learning=enable_learning,
        navigation_start_seconds=encounter.navigation_start_seconds,
        probe_every_controls=5,
        termination_geometry=termination_geometry,
        fallback_mapping="compensated"
        if bundle.config.model_compensation
        else "matched_uncompensated",
    )
    # Intentionally keep the actual scheduler's .003 s reserve and1.25 service multiplier.
    result = run_navigation_experiment(
        world,
        config,
        checkpoint,
        output,
        device=device,
        progress_callback=lambda method, index, count, reached: print(
            f"{method}: {index}/{count} controls; {reached} waypoints reached", flush=True
        ),
    )
    audits = {}
    with np.load(output / "dense_plant_states.npz", allow_pickle=False) as archive:
        for name in result.summary["methods"]:
            states = archive[name]
            times = np.arange(len(states)) * world.config.dt
            audits[name] = audit_recorded_collider_clearance(world, times, states)
    _write(output / "COLLIDER_GEOMETRY_AUDIT.json", audits)
    availability = snapshot_availability_at_arrival(
        result.summary, encounter.incoming.arrival_time_seconds
    )
    _write(output / "SNAPSHOT_AVAILABILITY_AT_ARRIVAL.json", availability)
    methods = {}
    for name, method in result.summary["methods"].items():
        methods[name] = {
            key: method[key]
            for key in (
                "termination",
                "termination_time_seconds",
                "termination_geometry",
                "collision_event",
                "waypoints_completed",
                "waypoints_total",
                "degraded_controls",
                "emergency_controls",
                "accepted_qp_controls",
                "finite_updates",
                "minimum_inflated_clearance_m",
                "minimum_physical_clearance_m",
                "controller_service",
                "learner_service",
                "service_exceeds_nominal_period_count",
                "execution_audit",
            )
        }
        methods[name].update(
            {
                "body_origin_envelope_breach_recorded": method[
                    "body_origin_enclosure_breach_recorded"
                ],
                "legacy_physical_collision_label_scope": (
                    "configured body-origin enclosure intersection"
                    if termination_geometry == "body_origin_enclosure"
                    else "definite modeled XML-sphere obstacle or floor intersection"
                ),
                "xml_sphere_geometric_intersection": audits[name]["actual_xml_sphere_geometry"],
                "xml_ground_geometric_intersection": audits[name]["actual_xml_ground_geometry"],
                **summarize_collision_observation(
                    audits[name],
                    termination_geometry=termination_geometry,
                    termination=method["termination"],
                ),
                "availability_at_arrival": availability[name],
            }
        )
    report = {
        "schema": "da_plcbf_continuous_persistent_wind_case_v2",
        "scope": (
            "continuous paired episode from shared calm hover; obstacles present from time zero"
        ),
        "encounter": asdict(encounter),
        "execution_mode": execution_mode,
        "enable_learning": enable_learning,
        "termination_geometry": termination_geometry,
        "checkpoint_npz_sha256": bundle.sha256,
        "controller_reserve_seconds": config.controller_reserve_seconds,
        "update_safety_factor": config.update_safety_factor,
        "command_period_seconds": world.config.control_period,
        "compensation_protocol": result.summary["compensation_protocol"],
        "runtime_feasibility": result.summary["runtime_feasibility"],
        "methods": methods,
        "positive_used_updates_with_zero_paced_deadline_misses": (
            execution_mode == "budgeted"
            and availability["adaptive"]["completed_updates_actually_used_before_arrival"] > 0
            and all(
                method["service_exceeds_nominal_period_count"] == 0 for method in methods.values()
            )
        ),
        "collision_scope": (
            "the flight runner retains commands after enclosure/shell breaches and terminates "
            "at the next control boundary after a definite XML-sphere obstacle/floor intersection; "
            "zero-straddling bounds remain unresolved; geometry is not measured MuJoCo contact"
            if termination_geometry == "modeled_collider"
            else "legacy enclosure termination censors later actual-collider outcomes; "
            "the retained-state XML-sphere audit is geometric and cannot infer later contact"
        ),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path("crazyflow/safety/da_plcbf/navigation_experiment.py"),
                Path("crazyflow/safety/da_plcbf/quad_rollouts.py"),
                Path("crazyflow/safety/da_plcbf/case_study_world.py"),
            )
        },
    }
    _write(output / "encounter.json", asdict(encounter))
    _write(output / "CASE_EPISODE_SUMMARY.json", report)
    return report


def main() -> None:
    """Load a typed selected encounter, execute it continuously, and optionally render locally."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encounter", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/da_plcbf/hover-explanation-20260905/learning/hover-probe-1/restoration_uncompensated_gated/initial_checkpoint"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execution-mode", choices=("deterministic", "budgeted"), default="deterministic"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--disable-learning", action="store_true")
    parser.add_argument(
        "--termination-geometry",
        choices=("modeled_collider", "body_origin_enclosure"),
        default="modeled_collider",
    )
    parser.add_argument("--unchanged-dynamics", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    data = json.loads(args.encounter.read_text())
    encounter = HoverEncounterConfig.from_dict(data.get("case_study_config", data))
    if args.unchanged_dynamics:
        from dataclasses import replace

        encounter = replace(encounter, wind_velocity=(0.0, 0.0, 0.0))
    report = run_case_episode(
        encounter,
        args.checkpoint,
        args.output,
        execution_mode=args.execution_mode,
        enable_learning=not args.disable_learning,
        termination_geometry=args.termination_geometry,
        device=jax.devices(args.device)[0],
    )
    print(
        json.dumps(
            {
                "directory": str(args.output),
                "methods": {
                    name: {
                        key: method[key]
                        for key in (
                            "termination",
                            "waypoints_completed",
                            "minimum_inflated_clearance_m",
                            "degraded_controls",
                            "finite_updates",
                        )
                    }
                    for name, method in report["methods"].items()
                },
                "positive_used_updates_with_zero_paced_deadline_misses": report[
                    "positive_used_updates_with_zero_paced_deadline_misses"
                ],
            }
        ),
        flush=True,
    )
    if args.render:
        from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
            ComparisonRenderConfig,
            render_comparison_video,
        )
        from crazyflow.safety.da_plcbf.online_constant_wind import load_online_constant_wind_result

        result = load_online_constant_wind_result(
            args.output / "navigation_comparison.npz", args.output / "navigation_comparison.json"
        )
        rendered = render_comparison_video(
            result.trace,
            args.output / "case_episode_demo.mp4",
            ComparisonRenderConfig(
                mode="demo",
                fps=args.fps,
                width=1600,
                height=900,
                comparison_note=(
                    "Persistent wind through the encounter · same model-aware nominal hover"
                ),
                hover_camera_distance=3.2,
            ),
        )
        print(
            json.dumps({"local_video": str(rendered.path), "frames": rendered.frame_count}),
            flush=True,
        )


if __name__ == "__main__":
    main()
