"""Replay a confirmed full episode, with an optional separately simulated contact fall.

``prepare-contact`` uses existing MuJoCo rigid-body contact physics only after a recorded
modeled-sphere intersection. ``render`` reads saved flight/contact poses and never reruns
control, learning, or contact dynamics. Videos remain local artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from crazyflow.safety.da_plcbf.case_study_world import (
    HoverEncounterConfig,
    audit_recorded_collider_clearance,
    build_hover_encounter_world,
)
from crazyflow.safety.da_plcbf.contact_replay import (
    ContactBody,
    ContactReplayConfig,
    ObstacleMotion,
    cf21b_contact_body,
    find_contact_trigger,
    run_contact_replay,
    save_contact_replay,
)

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.mujoco_comparison_video import ComparisonVideoTrace


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_contact(run: Path, output: Path) -> Path:
    """Use physical intersection as the sole trigger, retaining the scene's analytic clock."""
    from scipy.spatial.transform import Rotation, Slerp

    if output.exists():
        raise FileExistsError("refusing to overwrite contact evidence")
    scene = HoverEncounterConfig.from_dict(json.loads((run / "encounter.json").read_text()))
    world = build_hover_encounter_world(scene)
    with np.load(run / "dense_plant_states.npz") as source:
        states = source["fixed"]
    times = np.arange(len(states)) * scene.dt
    audit = audit_recorded_collider_clearance(world, times, states)
    collider = audit["actual_xml_sphere_geometry"]
    if collider["minimum_clearance_upper_bound_m"] >= 0:
        raise ValueError(
            "contact presentation requires a definite recorded XML-sphere intersection"
        )
    geometric_time = collider["first_chord_intersection_time_seconds"]
    if geometric_time is None:
        raise ValueError("no physical sphere crossing was recorded")
    timestep = 0.001
    support = np.arange(math.ceil((scene.duration_seconds + 0.02) / timestep) + 1) * timestep
    motion = ObstacleMotion(support, world.obstacle_kinematics(support)[0], world.obstacle_radii)
    fine_times = np.unique(
        np.r_[np.arange(math.floor(times[-1] / timestep) + 1) * timestep, times[-1]]
    )
    fine = np.column_stack([np.interp(fine_times, times, states[:, i]) for i in range(13)])
    fine[:, 3:7] = Slerp(times, Rotation.from_quat(states[:, 3:7]))(fine_times).as_quat()
    trigger = find_contact_trigger(
        fine_times, fine, motion, ContactReplayConfig(), kind="physical_contact"
    )
    event_error = abs(trigger.details["source_swept_event_time_seconds"] - geometric_time)
    if event_error > 0.002:
        raise ValueError("contact handoff disagrees with the independently audited physical event")
    config = ContactReplayConfig(
        duration_seconds=scene.duration_seconds - trigger.time_seconds, timestep=timestep
    )
    base = cf21b_contact_body()
    index = min(int(trigger.time_seconds / world.config.control_period), len(times) - 1)
    with np.load(run / "raw_diagnostics.npz") as raw:
        body = ContactBody(
            float(raw["fixed_actual_mass"][index]),
            raw["fixed_actual_inertia"][index],
            base.gravity,
            base.drag_matrix_body,
        )
    replay_times = (
        trigger.time_seconds
        + np.arange(math.ceil(config.duration_seconds / timestep) + 1) * timestep
    )
    wind = np.array([world.wind_at(t) for t in replay_times])
    replay = run_contact_replay(trigger, body, motion, config, wind_velocity_world=wind)
    if not replay.metadata["obstacle_contact_steps"]:
        raise ValueError("MuJoCo measured no obstacle contact; do not present a forced impact")
    if not replay.metadata["ground_contact_steps"]:
        raise ValueError("MuJoCo continuation did not reach the ground within the recorded scene")
    replay.metadata["source"] = {
        "directory": str(run.resolve()),
        "method": "fixed",
        "input_sha256": {
            name: _sha(run / name)
            for name in (
                "encounter.json",
                "navigation_comparison.npz",
                "navigation_comparison.json",
                "dense_plant_states.npz",
                "raw_diagnostics.npz",
            )
        },
        "audited_xml_intersection_seconds": geometric_time,
        "handoff_event_disagreement_seconds": event_error,
        "collision_trigger_scope": (
            "physical XML sphere only; neither clearance shell nor degradation"
        ),
    }
    replay.metadata["obstacle_motion"] = (
        "Original independently prescribed absolute-time sinusoids sampled at 1 ms; "
        "both obstacle positions and derivatives of that interpolant drive MuJoCo surfaces."
    )
    replay.metadata["presentation_source_sha256"] = _sha(Path(__file__))
    return save_contact_replay(replay, output)


def splice_contact_replay(
    trace: ComparisonVideoTrace, replay: dict[str, np.ndarray], metadata: dict
) -> tuple[ComparisonVideoTrace, dict]:
    """Display saved fixed-body contact poses; keep flight telemetry before handoff unchanged."""
    trigger = metadata["trigger"]
    if trigger["kind"] != "swept_collider_contact":
        raise ValueError(
            "only a physical collider trigger can be spliced into this collision video"
        )
    if not metadata.get("obstacle_contact_steps", 0):
        raise ValueError("a measured MuJoCo obstacle contact is required")
    when = float(trigger["time_seconds"])
    times = np.asarray(replay["time_seconds"], dtype=float)
    states = np.asarray(replay["full_state"], dtype=float)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("saved contact times must increase strictly")
    if abs(times[0] - when) > 1e-9 or states.shape != (len(times), 13):
        raise ValueError("saved contact states must start at the declared handoff")
    if not np.isfinite(states).all() or not np.isfinite(times).all():
        raise ValueError("saved contact states and times must be finite")
    if not trace.time_seconds[0] < when < trace.time_seconds[-1]:
        raise ValueError("contact handoff must lie inside the recorded flight interval")
    contact = trace.time_seconds >= when - 1e-10
    queries = trace.time_seconds[contact]
    if queries[-1] > times[-1] + 1e-9:
        raise ValueError("contact replay must cover the entire displayed continuation")
    right = np.clip(np.searchsorted(times, queries), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    chosen = np.where(abs(times[left] - queries) <= abs(times[right] - queries), left, right)
    fixed = trace.fixed
    if fixed.full_state is None or fixed.recorded_control_valid is None:
        raise ValueError("canonical flight full states and actual control masks are required")
    # Preserve saved contact precision while embedding the numerically identical flight prefix.
    full_state = fixed.full_state.astype(np.result_type(fixed.full_state.dtype, states.dtype))
    full_state[contact] = states[chosen]
    updates = {
        "full_state": full_state,
        "position": full_state[:, :3],
        "quaternion_xyzw": full_state[:, 3:7],
        "contact_replay": contact,
    }
    for name in (
        "recorded_control_valid",
        "qp_valid",
        "used_fallback",
        "used_emergency",
        "used_midpoint",
        "degraded",
        "collision_constraint_active",
        "missed_deadline",
    ):
        value = getattr(fixed, name)
        if value is not None:
            updated = value.copy()
            updated[contact] = False
            updates[name] = updated
    for name in (
        "applied_wrench",
        "nominal_wrench",
        "intervention_world",
        "intervention_norm",
        "selected_policy_dual",
        "executed_policy_dual",
        "gradient_norm",
        "parameter_update_norm",
        "controller_seconds",
        "learner_seconds",
    ):
        value = getattr(fixed, name)
        if value is not None:
            updated = value.copy()
            updated[contact] = 0
            updates[name] = updated
    previous = int(np.flatnonzero(~contact)[-1])
    for name in ("library_version", "cumulative_gradient_steps", "waypoint_index"):
        value = getattr(fixed, name)
        if value is not None:
            updated = value.copy()
            updated[contact] = updated[previous]
            updates[name] = updated
    collision = np.zeros(len(trace.time_seconds), dtype=bool)
    if fixed.physical_collision_recorded is not None:
        collision[:] = fixed.physical_collision_recorded
    collision[trace.time_seconds >= float(trigger.get("source_swept_event_time_seconds", when))] = (
        True
    )
    updates["physical_collision_recorded"] = collision
    presented = replace(trace, fixed=replace(fixed, **updates))
    presented.validate()
    return presented, {
        "scope": (
            "saved physical-contact continuation only; original fixed control stops at handoff"
        ),
        "trigger": trigger,
        "source_sample_indices": chosen.tolist(),
        "maximum_time_quantization_seconds": float(np.max(abs(times[chosen] - queries))),
        "adaptive_trace_object_unchanged": presented.adaptive is trace.adaptive,
        "fixed_precontact_states_exact": bool(
            np.array_equal(presented.fixed.full_state[~contact], fixed.full_state[~contact])
        ),
        "no_control_or_waypoint_credit_after_contact": True,
    }


def render(run: Path, output: Path, contact_directory: Path | None, azimuth: float) -> Path:
    """Render one fresh EGL process from saved canonical arrays and measured contact poses."""
    import crazyflow.safety.da_plcbf.mujoco_comparison_video as renderer
    from crazyflow.safety.da_plcbf.online_constant_wind import load_online_constant_wind_result

    if output.exists():
        raise FileExistsError("refusing to overwrite video evidence")
    paths = [
        run / "navigation_comparison.npz",
        run / "navigation_comparison.json",
        run / "CASE_EPISODE_SUMMARY.json",
    ]
    hashes = {str(p.resolve()): _sha(p) for p in paths}
    loaded = load_online_constant_wind_result(paths[0], paths[1])
    trace = loaded.trace
    report = json.loads(paths[2].read_text())
    scene = HoverEncounterConfig.from_dict(report["encounter"])
    if report["termination_geometry"] != "modeled_collider":
        raise ValueError("this presentation requires actual modeled-collider termination")
    if (
        report["methods"]["fixed"]["termination"] != "physical_collision"
        or report["methods"]["adaptive"]["termination"] != "completed"
        or len(scene.additional_incoming) != 1
    ):
        raise ValueError(
            "this caption protocol requires a confirmed two-mover collision/survival case"
        )
    splice = None
    if contact_directory:
        meta_path = contact_directory / "contact_replay.json"
        metadata = json.loads(meta_path.read_text())
        source = metadata["source"]
        if Path(source["directory"]).resolve() != run.resolve():
            raise ValueError("contact replay belongs to a different flight run")
        for name, expected in source["input_sha256"].items():
            if _sha(run / name) != expected:
                raise ValueError(f"contact source checksum mismatch: {name}")
        raw_path = contact_directory / "contact_replay.npz"
        if _sha(raw_path) != metadata["npz_sha256"]:
            raise ValueError("contact replay checksum mismatch")
        with np.load(raw_path) as raw:
            trace, splice = splice_contact_replay(trace, dict(raw), metadata)
        for path in (meta_path, raw_path, contact_directory / "contact_model.xml"):
            hashes[str(path.resolve())] = _sha(path)
    fixed_event = report["methods"]["fixed"]["xml_sphere_geometric_intersection"][
        "first_chord_intersection_time_seconds"
    ]
    completion = report["methods"]["adaptive"]["termination_time_seconds"]
    captions = []
    for when in trace.time_seconds:
        if when < scene.wind_onset_seconds:
            caption = (
                "Calm hover · same starting library · right learns from its own observed motion"
            )
        elif fixed_event is None or when < fixed_event:
            caption = (
                "Wind stays on · both see both movers and the same dynamics · "
                "right updates its skills"
            )
        elif when < scene.navigation_start_seconds:
            caption = (
                f"Fixed collided at {fixed_event:.3f} s · "
                "adapted is still avoiding the incoming obstacles"
            )
        elif completion is not None and when >= completion:
            caption = (
                "Adapted completed both waypoints · "
                "fixed task ended at its first physical collision"
            )
        else:
            caption = (
                "Encounter passed · adapted follows the waypoint route · "
                "fixed has no further control"
            )
        captions.append(caption)
    trace = replace(
        trace,
        title="Two incoming obstacles in persistent wind | fixed fallbacks vs online adaptation",
        left_label="FIXED FALLBACK LIBRARY",
        right_label="CONTINUOUS ONLINE ADAPTATION",
        phase_caption=np.asarray(captions),
    )
    config = renderer.ComparisonRenderConfig(
        mode="demo",
        width=1600,
        height=900,
        fps=20,
        camera_azimuth=azimuth,
        hover_camera_distance=3.5,
        camera_distance=5.2,
        freeze_after_termination=contact_directory is None,
        comparison_note=(
            f"Same physical spheres +{100 * scene.obstacle_clearance:g} cm optional clearance · "
            + (
                "after contact: separate MuJoCo motor-off continuation"
                if contact_directory
                else "recorded collision ends the fixed task"
            )
        ),
    )
    output.mkdir(parents=True)
    original = renderer._render_world
    checks = []

    def checked(*arguments: Any, **kwargs: Any) -> np.ndarray:
        frame = original(*arguments, **kwargs)
        mean = float(frame.mean())
        checks.append(
            {"index": int(arguments[3]), "world": int(arguments[4]), "scene_mean_rgb": mean}
        )
        if mean < 1:
            raise RuntimeError("black rendered scene panel; revise camera before publication")
        return frame

    renderer._render_world = checked
    try:
        video = renderer.render_comparison_video(trace, output / "comparison.mp4", config)
    finally:
        renderer._render_world = original
    if hashes != {name: _sha(Path(name)) for name in hashes}:
        raise ValueError("render inputs changed during presentation")
    metadata = {
        "scope": __doc__,
        "source_sha256": hashes,
        "splice": splice,
        "render_config": asdict(config),
        "frame_count": video.frame_count,
        "duration_seconds": video.frame_count / video.fps,
        "video_sha256": _sha(video.path),
        "source_time_support_seconds": [
            float(trace.time_seconds[0]),
            float(trace.time_seconds[-1]),
        ],
        "source_execution_mode": report["execution_mode"],
        "actual_collider_event_seconds": fixed_event,
        "adaptive_completion_seconds": completion,
        "source_finite_updates": report["methods"]["adaptive"]["finite_updates"],
        "render_source_sha256": {
            str(Path(__file__)): _sha(Path(__file__)),
            str(Path(renderer.__file__)): _sha(Path(renderer.__file__)),
        },
        "scene_checks": checks,
        "git_policy": "local video only; do not stage",
    }
    (output / "RENDER_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"video": str(video.path.resolve()), "frames": video.frame_count}), flush=True)
    return video.path


def main() -> None:
    """Keep contact physics and saved-pose rendering as explicit separate commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-contact", "render"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact", type=Path)
    parser.add_argument("--camera-azimuth", type=float, default=-45)
    args = parser.parse_args()
    if args.command == "prepare-contact":
        print(prepare_contact(args.run, args.output), flush=True)
    else:
        render(args.run, args.output, args.contact, args.camera_azimuth)


if __name__ == "__main__":
    main()
