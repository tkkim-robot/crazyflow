"""Render a recorded approach and a separately simulated motors-off contact continuation.

The renderer never steps a controller, learner, or contact simulation. Every displayed vehicle
pose is selected from a saved source array; the exact handoff pose comes from contact_replay.npz.
The cf21B mesh is a visual asset, while contact_model.xml owns the archived collision geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import jax.numpy as jnp
import mujoco
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from crazyflow import Sim
from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
    _add_polyline,
    _install_marker_shadow_categories,
)
from crazyflow.sim.visualize import change_material


@dataclass(frozen=True)
class ContactPresentation:
    """Selected, unchanged recorded samples and their source indices."""

    requested_times: np.ndarray
    times: np.ndarray
    states: np.ndarray
    obstacle_centers: np.ndarray
    obstacle_radii: np.ndarray
    contact_phase: np.ndarray
    source_indices: np.ndarray
    metadata: dict
    inputs: dict[str, str]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nearest_indices(times: np.ndarray, requested: np.ndarray) -> np.ndarray:
    """Select recorded samples without extrapolation or pose interpolation."""
    if np.any(np.diff(times) <= 0) or np.any(~np.isfinite(times)):
        raise ValueError("source times must be finite and strictly increasing")
    if requested.size and (requested.min() < times[0] - 1e-9 or requested.max() > times[-1] + 1e-9):
        raise ValueError("requested frame lies outside its recorded source")
    right = np.minimum(np.searchsorted(times, requested), len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(times[left] - requested) <= abs(times[right] - requested), left, right)


def load_presentation(
    directory: Path, *, prelude_seconds: float = 1.5, fps: float = 20, playback_rate: float = 0.5
) -> ContactPresentation:
    """Verify input hashes and splice saved samples at the declared simulation handoff."""
    if not np.isfinite([prelude_seconds, fps, playback_rate]).all():
        raise ValueError("presentation timing must be finite")
    if prelude_seconds < 0 or min(fps, playback_rate) <= 0:
        raise ValueError("prelude must be nonnegative; fps and playback rate must be positive")
    metadata_path = directory / "contact_replay.json"
    metadata = json.loads(metadata_path.read_text())
    inputs = {str(metadata_path.resolve()): _sha256(metadata_path)}
    for name, key in (("contact_replay.npz", "npz_sha256"), ("contact_model.xml", "xml_sha256")):
        path = directory / name
        actual = _sha256(path)
        if actual != metadata[key]:
            raise ValueError(f"contact input checksum mismatch: {path}")
        inputs[str(path.resolve())] = actual
    with np.load(directory / "contact_replay.npz", allow_pickle=False) as raw:
        contact = {key: raw[key] for key in raw.files}
    trigger = float(metadata["trigger"]["time_seconds"])
    if abs(contact["time_seconds"][0] - trigger) > 1e-9:
        raise ValueError("contact recording must begin at the declared handoff")
    source = metadata["source"]
    source_directory = Path(source.get("replay_directory", source.get("directory", "")))
    for name, expected in source.get("input_sha256", {}).items():
        path = Path(name) if Path(name).is_absolute() else source_directory / name
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"approach input checksum mismatch: {path}")
        inputs[str(path.resolve())] = actual
    if "replay_directory" in source:
        replay = json.loads((source_directory / "replay.json").read_text())
        scenario_path = Path(replay["source_directory"]) / "feasibility_reference.json"
        scenario = json.loads(scenario_path.read_text())["scenario"]
        branch = source["branch"]
        with np.load(source_directory / "replay.npz", allow_pickle=False) as raw:
            states = raw[f"branch_{branch}_states"]
        start = replay["pre_failure_closed_loop_branches"][branch]["start_time"]
        source_times = start + np.arange(len(states)) * scenario["dt"]
        mask = np.asarray(scenario["obstacle_mask"], dtype=bool)
        centers = np.asarray(scenario["obstacle_initial_centers"])[mask]
        velocity = np.asarray(scenario["obstacle_velocities"])[mask]
        source_centers = centers[None] + source_times[:, None, None] * velocity[None]
    elif "directory" in source:
        recording = json.loads((source_directory / "navigation_comparison.json").read_text())
        dt = recording["summary"]["world"]["config"]["dt"]
        with np.load(source_directory / "dense_plant_states.npz", allow_pickle=False) as raw:
            states = raw[source["method"]]
        source_times = np.arange(len(states)) * dt
        with np.load(source_directory / "navigation_comparison.npz", allow_pickle=False) as raw:
            times, centers = raw["time_seconds"], raw["obstacle_centers"]
        source_centers = np.stack(
            [
                np.interp(source_times, times, centers[:, i, j])
                for i in range(centers.shape[1])
                for j in range(3)
            ],
            axis=1,
        ).reshape(len(source_times), -1, 3)
    else:
        raise ValueError("a saved navigation or legacy approach source is required")
    # Anchor the frame grid to handoff, so no frame crosses or interpolates that boundary.
    frame_dt = playback_rate / fps
    prelude = min(prelude_seconds, trigger - float(source_times[0]))
    if prelude < -1e-9:
        raise ValueError("approach source starts after contact handoff")
    before = trigger - np.arange(int(np.floor(prelude / frame_dt)), 0, -1) * frame_dt
    available = source_times < trigger - 1e-9
    source_ids = np.flatnonzero(available)
    pre_indices = (
        np.minimum(_nearest_indices(source_times, before), source_ids[-1])
        if before.size
        else np.array([], dtype=int)
    )
    after = (
        trigger
        + np.arange(int(np.floor((contact["time_seconds"][-1] - trigger) / frame_dt + 1e-9)) + 1)
        * frame_dt
    )
    post_indices = _nearest_indices(contact["time_seconds"], after)
    model = ET.parse(directory / "contact_model.xml").getroot()
    radii = np.asarray(
        [
            float(model.find(f".//geom[@name='obstacle_geom_{i}']").attrib["size"].split()[0])
            for i in range(contact["obstacle_centers"].shape[1])
        ]
    )
    selected_states = np.concatenate((states[pre_indices], contact["full_state"][post_indices]))
    selected_centers = np.concatenate(
        (source_centers[pre_indices], contact["obstacle_centers"][post_indices])
    )
    if not np.isfinite(selected_states).all() or not np.isfinite(selected_centers).all():
        raise ValueError("recorded render poses and obstacle centers must be finite")
    return ContactPresentation(
        np.concatenate((before, after)),
        np.concatenate((source_times[pre_indices], contact["time_seconds"][post_indices])),
        selected_states,
        selected_centers,
        radii,
        np.concatenate((np.zeros(len(before), dtype=bool), np.ones(len(after), dtype=bool))),
        np.concatenate((pre_indices, post_indices)),
        metadata,
        inputs,
    )


def _first_event(metadata: dict, kind: str) -> float | None:
    return min(
        (row["time_seconds"] for row in metadata["contact_events"] if row["kind"] == kind),
        default=None,
    )


def _compose(
    scene: np.ndarray, presentation: ContactPresentation, index: int, args: argparse.Namespace
) -> np.ndarray:
    figure = Figure(figsize=(args.width / 100, args.height / 100), dpi=100, facecolor="#071018")
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes((0.006, 0.105, 0.988, 0.802))
    axis.imshow(scene)
    axis.axis("off")
    contact = presentation.contact_phase[index]
    t = presentation.times[index]
    trigger = presentation.metadata["trigger"]["time_seconds"]
    is_impact = _first_event(presentation.metadata, "obstacle_contact") is not None
    title = (
        "Recorded encounter → simulated contact and fall"
        if is_impact
        else "Recorded safety failure → simulated motors-off drop"
    )
    figure.text(0.018, 0.969, title, color="#f5f7fa", fontsize=16, weight="bold", va="center")
    figure.text(
        0.982,
        0.969,
        f"t = {t:.3f} s  ·  {args.playback_rate:g}× replay",
        color="#f5f7fa",
        fontsize=11,
        ha="right",
        va="center",
    )
    phase = (
        "MUJOCO CONTACT REPLAY · MOTORS OFF"
        if contact
        else "RECORDED APPROACH · ORIGINAL CONTROLLER"
    )
    figure.text(
        0.018, 0.931, phase, color="#ffbdc1" if contact else "#97efc5", fontsize=12, va="center"
    )
    figure.text(
        0.022,
        0.868,
        f"Body altitude {presentation.states[index, 2]:.2f} m",
        color="white",
        fontsize=11,
        bbox={"facecolor": "#102331", "edgecolor": "none", "alpha": 0.88, "pad": 5},
    )
    events = []
    for kind, label in (
        ("obstacle_contact", "Obstacle contact"),
        ("ground_contact", "Ground contact"),
    ):
        first = _first_event(presentation.metadata, kind)
        if first is not None and contact and t >= first - 1e-9:
            events.append(f"{label} at {first:.3f} s")
    event_text = "  ·  ".join(events) or (
        "Motors off; contact forces evolve in MuJoCo"
        if contact
        else f"Simulation handoff at {trigger:.3f} s"
    )
    figure.text(
        0.5, 0.078, event_text, color="#ffbdc1" if contact else "#d5e1eb", fontsize=12, ha="center"
    )
    figure.text(
        0.5,
        0.047,
        "White: recorded path · cyan: simulated path · actual cf21B mesh at saved poses",
        color="#a9bac7",
        fontsize=10,
        ha="center",
    )
    figure.text(
        0.5,
        0.019,
        "Separate motor-cut simulation after handoff; "
        "the original controller is no longer running.",
        color="#a9bac7",
        fontsize=10,
        ha="center",
    )
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def render(args: argparse.Namespace) -> Path:
    source = Path(__file__).resolve()
    source_bytes = source.read_bytes()
    common_renderer_sha256 = _sha256(
        source.parents[2] / "crazyflow/safety/da_plcbf/mujoco_comparison_video.py"
    )
    presentation = load_presentation(
        args.input_dir,
        prelude_seconds=args.prelude_seconds,
        fps=args.fps,
        playback_rate=args.playback_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    destination = args.output_dir / "contact_replay_demo.mp4"
    temporary = args.output_dir / ".contact_replay_demo.encoding.mp4"
    scene_width, scene_height = args.width, int(args.height * 0.812)
    camera = {
        "lookat": presentation.states[0, :3].copy(),
        "distance": args.camera_distance,
        "azimuth": args.camera_azimuth,
        "elevation": args.camera_elevation,
    }
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        drone="cf21B_500",
        device="cpu",
        fused_mjx_model=False,
        enable_contacts=False,
    )
    sim.max_visual_geom = 2000
    sim.mj_model.geom("floor").pos[2] = presentation.metadata["config"]["ground_height"]
    center = np.asarray(sim.mj_model.stat.center)
    radius = max(
        np.linalg.norm(presentation.states[:, :3] - center, axis=-1).max(),
        np.linalg.norm(presentation.obstacle_centers - center, axis=-1).max()
        + presentation.obstacle_radii.max(),
    )
    sim.mj_model.vis.map.shadowclip = max(
        sim.mj_model.vis.map.shadowclip, (radius + args.camera_distance) / sim.mj_model.stat.extent
    )
    sim.mj_model.vis.quality.shadowsize = max(4096, sim.mj_model.vis.quality.shadowsize)
    for material in ("led_top", "led_bot"):
        change_material(
            sim,
            material,
            np.asarray([0], dtype=np.int32),
            rgba=np.asarray([0.10, 0.86, 1.0, 1.0]),
            emission=np.asarray(1.0),
        )
    writer = imageio_ffmpeg.write_frames(
        str(temporary),
        (args.width, args.height),
        fps=args.fps,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        codec="libx264",
        quality=None,
        macro_block_size=2,
        ffmpeg_log_level="error",
        output_params=[
            "-preset",
            "medium",
            "-crf",
            "18",
            "-threads",
            "1",
            "-movflags",
            "+faststart",
        ],
    )
    trigger_index = int(np.flatnonzero(presentation.contact_phase)[0])
    ground_time = _first_event(presentation.metadata, "ground_contact")
    keyframes = {0, trigger_index, len(presentation.times) - 1}
    if ground_time is not None:
        keyframes.add(int(np.argmin(abs(presentation.times - ground_time - 0.10))))
    saved_frames = []
    try:
        writer.send(None)
        initialized = False
        for index, state in enumerate(presentation.states):
            states = sim.data.states.replace(
                pos=jnp.asarray(state[None, None, :3]), quat=jnp.asarray(state[None, None, 3:7])
            )
            core = sim.data.core.replace(
                mjx_synced=jnp.asarray(False), mjx_collision_synced=jnp.asarray(False)
            )
            sim.data = sim.data.replace(states=states, core=core)
            if not initialized:
                sim.render(
                    mode="rgb_array",
                    world=0,
                    camera=-1,
                    cam_config=camera,
                    width=scene_width,
                    height=scene_height,
                )
                _install_marker_shadow_categories(sim)
                initialized = True
            # A fixed metric camera follows translation only; the mesh keeps the recorded attitude.
            lookat = state[:3].copy()
            ground = presentation.metadata["config"]["ground_height"]
            lookat[2] = ground + max(0.35, 0.72 * (lookat[2] - ground))
            sim.viewer.viewer.cam.lookat[:] = lookat
            for center, radius in zip(
                presentation.obstacle_centers[index], presentation.obstacle_radii, strict=True
            ):
                sim.viewer.viewer.add_marker(
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    pos=center,
                    size=np.full(3, radius),
                    rgba=np.asarray([0.78, 0.09, 0.065, 1.0]),
                    label="",
                    category=int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
                )
            _add_polyline(
                sim,
                presentation.states[: min(index + 1, trigger_index + 1), :3],
                np.asarray([0.94, 1.0, 1.0, 0.9]),
                radius=0.006,
            )
            if index >= trigger_index:
                _add_polyline(
                    sim,
                    presentation.states[trigger_index : index + 1, :3],
                    np.asarray([0.16, 0.85, 1.0, 0.9]),
                    radius=0.006,
                )
            scene = sim.render(
                mode="rgb_array",
                world=0,
                camera=-1,
                cam_config=camera,
                width=scene_width,
                height=scene_height,
            )
            frame = _compose(scene, presentation, index, args)
            writer.send(frame)
            if index in keyframes:
                path = args.output_dir / f"frame_{index:04d}_{presentation.times[index]:.3f}s.png"
                Image.fromarray(frame).save(path)
                saved_frames.append(
                    {
                        "path": path.name,
                        "sha256": _sha256(path),
                        "frame_index": index,
                        "source_time_seconds": float(presentation.times[index]),
                    }
                )
        writer.close()
        writer = None
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        sim.close()
    arrays = args.output_dir / "rendered_source_samples.npz"
    np.savez_compressed(
        arrays,
        requested_times=presentation.requested_times,
        actual_source_times=presentation.times,
        full_state=presentation.states,
        obstacle_centers=presentation.obstacle_centers,
        contact_phase=presentation.contact_phase,
        source_indices=presentation.source_indices,
    )
    source_copy = args.output_dir / "CONTACT_RENDERER_SOURCE.py"
    source_copy.write_bytes(source_bytes)
    metadata = {
        "scope": __doc__,
        "input_sha256": presentation.inputs,
        "render_config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "command": shlex.join([sys.executable, str(source), *sys.argv[1:]]),
        "renderer_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "common_renderer_sha256": common_renderer_sha256,
        "video_sha256": _sha256(destination),
        "source_samples_sha256": _sha256(arrays),
        "frame_count": len(presentation.times),
        "duration_seconds": len(presentation.times) / args.fps,
        "actual_prelude_seconds": float(
            presentation.metadata["trigger"]["time_seconds"] - presentation.times[0]
        ),
        "maximum_source_time_quantization_seconds": float(
            np.max(abs(presentation.times - presentation.requested_times))
        ),
        "contact_trigger": presentation.metadata["trigger"],
        "first_obstacle_contact_seconds": _first_event(presentation.metadata, "obstacle_contact"),
        "first_ground_contact_seconds": ground_time,
        "representative_frames": saved_frames,
        "visual_model": "cf21B_500 mesh, unchanged physical scale, actual saved xyzw attitude",
        "physics_model": (
            "Archived contact_model.xml; mesh playback does not rerun physics "
            "or change inertia/colliders"
        ),
        "source_selection": (
            "Nearest saved samples within each source; frame grid anchored at handoff; "
            "no interpolation across handoff or trajectory extrapolation"
        ),
    }
    (args.output_dir / "render_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"video": str(destination), "frames": saved_frames}, indent=2))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prelude-seconds", type=float, default=1.5)
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--playback-rate", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--camera-distance", type=float, default=3.4)
    parser.add_argument("--camera-azimuth", type=float, default=-45)
    parser.add_argument("--camera-elevation", type=float, default=-22)
    args = parser.parse_args()
    if args.width < 640 or args.height < 360 or args.width % 2 or args.height % 2:
        parser.error("video dimensions must be even and at least 640×360")
    if (
        not np.isfinite([args.camera_distance, args.camera_azimuth, args.camera_elevation]).all()
        or args.camera_distance <= 0
    ):
        parser.error("camera parameters must be finite and distance positive")
    render(args)


if __name__ == "__main__":
    main()
