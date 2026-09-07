"""Render the new uncompensated wind flights as three synchronized panels."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from benchmark.da_plcbf_actuator_video import (
    ActuatorVideoConfig,
    ReplayEpisode,
    ReplaySample,
    _array_sha,
    _camera,
    _compose,
    _font,
    _jsonable,
    _markers,
    _sample_audit,
    load_episode,
    motor_site_positions,
    validate_pair,
)
from benchmark.da_plcbf_recovery_analysis import sha
from benchmark.da_plcbf_wind_learning_comparison import BASE

if TYPE_CHECKING:
    from collections.abc import Generator


METHODS = ("PD_F", "F2", "A_BAL")
LABELS = ("Handcrafted · frozen", "Learned · frozen", "Learned · adaptive")
OUTPUT = BASE / "video-v1"
TIMES = np.arange(861) / 20


def episode_for(method: str) -> ReplayEpisode:
    episode = load_episode(BASE / "flights" / method / "attempt-00", expected_method=method)
    assert episode.binding["checkpoint"]["runtime_actor_config"]["wind_feedforward"] is False
    return episode


def contact_samples(
    episode: ReplayEpisode, output: Path
) -> tuple[list[ReplaySample], ReplayEpisode, np.ndarray, dict[str, Any] | None]:
    """Replace only a collided suffix with explicitly separate rigid-body dynamics."""
    from crazyflow.safety.da_plcbf.contact_replay import (
        ContactBody,
        ContactReplayConfig,
        ObstacleMotion,
        cf21b_contact_body,
        find_contact_trigger,
        run_contact_replay,
        save_contact_replay,
    )

    samples = [episode.sample(t) for t in TIMES]
    mask = np.zeros(len(TIMES), dtype=bool)
    if episode.contact_time is None:
        return samples, episode, mask, None
    support = np.arange(43002) * 0.001
    obstacles = ObstacleMotion(
        support,
        np.asarray([episode.obstacle_centers(t) for t in support]),
        np.asarray(episode.world["obstacle_radii"]),
    )
    trigger = find_contact_trigger(
        episode.dense["time"],
        episode.dense["state"][:, :13],
        obstacles,
        ContactReplayConfig(),
        kind="physical_contact",
    )
    assert abs(trigger.time_seconds - episode.contact_time) < 0.002
    config = ContactReplayConfig(duration_seconds=43 - trigger.time_seconds)
    default = cf21b_contact_body()
    body = ContactBody(
        float(episode.controls["actual_mass"][-1]),
        episode.controls["actual_inertia"][-1],
        default.gravity,
        default.drag_matrix_body,
    )
    times = trigger.time_seconds + np.arange(math.ceil(config.duration_seconds / 0.001) + 1) * 0.001
    replay = run_contact_replay(
        trigger,
        body,
        obstacles,
        config,
        wind_velocity_world=np.asarray([episode.wind_at(t) for t in times]),
    )
    assert replay.metadata["ground_contact_steps"] > 0
    replay.metadata.update(
        {
            "source_episode": str(episode.directory),
            "source_files_sha256": episode.source_sha256,
            "source_contact_time": episode.contact_time,
            "driver_sha256": sha(Path(__file__)),
            "presentation_policy": (
                "At physical contact, switch to zero rotor thrust and MuJoCo rigid-body contact "
                "dynamics. This is a separately simulated crash presentation, not the controller's "
                "recorded suffix. Predictions stop at the handoff."
            ),
        }
    )
    contact_dir = save_contact_replay(replay, output / "contact")
    with np.load(contact_dir / "contact_replay.npz") as values:
        ct, cs = values["time_seconds"], values["full_state"]
    mask = TIMES >= ct[0]
    q = TIMES[mask]
    interpolated = np.column_stack([np.interp(q, ct, cs[:, i]) for i in range(13)])
    interpolated[:, 3:7] = Slerp(ct, Rotation.from_quat(cs[:, 3:7]))(q).as_quat()
    for index, state in zip(np.flatnonzero(mask), interpolated, strict=True):
        samples[index] = replace(
            samples[index],
            display_time=TIMES[index],
            state=np.r_[state, np.zeros(4)],
            command=np.zeros(4),
            actual_forces=np.zeros(4),
            control_index=None,
            contact_stopped=False,
            terminal_stopped=False,
        )
    prefix = episode.dense["time"] < ct[0]
    trail_episode = replace(
        episode,
        dense={
            **episode.dense,
            "time": np.r_[episode.dense["time"][prefix], ct],
            "state": np.vstack(
                (episode.dense["state"][prefix], np.column_stack((cs, np.zeros((len(cs), 4)))))
            ),
        },
    )
    return samples, trail_episode, mask, replay.metadata


def writer_for(path: Path, size: tuple[int, int]) -> Generator:
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size,
        fps=20,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        codec="libx264",
        quality=None,
        macro_block_size=2,
        output_params=[
            "-crf",
            "18",
            "-preset",
            "medium",
            "-threads",
            "1",
            "-movflags",
            "+faststart",
        ],
    )
    writer.send(None)
    return writer


def render_panel(method: str) -> None:
    from crazyflow import Sim
    from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
        _install_marker_shadow_categories,
        _set_two_world_poses,
    )

    output = OUTPUT / method
    output.mkdir(parents=True, exist_ok=False)
    episode = episode_for(method)
    sources = dict(episode.source_sha256)
    sources.update(
        {
            str(p.resolve()): sha(p)
            for p in (
                Path(__file__),
                Path("benchmark/da_plcbf_actuator_video.py"),
                Path("crazyflow/safety/da_plcbf/contact_replay.py"),
            )
        }
    )
    samples, trail_episode, contact_mask, contact = contact_samples(episode, output)
    pose = SimpleNamespace(
        position=np.asarray([s.state[:3] for s in samples]),
        quaternion_xyzw=np.asarray([s.state[3:7] for s in samples]),
    )
    trace = SimpleNamespace(fixed=pose, adaptive=pose)
    label = LABELS[METHODS.index(method)]
    config = ActuatorVideoConfig(camera_azimuth=0, left_label=label, right_label=label)
    sites = motor_site_positions()
    sim = Sim(
        n_worlds=2,
        n_drones=1,
        drone="cf21B_500",
        device="cpu",
        fused_mjx_model=False,
        enable_contacts=False,
    )
    sim.max_visual_geom = 6000
    extent = np.linalg.norm(trail_episode.dense["state"][:, :3], axis=1).max() + 5.4
    sim.mj_model.vis.map.shadowclip = max(
        sim.mj_model.vis.map.shadowclip, extent / sim.mj_model.stat.extent
    )
    sim.mj_model.vis.quality.shadowsize = max(4096, sim.mj_model.vis.quality.shadowsize)
    initial_camera = _camera(samples[0], config)
    _set_two_world_poses(sim, trace, 0)
    sim.render(
        mode="rgb_array", world=0, camera=-1, cam_config=initial_camera, width=800, height=694
    )
    _install_marker_shadow_categories(sim)
    writer = writer_for(output / ".encoding.mp4", (800, 900))
    try:
        with (output / "frame_audit.jsonl").open("x") as audit:
            for index, sample in enumerate(samples):
                camera = _camera(sample, config)
                _set_two_world_poses(sim, trace, index)
                sim.viewer.viewer.cam.lookat[:] = camera["lookat"]
                sim.viewer.viewer.cam.distance = camera["distance"]
                _markers(sim, trail_episode, sample, config, sites)
                viewport = sim.render(
                    mode="rgb_array",
                    world=0,
                    camera=-1,
                    cam_config=initial_camera,
                    width=800,
                    height=694,
                )
                # The right half always includes the update counter, including zero.
                pixels = _compose(
                    (viewport, viewport), (sample, sample), (episode, episode), config, 0.25
                )[:, 800:]
                frame = Image.fromarray(pixels)
                draw = ImageDraw.Draw(frame)
                draw.rounded_rectangle((20, 124, 535, 154), radius=5, fill="#102936")
                draw.text(
                    (30, 132),
                    "Direct wind feedforward: OFF · state feedback: ON",
                    font=_font(14),
                    fill="#d9e4eb",
                )
                if contact_mask[index]:
                    draw.rounded_rectangle((20, 164, 778, 199), radius=5, fill="#591b18")
                    draw.text(
                        (30, 174),
                        "Impact → motors off · MuJoCo contact replay · predictions stopped",
                        font=_font(15),
                        fill="#ffe2d7",
                    )
                draw.rectangle((0, 875, 800, 900), fill="#071018")
                pixels = np.asarray(frame)
                writer.send(pixels)
                row = _sample_audit(episode, sample)
                row.update(
                    {
                        "frame": index,
                        "physical_time": TIMES[index],
                        "method": method,
                        "contact_replay": bool(contact_mask[index]),
                        "raw_rgb_sha256": _array_sha(pixels),
                    }
                )
                audit.write(json.dumps(_jsonable(row)) + "\n")
                if index % 100 == 0 or index in (60, 140, 300, 380, 435, 500, 600, 740, 860):
                    frame.save(output / f"frame-{index:06d}.png")
                    audit.flush()
                    print(json.dumps({"method": method, "frame": index}), flush=True)
        writer.close()
        writer = None
        (output / ".encoding.mp4").rename(output / "panel.mp4")
        for path, digest in sources.items():
            assert sha(Path(path)) == digest, path
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "source_sha256": sources,
                    "video_sha256": sha(output / "panel.mp4"),
                    "frame_audit_sha256": sha(output / "frame_audit.jsonl"),
                    "method": method,
                    "frame_count": len(TIMES),
                    "contact_replay": contact,
                    "direct_wind_feedforward": False,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        if writer is not None:
            writer.close()
        sim.close()


def combine() -> None:
    episodes = [episode_for(method) for method in METHODS]
    for episode in episodes[1:]:
        validate_pair(
            episodes[0],
            episode,
            allow_different_learning_contract=True,
            allow_different_libraries=True,
        )
    manifests = [json.loads((OUTPUT / m / "manifest.json").read_text()) for m in METHODS]
    movies = [OUTPUT / m / "panel.mp4" for m in METHODS]
    for path, manifest in zip(movies, manifests, strict=True):
        assert sha(path) == manifest["video_sha256"]
    readers = [imageio_ffmpeg.read_frames(str(p), pix_fmt="rgb24") for p in movies]
    for reader in readers:
        info = next(reader)
        assert tuple(info["size"]) == (800, 900) and info["fps"] == 20
    writer = writer_for(OUTPUT / ".encoding.mp4", (2400, 900))
    try:
        for index in range(len(TIMES)):
            panels = [np.frombuffer(next(r), dtype=np.uint8).reshape(900, 800, 3) for r in readers]
            frame = Image.fromarray(np.concatenate(panels, axis=1))
            draw = ImageDraw.Draw(frame)
            for x in (800, 1600):
                draw.line([(x, 0), (x, 900)], fill="#33434c", width=1)
            legend = (
                "Colored: current predictions    Cyan: actual wind    White: flown path    "
                "Gold: executing checked backup    Shared motor scale: 0–0.25 N"
            )
            box = draw.textbbox((0, 0), legend, font=_font(13))
            draw.text(((2400 - box[2]) / 2, 880), legend, font=_font(13), fill="#aabac5")
            writer.send(np.asarray(frame))
            if index in (0, 60, 100, 140, 300, 380, 435, 500, 600, 740, 860):
                frame.save(OUTPUT / f"frame-{index:06d}.png")
        for reader in readers:
            assert next(reader, None) is None, "Unexpected extra panel frame"
        writer.close()
        writer = None
        destination = OUTPUT / "comparison.mp4"
        (OUTPUT / ".encoding.mp4").rename(destination)
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-i",
                str(destination),
                "-f",
                "null",
                "-",
            ],
            check=True,
        )
        (OUTPUT / "manifest.json").write_text(
            json.dumps(
                {
                    "video": str(destination),
                    "sha256": sha(destination),
                    "frame_count": len(TIMES),
                    "width": 2400,
                    "height": 900,
                    "fps": 20,
                    "full_decode_passed": True,
                    "panel_order": list(METHODS),
                    "panels": manifests,
                    "scope": (
                        "New recorded controller prefixes; labeled motor-off contact suffixes only."
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        print(destination, flush=True)
    finally:
        if writer is not None:
            writer.close()
        for reader in readers:
            reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*METHODS, "combine"))
    args = parser.parse_args()
    combine() if args.stage == "combine" else render_panel(args.stage)
