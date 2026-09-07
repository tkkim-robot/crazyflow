"""Three simultaneous recorded flights, with a labeled motor-off contact continuation."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from benchmark.da_plcbf_actuator_video import (
    ActuatorVideoConfig,
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
from benchmark.da_plcbf_recovery_analysis import BASE as SOURCE
from benchmark.da_plcbf_recovery_analysis import sha
from benchmark.da_plcbf_three_panel_contact import BASE


def main() -> None:
    output = BASE / "video-v1"
    output.mkdir(parents=True, exist_ok=False)
    episodes = [
        load_episode(SOURCE / "wind-demo-v1" / method / "attempt-00", expected_method=method)
        for method in ("PD_F", "F2", "A_BAL")
    ]
    for episode in episodes[1:]:
        validate_pair(
            episodes[0],
            episode,
            allow_different_learning_contract=True,
            allow_different_libraries=True,
        )
    contact_dir = BASE / "contact-v1"
    metadata = json.loads((contact_dir / "contact_replay.json").read_text())
    assert sha(contact_dir / "contact_replay.npz") == metadata["npz_sha256"]
    assert metadata["obstacle_contact_steps"] > 0 and metadata["ground_contact_steps"] > 0
    with np.load(contact_dir / "contact_replay.npz") as values:
        ct, cs = values["time_seconds"], values["full_state"]
        ground = values["ground_contact"]
    trigger = ct[0]
    times = np.arange(861) / 20
    samples = [[episode.sample(t) for t in times] for episode in episodes]
    contact_mask = times >= trigger
    q = times[contact_mask]
    interpolated = np.column_stack([np.interp(q, ct, cs[:, i]) for i in range(13)])
    interpolated[:, 3:7] = Slerp(ct, Rotation.from_quat(cs[:, 3:7]))(q).as_quat()
    first = int(np.flatnonzero(contact_mask)[0])
    for index, state in zip(np.flatnonzero(contact_mask), interpolated, strict=True):
        original = samples[0][index]
        samples[0][index] = replace(
            original,
            display_time=times[index],
            state=np.r_[state, np.zeros(4)],
            command=np.zeros(4),
            actual_forces=np.zeros(4),
            control_index=None,
            contact_stopped=False,
            terminal_stopped=False,
        )
    prefix = episodes[0].dense["time"] < trigger
    contact_episode = replace(
        episodes[0],
        dense={
            **episodes[0].dense,
            "time": np.r_[episodes[0].dense["time"][prefix], ct],
            "state": np.vstack(
                (episodes[0].dense["state"][prefix], np.column_stack((cs, np.zeros((len(cs), 4)))))
            ),
        },
    )
    pose = SimpleNamespace(
        position=np.asarray([s.state[:3] for s in samples[0]]),
        quaternion_xyzw=np.asarray([s.state[3:7] for s in samples[0]]),
    )
    pose_trace = SimpleNamespace(fixed=pose, adaptive=pose)
    config = ActuatorVideoConfig(
        camera_azimuth=0, left_label="Handcrafted · frozen", right_label="Handcrafted · frozen"
    )
    learned_config = replace(
        config, left_label="Learned · frozen", right_label="Learned · adaptive"
    )
    source_movies = [
        SOURCE / "wind-video-v1" / pair / "comparison.mp4" for pair in ("libraries", "adaptation")
    ]
    sources = {str(p): sha(p) for p in source_movies}
    sources.update(
        {
            str(p): sha(p)
            for p in (
                Path(__file__),
                Path("benchmark/da_plcbf_actuator_video.py"),
                contact_dir / "contact_replay.npz",
                contact_dir / "contact_replay.json",
            )
        }
    )
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    (output / "binding.json").write_text(
        json.dumps(
            {
                "source_sha256": sources,
                "panel_order": ["Handcrafted frozen", "Learned frozen", "Learned adaptive"],
                "width": 2400,
                "height": 900,
                "fps": 20,
                "force_scale_N": 0.25,
                "contact_handoff_seconds": trigger,
                "contact_policy": metadata["presentation_policy"],
                "learned_viewports": "Decoded original videos; no flight or learning rerun",
                "handcrafted_prefix": "Decoded original video until the contact handoff",
            },
            indent=2,
        )
        + "\n"
    )
    readers = [imageio_ffmpeg.read_frames(str(p), pix_fmt="rgb24") for p in source_movies]
    for reader in readers:
        info = next(reader)
        assert tuple(info["size"]) == (1600, 900) and info["fps"] == 20
    destination = output / "comparison.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(output / ".encoding.mp4"),
        (2400, 900),
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
    sim = None
    sites = motor_site_positions()
    try:
        with (output / "frame_audit.jsonl").open("x") as audit:
            for index, when in enumerate(times):
                old = [
                    np.frombuffer(next(r), dtype=np.uint8).reshape(900, 1600, 3) for r in readers
                ]
                left_image = old[0][66:760, :800]
                if contact_mask[index]:
                    from crazyflow import Sim
                    from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
                        _install_marker_shadow_categories,
                        _set_two_world_poses,
                    )

                    camera = _camera(samples[0][index], config)
                    if sim is None:
                        initial_camera = dict(camera)
                        sim = Sim(
                            n_worlds=2,
                            n_drones=1,
                            drone="cf21B_500",
                            device="cpu",
                            fused_mjx_model=False,
                            enable_contacts=False,
                        )
                        sim.max_visual_geom = 6000
                        extent = (
                            max(
                                np.linalg.norm(e.dense["state"][:, :3], axis=1).max()
                                for e in episodes
                            )
                            + 5.4
                        )
                        sim.mj_model.vis.map.shadowclip = max(
                            sim.mj_model.vis.map.shadowclip, extent / sim.mj_model.stat.extent
                        )
                        sim.mj_model.vis.quality.shadowsize = max(
                            4096, sim.mj_model.vis.quality.shadowsize
                        )
                        _set_two_world_poses(sim, pose_trace, index)
                        sim.render(
                            mode="rgb_array",
                            world=0,
                            camera=-1,
                            cam_config=camera,
                            width=800,
                            height=694,
                        )
                        _install_marker_shadow_categories(sim)
                    _set_two_world_poses(sim, pose_trace, index)
                    sim.viewer.viewer.cam.lookat[:] = camera["lookat"]
                    sim.viewer.viewer.cam.distance = camera["distance"]
                    _markers(sim, contact_episode, samples[0][index], config, sites)
                    left_image = sim.render(
                        mode="rgb_array",
                        world=0,
                        camera=-1,
                        cam_config=initial_camera,
                        width=800,
                        height=694,
                    )
                left = _compose(
                    (left_image, left_image),
                    (samples[0][index], samples[0][index]),
                    (episodes[0], episodes[0]),
                    config,
                    0.25,
                )[:, :800]
                right = _compose(
                    (old[1][66:760, :800], old[1][66:760, 800:]),
                    (samples[1][index], samples[2][index]),
                    tuple(episodes[1:]),
                    learned_config,
                    0.25,
                )
                frame = Image.fromarray(np.concatenate((left, right), axis=1))
                draw = ImageDraw.Draw(frame)
                for x in (800, 1600):
                    draw.line([(x, 0), (x, 900)], fill="#33434c", width=1)
                if contact_mask[index]:
                    draw.rounded_rectangle((20, 128, 765, 163), radius=5, fill="#591b18")
                    draw.text(
                        (32, 137),
                        "Impact → motors off · MuJoCo contact replay · predictions stopped",
                        font=_font(15),
                        fill="#ffe2d7",
                    )
                draw.rectangle((0, 875, 2400, 900), fill="#071018")
                legend = (
                    "Colored: current predictions    Cyan: actual wind    White: flown path    "
                    "Gold: executing checked backup    Shared motor scale: 0–0.25 N"
                )
                box = draw.textbbox((0, 0), legend, font=_font(13))
                draw.text(((2400 - box[2]) / 2, 880), legend, font=_font(13), fill="#aabac5")
                pixels = np.asarray(frame)
                writer.send(pixels)
                audit.write(
                    json.dumps(
                        _jsonable(
                            {
                                "frame": index,
                                "physical_time": when,
                                "raw_rgb_sha256": _array_sha(pixels),
                                "panels": [
                                    _sample_audit(e, s[index])
                                    for e, s in zip(episodes, samples, strict=True)
                                ],
                                "handcrafted_contact_replay": bool(contact_mask[index]),
                                "handcrafted_ground_contact": bool(
                                    ground[min(np.searchsorted(ct, when), len(ct) - 1)]
                                )
                                if contact_mask[index]
                                else False,
                                "learned_original_viewport_sha256": _array_sha(
                                    np.concatenate(
                                        (old[1][164:760, 1:800], old[1][164:760, 801:]), axis=1
                                    )
                                ),
                                "learned_composed_viewport_sha256": _array_sha(
                                    np.concatenate(
                                        (pixels[164:760, 801:1600], pixels[164:760, 1601:]), axis=1
                                    )
                                ),
                            }
                        )
                    )
                    + "\n"
                )
                audit.flush()
                if index % 100 == 0 or index in (first, 450, 470, 500, 860):
                    frame.save(output / f"frame-{index:06d}.png")
        writer.close()
        writer = None
        (output / ".encoding.mp4").rename(destination)
        for path, digest in sources.items():
            assert sha(Path(path)) == digest, path
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
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "video": str(destination),
                    "sha256": sha(destination),
                    "frame_count": len(times),
                    "full_decode_passed": True,
                    "source_sha256": sources,
                    "frame_audit_sha256": sha(output / "frame_audit.jsonl"),
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
        if sim is not None:
            sim.close()


if __name__ == "__main__":
    main()
