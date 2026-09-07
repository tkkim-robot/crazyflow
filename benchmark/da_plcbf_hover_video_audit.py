"""Verify every hover video frame against recorded flight or contact-replay states."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from benchmark.da_plcbf_actuator_video import _array_sha, _jsonable, _sample_audit, load_episode
from benchmark.da_plcbf_hover_return import BASE
from benchmark.da_plcbf_hover_video import METHODS, TIMES
from benchmark.da_plcbf_recovery_analysis import sha

OUTPUT = BASE / "video-v2"
SOURCE = BASE / "selected-replay-v2"


def main() -> None:
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    movie = Path(manifest["video"])
    assert manifest["full_decode_passed"] and sha(movie) == manifest["sha256"]
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames,duration",
                "-of",
                "json",
                str(movie),
            ]
        )
    )["streams"][0]
    assert (probe["width"], probe["height"], probe["avg_frame_rate"]) == (2400, 900, "20/1")
    assert int(probe["nb_read_frames"]) == len(TIMES)
    assert abs(float(probe["duration"]) - (TIMES[-1] + 0.05)) < 1e-6
    panels = []
    for method, binding in zip(METHODS, manifest["panels"], strict=True):
        directory = OUTPUT / method
        episode = load_episode(SOURCE / method / "attempt-00", expected_method=method)
        for name, digest in binding["source_sha256"].items():
            assert sha(Path(name)) == digest, name
        assert sha(directory / "panel.mp4") == binding["video_sha256"]
        assert sha(directory / "frame_audit.jsonl") == binding["frame_audit_sha256"]
        frames = [json.loads(s) for s in (directory / "frame_audit.jsonl").read_text().splitlines()]
        assert len(frames) == len(TIMES)
        ct, cs, rotation = None, None, None
        if binding["contact_replay"] is not None:
            metadata = json.loads((directory / "contact/contact_replay.json").read_text())
            assert sha(directory / "contact/contact_replay.npz") == metadata["npz_sha256"]
            assert metadata["ground_contact_steps"] > 0 and metadata["obstacle_contact_steps"] > 0
            assert not any(metadata["warning_counts"].values())
            with np.load(directory / "contact/contact_replay.npz") as contact:
                ct, cs = contact["time_seconds"], contact["full_state"]
            rotation = Slerp(ct, Rotation.from_quat(cs[:, 3:7]))
        contact_frames = 0
        for index, (frame, when) in enumerate(zip(frames, TIMES, strict=True)):
            assert frame["frame"] == index and frame["physical_time"] == when
            is_contact = ct is not None and when >= ct[0]
            assert frame["contact_replay"] == is_contact
            if is_contact:
                contact_frames += 1
                expected = np.asarray([np.interp(when, ct, cs[:, k]) for k in range(13)])
                expected[3:7] = rotation(when).as_quat()
                np.testing.assert_allclose(frame["state17"][:13], expected, atol=1e-12, rtol=0)
                assert frame["state17"][13:] == [0, 0, 0, 0]
                assert frame["available_control_index"] is None
                assert frame["command_N"] == frame["actual_force_N"] == [0, 0, 0, 0]
                assert not frame["executing_committed_backup"]
                np.testing.assert_array_equal(frame["wind_velocity_mps"], episode.wind_at(when))
            else:
                expected = _jsonable(_sample_audit(episode, episode.sample(when)))
                for key, value in expected.items():
                    assert frame[key] == value, (method, index, key)
        previews = sorted(directory.glob("frame-*.png"))
        for path in previews:
            index = int(path.stem.split("-")[1])
            assert _array_sha(np.asarray(Image.open(path))) == frames[index]["raw_rgb_sha256"]
        panels.append(
            {
                "method": method,
                "frames_verified": len(frames),
                "recorded_flight_frames_verified": len(frames) - contact_frames,
                "contact_frames_verified": contact_frames,
                "preview_hashes_verified": len(previews),
                "final_online_updates_used": frames[-1]["online_updates_used"],
            }
        )
    assert [r["final_online_updates_used"] for r in panels] == [0, 0, 799]
    verification = {
        "video_sha256": sha(movie),
        "ffprobe": probe,
        "panels": panels,
        "full_decode_passed": True,
        "recorded_prefix_states_commands_predictions_binding_verified": True,
        "contact_states_equal_separate_simulation": True,
    }
    (OUTPUT / "validation.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
