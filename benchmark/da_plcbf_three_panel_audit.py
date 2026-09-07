"""Authenticate the three-panel composition, contact suffix and compact review evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from benchmark.da_plcbf_actuator_video import _array_sha
from benchmark.da_plcbf_recovery_analysis import sha
from benchmark.da_plcbf_three_panel_contact import BASE


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = BASE / "video-v1"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["full_decode_passed"]
    movie = Path(manifest["video"])
    assert sha(movie) == manifest["sha256"]
    for name, digest in manifest["source_sha256"].items():
        assert sha(Path(name)) == digest, name
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
    frames = [json.loads(s) for s in (directory / "frame_audit.jsonl").read_text().splitlines()]
    assert len(frames) == int(probe["nb_read_frames"]) == 861
    assert abs(float(probe["duration"]) - 43.05) < 1e-6
    with np.load(BASE / "contact-v1/contact_replay.npz") as c:
        times, states = c["time_seconds"], c["full_state"]
    rotation = Slerp(times, Rotation.from_quat(states[:, 3:7]))
    contact_frames = 0
    for index, frame in enumerate(frames):
        assert frame["frame"] == index and abs(frame["physical_time"] - index / 20) < 1e-10
        assert (
            frame["learned_original_viewport_sha256"] == frame["learned_composed_viewport_sha256"]
        )
        when = frame["physical_time"]
        assert frame["handcrafted_contact_replay"] == (when >= times[0])
        if when >= times[0]:
            contact_frames += 1
            sample = frame["panels"][0]
            expected = np.asarray([np.interp(when, times, states[:, k]) for k in range(13)])
            expected[3:7] = rotation(when).as_quat()
            np.testing.assert_allclose(sample["state17"][:13], expected, atol=1e-12, rtol=0)
            assert sample["available_control_index"] is None
            assert sample["command_N"] == sample["actual_force_N"] == [0, 0, 0, 0]
            assert not sample["executing_committed_backup"]
    assert contact_frames > 0 and any(f["handcrafted_ground_contact"] for f in frames)
    assert all(
        not f["panels"][1]["contact_stopped"] and not f["panels"][2]["contact_stopped"]
        for f in frames
    )
    assert frames[-1]["panels"][1]["online_updates_used"] == 0
    assert frames[-1]["panels"][2]["online_updates_used"] == 1074
    previews = sorted(directory.glob("frame-*.png"))
    for path in previews:
        index = int(path.stem.split("-")[1])
        assert _array_sha(np.asarray(Image.open(path))) == frames[index]["raw_rgb_sha256"]
    verification = {
        "video_sha256": sha(movie),
        "ffprobe": probe,
        "frames_verified": len(frames),
        "contact_frames_verified": contact_frames,
        "preview_hashes_verified": len(previews),
        "learned_viewports_equal_original_decoded_source_before_encoding": True,
        "contact_states_equal_separate_simulation": True,
        "full_decode_passed": True,
    }
    (directory / "validation.json").write_text(json.dumps(verification, indent=2) + "\n")
    report_path = root / "docs/da_plcbf_three_panel_wind_report.md"
    report_path.write_text(
        report_path.read_text()
        + "\nFinal video validation passes: 861 frames at 2400×900 and 20 fps, full decoding, "
        + f"{contact_frames} contact-suffix frames matched to the separate simulation, "
        + "unchanged learned viewport interiors before encoding, and every saved preview hash.\n"
    )
    logs = BASE / "logs"
    logs.mkdir(exist_ok=True)
    for pattern in (
        "three-panel-contact*.log",
        "three-panel-video*.log",
        "three-panel-tests*.log",
        "wind-bias-diagnosis*.log",
    ):
        for path in Path("/tmp").glob(pattern):
            shutil.copy2(path, logs / path.name)
    output = BASE / "publication-v1"
    output.mkdir(exist_ok=False)
    inventory, members = [], {}
    for path in sorted(BASE.rglob("*")):
        if not path.is_file() or output in path.parents:
            continue
        name = str(path.relative_to(root))
        included = path.suffix != ".mp4" and path.suffix != ".png"
        inventory.append({"path": name, "sha256": sha(path), "included": included})
        if included:
            members[name] = path
    for path in (
        *root.glob("benchmark/da_plcbf_three_panel*.py"),
        root / "benchmark/da_plcbf_wind_bias_diagnosis.py",
        root / "crazyflow/safety/da_plcbf/contact_replay.py",
        root / "benchmark/da_plcbf_actuator_video.py",
        root / "docs/da_plcbf_three_panel_wind_report.md",
    ):
        members[str(path.relative_to(root))] = path
    archive = output / "review-evidence.tar.gz"
    expected = {name: sha(path) for name, path in members.items()}
    with tarfile.open(archive, "w:gz") as tar:
        for name, path in members.items():
            tar.add(path, arcname=name, recursive=False)
    with tarfile.open(archive, "r:gz") as tar:
        assert set(tar.getnames()) == set(expected)
        for member in tar:
            stream = tar.extractfile(member)
            assert stream is not None
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected[member.name]
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "archive_sha256": sha(archive),
                "all_members_verified": True,
                "members": expected,
                "scope": "Includes new contact/probe arrays; videos and PNGs hash-indexed.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
