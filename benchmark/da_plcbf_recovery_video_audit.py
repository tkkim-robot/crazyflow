"""Verify encoded dimensions, frame clocks, saved previews and source-linked replay data."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark.da_plcbf_actuator_video import _array_sha, _jsonable, _sample_audit, load_episode
from benchmark.da_plcbf_recovery_analysis import BASE, sha


def main() -> None:
    results = []
    for pair, methods in (("libraries", ("PD_F", "F2")), ("adaptation", ("F2", "A_BAL"))):
        directory = BASE / "wind-video-v1" / pair
        manifest = json.loads((directory / "manifest.json").read_text())
        movie = Path(manifest["video"])
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
        assert (probe["width"], probe["height"], probe["avg_frame_rate"]) == (1600, 900, "20/1")
        frames = [
            json.loads(line) for line in (directory / "frame_audit.jsonl").read_text().splitlines()
        ]
        assert len(frames) == int(probe["nb_read_frames"]) == 861
        assert abs(float(probe["duration"]) - 43.05) < 1e-6
        episodes = [
            load_episode(BASE / "wind-demo-v1" / m / "attempt-00", expected_method=m)
            for m in methods
        ]
        for index, frame in enumerate(frames):
            assert frame["frame"] == index
            assert abs(frame["physical_time_seconds"] - index / 20) < 1e-10
            for key, episode in zip(("F2", "A"), episodes, strict=True):
                sample = episode.sample(frame["physical_time_seconds"])
                expected = _jsonable(_sample_audit(episode, sample))
                assert frame[key] == expected, (pair, index, key)
        previews = sorted(directory.glob("frame-*.png"))
        for path in previews:
            index = int(path.stem.split("-")[1])
            assert _array_sha(np.asarray(Image.open(path))) == frames[index]["raw_rgb_sha256"]
        assert frames[0]["A"]["preflight"] and not frames[380]["A"]["preflight"]
        assert frames[60]["A"]["wind_velocity_mps"] == [1.6, 0.8, 0]
        assert frames[220]["A"]["wind_velocity_mps"] == [0, 0, 0]
        assert frames[460]["A"]["wind_velocity_mps"] == [1.2, -0.6, 0]
        assert frames[580]["A"]["wind_velocity_mps"] == [-1, 0.8, 0]
        assert frames[740]["A"]["wind_velocity_mps"] == [0, 0, 0]
        if pair == "libraries":
            stopped = [f["F2"] for f in frames if f["F2"]["contact_stopped"]]
            assert stopped and len({f["state17_sha256"] for f in stopped}) == 1
            assert any(f["F2"]["executing_committed_backup"] for f in frames)
        else:
            assert frames[-1]["A"]["online_updates_used"] == 1074
            assert all(f["F2"]["online_updates_used"] == 0 for f in frames)
        results.append(
            {
                "pair": pair,
                "video": str(movie),
                "sha256": sha(movie),
                "ffprobe": probe,
                "replayed_frames_verified": len(frames),
                "preview_hashes_verified": len(previews),
                "wind_and_stage_boundaries_verified": True,
                "full_decode_passed": manifest["full_decode_passed"],
            }
        )
    output = BASE / "wind-video-v1/validation.json"
    assert not output.exists()
    output.write_text(json.dumps({"videos": results}, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
