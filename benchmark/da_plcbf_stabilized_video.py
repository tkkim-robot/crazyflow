"""The primary hierarchy: handcrafted frozen, learned frozen, learned adaptive."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import imageio_ffmpeg

from benchmark.da_plcbf_actuator_video import ActuatorVideoConfig, render_pair
from benchmark.da_plcbf_numerical_stabilization_analysis import BASE, sha


def main() -> None:
    output = BASE / "primary-comparison-video-v1"
    output.mkdir(exist_ok=False)
    directory = BASE / "development-36-v3"
    videos = []
    for name, left, right, left_label, right_label, different in (
        ("initial_libraries", "PD_F", "F2", "Handcrafted · frozen", "Learned · frozen", True),
        ("neural_adaptation", "F2", "A_BAL", "Learned · frozen", "Learned · adaptive", False),
    ):
        config = ActuatorVideoConfig(
            camera_azimuth=0.0,
            left_method=left,
            right_method=right,
            left_label=left_label,
            right_label=right_label,
            allow_different_learning_contract=True,
            allow_different_libraries=different,
        )
        paths = [
            directory / f"harm__{method}__matched__inward" / "attempt-00"
            for method in (left, right)
        ]
        movie = render_pair(*paths, output / name, config)
        videos.append({"video": str(movie), "sha256": sha(movie), "config": asdict(config)})
    playlist = output / "segments.txt"
    playlist.write_text("".join("file '" + row["video"] + "'\n" for row in videos))
    movie = output / "handcrafted-learned-adaptive.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-n",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(playlist),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(movie),
        ],
        check=True,
    )
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(movie), "-f", "null", "-"],
        check=True,
    )
    manifest = {
        "video": str(movie),
        "sha256": sha(movie),
        "segments": videos,
        "full_decode_passed": True,
        "source_sha256": sha(Path(__file__)),
        "scope": (
            "Fixed historical harmful development scene; corrected learner and QP; "
            "no outcome selection."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(str(movie), flush=True)


if __name__ == "__main__":
    main()
