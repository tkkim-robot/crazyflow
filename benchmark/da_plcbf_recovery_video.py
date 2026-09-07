"""Render one complete wind demonstration comparison with authenticated replay."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import imageio_ffmpeg

from benchmark.da_plcbf_actuator_video import ActuatorVideoConfig, render_pair
from benchmark.da_plcbf_recovery_diagnosis import BASE, sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=("libraries", "adaptation"), required=True)
    args = parser.parse_args()
    left, right, left_label, right_label = (
        ("PD_F", "F2", "Handcrafted · frozen", "Learned · frozen")
        if args.pair == "libraries"
        else ("F2", "A_BAL", "Learned · frozen", "Learned · adaptive")
    )
    output = BASE / "wind-video-v1" / args.pair
    output.parent.mkdir(exist_ok=True)
    config = ActuatorVideoConfig(
        camera_azimuth=0.0,
        left_method=left,
        right_method=right,
        left_label=left_label,
        right_label=right_label,
        allow_different_learning_contract=True,
        allow_different_libraries=args.pair == "libraries",
    )
    movie = render_pair(
        BASE / "wind-demo-v1" / left / "attempt-00",
        BASE / "wind-demo-v1" / right / "attempt-00",
        output,
        config,
    )
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(movie), "-f", "null", "-"],
        check=True,
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "video": str(movie),
                "sha256": sha(movie),
                "config": asdict(config),
                "full_decode_passed": True,
                "driver_sha256": sha(Path(__file__)),
                "scope": "Fixed development demonstration with preflight and navigation wind.",
            },
            indent=2,
        )
        + "\n"
    )
    print(str(movie), flush=True)


if __name__ == "__main__":
    main()
