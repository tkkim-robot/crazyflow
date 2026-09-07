"""Render two explicitly labelled comparisons on the fixed development anchor.

Each segment is a complete, independently authenticated 14-second comparison.
The second segment restarts physical time at zero and changes both panel labels.
No flight is rerun and no outcome is used to select the scene or method pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import imageio_ffmpeg

from benchmark.da_plcbf_actuator_video import ActuatorVideoConfig, render_pair

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/da_plcbf/actuator-diagnostics-20260906/v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_comparison(campaign: Path, output: Path, *, camera_azimuth: float = 0.0) -> Path:
    """Keep both native videos, per-frame audits, and a checksum-bound concat manifest."""
    records = {}
    for path in sorted(campaign.glob("*/record.json")):
        record = json.loads(path.read_text())
        trial = record["trial"]
        if trial["world_key"] == "structured_30101" and trial["cell_id"] == "eta0.7_lag1_extra0":
            method = trial["arm"]["arm"]
            if method in records:
                raise ValueError(f"duplicate anchor record: {method}")
            records[method] = (path, record)
    output.mkdir(parents=True, exist_ok=False)
    segments = []
    for name, left, right, title in (
        ("learned", "F2", "A_BAL", "Learned"),
        ("pd", "PD_F", "PD_A", "PD"),
    ):
        config = ActuatorVideoConfig(
            camera_azimuth=camera_azimuth,
            left_method=left,
            right_method=right,
            left_label=f"{title} · frozen",
            right_label=f"{title} · adapting",
            allow_different_learning_contract=left == "F2",
        )
        lhs, rhs = records[left], records[right]
        destination = render_pair(
            Path(lhs[1]["episode_directory"]),
            Path(rhs[1]["episode_directory"]),
            output / name,
            config,
        )
        segments.append(
            {
                "name": name,
                "video": str(destination.resolve()),
                "video_sha256": sha(destination),
                "config": asdict(config),
                "source_records_sha256": {str(p.resolve()): sha(p) for p in (lhs[0], rhs[0])},
                "render_summary": json.loads((output / name / "render_summary.json").read_text()),
            }
        )
    playlist = output / "segments.txt"
    playlist.write_text(
        "".join("file '" + row["video"].replace("'", "'\\''") + "'\n" for row in segments)
    )
    destination = output / "learned-and-pd-comparison.mp4"
    command = [
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
        str(destination),
    ]
    subprocess.run(command, check=True)
    manifest = {
        "status": "completed",
        "source_sha256": {str(Path(__file__).resolve()): sha(Path(__file__))},
        "selection": (
            "Fixed review development anchor; no outcome-based scene selection in this iteration"
        ),
        "world_key": "structured_30101",
        "cell_id": "eta0.7_lag1_extra0",
        "segments": segments,
        "segment_boundary": (
            "Independent comparisons; physical timer restarts and method labels change"
        ),
        "video": str(destination.resolve()),
        "video_sha256": sha(destination),
        "concat_command": command,
        "qualification": (
            "Deterministic P0 development flights; no cross-build robustness "
            "or adaptation advantage implied"
        ),
    }
    (output / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=BASE / "library-comparison-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-azimuth", type=float, default=0.0)
    args = parser.parse_args()
    print(
        render_comparison(args.campaign, args.output, camera_azimuth=args.camera_azimuth),
        flush=True,
    )


if __name__ == "__main__":
    main()
