"""Restore the recorded wind-on/off preflight before the unchanged collision-case video.

The two experiments remain separate. The intro shows persistent deterministic learning;
the case retains its original paced state/checkpoint and every original decoded video frame.
No controller, learner, obstacle, physics, or source artifact is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BASE = Path("artifacts/da_plcbf")
PREFLIGHT = BASE / "hover-explanation-20260905/hover-wind-payload-navigation"
CASE = BASE / "closed-loop-search-20260905/videos/paced-collision-v2/comparison.mp4"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BOLD = FONT.with_name("DejaVuSans-Bold.ttf")
FPS, PREFLIGHT_FRAMES, TRANSITION_FRAMES = 20, 380, 60


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run(arguments: list[str]) -> str:
    result = subprocess.run(arguments, capture_output=True, text=True, check=True)
    return result.stdout


def probe(path: Path) -> dict:
    return json.loads(
        run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    )


def frame_hashes(path: Path) -> list[str]:
    """Fully decode video frames; compare pixels, not container headers or timestamps."""
    rows = run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "framemd5",
            "-",
        ]
    )
    return [
        line.rsplit(",", 1)[1].strip() for line in rows.splitlines() if not line.startswith("#")
    ]


def text_filter(
    path: Path,
    *,
    y: int,
    size: int,
    x: str = "28",
    bold: bool = False,
    color: str = "white",
    enable: str | None = None,
) -> str:
    # Text files avoid interpreting explanatory punctuation as FFmpeg filter syntax.
    value = (
        f"drawtext=fontfile={BOLD if bold else FONT}:textfile={path}:"
        f"fontsize={size}:fontcolor={color}:x={x}:y={y}"
    )
    return value + (f":enable='{enable}'" if enable else "")


def compose(output: Path, preflight: Path, case: Path) -> Path:
    output, preflight, case = output.resolve(), preflight.resolve(), case.resolve()
    source = preflight / "navigation_comparison_demo.mp4"
    for path in (source, case, FONT, BOLD):
        if not path.is_file():
            raise FileNotFoundError(path)
    # Paths are also used inside filter/concat grammars, independently of shell quoting.
    if any(character in str(output) for character in "'\\:;,[]"):
        raise ValueError("output path contains reserved FFmpeg filter characters")
    if output.exists():
        raise FileExistsError("refusing to overwrite a composed video or its provenance")
    metadata = json.loads((preflight / "navigation_comparison.json").read_text())
    world = metadata["summary"]["world"]["config"]
    if [event["time_seconds"] for event in world["wind_events"]] != [3.0, 11.0]:
        raise ValueError("preflight captions require the recorded 3 s / 11 s wind schedule")
    if world["wind_events"][1]["velocity"] != [0.0, 0.0, 0.0]:
        raise ValueError("preflight must actually remove the wind")
    if world["wind_events"][0]["velocity"] != [1.6, 0.8, 0.0]:
        raise ValueError("preflight captions require the recorded wind vector")
    if any(event["time_seconds"] < 19 for event in world["payload_events"]):
        raise ValueError("preflight must exclude payload changes")
    if metadata["summary"]["config"]["navigation_start_seconds"] < 19:
        raise ValueError("preflight must contain only hover commands")
    video_sources = {"preflight": source, "case": case}
    hashes = {name: digest(path) for name, path in video_sources.items()}
    streams = {name: probe(path)["streams"][0] for name, path in video_sources.items()}
    for stream in streams.values():
        if (
            stream["codec_name"],
            stream["width"],
            stream["height"],
            stream["pix_fmt"],
            stream["r_frame_rate"],
            stream["time_base"],
        ) != ("h264", 1600, 900, "yuv420p", "20/1", "1/10240"):
            raise ValueError("source must be the original compatible 1600x900, 20 fps H.264 video")
    output.mkdir(parents=True)
    labels = {
        "title": "Preflight | wind adaptation and recovery",
        "calm": "Calm hover | both robots hold position; colored curves predict fallback motions",
        "wind": "Wind on | fixed predictions drift; the adaptive library learns to restore them",
        "off": "Wind off | the learned correction is briefly wrong; adaptation continues",
        "recovery": "Calm again | adaptive fallback motions move back toward the original set",
        "payload": "NO PAYLOAD CHANGE | hovering throughout this preflight",
        "transition_title": "Preflight complete",
        "transition_case": "Now replaying the unchanged case study",
        "transition_reset": (
            "Original initial state and policy checkpoint | case clock restarts at 0 s"
        ),
        "transition_scope": (
            "Preflight learning is a separate demonstration; it is not carried into this replay."
        ),
    }
    texts = output / "captions"
    texts.mkdir()
    for name, label in labels.items():
        (texts / f"{name}.txt").write_text(label)
    filters = [
        f"trim=end_frame={PREFLIGHT_FRAMES}",
        "setpts=PTS-STARTPTS",
        "drawbox=x=0:y=0:w=1450:h=75:color=0x06121a:t=fill",
        text_filter(texts / "title.txt", y=17, size=23, bold=True),
        "drawbox=x=0:y=816:w=1600:h=28:color=0x06121a:t=fill",
        text_filter(texts / "payload.txt", y=822, size=15, x="(w-tw)/2", color="0xb4c2cd"),
    ]
    for name, start, end in (
        ("calm", 0, 59),
        ("wind", 60, 219),
        ("off", 220, 279),
        ("recovery", 280, 379),
    ):
        filters.append(
            text_filter(texts / f"{name}.txt", y=44, size=17, enable=f"between(n,{start},{end})")
        )
    encoding = [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "17",
        "-profile:v",
        "high",
        "-level:v",
        "4.0",
        "-pix_fmt",
        "yuv420p",
        "-video_track_timescale",
        "10240",
        "-movflags",
        "+faststart",
    ]
    commands = [
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            ",".join(filters),
            *encoding,
            str(output / "preflight.mp4"),
        ]
    ]
    card = [
        text_filter(texts / f"{name}.txt", y=y, size=size, x="(w-tw)/2", bold=bold, color=color)
        for name, y, size, bold, color in (
            ("transition_title", 285, 44, True, "white"),
            ("transition_case", 372, 29, False, "0x66d3d5"),
            ("transition_reset", 438, 21, False, "white"),
            ("transition_scope", 534, 18, False, "0xb4c2cd"),
        )
    ]
    commands.append(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x06121a:s=1600x900:r=20",
            "-frames:v",
            str(TRANSITION_FRAMES),
            "-vf",
            ",".join(card),
            *encoding,
            str(output / "transition.mp4"),
        ]
    )
    for command in commands:
        run(command)
    # Stream-copy the original case: even lossy re-encoding of its pixels is avoided.
    listing = output / "concat.txt"
    if "'" in str(case) or "\\" in str(case):
        raise ValueError("case path contains reserved concat characters")
    listing.write_text(
        "".join(
            f"file '{path}'\n"
            for path in (output / "preflight.mp4", output / "transition.mp4", case)
        )
    )
    video = output / "preflight_then_case.mp4"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listing),
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(video),
    ]
    commands.append(command)
    run(command)
    expected_case = frame_hashes(case)
    actual = frame_hashes(video)
    case_start = PREFLIGHT_FRAMES + TRANSITION_FRAMES
    if len(actual) != case_start + len(expected_case) or actual[case_start:] != expected_case:
        raise ValueError("composed video changed, lost, or duplicated original case frames")
    if any(digest(path) != hashes[name] for name, path in video_sources.items()):
        raise ValueError("a source recording changed during composition")
    report = {
        "scope": (
            "Presentation-only preflight intro; original case and its numerical evidence unchanged"
        ),
        "source_videos": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in video_sources.items()
        },
        "preflight_source_metadata_sha256": digest(preflight / "navigation_comparison.json"),
        "timeline": [
            {"video_seconds": [0, 3], "phase": "calm hover"},
            {"video_seconds": [3, 11], "phase": "wind on", "wind_velocity_m_s": [1.6, 0.8, 0]},
            {"video_seconds": [11, 19], "phase": "wind removed; persistent learning recovers"},
            {"video_seconds": [19, 22], "phase": "explicit separate-experiment transition"},
            {
                "video_seconds": [22, 22 + len(expected_case) / FPS],
                "phase": "unchanged recorded case; its source clock starts at zero",
            },
        ],
        "preflight_trim": {
            "source_seconds": [0, 19],
            "end_exclusive": True,
            "frames": PREFLIGHT_FRAMES,
            "no_payload_or_navigation_segment": True,
        },
        "case_decoded_frames": len(expected_case),
        "all_case_decoded_frames_exact": actual[case_start:] == expected_case,
        "frame_count": len(actual),
        "fps": FPS,
        "duration_seconds": len(actual) / FPS,
        "full_decode_passed": True,
        "video_sha256": digest(video),
        "script_sha256": digest(Path(__file__)),
        "commands": commands,
        "learning_scope": (
            "Archived preflight is deterministic learning; case retains measured paced execution"
        ),
        "reset_scope": (
            "Preflight updates are not transferred to the original case's state or library"
        ),
    }
    (output / "VIDEO_REVIEW.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "case_frame_md5.json").write_text(json.dumps(expected_case, indent=2) + "\n")
    print(
        json.dumps({"video": str(video), "frames": len(actual), "case_frames_exact": True}),
        flush=True,
    )
    return video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    parser.add_argument("--case-video", type=Path, default=CASE)
    args = parser.parse_args()
    compose(args.output, args.preflight, args.case_video)


if __name__ == "__main__":
    main()
