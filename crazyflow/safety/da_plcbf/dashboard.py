"""Deterministic offline dashboard rendering and strict MP4 validation.

The production simulator is never invoked here.  Frames are a pure function of an immutable trace
and are encoded by the project-pinned ``imageio-ffmpeg`` binary with H.264 settings recorded in run
provenance.  Replay equality is evaluated on decoded RGB frames, avoiding container timestamps.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio_ffmpeg
import numpy as np

from crazyflow.safety.da_plcbf.artifacts import ImmutableTrace, file_sha256

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True, slots=True)
class VideoValidation:
    """Decoded MP4 metadata and content checks."""

    codec: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    unique_frame_count: int
    maximum_mean_frame_change: float
    decoded_frames_sha256: str
    file_sha256: str


def render_dashboard(
    trace: ImmutableTrace,
    path: str | os.PathLike[str],
    *,
    fps: float = 10.0,
    size: tuple[int, int] = (640, 360),
) -> VideoValidation:
    """Render one deterministic dashboard frame per saved trace node to an H.264 MP4."""
    if not isinstance(trace, ImmutableTrace):
        raise TypeError("trace must be an ImmutableTrace")
    trace.validate()
    frame_rate = _positive_finite(fps, "fps")
    width, height = _validate_size(size)
    destination = Path(path)
    if destination.suffix.lower() != ".mp4":
        raise ValueError("dashboard path must end in .mp4")
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)

    temporary = destination.parent / f".{destination.name}.encoding.tmp.mp4"
    if temporary.exists():
        raise FileExistsError(temporary)
    writer = None
    try:
        writer = imageio_ffmpeg.write_frames(
            str(temporary),
            (width, height),
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=frame_rate,
            quality=None,
            bitrate=None,
            codec="libx264",
            macro_block_size=2,
            ffmpeg_log_level="error",
            ffmpeg_timeout=30,
            output_params=[
                "-preset",
                "medium",
                "-crf",
                "18",
                "-threads",
                "1",
                "-movflags",
                "+faststart",
                "-metadata",
                "creation_time=1970-01-01T00:00:00Z",
            ],
        )
        writer.send(None)
        for frame in dashboard_frames(trace, size=(width, height)):
            writer.send(frame)
        writer.close()
        writer = None
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
    return validate_mp4(
        destination,
        expected_codec="h264",
        expected_size=(width, height),
        expected_fps=frame_rate,
        expected_frame_count=trace.steps,
    )


def dashboard_frames(
    trace: ImmutableTrace, *, size: tuple[int, int] = (640, 360)
) -> Iterator[np.ndarray]:
    """Yield pure NumPy RGB frames for deterministic replay or an alternative encoder."""
    if not isinstance(trace, ImmutableTrace):
        raise TypeError("trace must be an ImmutableTrace")
    width, height = _validate_size(size)
    x_index, y_index, _ = _position_indices(trace)
    true_xy = trace.true_state[:, [x_index, y_index]]
    estimated_xy = trace.estimated_state[:, [x_index, y_index]]
    all_xy = np.concatenate((true_xy, estimated_xy), axis=0)
    lower = np.min(all_xy, axis=0)
    upper = np.max(all_xy, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    lower -= 0.15 * span + 0.1
    upper += 0.15 * span + 0.1
    minimum_margin = np.min(trace.hard_barriers, axis=1)
    margin_extent = max(float(np.max(np.abs(minimum_margin))), 1e-9)
    executed_policy_values = trace.policy_values[trace.executed_control]
    policy_extent = max(float(np.max(np.abs(executed_policy_values))), 1e-9)

    for step in range(trace.steps):
        frame = np.full((height, width, 3), (14, 19, 29), dtype=np.uint8)
        _fill_rect(frame, 12, 12, int(width * 0.64), height - 12, (23, 31, 45))
        _fill_rect(frame, int(width * 0.66), 12, width - 12, int(height * 0.48), (23, 31, 45))
        _fill_rect(
            frame, int(width * 0.66), int(height * 0.52), width - 12, height - 12, (23, 31, 45)
        )

        world_box = (26, 26, int(width * 0.62), height - 28)
        for fraction in (0.25, 0.5, 0.75):
            x = round(world_box[0] + fraction * (world_box[2] - world_box[0]))
            y = round(world_box[1] + fraction * (world_box[3] - world_box[1]))
            _draw_line(frame, x, world_box[1], x, world_box[3], (42, 53, 69))
            _draw_line(frame, world_box[0], y, world_box[2], y, (42, 53, 69))
        true_pixels = _map_points(true_xy[: step + 1], lower, upper, world_box)
        estimated_pixels = _map_points(estimated_xy[: step + 1], lower, upper, world_box)
        _draw_polyline(frame, estimated_pixels, (255, 177, 66), thickness=2)
        _draw_polyline(frame, true_pixels, (72, 202, 228), thickness=3)
        _draw_circle(frame, *true_pixels[-1], 5, (237, 246, 255))
        _draw_circle(frame, *estimated_pixels[-1], 3, (255, 177, 66))

        policy_left = int(width * 0.69)
        policy_right = width - 26
        policy_top = 28
        policy_bottom = int(height * 0.44)
        values = trace.policy_values[step]
        cell_height = max(1, (policy_bottom - policy_top) // len(values))
        midpoint = (policy_left + policy_right) // 2
        _draw_line(frame, midpoint, policy_top, midpoint, policy_bottom, (90, 100, 116))
        if bool(trace.executed_control[step]):
            for index, value in enumerate(values):
                normalized = float(np.clip(value / policy_extent, -1.0, 1.0))
                y0 = policy_top + index * cell_height
                y1 = min(y0 + max(1, cell_height - 1), policy_bottom)
                extent = round(abs(normalized) * (policy_right - policy_left) * 0.48)
                color = (57, 192, 122) if normalized >= 0.0 else (232, 79, 91)
                if normalized >= 0.0:
                    _fill_rect(frame, midpoint, y0, midpoint + extent, y1, color)
                else:
                    _fill_rect(frame, midpoint - extent, y0, midpoint, y1, color)
                if index == int(trace.selected_policy[step]):
                    _draw_rect(frame, policy_left, y0, policy_right, y1, (255, 255, 255))
        else:
            # A terminal no-control row carries exact zero sentinels.  Render a literal label rather
            # than green bars, which would falsely present those zeros as hard-safe policies.
            _fill_rect(frame, policy_left, policy_top, policy_right, policy_bottom, (35, 45, 60))
            _draw_rect(frame, policy_left, policy_top, policy_right, policy_bottom, (130, 138, 151))
            _draw_centered_bitmap_text(
                frame, "TERMINAL", policy_left, policy_right, policy_top + 8, (237, 246, 255)
            )
            _draw_centered_bitmap_text(
                frame, "NO CONTROL", policy_left, policy_right, policy_top + 20, (255, 177, 66)
            )

        graph_left = int(width * 0.69)
        graph_right = width - 26
        graph_top = int(height * 0.56)
        graph_bottom = height - 42
        zero_y = (graph_top + graph_bottom) // 2
        _draw_line(frame, graph_left, zero_y, graph_right, zero_y, (130, 138, 151))
        x_values = np.linspace(graph_left, graph_right, trace.steps).round().astype(int)
        y_values = (
            (zero_y - minimum_margin / margin_extent * (graph_bottom - graph_top) * 0.46)
            .round()
            .astype(int)
        )
        points = np.stack((x_values[: step + 1], y_values[: step + 1]), axis=1)
        _draw_polyline(frame, points, (57, 192, 122), thickness=2)
        cursor_x = int(x_values[step])
        _draw_line(frame, cursor_x, graph_top, cursor_x, graph_bottom, (237, 246, 255))

        status_color = (57, 192, 122)
        if bool(trace.failure[step]):
            status_color = (232, 79, 91)
        elif bool(trace.degraded[step]):
            status_color = (255, 177, 66)
        _fill_rect(frame, graph_left, height - 34, graph_right, height - 22, status_color)
        progress_right = round(12 + (width - 24) * (step + 1) / trace.steps)
        _fill_rect(frame, 12, height - 8, progress_right, height - 4, (72, 202, 228))
        yield frame


def validate_mp4(
    path: str | os.PathLike[str],
    *,
    expected_codec: str = "h264",
    expected_size: tuple[int, int] | None = None,
    expected_fps: float | None = None,
    expected_frame_count: int | None = None,
) -> VideoValidation:
    """Decode and validate codec, dimensions, timing, frame count, and non-static content."""
    source = Path(path)
    if expected_codec != "h264":
        raise ValueError("the version-1 dashboard schema requires decoded h264")
    if expected_frame_count is not None and (
        isinstance(expected_frame_count, bool)
        or not isinstance(expected_frame_count, int)
        or expected_frame_count < 2
    ):
        raise ValueError("expected_frame_count must be an integer of at least two")
    if source.suffix.lower() != ".mp4" or source.is_symlink() or not source.is_file():
        raise ValueError("video must be a regular non-symlink MP4 file")
    reader = imageio_ffmpeg.read_frames(str(source), pix_fmt="rgb24", bits_per_pixel=24)
    try:
        metadata = next(reader)
    except (OSError, RuntimeError, StopIteration) as error:
        reader.close()
        raise ValueError("MP4 metadata could not be decoded") from error
    try:
        if not isinstance(metadata, Mapping):
            raise ValueError("MP4 decoder returned invalid metadata")
        codec_text = str(metadata.get("codec", "")).lower()
        codec = "h264" if "h264" in codec_text or "avc" in codec_text else codec_text
        if codec != expected_codec:
            raise ValueError(f"MP4 codec is {codec_text!r}, expected {expected_codec!r}")
        raw_size = metadata.get("size")
        if not isinstance(raw_size, tuple) or len(raw_size) != 2:
            raise ValueError("MP4 has invalid dimensions metadata")
        width, height = (int(raw_size[0]), int(raw_size[1]))
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("MP4 dimensions must be positive and even")
        if expected_size is not None and (width, height) != _validate_size(expected_size):
            raise ValueError("MP4 dimensions do not match expected dimensions")
        fps = _positive_finite(metadata.get("fps"), "decoded MP4 fps")
        if expected_fps is not None and not math.isclose(
            fps, _positive_finite(expected_fps, "expected_fps"), rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("MP4 fps does not match expected fps")
    except Exception:
        reader.close()
        raise

    digest = hashlib.sha256(b"crazyflow.da_plcbf.decoded-video.v1\0")
    unique_hashes: set[bytes] = set()
    previous: np.ndarray | None = None
    maximum_change = 0.0
    frame_count = 0
    try:
        for raw_frame in reader:
            if len(raw_frame) != width * height * 3:
                raise ValueError("decoded MP4 frame has an unexpected byte size")
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(height, width, 3)
            frame_digest = hashlib.sha256(raw_frame).digest()
            unique_hashes.add(frame_digest)
            digest.update(frame_count.to_bytes(8, "little"))
            digest.update(raw_frame)
            if previous is not None:
                change = float(np.mean(np.abs(frame.astype(np.int16) - previous.astype(np.int16))))
                maximum_change = max(maximum_change, change)
            previous = np.array(frame, copy=True)
            frame_count += 1
    except (OSError, RuntimeError) as error:
        raise ValueError("MP4 frame decoding failed") from error
    finally:
        reader.close()
    if frame_count < 2:
        raise ValueError("MP4 must contain at least two frames")
    if expected_frame_count is not None and frame_count != int(expected_frame_count):
        raise ValueError("MP4 frame count does not match expected frame count")
    if len(unique_hashes) < 2 or maximum_change <= 0.1:
        raise ValueError("MP4 is static or its frame changes are below the validation threshold")
    duration = frame_count / fps
    metadata_duration = metadata.get("duration")
    if metadata_duration is not None:
        decoded_duration = _positive_finite(metadata_duration, "decoded MP4 duration")
        if not math.isclose(decoded_duration, duration, rel_tol=0.0, abs_tol=1.0 / fps):
            raise ValueError("MP4 duration metadata is inconsistent with frame count and fps")
    return VideoValidation(
        codec=codec,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
        unique_frame_count=len(unique_hashes),
        maximum_mean_frame_change=maximum_change,
        decoded_frames_sha256=digest.hexdigest(),
        file_sha256=file_sha256(source),
    )


def verify_dashboard_replay(
    trace: ImmutableTrace,
    reference_video: str | os.PathLike[str],
    *,
    fps: float,
    size: tuple[int, int],
) -> VideoValidation:
    """Re-render a dashboard and require byte-identical decoded RGB frames."""
    reference = validate_mp4(
        reference_video,
        expected_codec="h264",
        expected_size=size,
        expected_fps=fps,
        expected_frame_count=trace.steps,
    )
    with tempfile.TemporaryDirectory(prefix="crazyflow-da-plcbf-replay-") as directory:
        replay_path = Path(directory) / "replay.mp4"
        replay = render_dashboard(trace, replay_path, fps=fps, size=size)
    if not hmac.compare_digest(reference.decoded_frames_sha256, replay.decoded_frames_sha256):
        raise ValueError("dashboard replay decoded frames are not deterministic")
    return replay


def video_manifest_record(
    video_path: str | os.PathLike[str],
    source_trace_path: str,
    validation: VideoValidation,
    *,
    run_directory: str | os.PathLike[str],
) -> dict[str, Any]:
    """Create a manifest-ready video record from a validated encoded dashboard."""
    root = Path(run_directory).resolve()
    relative = Path(video_path).resolve().relative_to(root).as_posix()
    return {
        "path": relative,
        "source_trace_path": source_trace_path,
        "sha256": validation.file_sha256,
        "codec": validation.codec,
        "width": validation.width,
        "height": validation.height,
        "fps": validation.fps,
        "frame_count": validation.frame_count,
        "duration_seconds": validation.duration_seconds,
        "decoded_frames_sha256": validation.decoded_frames_sha256,
    }


def _position_indices(trace: ImmutableTrace) -> tuple[int, int, int]:
    names = [str(value) for value in trace.state_names]
    for candidates in (("position_x", "position_y", "position_z"), ("x", "y", "z")):
        if all(name in names for name in candidates):
            return tuple(names.index(name) for name in candidates)  # type: ignore[return-value]
    return (0, 1, 2)


def _validate_size(size: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(size, tuple) or len(size) != 2:
        raise TypeError("size must be a (width, height) tuple")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in size):
        raise TypeError("dashboard dimensions must be integers")
    width, height = size
    if width < 320 or height < 180 or width % 2 or height % 2:
        raise ValueError("dashboard dimensions must be even and at least 320x180")
    return width, height


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def _map_points(
    points: np.ndarray, lower: np.ndarray, upper: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    normalized = (points - lower) / (upper - lower)
    x = box[0] + normalized[:, 0] * (box[2] - box[0])
    y = box[3] - normalized[:, 1] * (box[3] - box[1])
    return np.stack((np.rint(x), np.rint(y)), axis=1).astype(int)


def _fill_rect(
    frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]
) -> None:
    if max(x0, x1) < 0 or min(x0, x1) >= frame.shape[1]:
        return
    if max(y0, y1) < 0 or min(y0, y1) >= frame.shape[0]:
        return
    left = max(0, min(frame.shape[1] - 1, min(x0, x1)))
    right = max(0, min(frame.shape[1] - 1, max(x0, x1)))
    top = max(0, min(frame.shape[0] - 1, min(y0, y1)))
    bottom = max(0, min(frame.shape[0] - 1, max(y0, y1)))
    if left <= right and top <= bottom:
        frame[top : bottom + 1, left : right + 1] = color


def _draw_rect(
    frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]
) -> None:
    _draw_line(frame, x0, y0, x1, y0, color)
    _draw_line(frame, x1, y0, x1, y1, color)
    _draw_line(frame, x1, y1, x0, y1, color)
    _draw_line(frame, x0, y1, x0, y0, color)


def _draw_polyline(
    frame: np.ndarray, points: np.ndarray, color: tuple[int, int, int], *, thickness: int
) -> None:
    if len(points) == 1:
        _draw_circle(frame, int(points[0, 0]), int(points[0, 1]), thickness, color)
        return
    for first, second in zip(points[:-1], points[1:], strict=True):
        for offset in range(-(thickness // 2), thickness - thickness // 2):
            _draw_line(
                frame,
                int(first[0]),
                int(first[1] + offset),
                int(second[0]),
                int(second[1] + offset),
                color,
            )


def _draw_line(
    frame: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]
) -> None:
    count = max(abs(x1 - x0), abs(y1 - y0)) + 1
    x = np.rint(np.linspace(x0, x1, count)).astype(int)
    y = np.rint(np.linspace(y0, y1, count)).astype(int)
    valid = (x >= 0) & (x < frame.shape[1]) & (y >= 0) & (y < frame.shape[0])
    frame[y[valid], x[valid]] = color


def _draw_circle(
    frame: np.ndarray, x: int, y: int, radius: int, color: tuple[int, int, int]
) -> None:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = xx * xx + yy * yy <= radius * radius
    x0, x1 = max(0, x - radius), min(frame.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(frame.shape[0], y + radius + 1)
    mask_x0 = x0 - (x - radius)
    mask_y0 = y0 - (y - radius)
    cropped = mask[mask_y0 : mask_y0 + y1 - y0, mask_x0 : mask_x0 + x1 - x0]
    frame[y0:y1, x0:x1][cropped] = color


_BITMAP_FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    " ": ("00000",) * 7,
}


def _draw_centered_bitmap_text(
    frame: np.ndarray, text: str, left: int, right: int, top: int, color: tuple[int, int, int]
) -> None:
    """Draw the small fixed terminal label without introducing a font/runtime dependency."""
    glyph_width = 5
    spacing = 1
    width = len(text) * glyph_width + max(len(text) - 1, 0) * spacing
    start = left + max((right - left + 1 - width) // 2, 0)
    for character_index, character in enumerate(text):
        glyph = _BITMAP_FONT[character]
        x_origin = start + character_index * (glyph_width + spacing)
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit == "1":
                    _fill_rect(
                        frame, x_origin + column, top + row, x_origin + column, top + row, color
                    )
