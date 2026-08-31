from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import imageio_ffmpeg
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.artifact_smoke import synthetic_trace
from crazyflow.safety.da_plcbf.dashboard import (
    dashboard_frames,
    render_dashboard,
    validate_mp4,
    verify_dashboard_replay,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.render


def _terminal_trace() -> object:
    trace = synthetic_trace("d" * 64, steps=5, dt=0.1)
    controls = [
        np.array(value, copy=True)
        for value in (trace.nominal_control, trace.filtered_control, trace.applied_control)
    ]
    for value in controls:
        value[-1] = 0.0
    policy = np.array(trace.policy_values, copy=True)
    training = np.array(trace.training_values, copy=True)
    policy[-1] = 0.0
    training[-1] = 0.0
    selected = np.array(trace.selected_policy, copy=True)
    selected[-1] = -1
    executed = np.ones(trace.steps, dtype=np.bool_)
    executed[-1] = False
    clipped = np.array(trace.clipped, copy=True)
    saturated = np.array(trace.saturated, copy=True)
    clipped[-1] = False
    saturated[-1] = False
    latency = np.array(trace.component_latency_seconds, copy=True)
    latency[-1] = 0.0
    solver = np.array(trace.solver_kkt_residual, copy=True)
    postcheck = np.array(trace.postcheck_residual, copy=True)
    gradient = np.array(trace.gradient_norm, copy=True)
    solver[-1] = postcheck[-1] = gradient[-1] = 0.0
    return replace(
        trace,
        nominal_control=controls[0],
        filtered_control=controls[1],
        applied_control=controls[2],
        executed_control=executed,
        policy_values=policy,
        training_values=training,
        selected_policy=selected,
        solver_kkt_residual=solver,
        postcheck_residual=postcheck,
        clipped=clipped,
        saturated=saturated,
        gradient_norm=gradient,
        component_latency_seconds=latency,
    )


def test_dashboard_frames_and_h264_replay_are_deterministic(tmp_path: Path) -> None:
    trace = synthetic_trace("b" * 64, steps=8, dt=0.1)
    first_frames = tuple(dashboard_frames(trace, size=(320, 180)))
    second_frames = tuple(dashboard_frames(trace, size=(320, 180)))
    assert len(first_frames) == trace.steps
    for first, second in zip(first_frames, second_frames, strict=True):
        np.testing.assert_array_equal(first, second)
        assert first.shape == (180, 320, 3)
        assert first.dtype == np.uint8

    path = tmp_path / "dashboard.mp4"
    validation = render_dashboard(trace, path, fps=8.0, size=(320, 180))
    assert validation.codec == "h264"
    assert (validation.width, validation.height) == (320, 180)
    assert validation.frame_count == trace.steps
    assert validation.fps == pytest.approx(8.0)
    assert validation.duration_seconds == pytest.approx(1.0)
    assert validation.unique_frame_count > 1
    assert validation.maximum_mean_frame_change > 0.1
    replay = verify_dashboard_replay(trace, path, fps=8.0, size=(320, 180))
    assert replay.decoded_frames_sha256 == validation.decoded_frames_sha256


def test_video_validator_rejects_static_content_and_wrong_expectations(tmp_path: Path) -> None:
    path = tmp_path / "static.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(path), (320, 180), fps=5.0, codec="libx264", quality=5, ffmpeg_log_level="error"
    )
    writer.send(None)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    for _ in range(3):
        writer.send(frame)
    writer.close()
    with pytest.raises(ValueError, match="static"):
        validate_mp4(path, expected_frame_count=3)

    trace = synthetic_trace("c" * 64, steps=4, dt=0.1)
    dynamic = tmp_path / "dynamic.mp4"
    render_dashboard(trace, dynamic, fps=5.0, size=(320, 180))
    with pytest.raises(ValueError, match="dimensions"):
        validate_mp4(dynamic, expected_size=(640, 360))
    with pytest.raises(ValueError, match="frame count"):
        validate_mp4(dynamic, expected_frame_count=5)


def test_terminal_frame_labels_no_control_instead_of_drawing_zero_sentinels_safe() -> None:
    trace = _terminal_trace()
    final = tuple(dashboard_frames(trace, size=(320, 180)))[-1]
    policy_panel = final[28:80, int(320 * 0.69) : 320 - 25]

    assert np.any(np.all(policy_panel == np.asarray((237, 246, 255)), axis=-1))
    assert np.any(np.all(policy_panel == np.asarray((255, 177, 66)), axis=-1))
    assert not np.any(np.all(policy_panel == np.asarray((57, 192, 122)), axis=-1))

    # The same numeric zeros are rendered differently when explicitly declared to be a real
    # controller row, proving that the final frame is driven by executed_control rather than value.
    legacy = replace(trace, executed_control=np.ones(trace.steps, dtype=np.bool_))
    legacy_final = tuple(dashboard_frames(legacy, size=(320, 180)))[-1]
    assert not np.array_equal(final, legacy_final)
