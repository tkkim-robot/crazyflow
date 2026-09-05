from dataclasses import replace

import numpy as np
import pytest

from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
    ComparisonRenderConfig,
    ComparisonVideoTrace,
    MethodVideoTrace,
    ObstacleTrack,
    _execution_status,
    _hud,
    _latest_coverage_probe,
    _probe_pause_index,
    _validate_method,
    comparison_video_frames,
)


def _method_trace(position: np.ndarray) -> MethodVideoTrace:
    steps, policies, horizon = len(position), 2, 3
    directions = np.asarray(((0.35, 0.20, 0.0), (0.35, -0.20, 0.0)))
    phase = np.linspace(0.0, 1.0, horizon)
    fallback = (
        position[:, None, None, :] + phase[None, None, :, None] * directions[None, :, None, :]
    )
    nominal = position[:, None, :] + phase[None, :, None] * np.asarray((0.45, 0.0, 0.0))
    return MethodVideoTrace(
        position=position,
        quaternion_xyzw=np.tile(np.asarray((0.0, 0.0, 0.0, 1.0)), (steps, 1)),
        nominal_rollout=nominal,
        fallback_rollouts=fallback,
        fallback_safe=np.ones((steps, policies), dtype=np.bool_),
        selected_policy=np.zeros((steps,), dtype=np.int32),
        selected_rollout=fallback[:, 0],
        intervention_world=np.tile(np.asarray((0.0, 0.1, 0.0)), (steps, 1)),
        intervention_norm=np.full((steps,), 0.1),
        descriptors=np.broadcast_to(directions, (steps, policies, 3)).copy(),
        library_version=np.arange(steps, dtype=np.int32),
        cumulative_gradient_steps=np.arange(steps, dtype=np.int32),
        diversity_loss=np.ones((steps,)),
        descriptor_target_loss=np.ones((steps,)),
        gradient_norm=np.ones((steps,)),
        parameter_update_norm=np.full((steps,), 0.01),
        minimum_library_value=np.full((steps,), 0.2),
        maximum_library_value=np.full((steps,), 0.5),
        selected_policy_value=np.full((steps,), 0.4),
        selected_policy_dual=np.full((steps,), 0.3),
        qp_valid=np.ones((steps,), dtype=np.bool_),
        used_fallback=np.zeros((steps,), dtype=np.bool_),
        degraded=np.zeros((steps,), dtype=np.bool_),
    )


@pytest.mark.unit
@pytest.mark.render
def test_mujoco_comparison_frame_smoke() -> None:
    time = np.asarray((0.0, 0.1))
    position = np.asarray(((0.0, 0.0, 0.8), (0.05, 0.0, 0.8)))
    method = _method_trace(position)
    trace = ComparisonVideoTrace(
        time_seconds=time,
        goal_position=np.asarray((0.9, 0.0, 0.8)),
        obstacles=(
            ObstacleTrack(
                centers=np.tile(np.asarray((0.55, 0.0, 0.8)), (len(time), 1)),
                physical_radius=0.10,
                inflated_radius=0.18,
            ),
        ),
        true_wind=np.asarray(((0.0, 0.0, 0.0), (0.8, 0.2, 0.0))),
        estimated_wind=np.asarray(((0.0, 0.0, 0.0), (0.5, 0.1, 0.0))),
        wind_change_time=0.1,
        descriptor_targets=np.asarray(((0.35, 0.20, 0.0), (0.35, -0.20, 0.0))),
        fixed=method,
        adaptive=method,
    )

    frame = next(
        comparison_video_frames(
            trace, ComparisonRenderConfig(fps=10.0, width=960, height=540, camera_distance=2.8)
        )
    )

    assert frame.shape == (540, 960, 3)
    assert frame.dtype == np.uint8
    assert np.unique(frame.reshape(-1, 3), axis=0).shape[0] > 100


@pytest.mark.unit
def test_renderer_reports_certificate_max_and_actual_execution() -> None:
    method = _method_trace(np.asarray(((0.0, 0.0, 0.8), (0.05, 0.0, 0.8))))
    hud = _hud(method, 0)
    assert "library H = max +0.500" in hud
    assert "selected H +0.400" in hud
    assert "PL-CBF dual 3.00e-01" in hud
    assert "min library" not in hud
    assert _execution_status(method, 0)[0] == "QP INTERVENING"
    fallback = replace(
        method, qp_valid=np.asarray((False, True)), used_fallback=np.asarray((True, False))
    )
    assert _execution_status(fallback, 0)[0] == "EXECUTING FALLBACK"
    degraded = replace(
        method, qp_valid=np.asarray((False, True)), degraded=np.asarray((True, False))
    )
    assert _execution_status(degraded, 0)[0] == "UNCERTIFIED BEST EFFORT"
    no_certificate = replace(method, maximum_library_value=np.asarray((-0.1, 0.2)))
    assert _execution_status(no_certificate, 0)[0] == "NO COLLISION CERTIFICATE"
    legacy = replace(method, maximum_library_value=None, qp_valid=None)
    assert "library H = max unrecorded" in _hud(legacy, 0)
    assert _execution_status(legacy, 0)[0] == "COMMAND STATUS NOT RECORDED"
    assert _execution_status(replace(method, control_mode="nominal"), 0)[0] == "EXECUTING NOMINAL"
    inconsistent = replace(method, used_fallback=np.asarray((True, False)))
    with pytest.raises(ValueError, match="valid QP"):
        _validate_method(inconsistent, 2, "method")


@pytest.mark.unit
def test_same_state_probe_uses_only_measured_past_values_and_explicit_pause() -> None:
    time = np.asarray((0.0, 0.05, 0.08, 0.2))
    method = _method_trace(np.tile(np.asarray((0.0, 0.0, 0.8)), (4, 1)))
    trace = ComparisonVideoTrace(
        time_seconds=time,
        goal_position=np.asarray((1.0, 0.0, 0.8)),
        obstacles=(),
        true_wind=np.zeros((4, 3)),
        estimated_wind=np.zeros((4, 3)),
        wind_change_time=0.2,
        descriptor_targets=np.zeros((2, 3)),
        fixed=method,
        adaptive=method,
        coverage_probes={
            "time_seconds": [0.05, 0.1],
            "fixed_h": [-0.2, -0.1],
            "adaptive_h": [0.3, 0.4],
            "fixed_safe_count": [0, 0],
            "adaptive_safe_count": [1, 2],
            "source": "adaptive state and model",
        },
    )
    trace.validate()
    assert _latest_coverage_probe(trace, 0) is None
    assert _latest_coverage_probe(trace, 1) == 0
    assert _latest_coverage_probe(trace, 2) == 0
    assert _latest_coverage_probe(trace, 3) == 1
    config = ComparisonRenderConfig(probe_pause_time=0.05, probe_pause_seconds=2.0)
    config.validate()
    assert _probe_pause_index(trace, config, np.arange(4)) == 1
    with pytest.raises(ValueError, match="recorded coverage probe"):
        _probe_pause_index(trace, replace(config, probe_pause_time=0.08), np.arange(4))
    with pytest.raises(ValueError, match="sampled video frame"):
        _probe_pause_index(trace, replace(config, probe_pause_time=0.1), np.arange(4))
