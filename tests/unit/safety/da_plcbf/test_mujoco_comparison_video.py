import numpy as np
import pytest

from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
    ComparisonRenderConfig,
    ComparisonVideoTrace,
    MethodVideoTrace,
    ObstacleTrack,
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
