"""The cheap discovery screen must retain the runtime's augmented, swept geometry."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.da_plcbf_case_attribution import run_branch
from benchmark.da_plcbf_case_discovery import cached_swept_values
from crazyflow.safety.da_plcbf.continuous_version_a import (
    RuntimeObstacleTrajectories,
    runtime_policy_values,
)


def test_cached_geometry_matches_runtime_with_moving_obstacles_and_nominal() -> None:
    rng = np.random.default_rng(831)
    positions = rng.normal(size=(5, 9, 3)).astype(np.float32)
    centers = rng.normal(size=(3, 9, 2, 3)).astype(np.float32)
    radii = rng.uniform(0.1, 0.5, (3, 2)).astype(np.float32)
    states = np.zeros((5, 9, 13), np.float32)
    states[..., :3] = positions
    states[..., 6] = 1
    values, _ = cached_swept_values(positions, centers, radii + 0.256)
    for index in range(3):
        runtime = runtime_policy_values(
            jnp.asarray(states),
            RuntimeObstacleTrajectories(
                jnp.asarray(centers[index]), jnp.asarray(radii[index]), jnp.ones((9, 2), dtype=bool)
            ),
            obstacle_clearance=0.15,
            ego_radius=0.106,
        )
        np.testing.assert_allclose(values[index], runtime.values, atol=7e-7)


def test_augmented_value_cannot_omit_safe_nominal_or_initial_state() -> None:
    # Nominal at x=2 is safe; the only fallback crosses the obstacle between safe endpoints.
    positions = np.asarray([[[2, 0, 0], [2, 0, 0]], [[-1, 0, 0], [1, 0, 0]]], float)
    centers = np.zeros((1, 2, 1, 3))
    values, clearance = cached_swept_values(positions, centers, np.asarray([[0.3]]))
    assert values[0, 1] < 0 < np.max(values[0])
    np.testing.assert_allclose(clearance[0], [1.7, -0.3])
    # An initially intersecting common state invalidates every future path.
    positions[:, 0] = 0
    values, _ = cached_swept_values(positions, centers, np.asarray([[0.3]]))
    assert np.max(values[0]) < 0


def test_branch_rejects_future_snapshot_before_any_control_or_training() -> None:
    world = SimpleNamespace(initial_state_time_seconds=4.0)
    with pytest.raises(ValueError, match="future learner snapshot"):
        run_branch(world, None, None, None, end=7.0, snapshot_available=4.04)


def test_branch_rejects_changed_prediction_hold_before_execution() -> None:
    world = SimpleNamespace(
        initial_state_time_seconds=4.0,
        config=SimpleNamespace(dt=.02, control_interval_steps=2),
    )
    checkpoint = SimpleNamespace(config=SimpleNamespace(dt=.02, control_interval_steps=1))
    with pytest.raises(ValueError, match="preserve checkpoint prediction and command cadence"):
        run_branch(world, checkpoint, None, None, end=7.0, snapshot_available=4.0)
