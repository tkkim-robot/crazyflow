"""Independent scientific-isolation checks for the corrected continuous controller."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.continuous_version_a import (
    RuntimeObstacleTrajectories,
    conservative_smooth_policy_values,
    runtime_policy_values,
)
from crazyflow.safety.da_plcbf.version_a_barriers import (
    RigidBodySafetySet,
    VersionABarrierConfig,
    VersionAModel,
    continuous_safety_halfspaces,
)


def test_smooth_min_is_conservative_and_differentiable_at_duplicate_and_competing_minima() -> None:
    """Tied obstacle branches and duplicate node/segment endpoints must remain usable."""
    obstacles = RuntimeObstacleTrajectories(
        centers=jnp.asarray([[[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]] * 3),
        radii=jnp.asarray([0.2, 0.2]),
        mask=jnp.ones((3, 2), dtype=bool),
    )
    temperature = 0.02

    def evaluate(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        states = jnp.zeros((1, 3, 13)).at[:, :, 6].set(1).at[:, :, 2].set(1)
        states = states.at[:, :, 0].set(x)
        hard = runtime_policy_values(states, obstacles, obstacle_clearance=0.1, ego_radius=0.05)
        smooth = conservative_smooth_policy_values(hard, temperature=temperature)
        return smooth[0], hard.values[0]

    value, gradient = jax.value_and_grad(lambda x: evaluate(x)[0])(jnp.asarray(0.0))
    hard = evaluate(jnp.asarray(0.0))[1]
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, 0.0, atol=1e-6)
    assert value <= hard
    assert hard - value <= temperature * np.log(10) + 1e-6
    for x in (-0.04, 0.04):
        point = jnp.asarray(x)
        derivative = jax.grad(lambda t: evaluate(t)[0])(point)
        finite_difference = (evaluate(point + 1e-3)[0] - evaluate(point - 1e-3)[0]) / 2e-3
        np.testing.assert_allclose(derivative, finite_difference, atol=2e-3)


def test_ego_radius_and_masked_padding_preserve_value_and_gradient() -> None:
    states = jnp.zeros((1, 2, 13)).at[:, :, 6].set(1).at[:, :, 0].set(1)
    obstacles = RuntimeObstacleTrajectories(
        centers=jnp.asarray([[[0.0, 0.0, 0.0], [jnp.nan, jnp.nan, jnp.nan]]] * 2),
        radii=jnp.asarray([0.2, jnp.nan]),
        mask=jnp.asarray([[True, False]] * 2),
    )

    def smooth(rollout: jax.Array) -> jax.Array:
        hard = runtime_policy_values(rollout, obstacles, obstacle_clearance=0.1, ego_radius=0.05)
        return conservative_smooth_policy_values(hard, temperature=0.005)[0]

    hard = runtime_policy_values(states, obstacles, obstacle_clearance=0.1, ego_radius=0.05)
    np.testing.assert_allclose(hard.values, [1.0 - 0.35**2], atol=1e-6)
    assert bool(hard.input_valid[0])
    assert np.all(np.isfinite(jax.grad(smooth)(states)))


def test_disabling_obstacle_hocbf_preserves_all_operational_faces() -> None:
    model = VersionAModel(
        jnp.asarray(1.0),
        jnp.asarray([0.0, 0.0, -9.81]),
        jnp.eye(3),
        jnp.eye(3),
        jnp.zeros((3, 3)),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros(3),
    )
    state = jnp.zeros(13).at[2].set(1).at[6].set(1)
    safety = RigidBodySafetySet(
        jnp.asarray([[0.0, 0.0, 1.0]]),
        jnp.asarray([0.2]),
        jnp.asarray([True]),
        jnp.asarray([-5.0, -5.0, 0.0]),
        jnp.asarray([5.0, 5.0, 5.0]),
        jnp.asarray(5.0),
        jnp.asarray(10.0),
        jnp.asarray(0.7),
    )
    full = continuous_safety_halfspaces(state, model, safety, VersionABarrierConfig())
    operational = continuous_safety_halfspaces(
        state, model, safety, VersionABarrierConfig(include_obstacle_hocbf=False)
    )
    assert not bool(full.domain_valid)
    assert bool(operational.domain_valid)
    np.testing.assert_array_equal(operational.enabled, [False] + [True] * 9)
    np.testing.assert_allclose(operational.matrix[1:], full.matrix[1:])
    np.testing.assert_allclose(operational.upper_bound[1:], full.upper_bound[1:])
    np.testing.assert_array_equal(operational.matrix[0], np.zeros(4))


def test_submillimetre_swept_segment_does_not_skip_interior_collision() -> None:
    states = jnp.zeros((1, 2, 13)).at[:, :, 6].set(1)
    states = states.at[0, :, 0].set(jnp.asarray([-1e-4, 1e-4]))
    obstacles = RuntimeObstacleTrajectories(
        centers=jnp.zeros((2, 1, 3)), radii=jnp.asarray([5e-5]), mask=jnp.ones((2, 1), dtype=bool)
    )
    values = runtime_policy_values(states, obstacles, obstacle_clearance=0.0)
    assert bool(values.input_valid[0])
    assert float(values.values[0]) < 0.0
    np.testing.assert_allclose(values.values[0], -((5e-5) ** 2), rtol=1e-5, atol=1e-12)
