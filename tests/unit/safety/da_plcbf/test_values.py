import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.values import (
    conservative_softmin,
    hard_policy_margins,
    swept_trajectory_constraints,
    training_policy_margins,
    trajectory_constraints,
)


def test_trajectory_constraints_have_safe_positive_signs_and_exact_values() -> None:
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.0, 0.0], [100.0, -100.0]]]),
        obstacle_radii=jnp.array([[1.0, 20.0]]),
        obstacle_mask=jnp.array([[True, False]]),
        arena_lower=jnp.array([[-2.0, -1.0]]),
        arena_upper=jnp.array([[3.0, 2.0]]),
        speed_limit=jnp.array([2.0]),
    )
    states = jnp.array([[[[1.25, 0.0, 0.0, 2.0], [0.5, 0.0, 0.0, 0.0], [3.5, 0.0, 3.0, 0.0]]]])

    constraints = trajectory_constraints(states, scenarios, safety_margin=0.25)

    assert constraints.shape == (1, 1, 3, 7)
    expected_unmasked = np.array(
        [
            [0.0, 0.65, 1.0 / 3.0, 0.35, 2.0 / 3.0, 0.0],
            [-0.6, 0.5, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0],
            [1.8, 1.1, 1.0 / 3.0, -0.1, 2.0 / 3.0, -1.25],
        ]
    )
    np.testing.assert_allclose(
        constraints[0, 0, :, [0, 2, 3, 4, 5, 6]].T, expected_unmasked, rtol=1e-7, atol=1e-7
    )
    assert np.all(np.isposinf(np.asarray(constraints[0, 0, :, 1])))

    assert constraints[0, 0, 0, 0] == pytest.approx(0.0)
    assert constraints[0, 0, 1, 0] < 0
    assert constraints[0, 0, 2, 0] > 0
    assert constraints[0, 0, 0, -1] == pytest.approx(0.0)
    assert constraints[0, 0, 1, -1] > 0
    assert constraints[0, 0, 2, -1] < 0


def test_masked_obstacles_cannot_change_hard_or_training_policy_margins() -> None:
    states = jnp.array([[[[0.0, 0.0, 0.0, 0.0], [0.2, 0.1, 0.1, -0.1]]]])
    common = {
        "obstacle_centers": jnp.array([[[5.0, 5.0], [0.0, 0.0]]]),
        "obstacle_radii": jnp.array([[0.5, 1000.0]]),
        "arena_lower": jnp.array([[-2.0, -2.0]]),
        "arena_upper": jnp.array([[2.0, 2.0]]),
        "speed_limit": jnp.array([1.0]),
    }
    padded = CircleScenarioBatch(obstacle_mask=jnp.array([[True, False]]), **common)
    one_obstacle = CircleScenarioBatch(
        obstacle_centers=common["obstacle_centers"][:, :1],
        obstacle_radii=common["obstacle_radii"][:, :1],
        obstacle_mask=jnp.array([[True]]),
        arena_lower=common["arena_lower"],
        arena_upper=common["arena_upper"],
        speed_limit=common["speed_limit"],
    )

    padded_constraints = trajectory_constraints(states, padded, safety_margin=0.1)
    compact_constraints = trajectory_constraints(states, one_obstacle, safety_margin=0.1)

    np.testing.assert_allclose(
        hard_policy_margins(padded_constraints), hard_policy_margins(compact_constraints)
    )
    np.testing.assert_allclose(
        training_policy_margins(padded_constraints, beta=12.0),
        training_policy_margins(compact_constraints, beta=12.0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_conservative_softmin_is_bounded_by_hard_minimum() -> None:
    values = jnp.array(
        [
            [[2.0, -1.0, 4.0, 0.5], [3.0, 1.0, -0.25, 5.0], [0.0, 2.0, 6.0, 1.0]],
            [[-2.0, 3.0, 1.5, 4.0], [1.0, 0.0, 2.5, 3.0], [5.0, -0.5, 2.0, 7.0]],
        ]
    )
    beta = 7.0

    smooth = conservative_softmin(values, beta=beta, axis=(-2, -1))
    hard = jnp.min(values, axis=(-2, -1))
    lower_bound = hard - np.log(values.shape[-2] * values.shape[-1]) / beta

    assert np.all(np.asarray(smooth) <= np.asarray(hard) + 1e-7)
    assert np.all(np.asarray(smooth) >= np.asarray(lower_bound) - 1e-6)


@pytest.mark.parametrize("shift", [-10_000.0, 10_000.0])
def test_conservative_softmin_is_numerically_stable_and_translation_equivariant(
    shift: float,
) -> None:
    values = jnp.array([-3.0, -0.5, 2.0, 8.0])
    baseline = conservative_softmin(values, beta=25.0)
    shifted = conservative_softmin(values + shift, beta=25.0)

    assert np.isfinite(np.asarray(shifted))
    np.testing.assert_allclose(shifted - shift, baseline, rtol=0.0, atol=5e-4)


def test_conservative_softmin_ignores_positive_infinity_padding() -> None:
    values = jnp.array([7.0, jnp.inf, jnp.inf])

    actual = conservative_softmin(values, beta=10.0)

    np.testing.assert_allclose(actual, 7.0, rtol=0.0, atol=1e-7)


def test_policy_margin_reductions_use_both_time_and_constraint_axes() -> None:
    constraints = jnp.array(
        [
            [[[3.0, 2.0], [1.0, 4.0]], [[-1.0, 7.0], [5.0, 6.0]]],
            [[[0.5, 8.0], [2.0, 3.0]], [[4.0, 1.5], [2.5, 9.0]]],
        ]
    )

    hard = hard_policy_margins(constraints)
    smooth = training_policy_margins(constraints, beta=20.0)

    np.testing.assert_array_equal(hard, np.array([[1.0, -1.0], [0.5, 1.5]]))
    assert np.all(np.asarray(smooth) <= np.asarray(hard) + 1e-7)
    assert hard.shape == smooth.shape == (2, 2)


def test_scenario_batch_is_a_jax_pytree_usable_through_jit() -> None:
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.zeros((2, 3, 2)),
        obstacle_radii=jnp.ones((2, 3)),
        obstacle_mask=jnp.ones((2, 3), dtype=bool),
        arena_lower=-jnp.ones((2, 2)),
        arena_upper=jnp.ones((2, 2)),
        speed_limit=jnp.array([1.0, 2.0]),
    )

    doubled = jax.jit(lambda batch: batch.replace(speed_limit=2 * batch.speed_limit))(scenarios)

    assert isinstance(doubled, CircleScenarioBatch)
    np.testing.assert_array_equal(doubled.speed_limit, np.array([2.0, 4.0]))


def test_swept_circle_barrier_detects_collision_between_safe_controller_ticks() -> None:
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.0, 0.0]]]),
        obstacle_radii=jnp.array([[0.4]]),
        obstacle_mask=jnp.array([[True]]),
        arena_lower=jnp.array([[-3.0, -3.0]]),
        arena_upper=jnp.array([[3.0, 3.0]]),
        speed_limit=jnp.array([3.0]),
    )
    states = jnp.array([[[[-1.0, 0.0, 2.0, 0.0], [1.0, 0.0, 2.0, 0.0]]]])
    actions = jnp.zeros((1, 1, 1, 2))

    sampled = trajectory_constraints(states, scenarios, safety_margin=0.0)
    swept = swept_trajectory_constraints(
        states, actions, scenarios, safety_margin=0.0, dt=1.0, action_scale=3.0
    )

    assert np.all(np.asarray(sampled[..., 0]) > 0)
    assert swept.shape == (1, 1, 1, 6)
    assert float(swept[0, 0, 0, 0]) < 0


def test_swept_arena_barrier_checks_quadratic_vertex_inside_interval() -> None:
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.zeros((1, 0, 1)),
        obstacle_radii=jnp.zeros((1, 0)),
        obstacle_mask=jnp.zeros((1, 0), dtype=bool),
        arena_lower=jnp.array([[0.0]]),
        arena_upper=jnp.array([[1.0]]),
        speed_limit=jnp.array([3.0]),
    )
    # p(t) = 0.1 - t + t^2: both endpoints are 0.1, but p(0.5) = -0.15.
    states = jnp.array([[[[0.1, -1.0], [0.1, 1.0]]]])
    actions = jnp.array([[[[2.0]]]])

    swept = swept_trajectory_constraints(
        states, actions, scenarios, safety_margin=0.0, dt=1.0, action_scale=2.0
    )

    assert float(swept[0, 0, 0, 0]) == pytest.approx(-0.15, abs=2e-7)


def test_dimensionless_node_and_swept_barriers_are_invariant_to_length_unit_scaling() -> None:
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.3, -0.2]]]),
        obstacle_radii=jnp.array([[0.4]]),
        obstacle_mask=jnp.array([[True]]),
        arena_lower=jnp.array([[-2.0, -3.0]]),
        arena_upper=jnp.array([[4.0, 5.0]]),
        speed_limit=jnp.array([2.5]),
    )
    states = jnp.array([[[[-1.0, 0.5, 0.4, -0.2], [-0.55, 0.25, 0.5, -0.3]]]])
    actions = jnp.array([[[[0.1, -0.1]]]])
    margin = 0.08
    dt = 1.0
    scale = 100.0
    scaled_scenarios = scenarios.replace(
        obstacle_centers=scale * scenarios.obstacle_centers,
        obstacle_radii=scale * scenarios.obstacle_radii,
        arena_lower=scale * scenarios.arena_lower,
        arena_upper=scale * scenarios.arena_upper,
        speed_limit=scale * scenarios.speed_limit,
    )

    node = trajectory_constraints(states, scenarios, margin)
    node_scaled = trajectory_constraints(scale * states, scaled_scenarios, scale * margin)
    swept = swept_trajectory_constraints(states, actions, scenarios, margin, dt, 2.0)
    swept_scaled = swept_trajectory_constraints(
        scale * states, scale * actions, scaled_scenarios, scale * margin, dt, scale * 2.0
    )

    np.testing.assert_allclose(node_scaled, node, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(swept_scaled, swept, rtol=2e-6, atol=2e-6)


def test_masked_nan_obstacle_padding_cannot_poison_values_or_state_gradients() -> None:
    states = jnp.array([[[[-1.0, 0.2, 0.3, -0.1], [-0.7, 0.1, 0.3, -0.1]]]])
    actions = jnp.zeros((1, 1, 1, 2))
    common = {
        "arena_lower": jnp.array([[-3.0, -3.0]]),
        "arena_upper": jnp.array([[3.0, 3.0]]),
        "speed_limit": jnp.array([2.0]),
    }
    padded = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.5, 0.0], [jnp.nan, jnp.nan]]]),
        obstacle_radii=jnp.array([[0.25, jnp.nan]]),
        obstacle_mask=jnp.array([[True, False]]),
        **common,
    )
    compact = CircleScenarioBatch(
        obstacle_centers=padded.obstacle_centers[:, :1],
        obstacle_radii=padded.obstacle_radii[:, :1],
        obstacle_mask=jnp.array([[True]]),
        **common,
    )

    def margin(candidate_states: jax.Array, scenario: CircleScenarioBatch) -> jax.Array:
        constraints = swept_trajectory_constraints(
            candidate_states, actions, scenario, 0.05, dt=1.0, action_scale=2.0
        )
        return training_policy_margins(constraints, beta=10.0)[0, 0]

    padded_value, padded_gradient = jax.value_and_grad(margin)(states, padded)
    compact_value, compact_gradient = jax.value_and_grad(margin)(states, compact)

    assert np.all(np.isfinite(np.asarray(padded_gradient)))
    np.testing.assert_allclose(padded_value, compact_value, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(padded_gradient, compact_gradient, rtol=1e-6, atol=1e-6)


def test_exact_zero_subgradients_are_finite_at_tiny_scale_on_nodes_and_sweeps() -> None:
    radius = 1e-6
    scenario = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.0, 0.0]]]),
        obstacle_radii=jnp.array([[radius]]),
        obstacle_mask=jnp.array([[True]]),
        arena_lower=jnp.array([[-5 * radius, -5 * radius]]),
        arena_upper=jnp.array([[5 * radius, 5 * radius]]),
        speed_limit=jnp.array([radius]),
    )
    states = jnp.zeros((1, 1, 2, 4))
    actions = jnp.zeros((1, 1, 1, 2))

    node_gradient = jax.grad(
        lambda candidate: jnp.sum(trajectory_constraints(candidate, scenario, 0.0))
    )(states)
    action_gradient = jax.grad(
        lambda candidate: jnp.sum(
            swept_trajectory_constraints(
                states, candidate, scenario, 0.0, dt=0.1, action_scale=radius
            )
        )
    )(actions)

    assert np.all(np.isfinite(np.asarray(node_gradient)))
    assert np.all(np.isfinite(np.asarray(action_gradient)))


def test_scenario_validation_accepts_masked_nan_padding_and_rejects_invalid_real_data() -> None:
    valid = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.0, 0.0], [jnp.nan, jnp.nan]]]),
        obstacle_radii=jnp.array([[0.4, jnp.nan]]),
        obstacle_mask=jnp.array([[True, False]]),
        arena_lower=jnp.array([[-2.0, -2.0]]),
        arena_upper=jnp.array([[2.0, 2.0]]),
        speed_limit=jnp.array([2.0]),
    )
    valid.validate()

    invalid_batches = (
        valid.replace(obstacle_centers=valid.obstacle_centers.at[0, 0, 0].set(jnp.nan)),
        valid.replace(obstacle_radii=valid.obstacle_radii.at[0, 0].set(-0.1)),
        valid.replace(obstacle_mask=jnp.array([[1, 0]])),
        valid.replace(arena_upper=jnp.array([[-2.0, 2.0]])),
        valid.replace(speed_limit=jnp.array([jnp.inf])),
        valid.replace(obstacle_radii=jnp.ones((1, 3))),
    )
    for invalid in invalid_batches:
        with pytest.raises(ValueError):
            invalid.validate()


@pytest.mark.parametrize(
    "replacement",
    [
        {"obstacle_centers": jnp.array([[[jnp.inf, 0.0], [jnp.nan, jnp.nan]]])},
        {"obstacle_centers": jnp.array([[[-jnp.inf, 0.0], [jnp.nan, jnp.nan]]])},
        {"obstacle_radii": jnp.array([[jnp.inf, jnp.nan]])},
        {"arena_lower": jnp.array([[-jnp.inf, -2.0]])},
        {"arena_upper": jnp.array([[jnp.inf, 2.0]])},
        {"speed_limit": jnp.array([jnp.inf])},
    ],
)
def test_device_constraints_fail_closed_for_infinite_real_scenario_data(
    replacement: dict[str, jax.Array],
) -> None:
    scenario = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.0, 0.0], [jnp.nan, jnp.nan]]]),
        obstacle_radii=jnp.array([[0.4, jnp.nan]]),
        obstacle_mask=jnp.array([[True, False]]),
        arena_lower=jnp.array([[-2.0, -2.0]]),
        arena_upper=jnp.array([[2.0, 2.0]]),
        speed_limit=jnp.array([2.0]),
    ).replace(**replacement)
    states = jnp.zeros((1, 1, 2, 4))
    actions = jnp.zeros((1, 1, 1, 2))

    node_values = jax.jit(lambda x, batch: trajectory_constraints(x, batch, 0.0))(states, scenario)
    swept_values = jax.jit(
        lambda x, u, batch: swept_trajectory_constraints(x, u, batch, 0.0, 0.1, 2.0)
    )(states, actions, scenario)

    assert np.isnan(np.asarray(node_values)).any()
    assert np.isnan(np.asarray(swept_values)).any()


@pytest.mark.parametrize("beta", [0.0, -1.0])
def test_conservative_softmin_rejects_nonpositive_temperature(beta: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        conservative_softmin(jnp.ones(3), beta=beta)
