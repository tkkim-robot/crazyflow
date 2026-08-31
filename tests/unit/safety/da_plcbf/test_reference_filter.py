import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.config import RolloutConfig
from crazyflow.safety.da_plcbf.policies import structured_velocity_policy
from crazyflow.safety.da_plcbf.qp import project_box_halfspace
from crazyflow.safety.da_plcbf.reference_filter import (
    box_halfspace_fraction_2d,
    reference_plcbf_filter,
)
from crazyflow.safety.da_plcbf.rollouts import rollout_structured_library
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.values import hard_policy_margins, swept_trajectory_constraints


@pytest.mark.parametrize(
    ("normal", "minimum", "lower", "upper", "expected"),
    [
        ([0.0, 0.0], -0.1, [-2.0, -3.0], [2.0, 1.0], 1.0),
        ([0.0, 0.0], 0.1, [-2.0, -3.0], [2.0, 1.0], 0.0),
        ([2.0, 0.0], -5.0, [-2.0, -3.0], [2.0, 1.0], 1.0),
        ([2.0, 0.0], 5.0, [-2.0, -3.0], [2.0, 1.0], 0.0),
        ([2.0, 0.0], 0.0, [-2.0, -3.0], [2.0, 1.0], 0.5),
        ([1.0, 1.0], 0.0, [0.0, 0.0], [1.0, 1.0], 1.0),
        ([1.0, 1.0], 2.0, [0.0, 0.0], [1.0, 1.0], 0.0),
        ([1.0, 1.0], 1.0, [0.0, 0.0], [1.0, 1.0], 0.5),
        ([1.0, 1.0], 0.5, [0.0, 0.0], [1.0, 1.0], 0.875),
        ([1.0, 1.0], 1.5, [0.0, 0.0], [1.0, 1.0], 0.125),
        ([-1.0, 1.0], 3.0, [-1.0, 2.0], [3.0, 6.0], 0.5),
    ],
)
def test_box_halfspace_fraction_has_exact_constant_one_and_two_dimensional_cases(
    normal: list[float], minimum: float, lower: list[float], upper: list[float], expected: float
) -> None:
    arguments = (jnp.array(normal), jnp.array(minimum), jnp.array(lower), jnp.array(upper))

    eager = box_halfspace_fraction_2d(*arguments)
    compiled = jax.jit(box_halfspace_fraction_2d)(*arguments)

    np.testing.assert_allclose(eager, expected, rtol=0.0, atol=2e-7)
    np.testing.assert_allclose(compiled, expected, rtol=0.0, atol=2e-7)


@pytest.mark.parametrize("seed", range(10))
def test_box_halfspace_fraction_matches_dense_deterministic_grid(seed: int) -> None:
    generator = np.random.default_rng(202_608_30 + seed)
    lower = generator.uniform(-3.0, 0.0, size=2)
    upper = lower + generator.uniform(0.25, 4.0, size=2)
    normal = generator.normal(size=2)
    support_lower = float(np.sum(np.minimum(normal * lower, normal * upper)))
    support_upper = float(np.sum(np.maximum(normal * lower, normal * upper)))
    support_width = support_upper - support_lower
    minimum = generator.uniform(
        support_lower - 0.15 * support_width, support_upper + 0.15 * support_width
    )
    grid_size = 801
    first = lower[0] + (np.arange(grid_size) + 0.5) * (upper[0] - lower[0]) / grid_size
    second = lower[1] + (np.arange(grid_size) + 0.5) * (upper[1] - lower[1]) / grid_size
    grid_fraction = np.mean(normal[0] * first[:, None] + normal[1] * second[None, :] >= minimum)

    exact = box_halfspace_fraction_2d(
        jnp.asarray(normal), jnp.asarray(minimum), jnp.asarray(lower), jnp.asarray(upper)
    )

    np.testing.assert_allclose(exact, grid_fraction, rtol=0.0, atol=3e-3)


def _filter_problem() -> tuple[
    jax.Array, jax.Array, CircleScenarioBatch, jax.Array, jax.Array, jax.Array, RolloutConfig
]:
    desired_velocities = jnp.array([[1.5, 0.0], [-1.5, 0.0], [0.0, 1.5], [0.0, -1.5], [0.5, 0.5]])
    state = jnp.array([-2.0, 0.2, -1.0, 0.1])
    scenario = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.0, 0.0]]]),
        obstacle_radii=jnp.array([[0.6]]),
        obstacle_mask=jnp.array([[True]]),
        arena_lower=jnp.array([[-3.0, -3.0]]),
        arena_upper=jnp.array([[3.0, 3.0]]),
        speed_limit=jnp.array([2.0]),
    )
    nominal_action = jnp.array([-2.0, -2.0])
    action_lower = jnp.array([-2.0, -2.0])
    action_upper = jnp.array([2.0, 2.0])
    config = RolloutConfig(
        dt=0.1, horizon=5, policy_gain=1.5, action_limit=2.0, safety_margin=0.05, softmin_beta=10.0
    )
    return (desired_velocities, state, scenario, nominal_action, action_lower, action_upper, config)


def _selected_policy_value(
    state: jax.Array,
    desired_velocity: jax.Array,
    scenario: CircleScenarioBatch,
    config: RolloutConfig,
) -> jax.Array:
    rollout = rollout_structured_library(
        desired_velocity[None, :], state[None, :], config, smooth_actions=True
    )
    constraints = swept_trajectory_constraints(
        rollout.states,
        rollout.actions,
        scenario,
        config.safety_margin,
        config.dt,
        config.action_limit,
    )
    return hard_policy_margins(constraints)[0, 0]


def test_certified_filter_selects_largest_admissible_fraction_and_enforces_qp() -> None:
    (desired_velocities, state, scenario, nominal_action, action_lower, action_upper, config) = (
        _filter_problem()
    )
    alpha = 4.0

    result = reference_plcbf_filter(
        desired_velocities,
        state,
        scenario,
        nominal_action,
        action_lower,
        action_upper,
        config,
        alpha=alpha,
    )

    eligible = (
        np.isfinite(np.asarray(result.cbf_values))
        & np.isfinite(np.asarray(result.hard_values))
        & (np.asarray(result.cbf_values) > 0)
        & (np.asarray(result.hard_values) >= 0)
        & (np.asarray(result.admissible_fractions) > 0)
    )
    expected_index = int(
        np.argmax(np.where(eligible, np.asarray(result.admissible_fractions), -np.inf))
    )
    assert int(result.selected_index) == expected_index
    assert np.count_nonzero(eligible) > 1
    assert bool(result.has_certificate)
    assert bool(result.qp_feasible)
    assert bool(result.qp_accepted)
    assert not bool(result.used_fallback)
    assert not bool(result.degraded)
    assert np.all(np.asarray(result.action) >= np.asarray(action_lower))
    assert np.all(np.asarray(result.action) <= np.asarray(action_upper))

    selected_target = desired_velocities[result.selected_index]
    selected_value, gradient = jax.value_and_grad(_selected_policy_value)(
        state, selected_target, scenario, config
    )
    velocity = state[2:]
    normal = gradient[2:]
    minimum = -jnp.dot(gradient[:2], velocity) - alpha * selected_value
    independent_qp = project_box_halfspace(
        nominal_action,
        jnp.ones_like(nominal_action),
        action_lower,
        action_upper,
        normal,
        minimum,
        tolerance=1e-6,
    )
    direct_residual = jnp.dot(normal, result.action) - minimum

    np.testing.assert_allclose(result.action, independent_qp.action, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.constraint_residual, direct_residual, rtol=0.0, atol=1e-6)
    assert result.constraint_residual >= -1e-6
    assert result.postcheck_interval_margin >= -1e-6
    assert result.postcheck_discrete_residual >= -1e-6
    assert result.applied_discrete_residual >= -1e-6


def test_reference_filter_is_jittable_and_reproduces_eager_result() -> None:
    arguments = _filter_problem()
    config = arguments[-1]
    dynamic_arguments = arguments[:-1]
    eager = reference_plcbf_filter(*dynamic_arguments, config)
    compiled = jax.jit(
        lambda policies, state, scenario, nominal, lower, upper: reference_plcbf_filter(
            policies, state, scenario, nominal, lower, upper, config
        )
    )(*dynamic_arguments)

    eager_leaves, eager_structure = jax.tree_util.tree_flatten(eager)
    compiled_leaves, compiled_structure = jax.tree_util.tree_flatten(compiled)
    assert eager_structure == compiled_structure
    for eager_leaf, compiled_leaf in zip(eager_leaves, compiled_leaves, strict=True):
        np.testing.assert_allclose(eager_leaf, compiled_leaf, rtol=2e-6, atol=2e-6)


def test_no_certificate_is_explicitly_degraded_and_applies_best_effort_policy() -> None:
    (desired_velocities, _state, scenario, nominal_action, action_lower, action_upper, config) = (
        _filter_problem()
    )
    unsafe_state = jnp.array([0.0, 0.0, 0.0, 0.0])

    result = reference_plcbf_filter(
        desired_velocities,
        unsafe_state,
        scenario,
        nominal_action,
        action_lower,
        action_upper,
        config,
    )
    best_effort_index = int(np.argmax(np.asarray(result.hard_values)))
    expected_action = structured_velocity_policy(
        unsafe_state,
        desired_velocities[best_effort_index],
        config.policy_gain,
        config.action_limit,
        smooth=True,
    )

    assert np.all(np.asarray(result.hard_values) < 0)
    assert not bool(result.has_certificate)
    assert not bool(result.qp_feasible)
    assert bool(result.degraded)
    assert int(result.selected_index) == best_effort_index
    np.testing.assert_allclose(result.action, expected_action, rtol=0.0, atol=0.0)


def test_filter_rejects_policy_that_crosses_obstacle_between_safe_endpoints() -> None:
    desired_velocities = jnp.array([[2.0, 0.0]])
    state = jnp.array([-1.0, 0.0, 2.0, 0.0])
    scenario = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.0, 0.0]]]),
        obstacle_radii=jnp.array([[0.4]]),
        obstacle_mask=jnp.array([[True]]),
        arena_lower=jnp.array([[-3.0, -3.0]]),
        arena_upper=jnp.array([[3.0, 3.0]]),
        speed_limit=jnp.array([3.0]),
    )
    config = RolloutConfig(
        dt=1.0, horizon=1, policy_gain=1.0, action_limit=3.0, safety_margin=0.0, softmin_beta=10.0
    )

    result = reference_plcbf_filter(
        desired_velocities,
        state,
        scenario,
        nominal_action=jnp.zeros(2),
        action_lower=-3.0 * jnp.ones(2),
        action_upper=3.0 * jnp.ones(2),
        config=config,
    )

    assert float(result.hard_values[0]) < 0
    assert not bool(result.has_certificate)
    assert bool(result.degraded)


def test_qp_action_is_rejected_when_tick_start_cbf_passes_but_held_action_collides() -> None:
    desired_velocities = jnp.array([[0.0, 0.0], [-2.0, 0.0], [2.0, 0.0], [0.0, -2.0], [0.0, 2.0]])
    state = jnp.array([-1.802811, -0.32184377, 3.0504575, 0.08565205])
    scenario = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[-0.01115045, 0.47646232]]]),
        obstacle_radii=jnp.array([[0.665414713]]),
        obstacle_mask=jnp.array([[True]]),
        arena_lower=jnp.array([[-5.0, -5.0]]),
        arena_upper=jnp.array([[5.0, 5.0]]),
        speed_limit=jnp.array([10.0]),
    )
    nominal_action = jnp.array([-1.2456342, 3.959339])
    config = RolloutConfig(
        dt=0.5, horizon=4, policy_gain=2.0, action_limit=4.0, safety_margin=0.0, softmin_beta=10.0
    )

    result = reference_plcbf_filter(
        desired_velocities,
        state,
        scenario,
        nominal_action,
        -4.0 * jnp.ones(2),
        4.0 * jnp.ones(2),
        config,
    )

    assert bool(result.has_certificate)
    assert bool(result.qp_feasible)
    assert result.postcheck_interval_margin < 0
    assert not bool(result.qp_accepted)
    assert bool(result.used_fallback)
    assert result.applied_interval_margin >= 0
    assert not bool(result.degraded)
    assert not np.allclose(result.action, nominal_action)


def test_fallback_with_safe_held_interval_but_lost_next_certificate_is_degraded() -> None:
    desired_velocities = jnp.array([[1.2, 0.0]])
    state = jnp.array([-1.5, 0.0, 1.2, 0.0])
    scenario = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.0, 0.0]]]),
        obstacle_radii=jnp.array([[0.4]]),
        obstacle_mask=jnp.array([[True]]),
        arena_lower=jnp.array([[-5.0, -5.0]]),
        arena_upper=jnp.array([[5.0, 5.0]]),
        speed_limit=jnp.array([10.0]),
    )
    config = RolloutConfig(
        dt=0.5, horizon=1, policy_gain=2.0, action_limit=4.0, safety_margin=0.0, softmin_beta=10.0
    )

    result = reference_plcbf_filter(
        desired_velocities,
        state,
        scenario,
        nominal_action=jnp.array([4.0, 0.0]),
        action_lower=-4.0 * jnp.ones(2),
        action_upper=4.0 * jnp.ones(2),
        config=config,
    )

    assert bool(result.has_certificate)
    assert not bool(result.qp_accepted)
    assert result.applied_interval_margin > 0
    assert result.fallback_discrete_residual < 0
    assert result.applied_discrete_residual < 0
    assert bool(result.degraded)


def test_certificate_rollout_uses_actual_runtime_action_box_not_configured_training_limit() -> None:
    desired_velocities = jnp.array([[-2.0, 0.0]])
    state = jnp.array([-2.0, 0.0, 1.0, 0.0])
    scenario = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.0, 0.0]]]),
        obstacle_radii=jnp.array([[0.4]]),
        obstacle_mask=jnp.array([[True]]),
        arena_lower=jnp.array([[-5.0, -5.0]]),
        arena_upper=jnp.array([[5.0, 5.0]]),
        speed_limit=jnp.array([10.0]),
    )
    config = RolloutConfig(
        dt=0.1, horizon=20, policy_gain=2.0, action_limit=4.0, safety_margin=0.0, softmin_beta=10.0
    )

    result = reference_plcbf_filter(
        desired_velocities,
        state,
        scenario,
        nominal_action=jnp.zeros(2),
        action_lower=jnp.zeros(2),
        action_upper=jnp.zeros(2),
        config=config,
    )

    assert result.hard_values[0] < 0
    assert not bool(result.has_certificate)
    assert not bool(result.qp_accepted)
    assert bool(result.degraded)


@pytest.mark.parametrize("minimum", [1e4, 1e8, 1e10])
def test_fraction_returns_zero_far_above_support_without_cancellation(minimum: float) -> None:
    fraction = box_halfspace_fraction_2d(
        jnp.array([1.0, 1.0]), jnp.array(minimum), jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])
    )

    assert float(fraction) == 0.0


@pytest.mark.parametrize("scale", [1e-20, 1e-7, 1.0, 1e7, 1e20])
def test_fraction_is_invariant_to_positive_halfspace_rescaling(scale: float) -> None:
    fraction = box_halfspace_fraction_2d(
        scale * jnp.array([1.0, 1.0]), jnp.asarray(0.5 * scale), jnp.zeros(2), jnp.ones(2)
    )

    np.testing.assert_allclose(fraction, 0.875, rtol=0.0, atol=2e-6)


def test_fraction_handles_below_support_degenerate_boxes_and_invalid_numbers() -> None:
    below = box_halfspace_fraction_2d(
        jnp.array([1.0, -2.0]), jnp.array(-100.0), jnp.array([-1.0, 0.5]), jnp.array([2.0, 0.5])
    )
    point_true = box_halfspace_fraction_2d(
        jnp.array([1.0, 1.0]), jnp.array(3.0), jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0])
    )
    point_false = box_halfspace_fraction_2d(
        jnp.array([1.0, 1.0]), jnp.array(3.1), jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0])
    )
    invalid = box_halfspace_fraction_2d(
        jnp.array([jnp.nan, 1.0]), jnp.array(0.0), -jnp.ones(2), jnp.ones(2)
    )

    assert float(below) == 1.0
    assert float(point_true) == 1.0
    assert float(point_false) == 0.0
    assert np.isnan(float(invalid))


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("alpha", float("nan")),
        ("alpha", float("inf")),
        ("qp_tolerance", float("nan")),
        ("qp_tolerance", float("inf")),
        ("qp_tolerance", -1.0),
    ],
)
def test_filter_rejects_nonfinite_or_negative_numeric_tolerances(
    keyword: str, value: float
) -> None:
    arguments = _filter_problem()

    with pytest.raises(ValueError):
        reference_plcbf_filter(*arguments, **{keyword: value})


def test_box_halfspace_fraction_rejects_nonplanar_shapes() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        box_halfspace_fraction_2d(jnp.ones(3), jnp.array(0.0), jnp.zeros(3), jnp.ones(3))
