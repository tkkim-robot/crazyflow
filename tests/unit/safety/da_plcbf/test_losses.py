from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.config import LibraryLossConfig, RolloutConfig
from crazyflow.safety.da_plcbf.descriptors import trajectory_descriptors
from crazyflow.safety.da_plcbf.losses import LibraryLossMetrics, library_loss
from crazyflow.safety.da_plcbf.rollouts import rollout_structured_library
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.values import hard_policy_margins, swept_trajectory_constraints


def _loss_inputs(
    *, dtype: jnp.dtype = jnp.float32, horizon: int = 5
) -> tuple[
    jax.Array,
    jax.Array,
    CircleScenarioBatch,
    jax.Array,
    jax.Array,
    jax.Array,
    RolloutConfig,
    LibraryLossConfig,
]:
    desired_velocities = jnp.array([[1.0, 0.2], [-0.4, 0.8], [0.2, -0.9]], dtype=dtype)
    initial_states = jnp.array([[-1.5, 0.0, 0.2, 0.0], [1.2, -0.4, -0.1, 0.2]], dtype=dtype)
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array(
            [[[0.1, 0.3], [100.0, 100.0]], [[-0.2, 0.7], [50.0, -50.0]]], dtype=dtype
        ),
        obstacle_radii=jnp.array([[0.35, 20.0], [0.4, 10.0]], dtype=dtype),
        obstacle_mask=jnp.array([[True, False], [True, False]]),
        arena_lower=jnp.full((2, 2), -4.0, dtype=dtype),
        arena_upper=jnp.full((2, 2), 4.0, dtype=dtype),
        speed_limit=jnp.array([3.0, 2.5], dtype=dtype),
    )
    target_codes = jnp.array(
        [
            [0.6, 0.1, 0.5, 0.0, 0.8, 0.1],
            [-0.2, 0.5, -0.1, 0.6, -0.3, 0.7],
            [0.1, -0.6, 0.2, -0.5, 0.2, -0.8],
        ],
        dtype=dtype,
    )
    active_desired_velocities = desired_velocities + jnp.array(
        [[0.05, -0.02], [-0.03, 0.04], [0.02, 0.01]], dtype=dtype
    )
    descriptor_scales = jnp.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)
    rollout_config = RolloutConfig(
        dt=0.08,
        horizon=horizon,
        policy_gain=1.3,
        action_limit=2.0,
        safety_margin=0.05,
        softmin_beta=8.0,
    )
    loss_config = LibraryLossConfig(covariance_regularizer=0.05)
    return (
        desired_velocities,
        initial_states,
        scenarios,
        target_codes,
        active_desired_velocities,
        descriptor_scales,
        rollout_config,
        loss_config,
    )


def _evaluate_loss(
    desired_velocities: jax.Array,
    initial_states: jax.Array,
    scenarios: CircleScenarioBatch,
    target_codes: jax.Array,
    active_desired_velocities: jax.Array,
    descriptor_scales: jax.Array,
    rollout_config: RolloutConfig,
    loss_config: LibraryLossConfig,
) -> tuple[jax.Array, LibraryLossMetrics]:
    return library_loss(
        desired_velocities,
        initial_states,
        scenarios,
        target_codes,
        active_desired_velocities,
        descriptor_scales,
        rollout_config,
        loss_config,
    )


def test_library_loss_and_every_reported_metric_are_finite_and_consistent() -> None:
    inputs = _loss_inputs()

    total, metrics = _evaluate_loss(*inputs)

    metric_values = np.asarray(jax.tree_util.tree_leaves(metrics), dtype=float)
    assert np.isfinite(np.asarray(total))
    assert np.all(np.isfinite(metric_values))
    np.testing.assert_allclose(total, metrics.total, rtol=0.0, atol=0.0)
    rollout_config = inputs[-2]
    loss_config = inputs[-1]
    reconstructed = (
        loss_config.coverage_weight * metrics.coverage
        + loss_config.redundancy_weight * metrics.redundancy
        + loss_config.diversity_weight * metrics.diversity
        + loss_config.code_weight * metrics.code
        + loss_config.action_weight * metrics.action
        + loss_config.action_rate_weight * metrics.action_rate
        + loss_config.terminal_weight * metrics.terminal
        + loss_config.trust_weight * metrics.trust
    )
    np.testing.assert_allclose(total, reconstructed, rtol=2e-7, atol=2e-7)
    velocity_scale = rollout_config.action_limit / rollout_config.policy_gain
    np.testing.assert_allclose(
        metrics.trust,
        jnp.mean(((inputs[0] - inputs[4]) / velocity_scale) ** 2),
        rtol=1e-7,
        atol=1e-7,
    )
    assert metrics.coverage >= 0
    assert metrics.code >= 0
    assert metrics.action >= 0
    assert metrics.action_rate >= 0
    assert metrics.terminal >= 0
    assert metrics.trust >= 0
    assert 0 <= metrics.hard_safe_fraction <= 1
    assert 0 <= metrics.smooth_safe_count <= inputs[0].shape[0]
    assert rollout_config.horizon == 5


def test_library_loss_normalizes_physical_targets_and_is_si_scale_invariant() -> None:
    inputs = _loss_inputs()
    original_total, original_metrics = _evaluate_loss(*inputs)
    rollouts = rollout_structured_library(inputs[0], inputs[1], inputs[-2], smooth_actions=True)
    normalized_descriptors = trajectory_descriptors(rollouts.states) / inputs[5]
    expected_code = jnp.mean((normalized_descriptors - inputs[3][:, None, :] / inputs[5]) ** 2)
    np.testing.assert_allclose(original_metrics.code, expected_code, rtol=1e-7, atol=1e-7)

    scale = 100.0
    desired, states, scenarios, targets, active, descriptor_scales, rollout, loss = inputs
    scaled_scenarios = scenarios.replace(
        obstacle_centers=scale * scenarios.obstacle_centers,
        obstacle_radii=scale * scenarios.obstacle_radii,
        arena_lower=scale * scenarios.arena_lower,
        arena_upper=scale * scenarios.arena_upper,
        speed_limit=scale * scenarios.speed_limit,
    )
    scaled_rollout = replace(
        rollout,
        action_limit=scale * rollout.action_limit,
        safety_margin=scale * rollout.safety_margin,
    )
    scaled_total, scaled_metrics = _evaluate_loss(
        scale * desired,
        scale * states,
        scaled_scenarios,
        scale * targets,
        scale * active,
        scale * descriptor_scales,
        scaled_rollout,
        loss,
    )

    np.testing.assert_allclose(scaled_total, original_total, rtol=4e-6, atol=4e-6)
    for scaled_value, original_value in zip(
        jax.tree.leaves(scaled_metrics), jax.tree.leaves(original_metrics), strict=True
    ):
        np.testing.assert_allclose(scaled_value, original_value, rtol=4e-6, atol=4e-6)


def test_hard_loss_diagnostics_are_exact_hard_rollout_reductions() -> None:
    inputs = _loss_inputs()
    desired_velocities, initial_states, scenarios = inputs[:3]
    rollout_config = inputs[-2]
    _, metrics = _evaluate_loss(*inputs)
    rollouts = rollout_structured_library(desired_velocities, initial_states, rollout_config)
    constraints = swept_trajectory_constraints(
        rollouts.states,
        rollouts.actions,
        scenarios,
        rollout_config.safety_margin,
        rollout_config.dt,
        rollout_config.action_limit,
    )
    margins = hard_policy_margins(constraints)
    best_by_scenario = jnp.max(margins, axis=0)

    np.testing.assert_allclose(metrics.hard_library_margin, jnp.min(best_by_scenario))
    np.testing.assert_allclose(
        metrics.hard_safe_fraction, jnp.mean(best_by_scenario >= 0), rtol=0.0, atol=0.0
    )

    colder_inputs = (*inputs[:-2], replace(rollout_config, softmin_beta=80.0), inputs[-1])
    colder_total, colder_metrics = _evaluate_loss(*colder_inputs)
    np.testing.assert_allclose(
        colder_metrics.hard_library_margin, metrics.hard_library_margin, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        colder_metrics.hard_safe_fraction, metrics.hard_safe_fraction, rtol=0.0, atol=0.0
    )
    assert np.isfinite(np.asarray(colder_total))


def test_library_loss_autodiff_matches_central_difference_for_policy_parameters() -> None:
    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        inputs = _loss_inputs(dtype=jnp.float64)
        desired_velocities = inputs[0]

        def objective(candidate: jax.Array) -> jax.Array:
            return _evaluate_loss(candidate, *inputs[1:])[0]

        autodiff = np.asarray(jax.grad(objective)(desired_velocities))
        base = np.asarray(desired_velocities)
        central = np.empty_like(base)
        step = 1e-5
        for index in np.ndindex(base.shape):
            positive = base.copy()
            negative = base.copy()
            positive[index] += step
            negative[index] -= step
            central[index] = (
                float(objective(jnp.asarray(positive))) - float(objective(jnp.asarray(negative)))
            ) / (2 * step)

        assert np.all(np.isfinite(autodiff))
        np.testing.assert_allclose(autodiff, central, rtol=3e-5, atol=3e-6)
    finally:
        jax.config.update("jax_enable_x64", previous_x64)


def test_library_loss_autodiff_matches_state_direction_central_difference() -> None:
    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        inputs = _loss_inputs(dtype=jnp.float64)
        initial_states = inputs[1]
        direction = jnp.array([[0.3, -0.2, 0.1, 0.4], [-0.25, 0.15, -0.35, 0.2]], dtype=jnp.float64)
        direction /= jnp.linalg.norm(direction)

        def objective(candidate: jax.Array) -> jax.Array:
            return _evaluate_loss(inputs[0], candidate, *inputs[2:])[0]

        autodiff_direction = jnp.vdot(jax.grad(objective)(initial_states), direction)
        step = 1e-5
        central_direction = (
            objective(initial_states + step * direction)
            - objective(initial_states - step * direction)
        ) / (2 * step)

        assert np.isfinite(np.asarray(autodiff_direction))
        np.testing.assert_allclose(autodiff_direction, central_direction, rtol=3e-5, atol=3e-6)
    finally:
        jax.config.update("jax_enable_x64", previous_x64)


def test_library_loss_is_finite_at_the_minimum_valid_rollout_horizon() -> None:
    inputs = _loss_inputs(horizon=1)

    total, metrics = _evaluate_loss(*inputs)

    assert np.isfinite(np.asarray(total))
    assert np.all(np.isfinite(np.asarray(jax.tree_util.tree_leaves(metrics), dtype=float)))


def test_hard_loss_diagnostic_detects_horizon_one_mid_interval_collision() -> None:
    rollout_config = RolloutConfig(
        dt=1.0, horizon=1, policy_gain=1.0, action_limit=3.0, safety_margin=0.0, softmin_beta=10.0
    )
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.0, 0.0]]]),
        obstacle_radii=jnp.array([[0.4]]),
        obstacle_mask=jnp.array([[True]]),
        arena_lower=jnp.array([[-3.0, -3.0]]),
        arena_upper=jnp.array([[3.0, 3.0]]),
        speed_limit=jnp.array([3.0]),
    )
    params = jnp.array([[2.0, 0.0]])

    _, metrics = library_loss(
        params,
        jnp.array([[-1.0, 0.0, 2.0, 0.0]]),
        scenarios,
        jnp.zeros((1, 6)),
        params,
        jnp.ones(6),
        rollout_config,
        LibraryLossConfig(covariance_regularizer=0.05),
    )

    assert metrics.hard_library_margin < 0
    assert metrics.hard_safe_fraction == 0


def test_best_policy_coverage_is_invariant_to_duplicating_the_entire_library() -> None:
    inputs = _loss_inputs()
    _, original = _evaluate_loss(*inputs)
    repeated_inputs = (
        jnp.repeat(inputs[0], 2, axis=0),
        inputs[1],
        inputs[2],
        jnp.repeat(inputs[3], 2, axis=0),
        jnp.repeat(inputs[4], 2, axis=0),
        inputs[5],
        inputs[6],
        inputs[7],
    )

    _, repeated = _evaluate_loss(*repeated_inputs)

    np.testing.assert_allclose(repeated.coverage, original.coverage, rtol=1e-7, atol=1e-7)


def test_complete_dimensionless_loss_is_invariant_to_metres_to_centimetres_scaling() -> None:
    inputs = _loss_inputs()
    scale = 100.0
    scenario = inputs[2]
    scaled_scenario = scenario.replace(
        obstacle_centers=scale * scenario.obstacle_centers,
        obstacle_radii=scale * scenario.obstacle_radii,
        arena_lower=scale * scenario.arena_lower,
        arena_upper=scale * scenario.arena_upper,
        speed_limit=scale * scenario.speed_limit,
    )
    scaled_rollout = replace(
        inputs[6],
        action_limit=scale * inputs[6].action_limit,
        safety_margin=scale * inputs[6].safety_margin,
    )
    scaled_inputs = (
        scale * inputs[0],
        scale * inputs[1],
        scaled_scenario,
        scale * inputs[3],
        scale * inputs[4],
        scale * inputs[5],
        scaled_rollout,
        inputs[7],
    )

    original_total, original_metrics = _evaluate_loss(*inputs)
    scaled_total, scaled_metrics = _evaluate_loss(*scaled_inputs)

    np.testing.assert_allclose(scaled_total, original_total, rtol=3e-6, atol=3e-6)
    for scaled_value, original_value in zip(
        jax.tree.leaves(scaled_metrics), jax.tree.leaves(original_metrics), strict=True
    ):
        np.testing.assert_allclose(scaled_value, original_value, rtol=3e-6, atol=3e-6)
