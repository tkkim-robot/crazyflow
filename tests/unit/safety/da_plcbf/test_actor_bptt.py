from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.actor import (
    SharedActorConfig,
    SharedActorSpec,
    initialize_shared_actor,
)
from crazyflow.safety.da_plcbf.actor_bptt import build_shared_actor_bptt_functions
from crazyflow.safety.da_plcbf.actor_losses import shared_actor_library_loss
from crazyflow.safety.da_plcbf.actor_rollouts import rollout_shared_actor_library
from crazyflow.safety.da_plcbf.config import LibraryLossConfig, RolloutConfig
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.values import hard_policy_margins, swept_trajectory_constraints


def _training_problem() -> tuple[object, ...]:
    spec = SharedActorSpec(
        base_codes=jnp.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.1, 0.5], [-1.0, -0.1, -0.5]]
        ),
        base_desired_velocities=jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.1], [1.0, -0.1]]),
        base_durations=jnp.full((4,), 2.2),
        adaptive_mask=jnp.array([False, False, True, True]),
    )
    rollout_config = RolloutConfig(
        dt=0.1, horizon=20, policy_gain=2.0, action_limit=3.0, safety_margin=0.05, softmin_beta=20.0
    )
    actor_config = SharedActorConfig(
        hidden_width=16,
        residual_scale=0.5,
        min_duration=0.2,
        max_duration=2.5,
        duration_transition=0.1,
    )
    loss_config = LibraryLossConfig(
        target_margin=0.02,
        coverage_softplus_temperature=0.05,
        safe_count_temperature=0.05,
        covariance_regularizer=0.05,
        coverage_weight=1.0,
        redundancy_weight=0.05,
        diversity_weight=0.001,
        code_weight=0.0,
        action_weight=0.0,
        action_rate_weight=0.0,
        terminal_weight=0.0,
        trust_weight=0.001,
    )
    params = initialize_shared_actor(
        jax.random.key(0), spec, dimension=2, n_obstacles=1, config=actor_config
    )
    train_states = jnp.array(
        [[-1.5, 0.0, 1.0, 0.0], [-1.5, 0.15, 1.0, 0.0], [-1.5, -0.15, 1.0, 0.0]]
    )
    train_scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.0, 0.0]], [[0.0, 0.05]], [[0.0, -0.05]]]),
        obstacle_radii=jnp.full((3, 1), 0.4),
        obstacle_mask=jnp.ones((3, 1), dtype=bool),
        arena_lower=jnp.full((3, 2), -3.0),
        arena_upper=jnp.full((3, 2), 3.0),
        speed_limit=jnp.full((3,), 3.0),
    )
    return (spec, rollout_config, actor_config, loss_config, params, train_states, train_scenarios)


def _library_margin(
    params: object,
    spec: SharedActorSpec,
    states: jax.Array,
    scenarios: CircleScenarioBatch,
    rollout_config: RolloutConfig,
    actor_config: SharedActorConfig,
) -> tuple[jax.Array, object]:
    rollouts = rollout_shared_actor_library(
        params, spec, states, scenarios, rollout_config, actor_config
    )
    constraints = swept_trajectory_constraints(
        rollouts.states,
        rollouts.actions,
        scenarios,
        rollout_config.safety_margin,
        rollout_config.dt,
        rollout_config.action_limit,
    )
    return jnp.max(hard_policy_margins(constraints), axis=0), rollouts


def test_shared_actor_bptt_learns_future_safety_and_preserves_structural_core() -> None:
    (spec, rollout_config, actor_config, loss_config, params, train_states, train_scenarios) = (
        _training_problem()
    )
    held_out_states = jnp.array(
        [[-1.55, 0.08, 1.1, -0.05], [-1.45, -0.08, 0.9, 0.05], [-1.6, 0.0, 1.15, 0.0]]
    )
    held_out_scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.08, -0.03]], [[-0.08, 0.03]], [[0.12, 0.0]]]),
        obstacle_radii=jnp.array([[0.38], [0.42], [0.36]]),
        obstacle_mask=jnp.ones((3, 1), dtype=bool),
        arena_lower=jnp.full((3, 2), -3.0),
        arena_upper=jnp.full((3, 2), 3.0),
        speed_limit=jnp.full((3,), 3.0),
    )
    functions = build_shared_actor_bptt_functions(
        spec,
        rollout_config,
        actor_config,
        loss_config,
        learning_rate=0.02,
        max_gradient_norm=10.0,
        burst_steps=100,
    )
    targets = jnp.zeros((4, 6))
    descriptor_scales = jnp.full((6,), 3.0)
    train_before, rollout_before = _library_margin(
        params, spec, train_states, train_scenarios, rollout_config, actor_config
    )
    held_out_before, _ = _library_margin(
        params, spec, held_out_states, held_out_scenarios, rollout_config, actor_config
    )

    final_state, history = functions.burst(
        functions.initialize(params),
        train_states,
        train_scenarios,
        targets,
        params,
        descriptor_scales,
    )
    train_after, rollout_after = _library_margin(
        final_state.params, spec, train_states, train_scenarios, rollout_config, actor_config
    )
    held_out_after, _ = _library_margin(
        final_state.params, spec, held_out_states, held_out_scenarios, rollout_config, actor_config
    )

    assert np.all(np.asarray(history.update_accepted))
    assert np.max(np.asarray(train_before)) < 0
    assert np.max(np.asarray(held_out_before)) < 0
    assert np.all(np.asarray(train_after) > 0)
    assert np.all(np.asarray(held_out_after) > 0)
    np.testing.assert_array_equal(rollout_after.states[:2], rollout_before.states[:2])
    np.testing.assert_array_equal(rollout_after.actions[:2], rollout_before.actions[:2])
    assert np.linalg.norm(np.asarray(final_state.params.output_kernel)) > 0.1
    assert np.linalg.norm(np.asarray(final_state.params.velocity_offsets[2:])) > 0.2
    assert np.linalg.norm(np.asarray(final_state.params.code_offsets[2:])) > 0.05
    assert np.all(np.asarray(final_state.params.velocity_offsets[:2]) == 0)
    assert np.all(np.asarray(final_state.params.code_offsets[:2]) == 0)


def test_shared_actor_loss_is_dimensionless_under_coherent_si_scaling() -> None:
    (spec, rollout_config, actor_config, loss_config, params, states, scenarios) = (
        _training_problem()
    )
    displacement_targets = spec.base_desired_velocities * spec.base_durations[:, None]
    targets = jnp.concatenate(
        (
            displacement_targets,
            spec.base_desired_velocities,
            jnp.zeros_like(spec.base_desired_velocities),
        ),
        axis=-1,
    )
    descriptor_scales = jnp.full((6,), 3.0)
    original = shared_actor_library_loss(
        params,
        spec,
        states,
        scenarios,
        targets,
        params,
        descriptor_scales,
        rollout_config,
        actor_config,
        loss_config,
    )
    scale = 100.0
    scaled_spec = spec.replace(base_desired_velocities=scale * spec.base_desired_velocities)
    scaled_params = params.replace(velocity_offsets=scale * params.velocity_offsets)
    scaled_scenarios = scenarios.replace(
        obstacle_centers=scale * scenarios.obstacle_centers,
        obstacle_radii=scale * scenarios.obstacle_radii,
        arena_lower=scale * scenarios.arena_lower,
        arena_upper=scale * scenarios.arena_upper,
        speed_limit=scale * scenarios.speed_limit,
    )
    scaled_rollout = replace(
        rollout_config,
        action_limit=scale * rollout_config.action_limit,
        safety_margin=scale * rollout_config.safety_margin,
    )
    scaled_actor = replace(actor_config, residual_scale=scale * actor_config.residual_scale)
    scaled = shared_actor_library_loss(
        scaled_params,
        scaled_spec,
        scale * states,
        scaled_scenarios,
        scale * targets,
        scaled_params,
        scale * descriptor_scales,
        scaled_rollout,
        scaled_actor,
        loss_config,
    )

    np.testing.assert_allclose(scaled[0], original[0], rtol=4e-6, atol=4e-6)
    for scaled_value, original_value in zip(
        jax.tree.leaves(scaled[1]), jax.tree.leaves(original[1]), strict=True
    ):
        np.testing.assert_allclose(scaled_value, original_value, rtol=4e-6, atol=4e-6)
