import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.bptt import build_bptt_functions, tree_all_finite
from crazyflow.safety.da_plcbf.config import LibraryLossConfig, RolloutConfig
from crazyflow.safety.da_plcbf.losses import library_loss
from crazyflow.safety.da_plcbf.rollouts import rollout_structured_library
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.values import hard_policy_margins, swept_trajectory_constraints


def _bptt_problem() -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    CircleScenarioBatch,
    jax.Array,
    jax.Array,
    RolloutConfig,
    LibraryLossConfig,
]:
    active_params = jnp.array([[0.5, -0.25], [-0.4, 0.3]])
    candidate_params = active_params + jnp.array([[0.8, -0.6], [-0.5, 0.7]])
    initial_states = jnp.array([[0.0, 0.0, 0.1, -0.2]])
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[8.0, 8.0]]]),
        obstacle_radii=jnp.array([[0.5]]),
        obstacle_mask=jnp.array([[True]]),
        arena_lower=jnp.array([[-10.0, -10.0]]),
        arena_upper=jnp.array([[10.0, 10.0]]),
        speed_limit=jnp.array([5.0]),
    )
    target_codes = jnp.zeros((2, 6))
    descriptor_scales = jnp.ones(6)
    rollout_config = RolloutConfig(
        dt=0.1, horizon=3, policy_gain=1.0, action_limit=3.0, safety_margin=0.0, softmin_beta=10.0
    )
    loss_config = LibraryLossConfig(
        covariance_regularizer=0.05,
        coverage_weight=0.0,
        redundancy_weight=0.0,
        diversity_weight=0.0,
        code_weight=0.0,
        action_weight=0.0,
        action_rate_weight=0.0,
        terminal_weight=0.0,
        trust_weight=1.0,
    )
    return (
        active_params,
        candidate_params,
        initial_states,
        scenarios,
        target_codes,
        descriptor_scales,
        rollout_config,
        loss_config,
    )


def _loss_at(
    params: jax.Array,
    initial_states: jax.Array,
    scenarios: CircleScenarioBatch,
    target_codes: jax.Array,
    active_params: jax.Array,
    descriptor_scales: jax.Array,
    rollout_config: RolloutConfig,
    loss_config: LibraryLossConfig,
) -> jax.Array:
    return library_loss(
        params,
        initial_states,
        scenarios,
        target_codes,
        active_params,
        descriptor_scales,
        rollout_config,
        loss_config,
    )[0]


def _assert_trees_allclose(first: object, second: object) -> None:
    first_leaves, first_structure = jax.tree_util.tree_flatten(first)
    second_leaves, second_structure = jax.tree_util.tree_flatten(second)
    assert first_structure == second_structure
    for first_leaf, second_leaf in zip(first_leaves, second_leaves, strict=True):
        np.testing.assert_allclose(first_leaf, second_leaf, rtol=2e-6, atol=2e-7)


def test_fixed_budget_burst_updates_only_candidate_and_reduces_loss() -> None:
    (
        active_params,
        candidate_params,
        initial_states,
        scenarios,
        target_codes,
        descriptor_scales,
        rollout_config,
        loss_config,
    ) = _bptt_problem()
    burst_steps = 6
    functions = build_bptt_functions(
        rollout_config,
        loss_config,
        learning_rate=0.05,
        max_gradient_norm=10.0,
        burst_steps=burst_steps,
    )
    state = functions.initialize(candidate_params)
    active_before = np.asarray(active_params).copy()
    initial_loss = _loss_at(
        state.params,
        initial_states,
        scenarios,
        target_codes,
        active_params,
        descriptor_scales,
        rollout_config,
        loss_config,
    )

    final_state, history = functions.burst(
        state, initial_states, scenarios, target_codes, active_params, descriptor_scales
    )
    final_loss = _loss_at(
        final_state.params,
        initial_states,
        scenarios,
        target_codes,
        active_params,
        descriptor_scales,
        rollout_config,
        loss_config,
    )

    np.testing.assert_array_equal(active_params, active_before)
    assert int(final_state.steps) == burst_steps
    assert history.gradient_norm.shape == (burst_steps,)
    assert history.parameter_delta_norm.shape == (burst_steps,)
    assert np.all(np.asarray(history.update_accepted))
    assert bool(tree_all_finite(final_state))
    assert bool(tree_all_finite(history))
    assert np.all(np.asarray(history.gradient_norm) > 0)
    assert np.all(np.asarray(history.parameter_delta_norm) > 0)
    assert not np.allclose(final_state.params, candidate_params)
    assert float(final_loss) < float(initial_loss)
    assert np.all(np.diff(np.asarray(history.loss.total)) < 0)


def test_fused_burst_reproduces_repeated_jitted_steps_and_has_exact_step_count() -> None:
    (
        active_params,
        candidate_params,
        initial_states,
        scenarios,
        target_codes,
        descriptor_scales,
        rollout_config,
        loss_config,
    ) = _bptt_problem()
    burst_steps = 4
    functions = build_bptt_functions(
        rollout_config, loss_config, learning_rate=0.03, burst_steps=burst_steps
    )
    initial = functions.initialize(candidate_params)
    arguments = (initial_states, scenarios, target_codes, active_params, descriptor_scales)

    fused_state, fused_history = functions.burst(initial, *arguments)
    reproduced_state, reproduced_history = functions.burst(initial, *arguments)
    sequential_state = initial
    sequential_metrics = []
    for _ in range(burst_steps):
        sequential_state, metrics = functions.step(sequential_state, *arguments)
        sequential_metrics.append(metrics)

    assert int(fused_state.steps) == burst_steps
    assert int(sequential_state.steps) == burst_steps
    _assert_trees_allclose(fused_state, reproduced_state)
    _assert_trees_allclose(fused_history, reproduced_history)
    _assert_trees_allclose(fused_state, sequential_state)
    np.testing.assert_allclose(
        fused_history.loss.total,
        jnp.stack([metrics.loss.total for metrics in sequential_metrics]),
        rtol=2e-6,
        atol=2e-7,
    )

    one_step_state, _ = functions.step(fused_state, *arguments)
    assert int(one_step_state.steps) == burst_steps + 1


def test_tree_all_finite_detects_nan_in_nested_pytrees_and_is_jittable() -> None:
    finite_tree = {"candidate": jnp.array([1.0, -2.0]), "optimizer": (jnp.array(3.0),)}
    nan_tree = {"candidate": jnp.array([1.0, jnp.nan]), "optimizer": (jnp.array(3.0),)}

    assert bool(tree_all_finite(finite_tree))
    assert not bool(tree_all_finite(nan_tree))
    assert bool(tree_all_finite({}))
    assert bool(jax.jit(tree_all_finite)(finite_tree))
    assert not bool(jax.jit(tree_all_finite)(nan_tree))


def _hard_library_margins(
    params: jax.Array,
    initial_states: jax.Array,
    scenarios: CircleScenarioBatch,
    config: RolloutConfig,
) -> jax.Array:
    rollouts = rollout_structured_library(params, initial_states, config, smooth_actions=True)
    constraints = swept_trajectory_constraints(
        rollouts.states,
        rollouts.actions,
        scenarios,
        config.safety_margin,
        config.dt,
        config.action_limit,
    )
    return jnp.max(hard_policy_margins(constraints), axis=0)


def test_bptt_temporal_credit_improves_future_safety_on_train_and_held_out_scenarios() -> None:
    config = RolloutConfig(
        dt=0.1, horizon=20, policy_gain=2.0, action_limit=3.0, safety_margin=0.05, softmin_beta=20.0
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
        trust_weight=0.0,
    )
    initial_params = jnp.array([[1.0, 0.1], [1.0, -0.1]])
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
    functions = build_bptt_functions(
        config, loss_config, learning_rate=0.03, max_gradient_norm=10.0, burst_steps=100
    )
    target_codes = jnp.zeros((2, 6))
    descriptor_scales = jnp.full((6,), 3.0)
    initial_state = functions.initialize(initial_params)
    train_before = _hard_library_margins(initial_params, train_states, train_scenarios, config)
    held_out_before = _hard_library_margins(
        initial_params, held_out_states, held_out_scenarios, config
    )

    final_state, history = functions.burst(
        initial_state,
        train_states,
        train_scenarios,
        target_codes,
        initial_params,
        descriptor_scales,
    )
    train_after = _hard_library_margins(final_state.params, train_states, train_scenarios, config)
    held_out_after = _hard_library_margins(
        final_state.params, held_out_states, held_out_scenarios, config
    )

    assert np.all(np.asarray(history.update_accepted))
    assert np.max(np.asarray(train_before)) < 0
    assert np.max(np.asarray(held_out_before)) < 0
    assert np.all(np.asarray(train_after) > 0)
    assert np.all(np.asarray(held_out_after) > 0)
    assert float(jnp.mean(train_after)) > float(jnp.mean(train_before)) + 0.5
    assert float(jnp.mean(held_out_after)) > float(jnp.mean(held_out_before)) + 0.5
    assert final_state.params[0, 1] > 0.5
    assert final_state.params[1, 1] < -0.5


def test_nonfinite_bptt_update_is_rejected_without_poisoning_candidate_or_optimizer() -> None:
    (
        active_params,
        candidate_params,
        initial_states,
        scenarios,
        target_codes,
        descriptor_scales,
        rollout_config,
        loss_config,
    ) = _bptt_problem()
    invalid_scenarios = scenarios.replace(
        obstacle_centers=scenarios.obstacle_centers.at[0, 0, 0].set(jnp.nan)
    )
    functions = build_bptt_functions(rollout_config, loss_config, burst_steps=1)
    initial = functions.initialize(candidate_params)

    rejected, metrics = functions.step(
        initial, initial_states, invalid_scenarios, target_codes, active_params, descriptor_scales
    )

    assert not bool(metrics.update_accepted)
    assert int(rejected.steps) == 1
    np.testing.assert_array_equal(rejected.params, initial.params)
    _assert_trees_allclose(rejected.optimizer_state, initial.optimizer_state)
    assert bool(tree_all_finite(rejected))
