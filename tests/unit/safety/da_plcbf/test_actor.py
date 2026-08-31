import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.actor import (
    SharedActorConfig,
    SharedActorSpec,
    actor_observation_size,
    initialize_shared_actor,
    shared_fallback_actions,
)
from crazyflow.safety.da_plcbf.actor_rollouts import rollout_shared_actor_library
from crazyflow.safety.da_plcbf.config import RolloutConfig
from crazyflow.safety.da_plcbf.policies import structured_velocity_policy
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch


def _problem() -> tuple[SharedActorSpec, CircleScenarioBatch, RolloutConfig, SharedActorConfig]:
    spec = SharedActorSpec(
        base_codes=jnp.array(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, -1.0, -0.5]]
        ),
        base_desired_velocities=jnp.array([[0.6, 0.0], [-0.6, 0.0], [0.0, 0.6], [0.0, -0.6]]),
        base_durations=jnp.array([2.0, 2.0, 1.2, 1.2]),
        adaptive_mask=jnp.array([False, False, True, True]),
    )
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[-0.2, 0.3], [jnp.nan, jnp.nan]], [[0.5, -0.4], [0.0, 0.8]]]),
        obstacle_radii=jnp.array([[0.3, jnp.nan], [0.2, 0.25]]),
        obstacle_mask=jnp.array([[True, False], [True, True]]),
        arena_lower=jnp.array([[-3.0, -2.0], [-2.0, -3.0]]),
        arena_upper=jnp.array([[3.0, 2.0], [2.0, 3.0]]),
        speed_limit=jnp.array([2.0, 2.5]),
    )
    rollout = RolloutConfig(
        dt=0.1, horizon=20, policy_gain=1.5, action_limit=2.0, safety_margin=0.05, softmin_beta=15.0
    )
    actor = SharedActorConfig(
        hidden_width=16,
        residual_scale=0.4,
        min_duration=0.2,
        max_duration=2.0,
        duration_transition=0.1,
    )
    return spec, scenarios, rollout, actor


def test_zero_residual_initialization_matches_structured_actor_before_duration_transition() -> None:
    spec, scenarios, rollout, actor = _problem()
    params = initialize_shared_actor(
        jax.random.key(4), spec, dimension=2, n_obstacles=2, config=actor
    )
    states = jnp.broadcast_to(
        jnp.array([[-1.0, 0.2, 0.1, -0.3], [0.4, -0.7, -0.2, 0.1]])[None, :, :], (4, 2, 4)
    )

    actions = shared_fallback_actions(
        params,
        spec,
        states,
        scenarios,
        elapsed=jnp.array(0.0),
        horizon_duration=2.0,
        policy_gain=rollout.policy_gain,
        action_limit=rollout.action_limit,
        config=actor,
    )
    expected = structured_velocity_policy(
        states,
        spec.base_desired_velocities[:, None, :],
        rollout.policy_gain,
        rollout.action_limit,
        smooth=True,
    )

    np.testing.assert_allclose(actions, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(params.output_kernel, 0.0)
    np.testing.assert_array_equal(params.output_bias, 0.0)
    assert params.input_kernel.shape == (actor_observation_size(2, 2) + 3, 16)


def test_structural_core_is_exactly_immutable_under_arbitrary_candidate_parameters() -> None:
    spec, scenarios, rollout, actor = _problem()
    base = initialize_shared_actor(
        jax.random.key(1), spec, dimension=2, n_obstacles=2, config=actor
    )
    candidate = jax.tree.map(
        lambda value: value + 0.7 * jnp.arange(value.size, dtype=value.dtype).reshape(value.shape),
        base,
    )
    states = jnp.broadcast_to(jnp.array([0.2, -0.1, 0.3, -0.2]), (4, 2, 4))
    arguments = {
        "spec": spec,
        "states": states,
        "scenarios": scenarios,
        "elapsed": jnp.array(0.4),
        "horizon_duration": 2.0,
        "policy_gain": rollout.policy_gain,
        "action_limit": rollout.action_limit,
        "config": actor,
    }

    active_actions = shared_fallback_actions(base, **arguments)
    candidate_actions = shared_fallback_actions(candidate, **arguments)

    np.testing.assert_array_equal(candidate_actions[:2], active_actions[:2])
    assert not np.allclose(candidate_actions[2:], active_actions[2:])


def test_duration_mask_reaches_exact_zero_velocity_brake_tail_without_shape_change() -> None:
    spec, scenarios, rollout, actor = _problem()
    params = initialize_shared_actor(
        jax.random.key(2), spec, dimension=2, n_obstacles=2, config=actor
    )
    states = jnp.broadcast_to(jnp.array([0.0, 0.0, 0.4, -0.2]), (4, 2, 4))
    tail = shared_fallback_actions(
        params,
        spec,
        states,
        scenarios,
        elapsed=jnp.array(1.5),
        horizon_duration=2.0,
        policy_gain=rollout.policy_gain,
        action_limit=rollout.action_limit,
        config=actor,
    )
    expected_brake = structured_velocity_policy(
        states[2:], jnp.zeros((2, 1, 2)), rollout.policy_gain, rollout.action_limit, smooth=True
    )

    np.testing.assert_allclose(tail[2:], expected_brake, rtol=1e-6, atol=1e-6)
    assert tail.shape == (4, 2, 2)


def test_shared_actor_rollout_is_jittable_batched_and_masked_nan_safe() -> None:
    spec, scenarios, rollout, actor = _problem()
    params = initialize_shared_actor(
        jax.random.key(3), spec, dimension=2, n_obstacles=2, config=actor
    )
    initial_states = jnp.array([[-1.0, 0.2, 0.1, 0.0], [0.4, -0.7, -0.2, 0.1]])
    eager = rollout_shared_actor_library(params, spec, initial_states, scenarios, rollout, actor)
    compiled = jax.jit(
        lambda candidate, states, batch: rollout_shared_actor_library(
            candidate, spec, states, batch, rollout, actor
        )
    )(params, initial_states, scenarios)

    assert eager.states.shape == (4, 2, 21, 4)
    assert eager.actions.shape == (4, 2, 20, 2)
    assert np.all(np.isfinite(np.asarray(eager.states)))
    assert np.all(np.isfinite(np.asarray(eager.actions)))
    np.testing.assert_allclose(compiled.states, eager.states, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(compiled.actions, eager.actions, rtol=1e-6, atol=1e-6)

    def objective(candidate: object) -> jax.Array:
        result = rollout_shared_actor_library(
            candidate, spec, initial_states, scenarios, rollout, actor
        )
        return jnp.sum(result.states[:, :, -1] ** 2)

    gradient = jax.grad(objective)(params)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(gradient))


def test_shared_parameter_count_does_not_multiply_network_by_policy_count() -> None:
    spec, _scenarios, _rollout, actor = _problem()
    params = initialize_shared_actor(
        jax.random.key(9), spec, dimension=2, n_obstacles=2, config=actor
    )
    network_count = sum(
        getattr(params, name).size
        for name in (
            "input_kernel",
            "input_bias",
            "hidden_kernel",
            "hidden_bias",
            "output_kernel",
            "output_bias",
        )
    )
    per_policy_count = (
        params.code_offsets.size + params.velocity_offsets.size + params.duration_offsets.size
    )

    assert network_count > 0
    assert per_policy_count == spec.base_codes.shape[0] * (3 + 2 + 1)
    assert params.output_kernel.shape[0] == actor.hidden_width


@pytest.mark.parametrize(
    "scenario_change",
    [
        {"obstacle_centers": jnp.full((2, 2, 2), jnp.inf)},
        {"obstacle_radii": jnp.full((2, 2), jnp.inf)},
        {"arena_upper": jnp.full((2, 2), jnp.inf)},
        {"speed_limit": jnp.full((2,), jnp.inf)},
    ],
)
def test_shared_actor_device_gate_fails_closed_for_invalid_real_scenarios(
    scenario_change: dict[str, jax.Array],
) -> None:
    spec, scenarios, rollout, actor = _problem()
    params = initialize_shared_actor(
        jax.random.key(5), spec, dimension=2, n_obstacles=2, config=actor
    )
    states = jnp.zeros((4, 2, 4))
    invalid = scenarios.replace(**scenario_change)

    actions = jax.jit(
        lambda state, batch: shared_fallback_actions(
            params,
            spec,
            state,
            batch,
            elapsed=jnp.array(0.0),
            horizon_duration=2.0,
            policy_gain=rollout.policy_gain,
            action_limit=rollout.action_limit,
            config=actor,
        )
    )(states, invalid)

    assert np.isnan(np.asarray(actions)).any()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_shared_actor_rejects_invalid_numeric_runtime_settings(value: float) -> None:
    spec, scenarios, rollout, actor = _problem()
    params = initialize_shared_actor(
        jax.random.key(6), spec, dimension=2, n_obstacles=2, config=actor
    )
    with pytest.raises(ValueError, match="finite and positive"):
        shared_fallback_actions(
            params,
            spec,
            jnp.zeros((4, 2, 4)),
            scenarios,
            elapsed=jnp.array(0.0),
            horizon_duration=value,
            policy_gain=rollout.policy_gain,
            action_limit=rollout.action_limit,
            config=actor,
        )
