import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.config import RolloutConfig
from crazyflow.safety.da_plcbf.double_integrator import double_integrator_step
from crazyflow.safety.da_plcbf.policies import structured_velocity_policy
from crazyflow.safety.da_plcbf.rollouts import rollout_structured_library


def test_double_integrator_step_matches_exact_constant_acceleration_solution() -> None:
    state = jnp.array([[1.0, -2.0, 0.5, -1.5], [-3.0, 4.0, 2.0, 0.25]])
    action = jnp.array([[2.0, -4.0], [-1.0, 3.0]])
    dt = 0.2

    actual = double_integrator_step(state, action, dt)
    position = np.asarray(state[:, :2])
    velocity = np.asarray(state[:, 2:])
    expected = np.concatenate(
        (
            position + dt * velocity + 0.5 * dt**2 * np.asarray(action),
            velocity + dt * np.asarray(action),
        ),
        axis=-1,
    )

    # CUDA and NumPy use different correctly rounded float32 tanh implementations; the observed
    # worst-case difference is below five ulps and is unrelated to the policy definition.
    np.testing.assert_allclose(actual, expected, rtol=5e-7, atol=5e-7)


def test_repeated_double_integrator_steps_match_closed_form_over_total_time() -> None:
    state = jnp.array([0.3, -0.7, 1.2, -0.4])
    action = jnp.array([-0.6, 1.5])
    dt = 0.03
    steps = 17

    actual = state
    for _ in range(steps):
        actual = double_integrator_step(actual, action, dt)

    total_time = steps * dt
    expected = np.concatenate(
        (
            np.asarray(state[:2])
            + total_time * np.asarray(state[2:])
            + 0.5 * total_time**2 * np.asarray(action),
            np.asarray(state[2:]) + total_time * np.asarray(action),
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_double_integrator_autodiff_jacobians_are_the_exact_linear_system() -> None:
    state = jnp.array([0.3, -0.2, 1.1, 0.4])
    action = jnp.array([-0.7, 0.9])
    dt = 0.125

    state_jacobian = jax.jacfwd(lambda value: double_integrator_step(value, action, dt))(state)
    action_jacobian = jax.jacfwd(lambda value: double_integrator_step(state, value, dt))(action)
    identity = np.eye(2)
    expected_state = np.block([[identity, dt * identity], [np.zeros((2, 2)), identity]])
    expected_action = np.concatenate((0.5 * dt**2 * identity, dt * identity), axis=0)

    np.testing.assert_allclose(state_jacobian, expected_state, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(action_jacobian, expected_action, rtol=1e-7, atol=1e-7)


def test_double_integrator_rejects_incompatible_state_and_action_dimensions() -> None:
    with pytest.raises(ValueError, match="position and velocity"):
        double_integrator_step(jnp.zeros(5), jnp.zeros(2), 0.1)


@pytest.mark.parametrize("smooth", [False, True])
def test_structured_velocity_policy_matches_its_bounded_feedback_definition(smooth: bool) -> None:
    state = jnp.array([[0.0, 0.0, 1.0, -2.0], [4.0, -3.0, -0.5, 0.25]])
    target = jnp.array([[3.0, -3.0], [-2.0, 2.0]])
    gain = 1.7
    limit = 2.5
    unconstrained = gain * (np.asarray(target) - np.asarray(state[:, 2:]))
    expected = (
        limit * np.tanh(unconstrained / limit) if smooth else np.clip(unconstrained, -limit, limit)
    )

    actual = structured_velocity_policy(state, target, gain, limit, smooth=smooth)

    # CUDA and NumPy use different correctly rounded float32 tanh implementations; the observed
    # worst-case difference is below five ulps and is unrelated to the policy definition.
    np.testing.assert_allclose(actual, expected, rtol=5e-7, atol=5e-7)
    assert np.all(np.abs(actual) <= limit)


def _sequential_reference_rollout(
    desired_velocities: np.ndarray,
    initial_states: np.ndarray,
    config: RolloutConfig,
    *,
    smooth: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n_policies, dimension = desired_velocities.shape
    n_scenarios = initial_states.shape[0]
    states = np.empty(
        (n_policies, n_scenarios, config.horizon + 1, 2 * dimension), dtype=np.float64
    )
    actions = np.empty((n_policies, n_scenarios, config.horizon, dimension), dtype=np.float64)
    for policy_index in range(n_policies):
        for scenario_index in range(n_scenarios):
            current = initial_states[scenario_index].astype(np.float64, copy=True)
            states[policy_index, scenario_index, 0] = current
            for time_index in range(config.horizon):
                raw_action = config.policy_gain * (
                    desired_velocities[policy_index] - current[dimension:]
                )
                if smooth:
                    action = config.action_limit * np.tanh(raw_action / config.action_limit)
                else:
                    action = np.clip(raw_action, -config.action_limit, config.action_limit)
                position = current[:dimension]
                velocity = current[dimension:]
                current = np.concatenate(
                    (
                        position + config.dt * velocity + 0.5 * config.dt**2 * action,
                        velocity + config.dt * action,
                    )
                )
                actions[policy_index, scenario_index, time_index] = action
                states[policy_index, scenario_index, time_index + 1] = current
    return states, actions


@pytest.mark.parametrize("smooth", [False, True])
def test_batched_rollout_matches_independent_sequential_reference(smooth: bool) -> None:
    desired_velocities = np.array([[1.2, -0.4], [-0.8, 1.1], [0.25, 0.6]])
    initial_states = np.array([[0.0, 1.0, -0.2, 0.3], [2.0, -1.0, 0.7, -0.5]], dtype=np.float64)
    config = RolloutConfig(dt=0.07, horizon=6, policy_gain=1.8, action_limit=1.3)

    expected_states, expected_actions = _sequential_reference_rollout(
        desired_velocities, initial_states, config, smooth=smooth
    )
    actual = rollout_structured_library(
        jnp.asarray(desired_velocities), jnp.asarray(initial_states), config, smooth_actions=smooth
    )

    assert actual.states.shape == (3, 2, 7, 4)
    assert actual.actions.shape == (3, 2, 6, 2)
    np.testing.assert_allclose(actual.states, expected_states, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual.actions, expected_actions, rtol=2e-6, atol=2e-6)


def test_rollout_is_jittable_without_changing_policy_scenario_broadcasting() -> None:
    desired_velocities = jnp.array([[0.5, -0.25], [-1.0, 0.75]])
    initial_states = jnp.array(
        [[0.0, 0.0, 0.1, 0.2], [1.0, -2.0, -0.3, 0.4], [-1.0, 1.0, 0.0, 0.0]]
    )
    config = RolloutConfig(dt=0.05, horizon=4, policy_gain=2.0, action_limit=1.5)
    eager = rollout_structured_library(desired_velocities, initial_states, config)

    compiled = jax.jit(lambda targets, states: rollout_structured_library(targets, states, config))(
        desired_velocities, initial_states
    )

    np.testing.assert_allclose(compiled.states, eager.states, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(compiled.actions, eager.actions, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("desired_velocities", "initial_states", "message"),
    [
        (jnp.zeros(2), jnp.zeros((1, 4)), "rank-two"),
        (jnp.zeros((1, 2)), jnp.zeros(4), "rank-two"),
        (jnp.zeros((1, 3)), jnp.zeros((2, 4)), "twice"),
    ],
)
def test_rollout_rejects_shape_contract_violations(
    desired_velocities: jax.Array, initial_states: jax.Array, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        rollout_structured_library(desired_velocities, initial_states, RolloutConfig())
