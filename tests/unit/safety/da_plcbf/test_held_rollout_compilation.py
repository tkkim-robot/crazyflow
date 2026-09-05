from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.online_constant_wind import build_cf21b_version_a_resources
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig, acceleration_to_feasible_wrench
from crazyflow.safety.da_plcbf.quad_rollouts import (
    direct_wrench_symplectic_step,
    zero_order_hold_rollout,
)


@pytest.mark.parametrize("hold_steps", (1, 2, 3, 5))
def test_unrolled_inner_hold_matches_literal_execution_and_gradient(hold_steps: int) -> None:
    resources = build_cf21b_version_a_resources()
    model = resources.model._replace(wind_velocity=jnp.array((1.6, 0.8, 0.0)))
    actuator = resources.actuator
    initial = jnp.array((0, 0, 1.4, 0, 0.05, 0, np.sqrt(1 - 0.05**2), 0.1, -0.1, 0, 0, 0, 0))
    horizon, dt = 7, 0.02  # Includes a partial final hold for both2-step and3-step commands.

    def command(state: jax.Array, step: jax.Array, gain: jax.Array) -> tuple[jax.Array, ...]:
        acceleration = jnp.array((gain, -0.6 * gain, 0.2)) - 0.7 * state[7:10]
        acceleration = acceleration.at[2].add(step * 0.01)
        result = acceleration_to_feasible_wrench(
            acceleration, state[3:7], state[10:13], model, actuator, QuadPolicyConfig()
        )
        return result.wrench, result.bounded_motor_forces, jnp.asarray(step, dtype=state.dtype)

    def scanned(gain: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        return zero_order_hold_rollout(
            initial,
            lambda state, step: command(state, step, gain),
            model,
            dt=dt,
            horizon=horizon,
            command_hold_steps=hold_steps,
        )

    def literal(gain: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        state = initial
        states, commands = [], []
        for step in range(horizon):
            if step % hold_steps == 0:
                current_command = command(state, jnp.asarray(step), gain)
            state = direct_wrench_symplectic_step(state, current_command[0], model, dt)
            states.append(state)
            commands.append(current_command)
        return jnp.stack(states), tuple(jnp.stack(values) for values in zip(*commands, strict=True))

    gain = jnp.asarray(12.0)  # Includes motor saturation under exactly the same allocator.
    actual, expected = jax.block_until_ready((jax.jit(scanned)(gain), jax.jit(literal)(gain)))
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(left, right, rtol=2e-5, atol=2e-7)
    assert actual[0].shape == (horizon, 13)
    np.testing.assert_array_equal(actual[1][2], np.arange(horizon) // hold_steps * hold_steps)
    assert np.all(np.asarray(actual[1][1]) >= np.asarray(actuator.thrust_min))
    assert np.all(np.asarray(actual[1][1]) <= np.asarray(actuator.thrust_max))
    for step in range(1, horizon):
        if step % hold_steps:
            np.testing.assert_array_equal(actual[1][0][step], actual[1][0][step - 1])

    def score(result: tuple[jax.Array, tuple[jax.Array, ...]]) -> jax.Array:
        return jnp.sum(result[0][:, :3] ** 2) + 0.1 * jnp.sum(result[1][1] ** 2)

    actual_gradient = jax.jit(jax.grad(lambda parameter: score(scanned(parameter))))(gain)
    expected_gradient = jax.jit(jax.grad(lambda parameter: score(literal(parameter))))(gain)
    assert np.isfinite(actual_gradient)
    assert abs(float(actual_gradient)) > 1e-8
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=3e-5, atol=1e-8)
