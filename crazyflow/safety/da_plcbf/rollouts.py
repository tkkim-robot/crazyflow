"""Fixed-shape batched policy-library rollouts."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.double_integrator import double_integrator_step
from crazyflow.safety.da_plcbf.policies import structured_velocity_policy

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.config import RolloutConfig


class RolloutBatch(NamedTuple):
    """Closed-loop states and actions for every policy/scenario pair."""

    states: Array
    """Shape ``(K, B, H + 1, 2 * D)``."""
    actions: Array
    """Shape ``(K, B, H, D)``."""


def rollout_structured_library(
    desired_velocities: Array,
    initial_states: Array,
    config: RolloutConfig,
    *,
    smooth_actions: bool = True,
    action_lower: Array | None = None,
    action_upper: Array | None = None,
) -> RolloutBatch:
    """Roll out every structured fallback from every initial state in one JAX scan."""
    config.validate()
    if desired_velocities.ndim != 2 or initial_states.ndim != 2:
        raise ValueError("desired_velocities and initial_states must both be rank-two arrays")
    dimension = desired_velocities.shape[-1]
    if initial_states.shape[-1] != 2 * dimension:
        raise ValueError("initial state dimension must be twice the velocity dimension")

    n_policies = desired_velocities.shape[0]
    n_scenarios = initial_states.shape[0]
    state = jnp.broadcast_to(
        initial_states[None, :, :], (n_policies, n_scenarios, initial_states.shape[-1])
    )
    targets = desired_velocities[:, None, :]

    def scan_step(current: Array, _: None) -> tuple[Array, tuple[Array, Array]]:
        action = structured_velocity_policy(
            current,
            targets,
            config.policy_gain,
            config.action_limit,
            smooth=smooth_actions,
            action_lower=action_lower,
            action_upper=action_upper,
        )
        following = double_integrator_step(current, action, config.dt)
        return following, (following, action)

    _, (future_states, actions) = jax.lax.scan(scan_step, state, None, length=config.horizon)
    future_states = jnp.moveaxis(future_states, 0, 2)
    actions = jnp.moveaxis(actions, 0, 2)
    states = jnp.concatenate((state[:, :, None, :], future_states), axis=2)
    return RolloutBatch(states=states, actions=actions)
