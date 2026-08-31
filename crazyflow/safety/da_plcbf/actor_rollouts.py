"""Fixed-shape rollouts for the shared latent-residual fallback library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.actor import shared_fallback_actions
from crazyflow.safety.da_plcbf.double_integrator import double_integrator_step
from crazyflow.safety.da_plcbf.rollouts import RolloutBatch

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.actor import (
        SharedActorConfig,
        SharedActorParams,
        SharedActorSpec,
    )
    from crazyflow.safety.da_plcbf.config import RolloutConfig
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch


def rollout_shared_actor_library(
    params: SharedActorParams,
    spec: SharedActorSpec,
    initial_states: Array,
    scenarios: CircleScenarioBatch,
    rollout_config: RolloutConfig,
    actor_config: SharedActorConfig,
) -> RolloutBatch:
    """Roll out every shared policy/scenario pair using the common finite horizon."""
    rollout_config.validate()
    actor_config.validate()
    if initial_states.ndim != 2:
        raise ValueError("initial_states must have shape (B, 2 * D)")
    n_policies = spec.base_codes.shape[0]
    n_scenarios = initial_states.shape[0]
    current = jnp.broadcast_to(
        initial_states[None, :, :], (n_policies, n_scenarios, initial_states.shape[-1])
    )
    horizon_duration = rollout_config.horizon * rollout_config.dt

    def step(state: Array, step_index: Array) -> tuple[Array, tuple[Array, Array]]:
        action = shared_fallback_actions(
            params,
            spec,
            state,
            scenarios,
            elapsed=step_index * rollout_config.dt,
            horizon_duration=horizon_duration,
            policy_gain=rollout_config.policy_gain,
            action_limit=rollout_config.action_limit,
            config=actor_config,
        )
        following = double_integrator_step(state, action, rollout_config.dt)
        return following, (following, action)

    _, (future_states, actions) = jax.lax.scan(
        step, current, jnp.arange(rollout_config.horizon, dtype=current.dtype)
    )
    future_states = jnp.moveaxis(future_states, 0, 2)
    actions = jnp.moveaxis(actions, 0, 2)
    states = jnp.concatenate((current[:, :, None, :], future_states), axis=2)
    return RolloutBatch(states=states, actions=actions)


__all__ = ["rollout_shared_actor_library"]
