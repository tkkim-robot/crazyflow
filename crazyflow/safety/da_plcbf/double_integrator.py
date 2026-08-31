"""Exact discrete-time planar double-integrator reference dynamics."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def double_integrator_step(state: Array, action: Array, dt: float) -> Array:
    """Advance ``[position, velocity]`` under constant acceleration for one step.

    Args:
        state: State with shape ``(..., 2 * D)``.
        action: Acceleration with shape ``(..., D)``.
        dt: Positive step duration in seconds.

    Returns:
        State at the end of the interval, with the same shape as ``state``.
    """
    if state.shape[-1] != 2 * action.shape[-1]:
        raise ValueError("state must contain position and velocity matching the action dimension")
    position, velocity = jnp.split(state, 2, axis=-1)
    next_position = position + dt * velocity + 0.5 * dt**2 * action
    next_velocity = velocity + dt * action
    return jnp.concatenate((next_position, next_velocity), axis=-1)
