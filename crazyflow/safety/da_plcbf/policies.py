"""Structured feedback policies for the DA-PLCBF reference problem."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def structured_velocity_policy(
    state: Array,
    desired_velocity: Array,
    gain: float,
    action_limit: float,
    *,
    smooth: bool,
    action_lower: Array | None = None,
    action_upper: Array | None = None,
) -> Array:
    """Track a desired velocity with the actuator bounds used by the certified rollout.

    Omitting ``action_lower`` and ``action_upper`` uses the symmetric training bounds
    ``[-action_limit, action_limit]``. Supplying them makes the fallback actor use the exact runtime
    box; a certificate therefore cannot rely on actions that will later be clipped away.
    """
    _, velocity = jnp.split(state, 2, axis=-1)
    unconstrained = gain * (desired_velocity - velocity)
    if (action_lower is None) != (action_upper is None):
        raise ValueError("action_lower and action_upper must be provided together")
    if action_lower is None:
        action_lower = -action_limit * jnp.ones_like(unconstrained)
        action_upper = action_limit * jnp.ones_like(unconstrained)
    else:
        action_lower = jnp.asarray(action_lower, dtype=unconstrained.dtype)
        action_upper = jnp.asarray(action_upper, dtype=unconstrained.dtype)
        if action_lower.shape[-1:] != unconstrained.shape[-1:]:
            raise ValueError("action bounds must match the action dimension")
    valid = jnp.isfinite(action_lower) & jnp.isfinite(action_upper) & (action_lower <= action_upper)
    if smooth:
        center = 0.5 * (action_lower + action_upper)
        half_width = 0.5 * (action_upper - action_lower)
        safe_half_width = jnp.where(half_width > 0, half_width, 1.0)
        bounded = center + half_width * jnp.tanh((unconstrained - center) / safe_half_width)
    else:
        bounded = jnp.clip(unconstrained, action_lower, action_upper)
    return jnp.where(valid, bounded, jnp.nan)
