"""Normalized trajectory descriptors used for library diversity and skill alignment."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def trajectory_descriptors(states: Array) -> Array:
    """Return displacement, mean velocity, and final velocity with shape ``(K, B, 3D)``."""
    if states.ndim != 4 or states.shape[-1] % 2:
        raise ValueError("states must have shape (K, B, T, 2 * D)")
    dimension = states.shape[-1] // 2
    position, velocity = jnp.split(states, 2, axis=-1)
    displacement = position[:, :, -1] - position[:, :, 0]
    mean_velocity = jnp.mean(velocity, axis=2)
    final_velocity = velocity[:, :, -1]
    return jnp.concatenate((displacement, mean_velocity, final_velocity), axis=-1).reshape(
        states.shape[0], states.shape[1], 3 * dimension
    )


__all__ = ["trajectory_descriptors"]
