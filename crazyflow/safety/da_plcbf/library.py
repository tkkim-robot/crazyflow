"""Deterministic structured policy-library construction for Crazyflow experiments."""

from __future__ import annotations

import math
from numbers import Integral

import jax.numpy as jnp
import numpy as np
from jax import Array

from crazyflow.safety.da_plcbf.actor import SharedActorParams, SharedActorSpec


def build_shared_quad_library_spec(
    *,
    policy_count: int = 64,
    code_size: int = 8,
    structural_policy_count: int = 8,
    minimum_speed: float = 0.25,
    maximum_speed: float = 1.25,
    minimum_duration: float = 0.35,
    maximum_duration: float = 1.5,
    dtype: jnp.dtype = jnp.float32,
) -> SharedActorSpec:
    """Create a deterministic task-agnostic spherical velocity/duration library.

    The first structural slots are hover/brake and the six signed Cartesian directions plus one
    slow diagonal.  All remaining directions use a Fibonacci sphere; speed and duration vary on
    independent coprime index cycles.  This is an initialization grid, not a runtime maneuver
    selector or state machine.  Every policy is evaluated concurrently by the same hard value.
    """
    for name, value in (
        ("policy_count", policy_count),
        ("code_size", code_size),
        ("structural_policy_count", structural_policy_count),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if policy_count < 8:
        raise ValueError("policy_count must be at least eight")
    if structural_policy_count < 8 or structural_policy_count > policy_count:
        raise ValueError("structural_policy_count must lie in [8, policy_count]")
    numeric = (minimum_speed, maximum_speed, minimum_duration, maximum_duration)
    if not all(math.isfinite(value) and value > 0 for value in numeric):
        raise ValueError("speed and duration bounds must be finite and positive")
    if minimum_speed > maximum_speed or minimum_duration > maximum_duration:
        raise ValueError("minimum speed/duration must not exceed its maximum")

    index = np.arange(policy_count, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / policy_count
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    directions = np.stack(
        (radius * np.cos(golden_angle * index), radius * np.sin(golden_angle * index), z), axis=-1
    )
    structural_directions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.5],
        ]
    )
    structural_directions[-1] /= np.linalg.norm(structural_directions[-1])
    directions[:8] = structural_directions

    speed_phase = ((index * 17.0) % policy_count) / max(policy_count - 1, 1)
    duration_phase = ((index * 29.0) % policy_count) / max(policy_count - 1, 1)
    speeds = minimum_speed + (maximum_speed - minimum_speed) * speed_phase
    speeds[0] = 0.0
    durations = minimum_duration + (maximum_duration - minimum_duration) * duration_phase
    desired_velocities = directions * speeds[:, None]

    # Deterministic Fourier features give every slot a distinct bounded latent identity without
    # injecting task goals. The first feature pair carries the normalized speed/duration scales.
    normalized_index = (index + 0.5) / policy_count
    features: list[np.ndarray] = []
    for feature in range(code_size):
        frequency = feature // 2 + 1
        phase = 2.0 * np.pi * frequency * normalized_index
        features.append(np.sin(phase) if feature % 2 == 0 else np.cos(phase))
    codes = np.stack(features, axis=-1)
    if code_size >= 1:
        codes[:, 0] = 2.0 * speed_phase - 1.0
    if code_size >= 2:
        codes[:, 1] = 2.0 * duration_phase - 1.0
    adaptive_mask = np.arange(policy_count) >= structural_policy_count
    return SharedActorSpec(
        base_codes=jnp.asarray(codes, dtype=dtype),
        base_desired_velocities=jnp.asarray(desired_velocities, dtype=dtype),
        base_durations=jnp.asarray(durations, dtype=dtype),
        adaptive_mask=jnp.asarray(adaptive_mask),
    )


def descriptor_targets_from_spec(spec: SharedActorSpec) -> Array:
    """Map each base skill to displacement/mean/final-velocity descriptor targets."""
    if spec.base_desired_velocities.ndim != 2 or spec.base_desired_velocities.shape[-1] != 3:
        raise ValueError("spec.base_desired_velocities must have shape (K, 3)")
    if spec.base_durations.shape != (spec.base_desired_velocities.shape[0],):
        raise ValueError("spec.base_durations must have shape (K,)")
    displacement = spec.base_desired_velocities * spec.base_durations[:, None]
    return jnp.concatenate(
        (displacement, spec.base_desired_velocities, jnp.zeros_like(spec.base_desired_velocities)),
        axis=-1,
    )


def slice_shared_actor_policy(
    params: SharedActorParams, spec: SharedActorSpec, index: Array
) -> tuple[SharedActorParams, SharedActorSpec]:
    """Select one policy slot while preserving the genuinely shared residual network."""
    if spec.base_codes.ndim != 2 or spec.base_codes.shape[0] < 1:
        raise ValueError("spec must contain at least one policy")
    if jnp.ndim(index) != 0:
        raise ValueError("index must be scalar")
    safe_index = jnp.clip(jnp.asarray(index, dtype=jnp.int32), 0, spec.base_codes.shape[0] - 1)

    def one(value: Array) -> Array:
        return jnp.take(value, safe_index, axis=0)[None, ...]

    selected_spec = SharedActorSpec(
        base_codes=one(spec.base_codes),
        base_desired_velocities=one(spec.base_desired_velocities),
        base_durations=one(spec.base_durations),
        adaptive_mask=one(spec.adaptive_mask),
    )
    selected_params = params.replace(
        code_offsets=one(params.code_offsets),
        velocity_offsets=one(params.velocity_offsets),
        duration_offsets=one(params.duration_offsets),
    )
    return selected_params, selected_spec


__all__ = [
    "build_shared_quad_library_spec",
    "descriptor_targets_from_spec",
    "slice_shared_actor_policy",
]
