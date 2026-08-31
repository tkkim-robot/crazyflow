"""Hard certificate values and conservative smooth training surrogates."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch


def conservative_softmin(
    values: Array, beta: float, axis: int | Sequence[int] | None = None, keepdims: bool = False
) -> Array:
    """Return the unnormalised log-sum-exp soft minimum.

    For finite inputs this value is no greater than the corresponding hard minimum. Consequently,
    a positive soft value implies every included value is positive. It is a training surrogate, not
    a replacement for hard validation.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    return -logsumexp(-beta * values, axis=axis, keepdims=keepdims) / beta


def _validate_trajectory_shapes(states: Array, scenarios: CircleScenarioBatch) -> int:
    """Validate common fixed-shape trajectory inputs and return the spatial dimension."""
    if states.ndim != 4:
        raise ValueError("states must have shape (K, B, T, 2 * D)")
    dimension = scenarios.obstacle_centers.shape[-1]
    if states.shape[-1] != 2 * dimension:
        raise ValueError("trajectory and scenario dimensions do not match")
    if states.shape[1] != scenarios.obstacle_centers.shape[0]:
        raise ValueError("trajectory and scenario batch sizes do not match")
    return dimension


@jax.custom_jvp
def _zero_subgradient_norm(values: Array) -> Array:
    """Exact Euclidean norm with the valid zero subgradient selected at the origin."""
    return jnp.sqrt(jnp.sum(values**2, axis=-1))


@_zero_subgradient_norm.defjvp
def _zero_subgradient_norm_jvp(
    primals: tuple[Array], tangents: tuple[Array]
) -> tuple[Array, Array]:
    (values,), (values_tangent,) = primals, tangents
    norm = jnp.sqrt(jnp.sum(values**2, axis=-1))
    denominator = jnp.where(norm > 0, norm, 1.0)
    tangent = jnp.sum(values * values_tangent, axis=-1) / denominator
    return norm, jnp.where(norm > 0, tangent, 0.0)


@jax.custom_jvp
def _zero_subgradient_abs(values: Array) -> Array:
    """Exact absolute value with zero selected from its subdifferential at the origin."""
    return jnp.abs(values)


@_zero_subgradient_abs.defjvp
def _zero_subgradient_abs_jvp(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
    (values,), (values_tangent,) = primals, tangents
    tangent = jnp.where(values > 0, values_tangent, jnp.where(values < 0, -values_tangent, 0.0))
    return jnp.abs(values), tangent


def trajectory_constraints(
    states: Array, scenarios: CircleScenarioBatch, safety_margin: float
) -> Array:
    """Evaluate dimensionless obstacle, arena, and speed barriers at trajectory nodes.

    Args:
        states: Trajectories with shape ``(K, B, T, 2 * D)``.
        scenarios: Fixed-shape scenario batch with matching ``B`` and ``D``.
        safety_margin: Extra obstacle clearance in metres.

    Returns:
        Dimensionless barrier values with shape ``(K, B, T, O + 2 * D + 1)`` and safe-positive
        sign. Circle clearance is normalized by effective radius, arena clearance by axis span,
        and squared-speed reserve by the squared speed limit. Consequently one minimum or soft
        minimum never mixes metres, squared metres, and squared metres per second.
    """
    _validate_trajectory_shapes(states, scenarios)
    if not math.isfinite(safety_margin) or safety_margin < 0:
        raise ValueError("safety_margin must be finite and nonnegative")

    position, velocity = jnp.split(states, 2, axis=-1)
    mask = scenarios.obstacle_mask
    # Sanitize padding *before* arithmetic. ``where`` after a NaN calculation has a NaN gradient
    # through JAX even when the padded branch is masked from the forward value.
    centers = jnp.where(mask[..., None], scenarios.obstacle_centers, 0.0)
    radii = jnp.where(mask, scenarios.obstacle_radii, 1.0)
    delta = position[..., None, :] - centers[None, :, None, :, :]
    effective_radius = jnp.where(mask, radii + safety_margin, 1.0)
    valid_center = jnp.all(jnp.isfinite(scenarios.obstacle_centers), axis=-1)
    valid_radius = (
        valid_center
        & jnp.isfinite(radii)
        & (radii > 0)
        & jnp.isfinite(effective_radius)
        & (effective_radius > 0)
    )
    radius_scale = jnp.where(valid_radius, effective_radius, 1.0)
    distance = _zero_subgradient_norm(delta)
    obstacle = (distance - radius_scale[None, :, None, :]) / radius_scale[None, :, None, :]
    obstacle = jnp.where(
        mask[None, :, None, :],
        jnp.where(valid_radius[None, :, None, :], obstacle, jnp.nan),
        jnp.inf,
    )

    arena_span = scenarios.arena_upper - scenarios.arena_lower
    valid_span = (
        jnp.isfinite(scenarios.arena_lower)
        & jnp.isfinite(scenarios.arena_upper)
        & jnp.isfinite(arena_span)
        & (arena_span > 0)
    )
    span_scale = jnp.where(valid_span, arena_span, 1.0)
    lower = (position - scenarios.arena_lower[None, :, None, :]) / span_scale[None, :, None, :]
    upper = (scenarios.arena_upper[None, :, None, :] - position) / span_scale[None, :, None, :]
    lower = jnp.where(valid_span[None, :, None, :], lower, jnp.nan)
    upper = jnp.where(valid_span[None, :, None, :], upper, jnp.nan)

    valid_speed = jnp.isfinite(scenarios.speed_limit) & (scenarios.speed_limit > 0)
    speed_scale = jnp.where(valid_speed, scenarios.speed_limit, 1.0)
    speed = 1.0 - jnp.sum(velocity**2, axis=-1) / speed_scale[None, :, None] ** 2
    speed = jnp.where(valid_speed[None, :, None], speed, jnp.nan)
    return jnp.concatenate((obstacle, lower, upper, speed[..., None]), axis=-1)


def swept_trajectory_constraints(
    states: Array,
    actions: Array,
    scenarios: CircleScenarioBatch,
    safety_margin: float,
    dt: float,
    action_scale: float,
) -> Array:
    """Evaluate conservative dimensionless barriers over every constant-acceleration interval.

    Circle clearance uses the exact closest point on the endpoint chord minus an upper bound on
    the parabola-to-chord deviation, ``||a|| dt² / 8``. Arena extrema are evaluated at both
    endpoints and the exact per-axis quadratic vertex. Squared speed is convex during a constant
    acceleration interval, so its maximum is attained at an endpoint. A nonnegative returned
    barrier therefore covers the complete interval, not only controller ticks.

    The guarantee assumes each endpoint is generated from the supplied start state and constant
    interval action. Authoritative DA-PLCBF rollouts satisfy that contract; callers evaluating
    arbitrary, dynamically inconsistent endpoint/action tuples must not interpret this standalone
    helper as a certificate.

    Returns:
        Values with shape ``(K, B, H, O + 2 * D + 1)``.
    """
    dimension = _validate_trajectory_shapes(states, scenarios)
    if actions.ndim != 4 or actions.shape[-1] != dimension:
        raise ValueError("actions must have shape (K, B, H, D)")
    if actions.shape[:2] != states.shape[:2] or actions.shape[2] + 1 != states.shape[2]:
        raise ValueError("actions must align with every interval between states")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(action_scale) or action_scale <= 0:
        raise ValueError("action_scale must be finite and positive")
    if not math.isfinite(safety_margin) or safety_margin < 0:
        raise ValueError("safety_margin must be finite and nonnegative")

    position, velocity = jnp.split(states, 2, axis=-1)
    position_start = position[:, :, :-1]
    position_end = position[:, :, 1:]
    velocity_start = velocity[:, :, :-1]
    velocity_end = velocity[:, :, 1:]

    mask = scenarios.obstacle_mask
    centers = jnp.where(mask[..., None], scenarios.obstacle_centers, 0.0)
    radii = jnp.where(mask, scenarios.obstacle_radii, 1.0)
    effective_radius = jnp.where(mask, radii + safety_margin, 1.0)
    valid_center = jnp.all(jnp.isfinite(scenarios.obstacle_centers), axis=-1)
    valid_radius = (
        valid_center
        & jnp.isfinite(radii)
        & (radii > 0)
        & jnp.isfinite(effective_radius)
        & (effective_radius > 0)
    )
    radius_scale = jnp.where(valid_radius, effective_radius, 1.0)

    segment = position_end - position_start
    center_offset = centers[None, :, None, :, :] - position_start[..., None, :]
    segment_squared = jnp.sum(segment**2, axis=-1)
    denominator = jnp.where(segment_squared > 0, segment_squared, 1.0)
    chord_fraction = (
        jnp.sum(center_offset * segment[..., None, :], axis=-1) / denominator[..., None]
    )
    chord_fraction = jnp.clip(chord_fraction, 0.0, 1.0)
    closest = position_start[..., None, :] + chord_fraction[..., None] * segment[..., None, :]
    chord_delta = closest - centers[None, :, None, :, :]
    chord_distance_lower = _zero_subgradient_norm(chord_delta)

    # The exact L1 norm upper-bounds L2; its chosen zero subgradient stays finite on GPU.
    acceleration_upper = jnp.sum(_zero_subgradient_abs(actions), axis=-1)
    parabola_deviation_upper = 0.125 * dt**2 * acceleration_upper
    swept_distance_lower = chord_distance_lower - parabola_deviation_upper[..., None]
    obstacle = (swept_distance_lower - radius_scale[None, :, None, :]) / radius_scale[
        None, :, None, :
    ]
    obstacle = jnp.where(
        mask[None, :, None, :],
        jnp.where(valid_radius[None, :, None, :], obstacle, jnp.nan),
        jnp.inf,
    )

    acceleration_nonzero = jnp.abs(actions) > 0
    safe_acceleration = jnp.where(acceleration_nonzero, actions, 1.0)
    vertex_time = jnp.clip(-velocity_start / safe_acceleration, 0.0, dt)
    vertex_time = jnp.where(acceleration_nonzero, vertex_time, 0.0)
    vertex_position = position_start + velocity_start * vertex_time + 0.5 * actions * vertex_time**2
    interval_minimum = jnp.minimum(jnp.minimum(position_start, position_end), vertex_position)
    interval_maximum = jnp.maximum(jnp.maximum(position_start, position_end), vertex_position)
    arena_span = scenarios.arena_upper - scenarios.arena_lower
    valid_span = (
        jnp.isfinite(scenarios.arena_lower)
        & jnp.isfinite(scenarios.arena_upper)
        & jnp.isfinite(arena_span)
        & (arena_span > 0)
    )
    span_scale = jnp.where(valid_span, arena_span, 1.0)
    lower = (interval_minimum - scenarios.arena_lower[None, :, None, :]) / span_scale[
        None, :, None, :
    ]
    upper = (scenarios.arena_upper[None, :, None, :] - interval_maximum) / span_scale[
        None, :, None, :
    ]
    lower = jnp.where(valid_span[None, :, None, :], lower, jnp.nan)
    upper = jnp.where(valid_span[None, :, None, :], upper, jnp.nan)

    valid_speed = jnp.isfinite(scenarios.speed_limit) & (scenarios.speed_limit > 0)
    speed_scale = jnp.where(valid_speed, scenarios.speed_limit, 1.0)
    speed_squared = jnp.maximum(
        jnp.sum(velocity_start**2, axis=-1), jnp.sum(velocity_end**2, axis=-1)
    )
    speed = 1.0 - speed_squared / speed_scale[None, :, None] ** 2
    speed = jnp.where(valid_speed[None, :, None], speed, jnp.nan)
    return jnp.concatenate((obstacle, lower, upper, speed[..., None]), axis=-1)


def hard_policy_margins(constraints: Array) -> Array:
    """Return the exact sampled minimum for each policy and scenario."""
    if constraints.ndim != 4:
        raise ValueError("constraints must have shape (K, B, T, C)")
    return jnp.min(constraints, axis=(-2, -1))


def training_policy_margins(constraints: Array, beta: float) -> Array:
    """Return conservative smooth minima for each policy and scenario."""
    if constraints.ndim != 4:
        raise ValueError("constraints must have shape (K, B, T, C)")
    return conservative_softmin(constraints, beta=beta, axis=(-2, -1))
