"""Auditable small convex projections used by the PL-CBF filter."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class BoxHalfspaceResult(NamedTuple):
    """Result of a weighted projection onto a box and one lower halfspace."""

    action: Array
    feasible: Array
    multiplier: Array
    constraint_residual: Array
    objective: Array


def project_box_halfspace(
    nominal: Array,
    weight: Array,
    lower: Array,
    upper: Array,
    normal: Array,
    minimum: Array,
    *,
    tolerance: float = 1e-7,
    iterations: int = 64,
) -> BoxHalfspaceResult:
    r"""Solve a diagonal weighted minimum-intervention QP exactly up to bisection tolerance.

    The problem is

    .. math::
        \min_u \tfrac12 (u-u_\mathrm{nom})^T W (u-u_\mathrm{nom})

    subject to ``lower <= u <= upper`` and ``normal @ u >= minimum``. The scalar dual is monotone;
    fixed-iteration bisection is deterministic and JIT compatible. If the halfspace and box do not
    intersect, ``feasible`` is false and ``action`` is the box point maximising ``normal @ u``.
    """
    if not (nominal.shape == weight.shape == lower.shape == upper.shape == normal.shape):
        raise ValueError("all vector inputs must have identical shapes")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")

    finite = jnp.all(
        jnp.isfinite(nominal)
        & jnp.isfinite(weight)
        & jnp.isfinite(lower)
        & jnp.isfinite(upper)
        & jnp.isfinite(normal)
    ) & jnp.isfinite(minimum)
    valid = finite & jnp.all(weight > 0) & jnp.all(lower <= upper)

    halfspace_scale = jnp.maximum(jnp.max(jnp.abs(normal)), jnp.abs(minimum))
    halfspace_scale = jnp.where(
        jnp.isfinite(halfspace_scale) & (halfspace_scale > 0), halfspace_scale, 1.0
    )
    scaled_normal = normal / halfspace_scale
    scaled_minimum = minimum / halfspace_scale
    scaled_tolerance = tolerance / halfspace_scale

    clipped = jnp.clip(nominal, lower, upper)
    maximizer = jnp.where(scaled_normal >= 0, upper, lower)
    maximum_value = jnp.dot(scaled_normal, maximizer)
    feasible = valid & (maximum_value >= scaled_minimum - scaled_tolerance)
    already_satisfied = jnp.dot(scaled_normal, clipped) >= scaled_minimum - scaled_tolerance

    nonzero = scaled_normal != 0
    safe_normal = jnp.where(nonzero, scaled_normal, 1)
    saturation_lambda = (maximizer - nominal) * weight / safe_normal
    saturation_lambda = jnp.where(nonzero, jnp.maximum(saturation_lambda, 0), 0)
    lambda_high = jnp.max(saturation_lambda) + 1

    def bisect(_: int, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
        lambda_low, lambda_upper = bounds
        midpoint = 0.5 * (lambda_low + lambda_upper)
        candidate = jnp.clip(nominal + midpoint * scaled_normal / weight, lower, upper)
        below = jnp.dot(scaled_normal, candidate) < scaled_minimum
        return (jnp.where(below, midpoint, lambda_low), jnp.where(below, lambda_upper, midpoint))

    _, multiplier = jax.lax.fori_loop(
        0, iterations, bisect, (jnp.zeros((), dtype=nominal.dtype), lambda_high)
    )
    projected = jnp.clip(nominal + multiplier * scaled_normal / weight, lower, upper)
    use_nominal = feasible & already_satisfied
    action = jnp.where(use_nominal, clipped, jnp.where(feasible, projected, maximizer))
    multiplier = jnp.where(use_nominal | ~feasible, 0, multiplier) / halfspace_scale
    residual = jnp.dot(normal, action) - minimum
    objective = 0.5 * jnp.sum(weight * (action - nominal) ** 2)
    return BoxHalfspaceResult(action, feasible, multiplier, residual, objective)
