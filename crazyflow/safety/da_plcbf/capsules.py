"""Exact fixed-shape capsule geometry for DA-PLCBF safety values.

A capsule is the Minkowski sum of a closed line segment and a sphere.  These helpers evaluate
point--capsule and swept segment--capsule clearances without replacing a capsule by sampled
spheres.  Positive dimensionless values are safe, zero is contact, and negative is penetration.
Masked slots are represented by ``+inf`` and never influence a hard minimum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from crazyflow.safety.da_plcbf.version_a_barriers import (
    ContinuousBarrierHalfspaces,
    VersionABarrierConfig,
    VersionAModel,
    validated_control_affine_terms,
)


class CapsuleObstacleSet(NamedTuple):
    """Fixed-shape capsule endpoints, radii, and enabled mask."""

    segment_start: Array
    segment_end: Array
    radii: Array
    mask: Array


class CapsuleValueResult(NamedTuple):
    """Per-capsule dimensionless values plus validated geometry metadata."""

    values: Array
    enabled: Array
    closest_fraction: Array
    input_valid: Array


class ContinuousCapsuleHalfspaces(NamedTuple):
    """Piecewise-smooth capsule HOCBF faces in ``matrix @ wrench <= upper_bound`` form."""

    matrix: Array
    upper_bound: Array
    raw_values: Array
    first_order_values: Array
    enabled: Array
    differentiable_region: Array
    input_valid: Array
    domain_valid: Array


class QuadCapsuleTrajectoryValues(NamedTuple):
    """Node, swept, hard, and conservative-soft values with leading axes ``(K, B)``."""

    node_values: Array
    node_enabled: Array
    segment_values: Array
    segment_enabled: Array
    input_valid: Array
    hard_policy_margins: Array
    smooth_policy_margins: Array


@dataclass(frozen=True, slots=True)
class CapsuleBarrierConfig:
    """Capsule-specific clearance and projection-seam tolerance."""

    clearance: float = 0.0
    projection_seam_tolerance: float = 1e-6

    def validate(self) -> None:
        """Reject settings that could silently weaken capsule constraints."""
        if not math.isfinite(self.clearance) or self.clearance < 0:
            raise ValueError("capsule clearance must be finite and nonnegative")
        if not math.isfinite(self.projection_seam_tolerance) or self.projection_seam_tolerance < 0:
            raise ValueError("projection_seam_tolerance must be finite and nonnegative")


def _validate_capsule_shapes(capsules: CapsuleObstacleSet) -> tuple[int, tuple[int, ...]]:
    start = jnp.asarray(capsules.segment_start)
    end = jnp.asarray(capsules.segment_end)
    radii = jnp.asarray(capsules.radii)
    mask = jnp.asarray(capsules.mask)
    if start.ndim < 2 or start.shape[-1] != 3:
        raise ValueError("capsule segment_start must have shape (..., capsules, 3)")
    if end.shape != start.shape:
        raise ValueError("capsule segment_end must match segment_start")
    leading = start.shape[:-2]
    capsule_count = start.shape[-2]
    if radii.shape != (*leading, capsule_count):
        raise ValueError("capsule radii must have shape (..., capsules)")
    if mask.shape != radii.shape:
        raise ValueError("capsule mask must match radii")
    if not jnp.issubdtype(mask.dtype, jnp.bool_):
        raise TypeError("capsule mask must have boolean dtype")
    if not jnp.issubdtype(start.dtype, jnp.floating):
        raise TypeError("capsule endpoints must have floating-point dtype")
    return capsule_count, leading


def validate_capsules(capsules: CapsuleObstacleSet, *, clearance: float = 0.0) -> Array:
    """Return a scalar validity flag while rejecting incompatible static shapes eagerly."""
    _validate_capsule_shapes(capsules)
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError("capsule clearance must be finite and nonnegative")
    start = jnp.asarray(capsules.segment_start)
    end = jnp.asarray(capsules.segment_end, dtype=start.dtype)
    radii = jnp.asarray(capsules.radii, dtype=start.dtype)
    mask = jnp.asarray(capsules.mask)
    real_valid = (~mask) | (
        jnp.all(jnp.isfinite(start), axis=-1)
        & jnp.all(jnp.isfinite(end), axis=-1)
        & jnp.isfinite(radii)
        & (radii > 0)
        & (radii + clearance > 0)
    )
    return jnp.all(real_valid)


def _safe_capsules(
    capsules: CapsuleObstacleSet, clearance: float, dtype: jnp.dtype
) -> tuple[Array, Array, Array, Array, Array]:
    valid = validate_capsules(capsules, clearance=clearance)
    mask = jnp.asarray(capsules.mask)
    start = jnp.asarray(capsules.segment_start, dtype=dtype)
    end = jnp.asarray(capsules.segment_end, dtype=dtype)
    radii = jnp.asarray(capsules.radii, dtype=dtype)
    safe_start = jnp.where(mask[..., None] & jnp.isfinite(start), start, 0.0)
    safe_end = jnp.where(mask[..., None] & jnp.isfinite(end), end, safe_start)
    safe_radii = jnp.where(mask & jnp.isfinite(radii) & (radii > 0), radii, 1.0)
    return safe_start, safe_end, safe_radii + clearance, mask, valid


def point_capsule_dimensionless_values(
    position: Array, capsules: CapsuleObstacleSet, *, clearance: float = 0.0
) -> CapsuleValueResult:
    """Evaluate exact point--capsule margins for matching leading batch axes."""
    _, leading = _validate_capsule_shapes(capsules)
    if position.shape != (*leading, 3):
        raise ValueError("position must have shape (..., 3) matching the capsule batch axes")
    if not jnp.issubdtype(position.dtype, jnp.floating):
        raise TypeError("position must have floating-point dtype")
    start, end, effective_radius, mask, geometry_valid = _safe_capsules(
        capsules, clearance, position.dtype
    )
    finite_position = jnp.all(jnp.isfinite(position))
    safe_position = jnp.where(jnp.isfinite(position), position, 0.0)
    spine = end - start
    relative = safe_position[..., None, :] - start
    denominator = jnp.sum(spine * spine, axis=-1)
    safe_denominator = jnp.where(denominator > 0, denominator, 1.0)
    fraction = jnp.clip(jnp.sum(relative * spine, axis=-1) / safe_denominator, 0.0, 1.0)
    fraction = jnp.where(denominator > 0, fraction, 0.0)
    closest = start + fraction[..., None] * spine
    distance_squared = jnp.sum((safe_position[..., None, :] - closest) ** 2, axis=-1)
    values = (distance_squared - effective_radius**2) / effective_radius**2
    values = jnp.where(mask, values, jnp.inf)
    finite_enabled = jnp.all(jnp.where(mask, jnp.isfinite(values), True))
    return CapsuleValueResult(
        values, mask, fraction, finite_position & geometry_valid & finite_enabled
    )


def _point_segment_distance_squared(point: Array, start: Array, end: Array) -> tuple[Array, Array]:
    displacement = end - start
    denominator = jnp.sum(displacement * displacement, axis=-1)
    safe_denominator = jnp.where(denominator > 0, denominator, 1.0)
    fraction = jnp.clip(
        jnp.sum((point - start) * displacement, axis=-1) / safe_denominator, 0.0, 1.0
    )
    fraction = jnp.where(denominator > 0, fraction, 0.0)
    closest = start + fraction[..., None] * displacement
    return jnp.sum((point - closest) ** 2, axis=-1), fraction


def swept_segment_capsule_dimensionless_values(
    motion_start: Array, motion_end: Array, capsules: CapsuleObstacleSet, *, clearance: float = 0.0
) -> CapsuleValueResult:
    """Evaluate exact swept line-segment clearance against every capsule spine.

    The minimum segment--segment distance is the minimum of the four endpoint projections and the
    unconstrained interior--interior solution when it lies in both closed segments.  This covers
    degenerate and parallel segments exactly without temporal or spatial sampling.
    """
    _, leading = _validate_capsule_shapes(capsules)
    expected = (*leading, 3)
    if motion_start.shape != expected or motion_end.shape != expected:
        raise ValueError("motion endpoints must have shape (..., 3) matching capsule batch axes")
    if not jnp.issubdtype(motion_start.dtype, jnp.floating):
        raise TypeError("motion endpoints must have floating-point dtype")
    if motion_end.dtype != motion_start.dtype:
        raise TypeError("motion endpoints must have the same dtype")
    capsule_start, capsule_end, effective_radius, mask, geometry_valid = _safe_capsules(
        capsules, clearance, motion_start.dtype
    )
    finite_motion = jnp.all(jnp.isfinite(motion_start)) & jnp.all(jnp.isfinite(motion_end))
    p0 = jnp.where(jnp.isfinite(motion_start), motion_start, 0.0)[..., None, :]
    p1 = jnp.where(jnp.isfinite(motion_end), motion_end, 0.0)[..., None, :]
    q0 = capsule_start
    q1 = capsule_end

    d_p0, t_p0 = _point_segment_distance_squared(p0, q0, q1)
    d_p1, t_p1 = _point_segment_distance_squared(p1, q0, q1)
    d_q0, s_q0 = _point_segment_distance_squared(q0, p0, p1)
    d_q1, s_q1 = _point_segment_distance_squared(q1, p0, p1)

    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = jnp.sum(u * u, axis=-1)
    c = jnp.sum(v * v, axis=-1)
    # Use the squared cross product instead of ``a * c - b**2`` so nearly parallel directions do
    # not lose their entire determinant to cancellation.  The conditioning test must be relative to
    # ``a * c``: a fixed floor such as one has units of length**4 and incorrectly classifies
    # ordinary short, perpendicular segments as parallel in float32.  A nonzero determinant below
    # the relative resolution threshold is neither treated as exactly parallel nor trusted; its
    # value is retained for diagnostics, but ``input_valid`` fails closed below.
    normal = jnp.cross(u, v)
    denominator = jnp.sum(normal * normal, axis=-1)
    nondegenerate = (a > 0) & (c > 0)
    relative_resolution = 64.0 * jnp.finfo(motion_start.dtype).eps * (a * c)
    interior_well_conditioned = nondegenerate & (denominator > relative_resolution)
    exactly_parallel = nondegenerate & (denominator == 0)
    geometry_resolved = (~nondegenerate) | exactly_parallel | interior_well_conditioned
    safe_denominator = jnp.where(interior_well_conditioned, denominator, 1.0)
    # Equivalent to ``(b*e-c*d)/denominator`` and ``(a*e-b*d)/denominator``, but the
    # scalar-triple-product form avoids another subtractive cancellation for skew short segments.
    interior_s = jnp.sum(jnp.cross(v, w) * normal, axis=-1) / safe_denominator
    interior_t = jnp.sum(jnp.cross(u, w) * normal, axis=-1) / safe_denominator
    interior_valid = (
        interior_well_conditioned
        & (interior_s >= 0)
        & (interior_s <= 1)
        & (interior_t >= 0)
        & (interior_t <= 1)
    )
    interior_difference = w + interior_s[..., None] * u - interior_t[..., None] * v
    interior_distance = jnp.where(
        interior_valid, jnp.sum(interior_difference * interior_difference, axis=-1), jnp.inf
    )

    candidates = jnp.stack((d_p0, d_p1, d_q0, d_q1, interior_distance), axis=-1)
    candidate_index = jnp.argmin(candidates, axis=-1)
    distance_squared = jnp.min(candidates, axis=-1)
    motion_fraction_candidates = jnp.stack(
        (jnp.zeros_like(t_p0), jnp.ones_like(t_p1), s_q0, s_q1, interior_s), axis=-1
    )
    motion_fraction = jnp.take_along_axis(
        motion_fraction_candidates, candidate_index[..., None], axis=-1
    )[..., 0]
    values = (distance_squared - effective_radius**2) / effective_radius**2
    values = jnp.where(mask, values, jnp.inf)
    finite_enabled = jnp.all(jnp.where(mask, jnp.isfinite(values), True))
    resolution_valid = jnp.all(jnp.where(mask, geometry_resolved, True))
    return CapsuleValueResult(
        values,
        mask,
        motion_fraction,
        finite_motion & geometry_valid & finite_enabled & resolution_valid,
    )


def quad_capsule_trajectory_values(
    states: Array,
    capsules: CapsuleObstacleSet,
    *,
    clearance: float = 0.0,
    softmin_beta: float = 40.0,
) -> QuadCapsuleTrajectoryValues:
    """Evaluate exact capsule values for quad states shaped ``(K, B, T, 13)``."""
    if states.ndim != 4 or states.shape[-1] != 13 or states.shape[2] < 2:
        raise ValueError("states must have shape (K, B, at_least_two_nodes, 13)")
    if not math.isfinite(softmin_beta) or softmin_beta <= 0:
        raise ValueError("softmin_beta must be finite and positive")
    capsule_count, leading = _validate_capsule_shapes(capsules)
    if leading != (states.shape[1],):
        raise ValueError("batched capsules must have leading shape (B,)")

    def one_scenario(
        nodes: Array, start: Array, end: Array, radii: Array, mask: Array
    ) -> tuple[Array, Array, Array, Array, Array]:
        scenario_capsules = CapsuleObstacleSet(start, end, radii, mask)
        node_results = jax.vmap(
            lambda position: point_capsule_dimensionless_values(
                position, scenario_capsules, clearance=clearance
            )
        )(nodes[:, :3])
        segment_results = jax.vmap(
            lambda first, second: swept_segment_capsule_dimensionless_values(
                first, second, scenario_capsules, clearance=clearance
            )
        )(nodes[:-1, :3], nodes[1:, :3])
        valid = jnp.all(node_results.input_valid) & jnp.all(segment_results.input_valid)
        return (
            node_results.values,
            node_results.enabled,
            segment_results.values,
            segment_results.enabled,
            valid,
        )

    evaluate_scenarios = jax.vmap(one_scenario, in_axes=(0, 0, 0, 0, 0))

    def one_policy(policy_states: Array) -> tuple[Array, Array, Array, Array, Array]:
        return evaluate_scenarios(
            policy_states,
            capsules.segment_start,
            capsules.segment_end,
            capsules.radii,
            capsules.mask,
        )

    node_values, node_enabled, segment_values, segment_enabled, valid = jax.vmap(one_policy)(states)
    if capsule_count == 0:
        hard = jnp.full(states.shape[:2], jnp.inf, dtype=states.dtype)
        smooth = hard
    else:
        flattened = jnp.concatenate(
            (
                jnp.where(node_enabled, node_values, jnp.inf).reshape(*states.shape[:2], -1),
                jnp.where(segment_enabled, segment_values, jnp.inf).reshape(*states.shape[:2], -1),
            ),
            axis=-1,
        )
        hard = jnp.min(flattened, axis=-1)
        smooth = -logsumexp(-softmin_beta * flattened, axis=-1) / softmin_beta
        # Roundoff in the log-sum-exp at large positive margins can exceed the exact minimum by a
        # few ulps.  The training surrogate's contract is conservative, so clamp only that
        # numerical excess to the already-computed exact hard value.
        smooth = jnp.minimum(smooth, hard)
        hard = jnp.where(valid, hard, -jnp.inf)
        smooth = jnp.where(valid, smooth, -jnp.inf)
    return QuadCapsuleTrajectoryValues(
        node_values, node_enabled, segment_values, segment_enabled, valid, hard, smooth
    )


def continuous_capsule_halfspaces(
    state: Array,
    model: VersionAModel,
    capsules: CapsuleObstacleSet,
    barrier_config: VersionABarrierConfig,
    capsule_config: CapsuleBarrierConfig = CapsuleBarrierConfig(),
) -> ContinuousCapsuleHalfspaces:
    r"""Construct exact static-capsule relative-degree-two HOCBF wrench faces.

    For a point outside a capsule, ``h`` is squared distance to its spine minus squared effective
    radius.  The squared distance to a closed segment is continuously differentiable.  Its Hessian
    is piecewise constant: ``2 I`` in endpoint regions and ``2 (I-vv^T/||v||^2)`` in the interior
    projection region.  At an exact projection seam the Hessian is not unique, so the row is
    retained for reporting but ``input_valid`` fails closed instead of selecting an arbitrary
    second derivative.
    """
    barrier_config.validate()
    capsule_config.validate()
    _validate_capsule_shapes(capsules)
    if state.shape != (13,):
        raise ValueError("state must have shape (13,)")
    if jnp.asarray(capsules.segment_start).ndim != 2:
        raise ValueError("continuous capsule halfspaces require an unbatched capsule set")

    start, end, effective_radius, mask, geometry_valid = _safe_capsules(
        capsules, capsule_config.clearance, state.dtype
    )
    control_terms = validated_control_affine_terms(
        state,
        model,
        model_tolerance=barrier_config.model_tolerance,
        quaternion_norm_tolerance=barrier_config.quaternion_norm_tolerance,
    )
    safe_state = jnp.where(jnp.isfinite(state), state, jnp.zeros_like(state))
    position = safe_state[:3]
    velocity = safe_state[7:10]
    spine = end - start
    relative = position[None, :] - start
    length_squared = jnp.sum(spine * spine, axis=-1)
    safe_length_squared = jnp.where(length_squared > 0, length_squared, 1.0)
    raw_fraction = jnp.sum(relative * spine, axis=-1) / safe_length_squared
    fraction = jnp.where(length_squared > 0, jnp.clip(raw_fraction, 0.0, 1.0), 0.0)
    closest = start + fraction[:, None] * spine
    displacement = position[None, :] - closest
    raw_values = jnp.sum(displacement * displacement, axis=-1) - effective_radius**2
    gradients = 2.0 * displacement

    identity = jnp.eye(3, dtype=state.dtype)
    direction_outer = spine[:, :, None] * spine[:, None, :] / safe_length_squared[:, None, None]
    interior_hessian = 2.0 * (identity[None, :, :] - direction_outer)
    endpoint_hessian = jnp.broadcast_to(2.0 * identity, interior_hessian.shape)
    interior = (length_squared > 0) & (raw_fraction > 0.0) & (raw_fraction < 1.0)
    hessian = jnp.where(interior[:, None, None], interior_hessian, endpoint_hessian)

    seam_tolerance = capsule_config.projection_seam_tolerance
    differentiable_region = (length_squared == 0) | (
        (jnp.abs(raw_fraction) > seam_tolerance) & (jnp.abs(raw_fraction - 1.0) > seam_tolerance)
    )
    h_dot = gradients @ velocity
    curvature = jnp.einsum("i,nij,j->n", velocity, hessian, velocity)
    acceleration_drift = control_terms.terms.drift[7:10]
    acceleration_control = control_terms.terms.input_matrix[7:10, :]
    control = gradients @ acceleration_control
    drift = curvature + gradients @ acceleration_drift
    first_order = h_dot + barrier_config.position_alpha_1 * raw_values
    upper_bound = (
        drift
        + (barrier_config.position_alpha_1 + barrier_config.position_alpha_2) * h_dot
        + barrier_config.position_alpha_1 * barrier_config.position_alpha_2 * raw_values
    )
    matrix = -control

    matrix = jnp.where(mask[:, None], matrix, 0.0)
    upper_bound = jnp.where(mask, upper_bound, 1.0)
    raw_values = jnp.where(mask, raw_values, jnp.inf)
    first_order = jnp.where(mask, first_order, jnp.inf)
    differentiable_region = jnp.where(mask, differentiable_region, True)
    finite = (
        jnp.all(jnp.where(mask[:, None], jnp.isfinite(matrix), True))
        & jnp.all(jnp.where(mask, jnp.isfinite(upper_bound), True))
        & jnp.all(jnp.where(mask, jnp.isfinite(raw_values), True))
        & jnp.all(jnp.where(mask, jnp.isfinite(first_order), True))
    )
    input_valid = (
        geometry_valid
        & control_terms.input_valid
        & finite
        & jnp.all(jnp.where(mask, differentiable_region, True))
    )
    in_domain = (raw_values >= -barrier_config.domain_tolerance) & (
        first_order >= -barrier_config.domain_tolerance
    )
    domain_valid = input_valid & jnp.all(jnp.where(mask, in_domain, True))
    return ContinuousCapsuleHalfspaces(
        matrix,
        upper_bound,
        raw_values,
        first_order,
        mask,
        differentiable_region,
        input_valid,
        domain_valid,
    )


def append_capsule_halfspaces(
    base: ContinuousBarrierHalfspaces, capsules: ContinuousCapsuleHalfspaces
) -> ContinuousBarrierHalfspaces:
    """Append enabled capsule rows to an existing Version-A analytic barrier set."""
    if base.matrix.ndim != 2 or base.matrix.shape[-1] != 4:
        raise ValueError("base barrier matrix must have shape (constraints, 4)")
    if capsules.matrix.ndim != 2 or capsules.matrix.shape[-1] != 4:
        raise ValueError("capsule barrier matrix must have shape (capsules, 4)")
    capsule_count = capsules.matrix.shape[0]
    return ContinuousBarrierHalfspaces(
        matrix=jnp.concatenate((base.matrix, capsules.matrix), axis=0),
        upper_bound=jnp.concatenate((base.upper_bound, capsules.upper_bound), axis=0),
        raw_values=jnp.concatenate((base.raw_values, capsules.raw_values), axis=0),
        first_order_values=jnp.concatenate(
            (base.first_order_values, capsules.first_order_values), axis=0
        ),
        relative_degrees=jnp.concatenate(
            (base.relative_degrees, jnp.full((capsule_count,), 2, dtype=jnp.int32)), axis=0
        ),
        enabled=jnp.concatenate((base.enabled, capsules.enabled), axis=0),
        relative_degree_valid=jnp.concatenate(
            (base.relative_degree_valid, capsules.differentiable_region), axis=0
        ),
        input_valid=base.input_valid & capsules.input_valid,
        domain_valid=base.domain_valid & capsules.domain_valid,
    )


__all__ = [
    "CapsuleBarrierConfig",
    "CapsuleObstacleSet",
    "CapsuleValueResult",
    "ContinuousCapsuleHalfspaces",
    "QuadCapsuleTrajectoryValues",
    "append_capsule_halfspaces",
    "continuous_capsule_halfspaces",
    "point_capsule_dimensionless_values",
    "quad_capsule_trajectory_values",
    "swept_segment_capsule_dimensionless_values",
    "validate_capsules",
]
