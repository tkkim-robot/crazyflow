"""Analytic-only Version-A CBF/HOCBF baseline with exact actuator postchecks.

This comparator deliberately has no policy-value row and no fallback policy.  It projects a
nominal airborne wrench onto Crazyflow's exact motor-force polytope and the configured analytic
CBF/HOCBF faces.  If the projection or its independent postcheck fails, the motor midpoint is
returned as explicitly degraded best effort; that outcome carries no safety certificate.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.capsules import (
    CapsuleBarrierConfig,
    CapsuleObstacleSet,
    append_capsule_halfspaces,
    continuous_capsule_halfspaces,
)
from crazyflow.safety.da_plcbf.direct_wrench import motor_forces_to_wrench, wrench_to_motor_forces
from crazyflow.safety.da_plcbf.polytope_qp import PolytopeQPResult, project_affine_polytope
from crazyflow.safety.da_plcbf.version_a_barriers import (
    ContinuousBarrierHalfspaces,
    RigidBodySafetySet,
    VersionABarrierConfig,
    VersionAModel,
    continuous_safety_halfspaces,
)
from crazyflow.safety.da_plcbf.version_a_filter import (
    ValidatedMotorPolytope,
    VersionAActuator,
    VersionAFilterConfig,
    validated_motor_polytope,
)


class AnalyticPostcheck(NamedTuple):
    """Independent actuator and analytic-barrier checks for one wrench."""

    motor_forces: Array
    minimum_motor_margin: Array
    allocation_roundtrip_error: Array
    analytic_residuals: Array
    minimum_analytic_residual: Array
    actuator_passed: Array
    analytic_passed: Array
    passed: Array


class VersionAAnalyticFilterResult(NamedTuple):
    """Applied analytic-only action, QP audit, and explicit degraded status."""

    action: Array
    qp: PolytopeQPResult
    motor_polytope: ValidatedMotorPolytope
    analytic_barriers: ContinuousBarrierHalfspaces
    qp_postcheck: AnalyticPostcheck
    applied_postcheck: AnalyticPostcheck
    qp_feasible: Array
    qp_accepted: Array
    used_midpoint: Array
    degraded: Array
    input_valid: Array
    action_executable: Array


def _positive_definite(weight: Array) -> Array:
    if weight.ndim == 1:
        return jnp.all(jnp.isfinite(weight)) & jnp.all(weight > 0)
    finite = jnp.all(jnp.isfinite(weight))
    safe = jnp.where(jnp.isfinite(weight), weight, jnp.eye(4, dtype=weight.dtype))
    symmetric = 0.5 * (safe + safe.T)
    scale = jnp.maximum(jnp.max(jnp.abs(safe)), jnp.finfo(weight.dtype).tiny)
    symmetry_valid = jnp.max(jnp.abs(safe - safe.T)) <= 1e-7 * scale
    eigenvalues = jnp.linalg.eigvalsh(symmetric)
    eigenvalue_scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), jnp.finfo(weight.dtype).tiny)
    return finite & symmetry_valid & (jnp.min(eigenvalues) > 1e-7 * eigenvalue_scale)


def _postcheck(
    wrench: Array,
    actuator: VersionAActuator,
    motor_polytope: ValidatedMotorPolytope,
    analytic: ContinuousBarrierHalfspaces,
    config: VersionAFilterConfig,
) -> AnalyticPostcheck:
    finite = jnp.all(jnp.isfinite(wrench))
    safe_wrench = jnp.where(jnp.isfinite(wrench), wrench, motor_polytope.midpoint_wrench)
    motor_forces = wrench_to_motor_forces(
        safe_wrench,
        L=actuator.arm_length,
        thrust2torque=actuator.thrust_to_torque,
        mixing_matrix=actuator.mixing_matrix,
    )
    reconstructed = motor_forces_to_wrench(
        motor_forces,
        L=actuator.arm_length,
        thrust2torque=actuator.thrust_to_torque,
        mixing_matrix=actuator.mixing_matrix,
    )
    motor_margin = jnp.min(
        jnp.concatenate(
            (motor_forces - motor_polytope.thrust_min, motor_polytope.thrust_max - motor_forces)
        )
    )
    wrench_scale = jnp.maximum(jnp.max(jnp.abs(safe_wrench)), 1.0)
    roundtrip_error = jnp.max(jnp.abs(reconstructed - safe_wrench)) / wrench_scale
    residuals = analytic.upper_bound - analytic.matrix @ safe_wrench
    minimum_residual = jnp.min(jnp.where(analytic.enabled, residuals, jnp.inf))
    actuator_passed = (
        motor_polytope.input_valid
        & finite
        & jnp.all(jnp.isfinite(motor_forces))
        & (motor_margin >= -config.motor_tolerance)
        & jnp.isfinite(roundtrip_error)
        & (roundtrip_error <= config.allocation_roundtrip_tolerance)
    )
    analytic_passed = (
        analytic.domain_valid
        & jnp.isfinite(minimum_residual)
        & (minimum_residual >= -config.barrier_tolerance)
    )
    return AnalyticPostcheck(
        motor_forces,
        motor_margin,
        roundtrip_error,
        residuals,
        minimum_residual,
        actuator_passed,
        analytic_passed,
        actuator_passed & analytic_passed,
    )


def version_a_analytic_filter(
    state: Array,
    nominal_wrench: Array,
    weight: Array,
    model: VersionAModel,
    actuator: VersionAActuator,
    safety: RigidBodySafetySet,
    barrier_config: VersionABarrierConfig,
    filter_config: VersionAFilterConfig = VersionAFilterConfig(),
    *,
    capsules: CapsuleObstacleSet | None = None,
    capsule_config: CapsuleBarrierConfig = CapsuleBarrierConfig(),
) -> VersionAAnalyticFilterResult:
    """Apply the clean-room analytic distance/limit CBF/HOCBF baseline.

    Optional capsule obstacles use the same exact piecewise capsule HOCBF rows as the full
    Version-A filter.  This keeps the comparator honest: it removes only the learned policy-value
    row, not analytic geometry that is available to every method.
    """
    barrier_config.validate()
    filter_config.validate()
    capsule_config.validate()
    if state.shape != (13,):
        raise ValueError("state must have shape (13,)")
    if nominal_wrench.shape != (4,):
        raise ValueError("nominal_wrench must have shape (4,)")
    if weight.shape not in ((4,), (4, 4)):
        raise ValueError("weight must have shape (4,) or (4, 4)")

    analytic = continuous_safety_halfspaces(state, model, safety, barrier_config)
    if capsules is not None:
        capsule_barriers = continuous_capsule_halfspaces(
            state, model, capsules, barrier_config, capsule_config
        )
        analytic = append_capsule_halfspaces(analytic, capsule_barriers)
    motor_polytope = validated_motor_polytope(
        actuator, state.dtype, allocation_model_tolerance=filter_config.allocation_model_tolerance
    )
    common_valid = (
        analytic.input_valid
        & analytic.domain_valid
        & motor_polytope.input_valid
        & jnp.all(jnp.isfinite(nominal_wrench))
        & _positive_definite(jnp.asarray(weight, dtype=state.dtype))
    )
    matrix = jnp.concatenate((motor_polytope.matrix, analytic.matrix), axis=0)
    upper_bound = jnp.concatenate((motor_polytope.upper_bound, analytic.upper_bound), axis=0)
    qp = project_affine_polytope(
        nominal_wrench,
        weight,
        matrix,
        upper_bound,
        tolerance=filter_config.qp_tolerance,
        rank_tolerance=filter_config.qp_rank_tolerance,
    )
    kkt_valid = (
        qp.feasible
        & (qp.primal_residual <= filter_config.kkt_tolerance)
        & (qp.dual_residual <= filter_config.kkt_tolerance)
        & (qp.stationarity_residual <= filter_config.kkt_tolerance)
        & (qp.complementarity_residual <= filter_config.kkt_tolerance)
    )
    qp_postcheck = _postcheck(qp.action, actuator, motor_polytope, analytic, filter_config)
    accepted = common_valid & kkt_valid & qp_postcheck.passed
    action = jnp.where(accepted, qp.action, motor_polytope.midpoint_wrench)
    action = jnp.where(motor_polytope.input_valid, action, jnp.full_like(action, jnp.nan))
    applied_postcheck = _postcheck(action, actuator, motor_polytope, analytic, filter_config)
    return VersionAAnalyticFilterResult(
        action,
        qp,
        motor_polytope,
        analytic,
        qp_postcheck,
        applied_postcheck,
        common_valid & qp.feasible,
        accepted,
        ~accepted,
        ~accepted,
        common_valid,
        applied_postcheck.actuator_passed,
    )


__all__ = ["AnalyticPostcheck", "VersionAAnalyticFilterResult", "version_a_analytic_filter"]
