"""Fail-closed continuous direct-wrench DA-PLCBF filter for Version A.

The filter projects a nominal wrench onto three auditable constraint groups:

* the exact coupled airborne motor-thrust polytope;
* every applicable analytic rigid-body CBF/HOCBF face; and
* one selected finite-horizon hard policy-value PL-CBF face.

An accepted wrench is independently re-evaluated against motor bounds, allocation round-trip,
analytic barrier residuals, the selected PL-CBF residual, and the QP KKT audit.  If projection is
rejected, the selected fallback wrench is used only when it passes the same checks.  Otherwise a
motor-midpoint best effort is returned with ``degraded=True``.  Invalid actuator parameters return
a NaN action sentinel: no airborne wrench can honestly be certified against an unknown polytope.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.capsules import (
    CapsuleBarrierConfig,
    CapsuleObstacleSet,
    append_capsule_halfspaces,
    continuous_capsule_halfspaces,
)
from crazyflow.safety.da_plcbf.direct_wrench import (
    motor_allocation_matrix,
    motor_forces_to_wrench,
    motor_thrust_inequalities,
    wrench_to_motor_forces,
)
from crazyflow.safety.da_plcbf.polytope_qp import PolytopeQPResult, project_affine_polytope
from crazyflow.safety.da_plcbf.selector import PolicySelection, SelectionConfig, select_hard_policy
from crazyflow.safety.da_plcbf.version_a_barriers import (
    ContinuousBarrierHalfspaces,
    RigidBodySafetySet,
    VersionABarrierConfig,
    VersionAModel,
    continuous_safety_halfspaces,
    validated_control_affine_terms,
)


class VersionAActuator(NamedTuple):
    """Unclipped Crazyflow motor allocation and airborne per-motor force limits."""

    arm_length: Array
    thrust_to_torque: Array
    mixing_matrix: Array
    thrust_min: Array
    thrust_max: Array


class PolicyLibraryCertificates(NamedTuple):
    """Hard finite-horizon values, generalized gradients, and first fallback wrenches.

    ``values`` and ``gradients`` should come from
    :func:`hard_finite_horizon_policy_certificate`.  ``gradient_valid`` must remain false for
    nonfinite rollouts or tied active minima.  All arrays have a common leading policy dimension.
    """

    values: Array
    gradients: Array
    gradient_valid: Array
    fallback_wrenches: Array


@dataclass(frozen=True, slots=True)
class VersionAFilterConfig:
    """Continuous-QP, postcheck, and selection tolerances.

    ``selection_requires_certified_fallback`` preserves the original conservative selector by
    default.  The minimal continuous demo disables it: that path selects a positive-valued policy
    before solving the QP and checks the selected policy action separately only when the QP fails.
    """

    policy_alpha: float = 2.0
    qp_tolerance: float = 2e-6
    qp_rank_tolerance: float = 1e-7
    kkt_tolerance: float = 5e-5
    motor_tolerance: float = 3e-6
    allocation_roundtrip_tolerance: float = 2e-5
    barrier_tolerance: float = 3e-6
    allocation_model_tolerance: float = 2e-5
    enforce_analytic_barriers: bool = True
    selection_requires_certified_fallback: bool = True

    def validate(self) -> None:
        """Reject nonfinite rates, negative tolerances, or nonboolean switches."""
        values = (
            self.policy_alpha,
            self.qp_tolerance,
            self.qp_rank_tolerance,
            self.kkt_tolerance,
            self.motor_tolerance,
            self.allocation_roundtrip_tolerance,
            self.barrier_tolerance,
            self.allocation_model_tolerance,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all Version-A filter values must be finite")
        if self.policy_alpha <= 0 or self.qp_rank_tolerance <= 0:
            raise ValueError("policy_alpha and qp_rank_tolerance must be positive")
        if (
            min(
                self.qp_tolerance,
                self.kkt_tolerance,
                self.motor_tolerance,
                self.allocation_roundtrip_tolerance,
                self.barrier_tolerance,
                self.allocation_model_tolerance,
            )
            < 0
        ):
            raise ValueError("all Version-A filter tolerances must be nonnegative")
        if not isinstance(self.enforce_analytic_barriers, bool):
            raise TypeError("enforce_analytic_barriers must be boolean")
        if not isinstance(self.selection_requires_certified_fallback, bool):
            raise TypeError("selection_requires_certified_fallback must be boolean")


class ValidatedMotorPolytope(NamedTuple):
    """Coupled motor-force faces plus allocation audit data."""

    matrix: Array
    upper_bound: Array
    allocation_matrix: Array
    force_to_wrench_matrix: Array
    thrust_min: Array
    thrust_max: Array
    midpoint_wrench: Array
    allocation_identity_error: Array
    input_valid: Array


class WrenchPostcheck(NamedTuple):
    """Independent exact checks for one proposed direct wrench."""

    finite: Array
    motor_forces: Array
    minimum_motor_margin: Array
    allocation_roundtrip_error: Array
    policy_barrier_residual: Array
    analytic_barrier_residuals: Array
    minimum_analytic_barrier_residual: Array
    actuator_passed: Array
    policy_barrier_passed: Array
    analytic_barriers_passed: Array
    passed: Array


class VersionAFilterResult(NamedTuple):
    """Applied wrench and complete selection/QP/postcheck diagnostics."""

    action: Array
    selected_index: Array
    selection: PolicySelection
    has_certificate: Array
    qp_feasible: Array
    qp_accepted: Array
    fallback_certified: Array
    used_fallback: Array
    used_midpoint: Array
    degraded: Array
    input_valid: Array
    action_executable: Array
    policy_values: Array
    policy_eligible: Array
    policy_admissible_fractions: Array
    policy_barrier_residuals_at_fallback: Array
    selected_policy_row: Array
    selected_policy_bound: Array
    motor_polytope: ValidatedMotorPolytope
    analytic_barriers: ContinuousBarrierHalfspaces
    qp: PolytopeQPResult
    qp_postcheck: WrenchPostcheck
    fallback_postcheck: WrenchPostcheck
    applied_postcheck: WrenchPostcheck


def _check_actuator_shapes(actuator: VersionAActuator) -> None:
    if jnp.asarray(actuator.arm_length).shape not in ((), (1,)):
        raise ValueError("actuator.arm_length must be scalar or shape (1,)")
    if jnp.asarray(actuator.thrust_to_torque).shape not in ((), (1,)):
        raise ValueError("actuator.thrust_to_torque must be scalar or shape (1,)")
    if jnp.asarray(actuator.mixing_matrix).shape != (3, 4):
        raise ValueError("actuator.mixing_matrix must have shape (3, 4)")
    for name in ("thrust_min", "thrust_max"):
        if jnp.asarray(getattr(actuator, name)).shape not in ((), (4,)):
            raise ValueError(f"actuator.{name} must be scalar or shape (4,)")


def validated_motor_polytope(
    actuator: VersionAActuator, dtype: jnp.dtype, *, allocation_model_tolerance: float = 2e-5
) -> ValidatedMotorPolytope:
    """Build the coupled airborne wrench polytope and verify both allocation maps."""
    if not math.isfinite(allocation_model_tolerance) or allocation_model_tolerance < 0:
        raise ValueError("allocation_model_tolerance must be finite and nonnegative")
    if not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError(f"dtype must be floating point, got {dtype}")
    _check_actuator_shapes(actuator)
    arm = jnp.reshape(jnp.asarray(actuator.arm_length, dtype=dtype), ())
    ratio = jnp.reshape(jnp.asarray(actuator.thrust_to_torque, dtype=dtype), ())
    mixing = jnp.asarray(actuator.mixing_matrix, dtype=dtype)
    lower = jnp.broadcast_to(jnp.asarray(actuator.thrust_min, dtype=dtype), (4,))
    upper = jnp.broadcast_to(jnp.asarray(actuator.thrust_max, dtype=dtype), (4,))
    finite = (
        jnp.isfinite(arm)
        & jnp.isfinite(ratio)
        & jnp.all(jnp.isfinite(mixing))
        & jnp.all(jnp.isfinite(lower))
        & jnp.all(jnp.isfinite(upper))
    )
    ordered = jnp.all(lower >= 0) & jnp.all(lower < upper)
    basic_valid = finite & (arm > 0) & (ratio > 0) & ordered
    canonical_mixing = jnp.asarray(
        [[-1.0, -1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]], dtype=dtype
    )
    safe_arm = jnp.where(basic_valid, arm, 1.0)
    safe_ratio = jnp.where(basic_valid, ratio, 1.0)
    safe_mixing = jnp.where(basic_valid, mixing, canonical_mixing)
    safe_lower = jnp.where(basic_valid, lower, jnp.zeros(4, dtype=dtype))
    safe_upper = jnp.where(basic_valid, upper, jnp.ones(4, dtype=dtype))
    allocation = motor_allocation_matrix(safe_mixing, L=safe_arm, thrust2torque=safe_ratio)
    motor_basis_wrenches = motor_forces_to_wrench(
        jnp.eye(4, dtype=dtype), L=safe_arm, thrust2torque=safe_ratio, mixing_matrix=safe_mixing
    )
    force_to_wrench_matrix = motor_basis_wrenches.T
    identity_error = jnp.max(jnp.abs(force_to_wrench_matrix @ allocation - jnp.eye(4, dtype=dtype)))
    input_valid = (
        basic_valid & jnp.isfinite(identity_error) & (identity_error <= allocation_model_tolerance)
    )
    constraints = motor_thrust_inequalities(
        thrust_min=safe_lower,
        thrust_max=safe_upper,
        L=safe_arm,
        thrust2torque=safe_ratio,
        mixing_matrix=safe_mixing,
    )
    midpoint_wrench = motor_forces_to_wrench(
        0.5 * (safe_lower + safe_upper),
        L=safe_arm,
        thrust2torque=safe_ratio,
        mixing_matrix=safe_mixing,
    )
    return ValidatedMotorPolytope(
        constraints.matrix,
        constraints.upper_bound,
        allocation,
        force_to_wrench_matrix,
        safe_lower,
        safe_upper,
        midpoint_wrench,
        identity_error,
        input_valid,
    )


def _check_library_shapes(library: PolicyLibraryCertificates) -> int:
    values = jnp.asarray(library.values)
    gradients = jnp.asarray(library.gradients)
    gradient_valid = jnp.asarray(library.gradient_valid)
    fallback = jnp.asarray(library.fallback_wrenches)
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError("library.values must have shape (positive_policy_count,)")
    policy_count = values.shape[0]
    if gradients.shape != (policy_count, 13):
        raise ValueError("library.gradients must have shape (policy_count, 13)")
    if gradient_valid.shape != (policy_count,) or not jnp.issubdtype(
        gradient_valid.dtype, jnp.bool_
    ):
        raise ValueError("library.gradient_valid must be boolean shape (policy_count,)")
    if fallback.shape != (policy_count, 4):
        raise ValueError("library.fallback_wrenches must have shape (policy_count, 4)")
    return policy_count


def _minimum_constraint_residual(matrix: Array, upper_bound: Array, wrench: Array) -> Array:
    if matrix.shape[0] == 0:
        return jnp.asarray(jnp.inf, dtype=wrench.dtype)
    return jnp.min(upper_bound - matrix @ wrench)


def _weight_is_positive_definite(weight: Array) -> Array:
    """Mirror the QP's weight contract before declaring a runtime input valid."""
    if weight.ndim == 1:
        return jnp.all(jnp.isfinite(weight)) & jnp.all(weight > 0)
    finite = jnp.all(jnp.isfinite(weight))
    safe_weight = jnp.where(jnp.isfinite(weight), weight, jnp.eye(4, dtype=weight.dtype))
    symmetric_weight = 0.5 * (safe_weight + safe_weight.T)
    scale = jnp.maximum(jnp.max(jnp.abs(safe_weight)), jnp.finfo(weight.dtype).tiny)
    symmetric_valid = jnp.max(jnp.abs(safe_weight - safe_weight.T)) <= 1e-7 * scale
    eigenvalues = jnp.linalg.eigvalsh(symmetric_weight)
    eigenvalue_scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), jnp.finfo(weight.dtype).tiny)
    return finite & symmetric_valid & (jnp.min(eigenvalues) > 1e-7 * eigenvalue_scale)


def motor_box_halfspace_fraction(
    wrench_row: Array, upper_bound: Array, motor_polytope: ValidatedMotorPolytope
) -> Array:
    """Return the exact motor-box volume fraction satisfying one wrench halfspace.

    The affine allocation is inverted first, so the problem is the CDF of a weighted sum of four
    independent uniform motor forces.  Inclusion-exclusion evaluates that CDF exactly for every
    nondegenerate coefficient count from one through four; zero coefficients reduce dimension
    without sampling.
    """
    if wrench_row.shape != (4,):
        raise ValueError("wrench_row must have shape (4,)")
    upper_bound = jnp.reshape(jnp.asarray(upper_bound, dtype=wrench_row.dtype), ())
    valid = (
        motor_polytope.input_valid & jnp.all(jnp.isfinite(wrench_row)) & jnp.isfinite(upper_bound)
    )
    scale = jnp.maximum(jnp.max(jnp.abs(wrench_row)), jnp.abs(upper_bound))
    scale = jnp.where(jnp.isfinite(scale) & (scale > 0), scale, 1.0)
    row = jnp.where(jnp.isfinite(wrench_row), wrench_row / scale, 0.0)
    bound = jnp.where(jnp.isfinite(upper_bound), upper_bound / scale, 0.0)
    coefficients = row @ motor_polytope.force_to_wrench_matrix
    lower_products = coefficients * motor_polytope.thrust_min
    upper_products = coefficients * motor_polytope.thrust_max
    interval_lower = jnp.minimum(lower_products, upper_products)
    widths = jnp.maximum(jnp.maximum(lower_products, upper_products) - interval_lower, 0.0)
    width_scale = jnp.maximum(jnp.max(widths), jnp.finfo(wrench_row.dtype).tiny)
    numerically_active = widths > 64.0 * jnp.finfo(wrench_row.dtype).eps * width_scale
    widths = jnp.where(numerically_active, widths, 0.0)
    sorted_widths = jnp.sort(widths)[::-1]
    active_count = jnp.sum(sorted_widths > 0, dtype=jnp.int32)
    threshold = bound - jnp.sum(interval_lower)
    support_upper = jnp.sum(widths)

    def constant_case() -> Array:
        return jnp.where(threshold >= 0, 1.0, 0.0)

    def cdf(active_dimension: int) -> Array:
        active_widths = sorted_widths[:active_dimension]
        subset_masks = jnp.asarray(
            tuple(itertools.product((0.0, 1.0), repeat=active_dimension)), dtype=wrench_row.dtype
        )
        offsets = subset_masks @ active_widths
        signs = jnp.where(jnp.sum(subset_masks, axis=-1).astype(jnp.int32) % 2 == 0, 1.0, -1.0)
        numerator = jnp.sum(signs * jnp.maximum(threshold - offsets, 0.0) ** active_dimension)
        denominator = math.factorial(active_dimension) * jnp.prod(active_widths)
        return numerator / denominator

    fraction = jax.lax.switch(
        active_count,
        (constant_case, lambda: cdf(1), lambda: cdf(2), lambda: cdf(3), lambda: cdf(4)),
    )
    fraction = jnp.where(
        active_count == 0,
        constant_case(),
        jnp.where(
            threshold <= 0,
            0.0,
            jnp.where(threshold >= support_upper, 1.0, jnp.clip(fraction, 0.0, 1.0)),
        ),
    )
    return jnp.where(valid & jnp.isfinite(fraction), fraction, jnp.nan)


def _wrench_postcheck(
    wrench: Array,
    *,
    actuator: VersionAActuator,
    motor_polytope: ValidatedMotorPolytope,
    policy_row: Array,
    policy_bound: Array,
    analytic_matrix: Array,
    analytic_bound: Array,
    has_certificate: Array,
    analytic_domain_valid: Array,
    config: VersionAFilterConfig,
) -> WrenchPostcheck:
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
    policy_residual = policy_bound - jnp.dot(policy_row, safe_wrench)
    analytic_residuals = analytic_bound - analytic_matrix @ safe_wrench
    analytic_residual = _minimum_constraint_residual(analytic_matrix, analytic_bound, safe_wrench)
    actuator_passed = (
        motor_polytope.input_valid
        & finite
        & jnp.all(jnp.isfinite(motor_forces))
        & (motor_margin >= -config.motor_tolerance)
        & jnp.isfinite(roundtrip_error)
        & (roundtrip_error <= config.allocation_roundtrip_tolerance)
    )
    policy_passed = (
        has_certificate
        & jnp.isfinite(policy_residual)
        & (policy_residual >= -config.barrier_tolerance)
    )
    analytic_passed = (~jnp.asarray(config.enforce_analytic_barriers)) | (
        analytic_domain_valid
        & jnp.isfinite(analytic_residual)
        & (analytic_residual >= -config.barrier_tolerance)
    )
    return WrenchPostcheck(
        finite,
        motor_forces,
        motor_margin,
        roundtrip_error,
        policy_residual,
        analytic_residuals,
        analytic_residual,
        actuator_passed,
        policy_passed,
        analytic_passed,
        actuator_passed & policy_passed & analytic_passed,
    )


def postcheck_version_a_action(
    wrench: Array,
    actuator: VersionAActuator,
    filtered: VersionAFilterResult,
    config: VersionAFilterConfig,
) -> WrenchPostcheck:
    """Recheck a runtime-selected wrench against the filter's exact selected constraints.

    A higher-level runtime can replace the continuous filter's command after a held-step or
    equal-horizon discrete check.  In that case ``filtered.applied_postcheck`` remains evidence for
    the command returned *by the continuous filter*, not the command ultimately selected by the
    runtime.  This helper binds a fresh postcheck to that final wrench while reusing the exact
    actuator model, selected policy face, and analytic barrier set from the original decision.
    """
    config.validate()
    if wrench.shape != (4,):
        raise ValueError("wrench must have shape (4,)")
    return _wrench_postcheck(
        wrench,
        actuator=actuator,
        motor_polytope=filtered.motor_polytope,
        policy_row=filtered.selected_policy_row,
        policy_bound=filtered.selected_policy_bound,
        analytic_matrix=filtered.analytic_barriers.matrix,
        analytic_bound=filtered.analytic_barriers.upper_bound,
        has_certificate=filtered.has_certificate,
        analytic_domain_valid=filtered.analytic_barriers.domain_valid,
        config=config,
    )


def version_a_plcbf_filter(
    state: Array,
    nominal_wrench: Array,
    weight: Array,
    library: PolicyLibraryCertificates,
    model: VersionAModel,
    actuator: VersionAActuator,
    safety: RigidBodySafetySet,
    barrier_config: VersionABarrierConfig,
    filter_config: VersionAFilterConfig = VersionAFilterConfig(),
    *,
    previous_policy_index: Array | None = None,
    selection_config: SelectionConfig = SelectionConfig(),
    capsules: CapsuleObstacleSet | None = None,
    capsule_config: CapsuleBarrierConfig = CapsuleBarrierConfig(),
) -> VersionAFilterResult:
    """Project one nominal direct wrench and apply a postchecked fallback on rejection.

    Policy selection is deterministic and task-independent.  The legacy/default mode requires a
    candidate's first wrench to satisfy actuator, analytic-barrier, and PL-CBF conditions before
    selection.  When ``selection_requires_certified_fallback`` is false, a finite positive-valued
    candidate with a valid gradient and actuator-executable first wrench is selectable before the
    QP is solved.  In either mode, the selector maximizes the exact motor-box admissible-volume
    fraction.  If none qualifies, the largest finite hard value supplies an explicitly uncertified
    best effort.
    """
    barrier_config.validate()
    filter_config.validate()
    selection_config.validate()
    if selection_config.minimum_hard_value < 0:
        raise ValueError("Version-A selection minimum_hard_value must be nonnegative")
    if state.shape != (13,):
        raise ValueError("state must have shape (13,)")
    if nominal_wrench.shape != (4,):
        raise ValueError("nominal_wrench must have shape (4,)")
    if weight.shape not in ((4,), (4, 4)):
        raise ValueError("weight must have shape (4,) or (4, 4)")
    _check_library_shapes(library)
    values = jnp.asarray(library.values, dtype=state.dtype)
    gradients = jnp.asarray(library.gradients, dtype=state.dtype)
    gradient_valid = jnp.asarray(library.gradient_valid, dtype=bool)
    fallback_wrenches = jnp.asarray(library.fallback_wrenches, dtype=state.dtype)

    control_terms = validated_control_affine_terms(
        state,
        model,
        model_tolerance=barrier_config.model_tolerance,
        quaternion_norm_tolerance=barrier_config.quaternion_norm_tolerance,
    )
    analytic = continuous_safety_halfspaces(state, model, safety, barrier_config)
    if capsules is not None:
        capsule_barriers = continuous_capsule_halfspaces(
            state, model, capsules, barrier_config, capsule_config
        )
        analytic = append_capsule_halfspaces(analytic, capsule_barriers)
    motor_polytope = validated_motor_polytope(
        actuator, state.dtype, allocation_model_tolerance=filter_config.allocation_model_tolerance
    )
    certificate_finite = (
        jnp.isfinite(values)
        & jnp.all(jnp.isfinite(gradients), axis=-1)
        & jnp.all(jnp.isfinite(fallback_wrenches), axis=-1)
    )
    safe_gradients = jnp.where(jnp.isfinite(gradients), gradients, 0.0)
    policy_drift = safe_gradients @ control_terms.terms.drift
    policy_control = safe_gradients @ control_terms.terms.input_matrix
    policy_rows = -policy_control
    policy_bounds = policy_drift + filter_config.policy_alpha * jnp.where(
        jnp.isfinite(values), values, 0.0
    )
    admissible_fractions = jax.vmap(
        lambda row, bound: motor_box_halfspace_fraction(row, bound, motor_polytope)
    )(policy_rows, policy_bounds)
    fallback_policy_residual = policy_bounds - jnp.sum(policy_rows * fallback_wrenches, axis=-1)
    fallback_motor_forces = fallback_wrenches @ motor_polytope.allocation_matrix.T
    fallback_motor_margin = jnp.min(
        jnp.concatenate(
            (
                fallback_motor_forces - motor_polytope.thrust_min,
                motor_polytope.thrust_max - fallback_motor_forces,
            ),
            axis=-1,
        ),
        axis=-1,
    )
    fallback_analytic_residual = jnp.min(
        analytic.upper_bound[None, :] - fallback_wrenches @ analytic.matrix.T, axis=-1
    )
    analytic_eligible = (~jnp.asarray(filter_config.enforce_analytic_barriers)) | (
        analytic.domain_valid
        & jnp.isfinite(fallback_analytic_residual)
        & (fallback_analytic_residual >= -filter_config.barrier_tolerance)
    )
    common_valid = (
        control_terms.input_valid
        & motor_polytope.input_valid
        & analytic.input_valid
        & jnp.all(jnp.isfinite(nominal_wrench))
        & _weight_is_positive_definite(jnp.asarray(weight, dtype=state.dtype))
    )
    certified_fallback_prerequisites = (
        common_valid
        & certificate_finite
        & gradient_valid
        & (fallback_motor_margin >= -filter_config.motor_tolerance)
        & (fallback_policy_residual >= -filter_config.barrier_tolerance)
        & analytic_eligible
    )
    executable_fallback_prerequisites = (
        common_valid
        & certificate_finite
        & gradient_valid
        & (fallback_motor_margin >= -filter_config.motor_tolerance)
    )
    selection_prerequisites = jnp.where(
        jnp.asarray(filter_config.selection_requires_certified_fallback),
        certified_fallback_prerequisites,
        executable_fallback_prerequisites,
    )
    selectable_scores = jnp.where(selection_prerequisites, admissible_fractions, -jnp.inf)
    previous_index = (
        jnp.asarray(-1, dtype=jnp.int32)
        if previous_policy_index is None
        else jnp.asarray(previous_policy_index, dtype=jnp.int32)
    )
    selection = select_hard_policy(values, selectable_scores, previous_index, selection_config)
    eligible = selection.eligible
    has_certificate = selection.has_certificate
    selected_index = selection.selected_index
    selected_row = policy_rows[selected_index]
    selected_bound = policy_bounds[selected_index]

    if filter_config.enforce_analytic_barriers:
        qp_matrix = jnp.concatenate(
            (motor_polytope.matrix, analytic.matrix, selected_row[None, :]), axis=0
        )
        qp_bound = jnp.concatenate(
            (motor_polytope.upper_bound, analytic.upper_bound, selected_bound[None]), axis=0
        )
    else:
        qp_matrix = jnp.concatenate((motor_polytope.matrix, selected_row[None, :]), axis=0)
        qp_bound = jnp.concatenate((motor_polytope.upper_bound, selected_bound[None]), axis=0)
    qp = project_affine_polytope(
        nominal_wrench,
        weight,
        qp_matrix,
        qp_bound,
        tolerance=filter_config.qp_tolerance,
        rank_tolerance=filter_config.qp_rank_tolerance,
    )
    qp_kkt_valid = (
        qp.feasible
        & (qp.primal_residual <= filter_config.kkt_tolerance)
        & (qp.dual_residual <= filter_config.kkt_tolerance)
        & (qp.stationarity_residual <= filter_config.kkt_tolerance)
        & (qp.complementarity_residual <= filter_config.kkt_tolerance)
    )
    qp_postcheck = _wrench_postcheck(
        qp.action,
        actuator=actuator,
        motor_polytope=motor_polytope,
        policy_row=selected_row,
        policy_bound=selected_bound,
        analytic_matrix=analytic.matrix,
        analytic_bound=analytic.upper_bound,
        has_certificate=has_certificate,
        analytic_domain_valid=analytic.domain_valid,
        config=filter_config,
    )
    qp_accepted = common_valid & has_certificate & qp_kkt_valid & qp_postcheck.passed

    selected_fallback = fallback_wrenches[selected_index]
    fallback_postcheck = _wrench_postcheck(
        selected_fallback,
        actuator=actuator,
        motor_polytope=motor_polytope,
        policy_row=selected_row,
        policy_bound=selected_bound,
        analytic_matrix=analytic.matrix,
        analytic_bound=analytic.upper_bound,
        has_certificate=has_certificate,
        analytic_domain_valid=analytic.domain_valid,
        config=filter_config,
    )
    fallback_certified = common_valid & has_certificate & fallback_postcheck.passed
    fallback_actuator_safe = fallback_postcheck.actuator_passed
    best_effort_action = jnp.where(
        fallback_actuator_safe, selected_fallback, motor_polytope.midpoint_wrench
    )
    used_midpoint = (~qp_accepted) & (~fallback_actuator_safe)
    applied = jnp.where(qp_accepted, qp.action, best_effort_action)
    applied = jnp.where(motor_polytope.input_valid, applied, jnp.full_like(applied, jnp.nan))
    applied_postcheck = _wrench_postcheck(
        applied,
        actuator=actuator,
        motor_polytope=motor_polytope,
        policy_row=selected_row,
        policy_bound=selected_bound,
        analytic_matrix=analytic.matrix,
        analytic_bound=analytic.upper_bound,
        has_certificate=has_certificate,
        analytic_domain_valid=analytic.domain_valid,
        config=filter_config,
    )
    input_valid = common_valid & certificate_finite[selected_index] & gradient_valid[selected_index]
    return VersionAFilterResult(
        applied,
        selected_index,
        selection,
        has_certificate,
        common_valid & has_certificate & qp.feasible,
        qp_accepted,
        fallback_certified,
        ~qp_accepted,
        used_midpoint,
        (~has_certificate) | ((~qp_accepted) & (~fallback_certified)),
        input_valid,
        applied_postcheck.actuator_passed,
        values,
        eligible,
        admissible_fractions,
        fallback_policy_residual,
        selected_row,
        selected_bound,
        motor_polytope,
        analytic,
        qp,
        qp_postcheck,
        fallback_postcheck,
        applied_postcheck,
    )


__all__ = [
    "PolicyLibraryCertificates",
    "ValidatedMotorPolytope",
    "VersionAActuator",
    "VersionAFilterConfig",
    "VersionAFilterResult",
    "WrenchPostcheck",
    "motor_box_halfspace_fraction",
    "postcheck_version_a_action",
    "validated_motor_polytope",
    "version_a_plcbf_filter",
]
