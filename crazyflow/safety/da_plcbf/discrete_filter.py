"""Trust-region discrete nonlinear PL-CBF filtering with exact acceptance checks.

This module deliberately separates the optimizer's local linear model from the hard nonlinear
decision.  A linearized proposal is never executed merely because its QP is feasible: the caller's
full transition/evaluation function is run again and every finite-horizon, held-interval, and
actuator residual must pass.  The helper is plant-agnostic so the same logic can be tested on small
reference systems and applied to Crazyflow's controller, allocation, rotor, and integration stack.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.qp import project_box_halfspace

if TYPE_CHECKING:
    from collections.abc import Callable


class DiscreteActionEvaluation(NamedTuple):
    """Hard nonlinear evidence for one proposed command.

    ``next_value`` is the selected policy's hard finite-horizon value after the complete plant
    transition. ``interval_margin`` is the minimum hard physical margin during the held-control
    transition (including substeps when applicable). ``actuator_residual`` is safe when nonpositive
    and records clipping, allocation mismatch, or another caller-defined actuator violation.
    ``applied_action`` is the actual plant input after the complete command path and is logged even
    when it differs in representation from the command.
    """

    next_value: Array
    interval_margin: Array
    actuator_residual: Array
    applied_action: Array


class DiscreteFilterResult(NamedTuple):
    """Command plus the complete linearization, postcheck, and fallback audit record."""

    action: Array
    applied_action: Array
    has_certificate: Array
    input_valid: Array
    proposal_feasible: Array
    proposal_accepted: Array
    fallback_input_valid: Array
    fallback_accepted: Array
    fallback_substituted: Array
    used_fallback: Array
    degraded: Array
    linearization_action: Array
    trust_lower: Array
    trust_upper: Array
    residual_gradient: Array
    linearized_halfspace_minimum: Array
    linearization_exact_residual: Array
    proposal_exact_residual: Array
    proposal_interval_margin: Array
    proposal_actuator_residual: Array
    fallback_exact_residual: Array
    fallback_interval_margin: Array
    fallback_actuator_residual: Array
    qp_action: Array
    qp_constraint_residual: Array
    qp_multiplier: Array
    qp_objective: Array


def discrete_nonlinear_plcbf_filter(
    nominal_action: Array,
    fallback_action: Array,
    action_lower: Array,
    action_upper: Array,
    weight: Array,
    trust_radius: Array,
    current_value: Array,
    has_certificate: Array,
    evaluate_action: Callable[[Array], DiscreteActionEvaluation],
    *,
    decay: float,
    tolerance: float = 1e-6,
    qp_iterations: int = 64,
) -> DiscreteFilterResult:
    r"""Filter one command using a local discrete PL-CBF model and exact nonlinear postcheck.

    The exact condition is

    .. math::
        V(F(x,u)) - \rho V(x) \ge 0,

    where ``rho=decay`` and both ``F`` and ``V`` are supplied by ``evaluate_action``. The condition
    is differentiated at the bounded nominal command, projected through a weighted box/halfspace
    QP inside ``trust_radius``, and then evaluated again without linearization. A proposal passes
    only if its exact discrete residual, held-interval margin, and actuator residual all satisfy the
    configured tolerance. Otherwise the supplied library fallback is evaluated by the same checks.

    If the fallback is nonfinite or outside the declared command box it is *not* silently clipped;
    a bounded zero command is substituted and ``fallback_substituted``/``degraded`` expose that
    best-effort path. No result can be accepted without a nonnegative current hard certificate.
    """
    vectors = (nominal_action, fallback_action, action_lower, action_upper, weight, trust_radius)
    if nominal_action.ndim != 1 or not all(
        value.shape == nominal_action.shape for value in vectors
    ):
        raise ValueError("all action, bound, weight, and trust vectors must share one 1-D shape")
    if nominal_action.size == 0:
        raise ValueError("the action dimension must be positive")
    if jnp.ndim(current_value) != 0 or jnp.ndim(has_certificate) != 0:
        raise ValueError("current_value and has_certificate must be scalars")
    if not math.isfinite(decay) or not 0 < decay <= 1:
        raise ValueError("decay must be finite and in (0, 1]")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if isinstance(qp_iterations, bool) or not isinstance(qp_iterations, int) or qp_iterations <= 0:
        raise ValueError("qp_iterations must be a positive integer")

    dtype = jnp.result_type(*vectors, current_value, jnp.float32)
    nominal_action = jnp.asarray(nominal_action, dtype=dtype)
    fallback_action = jnp.asarray(fallback_action, dtype=dtype)
    action_lower = jnp.asarray(action_lower, dtype=dtype)
    action_upper = jnp.asarray(action_upper, dtype=dtype)
    weight = jnp.asarray(weight, dtype=dtype)
    trust_radius = jnp.asarray(trust_radius, dtype=dtype)
    current_value = jnp.asarray(current_value, dtype=dtype)
    has_certificate = jnp.asarray(has_certificate, dtype=bool)
    tolerance_array = jnp.asarray(tolerance, dtype=dtype)

    bounds_valid = (
        jnp.all(jnp.isfinite(action_lower))
        & jnp.all(jnp.isfinite(action_upper))
        & jnp.all(action_lower <= action_upper)
    )
    safe_lower = jnp.where(bounds_valid, action_lower, -jnp.ones_like(action_lower))
    safe_upper = jnp.where(bounds_valid, action_upper, jnp.ones_like(action_upper))
    nominal_finite = jnp.all(jnp.isfinite(nominal_action))
    safe_nominal = jnp.where(nominal_finite, nominal_action, jnp.zeros_like(nominal_action))
    linearization_action = jnp.clip(safe_nominal, safe_lower, safe_upper)

    optimization_inputs_valid = (
        nominal_finite
        & bounds_valid
        & jnp.all(jnp.isfinite(weight))
        & jnp.all(weight > 0)
        & jnp.all(jnp.isfinite(trust_radius))
        & jnp.all(trust_radius > 0)
        & jnp.isfinite(current_value)
    )
    trust_lower = jnp.maximum(safe_lower, linearization_action - trust_radius)
    trust_upper = jnp.minimum(safe_upper, linearization_action + trust_radius)

    def exact_residual(action: Array) -> tuple[Array, DiscreteActionEvaluation]:
        evidence = evaluate_action(action)
        return evidence.next_value - decay * current_value, evidence

    (linearization_residual, _linearization_evidence), residual_gradient = jax.value_and_grad(
        exact_residual, has_aux=True
    )(linearization_action)
    halfspace_minimum = jnp.dot(residual_gradient, linearization_action) - linearization_residual
    qp = project_box_halfspace(
        linearization_action,
        weight,
        trust_lower,
        trust_upper,
        residual_gradient,
        halfspace_minimum,
        tolerance=tolerance,
        iterations=qp_iterations,
    )

    proposal_residual, proposal_evidence = exact_residual(qp.action)
    proposal_finite = (
        jnp.all(jnp.isfinite(qp.action))
        & jnp.isfinite(proposal_residual)
        & jnp.isfinite(proposal_evidence.interval_margin)
        & jnp.isfinite(proposal_evidence.actuator_residual)
        & jnp.all(jnp.isfinite(proposal_evidence.applied_action))
    )
    proposal_in_box = jnp.all(qp.action >= trust_lower - tolerance_array) & jnp.all(
        qp.action <= trust_upper + tolerance_array
    )
    current_certificate_valid = (
        has_certificate & jnp.isfinite(current_value) & (current_value >= -tolerance_array)
    )
    proposal_base_valid = current_certificate_valid & optimization_inputs_valid
    proposal_accepted = (
        proposal_base_valid
        & qp.feasible
        & proposal_finite
        & proposal_in_box
        & (proposal_residual >= -tolerance_array)
        & (proposal_evidence.interval_margin >= -tolerance_array)
        & (proposal_evidence.actuator_residual <= tolerance_array)
    )

    fallback_finite = jnp.all(jnp.isfinite(fallback_action))
    fallback_in_box = jnp.all(fallback_action >= safe_lower - tolerance_array) & jnp.all(
        fallback_action <= safe_upper + tolerance_array
    )
    fallback_input_valid = bounds_valid & fallback_finite & fallback_in_box
    bounded_zero = jnp.clip(jnp.zeros_like(fallback_action), safe_lower, safe_upper)
    evaluated_fallback = jnp.where(fallback_input_valid, fallback_action, bounded_zero)
    fallback_substituted = ~fallback_input_valid
    fallback_residual, fallback_evidence = exact_residual(evaluated_fallback)
    fallback_evidence_finite = (
        jnp.isfinite(fallback_residual)
        & jnp.isfinite(fallback_evidence.interval_margin)
        & jnp.isfinite(fallback_evidence.actuator_residual)
        & jnp.all(jnp.isfinite(fallback_evidence.applied_action))
    )
    fallback_accepted = (
        current_certificate_valid
        & fallback_input_valid
        & fallback_evidence_finite
        & (fallback_residual >= -tolerance_array)
        & (fallback_evidence.interval_margin >= -tolerance_array)
        & (fallback_evidence.actuator_residual <= tolerance_array)
    )

    use_fallback = ~proposal_accepted
    action = jnp.where(proposal_accepted, qp.action, evaluated_fallback)
    applied_action = jnp.where(
        proposal_accepted, proposal_evidence.applied_action, fallback_evidence.applied_action
    )
    return DiscreteFilterResult(
        action=action,
        applied_action=applied_action,
        has_certificate=has_certificate,
        input_valid=optimization_inputs_valid,
        proposal_feasible=proposal_base_valid & qp.feasible,
        proposal_accepted=proposal_accepted,
        fallback_input_valid=fallback_input_valid,
        fallback_accepted=fallback_accepted,
        fallback_substituted=fallback_substituted,
        used_fallback=use_fallback,
        degraded=(~current_certificate_valid)
        | (~optimization_inputs_valid)
        | (use_fallback & ~fallback_accepted),
        linearization_action=linearization_action,
        trust_lower=trust_lower,
        trust_upper=trust_upper,
        residual_gradient=residual_gradient,
        linearized_halfspace_minimum=halfspace_minimum,
        linearization_exact_residual=linearization_residual,
        proposal_exact_residual=proposal_residual,
        proposal_interval_margin=proposal_evidence.interval_margin,
        proposal_actuator_residual=proposal_evidence.actuator_residual,
        fallback_exact_residual=fallback_residual,
        fallback_interval_margin=fallback_evidence.interval_margin,
        fallback_actuator_residual=fallback_evidence.actuator_residual,
        qp_action=qp.action,
        qp_constraint_residual=qp.constraint_residual,
        qp_multiplier=qp.multiplier,
        qp_objective=qp.objective,
    )


__all__ = ["DiscreteActionEvaluation", "DiscreteFilterResult", "discrete_nonlinear_plcbf_filter"]
