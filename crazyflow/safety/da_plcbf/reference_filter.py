"""Auditable PL-CBF filter for the planar double-integrator reference system."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.double_integrator import double_integrator_step
from crazyflow.safety.da_plcbf.policies import structured_velocity_policy
from crazyflow.safety.da_plcbf.qp import project_box_halfspace
from crazyflow.safety.da_plcbf.rollouts import rollout_structured_library
from crazyflow.safety.da_plcbf.values import hard_policy_margins, swept_trajectory_constraints

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.config import RolloutConfig
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch


class ReferenceFilterResult(NamedTuple):
    """Action and complete certificate/selection diagnostics for one control cycle."""

    action: Array
    selected_index: Array
    has_certificate: Array
    qp_feasible: Array
    qp_accepted: Array
    used_fallback: Array
    degraded: Array
    hard_values: Array
    cbf_values: Array
    admissible_fractions: Array
    constraint_residual: Array
    postcheck_interval_margin: Array
    postcheck_discrete_residual: Array
    fallback_discrete_residual: Array
    applied_interval_margin: Array
    applied_discrete_residual: Array


def box_halfspace_fraction_2d(normal: Array, minimum: Array, lower: Array, upper: Array) -> Array:
    """Return the exact area fraction satisfying ``normal @ u >= minimum`` in a 2-D box.

    The dot product of a uniformly distributed box point is the sum of two independent uniforms.
    Its CDF has a closed-form inclusion-exclusion expression. Degenerate coefficients reduce to a
    one-dimensional uniform or a constant, without sampling or tuning parameters.
    """
    if normal.shape != (2,) or lower.shape != (2,) or upper.shape != (2,):
        raise ValueError("normal and bounds must be two-dimensional vectors")

    dtype = jnp.result_type(normal, minimum, lower, upper, jnp.float32)
    normal = jnp.asarray(normal, dtype=dtype)
    minimum = jnp.asarray(minimum, dtype=dtype)
    lower = jnp.asarray(lower, dtype=dtype)
    upper = jnp.asarray(upper, dtype=dtype)
    valid = (
        jnp.all(jnp.isfinite(normal))
        & jnp.isfinite(minimum)
        & jnp.all(jnp.isfinite(lower))
        & jnp.all(jnp.isfinite(upper))
        & jnp.all(lower <= upper)
    )
    halfspace_scale = jnp.maximum(jnp.max(jnp.abs(normal)), jnp.abs(minimum))
    halfspace_scale = jnp.where(
        jnp.isfinite(halfspace_scale) & (halfspace_scale > 0), halfspace_scale, 1.0
    )
    normal = normal / halfspace_scale
    minimum = minimum / halfspace_scale

    products_lower = normal * lower
    products_upper = normal * upper
    interval_lower = jnp.minimum(products_lower, products_upper)
    interval_upper = jnp.maximum(products_lower, products_upper)
    widths = interval_upper - interval_lower
    total_lower = jnp.sum(interval_lower)
    threshold = minimum - total_lower
    active_count = jnp.sum(widths > 0, dtype=jnp.int32)

    def constant_case() -> Array:
        return jnp.where(total_lower >= minimum, 1.0, 0.0)

    def one_dimensional_case() -> Array:
        width = jnp.max(widths)
        return jnp.clip((width - threshold) / width, 0.0, 1.0)

    def two_dimensional_case() -> Array:
        # A direct inclusion-exclusion formula catastrophically cancels when one coefficient is
        # tiny (a common autodiff result for a nearly irrelevant action axis).  This equivalent
        # piecewise CDF has the correct one-dimensional limit as the smaller width approaches zero.
        small = jnp.min(widths)
        large = jnp.max(widths)
        product_twice = 2.0 * small * large
        total = small + large
        cdf = jnp.where(
            threshold <= 0,
            0.0,
            jnp.where(
                threshold < small,
                threshold**2 / product_twice,
                jnp.where(
                    threshold <= large,
                    (threshold - 0.5 * small) / large,
                    jnp.where(
                        threshold < total, 1.0 - (total - threshold) ** 2 / product_twice, 1.0
                    ),
                ),
            ),
        )
        return jnp.clip(1.0 - cdf, 0.0, 1.0)

    fraction = jax.lax.switch(
        active_count, (constant_case, one_dimensional_case, two_dimensional_case)
    )
    support_upper = total_lower + jnp.sum(widths)
    fraction = jnp.where(
        minimum <= total_lower, 1.0, jnp.where(minimum >= support_upper, 0.0, fraction)
    )
    return jnp.where(valid, fraction, jnp.nan)


def _policy_value(
    desired_velocity: Array,
    state: Array,
    scenario: CircleScenarioBatch,
    config: RolloutConfig,
    action_lower: Array,
    action_upper: Array,
) -> Array:
    rollout = rollout_structured_library(
        desired_velocity[None, :],
        state[None, :],
        config,
        smooth_actions=True,
        action_lower=action_lower,
        action_upper=action_upper,
    )
    constraints = swept_trajectory_constraints(
        rollout.states,
        rollout.actions,
        scenario,
        config.safety_margin,
        config.dt,
        config.action_limit,
    )
    return hard_policy_margins(constraints)[0, 0]


def reference_plcbf_filter(
    desired_velocities: Array,
    state: Array,
    scenario: CircleScenarioBatch,
    nominal_action: Array,
    action_lower: Array,
    action_upper: Array,
    config: RolloutConfig,
    *,
    alpha: float = 2.0,
    qp_tolerance: float = 1e-6,
) -> ReferenceFilterResult:
    """Filter one nominal action using finite-horizon structured fallback certificates.

    ``scenario`` must contain exactly one current environment. A swept hard value defines runtime
    eligibility and the PL-CBF generalized gradient; the runtime certificate never substitutes the
    soft training value. If no policy is eligible, the result is explicitly degraded and applies
    the first action of the highest-hard-margin policy as a transparent, uncertified best effort.
    """
    config.validate()
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(qp_tolerance) or qp_tolerance < 0:
        raise ValueError("qp_tolerance must be finite and nonnegative")
    if scenario.obstacle_centers.shape[0] != 1:
        raise ValueError("the runtime reference filter accepts exactly one scenario")
    if state.ndim != 1 or state.shape[-1] != 2 * desired_velocities.shape[-1]:
        raise ValueError("state must be a single position/velocity vector")
    if nominal_action.shape != desired_velocities.shape[1:]:
        raise ValueError("nominal action dimension does not match the policy library")
    if nominal_action.shape != (2,):
        raise ValueError("the reference admissible-area selector is defined for 2-D actions")

    def value_and_gradient(target: Array) -> tuple[Array, Array]:
        return jax.value_and_grad(_policy_value, argnums=1)(
            target, state, scenario, config, action_lower, action_upper
        )

    cbf_values, gradients = jax.vmap(value_and_gradient)(desired_velocities)
    rollout = rollout_structured_library(
        desired_velocities,
        state[None, :],
        config,
        smooth_actions=True,
        action_lower=action_lower,
        action_upper=action_upper,
    )
    constraints = swept_trajectory_constraints(
        rollout.states,
        rollout.actions,
        scenario,
        config.safety_margin,
        config.dt,
        config.action_limit,
    )
    hard_values = hard_policy_margins(constraints)[:, 0]

    dimension = nominal_action.shape[0]
    _, velocity = jnp.split(state, 2, axis=-1)
    drift = jnp.sum(gradients[:, :dimension] * velocity[None, :], axis=-1)
    normals = gradients[:, dimension:]
    minimums = -drift - alpha * cbf_values
    admissible = jax.vmap(box_halfspace_fraction_2d, in_axes=(0, 0, None, None))(
        normals, minimums, action_lower, action_upper
    )

    finite = (
        jnp.isfinite(cbf_values)
        & jnp.isfinite(hard_values)
        & jnp.all(jnp.isfinite(gradients), axis=-1)
        & jnp.isfinite(admissible)
    )
    eligible = finite & (cbf_values > 0) & (hard_values >= 0) & (admissible > 0)
    has_certificate = jnp.any(eligible)
    safe_index = jnp.argmax(jnp.where(eligible, admissible, -jnp.inf))
    best_effort_index = jnp.argmax(jnp.where(jnp.isfinite(hard_values), hard_values, -jnp.inf))
    selected_index = jnp.where(has_certificate, safe_index, best_effort_index)

    selected_normal = normals[selected_index]
    selected_minimum = minimums[selected_index]
    qp = project_box_halfspace(
        nominal_action,
        jnp.ones_like(nominal_action),
        action_lower,
        action_upper,
        selected_normal,
        selected_minimum,
        tolerance=qp_tolerance,
    )
    qp_feasible = has_certificate & qp.feasible & (qp.constraint_residual >= -qp_tolerance)
    fallback_action = structured_velocity_policy(
        state,
        desired_velocities[selected_index],
        config.policy_gain,
        config.action_limit,
        smooth=True,
        action_lower=action_lower,
        action_upper=action_upper,
    )
    fallback_action = jnp.clip(fallback_action, action_lower, action_upper)
    fallback_finite = jnp.all(jnp.isfinite(fallback_action))
    bounded_zero = jnp.clip(jnp.zeros_like(nominal_action), action_lower, action_upper)
    fallback_action = jnp.where(fallback_finite, fallback_action, bounded_zero)

    def interval_margin(action_to_check: Array) -> tuple[Array, Array]:
        next_state = double_integrator_step(state, action_to_check, config.dt)
        interval_states = jnp.stack((state, next_state))[None, None, :, :]
        interval_actions = action_to_check[None, None, None, :]
        constraints = swept_trajectory_constraints(
            interval_states,
            interval_actions,
            scenario,
            config.safety_margin,
            config.dt,
            config.action_limit,
        )
        return jnp.min(constraints), next_state

    postcheck_interval_margin, qp_next_state = interval_margin(qp.action)
    next_cbf_value = _policy_value(
        desired_velocities[selected_index],
        qp_next_state,
        scenario,
        config,
        action_lower,
        action_upper,
    )
    postcheck_discrete_residual = (
        next_cbf_value - jnp.exp(-alpha * config.dt) * cbf_values[selected_index]
    )
    qp_accepted = (
        qp_feasible
        & jnp.isfinite(postcheck_interval_margin)
        & jnp.isfinite(postcheck_discrete_residual)
        & (postcheck_interval_margin >= -qp_tolerance)
        & (postcheck_discrete_residual >= -qp_tolerance)
    )
    fallback_interval_margin, fallback_next_state = interval_margin(fallback_action)
    fallback_next_cbf_value = _policy_value(
        desired_velocities[selected_index],
        fallback_next_state,
        scenario,
        config,
        action_lower,
        action_upper,
    )
    fallback_discrete_residual = (
        fallback_next_cbf_value - jnp.exp(-alpha * config.dt) * cbf_values[selected_index]
    )
    fallback_accepted = (
        has_certificate
        & fallback_finite
        & jnp.isfinite(fallback_interval_margin)
        & jnp.isfinite(fallback_discrete_residual)
        & (fallback_interval_margin >= -qp_tolerance)
        & (fallback_discrete_residual >= -qp_tolerance)
    )
    action = jnp.where(qp_accepted, qp.action, fallback_action)
    applied_interval_margin = jnp.where(
        qp_accepted, postcheck_interval_margin, fallback_interval_margin
    )
    applied_discrete_residual = jnp.where(
        qp_accepted, postcheck_discrete_residual, fallback_discrete_residual
    )
    return ReferenceFilterResult(
        action=action,
        selected_index=selected_index,
        has_certificate=has_certificate,
        qp_feasible=qp_feasible,
        qp_accepted=qp_accepted,
        used_fallback=~qp_accepted,
        degraded=~has_certificate | (~qp_accepted & ~fallback_accepted),
        hard_values=hard_values,
        cbf_values=cbf_values,
        admissible_fractions=admissible,
        constraint_residual=jnp.where(
            qp_accepted,
            qp.constraint_residual,
            jnp.dot(selected_normal, fallback_action) - selected_minimum,
        ),
        postcheck_interval_margin=postcheck_interval_margin,
        postcheck_discrete_residual=postcheck_discrete_residual,
        fallback_discrete_residual=fallback_discrete_residual,
        applied_interval_margin=applied_interval_margin,
        applied_discrete_residual=applied_discrete_residual,
    )
