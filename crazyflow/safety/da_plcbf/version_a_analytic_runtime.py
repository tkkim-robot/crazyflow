"""Held-step runtime adapter for the analytic-only Version-A baseline.

The continuous analytic CBF/HOCBF projection is a local condition.  This adapter therefore
advances its proposed wrench through the exact airborne plant for one held controller interval and
evaluates the complete node/swept physical margin.  Moving-sphere predictions, when supplied, are
checked with their relative motion over the same interval.  An unsafe proposal is replaced only by
the actuator midpoint and is explicitly degraded; the midpoint is never reported as certified.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.dynamic_rollouts import (
    DynamicSphereScenarioBatch,
    dynamic_quad_safety_values,
)
from crazyflow.safety.da_plcbf.quad_actor_losses import quad_safety_values
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.version_a_analytic_filter import (
    VersionAAnalyticFilterResult,
    version_a_analytic_filter,
)

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.version_a_barriers import (
        RigidBodySafetySet,
        VersionABarrierConfig,
        VersionAModel,
    )
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator, VersionAFilterConfig


class VersionAAnalyticRuntimeStep(NamedTuple):
    """Applied analytic-only action and exact held-interval evidence."""

    next_state: Array
    action: Array
    continuous_filter: VersionAAnalyticFilterResult
    proposal_interval_margin: Array
    midpoint_interval_margin: Array
    applied_interval_margin: Array
    proposal_interval_accepted: Array
    used_interval_midpoint: Array
    degraded: Array


def _slice_interval(window: DynamicSphereScenarioBatch) -> DynamicSphereScenarioBatch:
    """Return the observed/predicted two-node interval without changing sample axes."""
    return DynamicSphereScenarioBatch(
        obstacle_centers=window.obstacle_centers[:, :, :2],
        obstacle_radii=window.obstacle_radii[:, :, :2],
        obstacle_mask=window.obstacle_mask[:, :, :2],
        arena_lower=window.arena_lower,
        arena_upper=window.arena_upper,
        speed_limit=window.speed_limit,
        angular_rate_max=window.angular_rate_max,
        tilt_max_radians=window.tilt_max_radians,
    )


def _held_interval_margin(
    state: Array,
    wrench: Array,
    model: VersionAModel,
    safety: RigidBodySafetySet,
    barrier_config: VersionABarrierConfig,
    dt: float,
    prediction_window: DynamicSphereScenarioBatch | None,
) -> tuple[Array, Array]:
    next_state = direct_wrench_symplectic_step(state, wrench, model, dt)
    if prediction_window is None:
        pair = jnp.stack((state, next_state))[None, None]
        values = quad_safety_values(pair, safety, barrier_config, softmin_beta=1.0)
        return next_state, values.hard_policy_margins[0, 0]

    interval = _slice_interval(prediction_window)
    prediction_count = interval.obstacle_centers.shape[1]
    pair = jnp.stack((state, next_state))
    states = jnp.broadcast_to(pair[None, None, None], (1, 1, prediction_count, 2, 13))
    values = dynamic_quad_safety_values(states, interval, barrier_config, softmin_beta=1.0)
    return next_state, values.robust_hard_margins[0, 0]


def version_a_analytic_runtime_step(
    state: Array,
    nominal_wrench: Array,
    weight: Array,
    model: VersionAModel,
    actuator: VersionAActuator,
    safety: RigidBodySafetySet,
    barrier_config: VersionABarrierConfig,
    filter_config: VersionAFilterConfig,
    *,
    dt: float,
    prediction_window: DynamicSphereScenarioBatch | None = None,
    interval_tolerance: float = 1e-6,
) -> VersionAAnalyticRuntimeStep:
    """Project, postcheck, and advance one analytic-only controller interval.

    ``prediction_window`` must contain the current observation and at least one future node.  The
    continuous analytic rows still use the current observed sphere positions; the independent
    held-step gate additionally checks exact relative swept motion for every supplied prediction.
    """
    if state.shape != (13,) or nominal_wrench.shape != (4,):
        raise ValueError("state and nominal_wrench must have shapes (13,) and (4,)")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(interval_tolerance) or interval_tolerance < 0:
        raise ValueError("interval_tolerance must be finite and nonnegative")
    if prediction_window is not None:
        if prediction_window.obstacle_centers.shape[0] != 1:
            raise ValueError("runtime prediction_window must have B=1")
        if prediction_window.obstacle_centers.shape[2] < 2:
            raise ValueError("prediction_window must contain at least two nodes")

    filtered = version_a_analytic_filter(
        state,
        nominal_wrench,
        weight,
        model,
        actuator,
        safety._replace(
            obstacle_centers=safety.obstacle_centers[0],
            obstacle_radii=safety.obstacle_radii[0],
            obstacle_mask=safety.obstacle_mask[0],
            arena_lower=safety.arena_lower[0],
            arena_upper=safety.arena_upper[0],
            speed_max=safety.speed_max[0],
            angular_rate_max=safety.angular_rate_max[0],
            tilt_max_radians=safety.tilt_max_radians[0],
        ),
        barrier_config,
        filter_config,
    )
    proposal_next, proposal_margin = _held_interval_margin(
        state, filtered.action, model, safety, barrier_config, dt, prediction_window
    )
    midpoint = filtered.motor_polytope.midpoint_wrench
    midpoint_next, midpoint_margin = _held_interval_margin(
        state, midpoint, model, safety, barrier_config, dt, prediction_window
    )
    proposal_accepted = (
        filtered.qp_accepted
        & filtered.action_executable
        & jnp.isfinite(proposal_margin)
        & (proposal_margin >= -interval_tolerance)
    )
    use_midpoint = ~proposal_accepted
    action = jnp.where(proposal_accepted, filtered.action, midpoint)
    next_state = jnp.where(proposal_accepted, proposal_next, midpoint_next)
    applied_margin = jnp.where(proposal_accepted, proposal_margin, midpoint_margin)
    degraded = (
        filtered.degraded
        | use_midpoint
        | ~jnp.isfinite(applied_margin)
        | (applied_margin < -interval_tolerance)
    )
    return VersionAAnalyticRuntimeStep(
        next_state=next_state,
        action=action,
        continuous_filter=filtered,
        proposal_interval_margin=proposal_margin,
        midpoint_interval_margin=midpoint_margin,
        applied_interval_margin=applied_margin,
        proposal_interval_accepted=proposal_accepted,
        used_interval_midpoint=use_midpoint,
        degraded=degraded,
    )


__all__ = ["VersionAAnalyticRuntimeStep", "version_a_analytic_runtime_step"]
