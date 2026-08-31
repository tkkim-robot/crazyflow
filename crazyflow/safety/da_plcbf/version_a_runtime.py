"""One complete Version-A DA-PLCBF runtime decision with a held-step postcheck."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.capsules import (
    CapsuleBarrierConfig,
    CapsuleObstacleSet,
    quad_capsule_trajectory_values,
)
from crazyflow.safety.da_plcbf.certificates import (
    LibraryCertificateDiagnostics,
    version_a_shared_library_certificates,
    version_a_shared_library_values,
)
from crazyflow.safety.da_plcbf.library import slice_shared_actor_policy
from crazyflow.safety.da_plcbf.quad_actor_losses import quad_safety_values
from crazyflow.safety.da_plcbf.quad_policy import QuadWrenchCommand, waypoint_nominal_wrench
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.selector import SelectionConfig
from crazyflow.safety.da_plcbf.version_a_filter import (
    VersionAFilterResult,
    WrenchPostcheck,
    postcheck_version_a_action,
    version_a_plcbf_filter,
)

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.actor import (
        SharedActorConfig,
        SharedActorParams,
        SharedActorSpec,
    )
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
    from crazyflow.safety.da_plcbf.version_a_barriers import (
        RigidBodySafetySet,
        VersionABarrierConfig,
        VersionAModel,
    )
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator, VersionAFilterConfig


class VersionARuntimeStep(NamedTuple):
    """Applied state transition and complete certificate/filter/interval decision."""

    next_state: Array
    action: Array
    nominal: QuadWrenchCommand
    certificates: LibraryCertificateDiagnostics
    continuous_filter: VersionAFilterResult
    applied_continuous_postcheck: WrenchPostcheck
    discrete_contraction_factor: Array
    proposal_next_policy_value: Array
    fallback_next_policy_value: Array
    applied_next_policy_value: Array
    proposal_discrete_residual: Array
    fallback_discrete_residual: Array
    applied_discrete_residual: Array
    proposal_discrete_accepted: Array
    fallback_discrete_accepted: Array
    applied_discrete_applicable: Array
    proposal_interval_margin: Array
    fallback_interval_margin: Array
    applied_interval_margin: Array
    proposal_interval_accepted: Array
    fallback_interval_accepted: Array
    used_interval_fallback: Array
    used_interval_midpoint: Array
    degraded: Array


def _one_step_interval_margin(
    state: Array,
    wrench: Array,
    model: VersionAModel,
    safety: RigidBodySafetySet,
    barrier_config: VersionABarrierConfig,
    dt: float,
    capsules: CapsuleObstacleSet | None,
    capsule_config: CapsuleBarrierConfig,
) -> tuple[Array, Array]:
    next_state = direct_wrench_symplectic_step(state, wrench, model, dt)
    pair = jnp.stack((state, next_state))[None, None, ...]
    values = quad_safety_values(pair, safety, barrier_config, softmin_beta=1.0)
    margin = values.hard_policy_margins[0, 0]
    if capsules is not None:
        capsule_values = quad_capsule_trajectory_values(
            pair, capsules, clearance=capsule_config.clearance, softmin_beta=1.0
        )
        margin = jnp.minimum(margin, capsule_values.hard_policy_margins[0, 0])
    return next_state, margin


def _single_capsules(capsules: CapsuleObstacleSet) -> CapsuleObstacleSet:
    if capsules.segment_start.ndim != 3 or capsules.segment_start.shape[0] != 1:
        raise ValueError("runtime capsules must have shape (1, capsules, 3)")
    return CapsuleObstacleSet(
        capsules.segment_start[0], capsules.segment_end[0], capsules.radii[0], capsules.mask[0]
    )


def version_a_runtime_step(
    state: Array,
    target_position: Array,
    target_velocity: Array,
    params: SharedActorParams,
    spec: SharedActorSpec,
    scenarios: CircleScenarioBatch,
    safety: RigidBodySafetySet,
    model: VersionAModel,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    barrier_config: VersionABarrierConfig,
    filter_config: VersionAFilterConfig,
    *,
    dt: float,
    certificate_horizon: int,
    policy_gain: float,
    interval_tolerance: float = 1e-6,
    wrench_weight: Array | None = None,
    previous_policy_index: Array | None = None,
    selection_config: SelectionConfig = SelectionConfig(),
    capsules: CapsuleObstacleSet | None = None,
    capsule_config: CapsuleBarrierConfig = CapsuleBarrierConfig(),
) -> VersionARuntimeStep:
    r"""Evaluate, filter, exact-preview, and advance one airborne direct-wrench step.

    The continuous PL-CBF QP is local. Therefore both its proposal and the selected fallback are
    simulated through the configured held step. In addition to exact physical node/swept checks,
    the selected policy's hard value is recomputed at the successor with the same horizon. The
    sampled contraction condition is

    V_i(x_next; H) - exp(-policy_alpha * dt) * V_i(x; H) >= 0.

    A command is accepted only if its continuous, actuator, held-physical, and discrete
    same-policy/equal-horizon checks all pass. Otherwise the actuator midpoint is an explicitly
    degraded best effort; no finite-horizon certificate is claimed for it.
    """
    if state.shape != (13,) or target_position.shape != (3,) or target_velocity.shape != (3,):
        raise ValueError("state/target shapes must be (13,), (3,), and (3,)")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(interval_tolerance) or interval_tolerance < 0:
        raise ValueError("interval_tolerance must be finite and nonnegative")
    capsule_config.validate()
    selection_config.validate()
    if wrench_weight is None:
        wrench_weight = jnp.ones((4,), dtype=state.dtype)
    if wrench_weight.shape not in ((4,), (4, 4)):
        raise ValueError("wrench_weight must have shape (4,) or (4,4)")

    nominal = waypoint_nominal_wrench(
        state, target_position, target_velocity, model, actuator, quad_config
    )
    certificates = version_a_shared_library_certificates(
        state,
        params,
        spec,
        scenarios,
        safety,
        model,
        actuator,
        actor_config,
        quad_config,
        barrier_config,
        dt=dt,
        horizon=certificate_horizon,
        policy_gain=policy_gain,
        capsules=capsules,
        capsule_config=capsule_config,
    )
    unbatched_capsules = None if capsules is None else _single_capsules(capsules)
    filtered = version_a_plcbf_filter(
        state,
        nominal.wrench,
        wrench_weight,
        certificates.certificates,
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
        previous_policy_index=previous_policy_index,
        selection_config=selection_config,
        capsules=unbatched_capsules,
        capsule_config=capsule_config,
    )
    proposal_next, proposal_margin = _one_step_interval_margin(
        state, filtered.action, model, safety, barrier_config, dt, capsules, capsule_config
    )
    selected_fallback = certificates.certificates.fallback_wrenches[filtered.selected_index]
    fallback_next, fallback_margin = _one_step_interval_margin(
        state, selected_fallback, model, safety, barrier_config, dt, capsules, capsule_config
    )
    selected_params, selected_spec = slice_shared_actor_policy(
        params, spec, filtered.selected_index
    )

    def next_policy_value(next_state: Array) -> tuple[Array, Array]:
        evidence = version_a_shared_library_values(
            next_state,
            selected_params,
            selected_spec,
            scenarios,
            safety,
            model,
            actuator,
            actor_config,
            quad_config,
            barrier_config,
            dt=dt,
            horizon=certificate_horizon,
            policy_gain=policy_gain,
            capsules=capsules,
            capsule_config=capsule_config,
        )
        value = evidence.values[0]
        valid = evidence.rollout_valid[0] & evidence.includes_current_state[0] & jnp.isfinite(value)
        return value, valid

    proposal_next_value, proposal_next_valid = next_policy_value(proposal_next)
    fallback_next_value, fallback_next_valid = next_policy_value(fallback_next)
    current_value = certificates.certificates.values[filtered.selected_index]
    contraction_factor = jnp.asarray(math.exp(-filter_config.policy_alpha * dt), dtype=state.dtype)
    proposal_discrete_residual = proposal_next_value - contraction_factor * current_value
    fallback_discrete_residual = fallback_next_value - contraction_factor * current_value
    current_value_valid = (
        filtered.has_certificate
        & jnp.isfinite(current_value)
        & (current_value >= -interval_tolerance)
    )
    proposal_discrete_accepted = (
        current_value_valid
        & proposal_next_valid
        & (proposal_next_value >= -interval_tolerance)
        & jnp.isfinite(proposal_discrete_residual)
        & (proposal_discrete_residual >= -interval_tolerance)
    )
    fallback_discrete_accepted = (
        current_value_valid
        & fallback_next_valid
        & (fallback_next_value >= -interval_tolerance)
        & jnp.isfinite(fallback_discrete_residual)
        & (fallback_discrete_residual >= -interval_tolerance)
    )
    proposal_interval_accepted = (
        filtered.action_executable
        & (filtered.qp_accepted | filtered.fallback_certified)
        & jnp.isfinite(proposal_margin)
        & (proposal_margin >= -interval_tolerance)
    )
    fallback_interval_accepted = (
        filtered.fallback_certified
        & jnp.isfinite(fallback_margin)
        & (fallback_margin >= -interval_tolerance)
    )
    proposal_accepted = proposal_interval_accepted & proposal_discrete_accepted
    fallback_accepted = fallback_interval_accepted & fallback_discrete_accepted
    use_fallback = (~proposal_accepted) & fallback_accepted
    use_midpoint = (~proposal_accepted) & (~fallback_accepted)
    midpoint = filtered.motor_polytope.midpoint_wrench
    midpoint_next, midpoint_margin = _one_step_interval_margin(
        state, midpoint, model, safety, barrier_config, dt, capsules, capsule_config
    )
    action = jnp.where(
        proposal_accepted,
        filtered.action,
        jnp.where(fallback_accepted, selected_fallback, midpoint),
    )
    next_state = jnp.where(
        proposal_accepted, proposal_next, jnp.where(fallback_accepted, fallback_next, midpoint_next)
    )
    applied_margin = jnp.where(
        proposal_accepted,
        proposal_margin,
        jnp.where(fallback_accepted, fallback_margin, midpoint_margin),
    )
    applied_next_value = jnp.where(
        proposal_accepted,
        proposal_next_value,
        jnp.where(fallback_accepted, fallback_next_value, -jnp.inf),
    )
    applied_discrete_residual = jnp.where(
        proposal_accepted,
        proposal_discrete_residual,
        jnp.where(fallback_accepted, fallback_discrete_residual, -jnp.inf),
    )
    applied_discrete_applicable = proposal_accepted | fallback_accepted
    applied_continuous_postcheck = postcheck_version_a_action(
        action, actuator, filtered, filter_config
    )
    return VersionARuntimeStep(
        next_state,
        action,
        nominal,
        certificates,
        filtered,
        applied_continuous_postcheck,
        contraction_factor,
        proposal_next_value,
        fallback_next_value,
        applied_next_value,
        proposal_discrete_residual,
        fallback_discrete_residual,
        applied_discrete_residual,
        proposal_discrete_accepted,
        fallback_discrete_accepted,
        applied_discrete_applicable,
        proposal_margin,
        fallback_margin,
        applied_margin,
        proposal_interval_accepted,
        fallback_interval_accepted,
        use_fallback,
        use_midpoint,
        filtered.degraded
        | use_midpoint
        | (~applied_continuous_postcheck.passed)
        | (applied_margin < -interval_tolerance),
    )


__all__ = ["VersionARuntimeStep", "version_a_runtime_step"]
