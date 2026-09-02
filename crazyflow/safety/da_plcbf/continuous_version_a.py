"""Minimal continuous Version-A PL-CBF controller for the online demo.

This module deliberately has no policy-learning protocol.  A caller supplies two JAX-pure
rollout functions: one for the nominal policy and one for an immutable snapshot of the fallback
library.  Neither function receives obstacle data.  The controller then augments the two sets,
scores every rollout against the *current* point prediction of static and moving obstacles,
differentiate the hard values with JAX, and invokes the existing audited direct-wrench QP.

Only one :class:`~crazyflow.safety.da_plcbf.version_a_barriers.VersionAModel` is accepted.  It is
the current point estimate and is shared by the rollouts, value gradient, and QP.  Model particles,
candidate policy updates, policy admission, and rollback are intentionally outside this path.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.quad_policy import waypoint_nominal_wrench
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.selector import SelectionConfig
from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel
from crazyflow.safety.da_plcbf.version_a_filter import (
    PolicyLibraryCertificates,
    VersionAFilterResult,
    WrenchPostcheck,
    postcheck_version_a_action,
    version_a_plcbf_filter,
)

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
    from crazyflow.safety.da_plcbf.version_a_barriers import (
        RigidBodySafetySet,
        VersionABarrierConfig,
    )
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator, VersionAFilterConfig


class PolicyRollouts(NamedTuple):
    """One or more policy rollouts from a common current state.

    ``states`` has shape ``(policy_count, horizon + 1, 13)``, ``wrenches`` has shape
    ``(policy_count, horizon, 4)``, and ``valid`` has shape ``(policy_count,)``.  The initial state
    must be present at node zero so the finite-horizon value includes current safety.
    """

    states: Array
    wrenches: Array
    valid: Array


class RuntimeObstacleTrajectories(NamedTuple):
    """One point prediction for spherical obstacles over the controller horizon.

    ``centers`` has shape ``(horizon + 1, obstacle_count, 3)``.  ``mask`` has shape
    ``(horizon + 1, obstacle_count)`` so an observed moving obstacle can enter or leave the
    prediction window without introducing a particle/sample axis.  Radii are constant over the
    short rollout and have shape ``(obstacle_count,)``.
    """

    centers: Array
    radii: Array
    mask: Array


class RuntimePolicyValues(NamedTuple):
    """Hard node/swept values and active-minimum diagnostics for every candidate."""

    values: Array
    constraint_values: Array
    active_indices: Array
    second_value_gaps: Array
    input_valid: Array


class ContinuousVersionAStep(NamedTuple):
    """One immutable-snapshot PL-CBF decision and its executable postchecks."""

    action: Array
    nominal_action: Array
    next_estimated_state: Array
    candidates: PolicyRollouts
    values: RuntimePolicyValues
    gradients: Array
    gradient_valid: Array
    certificates: PolicyLibraryCertificates
    continuous_filter: VersionAFilterResult
    applied_postcheck: WrenchPostcheck
    selected_index: Array
    selected_is_nominal: Array
    safe_candidate_count: Array
    eligible_candidate_count: Array
    qp_held_margin: Array
    fallback_held_margin: Array
    applied_held_margin: Array
    qp_valid: Array
    fallback_valid: Array
    used_fallback: Array
    used_midpoint: Array
    degraded: Array
    qp_intervention_norm: Array


@dataclass(frozen=True, slots=True)
class ContinuousVersionAConfig:
    """Small set of demo-specific certificate and held-step settings."""

    dt: float = 0.02
    horizon: int = 50
    obstacle_clearance: float = 0.12
    interval_tolerance: float = 2e-6
    prefer_nominal_when_safe: bool = True

    def validate(self) -> None:
        """Reject non-finite geometry or invalid fixed rollout shapes."""
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if not math.isfinite(self.obstacle_clearance) or self.obstacle_clearance < 0:
            raise ValueError("obstacle_clearance must be finite and nonnegative")
        if not math.isfinite(self.interval_tolerance) or self.interval_tolerance < 0:
            raise ValueError("interval_tolerance must be finite and nonnegative")
        if not isinstance(self.prefer_nominal_when_safe, bool):
            raise TypeError("prefer_nominal_when_safe must be boolean")


RolloutFunction = Callable[[Array, VersionAModel], PolicyRollouts]


def _validate_rollout_shapes(
    rollouts: PolicyRollouts, *, horizon: int, require_single_policy: bool
) -> None:
    states = jnp.asarray(rollouts.states)
    wrenches = jnp.asarray(rollouts.wrenches)
    valid = jnp.asarray(rollouts.valid)
    if states.ndim != 3 or states.shape[1:] != (horizon + 1, 13):
        raise ValueError("rollout states must have shape (policies, horizon + 1, 13)")
    policy_count = states.shape[0]
    if require_single_policy and policy_count != 1:
        raise ValueError("the nominal rollout function must return exactly one policy")
    if not require_single_policy and policy_count < 1:
        raise ValueError("the fallback rollout function must return at least one policy")
    if wrenches.shape != (policy_count, horizon, 4):
        raise ValueError("rollout wrenches must have shape (policies, horizon, 4)")
    if valid.shape != (policy_count,) or not jnp.issubdtype(valid.dtype, jnp.bool_):
        raise ValueError("rollout valid must be boolean shape (policies,)")


def augmented_policy_rollouts(
    state: Array,
    nominal_rollout: RolloutFunction,
    fallback_rollouts: RolloutFunction,
    point_model: VersionAModel,
    *,
    horizon: int,
) -> PolicyRollouts:
    """Evaluate both policy sets with one point model and nominal policy at index zero."""
    if state.shape != (13,):
        raise ValueError("state must have shape (13,)")
    nominal = nominal_rollout(state, point_model)
    fallbacks = fallback_rollouts(state, point_model)
    _validate_rollout_shapes(nominal, horizon=horizon, require_single_policy=True)
    _validate_rollout_shapes(fallbacks, horizon=horizon, require_single_policy=False)
    states = jnp.concatenate((nominal.states, fallbacks.states), axis=0)
    wrenches = jnp.concatenate((nominal.wrenches, fallbacks.wrenches), axis=0)
    valid = jnp.concatenate((nominal.valid, fallbacks.valid), axis=0)
    scale = jnp.maximum(jnp.max(jnp.abs(state)), 1.0)
    tolerance = 32.0 * jnp.finfo(state.dtype).eps * scale
    includes_current = jnp.max(jnp.abs(states[:, 0] - state), axis=-1) <= tolerance
    return PolicyRollouts(states, wrenches, valid & includes_current)


def _validate_obstacle_shapes(obstacles: RuntimeObstacleTrajectories, *, horizon: int) -> None:
    centers = jnp.asarray(obstacles.centers)
    radii = jnp.asarray(obstacles.radii)
    mask = jnp.asarray(obstacles.mask)
    if centers.ndim != 3 or centers.shape[0] != horizon + 1 or centers.shape[-1] != 3:
        raise ValueError("obstacle centers must have shape (horizon + 1, obstacles, 3)")
    obstacle_count = centers.shape[1]
    if obstacle_count < 1:
        raise ValueError("the demo runtime requires at least one obstacle slot")
    if radii.shape != (obstacle_count,):
        raise ValueError("obstacle radii must have shape (obstacles,)")
    if mask.shape != (horizon + 1, obstacle_count) or not jnp.issubdtype(mask.dtype, jnp.bool_):
        raise ValueError("obstacle mask must be boolean shape (horizon + 1, obstacles)")


def runtime_policy_values(
    rollout_states: Array, obstacles: RuntimeObstacleTrajectories, *, obstacle_clearance: float
) -> RuntimePolicyValues:
    """Return exact hard spherical values using relative swept motion.

    A moving sphere is handled by subtracting its predicted centre from the quadrotor position at
    both interval endpoints.  The closest point of that relative segment gives the exact swept
    distance under piecewise-linear interpolation.  This is runtime-only geometry: no obstacle
    quantity is passed to either candidate policy.
    """
    if rollout_states.ndim != 3 or rollout_states.shape[-1] != 13:
        raise ValueError("rollout_states must have shape (policies, horizon + 1, 13)")
    horizon = rollout_states.shape[1] - 1
    if horizon <= 0:
        raise ValueError("rollout_states must contain at least two nodes")
    _validate_obstacle_shapes(obstacles, horizon=horizon)
    if not math.isfinite(obstacle_clearance) or obstacle_clearance < 0:
        raise ValueError("obstacle_clearance must be finite and nonnegative")

    centers = jnp.asarray(obstacles.centers, dtype=rollout_states.dtype)
    radii = jnp.asarray(obstacles.radii, dtype=rollout_states.dtype)
    mask = jnp.asarray(obstacles.mask, dtype=bool)
    relative = rollout_states[:, :, None, :3] - centers[None, ...]
    inflated_radius_squared = (radii + obstacle_clearance) ** 2
    node_values = jnp.sum(relative * relative, axis=-1) - inflated_radius_squared[None, None, :]
    node_values = jnp.where(mask[None, ...], node_values, jnp.inf)

    start = relative[:, :-1]
    delta = relative[:, 1:] - start
    denominator = jnp.sum(delta * delta, axis=-1)
    moving = denominator > 32.0 * jnp.finfo(rollout_states.dtype).eps
    safe_denominator = jnp.where(moving, denominator, 1.0)
    fraction = jnp.clip(-jnp.sum(start * delta, axis=-1) / safe_denominator, 0.0, 1.0)
    fraction = jnp.where(moving, fraction, 0.0)
    closest = start + fraction[..., None] * delta
    segment_values = jnp.sum(closest * closest, axis=-1) - inflated_radius_squared[None, None, :]
    segment_mask = mask[:-1] & mask[1:]
    segment_values = jnp.where(segment_mask[None, ...], segment_values, jnp.inf)

    flattened = jnp.concatenate(
        (
            node_values.reshape(rollout_states.shape[0], -1),
            segment_values.reshape(rollout_states.shape[0], -1),
        ),
        axis=-1,
    )
    finite_constraints = jnp.where(jnp.isfinite(flattened), flattened, jnp.inf)
    ordered = jnp.sort(finite_constraints, axis=-1)
    finite_count = jnp.sum(jnp.isfinite(flattened), axis=-1)
    values = ordered[:, 0]
    second = jnp.where(finite_count > 1, ordered[:, 1], jnp.inf)
    gaps = second - values
    active = jnp.argmin(finite_constraints, axis=-1)
    active_obstacles_valid = (~mask) | (
        jnp.all(jnp.isfinite(centers), axis=-1)
        & jnp.isfinite(radii)[None, :]
        & (radii[None, :] > 0)
    )
    input_valid = (
        jnp.all(jnp.isfinite(rollout_states), axis=(1, 2))
        & jnp.all(active_obstacles_valid)
        & (finite_count > 0)
        & jnp.isfinite(values)
    )
    values = jnp.where(input_valid, values, -jnp.inf)
    return RuntimePolicyValues(values, flattened, active, gaps, input_valid)


def rollout_waypoint_library(
    state: Array,
    target_positions: Array,
    target_velocities: Array,
    model: VersionAModel,
    actuator: VersionAActuator,
    quad_config: QuadPolicyConfig,
    *,
    dt: float,
    horizon: int,
    position_gain: float = 2.0,
    velocity_gain: float = 1.4,
) -> PolicyRollouts:
    """Roll out a deterministic set of waypoint policies through one point model."""
    if state.shape != (13,):
        raise ValueError("state must have shape (13,)")
    if target_positions.ndim != 2 or target_positions.shape[-1] != 3:
        raise ValueError("target_positions must have shape (policies, 3)")
    if target_velocities.shape != target_positions.shape:
        raise ValueError("target_velocities must match target_positions")
    if target_positions.shape[0] < 1:
        raise ValueError("at least one waypoint policy is required")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not all(math.isfinite(value) and value > 0 for value in (position_gain, velocity_gain)):
        raise ValueError("waypoint gains must be finite and positive")
    current = jnp.broadcast_to(state, (target_positions.shape[0], 13))

    def advance(carry: Array, _: None) -> tuple[Array, tuple[Array, Array]]:
        command = jax.vmap(
            lambda candidate_state, target_position, target_velocity: waypoint_nominal_wrench(
                candidate_state,
                target_position,
                target_velocity,
                model,
                actuator,
                quad_config,
                position_gain=position_gain,
                velocity_gain=velocity_gain,
            )
        )(carry, target_positions, target_velocities)
        following = direct_wrench_symplectic_step(carry, command.wrench, model, dt)
        return following, (following, command.wrench)

    _, (future, wrenches) = jax.lax.scan(advance, current, xs=None, length=horizon)
    future = jnp.moveaxis(future, 0, 1)
    wrenches = jnp.moveaxis(wrenches, 0, 1)
    states = jnp.concatenate((current[:, None, :], future), axis=1)
    valid = jnp.all(jnp.isfinite(states), axis=(1, 2)) & jnp.all(
        jnp.isfinite(wrenches), axis=(1, 2)
    )
    return PolicyRollouts(states, wrenches, valid)


def obstacle_agnostic_waypoint_callbacks(
    goal_position: Array,
    goal_velocity: Array,
    skill_displacements: Array,
    actuator: VersionAActuator,
    quad_config: QuadPolicyConfig,
    *,
    dt: float,
    horizon: int,
) -> tuple[RolloutFunction, RolloutFunction]:
    """Build a nominal callback and local, goal-free fallback skill callbacks.

    The fixed displacement vectors are descriptors of the skills, not obstacle-relative targets.
    At every control boundary they are anchored to the current position.  The goal enters only the
    separate one-policy nominal callback.
    """
    if goal_position.shape != (3,) or goal_velocity.shape != (3,):
        raise ValueError("goal_position and goal_velocity must have shape (3,)")
    if skill_displacements.ndim != 2 or skill_displacements.shape[-1] != 3:
        raise ValueError("skill_displacements must have shape (skills, 3)")
    if skill_displacements.shape[0] < 1:
        raise ValueError("at least one fallback skill is required")
    zero_skill_velocities = jnp.zeros_like(skill_displacements)

    def nominal(candidate_state: Array, point_model: VersionAModel) -> PolicyRollouts:
        return rollout_waypoint_library(
            candidate_state,
            goal_position[None, :],
            goal_velocity[None, :],
            point_model,
            actuator,
            quad_config,
            dt=dt,
            horizon=horizon,
        )

    def fallbacks(candidate_state: Array, point_model: VersionAModel) -> PolicyRollouts:
        return rollout_waypoint_library(
            candidate_state,
            candidate_state[None, :3] + skill_displacements,
            zero_skill_velocities,
            point_model,
            actuator,
            quad_config,
            dt=dt,
            horizon=horizon,
        )

    return nominal, fallbacks


def _held_margin(
    state: Array,
    wrench: Array,
    model: VersionAModel,
    obstacles: RuntimeObstacleTrajectories,
    config: ContinuousVersionAConfig,
) -> tuple[Array, Array]:
    next_state = direct_wrench_symplectic_step(state, wrench, model, config.dt)
    pair = jnp.stack((state, next_state))[None, ...]
    interval_obstacles = RuntimeObstacleTrajectories(
        obstacles.centers[:2], obstacles.radii, obstacles.mask[:2]
    )
    margin = runtime_policy_values(
        pair, interval_obstacles, obstacle_clearance=config.obstacle_clearance
    ).values[0]
    return next_state, margin


def continuous_version_a_step(
    state: Array,
    nominal_rollout: RolloutFunction,
    fallback_rollouts: RolloutFunction,
    obstacles: RuntimeObstacleTrajectories,
    estimated_model: VersionAModel,
    actuator: VersionAActuator,
    safety_limits: RigidBodySafetySet,
    barrier_config: VersionABarrierConfig,
    filter_config: VersionAFilterConfig,
    config: ContinuousVersionAConfig,
    *,
    wrench_weight: Array | None = None,
    previous_policy_index: Array | None = None,
    selection_config: SelectionConfig = SelectionConfig(),
) -> ContinuousVersionAStep:
    """Run one point-model, augmented-library continuous PL-CBF controller step.

    The returned decision is based on exactly one immutable policy snapshot because the supplied
    callback closures are evaluated wholly inside this call.  A publisher may replace its active
    parameters only between calls.  The QP action is used only after its exact actuator/barrier
    postcheck and an independent held-interval obstacle check.  The selected policy's first action
    is considered only when that QP proposal is invalid.
    """
    config.validate()
    barrier_config.validate()
    filter_config.validate()
    selection_config.validate()
    if state.shape != (13,):
        raise ValueError("state must have shape (13,)")
    _validate_obstacle_shapes(obstacles, horizon=config.horizon)
    if safety_limits.obstacle_centers.ndim != 2:
        raise ValueError("safety_limits must be an unbatched RigidBodySafetySet")
    if wrench_weight is None:
        wrench_weight = jnp.asarray([1.0, 2.0e4, 2.0e4, 2.0e4], dtype=state.dtype)
    if wrench_weight.shape not in ((4,), (4, 4)):
        raise ValueError("wrench_weight must have shape (4,) or (4, 4)")

    def rollouts_and_values(candidate_state: Array) -> tuple[Array, PolicyRollouts]:
        candidates = augmented_policy_rollouts(
            candidate_state,
            nominal_rollout,
            fallback_rollouts,
            estimated_model,
            horizon=config.horizon,
        )
        values = runtime_policy_values(
            candidates.states, obstacles, obstacle_clearance=config.obstacle_clearance
        ).values
        return values, candidates

    values_array, candidates = rollouts_and_values(state)
    gradients = jax.jacfwd(lambda candidate_state: rollouts_and_values(candidate_state)[0])(state)
    values = runtime_policy_values(
        candidates.states, obstacles, obstacle_clearance=config.obstacle_clearance
    )
    gradient_valid = (
        candidates.valid & values.input_valid & jnp.all(jnp.isfinite(gradients), axis=-1)
    )
    certificates = PolicyLibraryCertificates(
        values=values_array,
        gradients=gradients,
        gradient_valid=gradient_valid,
        fallback_wrenches=candidates.wrenches[:, 0],
    )
    current_safety = safety_limits._replace(
        obstacle_centers=jnp.asarray(obstacles.centers[0], dtype=state.dtype),
        obstacle_radii=jnp.asarray(obstacles.radii, dtype=state.dtype),
        obstacle_mask=jnp.asarray(obstacles.mask[0], dtype=bool),
    )
    nominal_action = candidates.wrenches[0, 0]
    demo_filter_config = replace(filter_config, selection_requires_certified_fallback=False)
    demo_selection_config = replace(
        selection_config, prefer_first_eligible=config.prefer_nominal_when_safe
    )
    filtered = version_a_plcbf_filter(
        state,
        nominal_action,
        wrench_weight,
        certificates,
        estimated_model,
        actuator,
        current_safety,
        barrier_config,
        demo_filter_config,
        previous_policy_index=previous_policy_index,
        selection_config=demo_selection_config,
    )
    selected_fallback = candidates.wrenches[filtered.selected_index, 0]
    qp_next, qp_margin = _held_margin(state, filtered.qp.action, estimated_model, obstacles, config)
    fallback_next, fallback_margin = _held_margin(
        state, selected_fallback, estimated_model, obstacles, config
    )
    qp_valid = (
        filtered.qp_accepted & jnp.isfinite(qp_margin) & (qp_margin >= -config.interval_tolerance)
    )
    selected_value = values.values[filtered.selected_index]
    fallback_valid = (
        filtered.has_certificate
        & filtered.fallback_postcheck.actuator_passed
        & jnp.isfinite(selected_value)
        & (selected_value >= selection_config.minimum_hard_value)
        & jnp.isfinite(fallback_margin)
        & (fallback_margin >= -config.interval_tolerance)
    )
    use_fallback = (~qp_valid) & fallback_valid
    use_midpoint = (~qp_valid) & (~fallback_valid)
    midpoint = filtered.motor_polytope.midpoint_wrench
    midpoint_next, midpoint_margin = _held_margin(
        state, midpoint, estimated_model, obstacles, config
    )
    action = jnp.where(
        qp_valid, filtered.qp.action, jnp.where(use_fallback, selected_fallback, midpoint)
    )
    next_state = jnp.where(qp_valid, qp_next, jnp.where(use_fallback, fallback_next, midpoint_next))
    applied_margin = jnp.where(
        qp_valid, qp_margin, jnp.where(use_fallback, fallback_margin, midpoint_margin)
    )
    applied_postcheck = postcheck_version_a_action(action, actuator, filtered, demo_filter_config)
    execution_valid = jnp.where(
        qp_valid,
        applied_postcheck.passed,
        jnp.where(use_fallback, filtered.fallback_postcheck.actuator_passed, False),
    )
    degraded = (
        use_midpoint
        | ~execution_valid
        | ~jnp.isfinite(applied_margin)
        | (applied_margin < -config.interval_tolerance)
    )
    safe = values.input_valid & (values.values >= selection_config.minimum_hard_value)
    return ContinuousVersionAStep(
        action=action,
        nominal_action=nominal_action,
        next_estimated_state=next_state,
        candidates=candidates,
        values=values,
        gradients=gradients,
        gradient_valid=gradient_valid,
        certificates=certificates,
        continuous_filter=filtered,
        applied_postcheck=applied_postcheck,
        selected_index=filtered.selected_index,
        selected_is_nominal=filtered.selected_index == 0,
        safe_candidate_count=jnp.sum(safe, dtype=jnp.int32),
        eligible_candidate_count=jnp.sum(filtered.policy_eligible, dtype=jnp.int32),
        qp_held_margin=qp_margin,
        fallback_held_margin=fallback_margin,
        applied_held_margin=applied_margin,
        qp_valid=qp_valid,
        fallback_valid=fallback_valid,
        used_fallback=use_fallback,
        used_midpoint=use_midpoint,
        degraded=degraded,
        qp_intervention_norm=jnp.linalg.norm(action - nominal_action),
    )


__all__ = [
    "ContinuousVersionAConfig",
    "ContinuousVersionAStep",
    "PolicyRollouts",
    "RuntimeObstacleTrajectories",
    "RuntimePolicyValues",
    "augmented_policy_rollouts",
    "continuous_version_a_step",
    "obstacle_agnostic_waypoint_callbacks",
    "rollout_waypoint_library",
    "runtime_policy_values",
]
