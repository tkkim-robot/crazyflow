"""Minimal continuous Version-A PL-CBF controller for the online demo.

This module deliberately has no policy-learning protocol.  A caller supplies two JAX-pure
rollout functions: one for the nominal policy and one for an immutable snapshot of the fallback
library.  Neither function receives obstacle data.  The controller then augments the two sets,
scores every rollout against the *current* point prediction of static and moving obstacles,
differentiates a conservative smooth lower bound with JAX, and invokes the audited direct-wrench
QP. Exact hard swept values remain independent certification and postcheck diagnostics.

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


QP_REJECTION_REASONS = (
    "invalid_input",
    "no_policy_certificate",
    "qp_infeasible",
    "kkt_failed",
    "actuator_failed",
    "policy_residual_failed",
    "analytic_barrier_failed",
    "held_interval_failed",
)


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
    smooth_values: Array
    selected_policy_dual: Array
    qp_rejection_flags: Array


@dataclass(frozen=True, slots=True)
class ContinuousVersionAConfig:
    """Collision-only finite-horizon values with independent operational constraints.

    The corrected defaults use a conservative smooth collision-value QP face and disable analytic
    obstacle HOCBF faces. Arena, altitude, speed, angular-rate, tilt, and actuators remain enforced
    by the instantaneous filter. They are not part of the horizon collision value. Set
    ``use_policy_constraint=False, analytic_obstacle_hocbf=True`` for the matched analytic-only
    baseline; that mode has no PL-CBF row and never executes a library fallback.
    """

    dt: float = 0.02
    horizon: int = 50
    obstacle_clearance: float = 0.12
    interval_tolerance: float = 2e-6
    prefer_nominal_when_safe: bool = True
    ego_radius: float = 0.0
    analytic_obstacle_hocbf: bool = False
    use_policy_constraint: bool = True
    smooth_min_temperature: float = 0.005

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
        if not math.isfinite(self.ego_radius) or self.ego_radius < 0:
            raise ValueError("ego_radius must be finite and nonnegative")
        if not math.isfinite(self.smooth_min_temperature) or self.smooth_min_temperature <= 0:
            raise ValueError("smooth_min_temperature must be finite and positive")
        if not isinstance(self.analytic_obstacle_hocbf, bool):
            raise TypeError("analytic_obstacle_hocbf must be boolean")
        if not isinstance(self.use_policy_constraint, bool):
            raise TypeError("use_policy_constraint must be boolean")
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
    rollout_states: Array,
    obstacles: RuntimeObstacleTrajectories,
    *,
    obstacle_clearance: float,
    ego_radius: float = 0.0,
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

    if not math.isfinite(ego_radius) or ego_radius < 0:
        raise ValueError("ego_radius must be finite and nonnegative")
    centers = jnp.asarray(obstacles.centers, dtype=rollout_states.dtype)
    radii = jnp.asarray(obstacles.radii, dtype=rollout_states.dtype)
    mask = jnp.asarray(obstacles.mask, dtype=bool)
    # Sanitise masked padding before differentiation: where(mask, NaN, inf) still risks NaN
    # cotangents even when the padded obstacle has no physical role.
    safe_centers = jnp.where(mask[..., None], centers, 0.0)
    safe_radii = jnp.where(jnp.any(mask, axis=0), radii, 0.0)
    relative = rollout_states[:, :, None, :3] - safe_centers[None, ...]
    inflated_radius_squared = (safe_radii + ego_radius + obstacle_clearance) ** 2
    node_values = jnp.sum(relative * relative, axis=-1) - inflated_radius_squared[None, None, :]
    node_values = jnp.where(mask[None, ...], node_values, jnp.inf)

    start = relative[:, :-1]
    delta = relative[:, 1:] - start
    denominator = jnp.sum(delta * delta, axis=-1)
    # Only a zero-length segment is stationary. An epsilon-sized distance threshold can
    # otherwise miss a collision between two very close, individually safe endpoints.
    moving = denominator > 0.0
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
        & jnp.all(jnp.where(mask[None, ...], jnp.isfinite(node_values), True), axis=(1, 2))
        & jnp.all(
            jnp.where(segment_mask[None, ...], jnp.isfinite(segment_values), True), axis=(1, 2)
        )
        & (finite_count > 0)
        & jnp.isfinite(values)
    )
    values = jnp.where(input_valid, values, -jnp.inf)
    return RuntimePolicyValues(values, flattened, active, gaps, input_valid)


def conservative_smooth_policy_values(values: RuntimePolicyValues, *, temperature: float) -> Array:
    """Smooth lower bound on the exact hard obstacle minimum, in squared metres.

    The unnormalised log-sum-exp obeys ``soft <= hard <= soft + temperature * log(N)``
    for N enabled node/segment constraints. Duplicate endpoint minima therefore have a single
    smooth, permutation-invariant derivative rather than causing every rollout to be rejected.
    Both this value and its derivative must be used together in the QP. Hard values are retained
    for collision diagnostics and held-interval postchecks. This regularises the outer minimum;
    it does not eliminate policy-selection switches or all piecewise rollout nonsmoothness.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    constraints = values.constraint_values
    finite = jnp.isfinite(constraints)
    # A dummy finite row keeps an invalid/all-masked candidate's derivative well-defined; the
    # input-valid flag still rejects it. +inf padding contributes exactly zero exponential mass.
    safe_constraints = jnp.where(finite, constraints, jnp.inf)
    safe_constraints = safe_constraints.at[:, 0].set(
        jnp.where(jnp.any(finite, axis=-1), safe_constraints[:, 0], 0.0)
    )
    smooth = -temperature * jax.scipy.special.logsumexp(-safe_constraints / temperature, axis=-1)
    return jnp.where(values.input_valid, smooth, -jnp.inf)


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
    model_compensation: bool = False,
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
                model_compensation=model_compensation,
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
        pair,
        interval_obstacles,
        obstacle_clearance=config.obstacle_clearance,
        ego_radius=config.ego_radius,
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
    is considered only when that QP proposal is invalid. An executable held-safe fallback whose
    horizon value is negative may be returned only as explicitly degraded best effort.
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
        hard_values = runtime_policy_values(
            candidates.states,
            obstacles,
            obstacle_clearance=config.obstacle_clearance,
            ego_radius=config.ego_radius,
        )
        smooth_values = conservative_smooth_policy_values(
            hard_values, temperature=config.smooth_min_temperature
        )
        return smooth_values, candidates

    # Reuse the primal rollout already evaluated by forward differentiation. A separate
    # rollout here duplicates a full horizon scan without changing the candidate snapshot.
    gradients, candidates = jax.jacfwd(rollouts_and_values, has_aux=True)(state)
    values = runtime_policy_values(
        candidates.states,
        obstacles,
        obstacle_clearance=config.obstacle_clearance,
        ego_radius=config.ego_radius,
    )
    smooth_values = conservative_smooth_policy_values(
        values, temperature=config.smooth_min_temperature
    )
    gradient_valid = (
        candidates.valid
        & values.input_valid
        & jnp.isfinite(smooth_values)
        & jnp.all(jnp.isfinite(gradients), axis=-1)
    )
    certificates = PolicyLibraryCertificates(
        values=values.values,
        gradients=gradients,
        gradient_valid=gradient_valid,
        fallback_wrenches=candidates.wrenches[:, 0],
        barrier_values=smooth_values,
    )
    current_safety = safety_limits._replace(
        obstacle_centers=jnp.asarray(obstacles.centers[0], dtype=state.dtype),
        obstacle_radii=jnp.asarray(obstacles.radii, dtype=state.dtype),
        obstacle_mask=jnp.asarray(obstacles.mask[0], dtype=bool),
    )
    nominal_action = candidates.wrenches[0, 0]
    demo_filter_config = replace(
        filter_config,
        selection_requires_certified_fallback=False,
        enforce_policy_barrier=config.use_policy_constraint,
    )
    demo_barrier_config = replace(
        barrier_config,
        include_obstacle_hocbf=config.analytic_obstacle_hocbf,
        obstacle_clearance=config.obstacle_clearance,
        ego_radius=config.ego_radius,
    )
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
        demo_barrier_config,
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
    fallback_executable = (
        config.use_policy_constraint
        & candidates.valid[filtered.selected_index]
        & values.input_valid[filtered.selected_index]
        & filtered.fallback_postcheck.actuator_passed
        & filtered.fallback_postcheck.analytic_barriers_passed
        & jnp.isfinite(fallback_margin)
        & (fallback_margin >= -config.interval_tolerance)
    )
    fallback_valid = (
        fallback_executable
        & jnp.isfinite(selected_value)
        & (selected_value >= selection_config.minimum_hard_value)
    )
    # Without a horizon certificate, execute the best-valued held-safe fallback as explicitly
    # degraded best effort. An arbitrary midpoint must not manufacture an ablation failure.
    use_fallback = (~qp_valid) & fallback_executable
    use_midpoint = (~qp_valid) & (~fallback_executable)
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
        qp_valid, applied_postcheck.passed, jnp.where(use_fallback, fallback_valid, False)
    )
    degraded = (
        use_midpoint
        | ~execution_valid
        | ~jnp.isfinite(applied_margin)
        | (applied_margin < -config.interval_tolerance)
    )
    safe = values.input_valid & (values.values >= selection_config.minimum_hard_value)
    qp_rejection_flags = jnp.stack(
        (
            ~filtered.input_valid,
            config.use_policy_constraint & ~filtered.has_certificate,
            ~filtered.qp.feasible,
            filtered.qp.feasible & ~filtered.qp_kkt_valid,
            filtered.qp.feasible & ~filtered.qp_postcheck.actuator_passed,
            filtered.qp.feasible
            & config.use_policy_constraint
            & ~filtered.qp_postcheck.policy_barrier_passed,
            filtered.qp.feasible & ~filtered.qp_postcheck.analytic_barriers_passed,
            filtered.qp_accepted
            & (~jnp.isfinite(qp_margin) | (qp_margin < -config.interval_tolerance)),
        )
    )
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
        smooth_values=smooth_values,
        selected_policy_dual=filtered.selected_policy_dual,
        qp_rejection_flags=qp_rejection_flags,
    )


__all__ = [
    "ContinuousVersionAConfig",
    "ContinuousVersionAStep",
    "PolicyRollouts",
    "QP_REJECTION_REASONS",
    "RuntimeObstacleTrajectories",
    "RuntimePolicyValues",
    "augmented_policy_rollouts",
    "continuous_version_a_step",
    "conservative_smooth_policy_values",
    "obstacle_agnostic_waypoint_callbacks",
    "rollout_waypoint_library",
    "runtime_policy_values",
]
