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

from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig, waypoint_nominal_wrench
from crazyflow.safety.da_plcbf.quad_rollouts import (
    direct_wrench_symplectic_step,
    zero_order_hold_rollout,
)
from crazyflow.safety.da_plcbf.selector import SelectionConfig
from crazyflow.safety.da_plcbf.version_a_barriers import (
    VersionAModel,
    continuous_safety_halfspaces,
    dimensionless_safety_values,
    validated_control_affine_terms,
)
from crazyflow.safety.da_plcbf.version_a_filter import (
    PolicyLibraryCertificates,
    VersionAFilterResult,
    WrenchPostcheck,
    postcheck_version_a_action,
    reproject_with_predictive_operational_faces,
    version_a_plcbf_filter,
)

if TYPE_CHECKING:
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
    "held_operational_failed",
)

EXECUTION_MODES = ("qp", "fallback", "emergency", "midpoint")


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
    short rollout and have shape ``(obstacle_count,)``. Optional velocities have shape
    ``(obstacle_count, 3)`` or match centers; they define the local absolute-time shift.
    Without them, adjacent prediction-node slopes define that shift.
    """

    centers: Array
    radii: Array
    mask: Array
    velocities: Array | None = None


class RuntimePolicyValues(NamedTuple):
    """Hard node/swept values and active-minimum diagnostics for every candidate.

    A valid empty/all-masked collision horizon has value +inf (vacuous), active index -1,
    and input_valid=True. Invalid inputs have value -inf. Solver placeholders are separate.
    """

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
    time_derivatives: Array
    selected_smooth_value: Array
    selected_time_derivative: Array
    collision_constraint_active: Array
    effective_smooth_temperature: Array
    smooth_gap_bound: Array
    used_emergency: Array
    execution_mode: Array
    executed_policy_dual: Array
    applied_held_operational_margin: Array
    applied_held_operational_residual: Array
    applied_held_operational_passed: Array
    emergency_postcheck: WrenchPostcheck
    qp_held_operational_residuals: Array
    fallback_held_operational_residuals: Array
    applied_held_operational_residuals: Array
    applied_held_physical_margins: Array
    predictive_operational_iterations: Array
    initial_qp_held_operational_residual: Array


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
    smooth_min_gap_budget: float | None = 0.03
    control_interval_steps: int = 1
    emergency_braking_gain: float = 2.0
    emergency_acceleration_limit: float = 4.0
    predictive_operational_iterations: int = 3

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
        if self.smooth_min_gap_budget is not None and (
            not math.isfinite(self.smooth_min_gap_budget) or self.smooth_min_gap_budget <= 0
        ):
            raise ValueError("smooth_min_gap_budget must be positive finite or None")
        if (
            isinstance(self.control_interval_steps, bool)
            or not isinstance(self.control_interval_steps, int)
            or not 1 <= self.control_interval_steps <= self.horizon
        ):
            raise ValueError("control_interval_steps must be an integer in [1, horizon]")
        if not all(
            math.isfinite(x) and x > 0
            for x in (self.emergency_braking_gain, self.emergency_acceleration_limit)
        ):
            raise ValueError(
                "emergency braking gain and acceleration limit must be positive finite"
            )
        if not isinstance(self.analytic_obstacle_hocbf, bool):
            raise TypeError("analytic_obstacle_hocbf must be boolean")
        if not isinstance(self.use_policy_constraint, bool):
            raise TypeError("use_policy_constraint must be boolean")
        if not isinstance(self.prefer_nominal_when_safe, bool):
            raise TypeError("prefer_nominal_when_safe must be boolean")
        if (
            isinstance(self.predictive_operational_iterations, bool)
            or not isinstance(self.predictive_operational_iterations, int)
            or not 0 <= self.predictive_operational_iterations <= 4
        ):
            raise ValueError("predictive_operational_iterations must be an integer in [0, 4]")
        if self.predictive_operational_iterations > 0 and self.control_interval_steps > 2:
            raise ValueError(
                "predictive operational refinement supports at most two integration substeps; "
                "set predictive_operational_iterations=0 for longer holds with unchanged postchecks"
            )


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
    if radii.shape != (obstacle_count,):
        raise ValueError("obstacle radii must have shape (obstacles,)")
    if mask.shape != (horizon + 1, obstacle_count) or not jnp.issubdtype(mask.dtype, jnp.bool_):
        raise ValueError("obstacle mask must be boolean shape (horizon + 1, obstacles)")
    if obstacles.velocities is not None and obstacles.velocities.shape not in (
        centers.shape,
        (obstacle_count, 3),
    ):
        raise ValueError("obstacle velocities must match centers or have shape (obstacles, 3)")


def obstacle_prediction_velocities(obstacles: RuntimeObstacleTrajectories, *, dt: float) -> Array:
    """Local absolute-time shift derivative, with prediction masks held fixed.

    Explicit velocities are preferred. Otherwise each node uses the following segment slope and
    the terminal node repeats the last slope. Shifting every node by that derivative defines a
    local affine prediction update; it is exact for constant-velocity predictions. Discrete mask
    changes and arbitrary perception replacements are not covered by this temporal derivative.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive finite")
    _validate_obstacle_shapes(obstacles, horizon=obstacles.centers.shape[0] - 1)
    if obstacles.velocities is not None:
        return jnp.broadcast_to(obstacles.velocities, obstacles.centers.shape)
    slope = jnp.diff(obstacles.centers, axis=0) / dt
    valid_segments = obstacles.mask[:-1] & obstacles.mask[1:]
    slope = jnp.where(valid_segments[..., None], slope, 0.0)
    return jnp.concatenate((slope, slope[-1:]), axis=0)


def shift_obstacle_prediction(
    obstacles: RuntimeObstacleTrajectories, absolute_time_delta: Array, *, dt: float
) -> RuntimeObstacleTrajectories:
    """Advance the prediction's absolute time under its declared local velocity field."""
    velocities = obstacle_prediction_velocities(obstacles, dt=dt)
    return obstacles._replace(
        centers=obstacles.centers + absolute_time_delta * velocities, velocities=velocities
    )


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
    if centers.shape[1] == 0:
        valid = jnp.all(jnp.isfinite(rollout_states), axis=(1, 2))
        shape = (rollout_states.shape[0],)
        return RuntimePolicyValues(
            jnp.where(valid, jnp.inf, -jnp.inf),
            jnp.empty((*shape, 0), rollout_states.dtype),
            jnp.full(shape, -1, dtype=jnp.int32),
            jnp.full(shape, jnp.inf),
            valid,
        )
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
    gaps = jnp.where(finite_count > 1, second - values, jnp.inf)
    active = jnp.where(finite_count > 0, jnp.argmin(finite_constraints, axis=-1), -1)
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
        & (((finite_count > 0) & jnp.isfinite(values)) | ~jnp.any(mask))
    )
    values = jnp.where(input_valid, values, -jnp.inf)
    return RuntimePolicyValues(values, flattened, active, gaps, input_valid)


def smooth_min_conservatism(
    values: RuntimePolicyValues, *, temperature: float, max_gap_budget: float | None = None
) -> tuple[Array, Array]:
    """Return the effective temperature and worst-case hard/soft gap in square metres.

    No normalisation is applied to log-sum-exp. With a gap budget, temperature is capped by
    budget/log(N), keeping conservatism bounded when resolution or duplicate counts change.
    This does not claim duplicate invariance: report N/temperature/bound with every experiment.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive finite")
    if max_gap_budget is not None and (not math.isfinite(max_gap_budget) or max_gap_budget <= 0):
        raise ValueError("max_gap_budget must be positive finite or None")
    count = jnp.sum(jnp.isfinite(values.constraint_values), axis=-1)
    log_count = jnp.log(jnp.maximum(count, 1).astype(values.values.dtype))
    effective = jnp.full_like(values.values, temperature)
    if max_gap_budget is not None:
        effective = jnp.minimum(
            effective, max_gap_budget / jnp.where(log_count > 0, log_count, 1.0)
        )
    return effective, effective * log_count


def conservative_smooth_policy_values(
    values: RuntimePolicyValues, *, temperature: float, max_gap_budget: float | None = None
) -> Array:
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
    if constraints.shape[-1] == 0:
        return jnp.where(values.input_valid, jnp.inf, -jnp.inf)
    finite = jnp.isfinite(constraints)
    temperature, _ = smooth_min_conservatism(
        values, temperature=temperature, max_gap_budget=max_gap_budget
    )
    # A dummy finite row keeps invalid/all-masked derivatives well-defined. Invalid inputs are
    # rejected; valid empty horizons retain +inf. Padding contributes zero exponential mass.
    safe_constraints = jnp.where(finite, constraints, jnp.inf)
    safe_constraints = safe_constraints.at[:, 0].set(
        jnp.where(jnp.any(finite, axis=-1), safe_constraints[:, 0], 0.0)
    )
    smooth = -temperature * jax.scipy.special.logsumexp(
        -safe_constraints / temperature[:, None], axis=-1
    )
    smooth = jnp.where(jnp.any(finite, axis=-1), smooth, jnp.inf)
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
    command_hold_steps: int = 1,
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

    def command_at_boundary(carry: Array, _: Array) -> tuple[Array]:
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
        return (command.wrench,)

    future, (wrenches,) = zero_order_hold_rollout(
        current,
        command_at_boundary,
        model,
        dt=dt,
        horizon=horizon,
        command_hold_steps=command_hold_steps,
    )
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
    command_hold_steps: int = 1,
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
            command_hold_steps=command_hold_steps,
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
            command_hold_steps=command_hold_steps,
        )

    return nominal, fallbacks


class HeldActionCheck(NamedTuple):
    """Checks for one constant wrench over every integration substep in a control hold."""

    next_state: Array
    collision_margin: Array
    operational_margin: Array
    operational_residual: Array
    operational_passed: Array
    operational_residuals: Array
    physical_margins: Array


def _held_action_check(
    state: Array,
    wrench: Array,
    model: VersionAModel,
    obstacles: RuntimeObstacleTrajectories,
    safety_limits: RigidBodySafetySet,
    barrier_config: VersionABarrierConfig,
    filter_config: VersionAFilterConfig,
    config: ContinuousVersionAConfig,
) -> HeldActionCheck:
    def advance(current: Array, _: None) -> tuple[Array, Array]:
        following = direct_wrench_symplectic_step(current, wrench, model, config.dt)
        return following, following

    next_state, future = jax.lax.scan(advance, state, None, length=config.control_interval_steps)
    nodes = jnp.concatenate((state[None, :], future), axis=0)
    interval_obstacles = RuntimeObstacleTrajectories(
        obstacles.centers[: config.control_interval_steps + 1],
        obstacles.radii,
        obstacles.mask[: config.control_interval_steps + 1],
    )
    margin = runtime_policy_values(
        nodes[None, ...],
        interval_obstacles,
        obstacle_clearance=config.obstacle_clearance,
        ego_radius=config.ego_radius,
    ).values[0]
    # Collision is checked independently above. These diagnostics cover all operational limits
    # at every held node and their continuous analytic residual at every substep start.
    operational_safety = safety_limits._replace(
        obstacle_mask=jnp.zeros_like(safety_limits.obstacle_mask)
    )
    operational_config = replace(barrier_config, include_obstacle_hocbf=False)
    physical = jax.vmap(
        lambda x: dimensionless_safety_values(x, operational_safety, operational_config)
    )(nodes)
    obstacle_count = operational_safety.obstacle_radii.shape[0]
    operational_margin = jnp.min(physical.values[:, obstacle_count:])
    halfspaces = jax.vmap(
        lambda x: continuous_safety_halfspaces(x, model, operational_safety, operational_config)
    )(nodes[:-1])
    residuals = halfspaces.upper_bound - jnp.einsum("nmd,d->nm", halfspaces.matrix, wrench)
    operational_residual = jnp.min(jnp.where(halfspaces.enabled, residuals, jnp.inf))
    passed = (
        jnp.all(physical.input_valid)
        & jnp.all(halfspaces.domain_valid)
        & (operational_margin >= -barrier_config.domain_tolerance)
        & (operational_residual >= -filter_config.barrier_tolerance)
    )
    return HeldActionCheck(
        next_state,
        margin,
        operational_margin,
        operational_residual,
        passed,
        residuals[:, obstacle_count:],
        physical.values[:, obstacle_count:],
    )


def predictive_operational_halfspaces(
    state: Array,
    reference_wrench: Array,
    model: VersionAModel,
    safety: RigidBodySafetySet,
    barrier_config: VersionABarrierConfig,
    *,
    dt: float,
    hold_steps: int,
) -> tuple[Array, Array]:
    """Linearize later substep CBF residuals through the actual held-wrench integrator.

    For each residual r_j(u), the affine face is -Dr_j(u0) u <= r_j(u0)-Dr_j(u0)u0.
    This includes the command's effect on the predicted state, including attitude and velocity.
    It is a local predictive correction, not a global sampled-data invariance bound. Independent
    nonlinear held-action postchecks retain the original limits and tolerances.
    """
    operational_config = replace(barrier_config, include_obstacle_hocbf=False)
    operational_safety = safety._replace(obstacle_mask=jnp.zeros_like(safety.obstacle_mask))
    obstacle_count = safety.obstacle_radii.shape[0]

    def residual(wrench: Array) -> Array:
        def advance(current: Array, _: None) -> tuple[Array, Array]:
            following = direct_wrench_symplectic_step(current, wrench, model, dt)
            halfspaces = continuous_safety_halfspaces(
                following, model, operational_safety, operational_config
            )
            values = halfspaces.upper_bound - halfspaces.matrix @ wrench
            return following, values[obstacle_count:]

        _, values = jax.lax.scan(advance, state, None, length=hold_steps - 1)
        return values.reshape(-1)

    value = residual(reference_wrench)
    derivative = jax.jacfwd(residual)(reference_wrench)
    return -derivative, value - derivative @ reference_wrench


def obstacle_agnostic_emergency_wrench(
    state: Array, model: VersionAModel, actuator: VersionAActuator, config: ContinuousVersionAConfig
) -> tuple[Array, Array]:
    """Predeclared wind-aware velocity brake with attitude/rate stabilization.

    It receives no obstacle, goal, policy library, or safety value. Known drag/wind feedforward
    compensates external force; hard motor limits bound the stabilizing command. This is an
    emergency best effort, not a collision-avoidance certificate or proof of recoverability.
    """
    command = waypoint_nominal_wrench(
        state,
        state[:3],
        jnp.zeros(3, state.dtype),
        model,
        actuator,
        QuadPolicyConfig(acceleration_limit=config.emergency_acceleration_limit),
        position_gain=1.0,
        velocity_gain=config.emergency_braking_gain,
        model_compensation=True,
    )
    valid_model = validated_control_affine_terms(state, model).input_valid
    return command.wrench, command.input_valid & valid_model


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
    horizon value is negative may be returned only as explicitly degraded best effort. If no
    selected fallback is executable, every method uses the same wind-aware stabilizing emergency
    controller; motor midpoint is reserved for invalid emergency resources.
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
            hard_values,
            temperature=config.smooth_min_temperature,
            max_gap_budget=config.smooth_min_gap_budget,
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
        values,
        temperature=config.smooth_min_temperature,
        max_gap_budget=config.smooth_min_gap_budget,
    )

    def time_shifted_values(time_delta: Array) -> Array:
        shifted = shift_obstacle_prediction(obstacles, time_delta, dt=config.dt)
        shifted_values = runtime_policy_values(
            candidates.states,
            shifted,
            obstacle_clearance=config.obstacle_clearance,
            ego_radius=config.ego_radius,
        )
        return conservative_smooth_policy_values(
            shifted_values,
            temperature=config.smooth_min_temperature,
            max_gap_budget=config.smooth_min_gap_budget,
        )

    time_derivatives = jax.jacfwd(time_shifted_values)(jnp.asarray(0.0, state.dtype))
    collision_active = jnp.any(obstacles.mask)
    prediction_velocities = obstacle_prediction_velocities(obstacles, dt=config.dt)
    # An invalid shifted prediction can take a constant -inf branch with a numerical zero AD
    # derivative. Validate the declared motion itself, including future active nodes, instead
    # of mistaking that zero for a valid temporal certificate. Masked padding remains harmless.
    prediction_motion_valid = jnp.all(
        (~obstacles.mask[..., None]) | jnp.isfinite(prediction_velocities)
    )
    effective_temperature, gap_bound = smooth_min_conservatism(
        values,
        temperature=config.smooth_min_temperature,
        max_gap_budget=config.smooth_min_gap_budget,
    )
    gradient_valid = (
        candidates.valid
        & values.input_valid
        & prediction_motion_valid
        & jnp.isfinite(smooth_values)
        & jnp.isfinite(time_derivatives)
        & jnp.all(jnp.isfinite(gradients), axis=-1)
    )
    certificates = PolicyLibraryCertificates(
        values=values.values,
        gradients=gradients,
        gradient_valid=gradient_valid,
        fallback_wrenches=candidates.wrenches[:, 0],
        barrier_values=smooth_values,
        time_derivatives=time_derivatives,
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
        collision_enabled=collision_active,
        obstacle_velocities=prediction_velocities[0],
    )
    selected_fallback = candidates.wrenches[filtered.selected_index, 0]

    def held(wrench: Array) -> HeldActionCheck:
        return _held_action_check(
            state,
            wrench,
            estimated_model,
            obstacles,
            current_safety,
            demo_barrier_config,
            demo_filter_config,
            config,
        )

    qp_check = held(filtered.qp.action)
    initial_qp_held_operational_residual = qp_check.operational_residual
    refinement_count = jnp.asarray(0, jnp.int32)
    if config.control_interval_steps > 1 and config.predictive_operational_iterations > 0:
        extra_count = 9 * (config.control_interval_steps - 1)

        def pad_rows(value: Array) -> Array:
            extra = jnp.zeros(extra_count, value.dtype)
            if demo_filter_config.enforce_policy_barrier:
                return jnp.concatenate((value[:-1], extra, value[-1:]))
            return jnp.concatenate((value, extra))

        filtered = filtered._replace(
            qp=filtered.qp._replace(
                multipliers=pad_rows(filtered.qp.multipliers),
                active_mask=pad_rows(filtered.qp.active_mask),
            )
        )

        def refine(
            _: int, carry: tuple[VersionAFilterResult, HeldActionCheck, Array]
        ) -> tuple[VersionAFilterResult, HeldActionCheck, Array]:
            prior, prior_check, count = carry

            def repair(_: None) -> tuple[VersionAFilterResult, HeldActionCheck, Array]:
                matrix, bound = predictive_operational_halfspaces(
                    state,
                    prior.qp.action,
                    estimated_model,
                    current_safety,
                    demo_barrier_config,
                    dt=config.dt,
                    hold_steps=config.control_interval_steps,
                )
                revised = reproject_with_predictive_operational_faces(
                    prior,
                    nominal_action,
                    wrench_weight,
                    actuator,
                    demo_filter_config,
                    matrix,
                    bound,
                    omitted_obstacle_rows=(
                        0 if demo_barrier_config.include_obstacle_hocbf else obstacles.radii.size
                    ),
                    selected_fallback_wrench=selected_fallback,
                )
                return revised, held(revised.qp.action), count + 1

            return jax.lax.cond(
                prior.qp_accepted & ~prior_check.operational_passed, repair, lambda _: carry, None
            )

        filtered, qp_check, refinement_count = jax.lax.fori_loop(
            0,
            config.predictive_operational_iterations,
            refine,
            (filtered, qp_check, refinement_count),
        )
    fallback_check = held(selected_fallback)
    qp_valid = (
        filtered.qp_accepted
        & (qp_check.collision_margin >= -config.interval_tolerance)
        & qp_check.operational_passed
    )
    selected_value = values.values[filtered.selected_index]
    fallback_executable = (
        config.use_policy_constraint
        & collision_active
        & candidates.valid[filtered.selected_index]
        & values.input_valid[filtered.selected_index]
        & filtered.fallback_postcheck.actuator_passed
        & filtered.fallback_postcheck.analytic_barriers_passed
        & fallback_check.operational_passed
        & (fallback_check.collision_margin >= -config.interval_tolerance)
    )
    fallback_valid = fallback_executable & (selected_value >= selection_config.minimum_hard_value)
    use_fallback = (~qp_valid) & fallback_executable
    emergency, emergency_resources_valid = obstacle_agnostic_emergency_wrench(
        state, estimated_model, actuator, config
    )
    emergency_postcheck = postcheck_version_a_action(
        emergency, actuator, filtered, demo_filter_config
    )
    emergency_executable = emergency_resources_valid & emergency_postcheck.actuator_passed
    use_emergency = (~qp_valid) & (~use_fallback) & emergency_executable
    use_midpoint = (~qp_valid) & (~use_fallback) & (~emergency_executable)
    midpoint = filtered.motor_polytope.midpoint_wrench
    action = jnp.where(
        qp_valid,
        filtered.qp.action,
        jnp.where(use_fallback, selected_fallback, jnp.where(use_emergency, emergency, midpoint)),
    )
    action = jnp.where(filtered.motor_polytope.input_valid, action, jnp.full_like(action, jnp.nan))
    applied_check = jax.lax.cond(
        qp_valid,
        lambda _: qp_check,
        lambda _: jax.lax.cond(
            use_fallback,
            lambda _: fallback_check,
            lambda _: held(jnp.where(use_emergency, emergency, midpoint)),
            None,
        ),
        None,
    )
    applied_postcheck = postcheck_version_a_action(action, actuator, filtered, demo_filter_config)
    execution_valid = jnp.where(qp_valid, applied_postcheck.passed, use_fallback & fallback_valid)
    degraded = (
        use_emergency
        | use_midpoint
        | ~execution_valid
        | ~applied_check.operational_passed
        | ~(applied_check.collision_margin >= -config.interval_tolerance)
    )
    safe = values.input_valid & (values.values >= selection_config.minimum_hard_value)
    qp_rejection_flags = jnp.stack(
        (
            ~filtered.input_valid,
            filtered.policy_constraint_active & ~filtered.has_certificate,
            ~filtered.qp.feasible,
            filtered.qp.feasible & ~filtered.qp_kkt_valid,
            filtered.qp.feasible & ~filtered.qp_postcheck.actuator_passed,
            filtered.qp.feasible
            & filtered.policy_constraint_active
            & ~filtered.qp_postcheck.policy_barrier_passed,
            filtered.qp.feasible & ~filtered.qp_postcheck.analytic_barriers_passed,
            filtered.qp_accepted & ~(qp_check.collision_margin >= -config.interval_tolerance),
            filtered.qp_accepted & ~qp_check.operational_passed,
        )
    )
    return ContinuousVersionAStep(
        action=action,
        nominal_action=nominal_action,
        next_estimated_state=jnp.where(
            jnp.all(jnp.isfinite(action)), applied_check.next_state, jnp.full_like(state, jnp.nan)
        ),
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
        qp_held_margin=qp_check.collision_margin,
        fallback_held_margin=fallback_check.collision_margin,
        applied_held_margin=applied_check.collision_margin,
        qp_valid=qp_valid,
        fallback_valid=fallback_valid,
        used_fallback=use_fallback,
        used_midpoint=use_midpoint,
        degraded=degraded,
        qp_intervention_norm=jnp.linalg.norm(action - nominal_action),
        smooth_values=smooth_values,
        selected_policy_dual=filtered.selected_policy_dual,
        qp_rejection_flags=qp_rejection_flags,
        time_derivatives=time_derivatives,
        selected_smooth_value=smooth_values[filtered.selected_index],
        selected_time_derivative=time_derivatives[filtered.selected_index],
        collision_constraint_active=collision_active,
        effective_smooth_temperature=effective_temperature,
        smooth_gap_bound=gap_bound,
        used_emergency=use_emergency,
        execution_mode=jnp.where(
            qp_valid, 0, jnp.where(use_fallback, 1, jnp.where(use_emergency, 2, 3))
        ),
        executed_policy_dual=jnp.where(qp_valid, filtered.selected_policy_dual, 0.0),
        applied_held_operational_margin=applied_check.operational_margin,
        applied_held_operational_residual=applied_check.operational_residual,
        applied_held_operational_passed=applied_check.operational_passed,
        emergency_postcheck=emergency_postcheck,
        qp_held_operational_residuals=qp_check.operational_residuals,
        fallback_held_operational_residuals=fallback_check.operational_residuals,
        applied_held_operational_residuals=applied_check.operational_residuals,
        applied_held_physical_margins=applied_check.physical_margins,
        predictive_operational_iterations=refinement_count,
        initial_qp_held_operational_residual=initial_qp_held_operational_residual,
    )


__all__ = [
    "ContinuousVersionAConfig",
    "ContinuousVersionAStep",
    "PolicyRollouts",
    "HeldActionCheck",
    "EXECUTION_MODES",
    "QP_REJECTION_REASONS",
    "RuntimeObstacleTrajectories",
    "RuntimePolicyValues",
    "augmented_policy_rollouts",
    "continuous_version_a_step",
    "conservative_smooth_policy_values",
    "obstacle_agnostic_waypoint_callbacks",
    "obstacle_agnostic_emergency_wrench",
    "obstacle_prediction_velocities",
    "shift_obstacle_prediction",
    "smooth_min_conservatism",
    "rollout_waypoint_library",
    "runtime_policy_values",
]
