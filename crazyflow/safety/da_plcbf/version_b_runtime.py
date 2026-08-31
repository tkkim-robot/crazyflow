"""Complete nonlinear Crazyflow runtime adapter for DA-PLCBF Version B.

Version B does not inherit the control-affine guarantee of Version A.  Its policy values are
recomputed through Crazyflow's force/torque controller, motor allocation and clipping, rotor lag,
first-principles dynamics, and configured integrator.  The convenience floor clamp is absent from
the supplied ``one_step`` function.  Every candidate command is independently replayed over all
held substeps, including swept static-obstacle checks, before the discrete PL-CBF result can be
accepted.

The exact discrete condition for the selected policy ``i`` is

``V_i(full_stack_step(x, u); H) - decay * V_i(x; H) >= 0``.

Both values use the same policy and the same horizon ``H``.  A local trust-region linearization is
only a proposal mechanism; :func:`discrete_nonlinear_plcbf_filter` performs the nonlinear
postcheck.  Missing certificates, internal allocation clipping, actuator violations, replay
disagreement, or an unsafe fallback are explicit degraded outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.discrete_filter import (
    DiscreteActionEvaluation,
    DiscreteFilterResult,
    discrete_nonlinear_plcbf_filter,
)
from crazyflow.safety.da_plcbf.full_stack import (
    FullStackCommandRollout,
    rollout_full_stack_force_torque,
)
from crazyflow.safety.da_plcbf.library import slice_shared_actor_policy
from crazyflow.safety.da_plcbf.quad_actor_losses import quad_safety_values
from crazyflow.safety.da_plcbf.quad_policy import shared_quad_fallback_wrenches
from crazyflow.safety.da_plcbf.selector import PolicySelection, SelectionConfig, select_hard_policy
from crazyflow.sim import functional as sim_functional

if TYPE_CHECKING:
    from collections.abc import Callable

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
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator
    from crazyflow.sim.data import SimData


@dataclass(frozen=True, slots=True)
class VersionBRuntimeConfig:
    """Static rollout, nonlinear-filter, and audit settings for one decision."""

    n_substeps: int = 2
    certificate_horizon: int = 4
    policy_gain: float = 1.8
    decay: float = 0.99
    tolerance: float = 1e-5
    qp_iterations: int = 64

    def validate(self) -> None:
        """Reject settings that change or weaken the exact discrete contract."""
        for name, value in (
            ("n_substeps", self.n_substeps),
            ("certificate_horizon", self.certificate_horizon),
            ("qp_iterations", self.qp_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.policy_gain) or self.policy_gain <= 0:
            raise ValueError("policy_gain must be finite and positive")
        if not math.isfinite(self.decay) or not 0 < self.decay <= 1:
            raise ValueError("decay must be finite and in (0, 1]")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")


class ExactHeldCommand(NamedTuple):
    """One held wrench executed and independently replayed through the full stack."""

    rollout: FullStackCommandRollout
    replay_final_data: Any
    state_trace: Array
    rotor_velocity_trace: Array
    constraint_values: Array
    interval_margin: Array
    node_interval_margin: Array
    replay_state_error: Array
    rotor_lower_residual: Array
    audit_residual: Array
    valid: Array


class VersionBLibraryCertificates(NamedTuple):
    """Hard full-stack values and audit traces for all shared fallback policies."""

    values: Array
    rollout_valid: Array
    fallback_wrenches: Array
    constraint_values: Array
    state_traces: Array
    held_interval_margins: Array
    actuator_residuals: Array
    replay_state_errors: Array
    policy_command_valid: Array
    first_raw_motor_forces: Array
    admissible_scores: Array


class VersionBActionEvidence(NamedTuple):
    """Exact action evidence plus the same-policy equal-horizon next certificate."""

    evaluation: DiscreteActionEvaluation
    held: ExactHeldCommand
    next_certificate: VersionBLibraryCertificates


class VersionBRuntimeStep(NamedTuple):
    """Applied nonlinear transition and complete selection/filter/postcheck record."""

    next_data: Any
    action: Array
    applied_motor_forces: Array
    certificates: VersionBLibraryCertificates
    admissible_scores: Array
    selection: PolicySelection
    selected_index: Array
    selected_value: Array
    selected_fallback: Array
    has_certificate: Array
    discrete_filter: DiscreteFilterResult
    applied_evidence: VersionBActionEvidence
    applied_exact_residual: Array
    applied_accepted: Array
    postcheck_replay_error: Array
    degraded: Array


def sim_data_to_version_b_state(data: SimData) -> Array:
    """Extract ``[pos, quat_xyzw, world_vel, body_rate]`` for one world/drone.

    Crazyflow stores scalar-last quaternions, so this is a direct concatenation rather than a
    reordering.  Rotor speed and controller memory remain in ``SimData`` and therefore remain part
    of the nonlinear certificate state even though the policy observation has 13 components.
    """
    if data.core.n_worlds != 1 or data.core.n_drones != 1:
        raise ValueError("Version-B runtime requires exactly one world and one drone")
    expected = {"pos": (1, 1, 3), "quat": (1, 1, 4), "vel": (1, 1, 3), "ang_vel": (1, 1, 3)}
    for name, shape in expected.items():
        if getattr(data.states, name).shape != shape:
            raise ValueError(f"states.{name} must have shape {shape}")
    return jnp.concatenate(
        (
            data.states.pos[0, 0],
            data.states.quat[0, 0],
            data.states.vel[0, 0],
            data.states.ang_vel[0, 0],
        )
    )


def replace_version_b_state(data: SimData, state: Array) -> SimData:
    """Replace the 13 observable plant coordinates without changing hidden actuator state."""
    if state.shape != (13,):
        raise ValueError("state must have shape (13,)")
    sim_data_to_version_b_state(data)
    states = data.states.replace(
        pos=data.states.pos.at[0, 0].set(state[:3]),
        quat=data.states.quat.at[0, 0].set(state[3:7]),
        vel=data.states.vel.at[0, 0].set(state[7:10]),
        ang_vel=data.states.ang_vel.at[0, 0].set(state[10:13]),
    )
    return data.replace(states=states)


def _single_safety(safety: RigidBodySafetySet) -> RigidBodySafetySet:
    """Remove the required singleton runtime scenario axis."""
    if safety.obstacle_centers.ndim != 3 or safety.obstacle_centers.shape[0] != 1:
        raise ValueError("Version-B runtime safety must contain exactly one scenario")
    return safety._replace(
        obstacle_centers=safety.obstacle_centers[0],
        obstacle_radii=safety.obstacle_radii[0],
        obstacle_mask=safety.obstacle_mask[0],
        arena_lower=safety.arena_lower[0],
        arena_upper=safety.arena_upper[0],
        speed_max=safety.speed_max[0],
        angular_rate_max=safety.angular_rate_max[0],
        tilt_max_radians=safety.tilt_max_radians[0],
    )


def _flatten_constraints(values: Any) -> Array:
    nodes = jnp.where(values.node_enabled, values.node_values, jnp.inf).reshape(
        values.node_values.shape[0], values.node_values.shape[1], -1
    )
    segments = jnp.where(
        values.segment_obstacle_enabled, values.segment_obstacle_values, jnp.inf
    ).reshape(values.node_values.shape[0], values.node_values.shape[1], -1)
    return jnp.concatenate((nodes, segments), axis=-1)


def _physical_state_vector(data: SimData) -> Array:
    """Vector used only for an independent deterministic execution crosscheck."""
    force_torque = data.controls.force_torque
    if force_torque is None:
        raise ValueError("Version-B runtime requires initialized force/torque control")
    return jnp.concatenate(
        (
            sim_data_to_version_b_state(data),
            data.states.rotor_vel[0, 0],
            data.states.force[0, 0],
            data.states.torque[0, 0],
            data.controls.rotor_vel[0, 0],
            force_torque.cmd[0, 0],
            force_torque.staged_cmd[0, 0],
            force_torque.steps[0].astype(data.states.pos.dtype),
            data.core.steps[0].astype(data.states.pos.dtype),
        )
    )


def execute_version_b_held_command(
    data: SimData,
    command: Array,
    one_step: Callable[[SimData], SimData],
    safety: RigidBodySafetySet,
    barrier_config: VersionABarrierConfig,
    *,
    n_substeps: int,
    tolerance: float,
) -> ExactHeldCommand:
    """Execute and independently replay one command with exact static safety checks.

    The authoritative transition comes from :func:`rollout_full_stack_force_torque`.  A second
    execution records every physical substep.  The recorded trace is evaluated with the same hard
    dimensionless constraints as the library, including exact closest-point checks on every
    straight segment between simulator nodes.  The two final physical states must agree.
    """
    barrier_config.validate()
    if command.shape != (4,):
        raise ValueError("command must have shape (4,)")
    if isinstance(n_substeps, bool) or not isinstance(n_substeps, int) or n_substeps <= 0:
        raise ValueError("n_substeps must be a positive integer")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    singleton_safety = _single_safety(safety)

    def node_margin(current: SimData) -> Array:
        from crazyflow.safety.da_plcbf.version_a_barriers import dimensionless_safety_values

        result = dimensionless_safety_values(
            sim_data_to_version_b_state(current), singleton_safety, barrier_config
        )
        margin = jnp.min(jnp.where(result.enabled, result.values, jnp.inf))
        return jnp.where(result.input_valid, margin, -jnp.inf)

    rollout = rollout_full_stack_force_torque(
        data, command, one_step, node_margin, n_substeps=n_substeps
    )
    staged = sim_functional.force_torque_control(data, command[None, None, :])

    def replay(carry: SimData, _: None) -> tuple[SimData, tuple[Array, Array]]:
        following = one_step(carry)
        return following, (sim_data_to_version_b_state(following), following.states.rotor_vel[0, 0])

    replay_final, (future_states, future_rotor_velocities) = jax.lax.scan(
        replay, staged, xs=None, length=n_substeps
    )
    state_trace = jnp.concatenate(
        (sim_data_to_version_b_state(data)[None, :], future_states), axis=0
    )
    rotor_trace = jnp.concatenate(
        (data.states.rotor_vel[0, 0][None, :], future_rotor_velocities), axis=0
    )
    safety_values = quad_safety_values(
        state_trace[None, None, ...], safety, barrier_config, softmin_beta=1.0
    )
    constraint_values = _flatten_constraints(safety_values)[0, 0]
    interval_margin = safety_values.hard_policy_margins[0, 0]
    replay_error = jnp.max(
        jnp.abs(_physical_state_vector(replay_final) - _physical_state_vector(rollout.final_data))
    )
    rotor_lower_residual = jnp.max(-rotor_trace)
    audit_residual = jnp.maximum(
        jnp.maximum(rollout.actuator_residual, replay_error), rotor_lower_residual
    )
    finite = (
        jnp.all(jnp.isfinite(command))
        & jnp.all(jnp.isfinite(state_trace))
        & jnp.all(jnp.isfinite(rotor_trace))
        & jnp.isfinite(interval_margin)
        & jnp.isfinite(rollout.interval_margin)
        & jnp.isfinite(audit_residual)
    )
    valid = (
        finite
        & rollout.command_committed
        & safety_values.input_valid[0, 0]
        & (audit_residual <= tolerance)
    )
    return ExactHeldCommand(
        rollout,
        replay_final,
        state_trace,
        rotor_trace,
        constraint_values,
        interval_margin,
        rollout.interval_margin,
        replay_error,
        rotor_lower_residual,
        audit_residual,
        valid,
    )


def _broadcast_sim_data(data: SimData, count: int) -> SimData:
    """Add a policy-vmap axis to every dynamic ``SimData`` leaf."""
    return jax.tree.map(lambda value: jnp.broadcast_to(value, (count, *value.shape)), data)


def version_b_shared_library_certificates(
    data: SimData,
    params: SharedActorParams,
    spec: SharedActorSpec,
    scenarios: CircleScenarioBatch,
    safety: RigidBodySafetySet,
    model: VersionAModel,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    barrier_config: VersionABarrierConfig,
    one_step: Callable[[SimData], SimData],
    *,
    n_substeps: int,
    horizon: int,
    policy_gain: float,
    tolerance: float,
) -> VersionBLibraryCertificates:
    """Compute hard shared-policy certificates through the complete nonlinear stack.

    Every policy begins from an identical copy of the full ``SimData``, including rotor speed,
    committed controller state, and control clock.  A policy command is held for ``n_substeps``;
    the common certificate contains ``horizon * n_substeps + 1`` physical nodes.  Invalid actor
    outputs or any command that relies on internal clipping invalidate that policy rather than
    improving its hard value.
    """
    actor_config.validate()
    quad_config.validate()
    barrier_config.validate()
    sim_data_to_version_b_state(data)
    if scenarios.obstacle_centers.shape[0] != 1:
        raise ValueError("Version-B runtime scenarios must contain exactly one scenario")
    if safety.obstacle_centers.shape[0] != 1:
        raise ValueError("Version-B runtime safety must contain exactly one scenario")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if isinstance(n_substeps, bool) or not isinstance(n_substeps, int) or n_substeps <= 0:
        raise ValueError("n_substeps must be a positive integer")
    if not math.isfinite(policy_gain) or policy_gain <= 0:
        raise ValueError("policy_gain must be finite and positive")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    policy_count = spec.base_codes.shape[0]
    if policy_count < 1:
        raise ValueError("the policy library must not be empty")
    decision_dt = n_substeps / data.core.freq
    horizon_duration = horizon * decision_dt
    policy_data = _broadcast_sim_data(data, policy_count)

    def advance(carry: SimData, step_index: Array) -> tuple[SimData, tuple[Array, ...]]:
        states = jnp.concatenate(
            (
                carry.states.pos[:, 0, 0],
                carry.states.quat[:, 0, 0],
                carry.states.vel[:, 0, 0],
                carry.states.ang_vel[:, 0, 0],
            ),
            axis=-1,
        )[:, None, :]
        commands = shared_quad_fallback_wrenches(
            params,
            spec,
            states,
            scenarios,
            model,
            actuator,
            elapsed=step_index * decision_dt,
            horizon_duration=horizon_duration,
            policy_gain=policy_gain,
            actor_config=actor_config,
            quad_config=quad_config,
        )
        wrenches = commands.wrench[:, 0]

        def execute(single_data: SimData, wrench: Array) -> ExactHeldCommand:
            return execute_version_b_held_command(
                single_data,
                wrench,
                one_step,
                safety,
                barrier_config,
                n_substeps=n_substeps,
                tolerance=tolerance,
            )

        held = jax.vmap(execute)(carry, wrenches)
        return held.rollout.final_data, (
            held.state_trace[:, 1:],
            wrenches,
            commands.input_valid[:, 0],
            held.interval_margin,
            held.audit_residual,
            held.replay_state_error,
            held.valid,
            held.rollout.raw_motor_forces,
        )

    _, outputs = jax.lax.scan(
        advance, policy_data, jnp.arange(horizon, dtype=data.states.pos.dtype)
    )
    (
        substep_states,
        commands,
        command_valid,
        interval_margins,
        actuator_residuals,
        replay_errors,
        held_valid,
        raw_motor_forces,
    ) = outputs
    substep_states = jnp.swapaxes(substep_states, 0, 1).reshape(
        policy_count, horizon * n_substeps, 13
    )
    initial = jnp.broadcast_to(
        sim_data_to_version_b_state(data)[None, None, :], (policy_count, 1, 13)
    )
    traces = jnp.concatenate((initial, substep_states), axis=1)
    safety_values = quad_safety_values(
        traces[:, None, ...], safety, barrier_config, softmin_beta=1.0
    )
    constraints = _flatten_constraints(safety_values)[:, 0]
    command_valid = jnp.swapaxes(command_valid, 0, 1)
    held_valid = jnp.swapaxes(held_valid, 0, 1)
    interval_margins = jnp.swapaxes(interval_margins, 0, 1)
    actuator_residuals = jnp.swapaxes(actuator_residuals, 0, 1)
    replay_errors = jnp.swapaxes(replay_errors, 0, 1)
    raw_motor_forces = jnp.swapaxes(raw_motor_forces, 0, 1)
    rollout_valid = (
        safety_values.input_valid[:, 0]
        & jnp.all(command_valid, axis=-1)
        & jnp.all(held_valid, axis=-1)
    )
    values = jnp.where(rollout_valid, safety_values.hard_policy_margins[:, 0], -jnp.inf)
    force_torque = data.controls.force_torque
    if force_torque is None:
        raise ValueError("Version-B runtime requires initialized force/torque control")
    lower = jnp.broadcast_to(
        jnp.asarray(force_torque.params["thrust_min"], dtype=data.states.pos.dtype), (4,)
    )
    upper = jnp.broadcast_to(
        jnp.asarray(force_torque.params["thrust_max"], dtype=data.states.pos.dtype), (4,)
    )
    span = upper - lower
    bound_data_valid = (
        jnp.all(jnp.isfinite(lower)) & jnp.all(jnp.isfinite(upper)) & jnp.all(span > 0)
    )
    safe_span = jnp.where(span > 0, span, 1.0)
    first_motor_forces = raw_motor_forces[:, 0]
    normalized_lower_headroom = (first_motor_forces - lower) / safe_span
    normalized_upper_headroom = (upper - first_motor_forces) / safe_span
    # Two times the minimum normalized distance to a motor face lies in [0, 1] for an exactly
    # feasible command and reaches one only at the center of every motor interval.  It is a local
    # admissible-set proxy, not a learned score or maneuver label.
    admissible_scores = 2.0 * jnp.min(
        jnp.concatenate((normalized_lower_headroom, normalized_upper_headroom), axis=-1), axis=-1
    )
    score_valid = rollout_valid & bound_data_valid & jnp.isfinite(admissible_scores)
    admissible_scores = jnp.where(score_valid, admissible_scores, jnp.nan)
    return VersionBLibraryCertificates(
        values,
        rollout_valid,
        commands[0],
        constraints,
        traces,
        interval_margins,
        actuator_residuals,
        replay_errors,
        command_valid,
        first_motor_forces,
        admissible_scores,
    )


def version_b_action_evidence(
    data: SimData,
    command: Array,
    selected_index: Array,
    params: SharedActorParams,
    spec: SharedActorSpec,
    scenarios: CircleScenarioBatch,
    safety: RigidBodySafetySet,
    model: VersionAModel,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    barrier_config: VersionABarrierConfig,
    one_step: Callable[[SimData], SimData],
    *,
    n_substeps: int,
    horizon: int,
    policy_gain: float,
    tolerance: float,
) -> VersionBActionEvidence:
    """Evaluate one command and the same selected policy's equal-horizon next value."""
    if jnp.ndim(selected_index) != 0:
        raise ValueError("selected_index must be scalar")
    held = execute_version_b_held_command(
        data, command, one_step, safety, barrier_config, n_substeps=n_substeps, tolerance=tolerance
    )
    selected_params, selected_spec = slice_shared_actor_policy(params, spec, selected_index)
    next_certificate = version_b_shared_library_certificates(
        held.rollout.final_data,
        selected_params,
        selected_spec,
        scenarios,
        safety,
        model,
        actuator,
        actor_config,
        quad_config,
        barrier_config,
        one_step,
        n_substeps=n_substeps,
        horizon=horizon,
        policy_gain=policy_gain,
        tolerance=tolerance,
    )
    next_value = next_certificate.values[0]
    audit_residual = jnp.where(
        held.valid & next_certificate.rollout_valid[0], held.audit_residual, jnp.inf
    )
    evaluation = DiscreteActionEvaluation(
        next_value=next_value,
        interval_margin=held.interval_margin,
        actuator_residual=audit_residual,
        applied_action=held.rollout.final_motor_forces,
    )
    return VersionBActionEvidence(evaluation, held, next_certificate)


def version_b_runtime_step(
    data: SimData,
    nominal_action: Array,
    params: SharedActorParams,
    spec: SharedActorSpec,
    scenarios: CircleScenarioBatch,
    safety: RigidBodySafetySet,
    model: VersionAModel,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    barrier_config: VersionABarrierConfig,
    one_step: Callable[[SimData], SimData],
    action_lower: Array,
    action_upper: Array,
    weight: Array,
    trust_radius: Array,
    config: VersionBRuntimeConfig,
    *,
    previous_policy_index: Array | None = None,
    selection_config: SelectionConfig | None = None,
) -> VersionBRuntimeStep:
    """Make and execute one exact nonlinear discrete PL-CBF decision.

    ``action_lower``/``action_upper`` bound the trust-region proposal only.  They need not pretend
    that the coupled motor-feasible wrench set is a box: the actual Crazyflow allocation is audited
    independently and any internal clipping rejects the proposal.  The library fallback is
    evaluated by the identical full-stack checks.  If neither path is certified, the returned
    command is best effort and ``degraded`` remains true.
    """
    config.validate()
    if selection_config is None:
        selection_config = SelectionConfig()
    selection_config.validate()
    if selection_config.minimum_hard_value < 0:
        raise ValueError("Version-B selection minimum_hard_value must be nonnegative")
    if previous_policy_index is None:
        previous_policy_index = jnp.asarray(-1, dtype=jnp.int32)
    if jnp.ndim(previous_policy_index) != 0:
        raise ValueError("previous_policy_index must be scalar")
    vectors = (nominal_action, action_lower, action_upper, weight, trust_radius)
    if any(value.shape != (4,) for value in vectors):
        raise ValueError("all runtime action vectors must have shape (4,)")
    current = version_b_shared_library_certificates(
        data,
        params,
        spec,
        scenarios,
        safety,
        model,
        actuator,
        actor_config,
        quad_config,
        barrier_config,
        one_step,
        n_substeps=config.n_substeps,
        horizon=config.certificate_horizon,
        policy_gain=config.policy_gain,
        tolerance=config.tolerance,
    )
    selection = select_hard_policy(
        current.values, current.admissible_scores, previous_policy_index, selection_config
    )
    has_certificate = selection.has_certificate
    selected_index = selection.selected_index
    selected_value = selection.selected_hard_value
    filter_value = jnp.where(has_certificate, selected_value, 0.0)
    selected_fallback = current.fallback_wrenches[selected_index]

    def evaluate(command: Array) -> DiscreteActionEvaluation:
        evidence = version_b_action_evidence(
            data,
            command,
            selected_index,
            params,
            spec,
            scenarios,
            safety,
            model,
            actuator,
            actor_config,
            quad_config,
            barrier_config,
            one_step,
            n_substeps=config.n_substeps,
            horizon=config.certificate_horizon,
            policy_gain=config.policy_gain,
            tolerance=config.tolerance,
        )
        return evidence.evaluation

    filtered = discrete_nonlinear_plcbf_filter(
        nominal_action,
        selected_fallback,
        action_lower,
        action_upper,
        weight,
        trust_radius,
        filter_value,
        has_certificate,
        evaluate,
        decay=config.decay,
        tolerance=config.tolerance,
        qp_iterations=config.qp_iterations,
    )
    applied = version_b_action_evidence(
        data,
        filtered.action,
        selected_index,
        params,
        spec,
        scenarios,
        safety,
        model,
        actuator,
        actor_config,
        quad_config,
        barrier_config,
        one_step,
        n_substeps=config.n_substeps,
        horizon=config.certificate_horizon,
        policy_gain=config.policy_gain,
        tolerance=config.tolerance,
    )
    exact_residual = applied.evaluation.next_value - config.decay * filter_value
    recorded_residual = jnp.where(
        filtered.used_fallback, filtered.fallback_exact_residual, filtered.proposal_exact_residual
    )
    recorded_interval = jnp.where(
        filtered.used_fallback, filtered.fallback_interval_margin, filtered.proposal_interval_margin
    )
    recorded_actuator = jnp.where(
        filtered.used_fallback,
        filtered.fallback_actuator_residual,
        filtered.proposal_actuator_residual,
    )
    postcheck_error = jnp.max(
        jnp.abs(
            jnp.stack(
                (
                    exact_residual - recorded_residual,
                    applied.evaluation.interval_margin - recorded_interval,
                    applied.evaluation.actuator_residual - recorded_actuator,
                )
            )
        )
    )
    decision_accepted = jnp.where(
        filtered.used_fallback, filtered.fallback_accepted, filtered.proposal_accepted
    )
    applied_accepted = (
        has_certificate
        & decision_accepted
        & jnp.isfinite(exact_residual)
        & (exact_residual >= -config.tolerance)
        & jnp.isfinite(applied.evaluation.interval_margin)
        & (applied.evaluation.interval_margin >= -config.tolerance)
        & jnp.isfinite(applied.evaluation.actuator_residual)
        & (applied.evaluation.actuator_residual <= config.tolerance)
        & jnp.isfinite(postcheck_error)
        & (postcheck_error <= config.tolerance)
    )
    return VersionBRuntimeStep(
        applied.held.rollout.final_data,
        filtered.action,
        applied.evaluation.applied_action,
        current,
        current.admissible_scores,
        selection,
        selected_index,
        selected_value,
        selected_fallback,
        has_certificate,
        filtered,
        applied,
        exact_residual,
        applied_accepted,
        postcheck_error,
        filtered.degraded | (~applied_accepted),
    )


__all__ = [
    "ExactHeldCommand",
    "VersionBActionEvidence",
    "VersionBLibraryCertificates",
    "VersionBRuntimeConfig",
    "VersionBRuntimeStep",
    "execute_version_b_held_command",
    "replace_version_b_state",
    "sim_data_to_version_b_state",
    "version_b_action_evidence",
    "version_b_runtime_step",
    "version_b_shared_library_certificates",
]
