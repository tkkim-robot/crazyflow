"""Discrete nonlinear DA-PLCBF runtime for finite moving-obstacle predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.direct_wrench import motor_forces_to_wrench, wrench_to_motor_forces
from crazyflow.safety.da_plcbf.discrete_filter import (
    DiscreteActionEvaluation,
    DiscreteFilterResult,
    discrete_nonlinear_plcbf_filter,
)
from crazyflow.safety.da_plcbf.dynamic_rollouts import (
    DynamicLibraryEvaluation,
    DynamicSphereScenarioBatch,
    dynamic_quad_safety_values,
    evaluate_dynamic_quad_library,
)
from crazyflow.safety.da_plcbf.library import slice_shared_actor_policy
from crazyflow.safety.da_plcbf.quad_policy import QuadWrenchCommand, waypoint_nominal_wrench
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.selector import PolicySelection, SelectionConfig, select_hard_policy

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.actor import (
        SharedActorConfig,
        SharedActorParams,
        SharedActorSpec,
    )
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
    from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig, VersionAModel
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


@dataclass(frozen=True, slots=True)
class DynamicFilterConfig:
    """Selection, trust-region, and exact discrete-condition settings."""

    decay: float = 0.98
    tolerance: float = 2e-6
    trust_fraction: float = 0.2
    current_action_tolerance: float = 2e-5
    softmin_beta: float = 40.0
    qp_iterations: int = 64
    selection: SelectionConfig = SelectionConfig()

    def validate(self) -> None:
        """Reject settings that weaken or obscure the exact postcheck contract."""
        values = (
            self.decay,
            self.tolerance,
            self.trust_fraction,
            self.current_action_tolerance,
            self.softmin_beta,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("dynamic filter settings must be finite")
        if not 0 < self.decay <= 1:
            raise ValueError("decay must lie in (0, 1]")
        if self.tolerance < 0 or self.current_action_tolerance < 0:
            raise ValueError("dynamic filter tolerances must be nonnegative")
        if not 0 < self.trust_fraction <= 1:
            raise ValueError("trust_fraction must lie in (0, 1]")
        if self.softmin_beta <= 0:
            raise ValueError("softmin_beta must be positive")
        if (
            isinstance(self.qp_iterations, bool)
            or not isinstance(self.qp_iterations, int)
            or self.qp_iterations <= 0
        ):
            raise ValueError("qp_iterations must be a positive integer")
        self.selection.validate()


class DynamicFilterStep(NamedTuple):
    """One applied direct-wrench transition and its finite-prediction audit."""

    next_state: Array
    motor_forces: Array
    wrench: Array
    nominal: QuadWrenchCommand
    library: DynamicLibraryEvaluation
    selection: PolicySelection
    filter: DiscreteFilterResult
    admissible_scores: Array
    applied_interval_margin: Array
    applied_next_value: Array
    degraded: Array


def _slice_nodes(
    scenarios: DynamicSphereScenarioBatch, start: int, stop: int | None
) -> DynamicSphereScenarioBatch:
    return DynamicSphereScenarioBatch(
        scenarios.obstacle_centers[:, :, start:stop],
        scenarios.obstacle_radii[:, :, start:stop],
        scenarios.obstacle_mask[:, :, start:stop],
        scenarios.arena_lower,
        scenarios.arena_upper,
        scenarios.speed_limit,
        scenarios.angular_rate_max,
        scenarios.tilt_max_radians,
    )


def _motor_headroom_score(motor_forces: Array, lower: Array, upper: Array) -> Array:
    span = upper - lower
    lower_fraction = (motor_forces - lower) / span
    upper_fraction = (upper - motor_forces) / span
    score = jnp.min(jnp.minimum(lower_fraction, upper_fraction), axis=-1)
    return jnp.where(jnp.all(jnp.isfinite(motor_forces), axis=-1), score, jnp.nan)


def dynamic_discrete_runtime_step(
    state: Array,
    target_position: Array,
    target_velocity: Array,
    previous_policy_index: Array,
    params: SharedActorParams,
    spec: SharedActorSpec,
    prediction_window: DynamicSphereScenarioBatch,
    model: VersionAModel,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    barrier_config: VersionABarrierConfig,
    filter_config: DynamicFilterConfig,
    *,
    dt: float,
    policy_gain: float,
) -> DynamicFilterStep:
    """Apply one exact-postchecked motor-force decision against moving predictions.

    ``prediction_window`` must contain ``certificate_horizon + 2`` nodes.  The current value uses
    all but the last node and the next value uses all but the first, so both sides of the discrete
    condition use the same finite horizon.  Motor forces are the optimization variables, making
    the coupled wrench polytope an exact box without downstream allocation clipping.
    """
    filter_config.validate()
    barrier_config.validate()
    actor_config.validate()
    quad_config.validate()
    if state.shape != (13,) or target_position.shape != (3,) or target_velocity.shape != (3,):
        raise ValueError("state/target shapes must be (13,), (3,), and (3,)")
    if prediction_window.obstacle_centers.shape[0] != 1:
        raise ValueError("runtime prediction window must have B=1")
    if prediction_window.obstacle_centers.shape[2] < 3:
        raise ValueError("prediction window must contain at least three nodes")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(policy_gain) or policy_gain <= 0:
        raise ValueError("policy_gain must be finite and positive")

    current_window = _slice_nodes(prediction_window, 0, -1)
    next_window = _slice_nodes(prediction_window, 1, None)
    interval_window = _slice_nodes(prediction_window, 0, 2)
    current_library = evaluate_dynamic_quad_library(
        params,
        spec,
        state,
        current_window,
        model,
        actuator,
        actor_config,
        quad_config,
        barrier_config,
        dt=dt,
        policy_gain=policy_gain,
        softmin_beta=filter_config.softmin_beta,
        current_action_tolerance=filter_config.current_action_tolerance,
    )
    lower = jnp.broadcast_to(jnp.asarray(actuator.thrust_min, dtype=state.dtype), (4,))
    upper = jnp.broadcast_to(jnp.asarray(actuator.thrust_max, dtype=state.dtype), (4,))
    scores = _motor_headroom_score(current_library.first_motor_forces, lower, upper)
    selection = select_hard_policy(
        current_library.hard_values, scores, previous_policy_index, filter_config.selection
    )
    selected_params, selected_spec = slice_shared_actor_policy(
        params, spec, selection.selected_index
    )
    # JAX 0.11.1's CUDA scan lowering can corrupt a closed scalar ``dt`` when the
    # selected-policy rollout has a singleton policy axis.  Evaluate two byte-identical
    # copies of that already selected policy and consume only the first result below.  This
    # is an execution-shape workaround, not an extra policy or a change to the decision.
    selected_spec = selected_spec.replace(
        base_codes=jnp.repeat(selected_spec.base_codes, 2, axis=0),
        base_desired_velocities=jnp.repeat(selected_spec.base_desired_velocities, 2, axis=0),
        base_durations=jnp.repeat(selected_spec.base_durations, 2, axis=0),
        adaptive_mask=jnp.repeat(selected_spec.adaptive_mask, 2, axis=0),
    )
    selected_params = selected_params.replace(
        code_offsets=jnp.repeat(selected_params.code_offsets, 2, axis=0),
        velocity_offsets=jnp.repeat(selected_params.velocity_offsets, 2, axis=0),
        duration_offsets=jnp.repeat(selected_params.duration_offsets, 2, axis=0),
    )
    fallback_motor = current_library.first_motor_forces[selection.selected_index]
    nominal = waypoint_nominal_wrench(
        state, target_position, target_velocity, model, actuator, quad_config
    )
    nominal_motor = nominal.bounded_motor_forces
    span = upper - lower

    def evaluate_action(motor_forces: Array) -> DiscreteActionEvaluation:
        wrench = motor_forces_to_wrench(
            motor_forces,
            L=actuator.arm_length,
            thrust2torque=actuator.thrust_to_torque,
            mixing_matrix=actuator.mixing_matrix,
        )
        reconstructed = wrench_to_motor_forces(
            wrench,
            L=actuator.arm_length,
            thrust2torque=actuator.thrust_to_torque,
            mixing_matrix=actuator.mixing_matrix,
        )
        next_state = direct_wrench_symplectic_step(state, wrench, model, dt)
        next_library = evaluate_dynamic_quad_library(
            selected_params,
            selected_spec,
            next_state,
            next_window,
            model,
            actuator,
            actor_config,
            quad_config,
            barrier_config,
            dt=dt,
            policy_gain=policy_gain,
            softmin_beta=filter_config.softmin_beta,
            current_action_tolerance=filter_config.current_action_tolerance,
        )
        prediction_count = interval_window.obstacle_centers.shape[1]
        pair = jnp.stack((state, next_state))
        interval_states = jnp.broadcast_to(pair, (1, 1, prediction_count, 2, 13))
        interval_values = dynamic_quad_safety_values(
            interval_states,
            interval_window,
            barrier_config,
            softmin_beta=filter_config.softmin_beta,
        )
        bound_residual = jnp.max(jnp.concatenate((lower - motor_forces, motor_forces - upper)))
        motor_scale = jnp.maximum(jnp.max(jnp.abs(motor_forces)), 1.0)
        roundtrip_error = jnp.max(jnp.abs(reconstructed - motor_forces)) / motor_scale
        actuator_residual = jnp.maximum(
            bound_residual, roundtrip_error - quad_config.allocation_tolerance
        )
        return DiscreteActionEvaluation(
            next_value=next_library.hard_values[0],
            interval_margin=interval_values.robust_hard_margins[0, 0],
            actuator_residual=actuator_residual,
            applied_action=motor_forces,
        )

    filtered = discrete_nonlinear_plcbf_filter(
        nominal_motor,
        fallback_motor,
        lower,
        upper,
        1.0 / span**2,
        filter_config.trust_fraction * span,
        selection.selected_hard_value,
        selection.has_certificate,
        evaluate_action,
        decay=filter_config.decay,
        tolerance=filter_config.tolerance,
        qp_iterations=filter_config.qp_iterations,
    )
    applied_evidence = evaluate_action(filtered.action)
    applied_wrench = motor_forces_to_wrench(
        filtered.action,
        L=actuator.arm_length,
        thrust2torque=actuator.thrust_to_torque,
        mixing_matrix=actuator.mixing_matrix,
    )
    next_state = direct_wrench_symplectic_step(state, applied_wrench, model, dt)
    degraded = filtered.degraded | (applied_evidence.interval_margin < -filter_config.tolerance)
    return DynamicFilterStep(
        next_state,
        filtered.action,
        applied_wrench,
        nominal,
        current_library,
        selection,
        filtered,
        scores,
        applied_evidence.interval_margin,
        applied_evidence.next_value,
        degraded,
    )


__all__ = ["DynamicFilterConfig", "DynamicFilterStep", "dynamic_discrete_runtime_step"]
