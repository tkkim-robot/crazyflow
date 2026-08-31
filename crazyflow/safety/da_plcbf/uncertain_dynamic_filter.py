"""Robust discrete PL-CBF filtering over dynamics and obstacle prediction axes.

The uncertainty contract in this module is deliberately Cartesian.  A finite dynamics set with
``R_m`` members and a moving-obstacle prediction set with ``R_o`` members produces ``R_m * R_o``
closed-loop trajectories for each fallback policy.  The actor and geometric controller never see
the dynamics-sample index: one controller model produces commands and only the realized rotor
forces and plant transition vary across ``R_m``.  The hard certificate is the exact minimum over
both axes.

The runtime command is sample independent.  The exact nonlinear postcheck advances that one motor
command through every dynamics sample, evaluates the selected fallback from every resulting state
against the complete future Cartesian set, and accepts only the worst residual and swept interval
margin.  This is a finite-sample simulation contract, not a distribution-free guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.direct_wrench import motor_forces_to_wrench, wrench_to_motor_forces
from crazyflow.safety.da_plcbf.discrete_filter import (
    DiscreteActionEvaluation,
    DiscreteFilterResult,
    discrete_nonlinear_plcbf_filter,
)
from crazyflow.safety.da_plcbf.dynamic_rollouts import (
    DynamicSafetyValues,
    DynamicSphereScenarioBatch,
    dynamic_quad_safety_values,
)
from crazyflow.safety.da_plcbf.library import slice_shared_actor_policy
from crazyflow.safety.da_plcbf.quad_policy import (
    QuadWrenchCommand,
    shared_quad_fallback_wrenches,
    waypoint_nominal_wrench,
)
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.selector import PolicySelection, select_hard_policy
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.actor import (
        SharedActorConfig,
        SharedActorParams,
        SharedActorSpec,
    )
    from crazyflow.safety.da_plcbf.dynamic_filter import DynamicFilterConfig
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
    from crazyflow.safety.da_plcbf.quad_uncertainty import VersionAModelSamples
    from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig, VersionAModel
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


class UncertainDynamicRolloutBatch(NamedTuple):
    """Policy rollouts with explicit ``[K,B,R_o,R_m,T]`` leading axes."""

    states: Array
    commanded_wrenches: Array
    realized_wrenches: Array
    commanded_motor_forces: Array
    realized_motor_forces: Array
    policy_valid: Array
    model_sample_valid: Array


class UncertainDynamicLibraryEvaluation(NamedTuple):
    """Exact Cartesian hard values and sample-independent first commands."""

    rollouts: UncertainDynamicRolloutBatch
    safety_values: DynamicSafetyValues
    hard_values: Array
    smooth_values: Array
    first_motor_forces: Array
    first_wrenches: Array
    first_action_consistent: Array
    policy_valid: Array


class UncertainDynamicFilterStep(NamedTuple):
    """One robust decision and all finite-sample nonlinear postcheck evidence."""

    motor_forces: Array
    wrench: Array
    nominal: QuadWrenchCommand
    library: UncertainDynamicLibraryEvaluation
    selection: PolicySelection
    filter: DiscreteFilterResult
    admissible_scores: Array
    sampled_next_states: Array
    applied_interval_margin: Array
    applied_next_value: Array
    degraded: Array


@dataclass(frozen=True, slots=True)
class CartesianUncertaintyConfig:
    """Numerical checks for the explicit ``R_o x R_m`` finite uncertainty set."""

    model_tolerance: float = 2e-5
    weight_tolerance: float = 2e-6
    first_action_tolerance: float = 2e-5

    def validate(self) -> None:
        """Reject negative or non-finite audit tolerances."""
        values = (self.model_tolerance, self.weight_tolerance, self.first_action_tolerance)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Cartesian uncertainty tolerances must be finite and nonnegative")


def _validate_shapes(
    scenarios: DynamicSphereScenarioBatch, model_samples: VersionAModelSamples
) -> tuple[int, int, int, int, int]:
    centers = scenarios.obstacle_centers
    if centers.ndim != 5 or centers.shape[-1] != 3:
        raise ValueError("obstacle_centers must have shape (B,R_o,T,O,3)")
    batch, obstacle_samples, nodes, obstacles, _ = centers.shape
    if min(batch, obstacle_samples, nodes, obstacles) <= 0 or nodes < 2:
        raise ValueError("dynamic scenario axes must be positive and contain at least two nodes")
    models = model_samples.models
    model_count = jnp.asarray(model_samples.sample_valid).shape[0]
    if model_count not in (4, 8):
        raise ValueError("dynamics uncertainty must contain exactly 4 or 8 samples")
    expected_model_shapes = {
        "mass": (model_count,),
        "gravity_vec": (model_count, 3),
        "inertia": (model_count, 3, 3),
        "inertia_inv": (model_count, 3, 3),
        "drag_matrix": (model_count, 3, 3),
        "wind_velocity": (model_count, 3),
        "external_force": (model_count, 3),
        "external_torque": (model_count, 3),
    }
    for name, shape in expected_model_shapes.items():
        if jnp.asarray(getattr(models, name)).shape != shape:
            raise ValueError(f"model_samples.models.{name} must have shape {shape}")
    if jnp.asarray(model_samples.rotor_efficiency).shape != (model_count, 4):
        raise ValueError("model_samples.rotor_efficiency must have shape (R_m,4)")
    if jnp.asarray(model_samples.weights).shape != (model_count,):
        raise ValueError("model_samples.weights must have shape (R_m,)")
    return batch, obstacle_samples, model_count, nodes, obstacles


def _model_validity(
    model_samples: VersionAModelSamples, config: CartesianUncertaintyConfig
) -> Array:
    models = model_samples.models
    count = model_samples.sample_valid.shape[0]
    finite = jnp.ones((count,), dtype=bool)
    for leaf in (*jax.tree.leaves(models), model_samples.rotor_efficiency):
        finite = finite & jnp.all(jnp.isfinite(leaf), axis=tuple(range(1, leaf.ndim)))
    symmetric = 0.5 * (models.inertia + jnp.swapaxes(models.inertia, -1, -2))
    inertia_positive = jnp.linalg.eigvalsh(symmetric)[:, 0] > 0
    identity = jnp.eye(3, dtype=models.inertia.dtype)
    inverse_error = jnp.max(jnp.abs(models.inertia @ models.inertia_inv - identity), axis=(-2, -1))
    weights = model_samples.weights
    metadata_valid = (
        jnp.all(jnp.isfinite(weights))
        & jnp.all(weights >= 0)
        & (jnp.abs(jnp.sum(weights) - 1.0) <= config.weight_tolerance)
        & jnp.isfinite(model_samples.retained_variance_fraction)
        & (model_samples.retained_variance_fraction >= 0)
        & (model_samples.retained_variance_fraction <= 1.0 + config.weight_tolerance)
        & (model_samples.model_version >= 0)
    )
    return (
        jnp.asarray(model_samples.sample_valid, dtype=bool)
        & finite
        & (models.mass > 0)
        & inertia_positive
        & (inverse_error <= config.model_tolerance)
        & jnp.all(model_samples.rotor_efficiency > 0, axis=-1)
        & metadata_valid
    )


def _repeat_obstacle_axis(
    scenarios: DynamicSphereScenarioBatch, model_count: int
) -> DynamicSphereScenarioBatch:
    """Repeat each obstacle prediction contiguously across the dynamics axis."""
    return DynamicSphereScenarioBatch(
        obstacle_centers=jnp.repeat(scenarios.obstacle_centers, model_count, axis=1),
        obstacle_radii=jnp.repeat(scenarios.obstacle_radii, model_count, axis=1),
        obstacle_mask=jnp.repeat(scenarios.obstacle_mask, model_count, axis=1),
        arena_lower=scenarios.arena_lower,
        arena_upper=scenarios.arena_upper,
        speed_limit=scenarios.speed_limit,
        angular_rate_max=scenarios.angular_rate_max,
        tilt_max_radians=scenarios.tilt_max_radians,
    )


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


def rollout_shared_quad_dynamic_library_under_uncertainty(
    params: SharedActorParams,
    spec: SharedActorSpec,
    initial_states: Array,
    scenarios: DynamicSphereScenarioBatch,
    controller_model: VersionAModel,
    model_samples: VersionAModelSamples,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    *,
    dt: float,
    policy_gain: float,
    uncertainty_config: CartesianUncertaintyConfig = CartesianUncertaintyConfig(),
) -> UncertainDynamicRolloutBatch:
    """Roll out one shared actor over the full obstacle/dynamics Cartesian product."""
    uncertainty_config.validate()
    actor_config.validate()
    quad_config.validate()
    batch, obstacle_count, model_count, nodes, _ = _validate_shapes(scenarios, model_samples)
    if initial_states.shape != (batch, 13):
        raise ValueError("initial_states must have shape (B,13)")
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(policy_gain) or policy_gain <= 0:
        raise ValueError("dt and policy_gain must be finite and positive")
    horizon = nodes - 1
    policy_count = spec.base_codes.shape[0]
    model_valid = _model_validity(model_samples, uncertainty_config)
    current = jnp.broadcast_to(
        initial_states[:, None, None, :], (batch, obstacle_count, model_count, 13)
    )
    current = jnp.broadcast_to(current[None], (policy_count, *current.shape))
    horizon_duration = horizon * dt

    def advance(state: Array, step_index: Array) -> tuple[Array, tuple[Array, ...]]:
        # Flatten B,R_o,R_m into the actor scenario axis.  Obstacle predictions repeat over R_m;
        # the controller model is common, so no dynamics sample is observable by the actor.
        flat_state = state.reshape((policy_count, batch * obstacle_count * model_count, 13))
        centers = scenarios.obstacle_centers[:, :, step_index]
        radii = scenarios.obstacle_radii[:, :, step_index]
        mask = scenarios.obstacle_mask[:, :, step_index]
        circles = CircleScenarioBatch(
            obstacle_centers=jnp.repeat(centers, model_count, axis=1).reshape(
                (batch * obstacle_count * model_count, centers.shape[-2], 3)
            ),
            obstacle_radii=jnp.repeat(radii, model_count, axis=1).reshape(
                (batch * obstacle_count * model_count, radii.shape[-1])
            ),
            obstacle_mask=jnp.repeat(mask, model_count, axis=1).reshape(
                (batch * obstacle_count * model_count, mask.shape[-1])
            ),
            arena_lower=jnp.repeat(scenarios.arena_lower, obstacle_count * model_count, axis=0),
            arena_upper=jnp.repeat(scenarios.arena_upper, obstacle_count * model_count, axis=0),
            speed_limit=jnp.repeat(scenarios.speed_limit, obstacle_count * model_count, axis=0),
        )
        command = shared_quad_fallback_wrenches(
            params,
            spec,
            flat_state,
            circles,
            controller_model,
            actuator,
            elapsed=step_index * dt,
            horizon_duration=horizon_duration,
            policy_gain=policy_gain,
            actor_config=actor_config,
            quad_config=quad_config,
        )

        def unflatten(value: Array) -> Array:
            return value.reshape(
                (policy_count, batch, obstacle_count, model_count, *value.shape[2:])
            )

        commanded_motor = unflatten(command.bounded_motor_forces)
        commanded_wrench = unflatten(command.wrench)
        command_valid = command.input_valid.reshape(
            (policy_count, batch, obstacle_count, model_count)
        )
        efficiency = model_samples.rotor_efficiency[None, None, None, :, :]
        realized_motor = commanded_motor * efficiency
        realized_wrench = motor_forces_to_wrench(
            realized_motor,
            L=actuator.arm_length,
            thrust2torque=actuator.thrust_to_torque,
            mixing_matrix=actuator.mixing_matrix,
        )

        # Broadcast model leaves explicitly over K,B,R_o and advance the complete Cartesian batch
        # in one call.  A nested ``vmap`` around the rotation exponential inside ``scan`` triggers
        # a CUDA/XLA shape corruption in JAX 0.11.1 (a scalar ``dt`` is lowered as ``[1,1]``).
        # This is mathematically the same sample-wise map and makes the intended
        # ``[K,B,R_o,R_m]`` semantics visible instead of relying on a compiler transform.
        leading = state.shape[:-1]
        models = model_samples.models
        batched_models = models._replace(
            mass=jnp.broadcast_to(models.mass[None, None, None, :], leading),
            gravity_vec=jnp.broadcast_to(models.gravity_vec[None, None, None, :, :], (*leading, 3)),
            inertia=jnp.broadcast_to(models.inertia[None, None, None, :, :, :], (*leading, 3, 3)),
            inertia_inv=jnp.broadcast_to(
                models.inertia_inv[None, None, None, :, :, :], (*leading, 3, 3)
            ),
            drag_matrix=jnp.broadcast_to(
                models.drag_matrix[None, None, None, :, :, :], (*leading, 3, 3)
            ),
            wind_velocity=jnp.broadcast_to(
                models.wind_velocity[None, None, None, :, :], (*leading, 3)
            ),
            external_force=jnp.broadcast_to(
                models.external_force[None, None, None, :, :], (*leading, 3)
            ),
            external_torque=jnp.broadcast_to(
                models.external_torque[None, None, None, :, :], (*leading, 3)
            ),
        )
        following = direct_wrench_symplectic_step(state, realized_wrench, batched_models, dt)
        valid = (
            command_valid
            & model_valid[None, None, None, :]
            & jnp.all(jnp.isfinite(commanded_motor), axis=-1)
            & jnp.all(jnp.isfinite(realized_motor), axis=-1)
            & jnp.all(jnp.isfinite(following), axis=-1)
        )
        following = jnp.where(valid[..., None], following, jnp.nan)
        return following, (
            following,
            commanded_wrench,
            realized_wrench,
            commanded_motor,
            realized_motor,
            valid,
        )

    _, outputs = jax.lax.scan(advance, current, jnp.arange(horizon, dtype=jnp.int32))
    future, commanded_wrench, realized_wrench, commanded_motor, realized_motor, valid = (
        jnp.moveaxis(value, 0, 4) for value in outputs
    )
    states = jnp.concatenate((current[..., None, :], future), axis=4)
    return UncertainDynamicRolloutBatch(
        states,
        commanded_wrench,
        realized_wrench,
        commanded_motor,
        realized_motor,
        valid,
        model_valid,
    )


def evaluate_uncertain_dynamic_quad_library(
    params: SharedActorParams,
    spec: SharedActorSpec,
    initial_states: Array,
    scenarios: DynamicSphereScenarioBatch,
    controller_model: VersionAModel,
    model_samples: VersionAModelSamples,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    barrier_config: VersionABarrierConfig,
    *,
    dt: float,
    policy_gain: float,
    softmin_beta: float,
    uncertainty_config: CartesianUncertaintyConfig = CartesianUncertaintyConfig(),
) -> UncertainDynamicLibraryEvaluation:
    """Return exact hard values reduced over independent obstacle and dynamics samples."""
    rollouts = rollout_shared_quad_dynamic_library_under_uncertainty(
        params,
        spec,
        initial_states,
        scenarios,
        controller_model,
        model_samples,
        actuator,
        actor_config,
        quad_config,
        dt=dt,
        policy_gain=policy_gain,
        uncertainty_config=uncertainty_config,
    )
    policy_count, batch, obstacle_count, model_count, nodes, _ = rollouts.states.shape
    combined_states = rollouts.states.reshape(
        (policy_count, batch, obstacle_count * model_count, nodes, 13)
    )
    combined_scenarios = _repeat_obstacle_axis(scenarios, model_count)
    values = dynamic_quad_safety_values(
        combined_states, combined_scenarios, barrier_config, softmin_beta=softmin_beta
    )
    first_motor_all = rollouts.commanded_motor_forces[..., 0, :]
    first_wrench_all = rollouts.commanded_wrenches[..., 0, :]
    first_motor = first_motor_all[:, :, 0, 0]
    first_wrench = first_wrench_all[:, :, 0, 0]
    motor_error = jnp.max(
        jnp.abs(first_motor_all - first_motor[:, :, None, None, :]), axis=(2, 3, 4)
    )
    motor_scale = jnp.maximum(jnp.max(jnp.abs(first_motor), axis=-1), 1.0)
    consistent = motor_error / motor_scale <= uncertainty_config.first_action_tolerance
    rollout_valid = (
        jnp.all(rollouts.policy_valid, axis=(2, 3, 4))
        & consistent
        & jnp.all(rollouts.model_sample_valid)
    )
    hard = jnp.where(rollout_valid, values.robust_hard_margins, -jnp.inf)
    smooth = jnp.where(rollout_valid, values.robust_smooth_margins, -jnp.inf)
    return UncertainDynamicLibraryEvaluation(
        rollouts, values, hard, smooth, first_motor, first_wrench, consistent, rollout_valid
    )


def _motor_headroom_score(motor_forces: Array, lower: Array, upper: Array) -> Array:
    span = upper - lower
    fraction = jnp.minimum((motor_forces - lower) / span, (upper - motor_forces) / span)
    return jnp.where(
        jnp.all(jnp.isfinite(motor_forces), axis=-1), jnp.min(fraction, axis=-1), jnp.nan
    )


def uncertain_dynamic_discrete_runtime_step(
    state: Array,
    target_position: Array,
    target_velocity: Array,
    previous_policy_index: Array,
    params: SharedActorParams,
    spec: SharedActorSpec,
    prediction_window: DynamicSphereScenarioBatch,
    controller_model: VersionAModel,
    model_samples: VersionAModelSamples,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    barrier_config: VersionABarrierConfig,
    filter_config: DynamicFilterConfig,
    *,
    dt: float,
    policy_gain: float,
    uncertainty_config: CartesianUncertaintyConfig = CartesianUncertaintyConfig(),
) -> UncertainDynamicFilterStep:
    """Apply one sample-independent command with an exact Cartesian robust postcheck."""
    filter_config.validate()
    uncertainty_config.validate()
    if state.shape != (13,) or target_position.shape != (3,) or target_velocity.shape != (3,):
        raise ValueError("state/target shapes must be (13,), (3,), and (3,)")
    if prediction_window.obstacle_centers.shape[0] != 1:
        raise ValueError("runtime prediction window must have B=1")
    if prediction_window.obstacle_centers.shape[2] < 3:
        raise ValueError("prediction window must contain at least three nodes")
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(policy_gain) or policy_gain <= 0:
        raise ValueError("dt and policy_gain must be finite and positive")

    current_window = _slice_nodes(prediction_window, 0, -1)
    next_window = _slice_nodes(prediction_window, 1, None)
    interval_window = _slice_nodes(prediction_window, 0, 2)
    library = evaluate_uncertain_dynamic_quad_library(
        params,
        spec,
        state[None],
        current_window,
        controller_model,
        model_samples,
        actuator,
        actor_config,
        quad_config,
        barrier_config,
        dt=dt,
        policy_gain=policy_gain,
        softmin_beta=filter_config.softmin_beta,
        uncertainty_config=uncertainty_config,
    )
    lower = jnp.broadcast_to(jnp.asarray(actuator.thrust_min, dtype=state.dtype), (4,))
    upper = jnp.broadcast_to(jnp.asarray(actuator.thrust_max, dtype=state.dtype), (4,))
    current_hard_values = library.hard_values[:, 0]
    current_motor_forces = library.first_motor_forces[:, 0]
    scores = _motor_headroom_score(current_motor_forces, lower, upper)
    selection = select_hard_policy(
        current_hard_values, scores, previous_policy_index, filter_config.selection
    )
    selected_params, selected_spec = slice_shared_actor_policy(
        params, spec, selection.selected_index
    )
    # JAX 0.11.1's CUDA scan lowering corrupts a closed scalar ``dt`` only when both the policy
    # and outer batch axes are singleton/non-singleton in this selected-policy shape.  Duplicate
    # the already selected slot into two byte-identical rows; this changes neither its command nor
    # hard value, and the exact postcheck still follows that one selected policy (no policy switch).
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
    fallback_motor = current_motor_forces[selection.selected_index]
    nominal = waypoint_nominal_wrench(
        state, target_position, target_velocity, controller_model, actuator, quad_config
    )
    span = upper - lower

    def sampled_next_states(motor_forces: Array) -> tuple[Array, Array, Array]:
        realized_motor = motor_forces[None, :] * model_samples.rotor_efficiency
        realized_wrench = motor_forces_to_wrench(
            realized_motor,
            L=actuator.arm_length,
            thrust2torque=actuator.thrust_to_torque,
            mixing_matrix=actuator.mixing_matrix,
        )
        states = direct_wrench_symplectic_step(
            jnp.broadcast_to(state, (realized_wrench.shape[0], 13)),
            realized_wrench,
            model_samples.models,
            dt,
        )
        return states, realized_motor, realized_wrench

    def evaluate_action(motor_forces: Array) -> DiscreteActionEvaluation:
        next_states, realized_motor, _ = sampled_next_states(motor_forces)
        # Each possible one-step successor becomes a B branch.  Every branch is then rolled
        # against the complete future R_o x R_m set, giving an explicit R_m(current) x
        # R_o(future) x R_m(future) robust residual.
        branch_count = next_states.shape[0]
        branched_window = DynamicSphereScenarioBatch(
            obstacle_centers=jnp.broadcast_to(
                next_window.obstacle_centers,
                (branch_count, *next_window.obstacle_centers.shape[1:]),
            ),
            obstacle_radii=jnp.broadcast_to(
                next_window.obstacle_radii, (branch_count, *next_window.obstacle_radii.shape[1:])
            ),
            obstacle_mask=jnp.broadcast_to(
                next_window.obstacle_mask, (branch_count, *next_window.obstacle_mask.shape[1:])
            ),
            arena_lower=jnp.broadcast_to(next_window.arena_lower, (branch_count, 3)),
            arena_upper=jnp.broadcast_to(next_window.arena_upper, (branch_count, 3)),
            speed_limit=jnp.broadcast_to(next_window.speed_limit, (branch_count,)),
            angular_rate_max=jnp.broadcast_to(next_window.angular_rate_max, (branch_count,)),
            tilt_max_radians=jnp.broadcast_to(next_window.tilt_max_radians, (branch_count,)),
        )
        next_library = evaluate_uncertain_dynamic_quad_library(
            selected_params,
            selected_spec,
            next_states,
            branched_window,
            controller_model,
            model_samples,
            actuator,
            actor_config,
            quad_config,
            barrier_config,
            dt=dt,
            policy_gain=policy_gain,
            softmin_beta=filter_config.softmin_beta,
            uncertainty_config=uncertainty_config,
        )

        # Exact swept interval across every current dynamics branch and obstacle prediction.
        model_count = next_states.shape[0]
        obstacle_count = interval_window.obstacle_centers.shape[1]
        pair = jnp.stack((jnp.broadcast_to(state, next_states.shape), next_states), axis=1)
        combined_states = jnp.broadcast_to(
            pair[None, None, :, :, :], (1, obstacle_count, model_count, 2, 13)
        )
        combined_states = combined_states.reshape((1, 1, obstacle_count * model_count, 2, 13))
        combined_interval = _repeat_obstacle_axis(interval_window, model_count)
        interval_values = dynamic_quad_safety_values(
            combined_states,
            combined_interval,
            barrier_config,
            softmin_beta=filter_config.softmin_beta,
        )
        reconstructed = wrench_to_motor_forces(
            motor_forces_to_wrench(
                motor_forces,
                L=actuator.arm_length,
                thrust2torque=actuator.thrust_to_torque,
                mixing_matrix=actuator.mixing_matrix,
            ),
            L=actuator.arm_length,
            thrust2torque=actuator.thrust_to_torque,
            mixing_matrix=actuator.mixing_matrix,
        )
        command_bound = jnp.max(jnp.concatenate((lower - motor_forces, motor_forces - upper)))
        effective_lower = model_samples.rotor_efficiency * lower[None, :]
        effective_upper = model_samples.rotor_efficiency * upper[None, :]
        realized_bound = jnp.max(
            jnp.concatenate((effective_lower - realized_motor, realized_motor - effective_upper))
        )
        scale = jnp.maximum(jnp.max(jnp.abs(motor_forces)), 1.0)
        roundtrip = jnp.max(jnp.abs(reconstructed - motor_forces)) / scale
        invalid_sample = jnp.where(
            jnp.all(_model_validity(model_samples, uncertainty_config)), 0.0, 1.0
        )
        actuator_residual = jnp.maximum(
            jnp.maximum(command_bound, realized_bound),
            jnp.maximum(roundtrip - quad_config.allocation_tolerance, invalid_sample),
        )
        return DiscreteActionEvaluation(
            next_value=jnp.min(next_library.hard_values),
            interval_margin=interval_values.robust_hard_margins[0, 0],
            actuator_residual=actuator_residual,
            applied_action=motor_forces,
        )

    filtered = discrete_nonlinear_plcbf_filter(
        nominal.bounded_motor_forces,
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
    evidence = evaluate_action(filtered.action)
    next_states, _, _ = sampled_next_states(filtered.action)
    wrench = motor_forces_to_wrench(
        filtered.action,
        L=actuator.arm_length,
        thrust2torque=actuator.thrust_to_torque,
        mixing_matrix=actuator.mixing_matrix,
    )
    degraded = filtered.degraded | (evidence.interval_margin < -filter_config.tolerance)
    return UncertainDynamicFilterStep(
        filtered.action,
        wrench,
        nominal,
        library,
        selection,
        filtered,
        scores,
        next_states,
        evidence.interval_margin,
        evidence.next_value,
        degraded,
    )


__all__ = [
    "CartesianUncertaintyConfig",
    "UncertainDynamicFilterStep",
    "UncertainDynamicLibraryEvaluation",
    "UncertainDynamicRolloutBatch",
    "evaluate_uncertain_dynamic_quad_library",
    "rollout_shared_quad_dynamic_library_under_uncertainty",
    "uncertain_dynamic_discrete_runtime_step",
]
