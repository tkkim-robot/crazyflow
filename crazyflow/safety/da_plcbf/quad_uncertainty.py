"""Finite-scenario uncertainty rollouts for the Version-A shared quadrotor actor.

This module adds the explicit dynamics-sample axis ``R`` required by DA-PLCBF.  A single
task-agnostic actor and a single controller model are shared by every sample; the sampled plant
parameters are *not* exposed to the policy.  Only the fixed safety-scenario fields are repeated.
Consequently, the adapter cannot accidentally leak a waypoint or another nominal-controller input
through the uncertainty axis.

The hard robust value is the minimum over the supplied finite set of ``R=4`` or ``R=8`` bounded
particles.  It is an empirical finite-scenario certificate only.  It is not a distribution-free,
chance-constrained, or real-world robustness guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.direct_wrench import motor_forces_to_wrench
from crazyflow.safety.da_plcbf.quad_actor_losses import QuadSafetyValues, quad_safety_values
from crazyflow.safety.da_plcbf.quad_policy import shared_quad_fallback_wrenches
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.actor import (
        SharedActorConfig,
        SharedActorParams,
        SharedActorSpec,
    )
    from crazyflow.safety.da_plcbf.estimator import DeterministicParameterSamples, EstimatorConfig
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
    from crazyflow.safety.da_plcbf.version_a_barriers import (
        RigidBodySafetySet,
        VersionABarrierConfig,
    )
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


_PARAMETER_SIZE = 11
_SUPPORTED_SAMPLE_COUNTS = (4, 8)


@dataclass(frozen=True, slots=True)
class QuadUncertaintyConfig:
    """Numerical checks for estimator-to-plant conversion and rollout metadata."""

    parameter_tolerance: float = 3e-5
    model_tolerance: float = 3e-5
    weight_tolerance: float = 2e-6

    def validate(self) -> None:
        """Reject nonfinite or negative audit tolerances."""
        values = (self.parameter_tolerance, self.model_tolerance, self.weight_tolerance)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("quad uncertainty tolerances must be finite and nonnegative")


class VersionAModelSamples(NamedTuple):
    """Stacked Version-A plants and estimator metadata with a leading sample axis.

    Every leaf of ``models`` has shape ``(R, ...)``.  ``rotor_efficiency`` maps commanded motor
    force to realized motor force and has shape ``(R, 4)``.  ``sample_valid`` is deliberately
    per-sample even though the estimator currently emits a single global validity bit.
    """

    models: VersionAModel
    rotor_efficiency: Array
    weights: Array
    sample_valid: Array
    retained_variance_fraction: Array
    model_version: Array


class UncertainQuadRolloutBatch(NamedTuple):
    """Direct-wrench traces with leading axes ``(K, B, R, H, ...)``.

    ``states`` contains the initial node and therefore has ``H + 1`` nodes.  Commanded motor
    forces pass through the common controller's normal actuator map.  Realized motor forces apply
    each plant sample's per-rotor efficiency before conversion to ``realized_wrenches``.
    """

    states: Array
    commanded_wrenches: Array
    realized_wrenches: Array
    desired_accelerations: Array
    raw_motor_forces: Array
    commanded_motor_forces: Array
    realized_motor_forces: Array
    policy_valid: Array
    sample_valid: Array
    weights: Array
    retained_variance_fraction: Array
    model_version: Array


class UncertainQuadSafetyValues(NamedTuple):
    """Exact safety values and finite-scenario robust policy margins."""

    node_values: Array
    node_enabled: Array
    segment_obstacle_values: Array
    segment_obstacle_enabled: Array
    input_valid: Array
    hard_sample_margins: Array
    smooth_sample_margins: Array
    robust_hard_policy_margins: Array
    robust_smooth_policy_margins: Array


def _check_nominal_model_shapes(model: VersionAModel) -> None:
    expected = {
        "mass": (),
        "gravity_vec": (3,),
        "inertia": (3, 3),
        "inertia_inv": (3, 3),
        "drag_matrix": (3, 3),
        "wind_velocity": (3,),
        "external_force": (3,),
        "external_torque": (3,),
    }
    for name, shape in expected.items():
        if jnp.asarray(getattr(model, name)).shape != shape:
            raise ValueError(f"controller_model.{name} must have shape {shape}")


def _check_parameter_sample_shapes(samples: DeterministicParameterSamples) -> int:
    vectors = jnp.asarray(samples.estimation_vectors)
    if vectors.ndim != 2 or vectors.shape[1] != _PARAMETER_SIZE:
        raise ValueError("estimation_vectors must have shape (R, 11)")
    sample_count = vectors.shape[0]
    if sample_count not in _SUPPORTED_SAMPLE_COUNTS:
        raise ValueError("the uncertainty axis must contain exactly R=4 or R=8 samples")
    expected = {
        "mass": (sample_count,),
        "inverse_mass": (sample_count,),
        "drag_acceleration": (sample_count, 3),
        "drag_force_coefficients": (sample_count, 3),
        "drag_matrix": (sample_count, 3, 3),
        "wind_velocity": (sample_count, 3),
        "rotor_efficiency": (sample_count, 4),
    }
    for name, shape in expected.items():
        if jnp.asarray(getattr(samples.parameters, name)).shape != shape:
            raise ValueError(f"samples.parameters.{name} must have shape {shape}")
    if jnp.asarray(samples.weights).shape != (sample_count,):
        raise ValueError("sample weights must have shape (R,)")
    if jnp.asarray(samples.valid).shape != ():
        raise ValueError("samples.valid must be scalar")
    if not jnp.issubdtype(jnp.asarray(samples.valid).dtype, jnp.bool_):
        raise TypeError("samples.valid must have boolean dtype")
    if jnp.asarray(samples.retained_variance_fraction).shape != ():
        raise ValueError("retained_variance_fraction must be scalar")
    version = jnp.asarray(samples.model_version)
    if version.shape != () or not jnp.issubdtype(version.dtype, jnp.integer):
        raise TypeError("model_version must be an integer scalar")
    return sample_count


def _check_stacked_model_shapes(samples: VersionAModelSamples) -> int:
    sample_valid = jnp.asarray(samples.sample_valid)
    if sample_valid.ndim != 1 or sample_valid.shape[0] not in _SUPPORTED_SAMPLE_COUNTS:
        raise ValueError("sample_valid must have shape (4,) or (8,)")
    if not jnp.issubdtype(sample_valid.dtype, jnp.bool_):
        raise TypeError("sample_valid must have boolean dtype")
    sample_count = sample_valid.shape[0]
    expected = {
        "mass": (sample_count,),
        "gravity_vec": (sample_count, 3),
        "inertia": (sample_count, 3, 3),
        "inertia_inv": (sample_count, 3, 3),
        "drag_matrix": (sample_count, 3, 3),
        "wind_velocity": (sample_count, 3),
        "external_force": (sample_count, 3),
        "external_torque": (sample_count, 3),
    }
    for name, shape in expected.items():
        if jnp.asarray(getattr(samples.models, name)).shape != shape:
            raise ValueError(f"model_samples.models.{name} must have shape {shape}")
    if jnp.asarray(samples.rotor_efficiency).shape != (sample_count, 4):
        raise ValueError("model_samples.rotor_efficiency must have shape (R, 4)")
    if jnp.asarray(samples.weights).shape != (sample_count,):
        raise ValueError("model_samples.weights must have shape (R,)")
    if jnp.asarray(samples.retained_variance_fraction).shape != ():
        raise ValueError("retained_variance_fraction must be scalar")
    version = jnp.asarray(samples.model_version)
    if version.shape != () or not jnp.issubdtype(version.dtype, jnp.integer):
        raise TypeError("model_version must be an integer scalar")
    return sample_count


def _circle_scenario_shapes(scenarios: CircleScenarioBatch) -> tuple[int, int, int]:
    centers = jnp.asarray(scenarios.obstacle_centers)
    if centers.ndim != 3:
        raise ValueError("obstacle_centers must have shape (B, O, D)")
    batch_size, obstacle_count, dimension = centers.shape
    if batch_size < 1 or dimension < 1:
        raise ValueError("scenario batch and spatial dimension must be positive")
    expected = {
        "obstacle_radii": (batch_size, obstacle_count),
        "obstacle_mask": (batch_size, obstacle_count),
        "arena_lower": (batch_size, dimension),
        "arena_upper": (batch_size, dimension),
        "speed_limit": (batch_size,),
    }
    for name, shape in expected.items():
        if jnp.asarray(getattr(scenarios, name)).shape != shape:
            raise ValueError(f"scenarios.{name} must have shape {shape}")
    if not jnp.issubdtype(jnp.asarray(scenarios.obstacle_mask).dtype, jnp.bool_):
        raise TypeError("scenarios.obstacle_mask must have boolean dtype")
    return batch_size, obstacle_count, dimension


def _relative_error(actual: Array, expected: Array) -> Array:
    scale = jnp.maximum(jnp.maximum(jnp.abs(actual), jnp.abs(expected)), 1.0)
    return jnp.max(jnp.abs(actual - expected) / scale, axis=tuple(range(1, actual.ndim)))


def version_a_model_samples_from_estimator(
    samples: DeterministicParameterSamples,
    controller_model: VersionAModel,
    estimator_config: EstimatorConfig,
    *,
    uncertainty_config: QuadUncertaintyConfig = QuadUncertaintyConfig(),
) -> VersionAModelSamples:
    """Convert bounded estimator particles into stacked Version-A plant samples.

    The fixed gravity, inertia, and external loads come from ``controller_model`` because the
    current low-dimensional estimator does not estimate them.  Numerical invalidity is represented
    by ``sample_valid=False`` instead of raising inside a jitted execution path.  Static shape and
    dtype errors are rejected eagerly.
    """
    estimator_config.validate()
    uncertainty_config.validate()
    _check_nominal_model_shapes(controller_model)
    sample_count = _check_parameter_sample_shapes(samples)
    parameters = samples.parameters
    dtype = jnp.asarray(parameters.mass).dtype

    def repeated(value: Array) -> Array:
        array = jnp.asarray(value, dtype=dtype)
        return jnp.broadcast_to(array, (sample_count, *array.shape))

    models = VersionAModel(
        mass=jnp.asarray(parameters.mass, dtype=dtype),
        gravity_vec=repeated(controller_model.gravity_vec),
        inertia=repeated(controller_model.inertia),
        inertia_inv=repeated(controller_model.inertia_inv),
        drag_matrix=jnp.asarray(parameters.drag_matrix, dtype=dtype),
        wind_velocity=jnp.asarray(parameters.wind_velocity, dtype=dtype),
        external_force=repeated(controller_model.external_force),
        external_torque=repeated(controller_model.external_torque),
    )

    vector = jnp.asarray(samples.estimation_vectors, dtype=dtype)
    reconstructed_vector = jnp.concatenate(
        (
            parameters.inverse_mass[:, None],
            parameters.drag_acceleration,
            parameters.wind_velocity,
            parameters.rotor_efficiency,
        ),
        axis=-1,
    )
    mass_lower, mass_upper = estimator_config.mass_bounds
    drag_lower, drag_upper = estimator_config.drag_acceleration_bounds
    lower = jnp.asarray(
        [
            1.0 / mass_upper,
            drag_lower,
            drag_lower,
            drag_lower,
            *estimator_config.wind_lower,
            *([estimator_config.efficiency_bounds[0]] * 4),
        ],
        dtype=dtype,
    )
    upper = jnp.asarray(
        [
            1.0 / mass_lower,
            drag_upper,
            drag_upper,
            drag_upper,
            *estimator_config.wind_upper,
            *([estimator_config.efficiency_bounds[1]] * 4),
        ],
        dtype=dtype,
    )
    tolerance = uncertainty_config.parameter_tolerance
    in_bounds = jnp.all((vector >= lower - tolerance) & (vector <= upper + tolerance), axis=-1)
    vector_consistent = _relative_error(vector, reconstructed_vector) <= tolerance
    inverse_consistent = jnp.abs(parameters.mass * parameters.inverse_mass - 1.0) <= tolerance
    drag_force_consistent = jnp.all(
        jnp.abs(
            parameters.drag_force_coefficients * parameters.inverse_mass[:, None]
            - parameters.drag_acceleration
        )
        <= tolerance * jnp.maximum(1.0, jnp.abs(parameters.drag_acceleration)),
        axis=-1,
    )
    expected_drag = -jnp.eye(3, dtype=dtype) * parameters.drag_force_coefficients[:, None, :]
    drag_matrix_consistent = _relative_error(parameters.drag_matrix, expected_drag) <= tolerance

    parameter_leaves = jax.tree.leaves(parameters)
    parameter_finite = jnp.all(
        jnp.stack(
            [
                jnp.all(jnp.isfinite(leaf), axis=tuple(range(1, leaf.ndim)))
                if leaf.ndim > 1
                else jnp.isfinite(leaf)
                for leaf in parameter_leaves
            ]
        ),
        axis=0,
    )
    weights = jnp.asarray(samples.weights, dtype=dtype)
    weights_valid = (
        jnp.all(jnp.isfinite(weights))
        & jnp.all(weights >= 0)
        & (jnp.abs(jnp.sum(weights) - 1.0) <= uncertainty_config.weight_tolerance)
    )
    retained = jnp.asarray(samples.retained_variance_fraction, dtype=dtype)
    metadata_valid = (
        jnp.asarray(samples.valid)
        & weights_valid
        & jnp.isfinite(retained)
        & (retained >= 0)
        & (retained <= 1.0 + uncertainty_config.parameter_tolerance)
        & (jnp.asarray(samples.model_version) >= 0)
    )
    sample_valid = (
        metadata_valid
        & parameter_finite
        & in_bounds
        & vector_consistent
        & inverse_consistent
        & drag_force_consistent
        & drag_matrix_consistent
    )
    return VersionAModelSamples(
        models=models,
        rotor_efficiency=jnp.asarray(parameters.rotor_efficiency, dtype=dtype),
        weights=weights,
        sample_valid=sample_valid,
        retained_variance_fraction=retained,
        model_version=jnp.asarray(samples.model_version),
    )


def duplicate_circle_scenarios_for_samples(
    scenarios: CircleScenarioBatch, sample_count: int
) -> CircleScenarioBatch:
    """Repeat only task-agnostic safety fields in ``B``-major, ``R``-minor order.

    ``CircleScenarioBatch`` intentionally has no goal, waypoint, reward, or nominal-controller
    field.  This explicit constructor makes that non-leakage boundary visible in code review.
    """
    _circle_scenario_shapes(scenarios)
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, Integral)
        or sample_count not in _SUPPORTED_SAMPLE_COUNTS
    ):
        raise ValueError("sample_count must be exactly 4 or 8")

    def repeat(value: Array) -> Array:
        return jnp.repeat(value, sample_count, axis=0)

    return CircleScenarioBatch(
        obstacle_centers=repeat(scenarios.obstacle_centers),
        obstacle_radii=repeat(scenarios.obstacle_radii),
        obstacle_mask=repeat(scenarios.obstacle_mask),
        arena_lower=repeat(scenarios.arena_lower),
        arena_upper=repeat(scenarios.arena_upper),
        speed_limit=repeat(scenarios.speed_limit),
    )


def _device_model_validity(
    samples: VersionAModelSamples, controller_model: VersionAModel, config: QuadUncertaintyConfig
) -> Array:
    models = samples.models
    sample_count = samples.sample_valid.shape[0]
    identity = jnp.eye(3, dtype=models.inertia.dtype)
    finite = jnp.ones((sample_count,), dtype=bool)
    for leaf in (*jax.tree.leaves(models), samples.rotor_efficiency):
        axes = tuple(range(1, leaf.ndim))
        finite = finite & jnp.all(jnp.isfinite(leaf), axis=axes)
    symmetric_inertia = 0.5 * (models.inertia + jnp.swapaxes(models.inertia, -1, -2))
    inertia_positive = jnp.linalg.eigvalsh(symmetric_inertia)[:, 0] > 0
    inverse_error = jnp.max(jnp.abs(models.inertia @ models.inertia_inv - identity), axis=(-2, -1))
    weights = samples.weights
    metadata_valid = (
        jnp.all(jnp.isfinite(weights))
        & jnp.all(weights >= 0)
        & (jnp.abs(jnp.sum(weights) - 1.0) <= config.weight_tolerance)
        & jnp.isfinite(samples.retained_variance_fraction)
        & (samples.retained_variance_fraction >= 0)
        & (samples.retained_variance_fraction <= 1.0 + config.parameter_tolerance)
        & (samples.model_version >= 0)
    )
    controller_finite = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(controller_model)])
    )
    return (
        samples.sample_valid
        & finite
        & (models.mass > 0)
        & inertia_positive
        & (inverse_error <= config.model_tolerance)
        & jnp.all(samples.rotor_efficiency > 0, axis=-1)
        & metadata_valid
        & controller_finite
    )


def _safe_models(
    samples: VersionAModelSamples, controller_model: VersionAModel, valid: Array
) -> VersionAModel:
    sample_count = valid.shape[0]

    def safe(sampled: Array, nominal: Array) -> Array:
        nominal_array = jnp.asarray(nominal, dtype=sampled.dtype)
        replacement = jnp.broadcast_to(nominal_array, (sample_count, *nominal_array.shape))
        mask = valid.reshape((sample_count, *(1 for _ in nominal_array.shape)))
        return jnp.where(mask, sampled, replacement)

    return jax.tree.map(safe, samples.models, controller_model)


def rollout_shared_quad_library_under_uncertainty(
    params: SharedActorParams,
    spec: SharedActorSpec,
    initial_states: Array,
    scenarios: CircleScenarioBatch,
    controller_model: VersionAModel,
    model_samples: VersionAModelSamples,
    actuator: VersionAActuator,
    *,
    dt: float,
    horizon: int,
    policy_gain: float,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    uncertainty_config: QuadUncertaintyConfig = QuadUncertaintyConfig(),
) -> UncertainQuadRolloutBatch:
    """Roll out one shared actor against every finite dynamics sample.

    The controller model and actuator are common across ``R``.  Each sample affects only realized
    motor force and plant evolution, so neither the actor nor its geometric wrench map receives the
    sample index or true sampled parameters.
    """
    uncertainty_config.validate()
    actor_config.validate()
    quad_config.validate()
    _check_nominal_model_shapes(controller_model)
    sample_count = _check_stacked_model_shapes(model_samples)
    batch_size, _, dimension = _circle_scenario_shapes(scenarios)
    if dimension != 3:
        raise ValueError("quad uncertainty scenarios must be three-dimensional")
    if initial_states.shape != (batch_size, 13):
        raise ValueError("initial_states must have shape (B, 13)")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not math.isfinite(policy_gain) or policy_gain <= 0:
        raise ValueError("policy_gain must be finite and positive")
    if spec.base_codes.ndim != 2 or spec.base_codes.shape[0] < 1:
        raise ValueError("spec.base_codes must have shape (positive K, Z)")

    policy_count = spec.base_codes.shape[0]
    sample_valid = _device_model_validity(model_samples, controller_model, uncertainty_config)
    safe_models = _safe_models(model_samples, controller_model, sample_valid)
    safe_efficiency = jnp.where(
        sample_valid[:, None], model_samples.rotor_efficiency, jnp.ones((sample_count, 4))
    )
    duplicated_scenarios = duplicate_circle_scenarios_for_samples(scenarios, sample_count)
    current = jnp.broadcast_to(
        initial_states[None, :, None, :], (policy_count, batch_size, sample_count, 13)
    )
    horizon_duration = horizon * dt

    def advance(state: Array, step_index: Array) -> tuple[Array, tuple[Array, ...]]:
        flattened_state = state.reshape((policy_count, batch_size * sample_count, 13))
        command = shared_quad_fallback_wrenches(
            params,
            spec,
            flattened_state,
            duplicated_scenarios,
            controller_model,
            actuator,
            elapsed=step_index * dt,
            horizon_duration=horizon_duration,
            policy_gain=policy_gain,
            actor_config=actor_config,
            quad_config=quad_config,
        )

        def unflatten(value: Array) -> Array:
            return value.reshape((policy_count, batch_size, sample_count, *value.shape[2:]))

        commanded_wrench = unflatten(command.wrench)
        desired_acceleration = unflatten(command.desired_acceleration)
        raw_motor = unflatten(command.raw_motor_forces)
        commanded_motor = unflatten(command.bounded_motor_forces)
        command_valid = command.input_valid.reshape((policy_count, batch_size, sample_count))
        realized_motor = commanded_motor * safe_efficiency[None, None, :, :]
        realized_wrench = motor_forces_to_wrench(
            realized_motor,
            L=actuator.arm_length,
            thrust2torque=actuator.thrust_to_torque,
            mixing_matrix=actuator.mixing_matrix,
        )

        def step_one(sample_state: Array, sample_wrench: Array, model: VersionAModel) -> Array:
            return direct_wrench_symplectic_step(sample_state, sample_wrench, model, dt)

        following = jax.vmap(step_one, in_axes=(2, 2, 0), out_axes=2)(
            state, realized_wrench, safe_models
        )
        finite_step = (
            jnp.all(jnp.isfinite(state), axis=-1)
            & jnp.all(jnp.isfinite(commanded_wrench), axis=-1)
            & jnp.all(jnp.isfinite(realized_motor), axis=-1)
            & jnp.all(jnp.isfinite(realized_wrench), axis=-1)
            & jnp.all(jnp.isfinite(following), axis=-1)
        )
        step_valid = command_valid & sample_valid[None, None, :] & finite_step
        following = jnp.where(step_valid[..., None], following, jnp.nan)
        return following, (
            following,
            commanded_wrench,
            realized_wrench,
            desired_acceleration,
            raw_motor,
            commanded_motor,
            realized_motor,
            step_valid,
        )

    _, outputs = jax.lax.scan(advance, current, jnp.arange(horizon, dtype=current.dtype))
    (
        future,
        commanded_wrench,
        realized_wrench,
        desired_acceleration,
        raw_motor,
        commanded_motor,
        realized_motor,
        policy_valid,
    ) = (jnp.moveaxis(value, 0, 3) for value in outputs)
    states = jnp.concatenate((current[:, :, :, None, :], future), axis=3)
    return UncertainQuadRolloutBatch(
        states=states,
        commanded_wrenches=commanded_wrench,
        realized_wrenches=realized_wrench,
        desired_accelerations=desired_acceleration,
        raw_motor_forces=raw_motor,
        commanded_motor_forces=commanded_motor,
        realized_motor_forces=realized_motor,
        policy_valid=policy_valid,
        sample_valid=sample_valid,
        weights=model_samples.weights,
        retained_variance_fraction=model_samples.retained_variance_fraction,
        model_version=model_samples.model_version,
    )


def uncertain_quad_safety_values(
    rollouts: UncertainQuadRolloutBatch,
    safety: RigidBodySafetySet,
    barrier_config: VersionABarrierConfig,
    *,
    softmin_beta: float,
) -> UncertainQuadSafetyValues:
    """Evaluate exact values and take the hard policy margin's minimum over ``R``.

    Sample weights do not weaken the robust minimum: every supplied finite sample must pass.  Any
    invalid model, command, state, or enabled safety value makes that sample's margin ``-inf`` and
    therefore fails the robust certificate closed.
    """
    states = jnp.asarray(rollouts.states)
    if states.ndim != 5 or states.shape[-1] != 13 or states.shape[3] < 2:
        raise ValueError("rollout states must have shape (K, B, R, at_least_two_nodes, 13)")
    policy_count, batch_size, sample_count, node_count, _ = states.shape
    if sample_count not in _SUPPORTED_SAMPLE_COUNTS:
        raise ValueError("rollout uncertainty axis must have size 4 or 8")
    horizon = node_count - 1
    expected = {
        "commanded_wrenches": (policy_count, batch_size, sample_count, horizon, 4),
        "realized_wrenches": (policy_count, batch_size, sample_count, horizon, 4),
        "desired_accelerations": (policy_count, batch_size, sample_count, horizon, 3),
        "raw_motor_forces": (policy_count, batch_size, sample_count, horizon, 4),
        "commanded_motor_forces": (policy_count, batch_size, sample_count, horizon, 4),
        "realized_motor_forces": (policy_count, batch_size, sample_count, horizon, 4),
        "policy_valid": (policy_count, batch_size, sample_count, horizon),
    }
    for name, shape in expected.items():
        if jnp.asarray(getattr(rollouts, name)).shape != shape:
            raise ValueError(f"rollouts.{name} must have shape {shape}")
    if not jnp.issubdtype(jnp.asarray(rollouts.policy_valid).dtype, jnp.bool_):
        raise TypeError("rollouts.policy_valid must have boolean dtype")
    if jnp.asarray(rollouts.sample_valid).shape != (sample_count,):
        raise ValueError("rollouts.sample_valid must have shape (R,)")
    if not jnp.issubdtype(jnp.asarray(rollouts.sample_valid).dtype, jnp.bool_):
        raise TypeError("rollouts.sample_valid must have boolean dtype")

    def evaluate_sample(sample_states: Array) -> QuadSafetyValues:
        return quad_safety_values(sample_states, safety, barrier_config, softmin_beta=softmin_beta)

    values = jax.vmap(evaluate_sample, in_axes=2, out_axes=2)(states)
    rollout_valid = (
        values.input_valid
        & jnp.all(rollouts.policy_valid, axis=-1)
        & rollouts.sample_valid[None, None, :]
    )
    hard = jnp.where(
        rollout_valid & jnp.isfinite(values.hard_policy_margins),
        values.hard_policy_margins,
        -jnp.inf,
    )
    smooth = jnp.where(
        rollout_valid & jnp.isfinite(values.smooth_policy_margins),
        values.smooth_policy_margins,
        -jnp.inf,
    )
    return UncertainQuadSafetyValues(
        node_values=values.node_values,
        node_enabled=values.node_enabled,
        segment_obstacle_values=values.segment_obstacle_values,
        segment_obstacle_enabled=values.segment_obstacle_enabled,
        input_valid=rollout_valid,
        hard_sample_margins=hard,
        smooth_sample_margins=smooth,
        robust_hard_policy_margins=jnp.min(hard, axis=2),
        robust_smooth_policy_margins=jnp.min(smooth, axis=2),
    )


__all__ = [
    "QuadUncertaintyConfig",
    "UncertainQuadRolloutBatch",
    "UncertainQuadSafetyValues",
    "VersionAModelSamples",
    "duplicate_circle_scenarios_for_samples",
    "rollout_shared_quad_library_under_uncertainty",
    "uncertain_quad_safety_values",
    "version_a_model_samples_from_estimator",
]
