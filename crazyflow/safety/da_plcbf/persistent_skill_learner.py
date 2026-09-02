"""Obstacle-agnostic online skill learning through one point dynamics model.

This module is the small corrected learning path used by the mechanism demonstration.  It is
deliberately independent of obstacle geometry, goals, PL-CBF values, candidate snapshots, and
admission gates.  The fallback actor observes only proprioceptive state relative to the start of
the skill, a fixed latent code, and elapsed phase.  Safety is evaluated later by the PL-CBF runtime.

One :class:`PersistentLearnerState` owns both the current parameters and AdamW state for an entire
episode.  Every finite optimizer step becomes the next library version.  NaN/Inf is the only reason
to skip a step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.struct import dataclass as struct_dataclass
from jax import Array
from jax.nn import softplus

from crazyflow.safety.da_plcbf.bptt import tree_all_finite
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig, acceleration_to_feasible_wrench
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step

if TYPE_CHECKING:
    from collections.abc import Callable

    from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


_PROPRIOCEPTIVE_FEATURES = 14  # displacement(3), attitude(4), v(3), omega(3), phase(1)
_DESCRIPTOR_SIZE = 9  # final displacement, mean velocity, terminal velocity


@dataclass(frozen=True, slots=True)
class PersistentSkillConfig:
    """Static actor, rollout, optimizer, and obstacle-free objective settings."""

    dt: float = 0.02
    horizon: int = 50
    hidden_width: int = 32
    policy_gain: float = 1.8
    acceleration_limit: float = 4.0
    residual_scale: float = 1.0
    duration_transition: float = 0.1
    position_scale: float = 1.0
    velocity_scale: float = 1.5
    angular_velocity_scale: float = 8.0
    descriptor_scales: tuple[float, ...] = (1.5, 1.5, 1.0, 1.5, 1.5, 1.0, 1.5, 1.5, 1.0)
    target_weight: float = 1.0
    diversity_weight: float = 0.01
    pairwise_weight: float = 0.05
    action_weight: float = 1e-3
    action_rate_weight: float = 1e-3
    saturation_weight: float = 1e-3
    trust_weight: float = 1e-4
    covariance_epsilon: float = 1e-3
    pairwise_sigma: float = 1.0
    saturation_temperature: float = 0.05
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_gradient_norm: float = 10.0

    def validate(self) -> None:
        """Reject shapes and scales that invalidate the fixed JAX computation."""
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, Integral):
            raise ValueError("horizon must be a positive integer")
        if self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if (
            isinstance(self.hidden_width, bool)
            or not isinstance(self.hidden_width, Integral)
            or self.hidden_width <= 0
        ):
            raise ValueError("hidden_width must be a positive integer")
        positive = (
            self.dt,
            self.policy_gain,
            self.acceleration_limit,
            self.duration_transition,
            self.position_scale,
            self.velocity_scale,
            self.angular_velocity_scale,
            self.covariance_epsilon,
            self.pairwise_sigma,
            self.saturation_temperature,
            self.learning_rate,
            self.max_gradient_norm,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("rollout, normalization, and optimizer scales must be positive finite")
        nonnegative = (
            self.residual_scale,
            self.target_weight,
            self.diversity_weight,
            self.pairwise_weight,
            self.action_weight,
            self.action_rate_weight,
            self.saturation_weight,
            self.trust_weight,
            self.weight_decay,
        )
        if not all(math.isfinite(value) and value >= 0 for value in nonnegative):
            raise ValueError("actor and objective weights must be finite and nonnegative")
        if len(self.descriptor_scales) != _DESCRIPTOR_SIZE or not all(
            math.isfinite(value) and value > 0 for value in self.descriptor_scales
        ):
            raise ValueError("descriptor_scales must contain nine positive finite values")


@struct_dataclass
class SkillLibrarySpec:
    """Fixed latent identities and obstacle-independent behavior targets."""

    latent_codes: Array
    base_desired_velocities: Array
    base_durations: Array
    target_descriptors: Array


@struct_dataclass
class SkillActorParams:
    """Trainable offsets and one shared latent-conditioned residual network."""

    velocity_offsets: Array
    duration_offsets: Array
    input_kernel: Array
    input_bias: Array
    hidden_kernel: Array
    hidden_bias: Array
    output_kernel: Array
    output_bias: Array


@struct_dataclass
class PersistentLearnerState:
    """Parameters and optimizer history that persist for the complete episode."""

    params: SkillActorParams
    previous_params: SkillActorParams
    optimizer_state: optax.OptState
    cumulative_gradient_steps: Array
    latest_dynamics_estimate: VersionAModel
    library_version: Array


class SkillRollout(NamedTuple):
    """Differentiable policy-library rollout under one current point model."""

    states: Array
    wrenches: Array
    desired_accelerations: Array
    raw_motor_forces: Array
    bounded_motor_forces: Array
    policy_valid: Array
    descriptors: Array


class SkillLossMetrics(NamedTuple):
    """Obstacle-free loss decomposition and realized descriptors."""

    total: Array
    descriptor_target: Array
    diversity: Array
    pairwise: Array
    action: Array
    action_rate: Array
    saturation: Array
    trust: Array
    rollout_valid_fraction: Array
    descriptors: Array


class PersistentStepMetrics(NamedTuple):
    """Diagnostics for one finite-guarded persistent optimizer step."""

    loss: SkillLossMetrics
    gradient_norm: Array
    parameter_update_norm: Array
    finite_update_applied: Array
    cumulative_gradient_steps: Array
    library_version: Array


class PersistentSkillFunctions(NamedTuple):
    """Bound, pre-jitted entry points for one obstacle-agnostic skill library."""

    initialize: Callable[[SkillActorParams, VersionAModel], PersistentLearnerState]
    rollout: Callable[[SkillActorParams, Array, VersionAModel], SkillRollout]
    loss: Callable[
        [SkillActorParams, Array, VersionAModel, SkillActorParams], tuple[Array, SkillLossMetrics]
    ]
    step: Callable[
        [PersistentLearnerState, Array, VersionAModel],
        tuple[PersistentLearnerState, PersistentStepMetrics],
    ]


def build_fibonacci_skill_spec(
    *,
    policy_count: int = 16,
    latent_size: int = 8,
    minimum_speed: float = 0.35,
    maximum_speed: float = 1.25,
    minimum_duration: float = 0.35,
    maximum_duration: float = 0.9,
    dtype: jnp.dtype = jnp.float32,
) -> SkillLibrarySpec:
    """Construct deterministic spherical skills without task or obstacle information."""
    if isinstance(policy_count, bool) or not isinstance(policy_count, Integral) or policy_count < 2:
        raise ValueError("policy_count must be an integer of at least two")
    if isinstance(latent_size, bool) or not isinstance(latent_size, Integral) or latent_size <= 0:
        raise ValueError("latent_size must be a positive integer")
    scales = (minimum_speed, maximum_speed, minimum_duration, maximum_duration)
    if not all(math.isfinite(value) and value > 0 for value in scales):
        raise ValueError("speed and duration bounds must be positive finite")
    if minimum_speed > maximum_speed or minimum_duration > maximum_duration:
        raise ValueError("minimum speed/duration must not exceed its maximum")

    index = np.arange(policy_count, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    vertical = 1.0 - 2.0 * (index + 0.5) / policy_count
    radial = np.sqrt(np.maximum(1.0 - vertical**2, 0.0))
    directions = np.stack(
        (radial * np.cos(golden_angle * index), radial * np.sin(golden_angle * index), vertical),
        axis=-1,
    )
    speed_fraction = ((index * 7.0) % policy_count) / max(policy_count - 1, 1)
    duration_fraction = ((index * 11.0) % policy_count) / max(policy_count - 1, 1)
    speeds = minimum_speed + (maximum_speed - minimum_speed) * speed_fraction
    durations = minimum_duration + (maximum_duration - minimum_duration) * duration_fraction
    desired_velocities = directions * speeds[:, None]

    normalized_index = (index + 0.5) / policy_count
    features = []
    for feature_index in range(latent_size):
        frequency = feature_index // 2 + 1
        angle = 2.0 * np.pi * frequency * normalized_index
        features.append(np.sin(angle) if feature_index % 2 == 0 else np.cos(angle))
    latent_codes = np.stack(features, axis=-1)
    displacement = desired_velocities * durations[:, None]
    target_descriptors = np.concatenate(
        (displacement, desired_velocities, np.zeros_like(desired_velocities)), axis=-1
    )
    return SkillLibrarySpec(
        latent_codes=jnp.asarray(latent_codes, dtype=dtype),
        base_desired_velocities=jnp.asarray(desired_velocities, dtype=dtype),
        base_durations=jnp.asarray(durations, dtype=dtype),
        target_descriptors=jnp.asarray(target_descriptors, dtype=dtype),
    )


def _validate_spec(spec: SkillLibrarySpec) -> tuple[int, int]:
    if spec.latent_codes.ndim != 2:
        raise ValueError("latent_codes must have shape (K, Z)")
    policy_count, latent_size = spec.latent_codes.shape
    expected = {
        "base_desired_velocities": (policy_count, 3),
        "base_durations": (policy_count,),
        "target_descriptors": (policy_count, _DESCRIPTOR_SIZE),
    }
    for name, shape in expected.items():
        if getattr(spec, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if policy_count < 2 or latent_size < 1:
        raise ValueError("the skill library needs at least two policies and one latent feature")
    return policy_count, latent_size


def _validate_params(
    params: SkillActorParams, spec: SkillLibrarySpec, config: PersistentSkillConfig
) -> None:
    policy_count, latent_size = _validate_spec(spec)
    input_size = _PROPRIOCEPTIVE_FEATURES + latent_size
    expected = {
        "velocity_offsets": (policy_count, 3),
        "duration_offsets": (policy_count,),
        "input_kernel": (input_size, config.hidden_width),
        "input_bias": (config.hidden_width,),
        "hidden_kernel": (config.hidden_width, config.hidden_width),
        "hidden_bias": (config.hidden_width,),
        "output_kernel": (config.hidden_width, 3),
        "output_bias": (3,),
    }
    for name, shape in expected.items():
        if getattr(params, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")


def initialize_skill_actor(
    key: Array, spec: SkillLibrarySpec, config: PersistentSkillConfig
) -> SkillActorParams:
    """Initialize one deterministic shared actor with small residual output weights."""
    config.validate()
    policy_count, latent_size = _validate_spec(spec)
    input_size = _PROPRIOCEPTIVE_FEATURES + latent_size
    first_key, hidden_key, output_key = jax.random.split(key, 3)

    def glorot(random_key: Array, fan_in: int, fan_out: int) -> Array:
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(
            random_key,
            (fan_in, fan_out),
            dtype=spec.latent_codes.dtype,
            minval=-limit,
            maxval=limit,
        )

    params = SkillActorParams(
        velocity_offsets=jnp.zeros((policy_count, 3), dtype=spec.latent_codes.dtype),
        duration_offsets=jnp.zeros((policy_count,), dtype=spec.latent_codes.dtype),
        input_kernel=glorot(first_key, input_size, config.hidden_width),
        input_bias=jnp.zeros((config.hidden_width,), dtype=spec.latent_codes.dtype),
        hidden_kernel=glorot(hidden_key, config.hidden_width, config.hidden_width),
        hidden_bias=jnp.zeros((config.hidden_width,), dtype=spec.latent_codes.dtype),
        output_kernel=0.01 * glorot(output_key, config.hidden_width, 3),
        output_bias=jnp.zeros((3,), dtype=spec.latent_codes.dtype),
    )
    _validate_params(params, spec, config)
    return params


def _duration_gate(elapsed: Array, duration: Array, transition: float) -> Array:
    coordinate = jnp.clip((duration + transition - elapsed) / (2.0 * transition), 0.0, 1.0)
    return coordinate**2 * (3.0 - 2.0 * coordinate)


def obstacle_agnostic_skill_actions(
    params: SkillActorParams,
    spec: SkillLibrarySpec,
    states: Array,
    skill_start_position: Array,
    phase: Array,
    config: PersistentSkillConfig,
) -> Array:
    """Evaluate every skill from state, skill-start displacement, latent, and phase only.

    Args:
        params: Current immutable library parameter snapshot.
        spec: Fixed latent codes and motion targets.
        states: Full quadrotor states with shape ``(K, 13)``.
        skill_start_position: Common rollout-start position with shape ``(3,)``.
        phase: Scalar elapsed fraction of the configured horizon.
        config: Static actor and rollout configuration.

    This interface intentionally has no goal, waypoint, obstacle, or safety argument.
    """
    if states.ndim != 2 or states.shape[-1] != 13:
        raise ValueError("states must have shape (K, 13)")
    if skill_start_position.shape != (3,) or phase.ndim != 0:
        raise ValueError("skill_start_position and phase must have shapes (3,) and ()")
    _validate_params(params, spec, config)
    if states.shape[0] != spec.latent_codes.shape[0]:
        raise ValueError("state policy axis must match the skill library")

    displacement = (states[:, :3] - skill_start_position) / config.position_scale
    quaternion = states[:, 3:7] / jnp.linalg.norm(states[:, 3:7], axis=-1, keepdims=True)
    velocity = states[:, 7:10] / config.velocity_scale
    angular_velocity = states[:, 10:13] / config.angular_velocity_scale
    phase_feature = jnp.broadcast_to(phase, (states.shape[0], 1))
    network_input = jnp.concatenate(
        (displacement, quaternion, velocity, angular_velocity, phase_feature, spec.latent_codes),
        axis=-1,
    )
    hidden = jnp.tanh(network_input @ params.input_kernel + params.input_bias)
    hidden = jnp.tanh(hidden @ params.hidden_kernel + params.hidden_bias)
    residual = config.residual_scale * jnp.tanh(hidden @ params.output_kernel + params.output_bias)

    elapsed = phase * (config.horizon * config.dt)
    duration = jnp.clip(
        spec.base_durations + params.duration_offsets,
        config.duration_transition,
        config.horizon * config.dt,
    )
    gate = _duration_gate(elapsed, duration, config.duration_transition)
    desired_velocity = spec.base_desired_velocities + params.velocity_offsets
    structured = config.policy_gain * (gate[:, None] * desired_velocity - states[:, 7:10])
    raw_action = structured + gate[:, None] * residual
    action = config.acceleration_limit * jnp.tanh(raw_action / config.acceleration_limit)
    input_finite = tree_all_finite((params, spec, states, skill_start_position, phase))
    return jnp.where(input_finite, action, jnp.full_like(action, jnp.nan))


def _trajectory_descriptors(states: Array) -> Array:
    displacement = states[:, -1, :3] - states[:, 0, :3]
    mean_velocity = jnp.mean(states[:, :, 7:10], axis=1)
    terminal_velocity = states[:, -1, 7:10]
    return jnp.concatenate((displacement, mean_velocity, terminal_velocity), axis=-1)


def rollout_skill_library(
    params: SkillActorParams,
    spec: SkillLibrarySpec,
    initial_state: Array,
    point_model: VersionAModel,
    actuator: VersionAActuator,
    config: PersistentSkillConfig,
) -> SkillRollout:
    """Roll out all skills under exactly one current estimated dynamics model."""
    config.validate()
    _validate_params(params, spec, config)
    if initial_state.shape != (13,):
        raise ValueError("initial_state must have shape (13,)")
    policy_count = spec.latent_codes.shape[0]
    current = jnp.broadcast_to(initial_state, (policy_count, 13))
    start_position = initial_state[:3]
    quad_config = QuadPolicyConfig(acceleration_limit=config.acceleration_limit)

    def advance(state: Array, step_index: Array) -> tuple[Array, tuple[Array, ...]]:
        phase = step_index / config.horizon
        desired_acceleration = obstacle_agnostic_skill_actions(
            params, spec, state, start_position, phase, config
        )
        command = acceleration_to_feasible_wrench(
            desired_acceleration,
            state[:, 3:7],
            state[:, 10:13],
            point_model,
            actuator,
            quad_config,
            smooth_motor_bounds=True,
        )
        following = direct_wrench_symplectic_step(state, command.wrench, point_model, config.dt)
        return following, (
            following,
            command.wrench,
            desired_acceleration,
            command.raw_motor_forces,
            command.bounded_motor_forces,
            command.input_valid,
        )

    _, outputs = jax.lax.scan(
        advance, current, jnp.arange(config.horizon, dtype=initial_state.dtype)
    )
    future, wrenches, accelerations, raw_motor, bounded_motor, valid = (
        jnp.moveaxis(value, 0, 1) for value in outputs
    )
    states = jnp.concatenate((current[:, None, :], future), axis=1)
    descriptors = _trajectory_descriptors(states)
    return SkillRollout(
        states, wrenches, accelerations, raw_motor, bounded_motor, valid, descriptors
    )


def _parameter_distance(params: SkillActorParams, reference: SkillActorParams) -> Array:
    terms = [
        jnp.mean((candidate - anchor) ** 2)
        for candidate, anchor in zip(
            jax.tree.leaves(params), jax.tree.leaves(reference), strict=True
        )
    ]
    return jnp.mean(jnp.stack(terms))


def obstacle_agnostic_skill_loss(
    params: SkillActorParams,
    spec: SkillLibrarySpec,
    initial_state: Array,
    point_model: VersionAModel,
    actuator: VersionAActuator,
    previous_params: SkillActorParams,
    config: PersistentSkillConfig,
) -> tuple[Array, SkillLossMetrics]:
    """Return target, diversity, effort, saturation, and trust losses with no safety term."""
    rollout = rollout_skill_library(params, spec, initial_state, point_model, actuator, config)
    scales = jnp.asarray(config.descriptor_scales, dtype=initial_state.dtype)
    descriptors = rollout.descriptors / scales
    targets = spec.target_descriptors / scales
    descriptor_target = jnp.mean((descriptors - targets) ** 2)

    centered = descriptors - jnp.mean(descriptors, axis=0, keepdims=True)
    covariance = centered.T @ centered / descriptors.shape[0]
    sign, logdet = jnp.linalg.slogdet(
        covariance
        + config.covariance_epsilon * jnp.eye(_DESCRIPTOR_SIZE, dtype=initial_state.dtype)
    )
    diversity = jnp.where(sign > 0, -logdet, jnp.inf)

    differences = descriptors[:, None, :] - descriptors[None, :, :]
    squared_distances = jnp.sum(differences**2, axis=-1)
    off_diagonal = 1.0 - jnp.eye(descriptors.shape[0], dtype=initial_state.dtype)
    pairwise = jnp.sum(
        off_diagonal * jnp.exp(-squared_distances / (config.pairwise_sigma**2))
    ) / jnp.sum(off_diagonal)

    action = jnp.mean((rollout.desired_accelerations / config.acceleration_limit) ** 2)
    action_rate = (
        jnp.mean((jnp.diff(rollout.desired_accelerations, axis=1) / config.acceleration_limit) ** 2)
        if config.horizon > 1
        else jnp.zeros((), dtype=initial_state.dtype)
    )
    lower = jnp.broadcast_to(jnp.asarray(actuator.thrust_min, dtype=initial_state.dtype), (4,))
    upper = jnp.broadcast_to(jnp.asarray(actuator.thrust_max, dtype=initial_state.dtype), (4,))
    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    motor_coordinate = (rollout.raw_motor_forces - center) / half_width
    excess = (
        softplus((jnp.abs(motor_coordinate) - 1.0) / config.saturation_temperature)
        * config.saturation_temperature
    )
    saturation = jnp.mean(excess**2)
    trust = _parameter_distance(params, previous_params)
    total = (
        config.target_weight * descriptor_target
        + config.diversity_weight * diversity
        + config.pairwise_weight * pairwise
        + config.action_weight * action
        + config.action_rate_weight * action_rate
        + config.saturation_weight * saturation
        + config.trust_weight * trust
    )
    rollout_valid = jnp.all(rollout.policy_valid, axis=1) & jnp.all(
        jnp.isfinite(rollout.states), axis=(1, 2)
    )
    finite_inputs = tree_all_finite((spec, point_model, actuator, previous_params))
    total = jnp.where(finite_inputs & jnp.all(rollout_valid) & jnp.isfinite(total), total, jnp.inf)
    return total, SkillLossMetrics(
        total,
        descriptor_target,
        diversity,
        pairwise,
        action,
        action_rate,
        saturation,
        trust,
        jnp.mean(rollout_valid),
        rollout.descriptors,
    )


def build_persistent_skill_learner(
    spec: SkillLibrarySpec,
    actuator: VersionAActuator,
    config: PersistentSkillConfig,
    *,
    device: jax.Device | None = None,
) -> PersistentSkillFunctions:
    """Build jitted rollout/loss/update functions with one persistent AdamW state."""
    config.validate()
    _validate_spec(spec)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
    )

    def initialize(params: SkillActorParams, point_model: VersionAModel) -> PersistentLearnerState:
        _validate_params(params, spec, config)
        return PersistentLearnerState(
            params=params,
            previous_params=params,
            optimizer_state=optimizer.init(params),
            cumulative_gradient_steps=jnp.zeros((), dtype=jnp.int32),
            latest_dynamics_estimate=point_model,
            library_version=jnp.zeros((), dtype=jnp.int32),
        )

    def rollout_bound(
        params: SkillActorParams, initial_state: Array, point_model: VersionAModel
    ) -> SkillRollout:
        return rollout_skill_library(params, spec, initial_state, point_model, actuator, config)

    def loss_bound(
        params: SkillActorParams,
        initial_state: Array,
        point_model: VersionAModel,
        previous_params: SkillActorParams,
    ) -> tuple[Array, SkillLossMetrics]:
        return obstacle_agnostic_skill_loss(
            params, spec, initial_state, point_model, actuator, previous_params, config
        )

    def update(
        state: PersistentLearnerState, initial_state: Array, point_model: VersionAModel
    ) -> tuple[PersistentLearnerState, PersistentStepMetrics]:
        def objective(candidate: SkillActorParams) -> tuple[Array, SkillLossMetrics]:
            return loss_bound(candidate, initial_state, point_model, state.previous_params)

        (_, loss_metrics), gradients = jax.value_and_grad(objective, has_aux=True)(state.params)
        updates, proposed_optimizer_state = optimizer.update(
            gradients, state.optimizer_state, params=state.params
        )
        proposed_params = optax.apply_updates(state.params, updates)
        finite_update = (
            tree_all_finite(loss_metrics)
            & tree_all_finite(gradients)
            & tree_all_finite(proposed_params)
            & tree_all_finite(proposed_optimizer_state)
        )
        params = jax.tree.map(
            lambda proposed, current: jnp.where(finite_update, proposed, current),
            proposed_params,
            state.params,
        )
        optimizer_state = jax.tree.map(
            lambda proposed, current: jnp.where(finite_update, proposed, current),
            proposed_optimizer_state,
            state.optimizer_state,
        )
        increment = finite_update.astype(jnp.int32)
        cumulative_steps = state.cumulative_gradient_steps + increment
        library_version = state.library_version + increment
        model_finite = tree_all_finite(point_model)
        latest_model = jax.tree.map(
            lambda proposed, current: jnp.where(model_finite, proposed, current),
            point_model,
            state.latest_dynamics_estimate,
        )
        parameter_update_norm = optax.tree.norm(
            jax.tree.map(lambda new, old: new - old, params, state.params)
        )
        next_state = state.replace(
            params=params,
            previous_params=jax.tree.map(
                lambda current, previous: jnp.where(finite_update, current, previous),
                state.params,
                state.previous_params,
            ),
            optimizer_state=optimizer_state,
            cumulative_gradient_steps=cumulative_steps,
            latest_dynamics_estimate=latest_model,
            library_version=library_version,
        )
        metrics = PersistentStepMetrics(
            loss=loss_metrics,
            gradient_norm=optax.tree.norm(gradients),
            parameter_update_norm=parameter_update_norm,
            finite_update_applied=finite_update,
            cumulative_gradient_steps=cumulative_steps,
            library_version=library_version,
        )
        return next_state, metrics

    return PersistentSkillFunctions(
        initialize=initialize,
        rollout=jax.jit(rollout_bound, device=device),
        loss=jax.jit(loss_bound, device=device),
        step=jax.jit(update, device=device),
    )


__all__ = [
    "PersistentLearnerState",
    "PersistentSkillConfig",
    "PersistentSkillFunctions",
    "PersistentStepMetrics",
    "SkillActorParams",
    "SkillLibrarySpec",
    "SkillLossMetrics",
    "SkillRollout",
    "build_fibonacci_skill_spec",
    "build_persistent_skill_learner",
    "initialize_skill_actor",
    "obstacle_agnostic_skill_actions",
    "obstacle_agnostic_skill_loss",
    "rollout_skill_library",
]
