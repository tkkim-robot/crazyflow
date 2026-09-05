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
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.struct import dataclass as struct_dataclass
from jax import Array
from jax.nn import softplus

from crazyflow.safety.da_plcbf.bptt import tree_all_finite
from crazyflow.safety.da_plcbf.direct_wrench import quaternion_to_rotation_matrix
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig, acceleration_to_feasible_wrench
from crazyflow.safety.da_plcbf.quad_rollouts import zero_order_hold_rollout

if TYPE_CHECKING:
    from collections.abc import Callable

    from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


_PROPRIOCEPTIVE_FEATURES = 14  # displacement(3), attitude(4), v(3), omega(3), phase(1)
_DESCRIPTOR_SIZE = 9  # final displacement, mean velocity, terminal velocity
_SPATIAL_DESCRIPTOR_SIZE = 3  # only independent displacement coordinates enter diversity


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
    terminal_braking_weight: float = 1.0
    attitude_weight: float = 0.02
    angular_rate_weight: float = 0.01
    covariance_epsilon: float = 1e-3
    pairwise_sigma: float = 1.0
    saturation_temperature: float = 0.05
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_gradient_norm: float = 10.0
    model_compensation: bool = False
    smooth_motor_bounds: bool = True
    initial_residual_scale: float = 0.01
    initial_skill_scale: float = 1.0
    control_interval_steps: int = 1
    velocity_offset_limit: float | None = None
    learn_durations: bool = True
    trainable_parameters: str = "all"
    max_parameter_update_norm: float | None = None
    gate_residual_with_skill_duration: bool = True

    def validate(self) -> None:
        """Reject shapes and scales that invalidate the fixed JAX computation."""
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, Integral):
            raise ValueError("horizon must be a positive integer")
        if self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if (
            isinstance(self.control_interval_steps, bool)
            or not isinstance(self.control_interval_steps, Integral)
            or not 1 <= self.control_interval_steps <= self.horizon
        ):
            raise ValueError("control_interval_steps must be an integer within the horizon")
        if self.velocity_offset_limit is not None and (
            not math.isfinite(self.velocity_offset_limit) or self.velocity_offset_limit <= 0
        ):
            raise ValueError("velocity_offset_limit must be positive finite or None")
        if not isinstance(self.learn_durations, bool):
            raise TypeError("learn_durations must be boolean")
        if not isinstance(self.gate_residual_with_skill_duration, bool):
            raise TypeError("gate_residual_with_skill_duration must be boolean")
        if self.trainable_parameters not in {"all", "network", "offsets"}:
            raise ValueError("trainable_parameters must be all, network, or offsets")
        if self.max_parameter_update_norm is not None and (
            not math.isfinite(self.max_parameter_update_norm) or self.max_parameter_update_norm <= 0
        ):
            raise ValueError("max_parameter_update_norm must be positive finite or None")
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
            self.terminal_braking_weight,
            self.attitude_weight,
            self.angular_rate_weight,
            self.weight_decay,
            self.initial_residual_scale,
        )
        if not all(math.isfinite(value) and value >= 0 for value in nonnegative):
            raise ValueError("actor and objective weights must be finite and nonnegative")
        if len(self.descriptor_scales) != _DESCRIPTOR_SIZE or not all(
            math.isfinite(value) and value > 0 for value in self.descriptor_scales
        ):
            raise ValueError("descriptor_scales must contain nine positive finite values")
        if not isinstance(self.model_compensation, bool) or not isinstance(
            self.smooth_motor_bounds, bool
        ):
            raise TypeError("model_compensation and smooth_motor_bounds must be boolean")
        if (
            not math.isfinite(self.initial_skill_scale)
            or not 0.0 <= self.initial_skill_scale <= 1.0
        ):
            raise ValueError("initial_skill_scale must lie in [0, 1]")


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
    behavior_accelerations: Array | None = None


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
    terminal_braking: Array
    attitude: Array
    angular_rate: Array
    trajectory_tracking: Array | float = 0.0
    velocity_tracking: Array | float = 0.0
    reference_retention: Array | float = 0.0


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
    horizon_duration: float = 1.0,
    dtype: jnp.dtype = jnp.float32,
) -> SkillLibrarySpec:
    """Construct deterministic spherical skills without task or obstacle information."""
    if isinstance(policy_count, bool) or not isinstance(policy_count, Integral) or policy_count < 2:
        raise ValueError("policy_count must be an integer of at least two")
    if isinstance(latent_size, bool) or not isinstance(latent_size, Integral) or latent_size <= 0:
        raise ValueError("latent_size must be a positive integer")
    scales = (minimum_speed, maximum_speed, minimum_duration, maximum_duration, horizon_duration)
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
        (displacement, displacement / horizon_duration, np.zeros_like(desired_velocities)), axis=-1
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
    """Initialize a shared actor with optional suppression of its directional scaffold.

    ``initial_skill_scale=1`` retains the structured directional starting library. Zero starts
    with velocity braking and the small random network residual; obstacle-independent descriptor
    targets and latent identities are retained for subsequent learning.
    """
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
        velocity_offsets=(config.initial_skill_scale - 1.0) * spec.base_desired_velocities,
        duration_offsets=jnp.zeros((policy_count,), dtype=spec.latent_codes.dtype),
        input_kernel=glorot(first_key, input_size, config.hidden_width),
        input_bias=jnp.zeros((config.hidden_width,), dtype=spec.latent_codes.dtype),
        hidden_kernel=glorot(hidden_key, config.hidden_width, config.hidden_width),
        hidden_bias=jnp.zeros((config.hidden_width,), dtype=spec.latent_codes.dtype),
        output_kernel=config.initial_residual_scale * glorot(output_key, config.hidden_width, 3),
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
    *,
    point_model: VersionAModel | None = None,
) -> Array:
    """Evaluate every skill from state, skill-start displacement, latent, and phase only.

    Args:
        params: Current immutable library parameter snapshot.
        spec: Fixed latent codes and motion targets.
        states: Full quadrotor states with shape ``(K, 13)``.
        skill_start_position: Common rollout-start position with shape ``(3,)``.
        phase: Scalar elapsed fraction of the configured horizon.
        config: Static actor and rollout configuration.
        point_model: Optional known dynamics for explicit drag/wind compensation only.

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
    offsets = params.velocity_offsets
    if config.velocity_offset_limit is not None:
        offsets = jnp.clip(offsets, -config.velocity_offset_limit, config.velocity_offset_limit)
    desired_velocity = spec.base_desired_velocities + offsets
    structured = config.policy_gain * (gate[:, None] * desired_velocity - states[:, 7:10])
    residual_gate = gate[:, None] if config.gate_residual_with_skill_duration else 1.0
    raw_action = structured + residual_gate * residual
    action = config.acceleration_limit * jnp.tanh(raw_action / config.acceleration_limit)
    if config.model_compensation:
        if point_model is None:
            raise ValueError("model compensation requires the current point dynamics model")
        # Compensation is outside the behavioral acceleration saturation; the allocator still
        # enforces the same physical motor bounds. This also preserves stationary hover.
        action = action + model_compensation_acceleration(states, point_model)
    input_finite = tree_all_finite((params, spec, states, skill_start_position, phase))
    return jnp.where(input_finite, action, jnp.full_like(action, jnp.nan))


def model_compensation_acceleration(states: Array, point_model: VersionAModel) -> Array:
    """Known force cancellation, separate from the bounded behavioral effort objective."""
    rotation = quaternion_to_rotation_matrix(states[..., 3:7])
    relative_body = (
        jnp.swapaxes(rotation, -1, -2) @ (states[..., 7:10] - point_model.wind_velocity)[..., None]
    )[..., 0]
    drag_body = (point_model.drag_matrix @ relative_body[..., None])[..., 0]
    external_world = (rotation @ drag_body[..., None])[..., 0] + point_model.external_force
    return -external_world / jnp.reshape(point_model.mass, ())


def _trajectory_descriptors(states: Array) -> Array:
    displacement = states[:, -1, :3] - states[:, 0, :3]
    # Symplectic Euler advances p with the following velocity. These samples therefore satisfy
    # displacement == duration * mean_velocity, matching the descriptor target construction.
    mean_velocity = jnp.mean(states[:, 1:, 7:10], axis=1)
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

    def command_at_boundary(state: Array, step_index: Array) -> tuple[Array, ...]:
        phase = step_index / config.horizon
        desired_acceleration = obstacle_agnostic_skill_actions(
            params, spec, state, start_position, phase, config, point_model=point_model
        )
        command = acceleration_to_feasible_wrench(
            desired_acceleration,
            state[:, 3:7],
            state[:, 10:13],
            point_model,
            actuator,
            quad_config,
            smooth_motor_bounds=config.smooth_motor_bounds,
        )
        behavior = desired_acceleration
        if config.model_compensation:
            behavior = behavior - model_compensation_acceleration(state, point_model)
        return (
            command.wrench,
            desired_acceleration,
            command.raw_motor_forces,
            command.bounded_motor_forces,
            command.input_valid,
            behavior,
        )

    future, outputs = zero_order_hold_rollout(
        current,
        command_at_boundary,
        point_model,
        dt=config.dt,
        horizon=config.horizon,
        command_hold_steps=config.control_interval_steps,
    )
    future, wrenches, accelerations, raw_motor, bounded_motor, valid, behavior = (
        jnp.moveaxis(value, 0, 1) for value in (future, *outputs)
    )
    states = jnp.concatenate((current[:, None, :], future), axis=1)
    descriptors = _trajectory_descriptors(states)
    return SkillRollout(
        states, wrenches, accelerations, raw_motor, bounded_motor, valid, descriptors, behavior
    )


def _parameter_distance(params: SkillActorParams, reference: SkillActorParams) -> Array:
    terms = [
        jnp.mean((candidate - anchor) ** 2)
        for candidate, anchor in zip(
            jax.tree.leaves(params), jax.tree.leaves(reference), strict=True
        )
    ]
    return jnp.mean(jnp.stack(terms))


class SpatialDescriptorLosses(NamedTuple):
    """Tracking and diversity losses on three independent displacement coordinates only."""

    target: Array
    diversity: Array
    pairwise: Array


def spatial_descriptor_losses(
    descriptors: Array, targets: Array, config: PersistentSkillConfig
) -> SpatialDescriptorLosses:
    """Evaluate displacement coverage while leaving braking and body motion separate.

    Nine-dimensional diagnostic descriptors remain accepted for saved-trace compatibility, but
    mean and terminal velocity columns never enter these losses. In particular, increasing
    terminal velocity cannot be rewarded as repertoire diversity.
    """
    if descriptors.ndim != 2 or descriptors.shape[0] < 2 or descriptors.shape[1] not in (3, 9):
        raise ValueError("descriptors must have shape (K, 3) or (K, 9), with K >= 2")
    if targets.shape != descriptors.shape:
        raise ValueError("targets must match descriptor shapes")
    scales = jnp.asarray(config.descriptor_scales[:3], dtype=descriptors.dtype)
    spatial = descriptors[:, :3] / scales
    spatial_targets = targets[:, :3] / scales
    target = jnp.mean((spatial - spatial_targets) ** 2)
    centered = spatial - jnp.mean(spatial, axis=0, keepdims=True)
    covariance = centered.T @ centered / spatial.shape[0]
    sign, logdet = jnp.linalg.slogdet(
        covariance
        + config.covariance_epsilon * jnp.eye(_SPATIAL_DESCRIPTOR_SIZE, dtype=spatial.dtype)
    )
    diversity = jnp.where(sign > 0, -logdet, jnp.inf)
    differences = spatial[:, None, :] - spatial[None, :, :]
    squared_distances = jnp.sum(differences**2, axis=-1)
    off_diagonal = 1.0 - jnp.eye(spatial.shape[0], dtype=spatial.dtype)
    pairwise = jnp.sum(
        off_diagonal * jnp.exp(-squared_distances / (config.pairwise_sigma**2))
    ) / jnp.sum(off_diagonal)
    return SpatialDescriptorLosses(target, diversity, pairwise)


def obstacle_agnostic_skill_loss(
    params: SkillActorParams,
    spec: SkillLibrarySpec,
    initial_state: Array,
    point_model: VersionAModel,
    actuator: VersionAActuator,
    previous_params: SkillActorParams,
    config: PersistentSkillConfig,
) -> tuple[Array, SkillLossMetrics]:
    """Learn spatial repertoire coverage with separate obstacle-free motion regularization."""
    rollout = rollout_skill_library(params, spec, initial_state, point_model, actuator, config)
    descriptor_target, diversity, pairwise = spatial_descriptor_losses(
        rollout.descriptors, spec.target_descriptors, config
    )
    terminal_braking = jnp.mean((rollout.states[:, -1, 7:10] / config.velocity_scale) ** 2)
    rotation = quaternion_to_rotation_matrix(rollout.states[:, :, 3:7])
    # 1-cos(tilt) is smooth at upright hover, unlike differentiating arccos at one.
    attitude = jnp.mean(1.0 - rotation[:, :, 2, 2])
    angular_rate = jnp.mean((rollout.states[:, :, 10:13] / config.angular_velocity_scale) ** 2)

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
        + config.terminal_braking_weight * terminal_braking
        + config.attitude_weight * attitude
        + config.angular_rate_weight * angular_rate
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
        terminal_braking,
        attitude,
        angular_rate,
    )


@dataclass(frozen=True, slots=True)
class SkillCompetencyThresholds:
    """Declared nominal-model diagnostics; these never gate learner updates or publication."""

    minimum_displacement_m: float = 0.10
    minimum_direction_cosine: float = 0.8
    minimum_occupied_fraction: float = 0.75
    minimum_aligned_fraction: float = 0.75
    minimum_endpoint_pairwise_mean_m: float = 0.30
    minimum_trajectory_pairwise_rms_mean_m: float = 0.15
    maximum_terminal_speed_mean_mps: float = 0.30
    maximum_terminal_speed_p95_mps: float = 0.50
    maximum_tilt_radians: float = 0.90
    maximum_angular_rate_rps: float = 12.0

    def validate(self) -> None:
        """Reject malformed thresholds rather than silently relabeling a repertoire."""
        if not all(math.isfinite(value) and value > 0.0 for value in asdict(self).values()):
            raise ValueError("competency thresholds must be positive and finite")
        for value in (
            self.minimum_direction_cosine,
            self.minimum_occupied_fraction,
            self.minimum_aligned_fraction,
        ):
            if value > 1.0:
                raise ValueError("direction cosine and occupancy/alignment fractions must be <= 1")
        if self.maximum_tilt_radians > math.pi:
            raise ValueError("maximum_tilt_radians must not exceed pi")


def skill_library_competency(
    rollout: SkillRollout,
    spec: SkillLibrarySpec,
    config: PersistentSkillConfig,
    *,
    thresholds: SkillCompetencyThresholds = SkillCompetencyThresholds(),
) -> dict[str, Any]:
    """Measure actual spatial motions and braking, never latent identities alone.

    Direction bins are nearest target-displacement directions on the unit sphere. A trajectory
    must travel the declared minimum distance before it occupies a bin. Pairwise trajectory
    spread compares synchronized relative positions on the same fixed metric scale. These
    obstacle-independent diagnostic criteria describe a checkpoint; they do not reject updates.
    """
    config.validate()
    thresholds.validate()
    policy_count, _ = _validate_spec(spec)
    states = np.asarray(rollout.states, dtype=np.float64)
    if states.shape != (policy_count, config.horizon + 1, 13):
        raise ValueError("rollout states must match the skill count and configured horizon")
    finite = np.all(np.isfinite(states), axis=(1, 2)) & np.all(
        np.asarray(rollout.policy_valid), axis=1
    )
    relative = states[:, :, :3] - states[:, :1, :3]
    displacement = relative[:, -1]
    norms = np.linalg.norm(displacement, axis=1)
    directions = displacement / np.maximum(norms[:, None], 1e-12)
    targets = np.asarray(spec.target_descriptors[:, :3], dtype=np.float64)
    target_norms = np.linalg.norm(targets, axis=1)
    target_directions = targets / np.maximum(target_norms[:, None], 1e-12)
    similarity = directions @ target_directions.T
    similarity[:, target_norms <= 1e-12] = -np.inf
    active = finite & (norms >= thresholds.minimum_displacement_m)
    if not np.any(target_norms > 1e-12):
        active[:] = False
    assigned = np.where(active, np.argmax(similarity, axis=1), -1)
    counts = np.bincount(assigned[active], minlength=policy_count)
    occupied = int(np.count_nonzero(counts))
    own_cosine = np.sum(directions * target_directions, axis=1)
    aligned = active & (own_cosine >= thresholds.minimum_direction_cosine)
    pair_mask = ~np.eye(policy_count, dtype=bool) & finite[:, None] & finite[None, :]
    endpoint_distances = np.linalg.norm(displacement[:, None] - displacement[None, :], axis=-1)
    trajectory_distances = np.sqrt(
        np.mean(np.sum((relative[:, None] - relative[None, :]) ** 2, axis=-1), axis=-1)
    )
    endpoint_mean = float(np.mean(endpoint_distances[pair_mask])) if np.any(pair_mask) else None
    trajectory_mean = float(np.mean(trajectory_distances[pair_mask])) if np.any(pair_mask) else None
    terminal_speed = np.linalg.norm(states[:, -1, 7:10], axis=1)
    quaternion = states[:, :, 3:7]
    quaternion = quaternion / np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-12)
    body_z_world_z = 1.0 - 2.0 * (quaternion[:, :, 0] ** 2 + quaternion[:, :, 1] ** 2)
    tilt = np.arccos(np.clip(body_z_world_z, -1.0, 1.0))
    rate = np.linalg.norm(states[:, :, 10:13], axis=-1)

    def statistics(values: np.ndarray) -> dict[str, float | None]:
        available = values[np.isfinite(values)]
        if not available.size:
            return {"mean": None, "p95": None, "maximum": None}
        return {
            "mean": float(np.mean(available)),
            "p95": float(np.percentile(available, 95)),
            "maximum": float(np.max(available)),
        }

    speed_stats = statistics(terminal_speed[finite])
    tilt_stats = statistics(tilt[finite])
    rate_stats = statistics(rate[finite])
    checks = {
        "all_rollouts_finite_and_actuator_valid": bool(np.all(finite)),
        "direction_occupancy": occupied / policy_count >= thresholds.minimum_occupied_fraction,
        "intended_direction_alignment": np.count_nonzero(aligned) / policy_count
        >= thresholds.minimum_aligned_fraction,
        "endpoint_spread": endpoint_mean is not None
        and endpoint_mean >= thresholds.minimum_endpoint_pairwise_mean_m,
        "trajectory_spread": trajectory_mean is not None
        and trajectory_mean >= thresholds.minimum_trajectory_pairwise_rms_mean_m,
        "terminal_braking_mean": speed_stats["mean"] is not None
        and speed_stats["mean"] <= thresholds.maximum_terminal_speed_mean_mps,
        "terminal_braking_p95": speed_stats["p95"] is not None
        and speed_stats["p95"] <= thresholds.maximum_terminal_speed_p95_mps,
        "tilt": tilt_stats["maximum"] is not None
        and tilt_stats["maximum"] <= thresholds.maximum_tilt_radians,
        "angular_rate": rate_stats["maximum"] is not None
        and rate_stats["maximum"] <= thresholds.maximum_angular_rate_rps,
    }
    # Native Python values make this diagnostic directly usable in JSON checkpoint metadata.
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "scope": "nominal point-model motion competence; no obstacle or task safety guarantee",
        "diversity_descriptor": "independent final displacement (3D)",
        "policy_count": policy_count,
        "thresholds": asdict(thresholds),
        "competency_checks": checks,
        "competent_under_declared_criteria": all(checks.values()),
        "finite_policy_fraction": float(np.mean(finite)),
        "active_direction_count": int(np.count_nonzero(active)),
        "occupied_direction_count": occupied,
        "occupied_direction_fraction": occupied / policy_count,
        "direction_bin_counts": counts.tolist(),
        "direction_bin_for_skill": assigned.tolist(),
        "aligned_direction_count": int(np.count_nonzero(aligned)),
        "aligned_direction_fraction": float(np.mean(aligned)),
        "own_target_direction_cosines": [float(v) if np.isfinite(v) else None for v in own_cosine],
        "displacement_norm_m": [float(v) if np.isfinite(v) else None for v in norms],
        "endpoint_pairwise_mean_m": endpoint_mean,
        "trajectory_pairwise_rms_mean_m": trajectory_mean,
        "terminal_speed_mps": speed_stats,
        "tilt_radians": tilt_stats,
        "angular_rate_rps": rate_stats,
        "spatial_target_rmse_m": (
            float(np.sqrt(np.mean((displacement[finite] - targets[finite]) ** 2)))
            if np.any(finite)
            else None
        ),
    }


def build_persistent_skill_learner(
    spec: SkillLibrarySpec,
    actuator: VersionAActuator,
    config: PersistentSkillConfig,
    *,
    device: jax.Device | None = None,
    loss_function: Callable[..., tuple[Array, SkillLossMetrics]] | None = None,
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
        iteration: Array | None = None,
    ) -> tuple[Array, SkillLossMetrics]:
        if loss_function is not None:
            return loss_function(
                params,
                initial_state,
                point_model,
                previous_params,
                jnp.asarray(0, dtype=jnp.int32) if iteration is None else iteration,
            )
        return obstacle_agnostic_skill_loss(
            params, spec, initial_state, point_model, actuator, previous_params, config
        )

    def update(
        state: PersistentLearnerState, initial_state: Array, point_model: VersionAModel
    ) -> tuple[PersistentLearnerState, PersistentStepMetrics]:
        def objective(candidate: SkillActorParams) -> tuple[Array, SkillLossMetrics]:
            return loss_bound(
                candidate, initial_state, point_model, state.previous_params, state.library_version
            )

        (_, loss_metrics), gradients = jax.value_and_grad(objective, has_aux=True)(state.params)
        updates, proposed_optimizer_state = optimizer.update(
            _trainable_skill_tree(gradients, config), state.optimizer_state, params=state.params
        )
        updates = _trainable_skill_tree(updates, config)
        if config.max_parameter_update_norm is not None:
            scale = jnp.minimum(
                1.0, config.max_parameter_update_norm / jnp.maximum(optax.tree.norm(updates), 1e-30)
            )
            updates = jax.tree.map(lambda update: scale * update, updates)
        proposed_params = optax.apply_updates(state.params, updates)
        if config.velocity_offset_limit is not None:
            proposed_params = proposed_params.replace(
                velocity_offsets=jnp.clip(
                    proposed_params.velocity_offsets,
                    -config.velocity_offset_limit,
                    config.velocity_offset_limit,
                )
            )
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

    # Inputs are explicitly placed by callers; avoid deprecated jax.jit(device=...).
    if device is not None:
        spec = jax.device_put(spec, device)
        actuator = jax.device_put(actuator, device)
    return PersistentSkillFunctions(
        initialize=initialize,
        rollout=jax.jit(rollout_bound),
        loss=jax.jit(loss_bound),
        step=jax.jit(update),
    )


def _trainable_skill_tree(
    tree: SkillActorParams, config: PersistentSkillConfig
) -> SkillActorParams:
    """Mask gradients and actual updates so frozen coordinates cannot drift through Adam history."""
    values = {}
    for name in SkillActorParams.__dataclass_fields__:
        trainable = True
        if name == "duration_offsets" and not config.learn_durations:
            trainable = False
        if config.trainable_parameters == "network" and name.endswith("_offsets"):
            trainable = False
        if config.trainable_parameters == "offsets" and not name.endswith("_offsets"):
            trainable = False
        value = getattr(tree, name)
        values[name] = value if trainable else jnp.zeros_like(value)
    return SkillActorParams(**values)


__all__ = [
    "PersistentLearnerState",
    "PersistentSkillConfig",
    "PersistentSkillFunctions",
    "PersistentStepMetrics",
    "SkillActorParams",
    "SkillCompetencyThresholds",
    "SkillLibrarySpec",
    "SkillLossMetrics",
    "SkillRollout",
    "SpatialDescriptorLosses",
    "build_fibonacci_skill_spec",
    "build_persistent_skill_learner",
    "initialize_skill_actor",
    "obstacle_agnostic_skill_actions",
    "obstacle_agnostic_skill_loss",
    "rollout_skill_library",
    "skill_library_competency",
    "spatial_descriptor_losses",
]
