"""Task-agnostic shared fallback actor with immutable structural seed policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax.struct import dataclass as struct_dataclass
from jax import Array

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch


@dataclass(frozen=True)
class SharedActorConfig:
    """Static architecture and duration settings for the shared residual actor."""

    hidden_width: int = 32
    residual_scale: float = 0.5
    min_duration: float = 0.2
    max_duration: float = 2.0
    duration_transition: float = 0.1

    def validate(self) -> None:
        """Validate architecture sizes and physical scales."""
        if (
            isinstance(self.hidden_width, bool)
            or not isinstance(self.hidden_width, Integral)
            or self.hidden_width <= 0
        ):
            raise ValueError("hidden_width must be a positive integer")
        if not math.isfinite(self.residual_scale) or self.residual_scale < 0:
            raise ValueError("residual_scale must be finite and nonnegative")
        if (
            not math.isfinite(self.min_duration)
            or not math.isfinite(self.max_duration)
            or self.min_duration < 0
            or self.max_duration <= self.min_duration
        ):
            raise ValueError(
                "duration bounds must be finite and satisfy 0 <= min_duration < max_duration"
            )
        if not math.isfinite(self.duration_transition) or self.duration_transition <= 0:
            raise ValueError("duration_transition must be finite and positive")


@struct_dataclass
class SharedActorSpec:
    """Fixed library identity, structural core, and base skill parameters.

    ``adaptive_mask=False`` marks structural policies. Their code, desired velocity, duration, and
    residual contribution are unaffected by every trainable parameter in :class:`SharedActorParams`.
    """

    base_codes: Array
    base_desired_velocities: Array
    base_durations: Array
    adaptive_mask: Array


@struct_dataclass
class SharedActorParams:
    """Candidate-trainable code/skill offsets and one residual MLP shared by all adaptive slots."""

    code_offsets: Array
    velocity_offsets: Array
    duration_offsets: Array
    input_kernel: Array
    input_bias: Array
    hidden_kernel: Array
    hidden_bias: Array
    output_kernel: Array
    output_bias: Array


def actor_observation_size(dimension: int, n_obstacles: int) -> int:
    """Return the fixed task-agnostic observation width.

    Features are normalized position, velocity, obstacle-relative position/radius/mask, arena
    clearance, and common-horizon phase. No nominal goal or task-controller state is present.
    """
    if dimension <= 0 or n_obstacles < 0:
        raise ValueError("dimension must be positive and n_obstacles nonnegative")
    return 4 * dimension + n_obstacles * (dimension + 2) + 1


def validate_shared_actor_shapes(
    params: SharedActorParams, spec: SharedActorSpec, *, dimension: int, n_obstacles: int
) -> None:
    """Reject inconsistent library and network shapes before tracing expensive rollouts."""
    if spec.base_codes.ndim != 2:
        raise ValueError("base_codes must have shape (K, Z)")
    n_policies, code_size = spec.base_codes.shape
    if spec.base_desired_velocities.shape != (n_policies, dimension):
        raise ValueError("base_desired_velocities must have shape (K, D)")
    if spec.base_durations.shape != (n_policies,):
        raise ValueError("base_durations must have shape (K,)")
    if spec.adaptive_mask.shape != (n_policies,):
        raise ValueError("adaptive_mask must have shape (K,)")
    if not jnp.issubdtype(spec.adaptive_mask.dtype, jnp.bool_):
        raise ValueError("adaptive_mask must have boolean dtype")
    if params.code_offsets.shape != spec.base_codes.shape:
        raise ValueError("code_offsets must match base_codes")
    if params.velocity_offsets.shape != spec.base_desired_velocities.shape:
        raise ValueError("velocity_offsets must match base_desired_velocities")
    if params.duration_offsets.shape != spec.base_durations.shape:
        raise ValueError("duration_offsets must match base_durations")
    observation_size = actor_observation_size(dimension, n_obstacles)
    hidden_width = params.input_bias.shape[0]
    expected = {
        "input_kernel": (observation_size + code_size, hidden_width),
        "hidden_kernel": (hidden_width, hidden_width),
        "hidden_bias": (hidden_width,),
        "output_kernel": (hidden_width, dimension),
        "output_bias": (dimension,),
    }
    for name, shape in expected.items():
        if getattr(params, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")


def initialize_shared_actor(
    key: Array,
    spec: SharedActorSpec,
    *,
    dimension: int,
    n_obstacles: int,
    config: SharedActorConfig,
) -> SharedActorParams:
    """Initialize a shared actor whose residual is exactly zero at creation."""
    config.validate()
    if spec.base_codes.ndim != 2:
        raise ValueError("base_codes must have shape (K, Z)")
    n_policies, code_size = spec.base_codes.shape
    observation_size = actor_observation_size(dimension, n_obstacles)
    first_key, second_key = jax.random.split(key)

    def glorot(random_key: Array, fan_in: int, fan_out: int) -> Array:
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(
            random_key, (fan_in, fan_out), dtype=spec.base_codes.dtype, minval=-limit, maxval=limit
        )

    params = SharedActorParams(
        code_offsets=jnp.zeros_like(spec.base_codes),
        velocity_offsets=jnp.zeros_like(spec.base_desired_velocities),
        duration_offsets=jnp.zeros_like(spec.base_durations),
        input_kernel=glorot(first_key, observation_size + code_size, config.hidden_width),
        input_bias=jnp.zeros((config.hidden_width,), dtype=spec.base_codes.dtype),
        hidden_kernel=glorot(second_key, config.hidden_width, config.hidden_width),
        hidden_bias=jnp.zeros((config.hidden_width,), dtype=spec.base_codes.dtype),
        output_kernel=jnp.zeros((config.hidden_width, dimension), dtype=spec.base_codes.dtype),
        output_bias=jnp.zeros((dimension,), dtype=spec.base_codes.dtype),
    )
    validate_shared_actor_shapes(params, spec, dimension=dimension, n_obstacles=n_obstacles)
    if n_policies == 0:
        raise ValueError("the policy library must not be empty")
    return params


def _task_agnostic_observation(
    states: Array, scenarios: CircleScenarioBatch, phase: Array
) -> Array:
    """Build normalized fixed-size observations for states shaped ``(K, B, 2D)``."""
    position, velocity = jnp.split(states, 2, axis=-1)
    mask = scenarios.obstacle_mask
    centers = jnp.where(mask[..., None], scenarios.obstacle_centers, 0.0)
    radii = jnp.where(mask, scenarios.obstacle_radii, 0.0)
    arena_span = scenarios.arena_upper - scenarios.arena_lower
    valid_span = arena_span > 0
    span = jnp.where(valid_span, arena_span, 1.0)
    position_normalized = (
        2.0 * (position - scenarios.arena_lower[None, :, :]) / span[None, :, :] - 1.0
    )
    position_normalized = jnp.where(valid_span[None, :, :], position_normalized, jnp.nan)
    valid_speed = scenarios.speed_limit > 0
    speed = jnp.where(valid_speed, scenarios.speed_limit, 1.0)
    velocity_normalized = velocity / speed[None, :, None]
    velocity_normalized = jnp.where(valid_speed[None, :, None], velocity_normalized, jnp.nan)

    obstacle_span = jnp.mean(span, axis=-1)
    relative = (centers[None, :, :, :] - position[:, :, None, :]) / obstacle_span[
        None, :, None, None
    ]
    radius_normalized = radii / obstacle_span[:, None]
    obstacle_features = jnp.concatenate(
        (
            relative,
            jnp.broadcast_to(radius_normalized[None, :, :, None], (*relative.shape[:-1], 1)),
            jnp.broadcast_to(mask[None, :, :, None], (*relative.shape[:-1], 1)),
        ),
        axis=-1,
    )
    obstacle_features = obstacle_features.reshape(states.shape[0], states.shape[1], -1)
    lower_clearance = (position - scenarios.arena_lower[None, :, :]) / span[None, :, :]
    upper_clearance = (scenarios.arena_upper[None, :, :] - position) / span[None, :, :]
    phase_feature = jnp.broadcast_to(phase, (*states.shape[:2], 1))
    return jnp.concatenate(
        (
            position_normalized,
            velocity_normalized,
            obstacle_features,
            lower_clearance,
            upper_clearance,
            phase_feature,
        ),
        axis=-1,
    )


def _compact_duration_gate(elapsed: Array, duration: Array, transition: float) -> Array:
    """Cubic compact transition from maneuver to an exact brake/hover tail."""
    coordinate = jnp.clip((duration + transition - elapsed) / (2.0 * transition), 0.0, 1.0)
    return coordinate**2 * (3.0 - 2.0 * coordinate)


def shared_fallback_actions(
    params: SharedActorParams,
    spec: SharedActorSpec,
    states: Array,
    scenarios: CircleScenarioBatch,
    *,
    elapsed: Array,
    horizon_duration: float,
    policy_gain: float,
    action_limit: float,
    config: SharedActorConfig,
) -> Array:
    """Evaluate all shared fallback policies for states shaped ``(K, B, 2D)``.

    Structural slots are exactly the bounded structured controller. Adaptive slots add a single
    shared latent-conditioned residual. At the end of each skill's duration, the same compact mask
    transitions both the maneuver and residual into a zero-velocity brake/hover tail.
    """
    config.validate()
    if states.ndim != 3:
        raise ValueError("states must have shape (K, B, 2 * D)")
    if states.shape[-1] % 2:
        raise ValueError("the state dimension must contain equal position and velocity halves")
    dimension = states.shape[-1] // 2
    n_obstacles = scenarios.obstacle_centers.shape[1]
    validate_shared_actor_shapes(params, spec, dimension=dimension, n_obstacles=n_obstacles)
    if states.shape[0] != spec.base_codes.shape[0]:
        raise ValueError("state policy axis must match the library")
    if states.shape[1] != scenarios.obstacle_centers.shape[0]:
        raise ValueError("state scenario axis must match scenarios")
    numeric_settings = (horizon_duration, policy_gain, action_limit)
    if not all(math.isfinite(value) and value > 0 for value in numeric_settings):
        raise ValueError(
            "horizon duration, policy gain, and action limit must be finite and positive"
        )

    adaptive = spec.adaptive_mask.astype(states.dtype)
    codes = spec.base_codes + adaptive[:, None] * params.code_offsets
    desired_velocity = spec.base_desired_velocities + adaptive[:, None] * params.velocity_offsets
    duration = jnp.clip(
        spec.base_durations + adaptive * params.duration_offsets,
        config.min_duration,
        config.max_duration,
    )
    gate = _compact_duration_gate(elapsed, duration[:, None], config.duration_transition)
    phase = jnp.asarray(elapsed / horizon_duration, dtype=states.dtype)
    observation = _task_agnostic_observation(states, scenarios, phase)
    code_batch = jnp.broadcast_to(codes[:, None, :], (*states.shape[:2], codes.shape[-1]))
    network_input = jnp.concatenate((observation, code_batch), axis=-1)
    hidden = jnp.tanh(network_input @ params.input_kernel + params.input_bias)
    hidden = jnp.tanh(hidden @ params.hidden_kernel + params.hidden_bias)
    residual = jnp.tanh(hidden @ params.output_kernel + params.output_bias)
    residual = config.residual_scale * adaptive[:, None, None] * gate[..., None] * residual

    _, velocity = jnp.split(states, 2, axis=-1)
    gated_desired_velocity = gate[..., None] * desired_velocity[:, None, :]
    structured = policy_gain * (gated_desired_velocity - velocity)
    action = action_limit * jnp.tanh((structured + residual) / action_limit)

    # Host-side ``CircleScenarioBatch.validate`` is the normal input boundary, but this public
    # function is also used inside JIT. Keep a device-side fail-closed gate so infinite real
    # obstacle data or limits cannot silently saturate the actor into an apparently valid action.
    mask = scenarios.obstacle_mask
    real_centers_finite = jnp.all(
        jnp.where(mask[..., None], jnp.isfinite(scenarios.obstacle_centers), True), axis=(-2, -1)
    )
    real_radii_valid = jnp.all(
        jnp.where(
            mask, jnp.isfinite(scenarios.obstacle_radii) & (scenarios.obstacle_radii > 0), True
        ),
        axis=-1,
    )
    arena_valid = jnp.all(
        jnp.isfinite(scenarios.arena_lower)
        & jnp.isfinite(scenarios.arena_upper)
        & (scenarios.arena_upper > scenarios.arena_lower),
        axis=-1,
    )
    speed_valid = jnp.isfinite(scenarios.speed_limit) & (scenarios.speed_limit > 0)
    scenario_valid = real_centers_finite & real_radii_valid & arena_valid & speed_valid
    policy_values = (
        spec.base_codes,
        spec.base_desired_velocities,
        spec.base_durations,
        *jax.tree.leaves(params),
    )
    policy_valid = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in policy_values]))
    elapsed_valid = jnp.all(jnp.isfinite(elapsed))
    valid = policy_valid & elapsed_valid & scenario_valid[None, :, None]
    return jnp.where(valid, action, jnp.nan)


__all__ = [
    "SharedActorConfig",
    "SharedActorParams",
    "SharedActorSpec",
    "actor_observation_size",
    "initialize_shared_actor",
    "shared_fallback_actions",
    "validate_shared_actor_shapes",
]
