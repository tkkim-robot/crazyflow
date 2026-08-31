"""PL-CBF-aligned learning objective for the 13-state shared quadrotor actor.

The objective is used only to propose a candidate snapshot.  Runtime certificates always use the
exact hard finite-horizon values and the independent post-checking filters.  Training uses a
conservative soft minimum over enabled physical constraints and swept sphere segments so gradients
cannot gain margin by skipping through an obstacle between simulator nodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.nn import sigmoid, softplus
from jax.scipy.special import logsumexp

from crazyflow.safety.da_plcbf.descriptors import trajectory_descriptors
from crazyflow.safety.da_plcbf.quad_rollouts import rollout_shared_quad_library
from crazyflow.safety.da_plcbf.version_a_barriers import (
    RigidBodySafetySet,
    dimensionless_safety_values,
)

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.actor import (
        SharedActorConfig,
        SharedActorParams,
        SharedActorSpec,
    )
    from crazyflow.safety.da_plcbf.config import LibraryLossConfig
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
    from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig, VersionAModel
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


@dataclass(frozen=True, slots=True)
class QuadLearningConfig:
    """Fixed-shape rollout and dimensionless regularization scales."""

    dt: float = 0.02
    horizon: int = 50
    policy_gain: float = 1.8
    softmin_beta: float = 40.0
    acceleration_scale: float = 4.0
    angular_velocity_scale: float = 8.0
    motor_saturation_temperature: float = 0.05

    def validate(self) -> None:
        """Reject settings that would invalidate rollout or normalized-loss semantics."""
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        values = (
            self.policy_gain,
            self.softmin_beta,
            self.acceleration_scale,
            self.angular_velocity_scale,
            self.motor_saturation_temperature,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("quad learning scales and temperatures must be finite and positive")


class QuadSafetyValues(NamedTuple):
    """Exact node/swept values with fixed ``(K, B, ...)`` leading axes."""

    node_values: Array
    node_enabled: Array
    segment_obstacle_values: Array
    segment_obstacle_enabled: Array
    input_valid: Array
    hard_policy_margins: Array
    smooth_policy_margins: Array


class QuadLibraryLossMetrics(NamedTuple):
    """Dimensionless candidate loss terms and hard held-in diagnostics."""

    total: Array
    coverage: Array
    redundancy: Array
    diversity: Array
    code: Array
    acceleration: Array
    action_rate: Array
    actuator_utilization: Array
    actuator_saturation: Array
    terminal: Array
    trust: Array
    hard_library_margin: Array
    hard_safe_fraction: Array
    smooth_safe_count: Array
    rollout_valid_fraction: Array


def rigid_body_safety_batch_from_circles(
    scenarios: CircleScenarioBatch, *, angular_rate_max: float, tilt_max_radians: float
) -> RigidBodySafetySet:
    """Lift a validated three-dimensional scenario batch into Version-A safety data."""
    scenarios.validate()
    if scenarios.obstacle_centers.shape[-1] != 3:
        raise ValueError("quadrotor safety scenarios must be three-dimensional")
    if not math.isfinite(angular_rate_max) or angular_rate_max <= 0:
        raise ValueError("angular_rate_max must be finite and positive")
    if (
        not math.isfinite(tilt_max_radians)
        or tilt_max_radians <= 0
        or tilt_max_radians > 0.5 * math.pi
    ):
        raise ValueError("tilt_max_radians must lie in (0, pi / 2]")
    batch_size = scenarios.obstacle_centers.shape[0]
    dtype = scenarios.obstacle_centers.dtype
    return RigidBodySafetySet(
        obstacle_centers=scenarios.obstacle_centers,
        obstacle_radii=scenarios.obstacle_radii,
        obstacle_mask=scenarios.obstacle_mask,
        arena_lower=scenarios.arena_lower,
        arena_upper=scenarios.arena_upper,
        speed_max=scenarios.speed_limit,
        angular_rate_max=jnp.full((batch_size,), angular_rate_max, dtype=dtype),
        tilt_max_radians=jnp.full((batch_size,), tilt_max_radians, dtype=dtype),
    )


def _validate_safety_batch_shapes(safety: RigidBodySafetySet, batch_size: int) -> None:
    centers = jnp.asarray(safety.obstacle_centers)
    if centers.ndim != 3 or centers.shape[0] != batch_size or centers.shape[-1] != 3:
        raise ValueError("batched obstacle_centers must have shape (B, O, 3)")
    obstacle_count = centers.shape[1]
    expected = {
        "obstacle_radii": (batch_size, obstacle_count),
        "obstacle_mask": (batch_size, obstacle_count),
        "arena_lower": (batch_size, 3),
        "arena_upper": (batch_size, 3),
        "speed_max": (batch_size,),
        "angular_rate_max": (batch_size,),
        "tilt_max_radians": (batch_size,),
    }
    for name, shape in expected.items():
        if jnp.asarray(getattr(safety, name)).shape != shape:
            raise ValueError(f"batched {name} must have shape {shape}")
    if not jnp.issubdtype(jnp.asarray(safety.obstacle_mask).dtype, jnp.bool_):
        raise ValueError("batched obstacle_mask must have boolean dtype")


def _one_scenario_node_values(
    nodes: Array,
    centers: Array,
    radii: Array,
    mask: Array,
    lower: Array,
    upper: Array,
    speed: Array,
    angular_rate: Array,
    tilt: Array,
    barrier_config: VersionABarrierConfig,
) -> tuple[Array, Array, Array]:
    safety = RigidBodySafetySet(centers, radii, mask, lower, upper, speed, angular_rate, tilt)
    result = jax.vmap(lambda node: dimensionless_safety_values(node, safety, barrier_config))(nodes)
    return result.values, result.enabled, result.input_valid


def quad_safety_values(
    states: Array,
    safety: RigidBodySafetySet,
    barrier_config: VersionABarrierConfig,
    *,
    softmin_beta: float,
) -> QuadSafetyValues:
    """Evaluate exact node and swept-sphere margins for ``states`` shaped ``(K,B,T,13)``."""
    barrier_config.validate()
    if states.ndim != 4 or states.shape[-1] != 13 or states.shape[2] < 2:
        raise ValueError("states must have shape (K, B, at_least_two_nodes, 13)")
    if not math.isfinite(softmin_beta) or softmin_beta <= 0:
        raise ValueError("softmin_beta must be finite and positive")
    _validate_safety_batch_shapes(safety, states.shape[1])

    evaluate_scenarios = jax.vmap(
        _one_scenario_node_values, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None)
    )

    def evaluate_policy(policy_states: Array) -> tuple[Array, Array, Array]:
        return evaluate_scenarios(
            policy_states,
            safety.obstacle_centers,
            safety.obstacle_radii,
            safety.obstacle_mask,
            safety.arena_lower,
            safety.arena_upper,
            safety.speed_max,
            safety.angular_rate_max,
            safety.tilt_max_radians,
            barrier_config,
        )

    node_values, node_enabled, node_valid = jax.vmap(evaluate_policy)(states)

    starts = states[:, :, :-1, :3]
    displacements = states[:, :, 1:, :3] - starts
    obstacle_mask = safety.obstacle_mask
    safe_centers = jnp.where(
        obstacle_mask[..., None] & jnp.isfinite(safety.obstacle_centers),
        safety.obstacle_centers,
        0.0,
    )
    safe_radii = jnp.where(
        obstacle_mask & jnp.isfinite(safety.obstacle_radii) & (safety.obstacle_radii > 0),
        safety.obstacle_radii,
        1.0,
    )
    centers = safe_centers[None, :, None, :, :]
    relative = starts[..., None, :] - centers
    denominator = jnp.sum(displacements**2, axis=-1, keepdims=True)
    safe_denominator = jnp.where(denominator > 0, denominator, 1.0)
    fraction = -jnp.sum(relative * displacements[..., None, :], axis=-1) / safe_denominator
    fraction = jnp.clip(fraction, 0.0, 1.0)
    closest = relative + fraction[..., None] * displacements[..., None, :]
    effective_radius = safe_radii + barrier_config.obstacle_clearance
    segment_values = (
        jnp.sum(closest**2, axis=-1) - effective_radius[None, :, None, :] ** 2
    ) / effective_radius[None, :, None, :] ** 2
    segment_enabled = jnp.broadcast_to(obstacle_mask[None, :, None, :], segment_values.shape)

    masked_nodes = jnp.where(node_enabled, node_values, jnp.inf)
    masked_segments = jnp.where(segment_enabled, segment_values, jnp.inf)
    flattened = jnp.concatenate(
        (
            masked_nodes.reshape(states.shape[0], states.shape[1], -1),
            masked_segments.reshape(states.shape[0], states.shape[1], -1),
        ),
        axis=-1,
    )
    finite_enabled_segments = jnp.all(
        jnp.where(segment_enabled, jnp.isfinite(segment_values), True), axis=(-2, -1)
    )
    valid = jnp.all(node_valid, axis=-1) & finite_enabled_segments
    hard = jnp.min(flattened, axis=-1)
    smooth = -logsumexp(-softmin_beta * flattened, axis=-1) / softmin_beta
    hard = jnp.where(valid, hard, -jnp.inf)
    smooth = jnp.where(valid, smooth, -jnp.inf)
    return QuadSafetyValues(
        node_values, node_enabled, segment_values, segment_enabled, valid, hard, smooth
    )


def _quad_actor_trust_region(
    candidate: SharedActorParams,
    active: SharedActorParams,
    velocity_scale: float,
    duration_scale: float,
) -> Array:
    terms = (
        jnp.mean((candidate.code_offsets - active.code_offsets) ** 2),
        jnp.mean(((candidate.velocity_offsets - active.velocity_offsets) / velocity_scale) ** 2),
        jnp.mean(((candidate.duration_offsets - active.duration_offsets) / duration_scale) ** 2),
        jnp.mean((candidate.input_kernel - active.input_kernel) ** 2),
        jnp.mean((candidate.input_bias - active.input_bias) ** 2),
        jnp.mean((candidate.hidden_kernel - active.hidden_kernel) ** 2),
        jnp.mean((candidate.hidden_bias - active.hidden_bias) ** 2),
        jnp.mean((candidate.output_kernel - active.output_kernel) ** 2),
        jnp.mean((candidate.output_bias - active.output_bias) ** 2),
    )
    return jnp.mean(jnp.stack(terms))


def quad_actor_library_loss(
    params: SharedActorParams,
    spec: SharedActorSpec,
    initial_states: Array,
    scenarios: CircleScenarioBatch,
    safety: RigidBodySafetySet,
    target_descriptors: Array,
    active_params: SharedActorParams,
    descriptor_scales: Array,
    model: VersionAModel,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_policy_config: QuadPolicyConfig,
    barrier_config: VersionABarrierConfig,
    learning_config: QuadLearningConfig,
    loss_config: LibraryLossConfig,
) -> tuple[Array, QuadLibraryLossMetrics]:
    """Compute the full dimensionless candidate objective through 13-state dynamics."""
    actor_config.validate()
    quad_policy_config.validate()
    barrier_config.validate()
    learning_config.validate()
    loss_config.validate()
    policy_count = spec.base_codes.shape[0]
    if initial_states.ndim != 2 or initial_states.shape[-1] != 13:
        raise ValueError("initial_states must have shape (B, 13)")
    if target_descriptors.shape != (policy_count, 9):
        raise ValueError("target_descriptors must have shape (K, 9)")
    if descriptor_scales.shape != (9,):
        raise ValueError("descriptor_scales must have shape (9,)")
    _validate_safety_batch_shapes(safety, initial_states.shape[0])

    rollouts = rollout_shared_quad_library(
        params,
        spec,
        initial_states,
        scenarios,
        model,
        actuator,
        dt=learning_config.dt,
        horizon=learning_config.horizon,
        policy_gain=learning_config.policy_gain,
        actor_config=actor_config,
        quad_config=quad_policy_config,
    )
    safety_values = quad_safety_values(
        rollouts.states, safety, barrier_config, softmin_beta=learning_config.softmin_beta
    )
    smooth_margins = safety_values.smooth_policy_margins
    hard_margins = safety_values.hard_policy_margins
    best_smooth = jnp.max(smooth_margins, axis=0)
    coverage = jnp.mean(
        softplus(
            (loss_config.target_margin - best_smooth) / loss_config.coverage_softplus_temperature
        )
    )
    soft_safe_count = jnp.sum(
        sigmoid((smooth_margins - loss_config.target_margin) / loss_config.safe_count_temperature),
        axis=0,
    )
    redundancy = -jnp.mean(jnp.log(loss_config.log_epsilon + soft_safe_count))

    translation = jnp.concatenate((rollouts.states[..., :3], rollouts.states[..., 7:10]), axis=-1)
    descriptors = trajectory_descriptors(translation) / descriptor_scales
    policy_descriptors = jnp.mean(descriptors, axis=1)
    centered = policy_descriptors - jnp.mean(policy_descriptors, axis=0, keepdims=True)
    covariance = centered.T @ centered / policy_count
    sign, logdet = jnp.linalg.slogdet(
        covariance + loss_config.covariance_regularizer * jnp.eye(9, dtype=initial_states.dtype)
    )
    diversity = jnp.where(sign > 0, -logdet, jnp.inf)
    normalized_targets = target_descriptors / descriptor_scales
    code = jnp.mean((descriptors - normalized_targets[:, None, :]) ** 2)

    acceleration = jnp.mean(
        (rollouts.desired_accelerations / learning_config.acceleration_scale) ** 2
    )
    action_rate = (
        jnp.mean(
            (jnp.diff(rollouts.desired_accelerations, axis=2) / learning_config.acceleration_scale)
            ** 2
        )
        if learning_config.horizon > 1
        else jnp.zeros((), dtype=initial_states.dtype)
    )
    lower = jnp.broadcast_to(jnp.asarray(actuator.thrust_min, dtype=initial_states.dtype), (4,))
    upper = jnp.broadcast_to(jnp.asarray(actuator.thrust_max, dtype=initial_states.dtype), (4,))
    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    motor_coordinate = (rollouts.bounded_motor_forces - center) / half_width
    actuator_utilization = jnp.mean(motor_coordinate**2)
    raw_coordinate = jnp.abs((rollouts.raw_motor_forces - center) / half_width) - 1.0
    actuator_saturation = jnp.mean(
        softplus(raw_coordinate / learning_config.motor_saturation_temperature)
        * learning_config.motor_saturation_temperature
    )
    final_velocity = rollouts.states[:, :, -1, 7:10] / (
        learning_config.acceleration_scale / learning_config.policy_gain
    )
    final_angular = rollouts.states[:, :, -1, 10:13] / learning_config.angular_velocity_scale
    terminal = jnp.mean(final_velocity**2) + jnp.mean(final_angular**2)
    trust = _quad_actor_trust_region(
        params,
        active_params,
        learning_config.acceleration_scale / learning_config.policy_gain,
        actor_config.max_duration - actor_config.min_duration,
    )
    total = (
        loss_config.coverage_weight * coverage
        + loss_config.redundancy_weight * redundancy
        + loss_config.diversity_weight * diversity
        + loss_config.code_weight * code
        + loss_config.action_weight * (acceleration + actuator_utilization + actuator_saturation)
        + loss_config.action_rate_weight * action_rate
        + loss_config.terminal_weight * terminal
        + loss_config.trust_weight * trust
    )
    rollout_valid = safety_values.input_valid & jnp.all(rollouts.policy_valid, axis=-1)
    total = jnp.where(jnp.all(rollout_valid) & jnp.isfinite(total), total, jnp.inf)
    hard_library = jnp.max(hard_margins, axis=0)
    metrics = QuadLibraryLossMetrics(
        total,
        coverage,
        redundancy,
        diversity,
        code,
        acceleration,
        action_rate,
        actuator_utilization,
        actuator_saturation,
        terminal,
        trust,
        jnp.min(hard_library),
        jnp.mean(hard_library >= 0),
        jnp.mean(soft_safe_count),
        jnp.mean(rollout_valid),
    )
    return total, metrics


__all__ = [
    "QuadLearningConfig",
    "QuadLibraryLossMetrics",
    "QuadSafetyValues",
    "quad_actor_library_loss",
    "quad_safety_values",
    "rigid_body_safety_batch_from_circles",
]
