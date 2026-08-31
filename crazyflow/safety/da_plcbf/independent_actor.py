"""Genuinely independent policy-library actor used by the architecture ablation.

Unlike :mod:`crazyflow.safety.da_plcbf.actor`, every policy here owns separate input, hidden, and
output network parameters.  The leading ``K`` axis appears on every network tensor; there is no
latent-conditioned network shared between policies.  The observation, structural-policy mask,
compact maneuver duration, feasible-wrench map, plant, and learning objective otherwise match the
shared-actor comparator so the architecture axis can be changed without silently changing the
safety objective.

As with every learned library in this package, these functions only generate fallback proposals.
Exact hard finite-horizon certificates, validation admission, and runtime post-checks remain
separate requirements.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass as struct_dataclass
from jax import Array
from jax.nn import sigmoid, softplus

from crazyflow.safety.da_plcbf.actor import (
    SharedActorConfig,
    SharedActorSpec,
    actor_observation_size,
)
from crazyflow.safety.da_plcbf.bptt import (
    BPTTFunctions,
    BPTTState,
    BPTTStepMetrics,
    tree_all_finite,
)
from crazyflow.safety.da_plcbf.descriptors import trajectory_descriptors
from crazyflow.safety.da_plcbf.quad_actor_losses import (
    QuadLearningConfig,
    QuadLibraryLossMetrics,
    quad_safety_values,
)
from crazyflow.safety.da_plcbf.quad_policy import (
    QuadPolicyConfig,
    QuadWrenchCommand,
    acceleration_to_feasible_wrench,
)
from crazyflow.safety.da_plcbf.quad_rollouts import QuadRolloutBatch, direct_wrench_symplectic_step

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.config import LibraryLossConfig
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
    from crazyflow.safety.da_plcbf.version_a_barriers import (
        RigidBodySafetySet,
        VersionABarrierConfig,
        VersionAModel,
    )
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


@struct_dataclass
class IndependentActorParams:
    """Per-policy offsets and per-policy MLP tensors, all with leading axis ``K``."""

    code_offsets: Array
    velocity_offsets: Array
    duration_offsets: Array
    input_kernel: Array
    input_bias: Array
    hidden_kernel: Array
    hidden_bias: Array
    output_kernel: Array
    output_bias: Array


def validate_independent_actor_shapes(
    params: IndependentActorParams, spec: SharedActorSpec, *, dimension: int, n_obstacles: int
) -> None:
    """Reject any missing independent-policy axis or inconsistent actor tensor."""
    if not isinstance(params, IndependentActorParams):
        raise TypeError("params must be IndependentActorParams")
    if spec.base_codes.ndim != 2:
        raise ValueError("base_codes must have shape (K, Z)")
    policy_count, code_size = spec.base_codes.shape
    if policy_count < 1:
        raise ValueError("the independent policy library must not be empty")
    if spec.base_desired_velocities.shape != (policy_count, dimension):
        raise ValueError("base_desired_velocities must have shape (K, D)")
    if spec.base_durations.shape != (policy_count,):
        raise ValueError("base_durations must have shape (K,)")
    if spec.adaptive_mask.shape != (policy_count,) or not jnp.issubdtype(
        spec.adaptive_mask.dtype, jnp.bool_
    ):
        raise ValueError("adaptive_mask must have boolean shape (K,)")
    observation_size = actor_observation_size(dimension, n_obstacles)
    if params.input_bias.ndim != 2 or params.input_bias.shape[0] != policy_count:
        raise ValueError("input_bias must have shape (K, hidden_width)")
    hidden_width = params.input_bias.shape[1]
    expected = {
        "code_offsets": (policy_count, code_size),
        "velocity_offsets": (policy_count, dimension),
        "duration_offsets": (policy_count,),
        "input_kernel": (policy_count, observation_size + code_size, hidden_width),
        "hidden_kernel": (policy_count, hidden_width, hidden_width),
        "hidden_bias": (policy_count, hidden_width),
        "output_kernel": (policy_count, hidden_width, dimension),
        "output_bias": (policy_count, dimension),
    }
    for name, shape in expected.items():
        if getattr(params, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")


def initialize_independent_actor(
    key: Array,
    spec: SharedActorSpec,
    *,
    dimension: int,
    n_obstacles: int,
    config: SharedActorConfig,
) -> IndependentActorParams:
    """Initialize ``K`` separate residual MLPs with exactly zero residual output."""
    config.validate()
    if spec.base_codes.ndim != 2:
        raise ValueError("base_codes must have shape (K, Z)")
    policy_count, code_size = spec.base_codes.shape
    if policy_count < 1:
        raise ValueError("the independent policy library must not be empty")
    observation_size = actor_observation_size(dimension, n_obstacles)
    keys = jax.random.split(key, 2 * policy_count)
    input_keys = keys[:policy_count]
    hidden_keys = keys[policy_count:]

    def glorot(random_keys: Array, fan_in: int, fan_out: int) -> Array:
        limit = jnp.sqrt(6.0 / (fan_in + fan_out))
        return jax.vmap(
            lambda random_key: jax.random.uniform(
                random_key,
                (fan_in, fan_out),
                dtype=spec.base_codes.dtype,
                minval=-limit,
                maxval=limit,
            )
        )(random_keys)

    params = IndependentActorParams(
        code_offsets=jnp.zeros_like(spec.base_codes),
        velocity_offsets=jnp.zeros_like(spec.base_desired_velocities),
        duration_offsets=jnp.zeros_like(spec.base_durations),
        input_kernel=glorot(input_keys, observation_size + code_size, config.hidden_width),
        input_bias=jnp.zeros((policy_count, config.hidden_width), dtype=spec.base_codes.dtype),
        hidden_kernel=glorot(hidden_keys, config.hidden_width, config.hidden_width),
        hidden_bias=jnp.zeros((policy_count, config.hidden_width), dtype=spec.base_codes.dtype),
        output_kernel=jnp.zeros(
            (policy_count, config.hidden_width, dimension), dtype=spec.base_codes.dtype
        ),
        output_bias=jnp.zeros((policy_count, dimension), dtype=spec.base_codes.dtype),
    )
    validate_independent_actor_shapes(params, spec, dimension=dimension, n_obstacles=n_obstacles)
    return params


def _task_agnostic_observation(
    states: Array, scenarios: CircleScenarioBatch, phase: Array
) -> Array:
    """Match the shared actor's goal-free observation for states shaped ``(K,B,2D)``."""
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
    ).reshape(states.shape[0], states.shape[1], -1)
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


def independent_fallback_actions(
    params: IndependentActorParams,
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
    """Evaluate all independent policies for states shaped ``(K,B,2D)``."""
    config.validate()
    if states.ndim != 3 or states.shape[-1] % 2:
        raise ValueError("states must have shape (K, B, 2 * D)")
    dimension = states.shape[-1] // 2
    validate_independent_actor_shapes(
        params, spec, dimension=dimension, n_obstacles=scenarios.obstacle_centers.shape[1]
    )
    if states.shape[:2] != (spec.base_codes.shape[0], scenarios.obstacle_centers.shape[0]):
        raise ValueError("state policy/scenario axes must match the library and scenarios")
    if not all(
        math.isfinite(value) and value > 0
        for value in (horizon_duration, policy_gain, action_limit)
    ):
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
    transition_coordinate = jnp.clip(
        (duration[:, None] + config.duration_transition - elapsed)
        / (2.0 * config.duration_transition),
        0.0,
        1.0,
    )
    gate = transition_coordinate**2 * (3.0 - 2.0 * transition_coordinate)
    phase = jnp.asarray(elapsed / horizon_duration, dtype=states.dtype)
    observation = _task_agnostic_observation(states, scenarios, phase)
    code_batch = jnp.broadcast_to(codes[:, None, :], (*states.shape[:2], codes.shape[-1]))
    network_input = jnp.concatenate((observation, code_batch), axis=-1)
    hidden = jnp.tanh(
        jnp.einsum("kbi,kih->kbh", network_input, params.input_kernel)
        + params.input_bias[:, None, :]
    )
    hidden = jnp.tanh(
        jnp.einsum("kbi,kih->kbh", hidden, params.hidden_kernel) + params.hidden_bias[:, None, :]
    )
    residual = jnp.tanh(
        jnp.einsum("kbi,kid->kbd", hidden, params.output_kernel) + params.output_bias[:, None, :]
    )
    residual = config.residual_scale * adaptive[:, None, None] * gate[..., None] * residual
    _, velocity = jnp.split(states, 2, axis=-1)
    structured = policy_gain * (gate[..., None] * desired_velocity[:, None, :] - velocity)
    actions = action_limit * jnp.tanh((structured + residual) / action_limit)

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
    valid = policy_valid & jnp.all(jnp.isfinite(elapsed)) & scenario_valid[None, :, None]
    return jnp.where(valid, actions, jnp.nan)


def independent_quad_fallback_wrenches(
    params: IndependentActorParams,
    spec: SharedActorSpec,
    states: Array,
    scenarios: CircleScenarioBatch,
    model: VersionAModel,
    actuator: VersionAActuator,
    *,
    elapsed: Array,
    horizon_duration: float,
    policy_gain: float,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
) -> QuadWrenchCommand:
    """Map each independent policy to a feasible direct wrench."""
    if states.ndim != 3 or states.shape[-1] != 13:
        raise ValueError("states must have shape (K, B, 13)")
    translational = jnp.concatenate((states[..., :3], states[..., 7:10]), axis=-1)
    acceleration = independent_fallback_actions(
        params,
        spec,
        translational,
        scenarios,
        elapsed=elapsed,
        horizon_duration=horizon_duration,
        policy_gain=policy_gain,
        action_limit=quad_config.acceleration_limit,
        config=actor_config,
    )
    return acceleration_to_feasible_wrench(
        acceleration,
        states[..., 3:7],
        states[..., 10:13],
        model,
        actuator,
        quad_config,
        smooth_motor_bounds=True,
    )


def rollout_independent_quad_library(
    params: IndependentActorParams,
    spec: SharedActorSpec,
    initial_states: Array,
    scenarios: CircleScenarioBatch,
    model: VersionAModel,
    actuator: VersionAActuator,
    *,
    dt: float,
    horizon: int,
    policy_gain: float,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
) -> QuadRolloutBatch:
    """Roll out every independent-policy/scenario pair through the same 13-state plant."""
    if initial_states.ndim != 2 or initial_states.shape[-1] != 13:
        raise ValueError("initial_states must have shape (B, 13)")
    if initial_states.shape[0] != scenarios.obstacle_centers.shape[0]:
        raise ValueError("initial state and scenario batches must match")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not math.isfinite(policy_gain) or policy_gain <= 0:
        raise ValueError("policy_gain must be finite and positive")
    policy_count = spec.base_codes.shape[0]
    current = jnp.broadcast_to(initial_states[None, ...], (policy_count, *initial_states.shape))
    horizon_duration = int(horizon) * dt

    def advance(state: Array, step_index: Array) -> tuple[Array, tuple[Array, ...]]:
        command = independent_quad_fallback_wrenches(
            params,
            spec,
            state,
            scenarios,
            model,
            actuator,
            elapsed=step_index * dt,
            horizon_duration=horizon_duration,
            policy_gain=policy_gain,
            actor_config=actor_config,
            quad_config=quad_config,
        )
        following = direct_wrench_symplectic_step(state, command.wrench, model, dt)
        return following, (
            following,
            command.wrench,
            command.desired_acceleration,
            command.raw_motor_forces,
            command.bounded_motor_forces,
            command.input_valid,
        )

    _, outputs = jax.lax.scan(advance, current, jnp.arange(int(horizon), dtype=current.dtype))
    future, wrench, acceleration, raw_motor, bounded_motor, valid = (
        jnp.moveaxis(value, 0, 2) for value in outputs
    )
    states = jnp.concatenate((current[:, :, None, :], future), axis=2)
    return QuadRolloutBatch(states, wrench, acceleration, raw_motor, bounded_motor, valid)


def _independent_trust_region(
    candidate: IndependentActorParams,
    active: IndependentActorParams,
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


def independent_quad_actor_library_loss(
    params: IndependentActorParams,
    spec: SharedActorSpec,
    initial_states: Array,
    scenarios: CircleScenarioBatch,
    safety: RigidBodySafetySet,
    target_descriptors: Array,
    active_params: IndependentActorParams,
    descriptor_scales: Array,
    model: VersionAModel,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_policy_config: QuadPolicyConfig,
    barrier_config: VersionABarrierConfig,
    learning_config: QuadLearningConfig,
    loss_config: LibraryLossConfig,
) -> tuple[Array, QuadLibraryLossMetrics]:
    """Apply the same PL-CBF-aligned objective to the independent architecture."""
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
    validate_independent_actor_shapes(
        params, spec, dimension=3, n_obstacles=scenarios.obstacle_centers.shape[1]
    )
    validate_independent_actor_shapes(
        active_params, spec, dimension=3, n_obstacles=scenarios.obstacle_centers.shape[1]
    )

    rollouts = rollout_independent_quad_library(
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
    trust = _independent_trust_region(
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


def build_independent_quad_actor_bptt_functions(
    spec: SharedActorSpec,
    model: VersionAModel,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_policy_config: QuadPolicyConfig,
    barrier_config: VersionABarrierConfig,
    learning_config: QuadLearningConfig,
    loss_config: LibraryLossConfig,
    *,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    max_gradient_norm: float = 10.0,
    burst_steps: int = 10,
) -> BPTTFunctions:
    """Build fixed-budget BPTT updates for the independent-policy comparator."""
    actor_config.validate()
    quad_policy_config.validate()
    barrier_config.validate()
    learning_config.validate()
    loss_config.validate()
    if not bool(jax.device_get(jnp.any(spec.adaptive_mask))):
        raise ValueError("independent actor BPTT requires at least one adaptive policy slot")
    if isinstance(burst_steps, bool) or not isinstance(burst_steps, Integral) or burst_steps <= 0:
        raise ValueError("burst_steps must be a positive integer")
    if not all(math.isfinite(value) and value > 0 for value in (learning_rate, max_gradient_norm)):
        raise ValueError("learning_rate and max_gradient_norm must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_gradient_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )

    def initialize(params: IndependentActorParams) -> BPTTState:
        return BPTTState(
            params=params,
            optimizer_state=optimizer.init(params),
            steps=jnp.zeros((), dtype=jnp.int32),
        )

    def update(
        state: BPTTState,
        initial_states: Array,
        scenarios: CircleScenarioBatch,
        safety: RigidBodySafetySet,
        target_descriptors: Array,
        active_params: IndependentActorParams,
        descriptor_scales: Array,
    ) -> tuple[BPTTState, BPTTStepMetrics]:
        def objective(candidate: IndependentActorParams) -> tuple[Array, QuadLibraryLossMetrics]:
            return independent_quad_actor_library_loss(
                candidate,
                spec,
                initial_states,
                scenarios,
                safety,
                target_descriptors,
                active_params,
                descriptor_scales,
                model,
                actuator,
                actor_config,
                quad_policy_config,
                barrier_config,
                learning_config,
                loss_config,
            )

        (_, loss_metrics), gradients = jax.value_and_grad(objective, has_aux=True)(state.params)
        updates, proposed_optimizer_state = optimizer.update(
            gradients, state.optimizer_state, params=state.params
        )
        proposed_params = optax.apply_updates(state.params, updates)
        accepted = (
            tree_all_finite(loss_metrics)
            & tree_all_finite(gradients)
            & tree_all_finite(proposed_params)
            & tree_all_finite(proposed_optimizer_state)
        )
        params = jax.tree.map(
            lambda proposed, current: jnp.where(accepted, proposed, current),
            proposed_params,
            state.params,
        )
        optimizer_state = jax.tree.map(
            lambda proposed, current: jnp.where(accepted, proposed, current),
            proposed_optimizer_state,
            state.optimizer_state,
        )
        metrics = BPTTStepMetrics(
            loss=loss_metrics,
            gradient_norm=optax.tree.norm(gradients),
            parameter_delta_norm=optax.tree.norm(
                jax.tree.map(lambda new, old: new - old, params, state.params)
            ),
            update_accepted=accepted,
        )
        return (
            state.replace(params=params, optimizer_state=optimizer_state, steps=state.steps + 1),
            metrics,
        )

    step = jax.jit(update)

    @jax.jit
    def burst(
        state: BPTTState,
        initial_states: Array,
        scenarios: CircleScenarioBatch,
        safety: RigidBodySafetySet,
        target_descriptors: Array,
        active_params: IndependentActorParams,
        descriptor_scales: Array,
    ) -> tuple[BPTTState, BPTTStepMetrics]:
        def body(current: BPTTState, _: None) -> tuple[BPTTState, BPTTStepMetrics]:
            return update(
                current,
                initial_states,
                scenarios,
                safety,
                target_descriptors,
                active_params,
                descriptor_scales,
            )

        return jax.lax.scan(body, state, None, length=int(burst_steps))

    return BPTTFunctions(initialize=initialize, step=step, burst=burst)


__all__ = [
    "IndependentActorParams",
    "build_independent_quad_actor_bptt_functions",
    "independent_fallback_actions",
    "independent_quad_actor_library_loss",
    "independent_quad_fallback_wrenches",
    "initialize_independent_actor",
    "rollout_independent_quad_library",
    "validate_independent_actor_shapes",
]
