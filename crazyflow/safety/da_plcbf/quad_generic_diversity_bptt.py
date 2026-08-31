"""Generic-diversity BPTT comparator for the offline/frozen learned library.

This objective intentionally contains no PL-CBF coverage, redundancy, safety-margin, or
active-snapshot trust term.  It trains the shared actor to span normalized trajectory descriptors
while staying close to its predeclared skill-code targets and using regularized actions.  Hard
safety is evaluated only after training and during the common runtime filter; it is not smuggled
into this SDCBF-style comparison objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import Array

from crazyflow.safety.da_plcbf.bptt import (
    BPTTFunctions,
    BPTTState,
    BPTTStepMetrics,
    tree_all_finite,
)
from crazyflow.safety.da_plcbf.descriptors import trajectory_descriptors
from crazyflow.safety.da_plcbf.quad_rollouts import rollout_shared_quad_library

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.actor import (
        SharedActorConfig,
        SharedActorParams,
        SharedActorSpec,
    )
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
    from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


@dataclass(frozen=True, slots=True)
class GenericDiversityConfig:
    """Dimensionless, predeclared weights for the generic comparator objective."""

    covariance_regularizer: float = 1e-4
    diversity_weight: float = 1.0
    descriptor_target_weight: float = 0.05
    action_weight: float = 1e-3
    action_rate_weight: float = 1e-3
    terminal_weight: float = 1e-3
    acceleration_scale: float = 4.0
    angular_velocity_scale: float = 8.0

    def validate(self) -> None:
        values = (
            self.covariance_regularizer,
            self.diversity_weight,
            self.descriptor_target_weight,
            self.action_weight,
            self.action_rate_weight,
            self.terminal_weight,
            self.acceleration_scale,
            self.angular_velocity_scale,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("generic-diversity scales and weights must be finite and positive")


class GenericDiversityMetrics(NamedTuple):
    """Loss decomposition emitted by every generic-diversity optimizer step."""

    total: Array
    diversity: Array
    descriptor_target: Array
    action: Array
    action_rate: Array
    terminal: Array
    rollout_valid_fraction: Array


def generic_diversity_loss(
    params: SharedActorParams,
    spec: SharedActorSpec,
    initial_states: Array,
    scenarios: CircleScenarioBatch,
    target_descriptors: Array,
    descriptor_scales: Array,
    model: VersionAModel,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    *,
    dt: float,
    horizon: int,
    policy_gain: float,
    config: GenericDiversityConfig = GenericDiversityConfig(),
) -> tuple[Array, GenericDiversityMetrics]:
    """Return a safety-agnostic descriptor diversity objective through the real quad plant."""
    config.validate()
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not math.isfinite(policy_gain) or policy_gain <= 0:
        raise ValueError("policy_gain must be finite and positive")
    policy_count = spec.base_codes.shape[0]
    if target_descriptors.shape != (policy_count, 9) or descriptor_scales.shape != (9,):
        raise ValueError("target_descriptors and descriptor_scales must have shapes (K,9) and (9,)")
    scales_valid = jnp.all(jnp.isfinite(descriptor_scales) & (descriptor_scales > 0))
    safe_scales = jnp.where(scales_valid, descriptor_scales, jnp.ones_like(descriptor_scales))

    rollout = rollout_shared_quad_library(
        params,
        spec,
        initial_states,
        scenarios,
        model,
        actuator,
        dt=dt,
        horizon=horizon,
        policy_gain=policy_gain,
        actor_config=actor_config,
        quad_config=quad_config,
    )
    translation = jnp.concatenate((rollout.states[..., :3], rollout.states[..., 7:10]), axis=-1)
    descriptors = trajectory_descriptors(translation) / safe_scales
    policy_descriptors = jnp.mean(descriptors, axis=1)
    centered = policy_descriptors - jnp.mean(policy_descriptors, axis=0, keepdims=True)
    covariance = centered.T @ centered / policy_count
    sign, logdet = jnp.linalg.slogdet(
        covariance + config.covariance_regularizer * jnp.eye(9, dtype=initial_states.dtype)
    )
    diversity = jnp.where(sign > 0, -logdet, jnp.inf)
    descriptor_target = jnp.mean((descriptors - target_descriptors[:, None, :] / safe_scales) ** 2)
    action = jnp.mean((rollout.desired_accelerations / config.acceleration_scale) ** 2)
    action_rate = (
        jnp.mean((jnp.diff(rollout.desired_accelerations, axis=2) / config.acceleration_scale) ** 2)
        if horizon > 1
        else jnp.zeros((), dtype=initial_states.dtype)
    )
    terminal = jnp.mean((rollout.states[:, :, -1, 7:10] / config.acceleration_scale) ** 2)
    terminal += jnp.mean((rollout.states[:, :, -1, 10:13] / config.angular_velocity_scale) ** 2)
    total = (
        config.diversity_weight * diversity
        + config.descriptor_target_weight * descriptor_target
        + config.action_weight * action
        + config.action_rate_weight * action_rate
        + config.terminal_weight * terminal
    )
    valid_fraction = jnp.mean(rollout.policy_valid)
    total = jnp.where(
        scales_valid & jnp.all(rollout.policy_valid) & jnp.isfinite(total), total, jnp.inf
    )
    return total, GenericDiversityMetrics(
        total, diversity, descriptor_target, action, action_rate, terminal, valid_fraction
    )


def build_quad_generic_diversity_bptt_functions(
    spec: SharedActorSpec,
    model: VersionAModel,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    *,
    dt: float,
    horizon: int,
    policy_gain: float,
    config: GenericDiversityConfig = GenericDiversityConfig(),
    learning_rate: float = 1e-3,
    max_gradient_norm: float = 10.0,
    burst_steps: int = 10,
) -> BPTTFunctions:
    """Build a fixed-budget BPTT burst for the offline generic-diversity protocol."""
    config.validate()
    if not bool(jax.device_get(jnp.any(spec.adaptive_mask))):
        raise ValueError("generic-diversity BPTT requires at least one adaptive policy slot")
    if isinstance(burst_steps, bool) or not isinstance(burst_steps, int) or burst_steps <= 0:
        raise ValueError("burst_steps must be a positive integer")
    if not all(math.isfinite(value) and value > 0 for value in (learning_rate, max_gradient_norm)):
        raise ValueError("optimizer scales must be finite and positive")
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_gradient_norm), optax.adam(learning_rate=learning_rate)
    )

    def initialize(params: SharedActorParams) -> BPTTState:
        return BPTTState(
            params=params,
            optimizer_state=optimizer.init(params),
            steps=jnp.zeros((), dtype=jnp.int32),
        )

    def update(
        state: BPTTState,
        initial_states: Array,
        scenarios: CircleScenarioBatch,
        target_descriptors: Array,
        descriptor_scales: Array,
    ) -> tuple[BPTTState, BPTTStepMetrics]:
        def objective(candidate: SharedActorParams) -> tuple[Array, GenericDiversityMetrics]:
            return generic_diversity_loss(
                candidate,
                spec,
                initial_states,
                scenarios,
                target_descriptors,
                descriptor_scales,
                model,
                actuator,
                actor_config,
                quad_config,
                dt=dt,
                horizon=horizon,
                policy_gain=policy_gain,
                config=config,
            )

        (_, loss_metrics), gradients = jax.value_and_grad(objective, has_aux=True)(state.params)
        updates, proposed_optimizer = optimizer.update(
            gradients, state.optimizer_state, params=state.params
        )
        proposed_params = optax.apply_updates(state.params, updates)
        accepted = (
            tree_all_finite(loss_metrics)
            & tree_all_finite(gradients)
            & tree_all_finite(proposed_params)
            & tree_all_finite(proposed_optimizer)
        )
        params = jax.tree.map(
            lambda proposed, current: jnp.where(accepted, proposed, current),
            proposed_params,
            state.params,
        )
        optimizer_state = jax.tree.map(
            lambda proposed, current: jnp.where(accepted, proposed, current),
            proposed_optimizer,
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
        target_descriptors: Array,
        descriptor_scales: Array,
    ) -> tuple[BPTTState, BPTTStepMetrics]:
        def body(current: BPTTState, _: None) -> tuple[BPTTState, BPTTStepMetrics]:
            return update(current, initial_states, scenarios, target_descriptors, descriptor_scales)

        return jax.lax.scan(body, state, None, length=burst_steps)

    return BPTTFunctions(initialize=initialize, step=step, burst=burst)


__all__ = [
    "GenericDiversityConfig",
    "GenericDiversityMetrics",
    "build_quad_generic_diversity_bptt_functions",
    "generic_diversity_loss",
]
