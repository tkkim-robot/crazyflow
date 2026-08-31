"""PL-CBF-aligned objective for the shared latent-residual fallback actor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.nn import sigmoid, softplus

from crazyflow.safety.da_plcbf.actor_rollouts import rollout_shared_actor_library
from crazyflow.safety.da_plcbf.descriptors import trajectory_descriptors
from crazyflow.safety.da_plcbf.losses import LibraryLossMetrics
from crazyflow.safety.da_plcbf.values import (
    hard_policy_margins,
    swept_trajectory_constraints,
    training_policy_margins,
)

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.safety.da_plcbf.actor import (
        SharedActorConfig,
        SharedActorParams,
        SharedActorSpec,
    )
    from crazyflow.safety.da_plcbf.config import LibraryLossConfig, RolloutConfig
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch


def _actor_trust_region(
    candidate: SharedActorParams,
    active: SharedActorParams,
    velocity_scale: float,
    duration_scale: float,
) -> Array:
    """Dimensionless field-balanced active/candidate trust penalty."""
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


def shared_actor_library_loss(
    params: SharedActorParams,
    spec: SharedActorSpec,
    initial_states: Array,
    scenarios: CircleScenarioBatch,
    target_descriptors: Array,
    active_params: SharedActorParams,
    descriptor_scales: Array,
    rollout_config: RolloutConfig,
    actor_config: SharedActorConfig,
    loss_config: LibraryLossConfig,
) -> tuple[Array, LibraryLossMetrics]:
    """Compute the complete dimensionless objective for one shared-actor scenario batch."""
    rollout_config.validate()
    actor_config.validate()
    loss_config.validate()
    n_policies = spec.base_codes.shape[0]
    dimension = initial_states.shape[-1] // 2
    descriptor_size = 3 * dimension
    if target_descriptors.shape != (n_policies, descriptor_size):
        raise ValueError("target_descriptors must have shape (K, 3 * D)")
    if descriptor_scales.shape != (descriptor_size,):
        raise ValueError("descriptor_scales must have shape (3 * D,)")

    rollouts = rollout_shared_actor_library(
        params, spec, initial_states, scenarios, rollout_config, actor_config
    )
    constraints = swept_trajectory_constraints(
        rollouts.states,
        rollouts.actions,
        scenarios,
        rollout_config.safety_margin,
        rollout_config.dt,
        rollout_config.action_limit,
    )
    smooth_margins = training_policy_margins(constraints, rollout_config.softmin_beta)
    hard_margins = hard_policy_margins(constraints)
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

    descriptors = trajectory_descriptors(rollouts.states) / descriptor_scales
    policy_descriptors = jnp.mean(descriptors, axis=1)
    centered = policy_descriptors - jnp.mean(policy_descriptors, axis=0, keepdims=True)
    covariance = centered.T @ centered / n_policies
    sign, logdet = jnp.linalg.slogdet(
        covariance + loss_config.covariance_regularizer * jnp.eye(covariance.shape[0])
    )
    diversity = jnp.where(sign > 0, -logdet, jnp.inf)
    normalized_targets = target_descriptors / descriptor_scales
    code = jnp.mean((descriptors - normalized_targets[:, None, :]) ** 2)

    action = jnp.mean((rollouts.actions / rollout_config.action_limit) ** 2)
    action_rate = (
        jnp.mean((jnp.diff(rollouts.actions, axis=2) / rollout_config.action_limit) ** 2)
        if rollout_config.horizon > 1
        else jnp.zeros(())
    )
    _, terminal_velocity = jnp.split(rollouts.states[:, :, -1], 2, axis=-1)
    velocity_scale = rollout_config.action_limit / rollout_config.policy_gain
    terminal = jnp.mean((terminal_velocity / velocity_scale) ** 2)
    trust = _actor_trust_region(
        params, active_params, velocity_scale, actor_config.max_duration - actor_config.min_duration
    )
    total = (
        loss_config.coverage_weight * coverage
        + loss_config.redundancy_weight * redundancy
        + loss_config.diversity_weight * diversity
        + loss_config.code_weight * code
        + loss_config.action_weight * action
        + loss_config.action_rate_weight * action_rate
        + loss_config.terminal_weight * terminal
        + loss_config.trust_weight * trust
    )
    hard_library = jnp.max(hard_margins, axis=0)
    metrics = LibraryLossMetrics(
        total=total,
        coverage=coverage,
        redundancy=redundancy,
        diversity=diversity,
        code=code,
        action=action,
        action_rate=action_rate,
        terminal=terminal,
        trust=trust,
        hard_library_margin=jnp.min(hard_library),
        hard_safe_fraction=jnp.mean(hard_library >= 0),
        smooth_safe_count=jnp.mean(soft_safe_count),
    )
    return total, metrics


__all__ = ["shared_actor_library_loss"]
