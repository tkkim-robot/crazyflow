"""PL-CBF-aligned fallback-library learning objective."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.nn import sigmoid, softplus

from crazyflow.safety.da_plcbf.descriptors import trajectory_descriptors
from crazyflow.safety.da_plcbf.rollouts import rollout_structured_library
from crazyflow.safety.da_plcbf.values import (
    hard_policy_margins,
    swept_trajectory_constraints,
    training_policy_margins,
)

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.config import LibraryLossConfig, RolloutConfig
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch


class LibraryLossMetrics(NamedTuple):
    """Scalar loss terms plus hard held-in diagnostics."""

    total: Array
    coverage: Array
    redundancy: Array
    diversity: Array
    code: Array
    action: Array
    action_rate: Array
    terminal: Array
    trust: Array
    hard_library_margin: Array
    hard_safe_fraction: Array
    smooth_safe_count: Array


def library_loss(
    desired_velocities: Array,
    initial_states: Array,
    scenarios: CircleScenarioBatch,
    target_codes: Array,
    active_desired_velocities: Array,
    descriptor_scales: Array,
    rollout_config: RolloutConfig,
    loss_config: LibraryLossConfig,
) -> tuple[Array, LibraryLossMetrics]:
    """Compute the complete Phase-1 DA-PLCBF objective and diagnostics."""
    rollout_config.validate()
    loss_config.validate()
    if desired_velocities.shape != active_desired_velocities.shape:
        raise ValueError("active and candidate policy parameters must have identical shapes")
    expected_descriptor_size = 3 * desired_velocities.shape[-1]
    if target_codes.shape != (desired_velocities.shape[0], expected_descriptor_size):
        raise ValueError("target codes must have shape (K, 3 * D)")
    if descriptor_scales.shape != (expected_descriptor_size,):
        raise ValueError("descriptor scales must have shape (3 * D,)")

    rollouts = rollout_structured_library(
        desired_velocities, initial_states, rollout_config, smooth_actions=True
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

    # The exact best-policy reduction is invariant to duplicating a policy. An unnormalised
    # log-sum-exp would falsely improve coverage by ``temperature * log(K)`` under collapse.
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
    covariance = centered.T @ centered / desired_velocities.shape[0]
    sign, logdet = jnp.linalg.slogdet(
        covariance + loss_config.covariance_regularizer * jnp.eye(covariance.shape[0])
    )
    diversity = jnp.where(sign > 0, -logdet, jnp.inf)
    # Both operands must use the same dimensionless coordinates. ``target_codes`` are physical
    # trajectory descriptors (displacement, mean velocity, and final velocity), whereas
    # ``descriptors`` above have already been normalized by the corresponding SI scales.
    code = jnp.mean((descriptors - target_codes[:, None, :] / descriptor_scales) ** 2)

    action = jnp.mean((rollouts.actions / rollout_config.action_limit) ** 2)
    action_rate = (
        jnp.mean(
            (
                (jnp.diff(rollouts.actions, axis=2) / rollout_config.dt)
                / (rollout_config.action_limit / rollout_config.dt)
            )
            ** 2
        )
        if rollout_config.horizon > 1
        else jnp.zeros(())
    )
    _, terminal_velocity = jnp.split(rollouts.states[:, :, -1], 2, axis=-1)
    velocity_scale = rollout_config.action_limit / rollout_config.policy_gain
    terminal = jnp.mean((terminal_velocity / velocity_scale) ** 2)
    trust = jnp.mean(((desired_velocities - active_desired_velocities) / velocity_scale) ** 2)

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
