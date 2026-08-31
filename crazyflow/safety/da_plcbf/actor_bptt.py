"""Fixed-budget truncated BPTT for the shared latent-residual candidate actor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax

from crazyflow.safety.da_plcbf.actor_losses import shared_actor_library_loss
from crazyflow.safety.da_plcbf.bptt import (
    BPTTFunctions,
    BPTTState,
    BPTTStepMetrics,
    tree_all_finite,
)

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.safety.da_plcbf.actor import (
        SharedActorConfig,
        SharedActorParams,
        SharedActorSpec,
    )
    from crazyflow.safety.da_plcbf.config import LibraryLossConfig, RolloutConfig
    from crazyflow.safety.da_plcbf.losses import LibraryLossMetrics
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch


def build_shared_actor_bptt_functions(
    spec: SharedActorSpec,
    rollout_config: RolloutConfig,
    actor_config: SharedActorConfig,
    loss_config: LibraryLossConfig,
    *,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    max_gradient_norm: float = 10.0,
    burst_steps: int = 10,
) -> BPTTFunctions:
    """Build pre-jitted candidate-only updates through shared closed-loop rollouts."""
    rollout_config.validate()
    actor_config.validate()
    loss_config.validate()
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative")
    if max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be positive")
    if burst_steps <= 0:
        raise ValueError("burst_steps must be positive")

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_gradient_norm),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
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
        active_params: SharedActorParams,
        descriptor_scales: Array,
    ) -> tuple[BPTTState, BPTTStepMetrics]:
        def objective(candidate: SharedActorParams) -> tuple[Array, LibraryLossMetrics]:
            return shared_actor_library_loss(
                candidate,
                spec,
                initial_states,
                scenarios,
                target_descriptors,
                active_params,
                descriptor_scales,
                rollout_config,
                actor_config,
                loss_config,
            )

        (_, loss_metrics), gradients = jax.value_and_grad(objective, has_aux=True)(state.params)
        updates, proposed_optimizer_state = optimizer.update(
            gradients, state.optimizer_state, params=state.params
        )
        proposed_params = optax.apply_updates(state.params, updates)
        update_accepted = (
            tree_all_finite(loss_metrics)
            & tree_all_finite(gradients)
            & tree_all_finite(proposed_params)
            & tree_all_finite(proposed_optimizer_state)
        )
        params = jax.tree.map(
            lambda proposed, current: jnp.where(update_accepted, proposed, current),
            proposed_params,
            state.params,
        )
        optimizer_state = jax.tree.map(
            lambda proposed, current: jnp.where(update_accepted, proposed, current),
            proposed_optimizer_state,
            state.optimizer_state,
        )
        metrics = BPTTStepMetrics(
            loss=loss_metrics,
            gradient_norm=optax.tree.norm(gradients),
            parameter_delta_norm=optax.tree.norm(
                jax.tree.map(lambda new, old: new - old, params, state.params)
            ),
            update_accepted=update_accepted,
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
        active_params: SharedActorParams,
        descriptor_scales: Array,
    ) -> tuple[BPTTState, BPTTStepMetrics]:
        def body(current: BPTTState, _: None) -> tuple[BPTTState, BPTTStepMetrics]:
            return update(
                current,
                initial_states,
                scenarios,
                target_descriptors,
                active_params,
                descriptor_scales,
            )

        return jax.lax.scan(body, state, None, length=burst_steps)

    return BPTTFunctions(initialize=initialize, step=step, burst=burst)


__all__ = ["build_shared_actor_bptt_functions"]
