"""Fixed-budget truncated BPTT through the 13-state direct-wrench quadrotor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax

from crazyflow.safety.da_plcbf.bptt import (
    BPTTFunctions,
    BPTTState,
    BPTTStepMetrics,
    tree_all_finite,
)
from crazyflow.safety.da_plcbf.quad_actor_losses import quad_actor_library_loss

if TYPE_CHECKING:
    from jax import Array

    from crazyflow.safety.da_plcbf.actor import (
        SharedActorConfig,
        SharedActorParams,
        SharedActorSpec,
    )
    from crazyflow.safety.da_plcbf.config import LibraryLossConfig
    from crazyflow.safety.da_plcbf.quad_actor_losses import QuadLearningConfig
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
    from crazyflow.safety.da_plcbf.version_a_barriers import (
        RigidBodySafetySet,
        VersionABarrierConfig,
        VersionAModel,
    )
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


def build_quad_actor_bptt_functions(
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
    device: jax.Device | None = None,
) -> BPTTFunctions:
    """Build pre-jitted, candidate-only BPTT steps with a fixed update budget."""
    actor_config.validate()
    quad_policy_config.validate()
    barrier_config.validate()
    learning_config.validate()
    loss_config.validate()
    if not bool(jax.device_get(jnp.any(spec.adaptive_mask))):
        raise ValueError("quad actor BPTT requires at least one adaptive policy slot")
    if not isinstance(burst_steps, int) or isinstance(burst_steps, bool) or burst_steps <= 0:
        raise ValueError("burst_steps must be a positive integer")
    if not all(jnp.isfinite(value) and value > 0 for value in (learning_rate, max_gradient_norm)):
        raise ValueError("learning_rate and max_gradient_norm must be finite and positive")
    if not jnp.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")

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
        safety: RigidBodySafetySet,
        target_descriptors: Array,
        active_params: SharedActorParams,
        descriptor_scales: Array,
    ) -> tuple[BPTTState, BPTTStepMetrics]:
        def objective(candidate: SharedActorParams) -> tuple[Array, object]:
            return quad_actor_library_loss(
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

    step = jax.jit(update, device=device)

    def burst_impl(
        state: BPTTState,
        initial_states: Array,
        scenarios: CircleScenarioBatch,
        safety: RigidBodySafetySet,
        target_descriptors: Array,
        active_params: SharedActorParams,
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

        return jax.lax.scan(body, state, None, length=burst_steps)

    burst = jax.jit(burst_impl, device=device)
    return BPTTFunctions(initialize=initialize, step=step, burst=burst)


def build_dynamic_model_quad_actor_bptt_functions(
    spec: SharedActorSpec,
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
    device: jax.Device | None = None,
) -> BPTTFunctions:
    """Build one cacheable BPTT executable whose dynamics model is a runtime argument.

    Unlike :func:`build_quad_actor_bptt_functions`, this variant does not close over model values.
    A changing estimator model with the same shapes therefore reuses the same compiled executable.
    The returned ``step`` and ``burst`` accept ``model`` as their final argument.
    """
    actor_config.validate()
    quad_policy_config.validate()
    barrier_config.validate()
    learning_config.validate()
    loss_config.validate()
    if not bool(jax.device_get(jnp.any(spec.adaptive_mask))):
        raise ValueError("quad actor BPTT requires at least one adaptive policy slot")
    if not isinstance(burst_steps, int) or isinstance(burst_steps, bool) or burst_steps <= 0:
        raise ValueError("burst_steps must be a positive integer")
    if not all(jnp.isfinite(value) and value > 0 for value in (learning_rate, max_gradient_norm)):
        raise ValueError("learning_rate and max_gradient_norm must be finite and positive")
    if not jnp.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")

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
        safety: RigidBodySafetySet,
        target_descriptors: Array,
        active_params: SharedActorParams,
        descriptor_scales: Array,
        model: VersionAModel,
    ) -> tuple[BPTTState, BPTTStepMetrics]:
        def objective(candidate: SharedActorParams) -> tuple[Array, object]:
            return quad_actor_library_loss(
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

    step = jax.jit(update, device=device)

    def burst_impl(
        state: BPTTState,
        initial_states: Array,
        scenarios: CircleScenarioBatch,
        safety: RigidBodySafetySet,
        target_descriptors: Array,
        active_params: SharedActorParams,
        descriptor_scales: Array,
        model: VersionAModel,
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
                model,
            )

        return jax.lax.scan(body, state, None, length=burst_steps)

    burst = jax.jit(burst_impl, device=device)
    return BPTTFunctions(initialize=initialize, step=step, burst=burst)


__all__ = ["build_dynamic_model_quad_actor_bptt_functions", "build_quad_actor_bptt_functions"]
