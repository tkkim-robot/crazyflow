"""Fixed-budget truncated BPTT for candidate fallback libraries."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass
from jax import Array

from crazyflow.safety.da_plcbf.losses import LibraryLossMetrics, library_loss

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from crazyflow.safety.da_plcbf.config import LibraryLossConfig, RolloutConfig
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch


@dataclass
class BPTTState:
    """Candidate parameters and optimizer state, kept separate from an active snapshot."""

    params: Array
    optimizer_state: optax.OptState
    steps: Array


class BPTTStepMetrics(NamedTuple):
    """Loss decomposition and gradient/update diagnostics for one optimizer step."""

    loss: LibraryLossMetrics
    gradient_norm: Array
    parameter_delta_norm: Array
    update_accepted: Array


class BPTTFunctions(NamedTuple):
    """Constructed initializer, single update, and fused fixed-size burst."""

    initialize: Callable[[Array], BPTTState]
    step: Callable[..., tuple[BPTTState, BPTTStepMetrics]]
    burst: Callable[..., tuple[BPTTState, BPTTStepMetrics]]


def build_bptt_functions(
    rollout_config: RolloutConfig,
    loss_config: LibraryLossConfig,
    *,
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    max_gradient_norm: float = 10.0,
    burst_steps: int = 10,
) -> BPTTFunctions:
    """Build pre-jitted DA-PLCBF candidate-update functions.

    The active parameters enter only as immutable trust-region anchors. Only ``BPTTState.params``
    are updated. The fused burst uses a fixed scan length so its latency is predictable after JIT
    compilation.
    """
    rollout_config.validate()
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

    def initialize(params: Array) -> BPTTState:
        return BPTTState(
            params=params,
            optimizer_state=optimizer.init(params),
            steps=jnp.zeros((), dtype=jnp.int32),
        )

    def update(
        state: BPTTState,
        initial_states: Array,
        scenarios: CircleScenarioBatch,
        target_codes: Array,
        active_params: Array,
        descriptor_scales: Array,
    ) -> tuple[BPTTState, BPTTStepMetrics]:
        def objective(candidate_params: Array) -> tuple[Array, LibraryLossMetrics]:
            return library_loss(
                candidate_params,
                initial_states,
                scenarios,
                target_codes,
                active_params,
                descriptor_scales,
                rollout_config,
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
        target_codes: Array,
        active_params: Array,
        descriptor_scales: Array,
    ) -> tuple[BPTTState, BPTTStepMetrics]:
        def body(current: BPTTState, _: None) -> tuple[BPTTState, BPTTStepMetrics]:
            return update(
                current, initial_states, scenarios, target_codes, active_params, descriptor_scales
            )

        return jax.lax.scan(body, state, None, length=burst_steps)

    return BPTTFunctions(initialize=initialize, step=step, burst=burst)


def tree_all_finite(tree: Any) -> Array:
    """Return whether every numeric leaf in a JAX PyTree is finite."""
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.array(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
