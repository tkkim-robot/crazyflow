"""Braking-tail expressiveness and honest zero-gradient restoration with persistent Adam."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from crazyflow.safety.da_plcbf.online_constant_wind import build_cf21b_version_a_resources
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    build_fibonacci_skill_spec,
    build_persistent_skill_learner,
    initialize_skill_actor,
    obstacle_agnostic_skill_actions,
)
from crazyflow.safety.da_plcbf.state_conditioned_learning import (
    ReferenceContract,
    ReferenceLearningConfig,
    build_reference_skill_learner,
    proprioceptive_state_bank,
)


def test_ungated_residual_can_change_braking_tail_and_default_cannot() -> None:
    config = PersistentSkillConfig(horizon=60, hidden_width=8, smooth_motor_bounds=False)
    spec = build_fibonacci_skill_spec(policy_count=4, latent_size=3, maximum_duration=0.7)
    params = initialize_skill_actor(jax.random.key(11), spec, config)
    params = jax.tree.map(jnp.zeros_like, params).replace(output_bias=jnp.asarray([0.4, 0.0, 0.0]))
    hover = proprioceptive_state_bank()[0][0]
    states = jnp.broadcast_to(hover, (4, 13))

    def action(bias: jax.Array, gated: bool) -> jax.Array:
        return obstacle_agnostic_skill_actions(
            params.replace(output_bias=params.output_bias.at[0].set(bias)),
            spec,
            states,
            hover[:3],
            jnp.asarray(1.0),
            replace(config, gate_residual_with_skill_duration=gated),
        )[0, 0]

    assert config.gate_residual_with_skill_duration
    assert float(action(jnp.asarray(0.4), True)) == 0.0
    assert float(jax.grad(lambda bias: action(bias, True))(jnp.asarray(0.4))) == 0.0
    assert float(action(jnp.asarray(0.4), False)) > 0.3
    assert float(jax.grad(lambda bias: action(bias, False))(jnp.asarray(0.4))) > 0.5


def test_rebased_tracking_zero_gradient_does_not_reset_existing_adam() -> None:
    resources = build_cf21b_version_a_resources()
    config = PersistentSkillConfig(
        horizon=8,
        hidden_width=8,
        control_interval_steps=2,
        model_compensation=False,
        smooth_motor_bounds=False,
    )
    spec = build_fibonacci_skill_spec(
        policy_count=4,
        latent_size=3,
        minimum_duration=0.04,
        maximum_duration=0.12,
        horizon_duration=0.16,
    )
    params = initialize_skill_actor(jax.random.key(12), spec, config)
    bank = proprioceptive_state_bank()[0]
    original = build_persistent_skill_learner(spec, resources.actuator, config)
    initial = original.initialize(params, resources.model)
    persistent, metric = original.step(initial, bank[0], resources.model)
    assert bool(metric.finite_update_applied)
    restored_config = replace(
        config,
        diversity_weight=0.0,
        pairwise_weight=0.0,
        action_weight=0.0,
        action_rate_weight=0.0,
        saturation_weight=0.0,
        trust_weight=0.0,
        attitude_weight=0.0,
        angular_rate_weight=0.0,
        terminal_braking_weight=0.0,
        learn_durations=False,
    )
    contract = ReferenceContract(
        persistent.params,
        resources.model,
        bank,
        restored_config,
        ReferenceLearningConfig(),
        spec,
        resources.actuator,
    )
    learner = build_reference_skill_learner(contract)
    gradient = jax.grad(
        lambda value: learner.loss(
            value, bank[0], resources.model, persistent.previous_params, persistent.library_version
        )[0]
    )(persistent.params)
    assert float(optax.tree.norm(gradient)) == pytest.approx(0.0, abs=1e-8)
    changed, metric = learner.step(persistent, bank[0], resources.model)
    assert bool(metric.finite_update_applied)
    assert float(metric.parameter_update_norm) > 0.0
    assert int(changed.library_version) == int(persistent.library_version) + 1
    np.testing.assert_array_equal(
        changed.params.duration_offsets, persistent.params.duration_offsets
    )
    with pytest.raises(ValueError, match="gate_residual_with_skill_duration"):
        build_reference_skill_learner(
            contract, replace(restored_config, gate_residual_with_skill_duration=False)
        )
