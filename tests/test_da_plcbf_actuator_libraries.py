"""Nominal handcrafted invariants and immutable union coverage at common states."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.actuator_learning import actuator_skill_actions
from crazyflow.safety.da_plcbf.actuator_libraries import (
    handcrafted_pd_contract,
    handcrafted_pd_spec,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_study import (
    build_actuator_controller,
    nominal_actuator_model,
)


def test_pd_has_distinct_directions_hover_and_no_network_behavior():
    model = nominal_actuator_model()
    contract = handcrafted_pd_contract(model)
    spec, config, params = contract.spec, contract.actor_config, contract.params
    assert spec.base_desired_velocities.shape == (16, 3)
    assert len(np.unique(np.asarray(spec.base_desired_velocities), axis=0)) == 16
    assert config.trainable_parameters == "offsets"
    assert config.residual_scale == 0
    states = jnp.broadcast_to(contract.anchors[0], (16, 17))
    for phase in (0.0, 0.9):
        original = actuator_skill_actions(
            params, spec, states, states[0, :3], jnp.array(phase), config
        )
        changed = params.replace(output_bias=jnp.ones(3) * 1000)
        np.testing.assert_array_equal(
            original,
            actuator_skill_actions(changed, spec, states, states[0, :3], jnp.array(phase), config),
        )
    np.testing.assert_allclose(original[0], 0, atol=1e-7)


def test_union_keeps_all_original_common_state_rollouts_after_adaptation():
    contract = handcrafted_pd_contract(nominal_actuator_model())
    config = replace(contract.actor_config, horizon=6)
    cfg = ActuatorFilterConfig(horizon=6)
    base = build_actuator_controller(contract.spec, config, cfg)
    union = build_actuator_controller(
        contract.spec, config, cfg, frozen_library=(contract.params, contract.spec, config)
    )
    y = contract.anchors[3]
    changed = contract.params.replace(velocity_offsets=jnp.ones((16, 3)) * 0.3)
    args = (y, changed, y[:3], contract.model)
    augmented = jax.block_until_ready(union.candidates(*args))
    frozen = jax.block_until_ready(base.candidates(y, contract.params, y[:3], contract.model))
    assert augmented.states.shape[0] == 33
    np.testing.assert_allclose(augmented.states[:17], frozen.states, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(augmented.commands[:17], frozen.commands, atol=2e-7, rtol=2e-6)
    np.testing.assert_array_equal(augmented.valid[:17], frozen.valid)


def test_union_rejects_different_hold_and_pd_rejects_impossible_duration():
    contract = handcrafted_pd_contract(nominal_actuator_model())
    with pytest.raises(ValueError, match="command|control_interval"):
        build_actuator_controller(
            contract.spec,
            contract.actor_config,
            frozen_library=(
                contract.params,
                contract.spec,
                replace(contract.actor_config, control_interval_steps=1),
            ),
        )
    with pytest.raises(ValueError, match="duration"):
        handcrafted_pd_spec(duration=2.0)
