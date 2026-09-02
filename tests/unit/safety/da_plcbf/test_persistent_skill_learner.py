from __future__ import annotations

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    build_fibonacci_skill_spec,
    build_persistent_skill_learner,
    initialize_skill_actor,
    obstacle_agnostic_skill_actions,
)
from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


def _problem() -> tuple[object, ...]:
    physical: dict[str, Any] = load_params("cf21B_500")
    model = VersionAModel(
        mass=jnp.asarray(physical["mass"]),
        gravity_vec=jnp.asarray(physical["gravity_vec"]),
        inertia=jnp.asarray(physical["J"]),
        inertia_inv=jnp.linalg.inv(jnp.asarray(physical["J"])),
        drag_matrix=jnp.asarray(physical["drag_matrix"]),
        wind_velocity=jnp.zeros(3),
        external_force=jnp.zeros(3),
        external_torque=jnp.zeros(3),
    )
    actuator = VersionAActuator(
        arm_length=jnp.asarray(physical["L"]),
        thrust_to_torque=jnp.asarray(physical["thrust2torque"]),
        mixing_matrix=jnp.asarray(physical["mixing_matrix"]),
        thrust_min=jnp.asarray(physical["thrust_min"]),
        thrust_max=jnp.asarray(physical["thrust_max"]),
    )
    config = PersistentSkillConfig(dt=0.02, horizon=4, hidden_width=8)
    spec = build_fibonacci_skill_spec(
        policy_count=4, latent_size=3, minimum_duration=0.04, maximum_duration=0.08
    )
    params = initialize_skill_actor(jax.random.key(7), spec, config)
    initial_state = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, actuator, config, spec, params, initial_state


def test_actor_public_input_is_proprioception_skill_and_phase_only() -> None:
    _, _, config, spec, params, initial_state = _problem()
    argument_names = tuple(inspect.signature(obstacle_agnostic_skill_actions).parameters)
    forbidden = ("goal", "waypoint", "obstacle", "scenario", "safety", "margin")
    assert not any(token in name for name in argument_names for token in forbidden)

    states = jnp.broadcast_to(initial_state, (spec.latent_codes.shape[0], 13))
    first = obstacle_agnostic_skill_actions(
        params, spec, states, initial_state[:3], jnp.asarray(0.25), config
    )
    second = obstacle_agnostic_skill_actions(
        params, spec, states, initial_state[:3], jnp.asarray(0.25), config
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (4, 3)
    assert np.all(np.isfinite(np.asarray(first)))


def test_persistent_adam_state_and_library_version_accumulate_across_steps() -> None:
    model, actuator, config, spec, params, initial_state = _problem()
    functions = build_persistent_skill_learner(spec, actuator, config)
    initial = functions.initialize(params, model)
    first, first_metrics = functions.step(initial, initial_state, model)
    second, second_metrics = functions.step(first, initial_state, model)
    jax.block_until_ready((second, second_metrics))

    assert bool(first_metrics.finite_update_applied)
    assert bool(second_metrics.finite_update_applied)
    assert int(first.library_version) == 1
    assert int(second.library_version) == 2
    assert int(second.cumulative_gradient_steps) == 2
    assert float(first_metrics.parameter_update_norm) > 0
    assert float(second_metrics.parameter_update_norm) > 0
    assert float(second_metrics.loss.trust) > 0
    assert np.all(np.isfinite(np.asarray(second_metrics.loss.descriptors)))

    integer_optimizer_leaves = [
        np.asarray(leaf)
        for leaf in jax.tree.leaves(second.optimizer_state)
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.integer)
    ]
    assert any(np.any(leaf == 2) for leaf in integer_optimizer_leaves)


def test_nan_micro_step_is_the_only_skip_and_preserves_last_finite_version() -> None:
    model, actuator, config, spec, params, initial_state = _problem()
    functions = build_persistent_skill_learner(spec, actuator, config)
    finite_state, _ = functions.step(functions.initialize(params, model), initial_state, model)
    skipped, metrics = functions.step(finite_state, initial_state.at[0].set(jnp.nan), model)
    jax.block_until_ready((skipped, metrics))

    assert not bool(metrics.finite_update_applied)
    assert int(skipped.library_version) == int(finite_state.library_version)
    assert int(skipped.cumulative_gradient_steps) == int(finite_state.cumulative_gradient_steps)
    for preserved, previous in zip(
        jax.tree.leaves(skipped.params), jax.tree.leaves(finite_state.params), strict=True
    ):
        np.testing.assert_array_equal(preserved, previous)
    for preserved, previous in zip(
        jax.tree.leaves(skipped.optimizer_state),
        jax.tree.leaves(finite_state.optimizer_state),
        strict=True,
    ):
        np.testing.assert_array_equal(preserved, previous)
