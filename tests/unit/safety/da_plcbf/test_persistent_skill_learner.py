from __future__ import annotations

import inspect
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    SkillCompetencyThresholds,
    build_fibonacci_skill_spec,
    build_persistent_skill_learner,
    initialize_skill_actor,
    obstacle_agnostic_skill_actions,
    rollout_skill_library,
    skill_library_competency,
    spatial_descriptor_losses,
)
from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator

if TYPE_CHECKING:
    from pathlib import Path


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


def test_model_compensation_cancels_known_drag_outside_behavior_saturation() -> None:
    model, _, config, spec, params, initial_state = _problem()
    states = jnp.broadcast_to(initial_state, (4, 13)).at[:, 7:10].set(jnp.asarray([0.4, -0.2, 0.0]))
    point_model = model._replace(wind_velocity=jnp.asarray([1.2, 0.3, 0.0]))
    raw = obstacle_agnostic_skill_actions(
        params, spec, states, initial_state[:3], jnp.asarray(0.5), config
    )
    compensated = obstacle_agnostic_skill_actions(
        params,
        spec,
        states,
        initial_state[:3],
        jnp.asarray(0.5),
        replace(config, model_compensation=True),
        point_model=point_model,
    )
    expected = -(model.drag_matrix @ (states[:, 7:10] - point_model.wind_velocity).T).T / model.mass
    np.testing.assert_allclose(compensated - raw, expected, rtol=2e-6, atol=1e-6)


def test_descriptor_targets_and_rollouts_obey_displacement_mean_velocity_identity() -> None:
    model, actuator, config, _, params, initial_state = _problem()
    duration = config.dt * config.horizon
    spec = build_fibonacci_skill_spec(
        policy_count=4,
        latent_size=3,
        minimum_duration=0.04,
        maximum_duration=0.08,
        horizon_duration=duration,
    )
    np.testing.assert_allclose(
        spec.target_descriptors[:, :3], duration * spec.target_descriptors[:, 3:6], atol=1e-7
    )
    rollout = rollout_skill_library(
        params, spec, initial_state, model, actuator, replace(config, smooth_motor_bounds=False)
    )
    np.testing.assert_allclose(
        rollout.descriptors[:, :3], duration * rollout.descriptors[:, 3:6], atol=2e-7
    )


def test_hard_motor_bounds_preserve_unsaturated_hover() -> None:
    model, actuator, config, spec, params, initial_state = _problem()
    spec = spec.replace(base_desired_velocities=jnp.zeros_like(spec.base_desired_velocities))
    rollout = rollout_skill_library(
        params,
        spec,
        initial_state,
        model,
        actuator,
        replace(config, smooth_motor_bounds=False, residual_scale=0.0),
    )
    np.testing.assert_allclose(
        rollout.states[:, :, :3],
        jnp.broadcast_to(initial_state[:3], rollout.states[:, :, :3].shape),
        atol=1e-7,
    )


def test_zero_initial_skill_scale_removes_directional_scaffold_but_preserves_targets() -> None:
    _, _, config, spec, _, initial_state = _problem()
    config = replace(config, initial_skill_scale=0.0, initial_residual_scale=0.0)
    params = initialize_skill_actor(jax.random.key(7), spec, config)
    states = jnp.broadcast_to(initial_state, (4, 13))
    action = obstacle_agnostic_skill_actions(
        params, spec, states, initial_state[:3], jnp.asarray(0.0), config
    )
    np.testing.assert_array_equal(params.velocity_offsets, -spec.base_desired_velocities)
    np.testing.assert_array_equal(action, 0.0)
    assert np.linalg.norm(np.asarray(spec.target_descriptors)) > 0.0
    assert np.linalg.norm(np.asarray(spec.latent_codes)) > 0.0
    moving = states.at[:, 7].set(0.2)
    braking_action = obstacle_agnostic_skill_actions(
        params, spec, moving, initial_state[:3], jnp.asarray(0.0), config
    )
    assert np.all(np.asarray(braking_action)[:, 0] < 0.0)


def test_nominal_model_compensation_cancels_vertical_wind_at_hover() -> None:
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig, waypoint_nominal_wrench
    from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step

    model, actuator, _, _, _, initial_state = _problem()
    model = model._replace(wind_velocity=jnp.asarray([0.0, 0.0, 0.8]))
    plain = waypoint_nominal_wrench(
        initial_state, initial_state[:3], jnp.zeros(3), model, actuator, QuadPolicyConfig()
    )
    compensated = waypoint_nominal_wrench(
        initial_state,
        initial_state[:3],
        jnp.zeros(3),
        model,
        actuator,
        QuadPolicyConfig(),
        model_compensation=True,
    )
    plain_next = direct_wrench_symplectic_step(initial_state, plain.wrench, model, 0.02)
    compensated_next = direct_wrench_symplectic_step(initial_state, compensated.wrench, model, 0.02)
    assert abs(float(plain_next[9])) > 1e-3
    np.testing.assert_allclose(compensated_next[7:10], 0.0, atol=1e-7)


def test_diversity_cannot_reward_redundant_mean_velocity_or_nonzero_terminal_velocity() -> None:
    config = PersistentSkillConfig()
    spatial = jnp.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    observations = jnp.concatenate((spatial, jnp.zeros((4, 6))), axis=1)
    targets = observations
    stationary = spatial_descriptor_losses(observations, targets, config)
    velocity_spread = observations.at[:, 3:].set(jnp.arange(24).reshape(4, 6))
    moving = spatial_descriptor_losses(velocity_spread, targets, config)
    np.testing.assert_array_equal(np.asarray(stationary), np.asarray(moving))
    gradients = jax.grad(
        lambda descriptors: spatial_descriptor_losses(descriptors, targets, config).diversity
    )(velocity_spread)
    np.testing.assert_array_equal(gradients[:, 3:], 0.0)
    assert np.linalg.norm(np.asarray(gradients[:, :3])) > 0.0


def test_competency_counts_actual_directions_not_latent_identities() -> None:
    model, actuator, config, spec, params, initial_state = _problem()
    config = replace(config, smooth_motor_bounds=False)
    rollout = rollout_skill_library(params, spec, initial_state, model, actuator, config)
    collapsed = rollout._replace(states=jnp.repeat(rollout.states[:1], 4, axis=0))
    diagnostics = skill_library_competency(
        collapsed, spec, config, thresholds=SkillCompetencyThresholds(minimum_displacement_m=1e-6)
    )
    assert diagnostics["policy_count"] == 4
    assert diagnostics["occupied_direction_count"] == 1
    assert diagnostics["endpoint_pairwise_mean_m"] == 0.0
    assert diagnostics["trajectory_pairwise_rms_mean_m"] == 0.0
    assert diagnostics["competent_under_declared_criteria"] is False
    assert diagnostics["terminal_speed_mps"]["maximum"] >= 0.0


def test_complete_checkpoint_resume_preserves_the_next_persistent_update(tmp_path: Path) -> None:
    from crazyflow.safety.da_plcbf.learner_checkpoint import (
        load_learner_checkpoint,
        save_learner_checkpoint,
    )

    model, actuator, config, spec, params, initial_state = _problem()
    model, actuator, spec, params, initial_state = jax.device_put(
        (model, actuator, spec, params, initial_state), jax.devices()[0]
    )
    functions = build_persistent_skill_learner(spec, actuator, config)
    persistent = functions.initialize(params, model)
    for _ in range(2):
        persistent, _ = functions.step(persistent, initial_state, model)
    stem = tmp_path / "shared_nominal_library"
    paths = save_learner_checkpoint(
        persistent,
        spec,
        config,
        actuator,
        initial_state,
        stem,
        metadata={"purpose": "common nominal-dynamics initialization"},
    )
    with np.load(paths[0], allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind in "biuf" for key in archive.files)
    loaded = load_learner_checkpoint(stem, device=jax.devices()[0])
    assert loaded.config == config
    assert loaded.metadata["purpose"] == "common nominal-dynamics initialization"
    for before, restored in zip(
        jax.tree.leaves(persistent), jax.tree.leaves(loaded.state), strict=True
    ):
        np.testing.assert_array_equal(before, restored)
    resumed_functions = build_persistent_skill_learner(loaded.spec, loaded.actuator, loaded.config)
    expected, expected_metrics = functions.step(persistent, initial_state, model)
    actual, actual_metrics = resumed_functions.step(
        loaded.state, loaded.physical_state, loaded.point_model
    )
    jax.block_until_ready((expected, actual, expected_metrics, actual_metrics))
    for before, restored in zip(
        jax.tree.leaves((expected, expected_metrics)),
        jax.tree.leaves((actual, actual_metrics)),
        strict=True,
    ):
        np.testing.assert_array_equal(before, restored)
    assert int(actual.library_version) == 3
    with pytest.raises(FileExistsError):
        save_learner_checkpoint(persistent, spec, config, actuator, initial_state, stem)


def test_checkpoint_integrity_error_is_detected_before_deserialization(tmp_path: Path) -> None:
    from crazyflow.safety.da_plcbf.learner_checkpoint import (
        load_learner_checkpoint,
        save_learner_checkpoint,
    )

    model, actuator, config, spec, params, initial_state = _problem()
    state = build_persistent_skill_learner(spec, actuator, config).initialize(params, model)
    stem = tmp_path / "corrupted"
    npz_path, _ = save_learner_checkpoint(state, spec, config, actuator, initial_state, stem)
    damaged = bytearray(npz_path.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    npz_path.write_bytes(damaged)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_learner_checkpoint(stem)
