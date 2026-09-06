"""Actuator-memory dataflow, full-state reference, persistent publication and continuation."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.actuator_dynamics import effort_lag_step, make_actuator_model
from crazyflow.safety.da_plcbf.actuator_learning import (
    ActuatorDynamicsSupport,
    ActuatorLearnerFunctions,
    ActuatorReferenceConfig,
    ActuatorReferenceContract,
    ActuatorSkillConfig,
    _braking_positive_part,
    _smooth_worst_skill,
    acceleration_to_actuator_command,
    actuator_proprioceptive_state_bank,
    actuator_reference_fingerprint,
    actuator_skill_actions,
    balanced_reference_terms,
    build_actuator_skill_learner,
    build_single_recovery_spec,
    initialize_actuator_skill_actor,
    load_actuator_learner_checkpoint,
    probe_actuator_library,
    rollout_actuator_skill_library,
    sample_actuator_training_models,
    save_actuator_learner_checkpoint,
    train_actuator_library,
)
from crazyflow.safety.da_plcbf.online_constant_wind import build_cf21b_version_a_resources
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    build_fibonacci_skill_spec,
    initialize_skill_actor,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def contract() -> ActuatorReferenceContract:
    resources = build_cf21b_version_a_resources()
    actuator = resources.actuator
    model = make_actuator_model(
        resources.model,
        L=float(actuator.arm_length),
        thrust2torque=float(actuator.thrust_to_torque),
        mixing_matrix=actuator.mixing_matrix,
        thrust_min=actuator.thrust_min,
        thrust_max=actuator.thrust_max,
        time_constants=(0.02, 0.025, 0.03, 0.035),
    )
    config = ActuatorSkillConfig(
        horizon=8,
        hidden_width=6,
        control_interval_steps=2,
        plant_substeps=1,
        initial_residual_scale=0.05,
        learn_durations=False,
    )
    spec = build_fibonacci_skill_spec(
        policy_count=3,
        latent_size=2,
        minimum_duration=0.04,
        maximum_duration=0.12,
        horizon_duration=config.horizon * config.dt,
    )
    params = initialize_actuator_skill_actor(jax.random.key(7), spec, config)
    anchors, _ = actuator_proprioceptive_state_bank(model)
    # Nontrivial speed and motor transient states, with the cached anchor batch kept small.
    selected = anchors[jnp.asarray([0, 5, 18])]
    return ActuatorReferenceContract(
        params, model, selected, spec, config, ActuatorReferenceConfig(anchor_batch_size=1)
    )


@pytest.fixture(scope="module")
def learner(contract: ActuatorReferenceContract) -> ActuatorLearnerFunctions:
    return build_actuator_skill_learner(contract)


def assert_tree_equal(left: Any, right: Any) -> None:
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_motor_observation_is_real_and_model_parameters_are_not_actor_inputs(
    contract: ActuatorReferenceContract,
) -> None:
    names = set(inspect.signature(actuator_skill_actions).parameters)
    assert names == {"params", "spec", "states", "skill_start_position", "phase", "config"}
    count, latent = contract.spec.latent_codes.shape
    assert contract.params.input_kernel.shape == (18 + latent, contract.actor_config.hidden_width)
    states = jnp.broadcast_to(contract.anchors[0], (count, 17))
    action = actuator_skill_actions(
        contract.params,
        contract.spec,
        states,
        states[0, :3],
        jnp.asarray(0.0),
        contract.actor_config,
    )
    changed = actuator_skill_actions(
        contract.params,
        contract.spec,
        states.at[:, 13].multiply(0.5),
        states[0, :3],
        jnp.asarray(0.0),
        contract.actor_config,
    )
    assert np.max(np.abs(np.asarray(action - changed))) > 1e-6
    old = initialize_skill_actor(jax.random.key(7), contract.spec, contract.actor_config)
    with pytest.raises(ValueError, match="17-state actor input_kernel"):
        actuator_skill_actions(
            old, contract.spec, states, states[0, :3], jnp.asarray(0.0), contract.actor_config
        )


def test_teacher_uses_identical_full_state_and_is_immutable(
    contract: ActuatorReferenceContract, learner: ActuatorLearnerFunctions
) -> None:
    initial = contract.anchors[-1].at[13].multiply(0.8)
    fingerprint = actuator_reference_fingerprint(contract)
    rollout = learner.rollout(contract.params, initial, contract.model)
    np.testing.assert_array_equal(
        np.asarray(rollout.states[:, 0]), np.broadcast_to(np.asarray(initial), (3, 17))
    )
    nominal = learner.loss(contract.params, initial, contract.model, contract.params)[1]
    assert float(nominal.trajectory_tracking) == pytest.approx(0.0, abs=1e-12)
    assert float(nominal.velocity_tracking) == pytest.approx(0.0, abs=1e-12)
    assert float(nominal.reference_retention) == pytest.approx(0.0, abs=1e-12)
    assert float(nominal.motor_state_tracking) == pytest.approx(0.0, abs=1e-12)
    changed_model = contract.model._replace(
        effectiveness=jnp.asarray([0.7, 1.0, 1.0, 1.0]),
        time_constants=contract.model.time_constants * 3,
    )
    changed = learner.loss(contract.params, initial, changed_model, contract.params)[1]
    assert float(changed.trajectory_tracking + changed.velocity_tracking) > 1e-8
    assert float(changed.reference_retention) > 0
    assert actuator_reference_fingerprint(contract) == fingerprint
    assert contract.learning_config.motor_state_target_weight == 0


def test_rollout_holds_commands_and_advances_motor_and_body_together(
    contract: ActuatorReferenceContract,
) -> None:
    initial = contract.anchors[-1]
    rollout = rollout_actuator_skill_library(
        contract.params, contract.spec, initial, contract.model, contract.actor_config
    )
    assert rollout.states.shape == (3, 9, 17)
    assert rollout.commands.shape == (3, 8, 4)
    assert rollout.valid.shape == (3,)
    np.testing.assert_array_equal(
        np.asarray(rollout.commands[:, ::2]), np.asarray(rollout.commands[:, 1::2])
    )
    first = effort_lag_step(
        jnp.broadcast_to(initial, (3, 17)),
        rollout.commands[:, 0],
        contract.model,
        contract.actor_config.dt,
    )
    np.testing.assert_allclose(
        np.asarray(first), np.asarray(rollout.states[:, 1]), rtol=1e-6, atol=2e-7
    )
    # The inverse uses the complete .04 s command hold even though body integration is .02 s.
    behavior = actuator_skill_actions(
        contract.params,
        contract.spec,
        jnp.broadcast_to(initial, (3, 17)),
        initial[:3],
        jnp.asarray(0.0),
        contract.actor_config,
    )
    allocation = acceleration_to_actuator_command(
        behavior, jnp.broadcast_to(initial, (3, 17)), contract.model, contract.actor_config
    )
    np.testing.assert_allclose(
        np.asarray(allocation.command), np.asarray(rollout.commands[:, 0]), rtol=1e-6, atol=2e-7
    )


def test_reference_loss_directional_gradient_matches_finite_difference(
    contract: ActuatorReferenceContract, learner: ActuatorLearnerFunctions
) -> None:
    model = contract.model._replace(time_constants=contract.model.time_constants * 1.6)
    initial = contract.anchors[0].at[7].set(0.4)

    def objective(delta: jax.Array) -> jax.Array:
        params = contract.params.replace(output_bias=contract.params.output_bias.at[0].add(delta))
        return learner.loss(params, initial, model, contract.params)[0]

    point = jnp.asarray(0.08)
    derivative = float(jax.grad(objective)(point))
    h = 1e-3
    numeric = float((objective(point + h) - objective(point - h)) / (2 * h))
    assert np.isfinite(derivative)
    assert derivative == pytest.approx(numeric, rel=0.03, abs=2e-5)


def test_finite_updates_always_publish_and_nonfinite_updates_preserve_adam(
    contract: ActuatorReferenceContract, learner: ActuatorLearnerFunctions
) -> None:
    model = contract.model._replace(time_constants=contract.model.time_constants * 1.7)
    state = learner.initialize(contract.params, contract.model)
    next_state, metrics = learner.step(state, contract.anchors[-1], model)
    assert bool(metrics.finite_update_applied)
    assert int(next_state.library_version) == int(next_state.cumulative_gradient_steps) == 1
    assert float(metrics.parameter_update_norm) > 0
    assert_tree_equal(next_state.previous_params, state.params)
    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree.leaves(state.optimizer_state),
            jax.tree.leaves(next_state.optimizer_state),
            strict=True,
        )
    )
    invalid, invalid_metrics = learner.step(
        next_state, contract.anchors[-1].at[13].set(jnp.nan), model
    )
    assert not bool(invalid_metrics.finite_update_applied)
    assert int(invalid.library_version) == 1
    assert_tree_equal(invalid.params, next_state.params)
    assert_tree_equal(invalid.optimizer_state, next_state.optimizer_state)
    assert_tree_equal(invalid.previous_params, next_state.previous_params)


def test_finite_update_is_not_rejected_when_its_objective_worsens(
    contract: ActuatorReferenceContract,
) -> None:
    config = replace(contract.actor_config, learning_rate=10.0)
    learner = build_actuator_skill_learner(contract, config)
    state = learner.initialize(contract.params, contract.model)
    initial = contract.anchors[1]
    before = float(learner.loss(state.params, initial, contract.model, state.previous_params)[0])
    following, metrics = learner.step(state, initial, contract.model)
    after = float(
        learner.loss(following.params, initial, contract.model, following.previous_params)[0]
    )
    assert np.isfinite(after) and after > before
    assert bool(metrics.finite_update_applied)
    assert int(following.library_version) == 1


def test_checkpoint_restores_identical_next_adam_update(
    contract: ActuatorReferenceContract, learner: ActuatorLearnerFunctions, tmp_path: Path
) -> None:
    model = contract.model._replace(time_constants=contract.model.time_constants * 1.8)
    physical = contract.anchors[-1]
    state = learner.initialize(contract.params, contract.model)
    state, _ = learner.step(state, physical, model)
    stem = tmp_path / "actuator_v1"
    save_actuator_learner_checkpoint(state, contract, physical, stem, metadata={"seed": 7})
    loaded = load_actuator_learner_checkpoint(stem)
    assert_tree_equal(loaded.state, state)
    assert_tree_equal(loaded.physical_state, physical)
    assert actuator_reference_fingerprint(loaded.contract) == actuator_reference_fingerprint(
        contract
    )
    resumed = build_actuator_skill_learner(loaded.contract, loaded.config)
    expected, _ = learner.step(state, physical, model)
    actual, _ = resumed.step(loaded.state, loaded.physical_state, model)
    assert_tree_equal(expected, actual)
    with pytest.raises(FileExistsError):
        save_actuator_learner_checkpoint(state, contract, physical, stem)
    manifest = json.loads(stem.with_suffix(".json").read_text())
    manifest["state_size"] = 13
    stem.with_suffix(".json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="observation schema"):
        load_actuator_learner_checkpoint(stem)


def test_bank_contains_moving_perturbed_and_consistent_motor_transients(
    contract: ActuatorReferenceContract,
) -> None:
    bank, labels = actuator_proprioceptive_state_bank(contract.model)
    assert bank.shape == (27, 17)
    assert len(labels) == 27
    assert np.count_nonzero(np.linalg.norm(np.asarray(bank[:, 7:10]), axis=1) > 0.1) > 20
    assert any("pitch" in label for label in labels)
    assert sum("motor_transient" in label for label in labels) == 12
    motor = bank[0, 13:17]
    command = jnp.clip(1.25 * motor, contract.model.command_lower, contract.model.command_upper)
    expected = effort_lag_step(bank[0], command, contract.model, 0.02, substeps=4)
    np.testing.assert_array_equal(np.asarray(bank[15]), np.asarray(expected))


def test_single_recovery_and_dr_training_are_explicit_and_budgeted(
    contract: ActuatorReferenceContract,
) -> None:
    spec = build_single_recovery_spec(latent_size=2)
    params = initialize_actuator_skill_actor(jax.random.key(41), spec, contract.actor_config)
    single = replace(contract, params=params, spec=spec)
    models = sample_actuator_training_models(
        contract.model,
        seed=41,
        count=2,
        support=ActuatorDynamicsSupport((0.85, 1.0), (0.02, 0.045)),
    )
    replay = sample_actuator_training_models(
        contract.model,
        seed=41,
        count=2,
        support=ActuatorDynamicsSupport((0.85, 1.0), (0.02, 0.045)),
    )
    assert_tree_equal(models, replay)
    state, report = train_actuator_library(
        single, seed=41, steps=2, models=models, initial_params=params
    )
    assert report["gradient_evaluations"] == 2
    assert report["finite_updates_published"] == 2
    assert report["student_integration_steps"] == 2 * 2 * 1 * 8
    assert report["training_model_count"] == 2
    assert report["first_step_seconds_including_compile"] > 0
    learner = build_actuator_skill_learner(single)
    metrics = learner.loss(state.params, single.anchors[1], models[0], state.previous_params)[1]
    assert float(metrics.diversity) == float(metrics.pairwise) == 0
    assert metrics.per_skill_position_error.shape == (1,)


def test_probe_reports_per_state_and_per_skill_motor_behavior_without_claiming_safety(
    contract: ActuatorReferenceContract,
) -> None:
    result = probe_actuator_library(
        contract.params, contract, contract.model, labels=("rest", "moving", "transient")
    )
    assert result["policy_count"] == result["state_count"] == 3
    assert result["maximum_position_tracking_rmse_m"] == pytest.approx(0, abs=1e-12)
    assert len(result["per_state"]) == 3
    for entry in result["per_state"]:
        assert len(entry["initial_state"]) == 17
        assert len(entry["terminal_speed_mps"]) == 3
        assert len(entry["command_saturation_fraction"]) == 3
    assert "no obstacle safety guarantee" in result["scope"]


def test_absolute_braking_revision_changes_only_bootstrap_terminal_penalty(
    contract: ActuatorReferenceContract, learner: ActuatorLearnerFunctions
) -> None:
    revised = replace(
        contract, learning_config=replace(contract.learning_config, reference_braking_excess=False)
    )
    revised_learner = build_actuator_skill_learner(revised)
    initial = contract.anchors[1]
    original = learner.loss(contract.params, initial, contract.model, contract.params)[1]
    changed = revised_learner.loss(contract.params, initial, contract.model, contract.params)[1]
    assert float(original.terminal_braking) == pytest.approx(0.0, abs=1e-9)
    assert float(changed.terminal_braking) > 0.0
    for name in original._fields:
        if name not in {"total", "terminal_braking"}:
            np.testing.assert_allclose(
                np.asarray(getattr(original, name)),
                np.asarray(getattr(changed, name)),
                atol=1e-7,
                rtol=1e-6,
            )
    assert float(changed.total - original.total) == pytest.approx(
        float(changed.terminal_braking) * contract.actor_config.terminal_braking_weight, rel=1e-5
    )
    assert actuator_reference_fingerprint(revised) != actuator_reference_fingerprint(contract)


def test_opt_in_current_gain_tuning_keeps_original_nominal_teacher_immutable(
    contract: ActuatorReferenceContract, learner: ActuatorLearnerFunctions
) -> None:
    current = replace(contract.actor_config, attitude_gain=4e-4, angular_rate_gain=3e-4)
    with pytest.raises(ValueError, match="allow_reference_gain_mismatch"):
        build_actuator_skill_learner(contract, current)
    initial = contract.anchors[1]
    fingerprint = actuator_reference_fingerprint(contract)
    teacher_before = learner.rollout(contract.params, initial, contract.model).states
    current = replace(current, allow_reference_gain_mismatch=True)
    tuned = build_actuator_skill_learner(contract, current)
    metrics = tuned.loss(contract.params, initial, contract.model, contract.params)[1]
    assert float(metrics.trajectory_tracking + metrics.velocity_tracking) > 1e-8
    teacher_after = learner.rollout(contract.params, initial, contract.model).states
    np.testing.assert_array_equal(np.asarray(teacher_before), np.asarray(teacher_after))
    assert actuator_reference_fingerprint(contract) == fingerprint
    assert contract.actor_config.attitude_gain == 8e-4
    assert contract.actor_config.angular_rate_gain == 2e-4


def test_compile_unroll_preserves_rollout_and_adam_with_immutable_teacher(
    contract: ActuatorReferenceContract, learner: ActuatorLearnerFunctions, tmp_path: Path
) -> None:
    fingerprint = actuator_reference_fingerprint(contract)
    config = replace(contract.actor_config, rollout_scan_unroll=2)
    current = build_actuator_skill_learner(contract, config)
    initial = contract.anchors[2]
    model = contract.model._replace(
        effectiveness=contract.model.effectiveness * jnp.asarray([0.7, 1.0, 0.85, 1.0]),
        time_constants=contract.model.time_constants * jnp.asarray([3.0, 1.6, 2.0, 1.0]),
    )
    state = learner.initialize(contract.params, contract.model)
    for actual, expected in (
        (
            current.rollout(state.params, initial, model),
            learner.rollout(state.params, initial, model),
        ),
        (current.step(state, initial, model), learner.step(state, initial, model)),
    ):
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=5e-5, atol=2e-6)
    assert actuator_reference_fingerprint(contract) == fingerprint
    assert contract.actor_config.rollout_scan_unroll == 1
    save_actuator_learner_checkpoint(state, contract, initial, tmp_path / "unroll2", config=config)
    restored = load_actuator_learner_checkpoint(tmp_path / "unroll2")
    assert restored.config.rollout_scan_unroll == 2
    assert restored.contract.actor_config.rollout_scan_unroll == 1
    assert actuator_reference_fingerprint(restored.contract) == fingerprint
    assert_tree_equal(restored.state, state)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="rollout_scan_unroll"):
            replace(config, rollout_scan_unroll=invalid).validate()


def test_declared_braking_huber_has_continuous_derivative_at_both_joins() -> None:
    delta = 0.01
    values = jnp.asarray([-0.02, 0.0, 0.005, 0.01, 0.02])
    actual = _braking_positive_part(values, delta)
    derivative = jax.vmap(jax.grad(lambda x: _braking_positive_part(x, delta)))(values)
    np.testing.assert_allclose(actual, [0.0, 0.0, 0.00125, 0.005, 0.015], atol=1e-8)
    np.testing.assert_allclose(derivative, [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-7)
    epsilon = 1e-6
    for join in (0.0, delta):
        sides = jnp.asarray([join - epsilon, join, join + epsilon])
        slopes = jax.vmap(jax.grad(lambda x: _braking_positive_part(x, delta)))(sides)
        assert np.ptp(np.asarray(slopes)) < 2 * epsilon / delta
    np.testing.assert_array_equal(_braking_positive_part(values, 0.0), jax.nn.relu(values))
    for invalid in (-0.01, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="reference_braking_huber_delta"):
            ActuatorReferenceConfig(reference_braking_huber_delta=invalid).validate()


def test_float64_physical_snapshot_round_trips_without_enabling_jax_x64(
    contract: ActuatorReferenceContract, learner: ActuatorLearnerFunctions, tmp_path: Path
) -> None:
    physical = np.asarray(contract.anchors[-1], dtype=np.float64).copy()
    physical += np.arange(1, 18, dtype=np.float64) * 1e-10
    assert np.any(physical != physical.astype(np.float32).astype(np.float64))
    state = learner.initialize(contract.params, contract.model)
    with jax.enable_x64(False):
        save_actuator_learner_checkpoint(state, contract, physical, tmp_path / "physical64")
        restored = load_actuator_learner_checkpoint(tmp_path / "physical64")
    assert isinstance(restored.physical_state, np.ndarray)
    assert restored.physical_state.dtype == np.dtype("float64")
    np.testing.assert_array_equal(restored.physical_state, physical)
    assert not restored.physical_state.flags.writeable
    assert_tree_equal(restored.state, state)


@pytest.mark.parametrize("mode", ["balanced_reference", "balanced_reference_braking"])
def test_balanced_reference_teacher_is_stationary_with_absolute_auxiliaries_present(
    contract: ActuatorReferenceContract, mode: str
) -> None:
    revised = replace(
        contract, learning_config=replace(contract.learning_config, objective_mode=mode)
    )
    learner = build_actuator_skill_learner(revised)
    initial = contract.anchors[-1]
    value, gradient = jax.value_and_grad(
        lambda params: learner.loss(params, initial, contract.model, contract.params)[0]
    )(contract.params)
    assert float(value) == pytest.approx(0.0, abs=1e-10)
    for leaf in jax.tree.leaves(gradient):
        np.testing.assert_allclose(leaf, 0, atol=1e-9)
    # Absolute motor/attitude costs are deliberately still configured in the shared actor.
    assert contract.actor_config.action_weight > 0
    assert contract.actor_config.attitude_weight > 0
    assert (
        "motor_effort"
        not in learner.loss(contract.params, initial, contract.model, contract.params)[1]._fields
    )
    following, metrics = learner.step(
        learner.initialize(contract.params, contract.model), initial, contract.model
    )
    assert bool(metrics.finite_update_applied)
    assert int(following.library_version) == 1
    assert_tree_equal(following.params, contract.params)


def test_balanced_recovery_prioritizes_worse_skill_and_has_stationary_braking_joins(
    contract: ActuatorReferenceContract,
) -> None:
    # Equal averaging gives every skill identical weight; smooth worst-skill does not.
    weights = jax.grad(lambda values: _smooth_worst_skill(values, 0.25))(
        jnp.asarray([0.01, 0.9, 0.01])
    )
    assert float(weights[1]) > 20 * float(weights[0])
    assert float(jnp.sum(weights)) == pytest.approx(1.0)
    settings = replace(contract.learning_config, objective_mode="balanced_reference")
    shape = (1, 3, contract.actor_config.horizon + 1, 17)
    reference = jnp.broadcast_to(contract.anchors[0], shape)
    reference = reference.at[..., 7].set(0.5)

    def terminal(delta: jax.Array) -> jax.Array:
        states = reference.at[..., -1, 7].add(delta)
        return jnp.sum(
            balanced_reference_terms(states, reference, contract.actor_config, settings)[3]
        )

    assert float(terminal(jnp.asarray(0.0))) == 0
    assert float(jax.grad(terminal)(jnp.asarray(0.0))) == 0
    assert float(terminal(jnp.asarray(0.1))) > float(terminal(jnp.asarray(-0.1))) > 0
    later = reference.at[..., -1, 0].add(0.2)
    early = reference.at[..., 1, 0].add(0.2)
    assert np.all(
        np.asarray(balanced_reference_terms(later, reference, contract.actor_config, settings)[2])
        == 0
    )
    assert np.all(
        np.asarray(balanced_reference_terms(early, reference, contract.actor_config, settings)[2])
        > 0
    )


def test_balanced_config_is_fingerprinted_and_round_trips_without_altering_teacher(
    contract: ActuatorReferenceContract, learner: ActuatorLearnerFunctions, tmp_path: Path
) -> None:
    repaired = replace(
        contract,
        learning_config=replace(contract.learning_config, objective_mode="balanced_reference"),
    )
    assert actuator_reference_fingerprint(repaired) != actuator_reference_fingerprint(contract)
    assert_tree_equal(repaired.params, contract.params)
    state = learner.initialize(contract.params, contract.model)
    save_actuator_learner_checkpoint(state, repaired, contract.anchors[0], tmp_path / "balanced")
    restored = load_actuator_learner_checkpoint(tmp_path / "balanced")
    assert restored.contract.learning_config.objective_mode == "balanced_reference"
    assert actuator_reference_fingerprint(restored.contract) == actuator_reference_fingerprint(
        repaired
    )
    assert_tree_equal(restored.state, state)
    for kwargs in (
        {"objective_mode": "unknown"},
        {"recovery_balance_temperature": 0},
        {"recovery_prefix_weight": -1},
        {"recovery_prefix_fraction": float("nan")},
    ):
        with pytest.raises(ValueError):
            replace(contract.learning_config, **kwargs).validate()


def test_braking_priority_changes_only_the_declared_reference_terminal_weight(
    contract: ActuatorReferenceContract,
) -> None:
    base = replace(
        contract,
        learning_config=replace(
            contract.learning_config, objective_mode="balanced_reference", anchor_batch_size=0
        ),
    )
    revised = replace(
        base,
        learning_config=replace(
            base.learning_config,
            objective_mode="balanced_reference_braking",
            recovery_braking_priority=10.0,
        ),
    )
    initial = contract.anchors[1]
    model = contract.model._replace(effectiveness=jnp.asarray([0.7, 1.0, 1.0, 1.0]))
    original = build_actuator_skill_learner(base).loss(
        contract.params, initial, model, contract.params
    )[1]
    changed = build_actuator_skill_learner(revised).loss(
        contract.params, initial, model, contract.params
    )[1]
    assert float(original.terminal_braking) > 0
    assert float(changed.total - original.total) == pytest.approx(
        9 * contract.actor_config.terminal_braking_weight * float(original.terminal_braking),
        rel=2e-5,
        abs=1e-6,
    )
    for name in original._fields:
        if name != "total":
            np.testing.assert_allclose(
                getattr(original, name), getattr(changed, name), rtol=2e-5, atol=1e-6
            )
