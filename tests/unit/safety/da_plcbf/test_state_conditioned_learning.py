from __future__ import annotations

import inspect
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.learner_checkpoint import save_learner_checkpoint
from crazyflow.safety.da_plcbf.online_constant_wind import build_cf21b_version_a_resources
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    build_fibonacci_skill_spec,
    build_persistent_skill_learner,
    initialize_skill_actor,
)
from crazyflow.safety.da_plcbf.state_conditioned_learning import (
    ReferenceContract,
    ReferenceLearningConfig,
    build_reference_skill_learner,
    build_reference_skill_learner_from_checkpoint,
    proprioceptive_state_bank,
    reference_contract_checkpoint_metadata,
    reference_trajectory_loss,
    save_reference_contract,
)

if TYPE_CHECKING:
    from pathlib import Path


def _contract() -> ReferenceContract:
    resources = build_cf21b_version_a_resources()
    config = PersistentSkillConfig(
        horizon=8,
        hidden_width=8,
        control_interval_steps=2,
        model_compensation=True,
        smooth_motor_bounds=False,
        velocity_offset_limit=0.15,
        learn_durations=False,
    )
    spec = build_fibonacci_skill_spec(
        policy_count=4,
        latent_size=3,
        minimum_duration=0.04,
        maximum_duration=0.12,
        horizon_duration=config.dt * config.horizon,
    )
    params = initialize_skill_actor(jax.random.key(7), spec, config)
    bank, _ = proprioceptive_state_bank()
    return ReferenceContract(
        params,
        resources.model,
        bank,
        config,
        ReferenceLearningConfig(anchor_batch_size=2),
        spec,
        resources.actuator,
    )


def test_reference_is_conditioned_on_velocity_and_immutable_across_changed_models() -> None:
    contract = _contract()
    learner = build_reference_skill_learner(contract)
    moving = contract.anchors[0].at[7].set(2.06)
    nominal = learner.loss(contract.params, moving, contract.model, contract.params)[1]
    assert float(nominal.trajectory_tracking) == pytest.approx(0.0, abs=1e-12)
    assert float(nominal.velocity_tracking) == pytest.approx(0.0, abs=1e-12)
    assert float(nominal.reference_retention) == pytest.approx(0.0, abs=1e-12)
    assert float(nominal.descriptor_target) == pytest.approx(0.0, abs=1e-12)
    # The same fast state does not magically satisfy a rest-designed displacement target.
    original = build_persistent_skill_learner(
        contract.spec, contract.actuator, contract.actor_config
    )
    original_metrics = original.loss(contract.params, moving, contract.model, contract.params)[1]
    assert float(original_metrics.descriptor_target) > 1e-3
    changed_model = contract.model._replace(wind_velocity=jnp.asarray([2.0, 0.5, 0.0]))
    changed = learner.loss(contract.params, moving, changed_model, contract.params)[1]
    assert float(changed.trajectory_tracking + changed.velocity_tracking) > 1e-7
    assert float(changed.reference_retention) > 0.0


def test_reference_objective_directional_derivative_matches_finite_difference() -> None:
    contract = _contract()
    learner = build_reference_skill_learner(contract)
    model = contract.model._replace(wind_velocity=jnp.asarray([0.7, -0.3, 0.0]))
    state = contract.anchors[0].at[7].set(0.7)
    params = contract.params.replace(output_bias=contract.params.output_bias + 0.12)

    def objective(delta: jax.Array) -> jax.Array:
        shifted = params.replace(output_bias=params.output_bias.at[0].add(delta))
        return learner.loss(shifted, state, model, contract.params)[0]

    derivative = float(jax.grad(objective)(jnp.asarray(0.0)))
    h = 1e-3
    finite_difference = float((objective(jnp.asarray(h)) - objective(jnp.asarray(-h))) / (2 * h))
    np.testing.assert_allclose(derivative, finite_difference, rtol=0.03, atol=2e-5)


def test_bounded_offsets_and_frozen_duration_preserve_persistent_optimizer_history() -> None:
    contract = _contract()
    original_config = replace(contract.actor_config, learning_rate=0.1, learn_durations=True)
    original = build_persistent_skill_learner(contract.spec, contract.actuator, original_config)
    state = original.initialize(contract.params, contract.model)
    state, _ = original.step(state, contract.anchors[0], contract.model)
    fixed_duration = np.asarray(state.params.duration_offsets).copy()
    assert np.linalg.norm(fixed_duration) > 0.0
    bounded = replace(original_config, velocity_offset_limit=0.01, learn_durations=False)
    learner = build_persistent_skill_learner(contract.spec, contract.actuator, bounded)
    following, metrics = learner.step(state, contract.anchors[0], contract.model)
    assert bool(metrics.finite_update_applied)
    assert int(following.library_version) == int(state.library_version) + 1
    np.testing.assert_array_equal(following.params.duration_offsets, fixed_duration)
    assert np.max(np.abs(following.params.velocity_offsets)) <= 0.010001
    integer_leaves = [
        np.asarray(x)
        for x in jax.tree.leaves(following.optimizer_state)
        if np.issubdtype(np.asarray(x).dtype, np.integer)
    ]
    assert any(np.any(x == 2) for x in integer_leaves)


def test_reference_contract_resume_preserves_next_bank_update(tmp_path: Path) -> None:
    contract = _contract()
    learner = build_reference_skill_learner(contract)
    state = learner.initialize(contract.params, contract.model)
    model = contract.model._replace(wind_velocity=jnp.asarray([1.0, 0.4, 0.0]))
    physical = contract.anchors[0].at[7].set(1.2)
    state, _ = learner.step(state, physical, model)
    save_reference_contract(contract, tmp_path / "nominal_reference")
    save_learner_checkpoint(
        state,
        contract.spec,
        contract.actor_config,
        contract.actuator,
        physical,
        tmp_path / "checkpoint",
    )
    loaded, restored_contract, restored_learner = build_reference_skill_learner_from_checkpoint(
        tmp_path / "checkpoint", device=jax.devices()[0]
    )
    assert loaded.metadata["reference_contract_binding"] == "legacy_unbound"
    expected = learner.step(state, physical, model)
    actual = restored_learner.step(loaded.state, physical, model)
    jax.block_until_ready((expected, actual))
    for before, after in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(restored_contract.model.wind_velocity, 0.0)


def test_signed_checkpoint_rejects_different_teacher_bank_with_same_spec(tmp_path: Path) -> None:
    contract = _contract()
    learner = build_reference_skill_learner(contract)
    initial = learner.initialize(contract.params, contract.model)
    save_reference_contract(contract, tmp_path / "nominal_reference")
    save_learner_checkpoint(
        initial,
        contract.spec,
        contract.actor_config,
        contract.actuator,
        contract.anchors[0],
        tmp_path / "checkpoint",
        metadata=reference_contract_checkpoint_metadata(tmp_path / "nominal_reference"),
    )
    matched, _, _ = build_reference_skill_learner_from_checkpoint(tmp_path / "checkpoint")
    assert matched.metadata["reference_contract_binding"] == "verified_npz_and_manifest_sha256"
    different = replace(contract, anchors=contract.anchors.at[0, 7].set(0.123))
    save_reference_contract(different, tmp_path / "different_reference")
    with pytest.raises(ValueError, match="checkpoint reference hash"):
        build_reference_skill_learner_from_checkpoint(
            tmp_path / "checkpoint", tmp_path / "different_reference"
        )
    manifest_path = tmp_path / "nominal_reference.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["learning_config"]["retention_weight"] += 1.0
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reference_contract_manifest_sha256"):
        build_reference_skill_learner_from_checkpoint(tmp_path / "checkpoint")


def test_bank_and_target_interfaces_do_not_receive_scene_or_safety_metadata() -> None:
    for function in (proprioceptive_state_bank, reference_trajectory_loss):
        names = inspect.signature(function).parameters
        assert not any(
            token in name
            for name in names
            for token in ("obstacle", "goal", "waypoint", "collision", "certificate")
        )
    bank, labels = proprioceptive_state_bank()
    assert bank.shape == (len(labels), 13)
    assert set(np.sign(np.asarray(bank[:, 7]))) == {-1.0, 0.0, 1.0}
    assert set(np.sign(np.asarray(bank[:, 8]))) == {-1.0, 0.0, 1.0}
    assert set(np.sign(np.asarray(bank[:, 9]))) == {-1.0, 0.0, 1.0}
    assert np.max(np.abs(np.asarray(bank[:, 10:13]))) > 0.0


@pytest.mark.parametrize("trainable", ["network", "offsets"])
def test_parameter_block_mask_and_step_norm_cap_publish_finite_updates(trainable: str) -> None:
    contract = _contract()
    config = replace(
        contract.actor_config,
        learning_rate=0.01,
        trainable_parameters=trainable,
        max_parameter_update_norm=0.001,
    )
    learner = build_reference_skill_learner(contract, config)
    initial = learner.initialize(contract.params, contract.model)
    model = contract.model._replace(wind_velocity=jnp.asarray([1.0, 0.5, 0.0]))
    following, metrics = learner.step(initial, contract.anchors[0], model)
    assert bool(metrics.finite_update_applied)
    assert int(following.library_version) == 1
    assert 0.0 < float(metrics.parameter_update_norm) <= 0.001001
    for name in contract.params.__dataclass_fields__:
        frozen = (
            name == "duration_offsets"
            or (trainable == "network" and name.endswith("_offsets"))
            or (trainable == "offsets" and not name.endswith("_offsets"))
        )
        if frozen:
            np.testing.assert_array_equal(
                getattr(following.params, name), getattr(initial.params, name)
            )
