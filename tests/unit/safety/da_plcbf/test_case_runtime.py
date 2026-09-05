from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.da_plcbf_case_runtime import prepare_runtime_variant, profile_runtime_variant
from crazyflow.safety.da_plcbf.learner_checkpoint import save_learner_checkpoint
from crazyflow.safety.da_plcbf.online_constant_wind import build_cf21b_version_a_resources
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    build_fibonacci_skill_spec,
    initialize_skill_actor,
)
from crazyflow.safety.da_plcbf.state_conditioned_learning import (
    ReferenceContract,
    ReferenceLearningConfig,
    build_reference_skill_learner,
    build_reference_skill_learner_from_checkpoint,
    proprioceptive_state_bank,
    reference_contract_checkpoint_metadata,
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
        model_compensation=False,
        smooth_motor_bounds=False,
        learn_durations=False,
    )
    spec = build_fibonacci_skill_spec(
        policy_count=4,
        latent_size=3,
        minimum_duration=0.04,
        maximum_duration=0.12,
        horizon_duration=config.dt * config.horizon,
    )
    return ReferenceContract(
        initialize_skill_actor(jax.random.key(11), spec, config),
        resources.model,
        proprioceptive_state_bank()[0],
        config,
        ReferenceLearningConfig(),
        spec,
        resources.actuator,
    )


def test_single_anchor_cycle_retains_full_bank_objective_at_fixed_parameters() -> None:
    contract = _contract()
    single = build_reference_skill_learner(
        replace(contract, learning_config=replace(contract.learning_config, anchor_batch_size=1))
    )
    full = build_reference_skill_learner(
        replace(
            contract,
            learning_config=replace(
                contract.learning_config, anchor_batch_size=len(contract.anchors)
            ),
        )
    )
    physical = contract.anchors[0]
    model = contract.model._replace(wind_velocity=jnp.array((1.6, 0.8, 0)))
    params = contract.params.replace(output_bias=contract.params.output_bias + 0.12)
    # Start at a nonzero version and traverse a complete bank. Each state-conditioned
    # retention target must still contribute to the original full-bank objective.
    retention = [
        float(
            single.loss(params, physical, model, params, jnp.asarray(version))[
                1
            ].reference_retention
        )
        for version in range(11, 11 + len(contract.anchors))
    ]
    full_retention = float(
        full.loss(params, physical, model, params, jnp.asarray(11))[1].reference_retention
    )
    assert max(retention) - min(retention) > 1e-8
    np.testing.assert_allclose(np.mean(retention), full_retention, rtol=2e-5, atol=1e-10)


def test_microbatch_revision_preserves_adam_teacher_and_completed_resume(tmp_path: Path) -> None:
    contract = _contract()
    original = build_reference_skill_learner(contract)
    model = contract.model._replace(wind_velocity=jnp.array((1.6, 0.8, 0)))
    physical = contract.anchors[0]
    persistent = original.initialize(contract.params, model)
    for _ in range(3):
        persistent, metrics = jax.block_until_ready(original.step(persistent, physical, model))
        assert bool(metrics.finite_update_applied)
    source_dir = tmp_path / "source"
    save_reference_contract(contract, source_dir / "nominal_reference")
    save_learner_checkpoint(
        persistent,
        contract.spec,
        contract.actor_config,
        contract.actuator,
        physical,
        source_dir / "checkpoint",
        metadata=reference_contract_checkpoint_metadata(source_dir / "nominal_reference"),
    )
    source, original_contract, _ = build_reference_skill_learner_from_checkpoint(
        source_dir / "checkpoint"
    )
    directory = tmp_path / "one-anchor"
    revised, revised_contract = prepare_runtime_variant(source, original_contract, directory, 1)
    for expected, actual in zip(
        jax.tree.leaves(source.state), jax.tree.leaves(revised.state), strict=True
    ):
        np.testing.assert_array_equal(expected, actual)
    assert revised_contract.learning_config.anchor_batch_size == 1
    assert revised_contract.actor_config == original_contract.actor_config
    for expected, actual in zip(
        jax.tree.leaves(
            (original_contract.params, original_contract.model, original_contract.anchors)
        ),
        jax.tree.leaves(
            (revised_contract.params, revised_contract.model, revised_contract.anchors)
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(expected, actual)
    summary = profile_runtime_variant(
        revised,
        revised_contract,
        directory,
        device=jax.devices("cpu")[0],
        warmup=2,
        repetitions=2,
        profile_components=False,
        contention_note="CPU correctness test only; no production service inference.",
    )
    assert summary["finite_completed_updates"] == 2
    assert summary["initial_library_version"] == 3
    assert summary["final_library_version"] == 5
    assert summary["every_finite_completed_update_has_bound_full_checkpoint"]
    assert not summary["paced_positive_updates_demonstrated"]
    assert summary["anchor_bank_indices_visited"] == [3, 4]
    # Resume the first measured snapshot, not the discarded warmup. It must reproduce
    # measured update two exactly, including optimizer history and the next bank anchor.
    first, _, learner = build_reference_skill_learner_from_checkpoint(
        directory / "completed_updates/u0001"
    )
    final, _, _ = build_reference_skill_learner_from_checkpoint(directory / "final_checkpoint")
    resumed, _ = jax.block_until_ready(learner.step(first.state, physical, model))
    for expected, actual in zip(
        jax.tree.leaves(final.state), jax.tree.leaves(resumed), strict=True
    ):
        np.testing.assert_array_equal(expected, actual)
    ledger = json.loads((directory / "completed_updates.json").read_text())
    assert ledger[0]["persistent_state_after_sha256"] == ledger[1]["persistent_state_before_sha256"]
    assert all(
        row["service_completed_perf_counter_ns"] > row["service_started_perf_counter_ns"]
        for row in ledger
    )
    with pytest.raises(ValueError, match="bound"):
        prepare_runtime_variant(
            replace(source, metadata={}), original_contract, tmp_path / "unbound", 1
        )
