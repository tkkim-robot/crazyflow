"""Phase continuity, full-path checks and episode-owned backup state."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.actuator_backup import (
    CommittedBackupController,
    checked_paths,
    context_rollouts,
)
from crazyflow.safety.da_plcbf.actuator_learning import (
    ActuatorSkillConfig,
    initialize_actuator_skill_actor,
    rollout_actuator_skill_library,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_study import (
    build_actuator_controller,
    nominal_actuator_model,
)
from crazyflow.safety.da_plcbf.continuous_version_a import RuntimeObstacleTrajectories
from crazyflow.safety.da_plcbf.persistent_skill_learner import build_fibonacci_skill_spec
from crazyflow.safety.da_plcbf.version_a_barriers import RigidBodySafetySet


@pytest.fixture(scope="module")
def inputs() -> tuple[Any, ...]:
    model = nominal_actuator_model()
    actor = ActuatorSkillConfig(horizon=8, hidden_width=4, control_interval_steps=2)
    config = ActuatorFilterConfig(horizon=8, command_hold_steps=2, sqp_iterations=0)
    spec = build_fibonacci_skill_spec(
        policy_count=3,
        latent_size=2,
        minimum_duration=0.04,
        maximum_duration=0.12,
        horizon_duration=0.16,
    )
    params = initialize_actuator_skill_actor(jax.random.key(7), spec, actor)
    hover = float(model.body.mass * -model.body.gravity_vec[2] / 4)
    state = jnp.asarray([0, 0, 1.4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, *([hover] * 4)])
    obstacles = RuntimeObstacleTrajectories(
        jnp.empty((9, 0, 3)), jnp.empty(0), jnp.empty((9, 0), dtype=bool)
    )
    safety = RigidBodySafetySet(
        jnp.empty((0, 3)),
        jnp.empty(0),
        jnp.empty(0, dtype=bool),
        jnp.asarray([-5.0, -5.0, 0.15]),
        jnp.asarray([5.0, 5.0, 4.0]),
        jnp.asarray(3.5),
        jnp.asarray(12.0),
        jnp.asarray(0.9),
    )
    return state, params, spec, model, actor, config, obstacles, safety


def test_phase_continuation_matches_original_prediction(inputs: tuple[Any, ...]) -> None:
    state, params, spec, model, actor, *_ = inputs
    original = jax.block_until_ready(
        rollout_actuator_skill_library(params, spec, state, model, actor)
    )
    skill, offset = 1, 2
    continued = jax.block_until_ready(
        context_rollouts(
            params, spec, original.states[skill, offset], model, actor, state[:3], jnp.asarray(0.04)
        )
    )
    np.testing.assert_allclose(
        continued.states[skill, :7], original.states[skill, offset:], atol=3e-6, rtol=2e-6
    )
    restarted = context_rollouts(
        params,
        spec,
        original.states[skill, offset],
        model,
        actor,
        original.states[skill, offset, :3],
        jnp.asarray(0.0),
    )
    assert (
        np.max(abs(np.asarray(restarted.commands[skill]) - np.asarray(continued.commands[skill])))
        > 1e-4
    )


def test_path_check_rejects_interior_operational_violation(inputs: tuple[Any, ...]) -> None:
    state, params, spec, model, actor, config, obstacles, safety = inputs
    batch = context_rollouts(params, spec, state, model, actor, state[:3], jnp.asarray(0.0))
    valid, _, _ = checked_paths(batch, model, obstacles, safety, config)
    assert np.all(valid)
    damaged = batch._replace(states=batch.states.at[1, 4, 7].set(10.0))
    valid, _, _ = checked_paths(damaged, model, obstacles, safety, config)
    np.testing.assert_array_equal(valid, [True, False, True])


def test_mismatched_skill_bank_is_rejected(inputs: tuple[Any, ...]) -> None:
    from types import SimpleNamespace

    state, params, spec, model, actor, config, obstacles, safety = inputs
    functions = build_actuator_controller(spec, actor, config)
    bad = SimpleNamespace(
        nominal=functions.nominal,
        emergency=functions.emergency,
        candidates=functions.candidates,
        controller=lambda *args: SimpleNamespace(
            certificates=SimpleNamespace(rollouts=SimpleNamespace(states=np.empty((7, 9, 17))))
        ),
    )
    governor = CommittedBackupController(bad, spec, actor, config)
    with pytest.raises(ValueError, match="matching skill bank"):
        governor.controller_at(
            state,
            params,
            model,
            obstacles,
            safety,
            jnp.asarray(0),
            state[:3],
            when=0.0,
            commit=True,
        )


def test_warmup_does_not_publish_backup_and_updates_do_not_mutate_it(
    inputs: tuple[Any, ...],
) -> None:
    state, params, spec, model, actor, config, obstacles, safety = inputs
    functions = build_actuator_controller(spec, actor, config)
    governor = CommittedBackupController(functions, spec, actor, config)
    arguments = (state, params, model, obstacles, safety, jnp.asarray(0), state[:3])
    governor.controller_at(*arguments, when=0.0, commit=False)
    assert governor.backup is None and governor.generation == 0
    step = governor.controller_at(*arguments, when=0.0, commit=True)
    saved = governor.backup
    assert saved is not None and step.backup_audit["has_checked_plan"]
    assert saved.started_at == config.command_period
    updated = params.replace(velocity_offsets=params.velocity_offsets + 0.1)
    later = governor.controller_at(
        step.applied.nodes[-1],
        updated,
        model,
        obstacles,
        safety,
        jnp.asarray(0),
        state[:3],
        when=0.04,
        commit=True,
    )
    assert governor.backup.started_at == saved.started_at
    assert governor.backup.skill_index == saved.skill_index
    assert governor.backup.certified_until > saved.certified_until
    assert later.backup_audit["successor_backup_kind"] == "retained"
    np.testing.assert_array_equal(governor.backup.params.velocity_offsets, params.velocity_offsets)
    expected = functions.candidates(step.applied.nodes[-1], updated, state[:3], model)
    np.testing.assert_allclose(
        later.certificates.rollouts.states, expected.states, atol=3e-6, rtol=2e-6
    )


def test_stored_tail_is_not_discarded_for_an_unchecked_horizon_extension(
    inputs: tuple[Any, ...],
) -> None:
    state, params, spec, model, actor, config, _, safety = inputs
    functions = build_actuator_controller(spec, actor, config)
    governor = CommittedBackupController(functions, spec, actor, config)
    path = context_rollouts(params, spec, state, model, actor, state[:3], jnp.asarray(0.0))
    centers = jnp.broadcast_to(jnp.asarray([8.0, 8.0, 8.0]), (9, 1, 3))
    centers = centers.at[7:, 0].set(path.states[1, -1, :3])
    obstacles = RuntimeObstacleTrajectories(
        centers, jnp.asarray([0.05]), jnp.ones((9, 1), dtype=bool)
    )
    arguments = (state, params, state[:3], jnp.asarray(0.0), model, obstacles, safety)
    full = governor._current(*arguments)
    tail = governor._remaining(*arguments, jnp.asarray(4))
    expired = governor._remaining(*arguments, jnp.asarray(1))
    assert not bool(full[0][1])
    assert bool(tail[0][1])
    assert not bool(expired[0][1])


def test_memory_snapshot_covers_retained_parameters_phase_deadline_and_generation(
    inputs: tuple[Any, ...],
) -> None:
    from dataclasses import replace

    from crazyflow.safety.da_plcbf.actuator_backup import BackupSnapshot
    from crazyflow.safety.da_plcbf.actuator_experiment import _hash_tree

    state, params, spec, _, actor, config, *_ = inputs
    governor = CommittedBackupController(
        build_actuator_controller(spec, actor, config), spec, actor, config
    )
    governor.generation = 9
    governor.backup = BackupSnapshot(params, state[:3], 1.2, 1, 7, 2.4)
    original = governor.backup
    memory = governor.memory_state()
    assert governor.backup is original
    assert memory["generation"] == 9
    assert memory["backup"]["params"] is params
    assert memory["backup"]["started_at"] == 1.2
    assert memory["backup"]["certified_until"] == 2.4
    baseline = _hash_tree(memory)
    for field, value in (
        ("params", params.replace(velocity_offsets=params.velocity_offsets + 0.1)),
        ("anchor", state[:3] + 0.1),
        ("started_at", 1.24),
        ("skill_index", 2),
        ("generation", 8),
        ("certified_until", 2.44),
    ):
        governor.backup = replace(original, **{field: value})
        assert _hash_tree(governor.memory_state()) != baseline
