"""Synthetic evidence tests; these fixtures are never study cases or outcome evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmark.da_plcbf_actuator_confirmation import (
    _intervention_contract,
    _recorded_inputs,
    audit_common_prefix,
    audit_prior_available,
    audit_shared_state,
    authenticate_boundary_snapshot,
    evaluate_candidate_branches,
    load_saved_run,
    main,
)
from crazyflow.safety.da_plcbf.actuator_dynamics import effort_lag_step
from crazyflow.safety.da_plcbf.actuator_experiment import (
    ActuatorEpisodeConfig,
    _filter_record,
    _hash_tree,
    _jsonable,
)
from crazyflow.safety.da_plcbf.actuator_learning import (
    ActuatorReferenceConfig,
    ActuatorReferenceContract,
    ActuatorSkillConfig,
    build_actuator_skill_learner,
    initialize_actuator_skill_actor,
    load_actuator_learner_checkpoint,
    save_actuator_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig, ActuatorRollouts
from crazyflow.safety.da_plcbf.actuator_study import (
    build_actuator_controller,
    initial_augmented_state,
    make_actuator_scene,
    nominal_actuator_model,
)
from crazyflow.safety.da_plcbf.continuous_version_a import RuntimeObstacleTrajectories
from crazyflow.safety.da_plcbf.persistent_skill_learner import build_fibonacci_skill_spec
from crazyflow.safety.da_plcbf.version_a_barriers import RigidBodySafetySet

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def checkpoint_parts() -> tuple[Any, ...]:
    model = nominal_actuator_model()
    actor = ActuatorSkillConfig(
        horizon=4, hidden_width=4, control_interval_steps=2, plant_substeps=1
    )
    spec = build_fibonacci_skill_spec(
        policy_count=2,
        latent_size=2,
        minimum_duration=0.04,
        maximum_duration=0.08,
        horizon_duration=0.08,
    )
    params = initialize_actuator_skill_actor(jax.random.key(912), spec, actor)
    body = np.asarray([0.123456789012345, 0, 1.4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    physical = initial_augmented_state(body, model).astype(np.float64)
    contract = ActuatorReferenceContract(
        params,
        model,
        jnp.asarray(physical[None]),
        spec,
        actor,
        ActuatorReferenceConfig(anchor_batch_size=1),
    )
    learner = build_actuator_skill_learner(contract)
    initial = learner.initialize(params, model)
    published = initial.replace(params=params.replace(output_bias=params.output_bias + 0.02))
    return model, contract, initial, published, physical


def _episode(
    directory: Path,
    parts: tuple[Any, ...],
    initial_checkpoint: Any,
    *,
    freeze: float | None = None,
    revert: float | None = None,
    no_fault: bool = False,
) -> tuple[Any, Path]:
    model, contract, initial, published, physical = parts
    scene = replace(make_actuator_scene(17, "navigation", "combined"), event_time=0.04)
    scene = replace(
        scene, world=replace(scene.world, config=replace(scene.world.config, duration_seconds=0.08))
    )
    if no_fault:
        scene = replace(scene, effectiveness_after=(1, 1, 1, 1), lag_multipliers_after=(1, 1, 1, 1))
    config = ActuatorEpisodeConfig(
        filter_config=ActuatorFilterConfig(horizon=4, sqp_iterations=0),
        freeze_learning_at=freeze,
        revert_control_params_at=revert,
        capture_times=(0.04,),
        save_checkpoints=True,
    )
    directory.mkdir()
    times = np.asarray([0, 0.04, 0.08], dtype=np.float64)
    states = np.repeat(physical[None], 3, axis=0)
    command = np.repeat(physical[None, 13:], 3, axis=0)
    dense = {
        "time": times,
        "state": states,
        "native_state": states.copy(),
        "command": command,
        "actual_forces": command.copy(),
        "applied_wrenches": np.zeros((3, 4)),
        "actual_effectiveness": np.ones((3, 4)),
        "actual_time_constants": np.repeat(np.asarray(model.time_constants)[None], 3, axis=0),
    }
    if not no_fault:
        dense["actual_effectiveness"][1:] = scene.effectiveness_after
    controls = {
        "time": times[:2],
        "actual_state": states[:2],
        "actual_state_sha256": np.asarray([_hash_tree(physical)] * 2),
        "controller_input_state": states[:2].astype(np.float32),
        "planned_command": command[:2].astype(np.float32),
        "command_applied": np.asarray([True, True]),
        "command_applied_at": times[:2],
        "library_version": np.asarray([0, 0]),
        "control_library_version": np.asarray([0, 0]),
        "control_params_sha256": np.asarray(
            [_hash_tree(initial.params), _hash_tree(initial.params if revert else published.params)]
        ),
        "control_params_reverted": np.asarray([False, revert is not None]),
        "cumulative_gradient_steps": np.asarray([0, 0]),
        "estimated_model_sha256": np.asarray([_hash_tree(model)] * 2),
        "estimated_effectiveness": np.repeat(np.asarray(model.effectiveness)[None], 2, axis=0),
        "estimated_time_constants": np.repeat(np.asarray(model.time_constants)[None], 2, axis=0),
        "estimated_mass": np.repeat(np.asarray(model.body.mass)[None], 2, axis=0),
        "estimated_inertia": np.repeat(np.asarray(model.body.inertia)[None], 2, axis=0),
        "selected_index": np.asarray([1, 1]),
        "goal": states[:2, :3],
    }
    controls["controller_input_state_sha256"] = np.asarray(
        [_hash_tree(row) for row in controls["controller_input_state"]]
    )
    binding = {
        "config": asdict(config),
        "scene": scene.metadata(),
        "initial_state_sha256": _hash_tree(physical),
        "initial_learner_sha256": _hash_tree(initial),
        "initial_params_sha256": _hash_tree(initial.params),
        "checkpoint": {"checkpoint_sha256": initial_checkpoint.sha256},
        "source_sha256": {"synthetic_fixture_only": "not-study-evidence"},
    }
    summary = {
        "status": "completed",
        "termination": "duration_complete",
        "physical_time_seconds": 0.08,
        "final_state_sha256": _hash_tree(physical),
        "physical_world_id": scene.metadata()["physical_world_id"],
        "revert_control_params_actual_boundary": revert,
        "freeze_learning_actual_boundary": freeze,
    }
    for name, data in (("binding", binding), ("summary", summary)):
        (directory / f"{name}.json").write_text(json.dumps(_jsonable(data)))
    np.savez(directory / "dense.npz", **dense)
    np.savez(directory / "controls.npz", **controls)
    np.savez(
        directory / "applications.npz",
        time=times[:2],
        state=states[:2],
        command=command[:2].astype(np.float32),
        control_index=np.asarray([0, 1]),
    )
    snapshot = directory / "boundary"
    save_actuator_learner_checkpoint(
        published,
        contract,
        physical,
        snapshot,
        metadata={
            "capture_stage": "sensing_boundary_after_publication",
            "used_at_sensing_boundary": True,
            "sensing_boundary_time_seconds": 0.04,
            "physical_time_seconds": 0.04,
            "physical_state_sha256": _hash_tree(physical),
            "published_learner_sha256": _hash_tree(published),
            "published_version": 0,
            "control_library_version": 0,
            "control_params_sha256": str(controls["control_params_sha256"][1]),
            "control_params_reverted": revert is not None,
            "checkpoint_sha256": initial_checkpoint.sha256,
        },
    )
    return load_saved_run(directory), snapshot


@pytest.fixture
def pair(tmp_path: Path, checkpoint_parts: tuple[Any, ...]) -> tuple[Any, ...]:
    _, contract, initial, _, physical = checkpoint_parts
    save_actuator_learner_checkpoint(initial, contract, physical, tmp_path / "initial")
    original = load_actuator_learner_checkpoint(tmp_path / "initial")
    left, lhs = _episode(tmp_path / "left", checkpoint_parts, original)
    right, rhs = _episode(tmp_path / "right", checkpoint_parts, original, freeze=0.04)
    return left, lhs, right, rhs, original


def test_checkpoint_authentication_preserves_physical_float64_and_adam(
    pair: tuple[Any, ...],
) -> None:
    left, lhs, *_ = pair
    authenticated = authenticate_boundary_snapshot(left, lhs, 0.04)
    assert authenticated.report["physical_state_dtype"] == "float64"
    assert authenticated.report["optimizer_state_sha256"]
    physical = authenticated.checkpoint.physical_state
    assert not np.array_equal(physical, physical.astype(np.float32).astype(np.float64))
    np.testing.assert_array_equal(physical, left.controls["actual_state"][1])


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("capture_stage", "physical_grid_without_sensing_boundary", "actual sensing-boundary"),
        ("published_learner_sha256", "tampered", "params/Adam/learner hash"),
        ("control_params_sha256", "tampered", "parameter-use hash"),
    ],
)
def test_boundary_rejects_unauthenticated_snapshot(
    pair: tuple[Any, ...], field: str, value: str, message: str
) -> None:
    left, lhs, *_ = pair
    path = lhs.with_suffix(".json")
    contents = json.loads(path.read_text())
    contents["metadata"][field] = value
    path.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match=message):
        authenticate_boundary_snapshot(left, lhs, 0.04)


def test_prefix_includes_cut_motor_state_but_excludes_intervention_command(
    pair: tuple[Any, ...],
) -> None:
    left, lhs, right, rhs, _ = pair
    right.dense["command"].flags.writeable = True
    right.dense["command"][1:] += 0.003
    result = audit_common_prefix(
        left, right, 0.04, kind="onset_freeze", left_snapshot=lhs, right_snapshot=rhs
    )
    assert result["authentication_passed"]
    right.dense["state"].flags.writeable = True
    right.dense["state"][0, 16] += 0.001
    result = audit_common_prefix(
        left, right, 0.04, kind="onset_freeze", left_snapshot=lhs, right_snapshot=rhs
    )
    assert not result["authentication_passed"]
    assert not result["physical_and_control_checks"]["dense_state_through_boundary"]["exact_equal"]


def test_prefix_rejects_extra_scheduler_change(pair: tuple[Any, ...]) -> None:
    left, _, right, _, _ = pair
    right.binding["config"]["controller_reserve_seconds"] += 0.001
    with pytest.raises(ValueError, match="controller_reserve_seconds differs"):
        _intervention_contract(left, right, "onset_freeze", 0.04, None)


def test_no_fault_and_reversion_have_distinct_contracts(
    pair: tuple[Any, ...], tmp_path: Path, checkpoint_parts: tuple[Any, ...]
) -> None:
    left, lhs, _, _, original = pair
    nominal, nominal_snapshot = _episode(
        tmp_path / "nominal", checkpoint_parts, original, no_fault=True
    )
    result = audit_common_prefix(
        left,
        nominal,
        0.04,
        kind="no_fault_control",
        left_snapshot=lhs,
        right_snapshot=nominal_snapshot,
    )
    assert result["authentication_passed"]
    frozen, frozen_snapshot = _episode(tmp_path / "frozen", checkpoint_parts, original, freeze=0.04)
    reverted, reverted_snapshot = _episode(
        tmp_path / "reverted", checkpoint_parts, original, freeze=0.04, revert=0.04
    )
    result = audit_common_prefix(
        frozen,
        reverted,
        0.04,
        kind="early_freeze_vs_revert",
        left_snapshot=frozen_snapshot,
        right_snapshot=reverted_snapshot,
        reference_time=0.08,
    )
    assert result["authentication_passed"] and result["nontrivial_parameter_reversion"]
    with pytest.raises(ValueError, match="early/late"):
        _intervention_contract(frozen, reverted, "late_reversion", 0.04, 0.08)
    late = _intervention_contract(frozen, reverted, "late_reversion", 0.04, 0.02)
    assert late["kind"] == "late_reversion"


def test_recorded_inputs_use_saved_observation_and_absolute_obstacle_clock(
    pair: tuple[Any, ...],
) -> None:
    left, _, *_ = pair
    state, model, obstacles, _, previous, _, _ = _recorded_inputs(left, 1, 0.04)
    np.testing.assert_array_equal(state, left.controls["controller_input_state"][1])
    assert _hash_tree(model) == left.controls["estimated_model_sha256"][1]
    assert int(previous) == 1
    scene = replace(make_actuator_scene(17, "navigation", "combined"), event_time=0.04)
    expected = scene.world.obstacle_prediction(0.04, dt=0.02, horizon=4)
    np.testing.assert_array_equal(obstacles.centers, expected.centers)
    assert not np.array_equal(
        obstacles.centers, scene.world.obstacle_prediction(0, horizon=4).centers
    )


def test_all_candidate_branches_keep_nominal_objective_and_exact_original_decision(
    checkpoint_parts: tuple[Any, ...],
) -> None:
    model, _, _, _, physical = checkpoint_parts
    state = jnp.asarray(physical)
    config = ActuatorFilterConfig(horizon=2, sqp_iterations=0, gradient_mode="directional")
    commands = jnp.stack((state[13:] + 0.004, state[13:] - 0.004, state[13:] + 0.002))

    def rollouts(y: Any, point: Any, _: Any) -> ActuatorRollouts:
        def one(command: Any) -> Any:
            def advance(current: Any, _: Any) -> tuple[Any, Any]:
                following = effort_lag_step(current, command, point, config.dt, substeps=1)
                return following, following

            _, future = jax.lax.scan(advance, y, None, length=config.horizon)
            return jnp.concatenate((y[None], future))

        return ActuatorRollouts(
            jax.vmap(one)(commands),
            jnp.broadcast_to(commands[:, None], (3, config.horizon, 4)),
            jnp.ones(3, dtype=bool),
        )

    obstacles = RuntimeObstacleTrajectories(
        jnp.broadcast_to(jnp.asarray([4, 3, 3], jnp.float32), (3, 1, 3)),
        jnp.asarray([0.1]),
        jnp.ones((3, 1), dtype=bool),
        jnp.zeros((3, 1, 3)),
    )
    safety = RigidBodySafetySet(
        obstacles.centers[0],
        obstacles.radii,
        obstacles.mask[0],
        jnp.asarray([-5, -5, 0.15]),
        jnp.asarray([5, 5, 4]),
        jnp.asarray(3.5),
        jnp.asarray(12.0),
        jnp.asarray(0.9),
    )
    audit = evaluate_candidate_branches(
        state, model, obstacles, safety, rollouts, state[13:], jnp.asarray(2, jnp.int32), config
    )
    assert audit.report["eligible_candidate_indices"] == [0, 1, 2]
    assert audit.report["eligible_candidates_with_full_qp_and_hold_audit"] == 3
    assert audit.report["original_selection_equivalence"]["exact_equal"]
    assert audit.report["forced_candidates_selected_as_requested"]
    for index, step in audit.forced.items():
        assert int(step.selected_index) == index
        assert bool(step.qp_valid)
        np.testing.assert_array_equal(step.nominal_action, commands[0])
        np.testing.assert_allclose(step.action, commands[0], atol=1e-7)
        assert step.certificates.rollouts.states.shape == (3, 3, 17)
        assert np.all(np.isnan(np.asarray(step.certificates.gradients)[:, :13]))
        assert not np.any(step.certificates.gradient_components_computed[:13])


def test_cli_retains_incomplete_request_and_source_without_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "incomplete"
    argv = [
        "shared",
        "--episode",
        str(tmp_path / "missing"),
        "--boundary-snapshot",
        "unused",
        "--original-checkpoint",
        "unused",
        "--time",
        "1.0",
        "--output",
        str(output),
    ]
    assert main(argv) == 1
    assert json.loads((output / "error.json").read_text())["status"] == "incomplete"
    assert (output / "request.json").is_file()
    assert (output / "source/da_plcbf_actuator_confirmation.py").is_file()
    with pytest.raises(FileExistsError):
        main(argv)


def test_shared_audit_replays_recorded_decision_and_retains_full_numeric_branches(
    pair: tuple[Any, ...],
) -> None:
    from benchmark.da_plcbf_actuator_confirmation import ROOT, _sha

    left, snapshot, _, _, original = pair
    available = load_actuator_learner_checkpoint(snapshot)
    state, model, obstacles, safety, previous, goal, config = _recorded_inputs(left, 1, 0.04)
    functions = build_actuator_controller(original.contract.spec, original.config, config)
    recorded = jax.block_until_ready(
        functions.controller(
            state, available.state.params, model, obstacles, safety, previous, goal
        )
    )
    for name, value in _filter_record(recorded, retain_rollouts=False).items():
        if name not in left.controls:
            left.controls[name] = np.repeat(np.asarray(value)[None], 2, axis=0)
        else:
            values = left.controls[name].copy()
            values[1] = value
            left.controls[name] = values
    command = left.controls["planned_command"].copy()
    command[1] = np.asarray(recorded.action)
    left.controls["planned_command"] = command
    left.controls["actual_effectiveness"] = left.controls["estimated_effectiveness"]
    left.controls["actual_time_constants"] = left.controls["estimated_time_constants"]
    left.binding["source_sha256"] = {
        path.name: _sha(path) for path in (ROOT / "crazyflow/safety/da_plcbf").glob("actuator_*.py")
    }
    report, arrays = audit_shared_state(
        left, 0.04, boundary_snapshot=snapshot, original_checkpoint=original.npz_path
    )
    assert report["confirmation_passed"]
    assert report["recorded_decision_reproduction"]["reproduced"]
    assert report["actual_control_parameter_set"] == "available"
    assert arrays["physical_state"].dtype == np.float64
    assert arrays["controller_input_state"].dtype == np.float32
    for name in ("original", "available"):
        assert report[name]["original_selection_equivalence"]["exact_equal"]
        assert arrays[f"{name}.normal.certificates.rollouts.states"].shape == (3, 5, 17)
        for candidate in report[name]["eligible_candidate_indices"]:
            assert f"{name}.candidate_{candidate}.qp_check.nodes" in arrays
            assert f"{name}.candidate_{candidate}.fallback_check.nodes" in arrays
            assert f"{name}.candidate_{candidate}.emergency_check.nodes" in arrays
    # An unapplied proposal cannot become an actually executed learned-row effect.
    applied = left.controls["command_applied"].copy()
    applied[1] = False
    left.controls["command_applied"] = applied
    report, _ = audit_shared_state(
        left, 0.04, boundary_snapshot=snapshot, original_checkpoint=original.npz_path
    )
    assert not report["control_effect_evidence"]["recorded_applied_learned_row_changes_action"]
    assert not report["control_effect_evidence"]["recorded_applied_learned_fallback"]


def test_enclosing_campaign_source_snapshot_and_attempt_are_authenticated(
    pair: tuple[Any, ...], tmp_path: Path
) -> None:
    from benchmark.da_plcbf_actuator_confirmation import _sha
    from crazyflow.safety.da_plcbf.actuator_protocol import content_sha256

    left, _, _, _, original = pair
    relative = "crazyflow/safety/da_plcbf/synthetic_fixture_only"
    copied = tmp_path / "source" / relative
    copied.parent.mkdir(parents=True)
    copied.write_text("synthetic source fixture; never study evidence")
    digest = _sha(copied)
    binding_path = left.directory / "binding.json"
    episode = json.loads(binding_path.read_text())
    episode["source_sha256"] = {"synthetic_fixture_only": digest}
    binding_path.write_text(json.dumps(episode))
    campaign = {
        "specification": {"purpose": "synthetic confirmation test"},
        "source_sha256": {relative: digest},
        "checkpoint_sha256": {"initial.npz": original.sha256},
    }
    campaign["sha256"] = content_sha256(campaign)
    (tmp_path / "campaign_binding.json").write_text(json.dumps(campaign))
    records = [{"summary_path": "left/summary.json", "summary": left.summary}]
    (tmp_path / "campaign_result.json").write_text(json.dumps({"records": records}))
    authenticated = load_saved_run(left.directory)
    assert authenticated.campaign_envelope["authenticated"]
    assert authenticated.campaign_envelope["source_snapshot_file_count"] == 1
    assert not authenticated.campaign_envelope["all_current_campaign_sources_match"]
    copied.write_text("changed source")
    with pytest.raises(ValueError, match="source snapshot is missing or changed"):
        load_saved_run(left.directory)


def test_authentic_initial_hover_and_losslessly_widened_p0_archive(pair: tuple[Any, ...]) -> None:
    left, snapshot, *_ = pair
    checkpoint = load_actuator_learner_checkpoint(snapshot)
    initial = np.asarray(checkpoint.physical_state)
    later = initial.astype(np.float32)
    dense = {name: value.copy() for name, value in left.dense.items()}
    dense["state"][1:] = later
    dense["native_state"][1:] = later
    dense = {name: np.insert(value, 1, value[1], axis=0) for name, value in dense.items()}
    dense["time"][1] = 0.04 - 2e-16
    controls = {name: value.copy() for name, value in left.controls.items()}
    controls["actual_state"][1] = later
    controls["actual_state_sha256"][1] = _hash_tree(later)
    summary = dict(left.summary, final_state_sha256=_hash_tree(later))
    (left.directory / "summary.json").write_text(json.dumps(summary))
    np.savez(left.directory / "dense.npz", **dense)
    np.savez(left.directory / "controls.npz", **controls)
    np.savez(
        left.directory / "applications.npz",
        time=np.asarray([0.0, 0.0, 0.04]),
        control_index=np.asarray([-1, 0, 1]),
        state=np.stack((initial, initial, later)),
        command=np.vstack((initial[13:], controls["planned_command"])),
    )
    mixed_snapshot = left.directory / "boundary-original32"
    save_actuator_learner_checkpoint(
        checkpoint.state,
        checkpoint.contract,
        later,
        mixed_snapshot,
        metadata={**checkpoint.metadata, "physical_state_sha256": _hash_tree(later)},
    )
    run = load_saved_run(left.directory)
    assert run.storage_authentication["initial_hover_application_count"] == 1
    assert run.storage_authentication["dense_state_storage_dtype"] == "float64"
    assert run.storage_authentication["final_physical_authenticated_dtype"] == "float32"
    assert run.storage_authentication["application_command_storage_dtype"] == "float64"
    assert run.storage_authentication["planned_command_storage_dtype"] == "float32"
    boundary = authenticate_boundary_snapshot(run, mixed_snapshot, 0.04)
    assert boundary.report["physical_state_dtype"] == "float32"
    assert boundary.report["control_state_storage_dtype"] == "float64"
    # Narrowing is accepted only when the stored values still exactly reproduce
    # the original numerical state and original recorded dtype/shape/byte hash.
    dense["state"][-1, 13] += 1e-13
    np.savez(left.directory / "dense.npz", **dense)
    with pytest.raises(ValueError, match="lossless dtype recovery"):
        load_saved_run(left.directory)


@pytest.fixture
def adjacent_publications(pair: tuple[Any, ...]) -> tuple[Any, ...]:
    """A tiny actual finite learner update, with separately authenticated control boundaries."""
    from benchmark.da_plcbf_actuator_confirmation import ROOT, _sha

    run, old_snapshot, _, _, original = pair
    prior = run.directory / "prior-boundary"
    current = run.directory / "current-boundary"
    old = load_actuator_learner_checkpoint(old_snapshot)
    state, model, *_ = _recorded_inputs(run, 0, 0.0)
    learner = build_actuator_skill_learner(original.contract, original.config)
    following, metrics = jax.block_until_ready(learner.step(original.state, state, model))
    assert bool(metrics.finite_update_applied)
    assert int(following.cumulative_gradient_steps) == 1
    for name in ("library_version", "control_library_version", "cumulative_gradient_steps"):
        run.controls[name] = np.asarray([0, 1])
    run.controls["control_params_sha256"] = np.asarray(
        [_hash_tree(original.state.params), _hash_tree(following.params)]
    )
    for index, published, path in ((0, original.state, prior), (1, following, current)):
        when = float(run.controls["time"][index])
        save_actuator_learner_checkpoint(
            published,
            original.contract,
            old.physical_state,
            path,
            metadata={
                **old.metadata,
                "sensing_boundary_time_seconds": when,
                "physical_time_seconds": when,
                "published_learner_sha256": _hash_tree(published),
                "published_version": index,
                "control_library_version": index,
                "control_params_sha256": _hash_tree(published.params),
            },
        )
        y, model, obstacles, safety, previous, goal, config = _recorded_inputs(run, index, when)
        functions = build_actuator_controller(original.contract.spec, original.config, config)
        step = jax.block_until_ready(
            functions.controller(y, published.params, model, obstacles, safety, previous, goal)
        )
        for name, value in {
            **_filter_record(step, retain_rollouts=False),
            "planned_command": np.asarray(step.action),
        }.items():
            rows = (
                run.controls[name].copy()
                if name in run.controls
                else np.repeat(np.asarray(value)[None], 2, axis=0)
            )
            rows[index] = value
            run.controls[name] = rows
    run.controls["actual_effectiveness"] = run.controls["estimated_effectiveness"]
    run.controls["actual_time_constants"] = run.controls["estimated_time_constants"]
    run.binding["source_sha256"] = {
        path.name: _sha(path) for path in (ROOT / "crazyflow/safety/da_plcbf").glob("actuator_*.py")
    }
    return run, prior, current, original


def test_shared_prior_authenticates_both_decisions_and_uses_later_fixed_inputs(
    adjacent_publications: tuple[Any, ...],
) -> None:
    run, prior, current, original = adjacent_publications
    report, arrays = audit_prior_available(
        run,
        0.04,
        prior_time=0.0,
        prior_snapshot=prior,
        boundary_snapshot=current,
        original_checkpoint=original.npz_path,
    )
    assert report["confirmation_passed"]
    assert report["comparison_kind"] == "prior_available_vs_available"
    assert report["publication_comparison"]["single_gradient_update"]
    assert report["publication_comparison"]["parameters_changed"]
    assert report["prior_recorded_decision_reproduction"]["reproduced"]
    assert report["recorded_decision_reproduction"]["reproduced"]
    assert report["prior_boundary"]["actual_control_index"] == 0
    assert report["boundary"]["actual_control_index"] == 1
    assert "original" not in report
    assert float(arrays["absolute_obstacle_clock_seconds"]) == 0.04
    assert float(arrays["prior_boundary.absolute_obstacle_clock_seconds"]) == 0.0
    for name in ("prior_available", "available"):
        assert report[name]["original_selection_equivalence"]["exact_equal"]
        np.testing.assert_array_equal(
            arrays[f"{name}.normal.certificates.rollouts.states"][:, 0],
            np.repeat(arrays["controller_input_state"][None], 3, axis=0),
        )
        for candidate in report[name]["eligible_candidate_indices"]:
            assert f"{name}.candidate_{candidate}.qp_check.nodes" in arrays


def test_shared_prior_keeps_true_initial_checkpoint_guard(
    adjacent_publications: tuple[Any, ...],
) -> None:
    run, prior, current, _ = adjacent_publications
    with pytest.raises(ValueError, match="not the run's initial checkpoint"):
        audit_prior_available(
            run,
            0.04,
            prior_time=0.0,
            prior_snapshot=prior,
            boundary_snapshot=current,
            original_checkpoint=current,
        )


def test_shared_prior_rejects_same_boundary_or_wrong_physical_snapshot(
    adjacent_publications: tuple[Any, ...],
) -> None:
    run, prior, current, original = adjacent_publications
    with pytest.raises(ValueError, match="adjacent actual sensing boundaries"):
        audit_prior_available(
            run,
            0.04,
            prior_time=0.04,
            prior_snapshot=current,
            boundary_snapshot=current,
            original_checkpoint=original.npz_path,
        )
    with pytest.raises(ValueError, match="sensing time differs"):
        audit_prior_available(
            run,
            0.04,
            prior_time=0.0,
            prior_snapshot=current,
            boundary_snapshot=current,
            original_checkpoint=original.npz_path,
        )


def test_shared_prior_rejects_unpublished_parameter_difference(pair: tuple[Any, ...]) -> None:
    run, current, _, _, original = pair
    old = load_actuator_learner_checkpoint(current)
    prior = run.directory / "prior-no-publication"
    save_actuator_learner_checkpoint(
        original.state,
        original.contract,
        old.physical_state,
        prior,
        metadata={
            **old.metadata,
            "sensing_boundary_time_seconds": 0.0,
            "physical_time_seconds": 0.0,
            "published_learner_sha256": _hash_tree(original.state),
            "control_params_sha256": _hash_tree(original.state.params),
        },
    )
    with pytest.raises(ValueError, match="newly published learner update"):
        audit_prior_available(
            run,
            0.04,
            prior_time=0.0,
            prior_snapshot=prior,
            boundary_snapshot=current,
            original_checkpoint=original.npz_path,
        )
