"""Paired no-learning identity and complete physical accounting in the navigation runner."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import crazyflow.safety.da_plcbf.navigation_experiment as experiment
from crazyflow.safety.da_plcbf.learner_checkpoint import (
    load_learner_checkpoint,
    save_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.navigation_experiment import (
    NavigationExperimentConfig,
    run_navigation_experiment,
)
from crazyflow.safety.da_plcbf.navigation_world import (
    NavigationWorld,
    NavigationWorldConfig,
    WindEvent,
    build_navigation_world,
)
from crazyflow.safety.da_plcbf.online_constant_wind import (
    build_cf21b_version_a_resources,
    load_online_constant_wind_result,
)
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    build_fibonacci_skill_spec,
    build_persistent_skill_learner,
    initialize_skill_actor,
)

if TYPE_CHECKING:
    from pathlib import Path


def _tiny_checkpoint(
    world: NavigationWorld, directory: Path, *, model_compensation: bool = True
) -> tuple[Path, jax.Device, Any]:
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        resources = build_cf21b_version_a_resources()
        learner_config = PersistentSkillConfig(
            horizon=6,
            hidden_width=8,
            control_interval_steps=2,
            model_compensation=model_compensation,
        )
        spec = build_fibonacci_skill_spec(
            policy_count=4,
            latent_size=3,
            minimum_duration=0.04,
            maximum_duration=0.08,
            horizon_duration=0.12,
        )
        params = initialize_skill_actor(jax.random.key(15), spec, learner_config)
        learner = build_persistent_skill_learner(
            spec, resources.actuator, learner_config, device=cpu
        )
        persistent = learner.initialize(params, resources.model)
        checkpoint = directory / "initial_checkpoint"
        save_learner_checkpoint(
            persistent,
            spec,
            learner_config,
            resources.actuator,
            jnp.asarray(world.initial_state, dtype=jnp.float32),
            checkpoint,
        )
    return checkpoint, cpu, persistent


def test_hover_wind_on_off_then_navigation_preserves_the_matched_mapping(tmp_path: Path) -> None:
    world = build_navigation_world(
        NavigationWorldConfig(
            obstacle_count=0,
            waypoint_count=2,
            duration_seconds=0.16,
            wind_events=(WindEvent(0.04, (1.0, 0.0, 0.0)), WindEvent(0.08, (0.0, 0.0, 0.0))),
        )
    )
    config = NavigationExperimentConfig(
        enable_learning=False,
        learner_kind="original",
        navigation_start_seconds=0.12,
        fallback_mapping="matched_uncompensated",
    )
    checkpoint, cpu, persistent = _tiny_checkpoint(world, tmp_path, model_compensation=False)
    result = run_navigation_experiment(world, config, checkpoint, tmp_path / "hover", device=cpu)
    np.testing.assert_array_equal(result.trace.task_phase, ["hover"] * 3 + ["navigation"] * 2)
    for method in result.methods.values():
        np.testing.assert_array_equal(
            method.goal_position[:3], np.tile(world.initial_state[:3].astype(np.float32), (3, 1))
        )
        np.testing.assert_array_equal(
            method.goal_position[3], world.waypoint_positions[0].astype(np.float32)
        )
        np.testing.assert_array_equal(method.library_version, int(persistent.library_version))
    np.testing.assert_array_equal(result.trace.fixed.full_state, result.trace.adaptive.full_state)
    assert result.summary["compensation_protocol"]["prefix"] is False
    assert result.summary["compensation_protocol"]["post_event"] is False
    assert result.summary["compensation_protocol"]["nominal_task_controller_compensation"] is True
    restored = load_online_constant_wind_result(
        tmp_path / "hover/navigation_comparison.npz", tmp_path / "hover/navigation_comparison.json"
    )
    np.testing.assert_array_equal(restored.trace.task_phase, result.trace.task_phase)
    np.testing.assert_array_equal(restored.trace.phase_caption, result.trace.phase_caption)


def test_old_undersized_enclosure_requires_explicit_reproduction_opt_in() -> None:
    world = build_navigation_world(NavigationWorldConfig(ego_radius=0.05))
    with pytest.raises(ValueError, match="enclose the cf21B"):
        NavigationExperimentConfig().validate(world)
    NavigationExperimentConfig(allow_legacy_point_enclosure=True).validate(world)


def test_modeled_termination_retains_controls_after_an_enclosure_only_breach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scripted plant isolates the observer; both modes retain identical real controller calls."""
    world = build_navigation_world(
        NavigationWorldConfig(obstacle_count=0, waypoint_count=2, duration_seconds=0.12)
    )
    initial = np.array([0, 0, 1.4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    world = replace(
        world,
        config=replace(world.config, obstacle_count=1),
        initial_state=initial,
        waypoint_positions=np.array([[2.0, 0, 1.4], [-2.0, 0, 1.4]]),
        obstacle_mean_centers=np.array([[0.5, 0, 1.4]]),
        obstacle_amplitudes=np.zeros((1, 3)),
        obstacle_angular_frequencies=np.zeros(1),
        obstacle_phases=np.zeros(1),
        obstacle_radii=np.array([0.1]),
    )
    checkpoint, cpu, _ = _tiny_checkpoint(world, tmp_path)

    def scripted_plant(state: Any, action: Any, model: Any, dt: float) -> Any:
        del action, model, dt
        return state.at[0].add(0.15)

    monkeypatch.setattr(experiment, "direct_wrench_symplectic_step", scripted_plant)
    results = {}
    for geometry in ("body_origin_enclosure", "modeled_collider"):
        config = NavigationExperimentConfig(
            enable_learning=False, learner_kind="original", termination_geometry=geometry
        )
        results[geometry] = run_navigation_experiment(
            world, config, checkpoint, tmp_path / geometry, device=cpu
        )
    legacy, modeled = (results[key] for key in ("body_origin_enclosure", "modeled_collider"))
    for method in ("fixed", "adaptive"):
        before = legacy.summary["methods"][method]
        after = modeled.summary["methods"][method]
        assert before["active_controls"] == 1
        assert before["termination_time_seconds"] == pytest.approx(0.04)
        assert after["active_controls"] == 2
        assert after["termination_time_seconds"] == pytest.approx(0.08)
        assert after["collision_event"]["collision_kind"] == "modeled_collider_obstacle"
        assert after["collision_observation"]["modeled_collider_collision"] is True
        assert after["body_origin_enclosure_breach_recorded"]
        assert after["waypoints_completed"] == 0
        assert not after["publications_and_inputs"][0]["modeled_collider_clearance_bounds_m"][
            "actual_xml_sphere_geometry"
        ][0] < 0
        np.testing.assert_array_equal(
            legacy.methods[method].applied_wrench[0], modeled.methods[method].applied_wrench[0]
        )


def test_no_learning_pair_preserves_state_action_parameter_and_optimizer_identity(
    tmp_path: Path,
) -> None:
    world = build_navigation_world(
        NavigationWorldConfig(obstacle_count=0, waypoint_count=2, duration_seconds=0.12)
    )
    config = NavigationExperimentConfig(
        enable_learning=False,
        learner_kind="original",
        probe_every_controls=1,
        learning_start_seconds=0.0,
    )
    checkpoint, cpu, persistent = _tiny_checkpoint(world, tmp_path)
    with jax.default_device(cpu):
        output = tmp_path / "run"
        result = run_navigation_experiment(world, config, checkpoint, output, device=cpu)
        final = load_learner_checkpoint(output / "final_adaptive_checkpoint", device=cpu)
    for name in (
        "full_state",
        "applied_wrench",
        "nominal_wrench",
        "fallback_rollouts",
        "library_version",
        "cumulative_gradient_steps",
        "qp_held_operational_residuals",
        "applied_held_physical_margins",
        "recorded_control_valid",
        "waypoint_index",
    ):
        np.testing.assert_array_equal(
            getattr(result.trace.fixed, name), getattr(result.trace.adaptive, name)
        )
    np.testing.assert_array_equal(
        result.trace.fixed.recorded_control_valid, [True, True, True, False]
    )
    assert result.summary["methods"]["fixed"]["active_controls"] == 3
    assert result.summary["methods"]["adaptive"]["finite_updates"] == 0
    assert result.summary["methods"]["adaptive"]["termination"] == "timeout"
    for original, restored in zip(
        jax.tree.leaves(persistent), jax.tree.leaves(final.state), strict=True
    ):
        np.testing.assert_array_equal(original, restored)
    with np.load(output / "dense_plant_states.npz") as dense:
        np.testing.assert_array_equal(dense["fixed"], dense["adaptive"])
        assert dense["fixed"].shape == (7, 13)
        np.testing.assert_array_equal(dense["fixed"][-1], result.trace.fixed.full_state[-1])
    with np.load(output / "raw_diagnostics.npz") as raw:
        assert raw["fixed_actual_operational_margins"].shape == (7, 9)
    for probe in result.summary["same_state_probes"]:
        assert probe["fallback_max_hard"] is None
        assert probe["augmented_max_hard"] is None
        assert probe["collision_constraint_active"] is False
        assert all(probe["candidate_input_valid"])
    json.dumps(result.summary, allow_nan=False)
    audit = result.summary["methods"]["fixed"]["execution_audit"]
    assert audit["actual_node_count"] == 7
    assert audit["all_actual_physical_nodes_pass"]
    assert audit["applied_motor_limit_violating_controls"] == 0
    restored = load_online_constant_wind_result(
        output / "navigation_comparison.npz", output / "navigation_comparison.json"
    )
    np.testing.assert_array_equal(restored.trace.fixed.full_state, result.trace.fixed.full_state)
    np.testing.assert_array_equal(
        restored.trace.fixed.recorded_control_valid, result.trace.fixed.recorded_control_valid
    )
    assert restored.summary["methods"]["fixed"]["execution_audit"] == audit
    assert {probe["anchor"] for probe in result.summary["same_state_probes"]} == {
        "fixed",
        "adaptive",
    }


@pytest.mark.parametrize("field", ["nominal_acceleration_limit", "estimator_response_rate"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 0.0])
def test_navigation_rejects_nonfinite_or_nonpositive_execution_config(
    field: str, invalid: float
) -> None:
    world = build_navigation_world(NavigationWorldConfig(obstacle_count=0, duration_seconds=0.12))
    with pytest.raises(ValueError, match="positive finite"):
        replace(NavigationExperimentConfig(), **{field: invalid}).validate(world)


def test_budgeted_runner_publishes_finite_updates_only_at_completed_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deterministic clock audits integration chronology; its times are not benchmarks."""
    world = build_navigation_world(
        NavigationWorldConfig(obstacle_count=0, waypoint_count=2, duration_seconds=0.12)
    )
    checkpoint, cpu, initial = _tiny_checkpoint(world, tmp_path)
    elapsed = [1000.0]

    def now() -> float:
        elapsed[0] += 0.0001
        return elapsed[0]

    def sleep(seconds: float) -> None:
        elapsed[0] += seconds

    monkeypatch.setattr(experiment, "time", SimpleNamespace(perf_counter=now, sleep=sleep))
    host_warmups = []
    warm_host = experiment._warm_host_diagnostics

    def audit_discarded_warmup(*args: Any) -> dict[str, Any]:
        original_state = np.array(args[2])
        original_snapshot = [np.array(leaf) for leaf in jax.tree.leaves(args[3])]
        diagnostics = warm_host(*args)
        np.testing.assert_array_equal(args[2], original_state)
        for before, after in zip(original_snapshot, jax.tree.leaves(args[3]), strict=True):
            np.testing.assert_array_equal(before, after)
        host_warmups.append(diagnostics)
        return diagnostics

    monkeypatch.setattr(experiment, "_warm_host_diagnostics", audit_discarded_warmup)
    config = NavigationExperimentConfig(
        execution_mode="budgeted",
        learner_kind="original",
        learning_start_seconds=0.0,
        update_every_controls=1,
        probe_every_controls=1,
        termination_geometry="modeled_collider",
    )
    with jax.default_device(cpu):
        result = run_navigation_experiment(
            world, config, checkpoint, tmp_path / "paced", device=cpu
        )
        final = load_learner_checkpoint(tmp_path / "paced/final_adaptive_checkpoint", device=cpu)
    active = result.trace.adaptive.recorded_control_valid
    np.testing.assert_array_equal(result.trace.adaptive.library_version[active], [0, 1, 2])
    summary = result.summary["methods"]["adaptive"]
    assert summary["finite_updates"] == 2
    assert int(final.state.library_version) == int(initial.library_version) + 2
    assert len(host_warmups) == 2
    assert all(row["discarded_record_count"] == 1 for row in host_warmups)
    for method in ("fixed", "adaptive"):
        np.testing.assert_array_equal(
            result.methods[method].full_state[0], world.initial_state.astype(np.float32)
        )
        assert result.methods[method].library_version[0] == int(initial.library_version)
        assert result.summary["methods"][method]["active_controls"] == 3
        assert result.summary["methods"][method]["warmup"]["host_diagnostics"][
            "discarded_record_count"
        ] == 1
    boundaries = summary["publications_and_inputs"]
    assert [row["completed_version"] for row in boundaries] == [1, 2, None]
    for before, after in zip(boundaries[:-1], boundaries[1:], strict=True):
        assert before["completed_wall_seconds"] <= after["started_wall_seconds"]
        assert after["started_wall_seconds"] >= after["scheduled_wall_seconds"]
        assert before["completed_version"] == after["version_used"]
    assert all(
        row[key] >= 0
        for row in boundaries
        for key in (
            "pre_controller_seconds",
            "host_recording_seconds",
            "plant_seconds",
            "collider_audit_seconds",
        )
    )
    assert summary["service_exceeds_nominal_period_count"] == 0
    assert result.summary["schedule"]["mode"] == "exogenous_deterministic_opportunity_mask"
    assert result.summary["schedule"]["actual_execution_mode"] == "budgeted"
    assert (
        result.summary["schedule"]["publication"] == "completed snapshot at actual paced boundary"
    )
    assert result.summary["runtime_feasibility"]["adaptive_online_runtime_feasible"]
