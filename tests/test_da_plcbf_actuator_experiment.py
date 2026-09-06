"""CPU clock tests for causal actuator application, complete episodes and publications."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from crazyflow.safety.da_plcbf import actuator_experiment as runtime
from crazyflow.safety.da_plcbf.actuator_independent import NumpyEffortPlant
from crazyflow.safety.da_plcbf.actuator_learning import ActuatorLearnerState, ActuatorSkillConfig
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_study import (
    ActuatorObservationConfig,
    initial_augmented_state,
    make_actuator_scene,
    nominal_actuator_model,
)


class FakeClock:
    """Advance service time only through declared work and pacing."""

    def __init__(self) -> None:
        """Use an epoch distinct from physical time zero."""
        self.now = 50.0

    def perf_counter(self) -> float:
        """Read the injected wall clock."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Apply an exact pacing wait."""
        self.now += seconds


@pytest.fixture
def harness(monkeypatch: Any, tmp_path: Any) -> Any:
    clock = FakeClock()
    monkeypatch.setattr(runtime, "time", clock)
    monkeypatch.setattr(runtime, "_synchronize", lambda value: value)
    model = nominal_actuator_model()
    original = make_actuator_scene(11, "structured", "nominal")
    world = replace(original.world, config=replace(original.world.config, duration_seconds=0.16))
    scene = replace(original, world=world, navigation_start=0.0, event_time=0.02)
    initial = initial_augmented_state(world.initial_state, model)
    actor = ActuatorSkillConfig(horizon=4, control_interval_steps=2)
    state = ActuatorLearnerState(
        params=np.asarray([0.0]),
        previous_params=np.asarray([-1.0]),
        optimizer_state=np.asarray([987.0]),
        cumulative_gradient_steps=np.asarray(128),
        latest_dynamics_estimate=model,
        library_version=np.asarray(128),
    )
    bundle = SimpleNamespace(
        state=state,
        config=actor,
        contract=SimpleNamespace(spec=SimpleNamespace(latent_codes=np.zeros((16, 2)))),
        physical_state=np.zeros(17),
        metadata={"mode": "nominal", "seed": 11},
        npz_path=tmp_path / "loaded.npz",
        json_path=tmp_path / "loaded.json",
        sha256="test-source-checkpoint",
    )
    controller_calls = []
    learner_calls = []
    settings = SimpleNamespace(
        controller_service=0.006, learner_service=0.001, invalid=False, raise_control=False
    )

    def controller(
        y: Any, params: Any, point: Any, obstacles: Any, safety: Any, previous: Any, goal: Any
    ) -> Any:
        if settings.raise_control:
            raise RuntimeError("deliberate numerical failure")
        clock.now += settings.controller_service
        controller_calls.append(
            {
                "state": np.asarray(y).copy(),
                "params": np.asarray(params).copy(),
                "eta": np.asarray(point.effectiveness).copy(),
                "model_sha256": runtime._hash_tree(point),
                "prediction_sha256": runtime._hash_tree(obstacles),
                "safety_sha256": runtime._hash_tree(safety),
            }
        )
        command = initial[13:] * 1.01 + float(np.asarray(params)[0]) * 1e-5
        return SimpleNamespace(
            action=command,
            selected_index=0,
            certificates=SimpleNamespace(
                rollouts=SimpleNamespace(states=np.broadcast_to(np.asarray(y), (2, 5, 17)))
            ),
        )

    def step(persistent: Any, observed: Any, point: Any) -> Any:
        service = settings.learner_service
        clock.now += service() if callable(service) else service
        learner_calls.append(
            {
                "state": np.asarray(observed).copy(),
                "eta": np.asarray(point.effectiveness).copy(),
                "optimizer": np.asarray(persistent.optimizer_state).copy(),
            }
        )
        following = persistent.replace(
            params=persistent.params + 1,
            previous_params=persistent.params,
            optimizer_state=persistent.optimizer_state + 7,
            cumulative_gradient_steps=persistent.cumulative_gradient_steps + 1,
            library_version=persistent.library_version + 1,
            latest_dynamics_estimate=point,
        )
        return following, SimpleNamespace(
            finite_update_applied=True,
            gradient_norm=2.0,
            parameter_update_norm=1.0,
            loss={"total": float("nan")},
        )

    def filter_record(value: Any, retain: bool) -> dict[str, Any]:
        return {
            "input_valid": not settings.invalid,
            "mode": "invalid_input" if settings.invalid else "qp",
            "selected_index": 0,
            "hard": np.ones(2),
            "smooth": np.ones(2) * 0.9,
        }

    monkeypatch.setattr(runtime, "_filter_record", filter_record)
    cfg = runtime.ActuatorEpisodeConfig(
        method="A",
        plant_level="P1",
        filter_config=ActuatorFilterConfig(horizon=4),
        warmup_calls=2,
        controller_reserve_seconds=0.0,
        update_safety_factor=1.0,
    )

    def run(
        *,
        scene_override: Any = None,
        bundle_override: Any = None,
        name: str = "episode",
        snapshot_callback: Any = None,
        **changes: Any,
    ) -> Any:
        return runtime.run_actuator_episode(
            scene if scene_override is None else scene_override,
            bundle if bundle_override is None else bundle_override,
            replace(cfg, **changes),
            tmp_path / name,
            controllerfunctions=SimpleNamespace(controller=controller),
            learner_functions=SimpleNamespace(step=step),
            snapshot_callback=snapshot_callback,
        )

    return SimpleNamespace(
        run=run,
        scene=scene,
        bundle=bundle,
        initial=initial,
        model=model,
        settings=settings,
        clock=clock,
        controller_calls=controller_calls,
        learner_calls=learner_calls,
        tmp_path=tmp_path,
    )


def assert_complete(result: Any) -> None:
    assert result.summary["error"] is None, result.summary["error"]
    assert result.summary["termination"] == "duration_complete"
    assert result.summary["full_episode_completed"]


def test_full_state_and_adam_warmup_are_preserved_and_updates_publish_next_boundary(
    harness: Any,
) -> None:
    result = harness.run()
    assert_complete(result)
    np.testing.assert_array_equal(result.dense_traces["state"][0], harness.initial)
    assert np.any(result.dense_traces["state"][0, 13:] != harness.bundle.physical_state[13:])
    np.testing.assert_array_equal(result.control_traces["library_version"], [128, 129, 130, 131])
    np.testing.assert_array_equal(harness.learner_calls[2]["optimizer"], [987.0])
    np.testing.assert_array_equal(result.final_learner_state.optimizer_state, [1008.0])
    assert result.summary["finite_credited_updates"] == 3
    assert result.summary["snapshot_publications"][0]["published_simulation_time"] == pytest.approx(
        0.04
    )
    for path in result.artifacts.values():
        assert path.exists()
    with np.load(result.artifacts["dense"], allow_pickle=False) as arrays:
        assert arrays["state"].shape[1] == 17
    assert json.loads(result.artifacts["summary"].read_text())["status"] == "completed"


def test_delayed_controller_holds_old_command_across_fault_and_skips_ticks(harness: Any) -> None:
    harness.settings.controller_service = 0.09
    scene = replace(harness.scene, effectiveness_after=(0.8, 1.0, 1.0, 1.0))
    result = harness.run(method="F2", execution_mode="delayed", scene_override=scene)
    assert_complete(result)
    assert result.control_traces["time"][0] == 0
    assert result.control_traces["time"][1] == pytest.approx(0.12)
    assert result.control_traces["command_applied_at"][0] == pytest.approx(0.09)
    assert result.control_traces["skipped_sensing_ticks"][0] == 2
    dense = result.dense_traces
    mask = dense["time"] < 0.09 - 1e-10
    np.testing.assert_allclose(
        dense["command"][mask], np.broadcast_to(harness.initial[13:], (np.sum(mask), 4))
    )
    changed = (dense["time"] >= 0.02 - 1e-10) & mask
    assert np.any(changed)
    np.testing.assert_allclose(dense["actual_effectiveness"][changed, 0], 0.8)
    # The sensing snapshot has no knowledge of the later event during its computation.
    np.testing.assert_array_equal(result.control_traces["estimated_effectiveness"][0], np.ones(4))
    assert not result.control_traces["command_applied"][1]


def test_delayed_learner_overrun_advances_new_command_and_publishes_at_next_available_tick(
    harness: Any,
) -> None:
    calls = iter([0.001, 0.001, 0.09, 0.001, 0.001])
    harness.settings.learner_service = lambda: next(calls)
    result = harness.run(execution_mode="delayed")
    assert_complete(result)
    np.testing.assert_allclose(result.control_traces["time"], [0.0, 0.12])
    np.testing.assert_array_equal(result.control_traces["library_version"], [128, 129])
    publication = result.summary["snapshot_publications"][0]
    assert publication["published_simulation_time"] == pytest.approx(0.12)
    assert publication["training_simulation_time"] == 0
    assert result.control_traces["skipped_sensing_ticks"][0] == 2
    assert not result.control_traces["learner_deadline_met"][0]
    np.testing.assert_array_equal(
        harness.learner_calls[2]["state"], result.control_traces["controller_input_state"][0]
    )
    np.testing.assert_allclose(harness.learner_calls[2]["state"], harness.initial, atol=3e-8)
    during_update = (result.dense_traces["time"] > 0.006) & (result.dense_traces["time"] < 0.096)
    assert np.any(during_update)
    np.testing.assert_allclose(
        result.dense_traces["command"][during_update],
        np.broadcast_to(result.control_traces["planned_command"][0], (np.sum(during_update), 4)),
    )


def inject_contact(monkeypatch: Any, when: float) -> None:
    def audit(world: Any, times: Any, states: Any) -> dict[str, Any]:
        return {
            "terminate": bool(times[-1] >= when - 1e-10),
            "collision_kind": "modeled_collider_obstacle",
            "first_intersection_time_seconds": when,
            "audit": {},
        }

    monkeypatch.setattr(runtime, "_audit_interval", audit)
    monkeypatch.setattr(
        runtime,
        "summarize_collision_observation",
        lambda *args, **kwargs: {"modeled_collider_collision": True},
    )


def test_contact_during_controller_prevents_new_command_and_learning(
    harness: Any, monkeypatch: Any
) -> None:
    inject_contact(monkeypatch, 0.03)
    harness.settings.controller_service = 0.07
    result = harness.run(execution_mode="delayed")
    assert result.summary["termination"] == "physical_collision"
    assert result.summary["physical_time_seconds"] == pytest.approx(0.03)
    assert result.summary["applied_control_count"] == 0
    assert result.summary["learner_calls"] == 0
    assert result.summary["final_cumulative_gradient_steps"] == 128
    assert len(harness.controller_calls) == 3  # two disposable warmups, one episode decision
    assert len(harness.learner_calls) == 2
    assert not result.summary["reached_full_duration"]


def test_contact_during_learner_discards_completion_and_preserves_adam(
    harness: Any, monkeypatch: Any
) -> None:
    inject_contact(monkeypatch, 0.03)
    calls = iter([0.001, 0.001, 0.07])
    harness.settings.learner_service = lambda: next(calls)
    result = harness.run(execution_mode="delayed")
    assert result.summary["termination"] == "physical_collision"
    assert result.summary["finite_uncredited_updates"] == 1
    assert result.summary["finite_credited_updates"] == 0
    assert result.summary["final_cumulative_gradient_steps"] == 128
    np.testing.assert_array_equal(result.final_learner_state.optimizer_state, [987.0])
    assert result.summary["snapshot_publications"] == []


def test_completed_task_holds_final_goal_until_duration(harness: Any) -> None:
    world = replace(harness.scene.world, waypoint_positions=harness.initial[None, :3].copy())
    result = harness.run(method="F2", scene_override=replace(harness.scene, world=world))
    assert_complete(result)
    assert result.summary["task_completed"]
    assert result.summary["task_completion_time_seconds"] == 0
    assert result.summary["control_count"] == 4
    np.testing.assert_allclose(
        result.control_traces["goal"], np.broadcast_to(harness.initial[:3], (4, 3))
    )


def test_collision_precedes_arrival_at_next_boundary(harness: Any, monkeypatch: Any) -> None:
    inject_contact(monkeypatch, 0.04)
    world = replace(harness.scene.world, waypoint_positions=harness.initial[None, :3].copy())
    scene = replace(harness.scene, world=world, navigation_start=0.04)
    result = harness.run(method="F2", scene_override=scene)
    assert result.summary["termination"] == "physical_collision"
    assert result.summary["waypoints_completed"] == 0
    assert len(result.control_traces["time"]) == 1


@pytest.mark.parametrize("contact_at_duration", [False, True])
def test_final_boundary_arrival_preserves_collision_precedence_and_service_counts(
    harness: Any, monkeypatch: Any, contact_at_duration: bool
) -> None:
    duration = harness.scene.world.config.duration_seconds
    radius = 0.125

    def scripted_motion(plant: NumpyEffortPlant, command: Any, dt: float) -> None:
        plant._state[0] = float(harness.initial[0]) + radius * ((plant.time + dt) / duration)
        plant._state[7] = radius / duration

    monkeypatch.setattr(NumpyEffortPlant, "_step", scripted_motion)
    if contact_at_duration:
        inject_contact(monkeypatch, duration)
    goal = np.asarray(harness.initial[:3], dtype=float).copy()
    goal[0] += 2 * radius
    world = replace(
        harness.scene.world,
        config=replace(harness.scene.world.config, reach_radius=radius),
        waypoint_positions=goal[None],
    )
    result = harness.run(scene_override=replace(harness.scene, world=world))

    assert result.summary["error"] is None, result.summary["error"]
    assert result.summary["full_episode_completed"]
    assert result.summary["physical_time_seconds"] == pytest.approx(duration)
    assert np.linalg.norm(result.final_state[:3] - goal) == pytest.approx(radius)
    assert np.all(
        np.linalg.norm(result.control_traces["actual_state"][:, :3] - goal, axis=1) > radius
    )
    assert result.summary["termination"] == (
        "physical_collision" if contact_at_duration else "duration_complete"
    )
    assert result.summary["waypoint_arrival_times_seconds"] == (
        [] if contact_at_duration else [pytest.approx(duration)]
    )
    assert result.summary["task_completed"] is not contact_at_duration
    assert result.summary["control_count"] == 4
    assert result.summary["applied_control_count"] == 4
    assert result.summary["actual_command_application_count"] == 5  # Includes initial hold.
    assert result.summary["learner_calls"] == 3
    assert len(harness.controller_calls) == 6  # Includes two disposable warmups.
    assert len(harness.learner_calls) == 5
    assert result.summary["final_completed_library_version"] == 131
    assert result.summary["final_published_library_version"] == 131
    assert len(result.summary["snapshot_publications"]) == 3


def test_freeze_and_revert_preserve_prefix_optimizer_and_separate_control_version(
    harness: Any,
) -> None:
    frozen = harness.run(name="frozen", freeze_learning_at=0.08)
    reverted = harness.run(name="reverted", freeze_learning_at=0.08, revert_control_params_at=0.08)
    assert_complete(frozen)
    assert_complete(reverted)
    prefix = frozen.dense_traces["time"] <= 0.08 + 1e-10
    np.testing.assert_array_equal(
        frozen.dense_traces["state"][prefix], reverted.dense_traces["state"][prefix]
    )
    np.testing.assert_array_equal(frozen.control_traces["library_version"], [128, 129, 130, 130])
    np.testing.assert_array_equal(reverted.control_traces["library_version"], [128, 129, 130, 130])
    np.testing.assert_array_equal(
        reverted.control_traces["control_library_version"], [128, 129, 128, 128]
    )
    np.testing.assert_array_equal(
        frozen.final_learner_state.optimizer_state, reverted.final_learner_state.optimizer_state
    )
    assert reverted.summary["freeze_learning_actual_boundary"] == pytest.approx(0.08)


def test_invalid_observation_is_unclipped_and_no_command_is_applied(
    harness: Any, monkeypatch: Any
) -> None:
    harness.settings.invalid = True

    def negative_motor(actual: Any, *args: Any) -> np.ndarray:
        observed = np.asarray(actual).copy()
        observed[13] = -1
        return observed

    monkeypatch.setattr(runtime, "observe_actuator_state", negative_motor)
    result = harness.run(method="F2")
    assert result.summary["status"] == "incomplete"
    assert result.summary["termination"] == "invalid_observation"
    assert result.control_traces["observed_state"][0, 13] == -1
    assert result.control_traces["actual_state"][0, 13] > 0
    assert result.summary["applied_control_count"] == 0


def test_paced_clock_is_not_replayed_as_physical_actuation_delay(harness: Any) -> None:
    result = harness.run(execution_mode="paced")
    assert_complete(result)
    np.testing.assert_allclose(result.control_traces["time"], [0.0, 0.04, 0.08, 0.12])
    np.testing.assert_allclose(
        result.control_traces["command_applied_at"], result.control_traces["time"]
    )
    assert result.summary["finite_credited_updates"] == 3
    assert result.summary["skipped_sensing_ticks"] == 0


def test_failures_are_preserved_and_output_directory_is_exclusive(harness: Any) -> None:
    harness.settings.raise_control = True
    result = harness.run(method="F2")
    assert result.summary["status"] == "incomplete"
    assert result.summary["error"]["message"] == "deliberate numerical failure"
    assert result.artifacts["error"].exists()
    assert result.artifacts["dense"].exists()
    with pytest.raises(FileExistsError):
        harness.run(method="F2")


def test_method_checkpoint_identity_rejects_dr_relabeling_and_single_mismatch(harness: Any) -> None:
    with pytest.raises(ValueError, match="separately trained"):
        harness.run(method="DR")
    with pytest.raises(ValueError, match="single-recovery"):
        harness.run(method="A1")


def test_native_plant_records_both_effort_and_rpm_state(harness: Any) -> None:
    result = harness.run(method="F2", plant_level="P2")
    assert_complete(result)
    assert result.summary["native_motor_state_units"] == "rpm"
    assert result.dense_traces["native_state"].shape == result.dense_traces["state"].shape
    np.testing.assert_allclose(
        result.dense_traces["native_state"][:, :13], result.dense_traces["state"][:, :13]
    )
    assert np.min(result.dense_traces["native_state"][:, 13:]) > 1000
    assert np.max(result.dense_traces["state"][:, 13:]) < 1


@pytest.mark.parametrize("capture_times", [(0.015, 0.075), [0.015, 0.075]])
def test_critical_capture_preserves_all_motor_components(harness: Any, capture_times: Any) -> None:
    captured = []
    # Public capture callbacks are observational and do not modify parameter publication.
    result = harness.run(
        method="F2", capture_times=capture_times, snapshot_callback=captured.append
    )
    assert_complete(result)
    assert np.any(np.isclose(result.dense_traces["time"], 0.015))
    assert np.any(np.isclose(result.dense_traces["time"], 0.075))
    assert result.final_state.shape == (17,)
    np.testing.assert_allclose(
        [row["physical_time_seconds"] for row in captured], [0, 0.015, 0.075, 0.16]
    )
    assert all(row["state"].shape == (17,) for row in captured)
    assert all(row["published_version"] == 128 for row in captured)


def test_observation_model_delay_uses_past_not_future_parameters(harness: Any) -> None:
    scene = replace(harness.scene, effectiveness_after=(0.8, 1.0, 1.0, 1.0))
    result = harness.run(
        method="F2",
        scene_override=scene,
        observation_config=ActuatorObservationConfig(parameter_delay_seconds=0.06),
    )
    assert_complete(result)
    actual = result.control_traces["actual_effectiveness"][:, 0]
    estimated = result.control_traces["estimated_effectiveness"][:, 0]
    np.testing.assert_allclose(actual, [1.0, 0.8, 0.8, 0.8])
    np.testing.assert_allclose(estimated, [1.0, 1.0, 0.8, 0.8])


def test_boundary_capture_binds_published_adam_and_actual_reverted_control_params(
    harness: Any,
) -> None:
    captured = []
    result = harness.run(
        capture_times=(0.0, 0.04, 0.055, 0.08, 0.16),
        freeze_learning_at=0.08,
        revert_control_params_at=0.08,
        snapshot_callback=captured.append,
    )
    assert_complete(result)
    critical = {
        round(row["physical_time_seconds"], 6): row
        for row in captured
        if row["label"].startswith("critical-")
    }
    for when, index, version, optimizer in (
        (0.0, 0, 128, 987),
        (0.04, 1, 129, 994),
        (0.08, 2, 130, 1001),
    ):
        snapshot = critical[when]
        assert snapshot["capture_stage"] == "sensing_boundary_after_publication"
        assert snapshot["used_at_sensing_boundary"]
        assert snapshot["published_version"] == version
        assert int(snapshot["learner_state"].library_version) == version
        np.testing.assert_array_equal(snapshot["learner_state"].optimizer_state, [optimizer])
        np.testing.assert_array_equal(
            snapshot["state"], result.control_traces["actual_state"][index]
        )
        assert (
            snapshot["physical_state_sha256"] == result.control_traces["actual_state_sha256"][index]
        )
        assert (
            snapshot["control_params_sha256"]
            == result.control_traces["control_params_sha256"][index]
        )
        assert (
            snapshot["control_library_version"]
            == result.control_traces["control_library_version"][index]
        )
    assert critical[0.08]["control_params_reverted"]
    assert critical[0.08]["control_library_version"] == 128
    assert critical[0.055]["capture_stage"] == "physical_off_grid"
    assert critical[0.055]["published_version"] == 129
    assert not critical[0.055]["used_at_sensing_boundary"]
    assert critical[0.16]["capture_stage"] == "physical_grid_without_sensing_boundary"
    assert not critical[0.16]["used_at_sensing_boundary"]


def test_delayed_skipped_grid_capture_keeps_original_snapshot_until_real_publication(
    harness: Any,
) -> None:
    durations = iter([0.001, 0.001, 0.09, 0.001])
    harness.settings.learner_service = lambda: next(durations)
    captured = []
    result = harness.run(
        execution_mode="delayed",
        capture_times=(0.04, 0.08, 0.12),
        snapshot_callback=captured.append,
    )
    assert_complete(result)
    critical = [row for row in captured if row["label"].startswith("critical-")]
    np.testing.assert_allclose(
        [row["physical_time_seconds"] for row in critical], [0.04, 0.08, 0.12]
    )
    assert [row["capture_stage"] for row in critical] == [
        "physical_grid_without_sensing_boundary",
        "physical_grid_without_sensing_boundary",
        "sensing_boundary_after_publication",
    ]
    assert [row["published_version"] for row in critical] == [128, 128, 129]
    for snapshot in critical[:2]:
        np.testing.assert_array_equal(snapshot["learner_state"].optimizer_state, [987.0])
        assert not snapshot["used_at_sensing_boundary"]
    np.testing.assert_array_equal(critical[-1]["state"], result.control_traces["actual_state"][1])
    np.testing.assert_array_equal(critical[-1]["learner_state"].optimizer_state, [994.0])


def test_recorded_model_cache_preserves_exact_fields_across_fault_and_recovery(
    harness: Any,
) -> None:
    scene = replace(
        harness.scene,
        event_time=0.04,
        recovery_time=0.10,
        effectiveness_after=(0.7, 1.0, 0.8, 1.0),
        lag_multipliers_after=(3.0, 1.4, 2.0, 1.0),
    )
    cache = runtime._RecordedActuatorModels(scene, harness.model)
    fields = ("command_lower", "command_upper", "effectiveness", "time_constants")
    times = [0, 0.04 - 2e-10, 0.04 - 1e-10, 0.04, 0.07, 0.10 - 2e-10, 0.10 - 1e-10, 0.10, 0.14]
    for when in times:
        expected = scene.model_at(when, harness.model)
        for actual, name in zip(cache.at(when), fields, strict=True):
            assert actual.dtype == np.asarray(getattr(expected, name)).dtype
            np.testing.assert_array_equal(actual, np.asarray(getattr(expected, name)))
            assert not actual.flags.writeable
    assert len(cache.cache) == 2


def test_cached_recording_preserves_every_dense_field(harness: Any, monkeypatch: Any) -> None:
    scene = replace(harness.scene, effectiveness_after=(0.8, 1, 1, 1), recovery_time=0.10)
    optimized = harness.run(method="F2", scene_override=scene, name="cached")

    class UncachedModels:
        """Retain the original per-node host conversion for exact parity comparison."""

        def __init__(self, source: Any, nominal: Any) -> None:
            """Bind the original model query inputs."""
            self.source, self.nominal = source, nominal

        def at(self, when: float) -> Any:
            """Return the original conversion on every queried node."""
            model = self.source.model_at(when, self.nominal)
            return tuple(
                np.asarray(getattr(model, field))
                for field in ("command_lower", "command_upper", "effectiveness", "time_constants")
            )

    monkeypatch.setattr(runtime, "_RecordedActuatorModels", UncachedModels)
    original = harness.run(method="F2", scene_override=scene, name="uncached")
    assert_complete(optimized)
    assert_complete(original)
    assert optimized.dense_traces.keys() == original.dense_traces.keys()
    for key in original.dense_traces:
        np.testing.assert_array_equal(
            optimized.dense_traces[key], original.dense_traces[key], err_msg=key
        )


@pytest.mark.parametrize("method", ["F2", "A"])
def test_observation_cache_preserves_complete_deterministic_run_and_finite_updates(
    harness: Any, method: str
) -> None:
    scene = replace(
        harness.scene,
        event_time=0.04,
        recovery_time=0.08,
        effectiveness_after=(0.8, 1.0, 0.9, 1.0),
        lag_multipliers_after=(2.0, 1.0, 1.5, 1.0),
    )
    observation = ActuatorObservationConfig(
        parameter_delay_seconds=0.02,
        effectiveness_bias=-0.01,
        lag_scale=1.1,
        position_noise_m=0.001,
        velocity_noise_mps=0.002,
        attitude_noise_rad=0.001,
        rate_noise_rps=0.001,
        motor_noise_N=0.00001,
        obstacle_position_bias_m=-0.003,
    )
    harness.clock.now = 50.0
    begin = len(harness.controller_calls)
    original = harness.run(
        name=f"{method}-original-inputs",
        method=method,
        scene_override=scene,
        observation_config=observation,
        cache_observation_inputs=False,
    )
    original_calls = harness.controller_calls[begin:]
    harness.clock.now = 50.0
    begin = len(harness.controller_calls)
    cached = harness.run(
        name=f"{method}-cached-inputs",
        method=method,
        scene_override=scene,
        observation_config=observation,
        cache_observation_inputs=True,
    )
    cached_calls = harness.controller_calls[begin:]
    assert_complete(original)
    assert_complete(cached)
    for kind in ("dense_traces", "control_traces"):
        expected, actual = getattr(original, kind), getattr(cached, kind)
        assert expected.keys() == actual.keys()
        for name in expected:
            assert expected[name].dtype == actual[name].dtype, name
            np.testing.assert_array_equal(expected[name], actual[name], err_msg=f"{kind}.{name}")
    assert len(original_calls) == len(cached_calls)
    for expected, actual in zip(original_calls, cached_calls, strict=True):
        for name in expected:
            np.testing.assert_array_equal(expected[name], actual[name], err_msg=name)
    assert runtime._hash_tree(original.final_learner_state) == runtime._hash_tree(
        cached.final_learner_state
    )
    assert cached.summary["finite_credited_updates"] == (3 if method == "A" else 0)
    assert original.summary["cache_observation_inputs_enabled"] is False
    assert original.summary["observation_model_materializations"] == []
    assert cached.summary["cache_observation_inputs_enabled"] is True
    assert cached.summary["observation_model_materializations"] == [
        {"requested_time_seconds": 0.0, "observed_time_seconds": 0.0, "active_fault_phase": False},
        {"requested_time_seconds": 0.08, "observed_time_seconds": 0.06, "active_fault_phase": True},
    ]


def test_whole_hold_precheck_skips_adjacent_audits_only_when_strictly_separated(
    harness: Any, monkeypatch: Any
) -> None:
    calls = []
    audit = runtime._audit_interval

    def traced(world: Any, times: Any, states: Any) -> Any:
        calls.append(len(times))
        return audit(world, times, states)

    monkeypatch.setattr(runtime, "_audit_interval", traced)
    result = harness.run(method="F2", plant_level="P0")
    assert_complete(result)
    assert len(calls) == result.summary["control_count"] + 1  # complete holds plus final audit
    assert all(count > 2 for count in calls)
    for lower in (-1e-8, 0.0, 1e-10, 1e-9, None, float("nan")):
        assert not runtime._strictly_separated(
            {
                "terminate": False,
                "audit": {
                    "actual_xml_sphere_geometry": {"minimum_clearance_lower_bound_m": lower},
                    "actual_xml_ground_geometry": {"minimum_clearance_lower_bound_m": 1.0},
                },
            }
        )
    monkeypatch.setattr(runtime, "_strictly_separated", lambda audited: False)
    original = harness.run(method="F2", plant_level="P0", name="all-adjacent-audits")
    assert_complete(original)
    for key in original.dense_traces:
        np.testing.assert_array_equal(
            result.dense_traces[key], original.dense_traces[key], err_msg=key
        )


def test_whole_hold_precheck_retains_exact_first_contact_prefix_and_geometry(
    harness: Any, monkeypatch: Any
) -> None:
    world = replace(
        harness.scene.world,
        config=replace(harness.scene.world.config, obstacle_count=1),
        obstacle_mean_centers=np.asarray([[0.11, 0, harness.initial[2] + 0.02]]),
        obstacle_amplitudes=np.zeros((1, 3)),
        obstacle_angular_frequencies=np.zeros(1),
        obstacle_phases=np.zeros(1),
        obstacle_radii=np.asarray([0.02]),
    )
    scene = replace(harness.scene, world=world, event_time=0.02, effectiveness_after=(0.8, 1, 1, 1))
    optimized = harness.run(method="F2", plant_level="P0", scene_override=scene, name="prechecked")
    monkeypatch.setattr(runtime, "_strictly_separated", lambda result: False)
    original = harness.run(
        method="F2", plant_level="P0", scene_override=scene, name="chronological"
    )
    assert (
        optimized.summary["termination"] == original.summary["termination"] == "physical_collision"
    )
    assert optimized.summary["physical_time_seconds"] < 0.02  # future fault remains unexecuted
    assert optimized.summary["collision"] == original.summary["collision"]
    assert len(optimized.control_traces["time"]) == len(original.control_traces["time"]) == 1
    for key in original.dense_traces:
        np.testing.assert_array_equal(
            optimized.dense_traces[key], original.dense_traces[key], err_msg=key
        )
    np.testing.assert_array_equal(optimized.final_state, original.final_state)
    np.testing.assert_array_equal(optimized.dense_traces["actual_effectiveness"], 1)


def test_snapshot_serialization_preserves_float64_physical_state(
    harness: Any, monkeypatch: Any
) -> None:
    recorded = []

    def save(
        state: Any, contract: Any, physical: Any, stem: Any, *, config: Any, metadata: Any
    ) -> Any:
        physical = np.asarray(physical)
        recorded.append((physical.copy(), metadata))
        arrays = stem.parent / (stem.name + ".npz")
        manifest = stem.parent / (stem.name + ".json")
        with arrays.open("xb") as stream:
            np.savez(stream, physical_state=physical)
        manifest.write_text(json.dumps({"metadata": metadata}))
        return arrays, manifest

    monkeypatch.setattr(runtime, "save_actuator_learner_checkpoint", save)
    result = harness.run(
        method="F2", plant_level="P1", save_checkpoints=True, capture_times=(0.04,)
    )
    assert_complete(result)
    for physical, metadata in recorded:
        assert physical.dtype == np.float64
        assert runtime._hash_tree(physical) == metadata["physical_state_sha256"]
    critical = next(
        physical for physical, metadata in recorded if metadata["label"].startswith("critical-")
    )
    np.testing.assert_array_equal(critical, result.control_traces["actual_state"][1])
    assert not np.array_equal(critical, critical.astype(np.float32).astype(np.float64))
