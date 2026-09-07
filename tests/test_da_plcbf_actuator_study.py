"""Physical pairing, absolute-clock observations, and common rescue contracts."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.actuator_learning import ActuatorSkillConfig
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_study import (
    DYNAMICS_CELLS,
    ActuatorObservationConfig,
    build_actuator_controller,
    initial_augmented_state,
    make_actuator_scene,
    nominal_actuator_model,
    observe_actuator_state,
    observed_actuator_model,
)
from crazyflow.safety.da_plcbf.persistent_skill_learner import build_fibonacci_skill_spec


def test_campaign_and_protocol_bind_indirect_runtime_dependencies():
    from benchmark.da_plcbf_actuator_protocol import runner_source_files as protocol_files
    from benchmark.da_plcbf_actuator_study import ROOT, runner_source_files

    assert protocol_files is runner_source_files
    relative = {str(path.relative_to(ROOT)) for path in runner_source_files()}
    assert {
        "crazyflow/safety/da_plcbf/deadline_schedule.py",
        "crazyflow/safety/da_plcbf/selector.py",
        "crazyflow/safety/da_plcbf/version_a_barriers.py",
        "crazyflow/safety/da_plcbf/version_a_filter.py",
        "crazyflow/safety/da_plcbf/polytope_qp.py",
        "crazyflow/safety/da_plcbf/navigation_experiment.py",
        "crazyflow/dynamics/first_principles/params.toml",
        "crazyflow/drones/cf21B_500.xml",
        "benchmark/da_plcbf_actuator_study.py",
    } <= relative
    assert all(Path(path).suffix != ".pyc" for path in relative)


@pytest.mark.parametrize("family", ["structured", "navigation"])
def test_dynamics_cells_preserve_geometry_and_world_clock(family: str):
    scenes = [make_actuator_scene(107, family, cell) for cell in DYNAMICS_CELLS]
    baseline = scenes[0].world
    times = np.asarray([0.0, 1.9, 2.0, 2.041, 13.1])
    centers, velocities = baseline.obstacle_kinematics(times)
    assert centers.shape == (len(times), baseline.config.obstacle_count, 3)
    assert velocities.shape == centers.shape
    assert np.isfinite(centers).all()
    step = 1e-5
    plus = baseline.obstacle_kinematics(times + step)[0]
    minus = baseline.obstacle_kinematics(times - step)[0]
    np.testing.assert_allclose((plus - minus) / (2 * step), velocities, atol=1e-8)
    for scene in scenes[1:]:
        np.testing.assert_array_equal(scene.world.initial_state, baseline.initial_state)
        np.testing.assert_array_equal(scene.world.waypoint_positions, baseline.waypoint_positions)
        np.testing.assert_array_equal(scene.world.obstacle_kinematics(times)[0], centers)
        assert not scene.world.initial_state.flags.writeable
    assert len({scene.metadata()["physical_world_id"] for scene in scenes}) == 4


def test_physical_identity_ignores_labels_but_includes_task_limits_and_motor_events():
    scene = make_actuator_scene(107, "navigation", "combined")
    identity = scene.metadata()["physical_world_id"]
    relabeled = replace(scene, scene_seed=991, family="administrative_alias")
    assert relabeled.metadata()["physical_world_id"] == identity
    assert replace(scene, event_time=2.04).metadata()["physical_world_id"] != identity
    world = replace(scene.world, config=replace(scene.world.config, speed_max=3.4))
    assert replace(scene, world=world).metadata()["physical_world_id"] != identity
    world = replace(scene.world, waypoint_positions=scene.world.waypoint_positions + [0, 0, 0.01])
    assert replace(scene, world=world).metadata()["physical_world_id"] != identity


def test_model_observation_is_current_or_past_with_no_future_event_leak():
    scene = make_actuator_scene(107, "navigation", "combined")
    nominal = nominal_actuator_model()
    oracle = ActuatorObservationConfig()
    delayed = ActuatorObservationConfig(parameter_delay_seconds=0.2)
    for when in (0.0, 1.5, 1.999):
        model = observed_actuator_model(scene, when, nominal, oracle)
        for a, b in zip(jax.tree.leaves(model), jax.tree.leaves(nominal), strict=True):
            np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(
        observed_actuator_model(scene, 2.0, nominal, oracle).effectiveness,
        np.asarray(scene.effectiveness_after, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        observed_actuator_model(scene, 2.1, nominal, delayed).effectiveness, nominal.effectiveness
    )
    np.testing.assert_array_equal(
        observed_actuator_model(scene, 2.2, nominal, delayed).time_constants,
        scene.model_at(2.0, nominal).time_constants,
    )


def test_common_noise_depends_on_absolute_clock_and_channel_not_query_order():
    scene = make_actuator_scene(109, "navigation", "combined")
    actual = initial_augmented_state(scene.world.initial_state, nominal_actuator_model())
    config = ActuatorObservationConfig(
        position_noise_m=0.002,
        velocity_noise_mps=0.005,
        attitude_noise_rad=0.002,
        rate_noise_rps=0.004,
        motor_noise_N=0.001,
    )
    first = observe_actuator_state(actual, 2.012, 109, config)
    observe_actuator_state(actual, 8.0, 109, config)
    np.testing.assert_array_equal(first, observe_actuator_state(actual, 2.012, 109, config))
    shifted = actual.copy()
    shifted[:3] += [0.1, -0.2, 0.3]
    shifted[7:10] += [0.2, 0.1, 0.1]
    second = observe_actuator_state(shifted, 2.012, 109, config)
    np.testing.assert_allclose(second - first, shifted - actual, atol=1e-15)
    assert not np.array_equal(first, observe_actuator_state(actual, 2.013, 109, config))
    assert not np.array_equal(first, observe_actuator_state(actual, 2.012, 110, config))
    assert np.isclose(np.linalg.norm(first[3:7]), 1)


def test_noisy_motor_observation_is_not_silently_clipped():
    scene = make_actuator_scene(109, "navigation", "combined")
    actual = initial_augmented_state(scene.world.initial_state, nominal_actuator_model())
    observed = observe_actuator_state(
        actual, 0.0, 109, ActuatorObservationConfig(motor_noise_N=100.0)
    )
    assert np.any((observed[13:] < 0) | (observed[13:] > 1))


def test_nominal_and_emergency_commands_share_f2_for_all_fallback_maps():
    scene = make_actuator_scene(107, "structured", "combined")
    model = scene.model_at(2.0, nominal_actuator_model())
    state = jnp.asarray(
        initial_augmented_state(scene.world.initial_state, nominal_actuator_model())
    )
    cfg = ActuatorSkillConfig(horizon=4, dt=0.02, control_interval_steps=2)
    spec = build_fibonacci_skill_spec(
        policy_count=4, minimum_duration=0.02, maximum_duration=0.06, horizon_duration=0.08
    )
    filters = ActuatorFilterConfig(horizon=4, dt=0.02, command_hold_steps=2)
    outputs = []
    for mode in ("F0", "F1", "F2"):
        fns = build_actuator_controller(spec, replace(cfg, adapter_mode=mode), filters)
        outputs.append(
            (
                fns.nominal(state, state[:3] + jnp.asarray([1.0, 0.2, 0.1]), model),
                fns.emergency(state, model),
            )
        )
    for output in outputs[1:]:
        for a, b in zip(jax.tree.leaves(outputs[0]), jax.tree.leaves(output), strict=True):
            np.testing.assert_array_equal(a, b)
