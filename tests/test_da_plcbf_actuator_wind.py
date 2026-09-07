"""Wind clock, physical model events and delayed observer cache must agree."""

from dataclasses import replace

import jax
import numpy as np

from benchmark.da_plcbf_recovery_wind import demonstration_scene
from crazyflow.safety.da_plcbf.actuator_inputs import CausalObservationInputCache
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_study import (
    ActuatorObservationConfig,
    make_actuator_scene,
    nominal_actuator_model,
    observed_actuator_model,
)


def test_wind_events_and_faults_compose_and_cache_preserves_delayed_observations():
    scene = replace(demonstration_scene(), recovery_time=31.0)
    nominal = nominal_actuator_model()
    observation = ActuatorObservationConfig(parameter_delay_seconds=0.08)
    cache = CausalObservationInputCache(scene, nominal, observation, ActuatorFilterConfig())
    assert cache.model_materializations == []
    for when in (0.0, 3.04, 3.08, 11.08, 21.08, 23.08, 29.08, 31.08, 37.08, 5.0):
        cached = cache.model_at(when)
        reference = observed_actuator_model(scene, when, nominal, observation)
        for a, b in zip(jax.tree.leaves(cached), jax.tree.leaves(reference), strict=True):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(
            cached.body.wind_velocity, scene.world.wind_at(max(0, when - 0.08))
        )
    events = scene.events(nominal)
    assert [event.time for event in events] == [3, 11, 21, 23, 29, 31, 37]
    for event in events:
        expected = scene.model_at(event.time, nominal)
        for a, b in zip(jax.tree.leaves(event.model), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(scene.model_at(31, nominal).body.wind_velocity, [-1, 0.8, 0])
    np.testing.assert_array_equal(scene.model_at(31, nominal).effectiveness, nominal.effectiveness)


def test_preflight_obstacle_clock_is_continuous_and_identity_records_wind():
    scene = demonstration_scene()
    original = make_actuator_scene(62104, "navigation", "effectiveness")
    center0, _ = original.world.obstacle_kinematics(0)
    for when in (0, 3, 18.999):
        centers, velocity = scene.world.obstacle_kinematics(when)
        np.testing.assert_array_equal(centers, center0)
        np.testing.assert_array_equal(velocity, np.zeros_like(velocity))
    for when in (19, 20, 25, 40):
        actual = scene.world.obstacle_kinematics(when)
        expected = original.world.obstacle_kinematics(when - 19)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a, b)
    assert "wind_events" not in original.physical_spec()
    assert len(scene.physical_spec()["wind_events"]) == 5
    assert scene.metadata()["physical_world_id"] != original.metadata()["physical_world_id"]


def test_recorded_wind_visual_matches_physics_and_changes_acceleration():
    from types import SimpleNamespace

    import jax.numpy as jnp

    from benchmark.da_plcbf_actuator_video import ReplayEpisode
    from crazyflow.safety.da_plcbf.actuator_dynamics import augmented_dynamics
    from crazyflow.safety.da_plcbf.actuator_study import initial_augmented_state

    scene = demonstration_scene()
    replay = SimpleNamespace(world=scene.world.metadata())
    nominal = nominal_actuator_model()
    state = jnp.asarray(initial_augmented_state(scene.world.initial_state, nominal))
    for when in (0, 3, 10, 11, 19, 23, 29, 37, 43):
        np.testing.assert_array_equal(
            ReplayEpisode.wind_at(replay, when), scene.world.wind_at(when)
        )
        np.testing.assert_array_equal(
            ReplayEpisode.obstacle_centers(replay, when), scene.world.obstacle_kinematics(when)[0]
        )
    np.testing.assert_allclose(ReplayEpisode.wind_displacement(replay, 11), [12.8, 6.4, 0])
    calm = augmented_dynamics(state, state[13:], nominal)
    windy = augmented_dynamics(state, state[13:], scene.model_at(3, nominal))
    assert np.linalg.norm(np.asarray(windy[7:10] - calm[7:10])) > 0.01
