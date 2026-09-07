"""Exact current/past input preparation without future-model materialization."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.actuator_inputs import CausalObservationInputCache, reference_inputs
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_study import (
    ActuatorObservationConfig,
    initial_augmented_state,
    make_actuator_scene,
    nominal_actuator_model,
)


def test_only_queried_current_past_model_phases_are_materialized() -> None:
    scene = replace(make_actuator_scene(1002, "structured", "combined"), recovery_time=4.0)
    query_times = []

    class ObservedScene:
        """Record the only model observations actually requested by the cache."""

        def __getattr__(self, name: str) -> Any:
            return getattr(scene, name)

        def model_at(self, when: float, nominal: Any) -> Any:
            query_times.append(when)
            return scene.model_at(when, nominal)

    cache = CausalObservationInputCache(
        ObservedScene(),
        nominal_actuator_model(),
        ActuatorObservationConfig(parameter_delay_seconds=0.08),
        ActuatorFilterConfig(horizon=2),
    )
    assert query_times == [] and cache.model_materializations == []
    cache.model_at(0.0)
    cache.model_at(2.04)
    assert query_times == [0.0]
    cache.model_at(2.08)
    assert query_times == [0.0, 2.0]
    cache.model_at(4.08)
    cache.model_at(1.0)
    assert query_times == [0.0, 2.0]
    assert all(
        row["observed_time_seconds"] <= row["requested_time_seconds"]
        for row in cache.model_materializations
    )


@pytest.mark.parametrize("physical_dtype", [np.float32, np.float64])
def test_integrated_inputs_keep_every_leaf_byte_including_signed_zero_and_noise(
    physical_dtype: Any,
) -> None:
    nominal = nominal_actuator_model()
    scene = replace(make_actuator_scene(1002, "structured", "combined"), recovery_time=4.0)
    scene = replace(
        scene,
        world=replace(
            scene.world,
            obstacle_mean_centers=np.full_like(scene.world.obstacle_mean_centers, -0.0),
            obstacle_amplitudes=np.zeros_like(scene.world.obstacle_amplitudes),
            obstacle_phases=np.full_like(scene.world.obstacle_phases, -0.0),
        ),
    )
    actual = initial_augmented_state(scene.world.initial_state, nominal).astype(physical_dtype)
    actual[0] = physical_dtype(0.123456789012345)
    observation = ActuatorObservationConfig(
        parameter_delay_seconds=0.08,
        effectiveness_bias=-0.02,
        lag_scale=1.1,
        position_noise_m=0.001,
        velocity_noise_mps=0.002,
        attitude_noise_rad=0.001,
        rate_noise_rps=0.003,
        motor_noise_N=0.00001,
        obstacle_position_bias_m=-0.0,
    )
    config = ActuatorFilterConfig(horizon=4)
    cache = CausalObservationInputCache(scene, nominal, observation, config)
    for when in (0.0, -0.0, 2.08 - 2e-10, 2.08 - 1e-10, 2.08, 4.08, 0.04, -0.0):
        expected = reference_inputs(actual, when, scene, nominal, observation, config)
        result = cache.at(actual, when)
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(result), strict=True):
            a, b = np.asarray(a), np.asarray(b)
            assert a.dtype == b.dtype and a.shape == b.shape
            assert a.tobytes() == b.tobytes()
