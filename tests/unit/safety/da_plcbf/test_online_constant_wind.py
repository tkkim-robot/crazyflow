from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.continuous_demo_scenarios import model_with_wind
from crazyflow.safety.da_plcbf.online_constant_wind import (
    OnlineConstantWindConfig,
    build_cf21b_version_a_resources,
    run_online_constant_wind_demo,
)
from crazyflow.safety.da_plcbf.point_wind_estimator import (
    PointWindEstimatorConfig,
    initialize_point_wind_estimator,
    update_point_wind_estimator,
)
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step


def test_point_wind_filter_recovers_the_single_telemetry_change_without_oracle_input() -> None:
    resources = build_cf21b_version_a_resources()
    true_wind = jnp.asarray([0.9, 0.55, 0.0], dtype=jnp.float32)
    true_model = model_with_wind(resources.model, true_wind)
    state = jnp.asarray(
        [0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32
    )
    hover = jnp.asarray([-resources.model.mass * resources.model.gravity_vec[2], 0.0, 0.0, 0.0])
    estimator = initialize_point_wind_estimator()
    errors = []
    for _ in range(180):
        following = direct_wrench_symplectic_step(state, hover, true_model, 0.02)
        update = update_point_wind_estimator(
            estimator,
            state,
            following,
            hover,
            resources.model,
            dt=0.02,
            config=PointWindEstimatorConfig(response_rate=2.4),
        )
        np.testing.assert_allclose(update.instantaneous_wind, true_wind, atol=5e-5, rtol=0.0)
        estimator = update.state
        state = following
        errors.append(float(jnp.linalg.norm(estimator.wind_velocity - true_wind)))

    assert np.all(np.diff(errors) < 0.0)
    assert errors[-1] < 4e-4
    assert int(estimator.update_count) == 180
    assert int(estimator.finite_update_count) == 180


def test_short_gpu_end_to_end_publishes_every_finite_step_and_improves_common_state_library() -> (
    None
):
    gpu = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpu:
        pytest.skip("the end-to-end corrected demonstration is explicitly a CUDA/JIT check")
    result = run_online_constant_wind_demo(OnlineConstantWindConfig(steps=405), device=gpu[0])
    summary = result.summary

    assert summary["true_wind_change_count"] == 1
    assert summary["wind_change_time_seconds"] == 4.0
    assert summary["prewind_max_full_state_component_difference"] == 0.0
    assert summary["adaptive_gradient_steps"] > 0
    assert summary["adaptive_library_version"] == summary["adaptive_gradient_steps"]
    assert summary["library_version_equals_finite_gradient_steps"] is True
    assert summary["all_attempted_bptt_updates_finite"] is True
    assert summary["adaptive_parameter_delta_norm"] > 0.0
    assert (
        summary["common_state_adaptive_descriptor_target_loss"]
        < summary["common_state_fixed_descriptor_target_loss"]
    )
    assert (
        summary["common_state_adaptive_diversity_loss"]
        < summary["common_state_fixed_diversity_loss"]
    )
    assert summary["adaptive_minimum_inflated_clearance_m"] > 0.0
    assert summary["adaptive_degraded_step_count"] == 0
    assert summary["acceptance_frozen_library_has_postwind_descriptor_bias"] is True
    assert summary["acceptance_adaptive_shared_probe_target_recovery"] is True
    assert summary["acceptance_adaptive_shared_probe_spread_recovery"] is True
