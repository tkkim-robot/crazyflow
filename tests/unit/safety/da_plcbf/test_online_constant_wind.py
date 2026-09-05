from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.continuous_demo_scenarios import (
    constant_wind_scenario,
    model_with_wind,
)
from crazyflow.safety.da_plcbf.online_constant_wind import (
    OnlineConstantWindConfig,
    build_cf21b_version_a_resources,
    comparison_trace_for_methods,
    load_online_constant_wind_result,
    run_online_constant_wind_demo,
    save_online_constant_wind_result,
)
from crazyflow.safety.da_plcbf.point_wind_estimator import (
    PointWindEstimatorConfig,
    initialize_point_wind_estimator,
    update_point_wind_estimator,
)
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step

if TYPE_CHECKING:
    from pathlib import Path


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


def test_matched_methods_publish_finite_updates_and_save_failure_diagnostics(
    tmp_path: Path,
) -> None:
    gpu = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpu:
        pytest.skip("the integrated controller comparison is explicitly a CUDA/JIT check")
    scenario = replace(constant_wind_scenario(), steps=12, horizon=4, wind_change_step=2)
    result = run_online_constant_wind_demo(
        OnlineConstantWindConfig(
            policy_count=4, probe_every_steps=1, probe_window_seconds=(0.08, 0.22)
        ),
        scenario=scenario,
        device=gpu[0],
    )
    summary = result.summary
    assert set(result.methods) == {"fixed", "adaptive", "analytic", "compensated"}
    assert summary["prewind_max_full_state_component_difference"] == 0.0
    assert summary["adaptive_gradient_steps"] > 0
    assert summary["adaptive_library_version"] == summary["adaptive_gradient_steps"]
    assert summary["all_attempted_bptt_updates_finite"] is True
    assert summary["adaptive_parameter_delta_norm"] > 0.0
    assert summary["methods"]["analytic"]["maximum_policy_dual"] == 0.0
    assert summary["shared_probes"]
    for probe in summary["shared_probes"]:
        assert set(probe["adaptive_state_coverage"]) == set(result.methods)
        assert len(probe["adaptive_state_full_state"]) == 13
        assert len(probe["adaptive_state_point_wind"]) == 3
        nominal_values = []
        for coverage in probe["adaptive_state_coverage"].values():
            nominal_values.append(coverage["shared_nominal_value"])
            assert coverage["maximum_library_value"] == max(
                coverage["shared_nominal_value"], coverage["maximum_fallback_value"]
            )
        assert len(set(nominal_values)) == 1
    for name, method in result.methods.items():
        assert method.maximum_library_value.shape == (12,)
        assert method.qp_rejection_flags.shape == (12, 8)
        assert method.estimated_wind.shape == (12, 3)
        assert summary["methods"][name]["controller_timing"]["count"] == 12
        assert summary["methods"][name]["point_wind_finite_updates"] == 12
        if name != "adaptive":
            np.testing.assert_array_equal(method.library_version, 0)
    trace_path, summary_path = save_online_constant_wind_result(result, tmp_path)
    loaded = load_online_constant_wind_result(trace_path, summary_path)
    pair = comparison_trace_for_methods(loaded, "analytic", "compensated")
    assert pair.fixed.control_mode == "analytic"
    assert pair.coverage_probes is not None
    assert pair.coverage_probes["time_seconds"].shape == (len(summary["shared_probes"]),)
    np.testing.assert_array_equal(
        pair.adaptive.maximum_library_value, result.methods["compensated"].maximum_library_value
    )
    assert loaded.summary["all_checks_passed"] is False  # Short failures remain saved evidence.


def test_relative_swept_clearance_accounts_for_between_sample_collision() -> None:
    from crazyflow.safety.da_plcbf.online_constant_wind import _swept_clearance

    positions = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    centers = np.zeros((2, 1, 3))
    minimum, step = _swept_clearance(positions, centers, np.asarray([0.25]))
    assert minimum == -0.25
    assert step == 0
