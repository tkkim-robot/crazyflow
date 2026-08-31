from __future__ import annotations

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.dynamic_filter import (
    DynamicFilterConfig,
    dynamic_discrete_runtime_step,
)
from crazyflow.safety.da_plcbf.dynamic_rollouts import DynamicSphereScenarioBatch
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig


def _helpers() -> dict[str, object]:
    return runpy.run_path(
        str(Path(__file__).with_name("test_dynamic_rollouts.py")),
        run_name="dynamic_rollout_test_helpers",
    )


def _safe_window(nodes: int = 5) -> DynamicSphereScenarioBatch:
    return DynamicSphereScenarioBatch(
        obstacle_centers=jnp.broadcast_to(jnp.array([1.5, 1.5, 1.5]), (1, 2, nodes, 1, 3)),
        obstacle_radii=jnp.full((1, 2, nodes, 1), 0.1),
        obstacle_mask=jnp.ones((1, 2, nodes, 1), dtype=bool),
        arena_lower=jnp.array([[-2.0, -2.0, 0.1]]),
        arena_upper=jnp.array([[2.0, 2.0, 2.0]]),
        speed_limit=jnp.array([3.0]),
        angular_rate_max=jnp.array([8.0]),
        tilt_max_radians=jnp.array([1.1]),
    )


def test_dynamic_discrete_step_uses_equal_horizons_and_exact_postcheck_on_gpu_jit_path() -> None:
    helper = _helpers()
    model, actuator = helper["_physical"]()
    spec, params, actor_config = helper["_actor"](1)
    initial = helper["_initial"]()[0]
    config = DynamicFilterConfig()

    def control(state: jax.Array) -> object:
        return dynamic_discrete_runtime_step(
            state,
            jnp.array([0.8, 0.0, 1.0]),
            jnp.zeros(3),
            jnp.array(-1),
            params,
            spec,
            _safe_window(),
            model,
            actuator,
            actor_config,
            QuadPolicyConfig(),
            VersionABarrierConfig(obstacle_clearance=0.05),
            config,
            dt=0.02,
            policy_gain=1.5,
        )

    result = jax.jit(control)(initial)
    expected_residual = (
        result.applied_next_value - config.decay * result.selection.selected_hard_value
    )

    assert bool(result.selection.has_certificate)
    assert bool(result.filter.proposal_accepted)
    assert not bool(result.filter.degraded)
    assert not bool(result.degraded)
    assert result.applied_interval_margin >= -config.tolerance
    np.testing.assert_allclose(expected_residual, result.filter.proposal_exact_residual, atol=1e-6)
    assert np.all(np.asarray(result.motor_forces) >= float(actuator.thrust_min) - 1e-6)
    assert np.all(np.asarray(result.motor_forces) <= float(actuator.thrust_max) + 1e-6)


def test_no_robustly_safe_dynamic_policy_is_explicitly_degraded_not_certified() -> None:
    helper = _helpers()
    model, actuator = helper["_physical"]()
    scenarios = helper["_dynamic_scenarios"](predictions=2, nodes=6)
    scenarios = scenarios._replace(
        obstacle_centers=scenarios.obstacle_centers.at[:, 1, 0].set(
            scenarios.obstacle_centers[:, 0, 0]
        )
    )
    spec, params, actor_config = helper["_actor"](1)
    result = dynamic_discrete_runtime_step(
        helper["_initial"]()[0],
        jnp.array([0.8, 0.0, 1.0]),
        jnp.zeros(3),
        jnp.array(-1),
        params,
        spec,
        scenarios,
        model,
        actuator,
        actor_config,
        QuadPolicyConfig(),
        VersionABarrierConfig(obstacle_clearance=0.05),
        DynamicFilterConfig(),
        dt=0.02,
        policy_gain=1.5,
    )

    assert np.all(np.asarray(result.library.hard_values) < 0)
    assert not bool(result.selection.has_certificate)
    assert not bool(result.filter.proposal_accepted)
    assert not bool(result.filter.fallback_accepted)
    assert bool(result.filter.degraded)
    assert bool(result.degraded)
    assert result.applied_interval_margin < 0
