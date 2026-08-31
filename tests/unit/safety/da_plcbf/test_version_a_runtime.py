from __future__ import annotations

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.capsules import CapsuleObstacleSet
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig
from crazyflow.safety.da_plcbf.version_a_filter import VersionAFilterConfig
from crazyflow.safety.da_plcbf.version_a_runtime import version_a_runtime_step
from crazyflow.safety.da_plcbf.version_b_evidence import _version_a_mapping


def _problem() -> tuple[object, ...]:
    helper = runpy.run_path(
        str(Path(__file__).with_name("test_certificates.py")), run_name="certificate_test_helpers"
    )
    return helper["_problem"]()


def test_complete_version_a_runtime_step_is_jittable_and_independently_replayable() -> None:
    model, actuator, spec, scenarios, safety, actor_config, params, state = _problem()

    def control(candidate_state: jax.Array) -> object:
        return version_a_runtime_step(
            candidate_state,
            jnp.array([1.0, 0.0, 1.0]),
            jnp.zeros(3),
            params,
            spec,
            scenarios,
            safety,
            model,
            actuator,
            actor_config,
            QuadPolicyConfig(),
            VersionABarrierConfig(minimum_tie_tolerance=1e-7),
            VersionAFilterConfig(),
            dt=0.02,
            certificate_horizon=4,
            policy_gain=1.5,
        )

    result = jax.jit(control)(state)
    independent = direct_wrench_symplectic_step(state, result.action, model, 0.02)

    assert bool(result.continuous_filter.has_certificate)
    assert bool(result.proposal_interval_accepted)
    assert bool(result.proposal_discrete_accepted)
    assert bool(result.applied_discrete_applicable)
    assert not bool(result.used_interval_midpoint)
    assert not bool(result.degraded)
    assert result.applied_interval_margin >= -1e-6
    assert result.applied_next_policy_value >= -1e-6
    assert result.applied_discrete_residual >= -1e-6
    np.testing.assert_allclose(result.next_state, independent, atol=1e-7)
    assert np.all(result.applied_continuous_postcheck.motor_forces >= actuator.thrust_min - 3e-6)
    assert np.all(result.applied_continuous_postcheck.motor_forces <= actuator.thrust_max + 3e-6)


def test_current_collision_cannot_be_reported_as_an_accepted_runtime_certificate() -> None:
    model, actuator, spec, scenarios, safety, actor_config, params, state = _problem()
    colliding = state.at[:3].set(scenarios.obstacle_centers[0, 0])
    result = version_a_runtime_step(
        colliding,
        jnp.array([1.0, 0.0, 1.0]),
        jnp.zeros(3),
        params,
        spec,
        scenarios,
        safety,
        model,
        actuator,
        actor_config,
        QuadPolicyConfig(),
        VersionABarrierConfig(minimum_tie_tolerance=1e-7),
        VersionAFilterConfig(),
        dt=0.02,
        certificate_horizon=4,
        policy_gain=1.5,
    )

    assert np.all(np.asarray(result.certificates.certificates.values) < 0)
    assert not bool(result.continuous_filter.has_certificate)
    assert not bool(result.proposal_interval_accepted)
    assert not bool(result.fallback_interval_accepted)
    assert bool(result.used_interval_midpoint)
    assert bool(result.degraded)
    assert result.applied_interval_margin < 0
    assert not bool(result.applied_continuous_postcheck.passed)


def test_capsule_collision_is_shared_by_certificate_filter_and_held_interval_postcheck() -> None:
    model, actuator, spec, scenarios, safety, actor_config, params, state = _problem()
    capsule = CapsuleObstacleSet(
        segment_start=jnp.array([[[0.1, 0.2, 0.7]]]),
        segment_end=jnp.array([[[0.1, 0.2, 1.3]]]),
        radii=jnp.array([[0.15]]),
        mask=jnp.array([[True]]),
    )
    result = version_a_runtime_step(
        state,
        jnp.array([1.0, 0.0, 1.0]),
        jnp.zeros(3),
        params,
        spec,
        scenarios,
        safety,
        model,
        actuator,
        actor_config,
        QuadPolicyConfig(),
        VersionABarrierConfig(minimum_tie_tolerance=1e-7),
        VersionAFilterConfig(),
        dt=0.02,
        certificate_horizon=2,
        policy_gain=1.5,
        capsules=capsule,
    )

    assert np.all(np.asarray(result.certificates.certificates.values) < 0)
    assert not bool(result.continuous_filter.has_certificate)
    assert not bool(result.proposal_interval_accepted)
    assert not bool(result.fallback_interval_accepted)
    assert bool(result.used_interval_midpoint)
    assert bool(result.degraded)
    assert float(result.applied_interval_margin) < 0.0


def test_currently_safe_continuous_proposal_is_rejected_when_equal_horizon_value_contracts() -> (
    None
):
    """Regression for a command that passed the local QP but lost its sampled certificate.

    This state was found by a fixed-seed boundary search.  The old runtime accepted the local
    continuous half-space and held-interval checks even though recomputing the *same selected
    policy* at the actual successor changed its H=4 hard value from positive to negative.
    """
    model, actuator, spec, scenarios, safety, actor_config, params, _ = _problem()
    state = jnp.array(
        [
            1.8656267,
            0.79516256,
            0.30006576,
            0.0,
            0.0,
            0.0,
            1.0,
            0.494835,
            -0.61596483,
            0.20097029,
            0.16192143,
            2.427488,
            -0.6541592,
        ]
    )
    target = jnp.array([0.3909463, -1.7122915, 0.26060957])

    result = version_a_runtime_step(
        state,
        target,
        jnp.zeros(3),
        params,
        spec,
        scenarios,
        safety,
        model,
        actuator,
        actor_config,
        QuadPolicyConfig(),
        VersionABarrierConfig(minimum_tie_tolerance=1e-7),
        VersionAFilterConfig(),
        dt=0.2,
        certificate_horizon=4,
        policy_gain=1.5,
    )

    selected = int(result.continuous_filter.selected_index)
    current_value = float(result.certificates.certificates.values[selected])
    assert current_value > 0.0
    assert bool(result.proposal_interval_accepted)
    assert float(result.proposal_next_policy_value) < 0.0
    assert float(result.proposal_discrete_residual) < 0.0
    assert not bool(result.proposal_discrete_accepted)
    assert not np.array_equal(
        np.asarray(result.action), np.asarray(result.continuous_filter.action)
    )
    assert bool(result.fallback_discrete_accepted)
    assert bool(result.used_interval_fallback)
    assert not bool(result.used_interval_midpoint)
    assert float(result.applied_next_policy_value) >= -1e-6
    assert float(result.applied_discrete_residual) >= -1e-6
    np.testing.assert_allclose(
        result.action, result.certificates.certificates.fallback_wrenches[selected], atol=1e-7
    )
    np.testing.assert_allclose(
        result.applied_continuous_postcheck.motor_forces,
        result.continuous_filter.fallback_postcheck.motor_forces,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.applied_continuous_postcheck.policy_barrier_residual,
        result.continuous_filter.fallback_postcheck.policy_barrier_residual,
        atol=1e-7,
    )
    assert not np.isclose(
        float(result.applied_continuous_postcheck.policy_barrier_residual),
        float(result.continuous_filter.applied_postcheck.policy_barrier_residual),
        atol=1e-5,
    )
    assert bool(result.applied_continuous_postcheck.passed)
    evidence = _version_a_mapping(result, result.nominal.wrench, latency=0.01, tolerance=2e-5)
    assert evidence["action"] == pytest.approx(np.asarray(result.action))
    assert evidence["continuous_policy_residual"] == pytest.approx(
        float(result.applied_continuous_postcheck.policy_barrier_residual)
    )
    assert evidence["continuous_policy_residual"] != pytest.approx(
        float(result.continuous_filter.applied_postcheck.policy_barrier_residual), abs=1e-5
    )
    assert evidence["applied_postcheck_passed"]
