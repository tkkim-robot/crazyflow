from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.capsules import CapsuleObstacleSet
from crazyflow.safety.da_plcbf.version_a_analytic_filter import version_a_analytic_filter
from crazyflow.safety.da_plcbf.version_a_barriers import (
    RigidBodySafetySet,
    VersionABarrierConfig,
    VersionAModel,
)
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator, VersionAFilterConfig


def _problem() -> tuple[VersionAModel, VersionAActuator, RigidBodySafetySet, jax.Array]:
    parameters: dict[str, Any] = load_params("cf21B_500")
    model = VersionAModel(
        mass=jnp.asarray(parameters["mass"]),
        gravity_vec=jnp.asarray(parameters["gravity_vec"]),
        inertia=jnp.asarray(parameters["J"]),
        inertia_inv=jnp.linalg.inv(jnp.asarray(parameters["J"])),
        drag_matrix=jnp.asarray(parameters["drag_matrix"]),
        wind_velocity=jnp.zeros(3),
        external_force=jnp.zeros(3),
        external_torque=jnp.zeros(3),
    )
    actuator = VersionAActuator(
        arm_length=jnp.asarray(parameters["L"]),
        thrust_to_torque=jnp.asarray(parameters["thrust2torque"]),
        mixing_matrix=jnp.asarray(parameters["mixing_matrix"]),
        thrust_min=jnp.asarray(parameters["thrust_min"]),
        thrust_max=jnp.asarray(parameters["thrust_max"]),
    )
    safety = RigidBodySafetySet(
        obstacle_centers=jnp.array([[1.0, 0.0, 1.0]]),
        obstacle_radii=jnp.array([0.2]),
        obstacle_mask=jnp.array([True]),
        arena_lower=jnp.array([-2.0, -2.0, 0.2]),
        arena_upper=jnp.array([2.0, 2.0, 2.0]),
        speed_max=jnp.asarray(3.0),
        angular_rate_max=jnp.asarray(8.0),
        tilt_max_radians=jnp.asarray(np.deg2rad(60.0)),
    )
    state = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return model, actuator, safety, state


def test_analytic_only_filter_accepts_safe_hover_without_policy_certificate() -> None:
    model, actuator, safety, state = _problem()
    hover = jnp.array([-float(model.mass * model.gravity_vec[2]), 0.0, 0.0, 0.0])
    barrier_config = VersionABarrierConfig()
    filter_config = VersionAFilterConfig()
    compiled = jax.jit(
        lambda current, nominal: version_a_analytic_filter(
            current, nominal, jnp.ones(4), model, actuator, safety, barrier_config, filter_config
        )
    )
    result = compiled(state, hover)

    assert bool(result.input_valid)
    assert bool(result.qp_accepted)
    assert not bool(result.degraded)
    assert bool(result.applied_postcheck.passed)
    np.testing.assert_allclose(result.action, hover, atol=3e-6)


def test_analytic_filter_projects_motor_infeasible_nominal_and_postchecks_result() -> None:
    model, actuator, safety, state = _problem()
    result = version_a_analytic_filter(
        state,
        jnp.array([2.0, 0.01, -0.01, 0.001]),
        jnp.ones(4),
        model,
        actuator,
        safety,
        VersionABarrierConfig(),
    )

    assert bool(result.qp_accepted)
    assert bool(result.applied_postcheck.actuator_passed)
    assert bool(result.applied_postcheck.analytic_passed)
    assert float(result.applied_postcheck.minimum_motor_margin) >= -3e-6


def test_analytic_filter_fails_closed_outside_hocbf_domain() -> None:
    model, actuator, safety, state = _problem()
    unsafe_state = state.at[0].set(1.0)
    result = version_a_analytic_filter(
        unsafe_state, jnp.zeros(4), jnp.ones(4), model, actuator, safety, VersionABarrierConfig()
    )

    assert not bool(result.input_valid)
    assert not bool(result.qp_accepted)
    assert bool(result.used_midpoint)
    assert bool(result.degraded)
    assert bool(result.action_executable)


def test_invalid_actuator_returns_nonexecutable_nan_sentinel() -> None:
    model, actuator, safety, state = _problem()
    invalid = actuator._replace(thrust_max=jnp.asarray(jnp.nan))
    result = version_a_analytic_filter(
        state, jnp.zeros(4), jnp.ones(4), model, invalid, safety, VersionABarrierConfig()
    )

    assert not bool(result.input_valid)
    assert not bool(result.action_executable)
    assert np.all(np.isnan(np.asarray(result.action)))


def test_analytic_comparator_uses_the_same_capsule_geometry_and_fails_closed_inside() -> None:
    model, actuator, safety, state = _problem()
    hover = jnp.array([-float(model.mass * model.gravity_vec[2]), 0.0, 0.0, 0.0])
    far_capsule = CapsuleObstacleSet(
        segment_start=jnp.array([[1.8, 0.0, 0.7]]),
        segment_end=jnp.array([[1.8, 0.0, 1.3]]),
        radii=jnp.array([0.1]),
        mask=jnp.array([True]),
    )
    without = version_a_analytic_filter(
        state, hover, jnp.ones(4), model, actuator, safety, VersionABarrierConfig()
    )
    with_far = version_a_analytic_filter(
        state,
        hover,
        jnp.ones(4),
        model,
        actuator,
        safety,
        VersionABarrierConfig(),
        capsules=far_capsule,
    )

    assert (
        with_far.analytic_barriers.matrix.shape[0] == without.analytic_barriers.matrix.shape[0] + 1
    )
    assert bool(with_far.qp_accepted)
    assert bool(with_far.applied_postcheck.passed)

    colliding_capsule = far_capsule._replace(
        segment_start=jnp.array([[0.0, 0.0, 0.7]]), segment_end=jnp.array([[0.0, 0.0, 1.3]])
    )
    colliding = version_a_analytic_filter(
        state,
        hover,
        jnp.ones(4),
        model,
        actuator,
        safety,
        VersionABarrierConfig(),
        capsules=colliding_capsule,
    )
    assert not bool(colliding.input_valid)
    assert not bool(colliding.qp_accepted)
    assert bool(colliding.degraded)
