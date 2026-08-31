from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.drones import load_params
from crazyflow.dynamics.first_principles import dynamics as first_principles_dynamics
from crazyflow.safety.da_plcbf.direct_wrench import (
    direct_wrench_dynamics,
    motor_forces_to_wrench,
    motor_thrust_feasibility_residual,
    motor_thrust_inequalities,
    quaternion_derivative_xyzw,
    quaternion_to_rotation_matrix,
    wrench_to_motor_forces,
)


def _cf21b() -> dict:
    params = load_params("cf21B_500")
    return {
        **params,
        "gravity_vec": np.asarray(params["gravity_vec"]),
        "J": np.asarray(params["J"]),
        "J_inv": np.linalg.inv(params["J"]),
        "rpm2thrust": np.asarray(params["rpm2thrust"]),
        "rpm2torque": np.asarray(params["rpm2torque"]),
        "mixing_matrix": np.asarray(params["mixing_matrix"]),
        "drag_matrix": np.asarray(params["drag_matrix"]),
        "rotor_dyn_coef": np.asarray(params["rotor_dyn_coef"]),
    }


def _allocation_params(params: dict) -> dict:
    return {
        "L": params["L"],
        "thrust2torque": params["thrust2torque"],
        "mixing_matrix": params["mixing_matrix"],
    }


@pytest.mark.unit
def test_rotation_matrix_matches_scipy_for_scaled_quaternions() -> None:
    generator = np.random.default_rng(4109)
    quaternions = Rotation.random(32, random_state=generator).as_quat()
    scales = generator.uniform(0.1, 8.0, size=(32, 1))
    scaled_quaternions = quaternions * scales

    actual = quaternion_to_rotation_matrix(scaled_quaternions)
    expected = Rotation.from_quat(scaled_quaternions).as_matrix()

    assert np.allclose(actual, expected, atol=2e-14)
    assert np.allclose(actual @ np.swapaxes(actual, -1, -2), np.eye(3), atol=2e-14)


@pytest.mark.unit
def test_quaternion_derivative_matches_body_rate_rotation_composition() -> None:
    quat = Rotation.from_euler("xyz", [0.31, -0.47, 1.02]).as_quat()
    body_ang_vel = np.array([0.4, -0.2, 0.8])
    step = 1e-6

    plus = (Rotation.from_quat(quat) * Rotation.from_rotvec(body_ang_vel * step)).as_quat()
    minus = (Rotation.from_quat(quat) * Rotation.from_rotvec(-body_ang_vel * step)).as_quat()
    # Quaternions have a double cover; align both finite-difference samples to the input branch.
    plus *= np.sign(np.dot(plus, quat))
    minus *= np.sign(np.dot(minus, quat))
    finite_difference = (plus - minus) / (2 * step)

    assert np.allclose(
        quaternion_derivative_xyzw(quat, body_ang_vel), finite_difference, atol=2e-10
    )


@pytest.mark.unit
def test_all_sixteen_cf21b_motor_box_vertices_round_trip_and_satisfy_polytope() -> None:
    params = _cf21b()
    low, high = params["thrust_min"], params["thrust_max"]
    bits = np.asarray(list(itertools.product((False, True), repeat=4)))
    motor_forces = np.where(bits, high, low)

    wrenches = motor_forces_to_wrench(motor_forces, **_allocation_params(params))
    recovered = wrench_to_motor_forces(wrenches, **_allocation_params(params))
    residuals = motor_thrust_feasibility_residual(
        wrenches, thrust_min=low, thrust_max=high, **_allocation_params(params)
    )
    constraints = motor_thrust_inequalities(
        thrust_min=low, thrust_max=high, **_allocation_params(params)
    )

    assert np.allclose(recovered, motor_forces, atol=2e-16)
    assert np.all(residuals <= 2e-16)
    assert np.all(constraints.matrix @ wrenches.T <= constraints.upper_bound[:, None] + 2e-16)


@pytest.mark.unit
def test_independent_wrench_component_box_contains_an_infeasible_corner() -> None:
    params = _cf21b()
    low, high = params["thrust_min"], params["thrust_max"]
    bits = np.asarray(list(itertools.product((False, True), repeat=4)))
    motor_vertices = np.where(bits, high, low)
    wrench_vertices = motor_forces_to_wrench(motor_vertices, **_allocation_params(params))
    component_min = np.min(wrench_vertices, axis=0)
    component_max = np.max(wrench_vertices, axis=0)

    # A component-wise wrench box would accept this corner, but no four motor forces in the
    # airborne box can realize maximum collective thrust and all maximum torques simultaneously.
    infeasible_box_corner = component_max
    implied_motor_forces = wrench_to_motor_forces(
        infeasible_box_corner, **_allocation_params(params)
    )
    residual = motor_thrust_feasibility_residual(
        infeasible_box_corner, thrust_min=low, thrust_max=high, **_allocation_params(params)
    )

    assert np.all(infeasible_box_corner >= component_min)
    assert np.all(infeasible_box_corner <= component_max)
    assert np.any((implied_motor_forces < low) | (implied_motor_forces > high))
    assert np.isclose(residual, (high - low) / 2, atol=2e-16)


@pytest.mark.unit
def test_direct_derivative_matches_first_principles_when_omitted_effects_are_removed() -> None:
    params = _cf21b()
    pos = np.array([0.2, -0.4, 1.1])
    quat = Rotation.from_euler("xyz", [0.22, -0.31, 0.73]).as_quat()
    vel = np.array([0.8, -0.35, 0.16])
    ang_vel = np.array([0.42, -0.17, 0.29])
    external_force = np.array([0.003, -0.002, 0.004])
    external_torque = np.array([2e-6, -3e-6, 1e-6])
    requested_motor_forces = np.array([0.055, 0.087, 0.119, 0.151])
    rotor_vel = motor_force2rotor_vel(requested_motor_forces, params["rpm2thrust"])
    actual_motor_forces = (
        params["rpm2thrust"][0]
        + params["rpm2thrust"][1] * rotor_vel
        + params["rpm2thrust"][2] * rotor_vel**2
    )
    actual_motor_torques = (
        params["rpm2torque"][0]
        + params["rpm2torque"][1] * rotor_vel
        + params["rpm2torque"][2] * rotor_vel**2
    )
    mixed_forces = params["mixing_matrix"] @ actual_motor_forces
    mixed_motor_torques = params["mixing_matrix"] @ actual_motor_torques
    exact_production_wrench = np.concatenate(
        (
            np.array([np.sum(actual_motor_forces)]),
            mixed_forces[:2] * params["L"],
            np.array([mixed_motor_torques[2]]),
        )
    )

    production = first_principles_dynamics(
        pos,
        quat,
        vel,
        ang_vel,
        rotor_vel,
        rotor_vel=rotor_vel,
        dist_f=external_force,
        dist_t=external_torque,
        mass=params["mass"],
        L=params["L"],
        # Version A intentionally omits propeller gyroscopic/inertial torque.
        prop_inertia=0.0,
        gravity_vec=params["gravity_vec"],
        J=params["J"],
        J_inv=params["J_inv"],
        rpm2thrust=params["rpm2thrust"],
        rpm2torque=params["rpm2torque"],
        mixing_matrix=params["mixing_matrix"],
        drag_matrix=params["drag_matrix"],
        rotor_dyn_coef=params["rotor_dyn_coef"],
    )
    direct = direct_wrench_dynamics(
        pos,
        quat,
        vel,
        ang_vel,
        exact_production_wrench,
        mass=params["mass"],
        gravity_vec=params["gravity_vec"],
        J=params["J"],
        J_inv=params["J_inv"],
        drag_matrix=params["drag_matrix"],
        external_force=external_force,
        external_torque=external_torque,
    )

    # Crazyflow integrates body angular velocity directly, rather than its legacy returned
    # quaternion derivative, so compare the production quantities used by the simulator.
    assert np.allclose(direct.pos_dot, production[0], atol=2e-14)
    assert np.allclose(direct.vel_dot, production[2], atol=2e-13)
    assert np.allclose(direct.ang_vel_dot, production[3], atol=2e-11)
    assert np.allclose(production[4], 0.0, atol=2e-13)
