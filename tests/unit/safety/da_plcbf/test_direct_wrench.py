from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.direct_wrench import (
    affine_feasibility_residual,
    apply_control_affine,
    control_affine_identity_residual,
    control_affine_terms,
    direct_wrench_dynamics,
    flatten_derivative,
    motor_allocation_matrix,
    motor_forces_to_wrench,
    motor_thrust_feasibility_residual,
    motor_thrust_inequalities,
    quaternion_derivative_xyzw,
    wrench_to_motor_forces,
)


@pytest.fixture
def cf21b() -> dict[str, Any]:
    params = load_params("cf21B_500")
    return {
        "mass": params["mass"],
        "gravity_vec": jnp.asarray(params["gravity_vec"]),
        "J": jnp.asarray(params["J"]),
        "J_inv": jnp.linalg.inv(jnp.asarray(params["J"])),
        "drag_matrix": jnp.asarray(params["drag_matrix"]),
        "L": params["L"],
        "thrust2torque": params["thrust2torque"],
        "mixing_matrix": jnp.asarray(params["mixing_matrix"]),
        "thrust_min": params["thrust_min"],
        "thrust_max": params["thrust_max"],
    }


def _state(batch_shape: tuple[int, ...] = ()) -> tuple[jax.Array, ...]:
    pos = jnp.zeros((*batch_shape, 3))
    quat = jnp.broadcast_to(jnp.array([0.0, 0.0, 0.0, 1.0]), (*batch_shape, 4))
    vel = jnp.zeros((*batch_shape, 3))
    ang_vel = jnp.zeros((*batch_shape, 3))
    return pos, quat, vel, ang_vel


def _dynamics_params(params: dict[str, Any]) -> dict[str, Any]:
    return {name: params[name] for name in ("mass", "gravity_vec", "J", "J_inv", "drag_matrix")}


def _allocation_params(params: dict[str, Any]) -> dict[str, Any]:
    return {name: params[name] for name in ("L", "thrust2torque", "mixing_matrix")}


@pytest.mark.unit
def test_cf21b_hover_is_stationary_and_inside_airborne_polytope(cf21b: dict[str, Any]) -> None:
    pos, quat, vel, ang_vel = _state()
    hover_wrench = jnp.array([cf21b["mass"] * 9.81, 0.0, 0.0, 0.0])

    derivative = direct_wrench_dynamics(
        pos, quat, vel, ang_vel, hover_wrench, **_dynamics_params(cf21b)
    )
    assert np.allclose(flatten_derivative(derivative), 0.0, atol=2e-6)

    motor_forces = wrench_to_motor_forces(hover_wrench, **_allocation_params(cf21b))
    assert np.allclose(motor_forces, hover_wrench[0] / 4, atol=1e-7)
    assert np.all(motor_forces > cf21b["thrust_min"])
    assert np.all(motor_forces < cf21b["thrust_max"])
    residual = motor_thrust_feasibility_residual(
        hover_wrench,
        thrust_min=cf21b["thrust_min"],
        thrust_max=cf21b["thrust_max"],
        **_allocation_params(cf21b),
    )
    assert residual < 0


@pytest.mark.unit
def test_quaternion_derivative_uses_scalar_last_body_rate_convention() -> None:
    quat = jnp.array([0.0, 0.0, 0.0, 1.0])
    body_rate = jnp.array([0.4, -0.2, 0.8])
    derivative = quaternion_derivative_xyzw(quat, body_rate)
    assert np.allclose(derivative, jnp.array([0.2, -0.1, 0.4, 0.0]))
    assert np.isclose(jnp.dot(quat, derivative), 0.0)


@pytest.mark.unit
def test_thrust_and_external_disturbances_use_documented_frames(cf21b: dict[str, Any]) -> None:
    pos, _, vel, ang_vel = _state()
    half_sqrt_two = np.sqrt(0.5)
    # +90 degrees about body/world y: body +z points along world +x.
    quat = jnp.array([0.0, half_sqrt_two, 0.0, half_sqrt_two])
    thrust = 0.2
    external_force = jnp.array([0.0, 0.1, 0.0])
    # Body +z also points world +x, so this world torque becomes body +z.
    external_torque = jnp.array([4.9e-5, 0.0, 0.0])
    wrench = jnp.array([thrust, 2.5e-5, 0.0, 0.0])

    derivative = direct_wrench_dynamics(
        pos,
        quat,
        vel,
        ang_vel,
        wrench,
        mass=0.1,
        gravity_vec=jnp.zeros(3),
        J=cf21b["J"],
        J_inv=cf21b["J_inv"],
        drag_matrix=jnp.zeros((3, 3)),
        external_force=external_force,
        external_torque=external_torque,
    )

    assert np.allclose(derivative.vel_dot, jnp.array([2.0, 1.0, 0.0]), atol=2e-6)
    assert np.allclose(derivative.ang_vel_dot, jnp.array([1.0, 0.0, 1.0]), atol=2e-5)


@pytest.mark.unit
def test_body_drag_uses_world_relative_air_velocity() -> None:
    pos, _, _, ang_vel = _state()
    half_sqrt_two = np.sqrt(0.5)
    # +90 degrees yaw: world +x is body -y.
    quat = jnp.array([0.0, 0.0, half_sqrt_two, half_sqrt_two])
    vel = jnp.array([1.0, 0.0, 0.0])
    wind = jnp.array([0.25, 0.0, 0.0])
    drag_matrix = jnp.diag(jnp.array([-1.0, -2.0, -3.0]))

    derivative = direct_wrench_dynamics(
        pos,
        quat,
        vel,
        ang_vel,
        jnp.zeros(4),
        mass=2.0,
        gravity_vec=jnp.zeros(3),
        J=jnp.eye(3),
        drag_matrix=drag_matrix,
        wind_velocity=wind,
    )
    assert np.allclose(derivative.vel_dot, jnp.array([-0.75, 0.0, 0.0]), atol=1e-6)

    no_relative_air = direct_wrench_dynamics(
        pos,
        quat,
        wind,
        ang_vel,
        jnp.zeros(4),
        mass=2.0,
        gravity_vec=jnp.zeros(3),
        J=jnp.eye(3),
        drag_matrix=drag_matrix,
        wind_velocity=wind,
    )
    assert np.allclose(no_relative_air.vel_dot, 0.0)


@pytest.mark.unit
def test_control_affine_identity_unbatched_and_batched(cf21b: dict[str, Any]) -> None:
    key = jax.random.key(12)
    keys = jax.random.split(key, 5)
    pos = jax.random.normal(keys[0], (7, 3))
    quat = jax.random.normal(keys[1], (7, 4))
    vel = jax.random.normal(keys[2], (7, 3))
    ang_vel = 0.3 * jax.random.normal(keys[3], (7, 3))
    wrench = jnp.concatenate(
        (
            0.3 + 0.1 * jax.random.uniform(keys[4], (7, 1)),
            1e-4 * jax.random.normal(keys[0], (7, 3)),
        ),
        axis=-1,
    )
    optional = {
        "wind_velocity": jnp.array([0.2, -0.1, 0.05]),
        "external_force": jnp.array([1e-3, -2e-3, 3e-3]),
        "external_torque": jnp.array([2e-6, 1e-6, -3e-6]),
    }

    residual = control_affine_identity_residual(
        pos, quat, vel, ang_vel, wrench, **_dynamics_params(cf21b), **optional
    )
    assert residual.shape == (7, 13)
    assert np.allclose(residual, 0.0, atol=2e-5)

    terms = control_affine_terms(
        pos[0], quat[0], vel[0], ang_vel[0], **_dynamics_params(cf21b), **optional
    )
    direct = direct_wrench_dynamics(
        pos[0], quat[0], vel[0], ang_vel[0], wrench[0], **_dynamics_params(cf21b), **optional
    )
    assert terms.drift.shape == (13,)
    assert terms.input_matrix.shape == (13, 4)
    assert np.allclose(
        apply_control_affine(terms, wrench[0]), flatten_derivative(direct), atol=2e-5
    )


@pytest.mark.unit
def test_batched_dynamics_matches_individual_evaluation(cf21b: dict[str, Any]) -> None:
    pos, quat, vel, ang_vel = _state((3, 2))
    vel = vel.at[..., 0].set(jnp.arange(6).reshape(3, 2) * 0.1)
    ang_vel = ang_vel.at[..., 2].set(0.2)
    wrench = jnp.broadcast_to(jnp.array([0.4, 1e-5, -2e-5, 3e-6]), (3, 2, 4))
    batched = flatten_derivative(
        direct_wrench_dynamics(pos, quat, vel, ang_vel, wrench, **_dynamics_params(cf21b))
    )

    for first in range(3):
        for second in range(2):
            individual = flatten_derivative(
                direct_wrench_dynamics(
                    pos[first, second],
                    quat[first, second],
                    vel[first, second],
                    ang_vel[first, second],
                    wrench[first, second],
                    **_dynamics_params(cf21b),
                )
            )
            assert np.allclose(batched[first, second], individual, atol=1e-6)


@pytest.mark.unit
def test_cf21b_allocation_round_trip_and_current_mixing_formula(cf21b: dict[str, Any]) -> None:
    motor_forces = jnp.array([[0.04, 0.06, 0.08, 0.1], [0.15, 0.11, 0.07, 0.03]], dtype=jnp.float32)
    wrench = motor_forces_to_wrench(motor_forces, **_allocation_params(cf21b))
    recovered = wrench_to_motor_forces(wrench, **_allocation_params(cf21b))
    assert np.allclose(recovered, motor_forces, atol=2e-7)

    single_wrench = jnp.array([0.4, 1e-3, -5e-4, 2e-4])
    scale = jnp.array([1 / cf21b["L"], 1 / cf21b["L"], 1 / cf21b["thrust2torque"]])
    expected = (single_wrench[:1] + (single_wrench[1:] * scale) @ cf21b["mixing_matrix"]) / 4
    actual = wrench_to_motor_forces(single_wrench, **_allocation_params(cf21b))
    assert np.allclose(actual, expected, atol=1e-7)

    allocation = motor_allocation_matrix(
        cf21b["mixing_matrix"], L=cf21b["L"], thrust2torque=cf21b["thrust2torque"]
    )
    assert np.allclose(allocation @ single_wrench, actual)


@pytest.mark.unit
def test_large_fp32_allocation_batch_does_not_use_low_precision_matmul(
    cf21b: dict[str, Any],
) -> None:
    values = jnp.linspace(
        float(cf21b["thrust_min"]), float(cf21b["thrust_max"]), 65_536 * 4
    ).reshape(65_536, 4)
    motor_forces = values[jnp.arange(values.shape[0]) * 8191 % values.shape[0]]
    wrench = motor_forces_to_wrench(motor_forces, **_allocation_params(cf21b))
    recovered = wrench_to_motor_forces(wrench, **_allocation_params(cf21b))
    error = jnp.max(jnp.abs(recovered - motor_forces))

    assert float(error) <= 2e-7


@pytest.mark.unit
def test_allocation_is_unclipped(cf21b: dict[str, Any]) -> None:
    wrench = jnp.array([4 * (cf21b["thrust_max"] + 0.05), 0.0, 0.0, 0.0])
    motor_forces = wrench_to_motor_forces(wrench, **_allocation_params(cf21b))
    assert np.all(motor_forces > cf21b["thrust_max"])
    assert np.allclose(motor_forces, cf21b["thrust_max"] + 0.05)


@pytest.mark.unit
def test_affine_motor_inequalities_and_residual_match_direct_bounds(cf21b: dict[str, Any]) -> None:
    constraints = motor_thrust_inequalities(
        thrust_min=cf21b["thrust_min"], thrust_max=cf21b["thrust_max"], **_allocation_params(cf21b)
    )
    assert constraints.matrix.shape == (8, 4)
    assert constraints.upper_bound.shape == (8,)

    force_sets = jnp.array(
        [
            [0.05, 0.06, 0.07, 0.08],
            [cf21b["thrust_max"], 0.1, 0.1, 0.1],
            [cf21b["thrust_max"] + 0.03, 0.1, 0.1, 0.1],
            [cf21b["thrust_min"] - 0.02, 0.1, 0.1, 0.1],
        ]
    )
    wrenches = motor_forces_to_wrench(force_sets, **_allocation_params(cf21b))
    residual = affine_feasibility_residual(wrenches, constraints)
    expected = jnp.maximum(
        jnp.max(force_sets - cf21b["thrust_max"], axis=-1),
        jnp.max(cf21b["thrust_min"] - force_sets, axis=-1),
    )
    assert np.allclose(residual, expected, atol=2e-7)
    assert residual[0] < 0
    assert np.isclose(residual[1], 0.0, atol=2e-7)
    assert residual[2] > 0
    assert residual[3] > 0


@pytest.mark.unit
def test_numpy_array_api_path() -> None:
    pos = np.zeros(3)
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    vel = np.array([0.1, -0.2, 0.3])
    ang_vel = np.array([0.2, 0.1, -0.1])
    derivative = direct_wrench_dynamics(
        pos,
        quat,
        vel,
        ang_vel,
        np.array([1.0, 0.1, 0.2, 0.3]),
        mass=1.0,
        gravity_vec=np.array([0.0, 0.0, -9.81]),
        J=np.eye(3),
        drag_matrix=-0.1 * np.eye(3),
    )
    assert all(isinstance(component, np.ndarray) for component in derivative)
    assert all(np.all(np.isfinite(component)) for component in derivative)


@pytest.mark.unit
def test_jit_and_gradients_are_finite(cf21b: dict[str, Any]) -> None:
    pos = jnp.array([0.2, -0.1, 0.7])
    quat = jnp.array([0.1, -0.2, 0.3, 0.9])
    vel = jnp.array([0.4, -0.3, 0.2])
    ang_vel = jnp.array([0.2, -0.1, 0.3])

    def objective(wrench: jax.Array) -> jax.Array:
        derivative = direct_wrench_dynamics(
            pos, quat, vel, ang_vel, wrench, **_dynamics_params(cf21b)
        )
        vector = flatten_derivative(derivative)
        motors = wrench_to_motor_forces(wrench, **_allocation_params(cf21b))
        return jnp.sum(vector**2) + jnp.sum(motors**2)

    def state_objective(state: jax.Array, wrench: jax.Array) -> jax.Array:
        derivative = direct_wrench_dynamics(
            state[:3], state[3:7], state[7:10], state[10:13], wrench, **_dynamics_params(cf21b)
        )
        return jnp.sum(flatten_derivative(derivative) ** 2)

    wrench = jnp.array([0.4, 2e-5, -3e-5, 4e-6])
    value, gradient = jax.jit(jax.value_and_grad(objective))(wrench)
    state_gradient = jax.jit(jax.grad(state_objective))(
        jnp.concatenate((pos, quat, vel, ang_vel)), wrench
    )
    allocation_jacobian = jax.jacrev(
        lambda value: wrench_to_motor_forces(value, **_allocation_params(cf21b))
    )(wrench)
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(state_gradient))
    assert np.all(np.isfinite(allocation_jacobian))
