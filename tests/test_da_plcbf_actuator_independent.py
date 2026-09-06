from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from crazyflow.drones import available_drones, load_params
from crazyflow.dynamics.core import load_params as load_dynamics_params
from crazyflow.dynamics.first_principles.dynamics import dynamics as native_dynamics
from crazyflow.safety.da_plcbf.actuator_dynamics import (
    ActuatorModel,
    effort_lag_step,
    fit_native_time_constant,
)
from crazyflow.safety.da_plcbf.actuator_independent import (
    ActuatorEvent,
    NativeRotorParameters,
    NativeRotorPlant,
    NumpyEffortPlant,
    effort_to_rpm,
    independent_effort_step,
    native_rotor_step,
    native_rotor_wrench,
    rpm_to_effort,
)
from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel


def _model(drone: str = "cf21B_500", *, drag: bool = True) -> ActuatorModel:
    p = load_params(drone)
    body = VersionAModel(
        np.asarray(p["mass"]),
        np.asarray(p["gravity_vec"]),
        np.asarray(p["J"]),
        np.linalg.inv(p["J"]),
        np.asarray(p["drag_matrix"]) if drag else np.zeros((3, 3)),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
    )
    mapping = np.vstack(
        (
            np.ones(4),
            np.array([p["L"], p["L"], p["thrust2torque"]])[:, None]
            * np.asarray(p["mixing_matrix"]),
        )
    )
    return ActuatorModel(
        body,
        np.ones(4),
        np.full(4, fit_native_time_constant(drone)["nominal_tau"]),
        np.full(4, p["thrust_min"]),
        np.full(4, p["thrust_max"]),
        mapping,
    )


def _state(model: ActuatorModel) -> np.ndarray:
    y = np.zeros(17)
    y[2], y[6] = 1.0, 1.0
    y[13:] = float(model.body.mass) * -model.body.gravity_vec[2] / 4
    return y


def test_p1_coupled_body_solution_converges_to_analytic_vertical_response() -> None:
    model = _model(drag=False)._replace(time_constants=np.full(4, 0.073))
    state, command, duration = _state(model), np.full(4, 0.14), 0.18
    initial = state[13]
    mass, gravity, tau, target = float(model.body.mass), model.body.gravity_vec[2], 0.073, 0.14
    decay = -np.expm1(-duration / tau)
    expected_velocity = gravity * duration + 4 / mass * (
        target * duration + (initial - target) * tau * decay
    )
    expected_position = (
        state[2]
        + gravity * duration**2 / 2
        + 4
        / mass
        * (target * duration**2 / 2 + (initial - target) * tau * (duration - tau * decay))
    )
    solutions = [
        independent_effort_step(state, command, model, duration, substeps=n) for n in (4, 8, 16, 32)
    ]
    errors = [np.linalg.norm(y[[2, 9]] - [expected_position, expected_velocity]) for y in solutions]
    assert all(a / b > 12 for a, b in zip(errors[:-1], errors[1:], strict=True))
    assert errors[-1] < 2e-8
    np.testing.assert_allclose(
        solutions[-1][13:], initial + decay * (command - initial), atol=1e-15
    )
    # Applying the endpoint force over the whole hold would produce a much larger impulse.
    end_force_velocity = duration * (gravity + 4 * solutions[-1][13] / mass)
    assert abs(end_force_velocity - expected_velocity) > 0.1


def test_p1_matches_predictor_with_independent_body_rotation_and_allocation_path() -> None:
    model = _model()._replace(
        effectiveness=np.array([0.7, 1, 0.85, 1]), time_constants=np.array([0.05, 0.08, 0.13, 0.2])
    )
    body = model.body._replace(
        wind_velocity=np.array([0.2, -0.1, 0.05]),
        external_force=np.array([0.001, 0.002, -0.003]),
        external_torque=np.array([2e-5, -3e-5, 1e-5]),
    )
    model = model._replace(body=body)
    state = _state(model)
    state[3:7] = Rotation.from_euler("xyz", [0.4, -0.2, 0.7]).as_quat()
    state[7:13] = [0.2, -0.3, 0.1, 0.3, -0.4, 0.2]
    command = np.array([0.08, 0.12, 0.15, 0.11])
    independent = independent_effort_step(state, command, model, 0.04, substeps=8)
    with jax.enable_x64():
        matched = effort_lag_step(jnp.asarray(state), jnp.asarray(command), model, 0.04, substeps=8)
    np.testing.assert_allclose(independent, matched, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("motor", range(4))
def test_p1_motor_order_and_torque_signs(motor: int) -> None:
    model = _model(drag=False)
    state = _state(model)
    state[13 + motor] += 0.01
    following = independent_effort_step(state, state[13:], model, 1e-5)
    np.testing.assert_array_equal(
        np.sign(following[10:13]), np.sign(model.force_to_wrench[1:, motor])
    )
    expected = np.linalg.solve(model.body.inertia, model.force_to_wrench[1:, motor] * 0.01)
    np.testing.assert_allclose(following[10:13] / 1e-5, expected, rtol=2e-7)


def test_p1_xyzw_rotates_body_thrust_into_world_and_does_not_clamp_floor() -> None:
    model = _model(drag=False)
    state = _state(model)
    state[2] = -0.2
    state[3:7] = Rotation.from_euler("y", np.pi / 2).as_quat()
    following = independent_effort_step(state, state[13:], model, 0.002)
    np.testing.assert_allclose(following[7:10], [9.81 * 0.002, 0, -9.81 * 0.002], atol=1e-13)
    assert following[2] < -0.2
    np.testing.assert_allclose(np.linalg.norm(following[3:7]), 1.0, atol=1e-15)


@pytest.mark.parametrize("drone", available_drones)
def test_native_conversion_roundtrip_full_range_and_nominal_pole(drone: str) -> None:
    p = load_params(drone)
    efforts = np.linspace(p["thrust_min"], p["thrust_max"], 1001)
    conversion = effort_to_rpm(efforts, p["rpm2thrust"], p["thrust_min"], p["thrust_max"])
    assert conversion.residual < 1e-15
    assert np.all(np.diff(conversion.rpm) > 0)
    np.testing.assert_allclose(conversion.reconstructed_effort, efforts, atol=1e-15)
    assert np.all(conversion.rpm >= conversion.rpm_lower)
    assert np.all(conversion.rpm <= conversion.rpm_upper)
    assert NativeRotorParameters.from_drone(drone).nominal_time_constant == pytest.approx(
        fit_native_time_constant(drone)["nominal_tau"], abs=1e-15
    )


def test_conversion_retains_offset_and_selects_physical_quadratic_branch() -> None:
    coefficients = np.array([0.01, -2e-6, 4e-10])
    effort = np.array([0.02, 0.08, 0.13, 0.2])
    result = effort_to_rpm(effort, coefficients, 0.02, 0.2)
    assert np.all(result.rpm > -coefficients[1] / (2 * coefficients[2]))
    np.testing.assert_allclose(rpm_to_effort(result.rpm, coefficients), effort, atol=1e-15)
    wrong_without_offset = effort_to_rpm(effort, [0.0, *coefficients[1:]], 0.02, 0.2)
    assert np.max(abs(wrong_without_offset.rpm - result.rpm)) > 100
    linear = effort_to_rpm(effort, [0.01, 2e-5, 0.0], 0.02, 0.2)
    np.testing.assert_allclose(linear.rpm, (effort - 0.01) / 2e-5)


@pytest.mark.parametrize(
    "effort,coefficients,lower,upper",
    [
        (0.21, [0, -2e-6, 4e-10], 0.02, 0.2),
        (0.019, [0, -2e-6, 4e-10], 0.02, 0.2),
        (0.05, [0.1, 2e-6, 4e-10], 0.02, 0.2),
        (0.05, [0.1, 0.0, 4e-10], 0.02, 0.2),
        (0.05, [0, 2e-6, -4e-10], 0.02, 0.2),
        (0.05, [0, 0, 0], 0.02, 0.2),
        (float("nan"), [0, -2e-6, 4e-10], 0.02, 0.2),
    ],
)
def test_conversion_rejects_invalid_bounds_and_curves(
    effort: float, coefficients: list[float], lower: float, upper: float
) -> None:
    with pytest.raises(ValueError):
        effort_to_rpm(effort, coefficients, lower, upper)


def test_p2_joint_integration_matches_native_continuous_equations() -> None:
    model, parameters = _model(), NativeRotorParameters.from_drone()
    native = NativeRotorPlant(_state(model), model, parameters).native_state
    native[3:7] = Rotation.from_euler("xyz", [0.2, -0.15, 0.3]).as_quat()
    native[7:13] = [0.3, -0.2, 0.1, 0.4, -0.3, 0.2]
    command = np.array([0.085, 0.15, 0.095, 0.13])
    rpm_command = effort_to_rpm(
        command, parameters.thrust_coefficients, model.command_lower, model.command_upper
    ).rpm
    native_parameters = load_dynamics_params(native_dynamics, parameters.drone)

    def rhs(_: float, y: np.ndarray) -> np.ndarray:
        derivative = native_dynamics(
            y[:3], y[3:7], y[7:10], y[10:13], rpm_command, rotor_vel=y[13:], **native_parameters
        )
        return np.concatenate(derivative)

    reference = solve_ivp(rhs, (0.0, 0.04), native, method="DOP853", rtol=2e-12, atol=1e-13).y[
        :, -1
    ]
    reference[3:7] /= np.linalg.norm(reference[3:7])
    coarse = native_rotor_step(native, command, model, parameters, 0.04, substeps=4)
    medium = native_rotor_step(native, command, model, parameters, 0.04, substeps=8)
    fine = native_rotor_step(native, command, model, parameters, 0.04, substeps=64)
    assert np.linalg.norm(coarse - reference) / np.linalg.norm(medium - reference) > 10
    np.testing.assert_allclose(fine, reference, atol=3e-7, rtol=1e-10)


def test_p2_preserves_asymmetric_rates_and_scales_effectiveness_exactly_once() -> None:
    model, parameters = _model(), NativeRotorParameters.from_drone()
    native = NativeRotorPlant(_state(model), model, parameters).native_state
    native[13:] *= [0.9, 1.0, 1.1, 1.05]
    native[10:13] = [0.5, -0.2, 0.1]
    rpm_command = native[13:] + [1000, -1000, 1000, -1000]
    nominal, rpm_dot, nominal_forces = native_rotor_wrench(native, rpm_command, model, parameters)
    eta = np.array([0.7, 0.85, 1.0, 0.9])
    changed = model._replace(effectiveness=eta)
    degraded, changed_dot, degraded_forces = native_rotor_wrench(
        native, rpm_command, changed, parameters
    )
    np.testing.assert_allclose(changed_dot, rpm_dot, atol=0)
    np.testing.assert_allclose(degraded_forces, eta * nominal_forces, atol=1e-16)
    without_inertia = replace(parameters, propeller_inertia=0)
    aero_nominal, _, _ = native_rotor_wrench(native, rpm_command, model, without_inertia)
    aero_degraded, _, _ = native_rotor_wrench(native, rpm_command, changed, without_inertia)
    np.testing.assert_allclose(degraded - aero_degraded, nominal - aero_nominal, atol=1e-18)
    assert np.linalg.norm(nominal[1:] - aero_nominal[1:]) > 1e-6
    expected_yaw = parameters.mixing_matrix[2] @ (
        eta * rpm_to_effort(native[13:], parameters.torque_coefficients)
    )
    assert aero_degraded[3] == pytest.approx(expected_yaw, abs=1e-16)
    slowed = model._replace(time_constants=2 * model.time_constants)
    _, slow_dot, _ = native_rotor_wrench(native, rpm_command, slowed, parameters)
    np.testing.assert_allclose(slow_dot, rpm_dot / 2)
    up, _, down, _ = parameters.rotor_coefficients
    assert up != down


@pytest.mark.parametrize("plant_type", [NumpyEffortPlant, NativeRotorPlant])
def test_events_preserve_full_state_and_are_invisible_until_active(plant_type: type) -> None:
    model, initial = _model(), _state(_model())
    changed = model._replace(
        effectiveness=np.array([0.7, 1, 1, 0.85]), time_constants=2 * model.time_constants
    )
    event_time = 0.013
    plant = plant_type(initial, model, events=[ActuatorEvent(event_time, changed)], max_step=0.004)
    untouched = plant_type(initial, model, max_step=0.004)
    snapshot = plant.model_snapshot()
    np.testing.assert_array_equal(snapshot.effectiveness, np.ones(4))
    with pytest.raises(ValueError):
        snapshot.effectiveness[0] = 0.5
    command = initial[13:].copy()
    trace = plant.advance(command, event_time)
    untouched.advance(command, event_time)
    np.testing.assert_array_equal(plant.native_state, untouched.native_state)
    np.testing.assert_allclose(plant.observe(), initial, atol=5e-15)
    np.testing.assert_array_equal(plant.model_snapshot().effectiveness, changed.effectiveness)
    np.testing.assert_array_equal(snapshot.effectiveness, np.ones(4))
    np.testing.assert_allclose(
        trace.actual_forces[-1], changed.effectiveness * initial[13:], atol=1e-16
    )
    assert trace.times[-1] == event_time
    after = plant.advance(command, 0.004)
    assert after.states[-1, 9] < after.states[0, 9]
    assert after.states.shape[1] == after.native_states.shape[1] == 17


def test_native_observation_is_effort_but_rpm_state_persists_and_curve_offset_works() -> None:
    model = _model()
    parameters = replace(
        NativeRotorParameters.from_drone(), thrust_coefficients=np.array([0.01, -2e-6, 4e-10])
    )
    initial = _state(model)
    plant = NativeRotorPlant(initial, model, parameters)
    np.testing.assert_allclose(plant.observe(), initial, atol=3e-17)
    assert np.min(plant.native_state[13:]) > 1000
    trace = plant.advance(np.full(4, 0.14), 0.02)
    np.testing.assert_allclose(
        trace.states[:, 13:],
        rpm_to_effort(trace.native_states[:, 13:], parameters.thrust_coefficients),
        atol=1e-15,
    )
    assert trace.conversion_residual < 1e-15
    assert np.all(trace.states[-1, 13:] > initial[13:])


def test_invalid_held_command_and_invalid_event_schedule_raise() -> None:
    model, state = _model(), _state(_model())
    for plant_type in (NumpyEffortPlant, NativeRotorPlant):
        plant = plant_type(state, model)
        with pytest.raises(ValueError, match="bounds"):
            plant.advance(model.command_upper + 1e-4, 0.01)
        assert plant.time == 0
        with pytest.raises(ValueError, match="strictly increasing"):
            plant_type(
                state, model, events=[ActuatorEvent(0.02, model), ActuatorEvent(0.01, model)]
            )


@pytest.mark.parametrize("plant_type", [NumpyEffortPlant, NativeRotorPlant])
def test_mid_hold_event_preserves_transient_motor_state_and_splits_the_actual_plant(
    plant_type: type,
) -> None:
    model, state = _model(), _state(_model())
    changed = model._replace(
        time_constants=model.time_constants * [1.5, 2, 1, 3],
        effectiveness=np.array([0.85, 1, 0.7, 1]),
    )
    parameters = NativeRotorParameters.from_drone()
    plant = plant_type(state, model, events=[ActuatorEvent(0.013, changed)], max_step=0.1)
    native_initial = plant.native_state
    command = np.array([0.14, 0.09, 0.15, 0.08])
    if plant_type is NativeRotorPlant:
        boundary = native_rotor_step(native_initial, command, model, parameters, 0.013)
        expected = native_rotor_step(boundary, command, changed, parameters, 0.017)
    else:
        boundary = independent_effort_step(native_initial, command, model, 0.013)
        expected = independent_effort_step(boundary, command, changed, 0.017)
    trace = plant.advance(command, 0.03)
    np.testing.assert_allclose(trace.times, [0, 0.013, 0.03], atol=1e-16)
    np.testing.assert_array_equal(trace.native_states[1], boundary)
    np.testing.assert_allclose(trace.native_states[-1], expected, rtol=1e-14, atol=1e-13)
    assert np.linalg.norm(trace.states[1, 13:] - state[13:]) > 1e-3


def test_native_step_rejects_low_speed_nonphysical_force_branch() -> None:
    model = _model()
    parameters = NativeRotorParameters.from_drone()
    native = NativeRotorPlant(_state(model), model, parameters).native_state
    native[13:] = 100.0
    with pytest.raises(ValueError, match="physical branch"):
        native_rotor_step(native, _state(model)[13:], model, parameters, 0.001)
