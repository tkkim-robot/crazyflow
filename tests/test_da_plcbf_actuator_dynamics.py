from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import lsq_linear

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.actuator_dynamics import (
    ActuatorModel,
    actual_motor_forces,
    actuator_state_valid,
    algebraic_control_affine_terms,
    algebraic_dynamics,
    algebraic_step,
    allocate_algebraic_wrench,
    allocate_wrench,
    applied_wrench,
    augmented_control_affine_terms,
    augmented_dynamics,
    effort_lag_step,
    exact_effort_update,
    fit_native_time_constant,
    hover_authority,
    make_actuator_model,
    normalized_command_metric,
    validate_actuator_model,
)
from crazyflow.safety.da_plcbf.direct_wrench import direct_wrench_dynamics, flatten_derivative
from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel


@pytest.fixture
def model() -> ActuatorModel:
    params = load_params("cf21B_500")
    body = VersionAModel(
        jnp.asarray(params["mass"]),
        jnp.asarray(params["gravity_vec"]),
        jnp.asarray(params["J"]),
        jnp.asarray(np.linalg.inv(params["J"])),
        jnp.asarray(params["drag_matrix"]),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros(3),
    )
    return make_actuator_model(
        body,
        L=params["L"],
        thrust2torque=params["thrust2torque"],
        mixing_matrix=params["mixing_matrix"],
        thrust_min=params["thrust_min"],
        thrust_max=params["thrust_max"],
        time_constants=fit_native_time_constant()["nominal_tau"],
    )


def initial(model: ActuatorModel) -> jax.Array:
    state = jnp.zeros(17).at[6].set(1).at[2].set(1)
    return state.at[13:].set(-model.body.mass * model.body.gravity_vec[2] / 4)


@pytest.mark.unit
def test_algebraic_identity_and_single_motor_torque_signs(model: ActuatorModel) -> None:
    state = initial(model)[:13]
    command = jnp.array([0.08, 0.09, 0.11, 0.12])
    b = model.body
    wrench = applied_wrench(command, model)
    expected = flatten_derivative(
        direct_wrench_dynamics(
            state[:3],
            state[3:7],
            state[7:10],
            state[10:13],
            wrench,
            mass=b.mass,
            gravity_vec=b.gravity_vec,
            J=b.inertia,
            J_inv=b.inertia_inv,
            drag_matrix=b.drag_matrix,
        )
    )
    np.testing.assert_allclose(algebraic_dynamics(state, command, model), expected)
    signs = np.array([[-1, -1, 1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1]])
    for motor in range(4):
        contribution = applied_wrench(jnp.eye(4)[motor], model)
        np.testing.assert_array_equal(np.sign(contribution[1:]), signs[:, motor])
    terms = algebraic_control_affine_terms(state, model)
    np.testing.assert_allclose(terms.drift + terms.input_matrix @ command, expected, atol=3e-5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "eta", [(1, 1, 1, 1), (0.85, 0.85, 0.85, 0.85), (0.7, 1, 1, 1), (0.7, 1, 0.7, 1)]
)
def test_effectiveness_once_static_reachable_wrench(model: ActuatorModel, eta: tuple) -> None:
    changed = model._replace(effectiveness=jnp.asarray(eta))
    desired_effort = jnp.array([0.09, 0.11, 0.1, 0.12])
    desired_wrench = applied_wrench(desired_effort, changed)
    result = allocate_algebraic_wrench(desired_wrench, changed)
    np.testing.assert_allclose(result.command, desired_effort, atol=3e-8)
    np.testing.assert_allclose(result.achieved_wrench, desired_wrench, atol=3e-8)
    np.testing.assert_allclose(
        actual_motor_forces(result.command, changed), desired_effort * jnp.asarray(eta), atol=3e-8
    )
    assert result.valid and not np.any(result.saturation)


@pytest.mark.unit
def test_f2_exact_reachable_endpoint_with_asymmetry(model: ActuatorModel) -> None:
    model = model._replace(
        effectiveness=jnp.array([0.7, 0.85, 1, 0.85]),
        time_constants=jnp.array([0.08, 0.12, 0.06, 0.16]),
    )
    state = jnp.array([0.10, 0.12, 0.09, 0.11])
    reachable = jnp.array([0.16, 0.07, 0.14, 0.10])
    endpoint = exact_effort_update(state, reachable, model.time_constants, 0.04)
    desired = applied_wrench(endpoint, model)
    result = allocate_wrench(desired, state, model, 0.04)
    np.testing.assert_allclose(result.command, reachable, atol=8e-8)
    np.testing.assert_allclose(result.achieved_wrench, desired, atol=5e-8)
    assert not np.any(result.saturation)
    assert float(result.normalized_residual) < 1e-7


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["F1", "F2"])
def test_bounded_allocator_matches_independent_coupled_lsq(model: ActuatorModel, mode: str) -> None:
    model = model._replace(effectiveness=jnp.array([0.70, 0.85, 1, 0.70]))
    rng = np.random.default_rng(903)
    scales = np.abs(np.asarray(model.force_to_wrench)[:, 0])
    for _ in range(12):
        s = rng.uniform(0.06, 0.17, 4)
        w = rng.uniform([0.2, -0.01, -0.01, -0.003], [0.95, 0.01, 0.01, 0.003])
        gain = -np.expm1(-0.04 / np.asarray(model.time_constants)) if mode == "F2" else np.ones(4)
        matrix = np.asarray(model.force_to_wrench) * np.asarray(model.effectiveness) * gain
        offset = np.asarray(model.force_to_wrench) @ (
            np.asarray(model.effectiveness) * (1 - gain) * s
        )
        oracle = lsq_linear(
            matrix / scales[:, None],
            (w - offset) / scales,
            bounds=(model.command_lower, model.command_upper),
            tol=1e-13,
        )
        actual = allocate_wrench(jnp.asarray(w), jnp.asarray(s), model, 0.04, mode=mode)
        np.testing.assert_allclose(actual.command, oracle.x, atol=2e-7)
        assert actual.valid
        assert np.all(actual.command >= model.command_lower)
        assert np.all(actual.command <= model.command_upper)


@pytest.mark.unit
def test_f0_f1_endpoint_diagnostics_use_actual_plant(model: ActuatorModel) -> None:
    model = model._replace(effectiveness=jnp.full(4, 0.7))
    state = initial(model)[13:]
    wrench = jnp.array([0.4, 0, 0, 0])
    for mode in ("F0", "F1", "F2"):
        result = allocate_wrench(wrench, state, model, 0.04, mode=mode)
        endpoint = exact_effort_update(state, result.command, model.time_constants, 0.04)
        np.testing.assert_allclose(result.achieved_wrench, applied_wrench(endpoint, model))
    assert float(allocate_wrench(wrench, state, model, 0.04, mode="F0").normalized_residual) > 0.05


@pytest.mark.unit
@pytest.mark.parametrize("tau", [1e-8, 1e-4, 0.06, 1e4, 1e12])
def test_exact_response_semigroup_and_extreme_positive_lag(tau: float) -> None:
    s = jnp.array([0.04, 0.07, 0.1, 0.16])
    u = jnp.array([0.16, 0.05, 0.13, 0.08])
    lag = jnp.full(4, tau)
    endpoint = exact_effort_update(s, u, lag, 0.04)
    expected = np.asarray(s, dtype=float) + (-np.expm1(-0.04 / tau)) * np.asarray(u - s)
    np.testing.assert_allclose(endpoint, expected, atol=2e-8)
    twice = exact_effort_update(exact_effort_update(s, u, lag, 0.017), u, lag, 0.023)
    np.testing.assert_allclose(twice, endpoint, atol=2e-8)
    assert np.all(np.isfinite(jax.jacrev(lambda x: exact_effort_update(s, x, lag, 0.04))(u)))


@pytest.mark.unit
def test_body_uses_evolving_force_and_converges(model: ActuatorModel) -> None:
    model = model._replace(body=model.body._replace(drag_matrix=jnp.zeros((3, 3))))
    state = initial(model)
    command = jnp.full(4, 0.16)
    dt = 0.1
    tau = float(model.time_constants[0])
    mass = float(model.body.mass)
    s0 = float(state[13])
    u = float(command[0])
    # Exact vertical solution; symmetric motors produce no rotation.
    expected_v = -9.81 * dt + 4 / mass * (u * dt + (s0 - u) * tau * (-np.expm1(-dt / tau)))
    expected_p = (
        1
        - 0.5 * 9.81 * dt**2
        + 4 / mass * (0.5 * u * dt**2 + (s0 - u) * (tau * dt - tau**2 * (-np.expm1(-dt / tau))))
    )
    results = [effort_lag_step(state, command, model, dt, substeps=n) for n in (1, 2, 4, 8)]
    errors = [np.linalg.norm(np.array([r[2] - expected_p, r[9] - expected_v])) for r in results]
    assert errors[1] < errors[0] / 8
    assert errors[2] < errors[1] / 5
    np.testing.assert_allclose(results[3][jnp.array([2, 9])], [expected_p, expected_v], atol=8e-7)
    endpoint_force_for_whole_hold_v = -9.81 * dt + 4 * float(results[-1][13]) / mass * dt
    assert abs(float(results[-1][9]) - endpoint_force_for_whole_hold_v) > 0.05


@pytest.mark.unit
def test_augmented_input_matrix_and_motor_state_ad(model: ActuatorModel) -> None:
    state = initial(model).at[7:10].set(jnp.array([0.4, -0.2, 0.1]))
    command = jnp.array([0.12, 0.13, 0.11, 0.14])
    terms = augmented_control_affine_terms(state, model)
    np.testing.assert_array_equal(terms.input_matrix[:13], 0)
    np.testing.assert_allclose(
        terms.drift + terms.input_matrix @ command,
        augmented_dynamics(state, command, model),
        atol=3e-7,
    )
    jac = jax.jacfwd(lambda u: augmented_dynamics(state, u, model))(command)
    np.testing.assert_allclose(jac, terms.input_matrix)

    def transition(y: jax.Array) -> jax.Array:
        return effort_lag_step(y, command, model, 0.04, substeps=2)

    forward, reverse = jax.jacfwd(transition)(state), jax.jacrev(transition)(state)
    np.testing.assert_allclose(forward, reverse, atol=2e-5, rtol=2e-5)
    epsilon = 1e-4
    for index in (3, 7, 13, 14, 15, 16):
        direction = jnp.eye(17)[index]
        fd = (transition(state + epsilon * direction) - transition(state - epsilon * direction)) / (
            2 * epsilon
        )
        np.testing.assert_allclose(reverse[:, index], fd, atol=7e-4, rtol=3e-3)


@pytest.mark.unit
def test_fault_model_replacement_preserves_state_and_only_eta_jumps_force(
    model: ActuatorModel,
) -> None:
    state = initial(model).at[13:].set(jnp.array([0.08, 0.11, 0.13, 0.09]))
    command = jnp.full(4, 0.12)
    saved = np.asarray(state).copy()
    lag_change = model._replace(time_constants=model.time_constants * 3)
    eta_change = model._replace(effectiveness=jnp.array([0.7, 1, 0.85, 1]))
    np.testing.assert_array_equal(state, saved)
    np.testing.assert_array_equal(
        applied_wrench(state[13:], lag_change), applied_wrench(state[13:], model)
    )
    assert not np.allclose(
        applied_wrench(state[13:], eta_change), applied_wrench(state[13:], model)
    )
    before = augmented_dynamics(state, command, model)
    after = augmented_dynamics(state, command, lag_change)
    np.testing.assert_array_equal(before[:13], after[:13])
    np.testing.assert_allclose(before[13:] / 3, after[13:], atol=1e-7)


@pytest.mark.unit
def test_positive_lag_approaches_separate_algebraic_model(model: ActuatorModel) -> None:
    state = initial(model)
    command = jnp.full(4, 0.14)
    algebraic = algebraic_step(state[:13], command, model, 0.04, substeps=40)
    errors = []
    for tau in (0.004, 0.001, 0.00025):
        changed = model._replace(time_constants=jnp.full(4, tau))
        integrated = effort_lag_step(state, command, changed, 0.04, substeps=160)
        errors.append(float(jnp.linalg.norm(integrated[:13] - algebraic)))
    assert errors[1] < 0.3 * errors[0]
    assert errors[2] < 0.3 * errors[1]


@pytest.mark.unit
def test_trim_coupled_authority_and_infeasible_scope(model: ActuatorModel) -> None:
    for eta in ([1, 1, 1, 1], [0.7, 1, 1, 1], [0.7, 1, 0.7, 1], [0.7, 0.7, 0.7, 0.7]):
        current = model._replace(effectiveness=jnp.asarray(eta))
        result = hover_authority(current)
        assert result["trim_feasible"]
        assert result["maneuver_classification"] == "witness_not_found"
        assert min(result["headroom_N"]) > 0
        assert np.min(result["torque_limits_negative_positive_Nm"]) > 0
        state = initial(current).at[13:].set(jnp.asarray(result["required_command_N"]))
        np.testing.assert_allclose(augmented_dynamics(state, state[13:], current), 0, atol=3e-5)
    current = model._replace(effectiveness=jnp.array([0.4, 1, 1, 1]))
    result = hover_authority(current)
    assert not result["trim_feasible"]
    assert result["trim_classification"] == "proven_infeasible"
    assert "not all flight" in result["infeasibility_scope"]
    # Total maximum thrust still exceeds weight: this alone is insufficient.
    assert float(jnp.sum(current.effectiveness * current.command_upper)) > 0.04338 * 9.81


@pytest.mark.unit
def test_model_rejects_zero_tau_and_invalid_mixer(model: ActuatorModel) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        validate_actuator_model(model._replace(time_constants=jnp.zeros(4)))
    with pytest.raises(ValueError, match="orthogonal"):
        validate_actuator_model(
            model._replace(force_to_wrench=model.force_to_wrench.at[1, 1].multiply(0.5))
        )
    # The algebraic path does not evaluate any inverse-time expression.
    state = initial(model)[:13]
    zero_tau = model._replace(time_constants=jnp.zeros(4))
    assert np.all(np.isfinite(algebraic_step(state, jnp.full(4, 0.1), zero_tau, 0.04)))
    with pytest.raises(ValueError, match="17"):
        effort_lag_step(state, jnp.full(4, 0.1), model, 0.04)


@pytest.mark.unit
def test_normalized_metric_state_audit_and_no_hidden_command_clip(model: ActuatorModel) -> None:
    state = initial(model)
    span = model.command_upper - model.command_lower
    np.testing.assert_allclose(span @ normalized_command_metric(model) @ span, 4, rtol=1e-6)
    command = jnp.full(4, 0.25)
    assert not actuator_state_valid(state, command, model)
    direct = effort_lag_step(state, command, model, 0.04)
    clipped = effort_lag_step(
        state, jnp.clip(command, model.command_lower, model.command_upper), model, 0.04
    )
    assert np.all(np.asarray(direct[13:]) > np.asarray(clipped[13:]))
    assert actuator_state_valid(state, jnp.full(4, 0.15), model)


@pytest.mark.unit
def test_local_native_fit_pole_coordinate_invariance() -> None:
    fit = fit_native_time_constant()
    params = load_params("cf21B_500")
    hover = fit["operating_points"][1]
    r = hover["rpm"]
    a, b, c = params["rpm2thrust"]
    thrust_slope = b + 2 * c * r
    assert thrust_slope > 0 and 0 < fit["nominal_tau"] < 0.2
    for pair, pole in (
        (params["rotor_dyn_coef"][:2], hover["pole_up_per_s"]),
        (params["rotor_dyn_coef"][2:], hover["pole_down_per_s"]),
    ):
        eps = 0.001
        # ds/dt=f'(r)*dr/dt and local du=f'(r)*dRPM: same local pole.
        rpm_dot = pair[0] * eps + pair[1] * ((r + eps) ** 2 - r**2)
        finite_difference_pole = thrust_slope * rpm_dot / (thrust_slope * eps)
        assert np.isclose(finite_difference_pole, pole, rtol=1e-6)
    assert max(abs(p["thrust_roundtrip_residual_N"]) for p in fit["operating_points"]) < 1e-15
