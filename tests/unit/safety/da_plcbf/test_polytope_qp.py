from __future__ import annotations

import itertools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import LinearConstraint, minimize

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.direct_wrench import (
    motor_forces_to_wrench,
    motor_thrust_inequalities,
    wrench_to_motor_forces,
)
from crazyflow.safety.da_plcbf.polytope_qp import PolytopeQPResult, project_affine_polytope


def _objective(action: np.ndarray, nominal: np.ndarray, weight: np.ndarray) -> float:
    weight_matrix = np.diag(weight) if weight.ndim == 1 else weight
    delta = action - nominal
    return float(0.5 * delta @ weight_matrix @ delta)


def _scipy_projection(
    nominal: np.ndarray,
    weight: np.ndarray,
    matrix: np.ndarray,
    upper_bound: np.ndarray,
    feasible_start: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Independent dense SLSQP oracle used only by tests."""
    weight_matrix = np.diag(weight) if weight.ndim == 1 else weight
    row_norms = np.linalg.norm(matrix, axis=-1)
    safe_row_norms = np.where(row_norms > 0, row_norms, 1.0)
    normalised_matrix = matrix / safe_row_norms[:, None]
    normalised_bound = upper_bound / safe_row_norms
    result = minimize(
        fun=lambda action: _objective(action, nominal, weight),
        x0=feasible_start,
        jac=lambda action: weight_matrix @ (action - nominal),
        constraints=LinearConstraint(normalised_matrix, -np.inf, normalised_bound),
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    assert result.success, result.message
    return result.x, float(result.fun)


def _assert_auditable_kkt(
    result: PolytopeQPResult,
    nominal: np.ndarray,
    weight: np.ndarray,
    matrix: np.ndarray,
    upper_bound: np.ndarray,
    *,
    tolerance: float = 3e-4,
) -> None:
    action = np.asarray(result.action)
    multipliers = np.asarray(result.multipliers)
    nominal = np.asarray(nominal, dtype=action.dtype)
    weight = np.asarray(weight, dtype=action.dtype)
    matrix = np.asarray(matrix, dtype=action.dtype)
    upper_bound = np.asarray(upper_bound, dtype=action.dtype)
    weight_matrix = np.diag(weight) if weight.ndim == 1 else weight
    constraint_residuals = matrix @ action - upper_bound
    stationarity = weight_matrix @ (action - nominal) + matrix.T @ multipliers

    assert bool(result.input_valid)
    assert bool(result.feasible)
    assert np.all(np.isfinite(action))
    assert np.max(constraint_residuals, initial=0.0) <= tolerance
    assert np.min(multipliers, initial=0.0) >= -tolerance
    assert np.max(np.abs(stationarity), initial=0.0) <= tolerance
    assert np.max(np.abs(multipliers * constraint_residuals), initial=0.0) <= tolerance
    assert int(result.active_count) == np.count_nonzero(result.active_mask)
    assert int(result.active_count) <= nominal.size
    assert np.all(multipliers[~np.asarray(result.active_mask)] == 0)

    expected_primal = max(float(np.max(constraint_residuals, initial=0.0)), 0.0)
    expected_dual = max(float(np.max(-multipliers, initial=0.0)), 0.0)
    expected_stationarity = float(np.max(np.abs(stationarity), initial=0.0))
    expected_complementarity = float(
        np.max(np.abs(multipliers * constraint_residuals), initial=0.0)
    )
    # XLA and NumPy use different reduction orders for these float32 products.  Account only for
    # that roundoff; the independently checked KKT bounds above remain the substantive test.
    audit_agreement = max(1e-5, 0.05 * tolerance)
    assert result.primal_residual == pytest.approx(expected_primal, abs=audit_agreement)
    assert result.dual_residual == pytest.approx(expected_dual, abs=audit_agreement)
    assert result.stationarity_residual == pytest.approx(expected_stationarity, abs=audit_agreement)
    assert result.complementarity_residual == pytest.approx(
        expected_complementarity, abs=audit_agreement
    )
    assert result.objective == pytest.approx(
        _objective(action, nominal, weight), rel=2e-5, abs=2e-6
    )


@pytest.mark.unit
def test_interior_nominal_is_returned_without_an_active_face() -> None:
    nominal = np.array([0.2, -0.3, 0.1])
    weight = np.array([[2.0, 0.2, 0.0], [0.2, 3.0, -0.1], [0.0, -0.1, 1.0]])
    matrix = np.concatenate((np.eye(3), -np.eye(3)), axis=0)
    upper_bound = np.ones(6)

    result = project_affine_polytope(
        jnp.asarray(nominal), jnp.asarray(weight), jnp.asarray(matrix), jnp.asarray(upper_bound)
    )

    np.testing.assert_allclose(result.action, nominal, rtol=0.0, atol=1e-7)
    assert result.objective == pytest.approx(0.0)
    assert int(result.active_count) == 0
    assert not np.any(result.active_mask)
    _assert_auditable_kkt(result, nominal, weight, matrix, upper_bound)


@pytest.mark.unit
def test_projection_onto_multiple_faces_and_corner_has_exact_kkt_record() -> None:
    nominal = np.array([3.0, -4.0, 0.25])
    weight = np.array([2.0, 0.5, 4.0])
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    upper_bound = np.ones(6)

    result = project_affine_polytope(
        jnp.asarray(nominal), jnp.asarray(weight), jnp.asarray(matrix), jnp.asarray(upper_bound)
    )

    np.testing.assert_allclose(result.action, np.array([1.0, -1.0, 0.25]), atol=1e-7)
    np.testing.assert_array_equal(
        result.active_mask, np.array([True, False, False, True, False, False])
    )
    assert int(result.active_count) == 2
    np.testing.assert_allclose(
        result.multipliers, np.array([4.0, 0.0, 0.0, 1.5, 0.0, 0.0]), atol=1e-6
    )
    _assert_auditable_kkt(result, nominal, weight, matrix, upper_bound)


@pytest.mark.unit
def test_redundant_and_rank_deficient_faces_do_not_hide_the_solution() -> None:
    nominal = np.array([4.0, 3.0])
    weight = np.array([[2.0, 0.25], [0.25, 1.5]])
    matrix = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],  # Same face with a different row scale.
            [1.0, 0.0],  # Exact duplicate.
            [0.0, 1.0],
            [1.0, 1.0],  # Also active at the corner, but dependent on rows 0 and 3.
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    upper_bound = np.array([1.0, 2.0, 1.0, 1.0, 2.0, 5.0, 5.0])

    result = project_affine_polytope(
        jnp.asarray(nominal), jnp.asarray(weight), jnp.asarray(matrix), jnp.asarray(upper_bound)
    )

    np.testing.assert_allclose(result.action, np.ones(2), atol=2e-6)
    assert int(result.active_count) == 2
    assert np.linalg.matrix_rank(matrix[np.asarray(result.active_mask)]) == 2
    _assert_auditable_kkt(result, nominal, weight, matrix, upper_bound)


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(8))
def test_random_feasible_problems_match_independent_scipy(seed: int) -> None:
    generator = np.random.default_rng(seed)
    dimension = 4
    constraint_count = 9
    feasible_start = generator.normal(scale=0.25, size=dimension)
    matrix = generator.normal(size=(constraint_count, dimension))
    # Deliberately vary row scales; internal normalisation must leave the answer unchanged.
    matrix *= 10.0 ** generator.uniform(-2.0, 2.0, size=(constraint_count, 1))
    slack = 10.0 ** generator.uniform(-1.0, 0.3, size=constraint_count)
    upper_bound = matrix @ feasible_start + slack
    nominal = feasible_start + generator.normal(scale=3.0, size=dimension)
    factor = generator.normal(size=(dimension, dimension))
    weight = factor.T @ factor + 0.5 * np.eye(dimension)

    expected_action, expected_objective = _scipy_projection(
        nominal, weight, matrix, upper_bound, feasible_start
    )
    result = project_affine_polytope(
        jnp.asarray(nominal, dtype=jnp.float32),
        jnp.asarray(weight, dtype=jnp.float32),
        jnp.asarray(matrix, dtype=jnp.float32),
        jnp.asarray(upper_bound, dtype=jnp.float32),
    )

    np.testing.assert_allclose(result.action, expected_action, rtol=5e-4, atol=5e-4)
    assert result.objective == pytest.approx(expected_objective, rel=8e-4, abs=5e-4)
    _assert_auditable_kkt(result, nominal, weight, matrix, upper_bound, tolerance=8e-4)


@pytest.mark.unit
def test_infeasible_polytope_returns_fail_closed_sentinel() -> None:
    # x <= 0 and x >= 1 cannot both hold.
    result = project_affine_polytope(
        jnp.array([0.4]), jnp.array([2.0]), jnp.array([[1.0], [-1.0]]), jnp.array([0.0, -1.0])
    )

    assert bool(result.input_valid)
    assert not bool(result.feasible)
    assert np.all(np.isnan(result.action))
    assert np.isinf(result.objective)
    assert np.isinf(result.primal_residual)
    assert np.isinf(result.dual_residual)
    assert np.isinf(result.stationarity_residual)
    assert np.isinf(result.complementarity_residual)
    assert int(result.active_count) == 0
    assert not np.any(result.active_mask)
    np.testing.assert_array_equal(result.multipliers, np.zeros(2))


@pytest.mark.unit
def test_impossible_zero_normal_face_is_reported_infeasible() -> None:
    result = project_affine_polytope(
        jnp.zeros(2), jnp.ones(2), jnp.array([[0.0, 0.0]]), jnp.array([-0.1])
    )
    assert bool(result.input_valid)
    assert not bool(result.feasible)
    assert np.all(np.isnan(result.action))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("nominal", "weight", "matrix", "upper_bound", "tolerance", "rank_tolerance"),
    [
        ([np.nan, 0.0], [1.0, 1.0], [[1.0, 0.0]], [1.0], 1e-6, 1e-7),
        ([0.0, 0.0], [1.0, -1.0], [[1.0, 0.0]], [1.0], 1e-6, 1e-7),
        ([0.0, 0.0], [1.0, np.inf], [[1.0, 0.0]], [1.0], 1e-6, 1e-7),
        ([0.0, 0.0], [[1.0, 2.0], [0.0, 1.0]], [[1.0, 0.0]], [1.0], 1e-6, 1e-7),
        ([0.0, 0.0], [[1.0, 0.0], [0.0, -1.0]], [[1.0, 0.0]], [1.0], 1e-6, 1e-7),
        ([0.0, 0.0], [1.0, 1.0], [[np.inf, 0.0]], [1.0], 1e-6, 1e-7),
        ([0.0, 0.0], [1.0, 1.0], [[1.0, 0.0]], [np.nan], 1e-6, 1e-7),
        ([0.0, 0.0], [1.0, 1.0], [[1.0, 0.0]], [1.0], -1.0, 1e-7),
        ([0.0, 0.0], [1.0, 1.0], [[1.0, 0.0]], [1.0], 1e-6, 0.0),
    ],
)
def test_nonfinite_or_invalid_runtime_data_is_explicitly_rejected(
    nominal: list[float],
    weight: list[float] | list[list[float]],
    matrix: list[list[float]],
    upper_bound: list[float],
    tolerance: float,
    rank_tolerance: float,
) -> None:
    result = project_affine_polytope(
        jnp.asarray(nominal),
        jnp.asarray(weight),
        jnp.asarray(matrix),
        jnp.asarray(upper_bound),
        tolerance=tolerance,
        rank_tolerance=rank_tolerance,
    )
    assert not bool(result.input_valid)
    assert not bool(result.feasible)
    assert np.all(np.isnan(result.action))
    assert np.isinf(result.objective)


@pytest.mark.unit
def test_empty_constraint_set_is_an_audited_unconstrained_projection() -> None:
    nominal = jnp.array([0.2, -0.4])
    result = project_affine_polytope(
        nominal, jnp.array([2.0, 3.0]), jnp.empty((0, 2)), jnp.empty((0,))
    )
    np.testing.assert_array_equal(result.action, nominal)
    assert bool(result.feasible)
    assert result.objective == pytest.approx(0.0)
    assert result.primal_residual == pytest.approx(0.0)
    assert result.dual_residual == pytest.approx(0.0)
    assert result.complementarity_residual == pytest.approx(0.0)


@pytest.mark.unit
def test_solver_is_jittable_and_vmappable() -> None:
    matrix = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    upper_bound = jnp.ones(4)
    nominals = jnp.array([[2.0, 3.0], [0.25, -0.5], [-4.0, 0.1]])
    weights = jnp.array([[1.0, 2.0], [3.0, 0.5], [2.0, 4.0]])
    compiled = jax.jit(
        jax.vmap(
            lambda nominal, weight: project_affine_polytope(nominal, weight, matrix, upper_bound)
        )
    )

    result = compiled(nominals, weights)

    np.testing.assert_allclose(result.action, np.array([[1.0, 1.0], [0.25, -0.5], [-1.0, 0.1]]))
    np.testing.assert_array_equal(result.feasible, np.ones(3, dtype=bool))
    np.testing.assert_array_equal(result.active_count, np.array([2, 0, 1]))
    assert np.all(np.asarray(result.primal_residual) <= 1e-6)
    assert np.all(np.asarray(result.stationarity_residual) <= 1e-6)


@pytest.mark.unit
def test_static_shape_and_dtype_contracts_raise_clear_errors() -> None:
    nominal = jnp.ones(2)
    with pytest.raises(ValueError, match="one-dimensional"):
        project_affine_polytope(jnp.ones((1, 2)), nominal, jnp.ones((1, 2)), jnp.ones(1))
    with pytest.raises(ValueError, match="weight"):
        project_affine_polytope(nominal, jnp.ones(3), jnp.ones((1, 2)), jnp.ones(1))
    with pytest.raises(ValueError, match="matrix"):
        project_affine_polytope(nominal, nominal, jnp.ones((1, 3)), jnp.ones(1))
    with pytest.raises(ValueError, match="upper_bound"):
        project_affine_polytope(nominal, nominal, jnp.ones((1, 2)), jnp.ones(2))
    with pytest.raises(ValueError, match="floating-point"):
        project_affine_polytope(
            jnp.ones(2, dtype=jnp.int32), jnp.ones(2), jnp.ones((1, 2)), jnp.ones(1)
        )


def _cf21b() -> dict[str, Any]:
    params = load_params("cf21B_500")
    return {
        "mass": params["mass"],
        "thrust_min": params["thrust_min"],
        "thrust_max": params["thrust_max"],
        "L": params["L"],
        "thrust2torque": params["thrust2torque"],
        "mixing_matrix": np.asarray(params["mixing_matrix"]),
    }


def _allocation_params(params: dict[str, Any]) -> dict[str, Any]:
    return {name: params[name] for name in ("L", "thrust2torque", "mixing_matrix")}


@pytest.mark.unit
def test_cf21b_eight_motor_faces_plus_plcbf_face_project_to_hover() -> None:
    params = _cf21b()
    motor_constraints = motor_thrust_inequalities(
        thrust_min=params["thrust_min"],
        thrust_max=params["thrust_max"],
        **_allocation_params(params),
    )
    hover_thrust = params["mass"] * 9.81
    plcbf_row = np.array([[-1.0, 0.0, 0.0, 0.0]])
    matrix = np.concatenate((np.asarray(motor_constraints.matrix), plcbf_row), axis=0)
    upper_bound = np.concatenate(
        (np.asarray(motor_constraints.upper_bound), np.array([-hover_thrust]))
    )
    nominal = np.zeros(4)
    weight = np.array([1.0, 2.0e4, 2.0e4, 2.0e4])

    result = project_affine_polytope(
        jnp.asarray(nominal), jnp.asarray(weight), jnp.asarray(matrix), jnp.asarray(upper_bound)
    )

    np.testing.assert_allclose(result.action, np.array([hover_thrust, 0.0, 0.0, 0.0]), atol=3e-6)
    assert result.active_mask[-1]
    motor_forces = wrench_to_motor_forces(result.action, **_allocation_params(params))
    assert np.all(motor_forces >= params["thrust_min"] - 2e-6)
    assert np.all(motor_forces <= params["thrust_max"] + 2e-6)
    _assert_auditable_kkt(result, nominal, weight, matrix, upper_bound, tolerance=5e-4)


@pytest.mark.unit
def test_cf21b_coupled_polytope_rejects_naive_wrench_box_corner_and_projects_it() -> None:
    params = _cf21b()
    low, high = params["thrust_min"], params["thrust_max"]
    motor_vertices = np.where(
        np.asarray(list(itertools.product((False, True), repeat=4))), high, low
    )
    wrench_vertices = motor_forces_to_wrench(motor_vertices, **_allocation_params(params))
    component_min = np.min(wrench_vertices, axis=0)
    component_max = np.max(wrench_vertices, axis=0)
    naive_box_corner = component_max

    # A component-wise wrench box accepts this point, although its implied individual motor
    # forces violate the coupled airborne actuator set.
    assert np.all(naive_box_corner >= component_min)
    assert np.all(naive_box_corner <= component_max)
    implied_motors = wrench_to_motor_forces(naive_box_corner, **_allocation_params(params))
    assert np.any((implied_motors < low) | (implied_motors > high))

    motor_constraints = motor_thrust_inequalities(
        thrust_min=low, thrust_max=high, **_allocation_params(params)
    )
    plcbf_minimum_thrust = params["mass"] * 9.81
    matrix = np.concatenate(
        (np.asarray(motor_constraints.matrix), np.array([[-1.0, 0.0, 0.0, 0.0]])), axis=0
    )
    upper_bound = np.concatenate(
        (np.asarray(motor_constraints.upper_bound), np.array([-plcbf_minimum_thrust]))
    )
    weight = np.array([1.0, 2.0e4, 2.0e4, 2.0e4])
    feasible_start = np.array([plcbf_minimum_thrust, 0.0, 0.0, 0.0])
    expected_action, expected_objective = _scipy_projection(
        naive_box_corner, weight, matrix, upper_bound, feasible_start
    )

    result = project_affine_polytope(
        jnp.asarray(naive_box_corner),
        jnp.asarray(weight),
        jnp.asarray(matrix),
        jnp.asarray(upper_bound),
    )
    projected_motors = wrench_to_motor_forces(result.action, **_allocation_params(params))

    assert bool(result.feasible)
    assert not np.allclose(result.action, naive_box_corner)
    assert np.all(projected_motors >= low - 3e-6)
    assert np.all(projected_motors <= high + 3e-6)
    np.testing.assert_allclose(result.action, expected_action, rtol=8e-4, atol=3e-5)
    assert result.objective == pytest.approx(expected_objective, rel=1e-3, abs=3e-5)
    _assert_auditable_kkt(result, naive_box_corner, weight, matrix, upper_bound, tolerance=8e-4)
