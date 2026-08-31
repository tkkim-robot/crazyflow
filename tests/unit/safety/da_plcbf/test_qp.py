import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import Bounds, minimize

from crazyflow.safety.da_plcbf.qp import BoxHalfspaceResult, project_box_halfspace


def _scipy_projection(
    nominal: np.ndarray,
    weight: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    normal: np.ndarray,
    minimum: float,
) -> tuple[np.ndarray, float]:
    maximizer = np.where(normal >= 0, upper, lower)
    result = minimize(
        fun=lambda action: 0.5 * np.sum(weight * (action - nominal) ** 2),
        x0=maximizer,
        jac=lambda action: weight * (action - nominal),
        bounds=Bounds(lower, upper),
        constraints={
            "type": "ineq",
            "fun": lambda action: np.dot(normal, action) - minimum,
            "jac": lambda _action: normal,
        },
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert result.success, result.message
    return result.x, float(result.fun)


def _assert_kkt(
    result: BoxHalfspaceResult,
    nominal: np.ndarray,
    weight: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    normal: np.ndarray,
    minimum: float,
    *,
    tolerance: float = 4e-5,
) -> None:
    action = np.asarray(result.action)
    multiplier = float(result.multiplier)
    residual = float(np.dot(normal, action) - minimum)
    reduced_gradient = weight * (action - nominal) - multiplier * normal
    at_lower = np.isclose(action, lower, atol=tolerance, rtol=0.0)
    at_upper = np.isclose(action, upper, atol=tolerance, rtol=0.0)
    free = ~(at_lower | at_upper)

    assert bool(result.feasible)
    assert multiplier >= -tolerance
    assert residual >= -tolerance
    assert abs(multiplier * residual) <= tolerance
    assert np.all(action >= lower - tolerance)
    assert np.all(action <= upper + tolerance)
    assert np.all(np.abs(reduced_gradient[free]) <= tolerance)
    assert np.all(reduced_gradient[at_lower] >= -tolerance)
    assert np.all(reduced_gradient[at_upper] <= tolerance)
    np.testing.assert_allclose(result.constraint_residual, residual, rtol=0.0, atol=tolerance)


def test_qp_returns_box_projection_when_halfspace_is_inactive() -> None:
    nominal = jnp.array([3.0, -2.0])
    weight = jnp.array([2.0, 5.0])
    lower = jnp.array([-1.0, -1.0])
    upper = jnp.array([1.0, 2.0])
    normal = jnp.array([1.0, 1.0])

    result = project_box_halfspace(nominal, weight, lower, upper, normal, jnp.array(-1.0))

    np.testing.assert_allclose(result.action, np.array([1.0, -1.0]), rtol=0.0, atol=0.0)
    assert bool(result.feasible)
    assert result.multiplier == pytest.approx(0.0)
    assert result.constraint_residual == pytest.approx(1.0)
    assert result.objective == pytest.approx(6.5)
    _assert_kkt(
        result,
        np.asarray(nominal),
        np.asarray(weight),
        np.asarray(lower),
        np.asarray(upper),
        np.asarray(normal),
        -1.0,
    )


def test_qp_matches_closed_form_for_an_interior_active_halfspace() -> None:
    nominal = jnp.array([0.0, 0.0])
    weight = jnp.array([2.0, 1.0])
    lower = jnp.array([-10.0, -10.0])
    upper = jnp.array([10.0, 10.0])
    normal = jnp.array([1.0, 2.0])

    result = project_box_halfspace(nominal, weight, lower, upper, normal, jnp.array(3.0))

    np.testing.assert_allclose(result.action, np.array([1.0 / 3.0, 4.0 / 3.0]), atol=2e-6)
    assert result.multiplier == pytest.approx(2.0 / 3.0, abs=2e-6)
    assert result.constraint_residual == pytest.approx(0.0, abs=2e-6)
    assert result.objective == pytest.approx(1.0, abs=3e-6)
    _assert_kkt(
        result,
        np.asarray(nominal),
        np.asarray(weight),
        np.asarray(lower),
        np.asarray(upper),
        np.asarray(normal),
        3.0,
    )


def test_qp_handles_box_saturation_on_the_active_solution() -> None:
    nominal = jnp.array([0.0, 0.0])
    weight = jnp.ones(2)
    lower = jnp.array([-2.0, -2.0])
    upper = jnp.array([0.2, 2.0])
    normal = jnp.ones(2)

    result = project_box_halfspace(nominal, weight, lower, upper, normal, jnp.array(1.0))

    np.testing.assert_allclose(result.action, np.array([0.2, 0.8]), atol=2e-6)
    assert result.multiplier == pytest.approx(0.8, abs=2e-6)
    assert result.constraint_residual == pytest.approx(0.0, abs=2e-6)
    _assert_kkt(
        result,
        np.asarray(nominal),
        np.asarray(weight),
        np.asarray(lower),
        np.asarray(upper),
        np.asarray(normal),
        1.0,
    )


def test_infeasible_halfspace_returns_box_support_maximizer_and_negative_residual() -> None:
    nominal = jnp.array([0.1, -0.3, 0.7])
    weight = jnp.array([1.0, 2.0, 3.0])
    lower = jnp.array([-1.0, -2.0, -3.0])
    upper = jnp.array([2.0, 1.0, 4.0])
    normal = jnp.array([2.0, -1.0, 0.0])
    minimum = jnp.array(7.0)

    result = project_box_halfspace(nominal, weight, lower, upper, normal, minimum)

    np.testing.assert_array_equal(result.action, np.array([2.0, -2.0, 4.0]))
    assert not bool(result.feasible)
    assert result.multiplier == pytest.approx(0.0)
    assert result.constraint_residual == pytest.approx(-1.0)
    assert np.dot(np.asarray(normal), np.asarray(result.action)) == pytest.approx(6.0)


@pytest.mark.parametrize("seed", range(8))
def test_qp_matches_independent_scipy_oracle_and_satisfies_kkt(seed: int) -> None:
    generator = np.random.default_rng(seed)
    dimension = 4
    lower = generator.uniform(-2.0, -0.4, size=dimension)
    upper = lower + generator.uniform(0.7, 2.5, size=dimension)
    nominal = 0.5 * (lower + upper) + generator.normal(scale=0.15, size=dimension)
    weight = generator.uniform(0.25, 3.0, size=dimension)
    normal = generator.normal(size=dimension)
    clipped = np.clip(nominal, lower, upper)
    maximizer = np.where(normal >= 0, upper, lower)
    clipped_value = float(np.dot(normal, clipped))
    maximum_value = float(np.dot(normal, maximizer))
    minimum = clipped_value + 0.37 * (maximum_value - clipped_value)

    expected_action, expected_objective = _scipy_projection(
        nominal, weight, lower, upper, normal, minimum
    )
    result = project_box_halfspace(
        jnp.asarray(nominal),
        jnp.asarray(weight),
        jnp.asarray(lower),
        jnp.asarray(upper),
        jnp.asarray(normal),
        jnp.asarray(minimum),
    )

    np.testing.assert_allclose(result.action, expected_action, rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(result.objective, expected_objective, rtol=5e-5, atol=5e-5)
    _assert_kkt(result, nominal, weight, lower, upper, normal, minimum)


def test_qp_is_jittable_and_vmappable_over_independent_problems() -> None:
    nominal = jnp.array([[0.0, 0.0], [3.0, -2.0], [0.1, -0.3], [-0.2, 0.4]])
    weight = jnp.array([[2.0, 1.0], [1.0, 3.0], [2.0, 2.0], [0.5, 4.0]])
    lower = jnp.array([[-10.0, -10.0], [-1.0, -1.0], [-1.0, -2.0], [-2.0, -1.0]])
    upper = jnp.array([[10.0, 10.0], [1.0, 2.0], [2.0, 1.0], [1.0, 3.0]])
    normal = jnp.array([[1.0, 2.0], [1.0, 1.0], [2.0, -1.0], [-1.0, 0.5]])
    minimum = jnp.array([3.0, -1.0, 7.0, 0.2])

    batched_solver = jax.jit(
        jax.vmap(
            lambda n, w, low, high, direction, bound: project_box_halfspace(
                n, w, low, high, direction, bound
            )
        )
    )
    batched = batched_solver(nominal, weight, lower, upper, normal, minimum)
    sequential = [
        project_box_halfspace(*arguments)
        for arguments in zip(nominal, weight, lower, upper, normal, minimum, strict=True)
    ]

    np.testing.assert_allclose(
        batched.action, jnp.stack([result.action for result in sequential]), atol=1e-6
    )
    np.testing.assert_array_equal(
        batched.feasible, jnp.stack([result.feasible for result in sequential])
    )
    np.testing.assert_allclose(
        batched.constraint_residual,
        jnp.stack([result.constraint_residual for result in sequential]),
        atol=1e-6,
    )
    np.testing.assert_array_equal(batched.feasible, np.array([True, True, False, True]))


def test_qp_rejects_static_input_contract_violations() -> None:
    vector = jnp.ones(2)
    with pytest.raises(ValueError, match="identical shapes"):
        project_box_halfspace(vector, vector, vector, vector, jnp.ones(3), jnp.array(0.0))
    with pytest.raises(ValueError, match="iterations"):
        project_box_halfspace(vector, vector, -vector, vector, vector, jnp.array(0.0), iterations=0)
    with pytest.raises(ValueError, match="tolerance"):
        project_box_halfspace(
            vector, vector, -vector, vector, vector, jnp.array(0.0), tolerance=-1.0
        )
    for invalid_tolerance in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="tolerance"):
            project_box_halfspace(
                vector, vector, -vector, vector, vector, jnp.array(0.0), tolerance=invalid_tolerance
            )


@pytest.mark.parametrize("scale", [1e-20, 1e-7, 1.0, 1e7, 1e20])
def test_qp_solution_is_invariant_to_positive_halfspace_rescaling(scale: float) -> None:
    result = project_box_halfspace(
        nominal=jnp.zeros(2),
        weight=jnp.ones(2),
        lower=jnp.zeros(2),
        upper=jnp.ones(2),
        normal=scale * jnp.ones(2),
        minimum=jnp.asarray(scale),
        tolerance=0.0,
    )

    assert bool(result.feasible)
    np.testing.assert_allclose(result.action, np.array([0.5, 0.5]), rtol=2e-6, atol=2e-6)
    assert result.constraint_residual >= -abs(scale) * 2e-6
