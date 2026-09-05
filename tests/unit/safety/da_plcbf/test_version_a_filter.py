from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.direct_wrench import motor_forces_to_wrench, wrench_to_motor_forces
from crazyflow.safety.da_plcbf.selector import SelectionConfig
from crazyflow.safety.da_plcbf.version_a_barriers import (
    RigidBodySafetySet,
    VersionABarrierConfig,
    VersionAModel,
    hard_finite_horizon_policy_certificate,
)
from crazyflow.safety.da_plcbf.version_a_filter import (
    PolicyLibraryCertificates,
    VersionAActuator,
    VersionAFilterConfig,
    VersionAFilterResult,
    motor_box_halfspace_fraction,
    validated_motor_polytope,
    version_a_plcbf_filter,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="module", autouse=True)
def _enable_x64() -> Iterator[None]:
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def setup() -> tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet]:
    params = load_params("cf21B_500")
    inertia = jnp.asarray(params["J"], dtype=jnp.float64)
    model = VersionAModel(
        jnp.asarray(params["mass"], dtype=jnp.float64),
        jnp.asarray(params["gravity_vec"], dtype=jnp.float64),
        inertia,
        jnp.linalg.inv(inertia),
        jnp.asarray(params["drag_matrix"], dtype=jnp.float64),
        jnp.zeros(3, dtype=jnp.float64),
        jnp.zeros(3, dtype=jnp.float64),
        jnp.zeros(3, dtype=jnp.float64),
    )
    actuator = VersionAActuator(
        jnp.asarray(params["L"], dtype=jnp.float64),
        jnp.asarray(params["thrust2torque"], dtype=jnp.float64),
        jnp.asarray(params["mixing_matrix"], dtype=jnp.float64),
        jnp.asarray(params["thrust_min"], dtype=jnp.float64),
        jnp.asarray(params["thrust_max"], dtype=jnp.float64),
    )
    safety = RigidBodySafetySet(
        jnp.empty((0, 3), dtype=jnp.float64),
        jnp.empty((0,), dtype=jnp.float64),
        jnp.empty((0,), dtype=bool),
        jnp.asarray([-5.0, -5.0, 0.0], dtype=jnp.float64),
        jnp.asarray([5.0, 5.0, 5.0], dtype=jnp.float64),
        jnp.asarray(5.0, dtype=jnp.float64),
        jnp.asarray(10.0, dtype=jnp.float64),
        jnp.asarray(0.7, dtype=jnp.float64),
    )
    return params, model, actuator, safety


def _state(*, altitude: float = 1.0) -> jax.Array:
    return jnp.asarray(
        [0.0, 0.0, altitude, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64
    )


def _library(hover: jax.Array, values: tuple[float, ...] = (1.0,)) -> PolicyLibraryCertificates:
    policy_count = len(values)
    gradients = jnp.zeros((policy_count, 13), dtype=jnp.float64).at[:, 9].set(1.0)
    return PolicyLibraryCertificates(
        jnp.asarray(values, dtype=jnp.float64),
        gradients,
        jnp.ones((policy_count,), dtype=bool),
        jnp.broadcast_to(hover, (policy_count, 4)),
    )


def _filter(
    setup_data: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
    *,
    state: jax.Array | None = None,
    nominal: jax.Array | None = None,
    weight: jax.Array | None = None,
    library: PolicyLibraryCertificates | None = None,
    actuator: VersionAActuator | None = None,
    previous_policy_index: jax.Array | None = None,
    selection_config: SelectionConfig = SelectionConfig(),
) -> VersionAFilterResult:
    params, model, default_actuator, safety = setup_data
    hover = jnp.asarray([params["mass"] * 9.81, 0.0, 0.0, 0.0], dtype=jnp.float64)
    return version_a_plcbf_filter(
        _state() if state is None else state,
        jnp.zeros(4, dtype=jnp.float64) if nominal is None else nominal,
        (jnp.asarray([1.0, 2.0e4, 2.0e4, 2.0e4], dtype=jnp.float64) if weight is None else weight),
        _library(hover) if library is None else library,
        model,
        default_actuator if actuator is None else actuator,
        safety,
        VersionABarrierConfig(),
        VersionAFilterConfig(),
        previous_policy_index=previous_policy_index,
        selection_config=selection_config,
    )


@pytest.mark.unit
def test_motor_polytope_validates_allocation_identity_and_airborne_midpoint(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
) -> None:
    _, _, actuator, _ = setup
    polytope = validated_motor_polytope(actuator, jnp.float64)
    midpoint_forces = wrench_to_motor_forces(
        polytope.midpoint_wrench,
        L=actuator.arm_length,
        thrust2torque=actuator.thrust_to_torque,
        mixing_matrix=actuator.mixing_matrix,
    )

    assert bool(polytope.input_valid)
    assert polytope.allocation_identity_error <= 2e-12
    np.testing.assert_allclose(
        midpoint_forces, 0.5 * (polytope.thrust_min + polytope.thrust_max), atol=2e-12
    )
    assert np.max(polytope.matrix @ polytope.midpoint_wrench - polytope.upper_bound) <= 2e-12


@pytest.mark.unit
def test_motor_box_halfspace_fraction_is_exact_and_scale_invariant(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
) -> None:
    _, _, actuator, _ = setup
    polytope = validated_motor_polytope(actuator, jnp.float64)
    row = polytope.allocation_matrix[0]
    bound = polytope.thrust_min[0] + 0.25 * (polytope.thrust_max[0] - polytope.thrust_min[0])

    assert motor_box_halfspace_fraction(row, bound, polytope) == pytest.approx(0.25, abs=2e-12)
    assert motor_box_halfspace_fraction(1e-20 * row, 1e-20 * bound, polytope) == pytest.approx(
        0.25, abs=2e-12
    )
    assert motor_box_halfspace_fraction(jnp.zeros(4), jnp.asarray(0.0), polytope) == pytest.approx(
        1.0
    )
    assert motor_box_halfspace_fraction(jnp.zeros(4), jnp.asarray(-1.0), polytope) == pytest.approx(
        0.0
    )
    collective_row = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    motor_range = polytope.thrust_max[0] - polytope.thrust_min[0]
    collective_lower = jnp.sum(polytope.thrust_min)
    assert motor_box_halfspace_fraction(
        collective_row, collective_lower + motor_range, polytope
    ) == pytest.approx(1.0 / 24.0, abs=2e-12)
    assert motor_box_halfspace_fraction(
        collective_row, collective_lower + 2.0 * motor_range, polytope
    ) == pytest.approx(0.5, abs=2e-12)


@pytest.mark.unit
def test_filter_projects_to_exact_plcbf_face_with_auditable_kkt_and_unchanged_allocation(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
) -> None:
    params, _, actuator, _ = setup
    result = _filter(setup)
    expected_thrust = params["mass"] * (9.81 - 2.0)
    motor_forces = wrench_to_motor_forces(
        result.action,
        L=actuator.arm_length,
        thrust2torque=actuator.thrust_to_torque,
        mixing_matrix=actuator.mixing_matrix,
    )
    reconstructed = motor_forces_to_wrench(
        motor_forces,
        L=actuator.arm_length,
        thrust2torque=actuator.thrust_to_torque,
        mixing_matrix=actuator.mixing_matrix,
    )

    assert bool(result.input_valid)
    assert bool(result.has_certificate)
    assert bool(result.qp_feasible)
    assert bool(result.qp_accepted)
    assert not bool(result.used_fallback)
    assert not bool(result.degraded)
    assert bool(result.action_executable)
    np.testing.assert_allclose(result.action, [expected_thrust, 0.0, 0.0, 0.0], atol=3e-9)
    assert result.qp.primal_residual <= 1e-10
    assert result.qp.dual_residual <= 1e-10
    assert result.qp.stationarity_residual <= 1e-8
    assert result.qp.complementarity_residual <= 1e-10
    assert result.applied_postcheck.minimum_motor_margin >= -1e-12
    assert result.applied_postcheck.policy_barrier_residual >= -1e-10
    assert result.applied_postcheck.minimum_analytic_barrier_residual >= -1e-10
    assert result.applied_postcheck.allocation_roundtrip_error <= 2e-12
    np.testing.assert_allclose(reconstructed, result.action, atol=2e-12)


@pytest.mark.unit
def test_policy_selection_maximizes_admissible_fraction_with_hard_value_tie_break(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
) -> None:
    params, _, _, _ = setup
    hover = jnp.asarray([params["mass"] * 9.81, 0.0, 0.0, 0.0], dtype=jnp.float64)
    result = _filter(setup, library=_library(hover, values=(0.2, 0.9, 0.5)))

    assert int(result.selected_index) == 1
    np.testing.assert_array_equal(result.policy_eligible, np.ones(3, dtype=bool))
    assert result.policy_admissible_fractions[1] == jnp.max(result.policy_admissible_fractions)
    assert bool(result.qp_accepted)


@pytest.mark.unit
def test_policy_selection_retains_an_eligible_incumbent_only_within_documented_hysteresis(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
) -> None:
    params, _, _, _ = setup
    hover = jnp.asarray([params["mass"] * 9.81, 0.0, 0.0, 0.0], dtype=jnp.float64)
    library = _library(hover, values=(0.2, 0.9))

    switched = _filter(
        setup,
        library=library,
        previous_policy_index=jnp.asarray(0),
        selection_config=SelectionConfig(switch_score_margin=0.0),
    )
    retained = _filter(
        setup,
        library=library,
        previous_policy_index=jnp.asarray(0),
        selection_config=SelectionConfig(switch_score_margin=1.0),
    )

    assert int(switched.selected_index) == 1
    assert bool(switched.selection.switched)
    assert not bool(switched.selection.retained_by_hysteresis)
    assert int(retained.selected_index) == 0
    assert not bool(retained.selection.switched)
    assert bool(retained.selection.retained_by_hysteresis)


@pytest.mark.unit
def test_negative_hard_value_threshold_cannot_relabel_an_unsafe_policy_as_certified(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
) -> None:
    with pytest.raises(ValueError, match="minimum_hard_value must be nonnegative"):
        _filter(setup, selection_config=SelectionConfig(minimum_hard_value=-0.1))


@pytest.mark.unit
def test_hard_rollout_certificate_and_spherical_hocbf_are_connected_end_to_end(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
) -> None:
    params, model, actuator, _ = setup
    safety = RigidBodySafetySet(
        jnp.asarray([[0.0, 0.0, 2.5]], dtype=jnp.float64),
        jnp.asarray([0.4], dtype=jnp.float64),
        jnp.asarray([True]),
        jnp.asarray([-100.0, -100.0, -100.0], dtype=jnp.float64),
        jnp.asarray([200.0, 300.0, 400.0], dtype=jnp.float64),
        jnp.asarray(5.0, dtype=jnp.float64),
        jnp.asarray(10.0, dtype=jnp.float64),
        jnp.asarray(0.7, dtype=jnp.float64),
    )
    state = _state().at[9].set(1.0)
    barrier_config = VersionABarrierConfig()
    certificate = hard_finite_horizon_policy_certificate(
        state, lambda x: x[None, :], safety, barrier_config
    )
    fallback = jnp.asarray([4.0 * params["thrust_min"], 0.0, 0.0, 0.0], dtype=jnp.float64)
    library = PolicyLibraryCertificates(
        certificate.value[None],
        certificate.gradient[None, :],
        certificate.gradient_valid[None],
        fallback[None, :],
    )
    nominal = jnp.asarray([params["mass"] * 9.81, 0.0, 0.0, 0.0], dtype=jnp.float64)

    result = version_a_plcbf_filter(
        state,
        nominal,
        jnp.asarray([1.0, 2.0e4, 2.0e4, 2.0e4], dtype=jnp.float64),
        library,
        model,
        actuator,
        safety,
        barrier_config,
        VersionAFilterConfig(),
    )

    assert bool(certificate.gradient_valid)
    assert bool(result.has_certificate)
    assert bool(result.qp_accepted)
    assert not bool(result.degraded)
    assert result.action[0] < nominal[0]
    assert result.applied_postcheck.analytic_barrier_residuals[0] >= -3e-9
    assert result.applied_postcheck.policy_barrier_residual >= -3e-9


@pytest.mark.unit
def test_no_safe_policy_uses_transparent_actuator_feasible_best_effort_and_is_degraded(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
) -> None:
    params, _, actuator, _ = setup
    hover = jnp.asarray([params["mass"] * 9.81, 0.0, 0.0, 0.0], dtype=jnp.float64)
    library = _library(hover, values=(-0.4, -0.1))
    result = _filter(setup, library=library)

    assert int(result.selected_index) == 1
    assert not bool(result.has_certificate)
    assert not bool(result.qp_feasible)
    assert not bool(result.qp_accepted)
    assert bool(result.used_fallback)
    assert not bool(result.used_midpoint)
    assert bool(result.degraded)
    assert bool(result.action_executable)
    np.testing.assert_allclose(result.action, hover, atol=1e-12)
    implied_forces = wrench_to_motor_forces(
        result.action,
        L=actuator.arm_length,
        thrust2torque=actuator.thrust_to_torque,
        mixing_matrix=actuator.mixing_matrix,
    )
    assert np.all(implied_forces >= params["thrust_min"])
    assert np.all(implied_forces <= params["thrust_max"])


@pytest.mark.unit
@pytest.mark.parametrize("invalid_source", ["state", "nominal", "gradient", "value"])
def test_nonfinite_runtime_inputs_never_produce_a_false_certificate(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
    invalid_source: str,
) -> None:
    params, _, _, _ = setup
    hover = jnp.asarray([params["mass"] * 9.81, 0.0, 0.0, 0.0], dtype=jnp.float64)
    state = _state()
    nominal = jnp.zeros(4, dtype=jnp.float64)
    library = _library(hover)
    if invalid_source == "state":
        state = state.at[0].set(jnp.nan)
    elif invalid_source == "nominal":
        nominal = nominal.at[1].set(jnp.inf)
    elif invalid_source == "gradient":
        library = library._replace(gradients=library.gradients.at[0, 3].set(jnp.nan))
    else:
        library = library._replace(values=library.values.at[0].set(jnp.nan))

    result = _filter(setup, state=state, nominal=nominal, library=library)

    assert not bool(result.input_valid)
    assert not bool(result.has_certificate)
    assert not bool(result.qp_accepted)
    assert bool(result.used_fallback)
    assert bool(result.degraded)
    assert bool(result.action_executable)
    assert np.all(np.isfinite(result.action))


@pytest.mark.unit
@pytest.mark.parametrize(
    "weight",
    [
        np.asarray([1.0, -1.0, 1.0, 1.0]),
        np.asarray([[1.0, 0.2, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], *np.eye(4)[2:]]),
    ],
)
def test_invalid_qp_weight_fails_closed_to_degraded_fallback(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
    weight: jax.Array,
) -> None:
    result = _filter(setup, weight=weight)

    assert not bool(result.input_valid)
    assert not bool(result.has_certificate)
    assert not bool(result.qp_accepted)
    assert bool(result.degraded)
    assert bool(result.action_executable)


@pytest.mark.unit
def test_invalid_allocation_is_a_nonexecutable_nan_sentinel_not_a_clipped_false_success(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
) -> None:
    _, _, actuator, _ = setup
    invalid_actuator = actuator._replace(thrust_to_torque=jnp.asarray(jnp.nan))
    result = _filter(setup, actuator=invalid_actuator)

    assert not bool(result.input_valid)
    assert not bool(result.has_certificate)
    assert not bool(result.qp_accepted)
    assert bool(result.degraded)
    assert not bool(result.action_executable)
    assert np.all(np.isnan(result.action))


@pytest.mark.unit
def test_outside_hocbf_domain_cannot_be_relabelled_as_a_valid_continuous_certificate(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
) -> None:
    result = _filter(setup, state=_state(altitude=-0.1))

    assert not bool(result.has_certificate)
    assert not bool(result.qp_accepted)
    assert bool(result.degraded)
    assert bool(result.used_fallback)


@pytest.mark.unit
def test_infeasible_fallback_wrench_uses_motor_midpoint_and_marks_degraded(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
) -> None:
    bad_fallback = jnp.asarray([[100.0, 100.0, 100.0, 100.0]], dtype=jnp.float64)
    library = PolicyLibraryCertificates(
        jnp.asarray([-1.0], dtype=jnp.float64),
        jnp.zeros((1, 13), dtype=jnp.float64),
        jnp.asarray([True]),
        bad_fallback,
    )
    result = _filter(setup, library=library)

    assert not bool(result.has_certificate)
    assert bool(result.used_fallback)
    assert bool(result.used_midpoint)
    assert bool(result.degraded)
    assert bool(result.action_executable)
    assert result.applied_postcheck.minimum_motor_margin >= -1e-12


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        VersionAFilterConfig(policy_alpha=np.nan),
        VersionAFilterConfig(qp_tolerance=-1.0),
        VersionAFilterConfig(qp_rank_tolerance=0.0),
    ],
)
def test_invalid_filter_configuration_is_rejected(config: VersionAFilterConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_analytic_only_omits_policy_row_and_matches_independent_baseline(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
) -> None:
    from crazyflow.safety.da_plcbf.version_a_analytic_filter import version_a_analytic_filter

    params, model, actuator, safety = setup
    hover = jnp.asarray([params["mass"] * 9.81, 0.0, 0.0, 0.0], dtype=jnp.float64)
    library = _library(hover, values=(-1.0,))._replace(gradient_valid=jnp.asarray([False]))
    weight = jnp.asarray([1.0, 2.0e4, 2.0e4, 2.0e4], dtype=jnp.float64)
    nominal = jnp.zeros(4, dtype=jnp.float64)
    result = version_a_plcbf_filter(
        _state(),
        nominal,
        weight,
        library,
        model,
        actuator,
        safety,
        VersionABarrierConfig(),
        VersionAFilterConfig(enforce_policy_barrier=False),
    )
    baseline = version_a_analytic_filter(
        _state(), nominal, weight, model, actuator, safety, VersionABarrierConfig()
    )
    assert not bool(result.has_certificate)
    assert bool(result.qp_accepted)
    assert bool(result.qp_kkt_valid)
    assert float(result.selected_policy_dual) == 0.0
    assert result.qp.multipliers.shape == baseline.qp.multipliers.shape
    np.testing.assert_allclose(result.action, baseline.action, atol=1e-12)
    np.testing.assert_allclose(result.qp.multipliers, baseline.qp.multipliers, atol=1e-12)


def test_conservative_barrier_uses_its_own_value_with_matching_gradient(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
) -> None:
    params, model, actuator, safety = setup
    hover = jnp.asarray([params["mass"] * 9.81, 0.0, 0.0, 0.0], dtype=jnp.float64)
    library = _library(hover, values=(1.0,))._replace(barrier_values=jnp.asarray([0.1]))
    result = version_a_plcbf_filter(
        _state(),
        hover,
        jnp.ones(4),
        library,
        model,
        actuator,
        safety,
        VersionABarrierConfig(),
        VersionAFilterConfig(enforce_analytic_barriers=False),
    )
    np.testing.assert_allclose(result.selected_policy_bound, -9.81 + 2.0 * 0.1, atol=1e-12)
    assert bool(result.qp_accepted)
    np.testing.assert_allclose(result.selected_policy_dual, result.qp.multipliers[-1])


@pytest.mark.parametrize(
    ("case", "policy", "expected_fast"),
    [
        ("nominal", True, True),
        ("single_policy", True, True),
        ("full_weight", True, True),
        ("multiple_faces", True, False),
        ("invalid_weight", True, False),
        ("nominal", False, True),
        ("multiple_faces", False, False),
    ],
)
def test_exact_fast_path_matches_exhaustive_qp(
    case: str, policy: bool, expected_fast: bool
) -> None:
    """A KKT shortcut is exact only when every additional face also permits its action."""
    from crazyflow.safety.da_plcbf.polytope_qp import project_affine_polytope
    from crazyflow.safety.da_plcbf.version_a_filter import _project_with_exact_fast_path

    dtype = jnp.float64
    matrix = jnp.concatenate((-jnp.eye(4), jnp.eye(4)))
    bound = jnp.ones(8)
    if policy:
        matrix = jnp.concatenate((matrix, jnp.asarray([[1.0, 0.5, 0.0, 0.0]])))
        bound = jnp.concatenate((bound, jnp.asarray([0.2])))
    nominal = jnp.asarray([0.7, 0.2, 0.0, 0.0], dtype=dtype)
    weight = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    if case == "nominal":
        nominal = jnp.zeros(4, dtype=dtype)
    elif case == "multiple_faces":
        nominal = jnp.asarray([3.0, -2.0, 0.5, 0.0], dtype=dtype)
    elif case == "invalid_weight":
        weight = weight.at[0].set(0.0)
    elif case == "full_weight":
        weight = jnp.diag(weight).at[0, 1].set(0.2).at[1, 0].set(0.2)
    config = VersionAFilterConfig(enforce_policy_barrier=policy)
    shortcut, used_fast = jax.jit(
        lambda point: _project_with_exact_fast_path(point, weight, matrix, bound, config)
    )(nominal)
    reference = project_affine_polytope(
        nominal,
        weight,
        matrix,
        bound,
        tolerance=config.qp_tolerance,
        rank_tolerance=config.qp_rank_tolerance,
    )
    assert bool(used_fast) == expected_fast
    assert bool(shortcut.feasible) == bool(reference.feasible)
    assert bool(shortcut.input_valid) == bool(reference.input_valid)
    np.testing.assert_allclose(
        shortcut.action, reference.action, atol=2e-6, rtol=2e-6, equal_nan=True
    )
    np.testing.assert_allclose(shortcut.objective, reference.objective, atol=3e-6, rtol=2e-6)
    if bool(shortcut.feasible):
        assert float(shortcut.primal_residual) <= config.kkt_tolerance
        assert float(shortcut.dual_residual) <= config.kkt_tolerance
        assert float(shortcut.stationarity_residual) <= config.kkt_tolerance
        assert float(shortcut.complementarity_residual) <= config.kkt_tolerance
        np.testing.assert_allclose(
            shortcut.multipliers, reference.multipliers, atol=2e-6, rtol=2e-6
        )


def test_explicit_time_derivative_enters_selected_policy_constraint(
    setup: tuple[dict[str, Any], VersionAModel, VersionAActuator, RigidBodySafetySet],
) -> None:
    params, model, actuator, safety = setup
    hover = jnp.asarray([params["mass"] * 9.81, 0.0, 0.0, 0.0], dtype=jnp.float64)
    library = _library(hover)._replace(time_derivatives=jnp.asarray([-0.7]))
    result = version_a_plcbf_filter(
        _state(),
        hover,
        jnp.ones(4),
        library,
        model,
        actuator,
        safety,
        VersionABarrierConfig(),
        VersionAFilterConfig(enforce_analytic_barriers=False),
    )
    np.testing.assert_allclose(result.selected_policy_bound, -9.81 - 0.7 + 2.0, atol=1e-12)
    assert bool(result.qp_accepted)
