from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.direct_wrench import (
    apply_control_affine,
    control_affine_identity_residual,
    quaternion_to_rotation_matrix,
)
from crazyflow.safety.da_plcbf.version_a_barriers import (
    RigidBodySafetySet,
    VersionABarrierConfig,
    VersionAModel,
    continuous_safety_halfspaces,
    dimensionless_safety_values,
    hard_finite_horizon_policy_certificate,
    safety_constraint_names,
    validated_control_affine_terms,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture(scope="module", autouse=True)
def _enable_x64() -> Iterator[None]:
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def cf21b() -> dict[str, Any]:
    return load_params("cf21B_500")


def _model(params: dict[str, Any], dtype: Any = jnp.float64) -> VersionAModel:
    inertia = jnp.asarray(params["J"], dtype=dtype)
    return VersionAModel(
        jnp.asarray(params["mass"], dtype=dtype),
        jnp.asarray(params["gravity_vec"], dtype=dtype),
        inertia,
        jnp.linalg.inv(inertia),
        jnp.asarray(params["drag_matrix"], dtype=dtype),
        jnp.zeros(3, dtype=dtype),
        jnp.zeros(3, dtype=dtype),
        jnp.zeros(3, dtype=dtype),
    )


def _state(
    *,
    pos: tuple[float, float, float] = (0.0, 0.0, 1.0),
    quat: np.ndarray | tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> jax.Array:
    return jnp.asarray((*pos, *quat, *vel, *ang_vel), dtype=jnp.float64)


def _safety(
    *,
    centers: np.ndarray | None = None,
    radii: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    lower: tuple[float, float, float] = (-5.0, -5.0, 0.0),
    upper: tuple[float, float, float] = (5.0, 5.0, 5.0),
    speed_max: float = 5.0,
    angular_rate_max: float = 10.0,
    tilt_max: float = 0.7,
) -> RigidBodySafetySet:
    if centers is None:
        centers = np.empty((0, 3))
    if radii is None:
        radii = np.empty((centers.shape[0],))
    if mask is None:
        mask = np.ones((centers.shape[0],), dtype=bool)
    return RigidBodySafetySet(
        jnp.asarray(centers, dtype=jnp.float64),
        jnp.asarray(radii, dtype=jnp.float64),
        jnp.asarray(mask, dtype=bool),
        jnp.asarray(lower, dtype=jnp.float64),
        jnp.asarray(upper, dtype=jnp.float64),
        jnp.asarray(speed_max, dtype=jnp.float64),
        jnp.asarray(angular_rate_max, dtype=jnp.float64),
        jnp.asarray(tilt_max, dtype=jnp.float64),
    )


@pytest.mark.unit
def test_dimensionless_values_have_explicit_sign_and_individual_physical_scales() -> None:
    state = _state(pos=(2.0, 0.0, 1.0), vel=(3.0, 0.0, 0.0))
    safety = _safety(
        centers=np.array([[0.0, 0.0, 1.0]]),
        radii=np.array([1.0]),
        lower=(-2.0, -4.0, 0.0),
        upper=(6.0, 4.0, 4.0),
    )

    result = dimensionless_safety_values(state, safety, VersionABarrierConfig())

    names = safety_constraint_names(1)
    values = dict(zip(names, np.asarray(result.values), strict=True))
    assert bool(result.input_valid)
    assert values["obstacle_0"] == pytest.approx(3.0)
    assert values["arena_x_lower"] == pytest.approx(0.5)
    assert values["arena_x_upper"] == pytest.approx(0.5)
    assert values["altitude_lower"] == pytest.approx(0.25)
    assert values["altitude_upper"] == pytest.approx(0.75)
    assert values["speed"] == pytest.approx(1.0 - 9.0 / 25.0)
    assert values["angular_rate"] == pytest.approx(1.0)
    assert values["tilt"] == pytest.approx(1.0)

    boundary = state.at[0].set(1.0)
    assert dimensionless_safety_values(boundary, safety, VersionABarrierConfig()).values[
        0
    ] == pytest.approx(0.0)
    violated = state.at[0].set(0.5)
    assert dimensionless_safety_values(violated, safety, VersionABarrierConfig()).values[0] < 0


@pytest.mark.unit
def test_hocbf_orders_units_and_hover_residuals_are_analytically_correct(
    cf21b: dict[str, Any],
) -> None:
    config = VersionABarrierConfig(
        position_alpha_1=2.0,
        position_alpha_2=2.0,
        speed_alpha=2.0,
        angular_rate_alpha=2.0,
        tilt_alpha_1=4.0,
        tilt_alpha_2=4.0,
    )
    state = _state()
    safety = _safety(centers=np.array([[0.0, 0.0, -1.0]]), radii=np.array([1.0]))
    barriers = continuous_safety_halfspaces(state, _model(cf21b), safety, config)
    hover = jnp.asarray([cf21b["mass"] * 9.81, 0.0, 0.0, 0.0], dtype=jnp.float64)
    residuals = barriers.upper_bound - barriers.matrix @ hover
    by_name = dict(zip(safety_constraint_names(1), np.asarray(residuals), strict=True))
    raw_by_name = dict(
        zip(safety_constraint_names(1), np.asarray(barriers.raw_values), strict=True)
    )

    assert bool(barriers.input_valid)
    assert bool(barriers.domain_valid)
    np.testing.assert_array_equal(
        barriers.relative_degrees, np.array([2, 2, 2, 2, 2, 2, 2, 1, 1, 2])
    )
    assert raw_by_name["obstacle_0"] == pytest.approx(3.0)
    assert raw_by_name["altitude_lower"] == pytest.approx(1.0)
    assert raw_by_name["speed"] == pytest.approx(25.0)
    assert raw_by_name["angular_rate"] == pytest.approx(100.0)
    assert by_name["obstacle_0"] == pytest.approx(12.0, abs=1e-10)
    assert by_name["altitude_lower"] == pytest.approx(4.0, abs=1e-10)
    assert by_name["altitude_upper"] == pytest.approx(16.0, abs=1e-10)
    assert by_name["speed"] == pytest.approx(50.0, abs=1e-10)
    assert by_name["angular_rate"] == pytest.approx(200.0, abs=1e-10)
    expected_tilt = 16.0 * (1.0 - np.cos(0.7))
    assert by_name["tilt"] == pytest.approx(expected_tilt, abs=1e-10)


def _finite_difference_hocbf_condition(
    state: jax.Array,
    wrench: jax.Array,
    model: VersionAModel,
    h_function: Callable[[jax.Array], jax.Array],
    alpha_1: float,
    alpha_2: float,
    epsilon: float = 2e-6,
) -> float:
    terms = validated_control_affine_terms(state, model).terms
    state_dot = apply_control_affine(terms, wrench)

    def psi(z: jax.Array) -> jax.Array:
        local_terms = validated_control_affine_terms(z, model).terms
        return jnp.dot(jax.grad(h_function)(z), local_terms.drift) + alpha_1 * h_function(z)

    plus = psi(state + epsilon * state_dot)
    minus = psi(state - epsilon * state_dot)
    derivative = (plus - minus) / (2.0 * epsilon)
    return float(derivative + alpha_2 * psi(state))


@pytest.mark.unit
def test_sphere_and_tilt_hocbf_lie_derivatives_match_independent_finite_difference(
    cf21b: dict[str, Any],
) -> None:
    quat = Rotation.from_euler("xyz", [0.18, -0.24, 0.31]).as_quat()
    state = _state(
        pos=(1.4, -0.3, 1.2), quat=quat, vel=(-0.2, 0.35, -0.1), ang_vel=(0.4, -0.25, 0.12)
    )
    safety = _safety(centers=np.array([[0.1, 0.2, 0.4]]), radii=np.array([0.45]))
    config = VersionABarrierConfig()
    model = _model(cf21b)
    wrench = jnp.asarray([0.42, 2.0e-4, -1.4e-4, 0.7e-5], dtype=jnp.float64)
    barriers = continuous_safety_halfspaces(state, model, safety, config)
    residuals = barriers.upper_bound - barriers.matrix @ wrench
    names = safety_constraint_names(1)

    center = safety.obstacle_centers[0]
    radius = safety.obstacle_radii[0]

    def sphere_h(z: jax.Array) -> jax.Array:
        relative = z[:3] - center
        return jnp.dot(relative, relative) - radius**2

    def tilt_h(z: jax.Array) -> jax.Array:
        return quaternion_to_rotation_matrix(z[3:7])[:, 2][2] - jnp.cos(safety.tilt_max_radians)

    sphere_expected = _finite_difference_hocbf_condition(
        state, wrench, model, sphere_h, config.position_alpha_1, config.position_alpha_2
    )
    tilt_expected = _finite_difference_hocbf_condition(
        state, wrench, model, tilt_h, config.tilt_alpha_1, config.tilt_alpha_2
    )
    sphere_index = names.index("obstacle_0")
    tilt_index = names.index("tilt")
    assert residuals[sphere_index] == pytest.approx(sphere_expected, rel=2e-7, abs=2e-7)
    assert residuals[tilt_index] == pytest.approx(tilt_expected, rel=3e-6, abs=3e-6)


@pytest.mark.unit
@pytest.mark.parametrize("seed", range(8))
def test_randomized_validated_terms_reconstruct_direct_wrench_dynamics(
    cf21b: dict[str, Any], seed: int
) -> None:
    generator = np.random.default_rng(seed)
    quat = Rotation.random(random_state=generator).as_quat()
    state = _state(
        pos=tuple(generator.normal(size=3)),
        quat=quat,
        vel=tuple(generator.normal(size=3)),
        ang_vel=tuple(generator.normal(size=3)),
    )
    wrench = jnp.asarray(generator.normal(size=4), dtype=jnp.float64)
    model = _model(cf21b)
    pos, quat_array, vel, ang_vel = state[:3], state[3:7], state[7:10], state[10:13]

    residual = control_affine_identity_residual(
        pos,
        quat_array,
        vel,
        ang_vel,
        wrench,
        mass=model.mass,
        gravity_vec=model.gravity_vec,
        J=model.inertia,
        J_inv=model.inertia_inv,
        drag_matrix=model.drag_matrix,
        wind_velocity=model.wind_velocity,
        external_force=model.external_force,
        external_torque=model.external_torque,
    )

    assert bool(validated_control_affine_terms(state, model).input_valid)
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=2e-11)


@pytest.mark.unit
def test_inertia_symmetry_validation_is_scale_aware_and_fails_closed(cf21b: dict[str, Any]) -> None:
    state = _state()
    model = _model(cf21b)
    tolerance = 2e-5
    scale = float(jnp.max(jnp.abs(model.inertia)))

    within = model.inertia.at[0, 1].add(0.5 * tolerance * scale)
    within_model = model._replace(inertia=within, inertia_inv=jnp.linalg.inv(within))
    outside = model.inertia.at[0, 1].add(2.0 * tolerance * scale)
    outside_model = model._replace(inertia=outside, inertia_inv=jnp.linalg.inv(outside))
    nonfinite = model.inertia.at[0, 1].set(jnp.nan)
    nonfinite_model = model._replace(inertia=nonfinite)

    assert bool(
        validated_control_affine_terms(state, within_model, model_tolerance=tolerance).input_valid
    )
    assert not bool(
        validated_control_affine_terms(state, outside_model, model_tolerance=tolerance).input_valid
    )
    assert not bool(
        validated_control_affine_terms(
            state, nonfinite_model, model_tolerance=tolerance
        ).input_valid
    )


@pytest.mark.unit
def test_hard_finite_horizon_value_has_unique_branch_finite_difference_gradient() -> None:
    safety = _safety(
        centers=np.array([[0.0, 0.0, 1.0]]),
        radii=np.array([1.0]),
        lower=(-100.0, -100.0, -100.0),
        upper=(100.0, 100.0, 100.0),
    )
    state = _state(pos=(1.2, 0.0, 1.0))
    config = VersionABarrierConfig(minimum_tie_tolerance=1e-8)

    def rollout(x: jax.Array) -> jax.Array:
        return jnp.stack((x, x.at[0].add(0.1)))

    certificate = hard_finite_horizon_policy_certificate(state, rollout, safety, config)
    epsilon = 1e-6
    plus = hard_finite_horizon_policy_certificate(
        state.at[0].add(epsilon), rollout, safety, config
    ).value
    minus = hard_finite_horizon_policy_certificate(
        state.at[0].add(-epsilon), rollout, safety, config
    ).value
    finite_difference = (plus - minus) / (2.0 * epsilon)

    assert certificate.value == pytest.approx(1.2**2 - 1.0, abs=1e-12)
    assert bool(certificate.input_valid)
    assert bool(certificate.gradient_valid)
    assert certificate.second_value_gap > config.minimum_tie_tolerance
    assert certificate.gradient[0] == pytest.approx(float(finite_difference), rel=2e-7)
    assert certificate.gradient[0] == pytest.approx(2.4, abs=1e-10)
    np.testing.assert_allclose(certificate.gradient[1:], 0.0, atol=1e-12)


@pytest.mark.unit
def test_tied_hard_minimum_remains_reportable_but_cannot_be_single_halfspace_certificate() -> None:
    state = _state(pos=(0.0, 0.0, 0.0))
    safety = _safety(lower=(-1.0, -1.0, -1.0), upper=(1.0, 1.0, 1.0))
    certificate = hard_finite_horizon_policy_certificate(
        state, lambda x: x[None, :], safety, VersionABarrierConfig()
    )

    assert bool(certificate.input_valid)
    assert np.isfinite(certificate.value)
    assert certificate.second_value_gap == pytest.approx(0.0)
    assert not bool(certificate.gradient_valid)


@pytest.mark.unit
def test_rollout_that_omits_current_state_cannot_supply_a_runtime_certificate() -> None:
    state = _state(pos=(1.2, 0.0, 1.0))
    safety = _safety(
        centers=np.array([[0.0, 0.0, 1.0]]),
        radii=np.array([1.0]),
        lower=(-100.0, -100.0, -100.0),
        upper=(100.0, 100.0, 100.0),
    )
    certificate = hard_finite_horizon_policy_certificate(
        state, lambda x: x.at[0].add(0.2)[None, :], safety, VersionABarrierConfig()
    )

    assert not bool(certificate.input_valid)
    assert not bool(certificate.gradient_valid)


@pytest.mark.unit
def test_masked_nonfinite_obstacle_padding_is_ignored_but_real_nonfinite_data_fails_closed(
    cf21b: dict[str, Any],
) -> None:
    state = _state()
    padded = _safety(
        centers=np.array([[np.nan, np.inf, np.nan]]),
        radii=np.array([np.nan]),
        mask=np.array([False]),
    )
    active = padded._replace(obstacle_mask=jnp.asarray([True]))

    padded_values = dimensionless_safety_values(state, padded, VersionABarrierConfig())
    padded_barriers = continuous_safety_halfspaces(
        state, _model(cf21b), padded, VersionABarrierConfig()
    )
    active_values = dimensionless_safety_values(state, active, VersionABarrierConfig())
    active_barriers = continuous_safety_halfspaces(
        state, _model(cf21b), active, VersionABarrierConfig()
    )

    assert bool(padded_values.input_valid)
    assert bool(padded_barriers.input_valid)
    assert np.isinf(padded_values.values[0])
    assert not bool(active_values.input_valid)
    assert not bool(active_barriers.input_valid)


@pytest.mark.unit
def test_invalid_state_or_inconsistent_inertia_inverse_is_not_a_valid_barrier_input(
    cf21b: dict[str, Any],
) -> None:
    state = _state().at[0].set(jnp.nan)
    model = _model(cf21b)
    bad_model = model._replace(inertia_inv=2.0 * model.inertia_inv)
    safety = _safety()

    state_result = continuous_safety_halfspaces(state, model, safety, VersionABarrierConfig())
    model_result = continuous_safety_halfspaces(
        _state(), bad_model, safety, VersionABarrierConfig()
    )

    assert not bool(state_result.input_valid)
    assert not bool(state_result.domain_valid)
    assert not bool(model_result.input_valid)
    assert not bool(model_result.domain_valid)

    nonunit_quaternion = _state().at[6].set(1.01)
    quaternion_result = continuous_safety_halfspaces(
        nonunit_quaternion, model, safety, VersionABarrierConfig()
    )
    assert not bool(quaternion_result.input_valid)
    assert not bool(quaternion_result.domain_valid)

    with pytest.raises(ValueError, match="floating-point dtype"):
        dimensionless_safety_values(
            jnp.arange(13, dtype=jnp.int32), safety, VersionABarrierConfig()
        )
