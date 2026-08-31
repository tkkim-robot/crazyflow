from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.capsules import (
    CapsuleBarrierConfig,
    CapsuleObstacleSet,
    continuous_capsule_halfspaces,
    point_capsule_dimensionless_values,
    quad_capsule_trajectory_values,
    swept_segment_capsule_dimensionless_values,
    validate_capsules,
)
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig, VersionAModel


def _capsules() -> CapsuleObstacleSet:
    return CapsuleObstacleSet(
        segment_start=jnp.array([[0.0, -0.5, 0.0], [2.0, 0.0, 0.0], [jnp.nan, 0.0, 0.0]]),
        segment_end=jnp.array([[0.0, 0.5, 0.0], [2.0, 0.0, 0.0], [jnp.nan, 0.0, 0.0]]),
        radii=jnp.array([0.25, 0.5, jnp.nan]),
        mask=jnp.array([True, True, False]),
    )


def _model() -> VersionAModel:
    parameters = load_params("cf21B_500")
    inertia = jnp.asarray(parameters["J"])
    return VersionAModel(
        mass=jnp.asarray(parameters["mass"]),
        gravity_vec=jnp.asarray(parameters["gravity_vec"]),
        inertia=inertia,
        inertia_inv=jnp.linalg.inv(inertia),
        drag_matrix=jnp.asarray(parameters["drag_matrix"]),
        wind_velocity=jnp.zeros(3),
        external_force=jnp.zeros(3),
        external_torque=jnp.zeros(3),
    )


def test_point_capsule_values_are_exact_dimensionless_and_masked() -> None:
    result = point_capsule_dimensionless_values(jnp.array([0.5, 0.0, 0.0]), _capsules())

    assert bool(result.input_valid)
    np.testing.assert_allclose(result.values[:2], np.array([3.0, 8.0]), atol=1e-6)
    assert np.isinf(float(result.values[2]))
    np.testing.assert_allclose(result.closest_fraction[:2], np.array([0.5, 0.0]))


def test_degenerate_capsule_reduces_exactly_to_sphere() -> None:
    capsule = CapsuleObstacleSet(
        segment_start=jnp.array([[1.0, 2.0, 3.0]]),
        segment_end=jnp.array([[1.0, 2.0, 3.0]]),
        radii=jnp.array([0.4]),
        mask=jnp.array([True]),
    )
    boundary = point_capsule_dimensionless_values(jnp.array([1.4, 2.0, 3.0]), capsule)

    np.testing.assert_allclose(boundary.values, 0.0, atol=2e-6)
    np.testing.assert_array_equal(boundary.closest_fraction, np.array([0.0]))


def test_swept_capsule_value_detects_between_node_collision() -> None:
    capsule = CapsuleObstacleSet(
        segment_start=jnp.array([[0.0, -0.5, 0.0]]),
        segment_end=jnp.array([[0.0, 0.5, 0.0]]),
        radii=jnp.array([0.2]),
        mask=jnp.array([True]),
    )
    start = jnp.array([-1.0, 0.0, 0.0])
    end = jnp.array([1.0, 0.0, 0.0])
    node_values = jnp.stack(
        (
            point_capsule_dimensionless_values(start, capsule).values,
            point_capsule_dimensionless_values(end, capsule).values,
        )
    )
    swept = swept_segment_capsule_dimensionless_values(start, end, capsule)

    assert np.all(np.asarray(node_values) > 0)
    np.testing.assert_allclose(swept.values, -1.0, atol=1e-6)
    np.testing.assert_allclose(swept.closest_fraction, 0.5, atol=1e-6)


def test_float32_short_perpendicular_segments_cannot_hide_an_interior_collision() -> None:
    """Regress the independent crossing geometry that a length**4 scale floor misclassified."""
    capsule = CapsuleObstacleSet(
        segment_start=jnp.array([[0.0, -0.05, 0.0]], dtype=jnp.float32),
        segment_end=jnp.array([[0.0, 0.05, 0.0]], dtype=jnp.float32),
        radii=jnp.array([0.005], dtype=jnp.float32),
        mask=jnp.array([True]),
    )
    start = jnp.array([-0.01, 0.0, 0.0], dtype=jnp.float32)
    end = jnp.array([0.01, 0.0, 0.0], dtype=jnp.float32)

    result = swept_segment_capsule_dimensionless_values(start, end, capsule)

    # The two independently specified line segments cross at their midpoints, hence their axis
    # distance is exactly zero and the dimensionless capsule value is (0-r**2)/r**2 = -1.
    assert bool(result.input_valid)
    np.testing.assert_allclose(result.values, -1.0, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(result.closest_fraction, 0.5, rtol=0.0, atol=2e-6)


@pytest.mark.parametrize("scale", (1e-2, 1.0, 1e2))
def test_float32_swept_capsule_crossing_is_scale_invariant(scale: float) -> None:
    capsule = CapsuleObstacleSet(
        segment_start=jnp.array([[0.0, -0.05, 0.0]], dtype=jnp.float32) * scale,
        segment_end=jnp.array([[0.0, 0.05, 0.0]], dtype=jnp.float32) * scale,
        radii=jnp.array([0.005 * scale], dtype=jnp.float32),
        mask=jnp.array([True]),
    )
    result = swept_segment_capsule_dimensionless_values(
        jnp.array([-0.01, 0.0, 0.0], dtype=jnp.float32) * scale,
        jnp.array([0.01, 0.0, 0.0], dtype=jnp.float32) * scale,
        capsule,
    )

    assert bool(result.input_valid)
    np.testing.assert_allclose(result.values, -1.0, rtol=0.0, atol=3e-6)
    np.testing.assert_allclose(result.closest_fraction, 0.5, rtol=0.0, atol=3e-6)


def test_unresolved_near_parallel_segments_fail_closed_with_finite_diagnostics() -> None:
    capsule = CapsuleObstacleSet(
        segment_start=jnp.array([[0.0, 0.01, 0.0]], dtype=jnp.float32),
        segment_end=jnp.array([[0.1, 0.01001, 0.0]], dtype=jnp.float32),
        radii=jnp.array([0.001], dtype=jnp.float32),
        mask=jnp.array([True]),
    )
    result = swept_segment_capsule_dimensionless_values(
        jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([0.1, 0.0, 0.0], dtype=jnp.float32),
        capsule,
    )

    assert np.all(np.isfinite(np.asarray(result.values)))
    assert np.all(np.isfinite(np.asarray(result.closest_fraction)))
    assert not bool(result.input_valid)


def test_parallel_overlapping_segments_have_zero_axis_distance() -> None:
    capsule = CapsuleObstacleSet(
        segment_start=jnp.array([[-0.5, 0.1, 0.0]]),
        segment_end=jnp.array([[0.5, 0.1, 0.0]]),
        radii=jnp.array([0.1]),
        mask=jnp.array([True]),
    )
    result = swept_segment_capsule_dimensionless_values(
        jnp.array([-1.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0]), capsule
    )

    assert bool(result.input_valid)
    np.testing.assert_allclose(result.values, 0.0, atol=2e-6)


def test_capsule_geometry_batches_and_is_differentiable_away_from_ties() -> None:
    base = _capsules()
    batch = CapsuleObstacleSet(
        segment_start=jnp.stack((base.segment_start, base.segment_start + 0.2)),
        segment_end=jnp.stack((base.segment_end, base.segment_end + 0.2)),
        radii=jnp.stack((base.radii, base.radii)),
        mask=jnp.stack((base.mask, base.mask)),
    )
    positions = jnp.array([[0.6, 0.0, 0.0], [0.8, 0.2, 0.2]])
    result = point_capsule_dimensionless_values(positions, batch)
    gradient = jax.grad(
        lambda position: point_capsule_dimensionless_values(position, base).values[0]
    )(positions[0])

    assert result.values.shape == (2, 3)
    assert bool(result.input_valid)
    assert np.all(np.isfinite(np.asarray(gradient)))
    np.testing.assert_allclose(gradient, np.array([19.2, 0.0, 0.0]), atol=2e-5)


def test_invalid_enabled_capsule_fails_closed_but_masked_padding_is_ignored() -> None:
    assert bool(validate_capsules(_capsules()))
    invalid = _capsules()._replace(radii=jnp.array([0.25, -0.5, jnp.nan]))
    result = point_capsule_dimensionless_values(jnp.zeros(3), invalid)

    assert not bool(result.input_valid)


def test_continuous_capsule_hocbf_matches_closed_form_vertical_residual() -> None:
    capsule = CapsuleObstacleSet(
        segment_start=jnp.array([[0.0, -0.5, 0.0]]),
        segment_end=jnp.array([[0.0, 0.5, 0.0]]),
        radii=jnp.array([0.2]),
        mask=jnp.array([True]),
    )
    model = _model()
    state = jnp.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    config = VersionABarrierConfig(position_alpha_1=2.0, position_alpha_2=3.0)
    result = continuous_capsule_halfspaces(state, model, capsule, config)
    hover = jnp.array([-model.mass * model.gravity_vec[2], 0.0, 0.0, 0.0])

    expected_h = 0.5**2 - 0.2**2
    expected_residual = config.position_alpha_1 * config.position_alpha_2 * expected_h
    actual_residual = result.upper_bound[0] - result.matrix[0] @ hover
    assert bool(result.input_valid)
    assert bool(result.domain_valid)
    np.testing.assert_allclose(result.raw_values, expected_h, atol=1e-6)
    np.testing.assert_allclose(result.first_order_values, 2.0 * expected_h, atol=1e-6)
    np.testing.assert_allclose(actual_residual, expected_residual, atol=2e-5)


def test_continuous_capsule_hocbf_fails_closed_at_projection_hessian_seam() -> None:
    capsule = CapsuleObstacleSet(
        segment_start=jnp.array([[0.0, 0.0, 0.0]]),
        segment_end=jnp.array([[1.0, 0.0, 0.0]]),
        radii=jnp.array([0.2]),
        mask=jnp.array([True]),
    )
    state = jnp.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = continuous_capsule_halfspaces(
        state,
        _model(),
        capsule,
        VersionABarrierConfig(),
        CapsuleBarrierConfig(projection_seam_tolerance=1e-5),
    )

    assert not bool(result.differentiable_region[0])
    assert not bool(result.input_valid)
    assert not bool(result.domain_valid)


def test_quad_capsule_values_preserve_policy_scenario_axes_and_swept_hard_minimum() -> None:
    capsules = CapsuleObstacleSet(
        segment_start=jnp.array([[[0.0, -0.5, 1.0]], [[0.0, 1.5, 1.0]]]),
        segment_end=jnp.array([[[0.0, 0.5, 1.0]], [[0.0, 2.5, 1.0]]]),
        radii=jnp.full((2, 1), 0.2),
        mask=jnp.ones((2, 1), dtype=bool),
    )
    identity = jnp.array([0.0, 0.0, 0.0, 1.0])
    zero_tail = jnp.zeros(6)
    first = jnp.concatenate((jnp.array([-1.0, 0.0, 1.0]), identity, zero_tail))
    second = first.at[0].set(1.0)
    scenario_states = jnp.stack((jnp.stack((first, second)), jnp.stack((first, second))))
    states = jnp.stack((scenario_states, scenario_states.at[:, :, 1].add(0.3)))
    values = quad_capsule_trajectory_values(states, capsules, softmin_beta=30.0)

    assert values.node_values.shape == (2, 2, 2, 1)
    assert values.segment_values.shape == (2, 2, 1, 1)
    np.testing.assert_allclose(values.hard_policy_margins[0, 0], -1.0, atol=1e-6)
    assert float(values.hard_policy_margins[0, 1]) > 0
    assert np.all(
        np.asarray(values.smooth_policy_margins) <= np.asarray(values.hard_policy_margins) + 1e-6
    )


@pytest.mark.parametrize("clearance", (-1.0, np.nan, np.inf))
def test_invalid_clearance_is_rejected(clearance: float) -> None:
    with pytest.raises(ValueError, match="clearance"):
        validate_capsules(_capsules(), clearance=clearance)
