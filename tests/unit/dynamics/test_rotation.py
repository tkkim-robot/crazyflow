"""Testing the selfimplemented rotations against scipy rotations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import array_api_strict as xp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import crazyflow.dynamics.utils.rotation as rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray

tol = 1e-6  # Since Jax by default works with 32 bit, the precision is worse


def create_uniform_quats(N: int = 1000, scale: float = 10) -> NDArray:
    """Creates an (n, 4) list with random quaternions."""
    # larger range because the function should be able to handle wrong length quaternions
    return np.random.uniform(-scale, scale, size=(N, 4))


def create_uniform_ang_vel(N: int = 1000, scale: float = 10) -> NDArray:
    """Creates an (n, 4) list with random quaternions."""
    # larger range because the function should be able to handle wrong length quaternions
    return np.random.uniform(-scale, scale, size=(N, 3))


@pytest.mark.unit
def test_ang_vel2quat_dot_uses_xyzw_body_rate_convention() -> None:
    quat = xp.asarray([0.0, 0.0, 0.0, 1.0])
    body_rate = xp.asarray([0.4, -0.2, 0.8])

    derivative = rotation.ang_vel2quat_dot(quat, body_rate)

    assert np.allclose(derivative, np.array([0.2, -0.1, 0.4, 0.0]))


@pytest.mark.unit
def test_ang_vel2quat_dot_matches_right_composed_scipy_finite_difference_and_batches() -> None:
    generator = np.random.default_rng(20260830)
    quats = R.random(8, random_state=generator).as_quat()
    body_rates = generator.normal(size=(8, 3))
    dt = 1e-7
    expected = []
    for quat, body_rate in zip(quats, body_rates, strict=True):
        following = (R.from_quat(quat) * R.from_rotvec(body_rate * dt)).as_quat()
        if np.dot(following, quat) < 0:
            following = -following
        expected.append((following - quat) / dt)
    expected = np.asarray(expected)

    batched = rotation.ang_vel2quat_dot(xp.asarray(quats), xp.asarray(body_rates))

    np.testing.assert_allclose(batched, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.sum(np.asarray(batched) * quats, axis=-1), 0.0, atol=1e-12)
    for index in range(quats.shape[0]):
        single = rotation.ang_vel2quat_dot(xp.asarray(quats[index]), xp.asarray(body_rates[index]))
        np.testing.assert_allclose(single, np.asarray(batched)[index], rtol=0.0, atol=1e-12)


@pytest.mark.unit
def test_symbolic_ang_vel2quat_dot_matches_numeric_xyzw_implementation() -> None:
    generator = np.random.default_rng(260830)
    quats = R.random(8, random_state=generator).as_quat()
    body_rates = generator.normal(size=(8, 3))

    numeric = rotation.ang_vel2quat_dot(quats, body_rates)
    symbolic = np.stack(
        [
            np.asarray(rotation.cs_ang_vel2quat_dot(quat, body_rate)).reshape(4)
            for quat, body_rate in zip(quats, body_rates, strict=True)
        ]
    )

    np.testing.assert_allclose(symbolic, numeric, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_ang_vel2rpy_rates_two_way():
    quats = xp.asarray(create_uniform_quats())
    ang_vels = xp.asarray(create_uniform_ang_vel())

    rpy_rates_two_way = rotation.ang_vel2rpy_rates(quats, ang_vels)
    ang_vels_two_way = rotation.rpy_rates2ang_vel(quats, rpy_rates_two_way)
    assert np.allclose(ang_vels, ang_vels_two_way), "Two way transform results are off."


@pytest.mark.unit
def test_ang_vel2rpy_rates_batching():
    quats = xp.asarray(create_uniform_quats())
    ang_vels = xp.asarray(create_uniform_ang_vel())

    # Calculate batched version
    rpy_rates_batched = rotation.ang_vel2rpy_rates(quats, ang_vels)

    # Compare to non-batched version
    for i in range(ang_vels.shape[0]):
        rpy_rates_non_batched = rotation.ang_vel2rpy_rates(quats[i, ...], ang_vels[i, ...])
        assert np.allclose(rpy_rates_non_batched, rpy_rates_batched[i, ...]), (
            "Batched and non-batched results differ."
        )


@pytest.mark.unit
def test_rpy_rates2ang_vel_batching():
    quats = xp.asarray(create_uniform_quats())
    rpy_rates = xp.asarray(create_uniform_ang_vel())

    # Calculate batched version
    ang_vel_batched = rotation.rpy_rates2ang_vel(quats, rpy_rates)

    # Compare to non-batched version
    for i in range(rpy_rates.shape[0]):
        ang_vel_non_batched = rotation.rpy_rates2ang_vel(quats[i, ...], rpy_rates[i, ...])
        assert np.allclose(ang_vel_non_batched, ang_vel_batched[i, ...]), (
            "Batched and non-batched results differ."
        )


@pytest.mark.unit
def test_ang_vel2rpy_rates_symbolic():
    quats = np.array(create_uniform_quats())
    ang_vels = np.array(create_uniform_ang_vel())

    # Calculate numeric version
    rpy_rates = rotation.ang_vel2rpy_rates(quats, ang_vels)

    # Compare to casadi implementation
    for i in range(len(ang_vels)):
        rpy_rates_cs = np.array(rotation.cs_ang_vel2rpy_rates(quats[i], ang_vels[i])).flatten()
        assert np.allclose(rpy_rates_cs, rpy_rates[i]), "Symbolic and numeric results differ."


@pytest.mark.unit
def test_rpy_rates2ang_vel_symbolic():
    quats = np.array(create_uniform_quats())
    rpy_rates = np.array(create_uniform_ang_vel())

    # Calculate numeric version
    ang_vels = rotation.rpy_rates2ang_vel(quats, rpy_rates)

    # Compare to casadi implementation
    for i in range(len(rpy_rates)):
        ang_vel_cs = np.array(rotation.cs_rpy_rates2ang_vel(quats[i], rpy_rates[i])).flatten()
        assert np.allclose(ang_vel_cs, ang_vels[i]), "Symbolic and numeric results differ."


@pytest.mark.unit
def test_ang_vel_deriv2rpy_rates_deriv_two_way():
    quats = xp.asarray(create_uniform_quats())
    ang_vels = xp.asarray(create_uniform_ang_vel())
    ang_vels_deriv = xp.asarray(create_uniform_ang_vel())
    rpy_rates = rotation.ang_vel2rpy_rates(quats, ang_vels)

    rpy_rates_deriv_two_way = rotation.ang_vel_deriv2rpy_rates_deriv(
        quats, ang_vels, ang_vels_deriv
    )
    ang_vels_deriv_two_way = rotation.rpy_rates_deriv2ang_vel_deriv(
        quats, rpy_rates, rpy_rates_deriv_two_way
    )
    assert np.allclose(ang_vels_deriv, ang_vels_deriv_two_way), "Two way transform results are off."


@pytest.mark.unit
def test_ang_vel_deriv2rpy_rates_deriv_batching():
    quats = xp.asarray(create_uniform_quats())
    ang_vels = xp.asarray(create_uniform_ang_vel())
    ang_vels_deriv = xp.asarray(create_uniform_ang_vel())

    # Calculate batched version
    rpy_rates_deriv_batched = rotation.ang_vel_deriv2rpy_rates_deriv(
        quats, ang_vels, ang_vels_deriv
    )

    # Compare to non-batched version
    for i in range(ang_vels.shape[0]):
        rpy_rates_deriv_non_batched = rotation.ang_vel_deriv2rpy_rates_deriv(
            quats[i, ...], ang_vels[i, ...], ang_vels_deriv[i, ...]
        )
        assert np.allclose(rpy_rates_deriv_non_batched, rpy_rates_deriv_batched[i, ...]), (
            "Batched and non-batched results differ."
        )


@pytest.mark.unit
def test_rpy_rates_deriv2ang_vel_deriv_batching():
    quats = np.array(create_uniform_quats())
    rpy_rates = np.array(create_uniform_ang_vel())
    rpy_rates_deriv = np.array(create_uniform_ang_vel())

    # Calculate batched version
    ang_vels_deriv_batched = rotation.rpy_rates_deriv2ang_vel_deriv(
        quats, rpy_rates, rpy_rates_deriv
    )

    # Compare to non-batched version
    for i in range(rpy_rates.shape[0]):
        ang_vels_deriv_non_batched = rotation.rpy_rates_deriv2ang_vel_deriv(
            quats[i, ...], rpy_rates[i, ...], rpy_rates_deriv[i, ...]
        )
        assert np.allclose(ang_vels_deriv_non_batched, ang_vels_deriv_batched[i, ...]), (
            "Batched and non-batched results differ."
        )


@pytest.mark.unit
def test_ang_vel_deriv2rpy_rates_deriv_symbolic():
    quats = np.array(create_uniform_quats())
    ang_vels = np.array(create_uniform_ang_vel())
    ang_vels_deriv = np.array(create_uniform_ang_vel())

    # Calculate batched version
    rpy_rates_deriv = rotation.ang_vel_deriv2rpy_rates_deriv(quats, ang_vels, ang_vels_deriv)

    # Compare to casadi implementation
    for i in range(ang_vels.shape[0]):
        # TODO test against casadi implementation
        rpy_rates_deriv_cs = np.array(
            rotation.cs_ang_vel_deriv2rpy_rates_deriv(quats[i], ang_vels[i], ang_vels_deriv[i])
        ).flatten()
        assert np.allclose(rpy_rates_deriv_cs, rpy_rates_deriv[i]), (
            "Symbolic and numeric results differ."
        )


@pytest.mark.unit
def test_rpy_rates_deriv2ang_vel_deriv_symbolic():
    quats = np.array(create_uniform_quats())
    rpy_rates = np.array(create_uniform_ang_vel())
    rpy_rates_deriv = np.array(create_uniform_ang_vel())

    # Calculate batched version
    ang_vels_deriv = rotation.rpy_rates_deriv2ang_vel_deriv(quats, rpy_rates, rpy_rates_deriv)

    # Compare to casadi implementation
    for i in range(len(rpy_rates)):
        # TODO test against casadi implementation
        ang_vels_deriv_cs = np.array(
            rotation.cs_rpy_rates_deriv2ang_vel_deriv(quats[i], rpy_rates[i], rpy_rates_deriv[i])
        ).flatten()
        assert np.allclose(ang_vels_deriv_cs, ang_vels_deriv[i]), (
            "Symbolic and numeric results differ."
        )


# TODO test ang_vel2rpy_rates (and deriv) conversions with jp and np arrays


@pytest.mark.unit
def test_quat2matrix_symbolic():
    quats = np.array(create_uniform_quats())
    for i, q in enumerate(quats):
        mat_scipy = R.from_quat(q).as_matrix()

        # compare casadi/symbolic implementation to scipy
        mat_cs = np.array(rotation.cs_quat2matrix_func(q))
        assert np.allclose(mat_cs, mat_scipy, atol=tol), "Symbolic quat->matrix differs from scipy."


@pytest.mark.unit
def test_rpy2matrix_symbolic():
    rpys = np.array(create_uniform_ang_vel())
    for i, rpy in enumerate(rpys):
        rpy = rpys[i]
        mat_scipy = R.from_euler("xyz", rpy).as_matrix()

        # compare casadi/symbolic implementation to scipy
        mat_cs = np.array(rotation.cs_rpy2matrix_func(rpy))
        assert np.allclose(mat_cs, mat_scipy, atol=tol), "Symbolic rpy->matrix differs from scipy."
