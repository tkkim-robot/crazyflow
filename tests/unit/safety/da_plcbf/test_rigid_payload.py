from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.direct_wrench import wrench_to_motor_forces
from crazyflow.safety.da_plcbf.online_constant_wind import build_cf21b_version_a_resources
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.rigid_payload import CenteredRigidPayload, hover_authority


def test_centered_box_payload_updates_mass_and_inertia_without_changing_allocation() -> None:
    resources = build_cf21b_version_a_resources()
    original_actuator = [np.asarray(x).copy() for x in jax.tree.leaves(resources.actuator)]
    payload = CenteredRigidPayload(mass=0.006, half_extents=(0.02, 0.03, 0.04))
    combined = payload.apply(resources.model)
    expected_delta = np.diag([0.03**2 + 0.04**2, 0.02**2 + 0.04**2, 0.02**2 + 0.03**2]) * 0.006 / 3
    np.testing.assert_allclose(combined.mass, resources.model.mass + 0.006, atol=1e-9)
    np.testing.assert_allclose(
        combined.inertia, resources.model.inertia + expected_delta, atol=2e-12
    )
    np.testing.assert_allclose(combined.inertia @ combined.inertia_inv, np.eye(3), atol=1e-6)
    assert np.all(np.linalg.eigvalsh(np.asarray(combined.inertia)) > 0)
    for field in (
        "drag_matrix",
        "wind_velocity",
        "external_force",
        "external_torque",
        "gravity_vec",
    ):
        np.testing.assert_array_equal(getattr(combined, field), getattr(resources.model, field))
    for before, after in zip(original_actuator, jax.tree.leaves(resources.actuator), strict=True):
        np.testing.assert_array_equal(before, after)

    # More hover force is required, but the same physical motor allocation remains in effect.
    hover = jnp.asarray([-combined.mass * combined.gravity_vec[2], 0.0, 0.0, 0.0])
    motors = wrench_to_motor_forces(
        hover,
        L=resources.actuator.arm_length,
        thrust2torque=resources.actuator.thrust_to_torque,
        mixing_matrix=resources.actuator.mixing_matrix,
    )
    np.testing.assert_allclose(motors, jnp.full(4, hover[0] / 4), atol=1e-7)
    initial = jnp.asarray([0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    following = direct_wrench_symplectic_step(initial, hover, combined, 0.02)
    np.testing.assert_allclose(following, initial, atol=1e-7)
    assert hover_authority(combined, resources.actuator)["hover_feasible"] is True
    assert (
        hover_authority(CenteredRigidPayload(0.2).apply(resources.model), resources.actuator)[
            "hover_feasible"
        ]
        is False
    )


def test_payload_collision_sphere_encloses_every_box_corner_and_original_body() -> None:
    payload = CenteredRigidPayload(0.006, (0.06, 0.025, 0.02))
    radius = payload.enclosing_radius(0.05)
    corners = np.asarray(list(itertools.product(*[[-x, x] for x in payload.half_extents])))
    assert np.all(np.linalg.norm(corners, axis=1) <= radius + 1e-12)
    assert radius >= 0.05
    assert payload.enclosing_radius(0.10) == 0.10
    with pytest.raises(ValueError):
        payload.enclosing_radius(-0.01)
