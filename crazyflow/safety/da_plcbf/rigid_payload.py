"""Known, centered rigid payload parameters for a prescribed attachment event.

The payload center coincides with the modeled rigid body's center of mass. This deliberately
avoids claiming an off-center or tethered model: rotor moment arms and allocation remain exactly
unchanged, while mass, box inertia, and the enclosing collision radius change consistently.
Attachment is a supplied parameter switch, not contact-resolved pickup or momentum exchange.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


@dataclass(frozen=True, slots=True)
class CenteredRigidPayload:
    """A rigid box with specified mass and half extents about the existing center of mass."""

    mass: float
    half_extents: tuple[float, float, float] = (0.025, 0.025, 0.025)

    def validate(self) -> None:
        if not math.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("payload mass must be positive finite")
        if len(self.half_extents) != 3 or not all(
            math.isfinite(x) and x > 0 for x in self.half_extents
        ):
            raise ValueError("payload half extents must be three positive finite lengths")

    def apply(self, model: VersionAModel) -> VersionAModel:
        """Return the same point-model type with the combined mass and exact centered inertia."""
        self.validate()
        original_inertia = np.asarray(model.inertia)
        if (
            not np.isfinite(np.asarray(model.mass)).all()
            or np.asarray(model.mass).size != 1
            or float(np.asarray(model.mass).reshape(())) <= 0
            or original_inertia.shape != (3, 3)
            or not np.all(np.isfinite(original_inertia))
            or not np.allclose(original_inertia, original_inertia.T, atol=1e-10, rtol=1e-6)
            or np.min(np.linalg.eigvalsh(original_inertia)) <= 0
        ):
            raise ValueError(
                "the original rigid body requires positive mass and symmetric SPD inertia"
            )
        x, y, z = self.half_extents
        box_inertia = (self.mass / 3.0) * jnp.diag(
            jnp.asarray([y * y + z * z, x * x + z * z, x * x + y * y], dtype=model.inertia.dtype)
        )
        inertia = model.inertia + box_inertia
        return model._replace(
            mass=model.mass + self.mass, inertia=inertia, inertia_inv=jnp.linalg.inv(inertia)
        )

    def enclosing_radius(self, drone_radius: float) -> float:
        """Enclose both the original drone sphere and the entire centered box."""
        self.validate()
        if not math.isfinite(drone_radius) or drone_radius < 0:
            raise ValueError("drone_radius must be nonnegative and finite")
        return max(drone_radius, float(np.linalg.norm(self.half_extents)))


def hover_authority(
    model: VersionAModel, actuator: VersionAActuator
) -> dict[str, float | bool | str]:
    """Check upright vertical authority at rest; horizontal trim/evasion is a separate test."""
    gravity_z = float(np.asarray(model.gravity_vec)[2])
    drag_at_rest = -np.asarray(model.drag_matrix) @ np.asarray(model.wind_velocity)
    hover = (
        -float(np.asarray(model.mass)) * gravity_z
        - float(np.asarray(model.external_force)[2])
        - float(drag_at_rest[2])
    )
    maximum = float(np.sum(np.broadcast_to(np.asarray(actuator.thrust_max), (4,))))
    minimum = float(np.sum(np.broadcast_to(np.asarray(actuator.thrust_min), (4,))))
    return {
        "scope": "upright collective vertical authority only; excludes horizontal trim and evasion",
        "required_hover_thrust_N": hover,
        "minimum_collective_N": minimum,
        "maximum_collective_N": maximum,
        "hover_feasible": minimum <= hover <= maximum,
        "collective_reserve_fraction": (maximum - hover) / maximum,
    }
