"""JAX-compatible data structures used by DA-PLCBF."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from flax.struct import dataclass

if TYPE_CHECKING:
    from jax import Array


@dataclass
class CircleScenarioBatch:
    """Fixed-shape planar safety scenarios.

    Every array has a leading scenario dimension ``B``. Obstacles use a fixed padded dimension
    ``O``; ``obstacle_mask`` marks real entries. Bounds and speed are hard operational constraints.

    Attributes:
        obstacle_centers: Circle centres with shape ``(B, O, D)``.
        obstacle_radii: Circle radii with shape ``(B, O)``.
        obstacle_mask: Valid-obstacle mask with shape ``(B, O)``.
        arena_lower: Lower position bounds with shape ``(B, D)``.
        arena_upper: Upper position bounds with shape ``(B, D)``.
        speed_limit: Maximum speed with shape ``(B,)``.
    """

    obstacle_centers: Array
    obstacle_radii: Array
    obstacle_mask: Array
    arena_lower: Array
    arena_upper: Array
    speed_limit: Array

    def validate(self) -> None:
        """Validate fixed shapes and finite values at a non-jitted input boundary.

        Masked obstacle padding may contain arbitrary values, including NaNs; real obstacles may
        not. This method intentionally performs host checks and should run before, not inside, a
        jitted rollout.
        """
        centers = np.asarray(self.obstacle_centers)
        radii = np.asarray(self.obstacle_radii)
        mask = np.asarray(self.obstacle_mask)
        lower = np.asarray(self.arena_lower)
        upper = np.asarray(self.arena_upper)
        speed = np.asarray(self.speed_limit)
        if centers.ndim != 3:
            raise ValueError("obstacle_centers must have shape (B, O, D)")
        batch_size, n_obstacles, dimension = centers.shape
        if batch_size <= 0 or dimension <= 0:
            raise ValueError("scenario batch and spatial dimensions must be positive")
        if radii.shape != (batch_size, n_obstacles):
            raise ValueError("obstacle_radii must have shape (B, O)")
        if mask.shape != (batch_size, n_obstacles) or mask.dtype != np.bool_:
            raise ValueError("obstacle_mask must be a boolean array with shape (B, O)")
        if lower.shape != (batch_size, dimension) or upper.shape != (batch_size, dimension):
            raise ValueError("arena bounds must have shape (B, D)")
        if speed.shape != (batch_size,):
            raise ValueError("speed_limit must have shape (B,)")
        if not np.all(np.isfinite(centers[mask])):
            raise ValueError("real obstacle centers must be finite")
        if not np.all(np.isfinite(radii[mask]) & (radii[mask] > 0)):
            raise ValueError("real obstacle radii must be finite and positive")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("arena bounds must be finite")
        if not np.all(upper > lower):
            raise ValueError("every arena upper bound must exceed its lower bound")
        if not np.all(np.isfinite(speed) & (speed > 0)):
            raise ValueError("speed limits must be finite and positive")
