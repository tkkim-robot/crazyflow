"""One-state wind estimator for the corrected online DA-PLCBF demonstration.

This module deliberately estimates only the current wind vector.  It has no covariance,
particles, uncertainty set, or Cartesian product.  With the known Version-A mass and drag model,
one measured translational acceleration is sufficient to infer the instantaneous wind.  A
first-order filter makes the estimate evolve visibly after the single wind step in the demo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.direct_wrench import quaternion_to_rotation_matrix

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel


@dataclass(frozen=True, slots=True)
class PointWindEstimatorConfig:
    """Configuration for the deterministic low-pass point estimate."""

    response_rate: float = 1.8
    component_limit: float = 5.0
    minimum_drag_singular_value: float = 1e-5

    def validate(self) -> None:
        values = (self.response_rate, self.component_limit, self.minimum_drag_singular_value)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("all point-wind estimator settings must be finite and positive")


class PointWindEstimatorState(NamedTuple):
    """Current point estimate and monotonic update counters."""

    wind_velocity: Array
    update_count: Array
    finite_update_count: Array


class PointWindEstimatorUpdate(NamedTuple):
    """Updated state plus the unfiltered inferred wind and numerical validity."""

    state: PointWindEstimatorState
    instantaneous_wind: Array
    measurement_valid: Array


def initialize_point_wind_estimator(
    *, dtype: jnp.dtype = jnp.float32, initial_wind: Array | None = None
) -> PointWindEstimatorState:
    """Create one point estimate; no uncertainty state is allocated."""
    wind = (
        jnp.zeros(3, dtype=dtype)
        if initial_wind is None
        else jnp.asarray(initial_wind, dtype=dtype)
    )
    if wind.shape != (3,):
        raise ValueError("initial_wind must have shape (3,)")
    return PointWindEstimatorState(
        wind_velocity=wind,
        update_count=jnp.zeros((), dtype=jnp.int32),
        finite_update_count=jnp.zeros((), dtype=jnp.int32),
    )


def update_point_wind_estimator(
    estimator: PointWindEstimatorState,
    previous_state: Array,
    next_state: Array,
    applied_wrench: Array,
    known_model: VersionAModel,
    *,
    dt: float,
    config: PointWindEstimatorConfig = PointWindEstimatorConfig(),
) -> PointWindEstimatorUpdate:
    """Infer and filter one wind measurement from consecutive Version-A states.

    The Version-A translational dynamics give

    ``m(a-g) - thrust - external_force = R D R.T (v-w)``.

    Solving that relation for ``w`` uses only measured state transition, the applied wrench, and
    the known fixed mass/drag parameters.  The true plant wind is never supplied to this function.
    """
    config.validate()
    if previous_state.shape != (13,) or next_state.shape != (13,):
        raise ValueError("previous_state and next_state must have shape (13,)")
    if applied_wrench.shape != (4,):
        raise ValueError("applied_wrench must have shape (4,)")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")

    dtype = previous_state.dtype
    rotation = quaternion_to_rotation_matrix(previous_state[3:7])
    mass = jnp.reshape(jnp.asarray(known_model.mass, dtype=dtype), ())
    gravity = jnp.asarray(known_model.gravity_vec, dtype=dtype)
    drag = jnp.asarray(known_model.drag_matrix, dtype=dtype)
    external_force = jnp.asarray(known_model.external_force, dtype=dtype)
    observed_acceleration = (next_state[7:10] - previous_state[7:10]) / dt
    thrust_world = applied_wrench[0] * rotation[:, 2]
    drag_world = mass * (observed_acceleration - gravity) - thrust_world - external_force
    drag_body = rotation.T @ drag_world

    singular_values = jnp.linalg.svd(drag, compute_uv=False)
    matrix_valid = (
        jnp.all(jnp.isfinite(singular_values))
        & (jnp.min(singular_values) >= config.minimum_drag_singular_value)
        & jnp.isfinite(mass)
        & (mass > 0)
    )
    safe_drag = jnp.where(matrix_valid, drag, -jnp.eye(3, dtype=dtype))
    relative_air_body = jnp.linalg.solve(safe_drag, drag_body)
    instantaneous = previous_state[7:10] - rotation @ relative_air_body
    instantaneous = jnp.clip(instantaneous, -config.component_limit, config.component_limit)
    measurement_valid = (
        matrix_valid
        & jnp.all(jnp.isfinite(previous_state))
        & jnp.all(jnp.isfinite(next_state))
        & jnp.all(jnp.isfinite(applied_wrench))
        & jnp.all(jnp.isfinite(instantaneous))
    )
    alpha = jnp.asarray(1.0 - math.exp(-config.response_rate * dt), dtype=dtype)
    proposed = estimator.wind_velocity + alpha * (instantaneous - estimator.wind_velocity)
    wind = jnp.where(measurement_valid, proposed, estimator.wind_velocity)
    state = PointWindEstimatorState(
        wind_velocity=wind,
        update_count=estimator.update_count + jnp.asarray(1, dtype=jnp.int32),
        finite_update_count=estimator.finite_update_count + measurement_valid.astype(jnp.int32),
    )
    return PointWindEstimatorUpdate(state, instantaneous, measurement_valid)


def model_with_point_wind(
    model: VersionAModel, estimator: PointWindEstimatorState
) -> VersionAModel:
    """Return the single current differentiable model used by learner and controller."""
    return model._replace(wind_velocity=estimator.wind_velocity)


__all__ = [
    "PointWindEstimatorConfig",
    "PointWindEstimatorState",
    "PointWindEstimatorUpdate",
    "initialize_point_wind_estimator",
    "model_with_point_wind",
    "update_point_wind_estimator",
]
