"""Rigid-body safety values and continuous barriers for DA-PLCBF Version A.

The direct-wrench state is ``[position(3), quaternion_xyzw(4), velocity(3),
body_angular_velocity(3)]``.  Spherical-obstacle, arena, altitude, and tilt constraints have
relative degree two and are represented by exponential HOCBF conditions.  Speed and angular-rate
constraints have relative degree one.  Every returned row uses the common convention
``matrix @ wrench <= upper_bound``; its exact continuous-time barrier residual is therefore
``upper_bound - matrix @ wrench``.

The finite-horizon value helpers use dimensionless physical margins and an exact hard minimum.
They deliberately reject a tied active minimum as a differentiable continuous PL-CBF certificate:
one arbitrary autodiff branch at a nonsmooth tie is not a valid single-halfspace contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.direct_wrench import (
    ControlAffineTerms,
    control_affine_terms,
    quaternion_to_rotation_matrix,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


STATE_SIZE = 13
WRENCH_SIZE = 4


class VersionAModel(NamedTuple):
    """Parameters of the airborne direct-wrench model, in SI units."""

    mass: Array
    gravity_vec: Array
    inertia: Array
    inertia_inv: Array
    drag_matrix: Array
    wind_velocity: Array
    external_force: Array
    external_torque: Array


class RigidBodySafetySet(NamedTuple):
    """Fixed-shape rigid-body safety geometry and limits.

    Masked obstacle slots may contain arbitrary padding.  Every unmasked obstacle is a sphere.
    ``arena_lower`` and ``arena_upper`` include altitude as their third Cartesian component.
    """

    obstacle_centers: Array
    obstacle_radii: Array
    obstacle_mask: Array
    arena_lower: Array
    arena_upper: Array
    speed_max: Array
    angular_rate_max: Array
    tilt_max_radians: Array


@dataclass(frozen=True, slots=True)
class VersionABarrierConfig:
    """Numerical and physical configuration for Version-A safety constraints."""

    obstacle_clearance: float = 0.0
    arena_clearance: float = 0.0
    position_alpha_1: float = 2.0
    position_alpha_2: float = 2.0
    speed_alpha: float = 2.0
    angular_rate_alpha: float = 2.0
    tilt_alpha_1: float = 4.0
    tilt_alpha_2: float = 4.0
    relative_degree_tolerance: float = 2e-6
    domain_tolerance: float = 1e-7
    minimum_tie_tolerance: float = 1e-6
    model_tolerance: float = 2e-5
    quaternion_norm_tolerance: float = 2e-4
    ego_radius: float = 0.0
    include_obstacle_hocbf: bool = True

    def validate(self) -> None:
        """Reject nonphysical or nonfinite barrier settings."""
        finite_values = (
            self.obstacle_clearance,
            self.ego_radius,
            self.arena_clearance,
            self.position_alpha_1,
            self.position_alpha_2,
            self.speed_alpha,
            self.angular_rate_alpha,
            self.tilt_alpha_1,
            self.tilt_alpha_2,
            self.relative_degree_tolerance,
            self.domain_tolerance,
            self.minimum_tie_tolerance,
            self.model_tolerance,
            self.quaternion_norm_tolerance,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("all Version-A barrier configuration values must be finite")
        if not isinstance(self.include_obstacle_hocbf, bool):
            raise TypeError("include_obstacle_hocbf must be boolean")
        if min(self.obstacle_clearance, self.ego_radius, self.arena_clearance) < 0:
            raise ValueError("physical clearances must be nonnegative")
        if (
            min(
                self.position_alpha_1,
                self.position_alpha_2,
                self.speed_alpha,
                self.angular_rate_alpha,
                self.tilt_alpha_1,
                self.tilt_alpha_2,
            )
            <= 0
        ):
            raise ValueError("all CBF and HOCBF rates must be positive")
        if (
            min(
                self.relative_degree_tolerance,
                self.domain_tolerance,
                self.minimum_tie_tolerance,
                self.model_tolerance,
                self.quaternion_norm_tolerance,
            )
            < 0
        ):
            raise ValueError("all Version-A numerical tolerances must be nonnegative")


class SafetyValueResult(NamedTuple):
    """Dimensionless physical margins in the documented constraint order."""

    values: Array
    enabled: Array
    input_valid: Array


class ContinuousBarrierHalfspaces(NamedTuple):
    r"""Continuous CBF/HOCBF faces and their domain diagnostics.

    ``raw_values`` retains each barrier's natural physical units.  ``first_order_values`` is
    :math:`\dot h + \alpha_1 h` for relative-degree-2 barriers and positive infinity for
    relative-degree-1 barriers.  A classical forward-invariance claim requires ``domain_valid``
    before the projected action is applied.
    """

    matrix: Array
    upper_bound: Array
    raw_values: Array
    first_order_values: Array
    relative_degrees: Array
    enabled: Array
    relative_degree_valid: Array
    input_valid: Array
    domain_valid: Array


class ValidatedControlAffineTerms(NamedTuple):
    """Sanitised control-affine terms plus validity of the original state/model."""

    terms: ControlAffineTerms
    input_valid: Array


class HardPolicyCertificate(NamedTuple):
    """Exact hard finite-horizon value and one differentiable active-branch gradient."""

    value: Array
    gradient: Array
    gradient_valid: Array
    active_index: Array
    second_value_gap: Array
    constraint_values: Array
    input_valid: Array


def split_state(state: Array) -> tuple[Array, Array, Array, Array]:
    """Split one flattened direct-wrench state into Crazyflow components."""
    if state.shape != (STATE_SIZE,):
        raise ValueError(f"state must have shape ({STATE_SIZE},), got {state.shape}")
    if not jnp.issubdtype(state.dtype, jnp.floating):
        raise ValueError(f"state must have floating-point dtype, got {state.dtype}")
    return state[:3], state[3:7], state[7:10], state[10:13]


def safety_constraint_names(obstacle_count: int) -> tuple[str, ...]:
    """Return stable names matching every safety-value and halfspace vector."""
    if (
        isinstance(obstacle_count, bool)
        or not isinstance(obstacle_count, int)
        or obstacle_count < 0
    ):
        raise ValueError("obstacle_count must be a nonnegative integer")
    obstacles = tuple(f"obstacle_{index}" for index in range(obstacle_count))
    return obstacles + (
        "arena_x_lower",
        "arena_y_lower",
        "altitude_lower",
        "arena_x_upper",
        "arena_y_upper",
        "altitude_upper",
        "speed",
        "angular_rate",
        "tilt",
    )


def _check_model_shapes(model: VersionAModel) -> None:
    expected = {
        "gravity_vec": (3,),
        "inertia": (3, 3),
        "inertia_inv": (3, 3),
        "drag_matrix": (3, 3),
        "wind_velocity": (3,),
        "external_force": (3,),
        "external_torque": (3,),
    }
    if jnp.asarray(model.mass).shape not in ((), (1,)):
        raise ValueError("model.mass must be scalar or shape (1,)")
    for name, shape in expected.items():
        actual = jnp.asarray(getattr(model, name)).shape
        if actual != shape:
            raise ValueError(f"model.{name} must have shape {shape}, got {actual}")


def _check_safety_shapes(safety: RigidBodySafetySet) -> None:
    centers = jnp.asarray(safety.obstacle_centers)
    radii = jnp.asarray(safety.obstacle_radii)
    mask = jnp.asarray(safety.obstacle_mask)
    if centers.ndim != 2 or centers.shape[-1] != 3:
        raise ValueError("obstacle_centers must have shape (obstacles, 3)")
    if radii.shape != centers.shape[:1] or mask.shape != centers.shape[:1]:
        raise ValueError("obstacle_radii and obstacle_mask must match obstacle count")
    if not jnp.issubdtype(mask.dtype, jnp.bool_):
        raise ValueError("obstacle_mask must have boolean dtype")
    for name in ("arena_lower", "arena_upper"):
        if jnp.asarray(getattr(safety, name)).shape != (3,):
            raise ValueError(f"{name} must have shape (3,)")
    for name in ("speed_max", "angular_rate_max", "tilt_max_radians"):
        if jnp.asarray(getattr(safety, name)).shape not in ((), (1,)):
            raise ValueError(f"{name} must be scalar or shape (1,)")


def _safe_state_and_validity(
    state: Array, quaternion_norm_tolerance: float = 2e-4
) -> tuple[Array, Array]:
    finite = jnp.all(jnp.isfinite(state))
    safe_state = jnp.where(jnp.isfinite(state), state, jnp.zeros_like(state))
    quat_norm = jnp.linalg.norm(safe_state[3:7])
    quaternion_valid = (quat_norm > 32 * jnp.finfo(state.dtype).eps) & (
        jnp.abs(quat_norm - 1.0) <= quaternion_norm_tolerance
    )
    identity_quaternion = jnp.asarray([0.0, 0.0, 0.0, 1.0], dtype=state.dtype)
    safe_quaternion = jnp.where(quaternion_valid, safe_state[3:7], identity_quaternion)
    safe_state = safe_state.at[3:7].set(safe_quaternion)
    return safe_state, finite & quaternion_valid


def _safe_model_and_validity(
    model: VersionAModel, dtype: jnp.dtype, model_tolerance: float
) -> tuple[VersionAModel, Array]:
    _check_model_shapes(model)
    mass = jnp.reshape(jnp.asarray(model.mass, dtype=dtype), ())
    gravity = jnp.asarray(model.gravity_vec, dtype=dtype)
    inertia = jnp.asarray(model.inertia, dtype=dtype)
    inertia_inv = jnp.asarray(model.inertia_inv, dtype=dtype)
    drag = jnp.asarray(model.drag_matrix, dtype=dtype)
    wind = jnp.asarray(model.wind_velocity, dtype=dtype)
    force = jnp.asarray(model.external_force, dtype=dtype)
    torque = jnp.asarray(model.external_torque, dtype=dtype)
    arrays: Sequence[Array] = (gravity, inertia, inertia_inv, drag, wind, force, torque)
    finite = jnp.isfinite(mass) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in arrays]))
    finite_inertia = jnp.where(jnp.all(jnp.isfinite(inertia)), inertia, jnp.eye(3, dtype=dtype))
    symmetric_inertia = 0.5 * (finite_inertia + finite_inertia.T)
    eigenvalues = jnp.linalg.eigvalsh(symmetric_inertia)
    inertia_positive = jnp.min(eigenvalues) > 0
    inertia_scale = jnp.maximum(jnp.max(jnp.abs(finite_inertia)), jnp.finfo(dtype).tiny)
    symmetry_error = jnp.max(jnp.abs(finite_inertia - finite_inertia.T)) / inertia_scale
    inverse_error = jnp.max(jnp.abs(finite_inertia @ inertia_inv - jnp.eye(3, dtype=dtype)))
    valid = (
        finite
        & (mass > 0)
        & inertia_positive
        & (symmetry_error <= model_tolerance)
        & (inverse_error <= model_tolerance)
    )
    safe_mass = jnp.where(valid, mass, jnp.asarray(1.0, dtype=dtype))
    safe_inertia = jnp.where(valid, finite_inertia, jnp.eye(3, dtype=dtype))
    safe_inverse = jnp.where(valid, inertia_inv, jnp.eye(3, dtype=dtype))

    def finite_or_zero(value: Array) -> Array:
        return jnp.where(jnp.isfinite(value), value, jnp.zeros_like(value))

    return (
        VersionAModel(
            safe_mass,
            finite_or_zero(gravity),
            safe_inertia,
            safe_inverse,
            finite_or_zero(drag),
            finite_or_zero(wind),
            finite_or_zero(force),
            finite_or_zero(torque),
        ),
        valid,
    )


def validated_control_affine_terms(
    state: Array,
    model: VersionAModel,
    *,
    model_tolerance: float = 2e-5,
    quaternion_norm_tolerance: float = 2e-4,
) -> ValidatedControlAffineTerms:
    """Return finite control-affine terms while retaining fail-closed input validity."""
    split_state(state)
    safe_state, state_valid = _safe_state_and_validity(state, quaternion_norm_tolerance)
    safe_model, model_valid = _safe_model_and_validity(model, state.dtype, model_tolerance)
    pos, quat, vel, ang_vel = split_state(safe_state)
    terms = control_affine_terms(
        pos,
        quat,
        vel,
        ang_vel,
        mass=safe_model.mass,
        gravity_vec=safe_model.gravity_vec,
        J=safe_model.inertia,
        J_inv=safe_model.inertia_inv,
        drag_matrix=safe_model.drag_matrix,
        wind_velocity=safe_model.wind_velocity,
        external_force=safe_model.external_force,
        external_torque=safe_model.external_torque,
    )
    finite_terms = jnp.all(jnp.isfinite(terms.drift)) & jnp.all(jnp.isfinite(terms.input_matrix))
    return ValidatedControlAffineTerms(terms, state_valid & model_valid & finite_terms)


def _safe_safety_data(
    safety: RigidBodySafetySet, config: VersionABarrierConfig, dtype: jnp.dtype
) -> tuple[RigidBodySafetySet, Array]:
    _check_safety_shapes(safety)
    centers = jnp.asarray(safety.obstacle_centers, dtype=dtype)
    radii = jnp.asarray(safety.obstacle_radii, dtype=dtype)
    mask = jnp.asarray(safety.obstacle_mask, dtype=bool)
    real_obstacle_valid = (~mask) | (
        jnp.all(jnp.isfinite(centers), axis=-1)
        & jnp.isfinite(radii)
        & (radii > 0)
        & (radii + config.ego_radius + config.obstacle_clearance > 0)
    )
    safe_centers = jnp.where(mask[:, None] & jnp.isfinite(centers), centers, 0.0)
    safe_radii = jnp.where(mask & jnp.isfinite(radii) & (radii > 0), radii, 1.0)
    lower = jnp.asarray(safety.arena_lower, dtype=dtype)
    upper = jnp.asarray(safety.arena_upper, dtype=dtype)
    speed_max = jnp.reshape(jnp.asarray(safety.speed_max, dtype=dtype), ())
    angular_rate_max = jnp.reshape(jnp.asarray(safety.angular_rate_max, dtype=dtype), ())
    tilt_max = jnp.reshape(jnp.asarray(safety.tilt_max_radians, dtype=dtype), ())
    arena_valid = (
        jnp.all(jnp.isfinite(lower))
        & jnp.all(jnp.isfinite(upper))
        & jnp.all(lower + config.arena_clearance < upper - config.arena_clearance)
    )
    limits_valid = (
        jnp.isfinite(speed_max)
        & (speed_max > 0)
        & jnp.isfinite(angular_rate_max)
        & (angular_rate_max > 0)
        & jnp.isfinite(tilt_max)
        & (tilt_max > 0)
        & (tilt_max <= 0.5 * jnp.pi)
    )
    valid = jnp.all(real_obstacle_valid) & arena_valid & limits_valid
    safe_lower = jnp.where(jnp.isfinite(lower), lower, -jnp.ones_like(lower))
    safe_upper = jnp.where(jnp.isfinite(upper), upper, jnp.ones_like(upper))
    safe_speed = jnp.where(jnp.isfinite(speed_max) & (speed_max > 0), speed_max, 1.0)
    safe_angular = jnp.where(
        jnp.isfinite(angular_rate_max) & (angular_rate_max > 0), angular_rate_max, 1.0
    )
    safe_tilt = jnp.where(
        jnp.isfinite(tilt_max) & (tilt_max > 0) & (tilt_max <= 0.5 * jnp.pi),
        tilt_max,
        0.25 * jnp.pi,
    )
    return (
        RigidBodySafetySet(
            safe_centers,
            safe_radii,
            mask,
            safe_lower,
            safe_upper,
            safe_speed,
            safe_angular,
            safe_tilt,
        ),
        valid,
    )


def dimensionless_safety_values(
    state: Array, safety: RigidBodySafetySet, config: VersionABarrierConfig
) -> SafetyValueResult:
    """Evaluate dimensionless margins for hard finite-horizon comparison.

    Positive is safe, zero is the physical boundary, and negative is a violation.  Dividing each
    quantity by its own physical scale prevents a hard minimum from mixing metres, squared metres,
    velocity squared, and dimensionless tilt.
    """
    config.validate()
    split_state(state)
    safe_state, state_valid = _safe_state_and_validity(state, config.quaternion_norm_tolerance)
    safe_safety, safety_valid = _safe_safety_data(safety, config, state.dtype)
    pos, quat, vel, ang_vel = split_state(safe_state)
    relative = pos[None, :] - safe_safety.obstacle_centers
    effective_radii = safe_safety.obstacle_radii + config.ego_radius + config.obstacle_clearance
    obstacle_values = (jnp.sum(relative**2, axis=-1) - effective_radii**2) / effective_radii**2
    obstacle_values = jnp.where(safe_safety.obstacle_mask, obstacle_values, jnp.inf)
    arena_span = safe_safety.arena_upper - safe_safety.arena_lower
    lower_values = (pos - (safe_safety.arena_lower + config.arena_clearance)) / arena_span
    upper_values = ((safe_safety.arena_upper - config.arena_clearance) - pos) / arena_span
    speed_value = 1.0 - jnp.sum(vel**2) / safe_safety.speed_max**2
    angular_value = 1.0 - jnp.sum(ang_vel**2) / safe_safety.angular_rate_max**2
    body_z_world = quaternion_to_rotation_matrix(quat)[:, 2]
    cosine_limit = jnp.cos(safe_safety.tilt_max_radians)
    tilt_value = (body_z_world[2] - cosine_limit) / (1.0 - cosine_limit)
    values = jnp.concatenate(
        (
            obstacle_values,
            lower_values,
            upper_values,
            speed_value[None],
            angular_value[None],
            tilt_value[None],
        )
    )
    enabled = jnp.concatenate((safe_safety.obstacle_mask, jnp.ones((9,), dtype=bool)), axis=0)
    finite_enabled = jnp.all(jnp.where(enabled, jnp.isfinite(values), True))
    return SafetyValueResult(values, enabled, state_valid & safety_valid & finite_enabled)


def _terms_for_safe_state(state: Array, model: VersionAModel) -> ControlAffineTerms:
    pos, quat, vel, ang_vel = split_state(state)
    return control_affine_terms(
        pos,
        quat,
        vel,
        ang_vel,
        mass=model.mass,
        gravity_vec=model.gravity_vec,
        J=model.inertia,
        J_inv=model.inertia_inv,
        drag_matrix=model.drag_matrix,
        wind_velocity=model.wind_velocity,
        external_force=model.external_force,
        external_torque=model.external_torque,
    )


def _relative_degree_one(
    h_function: Callable[[Array], Array], state: Array, terms: ControlAffineTerms, alpha: float
) -> tuple[Array, Array, Array, Array]:
    value, gradient = jax.value_and_grad(h_function)(state)
    drift = jnp.dot(gradient, terms.drift)
    control = gradient @ terms.input_matrix
    return -control, drift + alpha * value, value, jnp.asarray(jnp.inf, dtype=state.dtype)


def _relative_degree_two(
    h_function: Callable[[Array], Array],
    state: Array,
    model: VersionAModel,
    terms: ControlAffineTerms,
    alpha_1: float,
    alpha_2: float,
    relative_degree_tolerance: float,
) -> tuple[Array, Array, Array, Array, Array]:
    value, gradient = jax.value_and_grad(h_function)(state)
    direct_control = gradient @ terms.input_matrix

    def first_order(z: Array) -> Array:
        h_value, h_gradient = jax.value_and_grad(h_function)(z)
        return jnp.dot(h_gradient, _terms_for_safe_state(z, model).drift) + alpha_1 * h_value

    first_value, first_gradient = jax.value_and_grad(first_order)(state)
    drift = jnp.dot(first_gradient, terms.drift)
    control = first_gradient @ terms.input_matrix
    scale = jnp.maximum(jnp.linalg.norm(gradient) * jnp.linalg.norm(terms.input_matrix), 1.0)
    degree_valid = jnp.linalg.norm(direct_control) <= relative_degree_tolerance * scale
    return -control, drift + alpha_2 * first_value, value, first_value, degree_valid


def continuous_safety_halfspaces(
    state: Array,
    model: VersionAModel,
    safety: RigidBodySafetySet,
    config: VersionABarrierConfig,
    *,
    obstacle_velocities: Array | None = None,
) -> ContinuousBarrierHalfspaces:
    """Construct operational faces plus explicitly enabled obstacle HOCBF faces.

    Obstacle rows retain stable slots but are disabled independently of arena, altitude, speed,
    angular-rate, and tilt limits when ``include_obstacle_hocbf`` is false. This separates
    analytic collision avoidance from a rollout-value collision certificate. Optional obstacle
    velocities implement the explicit time terms for a constant-velocity spherical prediction;
    obstacle acceleration is assumed zero over this local analytic model.
    """
    config.validate()
    split_state(state)
    safe_state, state_valid = _safe_state_and_validity(state, config.quaternion_norm_tolerance)
    safe_model, model_valid = _safe_model_and_validity(model, state.dtype, config.model_tolerance)
    safe_safety, safety_valid = _safe_safety_data(safety, config, state.dtype)
    terms = _terms_for_safe_state(safe_state, safe_model)
    position, _, velocity, _ = split_state(safe_state)
    if obstacle_velocities is None:
        obstacle_velocities = jnp.zeros_like(safe_safety.obstacle_centers)
    if obstacle_velocities.shape != safe_safety.obstacle_centers.shape:
        raise ValueError("obstacle_velocities must match obstacle_centers")
    velocity_valid = jnp.all(
        jnp.where(safe_safety.obstacle_mask[:, None], jnp.isfinite(obstacle_velocities), True)
    )
    obstacle_velocities = jnp.where(
        safe_safety.obstacle_mask[:, None] & jnp.isfinite(obstacle_velocities),
        obstacle_velocities,
        0.0,
    )

    rows: list[Array] = []
    bounds: list[Array] = []
    raw_values: list[Array] = []
    first_values: list[Array] = []
    relative_degrees: list[int] = []
    degree_validity: list[Array] = []
    enabled: list[Array] = []

    def add_second_order(h_function: Callable[[Array], Array], is_enabled: Array) -> None:
        row, bound, raw, first, degree_valid = _relative_degree_two(
            h_function,
            safe_state,
            safe_model,
            terms,
            config.position_alpha_1,
            config.position_alpha_2,
            config.relative_degree_tolerance,
        )
        rows.append(jnp.where(is_enabled, row, jnp.zeros_like(row)))
        bounds.append(jnp.where(is_enabled, bound, jnp.asarray(1.0, state.dtype)))
        raw_values.append(jnp.where(is_enabled, raw, jnp.inf))
        first_values.append(jnp.where(is_enabled, first, jnp.inf))
        relative_degrees.append(2)
        degree_validity.append(jnp.where(is_enabled, degree_valid, True))
        enabled.append(is_enabled)

    for index in range(safe_safety.obstacle_centers.shape[0]):
        center = safe_safety.obstacle_centers[index]
        radius = safe_safety.obstacle_radii[index] + config.ego_radius + config.obstacle_clearance

        relative = position - center
        relative_velocity = velocity - obstacle_velocities[index]
        raw = jnp.dot(relative, relative) - radius**2
        h_dot = 2.0 * jnp.dot(relative, relative_velocity)
        first = h_dot + config.position_alpha_1 * raw
        row = -2.0 * relative @ terms.input_matrix[7:10]
        bound = (
            2.0 * jnp.dot(relative_velocity, relative_velocity)
            + 2.0 * jnp.dot(relative, terms.drift[7:10])
            + (config.position_alpha_1 + config.position_alpha_2) * h_dot
            + config.position_alpha_1 * config.position_alpha_2 * raw
        )
        is_enabled = safe_safety.obstacle_mask[index] & config.include_obstacle_hocbf
        rows.append(jnp.where(is_enabled, row, jnp.zeros_like(row)))
        bounds.append(jnp.where(is_enabled, bound, jnp.asarray(1.0, state.dtype)))
        raw_values.append(jnp.where(is_enabled, raw, jnp.inf))
        first_values.append(jnp.where(is_enabled, first, jnp.inf))
        relative_degrees.append(2)
        degree_validity.append(jnp.asarray(True))
        enabled.append(is_enabled)

    for axis in range(3):
        lower = safe_safety.arena_lower[axis] + config.arena_clearance

        def lower_h(z: Array, axis: int = axis, lower: Array = lower) -> Array:
            return z[axis] - lower

        add_second_order(lower_h, jnp.asarray(True))

    for axis in range(3):
        upper = safe_safety.arena_upper[axis] - config.arena_clearance

        def upper_h(z: Array, axis: int = axis, upper: Array = upper) -> Array:
            return upper - z[axis]

        add_second_order(upper_h, jnp.asarray(True))

    def speed_h(z: Array) -> Array:
        return safe_safety.speed_max**2 - jnp.dot(z[7:10], z[7:10])

    speed_row, speed_bound, speed_raw, speed_first = _relative_degree_one(
        speed_h, safe_state, terms, config.speed_alpha
    )
    rows.append(speed_row)
    bounds.append(speed_bound)
    raw_values.append(speed_raw)
    first_values.append(speed_first)
    relative_degrees.append(1)
    degree_validity.append(jnp.asarray(True))
    enabled.append(jnp.asarray(True))

    def angular_rate_h(z: Array) -> Array:
        return safe_safety.angular_rate_max**2 - jnp.dot(z[10:13], z[10:13])

    angular_row, angular_bound, angular_raw, angular_first = _relative_degree_one(
        angular_rate_h, safe_state, terms, config.angular_rate_alpha
    )
    rows.append(angular_row)
    bounds.append(angular_bound)
    raw_values.append(angular_raw)
    first_values.append(angular_first)
    relative_degrees.append(1)
    degree_validity.append(jnp.asarray(True))
    enabled.append(jnp.asarray(True))

    def tilt_h(z: Array) -> Array:
        body_z_world = quaternion_to_rotation_matrix(z[3:7])[:, 2]
        return body_z_world[2] - jnp.cos(safe_safety.tilt_max_radians)

    tilt_row, tilt_bound, tilt_raw, tilt_first, tilt_degree_valid = _relative_degree_two(
        tilt_h,
        safe_state,
        safe_model,
        terms,
        config.tilt_alpha_1,
        config.tilt_alpha_2,
        config.relative_degree_tolerance,
    )
    rows.append(tilt_row)
    bounds.append(tilt_bound)
    raw_values.append(tilt_raw)
    first_values.append(tilt_first)
    relative_degrees.append(2)
    degree_validity.append(tilt_degree_valid)
    enabled.append(jnp.asarray(True))

    matrix = jnp.stack(rows)
    upper_bound = jnp.stack(bounds)
    raw = jnp.stack(raw_values)
    first = jnp.stack(first_values)
    enabled_array = jnp.stack(enabled)
    degree_valid = jnp.stack(degree_validity)
    relative_degree_array = jnp.asarray(relative_degrees, dtype=jnp.int32)
    finite = (
        jnp.all(jnp.where(enabled_array[:, None], jnp.isfinite(matrix), True))
        & jnp.all(jnp.where(enabled_array, jnp.isfinite(upper_bound), True))
        & jnp.all(jnp.where(enabled_array, jnp.isfinite(raw), True))
        & jnp.all(
            jnp.where(enabled_array & (relative_degree_array == 2), jnp.isfinite(first), True)
        )
    )
    input_valid = (
        state_valid
        & model_valid
        & safety_valid
        & velocity_valid
        & finite
        & jnp.all(jnp.where(enabled_array, degree_valid, True))
    )
    in_domain = (raw >= -config.domain_tolerance) & (
        (relative_degree_array == 1) | (first >= -config.domain_tolerance)
    )
    domain_valid = input_valid & jnp.all(jnp.where(enabled_array, in_domain, True))
    return ContinuousBarrierHalfspaces(
        matrix,
        upper_bound,
        raw,
        first,
        relative_degree_array,
        enabled_array,
        degree_valid,
        input_valid,
        domain_valid,
    )


def hard_finite_horizon_policy_certificate(
    state: Array,
    rollout_function: Callable[[Array], Array],
    safety: RigidBodySafetySet,
    config: VersionABarrierConfig,
) -> HardPolicyCertificate:
    """Differentiate the unique active branch of an exact hard rollout minimum.

    ``rollout_function`` must return ``(horizon_nodes, 13)`` states and include the current state
    as its first node.  The value is the exact minimum of the dimensionless physical margins over
    all nodes and enabled constraints.  At a tie within ``minimum_tie_tolerance``, the value stays
    valid for reporting but ``gradient_valid`` is false, preventing use as one continuous
    halfspace.
    """
    config.validate()
    split_state(state)

    def flattened_values(x: Array) -> tuple[Array, Array]:
        rollout = rollout_function(x)
        if rollout.ndim != 2 or rollout.shape[-1] != STATE_SIZE or rollout.shape[0] < 1:
            raise ValueError("rollout_function must return shape (positive_horizon_nodes, 13)")
        results = jax.vmap(lambda node: dimensionless_safety_values(node, safety, config))(rollout)
        current_scale = jnp.maximum(jnp.max(jnp.abs(x)), 1.0)
        current_tolerance = 32.0 * jnp.finfo(x.dtype).eps * current_scale
        includes_current = jnp.max(jnp.abs(rollout[0] - x)) <= current_tolerance
        return results.values.reshape(-1), jnp.all(results.input_valid) & includes_current

    def value_function(x: Array) -> Array:
        values, _ = flattened_values(x)
        return jnp.min(values)

    value, gradient = jax.value_and_grad(value_function)(state)
    values, rollout_valid = flattened_values(state)
    finite_values = jnp.where(jnp.isfinite(values), values, jnp.inf)
    order = jnp.argsort(finite_values)
    active_index = order[0]
    finite_count = jnp.sum(jnp.isfinite(values), dtype=jnp.int32)
    second = jnp.where(finite_count > 1, finite_values[order[1]], jnp.inf)
    gap = second - finite_values[active_index]
    input_valid = rollout_valid & jnp.isfinite(value)
    gradient_valid = (
        input_valid
        & jnp.all(jnp.isfinite(gradient))
        & ((finite_count == 1) | (gap > config.minimum_tie_tolerance))
    )
    return HardPolicyCertificate(
        value, gradient, gradient_valid, active_index, gap, values, input_valid
    )


__all__ = [
    "ContinuousBarrierHalfspaces",
    "HardPolicyCertificate",
    "RigidBodySafetySet",
    "SafetyValueResult",
    "ValidatedControlAffineTerms",
    "VersionABarrierConfig",
    "VersionAModel",
    "continuous_safety_halfspaces",
    "dimensionless_safety_values",
    "hard_finite_horizon_policy_certificate",
    "safety_constraint_names",
    "split_state",
    "validated_control_affine_terms",
]
