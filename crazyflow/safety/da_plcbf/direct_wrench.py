"""Control-affine direct-wrench rigid-body model for DA-PLCBF Version A.

Version A treats collective thrust and body torque as the plant input.  It intentionally omits
rotor-speed dynamics, ground contact, and Crazyflow's special all-zero idle command.  Its actuator
set is the convex airborne set obtained by bounding each (unclipped) motor force.

The state convention matches Crazyflow: position and velocity are in the world frame, quaternion
is scalar-last ``(x, y, z, w)`` and rotates body vectors into the world frame, and angular velocity
is in the body frame.  External force and torque are both supplied in the world frame, matching
``SimState.force`` and ``SimState.torque``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from array_api_compat import array_namespace

if TYPE_CHECKING:
    from crazyflow._typing import Array


class DirectWrenchDerivative(NamedTuple):
    """Derivative of Crazyflow's 13-dimensional rigid-body state."""

    pos_dot: Array
    quat_dot: Array
    vel_dot: Array
    ang_vel_dot: Array


class ControlAffineTerms(NamedTuple):
    """Flattened ``x_dot = drift + input_matrix @ wrench`` representation."""

    drift: Array
    input_matrix: Array


class AffineMotorThrustConstraints(NamedTuple):
    r"""Affine motor bounds represented as ``matrix @ wrench <= upper_bound``."""

    matrix: Array
    upper_bound: Array


def _check_vector(name: str, value: Array, size: int) -> None:
    if value.ndim < 1 or value.shape[-1] != size:
        raise ValueError(f"{name} must have shape (..., {size}), got {value.shape}")


def _check_state_shapes(pos: Array, quat: Array, vel: Array, ang_vel: Array, wrench: Array) -> None:
    _check_vector("pos", pos, 3)
    _check_vector("quat", quat, 4)
    _check_vector("vel", vel, 3)
    _check_vector("ang_vel", ang_vel, 3)
    _check_vector("wrench", wrench, 4)
    batch_shape = pos.shape[:-1]
    if any(value.shape[:-1] != batch_shape for value in (quat, vel, ang_vel, wrench)):
        raise ValueError("state components and wrench must have identical leading dimensions")


def _scalar_field(value: Array | float, reference: Array) -> Array:
    """Convert a scalar or per-state scalar to a broadcastable trailing-singleton array."""
    xp = array_namespace(reference)
    result = xp.asarray(value, dtype=reference.dtype)
    if result.shape == reference.shape[:-1]:
        result = xp.expand_dims(result, axis=-1)
    return result


def quaternion_to_rotation_matrix(quat: Array) -> Array:
    """Return the body-to-world rotation matrix for an ``xyzw`` quaternion.

    Quaternions are normalized before constructing the matrix, as they are by Crazyflow's SciPy
    rotation path.  A zero quaternion is invalid and consequently produces non-finite values.
    """
    _check_vector("quat", quat, 4)
    xp = array_namespace(quat)
    quat = quat / xp.sqrt(xp.sum(quat**2, axis=-1, keepdims=True))
    x, y, z, w = (quat[..., index] for index in range(4))
    two = xp.asarray(2, dtype=quat.dtype)
    one = xp.asarray(1, dtype=quat.dtype)
    return xp.stack(
        (
            xp.stack(
                (one - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)), axis=-1
            ),
            xp.stack(
                (two * (x * y + z * w), one - two * (x * x + z * z), two * (y * z - x * w)), axis=-1
            ),
            xp.stack(
                (two * (x * z - y * w), two * (y * z + x * w), one - two * (x * x + y * y)), axis=-1
            ),
        ),
        axis=-2,
    )


def quaternion_derivative_xyzw(quat: Array, body_ang_vel: Array) -> Array:
    r"""Return ``0.5 * quat ⊗ [body_ang_vel, 0]`` in scalar-last ordering."""
    _check_vector("quat", quat, 4)
    _check_vector("body_ang_vel", body_ang_vel, 3)
    if quat.shape[:-1] != body_ang_vel.shape[:-1]:
        raise ValueError("quat and body_ang_vel must have identical leading dimensions")
    xp = array_namespace(quat, body_ang_vel)
    quat = quat / xp.sqrt(xp.sum(quat**2, axis=-1, keepdims=True))
    vector = quat[..., :3]
    scalar = quat[..., 3:4]
    vector_dot = scalar * body_ang_vel + xp.linalg.cross(vector, body_ang_vel)
    scalar_dot = -xp.sum(vector * body_ang_vel, axis=-1, keepdims=True)
    return xp.concat((vector_dot, scalar_dot), axis=-1) * xp.asarray(0.5, dtype=quat.dtype)


def direct_wrench_dynamics(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    wrench: Array,
    *,
    mass: Array | float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array | None = None,
    drag_matrix: Array,
    wind_velocity: Array | None = None,
    external_force: Array | None = None,
    external_torque: Array | None = None,
) -> DirectWrenchDerivative:
    r"""Evaluate direct collective-thrust/body-torque rigid-body dynamics.

    Args:
        pos: World position in metres, shape ``(..., 3)``.
        quat: Body-to-world quaternion in ``xyzw`` order, shape ``(..., 4)``.
        vel: World linear velocity in metres per second, shape ``(..., 3)``.
        ang_vel: Body angular velocity in radians per second, shape ``(..., 3)``.
        wrench: ``[collective_thrust, body_torque_x, body_torque_y, body_torque_z]`` with
            shape ``(..., 4)``.
        mass: Positive mass in kg, scalar or shape broadcastable to ``(..., 1)``.
        gravity_vec: World gravity acceleration, shape broadcastable to ``(..., 3)``.
        J: Body inertia matrix, shape broadcastable to ``(..., 3, 3)``.
        J_inv: Inverse body inertia. Computed from ``J`` when omitted.
        drag_matrix: Body-frame linear drag matrix. Crazyflow parameters use negative diagonal
            entries, shape broadcastable to ``(..., 3, 3)``.
        wind_velocity: World wind velocity. Drag acts on ``vel - wind_velocity``.
        external_force: External world-frame force acting at the centre of mass.
        external_torque: External world-frame torque acting at the centre of mass.

    Returns:
        Derivatives of position, quaternion, world velocity, and body angular velocity.

    Note:
        The function performs no clipping or contact handling.  It is the airborne Version-A
        model, not the existing rotor/controller simulation path.
    """
    _check_state_shapes(pos, quat, vel, ang_vel, wrench)
    xp = array_namespace(pos, quat, vel, ang_vel, wrench)
    mass = _scalar_field(mass, vel)
    gravity_vec = xp.asarray(gravity_vec, dtype=vel.dtype)
    inertia = xp.asarray(J, dtype=ang_vel.dtype)
    inverse_inertia = (
        xp.linalg.inv(inertia) if J_inv is None else xp.asarray(J_inv, dtype=ang_vel.dtype)
    )
    drag_matrix = xp.asarray(drag_matrix, dtype=vel.dtype)
    wind_velocity = (
        xp.zeros_like(vel) if wind_velocity is None else xp.asarray(wind_velocity, dtype=vel.dtype)
    )
    external_force = (
        xp.zeros_like(vel)
        if external_force is None
        else xp.asarray(external_force, dtype=vel.dtype)
    )
    external_torque = (
        xp.zeros_like(ang_vel)
        if external_torque is None
        else xp.asarray(external_torque, dtype=ang_vel.dtype)
    )

    rotation = quaternion_to_rotation_matrix(quat)
    rotation_transpose = rotation.mT
    body_z_world = rotation[..., :, 2]

    relative_air_velocity_body = (rotation_transpose @ (vel - wind_velocity)[..., None])[..., 0]
    drag_force_body = (drag_matrix @ relative_air_velocity_body[..., None])[..., 0]
    drag_force_world = (rotation @ drag_force_body[..., None])[..., 0]
    thrust_force_world = wrench[..., :1] * body_z_world
    vel_dot = gravity_vec + (thrust_force_world + drag_force_world + external_force) / mass

    external_torque_body = (rotation_transpose @ external_torque[..., None])[..., 0]
    angular_momentum = (inertia @ ang_vel[..., None])[..., 0]
    net_torque_body = wrench[..., 1:] + external_torque_body
    ang_vel_dot = (
        inverse_inertia @ (net_torque_body - xp.linalg.cross(ang_vel, angular_momentum))[..., None]
    )[..., 0]

    return DirectWrenchDerivative(
        pos_dot=vel,
        quat_dot=quaternion_derivative_xyzw(quat, ang_vel),
        vel_dot=vel_dot,
        ang_vel_dot=ang_vel_dot,
    )


def flatten_derivative(derivative: DirectWrenchDerivative) -> Array:
    """Flatten a structured derivative in ``[pos, quat, vel, ang_vel]`` state order."""
    xp = array_namespace(*derivative)
    return xp.concat(derivative, axis=-1)


def control_affine_terms(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    *,
    mass: Array | float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array | None = None,
    drag_matrix: Array,
    wind_velocity: Array | None = None,
    external_force: Array | None = None,
    external_torque: Array | None = None,
) -> ControlAffineTerms:
    r"""Return analytic drift and input map for ``x_dot = f(x) + g(x) wrench``.

    The flattened state order is ``[pos(3), quat(4), vel(3), ang_vel(3)]`` and the wrench order is
    ``[collective_thrust, body_torque(3)]``.  The returned shapes are ``(..., 13)`` and
    ``(..., 13, 4)``.
    """
    _check_vector("pos", pos, 3)
    _check_vector("quat", quat, 4)
    _check_vector("vel", vel, 3)
    _check_vector("ang_vel", ang_vel, 3)
    if any(value.shape[:-1] != pos.shape[:-1] for value in (quat, vel, ang_vel)):
        raise ValueError("state components must have identical leading dimensions")
    xp = array_namespace(pos, quat, vel, ang_vel)
    zero_wrench = xp.concat((xp.zeros_like(pos[..., :1]), xp.zeros_like(ang_vel)), axis=-1)
    drift = direct_wrench_dynamics(
        pos,
        quat,
        vel,
        ang_vel,
        zero_wrench,
        mass=mass,
        gravity_vec=gravity_vec,
        J=J,
        J_inv=J_inv,
        drag_matrix=drag_matrix,
        wind_velocity=wind_velocity,
        external_force=external_force,
        external_torque=external_torque,
    )

    mass = _scalar_field(mass, vel)
    inverse_inertia = (
        xp.linalg.inv(xp.asarray(J, dtype=ang_vel.dtype))
        if J_inv is None
        else xp.asarray(J_inv, dtype=ang_vel.dtype)
    )
    rotation = quaternion_to_rotation_matrix(quat)
    body_z_world = rotation[..., :, 2]
    zeros_3 = xp.zeros_like(vel)
    vel_input = xp.stack((body_z_world / mass, zeros_3, zeros_3, zeros_3), axis=-1)
    inverse_inertia = inverse_inertia + xp.zeros_like(rotation)
    angular_input = xp.concat((xp.expand_dims(zeros_3, axis=-1), inverse_inertia), axis=-1)
    zero_pos_input = xp.zeros_like(vel_input)
    zero_quat_input = xp.zeros(quat.shape + (4,), dtype=quat.dtype)
    input_matrix = xp.concat((zero_pos_input, zero_quat_input, vel_input, angular_input), axis=-2)
    return ControlAffineTerms(flatten_derivative(drift), input_matrix)


def apply_control_affine(terms: ControlAffineTerms, wrench: Array) -> Array:
    """Evaluate a flattened control-affine representation at ``wrench``."""
    _check_vector("wrench", wrench, 4)
    return terms.drift + (terms.input_matrix @ wrench[..., None])[..., 0]


def control_affine_identity_residual(
    pos: Array,
    quat: Array,
    vel: Array,
    ang_vel: Array,
    wrench: Array,
    *,
    mass: Array | float,
    gravity_vec: Array,
    J: Array,
    J_inv: Array | None = None,
    drag_matrix: Array,
    wind_velocity: Array | None = None,
    external_force: Array | None = None,
    external_torque: Array | None = None,
) -> Array:
    """Return direct derivative minus its analytic control-affine reconstruction."""
    direct = direct_wrench_dynamics(
        pos,
        quat,
        vel,
        ang_vel,
        wrench,
        mass=mass,
        gravity_vec=gravity_vec,
        J=J,
        J_inv=J_inv,
        drag_matrix=drag_matrix,
        wind_velocity=wind_velocity,
        external_force=external_force,
        external_torque=external_torque,
    )
    terms = control_affine_terms(
        pos,
        quat,
        vel,
        ang_vel,
        mass=mass,
        gravity_vec=gravity_vec,
        J=J,
        J_inv=J_inv,
        drag_matrix=drag_matrix,
        wind_velocity=wind_velocity,
        external_force=external_force,
        external_torque=external_torque,
    )
    return flatten_derivative(direct) - apply_control_affine(terms, wrench)


def motor_allocation_matrix(
    mixing_matrix: Array, *, L: Array | float, thrust2torque: Array | float
) -> Array:
    r"""Return the current Crazyflow map from wrench to four unclipped motor forces.

    For the configured X-frame mixing matrix this is the inverse of
    :func:`motor_forces_to_wrench`.  No motor bounds or idle-mode switch are applied.
    """
    if mixing_matrix.ndim < 2 or mixing_matrix.shape[-2:] != (3, 4):
        raise ValueError(f"mixing_matrix must have shape (..., 3, 4), got {mixing_matrix.shape}")
    xp = array_namespace(mixing_matrix)
    L = xp.asarray(L, dtype=mixing_matrix.dtype)
    thrust2torque = xp.asarray(thrust2torque, dtype=mixing_matrix.dtype)
    torque_scale = xp.stack((1 / L, 1 / L, 1 / thrust2torque), axis=-1)
    torque_columns = mixing_matrix.mT * torque_scale[..., None, :]
    collective_column = xp.ones_like(torque_columns[..., :1])
    return xp.concat((collective_column, torque_columns), axis=-1) / xp.asarray(
        4, dtype=mixing_matrix.dtype
    )


def wrench_to_motor_forces(
    wrench: Array, *, L: Array | float, thrust2torque: Array | float, mixing_matrix: Array
) -> Array:
    """Map ``[collective thrust, body torque]`` to motor forces without clipping."""
    _check_vector("wrench", wrench, 4)
    # The command determines the output namespace.  Physical parameters are commonly loaded as
    # NumPy arrays even when the command is a traced JAX value, so normalize them before doing any
    # array-API dispatch instead of asking ``array_namespace`` to reconcile incompatible arrays.
    xp = array_namespace(wrench)
    mixing_matrix = xp.asarray(mixing_matrix, dtype=wrench.dtype)
    allocation = motor_allocation_matrix(mixing_matrix, L=L, thrust2torque=thrust2torque)
    # This fixed 4x4 map is ill-conditioned by the small thrust-to-torque ratio.  A generic
    # CUDA matmul may select TF32 and introduce O(1e-4 N) round-trip error, enough to reject
    # otherwise valid policies.  The explicit four-term reduction stays in fp32 and is also
    # valid for the NumPy array-API path.
    return xp.sum(allocation * wrench[..., None, :], axis=-1)


def motor_forces_to_wrench(
    motor_forces: Array, *, L: Array | float, thrust2torque: Array | float, mixing_matrix: Array
) -> Array:
    """Map four motor forces to collective thrust and body torque."""
    _check_vector("motor_forces", motor_forces, 4)
    if mixing_matrix.ndim < 2 or mixing_matrix.shape[-2:] != (3, 4):
        raise ValueError(f"mixing_matrix must have shape (..., 3, 4), got {mixing_matrix.shape}")
    xp = array_namespace(motor_forces)
    mixing_matrix = xp.asarray(mixing_matrix, dtype=motor_forces.dtype)
    # See :func:`wrench_to_motor_forces`: avoid low-precision tensor-core lowering for this
    # tiny, condition-sensitive physical allocation map.
    mixed_forces = xp.sum(mixing_matrix * motor_forces[..., None, :], axis=-1)
    torque_scale = xp.stack(
        (
            xp.asarray(L, dtype=motor_forces.dtype),
            xp.asarray(L, dtype=motor_forces.dtype),
            xp.asarray(thrust2torque, dtype=motor_forces.dtype),
        ),
        axis=-1,
    )
    return xp.concat(
        (xp.sum(motor_forces, axis=-1, keepdims=True), mixed_forces * torque_scale), axis=-1
    )


def motor_thrust_inequalities(
    *,
    thrust_min: Array | float,
    thrust_max: Array | float,
    L: Array | float,
    thrust2torque: Array | float,
    mixing_matrix: Array,
) -> AffineMotorThrustConstraints:
    r"""Construct the airborne motor-force polytope in wrench coordinates.

    The result encodes ``matrix @ wrench <= upper_bound``.  The first four rows enforce motor
    maxima and the last four enforce motor minima.  ``thrust_min`` and ``thrust_max`` may be scalar
    or four-vectors.  The caller is responsible for supplying physically ordered, nonnegative
    airborne bounds; no special idle point is added to this convex set.
    """
    xp = array_namespace(mixing_matrix)
    allocation = motor_allocation_matrix(mixing_matrix, L=L, thrust2torque=thrust2torque)
    motor_shape_template = allocation[..., 0]
    lower = xp.ones_like(motor_shape_template) * xp.asarray(thrust_min, dtype=mixing_matrix.dtype)
    upper = xp.ones_like(motor_shape_template) * xp.asarray(thrust_max, dtype=mixing_matrix.dtype)
    matrix = xp.concat((allocation, -allocation), axis=-2)
    upper_bound = xp.concat((upper, -lower), axis=-1)
    return AffineMotorThrustConstraints(matrix, upper_bound)


def affine_feasibility_residual(wrench: Array, constraints: AffineMotorThrustConstraints) -> Array:
    r"""Return ``max(matrix @ wrench - upper_bound)`` for affine constraints.

    A nonpositive value is feasible, zero is on the boundary, and a positive value is the largest
    signed inequality violation in motor-force units.
    """
    _check_vector("wrench", wrench, 4)
    xp = array_namespace(wrench, constraints.matrix, constraints.upper_bound)
    row_residuals = (constraints.matrix @ wrench[..., None])[..., 0] - constraints.upper_bound
    return xp.max(row_residuals, axis=-1)


def motor_thrust_feasibility_residual(
    wrench: Array,
    *,
    thrust_min: Array | float,
    thrust_max: Array | float,
    L: Array | float,
    thrust2torque: Array | float,
    mixing_matrix: Array,
) -> Array:
    """Return the largest signed airborne motor-thrust-bound violation."""
    constraints = motor_thrust_inequalities(
        thrust_min=thrust_min,
        thrust_max=thrust_max,
        L=L,
        thrust2torque=thrust2torque,
        mixing_matrix=mixing_matrix,
    )
    return affine_feasibility_residual(wrench, constraints)


# Descriptive alias for callers that use "derivative" rather than "dynamics" terminology.
direct_wrench_derivative = direct_wrench_dynamics


__all__ = [
    "AffineMotorThrustConstraints",
    "ControlAffineTerms",
    "DirectWrenchDerivative",
    "affine_feasibility_residual",
    "apply_control_affine",
    "control_affine_identity_residual",
    "control_affine_terms",
    "direct_wrench_derivative",
    "direct_wrench_dynamics",
    "flatten_derivative",
    "motor_allocation_matrix",
    "motor_forces_to_wrench",
    "motor_thrust_feasibility_residual",
    "motor_thrust_inequalities",
    "quaternion_derivative_xyzw",
    "quaternion_to_rotation_matrix",
    "wrench_to_motor_forces",
]
