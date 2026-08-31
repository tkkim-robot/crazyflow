"""Task-agnostic shared fallback policies mapped to feasible quadrotor wrenches."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.actor import shared_fallback_actions
from crazyflow.safety.da_plcbf.direct_wrench import (
    motor_forces_to_wrench,
    quaternion_to_rotation_matrix,
    wrench_to_motor_forces,
)

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.actor import (
        SharedActorConfig,
        SharedActorParams,
        SharedActorSpec,
    )
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
    from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


@dataclass(frozen=True, slots=True)
class QuadPolicyConfig:
    """Geometric acceleration-to-wrench controller settings."""

    acceleration_limit: float = 4.0
    attitude_gain: float = 8e-4
    angular_rate_gain: float = 2e-4
    force_direction_tolerance: float = 1e-6
    heading_tolerance: float = 1e-6
    # The allocation routines use an explicit fp32 four-term reduction so CUDA cannot lower this
    # condition-sensitive map to TF32.  A 1,048,576-row RTX 4090 audit measured a 3.8e-8 N maximum
    # round-trip error; this tolerance leaves headroom without expanding the actual motor box.
    allocation_tolerance: float = 1e-6

    def validate(self) -> None:
        """Reject nonphysical or singular controller settings."""
        values = (
            self.acceleration_limit,
            self.attitude_gain,
            self.angular_rate_gain,
            self.force_direction_tolerance,
            self.heading_tolerance,
            self.allocation_tolerance,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("all quad policy settings must be finite and positive")


class QuadWrenchCommand(NamedTuple):
    """Feasible wrench plus pre-allocation controller diagnostics."""

    wrench: Array
    desired_acceleration: Array
    desired_rotation: Array
    raw_wrench: Array
    raw_motor_forces: Array
    bounded_motor_forces: Array
    input_valid: Array


def _normalize(vector: Array, tolerance: float) -> tuple[Array, Array]:
    norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    valid = jnp.isfinite(norm[..., 0]) & (norm[..., 0] > tolerance)
    safe_norm = jnp.where(valid[..., None], norm, 1.0)
    return vector / safe_norm, valid


def acceleration_to_feasible_wrench(
    desired_acceleration: Array,
    quaternion: Array,
    angular_velocity: Array,
    model: VersionAModel,
    actuator: VersionAActuator,
    config: QuadPolicyConfig,
    *,
    smooth_motor_bounds: bool = True,
) -> QuadWrenchCommand:
    """Map desired world acceleration to an explicitly motor-feasible airborne wrench.

    A fixed world ``+x`` heading defines yaw; the desired body ``+z`` axis follows the required
    world force. The fallback acceleration bound is chosen below gravity in experiments, keeping
    this construction away from its horizontal-force heading singularity. Invalid/singular inputs
    return NaN commands rather than a fabricated orientation.

    Feasibility is part of the policy map itself: raw motor forces are transformed through a smooth
    box map for BPTT (or an explicit hard map for the non-trained nominal controller), then mapped
    back to wrench coordinates. No later allocation clip is credited as a certificate.
    """
    config.validate()
    if desired_acceleration.shape[-1:] != (3,):
        raise ValueError("desired_acceleration must end in dimension three")
    if quaternion.shape != (*desired_acceleration.shape[:-1], 4):
        raise ValueError("quaternion leading dimensions must match desired_acceleration")
    if angular_velocity.shape != desired_acceleration.shape:
        raise ValueError("angular_velocity must match desired_acceleration")
    if not isinstance(smooth_motor_bounds, bool):
        raise TypeError("smooth_motor_bounds must be boolean")

    dtype = desired_acceleration.dtype
    mass = jnp.reshape(jnp.asarray(model.mass, dtype=dtype), ())
    gravity = jnp.asarray(model.gravity_vec, dtype=dtype)
    inertia = jnp.asarray(model.inertia, dtype=dtype)
    force_world = mass * (desired_acceleration - gravity)
    desired_body_z, force_valid = _normalize(force_world, config.force_direction_tolerance)
    heading = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0], dtype=dtype), force_world.shape)
    desired_body_y, heading_valid = _normalize(
        jnp.cross(desired_body_z, heading), config.heading_tolerance
    )
    desired_body_x = jnp.cross(desired_body_y, desired_body_z)
    desired_rotation = jnp.stack((desired_body_x, desired_body_y, desired_body_z), axis=-1)

    rotation = quaternion_to_rotation_matrix(quaternion)
    attitude_skew = jnp.swapaxes(desired_rotation, -1, -2) @ rotation - (
        jnp.swapaxes(rotation, -1, -2) @ desired_rotation
    )
    attitude_error = 0.5 * jnp.stack(
        (attitude_skew[..., 2, 1], attitude_skew[..., 0, 2], attitude_skew[..., 1, 0]), axis=-1
    )
    angular_momentum = (inertia @ angular_velocity[..., None])[..., 0]
    gyroscopic = jnp.cross(angular_velocity, angular_momentum)
    torque = (
        -config.attitude_gain * attitude_error
        - config.angular_rate_gain * angular_velocity
        + gyroscopic
    )
    collective = jnp.sum(force_world * rotation[..., :, 2], axis=-1, keepdims=True)
    raw_wrench = jnp.concatenate((collective, torque), axis=-1)

    arm = jnp.reshape(jnp.asarray(actuator.arm_length, dtype=dtype), ())
    ratio = jnp.reshape(jnp.asarray(actuator.thrust_to_torque, dtype=dtype), ())
    mixing = jnp.asarray(actuator.mixing_matrix, dtype=dtype)
    lower = jnp.broadcast_to(jnp.asarray(actuator.thrust_min, dtype=dtype), (4,))
    upper = jnp.broadcast_to(jnp.asarray(actuator.thrust_max, dtype=dtype), (4,))
    raw_motor = wrench_to_motor_forces(raw_wrench, L=arm, thrust2torque=ratio, mixing_matrix=mixing)
    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    safe_half_width = jnp.where(half_width > 0, half_width, 1.0)
    if smooth_motor_bounds:
        bounded_motor = center + half_width * jnp.tanh((raw_motor - center) / safe_half_width)
    else:
        bounded_motor = jnp.clip(raw_motor, lower, upper)
    wrench = motor_forces_to_wrench(bounded_motor, L=arm, thrust2torque=ratio, mixing_matrix=mixing)
    reconstructed_motor = wrench_to_motor_forces(
        wrench, L=arm, thrust2torque=ratio, mixing_matrix=mixing
    )
    motor_scale = jnp.maximum(jnp.max(jnp.abs(bounded_motor), axis=-1), 1.0)
    allocation_error = jnp.max(jnp.abs(reconstructed_motor - bounded_motor), axis=-1) / motor_scale
    finite_values = (
        desired_acceleration,
        quaternion,
        angular_velocity,
        mass,
        gravity,
        inertia,
        arm,
        ratio,
        mixing,
        lower,
        upper,
        wrench,
    )
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in finite_values]))
    actuator_valid = (
        finite
        & (mass > 0)
        & jnp.all(jnp.linalg.eigvalsh(0.5 * (inertia + inertia.T)) > 0)
        & (arm > 0)
        & (ratio > 0)
        & jnp.all(lower >= 0)
        & jnp.all(lower < upper)
        & jnp.isfinite(allocation_error)
        & (allocation_error <= config.allocation_tolerance)
    )
    input_valid = force_valid & heading_valid & actuator_valid
    nan_wrench = jnp.full_like(wrench, jnp.nan)
    return QuadWrenchCommand(
        wrench=jnp.where(input_valid[..., None], wrench, nan_wrench),
        desired_acceleration=desired_acceleration,
        desired_rotation=desired_rotation,
        raw_wrench=raw_wrench,
        raw_motor_forces=raw_motor,
        bounded_motor_forces=bounded_motor,
        input_valid=input_valid,
    )


def shared_quad_fallback_wrenches(
    params: SharedActorParams,
    spec: SharedActorSpec,
    states: Array,
    scenarios: CircleScenarioBatch,
    model: VersionAModel,
    actuator: VersionAActuator,
    *,
    elapsed: Array,
    horizon_duration: float,
    policy_gain: float,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
) -> QuadWrenchCommand:
    """Evaluate the shared task-agnostic library for full states shaped ``(K, B, 13)``."""
    quad_config.validate()
    if states.ndim != 3 or states.shape[-1] != 13:
        raise ValueError("states must have shape (K, B, 13)")
    translational_state = jnp.concatenate((states[..., :3], states[..., 7:10]), axis=-1)
    desired_acceleration = shared_fallback_actions(
        params,
        spec,
        translational_state,
        scenarios,
        elapsed=elapsed,
        horizon_duration=horizon_duration,
        policy_gain=policy_gain,
        action_limit=quad_config.acceleration_limit,
        config=actor_config,
    )
    return acceleration_to_feasible_wrench(
        desired_acceleration,
        states[..., 3:7],
        states[..., 10:13],
        model,
        actuator,
        quad_config,
        smooth_motor_bounds=True,
    )


def waypoint_nominal_wrench(
    state: Array,
    target_position: Array,
    target_velocity: Array,
    model: VersionAModel,
    actuator: VersionAActuator,
    config: QuadPolicyConfig,
    *,
    position_gain: float = 2.0,
    velocity_gain: float = 1.4,
) -> QuadWrenchCommand:
    """Separate waypoint task controller; targets never enter fallback-library observations."""
    config.validate()
    if state.shape != (13,) or target_position.shape != (3,) or target_velocity.shape != (3,):
        raise ValueError("state must be (13,) and waypoint position/velocity must be (3,)")
    if not all(math.isfinite(value) and value > 0 for value in (position_gain, velocity_gain)):
        raise ValueError("waypoint gains must be finite and positive")
    raw_acceleration = position_gain * (target_position - state[:3]) + velocity_gain * (
        target_velocity - state[7:10]
    )
    desired_acceleration = config.acceleration_limit * jnp.tanh(
        raw_acceleration / config.acceleration_limit
    )
    return acceleration_to_feasible_wrench(
        desired_acceleration,
        state[3:7],
        state[10:13],
        model,
        actuator,
        config,
        smooth_motor_bounds=False,
    )


__all__ = [
    "QuadPolicyConfig",
    "QuadWrenchCommand",
    "acceleration_to_feasible_wrench",
    "shared_quad_fallback_wrenches",
    "waypoint_nominal_wrench",
]
