"""Fixed-shape direct-wrench rollouts for the shared quadrotor fallback actor."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.direct_wrench import direct_wrench_dynamics
from crazyflow.safety.da_plcbf.quad_policy import shared_quad_fallback_wrenches

if TYPE_CHECKING:
    from collections.abc import Callable

    from crazyflow.safety.da_plcbf.actor import (
        SharedActorConfig,
        SharedActorParams,
        SharedActorSpec,
    )
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
    from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


class QuadRolloutBatch(NamedTuple):
    """Full state/wrench/actor traces with policy and scenario leading axes."""

    states: Array
    wrenches: Array
    desired_accelerations: Array
    raw_motor_forces: Array
    bounded_motor_forces: Array
    policy_valid: Array


def _integrate_body_rate_xyzw(quaternion: Array, body_rate: Array, dt: float) -> Array:
    """Right-compose the exact constant-body-rate quaternion exponential.

    This is equivalent to ``Rotation.from_quat(q) * Rotation.from_rotvec(omega * dt)`` but avoids
    a JAX 0.11.1 CUDA lowering bug inside batched ``lax.scan``.  ``sinc`` supplies the analytic
    zero-rate limit without a divide-by-zero branch.
    """
    quaternion = quaternion / jnp.linalg.norm(quaternion, axis=-1, keepdims=True)
    # Differentiate squared angle directly at zero rate. Taking norm(omega) first creates a
    # NaN norm derivative at omega=0 even if a later where chooses the analytic small-angle limit.
    angle_squared = dt**2 * jnp.sum(body_rate * body_rate, axis=-1, keepdims=True)
    small = angle_squared <= (32.0 * jnp.finfo(body_rate.dtype).eps) ** 2
    safe_angle = jnp.sqrt(jnp.where(small, 1.0, angle_squared))
    half_angle = 0.5 * safe_angle
    small_sinc = 1.0 - angle_squared / 24.0 + angle_squared * angle_squared / 1920.0
    half_sinc = jnp.where(small, small_sinc, 2.0 * jnp.sin(half_angle) / safe_angle)
    vector_scale = 0.5 * dt * half_sinc
    delta_vector = body_rate * vector_scale
    small_cosine = 1.0 - angle_squared / 8.0 + angle_squared * angle_squared / 384.0
    delta_scalar = jnp.where(small, small_cosine, jnp.cos(half_angle))
    vector = quaternion[..., :3]
    scalar = quaternion[..., 3:4]
    composed_vector = (
        scalar * delta_vector + delta_scalar * vector + jnp.linalg.cross(vector, delta_vector)
    )
    composed_scalar = scalar * delta_scalar - jnp.sum(vector * delta_vector, axis=-1, keepdims=True)
    composed = jnp.concatenate((composed_vector, composed_scalar), axis=-1)
    return composed / jnp.linalg.norm(composed, axis=-1, keepdims=True)


def direct_wrench_symplectic_step(
    state: Array, wrench: Array, model: VersionAModel, dt: float
) -> Array:
    """Advance a flattened Version-A state using Crazyflow's symplectic ordering."""
    if state.shape[-1:] != (13,) or wrench.shape != (*state.shape[:-1], 4):
        raise ValueError("state/wrench shapes must be (..., 13) and (..., 4)")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    derivative = direct_wrench_dynamics(
        state[..., :3],
        state[..., 3:7],
        state[..., 7:10],
        state[..., 10:13],
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
    next_velocity = state[..., 7:10] + derivative.vel_dot * dt
    next_angular_velocity = state[..., 10:13] + derivative.ang_vel_dot * dt
    next_position = state[..., :3] + next_velocity * dt
    # Flush subnormal values for the backend while retaining the physical derivative at zero.
    safe_angular_velocity = next_angular_velocity + jax.lax.stop_gradient(
        jnp.where(
            jnp.abs(next_angular_velocity) < jnp.finfo(state.dtype).smallest_normal,
            -next_angular_velocity,
            0.0,
        )
    )
    next_quaternion = _integrate_body_rate_xyzw(state[..., 3:7], safe_angular_velocity, dt)
    return jnp.concatenate(
        (next_position, next_quaternion, next_velocity, next_angular_velocity), axis=-1
    )


def zero_order_hold_rollout(
    initial_states: Array,
    command_function: Callable[[Array, Array], tuple[Array, ...]],
    model: VersionAModel,
    *,
    dt: float,
    horizon: int,
    command_hold_steps: int = 1,
) -> tuple[Array, tuple[Array, ...]]:
    """Integrate H steps while evaluating feedback only at declared command boundaries.

    The command callback receives the current state and the integration-step index, and returns
    a tuple whose first element is the wrench. Remaining elements are command diagnostics. The
    returned future states and command traces have a leading time axis of exactly ``horizon``.
    A partial final hold is truncated without extending the physical rollout horizon. The caller
    retains the original skill anchor; the step index advances rather than restarting its phase.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if (
        isinstance(command_hold_steps, bool)
        or not isinstance(command_hold_steps, int)
        or not 1 <= command_hold_steps <= horizon
    ):
        raise ValueError("command_hold_steps must be an integer in [1, horizon]")

    def hold(state: Array, boundary: Array) -> tuple[Array, tuple[Array, tuple[Array, ...]]]:
        command = command_function(state, boundary * command_hold_steps)

        def integrate(current: Array, _: None) -> tuple[Array, Array]:
            following = direct_wrench_symplectic_step(current, command[0], model, dt)
            return following, following

        # The physical hold usually contains two integration steps. Unroll this small
        # inner loop to avoid serial GPU loop overhead inside the command-horizon scan;
        # every integration state and held command is retained in its original order.
        final, future = jax.lax.scan(
            integrate, state, None, length=command_hold_steps, unroll=command_hold_steps
        )
        repeated = tuple(
            jnp.broadcast_to(value, (command_hold_steps, *value.shape)) for value in command
        )
        return final, (future, repeated)

    _, (future, commands) = jax.lax.scan(
        hold, initial_states, jnp.arange(math.ceil(horizon / command_hold_steps))
    )

    def truncate(value: Array) -> Array:
        return value.reshape((-1, *value.shape[2:]))[:horizon]

    return truncate(future), tuple(truncate(value) for value in commands)


def rollout_shared_quad_library(
    params: SharedActorParams,
    spec: SharedActorSpec,
    initial_states: Array,
    scenarios: CircleScenarioBatch,
    model: VersionAModel,
    actuator: VersionAActuator,
    *,
    dt: float,
    horizon: int,
    policy_gain: float,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    command_hold_steps: int = 1,
) -> QuadRolloutBatch:
    """Roll out every policy/scenario through the differentiable direct-wrench plant."""
    if initial_states.ndim != 2 or initial_states.shape[-1] != 13:
        raise ValueError("initial_states must have shape (B, 13)")
    if initial_states.shape[0] != scenarios.obstacle_centers.shape[0]:
        raise ValueError("initial state and scenario batches must match")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not math.isfinite(policy_gain) or policy_gain <= 0:
        raise ValueError("policy_gain must be finite and positive")
    policy_count = spec.base_codes.shape[0]
    current = jnp.broadcast_to(initial_states[None, ...], (policy_count, *initial_states.shape))
    horizon_duration = horizon * dt

    def command_at_boundary(state: Array, step_index: Array) -> tuple[Array, ...]:
        command = shared_quad_fallback_wrenches(
            params,
            spec,
            state,
            scenarios,
            model,
            actuator,
            elapsed=step_index * dt,
            horizon_duration=horizon_duration,
            policy_gain=policy_gain,
            actor_config=actor_config,
            quad_config=quad_config,
        )
        return (
            command.wrench,
            command.desired_acceleration,
            command.raw_motor_forces,
            command.bounded_motor_forces,
            command.input_valid,
        )

    future, outputs = zero_order_hold_rollout(
        current,
        command_at_boundary,
        model,
        dt=dt,
        horizon=horizon,
        command_hold_steps=command_hold_steps,
    )
    future, wrench, acceleration, raw_motor, bounded_motor, valid = (
        jnp.moveaxis(value, 0, 2) for value in (future, *outputs)
    )
    states = jnp.concatenate((current[:, :, None, :], future), axis=2)
    return QuadRolloutBatch(states, wrench, acceleration, raw_motor, bounded_motor, valid)


__all__ = [
    "QuadRolloutBatch",
    "direct_wrench_symplectic_step",
    "rollout_shared_quad_library",
    "zero_order_hold_rollout",
]
