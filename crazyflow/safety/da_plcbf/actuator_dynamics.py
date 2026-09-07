"""Differentiable nominal-effort actuator surrogate and shared bounded allocation.

See ``docs/da_plcbf_actuator_contract.md`` for units, clocks and model limitations.
A1 is algebraic; A2 strictly requires positive time constants. Neither physical
transition clips inputs or repairs an invalid state.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.direct_wrench import (
    ControlAffineTerms,
    control_affine_terms,
    direct_wrench_dynamics,
    flatten_derivative,
)

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel


BODY_STATE_SIZE = 13
AUGMENTED_STATE_SIZE = 17
COMMAND_SIZE = 4


class ActuatorModel(NamedTuple):
    """One current model snapshot; command and internal-effort units are Newtons."""

    body: VersionAModel
    effectiveness: Array
    time_constants: Array
    command_lower: Array
    command_upper: Array
    force_to_wrench: Array


class AllocationResult(NamedTuple):
    """Bounded command and actual one-hold endpoint audit (static for A1)."""

    command: Array
    requested_command: Array
    achieved_wrench: Array
    wrench_residual: Array
    normalized_residual: Array
    saturation: Array
    valid: Array


def make_actuator_model(
    body: VersionAModel,
    *,
    L: float,
    thrust2torque: float,
    mixing_matrix: Any,
    thrust_min: Any,
    thrust_max: Any,
    time_constants: Any,
    effectiveness: Any = (1.0, 1.0, 1.0, 1.0),
) -> ActuatorModel:
    """Construct and validate an immutable model outside traced computation."""
    mixing = np.asarray(mixing_matrix, dtype=float)
    if mixing.shape != (3, 4):
        raise ValueError("mixing_matrix must have shape (3, 4)")
    if not math.isfinite(L) or L <= 0 or not math.isfinite(thrust2torque) or thrust2torque <= 0:
        raise ValueError("L and thrust2torque must be finite and positive")
    matrix = np.vstack((np.ones(4), np.array([L, L, thrust2torque])[:, None] * mixing))
    model = ActuatorModel(
        body,
        jnp.asarray(np.broadcast_to(effectiveness, (4,))),
        jnp.asarray(np.broadcast_to(time_constants, (4,))),
        jnp.asarray(np.broadcast_to(thrust_min, (4,))),
        jnp.asarray(np.broadcast_to(thrust_max, (4,))),
        jnp.asarray(matrix),
    )
    validate_actuator_model(model)
    return model


def validate_actuator_model(model: ActuatorModel) -> None:
    """Reject invalid parameters and a mixer incompatible with the exact allocator."""
    for name in ("effectiveness", "time_constants", "command_lower", "command_upper"):
        value = np.asarray(getattr(model, name))
        if value.shape != (4,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be a finite four-vector")
    if np.any(np.asarray(model.effectiveness) <= 0) or np.any(np.asarray(model.effectiveness) > 1):
        raise ValueError("partial effectiveness must be in (0, 1]")
    if np.any(np.asarray(model.time_constants) <= 0):
        raise ValueError(
            "time_constants must be strictly positive; use algebraic functions for zero lag"
        )
    if np.any(np.asarray(model.command_lower) < 0) or np.any(
        np.asarray(model.command_upper) <= np.asarray(model.command_lower)
    ):
        raise ValueError("command bounds must satisfy 0 <= lower < upper")
    matrix = np.asarray(model.force_to_wrench)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("force_to_wrench must be a finite (4, 4) matrix")
    scales = np.abs(matrix[:, 0])
    if np.any(scales <= 0) or not np.allclose(matrix[0], 1):
        raise ValueError("force map must use collective thrust and nonzero torque scales")
    normalized = matrix / scales[:, None]
    if not np.allclose(normalized.T @ normalized, 4 * np.eye(4), atol=2e-6):
        raise ValueError("bounded inverse allocator requires the accepted orthogonal mixer")


def _check_shapes(state: Array, command: Array, size: int) -> None:
    if state.shape[-1:] != (size,) or command.shape != (*state.shape[:-1], 4):
        raise ValueError(f"state/command shapes must be (..., {size}) and (..., 4)")


def _check_clock(dt: float, substeps: int) -> None:
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps <= 0:
        raise ValueError("substeps must be a positive integer")


def actual_motor_forces(effort: Array, model: ActuatorModel) -> Array:
    """Apply effectiveness once to internal effort (A2) or command (A1)."""
    if effort.shape[-1:] != (4,):
        raise ValueError("effort must have shape (..., 4)")
    return effort * model.effectiveness


def applied_wrench(effort: Array, model: ActuatorModel) -> Array:
    """Map nominal-equivalent effort to actual thrust and body torque."""
    forces = actual_motor_forces(effort, model)
    # Explicit reduction avoids TF32 lowering for this small condition-sensitive map.
    return jnp.sum(model.force_to_wrench * forces[..., None, :], axis=-1)


def exact_effort_update(state: Array, command: Array, tau: Array, dt: float) -> Array:
    """Exact stable held-command motor endpoint for strictly positive tau."""
    if state.shape[-1:] != (4,) or state.shape != command.shape:
        raise ValueError("motor state and command must have matching (..., 4) shapes")
    gain = -jnp.expm1(-dt / tau)
    return state + gain * (command - state)


def _body_derivative(state: Array, wrench: Array, model: ActuatorModel) -> Array:
    body = model.body
    return flatten_derivative(
        direct_wrench_dynamics(
            state[..., :3],
            state[..., 3:7],
            state[..., 7:10],
            state[..., 10:13],
            wrench,
            mass=body.mass,
            gravity_vec=body.gravity_vec,
            J=body.inertia,
            J_inv=body.inertia_inv,
            drag_matrix=body.drag_matrix,
            wind_velocity=body.wind_velocity,
            external_force=body.external_force,
            external_torque=body.external_torque,
        )
    )


def algebraic_dynamics(state: Array, command: Array, model: ActuatorModel) -> Array:
    """A1 effectiveness-only 13-state derivative; no inverse time constants."""
    _check_shapes(state, command, BODY_STATE_SIZE)
    return _body_derivative(state, applied_wrench(command, model), model)


def augmented_dynamics(state: Array, command: Array, model: ActuatorModel) -> Array:
    """A2 continuous 17-state derivative with command acting only on motor rates."""
    _check_shapes(state, command, AUGMENTED_STATE_SIZE)
    effort = state[..., 13:17]
    body = _body_derivative(state[..., :13], applied_wrench(effort, model), model)
    return jnp.concatenate((body, (command - effort) / model.time_constants), axis=-1)


def _body_affine_terms(state: Array, model: ActuatorModel) -> ControlAffineTerms:
    body = model.body
    return control_affine_terms(
        state[..., :3],
        state[..., 3:7],
        state[..., 7:10],
        state[..., 10:13],
        mass=body.mass,
        gravity_vec=body.gravity_vec,
        J=body.inertia,
        J_inv=body.inertia_inv,
        drag_matrix=body.drag_matrix,
        wind_velocity=body.wind_velocity,
        external_force=body.external_force,
        external_torque=body.external_torque,
    )


def algebraic_control_affine_terms(state: Array, model: ActuatorModel) -> ControlAffineTerms:
    """A1 command matrix is G_body B diag(eta), with 13-state body drift."""
    if state.shape[-1:] != (13,):
        raise ValueError("algebraic state must have shape (..., 13)")
    terms = _body_affine_terms(state, model)
    matrix = jnp.einsum(
        "...ij,jk->...ik",
        terms.input_matrix,
        model.force_to_wrench * model.effectiveness,
        precision=jax.lax.Precision.HIGHEST,
    )
    return ControlAffineTerms(terms.drift, matrix)


def augmented_control_affine_terms(state: Array, model: ActuatorModel) -> ControlAffineTerms:
    """Return f_a=[body_dot(x,s),-s/tau], G_a=[0;diag(1/tau)]."""
    if state.shape[-1:] != (17,):
        raise ValueError("augmented state must have shape (..., 17)")
    effort = state[..., 13:17]
    drift = jnp.concatenate(
        (
            _body_derivative(state[..., :13], applied_wrench(effort, model), model),
            -effort / model.time_constants,
        ),
        axis=-1,
    )
    matrix = jnp.zeros((*state.shape, 4), dtype=state.dtype)
    matrix = matrix.at[..., 13:17, :].set(jnp.diag(1 / model.time_constants))
    return ControlAffineTerms(drift, matrix)


def _normalize_body_quaternion(state: Array) -> Array:
    quat = state[..., 3:7]
    return state.at[..., 3:7].set(quat / jnp.linalg.norm(quat, axis=-1, keepdims=True))


def _body_rk4(
    state: Array,
    start_wrench: Array,
    half_wrench: Array,
    end_wrench: Array,
    model: ActuatorModel,
    dt: float,
) -> Array:
    k1 = _body_derivative(state, start_wrench, model)
    k2 = _body_derivative(state + 0.5 * dt * k1, half_wrench, model)
    k3 = _body_derivative(state + 0.5 * dt * k2, half_wrench, model)
    k4 = _body_derivative(state + dt * k3, end_wrench, model)
    return _normalize_body_quaternion(state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6)


def algebraic_step(
    state: Array, command: Array, model: ActuatorModel, dt: float, *, substeps: int = 1
) -> Array:
    """Integrate the separate 13-state A1 model with held command and RK4."""
    _check_shapes(state, command, BODY_STATE_SIZE)
    _check_clock(dt, substeps)
    wrench = applied_wrench(command, model)

    def step(_: int, current: Array) -> Array:
        return _body_rk4(current, wrench, wrench, wrench, model, dt / substeps)

    return jax.lax.fori_loop(0, substeps, step, state)


def effort_lag_step(
    state: Array, command: Array, model: ActuatorModel, dt: float, *, substeps: int = 1
) -> Array:
    """Integrate A2 with exact evolving effort at every body RK4 stage."""
    _check_shapes(state, command, AUGMENTED_STATE_SIZE)
    _check_clock(dt, substeps)
    h = dt / substeps

    def step(_: int, current: Array) -> Array:
        effort = current[..., 13:17]
        half = exact_effort_update(effort, command, model.time_constants, h / 2)
        end = exact_effort_update(effort, command, model.time_constants, h)
        body = _body_rk4(
            current[..., :13],
            applied_wrench(effort, model),
            applied_wrench(half, model),
            applied_wrench(end, model),
            model,
            h,
        )
        return jnp.concatenate((body, end), axis=-1)

    return jax.lax.fori_loop(0, substeps, step, state)


def wrench_to_actual_forces(wrench: Array, model: ActuatorModel) -> Array:
    """Invert the accepted orthogonal mixing map without effectiveness or clipping."""
    if wrench.shape[-1:] != (4,):
        raise ValueError("wrench must have shape (..., 4)")
    scales = jnp.abs(model.force_to_wrench[:, 0])
    inverse = (model.force_to_wrench / scales[:, None] ** 2).T / 4
    return jnp.sum(inverse * wrench[..., None, :], axis=-1)


def normalized_command_metric(model: ActuatorModel) -> Array:
    """Positive-definite quadratic metric in dimensionless command coordinates."""
    return jnp.diag(1 / (model.command_upper - model.command_lower) ** 2)


def _allocation_result(
    requested: Array,
    desired_wrench: Array,
    endpoint_effort: Array,
    command: Array,
    model: ActuatorModel,
) -> AllocationResult:
    achieved = applied_wrench(endpoint_effort, model)
    residual = achieved - desired_wrench
    scales = jnp.abs(model.force_to_wrench[:, 0]) * jnp.sum(
        model.command_upper - model.command_lower
    )
    normalized = jnp.linalg.norm(residual / scales, axis=-1)
    saturation = (requested < model.command_lower) | (requested > model.command_upper)
    valid = (
        jnp.all(jnp.isfinite(requested), axis=-1)
        & jnp.all(jnp.isfinite(achieved), axis=-1)
        & jnp.all(jnp.isfinite(desired_wrench), axis=-1)
        & jnp.all(jnp.isfinite(command), axis=-1)
    )
    return AllocationResult(command, requested, achieved, residual, normalized, saturation, valid)


def allocate_wrench(
    wrench: Array, motor_state: Array, model: ActuatorModel, hold_dt: float, *, mode: str = "F2"
) -> AllocationResult:
    """Map desired wrench to F0/F1/F2 bounded commands; audit actual A2 endpoint.

    Infeasible F1 and F2 inverse commands are projected to bounds. Orthogonality
    makes this the exact normalized wrench least-squares solution, as derived in
    the contract. F2 matches a reachable endpoint, not the full ideal trajectory.
    """
    _check_clock(hold_dt, 1)
    if motor_state.shape != wrench.shape or wrench.shape[-1:] != (4,):
        raise ValueError("wrench and motor_state must have matching (..., 4) shapes")
    forces = wrench_to_actual_forces(wrench, model)
    if mode == "F0":
        requested = forces
    elif mode == "F1":
        requested = forces / model.effectiveness
    elif mode == "F2":
        target = forces / model.effectiveness
        gain = -jnp.expm1(-hold_dt / model.time_constants)
        requested = motor_state + (target - motor_state) / gain
    else:
        raise ValueError("mode must be F0, F1 or F2")
    command = jnp.clip(requested, model.command_lower, model.command_upper)
    endpoint = exact_effort_update(motor_state, command, model.time_constants, hold_dt)
    return _allocation_result(requested, wrench, endpoint, command, model)


def allocate_algebraic_wrench(
    wrench: Array, model: ActuatorModel, *, mode: str = "F1"
) -> AllocationResult:
    """Bounded A1 adapter and algebraic achieved-wrench diagnostics."""
    forces = wrench_to_actual_forces(wrench, model)
    if mode == "F0":
        requested = forces
    elif mode in ("F1", "F2"):
        requested = forces / model.effectiveness
    else:
        raise ValueError("mode must be F0, F1 or F2")
    command = jnp.clip(requested, model.command_lower, model.command_upper)
    return _allocation_result(requested, wrench, command, command, model)


def actuator_state_valid(state: Array, command: Array, model: ActuatorModel) -> Array:
    """Audit finite augmented state and physical command/state bounds without clipping."""
    _check_shapes(state, command, AUGMENTED_STATE_SIZE)
    effort = state[..., 13:17]
    return (
        jnp.all(jnp.isfinite(state), axis=-1)
        & jnp.all(jnp.isfinite(command), axis=-1)
        & (jnp.abs(jnp.linalg.norm(state[..., 3:7], axis=-1) - 1) <= 2e-4)
        & jnp.all((effort >= 0) & (effort <= model.command_upper), axis=-1)
        & jnp.all((command >= model.command_lower) & (command <= model.command_upper), axis=-1)
    )


def fit_native_time_constant(drone: str = "cf21B_500") -> dict[str, Any]:
    """Derive local thrust-coordinate poles from native RPM coefficients, not hardware data."""
    params = load_params(drone)
    thrust = np.asarray(params["rpm2thrust"], dtype=float)
    coef = np.asarray(params["rotor_dyn_coef"], dtype=float)
    hover = params["mass"] * np.linalg.norm(params["gravity_vec"]) / 4
    efforts = [float(params["thrust_min"]), float(hover), float(params["thrust_max"])]
    points = []
    for effort in efforts:
        a, b, c = thrust
        rpm = (-b + math.sqrt(b * b + 4 * c * (effort - a))) / (2 * c)
        pole_up = coef[0] + 2 * coef[1] * rpm
        pole_down = coef[2] + 2 * coef[3] * rpm
        points.append(
            {
                "effort_N": effort,
                "rpm": float(rpm),
                "pole_up_per_s": float(pole_up),
                "pole_down_per_s": float(pole_down),
                "tau_up_s": float(1 / pole_up),
                "tau_down_s": float(1 / pole_down),
                "symmetric_tau_s": float(2 / (pole_up + pole_down)),
                "thrust_roundtrip_residual_N": float(a + b * rpm + c * rpm * rpm - effort),
            }
        )
    parameter_path = Path(__file__).resolve().parents[2] / "drones" / "params.toml"
    return {
        "schema": "da_plcbf_native_local_pole_fit_v1",
        "drone": drone,
        "source": "crazyflow/drones/params.toml",
        "source_sha256": hashlib.sha256(parameter_path.read_bytes()).hexdigest(),
        "method": "RPM pole a+2*b*rpm; thrust-coordinate invariant; mean directional pole",
        "qualification": "Repository model reduction with TODOs; not hardware validation",
        "rpm2thrust": thrust.tolist(),
        "rotor_dyn_coef": coef.tolist(),
        "nominal_tau": points[1]["symmetric_tau_s"],
        "operating_points": points,
    }


def hover_authority(model: ActuatorModel) -> dict[str, Any]:
    """Solve coupled level-hover trim and fixed-collective pure-axis torque headroom.

    This specifies the force/torque trim with no external force, wind or torque.
    An infeasible unique command proves only that this specified trim is infeasible.
    Maneuver witness classification is intentionally independent.
    """
    validate_actuator_model(model)
    body = model.body
    if any(
        np.any(np.asarray(value) != 0)
        for value in (body.wind_velocity, body.external_force, body.external_torque)
    ) or np.any(np.asarray(body.gravity_vec)[:2] != 0):
        raise ValueError("level-hover authority requires zero disturbances and vertical gravity")
    matrix = np.asarray(model.force_to_wrench, dtype=float) * np.asarray(model.effectiveness)
    gravity = -float(np.asarray(body.gravity_vec)[2])
    desired = np.array([float(np.asarray(body.mass)) * gravity, 0.0, 0.0, 0.0])
    inverse = np.linalg.inv(matrix)
    command = inverse @ desired
    lower, upper = np.asarray(model.command_lower), np.asarray(model.command_upper)
    feasible = bool(np.all(command >= lower - 1e-9) and np.all(command <= upper + 1e-9))
    headroom = np.minimum(command - lower, upper - command)
    torque_limits = []
    for axis in range(1, 4):
        row = []
        for sign in (-1, 1):
            direction = sign * inverse[:, axis]
            distances = np.where(direction > 0, upper - command, command - lower)
            row.append(float(np.min(distances / np.abs(direction))) if feasible else None)
        torque_limits.append(row)
    return {
        "trim_feasible": feasible,
        "trim_classification": "trim_feasible" if feasible else "proven_infeasible",
        "infeasibility_scope": "specified level zero-rate hover; not all flight",
        "maneuver_classification": "witness_not_found",
        "required_command_N": command.tolist(),
        "headroom_N": headroom.tolist(),
        "trim_wrench_residual": (matrix @ command - desired).tolist(),
        "torque_limits_negative_positive_Nm": torque_limits,
        "effectiveness": np.asarray(model.effectiveness).tolist(),
        "time_constants_s": np.asarray(model.time_constants).tolist(),
    }
