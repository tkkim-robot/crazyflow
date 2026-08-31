"""Exact Crazyflow full-stack transitions for the discrete DA-PLCBF path.

The adapter keeps force/torque allocation, command clipping, rotor-speed dynamics, rigid-body
dynamics, and the configured Crazyflow integrator in the differentiated transition.  It
intentionally excludes Crazyflow's convenience floor-position clamp: contact or crossing the floor
must remain visible to authoritative safety metrics rather than becoming an artificial recovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.control import Control
from crazyflow.safety.da_plcbf.direct_wrench import wrench_to_motor_forces
from crazyflow.safety.da_plcbf.discrete_filter import DiscreteActionEvaluation
from crazyflow.sim import functional as sim_functional

if TYPE_CHECKING:
    from collections.abc import Callable

    from crazyflow.sim.data import SimData
    from crazyflow.sim.sim import Sim


class FullStackCommandRollout(NamedTuple):
    """Full transition and actuator audit for one held force/torque command."""

    final_data: Any
    interval_margin: Array
    raw_motor_forces: Array
    desired_motor_forces: Array
    final_motor_forces: Array
    command_bound_residual: Array
    allocation_roundtrip_error: Array
    physical_upper_residual: Array
    actuator_residual: Array
    command_committed: Array


def build_unclipped_full_stack_step(sim: Sim) -> Callable[[SimData], SimData]:
    """Build a pure one-substep function from ``sim`` without the floor clamp stage.

    The controller, dynamics, and integrator functions are the exact configured Crazyflow pipeline
    objects. Only ``clip_floor_pos`` is omitted. The returned transition does not mutate ``sim``.
    """
    if sim.control != Control.force_torque:
        raise ValueError("the full-stack force/torque adapter requires Control.force_torque")
    if "integration" not in sim.step_pipeline:
        raise ValueError("the Crazyflow pipeline has no integration stage")
    pipeline = tuple(
        function for name, function in sim.step_pipeline.items() if name != "clip_floor_pos"
    )

    @jax.jit
    def one_step(data: SimData) -> SimData:
        for function in pipeline:
            data = function(data)
        return data.replace(core=data.core.replace(mjx_synced=False, mjx_collision_synced=False))

    return one_step


def _motor_forces_from_rpm(rotor_rpm: Array, rpm2thrust: Array) -> Array:
    """Evaluate Crazyflow's physical thrust polynomial without clipping."""
    return rpm2thrust[0] + rpm2thrust[1] * rotor_rpm + rpm2thrust[2] * rotor_rpm**2


def rollout_full_stack_force_torque(
    data: SimData,
    command: Array,
    one_step: Callable[[SimData], SimData],
    state_margin: Callable[[SimData], Array],
    *,
    n_substeps: int,
) -> FullStackCommandRollout:
    """Hold one desired wrench through the complete nonlinear stack for fixed substeps.

    This adapter currently accepts one world and one drone, matching one runtime filter decision.
    Batched policy/scenario evaluation is obtained by vmapping the complete function over separate
    ``SimData`` instances. The lower motor-thrust bound is enforced on the *desired allocation*;
    rotor lag is allowed to pass below it, while any physical upper-bound overshoot is rejected.
    """
    if data.core.n_worlds != 1 or data.core.n_drones != 1:
        raise ValueError("runtime full-stack filtering currently requires one world and one drone")
    if data.controls.mode != Control.force_torque or data.controls.force_torque is None:
        raise ValueError("SimData must use initialized force/torque control")
    if command.shape != (4,):
        raise ValueError("command must be [collective thrust, body torque xyz]")
    if isinstance(n_substeps, bool) or not isinstance(n_substeps, int) or n_substeps <= 0:
        raise ValueError("n_substeps must be a positive integer")

    controller_params = data.controls.force_torque.params
    dynamics_params = data.params
    raw_motor_forces = wrench_to_motor_forces(
        command,
        L=controller_params["L"],
        thrust2torque=controller_params["thrust2torque"],
        mixing_matrix=controller_params["mixing_matrix"],
    )
    lower = jnp.broadcast_to(controller_params["thrust_min"], raw_motor_forces.shape)
    upper = jnp.broadcast_to(controller_params["thrust_max"], raw_motor_forces.shape)
    command_bound_residual = jnp.max(
        jnp.maximum(raw_motor_forces - upper, lower - raw_motor_forces)
    )
    command_committed = jnp.all(sim_functional.controllable(data))
    staged = sim_functional.force_torque_control(data, command[None, None, :])
    initial_margin = state_margin(staged)
    if initial_margin.shape != ():
        raise ValueError("state_margin must return a scalar")

    def advance(carry: SimData, _: None) -> tuple[SimData, tuple[Array, Array]]:
        following = one_step(carry)
        margin = state_margin(following)
        motor_forces = _motor_forces_from_rpm(
            following.states.rotor_vel[0, 0], dynamics_params.rpm2thrust
        )
        return following, (margin, motor_forces)

    final_data, (substep_margins, physical_motor_forces) = jax.lax.scan(
        advance, staged, xs=None, length=n_substeps
    )
    desired_motor_forces = _motor_forces_from_rpm(
        final_data.controls.rotor_vel[0, 0], dynamics_params.rpm2thrust
    )
    allocation_roundtrip_error = jnp.max(jnp.abs(desired_motor_forces - raw_motor_forces))
    physical_upper_residual = jnp.max(physical_motor_forces - upper)
    actuator_residual = jnp.maximum(
        jnp.maximum(command_bound_residual, allocation_roundtrip_error), physical_upper_residual
    )
    finite = (
        jnp.all(jnp.isfinite(command))
        & jnp.all(jnp.isfinite(raw_motor_forces))
        & jnp.all(jnp.isfinite(desired_motor_forces))
        & jnp.all(jnp.isfinite(physical_motor_forces))
        & jnp.all(jnp.isfinite(substep_margins))
        & jnp.isfinite(initial_margin)
    )
    actuator_residual = jnp.where(
        finite & command_committed, actuator_residual, jnp.asarray(jnp.inf, command.dtype)
    )
    return FullStackCommandRollout(
        final_data=final_data,
        interval_margin=jnp.minimum(initial_margin, jnp.min(substep_margins)),
        raw_motor_forces=raw_motor_forces,
        desired_motor_forces=desired_motor_forces,
        final_motor_forces=physical_motor_forces[-1],
        command_bound_residual=command_bound_residual,
        allocation_roundtrip_error=allocation_roundtrip_error,
        physical_upper_residual=physical_upper_residual,
        actuator_residual=actuator_residual,
        command_committed=command_committed,
    )


def full_stack_action_evaluation(
    data: SimData,
    command: Array,
    one_step: Callable[[SimData], SimData],
    state_margin: Callable[[SimData], Array],
    certificate_value: Callable[[SimData], Array],
    *,
    n_substeps: int,
) -> DiscreteActionEvaluation:
    """Return exact evidence consumable by :func:`discrete_nonlinear_plcbf_filter`."""
    rollout = rollout_full_stack_force_torque(
        data, command, one_step, state_margin, n_substeps=n_substeps
    )
    next_value = certificate_value(rollout.final_data)
    if next_value.shape != ():
        raise ValueError("certificate_value must return one selected hard policy value")
    return DiscreteActionEvaluation(
        next_value=next_value,
        interval_margin=rollout.interval_margin,
        actuator_residual=rollout.actuator_residual,
        applied_action=rollout.final_motor_forces,
    )


__all__ = [
    "FullStackCommandRollout",
    "build_unclipped_full_stack_step",
    "full_stack_action_evaluation",
    "rollout_full_stack_force_torque",
]
