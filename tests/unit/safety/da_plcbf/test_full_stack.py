from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.control import Control
from crazyflow.dynamics import Dynamics
from crazyflow.safety.da_plcbf.discrete_filter import discrete_nonlinear_plcbf_filter
from crazyflow.safety.da_plcbf.full_stack import (
    build_unclipped_full_stack_step,
    full_stack_action_evaluation,
    rollout_full_stack_force_torque,
)
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator


def _sim() -> Sim:
    return Sim(
        dynamics=Dynamics.first_principles,
        control=Control.force_torque,
        integrator=Integrator.symplectic_euler,
        freq=500,
        force_torque_freq=500,
        device="cpu",
        enable_mjx=False,
    )


def _hover_command(sim: Sim) -> jax.Array:
    mass = sim.data.params.mass[0, 0, 0]
    gravity = -sim.data.params.gravity_vec[2]
    return jnp.array([mass * gravity, 0.0, 0.0, 0.0])


def _airborne_data(sim: Sim) -> object:
    hover_motor_force = _hover_command(sim)[0] / 4
    rpm2thrust = sim.data.params.rpm2thrust
    hover_rpm = (
        -rpm2thrust[1]
        + jnp.sqrt(rpm2thrust[1] ** 2 - 4 * rpm2thrust[2] * (rpm2thrust[0] - hover_motor_force))
    ) / (2 * rpm2thrust[2])
    states = sim.data.states.replace(
        pos=sim.data.states.pos.at[0, 0, 2].set(1.0),
        rotor_vel=jnp.full_like(sim.data.states.rotor_vel, hover_rpm),
    )
    return sim.data.replace(states=states)


def test_unclipped_step_preserves_floor_crossing_that_normal_sim_hides() -> None:
    sim = _sim()
    falling = sim.data.replace(
        states=sim.data.states.replace(
            pos=sim.data.states.pos.at[0, 0, 2].set(-0.2),
            vel=sim.data.states.vel.at[0, 0, 2].set(-1.0),
        )
    )
    unclipped = build_unclipped_full_stack_step(sim)

    safety_next = unclipped(falling)
    normal_next = sim._step(falling, n_steps=1)

    assert safety_next.states.pos[0, 0, 2] < -0.2
    np.testing.assert_allclose(normal_next.states.pos[0, 0, 2], -0.001, atol=1e-7)


def test_full_stack_rollout_matches_independent_staged_crazyflow_steps() -> None:
    sim = _sim()
    data = _airborne_data(sim)
    command = _hover_command(sim)
    one_step = build_unclipped_full_stack_step(sim)

    def state_margin(current: object) -> jax.Array:
        return current.states.pos[0, 0, 2] - 0.1

    audited = rollout_full_stack_force_torque(data, command, one_step, state_margin, n_substeps=5)
    independent = data
    from crazyflow.sim import functional as functional  # local import keeps the oracle explicit

    independent = functional.force_torque_control(independent, command[None, None, :])
    for _ in range(5):
        independent = one_step(independent)

    np.testing.assert_allclose(audited.final_data.states.pos, independent.states.pos, atol=1e-7)
    np.testing.assert_allclose(audited.final_data.states.vel, independent.states.vel, atol=1e-7)
    np.testing.assert_allclose(audited.raw_motor_forces, command[0] / 4, rtol=1e-6)
    np.testing.assert_allclose(
        audited.desired_motor_forces, audited.raw_motor_forces, rtol=2e-5, atol=1e-8
    )
    assert bool(audited.command_committed)
    assert audited.actuator_residual <= 2e-6
    assert audited.interval_margin > 0


def test_out_of_polytope_command_is_exposed_even_though_controller_clips_it() -> None:
    sim = _sim()
    data = _airborne_data(sim)
    command = jnp.array([10.0, 0.0, 0.0, 0.0])
    rollout = rollout_full_stack_force_torque(
        data,
        command,
        build_unclipped_full_stack_step(sim),
        lambda current: current.states.pos[0, 0, 2],
        n_substeps=2,
    )

    assert rollout.command_bound_residual > 0
    assert rollout.allocation_roundtrip_error > 0
    assert rollout.actuator_residual > 0
    assert np.all(
        np.asarray(rollout.desired_motor_forces)
        <= float(sim.data.controls.force_torque.params["thrust_max"]) + 1e-7
    )


def test_full_stack_evaluation_is_differentiable_and_uses_final_hard_value() -> None:
    sim = _sim()
    data = _airborne_data(sim)
    one_step = build_unclipped_full_stack_step(sim)

    def evaluate(command: jax.Array) -> object:
        return full_stack_action_evaluation(
            data,
            command,
            one_step,
            lambda current: current.states.pos[0, 0, 2] - 0.1,
            lambda current: current.states.pos[0, 0, 2] - 0.2,
            n_substeps=3,
        )

    command = _hover_command(sim)
    eager = evaluate(command)
    compiled = jax.jit(evaluate)(command)
    gradient = jax.grad(lambda candidate: evaluate(candidate).next_value)(command)

    np.testing.assert_allclose(compiled.next_value, eager.next_value, atol=1e-7)
    np.testing.assert_allclose(eager.next_value, eager.interval_margin - 0.1, atol=1e-7)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert gradient[0] > 0


def test_discrete_filter_rejects_command_that_only_looks_safe_after_internal_clipping() -> None:
    sim = _sim()
    data = _airborne_data(sim)
    one_step = build_unclipped_full_stack_step(sim)
    hover = _hover_command(sim)

    def evaluate(command: jax.Array) -> object:
        return full_stack_action_evaluation(
            data,
            command,
            one_step,
            lambda current: current.states.pos[0, 0, 2] - 0.1,
            lambda current: current.states.pos[0, 0, 2] - 0.2,
            n_substeps=3,
        )

    result = discrete_nonlinear_plcbf_filter(
        nominal_action=jnp.array([10.0, 0.0, 0.0, 0.0]),
        fallback_action=hover,
        action_lower=jnp.array([0.0, -1.0, -1.0, -1.0]),
        action_upper=jnp.array([10.0, 1.0, 1.0, 1.0]),
        weight=jnp.ones(4),
        trust_radius=jnp.full(4, 10.0),
        current_value=jnp.array(0.8),
        has_certificate=jnp.array(True),
        evaluate_action=evaluate,
        decay=0.99,
        tolerance=1e-5,
    )

    assert result.proposal_exact_residual >= -1e-5
    assert result.proposal_actuator_residual > 0
    assert not bool(result.proposal_accepted)
    assert bool(result.fallback_accepted)
    assert bool(result.used_fallback)
    assert not bool(result.degraded)
    np.testing.assert_allclose(result.action, hover, atol=1e-7)
