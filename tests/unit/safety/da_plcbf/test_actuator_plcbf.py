from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.actuator_dynamics import (
    ActuatorModel,
    augmented_dynamics,
    effort_lag_step,
    make_actuator_model,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import (
    ActuatorFilterConfig,
    ActuatorRollouts,
    actuator_plcbf_step,
    actuator_policy_certificates,
    check_actuator_hold,
    command_box_fraction,
    held_actuator_nodes,
    operational_node_margins,
    predictive_operational_rows,
)
from crazyflow.safety.da_plcbf.continuous_version_a import RuntimeObstacleTrajectories
from crazyflow.safety.da_plcbf.online_constant_wind import build_cf21b_version_a_resources
from crazyflow.safety.da_plcbf.version_a_barriers import RigidBodySafetySet

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with jax.enable_x64():
        yield


def _resources() -> tuple[jax.Array, ActuatorModel, RigidBodySafetySet]:
    raw = load_params("cf21B_500")
    body, _ = build_cf21b_version_a_resources(dtype=jnp.float64)
    model = make_actuator_model(
        body,
        L=raw["L"],
        thrust2torque=raw["thrust2torque"],
        mixing_matrix=raw["mixing_matrix"],
        thrust_min=raw["thrust_min"],
        thrust_max=raw["thrust_max"],
        time_constants=(0.06, 0.08, 0.09, 0.07),
    )
    hover = float(body.mass * -body.gravity_vec[2] / 4)
    state = jnp.asarray(
        [0, 0, 1.4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, *([hover] * 4)], dtype=jnp.float64
    )
    safety = RigidBodySafetySet(
        jnp.empty((0, 3)),
        jnp.empty((0,)),
        jnp.empty((0,), dtype=bool),
        jnp.asarray([-5.0, -5.0, 0.15]),
        jnp.asarray([5.0, 5.0, 4.0]),
        jnp.asarray(3.5),
        jnp.asarray(12.0),
        jnp.asarray(0.9),
    )
    return state, model, safety


def _obstacles(
    config: ActuatorFilterConfig, *, moving: bool = False
) -> RuntimeObstacleTrajectories:
    time = jnp.arange(config.horizon + 1) * config.dt
    velocity = jnp.asarray([[0.17, -0.23, 0.11]]) if moving else jnp.zeros((1, 3))
    centers = jnp.asarray([1.7, 0.6, 1.8])[None, None] + time[:, None, None] * velocity[None]
    return RuntimeObstacleTrajectories(
        centers, jnp.asarray([0.3]), jnp.ones((config.horizon + 1, 1), dtype=bool), velocity
    )


def _empty_obstacles(config: ActuatorFilterConfig) -> RuntimeObstacleTrajectories:
    return RuntimeObstacleTrajectories(
        jnp.empty((config.horizon + 1, 0, 3)),
        jnp.empty((0,)),
        jnp.empty((config.horizon + 1, 0), dtype=bool),
    )


def _rollout(
    config: ActuatorFilterConfig, commands: jax.Array
) -> Callable[[jax.Array, ActuatorModel], ActuatorRollouts]:
    def run(y: jax.Array, model: ActuatorModel) -> ActuatorRollouts:
        def one(command: jax.Array) -> jax.Array:
            def advance(current: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
                following = effort_lag_step(current, command, model, config.dt, substeps=2)
                return following, following

            _, future = jax.lax.scan(advance, y, None, length=config.horizon)
            return jnp.concatenate((y[None], future))

        states = jax.vmap(one)(commands)
        return ActuatorRollouts(
            states,
            jnp.broadcast_to(commands[:, None], (len(commands), config.horizon, 4)),
            jnp.ones(len(commands), dtype=bool),
        )

    return run


def test_augmented_certificate_state_and_absolute_time_partials_match_finite_difference() -> None:
    state, model, _ = _resources()
    state = state.at[7:10].set(jnp.asarray([0.21, -0.14, 0.04]))
    state = state.at[10:13].set(jnp.asarray([0.11, -0.07, 0.03]))
    state = state.at[13:].add(jnp.asarray([0.003, -0.002, 0.001, -0.001]))
    config = ActuatorFilterConfig(horizon=6, sqp_iterations=0)
    commands = jnp.stack((state[13:], state[13:] + jnp.asarray([0.002, -0.001, 0.003, -0.002])))
    rollout = _rollout(config, commands)
    obstacles = _obstacles(config, moving=True)
    evaluate = jax.jit(lambda y, o: actuator_policy_certificates(y, rollout, model, o, config))
    certificate = evaluate(state, obstacles)
    assert bool(jnp.all(certificate.gradient_valid))
    direction = jnp.asarray(
        [
            0.2,
            -0.1,
            0.3,
            0.1,
            0.02,
            -0.03,
            0.0,
            -0.3,
            0.1,
            -0.1,
            0.2,
            -0.1,
            0.05,
            0.01,
            -0.02,
            0.01,
            0.03,
        ]
    )
    step = 1e-5
    difference = (
        evaluate(state + step * direction, obstacles).smooth_values
        - evaluate(state - step * direction, obstacles).smooth_values
    ) / (2 * step)
    np.testing.assert_allclose(certificate.gradients @ direction, difference, rtol=3e-6, atol=2e-7)
    plus = obstacles._replace(centers=obstacles.centers + step * obstacles.velocities)
    minus = obstacles._replace(centers=obstacles.centers - step * obstacles.velocities)
    time_difference = (
        evaluate(state, plus).smooth_values - evaluate(state, minus).smooth_values
    ) / (2 * step)
    np.testing.assert_allclose(certificate.time_derivatives, time_difference, rtol=2e-6, atol=1e-8)
    # Audit the complete affine inequality directly against dH/dt along f_a+G_a*u.
    command = commands[1]
    derivative = certificate.time_derivatives + certificate.gradients @ augmented_dynamics(
        state, command, model
    )
    direct = derivative + config.policy_alpha * certificate.smooth_values
    np.testing.assert_allclose(certificate.bounds - certificate.rows @ command, direct, atol=2e-12)
    np.testing.assert_allclose(
        certificate.rows, -certificate.gradients[:, 13:] / model.time_constants, atol=2e-12
    )


def test_command_volume_uses_physical_command_box_without_second_allocation() -> None:
    _, model, _ = _resources()
    midpoint = (model.command_lower + model.command_upper) / 2
    for row in (jnp.asarray([1.0, 0, 0, 0]), jnp.asarray([1.0, -2.0, 3.0, -4.0])):
        fraction = command_box_fraction(row, row @ midpoint, model)
        np.testing.assert_allclose(fraction, 0.5, atol=1e-12)
        changed = model._replace(
            effectiveness=jnp.asarray([0.7, 1, 0.85, 1]), time_constants=model.time_constants * 3
        )
        np.testing.assert_allclose(
            command_box_fraction(row, row @ midpoint, changed), fraction, atol=0
        )


def test_forward_reverse_jacobians_agree_without_changing_rollout_values() -> None:
    state, model, _ = _resources()
    state = state.at[7:10].set(jnp.asarray([0.15, -0.08, 0.03]))
    config = ActuatorFilterConfig(horizon=4, sqp_iterations=0)
    rollout = _rollout(config, jnp.stack((state[13:], state[13:] + 0.003)))
    obstacles = _obstacles(config, moving=True)
    forward = actuator_policy_certificates(state, rollout, model, obstacles, config)
    reverse = actuator_policy_certificates(
        state, rollout, model, obstacles, replace(config, gradient_mode="reverse")
    )
    np.testing.assert_array_equal(forward.rollouts.states, reverse.rollouts.states)
    np.testing.assert_array_equal(forward.smooth_values, reverse.smooth_values)
    np.testing.assert_allclose(forward.gradients, reverse.gradients, atol=1e-12)
    np.testing.assert_allclose(forward.time_derivatives, reverse.time_derivatives, atol=1e-12)


@pytest.mark.parametrize("moving", [False, True])
def test_directional_rows_match_complete_jacobian_and_executed_action(moving: bool) -> None:
    state, model, safety = _resources()
    state = state.at[7:10].set(jnp.asarray([0.15, -0.08, 0.03]))
    state = state.at[13:].multiply(jnp.asarray([0.95, 1.02, 1.03, 0.97]))
    model = model._replace(effectiveness=jnp.asarray([0.7, 1.0, 0.85, 1.0]))
    config = ActuatorFilterConfig(horizon=4, sqp_iterations=0)
    rollout = _rollout(config, jnp.stack((state[13:], state[13:] + 0.003)))
    obstacles = _obstacles(config, moving=moving)
    results = [
        jax.jit(
            lambda y: actuator_plcbf_step(
                y, rollout, model, obstacles, safety, state[13:], jnp.asarray(-1), cfg
            )
        )(state)
        for cfg in (config, replace(config, gradient_mode="directional"))
    ]
    full, direct = [result.certificates for result in results]
    np.testing.assert_allclose(full.rows, direct.rows, atol=2e-12)
    np.testing.assert_allclose(full.bounds, direct.bounds, atol=2e-12)
    np.testing.assert_allclose(full.drift_derivatives, direct.drift_derivatives, atol=2e-12)
    np.testing.assert_allclose(full.time_derivatives, direct.time_derivatives, atol=2e-12)
    np.testing.assert_allclose(full.gradients[:, 13:], direct.gradients[:, 13:], atol=2e-12)
    assert np.isnan(np.asarray(direct.gradients[:, :13])).all()
    np.testing.assert_array_equal(direct.gradient_components_computed, np.arange(17) >= 13)
    np.testing.assert_array_equal(full.gradient_valid, direct.gradient_valid)
    np.testing.assert_array_equal(full.eligible, direct.eligible)
    np.testing.assert_allclose(results[0].action, results[1].action, atol=2e-12)
    np.testing.assert_array_equal(results[0].execution_mode, results[1].execution_mode)


def test_held_checks_detect_intermediate_mover_and_evolving_force() -> None:
    state, model, safety = _resources()
    config = ActuatorFilterConfig(
        horizon=2, ego_radius=0.0, obstacle_clearance=0.0, sqp_iterations=0
    )
    obstacles = RuntimeObstacleTrajectories(
        jnp.asarray([[[-0.5, 0.0, 1.4]], [[0.0, 0.0, 1.4]], [[0.5, 0.0, 1.4]]]),
        jnp.asarray([0.04]),
        jnp.ones((3, 1), bool),
    )
    command = state[13:] + 0.005
    checked = check_actuator_hold(state, command, model, obstacles, safety, config)
    assert not bool(checked.collision_passed)
    assert bool(checked.command_motor_passed)
    assert float(checked.collision_margin) < -0.001
    np.testing.assert_array_equal(checked.nodes[0, 13:], state[13:])
    np.testing.assert_allclose(
        checked.nodes[-1, 13:],
        command + (state[13:] - command) * jnp.exp(-config.command_period / model.time_constants),
        atol=2e-16,
    )
    assert np.max(np.abs(np.asarray(checked.actual_forces[0] - checked.actual_forces[-1]))) > 0.001


def test_operational_rows_differentiate_augmented_held_prediction() -> None:
    state, model, safety = _resources()
    state = state.at[7:10].set(jnp.asarray([0.3, -0.2, 0.1]))
    state = state.at[10:13].set(jnp.asarray([0.4, -0.3, 0.2]))
    command = state[13:] + jnp.asarray([0.002, -0.001, 0.003, -0.002])
    config = ActuatorFilterConfig(horizon=4, sqp_iterations=0)
    matrix, bounds = predictive_operational_rows(state, command, model, safety, config)

    def margins(command: jax.Array) -> jax.Array:
        return jnp.min(
            operational_node_margins(
                held_actuator_nodes(state, command, model, config)[1:], safety
            ),
            axis=0,
        )

    direction = jnp.asarray([0.1, -0.2, 0.3, -0.4])
    step = 1e-6
    derivative = (margins(command + step * direction) - margins(command - step * direction)) / (
        2 * step
    )
    np.testing.assert_allclose(-matrix @ direction, derivative, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(bounds - matrix @ command, margins(command), atol=1e-15)


def test_empty_collision_horizon_accepts_hover_and_keeps_motor_memory() -> None:
    state, model, safety = _resources()
    config = ActuatorFilterConfig(horizon=4, sqp_iterations=0)
    command = state[13:]
    rollout = _rollout(config, command[None])
    result = jax.jit(
        lambda y: actuator_plcbf_step(
            y, rollout, model, _empty_obstacles(config), safety, command, jnp.asarray(-1), config
        )
    )(state)
    assert bool(result.qp_valid)
    assert int(result.execution_mode) == 0
    assert float(result.executed_policy_dual) == 0
    assert np.isposinf(np.asarray(result.certificates.hard.values[0]))
    np.testing.assert_allclose(result.next_estimated_state, state, atol=1e-14)


def test_emergency_cannot_bypass_effectiveness_or_lag() -> None:
    state, model, safety = _resources()
    model = model._replace(
        effectiveness=jnp.asarray([0.7, 1.0, 1.0, 1.0]), time_constants=model.time_constants * 3
    )
    config = ActuatorFilterConfig(horizon=2, sqp_iterations=0)
    obstacle = RuntimeObstacleTrajectories(
        jnp.broadcast_to(state[:3], (3, 1, 3)), jnp.asarray([0.3]), jnp.ones((3, 1), bool)
    )
    command = state[13:]
    emergency = jnp.minimum(command / model.effectiveness, model.command_upper)
    result = jax.jit(
        lambda y: actuator_plcbf_step(
            y,
            _rollout(config, command[None]),
            model,
            obstacle,
            safety,
            emergency,
            jnp.asarray(-1),
            config,
        )
    )(state)
    assert bool(result.degraded)
    assert int(result.execution_mode) == 3
    np.testing.assert_allclose(result.action, emergency)
    np.testing.assert_allclose(result.applied.actual_forces[0], state[13:] * model.effectiveness)
    assert not np.isclose(
        float(result.applied.actual_forces[-1, 0]), float(emergency[0] * model.effectiveness[0])
    )
    assert float(result.executed_policy_dual) == 0


def test_invalid_operational_limits_fail_closed() -> None:
    state, model, safety = _resources()
    safety = safety._replace(speed_max=jnp.asarray(-1.0))
    config = ActuatorFilterConfig(horizon=2, sqp_iterations=0)
    result = jax.jit(
        lambda y: actuator_plcbf_step(
            y,
            _rollout(config, state[None, 13:]),
            model,
            _empty_obstacles(config),
            safety,
            state[13:],
            jnp.asarray(-1),
            config,
        )
    )(state)
    assert int(result.execution_mode) == 4
    assert np.all(np.isnan(np.asarray(result.action)))


def test_refinement_retains_original_limits_and_rejects_initial_violation() -> None:
    state, model, safety = _resources()
    state = state.at[2].set(0.14)
    config = ActuatorFilterConfig(horizon=2, sqp_iterations=1)
    result = jax.jit(
        lambda y: actuator_plcbf_step(
            y,
            _rollout(config, state[None, 13:]),
            model,
            _empty_obstacles(config),
            safety,
            state[13:],
            jnp.asarray(-1),
            config,
        )
    )(state)
    assert not bool(result.qp_valid)
    assert not bool(result.applied.operational_passed)
    assert bool(result.degraded)
    assert int(result.sqp_iterations) == 1
    assert float(jnp.min(result.applied.operational_margins)) < 0


def test_rejects_legacy_state_dimension_and_invalid_configuration() -> None:
    state, model, _ = _resources()
    config = ActuatorFilterConfig(horizon=2)
    with pytest.raises(ValueError, match="17-state"):
        actuator_policy_certificates(
            state[:13], _rollout(config, state[None, 13:]), model, _empty_obstacles(config), config
        )
    with pytest.raises(ValueError, match="command hold"):
        replace(config, command_hold_steps=3).validate()
