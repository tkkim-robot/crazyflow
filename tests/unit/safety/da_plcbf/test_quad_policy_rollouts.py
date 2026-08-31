from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.actor import (
    SharedActorConfig,
    SharedActorSpec,
    initialize_shared_actor,
)
from crazyflow.safety.da_plcbf.quad_policy import (
    QuadPolicyConfig,
    acceleration_to_feasible_wrench,
    shared_quad_fallback_wrenches,
    waypoint_nominal_wrench,
)
from crazyflow.safety.da_plcbf.quad_rollouts import (
    direct_wrench_symplectic_step,
    rollout_shared_quad_library,
)
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


def _physical() -> tuple[VersionAModel, VersionAActuator]:
    params: dict[str, Any] = load_params("cf21B_500")
    model = VersionAModel(
        mass=jnp.asarray(params["mass"]),
        gravity_vec=jnp.asarray(params["gravity_vec"]),
        inertia=jnp.asarray(params["J"]),
        inertia_inv=jnp.linalg.inv(jnp.asarray(params["J"])),
        drag_matrix=jnp.asarray(params["drag_matrix"]),
        wind_velocity=jnp.zeros(3),
        external_force=jnp.zeros(3),
        external_torque=jnp.zeros(3),
    )
    actuator = VersionAActuator(
        arm_length=jnp.asarray(params["L"]),
        thrust_to_torque=jnp.asarray(params["thrust2torque"]),
        mixing_matrix=jnp.asarray(params["mixing_matrix"]),
        thrust_min=jnp.asarray(params["thrust_min"]),
        thrust_max=jnp.asarray(params["thrust_max"]),
    )
    return model, actuator


def _actor_problem() -> tuple[object, ...]:
    model, actuator = _physical()
    policy_count = 6
    spec = SharedActorSpec(
        base_codes=jnp.eye(policy_count, 3, dtype=jnp.float32),
        base_desired_velocities=jnp.array(
            [
                [0.8, 0.0, 0.0],
                [-0.8, 0.0, 0.0],
                [0.0, 0.8, 0.0],
                [0.0, -0.8, 0.0],
                [0.0, 0.0, 0.6],
                [0.0, 0.0, -0.6],
            ]
        ),
        base_durations=jnp.full((policy_count,), 1.0),
        adaptive_mask=jnp.array([False, False, True, True, True, True]),
    )
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.5, 0.0, 1.0]], [[-0.5, 0.0, 1.0]]]),
        obstacle_radii=jnp.full((2, 1), 0.15),
        obstacle_mask=jnp.ones((2, 1), dtype=bool),
        arena_lower=jnp.array([[-2.0, -2.0, 0.1], [-2.0, -2.0, 0.1]]),
        arena_upper=jnp.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
        speed_limit=jnp.full((2,), 3.0),
    )
    actor_config = SharedActorConfig(hidden_width=16, max_duration=1.5)
    quad_config = QuadPolicyConfig()
    params = initialize_shared_actor(
        jax.random.key(5), spec, dimension=3, n_obstacles=1, config=actor_config
    )
    return model, actuator, spec, scenarios, actor_config, quad_config, params


def test_hover_and_tilt_commands_have_correct_sign_and_exact_motor_feasibility() -> None:
    model, actuator = _physical()
    config = QuadPolicyConfig()
    quaternion = jnp.array([0.0, 0.0, 0.0, 1.0])
    angular_velocity = jnp.zeros(3)

    hover = acceleration_to_feasible_wrench(
        jnp.zeros(3), quaternion, angular_velocity, model, actuator, config
    )
    tilt_x = acceleration_to_feasible_wrench(
        jnp.array([2.0, 0.0, 0.0]), quaternion, angular_velocity, model, actuator, config
    )

    assert bool(hover.input_valid) and bool(tilt_x.input_valid)
    np.testing.assert_allclose(hover.raw_wrench[0], model.mass * 9.81, rtol=1e-6)
    np.testing.assert_allclose(hover.raw_wrench[1:], 0.0, atol=1e-8)
    assert tilt_x.raw_wrench[2] > 0  # positive pitch rotates body +z toward world +x
    for command in (hover, tilt_x):
        assert np.all(command.bounded_motor_forces >= actuator.thrust_min)
        assert np.all(command.bounded_motor_forces <= actuator.thrust_max)


def test_waypoint_goal_affects_only_nominal_controller_not_fallback_observation() -> None:
    model, actuator, spec, scenarios, actor_config, quad_config, params = _actor_problem()
    state = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    states = jnp.broadcast_to(state, (6, 2, 13))
    fallback_first = shared_quad_fallback_wrenches(
        params,
        spec,
        states,
        scenarios,
        model,
        actuator,
        elapsed=jnp.array(0.0),
        horizon_duration=1.0,
        policy_gain=1.5,
        actor_config=actor_config,
        quad_config=quad_config,
    )
    fallback_second = shared_quad_fallback_wrenches(
        params,
        spec,
        states,
        scenarios,
        model,
        actuator,
        elapsed=jnp.array(0.0),
        horizon_duration=1.0,
        policy_gain=1.5,
        actor_config=actor_config,
        quad_config=quad_config,
    )
    nominal_a = waypoint_nominal_wrench(
        state, jnp.array([1.0, 0.0, 1.0]), jnp.zeros(3), model, actuator, quad_config
    )
    nominal_b = waypoint_nominal_wrench(
        state, jnp.array([-1.0, 0.0, 1.0]), jnp.zeros(3), model, actuator, quad_config
    )

    np.testing.assert_array_equal(fallback_first.wrench, fallback_second.wrench)
    assert not np.allclose(nominal_a.wrench, nominal_b.wrench)


def test_direct_wrench_quad_rollout_is_jittable_differentiable_and_stationary_at_hover() -> None:
    model, actuator, spec, scenarios, actor_config, quad_config, params = _actor_problem()
    initial = jnp.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    def rollout(candidate: object) -> object:
        return rollout_shared_quad_library(
            candidate,
            spec,
            initial,
            scenarios,
            model,
            actuator,
            dt=0.01,
            horizon=8,
            policy_gain=1.5,
            actor_config=actor_config,
            quad_config=quad_config,
        )

    eager = rollout(params)
    compiled = jax.jit(rollout)(params)
    gradient = jax.grad(lambda candidate: jnp.sum(rollout(candidate).states[..., -1, :] ** 2))(
        params
    )

    assert eager.states.shape == (6, 2, 9, 13)
    assert eager.wrenches.shape == (6, 2, 8, 4)
    assert np.all(np.isfinite(np.asarray(eager.states)))
    assert np.all(np.asarray(eager.policy_valid))
    np.testing.assert_allclose(compiled.states, eager.states, atol=1e-6)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(gradient))

    hover_wrench = jnp.array([model.mass * 9.81, 0.0, 0.0, 0.0])
    stationary = direct_wrench_symplectic_step(initial[0], hover_wrench, model, 0.01)
    np.testing.assert_allclose(stationary, initial[0], atol=2e-6)
