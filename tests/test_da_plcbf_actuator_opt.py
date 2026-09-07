from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.actuator_dynamics import (
    fit_native_time_constant,
    make_actuator_model,
)
from crazyflow.safety.da_plcbf.actuator_learning import ActuatorSkillConfig
from crazyflow.safety.da_plcbf.actuator_opt import (
    ActuatorOPTConfig,
    ActuatorOPTController,
    build_actuator_opt,
    opt_plan_rollouts,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.continuous_version_a import (
    RuntimeObstacleTrajectories,
    runtime_policy_values,
)
from crazyflow.safety.da_plcbf.version_a_barriers import RigidBodySafetySet, VersionAModel


@pytest.fixture(scope="module")
def problem() -> tuple:
    params = load_params("cf21B_500")
    body = VersionAModel(
        jnp.asarray(params["mass"]),
        jnp.asarray(params["gravity_vec"]),
        jnp.asarray(params["J"]),
        jnp.asarray(np.linalg.inv(params["J"])),
        jnp.asarray(params["drag_matrix"]),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros(3),
    )
    model = make_actuator_model(
        body,
        L=params["L"],
        thrust2torque=params["thrust2torque"],
        mixing_matrix=params["mixing_matrix"],
        thrust_min=params["thrust_min"],
        thrust_max=params["thrust_max"],
        time_constants=fit_native_time_constant()["nominal_tau"],
    )
    hover = float(-body.mass * body.gravity_vec[2] / 4)
    state = jnp.zeros(17).at[2].set(1).at[6].set(1).at[13:].set(hover)
    config = ActuatorFilterConfig(
        dt=0.02,
        horizon=12,
        command_hold_steps=2,
        held_substeps=2,
        obstacle_clearance=0,
        ego_radius=0.025,
    )
    actor = ActuatorSkillConfig(dt=0.02, horizon=12, control_interval_steps=2)
    safety = RigidBodySafetySet(
        jnp.array([[0.0, 0.0, 3.0]]),
        jnp.array([0.015]),
        jnp.array([False]),
        jnp.array([-3.0, -3.0, 0.1]),
        jnp.array([3.0, 3.0, 3.0]),
        jnp.asarray(4.0),
        jnp.asarray(12.0),
        jnp.asarray(1.0),
    )
    clear = RuntimeObstacleTrajectories(
        jnp.tile(jnp.array([[[0.0, 0.0, 3.0]]]), (13, 1, 1)),
        jnp.array([0.015]),
        jnp.ones((13, 1), dtype=bool),
    )
    nominal = jnp.full((6, 4), hover)
    emergency = jnp.full(4, hover)
    return state, model, clear, safety, nominal, emergency, config, actor


@pytest.fixture(scope="module")
def controller(problem: tuple) -> ActuatorOPTController:
    *args, config, actor = problem
    value = build_actuator_opt(
        config, actor, ActuatorOPTConfig(control_knots=3, wall_time_budget=1.0, max_iterations=50)
    )
    assert value.warmup(*args) > 0
    return value


def obstacle_at(problem: tuple, z: float) -> RuntimeObstacleTrajectories:
    obstacle = problem[2]
    return obstacle._replace(centers=obstacle.centers.at[:, 0, 2].set(z))


@pytest.mark.unit
def test_clear_nominal_fast_path_and_full_augmented_trace(
    problem: tuple, controller: ActuatorOPTController
) -> None:
    *args, config, _ = problem
    result = controller.solve(*args, observation_time=2.0)
    assert result.feasible and result.mode == "nominal"
    assert not result.compilation_in_solve
    assert result.held_check.passed
    assert result.plan.states.shape == (13, 17)
    np.testing.assert_allclose(result.action, args[4][0])
    assert result.deadline_met and result.available_action is not None
    assert result.available_at == 2.0 + result.service_seconds
    assert not result.attempts
    trace = opt_plan_rollouts(result.plan, config)
    assert trace.states.shape == (1, 13, 17)
    assert trace.commands.shape == (1, 12, 4)
    np.testing.assert_array_equal(trace.commands[0, ::2], trace.commands[0, 1::2])
    with pytest.raises(ValueError):
        result.plan.commands[0, 0] = 0


@pytest.mark.unit
def test_static_future_collision_uses_real_constrained_solver(
    problem: tuple, controller: ActuatorOPTController
) -> None:
    state, model, _, safety, _, emergency, _, _ = problem
    obstacle = obstacle_at(problem, 1.07)
    nominal = jnp.full((6, 4), 0.15)
    result = controller.solve(state, model, obstacle, safety, nominal, emergency)
    assert result.input_valid
    assert result.attempts
    assert any(a.evaluations > 1 and a.iterations > 0 for a in result.attempts)
    assert result.feasible and result.plan is not None
    assert result.held_check.passed
    assert result.plan.minimum_collision_margin >= -2e-6
    assert result.plan.minimum_operational_margin >= -1e-6
    assert np.linalg.norm(result.action - np.asarray(nominal[0])) > 1e-4
    assert np.all(result.action >= model.command_lower)
    assert np.all(result.action <= model.command_upper)


@pytest.mark.unit
def test_current_collision_cannot_be_optimized_away_and_uses_emergency(
    problem: tuple, controller: ActuatorOPTController
) -> None:
    state, model, _, safety, nominal, emergency, _, _ = problem
    result = controller.solve(state, model, obstacle_at(problem, 1.0), safety, nominal, emergency)
    assert result.input_valid and not result.feasible and result.plan is None
    assert result.mode == "degraded"
    assert not result.held_check.collision_passed
    np.testing.assert_array_equal(result.action, emergency)
    assert result.attempts


@pytest.mark.unit
def test_moving_obstacle_changes_the_current_absolute_prediction(
    problem: tuple, controller: ActuatorOPTController
) -> None:
    state, model, clear, safety, nominal, emergency, _, _ = problem
    moving_centers = clear.centers.at[:, 0, 2].set(jnp.linspace(1.14, 1.025, 13))
    moving_centers = moving_centers.at[:, 0, 0].set(0.025)
    moving = clear._replace(centers=moving_centers)
    fixed = clear._replace(centers=jnp.repeat(moving.centers[:1], 13, axis=0))
    safe = controller.solve(state, model, fixed, safety, nominal, emergency, observation_time=4.0)
    threatened = controller.solve(
        state, model, moving, safety, nominal, emergency, observation_time=4.0
    )
    assert safe.mode == "nominal" and safe.feasible
    assert threatened.attempts
    assert threatened.mode != "nominal"
    # Even if this finite compute budget finds no recovery, the observed threat is not hidden.
    if threatened.plan is not None:
        assert threatened.plan.minimum_collision_margin >= -2e-6
    assert threatened.available_at >= 4.0


@pytest.mark.unit
def test_shifted_plan_revalidated_at_new_state_model_and_clock(
    problem: tuple, controller: ActuatorOPTController
) -> None:
    state, model, clear, safety, nominal, emergency, config, _ = problem
    prior = controller.solve(state, model, clear, safety, nominal, emergency, observation_time=1.0)
    assert prior.plan is not None
    blocked = obstacle_at(problem, 1.07)
    changed = model._replace(time_constants=1.5 * model.time_constants)
    result = controller.solve(
        state,
        changed,
        blocked,
        safety,
        jnp.full((6, 4), 0.15),
        emergency,
        previous_plan=prior.plan,
        previous_command=prior.action,
        observation_time=1.0 + config.command_period,
    )
    assert result.warm_start_revalidated and result.warm_start_feasible
    assert result.warm_start_shift == 1
    assert result.feasible and result.held_check.passed
    assert result.attempts[0].initialization == "warm_start"
    assert result.plan.observation_time == 1.0 + config.command_period
    # Reuse under a newly intersecting obstacle must lose feasible-plan status.
    intersecting = controller.solve(
        state,
        changed,
        obstacle_at(problem, 1.0),
        safety,
        nominal,
        emergency,
        previous_plan=prior.plan,
        observation_time=1.0 + config.command_period,
    )
    assert intersecting.warm_start_revalidated and not intersecting.warm_start_feasible
    assert not intersecting.feasible


@pytest.mark.unit
def test_actual_budget_miss_never_credits_a_late_action(problem: tuple) -> None:
    *args, config, actor = problem
    controller = build_actuator_opt(
        config,
        actor,
        ActuatorOPTConfig(control_knots=3, wall_time_budget=1e-9, initializations=("nominal",)),
    )
    result = controller.solve(*args, observation_time=7.0)
    assert result.compilation_in_solve
    assert result.service_seconds > controller.opt_config.wall_time_budget
    assert not result.deadline_met and result.available_action is None
    assert result.available_at == 7.0 + result.service_seconds
    assert result.budget_exhausted
    assert result.action.shape == (4,)


@pytest.mark.unit
def test_solver_exception_retains_only_checked_candidates(
    problem: tuple, controller: ActuatorOPTController, monkeypatch: pytest.MonkeyPatch
) -> None:
    import crazyflow.safety.da_plcbf.actuator_opt as module

    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr(module, "minimize", fail)
    state, model, _, safety, nominal, emergency, _, _ = problem
    result = controller.solve(state, model, obstacle_at(problem, 1.0), safety, nominal, emergency)
    assert not result.feasible
    assert any("injected solver failure" in attempt.status for attempt in result.attempts)
    np.testing.assert_array_equal(result.action, emergency)
    assert result.held_check.command_motor_passed
    assert not result.held_check.collision_passed


@pytest.mark.unit
def test_invalid_state_and_inconsistent_command_hold_are_explicit(
    problem: tuple, controller: ActuatorOPTController
) -> None:
    state, model, clear, safety, nominal, emergency, config, _ = problem
    invalid = controller.solve(state.at[13].set(jnp.nan), model, clear, safety, nominal, emergency)
    assert not invalid.input_valid and not invalid.feasible
    assert invalid.mode == "invalid_input"
    assert invalid.available_action is None and np.all(np.isnan(invalid.action))
    commands = jnp.repeat(nominal, 2, axis=0).at[1, 0].add(0.01)
    with pytest.raises(ValueError, match="unchanged inside"):
        controller.solve(state, model, clear, safety, commands, emergency)
    with pytest.raises(ValueError, match="17-state"):
        controller.solve(state[:13], model, clear, safety, nominal, emergency)
    with pytest.raises(ValueError, match="same command clock"):
        build_actuator_opt(
            config, ActuatorSkillConfig(dt=0.01, horizon=12, control_interval_steps=2)
        )


@pytest.mark.unit
def test_expired_plan_is_not_reused_and_warmup_does_not_install_a_plan(
    problem: tuple, controller: ActuatorOPTController
) -> None:
    *args, config, _ = problem
    prior = controller.solve(*args)
    assert prior.plan is not None
    controller.warmup(*args)
    state, model, _, safety, _, emergency = args
    result = controller.solve(
        state,
        model,
        obstacle_at(problem, 1.07),
        safety,
        jnp.full((6, 4), 0.15),
        emergency,
        previous_plan=prior.plan,
        observation_time=config.dt * config.horizon + 0.04,
    )
    assert not result.warm_start_revalidated
    assert all(a.initialization != "warm_start" for a in result.attempts)


@pytest.mark.unit
def test_optimizer_constraint_jacobian_matches_finite_differences(
    problem: tuple, controller: ActuatorOPTController
) -> None:
    state, model, obstacles, safety, nominal, emergency, _, _ = problem
    state = state.at[7:10].set(jnp.array([0.15, 0.08, 0.02]))
    args = controller._arguments(state, model, obstacles, safety, nominal, emergency, None)
    vector = jnp.linspace(0.42, 0.50, 12)
    jacobian, _ = controller._differentiate(vector, *args)
    direction = jnp.linspace(-0.2, 0.3, 12)
    epsilon = 1e-3
    plus, _ = controller._knot_problem(vector + epsilon * direction, *args)
    minus, _ = controller._knot_problem(vector - epsilon * direction, *args)
    finite_difference = (plus - minus) / (2 * epsilon)
    np.testing.assert_allclose(jacobian @ direction, finite_difference, atol=1e-3, rtol=5e-3)


@pytest.mark.unit
def test_almost_stationary_swept_rows_have_finite_exact_envelope_gradients(problem: tuple) -> None:
    state, _, obstacle, _, _, _, config, _ = problem
    nodes = jnp.repeat(state[None], 3, axis=0).at[:, 0].set(jnp.array([0.0, 1e-15, 2e-15]))
    obstacle = obstacle._replace(centers=obstacle.centers[:3], mask=obstacle.mask[:3])

    def function(y: jax.Array) -> jax.Array:
        return runtime_policy_values(
            y[None, :, :13],
            obstacle,
            obstacle_clearance=config.obstacle_clearance,
            ego_radius=config.ego_radius,
            envelope_derivative=True,
        ).constraint_values[0]

    values = function(nodes)
    shared = runtime_policy_values(
        nodes[None, :, :13],
        obstacle,
        obstacle_clearance=config.obstacle_clearance,
        ego_radius=config.ego_radius,
    )
    np.testing.assert_array_equal(values, shared.constraint_values[0])
    assert np.all(np.isfinite(jax.jacfwd(function)(nodes)))
    assert np.all(np.isfinite(jax.jacrev(function)(nodes)))
