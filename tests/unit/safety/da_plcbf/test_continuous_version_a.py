from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.continuous_demo_scenarios import (
    blocking_static_scenario,
    constant_wind_scenario,
    scenario_obstacle_window,
    scenario_safety_limits,
    scenario_true_wind,
)
from crazyflow.safety.da_plcbf.continuous_version_a import (
    ContinuousVersionAConfig,
    RuntimeObstacleTrajectories,
    continuous_version_a_step,
    obstacle_agnostic_waypoint_callbacks,
    runtime_policy_values,
)
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig, waypoint_nominal_wrench
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig, VersionAModel
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator, VersionAFilterConfig


def _resources() -> tuple[VersionAModel, VersionAActuator]:
    raw: dict[str, Any] = load_params("cf21B_500")
    dtype = jnp.float32
    inertia = jnp.asarray(raw["J"], dtype=dtype)
    model = VersionAModel(
        mass=jnp.asarray(raw["mass"], dtype=dtype),
        gravity_vec=jnp.asarray(raw["gravity_vec"], dtype=dtype),
        inertia=inertia,
        inertia_inv=jnp.linalg.inv(inertia),
        drag_matrix=jnp.asarray(raw["drag_matrix"], dtype=dtype),
        wind_velocity=jnp.zeros(3, dtype=dtype),
        external_force=jnp.zeros(3, dtype=dtype),
        external_torque=jnp.zeros(3, dtype=dtype),
    )
    actuator = VersionAActuator(
        arm_length=jnp.asarray(raw["L"], dtype=dtype),
        thrust_to_torque=jnp.asarray(raw["thrust2torque"], dtype=dtype),
        mixing_matrix=jnp.asarray(raw["mixing_matrix"], dtype=dtype),
        thrust_min=jnp.asarray(raw["thrust_min"], dtype=dtype),
        thrust_max=jnp.asarray(raw["thrust_max"], dtype=dtype),
    )
    return model, actuator


def test_blocking_static_nominal_collides_but_continuous_plcbf_avoids_and_resumes_goal() -> None:
    """One deterministic trajectory is the functional gate for the corrected Version-A core."""
    model, actuator = _resources()
    scenario = blocking_static_scenario()
    config = ContinuousVersionAConfig(
        dt=scenario.dt,
        horizon=scenario.horizon,
        obstacle_clearance=scenario.obstacle_clearance,
        ego_radius=scenario.ego_radius,
    )
    quad_config = QuadPolicyConfig()
    nominal_rollout, fallback_rollouts = obstacle_agnostic_waypoint_callbacks(
        scenario.goal_position,
        scenario.goal_velocity,
        scenario.skill_displacements,
        actuator,
        quad_config,
        dt=scenario.dt,
        horizon=scenario.horizon,
    )
    obstacles = scenario_obstacle_window(scenario, 0)
    safety = scenario_safety_limits(scenario)
    barrier = VersionABarrierConfig(obstacle_clearance=scenario.obstacle_clearance)
    filter_config = VersionAFilterConfig(policy_alpha=2.0)

    nominal_step = jax.jit(
        lambda state: direct_wrench_symplectic_step(
            state,
            waypoint_nominal_wrench(
                state, scenario.goal_position, scenario.goal_velocity, model, actuator, quad_config
            ).wrench,
            model,
            scenario.dt,
        )
    )
    filtered_step = jax.jit(
        lambda state, previous_index: continuous_version_a_step(
            state,
            nominal_rollout,
            fallback_rollouts,
            obstacles,
            model,
            actuator,
            safety,
            barrier,
            filter_config,
            config,
            previous_policy_index=previous_index,
        )
    )

    nominal_state = scenario.initial_state
    filtered_state = scenario.initial_state
    previous_index = jnp.asarray(-1, dtype=jnp.int32)
    nominal_minimum = np.inf
    filtered_minimum = np.inf
    maximum_intervention = 0.0
    maximum_policy_dual = 0.0
    degraded_steps = 0
    checked_augmented_library = False
    obstacle_center = scenario.obstacle_initial_centers[0]
    for _ in range(scenario.steps):
        nominal_state = nominal_step(nominal_state)
        decision = filtered_step(filtered_state, previous_index)
        filtered_state = direct_wrench_symplectic_step(
            filtered_state, decision.action, model, scenario.dt
        )
        previous_index = decision.selected_index
        nominal_minimum = min(
            nominal_minimum, float(jnp.linalg.norm(nominal_state[:3] - obstacle_center))
        )
        filtered_minimum = min(
            filtered_minimum, float(jnp.linalg.norm(filtered_state[:3] - obstacle_center))
        )
        maximum_intervention = max(maximum_intervention, float(decision.qp_intervention_norm))
        maximum_policy_dual = max(maximum_policy_dual, float(decision.selected_policy_dual))
        degraded_steps += int(decision.degraded)
        assert not bool(decision.used_fallback & decision.qp_valid)
        assert decision.candidates.states.shape[0] == scenario.skill_displacements.shape[0] + 1
        if not checked_augmented_library:
            np.testing.assert_allclose(
                decision.candidates.wrenches[0, 0], decision.nominal_action, atol=1e-7
            )
            assert int(decision.safe_candidate_count) >= 1
            obstacle_count = scenario.obstacle_radii.shape[0]
            assert not np.any(decision.continuous_filter.analytic_barriers.enabled[:obstacle_count])
            assert np.all(decision.continuous_filter.analytic_barriers.enabled[obstacle_count:])
            checked_augmented_library = True

    physical_radius = float(scenario.obstacle_radii[0])
    inflated_radius = physical_radius + config.ego_radius + config.obstacle_clearance
    assert nominal_minimum < physical_radius
    assert filtered_minimum > inflated_radius
    assert maximum_intervention > 1e-3
    assert maximum_policy_dual > 1e-6
    assert degraded_steps == 0
    assert float(jnp.linalg.norm(filtered_state[:3] - scenario.goal_position)) < 0.1


def test_dynamic_obstacle_value_uses_relative_swept_geometry_not_only_nodes() -> None:
    states = jnp.zeros((2, 3, 13), dtype=jnp.float32)
    states = states.at[:, :, 6].set(1.0)
    states = states.at[0, :, :3].set(
        jnp.asarray([[-1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    )
    states = states.at[1, :, :3].set(
        jnp.asarray([[-1.0, 1.5, 1.0], [0.0, 1.5, 1.0], [1.0, 1.5, 1.0]])
    )
    obstacles = RuntimeObstacleTrajectories(
        centers=jnp.asarray([[[0.0, -1.0, 1.0]], [[-1.0, 1.0, 1.0]], [[-2.0, 3.0, 1.0]]]),
        radii=jnp.asarray([0.2]),
        mask=jnp.ones((3, 1), dtype=bool),
    )
    values = jax.jit(
        lambda rollout_states: runtime_policy_values(
            rollout_states, obstacles, obstacle_clearance=0.1
        )
    )(states)

    assert float(values.values[0]) < 0.0
    assert float(values.values[1]) > 0.0
    assert bool(jnp.all(values.input_valid))


def test_constant_wind_scenario_has_exactly_one_zero_to_constant_transition() -> None:
    scenario = constant_wind_scenario()
    winds = np.asarray(
        jax.vmap(lambda step: scenario_true_wind(scenario, step))(jnp.arange(scenario.steps + 40))
    )
    transitions = np.flatnonzero(np.any(winds[1:] != winds[:-1], axis=-1)) + 1

    np.testing.assert_array_equal(transitions, [scenario.wind_change_step])
    assert np.all(winds[: scenario.wind_change_step] == 0.0)
    np.testing.assert_array_equal(
        winds[scenario.wind_change_step :],
        np.broadcast_to(np.asarray(scenario.wind_after), winds[scenario.wind_change_step :].shape),
    )
    assert np.all(np.isfinite(scenario.wind_after))
