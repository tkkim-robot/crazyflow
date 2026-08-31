from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.actor import (
    SharedActorConfig,
    SharedActorSpec,
    initialize_shared_actor,
)
from crazyflow.safety.da_plcbf.dynamic_rollouts import (
    DynamicSphereScenarioBatch,
    dynamic_quad_safety_values,
    dynamic_sphere_window_from_tape,
    evaluate_dynamic_quad_library,
    rollout_shared_quad_dynamic_library,
    validate_dynamic_sphere_batch,
)
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.quad_rollouts import rollout_shared_quad_library
from crazyflow.safety.da_plcbf.scenarios import ScenarioTapeConfig, generate_scenario_tape
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig, VersionAModel
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


def _dynamic_scenarios(*, predictions: int = 2, nodes: int = 6) -> DynamicSphereScenarioBatch:
    time = jnp.linspace(0.0, 1.0, nodes)
    first = jnp.stack((0.5 * jnp.ones_like(time), 0.2 * time, jnp.ones_like(time)), axis=-1)
    second = jnp.stack((-0.5 * jnp.ones_like(time), -0.2 * time, jnp.ones_like(time)), axis=-1)
    centers = jnp.stack((first, second), axis=0)[:predictions, :, None, :][None, ...]
    return DynamicSphereScenarioBatch(
        obstacle_centers=centers,
        obstacle_radii=jnp.full(centers.shape[:-1], 0.15),
        obstacle_mask=jnp.ones(centers.shape[:-1], dtype=bool),
        arena_lower=jnp.array([[-2.0, -2.0, 0.1]]),
        arena_upper=jnp.array([[2.0, 2.0, 2.0]]),
        speed_limit=jnp.array([3.0]),
        angular_rate_max=jnp.array([8.0]),
        tilt_max_radians=jnp.array([1.1]),
    )


def _actor(obstacles: int) -> tuple[SharedActorSpec, object, SharedActorConfig]:
    spec = SharedActorSpec(
        base_codes=jnp.eye(3),
        base_desired_velocities=jnp.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]),
        base_durations=jnp.full((3,), 0.5),
        adaptive_mask=jnp.array([False, True, True]),
    )
    config = SharedActorConfig(hidden_width=8, max_duration=1.0)
    params = initialize_shared_actor(
        jax.random.key(3), spec, dimension=3, n_obstacles=obstacles, config=config
    )
    return spec, params, config


def _initial() -> jax.Array:
    return jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


def test_scenario_tape_window_keeps_static_dynamic_and_prediction_axes() -> None:
    tape = generate_scenario_tape(
        91,
        ScenarioTapeConfig(
            steps=20,
            prediction_samples=4,
            static_capacity=2,
            static_count=1,
            dynamic_capacity=3,
            ballistic_count=1,
            crossing_count=1,
            pursuit_count=0,
            interceptor_count=1,
            random_attacker_count=0,
        ),
    )
    window = dynamic_sphere_window_from_tape(
        tape, start_index=2, horizon=5, speed_limit=1.25, angular_rate_max=8.0, tilt_max_radians=1.0
    )

    assert window.obstacle_centers.shape == (1, 4, 6, 5, 3)
    assert window.obstacle_mask.shape == (1, 4, 6, 5)
    np.testing.assert_allclose(
        np.asarray(window.obstacle_centers[0, :, :, :2]),
        np.broadcast_to(tape.static_positions[None, None], (4, 6, 2, 3)),
        rtol=1e-6,
        atol=1e-6,
    )
    dynamic_start = window.obstacle_centers[0, :, 0, 2:]
    np.testing.assert_allclose(
        dynamic_start,
        np.broadcast_to(tape.dynamic_positions[2][None, ...], dynamic_start.shape),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(window.speed_limit, [1.25])
    with pytest.raises(ValueError, match="extends past"):
        dynamic_sphere_window_from_tape(
            tape,
            start_index=18,
            horizon=5,
            speed_limit=1.25,
            angular_rate_max=8.0,
            tilt_max_radians=1.0,
        )


def test_runtime_window_hides_unreleased_ballistic_future_until_observed() -> None:
    tape = generate_scenario_tape(
        92,
        ScenarioTapeConfig(
            steps=40,
            prediction_samples=4,
            static_capacity=1,
            static_count=0,
            dynamic_capacity=1,
            ballistic_count=1,
            crossing_count=0,
            pursuit_count=0,
            interceptor_count=0,
            random_attacker_count=0,
        ),
    )
    release = int(tape.ballistic_release_index[0])
    assert release > 0
    before = dynamic_sphere_window_from_tape(
        tape,
        start_index=release - 1,
        horizon=2,
        speed_limit=0.75,
        angular_rate_max=8.0,
        tilt_max_radians=1.0,
    )
    observed = dynamic_sphere_window_from_tape(
        tape,
        start_index=release,
        horizon=2,
        speed_limit=0.75,
        angular_rate_max=8.0,
        tilt_max_radians=1.0,
    )

    assert not np.any(before.obstacle_mask)
    assert not np.any(observed.obstacle_mask[..., 0])
    assert np.all(observed.obstacle_mask[..., 1])
    np.testing.assert_allclose(before.speed_limit, [0.75])
    np.testing.assert_allclose(observed.speed_limit, [0.75])


def test_relative_swept_check_catches_a_moving_sphere_between_safe_nodes() -> None:
    first = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    states = jnp.stack((first, first))[None, None, None, ...]
    scenarios = DynamicSphereScenarioBatch(
        obstacle_centers=jnp.array([[[[[-1.0, 0.0, 1.0]], [[1.0, 0.0, 1.0]]]]]),
        obstacle_radii=jnp.full((1, 1, 2, 1), 0.2),
        obstacle_mask=jnp.ones((1, 1, 2, 1), dtype=bool),
        arena_lower=jnp.array([[-2.0, -2.0, 0.1]]),
        arena_upper=jnp.array([[2.0, 2.0, 2.0]]),
        speed_limit=jnp.array([3.0]),
        angular_rate_max=jnp.array([8.0]),
        tilt_max_radians=jnp.array([1.0]),
    )
    values = dynamic_quad_safety_values(
        states, scenarios, VersionABarrierConfig(), softmin_beta=40.0
    )

    assert np.min(np.asarray(values.node_values[..., 0])) > 0
    np.testing.assert_allclose(values.segment_obstacle_values, -1.0)
    np.testing.assert_allclose(values.robust_hard_margins, -1.0)
    assert np.all(
        np.asarray(values.robust_smooth_margins) <= np.asarray(values.robust_hard_margins) + 1e-6
    )


def test_robust_margin_is_exact_worst_finite_prediction_not_an_average() -> None:
    state = _initial()[0]
    states = jnp.broadcast_to(state, (1, 1, 2, 2, 13))
    centers = jnp.array(
        [[[[[1.0, 0.0, 1.0]], [[1.0, 0.0, 1.0]]], [[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]]]]
    )
    scenarios = DynamicSphereScenarioBatch(
        obstacle_centers=centers,
        obstacle_radii=jnp.full((1, 2, 2, 1), 0.2),
        obstacle_mask=jnp.ones((1, 2, 2, 1), dtype=bool),
        arena_lower=jnp.array([[-2.0, -2.0, 0.1]]),
        arena_upper=jnp.array([[2.0, 2.0, 2.0]]),
        speed_limit=jnp.array([3.0]),
        angular_rate_max=jnp.array([8.0]),
        tilt_max_radians=jnp.array([1.0]),
    )
    values = dynamic_quad_safety_values(
        states, scenarios, VersionABarrierConfig(), softmin_beta=40.0
    )

    assert values.prediction_hard_margins[0, 0, 0] > 0
    np.testing.assert_allclose(values.prediction_hard_margins[0, 0, 1], -1.0)
    np.testing.assert_allclose(values.robust_hard_margins[0, 0], -1.0)
    assert not np.isclose(
        float(values.robust_hard_margins[0, 0]),
        float(jnp.mean(values.prediction_hard_margins[0, 0])),
    )


def test_dynamic_rollout_is_jittable_differentiable_and_preserves_explicit_r_axis() -> None:
    model, actuator = _physical()
    scenarios = _dynamic_scenarios()
    spec, params, actor_config = _actor(scenarios.obstacle_centers.shape[-2])
    quad_config = QuadPolicyConfig()

    def rollout(candidate: object) -> object:
        return rollout_shared_quad_dynamic_library(
            candidate,
            spec,
            _initial(),
            scenarios,
            model,
            actuator,
            dt=0.02,
            policy_gain=1.5,
            actor_config=actor_config,
            quad_config=quad_config,
        )

    eager = rollout(params)
    compiled = jax.jit(rollout)(params)
    gradient = jax.grad(lambda candidate: jnp.sum(rollout(candidate).states[..., -1, :]))(params)

    assert eager.states.shape == (3, 1, 2, 6, 13)
    assert eager.wrenches.shape == (3, 1, 2, 5, 4)
    np.testing.assert_allclose(compiled.states, eager.states, atol=1e-6)
    assert np.all(np.asarray(eager.policy_valid))
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(gradient))


def test_runtime_dynamic_library_requires_one_common_current_action_across_predictions() -> None:
    model, actuator = _physical()
    scenarios = _dynamic_scenarios()
    # Predictions must agree at the observed current node. Their futures may differ.
    scenarios = scenarios._replace(
        obstacle_centers=scenarios.obstacle_centers.at[:, 1, 0].set(
            scenarios.obstacle_centers[:, 0, 0]
        )
    )
    spec, params, actor_config = _actor(scenarios.obstacle_centers.shape[-2])

    def evaluate(candidate: object) -> object:
        return evaluate_dynamic_quad_library(
            candidate,
            spec,
            _initial()[0],
            scenarios,
            model,
            actuator,
            actor_config,
            QuadPolicyConfig(),
            VersionABarrierConfig(),
            dt=0.02,
            policy_gain=1.5,
        )

    result = jax.jit(evaluate)(params)
    assert result.hard_values.shape == (3,)
    assert result.first_motor_forces.shape == (3, 4)
    assert np.all(np.asarray(result.first_action_consistent))
    assert np.all(np.asarray(result.policy_valid))

    # A learned observation-dependent residual must not receive contradictory "current" worlds.
    trained = params.replace(output_kernel=jnp.ones_like(params.output_kernel) * 0.1)
    contradictory = scenarios._replace(
        obstacle_centers=scenarios.obstacle_centers.at[0, 1, 0, 0, 0].add(0.4)
    )
    rejected = evaluate_dynamic_quad_library(
        trained,
        spec,
        _initial()[0],
        contradictory,
        model,
        actuator,
        actor_config,
        QuadPolicyConfig(),
        VersionABarrierConfig(),
        dt=0.02,
        policy_gain=1.5,
        current_action_tolerance=1e-7,
    )
    assert not np.all(np.asarray(rejected.first_action_consistent))
    assert np.any(np.isneginf(np.asarray(rejected.hard_values)))


def test_duplicate_stationary_predictions_match_static_rollout_exactly() -> None:
    model, actuator = _physical()
    nodes = 5
    center = jnp.array([[[0.5, 0.0, 1.0]]])
    static = CircleScenarioBatch(
        obstacle_centers=center,
        obstacle_radii=jnp.array([[0.15]]),
        obstacle_mask=jnp.ones((1, 1), dtype=bool),
        arena_lower=jnp.array([[-2.0, -2.0, 0.1]]),
        arena_upper=jnp.array([[2.0, 2.0, 2.0]]),
        speed_limit=jnp.array([3.0]),
    )
    dynamic = DynamicSphereScenarioBatch(
        obstacle_centers=jnp.broadcast_to(center[:, None, None], (1, 2, nodes, 1, 3)),
        obstacle_radii=jnp.full((1, 2, nodes, 1), 0.15),
        obstacle_mask=jnp.ones((1, 2, nodes, 1), dtype=bool),
        arena_lower=static.arena_lower,
        arena_upper=static.arena_upper,
        speed_limit=static.speed_limit,
        angular_rate_max=jnp.array([8.0]),
        tilt_max_radians=jnp.array([1.0]),
    )
    spec, params, actor_config = _actor(1)
    quad_config = QuadPolicyConfig()
    reference = rollout_shared_quad_library(
        params,
        spec,
        _initial(),
        static,
        model,
        actuator,
        dt=0.02,
        horizon=nodes - 1,
        policy_gain=1.5,
        actor_config=actor_config,
        quad_config=quad_config,
    )
    predicted = rollout_shared_quad_dynamic_library(
        params,
        spec,
        _initial(),
        dynamic,
        model,
        actuator,
        dt=0.02,
        policy_gain=1.5,
        actor_config=actor_config,
        quad_config=quad_config,
    )

    np.testing.assert_allclose(predicted.states[:, :, 0], reference.states, atol=1e-6)
    np.testing.assert_array_equal(predicted.states[:, :, 0], predicted.states[:, :, 1])
    np.testing.assert_allclose(predicted.wrenches[:, :, 0], reference.wrenches, atol=1e-6)


def test_host_validation_rejects_invalid_enabled_prediction() -> None:
    scenarios = _dynamic_scenarios()
    invalid = scenarios._replace(
        obstacle_centers=scenarios.obstacle_centers.at[0, 0, 0, 0, 0].set(jnp.nan)
    )
    with pytest.raises(ValueError, match="centers must be finite"):
        validate_dynamic_sphere_batch(invalid)
