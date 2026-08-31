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
from crazyflow.safety.da_plcbf.config import LibraryLossConfig
from crazyflow.safety.da_plcbf.descriptors import trajectory_descriptors
from crazyflow.safety.da_plcbf.library import descriptor_targets_from_spec
from crazyflow.safety.da_plcbf.quad_actor_bptt import (
    build_dynamic_model_quad_actor_bptt_functions,
    build_quad_actor_bptt_functions,
)
from crazyflow.safety.da_plcbf.quad_actor_losses import (
    QuadLearningConfig,
    quad_actor_library_loss,
    quad_safety_values,
    rigid_body_safety_batch_from_circles,
)
from crazyflow.safety.da_plcbf.quad_generic_diversity_bptt import (
    build_quad_generic_diversity_bptt_functions,
)
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.quad_rollouts import rollout_shared_quad_library
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.version_a_barriers import (
    RigidBodySafetySet,
    VersionABarrierConfig,
    VersionAModel,
)
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


def _problem() -> tuple[object, ...]:
    model, actuator = _physical()
    spec = SharedActorSpec(
        base_codes=jnp.array(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]
        ),
        base_desired_velocities=jnp.array(
            [[0.7, 0.0, 0.0], [-0.7, 0.0, 0.0], [0.7, 0.08, 0.0], [0.7, -0.08, 0.0]]
        ),
        base_durations=jnp.full((4,), 0.8),
        adaptive_mask=jnp.array([False, False, True, True]),
    )
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.0, 0.02, 1.0]], [[0.0, -0.02, 1.0]]]),
        obstacle_radii=jnp.full((2, 1), 0.20),
        obstacle_mask=jnp.ones((2, 1), dtype=bool),
        arena_lower=jnp.array([[-2.0, -2.0, 0.2], [-2.0, -2.0, 0.2]]),
        arena_upper=jnp.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
        speed_limit=jnp.full((2,), 2.5),
    )
    initial = jnp.array(
        [
            [-0.42, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.45, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.70, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    actor_config = SharedActorConfig(
        hidden_width=8,
        residual_scale=0.8,
        min_duration=0.1,
        max_duration=1.0,
        duration_transition=0.08,
    )
    quad_policy_config = QuadPolicyConfig(acceleration_limit=4.0)
    barrier_config = VersionABarrierConfig(obstacle_clearance=0.08)
    learning_config = QuadLearningConfig(dt=0.02, horizon=20, softmin_beta=30.0)
    loss_config = LibraryLossConfig(
        covariance_regularizer=0.05,
        code_weight=0.0,
        diversity_weight=1e-4,
        action_weight=1e-4,
        action_rate_weight=1e-4,
        terminal_weight=1e-4,
        trust_weight=1e-3,
    )
    safety = rigid_body_safety_batch_from_circles(
        scenarios, angular_rate_max=8.0, tilt_max_radians=np.deg2rad(65.0)
    )
    params = initialize_shared_actor(
        jax.random.key(17), spec, dimension=3, n_obstacles=1, config=actor_config
    )
    targets = jnp.zeros((4, 9))
    scales = jnp.array([2.0, 2.0, 1.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
    return (
        model,
        actuator,
        spec,
        scenarios,
        initial,
        actor_config,
        quad_policy_config,
        barrier_config,
        learning_config,
        loss_config,
        safety,
        params,
        targets,
        scales,
    )


def test_swept_sphere_value_detects_tunnelling_and_soft_value_is_conservative() -> None:
    first = jnp.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    second = first.at[0].set(1.0)
    states = jnp.stack((first, second))[None, None, ...]
    safety = RigidBodySafetySet(
        obstacle_centers=jnp.array([[[0.0, 0.0, 1.0]]]),
        obstacle_radii=jnp.array([[0.2]]),
        obstacle_mask=jnp.ones((1, 1), dtype=bool),
        arena_lower=jnp.array([[-2.0, -2.0, 0.1]]),
        arena_upper=jnp.array([[2.0, 2.0, 2.0]]),
        speed_max=jnp.array([3.0]),
        angular_rate_max=jnp.array([8.0]),
        tilt_max_radians=jnp.array([1.0]),
    )
    values = quad_safety_values(states, safety, VersionABarrierConfig(), softmin_beta=40.0)

    assert np.min(np.asarray(values.node_values[..., 0])) > 0
    assert float(values.segment_obstacle_values[0, 0, 0, 0]) == -1.0
    assert float(values.hard_policy_margins[0, 0]) == -1.0
    assert float(values.smooth_policy_margins[0, 0]) <= -1.0


def test_quad_loss_has_finite_end_to_end_bptt_gradient_and_conservative_margin() -> None:
    problem = _problem()
    (
        model,
        actuator,
        spec,
        scenarios,
        initial,
        actor_config,
        quad_policy_config,
        barrier_config,
        learning_config,
        loss_config,
        safety,
        params,
        targets,
        scales,
    ) = problem

    targets = descriptor_targets_from_spec(spec)

    def objective(candidate: object) -> tuple[jax.Array, object]:
        return quad_actor_library_loss(
            candidate,
            spec,
            initial,
            scenarios,
            safety,
            targets,
            params,
            scales,
            model,
            actuator,
            actor_config,
            quad_policy_config,
            barrier_config,
            learning_config,
            loss_config,
        )

    (loss, metrics), gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))(params)
    rollouts = rollout_shared_quad_library(
        params,
        spec,
        initial,
        scenarios,
        model,
        actuator,
        dt=learning_config.dt,
        horizon=learning_config.horizon,
        policy_gain=learning_config.policy_gain,
        actor_config=actor_config,
        quad_config=quad_policy_config,
    )
    values = quad_safety_values(
        rollouts.states, safety, barrier_config, softmin_beta=learning_config.softmin_beta
    )
    translation = jnp.concatenate((rollouts.states[..., :3], rollouts.states[..., 7:10]), axis=-1)
    normalized_descriptors = trajectory_descriptors(translation) / scales
    expected_code = jnp.mean((normalized_descriptors - targets[:, None, :] / scales) ** 2)

    assert np.isfinite(float(loss))
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(gradient))
    assert float(metrics.rollout_valid_fraction) == 1.0
    np.testing.assert_allclose(metrics.code, expected_code, rtol=1e-6, atol=1e-7)
    assert np.all(
        np.asarray(values.smooth_policy_margins) <= np.asarray(values.hard_policy_margins) + 1e-6
    )
    assert np.linalg.norm(np.asarray(gradient.velocity_offsets[2:])) > 1e-6


def test_quad_bptt_updates_candidate_only_and_preserves_structural_rollouts() -> None:
    (
        model,
        actuator,
        spec,
        scenarios,
        initial,
        actor_config,
        quad_policy_config,
        barrier_config,
        learning_config,
        loss_config,
        safety,
        params,
        targets,
        scales,
    ) = _problem()
    functions = build_quad_actor_bptt_functions(
        spec,
        model,
        actuator,
        actor_config,
        quad_policy_config,
        barrier_config,
        learning_config,
        loss_config,
        learning_rate=5e-3,
        burst_steps=2,
    )
    before = rollout_shared_quad_library(
        params,
        spec,
        initial,
        scenarios,
        model,
        actuator,
        dt=learning_config.dt,
        horizon=learning_config.horizon,
        policy_gain=learning_config.policy_gain,
        actor_config=actor_config,
        quad_config=quad_policy_config,
    )
    final, history = functions.burst(
        functions.initialize(params), initial, scenarios, safety, targets, params, scales
    )
    after = rollout_shared_quad_library(
        final.params,
        spec,
        initial,
        scenarios,
        model,
        actuator,
        dt=learning_config.dt,
        horizon=learning_config.horizon,
        policy_gain=learning_config.policy_gain,
        actor_config=actor_config,
        quad_config=quad_policy_config,
    )

    assert np.all(np.asarray(history.update_accepted))
    assert np.all(np.asarray(history.parameter_delta_norm) > 0)
    np.testing.assert_array_equal(after.states[:2], before.states[:2])
    np.testing.assert_array_equal(after.wrenches[:2], before.wrenches[:2])
    assert np.all(np.asarray(final.params.velocity_offsets[:2]) == 0)
    assert np.linalg.norm(np.asarray(final.params.velocity_offsets[2:])) > 0


def test_bptt_builders_reject_an_all_structural_library() -> None:
    (
        model,
        actuator,
        spec,
        _,
        _,
        actor_config,
        quad_policy_config,
        barrier_config,
        learning_config,
        loss_config,
        _,
        _,
        _,
        _,
    ) = _problem()
    structural = spec.replace(adaptive_mask=jnp.zeros_like(spec.adaptive_mask))

    with pytest.raises(ValueError, match="at least one adaptive policy"):
        build_quad_actor_bptt_functions(
            structural,
            model,
            actuator,
            actor_config,
            quad_policy_config,
            barrier_config,
            learning_config,
            loss_config,
        )
    with pytest.raises(ValueError, match="at least one adaptive policy"):
        build_dynamic_model_quad_actor_bptt_functions(
            structural,
            actuator,
            actor_config,
            quad_policy_config,
            barrier_config,
            learning_config,
            loss_config,
        )
    with pytest.raises(ValueError, match="at least one adaptive policy"):
        build_quad_generic_diversity_bptt_functions(
            structural,
            model,
            actuator,
            actor_config,
            quad_policy_config,
            dt=learning_config.dt,
            horizon=learning_config.horizon,
            policy_gain=learning_config.policy_gain,
        )


def test_generic_diversity_bptt_executes_a_nonzero_update() -> None:
    (
        model,
        actuator,
        spec,
        scenarios,
        initial,
        actor_config,
        quad_policy_config,
        _,
        learning_config,
        _,
        _,
        params,
        _,
        scales,
    ) = _problem()
    functions = build_quad_generic_diversity_bptt_functions(
        spec,
        model,
        actuator,
        actor_config,
        quad_policy_config,
        dt=learning_config.dt,
        horizon=4,
        policy_gain=learning_config.policy_gain,
        burst_steps=1,
    )
    trained, history = functions.burst(
        functions.initialize(params), initial, scenarios, descriptor_targets_from_spec(spec), scales
    )
    jax.block_until_ready((trained, history))

    assert bool(history.update_accepted[0])
    assert float(history.gradient_norm[0]) > 0.0
    assert float(history.parameter_delta_norm[0]) > 0.0


def test_dynamic_model_bptt_compiled_executable_reuses_shape_for_changed_model() -> None:
    (
        model,
        actuator,
        spec,
        scenarios,
        initial,
        actor_config,
        quad_policy_config,
        barrier_config,
        learning_config,
        loss_config,
        safety,
        params,
        targets,
        scales,
    ) = _problem()
    functions = build_dynamic_model_quad_actor_bptt_functions(
        spec,
        actuator,
        actor_config,
        quad_policy_config,
        barrier_config,
        learning_config,
        loss_config,
        burst_steps=1,
    )
    state = functions.initialize(params)
    arguments = (state, initial, scenarios, safety, targets, params, scales)
    compiled = functions.burst.lower(*arguments, model).compile()
    first, first_metrics = compiled(*arguments, model)
    changed_model = model._replace(
        mass=1.35 * model.mass, wind_velocity=jnp.asarray([0.7, -0.25, 0.1], dtype=model.mass.dtype)
    )
    second, second_metrics = compiled(*arguments, changed_model)
    jax.block_until_ready((first, first_metrics, second, second_metrics))

    for metrics in (first_metrics, second_metrics):
        assert np.all(np.asarray(metrics.update_accepted))
        assert np.all(np.isfinite(np.asarray(metrics.loss.total)))
        assert np.all(np.isfinite(np.asarray(metrics.gradient_norm)))
    assert not np.allclose(
        np.asarray(first_metrics.loss.total), np.asarray(second_metrics.loss.total)
    )
    assert not np.allclose(
        np.asarray(first.params.velocity_offsets), np.asarray(second.params.velocity_offsets)
    )


def test_quad_bptt_improves_hard_train_and_held_out_empirical_coverage() -> None:
    (
        model,
        actuator,
        spec,
        scenarios,
        initial,
        actor_config,
        quad_policy_config,
        barrier_config,
        learning_config,
        loss_config,
        safety,
        params,
        targets,
        scales,
    ) = _problem()
    held_scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.02, 0.04, 1.0]], [[-0.02, -0.04, 1.0]]]),
        obstacle_radii=jnp.array([[0.19], [0.21]]),
        obstacle_mask=jnp.ones((2, 1), dtype=bool),
        arena_lower=scenarios.arena_lower,
        arena_upper=scenarios.arena_upper,
        speed_limit=scenarios.speed_limit,
    )
    held_initial = (
        initial.at[0, 0].set(-0.44).at[0, 7].set(0.68).at[1, 0].set(-0.43).at[1, 7].set(0.67)
    )
    held_safety = rigid_body_safety_batch_from_circles(
        held_scenarios, angular_rate_max=8.0, tilt_max_radians=np.deg2rad(65.0)
    )

    def hard_library_margin(
        candidate: object,
        scenario_batch: CircleScenarioBatch,
        state_batch: jax.Array,
        safety_batch: RigidBodySafetySet,
    ) -> jax.Array:
        rollouts = rollout_shared_quad_library(
            candidate,
            spec,
            state_batch,
            scenario_batch,
            model,
            actuator,
            dt=learning_config.dt,
            horizon=learning_config.horizon,
            policy_gain=learning_config.policy_gain,
            actor_config=actor_config,
            quad_config=quad_policy_config,
        )
        values = quad_safety_values(
            rollouts.states, safety_batch, barrier_config, softmin_beta=learning_config.softmin_beta
        )
        return jnp.max(values.hard_policy_margins, axis=0)

    functions = build_quad_actor_bptt_functions(
        spec,
        model,
        actuator,
        actor_config,
        quad_policy_config,
        barrier_config,
        learning_config,
        loss_config,
        learning_rate=1e-2,
        burst_steps=100,
    )
    train_before = hard_library_margin(params, scenarios, initial, safety)
    held_before = hard_library_margin(params, held_scenarios, held_initial, held_safety)
    final, history = functions.burst(
        functions.initialize(params), initial, scenarios, safety, targets, params, scales
    )
    train_after = hard_library_margin(final.params, scenarios, initial, safety)
    held_after = hard_library_margin(final.params, held_scenarios, held_initial, held_safety)

    assert np.all(np.asarray(history.update_accepted))
    assert np.all(np.asarray(train_before) < 0)
    assert np.all(np.asarray(train_after) >= 0)
    assert np.all(np.asarray(held_after) > np.asarray(held_before) + 0.1)
    assert np.mean(np.asarray(held_after) >= 0) > np.mean(np.asarray(held_before) >= 0)


def test_nonfinite_quad_candidate_batch_is_rejected_without_parameter_mutation() -> None:
    (
        model,
        actuator,
        spec,
        scenarios,
        initial,
        actor_config,
        quad_policy_config,
        barrier_config,
        learning_config,
        loss_config,
        safety,
        params,
        targets,
        scales,
    ) = _problem()
    functions = build_quad_actor_bptt_functions(
        spec,
        model,
        actuator,
        actor_config,
        quad_policy_config,
        barrier_config,
        learning_config,
        loss_config,
        burst_steps=1,
    )
    state = functions.initialize(params)
    invalid_initial = initial.at[0, 0].set(jnp.nan)
    following, metrics = functions.step(
        state, invalid_initial, scenarios, safety, targets, params, scales
    )

    assert not bool(metrics.update_accepted)
    for current, rejected in zip(
        jax.tree.leaves(state.params), jax.tree.leaves(following.params), strict=True
    ):
        np.testing.assert_array_equal(rejected, current)
