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
    shared_fallback_actions,
)
from crazyflow.safety.da_plcbf.config import LibraryLossConfig
from crazyflow.safety.da_plcbf.descriptors import trajectory_descriptors
from crazyflow.safety.da_plcbf.independent_actor import (
    IndependentActorParams,
    build_independent_quad_actor_bptt_functions,
    independent_fallback_actions,
    independent_quad_actor_library_loss,
    initialize_independent_actor,
    rollout_independent_quad_library,
    validate_independent_actor_shapes,
)
from crazyflow.safety.da_plcbf.library import descriptor_targets_from_spec
from crazyflow.safety.da_plcbf.proposal_ablations import (
    HybridProposalConfig,
    ProposalBudget,
    run_hybrid_proposal_bptt,
)
from crazyflow.safety.da_plcbf.quad_actor_bptt import build_quad_actor_bptt_functions
from crazyflow.safety.da_plcbf.quad_actor_losses import (
    QuadLearningConfig,
    quad_actor_library_loss,
    rigid_body_safety_batch_from_circles,
)
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig, VersionAModel
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


def _planar_problem() -> tuple[SharedActorSpec, CircleScenarioBatch, SharedActorConfig]:
    spec = SharedActorSpec(
        base_codes=jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        base_desired_velocities=jnp.array([[0.4, 0.0], [0.0, 0.4], [-0.4, 0.0]]),
        base_durations=jnp.full((3,), 0.8),
        adaptive_mask=jnp.ones((3,), dtype=bool),
    )
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[0.2, 0.3]]]),
        obstacle_radii=jnp.array([[0.15]]),
        obstacle_mask=jnp.ones((1, 1), dtype=bool),
        arena_lower=jnp.array([[-2.0, -2.0]]),
        arena_upper=jnp.array([[2.0, 2.0]]),
        speed_limit=jnp.array([2.0]),
    )
    return spec, scenarios, SharedActorConfig(hidden_width=6, max_duration=1.0)


def test_independent_network_tensors_have_a_real_policy_axis_and_no_shared_weights() -> None:
    spec, scenarios, config = _planar_problem()
    params = initialize_independent_actor(
        jax.random.key(3), spec, dimension=2, n_obstacles=1, config=config
    )

    assert params.input_kernel.shape[0] == 3
    assert params.hidden_kernel.shape == (3, 6, 6)
    assert params.output_kernel.shape == (3, 6, 2)
    assert params.output_bias.shape == (3, 2)
    assert not np.array_equal(
        np.asarray(params.input_kernel[0]), np.asarray(params.input_kernel[1])
    )
    validate_independent_actor_shapes(params, spec, dimension=2, n_obstacles=1)

    shared = initialize_shared_actor(
        jax.random.key(3), spec, dimension=2, n_obstacles=1, config=config
    )
    assert shared.input_kernel.ndim == 2
    assert params.input_kernel.size == spec.base_codes.shape[0] * shared.input_kernel.size
    assert scenarios.obstacle_centers.shape[0] == 1


def test_changing_one_policy_network_changes_only_that_policy_action_under_jit() -> None:
    spec, scenarios, config = _planar_problem()
    params = initialize_independent_actor(
        jax.random.key(4), spec, dimension=2, n_obstacles=1, config=config
    )
    states = jnp.broadcast_to(jnp.array([0.1, -0.1, 0.0, 0.0]), (3, 1, 4))
    changed = params.replace(output_bias=params.output_bias.at[1].set(jnp.array([0.8, -0.6])))

    evaluate = jax.jit(
        lambda candidate: independent_fallback_actions(
            candidate,
            spec,
            states,
            scenarios,
            elapsed=jnp.array(0.1),
            horizon_duration=1.0,
            policy_gain=1.5,
            action_limit=2.0,
            config=config,
        )
    )
    before = evaluate(params)
    after = evaluate(changed)

    np.testing.assert_array_equal(after[0], before[0])
    np.testing.assert_array_equal(after[2], before[2])
    assert not np.array_equal(np.asarray(after[1]), np.asarray(before[1]))


def test_independent_structural_slot_remains_exact_under_its_private_parameter_changes() -> None:
    spec, scenarios, config = _planar_problem()
    spec = spec.replace(adaptive_mask=jnp.array([False, True, True]))
    params = initialize_independent_actor(
        jax.random.key(14), spec, dimension=2, n_obstacles=1, config=config
    )
    changed = params.replace(
        code_offsets=params.code_offsets.at[0].set(99.0),
        velocity_offsets=params.velocity_offsets.at[0].set(-99.0),
        duration_offsets=params.duration_offsets.at[0].set(99.0),
        input_kernel=params.input_kernel.at[0].set(99.0),
        input_bias=params.input_bias.at[0].set(99.0),
        hidden_kernel=params.hidden_kernel.at[0].set(99.0),
        hidden_bias=params.hidden_bias.at[0].set(99.0),
        output_kernel=params.output_kernel.at[0].set(99.0),
        output_bias=params.output_bias.at[0].set(99.0),
    )
    states = jnp.broadcast_to(jnp.array([0.1, -0.1, 0.2, 0.0]), (3, 1, 4))

    def evaluate(candidate: IndependentActorParams) -> jax.Array:
        return independent_fallback_actions(
            candidate,
            spec,
            states,
            scenarios,
            elapsed=jnp.array(0.2),
            horizon_duration=1.0,
            policy_gain=1.5,
            action_limit=2.0,
            config=config,
        )

    before = jax.jit(evaluate)(params)
    after = jax.jit(evaluate)(changed)
    np.testing.assert_array_equal(after[0], before[0])


def test_zero_residual_independent_and_shared_architectures_match_exactly() -> None:
    spec, scenarios, config = _planar_problem()
    independent = initialize_independent_actor(
        jax.random.key(9), spec, dimension=2, n_obstacles=1, config=config
    )
    shared = initialize_shared_actor(
        jax.random.key(10), spec, dimension=2, n_obstacles=1, config=config
    )
    states = jnp.broadcast_to(jnp.array([-0.3, 0.1, 0.2, -0.1]), (3, 1, 4))
    arguments = {
        "spec": spec,
        "states": states,
        "scenarios": scenarios,
        "elapsed": jnp.array(0.2),
        "horizon_duration": 1.0,
        "policy_gain": 1.4,
        "action_limit": 2.0,
        "config": config,
    }

    independent_actions = independent_fallback_actions(independent, **arguments)
    shared_actions = shared_fallback_actions(shared, **arguments)
    np.testing.assert_array_equal(independent_actions, shared_actions)


def _physical() -> tuple[VersionAModel, VersionAActuator]:
    values: dict[str, Any] = load_params("cf21B_500")
    model = VersionAModel(
        mass=jnp.asarray(values["mass"]),
        gravity_vec=jnp.asarray(values["gravity_vec"]),
        inertia=jnp.asarray(values["J"]),
        inertia_inv=jnp.linalg.inv(jnp.asarray(values["J"])),
        drag_matrix=jnp.asarray(values["drag_matrix"]),
        wind_velocity=jnp.zeros(3),
        external_force=jnp.zeros(3),
        external_torque=jnp.zeros(3),
    )
    actuator = VersionAActuator(
        arm_length=jnp.asarray(values["L"]),
        thrust_to_torque=jnp.asarray(values["thrust2torque"]),
        mixing_matrix=jnp.asarray(values["mixing_matrix"]),
        thrust_min=jnp.asarray(values["thrust_min"]),
        thrust_max=jnp.asarray(values["thrust_max"]),
    )
    return model, actuator


def _quad_problem() -> tuple[object, ...]:
    model, actuator = _physical()
    spec = SharedActorSpec(
        base_codes=jnp.array([[1.0, 0.0], [-1.0, 0.0]]),
        base_desired_velocities=jnp.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]),
        base_durations=jnp.full((2,), 0.3),
        adaptive_mask=jnp.ones((2,), dtype=bool),
    )
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[1.5, 1.5, 1.0]]]),
        obstacle_radii=jnp.array([[0.1]]),
        obstacle_mask=jnp.ones((1, 1), dtype=bool),
        arena_lower=jnp.array([[-2.0, -2.0, 0.2]]),
        arena_upper=jnp.array([[2.0, 2.0, 2.0]]),
        speed_limit=jnp.array([3.0]),
    )
    initial = jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    actor_config = SharedActorConfig(hidden_width=4, min_duration=0.1, max_duration=0.5)
    quad_config = QuadPolicyConfig(acceleration_limit=3.0)
    barrier_config = VersionABarrierConfig(obstacle_clearance=0.05)
    learning_config = QuadLearningConfig(dt=0.02, horizon=3, softmin_beta=20.0)
    loss_config = LibraryLossConfig(
        covariance_regularizer=0.1,
        diversity_weight=1e-5,
        code_weight=0.0,
        action_weight=1e-5,
        action_rate_weight=1e-5,
        terminal_weight=1e-5,
        trust_weight=1e-4,
    )
    safety = rigid_body_safety_batch_from_circles(
        scenarios, angular_rate_max=8.0, tilt_max_radians=1.1
    )
    params = initialize_independent_actor(
        jax.random.key(11), spec, dimension=3, n_obstacles=1, config=actor_config
    )
    targets = jnp.zeros((2, 9))
    scales = jnp.ones(9)
    return (
        model,
        actuator,
        spec,
        scenarios,
        initial,
        actor_config,
        quad_config,
        barrier_config,
        learning_config,
        loss_config,
        safety,
        params,
        targets,
        scales,
    )


def test_independent_quad_rollout_loss_gradient_and_bptt_are_finite() -> None:
    (
        model,
        actuator,
        spec,
        scenarios,
        initial,
        actor_config,
        quad_config,
        barrier_config,
        learning_config,
        loss_config,
        safety,
        params,
        targets,
        scales,
    ) = _quad_problem()

    targets = descriptor_targets_from_spec(spec)
    scales = jnp.asarray([2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 2.0, 2.5, 3.0])

    def objective(candidate: IndependentActorParams) -> tuple[jax.Array, object]:
        return independent_quad_actor_library_loss(
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
            quad_config,
            barrier_config,
            learning_config,
            loss_config,
        )

    (loss, metrics), gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))(params)
    rollout = jax.jit(
        lambda candidate: rollout_independent_quad_library(
            candidate,
            spec,
            initial,
            scenarios,
            model,
            actuator,
            dt=learning_config.dt,
            horizon=learning_config.horizon,
            policy_gain=learning_config.policy_gain,
            actor_config=actor_config,
            quad_config=quad_config,
        )
    )(params)
    functions = build_independent_quad_actor_bptt_functions(
        spec,
        model,
        actuator,
        actor_config,
        quad_config,
        barrier_config,
        learning_config,
        loss_config,
        learning_rate=1e-3,
        burst_steps=1,
    )
    final, update = functions.step(
        functions.initialize(params), initial, scenarios, safety, targets, params, scales
    )
    translation = jnp.concatenate((rollout.states[..., :3], rollout.states[..., 7:10]), axis=-1)
    normalized_descriptors = trajectory_descriptors(translation) / scales
    expected_code = jnp.mean((normalized_descriptors - targets[:, None, :] / scales) ** 2)

    assert np.isfinite(float(loss))
    assert float(metrics.rollout_valid_fraction) == 1.0
    np.testing.assert_allclose(metrics.code, expected_code, rtol=1e-6, atol=1e-7)
    assert rollout.states.shape == (2, 1, 4, 13)
    assert np.all(np.asarray(rollout.policy_valid))
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(gradient))
    assert bool(update.update_accepted)
    assert int(final.steps) == 1
    assert float(update.parameter_delta_norm) > 0


def test_hybrid_runner_integrates_with_the_real_shared_quad_bptt_objective() -> None:
    (
        model,
        actuator,
        spec,
        scenarios,
        initial,
        actor_config,
        quad_config,
        barrier_config,
        learning_config,
        loss_config,
        safety,
        _independent_params,
        targets,
        scales,
    ) = _quad_problem()
    params = initialize_shared_actor(
        jax.random.key(21), spec, dimension=3, n_obstacles=1, config=actor_config
    )

    def objective(candidate: object) -> jax.Array:
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
            quad_config,
            barrier_config,
            learning_config,
            loss_config,
        )[0]

    functions = build_quad_actor_bptt_functions(
        spec,
        model,
        actuator,
        actor_config,
        quad_config,
        barrier_config,
        learning_config,
        loss_config,
        learning_rate=1e-3,
        burst_steps=1,
    )
    result = run_hybrid_proposal_bptt(
        params,
        objective,
        functions,
        (initial, scenarios, safety, targets, params, scales),
        HybridProposalConfig(
            ProposalBudget(4, gradient_updates=1),
            seed=5,
            relative_stddev=1e-3,
            absolute_stddev=1e-5,
        ),
    )

    assert result.input_valid
    assert result.accounting.actual_objective_evaluations == 4
    assert result.accounting.gradient_evaluations == 1
    assert result.accounting.accepted_gradient_updates == 1
    assert np.isfinite(result.selected_loss)


def test_independent_shape_and_runtime_boundaries_fail_closed() -> None:
    spec, scenarios, config = _planar_problem()
    params = initialize_independent_actor(
        jax.random.key(5), spec, dimension=2, n_obstacles=1, config=config
    )
    malformed = params.replace(input_kernel=params.input_kernel[0])
    with pytest.raises(ValueError, match="input_kernel"):
        validate_independent_actor_shapes(malformed, spec, dimension=2, n_obstacles=1)
    with pytest.raises(ValueError, match="state policy/scenario"):
        independent_fallback_actions(
            params,
            spec,
            jnp.zeros((2, 1, 4)),
            scenarios,
            elapsed=jnp.array(0.0),
            horizon_duration=1.0,
            policy_gain=1.0,
            action_limit=1.0,
            config=config,
        )

    invalid = scenarios.replace(obstacle_centers=jnp.full((1, 1, 2), jnp.inf))
    actions = jax.jit(
        lambda batch: independent_fallback_actions(
            params,
            spec,
            jnp.zeros((3, 1, 4)),
            batch,
            elapsed=jnp.array(0.0),
            horizon_duration=1.0,
            policy_gain=1.0,
            action_limit=1.0,
            config=config,
        )
    )(invalid)
    assert np.isnan(np.asarray(actions)).all()
