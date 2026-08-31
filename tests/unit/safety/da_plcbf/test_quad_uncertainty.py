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
from crazyflow.safety.da_plcbf.estimator import (
    DeterministicParameterSamples,
    EstimatorConfig,
    deterministic_parameter_samples,
    initialize_estimator,
)
from crazyflow.safety.da_plcbf.quad_actor_losses import (
    quad_safety_values,
    rigid_body_safety_batch_from_circles,
)
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.quad_rollouts import rollout_shared_quad_library
from crazyflow.safety.da_plcbf.quad_uncertainty import (
    duplicate_circle_scenarios_for_samples,
    rollout_shared_quad_library_under_uncertainty,
    uncertain_quad_safety_values,
    version_a_model_samples_from_estimator,
)
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig, VersionAModel
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


def _physical() -> tuple[VersionAModel, VersionAActuator]:
    parameters: dict[str, Any] = load_params("cf21B_500")
    inertia = jnp.asarray(parameters["J"])
    return (
        VersionAModel(
            mass=jnp.asarray(parameters["mass"]),
            gravity_vec=jnp.asarray(parameters["gravity_vec"]),
            inertia=inertia,
            inertia_inv=jnp.linalg.inv(inertia),
            drag_matrix=jnp.asarray(parameters["drag_matrix"]),
            wind_velocity=jnp.zeros(3),
            external_force=jnp.zeros(3),
            external_torque=jnp.zeros(3),
        ),
        VersionAActuator(
            arm_length=jnp.asarray(parameters["L"]),
            thrust_to_torque=jnp.asarray(parameters["thrust2torque"]),
            mixing_matrix=jnp.asarray(parameters["mixing_matrix"]),
            thrust_min=jnp.asarray(parameters["thrust_min"]),
            thrust_max=jnp.asarray(parameters["thrust_max"]),
        ),
    )


def _actor_problem() -> tuple[object, ...]:
    model, actuator = _physical()
    spec = SharedActorSpec(
        base_codes=jnp.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
        base_desired_velocities=jnp.asarray(
            [[0.45, 0.0, 0.0], [-0.35, 0.0, 0.0], [0.0, 0.35, 0.0]]
        ),
        base_durations=jnp.asarray([0.4, 0.4, 0.4]),
        adaptive_mask=jnp.asarray([False, True, True]),
    )
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.asarray(
            [[[0.55, 0.0, 1.0], [0.0, 0.0, 0.0]], [[-0.45, 0.1, 1.1], [0.2, -0.5, 1.0]]]
        ),
        obstacle_radii=jnp.asarray([[0.18, jnp.nan], [0.16, 0.12]]),
        obstacle_mask=jnp.asarray([[True, False], [True, True]]),
        arena_lower=jnp.asarray([[-2.0, -2.0, 0.2], [-2.0, -2.0, 0.2]]),
        arena_upper=jnp.asarray([[2.0, 2.0, 2.2], [2.0, 2.0, 2.2]]),
        speed_limit=jnp.asarray([2.5, 2.5]),
    )
    initial_states = jnp.asarray(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, -0.1, 1.1, 0.0, 0.0, 0.0, 1.0, -0.1, 0.05, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    actor_config = SharedActorConfig(hidden_width=8, max_duration=0.8)
    quad_config = QuadPolicyConfig()
    params = initialize_shared_actor(
        jax.random.key(29), spec, dimension=3, n_obstacles=2, config=actor_config
    )
    safety = rigid_body_safety_batch_from_circles(
        scenarios, angular_rate_max=8.0, tilt_max_radians=np.deg2rad(65.0)
    )
    return (
        model,
        actuator,
        spec,
        scenarios,
        initial_states,
        actor_config,
        quad_config,
        params,
        safety,
    )


def _estimator_samples(
    model: VersionAModel, sample_count: int, covariance_diagonal: np.ndarray
) -> tuple[EstimatorConfig, DeterministicParameterSamples]:
    config = EstimatorConfig()
    state = initialize_estimator(
        config,
        mass=float(model.mass),
        drag_force_coefficients=-np.diag(np.asarray(model.drag_matrix)),
        wind_velocity=np.asarray(model.wind_velocity),
        rotor_efficiency=1.0,
        covariance=np.diag(covariance_diagonal),
        model_version=7,
    )
    return config, deterministic_parameter_samples(state, sample_count=sample_count, config=config)


def _rollout(
    params: object, model_samples: object, problem: tuple[object, ...], *, horizon: int = 6
) -> object:
    (model, actuator, spec, scenarios, initial_states, actor_config, quad_config, _, _) = problem
    return rollout_shared_quad_library_under_uncertainty(
        params,
        spec,
        initial_states,
        scenarios,
        model,
        model_samples,
        actuator,
        dt=0.01,
        horizon=horizon,
        policy_gain=1.4,
        actor_config=actor_config,
        quad_config=quad_config,
    )


@pytest.mark.parametrize("sample_count", [4, 8])
def test_scenario_duplication_and_fixed_uncertainty_shapes(sample_count: int) -> None:
    problem = _actor_problem()
    model, _, _, scenarios, _, _, _, params, _ = problem
    covariance = np.zeros(11)
    config, parameter_samples = _estimator_samples(model, sample_count, covariance)
    model_samples = version_a_model_samples_from_estimator(parameter_samples, model, config)
    duplicated = duplicate_circle_scenarios_for_samples(scenarios, sample_count)
    rollouts = _rollout(params, model_samples, problem)

    assert duplicated.obstacle_centers.shape == (2 * sample_count, 2, 3)
    assert set(duplicated.__dataclass_fields__) == {
        "obstacle_centers",
        "obstacle_radii",
        "obstacle_mask",
        "arena_lower",
        "arena_upper",
        "speed_limit",
    }
    for batch_index in range(2):
        start = batch_index * sample_count
        stop = start + sample_count
        expected = np.broadcast_to(
            np.asarray(scenarios.obstacle_centers[batch_index]), (sample_count, 2, 3)
        )
        np.testing.assert_array_equal(duplicated.obstacle_centers[start:stop], expected)
    assert rollouts.states.shape == (3, 2, sample_count, 7, 13)
    assert rollouts.commanded_wrenches.shape == (3, 2, sample_count, 6, 4)
    assert rollouts.policy_valid.shape == (3, 2, sample_count, 6)
    assert np.all(np.asarray(rollouts.sample_valid))


def test_identical_nominal_particles_numerically_duplicate_nominal_rollout() -> None:
    problem = _actor_problem()
    (model, actuator, spec, scenarios, initial_states, actor_config, quad_config, params, _) = (
        problem
    )
    config, parameter_samples = _estimator_samples(model, 4, np.zeros(11))
    model_samples = version_a_model_samples_from_estimator(parameter_samples, model, config)
    nominal = rollout_shared_quad_library(
        params,
        spec,
        initial_states,
        scenarios,
        model,
        actuator,
        dt=0.01,
        horizon=6,
        policy_gain=1.4,
        actor_config=actor_config,
        quad_config=quad_config,
    )
    uncertain = jax.jit(lambda candidate: _rollout(candidate, model_samples, problem))(params)

    for sample_index in range(4):
        # CUDA uses a larger GEMM after the B/R flattening, so fp32 feedback trajectories are
        # mathematically equivalent but not bit-identical to the smaller nominal batch.
        np.testing.assert_allclose(
            uncertain.states[:, :, sample_index], nominal.states, atol=2e-3, rtol=2e-3
        )
        np.testing.assert_allclose(
            uncertain.commanded_wrenches[:, :, sample_index], nominal.wrenches, atol=2e-4, rtol=2e-4
        )
        np.testing.assert_allclose(
            uncertain.realized_wrenches[:, :, sample_index], nominal.wrenches, atol=2e-4, rtol=2e-4
        )


def test_hard_robust_policy_margin_is_exact_minimum_over_r() -> None:
    problem = _actor_problem()
    model, _, _, _, _, _, _, params, safety = problem
    covariance = np.zeros(11)
    covariance[0] = 4.0
    covariance[7] = 0.006
    config, parameter_samples = _estimator_samples(model, 4, covariance)
    model_samples = version_a_model_samples_from_estimator(parameter_samples, model, config)
    rollouts = _rollout(params, model_samples, problem, horizon=10)
    barrier_config = VersionABarrierConfig(obstacle_clearance=0.04)
    values = uncertain_quad_safety_values(rollouts, safety, barrier_config, softmin_beta=35.0)
    manual_hard = []
    for sample_index in range(4):
        sample_values = quad_safety_values(
            rollouts.states[:, :, sample_index], safety, barrier_config, softmin_beta=35.0
        )
        manual_hard.append(sample_values.hard_policy_margins)
    manual = jnp.stack(manual_hard, axis=2)

    np.testing.assert_allclose(values.hard_sample_margins, manual, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        values.robust_hard_policy_margins, jnp.min(manual, axis=2), atol=2e-6, rtol=2e-6
    )
    # At the common initial state, commands cannot depend on the hidden R index.  Only the sampled
    # rotor gain changes realized force; later commands may diverge through observed state feedback.
    initial_commands = np.asarray(rollouts.commanded_motor_forces[..., 0, :])
    initial_realized = np.asarray(rollouts.realized_motor_forces[..., 0, :])
    assert np.max(np.ptp(initial_commands, axis=2)) < 1e-7
    assert np.max(np.ptp(initial_realized, axis=2)) > 1e-4
    assert np.max(np.ptp(np.asarray(rollouts.states), axis=2)) > 1e-6
    assert np.all(
        np.asarray(values.robust_smooth_policy_margins)
        <= np.asarray(values.robust_hard_policy_margins) + 1e-6
    )


def test_invalid_or_nonfinite_parameter_set_fails_closed() -> None:
    problem = _actor_problem()
    model, _, _, _, _, _, _, params, safety = problem
    config, valid_parameters = _estimator_samples(model, 4, np.zeros(11))
    invalid_parameters = valid_parameters._replace(valid=jnp.asarray(False))
    invalid_samples = version_a_model_samples_from_estimator(invalid_parameters, model, config)
    invalid_rollouts = _rollout(params, invalid_samples, problem)
    invalid_values = uncertain_quad_safety_values(
        invalid_rollouts, safety, VersionABarrierConfig(), softmin_beta=30.0
    )

    assert not np.any(np.asarray(invalid_samples.sample_valid))
    assert not np.any(np.asarray(invalid_rollouts.policy_valid))
    assert np.all(np.isnan(np.asarray(invalid_rollouts.states[..., 1:, :])))
    assert np.all(np.isneginf(np.asarray(invalid_values.robust_hard_policy_margins)))

    nonfinite_vectors = valid_parameters.estimation_vectors.at[0, 0].set(jnp.nan)
    nonfinite_parameters = valid_parameters._replace(estimation_vectors=nonfinite_vectors)
    checked = version_a_model_samples_from_estimator(nonfinite_parameters, model, config)
    assert not bool(checked.sample_valid[0])
    assert np.all(np.asarray(checked.sample_valid[1:]))
    checked_rollouts = _rollout(params, checked, problem)
    checked_values = uncertain_quad_safety_values(
        checked_rollouts, safety, VersionABarrierConfig(), softmin_beta=30.0
    )
    assert np.all(np.isneginf(np.asarray(checked_values.robust_hard_policy_margins)))


def test_bad_parameter_sample_shapes_are_rejected() -> None:
    model, *_ = _actor_problem()
    config, samples = _estimator_samples(model, 4, np.zeros(11))
    bad = samples._replace(weights=samples.weights[:-1])
    with pytest.raises(ValueError, match="weights"):
        version_a_model_samples_from_estimator(bad, model, config)


def test_uncertainty_rollout_executes_on_an_actual_gpu_when_available() -> None:
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpu_devices:
        pytest.skip("an actual GPU is not available in this environment")
    problem = _actor_problem()
    model, _, _, _, _, _, _, params, safety = problem
    covariance = np.zeros(11)
    covariance[0] = 1.0
    covariance[7] = 0.002
    config, parameter_samples = _estimator_samples(model, 8, covariance)
    model_samples = version_a_model_samples_from_estimator(parameter_samples, model, config)

    def compiled(candidate: object) -> tuple[jax.Array, jax.Array]:
        rollouts = _rollout(candidate, model_samples, problem, horizon=4)
        values = uncertain_quad_safety_values(
            rollouts, safety, VersionABarrierConfig(), softmin_beta=30.0
        )
        return rollouts.states, values.robust_hard_policy_margins

    states, margins = jax.jit(compiled)(params)
    states.block_until_ready()
    assert next(iter(states.devices())).platform == "gpu"
    assert states.shape == (3, 2, 8, 5, 13)
    assert np.all(np.isfinite(np.asarray(margins)))
