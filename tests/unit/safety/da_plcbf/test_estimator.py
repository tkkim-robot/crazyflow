from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.direct_wrench import (
    direct_wrench_dynamics,
    quaternion_to_rotation_matrix,
)
from crazyflow.safety.da_plcbf.estimator import (
    EstimatorConfig,
    EstimatorState,
    EstimatorUpdateStatus,
    RotorEfficiencyObservations,
    TranslationalObservations,
    deterministic_parameter_samples,
    estimation_vector,
    initialize_estimator,
    jit_deterministic_parameter_samples,
    jit_update_rotor_efficiency,
    jit_update_translational_estimate,
    physical_parameters,
    update_rotor_efficiency,
    update_translational_estimate,
)


def _accurate_config() -> EstimatorConfig:
    return EstimatorConfig(
        acceleration_noise_std=1e-3,
        motor_force_noise_std=1e-5,
        initial_covariance_diagonal=(1e4, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0),
    )


def _initial_state(config: EstimatorConfig | None = None) -> EstimatorState:
    config = _accurate_config() if config is None else config
    return initialize_estimator(
        config,
        mass=0.045,
        drag_force_coefficients=jnp.array([0.0015, 0.0015, 0.0015]),
        wind_velocity=jnp.zeros(3),
        rotor_efficiency=1.0,
    )


def _synthetic_translational_window(
    *,
    mass: float = 0.032,
    drag_force: jax.Array = jnp.array([0.002, 0.003, 0.004]),
    wind: jax.Array = jnp.array([0.6, -0.3, 0.2]),
    size: int = 128,
) -> TranslationalObservations:
    keys = jax.random.split(jax.random.key(73), 3)
    quaternions = jax.random.normal(keys[0], (size, 4))
    rotation = quaternion_to_rotation_matrix(quaternions)
    velocity = 2 * jax.random.normal(keys[1], (size, 3))
    thrust = 0.1 + 0.5 * jax.random.uniform(keys[2], (size,))
    gravity = jnp.broadcast_to(jnp.array([0.0, 0.0, -9.81]), (size, 3))
    acceleration = direct_wrench_dynamics(
        jnp.zeros((size, 3)),
        quaternions,
        velocity,
        jnp.zeros((size, 3)),
        jnp.concatenate((thrust[:, None], jnp.zeros((size, 3))), axis=-1),
        mass=mass,
        gravity_vec=gravity,
        J=jnp.eye(3),
        drag_matrix=-jnp.diag(drag_force),
        wind_velocity=wind,
    ).vel_dot
    return TranslationalObservations(
        rotation_body_to_world=rotation,
        velocity_world=velocity,
        acceleration_world=acceleration,
        collective_thrust=thrust,
        gravity_world=gravity,
        sample_mask=jnp.ones((size,), dtype=bool),
    )


def _assert_state_identical(actual: EstimatorState, expected: EstimatorState) -> None:
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


@pytest.mark.unit
def test_joint_translational_update_recovers_mass_drag_and_wind_change_under_excitation() -> None:
    config = _accurate_config()
    state = _initial_state(config)
    observations = _synthetic_translational_window()

    result = jit_update_translational_estimate(state, observations, sequence=8, config=config)
    parameters = physical_parameters(result.state)

    assert result.status == EstimatorUpdateStatus.ACCEPTED
    assert result.identifiability_score > config.normalized_rank_tolerance
    assert result.innovation_rmse < 1e-4
    np.testing.assert_allclose(parameters.mass, 0.032, rtol=2e-5)
    np.testing.assert_allclose(
        parameters.drag_force_coefficients, jnp.array([0.002, 0.003, 0.004]), rtol=1e-3
    )
    np.testing.assert_allclose(
        parameters.wind_velocity, jnp.array([0.6, -0.3, 0.2]), rtol=3e-4, atol=1e-4
    )
    assert result.state.model_version == 1
    assert result.state.last_translational_sequence == 8
    assert np.linalg.eigvalsh(np.asarray(result.state.covariance))[0] >= -2e-7


@pytest.mark.unit
def test_unexcited_translational_window_is_rejected_without_prior_rank_leakage() -> None:
    config = _accurate_config()
    state = _initial_state(config)
    size = 16
    observation = TranslationalObservations(
        rotation_body_to_world=jnp.broadcast_to(jnp.eye(3), (size, 3, 3)),
        velocity_world=jnp.zeros((size, 3)),
        acceleration_world=jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (size, 3)),
        collective_thrust=jnp.ones((size,)),
        gravity_world=jnp.broadcast_to(jnp.array([0.0, 0.0, -9.81]), (size, 3)),
        sample_mask=jnp.ones((size,), dtype=bool),
    )

    result = update_translational_estimate(state, observation, sequence=0, config=config)

    assert result.status == EstimatorUpdateStatus.UNIDENTIFIABLE
    assert result.identifiability_score < config.normalized_rank_tolerance
    _assert_state_identical(result.state, state)


@pytest.mark.unit
def test_stale_nonfinite_and_invalid_rotation_updates_preserve_state() -> None:
    config = _accurate_config()
    observations = _synthetic_translational_window(size=32)
    accepted = update_translational_estimate(
        _initial_state(config), observations, sequence=3, config=config
    ).state

    stale = update_translational_estimate(accepted, observations, sequence=3, config=config)
    assert stale.status == EstimatorUpdateStatus.STALE_SEQUENCE
    _assert_state_identical(stale.state, accepted)

    nonfinite_observations = observations._replace(
        acceleration_world=observations.acceleration_world.at[0, 0].set(jnp.nan)
    )
    nonfinite = update_translational_estimate(
        accepted, nonfinite_observations, sequence=4, config=config
    )
    assert nonfinite.status == EstimatorUpdateStatus.NONFINITE
    _assert_state_identical(nonfinite.state, accepted)

    invalid_rotation = observations._replace(
        rotation_body_to_world=observations.rotation_body_to_world.at[0].set(2 * jnp.eye(3))
    )
    invalid = update_translational_estimate(accepted, invalid_rotation, sequence=4, config=config)
    assert invalid.status == EstimatorUpdateStatus.INVALID_OBSERVATION
    _assert_state_identical(invalid.state, accepted)

    invalid_version = accepted._replace(model_version=jnp.asarray(-1, dtype=jnp.int32))
    rejected_version = jit_update_translational_estimate(
        invalid_version, observations, sequence=4, config=config
    )
    assert rejected_version.status == EstimatorUpdateStatus.INVALID_OBSERVATION
    _assert_state_identical(rejected_version.state, invalid_version)


@pytest.mark.unit
def test_masked_nonfinite_padding_is_ignored_by_jitted_update() -> None:
    config = _accurate_config()
    observations = _synthetic_translational_window(size=33)
    padded = observations._replace(
        rotation_body_to_world=observations.rotation_body_to_world.at[-1].set(jnp.nan),
        velocity_world=observations.velocity_world.at[-1].set(jnp.nan),
        acceleration_world=observations.acceleration_world.at[-1].set(jnp.nan),
        collective_thrust=observations.collective_thrust.at[-1].set(jnp.nan),
        gravity_world=observations.gravity_world.at[-1].set(jnp.nan),
        sample_mask=observations.sample_mask.at[-1].set(False),
    )

    result = jit_update_translational_estimate(
        _initial_state(config), padded, sequence=0, config=config
    )

    assert result.status == EstimatorUpdateStatus.ACCEPTED
    assert np.all(np.isfinite(np.asarray(estimation_vector(result.state))))


@pytest.mark.unit
def test_symmetric_then_per_rotor_efficiency_updates_recover_measured_changes() -> None:
    config = _accurate_config()
    state = _initial_state(config)
    command = 0.02 + 0.08 * jax.random.uniform(jax.random.key(19), (40, 4))
    mask = jnp.ones_like(command, dtype=bool)
    symmetric = RotorEfficiencyObservations(command, 0.73 * command, mask)

    symmetric_result = jit_update_rotor_efficiency(
        state, symmetric, sequence=5, mode="symmetric", config=config
    )
    assert symmetric_result.status == EstimatorUpdateStatus.ACCEPTED
    np.testing.assert_allclose(symmetric_result.state.rotor_efficiency, 0.73, rtol=2e-5)

    changed_efficiency = jnp.array([0.82, 0.91, 0.76, 0.88])
    per_rotor = RotorEfficiencyObservations(command, command * changed_efficiency, mask)
    per_rotor_result = jit_update_rotor_efficiency(
        symmetric_result.state, per_rotor, sequence=6, mode="per_rotor", config=config
    )

    assert per_rotor_result.status == EstimatorUpdateStatus.ACCEPTED
    np.testing.assert_allclose(
        per_rotor_result.state.rotor_efficiency, changed_efficiency, rtol=2e-5, atol=2e-6
    )
    assert per_rotor_result.state.model_version == 2
    assert per_rotor_result.state.last_rotor_sequence == 6
    assert np.linalg.eigvalsh(np.asarray(per_rotor_result.state.covariance))[0] >= -2e-7


@pytest.mark.unit
def test_per_rotor_fit_rejects_an_unexcited_channel_and_nonfinite_measurement() -> None:
    config = _accurate_config()
    state = _initial_state(config)
    command = jnp.full((12, 4), 0.05).at[:, 2].set(0)
    observations = RotorEfficiencyObservations(
        command, 0.8 * command, jnp.ones_like(command, dtype=bool)
    )

    unidentifiable = update_rotor_efficiency(
        state, observations, sequence=0, mode="per_rotor", config=config
    )
    assert unidentifiable.status == EstimatorUpdateStatus.UNIDENTIFIABLE
    _assert_state_identical(unidentifiable.state, state)

    nonfinite = observations._replace(
        realized_motor_forces=observations.realized_motor_forces.at[0, 0].set(jnp.inf)
    )
    rejected = update_rotor_efficiency(
        state, nonfinite, sequence=0, mode="symmetric", config=config
    )
    assert rejected.status == EstimatorUpdateStatus.NONFINITE
    _assert_state_identical(rejected.state, state)


@pytest.mark.unit
@pytest.mark.parametrize("sample_count", [4, 8])
def test_deterministic_samples_are_symmetric_bounded_and_match_retained_covariance(
    sample_count: int,
) -> None:
    config = _accurate_config()
    diagonal = np.linspace(1e-5, 11e-5, 11)
    state = initialize_estimator(
        config,
        mass=0.04,
        drag_force_coefficients=jnp.array([0.004, 0.0048, 0.0056]),
        wind_velocity=jnp.array([0.2, -0.1, 0.3]),
        rotor_efficiency=jnp.array([0.8, 0.85, 0.9, 0.95]),
        covariance=np.diag(diagonal),
        model_version=7,
    )

    samples = deterministic_parameter_samples(state, sample_count=sample_count, config=config)
    vectors = np.asarray(samples.estimation_vectors)
    center = np.asarray(estimation_vector(state))
    empirical_mean = np.sum(np.asarray(samples.weights)[:, None] * vectors, axis=0)
    centered = vectors - empirical_mean
    empirical_covariance = np.einsum("r,ri,rj->ij", np.asarray(samples.weights), centered, centered)
    rank = sample_count // 2
    eigenvalues, eigenvectors = np.linalg.eigh(np.diag(diagonal))
    expected = (eigenvectors[:, -rank:] * eigenvalues[-rank:]) @ eigenvectors[:, -rank:].T

    assert samples.valid
    assert samples.model_version == 7
    assert vectors.shape == (sample_count, 11)
    np.testing.assert_allclose(np.sum(samples.weights), 1.0)
    np.testing.assert_allclose(empirical_mean, center, atol=2e-6)
    np.testing.assert_allclose(empirical_covariance, expected, atol=2e-7)
    assert np.all(np.asarray(samples.parameters.mass) >= config.mass_bounds[0])
    assert np.all(np.asarray(samples.parameters.mass) <= config.mass_bounds[1])
    assert np.all(np.asarray(samples.parameters.rotor_efficiency) >= config.efficiency_bounds[0])
    assert np.all(np.asarray(samples.parameters.rotor_efficiency) <= config.efficiency_bounds[1])


@pytest.mark.unit
def test_samples_scale_symmetrically_at_bounds_and_flag_invalid_covariance() -> None:
    config = _accurate_config()
    covariance = np.eye(11) * 0.1
    state = initialize_estimator(
        config,
        mass=config.mass_bounds[0],
        drag_force_coefficients=jnp.full(
            (3,), config.drag_acceleration_bounds[0] * config.mass_bounds[0]
        ),
        wind_velocity=jnp.asarray(config.wind_upper),
        rotor_efficiency=config.efficiency_bounds[1],
        covariance=covariance,
    )
    samples = deterministic_parameter_samples(state, sample_count=8, config=config)

    assert samples.valid
    np.testing.assert_allclose(
        np.mean(np.asarray(samples.estimation_vectors), axis=0),
        np.asarray(estimation_vector(state)),
        atol=2e-6,
    )
    assert np.all(np.isfinite(np.asarray(samples.estimation_vectors)))

    invalid_state = state._replace(covariance=state.covariance.at[0, 0].set(-1.0))
    invalid = deterministic_parameter_samples(invalid_state, sample_count=4, config=config)
    assert not invalid.valid
    assert np.all(np.isfinite(np.asarray(invalid.estimation_vectors)))


@pytest.mark.unit
def test_jitted_samples_feed_vmapped_direct_wrench_rollouts() -> None:
    config = _accurate_config()
    state = update_translational_estimate(
        _initial_state(config), _synthetic_translational_window(size=64), sequence=0, config=config
    ).state
    samples = jit_deterministic_parameter_samples(state, sample_count=8, config=config)
    position = jnp.array([0.0, 0.0, 1.0])
    quaternion = jnp.array([0.0, 0.0, 0.0, 1.0])
    velocity = jnp.array([0.4, -0.2, 0.1])
    body_rate = jnp.zeros(3)
    wrench = jnp.array([0.35, 0.0, 0.0, 0.0])
    gravity = jnp.array([0.0, 0.0, -9.81])
    inertia = jnp.diag(jnp.array([2e-5, 2.1e-5, 3.2e-5]))

    derivatives = jax.jit(
        jax.vmap(
            lambda mass, drag, wind: (
                direct_wrench_dynamics(
                    position,
                    quaternion,
                    velocity,
                    body_rate,
                    wrench,
                    mass=mass,
                    gravity_vec=gravity,
                    J=inertia,
                    drag_matrix=drag,
                    wind_velocity=wind,
                ).vel_dot
            )
        )
    )(samples.parameters.mass, samples.parameters.drag_matrix, samples.parameters.wind_velocity)

    assert samples.valid
    assert derivatives.shape == (8, 3)
    assert np.all(np.isfinite(np.asarray(derivatives)))


@pytest.mark.unit
def test_initialization_rejects_non_psd_cross_block_and_out_of_bounds_parameters() -> None:
    config = _accurate_config()
    covariance = np.eye(11)
    covariance[0, 7] = covariance[7, 0] = 0.1
    with pytest.raises(ValueError, match="separate"):
        initialize_estimator(
            config,
            mass=0.04,
            drag_force_coefficients=jnp.full((3,), 0.002),
            wind_velocity=jnp.zeros(3),
            covariance=covariance,
        )
    not_psd = np.eye(11)
    not_psd[0, 0] = -1
    with pytest.raises(ValueError, match="positive semidefinite"):
        initialize_estimator(
            config,
            mass=0.04,
            drag_force_coefficients=jnp.full((3,), 0.002),
            wind_velocity=jnp.zeros(3),
            covariance=not_psd,
        )
    with pytest.raises(ValueError, match="bounds"):
        initialize_estimator(
            config,
            mass=1.0,
            drag_force_coefficients=jnp.full((3,), 0.002),
            wind_velocity=jnp.zeros(3),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "message"),
    [
        (replace(EstimatorConfig(), mass_bounds=(0.1, 0.1)), "mass_bounds"),
        (replace(EstimatorConfig(), acceleration_noise_std=jnp.inf), "acceleration_noise_std"),
        (replace(EstimatorConfig(), gauss_newton_iterations=1.5), "gauss_newton_iterations"),
        (replace(EstimatorConfig(), process_noise_diagonal=(0.0,) * 10), "process_noise_diagonal"),
    ],
)
def test_estimator_config_rejects_invalid_values(config: EstimatorConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


@pytest.mark.unit
def test_shapes_modes_sequences_and_sample_counts_fail_before_execution() -> None:
    config = _accurate_config()
    state = _initial_state(config)
    observations = _synthetic_translational_window(size=8)
    with pytest.raises(TypeError, match="sequence"):
        update_translational_estimate(state, observations, sequence=0.5, config=config)
    with pytest.raises(ValueError, match="sample_count"):
        deterministic_parameter_samples(state, sample_count=6, config=config)  # type: ignore[arg-type]
    rotor = RotorEfficiencyObservations(
        jnp.ones((2, 4)), jnp.ones((2, 4)), jnp.ones((2, 4), dtype=bool)
    )
    with pytest.raises(ValueError, match="mode"):
        update_rotor_efficiency(
            state,
            rotor,
            sequence=0,
            mode="global",
            config=config,  # type: ignore[arg-type]
        )
