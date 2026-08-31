"""Low-dimensional online dynamics estimation and deterministic uncertainty samples.

The translational estimator uses the Version-A model

``a = g + mu * T * R[:, 2] - R @ (kappa * (R.T @ (v - wind)))``

where ``mu = 1 / mass`` and ``kappa = positive_drag_force / mass``.  A bounded, iterated
Gauss--Newton update estimates ``[mu, kappa(3), wind(3)]`` from a fixed observation window.  The
measurement Jacobian must have full normalized column rank; the prior covariance is deliberately
*not* allowed to make an unidentifiable window appear identifiable.

These parameters are identifiable only when the window excites collective thrust, relative air
velocity, attitude, and all three wind/drag directions.  Mass and rotor efficiency are confounded
if only commanded thrust and translational acceleration are observed.  Rotor efficiency updates
therefore require measured realized motor forces (or an independently reconstructed equivalent).
The symmetric and per-rotor fits are separate models: fitting a global efficiency and four free
relative efficiencies simultaneously would be redundant without an additional gauge constraint.

Every accepted estimator update advances ``model_version``.  Stale, non-finite, physically invalid,
or locally rank-deficient updates return the input state unchanged.  The deterministic ``R=4`` and
``R=8`` particles retain the leading two or four covariance eigendirections, respectively.  They
are bounded, symmetric, equally weighted scenario samples, not a claim that four/eight particles
exactly reproduce an eleven-dimensional distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from numbers import Integral
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from crazyflow._typing import Array


_TRANSLATIONAL_DIM = 7
_ROTOR_DIM = 4
_PARAMETER_DIM = _TRANSLATIONAL_DIM + _ROTOR_DIM


class EstimatorUpdateStatus(IntEnum):
    """Machine-readable outcome of an estimator update."""

    ACCEPTED = 0
    STALE_SEQUENCE = 1
    NONFINITE = 2
    INVALID_OBSERVATION = 3
    UNIDENTIFIABLE = 4
    NUMERICAL_FAILURE = 5


@dataclass(frozen=True)
class EstimatorConfig:
    """Static bounds and numerical settings for the parametric estimator.

    Drag bounds apply to ``kappa = positive_drag_force / mass`` in ``s^-1``.  Covariance entries
    use the estimation-coordinate order ``[inverse_mass, kappa(3), wind(3), efficiency(4)]``.
    """

    mass_bounds: tuple[float, float] = (0.01, 0.25)
    drag_acceleration_bounds: tuple[float, float] = (1e-4, 5.0)
    wind_lower: tuple[float, float, float] = (-10.0, -10.0, -10.0)
    wind_upper: tuple[float, float, float] = (10.0, 10.0, 10.0)
    efficiency_bounds: tuple[float, float] = (0.2, 1.2)
    acceleration_noise_std: float = 0.03
    motor_force_noise_std: float = 5e-4
    initial_covariance_diagonal: tuple[float, ...] = (
        100.0,
        0.25,
        0.25,
        0.25,
        1.0,
        1.0,
        1.0,
        0.04,
        0.04,
        0.04,
        0.04,
    )
    process_noise_diagonal: tuple[float, ...] = (
        1.0,
        1e-3,
        1e-3,
        1e-3,
        1e-2,
        1e-2,
        1e-2,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
    )
    gauss_newton_iterations: int = 8
    normalized_rank_tolerance: float = 1e-5
    minimum_column_norm: float = 1e-8
    minimum_rotor_excitation: float = 1e-7
    covariance_floor: float = 1e-10
    psd_tolerance: float = 1e-6
    rotation_tolerance: float = 2e-3

    def validate(self) -> None:
        """Reject nonphysical bounds and numerically unsafe settings."""
        _validate_ordered_positive_bounds(self.mass_bounds, "mass_bounds")
        _validate_ordered_nonnegative_bounds(
            self.drag_acceleration_bounds, "drag_acceleration_bounds"
        )
        _validate_ordered_positive_bounds(self.efficiency_bounds, "efficiency_bounds")
        _validate_vector_bounds(self.wind_lower, self.wind_upper, "wind")
        for name, value in (
            ("acceleration_noise_std", self.acceleration_noise_std),
            ("motor_force_noise_std", self.motor_force_noise_std),
            ("normalized_rank_tolerance", self.normalized_rank_tolerance),
            ("minimum_column_norm", self.minimum_column_norm),
            ("minimum_rotor_excitation", self.minimum_rotor_excitation),
            ("covariance_floor", self.covariance_floor),
            ("psd_tolerance", self.psd_tolerance),
            ("rotation_tolerance", self.rotation_tolerance),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.gauss_newton_iterations, bool)
            or not isinstance(self.gauss_newton_iterations, Integral)
            or self.gauss_newton_iterations <= 0
        ):
            raise ValueError("gauss_newton_iterations must be a positive integer")
        _validate_covariance_diagonal(
            self.initial_covariance_diagonal, "initial_covariance_diagonal", strictly_positive=True
        )
        _validate_covariance_diagonal(
            self.process_noise_diagonal, "process_noise_diagonal", strictly_positive=False
        )


class EstimatorState(NamedTuple):
    """JAX PyTree containing the current estimate and independent block covariance."""

    inverse_mass: Array
    drag_acceleration: Array
    wind_velocity: Array
    rotor_efficiency: Array
    covariance: Array
    model_version: Array
    last_translational_sequence: Array
    last_rotor_sequence: Array


class DynamicsParameters(NamedTuple):
    """Physical dynamics parameters derived from estimation coordinates."""

    mass: Array
    inverse_mass: Array
    drag_acceleration: Array
    drag_force_coefficients: Array
    drag_matrix: Array
    wind_velocity: Array
    rotor_efficiency: Array


class TranslationalObservations(NamedTuple):
    """A fixed-size translational estimation window.

    ``collective_thrust`` is the measured *realized* airborne thrust, not merely a command.  Masked
    padding may contain non-finite sentinels and is ignored.  All real rotations must be proper
    body-to-world rotation matrices.
    """

    rotation_body_to_world: Array
    velocity_world: Array
    acceleration_world: Array
    collective_thrust: Array
    gravity_world: Array
    sample_mask: Array


class RotorEfficiencyObservations(NamedTuple):
    """Commanded and independently measured realized motor forces, shape ``[N, 4]``."""

    commanded_motor_forces: Array
    realized_motor_forces: Array
    sample_mask: Array


class EstimatorUpdate(NamedTuple):
    """Estimator state plus finite/rank diagnostics for one attempted update."""

    state: EstimatorState
    status: Array
    innovation_rmse: Array
    identifiability_score: Array
    information_condition_number: Array


class DeterministicParameterSamples(NamedTuple):
    """Bounded low-rank covariance particles suitable for ``jax.vmap`` rollouts."""

    parameters: DynamicsParameters
    estimation_vectors: Array
    weights: Array
    valid: Array
    retained_variance_fraction: Array
    model_version: Array


def initialize_estimator(
    config: EstimatorConfig,
    *,
    mass: float,
    drag_force_coefficients: Array,
    wind_velocity: Array,
    rotor_efficiency: Array | float = 1.0,
    covariance: Array | None = None,
    model_version: int = 0,
) -> EstimatorState:
    """Create a validated estimator state from physical parameters.

    Positive ``drag_force_coefficients`` correspond to ``-diag(coefficients)`` in
    :func:`direct_wrench_dynamics`.
    """
    config.validate()
    if not math.isfinite(mass) or not config.mass_bounds[0] <= mass <= config.mass_bounds[1]:
        raise ValueError("mass must be finite and inside mass_bounds")
    if (
        isinstance(model_version, bool)
        or not isinstance(model_version, Integral)
        or model_version < 0
    ):
        raise ValueError("model_version must be a nonnegative integer")

    drag_force = np.asarray(drag_force_coefficients, dtype=np.float64)
    wind = np.asarray(wind_velocity, dtype=np.float64)
    efficiency = np.broadcast_to(np.asarray(rotor_efficiency, dtype=np.float64), (4,)).copy()
    if drag_force.shape != (3,) or wind.shape != (3,):
        raise ValueError("drag_force_coefficients and wind_velocity must have shape (3,)")
    if not np.all(np.isfinite(drag_force)) or np.any(drag_force < 0):
        raise ValueError("drag_force_coefficients must be finite and nonnegative")
    if not np.all(np.isfinite(wind)):
        raise ValueError("wind_velocity must be finite")
    drag_acceleration = drag_force / mass
    lower, upper = _host_parameter_bounds(config)
    vector = np.concatenate(([1.0 / mass], drag_acceleration, wind, efficiency))
    below = (vector < lower) & ~np.isclose(vector, lower, rtol=1e-6, atol=1e-12)
    above = (vector > upper) & ~np.isclose(vector, upper, rtol=1e-6, atol=1e-12)
    if np.any(below) or np.any(above):
        raise ValueError("initial physical parameters violate configured bounds")
    vector = np.clip(vector, lower, upper)

    if covariance is None:
        covariance_array = np.diag(np.asarray(config.initial_covariance_diagonal, dtype=np.float64))
    else:
        covariance_array = np.asarray(covariance, dtype=np.float64)
    _validate_covariance(covariance_array, config.psd_tolerance)
    if not np.allclose(
        covariance_array[:_TRANSLATIONAL_DIM, _TRANSLATIONAL_DIM:],
        0,
        atol=config.psd_tolerance,
        rtol=0,
    ):
        raise ValueError("initial covariance must separate translational and rotor blocks")

    dtype = jnp.asarray(mass).dtype
    return EstimatorState(
        inverse_mass=jnp.asarray(vector[0], dtype=dtype),
        drag_acceleration=jnp.asarray(vector[1:4], dtype=dtype),
        wind_velocity=jnp.asarray(vector[4:7], dtype=dtype),
        rotor_efficiency=jnp.asarray(vector[7:11], dtype=dtype),
        covariance=jnp.asarray(covariance_array, dtype=dtype),
        model_version=jnp.asarray(model_version, dtype=jnp.int32),
        last_translational_sequence=jnp.asarray(-1, dtype=jnp.int32),
        last_rotor_sequence=jnp.asarray(-1, dtype=jnp.int32),
    )


def estimation_vector(state: EstimatorState) -> Array:
    """Return ``[inverse_mass, drag_acceleration(3), wind(3), efficiency(4)]``."""
    return jnp.concatenate(
        (
            jnp.reshape(state.inverse_mass, (1,)),
            state.drag_acceleration,
            state.wind_velocity,
            state.rotor_efficiency,
        )
    )


def physical_parameters(state: EstimatorState) -> DynamicsParameters:
    """Convert an estimator state to direct-wrench physical parameters."""
    return _vectors_to_parameters(estimation_vector(state))


def update_translational_estimate(
    state: EstimatorState,
    observations: TranslationalObservations,
    *,
    sequence: Array | int,
    config: EstimatorConfig,
) -> EstimatorUpdate:
    """Attempt one bounded, locally identifiable translational window update.

    The function is pure and can be jitted with ``config`` static.  An accepted update advances the
    model version exactly once.  Every rejected result contains the original state byte-for-byte.
    """
    config.validate()
    _check_estimator_state_shapes(state)
    _check_sequence(sequence)
    _check_translational_shapes(observations)
    mask = observations.sample_mask
    mask_float = mask.astype(state.inverse_mass.dtype)
    row_mask = mask_float[:, None]
    rotation = _masked(observations.rotation_body_to_world, mask[:, None, None])
    velocity = _masked(observations.velocity_world, mask[:, None])
    acceleration = _masked(observations.acceleration_world, mask[:, None])
    thrust = _masked(observations.collective_thrust, mask)
    gravity = _masked(observations.gravity_world, mask[:, None])

    state_vector = estimation_vector(state)
    lower, upper = _parameter_bounds(config, state_vector.dtype)
    finite_state = _state_is_finite(state)
    finite_observations = _masked_all_finite(
        mask,
        observations.rotation_body_to_world,
        observations.velocity_world,
        observations.acceleration_world,
        observations.collective_thrust,
        observations.gravity_world,
    )
    state_in_bounds = jnp.all((state_vector >= lower) & (state_vector <= upper))
    state_covariance_psd = _is_psd(state.covariance, config.psd_tolerance)
    state_covariance_blocked = _has_independent_covariance_blocks(
        state.covariance, config.psd_tolerance
    )
    state_versions_valid = _state_versions_valid(state)
    fresh = jnp.asarray(sequence, dtype=jnp.int32) > state.last_translational_sequence
    rotation_error = _rotation_error(rotation, mask)
    physically_valid = (
        jnp.all(jnp.where(mask, observations.collective_thrust >= 0, True))
        & (rotation_error <= config.rotation_tolerance)
        & (jnp.sum(mask_float) * 3 >= _TRANSLATIONAL_DIM)
    )

    safe_state_vector = jnp.clip(
        jnp.nan_to_num(state_vector, nan=0.0, posinf=0.0, neginf=0.0), lower, upper
    )
    safe_covariance = jnp.where(
        finite_state & state_covariance_psd,
        state.covariance,
        jnp.eye(_PARAMETER_DIM, dtype=state.covariance.dtype),
    )
    prior_mean = safe_state_vector[:_TRANSLATIONAL_DIM]
    process_noise = jnp.asarray(config.process_noise_diagonal, dtype=state_vector.dtype)
    prior_covariance = _symmetric(
        safe_covariance[:_TRANSLATIONAL_DIM, :_TRANSLATIONAL_DIM]
        + jnp.diag(process_noise[:_TRANSLATIONAL_DIM])
    )
    prior_precision = _positive_definite_inverse(prior_covariance, config.covariance_floor)
    noise_variance = jnp.asarray(config.acceleration_noise_std**2, dtype=state_vector.dtype)

    _, initial_jacobian = _translation_prediction_and_jacobian(
        prior_mean, rotation, velocity, thrust, gravity
    )
    initial_design = initial_jacobian.reshape((-1, _TRANSLATIONAL_DIM)) * row_mask.repeat(
        3, axis=1
    ).reshape((-1, 1))
    initial_rank = _normalized_information_diagnostics(initial_design, config.minimum_column_norm)

    def gauss_newton_body(_: int, theta: Array) -> Array:
        prediction, jacobian = _translation_prediction_and_jacobian(
            theta, rotation, velocity, thrust, gravity
        )
        weighted_jacobian = jacobian * row_mask[..., None]
        residual = (acceleration - prediction) * row_mask
        information = (
            prior_precision
            + jnp.einsum("npi,npj->ij", weighted_jacobian, weighted_jacobian) / noise_variance
        )
        gradient = (
            prior_precision @ (prior_mean - theta)
            + jnp.einsum("npi,np->i", weighted_jacobian, residual) / noise_variance
        )
        delta = jnp.linalg.solve(
            information + config.covariance_floor * jnp.eye(_TRANSLATIONAL_DIM), gradient
        )
        return jnp.clip(theta + delta, lower[:_TRANSLATIONAL_DIM], upper[:_TRANSLATIONAL_DIM])

    final_theta = jax.lax.fori_loop(
        0, config.gauss_newton_iterations, gauss_newton_body, prior_mean
    )
    final_prediction, final_jacobian = _translation_prediction_and_jacobian(
        final_theta, rotation, velocity, thrust, gravity
    )
    weighted_final_jacobian = final_jacobian * row_mask[..., None]
    final_design = weighted_final_jacobian.reshape((-1, _TRANSLATIONAL_DIM))
    final_rank = _normalized_information_diagnostics(final_design, config.minimum_column_norm)
    posterior_information = (
        prior_precision
        + jnp.einsum("npi,npj->ij", weighted_final_jacobian, weighted_final_jacobian)
        / noise_variance
    )
    posterior_covariance = _symmetric(
        _positive_definite_inverse(posterior_information, config.covariance_floor)
    )
    innovation = (acceleration - final_prediction) * row_mask
    observation_count = jnp.maximum(3 * jnp.sum(mask_float), 1)
    innovation_rmse = jnp.sqrt(jnp.sum(innovation**2) / observation_count)

    identifiable = (
        (initial_rank[0] >= config.normalized_rank_tolerance)
        & (final_rank[0] >= config.normalized_rank_tolerance)
        & jnp.isfinite(initial_rank[1])
        & jnp.isfinite(final_rank[1])
    )
    numerically_valid = (
        jnp.all(jnp.isfinite(final_theta))
        & jnp.all(jnp.isfinite(posterior_covariance))
        & _is_psd(posterior_covariance, config.psd_tolerance)
        & jnp.isfinite(innovation_rmse)
    )
    accepted = (
        fresh
        & finite_state
        & finite_observations
        & state_in_bounds
        & state_covariance_psd
        & state_covariance_blocked
        & state_versions_valid
        & physically_valid
        & identifiable
        & numerically_valid
    )

    candidate_covariance = jnp.zeros_like(state.covariance)
    candidate_covariance = candidate_covariance.at[:_TRANSLATIONAL_DIM, :_TRANSLATIONAL_DIM].set(
        posterior_covariance
    )
    candidate_covariance = candidate_covariance.at[_TRANSLATIONAL_DIM:, _TRANSLATIONAL_DIM:].set(
        state.covariance[_TRANSLATIONAL_DIM:, _TRANSLATIONAL_DIM:]
    )
    candidate = EstimatorState(
        inverse_mass=final_theta[0],
        drag_acceleration=final_theta[1:4],
        wind_velocity=final_theta[4:7],
        rotor_efficiency=state.rotor_efficiency,
        covariance=candidate_covariance,
        model_version=state.model_version + jnp.asarray(1, dtype=state.model_version.dtype),
        last_translational_sequence=jnp.asarray(sequence, dtype=jnp.int32),
        last_rotor_sequence=state.last_rotor_sequence,
    )
    next_state = jax.lax.cond(accepted, lambda: candidate, lambda: state)
    status = _update_status(
        fresh=fresh,
        finite=finite_state & finite_observations,
        valid=(
            state_in_bounds
            & state_covariance_psd
            & state_covariance_blocked
            & state_versions_valid
            & physically_valid
        ),
        identifiable=identifiable,
        numerical=numerically_valid,
    )
    return EstimatorUpdate(
        state=next_state,
        status=status,
        innovation_rmse=jnp.where(finite_observations, innovation_rmse, jnp.inf),
        identifiability_score=jnp.minimum(initial_rank[0], final_rank[0]),
        information_condition_number=jnp.maximum(initial_rank[1], final_rank[1]),
    )


def update_rotor_efficiency(
    state: EstimatorState,
    observations: RotorEfficiencyObservations,
    *,
    sequence: Array | int,
    mode: Literal["symmetric", "per_rotor"],
    config: EstimatorConfig,
) -> EstimatorUpdate:
    """Fit either one symmetric or four independent effective rotor efficiencies.

    ``realized_motor_forces = efficiency * commanded_motor_forces`` is fit with a Gaussian prior.
    The symmetric mode assumes all four efficiencies are the same.  The per-rotor mode requires
    excitation of every rotor and never infers an unexcited channel from the others.
    """
    config.validate()
    _check_estimator_state_shapes(state)
    _check_sequence(sequence)
    if mode not in ("symmetric", "per_rotor"):
        raise ValueError("mode must be 'symmetric' or 'per_rotor'")
    _check_rotor_shapes(observations)
    mask = observations.sample_mask
    mask_float = mask.astype(state.inverse_mass.dtype)
    command = _masked(observations.commanded_motor_forces, mask)
    realized = _masked(observations.realized_motor_forces, mask)
    state_vector = estimation_vector(state)
    lower, upper = _parameter_bounds(config, state_vector.dtype)

    finite_state = _state_is_finite(state)
    finite_observations = _masked_all_finite(
        mask, observations.commanded_motor_forces, observations.realized_motor_forces
    )
    state_in_bounds = jnp.all((state_vector >= lower) & (state_vector <= upper))
    state_covariance_psd = _is_psd(state.covariance, config.psd_tolerance)
    state_covariance_blocked = _has_independent_covariance_blocks(
        state.covariance, config.psd_tolerance
    )
    state_versions_valid = _state_versions_valid(state)
    physically_valid = jnp.all(
        jnp.where(
            mask,
            (observations.commanded_motor_forces >= 0) & (observations.realized_motor_forces >= 0),
            True,
        )
    )
    fresh = jnp.asarray(sequence, dtype=jnp.int32) > state.last_rotor_sequence

    safe_covariance = jnp.where(
        finite_state & state_covariance_psd,
        state.covariance,
        jnp.eye(_PARAMETER_DIM, dtype=state.covariance.dtype),
    )
    process_noise = jnp.asarray(config.process_noise_diagonal, dtype=state_vector.dtype)
    prior_covariance = _symmetric(
        safe_covariance[_TRANSLATIONAL_DIM:, _TRANSLATIONAL_DIM:]
        + jnp.diag(process_noise[_TRANSLATIONAL_DIM:])
    )
    noise_variance = jnp.asarray(config.motor_force_noise_std**2, dtype=state_vector.dtype)
    efficiency_lower = lower[_TRANSLATIONAL_DIM]
    efficiency_upper = upper[_TRANSLATIONAL_DIM]

    if mode == "symmetric":
        prior_mean = jnp.mean(state.rotor_efficiency)
        ones = jnp.ones((_ROTOR_DIM,), dtype=state_vector.dtype)
        prior_variance = (ones @ prior_covariance @ ones) / (_ROTOR_DIM**2)
        prior_variance = jnp.maximum(prior_variance, config.covariance_floor)
        data_information = jnp.sum(mask_float * command**2) / noise_variance
        data_rhs = jnp.sum(mask_float * command * realized) / noise_variance
        posterior_variance = 1 / (1 / prior_variance + data_information)
        scalar_efficiency = posterior_variance * (prior_mean / prior_variance + data_rhs)
        scalar_efficiency = jnp.clip(scalar_efficiency, efficiency_lower, efficiency_upper)
        proposed_efficiency = jnp.full_like(state.rotor_efficiency, scalar_efficiency)
        posterior_covariance = posterior_variance * jnp.ones_like(prior_covariance)
        excitation = jnp.sqrt(jnp.sum(mask_float * command**2))
        identifiable = excitation >= config.minimum_rotor_excitation
        identifiability_score = excitation
        condition_number = jnp.asarray(1.0, dtype=state_vector.dtype)
    else:
        prior_precision = _positive_definite_inverse(prior_covariance, config.covariance_floor)
        data_information = jnp.sum(mask_float * command**2, axis=0) / noise_variance
        data_rhs = jnp.sum(mask_float * command * realized, axis=0) / noise_variance
        posterior_information = prior_precision + jnp.diag(data_information)
        posterior_covariance = _symmetric(
            _positive_definite_inverse(posterior_information, config.covariance_floor)
        )
        proposed_efficiency = posterior_covariance @ (
            prior_precision @ state.rotor_efficiency + data_rhs
        )
        proposed_efficiency = jnp.clip(proposed_efficiency, efficiency_lower, efficiency_upper)
        excitation = jnp.sqrt(jnp.sum(mask_float * command**2, axis=0))
        identifiable = jnp.all(excitation >= config.minimum_rotor_excitation)
        identifiability_score = jnp.min(excitation)
        condition_number = jnp.max(excitation) / jnp.maximum(
            identifiability_score, config.covariance_floor
        )

    predicted = command * proposed_efficiency[None, :]
    residual = (realized - predicted) * mask_float
    observation_count = jnp.maximum(jnp.sum(mask_float), 1)
    innovation_rmse = jnp.sqrt(jnp.sum(residual**2) / observation_count)
    numerically_valid = (
        jnp.all(jnp.isfinite(proposed_efficiency))
        & jnp.all(jnp.isfinite(posterior_covariance))
        & _is_psd(posterior_covariance, config.psd_tolerance)
        & jnp.isfinite(innovation_rmse)
    )
    accepted = (
        fresh
        & finite_state
        & finite_observations
        & state_in_bounds
        & state_covariance_psd
        & state_covariance_blocked
        & state_versions_valid
        & physically_valid
        & identifiable
        & numerically_valid
    )

    candidate_covariance = jnp.zeros_like(state.covariance)
    candidate_covariance = candidate_covariance.at[:_TRANSLATIONAL_DIM, :_TRANSLATIONAL_DIM].set(
        state.covariance[:_TRANSLATIONAL_DIM, :_TRANSLATIONAL_DIM]
    )
    candidate_covariance = candidate_covariance.at[_TRANSLATIONAL_DIM:, _TRANSLATIONAL_DIM:].set(
        posterior_covariance
    )
    candidate = EstimatorState(
        inverse_mass=state.inverse_mass,
        drag_acceleration=state.drag_acceleration,
        wind_velocity=state.wind_velocity,
        rotor_efficiency=proposed_efficiency,
        covariance=candidate_covariance,
        model_version=state.model_version + jnp.asarray(1, dtype=state.model_version.dtype),
        last_translational_sequence=state.last_translational_sequence,
        last_rotor_sequence=jnp.asarray(sequence, dtype=jnp.int32),
    )
    next_state = jax.lax.cond(accepted, lambda: candidate, lambda: state)
    status = _update_status(
        fresh=fresh,
        finite=finite_state & finite_observations,
        valid=(
            state_in_bounds
            & state_covariance_psd
            & state_covariance_blocked
            & state_versions_valid
            & physically_valid
        ),
        identifiable=identifiable,
        numerical=numerically_valid,
    )
    return EstimatorUpdate(
        state=next_state,
        status=status,
        innovation_rmse=jnp.where(finite_observations, innovation_rmse, jnp.inf),
        identifiability_score=identifiability_score,
        information_condition_number=condition_number,
    )


def deterministic_parameter_samples(
    state: EstimatorState, *, sample_count: Literal[4, 8], config: EstimatorConfig
) -> DeterministicParameterSamples:
    """Return bounded, symmetric particles along leading covariance eigendirections.

    For ``sample_count = 2 * rank``, direction ``i`` has displacement
    ``sqrt(rank * eigenvalue_i) * eigenvector_i``.  Thus, away from bounds, the equal-weight sample
    covariance exactly reproduces the retained rank-``rank`` covariance approximation.  At bounds,
    symmetric direction scaling preserves the mean and physical feasibility while conservatively
    shrinking covariance.
    """
    config.validate()
    _check_estimator_state_shapes(state)
    if sample_count not in (4, 8):
        raise ValueError("sample_count must be exactly 4 or 8")
    center = estimation_vector(state)
    lower, upper = _parameter_bounds(config, center.dtype)
    state_finite = _state_is_finite(state)
    center_in_bounds = jnp.all((center >= lower) & (center <= upper))
    covariance_finite = jnp.all(jnp.isfinite(state.covariance))
    safe_center = jnp.clip(jnp.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0), lower, upper)
    safe_covariance = jnp.where(
        covariance_finite,
        _symmetric(state.covariance),
        jnp.eye(_PARAMETER_DIM, dtype=center.dtype) * config.covariance_floor,
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(safe_covariance)
    covariance_psd = eigenvalues[0] >= -config.psd_tolerance
    eigenvalues = jnp.maximum(eigenvalues, 0)
    rank = sample_count // 2
    selected_values = eigenvalues[-rank:][::-1]
    selected_vectors = eigenvectors[:, -rank:][:, ::-1].T
    raw_deltas = jnp.sqrt(rank * selected_values)[:, None] * selected_vectors
    symmetric_room = jnp.minimum(safe_center - lower, upper - safe_center)
    ratios = jnp.where(
        jnp.abs(raw_deltas) > config.covariance_floor,
        symmetric_room[None, :] / jnp.abs(raw_deltas),
        jnp.inf,
    )
    direction_scale = jnp.clip(jnp.min(ratios, axis=-1), 0, 1)
    deltas = raw_deltas * direction_scale[:, None]
    vectors = jnp.concatenate((safe_center[None, :] + deltas, safe_center[None, :] - deltas))
    weights = jnp.full((sample_count,), 1 / sample_count, dtype=center.dtype)
    captured_variance = jnp.sum(deltas**2) / rank
    total_variance = jnp.sum(eigenvalues)
    retained_fraction = jnp.where(
        total_variance > config.covariance_floor,
        captured_variance / total_variance,
        jnp.asarray(1.0, dtype=center.dtype),
    )
    valid = (
        state_finite
        & center_in_bounds
        & covariance_finite
        & covariance_psd
        & _state_versions_valid(state)
    )
    return DeterministicParameterSamples(
        parameters=_vectors_to_parameters(vectors),
        estimation_vectors=vectors,
        weights=weights,
        valid=valid,
        retained_variance_fraction=retained_fraction,
        model_version=state.model_version,
    )


def _translation_prediction_and_jacobian(
    theta: Array, rotation: Array, velocity: Array, thrust: Array, gravity: Array
) -> tuple[Array, Array]:
    inverse_mass = theta[0]
    drag_acceleration = theta[1:4]
    wind = theta[4:7]
    relative_air_body = jnp.einsum("nji,nj->ni", rotation, velocity - wind)
    drag_world = jnp.einsum("nij,nj->ni", rotation, drag_acceleration * relative_air_body)
    prediction = gravity + inverse_mass * thrust[:, None] * rotation[:, :, 2] - drag_world
    inverse_mass_jacobian = (thrust[:, None] * rotation[:, :, 2])[..., None]
    drag_jacobian = -rotation * relative_air_body[:, None, :]
    wind_jacobian = jnp.einsum("nij,nj,nkj->nik", rotation, drag_acceleration[None, :], rotation)
    jacobian = jnp.concatenate((inverse_mass_jacobian, drag_jacobian, wind_jacobian), axis=-1)
    return prediction, jacobian


def _normalized_information_diagnostics(
    design: Array, minimum_column_norm: float
) -> tuple[Array, Array]:
    column_norms = jnp.sqrt(jnp.sum(design**2, axis=0))
    normalized = design / jnp.maximum(column_norms, minimum_column_norm)
    information = _symmetric(normalized.T @ normalized)
    eigenvalues = jnp.linalg.eigvalsh(information)
    minimum = jnp.where(jnp.all(column_norms >= minimum_column_norm), eigenvalues[0], 0.0)
    condition = eigenvalues[-1] / jnp.maximum(minimum, jnp.finfo(design.dtype).eps)
    return minimum, condition


def _positive_definite_inverse(matrix: Array, floor: float) -> Array:
    eigenvalues, eigenvectors = jnp.linalg.eigh(_symmetric(matrix))
    inverse_values = 1 / jnp.maximum(eigenvalues, floor)
    return (eigenvectors * inverse_values[None, :]) @ eigenvectors.T


def _vectors_to_parameters(vectors: Array) -> DynamicsParameters:
    inverse_mass = vectors[..., 0]
    mass = 1 / inverse_mass
    drag_acceleration = vectors[..., 1:4]
    drag_force = drag_acceleration / inverse_mass[..., None]
    drag_matrix = -jnp.eye(3, dtype=vectors.dtype) * drag_force[..., None, :]
    return DynamicsParameters(
        mass=mass,
        inverse_mass=inverse_mass,
        drag_acceleration=drag_acceleration,
        drag_force_coefficients=drag_force,
        drag_matrix=drag_matrix,
        wind_velocity=vectors[..., 4:7],
        rotor_efficiency=vectors[..., 7:11],
    )


def _parameter_bounds(config: EstimatorConfig, dtype: jnp.dtype) -> tuple[Array, Array]:
    lower, upper = _host_parameter_bounds(config)
    return jnp.asarray(lower, dtype=dtype), jnp.asarray(upper, dtype=dtype)


def _host_parameter_bounds(config: EstimatorConfig) -> tuple[np.ndarray, np.ndarray]:
    mass_lower, mass_upper = config.mass_bounds
    drag_lower, drag_upper = config.drag_acceleration_bounds
    efficiency_lower, efficiency_upper = config.efficiency_bounds
    lower = np.asarray(
        [
            1 / mass_upper,
            drag_lower,
            drag_lower,
            drag_lower,
            *config.wind_lower,
            efficiency_lower,
            efficiency_lower,
            efficiency_lower,
            efficiency_lower,
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            1 / mass_lower,
            drag_upper,
            drag_upper,
            drag_upper,
            *config.wind_upper,
            efficiency_upper,
            efficiency_upper,
            efficiency_upper,
            efficiency_upper,
        ],
        dtype=np.float64,
    )
    return lower, upper


def _state_is_finite(state: EstimatorState) -> Array:
    return jnp.all(
        jnp.stack(
            [
                jnp.all(jnp.isfinite(state.inverse_mass)),
                jnp.all(jnp.isfinite(state.drag_acceleration)),
                jnp.all(jnp.isfinite(state.wind_velocity)),
                jnp.all(jnp.isfinite(state.rotor_efficiency)),
                jnp.all(jnp.isfinite(state.covariance)),
            ]
        )
    )


def _state_versions_valid(state: EstimatorState) -> Array:
    """Validate logical counters on device so malformed external states fail closed under JIT."""
    return (
        (state.model_version >= 0)
        & (state.last_translational_sequence >= -1)
        & (state.last_rotor_sequence >= -1)
    )


def _is_psd(matrix: Array, tolerance: float) -> Array:
    safe = jnp.nan_to_num(_symmetric(matrix), nan=0.0, posinf=0.0, neginf=0.0)
    symmetry_error = jnp.max(jnp.abs(matrix - matrix.T))
    return (
        jnp.all(jnp.isfinite(matrix))
        & (symmetry_error <= tolerance)
        & (jnp.linalg.eigvalsh(safe)[0] >= -tolerance)
    )


def _has_independent_covariance_blocks(matrix: Array, tolerance: float) -> Array:
    cross_block = matrix[:_TRANSLATIONAL_DIM, _TRANSLATIONAL_DIM:]
    return jnp.max(jnp.abs(cross_block)) <= tolerance


def _symmetric(matrix: Array) -> Array:
    return (matrix + matrix.T) / 2


def _rotation_error(rotation: Array, mask: Array) -> Array:
    identity = jnp.eye(3, dtype=rotation.dtype)
    gram_error = jnp.max(
        jnp.abs(jnp.swapaxes(rotation, -1, -2) @ rotation - identity), axis=(-2, -1)
    )
    determinant_error = jnp.abs(jnp.linalg.det(rotation) - 1)
    error = jnp.maximum(gram_error, determinant_error)
    return jnp.max(jnp.where(mask, error, 0.0))


def _masked(value: Array, mask: Array) -> Array:
    return jnp.where(mask, value, jnp.zeros((), dtype=value.dtype))


def _masked_all_finite(mask: Array, *values: Array) -> Array:
    result = jnp.asarray(True)
    for value in values:
        expanded_mask = mask.reshape(mask.shape + (1,) * (value.ndim - mask.ndim))
        result = result & jnp.all(jnp.where(expanded_mask, jnp.isfinite(value), True))
    return result


def _update_status(
    *, fresh: Array, finite: Array, valid: Array, identifiable: Array, numerical: Array
) -> Array:
    status = jnp.asarray(EstimatorUpdateStatus.ACCEPTED, dtype=jnp.int32)
    status = jnp.where(~numerical, EstimatorUpdateStatus.NUMERICAL_FAILURE, status)
    status = jnp.where(~identifiable, EstimatorUpdateStatus.UNIDENTIFIABLE, status)
    status = jnp.where(~valid, EstimatorUpdateStatus.INVALID_OBSERVATION, status)
    status = jnp.where(~finite, EstimatorUpdateStatus.NONFINITE, status)
    status = jnp.where(~fresh, EstimatorUpdateStatus.STALE_SEQUENCE, status)
    return status


def _check_translational_shapes(observations: TranslationalObservations) -> None:
    rotation = observations.rotation_body_to_world
    if rotation.ndim != 3 or rotation.shape[1:] != (3, 3):
        raise ValueError("rotation_body_to_world must have shape [N, 3, 3]")
    n_observations = rotation.shape[0]
    if n_observations == 0:
        raise ValueError("translational observation window must not be empty")
    for name, value in (
        ("velocity_world", observations.velocity_world),
        ("acceleration_world", observations.acceleration_world),
        ("gravity_world", observations.gravity_world),
    ):
        if value.shape != (n_observations, 3):
            raise ValueError(f"{name} must have shape [N, 3]")
    if observations.collective_thrust.shape != (n_observations,):
        raise ValueError("collective_thrust must have shape [N]")
    if observations.sample_mask.shape != (n_observations,):
        raise ValueError("sample_mask must have shape [N]")
    if observations.sample_mask.dtype != jnp.bool_:
        raise TypeError("sample_mask must have boolean dtype")


def _check_rotor_shapes(observations: RotorEfficiencyObservations) -> None:
    command = observations.commanded_motor_forces
    if command.ndim != 2 or command.shape[1] != 4:
        raise ValueError("commanded_motor_forces must have shape [N, 4]")
    if command.shape[0] == 0:
        raise ValueError("rotor observation window must not be empty")
    if observations.realized_motor_forces.shape != command.shape:
        raise ValueError("realized_motor_forces must match commanded_motor_forces")
    if observations.sample_mask.shape != command.shape:
        raise ValueError("sample_mask must have shape [N, 4]")
    if observations.sample_mask.dtype != jnp.bool_:
        raise TypeError("sample_mask must have boolean dtype")


def _check_estimator_state_shapes(state: EstimatorState) -> None:
    expected = (
        ("inverse_mass", state.inverse_mass, ()),
        ("drag_acceleration", state.drag_acceleration, (3,)),
        ("wind_velocity", state.wind_velocity, (3,)),
        ("rotor_efficiency", state.rotor_efficiency, (4,)),
        ("covariance", state.covariance, (_PARAMETER_DIM, _PARAMETER_DIM)),
        ("model_version", state.model_version, ()),
        ("last_translational_sequence", state.last_translational_sequence, ()),
        ("last_rotor_sequence", state.last_rotor_sequence, ()),
    )
    for name, value, shape in expected:
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    for name, value in (
        ("model_version", state.model_version),
        ("last_translational_sequence", state.last_translational_sequence),
        ("last_rotor_sequence", state.last_rotor_sequence),
    ):
        if not jnp.issubdtype(value.dtype, jnp.integer):
            raise TypeError(f"{name} must have integer dtype")


def _check_sequence(sequence: Array | int) -> None:
    array = jnp.asarray(sequence)
    if array.shape != () or not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError("sequence must be an integer scalar")


def _validate_covariance(covariance: np.ndarray, tolerance: float) -> None:
    if covariance.shape != (_PARAMETER_DIM, _PARAMETER_DIM):
        raise ValueError(f"covariance must have shape ({_PARAMETER_DIM}, {_PARAMETER_DIM})")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be finite")
    if not np.allclose(covariance, covariance.T, atol=tolerance, rtol=0):
        raise ValueError("covariance must be symmetric")
    if np.linalg.eigvalsh(covariance)[0] < -tolerance:
        raise ValueError("covariance must be positive semidefinite")


def _validate_ordered_positive_bounds(bounds: tuple[float, float], name: str) -> None:
    if (
        len(bounds) != 2
        or not all(math.isfinite(value) for value in bounds)
        or bounds[0] <= 0
        or bounds[0] >= bounds[1]
    ):
        raise ValueError(f"{name} must contain finite ordered positive bounds")


def _validate_ordered_nonnegative_bounds(bounds: tuple[float, float], name: str) -> None:
    if (
        len(bounds) != 2
        or not all(math.isfinite(value) for value in bounds)
        or bounds[0] < 0
        or bounds[0] >= bounds[1]
    ):
        raise ValueError(f"{name} must contain finite ordered nonnegative bounds")


def _validate_vector_bounds(
    lower: tuple[float, float, float], upper: tuple[float, float, float], name: str
) -> None:
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError(f"{name} bounds must have length three")
    if not all(math.isfinite(value) for value in (*lower, *upper)):
        raise ValueError(f"{name} bounds must be finite")
    if any(low >= high for low, high in zip(lower, upper, strict=True)):
        raise ValueError(f"{name} lower bounds must be below upper bounds")


def _validate_covariance_diagonal(
    diagonal: tuple[float, ...], name: str, *, strictly_positive: bool
) -> None:
    if len(diagonal) != _PARAMETER_DIM or not all(math.isfinite(value) for value in diagonal):
        raise ValueError(f"{name} must contain {_PARAMETER_DIM} finite values")
    minimum = 0 if not strictly_positive else np.nextafter(0.0, 1.0)
    if any(value < minimum for value in diagonal):
        qualifier = "positive" if strictly_positive else "nonnegative"
        raise ValueError(f"{name} values must be {qualifier}")


# Convenience wrappers make the intended static arguments explicit for callers that want compiled
# fixed-window online updates without repeating ``static_argnames``.
jit_update_translational_estimate = partial(jax.jit, static_argnames=("config",))(
    update_translational_estimate
)
jit_update_rotor_efficiency = partial(jax.jit, static_argnames=("mode", "config"))(
    update_rotor_efficiency
)
jit_deterministic_parameter_samples = partial(jax.jit, static_argnames=("sample_count", "config"))(
    deterministic_parameter_samples
)


__all__ = [
    "DeterministicParameterSamples",
    "DynamicsParameters",
    "EstimatorConfig",
    "EstimatorState",
    "EstimatorUpdate",
    "EstimatorUpdateStatus",
    "RotorEfficiencyObservations",
    "TranslationalObservations",
    "deterministic_parameter_samples",
    "estimation_vector",
    "initialize_estimator",
    "jit_deterministic_parameter_samples",
    "jit_update_rotor_efficiency",
    "jit_update_translational_estimate",
    "physical_parameters",
    "update_rotor_efficiency",
    "update_translational_estimate",
]
