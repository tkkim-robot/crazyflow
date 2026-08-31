"""Deterministic dynamic-obstacle scenarios and immutable scenario tapes.

The generators in this module describe the *environment*, not fallback-controller logic.  Every
attacker selects one documented trajectory model when the tape is created and then follows the
resulting fixed trajectory.  There are no contact impulses, mode transitions, floor clamps, or
post-generation obstacle corrections.

All randomness comes from stable, named JAX PRNG streams.  A scenario tape records the stream IDs,
root seed, and fold so paired methods can consume byte-identical conditions.  The NPZ serializer
uses sorted members and fixed ZIP metadata; the canonical content digest is independent of the
archive container.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import struct
import tempfile
import zipfile
from dataclasses import asdict, dataclass, fields
from enum import IntEnum
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from jax import Array


SCENARIO_TAPE_SCHEMA_VERSION = 3
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024

# Ballistic trial difficulty is assigned by fold, before any controller is evaluated.  Every ten
# consecutive folds contain four reference-collision encounters, four safety-buffer encounters,
# and two wider near misses.  Continuous values within the selected stratum remain seed-random.
BALLISTIC_ENCOUNTER_STRATUM_CYCLE = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2)
BALLISTIC_ENCOUNTER_STRATUM_NAMES = (
    "reference_collision_tube",
    "reference_safety_buffer",
    "reference_near_miss",
)


class DynamicObstacleKind(IntEnum):
    """Physical provenance of a fixed dynamic-obstacle slot."""

    PADDING = 0
    BALLISTIC = 1
    ATTACKER_DRONE = 2


class AttackerMode(IntEnum):
    """Finite attacker trajectory models selected at tape generation time."""

    NOT_ATTACKER = -1
    SCRIPTED_CROSSING = 0
    BOUNDED_PURSUIT = 1
    PREDICTIVE_INTERCEPTOR = 2


RNG_STREAM_NAMES = (
    "static_obstacles",
    "dynamic_radii",
    "ballistic_truth",
    "ballistic_predictions",
    "scripted_crossing",
    "bounded_pursuit",
    "predictive_interceptor",
    "random_attacker_truth",
    "random_attacker_predictions",
    "wind_schedule",
    "mass_schedule",
    "drag_schedule",
    "rotor_schedule",
    "estimator_acceleration_noise",
    "estimator_motor_force_noise",
)

SCHEDULE_NAMES = (
    "wind_step",
    "mass_step",
    "drag_step",
    "rotor_symmetric_step",
    "rotor_single_step",
)


def _stable_stream_id(name: str) -> int:
    """Map a semantic stream name to a stable unsigned 32-bit fold value."""
    return int.from_bytes(
        hashlib.sha256(f"crazyflow.da_plcbf.rng:{name}".encode()).digest()[:4], "little"
    )


RNG_STREAM_IDS: Mapping[str, int] = MappingProxyType(
    {name: _stable_stream_id(name) for name in RNG_STREAM_NAMES}
)


def named_rng_key(seed: int, stream: str, *, fold: int = 0) -> Array:
    """Return a JAX key folded by a stable semantic stream ID and trial fold.

    Args:
        seed: Unsigned 32-bit root seed.
        stream: One of :data:`RNG_STREAM_NAMES`.
        fold: Unsigned 32-bit paired-trial or scenario index.

    Returns:
        A legacy ``uint32[2]`` JAX PRNG key.  Using a legacy key makes the recorded seed/fold
        representation independent of JAX's typed-key implementation details.
    """
    root_seed = _validate_uint32(seed, "seed")
    fold_value = _validate_uint32(fold, "fold")
    try:
        stream_id = RNG_STREAM_IDS[stream]
    except KeyError as error:
        raise ValueError(f"unknown RNG stream {stream!r}") from error
    return jr.fold_in(jr.fold_in(jr.PRNGKey(root_seed), stream_id), fold_value)


@dataclass(frozen=True, slots=True)
class ScenarioTapeConfig:
    """Configuration for a fixed-shape DA-PLCBF scenario tape."""

    steps: int = 151
    dt: float = 0.02
    prediction_samples: int = 8
    static_capacity: int = 8
    static_count: int = 4
    dynamic_capacity: int = 10
    ballistic_count: int = 2
    crossing_count: int = 1
    pursuit_count: int = 1
    interceptor_count: int = 1
    random_attacker_count: int = 3
    arena_lower: tuple[float, float, float] = (-5.0, -5.0, 0.0)
    arena_upper: tuple[float, float, float] = (5.0, 5.0, 5.0)
    vehicle_initial_position: tuple[float, float, float] = (0.0, 0.0, 1.5)
    vehicle_initial_velocity: tuple[float, float, float] = (0.45, 0.10, 0.0)
    reference_initial_position: tuple[float, float, float] | None = None
    reference_initial_velocity: tuple[float, float, float] | None = None
    vehicle_radius: float = 0.12
    static_radius_range: tuple[float, float] = (0.25, 0.65)
    dynamic_radius_range: tuple[float, float] = (0.10, 0.20)
    crossing_fraction_range: tuple[float, float] = (0.35, 0.65)
    ball_release_fraction_range: tuple[float, float] = (0.05, 0.35)
    ball_velocity_lower: tuple[float, float, float] = (-2.5, -2.5, -0.5)
    ball_velocity_upper: tuple[float, float, float] = (2.5, 2.5, 2.0)
    ball_velocity_uncertainty: tuple[float, float, float] = (0.45, 0.45, 0.30)
    ballistic_safety_buffer: float = 0.03
    ballistic_near_miss_extra: float = 0.15
    ballistic_encounter_band_fraction_range: tuple[float, float] = (0.15, 0.85)
    ballistic_time_to_impact_cap: float = 1.0
    ballistic_time_to_impact_fraction_bins: tuple[tuple[float, float], ...] = (
        (0.25, 0.45),
        (0.45, 0.70),
        (0.70, 0.95),
    )
    ballistic_generation_max_attempts: int = 128
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    attacker_speed_range: tuple[float, float] = (0.8, 2.2)
    attacker_acceleration_limit: float = 3.5
    attacker_initial_speed_fraction: float = 0.20
    interceptor_prediction_horizon: float = 1.0
    wind_speed_limit: float = 3.0
    wind_gust_amplitude: float = 0.75
    wind_frequency_range: tuple[float, float] = (0.2, 0.8)
    mass_scale_bounds: tuple[float, float] = (1.0, 1.45)
    drag_scale_bounds: tuple[float, float] = (0.65, 1.45)
    rotor_efficiency_bounds: tuple[float, float] = (0.60, 1.0)
    rotor_single_efficiency_lower: float | None = None
    wind_change_fraction: float = 0.20
    mass_change_fraction: float = 0.35
    drag_change_fraction: float = 0.50
    rotor_symmetric_change_fraction: float = 0.65
    rotor_single_change_fraction: float = 0.80
    estimator_acceleration_noise_std: float = 0.03
    estimator_motor_force_noise_std: float = 5e-4

    def validate(self) -> None:
        """Validate shapes, finiteness, capacities, and physical parameter bounds."""
        _validate_positive_integer(self.steps, "steps", minimum=2)
        _validate_finite_positive(self.dt, "dt")
        _validate_positive_integer(self.prediction_samples, "prediction_samples")
        _validate_positive_integer(self.static_capacity, "static_capacity")
        _validate_nonnegative_integer(self.static_count, "static_count")
        _validate_positive_integer(self.dynamic_capacity, "dynamic_capacity")
        for name in (
            "ballistic_count",
            "crossing_count",
            "pursuit_count",
            "interceptor_count",
            "random_attacker_count",
        ):
            _validate_nonnegative_integer(getattr(self, name), name)
        if self.static_count > self.static_capacity:
            raise ValueError("static_count must not exceed static_capacity")
        required_dynamic = (
            self.ballistic_count
            + self.crossing_count
            + self.pursuit_count
            + self.interceptor_count
            + self.random_attacker_count
        )
        if required_dynamic > self.dynamic_capacity:
            raise ValueError("configured dynamic obstacles exceed dynamic_capacity")

        lower = _finite_vector(self.arena_lower, "arena_lower")
        upper = _finite_vector(self.arena_upper, "arena_upper")
        if not np.all(upper > lower):
            raise ValueError("every arena upper bound must exceed its lower bound")
        initial_position = _finite_vector(self.vehicle_initial_position, "vehicle_initial_position")
        _finite_vector(self.vehicle_initial_velocity, "vehicle_initial_velocity")
        if not np.all((initial_position >= lower) & (initial_position <= upper)):
            raise ValueError("vehicle_initial_position must lie inside the arena")
        if self.reference_initial_position is not None:
            reference_position = _finite_vector(
                self.reference_initial_position, "reference_initial_position"
            )
            if not np.all((reference_position >= lower) & (reference_position <= upper)):
                raise ValueError("reference_initial_position must lie inside the arena")
        if self.reference_initial_velocity is not None:
            _finite_vector(self.reference_initial_velocity, "reference_initial_velocity")
        _validate_finite_positive(self.vehicle_radius, "vehicle_radius")

        static_radius = _finite_interval(
            self.static_radius_range, "static_radius_range", positive=True
        )
        dynamic_radius = _finite_interval(
            self.dynamic_radius_range, "dynamic_radius_range", positive=True
        )
        if np.any(2.0 * static_radius[1] >= upper - lower):
            raise ValueError("largest static obstacle must fit strictly inside every arena axis")
        if np.any(2.0 * dynamic_radius[1] >= upper - lower):
            raise ValueError("largest dynamic obstacle must fit strictly inside every arena axis")

        crossing_range = np.asarray(self.crossing_fraction_range, dtype=np.float64)
        if (
            crossing_range.shape != (2,)
            or not np.all(np.isfinite(crossing_range))
            or crossing_range[0] <= 0.0
            or crossing_range[1] >= 1.0
            or crossing_range[1] < crossing_range[0]
        ):
            raise ValueError(
                "crossing_fraction_range must be finite, ordered, and strictly inside (0, 1)"
            )

        release_range = _finite_interval(
            self.ball_release_fraction_range, "ball_release_fraction_range"
        )
        if release_range[0] < 0.0 or release_range[1] >= 1.0:
            raise ValueError("ball release fractions must lie in [0, 1)")
        velocity_lower = _finite_vector(self.ball_velocity_lower, "ball_velocity_lower")
        velocity_upper = _finite_vector(self.ball_velocity_upper, "ball_velocity_upper")
        uncertainty = _finite_vector(self.ball_velocity_uncertainty, "ball_velocity_uncertainty")
        if not np.all(velocity_upper > velocity_lower):
            raise ValueError("every ball velocity upper bound must exceed its lower bound")
        if np.any(uncertainty < 0.0):
            raise ValueError("ball velocity uncertainty must be nonnegative")
        if np.any(2.0 * uncertainty > velocity_upper - velocity_lower):
            raise ValueError("ball velocity uncertainty leaves no valid truth-velocity interior")
        _validate_finite_positive(self.ballistic_safety_buffer, "ballistic_safety_buffer")
        _validate_finite_positive(self.ballistic_near_miss_extra, "ballistic_near_miss_extra")
        band_fraction = _finite_interval(
            self.ballistic_encounter_band_fraction_range, "ballistic_encounter_band_fraction_range"
        )
        if band_fraction[0] <= 0.0 or band_fraction[1] >= 1.0:
            raise ValueError("ballistic encounter band fractions must lie strictly inside (0, 1)")
        _validate_finite_positive(self.ballistic_time_to_impact_cap, "ballistic_time_to_impact_cap")
        impact_bins = np.asarray(self.ballistic_time_to_impact_fraction_bins, dtype=np.float64)
        if impact_bins.shape != (3, 2) or not np.all(np.isfinite(impact_bins)):
            raise ValueError("ballistic time-to-impact bins must contain three finite intervals")
        if (
            np.any(impact_bins[:, 0] <= 0.0)
            or np.any(impact_bins[:, 1] > 1.0)
            or np.any(impact_bins[:, 1] <= impact_bins[:, 0])
            or np.any(impact_bins[1:, 0] < impact_bins[:-1, 1])
        ):
            raise ValueError(
                "ballistic time-to-impact bins must be positive, ordered, disjoint, and <=1"
            )
        _validate_positive_integer(
            self.ballistic_generation_max_attempts, "ballistic_generation_max_attempts"
        )
        gravity = _finite_vector(self.gravity, "gravity")
        if gravity[2] >= 0.0 or np.linalg.norm(gravity) == 0.0:
            raise ValueError("gravity must be nonzero with a negative vertical component")

        _finite_interval(self.attacker_speed_range, "attacker_speed_range", positive=True)
        _validate_finite_positive(self.attacker_acceleration_limit, "attacker_acceleration_limit")
        initial_speed_fraction = _finite_scalar(
            self.attacker_initial_speed_fraction, "attacker_initial_speed_fraction"
        )
        if not 0.0 <= initial_speed_fraction <= 1.0:
            raise ValueError("attacker_initial_speed_fraction must lie in [0, 1]")
        _validate_finite_positive(
            self.interceptor_prediction_horizon, "interceptor_prediction_horizon"
        )

        _validate_finite_positive(self.wind_speed_limit, "wind_speed_limit")
        gust = _finite_scalar(self.wind_gust_amplitude, "wind_gust_amplitude")
        if gust < 0.0 or gust > 0.4 * self.wind_speed_limit:
            raise ValueError("wind_gust_amplitude must lie in [0, 0.4 * wind_speed_limit]")
        _finite_interval(self.wind_frequency_range, "wind_frequency_range", positive=True)
        _validate_scale_bounds(self.mass_scale_bounds, "mass_scale_bounds")
        _validate_scale_bounds(self.drag_scale_bounds, "drag_scale_bounds")
        rotor_bounds = _validate_scale_bounds(
            self.rotor_efficiency_bounds, "rotor_efficiency_bounds"
        )
        if rotor_bounds[1] > 1.0:
            raise ValueError("rotor efficiency upper bound must not exceed 1")
        single_rotor_lower = (
            rotor_bounds[0]
            if self.rotor_single_efficiency_lower is None
            else _finite_scalar(self.rotor_single_efficiency_lower, "rotor_single_efficiency_lower")
        )
        if single_rotor_lower <= 0.0 or single_rotor_lower > rotor_bounds[0]:
            raise ValueError(
                "rotor_single_efficiency_lower must be positive and no greater than the "
                "symmetric rotor lower bound"
            )

        fractions = (
            self.wind_change_fraction,
            self.mass_change_fraction,
            self.drag_change_fraction,
            self.rotor_symmetric_change_fraction,
            self.rotor_single_change_fraction,
        )
        fraction_values = np.asarray(
            [
                _finite_scalar(value, name)
                for value, name in zip(fractions, SCHEDULE_NAMES, strict=True)
            ]
        )
        if np.any((fraction_values < 0.0) | (fraction_values > 1.0)):
            raise ValueError("schedule change fractions must lie in [0, 1]")
        if np.any(np.diff(fraction_values) < 0.0):
            raise ValueError("schedule change fractions must be nondecreasing")
        for name in ("estimator_acceleration_noise_std", "estimator_motor_force_noise_std"):
            value = _finite_scalar(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True, slots=True, eq=False)
class ScenarioTape:
    """Content-addressed, immutable fixed-shape scenario data.

    Dynamic-obstacle arrays have a padded slot dimension ``M``.  Prediction arrays add a leading
    finite-scenario dimension ``R``.  ``dynamic_time_mask`` is shared by truth and predictions;
    ballistic slots become valid at their release node and attacker drones are valid throughout.
    """

    schema_version: np.ndarray
    root_seed: np.ndarray
    generation_fold: np.ndarray
    generator_config_sha256: np.ndarray
    rng_stream_names: np.ndarray
    rng_stream_ids: np.ndarray
    time: np.ndarray
    arena_lower: np.ndarray
    arena_upper: np.ndarray
    vehicle_initial_position: np.ndarray
    vehicle_initial_velocity: np.ndarray
    vehicle_radius: np.ndarray
    defender_reference_position: np.ndarray
    defender_reference_velocity: np.ndarray
    static_positions: np.ndarray
    static_radii: np.ndarray
    static_mask: np.ndarray
    dynamic_positions: np.ndarray
    dynamic_velocities: np.ndarray
    dynamic_radii: np.ndarray
    dynamic_slot_mask: np.ndarray
    dynamic_time_mask: np.ndarray
    dynamic_kind: np.ndarray
    attacker_mode: np.ndarray
    randomized_attacker: np.ndarray
    prediction_positions: np.ndarray
    prediction_velocities: np.ndarray
    prediction_attacker_mode: np.ndarray
    ballistic_release_index: np.ndarray
    ballistic_release_position: np.ndarray
    ballistic_release_velocity: np.ndarray
    ballistic_prediction_release_velocity: np.ndarray
    ballistic_velocity_lower: np.ndarray
    ballistic_velocity_upper: np.ndarray
    ballistic_velocity_uncertainty: np.ndarray
    ballistic_release_fraction_range: np.ndarray
    ballistic_safety_buffer: np.ndarray
    ballistic_near_miss_extra: np.ndarray
    ballistic_time_to_impact_cap: np.ndarray
    ballistic_time_to_impact_fraction_bins: np.ndarray
    ballistic_generation_max_attempts: np.ndarray
    ballistic_encounter_stratum: np.ndarray
    ballistic_time_to_impact_bin: np.ndarray
    ballistic_target_time: np.ndarray
    ballistic_time_to_impact: np.ndarray
    ballistic_intended_miss_vector: np.ndarray
    ballistic_intended_miss_distance: np.ndarray
    ballistic_realized_closest_time: np.ndarray
    ballistic_realized_closest_distance: np.ndarray
    ballistic_generation_attempts: np.ndarray
    gravity: np.ndarray
    dynamic_speed_limit: np.ndarray
    dynamic_acceleration_limit: np.ndarray
    interceptor_prediction_horizon: np.ndarray
    wind_velocity: np.ndarray
    wind_speed_limit: np.ndarray
    mass_scale: np.ndarray
    mass_scale_bounds: np.ndarray
    drag_scale: np.ndarray
    drag_scale_bounds: np.ndarray
    rotor_efficiency: np.ndarray
    rotor_efficiency_bounds: np.ndarray
    rotor_single_index: np.ndarray
    schedule_names: np.ndarray
    schedule_change_indices: np.ndarray
    estimator_acceleration_noise: np.ndarray
    estimator_acceleration_noise_std: np.ndarray
    estimator_motor_force_noise: np.ndarray
    estimator_motor_force_noise_std: np.ndarray
    contact_is_failure: np.ndarray

    def __post_init__(self) -> None:
        """Defensively copy every array, make it read-only, and validate the complete tape."""
        for item in fields(self):
            object.__setattr__(self, item.name, _frozen_array(getattr(self, item.name)))
        self.validate()

    @property
    def sha256(self) -> str:
        """Return the canonical SHA-256 digest of all tape fields."""
        return _canonical_tape_digest(self)

    @property
    def steps(self) -> int:
        """Return the number of fixed time nodes."""
        return int(self.time.shape[0])

    @property
    def prediction_samples(self) -> int:
        """Return the number of finite predicted trajectory scenarios."""
        return int(self.prediction_positions.shape[0])

    def validate(self) -> None:
        """Validate schema, shapes, finiteness, masks, dynamics, and physical bounds."""
        _require_scalar(self.schema_version, np.uint16, "schema_version")
        if int(self.schema_version) != SCENARIO_TAPE_SCHEMA_VERSION:
            raise ValueError("unsupported scenario-tape schema version")
        _require_scalar(self.root_seed, np.uint32, "root_seed")
        _require_scalar(self.generation_fold, np.uint32, "generation_fold")
        if (
            self.generator_config_sha256.shape != ()
            or self.generator_config_sha256.dtype.kind != "U"
        ):
            raise ValueError("generator_config_sha256 must be a scalar Unicode string")
        if len(str(self.generator_config_sha256)) != 64:
            raise ValueError("generator_config_sha256 must contain a SHA-256 hex digest")
        try:
            bytes.fromhex(str(self.generator_config_sha256))
        except ValueError as error:
            raise ValueError("generator_config_sha256 is not hexadecimal") from error

        expected_names = np.asarray(RNG_STREAM_NAMES)
        expected_ids = np.asarray(
            [RNG_STREAM_IDS[name] for name in RNG_STREAM_NAMES], dtype=np.uint32
        )
        if self.rng_stream_names.dtype.kind != "U" or not np.array_equal(
            self.rng_stream_names, expected_names
        ):
            raise ValueError("rng_stream_names do not match the schema")
        _require_dtype(self.rng_stream_ids, np.uint32, "rng_stream_ids")
        if not np.array_equal(self.rng_stream_ids, expected_ids):
            raise ValueError("rng_stream_ids do not match their stable names")

        _require_dtype(self.time, np.float64, "time")
        if self.time.ndim != 1 or self.time.size < 2 or not np.all(np.isfinite(self.time)):
            raise ValueError("time must be a finite one-dimensional array with at least two nodes")
        dt = np.diff(self.time)
        if (
            self.time[0] != 0.0
            or not np.all(dt > 0.0)
            or not np.allclose(dt, dt[0], rtol=0.0, atol=1e-14)
        ):
            raise ValueError("time must start at zero and use a strictly positive uniform grid")
        steps = self.time.size
        _require_float_shape(self.arena_lower, (3,), "arena_lower")
        _require_float_shape(self.arena_upper, (3,), "arena_upper")
        if not np.all(self.arena_upper > self.arena_lower):
            raise ValueError("arena bounds must be strictly ordered")
        _require_float_shape(self.vehicle_initial_position, (3,), "vehicle_initial_position")
        _require_float_shape(self.vehicle_initial_velocity, (3,), "vehicle_initial_velocity")
        _require_float_shape(self.vehicle_radius, (), "vehicle_radius")
        if float(self.vehicle_radius) <= 0.0:
            raise ValueError("vehicle_radius must be positive")
        if not np.all(
            (self.vehicle_initial_position >= self.arena_lower)
            & (self.vehicle_initial_position <= self.arena_upper)
        ):
            raise ValueError("vehicle_initial_position must lie inside the arena")
        _require_float_shape(
            self.defender_reference_position, (steps, 3), "defender_reference_position"
        )
        _require_float_shape(
            self.defender_reference_velocity, (steps, 3), "defender_reference_velocity"
        )
        expected_reference = (
            self.defender_reference_position[0][None, :]
            + self.time[:, None] * self.defender_reference_velocity[0][None, :]
        )
        if not np.allclose(
            self.defender_reference_position, expected_reference, rtol=1e-12, atol=1e-12
        ) or not np.array_equal(
            self.defender_reference_velocity,
            np.broadcast_to(self.defender_reference_velocity[0], (steps, 3)),
        ):
            raise ValueError("defender reference must be a constant-velocity trajectory")

        self._validate_static_obstacles()
        self._validate_dynamic_obstacles(float(dt[0]))
        self._validate_schedules()
        self._validate_estimator_observation_noise()

    def _validate_estimator_observation_noise(self) -> None:
        """Validate and reconstruct the two predeclared estimator-noise streams."""
        steps = self.time.size
        _require_float_shape(
            self.estimator_acceleration_noise, (steps, 3), "estimator_acceleration_noise"
        )
        _require_float_shape(
            self.estimator_motor_force_noise, (steps, 4), "estimator_motor_force_noise"
        )
        _require_float_shape(
            self.estimator_acceleration_noise_std, (), "estimator_acceleration_noise_std"
        )
        _require_float_shape(
            self.estimator_motor_force_noise_std, (), "estimator_motor_force_noise_std"
        )
        if (
            float(self.estimator_acceleration_noise_std) < 0.0
            or float(self.estimator_motor_force_noise_std) < 0.0
        ):
            raise ValueError("estimator observation-noise standard deviations must be nonnegative")
        expected = _generate_estimator_observation_noise(
            int(self.root_seed),
            int(self.generation_fold),
            steps,
            acceleration_std=float(self.estimator_acceleration_noise_std),
            motor_force_std=float(self.estimator_motor_force_noise_std),
        )
        if not np.array_equal(
            self.estimator_acceleration_noise, expected["estimator_acceleration_noise"]
        ) or not np.array_equal(
            self.estimator_motor_force_noise, expected["estimator_motor_force_noise"]
        ):
            raise ValueError(
                "estimator observation-noise sequences do not match their named RNG streams"
            )

    def _validate_static_obstacles(self) -> None:
        if self.static_positions.ndim != 2 or self.static_positions.shape[1:] != (3,):
            raise ValueError("static_positions must have shape (O, 3)")
        capacity = self.static_positions.shape[0]
        if capacity < 1:
            raise ValueError("static obstacle capacity must be positive")
        _require_float_shape(self.static_positions, (capacity, 3), "static_positions")
        _require_float_shape(self.static_radii, (capacity,), "static_radii")
        _require_bool_shape(self.static_mask, (capacity,), "static_mask")
        if not _is_prefix_mask(self.static_mask):
            raise ValueError("static_mask must mark a contiguous prefix")
        if np.any(self.static_radii[self.static_mask] <= 0.0):
            raise ValueError("real static-obstacle radii must be positive")
        real_positions = self.static_positions[self.static_mask]
        real_radii = self.static_radii[self.static_mask, None]
        if np.any(real_positions - real_radii < self.arena_lower) or np.any(
            real_positions + real_radii > self.arena_upper
        ):
            raise ValueError("static obstacles must fit inside the arena")
        if np.any(self.static_positions[~self.static_mask] != 0.0) or np.any(
            self.static_radii[~self.static_mask] != 0.0
        ):
            raise ValueError("padded static-obstacle fields must be zero")

    def _validate_dynamic_obstacles(self, dt: float) -> None:
        if self.dynamic_positions.ndim != 3 or self.dynamic_positions.shape[2:] != (3,):
            raise ValueError("dynamic_positions must have shape (T, M, 3)")
        steps, capacity, dimension = self.dynamic_positions.shape
        if dimension != 3 or steps != self.time.size or capacity < 1:
            raise ValueError("dynamic_positions must match the time grid with positive capacity")
        _require_float_shape(self.dynamic_velocities, (steps, capacity, 3), "dynamic_velocities")
        _require_float_shape(self.dynamic_radii, (capacity,), "dynamic_radii")
        _require_bool_shape(self.dynamic_slot_mask, (capacity,), "dynamic_slot_mask")
        _require_bool_shape(self.dynamic_time_mask, (steps, capacity), "dynamic_time_mask")
        _require_dtype_shape(self.dynamic_kind, np.int8, (capacity,), "dynamic_kind")
        _require_dtype_shape(self.attacker_mode, np.int8, (capacity,), "attacker_mode")
        _require_bool_shape(self.randomized_attacker, (capacity,), "randomized_attacker")
        if not _is_prefix_mask(self.dynamic_slot_mask):
            raise ValueError("dynamic_slot_mask must mark a contiguous prefix")
        if np.any(self.dynamic_radii[self.dynamic_slot_mask] <= 0.0):
            raise ValueError("real dynamic-obstacle radii must be positive")

        if self.prediction_positions.ndim != 4:
            raise ValueError("prediction_positions must have shape (R, T, M, 3)")
        samples = self.prediction_positions.shape[0]
        if samples < 1:
            raise ValueError("at least one prediction scenario is required")
        _require_float_shape(
            self.prediction_positions, (samples, steps, capacity, 3), "prediction_positions"
        )
        _require_float_shape(
            self.prediction_velocities, (samples, steps, capacity, 3), "prediction_velocities"
        )
        _require_dtype_shape(
            self.prediction_attacker_mode, np.int8, (samples, capacity), "prediction_attacker_mode"
        )
        _require_dtype_shape(
            self.ballistic_release_index, np.int32, (capacity,), "ballistic_release_index"
        )
        _require_float_shape(
            self.ballistic_release_position, (capacity, 3), "ballistic_release_position"
        )
        _require_float_shape(
            self.ballistic_release_velocity, (capacity, 3), "ballistic_release_velocity"
        )
        _require_float_shape(
            self.ballistic_prediction_release_velocity,
            (samples, capacity, 3),
            "ballistic_prediction_release_velocity",
        )
        _require_float_shape(self.ballistic_velocity_lower, (3,), "ballistic_velocity_lower")
        _require_float_shape(self.ballistic_velocity_upper, (3,), "ballistic_velocity_upper")
        _require_float_shape(
            self.ballistic_velocity_uncertainty, (3,), "ballistic_velocity_uncertainty"
        )
        if not np.all(self.ballistic_velocity_upper > self.ballistic_velocity_lower):
            raise ValueError("ballistic velocity bounds must be strictly ordered")
        if np.any(self.ballistic_velocity_uncertainty < 0.0) or np.any(
            2.0 * self.ballistic_velocity_uncertainty
            > self.ballistic_velocity_upper - self.ballistic_velocity_lower
        ):
            raise ValueError("ballistic velocity uncertainty is outside its declared bounds")
        _require_float_shape(
            self.ballistic_release_fraction_range, (2,), "ballistic_release_fraction_range"
        )
        if (
            self.ballistic_release_fraction_range[0] < 0.0
            or self.ballistic_release_fraction_range[1] <= self.ballistic_release_fraction_range[0]
            or self.ballistic_release_fraction_range[1] >= 1.0
        ):
            raise ValueError("ballistic release fractions must be ordered inside [0, 1)")
        _require_float_shape(self.ballistic_safety_buffer, (), "ballistic_safety_buffer")
        _require_float_shape(self.ballistic_near_miss_extra, (), "ballistic_near_miss_extra")
        _require_float_shape(self.ballistic_time_to_impact_cap, (), "ballistic_time_to_impact_cap")
        if (
            float(self.ballistic_safety_buffer) <= 0.0
            or float(self.ballistic_near_miss_extra) <= 0.0
            or float(self.ballistic_time_to_impact_cap) <= 0.0
        ):
            raise ValueError("ballistic encounter distances and time cap must be positive")
        _require_float_shape(
            self.ballistic_time_to_impact_fraction_bins,
            (3, 2),
            "ballistic_time_to_impact_fraction_bins",
        )
        impact_bins = self.ballistic_time_to_impact_fraction_bins
        if (
            np.any(impact_bins[:, 0] <= 0.0)
            or np.any(impact_bins[:, 1] > 1.0)
            or np.any(impact_bins[:, 1] <= impact_bins[:, 0])
            or np.any(impact_bins[1:, 0] < impact_bins[:-1, 1])
        ):
            raise ValueError("ballistic time-to-impact bins are invalid")
        _require_scalar(
            self.ballistic_generation_max_attempts, np.int32, "ballistic_generation_max_attempts"
        )
        if int(self.ballistic_generation_max_attempts) <= 0:
            raise ValueError("ballistic_generation_max_attempts must be positive")
        _require_dtype_shape(
            self.ballistic_encounter_stratum, np.int8, (capacity,), "ballistic_encounter_stratum"
        )
        _require_dtype_shape(
            self.ballistic_time_to_impact_bin, np.int8, (capacity,), "ballistic_time_to_impact_bin"
        )
        _require_float_shape(self.ballistic_target_time, (capacity,), "ballistic_target_time")
        _require_float_shape(self.ballistic_time_to_impact, (capacity,), "ballistic_time_to_impact")
        _require_float_shape(
            self.ballistic_intended_miss_vector, (capacity, 3), "ballistic_intended_miss_vector"
        )
        _require_float_shape(
            self.ballistic_intended_miss_distance, (capacity,), "ballistic_intended_miss_distance"
        )
        _require_float_shape(
            self.ballistic_realized_closest_time, (capacity,), "ballistic_realized_closest_time"
        )
        _require_float_shape(
            self.ballistic_realized_closest_distance,
            (capacity,),
            "ballistic_realized_closest_distance",
        )
        _require_dtype_shape(
            self.ballistic_generation_attempts,
            np.int32,
            (capacity,),
            "ballistic_generation_attempts",
        )
        _require_float_shape(self.gravity, (3,), "gravity")
        if self.gravity[2] >= 0.0 or np.linalg.norm(self.gravity) == 0.0:
            raise ValueError("gravity must be nonzero with a negative vertical component")
        _require_float_shape(self.dynamic_speed_limit, (capacity,), "dynamic_speed_limit")
        _require_float_shape(
            self.dynamic_acceleration_limit, (capacity,), "dynamic_acceleration_limit"
        )
        _require_float_shape(
            self.interceptor_prediction_horizon, (), "interceptor_prediction_horizon"
        )
        if float(self.interceptor_prediction_horizon) <= 0.0:
            raise ValueError("interceptor_prediction_horizon must be positive")

        real = self.dynamic_slot_mask
        padding = ~real
        valid_kinds = {int(DynamicObstacleKind.BALLISTIC), int(DynamicObstacleKind.ATTACKER_DRONE)}
        if any(int(value) not in valid_kinds for value in self.dynamic_kind[real]):
            raise ValueError("real dynamic slots have an unknown kind")
        if np.any(self.dynamic_kind[padding] != int(DynamicObstacleKind.PADDING)):
            raise ValueError("padded dynamic slots must use PADDING kind")
        if np.any(self.dynamic_radii[padding] != 0.0):
            raise ValueError("padded dynamic radii must be zero")
        if np.any(self.dynamic_time_mask[:, padding]):
            raise ValueError("padded dynamic slots cannot be active")
        if np.any(self.dynamic_positions[:, padding] != 0.0) or np.any(
            self.dynamic_velocities[:, padding] != 0.0
        ):
            raise ValueError("padded dynamic truth trajectories must be zero")
        if np.any(self.prediction_positions[:, :, padding] != 0.0) or np.any(
            self.prediction_velocities[:, :, padding] != 0.0
        ):
            raise ValueError("padded dynamic prediction trajectories must be zero")
        if np.any(self.dynamic_speed_limit[real] <= 0.0) or np.any(
            self.dynamic_acceleration_limit[real] <= 0.0
        ):
            raise ValueError("real dynamic slots require positive speed and acceleration bounds")
        if np.any(self.dynamic_speed_limit[padding] != 0.0) or np.any(
            self.dynamic_acceleration_limit[padding] != 0.0
        ):
            raise ValueError("padded dynamic motion bounds must be zero")

        ballistic = self.dynamic_kind == int(DynamicObstacleKind.BALLISTIC)
        attackers = self.dynamic_kind == int(DynamicObstacleKind.ATTACKER_DRONE)
        if np.any(self.attacker_mode[ballistic | padding] != int(AttackerMode.NOT_ATTACKER)):
            raise ValueError("only attacker-drone slots may have attacker modes")
        valid_modes = {
            int(AttackerMode.SCRIPTED_CROSSING),
            int(AttackerMode.BOUNDED_PURSUIT),
            int(AttackerMode.PREDICTIVE_INTERCEPTOR),
        }
        if any(int(value) not in valid_modes for value in self.attacker_mode[attackers]):
            raise ValueError("attacker slot has an unknown trajectory mode")
        if np.any(self.randomized_attacker & ~attackers):
            raise ValueError("only attacker-drone slots may be randomized attackers")
        if np.any(self.ballistic_release_index[~ballistic] != -1):
            raise ValueError("only ballistic slots may have release indices")
        if np.any(self.ballistic_release_index[ballistic] < 0) or np.any(
            self.ballistic_release_index[ballistic] >= steps
        ):
            raise ValueError("ballistic release index lies outside the time grid")
        if (
            np.any(self.ballistic_release_position[~ballistic] != 0.0)
            or np.any(self.ballistic_release_velocity[~ballistic] != 0.0)
            or np.any(self.ballistic_prediction_release_velocity[:, ~ballistic] != 0.0)
        ):
            raise ValueError("non-ballistic release fields must be zero")
        if (
            np.any(self.ballistic_encounter_stratum[~ballistic] != -1)
            or np.any(self.ballistic_time_to_impact_bin[~ballistic] != -1)
            or np.any(self.ballistic_target_time[~ballistic] != 0.0)
            or np.any(self.ballistic_time_to_impact[~ballistic] != 0.0)
            or np.any(self.ballistic_intended_miss_vector[~ballistic] != 0.0)
            or np.any(self.ballistic_intended_miss_distance[~ballistic] != 0.0)
            or np.any(self.ballistic_realized_closest_time[~ballistic] != 0.0)
            or np.any(self.ballistic_realized_closest_distance[~ballistic] != 0.0)
            or np.any(self.ballistic_generation_attempts[~ballistic] != 0)
        ):
            raise ValueError("non-ballistic encounter metadata must use zero/-1 sentinels")
        ballistic_slots = np.flatnonzero(ballistic)
        expected_stratum = BALLISTIC_ENCOUNTER_STRATUM_CYCLE[
            int(self.generation_fold) % len(BALLISTIC_ENCOUNTER_STRATUM_CYCLE)
        ]
        if np.any(self.ballistic_encounter_stratum[ballistic] != expected_stratum):
            raise ValueError(
                "ballistic encounter stratum does not match the predeclared fold cycle"
            )
        expected_bins = (
            int(self.generation_fold) + np.arange(ballistic_slots.size)
        ) % self.ballistic_time_to_impact_fraction_bins.shape[0]
        if not np.array_equal(
            self.ballistic_time_to_impact_bin[ballistic], expected_bins.astype(np.int8)
        ):
            raise ValueError(
                "ballistic time-to-impact bins do not match the predeclared fold cycle"
            )
        if np.any(self.ballistic_generation_attempts[ballistic] < 1) or np.any(
            self.ballistic_generation_attempts[ballistic]
            > int(self.ballistic_generation_max_attempts)
        ):
            raise ValueError("ballistic generation attempts exceed the declared bounded search")
        ballistic_position = self.ballistic_release_position[ballistic]
        ballistic_radius = self.dynamic_radii[ballistic, None]
        if np.any(ballistic_position - ballistic_radius < self.arena_lower) or np.any(
            ballistic_position + ballistic_radius > self.arena_upper
        ):
            raise ValueError("ballistic release spheres must fit inside the arena")
        if np.any(
            self.ballistic_release_velocity[ballistic]
            < self.ballistic_velocity_lower + self.ballistic_velocity_uncertainty
        ) or np.any(
            self.ballistic_release_velocity[ballistic]
            > self.ballistic_velocity_upper - self.ballistic_velocity_uncertainty
        ):
            raise ValueError("ballistic truth release velocity leaves the uncertainty interior")
        ballistic_predictions = self.ballistic_prediction_release_velocity[:, ballistic]
        if (
            np.any(ballistic_predictions < self.ballistic_velocity_lower)
            or np.any(ballistic_predictions > self.ballistic_velocity_upper)
            or np.any(
                np.abs(ballistic_predictions - self.ballistic_release_velocity[ballistic])
                > self.ballistic_velocity_uncertainty + 1e-12
            )
        ):
            raise ValueError("ballistic prediction release velocity exceeds declared uncertainty")
        expected_offsets = _fixed_ballistic_prediction_offsets(
            samples, self.ballistic_velocity_uncertainty
        )
        actual_offsets = (
            self.ballistic_prediction_release_velocity[:, ballistic]
            - self.ballistic_release_velocity[ballistic][None]
        )
        if not np.allclose(actual_offsets, expected_offsets[:, None, :], rtol=0.0, atol=2e-15):
            raise ValueError("ballistic prediction offsets are not the fixed declared support")

        expected_modes = np.broadcast_to(self.attacker_mode, (samples, capacity)).copy()
        random_mode_values = self.prediction_attacker_mode[:, self.randomized_attacker]
        if any(int(value) not in valid_modes for value in random_mode_values.ravel()):
            raise ValueError("random attacker prediction has an unknown finite mode")
        expected_modes[:, self.randomized_attacker] = random_mode_values
        if not np.array_equal(self.prediction_attacker_mode, expected_modes):
            raise ValueError("non-random prediction modes must match the truth mode")

        for slot in np.flatnonzero(ballistic):
            release = int(self.ballistic_release_index[slot])
            expected_mask = np.arange(steps) >= release
            if not np.array_equal(self.dynamic_time_mask[:, slot], expected_mask):
                raise ValueError("ballistic time mask must begin exactly at release")
            minimum_release = int(
                math.floor(self.ballistic_release_fraction_range[0] * (steps - 1))
            )
            maximum_release = int(
                math.floor(self.ballistic_release_fraction_range[1] * (steps - 1))
            )
            if release < minimum_release or release > maximum_release:
                raise ValueError("ballistic release index exceeds its declared fraction range")
            impact_time = float(self.ballistic_target_time[slot])
            impact_delay = float(self.ballistic_time_to_impact[slot])
            if not math.isclose(
                impact_time, float(self.time[release]) + impact_delay, rel_tol=0.0, abs_tol=2e-14
            ) or not (self.time[release] < impact_time <= self.time[-1]):
                raise ValueError("ballistic target time and time-to-impact are inconsistent")
            available = min(
                float(self.ballistic_time_to_impact_cap), float(self.time[-1] - self.time[release])
            )
            impact_fraction = impact_delay / available
            impact_bin = int(self.ballistic_time_to_impact_bin[slot])
            bin_lower, bin_upper = self.ballistic_time_to_impact_fraction_bins[impact_bin]
            if impact_fraction < bin_lower - 1e-14 or impact_fraction > bin_upper + 1e-14:
                raise ValueError("ballistic time-to-impact lies outside its declared bin")
            tau = self.time - self.time[release]
            active_tau = tau[release:, None]
            truth_position = (
                self.ballistic_release_position[slot]
                + active_tau * self.ballistic_release_velocity[slot]
                + 0.5 * active_tau**2 * self.gravity
            )
            truth_velocity = self.ballistic_release_velocity[slot] + active_tau * self.gravity
            if not np.allclose(
                self.dynamic_positions[release:, slot], truth_position, rtol=1e-12, atol=1e-12
            ) or not np.allclose(
                self.dynamic_velocities[release:, slot], truth_velocity, rtol=1e-12, atol=1e-12
            ):
                raise ValueError("ballistic truth trajectory is not analytic free flight")
            if np.any(self.dynamic_positions[:release, slot] != 0.0) or np.any(
                self.dynamic_velocities[:release, slot] != 0.0
            ):
                raise ValueError("pre-release ballistic truth padding must be zero")
            encounter_position = (
                self.ballistic_release_position[slot]
                + impact_delay * self.ballistic_release_velocity[slot]
                + 0.5 * impact_delay**2 * self.gravity
            )
            reference_at_impact = self.defender_reference_position[0] + (
                impact_time * self.defender_reference_velocity[0]
            )
            intended_vector = self.ballistic_intended_miss_vector[slot]
            intended_distance = float(self.ballistic_intended_miss_distance[slot])
            if not np.allclose(
                encounter_position - reference_at_impact, intended_vector, rtol=0.0, atol=2e-12
            ) or not math.isclose(
                float(np.linalg.norm(intended_vector)),
                intended_distance,
                rel_tol=0.0,
                abs_tol=2e-12,
            ):
                raise ValueError("ballistic intended closest-approach metadata is inconsistent")
            physical_radius = float(self.vehicle_radius + self.dynamic_radii[slot])
            stratum = int(self.ballistic_encounter_stratum[slot])
            distance_bounds = (
                (0.0, physical_radius),
                (physical_radius, physical_radius + float(self.ballistic_safety_buffer)),
                (
                    physical_radius + float(self.ballistic_safety_buffer),
                    physical_radius
                    + float(self.ballistic_safety_buffer)
                    + float(self.ballistic_near_miss_extra),
                ),
            )[stratum]
            if not distance_bounds[0] < intended_distance < distance_bounds[1]:
                raise ValueError("ballistic intended miss distance violates its encounter stratum")
            realized_time, realized_distance = _ballistic_reference_closest_approach(
                self.ballistic_release_position[slot],
                self.ballistic_release_velocity[slot],
                float(self.time[release]),
                float(self.time[-1]),
                self.gravity,
                self.defender_reference_position[0],
                self.defender_reference_velocity[0],
            )
            if (
                not math.isclose(
                    float(self.ballistic_realized_closest_time[slot]),
                    realized_time,
                    rel_tol=0.0,
                    abs_tol=2e-10,
                )
                or not math.isclose(
                    float(self.ballistic_realized_closest_distance[slot]),
                    realized_distance,
                    rel_tol=0.0,
                    abs_tol=2e-10,
                )
                or not math.isclose(realized_time, impact_time, rel_tol=0.0, abs_tol=2e-9)
                or not math.isclose(realized_distance, intended_distance, rel_tol=0.0, abs_tol=2e-9)
            ):
                raise ValueError("ballistic realized closest approach does not match its target")
            for sample in range(samples):
                scenario_velocity = self.ballistic_prediction_release_velocity[sample, slot]
                predicted_position = (
                    self.ballistic_release_position[slot]
                    + active_tau * scenario_velocity
                    + 0.5 * active_tau**2 * self.gravity
                )
                predicted_velocity = scenario_velocity + active_tau * self.gravity
                if not np.allclose(
                    self.prediction_positions[sample, release:, slot],
                    predicted_position,
                    rtol=1e-12,
                    atol=1e-12,
                ) or not np.allclose(
                    self.prediction_velocities[sample, release:, slot],
                    predicted_velocity,
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise ValueError("ballistic prediction is not analytic free flight")
                if np.any(self.prediction_positions[sample, :release, slot] != 0.0) or np.any(
                    self.prediction_velocities[sample, :release, slot] != 0.0
                ):
                    raise ValueError("pre-release ballistic prediction padding must be zero")

        if np.any(
            self.dynamic_time_mask[:, attackers]
            != np.broadcast_to(real[attackers], (steps, attackers.sum()))
        ):
            raise ValueError("attacker drones must be active for the complete time grid")
        self._validate_attacker_equations(attackers)
        self._validate_motion_bounds(dt)

    def _validate_attacker_equations(self, attackers: np.ndarray) -> None:
        """Reconstruct every fixed attacker trajectory from its declared model."""
        for slot in np.flatnonzero(attackers):
            expected_position, expected_velocity = _steered_attacker_trajectory(
                AttackerMode(int(self.attacker_mode[slot])),
                self.dynamic_positions[0, slot],
                self.dynamic_velocities[0, slot],
                float(self.dynamic_speed_limit[slot]),
                float(self.dynamic_acceleration_limit[slot]),
                float(self.interceptor_prediction_horizon),
                self.time,
                self.defender_reference_position,
                self.defender_reference_velocity,
            )
            if (
                self.attacker_mode[slot] == int(AttackerMode.SCRIPTED_CROSSING)
                and not self.randomized_attacker[slot]
            ):
                expected_velocity = np.broadcast_to(
                    self.dynamic_velocities[0, slot], self.dynamic_velocities[:, slot].shape
                )
                expected_position = (
                    self.dynamic_positions[0, slot]
                    + self.time[:, None] * self.dynamic_velocities[0, slot]
                )
            if not np.allclose(
                self.dynamic_positions[:, slot], expected_position, rtol=1e-12, atol=1e-12
            ) or not np.allclose(
                self.dynamic_velocities[:, slot], expected_velocity, rtol=1e-12, atol=1e-12
            ):
                raise ValueError("attacker truth trajectory violates its declared model")
            for sample in range(self.prediction_samples):
                if not self.randomized_attacker[slot]:
                    if not np.array_equal(
                        self.prediction_positions[sample, :, slot], self.dynamic_positions[:, slot]
                    ) or not np.array_equal(
                        self.prediction_velocities[sample, :, slot],
                        self.dynamic_velocities[:, slot],
                    ):
                        raise ValueError("fixed-mode attacker predictions must equal truth")
                    continue
                predicted_position, predicted_velocity = _steered_attacker_trajectory(
                    AttackerMode(int(self.prediction_attacker_mode[sample, slot])),
                    self.dynamic_positions[0, slot],
                    self.dynamic_velocities[0, slot],
                    float(self.dynamic_speed_limit[slot]),
                    float(self.dynamic_acceleration_limit[slot]),
                    float(self.interceptor_prediction_horizon),
                    self.time,
                    self.defender_reference_position,
                    self.defender_reference_velocity,
                )
                if not np.allclose(
                    self.prediction_positions[sample, :, slot],
                    predicted_position,
                    rtol=1e-12,
                    atol=1e-12,
                ) or not np.allclose(
                    self.prediction_velocities[sample, :, slot],
                    predicted_velocity,
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise ValueError("attacker prediction violates its selected finite mode")

    def _validate_motion_bounds(self, dt: float) -> None:
        tolerance = 2e-10
        truth_speed = np.linalg.norm(self.dynamic_velocities, axis=-1)
        if np.any(
            truth_speed[self.dynamic_time_mask]
            > np.broadcast_to(self.dynamic_speed_limit, truth_speed.shape)[self.dynamic_time_mask]
            + tolerance
        ):
            raise ValueError("truth trajectory exceeds its speed bound")
        prediction_mask = np.broadcast_to(
            self.dynamic_time_mask, self.prediction_velocities.shape[:-1]
        )
        prediction_speed = np.linalg.norm(self.prediction_velocities, axis=-1)
        prediction_limit = np.broadcast_to(
            self.dynamic_speed_limit, self.prediction_velocities.shape[:-1]
        )
        if np.any(
            prediction_speed[prediction_mask] > prediction_limit[prediction_mask] + tolerance
        ):
            raise ValueError("prediction trajectory exceeds its speed bound")

        adjacent = self.dynamic_time_mask[1:] & self.dynamic_time_mask[:-1]
        truth_acceleration = np.linalg.norm(np.diff(self.dynamic_velocities, axis=0) / dt, axis=-1)
        truth_limit = np.broadcast_to(self.dynamic_acceleration_limit, truth_acceleration.shape)
        if np.any(truth_acceleration[adjacent] > truth_limit[adjacent] + tolerance):
            raise ValueError("truth trajectory exceeds its acceleration bound")
        prediction_adjacent = np.broadcast_to(
            adjacent, (self.prediction_velocities.shape[0], *adjacent.shape)
        )
        prediction_acceleration = np.linalg.norm(
            np.diff(self.prediction_velocities, axis=1) / dt, axis=-1
        )
        prediction_acceleration_limit = np.broadcast_to(
            self.dynamic_acceleration_limit, prediction_acceleration.shape
        )
        if np.any(
            prediction_acceleration[prediction_adjacent]
            > prediction_acceleration_limit[prediction_adjacent] + tolerance
        ):
            raise ValueError("prediction trajectory exceeds its acceleration bound")

    def _validate_schedules(self) -> None:
        steps = self.time.size
        _require_float_shape(self.wind_velocity, (steps, 3), "wind_velocity")
        _require_float_shape(self.wind_speed_limit, (), "wind_speed_limit")
        if float(self.wind_speed_limit) <= 0.0 or np.any(
            np.linalg.norm(self.wind_velocity, axis=-1) > float(self.wind_speed_limit) + 1e-12
        ):
            raise ValueError("wind schedule exceeds its positive speed limit")
        _require_float_shape(self.mass_scale, (steps,), "mass_scale")
        _require_float_shape(self.mass_scale_bounds, (2,), "mass_scale_bounds")
        _require_float_shape(self.drag_scale, (steps, 3), "drag_scale")
        _require_float_shape(self.drag_scale_bounds, (2,), "drag_scale_bounds")
        _require_float_shape(self.rotor_efficiency, (steps, 4), "rotor_efficiency")
        _require_float_shape(self.rotor_efficiency_bounds, (2,), "rotor_efficiency_bounds")
        for values, bounds, name in (
            (self.mass_scale, self.mass_scale_bounds, "mass_scale"),
            (self.drag_scale, self.drag_scale_bounds, "drag_scale"),
            (self.rotor_efficiency, self.rotor_efficiency_bounds, "rotor_efficiency"),
        ):
            if bounds[0] <= 0.0 or bounds[1] < bounds[0] or not bounds[0] <= 1.0 <= bounds[1]:
                raise ValueError(f"{name} bounds must be positive, ordered, and contain one")
            if np.any(values < bounds[0]) or np.any(values > bounds[1]):
                raise ValueError(f"{name} schedule exceeds its declared bounds")
        _require_scalar(self.rotor_single_index, np.int8, "rotor_single_index")
        if not 0 <= int(self.rotor_single_index) < 4:
            raise ValueError("rotor_single_index must lie in [0, 4)")
        if self.schedule_names.dtype.kind != "U" or not np.array_equal(
            self.schedule_names, np.asarray(SCHEDULE_NAMES)
        ):
            raise ValueError("schedule_names do not match the schema")
        _require_dtype_shape(
            self.schedule_change_indices,
            np.int32,
            (len(SCHEDULE_NAMES),),
            "schedule_change_indices",
        )
        if (
            np.any(self.schedule_change_indices < 0)
            or np.any(self.schedule_change_indices >= steps)
            or np.any(np.diff(self.schedule_change_indices) < 0)
        ):
            raise ValueError("schedule change indices must be ordered nodes on the time grid")
        _, mass_index, drag_index, symmetric_index, single_index = (
            int(value) for value in self.schedule_change_indices
        )
        if np.any(self.mass_scale[:mass_index] != 1.0) or not np.all(
            self.mass_scale[mass_index:] == self.mass_scale[mass_index]
        ):
            raise ValueError("mass_scale must be one declared step change")
        if np.any(self.drag_scale[:drag_index] != 1.0) or not np.all(
            self.drag_scale[drag_index:] == self.drag_scale[drag_index]
        ):
            raise ValueError("drag_scale must be one declared vector step change")
        rotor_index = int(self.rotor_single_index)
        other_rotors = np.arange(4) != rotor_index
        if np.any(self.rotor_efficiency[:symmetric_index] != 1.0):
            raise ValueError("rotor efficiency must be nominal before its declared change")
        symmetric_value = self.rotor_efficiency[symmetric_index, np.flatnonzero(other_rotors)[0]]
        if not np.all(
            self.rotor_efficiency[symmetric_index:, other_rotors] == symmetric_value
        ) or np.any(
            self.rotor_efficiency[symmetric_index:single_index, rotor_index] != symmetric_value
        ):
            raise ValueError("rotor schedule must apply its declared symmetric change")
        if (
            not np.all(
                self.rotor_efficiency[single_index:, rotor_index]
                == self.rotor_efficiency[single_index, rotor_index]
            )
            or self.rotor_efficiency[single_index, rotor_index] > symmetric_value
        ):
            raise ValueError("rotor schedule must apply one non-increasing single-rotor change")
        _require_scalar(self.contact_is_failure, np.bool_, "contact_is_failure")
        if not bool(self.contact_is_failure):
            raise ValueError("DA-PLCBF scenario tapes must label every contact as failure")


@dataclass(frozen=True, slots=True, eq=False)
class ContactLabels:
    """Pure geometric contact-event labels with no contact response or state correction."""

    static: np.ndarray
    dynamic: np.ndarray
    any_contact: np.ndarray

    def __post_init__(self) -> None:
        """Defensively freeze and type-check every contact mask."""
        for item in fields(self):
            value = _frozen_array(getattr(self, item.name))
            if value.dtype != np.dtype(np.bool_):
                raise ValueError(f"{item.name} must be boolean")
            object.__setattr__(self, item.name, value)
        if self.static.ndim != 2 or self.dynamic.ndim != 2:
            raise ValueError("static and dynamic contact labels must have shape (T, slots)")
        if self.static.shape[0] != self.dynamic.shape[0] or self.any_contact.shape != (
            self.static.shape[0],
        ):
            raise ValueError("contact-label time dimensions must agree")
        if not np.array_equal(
            self.any_contact, np.any(self.static, axis=1) | np.any(self.dynamic, axis=1)
        ):
            raise ValueError("any_contact must be the exact union of all contact labels")


def generate_scenario_tape(
    seed: int, config: ScenarioTapeConfig | None = None, *, fold: int = 0
) -> ScenarioTape:
    """Generate one deterministic fixed-shape scenario tape.

    The trajectory tensor contains one truth path per obstacle plus a finite prediction set.  A
    randomized attacker independently selects one of the three declared attacker modes for truth
    and for each prediction scenario; its selected mode never changes along a trajectory.
    """
    resolved = ScenarioTapeConfig() if config is None else config
    if not isinstance(resolved, ScenarioTapeConfig):
        raise TypeError("config must be a ScenarioTapeConfig")
    resolved.validate()
    root_seed = _validate_uint32(seed, "seed")
    fold_value = _validate_uint32(fold, "fold")

    time = np.arange(resolved.steps, dtype=np.float64) * float(resolved.dt)
    arena_lower = np.asarray(resolved.arena_lower, dtype=np.float64)
    arena_upper = np.asarray(resolved.arena_upper, dtype=np.float64)
    vehicle_position = np.asarray(resolved.vehicle_initial_position, dtype=np.float64)
    vehicle_velocity = np.asarray(resolved.vehicle_initial_velocity, dtype=np.float64)
    reference_initial_position = np.asarray(
        resolved.vehicle_initial_position
        if resolved.reference_initial_position is None
        else resolved.reference_initial_position,
        dtype=np.float64,
    )
    reference_initial_velocity = np.asarray(
        resolved.vehicle_initial_velocity
        if resolved.reference_initial_velocity is None
        else resolved.reference_initial_velocity,
        dtype=np.float64,
    )
    reference_position = (
        reference_initial_position[None, :] + time[:, None] * reference_initial_velocity[None, :]
    )
    reference_velocity = np.broadcast_to(reference_initial_velocity, (resolved.steps, 3)).copy()

    static_positions, static_radii, static_mask = _generate_static_obstacles(
        root_seed, fold_value, resolved, arena_lower, arena_upper
    )
    arrays = _generate_dynamic_obstacles(
        root_seed,
        fold_value,
        resolved,
        time,
        arena_lower,
        arena_upper,
        reference_position,
        reference_velocity,
    )
    schedules = _generate_dynamics_schedules(root_seed, fold_value, resolved, time)
    estimator_noise = _generate_estimator_observation_noise(
        root_seed,
        fold_value,
        resolved.steps,
        acceleration_std=resolved.estimator_acceleration_noise_std,
        motor_force_std=resolved.estimator_motor_force_noise_std,
    )

    config_payload = json.dumps(
        asdict(resolved), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return ScenarioTape(
        schema_version=np.asarray(SCENARIO_TAPE_SCHEMA_VERSION, dtype=np.uint16),
        root_seed=np.asarray(root_seed, dtype=np.uint32),
        generation_fold=np.asarray(fold_value, dtype=np.uint32),
        generator_config_sha256=np.asarray(hashlib.sha256(config_payload).hexdigest()),
        rng_stream_names=np.asarray(RNG_STREAM_NAMES),
        rng_stream_ids=np.asarray(
            [RNG_STREAM_IDS[name] for name in RNG_STREAM_NAMES], dtype=np.uint32
        ),
        time=time,
        arena_lower=arena_lower,
        arena_upper=arena_upper,
        vehicle_initial_position=vehicle_position,
        vehicle_initial_velocity=vehicle_velocity,
        vehicle_radius=np.asarray(resolved.vehicle_radius, dtype=np.float64),
        defender_reference_position=reference_position,
        defender_reference_velocity=reference_velocity,
        static_positions=static_positions,
        static_radii=static_radii,
        static_mask=static_mask,
        **arrays,
        **schedules,
        **estimator_noise,
        contact_is_failure=np.asarray(True, dtype=np.bool_),
    )


def hard_contact_labels(vehicle_positions: Any, tape: ScenarioTape) -> ContactLabels:
    """Evaluate hard spherical contact labels without modifying either trajectory.

    Args:
        vehicle_positions: Finite vehicle centres with shape ``(T, 3)``.
        tape: Valid immutable scenario tape.

    Returns:
        Static, dynamic, and aggregate contact masks.  Equality is contact and therefore failure.
    """
    if not isinstance(tape, ScenarioTape):
        raise TypeError("tape must be a ScenarioTape")
    positions = np.asarray(vehicle_positions, dtype=np.float64)
    if positions.shape != (tape.steps, 3) or not np.all(np.isfinite(positions)):
        raise ValueError("vehicle_positions must be finite with shape (T, 3)")
    static_distance = np.linalg.norm(
        positions[:, None, :] - tape.static_positions[None, :, :], axis=-1
    )
    static = (
        static_distance <= tape.vehicle_radius + tape.static_radii[None, :]
    ) & tape.static_mask[None, :]
    dynamic_distance = np.linalg.norm(positions[:, None, :] - tape.dynamic_positions, axis=-1)
    dynamic = (
        dynamic_distance <= tape.vehicle_radius + tape.dynamic_radii[None, :]
    ) & tape.dynamic_time_mask
    return ContactLabels(
        static=static, dynamic=dynamic, any_contact=np.any(static, axis=1) | np.any(dynamic, axis=1)
    )


def save_scenario_tape(
    tape: ScenarioTape, path: str | os.PathLike[str], *, overwrite: bool = False
) -> str:
    """Atomically save a tape as a deterministic NPZ archive and return its content digest."""
    if not isinstance(tape, ScenarioTape):
        raise TypeError("tape must be a ScenarioTape")
    tape.validate()
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("scenario-tape path must end in .npz")
    if not destination.parent.exists():
        raise FileNotFoundError("scenario-tape parent directory does not exist")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)

    payload = _scenario_tape_arrays(tape)
    payload["content_sha256"] = np.asarray(tape.sha256)
    archive_bytes = _deterministic_npz_bytes(payload)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(archive_bytes)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return tape.sha256


def load_scenario_tape(path: str | os.PathLike[str]) -> ScenarioTape:
    """Load, validate, and hash-check a deterministic scenario-tape NPZ archive."""
    source = Path(path)
    if source.suffix.lower() != ".npz":
        raise ValueError("scenario-tape path must end in .npz")
    try:
        archive = zipfile.ZipFile(source, mode="r")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("scenario tape is not a valid NPZ archive") from error
    expected_fields = {item.name for item in fields(ScenarioTape)}
    expected_members = {f"{name}.npy" for name in (*expected_fields, "content_sha256")}
    try:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or set(names) != expected_members:
            raise ValueError("scenario tape has missing, duplicate, or unexpected members")
        if sum(info.file_size for info in infos) > _MAX_ARCHIVE_BYTES:
            raise ValueError("scenario tape exceeds the decompressed size limit")
        loaded: dict[str, np.ndarray] = {}
        for info in infos:
            try:
                member = archive.read(info)
                array = np.load(io.BytesIO(member), allow_pickle=False)
            except (EOFError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
                raise ValueError(f"invalid scenario-tape member {info.filename!r}") from error
            if not isinstance(array, np.ndarray) or array.dtype.hasobject:
                raise ValueError(
                    f"scenario-tape member {info.filename!r} is not a numeric/string array"
                )
            loaded[info.filename.removesuffix(".npy")] = array
    finally:
        archive.close()

    stored_digest_array = loaded.pop("content_sha256")
    if stored_digest_array.shape != () or stored_digest_array.dtype.kind != "U":
        raise ValueError("scenario-tape content_sha256 member is invalid")
    try:
        tape = ScenarioTape(**loaded)
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("scenario-tape payload failed schema validation") from error
    if not _constant_time_digest_equal(str(stored_digest_array), tape.sha256):
        raise ValueError("scenario-tape content digest mismatch")
    return tape


def _generate_static_obstacles(
    seed: int,
    fold: int,
    config: ScenarioTapeConfig,
    arena_lower: np.ndarray,
    arena_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.zeros((config.static_capacity, 3), dtype=np.float64)
    radii = np.zeros(config.static_capacity, dtype=np.float64)
    mask = np.arange(config.static_capacity) < config.static_count
    if config.static_count == 0:
        return positions, radii, mask
    key = named_rng_key(seed, "static_obstacles", fold=fold)
    radius_key, position_key = jr.split(key)
    radius_range = np.asarray(config.static_radius_range, dtype=np.float64)
    radii[: config.static_count] = _uniform(
        radius_key, (config.static_count,), radius_range[0], radius_range[1]
    )
    unit = _uniform(position_key, (config.static_count, 3), 0.0, 1.0)
    lower = arena_lower + radii[: config.static_count, None]
    upper = arena_upper - radii[: config.static_count, None]
    positions[: config.static_count] = lower + unit * (upper - lower)
    return positions, radii, mask


def _generate_dynamic_obstacles(
    seed: int,
    fold: int,
    config: ScenarioTapeConfig,
    time: np.ndarray,
    arena_lower: np.ndarray,
    arena_upper: np.ndarray,
    reference_position: np.ndarray,
    reference_velocity: np.ndarray,
) -> dict[str, np.ndarray]:
    steps = config.steps
    capacity = config.dynamic_capacity
    samples = config.prediction_samples
    positions = np.zeros((steps, capacity, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    prediction_positions = np.zeros((samples, steps, capacity, 3), dtype=np.float64)
    prediction_velocities = np.zeros_like(prediction_positions)
    radii = np.zeros(capacity, dtype=np.float64)
    slot_mask = np.zeros(capacity, dtype=np.bool_)
    time_mask = np.zeros((steps, capacity), dtype=np.bool_)
    kinds = np.full(capacity, int(DynamicObstacleKind.PADDING), dtype=np.int8)
    modes = np.full(capacity, int(AttackerMode.NOT_ATTACKER), dtype=np.int8)
    randomized = np.zeros(capacity, dtype=np.bool_)
    prediction_modes = np.full((samples, capacity), int(AttackerMode.NOT_ATTACKER), dtype=np.int8)
    release_indices = np.full(capacity, -1, dtype=np.int32)
    release_positions = np.zeros((capacity, 3), dtype=np.float64)
    release_velocities = np.zeros((capacity, 3), dtype=np.float64)
    prediction_release_velocities = np.zeros((samples, capacity, 3), dtype=np.float64)
    encounter_strata = np.full(capacity, -1, dtype=np.int8)
    impact_bins = np.full(capacity, -1, dtype=np.int8)
    target_times = np.zeros(capacity, dtype=np.float64)
    impact_delays = np.zeros(capacity, dtype=np.float64)
    intended_miss_vectors = np.zeros((capacity, 3), dtype=np.float64)
    intended_miss_distances = np.zeros(capacity, dtype=np.float64)
    realized_closest_times = np.zeros(capacity, dtype=np.float64)
    realized_closest_distances = np.zeros(capacity, dtype=np.float64)
    generation_attempts = np.zeros(capacity, dtype=np.int32)
    speed_limits = np.zeros(capacity, dtype=np.float64)
    acceleration_limits = np.zeros(capacity, dtype=np.float64)

    real_count = (
        config.ballistic_count
        + config.crossing_count
        + config.pursuit_count
        + config.interceptor_count
        + config.random_attacker_count
    )
    slot_mask[:real_count] = True
    if real_count:
        radius_key = named_rng_key(seed, "dynamic_radii", fold=fold)
        radius_range = np.asarray(config.dynamic_radius_range, dtype=np.float64)
        radii[:real_count] = _uniform(radius_key, (real_count,), radius_range[0], radius_range[1])

    slot = 0
    slot = _populate_ballistic_slots(
        seed,
        fold,
        config,
        time,
        arena_lower,
        arena_upper,
        reference_position,
        reference_velocity,
        slot,
        positions,
        velocities,
        prediction_positions,
        prediction_velocities,
        radii,
        time_mask,
        kinds,
        release_indices,
        release_positions,
        release_velocities,
        prediction_release_velocities,
        encounter_strata,
        impact_bins,
        target_times,
        impact_delays,
        intended_miss_vectors,
        intended_miss_distances,
        realized_closest_times,
        realized_closest_distances,
        generation_attempts,
        speed_limits,
        acceleration_limits,
    )
    attacker_groups = (
        (config.crossing_count, AttackerMode.SCRIPTED_CROSSING, "scripted_crossing", False),
        (config.pursuit_count, AttackerMode.BOUNDED_PURSUIT, "bounded_pursuit", False),
        (
            config.interceptor_count,
            AttackerMode.PREDICTIVE_INTERCEPTOR,
            "predictive_interceptor",
            False,
        ),
        (config.random_attacker_count, None, "random_attacker_truth", True),
    )
    for count, fixed_mode, stream, is_random in attacker_groups:
        if count == 0:
            continue
        key = named_rng_key(seed, stream, fold=fold)
        group_keys = jr.split(key, count)
        for group_index in range(count):
            selected_mode = (
                AttackerMode(
                    int(
                        jr.randint(
                            jr.fold_in(group_keys[group_index], 0),
                            (),
                            int(AttackerMode.SCRIPTED_CROSSING),
                            int(AttackerMode.PREDICTIVE_INTERCEPTOR) + 1,
                        )
                    )
                )
                if is_random
                else fixed_mode
            )
            assert selected_mode is not None
            trajectory = _generate_attacker(
                group_keys[group_index],
                selected_mode,
                config,
                time,
                arena_lower,
                arena_upper,
                reference_position,
                reference_velocity,
                exact_scripted_crossing=not is_random,
            )
            positions[:, slot], velocities[:, slot], max_speed = trajectory
            prediction_positions[:, :, slot] = positions[None, :, slot]
            prediction_velocities[:, :, slot] = velocities[None, :, slot]
            time_mask[:, slot] = True
            kinds[slot] = int(DynamicObstacleKind.ATTACKER_DRONE)
            modes[slot] = int(selected_mode)
            prediction_modes[:, slot] = int(selected_mode)
            randomized[slot] = is_random
            speed_limits[slot] = max_speed
            acceleration_limits[slot] = config.attacker_acceleration_limit
            slot += 1

    random_slots = np.flatnonzero(randomized)
    if random_slots.size:
        prediction_key = named_rng_key(seed, "random_attacker_predictions", fold=fold)
        keys = jr.split(prediction_key, samples * random_slots.size).reshape(
            samples, random_slots.size, 2
        )
        for sample in range(samples):
            for local_index, random_slot in enumerate(random_slots):
                key = keys[sample, local_index]
                selected_mode = AttackerMode(
                    int(
                        jr.randint(
                            jr.fold_in(key, 0),
                            (),
                            int(AttackerMode.SCRIPTED_CROSSING),
                            int(AttackerMode.PREDICTIVE_INTERCEPTOR) + 1,
                        )
                    )
                )
                # Predictions share the observed initial state and declared physical bounds.  Only
                # the finite attacker model (and its deterministic steering path) varies.
                predicted_position, predicted_velocity = _steered_attacker_trajectory(
                    selected_mode,
                    positions[0, random_slot],
                    velocities[0, random_slot],
                    speed_limits[random_slot],
                    config.attacker_acceleration_limit,
                    config.interceptor_prediction_horizon,
                    time,
                    reference_position,
                    reference_velocity,
                )
                prediction_positions[sample, :, random_slot] = predicted_position
                prediction_velocities[sample, :, random_slot] = predicted_velocity
                prediction_modes[sample, random_slot] = int(selected_mode)

    return {
        "dynamic_positions": positions,
        "dynamic_velocities": velocities,
        "dynamic_radii": radii,
        "dynamic_slot_mask": slot_mask,
        "dynamic_time_mask": time_mask,
        "dynamic_kind": kinds,
        "attacker_mode": modes,
        "randomized_attacker": randomized,
        "prediction_positions": prediction_positions,
        "prediction_velocities": prediction_velocities,
        "prediction_attacker_mode": prediction_modes,
        "ballistic_release_index": release_indices,
        "ballistic_release_position": release_positions,
        "ballistic_release_velocity": release_velocities,
        "ballistic_prediction_release_velocity": prediction_release_velocities,
        "ballistic_velocity_lower": np.asarray(config.ball_velocity_lower, dtype=np.float64),
        "ballistic_velocity_upper": np.asarray(config.ball_velocity_upper, dtype=np.float64),
        "ballistic_velocity_uncertainty": np.asarray(
            config.ball_velocity_uncertainty, dtype=np.float64
        ),
        "ballistic_release_fraction_range": np.asarray(
            config.ball_release_fraction_range, dtype=np.float64
        ),
        "ballistic_safety_buffer": np.asarray(config.ballistic_safety_buffer, dtype=np.float64),
        "ballistic_near_miss_extra": np.asarray(config.ballistic_near_miss_extra, dtype=np.float64),
        "ballistic_time_to_impact_cap": np.asarray(
            config.ballistic_time_to_impact_cap, dtype=np.float64
        ),
        "ballistic_time_to_impact_fraction_bins": np.asarray(
            config.ballistic_time_to_impact_fraction_bins, dtype=np.float64
        ),
        "ballistic_generation_max_attempts": np.asarray(
            config.ballistic_generation_max_attempts, dtype=np.int32
        ),
        "ballistic_encounter_stratum": encounter_strata,
        "ballistic_time_to_impact_bin": impact_bins,
        "ballistic_target_time": target_times,
        "ballistic_time_to_impact": impact_delays,
        "ballistic_intended_miss_vector": intended_miss_vectors,
        "ballistic_intended_miss_distance": intended_miss_distances,
        "ballistic_realized_closest_time": realized_closest_times,
        "ballistic_realized_closest_distance": realized_closest_distances,
        "ballistic_generation_attempts": generation_attempts,
        "gravity": np.asarray(config.gravity, dtype=np.float64),
        "dynamic_speed_limit": speed_limits,
        "dynamic_acceleration_limit": acceleration_limits,
        "interceptor_prediction_horizon": np.asarray(
            config.interceptor_prediction_horizon, dtype=np.float64
        ),
    }


def _fixed_ballistic_prediction_offsets(samples: int, uncertainty: np.ndarray) -> np.ndarray:
    """Return fixed bounded velocity offsets with the exact truth as scenario zero."""
    offsets = np.zeros((samples, 3), dtype=np.float64)
    if samples == 1:
        return offsets
    count = samples - 1
    indices = np.arange(count, dtype=np.float64)
    # A deterministic spherical lattice avoids seed-dependent support and keeps every component
    # inside [-1, 1].  Scenario zero is exactly the truth trajectory.
    z = -1.0 + 2.0 * (indices + 0.5) / count
    angle = math.pi * (3.0 - math.sqrt(5.0)) * indices
    radius = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    offsets[1:, 0] = radius * np.cos(angle)
    offsets[1:, 1] = radius * np.sin(angle)
    offsets[1:, 2] = z
    return offsets * np.asarray(uncertainty, dtype=np.float64)[None]


def _common_perpendicular(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return a deterministic unit vector orthogonal to both supplied 3-vectors."""
    normal = np.cross(first, second)
    norm = float(np.linalg.norm(normal))
    scale = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1.0)
    if norm <= 64.0 * np.finfo(np.float64).eps * scale:
        reference = second if np.linalg.norm(second) > 0.0 else first
        axis = np.eye(3)[int(np.argmin(np.abs(reference)))]
        normal = np.cross(reference, axis)
        norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("ballistic encounter has no finite perpendicular miss direction")
    return normal / norm


def _ballistic_reference_closest_approach(
    release_position: np.ndarray,
    release_velocity: np.ndarray,
    release_time: float,
    end_time: float,
    gravity: np.ndarray,
    reference_initial_position: np.ndarray,
    reference_velocity: np.ndarray,
) -> tuple[float, float]:
    """Compute the exact continuous closest approach to a constant-velocity reference."""
    duration = end_time - release_time
    reference_at_release = reference_initial_position + release_time * reference_velocity
    relative_position = release_position - reference_at_release
    relative_velocity = release_velocity - reference_velocity
    coefficients = np.asarray(
        (
            0.5 * np.dot(gravity, gravity),
            1.5 * np.dot(relative_velocity, gravity),
            np.dot(relative_velocity, relative_velocity) + np.dot(relative_position, gravity),
            np.dot(relative_position, relative_velocity),
        ),
        dtype=np.float64,
    )
    nonzero = np.flatnonzero(np.abs(coefficients) > 64.0 * np.finfo(np.float64).eps)
    roots = np.roots(coefficients[nonzero[0] :]) if nonzero.size else np.empty(0)
    candidates = [0.0, duration]
    for root in roots:
        if abs(float(np.imag(root))) <= 1e-10:
            value = float(np.real(root))
            if -1e-12 <= value <= duration + 1e-12:
                candidates.append(float(np.clip(value, 0.0, duration)))
    delays = np.asarray(candidates, dtype=np.float64)
    relative = (
        relative_position[None]
        + delays[:, None] * relative_velocity[None]
        + 0.5 * delays[:, None] ** 2 * gravity[None]
    )
    distances = np.linalg.norm(relative, axis=-1)
    best = int(np.argmin(distances))
    return release_time + float(delays[best]), float(distances[best])


def _populate_ballistic_slots(
    seed: int,
    fold: int,
    config: ScenarioTapeConfig,
    time: np.ndarray,
    arena_lower: np.ndarray,
    arena_upper: np.ndarray,
    reference_position: np.ndarray,
    reference_velocity: np.ndarray,
    first_slot: int,
    positions: np.ndarray,
    velocities: np.ndarray,
    prediction_positions: np.ndarray,
    prediction_velocities: np.ndarray,
    radii: np.ndarray,
    time_mask: np.ndarray,
    kinds: np.ndarray,
    release_indices: np.ndarray,
    release_positions: np.ndarray,
    release_velocities: np.ndarray,
    prediction_release_velocities: np.ndarray,
    encounter_strata: np.ndarray,
    impact_bins: np.ndarray,
    target_times: np.ndarray,
    impact_delays: np.ndarray,
    intended_miss_vectors: np.ndarray,
    intended_miss_distances: np.ndarray,
    realized_closest_times: np.ndarray,
    realized_closest_distances: np.ndarray,
    generation_attempts: np.ndarray,
    speed_limits: np.ndarray,
    acceleration_limits: np.ndarray,
) -> int:
    """Populate predeclared controller-independent reference encounters.

    Categorical difficulty is a fixed function of ``fold``.  Seeded draws choose only continuous
    values inside that stratum.  Each candidate samples a release node, impact delay, and interior
    release velocity; the release point is then constructed and the velocity is independently
    recovered from the ballistic endpoint equation.  Candidates outside physical bounds are
    rejected for at most ``ballistic_generation_max_attempts`` attempts.
    """
    count = config.ballistic_count
    if count == 0:
        return first_slot
    truth_key = named_rng_key(seed, "ballistic_truth", fold=fold)
    minimum_fraction, maximum_fraction = config.ball_release_fraction_range
    minimum_index = int(math.floor(minimum_fraction * (config.steps - 1)))
    maximum_index_exclusive = int(math.floor(maximum_fraction * (config.steps - 1))) + 1
    maximum_index_exclusive = min(maximum_index_exclusive, config.steps - 1)
    if maximum_index_exclusive <= minimum_index:
        raise ValueError("ballistic release range leaves no pre-terminal release node")
    lower = np.asarray(config.ball_velocity_lower, dtype=np.float64)
    upper = np.asarray(config.ball_velocity_upper, dtype=np.float64)
    uncertainty = np.asarray(config.ball_velocity_uncertainty, dtype=np.float64)
    interior_lower = lower + uncertainty
    interior_upper = upper - uncertainty
    prediction_offsets = _fixed_ballistic_prediction_offsets(config.prediction_samples, uncertainty)
    gravity = np.asarray(config.gravity, dtype=np.float64)
    horizon = float(time[-1])
    component_velocity_bound = np.maximum(np.abs(lower), np.abs(upper)) + np.abs(gravity) * horizon
    ballistic_speed_limit = float(np.linalg.norm(component_velocity_bound))
    stratum = BALLISTIC_ENCOUNTER_STRATUM_CYCLE[fold % len(BALLISTIC_ENCOUNTER_STRATUM_CYCLE)]
    fraction_lower, fraction_upper = config.ballistic_encounter_band_fraction_range
    impact_fraction_bins = np.asarray(
        config.ballistic_time_to_impact_fraction_bins, dtype=np.float64
    )

    for local_slot in range(count):
        slot = first_slot + local_slot
        slot_key = jr.fold_in(truth_key, local_slot)
        impact_bin = (fold + local_slot) % impact_fraction_bins.shape[0]
        solved: tuple[int, np.ndarray, np.ndarray, float, np.ndarray, float, float] | None = None
        for attempt in range(config.ballistic_generation_max_attempts):
            candidate_key = jr.fold_in(slot_key, attempt)
            index_key, impact_key, velocity_key, distance_key, orientation_key = jr.split(
                candidate_key, 5
            )
            release = int(jr.randint(index_key, (), minimum_index, maximum_index_exclusive))
            available = min(config.ballistic_time_to_impact_cap, float(time[-1] - time[release]))
            bin_lower, bin_upper = impact_fraction_bins[impact_bin]
            impact_fraction = float(_uniform(impact_key, (), bin_lower, bin_upper))
            impact_delay = available * impact_fraction
            target_time = float(time[release]) + impact_delay
            candidate_velocity = _uniform(velocity_key, (3,), interior_lower, interior_upper)
            reference_at_target = reference_position[0] + target_time * reference_velocity[0]
            relative_impact_velocity = (
                candidate_velocity + impact_delay * gravity - reference_velocity[0]
            )
            miss_normal = _common_perpendicular(relative_impact_velocity, gravity)
            orientation_sign = -1.0 if bool(jr.bernoulli(orientation_key)) else 1.0
            miss_normal *= orientation_sign
            physical_radius = config.vehicle_radius + float(radii[slot])
            band_lower, band_width = (
                (0.0, physical_radius),
                (physical_radius, config.ballistic_safety_buffer),
                (
                    physical_radius + config.ballistic_safety_buffer,
                    config.ballistic_near_miss_extra,
                ),
            )[stratum]
            within_band = float(_uniform(distance_key, (), fraction_lower, fraction_upper))
            intended_distance = band_lower + within_band * band_width
            miss_vector = intended_distance * miss_normal
            encounter_position = reference_at_target + miss_vector
            release_position = (
                encounter_position
                - impact_delay * candidate_velocity
                - 0.5 * impact_delay**2 * gravity
            )
            solved_velocity = (
                encounter_position - release_position - 0.5 * impact_delay**2 * gravity
            ) / impact_delay
            ball_radius = float(radii[slot])
            release_fits = np.all(release_position - ball_radius >= arena_lower) and np.all(
                release_position + ball_radius <= arena_upper
            )
            velocity_fits = np.all(solved_velocity >= interior_lower - 1e-14) and np.all(
                solved_velocity <= interior_upper + 1e-14
            )
            if not release_fits or not velocity_fits:
                continue
            realized_time, realized_distance = _ballistic_reference_closest_approach(
                release_position,
                solved_velocity,
                float(time[release]),
                float(time[-1]),
                gravity,
                reference_position[0],
                reference_velocity[0],
            )
            if not math.isclose(realized_time, target_time, rel_tol=0.0, abs_tol=2e-9):
                continue
            if not math.isclose(realized_distance, intended_distance, rel_tol=0.0, abs_tol=2e-9):
                continue
            solved = (
                release,
                release_position,
                solved_velocity,
                target_time,
                miss_vector,
                realized_time,
                realized_distance,
            )
            generation_attempts[slot] = attempt + 1
            break
        if solved is None:
            raise ValueError(
                "unable to construct bounded ballistic reference encounter "
                f"for fold={fold}, slot={local_slot} after "
                f"{config.ballistic_generation_max_attempts} attempts"
            )
        (
            release,
            release_position,
            solved_velocity,
            target_time,
            miss_vector,
            realized_time,
            realized_distance,
        ) = solved
        impact_delay = target_time - float(time[release])
        sampled_prediction_velocities = solved_velocity[None] + prediction_offsets
        release_indices[slot] = release
        release_positions[slot] = release_position
        release_velocities[slot] = solved_velocity
        prediction_release_velocities[:, slot] = sampled_prediction_velocities
        encounter_strata[slot] = stratum
        impact_bins[slot] = impact_bin
        target_times[slot] = target_time
        impact_delays[slot] = impact_delay
        intended_miss_vectors[slot] = miss_vector
        intended_miss_distances[slot] = np.linalg.norm(miss_vector)
        realized_closest_times[slot] = realized_time
        realized_closest_distances[slot] = realized_distance
        time_mask[release:, slot] = True
        kinds[slot] = int(DynamicObstacleKind.BALLISTIC)
        speed_limits[slot] = ballistic_speed_limit
        acceleration_limits[slot] = np.linalg.norm(gravity)
        tau = time[release:] - time[release]
        positions[release:, slot] = (
            release_position + tau[:, None] * solved_velocity + 0.5 * tau[:, None] ** 2 * gravity
        )
        velocities[release:, slot] = solved_velocity + tau[:, None] * gravity
        for sample in range(config.prediction_samples):
            sample_velocity = sampled_prediction_velocities[sample]
            prediction_positions[sample, release:, slot] = (
                release_position
                + tau[:, None] * sample_velocity
                + 0.5 * tau[:, None] ** 2 * gravity
            )
            prediction_velocities[sample, release:, slot] = sample_velocity + tau[:, None] * gravity
    return first_slot + count


def _generate_attacker(
    key: Array,
    mode: AttackerMode,
    config: ScenarioTapeConfig,
    time: np.ndarray,
    arena_lower: np.ndarray,
    arena_upper: np.ndarray,
    reference_position: np.ndarray,
    reference_velocity: np.ndarray,
    *,
    exact_scripted_crossing: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    position_key, velocity_key, speed_key, crossing_key = jr.split(key, 4)
    max_speed = float(
        _uniform(speed_key, (), config.attacker_speed_range[0], config.attacker_speed_range[1])
    )
    initial_position = arena_lower + _uniform(position_key, (3,), 0.0, 1.0) * (
        arena_upper - arena_lower
    )
    direction = _unit_vector(velocity_key)
    initial_speed = float(
        _uniform(
            jr.fold_in(velocity_key, 1), (), 0.0, config.attacker_initial_speed_fraction * max_speed
        )
    )
    initial_velocity = initial_speed * direction

    if mode == AttackerMode.SCRIPTED_CROSSING and exact_scripted_crossing:
        crossing_lower, crossing_upper = config.crossing_fraction_range
        crossing_fraction = (
            float(crossing_lower)
            if crossing_lower == crossing_upper
            else float(_uniform(crossing_key, (), crossing_lower, crossing_upper))
        )
        crossing_time = float(time[-1]) * crossing_fraction
        planar_direction = _unit_planar_vector(jr.fold_in(crossing_key, 1))
        velocity = max_speed * planar_direction
        reference_at_crossing = reference_position[0] + crossing_time * reference_velocity[0]
        initial_position = reference_at_crossing - velocity * crossing_time
        return (
            initial_position[None, :] + time[:, None] * velocity[None, :],
            np.broadcast_to(velocity, (config.steps, 3)).copy(),
            max_speed,
        )

    positions, velocities = _steered_attacker_trajectory(
        mode,
        initial_position,
        initial_velocity,
        max_speed,
        config.attacker_acceleration_limit,
        config.interceptor_prediction_horizon,
        time,
        reference_position,
        reference_velocity,
    )
    return positions, velocities, max_speed


def _steered_attacker_trajectory(
    mode: AttackerMode,
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    max_speed: float,
    acceleration_limit: float,
    prediction_horizon: float,
    time: np.ndarray,
    reference_position: np.ndarray,
    reference_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    steps = time.size
    dt = float(time[1] - time[0])
    positions = np.empty((steps, 3), dtype=np.float64)
    velocities = np.empty((steps, 3), dtype=np.float64)
    positions[0] = initial_position
    velocities[0] = initial_velocity
    if mode == AttackerMode.SCRIPTED_CROSSING:
        target = reference_position[steps // 2]
        direction = _normalized_or_zero(target - initial_position)
        constant_velocity = max_speed * direction
        positions[:] = initial_position + time[:, None] * constant_velocity
        velocities[:] = constant_velocity
        return positions, velocities

    for index in range(steps - 1):
        relative = reference_position[index] - positions[index]
        if mode == AttackerMode.BOUNDED_PURSUIT:
            aim_point = reference_position[index]
        elif mode == AttackerMode.PREDICTIVE_INTERCEPTOR:
            intercept_time = _constant_velocity_intercept_time(
                relative, reference_velocity[index], max_speed, prediction_horizon
            )
            aim_point = reference_position[index] + intercept_time * reference_velocity[index]
        else:
            raise ValueError(f"unsupported attacker mode {mode!r}")
        desired_velocity = max_speed * _normalized_or_zero(aim_point - positions[index])
        velocity_delta = desired_velocity - velocities[index]
        delta_norm = np.linalg.norm(velocity_delta)
        maximum_delta = acceleration_limit * dt
        if delta_norm > maximum_delta:
            velocity_delta *= maximum_delta / delta_norm
        velocities[index + 1] = velocities[index] + velocity_delta
        positions[index + 1] = positions[index] + 0.5 * dt * (
            velocities[index] + velocities[index + 1]
        )
    return positions, velocities


def _constant_velocity_intercept_time(
    relative_position: np.ndarray,
    target_velocity: np.ndarray,
    interceptor_speed: float,
    horizon: float,
) -> float:
    # Solve ||r + v*t||^2 = s^2*t^2.  Degenerate and unreachable cases use the
    # declared finite prediction horizon, not an unbounded extrapolation.
    a = float(np.dot(target_velocity, target_velocity) - interceptor_speed**2)
    b = float(2.0 * np.dot(relative_position, target_velocity))
    c = float(np.dot(relative_position, relative_position))
    roots: list[float] = []
    scale = max(abs(a), abs(b), abs(c), 1.0)
    if abs(a) <= 16.0 * np.finfo(np.float64).eps * scale:
        if abs(b) > 16.0 * np.finfo(np.float64).eps * scale:
            roots.append(-c / b)
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            root = math.sqrt(discriminant)
            roots.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
    positive = [value for value in roots if value >= 0.0 and math.isfinite(value)]
    return min(min(positive, default=horizon), horizon)


def _generate_dynamics_schedules(
    seed: int, fold: int, config: ScenarioTapeConfig, time: np.ndarray
) -> dict[str, np.ndarray]:
    fractions = np.asarray(
        (
            config.wind_change_fraction,
            config.mass_change_fraction,
            config.drag_change_fraction,
            config.rotor_symmetric_change_fraction,
            config.rotor_single_change_fraction,
        ),
        dtype=np.float64,
    )
    indices = np.floor(fractions * (config.steps - 1) + 0.5).astype(np.int32)

    wind_key = named_rng_key(seed, "wind_schedule", fold=fold)
    base_direction_key, base_magnitude_key, step_direction_key, step_magnitude_key, gust_key = (
        jr.split(wind_key, 5)
    )
    base = _unit_vector(base_direction_key) * float(
        _uniform(base_magnitude_key, (), 0.0, 0.2 * config.wind_speed_limit)
    )
    step = _unit_vector(step_direction_key) * float(
        _uniform(step_magnitude_key, (), 0.0, 0.4 * config.wind_speed_limit)
    )
    gust_direction_key, frequency_key, phase_key = jr.split(gust_key, 3)
    gust_direction = _unit_vector(gust_direction_key)
    frequency = float(
        _uniform(frequency_key, (), config.wind_frequency_range[0], config.wind_frequency_range[1])
    )
    phase = float(_uniform(phase_key, (), 0.0, 2.0 * math.pi))
    gust = (
        config.wind_gust_amplitude
        * np.sin(2.0 * math.pi * frequency * time + phase)[:, None]
        * gust_direction[None, :]
    )
    wind = base[None, :] + gust
    wind[indices[0] :] += step

    mass_key = named_rng_key(seed, "mass_schedule", fold=fold)
    mass_changed = float(
        _uniform(mass_key, (), config.mass_scale_bounds[0], config.mass_scale_bounds[1])
    )
    mass = np.ones(config.steps, dtype=np.float64)
    mass[indices[1] :] = mass_changed

    drag_key = named_rng_key(seed, "drag_schedule", fold=fold)
    drag_changed = _uniform(
        drag_key, (3,), config.drag_scale_bounds[0], config.drag_scale_bounds[1]
    )
    drag = np.ones((config.steps, 3), dtype=np.float64)
    drag[indices[2] :] = drag_changed

    rotor_key = named_rng_key(seed, "rotor_schedule", fold=fold)
    symmetric_key, rotor_index_key, single_key = jr.split(rotor_key, 3)
    symmetric = float(
        _uniform(
            symmetric_key, (), config.rotor_efficiency_bounds[0], config.rotor_efficiency_bounds[1]
        )
    )
    rotor_index = int(jr.randint(rotor_index_key, (), 0, 4))
    single_lower = (
        config.rotor_efficiency_bounds[0]
        if config.rotor_single_efficiency_lower is None
        else config.rotor_single_efficiency_lower
    )
    single = float(_uniform(single_key, (), single_lower, symmetric))
    rotor = np.ones((config.steps, 4), dtype=np.float64)
    rotor[indices[3] :] = symmetric
    rotor[indices[4] :, rotor_index] = single
    return {
        "wind_velocity": wind,
        "wind_speed_limit": np.asarray(config.wind_speed_limit, dtype=np.float64),
        "mass_scale": mass,
        "mass_scale_bounds": np.asarray(config.mass_scale_bounds, dtype=np.float64),
        "drag_scale": drag,
        "drag_scale_bounds": np.asarray(config.drag_scale_bounds, dtype=np.float64),
        "rotor_efficiency": rotor,
        "rotor_efficiency_bounds": np.asarray(
            (
                min(config.rotor_efficiency_bounds[0], single_lower),
                config.rotor_efficiency_bounds[1],
            ),
            dtype=np.float64,
        ),
        "rotor_single_index": np.asarray(rotor_index, dtype=np.int8),
        "schedule_names": np.asarray(SCHEDULE_NAMES),
        "schedule_change_indices": indices,
    }


def _generate_estimator_observation_noise(
    seed: int, fold: int, steps: int, *, acceleration_std: float, motor_force_std: float
) -> dict[str, np.ndarray]:
    """Generate independent, predeclared Gaussian estimator observation-noise sequences.

    Uniform PRNG bits come from two separately named streams.  A host-side Box--Muller transform
    makes the draw independent of JAX's backend-specific normal sampler while preserving a precise
    standard-deviation interpretation.  These arrays are exogenous tape inputs: a controller never
    influences them and the runtime must consume them by time index without resampling.
    """
    acceleration = acceleration_std * _standard_normal_from_bits(
        named_rng_key(seed, "estimator_acceleration_noise", fold=fold), (steps, 3)
    )
    motor_force = motor_force_std * _standard_normal_from_bits(
        named_rng_key(seed, "estimator_motor_force_noise", fold=fold), (steps, 4)
    )
    return {
        "estimator_acceleration_noise": np.asarray(acceleration, dtype=np.float64),
        "estimator_acceleration_noise_std": np.asarray(acceleration_std, dtype=np.float64),
        "estimator_motor_force_noise": np.asarray(motor_force, dtype=np.float64),
        "estimator_motor_force_noise_std": np.asarray(motor_force_std, dtype=np.float64),
    }


def _standard_normal_from_bits(key: Array, shape: tuple[int, ...]) -> np.ndarray:
    """Return a deterministic standard-normal array from raw uint32 PRNG words."""
    count = math.prod(shape)
    pair_count = (count + 1) // 2
    words = np.asarray(jr.bits(key, (pair_count, 2), dtype=jnp.uint32), dtype=np.uint32)
    unit = (words.astype(np.float64) + 0.5) / float(2**32)
    radius = np.sqrt(-2.0 * np.log(unit[:, 0]))
    angle = 2.0 * math.pi * unit[:, 1]
    samples = np.empty(2 * pair_count, dtype=np.float64)
    samples[0::2] = radius * np.cos(angle)
    samples[1::2] = radius * np.sin(angle)
    return samples[:count].reshape(shape)


def _uniform(key: Array, shape: tuple[int, ...], lower: Any, upper: Any) -> np.ndarray:
    unit = np.asarray(jr.uniform(key, shape, dtype=jnp.float32), dtype=np.float64)
    return np.asarray(lower, dtype=np.float64) + unit * (
        np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)
    )


def _unit_vector(key: Array) -> np.ndarray:
    # Uniform PRNG bits are backend-identical; transcendental work happens once on the host.  This
    # avoids the backend-specific approximation differences in ``jax.random.normal`` while still
    # sampling the sphere uniformly.
    unit = _uniform(key, (2,), 0.0, 1.0)
    vertical = 2.0 * unit[0] - 1.0
    azimuth = 2.0 * math.pi * unit[1]
    planar = math.sqrt(max(0.0, 1.0 - vertical * vertical))
    return np.asarray(
        (planar * math.cos(azimuth), planar * math.sin(azimuth), vertical), dtype=np.float64
    )


def _unit_planar_vector(key: Array) -> np.ndarray:
    angle = float(_uniform(key, (), 0.0, 2.0 * math.pi))
    return np.asarray((math.cos(angle), math.sin(angle), 0.0), dtype=np.float64)


def _normalized_or_zero(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return np.zeros_like(vector) if norm == 0.0 else vector / norm


def _scenario_tape_arrays(tape: ScenarioTape) -> dict[str, np.ndarray]:
    return {item.name: getattr(tape, item.name) for item in fields(tape)}


def _canonical_tape_digest(tape: ScenarioTape) -> str:
    digest = hashlib.sha256(b"crazyflow.da_plcbf.scenario_tape.v1\0")
    for name, value in sorted(_scenario_tape_arrays(tape).items()):
        name_bytes = name.encode("utf-8")
        digest.update(struct.pack("<I", len(name_bytes)))
        digest.update(name_bytes)
        _update_canonical_array_digest(digest, value)
    return digest.hexdigest()


def _update_canonical_array_digest(digest: Any, value: np.ndarray) -> None:
    """Hash an array with explicit dtype, shape, byte order, and string encoding."""
    array = np.asarray(value)
    digest.update(struct.pack("<I", array.ndim))
    for extent in array.shape:
        digest.update(struct.pack("<Q", extent))
    if array.dtype.kind in "SU":
        dtype_bytes = array.dtype.newbyteorder("<").str.encode("ascii")
        digest.update(struct.pack("<I", len(dtype_bytes)))
        digest.update(dtype_bytes)
        flattened = array.ravel(order="C")
        digest.update(struct.pack("<Q", flattened.size))
        for item in flattened:
            encoded = str(item).encode("utf-8")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
        return
    canonical_dtype = array.dtype.newbyteorder("<")
    dtype_bytes = canonical_dtype.str.encode("ascii")
    digest.update(struct.pack("<I", len(dtype_bytes)))
    digest.update(dtype_bytes)
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    raw = canonical.tobytes(order="C")
    digest.update(struct.pack("<Q", len(raw)))
    digest.update(raw)


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, value in sorted(arrays.items()):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(
                info, _npy_bytes(value), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )
    return output.getvalue()


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _constant_time_digest_equal(left: str, right: str) -> bool:
    import hmac

    return hmac.compare_digest(left, right)


def _frozen_array(value: Any) -> np.ndarray:
    array = np.array(value, copy=True, order="C")
    if array.dtype.hasobject or array.dtype.kind not in "biufcSU":
        raise ValueError(
            "scenario tape supports only boolean, numeric, and fixed-width string arrays"
        )
    array.setflags(write=False)
    return array


def _validate_uint32(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if not 0 <= converted <= np.iinfo(np.uint32).max:
        raise ValueError(f"{name} must fit in uint32")
    return converted


def _validate_positive_integer(value: int, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _validate_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _finite_scalar(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


def _validate_finite_positive(value: Real, name: str) -> None:
    if _finite_scalar(value, name) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _finite_vector(value: Sequence[Real], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    return array


def _finite_interval(value: Sequence[Real], name: str, *, positive: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)) or array[1] < array[0]:
        raise ValueError(f"{name} must be a finite ordered pair")
    if positive and array[0] <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return array


def _validate_scale_bounds(value: Sequence[Real], name: str) -> np.ndarray:
    bounds = _finite_interval(value, name, positive=True)
    if not bounds[0] <= 1.0 <= bounds[1]:
        raise ValueError(f"{name} must contain the nominal scale one")
    return bounds


def _require_scalar(value: np.ndarray, dtype: Any, name: str) -> None:
    _require_dtype_shape(value, dtype, (), name)


def _require_dtype(value: np.ndarray, dtype: Any, name: str) -> None:
    if value.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must have dtype {np.dtype(dtype)}")


def _require_dtype_shape(value: np.ndarray, dtype: Any, shape: tuple[int, ...], name: str) -> None:
    _require_dtype(value, dtype, name)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")


def _require_float_shape(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    _require_dtype_shape(value, np.float64, shape, name)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")


def _require_bool_shape(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    _require_dtype_shape(value, np.bool_, shape, name)


def _is_prefix_mask(mask: np.ndarray) -> bool:
    return not np.any(mask[1:] > mask[:-1])
