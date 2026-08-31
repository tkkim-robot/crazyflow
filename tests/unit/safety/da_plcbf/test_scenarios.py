from __future__ import annotations

import io
import math
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from crazyflow.safety.da_plcbf.scenarios import (
    BALLISTIC_ENCOUNTER_STRATUM_CYCLE,
    RNG_STREAM_IDS,
    RNG_STREAM_NAMES,
    SCENARIO_TAPE_SCHEMA_VERSION,
    AttackerMode,
    DynamicObstacleKind,
    ScenarioTape,
    ScenarioTapeConfig,
    generate_scenario_tape,
    hard_contact_labels,
    load_scenario_tape,
    named_rng_key,
    save_scenario_tape,
)

if TYPE_CHECKING:
    from pathlib import Path


def _config(**changes: Any) -> ScenarioTapeConfig:
    values: dict[str, Any] = {
        "steps": 61,
        "dt": 0.05,
        "prediction_samples": 5,
        "static_capacity": 5,
        "static_count": 3,
        "dynamic_capacity": 8,
        "ballistic_count": 2,
        "crossing_count": 1,
        "pursuit_count": 1,
        "interceptor_count": 1,
        "random_attacker_count": 2,
    }
    values.update(changes)
    return ScenarioTapeConfig(**values)


def _tape(seed: int = 17, **changes: Any) -> ScenarioTape:
    return generate_scenario_tape(seed, _config(**changes), fold=9)


def _archive_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def test_named_rng_streams_are_stable_explicit_and_folded() -> None:
    assert tuple(RNG_STREAM_IDS) == RNG_STREAM_NAMES
    assert len(set(RNG_STREAM_IDS.values())) == len(RNG_STREAM_NAMES)
    first = np.asarray(named_rng_key(123, "ballistic_truth", fold=4))
    repeated = np.asarray(named_rng_key(123, "ballistic_truth", fold=4))
    other_name = np.asarray(named_rng_key(123, "wind_schedule", fold=4))
    other_fold = np.asarray(named_rng_key(123, "ballistic_truth", fold=5))
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other_name)
    assert not np.array_equal(first, other_fold)

    with pytest.raises(ValueError, match="unknown RNG stream"):
        named_rng_key(123, "implicit-global-stream")
    with pytest.raises((TypeError, ValueError)):
        named_rng_key(-1, "ballistic_truth")


def test_same_seed_and_fold_are_bit_identical_and_different_inputs_vary() -> None:
    first = _tape(41)
    repeated = _tape(41)
    for item in fields(first):
        np.testing.assert_array_equal(getattr(first, item.name), getattr(repeated, item.name))
    assert first.sha256 == repeated.sha256

    different_seed = _tape(42)
    different_fold = generate_scenario_tape(41, _config(), fold=10)
    assert first.sha256 != different_seed.sha256
    assert first.sha256 != different_fold.sha256
    assert not np.array_equal(first.static_positions, different_seed.static_positions)
    assert not np.array_equal(first.wind_velocity, different_fold.wind_velocity)


def test_actual_initial_state_and_task_reference_are_independent_and_recorded() -> None:
    config = _config(
        vehicle_initial_position=(0.8, -0.4, 1.2),
        vehicle_initial_velocity=(-0.2, 0.3, 0.1),
        reference_initial_position=(0.0, 0.0, 1.5),
        reference_initial_velocity=(0.45, 0.10, 0.0),
    )
    tape = generate_scenario_tape(41, config, fold=10)

    np.testing.assert_array_equal(tape.vehicle_initial_position, config.vehicle_initial_position)
    np.testing.assert_array_equal(tape.vehicle_initial_velocity, config.vehicle_initial_velocity)
    expected_reference = (
        np.asarray(config.reference_initial_position)[None, :]
        + tape.time[:, None] * np.asarray(config.reference_initial_velocity)[None, :]
    )
    np.testing.assert_allclose(tape.defender_reference_position, expected_reference, atol=1e-14)
    np.testing.assert_array_equal(
        tape.defender_reference_velocity,
        np.broadcast_to(config.reference_initial_velocity, (config.steps, 3)),
    )
    assert not np.array_equal(tape.vehicle_initial_position, tape.defender_reference_position[0])
    assert not np.array_equal(tape.vehicle_initial_velocity, tape.defender_reference_velocity[0])


def test_tape_has_fixed_shapes_prefix_masks_and_immutable_storage() -> None:
    tape = _tape()
    assert tape.time.shape == (61,)
    assert tape.static_positions.shape == (5, 3)
    assert tape.dynamic_positions.shape == (61, 8, 3)
    assert tape.prediction_positions.shape == (5, 61, 8, 3)
    assert tape.estimator_acceleration_noise.shape == (61, 3)
    assert tape.estimator_motor_force_noise.shape == (61, 4)
    np.testing.assert_array_equal(tape.static_mask, [True, True, True, False, False])
    np.testing.assert_array_equal(
        tape.dynamic_slot_mask, [True, True, True, True, True, True, True, False]
    )
    assert np.all(tape.static_positions[~tape.static_mask] == 0.0)
    assert np.all(tape.dynamic_positions[:, ~tape.dynamic_slot_mask] == 0.0)
    assert np.all(tape.prediction_positions[:, :, ~tape.dynamic_slot_mask] == 0.0)
    for item in fields(tape):
        assert not getattr(tape, item.name).flags.writeable
    with pytest.raises(ValueError):
        tape.wind_velocity[0, 0] = 99.0

    changed_config_metadata = replace(tape, generator_config_sha256=np.asarray("f" * 64))
    assert changed_config_metadata.sha256 != tape.sha256
    with pytest.raises(ValueError, match="named RNG streams"):
        replace(tape, root_seed=np.asarray(18, dtype=np.uint32))


def test_ballistic_truth_and_uncertainty_are_analytic_without_floor_clipping() -> None:
    tape = _tape()
    slots = np.flatnonzero(tape.dynamic_kind == int(DynamicObstacleKind.BALLISTIC))
    assert slots.size == 2
    for slot in slots:
        release = int(tape.ballistic_release_index[slot])
        tau = tape.time[release:] - tape.time[release]
        expected_position = (
            tape.ballistic_release_position[slot]
            + tau[:, None] * tape.ballistic_release_velocity[slot]
            + 0.5 * tau[:, None] ** 2 * tape.gravity
        )
        expected_velocity = tape.ballistic_release_velocity[slot] + tau[:, None] * tape.gravity
        np.testing.assert_allclose(tape.dynamic_positions[release:, slot], expected_position)
        np.testing.assert_allclose(tape.dynamic_velocities[release:, slot], expected_velocity)
        assert not np.any(tape.dynamic_time_mask[:release, slot])
        assert np.all(tape.dynamic_time_mask[release:, slot])
        offsets = (
            tape.ballistic_prediction_release_velocity[:, slot]
            - tape.ballistic_release_velocity[slot]
        )
        np.testing.assert_array_equal(offsets[0], np.zeros(3))
        assert np.all(np.abs(offsets) <= tape.ballistic_velocity_uncertainty + 1e-12)
        assert np.all(
            tape.ballistic_prediction_release_velocity[:, slot] >= tape.ballistic_velocity_lower
        )
        assert np.all(
            tape.ballistic_prediction_release_velocity[:, slot] <= tape.ballistic_velocity_upper
        )
        assert tape.ballistic_generation_attempts[slot] >= 1
        assert tape.ballistic_target_time[slot] == pytest.approx(
            tape.time[release] + tape.ballistic_time_to_impact[slot]
        )
        assert tape.ballistic_realized_closest_time[slot] == pytest.approx(
            tape.ballistic_target_time[slot], abs=2e-9
        )
        assert tape.ballistic_realized_closest_distance[slot] == pytest.approx(
            tape.ballistic_intended_miss_distance[slot], abs=2e-9
        )

    # At least one ball has passed below the arena floor.  Its analytic free-flight state is kept;
    # the scenario generator does not clip it to z=0 or introduce a contact bounce.
    assert np.min(tape.dynamic_positions[:, slots, 2]) < tape.arena_lower[2] - 1.0


def test_ballistic_encounters_have_predeclared_fold_strata_over_one_hundred_folds() -> None:
    config = ScenarioTapeConfig(
        steps=202,
        dt=0.02,
        prediction_samples=4,
        static_capacity=4,
        static_count=0,
        dynamic_capacity=4,
        ballistic_count=2,
        crossing_count=0,
        pursuit_count=0,
        interceptor_count=0,
        random_attacker_count=0,
    )
    strata: list[int] = []
    impact_bins: list[int] = []
    physical_intersection_by_fold: list[bool] = []
    for fold in range(100):
        tape = generate_scenario_tape(12345, config, fold=fold)
        ballistic = tape.ballistic_encounter_stratum >= 0
        expected_stratum = BALLISTIC_ENCOUNTER_STRATUM_CYCLE[
            fold % len(BALLISTIC_ENCOUNTER_STRATUM_CYCLE)
        ]
        np.testing.assert_array_equal(tape.ballistic_encounter_stratum[ballistic], expected_stratum)
        strata.extend(tape.ballistic_encounter_stratum[ballistic].tolist())
        impact_bins.extend(tape.ballistic_time_to_impact_bin[ballistic].tolist())
        physical_radius = tape.vehicle_radius + tape.dynamic_radii[ballistic]
        physical_intersection_by_fold.append(
            bool(np.any(tape.ballistic_realized_closest_distance[ballistic] < physical_radius))
        )
        np.testing.assert_allclose(
            tape.ballistic_realized_closest_distance[ballistic],
            tape.ballistic_intended_miss_distance[ballistic],
            rtol=0.0,
            atol=2e-9,
        )

    # Two balls per fold preserve the predeclared 40/40/20 trial strata exactly.  Time bins are
    # cycled independently and differ by at most one sample, without looking at controller output.
    np.testing.assert_array_equal(np.bincount(strata, minlength=3), [80, 80, 40])
    np.testing.assert_array_equal(np.bincount(impact_bins, minlength=3), [67, 67, 66])
    assert sum(physical_intersection_by_fold) == 40


@pytest.mark.parametrize("steps", [2, 3, 4, 5, 6, 7, 8, 11])
@pytest.mark.parametrize(("seed", "fold"), [(913, 8), (73, 4), (0, 0), (41, 9)])
def test_ballistic_release_range_is_valid_on_short_time_grids(
    steps: int, seed: int, fold: int
) -> None:
    config = ScenarioTapeConfig(steps=steps, dt=0.1)
    tape = generate_scenario_tape(seed, config, fold=fold)
    ballistic = tape.dynamic_kind == int(DynamicObstacleKind.BALLISTIC)
    minimum = math.floor(config.ball_release_fraction_range[0] * (steps - 1))
    maximum = math.floor(config.ball_release_fraction_range[1] * (steps - 1))
    assert np.all(tape.ballistic_release_index[ballistic] >= minimum)
    assert np.all(tape.ballistic_release_index[ballistic] <= maximum)
    assert np.all(tape.ballistic_release_index[ballistic] < steps - 1)


def test_ballistic_encounter_metadata_is_schema_bound_and_tamper_evident() -> None:
    tape = _tape()
    assert int(tape.schema_version) == SCENARIO_TAPE_SCHEMA_VERSION == 3
    slot = int(np.flatnonzero(tape.dynamic_kind == int(DynamicObstacleKind.BALLISTIC))[0])

    bad_distance = np.array(tape.ballistic_realized_closest_distance, copy=True)
    bad_distance[slot] += 0.01
    with pytest.raises(ValueError, match="realized closest approach"):
        replace(tape, ballistic_realized_closest_distance=bad_distance)

    with pytest.raises(ValueError, match="unsupported scenario-tape schema"):
        replace(tape, schema_version=np.asarray(2, dtype=np.uint16))


def test_estimator_noise_is_predeclared_independent_and_tamper_evident() -> None:
    tape = _tape(73)
    repeated = _tape(73)
    other_fold = generate_scenario_tape(73, _config(), fold=10)

    assert "estimator_acceleration_noise" in RNG_STREAM_NAMES
    assert "estimator_motor_force_noise" in RNG_STREAM_NAMES
    assert tape.estimator_acceleration_noise_std == pytest.approx(0.03)
    assert tape.estimator_motor_force_noise_std == pytest.approx(5e-4)
    np.testing.assert_array_equal(
        tape.estimator_acceleration_noise, repeated.estimator_acceleration_noise
    )
    np.testing.assert_array_equal(
        tape.estimator_motor_force_noise, repeated.estimator_motor_force_noise
    )
    assert not np.array_equal(
        tape.estimator_acceleration_noise, other_fold.estimator_acceleration_noise
    )
    assert not np.array_equal(
        tape.estimator_motor_force_noise, other_fold.estimator_motor_force_noise
    )
    assert not np.array_equal(
        tape.estimator_acceleration_noise[:, :3], tape.estimator_motor_force_noise[:, :3]
    )

    corrupted = np.array(tape.estimator_acceleration_noise, copy=True)
    corrupted[4, 1] += 1e-3
    with pytest.raises(ValueError, match="named RNG streams"):
        replace(tape, estimator_acceleration_noise=corrupted)

    zero_noise = _tape(
        73, estimator_acceleration_noise_std=0.0, estimator_motor_force_noise_std=0.0
    )
    np.testing.assert_array_equal(zero_noise.estimator_acceleration_noise, 0.0)
    np.testing.assert_array_equal(zero_noise.estimator_motor_force_noise, 0.0)


def test_scripted_crossing_is_constant_velocity_and_crosses_reference() -> None:
    crossing_fraction = 0.37
    tape = _tape(
        crossing_fraction_range=(crossing_fraction, crossing_fraction),
        vehicle_initial_position=(0.4, -0.3, 1.2),
        vehicle_initial_velocity=(-0.1, 0.2, 0.0),
        reference_initial_position=(0.0, 0.0, 1.5),
        reference_initial_velocity=(0.45, 0.10, 0.0),
    )
    dedicated = (
        tape.attacker_mode == int(AttackerMode.SCRIPTED_CROSSING)
    ) & ~tape.randomized_attacker
    slots = np.flatnonzero(dedicated)
    assert slots.size == 1
    slot = int(slots[0])
    np.testing.assert_allclose(
        tape.dynamic_velocities[:, slot],
        np.broadcast_to(tape.dynamic_velocities[0, slot], (tape.steps, 3)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        tape.dynamic_positions[:, slot],
        tape.dynamic_positions[0, slot] + tape.time[:, None] * tape.dynamic_velocities[0, slot],
    )
    relative_position = tape.dynamic_positions[0, slot] - tape.defender_reference_position[0]
    relative_velocity = tape.dynamic_velocities[0, slot] - tape.defender_reference_velocity[0]
    crossing_time = -float(np.dot(relative_position, relative_velocity)) / float(
        np.dot(relative_velocity, relative_velocity)
    )
    assert crossing_time == pytest.approx(crossing_fraction * tape.time[-1], abs=2e-12)
    np.testing.assert_allclose(
        relative_position + crossing_time * relative_velocity, np.zeros(3), atol=2e-12
    )


@pytest.mark.parametrize(
    "mode", [AttackerMode.BOUNDED_PURSUIT, AttackerMode.PREDICTIVE_INTERCEPTOR]
)
def test_pursuit_and_interceptor_obey_speed_and_acceleration_bounds(mode: AttackerMode) -> None:
    tape = _tape()
    slots = np.flatnonzero((tape.attacker_mode == int(mode)) & ~tape.randomized_attacker)
    assert slots.size == 1
    slot = int(slots[0])
    speed = np.linalg.norm(tape.dynamic_velocities[:, slot], axis=-1)
    acceleration = np.linalg.norm(
        np.diff(tape.dynamic_velocities[:, slot], axis=0) / np.diff(tape.time)[:, None], axis=-1
    )
    assert np.max(speed) <= tape.dynamic_speed_limit[slot] + 1e-12
    assert np.max(acceleration) <= tape.dynamic_acceleration_limit[slot] + 1e-12

    if mode == AttackerMode.BOUNDED_PURSUIT:
        desired_direction = tape.defender_reference_position[0] - tape.dynamic_positions[0, slot]
        desired_direction /= np.linalg.norm(desired_direction)
        desired_velocity = tape.dynamic_speed_limit[slot] * desired_direction
        delta = desired_velocity - tape.dynamic_velocities[0, slot]
        maximum_delta = tape.dynamic_acceleration_limit[slot] * (tape.time[1] - tape.time[0])
        if np.linalg.norm(delta) > maximum_delta:
            delta *= maximum_delta / np.linalg.norm(delta)
        np.testing.assert_allclose(
            tape.dynamic_velocities[1, slot], tape.dynamic_velocities[0, slot] + delta
        )
        np.testing.assert_allclose(
            tape.dynamic_positions[1, slot],
            tape.dynamic_positions[0, slot]
            + 0.5
            * (tape.time[1] - tape.time[0])
            * (tape.dynamic_velocities[0, slot] + tape.dynamic_velocities[1, slot]),
        )


def test_random_attackers_are_finite_fixed_trajectory_scenarios() -> None:
    tape = _tape(31)
    random_slots = np.flatnonzero(tape.randomized_attacker)
    assert random_slots.size == 2
    valid_modes = {
        int(AttackerMode.SCRIPTED_CROSSING),
        int(AttackerMode.BOUNDED_PURSUIT),
        int(AttackerMode.PREDICTIVE_INTERCEPTOR),
    }
    assert set(tape.attacker_mode[random_slots]).issubset(valid_modes)
    assert set(tape.prediction_attacker_mode[:, random_slots].ravel()).issubset(valid_modes)
    assert len(set(tape.prediction_attacker_mode[:, random_slots].ravel())) >= 2
    # A mode is one scalar per complete trajectory, not a time-indexed transition signal.
    assert tape.prediction_attacker_mode.shape == (tape.prediction_samples, 8)
    assert np.all(np.isfinite(tape.prediction_positions[:, :, random_slots]))


def test_dynamics_schedules_are_bounded_and_have_declared_changes() -> None:
    tape = _tape()
    wind_index, mass_index, drag_index, symmetric_index, single_index = (
        int(value) for value in tape.schedule_change_indices
    )
    assert wind_index <= mass_index <= drag_index <= symmetric_index <= single_index
    assert np.max(np.linalg.norm(tape.wind_velocity, axis=-1)) <= tape.wind_speed_limit + 1e-12
    np.testing.assert_array_equal(tape.mass_scale[:mass_index], 1.0)
    np.testing.assert_array_equal(tape.mass_scale[mass_index:], tape.mass_scale[mass_index])
    np.testing.assert_array_equal(tape.drag_scale[:drag_index], 1.0)
    np.testing.assert_array_equal(
        tape.drag_scale[drag_index:],
        np.broadcast_to(tape.drag_scale[drag_index], tape.drag_scale[drag_index:].shape),
    )
    np.testing.assert_array_equal(tape.rotor_efficiency[:symmetric_index], 1.0)
    rotor = int(tape.rotor_single_index)
    others = np.arange(4) != rotor
    symmetric = tape.rotor_efficiency[symmetric_index, np.flatnonzero(others)[0]]
    np.testing.assert_array_equal(tape.rotor_efficiency[symmetric_index:, others], symmetric)
    assert tape.rotor_efficiency[single_index, rotor] <= symmetric
    assert np.all(tape.mass_scale > 0.0)
    assert np.all(tape.drag_scale > 0.0)
    assert np.all(tape.rotor_efficiency > 0.0)


def test_symmetric_and_single_rotor_lower_efficiencies_have_independent_effects() -> None:
    base = _tape(31, rotor_efficiency_bounds=(0.75, 1.0), rotor_single_efficiency_lower=0.50)
    changed_symmetric = _tape(
        31, rotor_efficiency_bounds=(0.90, 1.0), rotor_single_efficiency_lower=0.50
    )
    changed_single = _tape(
        31, rotor_efficiency_bounds=(0.75, 1.0), rotor_single_efficiency_lower=0.35
    )
    rotor = int(base.rotor_single_index)
    assert (
        rotor == int(changed_symmetric.rotor_single_index) == int(changed_single.rotor_single_index)
    )
    others = np.arange(4) != rotor
    symmetric_index = int(base.schedule_change_indices[3])
    single_index = int(base.schedule_change_indices[4])

    assert not np.array_equal(
        base.rotor_efficiency[symmetric_index:, others],
        changed_symmetric.rotor_efficiency[symmetric_index:, others],
    )
    np.testing.assert_array_equal(
        base.rotor_efficiency[:, others], changed_single.rotor_efficiency[:, others]
    )
    np.testing.assert_array_equal(
        base.rotor_efficiency[:single_index, rotor],
        changed_single.rotor_efficiency[:single_index, rotor],
    )
    assert not np.array_equal(
        base.rotor_efficiency[single_index:, rotor],
        changed_single.rotor_efficiency[single_index:, rotor],
    )
    np.testing.assert_array_equal(base.rotor_efficiency_bounds, (0.50, 1.0))
    np.testing.assert_array_equal(changed_single.rotor_efficiency_bounds, (0.35, 1.0))


def test_contacts_are_hard_labels_and_do_not_modify_trajectories() -> None:
    tape = _tape()
    ball = int(np.flatnonzero(tape.dynamic_kind == int(DynamicObstacleKind.BALLISTIC))[0])
    positions = np.array(tape.dynamic_positions[:, ball], copy=True)
    positions_before = positions.copy()
    digest_before = tape.sha256
    labels = hard_contact_labels(positions, tape)
    release = int(tape.ballistic_release_index[ball])
    assert np.all(labels.dynamic[release:, ball])
    assert np.all(labels.any_contact[release:])
    np.testing.assert_array_equal(positions, positions_before)
    assert tape.sha256 == digest_before
    assert bool(tape.contact_is_failure)
    for value in (labels.static, labels.dynamic, labels.any_contact):
        assert value.dtype == np.bool_
        assert not value.flags.writeable

    with pytest.raises(ValueError, match="finite"):
        hard_contact_labels(np.full((tape.steps, 3), np.nan), tape)


def test_npz_round_trip_has_stable_hash_and_deterministic_bytes(tmp_path: Path) -> None:
    tape = _tape()
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    assert save_scenario_tape(tape, first_path) == tape.sha256
    assert save_scenario_tape(tape, second_path) == tape.sha256
    assert first_path.read_bytes() == second_path.read_bytes()

    restored = load_scenario_tape(first_path)
    assert restored.sha256 == tape.sha256
    for item in fields(tape):
        np.testing.assert_array_equal(getattr(restored, item.name), getattr(tape, item.name))
        assert not getattr(restored, item.name).flags.writeable

    with pytest.raises(FileExistsError):
        save_scenario_tape(tape, first_path)
    save_scenario_tape(tape, first_path, overwrite=True)
    assert load_scenario_tape(first_path).sha256 == tape.sha256


def test_npz_rejects_digest_tampering_nonfinite_members_and_bad_container(tmp_path: Path) -> None:
    tape = _tape()
    valid_path = tmp_path / "valid.npz"
    save_scenario_tape(tape, valid_path)

    digest_path = tmp_path / "bad_digest.npz"
    digest_payload = _archive_payload(valid_path)
    digest_payload["content_sha256"] = np.asarray("0" * 64)
    np.savez(digest_path, **digest_payload)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_scenario_tape(digest_path)

    nonfinite_path = tmp_path / "nonfinite.npz"
    nonfinite_payload = _archive_payload(valid_path)
    nonfinite_payload["wind_velocity"][0, 0] = np.nan
    np.savez(nonfinite_path, **nonfinite_payload)
    with pytest.raises(ValueError, match="schema validation"):
        load_scenario_tape(nonfinite_path)

    missing_path = tmp_path / "missing.npz"
    np.savez(missing_path, time=tape.time)
    with pytest.raises(ValueError, match="missing, duplicate, or unexpected"):
        load_scenario_tape(missing_path)

    broken_path = tmp_path / "broken.npz"
    broken_path.write_bytes(io.BytesIO(b"not-a-zip").getvalue())
    with pytest.raises(ValueError, match="not a valid NPZ"):
        load_scenario_tape(broken_path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"dt": math.nan}, "dt"),
        ({"static_count": 6}, "static_count"),
        ({"dynamic_capacity": 2}, "dynamic obstacles"),
        ({"ball_velocity_uncertainty": (3.0, 0.1, 0.1)}, "uncertainty"),
        (
            {"ballistic_time_to_impact_fraction_bins": ((0.2, 0.5), (0.4, 0.7), (0.8, 0.9))},
            "time-to-impact bins",
        ),
        ({"ballistic_generation_max_attempts": 0}, "ballistic_generation_max_attempts"),
        ({"wind_gust_amplitude": 10.0}, "wind_gust_amplitude"),
        ({"estimator_acceleration_noise_std": -0.1}, "must be nonnegative"),
        ({"estimator_motor_force_noise_std": math.inf}, "must be a finite"),
        ({"rotor_efficiency_bounds": (0.5, 1.2)}, "must not exceed 1"),
        ({"rotor_single_efficiency_lower": 0.0}, "rotor_single_efficiency_lower"),
        ({"rotor_single_efficiency_lower": 0.8}, "rotor_single_efficiency_lower"),
        ({"mass_change_fraction": 0.1}, "nondecreasing"),
    ],
)
def test_invalid_generator_config_is_rejected(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_scenario_tape(1, _config(**change))


def test_nonfinite_corrupt_and_wrong_shape_tapes_are_rejected() -> None:
    tape = _tape()
    bad_wind = np.array(tape.wind_velocity, copy=True)
    bad_wind[3, 1] = np.inf
    with pytest.raises(ValueError, match="wind_velocity"):
        replace(tape, wind_velocity=bad_wind)

    bad_ball = np.array(tape.dynamic_positions, copy=True)
    bad_ball[10, 0, 0] += 0.1
    with pytest.raises(ValueError, match="analytic free flight"):
        replace(tape, dynamic_positions=bad_ball)

    with pytest.raises(ValueError, match="static_positions"):
        replace(tape, static_positions=np.zeros(3))


def test_zero_real_obstacles_still_produces_a_valid_fixed_capacity_tape() -> None:
    tape = _tape(
        static_count=0,
        ballistic_count=0,
        crossing_count=0,
        pursuit_count=0,
        interceptor_count=0,
        random_attacker_count=0,
    )
    assert not np.any(tape.static_mask)
    assert not np.any(tape.dynamic_slot_mask)
    assert not np.any(tape.dynamic_time_mask)
    assert np.all(tape.static_positions == 0.0)
    assert np.all(tape.dynamic_positions == 0.0)
    assert np.all(tape.ballistic_encounter_stratum == -1)
    assert np.all(tape.ballistic_time_to_impact_bin == -1)
    assert np.all(tape.ballistic_realized_closest_distance == 0.0)
    assert np.all(tape.ballistic_generation_attempts == 0)
    tape.validate()
