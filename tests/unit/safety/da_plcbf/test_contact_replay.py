"""Free-body transfer, swept triggering, and genuine obstacle/ground contact response."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from crazyflow.safety.da_plcbf.contact_replay import (
    ContactReplayConfig,
    ContactTrigger,
    ObstacleMotion,
    build_contact_model,
    cf21b_contact_body,
    find_contact_trigger,
    run_contact_replay,
    save_contact_replay,
)

if TYPE_CHECKING:
    from pathlib import Path


def _obstacle() -> ObstacleMotion:
    return ObstacleMotion(
        np.asarray([0.0, 10.0]),
        np.asarray([[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]]),
        np.asarray([0.22]),
    )


def _state() -> np.ndarray:
    return np.asarray([-0.8, 0, 1.2, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0.0])


def test_contact_model_preserves_point_inertia_and_existing_offset_collider() -> None:
    body = cf21b_contact_body()
    model, _ = build_contact_model(body, _obstacle(), ContactReplayConfig())
    drone = model.body("drone").id
    rotation = Rotation.from_quat(model.body_iquat[drone][[1, 2, 3, 0]]).as_matrix()
    inertia = rotation @ np.diag(model.body_inertia[drone]) @ rotation.T
    np.testing.assert_allclose(inertia, np.diag([25e-6, 28e-6, 49e-6]), atol=1e-15)
    assert model.body_mass[drone] == 0.04338
    np.testing.assert_array_equal(model.body_ipos[drone], [0, 0, 0])
    collider = model.geom("drone_collider")
    np.testing.assert_array_equal(collider.pos, [0, 0, 0.02])
    assert collider.size[0] == 0.086
    assert model.nq == 14 and model.nv == 12 and model.nu == 0 and model.nmocap == 0


def test_uncontacted_motion_is_integrated_free_fall_after_exact_state_transfer() -> None:
    body = replace(cf21b_contact_body(), drag_matrix_body=np.zeros((3, 3)))
    state = _state()
    state[3:7] = Rotation.from_euler("xyz", [0.4, -0.3, 0.2]).as_quat()
    state[10:13] = [0.1, -0.2, 0.3]
    obstacles = ObstacleMotion(np.asarray([0.0, 1.0]), np.empty((2, 0, 3)), np.empty(0))
    replay = run_contact_replay(
        ContactTrigger(0, state, "fixture_motor_cut", 0, None),
        body,
        obstacles,
        ContactReplayConfig(duration_seconds=0.05),
    )
    np.testing.assert_array_equal(replay.full_state[0], state)
    np.testing.assert_allclose(
        replay.full_state[-1, :3],
        state[:3] + state[7:10] * 0.05 + 0.5 * body.gravity * 0.05**2,
        atol=3e-4,
    )
    np.testing.assert_allclose(
        replay.full_state[-1, 7:10], state[7:10] + body.gravity * 0.05, atol=1e-10
    )
    assert not replay.ground_contact.any() and not replay.obstacle_contact.any()


def test_real_contact_deflects_body_then_ground_arrests_fall(tmp_path: Path) -> None:
    state = _state()
    replay = run_contact_replay(
        ContactTrigger(0, state, "fixture_motor_cut", 0, None), cf21b_contact_body(), _obstacle()
    )
    obstacle_time = next(
        event["time_seconds"] for event in replay.events if event["kind"] == "obstacle_contact"
    )
    ground_time = next(
        event["time_seconds"] for event in replay.events if event["kind"] == "ground_contact"
    )
    assert 0 < obstacle_time < ground_time < 1
    assert replay.contact_force_norm_N.max() > 1
    assert replay.full_state[-1, 0] < 0  # Contact deflects the initially positive x velocity.
    assert abs(replay.full_state[-1, 9]) < 0.01
    center = replay.full_state[-1, :3] + Rotation.from_quat(replay.full_state[-1, 3:7]).apply(
        [0, 0, 0.02]
    )
    assert abs(center[2] - 0.086) < 1e-4
    assert replay.metadata["minimum_contact_distance_m"] > -0.012
    assert replay.metadata["warning_counts"] == {}
    output = save_contact_replay(replay, tmp_path / "contact")
    saved = json.loads((output / "contact_replay.json").read_text())
    assert saved["contact_events"][0]["kind"] == "obstacle_contact"
    with np.load(output / "contact_replay.npz") as archive:
        np.testing.assert_array_equal(archive["full_state"], replay.full_state)
    with pytest.raises(FileExistsError):
        save_contact_replay(replay, output)


def test_swept_contact_starts_at_surface_and_shell_abort_is_distinct() -> None:
    times = np.arange(16) * 0.02
    states = np.tile(_state(), (len(times), 1))
    states[:, 0] = -0.8 + 4 * times
    states[:, 2], states[:, 7] = 1.0, 4.0
    physical = find_contact_trigger(times, states, _obstacle(), ContactReplayConfig())
    expected = (0.8 - np.sqrt((0.086 + 0.22) ** 2 - 0.02**2)) / 4
    assert physical.time_seconds == pytest.approx(expected, abs=1e-12)
    center = physical.full_state[:3] + [0, 0, 0.02]
    assert np.linalg.norm(center - [0, 0, 1]) - 0.306 == pytest.approx(0, abs=1e-12)
    shell = find_contact_trigger(
        times,
        states,
        _obstacle(),
        ContactReplayConfig(),
        kind="unsafe_shell",
        shell_ego_radius=0.05,
        shell_clearance=0.15,
    )
    assert shell.kind == "safety_abort_unsafe_shell"
    assert shell.time_seconds < physical.time_seconds
    assert shell.time_seconds == pytest.approx(0.095)


def test_obstacle_motion_uses_absolute_time_and_refuses_extrapolation() -> None:
    motion = ObstacleMotion(
        np.asarray([2.0, 5.0]), np.asarray([[[1.0, 0, 1]], [[4.0, 0, 1]]]), np.asarray([0.2])
    )
    replay = run_contact_replay(
        ContactTrigger(3.0, _state(), "fixture_motor_cut", 0, None),
        cf21b_contact_body(),
        motion,
        ContactReplayConfig(duration_seconds=0.05),
    )
    np.testing.assert_allclose(replay.obstacle_centers[:, 0, 0], replay.time_seconds - 1)
    np.testing.assert_array_equal(
        replay.obstacle_velocities[:, 0, 0], np.ones(len(replay.time_seconds))
    )
    with pytest.raises(ValueError, match="time support"):
        motion.sample(5.1)


def test_moving_obstacle_imparts_directional_momentum_to_stationary_drone() -> None:
    # Contact velocity must include the prescribed obstacle velocity. Updating a mocap
    # position alone fails this physical contract, despite apparently colliding visually.
    body = replace(cf21b_contact_body(), gravity=np.zeros(3), drag_matrix_body=np.zeros((3, 3)))
    state = np.asarray([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.0])
    motion = ObstacleMotion(
        np.asarray([0.0, 2.0]),
        np.asarray([[[-0.7, 0, 1.02]], [[0.3, 0, 1.02]]]),
        np.asarray([0.22]),
    )
    replay = run_contact_replay(
        ContactTrigger(0, state, "moving_obstacle_fixture_motor_cut", 0, None),
        body,
        motion,
        ContactReplayConfig(duration_seconds=1.5),
    )
    assert replay.obstacle_contact.any()
    assert not replay.ground_contact.any()
    assert 0.4 < replay.full_state[-1, 7] < 0.65  # Incoming obstacle speed is +0.5 m/s.
    assert replay.full_state[-1, 0] > 0.25
    np.testing.assert_allclose(
        replay.obstacle_velocities[:, 0], [[0.5, 0, 0]] * len(replay.time_seconds)
    )
    np.testing.assert_allclose(replay.obstacle_centers[:, 0, 0], -0.7 + 0.5 * replay.time_seconds)
    assert replay.metadata["obstacle_drive"]["maximum_step_position_correction_m"] < 1e-6
    assert replay.metadata["warning_counts"] == {}


def test_contact_solver_receives_obstacle_velocity_at_identical_contact_pose() -> None:
    body = replace(cf21b_contact_body(), gravity=np.zeros(3), drag_matrix_body=np.zeros((3, 3)))
    state = np.asarray([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.0])
    results = []
    for speed in (0.0, 0.5):
        # Both start with the same negligible penetration. A mocap-position-only scheme
        # produces the same first-step contact impulse, and fails this comparison.
        centers = np.asarray([[[-0.306 + 1e-7, 0, 1.02]], [[-0.306 + 1e-7 + speed, 0, 1.02]]])
        results.append(
            run_contact_replay(
                ContactTrigger(0, state, "contact_velocity_fixture", 0, None),
                body,
                ObstacleMotion(np.asarray([0.0, 1.0]), centers, np.asarray([0.22])),
                ContactReplayConfig(duration_seconds=0.002),
            )
        )
    stationary, moving = results
    assert abs(stationary.full_state[1, 7]) < 1e-5
    assert moving.full_state[1, 7] > 0.1
    assert moving.contact_force_norm_N[0] > 1


def test_safe_source_has_no_implicit_crash_trigger() -> None:
    times = np.asarray([0.0, 0.1])
    states = np.tile(_state(), (2, 1))
    with pytest.raises(ValueError, match="has no unsafe trigger"):
        find_contact_trigger(
            times,
            states,
            _obstacle(),
            ContactReplayConfig(),
            kind="unsafe",
            shell_ego_radius=0.05,
            shell_clearance=0.15,
            degraded_times=np.empty(0),
        )
