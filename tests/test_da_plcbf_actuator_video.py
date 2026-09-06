"""Explicitly synthetic artifact tests; these are not flight or rendering outcomes."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmark.da_plcbf_actuator_video import (
    ActuatorVideoConfig,
    _compose,
    frame_times,
    load_episode,
    motor_site_positions,
    validate_pair,
)


def synthetic_episode(path: Any, method: str, *, contact: float | None = None) -> Any:
    path.mkdir()
    times = np.asarray([0.0, 0.04, 0.08])
    state = np.zeros((3, 17))
    state[:, 0] = times
    state[:, 2] = 1.0
    state[:, 3:7] = Rotation.from_rotvec([[0, 0, 0], [0, 0, np.pi / 2], [0, 0, np.pi]]).as_quat()
    state[:, 13:] = 0.1
    eta = np.ones((3, 4))
    eta[1:, 0] = 0.5
    world = {
        "initial_state": state[0, :13].tolist(),
        "waypoint_positions": [[1, 0, 1]],
        "obstacle_mean_centers": [[2, 3, 4]],
        "obstacle_amplitudes": [[0.1, 0.2, 0.3]],
        "obstacle_angular_frequencies": [2.0],
        "obstacle_phases": [0.1],
        "obstacle_radii": [0.2],
    }
    binding = {
        "scene": {"world": world, "navigation_start": 0.0},
        "checkpoint": {"checkpoint_sha256": "synthetic-identical-initial-checkpoint"},
        "initial_state_sha256": "synthetic-identical-state",
        "initial_learner_sha256": "synthetic-identical-learner",
        "config": {
            "execution_mode": "delayed",
            "plant_level": "P1",
            "plant_step_seconds": 0.005,
            "filter_config": {"obstacle_clearance": 0.15, "ego_radius": 0.106},
            "observation_config": {},
        },
    }
    summary = {
        "status": "completed",
        "method": method,
        "termination": "duration_complete" if contact is None else "physical_collision",
        "physical_time_seconds": 0.08,
        "physical_world_id": "synthetic-common-world",
        "modeled_collider_collision": contact is not None,
        "collision": None if contact is None else {"first_intersection_time_seconds": contact},
        "waypoints_total": 1,
        "waypoint_arrival_times_seconds": [],
    }
    for name, payload in (("binding", binding), ("summary", summary)):
        (path / f"{name}.json").write_text(json.dumps(payload))
    np.savez(
        path / "dense.npz",
        time=times,
        state=state,
        actual_effectiveness=eta,
        actual_time_constants=np.full((3, 4), 0.03),
        actual_forces=eta * state[:, 13:],
        command=np.full((3, 4), 0.12),
    )
    candidates = np.broadcast_to(state[:2, None, None, :], (2, 3, 3, 17)).copy()
    candidates[:, :, :, 0] += np.arange(3)[None, :, None] * np.arange(3)[None, None, :] * 0.1
    np.savez(
        path / "controls.npz",
        time=times[:2],
        command_applied=np.asarray([True, True]),
        command_applied_at=np.asarray([0.03, 0.07]),
        candidate_states=candidates,
        candidate_valid=np.ones((2, 3), dtype=bool),
        eligible=np.ones((2, 3), dtype=bool),
        hard=np.ones((2, 3)),
        selected_index=np.asarray([1, 2]),
        goal=np.asarray([[1, 0, 1], [1, 0, 1]]),
        control_library_version=np.asarray([128, 129]),
    )
    np.savez(
        path / "applications.npz",
        time=np.asarray([0.0, 0.03, 0.07]),
        command=np.asarray([[0.1] * 4, [0.12] * 4, [0.14] * 4]),
    )
    return path


def test_prediction_visibility_uses_application_time_and_faults_are_right_continuous(
    tmp_path: Any,
) -> None:
    episode = load_episode(synthetic_episode(tmp_path / "F2", "F2"), expected_method="F2")
    before = episode.sample(0.02)
    assert before.control_index is None
    np.testing.assert_allclose(before.command, 0.1)
    assert episode.sample(0.03).control_index == 0
    assert episode.sample(0.069).control_index == 0
    assert episode.sample(0.07).control_index == 1
    np.testing.assert_allclose(episode.sample(0.07).command, 0.14)
    assert episode.sample(0.039).effectiveness[0] == 1
    assert episode.sample(0.04).effectiveness[0] == 0.5
    assert episode.sample(0.04).actual_forces[0] == pytest.approx(0.05)
    np.testing.assert_allclose(before.state[3:7], Rotation.from_rotvec([0, 0, np.pi / 4]).as_quat())
    np.testing.assert_allclose(
        episode.obstacle_centers(0.5),
        np.asarray([[2, 3, 4]]) + np.asarray([[0.1, 0.2, 0.3]]) * np.sin(1.1),
    )


def test_contact_freezes_recorded_prefix_and_cannot_reveal_later_control(tmp_path: Any) -> None:
    episode = load_episode(
        synthetic_episode(tmp_path / "F2", "F2", contact=0.065), expected_method="F2"
    )
    stopped = episode.sample(0.20)
    at_contact = episode.sample(0.065)
    assert stopped.display_time == 0.065
    assert stopped.contact_stopped
    assert stopped.terminal_stopped
    assert stopped.control_index == 0
    np.testing.assert_array_equal(stopped.state, at_contact.state)
    np.testing.assert_allclose(
        episode.obstacle_centers(stopped.display_time), episode.obstacle_centers(0.065)
    )


def test_pair_requires_same_world_initial_checkpoint_and_full_candidate_evidence(
    tmp_path: Any,
) -> None:
    left = load_episode(synthetic_episode(tmp_path / "F2", "F2"), expected_method="F2")
    right = load_episode(synthetic_episode(tmp_path / "A", "A"), expected_method="A")
    validate_pair(left, right)
    np.testing.assert_allclose(frame_times(left, right, 20), [0, 0.05, 0.08])
    right.binding["checkpoint"]["checkpoint_sha256"] = "a-different-initial-library"
    with pytest.raises(ValueError, match="same nominal checkpoint"):
        validate_pair(left, right)
    path = tmp_path / "F2" / "controls.npz"
    with np.load(path) as saved:
        values = {key: saved[key] for key in saved.files if key != "candidate_states"}
    np.savez(path, **values)
    with pytest.raises(ValueError, match="paths are never fabricated"):
        load_episode(path.parent, expected_method="F2")


def test_motor_annotation_mapping_and_synthetic_composition(tmp_path: Any) -> None:
    sites = motor_site_positions()
    np.testing.assert_array_equal(np.sign(sites[:, :2]), [[1, -1], [-1, -1], [-1, 1], [1, 1]])
    left = load_episode(synthetic_episode(tmp_path / "F2", "F2"), expected_method="F2")
    right = load_episode(synthetic_episode(tmp_path / "A", "A"), expected_method="A")
    config = ActuatorVideoConfig(width=960, height=540, synthetic_fixture=True)
    images = (np.zeros((416, 480, 3), dtype=np.uint8),) * 2
    frame = _compose(images, (left.sample(0.04), right.sample(0.04)), (left, right), config, 0.2)
    assert frame.shape == (540, 960, 3)
    assert frame.dtype == np.uint8
    assert np.max(frame) > 0
