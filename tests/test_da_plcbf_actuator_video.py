"""Explicitly synthetic artifact tests; these are not flight or rendering outcomes."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from benchmark.da_plcbf_actuator_video import (
    ActuatorVideoConfig,
    _compose,
    _online_updates_used,
    _prediction_affects_control,
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
        "initial_library_version": 128,
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
        mode=np.asarray(["qp", "fallback"]),
        qp_valid=np.asarray([True, False]),
        fallback_valid=np.asarray([False, True]),
        executed_policy_dual=np.asarray([0.01, 0.0]),
        goal=np.asarray([[1, 0, 1], [1, 0, 1]]),
        control_library_version=np.asarray([128, 129]),
    )
    np.savez(
        path / "applications.npz",
        time=np.asarray([0.0, 0.03, 0.07]),
        command=np.asarray([[0.1] * 4, [0.12] * 4, [0.14] * 4]),
    )
    return path


def test_online_update_count_waits_for_recorded_application(tmp_path: Any) -> None:
    episode = load_episode(synthetic_episode(tmp_path / "adaptive", "A"), expected_method="A")
    assert _online_updates_used(episode, episode.sample(0.0)) == 0
    assert _online_updates_used(episode, episode.sample(0.03)) == 0
    assert _online_updates_used(episode, episode.sample(0.069)) == 0
    assert _online_updates_used(episode, episode.sample(0.07)) == 1


@pytest.mark.parametrize(
    "corruption", [None, "normalization", "time", "state", "attitude", "force", "right_force"]
)
def test_legacy_force_limit_requires_exact_adjacent_event_evidence(
    tmp_path: Any, corruption: str | None
) -> None:
    directory = synthetic_episode(tmp_path / "adaptive", "A")
    binding = json.loads((directory / "binding.json").read_text())
    binding["scene"].update(event_time=0.04, effectiveness_after=[0.5, 1, 1, 1])
    (directory / "binding.json").write_text(json.dumps(binding))
    with np.load(directory / "dense.npz") as archive:
        dense = {key: archive[key][[0, 1, 1, 2]].copy() for key in archive.files}
    dense["time"] = np.asarray([0, np.nextafter(0.04, -np.inf), 0.04, 0.08])
    dense["actual_forces"][1] = dense["state"][1, 13:]
    if corruption == "time":
        dense["time"][1] = 0.04 - 1e-5
    elif corruption == "state":
        dense["state"][1, 0] += 0.01
    elif corruption == "attitude":
        dense["state"][1, 3:7] = Rotation.from_rotvec([0, 0, 0.01]).as_quat()
    elif corruption == "normalization":
        quaternion = np.asarray([0.02, -0.01, 0.03, 0.9993], dtype=np.float32)
        quaternion /= np.linalg.norm(quaternion)
        dense["state"][1, 3:7] = quaternion
        dense["state"][2, 3:7] = quaternion / np.linalg.norm(quaternion)
    elif corruption == "force":
        dense["actual_forces"][1, 0] += 0.01
    elif corruption == "right_force":
        dense["actual_forces"][2, 0] += 0.01
    np.savez(directory / "dense.npz", **dense)
    if corruption not in {None, "normalization"}:
        with pytest.raises(ValueError, match="unexplained recorded force"):
            load_episode(directory, expected_method="A")
        return
    episode = load_episode(directory, expected_method="A")
    assert len(episode.force_boundary_audit) == 1
    assert episode.sample(dense["time"][1]).effectiveness[0] == 1
    assert episode.sample(0.04).effectiveness[0] == 0.5
    np.testing.assert_array_equal(episode.dense["actual_forces"], dense["actual_forces"])


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


def test_loss_contract_comparison_still_requires_identical_complete_initial_state(tmp_path: Any):
    left = load_episode(synthetic_episode(tmp_path / "A", "A"), expected_method="A")
    right = load_episode(synthetic_episode(tmp_path / "A_BAL", "A_BAL"), expected_method="A_BAL")
    right.binding["checkpoint"]["checkpoint_sha256"] = "different-loss-contract"
    validate_pair(left, right, allow_different_learning_contract=True)
    right.binding["initial_learner_sha256"] = "different-Adam-history"
    with pytest.raises(ValueError, match="initial_learner_sha256"):
        validate_pair(left, right, allow_different_learning_contract=True)


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


def test_persistent_renderer_keeps_initial_config_while_live_camera_moves(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise moving recorded frames with a fake renderer, without JAX or GL initialization."""
    from benchmark import da_plcbf_actuator_video as video

    instances = []

    class FakeSim:
        def __init__(self, **_: Any) -> None:
            self.mj_model = SimpleNamespace(
                vis=SimpleNamespace(
                    map=SimpleNamespace(shadowclip=1.0), quality=SimpleNamespace(shadowsize=1)
                ),
                stat=SimpleNamespace(extent=1.0),
            )
            self.viewer = SimpleNamespace(
                viewer=SimpleNamespace(cam=SimpleNamespace(lookat=np.zeros(3), distance=0.0))
            )
            self.initial_config = None
            self.live_positions = []
            self.closed = False
            instances.append(self)

        def render(self, *, cam_config: dict, width: int, height: int, **_: Any) -> np.ndarray:
            if self.initial_config is None:
                self.initial_config = deepcopy(cam_config)
            else:
                assert self.initial_config.keys() == cam_config.keys()
                for key in cam_config:
                    np.testing.assert_array_equal(cam_config[key], self.initial_config[key])
            self.live_positions.append(self.viewer.viewer.cam.lookat.copy())
            return np.zeros((height, width, 3), dtype=np.uint8)

        def close(self) -> None:
            self.closed = True

    # The render loop imports these lazily; isolate this test from all real
    # simulator modules instead of instantiating a simulator or a JAX device.
    for name in ("crazyflow", "crazyflow.safety", "crazyflow.safety.da_plcbf"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["crazyflow"].Sim = FakeSim
    helpers = ModuleType("crazyflow.safety.da_plcbf.mujoco_comparison_video")
    helpers._install_marker_shadow_categories = lambda _: None
    helpers._set_two_world_poses = lambda *_: None
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)

    def fake_frames(filename: str, *_: Any, **__: Any) -> Any:
        with open(filename, "wb") as stream:
            while True:
                frame = yield
                assert frame.ndim == 3
                stream.write(b"synthetic-frame")

    encoder = ModuleType("imageio_ffmpeg")
    encoder.write_frames = fake_frames
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", encoder)
    monkeypatch.setattr(video, "_markers", lambda *_: None)
    monkeypatch.setattr(video, "_compose", lambda *_: np.zeros((540, 960, 3), np.uint8))
    left = synthetic_episode(tmp_path / "F2", "F2")
    right = synthetic_episode(tmp_path / "A", "A")
    result = video.render_pair(
        left,
        right,
        tmp_path / "video",
        ActuatorVideoConfig(width=960, height=540, synthetic_fixture=True),
    )
    assert result.is_file()
    assert len(instances) == 1 and instances[0].closed
    live = np.asarray(instances[0].live_positions[1:])
    assert np.ptp(live[:, 0]) > 0.04
    summary = json.loads((result.parent / "render_summary.json").read_text())
    assert summary["status"] == "completed" and summary["frame_count"] == 3


def test_prediction_ring_requires_an_actually_used_policy_branch(tmp_path: Any) -> None:
    episode = load_episode(synthetic_episode(tmp_path / "A", "A"), expected_method="A")
    assert not _prediction_affects_control(episode, episode.sample(0.02))
    qp_sample, fallback_sample = episode.sample(0.03), episode.sample(0.07)
    assert _prediction_affects_control(episode, qp_sample)
    assert _prediction_affects_control(episode, fallback_sample)
    episode.controls["executed_policy_dual"] = np.asarray([0.0, 0.0])
    assert not _prediction_affects_control(episode, qp_sample)
    assert _prediction_affects_control(episode, fallback_sample)
    episode.controls["executed_policy_dual"] = np.asarray([0.01, 0.0])
    episode.controls["mode"] = np.asarray(["emergency", "fallback"])
    assert not _prediction_affects_control(episode, qp_sample)
    episode.controls["mode"] = np.asarray(["qp", "fallback"])
    episode.controls["qp_valid"] = np.asarray([False, False])
    assert not _prediction_affects_control(episode, qp_sample)
    episode.controls["fallback_valid"] = np.asarray([False, False])
    assert not _prediction_affects_control(episode, fallback_sample)
    episode.controls["fallback_valid"] = np.asarray([False, True])
    eligible = episode.controls["eligible"].copy()
    eligible[1, 2] = False
    episode.controls["eligible"] = eligible
    assert not _prediction_affects_control(episode, fallback_sample)


def test_primary_library_comparison_preserves_physical_matching(tmp_path: Any) -> None:
    left = load_episode(synthetic_episode(tmp_path / "PD_F", "PD_F"), expected_method="PD_F")
    right = load_episode(synthetic_episode(tmp_path / "F2", "F2"), expected_method="F2")
    right.binding["checkpoint"]["checkpoint_sha256"] = "learned-initialization"
    right.binding["initial_learner_sha256"] = "different-parameters-and-optimizer"
    validate_pair(left, right, allow_different_libraries=True)
    right.binding["initial_state_sha256"] = "different-physical-state"
    with pytest.raises(ValueError, match="initial_state_sha256"):
        validate_pair(left, right, allow_different_libraries=True)
