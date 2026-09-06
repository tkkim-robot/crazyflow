"""Offline audits of runtime publication accounting and saved physical timing."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmark.da_plcbf_actuator_diagnostic_runtime import (
    SCHEMA,
    analyze_episode,
    classify_compute_processes,
    digest,
    load_protocol,
    prescribed_center_crossings,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def test_compute_process_classification_ignores_only_own_process_and_retains_desktop() -> None:
    own = "1234, /env/bin/python, 512"
    desktop = "4567, /usr/bin/anydesk, 0"
    result = classify_compute_processes([own, desktop], own_pid=1234)
    assert result == {
        "foreign_compute_processes": [desktop],
        "declared_zero_memory_desktop_contexts": [desktop],
        "blocking_foreign_compute_processes": [],
    }


@pytest.mark.parametrize(
    "line",
    [
        "4567, /usr/bin/anydesk, 1",
        "4567, /usr/bin/anydesk, N/A",
        "4567, /usr/bin/python, 0",
        "4567, /usr/bin/python, 512",
        "4567, /usr/bin/anydesk-helper, 0",
    ],
)
def test_foreign_compute_exception_requires_exact_anydesk_name_and_zero_memory(line: str) -> None:
    result = classify_compute_processes([line], own_pid=1234)
    assert result == {
        "foreign_compute_processes": [line],
        "declared_zero_memory_desktop_contexts": [],
        "blocking_foreign_compute_processes": [line],
    }


def test_prescribed_center_crossings_use_geometry_and_ignore_static_guards() -> None:
    physical = {
        "duration_seconds": 14.0,
        "obstacle_amplitudes": [[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]],
        "obstacle_angular_frequencies": [0.1, 0.2, 0.3, 0],
        "obstacle_phases": [-0.58, -0.98, -0.03, 0],
    }
    crossings = prescribed_center_crossings(physical)
    assert [row["obstacle_index"] for row in crossings] == [1, 0]
    assert [row["time_seconds"] for row in crossings] == pytest.approx([4.9, 5.8])
    physical["irrelevant_collision_time"] = 2.7
    assert prescribed_center_crossings(physical) == crossings


@pytest.mark.parametrize("phases", [[0.0], [-1.5]])
def test_prescribed_center_crossings_reject_crossings_outside_exposure(phases: list[float]) -> None:
    physical = {
        "duration_seconds": 2.0,
        "obstacle_amplitudes": [[1, 0, 0]],
        "obstacle_angular_frequencies": [0.1],
        "obstacle_phases": phases,
    }
    with pytest.raises(ValueError, match="no first positive crossing"):
        prescribed_center_crossings(physical)


def test_protocol_content_mutation_is_rejected_without_file_verification(tmp_path: Path) -> None:
    body = {"schema": SCHEMA, "fault_time_seconds": 2.0}
    envelope = {"protocol": body, "sha256": digest(body)}
    path = tmp_path / "protocol.json"
    _write_json(path, envelope)
    assert load_protocol(path, verify_files=False) == body
    envelope["protocol"]["fault_time_seconds"] = 0.4
    _write_json(path, envelope)
    with pytest.raises(ValueError, match="schema or content digest changed"):
        load_protocol(path, verify_files=False)


def test_negative_moving_obstacle_frequency_is_rejected() -> None:
    physical = {
        "duration_seconds": 14.0,
        "obstacle_amplitudes": [[1, 0, 0]],
        "obstacle_angular_frequencies": [-0.1],
        "obstacle_phases": [-0.49],
    }
    with pytest.raises(ValueError, match="frequencies must be positive"):
        prescribed_center_crossings(physical)


@pytest.fixture
def saved_episode(tmp_path: Path) -> tuple[Path, dict]:
    physical = {"duration_seconds": 5.0, "actuator_events": [{"time_seconds": 2.0}]}
    config = {
        "method": "F2",
        "execution_mode": "paced",
        "observation_config": {"parameter_delay_seconds": 0.2},
        "controller_reserve_seconds": 0.003,
    }
    protocol = {
        "checkpoints": {
            "A_BAL": {
                "complete_state_sha256": "startup-state",
                "latest_model_sha256": "nominal-model",
                "initial_version": 10,
            }
        },
        "physical_spec_sha256": digest(physical),
        "shared_episode_config": config,
        "fault_time_seconds": 2.0,
        "timing_landmarks": [
            {"label": "fault", "time_seconds": 2.0},
            {"label": "legacy_first_action_divergence", "time_seconds": 2.40},
            {"label": "balanced_first_action_divergence", "time_seconds": 2.48},
        ],
        "prescribed_center_crossings": [
            {"obstacle_index": 0, "time_seconds": 4.9},
            {"obstacle_index": 1, "time_seconds": 5.8},
        ],
    }
    updates = []
    for version, training, credited, finite in (
        (11, 1.0, True, True),
        (12, 1.8, True, True),
        (13, 2.0, True, True),
        (14, 4.2, True, True),
        (15, 4.9, True, True),  # Completed and pending at terminal time.
        (16, 4.95, False, True),  # Finishes after the flight; no flight credit.
        (17, 4.96, True, False),
    ):
        updates.append(
            {
                "computed_version": version,
                "training_simulation_time": training,
                "completion_credited": credited,
                "finite_update_applied": finite,
                "estimated_model_sha256": "nominal-model" if training < 2 else "fault-model",
                "completed_wall_time": 100 + training + 0.1,
                "service_seconds": 0.1,
            }
        )
    publications = [
        {
            "version": update["computed_version"],
            "training_simulation_time": update["training_simulation_time"],
            "completed_wall_time": update["completed_wall_time"],
            "published_wall_time": 100 + when,
            "published_simulation_time": when,
        }
        for update, when in zip(updates[:4], [1.9, 2.0, 4.8, 4.9], strict=True)
    ]
    summary = {
        "method": "A_BAL",
        "execution_mode": "asynchronous",
        "snapshot_publications": publications,
        "physical_time_seconds": 5.0,
        "status": "completed",
        "termination": "duration",
        "full_episode_completed": True,
        "modeled_collider_collision": False,
        "actual_operational_all_nodes_pass": True,
        "successful_full_episode": True,
        "waypoints_completed": 2,
        "waypoints_total": 2,
        "finite_credited_updates": 5,
        "finite_uncredited_updates": 1,
        "final_pending_library_version": 15,
        "learner_calls": 7,
        "control_count": 4,
        "controller_deadline_misses": 1,
        "learner_deadline_misses": 0,
        "skipped_sensing_ticks": 2,
        "controller_seconds": {"count": 4, "mean": 0.04},
        "execution_wall_seconds": 5.1,
        "simulator_and_audit_seconds": 0.1,
        "timing_contract": {"physical_clock": "elapsed wall time"},
        "error": None,
    }
    binding = {
        "initial_learner_sha256": "startup-state",
        "scene": {"physical_spec": physical},
        "config": {**deepcopy(config), "method": "A_BAL", "execution_mode": "asynchronous"},
    }
    warmup = {"calls": 3, "disposable": True}
    for name, value in (
        ("summary", summary),
        ("binding", binding),
        ("updates", updates),
        ("warmup", warmup),
    ):
        _write_json(tmp_path / f"{name}.json", value)
    np.savez(
        tmp_path / "controls.npz",
        time=np.array([0.0, 1.9, 2.0, 4.9]),
        library_version=np.array([10, 11, 12, 14]),
        estimated_model_sha256=np.array(
            ["nominal-model", "fault-model", "fault-model", "fault-model"]
        ),
        command_applied=np.array([False, True, True, True]),
        command_applied_at=np.array([np.nan, 1.95, 2.03, 4.95]),
        snapshot_age_seconds=np.array([0.0, 0.9, 0.2, 0.7]),
        boundary_lateness_seconds=np.array([0.0, 0.02, 0.04, 0.06]),
    )
    np.savez(tmp_path / "applications.npz", time=np.array([0.0, 1.95, 2.03, 4.95]))
    return tmp_path, protocol


def test_counts_require_actual_publication_and_strict_center_crossing_cutoffs(
    saved_episode: tuple[Path, dict],
) -> None:
    directory, protocol = saved_episode
    result = analyze_episode(directory, protocol)
    assert result["published_online_updates"] == 4
    assert result["credited_finite_completed_updates"] == 5
    assert result["uncredited_finite_completed_updates"] == 1
    assert result["completed_unpublished_version"] == 15
    assert result["published_before_fault"] == 1  # Publication at exactly 2 s is excluded.
    assert result["center_crossing_counts"][0] == {
        "obstacle_index": 0,
        "time_seconds": 4.9,
        "physical_prefix_reached_center_crossing": True,
        "published_before_center_crossing": 3,  # The publication at exactly 4.9 s is excluded.
        "postfault_published_before_center_crossing": 2,
        "postfault_trained_published_before_center_crossing": 1,
    }
    assert result["center_crossing_counts"][1]["physical_prefix_reached_center_crossing"] is False
    assert result["center_crossing_counts"][1]["published_before_center_crossing"] == 4
    assert result["warmup"] == {"calls": 3, "disposable": True}


def test_landmarks_count_only_publications_strictly_before_each_fixed_time(
    saved_episode: tuple[Path, dict],
) -> None:
    directory, protocol = saved_episode
    summary_path = directory / "summary.json"
    updates_path = directory / "updates.json"
    summary = json.loads(summary_path.read_text())
    updates = json.loads(updates_path.read_text())
    # Put publications exactly on both historical action-divergence landmarks.
    for index, publication_time, training_time in ((2, 2.40, 2.0), (3, 2.48, 2.40)):
        updates[index]["training_simulation_time"] = training_time
        updates[index]["completed_wall_time"] = 100 + training_time + 0.01
        publication = summary["snapshot_publications"][index]
        publication.update(
            training_simulation_time=training_time,
            completed_wall_time=updates[index]["completed_wall_time"],
            published_wall_time=100 + publication_time,
            published_simulation_time=publication_time,
        )
    _write_json(summary_path, summary)
    _write_json(updates_path, updates)
    result = analyze_episode(directory, protocol)
    for actual, landmark, count, postfault_count in zip(
        result["landmark_counts"], protocol["timing_landmarks"], [1, 2, 3], [0, 0, 1], strict=True
    ):
        assert actual["label"] == landmark["label"]
        assert actual["time_seconds"] == landmark["time_seconds"]
        assert actual["published_before_landmark"] == count
        assert actual["postfault_trained_published_before_landmark"] == postfault_count
        assert actual["physical_prefix_reached_landmark"] is True


def test_report_uses_applied_command_times_and_training_model_observation_age(
    saved_episode: tuple[Path, dict],
) -> None:
    directory, protocol = saved_episode
    result = analyze_episode(directory, protocol)
    delays = result["actual_sensing_to_command_application_seconds"]
    assert delays["count"] == 3  # Unapplied command with a NaN time is excluded.
    assert delays["nonfinite_count"] == 0
    assert delays["mean"] == pytest.approx((0.05 + 0.03 + 0.05) / 3)
    assert delays["maximum"] == pytest.approx(0.05)
    holds = result["actual_command_hold_seconds"]
    assert holds["count"] == 4
    assert holds["mean"] == pytest.approx(1.25)
    assert holds["maximum"] == pytest.approx(2.92)
    ages = result["model_observation_age_seconds"]
    assert ages["mean"] == pytest.approx((0.0 + 1.1 + 0.4 + 0.9) / 4)
    assert ages["maximum"] == pytest.approx(1.1)
    assert result["controls_with_training_model_mismatch"] == 2


@pytest.mark.parametrize("version", [16, 17, 999])
def test_publication_rejects_late_nonfinite_or_missing_updates(
    saved_episode: tuple[Path, dict], version: int
) -> None:
    directory, protocol = saved_episode
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    summary["snapshot_publications"][0]["version"] = version
    _write_json(path, summary)
    with pytest.raises(ValueError, match="completed finite credited update"):
        analyze_episode(directory, protocol)


@pytest.mark.parametrize("violation", ["unfinished", "after_terminal"])
def test_publication_rejects_impossible_completion_or_terminal_credit(
    saved_episode: tuple[Path, dict], violation: str
) -> None:
    directory, protocol = saved_episode
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    publication = summary["snapshot_publications"][-1]
    if violation == "unfinished":
        publication["completed_wall_time"] = publication["published_wall_time"] + 1
    else:
        publication["published_simulation_time"] = summary["physical_time_seconds"] + 0.1
    _write_json(path, summary)
    with pytest.raises(ValueError, match="publication"):
        analyze_episode(directory, protocol)


@pytest.mark.parametrize("violation", ["startup", "world", "settings"])
def test_mismatched_startup_world_or_control_settings_are_rejected(
    saved_episode: tuple[Path, dict], violation: str
) -> None:
    directory, protocol = saved_episode
    path = directory / "binding.json"
    binding = json.loads(path.read_text())
    if violation == "startup":
        binding["initial_learner_sha256"] = "different-optimizer-history"
    elif violation == "world":
        binding["scene"]["physical_spec"]["actuator_events"][0]["time_seconds"] = 0.4
    else:
        binding["config"]["controller_reserve_seconds"] = 0.0
    _write_json(path, binding)
    with pytest.raises(ValueError, match="differs|differ"):
        analyze_episode(directory, protocol)


def test_application_log_rejects_a_hold_extending_before_previous_application(
    saved_episode: tuple[Path, dict],
) -> None:
    directory, protocol = saved_episode
    np.savez(directory / "applications.npz", time=np.array([0.0, 2.0, 1.9]))
    with pytest.raises(ValueError, match="causal application/hold ordering"):
        analyze_episode(directory, protocol)


@pytest.mark.parametrize("field", ["completed_wall_time", "training_simulation_time"])
def test_publication_timestamps_must_match_actual_update(
    saved_episode: tuple[Path, dict], field: str
) -> None:
    directory, protocol = saved_episode
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    summary["snapshot_publications"][0][field] -= 0.01
    _write_json(path, summary)
    with pytest.raises(ValueError, match="differs from its actual update"):
        analyze_episode(directory, protocol)


@pytest.mark.parametrize("violation", ["duplicate", "backward", "startup_version"])
def test_publications_cannot_repeat_or_regress_versions(
    saved_episode: tuple[Path, dict], violation: str
) -> None:
    directory, protocol = saved_episode
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    publications = summary["snapshot_publications"]
    if violation == "duplicate":
        publications.insert(1, deepcopy(publications[0]))
    elif violation == "backward":
        publications[0], publications[1] = publications[1], publications[0]
    else:
        protocol["checkpoints"]["A_BAL"]["initial_version"] = publications[0]["version"]
    _write_json(path, summary)
    with pytest.raises(ValueError, match="advance once without duplicates"):
        analyze_episode(directory, protocol)
