"""Geometry proposals must retain valid scenes and avoid claiming rerun outcomes."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import benchmark.da_plcbf_closed_loop_refinement as refinement
from crazyflow.safety.da_plcbf.case_study_world import (
    HoverEncounterConfig,
    IncomingSphere,
    build_hover_encounter_world,
)

if TYPE_CHECKING:
    from pathlib import Path


def _paired_detour() -> tuple[HoverEncounterConfig, dict[str, np.ndarray]]:
    """Two complete old paths: one sizable evasive excursion, one smaller excursion."""
    scene = HoverEncounterConfig(
        incoming=IncomingSphere(
            arrival_time_seconds=5.0,
            direction=(0.0, 1.0, 0.0),
            crossing_offset=(0.0, 0.0, 0.65),
            radius_m=0.3,
            speed_m_s=2.0,
        ),
        duration_seconds=8.0,
        navigation_start_seconds=7.0,
        obstacle_clearance=0.02,
    )
    times = np.arange(401) * 0.02
    excursion = np.maximum(1 - np.abs(times - 5.0) / 1.2, 0)
    traces: dict[str, np.ndarray] = {}
    for method, amplitude in (("fixed", 1.2), ("adaptive", 0.25)):
        states = np.zeros((times.size, 13))
        states[:, :3] = scene.hover_position
        states[:, 0] = amplitude * excursion
        states[:, 6] = 1
        states[:, 7] = np.gradient(states[:, 0], times)
        traces[f"{method}_dense_times"] = times.copy()
        traces[f"{method}_dense_states"] = states
    return scene, traces


def _assert_initial_clearance(scene: HoverEncounterConfig) -> None:
    world = build_hover_encounter_world(scene)
    centers, _velocities = world.obstacle_kinematics(0.0)
    assert np.all(
        np.linalg.norm(centers - world.initial_state[:3], axis=1)
        > world.obstacle_radii + scene.ego_radius + scene.obstacle_clearance
    )


def test_guard_targets_old_fixed_detour_without_invalidating_initial_scene() -> None:
    scene, traces = _paired_detour()
    proposed, metadata = refinement.propose_escape_scene(
        scene, traces, "guard", np.random.default_rng(19)
    )
    _assert_initial_clearance(proposed)
    assert len(proposed.guards) == len(scene.guards) + 1
    assert proposed.incoming == scene.incoming
    assert proposed.additional_incoming == scene.additional_incoming
    assert proposed.ego_radius == scene.ego_radius == 0.106
    assert proposed.obstacle_clearance == scene.obstacle_clearance
    guard = proposed.guards[-1]
    center = np.asarray(scene.hover_position) + guard.offset
    # These old paths only direct proposal placement: a real rerun may evade differently.
    distances = {
        method: np.linalg.norm(traces[f"{method}_dense_states"][:, :3] - center, axis=1).min()
        - guard.radius_m
        for method in ("fixed", "adaptive")
    }
    assert distances["fixed"] < 0.086
    assert distances["adaptive"] > scene.ego_radius
    assert proposed.wind_velocity == scene.wind_velocity
    assert proposed.wind_onset_seconds == scene.wind_onset_seconds
    assert proposed.navigation_start_seconds == scene.navigation_start_seconds
    assert not metadata["used_local_fallback"]
    assert metadata["requested_strategy"] == "guard"
    assert "proposal only" in metadata["scope"]
    assert "not an executed closed-loop outcome" in metadata["scope"]
    assert "must rerun from time zero" in metadata["scope"]
    assert "outcome_class" not in metadata
    assert "promotion_candidate" not in metadata
    audits = metadata["old_recorded_path_audits"]
    assert audits["fixed"]["actual_xml_sphere_geometry"]["minimum_clearance_upper_bound_m"] < 0
    assert (
        audits["adaptive"]["actual_xml_sphere_geometry"]["minimum_clearance_lower_bound_m"] > 0
    )


def test_second_mover_has_exogenous_absolute_passage_through_measured_detour() -> None:
    scene, traces = _paired_detour()
    proposed, _metadata = refinement.propose_escape_scene(
        scene, traces, "mover", np.random.default_rng(21)
    )
    _assert_initial_clearance(proposed)
    assert len(proposed.additional_incoming) == 1
    assert proposed.incoming == scene.incoming
    assert proposed.guards == scene.guards
    sphere = proposed.additional_incoming[-1]
    assert scene.wind_onset_seconds < sphere.arrival_time_seconds < scene.duration_seconds
    world = build_hover_encounter_world(proposed)
    centers, velocities = world.obstacle_kinematics(sphere.arrival_time_seconds)
    np.testing.assert_allclose(
        centers[1], np.asarray(scene.hover_position) + sphere.crossing_offset
    )
    np.testing.assert_allclose(
        velocities[1],
        sphere.speed_m_s * np.asarray(sphere.direction) / np.linalg.norm(sphere.direction),
    )
    old_fixed_position = np.array(
        [
            np.interp(
                sphere.arrival_time_seconds,
                traces["fixed_dense_times"],
                traces["fixed_dense_states"][:, axis],
            )
            for axis in range(3)
        ]
    )
    assert np.linalg.norm(old_fixed_position - centers[1]) < sphere.radius_m + 0.086
    # Reusing the world at a later branch time must not restart either obstacle's clock.
    branch = build_hover_encounter_world(proposed, initial_time_seconds=3.0)
    np.testing.assert_array_equal(
        branch.obstacle_kinematics(sphere.arrival_time_seconds)[0], centers
    )


def test_diagnostics_report_recorded_rescue_actions_and_do_not_infer_avoidance_cause() -> None:
    scene, traces = _paired_detour()
    traces.update(
        fixed_time=np.array((2.0, 4.0, 4.5, 5.0, 7.0)),
        fixed_emergency=np.array((False, True, True, False, True)),
        fixed_fallback=np.array((False, False, False, True, False)),
        fixed_qp=np.array((True, False, False, False, False)),
    )
    diagnostic = refinement.escape_diagnostics(
        replace(scene, navigation_start_seconds=1.0), traces
    )
    assert diagnostic["recorded_control_counts"] == {"emergency": 2, "fallback": 1, "qp": 0}
    assert diagnostic["maximum_horizontal_displacement_m"] == pytest.approx(1.2)
    assert diagnostic["maximum_vertical_displacement_m"] == pytest.approx(0)
    assert 3.0 < diagnostic["first_recorded_response_seconds"] < 4.5
    assert set(diagnostic["labels"]) == {
        "early_recorded_motion",
        "lateral_escape_or_task_motion",
        "emergency_control_executed",
        "fallback_prefix_executed",
        "waypoint_leg_active_before_encounter",
    }
    assert "task motion alone is not attributed to avoidance" in diagnostic["scope"]


@pytest.mark.parametrize(
    ("invalid", "message"),
    (
        ("branch", "starting at time zero"),
        ("duplicate_time", "strictly increasing"),
        ("nonfinite", "finite"),
        ("quaternion", "normalized"),
    ),
)
def test_refinement_rejects_invalid_or_non_full_episode_paths(invalid: str, message: str) -> None:
    scene, traces = _paired_detour()
    if invalid == "branch":
        traces["fixed_dense_times"] += 1.0
    elif invalid == "duplicate_time":
        traces["fixed_dense_times"][1] = 0.0
    elif invalid == "nonfinite":
        traces["adaptive_dense_states"][1, 0] = np.nan
    elif invalid == "quaternion":
        traces["adaptive_dense_states"][1, 6] = 2.0
    with pytest.raises(ValueError, match=message):
        refinement.propose_escape_scene(scene, traces, "guard", np.random.default_rng(1))


def _parent(
    scene: HoverEncounterConfig,
    trial_id: str,
    objective: float,
    mapping: str = "uncompensated",
) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "scene": asdict(scene),
        "objective": objective,
        "mapping": mapping,
        "stage_b_accepted": False,
    }


def test_parent_selection_keeps_mapping_buffer_and_approach_diversity_without_H_gate() -> None:
    scene, _traces = _paired_detour()
    same_stratum = [
        _parent(
            replace(
                scene,
                incoming=replace(scene.incoming, crossing_offset=(0.1 * i, 0.0, 0.65)),
            ),
            f"same-{i}",
            i,
        )
        for i in range(3)
    ]
    other_strata = [
        _parent(scene, "compensated", 10.0, "compensated"),
        _parent(replace(scene, obstacle_clearance=0.0), "no-buffer", 11.0),
        _parent(
            replace(scene, incoming=replace(scene.incoming, direction=(-1.0, 1.0, 0.0))),
            "other-approach",
            12.0,
        ),
    ]
    # An identical geometry/mapping is one experimental identity, despite a new trial ID.
    duplicate = {**same_stratum[0], "trial_id": "duplicate", "objective": 0.5}
    seed_only_duplicate = _parent(replace(scene, seed=999), "seed-only-duplicate", 0.75)
    records = [
        *same_stratum,
        duplicate,
        seed_only_duplicate,
        *other_strata,
        {"trial_id": "unevaluated"},
    ]
    selected = refinement.select_parents(records, limit=4)
    assert {row["trial_id"] for row in selected} == {
        "same-0",
        "compensated",
        "no-buffer",
        "other-approach",
    }
    assert not any(row["stage_b_accepted"] for row in selected)
    assert selected == refinement.select_parents(records, limit=4)
    all_unique = refinement.select_parents(records, limit=20)
    assert len(all_unique) == 6
    assert "duplicate" not in {row["trial_id"] for row in all_unique}
    assert "seed-only-duplicate" not in {row["trial_id"] for row in all_unique}


def test_proposal_command_binds_its_exact_parent_scene_and_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene, traces = _paired_detour()
    parent = _parent(scene, "parent-000", 0.1)
    trial = tmp_path / parent["trial_id"]
    trial.mkdir()
    trace_path = trial / "traces.npz"
    np.savez_compressed(trace_path, **traces)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps(parent) + "\n")
    output = tmp_path / "proposed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "da_plcbf_closed_loop_refinement.py",
            "--parents",
            str(ledger),
            "--output",
            str(output),
            "--count",
            "1",
            "--strategy",
            "guard",
        ],
    )
    refinement.main()
    proposal = json.loads((output / "proposals.json").read_text())[0]
    diagnostic = json.loads((output / "ESCAPE_PROPOSAL_DIAGNOSTICS.json").read_text())[0]
    protocol = json.loads((output / "REFINEMENT_PROTOCOL.json").read_text())
    assert proposal["lineage"] == diagnostic["lineage"]
    assert proposal["lineage"]["trace_sha256"] == hashlib.sha256(
        trace_path.read_bytes()
    ).hexdigest()
    assert proposal["lineage"]["parent_scene_sha256"] == hashlib.sha256(
        json.dumps(parent["scene"], sort_keys=True).encode()
    ).hexdigest()
    assert proposal["parent"] == parent["trial_id"]
    assert protocol["planned_distinct_scene_mapping_pairs"] == 1
    assert "no new full episode has been executed" in protocol["scope"]
    assert not (output / "ledger.jsonl").exists()
    assert not (output / "traces.npz").exists()
    _assert_initial_clearance(HoverEncounterConfig.from_dict(proposal["scene"]))
