"""Outcome classification and causal full-episode orchestration for discovery."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import benchmark.da_plcbf_closed_loop_search as search
from crazyflow.safety.da_plcbf.case_study_world import HoverEncounterConfig, IncomingSphere
from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel

if TYPE_CHECKING:
    from pathlib import Path


def _method(**changes: Any) -> dict[str, Any]:
    return {
        "censored": False,
        "collider_lower_m": 0.10,
        "collider_upper_m": 0.101,
        "ground_lower_m": 1.0,
        "ground_upper_m": 1.001,
        "all_operational_nodes_pass": True,
        "encounter_completed": True,
        "termination": "completed",
        **changes,
    }


@pytest.mark.parametrize(
    ("fixed_collision", "adaptive_collision", "outcome"),
    [
        (False, False, "both_separated"),
        (True, False, "fixed_only_collision"),
        (False, True, "adaptive_only_collision"),
        (True, True, "both_collision"),
    ],
)
def test_all_four_observed_physical_outcomes_remain_visible(
    fixed_collision: bool, adaptive_collision: bool, outcome: str
) -> None:
    def side(collided: bool) -> dict[str, Any]:
        return (
            _method(
                collider_lower_m=-0.011,
                collider_upper_m=-0.010,
                encounter_completed=False,
                termination="physical_collision",
            )
            if collided
            else _method()
        )

    record = search.classify_pair(
        {"fixed": side(fixed_collision), "adaptive": side(adaptive_collision)}
    )
    assert record["outcome_class"] == outcome
    assert record["promotion_candidate"] == (fixed_collision and not adaptive_collision)


@pytest.mark.parametrize(
    "changes",
    [
        {"censored": True},
        {"collider_lower_m": -1e-5, "collider_upper_m": 1e-5},
        {"ground_lower_m": -1e-5, "ground_upper_m": 1e-5},
        {"termination": "timeout"},
        {"encounter_completed": False},
        {"all_operational_nodes_pass": False},
    ],
)
def test_incomplete_uncertain_or_operationally_failed_adaptive_run_is_not_promoted(
    changes: dict[str, Any],
) -> None:
    result = search.classify_pair(
        {
            "fixed": _method(collider_lower_m=-0.011, collider_upper_m=-0.010),
            "adaptive": _method(**changes),
        }
    )
    assert not result["promotion_candidate"]
    if changes.get("censored") or "collider_upper_m" in changes or "ground_upper_m" in changes:
        assert result["outcome_class"] == "censored_or_unresolved"


def test_shell_only_breach_and_ground_collision_have_different_outcomes() -> None:
    shell_breach = _method(shell_clearance_m=-0.12, enclosure_clearance_m=-0.001)
    assert search.physical_class(shell_breach) == "separated"
    floor_impact = _method(ground_lower_m=-0.005, ground_upper_m=-0.004)
    assert search.physical_class(floor_impact) == "collision"
    result = search.classify_pair({"fixed": shell_breach, "adaptive": floor_impact})
    assert result["outcome_class"] == "adaptive_only_collision"
    assert not result["promotion_candidate"]


def test_rejected_cached_candidates_reach_the_full_episode_proposal_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "publication" / "all_candidates.jsonl.gz"
    source.parent.mkdir()
    with gzip.open(source, "wt") as stream:
        for family in ("geometry-v1", "geometry-v2", "geometry-v3"):
            for mapping in ("uncompensated", "compensated"):
                for accepted in (False, True):
                    if family == "geometry-v1" and accepted:
                        continue
                    row = {
                        "family": family,
                        "case": mapping,
                        "stage_b_accepted": accepted,
                        "time_seconds": 4.0,
                        "arrival_delay": 1.0,
                        "direction": [1.0, 0.0, 0.0],
                        "speed": 2.0,
                        "radius": 0.4,
                        "crossing_offset": [0.0, 0.0, 0.0],
                        "anchor": "t0100",
                        "index": int(accepted),
                        "rejection": None if accepted else "no_adapted_H_advantage",
                        "adaptive_H": -0.5 if not accepted else 0.1,
                        "fixed_H": 0.1,
                    }
                    stream.write(json.dumps(row) + "\n")
    monkeypatch.setattr(search, "OLD", tmp_path)
    proposals = search.existing_proposals(seed=17, count=32)
    rejected = [p for p in proposals if p["proposal"] == "ungated_cached_rejected"]
    assert len(rejected) == 16
    assert {p["mapping"] for p in rejected} == {"compensated", "uncompensated"}
    assert {p["parent"]["family"] for p in rejected} == {
        "geometry-v1",
        "geometry-v2",
        "geometry-v3",
    }
    assert {p["scene"]["obstacle_clearance"] for p in proposals} == set(search.BUFFERS)
    assert proposals == search.existing_proposals(seed=17, count=32)
    for proposal in proposals:
        world = search.build_hover_encounter_world(
            HoverEncounterConfig.from_dict(proposal["scene"])
        )
        assert world.initial_state_time_seconds == 0.0


@pytest.mark.parametrize("family", search.FAMILIES)
def test_scene_families_preserve_geometry_and_have_the_requested_maneuver_demands(
    family: str,
) -> None:
    scene = search.random_scene(np.random.default_rng(47), family, seed=47, buffer=0.02)
    world = search.build_hover_encounter_world(scene)
    assert world.config.ego_radius == 0.106
    assert world.config.obstacle_clearance == 0.02
    assert len(scene.guards) == (2 if family == "guards" else 0)
    assert len(scene.additional_incoming) == (1 if family == "staggered" else 0)
    if family == "moving":
        assert scene.navigation_start_seconds < scene.incoming.arrival_time_seconds
        assert np.linalg.norm(world.initial_state[7:10]) > 0
    centers, _ = world.obstacle_kinematics(0.0)
    assert np.all(
        np.linalg.norm(centers - world.initial_state[:3], axis=-1)
        > world.obstacle_radii + world.config.ego_radius + world.config.obstacle_clearance
    )


def _stub_engine(
    monkeypatch: pytest.MonkeyPatch, *, reject_second_update: bool = False
) -> tuple[Any, dict[str, list[Any]]]:
    """Exercise the real driver/world/audits with inexpensive deterministic control objects."""
    engine = search.EpisodeEvaluator(jax.devices("cpu")[0])
    model = VersionAModel(
        jnp.asarray(0.03),
        jnp.asarray([0.0, 0.0, -9.81]),
        jnp.eye(3),
        jnp.eye(3),
        jnp.eye(3),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros(3),
    )
    initial = SimpleNamespace(params=jnp.asarray(0.0), library_version=660)
    bundle = SimpleNamespace(config=SimpleNamespace(horizon=4), state=initial, point_model=model)
    observed: dict[str, list[Any]] = {"controls": [], "updates": [], "resources": []}

    def controller(
        state: Any, params: Any, point: Any, prediction: Any, previous: Any, goal: Any
    ) -> dict[str, Any]:
        observed["controls"].append(
            {
                "state": np.asarray(state).copy(),
                "model": np.asarray(point.wind_velocity).copy(),
                "prediction": np.asarray(prediction.centers).copy(),
                "mask": np.asarray(prediction.mask).copy(),
                "goal": np.asarray(goal).copy(),
                "params": float(params),
            }
        )
        return {
            "action": jnp.asarray([params, 0.0, 0.0, 0.0]),
            "selected": jnp.asarray(1),
            "mode": jnp.asarray(0),
            "qp": jnp.asarray(True),
            "degraded": jnp.asarray(False),
            "emergency": jnp.asarray(False),
            "fallback": jnp.asarray(False),
            "dual": jnp.asarray(0.01),
            "hard": jnp.asarray([0.2, 0.3]),
            "smooth": jnp.asarray([0.1, 0.2]),
            "eligible": jnp.asarray([True, True]),
            "motor_minimum": jnp.asarray(0.1),
            "held_operational_pass": jnp.asarray(True),
        }

    def update(persistent: Any, state: Any, point: Any) -> tuple[Any, Any]:
        observed["updates"].append((np.asarray(state).copy(), np.asarray(point.wind_velocity)))
        finite = not (reject_second_update and len(observed["updates"]) == 2)
        following = (
            SimpleNamespace(
                params=persistent.params + 0.1, library_version=persistent.library_version + 1
            )
            if finite
            else persistent
        )
        return following, SimpleNamespace(finite_update_applied=finite)

    def resources(mapping: str, world: Any) -> tuple[Any, ...]:
        observed["resources"].append((mapping, world))
        return (
            bundle,
            None,
            SimpleNamespace(step=update),
            controller,
            lambda x, u, m: x.at[0].add(u[0] * world.config.dt),
        )

    monkeypatch.setattr(engine, "resources", resources)
    return engine, observed


def _short_scene() -> HoverEncounterConfig:
    return HoverEncounterConfig(
        incoming=IncomingSphere(
            arrival_time_seconds=0.12, radius_m=0.1, crossing_offset=(3.0, 0.0, 0.0)
        ),
        duration_seconds=0.24,
        wind_onset_seconds=0.08,
        navigation_start_seconds=0.12,
    )


def test_full_pair_starts_at_zero_with_matched_information_and_own_causal_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, observed = _stub_engine(monkeypatch, reject_second_update=True)
    scene = _short_scene()
    fixed, fixed_trace = engine.run_method(scene, "uncompensated", "fixed")
    adaptive, trace = engine.run_method(scene, "uncompensated", "adaptive")
    assert fixed["start_time_seconds"] == adaptive["start_time_seconds"] == 0.0
    assert fixed["termination"] == adaptive["termination"] == "timeout"
    np.testing.assert_array_equal(fixed_trace["dense_states"][0], trace["dense_states"][0])
    np.testing.assert_array_equal(trace["version_used"], [660, 661, 661, 662, 663, 664])
    np.testing.assert_array_equal(trace["completed_version"], [661, 661, 662, 663, 664, 664])
    np.testing.assert_array_equal(trace["finite_update"], [True, False, True, True, True, False])
    assert adaptive["finite_updates"] == 4
    assert fixed["finite_updates"] == 0
    assert adaptive["wind_onset_version"] == 661
    np.testing.assert_allclose(trace["publication_time"], [0.04, -1, 0.12, 0.16, 0.20, -1])
    np.testing.assert_allclose(trace["dense_times"], np.arange(13) * 0.02)
    fixed_controls, adaptive_controls = observed["controls"][:6], observed["controls"][6:]
    for index, (frozen, learned) in enumerate(zip(fixed_controls, adaptive_controls, strict=True)):
        for name in ("prediction", "mask", "model", "goal"):
            np.testing.assert_array_equal(frozen[name], learned[name])
        assert learned["mask"].all()
        expected_wind = (0.0, 0.0, 0.0) if index < 2 else scene.wind_velocity
        np.testing.assert_allclose(learned["model"], expected_wind)
    for index, (state, wind) in enumerate(observed["updates"]):
        np.testing.assert_array_equal(state, trace["state"][index])
        np.testing.assert_array_equal(wind, adaptive_controls[index]["model"])
    assert observed["resources"][0][1].config == observed["resources"][1][1].config
    # Actual commands, rather than a substituted prefix, drive all dense intervals.
    np.testing.assert_allclose(
        np.diff(trace["dense_states"][:, 0]), np.repeat(trace["action"][:, 0], 2) * 0.02, atol=2e-9
    )


def test_freeze_at_onset_shares_the_calm_history_then_holds_its_available_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _observed = _stub_engine(monkeypatch)
    scene = _short_scene()
    _, continuing = engine.run_method(scene, "uncompensated", "adaptive")
    frozen_summary, frozen = engine.run_method(
        scene, "uncompensated", "adaptive", freeze_at=scene.wind_onset_seconds
    )
    np.testing.assert_array_equal(continuing["state"][:3], frozen["state"][:3])
    np.testing.assert_array_equal(continuing["action"][:3], frozen["action"][:3])
    np.testing.assert_array_equal(frozen["version_used"], [660, 661, 662, 662, 662, 662])
    assert frozen_summary["wind_onset_version"] == frozen_summary["final_version"] == 662
    assert frozen_summary["finite_updates"] == 2
    assert not np.array_equal(continuing["state"][-1], frozen["state"][-1])


def test_changing_only_extra_clearance_preserves_actual_collision_geometry() -> None:
    scene = _short_scene()
    worlds = [
        search.build_hover_encounter_world(replace(scene, obstacle_clearance=b))
        for b in search.BUFFERS
    ]
    reference = worlds[0]
    for world in worlds[1:]:
        np.testing.assert_array_equal(world.obstacle_radii, reference.obstacle_radii)
        np.testing.assert_array_equal(world.initial_state, reference.initial_state)
        assert world.config.ego_radius == reference.config.ego_radius
        for name in (
            "speed_max",
            "angular_rate_max",
            "tilt_max_radians",
            "dt",
            "control_interval_steps",
        ):
            assert getattr(world.config, name) == getattr(reference.config, name)
        for when in (0.0, 0.08, 0.12, 0.24):
            for observed, expected in zip(
                world.obstacle_kinematics(when), reference.obstacle_kinematics(when), strict=True
            ):
                np.testing.assert_array_equal(observed, expected)


def test_physical_trial_identity_ignores_seed_but_retains_mapping_and_buffer() -> None:
    scene = {"seed": 1, "obstacle_clearance": 0.15, "wind_velocity": [1.6, 0.8, 0]}
    expected = search.physical_scene_identity(scene, "uncompensated")
    assert search.physical_scene_identity({**scene, "seed": 98}, "uncompensated") == expected
    assert search.physical_scene_identity(scene, "compensated") != expected
    assert (
        search.physical_scene_identity({**scene, "obstacle_clearance": 0}, "uncompensated")
        != expected
    )


def test_partial_attempt_preserves_original_files_without_overwriting_prior_attempt(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "fixed-only.txt").write_text("first attempt")
    first = search.preserve_partial_attempt(trial)
    assert first == tmp_path / "trial.partial-1"
    assert (first / "fixed-only.txt").read_text() == "first attempt"
    trial.mkdir()
    (trial / "fixed-only.txt").write_text("second attempt")
    second = search.preserve_partial_attempt(trial)
    assert second == tmp_path / "trial.partial-2"
    assert (second / "fixed-only.txt").read_text() == "second attempt"
    assert (first / "fixed-only.txt").read_text() == "first attempt"
    assert search.preserve_partial_attempt(trial) is None


def test_parameter_reversion_preserves_common_prefix_and_uses_original_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _ = _stub_engine(monkeypatch)
    scene = _short_scene()
    callbacks = []
    _, held = engine.run_method(scene, "uncompensated", "adaptive", freeze_at=0.12)
    summary, reverted = engine.run_method(
        scene, "uncompensated", "adaptive", revert_at=0.12, snapshot_callback=callbacks.append
    )
    np.testing.assert_array_equal(held["state"][:4], reverted["state"][:4])
    np.testing.assert_array_equal(held["action"][:3], reverted["action"][:3])
    np.testing.assert_array_equal(reverted["version_used"], [660, 661, 662, 660, 660, 660])
    np.testing.assert_array_equal(reverted["parameter_reverted"], [False] * 3 + [True] * 3)
    assert summary["finite_updates"] == 3
    assert int(callbacks[3]["snapshot"].library_version) == 663
    assert int(callbacks[3]["used_snapshot"].library_version) == 660
    assert summary["final_version"] == 663
    assert not np.array_equal(held["state"][-1], reverted["state"][-1])
