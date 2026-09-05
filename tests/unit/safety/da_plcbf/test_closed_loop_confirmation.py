"""Prespecified confirmation plans and outcome accounting, without controller execution."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

import benchmark.da_plcbf_closed_loop_confirmation as confirmation
from crazyflow.safety.da_plcbf.case_study_world import HoverEncounterConfig, IncomingSphere

if TYPE_CHECKING:
    from pathlib import Path


def test_neighborhood_is_reproducible_and_does_not_replace_invalid_draws() -> None:
    scene = HoverEncounterConfig(incoming=IncomingSphere(radius_m=0.001))
    first = confirmation.confirmation_plan(scene, seed=48791)
    second = confirmation.confirmation_plan(scene, seed=48791)
    assert first == second
    assert len(first["neighborhood"]) == 12
    invalid = [row for row in first["neighborhood"] if not row["admissible"]]
    assert invalid
    assert all(row["invalid_reason"] for row in invalid)
    assert [row["index"] for row in first["neighborhood"]] == list(range(12))
    for row in first["neighborhood"]:
        neighbor = HoverEncounterConfig.from_dict(row["scene"])
        np.testing.assert_allclose(
            np.asarray(neighbor.hover_position) + neighbor.waypoint_offsets,
            np.asarray(scene.hover_position) + scene.waypoint_offsets,
            rtol=0,
            atol=1e-14,
        )
        assert neighbor.obstacle_clearance == scene.obstacle_clearance
        assert neighbor.ego_radius == scene.ego_radius
    json.dumps(first, allow_nan=False)


@pytest.mark.parametrize("count", [0, 6, 11, True])
def test_confirmation_rejects_an_undersized_local_protocol(count: int) -> None:
    with pytest.raises(ValueError, match="twelve"):
        confirmation.confirmation_plan(HoverEncounterConfig(), neighborhood_count=count)


def test_plan_mode_never_initializes_an_execution_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "result.json"
    selected.write_text(json.dumps({"scene": asdict(HoverEncounterConfig())}))

    def forbidden_devices(*args: object) -> None:
        raise AssertionError("planning must precede all device/controller execution")

    monkeypatch.setattr(confirmation.jax, "devices", forbidden_devices)
    output = tmp_path / "plan"
    result = confirmation.run_confirmation(selected, output, mode="plan")
    assert (output / "protocol.json").exists()
    assert not (output / "trials.jsonl").exists()
    assert len(result["neighborhood"]) == 12
    with pytest.raises(FileExistsError):
        confirmation.run_confirmation(selected, output, mode="plan")


def test_calm_prefix_requires_the_actual_onset_input_and_catches_action_changes() -> None:
    arrays = {
        "time": np.array([0.0, 0.04, 0.08]),
        "state": np.zeros((3, 13)),
        "action": np.zeros((3, 4)),
        "hard": np.full((3, 2), -np.inf),
        "smooth": np.zeros((3, 2)),
        "version_used": np.array([0, 1, 2]),
    }
    good = confirmation.compare_calm_prefix(arrays, arrays, 0.04)
    assert good["shared_calm_execution_verified"]
    assert good["fields"]["hard"]["maximum_absolute_difference"] == 0
    json.dumps(good, allow_nan=False)
    changed = deepcopy(arrays)
    changed["action"][1, 0] = 0.01
    assert not confirmation.compare_calm_prefix(arrays, changed, 0.04)[
        "shared_calm_execution_verified"
    ]
    absent = {key: value[:1] for key, value in arrays.items()}
    assert not confirmation.compare_calm_prefix(absent, absent, 0.04)[
        "shared_calm_execution_verified"
    ]


def test_numerical_sensitivity_can_reject_an_apparent_collision_survival_pair() -> None:
    methods = {
        "fixed": {"collider_lower_m": -0.005, "collider_upper_m": -0.004, "ground_lower_m": 1.0},
        "adaptive": {"collider_lower_m": 0.02, "collider_upper_m": 0.021, "ground_lower_m": 1.0},
    }
    records = {
        key: {"methods": deepcopy(methods), "promotion_candidate": True} for key in (1, 2, 4)
    }
    good = confirmation.integration_sensitivity(records)
    assert good["collision_survival_target_survives_observed_integration_sensitivity"]
    records[2]["methods"]["fixed"]["collider_upper_m"] = 0.001
    bad = confirmation.integration_sensitivity(records)
    assert not bad["collision_survival_target_survives_observed_integration_sensitivity"]


def test_prediction_grid_is_not_silently_refined_with_the_plant() -> None:
    with pytest.raises(ValueError, match="prediction/.04 s hold"):
        confirmation.confirmation_plan(
            replace(HoverEncounterConfig(), dt=0.01, control_interval_steps=4)
        )


def test_parameter_intervention_is_a_prior_boundary_with_wind_learning() -> None:
    scene = HoverEncounterConfig()
    boundary = confirmation.attribution_boundary(
        scene, {"first_collider_intersection_seconds": 4.91}
    )
    assert boundary == pytest.approx(4.48)
    assert boundary / 0.04 == pytest.approx(round(boundary / 0.04))
    assert confirmation.attribution_boundary(
        scene, {"first_collider_intersection_seconds": 3.03}
    ) is None
    assert confirmation.attribution_boundary(scene, {}) == pytest.approx(7.6)


@pytest.mark.parametrize("boundary", [2.2, 2.23, float("nan"), 16.0])
def test_explicit_intervention_rejects_invalid_publication_boundaries(
    tmp_path: Path, boundary: float
) -> None:
    with pytest.raises(ValueError, match="post-wind control boundary"):
        confirmation.parameter_reversion_case(
            None,
            HoverEncounterConfig(wind_onset_seconds=2.2),
            "uncompensated",
            {},
            tmp_path / "case",
            boundary_seconds=boundary,
            boundary_reason="prespecified first trajectory divergence",
        )


def test_explicit_intervention_records_reason_before_any_execution(tmp_path: Path) -> None:
    def stop_before_execution(*args: object) -> None:
        raise RuntimeError("test stops before resource initialization")

    output = tmp_path / "intervention"
    with pytest.raises(RuntimeError, match="before resource initialization"):
        confirmation.parameter_reversion_case(
            SimpleNamespace(resources=stop_before_execution),
            HoverEncounterConfig(wind_onset_seconds=2.2),
            "uncompensated",
            {},
            output,
            boundary_seconds=3.0,
            boundary_reason="last boundary before first2cm observed path difference",
        )
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["boundary_seconds"] == 3.0
    assert protocol["selection_reason"] == "last boundary before first2cm observed path difference"
