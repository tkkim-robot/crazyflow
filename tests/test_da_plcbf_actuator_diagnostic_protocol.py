from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from benchmark.da_plcbf_actuator_diagnostic_protocol import (
    DEVELOPMENT,
    METHODS,
    PREFIX_FIELDS,
    VALIDATION,
    analyze_pair,
    build_proposal,
    factorial_cells,
    four_outcomes,
    load_sealed,
    seal,
)

if TYPE_CHECKING:
    from pathlib import Path


def _proposal() -> dict:
    return build_proposal(
        {
            "scene_seed": 30101,
            "library_seed": 11,
            "episode_config": {"execution_mode": "deterministic"},
        },
        resolver=lambda family, seed: {
            "initial_state": [0, 0, 1.4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "obstacle_mean_centers": [[seed / 10000, 0, 1.4]],
            "obstacle_amplitudes": [[0, 1, 0]],
            "duration_seconds": 14 if family == "structured" else 24,
        },
    )


def _pair() -> tuple[dict, dict]:
    shared = {
        "world_key": "structured_30101",
        "physical_spec_sha256": "0" * 64,
        "runtime_binding_sha256": "1" * 64,
        "library_seed": 11,
        "realization": "primary",
        "status": "completed",
        "actual_collision": False,
        "operational_violation": False,
        "safe_task_completion": True,
        "reached_full_duration": True,
        "postfault_publication_count": 0,
        "applied_commands_sha256": "2" * 64,
        "prefix": dict.fromkeys(PREFIX_FIELDS, "3" * 64),
    }
    frozen, adaptive = deepcopy(shared), deepcopy(shared)
    frozen["arm"] = "A_BAL_FREEZE_AT_FAULT"
    adaptive["arm"] = "A_BAL"
    adaptive["postfault_publication_count"] = 12
    adaptive["common_state_evaluation"] = {
        "shared_inputs_sha256": "4" * 64,
        "per_skill_values_sha256": "5" * 64,
        "actual_command_comparison_sha256": "6" * 64,
        "minimum_union_minus_frozen_hard_value": 0.0,
    }
    return frozen, adaptive


def test_fixed_splits_small_budget_and_no_new_outcome_selection() -> None:
    proposal = _proposal()
    assert set(DEVELOPMENT).isdisjoint(VALIDATION)
    assert len(VALIDATION) == 8
    assert proposal["selection"]["new_outcomes_consulted"] is False
    assert set(proposal["methods"]) == set(METHODS)
    assert proposal["budget"]["uncollapsed_episode_upper_bound"] == 148
    assert proposal["budget"]["unique_episodes"] == 144
    assert len({r["trial_id"] for r in proposal["trials"]}) == 144
    assert len(proposal["stage_trial_ids"]["fresh_validation"]) == 64
    assert (
        proposal["methods"]["F2_2K"]["policy_count"]
        == proposal["methods"]["UNION"]["policy_count"]
        == 32
    )


def test_factorial_holds_geometry_and_uses_aligned_fault_clock() -> None:
    proposal = _proposal()
    cells = factorial_cells()
    assert len(cells) == 11
    assert sum(c["no_change_control"] for c in cells) == 1
    for cell in cells:
        assert round(cell["event_time_seconds"] / 0.04) * 0.04 == pytest.approx(
            cell["event_time_seconds"]
        )
        assert cell["event_time_seconds"] + cell["extra_lead_seconds"] == pytest.approx(2.0)
        assert cell["effectiveness_after"][2:] == [1, 1]
        assert cell["lag_multipliers_after"][2:] == [1, 1]
    for world in proposal["worlds"].values():
        matching = [r for r in proposal["trials"] if r["world_key"] == world["world_key"]]
        assert {r["geometry_sha256"] for r in matching} == {world["geometry_sha256"]}
        for row in matching:
            if row["arm"]["arm"].endswith("_FREEZE_AT_FAULT"):
                cell = next(c for c in cells if c["cell_id"] == row["cell_id"])
                assert row["arm"]["freeze_learning_at"] == cell["event_time_seconds"]
                assert row["arm"]["runtime_method"] in {"A_BAL", "PD_A"}


def test_sealed_design_refuses_overwrite_and_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "sealed.json"
    proposal = _proposal()
    envelope = seal(path, proposal)
    proposal["selection"]["new_outcomes_consulted"] = True
    assert load_sealed(path)["selection"]["new_outcomes_consulted"] is False
    with pytest.raises(FileExistsError):
        seal(path, proposal)
    envelope["proposal"]["selection"]["validation"][0][1] = 30101
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="digest"):
        load_sealed(path)


def test_perturbed_physical_inputs_are_separately_hashed() -> None:
    proposal = _proposal()
    rows = [
        row
        for row in proposal["trials"]
        if row["world_key"] == "structured_30101"
        and row["cell_id"] == "eta0.7_lag1_extra0"
        and row["arm"]["arm"] == "A_BAL"
    ]
    hashes = {row["realization"]: row["physical_spec_sha256"] for row in rows}
    assert hashes["primary"] == hashes["fresh_build_1"] == hashes["fresh_build_2"]
    assert len({hashes["primary"], hashes["perturb_plus"], hashes["perturb_minus"]}) == 3


def test_duplicate_physical_worlds_cannot_be_disguised_as_new_validation_seeds() -> None:
    with pytest.raises(ValueError, match="distinct physical geometries"):
        build_proposal(
            {"scene_seed": 30101, "library_seed": 11, "episode_config": {}},
            resolver=lambda _family, _seed: {"initial_state": [0] * 13},
        )


@pytest.mark.parametrize(
    ("frozen", "adaptive", "expected"),
    [
        (True, True, "both_succeed"),
        (False, True, "adaptation_alone_succeeds"),
        (True, False, "frozen_alone_succeeds"),
        (False, False, "both_fail"),
    ],
)
def test_four_way_outcomes(frozen: bool, adaptive: bool, expected: str) -> None:
    assert four_outcomes(frozen, adaptive) == expected


@pytest.mark.parametrize("invalid", [0, 1, None, "False"])
def test_four_way_outcomes_do_not_coerce_missing_or_string_metrics(invalid: object) -> None:
    with pytest.raises(ValueError, match="booleans"):
        four_outcomes(invalid, True)


def test_timeouts_are_task_failures_even_when_both_collision_free() -> None:
    frozen, adaptive = _pair()
    frozen["safe_task_completion"] = False
    result = analyze_pair(frozen, adaptive)
    assert result["task_outcome"] == "adaptation_alone_succeeds"
    assert result["collision_free_outcome"] == "both_succeed"
    adaptive["postfault_publication_count"] = 0
    assert analyze_pair(frozen, adaptive)["postfault_adaptation_available"] is False


@pytest.mark.parametrize("field", PREFIX_FIELDS)
def test_rejects_changed_optimizer_history_or_other_prefix_inputs(field: str) -> None:
    frozen, adaptive = _pair()
    adaptive["prefix"][field] = "a" * 64
    with pytest.raises(ValueError, match="pre-fault history"):
        analyze_pair(frozen, adaptive)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("status", "interrupted", "incomplete"),
        ("actual_collision", True, "contradicts"),
        ("operational_violation", True, "contradicts"),
        ("reached_full_duration", False, "contradicts"),
        ("postfault_publication_count", -1, "publication count"),
        ("applied_commands_sha256", None, "actual command"),
    ],
)
def test_rejects_incomplete_or_contradictory_attempts(
    key: str, value: object, message: str
) -> None:
    frozen, adaptive = _pair()
    adaptive[key] = value
    with pytest.raises(ValueError, match=message):
        analyze_pair(frozen, adaptive)


def test_contact_terminated_attempt_is_valid_failure_but_unexplained_prefix_is_incomplete() -> None:
    frozen, adaptive = _pair()
    adaptive.update(safe_task_completion=False, reached_full_duration=False, actual_collision=True)
    assert analyze_pair(frozen, adaptive)["task_outcome"] == "frozen_alone_succeeds"
    adaptive["actual_collision"] = False
    with pytest.raises(ValueError, match="partial exposure"):
        analyze_pair(frozen, adaptive)


def test_requires_same_adaptive_parent_and_no_postfault_updates_in_frozen_arm() -> None:
    frozen, adaptive = _pair()
    frozen["arm"] = "F2"
    with pytest.raises(ValueError, match="same adaptive parent"):
        analyze_pair(frozen, adaptive)
    frozen["arm"] = "A_BAL_FREEZE_AT_FAULT"
    frozen["postfault_publication_count"] = 1
    with pytest.raises(ValueError, match="published a postfault"):
        analyze_pair(frozen, adaptive)


def test_requires_common_state_and_actual_executed_command_evidence() -> None:
    frozen, adaptive = _pair()
    del adaptive["common_state_evaluation"]["actual_command_comparison_sha256"]
    with pytest.raises(ValueError, match="actual-command evidence"):
        analyze_pair(frozen, adaptive)
