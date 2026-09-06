from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from benchmark.da_plcbf_actuator_diagnostic_batch import planned_pairs, resolve_record, summarize

if TYPE_CHECKING:
    from pathlib import Path


def proposal() -> dict:
    common = {
        "world_key": "same-world",
        "cell_id": "cell",
        "realization": "primary",
        "library_seed": 11,
    }
    return {
        "stage_trial_ids": {"matched_factorial": ["frozen", "adaptive"]},
        "factorial_cells": [{"cell_id": "cell", "event_time_seconds": 0.4}],
        "trials": [
            {
                **common,
                "trial_id": "frozen",
                "arm": {
                    "arm": "A_BAL_FREEZE_AT_FAULT",
                    "runtime_method": "A_BAL",
                    "freeze_learning_at": 0.4,
                },
            },
            {
                **common,
                "trial_id": "adaptive",
                "arm": {"arm": "A_BAL", "runtime_method": "A_BAL", "freeze_learning_at": None},
            },
        ],
    }


def test_design_pairs_do_not_depend_on_retained_successful_records():
    (pair,) = planned_pairs(proposal(), "matched_factorial")
    assert pair["event_time_seconds"] == 0.4
    assert set(pair["arms"]) == {"frozen", "adaptive"}


@pytest.mark.parametrize("mutation", ["different_realization", "wrong_event", "missing_arm"])
def test_unmatched_design_cannot_create_a_pair(mutation: str):
    value = proposal()
    if mutation == "different_realization":
        value["trials"][0]["realization"] = "fresh_build_1"
    elif mutation == "wrong_event":
        value["trials"][0]["arm"]["freeze_learning_at"] = 2.0
    else:
        value["trials"].pop()
    with pytest.raises(ValueError):
        planned_pairs(value, "matched_factorial")


def test_reuse_authenticates_original_actual_episode_parent(tmp_path: Path):
    original = tmp_path / "original" / "trial"
    supplied = tmp_path / "supplied"
    (original / "episode").mkdir(parents=True)
    (supplied / "trial").mkdir(parents=True)
    trial = {"trial_id": "trial"}
    record = {
        "trial": trial,
        "runtime_binding_sha256": "runtime",
        "episode_directory": str(original / "episode"),
    }
    for path in (original / "record.json", supplied / "trial/record.json"):
        path.write_text(json.dumps(record))
    campaigns = [
        {
            "directory": str(supplied),
            "binding": {"resolved_trial_ids": ["trial"], "scientific_runtime_sha256": "runtime"},
        }
    ]
    loaded, auth = resolve_record(trial, campaigns)
    assert loaded == record
    assert auth["original_source_campaign_directory"] == str(tmp_path / "original")
    changed = deepcopy(record)
    changed["runtime_binding_sha256"] = "changed"
    (original / "record.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="original episode-parent"):
        resolve_record(trial, campaigns)


def test_conflicting_copies_cannot_select_a_favorable_outcome(tmp_path: Path):
    trial = {"trial_id": "trial"}
    campaigns = []
    for index in range(2):
        path = tmp_path / str(index)
        (path / "trial").mkdir(parents=True)
        (path / "trial/record.json").write_text(
            json.dumps(
                {"trial": trial, "runtime_binding_sha256": "runtime", "success": bool(index)}
            )
        )
        campaigns.append(
            {
                "directory": str(path),
                "binding": {
                    "resolved_trial_ids": ["trial"],
                    "scientific_runtime_sha256": "runtime",
                },
            }
        )
    with pytest.raises(ValueError, match="conflicting"):
        resolve_record(trial, campaigns)


def test_rejected_and_pending_rows_stay_in_denominator_without_success_credit():
    result = summarize(
        [
            {
                "status": "pending",
                "admissible_pair": False,
                "task_outcome": "adaptation_alone_succeeds",
            },
            {"status": "rejected", "admissible_pair": False},
            {
                "status": "mechanism_pending",
                "admissible_pair": True,
                "task_outcome": "frozen_alone_succeeds",
                "collision_free_outcome": "both_succeed",
            },
        ]
    )
    assert result["planned_pair_count"] == 3
    assert result["unresolved_or_inadmissible_pair_count"] == 2
    assert result["task_outcomes"]["adaptation_alone_succeeds"] == 0
    assert result["task_outcomes"]["frozen_alone_succeeds"] == 1
