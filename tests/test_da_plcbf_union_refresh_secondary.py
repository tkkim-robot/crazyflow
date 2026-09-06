from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmark.da_plcbf_actuator_diagnostic_batch import planned_pairs
from benchmark.da_plcbf_actuator_diagnostic_protocol import SCHEMA, digest
from benchmark.da_plcbf_union_refresh_secondary import (
    ARMS,
    CELLS,
    NEW_WORLDS,
    PROTOCOL_ID,
    audit_calls,
    claim_trial,
    load_identity_rule,
    make_proposal,
)
from crazyflow.safety.da_plcbf.actuator_protocol import content_sha256

if TYPE_CHECKING:
    from pathlib import Path


def fixture_files(directory: Path) -> tuple[Path, Path, Path]:
    rule = {
        "protocol_id": PROTOCOL_ID,
        "new_worlds": [{"family": family, "scene_seed": seed} for family, seed in NEW_WORLDS],
        "arms": list(ARMS),
        "cells": list(CELLS),
        "maximum_new_physical_flights": 40,
        "new_outcomes_consulted": False,
        "immutable_core_count": 16,
        "adaptive_policy_count": 16,
        "refresh_rule": "fixed rule for test",
    }
    identity = directory / "identity.json"
    identity.write_text(json.dumps({"rule": rule, "sha256": digest(rule)}))
    geometry = {"initial_state": [0.0] * 17, "test_world_seed": 30101}
    original = {
        "planner_files": {},
        "worlds": {
            "structured_30101": {
                "geometry": geometry,
                "geometry_sha256": digest(geometry),
                "split": "development",
            }
        },
        "factorial_cells": [
            {
                "cell_id": cell,
                "event_time_seconds": 2.0,
                "no_change_control": index == 0,
                "effectiveness_after": [1.0 if index == 0 else 0.7] * 2 + [1.0] * 2,
                "lag_multipliers_after": [1.0] * 4,
            }
            for index, cell in enumerate(CELLS)
        ],
        "shared_control": {"execution_mode": "deterministic"},
    }
    original_path = directory / "original.json"
    original_path.write_text(
        json.dumps({"schema": SCHEMA, "proposal": original, "sha256": digest(original)})
    )
    binding = {"source_sha256": {}, "checkpoint_files_sha256": {}}
    binding["sha256"] = content_sha256(binding)
    (directory / "diagnostic_runtime_binding.json").write_text(json.dumps(binding))
    return identity, original_path, directory


def test_invalid_identity_is_rejected_before_geometry_opening(tmp_path: Path):
    identity, original, campaign = fixture_files(tmp_path)
    value = json.loads(identity.read_text())
    value["rule"]["new_worlds"][0]["scene_seed"] += 100
    value["sha256"] = digest(value["rule"])
    identity.write_text(json.dumps(value))
    calls = []
    with pytest.raises(ValueError, match="identities"):
        make_proposal(
            identity, original, campaign, geometry_resolver=lambda *args: calls.append(args)
        )
    assert calls == []


def test_secondary_design_is_exactly_sixteen_fresh_and_four_robust_pairs(tmp_path: Path):
    identity, original, campaign = fixture_files(tmp_path)
    proposal = make_proposal(
        identity,
        original,
        campaign,
        geometry_resolver=lambda family, seed: {
            "initial_state": [0.0] * 17,
            "test_family": family,
            "test_world_seed": seed,
        },
    )
    assert len(proposal["trials"]) == len({row["trial_id"] for row in proposal["trials"]}) == 40
    assert len(planned_pairs(proposal, "fresh_validation")) == 16
    assert len(planned_pairs(proposal, "numerical_robustness")) == 4
    robust = [row for row in proposal["trials"] if row["realization"] == "perturb_plus"]
    assert len(robust) == 2
    assert all(
        row["initial_position_x_delta_m"] == row["initial_velocity_x_delta_mps"] == 1e-5
        for row in robust
    )
    assert {row["arm"]["runtime_method"] for row in proposal["trials"]} == {"UNION"}


def test_duplicate_geometry_is_rejected_without_reseeding(tmp_path: Path):
    identity, original, campaign = fixture_files(tmp_path)
    with pytest.raises(ValueError, match="duplicate secondary geometry"):
        make_proposal(
            identity,
            original,
            campaign,
            geometry_resolver=lambda *_: {"initial_state": [0.0] * 17, "unique": True},
        )


def test_exclusive_claim_blocks_reruns_in_another_output_and_unplanned_trials(tmp_path: Path):
    trial = {"trial_id": "one"}
    proposal = {"trials": [trial], "execution": {"claim_directory": str(tmp_path / "claims")}}
    claim_trial(proposal, trial, tmp_path / "first/episode")
    with pytest.raises(FileExistsError):
        claim_trial(proposal, trial, tmp_path / "second/episode")
    with pytest.raises(ValueError, match="unplanned"):
        claim_trial(proposal, {"trial_id": "other"}, tmp_path / "third/episode")
    assert len(list((tmp_path / "claims").glob("*.json"))) == 1


def audit_fixture() -> tuple[list[dict], dict[str, np.ndarray]]:
    calls = [
        {
            "call_index": 0,
            "warmup": True,
            "parameter_sha256": "start",
            "preceding_call_parameter_sha256": None,
            "parameters_changed": False,
        },
        {
            "call_index": 1,
            "warmup": False,
            "state_sha256": "state",
            "parameter_sha256": "changed",
            "preceding_call_parameter_sha256": "start",
            "parameters_changed": True,
            "requested_previous_index": 17,
            "incumbent_in_adaptive_component": True,
            "refreshed": True,
            "effective_previous_index": -1,
            "action": [0.1] * 4,
        },
    ]
    controls = {
        "time": np.array([2.0]),
        "controller_input_state_sha256": np.array(["state"]),
        "control_params_sha256": np.array(["changed"]),
        "planned_command": np.array([[0.1] * 4]),
        "command_applied": np.array([True]),
    }
    return calls, controls


def test_wrapper_audit_proves_effective_index_and_actual_command():
    calls, controls = audit_fixture()
    physical = audit_calls(calls, controls)
    assert physical[0]["time_seconds"] == 2.0
    assert physical[0]["effective_previous_index"] == -1
    changed = deepcopy(calls)
    changed[1]["effective_previous_index"] = 17
    with pytest.raises(ValueError, match="effective index"):
        audit_calls(changed, controls)
    controls["planned_command"][0, 0] += 1e-8
    with pytest.raises(ValueError, match="command differs"):
        audit_calls(calls, controls)


def test_wrapper_audit_rejects_forged_fingerprint_memory():
    calls, controls = audit_fixture()
    calls[1]["preceding_call_parameter_sha256"] = "another"
    with pytest.raises(ValueError, match="fingerprint memory"):
        audit_calls(calls, controls)


def test_identity_cap_cannot_be_relaxed(tmp_path: Path):
    identity, _, _ = fixture_files(tmp_path)
    value = json.loads(identity.read_text())
    value["rule"]["maximum_new_physical_flights"] = 41
    value["sha256"] = digest(value["rule"])
    identity.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="flight cap"):
        load_identity_rule(identity)
