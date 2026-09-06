from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
import pytest

from crazyflow.safety.da_plcbf.actuator_protocol import (
    PROTOCOL_SCHEMA,
    TrialLedger,
    canonical_json,
    crossed_paired_bootstrap,
    freeze_protocol,
    load_sealed_protocol,
    paired_protocol_bootstrap,
    physical_world_id,
    seal_protocol,
    world_binary_wilson,
)

if TYPE_CHECKING:
    from pathlib import Path


def _manifest() -> dict:
    splits = {}
    for index, split in enumerate(("development", "validation", "test")):
        worlds = []
        for scene in range(3):
            physical = {
                "initial_state": [index, scene, 1, 0, 0, 0, 1] + [0] * 6 + [0.11] * 4,
                "duration_s": 2,
                "actuator_event": {"time_s": 0.5, "eta": [0.7, 1, 1, 1]},
                "obstacle_centers": [[1.2, scene * 0.1, 1]],
                "obstacle_radii": [0.15],
            }
            worlds.append(
                {
                    "world_id": physical_world_id(physical),
                    "physical_spec": physical,
                    "cell": "structured_combined",
                }
            )
        splits[split] = {
            "world_seeds": [1000 + 10 * index + i for i in range(3)],
            "library_seeds": [7, 11, 19],
            "worlds": worlds,
        }
    methods = {
        "F2": {
            "definition": "Frozen skill library with current-parameter lag-aware adapter",
            "configuration": {"adapter": "F2", "online_updates": False, "K": 4},
        },
        "A": {
            "definition": "Persistent finite-update publication with the same F2 adapter",
            "configuration": {"adapter": "F2", "online_updates": True, "K": 4},
        },
    }
    return {
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": "synthetic_unit_test_only",
        "source": {"commit": "a" * 40, "files_sha256": {"fixture.py": "b" * 64}},
        "models": {
            "P0": {
                "definition": "17-state effort surrogate",
                "parameters": {"eta": [0.7, 1, 1, 1], "tau_s": [0.06] * 4},
            }
        },
        "control": {
            "command_period_s": 0.04,
            "prediction_horizon_s": 1.2,
            "integration_step_s": 0.002,
        },
        "numeric_ranges": {"affected_effectiveness": [0.7, 1.0], "lag_multiplier": [1, 3]},
        "methods": methods,
        "training_support": {
            method: {
                "training_transitions": 200,
                "library_seeds": [7, 11, 19],
                "selection": "synthetic validation fixture",
            }
            for method in methods
        },
        "splits": splits,
        "outcomes": {
            "modeled_collision": {
                "definition": "Actual collider contact during common exposure",
                "units": "indicator",
                "kind": "binary",
            },
            "safe_task_completion": {
                "definition": "Task complete and clear through common exposure",
                "units": "indicator",
                "kind": "binary",
            },
        },
        "analysis": {
            "confidence_level": 0.95,
            "n_resamples": 1000,
            "resampling_seed": 301,
            "cluster_axes": ["world", "library_seed"],
            "interval": "percentile",
            "primary_multiplicity": "bonferroni",
            "secondary_multiplicity": "descriptive_only",
            "missing_results": "refuse_incomplete_matrix",
            "primary_comparisons": {
                "A_minus_F2": {
                    "method_a": "A",
                    "method_b": "F2",
                    "metrics": ["modeled_collision", "safe_task_completion"],
                    "cells": ["structured_combined"],
                }
            },
        },
        "budget_rationale": "Small synthetic test fixture; this is not a research benchmark budget",
    }


def _record(
    ledger: TrialLedger,
    *,
    scene: int,
    seed: int,
    method: str,
    attempt: str,
    status: str = "completed",
    collision: int = 0,
) -> None:
    world = ledger.protocol.manifest["splits"]["test"]["worlds"][scene]["world_id"]
    ledger.record_attempt(
        attempt_id=attempt,
        split="test",
        world_id=world,
        library_seed=seed,
        method=method,
        status=status,
        metrics={"modeled_collision": collision, "safe_task_completion": 1 - collision},
        reason="synthetic terminal result" if status == "completed" else "process interrupted",
    )


def _complete_ledger() -> TrialLedger:
    ledger = TrialLedger(freeze_protocol(_manifest()))
    for scene in range(3):
        for seed in [7, 11, 19]:
            for method in ["A", "F2"]:
                collision = int(method == "F2" and scene == 0)
                _record(
                    ledger,
                    scene=scene,
                    seed=seed,
                    method=method,
                    attempt=f"{scene}_{seed}_{method}",
                    collision=collision,
                )
    return ledger


def test_canonical_physical_identity_normalizes_numeric_equivalence_and_key_order() -> None:
    first = {"position": [1.0, -0.0, 2], "eta": [0.7, 1, 1, 1]}
    second = {"eta": [0.7, 1.0, 1, 1], "position": [1, 0, 2.0]}
    assert canonical_json(first) == canonical_json(second)
    assert physical_world_id(first) == physical_world_id(second)
    changed = deepcopy(first)
    changed["position"][0] = 1.000001
    assert physical_world_id(first) != physical_world_id(changed)
    with pytest.raises(ValueError, match="administrative"):
        physical_world_id(dict(first, library_seed=7))


def test_freeze_is_detached_immutable_and_runtime_config_cannot_change() -> None:
    manifest = _manifest()
    frozen = freeze_protocol(manifest)
    original_hash = frozen.sha256
    manifest["methods"]["A"]["configuration"]["K"] = 32
    assert frozen.sha256 == original_hash
    assert frozen.manifest["methods"]["A"]["configuration"]["K"] == 4
    detached = frozen.manifest
    detached["control"]["command_period_s"] = 0.05
    assert frozen.manifest["control"]["command_period_s"] == 0.04
    with pytest.raises(FrozenInstanceError):
        frozen.sha256 = "f" * 64
    with pytest.raises(ValueError, match="runtime section"):
        frozen.require_section("methods", manifest["methods"])
    frozen.require_section("control", _manifest()["control"])


def test_seal_refuses_overwrite_and_load_rejects_manifest_tampering(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    frozen = seal_protocol(path, _manifest())
    original = path.read_bytes()
    assert load_sealed_protocol(path) == frozen
    with pytest.raises(FileExistsError):
        seal_protocol(path, _manifest())
    assert path.read_bytes() == original
    envelope = json.loads(path.read_text())
    envelope["manifest"]["numeric_ranges"]["lag_multiplier"][1] = 4
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_sealed_protocol(path)


@pytest.mark.parametrize(
    "replacement", ["TBD", "TODO after test", None, float("nan"), float("inf"), lambda: 3]
)
def test_unresolved_or_nonfinite_protocol_values_cannot_be_sealed(replacement: object) -> None:
    manifest = _manifest()
    manifest["numeric_ranges"]["lag_multiplier"][1] = replacement
    with pytest.raises(ValueError):
        freeze_protocol(manifest)


def test_persistent_fault_recovery_null_is_resolved_without_changing_physical_identity() -> None:
    manifest = _manifest()
    world = manifest["splits"]["test"]["worlds"][0]
    world["physical_spec"]["actuator_events"] = [
        {"time_seconds": 0.5, "recovery_time_seconds": None, "effectiveness": [0.7, 1, 1, 1]}
    ]
    world["world_id"] = physical_world_id(world["physical_spec"])
    frozen = freeze_protocol(manifest)
    retained = frozen.manifest["splits"]["test"]["worlds"][0]
    assert retained["physical_spec"]["actuator_events"][0]["recovery_time_seconds"] is None
    assert retained["world_id"] == physical_world_id(world["physical_spec"])


@pytest.mark.parametrize("location", ["causal_configuration", "event_time", "wrong_physical_path"])
def test_persistent_fault_exception_does_not_allow_unrelated_nulls(location: str) -> None:
    manifest = _manifest()
    world = manifest["splits"]["test"]["worlds"][0]
    if location == "causal_configuration":
        manifest["methods"]["A"]["configuration"]["freeze_learning_at"] = None
    elif location == "event_time":
        world["physical_spec"]["actuator_events"] = [
            {"time_seconds": None, "recovery_time_seconds": None}
        ]
    else:
        world["physical_spec"]["recovery_time_seconds"] = None
    world["world_id"] = physical_world_id(world["physical_spec"])
    with pytest.raises(ValueError, match="unresolved placeholder"):
        freeze_protocol(manifest)


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate_physics",
        "duplicate_seed",
        "false_world_id",
        "invalid_range",
        "invalid_clock",
        "iid_analysis",
        "undeclared_comparison",
        "missing_training",
        "missing_policy",
    ],
)
def test_invalid_protocol_structure_cannot_be_frozen(problem: str) -> None:
    manifest = _manifest()
    if problem == "duplicate_physics":
        manifest["splits"]["test"]["worlds"][0] = deepcopy(
            manifest["splits"]["development"]["worlds"][0]
        )
    elif problem == "duplicate_seed":
        manifest["splits"]["test"]["world_seeds"][0] = manifest["splits"]["development"][
            "world_seeds"
        ][0]
    elif problem == "false_world_id":
        manifest["splits"]["test"]["worlds"][0]["world_id"] = "world-" + "f" * 64
    elif problem == "invalid_range":
        manifest["numeric_ranges"]["lag_multiplier"] = [3, 1]
    elif problem == "invalid_clock":
        manifest["control"]["integration_step_s"] = 0.08
    elif problem == "iid_analysis":
        manifest["analysis"]["cluster_axes"] = ["episode"]
    elif problem == "undeclared_comparison":
        manifest["analysis"]["primary_comparisons"]["A_minus_F2"]["method_b"] = "unimplemented"
    elif problem == "missing_training":
        del manifest["training_support"]["F2"]
    elif problem == "missing_policy":
        del manifest["analysis"]["missing_results"]
    with pytest.raises(ValueError):
        freeze_protocol(manifest)


def test_interrupted_and_completed_attempts_remain_distinct_for_same_physical_trial() -> None:
    ledger = TrialLedger(freeze_protocol(_manifest()))
    _record(ledger, scene=0, seed=7, method="A", attempt="interrupted_1", status="interrupted")
    coverage = ledger.coverage()
    assert coverage["attempts"] == coverage["incomplete_attempts"] == 1
    assert coverage["completed_trials"] == 0
    assert coverage["missing_trials"] == coverage["planned_trials"] == 18
    with pytest.raises(ValueError, match="incomplete paired"):
        ledger.paired_metric("A", "F2", "modeled_collision", cell="structured_combined")
    _record(ledger, scene=0, seed=7, method="A", attempt="completed_2")
    records = ledger.as_mapping()["records"]
    assert records[0]["attempt_id"] != records[1]["attempt_id"]
    assert records[0]["trial_id"] == records[1]["trial_id"]
    assert ledger.coverage()["completed_trials"] == 1
    assert ledger.coverage()["attempts"] == 2
    restored = TrialLedger.from_mapping(ledger.protocol, ledger.as_mapping())
    assert restored.as_mapping() == ledger.as_mapping()
    with pytest.raises(ValueError, match="rerun/replaced"):
        _record(ledger, scene=0, seed=7, method="A", attempt="unwanted_rerun", collision=1)


def test_ledger_rejects_duplicate_ids_wrong_support_and_post_outcome_screening() -> None:
    ledger = TrialLedger(freeze_protocol(_manifest()))
    _record(ledger, scene=0, seed=7, method="A", attempt="done")
    with pytest.raises(ValueError, match="already exists"):
        _record(ledger, scene=1, seed=7, method="A", attempt="done")
    with pytest.raises(ValueError, match="support"):
        _record(ledger, scene=1, seed=1234, method="A", attempt="wrong_seed")
    with pytest.raises(ValueError, match="after comparative outcomes"):
        _record(ledger, scene=0, seed=11, method="F2", attempt="posthoc", status="inadmissible")
    tampered = ledger.as_mapping()
    tampered["records"][0]["trial_id"] = "trial-" + "f" * 64
    with pytest.raises(ValueError, match="trial ID"):
        TrialLedger.from_mapping(ledger.protocol, tampered)


def test_complete_primary_analysis_uses_declared_pairs_and_multiplicity() -> None:
    ledger = _complete_ledger()
    assert ledger.coverage()["completed_trials"] == ledger.coverage()["planned_trials"] == 18
    result = paired_protocol_bootstrap(
        ledger, "A_minus_F2", "modeled_collision", cell="structured_combined"
    )
    assert result.estimate == pytest.approx(-1 / 3)
    assert result.confidence_level == pytest.approx(0.975)  # two predeclared primary metrics
    assert result.n_worlds == result.n_library_seeds == 3
    assert result.n_pairs == 9
    with pytest.raises(ValueError, match="not declared"):
        paired_protocol_bootstrap(
            ledger, "selected_after_outcomes", "modeled_collision", cell="structured_combined"
        )
    with pytest.raises(ValueError, match="outside the prespecified"):
        paired_protocol_bootstrap(
            ledger, "A_minus_F2", "modeled_collision", cell="favorable_subset"
        )


def test_crossed_bootstrap_retains_seed_uncertainty_that_world_replication_cannot_remove() -> None:
    seed_effect = np.array([-1.0, 0.0, 1.0])
    differences = np.tile(seed_effect, (60, 1))
    result = crossed_paired_bootstrap(
        differences, np.zeros_like(differences), n_resamples=4000, resampling_seed=99
    )
    assert result.lower <= -2 / 3
    assert result.upper >= 2 / 3
    assert result.world_only_lower == result.world_only_upper == 0
    assert result.per_seed_effects == (-1.0, 0.0, 1.0)
    assert "Few library seeds" in result.qualification
    assert result == crossed_paired_bootstrap(
        differences, np.zeros_like(differences), n_resamples=4000, resampling_seed=99
    )


def test_paired_bootstrap_cancels_shared_world_seed_noise() -> None:
    rng = np.random.default_rng(10)
    shared = rng.normal(size=(30, 3)) * 100
    result = crossed_paired_bootstrap(shared + 0.125, shared, n_resamples=1000, resampling_seed=1)
    assert result.estimate == pytest.approx(0.125, abs=2e-14)
    assert result.upper - result.lower < 1e-12


@pytest.mark.parametrize(
    "a,b",
    [
        (np.zeros(10), np.zeros(10)),
        (np.zeros((10, 3)), np.zeros((9, 3))),
        (np.zeros((10, 1)), np.zeros((10, 1))),
        (np.full((10, 3), np.nan), np.zeros((10, 3))),
    ],
)
def test_crossed_bootstrap_rejects_unpaired_incomplete_or_single_cluster_arrays(
    a: np.ndarray, b: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        crossed_paired_bootstrap(a, b, n_resamples=1000)


def test_world_binary_wilson_diagnostic_counts_worlds_and_has_nonzero_zero_event_upper() -> None:
    values = np.zeros((60, 3))
    result = world_binary_wilson(values)
    z = NormalDist().inv_cdf(0.975)
    assert result["events"] == 0
    assert result["worlds"] == 60
    assert result["upper"] == pytest.approx(z**2 / (60 + z**2))
    assert result["upper"] > 0.05
    assert result["conditional_on_tested_libraries"]
    values[0, 0] = values[0, 1] = 1
    assert world_binary_wilson(values)["events"] == 1
    assert (
        world_binary_wilson(np.repeat(values, 5, axis=1))["upper"]
        == world_binary_wilson(values)["upper"]
    )
    with pytest.raises(ValueError, match="zeros/ones"):
        world_binary_wilson(np.array([[0, 1], [0, np.nan]]))
