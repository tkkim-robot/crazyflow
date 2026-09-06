from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.da_plcbf_actuator_diagnostic_analysis import (
    DECISION_FIELDS,
    _history_without_service,
    arrays_digest,
    authenticate_diagnostic_record,
    checkpoint_groups,
    command_comparison,
    compare_prefix_arrays,
    prefix_arrays,
)
from benchmark.da_plcbf_actuator_diagnostic_protocol import file_digest


def test_numeric_hash_retains_dtype_shape_and_field_identity() -> None:
    first = {"body": np.array([1.0, 2.0], dtype=np.float32)}
    assert arrays_digest(first) == arrays_digest(deepcopy(first))
    assert arrays_digest(first) != arrays_digest({"body": first["body"].astype(np.float64)})
    assert arrays_digest(first) != arrays_digest({"body": first["body"].reshape(2, 1)})
    assert arrays_digest(first) != arrays_digest({"command": first["body"]})


def test_history_excludes_measured_wall_cost_but_retains_update_schedule_and_loss() -> None:
    one = {
        "started_wall_time": 123.0,
        "service_seconds": 0.02,
        "training_simulation_time": 1.96,
        "loss": {"total": 0.04},
        "version": 178,
    }
    two = dict(one, started_wall_time=456.0, service_seconds=0.9)
    assert _history_without_service(one) == _history_without_service(two)
    two["training_simulation_time"] = 1.92
    assert _history_without_service(one) != _history_without_service(two)


def _run() -> SimpleNamespace:
    times = np.array([0.0, 0.04, 0.08])
    states = np.zeros((3, 17), dtype=np.float64)
    return SimpleNamespace(
        controls={
            "time": times,
            "controller_seconds": times + 0.01,
            "planned_command": np.ones((3, 4)),
            "selected_index": np.zeros(3, dtype=int),
        },
        dense={"time": times, "state": states, "command": np.ones((3, 4))},
        applications={
            "time": times,
            "state": states.copy(),
            "command": np.ones((3, 4)),
            "control_index": np.arange(3),
        },
    )


def test_prefix_includes_fault_state_but_excludes_right_continuous_boundary_command() -> None:
    run = _run()
    prefix = prefix_arrays(run, 0.04)
    assert prefix["dense.state"].shape == (2, 17)
    assert prefix["dense.command"].shape == (1, 4)
    assert prefix["controls.planned_command"].shape == (1, 4)
    assert "controls.controller_seconds" not in prefix
    assert prefix["applications.command"].shape == (1, 4)


def test_optional_repertoire_logging_does_not_change_physical_prefix() -> None:
    one, two = _run(), _run()
    two.controls["candidate_states"] = np.zeros((3, 17, 61, 17))
    two.controls["candidate_commands"] = np.zeros((3, 17, 60, 4))
    two.controls["controller_seconds"] += 100
    assert arrays_digest(prefix_arrays(one, 0.08)) == arrays_digest(prefix_arrays(two, 0.08))
    two.controls["selected_index"][0] = 1
    assert arrays_digest(prefix_arrays(one, 0.08)) != arrays_digest(prefix_arrays(two, 0.08))


def test_exact_prefix_comparison_does_not_hide_small_roundoff_or_missing_fields() -> None:
    one = {"state": np.array([1.0], dtype=np.float32)}
    two = {"state": np.nextafter(one["state"], np.float32(2.0))}
    report = compare_prefix_arrays(one, two)
    assert report["exact_equal"] is False
    assert report["checks"]["state"]["maximum_finite_absolute_difference"] > 0
    assert compare_prefix_arrays(one, {})["exact_equal"] is False


def test_command_comparison_uses_actual_applications_and_identifies_common_state_divergence() -> (
    None
):
    one, two = _run(), _run()
    for run in (one, two):
        for key in DECISION_FIELDS:
            run.controls.setdefault(key, np.zeros(3))
    two.applications["command"][1, 2] += 0.001
    two.controls["planned_command"][2, 0] += 100
    report, arrays = command_comparison(one, two)
    assert report["first_divergence"]["time_seconds"] == 0.04
    assert report["first_divergence"]["identical_physical_state"] is True
    assert report["first_divergence"]["maximum_motor_command_difference_n"] == pytest.approx(0.001)
    assert report["equal_commands_before_first_divergence"] == 1
    assert np.array_equal(arrays["adaptive.command"], two.applications["command"])


def _checkpoint(stem: Path) -> None:
    named = {
        "physical_state": np.zeros(17, dtype=np.float64),
        "actor": np.ones(2, dtype=np.float32),
        "adam": np.zeros(2, dtype=np.float32),
        "counter": np.array(4, dtype=np.int32),
        "teacher": np.ones(2, dtype=np.float32),
    }
    np.savez(Path(f"{stem}.npz"), **named)

    def leaf(name: str) -> dict:
        return {"kind": "array", "key": name}

    def mapping(items: dict) -> dict:
        return {"kind": "dict", "items": items}

    metadata = {
        "npz_sha256": file_digest(Path(f"{stem}.npz")),
        "arrays": {
            key: {"dtype": value.dtype.str, "shape": list(value.shape)}
            for key, value in named.items()
        },
        "structure": mapping(
            {
                "physical_state": leaf("physical_state"),
                "state": mapping(
                    {
                        "params": mapping({"value": leaf("actor")}),
                        "optimizer_state": mapping({"mu": leaf("adam")}),
                        "cumulative_gradient_steps": leaf("counter"),
                    }
                ),
                "reference": mapping({"params": mapping({"value": leaf("teacher")})}),
            }
        ),
        "reference_actor_config": {},
        "reference_learning_config": {"retention": 5},
        "reference_sha256": "a" * 64,
        "config": {},
    }
    Path(f"{stem}.json").write_text(json.dumps(metadata))


def test_checkpoint_hashing_preserves_decimal_snapshot_name_and_catches_tampering(
    tmp_path: Path,
) -> None:
    stem = tmp_path / "critical-2.000000000"
    _checkpoint(stem)
    _metadata, groups = checkpoint_groups(stem)
    assert len(groups["optimizer_and_counters"]) == 64
    assert groups["physical"] != groups["actor"]
    Path(f"{stem}.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        checkpoint_groups(stem)


def test_checkpoint_declared_shape_must_match_the_archive(tmp_path: Path) -> None:
    stem = tmp_path / "critical-0.400000000"
    _checkpoint(stem)
    path = Path(f"{stem}.json")
    metadata = json.loads(path.read_text())
    metadata["arrays"]["adam"]["shape"] = [1, 2]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="dtype/shape"):
        checkpoint_groups(stem)


def test_record_authentication_rejects_supplied_context_not_recorded_at_runtime(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "trial/attempt-00"
    episode.mkdir(parents=True)
    record = {"episode_directory": str(episode), "summary": {}, "runtime_binding_sha256": "a" * 64}
    (episode.parent / "record.json").write_text(json.dumps(record))
    run = SimpleNamespace(directory=episode, summary={})
    with pytest.raises(ValueError, match="not the recorded one"):
        authenticate_diagnostic_record(run, {"runtime_binding_sha256": "b" * 64})
