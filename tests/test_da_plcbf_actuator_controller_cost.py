"""CPU-only tests for common-input profiling provenance, scheduling and accounting."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmark import da_plcbf_actuator_controller_cost as cost

if TYPE_CHECKING:
    from pathlib import Path


def test_all_300_calls_are_balanced_across_methods_cases_and_positions() -> None:
    plan = cost.balanced_plan(
        cost.METHODS, [label for label, _ in cost.CASES], 30, phase="measured"
    )
    assert len(plan) == 300
    cells = Counter((row["method"], row["case"]) for row in plan)
    assert cells == {(method, label): 30 for method in cost.METHODS for label, _ in cost.CASES}
    assert [row["sample_index"] for row in plan] == list(range(300))
    for method in cost.METHODS:
        for case, _ in cost.CASES:
            rows = [row for row in plan if row["method"] == method and row["case"] == case]
            assert Counter(row["method_position"] for row in rows) == dict.fromkeys(range(5), 6)
            assert Counter(row["case_position"] for row in rows) == {0: 15, 1: 15}
    for repetition in range(30):
        first_case = next(row["case"] for row in plan if row["repetition"] == repetition)
        assert first_case == cost.CASES[repetition % 2][0]
    first = [row["method"] for row in plan if row["repetition"] == 0 and row["case_position"] == 0]
    reversed_cycle = [
        row["method"] for row in plan if row["repetition"] == 5 and row["case_position"] == 0
    ]
    assert first == list(cost.METHODS)
    assert reversed_cycle == first[::-1]


@pytest.mark.parametrize(
    "methods,cases,repeats",
    [
        ([], ["c"], 1),
        (["a", "a"], ["c"], 1),
        (["a"], [], 1),
        (["a"], ["c", "c"], 1),
        (["a"], ["c"], 0),
        (["a"], ["c"], True),
    ],
)
def test_invalid_or_duplicate_plan_inputs_are_rejected(
    methods: list, cases: list, repeats: int
) -> None:
    with pytest.raises(ValueError):
        cost.balanced_plan(methods, cases, repeats, phase="measured")


def test_array_identity_retains_dtype_shape_signed_zero_and_weak_type() -> None:
    original = cost.array_description(np.array([0.0], np.float32), weak_type=False)
    for value, weak in [
        (np.array([-0.0], np.float32), False),
        (np.array([0.0], np.float64), False),
        (np.array(0.0, np.float32), False),
        (np.array([0.0], np.float32), True),
    ]:
        assert cost.array_description(value, weak_type=weak) != original


@pytest.mark.parametrize("mutation", ["signed_zero", "value", "dtype", "shape", "extra", "missing"])
def test_input_archive_value_dtype_shape_or_membership_change_is_rejected(
    tmp_path: Path, mutation: str
) -> None:
    path = tmp_path / "inputs.npz"
    value = np.array([1.0, -0.0], np.float32)
    leaf = {"array_key": "input", **cost.array_description(value, weak_type=False)}
    descriptions = [{"leaves": [leaf]}]
    np.savez(path, input=value)
    cost.verify_array_archive(path, descriptions)
    changed = {"input": value.copy()}
    if mutation == "signed_zero":
        changed["input"][1] = 0.0
    elif mutation == "value":
        changed["input"][0] = np.nextafter(np.float32(1), np.float32(2))
    elif mutation == "dtype":
        changed["input"] = value.astype(np.float64)
    elif mutation == "shape":
        changed["input"] = value.reshape(1, 2)
    elif mutation == "extra":
        changed["extra"] = value
    else:
        changed.clear()
    np.savez(path, **changed)
    with pytest.raises(ValueError, match="membership changed|array changed"):
        cost.verify_array_archive(path, descriptions)


def test_input_archive_rejects_duplicate_manifest_leaf_keys(tmp_path: Path) -> None:
    value = np.asarray([1.0], dtype=np.float32)
    leaf = {"array_key": "input", **cost.array_description(value, weak_type=False)}
    path = tmp_path / "inputs.npz"
    np.savez(path, input=value)
    with pytest.raises(ValueError, match="duplicate bound input array keys"):
        cost.verify_array_archive(path, [{"leaves": [leaf, deepcopy(leaf)]}])


@pytest.mark.parametrize("x64", [False, True])
def test_restore_tree_preserves_archived_bytes_dtype_and_scalar_weak_types(x64: bool) -> None:
    with jax.enable_x64(x64):
        original = {
            "weak": (jnp.asarray(7), jnp.asarray(-0.0), jnp.asarray(1 + 2j)),
            "strong": [
                np.asarray([0x80000000, 0x00000001, 0x3F800001], dtype=np.uint32).view(np.float32),
                np.asarray([-32768, 32767], dtype=np.int16),
                np.asarray(True),
            ],
        }
        arrays: dict[str, np.ndarray] = {}
        description = cost.describe_tree(original, "saved", arrays)
        template = {
            "weak": (-1, 2.0, 0j),
            "strong": [np.zeros(3, np.float32), np.zeros(2, np.int16), np.asarray(False)],
        }
        restored = cost.restore_tree(template, description, arrays)
        assert cost.describe_tree(restored, "saved", {}) == description
        assert all(jax.typeof(value).weak_type for value in restored["weak"])
        assert all(not jax.typeof(value).weak_type for value in restored["strong"])
        assert np.signbit(np.asarray(restored["weak"][1]))
        for actual, expected in zip(restored["strong"], original["strong"], strict=True):
            assert np.asarray(actual).tobytes() == expected.tobytes()
            assert np.asarray(actual).dtype == expected.dtype


@pytest.mark.parametrize("mutation", ["structure", "path", "bytes", "weak_vector"])
def test_restore_tree_rejects_changed_template_or_archive(mutation: str) -> None:
    arrays: dict[str, np.ndarray] = {}
    template = {"input": np.asarray([-0.0], dtype=np.float32)}
    description = cost.describe_tree(template, "saved", arrays)
    leaf = description["leaves"][0]
    if mutation == "structure":
        template["extra"] = np.asarray([1.0])
    elif mutation == "path":
        leaf["tree_path"] = "['different_input']"
    elif mutation == "bytes":
        arrays[leaf["array_key"]][0] = 0.0
    else:
        leaf["weak_type"] = True
    with pytest.raises(ValueError, match="structure|leaf path|transfer changed|must be scalars"):
        cost.restore_tree(template, description, arrays)


def test_protocol_mutation_is_detected_without_live_file_checks(tmp_path: Path) -> None:
    protocol = {"schema": cost.SCHEMA, "repetitions": 30}
    envelope = {"protocol": protocol, "sha256": cost.digest(protocol)}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(envelope))
    assert cost.load_protocol(path, verify_files=False) == protocol
    envelope["protocol"]["repetitions"] = 1
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="digest changed"):
        cost.load_protocol(path, verify_files=False)


def test_timers_exclude_result_description_recording_and_round_checks() -> None:
    plan = cost.balanced_plan(["a", "b"], ["c"], 2, phase="measured")
    events, saved = [], []
    now = 100.0

    def clock() -> float:
        events.append("clock")
        return now

    def invoke(method: str, case: str) -> str:
        nonlocal now
        events.append(f"call-{method}-{case}")
        now += 1.0
        return method

    def describe(result: str) -> dict:
        nonlocal now
        events.append("describe")
        now += 10.0
        return {"result": result}

    def record(row: dict) -> None:
        nonlocal now
        events.append("record")
        now += 100.0
        saved.append(row)

    def before_round(repetition: int) -> None:
        nonlocal now
        events.append(f"round-{repetition}")
        now += 1000.0

    rows = cost.run_timed_plan(
        plan,
        invoke,
        describe_result=describe,
        record=record,
        before_round=before_round,
        clock=clock,
    )
    assert [row["service_seconds"] for row in rows] == [1.0] * 4
    assert saved == rows
    assert [{key: row[key] for key in plan[0]} for row in rows] == plan
    assert rows[-1]["completed_wall_time"] - rows[0]["started_wall_time"] == 1334.0
    assert events[:6] == ["round-0", "clock", "call-a-c", "clock", "describe", "record"]
    assert events.count("round-0") == events.count("round-1") == 1


def test_failed_call_propagates_and_preserves_completed_call_prefix_in_order() -> None:
    plan = cost.balanced_plan(["a", "b", "c"], ["input"], 1, phase="measured")
    saved: list[dict[str, Any]] = []
    calls: list[tuple[str, str]] = []
    ticks = iter([1.0, 1.5, 2.0])
    failure = RuntimeError("controller call failed")

    def invoke(method: str, case: str) -> str:
        calls.append((method, case))
        if method == "b":
            raise failure
        return method

    with pytest.raises(RuntimeError) as raised:
        cost.run_timed_plan(
            plan,
            invoke,
            describe_result=lambda value: {"output": value},
            record=saved.append,
            clock=lambda: next(ticks),
        )
    assert raised.value is failure
    assert calls == [("a", "input"), ("b", "input")]
    assert len(saved) == 1
    assert saved[0]["sample_index"] == 0 and saved[0]["method"] == "a"
    assert saved[0]["service_seconds"] == 0.5


def _samples() -> tuple[list, dict]:
    methods = ["F2", "UNION", "F2_2K"]
    plan = cost.balanced_plan(methods, ["c"], 3, phase="measured")
    protocol = {
        "measurement_plan": plan,
        "cases": [{"case": "c", "time_seconds": 2.4}],
        "methods": {
            method: {"total_fallback_count": cost.FALLBACK_COUNTS[method]} for method in methods
        },
        "measured_calls_per_cell": 3,
        "comparison_scope": "common inputs only",
        "common_input_provenance": "exact archived test arrays",
        "parameter_contract": "fixed per-method test parameter states",
    }
    seconds = {"F2": 0.01, "UNION": 0.02, "F2_2K": 0.025}
    rows = [
        {
            **entry,
            "service_seconds": seconds[entry["method"]],
            "output_sha256": entry["method"],
            "execution_mode": 0,
        }
        for entry in plan
    ]
    return rows, protocol


def test_report_compares_equal_size_libraries_within_each_common_input() -> None:
    samples, protocol = _samples()
    result = cost.analyze_samples(samples, protocol)
    assert result["measured_call_count"] == 9
    comparison = next(row for row in result["comparisons"] if row["denominator"] == "F2_2K")
    assert comparison["numerator"] == "UNION"
    assert comparison["ratio_of_mean_service"] == pytest.approx(0.8)
    assert comparison["paired_service_ratio"]["count"] == 3
    assert comparison["paired_service_ratio"]["median"] == pytest.approx(0.8)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "misordered", "warmup", "nonfinite"])
def test_measurement_audit_rejects_missing_duplicate_warm_or_nonfinite_samples(
    mutation: str,
) -> None:
    samples, protocol = _samples()
    if mutation == "missing":
        samples.pop()
    elif mutation == "duplicate":
        samples[1] = deepcopy(samples[0])
    elif mutation == "misordered":
        samples[0], samples[1] = samples[1], samples[0]
    elif mutation == "warmup":
        samples[0]["phase"] = "warmup"
    else:
        samples[0]["service_seconds"] = float("nan")
    with pytest.raises(ValueError):
        cost.analyze_samples(samples, protocol)


def test_analysis_keeps_slow_outlier_in_denominator_means_and_paired_ratios() -> None:
    samples, protocol = _samples()
    selected = [row for row in samples if row["method"] == "F2"]
    selected[-1]["service_seconds"] = 1.0
    result = cost.analyze_samples(samples, protocol)
    f2 = next(row for row in result["groups"] if row["method"] == "F2")["service_seconds"]
    assert f2["count"] == 3
    assert f2["mean"] == pytest.approx(0.34)
    assert f2["median"] == pytest.approx(0.01)
    assert f2["p95"] == pytest.approx(0.901)
    assert f2["maximum"] == 1.0
    comparison = next(
        row
        for row in result["comparisons"]
        if row["numerator"] == "UNION" and row["denominator"] == "F2"
    )
    assert comparison["paired_repetition_count"] == 3
    assert comparison["ratio_of_mean_service"] == pytest.approx(0.02 / 0.34)
    assert comparison["paired_service_ratio"]["minimum"] == pytest.approx(0.02)
    assert comparison["paired_service_ratio"]["mean"] == pytest.approx(1.34)


def test_proc_stat_parser_does_not_confuse_parentheses_in_process_name() -> None:
    fields = ["S"] + ["0"] * 20
    fields[11], fields[12], fields[19] = "100", "20", "987"
    result = cost.parse_process_stat("123 (worker (one)) " + " ".join(fields))
    assert result == {
        "pid": 123,
        "comm": "worker (one)",
        "start_time_ticks": 987,
        "user_ticks": 100,
        "system_ticks": 20,
        "cpu_ticks": 120,
    }


def test_cpu_window_separates_own_and_nonself_work_and_detects_pid_reuse() -> None:
    def process(pid: int, start: int, ticks: int) -> dict:
        return {"pid": pid, "start_time_ticks": start, "comm": f"p{pid}", "cpu_ticks": ticks}

    before = {
        "monotonic_time_seconds": 10.0,
        "unix_time_seconds": 1000.0,
        "uptime_seconds": 100.0,
        "clock_ticks_per_second": 100,
        "aggregate_cpu_ticks": [100, 0, 50, 200, 0, 0, 0, 0],
        "processes": [process(1, 10, 100), process(2, 20, 50), process(3, 30, 70)],
    }
    after = {
        **before,
        "monotonic_time_seconds": 12.0,
        "unix_time_seconds": 1002.0,
        "uptime_seconds": 102.0,
        "aggregate_cpu_ticks": [200, 0, 100, 450, 0, 0, 0, 0],
        "processes": [process(1, 10, 180), process(2, 20, 70), process(3, 90, 5)],
    }
    result = cost.summarize_cpu_window(before, after, own_pid=1)
    assert result["own_cpu_seconds"] == pytest.approx(0.8)
    assert result["nonself_matched_process_cpu_seconds"] == pytest.approx(0.2)
    assert result["aggregate_busy_cpu_seconds"] == pytest.approx(1.5)
    assert result["aggregate_busy_fraction"] == pytest.approx(150 / 400)
    assert result["newly_observed_processes"][0]["start_time_ticks"] == 90
    assert result["disappeared_processes"][0]["start_time_ticks"] == 30


def test_failed_attempt_retains_completed_counts_without_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "attempt"

    def fail(_protocol: Path, target: Path) -> None:
        target.mkdir()
        (target / "measured-samples.jsonl").write_text("{}\n{}\n")
        raise RuntimeError("mock blocked occupancy")

    monkeypatch.setattr(cost, "_run_profile", fail)
    with pytest.raises(RuntimeError, match="blocked occupancy"):
        cost.run_profile(tmp_path / "protocol.json", output)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["completed_call_counts"] == {"warmup": 0, "measured": 2}
    assert failure["status"] == "failed"
