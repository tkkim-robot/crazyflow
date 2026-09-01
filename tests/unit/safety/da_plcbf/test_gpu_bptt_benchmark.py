"""Focused tests for the campaign-faithful GPU BPTT benchmark harness."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_benchmark() -> ModuleType:
    path = Path(__file__).resolve().parents[4] / "benchmark" / "da_plcbf_gpu_bptt.py"
    spec = importlib.util.spec_from_file_location("crazyflow_gpu_bptt_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark()
BPTTBenchmarkShape = benchmark.BPTTBenchmarkShape


def test_reference_shape_comes_from_final_campaign_defaults() -> None:
    assert BPTTBenchmarkShape.campaign_final() == BPTTBenchmarkShape(
        policies=64, batch=64, horizon=50, obstacles=8, burst_steps=10
    )


def test_tiny_cpu_burst_is_compiled_timed_and_checked_without_final_relabeling() -> None:
    tiny = BPTTBenchmarkShape(policies=9, batch=1, horizon=1, obstacles=2, burst_steps=2)

    result = benchmark.run_benchmark(
        device_name="cpu", shape=tiny, repeats=1, warmups=0, deadline_seconds=10.0
    )

    assert result["status"] == "ok"
    assert result["campaign_reference_shape"] == {
        "policies": 64,
        "batch": 64,
        "horizon": 50,
        "obstacles": 8,
        "burst_steps": 10,
    }
    assert (
        result["requested_shape"]
        == result["effective_shape"]
        == {"policies": 9, "batch": 1, "horizon": 1, "obstacles": 2, "burst_steps": 2}
    )
    assert not result["uses_exact_campaign_final_shape"]
    assert set(result["shape_overrides_from_campaign_final"]) == {
        "policies",
        "batch",
        "horizon",
        "obstacles",
        "burst_steps",
    }
    assert result["runtime_model_argument"]["supplied_to_compiled_burst"]
    assert result["correctness"]["passed"]
    assert result["correctness"]["all_updates_accepted"]
    assert result["correctness"]["optimizer_step_delta"] == 2
    assert result["correctness"]["any_gradient_norm_nonzero"]
    assert result["correctness"]["any_parameter_delta_norm_nonzero"]
    assert len(result["phases"]["timed_synchronized"]["raw_seconds"]) == 1
    assert result["phases"]["lowering_and_compilation_seconds"] > 0
    json.dumps(result, allow_nan=False, sort_keys=True)
