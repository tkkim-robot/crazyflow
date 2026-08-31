from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _load_benchmark() -> ModuleType:
    path = Path(__file__).resolve().parents[4] / "benchmark" / "da_plcbf.py"
    spec = importlib.util.spec_from_file_location("crazyflow_da_plcbf_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark()
PRESETS = benchmark.PRESETS
CONTENTION_SHAPES = benchmark.CONTENTION_SHAPES
BenchmarkSettings = benchmark.BenchmarkSettings
ShapePoint = benchmark.ShapePoint
_run_component = benchmark._run_component
main = benchmark.main
run_benchmark = benchmark.run_benchmark
summarize_timings = benchmark.summarize_timings


def _settings(**updates: object) -> BenchmarkSettings:
    values = {
        "repeats": 3,
        "warmups": 0,
        "deadline_seconds": 0.002,
        "max_estimated_bytes": 2 * 1024**3,
    }
    values.update(updates)
    return BenchmarkSettings(**values)


def test_timing_summary_retains_raw_samples_tails_and_strict_deadline_misses() -> None:
    result = summarize_timings([0.001, 0.002, 0.004, 0.008], 0.002)

    assert result["raw_seconds"] == [0.001, 0.002, 0.004, 0.008]
    assert result["median_seconds"] == pytest.approx(0.003)
    assert result["p95_seconds"] == pytest.approx(0.0074)
    assert result["p99_seconds"] == pytest.approx(0.00788)
    assert result["worst_seconds"] == 0.008
    assert result["deadline_misses"] == 2
    assert result["deadline_miss_fraction"] == 0.5


def test_named_presets_include_smoke_scale_and_independent_final_shape_probes() -> None:
    assert PRESETS["smoke"] == (ShapePoint("smoke", 2, 1, 4, 2),)
    scale = PRESETS["scale"]
    assert len({point.policies for point in scale}) > 1
    assert len({point.scenarios for point in scale}) > 1
    assert {point.uncertainty_samples for point in scale} == {4, 8}
    assert len({point.horizon for point in scale}) > 1
    final = PRESETS["final"]
    assert any(point.policies == 64 for point in final)
    assert any(point.scenarios == 64 for point in final)
    assert any(point.uncertainty_samples == 8 for point in final)
    assert any(point.horizon == 50 for point in final)
    assert ShapePoint("final-joint-k64-b64-r8-h50", 64, 64, 8, 50) in final
    for point in (*PRESETS["smoke"], *scale, *final):
        point.validate()


def test_contention_shapes_are_explicit_and_not_the_first_final_probe() -> None:
    assert CONTENTION_SHAPES == {
        "smoke": ShapePoint("smoke-contention-k2-b1-r4-h2", 2, 1, 4, 2),
        "scale": ShapePoint("scale-contention-k16-b16-r4-h20", 16, 16, 4, 20),
        "final": ShapePoint("final-contention-k64-b64-r8-h50", 64, 64, 8, 50),
    }
    assert CONTENTION_SHAPES["final"] != PRESETS["final"][0]
    for point in CONTENTION_SHAPES.values():
        point.validate()


def test_precompile_memory_guard_is_explicit_and_does_not_construct_problem() -> None:
    shape = ShapePoint("guard", 64, 64, 8, 50)
    settings = _settings(max_estimated_bytes=1)

    record, compiled = _run_component(
        "uncertain_rollout", shape, None, settings, "cpu", jax.devices("cpu")[0]
    )

    assert compiled is None
    assert record["status"] == "skipped"
    assert "memory" in record["reason"]
    assert record["estimated_live_bytes"] > record["memory_guard_bytes"]
    assert record["timing"] is None


def test_host_validation_smoke_is_json_strict_and_reproducible_from_request() -> None:
    result = run_benchmark(
        device_name="cpu",
        preset_name="smoke",
        components=("validation",),
        settings=_settings(),
        contention="none",
        command_arguments=["--preset", "smoke"],
    )

    encoded = json.dumps(result, allow_nan=False, sort_keys=True)
    restored = json.loads(encoded)
    measurement = restored["measurements"][0]
    assert restored["schema"] == benchmark.PERFORMANCE_ARTIFACT_SCHEMA
    assert restored["schema_version"] == benchmark.PERFORMANCE_ARTIFACT_SCHEMA_VERSION
    assert restored["request"]["preset"] == "smoke"
    assert measurement["status"] == "ok"
    assert measurement["component"] == "hard_candidate_admission_gate_only"
    assert measurement["execution_kind"] == "host_numpy_not_jittable"
    assert measurement["compile_seconds"] is None
    assert "candidate_rollouts" in measurement["scope"]["excludes"]
    assert "trajectory_descriptor_generation" in measurement["scope"]["excludes"]
    assert "campaign event traces" in measurement["scope"]["end_to_end_timing_source"]
    assert len(measurement["timing"]["raw_seconds"]) == 3
    assert measurement["report_passed"]
    assert measurement["report_integrity"]
    assert measurement["correctness"]["passed"]
    assert measurement["requested_shape"] == measurement["shape"]
    assert measurement["effective_shape"]["horizon"] is None
    assert restored["completion"]["all_requested_measurements_ok"]
    assert benchmark.verify_performance_artifact(restored)["valid"]


def test_cli_writes_the_same_canonical_json_that_it_prints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "benchmark.json"
    return_code = main(
        [
            "--device",
            "cpu",
            "--preset",
            "smoke",
            "--components",
            "validation",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--contention",
            "none",
            "--output",
            str(output),
        ]
    )

    printed = capsys.readouterr().out
    assert return_code == 0
    assert printed == output.read_text(encoding="utf-8")
    assert json.loads(printed)["measurements"][0]["status"] == "ok"


def test_performance_artifact_is_write_once(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "benchmark.json"
    document = run_benchmark(
        device_name="cpu",
        preset_name="smoke",
        components=("validation",),
        settings=_settings(repeats=1),
        contention="none",
    )

    benchmark._write_json(document, str(output))
    benchmark._write_json(document, str(output))
    changed = json.loads(json.dumps(document))
    changed["claim_caveats"].append("A distinct but internally valid caveat.")
    unsigned = {key: value for key, value in changed.items() if key != "integrity"}
    changed["integrity"]["digest"] = benchmark._canonical_digest(unsigned)
    with pytest.raises(FileExistsError, match="differs"):
        benchmark._write_json(changed, str(output))


def test_verifier_rejects_rehashed_raw_summary_and_status_fabrication() -> None:
    document = run_benchmark(
        device_name="cpu",
        preset_name="smoke",
        components=("validation",),
        settings=_settings(repeats=3),
        contention="none",
    )
    timing_tamper = json.loads(json.dumps(document))
    timing_tamper["measurements"][0]["timing"]["raw_seconds"][0] *= 0.5
    unsigned = {key: value for key, value in timing_tamper.items() if key != "integrity"}
    timing_tamper["integrity"]["digest"] = benchmark._canonical_digest(unsigned)
    with pytest.raises(ValueError, match="inconsistent with retained raw samples"):
        benchmark.verify_performance_artifact(timing_tamper)

    status_tamper = json.loads(json.dumps(document))
    status_tamper["measurements"][0]["correctness"]["report_integrity"] = False
    status_tamper["measurements"][0]["correctness"]["passed"] = False
    unsigned = {key: value for key, value in status_tamper.items() if key != "integrity"}
    status_tamper["integrity"]["digest"] = benchmark._canonical_digest(unsigned)
    with pytest.raises(ValueError, match="cannot count failed correctness as ok"):
        benchmark.verify_performance_artifact(status_tamper)

    extra_field = json.loads(json.dumps(document))
    extra_field["measurements"][0]["untrusted_note"] = "rehash should not bless schema drift"
    unsigned = {key: value for key, value in extra_field.items() if key != "integrity"}
    extra_field["integrity"]["digest"] = benchmark._canonical_digest(unsigned)
    with pytest.raises(ValueError, match="keys differ"):
        benchmark.verify_performance_artifact(extra_field)

    correctness_extra = json.loads(json.dumps(document))
    correctness_extra["measurements"][0]["correctness"]["self_reported_safe"] = True
    unsigned = {key: value for key, value in correctness_extra.items() if key != "integrity"}
    correctness_extra["integrity"]["digest"] = benchmark._canonical_digest(unsigned)
    with pytest.raises(ValueError, match="validation.correctness keys differ"):
        benchmark.verify_performance_artifact(correctness_extra)
