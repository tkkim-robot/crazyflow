from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from crazyflow.safety.da_plcbf import version_b_evidence as evidence_module
from crazyflow.safety.da_plcbf.version_b_evidence import (
    CORE_CAMPAIGN_SUBSTEPS,
    FIXED_CASE_COUNT,
    INTENDED_CERTIFICATE_HORIZON,
    INTENDED_POLICY_COUNT,
    MINIMUM_FINAL_RANDOMIZED_CASES,
    SHORT_INTERVAL_SUBSTEPS,
    comparison_profile,
    generate_matched_version_cases,
    load_version_comparison_artifact,
    matched_case_set_sha256,
    render_version_comparison_report,
    run_matched_version_comparison,
    save_version_comparison_artifact,
    validate_version_comparison_artifact,
)

_CLI_PATH = Path(__file__).resolve().parents[4] / "examples" / "da_plcbf" / "version_b_evidence.py"
_CLI_SPEC = importlib.util.spec_from_file_location("da_plcbf_version_b_evidence_cli", _CLI_PATH)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
cli_module = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(cli_module)


def test_profiles_predeclare_three_fixed_and_one_hundred_final_randomized_cases() -> None:
    smoke = comparison_profile("smoke", root_seed=17)
    final = comparison_profile("final", root_seed=17)
    smoke_cases = generate_matched_version_cases(smoke)
    final_cases = generate_matched_version_cases(final)

    assert len(smoke_cases) == FIXED_CASE_COUNT
    assert len(final_cases) == FIXED_CASE_COUNT + MINIMUM_FINAL_RANDOMIZED_CASES
    assert final.randomized_case_count == 100
    assert smoke.n_substeps == SHORT_INTERVAL_SUBSTEPS
    assert final.n_substeps == CORE_CAMPAIGN_SUBSTEPS
    assert [case.case_id for case in final_cases[:FIXED_CASE_COUNT]] == [
        "fixed-safe-airborne",
        "fixed-near-obstacle-approach",
        "fixed-colliding-initial-state",
    ]
    assert all(case.source == "randomized" for case in final_cases[FIXED_CASE_COUNT:])
    assert matched_case_set_sha256(final_cases) == matched_case_set_sha256(
        generate_matched_version_cases(final)
    )
    assert matched_case_set_sha256(final_cases) != matched_case_set_sha256(
        generate_matched_version_cases(comparison_profile("final", root_seed=18))
    )


@pytest.mark.integration
def test_real_cpu_smoke_is_strict_content_addressed_and_keeps_fail_closed_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_states: list[object] = []
    capture_source_state = evidence_module._source_state

    def record_source_state(repository: Path) -> object:
        state = capture_source_state(repository)
        source_states.append(state)
        return state

    monkeypatch.setattr(evidence_module, "_source_state", record_source_state)
    artifact = run_matched_version_comparison(
        comparison_profile("smoke", root_seed=23), device="cpu"
    )

    assert len(source_states) == 2
    assert source_states[0] == source_states[1]
    validate_version_comparison_artifact(artifact)
    assert artifact["summary"]["scheduled_cases"] == 3
    assert artifact["summary"]["version_a"]["operational_failures"] == 0
    assert artifact["summary"]["version_b"]["operational_failures"] == 0
    assert not artifact["full_shape_preflight"]["scheduled"]
    assert artifact["full_shape_preflight"]["matched_inputs"] is None
    assert not artifact["full_shape_preflight"]["matched_acceptance_postcheck_passed"]
    assert not artifact["full_shape_preflight"]["version_b_integration_supported"]
    assert artifact["full_shape_preflight"]["n_substeps"] == CORE_CAMPAIGN_SUBSTEPS
    assert artifact["full_shape_preflight"]["decision_deadline_seconds"] == 0.02
    short = artifact["short_interval_shape_probe"]
    assert not short["scheduled"]
    assert short["n_substeps"] == SHORT_INTERVAL_SUBSTEPS
    assert short["decision_deadline_seconds"] == 0.004
    collision = artifact["cases"][2]
    assert collision["matched_inputs"]["condition"] == "colliding_initial_state"
    assert not collision["version_a"]["claim_eligible"]
    assert not collision["version_b"]["claim_eligible"]
    assert collision["version_a"]["degraded"]
    assert collision["version_b"]["degraded"]

    path = tmp_path / "matched.json"
    digest = save_version_comparison_artifact(artifact, path)
    loaded = load_version_comparison_artifact(path)
    assert loaded["content_sha256"] == digest
    report = render_version_comparison_report(loaded)
    assert digest in report
    assert "does **not** transfer" in report
    assert "one `cpu` run" in report
    assert "realtime operation was not assessed" in report
    assert "realtime support: **supported**" not in report.lower()

    tampered = copy.deepcopy(loaded)
    tampered["cases"][0]["version_b"]["applied_exact_residual"] = 123.0
    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        validate_version_comparison_artifact(tampered)

    with monkeypatch.context() as source_change:
        source_change.setattr(evidence_module, "_source_tree_sha256", lambda _root: "0" * 64)
        with pytest.raises(ValueError, match="current source tree differs"):
            load_version_comparison_artifact(path)
        blocked_path = tmp_path / "blocked-after-source-drift.json"
        with pytest.raises(ValueError, match="current source tree differs"):
            save_version_comparison_artifact(artifact, blocked_path)
        assert not blocked_path.exists()
        historical = load_version_comparison_artifact(path, require_current_source=False)
        assert historical["content_sha256"] == digest


def test_source_guards_fail_closed_for_dirty_unavailable_and_changed_state() -> None:
    clean = evidence_module._SourceState("a" * 64, "b" * 40, "")
    evidence_module._require_clean_source(clean)
    evidence_module._require_unchanged_source(clean, clean)

    for status in (" M runtime.py", None):
        state = evidence_module._SourceState("a" * 64, "b" * 40, status)
        with pytest.raises(RuntimeError, match="requires a clean source tree"):
            evidence_module._require_clean_source(state)

    changed = evidence_module._SourceState("c" * 64, "b" * 40, "")
    with pytest.raises(RuntimeError, match="source/git state changed"):
        evidence_module._require_unchanged_source(clean, changed)


def test_cli_strict_verify_passes_current_and_clean_source_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "matched.json"
    calls: list[tuple[Path, Path, bool, bool]] = []

    def load(
        path: Path, *, repository: Path, require_current_source: bool, require_clean_source: bool
    ) -> dict[str, object]:
        calls.append((path, repository, require_current_source, require_clean_source))
        return {
            "content_sha256": "a" * 64,
            "protocol": {"profile": "final"},
            "summary": {"scheduled_cases": 103},
        }

    monkeypatch.setattr(cli_module, "load_version_comparison_artifact", load)
    cli_module.main(["--verify-artifact", str(source), "--require-clean-source"])

    assert calls == [(source.resolve(), cli_module.REPOSITORY, True, True)]
    report = json.loads(capsys.readouterr().out)
    assert report["current_source_verified"] is True
    assert report["clean_source_required"] is True
    assert report["scheduled_cases"] == 103


def test_cli_source_drift_failure_emits_no_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "matched.json"
    report = tmp_path / "matched.md"
    saves: list[Path] = []

    monkeypatch.setattr(cli_module, "comparison_profile", lambda *_args, **_kwargs: object())

    def fail_run(*_args: object, **kwargs: object) -> dict[str, object]:
        assert kwargs["require_clean_source"] is True
        raise RuntimeError("source/git state changed while Version A/B evidence was executing")

    monkeypatch.setattr(cli_module, "run_matched_version_comparison", fail_run)
    monkeypatch.setattr(
        cli_module,
        "save_version_comparison_artifact",
        lambda _artifact, path, **_kwargs: saves.append(path),
    )
    monkeypatch.setattr(
        cli_module,
        "save_version_comparison_report",
        lambda _artifact, path, **_kwargs: saves.append(path),
    )

    with pytest.raises(RuntimeError, match="source/git state changed"):
        cli_module.main(
            [
                "--profile",
                "final",
                "--device",
                "gpu",
                "--output",
                str(output),
                "--report",
                str(report),
                "--require-clean-source",
            ]
        )
    assert saves == []
    assert not output.exists()
    assert not report.exists()


def test_source_tree_digest_binds_runtime_assets_and_excludes_plan_docs_tests(
    tmp_path: Path,
) -> None:
    (tmp_path / "crazyflow" / "drones").mkdir(parents=True)
    (tmp_path / "examples" / "da_plcbf").mkdir(parents=True)
    (tmp_path / "benchmark").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "crazyflow" / "runtime.py").write_text("VALUE = 1\n")
    params = tmp_path / "crazyflow" / "drones" / "params.toml"
    params.write_text("mass = 0.5\n")
    (tmp_path / "crazyflow" / "drones" / "model.xml").write_text("<model/>\n")
    (tmp_path / "examples" / "da_plcbf" / "run.py").write_text("pass\n")
    (tmp_path / "benchmark" / "bench.py").write_text("pass\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (tmp_path / "pixi.lock").write_text("lock\n")
    baseline = evidence_module._source_tree_sha256(tmp_path)

    (tmp_path / "DA_PLCBF_PLAN.md").write_text("changed plan\n")
    (tmp_path / "docs" / "guide.md").write_text("changed docs\n")
    (tmp_path / "tests" / "test_runtime.py").write_text("changed tests\n")
    assert evidence_module._source_tree_sha256(tmp_path) == baseline
    evidence_module._require_current_source_digest(baseline, tmp_path)

    params.write_text("mass = 0.6\n")
    assert evidence_module._source_tree_sha256(tmp_path) != baseline
    with pytest.raises(ValueError, match="current source tree differs"):
        evidence_module._require_current_source_digest(baseline, tmp_path)


def test_final_preflight_specs_bind_exact_core_and_short_diagnostic_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = comparison_profile("final", root_seed=31)
    core = evidence_module._intended_shape_profile(profile, n_substeps=CORE_CAMPAIGN_SUBSTEPS)
    short = evidence_module._intended_shape_profile(profile, n_substeps=SHORT_INTERVAL_SUBSTEPS)
    assert (core.policy_count, core.certificate_horizon, core.n_substeps) == (
        INTENDED_POLICY_COUNT,
        INTENDED_CERTIFICATE_HORIZON,
        10,
    )
    assert core.n_substeps / 500.0 == 0.02
    assert (short.policy_count, short.certificate_horizon, short.n_substeps) == (
        INTENDED_POLICY_COUNT,
        INTENDED_CERTIFICATE_HORIZON,
        2,
    )
    assert short.n_substeps / 500.0 == 0.004

    calls: list[tuple[str, int]] = []

    def record_route(
        candidate: object, *, device: str, role: str, n_substeps: int
    ) -> dict[str, object]:
        assert candidate is profile
        assert device == "gpu"
        calls.append((role, n_substeps))
        return {"role": role, "n_substeps": n_substeps}

    monkeypatch.setattr(evidence_module, "_run_shape_preflight", record_route)
    evidence_module._run_full_shape_preflight(profile, device="gpu")
    evidence_module._run_short_interval_shape_probe(profile, device="gpu")
    assert calls == [
        ("core_campaign_intended_shape", CORE_CAMPAIGN_SUBSTEPS),
        ("short_interval_diagnostic", SHORT_INTERVAL_SUBSTEPS),
    ]


def test_version_b_command_ready_sync_waits_for_applied_acceptance_and_finite_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronized: list[object] = []

    def record(value: object) -> object:
        synchronized.append(value)
        return value

    monkeypatch.setattr(evidence_module.jax, "block_until_ready", record)
    result = SimpleNamespace(
        action=jnp.ones((4,)), applied_accepted=jnp.asarray(True), degraded=jnp.asarray(False)
    )
    accepted, degraded = evidence_module._synchronize_command_ready(result, version="b")

    assert accepted
    assert not degraded
    assert len(synchronized) == 1
    action, acceptance, synchronized_degraded = synchronized[0]
    assert action is result.action
    assert acceptance is result.applied_accepted
    assert synchronized_degraded is result.degraded

    invalid = SimpleNamespace(
        action=jnp.asarray((jnp.nan, 0.0, 0.0, 0.0)),
        applied_accepted=jnp.asarray(False),
        degraded=jnp.asarray(True),
    )
    with pytest.raises(ValueError, match="finite command-ready wrench"):
        evidence_module._synchronize_command_ready(invalid, version="b")
