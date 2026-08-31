from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from crazyflow.safety.da_plcbf import dynamics_knowledge_campaign as campaign


def _startup_fixture(
    *, root_seed: int = 73
) -> tuple[
    campaign.DynamicsKnowledgeProfile,
    object,
    campaign.ExperimentResources,
    campaign._StartupPreparation,
]:
    profile = campaign.dynamics_knowledge_profile("smoke")
    tape = campaign.generate_matched_dynamics_tape(profile, root_seed=root_seed, fold=0)
    resources = campaign.build_matched_resources(profile, tape, root_seed=root_seed, fold=0)
    active = campaign._initial_active_snapshot(resources)
    record = {
        "status": "failed",
        "failure_type": "InjectedStartupFailure",
        "failure_message": "retained test outcome",
        "publication_accepted": False,
        "active_digest": active.digest,
        "active_params_digest": active.params_digest,
        "active_version": active.version,
        "model_version": active.model_version,
        "training_objective": campaign.TRAINING_OBJECTIVE_ID,
        "uncertainty_aware_bptt_training": False,
        "hard_admission_dynamics_samples": campaign.ADAPTATION_VALIDATION_SAMPLES,
        "execution_seconds": 0.25,
    }
    return profile, tape, resources, campaign._StartupPreparation(active, record)


def test_exact_variant_registry_and_profile_shapes() -> None:
    assert [variant.variant_id for variant in campaign.VARIANTS] == [
        "oracle_point",
        "estimated_r0",
        "estimated_cartesian_r4",
        "estimated_cartesian_r8",
    ]
    oracle, point, r4, r8 = campaign.VARIANTS
    assert oracle.privileged_oracle_upper_bound
    assert not oracle.deployable_interpretation
    assert not oracle.causal_estimator_history_only
    assert not point.privileged_oracle_upper_bound
    assert point.causal_estimator_history_only
    assert [item.runtime_dynamics_samples for item in (oracle, point, r4, r8)] == [0, 0, 4, 8]
    assert all(
        item.filter_implementation == campaign.FILTER_IMPLEMENTATION_ID
        for item in campaign.VARIANTS
    )
    assert all(not item.uncertainty_aware_bptt_training for item in campaign.VARIANTS)

    final = campaign.dynamics_knowledge_profile("final")
    assert final.trials == 100
    assert final.intended_for_confirmatory_differences
    assert (
        final.trial.policy_count,
        final.trial.certificate_horizon,
        final.trial.training_scenario_count,
    ) == (64, 50, 64)
    assert final.trial.uncertainty_sample_count == 4


def test_shortened_final_schedule_is_explicitly_demoted() -> None:
    resolved = campaign.DynamicsKnowledgeCampaignConfig(
        profile="final", trials=3
    ).resolved_profile()
    assert resolved.name == "final"
    assert resolved.trials == 3
    assert not resolved.intended_for_confirmatory_differences
    assert (
        resolved.trial.policy_count,
        resolved.trial.certificate_horizon,
        resolved.trial.training_scenario_count,
    ) == (64, 50, 64)
    with pytest.raises(ValueError, match="fold indices"):
        campaign.DynamicsKnowledgeCampaignConfig(
            fold_start=int(np.iinfo(np.uint32).max), trials=2
        ).validate()


def test_matched_tape_is_deterministic_and_changes_only_declared_dynamics_axis() -> None:
    profile = campaign.dynamics_knowledge_profile("smoke")
    first = campaign.generate_matched_dynamics_tape(profile, root_seed=41, fold=2)
    second = campaign.generate_matched_dynamics_tape(profile, root_seed=41, fold=2)
    assert first.sha256 == second.sha256
    assert np.array_equal(first.rotor_efficiency, np.ones_like(first.rotor_efficiency))
    executed = slice(0, profile.trial.control_steps)
    assert np.any(first.mass_scale[executed] != 1.0)
    assert np.any(first.drag_scale[executed] != 1.0)
    assert np.any(first.wind_velocity[executed] != 0.0)


def _synthetic_metrics(*, failure: float, minimum_margin: float) -> dict[str, float | bool]:
    return {
        "operational_failure": bool(failure),
        "executed_degraded_any": False,
        "failure_any": bool(failure),
        "contact_any": bool(failure),
        "minimum_barrier_margin": minimum_margin,
        "post_change_minimum_barrier_margin": minimum_margin - 0.1,
        "degraded_fraction": failure,
        "tracking_position_rmse": 1.0 + failure,
        "normalized_estimation_rmse": 0.5 + failure,
        "fallback_use_fraction": failure,
    }


def test_final_aggregate_is_metric_level_holm_adjusted_and_never_blanket() -> None:
    config = campaign.DynamicsKnowledgeCampaignConfig(profile="final", root_seed=9)
    profile = config.resolved_profile()
    outcomes = []
    values = {
        "oracle_point": (0.0, 2.0),
        "estimated_r0": (1.0, -1.0),
        "estimated_cartesian_r4": (0.0, 0.5),
        "estimated_cartesian_r8": (0.0, 0.75),
    }
    for fold in range(profile.trials):
        for variant in campaign.VARIANTS:
            failure, margin = values[variant.variant_id]
            outcomes.append(
                {
                    "fold": fold,
                    "variant": asdict(variant),
                    "status": "complete",
                    "operational_failure": bool(failure),
                    "metrics": _synthetic_metrics(failure=failure, minimum_margin=margin),
                }
            )
    aggregate = campaign.aggregate_dynamics_knowledge_outcomes(config, profile, outcomes)
    assert aggregate["confirmatory_metric_family_eligible"]
    assert aggregate["confirmatory_metric_family_size"] == 4
    assert not aggregate["blanket_safety_superiority_supported"]
    primary = [item for item in aggregate["comparisons"] if item["analysis_role"] == "confirmatory"]
    assert len(primary) == 4
    assert all(item["paired_count"] == 100 for item in primary)
    assert all(item["holm_adjusted_pvalue"] is not None for item in primary)
    assert all(item["metric_level_improvement_supported"] for item in primary)
    assert not any(
        item["analysis_role"] == "predeclared_primary" for item in aggregate["comparisons"]
    )
    assert all(
        not item["blanket_safety_superiority_interpretation_permitted"]
        for item in aggregate["comparisons"]
    )

    duplicated = list(outcomes)
    duplicated[-1] = dict(duplicated[3])
    invalid = campaign.aggregate_dynamics_knowledge_outcomes(config, profile, duplicated)
    assert not invalid["schedule_complete"]
    assert not invalid["confirmatory_metric_family_eligible"]
    missing = [
        item
        for item in invalid["comparisons"]
        if item["analysis_role"] == "confirmatory" and item["missing_scheduled_pairs"]
    ]
    assert missing
    assert all(not item["confirmatory_eligible"] for item in missing)
    assert all(not item["metric_level_improvement_supported"] for item in missing)


def _valid_scientific_arrays(profile: campaign.DynamicsKnowledgeProfile) -> dict[str, np.ndarray]:
    steps = profile.trial.control_steps
    policies = profile.trial.policy_count
    command_valid = np.ones(steps, dtype=bool)
    command_valid[-1] = False
    selected = np.zeros(steps, dtype=np.int32)
    selected[-1] = -1
    timings = np.full(steps, 1e-3)
    timings[-1] = 0.0
    true_parameters = np.zeros((steps, campaign.DYNAMICS_PARAMETER_COUNT))
    true_parameters[:, 7:] = 1.0
    return {
        "states": np.zeros((steps, 13)),
        "state_valid": np.ones(steps, dtype=bool),
        "command_valid": command_valid,
        "commanded_motor_forces": np.zeros((steps, 4)),
        "realized_motor_forces": np.zeros((steps, 4)),
        "nominal_motor_forces": np.zeros((steps, 4)),
        "policy_hard_values": np.zeros((steps, policies)),
        "selected_policy": selected,
        "selected_hard_value": np.zeros(steps),
        "degraded": np.zeros(steps, dtype=bool),
        "proposal_accepted": np.zeros(steps, dtype=bool),
        "fallback_accepted": np.zeros(steps, dtype=bool),
        "used_fallback": np.zeros(steps, dtype=bool),
        "applied_interval_margin": np.zeros(steps),
        "applied_next_value": np.zeros(steps),
        "applied_exact_residual": np.zeros(steps),
        "controller_seconds": timings.copy(),
        "plant_seconds": timings.copy(),
        "estimator_seconds": np.zeros(steps),
        "adaptation_seconds": np.zeros(steps),
        "translation_update_status": np.full(steps, -1, dtype=np.int16),
        "rotor_update_status": np.full(steps, -1, dtype=np.int16),
        "estimator_model_version": np.zeros(steps, dtype=np.int32),
        "snapshot_version": np.zeros(steps, dtype=np.int32),
        "model_last_observation_transition": np.arange(steps, dtype=np.int32) - 1,
        "estimator_history_count": np.arange(steps, dtype=np.int32),
        "true_parameters": true_parameters,
        "estimated_parameters": true_parameters.copy(),
        "runtime_sample_parameters": np.zeros(
            (steps, campaign.MAX_RUNTIME_SAMPLES, campaign.DYNAMICS_PARAMETER_COUNT)
        ),
        "runtime_sample_valid": np.zeros((steps, campaign.MAX_RUNTIME_SAMPLES), dtype=bool),
        "adaptation_publication_accepted": np.zeros(steps, dtype=bool),
        "barrier_margins": np.ones((steps, 1)),
        "contact": np.zeros(steps, dtype=bool),
        "failure": np.zeros(steps, dtype=bool),
    }


def test_trace_semantics_reject_future_estimator_history() -> None:
    profile = campaign.dynamics_knowledge_profile("smoke")
    tape = campaign.generate_matched_dynamics_tape(profile, root_seed=3, fold=0)
    arrays = _valid_scientific_arrays(profile)
    estimated = next(item for item in campaign.VARIANTS if item.variant_id == "estimated_r0")
    assert campaign._trace_semantic_errors(arrays, estimated, profile, tape) == ()
    arrays["model_last_observation_transition"][2] = 2
    errors = campaign._trace_semantic_errors(arrays, estimated, profile, tape)
    assert any("future or same-transition" in error for error in errors)


def test_operational_failure_includes_executed_degraded_and_adaptation_failures() -> None:
    profile = campaign.dynamics_knowledge_profile("smoke")
    tape = campaign.generate_matched_dynamics_tape(profile, root_seed=3, fold=0)
    arrays = _valid_scientific_arrays(profile)
    arrays["degraded"][1] = True
    metrics = campaign._knowledge_metrics(arrays, tape, profile)
    assert metrics["operational_failure"]
    assert metrics["executed_degraded_any"]
    assert not metrics["failure_any"]

    arrays["degraded"][:] = False
    arrays["degraded"][-1] = True
    terminal_only = campaign._knowledge_metrics(arrays, tape, profile)
    assert not terminal_only["operational_failure"]
    assert not terminal_only["executed_degraded_any"]
    assert campaign._knowledge_metrics(arrays, tape, profile, startup_adaptation_status="failed")[
        "operational_failure"
    ]
    assert campaign._knowledge_metrics(
        arrays, tape, profile, adaptation_events=({"status": "failed"},)
    )["operational_failure"]


def test_sample_range_metric_is_explicitly_coordinatewise_not_coverage() -> None:
    profile = campaign.dynamics_knowledge_profile("smoke")
    tape = campaign.generate_matched_dynamics_tape(profile, root_seed=4, fold=0)
    arrays = _valid_scientific_arrays(profile)
    arrays["runtime_sample_valid"][0, :2] = True
    arrays["runtime_sample_parameters"][0, 0] = 0.0
    arrays["runtime_sample_parameters"][0, 1] = 1.0
    metrics = campaign._knowledge_metrics(arrays, tape, profile)
    assert metrics["axis_aligned_sample_range_enclosure_fraction"] == 1.0
    assert "true_parameter_sample_enclosure_fraction" not in metrics
    definition = campaign.AXIS_ALIGNED_SAMPLE_RANGE_METRIC_DEFINITION.lower()
    assert "coordinatewise" in definition
    assert "not convex-hull" in definition
    assert "not" in definition and "coverage" in definition


def test_atomic_outcome_flush_preserves_previous_ledger_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "outcomes.jsonl"
    first = {
        "fold": 0,
        "variant": {"variant_id": "oracle_point"},
        "status": "failed",
        "operational_failure": True,
    }
    second = {
        "fold": 0,
        "variant": {"variant_id": "estimated_r0"},
        "status": "failed",
        "operational_failure": True,
    }
    campaign._flush_outcomes(path, {(0, "oracle_point"): first})
    original = path.read_bytes()

    def interrupted_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace interruption")

    monkeypatch.setattr(campaign.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="injected replace interruption"):
        campaign._flush_outcomes(path, {(0, "oracle_point"): first, (0, "estimated_r0"): second})
    assert path.read_bytes() == original
    assert campaign._read_outcomes(path) == (first,)


def test_finalized_resume_is_strict_verified_byte_preserving_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = campaign.DynamicsKnowledgeCampaignConfig(profile="smoke", root_seed=17)
    profile = config.resolved_profile()
    repository = Path.cwd()
    configuration = campaign._configuration_mapping(config, profile, repository)
    campaign._atomic_json(tmp_path / "config.json", configuration)
    campaign._atomic_json(tmp_path / "manifest.json", {"schema_version": 2})
    campaign._atomic_json(tmp_path / "complete.marker", {"committed": True})
    verification = campaign.DynamicsKnowledgeVerification(True, (), 4, 4, 3, 1, 2)
    monkeypatch.setattr(
        campaign, "verify_dynamics_knowledge_campaign", lambda *_args, **_kwargs: verification
    )
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    before_mtime = {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    result = campaign.run_dynamics_knowledge_campaign(
        config, tmp_path, repository=repository, resume=True
    )
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert before == after
    assert before_mtime == {path.name: path.stat().st_mtime_ns for path in tmp_path.iterdir()}
    assert result.execution_complete
    assert result.failed_outcomes == 1
    assert result.operational_failures == 2

    monkeypatch.setattr(
        campaign,
        "verify_dynamics_knowledge_campaign",
        lambda *_args, **_kwargs: campaign.DynamicsKnowledgeVerification(
            False, ("provenance hash mismatch",), 4, 4, 3, 1, 2
        ),
    )
    with pytest.raises(ValueError, match="failed verification"):
        campaign.run_dynamics_knowledge_campaign(
            config, tmp_path, repository=repository, resume=True
        )
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


def test_trace_semantic_digest_detects_numeric_tampering(tmp_path: Path) -> None:
    metadata = np.frombuffer(b'{"schema_version":2}\n', dtype=np.uint8).copy()
    leaf = np.asarray([1.0, 2.0], dtype=np.float32)
    arrays = {
        "metadata_json_utf8": metadata,
        "states": np.arange(6, dtype=np.float64).reshape(2, 3),
        "final_param_leaf_000": leaf,
    }
    content = campaign._canonical_array_digest("dynamics-knowledge-trace-v2", arrays)
    leaf_digest = campaign._canonical_array_digest(
        "dynamics-knowledge-final-parameter-leaves-v1", {"final_param_leaf_000": leaf}
    )
    path = tmp_path / "trace.npz"
    campaign._atomic_npz(
        path,
        {
            **arrays,
            "content_digest": np.asarray(content),
            "final_parameter_leaf_digest": np.asarray(leaf_digest),
        },
    )
    _, parsed, loaded_content, loaded_leaf = campaign._load_trace_payload(path)
    assert parsed == {"schema_version": 2}
    assert loaded_content == content
    assert loaded_leaf == leaf_digest

    tampered = dict(arrays)
    tampered["states"] = arrays["states"].copy()
    tampered["states"][0, 0] += 1.0
    campaign._atomic_npz(
        path,
        {
            **tampered,
            "content_digest": np.asarray(content),
            "final_parameter_leaf_digest": np.asarray(leaf_digest),
        },
    )
    with pytest.raises(ValueError, match="semantic content digest"):
        campaign._load_trace_payload(path)


def test_common_startup_persists_exact_active_snapshot_and_recovers_sidecar(tmp_path: Path) -> None:
    _profile, tape, resources, prepared = _startup_fixture()
    campaign._save_or_verify_startup(tmp_path, 0, tape.sha256, prepared, resources)
    sidecar_path, bundle_path = campaign._startup_paths(tmp_path, 0)
    bundle_bytes = bundle_path.read_bytes()
    bundle_mtime = bundle_path.stat().st_mtime_ns

    loaded, sidecar = campaign._load_common_startup(
        tmp_path, 0, tape.sha256, resources, recover_missing_sidecar=False
    )
    assert loaded.active.digest == prepared.active.digest
    assert loaded.active.params_digest == prepared.active.params_digest
    assert loaded.active.structural_core_digest == prepared.active.structural_core_digest
    assert loaded.record == prepared.record
    assert sidecar["active_artifact_sha256"] == campaign._file_sha256(bundle_path)

    sidecar_path.unlink()
    recovered, recovered_sidecar = campaign._load_common_startup(
        tmp_path, 0, tape.sha256, resources, recover_missing_sidecar=True
    )
    assert recovered.active.digest == prepared.active.digest
    assert recovered_sidecar == campaign._read_object(sidecar_path)
    assert bundle_path.read_bytes() == bundle_bytes
    assert bundle_path.stat().st_mtime_ns == bundle_mtime


def test_common_startup_tampering_fails_closed(tmp_path: Path) -> None:
    _profile, tape, resources, prepared = _startup_fixture(root_seed=74)
    campaign._save_or_verify_startup(tmp_path, 0, tape.sha256, prepared, resources)
    sidecar_path, bundle_path = campaign._startup_paths(tmp_path, 0)
    sidecar = campaign._read_object(sidecar_path)
    sidecar["active_artifact_sha256"] = "0" * 64
    campaign._atomic_json(sidecar_path, sidecar)
    with pytest.raises(ValueError, match="does not bind active bundle"):
        campaign._load_common_startup(
            tmp_path, 0, tape.sha256, resources, recover_missing_sidecar=False
        )

    campaign._atomic_json(
        sidecar_path,
        campaign._startup_sidecar(
            tmp_path,
            bundle_path,
            campaign._load_startup_bundle(
                bundle_path, fold=0, tape_digest=tape.sha256, resources=resources
            )[1],
            campaign._load_startup_bundle(
                bundle_path, fold=0, tape_digest=tape.sha256, resources=resources
            )[2],
        ),
    )
    with np.load(bundle_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["param_leaf_000"].flat[0] += 1.0
    campaign._atomic_npz(bundle_path, arrays)
    with pytest.raises(ValueError, match="active content digest mismatch"):
        campaign._load_common_startup(
            tmp_path, 0, tape.sha256, resources, recover_missing_sidecar=False
        )


def test_runtime_provenance_is_complete_bound_and_write_once(tmp_path: Path) -> None:
    config = campaign.DynamicsKnowledgeCampaignConfig(profile="smoke", root_seed=75)
    profile = config.resolved_profile()
    configuration = campaign._configuration_mapping(config, profile, Path.cwd())
    timings = {name: 0.125 for name in campaign._RUNTIME_TIMING_KEYS}
    compiled = SimpleNamespace(compile_seconds=timings, warmup_seconds=timings)
    provenance = campaign._runtime_provenance(compiled, configuration)
    campaign._validate_runtime_provenance(provenance, configuration)
    assert set(provenance["compile_seconds"]) == campaign._RUNTIME_TIMING_KEYS
    assert set(provenance["warmup_seconds"]) == campaign._RUNTIME_TIMING_KEYS

    path = tmp_path / "provenance.json"
    campaign._write_once_json(path, provenance)
    original = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns
    with pytest.raises(FileExistsError):
        campaign._write_once_json(path, {**provenance, "platform": "changed"})
    assert path.read_bytes() == original
    assert path.stat().st_mtime_ns == original_mtime

    tampered = copy.deepcopy(provenance)
    tampered["compile_seconds"].pop("true_plant")
    with pytest.raises(ValueError, match="compile_seconds is incomplete"):
        campaign._validate_runtime_provenance(tampered, configuration)
    tampered = copy.deepcopy(provenance)
    tampered["source_tree_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source-tree digest"):
        campaign._validate_runtime_provenance(tampered, configuration)


def test_resume_rejects_outcomes_created_without_preexecution_provenance(tmp_path: Path) -> None:
    config = campaign.DynamicsKnowledgeCampaignConfig(profile="smoke", root_seed=76)
    profile = config.resolved_profile()
    configuration = campaign._configuration_mapping(config, profile, Path.cwd())
    campaign._atomic_json(tmp_path / "config.json", configuration)
    retained = {
        "fold": 0,
        "variant": {"variant_id": "oracle_point"},
        "status": "failed",
        "operational_failure": True,
    }
    campaign._flush_outcomes(tmp_path / "outcomes.jsonl", {(0, "oracle_point"): retained})
    with pytest.raises(ValueError, match="lack the immutable pre-execution runtime provenance"):
        campaign.run_dynamics_knowledge_campaign(config, tmp_path, repository=Path.cwd())


def test_interrupted_runner_reuses_startup_and_preserves_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = campaign.DynamicsKnowledgeCampaignConfig(profile="smoke", root_seed=77)
    profile = config.resolved_profile()
    tape = campaign.generate_matched_dynamics_tape(profile, root_seed=77, fold=0)
    resources = campaign.build_matched_resources(profile, tape, root_seed=77, fold=0)
    active = campaign._initial_active_snapshot(resources)
    startup = campaign._StartupPreparation(
        active,
        {
            "status": "failed",
            "failure_type": "RetainedStartupFailure",
            "failure_message": "simulated interruption fixture",
            "publication_accepted": False,
            "active_digest": active.digest,
            "active_params_digest": active.params_digest,
            "active_version": active.version,
            "model_version": active.model_version,
            "training_objective": campaign.TRAINING_OBJECTIVE_ID,
            "uncertainty_aware_bptt_training": False,
            "hard_admission_dynamics_samples": campaign.ADAPTATION_VALIDATION_SAMPLES,
            "execution_seconds": 0.25,
        },
    )
    configuration = campaign._configuration_mapping(config, profile, Path.cwd())
    campaign._atomic_json(tmp_path / "config.json", configuration)
    campaign._save_or_verify_startup(tmp_path, 0, tape.sha256, startup, resources)
    timings = {name: 0.125 for name in campaign._RUNTIME_TIMING_KEYS}
    compiled = SimpleNamespace(compile_seconds=timings, warmup_seconds=timings)
    provenance = campaign._runtime_provenance(compiled, configuration)
    campaign._write_once_json(tmp_path / "provenance.json", provenance)
    startup_files = campaign._startup_paths(tmp_path, 0)
    immutable_paths = (*startup_files, tmp_path / "provenance.json")
    immutable_bytes = {path: path.read_bytes() for path in immutable_paths}
    immutable_mtimes = {path: path.stat().st_mtime_ns for path in immutable_paths}

    monkeypatch.setattr(
        campaign,
        "prepare_common_startup_adaptation",
        lambda *_args, **_kwargs: pytest.fail("resume reran common BPTT startup"),
    )
    monkeypatch.setattr(
        campaign, "compile_knowledge_executables", lambda *_args, **_kwargs: compiled
    )
    monkeypatch.setattr(
        campaign.experiment_core, "_build_bptt_executable_pool", lambda *_args, **_kwargs: object()
    )

    def fail_trial(**_kwargs: object) -> None:
        raise RuntimeError("retained synthetic trial failure")

    monkeypatch.setattr(campaign, "execute_dynamics_knowledge_trial", fail_trial)
    first = campaign.run_dynamics_knowledge_campaign(
        config, tmp_path, repository=Path.cwd(), resume=True
    )
    assert first.execution_complete
    assert first.failed_outcomes == len(campaign.VARIANTS)
    assert all(path.read_bytes() == immutable_bytes[path] for path in immutable_paths)
    assert all(path.stat().st_mtime_ns == immutable_mtimes[path] for path in immutable_paths)

    retained_outcomes = (tmp_path / "outcomes.jsonl").read_bytes()
    retained_outcomes_mtime = (tmp_path / "outcomes.jsonl").stat().st_mtime_ns
    (tmp_path / "complete.marker").unlink()
    (tmp_path / "manifest.json").unlink()
    (tmp_path / "aggregates.json").unlink()
    monkeypatch.setattr(
        campaign,
        "compile_knowledge_executables",
        lambda *_args, **_kwargs: pytest.fail("all-outcomes resume recompiled executables"),
    )
    second = campaign.run_dynamics_knowledge_campaign(
        config, tmp_path, repository=Path.cwd(), resume=True
    )
    assert second.execution_complete
    assert second.failed_outcomes == len(campaign.VARIANTS)
    assert (tmp_path / "outcomes.jsonl").read_bytes() == retained_outcomes
    assert (tmp_path / "outcomes.jsonl").stat().st_mtime_ns == retained_outcomes_mtime
    assert all(path.read_bytes() == immutable_bytes[path] for path in immutable_paths)
    assert all(path.stat().st_mtime_ns == immutable_mtimes[path] for path in immutable_paths)


def test_source_tree_digest_binds_runtime_params_but_ignores_docs_and_tests(tmp_path: Path) -> None:
    (tmp_path / "crazyflow" / "drones").mkdir(parents=True)
    (tmp_path / "examples" / "da_plcbf").mkdir(parents=True)
    (tmp_path / "benchmark").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "crazyflow" / "runtime.py").write_text("VALUE = 1\n")
    params = tmp_path / "crazyflow" / "drones" / "params.toml"
    params.write_text("mass = 0.5\n")
    (tmp_path / "examples" / "da_plcbf" / "run.py").write_text("pass\n")
    (tmp_path / "benchmark" / "bench.py").write_text("pass\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (tmp_path / "pixi.lock").write_text("lock\n")
    baseline = campaign.source_tree_digest(tmp_path)

    (tmp_path / "DA_PLCBF_PLAN.md").write_text("changed plan\n")
    (tmp_path / "docs" / "guide.md").write_text("changed docs\n")
    (tmp_path / "tests" / "test_runtime.py").write_text("changed tests\n")
    assert campaign.source_tree_digest(tmp_path) == baseline

    params.write_text("mass = 0.6\n")
    assert campaign.source_tree_digest(tmp_path) != baseline
