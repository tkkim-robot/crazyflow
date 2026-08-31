from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from crazyflow.safety.da_plcbf.adaptation_evidence import validate_adaptation_evidence_binding
from crazyflow.safety.da_plcbf.artifacts import (
    collect_provenance,
    load_events,
    load_metrics,
    load_timing,
    load_trace,
    save_trace,
)
from crazyflow.safety.da_plcbf.campaign_artifacts import (
    CampaignArtifactStore,
    _source_tree_sha256,
    _validate_trace_physical_evidence,
    validate_current_source_tree,
    validate_persisted_campaign_evidence,
)
from crazyflow.safety.da_plcbf.dashboard_evidence import validate_dashboard_evidence_binding
from crazyflow.safety.da_plcbf.experiments import (
    REQUIRED_CONDITIONS,
    CampaignConfig,
    ConditionID,
    ExperimentConfig,
    _auxiliary_tape,
    _barrier_trace,
    _causal_history_indices,
    _estimator_history_entry,
    _finite_policy_evidence,
    _global_confirmatory_superiority_supported,
    _offline_training_batch,
    _resources_for_tape,
    _used_online_adaptation_versions,
    build_experiment_resources,
    generate_condition_tape,
    replay_dashboard_dynamics_evidence,
    run_campaign,
    run_trial,
    save_trial_run,
    scenario_config_for_condition,
)
from crazyflow.safety.da_plcbf.scientific_dashboard import change_annotations
from crazyflow.safety.da_plcbf.scientific_evaluation import AnalysisRole, make_paired_trial_schedule


def _small_config() -> ExperimentConfig:
    return ExperimentConfig(
        control_steps=3,
        certificate_horizon=1,
        policy_count=8,
        prediction_samples=4,
        training_scenario_count=2,
        bptt_burst_steps=1,
        adaptation_interval_steps=2,
        estimator_interval_steps=2,
        estimator_window_steps=3,
    )


def _assignments(methods: tuple[str, ...], condition: str = "static") -> tuple[object, ...]:
    return make_paired_trial_schedule(
        root_seed=19, methods=methods, conditions=(condition,), trials_per_condition=1
    ).assignments


def test_required_condition_configs_isolate_dynamic_obstacle_families() -> None:
    config = _small_config()
    static = scenario_config_for_condition(ConditionID.STATIC, config)
    dynamics = scenario_config_for_condition(ConditionID.DYNAMICS_CHANGE, config)
    ball = scenario_config_for_condition(ConditionID.BALLISTIC_BALL, config)
    interceptor = scenario_config_for_condition(ConditionID.INTERCEPTOR_DRONE, config)
    combined = scenario_config_for_condition(ConditionID.FALSIFICATION_COMBINED, config)

    assert REQUIRED_CONDITIONS == (
        "static",
        "dynamics_change",
        "ballistic_ball",
        "interceptor_drone",
    )
    assert static.ballistic_count == static.interceptor_count == 0
    assert dynamics.ballistic_count == dynamics.interceptor_count == 0
    assert ball.ballistic_count == 2 and ball.interceptor_count == 0
    assert interceptor.interceptor_count == 2 and interceptor.ballistic_count == 0
    assert combined.crossing_count == 1
    assert combined.ballistic_count == combined.interceptor_count == 0
    assert {item.steps for item in (static, dynamics, ball, interceptor, combined)} == {
        config.control_steps + config.certificate_horizon + 1
    }


def test_final_dynamics_changes_are_indexed_on_executed_not_lookahead_horizon() -> None:
    config = ExperimentConfig.final_defaults()
    tape = generate_condition_tape(ConditionID.DYNAMICS_CHANGE, config, seed=4, fold=2)

    np.testing.assert_array_equal(tape.schedule_change_indices, (30, 53, 75, 98, 120))
    assert np.all(tape.schedule_change_indices <= config.control_steps - 2)
    assert config.control_steps - 1 - int(tape.schedule_change_indices[-1]) == 30


def test_final_configuration_predeclares_k64_h50_and_one_hundred_pairs() -> None:
    campaign = CampaignConfig.final_core(root_seed=7)
    schedule = campaign.schedule()

    assert campaign.trial.policy_count == 64
    assert campaign.trial.certificate_horizon == 50
    assert campaign.trial.uncertainty_sample_count == 4
    assert campaign.trials_per_condition == 100
    assert schedule.final_claim_eligible
    assert len(schedule.assignments) == 7 * 4 * 100


def test_final_contract_rejects_subsets_and_shape_changes() -> None:
    final = CampaignConfig.final_core(root_seed=7)
    subset = CampaignConfig(
        trial=final.trial,
        methods=final.methods[:-1],
        conditions=final.conditions,
        trials_per_condition=100,
        intended_for_final_claim=True,
    )
    changed = CampaignConfig(
        trial=ExperimentConfig(control_steps=150),
        methods=final.methods,
        conditions=final.conditions,
        trials_per_condition=100,
        intended_for_final_claim=True,
    )

    assert any("seven ordered" in blocker for blocker in subset.final_contract_blockers())
    assert any("differs from final" in blocker for blocker in changed.final_contract_blockers())


def test_final_intended_campaign_requires_clean_committed_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development = CampaignConfig(
        trial=_small_config(),
        methods=("nominal_only", "analytic_cbf_hocbf"),
        conditions=("static",),
        trials_per_condition=1,
        root_seed=27,
        intended_for_final_claim=False,
    )
    repository = Path(__file__).resolve().parents[4]
    dirty = collect_provenance(repository)
    dirty["git"] = {**dirty["git"], "dirty": True}
    monkeypatch.setattr(
        "crazyflow.safety.da_plcbf.campaign_artifacts.collect_provenance", lambda _path: dirty
    )
    schedule = development.schedule()
    assignment = schedule.assignments[0]
    tape = generate_condition_tape(
        assignment.condition,
        development.trial,
        seed=assignment.scenario_root_seed,
        fold=assignment.scenario_fold,
    )
    tapes = {(assignment.condition, assignment.fold): tape}
    CampaignArtifactStore(
        tmp_path / "development", development, schedule, tapes, repository=repository, resume=False
    )
    assert json.loads((tmp_path / "development" / "provenance.json").read_text())["git"]["dirty"]

    final_intended = replace(development, trials_per_condition=100, intended_for_final_claim=True)
    final_schedule = final_intended.schedule()
    final_tapes = {}
    for final_assignment in final_schedule.assignments:
        if final_assignment.pair_key not in final_tapes:
            final_tapes[final_assignment.pair_key] = generate_condition_tape(
                final_assignment.condition,
                final_intended.trial,
                seed=final_assignment.scenario_root_seed,
                fold=final_assignment.scenario_fold,
            )
    with pytest.raises(ValueError, match="clean committed"):
        CampaignArtifactStore(
            tmp_path / "final",
            final_intended,
            final_schedule,
            final_tapes,
            repository=repository,
            resume=False,
        )
    assert not (tmp_path / "final").exists()


def test_online_history_is_causal_but_offline_batch_spans_auxiliary_tape() -> None:
    config = _small_config()
    tape = generate_condition_tape(ConditionID.DYNAMICS_CHANGE, config, seed=101, fold=3)
    online = _causal_history_indices(2, 8)
    _, _, offline = _offline_training_batch(tape, config)

    assert np.all(online <= 2)
    assert np.min(offline) == 0
    assert np.max(offline) == tape.steps - 1
    assert np.unique(offline).size == config.training_scenario_count


def test_auxiliary_tape_is_predeclared_independently_of_evaluation_fold() -> None:
    config = _small_config()
    first = generate_condition_tape(ConditionID.STATIC, config, seed=1, fold=1)
    second = generate_condition_tape(ConditionID.STATIC, config, seed=2, fold=2)
    first_aux = _auxiliary_tape(first, ConditionID.STATIC, config, purpose="hard-validation")
    second_aux = _auxiliary_tape(second, ConditionID.STATIC, config, purpose="hard-validation")

    assert first_aux.sha256 == second_aux.sha256
    assert first_aux.sha256 not in {first.sha256, second.sha256}


def test_runtime_barrier_geometry_matches_authoritative_vehicle_footprint() -> None:
    config = _small_config()
    tape = generate_condition_tape(ConditionID.STATIC, config, seed=3, fold=4)
    resources = build_experiment_resources(
        config,
        obstacle_count=tape.static_positions.shape[0] + tape.dynamic_positions.shape[1],
        initialization_seed=5,
    )
    bound = _resources_for_tape(resources, tape, config)

    assert bound.barrier_config.obstacle_clearance == (
        float(tape.vehicle_radius) + config.obstacle_clearance
    )
    assert bound.barrier_config.arena_clearance == float(tape.vehicle_radius)


def test_estimator_consumes_predeclared_tape_noise_without_perturbing_true_control() -> None:
    config = _small_config()
    tape = generate_condition_tape(ConditionID.DYNAMICS_CHANGE, config, seed=31, fold=7)
    resources = build_experiment_resources(
        config,
        obstacle_count=tape.static_positions.shape[0] + tape.dynamic_positions.shape[1],
        initialization_seed=2,
    )
    state = np.asarray(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.2, -0.1, 0.05, 0.0, 0.0, 0.0], dtype=np.float32
    )
    next_state = state.copy()
    next_state[7:10] += np.asarray([0.04, -0.02, 0.01], dtype=np.float32)
    commanded = np.asarray([0.08, 0.09, 0.10, 0.11], dtype=np.float32)
    realized = np.asarray([0.07, 0.08, 0.09, 0.10], dtype=np.float32)
    entry = _estimator_history_entry(
        state,
        next_state,
        commanded,
        realized,
        resources.model,
        resources.actuator,
        tape,
        1,
        config.dt,
    )

    expected_acceleration = (next_state[7:10] - state[7:10]) / config.dt
    np.testing.assert_allclose(
        entry[2] - expected_acceleration, tape.estimator_acceleration_noise[1], atol=2e-7
    )
    np.testing.assert_allclose(entry[6] - realized, tape.estimator_motor_force_noise[1], atol=2e-8)
    np.testing.assert_array_equal(entry[5], commanded)
    np.testing.assert_array_equal(realized, np.asarray([0.07, 0.08, 0.09, 0.10], dtype=np.float32))


def test_nominal_trial_is_real_immutable_trace_with_swept_failure_columns() -> None:
    config = _small_config()
    assignment = _assignments(("nominal_only", "analytic_cbf_hocbf"))[0]
    tape = generate_condition_tape(
        assignment.condition,
        config,
        seed=assignment.scenario_root_seed,
        fold=assignment.scenario_fold,
    )
    run = run_trial(assignment, tape, config)

    assert run.trace.scenario_tape_sha256 == tape.sha256
    assert run.trace.policy_values.shape == (config.control_steps, 1)
    assert np.all(run.trace.selected_policy == -1)
    assert not np.any(run.hard_certified_policy)
    assert "static_swept" in run.trace.barrier_names
    assert "dynamic_swept" in run.trace.barrier_names
    np.testing.assert_array_equal(
        run.trace.failure, run.trace.contact | np.any(run.trace.hard_barriers < 0.0, axis=1)
    )
    assert not np.array_equal(run.trace.true_state[0], run.trace.true_state[1])
    assert np.all(run.trace.component_latency_seconds >= 0.0)
    assert run.trace.latency_names.tolist() == [
        "controller",
        "plant",
        "estimator_tick_work",
        "command_preparation",
        "postprocessing",
        "wall_step",
        "command_ready",
    ]
    executed_latency = run.trace.component_latency_seconds[run.trace.executed_control]
    np.testing.assert_allclose(
        np.sum(executed_latency[:, :5], axis=1), executed_latency[:, 5], rtol=1e-12, atol=1e-12
    )
    assert np.all(executed_latency[:, 6] >= executed_latency[:, 3] + executed_latency[:, 0])
    assert run.compile_seconds["controller"] >= 0.0
    np.testing.assert_array_equal(
        run.dashboard_evidence.nominal_rollout_available, run.trace.executed_control
    )
    assert not np.any(run.dashboard_evidence.fallback_rollout_available)
    annotations = change_annotations(run.trace, tape, run.events, sidecar=run.dashboard_evidence)
    assert not any(item.kind == "dynamics-change" for item in annotations)
    preparation = next(event for event in run.events if event.name == "runtime_inputs_precomputed")
    assert preparation.details["excluded_from_warm_step_latency"] is True
    assert preparation.details["dynamic_prediction_contract"] == (
        "predeclared-exogenous-oracle-forecast-observed-active-slots-only-v1"
    )
    assert preparation.details["unobserved_dynamic_slots_masked_for_entire_horizon"] is True
    timing = next(event for event in run.events if event.name == "warm_step_timing_semantics")
    assert timing.details["wall_step_excludes_pacing_sleep"] is True


def test_paired_campaign_consumes_identical_tape_and_distinct_real_dispatches() -> None:
    config = _small_config()
    campaign = CampaignConfig(
        trial=config,
        methods=("nominal_only", "analytic_cbf_hocbf"),
        conditions=("static",),
        trials_per_condition=1,
        root_seed=23,
    )
    result = run_campaign(campaign)

    assert len(result.records) == 2
    assert len(result.trial_runs) == 2
    assert {run.tape.sha256 for run in result.trial_runs} == {result.trial_runs[0].tape.sha256}
    assert {run.assignment.pairing_id for run in result.trial_runs} == {
        result.trial_runs[0].assignment.pairing_id
    }
    assert not result.scientific_claim_eligible
    assert "schedule is not" in result.claim_blockers[-1]


def test_campaign_reuses_same_shape_controller_and_plant_executables() -> None:
    campaign = CampaignConfig(
        trial=_small_config(),
        methods=("nominal_only", "analytic_cbf_hocbf"),
        conditions=("static",),
        trials_per_condition=2,
        root_seed=24,
    )
    result = run_campaign(campaign)
    first, second = (run for run in result.trial_runs if run.assignment.method == "nominal_only")

    assert not first.compile_cache_hits["controller"]
    assert not first.compile_cache_hits["plant"]
    assert second.compile_cache_hits["controller"]
    assert second.compile_cache_hits["plant"]
    assert second.compile_seconds["controller"] == 0.0
    assert second.compile_seconds["plant"] == 0.0
    event = next(event for event in second.events if event.name == "compile_cache_accounting")
    assert event.details["warm_execution_excludes_compilation"] is True


def test_artifact_campaign_is_complete_resumable_and_retains_numeric_inference(
    tmp_path: Path,
) -> None:
    campaign = CampaignConfig(
        trial=_small_config(),
        methods=("nominal_only", "da_plcbf_full"),
        conditions=("static",),
        trials_per_condition=1,
        root_seed=26,
    )
    output = tmp_path / "campaign-run"
    first = run_campaign(campaign, output_directory=output)

    assert len(first.records) == 2
    assert len((output / "aggregate" / "outcomes.jsonl").read_text().splitlines()) == 2
    for name in (
        "paired_metrics.csv",
        "confidence_intervals.json",
        "paired_comparisons.json",
        "report.md",
        "scientific_report.md",
    ):
        assert (output / "aggregate" / name).is_file()
    before = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }

    resumed = run_campaign(campaign, output_directory=output, resume=True)
    after = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    assert not resumed.trial_runs
    assert resumed.records == first.records
    assert after == before
    comparison_payload = json.loads((output / "aggregate" / "paired_comparisons.json").read_text())
    assert comparison_payload["schema_version"] == 3
    assert not comparison_payload["global_confirmatory_superiority_supported"]
    assert comparison_payload["inference_config"]["analysis_role"] == "confirmatory"
    assert comparison_payload["exploratory_inference_config"]["analysis_role"] == "exploratory"
    report = (output / "aggregate" / "scientific_report.md").read_text()
    assert "Predeclared analysis roles" in report
    assert "never claim-eligible" in report
    reconstructed = validate_persisted_campaign_evidence(output)
    assert reconstructed.scientific_claim_eligible == first.scientific_claim_eligible
    validate_current_source_tree(output, repository=Path(__file__).resolve().parents[4])
    with np.testing.assert_raises_regex(ValueError, "current source tree differs"):
        validate_current_source_tree(output, repository=tmp_path)

    comparisons_path = output / "aggregate" / "paired_comparisons.json"
    original_comparisons = comparisons_path.read_bytes()
    comparison_mutations = (
        lambda value: value.__setitem__("execution_complete", not value["execution_complete"]),
        lambda value: value.__setitem__(
            "scientific_claim_eligible", not value["scientific_claim_eligible"]
        ),
        lambda value: value.__setitem__("schedule_sha256", "0" * 64),
        lambda value: value["claim_blockers"].append("forged campaign blocker"),
        lambda value: value["comparisons"][0].__setitem__(
            "conclusion", "forged comparison conclusion"
        ),
    )
    for mutate in comparison_mutations:
        comparisons = json.loads(original_comparisons)
        mutate(comparisons)
        comparisons_path.write_text(
            json.dumps(comparisons, sort_keys=True, separators=(",", ":")) + "\n"
        )
        with np.testing.assert_raises_regex(ValueError, "paired comparisons"):
            validate_persisted_campaign_evidence(output)
        comparisons_path.write_bytes(original_comparisons)

    scientific_report = output / "aggregate" / "scientific_report.md"
    original_scientific_report = scientific_report.read_bytes()
    scientific_report.write_bytes(original_scientific_report + b"forged conclusion\n")
    with np.testing.assert_raises_regex(ValueError, "scientific report"):
        validate_persisted_campaign_evidence(output)
    scientific_report.write_bytes(original_scientific_report)

    outcomes_path = output / "aggregate" / "outcomes.jsonl"
    original_outcomes = outcomes_path.read_bytes()
    outcomes = [json.loads(line) for line in original_outcomes.splitlines()]
    outcomes[0]["scientific_metrics"]["minimum_hard_margin"] += 0.25
    outcomes_path.write_text(
        "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in outcomes)
    )
    with np.testing.assert_raises_regex(ValueError, "scientific metrics"):
        validate_persisted_campaign_evidence(output)
    outcomes_path.write_bytes(original_outcomes)

    outcomes = [json.loads(line) for line in original_outcomes.splitlines()]
    outcomes[0]["method_claim_eligible"] = False
    outcomes[0]["claim_blockers"] = ["forged blocker"]
    outcomes_path.write_text(
        "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in outcomes)
    )
    with np.testing.assert_raises_regex(ValueError, "method eligibility"):
        validate_persisted_campaign_evidence(output)
    outcomes_path.write_bytes(original_outcomes)


def test_source_tree_digest_binds_runtime_plant_assets(tmp_path: Path) -> None:
    package = tmp_path / "crazyflow"
    example = tmp_path / "examples" / "da_plcbf"
    benchmark = tmp_path / "benchmark"
    for directory in (package / "drones", example, benchmark):
        directory.mkdir(parents=True)
    (package / "module.py").write_text("VALUE = 1\n")
    params = package / "drones" / "params.toml"
    params.write_text("[cf21B_500]\nmass = 0.04338\n")
    (example / "campaign.py").write_text("PROFILE = 'final'\n")
    (benchmark / "bptt.py").write_text("SHAPE = (64, 64, 50)\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (tmp_path / "pixi.lock").write_text("version: 1\n")

    before = _source_tree_sha256(tmp_path)
    params.write_text("[cf21B_500]\nmass = 0.05000\n")
    after_plant_change = _source_tree_sha256(tmp_path)
    assert after_plant_change != before

    documentation = tmp_path / "docs"
    documentation.mkdir()
    (documentation / "notes.md").write_text("non-runtime prose\n")
    assert _source_tree_sha256(tmp_path) == after_plant_change


def test_resume_revalidates_success_artifacts_before_skipping(tmp_path: Path) -> None:
    campaign = CampaignConfig(
        trial=_small_config(),
        methods=("nominal_only", "analytic_cbf_hocbf"),
        conditions=("static",),
        trials_per_condition=1,
        root_seed=27,
    )
    output = tmp_path / "tamper-run"
    run_campaign(campaign, output_directory=output)
    trace_path = next(output.glob("methods/*/*/*/trace.npz"))
    original = load_trace(trace_path)
    changed_state = original.true_state.copy()
    changed_state[0, 0] += 0.01
    trace_path.unlink()
    save_trace(replace(original, true_state=changed_state), trace_path)

    with np.testing.assert_raises_regex(ValueError, "digest mismatch"):
        run_campaign(campaign, output_directory=output, resume=True)


def test_physical_trace_evidence_is_recomputed_from_true_state_and_tape() -> None:
    config = _small_config()
    assignment = _assignments(("nominal_only", "analytic_cbf_hocbf"))[0]
    tape = generate_condition_tape(
        assignment.condition,
        config,
        seed=assignment.scenario_root_seed,
        fold=assignment.scenario_fold,
    )
    run = run_trial(assignment, tape, config)
    _validate_trace_physical_evidence(run.trace, tape, config, condition=assignment.condition)
    barriers = np.array(run.trace.hard_barriers, copy=True)
    barriers[0, 0] += 0.1
    tampered = replace(run.trace, hard_barriers=barriers)

    with np.testing.assert_raises_regex(ValueError, "hard barriers do not recompute"):
        _validate_trace_physical_evidence(tampered, tape, config, condition=assignment.condition)

    # Recomputing barriers alone must not legitimize an impossible trajectory. This forged trace
    # has the scheduled nonzero velocity at every node but no position change at all.
    stale_states = np.broadcast_to(run.trace.true_state[0], run.trace.true_state.shape).copy()
    stale_barriers, stale_contact, stale_failure = _barrier_trace(stale_states, tape, config)
    stale = replace(
        run.trace,
        true_state=stale_states,
        hard_barriers=stale_barriers,
        contact=stale_contact,
        failure=stale_failure,
        degraded=np.asarray(run.trace.degraded) | stale_failure,
    )
    assert np.linalg.norm(stale.true_state[0, 7:10]) > 0.0
    assert np.all(np.diff(stale.true_state[:, :3], axis=0) == 0.0)
    with np.testing.assert_raises_regex(ValueError, "true-state transition does not replay"):
        _validate_trace_physical_evidence(stale, tape, config, condition=assignment.condition)


def test_campaign_reuses_dynamic_bptt_and_estimator_executables_across_folds() -> None:
    config = ExperimentConfig(
        control_steps=3,
        certificate_horizon=1,
        policy_count=16,
        prediction_samples=4,
        training_scenario_count=2,
        bptt_burst_steps=1,
        adaptation_interval_steps=2,
        estimator_interval_steps=1,
        estimator_window_steps=3,
        validation_runtime_budget_seconds=60.0,
        validation_minimum_diversity=1e-8,
    )
    result = run_campaign(
        CampaignConfig(
            trial=config,
            methods=("nominal_only", "da_plcbf_full"),
            conditions=("dynamics_change",),
            trials_per_condition=2,
            root_seed=25,
        )
    )
    first, second = (run for run in result.trial_runs if run.assignment.method == "da_plcbf_full")

    assert not first.compile_cache_hits["controller"]
    assert not first.compile_cache_hits["estimator"]
    assert not first.compile_cache_hits["bptt_startup"]
    assert second.compile_cache_hits["controller"]
    assert second.compile_cache_hits["plant"]
    assert second.compile_cache_hits["estimator"]
    assert second.compile_cache_hits["bptt_startup"]
    assert second.compile_cache_hits["bptt_online"]
    assert second.compile_cache_hits["validation_startup"]
    assert second.compile_cache_hits["validation_online"]
    assert all(seconds == 0.0 for seconds in second.compile_seconds.values())
    cold = next(event for event in second.events if event.category == "cold_start")
    assert cold.details["bptt_compile_seconds"] == 0.0
    assert cold.details["bptt_warmup_seconds"] == 0.0
    assert cold.details["bptt_execution_seconds"] >= 0.0
    assert cold.details["admission_runtime_scope"] == (
        "complete_prepublication_candidate_job_excluding_compile_and_warmup"
    )
    assert cold.details["admission_runtime_seconds"] >= (
        cold.details["bptt_setup_seconds"]
        + cold.details["bptt_execution_seconds"]
        + cold.details["validation_seconds"]
    )
    assert cold.details["validation_report_seconds"] >= 0.0
    assert cold.details["admission_publication_included"] is False
    assert result.execution_complete
    assert result.inference_config is not None
    assert result.exploratory_inference_config is not None
    assert len(result.paired_comparisons) == 9
    assert (
        sum(item.analysis_role is AnalysisRole.CONFIRMATORY for item in result.paired_comparisons)
        == 3
    )
    assert (
        sum(item.analysis_role is AnalysisRole.EXPLORATORY for item in result.paired_comparisons)
        == 6
    )
    assert all(
        not item.superiority_supported
        for item in result.paired_comparisons
        if item.analysis_role is AnalysisRole.EXPLORATORY
    )
    assert not result.global_confirmatory_superiority_supported
    confirmatory = tuple(
        item
        for item in result.paired_comparisons
        if item.analysis_role is AnalysisRole.CONFIRMATORY
    )
    all_supported = tuple(replace(item, superiority_supported=True) for item in confirmatory)
    assert _global_confirmatory_superiority_supported(all_supported)
    assert not _global_confirmatory_superiority_supported(
        (replace(all_supported[0], superiority_supported=False), *all_supported[1:])
    )


def test_nonfinite_policy_evidence_fails_closed_without_numeric_substitution() -> None:
    policy = np.asarray([0.2, np.inf])
    training = np.asarray([0.1, 0.3])

    with np.testing.assert_raises_regex(RuntimeError, "non-finite policy"):
        _finite_policy_evidence(policy, training)


def test_online_adaptation_trial_gate_accepts_hard_rejection_but_use_proof_stays_strict() -> None:
    config = ExperimentConfig(
        control_steps=3,
        certificate_horizon=1,
        policy_count=16,
        prediction_samples=4,
        training_scenario_count=2,
        bptt_burst_steps=1,
        adaptation_interval_steps=2,
        estimator_interval_steps=1,
        estimator_window_steps=3,
        validation_runtime_budget_seconds=60.0,
        validation_minimum_diversity=1e-8,
    )
    assignment = _assignments(("nominal_only", "da_plcbf_full"), "dynamics_change")[1]
    tape = generate_condition_tape(
        assignment.condition,
        config,
        seed=assignment.scenario_root_seed,
        fold=assignment.scenario_fold,
    )
    run = run_trial(assignment, tape, config)
    cold = next(event for event in run.events if event.category == "cold_start")
    used_version = int(run.trace.snapshot_version[0])
    synthetic_online = replace(
        cold,
        category="adaptation",
        name="candidate_admitted",
        details={**cold.details, "published_snapshot_version": used_version},
    )
    terminal_only = replace(synthetic_online, step=run.trace.steps - 1)

    assert _used_online_adaptation_versions((synthetic_online,), run.trace) == (used_version,)
    assert _used_online_adaptation_versions((terminal_only,), run.trace) == ()
    assert not any("adaptation" in blocker for blocker in run.claim_blockers)


def test_trial_artifacts_round_trip_without_synthetic_trace(tmp_path: Path) -> None:
    config = _small_config()
    assignment = _assignments(("nominal_only", "analytic_cbf_hocbf"))[0]
    tape = generate_condition_tape(
        assignment.condition,
        config,
        seed=assignment.scenario_root_seed,
        fold=assignment.scenario_fold,
    )
    run = run_trial(assignment, tape, config)
    directory = save_trial_run(run, tmp_path)
    trace = load_trace(directory / "trace.npz")

    assert trace.content_sha256 == run.trace.content_sha256
    assert load_events(directory / "events.jsonl", trace=trace)
    load_metrics(directory / "metrics.json", trace=trace)
    timing = load_timing(directory / "timing.json", trace=trace)
    assert timing["warm_execution_excludes_compilation"] is True


def test_full_development_slice_logs_model_and_candidate_lifecycle() -> None:
    config = ExperimentConfig(
        control_steps=3,
        certificate_horizon=1,
        policy_count=16,
        prediction_samples=4,
        training_scenario_count=2,
        bptt_burst_steps=1,
        adaptation_interval_steps=2,
        estimator_interval_steps=1,
        estimator_window_steps=3,
        validation_runtime_budget_seconds=60.0,
        validation_minimum_diversity=1e-8,
    )
    assignment = _assignments(("nominal_only", "da_plcbf_full"), condition="dynamics_change")[1]
    tape = generate_condition_tape(
        assignment.condition,
        config,
        seed=assignment.scenario_root_seed,
        fold=assignment.scenario_fold,
    )
    run = run_trial(assignment, tape, config)
    names = {event.name for event in run.events}

    assert "candidate_submitted" in names
    assert any(name.startswith("candidate_") and name != "candidate_submitted" for name in names)
    assert "model_version_advanced" in names
    estimator_event = next(
        event for event in run.events if event.name == "estimator_update_executed"
    )
    assert estimator_event.details["event_only_latency_sample"] is True
    assert estimator_event.details["execution_seconds"] > 0.0
    assert np.max(run.trace.model_version) > 0
    assert run.trace.policy_values.shape == (3, 16)
    np.testing.assert_array_equal(run.trace.executed_control, np.asarray([True, True, False]))
    assert run.trace.gradient_norm[-1] == 0.0
    cold = next(event for event in run.events if event.category == "cold_start")
    assert cold.details["bptt_execution_scope"] == "compiled_burst_only"
    assert cold.details["bptt_compilation_excluded_from_execution_timing"] is True
    assert cold.details["validation_execution_synchronized"] is True
    scheduler = next(event for event in run.events if event.name == "logical_simulation_scheduler")
    assert scheduler.details["host_load_can_change_event_step"] is False
    assert scheduler.details["real_time_claim_eligible"] is False
    assert np.any(run.dashboard_evidence.fallback_rollout_available[:-1])
    assert np.any(run.dashboard_evidence.selected_rollout_available[:-1])
    replayed = replay_dashboard_dynamics_evidence(
        run.trace, tape, assignment.condition, assignment.method, config
    )
    np.testing.assert_allclose(run.dashboard_evidence.dynamics_true, replayed[0], atol=2e-7)
    np.testing.assert_allclose(run.dashboard_evidence.dynamics_estimated, replayed[1], atol=2e-7)
    np.testing.assert_allclose(
        run.dashboard_evidence.dynamics_uncertainty_samples, replayed[2], atol=2e-7
    )
    np.testing.assert_array_equal(
        run.dashboard_evidence.dynamics_uncertainty_available, replayed[3]
    )
    validate_dashboard_evidence_binding(
        run.dashboard_evidence, run.trace, tape, events=run.events, expected_dynamics=replayed
    )
    no_admissions = replace(
        run.dashboard_evidence,
        admission_recorded=np.asarray(False),
        candidate_present=np.zeros(run.trace.steps, dtype=np.bool_),
        candidate_admitted=np.zeros(run.trace.steps, dtype=np.bool_),
        candidate_rejected=np.zeros(run.trace.steps, dtype=np.bool_),
        admission_margin=np.zeros(run.trace.steps),
        admission_reason_names=np.asarray((), dtype="<U1"),
        admission_reason_index=np.full(run.trace.steps, -1, dtype=np.int16),
    )
    with pytest.raises(ValueError, match="admission/BPTT"):
        validate_dashboard_evidence_binding(no_admissions, run.trace, tape, events=run.events)
    descriptors = np.array(run.dashboard_evidence.normalized_descriptors, copy=True)
    descriptors[0, 0, 0] += 0.25
    with pytest.raises(ValueError, match="commitment"):
        validate_dashboard_evidence_binding(
            replace(run.dashboard_evidence, normalized_descriptors=descriptors),
            run.trace,
            tape,
            events=run.events,
        )
    estimates = np.array(run.dashboard_evidence.dynamics_estimated, copy=True)
    estimates[0, 0] += 0.1
    with pytest.raises(ValueError, match="commitment|independently replay"):
        validate_dashboard_evidence_binding(
            replace(run.dashboard_evidence, dynamics_estimated=estimates),
            run.trace,
            tape,
            expected_dynamics=replayed,
        )
    assert run.adaptation_evidence is not None
    validate_adaptation_evidence_binding(
        run.adaptation_evidence,
        run.trace,
        run.events,
        tape=tape,
        condition=assignment.condition,
        method=assignment.method,
        config=config,
    )
    assert np.any(run.dashboard_evidence.descriptor_available[:-1])
    assert np.any(run.dashboard_evidence.ghost_rollout_available[:-1])
    assert any("K=64" in blocker for blocker in run.claim_blockers)


def test_logical_adaptation_is_load_independent_except_recorded_wall_timing() -> None:
    config = ExperimentConfig(
        control_steps=3,
        certificate_horizon=1,
        policy_count=16,
        prediction_samples=4,
        training_scenario_count=2,
        bptt_burst_steps=1,
        adaptation_interval_steps=2,
        estimator_interval_steps=1,
        estimator_window_steps=3,
        validation_runtime_budget_seconds=60.0,
        validation_minimum_diversity=1e-8,
        adaptation_execution_mode="logical_simulation",
    )
    assignment = _assignments(("nominal_only", "da_plcbf_full"), "dynamics_change")[1]
    tape = generate_condition_tape(
        assignment.condition,
        config,
        seed=assignment.scenario_root_seed,
        fold=assignment.scenario_fold,
    )

    first = run_trial(assignment, tape, config)
    second = run_trial(assignment, tape, config)

    for name in (
        "true_state",
        "nominal_control",
        "filtered_control",
        "applied_control",
        "hard_barriers",
        "policy_values",
        "selected_policy",
        "snapshot_version",
        "model_version",
        "degraded",
        "failure",
    ):
        np.testing.assert_array_equal(getattr(first.trace, name), getattr(second.trace, name))
    first_decisions = tuple(
        (
            event.step,
            event.name,
            event.details.get("reason"),
            event.details.get("published_snapshot_version"),
            event.details.get("report_passed"),
            tuple(event.details.get("failed_gates", ())),
            event.details.get("training_model_version"),
            event.details.get("validation_model_version"),
        )
        for event in first.events
        if event.category == "adaptation"
        and event.name in {"candidate_admitted", "candidate_rejected"}
    )
    second_decisions = tuple(
        (
            event.step,
            event.name,
            event.details.get("reason"),
            event.details.get("published_snapshot_version"),
            event.details.get("report_passed"),
            tuple(event.details.get("failed_gates", ())),
            event.details.get("training_model_version"),
            event.details.get("validation_model_version"),
        )
        for event in second.events
        if event.category == "adaptation"
        and event.name in {"candidate_admitted", "candidate_rejected"}
    )
    # Snapshot/report digests deliberately bind exact measured validation timing, so they are
    # provenance rather than load-invariant numerical outputs.  Compare the complete hard gate
    # decision and publication semantics here; timing and its content-addressed lineage are
    # checked separately by artifact tests.
    assert first_decisions == second_decisions
    assert first_decisions
    np.testing.assert_array_equal(first.trace.latency_names, second.trace.latency_names)
    assert np.all(first.trace.component_latency_seconds >= 0.0)
    assert np.all(second.trace.component_latency_seconds >= 0.0)
