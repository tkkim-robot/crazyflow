from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf import adaptation_evidence as adaptation_evidence_module
from crazyflow.safety.da_plcbf.adaptation_evidence import (
    BPTT_EXECUTION_CONTRACT,
    validate_adaptation_evidence_binding,
)
from crazyflow.safety.da_plcbf.artifacts import (
    _validate_runtime_device_roles,
    collect_provenance,
    load_events,
    load_metrics,
    load_timing,
    load_trace,
    save_trace,
)
from crazyflow.safety.da_plcbf.campaign_artifacts import (
    CampaignArtifactStore,
    _base_airborne_plant,
    _replay_airborne_transitions,
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
    _authoritative_estimator_arguments,
    _authoritative_estimator_function,
    _authoritative_estimator_observations,
    _authoritative_model_samples,
    _authoritative_resources,
    _auxiliary_tape,
    _barrier_trace,
    _build_bptt_executable_pool,
    _causal_history_indices,
    _estimator_history_entry,
    _finite_policy_evidence,
    _global_confirmatory_superiority_supported,
    _initialize_authoritative_estimator,
    _offline_training_batch,
    _online_bptt_device,
    _plant_step,
    _replay_dashboard_dynamics_and_contexts,
    _resources_for_tape,
    _used_online_adaptation_versions,
    build_experiment_resources,
    generate_condition_tape,
    run_campaign,
    run_trial,
    save_trial_run,
    scenario_config_for_condition,
)
from crazyflow.safety.da_plcbf.scientific_dashboard import change_annotations
from crazyflow.safety.da_plcbf.scientific_evaluation import AnalysisRole, make_paired_trial_schedule
from crazyflow.safety.da_plcbf.snapshots import ActiveSnapshotStore
from crazyflow.safety.da_plcbf.validation import hard_validate_candidate
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


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


def _create_pending_campaign_store(
    root: Path, *, root_seed: int
) -> tuple[CampaignConfig, object, dict[tuple[str, int], object], Path]:
    campaign = CampaignConfig(
        trial=_small_config(),
        methods=("nominal_only", "analytic_cbf_hocbf"),
        conditions=("static",),
        trials_per_condition=1,
        root_seed=root_seed,
    )
    schedule = campaign.schedule()
    tapes: dict[tuple[str, int], object] = {}
    for assignment in schedule.assignments:
        key = (assignment.condition, assignment.fold)
        if key not in tapes:
            tapes[key] = generate_condition_tape(
                assignment.condition,
                campaign.trial,
                seed=assignment.scenario_root_seed,
                fold=assignment.scenario_fold,
            )
    repository = Path(__file__).resolve().parents[4]
    CampaignArtifactStore(root, campaign, schedule, tapes, repository=repository, resume=False)
    return campaign, schedule, tapes, repository


def test_online_bptt_pool_places_closed_constants_on_selected_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    resources = build_experiment_resources(config, obstacle_count=1, initialization_seed=0)
    assert all(
        leaf.device.platform == "cpu"
        for leaf in jax.tree.leaves((resources.spec, resources.initial_params))
    )
    captured: dict[str, object] = {}

    def fake_builder(
        spec: object,
        actuator: object,
        *_args: object,
        device: jax.Device | None = None,
        **_kwargs: object,
    ) -> object:
        captured["spec"] = spec
        captured["actuator"] = actuator
        captured["device"] = device
        return object()

    monkeypatch.setattr(
        "crazyflow.safety.da_plcbf.experiments.build_dynamic_model_quad_actor_bptt_functions",
        fake_builder,
    )
    pool = _build_bptt_executable_pool(resources, config)
    selected = _online_bptt_device()
    assert captured["device"] == pool.device
    assert pool.device_key == (selected.platform, selected.id)
    closed_leaves = jax.tree.leaves((captured["spec"], captured["actuator"]))
    assert closed_leaves
    assert all(leaf.device.platform == selected.platform for leaf in closed_leaves)


def test_trial_rejects_caller_supplied_policy_root_from_wrong_shared_seed() -> None:
    config = _small_config()
    assignment = _assignments(("nominal_only", "da_plcbf_full"))[0]
    tape = generate_condition_tape(
        assignment.condition,
        config,
        seed=assignment.scenario_root_seed,
        fold=assignment.scenario_fold,
    )
    obstacle_count = tape.static_positions.shape[0] + tape.dynamic_positions.shape[1]
    wrong_resources = build_experiment_resources(
        config,
        obstacle_count=obstacle_count,
        initialization_seed=(assignment.shared_stochastic_seed + 1) & 0xFFFFFFFF,
    )
    with pytest.raises(ValueError, match="CPU-canonical scheduled policy root"):
        run_trial(assignment, tape, config, resources=wrong_resources)


def test_authoritative_estimator_context_is_cpu_and_backend_source_independent() -> None:
    config = _small_config()
    cpu = jax.devices("cpu")[0]
    ambient = build_experiment_resources(config, obstacle_count=1, initialization_seed=0)
    # A stale ambient-backend inverse must never leak into proof resources or runtime contexts.
    ambient = replace(
        ambient, model=ambient.model._replace(inertia_inv=jnp.zeros_like(ambient.model.inertia_inv))
    )
    with jax.default_device(cpu):
        cpu_source = build_experiment_resources(config, obstacle_count=1, initialization_seed=0)

    ambient_estimator = _initialize_authoritative_estimator(ambient.model, ambient.estimator_config)
    cpu_estimator = _initialize_authoritative_estimator(
        cpu_source.model, cpu_source.estimator_config
    )
    ambient_context = _authoritative_model_samples(
        ambient.model, ambient_estimator, ambient.estimator_config, 4
    )
    cpu_context = _authoritative_model_samples(
        cpu_source.model, cpu_estimator, cpu_source.estimator_config, 4
    )
    ambient_proof = _authoritative_resources(ambient)
    cpu_proof = _authoritative_resources(cpu_source)

    for actual_tree, expected_tree in (
        (ambient_estimator, cpu_estimator),
        (ambient_context, cpu_context),
        (
            (ambient_proof.model, ambient_proof.actuator, ambient_proof.spec),
            (cpu_proof.model, cpu_proof.actuator, cpu_proof.spec),
        ),
    ):
        actual_leaves = jax.tree.leaves(actual_tree)
        expected_leaves = jax.tree.leaves(expected_tree)
        assert actual_leaves
        assert all(leaf.device.platform == "cpu" for leaf in actual_leaves)
        for actual, expected in zip(actual_leaves, expected_leaves, strict=True):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_authoritative_estimator_fresh_compiles_are_byte_exact_on_cpu() -> None:
    config = _small_config()
    resources = build_experiment_resources(config, obstacle_count=1, initialization_seed=0)
    estimator = _initialize_authoritative_estimator(resources.model, resources.estimator_config)
    observations = _authoritative_estimator_observations([], config.estimator_window_steps)
    arguments = _authoritative_estimator_arguments(estimator, observations, 0)
    first = (
        _authoritative_estimator_function(resources.estimator_config).lower(*arguments).compile()
    )
    second = (
        _authoritative_estimator_function(resources.estimator_config).lower(*arguments).compile()
    )
    first_output = first(*arguments)
    second_output = second(*arguments)
    first_leaves = jax.tree.leaves(first_output)
    second_leaves = jax.tree.leaves(second_output)
    assert first_leaves
    assert all(leaf.device.platform == "cpu" for leaf in first_leaves)
    for actual, expected in zip(first_leaves, second_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_gpu_producer_adaptation_validates_without_cross_process_bptt_replay(
    tmp_path: Path,
) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("cross-backend replay regression requires a GPU-default parent process")
    config = replace(
        _small_config(),
        policy_count=16,
        estimator_interval_steps=1,
        validation_minimum_diversity=1e-8,
    )
    assignment = _assignments(("nominal_only", "da_plcbf_full"))[1]
    tape = generate_condition_tape(
        assignment.condition,
        config,
        seed=assignment.scenario_root_seed,
        fold=assignment.scenario_fold,
    )
    run = run_trial(assignment, tape, config)
    assert run.adaptation_evidence is not None
    assert any(
        proof.context_step > 0 and proof.candidate.model_version > 0
        for proof in run.adaptation_evidence.decisions
    )
    method_directory = save_trial_run(run, tmp_path / "portable-run")
    tape_path = (
        tmp_path
        / "portable-run"
        / "scenario_tapes"
        / assignment.condition
        / f"{assignment.fold}.npz"
    )
    config_json = json.dumps(
        {name: getattr(config, name) for name in config.__dataclass_fields__}, sort_keys=True
    )
    validation_script = textwrap.dedent(
        """
        import json
        import sys
        from pathlib import Path

        from crazyflow.safety.da_plcbf.adaptation_evidence import (
            load_adaptation_evidence,
            validate_adaptation_evidence_binding,
        )
        from crazyflow.safety.da_plcbf.artifacts import load_events, load_trace
        from crazyflow.safety.da_plcbf.experiments import ExperimentConfig
        from crazyflow.safety.da_plcbf.scenarios import load_scenario_tape

        directory = Path(sys.argv[1])
        trace = load_trace(directory / "trace.npz")
        events = load_events(directory / "events.jsonl", trace=trace)
        evidence = load_adaptation_evidence(directory / "adaptation_evidence.npz")
        tape = load_scenario_tape(sys.argv[2])
        config = ExperimentConfig(**json.loads(sys.argv[3]))
        validate_adaptation_evidence_binding(
            evidence,
            trace,
            events,
            shared_stochastic_seed=int(sys.argv[4]),
            tape=tape,
            condition="static",
            method="da_plcbf_full",
            config=config,
        )
        """
    )
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["JAX_COMPILATION_CACHE_DIR"] = str(tmp_path / "fresh-cpu-cache")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            validation_script,
            str(method_directory),
            str(tape_path),
            config_json,
            str(assignment.shared_stochastic_seed),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_hard_barrier_tilt_uses_host_canonical_float64_geometry() -> None:
    config = _small_config()
    tape = generate_condition_tape("static", config, seed=101, fold=0)
    states = np.zeros((config.control_steps, 13), dtype=np.float64)
    states[:, :3] = tape.vehicle_initial_position
    states[:, 3:7] = np.asarray((0.0, 0.0, 0.0, 1.0))
    # Retained from a GPU-produced development trace whose float32 GPU and CPU quaternion
    # lowerings differed by two ulps in the reported tilt margin.
    states[1, 3:7] = np.asarray(
        (0.2016475647687912, 0.24650098383426666, 0.02752041257917881, 0.9475326538085938)
    )

    barriers, _, _ = _barrier_trace(states, tape, config)
    quaternions = states[:, 3:7]
    quaternions = quaternions / np.sqrt(np.sum(quaternions**2, axis=-1, keepdims=True))
    cosine_tilt = 1.0 - 2.0 * (
        quaternions[:, 0] * quaternions[:, 0] + quaternions[:, 1] * quaternions[:, 1]
    )
    tilt_limit = math.cos(config.tilt_max_radians)
    expected = (cosine_tilt - tilt_limit) / (1.0 - tilt_limit)

    assert barriers.dtype == np.float64
    np.testing.assert_array_equal(barriers[:, 5], expected)


def test_gpu_hard_barrier_trace_replays_exactly_in_fresh_cpu_process(tmp_path: Path) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("cross-backend barrier replay regression requires a GPU-default parent")
    config = _small_config()
    tape = generate_condition_tape("static", config, seed=101, fold=0)
    states = np.zeros((config.control_steps, 13), dtype=np.float64)
    states[:, :3] = tape.vehicle_initial_position
    states[:, 3:7] = np.asarray((0.0, 0.0, 0.0, 1.0))
    states[1, 3:7] = np.asarray(
        (0.2016475647687912, 0.24650098383426666, 0.02752041257917881, 0.9475326538085938)
    )
    gpu_barriers, _, _ = _barrier_trace(states, tape, config)
    states_path = tmp_path / "states.npy"
    cpu_barriers_path = tmp_path / "cpu-barriers.npy"
    np.save(states_path, states, allow_pickle=False)
    config_json = json.dumps(
        {name: getattr(config, name) for name in config.__dataclass_fields__}, sort_keys=True
    )
    validation_script = textwrap.dedent(
        """
        import json
        import sys

        import numpy as np

        from crazyflow.safety.da_plcbf.experiments import (
            ExperimentConfig,
            _barrier_trace,
            generate_condition_tape,
        )

        states = np.load(sys.argv[1], allow_pickle=False)
        config = ExperimentConfig(**json.loads(sys.argv[2]))
        tape = generate_condition_tape("static", config, seed=101, fold=0)
        barriers, _, _ = _barrier_trace(states, tape, config)
        np.save(sys.argv[3], barriers, allow_pickle=False)
        """
    )
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["JAX_COMPILATION_CACHE_DIR"] = str(tmp_path / "fresh-cpu-barrier-cache")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            validation_script,
            str(states_path),
            config_json,
            str(cpu_barriers_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cpu_barriers = np.load(cpu_barriers_path, allow_pickle=False)
    np.testing.assert_array_equal(cpu_barriers, gpu_barriers)


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    portable_runtime = json.loads((output / "provenance.json").read_text())
    portable_runtime["runtime"]["platform"] = "portable-verification-platform"
    with monkeypatch.context() as patch:
        patch.setattr(
            "crazyflow.safety.da_plcbf.campaign_artifacts.collect_provenance",
            lambda _repository: portable_runtime,
        )
        resumed = run_campaign(campaign, output_directory=output, resume=True)
    after = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    assert not resumed.trial_runs
    assert resumed.records == first.records
    assert after == before
    scipy_drift = json.loads(json.dumps(portable_runtime))
    scipy_drift["packages"]["scipy"] = "0.0.0-drift"
    with monkeypatch.context() as patch:
        patch.setattr(
            "crazyflow.safety.da_plcbf.campaign_artifacts.collect_provenance",
            lambda _repository: scipy_drift,
        )
        with pytest.raises(ValueError, match="analysis package scipy"):
            run_campaign(campaign, output_directory=output, resume=True)
    for runtime_field in ("python", "implementation"):
        runtime_drift = json.loads(json.dumps(portable_runtime))
        runtime_drift["runtime"][runtime_field] += "-drift"
        with monkeypatch.context() as patch:
            patch.setattr(
                "crazyflow.safety.da_plcbf.campaign_artifacts.collect_provenance",
                lambda _repository, current=runtime_drift: current,
            )
            with pytest.raises(ValueError, match=f"analysis runtime {runtime_field}"):
                run_campaign(campaign, output_directory=output, resume=True)
    with monkeypatch.context() as patch:
        patch.setattr(
            "crazyflow.safety.da_plcbf.campaign_artifacts._source_tree_sha256",
            lambda _repository: "0" * 64,
        )
        with pytest.raises(ValueError, match="source digest"):
            run_campaign(campaign, output_directory=output, resume=True)
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


def test_pending_resume_accepts_same_immutable_execution_identity(tmp_path: Path) -> None:
    root = tmp_path / "same-identity"
    campaign, schedule, tapes, repository = _create_pending_campaign_store(root, root_seed=281)
    resumed = CampaignArtifactStore(
        root, campaign, schedule, tapes, repository=repository, resume=True
    )
    assert not resumed.completed_keys()


@pytest.mark.parametrize("drift", ("python", "cpu_model", "device_inventory", "scipy"))
def test_pending_resume_rejects_runtime_or_device_drift_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    root = tmp_path / f"drift-{drift}"
    campaign, _, _, _ = _create_pending_campaign_store(root, root_seed=282)
    current = json.loads((root / "provenance.json").read_text())
    if drift == "python":
        current["runtime"]["python"] = "0.0.0-drift"
    elif drift == "cpu_model":
        current["hardware"]["cpu"] = "different-vendor-family-model-stepping"
    elif drift == "device_inventory":
        current["jax"]["devices"].append("drift-device:99")
    else:
        current["packages"]["scipy"] = "0.0.0-drift"
    monkeypatch.setattr(
        "crazyflow.safety.da_plcbf.campaign_artifacts.collect_provenance",
        lambda _repository: current,
    )
    monkeypatch.setattr(
        "crazyflow.safety.da_plcbf.experiments.run_trial",
        lambda *_args, **_kwargs: pytest.fail("runtime identity drift reached trial execution"),
    )
    expected_error = (
        "resume analysis" if drift in {"python", "scipy"} else "resume execution identity"
    )
    with pytest.raises(ValueError, match=expected_error):
        run_campaign(campaign, output_directory=root, resume=True)


def test_pending_resume_allows_nonexecution_descriptive_provenance_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "descriptive-drift"
    campaign, schedule, tapes, repository = _create_pending_campaign_store(root, root_seed=283)
    current = json.loads((root / "provenance.json").read_text())
    current["video"]["encoder_version"] += " descriptive-build-metadata-drift"
    monkeypatch.setattr(
        "crazyflow.safety.da_plcbf.campaign_artifacts.collect_provenance",
        lambda _repository: current,
    )
    resumed = CampaignArtifactStore(
        root, campaign, schedule, tapes, repository=repository, resume=True
    )
    assert not resumed.completed_keys()


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

    changed_filtered = np.array(run.trace.filtered_control, copy=True)
    changed_filtered[0, 0] += 0.01
    mismatched_actuator = replace(run.trace, filtered_control=changed_filtered)
    with np.testing.assert_raises_regex(ValueError, "applied control does not match"):
        _validate_trace_physical_evidence(
            mismatched_actuator, tape, config, condition=assignment.condition
        )

    changed_state = np.array(run.trace.true_state, copy=True)
    changed_state[1, 10] += 5e-5
    changed_barriers, changed_contact, changed_failure = _barrier_trace(changed_state, tape, config)
    near_threshold_tamper = replace(
        run.trace,
        true_state=changed_state,
        hard_barriers=changed_barriers,
        contact=changed_contact,
        failure=changed_failure,
        degraded=np.asarray(run.trace.degraded) | changed_failure,
    )
    with np.testing.assert_raises_regex(ValueError, "true-state transition does not replay"):
        _validate_trace_physical_evidence(
            near_threshold_tamper, tape, config, condition=assignment.condition
        )

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


def test_physical_trace_binds_exact_motor_names_and_executed_filtered_bounds() -> None:
    config = _small_config()
    assignment = _assignments(("nominal_only", "analytic_cbf_hocbf"))[0]
    tape = generate_condition_tape(
        assignment.condition,
        config,
        seed=assignment.scenario_root_seed,
        fold=assignment.scenario_fold,
    )
    run = run_trial(assignment, tape, config)

    changed_names = np.asarray(run.trace.control_names).copy()
    changed_names[0] = "motor_0"
    with pytest.raises(ValueError, match="four-motor schema"):
        _validate_trace_physical_evidence(
            replace(run.trace, control_names=changed_names),
            tape,
            config,
            condition=assignment.condition,
        )

    below_minimum = np.asarray(run.trace.filtered_control).copy()
    below_minimum[0, 0] = -1.0
    with pytest.raises(ValueError, match="outside tracked per-rotor thrust bounds"):
        _validate_trace_physical_evidence(
            replace(run.trace, filtered_control=below_minimum),
            tape,
            config,
            condition=assignment.condition,
        )

    above_maximum = np.asarray(run.trace.filtered_control).copy()
    above_maximum[0, 1] = 1.0
    with pytest.raises(ValueError, match="outside tracked per-rotor thrust bounds"):
        _validate_trace_physical_evidence(
            replace(run.trace, filtered_control=above_maximum),
            tape,
            config,
            condition=assignment.condition,
        )

    nonfinite = np.asarray(run.trace.filtered_control).copy()
    nonfinite[0, 2] = np.nan
    nonfinite_trace = replace(run.trace)
    object.__setattr__(nonfinite_trace, "filtered_control", nonfinite)
    with pytest.raises(ValueError, match="not finite four-rotor evidence"):
        _validate_trace_physical_evidence(
            nonfinite_trace, tape, config, condition=assignment.condition
        )

    terminal_is_not_executed = np.asarray(run.trace.filtered_control).copy()
    terminal_is_not_executed[-1] = np.asarray((-1.0, 1.0, np.nan, np.inf))
    terminal_trace = replace(run.trace)
    object.__setattr__(terminal_trace, "filtered_control", terminal_is_not_executed)
    _validate_trace_physical_evidence(terminal_trace, tape, config, condition=assignment.condition)


def test_physical_replay_binds_nonunit_per_rotor_efficiency_and_both_control_records() -> None:
    config = _small_config()
    assignment = _assignments(("nominal_only", "analytic_cbf_hocbf"), "dynamics_change")[0]
    tape = generate_condition_tape(
        assignment.condition,
        config,
        seed=assignment.scenario_root_seed,
        fold=assignment.scenario_fold,
    )
    executed_efficiency = np.asarray(tape.rotor_efficiency[: config.control_steps - 1])
    assert np.any(executed_efficiency != 1.0)
    run = run_trial(assignment, tape, config)

    _validate_trace_physical_evidence(run.trace, tape, config, condition=assignment.condition)

    changed_applied = np.array(run.trace.applied_control, copy=True)
    changed_applied[1, 0] += 1e-4
    with np.testing.assert_raises_regex(ValueError, "applied control does not match"):
        _validate_trace_physical_evidence(
            replace(run.trace, applied_control=changed_applied),
            tape,
            config,
            condition=assignment.condition,
        )

    changed_filtered = np.array(run.trace.filtered_control, copy=True)
    changed_filtered[1, 1] += 1e-4
    with np.testing.assert_raises_regex(ValueError, "applied control does not match"):
        _validate_trace_physical_evidence(
            replace(run.trace, filtered_control=changed_filtered),
            tape,
            config,
            condition=assignment.condition,
        )


def test_physical_replay_preserves_scalar_plant_execution_geometry() -> None:
    # This retained high-cancellation witness differs by 4.75e-5 in angular velocity when GPU XLA
    # is allowed to turn the scalar plant into a batched matmul.  The replay must instead preserve
    # the production scalar body's geometry; the existing eight-epsilon bound then remains strict.
    state = jnp.asarray(
        (
            0.54786408,
            0.89792085,
            2.4741976,
            -0.02637339,
            0.0048929313,
            -0.00019761101,
            0.99964017,
            0.21714544,
            0.53238428,
            0.56250781,
            0.064965561,
            -0.0010001627,
            0.0001234099,
        ),
        dtype=jnp.float32,
    )
    command = jnp.asarray((0.10483667, 0.10505788, 0.10739353, 0.10754688), dtype=jnp.float32)
    base_model, (arm_length, thrust_to_torque, mixing_matrix) = _base_airborne_plant()
    actuator = VersionAActuator(
        arm_length=arm_length,
        thrust_to_torque=thrust_to_torque,
        mixing_matrix=mixing_matrix,
        thrust_min=jnp.zeros(4, dtype=jnp.float32),
        thrust_max=jnp.ones(4, dtype=jnp.float32),
    )
    scalar_plant = jax.jit(
        lambda candidate_state, candidate_command, model, efficiency: _plant_step(
            candidate_state, candidate_command, model, efficiency, actuator, 0.02
        )
    )
    reference_state, reference_motor = scalar_plant(
        state, command, base_model, jnp.ones(4, dtype=jnp.float32)
    )

    count = 150
    replayed_state, replayed_motor = _replay_airborne_transitions(
        jnp.broadcast_to(state, (count, 13)),
        jnp.broadcast_to(command, (count, 4)),
        jnp.ones((count, 4), dtype=jnp.float32),
        jnp.ones(count, dtype=jnp.float32),
        jnp.ones((count, 3), dtype=jnp.float32),
        jnp.zeros((count, 3), dtype=jnp.float32),
        base_model,
        arm_length,
        thrust_to_torque,
        mixing_matrix,
        0.02,
    )
    actual = np.asarray(replayed_state, dtype=np.float64)
    expected = np.broadcast_to(np.asarray(reference_state, dtype=np.float64), actual.shape)
    epsilon = float(np.finfo(np.float32).eps)
    tolerance = 8.0 * epsilon * (1.0 + np.maximum(np.abs(actual), np.abs(expected)))
    assert np.all(np.abs(actual - expected) <= tolerance)
    np.testing.assert_array_equal(
        np.asarray(replayed_motor),
        np.broadcast_to(np.asarray(reference_motor), np.asarray(replayed_motor).shape),
    )


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


def test_full_development_slice_logs_model_and_candidate_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    provenance = collect_provenance(Path(__file__).resolve().parents[4])
    _validate_runtime_device_roles(run.events, provenance)
    names = {event.name for event in run.events}

    assert "candidate_submitted" in names
    assert any(name.startswith("candidate_") and name != "candidate_submitted" for name in names)
    assert "model_version_advanced" in names
    estimator_event = next(
        event for event in run.events if event.name == "estimator_update_executed"
    )
    assert estimator_event.details["event_only_latency_sample"] is True
    assert estimator_event.details["execution_seconds"] > 0.0
    runtime_device_event = next(
        event for event in run.events if event.name == "runtime_inputs_precomputed"
    )
    assert runtime_device_event.details["estimator_device"] == str(jax.devices("cpu")[0])
    for field, message in (
        ("controller_device", "controller/plant device"),
        ("estimator_device", "estimator device"),
    ):
        tampered_runtime_events = tuple(
            replace(event, details={**event.details, field: "unbound-device"})
            if event is runtime_device_event
            else event
            for event in run.events
        )
        with pytest.raises(ValueError, match=message):
            _validate_runtime_device_roles(tampered_runtime_events, provenance)
    with pytest.raises(ValueError, match="exactly one"):
        _validate_runtime_device_roles((*run.events, runtime_device_event), provenance)
    assert np.max(run.trace.model_version) > 0
    assert run.trace.policy_values.shape == (3, 16)
    np.testing.assert_array_equal(run.trace.executed_control, np.asarray([True, True, False]))
    assert run.trace.gradient_norm[-1] == 0.0
    cold = next(event for event in run.events if event.category == "cold_start")
    assert cold.details["bptt_execution_scope"] == "compiled_burst_only"
    assert cold.details["bptt_compilation_excluded_from_execution_timing"] is True
    assert cold.details["bptt_execution_contract"] == BPTT_EXECUTION_CONTRACT
    assert cold.details["execution_device_is_cpu"] is True
    assert cold.details["execution_device_is_gpu"] is False
    assert cold.details["bptt_execution_backend"] == "cpu"
    assert cold.details["bptt_cache_key"].startswith("online:cpu:")
    assert cold.details["bptt_compiled_cache_hit"] is False
    assert len(cold.details["bptt_input_digest"]) == 64
    assert cold.details["validation_execution_synchronized"] is True
    scheduler = next(event for event in run.events if event.name == "logical_simulation_scheduler")
    assert scheduler.details["host_load_can_change_event_step"] is False
    assert scheduler.details["real_time_claim_eligible"] is False
    assert scheduler.details["compiled_execution_backend"] == "cpu"
    assert scheduler.details["cache_key"].startswith("online:cpu:")
    assert scheduler.details["cache_hit"] is True
    assert scheduler.details["gpu_adaptation_included_in_wall_step"] is False
    assert scheduler.details["cpu_adaptation_included_in_wall_step"] is True
    online_decisions = tuple(
        event
        for event in run.events
        if event.category == "adaptation"
        and event.name in {"candidate_admitted", "candidate_rejected"}
    )
    assert online_decisions
    assert all(event.details["execution_device_is_cpu"] is True for event in online_decisions)
    assert all(event.details["execution_device_is_gpu"] is False for event in online_decisions)
    assert all(event.details["bptt_execution_backend"] == "cpu" for event in online_decisions)
    assert all(
        event.details["bptt_cache_key"].startswith("online:cpu:") for event in online_decisions
    )
    assert all(len(event.details["bptt_input_digest"]) == 64 for event in online_decisions)
    assert np.any(run.dashboard_evidence.fallback_rollout_available[:-1])
    assert np.any(run.dashboard_evidence.selected_rollout_available[:-1])
    replay_with_contexts = _replay_dashboard_dynamics_and_contexts(
        run.trace, tape, assignment.condition, assignment.method, config
    )
    replayed = replay_with_contexts[:4]
    assert all(
        leaf.device.platform == "cpu"
        for leaf in jax.tree.leaves((replay_with_contexts[4], replay_with_contexts[5]))
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
    assert all(
        proof.candidate.metadata["bptt_input_digest"]
        == next(
            event.details["bptt_input_digest"]
            for event in run.events
            if event.details.get("candidate_digest") == proof.candidate.digest
            and event.name == f"candidate_{proof.status}"
        )
        for proof in run.adaptation_evidence.decisions
    )

    def forbid_cross_process_replay(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("BPTT/hard-evidence kernels must not be replayed by the validator")

    monkeypatch.setattr(
        adaptation_evidence_module,
        "_compile_candidate_evidence_replay",
        forbid_cross_process_replay,
    )
    validate_adaptation_evidence_binding(
        run.adaptation_evidence,
        run.trace,
        run.events,
        shared_stochastic_seed=assignment.shared_stochastic_seed,
        tape=tape,
        condition=assignment.condition,
        method=assignment.method,
        config=config,
        provenance=provenance,
    )
    alternate_cpu_provenance = json.loads(json.dumps(provenance))
    alternate_cpu = "TFRT_CPU_999"
    alternate_cpu_provenance["jax"]["devices"].append(alternate_cpu)
    alternate_cpu_provenance["jax"]["cpu_devices"].append(alternate_cpu)
    for role in ("controller", "plant", "bptt", "validation"):
        alternate_cpu_provenance["jax"]["role_devices"][role] = alternate_cpu
    with pytest.raises(ValueError, match="BPTT device does not match run provenance"):
        validate_adaptation_evidence_binding(
            run.adaptation_evidence,
            run.trace,
            run.events,
            shared_stochastic_seed=assignment.shared_stochastic_seed,
            provenance=alternate_cpu_provenance,
        )
    first_decision = next(
        event
        for event in run.events
        if event.category == "cold_start"
        and event.name in {"candidate_admitted", "candidate_rejected"}
    )
    for field, value, message in (
        ("validation_execution_device", "unbound-device", "validation device"),
        ("validation_cache_key", "evidence:unbound:gpu:0", "device/cache identity"),
        ("validation_execution_synchronized", False, "device/cache identity"),
    ):
        tampered_validation_events = tuple(
            replace(event, details={**event.details, field: value})
            if event is first_decision
            else event
            for event in run.events
        )
        with pytest.raises(ValueError, match=message):
            validate_adaptation_evidence_binding(
                run.adaptation_evidence,
                run.trace,
                tampered_validation_events,
                shared_stochastic_seed=assignment.shared_stochastic_seed,
                provenance=provenance,
            )
    for field, value in (
        ("admission_runtime_scope", "unbound-scope"),
        ("admission_publication_included", True),
        ("bptt_compilation_excluded_from_execution_timing", False),
    ):
        tampered_timing_scope_events = tuple(
            replace(event, details={**event.details, field: value})
            if event is first_decision
            else event
            for event in run.events
        )
        with pytest.raises(ValueError, match="timing scope/accounting"):
            validate_adaptation_evidence_binding(
                run.adaptation_evidence,
                run.trace,
                tampered_timing_scope_events,
                shared_stochastic_seed=assignment.shared_stochastic_seed,
            )
    for field, value in (
        ("report_passed", not first_decision.details["report_passed"]),
        ("failed_gates", ["forged_gate"]),
        ("admission_margin", float(first_decision.details["admission_margin"]) + 1.0),
        (
            "minimum_coverage_threshold",
            float(first_decision.details["minimum_coverage_threshold"]) + 0.1,
        ),
    ):
        tampered_gate_events = tuple(
            replace(event, details={**event.details, field: value})
            if event is first_decision
            else event
            for event in run.events
        )
        with pytest.raises(ValueError, match="gate diagnostics"):
            validate_adaptation_evidence_binding(
                run.adaptation_evidence,
                run.trace,
                tampered_gate_events,
                shared_stochastic_seed=assignment.shared_stochastic_seed,
            )
    original_proof = run.adaptation_evidence.decisions[-1]
    weakened_thresholds = replace(
        original_proof.thresholds,
        maximum_runtime_seconds=original_proof.thresholds.maximum_runtime_seconds + 1.0,
    )
    weakened_report = hard_validate_candidate(
        original_proof.proposal_active,
        original_proof.candidate,
        original_proof.evidence,
        weakened_thresholds,
        current_model_version=original_proof.report.model_version,
    )
    weakened_publication = original_proof.publication_active
    weakened_reason = original_proof.publication_reason
    if original_proof.status != "expired":
        weakened_store = ActiveSnapshotStore(original_proof.decision_active)
        if original_proof.decision_model_version > weakened_store.model_version:
            weakened_store.advance_model_version(original_proof.decision_model_version)
        replayed_publication = weakened_store.admit(original_proof.candidate, weakened_report)
        assert replayed_publication.accepted is (original_proof.status == "admitted")
        weakened_publication = replayed_publication.active
        weakened_reason = replayed_publication.reason
    weakened_proof = replace(
        original_proof,
        thresholds=weakened_thresholds,
        report=weakened_report,
        publication_active=weakened_publication,
        publication_reason=weakened_reason,
    )
    weakened_evidence = replace(
        run.adaptation_evidence, decisions=(*run.adaptation_evidence.decisions[:-1], weakened_proof)
    )
    weakened_events = tuple(
        replace(
            event,
            details={
                **event.details,
                "report_digest": weakened_report.digest,
                "reason": weakened_reason,
                "report_passed": weakened_report.passed,
                "failed_gates": list(weakened_report.failed_gate_names),
                "admission_margin": float(
                    np.min(
                        np.asarray(weakened_report.candidate_local_best)
                        - np.asarray(weakened_report.active_local_best)
                        + weakened_thresholds.local_non_regression_tolerance
                    )
                ),
                "minimum_coverage_threshold": weakened_thresholds.minimum_coverage,
                "minimum_redundancy_threshold": weakened_thresholds.minimum_redundancy,
                "minimum_diversity_threshold": weakened_thresholds.minimum_diversity,
                "retention_tolerance": weakened_thresholds.local_non_regression_tolerance,
            },
        )
        if event.details.get("candidate_digest") == original_proof.candidate.digest
        and event.name == f"candidate_{original_proof.status}"
        else event
        for event in run.events
    )
    with pytest.raises(ValueError, match="thresholds do not match the configuration"):
        validate_adaptation_evidence_binding(
            weakened_evidence,
            run.trace,
            weakened_events,
            shared_stochastic_seed=assignment.shared_stochastic_seed,
            tape=tape,
            condition=assignment.condition,
            method=assignment.method,
            config=config,
        )
    tampered_runtime_evidence = replace(
        original_proof.evidence,
        runtime_seconds=np.asarray(
            [float(np.asarray(original_proof.evidence.runtime_seconds).item()) + 0.5]
        ),
    )
    tampered_runtime_report = hard_validate_candidate(
        original_proof.proposal_active,
        original_proof.candidate,
        tampered_runtime_evidence,
        original_proof.thresholds,
        current_model_version=original_proof.report.model_version,
    )
    tampered_runtime_publication = original_proof.publication_active
    tampered_runtime_reason = original_proof.publication_reason
    if original_proof.status != "expired":
        tampered_runtime_store = ActiveSnapshotStore(original_proof.decision_active)
        if original_proof.decision_model_version > tampered_runtime_store.model_version:
            tampered_runtime_store.advance_model_version(original_proof.decision_model_version)
        replayed_runtime_publication = tampered_runtime_store.admit(
            original_proof.candidate, tampered_runtime_report
        )
        assert replayed_runtime_publication.accepted is (original_proof.status == "admitted")
        tampered_runtime_publication = replayed_runtime_publication.active
        tampered_runtime_reason = replayed_runtime_publication.reason
    tampered_runtime_proof = replace(
        original_proof,
        evidence=tampered_runtime_evidence,
        report=tampered_runtime_report,
        publication_active=tampered_runtime_publication,
        publication_reason=tampered_runtime_reason,
    )
    tampered_runtime_artifact = replace(
        run.adaptation_evidence,
        decisions=(*run.adaptation_evidence.decisions[:-1], tampered_runtime_proof),
    )
    tampered_runtime_events = tuple(
        replace(
            event,
            details={
                **event.details,
                "report_digest": tampered_runtime_report.digest,
                "reason": tampered_runtime_reason,
                "report_passed": tampered_runtime_report.passed,
                "failed_gates": list(tampered_runtime_report.failed_gate_names),
            },
        )
        if event.details.get("candidate_digest") == original_proof.candidate.digest
        and event.name == f"candidate_{original_proof.status}"
        else event
        for event in run.events
    )
    with pytest.raises(ValueError, match="gate runtime.*decision event"):
        validate_adaptation_evidence_binding(
            tampered_runtime_artifact,
            run.trace,
            tampered_runtime_events,
            shared_stochastic_seed=assignment.shared_stochastic_seed,
        )

    tampered_contract_events = tuple(
        replace(event, details={**event.details, "bptt_execution_contract": "unbound-contract"})
        if event.category == "cold_start"
        else event
        for event in run.events
    )
    with pytest.raises(ValueError, match="execution contract"):
        validate_adaptation_evidence_binding(
            run.adaptation_evidence,
            run.trace,
            tampered_contract_events,
            shared_stochastic_seed=assignment.shared_stochastic_seed,
            tape=tape,
            condition=assignment.condition,
            method=assignment.method,
            config=config,
        )
    tampered_input_events = tuple(
        replace(event, details={**event.details, "bptt_input_digest": "0" * 64})
        if event.category == "cold_start"
        else event
        for event in run.events
    )
    with pytest.raises(ValueError, match="bptt_input_digest"):
        validate_adaptation_evidence_binding(
            run.adaptation_evidence,
            run.trace,
            tampered_input_events,
            shared_stochastic_seed=assignment.shared_stochastic_seed,
            tape=tape,
            condition=assignment.condition,
            method=assignment.method,
            config=config,
        )
    tampered_isolation_events = tuple(
        replace(event, details={**event.details, "execution_device_is_cpu": False})
        if event.category in {"cold_start", "adaptation"}
        and event.name in {"candidate_admitted", "candidate_rejected"}
        else event
        for event in run.events
    )
    with pytest.raises(ValueError, match="execution-device evidence"):
        validate_adaptation_evidence_binding(
            run.adaptation_evidence,
            run.trace,
            tampered_isolation_events,
            shared_stochastic_seed=assignment.shared_stochastic_seed,
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
