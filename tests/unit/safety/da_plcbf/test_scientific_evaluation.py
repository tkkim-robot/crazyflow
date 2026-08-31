from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from crazyflow.safety.da_plcbf.artifact_smoke import synthetic_trace
from crazyflow.safety.da_plcbf.scientific_evaluation import (
    MINIMUM_FINAL_PAIRED_TRIALS,
    AnalysisRole,
    BoundaryVariable,
    FalsificationAxis,
    LatencyMetrics,
    MetricDirection,
    PairedInferenceConfig,
    PairedTrialDataset,
    ScientificTrialMetrics,
    ScientificTrialRecord,
    TrialStatus,
    compare_paired_metric,
    confirmatory_bootstrap_replicates,
    derive_scientific_metrics,
    exact_binomial_interval,
    generate_boundary_candidates,
    make_paired_trial_schedule,
    operational_failure_rate,
    rng_provenance,
    scientific_event_rate,
    wilson_rate_interval,
)


def _tape_digest(condition: str, fold: int) -> str:
    return hashlib.sha256(f"{condition}:{fold}".encode()).hexdigest()


def _metrics(
    score: float, *, failed: bool = False, degraded_without_failure: bool = False
) -> ScientificTrialMetrics:
    degraded = failed or degraded_without_failure
    result = ScientificTrialMetrics(
        steps=2,
        duration_seconds=1.0,
        interval_safety_evidence=True,
        warm_execution_excludes_compilation=True,
        collision_steps=int(failed),
        constraint_violation_steps=int(failed),
        failure_steps=int(failed),
        any_collision=failed,
        any_constraint_violation=failed,
        any_failure=failed,
        minimum_hard_margin=score,
        certified_state_fraction=float(not failed),
        certified_time_fraction=float(not failed),
        degraded_state_fraction=float(degraded),
        degraded_duration_seconds=float(degraded),
        mean_intervention_norm=0.1,
        maximum_intervention_norm=0.2,
        intervention_integral=0.1,
        policy_switches=0,
        policy_switch_rate_hz=0.0,
        mean_normalized_estimation_error=0.1,
        maximum_normalized_estimation_error=0.2,
        recoveries=(),
        latencies=(LatencyMetrics("filter", 2, 0.1, 0.15, 0.19, 0.2, 0.25, 0),),
    )
    result.validate()
    return result


def test_operational_failure_counts_declared_degradation_but_not_nominal_unavailability() -> None:
    common = {
        "condition": "static",
        "fold": 0,
        "pairing_id": "a" * 64,
        "scenario_tape_sha256": "b" * 64,
        "status": TrialStatus.COMPLETE,
        "metrics": _metrics(0.1, degraded_without_failure=True),
    }
    assert ScientificTrialRecord(method="da_plcbf_full", **common).operational_failure
    assert not ScientificTrialRecord(method="nominal_only", **common).operational_failure

    physical = {**common, "metrics": _metrics(-0.1, failed=True)}
    assert ScientificTrialRecord(method="nominal_only", **physical).operational_failure


def _complete_dataset(
    *,
    trials: int,
    intended_for_final_claim: bool,
    candidate_scores: np.ndarray | None = None,
    execution_failure_fold: int | None = None,
) -> PairedTrialDataset:
    schedule = make_paired_trial_schedule(
        root_seed=42,
        methods=("candidate", "baseline"),
        conditions=("wind-change",),
        trials_per_condition=trials,
        intended_for_final_claim=intended_for_final_claim,
    )
    if candidate_scores is None:
        candidate_scores = np.ones(trials)
    records = []
    for assignment in schedule.assignments:
        execution_failure = (
            assignment.method == "candidate" and assignment.fold == execution_failure_fold
        )
        records.append(
            ScientificTrialRecord(
                method=assignment.method,
                condition=assignment.condition,
                fold=assignment.fold,
                pairing_id=assignment.pairing_id,
                scenario_tape_sha256=_tape_digest(assignment.condition, assignment.fold),
                status=(
                    TrialStatus.EXECUTION_FAILURE if execution_failure else TrialStatus.COMPLETE
                ),
                metrics=(
                    None
                    if execution_failure
                    else _metrics(
                        float(candidate_scores[assignment.fold])
                        if assignment.method == "candidate"
                        else 0.0
                    )
                ),
                failure_code="worker-crash" if execution_failure else None,
            )
        )
    dataset = PairedTrialDataset(schedule, tuple(records))
    dataset.validate()
    return dataset


def test_final_schedule_requires_100_pairs_and_has_matched_named_seeds() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        make_paired_trial_schedule(
            root_seed=7,
            methods=("candidate", "baseline"),
            conditions=("static",),
            trials_per_condition=99,
            intended_for_final_claim=True,
        )

    first = make_paired_trial_schedule(
        root_seed=7,
        methods=("candidate", "baseline"),
        conditions=("static", "ball"),
        trials_per_condition=MINIMUM_FINAL_PAIRED_TRIALS,
        intended_for_final_claim=True,
    )
    second = make_paired_trial_schedule(
        root_seed=7,
        methods=("candidate", "baseline"),
        conditions=("static", "ball"),
        trials_per_condition=MINIMUM_FINAL_PAIRED_TRIALS,
        intended_for_final_claim=True,
    )
    assert first == second
    assert first.final_claim_eligible
    matched = [item for item in first.assignments if item.condition == "static" and item.fold == 3]
    assert len(matched) == 2
    assert matched[0].pairing_id == matched[1].pairing_id
    assert matched[0].scenario_root_seed == matched[1].scenario_root_seed
    assert matched[0].shared_stochastic_seed == matched[1].shared_stochastic_seed
    assert matched[0].method_stochastic_seed != matched[1].method_stochastic_seed
    assert {name for name, _ in first.named_stream_ids} == {
        "scenario_tape",
        "paired_runtime",
        "method_runtime",
    }


def test_named_rng_provenance_is_complete_and_tamper_evident() -> None:
    provenance = rng_provenance(17, "paired_bootstrap", "static", "margin", 2_000)
    provenance.validate()
    assert provenance.labels == ("static", "margin", "2000")
    with pytest.raises(ValueError, match="does not match"):
        replace(provenance, labels=("different",)).validate()


def test_dataset_requires_every_failure_record_and_identical_tape_hashes() -> None:
    dataset = _complete_dataset(trials=3, intended_for_final_claim=False)
    with pytest.raises(ValueError, match="complete schedule"):
        PairedTrialDataset(dataset.schedule, dataset.records[:-1]).validate()

    records = list(dataset.records)
    records[1] = replace(records[1], scenario_tape_sha256="f" * 64)
    with pytest.raises(ValueError, match="identical scenario tape"):
        PairedTrialDataset(dataset.schedule, tuple(records)).validate()


def test_scientific_metrics_use_hard_mask_continuous_duration_and_suffix_recovery() -> None:
    trace = synthetic_trace("a" * 64, steps=6, dt=0.1)
    barriers = np.array(trace.hard_barriers, copy=True)
    barriers[2, 0] = -0.1
    contact = np.zeros(6, dtype=np.bool_)
    contact[3] = True
    failure = np.zeros(6, dtype=np.bool_)
    failure[2:4] = True
    degraded = np.zeros(6, dtype=np.bool_)
    degraded[1:3] = True
    selected = np.asarray((0, 0, 1, 1, -1, 0), dtype=np.int64)
    correction = np.broadcast_to(np.asarray((3.0, 4.0, 0.0, 0.0)), (6, 4))
    trace = replace(
        trace,
        hard_barriers=barriers,
        contact=contact,
        failure=failure,
        degraded=degraded,
        selected_policy=selected,
        filtered_control=trace.nominal_control + correction,
        applied_control=trace.nominal_control + correction,
    )
    certified = np.asarray(
        ((True, False), (False, False), (False, False), (True, False), (False, True), (True, True)),
        dtype=np.bool_,
    )
    estimation_error = np.broadcast_to(np.asarray((2.0, 4.0)), (6, 2))
    deadlines = {str(name): 0.01 for name in trace.latency_names}
    metrics = derive_scientific_metrics(
        trace,
        hard_certified_policy=certified,
        estimation_error=estimation_error,
        estimation_scale=np.asarray((2.0, 4.0)),
        change_indices=(1,),
        latency_deadlines_seconds=deadlines,
        interval_safety_evidence=True,
        warm_execution_excludes_compilation=True,
    )

    assert metrics.collision_steps == 1
    assert metrics.constraint_violation_steps == 1
    assert metrics.failure_steps == 2
    assert metrics.minimum_hard_margin == -0.1
    assert metrics.certified_state_fraction == pytest.approx(4 / 6)
    assert metrics.certified_time_fraction == pytest.approx(0.6)
    assert metrics.degraded_state_fraction == pytest.approx(2 / 6)
    assert metrics.degraded_duration_seconds == pytest.approx(0.2)
    assert metrics.mean_intervention_norm == pytest.approx(5.0)
    assert metrics.intervention_integral == pytest.approx(2.5)
    assert metrics.policy_switches == 1
    assert metrics.policy_switch_rate_hz == pytest.approx(2.0)
    assert metrics.mean_normalized_estimation_error == pytest.approx(np.sqrt(2.0))
    assert len(metrics.recoveries) == 1
    recovery = metrics.recoveries[0]
    assert recovery.change_index == 1
    assert recovery.change_time_seconds == pytest.approx(0.1)
    assert recovery.recovered_through_horizon
    assert recovery.recovery_time_seconds == pytest.approx(0.2)
    assert recovery.censor_time_seconds == pytest.approx(0.4)
    assert all(latency.count == 6 for latency in metrics.latencies)


def test_terminal_no_control_sentinel_is_excluded_without_dropping_final_interval() -> None:
    trace = synthetic_trace("a" * 64, steps=5, dt=0.1)
    zeros_control = np.zeros_like(trace.nominal_control)
    applied = np.zeros_like(trace.applied_control)
    applied[:4, 0] = np.asarray((1.0, 2.0, 3.0, 4.0))
    executed = np.asarray((True, True, True, True, False), dtype=np.bool_)
    selected = np.asarray((0, 0, 1, 1, -1), dtype=np.int64)
    zero_policy = np.zeros_like(trace.policy_values)
    zero_latency = np.array(trace.component_latency_seconds, copy=True)
    zero_latency[-1] = 0.0
    trace = replace(
        trace,
        nominal_control=zeros_control,
        filtered_control=applied,
        applied_control=applied,
        executed_control=executed,
        training_values=zero_policy,
        policy_values=zero_policy,
        selected_policy=selected,
        solver_kkt_residual=np.zeros(5),
        postcheck_residual=np.zeros(5),
        clipped=np.zeros(5, dtype=np.bool_),
        saturated=np.zeros(5, dtype=np.bool_),
        gradient_norm=np.zeros(5),
        component_latency_seconds=zero_latency,
    )
    certified = np.asarray(((False,), (False,), (True,), (True,), (False,)), dtype=np.bool_)
    metrics = derive_scientific_metrics(
        trace,
        hard_certified_policy=certified,
        estimation_error=np.zeros((5, 1)),
        estimation_scale=np.ones(1),
        change_indices=(1,),
        latency_deadlines_seconds={str(name): 0.01 for name in trace.latency_names},
        interval_safety_evidence=True,
        warm_execution_excludes_compilation=True,
    )

    assert metrics.mean_intervention_norm == pytest.approx(2.5)
    assert metrics.maximum_intervention_norm == pytest.approx(4.0)
    assert metrics.intervention_integral == pytest.approx(1.0)
    assert metrics.certified_state_fraction == pytest.approx(0.5)
    assert metrics.certified_time_fraction == pytest.approx(0.5)
    assert metrics.recoveries[0].recovery_time_seconds == pytest.approx(0.1)
    assert metrics.recoveries[0].censor_time_seconds == pytest.approx(0.3)
    assert all(latency.count == 4 for latency in metrics.latencies)


def test_scientific_metrics_refuse_soft_or_implicit_certification_and_mixed_scales() -> None:
    trace = synthetic_trace("a" * 64, steps=4, dt=0.1)
    deadlines = {str(name): 0.01 for name in trace.latency_names}
    with pytest.raises(ValueError, match="boolean"):
        derive_scientific_metrics(
            trace,
            hard_certified_policy=np.ones((4, 2)),
            estimation_error=np.zeros((4, 2)),
            estimation_scale=np.ones(2),
            latency_deadlines_seconds=deadlines,
            interval_safety_evidence=True,
            warm_execution_excludes_compilation=True,
        )
    with pytest.raises(ValueError, match="positive"):
        derive_scientific_metrics(
            trace,
            hard_certified_policy=np.ones((4, 2), dtype=np.bool_),
            estimation_error=np.zeros((4, 2)),
            estimation_scale=np.asarray((1.0, 0.0)),
            latency_deadlines_seconds=deadlines,
            interval_safety_evidence=True,
            warm_execution_excludes_compilation=True,
        )
    with pytest.raises(ValueError, match="substep checks"):
        derive_scientific_metrics(
            trace,
            hard_certified_policy=np.ones((4, 2), dtype=np.bool_),
            estimation_error=np.zeros((4, 2)),
            estimation_scale=np.ones(2),
            latency_deadlines_seconds=deadlines,
            interval_safety_evidence=False,
            warm_execution_excludes_compilation=True,
        )


def test_wilson_and_exact_binomial_intervals_cover_edge_cases() -> None:
    zero = wilson_rate_interval(0, 100)
    assert zero.rate == 0.0
    assert zero.lower == pytest.approx(0.0)
    assert 0.0 < zero.upper < 0.05
    all_success = exact_binomial_interval(100, 100)
    assert all_success.estimate == 1.0
    assert all_success.lower is not None and all_success.lower > 0.95
    assert all_success.upper == 1.0
    unavailable = exact_binomial_interval(0, 0)
    assert unavailable.estimate is None
    assert unavailable.lower is None


def test_paired_superiority_needs_both_adjusted_intervals_and_predeclared_schedule() -> None:
    dataset = _complete_dataset(trials=100, intended_for_final_claim=True)
    protocol = PairedInferenceConfig(
        confidence_level=0.95,
        bootstrap_replicates=2_000,
        familywise_comparisons=2,
        minimum_oriented_effect=0.2,
    )
    result = compare_paired_metric(
        dataset,
        condition="wind-change",
        candidate_method="candidate",
        baseline_method="baseline",
        metric_name="minimum_hard_margin",
        direction=MetricDirection.HIGHER_IS_BETTER,
        inference=protocol,
    )
    assert result.superiority_supported
    assert result.final_claim_eligible
    assert result.analysis_role is AnalysisRole.CONFIRMATORY
    assert result.bootstrap_distribution_degenerate
    assert result.bootstrap_resolution_sufficient
    assert result.bootstrap_interval == pytest.approx((1.0, 1.0))
    assert result.candidate_wins == 100
    assert result.win_probability_interval.lower is not None
    assert result.win_probability_interval.lower > 0.5
    assert result.bootstrap_rng == rng_provenance(
        42,
        "paired_bootstrap",
        "wind-change",
        "minimum_hard_margin",
        "candidate",
        "baseline",
        "higher_is_better",
        2_000,
        2,
    )

    exploratory = _complete_dataset(trials=100, intended_for_final_claim=False)
    exploratory_result = compare_paired_metric(
        exploratory,
        condition="wind-change",
        candidate_method="candidate",
        baseline_method="baseline",
        metric_name="minimum_hard_margin",
        direction=MetricDirection.HIGHER_IS_BETTER,
        inference=protocol,
    )
    assert not exploratory_result.superiority_supported
    assert "not eligible" in exploratory_result.conclusion


def test_bootstrap_resolution_is_sized_for_the_confirmatory_family() -> None:
    assert confirmatory_bootstrap_replicates(1) == 20_000
    assert confirmatory_bootstrap_replicates(48) == 200_000
    assert confirmatory_bootstrap_replicates(72) == 290_000
    protocol = PairedInferenceConfig(
        bootstrap_replicates=confirmatory_bootstrap_replicates(48), familywise_comparisons=48
    )
    assert protocol.expected_bootstrap_draws_per_tail >= 100.0


def test_underresolved_nondegenerate_bootstrap_and_exploratory_role_cannot_claim() -> None:
    scores = np.linspace(0.5, 1.5, 100)
    dataset = _complete_dataset(trials=100, intended_for_final_claim=True, candidate_scores=scores)
    underresolved = compare_paired_metric(
        dataset,
        condition="wind-change",
        candidate_method="candidate",
        baseline_method="baseline",
        metric_name="minimum_hard_margin",
        direction=MetricDirection.HIGHER_IS_BETTER,
        inference=PairedInferenceConfig(bootstrap_replicates=20_000, familywise_comparisons=144),
    )
    assert underresolved.expected_bootstrap_draws_per_tail < 4.0
    assert not underresolved.bootstrap_distribution_degenerate
    assert not underresolved.bootstrap_resolution_sufficient
    assert not underresolved.final_claim_eligible
    assert not underresolved.superiority_supported
    assert "fewer than 100 expected draws" in underresolved.conclusion

    exploratory = compare_paired_metric(
        dataset,
        condition="wind-change",
        candidate_method="candidate",
        baseline_method="baseline",
        metric_name="minimum_hard_margin",
        direction=MetricDirection.HIGHER_IS_BETTER,
        inference=PairedInferenceConfig(analysis_role=AnalysisRole.EXPLORATORY),
    )
    assert exploratory.bootstrap_resolution_sufficient
    assert not exploratory.final_claim_eligible
    assert not exploratory.superiority_supported
    assert exploratory.conclusion.startswith("exploratory only")


def test_paired_sign_gate_blocks_mean_only_result() -> None:
    scores = np.concatenate((np.full(40, 10.0), np.full(60, -1.0)))
    dataset = _complete_dataset(trials=100, intended_for_final_claim=True, candidate_scores=scores)
    result = compare_paired_metric(
        dataset,
        condition="wind-change",
        candidate_method="candidate",
        baseline_method="baseline",
        metric_name="minimum_hard_margin",
        direction=MetricDirection.HIGHER_IS_BETTER,
        inference=PairedInferenceConfig(bootstrap_replicates=2_000),
    )
    assert result.mean_oriented_improvement is not None
    assert result.mean_oriented_improvement > 0.0
    assert not result.superiority_supported
    assert result.candidate_losses > result.candidate_wins


def test_execution_failure_is_retained_blocks_continuous_claim_and_counts_in_rate() -> None:
    dataset = _complete_dataset(trials=100, intended_for_final_claim=True, execution_failure_fold=9)
    result = compare_paired_metric(
        dataset,
        condition="wind-change",
        candidate_method="candidate",
        baseline_method="baseline",
        metric_name="minimum_hard_margin",
        direction=MetricDirection.HIGHER_IS_BETTER,
        inference=PairedInferenceConfig(bootstrap_replicates=1_000),
    )
    assert result.missing_metric_pairs == 1
    assert not result.superiority_supported
    assert result.bootstrap_interval is None
    assert "retained rather than dropped" in result.conclusion
    rate = operational_failure_rate(dataset, method="candidate", condition="wind-change")
    assert rate.events == 1
    assert rate.trials == 100
    assert rate.rate == pytest.approx(0.01)
    paired_failure = compare_paired_metric(
        dataset,
        condition="wind-change",
        candidate_method="candidate",
        baseline_method="baseline",
        metric_name="operational_failure",
        direction=MetricDirection.LOWER_IS_BETTER,
        inference=PairedInferenceConfig(bootstrap_replicates=1_000),
    )
    assert paired_failure.missing_metric_pairs == 0
    assert paired_failure.candidate_losses == 1
    assert not paired_failure.superiority_supported
    with pytest.raises(ValueError, match="retained execution failure"):
        scientific_event_rate(
            dataset, method="candidate", condition="wind-change", event="collision"
        )


def _boundary_variables() -> tuple[BoundaryVariable, ...]:
    return tuple(
        BoundaryVariable(
            axis=axis,
            name=axis.value,
            component_names=("value",),
            lower=(-float(index + 1),),
            upper=(float(index + 1),),
            nominal=(0.0,),
        )
        for index, axis in enumerate(FalsificationAxis)
    )


def test_boundary_candidates_are_deterministic_bounded_and_cover_both_sides() -> None:
    variables = _boundary_variables()
    first = generate_boundary_candidates(
        variables,
        count=28,
        root_seed=13,
        search_name="declared-stress",
        boundary_band_fraction=0.03,
    )
    second = generate_boundary_candidates(
        variables,
        count=28,
        root_seed=13,
        search_name="declared-stress",
        boundary_band_fraction=0.03,
    )
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.focus_component, second.focus_component)
    assert first.rng == second.rng
    assert not first.values.flags.writeable
    covered = set(zip(first.focus_component.tolist(), first.focus_side.tolist(), strict=True))
    assert covered == {(component, side) for component in range(7) for side in (-1, 1)}


def test_boundary_generator_requires_every_phase7_axis_and_sufficient_coverage() -> None:
    variables = _boundary_variables()
    with pytest.raises(ValueError, match="cover Phase-7 axes"):
        generate_boundary_candidates(
            variables[:-1], count=20, root_seed=0, search_name="missing-axis"
        )
    with pytest.raises(ValueError, match="twice"):
        generate_boundary_candidates(variables, count=13, root_seed=0, search_name="too-small")
