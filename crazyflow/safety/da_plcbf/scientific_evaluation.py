"""Paired, finite-horizon scientific evaluation for DA-PLCBF.

This module is deliberately separate from the artifact writer.  Artifact summaries are useful for
replay integrity, but a final comparison needs a predeclared matched schedule, complete failure
accounting, and paired inference.  Nothing here manufactures controller outcomes: callers must
provide one record for every scheduled method/trial pair, and missing executions remain explicit
records that prevent claims for metrics that could not be observed.

All conclusions are limited to the recorded finite simulation horizon and sampled scenario tapes.
They are not infinite-horizon, distribution-free, hardware, or real-world safety guarantees.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import beta, binomtest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from crazyflow.safety.da_plcbf.artifacts import ImmutableTrace


MINIMUM_FINAL_PAIRED_TRIALS = 100
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
MINIMUM_EXPECTED_BOOTSTRAP_TAIL_DRAWS = 100.0
BOOTSTRAP_REPLICATE_ROUNDING = 10_000
FINITE_HORIZON_CLAIM_BOUNDARY = (
    "Metric-level superiority, when supported, applies only to the predeclared matched scenario "
    "tapes, constraints, numerical tolerances, and finite simulation horizon. It is not an "
    "infinite-horizon, distribution-free, hardware, or real-world safety guarantee."
)

_SLUG = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _require_slug(value: str, name: str) -> str:
    if not isinstance(value, str) or _SLUG.fullmatch(value) is None:
        raise ValueError(f"{name} must be a portable nonempty slug")
    return value


def _require_uint32(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= np.iinfo(np.uint32).max:
        raise ValueError(f"{name} must fit uint32")
    return result


def _require_nonnegative_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite(value: float, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _stable_u64(root_seed: int, stream_name: str, *labels: object) -> int:
    root = _require_uint32(root_seed, "root_seed")
    _require_slug(stream_name, "stream_name")
    payload = "\0".join((str(root), stream_name, *(str(label) for label in labels))).encode()
    digest = hashlib.sha256(b"crazyflow.da_plcbf.evaluation.v1\0" + payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def named_stream_id(stream_name: str) -> int:
    """Return a stable uint32 identifier for a semantic evaluation RNG stream."""
    _require_slug(stream_name, "stream_name")
    digest = hashlib.sha256(f"crazyflow.da_plcbf.evaluation:{stream_name}".encode()).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


@dataclass(frozen=True, slots=True)
class RNGProvenance:
    """Complete provenance for a deterministic NumPy random stream."""

    root_seed: int
    stream_name: str
    stream_id: int
    labels: tuple[str, ...]
    derived_seed: int
    algorithm: str = "numpy.PCG64"

    def validate(self) -> None:
        """Reject inconsistent or incompletely named random streams."""
        _require_uint32(self.root_seed, "rng.root_seed")
        _require_slug(self.stream_name, "rng.stream_name")
        if self.stream_id != named_stream_id(self.stream_name):
            raise ValueError("rng.stream_id does not match stream_name")
        if not isinstance(self.labels, tuple):
            raise TypeError("rng.labels must be an immutable tuple")
        if any(not isinstance(label, str) or "\0" in label for label in self.labels):
            raise ValueError("rng.labels must be strings without null separators")
        if not isinstance(self.derived_seed, Integral) or isinstance(self.derived_seed, bool):
            raise TypeError("rng.derived_seed must be an integer")
        if not 0 <= int(self.derived_seed) <= np.iinfo(np.uint64).max:
            raise ValueError("rng.derived_seed must fit uint64")
        if int(self.derived_seed) != _stable_u64(self.root_seed, self.stream_name, *self.labels):
            raise ValueError("rng.derived_seed does not match root, stream, and labels")
        if self.algorithm != "numpy.PCG64":
            raise ValueError("only the declared numpy.PCG64 stream is supported")


def rng_provenance(root_seed: int, stream_name: str, *labels: object) -> RNGProvenance:
    """Create stable, named RNG provenance without relying on Python's randomized hash."""
    label_strings = tuple(str(label) for label in labels)
    provenance = RNGProvenance(
        root_seed=_require_uint32(root_seed, "root_seed"),
        stream_name=_require_slug(stream_name, "stream_name"),
        stream_id=named_stream_id(stream_name),
        labels=label_strings,
        derived_seed=_stable_u64(root_seed, stream_name, *label_strings),
    )
    provenance.validate()
    return provenance


@dataclass(frozen=True, slots=True)
class TrialAssignment:
    """One method run in a matched condition/fold block."""

    method: str
    condition: str
    fold: int
    pairing_id: str
    scenario_root_seed: int
    scenario_fold: int
    shared_stochastic_seed: int
    method_stochastic_seed: int

    @property
    def key(self) -> tuple[str, str, int]:
        """Return the unique method/condition/fold key."""
        return (self.method, self.condition, self.fold)

    @property
    def pair_key(self) -> tuple[str, int]:
        """Return the condition/fold matched-block key."""
        return (self.condition, self.fold)


@dataclass(frozen=True, slots=True)
class PairedTrialSchedule:
    """Immutable complete factorial method schedule with matched exogenous randomness."""

    root_seed: int
    methods: tuple[str, ...]
    conditions: tuple[str, ...]
    trials_per_condition: int
    fold_start: int
    intended_for_final_claim: bool
    assignments: tuple[TrialAssignment, ...]

    @property
    def final_claim_eligible(self) -> bool:
        """Whether the schedule was predeclared and large enough for a final claim."""
        return self.intended_for_final_claim and (
            self.trials_per_condition >= MINIMUM_FINAL_PAIRED_TRIALS
        )

    @property
    def named_stream_ids(self) -> tuple[tuple[str, int], ...]:
        """Expose every schedule RNG stream name and stable ID for artifact provenance."""
        names = ("scenario_tape", "paired_runtime", "method_runtime")
        return tuple((name, named_stream_id(name)) for name in names)

    def validate(self) -> None:
        """Validate exact pairing, stable seeds, and the final-claim sample-size gate."""
        _require_uint32(self.root_seed, "schedule.root_seed")
        if not all(
            isinstance(value, tuple) for value in (self.methods, self.conditions, self.assignments)
        ):
            raise TypeError("schedule methods, conditions, and assignments must be tuples")
        if len(self.methods) < 2 or len(set(self.methods)) != len(self.methods):
            raise ValueError("schedule requires at least two unique methods")
        if not self.conditions or len(set(self.conditions)) != len(self.conditions):
            raise ValueError("schedule requires unique nonempty conditions")
        for method in self.methods:
            _require_slug(method, "schedule method")
        for condition in self.conditions:
            _require_slug(condition, "schedule condition")
        trials = _require_nonnegative_integer(
            self.trials_per_condition, "schedule.trials_per_condition"
        )
        if trials < 1:
            raise ValueError("schedule.trials_per_condition must be positive")
        fold_start = _require_uint32(self.fold_start, "schedule.fold_start")
        if fold_start + trials - 1 > np.iinfo(np.uint32).max:
            raise ValueError("scheduled folds exceed uint32")
        if not isinstance(self.intended_for_final_claim, bool):
            raise TypeError("schedule.intended_for_final_claim must be boolean")
        if self.intended_for_final_claim and trials < MINIMUM_FINAL_PAIRED_TRIALS:
            raise ValueError(
                f"final-claim schedules require at least {MINIMUM_FINAL_PAIRED_TRIALS} paired "
                "trials per condition"
            )
        expected = _schedule_assignments(
            self.root_seed, self.methods, self.conditions, trials, fold_start
        )
        if self.assignments != expected:
            raise ValueError("schedule assignments do not match its declared deterministic design")


def _pairing_id(root_seed: int, condition: str, fold: int) -> str:
    payload = f"{root_seed}\0{condition}\0{fold}".encode()
    return hashlib.sha256(b"crazyflow.da_plcbf.pair.v1\0" + payload).hexdigest()


def _schedule_assignments(
    root_seed: int,
    methods: tuple[str, ...],
    conditions: tuple[str, ...],
    trials: int,
    fold_start: int,
) -> tuple[TrialAssignment, ...]:
    output: list[TrialAssignment] = []
    for condition in conditions:
        scenario_root = _stable_u64(root_seed, "scenario_tape", condition) & 0xFFFFFFFF
        for fold in range(fold_start, fold_start + trials):
            pair_id = _pairing_id(root_seed, condition, fold)
            shared_seed = _stable_u64(root_seed, "paired_runtime", condition, fold)
            for method in methods:
                output.append(
                    TrialAssignment(
                        method=method,
                        condition=condition,
                        fold=fold,
                        pairing_id=pair_id,
                        scenario_root_seed=scenario_root,
                        scenario_fold=fold,
                        shared_stochastic_seed=shared_seed,
                        method_stochastic_seed=_stable_u64(
                            root_seed, "method_runtime", condition, fold, method
                        ),
                    )
                )
    return tuple(output)


def make_paired_trial_schedule(
    *,
    root_seed: int,
    methods: Sequence[str],
    conditions: Sequence[str],
    trials_per_condition: int,
    fold_start: int = 0,
    intended_for_final_claim: bool = False,
) -> PairedTrialSchedule:
    """Build a deterministic matched schedule, enforcing 100 pairs for final claims."""
    if isinstance(methods, str) or isinstance(conditions, str):
        raise TypeError("methods and conditions must be sequences of slugs, not strings")
    method_tuple = tuple(methods)
    condition_tuple = tuple(conditions)
    schedule = PairedTrialSchedule(
        root_seed=root_seed,
        methods=method_tuple,
        conditions=condition_tuple,
        trials_per_condition=trials_per_condition,
        fold_start=fold_start,
        intended_for_final_claim=intended_for_final_claim,
        assignments=_schedule_assignments(
            root_seed, method_tuple, condition_tuple, trials_per_condition, fold_start
        ),
    )
    schedule.validate()
    return schedule


@dataclass(frozen=True, slots=True)
class LatencyMetrics:
    """Raw-sample-derived warm-execution tail metrics for one component."""

    component: str
    count: int
    median_seconds: float
    p95_seconds: float
    p99_seconds: float
    worst_seconds: float
    deadline_seconds: float
    deadline_misses: int

    def validate(self) -> None:
        """Validate ordering, units, and count relationships."""
        _require_slug(self.component, "latency.component")
        count = _require_nonnegative_integer(self.count, "latency.count")
        if count < 1:
            raise ValueError("latency.count must be positive")
        values = (
            _finite_nonnegative(self.median_seconds, "latency.median_seconds"),
            _finite_nonnegative(self.p95_seconds, "latency.p95_seconds"),
            _finite_nonnegative(self.p99_seconds, "latency.p99_seconds"),
            _finite_nonnegative(self.worst_seconds, "latency.worst_seconds"),
        )
        if tuple(sorted(values)) != values:
            raise ValueError("latency quantiles must be nondecreasing")
        deadline = _finite(self.deadline_seconds, "latency.deadline_seconds")
        if deadline <= 0.0:
            raise ValueError("latency.deadline_seconds must be positive")
        misses = _require_nonnegative_integer(self.deadline_misses, "latency.deadline_misses")
        if misses > count:
            raise ValueError("latency.deadline_misses cannot exceed count")


@dataclass(frozen=True, slots=True)
class RecoveryMetrics:
    """Coverage recovery after one declared dynamics-change node."""

    change_index: int
    change_time_seconds: float
    recovered_through_horizon: bool
    recovery_time_seconds: float | None
    censor_time_seconds: float

    def validate(self) -> None:
        """Validate observed versus right-censored recovery representation."""
        _require_nonnegative_integer(self.change_index, "recovery.change_index")
        _finite_nonnegative(self.change_time_seconds, "recovery.change_time_seconds")
        censor = _finite_nonnegative(self.censor_time_seconds, "recovery.censor_time_seconds")
        if not isinstance(self.recovered_through_horizon, bool):
            raise TypeError("recovery.recovered_through_horizon must be boolean")
        if self.recovered_through_horizon:
            if self.recovery_time_seconds is None:
                raise ValueError("observed recovery requires recovery_time_seconds")
            if (
                _finite_nonnegative(self.recovery_time_seconds, "recovery.recovery_time_seconds")
                > censor
            ):
                raise ValueError("recovery time cannot exceed censor time")
        elif self.recovery_time_seconds is not None:
            raise ValueError("right-censored recovery cannot contain an observed time")


@dataclass(frozen=True, slots=True)
class ScientificTrialMetrics:
    """Complete finite-horizon metrics derived from one successfully recorded trace."""

    steps: int
    duration_seconds: float
    interval_safety_evidence: bool
    warm_execution_excludes_compilation: bool
    collision_steps: int
    constraint_violation_steps: int
    failure_steps: int
    any_collision: bool
    any_constraint_violation: bool
    any_failure: bool
    minimum_hard_margin: float
    certified_state_fraction: float
    certified_time_fraction: float
    degraded_state_fraction: float
    degraded_duration_seconds: float
    mean_intervention_norm: float
    maximum_intervention_norm: float
    intervention_integral: float
    policy_switches: int
    policy_switch_rate_hz: float
    mean_normalized_estimation_error: float
    maximum_normalized_estimation_error: float
    recoveries: tuple[RecoveryMetrics, ...]
    latencies: tuple[LatencyMetrics, ...]

    def validate(self) -> None:
        """Validate scalar domains and nested metric records."""
        if not isinstance(self.recoveries, tuple) or not isinstance(self.latencies, tuple):
            raise TypeError("metrics recoveries and latencies must be immutable tuples")
        steps = _require_nonnegative_integer(self.steps, "metrics.steps")
        if steps < 2:
            raise ValueError("metrics.steps must be at least two")
        duration = _finite(self.duration_seconds, "metrics.duration_seconds")
        if duration <= 0.0:
            raise ValueError("metrics.duration_seconds must be positive")
        if self.interval_safety_evidence is not True:
            raise ValueError(
                "scientific metrics require substep clearance or conservative swept/interval minima"
            )
        if self.warm_execution_excludes_compilation is not True:
            raise ValueError("warm latency samples must exclude compilation")
        for name in ("collision_steps", "constraint_violation_steps", "failure_steps"):
            count = _require_nonnegative_integer(getattr(self, name), f"metrics.{name}")
            if count > steps:
                raise ValueError(f"metrics.{name} cannot exceed steps")
        for name in ("any_collision", "any_constraint_violation", "any_failure"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"metrics.{name} must be boolean")
        if self.any_collision != (self.collision_steps > 0):
            raise ValueError("any_collision disagrees with collision_steps")
        if self.any_constraint_violation != (self.constraint_violation_steps > 0):
            raise ValueError("any_constraint_violation disagrees with violation steps")
        if self.any_failure != (self.failure_steps > 0):
            raise ValueError("any_failure disagrees with failure_steps")
        _finite(self.minimum_hard_margin, "metrics.minimum_hard_margin")
        for name in (
            "certified_state_fraction",
            "certified_time_fraction",
            "degraded_state_fraction",
        ):
            value = _finite(getattr(self, name), f"metrics.{name}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"metrics.{name} must lie in [0, 1]")
        for name in (
            "degraded_duration_seconds",
            "mean_intervention_norm",
            "maximum_intervention_norm",
            "intervention_integral",
            "policy_switch_rate_hz",
            "mean_normalized_estimation_error",
            "maximum_normalized_estimation_error",
        ):
            _finite_nonnegative(getattr(self, name), f"metrics.{name}")
        if self.degraded_duration_seconds > duration:
            raise ValueError("degraded duration cannot exceed trial duration")
        if self.mean_intervention_norm > self.maximum_intervention_norm:
            raise ValueError("mean intervention cannot exceed maximum intervention")
        if self.mean_normalized_estimation_error > self.maximum_normalized_estimation_error:
            raise ValueError("mean estimation error cannot exceed maximum estimation error")
        _require_nonnegative_integer(self.policy_switches, "metrics.policy_switches")
        if self.policy_switches > steps - 1:
            raise ValueError("policy_switches cannot exceed adjacent node count")
        recovery_indices = [recovery.change_index for recovery in self.recoveries]
        if recovery_indices != sorted(set(recovery_indices)):
            raise ValueError("recoveries must have unique increasing change indices")
        for recovery in self.recoveries:
            recovery.validate()
        latency_names = [latency.component for latency in self.latencies]
        if not latency_names or len(latency_names) != len(set(latency_names)):
            raise ValueError("metrics.latencies must have unique nonempty component names")
        latency_counts: set[int] = set()
        for latency in self.latencies:
            latency.validate()
            latency_counts.add(latency.count)
        if len(latency_counts) != 1 or next(iter(latency_counts)) not in {steps, steps - 1}:
            raise ValueError(
                "latency components must consistently cover all nodes or a T-1 control prefix"
            )

    def scalar(self, metric_name: str) -> float:
        """Return a supported scalar without silently reducing censored or component data."""
        if metric_name in {
            "steps",
            "collision_steps",
            "constraint_violation_steps",
            "failure_steps",
            "policy_switches",
        }:
            return float(getattr(self, metric_name))
        if metric_name in {"any_collision", "any_constraint_violation", "any_failure"}:
            return float(getattr(self, metric_name))
        allowed = {
            "duration_seconds",
            "minimum_hard_margin",
            "certified_state_fraction",
            "certified_time_fraction",
            "degraded_state_fraction",
            "degraded_duration_seconds",
            "mean_intervention_norm",
            "maximum_intervention_norm",
            "intervention_integral",
            "policy_switch_rate_hz",
            "mean_normalized_estimation_error",
            "maximum_normalized_estimation_error",
        }
        if metric_name not in allowed:
            raise ValueError(
                "unsupported scalar metric; recovery censoring and latency components require "
                "explicit accessors"
            )
        return float(getattr(self, metric_name))

    def latency_scalar(self, component: str, statistic: str) -> float:
        """Return one explicitly named latency statistic."""
        _require_slug(component, "latency component")
        if statistic not in {
            "median_seconds",
            "p95_seconds",
            "p99_seconds",
            "worst_seconds",
            "deadline_misses",
        }:
            raise ValueError("unsupported latency statistic")
        for latency in self.latencies:
            if latency.component == component:
                return float(getattr(latency, statistic))
        raise ValueError(f"latency component {component!r} is absent")


def _time_measure(mask: np.ndarray, time: np.ndarray) -> float:
    """Measure a node mask using left-held controller intervals; the terminal node has no width."""
    return float(np.sum(np.diff(time) * mask[:-1].astype(np.float64)))


def _coverage_recoveries(
    certified: np.ndarray, time: np.ndarray, change_indices: Sequence[int]
) -> tuple[RecoveryMetrics, ...]:
    if certified.shape != time.shape:
        raise ValueError("certified and time must be same-shape node vectors")
    indices = tuple(_require_nonnegative_integer(index, "change index") for index in change_indices)
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("change_indices must be strictly increasing and unique")
    # Certification is left-held over control intervals.  The final node is an observation with
    # zero width and therefore cannot make an otherwise certified suffix fail recovery.
    interval_certified = certified[:-1]
    suffix_certified = np.logical_and.accumulate(interval_certified[::-1])[::-1]
    output: list[RecoveryMetrics] = []
    for index in indices:
        if index >= time.size - 1:
            raise ValueError(
                "every change index must precede at least one complete control interval"
            )
        candidates = np.flatnonzero(suffix_certified[index:])
        censor = float(time[-1] - time[index])
        if candidates.size:
            recovered_index = index + int(candidates[0])
            output.append(
                RecoveryMetrics(
                    change_index=index,
                    change_time_seconds=float(time[index]),
                    recovered_through_horizon=True,
                    recovery_time_seconds=float(time[recovered_index] - time[index]),
                    censor_time_seconds=censor,
                )
            )
        else:
            output.append(
                RecoveryMetrics(
                    change_index=index,
                    change_time_seconds=float(time[index]),
                    recovered_through_horizon=False,
                    recovery_time_seconds=None,
                    censor_time_seconds=censor,
                )
            )
    return tuple(output)


def derive_scientific_metrics(
    trace: ImmutableTrace,
    *,
    hard_certified_policy: np.ndarray,
    estimation_error: np.ndarray,
    estimation_scale: np.ndarray,
    change_indices: Sequence[int] = (),
    latency_deadlines_seconds: Mapping[str, float],
    interval_safety_evidence: bool,
    warm_execution_excludes_compilation: bool,
) -> ScientificTrialMetrics:
    """Derive scientific metrics from raw arrays without inferring hard certification.

    ``hard_certified_policy`` must be the exact hard certificate mask with shape ``(T, K)``; soft
    training values are intentionally not thresholded here.  ``estimation_error`` has shape
    ``(T, E)`` and is normalized componentwise by the positive, predeclared ``estimation_scale`` so
    physically incompatible quantities are never combined without an explicit scale.  The two
    protocol booleans must explicitly confirm that hard margins/contact include dynamics-substep or
    conservative swept evidence and that compilation was excluded from every latency sample.
    """
    from crazyflow.safety.da_plcbf.artifacts import ImmutableTrace

    if not isinstance(trace, ImmutableTrace):
        raise TypeError("trace must be an ImmutableTrace")
    trace.validate()
    if interval_safety_evidence is not True:
        raise ValueError(
            "interval_safety_evidence must confirm substep checks or conservative swept minima"
        )
    if warm_execution_excludes_compilation is not True:
        raise ValueError("latency samples must exclude compilation")
    certified = np.asarray(hard_certified_policy)
    if certified.dtype != np.dtype(np.bool_) or certified.ndim != 2:
        raise ValueError("hard_certified_policy must be a boolean (T, K) array")
    if certified.shape[0] != trace.steps or certified.shape[1] < 1:
        raise ValueError("hard_certified_policy shape must be (trace.steps, positive K)")
    error = np.asarray(estimation_error, dtype=np.float64)
    scale = np.asarray(estimation_scale, dtype=np.float64)
    if error.ndim != 2 or error.shape[0] != trace.steps or error.shape[1] < 1:
        raise ValueError("estimation_error must have shape (trace.steps, positive E)")
    if scale.shape != (error.shape[1],) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("estimation_scale must be finite, positive, and match estimation_error")
    if not np.all(np.isfinite(error)):
        raise ValueError("estimation_error must be finite")
    if set(latency_deadlines_seconds) != {str(name) for name in trace.latency_names}:
        raise ValueError("latency deadlines must exactly match trace latency names")

    time = np.asarray(trace.time, dtype=np.float64)
    collision = np.asarray(trace.contact, dtype=np.bool_)
    constraint_violation = np.any(np.asarray(trace.hard_barriers) < 0.0, axis=1)
    failure = np.asarray(trace.failure, dtype=np.bool_)
    executed = np.asarray(trace.executed_control, dtype=np.bool_)
    has_certificate_all = np.any(certified, axis=1)
    has_certificate = has_certificate_all[executed]
    degraded_all = np.asarray(trace.degraded, dtype=np.bool_)
    degraded = degraded_all[executed]
    intervention = np.linalg.norm(
        trace.applied_control[executed] - trace.nominal_control[executed], axis=1
    )
    # Every physical interval [t_i,t_{i+1}) uses the command stored at its left node.  This is
    # intentionally independent of whether a legacy trace also labels its final zero-width node as
    # executed; schema-v2 experiment traces use an explicit final no-control sentinel instead.
    interval_intervention = np.linalg.norm(
        trace.applied_control[:-1] - trace.nominal_control[:-1], axis=1
    )
    normalized_estimation = np.linalg.norm(error / scale[None, :], axis=1)
    selected = np.asarray(trace.selected_policy)[executed]
    valid_adjacent = (selected[1:] >= 0) & (selected[:-1] >= 0)
    switches = int(np.count_nonzero(valid_adjacent & (selected[1:] != selected[:-1])))
    duration = float(time[-1] - time[0])

    latencies: list[LatencyMetrics] = []
    for column, raw_name in enumerate(trace.latency_names):
        name = str(raw_name)
        samples = np.asarray(trace.component_latency_seconds[executed, column], dtype=np.float64)
        deadline = _finite(latency_deadlines_seconds[name], f"deadline.{name}")
        if deadline <= 0.0:
            raise ValueError(f"deadline.{name} must be positive")
        latencies.append(
            LatencyMetrics(
                component=name,
                count=int(samples.size),
                median_seconds=float(np.percentile(samples, 50.0, method="linear")),
                p95_seconds=float(np.percentile(samples, 95.0, method="linear")),
                p99_seconds=float(np.percentile(samples, 99.0, method="linear")),
                worst_seconds=float(np.max(samples)),
                deadline_seconds=deadline,
                deadline_misses=int(np.count_nonzero(samples > deadline)),
            )
        )

    metrics = ScientificTrialMetrics(
        steps=trace.steps,
        duration_seconds=duration,
        interval_safety_evidence=True,
        warm_execution_excludes_compilation=True,
        collision_steps=int(np.count_nonzero(collision)),
        constraint_violation_steps=int(np.count_nonzero(constraint_violation)),
        failure_steps=int(np.count_nonzero(failure)),
        any_collision=bool(np.any(collision)),
        any_constraint_violation=bool(np.any(constraint_violation)),
        any_failure=bool(np.any(failure)),
        minimum_hard_margin=float(np.min(trace.hard_barriers)),
        certified_state_fraction=float(np.mean(has_certificate)),
        certified_time_fraction=_time_measure(has_certificate_all, time) / duration,
        degraded_state_fraction=float(np.mean(degraded)),
        degraded_duration_seconds=_time_measure(degraded_all, time),
        mean_intervention_norm=float(np.mean(intervention)),
        maximum_intervention_norm=float(np.max(intervention)),
        intervention_integral=float(np.sum(np.diff(time) * interval_intervention)),
        policy_switches=switches,
        policy_switch_rate_hz=switches / duration,
        mean_normalized_estimation_error=float(np.mean(normalized_estimation)),
        maximum_normalized_estimation_error=float(np.max(normalized_estimation)),
        recoveries=_coverage_recoveries(
            has_certificate_all, time, tuple(index for index in change_indices if executed[index])
        ),
        latencies=tuple(latencies),
    )
    metrics.validate()
    return metrics


class TrialStatus(str, Enum):
    """Whether a scheduled run produced a complete validated trace."""

    COMPLETE = "complete"
    EXECUTION_FAILURE = "execution_failure"


@dataclass(frozen=True, slots=True)
class ScientificTrialRecord:
    """One retained scheduled outcome, including failed executions with no trace metrics."""

    method: str
    condition: str
    fold: int
    pairing_id: str
    scenario_tape_sha256: str
    status: TrialStatus
    metrics: ScientificTrialMetrics | None
    failure_code: str | None = None
    failure_message: str | None = None

    @property
    def key(self) -> tuple[str, str, int]:
        """Return the unique method/condition/fold key."""
        return (self.method, self.condition, self.fold)

    @property
    def operational_failure(self) -> bool:
        """Count execution, physical, and declared safety-controller failures.

        A completed safety-controller trial is operationally failed when any executed interval is
        explicitly degraded, even if the best-effort command happens not to violate a physical
        constraint on that finite tape.  ``nominal_only`` is the one protocol-level exception: it
        has no safety controller by definition, and its trace uses ``degraded`` solely to encode
        certificate unavailability.  Its physical failures remain observable through
        :attr:`ScientificTrialMetrics.any_failure`.
        """
        if self.status is TrialStatus.EXECUTION_FAILURE:
            return True
        if self.metrics is None:
            return True
        physical_failure = self.metrics.any_failure
        declared_controller_failure = (
            self.method != "nominal_only" and self.metrics.degraded_state_fraction > 0.0
        )
        return bool(physical_failure or declared_controller_failure)

    def validate(self) -> None:
        """Require explicit failure status rather than silently missing metrics."""
        _require_slug(self.method, "record.method")
        _require_slug(self.condition, "record.condition")
        _require_uint32(self.fold, "record.fold")
        if _SHA256.fullmatch(self.pairing_id) is None:
            raise ValueError("record.pairing_id must be a sha256 digest")
        if _SHA256.fullmatch(self.scenario_tape_sha256) is None:
            raise ValueError("record.scenario_tape_sha256 must be a sha256 digest")
        if not isinstance(self.status, TrialStatus):
            raise TypeError("record.status must be TrialStatus")
        if self.status is TrialStatus.COMPLETE:
            if (
                self.metrics is None
                or self.failure_code is not None
                or self.failure_message is not None
            ):
                raise ValueError("complete records require metrics and no failure details")
            self.metrics.validate()
        else:
            if self.metrics is not None or self.failure_code is None:
                raise ValueError(
                    "execution failures require no metrics and an explicit failure_code"
                )
            _require_slug(self.failure_code, "record.failure_code")
            if self.failure_message is not None:
                if (
                    not isinstance(self.failure_message, str)
                    or not self.failure_message
                    or len(self.failure_message) > 1000
                    or any(character in self.failure_message for character in "\r\n\0")
                ):
                    raise ValueError("record.failure_message must be a bounded single line")


@dataclass(frozen=True, slots=True)
class PairedTrialDataset:
    """A complete scheduled result matrix; no run may be dropped."""

    schedule: PairedTrialSchedule
    records: tuple[ScientificTrialRecord, ...]

    def validate(self) -> None:
        """Require every scheduled result once and byte-identical tape hashes within each pair."""
        if not isinstance(self.records, tuple):
            raise TypeError("dataset.records must be an immutable tuple")
        self.schedule.validate()
        expected = {assignment.key: assignment for assignment in self.schedule.assignments}
        actual: dict[tuple[str, str, int], ScientificTrialRecord] = {}
        for record in self.records:
            record.validate()
            if record.key in actual:
                raise ValueError(f"duplicate trial record {record.key!r}")
            actual[record.key] = record
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise ValueError(
                f"records do not match complete schedule; missing={missing}, extra={extra}"
            )
        tape_by_pair: dict[tuple[str, int], str] = {}
        for key, record in actual.items():
            assignment = expected[key]
            if record.pairing_id != assignment.pairing_id:
                raise ValueError("record pairing_id does not match schedule")
            previous = tape_by_pair.setdefault(assignment.pair_key, record.scenario_tape_sha256)
            if record.scenario_tape_sha256 != previous:
                raise ValueError("paired methods must consume the identical scenario tape hash")

    def record(self, method: str, condition: str, fold: int) -> ScientificTrialRecord:
        """Return one validated record by its unique key."""
        key = (method, condition, fold)
        for record in self.records:
            if record.key == key:
                return record
        raise KeyError(key)


@dataclass(frozen=True, slots=True)
class RateInterval:
    """Binomial rate with a Wilson score uncertainty interval."""

    events: int
    trials: int
    rate: float
    lower: float
    upper: float
    confidence_level: float
    method: str = "wilson-score"


def wilson_rate_interval(events: int, trials: int, confidence_level: float = 0.95) -> RateInterval:
    """Compute a two-sided Wilson score interval, including zero/all-event edge cases."""
    successes = _require_nonnegative_integer(events, "events")
    count = _require_nonnegative_integer(trials, "trials")
    if count < 1 or successes > count:
        raise ValueError("trials must be positive and events cannot exceed trials")
    level = _finite(confidence_level, "confidence_level")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    proportion = successes / count
    z_value = NormalDist().inv_cdf(0.5 + level / 2.0)
    denominator = 1.0 + z_value**2 / count
    center = (proportion + z_value**2 / (2.0 * count)) / denominator
    half = (
        z_value
        * math.sqrt(proportion * (1.0 - proportion) / count + z_value**2 / (4.0 * count**2))
        / denominator
    )
    return RateInterval(
        events=successes,
        trials=count,
        rate=proportion,
        lower=max(0.0, center - half),
        upper=min(1.0, center + half),
        confidence_level=level,
    )


@dataclass(frozen=True, slots=True)
class ExactBinomialInterval:
    """Clopper-Pearson interval for a binomial success probability."""

    successes: int
    trials: int
    estimate: float | None
    lower: float | None
    upper: float | None
    confidence_level: float
    method: str = "clopper-pearson-exact"


def exact_binomial_interval(
    successes: int, trials: int, confidence_level: float = 0.95
) -> ExactBinomialInterval:
    """Compute a two-sided exact interval; zero trials is explicitly unavailable."""
    wins = _require_nonnegative_integer(successes, "successes")
    count = _require_nonnegative_integer(trials, "trials")
    if wins > count:
        raise ValueError("successes cannot exceed trials")
    level = _finite(confidence_level, "confidence_level")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if count == 0:
        return ExactBinomialInterval(wins, count, None, None, None, level)
    alpha = 1.0 - level
    lower = 0.0 if wins == 0 else float(beta.ppf(alpha / 2.0, wins, count - wins + 1))
    upper = 1.0 if wins == count else float(beta.ppf(1.0 - alpha / 2.0, wins + 1, count - wins))
    return ExactBinomialInterval(wins, count, wins / count, lower, upper, level)


class MetricDirection(str, Enum):
    """Predeclared direction of improvement for a scalar estimand."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class AnalysisRole(str, Enum):
    """Whether an endpoint was predeclared for confirmation or only exploration."""

    CONFIRMATORY = "confirmatory"
    EXPLORATORY = "exploratory"


def confirmatory_bootstrap_replicates(
    familywise_comparisons: int, *, confidence_level: float = 0.95
) -> int:
    """Size a percentile bootstrap for at least 100 expected draws per adjusted tail.

    This is a Monte Carlo *resolution* requirement, not a claim that percentile bootstrap has
    exact finite-sample coverage.  Rounding upward keeps the predeclared campaign configuration
    simple; the 72-endpoint core safety family uses 290,000 replicates rather than the unrounded
    288,000.
    """
    comparisons = _require_nonnegative_integer(familywise_comparisons, "familywise_comparisons")
    if comparisons < 1:
        raise ValueError("familywise_comparisons must be positive")
    level = _finite(confidence_level, "confidence_level")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    required = math.ceil(MINIMUM_EXPECTED_BOOTSTRAP_TAIL_DRAWS * 2.0 * comparisons / (1.0 - level))
    target = max(DEFAULT_BOOTSTRAP_REPLICATES, required)
    return int(math.ceil(target / BOOTSTRAP_REPLICATE_ROUNDING) * BOOTSTRAP_REPLICATE_ROUNDING)


@dataclass(frozen=True, slots=True)
class PairedInferenceConfig:
    """Predeclared paired bootstrap, multiplicity, and practical-effect protocol."""

    analysis_role: AnalysisRole = AnalysisRole.CONFIRMATORY
    confidence_level: float = 0.95
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES
    familywise_comparisons: int = 1
    minimum_oriented_effect: float = 0.0
    tie_tolerance: float = 0.0

    @property
    def adjusted_confidence_level(self) -> float:
        """Return the Bonferroni-adjusted two-sided confidence level."""
        return 1.0 - (1.0 - self.confidence_level) / self.familywise_comparisons

    @property
    def expected_bootstrap_draws_per_tail(self) -> float:
        """Return the expected replicate count beyond either adjusted CI endpoint."""
        return self.bootstrap_replicates * (1.0 - self.adjusted_confidence_level) / 2.0

    def validate(self) -> None:
        """Validate a final-evidence-capable paired inference protocol."""
        if not isinstance(self.analysis_role, AnalysisRole):
            raise TypeError("inference.analysis_role must be an AnalysisRole")
        level = _finite(self.confidence_level, "inference.confidence_level")
        if not 0.0 < level < 1.0:
            raise ValueError("inference.confidence_level must lie strictly between zero and one")
        replicates = _require_nonnegative_integer(
            self.bootstrap_replicates, "inference.bootstrap_replicates"
        )
        if replicates < 1_000:
            raise ValueError("paired bootstrap requires at least 1000 replicates")
        comparisons = _require_nonnegative_integer(
            self.familywise_comparisons, "inference.familywise_comparisons"
        )
        if comparisons < 1:
            raise ValueError("inference.familywise_comparisons must be positive")
        _finite_nonnegative(self.minimum_oriented_effect, "inference.minimum_oriented_effect")
        _finite_nonnegative(self.tie_tolerance, "inference.tie_tolerance")


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Conservative metric-level result for candidate relative to baseline."""

    condition: str
    metric_name: str
    candidate_method: str
    baseline_method: str
    analysis_role: AnalysisRole
    direction: MetricDirection
    pair_count: int
    missing_metric_pairs: int
    candidate_mean: float | None
    baseline_mean: float | None
    mean_oriented_improvement: float | None
    bootstrap_interval: tuple[float, float] | None
    candidate_wins: int
    ties: int
    candidate_losses: int
    win_probability_interval: ExactBinomialInterval
    exact_one_sided_sign_pvalue: float | None
    adjusted_confidence_level: float
    expected_bootstrap_draws_per_tail: float
    bootstrap_distribution_degenerate: bool
    bootstrap_resolution_sufficient: bool
    minimum_oriented_effect: float
    tie_tolerance: float
    final_claim_eligible: bool
    superiority_supported: bool
    conclusion: str
    bootstrap_rng: RNGProvenance
    finite_horizon_claim_boundary: str = FINITE_HORIZON_CLAIM_BOUNDARY


def _bootstrap_mean_interval(
    paired_improvement: np.ndarray,
    *,
    provenance: RNGProvenance,
    replicates: int,
    confidence_level: float,
) -> tuple[float, float]:
    if np.all(paired_improvement == paired_improvement[0]):
        value = float(paired_improvement[0])
        return (value, value)
    rng = np.random.Generator(np.random.PCG64(provenance.derived_seed))
    count = paired_improvement.size
    means = np.empty(replicates, dtype=np.float64)
    batch_size = min(replicates, max(1, 2_000_000 // count))
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, count, size=(stop - start, count), endpoint=False)
        means[start:stop] = np.mean(paired_improvement[indices], axis=1)
    alpha = 1.0 - confidence_level
    quantiles = np.quantile(means, (alpha / 2.0, 1.0 - alpha / 2.0), method="linear")
    return (float(quantiles[0]), float(quantiles[1]))


def compare_paired_metric(
    dataset: PairedTrialDataset,
    *,
    condition: str,
    candidate_method: str,
    baseline_method: str,
    metric_name: str,
    direction: MetricDirection,
    inference: PairedInferenceConfig | None = None,
    metric_getter: Callable[[ScientificTrialMetrics], float] | None = None,
) -> PairedComparison:
    """Compare a scalar metric with paired bootstrap and an exact paired sign analysis.

    A superiority statement is emitted only when the schedule was predeclared for final claims,
    has at least 100 complete matched pairs, the adjusted paired-bootstrap interval exceeds the
    predeclared effect threshold, and the exact adjusted win-probability interval exceeds 0.5.
    """
    if not isinstance(dataset, PairedTrialDataset):
        raise TypeError("dataset must be a PairedTrialDataset")
    dataset.validate()
    _require_slug(condition, "condition")
    _require_slug(candidate_method, "candidate_method")
    _require_slug(baseline_method, "baseline_method")
    _require_slug(metric_name, "metric_name")
    if candidate_method == baseline_method:
        raise ValueError("candidate and baseline methods must differ")
    if condition not in dataset.schedule.conditions:
        raise ValueError("condition is absent from schedule")
    if (
        candidate_method not in dataset.schedule.methods
        or baseline_method not in dataset.schedule.methods
    ):
        raise ValueError("candidate and baseline methods must be scheduled")
    if not isinstance(direction, MetricDirection):
        raise TypeError("direction must be MetricDirection")
    protocol = PairedInferenceConfig() if inference is None else inference
    if not isinstance(protocol, PairedInferenceConfig):
        raise TypeError("inference must be PairedInferenceConfig")
    protocol.validate()
    getter = (
        (lambda metrics: metrics.scalar(metric_name)) if metric_getter is None else metric_getter
    )

    candidate_values: list[float] = []
    baseline_values: list[float] = []
    missing = 0
    for fold in range(
        dataset.schedule.fold_start,
        dataset.schedule.fold_start + dataset.schedule.trials_per_condition,
    ):
        candidate = dataset.record(candidate_method, condition, fold)
        baseline = dataset.record(baseline_method, condition, fold)
        if metric_name == "operational_failure":
            if metric_getter is not None:
                raise ValueError("operational_failure uses its fixed conservative record accessor")
            if direction is not MetricDirection.LOWER_IS_BETTER:
                raise ValueError("lower_is_better is required for operational_failure")
            candidate_values.append(float(candidate.operational_failure))
            baseline_values.append(float(baseline.operational_failure))
            continue
        if candidate.metrics is None or baseline.metrics is None:
            missing += 1
            continue
        candidate_value = _finite(getter(candidate.metrics), f"candidate {metric_name}")
        baseline_value = _finite(getter(baseline.metrics), f"baseline {metric_name}")
        candidate_values.append(candidate_value)
        baseline_values.append(baseline_value)

    provenance = rng_provenance(
        dataset.schedule.root_seed,
        "paired_bootstrap",
        condition,
        metric_name,
        candidate_method,
        baseline_method,
        direction.value,
        protocol.bootstrap_replicates,
        protocol.familywise_comparisons,
    )
    pair_count = dataset.schedule.trials_per_condition
    expected_tail_draws = protocol.expected_bootstrap_draws_per_tail
    if missing:
        exact = exact_binomial_interval(0, 0, protocol.adjusted_confidence_level)
        return PairedComparison(
            condition=condition,
            metric_name=metric_name,
            candidate_method=candidate_method,
            baseline_method=baseline_method,
            analysis_role=protocol.analysis_role,
            direction=direction,
            pair_count=pair_count,
            missing_metric_pairs=missing,
            candidate_mean=None,
            baseline_mean=None,
            mean_oriented_improvement=None,
            bootstrap_interval=None,
            candidate_wins=0,
            ties=0,
            candidate_losses=0,
            win_probability_interval=exact,
            exact_one_sided_sign_pvalue=None,
            adjusted_confidence_level=protocol.adjusted_confidence_level,
            expected_bootstrap_draws_per_tail=expected_tail_draws,
            bootstrap_distribution_degenerate=False,
            bootstrap_resolution_sufficient=(
                expected_tail_draws >= MINIMUM_EXPECTED_BOOTSTRAP_TAIL_DRAWS
            ),
            minimum_oriented_effect=protocol.minimum_oriented_effect,
            tie_tolerance=protocol.tie_tolerance,
            final_claim_eligible=False,
            superiority_supported=False,
            conclusion=(
                "exploratory only: one or more scheduled execution failures lack this metric; "
                "failures were retained rather than dropped"
                if protocol.analysis_role is AnalysisRole.EXPLORATORY
                else "not supported: one or more scheduled execution failures lack this metric; "
                "failures were retained rather than dropped"
            ),
            bootstrap_rng=provenance,
        )

    candidate_array = np.asarray(candidate_values, dtype=np.float64)
    baseline_array = np.asarray(baseline_values, dtype=np.float64)
    oriented = (
        candidate_array - baseline_array
        if direction is MetricDirection.HIGHER_IS_BETTER
        else baseline_array - candidate_array
    )
    degenerate_bootstrap = bool(np.all(oriented == oriented[0]))
    resolution_sufficient = bool(
        degenerate_bootstrap or expected_tail_draws >= MINIMUM_EXPECTED_BOOTSTRAP_TAIL_DRAWS
    )
    interval = _bootstrap_mean_interval(
        oriented,
        provenance=provenance,
        replicates=protocol.bootstrap_replicates,
        confidence_level=protocol.adjusted_confidence_level,
    )
    wins = int(np.count_nonzero(oriented > protocol.tie_tolerance))
    losses = int(np.count_nonzero(oriented < -protocol.tie_tolerance))
    ties = int(oriented.size - wins - losses)
    discordant = wins + losses
    exact = exact_binomial_interval(wins, discordant, protocol.adjusted_confidence_level)
    sign_pvalue = (
        None
        if discordant == 0
        else float(binomtest(wins, discordant, p=0.5, alternative="greater").pvalue)
    )
    eligible = bool(
        protocol.analysis_role is AnalysisRole.CONFIRMATORY
        and dataset.schedule.final_claim_eligible
        and pair_count >= MINIMUM_FINAL_PAIRED_TRIALS
        and resolution_sufficient
    )
    threshold_pass = interval[0] > protocol.minimum_oriented_effect
    sign_pass = bool(exact.lower is not None and exact.lower > 0.5)
    alpha_adjusted = 1.0 - protocol.adjusted_confidence_level
    pvalue_pass = bool(sign_pvalue is not None and sign_pvalue < alpha_adjusted)
    supported = eligible and threshold_pass and sign_pass and pvalue_pass
    if protocol.analysis_role is AnalysisRole.EXPLORATORY:
        conclusion = (
            "exploratory only: this endpoint was not predeclared for confirmatory superiority"
        )
    elif not resolution_sufficient:
        conclusion = (
            "not eligible for a final claim: adjusted percentile-bootstrap tails have fewer than "
            f"{MINIMUM_EXPECTED_BOOTSTRAP_TAIL_DRAWS:.0f} expected draws"
        )
    elif supported:
        conclusion = (
            "supported for this predeclared metric and finite-horizon condition; no broader safety "
            "claim is implied"
        )
    elif not eligible:
        conclusion = (
            f"not eligible for a final claim: schedule must be predeclared with at least "
            f"{MINIMUM_FINAL_PAIRED_TRIALS} matched trials"
        )
    else:
        conclusion = "not supported: the conservative paired interval/sign criteria were not met"
    return PairedComparison(
        condition=condition,
        metric_name=metric_name,
        candidate_method=candidate_method,
        baseline_method=baseline_method,
        analysis_role=protocol.analysis_role,
        direction=direction,
        pair_count=pair_count,
        missing_metric_pairs=0,
        candidate_mean=float(np.mean(candidate_array)),
        baseline_mean=float(np.mean(baseline_array)),
        mean_oriented_improvement=float(np.mean(oriented)),
        bootstrap_interval=interval,
        candidate_wins=wins,
        ties=ties,
        candidate_losses=losses,
        win_probability_interval=exact,
        exact_one_sided_sign_pvalue=sign_pvalue,
        adjusted_confidence_level=protocol.adjusted_confidence_level,
        expected_bootstrap_draws_per_tail=expected_tail_draws,
        bootstrap_distribution_degenerate=degenerate_bootstrap,
        bootstrap_resolution_sufficient=resolution_sufficient,
        minimum_oriented_effect=protocol.minimum_oriented_effect,
        tie_tolerance=protocol.tie_tolerance,
        final_claim_eligible=eligible,
        superiority_supported=supported,
        conclusion=conclusion,
        bootstrap_rng=provenance,
    )


def operational_failure_rate(
    dataset: PairedTrialDataset, *, method: str, condition: str, confidence_level: float = 0.95
) -> RateInterval:
    """Report a Wilson interval counting execution failures and trace failures alike."""
    return scientific_event_rate(
        dataset,
        method=method,
        condition=condition,
        event="operational_failure",
        confidence_level=confidence_level,
    )


def scientific_event_rate(
    dataset: PairedTrialDataset,
    *,
    method: str,
    condition: str,
    event: str,
    confidence_level: float = 0.95,
) -> RateInterval:
    """Report Wilson uncertainty for a trial-level collision, violation, or failure event.

    Execution failures are always observable as ``operational_failure``.  For methods that declare
    a safety controller, any explicit degraded interval is also an operational failure even when
    best-effort execution remains physically lucky on that tape.  ``nominal_only`` has no safety
    controller, so its certificate-unavailable marker is excluded while its physical failures are
    still counted.  Execution failures do not reveal whether a physical collision or constraint
    violation occurred, so those event-specific rates are refused rather than estimated after
    dropping failed executions.
    """
    dataset.validate()
    if method not in dataset.schedule.methods or condition not in dataset.schedule.conditions:
        raise ValueError("method and condition must be present in the schedule")
    accessors: dict[str, Callable[[ScientificTrialMetrics], bool]] = {
        "collision": lambda metrics: metrics.any_collision,
        "constraint_violation": lambda metrics: metrics.any_constraint_violation,
        "trace_failure": lambda metrics: metrics.any_failure,
    }
    if event != "operational_failure" and event not in accessors:
        raise ValueError(
            "event must be collision, constraint_violation, trace_failure, or operational_failure"
        )
    failures = 0
    for fold in range(
        dataset.schedule.fold_start,
        dataset.schedule.fold_start + dataset.schedule.trials_per_condition,
    ):
        record = dataset.record(method, condition, fold)
        if event == "operational_failure":
            failures += int(record.operational_failure)
        else:
            if record.metrics is None:
                raise ValueError(
                    f"{event} rate is unavailable because a retained execution failure has no "
                    "physical trace"
                )
            failures += int(accessors[event](record.metrics))
    return wilson_rate_interval(failures, dataset.schedule.trials_per_condition, confidence_level)


class FalsificationAxis(str, Enum):
    """Required declared-bound axes for Phase-7 boundary search."""

    INITIAL_STATE = "initial_state"
    OBSTACLE_TIMING = "obstacle_timing"
    WIND = "wind"
    MASS = "mass"
    ROTOR_EFFICIENCY = "rotor_efficiency"
    ESTIMATOR_ERROR = "estimator_error"
    ACTUATOR_SATURATION = "actuator_saturation"


@dataclass(frozen=True, slots=True)
class BoundaryVariable:
    """A finite vector variable with declared admissible bounds and nominal value."""

    axis: FalsificationAxis
    name: str
    component_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    nominal: tuple[float, ...]

    def validate(self) -> None:
        """Validate dimensions, finite ordered bounds, and nominal containment."""
        if not isinstance(self.axis, FalsificationAxis):
            raise TypeError("boundary axis must be FalsificationAxis")
        _require_slug(self.name, "boundary variable name")
        if not all(
            isinstance(value, tuple)
            for value in (self.component_names, self.lower, self.upper, self.nominal)
        ):
            raise TypeError("boundary names, bounds, and nominal values must be tuples")
        if not self.component_names or len(set(self.component_names)) != len(self.component_names):
            raise ValueError("boundary component names must be unique and nonempty")
        for component in self.component_names:
            _require_slug(component, "boundary component name")
        dimension = len(self.component_names)
        if not (len(self.lower) == len(self.upper) == len(self.nominal) == dimension):
            raise ValueError("boundary vectors must match component_names")
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        nominal = np.asarray(self.nominal, dtype=np.float64)
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("boundary bounds must be finite")
        if not np.all(np.isfinite(nominal)):
            raise ValueError("boundary nominal values must be finite")
        if np.any(upper <= lower):
            raise ValueError("every boundary upper value must exceed its lower value")
        if np.any(nominal < lower) or np.any(nominal > upper):
            raise ValueError("boundary nominal values must lie inside declared bounds")


@dataclass(frozen=True, slots=True, eq=False)
class BoundaryCandidateSet:
    """Immutable randomized candidates with auditable focused-boundary provenance."""

    variables: tuple[BoundaryVariable, ...]
    flat_component_names: tuple[str, ...]
    values: np.ndarray
    focus_component: np.ndarray
    focus_side: np.ndarray
    boundary_band_fraction: float
    background_nominal_fraction: float
    rng: RNGProvenance

    def __post_init__(self) -> None:
        """Defensively copy and freeze numeric candidate arrays."""
        for name in ("values", "focus_component", "focus_side"):
            value = np.ascontiguousarray(getattr(self, name)).copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        self.validate()

    @property
    def count(self) -> int:
        """Number of generated candidates."""
        return int(self.values.shape[0])

    def validate(self) -> None:
        """Validate coverage, bounds, and each row's declared focused boundary."""
        if not isinstance(self.variables, tuple) or not isinstance(
            self.flat_component_names, tuple
        ):
            raise TypeError("candidate variables and component names must be immutable tuples")
        if not self.variables:
            raise ValueError("boundary candidate set requires variables")
        for variable in self.variables:
            variable.validate()
        expected_names = tuple(
            f"{variable.name}.{component}"
            for variable in self.variables
            for component in variable.component_names
        )
        if self.flat_component_names != expected_names:
            raise ValueError("flat component names do not match variables")
        dimension = len(expected_names)
        if self.values.dtype != np.dtype(np.float64) or self.values.ndim != 2:
            raise ValueError("candidate values must be a float64 matrix")
        if self.values.shape[1] != dimension or self.values.shape[0] < 2 * dimension:
            raise ValueError("candidate set must contain at least two rows per scalar component")
        if self.focus_component.dtype != np.dtype(np.int64) or self.focus_component.shape != (
            self.count,
        ):
            raise ValueError("focus_component must be int64[count]")
        if self.focus_side.dtype != np.dtype(np.int8) or self.focus_side.shape != (self.count,):
            raise ValueError("focus_side must be int8[count]")
        if np.any(self.focus_component < 0) or np.any(self.focus_component >= dimension):
            raise ValueError("focus component lies outside candidate dimension")
        if np.any((self.focus_side != -1) & (self.focus_side != 1)):
            raise ValueError("focus_side must contain only -1 and +1")
        band = _finite(self.boundary_band_fraction, "boundary_band_fraction")
        if not 0.0 < band <= 0.5:
            raise ValueError("boundary_band_fraction must lie in (0, 0.5]")
        background = _finite(self.background_nominal_fraction, "background_nominal_fraction")
        if not 0.0 <= background <= 1.0:
            raise ValueError("background_nominal_fraction must lie in [0, 1]")
        lower = np.concatenate([np.asarray(variable.lower) for variable in self.variables])
        upper = np.concatenate([np.asarray(variable.upper) for variable in self.variables])
        if not np.all(np.isfinite(self.values)):
            raise ValueError("candidate values must be finite")
        if np.any(self.values < lower) or np.any(self.values > upper):
            raise ValueError("candidate values exceed declared bounds")
        span = upper - lower
        rows = np.arange(self.count)
        focus_values = self.values[rows, self.focus_component]
        focus_lower = lower[self.focus_component]
        focus_upper = upper[self.focus_component]
        distance = np.where(
            self.focus_side < 0, focus_values - focus_lower, focus_upper - focus_values
        )
        if np.any(distance < 0.0) or np.any(distance > band * span[self.focus_component] + 1e-15):
            raise ValueError("focused candidate value lies outside its declared boundary band")
        covered = {
            (int(component), int(side))
            for component, side in zip(self.focus_component, self.focus_side, strict=True)
        }
        expected = {(component, side) for component in range(dimension) for side in (-1, 1)}
        if not expected.issubset(covered):
            raise ValueError("candidate set must focus both sides of every scalar bound")
        self.rng.validate()


def generate_boundary_candidates(
    variables: Sequence[BoundaryVariable],
    *,
    count: int,
    root_seed: int,
    search_name: str,
    boundary_band_fraction: float = 0.05,
    background_nominal_fraction: float = 0.25,
    require_all_phase7_axes: bool = True,
) -> BoundaryCandidateSet:
    """Generate deterministic randomized candidates near every declared lower/upper bound.

    Each row has one audited focus component sampled within ``boundary_band_fraction`` of a bound.
    Other components are randomized in a declared neighborhood of their nominal values.  The
    randomized ordering covers both sides of every component before targets repeat.
    """
    variable_tuple = tuple(variables)
    if not variable_tuple:
        raise ValueError("variables must be nonempty")
    for variable in variable_tuple:
        variable.validate()
    if len({variable.name for variable in variable_tuple}) != len(variable_tuple):
        raise ValueError("boundary variable names must be unique")
    _require_slug(search_name, "search_name")
    axes = {variable.axis for variable in variable_tuple}
    if require_all_phase7_axes and axes != set(FalsificationAxis):
        missing = sorted(axis.value for axis in set(FalsificationAxis) - axes)
        extra = sorted(axis.value for axis in axes - set(FalsificationAxis))
        raise ValueError(
            f"falsification variables do not cover Phase-7 axes; missing={missing}, extra={extra}"
        )
    lower = np.concatenate(
        [np.asarray(variable.lower, dtype=np.float64) for variable in variable_tuple]
    )
    upper = np.concatenate(
        [np.asarray(variable.upper, dtype=np.float64) for variable in variable_tuple]
    )
    nominal = np.concatenate(
        [np.asarray(variable.nominal, dtype=np.float64) for variable in variable_tuple]
    )
    dimension = lower.size
    candidate_count = _require_nonnegative_integer(count, "count")
    if candidate_count < 2 * dimension:
        raise ValueError(
            "count must be at least twice the flattened dimension for boundary coverage"
        )
    band = _finite(boundary_band_fraction, "boundary_band_fraction")
    if not 0.0 < band <= 0.5:
        raise ValueError("boundary_band_fraction must lie in (0, 0.5]")
    background = _finite(background_nominal_fraction, "background_nominal_fraction")
    if not 0.0 <= background <= 1.0:
        raise ValueError("background_nominal_fraction must lie in [0, 1]")
    variable_digest = hashlib.sha256(
        repr(
            tuple(
                (
                    variable.axis.value,
                    variable.name,
                    variable.component_names,
                    variable.lower,
                    variable.upper,
                    variable.nominal,
                )
                for variable in variable_tuple
            )
        ).encode()
    ).hexdigest()
    provenance = rng_provenance(
        root_seed,
        "boundary_falsification",
        search_name,
        candidate_count,
        f"{band:.17g}",
        f"{background:.17g}",
        variable_digest,
    )
    rng = np.random.Generator(np.random.PCG64(provenance.derived_seed))
    targets = np.asarray(
        [(component, side) for component in range(dimension) for side in (-1, 1)], dtype=np.int64
    )
    target_rows: list[np.ndarray] = []
    while sum(chunk.shape[0] for chunk in target_rows) < candidate_count:
        target_rows.append(targets[rng.permutation(targets.shape[0])])
    selected_targets = np.concatenate(target_rows, axis=0)[:candidate_count]

    span = upper - lower
    background_radius = background * span
    values = (
        nominal[None, :]
        + rng.uniform(-1.0, 1.0, size=(candidate_count, dimension)) * background_radius
    )
    values = np.clip(values, lower, upper)
    focus_component = selected_targets[:, 0].astype(np.int64)
    focus_side = selected_targets[:, 1].astype(np.int8)
    offset = rng.uniform(0.0, band, size=candidate_count) * span[focus_component]
    rows = np.arange(candidate_count)
    values[rows, focus_component] = np.where(
        focus_side < 0, lower[focus_component] + offset, upper[focus_component] - offset
    )
    names = tuple(
        f"{variable.name}.{component}"
        for variable in variable_tuple
        for component in variable.component_names
    )
    return BoundaryCandidateSet(
        variables=variable_tuple,
        flat_component_names=names,
        values=values.astype(np.float64),
        focus_component=focus_component,
        focus_side=focus_side,
        boundary_band_fraction=band,
        background_nominal_fraction=background,
        rng=provenance,
    )
