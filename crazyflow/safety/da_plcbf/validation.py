"""Fail-closed hard admission validation for DA-PLCBF candidate snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crazyflow.safety.da_plcbf.snapshots import PolicySnapshot


GATE_NAMES = (
    "active_integrity",
    "candidate_integrity",
    "parameter_schema_compatibility",
    "base_active_freshness",
    "model_version_freshness",
    "finite_values",
    "current_margin",
    "local_non_regression",
    "coverage",
    "redundancy",
    "diversity",
    "structural_core_preservation",
    "feasibility",
    "runtime_budget",
)
"""Stable ordered names emitted by every hard validation report."""


@dataclass(frozen=True, slots=True)
class HardValidationThresholds:
    """Explicit thresholds for all quantitative hard admission gates."""

    minimum_current_margin: float = 0.0
    safe_policy_margin: float = 0.0
    local_non_regression_tolerance: float = 0.0
    minimum_coverage: float = 1.0
    minimum_redundancy: int = 1
    minimum_diversity: float = 0.0
    minimum_feasible_fraction: float = 1.0
    maximum_runtime_seconds: float = 0.01

    def validate(self) -> None:
        """Reject nonsensical or nonfinite gate thresholds."""
        float_values = (
            self.minimum_current_margin,
            self.safe_policy_margin,
            self.local_non_regression_tolerance,
            self.minimum_coverage,
            self.minimum_diversity,
            self.minimum_feasible_fraction,
            self.maximum_runtime_seconds,
        )
        if not all(math.isfinite(value) for value in float_values):
            raise ValueError("all validation thresholds must be finite")
        if self.local_non_regression_tolerance < 0:
            raise ValueError("local_non_regression_tolerance must be nonnegative")
        if not 0.0 <= self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be in [0, 1]")
        if (
            isinstance(self.minimum_redundancy, bool)
            or not isinstance(self.minimum_redundancy, int)
            or self.minimum_redundancy < 1
        ):
            raise ValueError("minimum_redundancy must be a positive integer")
        if self.minimum_diversity < 0:
            raise ValueError("minimum_diversity must be nonnegative")
        if not 0.0 <= self.minimum_feasible_fraction <= 1.0:
            raise ValueError("minimum_feasible_fraction must be in [0, 1]")
        if self.maximum_runtime_seconds <= 0:
            raise ValueError("maximum_runtime_seconds must be positive")


@dataclass(frozen=True, slots=True)
class HardValidationEvidence:
    """Dense held-out evidence used to derive, rather than assert, admission metrics.

    ``candidate_local_policy_margins`` and ``active_local_policy_margins`` have shape ``[K, B]``;
    the non-regression check compares their best hard margin independently for every scenario.
    Descriptors have shape ``[K, D]`` and are normalized by the positive ``[D]`` scales before the
    minimum pairwise distance is computed.  Feasibility margins and runtime samples may have any
    nonempty shape and are conservatively reduced by feasible fraction and worst case respectively.
    """

    current_policy_margins: Any
    candidate_local_policy_margins: Any
    active_local_policy_margins: Any
    candidate_descriptors: Any
    descriptor_scales: Any
    feasibility_margins: Any
    runtime_seconds: Any
    validation_set_digest: str = ""


@dataclass(frozen=True, slots=True)
class GateResult:
    """One named, auditable hard-gate decision."""

    name: str
    passed: bool
    observed: str
    requirement: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Content-addressed validation report bound to exact active/candidate identities."""

    active_digest: str
    active_version: int
    candidate_digest: str
    candidate_version: int
    model_version: int
    validation_set_digest: str
    gates: tuple[GateResult, ...]
    candidate_local_best: tuple[float, ...]
    active_local_best: tuple[float, ...]
    local_non_regression_passes: tuple[bool, ...]
    digest: str

    @property
    def passed(self) -> bool:
        """Return true only when every required named gate is present once and passes."""
        names = tuple(gate.name for gate in self.gates)
        return names == GATE_NAMES and all(gate.passed for gate in self.gates)

    @property
    def failed_gate_names(self) -> tuple[str, ...]:
        """Return failed gate names in deterministic evaluation order."""
        return tuple(gate.name for gate in self.gates if not gate.passed)

    def as_log_records(self) -> tuple[dict[str, str | bool], ...]:
        """Return one serialization-ready log record per hard gate."""
        return tuple(
            {
                "name": gate.name,
                "passed": gate.passed,
                "observed": gate.observed,
                "requirement": gate.requirement,
                "detail": gate.detail,
            }
            for gate in self.gates
        )

    def verify_integrity(self) -> bool:
        """Verify that report bindings, metrics, and every gate match its SHA-256 digest."""
        if tuple(gate.name for gate in self.gates) != GATE_NAMES:
            return False
        return self.digest == _report_digest(
            active_digest=self.active_digest,
            active_version=self.active_version,
            candidate_digest=self.candidate_digest,
            candidate_version=self.candidate_version,
            model_version=self.model_version,
            validation_set_digest=self.validation_set_digest,
            gates=self.gates,
            candidate_local_best=self.candidate_local_best,
            active_local_best=self.active_local_best,
            local_non_regression_passes=self.local_non_regression_passes,
        )


def hard_validate_candidate(
    active: PolicySnapshot,
    candidate: PolicySnapshot,
    evidence: HardValidationEvidence,
    thresholds: HardValidationThresholds,
    *,
    current_model_version: int,
) -> ValidationReport:
    """Evaluate every admission gate and bind the result to the exact snapshot pair.

    Invalid evidence never skips a gate.  It causes the finite gate and each affected quantitative
    gate to fail with an explicit observation, preserving a stable log schema for rejected runs.
    """
    thresholds.validate()
    if isinstance(current_model_version, bool) or not isinstance(current_model_version, int):
        raise TypeError("current_model_version must be an integer")
    if current_model_version < 0:
        raise ValueError("current_model_version must be nonnegative")
    if not isinstance(evidence.validation_set_digest, str):
        raise TypeError("validation_set_digest must be a string")

    arrays, arrays_valid = _collect_arrays(evidence)
    finite_evidence = arrays_valid and all(
        bool(np.all(np.isfinite(array))) for array in arrays.values()
    )
    active_integrity = active.kind == "active" and active.verify_integrity()
    candidate_integrity = candidate.kind == "candidate" and candidate.verify_integrity()
    parameter_schema_compatible = (
        active.params_schema_digest == candidate.params_schema_digest
        if active_integrity and candidate_integrity
        else False
    )
    snapshots_finite = active.all_finite() and candidate.all_finite()

    base_fresh = (
        candidate.base_active_version == active.version
        and candidate.base_active_digest == active.digest
        and candidate.version > active.version
    )
    model_fresh = candidate.model_version == current_model_version

    candidate_local = arrays.get("candidate_local_policy_margins")
    expected_policy_count = (
        candidate_local.shape[0]
        if candidate_local is not None and candidate_local.ndim == 2
        else None
    )
    current_margin, current_ok = _current_maximum(
        arrays.get("current_policy_margins"), expected_policy_count
    )
    current_pass = current_ok and current_margin >= thresholds.minimum_current_margin

    (candidate_local_best, active_local_best, local_passes, local_shape_ok) = _local_non_regression(
        arrays.get("candidate_local_policy_margins"),
        arrays.get("active_local_policy_margins"),
        thresholds.local_non_regression_tolerance,
    )
    local_pass = local_shape_ok and bool(np.all(local_passes))

    coverage, redundancy, local_metrics_ok = _coverage_and_redundancy(
        candidate_local, thresholds.safe_policy_margin
    )
    coverage_pass = local_metrics_ok and coverage >= thresholds.minimum_coverage
    redundancy_pass = local_metrics_ok and redundancy >= thresholds.minimum_redundancy

    diversity, diversity_ok = _minimum_descriptor_distance(
        arrays.get("candidate_descriptors"), arrays.get("descriptor_scales"), expected_policy_count
    )
    diversity_pass = diversity_ok and diversity >= thresholds.minimum_diversity

    core_pass = (
        active.structural_core_digest == candidate.structural_core_digest
        if active_integrity and candidate_integrity
        else False
    )

    feasible_fraction, feasibility_ok = _feasible_fraction(arrays.get("feasibility_margins"))
    feasibility_pass = feasibility_ok and feasible_fraction >= thresholds.minimum_feasible_fraction

    worst_runtime, runtime_ok = _worst_nonnegative(arrays.get("runtime_seconds"))
    runtime_pass = runtime_ok and worst_runtime <= thresholds.maximum_runtime_seconds

    all_finite = active_integrity and candidate_integrity and snapshots_finite and finite_evidence
    failed_scenarios = tuple(int(index) for index in np.flatnonzero(~local_passes))
    gates = (
        GateResult(
            "active_integrity",
            active_integrity,
            _bool_text(active_integrity),
            "active kind and SHA-256 integrity valid",
        ),
        GateResult(
            "candidate_integrity",
            candidate_integrity,
            _bool_text(candidate_integrity),
            "candidate kind and SHA-256 integrity valid",
        ),
        GateResult(
            "parameter_schema_compatibility",
            parameter_schema_compatible,
            candidate.params_schema_digest,
            active.params_schema_digest,
        ),
        GateResult(
            "base_active_freshness",
            base_fresh,
            (
                f"base={candidate.base_active_version}:{candidate.base_active_digest}; "
                f"candidate_version={candidate.version}"
            ),
            f"base={active.version}:{active.digest}; candidate_version>{active.version}",
        ),
        GateResult(
            "model_version_freshness",
            model_fresh,
            str(candidate.model_version),
            str(current_model_version),
        ),
        GateResult(
            "finite_values",
            all_finite,
            _bool_text(all_finite),
            "all snapshot and evidence values finite",
        ),
        GateResult(
            "current_margin",
            current_pass,
            _number_text(current_margin, current_ok),
            f">={_number_text(thresholds.minimum_current_margin, True)}",
        ),
        GateResult(
            "local_non_regression",
            local_pass,
            f"{int(np.count_nonzero(local_passes))}/{local_passes.size} scenarios",
            (
                "candidate_best>=active_best-"
                f"{_number_text(thresholds.local_non_regression_tolerance, True)} per scenario"
            ),
            f"failed_scenarios={failed_scenarios}",
        ),
        GateResult(
            "coverage",
            coverage_pass,
            _number_text(coverage, local_metrics_ok),
            f">={_number_text(thresholds.minimum_coverage, True)}",
        ),
        GateResult(
            "redundancy",
            redundancy_pass,
            str(redundancy) if local_metrics_ok else "invalid",
            f">={thresholds.minimum_redundancy} safe policies in every scenario",
        ),
        GateResult(
            "diversity",
            diversity_pass,
            _number_text(diversity, diversity_ok),
            f">={_number_text(thresholds.minimum_diversity, True)} normalized distance",
        ),
        GateResult(
            "structural_core_preservation",
            core_pass,
            candidate.structural_core_digest,
            active.structural_core_digest,
        ),
        GateResult(
            "feasibility",
            feasibility_pass,
            _number_text(feasible_fraction, feasibility_ok),
            f">={_number_text(thresholds.minimum_feasible_fraction, True)}",
        ),
        GateResult(
            "runtime_budget",
            runtime_pass,
            _number_text(worst_runtime, runtime_ok),
            f"<={_number_text(thresholds.maximum_runtime_seconds, True)} seconds (worst case)",
        ),
    )

    candidate_best_tuple = tuple(float(value) for value in candidate_local_best)
    active_best_tuple = tuple(float(value) for value in active_local_best)
    local_pass_tuple = tuple(bool(value) for value in local_passes)
    digest = _report_digest(
        active_digest=active.digest,
        active_version=active.version,
        candidate_digest=candidate.digest,
        candidate_version=candidate.version,
        model_version=current_model_version,
        validation_set_digest=evidence.validation_set_digest,
        gates=gates,
        candidate_local_best=candidate_best_tuple,
        active_local_best=active_best_tuple,
        local_non_regression_passes=local_pass_tuple,
    )
    return ValidationReport(
        active_digest=active.digest,
        active_version=active.version,
        candidate_digest=candidate.digest,
        candidate_version=candidate.version,
        model_version=current_model_version,
        validation_set_digest=evidence.validation_set_digest,
        gates=gates,
        candidate_local_best=candidate_best_tuple,
        active_local_best=active_best_tuple,
        local_non_regression_passes=local_pass_tuple,
        digest=digest,
    )


def _collect_arrays(evidence: HardValidationEvidence) -> tuple[dict[str, np.ndarray], bool]:
    names = (
        "current_policy_margins",
        "candidate_local_policy_margins",
        "active_local_policy_margins",
        "candidate_descriptors",
        "descriptor_scales",
        "feasibility_margins",
        "runtime_seconds",
    )
    arrays: dict[str, np.ndarray] = {}
    valid = True
    for name in names:
        try:
            array = np.asarray(getattr(evidence, name))
            if array.dtype.kind not in "biuf":
                valid = False
                continue
            arrays[name] = array
        except (TypeError, ValueError):
            valid = False
    return arrays, valid


def _current_maximum(
    array: np.ndarray | None, expected_policy_count: int | None
) -> tuple[float, bool]:
    if (
        array is None
        or expected_policy_count is None
        or array.shape != (expected_policy_count,)
        or not bool(np.all(np.isfinite(array)))
    ):
        return math.nan, False
    return float(np.max(array)), True


def _local_non_regression(
    candidate: np.ndarray | None, active: np.ndarray | None, tolerance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    valid = (
        candidate is not None
        and active is not None
        and candidate.ndim == 2
        and active.ndim == 2
        and candidate.shape[0] > 0
        and active.shape[0] > 0
        and candidate.shape[1] > 0
        and candidate.shape == active.shape
        and bool(np.all(np.isfinite(candidate)))
        and bool(np.all(np.isfinite(active)))
    )
    if not valid:
        empty_float = np.empty(0, dtype=np.float64)
        return empty_float, empty_float, np.empty(0, dtype=bool), False
    candidate_best = np.max(candidate, axis=0)
    active_best = np.max(active, axis=0)
    passes = candidate_best >= active_best - tolerance
    return candidate_best, active_best, passes, True


def _coverage_and_redundancy(
    margins: np.ndarray | None, safe_margin: float
) -> tuple[float, int, bool]:
    valid = (
        margins is not None
        and margins.ndim == 2
        and margins.shape[0] > 0
        and margins.shape[1] > 0
        and bool(np.all(np.isfinite(margins)))
    )
    if not valid:
        return math.nan, 0, False
    counts = np.count_nonzero(margins >= safe_margin, axis=0)
    return float(np.mean(counts >= 1)), int(np.min(counts)), True


def _minimum_descriptor_distance(
    descriptors: np.ndarray | None, scales: np.ndarray | None, expected_policy_count: int | None
) -> tuple[float, bool]:
    valid = (
        descriptors is not None
        and scales is not None
        and descriptors.ndim == 2
        and scales.ndim == 1
        and expected_policy_count is not None
        and descriptors.shape[0] == expected_policy_count
        and descriptors.shape[0] >= 2
        and descriptors.shape[1] > 0
        and scales.shape == (descriptors.shape[1],)
        and bool(np.all(np.isfinite(descriptors)))
        and bool(np.all(np.isfinite(scales)))
        and bool(np.all(scales > 0))
    )
    if not valid:
        return math.nan, False
    normalized = descriptors / scales
    differences = normalized[:, None, :] - normalized[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    upper_triangle = distances[np.triu_indices(descriptors.shape[0], k=1)]
    return float(np.min(upper_triangle)), True


def _feasible_fraction(margins: np.ndarray | None) -> tuple[float, bool]:
    if margins is None or margins.size == 0 or not bool(np.all(np.isfinite(margins))):
        return math.nan, False
    return float(np.mean(margins >= 0.0)), True


def _worst_nonnegative(samples: np.ndarray | None) -> tuple[float, bool]:
    valid = (
        samples is not None
        and samples.size > 0
        and bool(np.all(np.isfinite(samples)))
        and bool(np.all(samples >= 0.0))
    )
    if not valid:
        return math.nan, False
    return float(np.max(samples)), True


def _number_text(value: float, valid: bool) -> str:
    return format(value, ".17g") if valid and math.isfinite(value) else "invalid"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _report_digest(
    *,
    active_digest: str,
    active_version: int,
    candidate_digest: str,
    candidate_version: int,
    model_version: int,
    validation_set_digest: str,
    gates: Sequence[GateResult],
    candidate_local_best: Sequence[float],
    active_local_best: Sequence[float],
    local_non_regression_passes: Sequence[bool],
) -> str:
    payload = {
        "active_digest": active_digest,
        "active_local_best": [_canonical_float(value) for value in active_local_best],
        "active_version": active_version,
        "candidate_digest": candidate_digest,
        "candidate_local_best": [_canonical_float(value) for value in candidate_local_best],
        "candidate_version": candidate_version,
        "gates": [
            {
                "detail": gate.detail,
                "name": gate.name,
                "observed": gate.observed,
                "passed": gate.passed,
                "requirement": gate.requirement,
            }
            for gate in gates
        ],
        "local_non_regression_passes": list(local_non_regression_passes),
        "model_version": model_version,
        "validation_set_digest": validation_set_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    hasher = hashlib.sha256()
    prefix = b"crazyflow.da_plcbf.validation_report.v1"
    hasher.update(struct.pack(">Q", len(prefix)))
    hasher.update(prefix)
    hasher.update(struct.pack(">Q", len(encoded)))
    hasher.update(encoded)
    return hasher.hexdigest()


def _canonical_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return value.hex()
