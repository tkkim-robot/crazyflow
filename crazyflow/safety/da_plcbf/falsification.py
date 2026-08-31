"""Deterministic fixed-budget empirical falsification with replayable evidence.

The executor evaluates every predeclared randomized boundary candidate, then performs a fixed
number of coordinate-refinement evaluations around the lowest observed margins.  Outcome values
may choose a refinement incumbent, but they never change the number of calls, component order,
step schedule, or retained records.  Exceptions, invalid returns, non-finite returns, safe results,
and counterexamples are all immutable evidence rows.

This is an empirical search over declared finite bounds.  Finding a negative-margin point is a
counterexample for the evaluated implementation and configuration.  Failing to find one is not a
proof of safety, feasibility, or robustness outside the evaluated points.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from crazyflow.safety.da_plcbf.scientific_evaluation import (
    BoundaryCandidateSet,
    BoundaryVariable,
    FalsificationAxis,
    RNGProvenance,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


FALSIFICATION_SCHEMA_VERSION = 1
FALSIFICATION_ALGORITHM = "fixed-budget-coordinate-refinement-v1"
FALSIFICATION_CLAIM_BOUNDARY = (
    "Results are empirical outcomes at the recorded bounded inputs for the identified evaluator. "
    "A retained counterexample falsifies the tested finite-margin claim at that input; absence of "
    "a counterexample does not prove safety, feasibility, robustness, or real-world performance."
)
_MAX_RESULT_BYTES = 128 * 1024 * 1024
_SLUG = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FalsificationStage(str, Enum):
    """Origin of one retained evaluator call."""

    RANDOMIZED = "randomized"
    REFINEMENT = "refinement"


class FalsificationStatus(str, Enum):
    """Machine-readable outcome of one pure-evaluator call."""

    SUCCESS = "success"
    EXCEPTION = "exception"
    INVALID_RETURN = "invalid-return"
    NONFINITE_RETURN = "nonfinite-return"


@dataclass(frozen=True, slots=True)
class FalsificationConfig:
    """Predeclared fixed refinement budget and counterexample threshold."""

    refinement_seed_count: int = 4
    refinement_rounds: int = 4
    initial_step_fraction: float = 0.10
    step_decay: float = 0.50
    counterexample_margin: float = 0.0
    maximum_failure_message_characters: int = 512

    def validate(self, *, candidate_count: int | None = None) -> None:
        """Reject adaptive, empty, non-finite, or over-wide search settings."""
        _positive_integer(self.refinement_seed_count, "refinement_seed_count")
        _positive_integer(self.refinement_rounds, "refinement_rounds")
        if candidate_count is not None:
            _positive_integer(candidate_count, "candidate_count")
            if self.refinement_seed_count > candidate_count:
                raise ValueError("refinement_seed_count must not exceed candidate_count")
        initial = _finite(self.initial_step_fraction, "initial_step_fraction")
        decay = _finite(self.step_decay, "step_decay")
        if not 0.0 < initial <= 1.0:
            raise ValueError("initial_step_fraction must lie in (0, 1]")
        if not 0.0 < decay <= 1.0:
            raise ValueError("step_decay must lie in (0, 1]")
        _finite(self.counterexample_margin, "counterexample_margin")
        _positive_integer(
            self.maximum_failure_message_characters, "maximum_failure_message_characters"
        )
        if self.maximum_failure_message_characters > 4096:
            raise ValueError("maximum_failure_message_characters must not exceed 4096")

    def evaluation_budget(self, candidates: BoundaryCandidateSet) -> int:
        """Return the exact evaluator-call count implied by this config and candidate set."""
        if not isinstance(candidates, BoundaryCandidateSet):
            raise TypeError("candidates must be a BoundaryCandidateSet")
        candidates.validate()
        self.validate(candidate_count=candidates.count)
        dimension = candidates.values.shape[1]
        return (
            candidates.count + 2 * dimension * self.refinement_seed_count * self.refinement_rounds
        )


@dataclass(frozen=True, slots=True)
class FalsificationEvaluation:
    """One retained randomized or coordinate-refinement evaluator call."""

    evaluation_index: int
    stage: FalsificationStage
    source_candidate_index: int
    refinement_seed_rank: int
    refinement_round: int
    refinement_component: int
    refinement_direction: int
    parent_evaluation_index: int
    values: tuple[float, ...]
    status: FalsificationStatus
    margin: float | None
    counterexample: bool
    error_type: str | None
    error_message: str | None

    @property
    def operational_failure(self) -> bool:
        """Whether the evaluator failed to return one finite scalar margin."""
        return self.status is not FalsificationStatus.SUCCESS

    def validate(self, *, dimension: int, config: FalsificationConfig) -> None:
        """Validate row shape, stage sentinels, and exact outcome semantics."""
        _nonnegative_integer(self.evaluation_index, "evaluation_index")
        if not isinstance(self.stage, FalsificationStage):
            raise TypeError("stage must be FalsificationStage")
        _nonnegative_integer(self.source_candidate_index, "source_candidate_index")
        if not isinstance(self.values, tuple) or len(self.values) != dimension:
            raise ValueError("evaluation values must be an immutable tuple matching the dimension")
        if any(
            not isinstance(value, Real) or not math.isfinite(float(value)) for value in self.values
        ):
            raise ValueError("evaluation values must be finite real scalars")
        if not isinstance(self.status, FalsificationStatus):
            raise TypeError("status must be FalsificationStatus")
        if not isinstance(self.counterexample, bool):
            raise TypeError("counterexample must be boolean")

        if self.stage is FalsificationStage.RANDOMIZED:
            if (
                self.refinement_seed_rank != -1
                or self.refinement_round != -1
                or self.refinement_component != -1
                or self.refinement_direction != 0
                or self.parent_evaluation_index != -1
            ):
                raise ValueError("randomized rows must use refinement sentinels")
        else:
            _nonnegative_integer(self.refinement_seed_rank, "refinement_seed_rank")
            _nonnegative_integer(self.refinement_round, "refinement_round")
            _nonnegative_integer(self.refinement_component, "refinement_component")
            _nonnegative_integer(self.parent_evaluation_index, "parent_evaluation_index")
            if self.refinement_direction not in (-1, 1):
                raise ValueError("refinement_direction must be -1 or +1")

        if self.status is FalsificationStatus.SUCCESS:
            margin = _finite(self.margin, "margin")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful evaluations cannot contain failure text")
            if self.counterexample != (margin <= config.counterexample_margin):
                raise ValueError("counterexample flag does not match the predeclared margin")
        else:
            if self.margin is not None or self.counterexample:
                raise ValueError(
                    "failed evaluations require no margin and cannot be counterexamples"
                )
            _failure_text(self.error_type, "error_type", maximum=256)
            _failure_text(
                self.error_message,
                "error_message",
                maximum=config.maximum_failure_message_characters,
            )


@dataclass(frozen=True, slots=True)
class FalsificationResult:
    """Complete, content-addressed empirical search evidence."""

    schema_version: int
    search_name: str
    evaluator_name: str
    evaluator_sha256: str
    candidates: BoundaryCandidateSet
    config: FalsificationConfig
    evaluations: tuple[FalsificationEvaluation, ...]
    algorithm: str = FALSIFICATION_ALGORITHM
    claim_boundary: str = FALSIFICATION_CLAIM_BOUNDARY

    @property
    def sha256(self) -> str:
        """Canonical digest of the complete result, including every evaluator outcome."""
        return _document_sha256(_result_document(self))

    @property
    def evaluation_budget(self) -> int:
        """Exact predeclared number of evaluator calls."""
        return self.config.evaluation_budget(self.candidates)

    @property
    def counterexamples(self) -> tuple[FalsificationEvaluation, ...]:
        """All successful evaluations at or below the declared counterexample margin."""
        return tuple(record for record in self.evaluations if record.counterexample)

    @property
    def failures(self) -> tuple[FalsificationEvaluation, ...]:
        """All retained evaluator exceptions and invalid/non-finite returns."""
        return tuple(record for record in self.evaluations if record.operational_failure)

    def validate(self) -> None:
        """Validate provenance, fixed budget, complete randomized rows, and refinement replay."""
        if self.schema_version != FALSIFICATION_SCHEMA_VERSION:
            raise ValueError("unsupported falsification-result schema version")
        _slug(self.search_name, "search_name")
        _slug(self.evaluator_name, "evaluator_name")
        _sha256(self.evaluator_sha256, "evaluator_sha256")
        if self.algorithm != FALSIFICATION_ALGORITHM:
            raise ValueError("falsification algorithm identifier is unsupported")
        if self.claim_boundary != FALSIFICATION_CLAIM_BOUNDARY:
            raise ValueError("falsification empirical claim boundary is not canonical")
        if not isinstance(self.candidates, BoundaryCandidateSet):
            raise TypeError("candidates must be a BoundaryCandidateSet")
        self.candidates.validate()
        if (
            self.candidates.rng.stream_name != "boundary_falsification"
            or not self.candidates.rng.labels
            or self.candidates.rng.labels[0] != self.search_name
        ):
            raise ValueError("search_name does not match boundary-candidate RNG provenance")
        if not isinstance(self.config, FalsificationConfig):
            raise TypeError("config must be FalsificationConfig")
        self.config.validate(candidate_count=self.candidates.count)
        if not isinstance(self.evaluations, tuple):
            raise TypeError("evaluations must be an immutable tuple")
        if len(self.evaluations) != self.evaluation_budget:
            raise ValueError("evaluation rows do not match the predeclared fixed budget")

        dimension = self.candidates.values.shape[1]
        lower, upper = _candidate_bounds(self.candidates)
        for index, record in enumerate(self.evaluations):
            if not isinstance(record, FalsificationEvaluation):
                raise TypeError("every evaluation must be FalsificationEvaluation")
            record.validate(dimension=dimension, config=self.config)
            if record.evaluation_index != index:
                raise ValueError("evaluation indices must be contiguous and ordered")
            values = np.asarray(record.values, dtype=np.float64)
            if np.any(values < lower) or np.any(values > upper):
                raise ValueError("evaluation values exceed the declared candidate bounds")

        for index in range(self.candidates.count):
            record = self.evaluations[index]
            if (
                record.stage is not FalsificationStage.RANDOMIZED
                or record.source_candidate_index != index
                or not np.array_equal(
                    np.asarray(record.values, dtype=np.float64), self.candidates.values[index]
                )
            ):
                raise ValueError("every randomized boundary candidate must be evaluated once first")

        seed_indices = _refinement_seed_indices(
            self.evaluations[: self.candidates.count], self.config.refinement_seed_count
        )
        cursor = self.candidates.count
        span = upper - lower
        for seed_rank, source_index in enumerate(seed_indices):
            current_index = source_index
            for refinement_round in range(self.config.refinement_rounds):
                round_indices: list[int] = []
                center = np.asarray(self.evaluations[current_index].values, dtype=np.float64)
                for component in range(dimension):
                    step = (
                        self.config.initial_step_fraction
                        * self.config.step_decay**refinement_round
                        * span[component]
                    )
                    for direction in (-1, 1):
                        record = self.evaluations[cursor]
                        expected = center.copy()
                        expected[component] = np.clip(
                            expected[component] + direction * step,
                            lower[component],
                            upper[component],
                        )
                        if (
                            record.stage is not FalsificationStage.REFINEMENT
                            or record.source_candidate_index != source_index
                            or record.refinement_seed_rank != seed_rank
                            or record.refinement_round != refinement_round
                            or record.refinement_component != component
                            or record.refinement_direction != direction
                            or record.parent_evaluation_index != current_index
                            or not np.array_equal(
                                np.asarray(record.values, dtype=np.float64), expected
                            )
                        ):
                            raise ValueError(
                                "refinement row does not match the fixed coordinate plan"
                            )
                        round_indices.append(cursor)
                        cursor += 1
                current_index = _best_finite_evaluation(
                    self.evaluations, (current_index, *round_indices)
                )
        if cursor != len(self.evaluations):
            raise ValueError("unexpected trailing falsification evaluations")


def boundary_candidate_set_sha256(candidates: BoundaryCandidateSet) -> str:
    """Return a canonical digest for a complete boundary candidate set and RNG provenance."""
    if not isinstance(candidates, BoundaryCandidateSet):
        raise TypeError("candidates must be a BoundaryCandidateSet")
    candidates.validate()
    return _document_sha256(_candidate_document(candidates))


def run_fixed_budget_falsification(
    candidates: BoundaryCandidateSet,
    evaluator: Callable[[np.ndarray], Real],
    *,
    search_name: str,
    evaluator_name: str,
    evaluator_sha256: str,
    config: FalsificationConfig | None = None,
) -> FalsificationResult:
    """Run every randomized row and a deterministic, fixed-budget coordinate refinement.

    The evaluator must be a pure function from one read-only ``float64[D]`` vector to one finite
    real margin, where lower values are worse.  The executor catches ordinary exceptions and
    retains them as operational failures.  It never retries, drops, or replaces a recorded call.
    """
    if not isinstance(candidates, BoundaryCandidateSet):
        raise TypeError("candidates must be a BoundaryCandidateSet")
    candidates.validate()
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    _slug(search_name, "search_name")
    _slug(evaluator_name, "evaluator_name")
    _sha256(evaluator_sha256, "evaluator_sha256")
    if (
        candidates.rng.stream_name != "boundary_falsification"
        or not candidates.rng.labels
        or candidates.rng.labels[0] != search_name
    ):
        raise ValueError("search_name does not match boundary-candidate RNG provenance")
    resolved = FalsificationConfig() if config is None else config
    if not isinstance(resolved, FalsificationConfig):
        raise TypeError("config must be FalsificationConfig")
    resolved.validate(candidate_count=candidates.count)

    evaluations: list[FalsificationEvaluation] = []

    def evaluate(
        values: np.ndarray,
        *,
        stage: FalsificationStage,
        source_candidate_index: int,
        seed_rank: int = -1,
        refinement_round: int = -1,
        component: int = -1,
        direction: int = 0,
        parent_index: int = -1,
    ) -> int:
        outcome = _evaluate_one(evaluator, values, resolved)
        index = len(evaluations)
        evaluations.append(
            FalsificationEvaluation(
                evaluation_index=index,
                stage=stage,
                source_candidate_index=source_candidate_index,
                refinement_seed_rank=seed_rank,
                refinement_round=refinement_round,
                refinement_component=component,
                refinement_direction=direction,
                parent_evaluation_index=parent_index,
                values=tuple(float(value) for value in values),
                **outcome,
            )
        )
        return index

    for candidate_index, values in enumerate(candidates.values):
        evaluate(
            values, stage=FalsificationStage.RANDOMIZED, source_candidate_index=candidate_index
        )

    lower, upper = _candidate_bounds(candidates)
    span = upper - lower
    seed_indices = _refinement_seed_indices(evaluations, resolved.refinement_seed_count)
    for seed_rank, source_index in enumerate(seed_indices):
        current_index = source_index
        for refinement_round in range(resolved.refinement_rounds):
            center = np.asarray(evaluations[current_index].values, dtype=np.float64)
            round_indices: list[int] = []
            for component in range(center.size):
                step = (
                    resolved.initial_step_fraction
                    * resolved.step_decay**refinement_round
                    * span[component]
                )
                for direction in (-1, 1):
                    proposal = center.copy()
                    proposal[component] = np.clip(
                        proposal[component] + direction * step, lower[component], upper[component]
                    )
                    round_indices.append(
                        evaluate(
                            proposal,
                            stage=FalsificationStage.REFINEMENT,
                            source_candidate_index=source_index,
                            seed_rank=seed_rank,
                            refinement_round=refinement_round,
                            component=component,
                            direction=direction,
                            parent_index=current_index,
                        )
                    )
            current_index = _best_finite_evaluation(evaluations, (current_index, *round_indices))

    result = FalsificationResult(
        schema_version=FALSIFICATION_SCHEMA_VERSION,
        search_name=search_name,
        evaluator_name=evaluator_name,
        evaluator_sha256=evaluator_sha256,
        candidates=candidates,
        config=resolved,
        evaluations=tuple(evaluations),
    )
    result.validate()
    return result


def verify_falsification_replay(
    result: FalsificationResult,
    evaluator: Callable[[np.ndarray], Real],
    *,
    absolute_tolerance: float = 0.0,
) -> str:
    """Re-evaluate every retained point and return the unchanged result digest on exact replay."""
    if not isinstance(result, FalsificationResult):
        raise TypeError("result must be FalsificationResult")
    result.validate()
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    tolerance = _finite(absolute_tolerance, "absolute_tolerance")
    if tolerance < 0.0:
        raise ValueError("absolute_tolerance must be nonnegative")
    for expected in result.evaluations:
        actual = _evaluate_one(
            evaluator, np.asarray(expected.values, dtype=np.float64), result.config
        )
        if actual["status"] is not expected.status:
            raise ValueError(
                f"falsification replay status mismatch at row {expected.evaluation_index}"
            )
        if expected.status is FalsificationStatus.SUCCESS:
            assert expected.margin is not None and actual["margin"] is not None
            if (
                not math.isclose(
                    float(actual["margin"]), expected.margin, rel_tol=0.0, abs_tol=tolerance
                )
                or actual["counterexample"] != expected.counterexample
            ):
                raise ValueError(
                    f"falsification replay margin mismatch at row {expected.evaluation_index}"
                )
        elif (
            actual["error_type"] != expected.error_type
            or actual["error_message"] != expected.error_message
        ):
            raise ValueError(
                f"falsification replay failure mismatch at row {expected.evaluation_index}"
            )
    return result.sha256


def save_falsification_result(
    result: FalsificationResult, path: str | os.PathLike[str], *, overwrite: bool = False
) -> str:
    """Atomically serialize complete replayable evidence to canonical JSON."""
    if not isinstance(result, FalsificationResult):
        raise TypeError("result must be FalsificationResult")
    result.validate()
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        raise ValueError("falsification-result path must end in .json")
    if not destination.parent.is_dir():
        raise FileNotFoundError("falsification-result parent directory does not exist")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    document = _result_document(result)
    document["content_sha256"] = result.sha256
    payload = _canonical_json(document) + b"\n"
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return result.sha256


def load_falsification_result(path: str | os.PathLike[str]) -> FalsificationResult:
    """Load strict JSON, reconstruct the fixed search plan, and verify its content digest."""
    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError("falsification-result path must end in .json")
    try:
        if source.stat().st_size > _MAX_RESULT_BYTES:
            raise ValueError("falsification result exceeds the size limit")
        document = json.loads(source.read_text(), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("falsification result is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError("falsification result root must be an object")
    stored_digest = document.pop("content_sha256", None)
    _sha256(stored_digest, "content_sha256")
    try:
        result = _result_from_document(document)
        result.validate()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("falsification result failed schema validation") from error
    if not _constant_time_equal(stored_digest, result.sha256):
        raise ValueError("falsification result content digest mismatch")
    return result


def _evaluate_one(
    evaluator: Callable[[np.ndarray], Real], values: np.ndarray, config: FalsificationConfig
) -> dict[str, Any]:
    input_values = np.ascontiguousarray(values, dtype=np.float64).copy()
    input_values.flags.writeable = False
    try:
        returned = evaluator(input_values)
    except Exception as error:  # noqa: BLE001 - every ordinary evaluator failure is evidence.
        return {
            "status": FalsificationStatus.EXCEPTION,
            "margin": None,
            "counterexample": False,
            "error_type": _bounded_failure_text(
                type(error).__qualname__, maximum=256, fallback="evaluator-exception"
            ),
            "error_message": _bounded_failure_text(
                str(error),
                maximum=config.maximum_failure_message_characters,
                fallback=type(error).__qualname__,
            ),
        }
    array = np.asarray(returned)
    if array.shape != () or array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        return {
            "status": FalsificationStatus.INVALID_RETURN,
            "margin": None,
            "counterexample": False,
            "error_type": "invalid-return",
            "error_message": "evaluator must return one real scalar margin",
        }
    margin = float(array)
    if not math.isfinite(margin):
        return {
            "status": FalsificationStatus.NONFINITE_RETURN,
            "margin": None,
            "counterexample": False,
            "error_type": "nonfinite-return",
            "error_message": "evaluator returned a non-finite margin",
        }
    return {
        "status": FalsificationStatus.SUCCESS,
        "margin": margin,
        "counterexample": margin <= config.counterexample_margin,
        "error_type": None,
        "error_message": None,
    }


def _refinement_seed_indices(
    evaluations: Sequence[FalsificationEvaluation], count: int
) -> tuple[int, ...]:
    successful = sorted(
        (
            (float(record.margin), record.evaluation_index)
            for record in evaluations
            if record.status is FalsificationStatus.SUCCESS
        ),
        key=lambda item: (item[0], item[1]),
    )
    selected = [index for _, index in successful[:count]]
    if len(selected) < count:
        selected_set = set(selected)
        selected.extend(
            record.evaluation_index
            for record in evaluations
            if record.evaluation_index not in selected_set
        )
    return tuple(selected[:count])


def _best_finite_evaluation(
    evaluations: Sequence[FalsificationEvaluation], indices: Sequence[int]
) -> int:
    finite = [
        (float(evaluations[index].margin), index)
        for index in indices
        if evaluations[index].status is FalsificationStatus.SUCCESS
    ]
    return min(finite, key=lambda item: (item[0], item[1]))[1] if finite else int(indices[0])


def _candidate_bounds(candidates: BoundaryCandidateSet) -> tuple[np.ndarray, np.ndarray]:
    lower = np.concatenate(
        [np.asarray(variable.lower, dtype=np.float64) for variable in candidates.variables]
    )
    upper = np.concatenate(
        [np.asarray(variable.upper, dtype=np.float64) for variable in candidates.variables]
    )
    return lower, upper


def _candidate_document(candidates: BoundaryCandidateSet) -> dict[str, Any]:
    return {
        "variables": [
            {
                "axis": variable.axis.value,
                "name": variable.name,
                "component_names": list(variable.component_names),
                "lower": [float(value) for value in variable.lower],
                "upper": [float(value) for value in variable.upper],
                "nominal": [float(value) for value in variable.nominal],
            }
            for variable in candidates.variables
        ],
        "flat_component_names": list(candidates.flat_component_names),
        "values": candidates.values.tolist(),
        "focus_component": candidates.focus_component.tolist(),
        "focus_side": candidates.focus_side.tolist(),
        "boundary_band_fraction": float(candidates.boundary_band_fraction),
        "background_nominal_fraction": float(candidates.background_nominal_fraction),
        "rng": {
            "root_seed": int(candidates.rng.root_seed),
            "stream_name": candidates.rng.stream_name,
            "stream_id": int(candidates.rng.stream_id),
            "labels": list(candidates.rng.labels),
            "derived_seed": int(candidates.rng.derived_seed),
            "algorithm": candidates.rng.algorithm,
        },
    }


def _result_document(result: FalsificationResult) -> dict[str, Any]:
    return {
        "schema_version": int(result.schema_version),
        "artifact_type": "crazyflow-da-plcbf-falsification",
        "search_name": result.search_name,
        "evaluator_name": result.evaluator_name,
        "evaluator_sha256": result.evaluator_sha256,
        "algorithm": result.algorithm,
        "claim_boundary": result.claim_boundary,
        "candidate_set_sha256": boundary_candidate_set_sha256(result.candidates),
        "candidates": _candidate_document(result.candidates),
        "config": {
            "refinement_seed_count": int(result.config.refinement_seed_count),
            "refinement_rounds": int(result.config.refinement_rounds),
            "initial_step_fraction": float(result.config.initial_step_fraction),
            "step_decay": float(result.config.step_decay),
            "counterexample_margin": float(result.config.counterexample_margin),
            "maximum_failure_message_characters": (
                int(result.config.maximum_failure_message_characters)
            ),
        },
        "evaluation_budget": int(result.evaluation_budget),
        "evaluations": [
            {
                "evaluation_index": int(record.evaluation_index),
                "stage": record.stage.value,
                "source_candidate_index": int(record.source_candidate_index),
                "refinement_seed_rank": int(record.refinement_seed_rank),
                "refinement_round": int(record.refinement_round),
                "refinement_component": int(record.refinement_component),
                "refinement_direction": int(record.refinement_direction),
                "parent_evaluation_index": int(record.parent_evaluation_index),
                "values": [float(value) for value in record.values],
                "status": record.status.value,
                "margin": None if record.margin is None else float(record.margin),
                "counterexample": bool(record.counterexample),
                "error_type": record.error_type,
                "error_message": record.error_message,
            }
            for record in result.evaluations
        ],
    }


def _result_from_document(document: Mapping[str, Any]) -> FalsificationResult:
    _exact_keys(
        document,
        {
            "schema_version",
            "artifact_type",
            "search_name",
            "evaluator_name",
            "evaluator_sha256",
            "algorithm",
            "claim_boundary",
            "candidate_set_sha256",
            "candidates",
            "config",
            "evaluation_budget",
            "evaluations",
        },
        "result",
    )
    if document["artifact_type"] != "crazyflow-da-plcbf-falsification":
        raise ValueError("falsification artifact type is invalid")
    candidates = _candidate_from_document(document["candidates"])
    if document["candidate_set_sha256"] != boundary_candidate_set_sha256(candidates):
        raise ValueError("embedded boundary-candidate digest mismatch")
    config_document = document["config"]
    _exact_keys(
        config_document,
        {
            "refinement_seed_count",
            "refinement_rounds",
            "initial_step_fraction",
            "step_decay",
            "counterexample_margin",
            "maximum_failure_message_characters",
        },
        "config",
    )
    config = FalsificationConfig(**config_document)
    evaluations_document = document["evaluations"]
    if not isinstance(evaluations_document, list):
        raise TypeError("evaluations must be a list")
    evaluations: list[FalsificationEvaluation] = []
    evaluation_keys = {
        "evaluation_index",
        "stage",
        "source_candidate_index",
        "refinement_seed_rank",
        "refinement_round",
        "refinement_component",
        "refinement_direction",
        "parent_evaluation_index",
        "values",
        "status",
        "margin",
        "counterexample",
        "error_type",
        "error_message",
    }
    for item in evaluations_document:
        _exact_keys(item, evaluation_keys, "evaluation")
        evaluations.append(
            FalsificationEvaluation(
                evaluation_index=item["evaluation_index"],
                stage=FalsificationStage(item["stage"]),
                source_candidate_index=item["source_candidate_index"],
                refinement_seed_rank=item["refinement_seed_rank"],
                refinement_round=item["refinement_round"],
                refinement_component=item["refinement_component"],
                refinement_direction=item["refinement_direction"],
                parent_evaluation_index=item["parent_evaluation_index"],
                values=tuple(item["values"]),
                status=FalsificationStatus(item["status"]),
                margin=item["margin"],
                counterexample=item["counterexample"],
                error_type=item["error_type"],
                error_message=item["error_message"],
            )
        )
    result = FalsificationResult(
        schema_version=document["schema_version"],
        search_name=document["search_name"],
        evaluator_name=document["evaluator_name"],
        evaluator_sha256=document["evaluator_sha256"],
        candidates=candidates,
        config=config,
        evaluations=tuple(evaluations),
        algorithm=document["algorithm"],
        claim_boundary=document["claim_boundary"],
    )
    if document["evaluation_budget"] != result.evaluation_budget:
        raise ValueError("serialized evaluation budget is inconsistent")
    return result


def _candidate_from_document(document: Mapping[str, Any]) -> BoundaryCandidateSet:
    _exact_keys(
        document,
        {
            "variables",
            "flat_component_names",
            "values",
            "focus_component",
            "focus_side",
            "boundary_band_fraction",
            "background_nominal_fraction",
            "rng",
        },
        "candidates",
    )
    variables_document = document["variables"]
    if not isinstance(variables_document, list):
        raise TypeError("candidate variables must be a list")
    variables: list[BoundaryVariable] = []
    for item in variables_document:
        _exact_keys(
            item,
            {"axis", "name", "component_names", "lower", "upper", "nominal"},
            "boundary variable",
        )
        variables.append(
            BoundaryVariable(
                axis=FalsificationAxis(item["axis"]),
                name=item["name"],
                component_names=tuple(item["component_names"]),
                lower=tuple(item["lower"]),
                upper=tuple(item["upper"]),
                nominal=tuple(item["nominal"]),
            )
        )
    rng_document = document["rng"]
    _exact_keys(
        rng_document,
        {"root_seed", "stream_name", "stream_id", "labels", "derived_seed", "algorithm"},
        "rng",
    )
    rng = RNGProvenance(
        root_seed=rng_document["root_seed"],
        stream_name=rng_document["stream_name"],
        stream_id=rng_document["stream_id"],
        labels=tuple(rng_document["labels"]),
        derived_seed=rng_document["derived_seed"],
        algorithm=rng_document["algorithm"],
    )
    return BoundaryCandidateSet(
        variables=tuple(variables),
        flat_component_names=tuple(document["flat_component_names"]),
        values=np.asarray(document["values"], dtype=np.float64),
        focus_component=np.asarray(document["focus_component"], dtype=np.int64),
        focus_side=np.asarray(document["focus_side"], dtype=np.int8),
        boundary_band_fraction=document["boundary_band_fraction"],
        background_nominal_fraction=document["background_nominal_fraction"],
        rng=rng,
    )


def _document_sha256(document: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(b"crazyflow.da_plcbf.falsification.v1\0")
    digest.update(_canonical_json(document))
    return digest.hexdigest()


def _canonical_json(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _exact_keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} has missing or unexpected fields")


def _constant_time_equal(left: str, right: str) -> bool:
    import hmac

    return hmac.compare_digest(left, right)


def _slug(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SLUG.fullmatch(value) is None:
        raise ValueError(f"{name} must be a portable nonempty slug")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite(value: Any, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _bounded_failure_text(value: Any, *, maximum: int, fallback: str) -> str:
    cleaned = " ".join(str(value).replace("\0", " ").split()) or fallback
    return cleaned[:maximum]


def _failure_text(value: Any, name: str, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(character in value for character in "\r\n\0")
    ):
        raise ValueError(f"{name} must be a bounded single-line string")
    return value


__all__ = [
    "FALSIFICATION_ALGORITHM",
    "FALSIFICATION_CLAIM_BOUNDARY",
    "FALSIFICATION_SCHEMA_VERSION",
    "FalsificationConfig",
    "FalsificationEvaluation",
    "FalsificationResult",
    "FalsificationStage",
    "FalsificationStatus",
    "boundary_candidate_set_sha256",
    "load_falsification_result",
    "run_fixed_budget_falsification",
    "save_falsification_result",
    "verify_falsification_replay",
]
