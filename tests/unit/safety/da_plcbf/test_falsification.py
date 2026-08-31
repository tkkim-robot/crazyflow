from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from crazyflow.safety.da_plcbf.falsification import (
    FALSIFICATION_CLAIM_BOUNDARY,
    FalsificationConfig,
    FalsificationStage,
    FalsificationStatus,
    boundary_candidate_set_sha256,
    load_falsification_result,
    run_fixed_budget_falsification,
    save_falsification_result,
    verify_falsification_replay,
)
from crazyflow.safety.da_plcbf.scientific_evaluation import (
    BoundaryCandidateSet,
    BoundaryVariable,
    FalsificationAxis,
    generate_boundary_candidates,
)

if TYPE_CHECKING:
    from pathlib import Path


def _candidates(*, search_name: str = "fixed-search") -> BoundaryCandidateSet:
    variables = (
        BoundaryVariable(
            axis=FalsificationAxis.INITIAL_STATE,
            name="state",
            component_names=("x", "v"),
            lower=(-1.0, -2.0),
            upper=(1.0, 2.0),
            nominal=(0.0, 0.0),
        ),
    )
    return generate_boundary_candidates(
        variables,
        count=8,
        root_seed=123,
        search_name=search_name,
        boundary_band_fraction=0.05,
        require_all_phase7_axes=False,
    )


def _quadratic(values: np.ndarray) -> float:
    target = np.asarray((0.35, -0.6))
    return float(np.sum((values - target) ** 2) - 0.15)


def test_executor_is_deterministic_fixed_budget_and_retains_every_call() -> None:
    candidates = _candidates()
    config = FalsificationConfig(
        refinement_seed_count=2, refinement_rounds=3, initial_step_fraction=0.25, step_decay=0.5
    )
    first = run_fixed_budget_falsification(
        candidates,
        _quadratic,
        search_name="fixed-search",
        evaluator_name="quadratic-margin",
        evaluator_sha256="a" * 64,
        config=config,
    )
    repeated = run_fixed_budget_falsification(
        candidates,
        _quadratic,
        search_name="fixed-search",
        evaluator_name="quadratic-margin",
        evaluator_sha256="a" * 64,
        config=config,
    )

    assert first.sha256 == repeated.sha256
    assert first.evaluations == repeated.evaluations
    assert first.evaluation_budget == 8 + 2 * 2 * 2 * 3
    assert len(first.evaluations) == first.evaluation_budget
    assert first.claim_boundary == FALSIFICATION_CLAIM_BOUNDARY
    assert "does not prove safety" in first.claim_boundary
    np.testing.assert_array_equal(
        np.asarray([record.values for record in first.evaluations[: candidates.count]]),
        candidates.values,
    )
    assert all(
        record.stage is FalsificationStage.RANDOMIZED
        for record in first.evaluations[: candidates.count]
    )
    assert all(
        record.stage is FalsificationStage.REFINEMENT
        for record in first.evaluations[candidates.count :]
    )
    random_minimum = min(
        record.margin
        for record in first.evaluations[: candidates.count]
        if record.margin is not None
    )
    overall_minimum = min(
        record.margin for record in first.evaluations if record.margin is not None
    )
    assert overall_minimum <= random_minimum
    assert first.counterexamples
    assert not first.failures
    assert boundary_candidate_set_sha256(candidates) == boundary_candidate_set_sha256(candidates)


def test_failures_invalid_returns_and_nonfinite_returns_are_never_discarded() -> None:
    candidates = _candidates(search_name="failure-search")
    exception_value = candidates.values[0].tobytes()
    nonfinite_value = candidates.values[1].tobytes()
    invalid_value = candidates.values[2].tobytes()

    def evaluator(values: np.ndarray) -> object:
        encoded = values.tobytes()
        if encoded == exception_value:
            raise RuntimeError("declared evaluator failure")
        if encoded == nonfinite_value:
            return np.nan
        if encoded == invalid_value:
            return np.asarray((1.0, 2.0))
        return float(np.sum(values**2) - 0.01)

    config = FalsificationConfig(refinement_seed_count=1, refinement_rounds=1)
    result = run_fixed_budget_falsification(
        candidates,
        evaluator,
        search_name="failure-search",
        evaluator_name="failure-fixture",
        evaluator_sha256="b" * 64,
        config=config,
    )

    assert len(result.evaluations) == result.evaluation_budget
    assert result.evaluations[0].status is FalsificationStatus.EXCEPTION
    assert result.evaluations[1].status is FalsificationStatus.NONFINITE_RETURN
    assert result.evaluations[2].status is FalsificationStatus.INVALID_RETURN
    assert len(result.failures) >= 3
    assert all(record.margin is None and not record.counterexample for record in result.failures)
    assert verify_falsification_replay(result, evaluator) == result.sha256


def test_refinement_structure_is_validated_and_cannot_be_outcome_pruned() -> None:
    result = run_fixed_budget_falsification(
        _candidates(),
        _quadratic,
        search_name="fixed-search",
        evaluator_name="quadratic-margin",
        evaluator_sha256="c" * 64,
        config=FalsificationConfig(refinement_seed_count=1, refinement_rounds=1),
    )
    with pytest.raises(ValueError, match="fixed budget"):
        replace(result, evaluations=result.evaluations[:-1]).validate()

    records = list(result.evaluations)
    refinement_index = result.candidates.count
    record = records[refinement_index]
    values = list(record.values)
    values[0] += 1e-4
    records[refinement_index] = replace(record, values=tuple(values))
    with pytest.raises(ValueError, match="coordinate plan"):
        replace(result, evaluations=tuple(records)).validate()


def test_json_round_trip_is_deterministic_digest_checked_and_replayable(tmp_path: Path) -> None:
    result = run_fixed_budget_falsification(
        _candidates(),
        _quadratic,
        search_name="fixed-search",
        evaluator_name="quadratic-margin",
        evaluator_sha256="d" * 64,
        config=FalsificationConfig(refinement_seed_count=2, refinement_rounds=2),
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert save_falsification_result(result, first) == result.sha256
    assert save_falsification_result(result, second) == result.sha256
    assert first.read_bytes() == second.read_bytes()
    restored = load_falsification_result(first)
    assert restored.sha256 == result.sha256
    assert restored.evaluations == result.evaluations
    np.testing.assert_array_equal(restored.candidates.values, result.candidates.values)
    assert verify_falsification_replay(restored, _quadratic) == result.sha256

    payload = json.loads(first.read_text())
    payload["evaluations"][0]["margin"] += 0.125
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_falsification_result(tampered)

    with pytest.raises(ValueError, match="margin mismatch"):
        verify_falsification_replay(result, lambda values: _quadratic(values) + 0.1)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (FalsificationConfig(refinement_seed_count=0), "positive integer"),
        (FalsificationConfig(refinement_rounds=0), "positive integer"),
        (FalsificationConfig(initial_step_fraction=0.0), "initial_step_fraction"),
        (FalsificationConfig(step_decay=1.1), "step_decay"),
        (FalsificationConfig(maximum_failure_message_characters=5000), "must not exceed"),
    ],
)
def test_invalid_fixed_budget_config_is_rejected(config: FalsificationConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate(candidate_count=8)
