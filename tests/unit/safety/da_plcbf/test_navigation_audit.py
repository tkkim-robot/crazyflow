"""Rejected proposal diagnostics remain distinct from finite executed-command evidence."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from crazyflow.safety.da_plcbf.navigation_audit import audit_navigation_execution
from crazyflow.safety.da_plcbf.navigation_world import (
    NavigationWorld,
    NavigationWorldConfig,
    build_navigation_world,
)
from crazyflow.safety.da_plcbf.version_a_barriers import safety_constraint_names


def _records() -> tuple[np.ndarray, SimpleNamespace, NavigationWorld]:
    world = build_navigation_world(NavigationWorldConfig(obstacle_count=0, duration_seconds=0.12))
    states = np.zeros((5, 13))
    states[:, 2] = states[:, 6] = 1.0
    residuals = np.ones((3, 2, 9))
    residuals[-1] = np.nan  # Terminal padding is not a recorded control.
    motors = np.full((3, 8), 0.01)
    motors[-1] = np.nan
    trace = SimpleNamespace(
        position=np.zeros((3, 3)),
        recorded_control_valid=np.asarray([True, True, False]),
        actuator_margins=motors,
        qp_valid=np.asarray([False, False, False]),
        used_fallback=np.asarray([True, True, False]),
        qp_held_operational_residuals=residuals.copy(),
        fallback_held_operational_residuals=residuals.copy(),
        applied_held_operational_residuals=residuals.copy(),
    )
    return states, trace, world


def test_rejected_nonfinite_qp_keeps_raw_evidence_and_strict_json_summary() -> None:
    states, trace, world = _records()
    proposed = trace.qp_held_operational_residuals
    proposed[:2, :, 0] = np.nan
    proposed[0, 1, 1] = -np.inf
    proposed[0, 0, 2] = -1e-4
    proposed[1, 1, 3] = np.inf
    before = proposed.copy()

    summary, margins = audit_navigation_execution(states, trace, world)

    names = safety_constraint_names(0)
    assert summary["qp_predicted_derivative_minimum_by_constraint"][names[0]] is None
    assert summary["qp_predicted_derivative_minimum_by_constraint"][names[1]] == 1.0
    assert summary["qp_predicted_derivative_finite_entries_by_constraint"][names[0]] == 0
    assert summary["qp_predicted_derivative_finite_entries_by_constraint"][names[1]] == 3
    assert summary["qp_predicted_derivative_nonfinite_controls"] == 2
    assert summary["qp_rejected_nonfinite_proposal_controls"] == 2
    assert summary["qp_predicted_derivative_violating_controls"] == 1
    assert summary["applied_predicted_derivative_nonfinite_controls"] == 0
    assert summary["applied_predicted_derivative_violating_controls"] == 0
    assert summary["all_actual_physical_nodes_pass"]
    assert margins.shape == (5, 9)
    np.testing.assert_array_equal(proposed, before)
    restored = json.loads(json.dumps(summary, allow_nan=False))
    assert restored["qp_predicted_derivative_minimum_by_constraint"][names[0]] is None


def test_unused_nonfinite_fallback_is_reported_separately_from_execution() -> None:
    states, trace, world = _records()
    trace.qp_valid[0] = True
    trace.used_fallback[0] = False
    trace.fallback_held_operational_residuals[0] = np.nan

    summary, _ = audit_navigation_execution(states, trace, world)

    assert summary["fallback_unexecuted_nonfinite_proposal_controls"] == 1
    assert summary["fallback_predicted_derivative_violating_controls"] == 0
    assert summary["applied_predicted_derivative_violating_controls"] == 0
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("prefix", ["qp", "fallback", "applied"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_accepted_or_executed_residuals_must_remain_finite(prefix: str, bad_value: float) -> None:
    states, trace, world = _records()
    trace.qp_valid[0] = True
    getattr(trace, f"{prefix}_held_operational_residuals")[0, 0, 0] = bad_value

    with pytest.raises(ValueError, match=f"accepted or executed {prefix} held residuals"):
        audit_navigation_execution(states, trace, world)
