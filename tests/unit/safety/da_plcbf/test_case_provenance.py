"""Clock, validity and executed-constraint provenance for selected case studies."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

import benchmark.da_plcbf_case_attribution as attribution
from benchmark.da_plcbf_case_confirmation import critical_skill_from_controls
from benchmark.da_plcbf_case_discovery import (
    cached_swept_values,
    require_valid_atlas_anchor,
    validate_atlas_branch_snapshot,
)
from crazyflow.safety.da_plcbf.persistent_skill_learner import PersistentLearnerState

if TYPE_CHECKING:
    from typing import Any


def test_snapshot_metadata_binds_the_physical_branch_clock() -> None:
    snapshot = SimpleNamespace(
        metadata={"available_time_seconds": 4.0, "training_before_seconds": 4.0}
    )
    assert validate_atlas_branch_snapshot(snapshot, 4.0) == 4.0
    for when in (3.96, 4.04):
        with pytest.raises(ValueError, match="authenticated atlas"):
            validate_atlas_branch_snapshot(snapshot, when)
    snapshot.metadata["training_before_seconds"] = 4.04
    with pytest.raises(ValueError, match="authenticated atlas"):
        validate_atlas_branch_snapshot(snapshot, 4.0)
    with pytest.raises(ValueError, match="authenticated atlas"):
        validate_atlas_branch_snapshot(SimpleNamespace(metadata={}), 4.0)


def test_geometrically_clear_invalid_policy_cannot_supply_a_cached_certificate() -> None:
    paths = np.asarray([[[0, 0, 0], [0, 0, 0]], [[2, 0, 0], [2, 0, 0]]], float)
    centers = np.zeros((1, 2, 1, 3))
    values, clearances = cached_swept_values(
        paths, centers, np.asarray([[0.3]]), policy_valid=np.asarray([True, False])
    )
    assert np.max(values) < 0
    assert np.isneginf(values[0, 1]) and np.isneginf(clearances[0, 1])
    paths[1] = np.nan
    unmasked, _ = cached_swept_values(paths, centers, np.asarray([[0.3]]))
    np.testing.assert_array_equal(unmasked, values)


def test_legacy_atlas_needs_its_recorded_actuator_validity_evidence() -> None:
    data = {
        f"t0100_{name}": np.zeros((1, 2, 13))
        for name in ("fixed", "adaptive", "nominal", "emergency")
    }
    anchor = {
        "key": "t0100",
        "methods": {
            name: {
                "competency": {
                    "competency_checks": {"all_rollouts_finite_and_actuator_valid": True}
                }
            }
            for name in ("fixed", "adaptive")
        },
    }
    require_valid_atlas_anchor(data, anchor)
    anchor["methods"]["fixed"]["competency"]["competency_checks"][
        "all_rollouts_finite_and_actuator_valid"
    ] = False
    with pytest.raises(ValueError, match="verified all-valid"):
        require_valid_atlas_anchor(data, anchor)
    data["t0100_fixed_valid"] = np.asarray([True])
    require_valid_atlas_anchor(data, anchor)
    data["t0100_fixed_valid"][0] = False
    with pytest.raises(ValueError, match="verified all-valid"):
        require_valid_atlas_anchor(data, anchor)


def test_critical_skill_comes_from_executed_intervention_not_unused_or_terminal_proposals() -> None:
    rows = [
        {"executed": False, "qp": True, "selected": 8, "dual": 0.2},
        {"executed": True, "qp": False, "selected": 7, "dual": 0.4},
        {"executed": True, "qp": True, "selected": 6, "dual": 0.0},
        {"executed": True, "qp": True, "selected": 0, "dual": 0.2},
        {"executed": True, "qp": True, "selected": 3, "dual": 0.01},
    ]
    assert critical_skill_from_controls(rows) == 2
    with pytest.raises(ValueError, match="recorded learned positive-dual"):
        critical_skill_from_controls(rows[:-1])


@pytest.mark.parametrize("record_provenance", [False, True])
def test_rejected_update_does_not_refresh_snapshot_age_or_count_terminal_action(
    monkeypatch: pytest.MonkeyPatch,
    record_provenance: bool,
) -> None:
    monkeypatch.setattr(attribution.jax, "jit", lambda f: f)
    monkeypatch.setattr(attribution.jax, "block_until_ready", lambda x: x)
    monkeypatch.setattr(attribution, "direct_wrench_symplectic_step", lambda x, u, m, dt: x)
    geometry = {
        "actual_xml_sphere_geometry": {"minimum_clearance_upper_bound_m": 1.0},
        "actual_xml_ground_geometry": {"minimum_clearance_upper_bound_m": 1.0},
    }
    monkeypatch.setattr(attribution, "audit_recorded_collider_clearance", lambda *args: geometry)
    world = SimpleNamespace(
        initial_state_time_seconds=4.0,
        initial_state=np.asarray([0, 0, 1.4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        config=SimpleNamespace(dt=0.02, control_interval_steps=2, control_period=0.04),
        case_study_config=SimpleNamespace(hover_position=(0, 0, 1.4), navigation_start_seconds=10),
        dynamics_at=lambda when, model: SimpleNamespace(model=model),
        obstacle_prediction=lambda when, horizon: None,
    )
    bundle = SimpleNamespace(
        config=SimpleNamespace(dt=0.02, control_interval_steps=2, horizon=4), point_model=None
    )
    state = PersistentLearnerState(
        params=np.asarray([0.0], dtype=np.float32),
        previous_params=np.asarray([-0.1], dtype=np.float32),
        optimizer_state={"momentum": np.asarray([0.2], dtype=np.float32)},
        cumulative_gradient_steps=np.asarray(1),
        latest_dynamics_estimate=None,
        library_version=np.asarray(1),
    )
    updates = iter((False, True))

    def update(persistent: Any, x: Any, model: Any) -> tuple[Any, Any]:
        finite = next(updates)
        following = (
            persistent.replace(
                params=persistent.params + 0.25,
                previous_params=persistent.params,
                optimizer_state={"momentum": persistent.optimizer_state["momentum"] + 0.05},
                cumulative_gradient_steps=persistent.cumulative_gradient_steps + 1,
                library_version=persistent.library_version + 1,
            )
            if finite
            else persistent
        )
        return following, SimpleNamespace(finite_update_applied=finite)

    def controller(*args: Any) -> Any:
        return SimpleNamespace(
            action=np.zeros(4),
            selected_index=1,
            execution_mode=2,
            qp_valid=False,
            degraded=True,
            values=SimpleNamespace(values=np.asarray([0.1, 0.2])),
            smooth_values=np.asarray([-0.01, -0.02]),
            continuous_filter=SimpleNamespace(policy_eligible=np.asarray([False, False])),
            executed_policy_dual=0.0,
            qp_rejection_flags=np.zeros(9, dtype=bool),
        )

    summary, rows, _, times, _ = attribution.run_branch(
        world,
        bundle,
        state,
        controller,
        end=4.08,
        learner=SimpleNamespace(step=update),
        snapshot_available=3.96,
        record_provenance=record_provenance,
    )
    assert [row["snapshot_available_time"] for row in rows] == [3.96, 3.96, 4.08]
    assert [row["executed"] for row in rows] == [True, True, False]
    assert summary["finite_updates"] == 1 and summary["executed_controls"] == 2
    assert summary["last_used_version"] == 1 and rows[-1]["version"] == 2
    assert summary["first_no_hard_collision_path"] is None
    assert summary["first_no_eligible_certificate"] == 4.0
    np.testing.assert_allclose(times, [4.0, 4.02, 4.04, 4.06, 4.08], atol=1e-12)
    if record_provenance:
        initial = attribution.persistent_provenance(state)
        published = [row["published_learner_state"] for row in rows]
        assert published[0] == published[1] == initial
        rejected, applied = [row["completed_update"] for row in rows[:-1]]
        assert rejected["publication_time_seconds"] is None
        assert rejected["before"] == rejected["after"] == initial
        assert applied["before"] == initial and applied["after"] == published[2]
        assert applied["publication_time_seconds"] == rows[2]["snapshot_available_time"]
        assert applied["after"]["parameters_sha256"] != initial["parameters_sha256"]
        assert applied["after"]["optimizer_state_sha256"] != initial["optimizer_state_sha256"]
        assert applied["after"]["previous_parameters_sha256"] == initial["parameters_sha256"]
        assert summary["continuation_provenance"]["final"] == published[2]
        assert "completed_update" not in rows[-1]
        for current, following in zip(rows[:-1], rows[1:], strict=True):
            update_record = current["completed_update"]
            assert (
                current["controller_completed_perf_counter_ns"]
                <= update_record["started_perf_counter_ns"]
                <= update_record["completed_perf_counter_ns"]
                <= following["controller_started_perf_counter_ns"]
            )
    else:
        assert all("published_learner_state" not in row for row in rows)
