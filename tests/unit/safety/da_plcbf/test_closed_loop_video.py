"""Contact presentation must not manufacture controller or learning continuation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from typing import Any

import numpy as np
import pytest

import benchmark.da_plcbf_closed_loop_video as video
from crazyflow.safety.da_plcbf.mujoco_comparison_video import ComparisonVideoTrace, MethodVideoTrace


def _method(position: np.ndarray) -> MethodVideoTrace:
    count = len(position)
    states = np.zeros((count, 13))
    states[:, :3] = position
    states[:, 6] = 1
    trajectories = np.repeat(position[:, None, :], 3, axis=1)
    return MethodVideoTrace(
        position=position.copy(),
        quaternion_xyzw=states[:, 3:7].copy(),
        nominal_rollout=trajectories.copy(),
        fallback_rollouts=np.repeat(trajectories[:, None], 2, axis=1),
        fallback_safe=np.ones((count, 2), dtype=bool),
        selected_policy=np.zeros(count, dtype=np.int32),
        selected_rollout=trajectories.copy(),
        intervention_world=np.ones((count, 3)) * 0.1,
        intervention_norm=np.full(count, np.sqrt(3) * 0.1),
        descriptors=np.zeros((count, 2, 3)),
        library_version=660 + np.arange(count, dtype=np.int32),
        cumulative_gradient_steps=np.arange(count, dtype=np.int32),
        diversity_loss=np.ones(count),
        descriptor_target_loss=np.ones(count),
        gradient_norm=np.ones(count),
        parameter_update_norm=np.full(count, 0.01),
        minimum_library_value=np.full(count, 0.2),
        maximum_library_value=np.full(count, 0.4),
        selected_policy_value=np.full(count, 0.3),
        selected_policy_dual=np.full(count, 0.1),
        qp_valid=np.ones(count, dtype=bool),
        used_fallback=np.zeros(count, dtype=bool),
        degraded=np.zeros(count, dtype=bool),
        full_state=states,
        applied_wrench=np.full((count, 4), 0.2),
        nominal_wrench=np.full((count, 4), 0.1),
        used_emergency=np.zeros(count, dtype=bool),
        executed_policy_dual=np.full(count, 0.1),
        controller_seconds=np.full(count, 0.005),
        learner_seconds=np.full(count, 0.010),
        collision_constraint_active=np.ones(count, dtype=bool),
        goal_position=np.tile((2.0, 0.0, 1.4), (count, 1)),
        waypoint_index=np.arange(count, dtype=np.int32),
        recorded_control_valid=np.ones(count, dtype=bool),
        physical_collision_recorded=np.zeros(count, dtype=bool),
        contact_replay=np.zeros(count, dtype=bool),
    )


def _inputs() -> tuple[ComparisonVideoTrace, dict[str, np.ndarray], dict[str, Any]]:
    times = np.arange(6) * 0.1
    position = np.column_stack((times, np.zeros(6), np.full(6, 1.4)))
    trace = ComparisonVideoTrace(
        time_seconds=times,
        goal_position=np.array((2.0, 0.0, 1.4)),
        obstacles=(),
        true_wind=np.zeros((6, 3)),
        estimated_wind=np.zeros((6, 3)),
        wind_change_time=0.1,
        descriptor_targets=np.zeros((2, 3)),
        fixed=_method(position),
        adaptive=_method(position + np.array((0.0, 0.2, 0.0))),
    )
    contact_times = np.array((0.25, 0.31, 0.37, 0.43, 0.49, 0.55))
    contact_states = np.zeros((len(contact_times), 13))
    contact_states[:, 0] = 10 + np.arange(len(contact_times))
    contact_states[:, 2] = 1.0 - np.arange(len(contact_times)) * 0.15
    contact_states[:, 3:7] = (0.6, 0.0, 0.0, 0.8)
    replay = {"time_seconds": contact_times, "full_state": contact_states}
    metadata = {
        "trigger": {"kind": "swept_collider_contact", "time_seconds": 0.25},
        "obstacle_contact_steps": 8,
    }
    return trace, replay, metadata


def test_contact_splice_preserves_recorded_prefix_adaptive_branch_and_saved_poses() -> None:
    trace, replay, metadata = _inputs()
    original = deepcopy(trace)
    result, audit = video.splice_contact_replay(trace, replay, metadata)
    before = trace.time_seconds < metadata["trigger"]["time_seconds"]
    after = ~before
    np.testing.assert_array_equal(result.time_seconds, original.time_seconds)
    for field in fields(MethodVideoTrace):
        expected = getattr(original.fixed, field.name)
        actual = getattr(result.fixed, field.name)
        unchanged_input = getattr(trace.fixed, field.name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual[before], expected[before], err_msg=field.name)
            np.testing.assert_array_equal(unchanged_input, expected, err_msg=field.name)
        else:
            assert unchanged_input == expected
        expected_adaptive = getattr(original.adaptive, field.name)
        actual_adaptive = getattr(result.adaptive, field.name)
        if isinstance(expected_adaptive, np.ndarray):
            np.testing.assert_array_equal(actual_adaptive, expected_adaptive, err_msg=field.name)
        else:
            assert actual_adaptive == expected_adaptive
    for recorded in result.fixed.full_state[after]:
        assert np.any(np.all(replay["full_state"] == recorded, axis=1))
    np.testing.assert_array_equal(
        result.fixed.full_state[after], replay["full_state"][audit["source_sample_indices"]]
    )
    assert audit["maximum_time_quantization_seconds"] == pytest.approx(0.03)
    assert audit["adaptive_trace_object_unchanged"]
    assert audit["fixed_precontact_states_exact"]
    assert audit["no_control_or_waypoint_credit_after_contact"]
    np.testing.assert_array_equal(result.fixed.position[after], result.fixed.full_state[after, :3])
    np.testing.assert_array_equal(
        result.fixed.quaternion_xyzw[after], result.fixed.full_state[after, 3:7]
    )
    assert np.all(result.fixed.contact_replay[after])
    assert np.all(result.fixed.physical_collision_recorded[after])
    result.validate()


def test_contact_presentation_has_no_commands_updates_or_waypoint_credit_after_handoff() -> None:
    trace, replay, metadata = _inputs()
    result, _audit = video.splice_contact_replay(trace, replay, metadata)
    after = trace.time_seconds >= metadata["trigger"]["time_seconds"]
    last_control = np.flatnonzero(~after)[-1]
    for name in (
        "recorded_control_valid",
        "qp_valid",
        "used_fallback",
        "used_emergency",
        "collision_constraint_active",
    ):
        assert not np.any(getattr(result.fixed, name)[after]), name
    for name in (
        "applied_wrench",
        "nominal_wrench",
        "intervention_world",
        "intervention_norm",
        "selected_policy_dual",
        "executed_policy_dual",
        "gradient_norm",
        "parameter_update_norm",
        "controller_seconds",
        "learner_seconds",
    ):
        np.testing.assert_array_equal(getattr(result.fixed, name)[after], 0, err_msg=name)
    for name in ("library_version", "cumulative_gradient_steps", "waypoint_index"):
        np.testing.assert_array_equal(
            getattr(result.fixed, name)[after],
            getattr(trace.fixed, name)[last_control],
            err_msg=name,
        )
    np.testing.assert_array_equal(result.adaptive.library_version, trace.adaptive.library_version)
    np.testing.assert_array_equal(result.adaptive.waypoint_index, trace.adaptive.waypoint_index)


def test_contact_rows_stop_all_modes_while_ordinary_mode_agreement_remains_strict() -> None:
    trace, replay, metadata = _inputs()
    modes = np.asarray((0, 1, 2, 2, 2, 2), dtype=np.int32)
    trace = replace(
        trace,
        fixed=replace(
            trace.fixed,
            execution_mode=modes,
            qp_valid=modes == 0,
            used_fallback=modes == 1,
            used_emergency=modes == 2,
            used_midpoint=modes == 3,
        ),
    )
    trace.validate()
    presented, _ = video.splice_contact_replay(trace, replay, metadata)
    contact = presented.fixed.contact_replay
    np.testing.assert_array_equal(presented.fixed.execution_mode, modes)
    for name in ("qp_valid", "used_fallback", "used_emergency", "used_midpoint"):
        assert not np.any(getattr(presented.fixed, name)[contact])
    presented.validate()
    bad = presented.fixed.qp_valid.copy()
    bad[0] = False
    with pytest.raises(ValueError, match="qp_valid must agree with execution_mode"):
        replace(presented, fixed=replace(presented.fixed, qp_valid=bad)).validate()
    resumed = presented.fixed.recorded_control_valid.copy()
    resumed[-1] = True
    with pytest.raises(ValueError, match="contact replay must have recorded controls inactive"):
        replace(
            presented, fixed=replace(presented.fixed, recorded_control_valid=resumed)
        ).validate()


@pytest.mark.parametrize("kind", ("safety_shell_abort", "envelope_breach", "operational_abort"))
def test_nonphysical_abort_cannot_be_presented_as_an_observed_collision(kind: str) -> None:
    trace, replay, metadata = _inputs()
    metadata["trigger"]["kind"] = kind
    with pytest.raises(ValueError, match="physical"):
        video.splice_contact_replay(trace, replay, metadata)


@pytest.mark.parametrize(
    ("invalid", "message"),
    (
        ("no_contact", "measured MuJoCo obstacle contact"),
        ("wrong_start", "declared handoff"),
        ("duplicate_time", "increase strictly"),
        ("nonfinite_pose", "finite"),
        ("nonfinite_time", "finite"),
        ("short_replay", "entire displayed continuation"),
        ("outside_flight", "inside the recorded flight interval"),
    ),
)
def test_splice_rejects_unverified_incomplete_or_invalid_contact_recordings(
    invalid: str, message: str
) -> None:
    trace, replay, metadata = _inputs()
    if invalid == "no_contact":
        metadata["obstacle_contact_steps"] = 0
    elif invalid == "wrong_start":
        replay["time_seconds"][0] += 0.001
    elif invalid == "duplicate_time":
        replay["time_seconds"][1] = replay["time_seconds"][0]
    elif invalid == "nonfinite_pose":
        replay["full_state"][1, 0] = np.nan
    elif invalid == "nonfinite_time":
        replay["time_seconds"][1] = np.nan
    elif invalid == "short_replay":
        replay = {name: value[:-1] for name, value in replay.items()}
    elif invalid == "outside_flight":
        metadata["trigger"]["time_seconds"] = 0.0
        replay["time_seconds"] -= 0.25
    with pytest.raises(ValueError, match=message):
        video.splice_contact_replay(trace, replay, metadata)
