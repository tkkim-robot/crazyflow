from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import crazyflow.safety.da_plcbf.scientific_dashboard as scientific_dashboard_module
from crazyflow.safety.da_plcbf.artifact_smoke import synthetic_trace
from crazyflow.safety.da_plcbf.artifacts import ArtifactEvent, load_trace, save_trace
from crazyflow.safety.da_plcbf.scenarios import (
    ScenarioTapeConfig,
    generate_scenario_tape,
    load_scenario_tape,
    save_scenario_tape,
)
from crazyflow.safety.da_plcbf.scientific_dashboard import (
    VISUAL_REVIEW_SCHEMA_VERSION,
    VisualReviewCheck,
    VisualReviewRecord,
    change_annotations,
    evidence_inventory,
    extract_keyframes,
    load_visual_review_record,
    render_contact_sheet,
    render_scientific_dashboard,
    scientific_dashboard_frames,
    select_keyframe_indices,
    verify_scientific_dashboard_replay,
    write_visual_review_record,
)

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.artifacts import ImmutableTrace
    from crazyflow.safety.da_plcbf.scenarios import ScenarioTape


_CHECK_NAMES = (
    "original_resolution_inspected",
    "labels_legible_without_console",
    "unsafe_and_degraded_visibly_distinct",
    "overlays_agree_with_trace",
    "event_annotations_agree_with_trace",
    "camera_and_occlusion_acceptable",
    "scales_units_and_timing_clear",
    "unavailable_evidence_explicit",
)


def _trace_and_tape() -> tuple[ImmutableTrace, ScenarioTape, tuple[ArtifactEvent, ...]]:
    tape = generate_scenario_tape(73, ScenarioTapeConfig(steps=6, dt=0.1), fold=4)
    base = synthetic_trace(tape.sha256, steps=tape.steps, dt=0.1)
    barriers = np.array(base.hard_barriers, copy=True)
    barriers[2, 0] = -0.04
    failure = np.array(base.failure, copy=True)
    failure[2] = True
    failure[4] = True
    degraded = np.array(base.degraded, copy=True)
    degraded[1] = True
    contact = np.array(base.contact, copy=True)
    contact[4] = True
    snapshot = np.array(base.snapshot_version, copy=True)
    snapshot[3:] = 1
    model = np.array(base.model_version, copy=True)
    model[4:] = 1
    postcheck = np.array(base.postcheck_residual, copy=True)
    postcheck[2] = -0.04
    trace = replace(
        base,
        hard_barriers=barriers,
        failure=failure,
        degraded=degraded,
        contact=contact,
        snapshot_version=snapshot,
        model_version=model,
        postcheck_residual=postcheck,
    )
    events = (
        ArtifactEvent(
            sequence=0,
            step=3,
            time_seconds=float(trace.time[3]),
            category="learner",
            name="candidate-admitted",
            severity="info",
            snapshot_version=1,
            model_version=0,
            details={"reason": "hard-gates-passed"},
        ),
        ArtifactEvent(
            sequence=1,
            step=4,
            time_seconds=float(trace.time[4]),
            category="safety",
            name="contact-detected",
            severity="failure",
            snapshot_version=1,
            model_version=1,
            details={"hard_failure": True},
        ),
    )
    return trace, tape, events


def _passing_review(
    trace: ImmutableTrace, video: object, keyframes: tuple[int, ...]
) -> VisualReviewRecord:
    validation = video.validation  # type: ignore[attr-defined]
    checks = tuple(
        VisualReviewCheck(
            name=name,
            status="pass",
            note=f"Inspected recorded evidence for {name.replace('_', ' ')}.",
        )
        for name in _CHECK_NAMES
    )
    return VisualReviewRecord(
        schema_version=VISUAL_REVIEW_SCHEMA_VERSION,
        reviewer="Codex visual inspector",
        reviewer_kind="agent",
        reviewed_utc="2026-08-30T20:00:00Z",
        disposition="pass",
        trace_content_sha256=trace.content_sha256,
        scenario_tape_sha256=str(trace.scenario_tape_sha256),
        dashboard_evidence_sha256=None,
        video_file_sha256=validation.file_sha256,
        decoded_frames_sha256=validation.decoded_frames_sha256,
        frame_width=validation.width,
        frame_height=validation.height,
        keyframe_indices=keyframes,
        checks=checks,
        notes=("Inspected every extracted keyframe and the contact sheet at original resolution.",),
        revisions=("Increased unavailable-evidence contrast before this passing inspection.",),
    )


def _with_terminal_observation(trace: ImmutableTrace) -> ImmutableTrace:
    controls = [
        np.array(value, copy=True)
        for value in (trace.nominal_control, trace.filtered_control, trace.applied_control)
    ]
    for value in controls:
        value[-1] = 0.0
    policy = np.array(trace.policy_values, copy=True)
    training = np.array(trace.training_values, copy=True)
    policy[-1] = 0.0
    training[-1] = 0.0
    selected = np.array(trace.selected_policy, copy=True)
    selected[-1] = -1
    executed = np.ones(trace.steps, dtype=np.bool_)
    executed[-1] = False
    clipped = np.array(trace.clipped, copy=True)
    saturated = np.array(trace.saturated, copy=True)
    clipped[-1] = False
    saturated[-1] = False
    latency = np.array(trace.component_latency_seconds, copy=True)
    latency[-1] = 0.0
    solver = np.array(trace.solver_kkt_residual, copy=True)
    postcheck = np.array(trace.postcheck_residual, copy=True)
    gradient = np.array(trace.gradient_norm, copy=True)
    solver[-1] = postcheck[-1] = gradient[-1] = 0.0
    return replace(
        trace,
        nominal_control=controls[0],
        filtered_control=controls[1],
        applied_control=controls[2],
        executed_control=executed,
        policy_values=policy,
        training_values=training,
        selected_policy=selected,
        solver_kkt_residual=solver,
        postcheck_residual=postcheck,
        clipped=clipped,
        saturated=saturated,
        gradient_norm=gradient,
        component_latency_seconds=latency,
    )


def test_inventory_and_annotations_never_relabel_missing_trace_fields() -> None:
    trace, tape, events = _trace_and_tape()
    inventory = {item.key: item for item in evidence_inventory(trace, tape, events)}
    assert inventory["actual_trajectory"].status == "recorded"
    assert inventory["prediction_tubes"].status == "scenario-recorded"
    assert inventory["candidate_admission"].status == "recorded"
    assert inventory["nominal_rollout"].status == "unavailable"
    assert "controls, not a nominal rollout" in inventory["nominal_rollout"].source
    assert inventory["fallback_rollouts"].status == "unavailable"
    assert inventory["descriptors"].status == "unavailable"
    assert inventory["dynamics_uncertainty"].status == "unavailable"

    annotations = change_annotations(trace, tape, events)
    labels = {(item.step, item.label) for item in annotations}
    assert (1, "degraded onset") in labels
    assert (2, "unsafe onset") in labels
    assert (3, "snapshot swap") in labels
    assert (3, "candidate-admitted") in labels
    assert (4, "contact") in labels
    assert (4, "model update") in labels
    for name, index in zip(tape.schedule_names, tape.schedule_change_indices, strict=True):
        assert (int(index), str(name)) in labels


def test_frames_are_deterministic_legible_size_and_require_exact_tape_binding() -> None:
    trace, tape, events = _trace_and_tape()
    first = next(scientific_dashboard_frames(trace, tape=tape, events=events, size=(1280, 720)))
    second = next(scientific_dashboard_frames(trace, tape=tape, events=events, size=(1280, 720)))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (720, 1280, 3)
    assert first.dtype == np.uint8
    assert np.unique(first.reshape(-1, 3), axis=0).shape[0] > 100

    with pytest.raises(ValueError, match="at least 1280x720"):
        next(scientific_dashboard_frames(trace, tape=tape, size=(640, 360)))
    wrong_tape = generate_scenario_tape(74, ScenarioTapeConfig(steps=6, dt=0.1), fold=4)
    with pytest.raises(ValueError, match="digest"):
        next(scientific_dashboard_frames(trace, tape=wrong_tape, size=(1280, 720)))


def test_keyframe_selection_prioritizes_safety_and_change_evidence() -> None:
    trace, tape, _ = _trace_and_tape()
    indices = select_keyframe_indices(trace, tape=tape, count=6)
    assert indices == tuple(sorted(set(indices)))
    assert {0, 1, 2, trace.steps - 1}.issubset(indices)


def test_terminal_observation_masks_every_control_derived_visual_summary() -> None:
    base, tape, events = _trace_and_tape()
    trace = _with_terminal_observation(base)

    counts = scientific_dashboard_module._safe_policy_counts(trace)
    assert np.all(np.isfinite(counts[:-1]))
    assert np.isnan(counts[-1])
    for values in (
        trace.policy_values,
        trace.loss_terms,
        trace.gradient_norm,
        trace.nominal_control,
        trace.filtered_control,
        trace.applied_control,
        trace.component_latency_seconds,
        trace.solver_kkt_residual,
        trace.postcheck_residual,
    ):
        display = scientific_dashboard_module._mask_nonexecuted_rows(trace, values)
        np.testing.assert_array_equal(display[:-1], np.asarray(values[:-1], dtype=np.float64))
        assert np.all(np.isnan(display[-1]))
    status, _ = scientific_dashboard_module._status(trace, trace.steps - 1)
    assert status.startswith("TERMINAL OBSERVATION")

    final = tuple(scientific_dashboard_frames(trace, tape=tape, events=events, size=(1280, 720)))[
        -1
    ]
    legacy = replace(trace, executed_control=np.ones(trace.steps, dtype=np.bool_))
    legacy_final = tuple(
        scientific_dashboard_frames(legacy, tape=tape, events=events, size=(1280, 720))
    )[-1]
    assert not np.array_equal(final, legacy_final)


def test_visual_review_record_rejects_automatic_or_incomplete_pass_claims() -> None:
    trace, _, _ = _trace_and_tape()
    checks = tuple(
        VisualReviewCheck(name=name, status="pass", note="Explicitly inspected this visual gate.")
        for name in _CHECK_NAMES
    )
    base = VisualReviewRecord(
        schema_version=1,
        reviewer="review agent",
        reviewer_kind="agent",
        reviewed_utc="2026-08-30T20:00:00+00:00",
        disposition="pass",
        trace_content_sha256=trace.content_sha256,
        scenario_tape_sha256=str(trace.scenario_tape_sha256),
        dashboard_evidence_sha256=None,
        video_file_sha256="a" * 64,
        decoded_frames_sha256="b" * 64,
        frame_width=1280,
        frame_height=720,
        keyframe_indices=(0, 2, 5),
        checks=checks,
        notes=("Inspected the full-resolution frames, labels, scales, colors, and overlays.",),
        revisions=(),
    )
    base.validate()
    with pytest.raises(ValueError, match="failed check"):
        replace(base, checks=(replace(checks[0], status="fail"), *checks[1:])).validate()
    with pytest.raises(ValueError, match="every required check"):
        replace(base, checks=checks[:-1]).validate()
    with pytest.raises(ValueError, match="inspection note"):
        replace(base, notes=()).validate()


@pytest.mark.render
def test_saved_schema_trace_renders_replays_and_produces_review_artifacts(tmp_path: Path) -> None:
    trace, tape, events = _trace_and_tape()
    trace_path = tmp_path / "trace.npz"
    tape_path = tmp_path / "tape.npz"
    save_trace(trace, trace_path)
    save_scenario_tape(tape, tape_path)
    saved_trace = load_trace(trace_path)
    saved_tape = load_scenario_tape(tape_path)

    video_path = tmp_path / "scientific-dashboard.mp4"
    rendered = render_scientific_dashboard(
        saved_trace, video_path, tape=saved_tape, events=events, fps=5.0, size=(1280, 720)
    )
    assert rendered.validation.codec == "h264"
    assert rendered.validation.frame_count == saved_trace.steps
    assert rendered.validation.width == 1280
    assert rendered.validation.height == 720
    replay = verify_scientific_dashboard_replay(
        saved_trace, video_path, tape=saved_tape, events=events, fps=5.0, size=(1280, 720)
    )
    assert replay.validation.decoded_frames_sha256 == rendered.validation.decoded_frames_sha256

    keyframes = extract_keyframes(
        video_path, saved_trace, tmp_path / "keyframes", tape=saved_tape, count=5
    )
    assert all(record.width == 1280 and record.height == 720 for record in keyframes)
    assert all(Path(record.path).is_file() for record in keyframes)
    sheet = render_contact_sheet(keyframes, tmp_path / "contact-sheet.png", columns=3)
    assert Path(sheet.path).is_file()
    assert sheet.keyframe_steps == tuple(record.step for record in keyframes)

    review = _passing_review(saved_trace, rendered, tuple(record.step for record in keyframes))
    review_path = tmp_path / "visual_review.md"
    digest = write_visual_review_record(review, review_path)
    assert len(digest) == 64
    assert load_visual_review_record(review_path) == review
