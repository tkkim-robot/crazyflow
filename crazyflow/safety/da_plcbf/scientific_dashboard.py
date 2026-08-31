"""Evidence-faithful offline scientific dashboards and visual-review records.

This module deliberately renders only fields that are present in an :class:`ImmutableTrace` or
an exactly bound :class:`ScenarioTape`.  In particular, controls are never integrated into an
invented trajectory and scenario references are never relabelled as controller-nominal rollouts.
Panels explicitly say ``UNAVAILABLE`` when the version-1 artifact schema cannot carry the requested
evidence.

Frames are rendered with Matplotlib's non-interactive Agg canvas and encoded by the project's
pinned, offline ``imageio-ffmpeg`` executable.  Simulation and controller code are not imported or
executed during replay.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from crazyflow.safety.da_plcbf.artifacts import ArtifactEvent, ImmutableTrace, file_sha256
from crazyflow.safety.da_plcbf.dashboard import VideoValidation, validate_mp4
from crazyflow.safety.da_plcbf.dashboard_evidence import (
    DashboardEvidence,
    validate_dashboard_evidence_binding,
)

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.scenarios import ScenarioTape


SCIENTIFIC_DASHBOARD_SCHEMA_VERSION = 1
VISUAL_REVIEW_SCHEMA_VERSION = 1

_BACKGROUND = "#0b111b"
_PANEL = "#121c2b"
_GRID = "#314057"
_TEXT = "#ecf3fb"
_MUTED = "#93a4b8"
_CYAN = "#46c7e8"
_ORANGE = "#ffb04a"
_GREEN = "#3bc17c"
_RED = "#e8505b"
_PURPLE = "#ab83ff"
_YELLOW = "#f4d35e"

_REQUIRED_REVIEW_CHECKS = (
    "original_resolution_inspected",
    "labels_legible_without_console",
    "unsafe_and_degraded_visibly_distinct",
    "overlays_agree_with_trace",
    "event_annotations_agree_with_trace",
    "camera_and_occlusion_acceptable",
    "scales_units_and_timing_clear",
    "unavailable_evidence_explicit",
)


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """Availability and provenance of one requested visual element."""

    key: str
    label: str
    status: Literal["recorded", "scenario-recorded", "unavailable"]
    source: str


@dataclass(frozen=True, slots=True)
class ChangeAnnotation:
    """One deterministic annotation derived from recorded data."""

    step: int
    label: str
    kind: str


@dataclass(frozen=True, slots=True)
class ScientificDashboardResult:
    """Validated scientific video plus its evidence inventory and encoder identity."""

    validation: VideoValidation
    evidence: tuple[EvidenceItem, ...]
    encoder_executable: str
    schema_version: int = SCIENTIFIC_DASHBOARD_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class KeyframeRecord:
    """One decoded, full-resolution frame extracted from the encoded video."""

    step: int
    time_seconds: float
    path: str
    width: int
    height: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ContactSheetRecord:
    """Metadata for a deterministic PNG contact sheet."""

    path: str
    width: int
    height: int
    keyframe_steps: tuple[int, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class VisualReviewCheck:
    """A review decision with a mandatory, evidence-specific note."""

    name: str
    status: Literal["pass", "fail"]
    note: str

    def validate(self) -> None:
        """Reject unknown checks, invalid decisions, and empty inspection notes."""
        if self.name not in _REQUIRED_REVIEW_CHECKS:
            raise ValueError(f"unknown visual-review check {self.name!r}")
        if self.status not in {"pass", "fail"}:
            raise ValueError("visual-review check status must be pass or fail")
        if not isinstance(self.note, str) or len(self.note.strip()) < 8:
            raise ValueError("every visual-review check requires a substantive inspection note")


@dataclass(frozen=True, slots=True)
class VisualReviewRecord:
    """Structured evidence that a person or agent actually inspected a final video.

    Constructing this record is intentionally not part of rendering.  A renderer can establish
    codec and pixel properties, but it cannot truthfully auto-assert legibility or visual clarity.
    """

    schema_version: int
    reviewer: str
    reviewer_kind: Literal["human", "agent"]
    reviewed_utc: str
    disposition: Literal["pass", "revise"]
    trace_content_sha256: str
    scenario_tape_sha256: str | None
    dashboard_evidence_sha256: str | None
    video_file_sha256: str
    decoded_frames_sha256: str
    frame_width: int
    frame_height: int
    keyframe_indices: tuple[int, ...]
    checks: tuple[VisualReviewCheck, ...]
    notes: tuple[str, ...]
    revisions: tuple[str, ...]

    def validate(self) -> None:
        """Require complete checks, original-resolution evidence, and explicit notes."""
        if self.schema_version != VISUAL_REVIEW_SCHEMA_VERSION:
            raise ValueError("unsupported visual-review schema version")
        if not isinstance(self.reviewer, str) or len(self.reviewer.strip()) < 2:
            raise ValueError("reviewer must be identified")
        if self.reviewer_kind not in {"human", "agent"}:
            raise ValueError("reviewer_kind must be human or agent")
        try:
            reviewed = datetime.fromisoformat(self.reviewed_utc.replace("Z", "+00:00"))
        except (TypeError, ValueError) as error:
            raise ValueError("reviewed_utc must be an ISO-8601 timestamp") from error
        if reviewed.tzinfo is None or reviewed.utcoffset() is None:
            raise ValueError("reviewed_utc must include a timezone")
        if self.disposition not in {"pass", "revise"}:
            raise ValueError("disposition must be pass or revise")
        for value, name in (
            (self.trace_content_sha256, "trace_content_sha256"),
            (self.video_file_sha256, "video_file_sha256"),
            (self.decoded_frames_sha256, "decoded_frames_sha256"),
        ):
            _require_sha256(value, name)
        if self.scenario_tape_sha256 is not None:
            _require_sha256(self.scenario_tape_sha256, "scenario_tape_sha256")
        if self.dashboard_evidence_sha256 is not None:
            _require_sha256(self.dashboard_evidence_sha256, "dashboard_evidence_sha256")
        _validate_size((self.frame_width, self.frame_height))
        if (
            not self.keyframe_indices
            or tuple(sorted(set(self.keyframe_indices))) != self.keyframe_indices
            or any(step < 0 for step in self.keyframe_indices)
        ):
            raise ValueError("keyframe_indices must be nonempty, sorted, unique, and nonnegative")
        if len(self.checks) != len(_REQUIRED_REVIEW_CHECKS):
            raise ValueError("visual review must contain every required check exactly once")
        check_names = tuple(check.name for check in self.checks)
        if set(check_names) != set(_REQUIRED_REVIEW_CHECKS) or len(set(check_names)) != len(
            check_names
        ):
            raise ValueError("visual review check names are missing or duplicated")
        for check in self.checks:
            check.validate()
        if self.disposition == "pass" and any(check.status != "pass" for check in self.checks):
            raise ValueError("a passing visual review cannot contain a failed check")
        if self.disposition == "revise" and all(check.status == "pass" for check in self.checks):
            raise ValueError("a revise disposition must identify at least one failed check")
        _validate_notes(self.notes, "notes", required=True)
        _validate_notes(self.revisions, "revisions", required=self.disposition == "revise")


def evidence_inventory(
    trace: ImmutableTrace,
    tape: ScenarioTape | None = None,
    events: Sequence[ArtifactEvent] | None = None,
    *,
    sidecar: DashboardEvidence | None = None,
) -> tuple[EvidenceItem, ...]:
    """Describe exactly which requested dashboard evidence is present and where it came from."""
    trace.validate()
    _validate_tape_binding(trace, tape)
    _validate_sidecar_binding(trace, tape, sidecar)
    validated_events = _validate_events(trace, events)
    has_position = _position_indices(trace) is not None
    dynamics_fields = _dynamics_state_indices(trace)
    has_bptt = (
        sidecar is not None
        and sidecar.bptt_timing_names.size > 0
        and np.any(sidecar.bptt_timing_available)
    ) or any("bptt" in str(name).lower() for name in trace.latency_names)
    has_admission_event = any(
        "admit" in event.name.lower() or "swap" in event.name.lower() for event in validated_events
    ) or (sidecar is not None and bool(sidecar.admission_recorded))
    has_nominal = sidecar is not None and np.any(sidecar.nominal_rollout_available)
    has_fallback = sidecar is not None and np.any(sidecar.fallback_rollout_available)
    has_selected = sidecar is not None and np.any(sidecar.selected_rollout_available)
    has_prediction = sidecar is not None and np.any(sidecar.prediction_available)
    has_descriptors = sidecar is not None and np.any(sidecar.descriptor_available)
    has_dynamics_truth = sidecar is not None and np.any(sidecar.dynamics_true_available)
    has_dynamics_estimate = sidecar is not None and np.any(sidecar.dynamics_estimated_available)
    has_dynamics_uncertainty = sidecar is not None and np.any(
        sidecar.dynamics_uncertainty_available
    )
    has_ghosts = sidecar is not None and np.any(sidecar.ghost_rollout_available)
    return (
        EvidenceItem(
            "actual_trajectory",
            "actual/estimated trajectory",
            "recorded" if has_position else "unavailable",
            "trace true_state/estimated_state position columns"
            if has_position
            else "no recognized position_x/y/z or x/y/z state names",
        ),
        EvidenceItem(
            "scenario_reference",
            "scenario defender reference",
            "scenario-recorded" if tape is not None else "unavailable",
            "ScenarioTape.defender_reference_position"
            if tape is not None
            else "no ScenarioTape supplied",
        ),
        EvidenceItem(
            "nominal_rollout",
            "controller nominal rollout trajectory",
            "recorded" if has_nominal else "unavailable",
            "DashboardEvidence.nominal_rollout_positions"
            if has_nominal
            else "ImmutableTrace stores nominal controls, not a nominal rollout trajectory",
        ),
        EvidenceItem(
            "fallback_rollouts",
            "fallback rollout trajectories",
            "recorded" if has_fallback else "unavailable",
            "DashboardEvidence.fallback_rollout_positions"
            if has_fallback
            else "version-1 ImmutableTrace stores values, not rollout states",
        ),
        EvidenceItem(
            "selected_rollout",
            "selected rollout trajectory",
            "recorded" if has_selected else "unavailable",
            "DashboardEvidence.selected_rollout_positions"
            if has_selected
            else "version-1 ImmutableTrace stores selection index, not rollout states",
        ),
        EvidenceItem(
            "prediction_tubes",
            "dynamic-obstacle prediction ensemble",
            "recorded"
            if has_prediction
            else ("scenario-recorded" if tape is not None else "unavailable"),
            "DashboardEvidence.prediction_positions"
            if has_prediction
            else (
                "ScenarioTape.prediction_positions"
                if tape is not None
                else "no recorded sidecar prediction or ScenarioTape supplied"
            ),
        ),
        EvidenceItem(
            "policy_values",
            "hard policy values and selection",
            "recorded",
            "trace policy_values/selected_policy",
        ),
        EvidenceItem(
            "descriptors",
            "normalized trajectory descriptors",
            "recorded" if has_descriptors else "unavailable",
            "DashboardEvidence.normalized_descriptors"
            if has_descriptors
            else "version-1 ImmutableTrace has no descriptor field",
        ),
        EvidenceItem(
            "dynamics_truth",
            "true dynamics schedule",
            "recorded"
            if has_dynamics_truth
            else ("scenario-recorded" if tape is not None else "unavailable"),
            "DashboardEvidence.dynamics_true"
            if has_dynamics_truth
            else (
                "ScenarioTape wind/mass/drag/rotor schedules"
                if tape is not None
                else "no recorded dynamics truth or ScenarioTape supplied"
            ),
        ),
        EvidenceItem(
            "dynamics_estimate",
            "estimated dynamics",
            "recorded" if has_dynamics_estimate or dynamics_fields else "unavailable",
            "DashboardEvidence.dynamics_estimated"
            if has_dynamics_estimate
            else (
                "recognized true_state/estimated_state dynamics columns"
                if dynamics_fields
                else "no recorded dynamics estimate"
            ),
        ),
        EvidenceItem(
            "dynamics_uncertainty",
            "dynamics uncertainty rollouts",
            "recorded" if has_dynamics_uncertainty else "unavailable",
            "DashboardEvidence.dynamics_uncertainty_samples"
            if has_dynamics_uncertainty
            else "neither version-1 trace nor ScenarioTape stores estimator uncertainty rollouts",
        ),
        EvidenceItem(
            "losses", "loss terms and gradient norm", "recorded", "trace loss_terms/gradient_norm"
        ),
        EvidenceItem(
            "candidate_admission",
            "candidate admission decision",
            "recorded" if has_admission_event else "unavailable",
            "DashboardEvidence admission arrays or recorded ArtifactEvent"
            if has_admission_event
            else "snapshot transitions show swaps, not rejected candidate decisions",
        ),
        EvidenceItem(
            "bptt_time",
            "BPTT component latency",
            "recorded" if has_bptt else "unavailable",
            "DashboardEvidence BPTT timing or trace latency column containing 'bptt'"
            if has_bptt
            else "no latency name contains 'bptt'",
        ),
        EvidenceItem(
            "actions_solver_latency",
            "controls, intervention, solver residuals, and latency",
            "recorded",
            "trace control, residual, and component-latency arrays",
        ),
        EvidenceItem(
            "ghost_rollouts",
            "pre-change/post-change/adapted ghost rollouts",
            "recorded" if has_ghosts else "unavailable",
            "DashboardEvidence.ghost_rollout_positions"
            if has_ghosts
            else "version-1 ImmutableTrace has no ghost-rollout field",
        ),
    )


def _recorded_schedule_change(sidecar: DashboardEvidence, label: str, step: int) -> bool:
    """Whether a tape schedule marker changed the controller's recorded true dynamics.

    Every condition uses the same fixed-shape tape schema, but static/obstacle-only experiments do
    not apply the tape's dynamics challenge schedule to their true plant.  Filtering annotations
    through the sidecar prevents those dormant tape fields from being drawn as executed events.
    """
    if step <= 0 or step >= sidecar.steps:
        return False
    prefixes = {
        "wind_step": ("wind_",),
        "mass_step": ("mass_",),
        "drag_step": ("drag_",),
        "rotor_symmetric_step": ("rotor_efficiency_",),
        "rotor_single_step": ("rotor_efficiency_",),
    }.get(label, ())
    names = tuple(str(name) for name in sidecar.dynamics_parameter_names)
    indices = tuple(
        index for index, name in enumerate(names) if any(name.startswith(item) for item in prefixes)
    )
    if not indices:
        return False
    available = sidecar.dynamics_true_available[[step - 1, step]][:, indices]
    if not np.all(available):
        return False
    previous = sidecar.dynamics_true[step - 1, indices]
    current = sidecar.dynamics_true[step, indices]
    return not np.allclose(previous, current, rtol=1e-7, atol=1e-9)


def change_annotations(
    trace: ImmutableTrace,
    tape: ScenarioTape | None = None,
    events: Sequence[ArtifactEvent] | None = None,
    *,
    sidecar: DashboardEvidence | None = None,
) -> tuple[ChangeAnnotation, ...]:
    """Return sorted annotations derived only from trace/tape/event transitions."""
    trace.validate()
    _validate_tape_binding(trace, tape)
    _validate_sidecar_binding(trace, tape, sidecar)
    validated_events = _validate_events(trace, events)
    annotations: list[ChangeAnnotation] = []
    if tape is not None:
        for name, step in zip(tape.schedule_names, tape.schedule_change_indices, strict=True):
            if sidecar is None or _recorded_schedule_change(sidecar, str(name), int(step)):
                annotations.append(ChangeAnnotation(int(step), str(name), "dynamics-change"))
    for step in np.flatnonzero(np.diff(trace.snapshot_version) != 0) + 1:
        annotations.append(ChangeAnnotation(int(step), "snapshot swap", "snapshot"))
    for step in np.flatnonzero(np.diff(trace.model_version) != 0) + 1:
        annotations.append(ChangeAnnotation(int(step), "model update", "model"))
    for field, label, kind in (
        (trace.degraded, "degraded onset", "degraded"),
        (trace.failure, "unsafe onset", "failure"),
        (trace.contact, "contact", "contact"),
    ):
        onset = np.flatnonzero(field & ~np.concatenate((np.asarray([False]), field[:-1])))
        annotations.extend(ChangeAnnotation(int(step), label, kind) for step in onset)
    for event in validated_events:
        if (
            event.severity != "info"
            or "admit" in event.name.lower()
            or "swap" in event.name.lower()
        ):
            annotations.append(ChangeAnnotation(event.step, event.name, f"event-{event.severity}"))
    if sidecar is not None and bool(sidecar.admission_recorded):
        for step in np.flatnonzero(sidecar.candidate_admitted):
            annotations.append(ChangeAnnotation(int(step), "candidate admitted", "admission"))
        for step in np.flatnonzero(sidecar.candidate_rejected):
            reason = int(sidecar.admission_reason_index[step])
            label = "candidate rejected"
            if reason >= 0:
                label += f": {sidecar.admission_reason_names[reason]}"
            annotations.append(ChangeAnnotation(int(step), label, "rejection"))
    unique = {(item.step, item.label, item.kind): item for item in annotations}
    return tuple(sorted(unique.values(), key=lambda item: (item.step, item.kind, item.label)))


def scientific_dashboard_frames(
    trace: ImmutableTrace,
    *,
    tape: ScenarioTape | None = None,
    sidecar: DashboardEvidence | None = None,
    events: Sequence[ArtifactEvent] | None = None,
    size: tuple[int, int] = (1600, 900),
) -> Iterator[np.ndarray]:
    """Yield synchronized, labelled RGB dashboard frames without running the simulator."""
    if not isinstance(trace, ImmutableTrace):
        raise TypeError("trace must be an ImmutableTrace")
    trace.validate()
    width, height = _validate_size(size)
    _validate_tape_binding(trace, tape)
    _validate_sidecar_binding(trace, tape, sidecar)
    validated_events = _validate_events(trace, events)
    inventory = evidence_inventory(trace, tape, validated_events, sidecar=sidecar)
    annotations = change_annotations(trace, tape, validated_events, sidecar=sidecar)
    context = _RenderContext(trace, tape, sidecar, validated_events, inventory, annotations)
    for step in range(trace.steps):
        yield _render_frame(context, step, width, height)


def render_scientific_dashboard(
    trace: ImmutableTrace,
    path: str | os.PathLike[str],
    *,
    tape: ScenarioTape | None = None,
    sidecar: DashboardEvidence | None = None,
    events: Sequence[ArtifactEvent] | None = None,
    fps: float = 15.0,
    size: tuple[int, int] = (1600, 900),
) -> ScientificDashboardResult:
    """Atomically encode a 720p-or-larger scientific dashboard as offline H.264."""
    if not isinstance(trace, ImmutableTrace):
        raise TypeError("trace must be an ImmutableTrace")
    frame_rate = _positive_finite(fps, "fps")
    width, height = _validate_size(size)
    destination = Path(path)
    if destination.suffix.lower() != ".mp4":
        raise ValueError("scientific dashboard path must end in .mp4")
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    encoder = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    if not encoder.is_file():
        raise RuntimeError("pinned imageio-ffmpeg executable is unavailable")
    temporary = destination.parent / f".{destination.name}.encoding.tmp.mp4"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    writer = None
    try:
        writer = imageio_ffmpeg.write_frames(
            str(temporary),
            (width, height),
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=frame_rate,
            quality=None,
            bitrate=None,
            codec="libx264",
            macro_block_size=2,
            ffmpeg_log_level="error",
            ffmpeg_timeout=60,
            output_params=[
                "-preset",
                "medium",
                "-crf",
                "18",
                "-threads",
                "1",
                "-movflags",
                "+faststart",
                "-metadata",
                "creation_time=1970-01-01T00:00:00Z",
            ],
        )
        writer.send(None)
        for frame in scientific_dashboard_frames(
            trace, tape=tape, sidecar=sidecar, events=events, size=(width, height)
        ):
            writer.send(frame)
        writer.close()
        writer = None
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
    validation = validate_mp4(
        destination,
        expected_codec="h264",
        expected_size=(width, height),
        expected_fps=frame_rate,
        expected_frame_count=trace.steps,
    )
    return ScientificDashboardResult(
        validation=validation,
        evidence=evidence_inventory(trace, tape, events, sidecar=sidecar),
        encoder_executable=str(encoder),
    )


def verify_scientific_dashboard_replay(
    trace: ImmutableTrace,
    reference_video: str | os.PathLike[str],
    *,
    tape: ScenarioTape | None = None,
    sidecar: DashboardEvidence | None = None,
    events: Sequence[ArtifactEvent] | None = None,
    fps: float,
    size: tuple[int, int],
) -> ScientificDashboardResult:
    """Re-render offline and require byte-identical decoded RGB frames."""
    width, height = _validate_size(size)
    frame_rate = _positive_finite(fps, "fps")
    reference = validate_mp4(
        reference_video,
        expected_codec="h264",
        expected_size=(width, height),
        expected_fps=frame_rate,
        expected_frame_count=trace.steps,
    )
    with tempfile.TemporaryDirectory(prefix="crazyflow-da-plcbf-scientific-replay-") as directory:
        replay = render_scientific_dashboard(
            trace,
            Path(directory) / "replay.mp4",
            tape=tape,
            sidecar=sidecar,
            events=events,
            fps=frame_rate,
            size=(width, height),
        )
    if replay.validation.decoded_frames_sha256 != reference.decoded_frames_sha256:
        raise ValueError("scientific dashboard decoded frames are not deterministic")
    return replay


def select_keyframe_indices(
    trace: ImmutableTrace,
    *,
    tape: ScenarioTape | None = None,
    sidecar: DashboardEvidence | None = None,
    count: int = 8,
) -> tuple[int, ...]:
    """Select deterministic review frames, prioritizing safety and change events."""
    if not isinstance(trace, ImmutableTrace):
        raise TypeError("trace must be an ImmutableTrace")
    trace.validate()
    _validate_tape_binding(trace, tape)
    _validate_sidecar_binding(trace, tape, sidecar)
    if isinstance(count, bool) or not isinstance(count, Integral):
        raise TypeError("count must be an integer")
    requested = int(count)
    if requested < 2 or requested > trace.steps:
        raise ValueError("count must lie between two and the trace length")
    margin = np.min(trace.hard_barriers, axis=1)
    intervention = np.linalg.norm(trace.filtered_control - trace.nominal_control, axis=1)
    executed_intervention = np.where(trace.executed_control, intervention, -np.inf)
    priority = [0, trace.steps - 1, int(np.argmin(margin)), int(np.argmax(executed_intervention))]
    for field in (trace.failure, trace.degraded, trace.contact):
        priority.extend(
            int(step)
            for step in np.flatnonzero(field & ~np.concatenate((np.asarray([False]), field[:-1])))
        )
    priority.extend(int(step) for step in np.flatnonzero(np.diff(trace.snapshot_version) != 0) + 1)
    priority.extend(int(step) for step in np.flatnonzero(np.diff(trace.model_version) != 0) + 1)
    if tape is not None:
        priority.extend(
            item.step
            for item in change_annotations(trace, tape, sidecar=sidecar)
            if item.kind == "dynamics-change"
        )
    if sidecar is not None and bool(sidecar.admission_recorded):
        priority.extend(int(step) for step in np.flatnonzero(sidecar.candidate_admitted))
        priority.extend(int(step) for step in np.flatnonzero(sidecar.candidate_rejected))
    priority.extend(np.linspace(0, trace.steps - 1, requested, dtype=int).tolist())
    selected: list[int] = []
    for step in priority:
        if 0 <= step < trace.steps and step not in selected:
            selected.append(step)
        if len(selected) == requested:
            break
    return tuple(sorted(selected))


def extract_keyframes(
    video_path: str | os.PathLike[str],
    trace: ImmutableTrace,
    destination: str | os.PathLike[str],
    *,
    tape: ScenarioTape | None = None,
    sidecar: DashboardEvidence | None = None,
    count: int = 8,
) -> tuple[KeyframeRecord, ...]:
    """Decode selected video frames and save full-resolution deterministic PNGs."""
    source = Path(video_path)
    validation = validate_mp4(source, expected_frame_count=trace.steps)
    indices = select_keyframe_indices(trace, tape=tape, sidecar=sidecar, count=count)
    output = Path(destination)
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir()
    wanted = set(indices)
    records: list[KeyframeRecord] = []
    reader = imageio_ffmpeg.read_frames(str(source), pix_fmt="rgb24", bits_per_pixel=24)
    try:
        metadata = next(reader)
        raw_size = metadata.get("size") if isinstance(metadata, Mapping) else None
        if raw_size != (validation.width, validation.height):
            raise ValueError("decoded keyframe dimensions disagree with validated video")
        for step, raw in enumerate(reader):
            if step not in wanted:
                continue
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                validation.height, validation.width, 3
            )
            frame_path = output / f"keyframe-{step:06d}.png"
            iio.imwrite(frame_path, frame, extension=".png", compress_level=9)
            records.append(
                KeyframeRecord(
                    step=step,
                    time_seconds=float(trace.time[step]),
                    path=str(frame_path.resolve()),
                    width=validation.width,
                    height=validation.height,
                    sha256=file_sha256(frame_path),
                )
            )
    finally:
        reader.close()
    if tuple(record.step for record in records) != indices:
        raise ValueError("encoded video did not contain every selected keyframe")
    return tuple(records)


def render_contact_sheet(
    keyframes: Sequence[KeyframeRecord],
    path: str | os.PathLike[str],
    *,
    title: str = "DA-PLCBF scientific dashboard review",
    columns: int = 4,
) -> ContactSheetRecord:
    """Render a deterministic labelled PNG contact sheet from extracted keyframes."""
    if isinstance(keyframes, (str, bytes)) or not isinstance(keyframes, Sequence) or not keyframes:
        raise ValueError("keyframes must be a nonempty sequence")
    if isinstance(columns, bool) or not isinstance(columns, Integral) or int(columns) < 1:
        raise ValueError("columns must be a positive integer")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be nonempty")
    records = tuple(keyframes)
    if tuple(sorted(record.step for record in records)) != tuple(record.step for record in records):
        raise ValueError("keyframes must be sorted by step")
    frames = []
    dimensions = set()
    for record in records:
        source = Path(record.path)
        if file_sha256(source) != record.sha256:
            raise ValueError("keyframe digest mismatch")
        frame = np.asarray(iio.imread(source), dtype=np.uint8)
        if frame.shape != (record.height, record.width, 3):
            raise ValueError("keyframe dimensions disagree with its record")
        dimensions.add((record.width, record.height))
        frames.append(frame)
    if len(dimensions) != 1:
        raise ValueError("all keyframes must have identical dimensions")
    destination = Path(path)
    if destination.suffix.lower() != ".png":
        raise ValueError("contact sheet path must end in .png")
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    column_count = min(int(columns), len(records))
    row_count = math.ceil(len(records) / column_count)
    width = column_count * 400
    height = 54 + row_count * 250
    figure = Figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor=_BACKGROUND)
    canvas = FigureCanvasAgg(figure)
    figure.text(0.015, 0.975, title, color=_TEXT, fontsize=13, weight="bold", va="top")
    grid = figure.add_gridspec(
        row_count,
        column_count,
        left=0.015,
        right=0.985,
        bottom=0.02,
        top=0.91,
        wspace=0.04,
        hspace=0.20,
    )
    for index, (record, frame) in enumerate(zip(records, frames, strict=True)):
        axis = figure.add_subplot(grid[index // column_count, index % column_count])
        axis.imshow(frame)
        axis.set_title(
            f"step {record.step}  •  t={record.time_seconds:.3f} s", color=_TEXT, fontsize=8, pad=3
        )
        axis.set_axis_off()
    for index in range(len(records), row_count * column_count):
        axis = figure.add_subplot(grid[index // column_count, index % column_count])
        axis.set_axis_off()
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    iio.imwrite(destination, image, extension=".png", compress_level=9)
    return ContactSheetRecord(
        path=str(destination.resolve()),
        width=width,
        height=height,
        keyframe_steps=tuple(record.step for record in records),
        sha256=file_sha256(destination),
    )


def write_visual_review_record(record: VisualReviewRecord, path: str | os.PathLike[str]) -> str:
    """Write a strict, machine-readable Markdown visual-review record and return its digest."""
    if not isinstance(record, VisualReviewRecord):
        raise TypeError("record must be a VisualReviewRecord")
    record.validate()
    destination = Path(path)
    if destination.suffix.lower() != ".md":
        raise ValueError("visual review path must end in .md")
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    payload = _review_mapping(record)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    checks = "\n".join(
        f"- [{'x' if check.status == 'pass' else ' '}] `{check.name}` — {check.note}"
        for check in record.checks
    )
    text = (
        "# DA-PLCBF visual review\n\n"
        f"Disposition: **{record.disposition}**  \n"
        f"Reviewer: {record.reviewer} ({record.reviewer_kind})  \n"
        f"Reviewed: {record.reviewed_utc}\n\n"
        "## Inspection checks\n\n"
        f"{checks}\n\n"
        "## Structured record\n\n"
        "```json\n"
        f"{encoded}\n"
        "```\n"
    )
    destination.write_text(text, encoding="utf-8")
    return file_sha256(destination)


def load_visual_review_record(path: str | os.PathLike[str]) -> VisualReviewRecord:
    """Load and revalidate the canonical JSON block from a visual-review Markdown file."""
    source = Path(path)
    if source.is_symlink() or not source.is_file() or source.suffix.lower() != ".md":
        raise ValueError("visual review must be a regular Markdown file")
    text = source.read_text(encoding="utf-8")
    start_token = "```json\n"
    end_token = "\n```\n"
    if text.count(start_token) != 1:
        raise ValueError("visual review must contain exactly one structured JSON block")
    start = text.index(start_token) + len(start_token)
    end = text.find(end_token, start)
    if end < 0 or text.find(end_token, end + len(end_token)) >= 0:
        raise ValueError("visual review has malformed structured JSON fencing")
    try:
        payload = json.loads(text[start:end])
    except json.JSONDecodeError as error:
        raise ValueError("visual review structured record is invalid JSON") from error
    record = _review_from_mapping(payload)
    record.validate()
    canonical = json.dumps(
        _review_mapping(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    if text[start:end] != canonical:
        raise ValueError("visual review structured record is not canonical")
    return record


class _RenderContext:
    """Precomputed immutable rendering inputs."""

    def __init__(
        self,
        trace: ImmutableTrace,
        tape: ScenarioTape | None,
        sidecar: DashboardEvidence | None,
        events: tuple[ArtifactEvent, ...],
        evidence: tuple[EvidenceItem, ...],
        annotations: tuple[ChangeAnnotation, ...],
    ) -> None:
        self.trace = trace
        self.tape = tape
        self.sidecar = sidecar
        self.events = events
        self.evidence = evidence
        self.annotations = annotations
        self.position_indices = _position_indices(trace)
        self.dynamics_indices = _dynamics_state_indices(trace)
        self.minimum_margin = np.min(trace.hard_barriers, axis=1)
        self.policy_values = _mask_nonexecuted_rows(trace, trace.policy_values)
        self.loss_terms = _mask_nonexecuted_rows(trace, trace.loss_terms)
        self.gradient_norm = _mask_nonexecuted_rows(trace, trace.gradient_norm)
        self.nominal_control = _mask_nonexecuted_rows(trace, trace.nominal_control)
        self.filtered_control = _mask_nonexecuted_rows(trace, trace.filtered_control)
        self.applied_control = _mask_nonexecuted_rows(trace, trace.applied_control)
        self.component_latency = _mask_nonexecuted_rows(trace, trace.component_latency_seconds)
        self.solver_kkt_residual = _mask_nonexecuted_rows(trace, trace.solver_kkt_residual)
        self.postcheck_residual = _mask_nonexecuted_rows(trace, trace.postcheck_residual)
        self.intervention = np.linalg.norm(self.filtered_control - self.nominal_control, axis=1)
        self.safe_policy_count = _safe_policy_counts(trace)


def _render_frame(context: _RenderContext, step: int, width: int, height: int) -> np.ndarray:
    trace = context.trace
    figure = Figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor=_BACKGROUND)
    canvas = FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        3, 4, left=0.035, right=0.985, bottom=0.055, top=0.885, wspace=0.31, hspace=0.46
    )
    world = figure.add_subplot(grid[0, 0:2])
    policy = figure.add_subplot(grid[0, 2])
    margin = figure.add_subplot(grid[0, 3])
    dynamics = figure.add_subplot(grid[1, 0])
    learning = figure.add_subplot(grid[1, 1:3])
    controls = figure.add_subplot(grid[1, 3])
    latency = figure.add_subplot(grid[2, 0:2])
    descriptor = figure.add_subplot(grid[2, 2])
    evidence = figure.add_subplot(grid[2, 3])
    axes = (world, policy, margin, dynamics, learning, controls, latency, descriptor, evidence)
    for axis in axes:
        _style_axis(axis)

    status, status_color = _status(trace, step)
    control_label = (
        f"policy={int(trace.selected_policy[step])}"
        if bool(trace.executed_control[step])
        else "TERMINAL OBSERVATION • NO CONTROL"
    )
    figure.text(
        0.035,
        0.975,
        "DA-PLCBF • finite-horizon scientific replay",
        color=_TEXT,
        fontsize=15,
        weight="bold",
        va="top",
    )
    figure.text(
        0.985,
        0.975,
        (
            f"t={trace.time[step]:.3f} s  |  step {step + 1}/{trace.steps}  |  "
            f"{control_label}  |  "
            f"margin={context.minimum_margin[step]:+.3e}  |  "
            f"snapshot={int(trace.snapshot_version[step])}  model={int(trace.model_version[step])}"
        ),
        color=_TEXT,
        fontsize=9,
        ha="right",
        va="top",
    )
    figure.add_artist(
        Rectangle(
            (0.0, 0.921),
            1.0,
            0.018,
            transform=figure.transFigure,
            facecolor=status_color,
            edgecolor="none",
        )
    )
    figure.text(
        0.5, 0.930, status, color=_BACKGROUND, fontsize=8, weight="bold", ha="center", va="center"
    )

    _plot_world(world, context, step, status_color)
    _plot_policy(policy, context, step)
    _plot_margin(margin, context, step)
    _plot_dynamics(dynamics, context, step)
    _plot_learning(learning, context, step)
    _plot_controls(controls, context, step)
    _plot_latency(latency, context, step)
    _plot_descriptor(descriptor, context, step)
    _plot_evidence(evidence, context, step)
    for rendered_axis in figure.axes:
        for title in (rendered_axis.title, rendered_axis._left_title, rendered_axis._right_title):
            title.set_color(_TEXT)
            title.set_fontsize(8.2)
    figure.text(
        0.5,
        0.018,
        "Numeric trace is authoritative • missing evidence is labelled, never inferred • "
        "safety claim is finite-horizon only",
        color=_MUTED,
        fontsize=7.5,
        ha="center",
    )
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    if frame.shape != (height, width, 3):
        raise RuntimeError("Agg canvas produced unexpected dashboard dimensions")
    return frame


def _plot_world(axis: Any, context: _RenderContext, step: int, status_color: str) -> None:
    trace, tape = context.trace, context.tape
    axis.set_title("Isometric world • recorded geometry", loc="left")
    axis.set_xlabel("projected horizontal axis")
    axis.set_ylabel("projected height axis")
    if context.position_indices is None:
        _unavailable(axis, "ACTUAL TRAJECTORY UNAVAILABLE\nno recognized position columns")
        return
    indices = context.position_indices
    actual = trace.true_state[:, indices]
    estimated = trace.estimated_state[:, indices]
    actual_uv = _isometric(actual)
    estimated_uv = _isometric(estimated)
    world_points = [actual_uv, estimated_uv]
    if tape is not None:
        reference_uv = _isometric(tape.defender_reference_position)
        world_points.append(reference_uv)
        axis.plot(
            reference_uv[:, 0],
            reference_uv[:, 1],
            color=_MUTED,
            linestyle=":",
            linewidth=1.0,
            label="scenario reference (not nominal rollout)",
        )
        sidecar_prediction = context.sidecar is not None and np.any(
            context.sidecar.prediction_available[step]
        )
        _plot_scenario_geometry(
            axis, tape, step, world_points, plot_predictions=not sidecar_prediction
        )
    if context.sidecar is not None:
        _plot_sidecar_world(axis, context.sidecar, step, world_points)
    points = np.concatenate(world_points, axis=0)
    span = np.maximum(np.ptp(points, axis=0), 0.5)
    center = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
    axis.set_xlim(center[0] - 0.62 * span[0], center[0] + 0.62 * span[0])
    axis.set_ylim(center[1] - 0.62 * span[1], center[1] + 0.62 * span[1])
    axis.plot(
        estimated_uv[: step + 1, 0],
        estimated_uv[: step + 1, 1],
        color=_ORANGE,
        linestyle="--",
        linewidth=1.2,
        label="estimated state",
    )
    axis.plot(
        actual_uv[: step + 1, 0],
        actual_uv[: step + 1, 1],
        color=_CYAN,
        linewidth=2.2,
        label="actual state",
    )
    unsafe = np.flatnonzero(trace.failure[: step + 1])
    degraded = np.flatnonzero(trace.degraded[: step + 1] & ~trace.failure[: step + 1])
    if degraded.size:
        axis.scatter(
            actual_uv[degraded, 0], actual_uv[degraded, 1], s=24, c=_ORANGE, marker="s", zorder=8
        )
    if unsafe.size:
        axis.scatter(actual_uv[unsafe, 0], actual_uv[unsafe, 1], s=34, c=_RED, marker="x", zorder=9)
    axis.scatter(
        actual_uv[step, 0],
        actual_uv[step, 1],
        s=65,
        facecolors=status_color,
        edgecolors=_TEXT,
        linewidths=1.0,
        zorder=10,
    )
    missing = []
    sidecar = context.sidecar
    if sidecar is None or not sidecar.nominal_rollout_available[step]:
        missing.append("nominal")
    if sidecar is None or not np.any(sidecar.fallback_rollout_available[step]):
        missing.append("fallback")
    if sidecar is None or not sidecar.selected_rollout_available[step]:
        missing.append("selected")
    if sidecar is None or not np.any(sidecar.ghost_rollout_available[step]):
        missing.append("ghost")
    if missing:
        axis.text(
            0.01,
            0.01,
            f"rollouts unavailable now: {', '.join(missing)}",
            transform=axis.transAxes,
            color=_YELLOW,
            fontsize=6.5,
            va="bottom",
        )
    axis.legend(
        loc="upper right",
        fontsize=5.2,
        frameon=False,
        labelcolor=_TEXT,
        ncol=4,
        columnspacing=0.7,
        handlelength=1.5,
        borderaxespad=0.2,
    )


def _plot_scenario_geometry(
    axis: Any,
    tape: ScenarioTape,
    step: int,
    world_points: list[np.ndarray],
    *,
    plot_predictions: bool,
) -> None:
    static = tape.static_positions[tape.static_mask]
    if len(static):
        projected = _isometric(static)
        world_points.append(projected)
        axis.scatter(
            projected[:, 0],
            projected[:, 1],
            s=26,
            facecolors="none",
            edgecolors=_PURPLE,
            linewidths=1.0,
            label="static obstacle centers",
        )
    dynamic_mask = tape.dynamic_slot_mask & tape.dynamic_time_mask[step]
    dynamic = tape.dynamic_positions[step, dynamic_mask]
    if len(dynamic):
        projected = _isometric(dynamic)
        world_points.append(projected)
        axis.scatter(
            projected[:, 0],
            projected[:, 1],
            s=30,
            c=_RED,
            marker="D",
            label="dynamic truth",
            zorder=7,
        )
    if plot_predictions:
        for slot in np.flatnonzero(dynamic_mask):
            active_future = tape.dynamic_time_mask[step:, slot]
            for sample in range(tape.prediction_samples):
                prediction = tape.prediction_positions[sample, step:, slot][active_future]
                if len(prediction) < 2:
                    continue
                projected = _isometric(prediction)
                world_points.append(projected)
                axis.plot(projected[:, 0], projected[:, 1], color=_RED, alpha=0.08, linewidth=0.55)
    axis.text(
        0.01,
        0.07,
        "obstacle markers show centers; hard-margin panel carries clearance",
        transform=axis.transAxes,
        color=_MUTED,
        fontsize=6.0,
        va="bottom",
    )


def _plot_sidecar_world(
    axis: Any, sidecar: DashboardEvidence, step: int, world_points: list[np.ndarray]
) -> None:
    if sidecar.nominal_rollout_available[step]:
        projected = _isometric(sidecar.nominal_rollout_positions[step])
        world_points.append(projected)
        axis.plot(
            projected[:, 0],
            projected[:, 1],
            color=_YELLOW,
            linestyle=":",
            linewidth=1.5,
            label="recorded held nominal preview",
        )
    first_fallback = True
    for policy in np.flatnonzero(sidecar.fallback_rollout_available[step]):
        projected = _isometric(sidecar.fallback_rollout_positions[step, policy])
        world_points.append(projected)
        axis.plot(
            projected[:, 0],
            projected[:, 1],
            color=_PURPLE,
            alpha=0.20,
            linewidth=0.65,
            label="recorded fallback library" if first_fallback else None,
        )
        first_fallback = False
    if sidecar.selected_rollout_available[step]:
        projected = _isometric(sidecar.selected_rollout_positions[step])
        world_points.append(projected)
        axis.plot(
            projected[:, 0],
            projected[:, 1],
            color=_GREEN,
            linewidth=2.2,
            label="recorded selected rollout",
        )
    ghost_colors = (_MUTED, _ORANGE, _CYAN, _PURPLE)
    for ghost in np.flatnonzero(sidecar.ghost_rollout_available[step]):
        projected = _isometric(sidecar.ghost_rollout_positions[step, ghost])
        world_points.append(projected)
        axis.plot(
            projected[:, 0],
            projected[:, 1],
            color=ghost_colors[ghost % len(ghost_colors)],
            linestyle="--",
            linewidth=1.0,
            label=f"ghost: {sidecar.ghost_rollout_names[ghost]}",
        )
    labelled_prediction = False
    prediction = sidecar.prediction_positions[step]
    available = sidecar.prediction_available[step]
    for sample in range(prediction.shape[0]):
        for obstacle in range(prediction.shape[1]):
            mask = available[sample, obstacle]
            if not np.any(mask):
                continue
            projected = _isometric(prediction[sample, obstacle])
            world_points.append(projected[mask])
            projected = np.array(projected, copy=True)
            projected[~mask] = np.nan
            axis.plot(
                projected[:, 0],
                projected[:, 1],
                color=_RED,
                alpha=0.14,
                linewidth=0.7,
                label="recorded prediction tube" if not labelled_prediction else None,
            )
            labelled_prediction = True


def _plot_policy(axis: Any, context: _RenderContext, step: int) -> None:
    trace = context.trace
    axis.set_title("Hard policy values + selection", loc="left")
    finite = np.abs(context.policy_values[np.isfinite(context.policy_values)])
    extent = max(float(np.max(finite)), 1e-12)
    axis.imshow(
        np.ma.masked_invalid(context.policy_values.T),
        aspect="auto",
        origin="lower",
        extent=(trace.time[0], trace.time[-1], -0.5, len(trace.policy_names) - 0.5),
        cmap="RdYlGn",
        vmin=-extent,
        vmax=extent,
        interpolation="nearest",
    )
    selected = trace.selected_policy[: step + 1].astype(float)
    selected[selected < 0] = np.nan
    axis.plot(trace.time[: step + 1], selected, color="white", linewidth=1.0)
    axis.axvline(trace.time[step], color=_TEXT, linewidth=0.8)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("policy index")
    axis.text(
        0.99,
        0.02,
        "green ≥ 0 hard-certified",
        transform=axis.transAxes,
        ha="right",
        color=_TEXT,
        fontsize=6.2,
    )
    if not bool(trace.executed_control[step]):
        axis.text(
            0.5,
            0.5,
            "TERMINAL OBSERVATION\nNO CONTROL OR POLICY EVALUATION",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color=_YELLOW,
            fontsize=7,
            weight="bold",
            bbox={"facecolor": _PANEL, "edgecolor": _GRID, "alpha": 0.92, "pad": 4},
        )


def _plot_margin(axis: Any, context: _RenderContext, step: int) -> None:
    trace = context.trace
    axis.set_title("Hard barriers • status is not surrogate", loc="left")
    _shade_status(axis, trace)
    for index, name in enumerate(trace.barrier_names):
        axis.plot(
            trace.time[: step + 1],
            trace.hard_barriers[: step + 1, index],
            linewidth=1.0,
            label=str(name),
        )
    axis.plot(
        trace.time[: step + 1],
        context.minimum_margin[: step + 1],
        color=_TEXT,
        linewidth=1.8,
        label="minimum",
    )
    axis.axhline(0.0, color=_RED, linewidth=1.0)
    _time_cursor(axis, trace, step)
    _annotate_changes(axis, context, step)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("dimensionless hard value")
    axis.legend(fontsize=5.5, frameon=False, labelcolor=_TEXT, ncol=2)


def _plot_dynamics(axis: Any, context: _RenderContext, step: int) -> None:
    trace, tape, sidecar = context.trace, context.tape, context.sidecar
    axis.set_title("Dynamics truth / estimate / uncertainty", loc="left")
    plotted = False
    has_sidecar_truth = sidecar is not None and np.any(sidecar.dynamics_true_available)
    has_sidecar_estimate = sidecar is not None and np.any(sidecar.dynamics_estimated_available)
    has_sidecar_uncertainty = sidecar is not None and np.any(sidecar.dynamics_uncertainty_available)
    if tape is not None and not has_sidecar_truth:
        denominator = max(float(tape.wind_speed_limit), 1e-12)
        series = (
            (tape.mass_scale, "mass scale", _CYAN),
            (tape.drag_scale, "drag scale", _PURPLE),
            (np.min(tape.rotor_efficiency, axis=1), "min rotor efficiency", _ORANGE),
            (np.linalg.norm(tape.wind_velocity, axis=1) / denominator, "wind / limit", _GREEN),
        )
        for values, label, color in series:
            axis.plot(
                trace.time[: step + 1], values[: step + 1], label=f"true {label}", color=color
            )
        plotted = True
    if sidecar is not None and sidecar.dynamics_parameter_names.size:
        colors = (_CYAN, _PURPLE, _ORANGE, _GREEN, _YELLOW, _MUTED)
        for parameter, name in enumerate(sidecar.dynamics_parameter_names):
            color = colors[parameter % len(colors)]
            truth_available = np.any(sidecar.dynamics_true_available[: step + 1, parameter])
            if truth_available:
                values = np.where(
                    sidecar.dynamics_true_available[: step + 1, parameter],
                    sidecar.dynamics_true[: step + 1, parameter],
                    np.nan,
                )
                axis.plot(
                    trace.time[: step + 1], values, color=color, linewidth=1.1, label=str(name)
                )
                plotted = True
            if np.any(sidecar.dynamics_estimated_available[: step + 1, parameter]):
                values = np.where(
                    sidecar.dynamics_estimated_available[: step + 1, parameter],
                    sidecar.dynamics_estimated[: step + 1, parameter],
                    np.nan,
                )
                axis.plot(
                    trace.time[: step + 1],
                    values,
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    label=None if truth_available else str(name),
                )
                plotted = True
            if has_sidecar_uncertainty:
                lower = np.full(step + 1, np.nan)
                upper = np.full(step + 1, np.nan)
                for time_index in range(step + 1):
                    mask = sidecar.dynamics_uncertainty_available[time_index]
                    if np.any(mask):
                        values = sidecar.dynamics_uncertainty_samples[time_index, mask, parameter]
                        lower[time_index] = np.min(values)
                        upper[time_index] = np.max(values)
                if np.any(np.isfinite(lower)):
                    axis.fill_between(
                        trace.time[: step + 1], lower, upper, color=color, alpha=0.10, linewidth=0.0
                    )
    for index, name in context.dynamics_indices:
        axis.plot(
            trace.time[: step + 1],
            trace.true_state[: step + 1, index],
            color=_CYAN,
            linewidth=1.0,
            label=f"trace true {name}",
        )
        axis.plot(
            trace.time[: step + 1],
            trace.estimated_state[: step + 1, index],
            color=_ORANGE,
            linestyle="--",
            linewidth=1.0,
            label=f"estimate {name}",
        )
        plotted = True
    _time_cursor(axis, trace, step)
    _annotate_changes(axis, context, step, kinds={"dynamics-change", "model"})
    axis.set_xlabel("time [s]")
    axis.set_ylabel("recorded value / declared ratio")
    if plotted:
        legend = axis.legend(
            fontsize=5.2,
            frameon=False,
            labelcolor=_TEXT,
            ncol=2,
            loc="upper left",
            title="solid=true • dashed=estimate • band=uncertainty"
            if has_sidecar_uncertainty
            else None,
            title_fontsize=5.0,
        )
        if legend.get_title() is not None:
            legend.get_title().set_color(_TEXT)
    else:
        _unavailable(axis, "DYNAMICS TRUTH + ESTIMATE UNAVAILABLE")
    if not has_sidecar_estimate and not context.dynamics_indices:
        axis.text(
            0.01,
            0.03,
            "online estimate: UNAVAILABLE",
            transform=axis.transAxes,
            color=_YELLOW,
            fontsize=6.3,
        )
    if not has_sidecar_uncertainty:
        axis.text(
            0.99,
            0.03,
            "dynamics uncertainty rollout: UNAVAILABLE",
            transform=axis.transAxes,
            color=_YELLOW,
            fontsize=6.3,
            ha="right",
        )


def _plot_learning(axis: Any, context: _RenderContext, step: int) -> None:
    trace = context.trace
    axis.set_title("Learning • loss / coverage / gradient / admission", loc="left")
    for index, name in enumerate(trace.loss_term_names):
        axis.plot(
            trace.time[: step + 1],
            context.loss_terms[: step + 1, index],
            linewidth=0.9,
            label=str(name),
        )
    axis.plot(
        trace.time[: step + 1],
        context.gradient_norm[: step + 1],
        color=_TEXT,
        linestyle=":",
        linewidth=1.2,
        label="gradient norm",
    )
    safe_fraction = context.safe_policy_count / len(trace.policy_names)
    axis.plot(
        trace.time[: step + 1],
        safe_fraction[: step + 1],
        color=_GREEN,
        linewidth=1.4,
        label="safe-policy fraction",
    )
    sidecar = context.sidecar
    use_sidecar_bptt = (
        sidecar is not None
        and sidecar.bptt_timing_names.size > 0
        and np.any(sidecar.bptt_timing_available)
    )
    bptt_indices = (
        []
        if use_sidecar_bptt
        else [
            index for index, name in enumerate(trace.latency_names) if "bptt" in str(name).lower()
        ]
    )
    bptt_axis = None
    bptt_lines = []
    for index in bptt_indices:
        if bptt_axis is None:
            bptt_axis = axis.twinx()
            bptt_axis.set_facecolor("none")
            bptt_axis.tick_params(colors=_ORANGE, labelsize=6)
            bptt_axis.spines["right"].set_color(_GRID)
            bptt_axis.set_ylabel("BPTT latency [ms]", color=_ORANGE, fontsize=7)
        (line,) = bptt_axis.plot(
            trace.time[: step + 1],
            1000.0 * context.component_latency[: step + 1, index],
            color=_ORANGE,
            linestyle="--",
            linewidth=0.9,
            label=f"{trace.latency_names[index]} [ms]",
        )
        bptt_lines.append(line)
    if use_sidecar_bptt and sidecar is not None:
        for index, name in enumerate(sidecar.bptt_timing_names):
            if not np.any(sidecar.bptt_timing_available[: step + 1, index]):
                continue
            if bptt_axis is None:
                bptt_axis = axis.twinx()
                bptt_axis.set_facecolor("none")
                bptt_axis.tick_params(colors=_ORANGE, labelsize=6)
                bptt_axis.spines["right"].set_color(_GRID)
                bptt_axis.set_ylabel("BPTT latency [ms]", color=_ORANGE, fontsize=7)
            values = np.where(
                sidecar.bptt_timing_available[: step + 1, index],
                1000.0 * sidecar.bptt_timing_seconds[: step + 1, index],
                np.nan,
            )
            (line,) = bptt_axis.plot(
                trace.time[: step + 1],
                values,
                linestyle="--",
                linewidth=0.9,
                label=f"BPTT {name} [ms]",
            )
            bptt_lines.append(line)
    has_sidecar_admission = sidecar is not None and bool(sidecar.admission_recorded)
    if has_sidecar_admission and sidecar is not None:
        candidate_values = np.where(
            sidecar.candidate_present[: step + 1], sidecar.admission_margin[: step + 1], np.nan
        )
        axis.plot(
            trace.time[: step + 1],
            candidate_values,
            color=_PURPLE,
            linewidth=0.8,
            label="candidate admission margin",
        )
        admitted = np.flatnonzero(sidecar.candidate_admitted[: step + 1])
        rejected = np.flatnonzero(sidecar.candidate_rejected[: step + 1])
        if admitted.size:
            axis.scatter(
                trace.time[admitted],
                sidecar.admission_margin[admitted],
                color=_GREEN,
                marker="^",
                s=25,
                label="admitted",
                zorder=8,
            )
        if rejected.size:
            axis.scatter(
                trace.time[rejected],
                sidecar.admission_margin[rejected],
                color=_RED,
                marker="x",
                s=28,
                label="rejected",
                zorder=8,
            )
    _time_cursor(axis, trace, step)
    _annotate_changes(
        axis,
        context,
        step,
        kinds={"snapshot", "admission", "rejection", "event-warning", "event-failure"},
    )
    axis.set_xlabel("time [s]")
    axis.set_ylabel("loss / gradient / safe fraction")
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(bptt_lines)
    labels.extend(line.get_label() for line in bptt_lines)
    axis.legend(handles, labels, fontsize=5.5, frameon=False, labelcolor=_TEXT, ncol=3)
    has_admission = (
        any(
            "admit" in event.name.lower() or "swap" in event.name.lower()
            for event in context.events
        )
        or has_sidecar_admission
    )
    if not has_admission:
        axis.text(
            0.99,
            0.03,
            "candidate decision: UNAVAILABLE • snapshot swaps only",
            transform=axis.transAxes,
            color=_YELLOW,
            fontsize=6.3,
            ha="right",
        )


def _plot_controls(axis: Any, context: _RenderContext, step: int) -> None:
    trace = context.trace
    axis.set_title("Actions + intervention (native units)", loc="left")
    for values, label, color, style in (
        (context.nominal_control, "‖nominal‖", _MUTED, ":"),
        (context.filtered_control, "‖filtered‖", _CYAN, "-"),
        (context.applied_control, "‖applied‖", _GREEN, "--"),
    ):
        norm = np.linalg.norm(values, axis=1)
        axis.plot(
            trace.time[: step + 1], norm[: step + 1], label=label, color=color, linestyle=style
        )
    axis.plot(
        trace.time[: step + 1],
        context.intervention[: step + 1],
        label="‖filtered−nominal‖",
        color=_ORANGE,
        linewidth=1.5,
    )
    _time_cursor(axis, trace, step)
    axis.set_xlabel("time [s]")
    axis.set_ylabel("control vector norm")
    axis.legend(fontsize=5.5, frameon=False, labelcolor=_TEXT)
    flags = []
    if not bool(trace.executed_control[step]):
        flags.append("TERMINAL OBSERVATION • NO CONTROL")
    elif trace.clipped[step]:
        flags.append("CLIPPED")
    if bool(trace.executed_control[step]) and trace.saturated[step]:
        flags.append("SATURATED")
    axis.text(
        0.99,
        0.03,
        " • ".join(flags) if flags else "no clip/saturation flag",
        transform=axis.transAxes,
        ha="right",
        color=_YELLOW if not bool(trace.executed_control[step]) else (_RED if flags else _MUTED),
        fontsize=6.2,
    )


def _plot_latency(axis: Any, context: _RenderContext, step: int) -> None:
    trace = context.trace
    axis.set_title("Latency + solver/post-check residuals • status code unavailable", loc="left")
    for index, name in enumerate(trace.latency_names):
        axis.plot(
            trace.time[: step + 1],
            1000.0 * context.component_latency[: step + 1, index],
            linewidth=1.0,
            label=f"{name} [ms]",
        )
    axis.set_xlabel("time [s]")
    axis.set_ylabel("latency [ms]")
    _time_cursor(axis, trace, step)
    axis.legend(fontsize=5.4, frameon=False, labelcolor=_TEXT, ncol=3, loc="upper left")
    residual_axis = axis.twinx()
    residual_axis.set_facecolor("none")
    positive_kkt = np.maximum(context.solver_kkt_residual, np.finfo(np.float64).tiny)
    residual_axis.plot(
        trace.time[: step + 1],
        np.log10(positive_kkt[: step + 1]),
        color=_PURPLE,
        linestyle=":",
        linewidth=1.0,
        label="log10 KKT residual",
    )
    residual_axis.plot(
        trace.time[: step + 1],
        context.postcheck_residual[: step + 1],
        color=_YELLOW,
        linestyle="--",
        linewidth=1.0,
        label="post-check residual",
    )
    residual_axis.tick_params(colors=_MUTED, labelsize=6)
    residual_axis.spines["right"].set_color(_GRID)
    residual_axis.set_ylabel("residual / log residual", color=_MUTED, fontsize=7)
    residual_axis.legend(fontsize=5.4, frameon=False, labelcolor=_TEXT, loc="upper right")


def _plot_descriptor(axis: Any, context: _RenderContext, step: int) -> None:
    axis.set_title("Normalized trajectory descriptors", loc="left")
    sidecar = context.sidecar
    if (
        sidecar is None
        or sidecar.descriptor_names.size == 0
        or not np.any(sidecar.descriptor_available[step])
    ):
        _unavailable(
            axis,
            "UNAVAILABLE AT THIS STEP\n"
            "no descriptor vector is reconstructed\n"
            "from state or action history",
        )
        return
    values = np.array(sidecar.normalized_descriptors[step].T, copy=True)
    values[:, ~sidecar.descriptor_available[step]] = np.nan
    extent = max(float(np.nanmax(np.abs(values))), 1.0)
    axis.imshow(
        values,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-extent,
        vmax=extent,
        interpolation="nearest",
    )
    selected = int(context.trace.selected_policy[step])
    if selected >= 0:
        axis.axvline(selected, color=_TEXT, linewidth=1.0)
    axis.set_xlabel("policy index")
    axis.set_ylabel("normalized descriptor")
    if len(sidecar.descriptor_names) <= 12:
        axis.set_yticks(np.arange(len(sidecar.descriptor_names)))
        axis.set_yticklabels([str(name) for name in sidecar.descriptor_names], fontsize=5.5)
    axis.text(
        0.99,
        0.02,
        "recorded dimensionless normalized values",
        transform=axis.transAxes,
        color=_TEXT,
        fontsize=5.8,
        ha="right",
    )


def _plot_evidence(axis: Any, context: _RenderContext, step: int) -> None:
    trace = context.trace
    axis.set_title("Evidence / changes through current step", loc="left")
    axis.set_xticks([])
    axis.set_yticks([])
    unavailable = [item.label for item in context.evidence if item.status == "unavailable"]
    recent = [item for item in context.annotations if item.step <= step][-5:]
    if bool(trace.executed_control[step]):
        control_lines = (
            f"hard-safe policies  {int(context.safe_policy_count[step])}/{len(trace.policy_names)}",
            f"KKT residual  {context.solver_kkt_residual[step]:.2e}",
            f"post-check  {context.postcheck_residual[step]:+.2e}",
        )
    else:
        control_lines = (
            "TERMINAL OBSERVATION • NO CONTROL",
            "hard-safe policies  N/A",
            "KKT / post-check / latency  N/A",
        )
    lines = [f"STATUS  {_status(trace, step)[0]}", *control_lines, "", "RECORDED CHANGES"]
    lines.extend(f"t={trace.time[item.step]:.3f}s  {item.label}" for item in recent)
    if not recent:
        lines.append("none yet")
    lines.extend(("", f"UNAVAILABLE ELEMENTS ({len(unavailable)})"))
    lines.extend(f"• {label}" for label in unavailable[:5])
    if len(unavailable) > 5:
        lines.append(f"• +{len(unavailable) - 5} more (inventory)")
    axis.text(
        0.03,
        0.95,
        "\n".join(lines),
        transform=axis.transAxes,
        color=_TEXT,
        fontsize=6.3,
        va="top",
        linespacing=1.15,
    )


def _style_axis(axis: Any) -> None:
    axis.set_facecolor(_PANEL)
    axis.tick_params(colors=_MUTED, labelsize=6)
    for spine in axis.spines.values():
        spine.set_color(_GRID)
    axis.grid(True, color=_GRID, alpha=0.35, linewidth=0.5)
    axis.title.set_color(_TEXT)
    axis.title.set_fontsize(8.2)
    axis.xaxis.label.set_color(_MUTED)
    axis.yaxis.label.set_color(_MUTED)
    axis.xaxis.label.set_size(6.5)
    axis.yaxis.label.set_size(6.5)


def _unavailable(axis: Any, message: str) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(
        0.5,
        0.5,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        color=_YELLOW,
        fontsize=8,
        weight="bold",
        linespacing=1.4,
    )


def _status(trace: ImmutableTrace, step: int) -> tuple[str, str]:
    terminal = not bool(trace.executed_control[step])
    if bool(trace.contact[step]):
        status, color = "UNSAFE • CONTACT", _RED
    elif bool(trace.failure[step]):
        status, color = "UNSAFE • HARD CONSTRAINT FAILURE", _RED
    elif bool(trace.degraded[step]):
        status, color = "DEGRADED • NO CERTIFIED FALLBACK", _ORANGE
    else:
        status, color = "SAFE IN RECORDED FINITE-HORIZON CHECKS", _GREEN
    return (f"TERMINAL OBSERVATION • {status}" if terminal else status), color


def _mask_nonexecuted_rows(trace: ImmutableTrace, values: np.ndarray) -> np.ndarray:
    """Return float display data with no-control sentinels represented as unavailable, not zero."""
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[0] != trace.steps:
        raise ValueError("time-leading display data must have trace.steps rows")
    masked = np.asarray(array, dtype=np.float64).copy()
    masked[~trace.executed_control] = np.nan
    return masked


def _safe_policy_counts(trace: ImmutableTrace) -> np.ndarray:
    """Count hard-safe policies only at executed controller rows; terminal rows are unavailable."""
    counts = np.count_nonzero(trace.policy_values >= 0.0, axis=1).astype(np.float64)
    counts[~trace.executed_control] = np.nan
    return counts


def _shade_status(axis: Any, trace: ImmutableTrace) -> None:
    dt = float(np.median(np.diff(trace.time)))
    for step in np.flatnonzero(trace.degraded & ~trace.failure):
        axis.axvspan(
            trace.time[step] - 0.5 * dt,
            trace.time[step] + 0.5 * dt,
            color=_ORANGE,
            alpha=0.12,
            linewidth=0,
        )
    for step in np.flatnonzero(trace.failure):
        axis.axvspan(
            trace.time[step] - 0.5 * dt,
            trace.time[step] + 0.5 * dt,
            color=_RED,
            alpha=0.16,
            linewidth=0,
        )


def _time_cursor(axis: Any, trace: ImmutableTrace, step: int) -> None:
    axis.axvline(trace.time[step], color=_TEXT, linewidth=0.7, alpha=0.8)
    axis.set_xlim(float(trace.time[0]), float(trace.time[-1]))


def _annotate_changes(
    axis: Any, context: _RenderContext, step: int, *, kinds: set[str] | None = None
) -> None:
    colors = {
        "dynamics-change": _PURPLE,
        "snapshot": _GREEN,
        "model": _CYAN,
        "admission": _GREEN,
        "rejection": _RED,
        "degraded": _ORANGE,
        "failure": _RED,
        "contact": _RED,
        "event-warning": _ORANGE,
        "event-failure": _RED,
    }
    shown = 0
    for item in context.annotations:
        if item.step > step or (kinds is not None and item.kind not in kinds):
            continue
        axis.axvline(
            context.trace.time[item.step],
            color=colors.get(item.kind, _MUTED),
            linewidth=0.65,
            alpha=0.65,
        )
        if shown < 3:
            axis.text(
                context.trace.time[item.step],
                0.98 - 0.09 * shown,
                item.label,
                transform=axis.get_xaxis_transform(),
                color=colors.get(item.kind, _MUTED),
                fontsize=5.3,
                rotation=90,
                va="top",
                ha="right",
            )
            shown += 1


def _isometric(position: np.ndarray) -> np.ndarray:
    value = np.asarray(position, dtype=np.float64)
    horizontal = math.sqrt(3.0) * 0.5 * (value[..., 0] - value[..., 1])
    vertical = value[..., 2] + 0.5 * (value[..., 0] + value[..., 1])
    return np.stack((horizontal, vertical), axis=-1)


def _position_indices(trace: ImmutableTrace) -> tuple[int, int, int] | None:
    names = tuple(str(name) for name in trace.state_names)
    for candidates in (("position_x", "position_y", "position_z"), ("x", "y", "z")):
        if all(candidate in names for candidate in candidates):
            return tuple(names.index(candidate) for candidate in candidates)  # type: ignore[return-value]
    return None


def _dynamics_state_indices(trace: ImmutableTrace) -> tuple[tuple[int, str], ...]:
    recognized = {
        "mass_scale",
        "drag_scale",
        "wind_x",
        "wind_y",
        "wind_z",
        "rotor_efficiency_0",
        "rotor_efficiency_1",
        "rotor_efficiency_2",
        "rotor_efficiency_3",
    }
    return tuple(
        (index, str(name))
        for index, name in enumerate(trace.state_names)
        if str(name) in recognized
    )


def _validate_tape_binding(trace: ImmutableTrace, tape: ScenarioTape | None) -> None:
    if tape is None:
        return
    from crazyflow.safety.da_plcbf.scenarios import ScenarioTape

    if not isinstance(tape, ScenarioTape):
        raise TypeError("tape must be a ScenarioTape or None")
    tape.validate()
    if str(trace.scenario_tape_sha256) != tape.sha256:
        raise ValueError("trace scenario digest does not match the supplied ScenarioTape")
    if tape.steps < trace.steps or not np.array_equal(trace.time, tape.time[: trace.steps]):
        raise ValueError("trace time grid must exactly match a prefix of the ScenarioTape")


def _validate_sidecar_binding(
    trace: ImmutableTrace, tape: ScenarioTape | None, sidecar: DashboardEvidence | None
) -> None:
    if sidecar is None:
        return
    validate_dashboard_evidence_binding(sidecar, trace, tape)


def _validate_events(
    trace: ImmutableTrace, events: Sequence[ArtifactEvent] | None
) -> tuple[ArtifactEvent, ...]:
    if events is None:
        return ()
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise TypeError("events must be a sequence of ArtifactEvent values")
    validated = tuple(events)
    for index, event in enumerate(validated):
        if not isinstance(event, ArtifactEvent):
            raise TypeError("events must contain only ArtifactEvent values")
        event.validate()
        if event.sequence != index:
            raise ValueError("event sequence values must be contiguous from zero")
        if event.step >= trace.steps:
            raise ValueError("event step lies outside trace")
        if not math.isclose(event.time_seconds, float(trace.time[event.step]), abs_tol=1e-12):
            raise ValueError("event time does not match trace")
        if event.snapshot_version != int(trace.snapshot_version[event.step]):
            raise ValueError("event snapshot version does not match trace")
        if event.model_version != int(trace.model_version[event.step]):
            raise ValueError("event model version does not match trace")
    return validated


def _validate_size(size: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(size, tuple) or len(size) != 2:
        raise TypeError("size must be a (width, height) tuple")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in size):
        raise TypeError("dashboard dimensions must be integers")
    width, height = int(size[0]), int(size[1])
    if width < 1280 or height < 720 or width % 2 or height % 2:
        raise ValueError("scientific dashboard must use even dimensions of at least 1280x720")
    return width, height


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest") from error


def _validate_notes(values: tuple[str, ...], name: str, *, required: bool) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if required and not values:
        raise ValueError(f"{name} must contain an inspection note")
    if any(not isinstance(value, str) or len(value.strip()) < 8 for value in values):
        raise ValueError(f"every {name} entry must be a substantive string")


def _review_mapping(record: VisualReviewRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["keyframe_indices"] = list(record.keyframe_indices)
    payload["checks"] = [asdict(check) for check in record.checks]
    payload["notes"] = list(record.notes)
    payload["revisions"] = list(record.revisions)
    return payload


def _review_from_mapping(payload: Any) -> VisualReviewRecord:
    if not isinstance(payload, Mapping):
        raise ValueError("visual review structured record must be an object")
    required = {
        "schema_version",
        "reviewer",
        "reviewer_kind",
        "reviewed_utc",
        "disposition",
        "trace_content_sha256",
        "scenario_tape_sha256",
        "dashboard_evidence_sha256",
        "video_file_sha256",
        "decoded_frames_sha256",
        "frame_width",
        "frame_height",
        "keyframe_indices",
        "checks",
        "notes",
        "revisions",
    }
    if set(payload) != required:
        raise ValueError("visual review structured record has missing or extra keys")
    raw_checks = payload["checks"]
    if not isinstance(raw_checks, list):
        raise ValueError("visual review checks must be a list")
    checks = []
    for raw in raw_checks:
        if not isinstance(raw, Mapping) or set(raw) != {"name", "status", "note"}:
            raise ValueError("visual review check has missing or extra keys")
        checks.append(VisualReviewCheck(**raw))
    try:
        return VisualReviewRecord(
            schema_version=payload["schema_version"],
            reviewer=payload["reviewer"],
            reviewer_kind=payload["reviewer_kind"],
            reviewed_utc=payload["reviewed_utc"],
            disposition=payload["disposition"],
            trace_content_sha256=payload["trace_content_sha256"],
            scenario_tape_sha256=payload["scenario_tape_sha256"],
            dashboard_evidence_sha256=payload["dashboard_evidence_sha256"],
            video_file_sha256=payload["video_file_sha256"],
            decoded_frames_sha256=payload["decoded_frames_sha256"],
            frame_width=payload["frame_width"],
            frame_height=payload["frame_height"],
            keyframe_indices=tuple(payload["keyframe_indices"]),
            checks=tuple(checks),
            notes=tuple(payload["notes"]),
            revisions=tuple(payload["revisions"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("visual review structured record contains invalid field types") from error
