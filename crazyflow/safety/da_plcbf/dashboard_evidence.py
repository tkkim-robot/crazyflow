"""Strict content-addressed sidecars for scientific-dashboard-only evidence.

The immutable control trace remains the authority for executed state, controls, barriers, and
status.  This sidecar carries high-dimensional *recorded* review evidence that does not fit trace
schema v1: finite-horizon rollout positions, normalized descriptors, estimator uncertainty,
candidate admissions, and decomposed BPTT timings.  Missing samples have explicit masks and their
numeric storage must be exactly zero, preventing stale filler values from appearing as evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import struct
import tempfile
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from crazyflow.safety.da_plcbf.artifacts import ArtifactEvent, ImmutableTrace
    from crazyflow.safety.da_plcbf.scenarios import ScenarioTape


DASHBOARD_EVIDENCE_SCHEMA_VERSION = 1
_DIGEST_PREFIX = b"crazyflow.da_plcbf.dashboard-evidence.v1\0"
_MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True, eq=False)
class DashboardEvidence:
    """Immutable arrays recorded by a run for later scientific visual review.

    Shape contract, where ``T`` is trace steps, ``K`` is policy count, ``H`` is rollout nodes,
    ``Hp`` is prediction nodes, ``R`` is prediction samples, ``M`` is obstacle slots, ``D`` is
    descriptor dimensions, ``P`` is dynamics parameters, ``Ru`` is uncertainty samples, ``G`` is
    ghost-rollout variants, ``A`` is admission reasons, and ``Q`` is BPTT timing components:

    - policy_names ``[K]`` (must exactly equal the bound trace)
    - rollout_time ``[H]``; nominal ``[T,H,3]``; fallback ``[T,K,H,3]``; selected ``[T,H,3]``
    - ghost names ``[G]`` and positions ``[T,G,H,3]``
    - prediction time ``[Hp]``, positions ``[T,R,M,Hp,3]``, mask ``[T,R,M,Hp]``
    - descriptor names ``[D]`` and normalized descriptors ``[T,K,D]``
    - dynamics names ``[P]``, truth/estimate ``[T,P]``, uncertainty ``[T,Ru,P]``
    - admission arrays ``[T]`` with reason names ``[A]``
    - BPTT timing names ``[Q]`` and seconds/availability ``[T,Q]``

    Every optional sample has an availability mask.  Values behind false masks must be zero.
    """

    schema_version: np.ndarray
    trace_content_sha256: np.ndarray
    scenario_tape_sha256: np.ndarray
    policy_names: np.ndarray
    rollout_time: np.ndarray
    nominal_rollout_positions: np.ndarray
    nominal_rollout_available: np.ndarray
    fallback_rollout_positions: np.ndarray
    fallback_rollout_available: np.ndarray
    selected_rollout_positions: np.ndarray
    selected_rollout_available: np.ndarray
    ghost_rollout_names: np.ndarray
    ghost_rollout_positions: np.ndarray
    ghost_rollout_available: np.ndarray
    prediction_time: np.ndarray
    prediction_positions: np.ndarray
    prediction_available: np.ndarray
    descriptor_names: np.ndarray
    normalized_descriptors: np.ndarray
    descriptor_available: np.ndarray
    dynamics_parameter_names: np.ndarray
    dynamics_true: np.ndarray
    dynamics_true_available: np.ndarray
    dynamics_estimated: np.ndarray
    dynamics_estimated_available: np.ndarray
    dynamics_uncertainty_samples: np.ndarray
    dynamics_uncertainty_available: np.ndarray
    admission_recorded: np.ndarray
    candidate_present: np.ndarray
    candidate_admitted: np.ndarray
    candidate_rejected: np.ndarray
    admission_margin: np.ndarray
    admission_reason_names: np.ndarray
    admission_reason_index: np.ndarray
    bptt_timing_names: np.ndarray
    bptt_timing_seconds: np.ndarray
    bptt_timing_available: np.ndarray

    def __post_init__(self) -> None:
        """Defensively copy, freeze, and validate every member."""
        for item in fields(self):
            object.__setattr__(self, item.name, _frozen_array(getattr(self, item.name)))
        self.validate()

    @property
    def steps(self) -> int:
        """Number of controller nodes represented by the sidecar."""
        return int(self.nominal_rollout_available.shape[0])

    @property
    def content_sha256(self) -> str:
        """Semantic digest independent of NPZ metadata."""
        return _canonical_digest(_evidence_arrays(self))

    def validate(self) -> None:
        """Validate names, shapes, masks, finiteness, and zero-fill invariants."""
        _scalar_integer(self.schema_version, "schema_version", DASHBOARD_EVIDENCE_SCHEMA_VERSION)
        _scalar_digest(self.trace_content_sha256, "trace_content_sha256")
        _scalar_digest(self.scenario_tape_sha256, "scenario_tape_sha256")
        policy_names = _names(self.policy_names, "policy_names", minimum=1)
        steps = _boolean_vector_length(self.nominal_rollout_available, "nominal_rollout_available")
        if steps < 2:
            raise ValueError("dashboard evidence must contain at least two steps")

        rollout_nodes = _time_axis(self.rollout_time, "rollout_time")
        _float_shape(
            self.nominal_rollout_positions, (steps, rollout_nodes, 3), "nominal_rollout_positions"
        )
        _float_shape(
            self.fallback_rollout_positions,
            (steps, len(policy_names), rollout_nodes, 3),
            "fallback_rollout_positions",
        )
        _boolean_shape(
            self.fallback_rollout_available,
            (steps, len(policy_names)),
            "fallback_rollout_available",
        )
        _float_shape(
            self.selected_rollout_positions, (steps, rollout_nodes, 3), "selected_rollout_positions"
        )
        _boolean_shape(self.selected_rollout_available, (steps,), "selected_rollout_available")
        ghost_names = _names(self.ghost_rollout_names, "ghost_rollout_names", minimum=0)
        _float_shape(
            self.ghost_rollout_positions,
            (steps, len(ghost_names), rollout_nodes, 3),
            "ghost_rollout_positions",
        )
        _boolean_shape(
            self.ghost_rollout_available, (steps, len(ghost_names)), "ghost_rollout_available"
        )
        rollout_masks = (
            self.nominal_rollout_available,
            self.fallback_rollout_available,
            self.selected_rollout_available,
            self.ghost_rollout_available,
        )
        if rollout_nodes == 0 and any(np.any(mask) for mask in rollout_masks):
            raise ValueError("rollout availability requires at least two rollout-time nodes")
        _zero_where_unavailable(
            self.nominal_rollout_positions,
            self.nominal_rollout_available[:, None, None],
            "nominal_rollout_positions",
        )
        _zero_where_unavailable(
            self.fallback_rollout_positions,
            self.fallback_rollout_available[:, :, None, None],
            "fallback_rollout_positions",
        )
        _zero_where_unavailable(
            self.selected_rollout_positions,
            self.selected_rollout_available[:, None, None],
            "selected_rollout_positions",
        )
        _zero_where_unavailable(
            self.ghost_rollout_positions,
            self.ghost_rollout_available[:, :, None, None],
            "ghost_rollout_positions",
        )

        prediction_nodes = _time_axis(self.prediction_time, "prediction_time")
        if self.prediction_positions.ndim != 5 or self.prediction_positions.shape[0] != steps:
            raise ValueError("prediction_positions must have shape [T,R,M,Hp,3]")
        prediction_shape = self.prediction_positions.shape
        if prediction_shape[-2:] != (prediction_nodes, 3):
            raise ValueError("prediction_positions does not match prediction_time or xyz")
        _finite_float(self.prediction_positions, "prediction_positions")
        _boolean_shape(self.prediction_available, prediction_shape[:-1], "prediction_available")
        if prediction_nodes == 0 and np.any(self.prediction_available):
            raise ValueError("prediction availability requires at least two prediction-time nodes")
        _zero_where_unavailable(
            self.prediction_positions, self.prediction_available[..., None], "prediction_positions"
        )

        descriptor_names = _names(self.descriptor_names, "descriptor_names", minimum=0)
        _float_shape(
            self.normalized_descriptors,
            (steps, len(policy_names), len(descriptor_names)),
            "normalized_descriptors",
        )
        _boolean_shape(
            self.descriptor_available, (steps, len(policy_names)), "descriptor_available"
        )
        if not descriptor_names and np.any(self.descriptor_available):
            raise ValueError("descriptor availability requires descriptor names")
        _zero_where_unavailable(
            self.normalized_descriptors,
            self.descriptor_available[:, :, None],
            "normalized_descriptors",
        )

        parameter_names = _names(
            self.dynamics_parameter_names, "dynamics_parameter_names", minimum=0
        )
        parameter_count = len(parameter_names)
        _float_shape(self.dynamics_true, (steps, parameter_count), "dynamics_true")
        _boolean_shape(
            self.dynamics_true_available, (steps, parameter_count), "dynamics_true_available"
        )
        _float_shape(self.dynamics_estimated, (steps, parameter_count), "dynamics_estimated")
        _boolean_shape(
            self.dynamics_estimated_available,
            (steps, parameter_count),
            "dynamics_estimated_available",
        )
        if self.dynamics_uncertainty_samples.ndim != 3:
            raise ValueError("dynamics_uncertainty_samples must have shape [T,Ru,P]")
        uncertainty_shape = self.dynamics_uncertainty_samples.shape
        if uncertainty_shape[0] != steps or uncertainty_shape[2] != parameter_count:
            raise ValueError("dynamics_uncertainty_samples has inconsistent T or P axes")
        _finite_float(self.dynamics_uncertainty_samples, "dynamics_uncertainty_samples")
        _boolean_shape(
            self.dynamics_uncertainty_available,
            uncertainty_shape[:2],
            "dynamics_uncertainty_available",
        )
        _zero_where_unavailable(self.dynamics_true, self.dynamics_true_available, "dynamics_true")
        _zero_where_unavailable(
            self.dynamics_estimated, self.dynamics_estimated_available, "dynamics_estimated"
        )
        _zero_where_unavailable(
            self.dynamics_uncertainty_samples,
            self.dynamics_uncertainty_available[:, :, None],
            "dynamics_uncertainty_samples",
        )

        _scalar_boolean(self.admission_recorded, "admission_recorded")
        for name in ("candidate_present", "candidate_admitted", "candidate_rejected"):
            _boolean_shape(getattr(self, name), (steps,), name)
        _float_shape(self.admission_margin, (steps,), "admission_margin")
        reason_names = _names(self.admission_reason_names, "admission_reason_names", minimum=0)
        if (
            self.admission_reason_index.dtype.kind not in "iu"
            or self.admission_reason_index.shape != (steps,)
        ):
            raise ValueError("admission_reason_index must be an integer vector [T]")
        reason_index = self.admission_reason_index.astype(np.int64)
        if np.any((reason_index < -1) | (reason_index >= len(reason_names))):
            raise ValueError("admission_reason_index contains an invalid name index")
        if np.any(self.candidate_admitted & self.candidate_rejected):
            raise ValueError("a candidate cannot be admitted and rejected at the same step")
        if np.any((self.candidate_admitted | self.candidate_rejected) & ~self.candidate_present):
            raise ValueError("admitted/rejected flags require candidate_present")
        if np.any((reason_index >= 0) & ~self.candidate_present):
            raise ValueError("an admission reason requires candidate_present")
        if np.any(self.candidate_rejected & (reason_index < 0)):
            raise ValueError("every rejected candidate requires a recorded reason")
        if not bool(self.admission_recorded):
            if any(
                np.any(value)
                for value in (
                    self.candidate_present,
                    self.candidate_admitted,
                    self.candidate_rejected,
                )
            ) or np.any(reason_index != -1):
                raise ValueError("unrecorded admissions cannot contain candidate evidence")
        if np.any(self.admission_margin[~self.candidate_present] != 0.0):
            raise ValueError("admission_margin must be zero when no candidate is present")

        timing_names = _names(self.bptt_timing_names, "bptt_timing_names", minimum=0)
        _float_shape(self.bptt_timing_seconds, (steps, len(timing_names)), "bptt_timing_seconds")
        if np.any(self.bptt_timing_seconds < 0.0):
            raise ValueError("bptt_timing_seconds must be nonnegative")
        _boolean_shape(
            self.bptt_timing_available, (steps, len(timing_names)), "bptt_timing_available"
        )
        _zero_where_unavailable(
            self.bptt_timing_seconds, self.bptt_timing_available, "bptt_timing_seconds"
        )


def validate_dashboard_evidence_binding(
    evidence: DashboardEvidence,
    trace: ImmutableTrace,
    tape: ScenarioTape | None = None,
    *,
    events: Sequence[ArtifactEvent] | None = None,
    expected_dynamics: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    """Require exact trace/tape/event bindings and independently replayed dynamics evidence."""
    from crazyflow.safety.da_plcbf.artifacts import ImmutableTrace

    if not isinstance(evidence, DashboardEvidence):
        raise TypeError("evidence must be DashboardEvidence")
    if not isinstance(trace, ImmutableTrace):
        raise TypeError("trace must be ImmutableTrace")
    evidence.validate()
    trace.validate()
    if str(evidence.trace_content_sha256) != trace.content_sha256:
        raise ValueError("dashboard evidence trace digest does not match the trace")
    if str(evidence.scenario_tape_sha256) != str(trace.scenario_tape_sha256):
        raise ValueError("dashboard evidence scenario digest does not match the trace")
    if evidence.steps != trace.steps:
        raise ValueError("dashboard evidence step count does not match the trace")
    if not np.array_equal(evidence.policy_names, trace.policy_names):
        raise ValueError("dashboard evidence policy names do not match the trace")
    if tape is not None:
        from crazyflow.safety.da_plcbf.scenarios import ScenarioTape

        if not isinstance(tape, ScenarioTape):
            raise TypeError("tape must be ScenarioTape or None")
        tape.validate()
        if tape.sha256 != str(evidence.scenario_tape_sha256):
            raise ValueError("dashboard evidence scenario digest does not match the tape")
        if tape.steps < evidence.steps:
            raise ValueError("dashboard evidence has more recorded steps than the tape")
        expected_positions, expected_available, expected_time = _prediction_evidence_from_tape(
            tape, steps=evidence.steps, prediction_nodes=evidence.prediction_time.size
        )
        if not np.array_equal(evidence.prediction_time, expected_time):
            raise ValueError("dashboard prediction time does not match the scenario tape")
        if not np.array_equal(evidence.prediction_available, expected_available):
            raise ValueError("dashboard prediction availability does not match the scenario tape")
        if not np.array_equal(evidence.prediction_positions, expected_positions):
            raise ValueError("dashboard prediction positions do not match the scenario tape")
    if events is not None:
        expected_admission = _admission_evidence_from_events(events, steps=evidence.steps)
        actual_admission = (
            evidence.admission_recorded,
            evidence.candidate_present,
            evidence.candidate_admitted,
            evidence.candidate_rejected,
            evidence.admission_margin,
            evidence.admission_reason_names,
            evidence.admission_reason_index,
            evidence.bptt_timing_names,
            evidence.bptt_timing_seconds,
            evidence.bptt_timing_available,
        )
        if any(
            not np.array_equal(actual, expected)
            for actual, expected in zip(actual_admission, expected_admission, strict=True)
        ):
            raise ValueError("dashboard admission/BPTT evidence does not recompute from events")
        commitments = tuple(
            event
            for event in events
            if event.category == "runtime" and event.name == "dashboard_evidence_committed"
        )
        if len(commitments) != 1:
            raise ValueError("dashboard evidence requires exactly one canonical commitment event")
        details = commitments[0].details
        expected_details = {
            "dashboard_evidence_sha256": evidence.content_sha256,
            "trace_content_sha256": trace.content_sha256,
            "schema_version": DASHBOARD_EVIDENCE_SCHEMA_VERSION,
        }
        if dict(details) != expected_details:
            raise ValueError("dashboard evidence commitment event does not match the sidecar")
    if expected_dynamics is not None:
        if len(expected_dynamics) != 4:
            raise ValueError("expected_dynamics must contain truth, estimate, samples, and mask")
        actual = (
            evidence.dynamics_true,
            evidence.dynamics_estimated,
            evidence.dynamics_uncertainty_samples,
            evidence.dynamics_uncertainty_available,
        )
        names = ("truth", "estimate", "uncertainty samples", "uncertainty mask")
        for name, observed, expected in zip(names, actual, expected_dynamics, strict=True):
            if observed.dtype == np.bool_:
                matches = np.array_equal(observed, expected)
            else:
                matches = np.allclose(observed, expected, rtol=2e-7, atol=2e-7)
            if observed.shape != np.asarray(expected).shape or not matches:
                raise ValueError(f"dashboard dynamics {name} does not independently replay")


def _admission_evidence_from_events(
    events: Sequence[ArtifactEvent], *, steps: int
) -> tuple[np.ndarray, ...]:
    """Reconstruct the exact candidate-decision and BPTT timing arrays from canonical events."""
    candidate_present = np.zeros(steps, dtype=np.bool_)
    candidate_admitted = np.zeros(steps, dtype=np.bool_)
    candidate_rejected = np.zeros(steps, dtype=np.bool_)
    admission_margin = np.zeros(steps, dtype=np.float64)
    reason_by_step: dict[int, str] = {}
    timing_names = ("setup", "compile", "warmup", "execution", "validation")
    timing_keys = tuple(f"bptt_{name}_seconds" for name in timing_names)
    bptt_timing = np.zeros((steps, len(timing_names)), dtype=np.float64)
    bptt_available = np.zeros((steps, len(timing_names)), dtype=np.bool_)
    for event in events:
        if event.name in {"candidate_admitted", "candidate_rejected"}:
            candidate_present[event.step] = True
            candidate_admitted[event.step] = event.name == "candidate_admitted"
            candidate_rejected[event.step] = event.name == "candidate_rejected"
            admission_margin[event.step] = float(event.details.get("admission_margin", 0.0))
            failed = event.details.get("failed_gates", [])
            if candidate_rejected[event.step] and isinstance(failed, list) and failed:
                reason_by_step[event.step] = "failed_" + "_".join(str(name) for name in failed)
            else:
                reason_by_step[event.step] = str(event.details.get("reason", "hard_gates_passed"))
        for timing_index, key in enumerate(timing_keys):
            value = event.details.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                bptt_timing[event.step, timing_index] = float(value)
                bptt_available[event.step, timing_index] = True
    reason_names = tuple(dict.fromkeys(reason_by_step.values()))
    lookup = {name: index for index, name in enumerate(reason_names)}
    reason_index = np.full(steps, -1, dtype=np.int16)
    for step, reason in reason_by_step.items():
        reason_index[step] = lookup[reason]
    return (
        np.asarray(bool(reason_by_step)),
        candidate_present,
        candidate_admitted,
        candidate_rejected,
        admission_margin,
        np.asarray(reason_names, dtype="<U128"),
        reason_index,
        np.asarray(timing_names),
        bptt_timing,
        bptt_available,
    )


def _prediction_evidence_from_tape(
    tape: ScenarioTape, *, steps: int, prediction_nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the exact observed-slot prediction sidecar arrays from a scenario tape."""
    from crazyflow.safety.da_plcbf.scenarios import ScenarioTape

    if not isinstance(tape, ScenarioTape):
        raise TypeError("tape must be a ScenarioTape")
    tape.validate()
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("steps must be an integer >=2")
    if (
        isinstance(prediction_nodes, bool)
        or not isinstance(prediction_nodes, int)
        or prediction_nodes < 0
    ):
        raise ValueError("prediction_nodes must be a nonnegative integer")
    if steps > tape.steps or steps + prediction_nodes - 1 > tape.steps:
        raise ValueError("dashboard prediction horizon exceeds the scenario tape")

    samples = tape.prediction_samples
    obstacles = tape.dynamic_positions.shape[1]
    positions = np.zeros((steps, samples, obstacles, prediction_nodes, 3), dtype=np.float64)
    available = np.zeros((steps, samples, obstacles, prediction_nodes), dtype=np.bool_)
    for step in range(steps):
        stop = step + prediction_nodes
        raw_positions = np.transpose(tape.prediction_positions[:, step:stop], (0, 2, 1, 3))
        future_mask = np.transpose(tape.dynamic_time_mask[step:stop], (1, 0))
        observed_active = tape.dynamic_time_mask[step] & tape.dynamic_slot_mask
        raw_available = np.logical_and(
            np.broadcast_to(future_mask[None], raw_positions.shape[:-1]),
            np.broadcast_to(observed_active[None, :, None], raw_positions.shape[:-1]),
        )
        available[step] = raw_available
        positions[step] = np.where(raw_available[..., None], raw_positions, 0.0)
    prediction_time = np.asarray(tape.time[:prediction_nodes] - tape.time[0], dtype=np.float64)
    return positions, available, prediction_time


def save_dashboard_evidence(
    evidence: DashboardEvidence, path: str | os.PathLike[str], *, overwrite: bool = False
) -> str:
    """Atomically save a deterministic NPZ and return its semantic content digest."""
    if not isinstance(evidence, DashboardEvidence):
        raise TypeError("evidence must be DashboardEvidence")
    evidence.validate()
    destination = _destination(path, overwrite=overwrite)
    arrays = _evidence_arrays(evidence)
    arrays["content_sha256"] = np.asarray(evidence.content_sha256)
    payload = _deterministic_npz(arrays)
    _atomic_write(destination, payload, overwrite=overwrite)
    return evidence.content_sha256


def load_dashboard_evidence(path: str | os.PathLike[str]) -> DashboardEvidence:
    """Strictly load, validate, and digest-check a dashboard-evidence NPZ."""
    source = Path(path)
    if source.suffix.lower() != ".npz":
        raise ValueError("dashboard evidence path must end in .npz")
    expected = {item.name for item in fields(DashboardEvidence)} | {"content_sha256"}
    loaded = _load_strict_npz(source, expected)
    recorded_digest = _scalar_digest(loaded.pop("content_sha256"), "content_sha256")
    try:
        evidence = DashboardEvidence(**loaded)
    except (TypeError, ValueError) as error:
        raise ValueError("dashboard evidence failed schema validation") from error
    if not hmac.compare_digest(recorded_digest, evidence.content_sha256):
        raise ValueError("dashboard evidence content digest mismatch")
    return evidence


def _evidence_arrays(evidence: DashboardEvidence) -> dict[str, np.ndarray]:
    return {item.name: getattr(evidence, item.name) for item in fields(evidence)}


def _frozen_array(value: Any) -> np.ndarray:
    array = np.array(value, copy=True, order="C")
    if array.dtype.hasobject or array.dtype.kind not in "biufSU":
        raise ValueError("dashboard evidence supports real numeric, boolean, and string arrays")
    array.setflags(write=False)
    return array


def _scalar_integer(value: np.ndarray, name: str, expected: int) -> None:
    if value.shape != () or value.dtype.kind not in "iu" or int(value) != expected:
        raise ValueError(f"{name} must be scalar integer {expected}")


def _scalar_boolean(value: np.ndarray, name: str) -> bool:
    if value.shape != () or value.dtype != np.bool_:
        raise ValueError(f"{name} must be a scalar boolean")
    return bool(value)


def _scalar_digest(value: np.ndarray, name: str) -> str:
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar Unicode SHA-256 digest")
    text = str(value)
    if len(text) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        bytes.fromhex(text)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 digest") from error
    return text


def _names(value: np.ndarray, name: str, *, minimum: int) -> tuple[str, ...]:
    if value.dtype.kind != "U" or value.ndim != 1 or len(value) < minimum:
        raise ValueError(f"{name} must be a Unicode vector with at least {minimum} entries")
    names = tuple(str(item) for item in value)
    if len(set(names)) != len(names) or any(not item.strip() for item in names):
        raise ValueError(f"{name} must contain unique nonempty names")
    return names


def _time_axis(value: np.ndarray, name: str) -> int:
    _finite_float(value, name)
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if value.size == 0:
        return 0
    if value.size < 2 or value[0] != 0.0 or np.any(np.diff(value) <= 0.0):
        raise ValueError(f"{name} must be empty or start at zero and strictly increase")
    return int(value.size)


def _finite_float(value: np.ndarray, name: str) -> None:
    if value.dtype.kind != "f" or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite floating-point array")


def _float_shape(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    _finite_float(value, name)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")


def _boolean_shape(value: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if value.dtype != np.bool_ or value.shape != shape:
        raise ValueError(f"{name} must be a boolean array with shape {shape}")


def _boolean_vector_length(value: np.ndarray, name: str) -> int:
    if value.dtype != np.bool_ or value.ndim != 1:
        raise ValueError(f"{name} must be a boolean vector")
    return int(value.size)


def _zero_where_unavailable(value: np.ndarray, mask: np.ndarray, name: str) -> None:
    if value.size and np.any(np.where(np.broadcast_to(mask, value.shape), 0.0, value) != 0.0):
        raise ValueError(f"{name} must be exactly zero behind false availability masks")


def _destination(path: str | os.PathLike[str], *, overwrite: bool) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("dashboard evidence path must end in .npz")
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    if destination.is_symlink():
        raise ValueError("dashboard evidence destination must not be a symlink")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    return destination


def _atomic_write(destination: Path, payload: bytes, *, overwrite: bool) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, value in sorted(arrays.items()):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(
                info, _npy_bytes(value), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )
    return output.getvalue()


def _canonical_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(_DIGEST_PREFIX)
    for name, value in sorted(arrays.items()):
        encoded_name = name.encode()
        encoded_array = _npy_bytes(value)
        digest.update(struct.pack("<I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<Q", len(encoded_array)))
        digest.update(encoded_array)
    return digest.hexdigest()


def _load_strict_npz(source: Path, expected: set[str]) -> dict[str, np.ndarray]:
    if source.is_symlink() or not source.is_file():
        raise ValueError("dashboard evidence must be a regular non-symlink NPZ")
    try:
        archive = zipfile.ZipFile(source, mode="r")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("dashboard evidence is not a valid NPZ") from error
    expected_members = {f"{name}.npy" for name in expected}
    try:
        infos = archive.infolist()
        member_names = [info.filename for info in infos]
        if len(member_names) != len(set(member_names)) or set(member_names) != expected_members:
            raise ValueError("dashboard evidence has missing, duplicate, or unexpected members")
        if sum(info.file_size for info in infos) > _MAX_ARCHIVE_BYTES:
            raise ValueError("dashboard evidence exceeds the decompressed size limit")
        loaded: dict[str, np.ndarray] = {}
        for info in infos:
            try:
                raw = archive.read(info)
                array = np.load(io.BytesIO(raw), allow_pickle=False)
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                raise ValueError(f"invalid dashboard evidence member {info.filename!r}") from error
            if not isinstance(array, np.ndarray) or array.dtype.hasobject:
                raise ValueError("dashboard evidence members must be non-object arrays")
            loaded[info.filename.removesuffix(".npy")] = array
    finally:
        archive.close()
    return loaded
