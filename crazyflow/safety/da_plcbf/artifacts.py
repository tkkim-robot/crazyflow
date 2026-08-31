"""Strict, content-addressed DA-PLCBF experiment artifacts.

Control runs write numeric data first and render it later.  This module deliberately keeps the
artifact format independent of JAX so validation and replay can run on a CPU-only machine.  NPZ
members are fixed-schema, pickle-free, immutable after loading, and saved with deterministic ZIP
metadata.  JSON is canonical and rejects non-finite numbers.

The schemas are intentionally versioned at their outer boundary.  A schema change must increment
the corresponding version rather than silently accepting a new field with different semantics.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

TRACE_SCHEMA_VERSION = 2
RUN_CONFIG_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 1
SEEDS_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1
METRICS_SCHEMA_VERSION = 1
TIMING_SCHEMA_VERSION = 1
AGGREGATE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1

SAFETY_CLAIM_BOUNDARY = (
    "Under the logged model/scenario samples, constraints, numerical tolerances, and finite "
    "horizon, the hard rollout and filter checks observed the reported margins and violation "
    "rates. This does not prove infinite-horizon, distribution-free, real-world, or hardware "
    "safety."
)

_SLUG_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_MAX_NPZ_BYTES = 2 * 1024 * 1024 * 1024
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_JSONL_LINE_BYTES = 4 * 1024 * 1024
_TRACE_PREFIX = b"crazyflow.da_plcbf.trace.v2\0"


@dataclass(frozen=True, slots=True, eq=False)
class ImmutableTrace:
    """Fixed-schema numeric control trace used for metrics and offline replay.

    The time-leading arrays all have length ``T``.  Names bind otherwise anonymous feature axes,
    so a replay cannot silently reinterpret a barrier, policy, loss, or latency column.  Actual
    hard constraint violations and contact are required to be represented in ``failure``.
    """

    schema_version: np.ndarray
    scenario_tape_sha256: np.ndarray
    time: np.ndarray
    state_names: np.ndarray
    control_names: np.ndarray
    barrier_names: np.ndarray
    policy_names: np.ndarray
    loss_term_names: np.ndarray
    latency_names: np.ndarray
    true_state: np.ndarray
    estimated_state: np.ndarray
    nominal_control: np.ndarray
    filtered_control: np.ndarray
    applied_control: np.ndarray
    executed_control: np.ndarray
    hard_barriers: np.ndarray
    training_values: np.ndarray
    policy_values: np.ndarray
    selected_policy: np.ndarray
    snapshot_version: np.ndarray
    model_version: np.ndarray
    solver_kkt_residual: np.ndarray
    postcheck_residual: np.ndarray
    clipped: np.ndarray
    saturated: np.ndarray
    degraded: np.ndarray
    contact: np.ndarray
    failure: np.ndarray
    loss_terms: np.ndarray
    gradient_norm: np.ndarray
    component_latency_seconds: np.ndarray

    def __post_init__(self) -> None:
        """Copy every leaf into read-only C-contiguous storage, then validate it."""
        for item in fields(self):
            object.__setattr__(self, item.name, _frozen_array(getattr(self, item.name)))
        self.validate()

    @property
    def steps(self) -> int:
        """Number of saved controller nodes."""
        return int(self.time.shape[0])

    @property
    def content_sha256(self) -> str:
        """Canonical digest of names, dtypes, shapes, and values, independent of ZIP metadata."""
        return _canonical_array_digest(_TRACE_PREFIX, _trace_arrays(self))

    def validate(self) -> None:
        """Validate exact dtypes, shapes, finiteness, names, and failure visibility."""
        _require_scalar_integer(self.schema_version, "schema_version", TRACE_SCHEMA_VERSION)
        scenario_digest = _require_scalar_string(self.scenario_tape_sha256, "scenario_tape_sha256")
        _require_sha256(scenario_digest, "scenario_tape_sha256")

        _require_float(self.time, "time", ndim=1)
        if self.time.size < 2:
            raise ValueError("time must contain at least two controller nodes")
        if self.time[0] != 0.0 or np.any(np.diff(self.time) <= 0.0):
            raise ValueError("time must start at zero and be strictly increasing")
        steps = self.steps

        state_names = _validate_names(self.state_names, "state_names", minimum=3)
        control_names = _validate_names(self.control_names, "control_names")
        barrier_names = _validate_names(self.barrier_names, "barrier_names")
        policy_names = _validate_names(self.policy_names, "policy_names")
        loss_names = _validate_names(self.loss_term_names, "loss_term_names")
        latency_names = _validate_names(self.latency_names, "latency_names")

        _require_float_matrix(self.true_state, "true_state", steps, len(state_names))
        _require_float_matrix(self.estimated_state, "estimated_state", steps, len(state_names))
        for name in ("nominal_control", "filtered_control", "applied_control"):
            _require_float_matrix(getattr(self, name), name, steps, len(control_names))
        _require_bool_vector(self.executed_control, "executed_control", steps)
        if not np.any(self.executed_control):
            raise ValueError("executed_control must contain at least one executed transition")
        if np.any(self.executed_control[1:] > self.executed_control[:-1]):
            raise ValueError("executed_control must be a contiguous true prefix")
        if np.count_nonzero(~self.executed_control) > 1:
            raise ValueError("trace schema v2 permits at most one terminal no-control row")
        _require_float_matrix(self.hard_barriers, "hard_barriers", steps, len(barrier_names))
        _require_float_matrix(self.training_values, "training_values", steps, len(policy_names))
        _require_float_matrix(self.policy_values, "policy_values", steps, len(policy_names))
        _require_float_matrix(self.loss_terms, "loss_terms", steps, len(loss_names))
        _require_float_matrix(
            self.component_latency_seconds, "component_latency_seconds", steps, len(latency_names)
        )
        if np.any(self.component_latency_seconds < 0.0):
            raise ValueError("component_latency_seconds must be nonnegative")

        for name in ("selected_policy", "snapshot_version", "model_version"):
            value = getattr(self, name)
            _require_integer_vector(value, name, steps)
        if np.any((self.selected_policy < -1) | (self.selected_policy >= len(policy_names))):
            raise ValueError("selected_policy entries must be -1 or valid policy indices")
        if np.any(self.snapshot_version < 0) or np.any(self.model_version < 0):
            raise ValueError("snapshot_version and model_version must be nonnegative")

        for name in ("solver_kkt_residual", "postcheck_residual", "gradient_norm"):
            value = getattr(self, name)
            _require_float(value, name, ndim=1, shape=(steps,))
        if np.any(self.solver_kkt_residual < 0.0) or np.any(self.gradient_norm < 0.0):
            raise ValueError("solver_kkt_residual and gradient_norm must be nonnegative")

        for name in ("clipped", "saturated", "degraded", "contact", "failure"):
            _require_bool_vector(getattr(self, name), name, steps)
        terminal = ~self.executed_control
        if np.any(terminal):
            zero_fields = (
                self.nominal_control,
                self.filtered_control,
                self.applied_control,
                self.training_values,
                self.policy_values,
                self.solver_kkt_residual,
                self.postcheck_residual,
                self.gradient_norm,
                self.component_latency_seconds,
            )
            if any(np.any(field[terminal] != 0.0) for field in zero_fields):
                raise ValueError("terminal no-control rows must use exact zero numeric sentinels")
            if np.any(self.selected_policy[terminal] != -1):
                raise ValueError("terminal no-control rows must select policy -1")
            if np.any(self.clipped[terminal] | self.saturated[terminal]):
                raise ValueError("terminal no-control rows cannot be clipped or saturated")
        constraint_violation = np.any(self.hard_barriers < 0.0, axis=1)
        if np.any(constraint_violation & ~self.failure):
            raise ValueError("every negative hard barrier must be visible in failure")
        if np.any(self.contact & ~self.failure):
            raise ValueError("contact is a hard failure and must be visible in failure")


@dataclass(frozen=True, slots=True)
class ArtifactEvent:
    """One ordered JSONL runtime, learner, solver, or failure event."""

    sequence: int
    step: int
    time_seconds: float
    category: str
    name: str
    severity: str
    snapshot_version: int
    model_version: int
    details: Mapping[str, Any]

    def validate(self) -> None:
        """Validate an event independently of a particular trace."""
        _require_nonnegative_int(self.sequence, "event.sequence")
        _require_nonnegative_int(self.step, "event.step")
        _require_finite_nonnegative(self.time_seconds, "event.time_seconds")
        _require_slug(self.category, "event.category")
        _require_slug(self.name, "event.name")
        if self.severity not in {"info", "warning", "failure"}:
            raise ValueError("event.severity must be info, warning, or failure")
        _require_nonnegative_int(self.snapshot_version, "event.snapshot_version")
        _require_nonnegative_int(self.model_version, "event.model_version")
        if not isinstance(self.details, Mapping):
            raise TypeError("event.details must be a mapping")
        _validate_json_value(dict(self.details), "event.details")


def save_trace(
    trace: ImmutableTrace, path: str | os.PathLike[str], *, overwrite: bool = False
) -> str:
    """Atomically save a deterministic trace NPZ and return its semantic content digest."""
    if not isinstance(trace, ImmutableTrace):
        raise TypeError("trace must be an ImmutableTrace")
    trace.validate()
    destination = _destination(path, suffix=".npz", overwrite=overwrite)
    payload = _trace_arrays(trace)
    payload["content_sha256"] = np.asarray(trace.content_sha256)
    _atomic_write_bytes(destination, _deterministic_npz_bytes(payload), overwrite=overwrite)
    return trace.content_sha256


def load_trace(path: str | os.PathLike[str]) -> ImmutableTrace:
    """Load an immutable trace and reject malformed, nonfinite, or digest-mismatched archives."""
    source = Path(path)
    if source.suffix.lower() != ".npz":
        raise ValueError("trace path must end in .npz")
    expected = {item.name for item in fields(ImmutableTrace)} | {"content_sha256"}
    loaded = _load_strict_npz(source, expected)
    stored = loaded.pop("content_sha256")
    digest = _require_scalar_string(stored, "content_sha256")
    _require_sha256(digest, "content_sha256")
    try:
        trace = ImmutableTrace(**loaded)
    except (TypeError, ValueError) as error:
        raise ValueError("trace payload failed schema validation") from error
    if not hmac.compare_digest(digest, trace.content_sha256):
        raise ValueError("trace content digest mismatch")
    return trace


def write_run_config(config: Mapping[str, Any], path: str | os.PathLike[str]) -> str:
    """Validate and canonically write ``config.json``; return its file digest."""
    validated = validate_run_config(config)
    return _write_json(validated, path)


def validate_run_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated copy of the version-1 experiment configuration."""
    data = _require_exact_mapping(
        config,
        {
            "schema_version",
            "experiment_id",
            "description",
            "control_dt_seconds",
            "horizon_steps",
            "paired_trials",
            "trials_per_condition",
            "methods",
            "conditions",
            "parameters",
        },
        "run config",
    )
    _require_schema(data, RUN_CONFIG_SCHEMA_VERSION, "run config")
    _require_slug(data["experiment_id"], "config.experiment_id")
    _require_nonempty_string(data["description"], "config.description")
    _require_finite_positive(data["control_dt_seconds"], "config.control_dt_seconds")
    _require_positive_int(data["horizon_steps"], "config.horizon_steps")
    if not isinstance(data["paired_trials"], bool):
        raise TypeError("config.paired_trials must be boolean")
    _require_positive_int(data["trials_per_condition"], "config.trials_per_condition")
    _validate_slug_list(data["methods"], "config.methods")
    _validate_slug_list(data["conditions"], "config.conditions")
    if not isinstance(data["parameters"], Mapping):
        raise TypeError("config.parameters must be a mapping")
    _validate_json_value(dict(data["parameters"]), "config.parameters")
    return data


def write_seeds(seeds: Mapping[str, Any], path: str | os.PathLike[str]) -> str:
    """Validate and canonically write named RNG and pairing metadata."""
    validated = validate_seeds(seeds)
    return _write_json(validated, path)


def validate_seeds(seeds: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated copy of version-1 seed metadata."""
    data = _require_exact_mapping(
        seeds,
        {"schema_version", "root_seed", "folds", "named_streams", "scenario_tapes", "pairing_id"},
        "seeds",
    )
    _require_schema(data, SEEDS_SCHEMA_VERSION, "seeds")
    _require_uint32(data["root_seed"], "seeds.root_seed")
    if not isinstance(data["folds"], list) or not data["folds"]:
        raise ValueError("seeds.folds must be a nonempty list")
    folds = [_require_uint32(value, "seeds.folds[]") for value in data["folds"]]
    if folds != sorted(folds) or len(folds) != len(set(folds)):
        raise ValueError("seeds.folds must be sorted and unique")
    if not isinstance(data["named_streams"], Mapping) or not data["named_streams"]:
        raise ValueError("seeds.named_streams must be a nonempty mapping")
    for name, value in data["named_streams"].items():
        _require_slug(name, "seeds.named_streams key")
        _require_uint32(value, f"seeds.named_streams.{name}")
    if not isinstance(data["scenario_tapes"], list) or not data["scenario_tapes"]:
        raise ValueError("seeds.scenario_tapes must be a nonempty list")
    tape_records = [_validate_seed_tape_record(record) for record in data["scenario_tapes"]]
    keys = [(record["condition"], record["fold"]) for record in tape_records]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("seeds.scenario_tapes condition/fold keys must be sorted and unique")
    if any(record["fold"] not in folds for record in tape_records):
        raise ValueError("every scenario-tape fold must appear in seeds.folds")
    if {record["fold"] for record in tape_records} != set(folds):
        raise ValueError("every seeds.folds entry must have at least one scenario-tape mapping")
    _require_unambiguous_tape_paths(tape_records, include_file_digest=False)
    _require_slug(data["pairing_id"], "seeds.pairing_id")
    return data


def validate_trace_scenario_binding(
    trace: ImmutableTrace, *, condition: str, fold: int, seeds: Mapping[str, Any]
) -> dict[str, Any]:
    """Require a trace to bind to the unique mapped tape for its condition and paired fold."""
    if not isinstance(trace, ImmutableTrace):
        raise TypeError("trace must be an ImmutableTrace")
    trace.validate()
    condition_name = _require_slug(condition, "trace condition")
    fold_value = _require_uint32(fold, "trace fold")
    seed_data = validate_seeds(seeds)
    matches = [
        record
        for record in seed_data["scenario_tapes"]
        if record["condition"] == condition_name and record["fold"] == fold_value
    ]
    if len(matches) != 1:
        raise ValueError("trace condition/fold must have exactly one scenario-tape mapping")
    record = matches[0]
    if str(trace.scenario_tape_sha256) != record["content_sha256"]:
        raise ValueError("trace semantic scenario-tape digest does not match its mapping")
    return record


def collect_provenance(repository: str | os.PathLike[str]) -> dict[str, Any]:
    """Collect git, runtime, device, package, and pinned video-backend provenance.

    Collection is best effort only for optional hardware fields.  The resulting object always
    passes :func:`validate_provenance`; absent GPUs are represented by an empty list, never by an
    invented device.
    """
    root = Path(repository).resolve()
    commit = _run_text(["git", "rev-parse", "HEAD"], cwd=root)
    branch = _run_text(["git", "branch", "--show-current"], cwd=root)
    dirty = bool(_run_text(["git", "status", "--porcelain"], cwd=root))
    import imageio_ffmpeg

    encoder = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    encoder_information = _run_text([str(encoder), "-version"]).splitlines()
    encoder_version = encoder_information[0]
    codec_library_version = next(
        (line.strip() for line in encoder_information if line.strip().startswith("libavcodec")),
        "unavailable",
    )
    encoder_digest = file_sha256(encoder)
    gpus = _query_gpus()

    jax_data = _query_jax_runtime()
    packages: dict[str, str] = {}
    for package in ("crazyflow", "numpy", "jax", "jaxlib", "imageio", "imageio-ffmpeg"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "unavailable"

    data: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "git": {"commit": commit, "branch": branch or "detached", "dirty": dirty},
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "hardware": {"cpu": platform.processor() or platform.machine(), "gpus": gpus},
        "jax": {
            "version": jax_data["version"],
            "jaxlib_version": jax_data["jaxlib_version"],
            "backend": jax_data["backend"],
            "devices": jax_data["devices"],
        },
        "packages": packages,
        "video": {
            "backend": "imageio-ffmpeg",
            "package_version": imageio_ffmpeg.__version__,
            "encoder_executable": str(encoder),
            "encoder_sha256": encoder_digest,
            "encoder_version": encoder_version,
            "codec": "libx264",
            "codec_library_version": codec_library_version,
        },
    }
    return validate_provenance(data)


def write_provenance(provenance: Mapping[str, Any], path: str | os.PathLike[str]) -> str:
    """Validate and canonically write software/hardware provenance."""
    validated = validate_provenance(provenance)
    return _write_json(validated, path)


def validate_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated copy of version-1 provenance metadata."""
    data = _require_exact_mapping(
        provenance,
        {"schema_version", "git", "runtime", "hardware", "jax", "packages", "video"},
        "provenance",
    )
    _require_schema(data, PROVENANCE_SCHEMA_VERSION, "provenance")
    git = _require_exact_mapping(data["git"], {"commit", "branch", "dirty"}, "provenance.git")
    if not isinstance(git["commit"], str) or not _GIT_COMMIT_PATTERN.fullmatch(git["commit"]):
        raise ValueError("provenance.git.commit must be a lowercase 40-character git hash")
    _require_nonempty_string(git["branch"], "provenance.git.branch")
    if not isinstance(git["dirty"], bool):
        raise TypeError("provenance.git.dirty must be boolean")

    runtime = _require_exact_mapping(
        data["runtime"], {"python", "implementation", "platform", "machine"}, "provenance.runtime"
    )
    for name, value in runtime.items():
        _require_nonempty_string(value, f"provenance.runtime.{name}")

    hardware = _require_exact_mapping(data["hardware"], {"cpu", "gpus"}, "provenance.hardware")
    _require_nonempty_string(hardware["cpu"], "provenance.hardware.cpu")
    if not isinstance(hardware["gpus"], list):
        raise TypeError("provenance.hardware.gpus must be a list")
    gpu_keys = {"index", "name", "driver_version", "memory_total_bytes", "uuid"}
    for index, gpu_value in enumerate(hardware["gpus"]):
        gpu = _require_exact_mapping(gpu_value, gpu_keys, f"provenance.hardware.gpus[{index}]")
        _require_nonnegative_int(gpu["index"], f"gpu[{index}].index")
        for key in ("name", "driver_version", "uuid"):
            _require_nonempty_string(gpu[key], f"gpu[{index}].{key}")
        _require_positive_int(gpu["memory_total_bytes"], f"gpu[{index}].memory_total_bytes")

    jax_data = _require_exact_mapping(
        data["jax"], {"version", "jaxlib_version", "backend", "devices"}, "provenance.jax"
    )
    for name in ("version", "jaxlib_version", "backend"):
        _require_nonempty_string(jax_data[name], f"provenance.jax.{name}")
    if not isinstance(jax_data["devices"], list) or not jax_data["devices"]:
        raise ValueError("provenance.jax.devices must be a nonempty list")
    for device in jax_data["devices"]:
        _require_nonempty_string(device, "provenance.jax.devices[]")

    if not isinstance(data["packages"], Mapping) or not data["packages"]:
        raise ValueError("provenance.packages must be a nonempty mapping")
    for name, version in data["packages"].items():
        _require_nonempty_string(name, "provenance.packages key")
        _require_nonempty_string(version, f"provenance.packages.{name}")

    video = _require_exact_mapping(
        data["video"],
        {
            "backend",
            "package_version",
            "encoder_executable",
            "encoder_sha256",
            "encoder_version",
            "codec",
            "codec_library_version",
        },
        "provenance.video",
    )
    if video["backend"] != "imageio-ffmpeg":
        raise ValueError("provenance.video.backend must be imageio-ffmpeg")
    if video["package_version"] != "0.6.0":
        raise ValueError("provenance.video.package_version must match the pinned 0.6.0 backend")
    for name in ("encoder_executable", "encoder_version", "codec", "codec_library_version"):
        _require_nonempty_string(video[name], f"provenance.video.{name}")
    _require_sha256(video["encoder_sha256"], "provenance.video.encoder_sha256")
    if video["codec"] != "libx264":
        raise ValueError("provenance.video.codec must be libx264")
    return data


def write_events(
    events: Sequence[ArtifactEvent],
    path: str | os.PathLike[str],
    *,
    trace: ImmutableTrace | None = None,
) -> str:
    """Atomically write ordered canonical JSONL events and return the file digest."""
    validated = _validate_events(events, trace=trace)
    lines = [_canonical_json_bytes(_event_mapping(event)).rstrip(b"\n") for event in validated]
    destination = _destination(path, suffix=".jsonl", overwrite=False)
    _atomic_write_bytes(destination, b"\n".join(lines) + b"\n", overwrite=False)
    return file_sha256(destination)


def load_events(
    path: str | os.PathLike[str], *, trace: ImmutableTrace | None = None
) -> tuple[ArtifactEvent, ...]:
    """Load strict JSONL events, rejecting blank, oversized, malformed, or extra-key records."""
    source = Path(path)
    if source.suffix.lower() != ".jsonl":
        raise ValueError("events path must end in .jsonl")
    if source.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError("events file exceeds size limit")
    events: list[ArtifactEvent] = []
    with source.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if len(line) > _MAX_JSONL_LINE_BYTES:
                raise ValueError(f"events line {line_number} exceeds size limit")
            if not line.endswith(b"\n") or not line.strip():
                raise ValueError(f"events line {line_number} is blank or unterminated")
            try:
                raw = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"events line {line_number} is invalid JSON") from error
            data = _require_exact_mapping(
                raw,
                {
                    "schema_version",
                    "sequence",
                    "step",
                    "time_seconds",
                    "category",
                    "name",
                    "severity",
                    "snapshot_version",
                    "model_version",
                    "details",
                },
                f"events line {line_number}",
            )
            _require_schema(data, EVENT_SCHEMA_VERSION, f"events line {line_number}")
            events.append(
                ArtifactEvent(
                    sequence=data["sequence"],
                    step=data["step"],
                    time_seconds=data["time_seconds"],
                    category=data["category"],
                    name=data["name"],
                    severity=data["severity"],
                    snapshot_version=data["snapshot_version"],
                    model_version=data["model_version"],
                    details=data["details"],
                )
            )
    return _validate_events(events, trace=trace)


def derive_metrics(trace: ImmutableTrace) -> dict[str, Any]:
    """Derive all version-1 scalar metrics directly from an immutable trace."""
    if not isinstance(trace, ImmutableTrace):
        raise TypeError("trace must be an ImmutableTrace")
    trace.validate()
    executed = np.asarray(trace.executed_control, dtype=np.bool_)
    intervention = np.linalg.norm(
        trace.applied_control[executed] - trace.nominal_control[executed], axis=1
    )
    violation = np.any(trace.hard_barriers < 0.0, axis=1)
    selected = trace.selected_policy[executed]
    valid_pair = (selected[1:] >= 0) & (selected[:-1] >= 0)
    switches = np.count_nonzero(valid_pair & (selected[1:] != selected[:-1]))
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "trace_content_sha256": trace.content_sha256,
        "scenario_tape_sha256": str(trace.scenario_tape_sha256),
        "steps": trace.steps,
        "duration_seconds": float(trace.time[-1] - trace.time[0]),
        "minimum_hard_margin": float(np.min(trace.hard_barriers)),
        "violation_steps": int(np.count_nonzero(violation)),
        "contact_steps": int(np.count_nonzero(trace.contact)),
        "degraded_steps": int(np.count_nonzero(trace.degraded[executed])),
        "failure_steps": int(np.count_nonzero(trace.failure)),
        "mean_intervention_norm": float(np.mean(intervention)),
        "maximum_intervention_norm": float(np.max(intervention)),
        "policy_switches": int(switches),
        "maximum_solver_kkt_residual": float(np.max(trace.solver_kkt_residual[executed])),
        "minimum_postcheck_residual": float(np.min(trace.postcheck_residual[executed])),
    }


def write_metrics(trace: ImmutableTrace, path: str | os.PathLike[str]) -> str:
    """Derive, validate, and write ``metrics.json``."""
    return _write_json(validate_metrics(derive_metrics(trace), trace=trace), path)


def validate_metrics(
    metrics: Mapping[str, Any], *, trace: ImmutableTrace | None = None
) -> dict[str, Any]:
    """Validate metric schema and optionally require exact agreement with a trace."""
    keys = {
        "schema_version",
        "trace_content_sha256",
        "scenario_tape_sha256",
        "steps",
        "duration_seconds",
        "minimum_hard_margin",
        "violation_steps",
        "contact_steps",
        "degraded_steps",
        "failure_steps",
        "mean_intervention_norm",
        "maximum_intervention_norm",
        "policy_switches",
        "maximum_solver_kkt_residual",
        "minimum_postcheck_residual",
    }
    data = _require_exact_mapping(metrics, keys, "metrics")
    _require_schema(data, METRICS_SCHEMA_VERSION, "metrics")
    _require_sha256(data["trace_content_sha256"], "metrics.trace_content_sha256")
    _require_sha256(data["scenario_tape_sha256"], "metrics.scenario_tape_sha256")
    for name in (
        "steps",
        "violation_steps",
        "contact_steps",
        "degraded_steps",
        "failure_steps",
        "policy_switches",
    ):
        _require_nonnegative_int(data[name], f"metrics.{name}")
    if data["steps"] < 2:
        raise ValueError("metrics.steps must be at least two")
    for name in (
        "duration_seconds",
        "mean_intervention_norm",
        "maximum_intervention_norm",
        "maximum_solver_kkt_residual",
    ):
        _require_finite_nonnegative(data[name], f"metrics.{name}")
    for name in ("minimum_hard_margin", "minimum_postcheck_residual"):
        _require_finite(data[name], f"metrics.{name}")
    if data["mean_intervention_norm"] > data["maximum_intervention_norm"]:
        raise ValueError("mean intervention cannot exceed maximum intervention")
    if trace is not None:
        expected = derive_metrics(trace)
        _require_json_numeric_agreement(data, expected, "metrics")
    return data


def load_metrics(
    path: str | os.PathLike[str], *, trace: ImmutableTrace | None = None
) -> dict[str, Any]:
    """Load strict metrics JSON and optionally cross-check it against a trace."""
    return validate_metrics(_read_json(path), trace=trace)


def derive_timing(
    trace: ImmutableTrace,
    *,
    compile_seconds: Mapping[str, float],
    deadline_seconds: Mapping[str, float],
) -> dict[str, Any]:
    """Build raw timing samples and fixed-protocol tail summaries from a trace."""
    names = [str(value) for value in trace.latency_names]
    if set(compile_seconds) != set(names) or set(deadline_seconds) != set(names):
        raise ValueError("compile_seconds and deadline_seconds must exactly match latency_names")
    components: dict[str, Any] = {}
    for index, name in enumerate(names):
        compile_time = _require_finite_nonnegative(compile_seconds[name], f"compile_seconds.{name}")
        deadline = _require_finite_positive(deadline_seconds[name], f"deadline_seconds.{name}")
        samples = np.asarray(
            trace.component_latency_seconds[trace.executed_control, index], dtype=np.float64
        )
        components[name] = {
            "compile_seconds": compile_time,
            "deadline_seconds": deadline,
            "samples_seconds": [float(value) for value in samples],
            "count": int(samples.size),
            "median_seconds": float(np.percentile(samples, 50.0, method="linear")),
            "p95_seconds": float(np.percentile(samples, 95.0, method="linear")),
            "p99_seconds": float(np.percentile(samples, 99.0, method="linear")),
            "worst_seconds": float(np.max(samples)),
            "deadline_misses": int(np.count_nonzero(samples > deadline)),
        }
    return {
        "schema_version": TIMING_SCHEMA_VERSION,
        "trace_content_sha256": trace.content_sha256,
        "units": "seconds",
        "percentile_method": "numpy-linear",
        "warm_execution_excludes_compilation": True,
        "components": components,
    }


def write_timing(
    trace: ImmutableTrace,
    path: str | os.PathLike[str],
    *,
    compile_seconds: Mapping[str, float],
    deadline_seconds: Mapping[str, float],
) -> str:
    """Derive, validate, and write raw timing samples and tail summaries."""
    timing = derive_timing(
        trace, compile_seconds=compile_seconds, deadline_seconds=deadline_seconds
    )
    return _write_json(validate_timing(timing, trace=trace), path)


def validate_timing(
    timing: Mapping[str, Any], *, trace: ImmutableTrace | None = None
) -> dict[str, Any]:
    """Validate timing samples, recompute every summary, and optionally bind to a trace."""
    data = _require_exact_mapping(
        timing,
        {
            "schema_version",
            "trace_content_sha256",
            "units",
            "percentile_method",
            "warm_execution_excludes_compilation",
            "components",
        },
        "timing",
    )
    _require_schema(data, TIMING_SCHEMA_VERSION, "timing")
    _require_sha256(data["trace_content_sha256"], "timing.trace_content_sha256")
    if data["units"] != "seconds" or data["percentile_method"] != "numpy-linear":
        raise ValueError("timing units/method must be seconds/numpy-linear")
    if data["warm_execution_excludes_compilation"] is not True:
        raise ValueError("warm execution must explicitly exclude compilation")
    if not isinstance(data["components"], Mapping) or not data["components"]:
        raise ValueError("timing.components must be a nonempty mapping")
    component_keys = {
        "compile_seconds",
        "deadline_seconds",
        "samples_seconds",
        "count",
        "median_seconds",
        "p95_seconds",
        "p99_seconds",
        "worst_seconds",
        "deadline_misses",
    }
    for name, raw_component in data["components"].items():
        _require_slug(name, "timing component name")
        component = _require_exact_mapping(
            raw_component, component_keys, f"timing.components.{name}"
        )
        _require_finite_nonnegative(
            component["compile_seconds"], f"timing.components.{name}.compile_seconds"
        )
        deadline = _require_finite_positive(
            component["deadline_seconds"], f"timing.components.{name}.deadline_seconds"
        )
        if not isinstance(component["samples_seconds"], list) or not component["samples_seconds"]:
            raise ValueError(f"timing component {name!r} must contain raw samples")
        samples = np.asarray(component["samples_seconds"], dtype=np.float64)
        if samples.ndim != 1 or not np.all(np.isfinite(samples)) or np.any(samples < 0.0):
            raise ValueError(f"timing component {name!r} has invalid samples")
        expected = {
            "count": int(samples.size),
            "median_seconds": float(np.percentile(samples, 50.0, method="linear")),
            "p95_seconds": float(np.percentile(samples, 95.0, method="linear")),
            "p99_seconds": float(np.percentile(samples, 99.0, method="linear")),
            "worst_seconds": float(np.max(samples)),
            "deadline_misses": int(np.count_nonzero(samples > deadline)),
        }
        _require_json_numeric_agreement(
            {key: component[key] for key in expected}, expected, f"timing.components.{name}"
        )
    if trace is not None:
        if data["trace_content_sha256"] != trace.content_sha256:
            raise ValueError("timing trace digest does not match trace")
        names = [str(value) for value in trace.latency_names]
        if set(data["components"]) != set(names):
            raise ValueError("timing components must match trace latency_names")
        for index, name in enumerate(names):
            samples = np.asarray(data["components"][name]["samples_seconds"], dtype=np.float64)
            if not np.array_equal(
                samples, trace.component_latency_seconds[trace.executed_control, index]
            ):
                raise ValueError(f"timing samples for {name!r} do not match the trace")
    return data


def load_timing(
    path: str | os.PathLike[str], *, trace: ImmutableTrace | None = None
) -> dict[str, Any]:
    """Load strict timing JSON and optionally cross-check its raw samples against a trace."""
    return validate_timing(_read_json(path), trace=trace)


AGGREGATE_COLUMNS = (
    "method",
    "condition",
    "seed",
    "scenario_tape_sha256",
    "trace_content_sha256",
    "steps",
    "duration_seconds",
    "minimum_hard_margin",
    "violation_steps",
    "contact_steps",
    "degraded_steps",
    "failure_steps",
    "mean_intervention_norm",
    "maximum_intervention_norm",
    "policy_switches",
    "maximum_solver_kkt_residual",
    "minimum_postcheck_residual",
)


def aggregate_row(
    method: str, condition: str, seed: int, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind one validated metrics record to its paired method/condition/seed key."""
    _require_slug(method, "aggregate.method")
    _require_slug(condition, "aggregate.condition")
    _require_uint32(seed, "aggregate.seed")
    values = validate_metrics(metrics)
    return {
        "method": method,
        "condition": condition,
        "seed": seed,
        **{name: values[name] for name in AGGREGATE_COLUMNS[3:]},
    }


def write_paired_metrics_csv(
    rows: Sequence[Mapping[str, Any]], path: str | os.PathLike[str]
) -> str:
    """Write deterministic strict aggregate CSV sorted by condition, seed, and method."""
    validated = _validate_aggregate_rows(rows)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=AGGREGATE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(
        validated, key=lambda item: (item["condition"], item["seed"], item["method"])
    ):
        writer.writerow({key: _csv_value(row[key]) for key in AGGREGATE_COLUMNS})
    destination = _destination(path, suffix=".csv", overwrite=False)
    _atomic_write_bytes(destination, output.getvalue().encode(), overwrite=False)
    return file_sha256(destination)


def load_paired_metrics_csv(path: str | os.PathLike[str]) -> tuple[dict[str, Any], ...]:
    """Load strict aggregate CSV with fixed headers, types, ordering, and unique paired keys."""
    source = Path(path)
    if source.suffix.lower() != ".csv":
        raise ValueError("paired metrics path must end in .csv")
    if source.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError("paired metrics CSV exceeds size limit")
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != AGGREGATE_COLUMNS:
            raise ValueError("paired metrics CSV header does not match schema")
        rows: list[dict[str, Any]] = []
        int_columns = {
            "seed",
            "steps",
            "violation_steps",
            "contact_steps",
            "degraded_steps",
            "failure_steps",
            "policy_switches",
        }
        float_columns = (
            set(AGGREGATE_COLUMNS)
            - int_columns
            - {"method", "condition", "scenario_tape_sha256", "trace_content_sha256"}
        )
        for row in reader:
            converted: dict[str, Any] = dict(row)
            try:
                for name in int_columns:
                    converted[name] = int(row[name])
                for name in float_columns:
                    converted[name] = float(row[name])
            except ValueError as error:
                raise ValueError("paired metrics CSV contains an invalid number") from error
            rows.append(converted)
    return _validate_aggregate_rows(rows)


def write_confidence_intervals(
    rows: Sequence[Mapping[str, Any]],
    path: str | os.PathLike[str],
    *,
    confidence_level: float = 0.95,
) -> str:
    """Write descriptive intervals, explicitly unavailable for groups with fewer than two runs.

    This foundation intentionally does not choose a scientific paired-test protocol.  Final runs
    must replace ``method summaries`` with the predeclared paired analysis where appropriate.
    """
    validated = _validate_aggregate_rows(rows)
    level = _require_finite(confidence_level, "confidence_level")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    payload = _confidence_interval_payload(validated, level)
    return _write_json(payload, path)


def _confidence_interval_payload(
    rows: Sequence[Mapping[str, Any]], confidence_level: float
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["condition"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (method, condition), group in sorted(groups.items()):
        margins = np.asarray([row["minimum_hard_margin"] for row in group], dtype=np.float64)
        failures = np.asarray([row["failure_steps"] for row in group], dtype=np.float64)
        summary: dict[str, Any] = {
            "method": method,
            "condition": condition,
            "count": len(group),
            "minimum_hard_margin_mean": float(np.mean(margins)),
            "failure_steps_mean": float(np.mean(failures)),
        }
        if len(group) < 2:
            summary.update(
                {
                    "interval_available": False,
                    "unavailable_reason": "at least two paired trials are required",
                    "minimum_hard_margin_mean_interval": None,
                    "failure_steps_mean_interval": None,
                }
            )
        else:
            # Normal intervals are only descriptive here and are labeled as such in the schema.
            z_value = _normal_quantile(0.5 + confidence_level / 2.0)
            margin_half = z_value * float(np.std(margins, ddof=1)) / math.sqrt(len(group))
            failure_half = z_value * float(np.std(failures, ddof=1)) / math.sqrt(len(group))
            summary.update(
                {
                    "interval_available": True,
                    "unavailable_reason": None,
                    "minimum_hard_margin_mean_interval": [
                        float(np.mean(margins) - margin_half),
                        float(np.mean(margins) + margin_half),
                    ],
                    "failure_steps_mean_interval": [
                        float(np.mean(failures) - failure_half),
                        float(np.mean(failures) + failure_half),
                    ],
                }
            )
        summaries.append(summary)
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "confidence_level": confidence_level,
        "interval_method": "descriptive-normal-not-a-paired-superiority-test",
        "summaries": summaries,
    }


def write_aggregate_report(
    rows: Sequence[Mapping[str, Any]], path: str | os.PathLike[str], *, scientific_evidence: bool
) -> str:
    """Write a compact aggregate Markdown report without inferring superiority."""
    validated = _validate_aggregate_rows(rows)
    if not isinstance(scientific_evidence, bool):
        raise TypeError("scientific_evidence must be boolean")
    payload = _aggregate_report_text(validated, scientific_evidence=scientific_evidence)
    destination = _destination(path, suffix=".md", overwrite=False)
    _atomic_write_bytes(destination, payload.encode(), overwrite=False)
    return file_sha256(destination)


def _aggregate_report_text(rows: Sequence[Mapping[str, Any]], *, scientific_evidence: bool) -> str:
    heading = "# DA-PLCBF aggregate report"
    status = (
        "Scientific evidence run; interpret only within the claim boundary below."
        if scientific_evidence
        else "Synthetic schema/replay smoke only; this is not scientific or safety evidence."
    )
    lines = [
        heading,
        "",
        status,
        "",
        f"> {SAFETY_CLAIM_BOUNDARY}",
        "",
        "This table is descriptive. It does not assert statistical superiority.",
        "",
        "| Method | Condition | Seed | Minimum hard margin | Failures | Degraded |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["condition"], item["seed"], item["method"])):
        lines.append(
            "| {method} | {condition} | {seed} | {margin:.9g} | {failure} | {degraded} |".format(
                method=row["method"],
                condition=row["condition"],
                seed=row["seed"],
                margin=row["minimum_hard_margin"],
                failure=row["failure_steps"],
                degraded=row["degraded_steps"],
            )
        )
    return "\n".join(lines) + "\n"


def write_manifest(
    run_directory: str | os.PathLike[str],
    *,
    run_id: str,
    status: str,
    scientific_evidence: bool,
    replay_command: str,
    video_records: Sequence[Mapping[str, Any]],
    created_utc: str | None = None,
) -> str:
    """Finalize ``manifest.json`` after every evidence file except ``SHA256SUMS`` exists."""
    root = Path(run_directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError("run directory does not exist")
    _require_slug(run_id, "manifest.run_id")
    if root.name != run_id:
        raise ValueError("run directory basename must equal run_id")
    if status not in {"incomplete", "complete", "synthetic-smoke"}:
        raise ValueError("manifest.status is invalid")
    if not isinstance(scientific_evidence, bool):
        raise TypeError("scientific_evidence must be boolean")
    if scientific_evidence != (status == "complete"):
        raise ValueError("only complete runs may be labeled scientific evidence")
    _require_nonempty_string(replay_command, "manifest.replay_command")
    timestamp = created_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _validate_utc_timestamp(timestamp)
    for name in ("manifest.json", "SHA256SUMS"):
        if (root / name).exists():
            raise FileExistsError(root / name)

    config = validate_run_config(_read_json(root / "config.json"))
    provenance = validate_provenance(_read_json(root / "provenance.json"))
    seeds = validate_seeds(_read_json(root / "seeds.json"))
    expected_tape_keys = {
        (condition, fold) for condition in config["conditions"] for fold in seeds["folds"]
    }
    mapped_tape_keys = {(record["condition"], record["fold"]) for record in seeds["scenario_tapes"]}
    if mapped_tape_keys != expected_tape_keys:
        raise ValueError("seed scenario-tape mapping must cover every configured condition/fold")
    del provenance

    file_records = _scan_file_records(root)
    tape_records = _build_manifest_tape_records(root, seeds["scenario_tapes"], file_records)
    videos = sorted(
        (_validate_video_record(record) for record in video_records), key=lambda item: item["path"]
    )
    if scientific_evidence:
        validate_campaign_visual_reviews(root, videos, require_all=True, require_final_core=True)
    indexed_paths = {record["path"] for record in file_records}
    for record in videos:
        if record["path"] not in indexed_paths or record["source_trace_path"] not in indexed_paths:
            raise ValueError("video and source trace must both be indexed manifest files")
        if _record_for_path(file_records, record["path"])["sha256"] != record["sha256"]:
            raise ValueError("video record SHA-256 does not match the video file")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "crazyflow-da-plcbf",
        "run_id": run_id,
        "created_utc": timestamp,
        "status": status,
        "scientific_evidence": scientific_evidence,
        "claim_boundary": SAFETY_CLAIM_BOUNDARY,
        "hash_algorithm": "sha256",
        "replay_command": replay_command,
        "config_sha256": file_sha256(root / "config.json"),
        "provenance_sha256": file_sha256(root / "provenance.json"),
        "seeds_sha256": file_sha256(root / "seeds.json"),
        "scenario_tapes": tape_records,
        "files": file_records,
        "videos": videos,
    }
    validated = validate_manifest(manifest, run_directory=root)
    return _write_json(validated, root / "manifest.json")


def validate_manifest(
    manifest: Mapping[str, Any], *, run_directory: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """Validate manifest schema and optionally verify every indexed file."""
    data = _require_exact_mapping(
        manifest,
        {
            "schema_version",
            "artifact_type",
            "run_id",
            "created_utc",
            "status",
            "scientific_evidence",
            "claim_boundary",
            "hash_algorithm",
            "replay_command",
            "config_sha256",
            "provenance_sha256",
            "seeds_sha256",
            "scenario_tapes",
            "files",
            "videos",
        },
        "manifest",
    )
    _require_schema(data, MANIFEST_SCHEMA_VERSION, "manifest")
    if data["artifact_type"] != "crazyflow-da-plcbf":
        raise ValueError("manifest.artifact_type is invalid")
    _require_slug(data["run_id"], "manifest.run_id")
    _validate_utc_timestamp(data["created_utc"])
    if data["status"] not in {"incomplete", "complete", "synthetic-smoke"}:
        raise ValueError("manifest.status is invalid")
    if not isinstance(data["scientific_evidence"], bool):
        raise TypeError("manifest.scientific_evidence must be boolean")
    if data["scientific_evidence"] != (data["status"] == "complete"):
        raise ValueError("manifest status/evidence label is inconsistent")
    if data["claim_boundary"] != SAFETY_CLAIM_BOUNDARY:
        raise ValueError("manifest claim boundary must match the finite-horizon contract")
    if data["hash_algorithm"] != "sha256":
        raise ValueError("manifest hash_algorithm must be sha256")
    _require_nonempty_string(data["replay_command"], "manifest.replay_command")
    for name in ("config_sha256", "provenance_sha256", "seeds_sha256"):
        _require_sha256(data[name], f"manifest.{name}")
    if not isinstance(data["scenario_tapes"], list) or not data["scenario_tapes"]:
        raise ValueError("manifest.scenario_tapes must be a nonempty list")
    tapes = [_validate_manifest_tape_record(record) for record in data["scenario_tapes"]]
    tape_keys = [(record["condition"], record["fold"]) for record in tapes]
    if tape_keys != sorted(tape_keys) or len(tape_keys) != len(set(tape_keys)):
        raise ValueError("manifest scenario-tape keys must be sorted and unique")
    _require_unambiguous_tape_paths(tapes, include_file_digest=True)

    if not isinstance(data["files"], list) or not data["files"]:
        raise ValueError("manifest.files must be a nonempty list")
    files = [_validate_file_record(record) for record in data["files"]]
    paths = [record["path"] for record in files]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("manifest file paths must be sorted and unique")
    if not isinstance(data["videos"], list):
        raise TypeError("manifest.videos must be a list")
    videos = [_validate_video_record(record) for record in data["videos"]]
    video_paths = [record["path"] for record in videos]
    if video_paths != sorted(video_paths) or len(video_paths) != len(set(video_paths)):
        raise ValueError("manifest video records must be sorted and unique")
    indexed_video_paths = {record["path"] for record in files if record["role"] == "video"}
    if set(video_paths) != indexed_video_paths:
        raise ValueError("manifest video records must cover every and only indexed video file")
    if data["status"] in {"complete", "synthetic-smoke"} and not videos:
        raise ValueError("complete and synthetic-smoke manifests require at least one video")

    if run_directory is not None:
        root = Path(run_directory).resolve()
        if root.name != data["run_id"]:
            raise ValueError("manifest run_id does not match its directory")
        current = _scan_file_records(root)
        if current != files:
            raise ValueError("manifest file inventory does not match the run directory")
        bindings = {
            "config.json": data["config_sha256"],
            "provenance.json": data["provenance_sha256"],
            "seeds.json": data["seeds_sha256"],
        }
        for path, digest in bindings.items():
            if _record_for_path(files, path)["sha256"] != digest:
                raise ValueError(f"manifest binding for {path} does not match file inventory")
        rebuilt_tapes = _build_manifest_tape_records(
            root,
            [
                {key: record[key] for key in ("condition", "fold", "path", "content_sha256")}
                for record in tapes
            ],
            files,
        )
        if rebuilt_tapes != tapes:
            raise ValueError("manifest scenario-tape records do not match files on disk")
    return data


def write_sha256sums(run_directory: str | os.PathLike[str]) -> str:
    """Write an exact sorted inventory of every run file except ``SHA256SUMS`` itself."""
    root = Path(run_directory).resolve()
    destination = root / "SHA256SUMS"
    if destination.exists():
        raise FileExistsError(destination)
    records = _scan_checksum_records(root)
    payload = "".join(f"{digest}  {path}\n" for path, digest in records).encode()
    _atomic_write_bytes(destination, payload, overwrite=False)
    return file_sha256(destination)


def verify_sha256sums(run_directory: str | os.PathLike[str]) -> None:
    """Reject missing, extra, duplicate, unsafe-path, or content-mismatched checksum entries."""
    root = Path(run_directory).resolve()
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    if checksum_path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError("SHA256SUMS exceeds size limit")
    entries: list[tuple[str, str]] = []
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        if not re.fullmatch(r"[0-9a-f]{64}  [^\r\n]+", line):
            raise ValueError(f"invalid SHA256SUMS line {line_number}")
        digest, relative = line.split("  ", maxsplit=1)
        _validate_relative_path(relative)
        if relative == "SHA256SUMS":
            raise ValueError("SHA256SUMS must not hash itself")
        entries.append((relative, digest))
    if entries != sorted(entries) or len(entries) != len({path for path, _ in entries}):
        raise ValueError("SHA256SUMS paths must be sorted and unique")
    expected = [(path, digest) for path, digest in _scan_checksum_records(root)]
    if entries != expected:
        raise ValueError("SHA256SUMS inventory or content digest mismatch")


def validate_run_artifacts(
    run_directory: str | os.PathLike[str],
    *,
    verify_replay: bool = False,
    _validated_campaign: Any | None = None,
) -> dict[str, Any]:
    """Validate a complete run across hashes, schemas, trace metrics, video, and optional replay."""
    root = Path(run_directory).resolve()
    verify_sha256sums(root)
    manifest = validate_manifest(_read_json(root / "manifest.json"), run_directory=root)
    config = validate_run_config(_read_json(root / "config.json"))
    provenance = validate_provenance(_read_json(root / "provenance.json"))
    seeds = validate_seeds(_read_json(root / "seeds.json"))
    if file_sha256(root / "config.json") != manifest["config_sha256"]:
        raise ValueError("config file does not match manifest")
    if file_sha256(root / "provenance.json") != manifest["provenance_sha256"]:
        raise ValueError("provenance file does not match manifest")
    if file_sha256(root / "seeds.json") != manifest["seeds_sha256"]:
        raise ValueError("seeds file does not match manifest")
    seed_tapes = {
        (record["condition"], record["fold"]): record for record in seeds["scenario_tapes"]
    }
    manifest_tapes = {
        (record["condition"], record["fold"]): record for record in manifest["scenario_tapes"]
    }
    expected_tape_keys = {
        (condition, fold) for condition in config["conditions"] for fold in seeds["folds"]
    }
    if set(seed_tapes) != expected_tape_keys or set(manifest_tapes) != expected_tape_keys:
        raise ValueError("scenario-tape mappings must cover every configured condition/fold")
    for key, seed_record in seed_tapes.items():
        manifest_record = manifest_tapes[key]
        if any(
            seed_record[name] != manifest_record[name]
            for name in ("condition", "fold", "path", "content_sha256")
        ):
            raise ValueError("seed and manifest scenario-tape mappings disagree")
    if manifest["scientific_evidence"] and (
        config["paired_trials"] is not True or config["trials_per_condition"] < 100
    ):
        raise ValueError("scientific evidence requires at least 100 paired trials per condition")
    if manifest["scientific_evidence"]:
        from crazyflow.safety.da_plcbf.baselines import MethodID
        from crazyflow.safety.da_plcbf.experiments import REQUIRED_CONDITIONS

        if tuple(config["methods"]) != tuple(item.value for item in MethodID):
            raise ValueError("scientific evidence requires exactly the seven ordered core methods")
        if tuple(config["conditions"]) != REQUIRED_CONDITIONS:
            raise ValueError(
                "scientific evidence requires exactly the four ordered core conditions"
            )
        parameters = config["parameters"]
        if (
            not isinstance(parameters, dict)
            or parameters.get("intended_for_final_claim") is not True
        ):
            raise ValueError("scientific evidence must be predeclared for a final claim")

    method_records: list[dict[str, Any]] = []
    trace_paths = sorted(root.glob("methods/*/*/*/trace.npz"))
    if not trace_paths:
        raise ValueError("run contains no method traces")
    for trace_path in trace_paths:
        relative_parts = trace_path.relative_to(root).parts
        if len(relative_parts) != 5:
            raise ValueError("method trace path does not match method/condition/seed schema")
        _, method, condition, seed_text, filename = relative_parts
        if filename != "trace.npz":
            raise AssertionError("internal trace glob invariant failed")
        _require_slug(method, "method path")
        _require_slug(condition, "condition path")
        try:
            seed = int(seed_text)
        except ValueError as error:
            raise ValueError("method seed directory must be an integer") from error
        _require_uint32(seed, "method seed directory")
        method_dir = trace_path.parent
        expected_names = {"trace.npz", "events.jsonl", "metrics.json", "timing.json"}
        extended_names = {*expected_names, "dashboard_evidence.npz"}
        online_names = {*extended_names, "adaptation_evidence.npz"}
        actual_names = {path.name for path in method_dir.iterdir() if path.is_file()}
        if actual_names not in (expected_names, extended_names, online_names):
            raise ValueError(f"method artifact files do not match schema in {method_dir}")
        trace = load_trace(trace_path)
        validate_trace_scenario_binding(trace, condition=condition, fold=seed, seeds=seeds)
        events = load_events(method_dir / "events.jsonl", trace=trace)
        metrics = load_metrics(method_dir / "metrics.json", trace=trace)
        load_timing(method_dir / "timing.json", trace=trace)
        if "dashboard_evidence.npz" in actual_names:
            from crazyflow.safety.da_plcbf.dashboard_evidence import (
                load_dashboard_evidence,
                validate_dashboard_evidence_binding,
            )

            sidecar = load_dashboard_evidence(method_dir / "dashboard_evidence.npz")
            from crazyflow.safety.da_plcbf.scenarios import load_scenario_tape

            tape_record = seed_tapes[(condition, seed)]
            tape = load_scenario_tape(root / tape_record["path"])
            validate_dashboard_evidence_binding(sidecar, trace, tape, events=events)
        if "adaptation_evidence.npz" in actual_names:
            from crazyflow.safety.da_plcbf.adaptation_evidence import (
                load_adaptation_evidence,
                validate_adaptation_evidence_binding,
            )

            adaptation_evidence = load_adaptation_evidence(method_dir / "adaptation_evidence.npz")
            validate_adaptation_evidence_binding(adaptation_evidence, trace, events)
        method_records.append(
            {
                "method": method,
                "condition": condition,
                "seed": seed,
                "trace": trace,
                "events": events,
                "metrics": metrics,
            }
        )

    if config["trials_per_condition"] != len(seeds["folds"]):
        raise ValueError("trials_per_condition must equal the number of recorded paired folds")
    expected_keys = {
        (method, condition, fold)
        for method in config["methods"]
        for condition in config["conditions"]
        for fold in seeds["folds"]
    }
    actual_keys = {
        (record["method"], record["condition"], record["seed"]) for record in method_records
    }
    outcomes_path = root / "aggregate" / "outcomes.jsonl"
    failed_outcome_keys: set[tuple[str, str, int]] = set()
    if outcomes_path.is_file():
        outcome_keys: set[tuple[str, str, int]] = set()
        successful_outcome_keys: set[tuple[str, str, int]] = set()
        trace_digest_by_key = {
            (record["method"], record["condition"], record["seed"]): record["trace"].content_sha256
            for record in method_records
        }
        for line_number, line in enumerate(outcomes_path.read_text().splitlines(), 1):
            try:
                outcome = json.loads(line)
                assignment = outcome["assignment"]
                key = (assignment["method"], assignment["condition"], int(assignment["fold"]))
                status = outcome["status"]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid campaign outcome line {line_number}") from error
            if key in outcome_keys or key not in expected_keys:
                raise ValueError("campaign outcomes are duplicated or outside the schedule")
            outcome_keys.add(key)
            if status == "complete":
                successful_outcome_keys.add(key)
                if outcome.get("trace_content_sha256") != trace_digest_by_key.get(key):
                    raise ValueError("successful campaign outcome trace digest mismatch")
            elif status == "execution_failure":
                failed_outcome_keys.add(key)
                if outcome.get("trace_content_sha256") is not None:
                    raise ValueError("failed campaign outcome cannot bind a trace")
            else:
                raise ValueError("campaign outcome status is invalid")
        if outcome_keys != expected_keys:
            raise ValueError("campaign outcomes do not cover the configured paired matrix")
        if successful_outcome_keys != actual_keys:
            raise ValueError("successful campaign outcomes and method traces disagree")
    elif actual_keys != expected_keys:
        raise ValueError(
            "method artifacts do not form the configured paired method/condition/fold matrix"
        )
    if len(actual_keys) != len(method_records):
        raise ValueError("method artifact keys are duplicated")
    if manifest["scientific_evidence"] and failed_outcome_keys:
        raise ValueError("scientific evidence cannot contain execution-failure outcomes")
    if outcomes_path.is_file():
        from crazyflow.safety.da_plcbf.campaign_artifacts import (
            validate_persisted_campaign_evidence,
        )

        reconstructed = (
            validate_persisted_campaign_evidence(root)
            if _validated_campaign is None
            else _validated_campaign
        )
        if manifest["scientific_evidence"] is not reconstructed.scientific_claim_eligible:
            raise ValueError(
                "manifest scientific-evidence status disagrees with canonical campaign evidence"
            )
    elif manifest["scientific_evidence"]:
        raise ValueError("scientific evidence requires canonical campaign outcomes")

    rows = load_paired_metrics_csv(root / "aggregate" / "paired_metrics.csv")
    expected_rows = tuple(
        aggregate_row(record["method"], record["condition"], record["seed"], record["metrics"])
        for record in method_records
    )
    if _sort_aggregate_rows(rows) != _sort_aggregate_rows(expected_rows):
        raise ValueError("aggregate CSV does not agree with method metrics")
    _validate_confidence_intervals(
        _read_json(root / "aggregate" / "confidence_intervals.json"), rows=rows
    )
    report = (root / "aggregate" / "report.md").read_text(encoding="utf-8")
    expected_report = _aggregate_report_text(
        rows, scientific_evidence=manifest["scientific_evidence"]
    )
    if report != expected_report:
        raise ValueError("aggregate report does not agree with paired metrics CSV")

    from crazyflow.safety.da_plcbf.dashboard import validate_mp4, verify_dashboard_replay

    video_results = []
    for record in manifest["videos"]:
        result = validate_mp4(
            root / record["path"],
            expected_codec=record["codec"],
            expected_size=(record["width"], record["height"]),
            expected_fps=record["fps"],
            expected_frame_count=record["frame_count"],
        )
        if result.decoded_frames_sha256 != record["decoded_frames_sha256"]:
            raise ValueError("video decoded-frame digest does not match manifest")
        if not math.isclose(result.duration_seconds, record["duration_seconds"], abs_tol=1e-9):
            raise ValueError("video duration does not match manifest")
        if verify_replay or manifest["scientific_evidence"]:
            trace = load_trace(root / record["source_trace_path"])
            if record.get("renderer", "basic-dashboard-v1") == "scientific-dashboard-v1":
                from crazyflow.safety.da_plcbf.dashboard_evidence import load_dashboard_evidence
                from crazyflow.safety.da_plcbf.scenarios import load_scenario_tape
                from crazyflow.safety.da_plcbf.scientific_dashboard import (
                    verify_scientific_dashboard_replay,
                )

                trace_relative = PurePosixPath(record["source_trace_path"])
                if len(trace_relative.parts) != 5:
                    raise ValueError("scientific video source trace path is malformed")
                _, _, condition, fold_text, _ = trace_relative.parts
                method_directory = root / trace_relative.parent
                verify_scientific_dashboard_replay(
                    trace,
                    root / record["path"],
                    tape=load_scenario_tape(
                        root / "scenario_tapes" / condition / f"{int(fold_text)}.npz"
                    ),
                    sidecar=load_dashboard_evidence(method_directory / "dashboard_evidence.npz"),
                    events=load_events(method_directory / "events.jsonl", trace=trace),
                    fps=record["fps"],
                    size=(record["width"], record["height"]),
                )
            else:
                verify_dashboard_replay(
                    trace,
                    root / record["path"],
                    fps=record["fps"],
                    size=(record["width"], record["height"]),
                )
        video_results.append(result)

    if manifest["status"] != "synthetic-smoke":
        review_files = {
            record["path"] for record in manifest["files"] if record["role"] == "visual-review"
        }
        validate_campaign_visual_reviews(
            root,
            manifest["videos"],
            require_all=manifest["scientific_evidence"] or bool(review_files),
            require_final_core=manifest["scientific_evidence"],
        )

    if provenance["video"]["package_version"] != "0.6.0":
        raise ValueError("run did not record the pinned video backend")
    if manifest["scientific_evidence"] and (
        provenance["jax"]["backend"] == "unavailable"
        or any(version == "unavailable" for version in provenance["packages"].values())
        or provenance["video"]["codec_library_version"] == "unavailable"
    ):
        raise ValueError("scientific evidence requires complete JAX and package provenance")
    if manifest["scientific_evidence"] and provenance["git"]["dirty"]:
        raise ValueError("scientific evidence requires a clean committed source tree")
    return {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "scientific_evidence": manifest["scientific_evidence"],
        "method_runs": len(method_records),
        "videos": len(video_results),
        "scenario_tapes": len({record["path"] for record in manifest["scenario_tapes"]}),
        "config": config,
    }


def validate_campaign_visual_reviews(
    run_directory: str | os.PathLike[str],
    video_records: Sequence[Mapping[str, Any]],
    *,
    require_all: bool,
    require_final_core: bool,
) -> tuple[str, ...]:
    """Parse and bind canonical visual reviews to their exact trace, tape, sidecar, and video.

    Development campaigns may carry a reviewed subset while remaining explicitly non-scientific.
    A scientific final-core campaign is stricter: it must contain exactly one scientific-dashboard
    video for ``da_plcbf_full`` in each of the four ordered core conditions, and every video must
    have a passing review.  Synthetic codec/replay smoke artifacts are deliberately validated by
    their separate non-scientific path and never auto-promoted to human/agent visual review.
    """
    root = Path(run_directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not isinstance(require_all, bool) or not isinstance(require_final_core, bool):
        raise TypeError("visual-review requirement flags must be boolean")
    videos = tuple(_validate_video_record(record) for record in video_records)
    if require_final_core:
        _validate_final_core_video_coverage(videos)

    regular_files = _regular_run_files(root, exclude={"manifest.json", "SHA256SUMS"})
    review_files = {
        relative
        for relative, _ in regular_files
        if relative == "visual_review.md" or relative.startswith("visual_reviews/")
    }
    expected_paths: list[str] = []
    resolved_paths: list[Path] = []
    for video in videos:
        stem = PurePosixPath(video["path"]).stem
        per_video = root / "visual_reviews" / f"{stem}.md"
        singleton = root / "visual_review.md"
        if per_video.is_file():
            review_path = per_video
        elif len(videos) == 1 and singleton.is_file():
            review_path = singleton
        else:
            review_path = per_video
        expected_paths.append(review_path.relative_to(root).as_posix())
        resolved_paths.append(review_path)

    if require_all and any(not path.is_file() for path in resolved_paths):
        missing = [
            path.relative_to(root).as_posix() for path in resolved_paths if not path.is_file()
        ]
        raise ValueError(f"missing canonical visual review(s): {','.join(missing)}")
    if require_all and review_files != set(expected_paths):
        raise ValueError("visual-review files must cover every and only rendered video")
    if not require_all:
        if review_files:
            raise ValueError("partial visual-review sets are not allowed")
        return ()

    from crazyflow.safety.da_plcbf.dashboard_evidence import load_dashboard_evidence
    from crazyflow.safety.da_plcbf.scenarios import load_scenario_tape
    from crazyflow.safety.da_plcbf.scientific_dashboard import load_visual_review_record

    seeds = validate_seeds(_read_json(root / "seeds.json"))
    tape_paths = {
        (record["condition"], int(record["fold"])): record["path"]
        for record in seeds["scenario_tapes"]
    }
    reviewed: list[str] = []
    for video, review_path in zip(videos, resolved_paths, strict=True):
        review = load_visual_review_record(review_path)
        if review.disposition != "pass":
            raise ValueError(f"visual review is not passing: {review_path.relative_to(root)}")
        source = PurePosixPath(video["source_trace_path"])
        if len(source.parts) != 5 or source.parts[0] != "methods" or source.name != "trace.npz":
            raise ValueError("visual-review source trace path is malformed")
        condition = source.parts[2]
        try:
            fold = int(source.parts[3])
            tape_relative = tape_paths[(condition, fold)]
        except (KeyError, ValueError) as error:
            raise ValueError(
                "visual-review source trace has no exact scenario-tape mapping"
            ) from error
        trace_path = root / video["source_trace_path"]
        trace = load_trace(trace_path)
        tape = load_scenario_tape(root / tape_relative)
        sidecar_path = trace_path.with_name("dashboard_evidence.npz")
        sidecar = None
        sidecar_digest: str | None = None
        if sidecar_path.is_file():
            sidecar = load_dashboard_evidence(sidecar_path)
            sidecar_digest = sidecar.content_sha256
        if video.get("renderer", "basic-dashboard-v1") == "scientific-dashboard-v1" and (
            sidecar_digest is None
        ):
            raise ValueError("scientific dashboard visual review requires bound dashboard evidence")
        bindings = (
            review.trace_content_sha256 == trace.content_sha256,
            review.scenario_tape_sha256 == tape.sha256,
            review.dashboard_evidence_sha256 == sidecar_digest,
            review.video_file_sha256 == video["sha256"],
            review.video_file_sha256 == file_sha256(root / video["path"]),
            review.decoded_frames_sha256 == video["decoded_frames_sha256"],
            review.frame_width == video["width"],
            review.frame_height == video["height"],
            all(index < trace.steps for index in review.keyframe_indices),
        )
        if not all(bindings):
            raise ValueError(f"visual review does not bind exactly to {video['path']}")
        if require_final_core:
            _validate_review_frame_artifacts(
                root,
                stem=stem,
                keyframe_indices=review.keyframe_indices,
                expected_size=(video["width"], video["height"]),
                trace=trace,
                tape=tape,
                sidecar=sidecar,
                video_path=root / video["path"],
                contact_sheet_title=(f"{source.parts[1]} · {condition} · paired fold {fold}"),
            )
        reviewed.append(review_path.relative_to(root).as_posix())
    return tuple(reviewed)


def _validate_review_frame_artifacts(
    root: Path,
    *,
    stem: str,
    keyframe_indices: Sequence[int],
    expected_size: tuple[int, int],
    trace: ImmutableTrace,
    tape: Any,
    sidecar: Any,
    video_path: Path,
    contact_sheet_title: str,
) -> None:
    """Decode and bind canonical review PNGs to exact MP4 frames and contact-sheet pixels."""
    import imageio.v3 as iio
    import imageio_ffmpeg

    from crazyflow.safety.da_plcbf.scientific_dashboard import (
        KeyframeRecord,
        render_contact_sheet,
        select_keyframe_indices,
    )

    canonical_indices = select_keyframe_indices(
        trace, tape=tape, sidecar=sidecar, count=min(8, trace.steps)
    )
    if tuple(keyframe_indices) != canonical_indices:
        raise ValueError("final review keyframe indices are not the canonical eight-frame set")
    keyframe_directory = root / "keyframes" / stem
    expected_names = {f"keyframe-{index:06d}.png" for index in canonical_indices}
    if not keyframe_directory.is_dir() or keyframe_directory.is_symlink():
        raise ValueError(f"missing canonical keyframe directory for {stem}")
    actual_names = {
        path.name
        for path in keyframe_directory.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual_names != expected_names or any(
        path.is_dir() for path in keyframe_directory.iterdir()
    ):
        raise ValueError(f"keyframe PNGs do not exactly match the visual review for {stem}")
    width, height = expected_size
    decoded: dict[int, np.ndarray] = {}
    reader = imageio_ffmpeg.read_frames(str(video_path), pix_fmt="rgb24", bits_per_pixel=24)
    try:
        metadata = next(reader)
        if not isinstance(metadata, Mapping) or metadata.get("size") != expected_size:
            raise ValueError("review video decoded size does not match its manifest")
        wanted = set(canonical_indices)
        for step, raw in enumerate(reader):
            if step in wanted:
                decoded[step] = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
    finally:
        reader.close()
    if tuple(sorted(decoded)) != canonical_indices:
        raise ValueError("review video does not contain every canonical keyframe")

    records = []
    for index in canonical_indices:
        path = keyframe_directory / f"keyframe-{index:06d}.png"
        frame = _decode_rgb_png(path, iio=iio)
        if frame.shape != (height, width, 3):
            raise ValueError(f"review keyframe is not full video resolution: {path.name}")
        if not np.array_equal(frame, decoded[index]):
            raise ValueError(f"review keyframe pixels do not match decoded MP4 frame: {path.name}")
        records.append(
            KeyframeRecord(
                step=index,
                time_seconds=float(trace.time[index]),
                path=str(path.resolve()),
                width=width,
                height=height,
                sha256=file_sha256(path),
            )
        )
    contact_sheet = root / "contact_sheets" / f"{stem}.png"
    actual_sheet = _decode_rgb_png(contact_sheet, iio=iio)
    with tempfile.TemporaryDirectory(prefix="crazyflow-da-plcbf-contact-check-") as directory:
        expected_path = Path(directory) / "expected.png"
        expected_record = render_contact_sheet(records, expected_path, title=contact_sheet_title)
        expected_sheet = _decode_rgb_png(expected_path, iio=iio)
        if not np.array_equal(actual_sheet, expected_sheet) or (
            file_sha256(contact_sheet) != expected_record.sha256
        ):
            raise ValueError("review contact sheet does not deterministically match keyframes")


def _decode_rgb_png(path: Path, *, iio: Any) -> np.ndarray:
    """Fully decode a regular PNG and require canonical uint8 RGB pixels."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular PNG artifact: {path}")
    try:
        frame = np.asarray(iio.imread(path, extension=".png"))
    except Exception as error:
        raise ValueError(f"artifact is not a decodable PNG: {path}") from error
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"PNG must decode to uint8 RGB pixels: {path}")
    return np.ascontiguousarray(frame)


def _validate_final_core_video_coverage(video_records: Sequence[Mapping[str, Any]]) -> None:
    """Require one full-method scientific video for every and only final core condition."""
    from crazyflow.safety.da_plcbf.experiments import REQUIRED_CONDITIONS

    if len(video_records) != len(REQUIRED_CONDITIONS):
        raise ValueError("final scientific evidence requires exactly four reviewed videos")
    observed: list[str] = []
    for record in video_records:
        if record.get("renderer", "basic-dashboard-v1") != "scientific-dashboard-v1":
            raise ValueError("final scientific evidence requires scientific-dashboard videos")
        source = PurePosixPath(record["source_trace_path"])
        if (
            len(source.parts) != 5
            or source.parts[0] != "methods"
            or source.parts[1] != "da_plcbf_full"
            or source.name != "trace.npz"
        ):
            raise ValueError("final videos must replay da_plcbf_full method traces")
        try:
            _require_uint32(int(source.parts[3]), "final video fold")
        except ValueError as error:
            raise ValueError("final video fold path must be a uint32 integer") from error
        observed.append(source.parts[2])
    if tuple(sorted(observed)) != tuple(sorted(REQUIRED_CONDITIONS)):
        raise ValueError("final videos must cover every and only the four required conditions once")


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of a regular file without following a symlink."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"not a regular non-symlink file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace_arrays(trace: ImmutableTrace) -> dict[str, np.ndarray]:
    return {item.name: getattr(trace, item.name) for item in fields(trace)}


def _event_mapping(event: ArtifactEvent) -> dict[str, Any]:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "sequence": event.sequence,
        "step": event.step,
        "time_seconds": event.time_seconds,
        "category": event.category,
        "name": event.name,
        "severity": event.severity,
        "snapshot_version": event.snapshot_version,
        "model_version": event.model_version,
        "details": dict(event.details),
    }


def _validate_events(
    events: Sequence[ArtifactEvent], *, trace: ImmutableTrace | None
) -> tuple[ArtifactEvent, ...]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence) or not events:
        raise ValueError("events must be a nonempty sequence")
    validated = tuple(events)
    for index, event in enumerate(validated):
        if not isinstance(event, ArtifactEvent):
            raise TypeError("every event must be an ArtifactEvent")
        event.validate()
        if event.sequence != index:
            raise ValueError("event sequence numbers must be contiguous from zero")
        if index and event.time_seconds < validated[index - 1].time_seconds:
            raise ValueError("event times must be nondecreasing")
        if trace is not None:
            if event.step >= trace.steps:
                raise ValueError("event step lies outside the trace")
            if not math.isclose(
                event.time_seconds, float(trace.time[event.step]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("event time does not match its trace step")
            if event.snapshot_version != int(trace.snapshot_version[event.step]):
                raise ValueError("event snapshot version does not match the trace")
            if event.model_version != int(trace.model_version[event.step]):
                raise ValueError("event model version does not match the trace")
    return validated


def _validate_aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("aggregate rows must be a nonempty sequence")
    validated: list[dict[str, Any]] = []
    keys = set(AGGREGATE_COLUMNS)
    for index, raw in enumerate(rows):
        row = _require_exact_mapping(raw, keys, f"aggregate row {index}")
        _require_slug(row["method"], f"aggregate row {index}.method")
        _require_slug(row["condition"], f"aggregate row {index}.condition")
        _require_uint32(row["seed"], f"aggregate row {index}.seed")
        metrics = {
            "schema_version": METRICS_SCHEMA_VERSION,
            **{key: row[key] for key in keys - {"method", "condition", "seed"}},
        }
        validate_metrics(metrics)
        validated.append(row)
    paired_keys = [(row["method"], row["condition"], row["seed"]) for row in validated]
    if len(paired_keys) != len(set(paired_keys)):
        raise ValueError("aggregate method/condition/seed keys must be unique")
    return tuple(validated)


def _sort_aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(
        dict(row)
        for row in sorted(rows, key=lambda item: (item["condition"], item["seed"], item["method"]))
    )


def _validate_confidence_intervals(
    data: Mapping[str, Any], *, rows: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    result = _require_exact_mapping(
        data,
        {"schema_version", "confidence_level", "interval_method", "summaries"},
        "confidence intervals",
    )
    _require_schema(result, AGGREGATE_SCHEMA_VERSION, "confidence intervals")
    confidence = _require_finite(result["confidence_level"], "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence level must lie strictly between zero and one")
    if result["interval_method"] != "descriptive-normal-not-a-paired-superiority-test":
        raise ValueError("confidence interval method label is invalid")
    if not isinstance(result["summaries"], list) or not result["summaries"]:
        raise ValueError("confidence interval summaries must be nonempty")
    summary_keys = {
        "method",
        "condition",
        "count",
        "minimum_hard_margin_mean",
        "failure_steps_mean",
        "interval_available",
        "unavailable_reason",
        "minimum_hard_margin_mean_interval",
        "failure_steps_mean_interval",
    }
    for index, value in enumerate(result["summaries"]):
        summary = _require_exact_mapping(value, summary_keys, f"CI summary {index}")
        _require_slug(summary["method"], f"CI summary {index}.method")
        _require_slug(summary["condition"], f"CI summary {index}.condition")
        _require_positive_int(summary["count"], f"CI summary {index}.count")
        _require_finite(summary["minimum_hard_margin_mean"], "CI margin mean")
        _require_finite_nonnegative(summary["failure_steps_mean"], "CI failure mean")
        if not isinstance(summary["interval_available"], bool):
            raise TypeError("CI interval_available must be boolean")
        intervals = (
            summary["minimum_hard_margin_mean_interval"],
            summary["failure_steps_mean_interval"],
        )
        if summary["interval_available"]:
            if summary["count"] < 2 or summary["unavailable_reason"] is not None:
                raise ValueError("available CI has inconsistent availability metadata")
            for interval in intervals:
                if not isinstance(interval, list) or len(interval) != 2:
                    raise ValueError("available CI must have a two-element interval")
                lower = _require_finite(interval[0], "CI lower")
                upper = _require_finite(interval[1], "CI upper")
                if lower > upper:
                    raise ValueError("CI lower bound exceeds upper bound")
        else:
            _require_nonempty_string(summary["unavailable_reason"], "CI unavailable reason")
            if any(interval is not None for interval in intervals):
                raise ValueError("unavailable CI interval fields must be null")
    if rows is not None:
        expected = _confidence_interval_payload(
            _validate_aggregate_rows(rows), float(result["confidence_level"])
        )
        if _canonical_json_bytes(result) != _canonical_json_bytes(expected):
            raise ValueError("confidence intervals do not agree with paired metrics CSV")
    return result


def _validate_file_record(record: Mapping[str, Any]) -> dict[str, Any]:
    data = _require_exact_mapping(record, {"path", "sha256", "bytes", "role"}, "file record")
    _validate_relative_path(data["path"])
    if data["path"] in {"manifest.json", "SHA256SUMS"}:
        raise ValueError("manifest inventory must not include its metadata files")
    _require_sha256(data["sha256"], "file record sha256")
    _require_nonnegative_int(data["bytes"], "file record bytes")
    _require_slug(data["role"], "file record role")
    return data


def _validate_seed_tape_record(record: Mapping[str, Any]) -> dict[str, Any]:
    data = _require_exact_mapping(
        record, {"condition", "fold", "path", "content_sha256"}, "seed tape record"
    )
    condition = _require_slug(data["condition"], "seed tape condition")
    fold = _require_uint32(data["fold"], "seed tape fold")
    path = _validate_relative_path(data["path"])
    allowed = {f"scenario_tapes/{fold}.npz", f"scenario_tapes/{condition}/{fold}.npz"}
    if path not in allowed:
        raise ValueError(
            "scenario-tape path must be canonical shared <fold>.npz or condition/<fold>.npz"
        )
    _require_sha256(data["content_sha256"], "seed tape content_sha256")
    return data


def _validate_manifest_tape_record(record: Mapping[str, Any]) -> dict[str, Any]:
    data = _require_exact_mapping(
        record,
        {"condition", "fold", "path", "content_sha256", "file_sha256"},
        "manifest tape record",
    )
    _validate_seed_tape_record(
        {key: data[key] for key in ("condition", "fold", "path", "content_sha256")}
    )
    _require_sha256(data["file_sha256"], "manifest tape file_sha256")
    return data


def _require_unambiguous_tape_paths(
    records: Sequence[Mapping[str, Any]], *, include_file_digest: bool
) -> None:
    bindings: dict[str, tuple[str, ...]] = {}
    for record in records:
        digest_binding = (record["content_sha256"],)
        if include_file_digest:
            digest_binding += (record["file_sha256"],)
        previous = bindings.setdefault(record["path"], digest_binding)
        if previous != digest_binding:
            raise ValueError("one scenario-tape path cannot have ambiguous digest bindings")


def _build_manifest_tape_records(
    root: Path, seed_records: Sequence[Mapping[str, Any]], file_records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    semantic_by_path: dict[str, str] = {}
    result = []
    for raw_record in seed_records:
        record = _validate_seed_tape_record(raw_record)
        path = record["path"]
        if path not in semantic_by_path:
            semantic_by_path[path] = _validated_scenario_tape_digest(root / path)
        semantic = semantic_by_path[path]
        if semantic != record["content_sha256"]:
            raise ValueError("seed mapping semantic digest does not match its scenario tape")
        file_record = _record_for_path(file_records, path)
        if file_record["role"] != "scenario-tape":
            raise ValueError("mapped scenario tape has the wrong manifest file role")
        result.append({**record, "file_sha256": file_record["sha256"]})
    result.sort(key=lambda item: (item["condition"], item["fold"]))
    indexed = {record["path"] for record in file_records if record["role"] == "scenario-tape"}
    mapped = {record["path"] for record in result}
    if indexed != mapped:
        raise ValueError("every and only indexed scenario tapes must appear in the seed mapping")
    return result


def _validate_video_record(record: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "path",
        "source_trace_path",
        "sha256",
        "codec",
        "width",
        "height",
        "fps",
        "frame_count",
        "duration_seconds",
        "decoded_frames_sha256",
    }
    keys = set(record)
    if keys not in (required, required | {"renderer"}):
        raise ValueError("video record has missing or unexpected fields")
    data = dict(record)
    renderer = data.get("renderer", "basic-dashboard-v1")
    if renderer not in {"basic-dashboard-v1", "scientific-dashboard-v1"}:
        raise ValueError("video record renderer is unsupported")
    if "renderer" in data:
        data["renderer"] = renderer
    _validate_relative_path(data["path"])
    _validate_relative_path(data["source_trace_path"])
    if not data["path"].endswith(".mp4") or not data["source_trace_path"].endswith("trace.npz"):
        raise ValueError("video/source paths have invalid suffixes")
    for name in ("sha256", "decoded_frames_sha256"):
        _require_sha256(data[name], f"video record {name}")
    if data["codec"] != "h264":
        raise ValueError("video record codec must be decoded as h264")
    for name in ("width", "height", "frame_count"):
        _require_positive_int(data[name], f"video record {name}")
    if data["width"] % 2 or data["height"] % 2:
        raise ValueError("H.264 dashboard dimensions must be even")
    _require_finite_positive(data["fps"], "video record fps")
    _require_finite_positive(data["duration_seconds"], "video record duration_seconds")
    expected_duration = data["frame_count"] / data["fps"]
    if not math.isclose(data["duration_seconds"], expected_duration, abs_tol=1.0 / data["fps"]):
        raise ValueError("video record duration is inconsistent with frames/fps")
    return data


def _scan_file_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for relative, path in _regular_run_files(root, exclude={"manifest.json", "SHA256SUMS"}):
        records.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "role": _artifact_role(relative),
            }
        )
    return records


def _scan_checksum_records(root: Path) -> list[tuple[str, str]]:
    return [
        (relative, file_sha256(path))
        for relative, path in _regular_run_files(root, exclude={"SHA256SUMS"})
    ]


def _regular_run_files(root: Path, *, exclude: set[str]) -> list[tuple[str, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    values: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"artifact tree must not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        _validate_relative_path(relative)
        values.append((relative, path))
    return sorted(values)


def _record_for_path(records: Sequence[Mapping[str, Any]], path: str) -> Mapping[str, Any]:
    matches = [record for record in records if record["path"] == path]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain exactly one file record for {path}")
    return matches[0]


def _artifact_role(relative: str) -> str:
    exact = {
        "config.json": "config",
        "provenance.json": "provenance",
        "seeds.json": "seeds",
        "aggregate/paired_metrics.csv": "aggregate-metrics",
        "aggregate/confidence_intervals.json": "aggregate-intervals",
        "aggregate/outcomes.jsonl": "scientific-outcomes",
        "aggregate/paired_comparisons.json": "scientific-comparisons",
        "aggregate/video_records.json": "video-index",
        "aggregate/report.md": "aggregate-report",
        "aggregate/scientific_report.md": "scientific-report",
    }
    if relative in exact:
        return exact[relative]
    if relative.startswith("scenario_tapes/") and relative.endswith(".npz"):
        return "scenario-tape"
    name = PurePosixPath(relative).name
    method_names = {
        "trace.npz": "trace",
        "dashboard_evidence.npz": "dashboard-evidence",
        "events.jsonl": "events",
        "metrics.json": "metrics",
        "timing.json": "timing",
    }
    if relative.startswith("methods/") and name in method_names:
        return method_names[name]
    prefix_roles = {
        "checkpoints/": "checkpoint",
        "plots/": "plot",
        "videos/": "video",
        "keyframes/": "keyframe",
        "contact_sheets/": "contact-sheet",
        "visual_reviews/": "visual-review",
    }
    for prefix, role in prefix_roles.items():
        if relative.startswith(prefix):
            return role
    if relative == "visual_review.md":
        return "visual-review"
    raise ValueError(f"file is outside the version-1 artifact layout: {relative}")


def _query_gpus() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    command = [
        executable,
        "--query-gpu=index,name,driver_version,memory.total,uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = _run_text(command)
    except RuntimeError:
        return []
    result = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            result.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "driver_version": parts[2],
                    "memory_total_bytes": int(float(parts[3]) * 1024 * 1024),
                    "uuid": parts[4],
                }
            )
        except ValueError:
            continue
    return result


def _query_jax_runtime() -> dict[str, Any]:
    script = (
        "import json, jax, jaxlib; "
        "print(json.dumps({'version':jax.__version__,'jaxlib_version':jaxlib.__version__,"
        "'backend':jax.default_backend(),'devices':[str(x) for x in jax.devices()]}))"
    )
    try:
        output = _run_text([sys.executable, "-c", script])
        value = json.loads(output.splitlines()[-1])
        if not isinstance(value, dict):
            raise ValueError
        for name in ("version", "jaxlib_version", "backend"):
            _require_nonempty_string(value[name], f"queried jax.{name}")
        if not isinstance(value["devices"], list) or not value["devices"]:
            raise ValueError
        return value
    except (KeyError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        return {
            "version": "unavailable",
            "jaxlib_version": "unavailable",
            "backend": "unavailable",
            "devices": [f"unavailable:{type(error).__name__}"],
        }


def _validated_scenario_tape_digest(path: Path) -> str:
    """Validate a scenario tape in an isolated process so replay never initializes JAX."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("scenario tape must be a regular non-symlink file")
    script = (
        "import sys; "
        "from crazyflow.safety.da_plcbf.scenarios import load_scenario_tape; "
        "print(load_scenario_tape(sys.argv[1]).sha256)"
    )
    try:
        output = _run_text([sys.executable, "-c", script, str(path)])
    except RuntimeError as error:
        raise ValueError("scenario tape failed isolated schema validation") from error
    digest = output.splitlines()[-1] if output else ""
    return _require_sha256(digest, "validated scenario-tape digest")


def _run_text(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            list(command), cwd=cwd, check=True, capture_output=True, text=True, timeout=30.0
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"command failed: {command[0]}") from error
    return result.stdout.strip()


def _write_json(data: Mapping[str, Any], path: str | os.PathLike[str]) -> str:
    destination = _destination(path, suffix=".json", overwrite=False)
    _validate_json_value(dict(data), destination.name)
    _atomic_write_bytes(destination, _canonical_json_bytes(data), overwrite=False)
    return file_sha256(destination)


def _read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError("JSON artifact path must end in .json")
    if source.is_symlink() or not source.is_file():
        raise ValueError("JSON artifact must be a regular non-symlink file")
    payload = source.read_bytes()
    if len(payload) > _MAX_JSON_BYTES:
        raise ValueError("JSON artifact exceeds size limit")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("JSON object contains duplicate keys")
        return dict(pairs)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        value = json.loads(payload, object_pairs_hook=pairs_hook, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid JSON artifact") from error
    if not isinstance(value, dict):
        raise ValueError("JSON artifact root must be an object")
    _validate_json_value(value, source.name)
    return value


def _canonical_json_bytes(data: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            )
            + "\n"
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("artifact contains a non-canonical JSON value") from error


def _atomic_write_bytes(destination: Path, payload: bytes, *, overwrite: bool) -> None:
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
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


def _destination(path: str | os.PathLike[str], *, suffix: str, overwrite: bool) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != suffix:
        raise ValueError(f"artifact path must end in {suffix}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    if destination.is_symlink():
        raise ValueError("artifact destination must not be a symlink")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    return destination


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, value in sorted(arrays.items()):
            if not _SLUG_PATTERN.fullmatch(name):
                raise ValueError(f"invalid NPZ member name: {name}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(
                info, _npy_bytes(value), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )
    return output.getvalue()


def _load_strict_npz(source: Path, expected: set[str]) -> dict[str, np.ndarray]:
    if source.is_symlink() or not source.is_file():
        raise ValueError("NPZ artifact must be a regular non-symlink file")
    try:
        archive = zipfile.ZipFile(source, mode="r")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("artifact is not a valid NPZ archive") from error
    expected_members = {f"{name}.npy" for name in expected}
    try:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or set(names) != expected_members:
            raise ValueError("NPZ has missing, duplicate, or unexpected members")
        if sum(info.file_size for info in infos) > _MAX_NPZ_BYTES:
            raise ValueError("NPZ exceeds decompressed size limit")
        loaded: dict[str, np.ndarray] = {}
        for info in infos:
            if info.is_dir() or info.file_size < 0:
                raise ValueError("NPZ contains an invalid member")
            try:
                member = archive.read(info)
                array = np.load(io.BytesIO(member), allow_pickle=False)
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                raise ValueError(f"invalid NPZ member {info.filename!r}") from error
            if not isinstance(array, np.ndarray) or array.dtype.hasobject:
                raise ValueError("NPZ members must be non-object NumPy arrays")
            loaded[info.filename.removesuffix(".npy")] = array
    finally:
        archive.close()
    return loaded


def _npy_bytes(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    array = np.asarray(value)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _canonical_array_digest(prefix: bytes, arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(prefix)
    for name, value in sorted(arrays.items()):
        encoded_name = name.encode()
        encoded_array = _npy_bytes(value)
        digest.update(struct.pack("<I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack("<Q", len(encoded_array)))
        digest.update(encoded_array)
    return digest.hexdigest()


def _frozen_array(value: Any) -> np.ndarray:
    array = np.array(value, copy=True, order="C")
    if array.dtype.hasobject or array.dtype.kind not in "biufSU":
        raise ValueError("trace supports only boolean, real numeric, and fixed-width string arrays")
    array.setflags(write=False)
    return array


def _require_float(
    value: np.ndarray, name: str, *, ndim: int, shape: tuple[int, ...] | None = None
) -> None:
    if value.dtype.kind != "f" or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-D floating-point array")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")


def _require_float_matrix(value: np.ndarray, name: str, rows: int, columns: int) -> None:
    _require_float(value, name, ndim=2, shape=(rows, columns))


def _require_integer_vector(value: np.ndarray, name: str, length: int) -> None:
    if value.dtype.kind not in "iu" or value.shape != (length,):
        raise ValueError(f"{name} must be an integer vector with length {length}")


def _require_bool_vector(value: np.ndarray, name: str, length: int) -> None:
    if value.dtype != np.bool_ or value.shape != (length,):
        raise ValueError(f"{name} must be a boolean vector with length {length}")


def _require_scalar_integer(value: np.ndarray, name: str, expected: int) -> None:
    if value.shape != () or value.dtype.kind not in "iu" or int(value) != expected:
        raise ValueError(f"{name} must be scalar integer {expected}")


def _require_scalar_string(value: np.ndarray, name: str) -> str:
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar Unicode array")
    return str(value)


def _validate_names(value: np.ndarray, name: str, *, minimum: int = 1) -> tuple[str, ...]:
    if value.dtype.kind != "U" or value.ndim != 1 or value.size < minimum:
        raise ValueError(f"{name} must be a Unicode vector with at least {minimum} entries")
    names = tuple(str(item) for item in value)
    if len(names) != len(set(names)):
        raise ValueError(f"{name} entries must be unique")
    for item in names:
        _require_slug(item, f"{name} entry")
    return names


def _require_exact_mapping(
    value: Mapping[str, Any], expected: set[str], name: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    data = dict(value)
    missing = expected - set(data)
    extra = set(data) - expected
    if missing or extra:
        raise ValueError(f"{name} has missing keys {sorted(missing)} or extra keys {sorted(extra)}")
    return data


def _require_schema(data: Mapping[str, Any], expected: int, name: str) -> None:
    if (
        isinstance(data["schema_version"], bool)
        or not isinstance(data["schema_version"], Integral)
        or data["schema_version"] != expected
    ):
        raise ValueError(f"{name}.schema_version must be {expected}")


def _require_slug(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SLUG_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a portable slug")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must be a nonempty string without control characters")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_finite_nonnegative(value: Any, name: str) -> float:
    result = _require_finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _require_finite_positive(value: Any, name: str) -> float:
    result = _require_finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _require_positive_int(value: Any, name: str) -> int:
    result = _require_nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _require_uint32(value: Any, name: str) -> int:
    result = _require_nonnegative_int(value, name)
    if result > np.iinfo(np.uint32).max:
        raise ValueError(f"{name} must fit in uint32")
    return result


def _validate_slug_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    values = tuple(_require_slug(item, f"{name}[]") for item in value)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} entries must be unique")
    return values


def _validate_json_value(value: Any, name: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and any(
            ord(char) < 32 and char not in "\t\n\r" for char in value
        ):
            raise ValueError(f"{name} contains a forbidden control character")
        return
    if isinstance(value, Integral):
        return
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} object keys must be nonempty strings")
            _validate_json_value(item, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{name}[{index}]")
        return
    raise TypeError(
        f"{name} contains a value unsupported by canonical JSON: {type(value).__name__}"
    )


def _require_json_numeric_agreement(
    actual: Mapping[str, Any], expected: Mapping[str, Any], name: str
) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{name} keys do not agree")
    for key in expected:
        left = actual[key]
        right = expected[key]
        if isinstance(right, bool) or isinstance(right, str) or right is None:
            if left != right:
                raise ValueError(f"{name}.{key} does not agree")
        elif isinstance(right, Integral):
            if isinstance(left, bool) or left != right:
                raise ValueError(f"{name}.{key} does not agree")
        elif isinstance(right, Real):
            if (
                isinstance(left, bool)
                or not isinstance(left, Real)
                or not math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)
            ):
                raise ValueError(f"{name}.{key} does not agree")
        else:
            raise TypeError(f"unsupported comparison value for {name}.{key}")


def _validate_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("artifact path must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("artifact path must not be absolute or contain traversal")
    return value


def _validate_utc_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("created_utc must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError("created_utc is not a valid ISO-8601 timestamp") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError("created_utc must be UTC")
    return value


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid aggregate CSV value")
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        finite = _require_finite(value, "CSV value")
        return format(finite, ".17g")
    if isinstance(value, str):
        return value
    raise TypeError("unsupported aggregate CSV value")


def _normal_quantile(probability: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(probability)
