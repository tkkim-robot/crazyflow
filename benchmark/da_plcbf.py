"""Reproducible latency and scaling benchmark for the DA-PLCBF implementation.

This benchmark measures the implementation in this repository; it is not a reproduction of any
paper timing.  JAX lowering/compilation is timed separately from synchronized warm executions.
Every successful measurement retains its raw samples, tail statistics, deadline misses, compiler
memory analysis, requested/effective shape, and host/device/git provenance.

Examples::

    pixi run -e tests python benchmark/da_plcbf.py --device cpu --preset smoke \
        --repeats 3 --warmups 1 --contention cpu

    pixi run -e gpu-tests python benchmark/da_plcbf.py --device gpu --preset final \
        --repeats 50 --warmups 5 --contention cpu,gpu \
        --output artifacts/da_plcbf/performance-gpu.json

The ``final`` preset is a fixed sweep containing a joint ``K=64, B=64, R=8, H=50`` probe plus
one-factor probes.  An estimated-memory guard runs before compilation.  A resource exhaustion or
unsupported full-stack shape is serialized as an explicit failed/skipped record; it is never
silently omitted or relabeled as a smaller benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax import Array

from crazyflow import __version__ as crazyflow_version
from crazyflow.control import Control
from crazyflow.drones import load_params
from crazyflow.dynamics import Dynamics
from crazyflow.safety.da_plcbf.actor import (
    SharedActorConfig,
    SharedActorParams,
    SharedActorSpec,
    initialize_shared_actor,
)
from crazyflow.safety.da_plcbf.config import LibraryLossConfig
from crazyflow.safety.da_plcbf.full_stack import build_unclipped_full_stack_step
from crazyflow.safety.da_plcbf.polytope_qp import project_affine_polytope
from crazyflow.safety.da_plcbf.quad_actor_bptt import build_quad_actor_bptt_functions
from crazyflow.safety.da_plcbf.quad_actor_losses import (
    QuadLearningConfig,
    rigid_body_safety_batch_from_circles,
)
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.quad_rollouts import rollout_shared_quad_library
from crazyflow.safety.da_plcbf.quad_uncertainty import (
    VersionAModelSamples,
    rollout_shared_quad_library_under_uncertainty,
)
from crazyflow.safety.da_plcbf.snapshots import create_active_snapshot, create_candidate_snapshot
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.validation import (
    HardValidationEvidence,
    HardValidationThresholds,
    hard_validate_candidate,
)
from crazyflow.safety.da_plcbf.version_a_barriers import (
    RigidBodySafetySet,
    VersionABarrierConfig,
    VersionAModel,
    continuous_safety_halfspaces,
)
from crazyflow.safety.da_plcbf.version_a_filter import (
    VersionAActuator,
    VersionAFilterConfig,
    validated_motor_polytope,
)
from crazyflow.safety.da_plcbf.version_a_runtime import version_a_runtime_step
from crazyflow.safety.da_plcbf.version_b_runtime import (
    VersionBRuntimeConfig,
    version_b_runtime_step,
)
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator

ComponentName = Literal[
    "rollout", "uncertain_rollout", "bptt", "version_a", "qp", "validation", "version_b"
]
ALL_COMPONENTS: tuple[ComponentName, ...] = (
    "rollout",
    "uncertain_rollout",
    "bptt",
    "version_a",
    "qp",
    "validation",
    "version_b",
)
PERFORMANCE_ARTIFACT_SCHEMA = "crazyflow.da_plcbf.performance.v2"
PERFORMANCE_ARTIFACT_SCHEMA_VERSION = 2

COMPONENT_LABELS: dict[ComponentName, str] = {
    "rollout": "shared_quad_rollout_forward",
    "uncertain_rollout": "shared_quad_uncertainty_rollout_forward",
    "bptt": "quad_bptt_backward_optimizer_step",
    "version_a": "version_a_active_filter_end_to_end",
    "qp": "version_a_exhaustive_active_set_qp",
    "validation": "hard_candidate_admission_gate_only",
    "version_b": "version_b_full_stack_runtime",
}


@dataclass(frozen=True, slots=True)
class ShapePoint:
    """One explicit static-shape point in a named benchmark preset."""

    name: str
    policies: int
    scenarios: int
    uncertainty_samples: int
    horizon: int

    def validate(self) -> None:
        """Reject invalid or unsupported axes before constructing JAX values."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("shape point name must be nonempty")
        for name, value in (
            ("policies", self.policies),
            ("scenarios", self.scenarios),
            ("horizon", self.horizon),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.uncertainty_samples not in (4, 8):
            raise ValueError("uncertainty_samples must be exactly 4 or 8")


PRESETS: dict[str, tuple[ShapePoint, ...]] = {
    "smoke": (ShapePoint("smoke", 2, 1, 4, 2),),
    "scale": (
        ShapePoint("scale-version-b-probe", 2, 1, 4, 2),
        ShapePoint("scale-anchor", 4, 2, 4, 5),
        ShapePoint("scale-k16", 16, 2, 4, 5),
        ShapePoint("scale-b16", 4, 16, 4, 5),
        ShapePoint("scale-r8", 4, 2, 8, 5),
        ShapePoint("scale-h20", 4, 2, 4, 20),
    ),
    "final": (
        ShapePoint("final-version-b-probe", 2, 1, 4, 2),
        ShapePoint("final-anchor", 8, 8, 4, 20),
        ShapePoint("final-k64", 64, 8, 4, 20),
        ShapePoint("final-b64", 8, 64, 4, 20),
        ShapePoint("final-r8", 8, 8, 8, 20),
        ShapePoint("final-h50", 8, 8, 4, 50),
        ShapePoint("final-joint-k64-b64-r8-h50", 64, 64, 8, 50),
    ),
}

# Contention is intentionally a separate, exact request rather than an incidental reuse of the
# first point in a measurement sweep.  In particular, the final profile must expose a failure or
# resource limit at the real final shape instead of producing an attractive K=2 probe number.
CONTENTION_SHAPES: dict[str, ShapePoint] = {
    "smoke": ShapePoint("smoke-contention-k2-b1-r4-h2", 2, 1, 4, 2),
    "scale": ShapePoint("scale-contention-k16-b16-r4-h20", 16, 16, 4, 20),
    "final": ShapePoint("final-contention-k64-b64-r8-h50", 64, 64, 8, 50),
}


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    """Host-side timing and resource settings."""

    repeats: int
    warmups: int
    deadline_seconds: float
    max_estimated_bytes: int

    def validate(self) -> None:
        """Validate settings without silently changing requested repetitions."""
        if isinstance(self.repeats, bool) or not isinstance(self.repeats, int) or self.repeats <= 0:
            raise ValueError("repeats must be a positive integer")
        if isinstance(self.warmups, bool) or not isinstance(self.warmups, int) or self.warmups < 0:
            raise ValueError("warmups must be a nonnegative integer")
        if not math.isfinite(self.deadline_seconds) or self.deadline_seconds <= 0:
            raise ValueError("deadline must be finite and positive")
        if (
            isinstance(self.max_estimated_bytes, bool)
            or not isinstance(self.max_estimated_bytes, int)
            or self.max_estimated_bytes <= 0
        ):
            raise ValueError("max_estimated_bytes must be a positive integer")


class QuadProblem(NamedTuple):
    """Static and dynamic values shared by direct-wrench benchmark components."""

    model: VersionAModel
    actuator: VersionAActuator
    spec: SharedActorSpec
    scenarios: CircleScenarioBatch
    safety: RigidBodySafetySet
    initial_states: Array
    params: SharedActorParams
    actor_config: SharedActorConfig
    quad_config: QuadPolicyConfig
    barrier_config: VersionABarrierConfig
    learning_config: QuadLearningConfig
    loss_config: LibraryLossConfig
    targets: Array
    descriptor_scales: Array


class VersionBProblem(NamedTuple):
    """One-world full-stack data and all runtime dependencies."""

    data: Any
    nominal: Array
    params: SharedActorParams
    spec: SharedActorSpec
    scenarios: CircleScenarioBatch
    safety: RigidBodySafetySet
    model: VersionAModel
    actuator: VersionAActuator
    actor_config: SharedActorConfig
    quad_config: QuadPolicyConfig
    barrier_config: VersionABarrierConfig
    one_step: Callable[[Any], Any]
    action_lower: Array
    action_upper: Array
    weight: Array
    trust_radius: Array
    config: VersionBRuntimeConfig


def summarize_timings(samples: list[float], deadline_seconds: float) -> dict[str, Any]:
    """Return raw synchronized samples and deterministic tail summaries."""
    if not samples:
        raise ValueError("at least one timing sample is required")
    values = np.asarray(samples, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("timing samples must be finite and nonnegative")
    misses = int(np.count_nonzero(values > deadline_seconds))
    return {
        "raw_seconds": [float(value) for value in values],
        "median_seconds": float(np.quantile(values, 0.50)),
        "p95_seconds": float(np.quantile(values, 0.95)),
        "p99_seconds": float(np.quantile(values, 0.99)),
        "worst_seconds": float(np.max(values)),
        "deadline_seconds": float(deadline_seconds),
        "deadline_misses": misses,
        "deadline_miss_fraction": float(misses / values.size),
    }


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _canonical_digest(document_without_integrity: dict[str, Any]) -> str:
    encoded = json.dumps(
        document_without_integrity, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_provenance() -> dict[str, Any]:
    root = _repository_root()

    def command(*arguments: str) -> str:
        try:
            return subprocess.run(
                ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    status = command("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": status not in ("", "unknown"),
    }


def _source_tree_binding() -> dict[str, Any]:
    """Hash benchmark code and executable Crazyflow/config inputs."""
    root = _repository_root()
    paths = [root / "benchmark" / "da_plcbf.py", root / "pyproject.toml", root / "pixi.lock"]
    included_suffixes = {".json", ".py", ".stl", ".toml", ".xml", ".yaml", ".yml"}
    paths.extend(
        path
        for path in (root / "crazyflow").rglob("*")
        if path.is_file() and path.suffix in included_suffixes and "__pycache__" not in path.parts
    )
    unique_paths = sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())
    aggregate = hashlib.sha256()
    for path in unique_paths:
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    git = _git_provenance()
    return {
        "scope": (
            "benchmark/da_plcbf.py + crazyflow/**/*.{py,json,stl,toml,xml,yaml,yml} + lock/config"
        ),
        "tree_sha256": aggregate.hexdigest(),
        "file_count": len(unique_paths),
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "git_dirty": git["dirty"],
    }


def _device_attributes(device: jax.Device) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "id": int(device.id),
        "platform": str(device.platform),
        "device_kind": str(device.device_kind),
    }
    for name in ("compute_capability", "local_hardware_id", "process_index", "slice_index"):
        value = getattr(device, name, None)
        if value is not None and isinstance(value, (str, int, float, bool, tuple)):
            attributes[name] = value
    return attributes


def _memory_stats(device: jax.Device) -> dict[str, int] | None:
    try:
        stats = device.memory_stats()
    except (AttributeError, RuntimeError):
        return None
    if stats is None:
        return None
    return {
        str(key): int(value) for key, value in stats.items() if isinstance(value, (int, np.integer))
    }


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return int(value if sys.platform == "darwin" else value * 1024)


def _compiled_memory_analysis(compiled: Any) -> dict[str, int] | None:
    try:
        analysis = compiled.memory_analysis()
    except (AttributeError, RuntimeError):
        return None
    if analysis is None:
        return None
    fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
    )
    return {
        name: int(getattr(analysis, name))
        for name in fields
        if getattr(analysis, name, None) is not None
    }


def _memory_availability(values: dict[str, int] | None, source: str) -> dict[str, Any]:
    if values is None:
        return {
            "available": False,
            "source": source,
            "values": None,
            "reason": "backend did not expose this memory measurement",
        }
    return {"available": True, "source": source, "values": values, "reason": None}


def _estimate_component_bytes(component: ComponentName, shape: ShapePoint) -> int:
    """Conservative static live-set estimate used only for a pre-compilation guard."""
    k, batch, samples, horizon = (
        shape.policies,
        shape.scenarios,
        shape.uncertainty_samples,
        shape.horizon,
    )
    float_bytes = 4
    if component == "rollout":
        elements = k * batch * (horizon + 1) * 48
        multiplier = 10
    elif component == "uncertain_rollout":
        elements = k * batch * samples * (horizon + 1) * 56
        multiplier = 12
    elif component == "bptt":
        elements = k * batch * (horizon + 1) * 64
        multiplier = 36
    elif component == "version_a":
        elements = k * (horizon + 1) * 96
        multiplier = 30
    elif component == "validation":
        elements = k * batch * 4
        multiplier = 8
    elif component == "version_b":
        elements = k * (horizon + 1) * 512
        multiplier = 80
    else:
        elements = 4096
        multiplier = 32
    return max(1 << 20, int(elements * float_bytes * multiplier))


def _effective_shape(component: ComponentName, shape: ShapePoint) -> dict[str, int | None]:
    """Expose only axes actually consumed by a component; never imply sliced axes were measured."""
    if component == "uncertain_rollout":
        return {
            "policies": shape.policies,
            "scenarios": shape.scenarios,
            "uncertainty_samples": shape.uncertainty_samples,
            "horizon": shape.horizon,
            "held_substeps": None,
        }
    if component in ("rollout", "bptt"):
        return {
            "policies": shape.policies,
            "scenarios": shape.scenarios,
            "uncertainty_samples": None,
            "horizon": shape.horizon,
            "held_substeps": None,
        }
    if component == "version_a":
        return {
            "policies": shape.policies,
            "scenarios": 1,
            "uncertainty_samples": None,
            "horizon": shape.horizon,
            "held_substeps": 1,
        }
    if component == "version_b":
        return {
            "policies": shape.policies,
            "scenarios": 1,
            "uncertainty_samples": None,
            "horizon": shape.horizon,
            "held_substeps": 2,
        }
    if component == "validation":
        return {
            "policies": shape.policies,
            "scenarios": shape.scenarios,
            "uncertainty_samples": None,
            "horizon": None,
            "held_substeps": None,
        }
    return {
        "policies": None,
        "scenarios": None,
        "uncertainty_samples": None,
        "horizon": None,
        "held_substeps": None,
    }


def _record_identity(component: ComponentName, shape: ShapePoint) -> dict[str, Any]:
    requested = asdict(shape)
    return {
        "measurement_id": f"{shape.name}:{component}",
        "component_key": component,
        "component": COMPONENT_LABELS[component],
        "shape": requested,
        "requested_shape": requested,
        "effective_shape": _effective_shape(component, shape),
    }


def _skipped_record(
    component: ComponentName,
    shape: ShapePoint,
    reason: str,
    estimated_bytes: int,
    settings: BenchmarkSettings,
) -> dict[str, Any]:
    return {
        **_record_identity(component, shape),
        "status": "skipped",
        "failure_stage": "pre_execution_guard",
        "reason": reason,
        "estimated_live_bytes": estimated_bytes,
        "memory_guard_bytes": settings.max_estimated_bytes,
        "compile_seconds": None,
        "warmup_seconds": [],
        "timing": None,
    }


def _failed_record(
    component: ComponentName,
    shape: ShapePoint,
    error: BaseException,
    estimated_bytes: int,
    settings: BenchmarkSettings,
) -> dict[str, Any]:
    return {
        **_record_identity(component, shape),
        "status": "failed",
        "failure_stage": "construction_compilation_or_execution",
        "reason": f"{type(error).__name__}: {error}",
        "estimated_live_bytes": estimated_bytes,
        "memory_guard_bytes": settings.max_estimated_bytes,
        "compile_seconds": None,
        "warmup_seconds": [],
        "timing": None,
    }


def _block(value: Any) -> None:
    jax.block_until_ready(value)


def benchmark_jitted(
    component: ComponentName,
    shape: ShapePoint,
    function: Any,
    arguments: tuple[Any, ...],
    settings: BenchmarkSettings,
    device: jax.Device,
    *,
    work_units: int,
    estimated_bytes: int,
) -> tuple[dict[str, Any], Any]:
    """Compile explicitly, then time only synchronized executions of one static program."""
    before_stats = _memory_stats(device)
    rss_before = _rss_bytes()
    compile_start = time.perf_counter()
    compiled = function.lower(*arguments).compile()
    compile_seconds = time.perf_counter() - compile_start
    compile_memory = _compiled_memory_analysis(compiled)

    warmup_seconds: list[float] = []
    for _ in range(settings.warmups):
        start = time.perf_counter()
        _block(compiled(*arguments))
        warmup_seconds.append(time.perf_counter() - start)

    raw_seconds: list[float] = []
    last_output: Any = None
    for _ in range(settings.repeats):
        start = time.perf_counter()
        last_output = compiled(*arguments)
        _block(last_output)
        raw_seconds.append(time.perf_counter() - start)

    timing = summarize_timings(raw_seconds, settings.deadline_seconds)
    timing["median_work_units_per_second"] = (
        float(work_units / timing["median_seconds"]) if timing["median_seconds"] > 0 else None
    )
    record = {
        **_record_identity(component, shape),
        "status": "ok",
        "failure_stage": None,
        "reason": None,
        "execution_kind": "jax_compiled_synchronized",
        "compile_seconds": compile_seconds,
        "warmup_seconds": warmup_seconds,
        "timing": timing,
        "work_units_per_execution": work_units,
        "estimated_live_bytes": estimated_bytes,
        "memory_guard_bytes": settings.max_estimated_bytes,
        "compiled_memory": compile_memory,
        "process_max_rss_bytes_before": rss_before,
        "process_max_rss_bytes_after": _rss_bytes(),
        "device_memory_before": before_stats,
        "device_memory_after": _memory_stats(device),
        "memory_evidence": {
            "compiled_program": _memory_availability(
                compile_memory, "JAX compiled executable memory_analysis"
            ),
            "process_peak_rss_available": True,
            "process_peak_rss_scope": (
                "cumulative process lifetime; before/after values are not an isolated interval peak"
            ),
            "device_allocator_before": _memory_availability(
                before_stats, "JAX device allocator snapshot before compilation"
            ),
            "device_allocator_after": _memory_availability(
                _memory_stats(device), "JAX device allocator snapshot after timed executions"
            ),
            "device_interval_peak_available": False,
            "device_interval_peak_reason": (
                "JAX provides allocator snapshots on some backends, not a portable resettable "
                "per-measurement peak counter"
            ),
        },
    }
    return record, compiled


def _attach_correctness(
    record: dict[str, Any], details: dict[str, Any], *, passed: bool, reason: str
) -> None:
    """Make a failed semantic postcheck impossible to retain an ``ok`` status."""
    record["correctness"] = {"passed": bool(passed), **details}
    if not passed:
        record["status"] = "failed"
        record["failure_stage"] = "untimed_correctness_postcheck"
        record["reason"] = reason


def _device_put(tree: Any, device: jax.Device) -> Any:
    def place(value: Any) -> Any:
        if isinstance(value, (jax.Array, np.ndarray, np.generic)):
            return jax.device_put(value, device)
        return value

    return jax.tree.map(place, tree)


def _physical_model() -> tuple[VersionAModel, VersionAActuator]:
    parameters: dict[str, Any] = load_params("cf21B_500")
    model = VersionAModel(
        mass=jnp.asarray(parameters["mass"]),
        gravity_vec=jnp.asarray(parameters["gravity_vec"]),
        inertia=jnp.asarray(parameters["J"]),
        inertia_inv=jnp.linalg.inv(jnp.asarray(parameters["J"])),
        drag_matrix=jnp.asarray(parameters["drag_matrix"]),
        wind_velocity=jnp.zeros(3),
        external_force=jnp.zeros(3),
        external_torque=jnp.zeros(3),
    )
    actuator = VersionAActuator(
        arm_length=jnp.asarray(parameters["L"]),
        thrust_to_torque=jnp.asarray(parameters["thrust2torque"]),
        mixing_matrix=jnp.asarray(parameters["mixing_matrix"]),
        thrust_min=jnp.asarray(parameters["thrust_min"]),
        thrust_max=jnp.asarray(parameters["thrust_max"]),
    )
    return model, actuator


def _actor_spec(policy_count: int, *, dtype: jnp.dtype = jnp.float32) -> SharedActorSpec:
    phase = 2.0 * jnp.pi * jnp.arange(policy_count, dtype=dtype) / policy_count
    codes = jnp.stack((jnp.cos(phase), jnp.sin(phase), jnp.cos(2.0 * phase)), axis=-1)
    velocities = jnp.stack(
        (0.45 + 0.15 * jnp.cos(phase), 0.25 * jnp.sin(phase), 0.08 * jnp.cos(phase)), axis=-1
    )
    structural_count = min(2, max(1, policy_count - 1))
    return SharedActorSpec(
        base_codes=codes,
        base_desired_velocities=velocities,
        base_durations=jnp.full((policy_count,), 0.55, dtype=dtype),
        adaptive_mask=(jnp.arange(policy_count) >= structural_count),
    )


def _circle_scenarios(batch_size: int, *, dtype: jnp.dtype = jnp.float32) -> CircleScenarioBatch:
    offsets = jnp.linspace(-0.12, 0.12, batch_size, dtype=dtype)
    centers = jnp.stack((jnp.full_like(offsets, 0.2), offsets, jnp.ones_like(offsets)), axis=-1)
    return CircleScenarioBatch(
        obstacle_centers=centers[:, None, :],
        obstacle_radii=jnp.full((batch_size, 1), 0.16, dtype=dtype),
        obstacle_mask=jnp.ones((batch_size, 1), dtype=bool),
        arena_lower=jnp.broadcast_to(jnp.array([-3.0, -3.0, 0.1], dtype=dtype), (batch_size, 3)),
        arena_upper=jnp.broadcast_to(jnp.array([3.0, 3.0, 3.0], dtype=dtype), (batch_size, 3)),
        speed_limit=jnp.full((batch_size,), 3.0, dtype=dtype),
    )


def _quad_problem(shape: ShapePoint, device: jax.Device) -> QuadProblem:
    model, actuator = _physical_model()
    spec = _actor_spec(shape.policies)
    scenarios = _circle_scenarios(shape.scenarios)
    y = jnp.linspace(-0.08, 0.08, shape.scenarios)
    initial = jnp.zeros((shape.scenarios, 13), dtype=jnp.float32)
    initial = initial.at[:, 0].set(-0.7).at[:, 1].set(y).at[:, 2].set(1.0)
    initial = initial.at[:, 6].set(1.0).at[:, 7].set(0.35)
    actor_config = SharedActorConfig(
        hidden_width=32,
        residual_scale=0.5,
        min_duration=0.1,
        max_duration=1.0,
        duration_transition=0.08,
    )
    quad_config = QuadPolicyConfig(acceleration_limit=4.0)
    barrier_config = VersionABarrierConfig(obstacle_clearance=0.08)
    learning_config = QuadLearningConfig(dt=0.02, horizon=shape.horizon, softmin_beta=30.0)
    loss_config = LibraryLossConfig(covariance_regularizer=0.05)
    safety = rigid_body_safety_batch_from_circles(
        scenarios, angular_rate_max=8.0, tilt_max_radians=1.13
    )
    params = initialize_shared_actor(
        jax.random.key(23), spec, dimension=3, n_obstacles=1, config=actor_config
    )
    problem = QuadProblem(
        model=model,
        actuator=actuator,
        spec=spec,
        scenarios=scenarios,
        safety=safety,
        initial_states=initial,
        params=params,
        actor_config=actor_config,
        quad_config=quad_config,
        barrier_config=barrier_config,
        learning_config=learning_config,
        loss_config=loss_config,
        targets=jnp.zeros((shape.policies, 9), dtype=jnp.float32),
        descriptor_scales=jnp.array([3.0, 3.0, 2.9, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
    )
    return _device_put(problem, device)


def _model_samples(problem: QuadProblem, sample_count: int) -> VersionAModelSamples:
    model = problem.model

    def repeat(value: Array) -> Array:
        return jnp.broadcast_to(value, (sample_count, *value.shape))

    scale = jnp.linspace(0.94, 1.06, sample_count, dtype=problem.initial_states.dtype)
    sampled_model = VersionAModel(
        mass=model.mass * scale,
        gravity_vec=repeat(model.gravity_vec),
        inertia=repeat(model.inertia),
        inertia_inv=repeat(model.inertia_inv),
        drag_matrix=repeat(model.drag_matrix),
        wind_velocity=repeat(model.wind_velocity)
        .at[:, 0]
        .set(jnp.linspace(-0.3, 0.3, sample_count)),
        external_force=repeat(model.external_force),
        external_torque=repeat(model.external_torque),
    )
    rotor_scale = jnp.linspace(0.9, 1.0, sample_count)[:, None]
    rotor_pattern = jnp.array([[1.0, 0.98, 0.96, 0.94]])
    return VersionAModelSamples(
        models=sampled_model,
        rotor_efficiency=rotor_scale * rotor_pattern,
        weights=jnp.full((sample_count,), 1.0 / sample_count),
        sample_valid=jnp.ones((sample_count,), dtype=bool),
        retained_variance_fraction=jnp.asarray(1.0),
        model_version=jnp.asarray(1, dtype=jnp.int32),
    )


def _single_safety(safety: RigidBodySafetySet) -> RigidBodySafetySet:
    return safety._replace(
        obstacle_centers=safety.obstacle_centers[0],
        obstacle_radii=safety.obstacle_radii[0],
        obstacle_mask=safety.obstacle_mask[0],
        arena_lower=safety.arena_lower[0],
        arena_upper=safety.arena_upper[0],
        speed_max=safety.speed_max[0],
        angular_rate_max=safety.angular_rate_max[0],
        tilt_max_radians=safety.tilt_max_radians[0],
    )


def _run_rollout(
    problem: QuadProblem, shape: ShapePoint, settings: BenchmarkSettings, device: jax.Device
) -> tuple[dict[str, Any], Any]:
    function = jax.jit(
        lambda params, initial: rollout_shared_quad_library(
            params,
            problem.spec,
            initial,
            problem.scenarios,
            problem.model,
            problem.actuator,
            dt=problem.learning_config.dt,
            horizon=shape.horizon,
            policy_gain=problem.learning_config.policy_gain,
            actor_config=problem.actor_config,
            quad_config=problem.quad_config,
        )
    )
    estimated = _estimate_component_bytes("rollout", shape)
    record, compiled = benchmark_jitted(
        "rollout",
        shape,
        function,
        (problem.params, problem.initial_states),
        settings,
        device,
        work_units=shape.policies * shape.scenarios * shape.horizon,
        estimated_bytes=estimated,
    )
    output = compiled(problem.params, problem.initial_states)
    _block(output)
    details = {
        "all_states_finite": bool(np.all(np.isfinite(np.asarray(output.states)))),
        "all_policy_steps_valid": bool(np.all(np.asarray(output.policy_valid))),
        "state_shape": list(output.states.shape),
        "policy_valid_shape": list(output.policy_valid.shape),
        "shape_matches_request": output.states.shape
        == (shape.policies, shape.scenarios, shape.horizon + 1, 13)
        and output.policy_valid.shape == (shape.policies, shape.scenarios, shape.horizon),
    }
    _attach_correctness(
        record,
        details,
        passed=all(
            details[name]
            for name in ("all_states_finite", "all_policy_steps_valid", "shape_matches_request")
        ),
        reason="forward rollout produced invalid/nonfinite output or an unexpected static shape",
    )
    record["latency_scope"] = "complete direct-wrench library rollout"
    return record, compiled


def _run_uncertain_rollout(
    problem: QuadProblem, shape: ShapePoint, settings: BenchmarkSettings, device: jax.Device
) -> tuple[dict[str, Any], Any]:
    samples = _model_samples(problem, shape.uncertainty_samples)
    function = jax.jit(
        lambda params, initial, models: rollout_shared_quad_library_under_uncertainty(
            params,
            problem.spec,
            initial,
            problem.scenarios,
            problem.model,
            models,
            problem.actuator,
            dt=problem.learning_config.dt,
            horizon=shape.horizon,
            policy_gain=problem.learning_config.policy_gain,
            actor_config=problem.actor_config,
            quad_config=problem.quad_config,
        )
    )
    estimated = _estimate_component_bytes("uncertain_rollout", shape)
    record, compiled = benchmark_jitted(
        "uncertain_rollout",
        shape,
        function,
        (problem.params, problem.initial_states, samples),
        settings,
        device,
        work_units=(shape.policies * shape.scenarios * shape.uncertainty_samples * shape.horizon),
        estimated_bytes=estimated,
    )
    output = compiled(problem.params, problem.initial_states, samples)
    _block(output)
    details = {
        "all_states_finite": bool(np.all(np.isfinite(np.asarray(output.states)))),
        "all_policy_steps_valid": bool(np.all(np.asarray(output.policy_valid))),
        "all_samples_valid": bool(np.all(np.asarray(output.sample_valid))),
        "state_shape": list(output.states.shape),
        "policy_valid_shape": list(output.policy_valid.shape),
        "shape_matches_request": output.states.shape
        == (shape.policies, shape.scenarios, shape.uncertainty_samples, shape.horizon + 1, 13)
        and output.policy_valid.shape
        == (shape.policies, shape.scenarios, shape.uncertainty_samples, shape.horizon),
    }
    _attach_correctness(
        record,
        details,
        passed=all(
            details[name]
            for name in (
                "all_states_finite",
                "all_policy_steps_valid",
                "all_samples_valid",
                "shape_matches_request",
            )
        ),
        reason=(
            "uncertainty rollout produced invalid/nonfinite output or an unexpected static shape"
        ),
    )
    record["latency_scope"] = "complete finite-R direct-wrench uncertainty rollout"
    return record, compiled


def _build_bptt_call(problem: QuadProblem) -> tuple[Any, tuple[Any, ...]]:
    functions = build_quad_actor_bptt_functions(
        problem.spec,
        problem.model,
        problem.actuator,
        problem.actor_config,
        problem.quad_config,
        problem.barrier_config,
        problem.learning_config,
        problem.loss_config,
        learning_rate=1e-3,
        burst_steps=1,
    )
    state = functions.initialize(problem.params)
    arguments = (
        state,
        problem.initial_states,
        problem.scenarios,
        problem.safety,
        problem.targets,
        problem.params,
        problem.descriptor_scales,
    )
    return functions.step, arguments


def _run_bptt(
    problem: QuadProblem, shape: ShapePoint, settings: BenchmarkSettings, device: jax.Device
) -> tuple[dict[str, Any], Any]:
    function, arguments = _build_bptt_call(problem)
    estimated = _estimate_component_bytes("bptt", shape)
    record, compiled = benchmark_jitted(
        "bptt",
        shape,
        function,
        arguments,
        settings,
        device,
        work_units=shape.policies * shape.scenarios * shape.horizon,
        estimated_bytes=estimated,
    )
    record["includes"] = ["full_13_state_rollout", "reverse_mode_gradient", "optimizer_update"]
    updated, metrics = compiled(*arguments)
    _block((updated, metrics))
    gradient_norm = float(np.asarray(metrics.gradient_norm))
    delta_norm = float(np.asarray(metrics.parameter_delta_norm))
    updated_finite = all(
        bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in jax.tree.leaves(updated.params)
    )
    details = {
        "update_accepted": bool(np.asarray(metrics.update_accepted)),
        "gradient_norm": gradient_norm,
        "parameter_delta_norm": delta_norm,
        "finite_nonzero_gradient": math.isfinite(gradient_norm) and gradient_norm > 0,
        "finite_nonzero_parameter_delta": math.isfinite(delta_norm) and delta_norm > 0,
        "updated_parameters_finite": updated_finite,
    }
    _attach_correctness(
        record,
        details,
        passed=all(
            details[name]
            for name in (
                "update_accepted",
                "finite_nonzero_gradient",
                "finite_nonzero_parameter_delta",
                "updated_parameters_finite",
            )
        ),
        reason="BPTT update was rejected, nonfinite, or did not change candidate parameters",
    )
    record["latency_scope"] = "one fused rollout + reverse-mode gradient + optimizer update"
    return record, compiled


def _runtime_problem(problem: QuadProblem) -> QuadProblem:
    """Take the first scenario while retaining the requested policy library and horizon."""
    return problem._replace(
        scenarios=jax.tree.map(lambda value: value[:1], problem.scenarios),
        safety=jax.tree.map(lambda value: value[:1], problem.safety),
        initial_states=problem.initial_states[:1],
    )


def _build_version_a_call(problem: QuadProblem, shape: ShapePoint) -> tuple[Any, tuple[Any, ...]]:
    runtime = _runtime_problem(problem)
    function = jax.jit(
        lambda state: version_a_runtime_step(
            state,
            jnp.array([1.0, 0.0, 1.0], dtype=state.dtype),
            jnp.zeros(3, dtype=state.dtype),
            runtime.params,
            runtime.spec,
            runtime.scenarios,
            runtime.safety,
            runtime.model,
            runtime.actuator,
            runtime.actor_config,
            runtime.quad_config,
            runtime.barrier_config,
            VersionAFilterConfig(),
            dt=runtime.learning_config.dt,
            certificate_horizon=shape.horizon,
            policy_gain=runtime.learning_config.policy_gain,
        )
    )
    return function, (runtime.initial_states[0],)


def _run_version_a(
    problem: QuadProblem, shape: ShapePoint, settings: BenchmarkSettings, device: jax.Device
) -> tuple[dict[str, Any], Any]:
    function, arguments = _build_version_a_call(problem, shape)
    estimated = _estimate_component_bytes("version_a", shape)
    record, compiled = benchmark_jitted(
        "version_a",
        shape,
        function,
        arguments,
        settings,
        device,
        work_units=shape.policies * shape.horizon,
        estimated_bytes=estimated,
    )
    record["includes"] = [
        "hard_library_certificates",
        "certificate_state_jacobian",
        "policy_selection",
        "continuous_filter",
        "independent_postchecks",
        "held_step_postcheck",
    ]
    output = compiled(*arguments)
    _block(output)
    kkt_max = float(
        max(
            np.asarray(output.continuous_filter.qp.primal_residual),
            np.asarray(output.continuous_filter.qp.dual_residual),
            np.asarray(output.continuous_filter.qp.stationarity_residual),
            np.asarray(output.continuous_filter.qp.complementarity_residual),
        )
    )
    details = {
        "has_certificate": bool(np.asarray(output.continuous_filter.has_certificate)),
        "proposal_interval_accepted": bool(np.asarray(output.proposal_interval_accepted)),
        "degraded": bool(np.asarray(output.degraded)),
        "applied_continuous_postcheck_passed": bool(
            np.asarray(output.applied_continuous_postcheck.passed)
        ),
        "action_finite": bool(np.all(np.isfinite(np.asarray(output.action)))),
        "next_state_finite": bool(np.all(np.isfinite(np.asarray(output.next_state)))),
        "applied_interval_margin": float(np.asarray(output.applied_interval_margin)),
        "qp_kkt_max_residual": kkt_max,
    }
    passed = (
        details["has_certificate"]
        and not details["degraded"]
        and details["applied_continuous_postcheck_passed"]
        and details["action_finite"]
        and details["next_state_finite"]
        and math.isfinite(details["applied_interval_margin"])
        and math.isfinite(kkt_max)
    )
    _attach_correctness(
        record,
        details,
        passed=passed,
        reason="Version-A command-ready decision degraded or failed its independent postchecks",
    )
    record["latency_scope"] = (
        "command-ready full Version-A decision at one 20 ms control interval; "
        "B/R requested axes are not consumed"
    )
    record["deadline_interpretation"] = (
        "20 ms is the configured 50 Hz budget, not a real-time guarantee"
    )
    return record, compiled


def _build_qp_call(problem: QuadProblem) -> tuple[Any, tuple[Any, ...]]:
    state = problem.initial_states[0]
    single_safety = _single_safety(problem.safety)
    motor = validated_motor_polytope(problem.actuator, state.dtype)
    analytic = continuous_safety_halfspaces(
        state, problem.model, single_safety, problem.barrier_config
    )
    policy_row = jnp.zeros((1, 4), dtype=state.dtype)
    policy_bound = jnp.ones((1,), dtype=state.dtype)
    matrix = jnp.concatenate((motor.matrix, analytic.matrix, policy_row), axis=0)
    bound = jnp.concatenate((motor.upper_bound, analytic.upper_bound, policy_bound), axis=0)
    nominal = motor.midpoint_wrench
    weight = jnp.ones((4,), dtype=state.dtype)
    function = jax.jit(
        lambda current_nominal: project_affine_polytope(
            current_nominal, weight, matrix, bound, tolerance=2e-6, rank_tolerance=1e-7
        )
    )
    return function, (nominal,)


def _run_qp(
    problem: QuadProblem, shape: ShapePoint, settings: BenchmarkSettings, device: jax.Device
) -> tuple[dict[str, Any], Any]:
    function, arguments = _build_qp_call(problem)
    estimated = _estimate_component_bytes("qp", shape)
    record, compiled = benchmark_jitted(
        "qp", shape, function, arguments, settings, device, work_units=1, estimated_bytes=estimated
    )
    record["shape_scope"] = "QP constraint count is physical-model-defined; K/B/R/H are n/a"
    output = compiled(*arguments)
    _block(output)
    details = {
        "feasible": bool(np.asarray(output.feasible)),
        "input_valid": bool(np.asarray(output.input_valid)),
        "primal_residual": float(np.asarray(output.primal_residual)),
        "dual_residual": float(np.asarray(output.dual_residual)),
        "stationarity_residual": float(np.asarray(output.stationarity_residual)),
        "complementarity_residual": float(np.asarray(output.complementarity_residual)),
    }
    passed = (
        details["feasible"]
        and details["input_valid"]
        and all(
            math.isfinite(details[name])
            for name in (
                "primal_residual",
                "dual_residual",
                "stationarity_residual",
                "complementarity_residual",
            )
        )
    )
    _attach_correctness(
        record,
        details,
        passed=passed,
        reason="active-set QP output was infeasible, invalid, or nonfinite",
    )
    record["latency_scope"] = "one physical-model-defined four-variable active-set QP"
    return record, compiled


def _validation_inputs(shape: ShapePoint) -> tuple[Any, ...]:
    k, batch = shape.policies, shape.scenarios
    active_params = {"adaptive": np.zeros((k, 3), dtype=np.float32)}
    candidate_params = {"adaptive": np.full((k, 3), 0.01, dtype=np.float32)}
    structural = {"codes": np.asarray(_actor_spec(k).base_codes)}
    active = create_active_snapshot(
        active_params, version=1, model_version=2, structural_core=structural
    )
    candidate = create_candidate_snapshot(candidate_params, version=2, base_active=active)
    scenario_axis = np.linspace(0.10, 0.30, batch, dtype=np.float64)
    policy_axis = np.linspace(0.0, 0.05, k, dtype=np.float64)[:, None]
    active_margins = scenario_axis[None, :] + policy_axis
    candidate_margins = active_margins + 0.01
    descriptor_dimension = max(2, min(9, k))
    descriptors = np.zeros((k, descriptor_dimension), dtype=np.float64)
    descriptors[np.arange(k), np.arange(k) % descriptor_dimension] = np.arange(1, k + 1)
    evidence = HardValidationEvidence(
        current_policy_margins=candidate_margins[:, 0],
        candidate_local_policy_margins=candidate_margins,
        active_local_policy_margins=active_margins,
        candidate_descriptors=descriptors,
        descriptor_scales=np.ones(descriptor_dimension),
        feasibility_margins=np.ones((k, batch)),
        runtime_seconds=np.full((batch,), 1e-4),
        validation_set_digest=f"benchmark-{shape.name}",
    )
    thresholds = HardValidationThresholds(
        minimum_coverage=1.0,
        minimum_redundancy=1,
        minimum_diversity=0.0,
        maximum_runtime_seconds=1.0,
    )
    return active, candidate, evidence, thresholds


def _run_validation(shape: ShapePoint, settings: BenchmarkSettings) -> tuple[dict[str, Any], None]:
    active, candidate, evidence, thresholds = _validation_inputs(shape)

    def evaluate() -> Any:
        return hard_validate_candidate(
            active, candidate, evidence, thresholds, current_model_version=2
        )

    warmups: list[float] = []
    for _ in range(settings.warmups):
        start = time.perf_counter()
        report = evaluate()
        warmups.append(time.perf_counter() - start)
        if not report.verify_integrity():
            raise RuntimeError("hard validation produced an invalid report digest")
    samples: list[float] = []
    report = None
    for _ in range(settings.repeats):
        start = time.perf_counter()
        report = evaluate()
        samples.append(time.perf_counter() - start)
    assert report is not None
    timing = summarize_timings(samples, settings.deadline_seconds)
    timing["median_work_units_per_second"] = (
        float(shape.policies * shape.scenarios / timing["median_seconds"])
        if timing["median_seconds"] > 0
        else None
    )
    report_integrity = report.verify_integrity()
    passed = bool(report.passed and report_integrity)
    identity = _record_identity("validation", shape)
    return (
        {
            **identity,
            "status": "ok" if passed else "failed",
            "failure_stage": None if passed else "untimed_correctness_postcheck",
            "reason": None if passed else "hard candidate admission report failed or was invalid",
            "execution_kind": "host_numpy_not_jittable",
            "scope": {
                "includes": [
                    "snapshot_integrity_and_freshness_checks",
                    "hard_non_regression_gate_reductions",
                    "report_digest_construction",
                ],
                "excludes": [
                    "candidate_rollouts",
                    "active_rollouts",
                    "trajectory_descriptor_generation",
                    "feasibility_evidence_generation",
                    "runtime_evidence_generation",
                ],
                "end_to_end_timing_source": (
                    "campaign event traces include evidence generation and are authoritative for "
                    "end-to-end candidate validation latency"
                ),
            },
            "compile_seconds": None,
            "compile_not_applicable_reason": (
                "snapshot hashing and immutable host admission reports intentionally use "
                "NumPy/Python"
            ),
            "warmup_seconds": warmups,
            "timing": timing,
            "work_units_per_execution": shape.policies * shape.scenarios,
            "report_passed": report.passed,
            "report_integrity": report_integrity,
            "report_digest": report.digest,
            "failed_gates": list(report.failed_gate_names),
            "correctness": {
                "passed": passed,
                "report_passed": bool(report.passed),
                "report_integrity": bool(report_integrity),
            },
            "latency_scope": "candidate admission gate only; evidence generation is excluded",
            "deadline_interpretation": (
                "The common deadline is descriptive for this host-only gate and is not the "
                "end-to-end candidate latency budget."
            ),
            "estimated_live_bytes": _estimate_component_bytes("validation", shape),
            "memory_guard_bytes": settings.max_estimated_bytes,
            "process_max_rss_bytes_after": _rss_bytes(),
        },
        None,
    )


def _version_b_problem(shape: ShapePoint, device_name: str, device: jax.Device) -> VersionBProblem:
    sim = Sim(
        dynamics=Dynamics.first_principles,
        control=Control.force_torque,
        integrator=Integrator.symplectic_euler,
        freq=500,
        force_torque_freq=500,
        device=device_name,
        enable_mjx=False,
    )
    controller = sim.data.controls.force_torque.params
    physical = sim.data.params
    mass = physical.mass[0, 0, 0]
    gravity = physical.gravity_vec
    hover = jnp.array([mass * -gravity[2], 0.0, 0.0, 0.0])
    hover_motor_force = hover[0] / 4
    rpm2thrust = physical.rpm2thrust
    hover_rpm = (
        -rpm2thrust[1]
        + jnp.sqrt(rpm2thrust[1] ** 2 - 4 * rpm2thrust[2] * (rpm2thrust[0] - hover_motor_force))
    ) / (2 * rpm2thrust[2])
    data = sim.data.replace(
        states=sim.data.states.replace(
            pos=sim.data.states.pos.at[0, 0, 2].set(1.0),
            rotor_vel=jnp.full_like(sim.data.states.rotor_vel, hover_rpm),
        )
    )
    model = VersionAModel(
        mass=mass,
        gravity_vec=gravity,
        inertia=physical.J[0, 0],
        inertia_inv=physical.J_inv[0, 0],
        drag_matrix=physical.drag_matrix,
        wind_velocity=jnp.zeros(3),
        external_force=jnp.zeros(3),
        external_torque=jnp.zeros(3),
    )
    actuator = VersionAActuator(
        arm_length=controller["L"],
        thrust_to_torque=controller["thrust2torque"],
        mixing_matrix=controller["mixing_matrix"],
        thrust_min=controller["thrust_min"],
        thrust_max=controller["thrust_max"],
    )
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.zeros((1, 1, 3)),
        obstacle_radii=jnp.ones((1, 1)),
        obstacle_mask=jnp.zeros((1, 1), dtype=bool),
        arena_lower=jnp.array([[-4.0, -4.0, 0.1]]),
        arena_upper=jnp.array([[4.0, 4.0, 4.1]]),
        speed_limit=jnp.array([8.0]),
    )
    safety = rigid_body_safety_batch_from_circles(
        scenarios, angular_rate_max=20.0, tilt_max_radians=1.4
    )
    spec = _actor_spec(shape.policies)
    actor_config = SharedActorConfig(hidden_width=32, max_duration=1.0)
    actor_params = initialize_shared_actor(
        jax.random.key(31), spec, dimension=3, n_obstacles=1, config=actor_config
    )
    problem = VersionBProblem(
        data=data,
        nominal=hover,
        params=actor_params,
        spec=spec,
        scenarios=scenarios,
        safety=safety,
        model=model,
        actuator=actuator,
        actor_config=actor_config,
        quad_config=QuadPolicyConfig(),
        barrier_config=VersionABarrierConfig(),
        one_step=build_unclipped_full_stack_step(sim),
        action_lower=jnp.array([0.0, -1.0, -1.0, -1.0]),
        action_upper=jnp.array([10.0, 1.0, 1.0, 1.0]),
        weight=jnp.ones(4),
        trust_radius=jnp.array([10.0, 1.0, 1.0, 1.0]),
        config=VersionBRuntimeConfig(
            n_substeps=2,
            certificate_horizon=shape.horizon,
            policy_gain=1.5,
            decay=0.99,
            tolerance=2e-5,
            qp_iterations=32,
        ),
    )
    return _device_put(problem, device)


def _run_version_b(
    shape: ShapePoint, settings: BenchmarkSettings, device_name: str, device: jax.Device
) -> tuple[dict[str, Any], Any]:
    estimated = _estimate_component_bytes("version_b", shape)
    # This path nests exact full-stack rollouts inside finite-difference nonlinear filtering.
    # A bounded probe is intentional; a larger request is recorded, never silently downscaled.
    if shape.policies > 4 or shape.horizon > 4 or shape.scenarios != 1:
        return (
            _skipped_record(
                "version_b",
                shape,
                "explicit full-stack probe guard requires K<=4, B=1, H<=4",
                estimated,
                settings,
            ),
            None,
        )
    problem = _version_b_problem(shape, device_name, device)
    function = jax.jit(
        lambda data, nominal: version_b_runtime_step(
            data,
            nominal,
            problem.params,
            problem.spec,
            problem.scenarios,
            problem.safety,
            problem.model,
            problem.actuator,
            problem.actor_config,
            problem.quad_config,
            problem.barrier_config,
            problem.one_step,
            problem.action_lower,
            problem.action_upper,
            problem.weight,
            problem.trust_radius,
            problem.config,
        )
    )
    record, compiled = benchmark_jitted(
        "version_b",
        shape,
        function,
        (problem.data, problem.nominal),
        settings,
        device,
        work_units=shape.policies * shape.horizon * problem.config.n_substeps,
        estimated_bytes=estimated,
    )
    record["includes"] = [
        "force_torque_control",
        "motor_allocation_and_clipping_audit",
        "rotor_lag",
        "unclamped_first_principles_dynamics",
        "swept_interval_postcheck",
        "same_policy_equal_horizon_residual",
    ]
    output = compiled(problem.data, problem.nominal)
    _block(output)
    next_state_finite = all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree.leaves(output.next_data.states)
    )
    details = {
        "has_certificate": bool(np.asarray(output.has_certificate)),
        "applied_accepted": bool(np.asarray(output.applied_accepted)),
        "degraded": bool(np.asarray(output.degraded)),
        "action_finite": bool(np.all(np.isfinite(np.asarray(output.action)))),
        "next_state_finite": next_state_finite,
        "postcheck_replay_error": float(np.asarray(output.postcheck_replay_error)),
        "applied_exact_residual": float(np.asarray(output.applied_exact_residual)),
    }
    passed = (
        details["has_certificate"]
        and details["applied_accepted"]
        and not details["degraded"]
        and details["action_finite"]
        and details["next_state_finite"]
        and math.isfinite(details["postcheck_replay_error"])
        and math.isfinite(details["applied_exact_residual"])
    )
    _attach_correctness(
        record,
        details,
        passed=passed,
        reason="Version-B command-ready decision degraded or failed its exact replay postchecks",
    )
    record["latency_scope"] = (
        "command-ready full Version-B diagnostic decision over two 500 Hz held substeps (4 ms); "
        "B/R requested axes are not consumed"
    )
    record["deadline_interpretation"] = (
        "The common 20 ms deadline is descriptive; this two-substep diagnostic is not a 50 Hz "
        "held-command configuration."
    )
    return record, compiled


def _run_component(
    component: ComponentName,
    shape: ShapePoint,
    problem: QuadProblem | None,
    settings: BenchmarkSettings,
    device_name: str,
    device: jax.Device,
) -> tuple[dict[str, Any], Any]:
    estimated = _estimate_component_bytes(component, shape)
    if estimated > settings.max_estimated_bytes:
        return (
            _skipped_record(
                component,
                shape,
                "pre-compilation memory guard: estimated live set exceeds --max-estimated-gib",
                estimated,
                settings,
            ),
            None,
        )
    try:
        if component == "validation":
            return _run_validation(shape, settings)
        if component == "version_b":
            return _run_version_b(shape, settings, device_name, device)
        if problem is None:
            raise RuntimeError("direct-wrench benchmark problem was not constructed")
        if component == "rollout":
            return _run_rollout(problem, shape, settings, device)
        if component == "uncertain_rollout":
            return _run_uncertain_rollout(problem, shape, settings, device)
        if component == "bptt":
            return _run_bptt(problem, shape, settings, device)
        if component == "version_a":
            return _run_version_a(problem, shape, settings, device)
        if component == "qp":
            return _run_qp(problem, shape, settings, device)
        raise ValueError(f"unknown benchmark component: {component}")
    except (MemoryError, RuntimeError, ValueError, FloatingPointError) as error:
        return _failed_record(component, shape, error, estimated, settings), None


def _parse_components(value: str) -> tuple[ComponentName, ...]:
    if value.strip().lower() == "all":
        return ALL_COMPONENTS
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(names) - set(ALL_COMPONENTS))
    if invalid:
        raise ValueError(f"unknown components: {', '.join(invalid)}")
    if not names:
        raise ValueError("at least one component is required")
    return names  # type: ignore[return-value]


def _resolve_contention(value: str, controller_device: jax.Device) -> tuple[str, ...]:
    requested = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if requested == ("none",):
        return ()
    if requested == ("auto",):
        return ("cpu", "gpu") if controller_device.platform == "gpu" else ("cpu",)
    invalid = sorted(set(requested) - {"cpu", "gpu"})
    if invalid or not requested:
        raise ValueError("contention must be auto, none, cpu, gpu, or a comma-separated pair")
    return requested


def _compile_without_timing(
    function: Any, arguments: tuple[Any, ...]
) -> tuple[Any, float, float, Any]:
    start = time.perf_counter()
    compiled = function.lower(*arguments).compile()
    compile_seconds = time.perf_counter() - start
    warmup_start = time.perf_counter()
    output = compiled(*arguments)
    _block(output)
    warmup_seconds = time.perf_counter() - warmup_start
    return compiled, compile_seconds, warmup_seconds, output


def _contention_experiment(
    shape: ShapePoint,
    settings: BenchmarkSettings,
    controller_problem: QuadProblem,
    controller_device: jax.Device,
    worker_platforms: tuple[str, ...],
) -> dict[str, Any]:
    """Measure one compiled Version-A controller idle and under real BPTT device work."""
    controller_function, controller_arguments = _build_version_a_call(controller_problem, shape)
    controller, controller_compile, controller_warmup, controller_output = _compile_without_timing(
        controller_function, controller_arguments
    )
    controller_correct = (
        bool(np.asarray(controller_output.continuous_filter.has_certificate))
        and bool(np.asarray(controller_output.applied_continuous_postcheck.passed))
        and not bool(np.asarray(controller_output.degraded))
        and bool(np.all(np.isfinite(np.asarray(controller_output.action))))
    )
    if not controller_correct:
        return {
            "status": "failed",
            "failure_stage": "controller_warmup_correctness_postcheck",
            "reason": "contention controller degraded or failed its public action postcheck",
            "controller": COMPONENT_LABELS["version_a"],
            "requested_shape": asdict(shape),
            "effective_controller_shape": _effective_shape("version_a", shape),
            "controller_device": _device_attributes(controller_device),
            "controller_compile_seconds": controller_compile,
            "controller_warmup_seconds": controller_warmup,
            "idle_controller_timing": None,
            "loaded_controller_timings": [],
        }

    def measure() -> dict[str, Any]:
        samples: list[float] = []
        for _ in range(settings.repeats):
            start = time.perf_counter()
            _block(controller(*controller_arguments))
            samples.append(time.perf_counter() - start)
        return summarize_timings(samples, settings.deadline_seconds)

    idle = measure()
    loaded: list[dict[str, Any]] = []
    final_scale = max(shape.policies, shape.scenarios, shape.horizon) >= 50
    worker_startup_timeout_seconds = 600.0 if final_scale else 60.0
    for worker_platform in worker_platforms:
        available = jax.devices(worker_platform)
        if not available:
            loaded.append(
                {
                    "worker_platform": worker_platform,
                    "status": "unavailable",
                    "reason": f"JAX exposes no {worker_platform} device",
                }
            )
            continue
        worker_device = available[0]
        worker_problem = _quad_problem(shape, worker_device)
        worker_function, worker_arguments = _build_bptt_call(worker_problem)
        try:
            worker, worker_compile, worker_warmup, worker_output = _compile_without_timing(
                worker_function, worker_arguments
            )
        except (MemoryError, RuntimeError, ValueError) as error:
            loaded.append(
                {
                    "worker_platform": worker_platform,
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        worker_state, worker_metrics = worker_output
        worker_gradient = float(np.asarray(worker_metrics.gradient_norm))
        worker_delta = float(np.asarray(worker_metrics.parameter_delta_norm))
        worker_correct = (
            bool(np.asarray(worker_metrics.update_accepted))
            and math.isfinite(worker_gradient)
            and worker_gradient > 0
            and math.isfinite(worker_delta)
            and worker_delta > 0
            and all(
                bool(np.all(np.isfinite(np.asarray(leaf))))
                for leaf in jax.tree.leaves(worker_state.params)
            )
        )
        if not worker_correct:
            loaded.append(
                {
                    "worker_platform": worker_platform,
                    "status": "failed",
                    "reason": "BPTT contention worker failed its untimed correctness postcheck",
                    "worker_compile_seconds": worker_compile,
                    "worker_warmup_seconds": worker_warmup,
                }
            )
            continue

        stop = threading.Event()
        ready = threading.Event()
        worker_samples: list[float] = []
        worker_completed = [0]
        worker_errors: list[str] = []

        def work() -> None:
            try:
                start = time.perf_counter()
                _block(worker(*worker_arguments))
                worker_samples.append(time.perf_counter() - start)
                worker_completed[0] += 1
                ready.set()
                while not stop.is_set():
                    start = time.perf_counter()
                    _block(worker(*worker_arguments))
                    elapsed = time.perf_counter() - start
                    worker_completed[0] += 1
                    if len(worker_samples) < max(100, settings.repeats):
                        worker_samples.append(elapsed)
            except BaseException as error:  # pragma: no cover - backend/runtime dependent
                worker_errors.append(f"{type(error).__name__}: {error}")
                ready.set()

        thread = threading.Thread(target=work, name=f"da-plcbf-{worker_platform}-bptt")
        thread.start()
        if not ready.wait(timeout=worker_startup_timeout_seconds):
            stop.set()
            thread.join(timeout=worker_startup_timeout_seconds)
            loaded.append(
                {
                    "worker_platform": worker_platform,
                    "status": "failed",
                    "reason": (
                        "BPTT worker did not finish its first synchronized update within "
                        f"{worker_startup_timeout_seconds:.0f} seconds"
                    ),
                }
            )
            continue
        try:
            loaded_timing = measure()
        finally:
            stop.set()
            thread.join(timeout=worker_startup_timeout_seconds)
        if thread.is_alive():
            loaded.append(
                {
                    "worker_platform": worker_platform,
                    "status": "failed",
                    "reason": "BPTT worker did not stop after its synchronized execution",
                }
            )
            continue
        if worker_errors:
            loaded.append(
                {"worker_platform": worker_platform, "status": "failed", "reason": worker_errors[0]}
            )
            continue
        loaded.append(
            {
                "worker_platform": worker_platform,
                "status": "ok",
                "worker_compile_seconds": worker_compile,
                "worker_warmup_seconds": worker_warmup,
                "worker_completed_updates": worker_completed[0],
                "worker_timing": summarize_timings(worker_samples, settings.deadline_seconds),
                "worker_raw_samples_truncated": worker_completed[0] > len(worker_samples),
                "controller_timing": loaded_timing,
                "median_slowdown_ratio": (loaded_timing["median_seconds"] / idle["median_seconds"]),
                "deadline_miss_delta": (
                    loaded_timing["deadline_miss_fraction"] - idle["deadline_miss_fraction"]
                ),
            }
        )
    successful_workers = sum(record["status"] == "ok" for record in loaded)
    overall_status = "ok" if successful_workers == len(worker_platforms) else "partial"
    if successful_workers == 0:
        overall_status = "failed"
    return {
        "status": overall_status,
        "failure_stage": None if overall_status == "ok" else "contention_worker",
        "reason": (
            None
            if overall_status == "ok"
            else (
                f"{successful_workers}/{len(worker_platforms)} requested contention workers "
                "succeeded"
            )
        ),
        "controller": COMPONENT_LABELS["version_a"],
        "requested_shape": asdict(shape),
        "effective_controller_shape": _effective_shape("version_a", shape),
        "effective_worker_shape": _effective_shape("bptt", shape),
        "controller_device": _device_attributes(controller_device),
        "controller_compile_seconds": controller_compile,
        "controller_warmup_seconds": controller_warmup,
        "worker_startup_timeout_seconds": worker_startup_timeout_seconds,
        "idle_controller_timing": idle,
        "loaded_controller_timings": loaded,
        "interpretation": (
            "This is process-local device contention, not an operating-system real-time guarantee."
        ),
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _provenance(device: jax.Device, arguments: list[str], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *arguments],
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "cpu_model": _cpu_model(),
        },
        "software": {
            "crazyflow": crazyflow_version,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
        },
        "source": source,
        "device": _device_attributes(device),
        "visible_devices": [_device_attributes(item) for item in jax.local_devices()],
    }


def run_benchmark(
    *,
    device_name: str,
    preset_name: str,
    components: tuple[ComponentName, ...],
    settings: BenchmarkSettings,
    contention: str,
    command_arguments: list[str] | None = None,
) -> dict[str, Any]:
    """Run one named preset and return a JSON-ready benchmark document."""
    settings.validate()
    if preset_name not in PRESETS:
        raise ValueError(f"unknown preset: {preset_name}")
    devices = jax.devices(device_name)
    if not devices:
        raise RuntimeError(f"JAX exposes no device for platform {device_name!r}")
    device = devices[0]
    source_before = _source_tree_binding()
    shapes = PRESETS[preset_name]
    contention_shape = CONTENTION_SHAPES[preset_name]
    contention_shape.validate()
    worker_platforms = _resolve_contention(contention, device)
    records: list[dict[str, Any]] = []
    for shape in shapes:
        shape.validate()
        needs_problem = any(
            component not in ("validation", "version_b") for component in components
        )
        problem = None
        problem_error: BaseException | None = None
        if needs_problem:
            try:
                problem = _quad_problem(shape, device)
            except (MemoryError, RuntimeError, ValueError, FloatingPointError) as error:
                problem_error = error
        for component in components:
            if problem_error is not None and component not in ("validation", "version_b"):
                record = _failed_record(
                    component,
                    shape,
                    problem_error,
                    _estimate_component_bytes(component, shape),
                    settings,
                )
            else:
                record, _ = _run_component(component, shape, problem, settings, device_name, device)
            records.append(record)

    contention_record: dict[str, Any]
    if worker_platforms:
        estimated = _estimate_component_bytes(
            "version_a", contention_shape
        ) + _estimate_component_bytes("bptt", contention_shape)
        if estimated > settings.max_estimated_bytes:
            contention_record = {
                "status": "skipped",
                "reason": "combined controller/worker estimate exceeds memory guard",
                "requested_shape": asdict(contention_shape),
                "estimated_live_bytes": estimated,
                "memory_guard_bytes": settings.max_estimated_bytes,
            }
        else:
            try:
                contention_problem = _quad_problem(contention_shape, device)
                contention_record = _contention_experiment(
                    contention_shape, settings, contention_problem, device, worker_platforms
                )
            except (MemoryError, RuntimeError, ValueError) as error:
                contention_record = {
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                    "requested_shape": asdict(contention_shape),
                    "estimated_live_bytes": estimated,
                    "memory_guard_bytes": settings.max_estimated_bytes,
                }
    else:
        contention_record = {
            "status": "not_requested",
            "reason": "pass --contention auto, cpu, gpu, or cpu,gpu to measure interference",
            "requested_shape": asdict(contention_shape),
        }

    source_after = _source_tree_binding()
    if source_after != source_before:
        raise RuntimeError("source/git state changed while the performance benchmark was running")
    status_counts = {
        status: sum(record["status"] == status for record in records)
        for status in ("ok", "failed", "skipped")
    }
    document = {
        "schema": PERFORMANCE_ARTIFACT_SCHEMA,
        "schema_version": PERFORMANCE_ARTIFACT_SCHEMA_VERSION,
        "provenance": _provenance(device, command_arguments or [], source_after),
        "request": {
            "device": device_name,
            "preset": preset_name,
            "components": list(components),
            "settings": asdict(settings),
            "contention": contention,
        },
        "preset_shapes": [asdict(shape) for shape in shapes],
        "measurements": records,
        "contention_experiment": contention_record,
        "completion": {
            "requested_measurements": len(shapes) * len(components),
            "recorded_measurements": len(records),
            "status_counts": status_counts,
            "all_requested_measurements_ok": status_counts
            == {"ok": len(records), "failed": 0, "skipped": 0},
            "contention_status": contention_record["status"],
            "source_clean": not source_after["git_dirty"],
            "claim_grade_source_eligible": not source_after["git_dirty"],
        },
        "claim_caveats": [
            "These timings measure this repository and configuration, not paper-equivalent "
            "training.",
            "Finite R is a supplied-scenario minimum, not a distribution-free robustness "
            "guarantee.",
            "Synchronized process latency is not a hard real-time or hardware-deployment "
            "guarantee.",
            "Percentiles describe only the retained finite sample; they are not population bounds.",
            "Compilation, warmups, timed control, candidate validation, and contention are "
            "separated.",
            "The admission-gate timing excludes rollout, descriptor, feasibility, and other "
            "evidence generation; campaign events provide end-to-end validation latency.",
            "A skipped/failed shape is evidence of no measurement and must not be treated as "
            "success.",
            "Artifact hashing detects corruption and inconsistent derived fields, but is not a "
            "signature or trusted hardware timing attestation.",
        ],
    }
    document = _jsonable(document)
    document["integrity"] = {
        "algorithm": "sha256-canonical-json",
        "digest": _canonical_digest(document),
        "authenticity_limit": (
            "Integrity validation is not an external signature or trusted hardware attestation."
        ),
    }
    verify_performance_artifact(document)
    return document


def _require_exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _finite_number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or (nonnegative and numeric < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


def _require_close(actual: Any, expected: float, name: str) -> None:
    numeric = _finite_number(actual, name)
    if not math.isclose(numeric, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{name} is inconsistent with retained raw samples")


def _verify_timing(
    value: Any,
    *,
    settings: BenchmarkSettings,
    name: str,
    expected_samples: int | None,
    work_units: int | None = None,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    expected_keys = {
        "raw_seconds",
        "median_seconds",
        "p95_seconds",
        "p99_seconds",
        "worst_seconds",
        "deadline_seconds",
        "deadline_misses",
        "deadline_miss_fraction",
    }
    if work_units is not None:
        expected_keys.add("median_work_units_per_second")
    _require_exact_keys(value, expected_keys, name)
    raw = value["raw_seconds"]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name}.raw_seconds must be a nonempty array")
    if expected_samples is not None and len(raw) != expected_samples:
        raise ValueError(f"{name} raw sample count does not match the request")
    normalized_raw = [
        _finite_number(sample, f"{name}.raw_seconds[{index}]", nonnegative=True)
        for index, sample in enumerate(raw)
    ]
    if any(sample <= 0 for sample in normalized_raw):
        raise ValueError(f"{name}.raw_seconds must be strictly positive")
    recomputed = summarize_timings(normalized_raw, settings.deadline_seconds)
    for field in ("median_seconds", "p95_seconds", "p99_seconds", "worst_seconds"):
        _require_close(value[field], recomputed[field], f"{name}.{field}")
    _require_close(value["deadline_seconds"], settings.deadline_seconds, f"{name}.deadline_seconds")
    if value["deadline_misses"] != recomputed["deadline_misses"]:
        raise ValueError(f"{name}.deadline_misses is inconsistent with raw samples")
    _require_close(
        value["deadline_miss_fraction"],
        recomputed["deadline_miss_fraction"],
        f"{name}.deadline_miss_fraction",
    )
    if work_units is not None:
        expected_rate = (
            work_units / recomputed["median_seconds"] if recomputed["median_seconds"] > 0 else None
        )
        if expected_rate is None:
            if value["median_work_units_per_second"] is not None:
                raise ValueError(f"{name} zero-duration throughput must be null")
        else:
            _require_close(
                value["median_work_units_per_second"],
                expected_rate,
                f"{name}.median_work_units_per_second",
            )


def _work_units(component: ComponentName, shape: ShapePoint) -> int:
    if component in ("rollout", "bptt"):
        return shape.policies * shape.scenarios * shape.horizon
    if component == "uncertain_rollout":
        return shape.policies * shape.scenarios * shape.uncertainty_samples * shape.horizon
    if component == "version_a":
        return shape.policies * shape.horizon
    if component == "version_b":
        return shape.policies * shape.horizon * 2
    if component == "validation":
        return shape.policies * shape.scenarios
    return 1


def _verify_correctness(record: dict[str, Any], component: ComponentName) -> bool:
    correctness = record.get("correctness")
    if not isinstance(correctness, dict) or not isinstance(correctness.get("passed"), bool):
        raise ValueError(f"{record['measurement_id']} requires explicit correctness.passed")
    if component == "rollout":
        _require_exact_keys(
            correctness,
            {
                "passed",
                "all_states_finite",
                "all_policy_steps_valid",
                "state_shape",
                "policy_valid_shape",
                "shape_matches_request",
            },
            "rollout.correctness",
        )
        requested = record["requested_shape"]
        expected_states = [
            requested["policies"],
            requested["scenarios"],
            requested["horizon"] + 1,
            13,
        ]
        expected_valid = [requested["policies"], requested["scenarios"], requested["horizon"]]
        shape_matches = (
            correctness["state_shape"] == expected_states
            and correctness["policy_valid_shape"] == expected_valid
        )
        if correctness["shape_matches_request"] is not shape_matches:
            raise ValueError("rollout shape_matches_request is inconsistent with retained shapes")
        names = ("all_states_finite", "all_policy_steps_valid", "shape_matches_request")
    elif component == "uncertain_rollout":
        _require_exact_keys(
            correctness,
            {
                "passed",
                "all_states_finite",
                "all_policy_steps_valid",
                "all_samples_valid",
                "state_shape",
                "policy_valid_shape",
                "shape_matches_request",
            },
            "uncertain_rollout.correctness",
        )
        requested = record["requested_shape"]
        expected_states = [
            requested["policies"],
            requested["scenarios"],
            requested["uncertainty_samples"],
            requested["horizon"] + 1,
            13,
        ]
        expected_valid = [
            requested["policies"],
            requested["scenarios"],
            requested["uncertainty_samples"],
            requested["horizon"],
        ]
        shape_matches = (
            correctness["state_shape"] == expected_states
            and correctness["policy_valid_shape"] == expected_valid
        )
        if correctness["shape_matches_request"] is not shape_matches:
            raise ValueError(
                "uncertainty rollout shape_matches_request is inconsistent with retained shapes"
            )
        names = (
            "all_states_finite",
            "all_policy_steps_valid",
            "all_samples_valid",
            "shape_matches_request",
        )
    elif component == "bptt":
        _require_exact_keys(
            correctness,
            {
                "passed",
                "update_accepted",
                "gradient_norm",
                "parameter_delta_norm",
                "finite_nonzero_gradient",
                "finite_nonzero_parameter_delta",
                "updated_parameters_finite",
            },
            "bptt.correctness",
        )
        names = (
            "update_accepted",
            "finite_nonzero_gradient",
            "finite_nonzero_parameter_delta",
            "updated_parameters_finite",
        )
        gradient = _finite_number(correctness["gradient_norm"], "correctness.gradient_norm")
        delta = _finite_number(
            correctness["parameter_delta_norm"], "correctness.parameter_delta_norm"
        )
        if correctness["finite_nonzero_gradient"] is not (gradient > 0):
            raise ValueError("finite_nonzero_gradient is inconsistent with gradient_norm")
        if correctness["finite_nonzero_parameter_delta"] is not (delta > 0):
            raise ValueError(
                "finite_nonzero_parameter_delta is inconsistent with parameter_delta_norm"
            )
    elif component == "version_a":
        _require_exact_keys(
            correctness,
            {
                "passed",
                "has_certificate",
                "proposal_interval_accepted",
                "degraded",
                "applied_continuous_postcheck_passed",
                "action_finite",
                "next_state_finite",
                "applied_interval_margin",
                "qp_kkt_max_residual",
            },
            "version_a.correctness",
        )
        for name in ("applied_interval_margin", "qp_kkt_max_residual"):
            _finite_number(correctness.get(name), f"correctness.{name}")
        for name in (
            "has_certificate",
            "proposal_interval_accepted",
            "degraded",
            "applied_continuous_postcheck_passed",
            "action_finite",
            "next_state_finite",
        ):
            if not isinstance(correctness[name], bool):
                raise ValueError(f"version_a.correctness.{name} must be boolean")
        names = (
            "has_certificate",
            "applied_continuous_postcheck_passed",
            "action_finite",
            "next_state_finite",
        )
        derived = (
            all(correctness.get(name) is True for name in names)
            and correctness.get("degraded") is False
        )
        if correctness["passed"] != derived:
            raise ValueError("Version-A correctness.passed is inconsistent with public postchecks")
        return derived
    elif component == "qp":
        _require_exact_keys(
            correctness,
            {
                "passed",
                "feasible",
                "input_valid",
                "primal_residual",
                "dual_residual",
                "stationarity_residual",
                "complementarity_residual",
            },
            "qp.correctness",
        )
        for name in (
            "primal_residual",
            "dual_residual",
            "stationarity_residual",
            "complementarity_residual",
        ):
            _finite_number(correctness.get(name), f"correctness.{name}")
        names = ("feasible", "input_valid")
    elif component == "validation":
        _require_exact_keys(
            correctness, {"passed", "report_passed", "report_integrity"}, "validation.correctness"
        )
        names = ("report_passed", "report_integrity")
    else:
        _require_exact_keys(
            correctness,
            {
                "passed",
                "has_certificate",
                "applied_accepted",
                "degraded",
                "action_finite",
                "next_state_finite",
                "postcheck_replay_error",
                "applied_exact_residual",
            },
            "version_b.correctness",
        )
        for name in ("postcheck_replay_error", "applied_exact_residual"):
            _finite_number(correctness.get(name), f"correctness.{name}")
        for name in (
            "has_certificate",
            "applied_accepted",
            "degraded",
            "action_finite",
            "next_state_finite",
        ):
            if not isinstance(correctness[name], bool):
                raise ValueError(f"version_b.correctness.{name} must be boolean")
        names = ("has_certificate", "applied_accepted", "action_finite", "next_state_finite")
        derived = (
            all(correctness.get(name) is True for name in names)
            and correctness.get("degraded") is False
        )
        if correctness["passed"] != derived:
            raise ValueError("Version-B correctness.passed is inconsistent with exact postchecks")
        return derived
    for name in names:
        if not isinstance(correctness[name], bool):
            raise ValueError(f"{component}.correctness.{name} must be boolean")
    derived = all(correctness[name] for name in names)
    if correctness["passed"] != derived:
        raise ValueError(f"{component} correctness.passed is inconsistent with component evidence")
    return derived


def _parse_settings(value: Any) -> BenchmarkSettings:
    if not isinstance(value, dict):
        raise ValueError("request.settings must be an object")
    _require_exact_keys(value, {field.name for field in fields(BenchmarkSettings)}, "settings")
    settings = BenchmarkSettings(**value)
    settings.validate()
    if _jsonable(asdict(settings)) != value:
        raise ValueError("request.settings contains noncanonical values")
    return settings


_IDENTITY_KEYS = {
    "measurement_id",
    "component_key",
    "component",
    "shape",
    "requested_shape",
    "effective_shape",
}
_NONEXECUTED_RECORD_KEYS = _IDENTITY_KEYS | {
    "status",
    "failure_stage",
    "reason",
    "estimated_live_bytes",
    "memory_guard_bytes",
    "compile_seconds",
    "warmup_seconds",
    "timing",
}
_JITTED_RECORD_KEYS = _IDENTITY_KEYS | {
    "status",
    "failure_stage",
    "reason",
    "execution_kind",
    "compile_seconds",
    "warmup_seconds",
    "timing",
    "work_units_per_execution",
    "estimated_live_bytes",
    "memory_guard_bytes",
    "compiled_memory",
    "process_max_rss_bytes_before",
    "process_max_rss_bytes_after",
    "device_memory_before",
    "device_memory_after",
    "memory_evidence",
    "correctness",
    "latency_scope",
}
_JITTED_COMPONENT_EXTRA_KEYS: dict[ComponentName, set[str]] = {
    "rollout": set(),
    "uncertain_rollout": set(),
    "bptt": {"includes"},
    "version_a": {"includes", "deadline_interpretation"},
    "qp": {"shape_scope"},
    "validation": set(),
    "version_b": {"includes", "deadline_interpretation"},
}
_VALIDATION_RECORD_KEYS = _IDENTITY_KEYS | {
    "status",
    "failure_stage",
    "reason",
    "execution_kind",
    "scope",
    "compile_seconds",
    "compile_not_applicable_reason",
    "warmup_seconds",
    "timing",
    "work_units_per_execution",
    "report_passed",
    "report_integrity",
    "report_digest",
    "failed_gates",
    "correctness",
    "latency_scope",
    "deadline_interpretation",
    "estimated_live_bytes",
    "memory_guard_bytes",
    "process_max_rss_bytes_after",
}


def _verify_measurement(
    record: Any, *, component: ComponentName, shape: ShapePoint, settings: BenchmarkSettings
) -> None:
    if not isinstance(record, dict):
        raise ValueError("measurement records must be objects")
    identity = _jsonable(_record_identity(component, shape))
    for name, expected in identity.items():
        if record.get(name) != expected:
            raise ValueError(f"measurement {identity['measurement_id']} has incorrect {name}")
    if record.get("estimated_live_bytes") != _estimate_component_bytes(component, shape):
        raise ValueError(f"{identity['measurement_id']} estimated memory is inconsistent")
    if record.get("memory_guard_bytes") != settings.max_estimated_bytes:
        raise ValueError(f"{identity['measurement_id']} memory guard does not match request")
    status = record.get("status")
    if status not in ("ok", "failed", "skipped"):
        raise ValueError(f"{identity['measurement_id']} has an invalid status")
    if record.get("timing") is None:
        expected_record_keys = _NONEXECUTED_RECORD_KEYS
    elif component == "validation":
        expected_record_keys = _VALIDATION_RECORD_KEYS
    else:
        expected_record_keys = _JITTED_RECORD_KEYS | _JITTED_COMPONENT_EXTRA_KEYS[component]
    _require_exact_keys(record, expected_record_keys, identity["measurement_id"])
    if status == "skipped":
        if (
            record.get("failure_stage") != "pre_execution_guard"
            or not isinstance(record.get("reason"), str)
            or record.get("timing") is not None
            or record.get("compile_seconds") is not None
            or record.get("warmup_seconds") != []
        ):
            raise ValueError(f"{identity['measurement_id']} has an invalid skipped record")
        return
    timing = record.get("timing")
    if timing is None:
        if status != "failed" or not isinstance(record.get("reason"), str):
            raise ValueError(f"{identity['measurement_id']} missing timing without failure")
        return
    work_units = _work_units(component, shape)
    if record.get("work_units_per_execution") != work_units:
        raise ValueError(f"{identity['measurement_id']} work unit count is inconsistent")
    _verify_timing(
        timing,
        settings=settings,
        name=f"{identity['measurement_id']}.timing",
        expected_samples=settings.repeats,
        work_units=work_units,
    )
    warmups = record.get("warmup_seconds")
    if not isinstance(warmups, list) or len(warmups) != settings.warmups:
        raise ValueError(f"{identity['measurement_id']} warmup count does not match request")
    for index, sample in enumerate(warmups):
        _finite_number(sample, f"warmup_seconds[{index}]", nonnegative=True)
    if component == "validation":
        if record.get("compile_seconds") is not None:
            raise ValueError("host validation compile_seconds must be null")
    else:
        _finite_number(record.get("compile_seconds"), "compile_seconds", nonnegative=True)
    correctness_passed = _verify_correctness(record, component)
    if status == "ok":
        if (
            not correctness_passed
            or record.get("failure_stage") is not None
            or record.get("reason") is not None
        ):
            raise ValueError(f"{identity['measurement_id']} cannot count failed correctness as ok")
    elif correctness_passed or record.get("failure_stage") != "untimed_correctness_postcheck":
        raise ValueError(f"{identity['measurement_id']} failed status is inconsistent")


def _verify_contention(
    value: Any, *, request: dict[str, Any], settings: BenchmarkSettings, shape: ShapePoint
) -> None:
    if not isinstance(value, dict):
        raise ValueError("contention_experiment must be an object")
    requested_workers = _resolve_contention(
        request["contention"], jax.devices(request["device"])[0]
    )
    if not requested_workers:
        _require_exact_keys(value, {"status", "reason", "requested_shape"}, "contention_experiment")
        if value.get("status") != "not_requested" or value.get("requested_shape") != _jsonable(
            asdict(shape)
        ):
            raise ValueError("not-requested contention record is inconsistent")
        return
    if value.get("requested_shape") != _jsonable(asdict(shape)):
        raise ValueError("contention requested shape is inconsistent")
    if value.get("status") in ("skipped", "failed") and "idle_controller_timing" not in value:
        _require_exact_keys(
            value,
            {"status", "reason", "requested_shape", "estimated_live_bytes", "memory_guard_bytes"},
            "contention_experiment",
        )
        return
    if value.get("status") == "failed" and value.get("idle_controller_timing") is None:
        _require_exact_keys(
            value,
            {
                "status",
                "failure_stage",
                "reason",
                "controller",
                "requested_shape",
                "effective_controller_shape",
                "controller_device",
                "controller_compile_seconds",
                "controller_warmup_seconds",
                "idle_controller_timing",
                "loaded_controller_timings",
            },
            "contention_experiment",
        )
        return
    if value.get("status") not in ("ok", "partial", "failed"):
        raise ValueError("contention status is invalid")
    _require_exact_keys(
        value,
        {
            "status",
            "failure_stage",
            "reason",
            "controller",
            "requested_shape",
            "effective_controller_shape",
            "effective_worker_shape",
            "controller_device",
            "controller_compile_seconds",
            "controller_warmup_seconds",
            "worker_startup_timeout_seconds",
            "idle_controller_timing",
            "loaded_controller_timings",
            "interpretation",
        },
        "contention_experiment",
    )
    _verify_timing(
        value.get("idle_controller_timing"),
        settings=settings,
        name="contention.idle_controller_timing",
        expected_samples=settings.repeats,
    )
    loaded = value.get("loaded_controller_timings")
    if not isinstance(loaded, list) or len(loaded) != len(requested_workers):
        raise ValueError("contention worker record count does not match request")
    successful = 0
    idle_median = value["idle_controller_timing"]["median_seconds"]
    idle_miss = value["idle_controller_timing"]["deadline_miss_fraction"]
    for index, worker in enumerate(loaded):
        if (
            not isinstance(worker, dict)
            or worker.get("worker_platform") != requested_workers[index]
        ):
            raise ValueError("contention worker order/platform does not match request")
        if worker.get("status") != "ok":
            if not isinstance(worker.get("reason"), str):
                raise ValueError("failed/unavailable contention worker requires a reason")
            allowed_failure_keys = {"worker_platform", "status", "reason"}
            allowed_failure_with_compile = allowed_failure_keys | {
                "worker_compile_seconds",
                "worker_warmup_seconds",
            }
            if set(worker) not in (allowed_failure_keys, allowed_failure_with_compile):
                raise ValueError("failed contention worker has unexpected fields")
            continue
        _require_exact_keys(
            worker,
            {
                "worker_platform",
                "status",
                "worker_compile_seconds",
                "worker_warmup_seconds",
                "worker_completed_updates",
                "worker_timing",
                "worker_raw_samples_truncated",
                "controller_timing",
                "median_slowdown_ratio",
                "deadline_miss_delta",
            },
            f"contention.worker[{index}]",
        )
        successful += 1
        _finite_number(
            worker.get("worker_compile_seconds"), "worker_compile_seconds", nonnegative=True
        )
        _finite_number(
            worker.get("worker_warmup_seconds"), "worker_warmup_seconds", nonnegative=True
        )
        _verify_timing(
            worker.get("controller_timing"),
            settings=settings,
            name=f"contention.worker[{index}].controller_timing",
            expected_samples=settings.repeats,
        )
        _verify_timing(
            worker.get("worker_timing"),
            settings=settings,
            name=f"contention.worker[{index}].worker_timing",
            expected_samples=None,
        )
        _require_close(
            worker.get("median_slowdown_ratio"),
            worker["controller_timing"]["median_seconds"] / idle_median,
            "median_slowdown_ratio",
        )
        _require_close(
            worker.get("deadline_miss_delta"),
            worker["controller_timing"]["deadline_miss_fraction"] - idle_miss,
            "deadline_miss_delta",
        )
    expected_status = "ok" if successful == len(requested_workers) else "partial"
    if successful == 0:
        expected_status = "failed"
    if value.get("status") != expected_status:
        raise ValueError("contention aggregate status is inconsistent with worker records")


def verify_performance_artifact(
    artifact: dict[str, Any] | Path,
    *,
    require_current_source: bool = False,
    require_current_runtime: bool = False,
    require_clean_source: bool = False,
) -> dict[str, Any]:
    """Strictly verify schema, raw-derived values, status accounting, and bindings."""
    if isinstance(artifact, Path):
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError("performance artifact must be a regular non-symlink file")

        def reject_constant(value: str) -> None:
            raise ValueError(f"nonfinite JSON constant is forbidden: {value}")

        document = json.loads(artifact.read_text(encoding="utf-8"), parse_constant=reject_constant)
    else:
        document = artifact
    if not isinstance(document, dict):
        raise ValueError("performance artifact must be an object")
    _require_exact_keys(
        document,
        {
            "schema",
            "schema_version",
            "provenance",
            "request",
            "preset_shapes",
            "measurements",
            "contention_experiment",
            "completion",
            "claim_caveats",
            "integrity",
        },
        "artifact",
    )
    if (
        document["schema"] != PERFORMANCE_ARTIFACT_SCHEMA
        or document["schema_version"] != PERFORMANCE_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported performance artifact schema")
    integrity = document["integrity"]
    if not isinstance(integrity, dict):
        raise ValueError("integrity must be an object")
    _require_exact_keys(integrity, {"algorithm", "digest", "authenticity_limit"}, "integrity")
    if integrity["algorithm"] != "sha256-canonical-json":
        raise ValueError("unsupported integrity algorithm")
    if (
        not isinstance(integrity["digest"], str)
        or len(integrity["digest"]) != 64
        or any(character not in "0123456789abcdef" for character in integrity["digest"])
    ):
        raise ValueError("integrity.digest must be a lowercase SHA-256 digest")
    unsigned = {key: value for key, value in document.items() if key != "integrity"}
    if integrity["digest"] != _canonical_digest(unsigned):
        raise ValueError("performance artifact digest mismatch")
    request = document["request"]
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    _require_exact_keys(
        request, {"device", "preset", "components", "settings", "contention"}, "request"
    )
    if request["device"] not in ("cpu", "gpu") or request["preset"] not in PRESETS:
        raise ValueError("request device/preset is invalid")
    components = request["components"]
    if (
        not isinstance(components, list)
        or not components
        or len(components) != len(set(components))
        or any(component not in ALL_COMPONENTS for component in components)
    ):
        raise ValueError("request components must be a unique nonempty known list")
    _resolve_contention(request["contention"], jax.devices(request["device"])[0])
    settings = _parse_settings(request["settings"])
    shapes = PRESETS[request["preset"]]
    expected_shapes = _jsonable([asdict(shape) for shape in shapes])
    if document["preset_shapes"] != expected_shapes:
        raise ValueError("preset_shapes does not match the named immutable preset")
    records = document["measurements"]
    if not isinstance(records, list) or len(records) != len(shapes) * len(components):
        raise ValueError("measurement matrix is incomplete")
    index = 0
    status_counts = {status: 0 for status in ("ok", "failed", "skipped")}
    for shape in shapes:
        for component_name in components:
            component = component_name
            _verify_measurement(records[index], component=component, shape=shape, settings=settings)
            status_counts[records[index]["status"]] += 1
            index += 1
    contention_shape = CONTENTION_SHAPES[request["preset"]]
    _verify_contention(
        document["contention_experiment"],
        request=request,
        settings=settings,
        shape=contention_shape,
    )
    provenance = document["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object")
    _require_exact_keys(
        provenance,
        {"timestamp_utc", "command", "host", "software", "source", "device", "visible_devices"},
        "provenance",
    )
    try:
        created = datetime.fromisoformat(provenance["timestamp_utc"])
    except (TypeError, ValueError) as error:
        raise ValueError("provenance.timestamp_utc must be ISO-8601") from error
    if created.tzinfo is None:
        raise ValueError("provenance.timestamp_utc must be timezone-aware")
    if not isinstance(provenance["command"], list) or not all(
        isinstance(argument, str) for argument in provenance["command"]
    ):
        raise ValueError("provenance.command must be a string array")
    host = provenance["host"]
    if not isinstance(host, dict):
        raise ValueError("provenance.host must be an object")
    _require_exact_keys(
        host,
        {"hostname", "platform", "python", "logical_cpu_count", "cpu_model"},
        "provenance.host",
    )
    if not all(
        isinstance(host[name], str) for name in ("hostname", "platform", "python", "cpu_model")
    ):
        raise ValueError("provenance.host string fields have invalid types")
    if host["logical_cpu_count"] is not None and (
        isinstance(host["logical_cpu_count"], bool)
        or not isinstance(host["logical_cpu_count"], int)
        or host["logical_cpu_count"] <= 0
    ):
        raise ValueError("logical_cpu_count must be null or a positive integer")
    source = provenance["source"]
    if not isinstance(source, dict):
        raise ValueError("provenance.source must be an object")
    _require_exact_keys(
        source,
        {"scope", "tree_sha256", "file_count", "git_commit", "git_branch", "git_dirty"},
        "provenance.source",
    )
    if not isinstance(source["git_dirty"], bool):
        raise ValueError("source.git_dirty must be boolean")
    if (
        not isinstance(source["tree_sha256"], str)
        or len(source["tree_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in source["tree_sha256"])
    ):
        raise ValueError("source.tree_sha256 must be a lowercase SHA-256 digest")
    if (
        isinstance(source["file_count"], bool)
        or not isinstance(source["file_count"], int)
        or source["file_count"] <= 0
    ):
        raise ValueError("source.file_count must be a positive integer")
    if not all(isinstance(source[name], str) for name in ("scope", "git_commit", "git_branch")):
        raise ValueError("source textual bindings have invalid types")
    if require_clean_source and source["git_dirty"]:
        raise ValueError("claim-grade verification requires a clean source tree")
    device = provenance["device"]
    if not isinstance(device, dict) or device.get("platform") != request["device"]:
        raise ValueError("actual device platform does not match requested backend")
    required_device_keys = {"id", "platform", "device_kind"}
    allowed_device_keys = required_device_keys | {
        "compute_capability",
        "local_hardware_id",
        "process_index",
        "slice_index",
    }
    if not required_device_keys <= set(device) or not set(device) <= allowed_device_keys:
        raise ValueError("provenance.device keys are invalid")
    if (
        isinstance(device["id"], bool)
        or not isinstance(device["id"], int)
        or not isinstance(device["platform"], str)
        or not isinstance(device["device_kind"], str)
    ):
        raise ValueError("provenance.device base fields have invalid types")
    software = provenance["software"]
    if not isinstance(software, dict):
        raise ValueError("provenance.software must be an object")
    _require_exact_keys(software, {"crazyflow", "jax", "jaxlib", "numpy"}, "software")
    if not all(isinstance(version, str) for version in software.values()):
        raise ValueError("software versions must be strings")
    visible_devices = provenance["visible_devices"]
    if not isinstance(visible_devices, list) or not visible_devices:
        raise ValueError("visible_devices must be a nonempty array")
    for visible in visible_devices:
        if not isinstance(visible, dict):
            raise ValueError("visible device records must be objects")
        if not required_device_keys <= set(visible) or not set(visible) <= allowed_device_keys:
            raise ValueError("visible device record keys are invalid")
    completion = document["completion"]
    expected_completion = {
        "requested_measurements": len(records),
        "recorded_measurements": len(records),
        "status_counts": status_counts,
        "all_requested_measurements_ok": status_counts
        == {"ok": len(records), "failed": 0, "skipped": 0},
        "contention_status": document["contention_experiment"]["status"],
        "source_clean": not source["git_dirty"],
        "claim_grade_source_eligible": not source["git_dirty"],
    }
    if completion != expected_completion:
        raise ValueError("completion summary is inconsistent with measurement records")
    if not isinstance(document["claim_caveats"], list) or not all(
        isinstance(item, str) and item for item in document["claim_caveats"]
    ):
        raise ValueError("claim_caveats must be a nonempty string array")
    if require_current_source and _source_tree_binding() != source:
        raise ValueError("artifact source/git binding does not match the current checkout")
    if require_current_runtime:
        current_device = _device_attributes(jax.devices(request["device"])[0])
        if current_device != device:
            raise ValueError("artifact device binding does not match the current runtime")
        expected_versions = {
            "crazyflow": crazyflow_version,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
        }
        if software != expected_versions:
            raise ValueError("artifact software binding does not match the current runtime")
    return {
        "valid": True,
        "schema": PERFORMANCE_ARTIFACT_SCHEMA,
        "source_clean": not source["git_dirty"],
        "all_requested_measurements_ok": completion["all_requested_measurements_ok"],
        "current_source_checked": require_current_source,
        "current_runtime_checked": require_current_runtime,
        "integrity_is_authentication": False,
    }


def _write_once_atomic(payload: str, path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"existing performance artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if (
                path.is_symlink()
                or not path.is_file()
                or path.read_text(encoding="utf-8") != payload
            ):
                raise FileExistsError(f"existing performance artifact differs: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(document: dict[str, Any], output: str | None) -> str:
    verify_performance_artifact(document)
    encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is not None:
        path = Path(output).expanduser().resolve()
        _write_once_atomic(encoded, path)
    return encoded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument(
        "--components",
        default="all",
        help=f"comma-separated subset of {','.join(ALL_COMPONENTS)}, or all",
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--deadline-ms", type=float, default=20.0)
    parser.add_argument("--max-estimated-gib", type=float, default=12.0)
    parser.add_argument(
        "--contention",
        default="auto",
        help="auto, none, cpu, gpu, or cpu,gpu; worker executes real DA quad BPTT updates",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--verify-artifact",
        type=Path,
        help="Strictly verify a saved artifact against the current source/runtime, then exit.",
    )
    parser.add_argument(
        "--require-clean-source",
        action="store_true",
        help="Reject execution/verification when the source tree has uncommitted changes.",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    """CLI entry point emitting one canonical JSON document."""
    raw_arguments = sys.argv[1:] if arguments is None else arguments
    parser = _parser()
    args = parser.parse_args(raw_arguments)
    if args.verify_artifact is not None:
        try:
            report = verify_performance_artifact(
                args.verify_artifact.expanduser().resolve(),
                require_current_source=True,
                require_current_runtime=True,
                require_clean_source=args.require_clean_source,
            )
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        print(json.dumps(report, sort_keys=True))
        return 0
    if not math.isfinite(args.max_estimated_gib) or args.max_estimated_gib <= 0:
        parser.error("--max-estimated-gib must be finite and positive")
    if args.require_clean_source and _source_tree_binding()["git_dirty"]:
        parser.error("--require-clean-source was requested but the source tree is dirty")
    settings = BenchmarkSettings(
        repeats=args.repeats,
        warmups=args.warmups,
        deadline_seconds=args.deadline_ms / 1000.0,
        max_estimated_bytes=round(args.max_estimated_gib * 1024**3),
    )
    try:
        components = _parse_components(args.components)
        document = run_benchmark(
            device_name=args.device,
            preset_name=args.preset,
            components=components,
            settings=settings,
            contention=args.contention,
            command_arguments=list(raw_arguments),
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(_write_json(document, args.output), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
