"""Campaign-faithful synchronized benchmark for one online DA-PLCBF BPTT burst.

The default request is derived directly from :meth:`ExperimentConfig.final_defaults`: 64 fallback
policies, 64 training scenarios, a 50-step differentiable 13-state rollout, eight obstacle slots,
and ten fused AdamW updates.  The estimated dynamics model is supplied as a runtime argument to
the dynamic-model BPTT executable, matching the online adaptation path.

Examples::

    pixi run -e gpu-tests python benchmark/da_plcbf_gpu_bptt.py \
        --device gpu --repeats 20 --warmups 3 \
        --output artifacts/da_plcbf/gpu-bptt-full-shape.json

    # Explicitly non-final, tiny CPU probe for development only.
    pixi run -e tests python benchmark/da_plcbf_gpu_bptt.py \
        --device cpu --policies 9 --batch 1 --horizon 1 --obstacles 2 \
        --burst-steps 2 --repeats 1 --warmups 0

The benchmark never substitutes a smaller shape or a CPU device after a failed request.  Both the
campaign reference, requested shape, and array-derived effective shape are serialized so results
from an override cannot be presented as the final configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from crazyflow import __version__ as crazyflow_version
from crazyflow.safety.da_plcbf.experiments import ExperimentConfig, build_experiment_resources
from crazyflow.safety.da_plcbf.library import descriptor_targets_from_spec
from crazyflow.safety.da_plcbf.quad_actor_bptt import build_dynamic_model_quad_actor_bptt_functions
from crazyflow.safety.da_plcbf.quad_actor_losses import (
    QuadLearningConfig,
    rigid_body_safety_batch_from_circles,
)
from crazyflow.safety.da_plcbf.scenarios import ScenarioTapeConfig
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch

SCHEMA = "crazyflow.da_plcbf.campaign_gpu_bptt.v1"


@dataclass(frozen=True, slots=True)
class BPTTBenchmarkShape:
    """Every static axis consumed by one fused candidate-update burst."""

    policies: int
    batch: int
    horizon: int
    obstacles: int
    burst_steps: int

    def validate(self) -> None:
        """Reject unsupported shapes rather than silently changing them."""
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        # The campaign library has eight immutable structural policies and needs at least one
        # adaptive slot for BPTT to have candidate parameters it is allowed to change.
        if self.policies < 9:
            raise ValueError("policies must be at least 9 (8 structural + >=1 adaptive)")

    @classmethod
    def campaign_final(cls) -> BPTTBenchmarkShape:
        """Derive the authoritative reference shape from the campaign config."""
        config = ExperimentConfig.final_defaults()
        return cls(
            policies=config.policy_count,
            batch=config.training_scenario_count,
            horizon=config.certificate_horizon,
            obstacles=config.static_capacity + config.dynamic_capacity,
            burst_steps=config.bptt_burst_steps,
        )


class BPTTProblem(NamedTuple):
    """Device-resident runtime arguments and static BPTT dependencies."""

    functions: Any
    arguments: tuple[Any, ...]
    learning_config: QuadLearningConfig


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


def _source_digest() -> str:
    """Bind the result to this harness and the code implementing the BPTT graph."""
    root = _repository_root()
    paths = (
        root / "benchmark" / "da_plcbf_gpu_bptt.py",
        root / "crazyflow" / "safety" / "da_plcbf" / "experiments.py",
        root / "crazyflow" / "safety" / "da_plcbf" / "quad_actor_bptt.py",
        root / "crazyflow" / "safety" / "da_plcbf" / "quad_actor_losses.py",
        root / "crazyflow" / "safety" / "da_plcbf" / "quad_rollouts.py",
        root / "crazyflow" / "safety" / "da_plcbf" / "direct_wrench.py",
        root / "pyproject.toml",
        root / "pixi.lock",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _device_metadata(device: jax.Device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "id": int(device.id),
        "platform": str(device.platform),
        "device_kind": str(device.device_kind),
    }
    for name in ("compute_capability", "local_hardware_id", "process_index", "slice_index"):
        value = getattr(device, name, None)
        if isinstance(value, (str, int, float, bool, tuple)):
            metadata[name] = value
    return metadata


def _select_device(requested: str) -> jax.Device:
    """Select exactly the requested backend; no GPU-to-CPU fallback is permitted."""
    if requested not in {"cpu", "gpu"}:
        raise ValueError("device must be exactly 'cpu' or 'gpu'")
    try:
        devices = jax.devices(requested)
    except RuntimeError as error:
        raise RuntimeError(f"requested JAX {requested} backend is unavailable") from error
    if not devices:
        raise RuntimeError(f"requested JAX {requested} backend has no devices")
    return devices[0]


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


def _compiled_memory_analysis(compiled: Any) -> dict[str, int] | None:
    try:
        analysis = compiled.memory_analysis()
    except (AttributeError, RuntimeError):
        return None
    if analysis is None:
        return None
    names = (
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
        for name in names
        if getattr(analysis, name, None) is not None
    }


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _device_put(tree: Any, device: jax.Device) -> Any:
    def place(value: Any) -> Any:
        if isinstance(value, (jax.Array, np.ndarray, np.generic)):
            return jax.device_put(value, device)
        return value

    return jax.tree.map(place, tree)


def _scenario_batch(shape: BPTTBenchmarkShape) -> tuple[jax.Array, CircleScenarioBatch]:
    """Construct deterministic fixed-size, nondegenerate campaign-style training rows."""
    dtype = jnp.float32
    batch_phase = jnp.linspace(-0.16, 0.16, shape.batch, dtype=dtype)
    initial = jnp.zeros((shape.batch, 13), dtype=dtype)
    initial = initial.at[:, 0].set(-0.75)
    initial = initial.at[:, 1].set(batch_phase)
    initial = initial.at[:, 2].set(1.0)
    initial = initial.at[:, 6].set(1.0)
    initial = initial.at[:, 7].set(0.30)

    obstacle_index = jnp.arange(shape.obstacles, dtype=dtype)
    obstacle_angle = 2.0 * jnp.pi * obstacle_index / shape.obstacles
    centers = jnp.stack(
        (
            0.35 + 0.55 * jnp.cos(obstacle_angle),
            0.70 * jnp.sin(obstacle_angle),
            1.05 + 0.18 * jnp.sin(2.0 * obstacle_angle),
        ),
        axis=-1,
    )
    centers = jnp.broadcast_to(centers[None, :, :], (shape.batch, shape.obstacles, 3))
    centers = centers.at[:, :, 1].add(0.15 * batch_phase[:, None])
    scenarios = CircleScenarioBatch(
        obstacle_centers=centers,
        obstacle_radii=jnp.full((shape.batch, shape.obstacles), 0.12, dtype=dtype),
        obstacle_mask=jnp.ones((shape.batch, shape.obstacles), dtype=bool),
        arena_lower=jnp.broadcast_to(jnp.asarray([-5.0, -5.0, 0.1], dtype=dtype), (shape.batch, 3)),
        arena_upper=jnp.broadcast_to(jnp.asarray([5.0, 5.0, 4.0], dtype=dtype), (shape.batch, 3)),
        speed_limit=jnp.full((shape.batch,), 3.0, dtype=dtype),
    )
    return initial, scenarios


def build_problem(shape: BPTTBenchmarkShape, device: jax.Device) -> BPTTProblem:
    """Build the production dynamic-model BPTT call at one exact static shape."""
    shape.validate()
    final = ExperimentConfig.final_defaults(random_seed=0)
    config = replace(
        final,
        policy_count=shape.policies,
        training_scenario_count=shape.batch,
        certificate_horizon=shape.horizon,
        bptt_burst_steps=shape.burst_steps,
    )
    config.validate()
    resources = build_experiment_resources(
        config, obstacle_count=shape.obstacles, initialization_seed=23
    )
    vehicle_radius = ScenarioTapeConfig().vehicle_radius
    resources = replace(
        resources,
        barrier_config=replace(
            resources.barrier_config,
            obstacle_clearance=vehicle_radius + config.obstacle_clearance,
            arena_clearance=vehicle_radius,
        ),
    )
    learning = QuadLearningConfig(
        dt=config.dt, horizon=config.certificate_horizon, policy_gain=config.policy_gain
    )
    initial_states, scenarios = _scenario_batch(shape)
    safety = rigid_body_safety_batch_from_circles(
        scenarios,
        angular_rate_max=config.angular_rate_max,
        tilt_max_radians=config.tilt_max_radians,
    )

    # This deliberately differs from the static nominal model and enters ``burst`` as its final
    # runtime argument.  The executable can therefore be reused after estimator/wind changes.
    runtime_model = resources.model._replace(
        mass=resources.model.mass * jnp.asarray(1.05, dtype=resources.model.mass.dtype),
        wind_velocity=jnp.asarray([0.35, -0.20, 0.05], dtype=jnp.float32),
    )
    targets = descriptor_targets_from_spec(resources.spec)
    descriptor_scales = jnp.asarray(
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=jnp.float32
    )
    resources = _device_put(resources, device)
    arguments_without_state = _device_put(
        (
            initial_states,
            scenarios,
            safety,
            targets,
            resources.initial_params,
            descriptor_scales,
            runtime_model,
        ),
        device,
    )
    with jax.default_device(device):
        functions = build_dynamic_model_quad_actor_bptt_functions(
            resources.spec,
            resources.actuator,
            resources.actor_config,
            resources.quad_config,
            resources.barrier_config,
            learning,
            resources.loss_config,
            burst_steps=shape.burst_steps,
            device=device,
        )
        state = _device_put(functions.initialize(resources.initial_params), device)
    return BPTTProblem(functions, (state, *arguments_without_state), learning)


def _timing_summary(samples: list[float], deadline_seconds: float) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("timing samples must be nonempty, finite, and nonnegative")
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


def _tree_l2_delta(new: Any, old: Any) -> float:
    squared = [
        np.sum((np.asarray(new_leaf) - np.asarray(old_leaf)) ** 2, dtype=np.float64)
        for new_leaf, old_leaf in zip(jax.tree.leaves(new), jax.tree.leaves(old), strict=True)
    ]
    return float(np.sqrt(np.sum(squared, dtype=np.float64)))


def _effective_shape(problem: BPTTProblem, output: Any) -> BPTTBenchmarkShape:
    state, metrics = output
    _, initial_states, scenarios, _, targets, _, _, _ = problem.arguments
    return BPTTBenchmarkShape(
        policies=int(targets.shape[0]),
        batch=int(initial_states.shape[0]),
        horizon=int(problem.learning_config.horizon),
        obstacles=int(scenarios.obstacle_centers.shape[1]),
        burst_steps=int(metrics.update_accepted.shape[0]),
    )


def _correctness(problem: BPTTProblem, output: Any, shape: BPTTBenchmarkShape) -> dict[str, Any]:
    initial_state = problem.arguments[0]
    updated, metrics = output
    accepted = np.asarray(metrics.update_accepted)
    gradients = np.asarray(metrics.gradient_norm)
    deltas = np.asarray(metrics.parameter_delta_norm)
    initial_steps = int(np.asarray(initial_state.steps))
    final_steps = int(np.asarray(updated.steps))
    effective = _effective_shape(problem, output)
    leaves_finite = all(
        bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in jax.tree.leaves(updated)
    ) and all(bool(np.all(np.isfinite(np.asarray(leaf)))) for leaf in jax.tree.leaves(metrics))
    parameter_l2_delta = _tree_l2_delta(updated.params, initial_state.params)
    checks = {
        "all_updates_accepted": bool(accepted.shape == (shape.burst_steps,) and np.all(accepted)),
        "all_gradient_norms_finite": bool(np.all(np.isfinite(gradients))),
        "any_gradient_norm_nonzero": bool(np.any(gradients > 0)),
        "all_parameter_delta_norms_finite": bool(np.all(np.isfinite(deltas))),
        "any_parameter_delta_norm_nonzero": bool(np.any(deltas > 0)),
        "all_output_leaves_finite": leaves_finite,
        "optimizer_steps_advanced_by_burst": final_steps - initial_steps == shape.burst_steps,
        "candidate_parameters_changed": math.isfinite(parameter_l2_delta)
        and parameter_l2_delta > 0,
        "effective_shape_matches_request": effective == shape,
    }
    return {
        "passed": all(checks.values()),
        **checks,
        "initial_optimizer_steps": initial_steps,
        "final_optimizer_steps": final_steps,
        "optimizer_step_delta": final_steps - initial_steps,
        "gradient_norms": gradients.tolist(),
        "parameter_delta_norms": deltas.tolist(),
        "candidate_parameter_l2_delta": parameter_l2_delta,
    }


def run_benchmark(
    *,
    device_name: str = "gpu",
    shape: BPTTBenchmarkShape | None = None,
    repeats: int = 20,
    warmups: int = 3,
    deadline_seconds: float = 0.05,
) -> dict[str, Any]:
    """Compile, warm, and synchronously time the exact requested fused burst."""
    reference = BPTTBenchmarkShape.campaign_final()
    requested = reference if shape is None else shape
    requested.validate()
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("warmups must be a nonnegative integer")
    if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be finite and positive")

    device = _select_device(device_name)
    memory_before = _memory_stats(device)
    rss_before = _rss_bytes()
    construction_start = time.perf_counter()
    problem = build_problem(requested, device)
    construction_seconds = time.perf_counter() - construction_start

    compile_start = time.perf_counter()
    compiled = problem.functions.burst.lower(*problem.arguments).compile()
    compile_seconds = time.perf_counter() - compile_start
    compiler_memory = _compiled_memory_analysis(compiled)

    warmup_samples: list[float] = []
    for _ in range(warmups):
        start = time.perf_counter()
        warm_output = compiled(*problem.arguments)
        jax.block_until_ready(warm_output)
        warmup_samples.append(time.perf_counter() - start)

    timed_samples: list[float] = []
    output: Any = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = compiled(*problem.arguments)
        jax.block_until_ready(output)
        timed_samples.append(time.perf_counter() - start)
    assert output is not None

    effective = _effective_shape(problem, output)
    correctness = _correctness(problem, output, requested)
    timing = _timing_summary(timed_samples, deadline_seconds)
    timing["median_bursts_per_second"] = (
        float(1.0 / timing["median_seconds"]) if timing["median_seconds"] > 0 else None
    )
    timing["median_optimizer_steps_per_second"] = (
        float(requested.burst_steps / timing["median_seconds"])
        if timing["median_seconds"] > 0
        else None
    )
    timing["meets_deadline_for_every_sample"] = timing["deadline_misses"] == 0

    overrides = {
        name: {"campaign_final": getattr(reference, name), "requested": value}
        for name, value in asdict(requested).items()
        if value != getattr(reference, name)
    }
    runtime_model = problem.arguments[-1]
    result = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "ok" if correctness["passed"] else "failed_correctness",
        "measurement_scope": (
            f"one fused dynamic-model BPTT burst: {requested.burst_steps} sequential full "
            f"13-state differentiable rollouts of horizon {requested.horizon}, reverse-mode "
            "gradients, and AdamW candidate updates at the explicitly requested shape"
        ),
        "campaign_reference_shape": asdict(reference),
        "requested_shape": asdict(requested),
        "effective_shape": asdict(effective),
        "uses_exact_campaign_final_shape": requested == reference and effective == reference,
        "shape_overrides_from_campaign_final": overrides,
        "no_device_or_shape_fallback": True,
        "request": {
            "device": device_name,
            "repeats": repeats,
            "warmups": warmups,
            "deadline_seconds": deadline_seconds,
        },
        "device": _device_metadata(device),
        "runtime_model_argument": {
            "supplied_to_compiled_burst": True,
            "argument_index": len(problem.arguments) - 1,
            "mass_kg": float(np.asarray(runtime_model.mass)),
            "wind_velocity_m_per_s": np.asarray(runtime_model.wind_velocity).tolist(),
        },
        "input_fixture": {
            "resource_builder": "experiments.build_experiment_resources",
            "vehicle_footprint_clearance_matches_scenario_tape_default": True,
            "training_rows": "deterministic fixed-shape timing fixture",
            "evaluation_tape_rows_used": False,
            "reason": (
                "timing isolates the campaign graph and static axes without consuming scientific "
                "evaluation data; numerical rows do not change the compiled graph shape"
            ),
        },
        "phases": {
            "problem_construction_seconds": construction_seconds,
            "lowering_and_compilation_seconds": compile_seconds,
            "warmup_synchronized_seconds": warmup_samples,
            "timed_synchronized": timing,
        },
        "memory": {
            "compiler_memory_analysis": compiler_memory,
            "device_allocator_before": memory_before,
            "device_allocator_after": _memory_stats(device),
            "process_max_rss_bytes_before": rss_before,
            "process_max_rss_bytes_after": _rss_bytes(),
        },
        "correctness": correctness,
        "provenance": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "crazyflow": crazyflow_version,
            "git": _git_provenance(),
            "source_sha256": _source_digest(),
        },
        "interpretation": {
            "deadline_hz": 1.0 / deadline_seconds,
            "timing_excludes_compilation": True,
            "timing_excludes_warmup": True,
            "every_timed_sample_is_device_synchronized": True,
            "this_is_candidate_training_only": True,
            "hard_admission_validation_and_snapshot_publication_are_excluded": True,
        },
    }
    # Prove the result is strict JSON before returning it to callers or writing it to disk.
    return json.loads(json.dumps(result, allow_nan=False, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    reference = BPTTBenchmarkShape.campaign_final()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--policies", type=int, default=reference.policies)
    parser.add_argument("--batch", type=int, default=reference.batch)
    parser.add_argument("--horizon", type=int, default=reference.horizon)
    parser.add_argument("--obstacles", type=int, default=reference.obstacles)
    parser.add_argument("--burst-steps", type=int, default=reference.burst_steps)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--deadline-ms",
        type=float,
        default=50.0,
        help="Complete-burst latency target; 50 ms corresponds to a 20 Hz update rate.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _write_once(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"refusing to replace different benchmark artifact: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    shape = BPTTBenchmarkShape(
        policies=args.policies,
        batch=args.batch,
        horizon=args.horizon,
        obstacles=args.obstacles,
        burst_steps=args.burst_steps,
    )
    result = run_benchmark(
        device_name=args.device,
        shape=shape,
        repeats=args.repeats,
        warmups=args.warmups,
        deadline_seconds=args.deadline_ms / 1000.0,
    )
    payload = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        _write_once(args.output, payload)
    sys.stdout.write(payload)
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
