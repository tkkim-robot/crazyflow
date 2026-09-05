"""Profile a reference-learner continuation with explicit rotating-anchor microbatches.

Only the number of rotating retention anchors changes. Teacher, anchor bank, actor, model,
parameters, previous parameters and Adam history remain intact. Synchronized learner service
is measured separately from controller deadlines; this program is not a paced flight run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.learner_checkpoint import (
    LearnerCheckpoint,
    load_learner_checkpoint,
    save_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.persistent_skill_learner import rollout_skill_library
from crazyflow.safety.da_plcbf.state_conditioned_learning import (
    ReferenceContract,
    build_reference_skill_learner,
    build_reference_skill_learner_from_checkpoint,
    load_reference_contract,
    reference_contract_checkpoint_metadata,
    save_reference_contract,
)


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _tree_digest(value: Any) -> str:
    digest = hashlib.sha256(str(jax.tree.structure(value)).encode())
    for leaf in jax.tree.leaves(value):
        array = np.asarray(leaf)
        digest.update(json.dumps([array.shape, array.dtype.str]).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _statistics(seconds: list[float]) -> dict[str, Any]:
    values = np.asarray(seconds)
    return {
        "sample_count": len(values),
        "minimum_seconds": float(values.min()),
        "median_seconds": float(np.median(values)),
        "mean_seconds": float(values.mean()),
        "p95_seconds": float(np.percentile(values, 95)),
        "maximum_seconds": float(values.max()),
    }


def _finite_scalar(value: Any) -> float | None:
    result = float(np.asarray(value))
    return result if math.isfinite(result) else None


def prepare_runtime_variant(
    source: LearnerCheckpoint, contract: ReferenceContract, directory: Path, anchor_batch_size: int
) -> tuple[LearnerCheckpoint, ReferenceContract]:
    """Save an exact persistent continuation with only anchor microbatch size changed."""
    if source.metadata.get("reference_contract_binding") != "verified_npz_and_manifest_sha256":
        raise ValueError("runtime comparison requires a checkpoint bound to its exact teacher")
    settings = replace(contract.learning_config, anchor_batch_size=anchor_batch_size)
    settings.validate()
    if anchor_batch_size > len(contract.anchors):
        raise ValueError("anchor microbatch must not exceed the immutable bank")
    directory.mkdir(parents=True, exist_ok=False)
    revised = replace(contract, learning_config=settings)
    save_reference_contract(revised, directory / "nominal_reference")
    binding = reference_contract_checkpoint_metadata(directory / "nominal_reference")
    save_learner_checkpoint(
        source.state,
        source.spec,
        source.config,
        source.actuator,
        source.physical_state,
        directory / "initial_checkpoint",
        metadata={
            **source.metadata,
            **binding,
            "runtime_revision": {
                "only_changed_configuration": "learning_config.anchor_batch_size",
                "source_anchor_batch_size": contract.learning_config.anchor_batch_size,
                "anchor_batch_size": anchor_batch_size,
                "source_checkpoint_npz_sha256": source.sha256,
                "source_checkpoint_path": str(source.npz_path),
                "interpretation": (
                    "same rotating retention bank, fewer anchors per update"
                    if anchor_batch_size
                    else "current-state-only diagnostic ablation; no anchor retention loss"
                ),
            },
        },
    )
    device = source.physical_state.device
    loaded = load_learner_checkpoint(directory / "initial_checkpoint", device=device)
    restored_contract = load_reference_contract(directory / "nominal_reference", device=device)
    source_identity = _tree_digest(
        (source.state, source.spec, source.actuator, source.physical_state)
    )
    restored_identity = _tree_digest(
        (loaded.state, loaded.spec, loaded.actuator, loaded.physical_state)
    )
    teacher_identity = _tree_digest(
        (contract.params, contract.model, contract.anchors, contract.spec, contract.actuator)
    )
    restored_teacher_identity = _tree_digest(
        (
            restored_contract.params,
            restored_contract.model,
            restored_contract.anchors,
            restored_contract.spec,
            restored_contract.actuator,
        )
    )
    if source_identity != restored_identity or teacher_identity != restored_teacher_identity:
        raise AssertionError("runtime revision changed a persistent state or nominal teacher array")
    if source.config != loaded.config or contract.actor_config != restored_contract.actor_config:
        raise AssertionError("runtime revision changed actor configuration")
    _write(
        directory / "continuation_identity.json",
        {
            "only_configuration_change": {
                "learning_config.anchor_batch_size": [
                    contract.learning_config.anchor_batch_size,
                    anchor_batch_size,
                ]
            },
            "source_and_variant_persistent_state_equal": True,
            "source_and_variant_teacher_model_bank_equal": True,
            "persistent_continuation_sha256": source_identity,
            "teacher_model_bank_sha256": teacher_identity,
            "source_reference_learning_config": asdict(contract.learning_config),
            "variant_reference_learning_config": asdict(restored_contract.learning_config),
            **binding,
        },
    )
    return loaded, restored_contract


def _time_callable(function: Callable[[], Any], warmup: int, repetitions: int) -> dict[str, Any]:
    for _ in range(warmup):
        jax.block_until_ready(function())
    seconds = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        jax.block_until_ready(function())
        seconds.append((time.perf_counter_ns() - start) * 1e-9)
    return {**_statistics(seconds), "samples_seconds": seconds}


def profile_runtime_variant(
    bundle: LearnerCheckpoint,
    contract: ReferenceContract,
    directory: Path,
    *,
    device: jax.Device,
    warmup: int = 5,
    repetitions: int = 50,
    contention_note: str = "Device isolation not established; other workloads may be active.",
    profile_components: bool = True,
) -> dict[str, Any]:
    """Measure completed synchronized updates and save every finite full-state snapshot."""
    if warmup < 1 or repetitions < 1:
        raise ValueError("warmup and measured repetitions must be positive")
    learner = build_reference_skill_learner(contract, bundle.config, device=device)
    source_state, physical, model = jax.device_put(
        (bundle.state, bundle.physical_state, bundle.point_model), device
    )
    jax.block_until_ready((source_state, physical, model))
    warmup_state = source_state
    for _ in range(warmup):
        warmup_state, _ = jax.block_until_ready(learner.step(warmup_state, physical, model))
    # The warmup continuation is deliberately discarded: measured update one resumes the
    # saved state exactly, including its original anchor-bank phase and optimizer history.
    persistent = source_state
    snapshots = directory / "completed_updates"
    snapshots.mkdir(exist_ok=False)
    save_reference_contract(contract, snapshots / "nominal_reference")
    binding = reference_contract_checkpoint_metadata(snapshots / "nominal_reference")
    ledger = []
    times = []
    count = contract.learning_config.anchor_batch_size
    initial_version = int(np.asarray(source_state.library_version))
    initial_steps = int(np.asarray(source_state.cumulative_gradient_steps))
    bank_count = len(contract.anchors)
    for index in range(repetitions):
        before_version = int(np.asarray(persistent.library_version))
        before_steps = int(np.asarray(persistent.cumulative_gradient_steps))
        before_identity = _tree_digest(persistent)
        jax.block_until_ready((persistent, physical, model))
        started = time.perf_counter_ns()
        following, metrics = jax.block_until_ready(learner.step(persistent, physical, model))
        completed = time.perf_counter_ns()
        service = (completed - started) * 1e-9
        times.append(service)
        finite = bool(np.asarray(metrics.finite_update_applied))
        after_version = int(np.asarray(following.library_version))
        after_steps = int(np.asarray(following.cumulative_gradient_steps))
        if after_version != before_version + int(finite) or after_steps != before_steps + int(
            finite
        ):
            raise AssertionError("completed update counters do not match finite publication")
        snapshot_stem = snapshots / f"u{index + 1:04d}"
        row = {
            "attempt": index + 1,
            "finite_update_applied": finite,
            "library_version_before": before_version,
            "library_version_after": after_version,
            "gradient_steps_before": before_steps,
            "gradient_steps_after": after_steps,
            "anchor_indices": (
                (before_version * max(count, 1) + np.arange(count)) % bank_count
            ).tolist(),
            "service_started_perf_counter_ns": started,
            "service_completed_perf_counter_ns": completed,
            "synchronized_service_seconds": service,
            "loss": _finite_scalar(metrics.loss.total),
            "gradient_norm": _finite_scalar(metrics.gradient_norm),
            "parameter_update_norm": _finite_scalar(metrics.parameter_update_norm),
            "persistent_state_before_sha256": before_identity,
            "persistent_state_after_sha256": _tree_digest(following),
            "completed_checkpoint": None,
            "completed_checkpoint_npz_sha256": None,
        }
        if finite:
            npz, _ = save_learner_checkpoint(
                following,
                bundle.spec,
                bundle.config,
                bundle.actuator,
                physical,
                snapshot_stem,
                metadata={**binding, "runtime_completed_update": row},
            )
            row["completed_checkpoint"] = str(snapshot_stem)
            row["completed_checkpoint_npz_sha256"] = hashlib.sha256(npz.read_bytes()).hexdigest()
        ledger.append(row)
        persistent = following
    save_learner_checkpoint(
        persistent,
        bundle.spec,
        bundle.config,
        bundle.actuator,
        physical,
        directory / "final_checkpoint",
        metadata={
            **reference_contract_checkpoint_metadata(directory / "nominal_reference"),
            "runtime_profile_repetitions": repetitions,
            "runtime_profile_initial_checkpoint_npz_sha256": bundle.sha256,
        },
    )
    _write(directory / "completed_updates.json", ledger)
    components = {}
    if profile_components:
        teacher = jax.jit(
            lambda state: (
                rollout_skill_library(
                    contract.params,
                    contract.spec,
                    state,
                    contract.model,
                    contract.actuator,
                    contract.actor_config,
                ).states
            )
        )

        @jax.jit
        def current_model_batch(params: Any, state: jax.Array, version: jax.Array) -> Any:
            indices = (version * max(count, 1) + jnp.arange(count)) % bank_count
            initial_states = jnp.concatenate((state[None], contract.anchors[indices]), axis=0)
            return jax.vmap(
                lambda anchor: rollout_skill_library(
                    params, contract.spec, anchor, model, contract.actuator, bundle.config
                )
            )(initial_states)

        gradient = jax.jit(jax.value_and_grad(learner.loss, has_aux=True))
        arguments = (
            source_state.params,
            physical,
            model,
            source_state.previous_params,
            source_state.library_version,
        )
        components = {
            "same_state_nominal_teacher_rollout": _time_callable(
                lambda: teacher(physical), warmup, repetitions
            ),
            "current_model_current_state_plus_rotating_anchors": _time_callable(
                lambda: current_model_batch(
                    source_state.params, physical, source_state.library_version
                ),
                warmup,
                repetitions,
            ),
            "full_reference_objective_forward": _time_callable(
                lambda: learner.loss(*arguments), warmup, repetitions
            ),
            "full_reference_value_and_gradient": _time_callable(
                lambda: gradient(*arguments), warmup, repetitions
            ),
        }
    finite_count = sum(row["finite_update_applied"] for row in ledger)
    summary = {
        "schema": "da_plcbf_reference_microbatch_runtime_v1",
        "scope": (
            "synchronized learner service experiment; not a paced flight "
            "or controller deadline result"
        ),
        "contention_note": contention_note,
        "device": str(device),
        "device_kind": device.device_kind,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "anchor_batch_size": count,
        "anchor_bank_count": bank_count,
        "anchor_bank_indices_visited": sorted({i for row in ledger for i in row["anchor_indices"]}),
        "source_checkpoint_npz_sha256": bundle.sha256,
        "initial_library_version": initial_version,
        "final_library_version": int(np.asarray(persistent.library_version)),
        "initial_gradient_steps": initial_steps,
        "final_gradient_steps": int(np.asarray(persistent.cumulative_gradient_steps)),
        "warmup_updates_discarded": warmup,
        "measured_attempts": repetitions,
        "finite_completed_updates": finite_count,
        "every_finite_completed_update_has_bound_full_checkpoint": all(
            row["completed_checkpoint"] is not None
            for row in ledger
            if row["finite_update_applied"]
        ),
        "update_service": _statistics(times),
        "timing_exclusions": [
            "compilation and discarded warmup",
            "input transfer and synchronization before service call",
            "checkpoint serialization, array hashing and ledger output",
            "controller computation, physical integration and actual scheduling",
        ],
        "component_probe_scope": (
            "separately compiled fixed-state probes; nonadditive because fusion, outputs and "
            "memory traffic differ from the complete learner update"
        ),
        "component_probes": components,
        "production_period_seconds": bundle.config.dt * bundle.config.control_interval_steps,
        "paced_positive_updates_demonstrated": False,
    }
    _write(directory / "runtime.json", summary)
    return summary


def main() -> None:
    """Prepare and profile independent full-state continuations for declared microbatches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--anchor-batch", type=int, nargs="+", default=(2, 1, 0))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-components", action="store_true")
    parser.add_argument(
        "--contention-note",
        default="Device isolation not established; other workloads may be active.",
    )
    args = parser.parse_args()
    if (
        args.warmup < 1
        or args.repetitions < 1
        or len(set(args.anchor_batch)) != len(args.anchor_batch)
    ):
        parser.error("warmup/repetitions must be positive and anchor batches must be unique")
    device = jax.devices(args.device)[0]
    source, contract, _ = build_reference_skill_learner_from_checkpoint(
        args.checkpoint, device=device
    )
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for count in args.anchor_batch:
        directory = args.output / f"anchors-{count}"
        bundle, revised = prepare_runtime_variant(source, contract, directory, count)
        if not args.prepare_only:
            summary = profile_runtime_variant(
                bundle,
                revised,
                directory,
                device=device,
                warmup=args.warmup,
                repetitions=args.repetitions,
                contention_note=args.contention_note,
                profile_components=not args.skip_components,
            )
            summaries.append(summary)
            print(
                json.dumps(
                    {
                        "anchor_batch": count,
                        "finite_updates": summary["finite_completed_updates"],
                        "service": summary["update_service"],
                    }
                ),
                flush=True,
            )
    _write(
        args.output / "runtime_comparison.json",
        {"variants": summaries, "prepare_only": args.prepare_only},
    )


if __name__ == "__main__":
    main()
