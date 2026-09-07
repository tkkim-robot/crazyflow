"""Bind one saved nominal PD loss input and isolate teacher/student computation paths.

CPU sealing does not evaluate a rollout, loss, gradient, or optimizer. A separately
assigned GPU session evaluates a fixed list of pure graphs at frozen inputs. It
never calls learner.step, advances a flight, changes a tolerance, or publishes.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmark.da_plcbf_actuator_controller_cost import (
    COMPILATION_CACHE,
    _checkpoint_description,
    describe_tree,
    restore_tree,
    verify_array_archive,
)
from benchmark.da_plcbf_actuator_diagnostic_runtime import (
    digest,
    file_digest,
    gpu_snapshot,
    write_json,
)

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/da_plcbf/actuator-diagnostics-20260906/v1"
EPISODE = BASE / (
    "fresh-validation-v1/"
    "7158c904d8ffabf2f8c0662e3ed58fd3b79f1d065348c8f08b9a287cc40b4e00/attempt-00"
)
PROTOCOL = BASE / "pd-loss-origin-protocol-v1/protocol.json"
SCHEMA = "actuator_pd_nominal_loss_origin_v1"
CASES = ("current", "anchor0", "anchor1")
GRAPH_PLAN = (
    "production_loss",
    "production_loss_value_and_grad",
    "instrumented_original_layout",
    "instrumented_full_anchor_teacher",
    "teacher_closed_same_batch",
    "teacher_dynamic_same_batch",
    "identical_stopped_reference_value_and_grad",
)


def _byte_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.dtype != right.dtype:
        return np.ones(left.shape, dtype=bool)
    return np.any(
        np.ascontiguousarray(left).view(np.uint8).reshape((*left.shape, left.dtype.itemsize))
        != np.ascontiguousarray(right).view(np.uint8).reshape((*right.shape, right.dtype.itemsize)),
        axis=-1,
    )


def compare_arrays(left: Any, right: Any) -> dict[str, Any]:
    """Distinguish exact representation changes from their numerical magnitude."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("compared arrays must have identical shapes")
    if left.dtype.kind not in "biuf" or right.dtype.kind not in "biuf":
        raise ValueError("compared arrays must be real numeric arrays")
    changed = _byte_difference(left, right)
    first = np.argwhere(changed)
    finite = bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right)))
    delta = left.astype(np.float64) - right.astype(np.float64)
    return {
        "shape": list(left.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "dtype_equal": left.dtype == right.dtype,
        "byte_equal": not bool(np.any(changed)),
        "finite": finite,
        "l2_delta": float(np.linalg.norm(delta.ravel())) if finite else None,
        "max_absolute_delta": float(np.max(np.abs(delta), initial=0)) if finite else None,
        "first_differing_index": first[0].tolist() if len(first) else None,
    }


def compare_rollout_states(student: Any, reference: Any) -> list[dict[str, Any]]:
    """Locate the earliest differing integration node independently of skill storage order."""
    student, reference = np.asarray(student), np.asarray(reference)
    if (
        student.shape != reference.shape
        or student.ndim != 4
        or student.shape[0] != len(CASES)
        or student.shape[-1] != 17
        or not all(student.shape)
    ):
        raise ValueError("rollout states require matching (3,K,T,17) shapes")
    rows = []
    for case, left, right in zip(CASES, student, reference, strict=True):
        changed = np.argwhere(_byte_difference(left, right))
        finite = np.all(np.isfinite(left)) and np.all(np.isfinite(right))
        rows.append(
            {
                "case": case,
                "comparison": compare_arrays(left, right),
                "first_differing_node": int(np.min(changed[:, 1])) if len(changed) else None,
                "per_component_max_absolute_delta": (
                    np.max(np.abs(left.astype(float) - right.astype(float)), axis=(0, 1)).tolist()
                    if finite
                    else None
                ),
            }
        )
    return rows


def _saved_case(episode: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load and authenticate the first actual learner input without evaluating the learner."""
    import jax.numpy as jnp

    from crazyflow.safety.da_plcbf.actuator_experiment import _hash_tree
    from crazyflow.safety.da_plcbf.actuator_learning import load_actuator_learner_checkpoint

    binding = json.loads((episode / "binding.json").read_text())
    summary = json.loads((episode / "summary.json").read_text())
    updates = json.loads((episode / "updates.json").read_text())
    checkpoint = load_actuator_learner_checkpoint(episode / "snapshots/initial")
    scene = binding["scene"]
    if (
        summary["method"] != "PD_A"
        or scene["family"] != "navigation"
        or scene["scene_seed"] != 62104
        or scene["physical_spec"]["actuator_events"]
        or scene["effectiveness_after"] != [1.0] * 4
        or scene["lag_multipliers_after"] != [1.0] * 4
    ):
        raise ValueError("loss-origin probe requires the declared nominal navigation62104 case")
    if (
        _hash_tree(checkpoint.state) != binding["initial_learner_sha256"]
        or int(checkpoint.state.library_version) != 0
        or int(checkpoint.state.cumulative_gradient_steps) != 0
        or asdict(checkpoint.config) != asdict(checkpoint.contract.actor_config)
        or checkpoint.contract.learning_config.objective_mode != "balanced_reference_braking"
        or checkpoint.contract.learning_config.anchor_batch_size != 2
    ):
        raise ValueError("saved initial learner state, actor config or objective changed")
    import jax

    if any(
        np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(checkpoint.state.optimizer_state)
    ):
        raise ValueError("initial saved Adam state must be identically zero")
    teacher_params = _hash_tree(checkpoint.contract.params)
    if any(
        _hash_tree(params) != teacher_params
        for params in (checkpoint.state.params, checkpoint.state.previous_params)
    ):
        raise ValueError("initial current and previous parameters must exactly equal the teacher")
    teacher_model = _hash_tree(checkpoint.contract.model)
    if _hash_tree(checkpoint.state.latest_dynamics_estimate) != teacher_model:
        raise ValueError("initial saved learner model differs from the teacher")
    with np.load(episode / "controls.npz", allow_pickle=False) as controls:
        if float(controls["time"][0]) != 0.0:
            raise ValueError("first saved control is not the time-zero input")
        initial = jnp.asarray(controls["controller_input_state"][0])
        state_hash = str(controls["controller_input_state_sha256"][0])
        if _hash_tree(initial) != state_hash:
            raise ValueError("saved controller input state does not authenticate")
        if str(controls["estimated_model_sha256"][0]) != teacher_model:
            raise ValueError("actual time-zero learner model bytes differ from the teacher")
    first = updates[0]
    if (
        first["training_simulation_time"] != 0.0
        or first["sensed_state_sha256"] != state_hash
        or first["estimated_model_sha256"] != teacher_model
        or first["computed_version"] != 1
    ):
        raise ValueError("original first update does not match the authenticated frozen inputs")
    inputs = {
        "runtime": {
            "params": checkpoint.state.params,
            "initial_state": initial,
            "model": checkpoint.state.latest_dynamics_estimate,
            "previous": checkpoint.state.previous_params,
            "iteration": checkpoint.state.library_version,
        },
        "reference": {
            "params": checkpoint.contract.params,
            "model": checkpoint.contract.model,
            "anchors": checkpoint.contract.anchors,
            "spec": asdict(checkpoint.contract.spec),
        },
    }
    metadata = {
        "episode": str(episode.resolve()),
        "physical_world_id": summary["physical_world_id"],
        "checkpoint": _checkpoint_description(checkpoint),
        "reference_sha256": binding["checkpoint"]["runtime_reference_sha256"],
        "recorded_first_update": first,
        "initial_learner_sha256": binding["initial_learner_sha256"],
        "runtime_state_sha256": state_hash,
        "runtime_model_sha256": teacher_model,
        "runtime_teacher_parameter_sha256": teacher_params,
        "input_model_provenance": (
            "saved checkpoint model has the same runtime tree/dtype/shape/byte digest as "
            "the model supplied to the first actual update; runtime hashes do not bind weak_type"
        ),
        "anchor_indices": [0, 1],
        "full_anchor_bank_shape": list(checkpoint.contract.anchors.shape),
        "shared_skill_specification": True,
        "teacher_runtime_actor_config_equal": True,
        "initial_adam_all_zero": True,
        "original_episode_source_sha256": binding["source_sha256"],
    }
    return checkpoint, inputs, metadata


def load_protocol(path: Path) -> dict[str, Any]:
    """Authenticate all frozen sources, original evidence and prepared numerical inputs."""
    envelope = json.loads(path.read_text())
    protocol = envelope["protocol"]
    if protocol.get("schema") != SCHEMA or digest(protocol) != envelope.get("sha256"):
        raise ValueError("loss-origin protocol schema or content digest changed")
    for name, expected in protocol["files_sha256"].items():
        if file_digest(Path(name)) != expected:
            raise ValueError(f"bound loss-origin file changed: {name}")
    archive = Path(protocol["input_archive"])
    if file_digest(archive) != protocol["input_archive_sha256"]:
        raise ValueError("loss-origin input archive bytes changed")
    verify_array_archive(archive, [protocol["input_description"]])
    return protocol


def seal_protocol(output: Path, *, episode: Path = EPISODE) -> dict[str, Any]:
    """Prepare immutable inputs on CPU after source review, without any rollout evaluation."""
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("CPU seal requires explicit JAX_PLATFORMS=cpu before importing JAX")
    import jax

    from benchmark.da_plcbf_actuator_diagnostics import source_binding

    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("loss-origin protocol sealing must use only CPU")
    episode = episode.resolve()
    checkpoint, inputs, metadata = _saved_case(episode)
    arrays: dict[str, np.ndarray] = {}
    description = describe_tree(inputs, "pd_t0", arrays)
    sources = source_binding()
    for name in (
        "benchmark/da_plcbf_actuator_diagnostic_runtime.py",
        "benchmark/da_plcbf_actuator_controller_cost.py",
        "benchmark/da_plcbf_actuator_pd_loss_origin.py",
        "tests/test_da_plcbf_actuator_pd_loss_origin.py",
    ):
        sources[name] = file_digest(ROOT / name)
    for name, expected in metadata["original_episode_source_sha256"].items():
        if file_digest(ROOT / "crazyflow/safety/da_plcbf" / name) != expected:
            raise ValueError(f"original episode numerical source changed: {name}")
    files = {str(ROOT / name): sha for name, sha in sources.items()}
    for path in (
        episode / "binding.json",
        episode / "summary.json",
        episode / "controls.npz",
        episode / "updates.json",
        checkpoint.npz_path,
        checkpoint.json_path,
    ):
        files[str(path.resolve())] = file_digest(path)
    metadata_report = BASE / "pd-loss-origin-metadata-v1/report.json"
    files[str(metadata_report)] = file_digest(metadata_report)
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "inputs.npz"
    np.savez_compressed(archive, **arrays)
    protocol = {
        "schema": SCHEMA,
        "case": metadata,
        "input_description": description,
        "input_archive": str(archive.resolve()),
        "input_archive_sha256": file_digest(archive),
        "files_sha256": files,
        "source_sha256": sources,
        "jax_version_at_seal": jax.__version__,
        "jax_x64_enabled": bool(jax.config.x64_enabled),
        "compilation_cache_directory": COMPILATION_CACHE,
        "graph_plan": list(GRAPH_PLAN),
        "production_anchor_setup": (
            "build_actuator_skill_learner evaluates its original full-bank cached teacher "
            "states once; the probe reads this exact closed-over cache without modifying it"
        ),
        "execution_contract": (
            "one evaluation of each fixed graph at the same frozen time-zero inputs; no "
            "learner.step, optimizer update, flight, publication, tolerance or precision repair; "
            "execution timings are descriptive only, with no latency claim or CPU quiet claim"
        ),
        "compilation_qualification": (
            "production learner.loss and its value_and_grad are new pure executions; neither "
            "is claimed to recreate the original optimizer-fused learner.step executable. "
            "Instrumentation and extra outputs can change compilation. Matched inputs plus "
            "different graph outputs localize a discrepancy without proving a sole cause"
        ),
        "anchor_command_qualification": (
            "the production anchor cache stores states only; full anchor commands come from "
            "a separate explicitly instrumented full-bank teacher graph, whose states are "
            "compared to the actual production cache before interpreting its commands"
        ),
    }
    protocol = json.loads(json.dumps(protocol))
    envelope = {"protocol": protocol, "sha256": digest(protocol)}
    write_json(output / "protocol.json", envelope, exclusive=True)
    for name in sources:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    for name in files:
        if name not in {str(ROOT / source) for source in sources}:
            relative = (
                Path(name).relative_to(episode)
                if Path(name).is_relative_to(episode)
                else Path(name).name
            )
            destination = output / "saved-episode" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(name, destination)
    load_protocol(output / "protocol.json")
    return envelope


def _json_numeric(value: Any) -> Any:
    if hasattr(value, "_asdict"):
        return {key: _json_numeric(item) for key, item in value._asdict().items()}
    if isinstance(value, dict):
        return {key: _json_numeric(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_numeric(item) for item in value]
    array = np.asarray(value)
    if array.ndim:
        return _json_numeric(array.tolist())
    scalar = array.item()
    if isinstance(scalar, float) and not np.isfinite(scalar):
        return str(scalar)
    return scalar


def _gradient_description(gradient: Any) -> dict[str, Any]:
    import jax

    leaves = [np.asarray(leaf, dtype=float) for leaf in jax.tree.leaves(gradient)]
    norm = float(np.sqrt(sum(np.sum(v * v) for v in leaves)))
    maximum = float(max(np.max(np.abs(v), initial=0) for v in leaves))
    return {
        "l2_norm_float64_host_reduction": norm if np.isfinite(norm) else None,
        "maximum_absolute": maximum if np.isfinite(maximum) else None,
        "all_exact_zero": all(np.all(v == 0) for v in leaves),
        "all_finite": all(np.all(np.isfinite(v)) for v in leaves),
    }


def _run_probe(protocol_path: Path, output: Path, *, gpu_slot_assigned: bool) -> dict[str, Any]:
    if not gpu_slot_assigned:
        raise RuntimeError("GPU execution requires the explicit root-assigned-slot flag")
    protocol = load_protocol(protocol_path)
    if os.environ.get("JAX_COMPILATION_CACHE_DIR") != protocol["compilation_cache_directory"]:
        raise RuntimeError("GPU execution requires the explicitly sealed historical JAX cache")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol_path, output / "protocol.json")

    def occupancy(label: str) -> None:
        observed = gpu_snapshot()
        write_json(output / f"gpu-{label}.json", observed, exclusive=True)
        if observed["blocking_foreign_compute_processes"]:
            raise RuntimeError("foreign GPU compute process blocks loss-origin execution")

    occupancy("before")
    import jax
    import jax.numpy as jnp

    from crazyflow.safety.da_plcbf.actuator_experiment import _hash_tree
    from crazyflow.safety.da_plcbf.actuator_learning import (
        _balanced_reference_loss,
        build_actuator_skill_learner,
        rollout_actuator_skill_library,
    )
    from crazyflow.safety.da_plcbf.persistent_skill_learner import SkillLibrarySpec

    if jax.default_backend() != "gpu" or any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("loss-origin execution requires the assigned CUDA device")
    if (
        bool(jax.config.x64_enabled) != protocol["jax_x64_enabled"]
        or jax.__version__ != protocol["jax_version_at_seal"]
    ):
        raise ValueError("JAX version or precision differs from the sealed context")
    checkpoint, template, metadata = _saved_case(Path(protocol["case"]["episode"]))
    if json.loads(json.dumps(metadata)) != protocol["case"]:
        raise ValueError("saved input or checkpoint authentication changed after sealing")
    with np.load(protocol["input_archive"], allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    restored = restore_tree(template, protocol["input_description"], saved)
    if describe_tree(restored, "pd_t0", {}) != protocol["input_description"]:
        raise ValueError("restored runtime inputs differ from their sealed representation")
    runtime, reference = restored["runtime"], restored["reference"]
    contract = replace(
        checkpoint.contract,
        params=reference["params"],
        model=reference["model"],
        anchors=reference["anchors"],
        spec=SkillLibrarySpec(**reference["spec"]),
    )
    config, settings = checkpoint.config, contract.learning_config
    args = tuple(
        runtime[key] for key in ("params", "initial_state", "model", "previous", "iteration")
    )
    params, initial, model, previous, iteration = jax.block_until_ready(args)
    initial_batch = jnp.concatenate((initial[None], contract.anchors[:2]), axis=0)
    began = time.perf_counter()
    learner = build_actuator_skill_learner(contract, config)
    anchor_reference = inspect.getclosurevars(learner.loss.__wrapped__).nonlocals[
        "anchor_reference"
    ]
    jax.block_until_ready(anchor_reference)
    setup_seconds = time.perf_counter() - began
    np.savez_compressed(
        output / "production-cached-anchor-states.npz", states=np.asarray(anchor_reference)
    )
    anchor_hash = _hash_tree(anchor_reference)
    write_json(
        output / "environment.json",
        {
            "jax_version": jax.__version__,
            "jax_x64_enabled": bool(jax.config.x64_enabled),
            "default_matmul_precision": str(jax.config.jax_default_matmul_precision),
            "jax_platforms_environment": os.environ.get("JAX_PLATFORMS"),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "jax_compilation_cache_dir": str(jax.config.jax_compilation_cache_dir),
            "devices": [str(device) for device in jax.devices()],
            "production_anchor_setup_seconds": setup_seconds,
            "production_anchor_cache_sha256": anchor_hash,
            "production_anchor_cache_shape": list(anchor_reference.shape),
            "optimizer_step_calls": 0,
        },
        exclusive=True,
    )
    graph_records, results = [], {}

    def evaluate(name: str, function: Callable, arguments: tuple) -> Any:
        if name != GRAPH_PLAN[len(graph_records)]:
            raise ValueError("pure graph evaluation order differs from sealed fixed plan")
        occupancy(name)
        lowered = function.lower(*arguments)
        hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
        hlo_path = output / f"{name}.lowered.hlo.txt"
        hlo_path.write_text(hlo)
        compiled = lowered.compile()
        optimized_path = output / f"{name}.compiled.hlo.txt"
        optimized = compiled.as_text()
        optimized_path.write_text(optimized)
        started = time.perf_counter()
        result = jax.block_until_ready(compiled(*arguments))
        elapsed = time.perf_counter() - started
        arrays = {
            f"leaf-{index:04d}": np.asarray(leaf)
            for index, leaf in enumerate(jax.tree.leaves(result))
        }
        archive = output / f"{name}.npz"
        np.savez_compressed(archive, **arrays)
        paths, tree = jax.tree_util.tree_flatten_with_path(result)
        record = {
            "name": name,
            "evaluation_index": len(graph_records),
            "synchronized_execution_seconds": elapsed,
            "result_sha256": _hash_tree(result),
            "array_archive": str(archive.resolve()),
            "array_archive_sha256": file_digest(archive),
            "tree_structure": str(tree),
            "array_paths": {
                f"leaf-{index:04d}": jax.tree_util.keystr(path)
                for index, (path, _leaf) in enumerate(paths)
            },
            "lowered_hlo_sha256": file_digest(hlo_path),
            "compiled_hlo_sha256": file_digest(optimized_path),
            "all_finite": all(np.all(np.isfinite(array)) for array in arrays.values()),
        }
        write_json(output / f"{name}.json", record, exclusive=True)
        graph_records.append(record)
        results[name] = result
        return result

    production = evaluate("production_loss", learner.loss, args)
    differentiated = evaluate(
        "production_loss_value_and_grad",
        jax.jit(jax.value_and_grad(learner.loss, has_aux=True)),
        args,
    )

    def original_layout(p: Any, state: Any, current_model: Any, version: Any) -> tuple:
        indices = (version * 2 + jnp.arange(2)) % len(contract.anchors)
        starts = jnp.concatenate((state[None], contract.anchors[indices]), axis=0)
        actual = jax.vmap(
            lambda y: rollout_actuator_skill_library(p, contract.spec, y, current_model, config)
        )(starts)
        teacher = rollout_actuator_skill_library(
            contract.params, contract.spec, state, contract.model, contract.actor_config
        )
        references = jax.lax.stop_gradient(
            jnp.concatenate((teacher.states[None], anchor_reference[indices]), axis=0)
        )
        loss = _balanced_reference_loss(
            p, previous, actual, references, current_model, config, settings
        )
        return actual, teacher, references, loss

    actual, teacher, references, instrumented_loss = evaluate(
        "instrumented_original_layout",
        jax.jit(original_layout),
        (params, initial, model, iteration),
    )

    def closed_teacher(starts: Any) -> Any:
        return jax.vmap(
            lambda y: rollout_actuator_skill_library(
                contract.params, contract.spec, y, contract.model, contract.actor_config
            )
        )(starts)

    full_teacher = evaluate(
        "instrumented_full_anchor_teacher", jax.jit(closed_teacher), (contract.anchors,)
    )
    closed = evaluate("teacher_closed_same_batch", jax.jit(closed_teacher), (initial_batch,))

    def dynamic_teacher(p: Any, starts: Any, current_model: Any) -> Any:
        return jax.vmap(
            lambda y: rollout_actuator_skill_library(
                p, contract.spec, y, current_model, contract.actor_config
            )
        )(starts)

    dynamic = evaluate(
        "teacher_dynamic_same_batch", jax.jit(dynamic_teacher), (params, initial_batch, model)
    )

    def identical_reference(p: Any, state: Any, current_model: Any, old: Any, version: Any) -> Any:
        indices = (version * 2 + jnp.arange(2)) % len(contract.anchors)
        starts = jnp.concatenate((state[None], contract.anchors[indices]), axis=0)
        same_actual = jax.vmap(
            lambda y: rollout_actuator_skill_library(p, contract.spec, y, current_model, config)
        )(starts)
        return _balanced_reference_loss(
            p,
            old,
            same_actual,
            jax.lax.stop_gradient(same_actual.states),
            current_model,
            config,
            settings,
        )

    identical = evaluate(
        "identical_stopped_reference_value_and_grad",
        jax.jit(jax.value_and_grad(identical_reference, has_aux=True)),
        args,
    )
    if _checkpoint_description(checkpoint) != protocol["case"]["checkpoint"]:
        raise ValueError("pure probes changed a complete saved learner state")
    if describe_tree(restored, "pd_t0", {}) != protocol["input_description"]:
        raise ValueError("pure probes changed sealed current or teacher inputs")
    if _hash_tree(anchor_reference) != anchor_hash:
        raise ValueError("pure probes changed the original production teacher cache")
    load_protocol(protocol_path)
    occupancy("after")
    original = protocol["case"]["recorded_first_update"]
    reference_commands = np.concatenate(
        (np.asarray(teacher.commands)[None], np.asarray(full_teacher.commands)[:2]), axis=0
    )
    report = {
        "status": "completed",
        "protocol_sha256": digest(protocol),
        "case": protocol["case"],
        "execution_contract": protocol["execution_contract"],
        "compilation_qualification": protocol["compilation_qualification"],
        "anchor_command_qualification": protocol["anchor_command_qualification"],
        "graph_evaluations": graph_records,
        "production_anchor_setup_evaluations": 1,
        "optimizer_step_calls": 0,
        "flight_calls": 0,
        "publications": 0,
        "all_sealed_inputs_and_complete_states_unchanged": True,
        "recorded_original_loss": original["loss"],
        "recorded_original_gradient_norm": original["gradient_norm"],
        "recorded_original_parameter_update_norm": original["parameter_update_norm"],
        "production_pure_loss": _json_numeric(production),
        "production_value_and_grad_loss": _json_numeric(differentiated[0]),
        "production_gradient": _gradient_description(differentiated[1]),
        "instrumented_original_layout_loss": _json_numeric(instrumented_loss),
        "original_vs_production_total": compare_arrays(
            np.asarray(original["loss"]["total"], dtype=np.asarray(production[0]).dtype),
            production[0],
        ),
        "student_vs_original_reference_states": compare_rollout_states(actual.states, references),
        "closed_same_batch_vs_original_reference_states": compare_rollout_states(
            closed.states, references
        ),
        "dynamic_same_batch_vs_student_states": compare_rollout_states(
            dynamic.states, actual.states
        ),
        "closed_vs_dynamic_same_batch_states": compare_rollout_states(
            closed.states, dynamic.states
        ),
        "instrumented_full_anchor_vs_production_cached_states": compare_arrays(
            full_teacher.states, anchor_reference
        ),
        "student_vs_instrumented_reference_commands": compare_arrays(
            actual.commands, reference_commands
        ),
        "identical_stopped_reference_loss": _json_numeric(identical[0]),
        "identical_stopped_reference_gradient": _gradient_description(identical[1]),
        "identical_stopped_reference_exact_zero": (
            float(identical[0][0]) == 0 and _gradient_description(identical[1])["all_exact_zero"]
        ),
    }
    write_json(output / "report.json", report, exclusive=True)
    return report


def run_probe(
    protocol_path: Path, output: Path, *, gpu_slot_assigned: bool = False
) -> dict[str, Any]:
    """Retain an exclusive failed attempt and its completed graph count without automatic retry."""
    existed = output.exists()
    try:
        return _run_probe(protocol_path, output, gpu_slot_assigned=gpu_slot_assigned)
    except (Exception, KeyboardInterrupt) as error:
        if not existed and output.is_dir():
            write_json(
                output / "failure.json",
                {
                    "status": "failed" if isinstance(error, Exception) else "interrupted",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "completed_graph_count": sum(
                        (output / f"{name}.json").exists() for name in GRAPH_PLAN
                    ),
                    "policy": "retained without replacement or automatic rerun",
                },
                exclusive=True,
            )
        raise


def main() -> None:
    """Keep reviewed CPU preparation separate from the explicitly assigned CUDA session."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--output", type=Path, default=PROTOCOL.parent)
    seal.add_argument("--episode", type=Path, default=EPISODE)
    run = commands.add_parser("run")
    run.add_argument("--protocol", type=Path, default=PROTOCOL)
    run.add_argument("--output", type=Path, default=BASE / "pd-loss-origin-measured-v1")
    run.add_argument("--gpu-slot-assigned", action="store_true")
    args = parser.parse_args()
    if args.command == "seal":
        result = seal_protocol(args.output, episode=args.episode)
        print(json.dumps({"protocol_sha256": result["sha256"], "output": str(args.output)}))
    else:
        result = run_probe(args.protocol, args.output, gpu_slot_assigned=args.gpu_slot_assigned)
        print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
