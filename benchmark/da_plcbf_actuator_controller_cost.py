"""Seal and microprofile complete controllers on two immutable saved F2 inputs.

This measures synchronized controller service only. No plant advances, learner
steps, publications, or new flight outcomes are produced. Warmups and first-call
costs remain separate from the balanced sequence of 300 measured calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmark.da_plcbf_actuator_diagnostic_runtime import (
    digest,
    file_digest,
    gpu_snapshot,
    write_json,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/da_plcbf/actuator-diagnostics-20260906/v1"
EPISODE = BASE / (
    "library-comparison-v1/"
    "d81d72d8efbcb42fc4bce9ffd512594e239ade5b3fe07604552efadec2377c17/attempt-00"
)
PROTOCOL = BASE / "controller-cost-protocol-v1/protocol.json"
SCHEMA = "actuator_common_input_controller_cost_v1"
METHODS = ("F2", "UNION", "F2_2K", "A_BAL", "PD_F")
FALLBACK_COUNTS = {"F2": 16, "UNION": 32, "F2_2K": 32, "A_BAL": 16, "PD_F": 16}
CASES = (("prefault_t0", 0.0), ("postfault_t2.4", 2.4))
WARMUP_CALLS = 3
REPETITIONS = 30
COMPILATION_CACHE = "/tmp/crazyflow-actuator-diagnostic-jax-v1"
PARAMETER_SOURCE_TRIALS = {
    "F2": "d81d72d8efbcb42fc4bce9ffd512594e239ade5b3fe07604552efadec2377c17",
    "A_BAL": "8156b0c402449c557f1bcf63f53fe6fc29b6ac2d3c5f3f26879f075ddd5a8576",
    "PD_F": "d12260becfd55327f979957ce6574a5d638677bd5476f39787fafed4973ec25a",
    "UNION": "0c37ef603c917db46e440c3e38f4df2c567987764f07d58faa3f33458b961757",
    "F2_2K": "6d4485f84e8cfa30932ff2348a9665e83375abfb7956676bb8d075a13ea9e9e3",
}


def balanced_plan(
    methods: Sequence[str], cases: Sequence[str], repetitions: int, *, phase: str
) -> list[dict[str, Any]]:
    """Rotate all methods through all positions, reversing alternate complete cycles."""
    methods, cases = tuple(methods), tuple(cases)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be nonempty and unique")
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("cases must be nonempty and unique")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if phase not in {"warmup", "measured"}:
        raise ValueError("unknown profiling phase")
    plan = []
    for repetition in range(repetitions):
        offset = repetition % len(methods)
        method_order = methods[offset:] + methods[:offset]
        if (repetition // len(methods)) % 2:
            method_order = method_order[::-1]
        case_order = cases if repetition % 2 == 0 else cases[::-1]
        for case_position, case in enumerate(case_order):
            for method_position, method in enumerate(method_order):
                plan.append(
                    {
                        "sample_index": len(plan),
                        "phase": phase,
                        "repetition": repetition,
                        "case": case,
                        "method": method,
                        "case_position": case_position,
                        "method_position": method_position,
                    }
                )
    return plan


def array_description(value: Any, *, weak_type: bool) -> dict[str, Any]:
    """Bind an input's exact representation, including signed zero and scalar weak type."""
    array = np.asarray(value)
    if array.dtype.kind not in "biufc" or not np.all(np.isfinite(array)):
        raise ValueError("controller inputs must contain finite numeric arrays")
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "weak_type": bool(weak_type),
    }


def describe_tree(tree: Any, prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Record every JAX input leaf without executing a controller or retaining device layout."""
    import jax

    leaves, structure = jax.tree_util.tree_flatten_with_path(tree)
    records = []
    for index, (path, value) in enumerate(leaves):
        key = f"{prefix}.leaf-{index:04d}"
        array = np.asarray(value)
        arrays[key] = array.copy()
        records.append(
            {
                "array_key": key,
                "tree_path": jax.tree_util.keystr(path),
                **array_description(
                    value, weak_type=getattr(jax.typeof(value), "weak_type", False)
                ),
            }
        )
    return {"tree_structure": str(structure), "leaves": records}


def verify_array_archive(path: Path, descriptions: Sequence[dict[str, Any]]) -> None:
    """Reject archive value, dtype, shape, or membership changes using the bound leaf manifests."""
    leaves = [leaf for description in descriptions for leaf in description["leaves"]]
    if len({leaf["array_key"] for leaf in leaves}) != len(leaves):
        raise ValueError("duplicate bound input array keys")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {leaf["array_key"] for leaf in leaves}:
            raise ValueError("bound controller input archive membership changed")
        for leaf in leaves:
            actual = array_description(archive[leaf["array_key"]], weak_type=leaf["weak_type"])
            if any(actual[name] != leaf[name] for name in actual):
                raise ValueError(f"bound controller input array changed: {leaf['array_key']}")


def restore_tree(template: Any, description: dict[str, Any], archive: dict[str, np.ndarray]) -> Any:
    """Transfer sealed leaf bytes instead of numerically regenerating device-side inputs."""
    import jax
    import jax.numpy as jnp

    leaves, structure = jax.tree_util.tree_flatten_with_path(template)
    if str(structure) != description["tree_structure"] or len(leaves) != len(description["leaves"]):
        raise ValueError("common input template structure changed")
    restored = []
    for (path, _value), expected in zip(leaves, description["leaves"], strict=True):
        if jax.tree_util.keystr(path) != expected["tree_path"]:
            raise ValueError("common input template leaf path changed")
        array = archive[expected["array_key"]]
        if expected["weak_type"]:
            if array.shape:
                raise ValueError("weak common inputs must be scalars")
            value = jnp.asarray(array.item())
        else:
            value = jax.device_put(array)
        actual = array_description(value, weak_type=getattr(jax.typeof(value), "weak_type", False))
        if any(actual[key] != expected[key] for key in actual):
            raise ValueError("sealed input transfer changed value, dtype or weak type")
        restored.append(value)
    return jax.tree_util.tree_unflatten(structure, restored)


def load_protocol(path: Path, *, verify_files: bool = True) -> dict[str, Any]:
    """Authenticate the sealed schedule and all live source/checkpoint/input bytes."""
    envelope = json.loads(path.read_text())
    protocol = envelope["protocol"]
    if protocol.get("schema") != SCHEMA or envelope.get("sha256") != digest(protocol):
        raise ValueError("controller-cost protocol schema or digest changed")
    if verify_files:
        for name, expected in protocol["files_sha256"].items():
            if file_digest(Path(name)) != expected:
                raise ValueError(f"bound controller-cost input changed: {name}")
        if file_digest(Path(protocol["input_archive"])) != protocol["input_archive_sha256"]:
            raise ValueError("bound controller input archive bytes changed")
        verify_array_archive(
            Path(protocol["input_archive"]),
            [case["common_input_description"] for case in protocol["cases"]],
        )
    return protocol


def _checkpoint_description(bundle: Any) -> dict[str, Any]:
    from crazyflow.safety.da_plcbf.actuator_experiment import _hash_tree

    description = {
        "stem": str(bundle.npz_path.with_suffix("").resolve()),
        "npz_file_sha256": file_digest(bundle.npz_path),
        "json_file_sha256": file_digest(bundle.json_path),
        "complete_state_sha256": _hash_tree(bundle.state),
        "params_sha256": _hash_tree(bundle.state.params),
        "optimizer_sha256": _hash_tree(bundle.state.optimizer_state),
        "model_sha256": _hash_tree(bundle.state.latest_dynamics_estimate),
        "library_version": int(bundle.state.library_version),
        "cumulative_gradient_steps": int(bundle.state.cumulative_gradient_steps),
        "actor_config": asdict(bundle.config),
        "learning_config": asdict(bundle.contract.learning_config),
    }
    return json.loads(json.dumps(description))


def _recorded_cases(episode: Path) -> tuple[Any, list[dict[str, Any]], dict[str, tuple], dict]:
    import jax.numpy as jnp

    from benchmark.da_plcbf_actuator_confirmation import _recorded_inputs, load_saved_run

    run = load_saved_run(episode)
    if run.summary["method"] != "F2" or run.binding["scene"]["scene_seed"] != 30101:
        raise ValueError("common inputs require the declared saved F2 seed-30101 episode")
    rows, values, arrays = [], {}, {}
    for label, when in CASES:
        index = run.boundary_index(when)
        state, model, obstacles, safety, previous, goal, config = _recorded_inputs(run, index, when)
        # Match control_call's Python-int conversion, including its weak scalar type.
        previous = jnp.asarray(int(previous))
        common = (state, model, obstacles, safety, previous, goal)
        values[label] = common
        rows.append(
            {
                "case": label,
                "time_seconds": when,
                "saved_control_index": index,
                "controller_input_state_sha256": str(
                    run.controls["controller_input_state_sha256"][index]
                ),
                "estimated_model_sha256": str(run.controls["estimated_model_sha256"][index]),
                "previous_selected_index": int(previous),
                "filter_config": asdict(config),
                "common_input_description": describe_tree(common, label, arrays),
            }
        )
    return run, rows, values, arrays


def _resolve_methods(resources: Any, filter_config: Any) -> tuple[dict, dict]:
    descriptions, resolved = {}, {}
    for method in METHODS:
        bundle, functions, _unused_learner, metadata = resources.resolve(method, filter_config)
        if metadata["total_fallback_count"] != FALLBACK_COUNTS[method]:
            raise ValueError(f"unexpected fallback policy count for {method}")
        core = resources.bundle("F2") if method in {"UNION", "F2_2K"} else None
        descriptions[method] = {
            "checkpoint": _checkpoint_description(bundle),
            "immutable_core": _checkpoint_description(core) if core is not None else None,
            "resolved_metadata": metadata,
            "total_fallback_count": FALLBACK_COUNTS[method],
            "total_candidate_count_including_nominal": FALLBACK_COUNTS[method] + 1,
            "controller_wrapper": type(functions.controller).__name__,
        }
        if descriptions[method]["controller_wrapper"] != "PackedActuatorController":
            raise ValueError("cost benchmark requires the runtime's complete packed controller")
        resolved[method] = (bundle, functions.controller)
    return json.loads(json.dumps(descriptions)), resolved


def _parameter_cells(resources: Any, common_run: Any) -> tuple[dict, dict, dict]:
    """Authenticate each method's own published 2.4 s parameters while fixing common inputs."""
    from benchmark.da_plcbf_actuator_confirmation import (
        authenticate_boundary_snapshot,
        load_saved_run,
    )
    from crazyflow.safety.da_plcbf.actuator_experiment import _hash_tree

    descriptions = {label: {} for label, _when in CASES}
    bundles, source_files = {}, {}
    for method in METHODS:
        startup = resources.bundle(method)
        directory = common_run.directory.parents[1] / PARAMETER_SOURCE_TRIALS[method] / "attempt-00"
        own_run = common_run if directory == common_run.directory else load_saved_run(directory)
        if (
            own_run.summary["method"] != method
            or own_run.binding["scene"]["physical_spec"]
            != common_run.binding["scene"]["physical_spec"]
            or own_run.binding["config"]["filter_config"]
            != common_run.binding["config"]["filter_config"]
            or own_run.binding["initial_learner_sha256"] != _hash_tree(startup.state)
        ):
            raise ValueError(
                "parameter-source method/world/filter/startup differs from declared anchor"
            )
        for name in (
            "binding.json",
            "summary.json",
            "dense.npz",
            "controls.npz",
            "applications.npz",
        ):
            path = directory / name
            source_files[str(path.resolve())] = file_digest(path)
        for label, when in CASES:
            if when == 0:
                bundle = startup
                authentication = {
                    "authenticated": True,
                    "source": "complete startup checkpoint authenticated against source episode",
                    "control_params_sha256": str(own_run.controls["control_params_sha256"][0]),
                }
                if authentication["control_params_sha256"] != _hash_tree(bundle.state.params):
                    raise ValueError("saved time-zero control parameters differ from startup")
            else:
                authenticated = authenticate_boundary_snapshot(
                    own_run, directory / "snapshots" / f"critical-{when:.9f}", when
                )
                bundle, authentication = authenticated.checkpoint, authenticated.report
                if authentication["control_params_reverted"]:
                    raise ValueError("profile requires the actual unreverted published parameters")
                if bundle.config != startup.config or _hash_tree(
                    asdict(bundle.contract.spec)
                ) != _hash_tree(asdict(startup.contract.spec)):
                    raise ValueError(
                        "published parameter architecture differs from startup controller"
                    )
                source_files.update(authentication["source_sha256"])
            descriptions[label][method] = {
                "checkpoint": _checkpoint_description(bundle),
                "source_episode": str(directory.resolve()),
                "authentication": authentication,
            }
            bundles[method, label] = bundle
    return json.loads(json.dumps(descriptions)), bundles, source_files


def _controller_args(common: tuple, params: Any) -> tuple:
    state, model, obstacles, safety, previous, goal = common
    return state, params, model, obstacles, safety, previous, goal


def seal_protocol(output: Path, *, episode: Path = EPISODE, base: Path = BASE) -> dict[str, Any]:
    """Prepare an exclusive CPU-only protocol; call only after source review is complete."""
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("CPU sealing requires explicit JAX_PLATFORMS=cpu before JAX import")
    import jax

    from benchmark.da_plcbf_actuator_diagnostics import DiagnosticResources, source_binding
    from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig

    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("protocol preparation must run exclusively on CPU")
    run, cases, common, arrays = _recorded_cases(episode)
    resources = DiagnosticResources(base)
    methods, resolved = _resolve_methods(
        resources, ActuatorFilterConfig(**cases[0]["filter_config"])
    )
    if (
        methods["F2"]["checkpoint"]["complete_state_sha256"]
        != run.binding["initial_learner_sha256"]
    ):
        raise ValueError("F2 profile checkpoint differs from the saved source episode startup")
    parameter_cells, cell_bundles, parameter_source_files = _parameter_cells(resources, run)
    input_signatures = {}
    for case in cases:
        label = case["case"]
        input_signatures[label] = {}
        for method in resolved:
            bundle = cell_bundles[method, label]
            # Full parameter values are bound by checkpoint files; descriptions also bind
            # their exact call-site dtype and weak type alongside every shared input leaf.
            input_signatures[label][method] = describe_tree(
                _controller_args(common[label], bundle.state.params), f"{label}.{method}", {}
            )
    sources = source_binding()
    for name in (
        "benchmark/da_plcbf_actuator_confirmation.py",
        "benchmark/da_plcbf_actuator_diagnostic_runtime.py",
        "benchmark/da_plcbf_actuator_controller_cost.py",
        "tests/test_da_plcbf_actuator_controller_cost.py",
    ):
        sources[name] = file_digest(ROOT / name)
    raw_files = {
        str((episode / name).resolve()): file_digest(episode / name)
        for name in (
            "binding.json",
            "summary.json",
            "controls.npz",
            "dense.npz",
            "applications.npz",
        )
    }
    checkpoint_files = {}
    for description in methods.values():
        for checkpoint in (description["checkpoint"], description["immutable_core"]):
            if checkpoint is not None:
                for suffix in ("npz", "json"):
                    checkpoint_files[f"{checkpoint['stem']}.{suffix}"] = checkpoint[
                        f"{suffix}_file_sha256"
                    ]
    for cells in parameter_cells.values():
        for description in cells.values():
            checkpoint = description["checkpoint"]
            for suffix in ("npz", "json"):
                checkpoint_files[f"{checkpoint['stem']}.{suffix}"] = checkpoint[
                    f"{suffix}_file_sha256"
                ]
    files = {
        **raw_files,
        **parameter_source_files,
        **checkpoint_files,
        **{str(ROOT / name): value for name, value in sources.items()},
    }
    output.mkdir(parents=True, exist_ok=False)
    input_archive = output / "common-inputs.npz"
    np.savez_compressed(input_archive, **arrays)
    protocol = {
        "schema": SCHEMA,
        "role": "common-input controller service microprofile, not a flight or learner benchmark",
        "source_episode": str(episode.resolve()),
        "source_episode_physical_world_id": run.summary["physical_world_id"],
        "source_episode_config": run.binding["config"],
        "source_episode_artifact_sha256": raw_files,
        "resource_base": str(base.resolve()),
        "methods": methods,
        "parameter_cells": parameter_cells,
        "parameter_source_files_sha256": parameter_source_files,
        "cases": cases,
        "controller_call_inputs": input_signatures,
        "input_archive": str(input_archive.resolve()),
        "input_archive_sha256": file_digest(input_archive),
        "checkpoint_files_sha256": checkpoint_files,
        "source_sha256": sources,
        "files_sha256": files,
        "jax_version_at_cpu_seal": jax.__version__,
        "jax_x64_enabled": bool(jax.config.x64_enabled),
        "warmup_calls_per_cell": WARMUP_CALLS,
        "measured_calls_per_cell": REPETITIONS,
        "warmup_plan": balanced_plan(METHODS, tuple(common), WARMUP_CALLS, phase="warmup"),
        "measurement_plan": balanced_plan(METHODS, tuple(common), REPETITIONS, phase="measured"),
        "compilation_cache_directory": COMPILATION_CACHE,
        "cache_contract": (
            "reuse explicitly declared historical cache; record first calls, "
            "not guaranteed cold compilation"
        ),
        "common_input_provenance": (
            "state and current model authenticate against saved hashes; goal and previous index "
            "come from saved controls; obstacle/safety arrays are reconstructed from sealed "
            "geometry because original device bytes were not logged; exact CPU-prepared bytes "
            "are archived and transferred unchanged for all measured methods"
        ),
        "parameter_contract": (
            "time zero uses each complete startup checkpoint; time 2.4 uses each method's own "
            "authenticated published/control snapshot; common physical inputs remain those of F2; "
            "UNION and F2_2K immutable cores remain the startup F2 checkpoint"
        ),
        "measurement_contract": (
            "perf_counter around the complete packed controller call plus block_until_ready; "
            "inputs are prepared and synchronized first; output hashing and file writes "
            "occur after timing"
        ),
        "warmup_contract": (
            "three disposable calls per method and case; first calls may include schema, "
            "compilation or cache loading and are retained separately; shared executable "
            "caches are recorded"
        ),
        "comparison_scope": (
            "UNION32 versus F2_2K32 and F2_16 at identical common state/model/obstacles/goal/"
            "previous index; method parameters differ as bound; one GPU/process session, "
            "descriptive service costs only; "
            "no learner contention, physical command delay, throughput or flight-safety claim"
        ),
        "occupancy_contract": (
            "retain every foreign context; only an exactly named anydesk context with reported "
            "zero compute memory is allowed; every other foreign compute process blocks execution"
        ),
        "cpu_observation_contract": (
            "the root assigns the experiment slot; /proc/stat and per-PID stat snapshots bracket "
            "the measured window; report aggregate and own/nonself observed CPU deltas with "
            "new/disappeared processes, without claiming all system activity is absent"
        ),
        "state_contract": (
            "no learner calls, state updates, physical evolution, policy publications "
            "or outcome selection"
        ),
    }
    protocol = json.loads(json.dumps(protocol))
    envelope = {"protocol": protocol, "sha256": digest(protocol)}
    write_json(output / "protocol.json", envelope, exclusive=True)
    for name in sources:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    for name in raw_files:
        destination = output / "source-episode" / Path(name).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, destination)
    for method, description in methods.items():
        for kind in ("checkpoint", "immutable_core"):
            checkpoint = description[kind]
            if checkpoint is not None:
                destination = output / "checkpoints" / method / kind
                destination.mkdir(parents=True, exist_ok=True)
                for suffix in ("npz", "json"):
                    shutil.copy2(
                        f"{checkpoint['stem']}.{suffix}", destination / f"deployment.{suffix}"
                    )
    for label, cells in parameter_cells.items():
        for method, description in cells.items():
            destination = output / "parameter-cells" / label / method
            destination.mkdir(parents=True, exist_ok=True)
            for suffix in ("npz", "json"):
                shutil.copy2(
                    f"{description['checkpoint']['stem']}.{suffix}",
                    destination / f"deployment.{suffix}",
                )
    load_protocol(output / "protocol.json")
    return envelope


def run_timed_plan(
    plan: Sequence[dict[str, Any]],
    invoke: Callable[[str, str], Any],
    *,
    describe_result: Callable[[Any], dict[str, Any]],
    record: Callable[[dict[str, Any]], None],
    before_round: Callable[[int], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> list[dict[str, Any]]:
    """Time only the supplied synchronized operation; retain evidence outside its interval."""
    rows, prior_round = [], None
    for entry in plan:
        if entry["repetition"] != prior_round:
            if before_round is not None:
                before_round(entry["repetition"])
            prior_round = entry["repetition"]
        started = clock()
        result = invoke(entry["method"], entry["case"])
        completed = clock()
        elapsed = completed - started
        if not np.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("controller service interval must be finite and positive")
        row = {
            **entry,
            "started_wall_time": started,
            "completed_wall_time": completed,
            "service_seconds": elapsed,
            **describe_result(result),
        }
        record(row)
        rows.append(row)
    return rows


def parse_process_stat(line: str) -> dict[str, Any]:
    """Read only PID, comm, CPU counters and process start identity from Linux procfs."""
    prefix, remainder = line.split("(", 1)
    comm, suffix = remainder.rsplit(")", 1)
    fields = suffix.split()
    user, system = int(fields[11]), int(fields[12])
    return {
        "pid": int(prefix.strip()),
        "comm": comm,
        "start_time_ticks": int(fields[19]),
        "user_ticks": user,
        "system_ticks": system,
        "cpu_ticks": user + system,
    }


def cpu_snapshot(proc: Path = Path("/proc")) -> dict[str, Any]:
    """Observe aggregate and per-process CPU counters without recording command arguments."""
    started = time.perf_counter()
    aggregate = next(
        line for line in (proc / "stat").read_text().splitlines() if line.startswith("cpu ")
    )
    values = [int(value) for value in aggregate.split()[1:9]]
    if len(values) != 8:
        raise ValueError("Linux aggregate CPU stat requires eight counters")
    processes, skipped = [], []
    for entry in proc.iterdir():
        if entry.name.isdecimal():
            try:
                processes.append(parse_process_stat((entry / "stat").read_text()))
            except (OSError, ValueError, IndexError):
                skipped.append(int(entry.name))
    return {
        "monotonic_time_seconds": started,
        "unix_time_seconds": time.time(),
        "snapshot_completed_monotonic_seconds": time.perf_counter(),
        "clock_ticks_per_second": int(os.sysconf("SC_CLK_TCK")),
        "logical_cpu_count": os.cpu_count(),
        "uptime_seconds": float((proc / "uptime").read_text().split()[0]),
        # Linux already includes guest time in user/nice, so guest columns are not summed again.
        "aggregate_cpu_ticks": values,
        "processes": sorted(processes, key=lambda row: row["pid"]),
        "unreadable_or_exited_pids": sorted(skipped),
    }


def summarize_cpu_window(before: dict, after: dict, own_pid: int) -> dict[str, Any]:
    """Report CPU work observed across the whole measurement window with explicit coverage."""
    wall = after["monotonic_time_seconds"] - before["monotonic_time_seconds"]
    hz = before["clock_ticks_per_second"]
    if (
        wall <= 0
        or hz != after["clock_ticks_per_second"]
        or after["uptime_seconds"] < before["uptime_seconds"]
    ):
        raise ValueError("CPU observation window has incompatible or nonmonotonic clocks")
    delta = np.asarray(after["aggregate_cpu_ticks"]) - before["aggregate_cpu_ticks"]
    if np.any(delta < 0):
        raise ValueError("aggregate CPU counters decreased during the measured window")
    start = {(row["pid"], row["start_time_ticks"]): row for row in before["processes"]}
    end = {(row["pid"], row["start_time_ticks"]): row for row in after["processes"]}
    rows = []
    for key in start.keys() & end.keys():
        ticks = end[key]["cpu_ticks"] - start[key]["cpu_ticks"]
        if ticks < 0:
            raise ValueError("matched process CPU counter decreased")
        if ticks:
            rows.append(
                {
                    "pid": key[0],
                    "comm": end[key]["comm"],
                    "cpu_ticks": ticks,
                    "cpu_seconds": ticks / hz,
                    "percent_of_one_logical_cpu": 100 * ticks / hz / wall,
                }
            )
    rows.sort(key=lambda row: (-row["cpu_ticks"], row["pid"]))
    own = [row for row in rows if row["pid"] == own_pid]
    other = [row for row in rows if row["pid"] != own_pid]
    total = int(np.sum(delta))
    idle = int(delta[3] + delta[4])
    return {
        "started_monotonic_seconds": before["monotonic_time_seconds"],
        "completed_monotonic_seconds": after["monotonic_time_seconds"],
        "started_unix_time_seconds": before["unix_time_seconds"],
        "completed_unix_time_seconds": after["unix_time_seconds"],
        "wall_seconds": wall,
        "aggregate_cpu_seconds": total / hz,
        "aggregate_busy_cpu_seconds": (total - idle) / hz,
        "aggregate_busy_fraction": (total - idle) / total if total else None,
        "own_pid": own_pid,
        "own_cpu_seconds": sum(row["cpu_seconds"] for row in own),
        "nonself_matched_process_cpu_seconds": sum(row["cpu_seconds"] for row in other),
        "own_process_activity": own,
        "nonself_process_activity": other,
        "newly_observed_processes": [end[key] for key in sorted(end.keys() - start.keys())],
        "disappeared_processes": [start[key] for key in sorted(start.keys() - end.keys())],
        "coverage": (
            "process deltas cover matching PID/start-time pairs only; newly observed, vanished "
            "or unreadable processes may have consumed additional CPU; aggregate counters cover "
            "the system; the window includes audits and telemetry between timed controller calls"
        ),
    }


def sample_statistics(values: Sequence[float]) -> dict[str, Any]:
    """Describe all measured samples without dropping slow or nonfinite observations."""
    samples = np.asarray(values, dtype=float)
    if not samples.size or not np.all(np.isfinite(samples)) or np.any(samples <= 0):
        raise ValueError("service samples must be nonempty, finite and positive")
    return {
        "count": int(samples.size),
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "p95": float(np.percentile(samples, 95)),
        "maximum": float(np.max(samples)),
        "minimum": float(np.min(samples)),
    }


def analyze_samples(samples: Sequence[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    """Verify the complete bound plan and compare methods within each common-input case."""
    plan = protocol["measurement_plan"]
    if len(samples) != len(plan):
        raise ValueError("incomplete controller-cost measurements; preserve this attempt")
    for sample, expected in zip(samples, plan, strict=True):
        if any(sample.get(key) != value for key, value in expected.items()):
            raise ValueError("measured call identity/order differs from the sealed plan")
    groups, comparisons = [], []
    by_cell = {}
    for case in protocol["cases"]:
        label = case["case"]
        for method, description in protocol["methods"].items():
            selected = [row for row in samples if row["case"] == label and row["method"] == method]
            if len(selected) != protocol["measured_calls_per_cell"]:
                raise ValueError("method/case measurement denominator differs from protocol")
            by_cell[label, method] = {row["repetition"]: row["service_seconds"] for row in selected}
            if len(by_cell[label, method]) != len(selected):
                raise ValueError("duplicate method/case/repetition timing sample")
            groups.append(
                {
                    "case": label,
                    "time_seconds": case["time_seconds"],
                    "method": method,
                    "total_fallback_count": description["total_fallback_count"],
                    "service_seconds": sample_statistics(
                        [row["service_seconds"] for row in selected]
                    ),
                    "method_position_counts": dict(
                        sorted(Counter(row["method_position"] for row in selected).items())
                    ),
                    "distinct_output_hashes": len({row.get("output_sha256") for row in selected}),
                    "execution_mode_counts": dict(
                        Counter(str(row.get("execution_mode")) for row in selected)
                    ),
                }
            )
        for numerator, denominator in (
            ("UNION", "F2_2K"),
            ("UNION", "F2"),
            ("F2_2K", "F2"),
            ("A_BAL", "F2"),
            ("PD_F", "F2"),
        ):
            if numerator not in protocol["methods"] or denominator not in protocol["methods"]:
                continue
            left, right = by_cell[label, numerator], by_cell[label, denominator]
            ratios = [left[index] / right[index] for index in sorted(left)]
            comparisons.append(
                {
                    "case": label,
                    "numerator": numerator,
                    "denominator": denominator,
                    "paired_repetition_count": len(ratios),
                    "ratio_of_mean_service": float(
                        np.mean(list(left.values())) / np.mean(list(right.values()))
                    ),
                    "paired_service_ratio": sample_statistics(ratios),
                }
            )
    return {
        "status": "completed",
        "measured_call_count": len(samples),
        "groups": groups,
        "comparisons": comparisons,
        "comparison_scope": protocol["comparison_scope"],
    }


def _run_profile(protocol_path: Path, output: Path) -> dict[str, Any]:
    """Execute the reviewed protocol only in an explicitly assigned GPU slot."""
    protocol = load_protocol(protocol_path)
    if os.environ.get("JAX_COMPILATION_CACHE_DIR") != protocol["compilation_cache_directory"]:
        raise RuntimeError(
            "set JAX_COMPILATION_CACHE_DIR to the explicitly sealed historical cache"
        )
    import jax

    from benchmark.da_plcbf_actuator_diagnostics import DiagnosticResources
    from crazyflow.safety.da_plcbf.actuator_experiment import _hash_tree
    from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig

    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol_path, output / "protocol.json")
    began = time.perf_counter()

    def occupancy(phase: str, repetition: int) -> None:
        observation = gpu_snapshot()
        write_json(output / f"gpu-{phase}-{repetition:02d}.json", observation, exclusive=True)
        if observation["blocking_foreign_compute_processes"]:
            raise RuntimeError(
                "foreign GPU compute process visible; measured controller work blocked"
            )

    occupancy("before", 0)
    if jax.default_backend() != "gpu" or any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("controller cost execution requires the assigned GPU backend")
    if bool(jax.config.x64_enabled) != protocol["jax_x64_enabled"]:
        raise ValueError("GPU precision context differs from bound CPU input descriptions")
    source_run, templates, common, _arrays = _recorded_cases(Path(protocol["source_episode"]))
    cases = protocol["cases"]
    with np.load(protocol["input_archive"], allow_pickle=False) as archive:
        saved_arrays = {name: archive[name] for name in archive.files}
    for template, expected in zip(templates, cases, strict=True):
        if any(
            template[key] != value
            for key, value in expected.items()
            if key != "common_input_description"
        ):
            raise ValueError("saved common input boundary identity changed")
        label = expected["case"]
        common[label] = restore_tree(
            common[label], expected["common_input_description"], saved_arrays
        )
        restored = describe_tree(common[label], label, {})
        if restored != expected["common_input_description"]:
            raise ValueError("common input restoration differs from sealed array descriptions")
    resources = DiagnosticResources(Path(protocol["resource_base"]))
    for method, description in protocol["methods"].items():
        resources.paths[method] = Path(description["checkpoint"]["stem"])
    descriptions, resolved = _resolve_methods(
        resources, ActuatorFilterConfig(**cases[0]["filter_config"])
    )
    if descriptions != protocol["methods"]:
        raise ValueError(
            "resolved checkpoint, core, controller or startup state differs from protocol"
        )
    parameter_cells, cell_bundles, _parameter_files = _parameter_cells(resources, source_run)
    if parameter_cells != protocol["parameter_cells"]:
        raise ValueError("per-case published parameter snapshot or authentication changed")
    arguments, signatures = {}, {}
    for case in cases:
        label = case["case"]
        signatures[label] = {}
        for method in resolved:
            bundle = cell_bundles[method, label]
            args = jax.block_until_ready(_controller_args(common[label], bundle.state.params))
            arguments[method, label] = args
            signatures[label][method] = describe_tree(args, f"{label}.{method}", {})
    if signatures != protocol["controller_call_inputs"]:
        raise ValueError("full controller call inputs differ from bound values/dtypes/weak types")
    write_json(
        output / "resolved-inputs.json",
        {
            "cases": cases,
            "methods": descriptions,
            "parameter_cells": parameter_cells,
            "signatures": signatures,
            "input_transfer": "exact sealed NPZ leaves, prepared before timing",
        },
        exclusive=True,
    )
    write_json(
        output / "environment.json",
        {
            "jax_version": jax.__version__,
            "jax_x64_enabled": bool(jax.config.x64_enabled),
            "compilation_cache_environment": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
            "compilation_cache_resolved": str(jax.config.jax_compilation_cache_dir),
            "cache_contract": protocol["cache_contract"],
            "setup_seconds": time.perf_counter() - began,
            "controller_object_groups": {
                method: [other for other in METHODS if resolved[other][1] is controller]
                for method, (_bundle, controller) in resolved.items()
            },
            "controller_cache_entries_before_warmup": {
                method: controller.cache_size for method, (_bundle, controller) in resolved.items()
            },
        },
        exclusive=True,
    )

    def invoke(method: str, case: str) -> Any:
        return jax.block_until_ready(resolved[method][1](*arguments[method, case]))

    def describe_result(result: Any) -> dict[str, Any]:
        def number(value: Any) -> float | None:
            value = float(value)
            return value if np.isfinite(value) else None

        return {
            "output_sha256": _hash_tree(result),
            "execution_mode": int(result.execution_mode),
            "selected_index": int(result.selected_index),
            "qp_valid": bool(result.qp_valid),
            "sqp_iterations": int(result.sqp_iterations),
            "eligible_policy_count": int(np.sum(result.certificates.eligible)),
            "action": [number(value) for value in np.asarray(result.action)],
            "action_finite": bool(np.all(np.isfinite(result.action))),
            "qp_rejection_flags": np.asarray(result.qp_rejection_flags).tolist(),
            "applied_passed": bool(result.applied.passed),
            "qp_held_passed": bool(result.qp_check.passed),
            "fallback_valid": bool(result.fallback_valid),
            "emergency_valid": bool(result.emergency_valid),
            "executed_policy_residual": number(result.policy_residual),
            "qp_primal_residual": number(result.qp.primal_residual),
            "qp_stationarity_residual": number(result.qp.stationarity_residual),
            "qp_complementarity_residual": number(result.qp.complementarity_residual),
        }

    def recorder(path: Path) -> Callable[[dict[str, Any]], None]:
        path.touch(exist_ok=False)

        def save(row: dict[str, Any]) -> None:
            with path.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

        return save

    warmups = run_timed_plan(
        protocol["warmup_plan"],
        invoke,
        describe_result=describe_result,
        record=recorder(output / "warmup-samples.jsonl"),
        before_round=lambda repeat: occupancy("warmup", repeat),
    )
    write_json(
        output / "warmup.json",
        {
            "contract": protocol["warmup_contract"],
            "samples": warmups,
            "first_call_samples": [row for row in warmups if row["repetition"] == 0],
            "steady_warmup_samples": [row for row in warmups if row["repetition"] > 0],
            "controller_cache_entries_after_warmup": {
                method: controller.cache_size for method, (_bundle, controller) in resolved.items()
            },
        },
        exclusive=True,
    )
    load_protocol(protocol_path)
    after_warmup, _unused = _resolve_methods(
        resources, ActuatorFilterConfig(**cases[0]["filter_config"])
    )
    if after_warmup != descriptions:
        raise ValueError("disposable warmup changed a bound startup learner or immutable core")
    expected_cell_states = {
        (method, label): cell["checkpoint"]
        for label, cells in parameter_cells.items()
        for method, cell in cells.items()
    }

    def check_cell_states() -> None:
        actual = {
            identity: _checkpoint_description(bundle) for identity, bundle in cell_bundles.items()
        }
        if actual != expected_cell_states:
            raise ValueError("profiling changed a complete per-case published learner state")

    check_cell_states()
    cpu_before = cpu_snapshot()
    write_json(output / "cpu-before.json", cpu_before, exclusive=True)
    try:
        samples = run_timed_plan(
            protocol["measurement_plan"],
            invoke,
            describe_result=describe_result,
            record=recorder(output / "measured-samples.jsonl"),
            before_round=lambda repeat: occupancy("measured", repeat),
        )
    finally:
        cpu_after = cpu_snapshot()
        write_json(output / "cpu-after.json", cpu_after, exclusive=True)
        cpu_window = summarize_cpu_window(cpu_before, cpu_after, os.getpid())
        write_json(output / "cpu-window.json", cpu_window, exclusive=True)
    after, _unused = _resolve_methods(resources, ActuatorFilterConfig(**cases[0]["filter_config"]))
    if after != descriptions:
        raise ValueError("controller profiling changed a startup learner or immutable core")
    check_cell_states()
    for label in common:
        if describe_tree(common[label], label, {}) != next(
            case["common_input_description"] for case in cases if case["case"] == label
        ):
            raise ValueError("profiling changed a sealed common controller input")
    occupancy("after", 0)
    load_protocol(protocol_path)
    report = analyze_samples(samples, protocol)
    report.update(
        protocol_content_sha256=digest(protocol),
        common_input_provenance=protocol["common_input_provenance"],
        parameter_contract=protocol["parameter_contract"],
        startup_complete_states_unchanged=True,
        immutable_core_states_unchanged=True,
        per_case_complete_published_states_unchanged=True,
        learner_calls=0,
        flight_calls=0,
        warmup_call_count=len(warmups),
        wall_seconds_including_setup_warmup_audits=time.perf_counter() - began,
        first_call_samples=[row for row in warmups if row["repetition"] == 0],
        cpu_window=cpu_window,
        first_measured_call_started_wall_time=samples[0]["started_wall_time"],
        last_measured_call_completed_wall_time=samples[-1]["completed_wall_time"],
    )
    write_json(output / "report.json", report, exclusive=True)
    write_report(output / "report.md", report)
    return report


def run_profile(protocol_path: Path, output: Path) -> dict[str, Any]:
    """Retain any failed attempt with its completed call counts; never overwrite an attempt."""
    existed = output.exists()
    try:
        return _run_profile(protocol_path, output)
    except (Exception, KeyboardInterrupt) as error:
        if not existed and output.is_dir():
            counts = {}
            for phase in ("warmup", "measured"):
                path = output / f"{phase}-samples.jsonl"
                counts[phase] = len(path.read_text().splitlines()) if path.exists() else 0
            write_json(
                output / "failure.json",
                {
                    "status": "failed" if isinstance(error, Exception) else "interrupted",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "completed_call_counts": counts,
                    "policy": "attempt retained; no automatic replacement or rerun",
                },
                exclusive=True,
            )
        raise


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Present complete synchronized cost and equal-size comparisons without flight claims."""
    lines = [
        report["comparison_scope"],
        "",
        report["common_input_provenance"],
        "",
        report["parameter_contract"],
        "",
        "| Saved input | Method | Fallbacks | Calls | Median ms | p95 ms | Max ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["groups"]:
        stats = row["service_seconds"]
        lines.append(
            f"| {row['case']} | {row['method']} | {row['total_fallback_count']} | "
            f"{stats['count']} | {1000 * stats['median']:.3f} | {1000 * stats['p95']:.3f} | "
            f"{1000 * stats['maximum']:.3f} |"
        )
    lines.extend(
        ["", "| Saved input | Cost ratio | Ratio of means | Paired median |", "|---|---|---:|---:|"]
    )
    for row in report["comparisons"]:
        lines.append(
            f"| {row['case']} | {row['numerator']} / {row['denominator']} | "
            f"{row['ratio_of_mean_service']:.3f} | {row['paired_service_ratio']['median']:.3f} |"
        )
    lines.extend(
        [
            "",
            "All 300 measured calls follow the sealed rotating/reversing order. "
            "Thirty disposable warmup calls, including each cell's first call, "
            "are retained separately. Parameters, optimizer states, immutable cores and "
            "common controller inputs remain unchanged. Raw timing samples, decisions, "
            "output hashes and GPU occupancy observations are retained.",
        ]
    )
    with path.open("x") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> None:
    """Separate reviewed CPU protocol preparation from root-scheduled GPU execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--output", type=Path, default=PROTOCOL.parent)
    seal.add_argument("--episode", type=Path, default=EPISODE)
    seal.add_argument("--base", type=Path, default=BASE)
    run = commands.add_parser("run")
    run.add_argument("--protocol", type=Path, default=PROTOCOL)
    run.add_argument("--output", type=Path, default=BASE / "controller-cost-measured-v1")
    args = parser.parse_args()
    if args.command == "seal":
        sealed = seal_protocol(args.output, episode=args.episode, base=args.base)
        print(json.dumps({"sha256": sealed["sha256"], "output": str(args.output)}))
    else:
        report = run_profile(args.protocol, args.output)
        print(
            json.dumps(
                {"measured_calls": report["measured_call_count"], "output": str(args.output)}
            )
        )


if __name__ == "__main__":
    main()
