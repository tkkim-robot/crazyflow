"""Seal, execute and audit eight dedicated actuator runtime diagnostic flights.

The paced and asynchronous modes have different physical command-delay contracts.
Their outcomes describe those complete systems; they do not isolate scheduling
alone. Every method starts from the same predeclared checkpoint on each flight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/da_plcbf/actuator-diagnostics-20260906/v1"
DESIGN = ROOT / "artifacts/da_plcbf/actuator-diagnostic-20260906/v1/protocol-sealed-v1.json"
PROTOCOL = BASE / "runtime-protocol-v3/protocol.json"
SCHEMA = "actuator_dedicated_runtime_v1"
METHODS = ("F2", "A_BAL", "PD_F", "PD_A")
MODES = ("paced", "asynchronous")


def digest(value: Any) -> str:
    """Hash exact canonical JSON content without accepting nonfinite numbers."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    """Hash retained file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    """Write already normalized evidence, with optional overwrite refusal."""
    with path.open("x" if exclusive else "w") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def statistics(values: Any) -> dict[str, Any]:
    """Keep the sample denominator and nonfinite count visible."""
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(values.size - finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "p95": float(np.percentile(finite, 95)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def prescribed_center_crossings(physical: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive the first positive harmonic zero crossing of each moving obstacle."""
    rows = []
    for index, (amplitude, omega, phase) in enumerate(
        zip(
            physical["obstacle_amplitudes"],
            physical["obstacle_angular_frequencies"],
            physical["obstacle_phases"],
            strict=True,
        )
    ):
        if np.linalg.norm(amplitude) == 0 or omega == 0:
            continue
        if omega < 0:
            raise ValueError("sealed incoming-obstacle frequencies must be positive")
        # The structured scene uses a scalar harmonic phase per obstacle.
        k = int(np.floor(float(phase) / np.pi)) + 1
        crossing = (k * np.pi - float(phase)) / float(omega)
        if not 0 < crossing <= physical["duration_seconds"]:
            raise ValueError("moving obstacle has no first positive crossing in this episode")
        rows.append({"obstacle_index": index, "time_seconds": crossing})
    if not rows:
        raise ValueError("runtime anchor must contain moving incoming obstacles")
    return sorted(rows, key=lambda row: row["time_seconds"])


def seal_protocol(output: Path, *, base: Path = BASE, design: Path = DESIGN) -> dict[str, Any]:
    """Resolve a CPU-only design and checkpoint/source identities before runtime outcomes."""
    from benchmark.da_plcbf_actuator_diagnostic_protocol import load_sealed
    from benchmark.da_plcbf_actuator_diagnostics import (
        DiagnosticResources,
        resolve_trial_scene,
        source_binding,
    )
    from crazyflow.safety.da_plcbf.actuator_experiment import ActuatorEpisodeConfig, _hash_tree
    from crazyflow.safety.da_plcbf.actuator_learning import actuator_reference_fingerprint
    from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
    from crazyflow.safety.da_plcbf.actuator_study import ActuatorObservationConfig

    proposal = load_sealed(design)
    anchor = next(
        trial
        for trial in proposal["trials"]
        if trial["world_key"] == "structured_30101"
        and trial["cell_id"] == "eta0.7_lag1_extra0"
        and trial["realization"] == "primary"
        and trial["arm"]["arm"] == "F2"
    )
    configuration = dict(proposal["shared_control"])
    filters = ActuatorFilterConfig(**configuration.pop("filter_config"))
    observation = ActuatorObservationConfig(**configuration.pop("observation_config", {}))
    configuration.update(
        method="F2",
        execution_mode="paced",
        save_checkpoints=True,
        retain_rollouts=False,
        capture_times=(),
        freeze_learning_at=None,
    )
    config = ActuatorEpisodeConfig(
        filter_config=filters, observation_config=observation, **configuration
    )
    scene = resolve_trial_scene(proposal, anchor, filters)
    resources = DiagnosticResources(base)
    checkpoints, files = {}, {}
    for method in METHODS:
        bundle = resources.bundle(method)
        checkpoints[method] = {
            "stem": str(resources.paths[method].resolve()),
            "complete_state_sha256": _hash_tree(bundle.state),
            "params_sha256": _hash_tree(bundle.state.params),
            "optimizer_sha256": _hash_tree(bundle.state.optimizer_state),
            "latest_model_sha256": _hash_tree(bundle.state.latest_dynamics_estimate),
            "initial_version": int(bundle.state.library_version),
            "initial_gradient_steps": int(bundle.state.cumulative_gradient_steps),
            "fallback_policy_count": int(bundle.contract.spec.latent_codes.shape[0]),
            "candidate_count_including_nominal": int(bundle.contract.spec.latent_codes.shape[0])
            + 1,
            "reference_sha256": actuator_reference_fingerprint(bundle.contract),
            "actor_config": asdict(bundle.config),
            "objective_config": asdict(bundle.contract.learning_config),
        }
        config.validate(scene, bundle)
        for path in (bundle.npz_path, bundle.json_path):
            files[str(path.resolve())] = file_digest(path)
    for frozen, adaptive in (("F2", "A_BAL"), ("PD_F", "PD_A")):
        if (
            checkpoints[frozen]["complete_state_sha256"]
            != checkpoints[adaptive]["complete_state_sha256"]
        ):
            raise ValueError("frozen/adaptive comparator startup states differ")
    source = source_binding()
    source[str(Path(__file__).resolve().relative_to(ROOT))] = file_digest(Path(__file__))
    physical = scene.physical_spec()
    body = {
        "schema": SCHEMA,
        "selected_before_runtime_outcomes": True,
        "role": "single historical development geometry; no held-out or population claim",
        "parent_design": str(design.resolve()),
        "parent_design_file_sha256": file_digest(design),
        "anchor_trial": anchor,
        "physical_spec": physical,
        "physical_spec_sha256": digest(physical),
        "world_metadata": scene.metadata(),
        "duration_seconds": float(scene.world.config.duration_seconds),
        "fault_time_seconds": 2.0,
        "prescribed_center_crossings": prescribed_center_crossings(physical),
        "center_crossing_definition": (
            "first positive zero crossing of each prescribed incoming obstacle harmonic, "
            "derived from sealed geometry; obstacle-center crossing, not first threat, "
            "first contact, or a needed-by guarantee; contact can occur earlier"
        ),
        "timing_landmarks": [
            {"label": "fault_onset", "time_seconds": 2.0},
            {"label": "legacy_deterministic_first_action_divergence", "time_seconds": 2.40},
            {"label": "balanced_deterministic_first_action_divergence", "time_seconds": 2.48},
        ],
        "timing_landmark_scope": (
            "fault onset is exogenous; 2.40 and 2.48 seconds are previously observed "
            "deterministic development action-divergence landmarks supplied before runtime "
            "outcomes; they are descriptive reference cuts, not collision/necessity deadlines"
        ),
        "shared_episode_config": asdict(config),
        "shared_episode_config_sha256": digest(asdict(config)),
        "checkpoints": checkpoints,
        "checkpoint_files_sha256": files,
        "source_sha256": source,
        "ordered_flights": [
            {"flight_id": f"{method}-{mode}", "method": method, "execution_mode": mode}
            for method in METHODS
            for mode in MODES
        ],
        "warmup": {
            "calls_per_flight": config.warmup_calls,
            "rule": "warm the applicable controller and learner disposably before every flight",
            "persistent_startup_state": "reload same bound complete checkpoint; no warmup credit",
        },
        "timing_contracts": {
            "paced": (
                "serialized measured controller/learner service with unchanged reserve gate; "
                "fixed nondelayed physical command holds; wall lateness reported"
            ),
            "asynchronous": (
                "real background learner, wall-paced physical sensing, delayed command "
                "application; simulator/audit overhead consumes physical wall budget; "
                "unchanged reserve setting is recorded but no serial admission gate applies"
            ),
        },
        "primary_observations": [
            "finite online snapshots published during flight and before declared timing landmarks",
            "postfault-trained snapshots published before each declared timing landmark",
            "full physical duration, collision, operational limits and safe task completion",
            "controller misses/lateness, actual command delay/holds and snapshot/model ages",
            "real learner/controller host overlap and service under observed contention",
        ],
        "publication_policy": (
            "publish every finite completed result at the next actual sensing boundary; "
            "a result pending at terminal time is not publication; "
            "late completion has no flight credit"
        ),
        "comparison_scope": (
            "compare frozen/adaptive arms within each execution mode; cross-mode comparisons "
            "include different physical-delay semantics, not pure scheduling superiority; "
            "one flight per cell, no hard real-time or concurrent-GPU-kernel claim"
        ),
        "gpu_execution": (
            "root assigns an exclusive experiment slot; refuse competing foreign compute "
            "processes immediately before execution. Preserve the existing AnyDesk desktop "
            "service when it reports exactly zero compute memory; this exception is not proof "
            "of total device inactivity or absence of desktop overhead"
        ),
        "attempt_policy": "retain every attempt; never replace an unfavorable or incomplete record",
        "amendment": {
            "previous_protocol": str(BASE / "runtime-protocol-v2/protocol.json"),
            "previous_protocol_file_sha256": file_digest(
                BASE / "runtime-protocol-v2/protocol.json"
            ),
            "reason": (
                "v2 preflight observed existing AnyDesk with zero compute memory; retain and "
                "disclose that context while blocking competing experiment processes. "
                "No flight has run; prior protocols and attempts remain retained"
            ),
        },
    }
    # Normalize tuples and NumPy-free metadata before hashing the immutable envelope.
    body = json.loads(json.dumps(body))
    envelope = {"protocol": body, "sha256": digest(body)}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", envelope, exclusive=True)
    for name in source:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    return envelope


def load_protocol(path: Path, *, verify_files: bool = True) -> dict[str, Any]:
    """Authenticate the design and optionally the exact live execution inputs."""
    envelope = json.loads(path.read_text())
    body = envelope["protocol"]
    if body.get("schema") != SCHEMA or digest(body) != envelope.get("sha256"):
        raise ValueError("runtime protocol schema or content digest changed")
    if verify_files:
        paths = {
            **body["checkpoint_files_sha256"],
            **{str(ROOT / name): sha for name, sha in body["source_sha256"].items()},
            body["parent_design"]: body["parent_design_file_sha256"],
        }
        for name, expected in paths.items():
            if file_digest(Path(name)) != expected:
                raise ValueError(f"bound runtime input changed: {name}")
    return body


def classify_compute_processes(lines: list[str], own_pid: int) -> dict[str, list[str]]:
    """Retain every foreign context while distinguishing a declared desktop exception."""
    foreign, desktop, blocking = [], [], []
    for line in lines:
        fields = [field.strip() for field in line.split(",")]
        if fields[0] == str(own_pid):
            continue
        foreign.append(line)
        if len(fields) == 3 and Path(fields[1]).name == "anydesk" and fields[2] == "0":
            desktop.append(line)
        else:
            blocking.append(line)
    return {
        "foreign_compute_processes": foreign,
        "declared_zero_memory_desktop_contexts": desktop,
        "blocking_foreign_compute_processes": blocking,
    }


def gpu_snapshot() -> dict[str, Any]:
    """Record hardware and compute-process occupancy without synthetic timing."""
    import jax

    result = {
        "pid": os.getpid(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "JAX_COMPILATION_CACHE_DIR",
                "XLA_FLAGS",
            )
        },
    }
    for key, query in (
        (
            "gpu",
            "--query-gpu=name,uuid,driver_version,pstate,temperature.gpu,utilization.gpu,memory.used",
        ),
        ("compute_processes", "--query-compute-apps=pid,process_name,used_gpu_memory"),
    ):
        completed = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"cannot observe uncontended GPU occupancy: {completed.stderr}")
        result[key] = completed.stdout.strip().splitlines()
    result.update(classify_compute_processes(result["compute_processes"], os.getpid()))
    if not all(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("measured runtime campaign requires actual GPU execution")
    return result


def analyze_episode(directory: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Derive timing/outcome observations from actual saved commands and publications."""
    summary = json.loads((directory / "summary.json").read_text())
    binding = json.loads((directory / "binding.json").read_text())
    updates = json.loads((directory / "updates.json").read_text())
    warmup = json.loads((directory / "warmup.json").read_text())
    with np.load(directory / "controls.npz", allow_pickle=False) as saved:
        controls = {key: saved[key] for key in saved.files}
    with np.load(directory / "applications.npz", allow_pickle=False) as saved:
        applications = {key: saved[key] for key in saved.files}
    method, mode = summary["method"], summary["execution_mode"]
    expected = protocol["checkpoints"][method]
    if binding["initial_learner_sha256"] != expected["complete_state_sha256"]:
        raise ValueError("measured startup state differs from prepared complete checkpoint")
    if digest(binding["scene"]["physical_spec"]) != protocol["physical_spec_sha256"]:
        raise ValueError("measured physical world differs from the sealed runtime world")
    config = dict(binding["config"])
    config.update(method="F2", execution_mode="paced")
    if config != protocol["shared_episode_config"]:
        raise ValueError("measured control settings differ from the shared runtime contract")
    publications = summary["snapshot_publications"]
    by_version = {
        row["computed_version"]: row
        for row in updates
        if row["completion_credited"] and row["finite_update_applied"]
    }
    last_version = expected["initial_version"]
    for row in publications:
        update = by_version.get(row["version"])
        if update is None or row["completed_wall_time"] > row["published_wall_time"]:
            raise ValueError("publication lacks an actually completed finite credited update")
        if row["version"] <= last_version:
            raise ValueError("published versions must advance once without duplicates")
        if any(
            row[key] != update[key] for key in ("completed_wall_time", "training_simulation_time")
        ):
            raise ValueError("publication completion/training time differs from its actual update")
        if row["published_simulation_time"] > summary["physical_time_seconds"]:
            raise ValueError("publication was credited after the physical terminal time")
        last_version = row["version"]
    center_crossing_counts = []
    for threat in protocol["prescribed_center_crossings"]:
        cutoff = threat["time_seconds"]
        before = [row for row in publications if row["published_simulation_time"] < cutoff]
        center_crossing_counts.append(
            {
                **threat,
                "physical_prefix_reached_center_crossing": summary["physical_time_seconds"]
                >= cutoff,
                "published_before_center_crossing": len(before),
                "postfault_published_before_center_crossing": sum(
                    row["published_simulation_time"] >= protocol["fault_time_seconds"]
                    for row in before
                ),
                "postfault_trained_published_before_center_crossing": sum(
                    row["training_simulation_time"] >= protocol["fault_time_seconds"]
                    for row in before
                ),
            }
        )
    times = controls.get("time", np.asarray([]))
    landmark_counts = []
    for landmark in protocol.get("timing_landmarks", []):
        before = [
            row
            for row in publications
            if row["published_simulation_time"] < landmark["time_seconds"]
        ]
        landmark_counts.append(
            {
                **landmark,
                "physical_prefix_reached_landmark": summary["physical_time_seconds"]
                >= landmark["time_seconds"],
                "published_before_landmark": len(before),
                "postfault_trained_published_before_landmark": sum(
                    row["training_simulation_time"] >= protocol["fault_time_seconds"]
                    for row in before
                ),
            }
        )
    model_ages, stale = [], []
    for index, when in enumerate(times):
        version = int(controls["library_version"][index])
        update = by_version.get(version)
        training_time = 0.0 if update is None else update["training_simulation_time"]
        model_time = max(
            0.0, training_time - binding["config"]["observation_config"]["parameter_delay_seconds"]
        )
        model_hash = (
            expected["latest_model_sha256"] if update is None else update["estimated_model_sha256"]
        )
        model_ages.append(float(when - model_time))
        stale.append(model_hash != str(controls["estimated_model_sha256"][index]))
    applied = controls.get("command_applied", np.asarray([], dtype=bool)).astype(bool)
    delays = controls.get("command_applied_at", np.asarray([]))[applied] - times[applied]
    holds = np.diff(np.append(applications["time"], summary["physical_time_seconds"]))
    if np.any(holds < -1e-9) or np.any(delays < -1e-9):
        raise ValueError("actual commands violate causal application/hold ordering")
    return {
        "method": method,
        "execution_mode": mode,
        "episode_directory": str(directory.resolve()),
        "status": summary["status"],
        "termination": summary["termination"],
        "full_episode_completed": summary["full_episode_completed"],
        "physical_time_seconds": summary["physical_time_seconds"],
        "collision": summary["modeled_collider_collision"],
        "operational_all_nodes_pass": summary["actual_operational_all_nodes_pass"],
        "safe_task_completion": summary["successful_full_episode"],
        "waypoints_completed": summary["waypoints_completed"],
        "waypoints_total": summary["waypoints_total"],
        "published_online_updates": len(publications),
        "published_before_fault": sum(
            row["published_simulation_time"] < protocol["fault_time_seconds"]
            for row in publications
        ),
        "center_crossing_counts": center_crossing_counts,
        "landmark_counts": landmark_counts,
        "postfault_adaptation_available_before_center_crossing_and_safe_task": bool(
            summary["successful_full_episode"]
            and center_crossing_counts[0]["postfault_trained_published_before_center_crossing"] > 0
        ),
        "credited_finite_completed_updates": summary["finite_credited_updates"],
        "uncredited_finite_completed_updates": summary["finite_uncredited_updates"],
        "completed_unpublished_version": summary["final_pending_library_version"],
        "learner_calls": summary["learner_calls"],
        "controller_calls": summary["control_count"],
        "controller_deadline_misses": summary["controller_deadline_misses"],
        "controller_grid_deadline_misses": summary.get("controller_grid_deadline_misses"),
        "learner_deadline_misses": summary["learner_deadline_misses"],
        "skipped_sensing_ticks": summary["skipped_sensing_ticks"],
        "controller_seconds": summary["controller_seconds"],
        "learner_seconds": statistics([row["service_seconds"] for row in updates]),
        "gradient_norm": statistics(
            [row["gradient_norm"] for row in updates if row.get("gradient_norm") is not None]
        ),
        "parameter_update_norm": statistics(
            [
                row["parameter_update_norm"]
                for row in updates
                if row.get("parameter_update_norm") is not None
            ]
        ),
        "snapshot_age_seconds": statistics(controls.get("snapshot_age_seconds", [])),
        "model_observation_age_seconds": statistics(model_ages),
        "controls_with_training_model_mismatch": int(np.sum(stale)),
        "boundary_lateness_seconds": statistics(controls.get("boundary_lateness_seconds", [])),
        "actual_sensing_to_command_application_seconds": statistics(delays),
        "actual_command_hold_seconds": statistics(holds),
        "application_count_including_initial": int(len(applications["time"])),
        "execution_wall_seconds": summary["execution_wall_seconds"],
        "simulator_and_audit_seconds": summary["simulator_and_audit_seconds"],
        "asynchronous_learner": summary.get("asynchronous_learner"),
        "warmup": warmup,
        "complete_startup_state_sha256": binding["initial_learner_sha256"],
        "timing_contract": summary["timing_contract"],
        "error": summary["error"],
    }


def run_campaign(protocol_path: Path, output: Path) -> dict[str, Any]:
    """Run all eight declared full flights serially in an assigned uncontended GPU slot."""
    from benchmark.da_plcbf_actuator_diagnostic_protocol import load_sealed
    from benchmark.da_plcbf_actuator_diagnostics import DiagnosticResources, resolve_trial_scene
    from crazyflow.safety.da_plcbf.actuator_experiment import (
        ActuatorEpisodeConfig,
        run_actuator_episode,
    )
    from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
    from crazyflow.safety.da_plcbf.actuator_study import ActuatorObservationConfig

    protocol = load_protocol(protocol_path)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol_path, output / "protocol.json")
    write_json(
        output / "binding.json",
        {
            "protocol_path": str(protocol_path.resolve()),
            "protocol_file_sha256": file_digest(protocol_path),
            "protocol_content_sha256": digest(protocol),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        exclusive=True,
    )
    resources = DiagnosticResources()
    for method in METHODS:
        resources.paths[method] = Path(protocol["checkpoints"][method]["stem"])
    proposal = load_sealed(Path(protocol["parent_design"]))
    rows = []
    began = time.perf_counter()
    for flight in protocol["ordered_flights"]:
        load_protocol(protocol_path)
        trial_output = output / flight["flight_id"]
        trial_output.mkdir()
        occupancy = gpu_snapshot()
        write_json(trial_output / "gpu-before.json", occupancy, exclusive=True)
        if occupancy["blocking_foreign_compute_processes"]:
            raise RuntimeError("foreign GPU compute process visible; no measured flight launched")
        values = dict(protocol["shared_episode_config"])
        filters = ActuatorFilterConfig(**values.pop("filter_config"))
        observation = ActuatorObservationConfig(**values.pop("observation_config"))
        values.update(method=flight["method"], execution_mode=flight["execution_mode"])
        values["capture_times"] = tuple(values["capture_times"])
        config = ActuatorEpisodeConfig(
            filter_config=filters, observation_config=observation, **values
        )
        scene = resolve_trial_scene(proposal, protocol["anchor_trial"], filters)
        bundle, controller, learner, metadata = resources.resolve(flight["method"], filters)
        write_json(trial_output / "resolved_method.json", metadata, exclusive=True)
        print(json.dumps({"starting": flight, "completed": len(rows), "planned": 8}), flush=True)
        result = run_actuator_episode(
            scene,
            bundle,
            config,
            trial_output / "attempt-00",
            controllerfunctions=controller,
            learner_functions=learner,
        )
        row = analyze_episode(trial_output / "attempt-00", protocol)
        row["flight_id"] = flight["flight_id"]
        row["evidence_sha256"] = {
            str(path.resolve()): file_digest(path) for path in result.artifacts.values()
        }
        occupancy_after = gpu_snapshot()
        write_json(trial_output / "gpu-after.json", occupancy_after, exclusive=True)
        row["foreign_gpu_compute_observed_after_flight"] = bool(
            occupancy_after["foreign_compute_processes"]
        )
        row["competing_gpu_compute_observed_after_flight"] = bool(
            occupancy_after["blocking_foreign_compute_processes"]
        )
        row["declared_desktop_context_observed_after_flight"] = occupancy_after[
            "declared_zero_memory_desktop_contexts"
        ]
        row["gpu_occupancy_observation_scope"] = (
            "process snapshots immediately before setup and after flight; "
            "root assigns exclusive experiment slot; declared zero-memory desktop context "
            "remains observable; not continuous tracing or proof of total device inactivity"
        )
        row["environment_evidence_sha256"] = {
            str(path.resolve()): file_digest(path)
            for path in (
                trial_output / "gpu-before.json",
                trial_output / "gpu-after.json",
                trial_output / "resolved_method.json",
            )
        }
        write_json(trial_output / "record.json", row, exclusive=True)
        rows.append(row)
        write_json(output / "progress.json", {"planned": 8, "records": rows})
        print(
            json.dumps(
                {
                    "finished": flight,
                    "status": row["status"],
                    "publications": row["published_online_updates"],
                    "safe_task_completion": row["safe_task_completion"],
                }
            ),
            flush=True,
        )
        load_protocol(protocol_path)
    report = {
        "planned": 8,
        "records": rows,
        "completed_attempts": sum(row["status"] == "completed" for row in rows),
        "full_duration_flights": sum(row["full_episode_completed"] for row in rows),
        "flights_with_foreign_gpu_compute_observed": sum(
            row["foreign_gpu_compute_observed_after_flight"] for row in rows
        ),
        "flights_with_competing_gpu_compute_observed": sum(
            row["competing_gpu_compute_observed_after_flight"] for row in rows
        ),
        "wall_seconds_including_warmup_and_serialization": time.perf_counter() - began,
        "protocol_content_sha256": digest(protocol),
        "comparison_scope": protocol["comparison_scope"],
    }
    write_json(output / "report.json", report, exclusive=True)
    write_report(output / "report.md", report)
    return report


def analyze_campaign(input_directory: Path, output: Path) -> dict[str, Any]:
    """Rebuild a report using authenticated retained evidence without GPU execution."""
    protocol_path = input_directory / "protocol.json"
    protocol = load_protocol(protocol_path, verify_files=False)
    binding = json.loads((input_directory / "binding.json").read_text())
    if binding["protocol_file_sha256"] != file_digest(protocol_path) or binding[
        "protocol_content_sha256"
    ] != digest(protocol):
        raise ValueError("retained campaign protocol differs from its execution binding")
    archive = Path(binding["protocol_path"]).parent / "source"
    for name, expected in protocol["source_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("bound source path escapes the protocol source archive")
        if file_digest(archive / relative) != expected:
            raise ValueError(f"archived runtime source differs from its binding: {name}")
    rows, missing = [], []
    for flight in protocol["ordered_flights"]:
        record_path = input_directory / flight["flight_id"] / "record.json"
        if not record_path.exists():
            missing.append(flight["flight_id"])
            continue
        saved = json.loads(record_path.read_text())
        for name, expected in {
            **saved["evidence_sha256"],
            **saved["environment_evidence_sha256"],
        }.items():
            if file_digest(Path(name)) != expected:
                raise ValueError(f"retained flight evidence changed: {name}")
        row = analyze_episode(Path(saved["episode_directory"]), protocol)
        if any(saved.get(key) != value for key, value in row.items()):
            raise ValueError("retained derived record differs from its authenticated raw evidence")
        rows.append(saved)
    report = {
        "planned": len(protocol["ordered_flights"]),
        "records": rows,
        "missing_flights": missing,
        "completed_attempts": sum(row["status"] == "completed" for row in rows),
        "full_duration_flights": sum(row["full_episode_completed"] for row in rows),
        "protocol_content_sha256": digest(protocol),
        "comparison_scope": protocol["comparison_scope"],
        "analysis_input_directory": str(input_directory.resolve()),
        "analysis_source_sha256": file_digest(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report, exclusive=True)
    write_report(output / "report.md", report)
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Present measured publication/outcome evidence without cross-mode causal claims."""
    lines = [
        report["comparison_scope"],
        "",
        (
            "| Method | Mode | Published | Before first center crossing | "
            "Postfault before center crossing | "
            "Safe task | Collision | Controller misses | Command delay p95 (ms) |"
        ),
        "|---|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in report["records"]:
        threat = row["center_crossing_counts"][0]
        delay = row["actual_sensing_to_command_application_seconds"]["p95"]
        delay_text = "unavailable" if delay is None else f"{1000 * delay:.3f}"
        lines.append(
            f"| {row['method']} | {row['execution_mode']} | {row['published_online_updates']} "
            f"| {threat['published_before_center_crossing']} "
            f"| {threat['postfault_trained_published_before_center_crossing']} "
            f"| {row['safe_task_completion']} | {row['collision']} "
            f"| {row['controller_deadline_misses']} "
            f"| {delay_text} |"
        )
    lines.extend(
        [
            "",
            "Strict-before publication counts at the declared reference landmarks:",
            "",
            "| Method | Mode | Landmark (s) | Published | Postfault trained | Prefix reached cut |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in report["records"]:
        for landmark in row["landmark_counts"]:
            lines.append(
                f"| {row['method']} | {row['execution_mode']} | "
                f"{landmark['label']} ({landmark['time_seconds']:.2f}) | "
                f"{landmark['published_before_landmark']} | "
                f"{landmark['postfault_trained_published_before_landmark']} | "
                f"{landmark['physical_prefix_reached_landmark']} |"
            )
    lines.extend(
        [
            "",
            (
                "Publication counts cover only the executed physical prefix. "
                "Pending terminal results "
                "and late completions are not publications. Paced command delays are zero by its "
                "nondelayed physical model; this does not establish instantaneous real actuation. "
                "Async host overlap does not establish concurrent GPU kernels or speedup. "
                "The JSON report retains warmups, all service/age samples and denominators, "
                "actual command holds, limits, late work, and checksummed raw evidence."
                " The 4.90045 s crossing is an obstacle-center crossing, not first contact or a "
                "needed-by deadline; physical contacts in this geometry can precede it."
            ),
        ]
    )
    with path.open("x") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> None:
    """Keep protocol construction separate from the explicitly scheduled GPU execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--output", type=Path, default=PROTOCOL.parent)
    seal.add_argument("--base", type=Path, default=BASE)
    seal.add_argument("--design", type=Path, default=DESIGN)
    run = commands.add_parser("run")
    run.add_argument("--protocol", type=Path, default=PROTOCOL)
    run.add_argument("--output", type=Path, default=BASE / "runtime-measured-v1")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        result = seal_protocol(args.output, base=args.base, design=args.design)
        print(json.dumps({"sha256": result["sha256"], "output": str(args.output)}))
    elif args.command == "run":
        run_campaign(args.protocol, args.output)
    else:
        analyze_campaign(args.input, args.output)


if __name__ == "__main__":
    main()
