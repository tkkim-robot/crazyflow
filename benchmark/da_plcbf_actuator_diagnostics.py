"""Execute the prospectively declared actuator learner/library diagnostic iteration.

Every arm starts from a checksummed complete checkpoint. Freeze-at-fault arms run
the same pre-fault learner and controller history, then stop learning. Union arms
retain the original frozen policies alongside all completed adaptive updates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import numpy as np

from benchmark.da_plcbf_actuator_study import ROOT, runner_source_files, write_json
from crazyflow.safety.da_plcbf.actuator_compute import PackedActuatorController
from crazyflow.safety.da_plcbf.actuator_experiment import (
    ActuatorEpisodeConfig,
    run_actuator_episode,
)
from crazyflow.safety.da_plcbf.actuator_learning import (
    actuator_reference_fingerprint,
    build_actuator_skill_learner,
    load_actuator_learner_checkpoint,
    save_actuator_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.actuator_libraries import library_size_metadata
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_protocol import content_sha256
from crazyflow.safety.da_plcbf.actuator_study import (
    ADAPTIVE_METHODS,
    ActuatorObservationConfig,
    build_actuator_controller,
    make_actuator_scene,
)

PREVIOUS = ROOT / "artifacts/da_plcbf/actuator-study-20260906/v1"
BASE = ROOT / "artifacts/da_plcbf/actuator-diagnostics-20260906/v1"
NOMINAL = PREVIOUS / "behavior-nominal128-seed11-v1/deployment"
DR = PREVIOUS / "behavior-dr512-seed11-v1/deployment"
PROTOCOL = ROOT / "artifacts/da_plcbf/actuator-diagnostic-20260906/v1/protocol-sealed-v1.json"


def sha(path: Path) -> str:
    """Exact file digest, distinct from canonical JSON content digests."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_balanced(output: Path) -> dict:
    """Retain the full learned checkpoint, changing only the declared loss contract."""
    source = load_actuator_learner_checkpoint(NOMINAL)
    output.mkdir(parents=True, exist_ok=False)
    objective = replace(
        source.contract.learning_config,
        objective_mode="balanced_reference_braking",
        recovery_braking_priority=10.0,
    )
    contract = replace(source.contract, learning_config=objective)
    paths = save_actuator_learner_checkpoint(
        source.state,
        contract,
        source.physical_state,
        output / "deployment",
        config=source.config,
        metadata={
            **source.metadata,
            "diagnostic_objective": asdict(objective),
            "derived_from_npz": str(source.npz_path),
            "derived_from_sha256": source.sha256,
            "full_optimizer_history_preserved": True,
        },
    )
    restored = load_actuator_learner_checkpoint(output / "deployment")
    for left, right in zip(
        jax.tree.leaves(source.state), jax.tree.leaves(restored.state), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    result = {
        "source_checkpoint": str(NOMINAL),
        "source_sha256": source.sha256,
        "files_sha256": {str(p): sha(p) for p in paths},
        "all_complete_state_arrays_exact": True,
        "initial_params_previous_params_Adam_model_counters_preserved": True,
        "new_reference_sha256": actuator_reference_fingerprint(contract),
        "objective": asdict(objective),
    }
    write_json(output / "summary.json", result)
    return result


class DiagnosticResources:
    """Shared executable caches, without sharing mutable learner state between episodes."""

    def __init__(self, base: Path = BASE) -> None:
        """Bind the selected objective and all comparator checkpoints explicitly."""
        self.paths = {
            "F2": NOMINAL,
            "A": NOMINAL,
            "DR": DR,
            "F2_2K": DR,
            "A_BAL": base / "balanced-prepare-v2/deployment",
            "UNION": base / "balanced-prepare-v2/deployment",
            "PD_F": base / "pd-prepare-v2/deployment",
            "PD_A": base / "pd-prepare-v2/deployment",
        }
        self.bundles: dict[str, Any] = {}
        self.controllers: dict[str, Any] = {}
        self.learners: dict[str, Any] = {}

    def bundle(self, method: str) -> Any:
        """Load once from checksummed bytes and return an immutable checkpoint object."""
        path = self.paths[method]
        key = str(path)
        if key not in self.bundles:
            self.bundles[key] = load_actuator_learner_checkpoint(path)
        return self.bundles[key]

    def resolve(self, method: str, filter_config: ActuatorFilterConfig) -> tuple[Any, ...]:
        """Resolve a static core only for the explicitly labeled equal-count comparisons."""
        bundle = self.bundle(method)
        core = self.bundle("F2") if method in {"UNION", "F2_2K"} else None
        metadata = {
            **library_size_metadata(bundle),
            "current_checkpoint": str(self.paths[method]),
            "current_npz_sha256": bundle.sha256,
            "current_json_sha256": sha(bundle.json_path),
            "current_reference_sha256": actuator_reference_fingerprint(bundle.contract),
            "objective": asdict(bundle.contract.learning_config),
            "actor_config": asdict(bundle.config),
            "immutable_core": None
            if core is None
            else {
                "checkpoint": str(NOMINAL),
                "npz_sha256": core.sha256,
                "json_sha256": sha(core.json_path),
                **library_size_metadata(core),
            },
        }
        metadata["total_fallback_count"] = metadata["fallback_policy_count"] + (
            0 if core is None else int(core.contract.spec.latent_codes.shape[0])
        )
        controller_key = content_sha256(
            {
                "actor_config": asdict(bundle.config),
                "spec": {
                    k: np.asarray(v).tolist() for k, v in asdict(bundle.contract.spec).items()
                },
                "filter": asdict(filter_config),
                "core": metadata["immutable_core"],
            }
        )
        if controller_key not in self.controllers:
            frozen = None if core is None else (core.state.params, core.contract.spec, core.config)
            functions = build_actuator_controller(
                bundle.contract.spec, bundle.config, filter_config, frozen_library=frozen
            )
            self.controllers[controller_key] = functions._replace(
                controller=PackedActuatorController(functions.controller)
            )
        if method in ADAPTIVE_METHODS and str(self.paths[method]) not in self.learners:
            self.learners[str(self.paths[method])] = build_actuator_skill_learner(
                bundle.contract, bundle.config
            )
        return (
            bundle,
            self.controllers[controller_key],
            self.learners.get(str(self.paths[method])),
            metadata,
        )


def source_binding() -> dict[str, str]:
    """Bind original runtime envelope plus the new preparation and protocol entrypoints."""
    paths = set(runner_source_files())
    paths.update(
        (Path(__file__).resolve(), ROOT / "benchmark/da_plcbf_actuator_diagnostic_protocol.py")
    )
    paths.add(ROOT / "benchmark/da_plcbf_actuator_pd_prepare.py")
    paths.add(ROOT / "benchmark/da_plcbf_actuator_loss_diagnosis.py")
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths) if p.exists()}


def resolve_trial_scene(proposal: dict, trial: dict, filters: ActuatorFilterConfig) -> Any:
    """Authenticate the actual geometry and perturbed physical scene against the sealed design."""
    from benchmark.da_plcbf_actuator_diagnostic_protocol import digest

    world = proposal["worlds"][trial["world_key"]]
    cell = next(c for c in proposal["factorial_cells"] if c["cell_id"] == trial["cell_id"])
    scene = make_actuator_scene(
        world["scene_seed"],
        world["family"],
        "effectiveness",
        control_period=filters.command_period,
        dt=filters.dt,
    )
    geometry = scene.physical_spec()
    geometry.pop("actuator_events")
    if digest(geometry) != world["geometry_sha256"]:
        raise ValueError("resolved geometry differs from the sealed physical geometry")
    initial = scene.world.initial_state.copy()
    initial[0] += trial.get("initial_position_x_delta_m", 0.0)
    initial[7] += trial.get("initial_velocity_x_delta_mps", 0.0)
    initial.setflags(write=False)
    scene = replace(
        scene,
        world=replace(scene.world, initial_state=initial),
        event_time=cell["event_time_seconds"],
        effectiveness_after=tuple(cell["effectiveness_after"]),
        lag_multipliers_after=tuple(cell["lag_multipliers_after"]),
    )
    if digest(scene.physical_spec()) != trial["physical_spec_sha256"]:
        raise ValueError(
            "actual fault/initial state differs from the sealed physical specification"
        )
    return scene


def run_stage(
    protocol: Path,
    stage: str,
    output: Path,
    *,
    base: Path = BASE,
    only_arms: tuple[str, ...] = (),
    only_worlds: tuple[str, ...] = (),
    realization: str | None = None,
    resume: bool = False,
    reuse: tuple[Path, ...] = (),
    execution_mode: str | None = None,
) -> dict:
    """Complete selected sealed trials; runtime failures and all attempts remain recorded."""
    from benchmark.da_plcbf_actuator_diagnostic_protocol import load_sealed

    proposal = load_sealed(protocol)
    ids = set(proposal["stage_trial_ids"][stage])
    trials = [
        t
        for t in proposal["trials"]
        if t["trial_id"] in ids
        and (not only_arms or t["arm"]["arm"] in only_arms)
        and (not only_worlds or t["world_key"] in only_worlds)
        and (realization is None or t["realization"] == realization)
    ]
    if not trials:
        raise ValueError("no sealed trials match the requested stage/filter")
    # Diagnostic anchor and its old harmonic fault are inspected first, without new-outcome input.
    trials.sort(
        key=lambda t: (
            t["world_key"] != "structured_30101",
            t["world_key"],
            t["cell_id"] != "eta0.7_lag1_extra0",
            t["cell_id"],
            t["realization"],
            t["arm"]["arm"],
        )
    )
    resources = DiagnosticResources(base)
    checkpoint_files = {}
    for method in resources.paths:
        bundle = resources.bundle(method)
        for path in (bundle.npz_path, bundle.json_path):
            checkpoint_files[str(path)] = sha(path)
    binding = {
        "protocol": str(protocol.resolve()),
        "protocol_file_sha256": sha(protocol),
        "source_sha256": source_binding(),
        "checkpoint_files_sha256": checkpoint_files,
        "scientific_runtime_sha256": content_sha256(
            {"source": source_binding(), "checkpoints": checkpoint_files}
        ),
        "stage": stage,
        "resolved_trial_ids": [t["trial_id"] for t in trials],
        "execution_mode_override": execution_mode,
        "ordering": "historical anchor/fault first, then deterministic lexical order",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    binding["sha256"] = content_sha256(binding)
    if resume:
        if json.loads((output / "diagnostic_runtime_binding.json").read_text()) != binding:
            raise ValueError("resume requires identical protocol, source, checkpoints and trials")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "diagnostic_runtime_binding.json", binding)
        for name in binding["source_sha256"]:
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        write_json(
            output / "environment.json",
            {
                "devices": [str(d) for d in jax.devices()],
                "jax_version": jax.__version__,
                "compilation_cache": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
                "execution_mode": execution_mode or "deterministic",
                "timing_claim": (
                    "deterministic service samples are descriptive; dedicated runtime follows"
                ),
            },
        )
    began = time.perf_counter()
    records = []
    for trial in trials:
        trial_output = output / trial["trial_id"]
        record_path = trial_output / "record.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record["summary"]["status"] == "completed":
                records.append(record)
                continue
        reused = None
        for parent in reuse:
            path = parent / trial["trial_id"] / "record.json"
            if path.exists():
                candidate = json.loads(path.read_text())
                if candidate["summary"]["status"] == "completed":
                    old_binding = json.loads(
                        (parent / "diagnostic_runtime_binding.json").read_text()
                    )
                    if any(
                        old_binding["source_sha256"].get(k) != v
                        for k, v in binding["source_sha256"].items()
                    ):
                        raise ValueError("reuse requires identical complete numerical source")
                    if candidate["trial"] != trial:
                        raise ValueError("reused trial is not the exact sealed identity")
                    for name, expected in candidate["evidence_sha256"].items():
                        if sha(Path(name)) != expected:
                            raise ValueError("reused episode evidence changed")
                    reused = candidate
                    break
        if reused is not None:
            trial_output.mkdir(parents=True, exist_ok=True)
            write_json(record_path, reused)
            records.append(reused)
            continue
        configuration = dict(proposal["shared_control"])
        filters = ActuatorFilterConfig(**configuration.pop("filter_config", {}))
        observation = ActuatorObservationConfig(**configuration.pop("observation_config", {}))
        scene = resolve_trial_scene(proposal, trial, filters)
        method = trial["arm"]["runtime_method"]
        configuration.update(
            {
                "method": method,
                "freeze_learning_at": trial["arm"]["freeze_learning_at"],
                "capture_times": tuple(scene.event_time + d for d in (0.0, 0.4, 0.8)),
                "save_checkpoints": True,
                "retain_rollouts": stage == "library_comparison",
            }
        )
        if execution_mode is not None:
            configuration["execution_mode"] = execution_mode
        config = ActuatorEpisodeConfig(
            filter_config=filters, observation_config=observation, **configuration
        )
        bundle, controller, learner, metadata = resources.resolve(method, filters)
        trial_output.mkdir(parents=True, exist_ok=True)
        attempt = len(list(trial_output.glob("attempt-*")))
        episode_output = trial_output / f"attempt-{attempt:02d}"
        print(
            json.dumps({"starting": trial, "completed": len(records), "planned": len(trials)}),
            flush=True,
        )
        result = run_actuator_episode(
            scene,
            bundle,
            config,
            episode_output,
            controllerfunctions=controller,
            learner_functions=learner,
        )
        evidence = {str(p.resolve()): sha(p) for p in episode_output.rglob("*") if p.is_file()}
        record = {
            "trial": trial,
            "method_metadata": metadata,
            "summary": result.summary,
            "episode_directory": str(episode_output.resolve()),
            "campaign_binding_sha256": binding["sha256"],
            "runtime_binding_sha256": binding["scientific_runtime_sha256"],
            "evidence_sha256": evidence,
        }
        write_json(record_path, record)
        records.append(record)
        write_json(output / "campaign_progress.json", {"planned": len(trials), "records": records})
        print(
            json.dumps(
                {
                    "finished": trial["arm"]["arm"],
                    "world": trial["world_key"],
                    "cell": trial["cell_id"],
                    "status": result.summary["status"],
                    "termination": result.summary["termination"],
                    "waypoints": result.summary["waypoints_completed"],
                    "clearance": result.summary.get("modeled_collider_clearance_lower_m"),
                }
            ),
            flush=True,
        )
        if source_binding() != binding["source_sha256"]:
            raise RuntimeError(
                "numerical source changed during a bound campaign; preserve this attempt"
            )
    report = {
        "planned": len(trials),
        "records": records,
        "completed": sum(r["summary"]["status"] == "completed" for r in records),
        "campaign_binding_sha256": binding["sha256"],
        "wall_seconds_this_invocation": time.perf_counter() - began,
    }
    write_json(output / "campaign_result.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-balanced")
    prepare.add_argument("--output", type=Path, default=BASE / "balanced-prepare-v2")
    run = commands.add_parser("run")
    run.add_argument("--protocol", type=Path, default=PROTOCOL)
    run.add_argument("--stage", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--base", type=Path, default=BASE)
    run.add_argument("--only-arm", action="append", default=[])
    run.add_argument("--only-world", action="append", default=[])
    run.add_argument("--realization")
    run.add_argument("--reuse", type=Path, action="append", default=[])
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--execution-mode", choices=("deterministic", "paced", "delayed", "asynchronous")
    )
    args = parser.parse_args()
    if args.command == "prepare-balanced":
        print(json.dumps(prepare_balanced(args.output), indent=2))
    else:
        run_stage(
            args.protocol,
            args.stage,
            args.output,
            base=args.base,
            only_arms=tuple(args.only_arm),
            only_worlds=tuple(args.only_world),
            realization=args.realization,
            reuse=tuple(args.reuse),
            resume=args.resume,
            execution_mode=args.execution_mode,
        )


if __name__ == "__main__":
    main()
