from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.actuator_compute import PackedActuatorController
from crazyflow.safety.da_plcbf.actuator_experiment import (
    ActuatorEpisodeConfig,
    run_actuator_episode,
)
from crazyflow.safety.da_plcbf.actuator_learning import (
    build_actuator_skill_learner,
    load_actuator_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.actuator_opt import ActuatorOPTConfig, build_actuator_opt
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_protocol import content_sha256, load_sealed_protocol
from crazyflow.safety.da_plcbf.actuator_study import (
    ActuatorObservationConfig,
    build_actuator_controller,
    initial_augmented_state,
    make_actuator_scene,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/da_plcbf/actuator-study-20260906/v1"
DEFAULT_CHECKPOINT = STUDY / "behavior-nominal128-seed11-v1/deployment"


def runner_source_files(root: Path = ROOT) -> list[Path]:
    """Bind the full local runtime, dynamics parameters, and collider definitions.

    A conservative package envelope also covers indirect imports such as deadline
    scheduling, QP selection, quaternion operations, and collision auditing. Meshes
    are presentation assets: physical contact uses the bound XML sphere geometry.
    The protocol builder imports this function so its envelope cannot drift.
    """
    package = root / "crazyflow"
    return sorted(
        path
        for path in package.rglob("*")
        if path.is_file() and path.suffix in {".py", ".toml", ".xml"}
    ) + [root / "benchmark/da_plcbf_actuator_study.py"]


def _clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return _clean(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def timing_summary(seconds: list[float]) -> dict[str, float | int]:
    values = np.asarray(seconds)
    return {
        "count": len(values),
        "mean_seconds": float(np.mean(values)),
        **{f"p{p}_seconds": float(np.percentile(values, p)) for p in (50, 95, 99)},
        "max_seconds": float(np.max(values)),
    }


def resolve_scene(specification: dict[str, Any], cfg: ActuatorFilterConfig) -> Any:
    """Resolve explicit predeclared sanity/causal variants before executing a method."""
    scene = make_actuator_scene(
        specification["scene_seed"],
        specification["family"],
        specification["dynamics_cell"],
        dt=cfg.dt,
        control_period=cfg.command_period,
    )
    world = scene.world
    world_changes = specification.get("world_config", {})
    if world_changes:
        configuration = replace(world.config, **world_changes)
        configuration.validate()
        world = replace(world, config=configuration)
    obstacle_mode = specification.get("obstacle_mode", "prescribed")
    if obstacle_mode == "none":
        world = replace(
            world,
            config=replace(world.config, obstacle_count=0),
            obstacle_mean_centers=np.zeros((0, 3)),
            obstacle_amplitudes=np.zeros((0, 3)),
            obstacle_angular_frequencies=np.zeros(0),
            obstacle_phases=np.zeros(0),
            obstacle_radii=np.zeros(0),
        )
    elif obstacle_mode == "static":
        world = replace(
            world,
            obstacle_mean_centers=world.obstacle_kinematics(0.0)[0],
            obstacle_amplitudes=np.zeros_like(world.obstacle_amplitudes),
        )
    elif obstacle_mode != "prescribed":
        raise ValueError("unknown obstacle_mode")
    changes = dict(specification.get("scene_changes", {}))
    for key in ("effectiveness_after", "lag_multipliers_after"):
        if key in changes:
            changes[key] = tuple(changes[key])
    return replace(scene, world=world, **changes)


class CampaignResources:
    """Reuse compiled functions; every episode still restores its own complete checkpoint."""

    def __init__(self, specification: dict[str, Any]) -> None:
        """Bind caches to one explicit campaign specification."""
        self.specification = specification
        self.bundles: dict[str, Any] = {}
        self.controllers: dict[str, Any] = {}
        self.learners: dict[str, Any] = {}
        self.optimizers: dict[str, Any] = {}

    def resolve(self, method: str, seed: int, configuration: dict[str, Any]) -> tuple[Any, ...]:
        """Resolve a checkpoint and the shared compiled methods without changing its state."""
        checkpoint = Path(self.specification["checkpoints"][method][str(seed)])
        checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
        if str(checkpoint) not in self.bundles:
            self.bundles[str(checkpoint)] = load_actuator_learner_checkpoint(checkpoint)
        bundle = self.bundles[str(checkpoint)]
        actor = bundle.config
        if method in {"F0", "F1"}:
            actor = replace(actor, adapter_mode=method)
        episode = dict(configuration)
        filter_configuration = ActuatorFilterConfig(**episode.pop("filter_config", {}))
        observation = ActuatorObservationConfig(**episode.pop("observation_config", {}))
        optimization = ActuatorOPTConfig(**episode.pop("opt_config", {}))
        config = ActuatorEpisodeConfig(
            method=method,
            filter_config=filter_configuration,
            observation_config=observation,
            **episode,
        )
        controller_key = content_sha256(
            {
                "actor": asdict(actor),
                "filter": asdict(filter_configuration),
                "spec": _clean(asdict(bundle.contract.spec)),
            }
        )
        if controller_key not in self.controllers:
            functions = build_actuator_controller(bundle.contract.spec, actor, filter_configuration)
            if self.specification.get("packed_controller", True):
                functions = functions._replace(
                    controller=PackedActuatorController(functions.controller)
                )
            self.controllers[controller_key] = functions
        learner_key = str(checkpoint)
        if method in {"A", "A1"} and learner_key not in self.learners:
            self.learners[learner_key] = build_actuator_skill_learner(
                bundle.contract, bundle.config
            )
        optimizer_key = content_sha256(
            {
                "filter": asdict(filter_configuration),
                "actor": asdict(actor),
                "optimizer": asdict(optimization),
            }
        )
        if method == "OPT" and optimizer_key not in self.optimizers:
            self.optimizers[optimizer_key] = build_actuator_opt(
                filter_configuration, actor, optimization
            )
        return (
            bundle,
            config,
            self.controllers[controller_key],
            self.learners.get(learner_key),
            self.optimizers.get(optimizer_key),
        )


def run_campaign(specification_path: Path, output: Path, *, resume: bool = False) -> dict[str, Any]:
    """Run all resolved method/world/seed cells, retaining failures and exclusive attempts."""
    specification = json.loads(specification_path.read_text())
    required = {"purpose", "worlds", "methods", "library_seeds", "checkpoints", "episode_config"}
    if not required.issubset(specification):
        raise ValueError(
            f"campaign specification missing {sorted(required - specification.keys())}"
        )
    source_files = runner_source_files()
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    checkpoint_hashes = {}
    for paths in specification["checkpoints"].values():
        for stem in paths.values():
            path = Path(stem)
            path = path if path.is_absolute() else ROOT / path
            for suffix in (".npz", ".json"):
                file = path.with_suffix(suffix)
                checkpoint_hashes[str(file.relative_to(ROOT))] = hashlib.sha256(
                    file.read_bytes()
                ).hexdigest()
    if {"F2", "A"}.issubset(specification["methods"]):
        for seed in specification["library_seeds"]:
            digests = []
            for method in ("F2", "A"):
                stem = ROOT / specification["checkpoints"][method][str(seed)]
                digests.append(
                    tuple(
                        hashlib.sha256(stem.with_suffix(suffix).read_bytes()).hexdigest()
                        for suffix in (".npz", ".json")
                    )
                )
            if digests[0] != digests[1]:
                raise ValueError("F2 and A must start from the identical complete checkpoint")
        common_fields = {
            "filter_config",
            "observation_config",
            "cache_observation_inputs",
            "plant_level",
            "plant_step_seconds",
            "execution_mode",
        }
        for field in common_fields:
            values = [
                specification.get("method_config", {})
                .get(method, {})
                .get(field, specification["episode_config"].get(field))
                for method in ("F2", "A")
            ]
            if values[0] != values[1]:
                raise ValueError(f"F2 and A must share {field}")
    binding = {
        "specification": specification,
        "source_sha256": source_hashes,
        "checkpoint_sha256": checkpoint_hashes,
    }
    split = specification.get("split", "development")
    if split not in {"development", "validation", "test"}:
        raise ValueError("campaign split must be development, validation, or test")
    if split == "test":
        protocol_path = specification.get("sealed_protocol_path")
        if not protocol_path:
            raise ValueError("test evaluation requires an already sealed protocol")
        frozen = load_sealed_protocol(ROOT / protocol_path)
        manifest = frozen.manifest
        if manifest.get("checkpoint_sha256") != checkpoint_hashes:
            raise ValueError("checkpoint bytes differ from the sealed protocol")
        unsigned = {
            key: value for key, value in specification.items() if key != "sealed_protocol_path"
        }
        if manifest.get("campaign_specification_sha256") != content_sha256(unsigned):
            raise ValueError("test campaign differs from the sealed specification")
        if any(
            source_hashes.get(path) != digest
            for path, digest in manifest["source"]["files_sha256"].items()
        ):
            raise ValueError("source differs from the sealed protocol")
        if set(specification["methods"]) != set(manifest["methods"]):
            raise ValueError("test campaign method matrix differs from the sealed protocol")
        if specification["library_seeds"] != manifest["splits"]["test"]["library_seeds"]:
            raise ValueError("test library seeds differ from the sealed protocol")
        cfg = ActuatorFilterConfig(**specification["episode_config"].get("filter_config", {}))
        physical = [
            resolve_scene(world, cfg).metadata()["physical_world_id"]
            for world in specification["worlds"]
        ]
        if physical != [world["world_id"] for world in manifest["splits"]["test"]["worlds"]]:
            raise ValueError("test physical scenes differ from the sealed protocol")
        binding["sealed_protocol_sha256"] = frozen.sha256
    binding["sha256"] = content_sha256(binding)
    if resume:
        previous = json.loads((output / "campaign_binding.json").read_text())
        if previous != binding:
            raise ValueError("resume requires identical source, checkpoints, and specification")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "campaign_binding.json", binding)
        source_directory = output / "source"
        for file in source_files:
            destination = source_directory / file.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
        write_json(
            output / "environment.json",
            {
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "jax_version": jax.__version__,
                "devices": [str(device) for device in jax.devices()],
                "working_tree_differs_from_commit": bool(
                    subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
                ),
            },
        )
    resources = CampaignResources(specification)
    records = []
    began = time.perf_counter()
    planned = (
        len(specification["worlds"])
        * len(specification["library_seeds"])
        * len(specification["methods"])
    )
    for world_index, world_specification in enumerate(specification["worlds"]):
        for library_seed in specification["library_seeds"]:
            # Rotate order without using any outcome; avoid every adaptive run following F2.
            methods = list(specification["methods"])
            offset = (world_index + library_seed) % len(methods)
            methods = methods[offset:] + methods[:offset]
            for method in methods:
                name = f"world-{world_index:04d}-seed-{library_seed}-{method}"
                trial_directory = output / name
                completed = []
                if trial_directory.exists():
                    for summary_path in sorted(trial_directory.glob("attempt-*/summary.json")):
                        summary = json.loads(summary_path.read_text())
                        if summary["status"] == "completed":
                            completed.append((summary_path, summary))
                if len(completed) > 1:
                    raise ValueError("a planned trial has multiple completed attempts")
                if completed:
                    summary_path, summary = completed[0]
                    records.append(
                        {
                            "trial": name,
                            "world_index": world_index,
                            "library_seed": library_seed,
                            "method": method,
                            "summary_path": str(summary_path.relative_to(output)),
                            "summary": summary,
                        }
                    )
                    continue
                configuration = dict(specification["episode_config"])
                configuration.update(world_specification.get("episode_config", {}))
                configuration.update(specification.get("method_config", {}).get(method, {}))
                bundle, config, controller, learner, optimizer = resources.resolve(
                    method, library_seed, configuration
                )
                scene = resolve_scene(world_specification, config.filter_config)
                trial_directory.mkdir(exist_ok=True)
                attempt = len(list(trial_directory.glob("attempt-*")))
                directory = trial_directory / f"attempt-{attempt:02d}"
                print(
                    json.dumps(
                        {
                            "starting": name,
                            "attempt": attempt,
                            "completed": len(records),
                            "planned": planned,
                            "physical_world_id": scene.metadata()["physical_world_id"],
                        }
                    ),
                    flush=True,
                )
                result = run_actuator_episode(
                    scene,
                    bundle,
                    config,
                    directory,
                    controllerfunctions=controller,
                    learner_functions=learner,
                    opt_controller=optimizer,
                )
                summary = result.summary
                records.append(
                    {
                        "trial": name,
                        "world_index": world_index,
                        "library_seed": library_seed,
                        "method": method,
                        "summary_path": str((directory / "summary.json").relative_to(output)),
                        "summary": summary,
                    }
                )
                write_json(
                    output / "campaign_progress.json", {"planned": planned, "records": records}
                )
                print(
                    json.dumps(
                        _clean(
                            {
                                "finished": name,
                                "status": summary["status"],
                                "termination": summary["termination"],
                                "waypoints": summary["waypoints_completed"],
                                "collider_clearance": summary.get(
                                    "modeled_collider_clearance_lower_m"
                                ),
                                "wall_seconds": summary["execution_wall_seconds"],
                            }
                        )
                    ),
                    flush=True,
                )
    result = {
        "campaign_sha256": binding["sha256"],
        "purpose": specification["purpose"],
        "planned": planned,
        "completed": sum(row["summary"]["status"] == "completed" for row in records),
        "wall_seconds_this_invocation": time.perf_counter() - began,
        "records": records,
    }
    write_json(output / "campaign_result.json", result)
    return result


def profile_controller(
    checkpoint: Path,
    output: Path,
    count: int,
    *,
    gradient_mode: str = "forward",
    packed: bool = False,
    check_full_ad: bool = False,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    bundle = load_actuator_learner_checkpoint(checkpoint)
    cfg = ActuatorFilterConfig(
        dt=bundle.config.dt,
        horizon=bundle.config.horizon,
        command_hold_steps=bundle.config.control_interval_steps,
        gradient_mode=gradient_mode,
    )
    functions = build_actuator_controller(bundle.contract.spec, bundle.config, cfg)
    reference = (
        build_actuator_controller(
            bundle.contract.spec, bundle.config, replace(cfg, gradient_mode="forward")
        )
        if check_full_ad
        else None
    )
    if packed:
        functions = functions._replace(controller=PackedActuatorController(functions.controller))
    scene = make_actuator_scene(
        101, "structured", "combined", dt=cfg.dt, control_period=cfg.command_period
    )
    state = jnp.asarray(
        initial_augmented_state(scene.world.initial_state, bundle.contract.model), dtype=jnp.float32
    )
    safety = scene.world.safety_limits()
    goal = jnp.asarray(scene.world.initial_state[:3], dtype=jnp.float32)
    profiles = {}
    for name, when in (("nominal_early", 0.0), ("combined_onset", scene.event_time)):
        point = scene.model_at(when, bundle.contract.model)
        obstacles = scene.world.obstacle_prediction(when, dt=cfg.dt, horizon=cfg.horizon)
        args = (
            state,
            bundle.state.params,
            point,
            obstacles,
            safety,
            jnp.asarray(-1, jnp.int32),
            goal,
        )
        start = time.perf_counter()
        first = functions.controller(*args)
        jax.block_until_ready(first)
        first_seconds = time.perf_counter() - start
        service, transfer = [], []
        for _ in range(count):
            start = time.perf_counter()
            result = functions.controller(*args)
            jax.block_until_ready(result)
            completed = time.perf_counter()
            host = jax.device_get(result)
            transferred = time.perf_counter()
            service.append(completed - start)
            transfer.append(transferred - completed)
        profiles[name] = {
            "first_call_seconds_including_any_compile": first_seconds,
            "call_service": timing_summary(service),
            "host_transfer": timing_summary(transfer),
            "execution_mode": int(host.execution_mode),
            "qp_valid": bool(host.qp_valid),
            "eligible_count": int(np.sum(host.certificates.eligible)),
            "finite_gradient_count": int(np.sum(host.certificates.gradient_valid)),
            "policy_count_including_nominal": len(host.certificates.eligible),
            "selected_policy": int(host.selected_index),
            "executed_policy_dual": float(host.executed_policy_dual),
            "sqp_iterations": int(host.sqp_iterations),
            "action": np.asarray(host.action),
            "hard_values": host.certificates.hard.values,
            "smooth_values": host.certificates.smooth_values,
            "gradient_components_computed": host.certificates.gradient_components_computed,
        }
        if reference is not None:
            full = jax.device_get(reference.controller(*args))
            profiles[name]["full_ad_parity"] = {
                "maximum_row_difference": float(
                    np.max(np.abs(full.certificates.rows - host.certificates.rows))
                ),
                "maximum_bound_difference": float(
                    np.max(np.abs(full.certificates.bounds - host.certificates.bounds))
                ),
                "maximum_action_difference": float(np.max(np.abs(full.action - host.action))),
                "hard_values_identical": bool(
                    np.array_equal(full.certificates.hard.values, host.certificates.hard.values)
                ),
                "eligibility_identical": bool(
                    np.array_equal(full.certificates.eligible, host.certificates.eligible)
                ),
                "selected_index_identical": bool(
                    np.array_equal(full.selected_index, host.selected_index)
                ),
                "execution_mode_identical": bool(
                    np.array_equal(full.execution_mode, host.execution_mode)
                ),
            }
    record = {
        "purpose": "development computation pilot; repeated same-state calls are not full episodes",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": bundle.sha256,
        "filter_config": asdict(cfg),
        "actor_config": asdict(bundle.config),
        "device": str(jax.devices()[0]),
        "packed_host_transport": packed,
        "call_service_scope": "compiled call, synchronization, and packed host transport"
        if packed
        else "compiled call and synchronization; separate host transfer",
        "profiles": profiles,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "crazyflow/safety/da_plcbf/actuator_plcbf.py", Path(__file__))
        },
    }
    write_json(output / "profile.json", record)
    for relative, digest in record["source_sha256"].items():
        source = ROOT / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError("source changed while the profile was running")
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print(json.dumps(_clean(record), indent=2), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Actuator study: reproducible staged numerical evaluation."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser(
        "profile", help="Measure complete compiled filter service before setting budgets"
    )
    profile.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    profile.add_argument("--output", type=Path, required=True)
    profile.add_argument("--count", type=int, default=32)
    profile.add_argument(
        "--gradient-mode", choices=("forward", "reverse", "directional"), default="forward"
    )
    profile.add_argument("--packed", action="store_true")
    profile.add_argument("--check-full-ad", action="store_true")
    campaign = commands.add_parser(
        "campaign", help="Execute a fully specified development or sealed matrix"
    )
    campaign.add_argument("--specification", type=Path, required=True)
    campaign.add_argument("--output", type=Path, required=True)
    campaign.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "profile":
        if args.count < 2:
            parser.error("profile count must be at least two")
        profile_controller(
            args.checkpoint,
            args.output,
            args.count,
            gradient_mode=args.gradient_mode,
            packed=args.packed,
            check_full_ad=args.check_full_ad,
        )
    elif args.command == "campaign":
        run_campaign(args.specification, args.output, resume=args.resume)


if __name__ == "__main__":
    main()
