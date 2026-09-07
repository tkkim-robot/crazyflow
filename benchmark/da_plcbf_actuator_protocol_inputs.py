"""Prepare reviewable inputs for the declared 384-episode actuator mini benchmark.

This CPU-only setup resolves geometry and checks JSON/checkpoint identities. It
does not execute controllers, learners, plants, the protocol builder, or sealing.
The builder supplies the committed-source envelope in a later explicit step.

Example:
    python -m benchmark.da_plcbf_actuator_protocol_inputs \
        --output artifacts/da_plcbf/actuator-study-20260906/v1/protocol-inputs-main384-draft-v1
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/da_plcbf/actuator-study-20260906/v1"
FAMILIES = ("structured", "navigation")
CELLS = ("nominal", "effectiveness", "lag", "combined")
METHODS = ("F2", "DR", "A", "OPT")
LIBRARY_SEEDS = (11, 23, 37)
REPLICATES = 4


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(_plain(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _campaign(filter_config: Any) -> dict[str, Any]:
    worlds = [
        {
            "family": family,
            "scene_seed": 30000 + family_index * 1000 + cell_index * 100 + replicate,
            "dynamics_cell": cell,
            "cell": f"{family}_{cell}",
        }
        for family_index, family in enumerate(FAMILIES)
        for cell_index, cell in enumerate(CELLS)
        for replicate in range(1, REPLICATES + 1)
    ]
    checkpoints = {
        method: {
            str(seed): _relative(
                STUDY
                / (
                    f"behavior-dr512-seed{seed}-v1/deployment"
                    if method == "DR"
                    else f"behavior-nominal128-seed{seed}-v1/deployment"
                )
            )
            for seed in LIBRARY_SEEDS
        }
        for method in METHODS
    }
    return {
        "purpose": (
            "Declared main mini benchmark: 32 independently generated physical worlds "
            "(four distinct geometry seeds in each of eight family/dynamics cells), "
            "crossed with three nominal-library seeds and four methods for 384 episodes. "
            "Original 40ms control and objective; no scene rejection based on H or outcomes. "
            "This specification is a review input and requires separate protocol sealing."
        ),
        "split": "test",
        "worlds": worlds,
        "methods": list(METHODS),
        "library_seeds": list(LIBRARY_SEEDS),
        "checkpoints": checkpoints,
        "episode_config": {
            "execution_mode": "deterministic",
            "plant_level": "P0",
            "plant_step_seconds": 0.005,
            "warmup_calls": 3,
            "filter_config": asdict(filter_config),
        },
        "packed_controller": True,
    }


def _checkpoint_inventory(campaign: dict[str, Any]) -> dict[str, Any]:
    inventory = {}
    for method in METHODS:
        inventory[method] = {}
        for seed in LIBRARY_SEEDS:
            stem = ROOT / campaign["checkpoints"][method][str(seed)]
            manifest = json.loads(stem.with_suffix(".json").read_text())
            expected_mode, expected_steps = ("dr", 512) if method == "DR" else ("nominal", 128)
            spec = manifest["structure"]["items"]["reference"]["items"]["spec"]["items"]
            latent_shape = manifest["arrays"][spec["latent_codes"]["key"]]["shape"]
            if (
                manifest["format"] != "crazyflow.actuator_skill_checkpoint"
                or manifest["state_size"] != 17
                or manifest["metadata"].get("mode") != expected_mode
                or manifest["metadata"].get("seed") != seed
                or manifest["library_version"] != expected_steps
                or manifest["cumulative_gradient_steps"] != expected_steps
                or latent_shape != [16, 8]
            ):
                raise ValueError(f"checkpoint provenance mismatch for {method}/seed{seed}")
            npz_hash = _sha256(stem.with_suffix(".npz"))
            if npz_hash != manifest["npz_sha256"]:
                raise ValueError(f"checkpoint checksum mismatch for {stem}")
            inventory[method][str(seed)] = {
                "stem": _relative(stem),
                "npz_sha256": npz_hash,
                "json_sha256": _sha256(stem.with_suffix(".json")),
                "reference_sha256": manifest["reference_sha256"],
                "mode": expected_mode,
                "gradient_steps": expected_steps,
                "policy_count": latent_shape[0],
            }
    for seed in map(str, LIBRARY_SEEDS):
        if inventory["F2"][seed] != inventory["A"][seed]:
            raise ValueError("F2 and A must retain identical complete checkpoint provenance")
        if any(
            inventory[method][seed]["reference_sha256"] != inventory["A"][seed]["reference_sha256"]
            for method in ("DR", "OPT")
        ):
            raise ValueError("every primary method must retain the same nominal teacher per seed")
    return inventory


def _support_inputs(directory: Path, common: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    from benchmark.da_plcbf_actuator_study import resolve_scene
    from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
    from crazyflow.safety.da_plcbf.actuator_protocol import content_sha256

    paths = sorted(directory.rglob("*.json"))
    if not paths:
        raise ValueError("no current specification JSON files found")
    unique: dict[str, dict[str, Any]] = {"development": {}, "validation": {}}
    libraries: dict[str, set[int]] = {"development": set(), "validation": set()}
    records, excluded, hashes = [], [], {}
    for path in paths:
        payload = path.read_bytes()
        hashes[_relative(path)] = hashlib.sha256(payload).hexdigest()
        spec = json.loads(payload)
        split = spec.get("split")
        if (
            split not in unique
            or not isinstance(spec.get("worlds"), list)
            or not spec["worlds"]
            or "profil" in path.stem.lower()
            or "selection" in path.stem.lower()
        ):
            excluded.append(
                {
                    "path": _relative(path),
                    "reason": (
                        "not a development/validation world specification or is selection/profiling"
                    ),
                }
            )
            continue
        seeds = spec.get("library_seeds")
        if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError(f"explicit library seeds required in {path}")
        libraries[split].update(seeds)
        for index, original in enumerate(spec["worlds"]):
            raw = copy.deepcopy(original)
            episode = dict(spec.get("episode_config", {}))
            episode.update(raw.get("episode_config", {}))
            effective = ActuatorFilterConfig(**episode.get("filter_config", {}))
            if (effective.dt, effective.horizon, effective.command_hold_steps) != (
                common.dt,
                common.horizon,
                common.command_hold_steps,
            ):
                raise ValueError(f"support world clocks differ from main contract in {path}")
            metadata = _plain(resolve_scene(raw, effective).metadata())
            # Timing/estimator/learner variants do not define a new physical world.
            raw.pop("episode_config", None)
            if (
                resolve_scene(raw, common).metadata()["physical_world_id"]
                != metadata["physical_world_id"]
            ):
                raise ValueError("stripping nonphysical overrides changed a support world")
            raw.setdefault("cell", f"{raw['family']}_{raw['dynamics_cell']}")
            identity = metadata["physical_world_id"]
            record = {
                "specification_path": _relative(path),
                "specification_sha256": hashes[_relative(path)],
                "source_world_index": index,
                "split": split,
                "world_id": identity,
                "scene_seed": raw["scene_seed"],
                "original_world": original,
                "library_seeds": seeds,
            }
            records.append(record)
            complexity = (
                len(raw.get("world_config", {}))
                + len(raw.get("scene_changes", {}))
                + int(raw.get("obstacle_mode", "prescribed") != "prescribed")
            )
            rank = (complexity, content_sha256(raw), _relative(path), index)
            existing = unique[split].get(identity)
            if existing is None:
                unique[split][identity] = {
                    "rank": rank,
                    "world": raw,
                    "physical_spec": metadata["physical_spec"],
                    "aliases": [record],
                }
            else:
                existing["aliases"].append(record)
                if rank < existing["rank"]:
                    existing.update(rank=rank, world=raw)
    supports, details = {}, {}
    used_identities: set[str] = set()
    used_generator_seeds: set[int] = set()
    for split in ("development", "validation"):
        worlds = sorted(
            unique[split].values(),
            key=lambda item: (
                item["world"]["family"],
                item["world"]["scene_seed"],
                item["world"]["cell"],
                content_sha256(item["world"]),
            ),
        )
        if not worlds:
            raise ValueError(f"no {split} physical support found")
        identities = set(unique[split])
        generator_seeds = {alias["scene_seed"] for item in worlds for alias in item["aliases"]}
        if used_identities & identities or used_generator_seeds & generator_seeds:
            raise ValueError(
                "development and validation contain overlapping physics or generator seeds"
            )
        used_identities.update(identities)
        used_generator_seeds.update(generator_seeds)
        supports[split] = {
            "purpose": (
                "Deduplicated physical support from every current declared specification, "
                "including planned expansion, candidate checks and secondary variants. "
                "Membership does not imply that every declared episode has run."
            ),
            "split": split,
            "worlds": [item["world"] for item in worlds],
            "library_seeds": sorted(libraries[split]),
        }
        details[split] = [
            {
                "world_id": content_identity,
                "representative": item["world"],
                "physical_spec": item["physical_spec"],
                "source_aliases": item["aliases"],
            }
            for content_identity, item in sorted(unique[split].items())
        ]
    return supports, {
        "specifications_directory": _relative(directory),
        "source_file_sha256": hashes,
        "included_world_entries": len(records),
        "included_specifications": sorted({row["specification_path"] for row in records}),
        "excluded_files": excluded,
        "deduplication": "complete resolve_scene physical_world_id; no outcomes or H values used",
        "unique_worlds_and_aliases": details,
    }


def _generator_support() -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Closed enclosing supports, retaining correlations in the accompanying formulas."""
    ranges = {
        "effectiveness_including_healthy": [0.70, 1.0],
        "affected_effectiveness_choice": [0.70, 0.85],
        "lag_multiplier_including_healthy": [1.0, 3.0],
        "affected_lag_multiplier_choice": [1.5, 3.0],
        "fault_time_s": [2.0, 2.0],
        "affected_motor_count": [1, 4],
        "affected_motor_index": [0, 3],
        "command_period_s": [0.04, 0.04],
        "prediction_step_s": [0.02, 0.02],
        "prediction_horizon_steps": [60, 60],
        "prediction_horizon_s": [1.2, 1.2],
        "primary_plant_step_s": [0.005, 0.005],
        "arena_x_m": [-5.0, 5.0],
        "arena_y_m": [-4.0, 4.0],
        "arena_z_m": [0.15, 4.0],
        "arena_clearance_m": [0.08, 0.08],
        "operational_center_x_m": [-4.92, 4.92],
        "operational_center_y_m": [-3.92, 3.92],
        "operational_center_z_m": [0.23, 3.92],
        "ego_safety_radius_m": [0.106, 0.106],
        "xml_collider_radius_m": [0.086, 0.086],
        "requested_obstacle_clearance_m": [0.15, 0.15],
        "waypoint_reach_radius_m": [0.4, 0.4],
        "speed_max_m_s": [3.5, 3.5],
        "angular_rate_max_rad_s": [12.0, 12.0],
        "tilt_max_rad": [0.9, 0.9],
        "wind_each_component_m_s": [0.0, 0.0],
        "initial_body_rate_each_component_rad_s": [0.0, 0.0],
        "structured_duration_s": [14.0, 14.0],
        "structured_obstacle_count": [4, 4],
        "structured_moving_obstacle_count": [2, 2],
        "structured_guard_count": [2, 2],
        "structured_waypoint_count": [2, 2],
        "structured_initial_velocity_each_component_m_s": [-0.35, 0.35],
        "structured_azimuth_rad": [-math.pi, math.pi],
        "structured_first_raw_direction_z": [-0.16, 0.16],
        "structured_second_raw_direction_z": [-0.12, 0.12],
        "structured_raw_and_normalized_direction_each_xy": [-1.0, 1.0],
        "structured_first_arrival_after_fault_s": [1.6, 4.0],
        "structured_first_arrival_s": [3.6, 6.0],
        "structured_second_arrival_gap_s": [0.4, 1.4],
        "structured_second_arrival_s": [4.0, 7.4],
        "structured_mover_speed_at_arrival_m_s": [1.2, 2.8],
        "structured_first_radius_m": [0.38, 0.67],
        "structured_second_radius_m": [0.32, 0.58],
        "structured_first_crossing_offset_xy_m": [-0.12, 0.12],
        "structured_first_crossing_offset_z_m": [-0.06, 0.06],
        "structured_second_crossing_offset_each_m": [-0.20, 0.20],
        "structured_mover_amplitude_norm_m": [20.0, 20.0],
        "structured_mover_amplitude_each_xy_m": [-20.0, 20.0],
        "structured_mover_angular_frequency_rad_s": [0.06, 0.14],
        "structured_first_phase_rad": [-0.84, -0.216],
        "structured_second_phase_rad": [-1.036, -0.24],
        "structured_first_mean_each_xy_m": [-0.12, 0.12],
        "structured_first_mean_z_m": [1.34, 1.46],
        "structured_second_mean_each_xy_m": [-0.20, 0.20],
        "structured_second_mean_z_m": [1.20, 1.60],
        "structured_guard_distance_parameter_m": [1.0, 1.5],
        "structured_guard_radius_m": [0.22, 0.34],
        "structured_guard_mean_each_xy_m": [-1.5, 1.5],
        "structured_guard_offset_z_m": [-0.18, 0.18],
        "structured_guard_mean_z_m": [1.22, 1.58],
        "structured_guard_amplitude_frequency_phase": [0.0, 0.0],
        "structured_navigation_start_after_second_arrival_s": [1.2, 1.24],
        "structured_navigation_start_s": [5.2, 8.6],
        "navigation_duration_s": [24.0, 24.0],
        "navigation_start_s": [0.0, 0.0],
        "navigation_obstacle_count": [6, 6],
        "navigation_waypoint_count": [4, 4],
        "navigation_initial_velocity_each_component_m_s": [-0.25, 0.25],
        "navigation_mean_perturbation_each_coordinate_m": [-0.12, 0.12],
        "navigation_obstacle_period_s": [6.0, 10.0],
        "navigation_obstacle_angular_frequency_rad_s": [2 * math.pi / 10, 2 * math.pi / 6],
        "navigation_arrival_perturbation_s": [-0.6, 0.6],
        "navigation_obstacle_radius_m": [0.30, 0.50],
    }
    vectors = {
        "arena_lower_m": [-5.0, -4.0, 0.15],
        "arena_upper_m": [5.0, 4.0, 4.0],
        "xml_collider_offset_body_m": [0.0, 0.0, 0.02],
        "initial_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "structured_initial_position_m": [0.0, 0.0, 1.4],
        "structured_waypoints_m": [[-1.5, 0.9, 1.8], [-2.0, -0.8, 1.4]],
        "navigation_initial_position_m": [-2.5, -0.7, 1.4],
        "navigation_waypoints_m": [
            [1.8, -0.8, 1.6],
            [2.0, 1.3, 2.2],
            [-1.6, 1.3, 2.7],
            [-2.1, -0.8, 1.4],
        ],
        "navigation_mean_centers_base_m": [
            [-0.5, -0.8, 1.5],
            [1.9, 0.3, 1.9],
            [0.5, 1.3, 2.4],
            [-1.7, 0.2, 2.0],
            [0.0, -0.5, 2.1],
            [-1.4, 0.9, 1.8],
        ],
        "navigation_amplitudes_m": [
            [0.0, 2.8, 0.25],
            [2.2, 0.0, 0.35],
            [0.0, 2.6, 0.35],
            [2.5, 0.0, 0.25],
            [0.0, 2.8, 0.45],
            [2.3, 0.0, 0.4],
        ],
        "navigation_arrivals_base_s": [2.8, 6.0, 10.0, 15.0, 19.0, 21.0],
    }

    def fixed(prefix: str, value: Any) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                fixed(f"{prefix}_{index}", item)
        else:
            ranges[prefix] = [value, value]

    for name, vector in vectors.items():
        fixed(name, vector)
    for label, raw_z in (("first", 0.16), ("second", 0.12)):
        z = raw_z / math.sqrt(1 + raw_z * raw_z)
        ranges[f"structured_{label}_normalized_direction_z"] = [-z, z]
        ranges[f"structured_{label}_amplitude_z_m"] = [-20 * z, 20 * z]
    for index, (mean, amplitude, arrival) in enumerate(
        zip(
            vectors["navigation_mean_centers_base_m"],
            vectors["navigation_amplitudes_m"],
            vectors["navigation_arrivals_base_s"],
            strict=True,
        )
    ):
        for axis, center, magnitude in zip("xyz", mean, amplitude, strict=True):
            ranges[f"navigation_obstacle{index}_mean_{axis}_m"] = [center - 0.12, center + 0.12]
            ranges[f"navigation_obstacle{index}_all_time_center_{axis}_m"] = [
                center - 0.12 - abs(magnitude),
                center + 0.12 + abs(magnitude),
            ]
        ranges[f"navigation_obstacle{index}_arrival_s"] = [arrival - 0.6, arrival + 0.6]
        ranges[f"navigation_obstacle{index}_phase_rad"] = [
            -(2 * math.pi / 6) * (arrival + 0.6),
            -(2 * math.pi / 10) * (arrival - 0.6),
        ]
    support = {
        "range_semantics": "closed enclosing supports; derived coordinates remain correlated",
        "fixed_geometry": vectors,
        "effectiveness_choices": [0.70, 0.85],
        "lag_multiplier_choices": [1.5, 2.0, 3.0],
        "affected_motor_patterns_zero_based": [[0], [0, 1], [0, 2], [0, 1, 2, 3]],
        "fault_choice_law": (
            "independent uniform discrete choices; inactive factors/motors remain one"
        ),
        "continuous_draw_law": "NumPy default_rng uniform draws on the declared intervals",
        "structured_direction_formula": (
            "first=[cos(theta),sin(theta),z1]; second=[-sin(theta),cos(theta),z2]; "
            "each mover direction is normalized, amplitude=20*direction, omega=speed/20, "
            "phase=-omega*arrival"
        ),
        "structured_guard_formula": (
            "guard centers=hover plus/minus distance*RAW second direction; "
            "guard directions are not normalized; static amplitude/frequency/phase zero"
        ),
        "structured_navigation_formula": "ceil((second_arrival+1.2)/0.04)*0.04",
        "navigation_formula": (
            "mean=base+per-coordinate U[-0.12,0.12]; fixed amplitude; "
            "omega=2*pi/U[6,10]; arrival=base+U[-0.6,0.6]; phase=-omega*arrival"
        ),
        "obstacle_motion": "center(t)=mean+amplitude*sin(omega*t+phase); no clipping to ego arena",
        "initial_motor_state": "nominal level-hover motor effort; no fault-time reset",
        "wind_and_payload": "zero wind, unchanged body mass/inertia, no payload events",
        "main_seed_formula": "30000+family_index*1000+cell_index*100+rep, rep=1..4",
        "family_index_order": list(FAMILIES),
        "cell_index_order": list(CELLS),
        "main_geometry_independence": "distinct generator seed in every family/cell/replicate",
    }
    return ranges, support


def _nominal_numeric_ranges(stem: Path) -> dict[str, list[float]]:
    """Bind numeric model constants directly to the existing nominal checkpoint arrays."""
    import numpy as np

    manifest = json.loads(stem.with_suffix(".json").read_text())
    model = manifest["structure"]["items"]["reference"]["items"]["model"]
    ranges = {}
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as arrays:

        def visit(node: dict, prefix: str) -> None:
            if node["kind"] == "dict":
                for name, child in node["items"].items():
                    visit(child, f"{prefix}_{name}")
            elif node["kind"] == "array":
                array = arrays[node["key"]]
                for index in np.ndindex(array.shape):
                    key = prefix + "".join(f"_{axis}" for axis in index)
                    value = float(array[index])
                    ranges[key] = [value, value]
            else:
                raise ValueError("unexpected numeric model checkpoint structure")

        visit(model, "nominal_model")
        tau = arrays[model["items"]["time_constants"]["key"]]
        lower = arrays[model["items"]["command_lower"]["key"]]
        upper = arrays[model["items"]["command_upper"]["key"]]
        ranges["active_motor_time_constant_s"] = [float(tau.min()), 3 * float(tau.max())]
        ranges["physical_command_effort_N"] = [float(lower.min()), float(upper.max())]
    return ranges


def _training_support(inventory: dict[str, Any]) -> dict[str, Any]:
    nominal = {
        "library_seeds": list(LIBRARY_SEEDS),
        "policy_count": 16,
        "nominal_finite_gradient_updates_per_seed": 128,
        "nominal_initialization_and_continuation": {
            "11": "selected absolute-braking preparation32 updates, then exact-Adam continuation96",
            "23": "fresh nominal initialization followed by128 absolute-braking updates",
            "37": "fresh nominal initialization followed by128 absolute-braking updates",
        },
        "nominal_training_state_count": 27,
        "nominal_disjoint_validation_state_count": 16,
        "state_bank_scope": (
            "15 body states plus 12 motor/body transient states; full 17-state observation"
        ),
        "bootstrap_braking_objective": (
            "absolute terminal braking; other declared weights unchanged"
        ),
        "deployment_reference": (
            "freeze nominal128 params/model/gains as teacher; restore reference-braking excess; "
            "preserve all nominal Adam moments and128 counters"
        ),
        "skill_speed_m_s": [0.35, 1.25],
        "skill_duration_s": [0.35, 0.9],
        "prediction_steps": 60,
        "prediction_step_s": 0.02,
        "retention_anchor_batch_size": 2,
        "retention_weight": 5.0,
        "actor_and_loss_information": (
            "proprioception/model and immutable behavior targets; no goals or obstacles"
        ),
        "evidence": {
            "nominal": _relative(STUDY / "M3_BEHAVIOR_SUMMARY.json"),
            "recovery": _relative(STUDY / "M4_RECOVERY_THREE_SEED_SUMMARY.json"),
        },
    }
    dr = {
        "library_seeds": list(LIBRARY_SEEDS),
        "policy_count": 16,
        "common_nominal_teacher_preparation": copy.deepcopy(nominal),
        "nominal_resources_role": (
            "same per-seed immutable nominal128 teacher; DR actor is separately initialized"
        ),
        "dr_finite_gradient_updates_per_seed": 512,
        "dr_initialization": "fresh actor and fresh Adam, independently for each seed 11/23/37",
        "training_state_count": 27,
        "disjoint_validation_state_count": 16,
        "retention_anchor_batch_size": 2,
        "retention_weight": 5.0,
        "braking_objective": "excess over the immutable nominal teacher",
        "training_current_models": 64,
        "disjoint_validation_current_models": 4,
        "effectiveness_support": [0.70, 1.0],
        "lag_multiplier_support": [1.0, 3.0],
        "validation_update_counts": [0, 128, 256, 384, 512],
        "deployed_update_count_by_seed": {str(seed): 512 for seed in LIBRARY_SEEDS},
        "selection": "prespecified validation score; no main-test state or outcome",
        "student_integration_steps_per_seed": 1474560,
        "integration_count_scope": (
            "512 updates * (1 current + 2 retention states) * 16 policies * 60 steps; "
            "excludes teacher setup"
        ),
        "dr_evidence": _relative(STUDY / "M3_DR_THREE_SEED_SUMMARY.json"),
        "online_actor_updates": "none; frozen DR512 at deployment",
    }
    return {
        "F2": {**copy.deepcopy(nominal), "online_actor_updates": "none; frozen nominal128"},
        "A": {
            **copy.deepcopy(nominal),
            "online_actor_updates": (
                "persistent Adam from nominal128; finite updates without quality admission; "
                "deterministic opportunities; completed snapshots publish at later boundaries"
            ),
            "initial_checkpoint_provenance": inventory["A"],
        },
        "DR": dr,
        "OPT": {
            "offline_optimizer_training_updates": 0,
            "common_nominal_resources": copy.deepcopy(nominal),
            "online_actor_updates": "none; bounded direct-command optimization",
            "control_knots": 6,
            "max_iterations": 80,
            "wall_time_budget_s": 0.04,
            "finalization_reserve_fraction": 0.10,
            "function_tolerance": 1e-7,
            "command_change_weight": 0.005,
            "effort_weight": 0.0001,
            "evasion_acceleration_m_s2": 2.0,
            "brake_gain": 2.0,
            "initializations": ["nominal", "brake", "left", "right", "up"],
            "previous_plan": "shifted and revalidated on every solve when available",
            "development_pilot": {
                "path": (
                    "artifacts/da_plcbf_actuator/opt_cpu_development_v1/"
                    "final_shared_geometry/summary.json"
                ),
                "case_names": ["clear_hover", "future_vertical_collision"],
                "service_budgets_s": [0.04, 0.20],
                "one_decision_runs": 4,
                "scope": (
                    "CPU service/implementation microbenchmark with0.025m ego geometry; "
                    "not physical-collider episodes or repeated deadline evidence. "
                    "The0.20s sensitivity is not a40ms deployment result."
                ),
                "main_budget_choice": "native40ms control period; no superiority selection claim",
            },
            "shared_physical_geometry_nominal_sanity": _relative(STUDY / "gate-d-original-v1"),
        },
    }


def prepare(output: Path, specifications: Path) -> dict[str, Any]:
    # Geometry resolution and NPZ/JSON verification are CPU-only and do not compile a controller.
    os.environ["JAX_PLATFORMS"] = "cpu"
    from benchmark.da_plcbf_actuator_protocol import PRIMARY_METRICS, _split_support
    from benchmark.da_plcbf_actuator_study import runner_source_files
    from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
    from crazyflow.safety.da_plcbf.actuator_protocol import content_sha256

    common = ActuatorFilterConfig()
    if (common.dt, common.horizon, common.command_period, common.gradient_mode) != (
        0.02,
        60,
        0.04,
        "forward",
    ):
        raise ValueError("current filter defaults differ from the declared original configuration")
    campaign = _campaign(common)
    inventory = _checkpoint_inventory(campaign)
    supports, collection = _support_inputs(specifications.resolve(), common)
    resolved = {
        **{name: _split_support(value, common, name) for name, value in supports.items()},
        "test": _split_support(campaign, common, "test"),
    }
    seen_seeds, seen_physics = set(), set()
    for split in ("development", "validation", "test"):
        seeds = set(resolved[split]["world_seeds"])
        identities = [world["world_id"] for world in resolved[split]["worlds"]]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate physical worlds within {split}")
        if seeds & seen_seeds or set(identities) & seen_physics:
            raise ValueError("development/validation/test world seeds and physics must be disjoint")
        seen_seeds.update(seeds)
        seen_physics.update(identities)
    counts = Counter(world["cell"] for world in resolved["test"]["worlds"])
    if len(counts) != 8 or set(counts.values()) != {REPLICATES}:
        raise ValueError("main input must contain four physical worlds in each of eight cells")
    episodes = len(campaign["worlds"]) * len(LIBRARY_SEEDS) * len(METHODS)
    if episodes != 384 or len(resolved["test"]["world_seeds"]) != 32:
        raise ValueError("declared main matrix or independent geometry count changed")
    ranges, generator = _generator_support()
    ranges.update(_nominal_numeric_ranges(ROOT / campaign["checkpoints"]["A"]["11"]))
    pilot_path = STUDY / "M4_DEVELOPMENT_PILOT_SUMMARY.json"
    pilot = json.loads(pilot_path.read_text())
    if pilot["completed"] != 32:
        raise ValueError("budget basis must be the completed32-episode development pilot")
    pilot_seconds = float(pilot["wall_seconds_including_warmup"])
    projected_minutes = pilot_seconds / 32 * episodes / 60
    z = NormalDist().inv_cdf(0.975)
    zero_event_upper = z * z / (REPLICATES + z * z)
    metrics = list(PRIMARY_METRICS)
    comparison_count = 3 * len(metrics) * len(counts)
    if comparison_count != 120:
        raise ValueError("primary multiplicity must cover all3*5*8=120 comparisons")
    decisions = {
        "protocol_id": "actuator-study-20260906-mini-main384-v1",
        "numeric_ranges": ranges,
        "training_support": _training_support(inventory),
        "uncertainty": {
            "primary_model_information": (
                "oracle current active effectiveness/time constants; no future faults"
            ),
            "prediction": "one current point model frozen through every predicted rollout",
            "primary_observation_error": "zero state noise, zero model bias, zero parameter delay",
            "generator_support": generator,
            "secondary_scope": (
                "separately declared validation-only model/integration/bias/noise/timing and "
                "A1/retention variants; descriptive only; no tuning on main outcomes"
            ),
            "sampling": {
                "physical_worlds_per_cell": REPLICATES,
                "independent_physical_worlds_total": 32,
                "library_seeds": list(LIBRARY_SEEDS),
                "methods": list(METHODS),
                "episode_count": episodes,
                "zero_event_world_wilson95_upper_with_four_worlds": zero_event_upper,
                "precision_scope": (
                    "only four independent worlds per cell; library repeats are not new worlds"
                ),
            },
        },
        "analysis": {
            "confidence_level": 0.95,
            "n_resamples": 20000,
            "resampling_seed": 20260906,
            "cluster_axes": ["world", "library_seed"],
            "interval": "percentile",
            "primary_multiplicity": "bonferroni",
            "secondary_multiplicity": "descriptive_only",
            "missing_results": "refuse_incomplete_matrix",
        },
        "budget_rationale": (
            f"A bounded mini benchmark with32 independently generated physical worlds: "
            f"4 per family/dynamics cell across8 cells, crossed with3 library seeds and4 methods "
            f"for384 episodes. The completed clean32-episode development pilot took "
            f"{pilot_seconds:.2f}s including warmup; proportional planning projection is "
            f"{projected_minutes:.2f}min, approximately90min, with different OPT service and "
            f"episode lengths limiting that estimate. Four independent worlds per cell imply "
            f"wide precision: even zero events has a two-sided95% Wilson upper bound "
            f"{zero_event_upper:.4f} (about0.49) at the world level. This is not the5760-episode "
            f"default, a power claim, or a test-tuned budget. All120 primary A-minus-F2/DR/OPT "
            f"metric/cell comparisons receive Bonferroni correction; "
            f"incomplete matrices are refused."
        ),
    }
    # Verify the inventory did not change while resolving this draft snapshot.
    current_paths = sorted(specifications.resolve().rglob("*.json"))
    current_hashes = {_relative(path): _sha256(path) for path in current_paths}
    if current_hashes != collection["source_file_sha256"]:
        raise ValueError(
            "specifications changed during preparation; rerun on one consistent snapshot"
        )
    source_paths = [*runner_source_files(ROOT), Path(__file__).resolve()]
    verification = {
        "status": "structurally_verified_draft_inputs_only",
        "controller_learner_or_plant_calls": 0,
        "protocol_builder_called": False,
        "sealed": False,
        "source_commit_policy": (
            "filled by committed_sources in the later explicit protocol builder"
        ),
        "source_file_sha256": {_relative(path): _sha256(path) for path in source_paths},
        "checkpoint_inventory": inventory,
        "unique_physical_worlds_by_split": {
            split: len(value["worlds"]) for split, value in resolved.items()
        },
        "world_seeds_by_split": {split: value["world_seeds"] for split, value in resolved.items()},
        "library_seeds_by_split": {
            split: value["library_seeds"] for split, value in resolved.items()
        },
        "cross_split_physics_and_generator_seeds_disjoint": True,
        "test_world_count_by_cell": dict(sorted(counts.items())),
        "planned_main_episodes": episodes,
        "primary_metrics": metrics,
        "primary_methods_compared_to_A": ["F2", "DR", "OPT"],
        "primary_comparison_count": comparison_count,
        "bonferroni_per_comparison_confidence": 1 - 0.05 / comparison_count,
        "budget_pilot_path": _relative(pilot_path),
        "budget_pilot_sha256": _sha256(pilot_path),
        "pilot_seconds": pilot_seconds,
        "projected_main_minutes": projected_minutes,
        "input_content_sha256": {
            "main_campaign": content_sha256(campaign),
            "decisions": content_sha256(decisions),
            **{split: content_sha256(value) for split, value in supports.items()},
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "main-campaign.json", campaign)
    _write(output / "decisions.json", decisions)
    _write(output / "development.json", supports["development"])
    _write(output / "validation.json", supports["validation"])
    _write(output / "support-provenance.json", collection)
    _write(output / "resolved-support.json", resolved)
    _write(output / "verification.json", verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--specifications", type=Path, default=STUDY / "specifications")
    args = parser.parse_args()
    report = prepare(args.output.resolve(), args.specifications.resolve())
    print(json.dumps({"output": str(args.output.resolve()), **report}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
