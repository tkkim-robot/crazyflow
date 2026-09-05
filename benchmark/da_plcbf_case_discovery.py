"""Obstacle-independent behavior atlas and bounded, all-candidate encounter search.

Atlas snapshots come from a recorded shared nominal hover prefix with persistent wind.
Geometry is chosen only after those snapshots exist. This is selected-case development,
not an unbiased campaign or evidence of paced online availability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.da_plcbf_hover_mechanism_probe import trajectory_metrics
from crazyflow.safety.da_plcbf.continuous_version_a import (
    ContinuousVersionAConfig,
    obstacle_agnostic_emergency_wrench,
    rollout_waypoint_library,
)
from crazyflow.safety.da_plcbf.learner_checkpoint import save_learner_checkpoint
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig, waypoint_nominal_wrench
from crazyflow.safety.da_plcbf.quad_rollouts import (
    direct_wrench_symplectic_step,
    zero_order_hold_rollout,
)
from crazyflow.safety.da_plcbf.state_conditioned_learning import (
    build_reference_skill_learner_from_checkpoint,
    reference_contract_checkpoint_metadata,
    save_reference_contract,
)

SOURCE = Path("artifacts/da_plcbf/hover-explanation-20260905/learning/hover-probe-1")
CASE_CHECKPOINTS = {
    "uncompensated": SOURCE / "restoration_uncompensated_gated/initial_checkpoint",
    "compensated": SOURCE / "restoration_compensated/initial_checkpoint",
    "ungated": SOURCE / "restoration_uncompensated_full_residual/initial_checkpoint",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_atlas_branch_snapshot(bundle: Any, branch_time: float) -> float:
    """Bind the branch clock to the recorded physical state and available snapshot time."""
    available = bundle.metadata.get("available_time_seconds")
    trained_before = bundle.metadata.get("training_before_seconds")
    if (
        available is None
        or trained_before is None
        or not np.isfinite((available, trained_before, branch_time)).all()
        or available < 0
        or trained_before < 0
        or trained_before > available
        or not np.isclose(available, branch_time, rtol=0, atol=1e-9)
    ):
        raise ValueError("branch clock must match the authenticated atlas state/snapshot time")
    return float(available)


def cached_swept_values(
    positions: np.ndarray,
    centers: np.ndarray,
    effective_radii: np.ndarray,
    *,
    policy_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact relative-chord minimum, batched over geometry and cached ego policies.

    positions: (K,T,3); centers: (B,T,O,3); radii: (B,O).
    Includes both endpoints, therefore the shared initial state and nominal candidate.
    Invalid policy rows have value and clearance -inf, never a positive geometry certificate.
    """
    positions, centers, effective_radii = map(np.asarray, (positions, centers, effective_radii))
    if positions.ndim != 3 or positions.shape[-1] != 3 or positions.shape[1] < 2:
        raise ValueError("positions must have shape (policies, at least two nodes, 3)")
    if (
        centers.ndim != 4
        or centers.shape[1] != positions.shape[1]
        or centers.shape[-1] != 3
        or centers.shape[2] < 1
        or effective_radii.shape != (centers.shape[0], centers.shape[2])
        or not np.isfinite(centers).all()
        or not np.isfinite(effective_radii).all()
        or np.any(effective_radii < 0)
    ):
        raise ValueError("geometry must have matching finite nodes and nonnegative radii")
    valid = np.all(np.isfinite(positions), axis=(1, 2))
    if policy_valid is not None:
        policy_valid = np.asarray(policy_valid)
        if policy_valid.shape != valid.shape or policy_valid.dtype != bool:
            raise ValueError("policy_valid must be a boolean vector matching the policy axis")
        valid &= policy_valid
    positions = np.where(valid[:, None, None], positions, 0)
    relative = positions[None, :, :, None] - centers[:, None]
    start = relative[:, :, :-1]
    delta = np.diff(relative, axis=2)
    denominator = np.sum(delta * delta, axis=-1)
    fraction = np.clip(
        -np.sum(start * delta, axis=-1) / np.where(denominator > 0, denominator, 1), 0, 1
    )
    closest = start + fraction[..., None] * delta
    distances = np.sqrt(np.sum(closest**2, axis=-1))
    values = distances**2 - effective_radii[:, None, None] ** 2
    return np.where(valid[None], np.min(values, axis=(2, 3)), -np.inf), np.where(
        valid[None], np.min(distances - effective_radii[:, None, None], axis=(2, 3)), -np.inf
    )


def require_valid_atlas_anchor(data: Any, anchor: dict) -> None:
    """Fail closed when an atlas cannot substantiate its unmasked complete library.

    The bounded geometry search currently requires all atlas policies to be valid. New atlases
    retain their per-policy masks; older ones must have the contemporaneously recorded all-valid
    competency check. This is a diagnostic input requirement, never a learner admission gate.
    """
    key = anchor["key"]
    for method in ("fixed", "adaptive"):
        mask_key = f"{key}_{method}_valid"
        if mask_key in data:
            mask = np.asarray(data[mask_key])
            valid = (
                mask.dtype == bool
                and mask.shape == (data[f"{key}_{method}"].shape[0],)
                and bool(mask.all())
            )
        else:
            valid = (
                anchor["methods"][method]["competency"]["competency_checks"].get(
                    "all_rollouts_finite_and_actuator_valid", False
                )
                is True
            )
        if not valid or not np.isfinite(data[f"{key}_{method}"]).all():
            raise ValueError("geometry search requires a verified all-valid atlas library")
    for shared in ("nominal", "emergency"):
        mask_key = f"{key}_{shared}_valid"
        if not np.isfinite(data[f"{key}_{shared}"]).all() or (
            mask_key in data and not np.asarray(data[mask_key]).all()
        ):
            raise ValueError("geometry search requires valid shared nominal/emergency rollouts")


def make_atlas(output: Path, cases: list[str], device: jax.Device) -> None:
    output.mkdir(parents=True, exist_ok=False)
    period, dt = 0.04, 0.02
    anchors = [3.0, 4.0, 5.0, 7.0, 9.0]
    protocol = {
        "scope": "obstacle-free controlled prefix; geometry discovery happens afterward",
        "wind_onset_seconds": 3.0,
        "wind_velocity": [1.6, 0.8, 0.0],
        "snapshot_times_seconds": anchors,
        "integration_dt": dt,
        "command_hold_seconds": period,
        "learning": "one complete finite update each period, published at next boundary",
        "cases": cases,
        "source_commit": "c653e0b522654afd547a43bc93d7f74b545c6a08",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "geometry_or_goal_enters_learner": False,
    }
    write_json(output / "protocol.json", protocol)
    for case in cases:
        directory = output / case
        directory.mkdir()
        bundle, contract, learner = build_reference_skill_learner_from_checkpoint(
            CASE_CHECKPOINTS[case], device=device
        )
        save_reference_contract(contract, directory / "nominal_reference")
        binding = reference_contract_checkpoint_metadata(directory / "nominal_reference")
        nominal_model = contract.model
        wind_model = nominal_model._replace(
            wind_velocity=jax.device_put(jnp.asarray(protocol["wind_velocity"]), device)
        )
        x = jax.device_put(jnp.asarray([0, 0, 1.4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.0]), device)
        goal = x[:3]
        quad = QuadPolicyConfig(acceleration_limit=1.2)
        nominal_action = jax.jit(
            lambda state, model: (
                waypoint_nominal_wrench(
                    state,
                    goal,
                    jnp.zeros(3),
                    model,
                    bundle.actuator,
                    quad,
                    position_gain=2.0,
                    velocity_gain=2.8,
                    model_compensation=True,
                ).wrench
            )
        )
        nominal_rollout = jax.jit(
            lambda state, model: rollout_waypoint_library(
                state,
                goal[None],
                jnp.zeros((1, 3)),
                model,
                bundle.actuator,
                quad,
                dt=dt,
                horizon=bundle.config.horizon,
                position_gain=2.0,
                velocity_gain=2.8,
                model_compensation=True,
                command_hold_steps=2,
            )
        )
        runtime = ContinuousVersionAConfig(
            dt=dt, horizon=bundle.config.horizon, control_interval_steps=2
        )

        @jax.jit
        def emergency_rollout(state: Any, model: Any) -> tuple[Any, Any, Any]:
            future, commands = zero_order_hold_rollout(
                state,
                lambda current, _: obstacle_agnostic_emergency_wrench(
                    current, model, bundle.actuator, runtime
                ),
                model,
                dt=dt,
                horizon=bundle.config.horizon,
                command_hold_steps=2,
            )
            return jnp.concatenate((state[None], future)), commands[0], jnp.all(commands[1])

        plant = jax.jit(
            lambda state, action, model: direct_wrench_symplectic_step(state, action, model, dt)
        )
        persistent = bundle.state
        fixed = persistent.params
        arrays, diagnostics, updates = {}, [], []
        states, actions = [np.asarray(x)], []
        for index in range(round(anchors[-1] / period) + 1):
            when = round(index * period, 10)
            model = nominal_model if when < 3 else wind_model
            if when in anchors:
                reference = learner.rollout(contract.params, x, nominal_model)
                fixed_rollout = learner.rollout(fixed, x, model)
                adaptive_rollout = learner.rollout(persistent.params, x, model)
                nominal = nominal_rollout(x, model)
                emergency, emergency_wrenches, emergency_valid = emergency_rollout(x, model)
                jax.block_until_ready(
                    (reference, fixed_rollout, adaptive_rollout, nominal, emergency)
                )
                key = f"t{index:04d}"
                save_learner_checkpoint(
                    persistent,
                    bundle.spec,
                    bundle.config,
                    bundle.actuator,
                    x,
                    directory / key,
                    metadata={
                        **binding,
                        "available_time_seconds": when,
                        "training_before_seconds": when,
                        "prefix_kind": "shared obstacle-free compensated nominal hover",
                    },
                )
                arrays[f"{key}_state"] = np.asarray(x)
                arrays[f"{key}_reference"] = np.asarray(reference.states)
                arrays[f"{key}_nominal"] = np.asarray(nominal.states)
                arrays[f"{key}_nominal_valid"] = np.asarray(nominal.valid)
                arrays[f"{key}_emergency"] = np.asarray(emergency)[None]
                arrays[f"{key}_emergency_valid"] = np.asarray(
                    [bool(emergency_valid) and np.isfinite(emergency).all()]
                )
                arrays[f"{key}_emergency_wrenches"] = np.asarray(emergency_wrenches)
                row = {
                    "key": key,
                    "time_seconds": when,
                    "completed_wind_updates": max(0, index - 75),
                    "version": int(persistent.library_version),
                    "methods": {},
                }
                for name, result in (("fixed", fixed_rollout), ("adaptive", adaptive_rollout)):
                    arrays[f"{key}_{name}"] = np.asarray(result.states)
                    arrays[f"{key}_{name}_wrenches"] = np.asarray(result.wrenches)
                    arrays[f"{key}_{name}_motors"] = np.asarray(result.bounded_motor_forces)
                    arrays[f"{key}_{name}_valid"] = np.asarray(
                        jnp.all(result.policy_valid, axis=1)
                        & jnp.all(jnp.isfinite(result.states), axis=(1, 2))
                    )
                    row["methods"][name] = trajectory_metrics(
                        result, np.asarray(reference.states), bundle.spec, bundle.config
                    )
                    row["methods"][name]["per_skill_position_rmse_m"] = np.sqrt(
                        np.mean(
                            (
                                np.asarray(result.states[..., :3])
                                - np.asarray(reference.states[..., :3])
                            )
                            ** 2,
                            axis=(1, 2),
                        )
                    ).tolist()
                diagnostics.append(row)
                print(
                    case,
                    when,
                    {k: round(v["tracking_position_rmse_m"], 5) for k, v in row["methods"].items()},
                    flush=True,
                )
            if index == round(anchors[-1] / period):
                break
            action = nominal_action(x, model)
            jax.block_until_ready((action, x, model))
            started = time.perf_counter()
            following, metrics = jax.block_until_ready(learner.step(persistent, x, model))
            updates.append(
                {
                    "training_time": when,
                    "available_time": round(when + period, 10),
                    "version": int(following.library_version),
                    "finite": bool(metrics.finite_update_applied),
                    "loss": float(metrics.loss.total),
                    "service_seconds": time.perf_counter() - started,
                }
            )
            actions.append(np.asarray(action))
            for _ in range(2):
                x = plant(x, action, model)
                states.append(np.asarray(x))
            persistent = following
        arrays["prefix_states"] = np.asarray(states)
        arrays["prefix_wrenches"] = np.asarray(actions)
        np.savez_compressed(directory / "atlas.npz", **arrays)
        write_json(directory / "atlas.json", diagnostics)
        write_json(directory / "updates.json", updates)
        write_json(
            directory / "source.json",
            {
                "checkpoint": str(CASE_CHECKPOINTS[case]),
                "sha256": bundle.sha256,
                "initial_version": int(bundle.state.library_version),
            },
        )


def geometry_search(atlas: Path, output: Path, *, seed: int = 19301, power: int = 9) -> None:
    from scipy.stats import qmc

    output.mkdir(parents=True, exist_ok=False)
    # Fixed before evaluating any geometry. Units of H are m²; these exceed fp32 noise.
    epsilon_fixed, epsilon_adaptive = 0.002, 0.025
    protocol = {
        "seed": seed,
        "candidates_per_anchor": 2**power,
        "fixed_max_H_upper": -epsilon_fixed,
        "adaptive_max_H_lower": epsilon_adaptive,
        "arrival_delay_range_s": [0.4, 1.1],
        "speed_range_m_s": [1.0, 3.0],
        "radius_range_m": [0.25, 0.85],
        "offset_range_m": [-0.15, 0.15],
        "trajectory_amplitude_m": 30,
        "ego_enclosure_m": 0.106,
        "shell_clearance_m": 0.15,
        "selection": "seeded Sobol development search; all outcomes retained",
        "guard_count": 0,
        "atlas": str(atlas),
    }
    write_json(output / "protocol.json", protocol)
    unit = qmc.Sobol(d=8, scramble=True, seed=seed).random_base2(power)
    directions = np.stack(
        (
            np.cos(2 * np.pi * unit[:, 0]),
            np.sin(2 * np.pi * unit[:, 0]),
            0.8 * (2 * unit[:, 1] - 1),
        ),
        axis=-1,
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    delay, speed, radius = 0.4 + 0.7 * unit[:, 2], 1 + 2 * unit[:, 3], 0.25 + 0.6 * unit[:, 4]
    offset = -0.15 + 0.3 * unit[:, 5:8]
    times = np.arange(61) * 0.02
    ledger, arrays = [], {}
    for directory in sorted(atlas.iterdir()):
        if not directory.is_dir() or not (directory / "atlas.npz").exists():
            continue
        with np.load(directory / "atlas.npz") as data:
            for anchor in json.loads((directory / "atlas.json").read_text()):
                if anchor["time_seconds"] == 3:
                    continue
                require_valid_atlas_anchor(data, anchor)
                key = anchor["key"]
                center = data[f"{key}_state"][:3]
                crossing = center[None] + offset
                centers = crossing[:, None, None] + (30 * directions)[:, None, None] * np.sin(
                    (speed / 30)[:, None, None, None]
                    * (times[None, :, None, None] - delay[:, None, None, None])
                )
                # Policy 0 is the shared nominal, 1..K fallbacks, K+1 emergency, K+2 stationary.
                shared = [
                    data[f"{key}_nominal"],
                    data[f"{key}_emergency"],
                    np.broadcast_to(data[f"{key}_state"], (1, 61, 13)),
                ]
                values, clearances = {}, {}
                for method in ("fixed", "adaptive"):
                    paths = np.concatenate((shared[0], data[f"{key}_{method}"], *shared[1:]))
                    values[method], clearances[method] = cached_swept_values(
                        paths[..., :3], centers, (radius + 0.256)[:, None]
                    )
                    arrays[f"{directory.name}_{key}_{method}_H"] = values[method]
                    arrays[f"{directory.name}_{key}_{method}_clearance"] = clearances[method]
                initial = np.linalg.norm(center - centers[:, 0, 0], axis=-1) - radius - 0.256
                for i in range(len(unit)):
                    fixed, adaptive = values["fixed"][i], values["adaptive"][i]
                    fmax, amax = float(max(fixed[:-2])), float(max(adaptive[:-2]))
                    accepted = (
                        initial[i] > 0.05 and fmax < -epsilon_fixed and amax > epsilon_adaptive
                    )
                    ledger.append(
                        {
                            "case": directory.name,
                            "anchor": key,
                            "index": i,
                            "time_seconds": anchor["time_seconds"],
                            "arrival_delay": float(delay[i]),
                            "direction": directions[i].tolist(),
                            "speed": float(speed[i]),
                            "radius": float(radius[i]),
                            "crossing_offset": (crossing[i] - np.array([0, 0, 1.4])).tolist(),
                            "initial_shell_clearance": float(initial[i]),
                            "fixed_H": fmax,
                            "adaptive_H": amax,
                            "fixed_best": int(np.argmax(fixed[:-2])),
                            "adaptive_best": int(np.argmax(adaptive[:-2])),
                            "nominal_blocked": bool(fixed[0] < 0),
                            "emergency_blocked": bool(fixed[-2] < 0),
                            "hover_blocked": bool(fixed[-1] < 0),
                            "stage_b_accepted": bool(accepted),
                            "rejection": None
                            if accepted
                            else (
                                "initially_unsafe"
                                if initial[i] <= 0.05
                                else "fixed_has_certificate"
                                if fmax >= -epsilon_fixed
                                else "adaptive_margin_insufficient"
                            ),
                        }
                    )
    write_json(output / "ledger.json", ledger)
    np.savez_compressed(output / "per_policy_values.npz", **arrays)
    selected = sorted(
        [r for r in ledger if r["stage_b_accepted"]],
        key=lambda r: r["adaptive_H"] - r["fixed_H"],
        reverse=True,
    )
    write_json(output / "selected.json", selected)
    counts = {
        case: sum(r["stage_b_accepted"] for r in ledger if r["case"] == case)
        for case in {r["case"] for r in ledger}
    }
    print("Searched", len(ledger), "Stage B accepted", counts, flush=True)


def refine_geometry(
    atlas: Path, output: Path, *, seed: int = 19302, power: int = 9, physical: bool = False
) -> None:
    """Declared contrastive radius fit after the initial family proved over-constrained."""
    from scipy.stats import qmc

    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "protocol.json",
        {
            "seed": seed,
            "candidates_per_anchor": 2**power,
            "reason": "v1 central large spheres exhausted both repertoires",
            "offset_bounds_m": [-0.6, 0.6],
            "radius_bounds_m": [0.08, 0.85 if physical else 0.6],
            "arrival_delay_bounds_s": [0.5, 1.15],
            "speed_bounds_m_s": [1.0, 3.0],
            "radius_selection": (
                "adaptive maxH=.035 m2 then clipped to bounds; "
                "emergency must intersect actual oriented asset"
                if physical
                else "midpoint of squared fixed/adaptive augmented minimum distances, "
                "clipped to bounds"
            ),
            "H_thresholds_m2": [-0.002, 0.025],
            "required_threats": ["nominal", "emergency", "stationary"],
            "scope": "contrastive selected development; no obstacle inputs to learner",
        },
    )
    unit = qmc.Sobol(d=7, scramble=True, seed=seed).random_base2(power)
    directions = np.stack(
        (
            np.cos(2 * np.pi * unit[:, 0]),
            np.sin(2 * np.pi * unit[:, 0]),
            0.8 * (2 * unit[:, 1] - 1),
        ),
        axis=-1,
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    delay, speed = 0.5 + 0.65 * unit[:, 2], 1 + 2 * unit[:, 3]
    offset = -0.6 + 1.2 * unit[:, 4:7]
    times = np.arange(61) * 0.02
    ledger, arrays = [], {}
    for directory in sorted(atlas.iterdir()):
        if not directory.is_dir() or not (directory / "atlas.npz").exists():
            continue
        with np.load(directory / "atlas.npz") as data:
            for anchor in json.loads((directory / "atlas.json").read_text()):
                if anchor["time_seconds"] == 3:
                    continue
                require_valid_atlas_anchor(data, anchor)
                key = anchor["key"]
                center = data[f"{key}_state"][:3]
                crossing = center[None] + offset
                centers = crossing[:, None, None] + (30 * directions)[:, None, None] * np.sin(
                    (speed / 30)[:, None, None, None]
                    * (times[None, :, None, None] - delay[:, None, None, None])
                )
                squared, asset_squared = {}, {}
                for method in ("fixed", "adaptive"):
                    paths = np.concatenate(
                        (
                            data[f"{key}_nominal"],
                            data[f"{key}_{method}"],
                            data[f"{key}_emergency"],
                            np.broadcast_to(data[f"{key}_state"], (1, 61, 13)),
                        )
                    )
                    squared[method], _ = cached_swept_values(
                        paths[..., :3], centers, np.zeros((len(unit), 1))
                    )
                    qx, qy, qz, qw = np.moveaxis(paths[..., 3:7], -1, 0)
                    asset_centers = paths[..., :3] + 0.02 * np.stack(
                        (
                            2 * (qx * qz + qy * qw),
                            2 * (qy * qz - qx * qw),
                            1 - 2 * (qx * qx + qy * qy),
                        ),
                        axis=-1,
                    )
                    asset_squared[method], _ = cached_swept_values(
                        asset_centers, centers, np.zeros((len(unit), 1))
                    )
                fdist = squared["fixed"][:, :-2].max(axis=1)
                adist = squared["adaptive"][:, :-2].max(axis=1)
                radius = (
                    np.clip(np.sqrt(np.maximum(adist - 0.035, 0)) - 0.256, 0.08, 0.85)
                    if physical
                    else np.clip(np.sqrt((fdist + adist) / 2) - 0.256, 0.08, 0.6)
                )
                values = {m: v - (radius[:, None] + 0.256) ** 2 for m, v in squared.items()}
                initial = np.linalg.norm(center - centers[:, 0, 0], axis=-1) - radius - 0.256
                for method in values:
                    arrays[f"{directory.name}_{key}_{method}_H"] = values[method]
                for i in range(len(unit)):
                    fixed, adaptive = values["fixed"][i], values["adaptive"][i]
                    fmax, amax = float(max(fixed[:-2])), float(max(adaptive[:-2]))
                    threats = fixed[0] < 0 and fixed[-2] < 0 and fixed[-1] < 0
                    fixed_asset = asset_squared["fixed"][i] - (radius[i] + 0.086) ** 2
                    accepted = (
                        initial[i] > 0.05
                        and fmax < -0.002
                        and amax > 0.025
                        and threats
                        and (not physical or fixed_asset[-2] < -0.0001)
                    )
                    ledger.append(
                        {
                            "case": directory.name,
                            "anchor": key,
                            "index": i,
                            "time_seconds": anchor["time_seconds"],
                            "arrival_delay": float(delay[i]),
                            "direction": directions[i].tolist(),
                            "speed": float(speed[i]),
                            "radius": float(radius[i]),
                            "crossing_offset": (crossing[i] - np.array([0, 0, 1.4])).tolist(),
                            "initial_shell_clearance": float(initial[i]),
                            "fixed_H": fmax,
                            "adaptive_H": amax,
                            "fixed_asset_H": float(max(fixed_asset[:-2])),
                            "emergency_asset_H": float(fixed_asset[-2]),
                            "fixed_best": int(np.argmax(fixed[:-2])),
                            "adaptive_best": int(np.argmax(adaptive[:-2])),
                            "nominal_blocked": bool(fixed[0] < 0),
                            "emergency_blocked": bool(fixed[-2] < 0),
                            "hover_blocked": bool(fixed[-1] < 0),
                            "stage_b_accepted": bool(accepted),
                            "rejection": None
                            if accepted
                            else (
                                "initially_unsafe"
                                if initial[i] <= 0.05
                                else "fixed_has_certificate"
                                if fmax >= -0.002
                                else "adaptive_margin_insufficient"
                                if amax <= 0.025
                                else "shared_escape"
                            ),
                        }
                    )
    write_json(output / "ledger.json", ledger)
    np.savez_compressed(output / "per_policy_values.npz", **arrays)
    selected = sorted(
        [r for r in ledger if r["stage_b_accepted"]],
        key=lambda r: -r["fixed_asset_H"] if physical else r["adaptive_H"] - r["fixed_H"],
        reverse=True,
    )
    write_json(output / "selected.json", selected)
    print(
        "Refinement searched",
        len(ledger),
        "accepted",
        {c: sum(r["case"] == c for r in selected) for c in {r["case"] for r in ledger}},
        flush=True,
    )


def screen_full_qps(atlas: Path, selected: Path, output: Path, *, limit: int, device: Any) -> None:
    from benchmark.da_plcbf_case_attribution import encounter_from_row
    from crazyflow.safety.da_plcbf.case_study_world import build_hover_encounter_world
    from crazyflow.safety.da_plcbf.learner_checkpoint import load_learner_checkpoint
    from crazyflow.safety.da_plcbf.navigation_experiment import NavigationExperimentConfig
    from crazyflow.safety.da_plcbf.policy_qp_audit import (
        make_navigation_policy_qp_auditor,
        summarize_policy_qp_audit,
    )

    output.mkdir(parents=True, exist_ok=False)
    selections = json.loads(selected.read_text())
    ledger = []
    for case in sorted({r["case"] for r in selections}):
        source = load_learner_checkpoint(CASE_CHECKPOINTS[case], device=device)
        auditor = None
        for rank, row in enumerate([r for r in selections if r["case"] == case][:limit]):
            bundle = load_learner_checkpoint(atlas / case / row["anchor"], device=device)
            validate_atlas_branch_snapshot(bundle, row["time_seconds"])
            cfg = encounter_from_row(row)
            world = build_hover_encounter_world(
                cfg,
                initial_state=np.asarray(bundle.physical_state),
                initial_time_seconds=row["time_seconds"],
            )
            config = NavigationExperimentConfig(
                navigation_start_seconds=cfg.navigation_start_seconds,
                fallback_mapping="compensated"
                if source.config.model_compensation
                else "matched_uncompensated",
            )
            if auditor is None:
                auditor = make_navigation_policy_qp_auditor(world, source, config)
            prediction = world.obstacle_prediction(
                row["time_seconds"], horizon=bundle.config.horizon
            )
            model = world.dynamics_at(row["time_seconds"], source.point_model).model
            result = {}
            runtime = ContinuousVersionAConfig(
                dt=0.02,
                horizon=60,
                control_interval_steps=2,
                ego_radius=0.106,
                obstacle_clearance=0.15,
            )
            for method, params in (
                ("fixed", source.state.params),
                ("adaptive", bundle.state.params),
            ):
                audit = auditor(
                    bundle.physical_state,
                    params,
                    model,
                    prediction,
                    -1,
                    jnp.asarray(cfg.hover_position),
                )
                result[method] = summarize_policy_qp_audit(audit, prediction, runtime)
                result[method]["cache_runtime_max_H_difference_m2"] = (
                    float(max(np.asarray(audit.runtime.values.values))) - row[f"{method}_H"]
                )
                result[method]["stage_b_runtime_confirmed"] = bool(
                    max(np.asarray(audit.runtime.values.values)) < -0.002
                    if method == "fixed"
                    else max(np.asarray(audit.runtime.values.values)) > 0.025
                )
            filename = f"{case}-{rank:03d}-{row['anchor']}-{row['index']}.json"
            write_json(output / filename, {"selection": row, "methods": result})
            ledger.append(
                {
                    "file": filename,
                    "selection": row,
                    "methods": {
                        name: {
                            "eligible": r["eligible_full_qp_count"],
                            "accepted_full_qps": r["eligible_accepted_held_qp_count"],
                            "runtime": r["runtime"],
                        }
                        for name, r in result.items()
                    },
                }
            )
            write_json(output / "ledger.json", ledger)
            print(
                filename,
                {
                    name: (r["eligible_full_qp_count"], r["eligible_accepted_held_qp_count"])
                    for name, r in result.items()
                },
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("atlas", "geometry", "refine", "refine-physical", "qp"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atlas", type=Path)
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(CASE_CHECKPOINTS),
        default=["uncompensated", "compensated"],
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=19301)
    parser.add_argument("--power", type=int, default=9)
    args = parser.parse_args()
    if args.stage == "atlas":
        make_atlas(args.output_dir, args.cases, jax.devices(args.device)[0])
    elif args.stage == "qp":
        if args.atlas is None or args.selected is None:
            parser.error("qp requires --atlas and --selected")
        screen_full_qps(
            args.atlas,
            args.selected,
            args.output_dir,
            limit=args.limit,
            device=jax.devices(args.device)[0],
        )
    else:
        if args.atlas is None:
            parser.error("geometry requires --atlas")
        if args.stage.startswith("refine"):
            refine_geometry(
                args.atlas,
                args.output_dir,
                seed=args.seed,
                power=args.power,
                physical=args.stage == "refine-physical",
            )
        else:
            geometry_search(args.atlas, args.output_dir, seed=args.seed, power=args.power)


if __name__ == "__main__":
    main()
