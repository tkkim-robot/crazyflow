"""Obstacle-free actuator gain, recovery and control-witness diagnostics.

These experiments hold an explicit proprioceptive state bank fixed. They are not complete
flight episodes, safety claims, or delayed-command simulations. Nominal teacher parameters,
model, gains and initial motor-state correspondence remain immutable throughout every run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.da_plcbf_actuator_behavior import validation_state_bank
from crazyflow.safety.da_plcbf.actuator_dynamics import (
    ActuatorModel,
    applied_wrench,
    augmented_dynamics,
)
from crazyflow.safety.da_plcbf.actuator_learning import (
    ActuatorLearnerCheckpoint,
    ActuatorReferenceContract,
    ActuatorSkillConfig,
    actuator_loss_gradient_contributions,
    actuator_reference_fingerprint,
    build_actuator_skill_learner,
    load_actuator_learner_checkpoint,
    rollout_actuator_skill_library,
    save_actuator_learner_checkpoint,
)

# Original, increased damping, then two reductions of attitude stiffness at that same damping.
GAIN_PAIRS = ((8e-4, 2e-4), (8e-4, 3e-4), (6e-4, 3e-4), (4e-4, 3e-4))
DEVELOPMENT_INDICES = (4, 5, 6, 13, 21, 26)
LEAD_TIMES = (0.04, 0.08, 0.16, 0.32, 0.64, 1.2)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {name: _jsonable(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def changed_model(nominal: ActuatorModel, cell: str) -> ActuatorModel:
    """Current fault parameters only; no schedule or future fault is an actor input."""
    if cell not in {"nominal", "effectiveness", "lag", "combined"}:
        raise ValueError("unknown dynamics cell")
    eta = (0.7, 1.0, 0.85, 1.0) if cell == "combined" else (0.7, 1.0, 1.0, 1.0)
    return nominal._replace(
        effectiveness=jnp.asarray(eta, dtype=nominal.effectiveness.dtype)
        if cell in {"effectiveness", "combined"}
        else nominal.effectiveness,
        time_constants=nominal.time_constants * jnp.asarray([3.0, 1.6, 2.0, 1.0])
        if cell in {"lag", "combined"}
        else nominal.time_constants,
    )


def _gain_config(reference: ActuatorSkillConfig, index: int) -> ActuatorSkillConfig:
    attitude, rate = GAIN_PAIRS[index]
    return replace(
        reference,
        attitude_gain=attitude,
        angular_rate_gain=rate,
        allow_reference_gain_mismatch=(
            attitude != reference.attitude_gain or rate != reference.angular_rate_gain
        ),
    )


def _source_manifest(
    checkpoint: ActuatorLearnerCheckpoint, args: argparse.Namespace
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    names = (
        "benchmark/da_plcbf_actuator_recovery.py",
        "benchmark/da_plcbf_actuator_behavior.py",
        "crazyflow/safety/da_plcbf/actuator_learning.py",
        "crazyflow/safety/da_plcbf/actuator_dynamics.py",
        "crazyflow/safety/da_plcbf/persistent_skill_learner.py",
        "crazyflow/safety/da_plcbf/direct_wrench.py",
    )
    sources = {}
    for name in names:
        payload = (root / name).read_bytes()
        snapshot = Path(args.output) / "source_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open("xb") as stream:
            stream.write(payload)
        sources[name] = hashlib.sha256(payload).hexdigest()
    return {
        "scope": "fixed obstacle-free state-bank diagnostic; no complete flight/safety conclusion",
        "checkpoint_stem": str(checkpoint.json_path.with_suffix("").resolve()),
        "checkpoint_npz_sha256": checkpoint.sha256,
        "initial_library_version": int(checkpoint.state.library_version),
        "reference_sha256": actuator_reference_fingerprint(checkpoint.contract),
        "reference_actor_config": asdict(checkpoint.contract.actor_config),
        "reference_learning_config": asdict(checkpoint.contract.learning_config),
        "arguments": vars(args),
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "sources": sources,
    }


class Evaluator:
    """Reusable numerical probes with dynamic parameter/model inputs and immutable teacher."""

    def __init__(
        self, contract: ActuatorReferenceContract, current_config: ActuatorSkillConfig
    ) -> None:
        """Bind matched current control and a separately immutable nominal teacher."""
        self.contract = contract
        self.config = current_config
        self.actual = jax.jit(
            jax.vmap(
                lambda params, state, model: rollout_actuator_skill_library(
                    params, contract.spec, state, model, current_config
                ),
                in_axes=(None, 0, None),
            )
        )
        self.teacher = jax.jit(
            jax.vmap(
                lambda state: rollout_actuator_skill_library(
                    contract.params, contract.spec, state, contract.model, contract.actor_config
                )
            )
        )
        self.cache: dict[str, Any] = {}

    def evaluate(
        self, params: Any, model: ActuatorModel, states: jax.Array
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Return body-motion, operational and per-skill errors plus complete numeric traces."""
        key = hashlib.sha256(np.asarray(states).tobytes()).hexdigest()
        if key not in self.cache:
            self.cache[key] = self.teacher(states)
        teacher = self.cache[key]
        actual = self.actual(params, states, model)
        jax.block_until_ready((teacher, actual))
        values = np.asarray(actual.states)
        references = np.asarray(teacher.states)
        nodes = sorted(
            {
                max(1, round(self.config.horizon * f))
                for f in self.contract.learning_config.trajectory_fractions
            }
        )
        position_error = values[:, :, nodes, :3] - references[:, :, nodes, :3]
        velocity_error = values[:, :, nodes, 7:10] - references[:, :, nodes, 7:10]
        position_rmse = np.sqrt(np.mean(position_error**2, axis=(-1, -2)))
        velocity_rmse = np.sqrt(np.mean(velocity_error**2, axis=(-1, -2)))
        terminal = np.linalg.norm(values[:, :, -1, 7:10], axis=-1)
        tilt = np.arccos(np.clip(1 - 2 * (values[..., 3] ** 2 + values[..., 4] ** 2), -1, 1))
        rate = np.linalg.norm(values[..., 10:13], axis=-1)
        speed = np.linalg.norm(values[..., 7:10], axis=-1)
        requested = np.asarray(actual.requested_commands)
        saturation = (requested < np.asarray(model.command_lower)) | (
            requested > np.asarray(model.command_upper)
        )
        valid = np.all(np.asarray(actual.policy_valid), axis=-1) & np.all(
            np.isfinite(values), axis=(-1, -2)
        )
        uniform = np.linalg.norm(values[..., :3] - references[..., :3], axis=-1)
        relative = values[..., :3] - values[:, :, :1, :3]
        pairwise = np.sqrt(
            np.mean(np.sum((relative[:, :, None] - relative[:, None, :]) ** 2, axis=-1), axis=-1)
        )
        mask = ~np.eye(values.shape[1], dtype=bool)
        spread = np.mean(pairwise[:, mask], axis=-1) if np.any(mask) else np.zeros(len(states))
        bounds_low, bounds_high = np.array([-4.92, -3.92, 0.23]), np.array([4.92, 3.92, 3.92])
        arena = np.minimum(values[..., :3] - bounds_low, bounds_high - values[..., :3])
        report = {
            "position_rmse_mean_m": float(np.mean(position_rmse)),
            "position_rmse_max_m": float(np.max(position_rmse)),
            "velocity_rmse_mean_mps": float(np.mean(velocity_rmse)),
            "velocity_rmse_max_mps": float(np.max(velocity_rmse)),
            "uniform_position_error_max_m": float(np.max(uniform)),
            "terminal_speed_max_mps": float(np.max(terminal)),
            "tilt_max_rad": float(np.max(tilt)),
            "body_rate_max_rps": float(np.max(rate)),
            "speed_max_mps": float(np.max(speed)),
            "arena_margin_min_m": float(np.min(arena)),
            "saturation_fraction": float(np.mean(saturation)),
            "finite_valid_fraction": float(np.mean(valid)),
            "trajectory_spread_min_m": float(np.min(spread)) if values.shape[1] > 1 else None,
            "prefix_times_s": [node * self.config.dt for node in nodes],
            "position_rmse_by_skill_m": np.sqrt(
                np.mean(position_error**2, axis=(0, 2, 3))
            ).tolist(),
            "velocity_rmse_by_skill_mps": np.sqrt(
                np.mean(velocity_error**2, axis=(0, 2, 3))
            ).tolist(),
            "prefix_position_rmse_by_skill_m": np.sqrt(
                np.mean(position_error**2, axis=(0, 3))
            ).tolist(),
            "prefix_velocity_rmse_by_skill_mps": np.sqrt(
                np.mean(velocity_error**2, axis=(0, 3))
            ).tolist(),
            "uniform_position_error_by_skill_m": np.max(uniform, axis=(0, 2)).tolist(),
            "terminal_speed_by_skill_mps": np.max(terminal, axis=0).tolist(),
            "tilt_by_skill_rad": np.max(tilt, axis=(0, 2)).tolist(),
            "body_rate_by_skill_rps": np.max(rate, axis=(0, 2)).tolist(),
            "position_rmse_by_state_and_skill_m": position_rmse.tolist(),
            "velocity_rmse_by_state_and_skill_mps": velocity_rmse.tolist(),
            "operational_pass": bool(
                np.all(valid)
                and np.max(tilt) <= 0.9
                and np.max(rate) <= 12
                and np.max(speed) <= 3.5
                and np.min(arena) >= 0
            ),
            "tracking_and_braking_pass": bool(
                np.all(valid)
                and np.max(position_rmse) <= 0.3
                and np.max(velocity_rmse) <= 0.6
                and np.max(terminal) <= 0.8
            ),
        }
        arrays = {
            name: np.asarray(value) for name, value in zip(actual._fields, actual, strict=True)
        }
        arrays.update(
            {
                "reference_states": references,
                "reference_commands": np.asarray(teacher.commands),
                "initial_states": np.asarray(states),
            }
        )
        return report, arrays


def _save_evaluation(
    output: Path,
    name: str,
    evaluator: Evaluator,
    params: Any,
    model: ActuatorModel,
    states: jax.Array,
) -> dict[str, Any]:
    report, arrays = evaluator.evaluate(params, model, states)
    _write_json(output / f"{name}.json", report)
    with (output / f"{name}.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return report


def _brief(report: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in report.items() if isinstance(value, (float, bool))}


def fault_jump_audit(
    states: jax.Array, nominal: ActuatorModel, model: ActuatorModel
) -> dict[str, Any]:
    """Quantify the unavoidable instantaneous jump, without asserting finite-time infeasibility."""
    before_wrench = applied_wrench(states[:, 13:17], nominal)
    after_wrench = applied_wrench(states[:, 13:17], model)
    lower = jnp.broadcast_to(model.command_lower, (len(states), 4))
    upper = jnp.broadcast_to(model.command_upper, (len(states), 4))
    before = augmented_dynamics(states, lower, nominal)
    after_lower = augmented_dynamics(states, lower, model)
    after_upper = augmented_dynamics(states, upper, model)
    delta = np.asarray(after_lower - before)
    return {
        "scope": (
            "instantaneous body acceleration cannot change through command while x,s are fixed; "
            "this is not a finite-horizon infeasibility proof"
        ),
        "before_wrench": np.asarray(before_wrench).tolist(),
        "after_wrench": np.asarray(after_wrench).tolist(),
        "wrench_jump": np.asarray(after_wrench - before_wrench).tolist(),
        "linear_acceleration_jump_mps2": delta[:, 7:10].tolist(),
        "body_angular_acceleration_jump_rps2": delta[:, 10:13].tolist(),
        "maximum_linear_acceleration_jump_mps2": float(
            np.max(np.linalg.norm(delta[:, 7:10], axis=-1))
        ),
        "maximum_body_angular_acceleration_jump_rps2": float(
            np.max(np.linalg.norm(delta[:, 10:13], axis=-1))
        ),
        "lower_vs_upper_command_instant_body_derivative_max_difference": float(
            jnp.max(jnp.abs(after_lower[:, :13] - after_upper[:, :13]))
        ),
        "initial_states_unchanged": np.asarray(states).tolist(),
    }


def _development_score(report: dict[str, Any]) -> float:
    if report["finite_valid_fraction"] < 1:
        return math.inf
    return (
        5 * report["position_rmse_mean_m"] ** 2
        + report["velocity_rmse_mean_mps"] ** 2
        + 4 * max(0, report["terminal_speed_max_mps"] - 0.8) ** 2
        + 10 * max(0, report["tilt_max_rad"] - 0.9) ** 2
        + max(0, report["body_rate_max_rps"] / 12 - 1) ** 2
        + max(0, report["speed_max_mps"] / 3.5 - 1) ** 2
        + max(0, -report["arena_margin_min_m"]) ** 2
    )


def run_gain_sweep(
    args: argparse.Namespace, checkpoint: ActuatorLearnerCheckpoint, output: Path
) -> dict[str, Any]:
    contract = checkpoint.contract
    nominal = contract.model
    combined = changed_model(nominal, "combined")
    bank = contract.anchors
    manifest = _source_manifest(checkpoint, args)
    manifest.update(
        {
            "gain_pairs": GAIN_PAIRS,
            "selection_rule": (
                "first require nominal operational and tracking/braking competence; then minimize "
                "combined development score + .25 nominal score, with original retained"
            ),
            "development_score": (
                "5 mean_position_rmse^2 + mean_velocity_rmse^2 + 4 terminal_excess(.8)^2 + "
                "10 tilt_excess(.9)^2 + normalized rate/speed excess + arena violation"
            ),
            "selection_states": "saved 27-state development bank only; validation held aside",
        }
    )
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "fault_jump.json", fault_jump_audit(bank, nominal, combined))
    records = []
    for index in range(len(GAIN_PAIRS)):
        config = _gain_config(checkpoint.config, index)
        evaluator = Evaluator(contract, config)
        results = {}
        for name, model in (("nominal", nominal), ("combined", combined)):
            results[name] = _save_evaluation(
                output,
                f"gain_{index}_{name}_development",
                evaluator,
                checkpoint.state.params,
                model,
                bank,
            )
        nominal_pass = (
            results["nominal"]["operational_pass"]
            and results["nominal"]["tracking_and_braking_pass"]
        )
        score = _development_score(results["combined"]) + 0.25 * _development_score(
            results["nominal"]
        )
        record = {
            "gain_index": index,
            "current_actor_config": asdict(config),
            "nominal_pass": nominal_pass,
            "selection_score": score,
            "development": results,
        }
        records.append(record)
        print(
            json.dumps(
                _jsonable(
                    {
                        "event": "gain_probe",
                        "gain_index": index,
                        "nominal_pass": nominal_pass,
                        "selection_score": score,
                        "nominal": _brief(results["nominal"]),
                        "combined": _brief(results["combined"]),
                    }
                )
            ),
            flush=True,
        )
    selected = min(
        records, key=lambda record: (not record["nominal_pass"], record["selection_score"])
    )
    config = _gain_config(checkpoint.config, selected["gain_index"])
    validation, labels = validation_state_bank(nominal)
    validation_results = {}
    evaluator = Evaluator(contract, config)
    for cell in ("nominal", "effectiveness", "lag", "combined"):
        validation_results[cell] = _save_evaluation(
            output,
            f"selected_{cell}_validation",
            evaluator,
            checkpoint.state.params,
            changed_model(nominal, cell),
            validation,
        )
    dr_results = {}
    if args.dr_checkpoint:
        dr = load_actuator_learner_checkpoint(args.dr_checkpoint)
        if actuator_reference_fingerprint(dr.contract) != actuator_reference_fingerprint(contract):
            raise ValueError("gain comparison DR must use the same nominal teacher as its seed")
        dr_evaluator = Evaluator(dr.contract, config)
        for cell in ("nominal", "effectiveness", "lag", "combined"):
            dr_results[cell] = _save_evaluation(
                output,
                f"selected_DR_{cell}_validation",
                dr_evaluator,
                dr.state.params,
                changed_model(nominal, cell),
                validation,
            )
    summary = {
        "selected": selected,
        "all_candidates": records,
        "validation_labels": labels,
        "frozen_F2_validation": validation_results,
        "frozen_DR_validation": dr_results,
        "reference_sha256": actuator_reference_fingerprint(contract),
    }
    _write_json(output / "summary.json", summary)
    print(
        json.dumps(
            _jsonable(
                {
                    "event": "gain_selection",
                    "gain_index": selected["gain_index"],
                    "F2_combined_validation": _brief(validation_results["combined"]),
                    "DR_combined_validation": _brief(dr_results["combined"])
                    if dr_results
                    else None,
                }
            )
        ),
        flush=True,
    )
    return summary


def _recovery_plots(
    output: Path,
    evaluations: list[dict[str, Any]],
    frozen: dict[str, Any],
    dr: dict[str, Any] | None,
) -> None:
    """Show all skill errors and operational metrics without merging independent states."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    updates = np.array([row["adaptation_updates"] for row in evaluations])
    elapsed = np.array([row["available_after_seconds"] for row in evaluations])
    skill_count = len(evaluations[0]["validation"]["position_rmse_by_skill_m"])
    colors = plt.get_cmap("tab20")(np.linspace(0, 0.95, skill_count))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    definitions = (
        ("position_rmse_by_skill_m", "Prefix position RMSE", "m", None),
        ("velocity_rmse_by_skill_mps", "Prefix velocity RMSE", "m/s", None),
        ("terminal_speed_by_skill_mps", "Maximum terminal speed", "m/s", 0.8),
        ("tilt_by_skill_rad", "Maximum tilt", "rad", 0.9),
        ("body_rate_by_skill_rps", "Maximum body rate", "rad/s", 12),
        ("uniform_position_error_by_skill_m", "Maximum trajectory position error", "m", None),
    )
    for axis, (name, title, units, threshold) in zip(axes.flat, definitions, strict=True):
        data = np.asarray([entry["validation"][name] for entry in evaluations])
        for skill in range(skill_count):
            axis.plot(updates, data[:, skill], color=colors[skill], lw=1.25, label=str(skill))
        if threshold is not None:
            axis.axhline(threshold, color="black", ls="--", lw=1)
        axis.set(title=title, xlabel="Completed adaptation updates", ylabel=units)
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=16, title="Fixed skill identity", frameon=False
    )
    fig.suptitle("Recovery on held-out fixed motion/motor states; no flight episode")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(output / "per_skill_recovery.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for axis, name, ylabel in (
        (axes[0], "position_rmse_mean_m", "Mean prefix position RMSE (m)"),
        (axes[1], "velocity_rmse_mean_mps", "Mean prefix velocity RMSE (m/s)"),
    ):
        axis.plot(
            elapsed, [row["validation"][name] for row in evaluations], "o-", label="Persistent A"
        )
        axis.axhline(frozen[name], color="tab:orange", ls="--", label="Frozen F2")
        if dr is not None:
            axis.axhline(dr[name], color="tab:green", ls=":", label="Frozen DR + F2")
        axis.set(xlabel="Actual completed learner-only elapsed time (s)", ylabel=ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Synchronized learning only; prewarm/probes excluded, no controller contention")
    fig.tight_layout()
    fig.savefig(output / "completed_time_recovery.png", dpi=160)
    plt.close(fig)

    prefixes = np.asarray(evaluations[-1]["validation"]["prefix_position_rmse_by_skill_m"])
    panels = [
        ("Frozen F2", np.asarray(frozen["prefix_position_rmse_by_skill_m"])),
        ("Persistent A", prefixes),
    ]
    if dr is not None:
        panels.insert(1, ("Frozen DR + F2", np.asarray(dr["prefix_position_rmse_by_skill_m"])))
    upper = max(float(np.max(values)) for _, values in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 5), squeeze=False)
    for axis, (name, values) in zip(axes[0], panels, strict=True):
        handle = axis.imshow(values, aspect="auto", vmin=0, vmax=upper, cmap="magma")
        axis.set(title=name, xlabel="Prefix time (s)", ylabel="Skill identity")
        axis.set_xticks(range(values.shape[1]), labels=frozen["prefix_times_s"])
        axis.set_yticks(range(skill_count))
    fig.colorbar(
        handle, ax=axes[0].tolist(), label="Position RMSE over held-out states (m)", fraction=0.025
    )
    fig.savefig(output / "prefix_recovery.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _gradient_plot(output: Path, initial: dict[str, Any], final: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(initial["gradient_components"])
    first = [initial["gradient_components"][name]["gradient_norm"] for name in names]
    last = [final["gradient_components"][name]["gradient_norm"] for name in names]
    positions = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(positions - 0.18, np.maximum(first, 1e-12), height=0.35, label="Initial")
    axis.barh(positions + 0.18, np.maximum(last, 1e-12), height=0.35, label="Final")
    axis.set_yticks(positions, names)
    axis.set(
        xscale="log",
        xlabel="Raw component gradient norm",
        title="Same current state; recorded rotating-anchor iteration",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "gradient_contributions.png", dpi=160)
    plt.close(fig)


def run_adaptation(
    args: argparse.Namespace,
    checkpoint: ActuatorLearnerCheckpoint,
    output: Path,
    compiled: dict[str, tuple[Any, Evaluator]] | None = None,
) -> dict[str, Any]:
    """Publish every finite persistent step, then probe saved completed snapshots afterward."""
    contract = checkpoint.contract
    if args.revision == "retention10":
        contract = replace(
            contract, learning_config=replace(contract.learning_config, retention_weight=10.0)
        )
    elif args.revision == "braking_huber":
        contract = replace(
            contract,
            learning_config=replace(contract.learning_config, reference_braking_huber_delta=0.01),
        )
    gain_index = 0
    if args.gain_report:
        gain_index = int(json.loads(Path(args.gain_report).read_text())["selected"]["gain_index"])
    config = _gain_config(checkpoint.config, gain_index)
    if args.unroll is not None:
        config = replace(config, rollout_scan_unroll=args.unroll)
    nominal = contract.model
    model = changed_model(nominal, args.cell)
    training = contract.anchors[jnp.asarray(DEVELOPMENT_INDICES)]
    validation, labels = validation_state_bank(nominal)
    original_reference = actuator_reference_fingerprint(checkpoint.contract)
    initial = checkpoint.state
    initial_version = int(initial.library_version)
    manifest = _source_manifest(checkpoint, args)
    manifest.update(
        {
            "current_actor_config": asdict(config),
            "training_indices_in_saved_bank": DEVELOPMENT_INDICES,
            "training_states": np.asarray(training).tolist(),
            "validation_labels": labels,
            "validation_bank_sha256": hashlib.sha256(np.asarray(validation).tobytes()).hexdigest(),
            "current_effectiveness": np.asarray(model.effectiveness).tolist(),
            "current_time_constants_s": np.asarray(model.time_constants).tolist(),
            "active_reference_sha256": actuator_reference_fingerprint(contract),
            "timing_scope": (
                "Uninterrupted synchronized learning from fixed virtual states; reference "
                "compilation "
                "and an unpublished nominal-model warmup are completed first. Snapshot probes and "
                "serialization run afterward. These are actual isolated learner completion times, "
                "not controller-contention deadlines or evolving-flight latency results."
            ),
            "lead_times_s": LEAD_TIMES,
        }
    )
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "fault_jump.json", fault_jump_audit(validation, nominal, model))
    cache_key = actuator_reference_fingerprint(contract) + json.dumps(
        asdict(config), sort_keys=True
    )
    if compiled is None:
        learner, evaluator = (
            build_actuator_skill_learner(contract, config),
            Evaluator(contract, config),
        )
    else:
        if cache_key not in compiled:
            compiled[cache_key] = (
                build_actuator_skill_learner(contract, config),
                Evaluator(contract, config),
            )
        learner, evaluator = compiled[cache_key]
    frozen_validation = _save_evaluation(
        output, "frozen_F2_validation", evaluator, initial.params, model, validation
    )
    frozen_development = _save_evaluation(
        output, "frozen_F2_development", evaluator, initial.params, model, contract.anchors
    )
    dr_validation = None
    if args.dr_checkpoint:
        dr = load_actuator_learner_checkpoint(args.dr_checkpoint)
        if actuator_reference_fingerprint(dr.contract) != original_reference:
            raise ValueError("DR comparator must share the nominal seed's immutable teacher")
        dr_evaluator = Evaluator(dr.contract, config)
        dr_validation = _save_evaluation(
            output, "frozen_DR_validation", dr_evaluator, dr.state.params, model, validation
        )
    save_actuator_learner_checkpoint(
        initial,
        contract,
        training[0],
        output / "initial",
        config=config,
        metadata={
            "mode": "adaptive_recovery_diagnostic",
            "seed": checkpoint.metadata.get("seed"),
            "adaptation_updates": 0,
            "physical_state_scope": "fixed virtual training-bank state",
        },
    )
    print(
        json.dumps(
            _jsonable(
                {
                    "event": "recovery_baseline",
                    "cell": args.cell,
                    "revision": args.revision,
                    "gain_index": gain_index,
                    "F2": _brief(frozen_validation),
                    "DR": _brief(dr_validation) if dr_validation else None,
                }
            )
        ),
        flush=True,
    )
    warmup_begin = time.perf_counter()
    warmup_proposal = learner.step(initial, training[0], nominal)
    jax.block_until_ready(warmup_proposal)
    warmup_seconds = time.perf_counter() - warmup_begin
    # The proposal was only a graph warmup on nominal parameters and is never installed.
    del warmup_proposal
    state = initial
    pending = []
    begin = time.perf_counter()
    for index in range(args.steps):
        source_index = index % len(training)
        step_begin = time.perf_counter()
        state, metrics = learner.step(state, training[source_index], model)
        jax.block_until_ready((state, metrics))
        completed = time.perf_counter()
        pending.append((state, metrics, source_index, step_begin - begin, completed - begin))
    learning_seconds = time.perf_counter() - begin
    trace = []
    for index, (snapshot, metrics, source_index, started, ended) in enumerate(pending):
        trace.append(
            {
                "attempt": index + 1,
                "library_version": int(snapshot.library_version),
                "published_adaptation_updates": int(snapshot.library_version) - initial_version,
                "finite_update_published": bool(metrics.finite_update_applied),
                "source_index": int(DEVELOPMENT_INDICES[source_index]),
                "source_state_age_at_completion_s": ended,
                "started_after_s": started,
                "completed_after_s": ended,
                "step_seconds": ended - started,
                "loss_before_update": float(metrics.loss.total),
                "gradient_norm": float(metrics.gradient_norm),
                "parameter_update_norm": float(metrics.parameter_update_norm),
                "loss_terms": {
                    name: float(getattr(metrics.loss, name)) for name in metrics.loss._fields[1:15]
                },
                "per_skill_position_squared_error_normalized": np.asarray(
                    metrics.loss.per_skill_position_error
                ).tolist(),
                "per_skill_velocity_squared_error_normalized": np.asarray(
                    metrics.loss.per_skill_velocity_error
                ).tolist(),
            }
        )
    _write_json(output / "publication_trace.json", trace)
    selected = {0, args.steps, *range(16, args.steps + 1, 16)}
    availability = []
    for lead in LEAD_TIMES:
        available = [index for index, row in enumerate(trace) if row["completed_after_s"] <= lead]
        attempt = available[-1] + 1 if available else 0
        selected.add(attempt)
        availability.append(
            {
                "lead_time_s": lead,
                "completed_attempt": attempt,
                "library_version": int(pending[attempt - 1][0].library_version)
                if attempt
                else initial_version,
            }
        )
    evaluations = []
    for attempt in sorted(selected):
        if attempt == 0:
            snapshot, elapsed, source = initial, 0.0, training[0]
            validation_report, development_report = frozen_validation, frozen_development
        else:
            snapshot, _, source_index, _, elapsed = pending[attempt - 1]
            source = training[source_index]
            validation_report = _save_evaluation(
                output,
                f"updates_{attempt:04d}_validation",
                evaluator,
                snapshot.params,
                model,
                validation,
            )
            development_report = _save_evaluation(
                output,
                f"updates_{attempt:04d}_development",
                evaluator,
                snapshot.params,
                model,
                contract.anchors,
            )
            save_actuator_learner_checkpoint(
                snapshot,
                contract,
                source,
                output / f"updates_{attempt:04d}",
                config=config,
                metadata={
                    "mode": "adaptive_recovery_diagnostic",
                    "seed": checkpoint.metadata.get("seed"),
                    "adaptation_updates": int(snapshot.library_version) - initial_version,
                    "completed_after_s": elapsed,
                    "revision": args.revision,
                    "physical_state_scope": (
                        "last fixed virtual training-bank state; not a flight episode"
                    ),
                },
            )
        evaluations.append(
            {
                "attempt": attempt,
                "adaptation_updates": int(snapshot.library_version) - initial_version,
                "library_version": int(snapshot.library_version),
                "available_after_seconds": elapsed,
                "validation": validation_report,
                "development": development_report,
            }
        )
    final = evaluations[-1]["validation"]
    summary = {
        "cell": args.cell,
        "revision": args.revision,
        "gain_index": gain_index,
        "gradient_evaluations_published_loop": args.steps,
        "warmup_gradient_evaluations": 1,
        "warmup_model": "nominal; no future true fault parameters used for compilation",
        "warmup_proposal_published": False,
        "warmup_seconds_including_compile": warmup_seconds,
        "completed_learning_seconds": learning_seconds,
        "step_mean_seconds": float(np.mean([row["step_seconds"] for row in trace]))
        if trace
        else None,
        "finite_updates_published": int(state.library_version) - initial_version,
        "initial_library_version": initial_version,
        "final_library_version": int(state.library_version),
        "reference_sha256": actuator_reference_fingerprint(contract),
        "original_reference_sha256": original_reference,
        "current_actor_config": asdict(config),
        "availability": availability,
        "frozen_F2": frozen_validation,
        "frozen_DR": dr_validation,
        "final_adaptive": final,
        "relative_mean_position_error_reduction_vs_F2": 1
        - final["position_rmse_mean_m"] / frozen_validation["position_rmse_mean_m"]
        if frozen_validation["position_rmse_mean_m"] > 1e-5
        else None,
        "relative_mean_velocity_error_reduction_vs_F2": 1
        - final["velocity_rmse_mean_mps"] / frozen_validation["velocity_rmse_mean_mps"]
        if frozen_validation["velocity_rmse_mean_mps"] > 1e-5
        else None,
        "scope": (
            "fixed held-out state-bank recovery and actual isolated learner availability; "
            "no closed-loop safety result"
        ),
    }
    _write_json(output / "evaluations.json", evaluations)
    _write_json(output / "summary.json", summary)
    _recovery_plots(output, evaluations, frozen_validation, dr_validation)
    print(
        json.dumps(
            _jsonable(
                {
                    "event": "recovery_complete",
                    "cell": args.cell,
                    "revision": args.revision,
                    "updates": summary["finite_updates_published"],
                    "step_mean_seconds": summary["step_mean_seconds"],
                    "availability": availability,
                    "F2": _brief(frozen_validation),
                    "DR": _brief(dr_validation) if dr_validation else None,
                    "A": _brief(final),
                }
            )
        ),
        flush=True,
    )
    if args.audit_gradients:
        audit_source = training[-1]
        initial_gradients = actuator_loss_gradient_contributions(
            learner, initial, audit_source, model
        )
        final_gradients = actuator_loss_gradient_contributions(learner, state, audit_source, model)
        for name, gradients, audited in (
            ("initial", initial_gradients, initial),
            ("final", final_gradients, state),
        ):
            gradients.update(
                {
                    "current_state": np.asarray(audit_source).tolist(),
                    "retention_iteration": int(audited.library_version),
                    "reference_sha256": actuator_reference_fingerprint(contract),
                }
            )
            _write_json(output / f"gradients_{name}.json", gradients)
        _gradient_plot(output, initial_gradients, final_gradients)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gains", "adapt", "matrix"), default="gains")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dr-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gain-report")
    parser.add_argument(
        "--cell", choices=("nominal", "effectiveness", "lag", "combined"), default="combined"
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--revision", choices=("none", "retention10", "braking_huber"), default="none"
    )
    parser.add_argument(
        "--unroll",
        type=int,
        choices=(1, 2, 4),
        help="Adaptation compile override; default is saved config",
    )
    parser.add_argument("--audit-gradients", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = load_actuator_learner_checkpoint(args.checkpoint)
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.stage == "gains":
        run_gain_sweep(args, checkpoint, output)
    elif args.stage == "matrix":
        compiled: dict[str, tuple[Any, Evaluator]] = {}
        summaries = {}
        for cell in ("nominal", "effectiveness", "lag", "combined"):
            cell_output = output / cell
            cell_output.mkdir(exist_ok=False)
            cell_args = argparse.Namespace(
                **(vars(args) | {"stage": "adapt", "cell": cell, "output": str(cell_output)})
            )
            summaries[cell] = run_adaptation(cell_args, checkpoint, cell_output, compiled)
        _write_json(
            output / "summary.json",
            {
                "cells": summaries,
                "scope": (
                    "four independent fixed-state recovery runs from identical saved Adam state"
                ),
                "compiled_graphs_reused": True,
                "optimizer_state_reused_across_cells": False,
            },
        )
    else:
        run_adaptation(args, checkpoint, output)


if __name__ == "__main__":
    main()
