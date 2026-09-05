"""Matched-hover wind-on/off behavior probes; no obstacle input, navigation or visual fabrication.

The principal compensated candidate is retained as a baseline. Separately named restoration
cases rebase targets to competent initial parameters, preserve Adam, and compare identical
fallback mappings in both panes. The optional ungated residual can act during the braking tail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from benchmark.da_plcbf_reference_ablation import _write
from benchmark.da_plcbf_update_stabilization import _parameter_change
from crazyflow.safety.da_plcbf.learner_checkpoint import save_learner_checkpoint
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    rollout_skill_library,
    skill_library_competency,
)
from crazyflow.safety.da_plcbf.state_conditioned_learning import (
    build_reference_skill_learner,
    build_reference_skill_learner_from_checkpoint,
    load_reference_contract,
    reference_contract_checkpoint_metadata,
    save_reference_contract,
)

CASES = (
    "principal_compensated",
    "restoration_compensated",
    "restoration_uncompensated_gated",
    "restoration_uncompensated_full_residual",
)


def restoration_config(config: PersistentSkillConfig) -> PersistentSkillConfig:
    """Position/velocity tracking only, including terminal velocity; no raw regularizers."""
    return replace(
        config,
        diversity_weight=0.0,
        pairwise_weight=0.0,
        action_weight=0.0,
        action_rate_weight=0.0,
        saturation_weight=0.0,
        trust_weight=0.0,
        attitude_weight=0.0,
        angular_rate_weight=0.0,
        terminal_braking_weight=0.0,
        learn_durations=False,
    )


def trajectory_metrics(
    rollout: Any, reference: np.ndarray, spec: Any, config: PersistentSkillConfig
) -> dict[str, Any]:
    states = np.asarray(rollout.states)
    actual = states[..., :3] - states[:, :1, :3]
    target = reference[..., :3] - reference[:, :1, :3]
    error = actual - target
    centered = actual - actual.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    return {
        "tracking_position_rmse_m": float(np.sqrt(np.mean(error**2))),
        "centered_shape_rmse_m": float(np.sqrt(np.mean((centered - target_centered) ** 2))),
        "endpoint_centroid_m": actual[:, -1].mean(axis=0).tolist(),
        "endpoint_centroid_error_m": float(np.linalg.norm(error[:, -1].mean(axis=0))),
        "trajectory_centroid_error_rms_m": float(
            np.sqrt(np.mean(np.sum(error.mean(axis=0) ** 2, axis=-1)))
        ),
        "mean_endpoint_error_m": float(np.mean(np.linalg.norm(error[:, -1], axis=-1))),
        "maximum_reference_point_error_m": float(np.max(np.linalg.norm(error, axis=-1))),
        "terminal_velocity_tracking_rmse_mps": float(
            np.sqrt(np.mean((states[:, -1, 7:10] - reference[:, -1, 7:10]) ** 2))
        ),
        "competency": skill_library_competency(rollout, spec, config),
    }


def analyze_recorded_hover(directory: Path, output: Path, *, device: jax.Device) -> None:
    """Measure saved physical hover and matched probes against the immutable nominal teacher."""
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((directory / "navigation_comparison.json").read_text())
    summary = metadata["summary"]
    world = summary["world"]
    dt = world["config"]["dt"]
    period = dt * world["config"]["control_interval_steps"]
    stop = summary["task_phases"]["hover_until_seconds"]
    target = np.asarray(world["initial_state"][:3])
    contract = load_reference_contract(directory / "nominal_reference", device=device)
    with np.load(directory / "dense_plant_states.npz", allow_pickle=False) as archive:
        dense = {key: archive[key] for key in archive.files}
    with np.load(directory / "same_state_probe_trajectories.npz", allow_pickle=False) as archive:
        trajectories = {key: archive[key] for key in archive.files}
    with np.load(directory / "raw_diagnostics.npz", allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    phases = [0.0] + [x["time_seconds"] for x in world["config"]["wind_events"]]
    phases += [x["time_seconds"] for x in world["config"]["payload_events"]] + [stop]
    phases = sorted(set(x for x in phases if x <= stop))
    physical, plot_arrays = {}, {}
    for method, states in dense.items():
        times = np.arange(len(states)) * dt
        error = np.linalg.norm(states[:, :3] - target, axis=-1)
        qx, qy, qz, qw = states[:, 3:7].T
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
        tilt = np.arccos(np.clip(1 - 2 * (qx**2 + qy**2), -1, 1))
        physical[method] = []
        for begin, end in zip(phases[:-1], phases[1:], strict=True):
            mask = (times >= begin) & (times < end)
            physical[method].append(
                {
                    "start_seconds": begin,
                    "end_seconds": end,
                    "maximum_position_error_m": float(np.max(error[mask])),
                    "rms_position_error_m": float(np.sqrt(np.mean(error[mask] ** 2))),
                    "maximum_absolute_roll_degrees": float(np.rad2deg(np.max(abs(roll[mask])))),
                    "maximum_absolute_pitch_degrees": float(np.rad2deg(np.max(abs(pitch[mask])))),
                    "maximum_tilt_degrees": float(np.rad2deg(np.max(tilt[mask]))),
                }
            )
        mask = times <= stop
        plot_arrays[f"{method}_time"] = times[mask]
        plot_arrays[f"{method}_position_error"] = error[mask]
        plot_arrays[f"{method}_tilt_degrees"] = np.rad2deg(tilt[mask])
    keys = sorted(
        {
            (int(key.split("_")[0]), key.split("_")[1])
            for key in trajectories
            if int(key.split("_")[0]) * period < stop
        }
    )
    initial_states = []
    for index, anchor in keys:
        fixed = trajectories[f"{index}_{anchor}_fixed_full_states"]
        adaptive = trajectories[f"{index}_{anchor}_adaptive_full_states"]
        np.testing.assert_array_equal(fixed[:, 0], adaptive[:, 0])
        initial_states.append(fixed[1, 0])
    reference_fn = jax.jit(
        jax.vmap(
            lambda state: rollout_skill_library(
                contract.params,
                contract.spec,
                state,
                contract.model,
                contract.actuator,
                contract.actor_config,
            ).states
        )
    )
    references = np.asarray(
        jax.block_until_ready(reference_fn(jax.device_put(np.asarray(initial_states), device)))
    )
    probes = []
    for (index, anchor), reference in zip(keys, references, strict=True):
        row = {
            "time_seconds": index * period,
            "anchor": anchor,
            "state": trajectories[f"{index}_{anchor}_fixed_full_states"][1, 0].tolist(),
            "point_wind": raw[f"{anchor}_point_wind"][index].tolist(),
            "actual_mass": float(raw[f"{anchor}_actual_mass"][index]),
            "methods": {},
        }
        for method in ("fixed", "adaptive"):
            states = trajectories[f"{index}_{anchor}_{method}_full_states"][1:]
            # Only spatial/kinematic diagnostics are used here. Saved states do not contain
            # the complete actuator-validity trace needed to assert full skill competence.
            metrics = trajectory_metrics(
                SimpleNamespace(states=states, policy_valid=np.isfinite(states).all(axis=-1)),
                reference,
                contract.spec,
                contract.actor_config,
            )
            competency = metrics.pop("competency")
            for field in (
                "occupied_direction_count",
                "aligned_direction_count",
                "endpoint_pairwise_mean_m",
                "trajectory_pairwise_rms_mean_m",
                "terminal_speed_mps",
            ):
                metrics[field] = competency[field]
            metrics["direction_bins"] = competency["direction_bin_for_skill"]
            row["methods"][method] = metrics
        probes.append(row)
    evidence = {
        "source_run": str(directory),
        "scope": (
            "recorded physical hover plus same-state/current-model fallback probes; "
            "reference is the saved teacher under its fixed nominal model at that same state"
        ),
        "reference_nominal_model_never_rebased_after_events": True,
        "hover_until_seconds": stop,
        "physical": physical,
        "same_state_probes": probes,
        "source_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in (
                "navigation_comparison.json",
                "navigation_comparison.npz",
                "dense_plant_states.npz",
                "same_state_probe_trajectories.npz",
                "raw_diagnostics.npz",
                "nominal_reference.json",
                "nominal_reference.npz",
            )
        },
    }
    _write(output / "actual_hover_analysis.json", evidence)
    np.savez_compressed(
        output / "reference_and_physical.npz",
        reference_states=references,
        reference_control_indices=np.asarray([x[0] for x in keys]),
        reference_anchor=np.asarray([x[1] for x in keys]),
        **plot_arrays,
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
    selected = [row for row in probes if row["anchor"] == "adaptive"]
    for method, color in (("fixed", "#6b7280"), ("adaptive", "#087e8b")):
        times = plot_arrays[f"{method}_time"]
        axes[0, 0].plot(times, plot_arrays[f"{method}_position_error"], color=color, label=method)
        axes[0, 1].plot(times, plot_arrays[f"{method}_tilt_degrees"], color=color, label=method)
        for ax, field in (
            (axes[1, 0], "tracking_position_rmse_m"),
            (axes[1, 1], "endpoint_centroid_error_m"),
            (axes[2, 0], "centered_shape_rmse_m"),
            (axes[2, 1], "occupied_direction_count"),
        ):
            ax.plot(
                [row["time_seconds"] for row in selected],
                [row["methods"][method][field] for row in selected],
                color=color,
                marker="o",
                markersize=3,
                label=method,
            )
    labels = (
        "Physical hover position error (m)",
        "Physical vehicle tilt (degrees)",
        "Same-state path tracking RMSE (m)",
        "Same-state endpoint centroid error (m)",
        "Same-state centered shape RMSE (m)",
        "Same-state occupied direction bins / 16",
    )
    for ax, label in zip(axes.flat, labels, strict=True):
        ax.set_title(label, fontsize=10)
        ax.set_xlim(0, stop)
        ax.grid(alpha=0.2)
        for event in phases[1:-1]:
            ax.axvline(event, color="black", alpha=0.25, linestyle=":")
    for ax in axes[-1]:
        ax.set_xlabel("Recorded simulation time (s)")
    axes[0, 0].legend()
    fig.suptitle(
        "Recorded hover and fallback restoration\n"
        "Shared adaptive state/current model; dotted lines mark prescribed wind/payload events",
        fontsize=12,
    )
    fig.savefig(output / "actual_hover_analysis.png", dpi=180)
    fig.savefig(output / "actual_hover_analysis.pdf")
    plt.close(fig)
    print(f"Saved physical and matched-state hover analysis: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/da_plcbf/learning-revision-20260905/reference-ablation-seed7/candidate/checkpoint"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--navigation-run", type=Path)
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--wind", type=float, nargs=3, default=[1.6, 0.8, 0.0])
    parser.add_argument("--updates-on", type=int, default=200)
    parser.add_argument("--updates-off", type=int, default=200)
    parser.add_argument("--nominal-settle-updates", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if args.navigation_run is not None:
        analyze_recorded_hover(
            args.navigation_run, args.output_dir, device=jax.devices(args.device)[0]
        )
        return
    if min(args.updates_on, args.updates_off) < 1 or args.nominal_settle_updates < 0:
        raise ValueError("on/off update counts must be positive; nominal settling nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = jax.devices(args.device)[0]
    source, source_contract, _ = build_reference_skill_learner_from_checkpoint(
        args.checkpoint, device=device
    )
    hover = jax.device_put(
        jnp.asarray([0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), device
    )
    nominal = source_contract.model
    wind_model = nominal._replace(
        wind_velocity=jax.device_put(jnp.asarray(args.wind, dtype=jnp.float32), device)
    )
    summary = {}
    _write(
        args.output_dir / "protocol.json",
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": source.sha256,
            "cases": args.cases,
            "wind": args.wind,
            "updates_on": args.updates_on,
            "updates_off": args.updates_off,
            "nominal_settle_updates": args.nominal_settle_updates,
            "hover_state": np.asarray(hover).tolist(),
            "scope": (
                "fixed-proprioception behavior probes; not a closed-loop hover or navigation run"
            ),
            "physics_scope": (
                "identical known point model/actuator limits for fixed and adaptive at each probe"
            ),
            "nominal_semantics": (
                "a separate model-aware nominal may hold the physical vehicle "
                "while both fallback actors omit feedforward"
            ),
            "source_sha256": {
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in (
                    __file__,
                    "crazyflow/safety/da_plcbf/persistent_skill_learner.py",
                    "crazyflow/safety/da_plcbf/state_conditioned_learning.py",
                )
            },
        },
    )
    for case in args.cases:
        directory = args.output_dir / case
        directory.mkdir()
        config = source.config
        persistent = source.state
        contract = source_contract
        if case != "principal_compensated":
            config = replace(
                restoration_config(config),
                model_compensation=case == "restoration_compensated",
                gate_residual_with_skill_duration=case != "restoration_uncompensated_full_residual",
            )
            contract = replace(contract, params=persistent.params, actor_config=config)
        learner = build_reference_skill_learner(contract, config, device=device)
        warmup = []
        if case != "principal_compensated":
            for index in range(args.nominal_settle_updates):
                persistent, metrics = jax.block_until_ready(
                    learner.step(persistent, hover, nominal)
                )
                warmup.append(
                    {
                        "update": index + 1,
                        "finite": bool(metrics.finite_update_applied),
                        "loss": float(metrics.loss.total),
                        "gradient_norm": float(metrics.gradient_norm),
                        "parameter_update_norm": float(metrics.parameter_update_norm),
                    }
                )
            contract = replace(contract, params=persistent.params)
            learner = build_reference_skill_learner(contract, config, device=device)
        fixed = persistent.params
        reference = jax.block_until_ready(learner.rollout(fixed, hover, nominal))
        target_states = np.asarray(reference.states)
        save_reference_contract(contract, directory / "nominal_reference")
        binding = reference_contract_checkpoint_metadata(directory / "nominal_reference")
        save_learner_checkpoint(
            persistent,
            source.spec,
            config,
            source.actuator,
            hover,
            directory / "initial_checkpoint",
            metadata={"case": case, **binding},
        )
        _write(directory / "warmup.json", warmup)
        gradient = jax.grad(
            lambda params: learner.loss(
                params, hover, nominal, persistent.previous_params, persistent.library_version
            )[0]
        )(persistent.params)
        calm_copy = persistent
        calm = []
        for index in range(20):
            calm_copy, metrics = jax.block_until_ready(learner.step(calm_copy, hover, nominal))
            calm.append(
                {
                    "update": index + 1,
                    "loss": float(metrics.loss.total),
                    "gradient_norm": float(metrics.gradient_norm),
                    "parameter_update_norm": float(metrics.parameter_update_norm),
                }
            )
        calm_rollout = jax.block_until_ready(learner.rollout(calm_copy.params, hover, nominal))
        _write(
            directory / "calm_drift.json",
            {
                "scope": (
                    "discarded 20-update copy; full initial Adam retained; "
                    "does not train wind initial state"
                ),
                "initial_loss_gradient_norm": float(optax.tree.norm(gradient)),
                "after_20_metrics": trajectory_metrics(
                    calm_rollout, target_states, source.spec, config
                ),
                "parameter_delta": _parameter_change(fixed, calm_copy.params),
                "updates": calm,
            },
        )
        arrays = {
            "hover_state": np.asarray(hover),
            "initial_nominal_states": target_states,
            "calm_copy_after_20_states": np.asarray(calm_rollout.states),
        }
        probes, updates = [], []

        def probe(stage: str, completed: int) -> None:
            pair_record = {
                "stage": stage,
                "stage_updates": completed,
                "library_version": int(persistent.library_version),
                "parameter_delta": _parameter_change(fixed, persistent.params),
                "models": {},
            }
            for model_name, model in (("nominal", nominal), ("wind", wind_model)):
                pair = {}
                paths = {}
                for method, params in (("fixed", fixed), ("adaptive", persistent.params)):
                    result = jax.block_until_ready(learner.rollout(params, hover, model))
                    pair[method] = trajectory_metrics(result, target_states, source.spec, config)
                    paths[method] = np.asarray(result.states[..., :3])
                    key = f"{stage}__u{completed:04d}__{model_name}__{method}"
                    for quantity in (
                        "states",
                        "wrenches",
                        "raw_motor_forces",
                        "bounded_motor_forces",
                    ):
                        arrays[f"{key}__{quantity}"] = np.asarray(getattr(result, quantity))
                pair["matched_fixed_adaptive_max_point_distance_m"] = float(
                    np.max(np.linalg.norm(paths["fixed"] - paths["adaptive"], axis=-1))
                )
                pair["point_wind"] = np.asarray(model.wind_velocity).tolist()
                pair_record["models"][model_name] = pair
            probes.append(pair_record)
            save_learner_checkpoint(
                persistent,
                source.spec,
                config,
                source.actuator,
                hover,
                directory / f"{stage}-{completed:04d}",
                metadata={"stage": stage, "stage_updates": completed, **binding},
            )
            on = pair_record["models"]["wind" if stage == "wind_on" else "nominal"]["adaptive"]
            print(
                f"{case} {stage} {completed}: "
                f"active-model RMSE={on['tracking_position_rmse_m']:.5f} "
                f"bins={on['competency']['occupied_direction_count']}",
                flush=True,
            )

        for stage, model, count in (
            ("wind_on", wind_model, args.updates_on),
            ("wind_off", nominal, args.updates_off),
        ):
            probe(stage, 0)
            checkpoints = {1, 4, 20, 50, 100, count}
            for index in range(count):
                persistent, metrics = jax.block_until_ready(learner.step(persistent, hover, model))
                updates.append(
                    {
                        "stage": stage,
                        "update": index + 1,
                        "version": int(persistent.library_version),
                        "finite": bool(metrics.finite_update_applied),
                        "loss": float(metrics.loss.total),
                        "gradient_norm": float(metrics.gradient_norm),
                        "parameter_update_norm": float(metrics.parameter_update_norm),
                    }
                )
                if index + 1 in checkpoints:
                    probe(stage, index + 1)
        np.savez_compressed(directory / "probe_rollouts.npz", **arrays)
        _write(directory / "updates.json", updates)
        _write(directory / "probes.json", probes)
        outcome = {
            "case": case,
            "config": asdict(config),
            "reference_config": asdict(contract.learning_config),
            "initial_competency": skill_library_competency(reference, source.spec, config),
            "finite_updates": sum(row["finite"] for row in updates),
            "final_wind_on": next(
                row
                for row in probes
                if row["stage"] == "wind_on" and row["stage_updates"] == args.updates_on
            ),
            "final_wind_off": probes[-1],
        }
        _write(directory / "summary.json", outcome)
        summary[case] = outcome
    _write(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
