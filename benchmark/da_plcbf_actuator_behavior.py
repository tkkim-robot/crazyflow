"""Bounded obstacle-free actuator competence pilot with immutable numerical evidence.

Run with the repository's GPU Pixi environment. The default performs only the first seed's
32-update profile. Subsequent runs require a fresh output directory; ``--resume`` continues
the exact checkpoint's Adam history instead of restarting preparation.
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

from crazyflow.safety.da_plcbf.actuator_dynamics import (
    ActuatorModel,
    effort_lag_step,
    fit_native_time_constant,
    make_actuator_model,
)
from crazyflow.safety.da_plcbf.actuator_learning import (
    ActuatorDynamicsSupport,
    ActuatorReferenceConfig,
    ActuatorReferenceContract,
    ActuatorSkillConfig,
    actuator_proprioceptive_state_bank,
    actuator_reference_fingerprint,
    build_actuator_skill_learner,
    build_single_recovery_spec,
    initialize_actuator_skill_actor,
    load_actuator_learner_checkpoint,
    probe_actuator_library,
    rollout_actuator_skill_library,
    sample_actuator_training_models,
    save_actuator_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.online_constant_wind import build_cf21b_version_a_resources
from crazyflow.safety.da_plcbf.persistent_skill_learner import build_fibonacci_skill_spec


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path: Path, values: Any) -> None:
    with path.open("x") as stream:
        stream.write(
            json.dumps(_jsonable(values), indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def nominal_actuator_model() -> tuple[ActuatorModel, dict[str, Any]]:
    resources = build_cf21b_version_a_resources()
    actuator = resources.actuator
    fit = fit_native_time_constant()
    model = make_actuator_model(
        resources.model._replace(wind_velocity=jnp.zeros(3)),
        L=float(actuator.arm_length),
        thrust2torque=float(actuator.thrust_to_torque),
        mixing_matrix=actuator.mixing_matrix,
        thrust_min=actuator.thrust_min,
        thrust_max=actuator.thrust_max,
        time_constants=fit["nominal_tau"],
    )
    return model, fit


def validation_state_bank(model: ActuatorModel) -> tuple[jax.Array, tuple[str, ...]]:
    """Disjoint motion and coupled motor transients for checkpoint selection, without scenes."""
    training, _ = actuator_proprioceptive_state_bank(model)
    rng = np.random.default_rng(4107)
    states = []
    labels = []
    for index in range(16):
        speed = (0.35, 0.9, 1.8, 2.3)[index % 4]
        angle = (index + 0.5) * math.pi * (3 - math.sqrt(5))
        velocity = (speed * math.cos(angle), speed * math.sin(angle), (-1) ** index * 0.22)
        roll = (-1) ** index * (0.08 if index % 3 else 0.22)
        pitch = (-1) ** (index // 2) * (0.12 if index % 3 else 0.18)
        sr, cr, sp, cp = (
            math.sin(roll / 2),
            math.cos(roll / 2),
            math.sin(pitch / 2),
            math.cos(pitch / 2),
        )
        state = training[0].at[3:7].set(jnp.asarray([sr * cp, cr * sp, -sr * sp, cr * cp]))
        state = state.at[7:10].set(jnp.asarray(velocity))
        state = state.at[10:13].set(jnp.asarray(rng.uniform(-0.4, 0.4, size=3)))
        command = jnp.clip(
            state[13:17] * jnp.asarray(rng.uniform(0.82, 1.18, size=4)),
            model.command_lower,
            model.command_upper,
        )
        state = effort_lag_step(state, command, model, 0.015, substeps=3)
        states.append(state)
        labels.append(f"validation_v{speed:.2f}_attitude_motor_{index}")
    bank = jnp.stack(states)
    if any(np.array_equal(np.asarray(a), np.asarray(b)) for a in bank for b in training):
        raise AssertionError("validation bank overlaps an exact training state")
    return bank, tuple(labels)


def _compact_probe(probe: dict[str, Any]) -> dict[str, Any]:
    names = (
        "maximum_position_tracking_rmse_m",
        "maximum_velocity_tracking_rmse_mps",
        "maximum_terminal_speed_mps",
        "maximum_tilt_radians",
        "maximum_angular_rate_rps",
        "mean_command_saturation_fraction",
        "competent_under_declared_criteria",
        "competence_checks",
    )
    return {name: probe[name] for name in names}


def _validation_score(probes: list[dict[str, Any]]) -> float:
    """Prespecified DR selection score; independent of every test scene and online update."""
    values = []
    for probe in probes:
        if not probe["competence_checks"]["all_rollouts_finite_and_valid"]:
            return math.inf
        position = probe["maximum_position_tracking_rmse_m"]
        velocity = probe["maximum_velocity_tracking_rmse_mps"]
        braking = max(0.0, probe["maximum_terminal_speed_mps"] - 0.8)
        tilt = max(0.0, probe["maximum_tilt_radians"] - 0.9)
        rate = max(0.0, probe["maximum_angular_rate_rps"] - 12.0)
        values.append(5 * position**2 + velocity**2 + braking**2 + tilt**2 + (rate / 8) ** 2)
    return float(np.mean(values))


def _save_probe(
    output: Path,
    name: str,
    params: Any,
    contract: ActuatorReferenceContract,
    model: ActuatorModel,
    states: jax.Array,
    labels: tuple[str, ...],
    *,
    config: ActuatorSkillConfig | None = None,
) -> dict[str, Any]:
    selected = contract.actor_config if config is None else config
    report = probe_actuator_library(
        params, contract, model, states=states, labels=labels, config=selected
    )
    _write_json(output / f"{name}.json", report)
    rollout = jax.jit(
        jax.vmap(
            lambda y: rollout_actuator_skill_library(params, contract.spec, y, model, selected)
        )
    )(states)
    jax.block_until_ready(rollout)
    with (output / f"{name}.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            **{
                field: np.asarray(value)
                for field, value in zip(rollout._fields, rollout, strict=True)
            },
        )
    return report


def _plots(output: Path, trace: list[dict[str, Any]]) -> None:
    if not trace:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [entry["library_version"] for entry in trace]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(steps, [entry["loss_before_update"] for entry in trace])
    axes[0].set(
        title="Obstacle-free loss before publication",
        xlabel="Published version",
        ylabel="Weighted loss",
    )
    position = np.asarray([entry["per_skill_position_error"] for entry in trace])
    velocity = np.asarray([entry["per_skill_velocity_error"] for entry in trace])
    for index in range(position.shape[1]):
        axes[1].plot(steps, position[:, index], alpha=0.65)
        axes[2].plot(steps, velocity[:, index], alpha=0.65)
    axes[1].set(
        title="Each skill: normalized position error",
        xlabel="Published version",
        ylabel="Mean squared error",
    )
    axes[2].set(
        title="Each skill: normalized velocity error",
        xlabel="Published version",
        ylabel="Mean squared error",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("behavior pilot requires the declared GPU; refusing silent CPU fallback")
    nominal, tau_fit = nominal_actuator_model()
    training, training_labels = actuator_proprioceptive_state_bank(nominal)
    validation, validation_labels = validation_state_bank(nominal)
    if args.resume:
        checkpoint = load_actuator_learner_checkpoint(args.resume)
        contract, state, config = checkpoint.contract, checkpoint.state, checkpoint.config
        if args.mode != "dr" and checkpoint.metadata.get("seed", args.seed) != args.seed:
            raise ValueError("nominal resume must retain its original library seed")
        if not np.array_equal(np.asarray(training), np.asarray(contract.anchors)):
            raise ValueError("resume training bank differs from the saved immutable anchors")
        if args.absolute_braking:
            contract = replace(
                contract,
                learning_config=replace(contract.learning_config, reference_braking_excess=False),
            )
        if args.mode == "dr":
            # This is a new offline baseline with a separately initialized optimizer/history.
            # Nominal teacher remains the trained deployment checkpoint's immutable reference.
            learner = build_actuator_skill_learner(contract, config)
            params = initialize_actuator_skill_actor(
                jax.random.key(args.seed), contract.spec, config
            )
            state = learner.initialize(params, nominal)
    else:
        config = ActuatorSkillConfig(horizon=60, control_interval_steps=2)
        if args.mode == "single":
            spec = build_single_recovery_spec()
        else:
            spec = build_fibonacci_skill_spec(policy_count=16, horizon_duration=1.2)
        params = initialize_actuator_skill_actor(jax.random.key(args.seed), spec, config)
        contract = ActuatorReferenceContract(
            params,
            nominal,
            training,
            spec,
            config,
            ActuatorReferenceConfig(reference_braking_excess=not args.absolute_braking),
        )
        learner = build_actuator_skill_learner(contract)
        state = learner.initialize(params, nominal)
    learner = build_actuator_skill_learner(contract, config)
    original_version = int(state.library_version)
    attempt_offset = (
        int(checkpoint.metadata.get("training_attempts", original_version))
        if args.resume and args.mode != "dr"
        else 0
    )
    model_support = ActuatorDynamicsSupport(
        (0.7, 1.0), (tau_fit["nominal_tau"], 3 * tau_fit["nominal_tau"])
    )
    models = (
        sample_actuator_training_models(
            nominal, seed=args.seed + 1000, count=64, support=model_support
        )
        if args.mode == "dr"
        else (nominal,)
    )
    validation_models = (
        sample_actuator_training_models(nominal, seed=7001, count=4, support=model_support)
        if args.mode == "dr"
        else (nominal,)
    )
    contract_hash = actuator_reference_fingerprint(contract)
    _write_json(
        output / "manifest.json",
        {
            "seed": args.seed,
            "mode": args.mode,
            "requested_gradient_evaluations": args.steps,
            "initial_library_version": original_version,
            "initial_training_attempts": attempt_offset,
            "reference_sha256": contract_hash,
            "config": asdict(config),
            "learning_config": asdict(contract.learning_config),
            "tau_fit": tau_fit,
            "devices": [str(device) for device in jax.devices()],
            "jax_version": jax.__version__,
            "training_state_labels": training_labels,
            "validation_state_labels": validation_labels,
            "training_bank_sha256": hashlib.sha256(np.asarray(training).tobytes()).hexdigest(),
            "validation_bank_sha256": hashlib.sha256(np.asarray(validation).tobytes()).hexdigest(),
            "validation_policy": (
                "selection uses disjoint validation states and models; no test scenes or obstacles"
            ),
            "dr_selection_score": (
                "mean over validation dynamics of 5*max_position_rmse^2 + "
                "max_velocity_rmse^2 + relu(max_terminal_speed-.8)^2 + "
                "relu(max_tilt-.9)^2 + (relu(max_rate-12)/8)^2; finite trajectories required"
            ),
            "dynamics_support": asdict(model_support) if args.mode == "dr" else None,
            "training_models": [
                {
                    "effectiveness": np.asarray(m.effectiveness).tolist(),
                    "time_constants_s": np.asarray(m.time_constants).tolist(),
                }
                for m in models
            ],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_sha256_by_path": {
                name: hashlib.sha256(
                    (Path(__file__).resolve().parents[1] / name).read_bytes()
                ).hexdigest()
                for name in (
                    "benchmark/da_plcbf_actuator_behavior.py",
                    "crazyflow/safety/da_plcbf/actuator_learning.py",
                    "crazyflow/safety/da_plcbf/actuator_dynamics.py",
                    "crazyflow/safety/da_plcbf/persistent_skill_learner.py",
                    "crazyflow/safety/da_plcbf/direct_wrench.py",
                )
            },
        },
    )
    save_actuator_learner_checkpoint(
        state,
        contract,
        training[0],
        output / "initial",
        config=config,
        metadata={"seed": args.seed, "mode": args.mode},
    )
    initial_train = _save_probe(
        output,
        "initial_training",
        state.params,
        contract,
        nominal,
        training,
        training_labels,
        config=config,
    )
    initial_val = _save_probe(
        output,
        "initial_validation",
        state.params,
        contract,
        nominal,
        validation,
        validation_labels,
        config=config,
    )
    print(
        json.dumps(
            {
                "event": "initial_probe",
                "seed": args.seed,
                "mode": args.mode,
                "training": _compact_probe(initial_train),
                "validation": _compact_probe(initial_val),
            }
        ),
        flush=True,
    )
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(training))
    # Reconstruct the deterministic offline sampling stream without resetting continuation.
    for index in range(attempt_offset):
        if index and index % len(order) == 0:
            order = rng.permutation(len(training))
        rng.integers(len(models))
    trace: list[dict[str, Any]] = []
    validation_records = []
    if args.mode == "dr":
        initial_models = []
        for model_index, model in enumerate(validation_models):
            probe = _save_probe(
                output,
                f"initial_validation_model{model_index}",
                state.params,
                contract,
                model,
                validation,
                validation_labels,
                config=config,
            )
            initial_models.append(_compact_probe(probe))
        validation_records.append(
            {
                "gradient_evaluations": 0,
                "library_version": int(state.library_version),
                "models": initial_models,
                "checkpoint": "initial",
                "selection_score": _validation_score(initial_models),
            }
        )
        print(
            json.dumps(
                {
                    "event": "initial_dr_validation",
                    "models": initial_models,
                    "selection_score": _validation_score(initial_models),
                }
            ),
            flush=True,
        )
    begin_training = time.perf_counter()
    for index in range(args.steps):
        attempt = attempt_offset + index
        if attempt and attempt % len(order) == 0:
            order = rng.permutation(len(training))
        state_index = int(order[attempt % len(order)])
        model_index = int(rng.integers(len(models)))
        begin = time.perf_counter()
        state, metrics = learner.step(state, training[state_index], models[model_index])
        jax.block_until_ready((state, metrics))
        elapsed = time.perf_counter() - begin
        trace.append(
            {
                "gradient_evaluation": attempt + 1,
                "state_index": state_index,
                "model_index": model_index,
                "finite_update_published": bool(metrics.finite_update_applied),
                "library_version": int(state.library_version),
                "loss_before_update": float(metrics.loss.total),
                "loss_terms": {
                    name: float(getattr(metrics.loss, name)) for name in metrics.loss._fields[1:15]
                },
                "per_skill_position_error": np.asarray(
                    metrics.loss.per_skill_position_error
                ).tolist(),
                "per_skill_velocity_error": np.asarray(
                    metrics.loss.per_skill_velocity_error
                ).tolist(),
                "gradient_norm": float(metrics.gradient_norm),
                "parameter_update_norm": float(metrics.parameter_update_norm),
                "wall_seconds": elapsed,
            }
        )
        if index == 0 or (index + 1) % 16 == 0:
            print(
                json.dumps(
                    {
                        "event": "training_progress",
                        "seed": args.seed,
                        "mode": args.mode,
                        "completed": index + 1,
                        "version": int(state.library_version),
                        "loss": _jsonable(float(metrics.loss.total)),
                        "last_step_seconds": elapsed,
                    }
                ),
                flush=True,
            )
        if (index + 1) % args.validation_interval == 0 or index + 1 == args.steps:
            milestone = f"step_{attempt + 1:04d}"
            save_actuator_learner_checkpoint(
                state,
                contract,
                training[0],
                output / milestone,
                config=config,
                metadata={"seed": args.seed, "mode": args.mode, "training_attempts": attempt + 1},
            )
            validations = []
            for model_index, model in enumerate(validation_models):
                probe = _save_probe(
                    output,
                    f"{milestone}_validation_model{model_index}",
                    state.params,
                    contract,
                    model,
                    validation,
                    validation_labels,
                    config=config,
                )
                validations.append(_compact_probe(probe))
            validation_records.append(
                {
                    "gradient_evaluations": index + 1,
                    "library_version": int(state.library_version),
                    "models": validations,
                    "checkpoint": milestone,
                    "selection_score": _validation_score(validations),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "validation",
                        "seed": args.seed,
                        "mode": args.mode,
                        "completed": index + 1,
                        "models": validations,
                    }
                ),
                flush=True,
            )
    train_seconds = time.perf_counter() - begin_training
    _write_json(output / "training_trace.json", trace)
    _write_json(output / "validation_trace.json", validation_records)
    _plots(output, trace)
    published_updates = int(state.library_version) - original_version
    last_library_version = int(state.library_version)
    selected_validation = None
    if args.mode == "dr":
        selected_validation = min(validation_records, key=lambda record: record["selection_score"])
        selected_checkpoint = load_actuator_learner_checkpoint(
            output / selected_validation["checkpoint"]
        )
        state = selected_checkpoint.state
    # Nominal deployment teacher is immutable after preparation, with the optimizer unchanged.
    deployment_contract = (
        replace(
            contract,
            params=state.params,
            learning_config=replace(contract.learning_config, reference_braking_excess=True),
        )
        if args.mode in {"nominal", "single"}
        else contract
    )
    save_actuator_learner_checkpoint(
        state,
        deployment_contract,
        training[0],
        output / "deployment",
        config=config,
        metadata={
            "seed": args.seed,
            "mode": args.mode,
            "bootstrap_reference_sha256": contract_hash,
            "training_attempts": attempt_offset + args.steps,
        },
    )
    final_train = _save_probe(
        output,
        "final_training",
        state.params,
        deployment_contract,
        nominal,
        training,
        training_labels,
        config=config,
    )
    final_val = _save_probe(
        output,
        "final_validation",
        state.params,
        deployment_contract,
        nominal,
        validation,
        validation_labels,
        config=config,
    )
    summary = {
        "seed": args.seed,
        "mode": args.mode,
        "gradient_evaluations": args.steps,
        "finite_updates_published": published_updates,
        "last_library_version": last_library_version,
        "deployed_library_version": int(state.library_version),
        "selected_validation": selected_validation,
        "first_step_seconds_including_compile": trace[0]["wall_seconds"] if trace else None,
        "steady_step_mean_seconds": float(np.mean([entry["wall_seconds"] for entry in trace[1:]]))
        if len(trace) > 1
        else None,
        "training_seconds_including_periodic_validation": train_seconds,
        "student_integration_steps": args.steps
        * (1 + contract.learning_config.anchor_batch_size)
        * contract.spec.latent_codes.shape[0]
        * config.horizon,
        "final_training": _compact_probe(final_train),
        "final_validation": _compact_probe(final_val),
        "deployment_reference_sha256": actuator_reference_fingerprint(deployment_contract),
    }
    if args.adapter_probes:
        models_for_probe = {
            "nominal": nominal,
            "effectiveness_only": nominal._replace(effectiveness=jnp.asarray([0.7, 1.0, 1.0, 1.0])),
            "lag_only": nominal._replace(
                time_constants=nominal.time_constants * jnp.asarray([3.0, 1.6, 2.0, 1.0])
            ),
            "combined": nominal._replace(
                effectiveness=jnp.asarray([0.7, 1.0, 0.85, 1.0]),
                time_constants=nominal.time_constants * jnp.asarray([3.0, 1.6, 2.0, 1.0]),
            ),
        }
        comparisons = {}
        for name, model in models_for_probe.items():
            comparisons[name] = {}
            for mode in ("F0", "F1", "F2"):
                probe = _save_probe(
                    output,
                    f"adapter_{name}_{mode}",
                    state.params,
                    deployment_contract,
                    model,
                    validation,
                    validation_labels,
                    config=replace(config, adapter_mode=mode),
                )
                comparisons[name][mode] = _compact_probe(probe)
        summary["frozen_adapter_probes"] = comparisons
    _write_json(output / "summary.json", summary)
    print(json.dumps({"event": "complete", **summary}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--mode", choices=("nominal", "dr", "single"), default="nominal")
    parser.add_argument("--resume")
    parser.add_argument("--validation-interval", type=int, default=32)
    parser.add_argument("--adapter-probes", action="store_true")
    parser.add_argument("--absolute-braking", action="store_true")
    args = parser.parse_args()
    if args.steps < 0 or args.validation_interval < 1:
        parser.error("steps must be nonnegative and validation-interval positive")
    if args.mode == "dr" and not args.resume:
        parser.error("DR requires an explicit immutable trained nominal teacher via --resume")
    run_pilot(args)


if __name__ == "__main__":
    main()
