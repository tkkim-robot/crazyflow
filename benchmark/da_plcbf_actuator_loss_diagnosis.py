"""Matched nominal/effectiveness objective and Adam-interference diagnostic.

Every arm starts from one complete competent checkpoint. Development states drive learning;
the disjoint validation bank is used only for measurement. There is no scene, obstacle,
certificate or QP input. Outputs use exclusive creation and never modify earlier evidence.
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
import optax

from benchmark.da_plcbf_actuator_behavior import validation_state_bank
from benchmark.da_plcbf_actuator_recovery import DEVELOPMENT_INDICES, Evaluator, changed_model
from crazyflow.safety.da_plcbf.actuator_dynamics import effort_lag_step
from crazyflow.safety.da_plcbf.actuator_learning import (
    actuator_reference_fingerprint,
    build_actuator_skill_learner,
    load_actuator_learner_checkpoint,
    save_actuator_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.persistent_skill_learner import _trainable_skill_tree


def write_json(path: Path, payload: Any) -> None:
    """Fail on an existing output or nonfinite/unserializable evidence."""
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def parameter_distance(a: Any, b: Any) -> float:
    return float(optax.tree.norm(jax.tree.map(lambda x, y: x - y, a, b)))


def fresh_validation_state_bank(
    model: Any, nominal_states: jax.Array, seed: int
) -> tuple[Any, Any]:
    """New independent directions, attitude/rate perturbations and coupled motor transients."""
    rng = np.random.default_rng(seed)
    states, labels = [], []
    for index in range(16):
        speed = (0.35, 0.9, 1.8, 2.3)[index % 4]
        angle = rng.uniform(-math.pi, math.pi)
        roll, pitch = rng.uniform(-0.22, 0.22), rng.uniform(-0.18, 0.18)
        sr, cr, sp, cp = (
            math.sin(roll / 2),
            math.cos(roll / 2),
            math.sin(pitch / 2),
            math.cos(pitch / 2),
        )
        state = nominal_states[0].at[3:7].set(jnp.asarray([sr * cp, cr * sp, -sr * sp, cr * cp]))
        state = state.at[7:10].set(
            jnp.asarray(
                [speed * math.cos(angle), speed * math.sin(angle), rng.uniform(-0.22, 0.22)]
            )
        )
        state = state.at[10:13].set(jnp.asarray(rng.uniform(-0.4, 0.4, size=3)))
        command = jnp.clip(
            state[13:17] * jnp.asarray(rng.uniform(0.82, 1.18, size=4)),
            model.command_lower,
            model.command_upper,
        )
        states.append(effort_lag_step(state, command, model, 0.015, substeps=3))
        labels.append(f"fresh_validation_seed{seed}_state{index}")
    return jnp.stack(states), tuple(labels)


def component_weights(contract: Any, config: Any, names: tuple[str, ...]) -> np.ndarray:
    """Weights of the exact additive objective, including the opt-in recovery branch."""
    settings = contract.learning_config
    braking_weight = config.terminal_braking_weight * (
        settings.recovery_braking_priority
        if settings.objective_mode == "balanced_reference_braking"
        else 1.0
    )
    mapping = {
        "total": 1.0,
        "trajectory_tracking": settings.trajectory_weight,
        "velocity_tracking": settings.velocity_weight,
        "reference_retention": settings.retention_weight,
        "trust": config.trust_weight,
        "terminal_braking": braking_weight,
        "prefix_tracking": settings.recovery_prefix_weight,
        "attitude": config.attitude_weight,
        "angular_rate": config.angular_rate_weight,
        "motor_effort": config.action_weight,
        "command_change": config.action_rate_weight,
        "saturation": config.saturation_weight,
        "motor_state_tracking": settings.motor_state_target_weight,
        "descriptor_target": config.target_weight,
        "diversity": config.diversity_weight,
        "pairwise": config.pairwise_weight,
    }
    return np.asarray([mapping[name] for name in names])


def build_component_probe(
    learner: Any, contract: Any, config: Any, state: Any, initial: Any
) -> Any:
    """One scalar reverse-AD executable reused across terms, states and current models.

    Avoids a large vector-Jacobian compiler executable that changes the rollout numerical
    kernel for each number of terms. It measures the trainable gradient actually presented
    to clipping/Adam; finite perturbations below are diagnostic gradient steps, not Adam.
    """
    sample = learner.loss(
        state.params, initial, contract.model, state.previous_params, state.library_version
    )[1]
    names = tuple(
        name
        for name in sample._fields
        if name != "rollout_valid_fraction" and getattr(sample, name).ndim == 0
    )

    def scalar(
        params: Any, previous: Any, physical: Any, model: Any, iteration: Any, selector: Any
    ) -> tuple[jax.Array, jax.Array]:
        metrics = learner.loss(params, physical, model, previous, iteration)[1]
        values = jnp.stack([getattr(metrics, name) for name in names])
        return jnp.vdot(selector, values), values

    differentiate = jax.jit(jax.value_and_grad(scalar, has_aux=True))

    def probe(state: Any, physical: Any, model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        gradients = {}
        flat = []
        begin = time.perf_counter()
        for index, name in enumerate(names):
            (_, values), gradient = differentiate(
                state.params,
                state.previous_params,
                physical,
                model,
                state.library_version,
                jnp.asarray(np.eye(len(names), dtype=np.float32)[index]),
            )
            gradient = _trainable_skill_tree(gradient, config)
            jax.block_until_ready((values, gradient))
            gradients[name] = gradient
            flat.append(
                np.concatenate([np.asarray(x).reshape(-1) for x in jax.tree.leaves(gradient)])
            )
        flat = np.asarray(flat, dtype=np.float64)
        weights = component_weights(contract, config, names)
        weighted = flat * weights[:, None]
        norms = np.linalg.norm(flat, axis=1)
        denominator = norms[:, None] * norms[None, :]
        cosine = np.divide(
            flat @ flat.T, denominator, out=np.zeros_like(denominator), where=denominator > 0
        )
        remainder = weighted[1:].sum(axis=0) - flat[0]
        entries = {
            name: {
                "value": float(values[index]),
                "weight": float(weights[index]),
                "raw_gradient_norm": float(norms[index]),
                "weighted_gradient_norm": float(np.linalg.norm(weighted[index])),
                "cosine_with_total": float(cosine[index, 0]) if norms[index] * norms[0] else None,
            }
            for index, name in enumerate(names)
        }
        return {
            "names": names,
            "components": entries,
            "cosine_matrix": cosine.tolist(),
            "zero_gradient_cosines_encoded_as_zero": True,
            "weighted_sum_total_gradient_error_norm": float(np.linalg.norm(remainder)),
            "seconds_including_any_compile": time.perf_counter() - begin,
        }, gradients

    return probe


def behavior_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Signed changes preserve each skill/prefix; improvements and regressions remain visible."""
    fields = (
        "position_rmse_by_skill_m",
        "velocity_rmse_by_skill_mps",
        "prefix_position_rmse_by_skill_m",
        "prefix_velocity_rmse_by_skill_mps",
        "terminal_speed_by_skill_mps",
        "position_rmse_by_state_and_skill_m",
        "velocity_rmse_by_state_and_skill_mps",
    )
    result = {
        name: (np.asarray(after[name]) - np.asarray(before[name])).tolist() for name in fields
    }
    result.update(
        {
            name: after[name] - before[name]
            for name in (
                "position_rmse_mean_m",
                "position_rmse_max_m",
                "velocity_rmse_mean_mps",
                "velocity_rmse_max_mps",
                "terminal_speed_max_mps",
            )
        }
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = load_actuator_learner_checkpoint(args.checkpoint)
    contract, config, shared = checkpoint.contract, checkpoint.config, checkpoint.state
    development = contract.anchors
    validation, validation_labels = (
        validation_state_bank(contract.model)
        if args.validation_seed is None
        else fresh_validation_state_bank(contract.model, development, args.validation_seed)
    )
    if any(np.array_equal(a, b) for a in np.asarray(development) for b in np.asarray(validation)):
        raise AssertionError("development and validation states overlap")
    if args.steps < 1 or args.perturbation_norm <= 0:
        raise ValueError("steps and perturbation norm must be positive")
    train_indices = tuple(index for index in DEVELOPMENT_INDICES if index < len(development))
    if not train_indices:
        train_indices = tuple(range(len(development)))
    sequence = [train_indices[index % len(train_indices)] for index in range(args.steps)]
    with (output / "inputs.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            development=np.asarray(development),
            validation=np.asarray(validation),
            training_indices=sequence,
        )
    sources = {}
    for name in (
        "benchmark/da_plcbf_actuator_loss_diagnosis.py",
        "benchmark/da_plcbf_actuator_recovery.py",
        "benchmark/da_plcbf_actuator_behavior.py",
        "crazyflow/safety/da_plcbf/actuator_learning.py",
        "crazyflow/safety/da_plcbf/actuator_dynamics.py",
        "crazyflow/safety/da_plcbf/persistent_skill_learner.py",
        "crazyflow/safety/da_plcbf/direct_wrench.py",
    ):
        payload = (Path(__file__).resolve().parents[1] / name).read_bytes()
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        sources[name] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "scope": "obstacle-free diagnosis; not a collision or online-timing result",
        "checkpoint": str(checkpoint.json_path.resolve()),
        "checkpoint_sha256": checkpoint.sha256,
        "reference_sha256": actuator_reference_fingerprint(contract),
        "library_version": int(shared.library_version),
        "config": asdict(config),
        "reference_learning_config": asdict(contract.learning_config),
        "arguments": vars(args),
        "sources": sources,
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "training_state_sequence": sequence,
        "development_count": len(development),
        "validation_count": len(validation),
        "validation_labels": validation_labels,
        "exact_state_overlap": False,
        "gradient_probe_states": {
            "development": train_indices[0],
            "validation": min(7, len(validation) - 1),
        },
        "optimizer_reset": (
            "reset moments/count once at fault/control onset; preserve all params, "
            "previous params, public counters"
        ),
        "finite_update_policy": (
            "every finite update published; no rejection, rollback or best-checkpoint selection"
        ),
        "repair_hypothesis": (
            "restore teacher equilibrium; emphasize individual skill prefix and braking recovery"
        ),
        "selection_rule": (
            "all prespecified arms reported at identical update counts; "
            "no validation-based selection"
        ),
    }
    write_json(output / "manifest.json", manifest)
    evaluator = Evaluator(contract, config)
    banks = {"development": development, "validation": validation}
    frozen = {}
    for cell in args.cells:
        model = changed_model(contract.model, cell)
        frozen[cell] = {}
        for name, bank in banks.items():
            report, arrays = evaluator.evaluate(shared.params, model, bank)
            frozen[cell][name] = report
            write_json(output / f"frozen_{cell}_{name}.json", report)
            with (output / f"frozen_{cell}_{name}.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
    results = []
    for mode in args.objectives:
        current_contract = replace(
            contract, learning_config=replace(contract.learning_config, objective_mode=mode)
        )
        learner = build_actuator_skill_learner(current_contract, config)
        probe = build_component_probe(
            learner, current_contract, config, shared, development[train_indices[0]]
        )
        for cell in args.cells:
            model = changed_model(contract.model, cell)
            print(json.dumps({"event": "components", "objective": mode, "cell": cell}), flush=True)
            for name, index in manifest["gradient_probe_states"].items():
                diagnostic, gradients = probe(shared, banks[name][index], model)
                diagnostic.update(
                    {"bank": name, "state_index": index, "objective": mode, "cell": cell}
                )
                if name == "development":
                    effects = {}
                    for component, gradient in gradients.items():
                        norm = float(optax.tree.norm(gradient))
                        if norm <= 1e-15:
                            continue
                        perturbed = jax.tree.map(
                            lambda p, g: p - args.perturbation_norm * g / norm,
                            shared.params,
                            gradient,
                        )
                        effects[component] = {}
                        for bank_name, bank in banks.items():
                            after, _ = evaluator.evaluate(perturbed, model, bank)
                            effects[component][bank_name] = behavior_delta(
                                frozen[cell][bank_name], after
                            )
                    diagnostic["normalized_gradient_descent_effects"] = effects
                    diagnostic["perturbation_norm"] = args.perturbation_norm
                    diagnostic["perturbation_interpretation"] = (
                        "same raw trainable-gradient displacement norm; "
                        "not actual Adam or weighted-component update magnitude"
                    )
                write_json(output / f"gradients_{mode}_{cell}_{name}.json", diagnostic)
            for history in ("persistent", "reset"):
                state = shared
                if history == "reset":
                    state = shared.replace(
                        optimizer_state=learner.initialize(
                            shared.params, contract.model
                        ).optimizer_state
                    )
                initial_state = state
                first = None
                snapshots = {}
                trace = []
                for update, state_index in enumerate(sequence, start=1):
                    previous = state
                    start = time.perf_counter()
                    state, metrics = learner.step(state, development[state_index], model)
                    jax.block_until_ready((state, metrics))
                    if not bool(metrics.finite_update_applied):
                        raise FloatingPointError(
                            f"nonfinite {mode}/{cell}/{history} update {update}"
                        )
                    trace.append(
                        {
                            "update": update,
                            "state_index": state_index,
                            "version": int(state.library_version),
                            "loss_before": float(metrics.loss.total),
                            "gradient_norm": float(metrics.gradient_norm),
                            "update_norm": float(metrics.parameter_update_norm),
                            "seconds_including_any_compile": time.perf_counter() - start,
                        }
                    )
                    if first is None:
                        first = state
                    if update in {1, 8, 32, 64, args.steps}:
                        snapshots[str(update)] = {}
                        for bank_name, bank in banks.items():
                            report, _ = evaluator.evaluate(state.params, model, bank)
                            snapshots[str(update)][bank_name] = report
                final_previous = previous
                measures = {}
                for bank_name, bank in banks.items():
                    before_last, _ = evaluator.evaluate(final_previous.params, model, bank)
                    final = snapshots[str(args.steps)][bank_name]
                    measures[bank_name] = {
                        "cumulative_delta": behavior_delta(frozen[cell][bank_name], final),
                        "last_update_delta": behavior_delta(before_last, final),
                        "previous_update_snapshot": before_last,
                    }
                stem = output / f"checkpoint_{mode}_{cell}_{history}"
                save_actuator_learner_checkpoint(
                    state,
                    current_contract,
                    development[sequence[-1]],
                    stem,
                    config=config,
                    metadata={
                        "diagnosis": True,
                        "objective": mode,
                        "cell": cell,
                        "optimizer_history": history,
                    },
                )
                arm = {
                    "objective": mode,
                    "cell": cell,
                    "optimizer_history": history,
                    "initial_version": int(initial_state.library_version),
                    "final_version": int(state.library_version),
                    "parameter_delta_first": parameter_distance(first.params, shared.params),
                    "parameter_delta_cumulative": parameter_distance(state.params, shared.params),
                    "parameter_delta_last": parameter_distance(state.params, final_previous.params),
                    "trace": trace,
                    "snapshots": snapshots,
                    "effects": measures,
                }
                write_json(output / f"arm_{mode}_{cell}_{history}.json", arm)
                results.append(arm)
                print(
                    json.dumps(
                        {
                            "event": "arm_complete",
                            "objective": mode,
                            "cell": cell,
                            "optimizer_history": history,
                            "updates": args.steps,
                            "validation_final_terminal_speed": snapshots[str(args.steps)][
                                "validation"
                            ]["terminal_speed_max_mps"],
                        }
                    ),
                    flush=True,
                )
    summary = {"manifest": manifest, "frozen": frozen, "arms": results}
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--perturbation-norm", type=float, default=0.001)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=("legacy", "balanced_reference", "balanced_reference_braking"),
        default=("legacy", "balanced_reference"),
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        choices=("nominal", "effectiveness"),
        default=("nominal", "effectiveness"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
