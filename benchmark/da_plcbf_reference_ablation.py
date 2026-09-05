"""Small fixed obstacle-free learner ablation; preserves every tested outcome and raw rollout.

This is deterministic behavior development, not a navigation experiment or throughput benchmark.
Every variant starts with identical parameters and Adam history. The promoted deployment file
contains the shared NOMINAL checkpoint with the chosen update configuration, never wind-trained
parameters disguised as an online initial condition.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from functools import cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.direct_wrench import quaternion_to_rotation_matrix
from crazyflow.safety.da_plcbf.learner_checkpoint import (
    load_learner_checkpoint,
    save_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.online_constant_wind import build_cf21b_version_a_resources
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    build_fibonacci_skill_spec,
    build_persistent_skill_learner,
    initialize_skill_actor,
    skill_library_competency,
)
from crazyflow.safety.da_plcbf.rigid_payload import CenteredRigidPayload, hover_authority
from crazyflow.safety.da_plcbf.state_conditioned_learning import (
    ReferenceContract,
    ReferenceLearningConfig,
    build_reference_skill_learner,
    loss_gradient_contributions,
    proprioceptive_state_bank,
    reference_contract_checkpoint_metadata,
    save_reference_contract,
)


def _write(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _weights(
    config: PersistentSkillConfig, reference: ReferenceLearningConfig | None
) -> dict[str, float]:
    result = {
        "descriptor_target": config.target_weight,
        "terminal_braking": config.terminal_braking_weight,
        "diversity": config.diversity_weight,
        "pairwise": config.pairwise_weight,
        "action": config.action_weight,
        "action_rate": config.action_rate_weight,
        "saturation": config.saturation_weight,
        "trust": config.trust_weight,
    }
    result.update(
        trajectory_tracking=reference.trajectory_weight if reference else 0.0,
        velocity_tracking=reference.velocity_weight if reference else 0.0,
        reference_retention=reference.retention_weight if reference else 0.0,
    )
    return result


def _behavior(
    rollout: Any,
    reference_states: np.ndarray,
    initial: np.ndarray,
    config: PersistentSkillConfig,
    actuator: Any,
) -> dict[str, Any]:
    states = np.asarray(rollout.states)
    terminal = np.linalg.norm(states[:, :, -1, 7:10], axis=-1)
    error = states[..., :3] - reference_states[..., :3]
    raw = np.asarray(rollout.raw_motor_forces)
    bounded = np.asarray(rollout.bounded_motor_forces)
    lower, upper = np.asarray(actuator.thrust_min), np.asarray(actuator.thrust_max)
    rotations = np.asarray(quaternion_to_rotation_matrix(jnp.asarray(states[..., 3:7])))
    displacement = states[..., :3] - initial[:, None, None, :3]
    speeds = np.linalg.norm(initial[:, 7:10], axis=-1)
    direction = initial[:, 7:10] / np.maximum(speeds[:, None], 1e-12)
    braking_travel = np.max(np.sum(displacement * direction[:, None, None, :], axis=-1), axis=-1)
    return {
        "trajectory_position_rmse_m": float(np.sqrt(np.mean(error**2))),
        "per_state_trajectory_position_rmse_m": np.sqrt(np.mean(error**2, axis=(1, 2, 3))).tolist(),
        "terminal_speed_mean_mps": float(np.mean(terminal)),
        "terminal_speed_p95_mps": float(np.percentile(terminal, 95)),
        "per_state_terminal_speed_mean_mps": np.mean(terminal, axis=1).tolist(),
        "maximum_tilt_rad": float(np.max(np.arccos(np.clip(rotations[..., 2, 2], -1, 1)))),
        "maximum_angular_rate_rps": float(np.max(np.linalg.norm(states[..., 10:13], axis=-1))),
        "raw_motor_saturation_fraction": float(np.mean((raw < lower) | (raw > upper))),
        "minimum_motor_upper_reserve_N": float(np.min(upper - bounded)),
        "minimum_motor_lower_reserve_N": float(np.min(bounded - lower)),
        "maximum_realized_horizontal_acceleration_mps2": float(
            np.max(np.linalg.norm(np.diff(states[..., 7:9], axis=2) / config.dt, axis=-1))
        ),
        "per_state_mean_forward_braking_travel_m": np.mean(braking_travel, axis=1).tolist(),
        "all_finite_and_actuator_valid": bool(
            np.all(rollout.policy_valid) and np.all(np.isfinite(states))
        ),
    }


@cache
def _batched_rollout(rollout: Any) -> Any:
    return jax.jit(jax.vmap(rollout, in_axes=(None, 0, None)))


def _evaluate(learner: Any, params: Any, bank: Any, model: Any) -> Any:
    result = _batched_rollout(learner.rollout)(params, bank, model)
    jax.block_until_ready(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--reference-warmup-steps", type=int, default=60)
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--policy-count", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-gradient-probes", action="store_true")
    parser.add_argument(
        "--event-record",
        type=Path,
        default=Path(
            "artifacts/da_plcbf/competent-revision-20260904/wind-estimated-8/dense_plant_states.npz"
        ),
    )
    args = parser.parse_args()
    if min(args.warmup_steps, args.reference_warmup_steps, args.updates) < 1:
        raise ValueError("warmup and update counts must be positive")
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=False)
    device = jax.devices(args.device)[0]
    config = PersistentSkillConfig(
        horizon=args.horizon,
        control_interval_steps=2,
        learning_rate=0.001,
        acceleration_limit=2.5,
        target_weight=10.0,
        diversity_weight=0.001,
        pairwise_weight=0.005,
        trust_weight=0.001,
        terminal_braking_weight=2.0,
        model_compensation=True,
        smooth_motor_bounds=False,
        velocity_offset_limit=0.35,
    )
    resources = jax.device_put(build_cf21b_version_a_resources(), device)
    with jax.default_device(device):
        bank, labels = proprioceptive_state_bank()
        with np.load(args.event_record, allow_pickle=False) as data:
            event = jnp.asarray(data["fixed"][200])
        bank = jnp.concatenate((bank, event[None]), axis=0)
        labels = (*labels, "recorded_event_proprioception")
        spec = build_fibonacci_skill_spec(
            policy_count=args.policy_count,
            maximum_speed=0.9,
            maximum_duration=0.7,
            horizon_duration=config.dt * config.horizon,
        )
        params = initialize_skill_actor(jax.random.key(args.seed), spec, config)
    original = build_persistent_skill_learner(spec, resources.actuator, config, device=device)
    persistent = original.initialize(params, resources.model)
    _write(
        directory / "protocol.json",
        {
            "config": asdict(config),
            "warmup_steps": args.warmup_steps,
            "reference_warmup_steps": args.reference_warmup_steps,
            "updates_per_variant": args.updates,
            "seed": args.seed,
            "bank_labels": labels,
            "bank_states": np.asarray(bank).tolist(),
            "event_source": str(args.event_record),
            "wind": [4.0, 1.6, 0.0],
            "scope": "fixed-state oracle behavior development; no safety or navigation outcome",
            "promotion_rule": (
                "reference variants only: all finite; trajectory RMSE <= frozen; "
                "terminal p95 <= frozen+0.05m/s; saturation <= frozen+0.01; "
                "minimize RMSE then braking excess"
            ),
            "provisional_checkpoint": "nominal state only; no wind-adapted parameters",
        },
    )
    warmup = []
    for index in range(args.warmup_steps):
        state = bank[0].at[7].set(0.5 if index % 5 == 4 else 0.0)
        persistent, metric = original.step(persistent, state, resources.model)
        jax.block_until_ready((persistent, metric))
        warmup.append(
            {
                "iteration": index,
                "finite": bool(metric.finite_update_applied),
                "loss": float(metric.loss.total),
            }
        )
    teacher = persistent.params
    contract = ReferenceContract(
        teacher, resources.model, bank, config, ReferenceLearningConfig(), spec, resources.actuator
    )
    revised_config = replace(config, learning_rate=0.0005, learn_durations=False)
    reference = build_reference_skill_learner(contract, revised_config, device=device)
    for index in range(args.reference_warmup_steps):
        persistent, metric = reference.step(persistent, bank[index % len(bank)], resources.model)
        jax.block_until_ready((persistent, metric))
        warmup.append(
            {
                "iteration": args.warmup_steps + index,
                "finite": bool(metric.finite_update_applied),
                "loss": float(metric.loss.total),
            }
        )
    _write(directory / "warmup.json", warmup)

    def save_nominal_candidate(
        target: Path,
        selected_config: PersistentSkillConfig,
        selected_contract: ReferenceContract,
        metadata: dict[str, Any],
    ) -> None:
        target.mkdir(parents=True, exist_ok=False)
        save_reference_contract(selected_contract, target / "nominal_reference")
        save_learner_checkpoint(
            persistent,
            spec,
            selected_config,
            resources.actuator,
            bank[0],
            target / "checkpoint",
            metadata={
                **metadata,
                **reference_contract_checkpoint_metadata(target / "nominal_reference"),
            },
        )

    save_nominal_candidate(
        directory / "provisional",
        revised_config,
        contract,
        {"scope": "provisional nominal checkpoint; unpromoted update config"},
    )
    print(
        json.dumps({"provisional_checkpoint": str(directory / "provisional/checkpoint")}),
        flush=True,
    )
    target = _evaluate(original, teacher, bank, resources.model)
    target_states = np.asarray(target.states)
    nominal = _evaluate(reference, persistent.params, bank, resources.model)
    _write(
        directory / "nominal_competency.json",
        {
            label: skill_library_competency(jax.tree.map(lambda x: x[index], nominal), spec, config)
            for index, label in enumerate(labels)
        },
    )
    changed_model = resources.model._replace(wind_velocity=jnp.asarray([4.0, 1.6, 0.0]))
    frozen = _evaluate(reference, persistent.params, bank, changed_model)
    frozen_metrics = _behavior(frozen, target_states, np.asarray(bank), config, resources.actuator)
    np.savez_compressed(
        directory / "initial_rollouts.npz",
        states=nominal.states,
        reference_states=target_states,
        frozen_changed_states=frozen.states,
        frozen_raw_motors=frozen.raw_motor_forces,
        bank=bank,
    )
    _write(directory / "frozen_behavior.json", frozen_metrics)

    # Fixed old/new physics × off/on mapping matrix from the same state and parameters.
    matrix, matrix_arrays = [], {}
    models = {
        "nominal": resources.model,
        "wind": changed_model,
        "payload25": CenteredRigidPayload(float(resources.model.mass) * 0.25).apply(
            resources.model
        ),
    }
    for model_name, model in models.items():
        for compensated in (False, True):
            actor_config = replace(config, model_compensation=compensated)
            evaluator = build_persistent_skill_learner(
                spec, resources.actuator, actor_config, device=device
            )
            rollouts = _evaluate(
                evaluator, persistent.params, bank[jnp.asarray([0, len(bank) - 1])], model
            )
            key = f"{model_name}__compensation_{int(compensated)}"
            for quantity in (
                "states",
                "raw_motor_forces",
                "bounded_motor_forces",
                "desired_accelerations",
            ):
                matrix_arrays[f"{key}_{quantity}"] = np.asarray(getattr(rollouts, quantity))
            matrix.append(
                {
                    "key": key,
                    "model": {
                        name: np.asarray(getattr(model, name)).tolist() for name in model._fields
                    },
                    "hover_authority": hover_authority(model, resources.actuator),
                    "behavior": _behavior(
                        rollouts,
                        target_states[[0, -1]],
                        np.asarray(bank)[[0, -1]],
                        config,
                        resources.actuator,
                    ),
                    "competency": [
                        skill_library_competency(
                            jax.tree.map(lambda x: x[i], rollouts), spec, actor_config
                        )
                        for i in range(2)
                    ],
                }
            )
    np.savez_compressed(directory / "compensation_physics_matrix.npz", **matrix_arrays)
    _write(directory / "compensation_physics_matrix.json", matrix)

    legacy = load_learner_checkpoint(
        "artifacts/da_plcbf/competent-revision-20260904/shared-checkpoint-1/competent_checkpoint",
        device=device,
    )
    legacy_initial = jnp.stack((bank[0], legacy.physical_state))
    legacy_records, legacy_arrays = [], {}
    for compensated in (False, True):
        legacy_config = replace(legacy.config, model_compensation=compensated)
        evaluator = build_persistent_skill_learner(
            legacy.spec, legacy.actuator, legacy_config, device=device
        )
        for changed in (False, True):
            model = legacy.point_model._replace(
                wind_velocity=jnp.asarray([4.0, 1.6, 0.0]) if changed else jnp.zeros(3)
            )
            rollout = _evaluate(evaluator, legacy.state.params, legacy_initial, model)
            key = f"changed_{int(changed)}__compensation_{int(compensated)}"
            for quantity in ("states", "raw_motor_forces", "bounded_motor_forces"):
                legacy_arrays[f"{key}_{quantity}"] = np.asarray(getattr(rollout, quantity))
            legacy_records.append(
                {
                    "key": key,
                    "checkpoint_sha256": legacy.sha256,
                    "actor_config": asdict(legacy_config),
                    "competency": [
                        skill_library_competency(
                            jax.tree.map(lambda x: x[i], rollout), legacy.spec, legacy_config
                        )
                        for i in range(2)
                    ],
                }
            )
    np.savez_compressed(directory / "legacy_compensation_2x2.npz", **legacy_arrays)
    _write(directory / "legacy_compensation_2x2.json", legacy_records)

    variants = [(f"original_current_lr{scale}", False, 0, scale) for scale in (0.25, 0.5, 1.0)]
    variants += [(f"reference_bank_lr{scale}", True, 2, scale) for scale in (0.25, 0.5, 1.0)]
    variants += [("reference_current_lr0.5", True, 0, 0.5)]
    outcomes = []
    for name, use_reference, anchor_count, multiplier in variants:
        trial_dir = directory / name
        trial_dir.mkdir()
        trial_config = replace(
            config,
            learning_rate=config.learning_rate * multiplier,
            learn_durations=not use_reference,
            velocity_offset_limit=0.35 if use_reference else None,
        )
        trial_contract = replace(
            contract,
            learning_config=replace(contract.learning_config, anchor_batch_size=anchor_count),
        )
        learner = (
            build_reference_skill_learner(trial_contract, trial_config, device=device)
            if use_reference
            else build_persistent_skill_learner(
                spec, resources.actuator, trial_config, device=device
            )
        )
        reference_metadata = {}
        if use_reference:
            save_reference_contract(trial_contract, trial_dir / "nominal_reference")
            reference_metadata = reference_contract_checkpoint_metadata(
                trial_dir / "nominal_reference"
            )
        state = persistent
        if not args.skip_gradient_probes and name in {
            "original_current_lr1.0",
            "reference_bank_lr0.5",
        }:
            gradient = loss_gradient_contributions(
                learner,
                state.params,
                event,
                changed_model,
                state.previous_params,
                component_weights=_weights(
                    trial_config, trial_contract.learning_config if use_reference else None
                ),
                iteration=state.library_version,
                trainable_config=trial_config,
            )
            _write(trial_dir / "initial_gradient_components.json", gradient)
        steps = []
        for index in range(args.updates):
            state, metrics = learner.step(state, event, changed_model)
            jax.block_until_ready((state, metrics))
            steps.append(
                {
                    "update": index + 1,
                    "version": int(state.library_version),
                    "finite": bool(metrics.finite_update_applied),
                    "loss": float(metrics.loss.total),
                    "gradient_norm": float(metrics.gradient_norm),
                    "parameter_update_norm": float(metrics.parameter_update_norm),
                    "terminal_braking_loss": float(metrics.loss.terminal_braking),
                    "reference_retention_loss": float(metrics.loss.reference_retention),
                }
            )
            if index + 1 in {1, 4, 8, 20}:
                save_learner_checkpoint(
                    state,
                    spec,
                    trial_config,
                    resources.actuator,
                    event,
                    trial_dir / f"update_{index + 1:03d}",
                    metadata={
                        "scope": "fixed-state development; early parameter-change probe",
                        **reference_metadata,
                    },
                )
        actual = _evaluate(learner, state.params, bank, changed_model)
        behavior = _behavior(actual, target_states, np.asarray(bank), config, resources.actuator)
        qualified = (
            all(x["finite"] for x in steps)
            and behavior["trajectory_position_rmse_m"]
            <= frozen_metrics["trajectory_position_rmse_m"]
            and behavior["terminal_speed_p95_mps"]
            <= frozen_metrics["terminal_speed_p95_mps"] + 0.05
            and behavior["raw_motor_saturation_fraction"]
            <= frozen_metrics["raw_motor_saturation_fraction"] + 0.01
        )
        outcome = {
            "name": name,
            "config": asdict(trial_config),
            "reference": use_reference,
            "reference_config": asdict(trial_contract.learning_config) if use_reference else None,
            "behavior": behavior,
            "finite_updates": sum(x["finite"] for x in steps),
            "qualified_under_predeclared_behavior_rule": qualified,
        }
        outcomes.append(outcome)
        _write(trial_dir / "result.json", outcome)
        _write(trial_dir / "updates.json", steps)
        np.savez_compressed(
            trial_dir / "rollouts.npz",
            states=actual.states,
            raw_motor_forces=actual.raw_motor_forces,
            bounded_motor_forces=actual.bounded_motor_forces,
            reference_states=target_states,
            bank=bank,
        )
        save_learner_checkpoint(
            state,
            spec,
            trial_config,
            resources.actuator,
            event,
            trial_dir / "final_checkpoint",
            metadata={"scope": "fixed-state wind adaptation ablation", **reference_metadata},
        )
        print(json.dumps(outcome), flush=True)
    candidates = [
        x for x in outcomes if x["reference"] and x["qualified_under_predeclared_behavior_rule"]
    ]
    qualified_promotion = bool(candidates)
    if not candidates:
        candidates = [x for x in outcomes if x["reference"]]
    chosen = min(
        candidates,
        key=lambda x: (
            x["behavior"]["trajectory_position_rmse_m"],
            x["behavior"]["terminal_speed_p95_mps"],
        ),
    )
    selected_config = PersistentSkillConfig(
        **{**chosen["config"], "descriptor_scales": tuple(chosen["config"]["descriptor_scales"])}
    )
    settings = ReferenceLearningConfig(
        **{
            **chosen["reference_config"],
            "trajectory_fractions": tuple(chosen["reference_config"]["trajectory_fractions"]),
        }
    )
    save_nominal_candidate(
        directory / "candidate",
        selected_config,
        replace(contract, learning_config=settings),
        {
            "selected_variant": chosen["name"],
            "qualified_behavior_promotion": qualified_promotion,
            "scope": (
                "same nominal parameters/Adam history; update config selected on "
                "fixed-state development only"
            ),
        },
    )
    _write(
        directory / "ablation_summary.json",
        {
            "frozen": frozen_metrics,
            "variants": outcomes,
            "selected_variant": chosen["name"],
            "qualified_behavior_promotion": qualified_promotion,
            "navigation_benefit_demonstrated": False,
        },
    )


if __name__ == "__main__":
    main()
