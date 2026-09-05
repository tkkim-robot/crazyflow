"""Declared parameter-group and step-cap development ablation after an early coverage loss.

This does not alter the frozen navigation campaign or use obstacles to admit updates. All
variants resume the same nominal parameters and Adam history, and publish every finite update.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.da_plcbf_reference_ablation import _behavior, _evaluate, _write
from crazyflow.safety.da_plcbf.learner_checkpoint import (
    load_learner_checkpoint,
    save_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.state_conditioned_learning import (
    build_reference_skill_learner,
    load_reference_contract,
    reference_contract_checkpoint_metadata,
    save_reference_contract,
)

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.persistent_skill_learner import SkillActorParams


def _parameter_change(before: SkillActorParams, after: SkillActorParams) -> dict[str, float]:
    result = {
        field.name: float(
            np.linalg.norm(np.asarray(getattr(after, field.name) - getattr(before, field.name)))
        )
        for field in fields(before)
    }
    result["network"] = float(
        np.sqrt(sum(value**2 for key, value in result.items() if not key.endswith("offsets")))
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = jax.devices(args.device)[0]
    initial = load_learner_checkpoint(args.source_dir / "provisional/checkpoint", device=device)
    contract = load_reference_contract(
        args.source_dir / "provisional/nominal_reference", device=device
    )
    config = replace(initial.config, learn_durations=False, velocity_offset_limit=0.35)
    bank = contract.anchors
    event = bank[-1]
    model = contract.model._replace(wind_velocity=jnp.asarray([4.0, 1.6, 0.0]))
    evaluator = build_reference_skill_learner(contract, config, device=device)
    reference = _evaluate(evaluator, contract.params, bank, contract.model)
    frozen = _evaluate(evaluator, initial.state.params, bank, model)
    targets = np.asarray(reference.states)
    frozen_metrics = _behavior(frozen, targets, np.asarray(bank), config, initial.actuator)
    variants = [
        (f"reference_{group}_lr{scale}", group, scale, None)
        for group in ("network", "offsets")
        for scale in (0.25, 0.5, 1.0)
    ] + [("reference_all_lr1.0_stepcap0.002", "all", 1.0, 0.002)]
    _write(
        args.output_dir / "protocol.json",
        {
            "source_directory": str(args.source_dir),
            "initial_checkpoint_sha256": initial.sha256,
            "initial_library_version": int(initial.state.library_version),
            "state": np.asarray(event).tolist(),
            "wind": np.asarray(model.wind_velocity).tolist(),
            "updates": 80,
            "snapshots": [0, 1, 4, 8, 20, 80],
            "variants_declared_before_execution": variants,
            "scope": (
                "development extension after reference-bank early certificate loss; "
                "same fixed state/model/initial params/Adam; "
                "no obstacle input or update admission; "
                "no promotion into already frozen held-out navigation campaign"
            ),
        },
    )
    provisional = args.output_dir / "provisional"
    provisional.mkdir()
    save_reference_contract(contract, provisional / "nominal_reference")
    save_learner_checkpoint(
        initial.state,
        initial.spec,
        config,
        initial.actuator,
        initial.physical_state,
        provisional / "checkpoint",
        metadata={
            "source_checkpoint_sha256": initial.sha256,
            **reference_contract_checkpoint_metadata(provisional / "nominal_reference"),
        },
    )
    _write(args.output_dir / "frozen_behavior.json", frozen_metrics)
    np.savez_compressed(
        args.output_dir / "initial_rollouts.npz",
        bank=bank,
        reference_states=targets,
        frozen_changed_states=frozen.states,
        frozen_raw_motors=frozen.raw_motor_forces,
    )
    outcomes = []
    for name, group, multiplier, cap in variants:
        trial_dir = args.output_dir / name
        trial_dir.mkdir()
        trial_config = replace(
            config,
            learning_rate=0.001 * multiplier,
            trainable_parameters=group,
            max_parameter_update_norm=cap,
        )
        learner = build_reference_skill_learner(contract, trial_config, device=device)
        save_reference_contract(contract, trial_dir / "nominal_reference")
        reference_metadata = reference_contract_checkpoint_metadata(trial_dir / "nominal_reference")
        state = initial.state
        steps, snapshots, arrays = [], [], {}
        for index in range(80):
            before = state.params
            state, metric = learner.step(state, event, model)
            jax.block_until_ready((state, metric))
            steps.append(
                {
                    "update": index + 1,
                    "version": int(state.library_version),
                    "finite": bool(metric.finite_update_applied),
                    "loss": float(metric.loss.total),
                    "gradient_norm": float(metric.gradient_norm),
                    "parameter_update_norm": float(metric.parameter_update_norm),
                    "parameter_group_step_norms": _parameter_change(before, state.params),
                    "terminal_braking_loss": float(metric.loss.terminal_braking),
                    "reference_retention_loss": float(metric.loss.reference_retention),
                }
            )
            if index + 1 in {1, 4, 8, 20, 80}:
                rollout = _evaluate(learner, state.params, bank, model)
                behavior = _behavior(rollout, targets, np.asarray(bank), config, initial.actuator)
                snapshots.append(
                    {
                        "completed_updates": index + 1,
                        "behavior": behavior,
                        "parameter_change_from_initial": _parameter_change(
                            initial.state.params, state.params
                        ),
                    }
                )
                for quantity in ("states", "raw_motor_forces", "bounded_motor_forces"):
                    arrays[f"update_{index + 1:03d}_{quantity}"] = np.asarray(
                        getattr(rollout, quantity)
                    )
                save_learner_checkpoint(
                    state,
                    initial.spec,
                    trial_config,
                    initial.actuator,
                    event,
                    trial_dir / ("final_checkpoint" if index == 79 else f"update_{index + 1:03d}"),
                    metadata={
                        "scope": "parameter-group development extension; no safety admission",
                        "source_checkpoint_sha256": initial.sha256,
                        **reference_metadata,
                    },
                )
        outcome = {
            "name": name,
            "config": asdict(trial_config),
            "reference": True,
            "reference_config": asdict(contract.learning_config),
            "behavior": snapshots[-1]["behavior"],
            "snapshots": snapshots,
            "finite_updates": sum(step["finite"] for step in steps),
        }
        outcomes.append(outcome)
        _write(trial_dir / "result.json", outcome)
        _write(trial_dir / "updates.json", steps)
        np.savez_compressed(
            trial_dir / "rollouts.npz", bank=bank, reference_states=targets, **arrays
        )
        print(
            f"{name}: finite={outcome['finite_updates']}/80 "
            f"RMSE={outcome['behavior']['trajectory_position_rmse_m']:.6f} "
            f"terminal_mean={outcome['behavior']['terminal_speed_mean_mps']:.6f}",
            flush=True,
        )
    _write(
        args.output_dir / "ablation_summary.json",
        {
            "frozen": frozen_metrics,
            "variants": outcomes,
            "navigation_benefit_demonstrated": False,
            "promotion_into_frozen_navigation_campaign": False,
        },
    )


if __name__ == "__main__":
    main()
