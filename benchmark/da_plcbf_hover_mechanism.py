"""Same-complete-state, same-model repertoire recovery probes for hover experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.da_plcbf_hover_followup import restore_scene
from benchmark.da_plcbf_recovery_analysis import sha
from crazyflow.safety.da_plcbf.actuator_experiment import _hash_tree
from crazyflow.safety.da_plcbf.actuator_learning import (
    load_actuator_learner_checkpoint,
    rollout_actuator_skill_library,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.case / "scene-plan.json").read_text())
    scene = restore_scene(plan["scene"])
    frozen = load_actuator_learner_checkpoint(args.case / "F2/attempt-00/snapshots/initial")
    source = args.case / "A_BAL/attempt-00/snapshots"
    samples = sorted(source.glob("critical-*.json"), key=lambda p: float(p.stem.split("-", 1)[1]))
    config = frozen.config

    @jax.jit
    def evaluate(params: Any, models: Any, state: jax.Array) -> jax.Array:
        return jax.lax.map(
            lambda pair: rollout_actuator_skill_library(
                pair[0], frozen.contract.spec, state, pair[1], config
            ),
            (params, models),
        ).states

    rows, arrays = [], {}
    for path in samples:
        adaptive = load_actuator_learner_checkpoint(path.with_suffix(""))
        when = adaptive.metadata["physical_time_seconds"]
        state = jnp.asarray(adaptive.physical_state)
        model = scene.model_at(when, frozen.contract.model)
        params = jax.tree.map(
            lambda *v: jnp.stack(v),
            adaptive.contract.params,
            frozen.state.params,
            adaptive.state.params,
        )
        models = jax.tree.map(lambda *v: jnp.stack(v), adaptive.contract.model, model, model)
        values = np.asarray(evaluate(params, models, state))
        assert np.all(np.isfinite(values))
        reference = values[0]
        row = {
            "time_seconds": when,
            "complete_state_sha256": _hash_tree(state),
            "current_model_sha256": _hash_tree(model),
            "adaptive_online_versions": int(adaptive.state.library_version)
            - int(frozen.state.library_version),
            "snapshot_sha256": sha(path.with_suffix(".npz")),
            "libraries": {},
        }
        for label, value in zip(("frozen", "adaptive"), values[1:], strict=True):
            delta = value - reference
            position = np.sqrt(np.mean(np.sum(delta[..., :3] ** 2, axis=-1), axis=-1))
            velocity = np.sqrt(np.mean(np.sum(delta[..., 7:10] ** 2, axis=-1), axis=-1))
            terminal = np.linalg.norm(value[:, -1, 7:10], axis=-1)
            excess = np.maximum(terminal - np.linalg.norm(reference[:, -1, 7:10], axis=-1), 0)
            row["libraries"][label] = {
                "mean_position_rmse_m": float(np.mean(position)),
                "worst_position_rmse_m": float(np.max(position)),
                "mean_velocity_rmse_mps": float(np.mean(velocity)),
                "worst_terminal_speed_mps": float(np.max(terminal)),
                "mean_terminal_speed_excess_mps": float(np.mean(excess)),
                "per_skill_position_rmse_m": position.tolist(),
                "per_skill_terminal_speed_mps": terminal.tolist(),
            }
        arrays[f"t{when:.9f}-states"] = values
        arrays[f"t{when:.9f}-initial"] = np.asarray(state)
        rows.append(row)
    np.savez_compressed(args.output / "rollouts.npz", **arrays)
    report = {
        "rows": rows,
        "scene_spec": plan.get("spec", plan.get("trial")),
        "actor_config": asdict(config),
        "source_initial_checkpoint_sha256": frozen.sha256,
        "driver_sha256": sha(Path(__file__)),
        "scope": "Each comparison fixes the complete physical/motor state, current model "
        "and horizon. The immutable nominal teacher is evaluated from that same state. "
        "These are behavior-recovery metrics, not proof of obstacle avoidance.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "driver.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            [
                {k: v for k, v in r.items() if k != "libraries"}
                | {
                    label: {k: v for k, v in values.items() if not k.startswith("per_skill")}
                    for label, values in r["libraries"].items()
                }
                for r in rows
            ],
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
