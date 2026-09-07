"""Same-state repertoire and governor probes, preserving incoming adaptive memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.da_plcbf_hover_followup import restore_scene
from benchmark.da_plcbf_hover_return import resources
from benchmark.da_plcbf_recovery_diagnosis import sha, write
from crazyflow.safety.da_plcbf.actuator_backup import BackupSnapshot, CommittedBackupController
from crazyflow.safety.da_plcbf.actuator_experiment import _hash_tree
from crazyflow.safety.da_plcbf.actuator_inputs import CausalObservationInputCache
from crazyflow.safety.da_plcbf.actuator_learning import load_actuator_learner_checkpoint
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_study import (
    ActuatorObservationConfig,
    nominal_actuator_model,
    observe_actuator_state,
)
from crazyflow.safety.da_plcbf.persistent_skill_learner import SkillActorParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.case / "scene-plan.json").read_text())
    scene = restore_scene(plan["scene"])
    episode = args.case / "A_BAL/attempt-00"
    binding = json.loads((episode / "binding.json").read_text())
    filters = ActuatorFilterConfig(**binding["config"]["filter_config"])
    bundle, functions, _, _ = resources().resolve("A_BAL", filters)
    governor = CommittedBackupController(functions, bundle.contract.spec, bundle.config, filters)
    with np.load(episode / "controls.npz") as archive:
        controls = {
            k: archive[k]
            for k in (
                "time",
                "planned_command",
                "control_params_sha256",
                "backup_backup_params_sha256",
                "backup_executed_backup",
                "selected_index",
            )
        }
    observation = ActuatorObservationConfig(**binding["config"]["observation_config"])
    inputs = CausalObservationInputCache(scene, nominal_actuator_model(), observation, filters)
    inputs.prediction_inputs(0.0)
    records = []
    for when in (1.96, 2.0, 3.0, 4.0, 5.4, 5.6, 6.0, 16.0, 18.0, 20.0, 22.0):
        checkpoint = load_actuator_learner_checkpoint(episode / f"snapshots/critical-{when:.9f}")
        state = jnp.asarray(
            observe_actuator_state(checkpoint.physical_state, when, scene.scene_seed, observation)
        )
        memory = checkpoint.metadata["controller_memory"]
        model = inputs.model_at(when)
        obstacles, safety = inputs.prediction_inputs(when)
        index = int(np.flatnonzero(np.isclose(controls["time"], when, atol=1e-10, rtol=0))[0])
        row = dict(
            time_seconds=when,
            state_sha256=_hash_tree(state),
            model_sha256=_hash_tree(model),
            incoming_memory_sha256=checkpoint.metadata["controller_memory_sha256"],
            contexts={},
        )
        for label, params, retain in [
            ("adaptive_actual_memory", checkpoint.state.params, True),
            ("frozen_actual_memory", bundle.state.params, True),
            ("adaptive_empty_memory", checkpoint.state.params, False),
            ("frozen_empty_memory", bundle.state.params, False),
        ]:
            mem = memory["governor"]
            governor.generation = mem["generation"] if retain else 0
            backup = mem["backup"] if retain else None
            governor.backup = (
                None
                if backup is None
                else BackupSnapshot(
                    **{
                        **backup,
                        "params": SkillActorParams(
                            **{
                                k: jnp.asarray(v, dtype=jnp.float32)
                                for k, v in backup["params"].items()
                            }
                        ),
                        "anchor": jnp.asarray(backup["anchor"], dtype=jnp.float32),
                    }
                )
            )
            step = governor.controller_at(
                state,
                params,
                model,
                obstacles,
                safety,
                jnp.asarray(memory["previous_selected_index"]),
                jnp.asarray(scene.world.initial_state[:3]),
                when=when,
                commit=True,
            )
            jax.block_until_ready(step.action)
            audit = step.backup_audit
            values = dict(
                params_sha256=_hash_tree(params),
                eligible_indices=np.flatnonzero(np.asarray(step.certificates.eligible)).tolist(),
                selected_index=int(step.selected_index),
                selected_policy_dual=float(step.selected_policy_dual),
                executed_policy_dual=float(step.executed_policy_dual),
                mode=int(step.execution_mode),
                action=np.asarray(step.action).tolist(),
                qp_valid=bool(step.qp_valid),
                backup={
                    k: v
                    for k, v in audit.items()
                    if k not in ("checked_trajectory", "proposal_original_action")
                },
            )
            if label == "adaptive_actual_memory":
                np.testing.assert_array_equal(
                    np.asarray(step.action), controls["planned_command"][index]
                )
                assert _hash_tree(params) == controls["control_params_sha256"][index]
                assert (
                    audit["backup_params_sha256"] == controls["backup_backup_params_sha256"][index]
                )
                assert audit["executed_backup"] == bool(controls["backup_executed_backup"][index])
                values["exact_recorded_decision_reproduced"] = True
            row["contexts"][label] = values
        records.append(row)
        write(
            args.output / "report.json",
            dict(
                records=records,
                complete=len(records) == 11,
                scope="All probes fix the complete 17-state, current model, obstacle clock, "
                "goal and selector. Actual-memory probes also preserve incoming stored backup "
                "parameters, anchor, phase and deadline. Empty-memory probes isolate fresh-bank "
                "availability and are diagnostic resets, not flight ablations. Adaptive "
                "actual-memory action and stored-parameter hashes must reproduce exactly.",
            ),
        )
        print(
            json.dumps(
                {
                    "time": when,
                    "contexts": {
                        k: {
                            a: b
                            for a, b in v.items()
                            if a in ("eligible_indices", "mode", "qp_valid")
                        }
                        for k, v in row["contexts"].items()
                    },
                }
            ),
            flush=True,
        )
    (args.output / "driver.py").write_bytes(Path(__file__).read_bytes())
    write(
        args.output / "sources.json",
        {
            "driver_sha256": sha(Path(__file__)),
            "source_binding_sha256": sha(episode / "binding.json"),
        },
    )


if __name__ == "__main__":
    main()
