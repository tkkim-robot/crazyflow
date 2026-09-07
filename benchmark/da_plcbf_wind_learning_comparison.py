"""Same three wind flights with direct wind feedforward disabled in every arm."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from benchmark.da_plcbf_actuator_diagnostics import DiagnosticResources
from benchmark.da_plcbf_recovery_diagnosis import begin, write
from benchmark.da_plcbf_recovery_wind import demonstration_scene
from crazyflow.safety.da_plcbf.actuator_experiment import (
    ActuatorEpisodeConfig,
    run_actuator_episode,
)
from crazyflow.safety.da_plcbf.actuator_learning import (
    load_actuator_learner_checkpoint,
    save_actuator_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/da_plcbf/wind-learning-comparison-20260907/v1"


def main() -> None:
    begin(BASE, "Fixed three-arm wind comparison; direct wind feedforward disabled in all arms.")
    (BASE / "driver.py").write_bytes(Path(__file__).read_bytes())
    resources = DiagnosticResources()
    initial = {}
    for method in ("PD_F", "F2", "A_BAL"):
        source = resources.bundle(method)
        config = replace(source.config, wind_feedforward=False)
        contract = replace(
            source.contract,
            actor_config=replace(source.contract.actor_config, wind_feedforward=False),
        )
        stem = BASE / "prepared" / method / "deployment"
        save_actuator_learner_checkpoint(
            source.state,
            contract,
            source.physical_state,
            stem,
            config=config,
            metadata={
                **source.metadata,
                "source_npz_sha256": source.sha256,
                "source_checkpoint": str(source.npz_path),
                "wind_feedforward": False,
                "initial_optimizer_preserved": True,
            },
        )
        restored = load_actuator_learner_checkpoint(stem)
        for a, b in zip(
            jax.tree.leaves(source.state), jax.tree.leaves(restored.state), strict=True
        ):
            np.testing.assert_array_equal(a, b)
        initial[method] = restored
        resources.paths[method] = stem
    for a, b in zip(
        jax.tree.leaves(initial["F2"].state), jax.tree.leaves(initial["A_BAL"].state), strict=True
    ):
        np.testing.assert_array_equal(a, b)
    scene = demonstration_scene()
    write(
        BASE / "plan.json",
        {
            "scene": scene.metadata(),
            "methods": ["PD_F", "F2", "A_BAL"],
            "wind_feedforward": False,
            "retained_shared_behavior": (
                "Ordinary feedback, calm-air drag compensation and lag-aware allocation"
            ),
            "wind_information": (
                "Actual wind retained in physics, safety prediction and learner model; "
                "excluded only from direct motor feedforward"
            ),
            "same_complete_learned_initial_state": True,
            "source_scene": "Identical 43-second scene to previous compensated three-panel video",
        },
    )
    rows = []
    filters = ActuatorFilterConfig()
    for method in ("PD_F", "F2", "A_BAL"):
        bundle, controller, learner, metadata = resources.resolve(method, filters)
        assert (
            not bundle.config.wind_feedforward and not bundle.contract.actor_config.wind_feedforward
        )
        print(json.dumps({"starting": method}), flush=True)
        config = ActuatorEpisodeConfig(
            method=method,
            command_governor="committed_backup",
            allow_wind=True,
            filter_config=filters,
            retain_rollouts=True,
            save_checkpoints=True,
            capture_times=(2.96, 3.0, 3.04, 4.0, 7.0, 10.96, 11.0, 18.96, 19.0, 21.0, 25.0),
        )
        result = run_actuator_episode(
            scene,
            bundle,
            config,
            BASE / "flights" / method / "attempt-00",
            controllerfunctions=controller,
            learner_functions=learner,
        )
        rows.append({"method": method, "summary": result.summary, "metadata": metadata})
        write(BASE / "report.json", {"records": rows, "planned": 3})
        print(
            json.dumps(
                {
                    "finished": method,
                    "termination": result.summary["termination"],
                    "safe_task": result.summary["successful_full_episode"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
