"""Fixed development demonstration: preflight then navigation with physical wind."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from benchmark.da_plcbf_actuator_diagnostics import DiagnosticResources
from benchmark.da_plcbf_recovery_diagnosis import begin, write
from crazyflow.safety.da_plcbf.actuator_experiment import (
    ActuatorEpisodeConfig,
    run_actuator_episode,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_study import ActuatorScene, make_actuator_scene
from crazyflow.safety.da_plcbf.navigation_world import WindEvent


def demonstration_scene() -> ActuatorScene:
    scene = make_actuator_scene(62104, "navigation", "effectiveness")
    config = replace(
        scene.world.config,
        duration_seconds=43.0,
        obstacle_time_offset_seconds=19.0,
        wind_events=(
            WindEvent(3.0, (1.6, 0.8, 0.0)),
            WindEvent(11.0, (0.0, 0.0, 0.0)),
            WindEvent(23.0, (1.2, -0.6, 0.0)),
            WindEvent(29.0, (-1.0, 0.8, 0.0)),
            WindEvent(37.0, (0.0, 0.0, 0.0)),
        ),
    )
    config.validate()
    return replace(
        scene, world=replace(scene.world, config=config), navigation_start=19.0, event_time=21.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    begin(args.output, "One fixed wind demonstration; three methods; no new validation worlds.")
    (args.output / "driver.py").write_bytes(Path(__file__).read_bytes())
    scene = demonstration_scene()
    write(
        args.output / "plan.json",
        {
            "scene": scene.metadata(),
            "methods": ["PD_F", "F2", "A_BAL"],
            "shared_adapter": "F2 with current observed wind; no future wind schedule",
            "learning": "persistent learner and optimizer across preflight/navigation",
        },
    )
    resources = DiagnosticResources()
    fc = ActuatorFilterConfig()
    records = []
    for method in ("PD_F", "F2", "A_BAL"):
        bundle, controller, learner, metadata = resources.resolve(method, fc)
        config = ActuatorEpisodeConfig(
            method=method,
            command_governor="committed_backup",
            allow_wind=True,
            filter_config=fc,
            retain_rollouts=True,
            save_checkpoints=True,
        )
        print(json.dumps({"starting": method}), flush=True)
        result = run_actuator_episode(
            scene,
            bundle,
            config,
            args.output / method / "attempt-00",
            controllerfunctions=controller,
            learner_functions=learner,
        )
        records.append({"method": method, "summary": result.summary, "metadata": metadata})
        write(args.output / "report.json", {"records": records, "planned": 3})
        print(
            json.dumps(
                {
                    "finished": method,
                    "safe_task": result.summary["successful_full_episode"],
                    "termination": result.summary["termination"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
