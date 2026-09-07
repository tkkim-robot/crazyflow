"""Closed three-case comparison of the committed-backup command governor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.da_plcbf_actuator_diagnostics import DiagnosticResources, resolve_trial_scene
from benchmark.da_plcbf_recovery_diagnosis import OLD, begin, sha, write
from crazyflow.safety.da_plcbf.actuator_experiment import (
    ActuatorEpisodeConfig,
    run_actuator_episode,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_study import ActuatorObservationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output
    if not args.resume:
        begin(
            output,
            "Fixed harm, gain and no-change development cases; "
            "shared command governor in all arms.",
        )
        (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    else:
        sources = json.loads((output / "execution.json").read_text())["source_sha256"]
        from benchmark.da_plcbf_recovery_diagnosis import ROOT

        assert all(sha(ROOT / name) == value for name, value in sources.items())
        assert sha(output / "driver.py") == sha(Path(__file__))
    prior = json.loads((OLD / "sealed-v4/protocol.json").read_text())
    resources = DiagnosticResources()
    records = []
    trials = [
        (case, method)
        for case in ("harm", "gain", "no_change_regression")
        for method in ("A_BAL", "F2", "PD_F", "ONSET_FREEZE")
    ]
    write(
        output / "plan.json",
        {
            "trials": trials,
            "prior_protocol_sha256": sha(OLD / "sealed-v4/protocol.json"),
            "governor": "committed_backup",
            "learner_numerics": "matched",
            "qp_numerics": "inward",
        },
    )
    for case, method in trials[: args.max_trials]:
        directory = output / f"{case}-{method}"
        if directory.exists():
            assert args.resume
            records.append(json.loads((directory / "record.json").read_text()))
            continue
        trial = next(
            t
            for t in prior["trials"]
            if t["case"] == case
            and t["method"] == method
            and t["learner_numerics"] == "matched"
            and t["qp_numerics"] == "inward"
        )
        original = OLD / "development-36-v3" / trial["id"] / "attempt-00"
        binding = json.loads((original / "binding.json").read_text())
        values = binding["config"]
        fc = ActuatorFilterConfig(**values.pop("filter_config"))
        observation = ActuatorObservationConfig(**values.pop("observation_config"))
        values.update(
            command_governor="committed_backup", retain_rollouts=True, save_checkpoints=True
        )
        config = ActuatorEpisodeConfig(filter_config=fc, observation_config=observation, **values)
        scene = resolve_trial_scene(prior["old_proposal"], trial["original_trial"], fc)
        bundle, controller, learner, metadata = resources.resolve(config.method, fc)
        directory.mkdir()
        print(json.dumps({"starting": case, "method": method}), flush=True)
        result = run_actuator_episode(
            scene,
            bundle,
            config,
            directory / "attempt-00",
            controllerfunctions=controller,
            learner_functions=learner,
        )
        record = {
            "case": case,
            "method": method,
            "summary": result.summary,
            "source_baseline": str(original),
            "method_metadata": metadata,
            "episode": str((directory / "attempt-00").resolve()),
        }
        write(directory / "record.json", record)
        records.append(record)
        write(
            output / "report.json", {"records": records, "planned": 12, "completed": len(records)}
        )
        print(
            json.dumps(
                {
                    "finished": case,
                    "method": method,
                    "termination": result.summary["termination"],
                    "safe_task": result.summary["successful_full_episode"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
