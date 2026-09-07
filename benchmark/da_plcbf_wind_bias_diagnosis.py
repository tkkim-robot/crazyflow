"""Separate shared wind compensation from visible changes in learned fallback paths."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.da_plcbf_recovery_analysis import BASE as SOURCE
from benchmark.da_plcbf_recovery_analysis import sha
from benchmark.da_plcbf_three_panel_contact import BASE
from crazyflow.safety.da_plcbf.actuator_learning import (
    load_actuator_learner_checkpoint,
    rollout_actuator_skill_library,
)


def main() -> None:
    output = BASE / "wind-bias-v2"
    output.mkdir(parents=True, exist_ok=False)
    directory = SOURCE / "wind-demo-v1"
    bundle = load_actuator_learner_checkpoint(directory / "F2/attempt-00/snapshots/initial")
    model = bundle.contract.model
    with np.load(directory / "F2/attempt-00/controls.npz") as controls:
        index = int(np.argmin(abs(controls["time"] - 2.96)))
        state = jnp.asarray(controls["controller_input_state"][index])
    wind = jnp.asarray([1.6, 0.8, 0.0])
    windy = model._replace(body=model.body._replace(wind_velocity=wind))
    calm = model._replace(body=model.body._replace(wind_velocity=jnp.zeros(3)))
    direction = np.asarray(wind) / np.linalg.norm(wind)
    rows = []
    saved = {}
    for compensated in (True, False):
        config = replace(bundle.config, model_compensation=compensated)
        rollout = jax.jit(
            lambda m: rollout_actuator_skill_library(
                bundle.state.params, bundle.contract.spec, state, m, config
            )
        )
        before, after = [np.asarray(rollout(m).states) for m in (calm, windy)]
        delta = after[..., :3] - before[..., :3]
        rows.append(
            {
                "model_compensation": compensated,
                "mean_endpoint_shift_along_wind_m": float(np.mean(delta[:, -1] @ direction)),
                "endpoint_rms_shift_m": float(np.sqrt(np.mean(np.sum(delta[:, -1] ** 2, axis=-1)))),
                "whole_path_rms_shift_m": float(np.sqrt(np.mean(np.sum(delta**2, axis=-1)))),
            }
        )
        saved[f"compensation_{compensated}_calm"] = before
        saved[f"compensation_{compensated}_wind"] = after
    actual = []
    with (
        np.load(directory / "F2/attempt-00/controls.npz") as frozen,
        np.load(directory / "A_BAL/attempt-00/controls.npz") as adaptive,
    ):
        for when in (2.96, 3.0, 3.04, 4.0, 7.0, 10.96, 11.0, 18.96, 21.0, 25.0):
            index = int(np.argmin(abs(frozen["time"] - when)))
            a, f = [c["candidate_states"][index, 1:, :, :3] for c in (adaptive, frozen)]
            delta = (a - a[:, :1]) - (f - f[:, :1])
            actual.append(
                {
                    "time": when,
                    "relative_fan_rms_difference_m": float(
                        np.sqrt(np.mean(np.sum(delta**2, axis=-1)))
                    ),
                    "adaptive_online_versions": int(adaptive["control_library_version"][index])
                    - int(adaptive["control_library_version"][0]),
                }
            )
    old_path = (
        SOURCE.parents[1]
        / "hover-explanation-20260905/hover-wind-payload-navigation/navigation_comparison.json"
    )
    old = json.loads(old_path.read_text())
    report = {
        "old_preflight_config": old["summary"]["config"],
        "old_preflight_source_sha256": sha(old_path),
        "current_actor_config_model_compensation": bundle.config.model_compensation,
        "same_state_time": 2.96,
        "same_parameters_sha256": bundle.sha256,
        "wind_mps": [1.6, 0.8, 0],
        "same_state_compensation_ablation": rows,
        "actual_video_relative_fan_comparison": actual,
        "qualification": (
            "Compensation ablation holds state, learned parameters, wind, motor model and horizon "
            "fixed. It is a rollout diagnostic, not a new flight or retrained comparison. "
            "Actual video fan comparison uses each flight's own recorded state."
        ),
        "driver_sha256": sha(Path(__file__)),
    }
    np.savez_compressed(output / "rollouts.npz", state=np.asarray(state), **saved)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
