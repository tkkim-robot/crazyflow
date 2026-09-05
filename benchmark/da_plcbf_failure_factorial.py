"""Small full factorial at pre-failure recorded states; no factor is a learning input.

This isolates fallback behavior values. The task nominal command and prediction are saved once
per recorded boundary and excluded from fallback maxima, so augmenting with a task policy cannot
hide a library loss. Interactions are reported as full cells, not a uniquely causal ordered sum.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.competent_library_experiment import (
    CompetentExperimentConfig,
    _scenario,
)
from crazyflow.safety.da_plcbf.continuous_demo_scenarios import (
    model_with_wind,
    scenario_obstacle_window,
)
from crazyflow.safety.da_plcbf.continuous_version_a import (
    conservative_smooth_policy_values,
    runtime_policy_values,
)
from crazyflow.safety.da_plcbf.learner_checkpoint import load_learner_checkpoint
from crazyflow.safety.da_plcbf.persistent_skill_learner import rollout_skill_library


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    replay = json.loads((args.replay / "replay.json").read_text())
    source = Path(replay["source_directory"])
    metadata = json.loads((source / "competent_comparison.json").read_text())["summary"]
    config = CompetentExperimentConfig(**metadata["config"])
    scenario = _scenario(config)
    device = jax.devices("gpu")[0]
    initial = load_learner_checkpoint(Path(metadata["checkpoint"]), device=device)
    with np.load(source / "competent_comparison.npz", allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    evaluators = {}
    for compensation, hold in itertools.product((False, True), (1, 2)):
        learning_config = replace(
            initial.config, model_compensation=compensation, control_interval_steps=hold
        )

        def evaluate(
            params: Any,
            state: Any,
            model: Any,
            obstacles: Any,
            learning_config: Any = learning_config,
        ) -> Any:
            rollout = rollout_skill_library(
                params, initial.spec, state, model, initial.actuator, learning_config
            )
            values = runtime_policy_values(
                rollout.states,
                obstacles,
                obstacle_clearance=scenario.obstacle_clearance,
                ego_radius=scenario.ego_radius,
            )
            smooth = conservative_smooth_policy_values(
                values, temperature=0.005, max_gap_budget=0.03
            )
            return rollout, values, smooth

        evaluators[compensation, hold] = jax.jit(evaluate)
    arrays, rows = {}, []
    for when in (4.04, 4.16, 4.68):
        index = int(np.flatnonzero(np.isclose(trace["time_seconds"], when))[0])
        current = load_learner_checkpoint(args.replay / f"snapshot-{index:04d}", device=device)
        preceding = load_learner_checkpoint(
            args.replay / f"snapshot-{index - 1:04d}", device=device
        )
        params = {
            "initial": initial.state.params,
            "previous": preceding.state.params,
            "current": current.state.params,
        }
        models = {
            "estimate": model_with_wind(
                initial.point_model, jnp.asarray(trace["adaptive_estimated_wind"][index])
            ),
            "oracle": model_with_wind(initial.point_model, jnp.asarray(config.wind_after)),
        }
        states = {
            "previous": jnp.asarray(trace["adaptive_full_state"][index - 1]),
            "current": jnp.asarray(trace["adaptive_full_state"][index]),
        }
        prediction = scenario_obstacle_window(scenario, round(when / config.dt))
        arrays[f"boundary_{index}_nominal_wrench"] = trace["adaptive_nominal_wrench"][index]
        arrays[f"boundary_{index}_nominal_rollout"] = trace["adaptive_nominal_rollout"][index]
        for parameter, model, state, compensation, hold in itertools.product(
            params, models, states, (False, True), (1, 2)
        ):
            rollout, values, smooth = jax.block_until_ready(
                evaluators[compensation, hold](
                    params[parameter], states[state], models[model], prediction
                )
            )
            key = f"b{index}_{parameter}_{model}_{state}_comp{int(compensation)}_hold{hold}"
            for name, value in {
                "states": rollout.states,
                "wrenches": rollout.wrenches,
                "motors": rollout.bounded_motor_forces,
                "hard": values.values,
                "smooth": smooth,
                "valid": rollout.policy_valid,
            }.items():
                arrays[f"{key}_{name}"] = np.asarray(value)
            terminal = np.linalg.norm(np.asarray(rollout.states[:, -1, 7:10]), axis=-1)
            rows.append(
                {
                    "key": key,
                    "boundary_time": when,
                    "parameters": parameter,
                    "model": model,
                    "state": state,
                    "compensation": compensation,
                    "policy_hold_steps": hold,
                    "actual_command_hold_steps": config.control_interval_steps,
                    "fallback_max_hard": float(np.max(values.values)),
                    "fallback_max_smooth": float(np.max(smooth)),
                    "nonnegative_smooth_count": int(np.sum(np.asarray(smooth) >= 0)),
                    "terminal_speed_mean": float(np.mean(terminal)),
                    "terminal_speed_p95": float(np.percentile(terminal, 95)),
                    "previous_version": int(preceding.state.library_version),
                    "current_version": int(current.state.library_version),
                }
            )
    np.savez_compressed(args.output / "factorial_trajectories.npz", **arrays)
    differences = []
    for cell in rows:
        if cell["parameters"] != "current":
            continue
        for before in ("previous", "initial"):
            match = next(
                row
                for row in rows
                if row["parameters"] == before
                and all(
                    row[key] == cell[key]
                    for key in (
                        "boundary_time",
                        "model",
                        "state",
                        "compensation",
                        "policy_hold_steps",
                    )
                )
            )
            differences.append(
                {
                    **{
                        key: cell[key]
                        for key in (
                            "boundary_time",
                            "model",
                            "state",
                            "compensation",
                            "policy_hold_steps",
                        )
                    },
                    "parameter_reference": before,
                    "delta_fallback_max_hard": cell["fallback_max_hard"]
                    - match["fallback_max_hard"],
                    "delta_terminal_speed_mean": cell["terminal_speed_mean"]
                    - match["terminal_speed_mean"],
                }
            )
    report = {
        "source_replay": str(args.replay),
        "cells": rows,
        "fixed_state_model_parameter_differences": differences,
        "scope": (
            "Full 3 parameter x 2 model x 2 state x 2 compensation x 2 cadence cells at each "
            "boundary. Absolute-time prediction fixed per boundary. Fallback-only values; task "
            "nominal stored separately. Interactions do not imply a unique causal decomposition."
        ),
    }
    (args.output / "factorial.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"output": str(args.output), "cells": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
