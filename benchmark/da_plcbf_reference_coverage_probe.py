"""Offline same-state/model coverage and QP checks of fixed ablation snapshots.

Geometry appears only here, after obstacle-free optimization has completed. This tool never
changes parameters, accepts a learner update, or claims that a counterfactual is an executed path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.continuous_demo_scenarios import (
    constant_wind_scenario,
    scenario_obstacle_window,
)
from crazyflow.safety.da_plcbf.continuous_version_a import QP_REJECTION_REASONS
from crazyflow.safety.da_plcbf.learner_checkpoint import load_learner_checkpoint
from crazyflow.safety.da_plcbf.online_constant_wind import VersionAResources, _make_controller


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--legacy-replay",
        type=Path,
        default=Path("artifacts/da_plcbf/navigation-revision-20260905/legacy-estimated-replay"),
    )
    parser.add_argument(
        "--legacy-trace",
        type=Path,
        default=Path(
            "artifacts/da_plcbf/competent-revision-20260904/wind-estimated-8/competent_comparison.npz"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = jax.devices(args.device)[0]
    initial = load_learner_checkpoint(args.ablation_dir / "provisional/checkpoint", device=device)
    summary = json.loads((args.ablation_dir / "ablation_summary.json").read_text())
    scenario = replace(
        constant_wind_scenario(), dt=initial.config.dt, horizon=initial.config.horizon
    )
    resources = VersionAResources(initial.point_model, initial.actuator)
    controllers = {}
    for bounded in (False, True):
        actor_config = replace(initial.config, velocity_offset_limit=0.35 if bounded else None)
        controllers[bounded] = _make_controller(
            scenario,
            resources,
            initial.spec,
            actor_config,
            nominal_acceleration_limit=1.2,
            waypoint_position_gain=2.0,
            waypoint_velocity_gain=2.8,
            device=device,
            nominal_model_compensation=True,
            control_interval_steps=actor_config.control_interval_steps,
        )
    with np.load(args.legacy_trace, allow_pickle=False) as data:
        recorded_wind = data["adaptive_estimated_wind"]
        recorded_state = data["adaptive_full_state"]
    rows, arrays, common_nominal = [], {}, {}
    for boundary in (100, 104):
        physical = load_learner_checkpoint(
            args.legacy_replay / f"snapshot-{boundary:04d}", device=device
        )
        np.testing.assert_array_equal(physical.physical_state, recorded_state[boundary])
        when = float(physical.metadata["simulation_time"])
        obstacles = scenario_obstacle_window(scenario, round(when / scenario.dt))
        previous = jnp.asarray(physical.metadata["previous_policy_index"], dtype=jnp.int32)
        for model_name, wind in (
            ("estimate", recorded_wind[boundary]),
            ("oracle", [4.0, 1.6, 0.0]),
        ):
            model = resources.model._replace(wind_velocity=jnp.asarray(wind, dtype=jnp.float32))
            for variant in summary["variants"]:
                name = variant["name"]
                bounded = variant["config"]["velocity_offset_limit"] is not None
                for updates in (0, 1, 4, 8, 20, 80):
                    checkpoint = (
                        initial
                        if updates == 0
                        else load_learner_checkpoint(
                            args.ablation_dir
                            / name
                            / ("final_checkpoint" if updates == 80 else f"update_{updates:03d}"),
                            device=device,
                        )
                    )
                    decision = controllers[bounded](
                        physical.physical_state, checkpoint.state.params, model, obstacles, previous
                    )
                    jax.block_until_ready(decision)
                    hard, smooth = (
                        np.asarray(decision.values.values),
                        np.asarray(decision.smooth_values),
                    )
                    candidate_valid = np.asarray(decision.candidates.valid) & np.asarray(
                        decision.values.input_valid
                    )
                    eligible = np.asarray(decision.continuous_filter.policy_eligible)
                    key = f"t{boundary}__{model_name}__{name}__u{updates:03d}"
                    nominal_key = (boundary, model_name)
                    nominal = np.asarray(decision.nominal_action)
                    if nominal_key not in common_nominal:
                        common_nominal[nominal_key] = nominal
                    np.testing.assert_array_equal(nominal, common_nominal[nominal_key])
                    arrays[f"{key}_states"] = np.asarray(decision.candidates.states)
                    arrays[f"{key}_actions"] = np.asarray(decision.candidates.wrenches)
                    flags = np.asarray(decision.qp_rejection_flags)
                    rows.append(
                        {
                            "key": key,
                            "time_seconds": when,
                            "model": model_name,
                            "wind": np.asarray(model.wind_velocity).tolist(),
                            "state": np.asarray(physical.physical_state).tolist(),
                            "variant": name,
                            "completed_updates": updates,
                            "checkpoint_sha256": checkpoint.sha256,
                            "nominal_action": nominal.tolist(),
                            "applied_action": np.asarray(decision.action).tolist(),
                            "hard_values": hard.tolist(),
                            "smooth_values": smooth.tolist(),
                            "candidate_valid": candidate_valid.tolist(),
                            "eligible": eligible.tolist(),
                            "fallback_max_hard": float(np.max(hard[1:])),
                            "fallback_max_smooth": float(np.max(smooth[1:])),
                            "fallback_hard_safe_count": int(
                                np.sum((hard[1:] >= 0) & candidate_valid[1:])
                            ),
                            "fallback_eligible_count": int(np.sum(eligible[1:])),
                            "augmented_max_hard": float(np.max(hard)),
                            "nominal_hard": float(hard[0]),
                            "qp_accepted": bool(decision.qp_valid),
                            "execution_mode": int(decision.execution_mode),
                            "degraded": bool(decision.degraded),
                            "selected_index": int(decision.selected_index),
                            "rejection_reasons": [
                                name
                                for name, flag in zip(QP_REJECTION_REASONS, flags, strict=True)
                                if flag
                            ],
                        }
                    )
    np.savez_compressed(args.output_dir / "trajectories.npz", **arrays)
    result = {
        "scope": (
            "offline counterfactuals from new nominal teacher and fixed-state-trained snapshots; "
            "not legacy replay or executed navigation"
        ),
        "ablation_directory": str(args.ablation_dir),
        "legacy_trace": str(args.legacy_trace),
        "controller_source_sha256": hashlib.sha256(
            Path("crazyflow/safety/da_plcbf/online_constant_wind.py").read_bytes()
        ).hexdigest(),
        "common_nominal_action_verified": True,
        "cells": rows,
    }
    with (args.output_dir / "coverage.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output_dir), "cells": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
