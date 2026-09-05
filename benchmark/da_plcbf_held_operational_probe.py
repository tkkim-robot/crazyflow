"""Reproduce and revise the immutable crossing 6.2 s held-operational fixture."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.competent_library_experiment import CompetentExperimentConfig
from crazyflow.safety.da_plcbf.continuous_demo_scenarios import (
    ContinuousDemoScenario,
    scenario_obstacle_window,
    scenario_safety_limits,
)
from crazyflow.safety.da_plcbf.continuous_version_a import (
    ContinuousVersionAConfig,
    PolicyRollouts,
    continuous_version_a_step,
    rollout_waypoint_library,
)
from crazyflow.safety.da_plcbf.learner_checkpoint import load_learner_checkpoint
from crazyflow.safety.da_plcbf.persistent_skill_learner import rollout_skill_library
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.version_a_barriers import (
    VersionABarrierConfig,
    VersionAModel,
    safety_constraint_names,
)
from crazyflow.safety.da_plcbf.version_a_filter import VersionAFilterConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--timing-calls", type=int, default=5)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    directory = args.input_directory
    fixture = json.loads((directory / "alternative_policy_probe.json").read_text())
    summary = json.loads((directory / "competent_comparison.json").read_text())["summary"]
    config = CompetentExperimentConfig(**summary["config"])
    device = jax.devices(args.device)[0]
    bundle = load_learner_checkpoint(Path(summary["checkpoint"]), device=device)
    state = jax.device_put(jnp.asarray(fixture["state"]), device)
    model = jax.device_put(
        VersionAModel(**{k: jnp.asarray(v) for k, v in fixture["point_model"].items()}), device
    )
    scenario_record = json.loads((directory / "feasibility_reference.json").read_text())["scenario"]
    scalar_fields = {
        "name",
        "dt",
        "steps",
        "horizon",
        "obstacle_clearance",
        "wind_change_step",
        "ego_radius",
    }
    scenario = ContinuousDemoScenario(
        **{k: v if k in scalar_fields else jnp.asarray(v) for k, v in scenario_record.items()}
    )
    prediction = scenario_obstacle_window(scenario, round(fixture["time_seconds"] / config.dt))
    safety = scenario_safety_limits(scenario)
    barriers = VersionABarrierConfig(
        obstacle_clearance=scenario.obstacle_clearance,
        arena_clearance=0.08,
        ego_radius=scenario.ego_radius,
        include_obstacle_hocbf=False,
    )
    records = []
    arrays = {}
    for cadence, iterations in ((1, 0), (2, 0), (2, 3)):
        learner_config = replace(
            bundle.config, model_compensation=True, control_interval_steps=cadence
        )
        continuous = ContinuousVersionAConfig(
            dt=config.dt,
            horizon=config.horizon,
            obstacle_clearance=scenario.obstacle_clearance,
            ego_radius=scenario.ego_radius,
            prefer_nominal_when_safe=False,
            control_interval_steps=2,
            predictive_operational_iterations=iterations,
        )

        def nominal(x: jax.Array, point: VersionAModel) -> PolicyRollouts:
            return rollout_waypoint_library(
                x,
                scenario.goal_position[None],
                scenario.goal_velocity[None],
                point,
                bundle.actuator,
                QuadPolicyConfig(acceleration_limit=config.nominal_acceleration_limit),
                dt=config.dt,
                horizon=config.horizon,
                position_gain=2.0,
                velocity_gain=2.8,
                model_compensation=True,
                command_hold_steps=cadence,
            )

        def fallbacks(x: jax.Array, point: VersionAModel) -> PolicyRollouts:
            r = rollout_skill_library(
                bundle.state.params, bundle.spec, x, point, bundle.actuator, learner_config
            )
            return PolicyRollouts(r.states, r.wrenches, jnp.all(r.policy_valid, axis=1))

        fn = jax.jit(
            lambda x: continuous_version_a_step(
                x,
                nominal,
                fallbacks,
                prediction,
                model,
                bundle.actuator,
                safety,
                barriers,
                VersionAFilterConfig(),
                continuous,
            )
        )
        result = jax.block_until_ready(fn(state))
        label = f"policy_hold_{cadence}_predictive_{iterations}"
        if cadence == 1 and iterations == 0:
            np.testing.assert_allclose(
                result.continuous_filter.qp.action,
                fixture["original"]["qp_action"],
                atol=2e-7,
                rtol=2e-5,
            )
            np.testing.assert_allclose(
                result.selected_smooth_value,
                fixture["original"]["selected_smooth_value"],
                atol=2e-6,
            )
            assert bool(result.qp_valid) == fixture["original"]["qp_accepted_including_hold"]
        timings = []
        for _ in range(args.timing_calls):
            started = time.perf_counter()
            jax.block_until_ready(fn(state))
            timings.append(time.perf_counter() - started)
        residuals = np.asarray(result.qp_held_operational_residuals)
        substep, row = np.unravel_index(np.argmin(residuals), residuals.shape)
        records.append(
            dict(
                label=label,
                policy_period_seconds=cadence * config.dt,
                qp_period_seconds=0.04,
                integration_dt=config.dt,
                selected_index=int(result.selected_index),
                hard_value=float(result.values.values[result.selected_index]),
                qp_action=np.asarray(result.continuous_filter.qp.action).tolist(),
                initial_held_operational_residual=float(
                    result.initial_qp_held_operational_residual
                ),
                minimum_held_operational_residual=float(residuals[substep, row]),
                limiting_constraint=safety_constraint_names(0)[row],
                limiting_substep=int(substep),
                limiting_time_offset_seconds=float(substep * config.dt),
                qp_accepted=bool(result.qp_valid),
                execution_mode=int(result.execution_mode),
                predictive_iterations=int(result.predictive_operational_iterations),
                applied_physical_margin=float(result.applied_held_operational_margin),
                held_collision_margin=float(result.qp_held_margin),
                kkt_valid=bool(result.continuous_filter.qp_kkt_valid),
                service_seconds=timings,
            )
        )
        for field in (
            "qp_held_operational_residuals",
            "fallback_held_operational_residuals",
            "applied_held_operational_residuals",
            "applied_held_physical_margins",
            "action",
            "qp_rejection_flags",
        ):
            arrays[label + "_" + field] = np.asarray(getattr(result, field))
        arrays[label + "_candidate_states"] = np.asarray(result.candidates.states)
        arrays[label + "_candidate_wrenches"] = np.asarray(result.candidates.wrenches)
        print(json.dumps(records[-1]), flush=True)
    np.savez_compressed(
        args.output_directory / "held_operational_probe.npz", state=np.asarray(state), **arrays
    )
    (args.output_directory / "held_operational_probe.json").write_text(
        json.dumps(
            dict(
                source_fixture=str(directory / "alternative_policy_probe.json"),
                operational_constraint_names=safety_constraint_names(0),
                records=records,
                interpretation=(
                    "Local predictive correction with unchanged nonlinear held postcheck; "
                    "no continuous-time or global sampled-data guarantee."
                ),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
