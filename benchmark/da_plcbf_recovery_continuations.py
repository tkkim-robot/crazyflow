"""Controlled physical continuations and phase-faithful fallback replays.

Every branch starts at a saved complete physical state and absolute obstacle
clock. Snapshot-holding and restoration here are interventions, not update gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.da_plcbf_actuator_confirmation import (
    _recorded_inputs,
    _world_from_binding,
    load_saved_run,
)
from benchmark.da_plcbf_actuator_diagnostics import DiagnosticResources
from benchmark.da_plcbf_recovery_diagnosis import DIAGNOSTIC_HARM, begin, sha, snapshot, write
from crazyflow.safety.da_plcbf.actuator_backup import CommittedBackupController
from crazyflow.safety.da_plcbf.actuator_compute import MatchedEffortPlant
from crazyflow.safety.da_plcbf.actuator_experiment import (
    _audit_interval,
    _filter_record,
    _hash_tree,
    _operational_margins,
    _strictly_separated,
)
from crazyflow.safety.da_plcbf.actuator_learning import (
    acceleration_to_actuator_command,
    actuator_skill_actions,
)
from crazyflow.safety.da_plcbf.actuator_study import (
    ActuatorObservationConfig,
    observe_actuator_state,
)

STARTS = (2.48, 3.2, 3.64, 3.68)


def continuation(run: Any, resources: Any, when: float, mode: str, directory: Path) -> dict:
    directory.mkdir(exist_ok=False)
    index = run.boundary_index(when)
    saved, onset = snapshot(index), snapshot(run.boundary_index(2.0))
    y, model, _, _, previous, _, fc = _recorded_inputs(run, index, when)
    assert _hash_tree(saved.state.params) == str(run.controls["control_params_sha256"][index])
    np.testing.assert_array_equal(saved.physical_state, run.controls["actual_state"][index])
    bundle, functions, learner, _ = resources.resolve("A_BAL", fc)
    governor = (
        CommittedBackupController(functions, bundle.contract.spec, bundle.config, fc)
        if mode == "guard"
        else None
    )
    world = _world_from_binding(run)
    observation = ActuatorObservationConfig(**run.binding["config"]["observation_config"])
    plant = MatchedEffortPlant(
        saved.physical_state, model, time=when, max_step=run.binding["config"]["plant_step_seconds"]
    )
    initial = plant.observe().copy()
    state = saved.state
    count = int(bundle.contract.spec.latent_codes.shape[0])
    skill = int(run.controls["selected_index"][index]) - 1
    assert skill >= 0, "declared phase replay starts must select a fallback, not the mission"
    phase_replay = mode in {"faithful", "restart"}
    end = when + fc.dt * fc.horizon if phase_replay else world.config.duration_seconds
    arrivals = [t for t in run.summary["waypoint_arrival_times_seconds"] if t < when]
    navigation_start = run.binding["scene"]["navigation_start"]
    times, states, commands, controls = [when], [initial], [], []
    collision = None
    min_operational = float("inf")

    @jax.jit
    def skill_command(current: Any, anchor: Any, elapsed: Any) -> Any:
        batch = jnp.broadcast_to(current, (count, 17))
        acceleration = actuator_skill_actions(
            saved.state.params,
            bundle.contract.spec,
            batch,
            anchor,
            elapsed / (fc.horizon * fc.dt),
            bundle.config,
        )
        return acceleration_to_actuator_command(acceleration, batch, model, bundle.config).command[
            skill
        ]

    tick = index
    while plant.time < end - 1e-10:
        current_time = tick * fc.command_period
        actual = plant.observe()
        if current_time >= navigation_start - 1e-10 and len(arrivals) < len(
            world.waypoint_positions
        ):
            if (
                np.linalg.norm(actual[:3] - world.waypoint_positions[len(arrivals)])
                <= world.config.reach_radius
            ):
                arrivals.append(current_time)
        goal_host = np.asarray(
            world.initial_state[:3]
            if current_time < navigation_start - 1e-10
            else world.waypoint_positions[min(len(arrivals), len(world.waypoint_positions) - 1)]
        )
        if tick == index:
            np.testing.assert_array_equal(goal_host, run.controls["goal"][index])
        goal = jnp.asarray(goal_host)
        observed = jnp.asarray(
            observe_actuator_state(
                actual, current_time, run.binding["scene"]["scene_seed"], observation
            )
        )
        obstacles = world.obstacle_prediction(current_time, dt=fc.dt, horizon=fc.horizon)
        safety = world.safety_limits(current_time)
        if phase_replay:
            anchor = y[:3] if mode == "faithful" else observed[:3]
            elapsed = (tick - index) * fc.command_period if mode == "faithful" else 0.0
            command = np.asarray(
                jax.block_until_ready(
                    skill_command(observed, anchor, jnp.asarray(elapsed, y.dtype))
                )
            )
            record = {
                "time": current_time,
                "mode": mode,
                "skill": skill,
                "elapsed_phase_seconds": elapsed,
                "anchor": np.asarray(anchor),
            }
        else:
            params = onset.state.params if mode == "restore_onset" else state.params
            step = (
                functions.controller(observed, params, model, obstacles, safety, previous, goal)
                if governor is None
                else governor.controller_at(
                    observed,
                    params,
                    model,
                    obstacles,
                    safety,
                    previous,
                    goal,
                    when=current_time,
                    commit=True,
                )
            )
            if tick == index:
                np.testing.assert_array_equal(
                    step.nominal_action, run.controls["nominal_command"][index]
                )
                if mode == "guard":
                    np.testing.assert_array_equal(
                        step.backup_audit["proposal_original_action"],
                        run.controls["planned_command"][index],
                    )
                elif mode != "restore_onset":
                    np.testing.assert_array_equal(
                        step.action, run.controls["planned_command"][index]
                    )
            command = np.asarray(step.action)
            previous = jnp.asarray(step.selected_index, jnp.int32)
            record = {
                "time": current_time,
                **_filter_record(step, False),
                "params_sha256": _hash_tree(params),
            }
        record["action"] = command
        controls.append(record)

        def audit_nodes(t: np.ndarray, x: np.ndarray) -> bool:
            nonlocal collision
            audit = _audit_interval(world, t, x)
            if audit["terminate"]:
                collision = audit
            return bool(audit["terminate"])

        def separated(t: np.ndarray, x: np.ndarray) -> bool:
            return _strictly_separated(_audit_interval(world, t, x))

        trace = plant.advance_audited(
            command,
            min(end, (tick + 1) * fc.command_period) - plant.time,
            audit_nodes,
            separation_precheck=separated,
        )
        times.extend(trace.times[1:].tolist())
        states.extend(trace.states[1:])
        commands.extend(np.broadcast_to(command, (len(trace.times) - 1, 4)))
        min_operational = min(
            min_operational,
            min(
                float(np.min(_operational_margins(world, x, fc.arena_clearance)))
                for x in trace.states
            ),
        )
        if collision is not None:
            break
        if mode in {"continue", "guard"} and plant.time < end - 1e-10:
            state, _ = jax.block_until_ready(learner.step(state, observed, model))
        tick += 1
    physical = np.asarray(states)
    report = {
        "mode": mode,
        "start_time": when,
        "end_time": plant.time,
        "planned_end_time": end,
        "collision": collision,
        "waypoints": len(arrivals),
        "waypoint_arrivals": arrivals,
        "minimum_operational_margin": min_operational,
        "safe_task": not phase_replay
        and collision is None
        and min_operational >= -1e-6
        and len(arrivals) == len(world.waypoint_positions),
        "source_episode": str(DIAGNOSTIC_HARM),
        "initial_physical_sha256": _hash_tree(initial),
        "initial_learner_sha256": _hash_tree(saved.state),
        "source_control_index": index,
        "selected_library_skill": skill,
        "horizon_replay_only": phase_replay,
        "prediction_initial_hard_value": float(run.controls["hard"][index, skill + 1]),
    }
    if phase_replay and collision is None:
        node_times = when + np.arange(fc.horizon + 1) * fc.dt
        locations = [int(np.argmin(abs(np.asarray(times) - t))) for t in node_times]
        predicted = run.controls["candidate_states"][index, skill + 1]
        report["prediction_vs_execution_max_state_delta"] = float(
            np.max(abs(physical[locations] - predicted))
        )
        report["prediction_vs_execution_max_position_delta"] = float(
            np.max(abs(physical[locations, :3] - predicted[:, :3]))
        )
    np.savez_compressed(
        directory / "physical.npz",
        time=np.asarray(times),
        state=physical,
        command=np.asarray(commands),
    )
    write(directory / "controls.json", controls)
    write(directory / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--guard-only", action="store_true")
    args = parser.parse_args()
    output = args.output
    begin(
        output,
        "Four fixed pre-loss times; each has continuation, held snapshot, onset restoration, "
        "faithful fallback and phase-reset fallback.",
    )
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    run = load_saved_run(DIAGNOSTIC_HARM)
    resources = DiagnosticResources()
    rows = []
    modes = ("continue", "hold_current", "restore_onset")
    if not args.control_only:
        modes += ("faithful", "restart")
    starts = STARTS
    if args.guard_only:
        modes, starts = ("guard",), STARTS[-1:]
    for when in starts:
        for mode in modes:
            print(json.dumps({"starting": when, "mode": mode}), flush=True)
            report = continuation(run, resources, when, mode, output / f"{when:.2f}-{mode}")
            rows.append(report)
            write(
                output / "report.json",
                {
                    "rows": rows,
                    "planned": len(starts) * len(modes),
                    "source_controls_sha256": sha(DIAGNOSTIC_HARM / "controls.npz"),
                },
            )
            print(
                json.dumps(
                    {
                        "finished": when,
                        "mode": mode,
                        "collision": report["collision"] is not None,
                        "end": report["end_time"],
                        "safe_task": report["safe_task"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
