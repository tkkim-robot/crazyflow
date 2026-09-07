"""Reproduce selected hover cases and run learning-freeze or timing follow-ups."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import numpy as np

from benchmark.da_plcbf_actuator_diagnostics import DiagnosticResources
from benchmark.da_plcbf_hover_return import METHODS, resources
from benchmark.da_plcbf_recovery_diagnosis import begin, sha, write
from crazyflow.safety.da_plcbf.actuator_experiment import (
    ActuatorEpisodeConfig,
    _hash_tree,
    run_actuator_episode,
)
from crazyflow.safety.da_plcbf.actuator_hover import HoverReturnTask
from crazyflow.safety.da_plcbf.actuator_learning import load_actuator_learner_checkpoint
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_study import ActuatorObservationConfig, ActuatorScene
from crazyflow.safety.da_plcbf.navigation_world import (
    NavigationWorld,
    NavigationWorldConfig,
    WindEvent,
)


def restore_scene(metadata: dict[str, Any]) -> ActuatorScene:
    """Restore exact serialized geometry, including nonplanar mover offsets."""
    w = metadata["world"]
    options = dict(w["config"])
    options["wind_events"] = tuple(WindEvent(**event) for event in options["wind_events"])
    assert not options["payload_events"]
    options["payload_events"] = ()
    config = NavigationWorldConfig(**options)
    config.validate()
    arrays = {
        name: np.asarray(w[name])
        for name in (
            "initial_state",
            "waypoint_positions",
            "obstacle_mean_centers",
            "obstacle_amplitudes",
            "obstacle_angular_frequencies",
            "obstacle_phases",
            "obstacle_radii",
        )
    }
    # JSON empty lists need their declared vector dimensions restored.
    for name in ("obstacle_mean_centers", "obstacle_amplitudes"):
        arrays[name] = arrays[name].reshape(-1, 3)
    world = NavigationWorld(config=config, **arrays)
    values = {
        name: metadata[name]
        for name in (
            "family",
            "scene_seed",
            "dynamics_cell",
            "event_time",
            "effectiveness_after",
            "lag_multipliers_after",
            "navigation_start",
            "recovery_time",
        )
    }
    scene = ActuatorScene(world=world, **values)
    assert scene.metadata()["physical_world_id"] == metadata["physical_world_id"]
    return scene


def verify_common_history(source: Path, frozen: Path, when: float) -> dict[str, Any]:
    """A from-time-zero rerun preserves the governor history; freezing never resets it."""
    a = load_actuator_learner_checkpoint(source / f"snapshots/critical-{when:.9f}")
    b = load_actuator_learner_checkpoint(frozen / f"snapshots/critical-{when:.9f}")
    assert _hash_tree(a.state) == _hash_tree(b.state)
    np.testing.assert_array_equal(a.physical_state, b.physical_state)
    assert a.metadata["controller_memory_sha256"] == b.metadata["controller_memory_sha256"]
    assert a.metadata["controller_memory"] == b.metadata["controller_memory"]
    keys = (
        "time",
        "controller_input_state",
        "goal",
        "planned_command",
        "command_applied",
        "previous_command",
        "control_params_sha256",
        "control_library_version",
        "selected_index",
        "eligible",
        "qp_valid",
        "backup_proposal_accepted",
        "backup_executed_backup",
        "backup_used_stored_tail",
        "backup_has_checked_plan",
        "backup_remaining_checked_seconds",
        "backup_backup_skill_index",
        "backup_backup_generation",
        "backup_backup_anchor",
        "backup_backup_started_at",
        "backup_backup_certified_until",
        "backup_backup_params_sha256",
        "backup_checked_trajectory",
    )
    with np.load(source / "controls.npz") as x, np.load(frozen / "controls.npz") as y:
        left, right = x["time"] <= when + 1e-10, y["time"] <= when + 1e-10
        for key in keys:
            np.testing.assert_array_equal(x[key][left], y[key][right], err_msg=key)
        after = y["time"] >= when - 1e-10
        assert np.all(y["control_library_version"][after] == int(b.state.library_version))
        count = int(np.count_nonzero(left))
    memory = a.metadata["controller_memory"]
    backup = memory["governor"]["backup"]
    return {
        "freeze_time_seconds": when,
        "matched_control_count_through_onset": count,
        "full_learner_and_adam_sha256": _hash_tree(a.state),
        "complete_controller_memory_sha256": a.metadata["controller_memory_sha256"],
        "selector_memory": memory["previous_selected_index"],
        "retained_backup_present": backup is not None,
        "retained_backup_anchor": None if backup is None else backup["anchor"],
        "retained_backup_started_at": None if backup is None else backup["started_at"],
        "retained_backup_deadline": None if backup is None else backup["certified_until"],
        "semantics": "Both runs begin from identical complete initialization. Every pre-onset "
        "state, command, selector and retained backup matches. Freeze stops subsequent learner "
        "updates; it never clears or rebuilds the stored backup.",
    }


def run_scene_case(
    output: Path,
    scene: ActuatorScene,
    task: HoverReturnTask,
    cache: DiagnosticResources,
    *,
    plan: dict[str, Any],
    capture_times: tuple[float, ...],
    retain_rollouts: bool = False,
) -> list[dict[str, Any]]:
    """Run a complete matched triplet for explicitly supplied exogenous geometry."""
    output.mkdir(parents=True, exist_ok=False)
    write(output / "scene-plan.json", {**plan, "scene": scene.metadata(), "task": asdict(task)})
    rows = []
    for method in METHODS:
        bundle, controller, learner, metadata = cache.resolve(method, ActuatorFilterConfig())
        config = ActuatorEpisodeConfig(
            method=method,
            command_governor="committed_backup",
            allow_wind=True,
            hover_task=task,
            capture_times=tuple(sorted(set(capture_times))),
            save_checkpoints=True,
            retain_rollouts=retain_rollouts,
        )
        print(json.dumps({"starting": str(output), "method": method}), flush=True)
        result = run_actuator_episode(
            scene,
            bundle,
            config,
            output / method / "attempt-00",
            controllerfunctions=controller,
            learner_functions=learner,
        )
        rows.append({"method": method, "summary": result.summary, "metadata": metadata})
        write(output / "report.json", {"records": rows, "planned": len(METHODS)})
        print(
            json.dumps(
                {
                    "finished": str(output),
                    "method": method,
                    "termination": result.summary["termination"],
                    "safe_return": result.summary["successful_full_episode"],
                }
            ),
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("replay", "freeze", "delayed", "asynchronous", "compensated"),
        default="replay",
    )
    parser.add_argument("--freeze-at", type=float, default=2.0)
    parser.add_argument("--capture", type=float, nargs="*", default=[])
    args = parser.parse_args()
    plan = json.loads((args.case / "scene-plan.json").read_text())
    scene = restore_scene(plan["scene"])
    task_options = dict(plan["task"])
    task_options["return_windows"] = tuple(tuple(w) for w in task_options["return_windows"])
    task = HoverReturnTask(**task_options)
    methods = (
        ("A_BAL",)
        if args.mode == "freeze"
        else ("F2", "A_BAL")
        if args.mode == "compensated"
        else METHODS
    )
    begin(args.output, f"Selected hover follow-up: {args.mode}; actual source geometry preserved.")
    (args.output / "driver.py").write_bytes(Path(__file__).read_bytes())
    write(
        args.output / "scene-plan.json",
        {**plan, "source_case": str(args.case.resolve()), "followup_mode": args.mode},
    )
    cache = DiagnosticResources() if args.mode == "compensated" else resources()
    rows = []
    for method in methods:
        original = args.case / method / "attempt-00"
        binding = json.loads((original / "binding.json").read_text())
        options = dict(binding["config"])
        options["filter_config"] = ActuatorFilterConfig(**options["filter_config"])
        options["observation_config"] = ActuatorObservationConfig(**options["observation_config"])
        options["hover_task"] = task
        options["retain_rollouts"] = args.mode not in ("delayed", "asynchronous")
        options["save_checkpoints"] = True
        options["freeze_learning_at"] = args.freeze_at if args.mode == "freeze" else None
        options["execution_mode"] = (
            args.mode if args.mode in ("delayed", "asynchronous") else "deterministic"
        )
        capture = set(options["capture_times"]) | set(args.capture)
        if args.mode == "freeze":
            capture.add(args.freeze_at)
        options["capture_times"] = tuple(sorted(capture))
        config = ActuatorEpisodeConfig(**options)
        bundle, controller, learner, metadata = cache.resolve(method, config.filter_config)
        source_initial = load_actuator_learner_checkpoint(original / "snapshots/initial")
        for x, y in zip(
            jax.tree.leaves(bundle.state), jax.tree.leaves(source_initial.state), strict=True
        ):
            np.testing.assert_array_equal(x, y)
        print(json.dumps({"starting": method, "mode": args.mode}), flush=True)
        result = run_actuator_episode(
            scene,
            bundle,
            config,
            args.output / method / "attempt-00",
            controllerfunctions=controller,
            learner_functions=learner,
        )
        row = {
            "method": method,
            "summary": result.summary,
            "metadata": metadata,
            "source_binding_sha256": sha(original / "binding.json"),
        }
        if args.mode == "freeze" and result.summary["physical_time_seconds"] >= args.freeze_at:
            row["common_history"] = verify_common_history(
                original, args.output / method / "attempt-00", args.freeze_at
            )
        rows.append(row)
        write(
            args.output / "report.json",
            {"mode": args.mode, "records": rows, "planned": len(methods), "task": asdict(task)},
        )
        print(
            json.dumps(
                {
                    "finished": method,
                    "termination": result.summary["termination"],
                    "safe_return": result.summary["successful_full_episode"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
