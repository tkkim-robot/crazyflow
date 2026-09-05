"""Create separate MuJoCo motor-off contact artifacts from recorded failures or a fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import numpy as np

from crazyflow.safety.da_plcbf.contact_replay import (
    ContactReplayConfig,
    ContactTrigger,
    ObstacleMotion,
    cf21b_contact_body,
    find_contact_trigger,
    navigation_contact_replay,
    run_contact_replay,
    save_contact_replay,
)


def main() -> None:
    """Run actual contact dynamics without rerunning the original controller or learner."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--legacy-replay", type=Path)
    source.add_argument("--fixture", action="store_true")
    parser.add_argument("--method", choices=("fixed", "adaptive"), default="adaptive")
    parser.add_argument("--branch", default="estimated_learning")
    parser.add_argument(
        "--trigger",
        choices=("physical_contact", "unsafe_shell", "degraded", "unsafe"),
        default="unsafe",
    )
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite contact evidence")
    config = ContactReplayConfig(duration_seconds=args.duration, timestep=args.timestep)
    if args.run:
        result = navigation_contact_replay(
            args.run, method=args.method, trigger_kind=args.trigger, config=config
        )
    elif args.fixture:
        state = np.asarray([-0.8, 0, 1.2, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0.0])
        obstacles = ObstacleMotion(
            np.asarray([0.0, args.duration + 1]),
            np.asarray([[[0.0, 0, 1]], [[0.0, 0, 1]]]),
            np.asarray([0.22]),
        )
        result = run_contact_replay(
            ContactTrigger(0, state, "synthetic_impact_fixture_motor_cut", 0, None),
            cf21b_contact_body(),
            obstacles,
            config,
        )
        result.metadata["source"] = {"kind": "explicit synthetic impact/drop fixture"}
    else:
        replay_json = args.legacy_replay / "replay.json"
        replay = json.loads(replay_json.read_text())
        source_directory = Path(replay["source_directory"])
        scenario_path = source_directory / "feasibility_reference.json"
        scenario = json.loads(scenario_path.read_text())["scenario"]
        source_config = json.loads((source_directory / "competent_comparison.json").read_text())[
            "summary"
        ]["config"]
        replay_npz = args.legacy_replay / "replay.npz"
        with np.load(replay_npz, allow_pickle=False) as raw:
            states = raw[f"branch_{args.branch}_states"]
            modes = raw[f"branch_{args.branch}_modes"]
        start = replay["pre_failure_closed_loop_branches"][args.branch]["start_time"]
        times = start + np.arange(len(states)) * scenario["dt"]
        support = np.asarray([start, times[-1] + args.duration + 1])
        mask = np.asarray(scenario["obstacle_mask"], dtype=bool)
        centers = np.asarray(scenario["obstacle_initial_centers"])[mask]
        velocities = np.asarray(scenario["obstacle_velocities"])[mask]
        obstacles = ObstacleMotion(
            support,
            centers[None] + support[:, None, None] * velocities[None],
            np.asarray(scenario["obstacle_radii"])[mask],
        )
        control_times = (
            start + np.arange(len(modes)) * scenario["dt"] * source_config["control_interval_steps"]
        )
        trigger = find_contact_trigger(
            times,
            states,
            obstacles,
            config,
            kind=args.trigger,
            shell_ego_radius=scenario["ego_radius"],
            shell_clearance=scenario["obstacle_clearance"],
            degraded_times=control_times[modes == 2],
        )
        replay_times = (
            trigger.time_seconds
            + np.arange(int(np.ceil(args.duration / args.timestep)) + 1) * args.timestep
        )
        wind = np.tile(scenario["wind_before"], (len(replay_times), 1))
        wind[replay_times >= scenario["wind_change_step"] * scenario["dt"]] = scenario["wind_after"]
        result = run_contact_replay(
            trigger, cf21b_contact_body(), obstacles, config, wind_velocity_world=wind
        )
        result.metadata["source"] = {
            "kind": "archived legacy pre-failure branch",
            "replay_directory": str(args.legacy_replay.resolve()),
            "branch": args.branch,
            "recorded_point_model_ego_radius_m": scenario["ego_radius"],
            "input_sha256": {
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (replay_json, replay_npz, scenario_path)
            },
        }
        result.metadata["obstacle_motion"] = (
            "Archived constant-velocity trajectories evaluated at their original absolute time."
        )
    result.metadata["command"] = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    )
    result.metadata["cli_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    directory = save_contact_replay(result, args.output)
    (directory / "REPRODUCE.txt").write_text(
        result.metadata["command"]
        + "\n\nUse a fresh output directory. "
        "This creates a separately labeled motor-off continuation.\n"
    )
    print(
        json.dumps(
            {
                "output": str(directory),
                "trigger": result.metadata["trigger"],
                "obstacle_contact_steps": result.metadata["obstacle_contact_steps"],
                "ground_contact_steps": result.metadata["ground_contact_steps"],
                "contact_events": result.events[:5],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
