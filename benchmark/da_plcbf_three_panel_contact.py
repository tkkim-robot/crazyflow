"""Motor-off contact continuation of the authenticated handcrafted wind flight."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from benchmark.da_plcbf_actuator_video import load_episode
from benchmark.da_plcbf_recovery_analysis import BASE as SOURCE
from benchmark.da_plcbf_recovery_analysis import sha
from crazyflow.safety.da_plcbf.contact_replay import (
    ContactBody,
    ContactReplayConfig,
    ObstacleMotion,
    cf21b_contact_body,
    find_contact_trigger,
    run_contact_replay,
    save_contact_replay,
)

BASE = SOURCE.parents[1] / "three-panel-wind-20260907/v1"


def main() -> None:
    episode = load_episode(SOURCE / "wind-demo-v1/PD_F/attempt-00", expected_method="PD_F")
    support = np.arange(43002) * 0.001
    obstacles = ObstacleMotion(
        support,
        np.asarray([episode.obstacle_centers(t) for t in support]),
        np.asarray(episode.world["obstacle_radii"]),
    )
    trigger = find_contact_trigger(
        episode.dense["time"],
        episode.dense["state"][:, :13],
        obstacles,
        ContactReplayConfig(),
        kind="physical_contact",
    )
    assert abs(trigger.time_seconds - episode.contact_time) < 0.002
    config = ContactReplayConfig(duration_seconds=43 - trigger.time_seconds)
    default = cf21b_contact_body()
    body = ContactBody(
        float(episode.controls["actual_mass"][-1]),
        episode.controls["actual_inertia"][-1],
        default.gravity,
        default.drag_matrix_body,
    )
    times = trigger.time_seconds + np.arange(math.ceil(config.duration_seconds / 0.001) + 1) * 0.001
    replay = run_contact_replay(
        trigger,
        body,
        obstacles,
        config,
        wind_velocity_world=np.asarray([episode.wind_at(t) for t in times]),
    )
    assert replay.metadata["obstacle_contact_steps"] > 0
    assert replay.metadata["ground_contact_steps"] > 0
    replay.metadata["source_episode"] = str(episode.directory)
    replay.metadata["source_files_sha256"] = episode.source_sha256
    replay.metadata["source_contact_time"] = episode.contact_time
    replay.metadata["presentation_policy"] = (
        "At physical contact, switch to zero rotor thrust and MuJoCo rigid-body contact dynamics. "
        "This is a separately simulated crash presentation, not the controller's recorded suffix."
    )
    replay.metadata["driver_sha256"] = sha(Path(__file__))
    output = save_contact_replay(replay, BASE / "contact-v1")
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(replay.metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
