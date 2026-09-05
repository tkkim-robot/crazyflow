from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from crazyflow.safety.da_plcbf.navigation_experiment import _append_terminal_record
from crazyflow.safety.da_plcbf.navigation_world import (
    PayloadEvent,
    WindEvent,
    build_navigation_world,
)
from examples.da_plcbf.navigation_demo import load_world_config

if TYPE_CHECKING:
    from pathlib import Path


def test_terminal_padding_keeps_final_physical_state_without_fictitious_service() -> None:
    initial = np.asarray((-1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0), dtype=float)
    terminal = initial.copy()
    terminal[0] = 1
    records = {
        "position": [initial[:3].copy()],
        "quaternion_xyzw": [initial[3:7].copy()],
        "full_state": [initial.copy()],
        "controller_seconds": [0.019],
        "learner_seconds": [0.011],
        "missed_deadline": [True],
        "fallback_rollouts": [np.zeros((2, 3, 3))],
    }
    _append_terminal_record(records, terminal)
    np.testing.assert_array_equal(records["full_state"][-1], terminal)
    np.testing.assert_array_equal(records["position"][-1], terminal[:3])
    np.testing.assert_array_equal(records["full_state"][0], initial)
    assert records["controller_seconds"][-1] == records["learner_seconds"][-1] == 0.0
    assert records["missed_deadline"][-1] is False
    _append_terminal_record(records, terminal)
    assert all(len(values) == 3 for values in records.values())
    np.testing.assert_array_equal(records["position"][-1], records["position"][-2])


def test_cli_config_round_trip_preserves_composed_events_and_shared_world(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "seed": 5,
                "wind_events": [{"time_seconds": 4, "velocity": [1, 2, 0]}],
                "payload_events": [{"time_seconds": 8, "mass_fraction": 0.25}],
            }
        )
    )
    config = load_world_config(path)
    assert config.wind_events == (WindEvent(4, (1, 2, 0)),)
    assert config.payload_events == (PayloadEvent(8, 0.25),)
    world = build_navigation_world(config)
    saved = tmp_path / "world.json"
    saved.write_text(json.dumps(world.metadata()))
    replay = build_navigation_world(load_world_config(saved))
    np.testing.assert_array_equal(replay.waypoint_positions, world.waypoint_positions)
    np.testing.assert_array_equal(
        replay.obstacle_kinematics(7.12)[0], world.obstacle_kinematics(7.12)[0]
    )
