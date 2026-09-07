"""Real-clock regression for sensing ticks lost during terminal asynchronous catch-up."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_da_plcbf_actuator_experiment import harness as harness

from crazyflow.safety.da_plcbf import actuator_experiment as runtime
from crazyflow.safety.da_plcbf.actuator_independent import NumpyEffortPlant
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig


def test_terminal_callback_stall_counts_remaining_sensing_ticks(
    harness: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr(runtime, "time", time)
    # Preserve the P1 clock and audits while making physical evolution stationary.
    monkeypatch.setattr(NumpyEffortPlant, "_step", lambda plant, command, dt: None)
    callbacks: list[dict[str, Any]] = []

    def controller(observed: Any, params: Any, *args: Any) -> Any:
        return SimpleNamespace(
            action=harness.initial[13:].copy(),
            selected_index=0,
            certificates=SimpleNamespace(
                rollouts=SimpleNamespace(states=np.broadcast_to(np.asarray(observed), (2, 5, 17)))
            ),
        )

    def stall_after_first_control(payload: dict[str, Any]) -> None:
        callbacks.append(payload)
        time.sleep(0.18)

    config = runtime.ActuatorEpisodeConfig(
        method="F2",
        execution_mode="asynchronous",
        plant_level="P1",
        filter_config=ActuatorFilterConfig(horizon=4),
        warmup_calls=2,
    )
    result = runtime.run_actuator_episode(
        harness.scene,
        harness.bundle,
        config,
        harness.tmp_path / "terminal-stall",
        controllerfunctions=SimpleNamespace(controller=controller),
        progress_callback=stall_after_first_control,
    )

    assert result.summary["error"] is None, result.summary["error"]
    assert result.summary["termination"] == "duration_complete"
    assert result.summary["full_episode_completed"]
    assert result.summary["physical_time_seconds"] == pytest.approx(0.16)
    assert result.summary["control_count"] == 1
    assert len(callbacks) == 1
    assert callbacks[0]["control_index"] == 0
    # The duration endpoint at .16 is excluded: only .04, .08, and .12 were missed.
    assert result.summary["skipped_sensing_ticks"] == 3
    np.testing.assert_array_equal(result.control_traces["skipped_sensing_ticks"], [3])
    np.testing.assert_allclose(result.control_traces["resolved_physical_time"], [0.16])
    events = [json.loads(line) for line in result.artifacts["events"].read_text().splitlines()]
    terminal = [event for event in events if event["event"] == "asynchronous_terminal_hold"]
    assert len(terminal) == 1
    assert terminal[0]["time"] == pytest.approx(0.16)
    # Service can itself cross a tick on a loaded CPU; the terminal event must
    # account for every remaining tick after the already-recorded control service.
    resolved = [event for event in events if event["event"] == "control_resolved"]
    assert len(resolved) == 1
    assert terminal[0]["skipped_sensing_ticks"] + resolved[0]["skipped_sensing_ticks"] == 3
    assert result.summary["learner_calls"] == 0
    assert result.summary["snapshot_publications"] == []
