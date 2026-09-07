"""Audit actual delayed application times, held commands, and available learner versions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmark.da_plcbf_hover_return import BASE, METHODS
from benchmark.da_plcbf_recovery_diagnosis import sha, write


def main() -> None:
    root = BASE / "selected-delayed-v1"
    records = []
    for method in METHODS:
        episode = root / method / "attempt-00"
        summary = json.loads((episode / "summary.json").read_text())
        binding = json.loads((episode / "binding.json").read_text())
        assert binding["config"]["execution_mode"] == "delayed"
        assert binding["config"]["retain_rollouts"] is False
        with np.load(episode / "controls.npz") as a:
            c = {k: a[k] for k in a.files}
        with np.load(episode / "dense.npz") as a:
            t, commands = a["time"], a["command"]
        with np.load(episode / "applications.npz") as a:
            application = {k: a[k] for k in a.files}
        selected = c["command_applied"].astype(bool)
        np.testing.assert_allclose(
            c["command_applied_at"][selected] - c["time"][selected],
            c["controller_seconds"][selected],
            rtol=0,
            atol=1e-8,
        )
        checked_nodes = 0
        for i in range(len(c["time"])):
            start = c["time"][i]
            stop = c["command_applied_at"][i] if selected[i] else summary["physical_time_seconds"]
            mask = (t > start + 1e-9) & (t < stop - 1e-9)
            np.testing.assert_array_equal(
                commands[mask], np.broadcast_to(c["previous_command"][i], commands[mask].shape)
            )
            checked_nodes += int(mask.sum())
        for j, i in enumerate(application["control_index"]):
            if i < 0:
                continue
            assert selected[i]
            np.testing.assert_array_equal(application["command"][j], c["planned_command"][i])
            assert abs(application["time"][j] - c["command_applied_at"][i]) < 1e-9
        # No update fits this measured run's available budget: verify this at every control.
        assert summary["finite_credited_updates"] == 0
        assert np.all(c["control_library_version"] == summary["initial_library_version"])
        records.append(
            dict(
                method=method,
                termination=summary["termination"],
                collision_time_seconds=summary["physical_time_seconds"],
                actual_old_command_service_nodes_verified=checked_nodes,
                exact_application_commands_verified=True,
                measured_application_delay_verified=True,
                actual_updates_used=0,
                control_count=summary["control_count"],
                controller_seconds=summary["controller_seconds"],
                controller_deadline_misses=summary["controller_deadline_misses"],
                skipped_sensing_ticks=summary["skipped_sensing_ticks"],
                uncovered_hold_intervals=summary["online_held_check_uncovered_intervals"],
                all_hold_intervals_covered=summary["online_held_check_timing_all_covered"],
                maximum_hold_seconds=summary["maximum_actual_command_hold_seconds"],
                actual_operational_all_nodes_pass=summary["actual_operational_all_nodes_pass"],
                sources={
                    str(p): sha(p)
                    for p in (
                        episode / "controls.npz",
                        episode / "dense.npz",
                        episode / "applications.npz",
                        episode / "summary.json",
                    )
                },
            )
        )
    write(
        root / "timing-audit.json",
        dict(
            records=records,
            environment=json.loads((root / "timing-environment.json").read_text()),
            scope="Measured delayed execution under observed concurrent GPU load. "
            "Old commands actually act during controller service; no unavailable updates are "
            "credited. This failed reproduction does not establish exclusive-device limits, "
            "warm worst-case execution time, asynchronous safety or deployment feasibility.",
        ),
    )
    (root / "timing-audit-driver.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(records, indent=2), flush=True)


if __name__ == "__main__":
    main()
