"""Audit the three new wind flights and quantify their displayed fallback differences."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmark.da_plcbf_actuator_video import load_episode, validate_pair
from benchmark.da_plcbf_recovery_analysis import BASE as PREVIOUS
from benchmark.da_plcbf_recovery_analysis import sha
from benchmark.da_plcbf_wind_learning_comparison import BASE


def main() -> None:
    output = BASE / "analysis-v1"
    output.mkdir(exist_ok=False)
    episodes = [
        load_episode(BASE / "flights" / m / "attempt-00", expected_method=m)
        for m in ("PD_F", "F2", "A_BAL")
    ]
    rows = []
    for episode in episodes:
        s, c = episode.summary, episode.controls
        method = s["method"]
        previous = json.loads(
            (PREVIOUS / "wind-demo-v1" / method / "attempt-00/binding.json").read_text()
        )
        assert episode.binding["scene"] == previous["scene"]
        config = episode.binding["checkpoint"]["runtime_actor_config"]
        assert config["wind_feedforward"] is False and config["model_compensation"] is True
        times = c["time"]
        expected = np.asarray([episode.wind_at(t) for t in times], dtype=np.float32)
        np.testing.assert_array_equal(c["actual_wind_velocity"], expected)
        np.testing.assert_array_equal(c["estimated_wind_velocity"], expected)
        np.testing.assert_array_equal(
            episode.dense["actual_wind_velocity"],
            [episode.wind_at(t) for t in episode.dense["time"]],
        )
        versions = c["control_library_version"] - s["initial_library_version"]
        if method != "A_BAL":
            np.testing.assert_array_equal(versions, np.zeros_like(versions))
        preflight = times < 19
        has_plan = c["backup_has_checked_plan"]
        rows.append(
            {
                "method": method,
                "termination": s["termination"],
                "end_time_seconds": s["physical_time_seconds"],
                "waypoints_completed": s["waypoints_completed"],
                "waypoint_arrival_times_seconds": s["waypoint_arrival_times_seconds"],
                "successful_full_episode": s["successful_full_episode"],
                "actual_operational_all_nodes_pass": s["actual_operational_all_nodes_pass"],
                "finite_credited_updates": s["finite_credited_updates"],
                "preflight_online_versions_used": int(np.max(versions[preflight])),
                "final_online_versions_used": int(versions[-1]),
                "unchecked_controls": int(np.count_nonzero(~has_plan)),
                "executed_backup_controls": int(np.count_nonzero(c["backup_executed_backup"])),
                "held_timing_covered": s["online_held_check_timing_all_covered"],
                "same_scene_as_previous_video": True,
                "actual_and_estimated_wind_match_recorded_schedule": True,
                "wind_feedforward": False,
                "source_sha256": episode.source_sha256,
            }
        )
    for episode in episodes[1:]:
        validate_pair(
            episodes[0],
            episode,
            allow_different_learning_contract=True,
            allow_different_libraries=True,
        )
    frozen, adaptive = [e.controls for e in episodes[1:]]
    fan_rows = []
    for when in (2.96, 3.0, 3.04, 4.0, 7.0, 10.96, 11.0, 18.96, 21.0, 25.0):
        indices = [int(np.argmin(abs(c["time"] - when))) for c in (frozen, adaptive)]
        for c, index in zip((frozen, adaptive), indices, strict=True):
            assert abs(c["time"][index] - when) < 1e-9
        f, a = [
            c["candidate_states"][i, 1:, :, :3]
            for c, i in zip((frozen, adaptive), indices, strict=True)
        ]
        delta = (a - a[:, :1]) - (f - f[:, :1])
        fan_rows.append(
            {
                "time_seconds": when,
                "relative_fan_rms_difference_m": float(np.sqrt(np.mean(np.sum(delta**2, axis=-1)))),
                "relative_endpoint_rms_difference_m": float(
                    np.sqrt(np.mean(np.sum(delta[:, -1] ** 2, axis=-1)))
                ),
                "adaptive_online_versions_used": int(
                    adaptive["control_library_version"][indices[1]]
                )
                - episodes[2].summary["initial_library_version"],
            }
        )
    report = {
        "flights": rows,
        "actual_video_relative_fan_comparison": fan_rows,
        "qualification": (
            "Fixed development demonstration, not fresh validation. Fan differences compare "
            "recorded predictions at each flight's own state; "
            "they do not isolate parameter effects. "
            "Same initial learned parameters and optimizer; only A_BAL updates online."
        ),
        "driver_sha256": sha(Path(__file__)),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
