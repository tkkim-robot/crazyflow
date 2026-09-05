"""Audit seeded scene geometry against a prescribed constant-speed route, without a plant.

This is a scene-severity diagnostic, not closed-loop progress, feasibility, or an adaptive result.
The actual experiment must also record nominal controller predictions and executed-QP influence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from crazyflow.safety.da_plcbf.continuous_version_a import RuntimeObstacleTrajectories
from crazyflow.safety.da_plcbf.navigation_world import (
    NavigationWorldConfig,
    build_navigation_world,
    nominal_encounter_metrics,
)


def geometry_audit(seed: int, obstacle_count: int, speed: float = 1.2) -> dict[str, object]:
    world = build_navigation_world(NavigationWorldConfig(seed=seed, obstacle_count=obstacle_count))
    vertices = np.vstack((world.initial_state[:3], world.waypoint_positions))
    durations = np.linalg.norm(np.diff(vertices, axis=0), axis=1) / speed
    arrivals = np.concatenate(([0.0], np.cumsum(durations)))
    times = np.arange(0.0, world.config.duration_seconds, 0.2)
    records = []
    encountered = set()
    for time in times:
        future = time + np.arange(61) * world.config.dt
        route = np.column_stack(
            [np.interp(future, arrivals, vertices[:, axis]) for axis in range(3)]
        )
        centers, velocities = world.obstacle_kinematics(future)
        prediction = RuntimeObstacleTrajectories(
            centers, world.obstacle_radii, np.ones(centers.shape[:-1], dtype=bool), velocities
        )
        records.append(
            nominal_encounter_metrics(
                route,
                prediction,
                dt=world.config.dt,
                ego_radius=world.config.ego_radius,
                obstacle_clearance=world.config.obstacle_clearance,
            )
        )
        node_clearance = (
            np.linalg.norm(route[:, None] - centers, axis=-1)
            - world.obstacle_radii
            - world.config.ego_radius
            - world.config.obstacle_clearance
        )
        encountered.update(np.flatnonzero(np.any(node_clearance <= 0, axis=0)).tolist())
    blocked = np.asarray([record["nominal_blocked"] for record in records])
    episodes = np.count_nonzero(blocked & ~np.concatenate(([False], blocked[:-1])))
    return {
        "seed": seed,
        "obstacle_count": obstacle_count,
        "reference_speed_m_per_s": speed,
        "blocked_prediction_fraction": float(np.mean(blocked)),
        "separated_blocked_prediction_intervals": int(episodes),
        "distinct_node_threatening_obstacles": sorted(encountered),
        "maximum_simultaneous_predicted_threats": max(
            row["peak_simultaneous_threats"] for row in records
        ),
        "reference_waypoints_reached_before_timeout": int(
            np.count_nonzero(arrivals[1:] < world.config.duration_seconds)
        ),
        "reference_altitude_range_m": float(np.ptp(vertices[:, 2])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = [geometry_audit(seed, count) for count in (8, 16, 32) for seed in range(args.seeds)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": (
                    "Prescribed constant-speed polyline only; no plant/controller/learner; "
                    "not a feasibility witness"
                ),
                "prediction_seconds": 1.2,
                "prediction_integration_dt": 0.02,
                "query_period_seconds": 0.2,
                "records": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for count in (8, 16, 32):
        records = [row for row in results if row["obstacle_count"] == count]
        print(
            json.dumps(
                {
                    "obstacles": count,
                    "blocked_fraction_median": float(
                        np.median([r["blocked_prediction_fraction"] for r in records])
                    ),
                    "blocked_intervals_range": [
                        min(r["separated_blocked_prediction_intervals"] for r in records),
                        max(r["separated_blocked_prediction_intervals"] for r in records),
                    ],
                    "distinct_threatening_obstacles_range": [
                        min(len(r["distinct_node_threatening_obstacles"]) for r in records),
                        max(len(r["distinct_node_threatening_obstacles"]) for r in records),
                    ],
                }
            )
        )


if __name__ == "__main__":
    main()
