"""Paired descriptive statistics over completed predeclared navigation world draws."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(center - radius), float(center + radius)]


def _paired(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values)
    if not len(x):
        return {"count": 0, "mean": None, "median": None, "bootstrap_mean_95_interval": None}
    rng = np.random.default_rng(57291)
    means = np.mean(x[rng.integers(0, len(x), size=(10000, len(x)))], axis=1)
    return {
        "count": len(x),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "bootstrap_mean_95_interval": np.percentile(means, [2.5, 97.5]).tolist(),
        "individual_differences": x.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", default="CAMPAIGN_PROTOCOL.json")
    parser.add_argument("--pattern", default="heldout-*/*/navigation_comparison.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a campaign summary")
    protocol_path = args.root / args.protocol
    protocol = json.loads(protocol_path.read_text())
    expected = {
        (seed, condition)
        for seed in protocol["heldout_world_seeds"]
        for condition in protocol["conditions"]
    }
    seen, runs = set(), []
    for path in sorted(args.root.glob(args.pattern)):
        summary = json.loads(path.read_text())["summary"]
        seed = summary["world"]["config"]["seed"]
        condition = path.parent.name.split("-seed")[0]
        key = seed, condition
        if key in seen or key not in expected:
            raise ValueError(f"unexpected or duplicated world draw: {key}")
        seen.add(key)
        if summary["world"]["config"]["obstacle_count"] != protocol["obstacles"]:
            raise ValueError("obstacle count differs from the declared condition")
        if summary["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
            raise ValueError("checkpoint changed across held-out runs")
        source = json.loads((path.parent / "SOURCE_SHA256.json").read_text())
        for name, digest in source.items():
            if protocol["files"][name] != digest:
                raise ValueError("runner source differs from frozen campaign")
        runs.append(
            {
                "seed": seed,
                "condition": condition,
                "path": str(path.parent),
                "methods": summary["methods"],
            }
        )
    if seen != expected:
        raise ValueError(f"campaign incomplete: missing {sorted(expected - seen)}")
    output = {
        "protocol": str(protocol_path),
        "paired_world_count": len(runs),
        "scope": (
            "Fixed teacher seed7; seeded route jitter and absolute-time moving-obstacle "
            "phases. Descriptive Wilson success intervals and paired bootstrap mean "
            "differences; no general safety guarantee."
        ),
        "conditions": {},
    }
    for condition in protocol["conditions"]:
        cells = [run for run in runs if run["condition"] == condition]
        statistics = {
            "count": len(cells),
            "world_seeds": [cell["seed"] for cell in cells],
            "methods": {},
        }
        for method in ("fixed", "adaptive"):
            measurements = [cell["methods"][method] for cell in cells]
            count = len(measurements)
            criteria = {
                "physical_collision_free": [not row["physical_collision"] for row in measurements],
                "positive_shell_clearance": [
                    row["minimum_inflated_clearance_m"] > 0 for row in measurements
                ],
                "all_waypoints_completed": [
                    row["termination"] == "completed" for row in measurements
                ],
                "zero_degraded_controls": [row["degraded_controls"] == 0 for row in measurements],
            }
            statistics["methods"][method] = {
                "criteria": {
                    key: {
                        "count": sum(mask),
                        "fraction": sum(mask) / count,
                        "wilson_95_interval": _wilson(sum(mask), count),
                    }
                    for key, mask in criteria.items()
                },
                "waypoint_counts": [row["waypoints_completed"] for row in measurements],
                "minimum_shell_clearance_m": min(
                    row["minimum_inflated_clearance_m"] for row in measurements
                ),
                "median_completion_or_timeout_seconds": float(
                    np.median(
                        [
                            row["termination_time_seconds"]
                            if row["termination"] == "completed"
                            else protocol["duration_seconds"]
                            for row in measurements
                        ]
                    )
                ),
                "nominal_blocked_fractions": [
                    row["nominal_blocked_fraction"] for row in measurements
                ],
                "nominal_encounter_episode_counts": [
                    row["separate_nominal_encounter_episodes"] for row in measurements
                ],
                "finite_update_counts": [row["finite_updates"] for row in measurements],
                "degraded_control_counts": [row["degraded_controls"] for row in measurements],
                "positive_executed_dual_counts": [
                    row["executed_positive_policy_dual_controls"] for row in measurements
                ],
            }
        statistics["paired_adaptive_minus_fixed_waypoints"] = _paired(
            [
                cell["methods"]["adaptive"]["waypoints_completed"]
                - cell["methods"]["fixed"]["waypoints_completed"]
                for cell in cells
            ]
        )
        statistics["paired_adaptive_minus_fixed_completion_seconds_both_complete"] = _paired(
            [
                cell["methods"]["adaptive"]["termination_time_seconds"]
                - cell["methods"]["fixed"]["termination_time_seconds"]
                for cell in cells
                if all(
                    cell["methods"][method]["termination"] == "completed"
                    for method in ("fixed", "adaptive")
                )
            ]
        )
        statistics["paired_adaptive_minus_fixed_shell_margin_m"] = _paired(
            [
                cell["methods"]["adaptive"]["minimum_inflated_clearance_m"]
                - cell["methods"]["fixed"]["minimum_inflated_clearance_m"]
                for cell in cells
            ]
        )
        output["conditions"][condition] = statistics
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
