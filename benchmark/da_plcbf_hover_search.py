"""Bounded, fully retained search for informative hover/return comparisons."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmark.da_plcbf_hover_return import BASE, HoverSceneSpec, resources, run_case
from benchmark.da_plcbf_recovery_diagnosis import begin, write


def search_specs(seed: int, count: int) -> list[HoverSceneSpec]:
    """Sample a fixed development grid before any flight outcomes are observed."""
    rng = np.random.default_rng(seed)
    specs = []
    for index in range(count):
        wind = (1.6, 1.8, 2.0)[index % 3]
        first = (
            (3.2, 4.0)[(index // 2) % 2] if index < 4 else float(rng.choice((3.0, 3.4, 4.0, 5.2)))
        )
        speed = 1.2 if index < 2 else 1.4 if index < 4 else float(rng.choice((0.8, 1.0, 1.2, 1.4)))
        radius = 0.65 if index < 4 else float(rng.choice((0.45, 0.6, 0.75)))
        stagger = 0.8 if index < 2 else 0.4 if index < 4 else float(rng.choice((0.4, 0.8, 1.2)))
        rotation = 0.2 if index < 2 else -0.4 if index < 4 else float(rng.uniform(-np.pi, np.pi))
        offset = 0.0 if index < 2 else 0.2 if index < 4 else float(rng.choice((-0.3, 0.0, 0.3)))
        angle = 0.463647609 if index < 4 else float(rng.uniform(-np.pi, np.pi))
        specs.append(
            HoverSceneSpec(
                seed=seed + index,
                wind_speed=wind,
                wind_angle=angle,
                reversal_angle=angle + 2.154346269,
                first_encounter=first,
                stagger=stagger,
                half_period=16.0,
                amplitude=speed * 16.0 / np.pi,
                radius=radius,
                path_rotation=rotation,
                lateral_offset=offset,
            )
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "search-v1")
    parser.add_argument("--seed", type=int, default=71200)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.count <= 32:
        raise ValueError("one development batch contains 1–32 complete triplets")
    begin(args.output, "Fixed bounded development search; every full triplet retained.")
    (args.output / "driver.py").write_bytes(Path(__file__).read_bytes())
    specs = search_specs(args.seed, args.count)
    write(
        args.output / "plan.json",
        {
            "seed": args.seed,
            "specs": [asdict(s) for s in specs],
            "methods": ["PD_F", "F2", "A_BAL"],
            "wind_range_basis": "All methods passed obstacle-free 1.4–2.2 m/s calibration.",
            "motivation": "Vary first encounter delay from wind onset to overlap the measured "
            "roughly one-to-four-second repertoire recovery response.",
            "primary_candidate_rule": "Adaptive full collision-free operationally valid return; "
            "both frozen baselines have observed modeled-collider collisions.",
            "selection_policy": (
                "Finish the complete sealed batch; retain all negative and tied cases."
            ),
        },
    )
    cache, records, candidates = resources(), [], []
    for index, spec in enumerate(specs):
        rows = run_case(args.output / f"case-{index:03d}", spec, cache)
        by_method = {r["method"]: r["summary"] for r in rows}
        candidate = by_method["A_BAL"]["successful_full_episode"] and all(
            by_method[m]["modeled_collider_collision"] is True for m in ("PD_F", "F2")
        )
        record = {
            "case": index,
            "spec": asdict(spec),
            "adaptive_only_survival_return": candidate,
            "outcomes": {
                m: {
                    k: s[k]
                    for k in (
                        "termination",
                        "physical_time_seconds",
                        "modeled_collider_collision",
                        "successful_full_episode",
                        "actual_operational_all_nodes_pass",
                        "hover_return",
                    )
                }
                for m, s in by_method.items()
            },
        }
        records.append(record)
        if candidate:
            candidates.append(index)
        write(
            args.output / "report.json",
            {
                "records": records,
                "candidates": candidates,
                "planned": args.count,
                "complete": index + 1 == args.count,
            },
        )
        print(
            json.dumps(
                {"case_complete": index, "candidate": candidate, "candidate_indices": candidates}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
