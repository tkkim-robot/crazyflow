"""Generate source-backed static figures for the controlled and continuous case study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from benchmark.da_plcbf_case_discovery import write_json
from crazyflow.safety.da_plcbf.case_study_world import (
    HoverEncounterConfig,
    build_hover_encounter_world,
)


def build_figures(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    sources = []

    def read(path: Path) -> dict | list:
        sources.append(path)
        return json.loads(path.read_text())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"frozen": "#b86635", "adapted_held": "#087f8c"}
    labels = {"frozen": "Frozen", "adapted_held": "Adapted snapshot"}
    cases = (
        ("uncompensated", "uncompensated-000-t0100-132"),
        ("compensated", "compensated-002-t0100-487"),
    )
    summary = {}
    for column, (case, selected) in enumerate(cases):
        selected_directory = root / "closed-loop-v3" / selected
        cfg = HoverEncounterConfig.from_dict(read(selected_directory / "encounter.json"))
        # Query-only world; initial clearance validation uses the actual t=0 default state.
        world = build_hover_encounter_world(cfg)
        confirmation = root / f"confirmation-{case}-v1"
        dense_path = confirmation / "causal_dense_states.npz"
        sources.append(dense_path)
        with np.load(dense_path) as data:
            for method in colors:
                states, times = data[f"{method}_states"], data[f"{method}_times"]
                centers, _ = world.obstacle_kinematics(times)
                clearance = np.min(
                    np.linalg.norm(states[:, None, :3] - centers, axis=-1)
                    - world.obstacle_radii[None]
                    - 0.256,
                    axis=1,
                )
                axes[0, column].plot(
                    times, clearance * 100, color=colors[method], label=labels[method]
                )
        axes[0, column].axhline(0, color="black", linewidth=0.8)
        axes[0, column].axvline(
            cfg.incoming.arrival_time_seconds, color="black", alpha=0.3, linestyle=":"
        )
        axes[0, column].set(
            title=f"Encounter detail: matched {case} pair",
            xlabel="Absolute simulation time (s)",
            ylabel="Distance outside safety shell (cm)",
            xlim=(4.6, 5.6),
            ylim=(-12, 12),
        )
        axes[0, column].legend(loc="lower right")
        neighbors = read(confirmation / "neighbor_ledger.json")
        margins = {
            method: [
                row["methods"][method]["geometry_audit"]["safety_shell"]["minimum_clearance_m"]
                * 100
                for row in neighbors
            ]
            for method in colors
        }
        for index in range(len(neighbors)):
            axes[1, column].plot(
                [0, 1],
                [margins[method][index] for method in colors],
                color="#bbc2c5",
                alpha=0.65,
                linewidth=1,
            )
        for index, method in enumerate(colors):
            axes[1, column].scatter(
                np.full(len(neighbors), index),
                margins[method],
                color=colors[method],
                s=28,
                zorder=3,
            )
        axes[1, column].axhline(0, color="black", linewidth=0.8)
        axes[1, column].set(
            xticks=[0, 1],
            xticklabels=list(labels.values()),
            xlim=(-0.25, 1.25),
            ylabel="Minimum shell clearance (cm)",
            title="12 frozen-protocol neighborhood perturbations",
        )
        summary[case] = {
            method: {
                "positive_shell": sum(x > 0 for x in values),
                "minimum_cm": min(values),
                "maximum_cm": max(values),
            }
            for method, values in margins.items()
        }
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Recovered fallback certificates preserve the requested margin in controlled branches\n"
        "All physical colliders remain separated; selected cases, not broad superiority",
        fontsize=12,
    )
    fig.savefig(output / "controlled_case_results.png", dpi=170)
    fig.savefig(output / "controlled_case_results.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    files = [
        root / "runtime-profile-v1/anchors-2/runtime.json",
        root / "runtime-unroll-v1/profile/anchors-2/runtime.json",
    ]
    runtime = [read(path)["update_service"] for path in files]
    means = [r["median_seconds"] * 1000 for r in runtime]
    axes[0].bar([0, 1], means, color=["#8a9398", "#087f8c"], width=0.5)
    for i, mean in enumerate(means):
        axes[0].text(i, mean + 0.4, f"{mean:.2f} ms", ha="center")
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Original nested loop", "Bounded inner unroll"],
        ylabel="Full learner update median (ms)",
        ylim=(0, 24),
        title="Same objective, anchors, integration and command hold",
    )
    raw_path = root / "continuous-paced-v1/navigation_comparison.npz"
    sources.append(raw_path)
    with np.load(raw_path) as data:
        times = data["time_seconds"]
        versions = data["adaptive_library_version"]
        axes[1].step(times, versions - versions[0], where="post", color="#087f8c")
    axes[1].axvline(3, color="black", alpha=0.3, linestyle=":")
    axes[1].axvline(4.903185042785481, color="black", alpha=0.3, linestyle=":")
    axes[1].set(
        xlabel="Paced simulation time (s)",
        ylabel="Completed updates used by control",
        title="Actual paced publication; zero controller deadline misses",
    )
    fig.suptitle(
        "Real online updates now fit the unchanged 40 ms schedule\n"
        "In the continuous scene both methods finish safely; no safety advantage is claimed",
        fontsize=12,
    )
    fig.savefig(output / "paced_compute_results.png", dpi=170)
    fig.savefig(output / "paced_compute_results.pdf")
    plt.close(fig)
    write_json(
        output / "source_manifest.json",
        {
            "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            "neighborhood_summary": summary,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_figures(args.root, args.output)


if __name__ == "__main__":
    main()
