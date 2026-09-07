"""Scientific plots of the selected source flights and same-state behavior probes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.da_plcbf_hover_return import BASE
from benchmark.da_plcbf_recovery_diagnosis import sha, write


def main() -> None:
    output = BASE / "figures-v1"
    output.mkdir(exist_ok=False)
    mechanism = BASE / "selected-mechanism-v1/report.json"
    rows = json.loads(mechanism.read_text())["rows"]
    times = np.array([r["time_seconds"] for r in rows])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, layout="constrained")
    colors = {"PD_F": "#a36a00", "F2": "#cc503e", "A_BAL": "#007c91"}
    labels = {"PD_F": "Handcrafted frozen", "F2": "Learned frozen", "A_BAL": "Learned adaptive"}
    sources = {str(mechanism): sha(mechanism)}
    for ax in axes.flat:
        ax.axvspan(2, 9, color="#bceaf0", alpha=0.35, zorder=0)
        ax.axvspan(16, 25.2, color="#bceaf0", alpha=0.35, zorder=0)
        ax.set_xlim(0, 32)
        ax.grid(alpha=0.2)
    for label, color in [("frozen", colors["F2"]), ("adaptive", colors["A_BAL"])]:
        for ax, key in zip(
            axes[0], ["mean_position_rmse_m", "mean_terminal_speed_excess_mps"], strict=True
        ):
            values = [r["libraries"][label][key] for r in rows]
            ax.plot(times, values, "o-", ms=3, label=f"Learned {label}", color=color)
    axes[0, 0].set(
        title="Same complete state and current model: path recovery",
        ylabel="Mean position RMSE vs nominal teacher (m)",
    )
    axes[0, 1].set(
        title="Same complete state and current model: braking recovery",
        ylabel="Mean terminal speed excess vs teacher (m/s)",
    )
    for method in colors:
        path = BASE / "direction-refinement-v1/case-003" / method / "attempt-00/dense.npz"
        summary = json.loads((path.parent / "summary.json").read_text())
        with np.load(path) as a:
            t = a["time"]
            states = a["state"]
        home = np.asarray(summary["hover_return"]["home_position_m"])
        distance = np.linalg.norm(states[:, :3] - home, axis=1)
        speed = np.linalg.norm(states[:, 7:10], axis=1)
        for ax, values in zip(axes[1], [distance, speed], strict=True):
            ax.plot(t, values, label=labels[method], color=colors[method], lw=1.6)
            if summary["modeled_collider_collision"]:
                ax.plot(t[-1], values[-1], "x", color=colors[method], ms=8, mew=2)
        sources[str(path)] = sha(path)
    axes[1, 0].set(
        title="Recorded flight: station-keeping and both returns",
        ylabel="Home distance (m)",
        xlabel="Physical time (s)",
    )
    axes[1, 1].set(
        title="Recorded flight: speed (× marks collision)",
        ylabel="Speed (m/s)",
        xlabel="Physical time (s)",
    )
    for ax, tolerance in zip(axes[1], [0.7, 0.35], strict=True):
        for lo, hi in [(10, 15.6), (26, 32)]:
            ax.hlines(tolerance, lo, hi, linestyles="dashed", color="#747474", lw=1)
    axes[1, 0].hlines(0.15, 31, 32, color="black", lw=2)
    axes[1, 1].hlines(0.15, 31, 32, color="black", lw=2)
    axes[0, 0].legend(loc="upper right")
    axes[1, 0].legend(loc="upper right")
    fig.suptitle(
        "Selected wind-only hover case · blue shading: wind active · "
        "identical six prescribed movers",
        fontsize=14,
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"mechanism-and-return.{suffix}", dpi=160)
    plt.close(fig)
    write(
        output / "manifest.json",
        dict(
            sources=sources,
            driver_sha256=sha(Path(__file__)),
            scope="Top: immutable teacher and both libraries evaluated at the same adaptive "
            "physical state. Bottom: each recorded closed-loop flight; no synthetic continuation.",
        ),
    )


if __name__ == "__main__":
    main()
