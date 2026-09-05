"""Plot recorded outcomes without rerunning or changing any controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from crazyflow.safety.da_plcbf.online_constant_wind import OnlineConstantWindResult


def plot_mechanism_comparison(result: OnlineConstantWindResult, output: Path) -> Path:
    """Save a compact route, clearance, certificate, and learning comparison."""
    colors = {
        "fixed": "#d95f40",
        "adaptive": "#167c62",
        "analytic": "#777d8a",
        "compensated": "#447bc2",
    }
    labels = {
        "fixed": "Fixed PL-CBF",
        "adaptive": "Online BPTT + PL-CBF",
        "analytic": "Analytic HOCBF",
        "compensated": "Fixed + wind feedforward",
    }
    trace = result.trace
    time = trace.time_seconds
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 8.2), constrained_layout=True)
    figure.suptitle(
        "One permanent wind change · matched controllers · independent point estimates", fontsize=15
    )
    for name, method in result.methods.items():
        color, label = colors[name], labels[name]
        physical_clearance = np.min(
            np.stack(
                [
                    np.linalg.norm(method.position - obstacle.centers, axis=1)
                    - obstacle.physical_radius
                    - trace.drone_radius
                    for obstacle in trace.obstacles
                ]
            ),
            axis=0,
        )
        collisions = np.flatnonzero(physical_clearance <= 0)
        end = int(collisions[0]) + 1 if collisions.size else len(time)
        # Post-impact free-flight replay is not a meaningful continuation of physical navigation.
        # Retain the complete trace on disk and explicitly stop failed curves at first contact.
        if collisions.size:
            label += " (collision ×)"
            axes[0, 0].scatter(*method.position[end - 1, :2], marker="x", s=65, color=color)
        axes[0, 0].plot(
            method.position[:end, 0], method.position[:end, 1], color=color, label=label, lw=2
        )
        clearance = np.min(
            np.stack(
                [
                    np.linalg.norm(method.position - obstacle.centers, axis=1)
                    - obstacle.inflated_radius
                    for obstacle in trace.obstacles
                ]
            ),
            axis=0,
        )
        axes[0, 1].plot(time[:end], clearance[:end], color=color, label=label, lw=1.8)
        if method.maximum_library_value is not None and name != "analytic":
            axes[1, 0].plot(time, method.maximum_library_value, color=color, label=label, lw=1.8)
        if name != "analytic" and not any(
            "adaptive_state_coverage" in probe for probe in result.summary.get("shared_probes", [])
        ):
            axes[1, 1].step(
                time,
                method.fallback_safe.sum(axis=1),
                color=color,
                label=label,
                lw=1.8,
                where="post",
            )
    for obstacle in trace.obstacles:
        center = obstacle.centers[0]
        axes[0, 0].add_patch(
            plt.Circle(center[:2], obstacle.inflated_radius, color="#d8d9dd", alpha=0.6)
        )
        axes[0, 0].add_patch(
            plt.Circle(center[:2], obstacle.physical_radius, fill=False, color="#717680")
        )
    axes[0, 0].scatter(*trace.goal_position[:2], marker="*", s=140, color="#d7a927", zorder=5)
    axes[0, 0].set(
        title="Paths in top view; failed curves end at first collision",
        xlabel="x [m]",
        ylabel="y [m]",
    )
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend(fontsize=8, loc="best")
    axes[0, 1].set(
        title="3D clearance beyond obstacle + drone + margin",
        xlabel="Simulation time [s]",
        ylabel="Shell clearance [m]",
    )
    axes[1, 0].set(
        title="Collision library value: max over nominal + fallbacks",
        xlabel="Simulation time [s]",
        ylabel="Hard swept H [m²]",
        ylim=(-0.6, 1.5),
    )
    probes = [
        probe
        for probe in result.summary.get("shared_probes", [])
        if "adaptive_state_coverage" in probe
    ]
    if probes:
        probe_time = [probe["time_seconds"] for probe in probes]
        for name in ("fixed", "compensated", "adaptive"):
            probe_values = [
                probe["adaptive_state_coverage"][name]["maximum_library_value"] for probe in probes
            ]
            axes[1, 1].plot(probe_time, probe_values, color=colors[name], label=labels[name], lw=2)
        axes[1, 1].set(
            title="Same adaptive state / model: counterfactual library H",
            xlabel="Simulation time [s]",
            ylabel="Hard swept H [m²]",
            ylim=(-0.5, 0.8),
            xlim=(trace.wind_change_time, min(trace.wind_change_time + 5, time[-1])),
        )
    else:
        axes[1, 1].set(
            title="Collision-safe fallback count at each method's state",
            xlabel="Simulation time [s]",
            ylabel="Number of skills",
            ylim=(-0.4, trace.fixed.fallback_safe.shape[1] + 0.4),
        )
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.axvline(trace.wind_change_time, color="#696969", ls="--", lw=1)
        axis.axhline(0, color="#777777", lw=0.8)
    for axis in axes.flat:
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output
