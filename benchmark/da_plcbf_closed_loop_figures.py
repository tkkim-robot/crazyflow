"""Plot source-bound full-episode clearances, causal timing, and matched buffer outcomes.

Clearance lines evaluate the oriented XML sphere at recorded plant nodes, joined for display.
Swept first-contact times and minimum-clearance bounds come from the independent saved audits.
No flight is extended after its recorded stop, and penetration depth is not crash severity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from crazyflow.safety.da_plcbf.case_study_world import (
    CF21B_XML_COLLIDER_OFFSET_BODY_M,
    CF21B_XML_COLLIDER_RADIUS_M,
    HoverEncounterConfig,
    build_hover_encounter_world,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

COLORS = {"fixed": "#b86635", "adaptive": "#087f8c", "onset": "#8360a7", "held": "#356fa6"}


def xml_node_clearance(
    scene: HoverEncounterConfig, times: np.ndarray, states: np.ndarray
) -> np.ndarray:
    """Evaluate physical XML-sphere surface distance without an optional safety buffer."""
    from scipy.spatial.transform import Rotation

    world = build_hover_encounter_world(scene)
    robot = states[:, :3] + Rotation.from_quat(states[:, 3:7]).apply(
        CF21B_XML_COLLIDER_OFFSET_BODY_M
    )
    obstacles = world.obstacle_kinematics(times)[0]
    return np.min(
        np.linalg.norm(robot[:, None] - obstacles, axis=-1)
        - world.obstacle_radii
        - CF21B_XML_COLLIDER_RADIUS_M,
        axis=1,
    )


def build_figures(root: Path, output: Path) -> None:
    """Build two compact PNG/PDF figures from immutable executed-source arrays and summaries."""
    output.mkdir(parents=True, exist_ok=False)
    sources, plotted = {}, {}

    def bind(path: Path) -> None:
        sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()

    def read(path: Path) -> dict:
        bind(path)
        return json.loads(path.read_text())

    def arrays(path: Path) -> dict:
        bind(path)
        with np.load(path) as source:
            return dict(source)

    confirmation = root / "confirmation-staggered-0000-v1"
    targeted = root / "targeted-confirmation-v1"
    original = {
        mapping: read(confirmation / mapping / "original/result.json")
        for mapping in ("uncompensated", "compensated")
    }
    traces = {
        mapping: arrays(confirmation / mapping / "original/traces.npz") for mapping in original
    }
    scene = HoverEncounterConfig.from_dict(original["uncompensated"]["scene"])
    if original["uncompensated"]["scene"] != original["compensated"]["scene"]:
        raise ValueError("the plotted mappings must share the exact physical scene")
    onset = read(confirmation / "uncompensated/freeze_at_wind_onset/result.json")
    onset_trace = arrays(confirmation / "uncompensated/freeze_at_wind_onset/traces.npz")
    early = read(targeted / "uncompensated/early_parameter_reversion/result.json")
    held_trace = arrays(
        targeted / "uncompensated/early_parameter_reversion/held_learned_traces.npz"
    )
    reverted_trace = arrays(
        targeted / "uncompensated/early_parameter_reversion/reverted_initial_traces.npz"
    )
    initial_version = original["uncompensated"]["methods"]["adaptive"]["initial_version"]
    onset_version = onset["methods"]["fixed"]["wind_onset_version"]
    held_version = early["provenance"]["held_learned"]["used_snapshot"]["library_version"]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.88, bottom=0.22, hspace=0.50, wspace=0.25)

    def clearance_line(
        ax: Axes, trace: dict, prefix: str, result: dict, label: str, color: str, name: str
    ) -> None:
        times, states = trace[f"{prefix}dense_times"], trace[f"{prefix}dense_states"]
        values = xml_node_clearance(scene, times, states)
        event = result["first_collider_intersection_seconds"]
        if event is not None:
            before = times < event
            ax.plot(
                np.r_[times[before], event],
                np.r_[values[before] * 100, 0],
                label=label,
                color=color,
                linewidth=1.8,
            )
            ax.plot(
                np.r_[event, times[~before]],
                np.r_[0, values[~before] * 100],
                color=color,
                linewidth=1.2,
                linestyle="--",
            )
            ax.scatter([event], [0], marker="x", s=52, color=color, zorder=5)
            ax.annotate(
                f"first contact {event:.3f} s",
                (event, 0),
                xytext=(event + 0.12, -6),
                fontsize=8.5,
                color=color,
            )
        else:
            ax.plot(times, values * 100, label=label, color=color, linewidth=1.8)
        plotted[f"{name}_times"] = times
        plotted[f"{name}_xml_node_clearance_m"] = values

    for column, mapping in enumerate(("uncompensated", "compensated")):
        ax = axes[0, column]
        for method in ("fixed", "adaptive"):
            clearance_line(
                ax,
                traces[mapping],
                f"{method}_",
                original[mapping]["methods"][method],
                "Fixed" if method == "fixed" else "Online adaptation",
                COLORS[method],
                f"{mapping}_{method}",
            )
        ax.set(
            title=f"{'A' if column == 0 else 'B'}  Full episode · {mapping} fallbacks",
            xlabel="Absolute simulation time (s)",
            ylabel="Actual XML-sphere clearance (cm)",
            xlim=(3.6, 5.8),
            ylim=(-9, 80),
        )
        ax.axhline(0, color="black", linewidth=0.8)
        for mover in (scene.incoming, *scene.additional_incoming):
            ax.axvline(
                mover.arrival_time_seconds, color="#777777", linewidth=0.7, alpha=0.4, linestyle=":"
            )
        ax.legend(loc="upper right", frameon=True, framealpha=0.85, fontsize=9)
        if mapping == "compensated":
            ax.text(
                0.97,
                0.17,
                "Both methods complete safely",
                ha="right",
                transform=ax.transAxes,
                color="#3f555a",
                fontsize=9,
            )

    ax = axes[1, 0]
    version_rows = (
        (
            traces["uncompensated"],
            "adaptive_",
            "Continued online learning",
            COLORS["adaptive"],
            "-",
        ),
        (onset_trace, "", "Freeze at wind onset", COLORS["onset"], "--"),
        (held_trace, "", "Hold available version at 3 s", COLORS["held"], "-"),
        (reverted_trace, "", "Restore original parameters at 3 s", COLORS["fixed"], "--"),
    )
    for index, (record, prefix, label, color, style) in enumerate(version_rows):
        times, versions = record[f"{prefix}time"], record[f"{prefix}version_used"]
        ax.step(
            times, versions, where="post", label=label, color=color, linestyle=style, linewidth=1.6
        )
        plotted[f"version_{index}_times"] = times
        plotted[f"version_{index}_used"] = versions
    ax.axvline(scene.wind_onset_seconds, color="#5d5d5d", linestyle=":", linewidth=1)
    ax.axvline(early["boundary_seconds"], color="#5d5d5d", linestyle=":", linewidth=1)
    ax.text(
        scene.wind_onset_seconds - 0.06, 807, "wind on\n2.2 s", ha="right", va="top", fontsize=8.5
    )
    ax.text(
        early["boundary_seconds"] + 0.06,
        807,
        "parameter intervention\n3.0 s",
        ha="left",
        va="top",
        fontsize=8.5,
    )
    ax.set(
        title="C  Only causally available parameters are applied",
        xlabel="Absolute simulation time (s)",
        ylabel="Applied parameter snapshot version",
        xlim=(0, 5.6),
        ylim=(647, 815),
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.24),
        ncol=2,
        fontsize=8.0,
        frameon=False,
        borderaxespad=0,
    )

    ax = axes[1, 1]
    clearance_line(
        ax,
        reverted_trace,
        "",
        early["methods"]["fixed"],
        "Restore original at 3 s",
        COLORS["fixed"],
        "reverted_at_3s",
    )
    clearance_line(
        ax,
        held_trace,
        "",
        early["methods"]["adaptive"],
        "Hold available learned version",
        COLORS["held"],
        "held_at_3s",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        title="D  Same full-scene prefix; parameters differ from 3 s",
        xlabel="Absolute simulation time (s)",
        ylabel="Actual XML-sphere clearance (cm)",
        xlim=(3.6, 5.8),
        ylim=(-9, 80),
    )
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.85)
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.suptitle(
        "Collision prevention in the selected full scene, with a causal parameter intervention\n"
        "The stronger model-compensated frozen comparison also succeeds",
        fontsize=13,
        y=0.97,
    )
    fig.text(
        0.075,
        0.045,
        "Clearance: oriented XML sphere at recorded plant nodes; ×: swept first contact.\n"
        "Dashed after ×: remainder of the final control hold, without contact physics; "
        "depth is not impact severity.\n"
        f"C: deterministic next-boundary execution. At {early['boundary_seconds']:g} s, "
        f"version {held_version} includes {onset_version - initial_version} calm "
        f"+{held_version - onset_version} wind updates.",
        fontsize=8.4,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"closed_loop_and_causal_results.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.78, bottom=0.25, wspace=0.28)
    buffers = (0.15, 0.05, 0.02, 0.0)
    buffer_table = {}
    for ax, mapping in zip(axes, ("uncompensated", "compensated"), strict=True):
        buffer_table[mapping] = {}
        for column, buffer in enumerate(buffers):
            row = (
                original[mapping]
                if buffer == 0.15
                else read(targeted / mapping / f"clearance_{buffer:.2f}/result.json")
            )
            matched_scene = dict(row["scene"])
            matched_scene["obstacle_clearance"] = scene.obstacle_clearance
            if matched_scene != original[mapping]["scene"]:
                raise ValueError("a buffer comparison changes another scene parameter")
            buffer_table[mapping][str(buffer)] = {}
            for y, method in enumerate(("fixed", "adaptive")):
                measured = row["methods"][method]
                collision = measured["collider_upper_m"] < 0
                text = (
                    f"COLLISION\n{measured['first_collider_intersection_seconds']:.3f} s"
                    if collision
                    else f"CLEAR\n+{100 * measured['collider_lower_m']:.2f} cm"
                )
                color = "#f5ded1" if collision else "#d8ecee"
                ax.add_patch(
                    Rectangle((column - 0.48, y - 0.45), 0.96, 0.9, color=color, linewidth=0)
                )
                ax.text(
                    column,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=COLORS["fixed"] if collision else "#075b63",
                )
                buffer_table[mapping][str(buffer)][method] = {
                    "collision": collision,
                    "collider_lower_m": measured["collider_lower_m"],
                    "first_collider_intersection_seconds": measured[
                        "first_collider_intersection_seconds"
                    ],
                    "termination": measured["termination"],
                }
        ax.set(
            xlim=(-0.5, 3.5),
            ylim=(1.5, -0.5),
            xticks=range(4),
            xticklabels=("15 cm", "5 cm", "2 cm", "0 cm"),
            yticks=(0, 1),
            yticklabels=("Fixed", "Adaptive"),
            xlabel="Optional obstacle clearance",
            title=f"Matched {mapping} pair",
        )
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        "Reducing the optional buffer is unnecessary for the uncompensated result\n"
        "The compensated fixed baseline succeeds at every tested buffer",
        fontsize=13,
        y=0.96,
    )
    fig.text(
        0.085,
        0.09,
        "Same physical obstacle radii and 0.106 m ego enclosure in all eight paired episodes.\n"
        "CLEAR gives audited minimum positive XML-sphere clearance; "
        "COLLISION gives first swept contact time.\n"
        "Penetration depths depend on the first control-hold stop boundary "
        "and are not compared as crash severity.",
        fontsize=8.5,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"matched_buffer_outcomes.{extension}", dpi=180)
    plt.close(fig)
    np.savez_compressed(output / "figure_data.npz", **plotted)
    (output / "source_manifest.json").write_text(
        json.dumps(
            {
                "scope": __doc__,
                "inputs_sha256": sources,
                "figure_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "buffer_table": buffer_table,
                "onset_snapshot_version": onset["methods"]["fixed"]["wind_onset_version"],
                "intervention_snapshot_version": early["provenance"]["held_learned"][
                    "used_snapshot"
                ]["library_version"],
                "outputs_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(output.iterdir())
                },
            },
            indent=2,
        )
        + "\n"
    )
    if any(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected
        for path, expected in sources.items()
    ):
        raise ValueError("an input changed during figure generation")


def main() -> None:
    """Generate figures with a fresh destination and exact input/source checksums."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_figures(args.root, args.output)


if __name__ == "__main__":
    main()
