"""Render complete paired navigation campaigns from JSON, without rerunning experiments.

Statistics are consumed from da_plcbf_navigation_summary.py outputs. No uncertainty estimates
are fitted here. Every predeclared seed must exist before any output directory is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

COLORS = {"fixed": "#2364AA", "adaptive": "#D86A24", "effect": "#66449A"}
METHOD_LABELS = {"fixed": "Frozen", "adaptive": "Adaptive"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_group(
    root: Path, protocol_name: str, pattern: str, statistics_path: Path, condition: str
) -> dict[str, Any]:
    protocol_path = root / protocol_name
    protocol = json.loads(protocol_path.read_text())
    expected = set(protocol["heldout_world_seeds"])
    seen: dict[int, dict[str, Any]] = {}
    for path in sorted(root.glob(pattern)):
        if path.parent.name.split("-seed")[0] != condition:
            continue
        summary = json.loads(path.read_text())["summary"]
        world = summary["world"]["config"]
        seed = world["seed"]
        if seed not in expected or seed in seen:
            raise ValueError(f"unexpected or repeated world seed {seed}: {path}")
        if summary["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
            raise ValueError("checkpoint differs from the frozen protocol")
        if world["obstacle_count"] != protocol["obstacles"]:
            raise ValueError("obstacle count differs from the frozen protocol")
        if (
            world["waypoint_count"] != 8
            or not world["moving_obstacles"]
            or world["duration_seconds"] != protocol["duration_seconds"]
            or world["duration_seconds"] != 40
            or not np.isclose(world["dt"] * world["control_interval_steps"], 0.04)
        ):
            raise ValueError("run differs from the eight-waypoint, 40 s / 40 ms figure contract")
        if condition == "unchanged":
            if world["wind_events"] or world["payload_events"]:
                raise ValueError("unchanged condition contains disturbance events")
        elif (
            [event["time_seconds"] for event in world["wind_events"]] != [8, 24]
            or [event["time_seconds"] for event in world["payload_events"]] != [16]
            or world["payload_events"][0]["mass_fraction"] != 0.25
        ):
            raise ValueError("combined condition differs from the labeled event schedule")
        if summary["config"]["model_information"] != "oracle":
            raise ValueError("these campaign figures require the declared oracle comparison")
        source_path = path.parent / "SOURCE_SHA256.json"
        source = json.loads(source_path.read_text())
        if "crazyflow/safety/da_plcbf/navigation_experiment.py" not in source or any(
            protocol["files"].get(name) != digest for name, digest in source.items()
        ):
            raise ValueError("run source differs from its frozen protocol")
        seen[seed] = {
            "seed": seed,
            "path": path,
            "source_path": source_path,
            "methods": summary["methods"],
        }
    if set(seen) != expected:
        raise ValueError(
            f"incomplete {protocol['obstacles']}/{condition}: "
            f"missing {sorted(expected - set(seen))}"
        )
    statistics = json.loads(statistics_path.read_text())["conditions"][condition]
    rows = [seen[seed] for seed in sorted(seen)]
    if statistics["count"] != len(rows) or set(statistics["world_seeds"]) != expected:
        raise ValueError("saved statistics do not describe the same complete world set")
    for method in ("fixed", "adaptive"):
        complete = sum(row["methods"][method]["termination"] == "completed" for row in rows)
        if (
            complete
            != statistics["methods"][method]["criteria"]["all_waypoints_completed"]["count"]
        ):
            raise ValueError("completion counts disagree with saved statistics")
    for metric, key in (
        ("minimum_inflated_clearance_m", "paired_adaptive_minus_fixed_shell_margin_m"),
        ("waypoints_completed", "paired_adaptive_minus_fixed_waypoints"),
        (
            "termination_time_seconds",
            "paired_adaptive_minus_fixed_completion_seconds_both_complete",
        ),
    ):
        selected = (
            rows
            if metric != "termination_time_seconds"
            else [
                row
                for row in rows
                if all(
                    row["methods"][m]["termination"] == "completed" for m in ("fixed", "adaptive")
                )
            ]
        )
        differences = [
            row["methods"]["adaptive"][metric] - row["methods"]["fixed"][metric] for row in selected
        ]
        if len(differences) != statistics[key]["count"]:
            raise ValueError("paired sample counts disagree with saved statistics")
        if differences and not np.isclose(
            np.mean(differences), statistics[key]["mean"], rtol=1e-10, atol=1e-12
        ):
            raise ValueError("paired mean disagrees with saved statistics")
        if differences and not np.allclose(
            np.sort(differences),
            np.sort(statistics[key]["individual_differences"]),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("paired observations disagree with saved statistics")
    dynamics = "unchanged dynamics" if condition == "unchanged" else "wind + payload"
    short_dynamics = "unchanged" if condition == "unchanged" else "combined"
    return {
        "title": f"{protocol['obstacles']} obstacles · {dynamics}",
        "short_title": f"{protocol['obstacles']} obstacles\n{short_dynamics}",
        "protocol": protocol_path,
        "statistics_path": statistics_path,
        "statistics": statistics,
        "rows": rows,
    }


def _style(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.65)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=9)


def _raw_figure(groups: list[dict[str, Any]]) -> Any:
    figure, axes = plt.subplots(3, 3, figsize=(14.3, 10.8), sharey="row")
    figure.subplots_adjust(
        left=0.082, right=0.985, top=0.875, bottom=0.115, hspace=0.36, wspace=0.14
    )
    figure.suptitle(
        "Matched 3D waypoint navigation: paired world results",
        x=0.082,
        y=0.977,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.082,
        0.943,
        "Same initial library, model information and actuator limits · "
        "40 s episode limit · 40 ms control period",
        fontsize=11,
        color="#374151",
    )
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=COLORS[m],
                marker="o" if m == "fixed" else "s",
                linestyle="",
                label=METHOD_LABELS[m],
            )
            for m in ("fixed", "adaptive")
        ],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.954),
        ncol=2,
        frameon=False,
    )
    for column, group in enumerate(groups):
        rows, statistics = group["rows"], group["statistics"]
        positions = np.arange(len(rows))
        labels = [str(row["seed"]) for row in rows]
        for axis in axes[:, column]:
            _style(axis)
            axis.set_xticks(positions, labels, rotation=45)
            axis.set_xlim(-0.5, len(rows) - 0.5)
        axes[0, column].set_title(group["title"], fontsize=12, pad=14, fontweight="bold")
        for row_index, metric in ((0, "minimum_inflated_clearance_m"), (1, "waypoints_completed")):
            axis = axes[row_index, column]
            frozen = np.asarray([row["methods"]["fixed"][metric] for row in rows])
            adaptive = np.asarray([row["methods"]["adaptive"][metric] for row in rows])
            for x, left, right in zip(positions, frozen, adaptive, strict=True):
                axis.plot(
                    [x - 0.12, x + 0.12], [left, right], color="#ABB2BC", linewidth=1, zorder=1
                )
            axis.scatter(
                positions - 0.12, frozen, color=COLORS["fixed"], s=31, marker="o", zorder=3
            )
            axis.scatter(
                positions + 0.12, adaptive, color=COLORS["adaptive"], s=29, marker="s", zorder=3
            )
            if row_index == 0:
                axis.axhline(0, color="#AC283B", linewidth=1.0, linestyle="--")
            else:
                complete = {
                    m: statistics["methods"][m]["criteria"]["all_waypoints_completed"]["count"]
                    for m in ("fixed", "adaptive")
                }
                axis.text(
                    0.02,
                    1.025,
                    f"All 8: frozen {complete['fixed']}/{len(rows)}, "
                    f"adaptive {complete['adaptive']}/{len(rows)}",
                    transform=axis.transAxes,
                    va="bottom",
                    fontsize=9,
                )
        axis = axes[2, column]
        axis.axhline(0, color="#6B7280", linewidth=1, linestyle="--")
        included = []
        for x, row in zip(positions, rows, strict=True):
            both = all(
                row["methods"][m]["termination"] == "completed" for m in ("fixed", "adaptive")
            )
            if both:
                delta = (
                    row["methods"]["adaptive"]["termination_time_seconds"]
                    - row["methods"]["fixed"]["termination_time_seconds"]
                )
                axis.plot([x, x], [0, delta], color=COLORS["effect"], alpha=0.25, linewidth=1)
                axis.scatter(x, delta, color=COLORS["effect"], s=32, zorder=3)
                included.append(row["seed"])
            else:
                axis.text(
                    x,
                    0.045,
                    "×",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    color="#888888",
                    fontsize=12,
                )
        axis.text(
            0.02,
            0.95,
            f"Both complete: {len(included)}/{len(rows)} pairs",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
        axis.set_xlabel("World seed", fontsize=10)
    axes[0, 0].set_ylabel("Minimum inflated-shell clearance (m)", fontsize=10)
    axes[1, 0].set_ylabel("Waypoints reached / 8", fontsize=10)
    axes[1, 0].set_ylim(-0.3, 9.0)
    axes[1, 0].set_yticks([0, 2, 4, 6, 8])
    axes[2, 0].set_ylabel("Completion time: adaptive − frozen (s)", fontsize=10)
    low, high = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(min(low, -0.001), high)
    low, high = axes[2, 0].get_ylim()
    span = max(high - low, 1.0)
    axes[2, 0].set_ylim(low - 0.04 * span, high + 0.13 * span)
    figure.text(
        0.082,
        0.056,
        "Moving obstacles in every condition. Combined: wind changes at 8 and 24 s; "
        "centered +25% mass at 16 s.",
        fontsize=9,
        color="#374151",
    )
    figure.text(
        0.082,
        0.032,
        "Completion-time differences include only pairs where both methods finish; "
        "× marks excluded pairs. Lower time differences favor adaptation.",
        fontsize=9,
        color="#374151",
    )
    return figure


def _summary_figure(groups: list[dict[str, Any]]) -> Any:
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 7.5))
    figure.subplots_adjust(left=0.19, right=0.94, top=0.84, bottom=0.14, hspace=0.55, wspace=0.40)
    figure.suptitle(
        "Paired effects and completion uncertainty",
        x=0.065,
        y=0.96,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.065,
        0.914,
        "Adaptive − frozen · diamonds show paired means · "
        "horizontal intervals come from the saved campaign statistics",
        fontsize=10,
        color="#374151",
    )
    labels = [group["short_title"] for group in groups]
    y = np.arange(len(groups))
    for axis in axes.flat:
        _style(axis)
        axis.set_yticks(y, labels)
        axis.set_ylim(len(groups) - 0.55, -0.55)
    for axis, key, title, xlabel in (
        (
            axes[0, 0],
            "paired_adaptive_minus_fixed_shell_margin_m",
            "Minimum shell clearance",
            "Paired difference (m) · higher favors adaptation",
        ),
        (
            axes[0, 1],
            "paired_adaptive_minus_fixed_waypoints",
            "Waypoint progress",
            "Paired difference (waypoints) · higher favors adaptation",
        ),
        (
            axes[1, 0],
            "paired_adaptive_minus_fixed_completion_seconds_both_complete",
            "Completion time · only pairs both complete",
            "Paired difference (s) · lower favors adaptation",
        ),
    ):
        axis.axvline(0, color="#777777", linewidth=0.9, linestyle="--")
        for index, group in enumerate(groups):
            stat = group["statistics"][key]
            if not stat["count"]:
                axis.text(
                    0.5,
                    index,
                    "No qualifying pairs",
                    transform=axis.get_yaxis_transform(),
                    ha="center",
                    fontsize=9,
                )
                continue
            low, high = stat["bootstrap_mean_95_interval"]
            axis.hlines(index, low, high, color=COLORS["effect"], linewidth=2.5)
            axis.plot(stat["mean"], index, marker="D", color=COLORS["effect"], markersize=6)
            axis.text(
                1.02,
                index,
                f"n={stat['count']}",
                transform=axis.get_yaxis_transform(),
                va="center",
                fontsize=8,
            )
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=11)
        axis.set_xlabel(xlabel, fontsize=9)
        axis.margins(x=0.18)
    axis = axes[1, 1]
    for index, group in enumerate(groups):
        for method, offset in (("fixed", -0.13), ("adaptive", 0.13)):
            stat = group["statistics"]["methods"][method]["criteria"]["all_waypoints_completed"]
            low, high = stat["wilson_95_interval"]
            axis.hlines(index + offset, low, high, color=COLORS[method], linewidth=2)
            axis.plot(
                stat["fraction"],
                index + offset,
                marker="o" if method == "fixed" else "s",
                color=COLORS[method],
                markersize=5,
            )
    axis.set_xlim(-0.03, 1.04)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    axis.set_title("All 8 waypoints completed", loc="left", fontsize=11, fontweight="bold", pad=11)
    axis.set_xlabel("Completion fraction · 95% Wilson interval", fontsize=9)
    figure.legend(
        handles=[
            Line2D(
                [], [], color=COLORS[m], marker="o" if m == "fixed" else "s", label=METHOD_LABELS[m]
            )
            for m in ("fixed", "adaptive")
        ],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.015),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.065,
        0.065,
        "Paired intervals: 95% bootstrap intervals for the mean difference. "
        "Completion intervals: Wilson, per method.",
        fontsize=9,
        color="#374151",
    )
    figure.text(
        0.065,
        0.035,
        "Descriptive uncertainty over these world draws with one fixed learner checkpoint; "
        "no general safety guarantee.",
        fontsize=9,
        color="#374151",
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--main-summary", type=Path, required=True)
    parser.add_argument("--dense-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = [
        _load_group(
            args.root,
            "CAMPAIGN_PROTOCOL.json",
            "heldout-*/*/navigation_comparison.json",
            args.main_summary,
            condition,
        )
        for condition in ("unchanged", "combined")
    ]
    groups.append(
        _load_group(
            args.root,
            "DENSE_CONFIRMATION_PROTOCOL.json",
            "dense-confirmation*/*/navigation_comparison.json",
            args.dense_summary,
            "combined",
        )
    )
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 11,
            "savefig.facecolor": "white",
        }
    )
    outputs = []
    for name, figure in (
        ("navigation_paired_worlds", _raw_figure(groups)),
        ("navigation_paired_uncertainty", _summary_figure(groups)),
    ):
        for extension in ("png", "pdf"):
            path = args.output / f"{name}.{extension}"
            figure.savefig(path, dpi=190)
            outputs.append(path)
        plt.close(figure)
    command = shlex.join(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(args.root.resolve()),
            "--main-summary",
            str(args.main_summary.resolve()),
            "--dense-summary",
            str(args.dense_summary.resolve()),
            "--output",
            str(args.output.resolve()),
        ]
    )
    (args.output / "REPRODUCE.txt").write_text(
        command
        + "\n\nUse a fresh --output directory when reproducing; existing artifacts are kept.\n"
    )
    inputs = {
        str(path.resolve()): _sha(path)
        for group in groups
        for path in (
            group["protocol"],
            group["statistics_path"],
            *[row["path"] for row in group["rows"]],
            *[row["source_path"] for row in group["rows"]],
        )
    }
    reporting_patch = args.root / "CAMPAIGN_OUTPUT_PATCH.json"
    if reporting_patch.exists():
        inputs[str(reporting_patch.resolve())] = _sha(reporting_patch)
    (args.output / "FIGURE_PROVENANCE.json").write_text(
        json.dumps(
            {
                "scope": (
                    "Complete predeclared campaigns only; source JSON and saved statistics, "
                    "no simulation rerun."
                ),
                "command": command,
                "script_sha256": _sha(Path(__file__)),
                "software": {
                    "python": sys.version,
                    "matplotlib": matplotlib.__version__,
                    "numpy": np.__version__,
                },
                "inputs": inputs,
                "outputs": {path.name: _sha(path) for path in outputs},
                "groups": [
                    {
                        "title": group["title"],
                        "world_seeds": [row["seed"] for row in group["rows"]],
                        "statistics": group["statistics"],
                    }
                    for group in groups
                ],
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paired_world_counts": [len(group["rows"]) for group in groups],
                "files": [str(path) for path in outputs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
