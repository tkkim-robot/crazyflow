"""Render publication figures from completed, authenticated actuator-study artifacts.

Example (all paths must name actual retained outputs)::

    python -m benchmark.da_plcbf_actuator_figures --study-root /abs/study/v1 \
        --main-campaign /abs/main --sealed-analysis /abs/analysis/analysis.json \
        --secondary-map /abs/secondary-roles.json --k-profile /abs/k-cost \
        --output /abs/fresh-figures

Secondary-map keys are explicit experimental roles, never inferred from directory
names: oracle, a1, retention_off, fine_p0, p1, p2, effectiveness_bias, lag_scale,
parameter_delay, motor_noise, position_noise, obstacle_bias, paced, delayed.
Values are campaign directories. The last two observation factors are optional.
Required missing inputs refuse a figure by default; --missing skip records why it
was omitted. Corrupt or inconsistent supplied artifacts always raise an error.
No training, simulation, bootstrap resampling, or synthetic outcomes are performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

# Crazyflow's package registration imports JAX even for its NumPy protocol module.
# Artifact rendering must never reserve a GPU used by a concurrent experiment.
os.environ["JAX_PLATFORMS"] = "cpu"

from benchmark.da_plcbf_actuator_analysis import (  # noqa: E402
    SCHEMA,
    Sources,
    aggregate_groups,
    read_campaign,
)
from crazyflow.safety.da_plcbf.actuator_protocol import (  # noqa: E402
    TrialLedger,
    content_sha256,
    load_sealed_protocol,
)

METHODS = ("F2", "DR", "A", "OPT")
COLORS = {"F2": "#64748b", "DR": "#d8891c", "A": "#007d85", "OPT": "#8556a6", "A1": "#ba4865"}
FIGURES = ("main", "recovery", "compute", "ablations", "mismatch", "timing")
RECOVERY_METRICS = {
    "position_rmse_max_m": ("Maximum position RMSE (m)", 0.3),
    "terminal_speed_max_mps": ("Maximum terminal speed (m/s)", 0.8),
}
ROLES = {
    "oracle",
    "a1",
    "retention_off",
    "fine_p0",
    "p1",
    "p2",
    "effectiveness_bias",
    "lag_scale",
    "parameter_delay",
    "motor_noise",
    "position_noise",
    "obstacle_bias",
    "paced",
    "delayed",
}


class MissingEvidence(RuntimeError):
    """A required artifact or complete matrix has not yet been produced."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def same_number(actual: Any, expected: Any, message: str) -> None:
    require(
        isinstance(actual, (int, float))
        and isinstance(expected, (int, float))
        and math.isfinite(actual)
        and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
        message,
    )


def file_required(path: Path | None, label: str) -> Path:
    if path is None or not path.is_file():
        raise MissingEvidence(f"{label} is missing: {path}")
    return path.resolve()


def child(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    require(path.is_relative_to(directory.resolve()), "artifact path escapes its directory")
    return path


def key(row: dict[str, Any]) -> tuple[str, int, str]:
    return row["physical_world_id"], row["library_seed"], row["method"]


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), "refusing an empty plot-data CSV")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=sorted(set().union(*(row.keys() for row in rows)))
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for name, value in row.items()
                }
            )


def flat(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            name: leaf
            for k, v in value.items()
            for name, leaf in flat(v, f"{prefix}.{k}".strip(".")).items()
        }
    return {prefix: value}


class Evidence:
    """Cache authenticated files and retain exact source hashes for the figure manifest."""

    def __init__(self, args: argparse.Namespace):
        """Resolve explicit role mappings without inspecting any outcome."""
        self.args = args
        self.sources = Sources()
        self.campaigns: dict[str, dict[str, Any]] = {}
        self.main_data: tuple[dict, dict] | None = None
        self.roles: dict[str, Path] = {}
        if args.secondary_map is not None:
            path = file_required(args.secondary_map, "secondary role mapping")
            mapping = self.sources.read(path)
            require(isinstance(mapping, dict), "secondary map must be a role-to-directory object")
            require(
                set(mapping).issubset(ROLES), f"unsupported secondary roles: {set(mapping) - ROLES}"
            )
            for role, value in mapping.items():
                require(
                    isinstance(value, str) and bool(value),
                    "secondary paths must be nonempty strings",
                )
                self.roles[role] = (path.parent / value).resolve()

    def campaign(self, directory: Path | None, label: str) -> dict[str, Any]:
        """Read and authenticate a complete campaign once."""
        if directory is None:
            raise MissingEvidence(f"No explicit directory supplied for {label}")
        directory = directory.resolve()
        file_required(directory / "campaign_result.json", label)
        cache_key = str(directory)
        if cache_key not in self.campaigns:
            dataset = read_campaign(directory, self.sources)
            dataset["binding"] = self.sources.read(directory / "campaign_binding.json")
            self.campaigns[cache_key] = dataset
        dataset = self.campaigns[cache_key]
        if dataset["completed_trials"] != dataset["planned_trials"]:
            raise MissingEvidence(
                f"{label} is incomplete: {dataset['completed_trials']}/{dataset['planned_trials']}"
            )
        for row in dataset["rows"]:
            require(
                row["status"] == "completed", f"completed count hides an incomplete row in {label}"
            )
            if any(
                row["metrics"].get(metric) is None
                for metric in (
                    "actual_collision",
                    "safe_task_completion",
                    "collider_clearance_lower_m",
                    "controller_deadline_miss_rate",
                )
            ):
                raise MissingEvidence(f"Unresolved required outcome in {label}: {key(row)}")
        return dataset

    def role(self, role: str) -> dict[str, Any]:
        """Resolve an explicitly named experimental role."""
        dataset = self.campaign(self.roles.get(role), role)
        spec = dataset["binding"]["specification"]
        require(spec.get("split") == "validation", f"{role} is not a validation campaign")
        if role not in {"paced", "delayed"}:
            require(
                spec["library_seeds"] == [11]
                and {(w["scene_seed"], w["family"], w["dynamics_cell"]) for w in spec["worlds"]}
                == {(2003, "structured", "combined"), (2103, "navigation", "combined")},
                f"{role} differs from the declared two-world validation support",
            )
        for row in dataset["rows"]:
            expected_mode = role if role in {"paced", "delayed"} else "deterministic"
            require(
                self.configuration(row)["config"]["execution_mode"] == expected_mode,
                f"{role} does not contain {expected_mode} execution",
            )
        return dataset

    def main(self) -> tuple[dict, dict]:
        """Cross-check the actual test matrix against its seal, ledger and analysis."""
        if self.main_data is not None:
            return self.main_data
        analysis_path = file_required(self.args.sealed_analysis, "sealed analysis JSON")
        analysis = self.sources.read(analysis_path)
        if self.args.analysis_sha256 is not None:
            require(
                self.sources.hashes[str(analysis_path)] == self.args.analysis_sha256,
                "analysis bytes differ from the independently supplied SHA-256 digest",
            )
        require(
            analysis.get("schema") == SCHEMA and analysis.get("input_kind") == "sealed_ledger",
            "main figure requires a sealed-ledger analysis",
        )
        input_objects = []
        require(bool(analysis.get("input_sha256")), "analysis has no authenticated inputs")
        for name, expected in analysis["input_sha256"].items():
            path = Path(name)
            path = path if path.is_absolute() else analysis_path.parent / path
            file_required(path, "analysis input")
            require(self.sources.bind(path) == expected, f"analysis input hash mismatch: {path}")
            if path.suffix == ".json":
                input_objects.append((path, self.sources.read(path)))
        protocols = [
            (path, obj)
            for path, obj in input_objects
            if str(obj.get("schema", "")).startswith("sealed_da_plcbf_actuator_protocol")
        ]
        ledgers = [
            obj
            for _, obj in input_objects
            if obj.get("schema") == "da_plcbf_actuator_trial_ledger_v1"
        ]
        require(
            len(protocols) == len(ledgers) == 1,
            "analysis must bind exactly one sealed protocol and ledger",
        )
        protocol = load_sealed_protocol(protocols[0][0])
        ledger = TrialLedger.from_mapping(protocol, ledgers[0])
        require(analysis["protocol_sha256"] == protocol.sha256, "analysis protocol mismatch")
        require(
            analysis["settings"] == protocol.manifest["analysis"],
            "analysis settings differ from seal",
        )
        require(
            analysis["metrics"] == protocol.manifest["outcomes"],
            "analysis outcomes differ from seal",
        )
        require(analysis["coverage"] == ledger.coverage(), "analysis coverage differs from ledger")
        require(
            analysis["attempts"] == ledger.as_mapping()["records"],
            "analysis attempts differ from ledger",
        )
        dataset = self.campaign(self.args.main_campaign, "main campaign")
        binding, manifest = dataset["binding"], protocol.manifest
        expected_analysis_source = manifest["source"]["analysis_files_sha256"][
            "benchmark/da_plcbf_actuator_analysis.py"
        ]
        require(
            analysis.get("analysis_source_sha256") == expected_analysis_source,
            "analysis implementation differs from the sealed implementation",
        )
        require(
            self.sources.bind(Path(__file__).with_name("da_plcbf_actuator_analysis.py"))
            == expected_analysis_source,
            "current metric adapter differs from the sealed analysis implementation",
        )
        spec = binding["specification"]
        unsigned = {k: v for k, v in spec.items() if k != "sealed_protocol_path"}
        require(
            spec.get("split") == "test"
            and content_sha256(unsigned) == manifest["campaign_specification_sha256"],
            "main campaign is outside the sealed test specification",
        )
        require(
            binding.get("sealed_protocol_sha256") == protocol.sha256,
            "main campaign does not bind this seal",
        )
        require(
            binding["source_sha256"] == manifest["source"]["files_sha256"]
            and binding["checkpoint_sha256"] == manifest["checkpoint_sha256"],
            "main source/checkpoint bytes differ from seal",
        )
        test = manifest["splits"]["test"]
        expected_groups = [
            {
                "group": content_sha256({"protocol": protocol.sha256, "cell": cell}),
                "dataset": protocol.sha256,
                "cell": cell,
                "worlds": [w["world_id"] for w in test["worlds"] if w["cell"] == cell],
                "library_seeds": test["library_seeds"],
                "methods": list(manifest["methods"]),
            }
            for cell in sorted({w["cell"] for w in test["worlds"]})
        ]
        require(analysis["groups"] == expected_groups, "analysis groups differ from sealed support")
        require(
            len(expected_groups) == 8
            and all(len(group["worlds"]) == 4 for group in expected_groups)
            and test["library_seeds"] == [11, 23, 37]
            and set(manifest["methods"]) == set(METHODS),
            "main figure requires the sealed 32-world, three-library, four-method matrix",
        )
        actual_attempts = {}
        for attempt in dataset["attempts"]:
            path = Path(attempt["path"])
            attempt_name = path.parent.name if path.name == "summary.json" else path.name
            identity = f"{dataset['id']}:{attempt_name}:{attempt['trial']}"
            require(identity not in actual_attempts, "duplicate retained main attempt")
            actual_attempts[identity] = attempt
        require(
            set(actual_attempts) == {r["attempt_id"] for r in ledger.as_mapping()["records"]},
            "ledger omits or invents a retained main attempt",
        )
        raw = {key(row): row for row in dataset["rows"]}
        analyzed = {key(row): row for row in analysis["rows"]}
        require(
            len(raw) == len(dataset["rows"]) and len(analyzed) == len(analysis["rows"]),
            "duplicate main trial identities",
        )
        require(raw.keys() == analyzed.keys(), "analysis and actual main trial identities differ")
        completed = {
            (r["world_id"], r["library_seed"], r["method"]): r
            for r in ledger.as_mapping()["records"]
            if r["split"] == "test" and r["status"] == "completed"
        }
        require(completed.keys() == raw.keys(), "ledger lacks the actual completed main matrix")
        for identity, row in raw.items():
            other = analyzed[identity]
            sealed_group = next(
                group for group in expected_groups if identity[0] in group["worlds"]
            )
            require(
                other["status"] == "completed"
                and row["cell"] == other["cell"] == sealed_group["cell"]
                and other["dataset"] == protocol.sha256
                and other["world"] == other["physical_world_id"]
                and other["group"] == sealed_group["group"],
                "main row status or cell differs from analysis",
            )
            self.checkpoint(dataset, row)
            for metric in analysis["metrics"]:
                same_number(
                    other["metrics"].get(metric),
                    row["metrics"].get(metric),
                    f"main raw/analysis metric mismatch: {identity}, {metric}",
                )
                same_number(
                    other["metrics"].get(metric),
                    completed[identity]["metrics"].get(metric),
                    f"main ledger/analysis metric mismatch: {identity}, {metric}",
                )
        recalculated = aggregate_groups(
            analysis["rows"],
            analysis["groups"],
            analysis["metrics"],
            analysis["settings"]["confidence_level"],
        )
        require(
            recalculated == analysis["aggregates"],
            "main aggregates do not reproduce from the retained rows",
        )
        settings = analysis["settings"]
        comparisons = settings["primary_comparisons"]
        expected_pairs = {
            (name, cell, metric)
            for name, comp in comparisons.items()
            for cell, metric in itertools.product(comp["cells"], comp["metrics"])
        }
        observed_pairs = {
            (p["comparison_id"], p["cell"], p["metric"]) for p in analysis["paired_comparisons"]
        }
        require(
            expected_pairs == observed_pairs
            and len(observed_pairs) == len(analysis["paired_comparisons"]),
            "sealed paired-comparison family is incomplete or duplicated",
        )
        confidence = 1 - (1 - settings["confidence_level"]) / len(expected_pairs)
        for pair in analysis["paired_comparisons"]:
            comp = comparisons[pair["comparison_id"]]
            group = next(g for g in expected_groups if g["cell"] == pair["cell"])
            require(
                all(pair[name] == value for name, value in group.items()),
                "paired contrast groups differ from the sealed support",
            )
            require(
                pair["method_a"] == comp["method_a"]
                and pair["method_b"] == comp["method_b"]
                and pair["effect_direction"] == "method_a_minus_method_b",
                "paired comparison direction differs from seal",
            )
            rows = [r for r in analysis["rows"] if r["cell"] == pair["cell"]]
            a = [r["metrics"][pair["metric"]] for r in rows if r["method"] == pair["method_a"]]
            b = [r["metrics"][pair["metric"]] for r in rows if r["method"] == pair["method_b"]]
            same_number(
                pair["estimate"],
                np.mean(a) - np.mean(b),
                "paired estimate differs from actual rows",
            )
            interval = pair.get("interval")
            if interval is None:
                raise MissingEvidence(f"Sealed interval unavailable: {pair.get('reason', pair)}")
            same_number(interval["estimate"], pair["estimate"], "paired interval estimate mismatch")
            same_number(
                interval["confidence_level"],
                confidence,
                "paired interval multiplicity differs from seal",
            )
            worlds = next(g["worlds"] for g in expected_groups if g["cell"] == pair["cell"])
            effects = [
                float(
                    np.mean(
                        [
                            raw[(world, seed, pair["method_a"])]["metrics"][pair["metric"]]
                            - raw[(world, seed, pair["method_b"])]["metrics"][pair["metric"]]
                            for world in worlds
                        ]
                    )
                )
                for seed in test["library_seeds"]
            ]
            require(
                np.allclose(interval["per_seed_effects"], effects, rtol=1e-10, atol=1e-12)
                and interval["n_worlds"] == len(worlds)
                and interval["n_library_seeds"] == len(test["library_seeds"])
                and interval["n_pairs"] == len(worlds) * len(test["library_seeds"])
                and interval["n_resamples"] == settings["n_resamples"]
                and interval["resampling_seed"] == settings["resampling_seed"],
                "paired interval counts, seeds or per-library effects differ from the seal",
            )
            require(
                all(
                    isinstance(interval[name], (int, float)) and math.isfinite(interval[name])
                    for name in ("lower", "upper", "world_only_lower", "world_only_upper")
                )
                and interval["lower"] <= interval["upper"]
                and interval["world_only_lower"] <= interval["world_only_upper"],
                "invalid or reversed confidence limits",
            )
            if analysis["metrics"][pair["metric"]]["kind"] == "binary":
                require(
                    -1 <= interval["lower"] <= interval["upper"] <= 1,
                    "binary difference interval is outside its support",
                )
        self.main_data = dataset, analysis
        return self.main_data

    def summary(self, row: dict) -> dict:
        """Bind an actual retained episode summary."""
        return self.sources.read(Path(row["summary_path"]))

    def configuration(self, row: dict) -> dict:
        """Read the resolved per-episode configuration and initialization."""
        return self.sources.read(Path(row["summary_path"]).parent / "binding.json")

    def checkpoint(self, dataset: dict, row: dict) -> dict:
        """Authenticate the exact initial checkpoint bytes without JAX deserialization."""
        bound = self.configuration(row)["checkpoint"]
        root = Path(__file__).resolve().parents[1]
        expected_stem = (
            root
            / dataset["binding"]["specification"]["checkpoints"][row["method"]][
                str(row["library_seed"])
            ]
        ).resolve()
        expected = {
            str((root / name).resolve()): digest
            for name, digest in dataset["binding"]["checkpoint_sha256"].items()
        }
        for name in ("checkpoint_json", "checkpoint_npz"):
            path = Path(bound[name]).resolve()
            suffix = ".json" if name.endswith("json") else ".npz"
            require(
                path == expected_stem.with_suffix(suffix),
                "trial uses another method or library seed's checkpoint",
            )
            require(str(path) in expected, "trial checkpoint is absent from campaign binding")
            require(
                self.sources.bind(path) == expected[str(path)],
                "checkpoint bytes differ from campaign binding",
            )
        document = self.sources.read(Path(bound["checkpoint_json"]))
        require(
            document["metadata"]["seed"] == row["library_seed"],
            "checkpoint library seed differs from the trial",
        )
        require(
            document["npz_sha256"]
            == bound["checkpoint_sha256"]
            == self.sources.hashes[str(Path(bound["checkpoint_npz"]).resolve())],
            "checkpoint JSON, numeric payload and trial binding differ",
        )
        return document


def style() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
        }
    )
    return plt


def save_figure(
    plt: Any, fig: Any, output: Path, name: str, rows: list[dict], caption: str
) -> dict:
    fig.savefig(output / f"{name}.png", dpi=220)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)
    write_csv(output / f"{name}.csv", rows)
    return {
        "name": name,
        "png": f"{name}.png",
        "pdf": f"{name}.pdf",
        "data": f"{name}.csv",
        "caption": caption,
    }


def short_cell(cell: str) -> str:
    return (
        cell.replace("/prescribed", "")
        .replace("structured", "Struct.")
        .replace("navigation", "Nav.")
        .replace("effectiveness", "effect.")
        .replace("/", "\n")
        .replace("_", "\n")
    )


def figure_main(e: Evidence, output: Path) -> dict:
    _, analysis = e.main()
    cells = sorted({row["cell"] for row in analysis["rows"]})
    require(
        len(cells) == 8 and {r["method"] for r in analysis["rows"]} == set(METHODS),
        "main figure requires the specified eight cells and four methods",
    )
    plt = style()
    fig, axes = plt.subplots(
        2, 2, figsize=(11.2, 7.2), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained"
    )
    rows = []
    metrics = [
        ("actual_collision", "Modeled collision", "Negative favors A"),
        ("safe_task_completion", "Safe task completion", "Positive favors A"),
    ]
    confidence = None
    for panel, (metric, label, direction) in enumerate(metrics):
        ax, forest = axes[panel]
        x = np.arange(len(cells))
        for i, method in enumerate(METHODS):
            values = []
            for cell in cells:
                matches = [
                    r
                    for r in analysis["aggregates"]
                    if (r["cell"], r["method"], r["metric"]) == (cell, method, metric)
                ]
                require(
                    len(matches) == 1 and matches[0]["complete"],
                    "main aggregate missing or ambiguous",
                )
                row = matches[0]
                values.append(row["estimate"])
                rows.append(
                    {
                        "panel": "balanced_episode_rate",
                        "cell": cell,
                        "method": method,
                        "metric": metric,
                        "estimate": row["estimate"],
                        "n_episodes": row["n_expected"],
                        "n_worlds": len(row["worlds"]),
                        "n_library_seeds": len(row["library_seeds"]),
                    }
                )
            ax.bar(x + (i - 1.5) * 0.19, values, 0.18, color=COLORS[method], label=method)
        ax.set(
            xticks=x,
            xticklabels=[short_cell(c) for c in cells],
            ylim=(0, 1.06),
            ylabel="Balanced episode proportion",
            title=f"{'AC'[panel]}. {label}",
        )
        ax.grid(axis="y", alpha=0.2)
        if panel == 0:
            ax.legend(ncol=4, loc="upper left", frameon=False)
        for i, cell in enumerate(cells):
            pairs = [
                p
                for p in analysis["paired_comparisons"]
                if p["cell"] == cell
                and p["metric"] == metric
                and p["method_a"] == "A"
                and p["method_b"] == "F2"
            ]
            require(len(pairs) == 1, "A minus F2 sealed contrast missing or ambiguous")
            pair = pairs[0]
            interval = pair["interval"]
            confidence = interval["confidence_level"]
            forest.plot(
                [interval["lower"], interval["upper"]], [i, i], color=COLORS["A"], linewidth=2
            )
            forest.scatter(pair["estimate"], i, color=COLORS["A"], s=25, zorder=3)
            rows.append(
                {
                    "panel": "sealed_paired_contrast",
                    "cell": cell,
                    "metric": metric,
                    "method": "A minus F2",
                    **interval,
                }
            )
        forest.axvline(0, color=".55", linewidth=0.8)
        forest.set(
            yticks=np.arange(len(cells)),
            yticklabels=[c.replace("/prescribed", "") for c in cells],
            xlabel=f"A − F2 proportion difference\n{direction}",
            title=f"{'BD'[panel]}. Prespecified paired contrast",
        )
        forest.invert_yaxis()
        forest.grid(axis="x", alpha=0.15)
    return save_figure(
        plt,
        fig,
        output,
        "01_main_safety",
        rows,
        "Main sealed test matrix: four worlds per cell and three reused library seeds, giving "
        "12 paired episodes per method and cell. Balanced episode proportions are shown with "
        "paired A minus F2 approximate crossed world/library percentile bootstrap intervals "
        "from the retained analysis, whose inputs are authenticated. Intervals use nominal "
        f"{confidence:.8%} individual confidence after the sealed Bonferroni correction "
        "(95% family target); no world-any-event Wilson interval is placed "
        "on an episode-rate bar. Modeled collider intersection is not measured hardware contact. "
        "Counts and all plotted numbers are in the CSV. Bootstrap samples are not regenerated "
        "by this renderer; a separately supplied analysis digest can additionally authenticate "
        "the saved interval bytes.",
    )


def load_recovery(e: Evidence) -> tuple[list[dict], list[dict]]:
    rows, censored = [], []
    bank = None
    repo = Path(__file__).resolve().parents[1]
    for seed in (11, 23, 37):
        directory = e.args.study_root / f"recovery-matrix-seed{seed}-v1" / "combined"
        file_required(directory / "summary.json", f"combined recovery seed {seed}")
        manifest = e.sources.read(directory / "manifest.json")
        require(
            manifest["arguments"]["cell"] == "combined"
            and manifest["arguments"]["revision"] == "none"
            and manifest["initial_library_version"] == 128,
            "recovery source is not the original combined experiment",
        )
        require(
            manifest["reference_sha256"] == manifest["active_reference_sha256"],
            "recovery teacher/objective changed",
        )
        bank = bank or manifest["validation_bank_sha256"]
        require(manifest["validation_bank_sha256"] == bank, "held-out recovery state banks differ")
        for relative, expected in manifest["sources"].items():
            require(
                e.sources.bind(child(directory / "source_snapshot", relative)) == expected,
                "recovery source snapshot hash mismatch",
            )
        parent = Path(manifest["checkpoint_stem"])
        parent = parent if parent.is_absolute() else repo / parent
        require(
            e.sources.bind(parent.with_suffix(".npz")) == manifest["checkpoint_npz_sha256"],
            "recovery initial checkpoint hash mismatch",
        )
        e.sources.bind(parent.with_suffix(".json"))
        dr = Path(manifest["arguments"]["dr_checkpoint"])
        dr = dr if dr.is_absolute() else repo / dr
        dr_checkpoint = e.sources.read(dr.with_suffix(".json"))
        require(
            e.sources.bind(dr.with_suffix(".npz")) == dr_checkpoint["npz_sha256"]
            and dr_checkpoint["reference_sha256"] == manifest["reference_sha256"]
            and dr_checkpoint["metadata"]["mode"] == "dr"
            and dr_checkpoint["metadata"]["seed"] == seed
            and dr_checkpoint["library_version"] == 512,
            "recovery DR baseline lacks its matched immutable teacher/seed provenance",
        )
        trace = e.sources.read(directory / "publication_trace.json")
        evaluations = e.sources.read(directory / "evaluations.json")
        summary = e.sources.read(directory / "summary.json")
        require(
            len(trace) == manifest["arguments"]["steps"] == 128,
            "recovery publication trace is incomplete",
        )
        for n, entry in enumerate(trace, 1):
            require(
                entry["attempt"] == entry["published_adaptation_updates"] == n
                and entry["finite_update_published"]
                and entry["library_version"] == 128 + n,
                "recovery publication/version trace is inconsistent",
            )
            require(
                entry["completed_after_s"]
                >= entry["started_after_s"]
                >= (trace[n - 2]["completed_after_s"] if n > 1 else 0),
                "recovery measured availability is noncausal",
            )
        observed_updates = [v["adaptation_updates"] for v in evaluations]
        require(
            len(set(observed_updates)) == len(observed_updates)
            and set(range(0, 129, 16)).issubset(observed_updates),
            "recovery common probe grid missing or duplicated",
        )
        for evaluation in evaluations:
            n = evaluation["adaptation_updates"]
            same_number(
                evaluation["available_after_seconds"],
                0 if n == 0 else trace[n - 1]["completed_after_s"],
                "recovery evaluation availability differs from actual completion",
            )
            name = "initial" if n == 0 else f"updates_{n:04d}"
            checkpoint = e.sources.read(directory / f"{name}.json")
            require(
                checkpoint["library_version"] == 128 + n
                and checkpoint["reference_sha256"] == manifest["reference_sha256"],
                "recovery checkpoint version or teacher mismatch",
            )
            require(
                e.sources.bind(directory / f"{name}.npz") == checkpoint["npz_sha256"],
                "recovery checkpoint payload mismatch",
            )
            report_name = "frozen_F2_validation" if n == 0 else f"{name}_validation"
            report = e.sources.read(directory / f"{report_name}.json")
            require(
                report == evaluation["validation"],
                "recovery evaluation differs from retained validation report",
            )
            raw_path = directory / f"{report_name}.npz"
            verify_recovery_raw(e, raw_path, report, manifest, bank)
            for metric in (*RECOVERY_METRICS, "velocity_rmse_max_mps"):
                rows.append(
                    {
                        "seed": seed,
                        "method": "A",
                        "updates": n,
                        "library_version": 128 + n,
                        "available_after_seconds": evaluation["available_after_seconds"],
                        "metric": metric,
                        "value": report[metric],
                        "common_grid": n % 16 == 0,
                        "tracking_and_braking_pass": report["tracking_and_braking_pass"],
                        "operational_pass": report["operational_pass"],
                    }
                )
        for method in ("F2", "DR"):
            report = e.sources.read(directory / f"frozen_{method}_validation.json")
            verify_recovery_raw(
                e, directory / f"frozen_{method}_validation.npz", report, manifest, bank
            )
            for metric in (*RECOVERY_METRICS, "velocity_rmse_max_mps"):
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "updates": 0,
                        "metric": metric,
                        "value": report[metric],
                        "common_grid": True,
                        "tracking_and_braking_pass": report["tracking_and_braking_pass"],
                        "operational_pass": report["operational_pass"],
                    }
                )
        for metric, (_, threshold) in RECOVERY_METRICS.items():
            passed = [v for v in evaluations if v["validation"][metric] <= threshold]
            censored.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "threshold": threshold,
                    "first_observed_passing_update": passed[0]["adaptation_updates"]
                    if passed
                    else None,
                    "right_censored_after_updates": None if passed else 128,
                    "interpolation": "none; observed probe grid only",
                }
            )
        final = next(v for v in evaluations if v["adaptation_updates"] == 128)
        require(
            summary["final_adaptive"] == final["validation"],
            "recovery summary differs from final held-out probe",
        )
    return rows, censored


def verify_recovery_raw(e: Evidence, path: Path, report: dict, manifest: dict, bank: str) -> None:
    e.sources.bind(path)
    with np.load(path, allow_pickle=False) as arrays:
        actual_bank = hashlib.sha256(
            np.ascontiguousarray(arrays["initial_states"]).tobytes()
        ).hexdigest()
        require(actual_bank == bank, "recovery NPZ contains a different validation bank")
        actual, reference = arrays["states"], arrays["reference_states"]
        require(
            actual.shape == reference.shape == (16, 16, 61, 17)
            and np.isfinite(actual).all()
            and np.isfinite(reference).all(),
            "recovery full-state trajectories are malformed or nonfinite",
        )
        config = manifest["current_actor_config"]
        nodes = sorted(
            {
                max(1, round(config["horizon"] * f))
                for f in manifest["reference_learning_config"]["trajectory_fractions"]
            }
        )
        for metric, selection in (
            ("position_rmse_max_m", slice(0, 3)),
            ("velocity_rmse_max_mps", slice(7, 10)),
        ):
            error = actual[:, :, nodes, selection] - reference[:, :, nodes, selection]
            same_number(
                report[metric],
                float(np.sqrt(np.mean(error**2, axis=(-1, -2))).max()),
                f"recovery {metric} differs from raw trajectories",
            )
        same_number(
            report["terminal_speed_max_mps"],
            float(np.linalg.norm(actual[:, :, -1, 7:10], axis=-1).max()),
            "recovery braking metric differs from raw trajectories",
        )
        require(
            report["tracking_and_braking_pass"]
            == (
                report["finite_valid_fraction"] == 1
                and report["position_rmse_max_m"] <= 0.3
                and report["velocity_rmse_max_mps"] <= 0.6
                and report["terminal_speed_max_mps"] <= 0.8
            ),
            "recovery threshold outcome differs from the declared metrics",
        )


def figure_recovery(e: Evidence, output: Path) -> dict:
    rows, censored = load_recovery(e)
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), layout="constrained")
    data = list(rows)
    for ax, (metric, (label, threshold)) in zip(axes, RECOVERY_METRICS.items(), strict=True):
        updates = list(range(0, 129, 16))
        values = np.array(
            [
                [
                    next(
                        r["value"]
                        for r in rows
                        if r["method"] == "A"
                        and r["metric"] == metric
                        and r["seed"] == seed
                        and r["updates"] == n
                    )
                    for n in updates
                ]
                for seed in (11, 23, 37)
            ]
        )
        for values_seed in values:
            ax.plot(updates, values_seed, color=COLORS["A"], alpha=0.22, linewidth=0.8)
        ax.fill_between(updates, values.min(0), values.max(0), color=COLORS["A"], alpha=0.18)
        ax.plot(
            updates,
            values.mean(0),
            color=COLORS["A"],
            linewidth=2,
            marker="o",
            markersize=3,
            label="A: mean + min–max, 3 seeds",
        )
        for n, mean, low, high in zip(
            updates, values.mean(0), values.min(0), values.max(0), strict=True
        ):
            data.append(
                {
                    "method": "A",
                    "metric": metric,
                    "updates": n,
                    "statistic": "three_seed_summary",
                    "value": mean,
                    "minimum": low,
                    "maximum": high,
                }
            )
        for method in ("F2", "DR"):
            baseline = [r["value"] for r in rows if r["method"] == method and r["metric"] == metric]
            ax.axhspan(min(baseline), max(baseline), color=COLORS[method], alpha=0.12)
            ax.axhline(
                np.mean(baseline),
                color=COLORS[method],
                linestyle="--",
                linewidth=1.4,
                label=f"{method}: frozen, 3 seeds",
            )
            data.append(
                {
                    "method": method,
                    "metric": metric,
                    "statistic": "three_seed_summary",
                    "value": float(np.mean(baseline)),
                    "minimum": min(baseline),
                    "maximum": max(baseline),
                }
            )
        ax.axhline(
            threshold, color=".25", linestyle=":", linewidth=1.2, label=f"Threshold: {threshold:g}"
        )
        unmet = sum(
            c["right_censored_after_updates"] is not None for c in censored if c["metric"] == metric
        )
        ax.set(
            xlabel="Completed adaptation updates (fixed state bank)",
            ylabel=label,
            xlim=(-2, 133),
            title=f"Combined fault · {unmet}/3 seeds censored at 128 updates",
        )
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=7.5)
    write_csv(output / "02_recovery_censoring.csv", censored)
    return save_figure(
        plt,
        fig,
        output,
        "02_fixed_bank_recovery",
        data,
        "Obstacle-free combined-fault recovery on the same held-out bank of 16 physical states "
        "and 16 skills. Curves report the maximum over states and skills, with arithmetic mean "
        "and min–max across only three library seeds; the bands are not confidence intervals. "
        "Frozen F2 and DR use the same bank. Unmet thresholds are right-censored after 128 "
        "updates, never set to zero. A full tracking/braking pass also requires velocity RMSE "
        "≤ 0.6 m/s. The x-axis is completed fixed-bank updates, not physical flight time; measured "
        "isolated update availability is retained in the CSV. See "
        "[censoring data](02_recovery_censoring.csv) for observed crossings without interpolation.",
    )


def timing_rows(e: Evidence, dataset: dict, role: str) -> list[dict]:
    output = []
    for row in dataset["rows"]:
        summary = e.summary(row)
        wall, exposure = summary.get("execution_wall_seconds"), summary.get("physical_time_seconds")
        require(
            isinstance(wall, (int, float))
            and wall > 0
            and isinstance(exposure, (int, float))
            and exposure > 0,
            "complete execution wall time or physical exposure missing",
        )
        output.append(
            {
                "role": role,
                "cell": row["cell"],
                "physical_world_id": row["physical_world_id"],
                "library_seed": row["library_seed"],
                "method": row["method"],
                "execution_wall_seconds": wall,
                "physical_time_seconds": exposure,
                "execution_wall_per_physical_second": wall / exposure,
                "controller_mean_seconds": summary["controller_seconds"]["mean"],
                "learner_mean_seconds": summary.get("learner_seconds", {}).get("mean"),
                "simulator_and_audit_seconds": summary.get("simulator_and_audit_seconds"),
                "diagnostic_callback_seconds": summary.get("diagnostic_callback_seconds"),
                "control_count": summary["control_count"],
                "termination": summary.get("termination"),
                **row["metrics"],
            }
        )
    return output


def figure_compute(e: Evidence, output: Path) -> dict:
    dataset, _ = e.main()
    require(
        all(
            e.configuration(r)["config"]["execution_mode"] == "deterministic"
            for r in dataset["rows"]
        ),
        "main execution-cost figure requires deterministic execution",
    )
    rows = timing_rows(e, dataset, "main")
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.3), layout="constrained")
    cells = sorted({r["cell"] for r in rows})
    plotted = list(rows)
    for mi, method in enumerate(METHODS):
        for ci, cell in enumerate(cells):
            group = [r for r in rows if r["method"] == method and r["cell"] == cell]
            cost = np.mean([r["execution_wall_per_physical_second"] for r in group])
            collision = np.mean([r["actual_collision"] for r in group])
            axes[0].scatter(
                mi + (ci - (len(cells) - 1) / 2) * 0.065,
                cost,
                color=COLORS[method],
                marker="o" if "structured" in cell else "^",
                s=35,
            )
            axes[1].scatter(
                cost,
                collision,
                color=COLORS[method],
                marker="o" if "structured" in cell else "^",
                s=35,
                label=method if ci == 0 else None,
            )
            plotted.append(
                {
                    "statistic": "balanced_cell_mean",
                    "cell": cell,
                    "method": method,
                    "execution_wall_per_physical_second": float(cost),
                    "actual_collision": float(collision),
                    "n_episodes": len(group),
                }
            )
    axes[0].set(
        xticks=np.arange(len(METHODS)),
        xticklabels=METHODS,
        ylabel="Execution wall seconds / physical second",
        title="A. Full execution cost · one point per cell",
    )
    axes[1].set(
        xlabel="Execution wall seconds / physical second",
        ylabel="Modeled collision proportion",
        ylim=(-0.04, 1.04),
        title="B. Safety and measured execution cost",
    )
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.18)
    return save_figure(
        plt,
        fig,
        output,
        "03_safety_and_execution_cost",
        plotted,
        "Actual main-campaign execution wall time divided by executed physical exposure, first "
        "calculated per episode and then equally averaged within each cell/method. Wall time "
        "includes controller, adaptive learner when run, simulation/audit, diagnostics and loop "
        "overhead; it excludes compilation warmup and final artifact serialization. Early "
        "collision termination contributes only its executed prefix. Circles denote structured "
        "worlds and triangles navigation worlds. This deterministic execution diagnostic is "
        "not a real-time guarantee and does not substitute controller-only timing for A’s "
        "complete cost.",
    )


def paired_secondary(
    e: Evidence, role: str, baseline_role: str, allowed: set[str], methods: set[str] | None = None
) -> list[dict]:
    alternative, baseline = e.role(role), e.role(baseline_role)
    require(
        alternative["binding"]["source_sha256"] == baseline["binding"]["source_sha256"],
        f"{role} source differs from {baseline_role}; not a one-factor comparison",
    )
    selected = methods or set(alternative["binding"]["specification"]["methods"])
    base = {key(r): r for r in baseline["rows"] if r["method"] in selected}
    alt = {key(r): r for r in alternative["rows"] if r["method"] in selected}
    require(
        base.keys() == alt.keys() and bool(alt),
        f"{role}/{baseline_role} physical world, library-seed or method matrix differs",
    )
    output = []
    for identity, row in alt.items():
        ref = base[identity]
        config_a, config_b = e.configuration(row), e.configuration(ref)
        a, b = flat(config_a["config"]), flat(config_b["config"])
        differences = {name for name in a.keys() | b.keys() if a.get(name) != b.get(name)}
        require(
            differences == allowed,
            f"{role} has missing or undeclared episode-config differences: {differences ^ allowed}",
        )
        factor_values = {
            "fine_p0": ("plant_step_seconds", 0.005, 0.002),
            "p1": ("plant_level", "P0", "P1"),
            "p2": ("plant_level", "P0", "P2"),
            "effectiveness_bias": ("observation_config.effectiveness_bias", 0.0, 0.1),
            "lag_scale": ("observation_config.lag_scale", 1.0, 1.2),
            "parameter_delay": ("observation_config.parameter_delay_seconds", 0.0, 0.08),
            "motor_noise": ("observation_config.motor_noise_N", 0.0, 0.002),
            "position_noise": ("observation_config.position_noise_m", 0.0, 0.01),
            "obstacle_bias": ("observation_config.obstacle_position_bias_m", 0.0, 0.03),
        }
        if role in factor_values:
            field, old, new = factor_values[role]
            require(a[field] == new and b[field] == old, f"{role} has a different factor magnitude")
        for field in (
            "initial_learner_sha256",
            "initial_optimizer_sha256",
            "initial_params_sha256",
            "initial_library_version",
            "initial_state_sha256",
            "initial_state",
        ):
            require(config_a[field] == config_b[field], f"{role} changes initial {field}")
        require(
            config_a["scene"]["physical_spec"] == config_b["scene"]["physical_spec"],
            f"{role} physical scene differs",
        )
        checkpoint_a, checkpoint_b = e.checkpoint(alternative, row), e.checkpoint(baseline, ref)
        normalized_a, normalized_b = (
            normalized_checkpoint(checkpoint_a),
            normalized_checkpoint(checkpoint_b),
        )
        require(
            normalized_a["config"] == normalized_b["config"], f"{role} changes actor configuration"
        )
        if role != "retention_off":
            require(checkpoint_a == checkpoint_b, f"{role} checkpoint differs")
        else:
            off, on = flat(normalized_a), flat(normalized_b)
            changed = {name for name in off.keys() | on.keys() if off.get(name) != on.get(name)}
            require(
                all(
                    name in {"reference_sha256", "reference_learning_config.retention_weight"}
                    or name.startswith("metadata.ablation_provenance.")
                    for name in changed
                )
                and checkpoint_a["reference_learning_config"]["retention_weight"] == 0
                and checkpoint_b["reference_learning_config"]["retention_weight"] == 5
                and checkpoint_a["reference_sha256"] != checkpoint_b["reference_sha256"],
                "retention-off checkpoint is not an exact declared retention 5 to 0 continuation",
            )
            provenance = checkpoint_a["metadata"]["ablation_provenance"]
            require(
                provenance["parent_npz_sha256"] == checkpoint_b["npz_sha256"]
                and provenance["parent_reference_sha256"] == checkpoint_b["reference_sha256"]
                and provenance["training_updates_performed"] == 0,
                "retention-off parent provenance differs from the oracle actor",
            )
        for metric in (
            "actual_collision",
            "safe_task_completion",
            "collider_clearance_lower_m",
            "controller_deadline_miss_rate",
        ):
            output.append(
                {
                    "role": role,
                    "baseline_role": baseline_role,
                    "physical_world_id": identity[0],
                    "library_seed": identity[1],
                    "method": identity[2],
                    "cell": row["cell"],
                    "metric": metric,
                    "value": row["metrics"][metric],
                    "baseline_value": ref["metrics"][metric],
                    "difference": row["metrics"][metric] - ref["metrics"][metric],
                    "actual_config_differences": sorted(differences),
                    "inference": (
                        "descriptive paired physical world; no crossed confidence interval"
                    ),
                }
            )
    return output


def normalized_checkpoint(document: dict) -> dict:
    # Explicit serialization defaults introduced after the original checkpoint are
    # semantically identical; only their documented default values may be omitted.
    value = json.loads(json.dumps(document))
    for field in ("config", "reference_actor_config"):
        if value[field].get("rollout_scan_unroll", 1) == 1:
            value[field].pop("rollout_scan_unroll", None)
        if value[field].get("allow_reference_gain_mismatch", False) is False:
            value[field].pop("allow_reference_gain_mismatch", None)
    if value["reference_learning_config"].get("reference_braking_huber_delta", 0) == 0:
        value["reference_learning_config"].pop("reference_braking_huber_delta", None)
    return value


def load_k_profile(e: Evidence) -> list[dict]:
    directory = e.args.k_profile
    if directory is None:
        raise MissingEvidence("No actual K-cost profile supplied")
    file_required(directory / "summary.json", "completed K-cost profile")
    summary = e.sources.read(directory / "summary.json")
    binding = e.sources.read(directory / "binding.json")
    require(
        summary["status"] == "completed_cost_profile" and summary["proposals_published"] == 0,
        "K profile is incomplete or published learning proposals",
    )
    require(all(summary.get(k) == v for k, v in binding.items()), "K summary differs from binding")
    require(
        summary["counts"] == [4, 8, 16, 32] and summary["source_policy_count"] == 16,
        "K profile counts differ from specified ablation",
    )
    for relative, expected in binding["source_sha256"].items():
        require(
            e.sources.bind(child(directory / "source_snapshot", relative)) == expected,
            "K profile source hash mismatch",
        )
    initializations = e.sources.read(directory / "initializations.json")
    require(initializations == summary["initializations"], "K initialization provenance differs")
    for count in summary["counts"]:
        if count != 16:
            require(
                initializations[str(count)]["changed_k_competence"] == "unavailable",
                "changed-K competence unexpectedly relabeled",
            )
        e.sources.bind(directory / f"K{count}_cost_inputs.npz")
    observations = e.sources.read(directory / "observations.json")
    require(len(observations) == 4, "K profile requires four fixed observations")
    trace_path = directory / "timing_trace.jsonl"
    e.sources.bind(trace_path)
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    identities = {(r["K"], r["observation"], r["component"], r["repetition"]) for r in rows}
    expected = {
        (k, obs, component, repetition)
        for k in summary["counts"]
        for obs in summary["timings"][str(k)]
        for component in ("controller", "learner")
        for repetition in range(binding["repetitions"])
    }
    require(
        identities == expected and len(rows) == len(identities),
        "K timing trace is incomplete or duplicated",
    )
    for count in summary["counts"]:
        for observation, components in summary["timings"][str(count)].items():
            e.sources.bind(directory / f"observation_{observation}.npz")
            for component, stats in components.items():
                values = [
                    r["seconds"]
                    for r in rows
                    if (r["K"], r["observation"], r["component"]) == (count, observation, component)
                ]
                require(all(math.isfinite(v) and v > 0 for v in values), "invalid K timing sample")
                same_number(
                    stats["mean_seconds"], np.mean(values), "K mean does not reproduce from trace"
                )
                same_number(
                    stats["p95_seconds"],
                    np.quantile(values, 0.95),
                    "K p95 does not reproduce from trace",
                )
    return rows


def figure_ablations(e: Evidence, output: Path) -> dict:
    a1 = e.role("a1")
    require(
        set(a1["binding"]["specification"]["methods"]) == {"F2", "A", "A1"},
        "A1 validation campaign must contain A1, F2 and A",
    )
    retention = paired_secondary(e, "retention_off", "oracle", set(), {"A"})
    cost = load_k_profile(e)
    rows = []
    for row in a1["rows"]:
        require(row["library_seed"] == 11, "A1 validation subset must use library seed 11")
        document = e.checkpoint(a1, row)
        require(
            document["metadata"]["mode"] == ("single" if row["method"] == "A1" else "nominal")
            and document["library_version"] == 128,
            "A1 subset lacks the specified independently trained initial checkpoint",
        )
        latent = document["structure"]["items"]["reference"]["items"]["spec"]["items"][
            "latent_codes"
        ]["key"]
        require(
            document["arrays"][latent]["shape"] == [1 if row["method"] == "A1" else 16, 8],
            "A1 subset checkpoint has a different library size",
        )
        reports = e.configuration(row)["checkpoint"]["competence_reports"]
        require(
            "final_validation.json" in reports
            and reports["final_validation.json"]["report"]["competent_under_declared_criteria"],
            "A1 subset initial actor has no retained passing validation evidence",
        )
        for metric in ("safe_task_completion", "collider_clearance_lower_m", "actual_collision"):
            rows.append(
                {
                    "panel": "A1_validation",
                    "method": row["method"],
                    "cell": row["cell"],
                    "physical_world_id": row["physical_world_id"],
                    "library_seed": row["library_seed"],
                    "metric": metric,
                    "value": row["metrics"][metric],
                }
            )
    for row in retention:
        rows.append({"panel": "retention_validation", **row})
    plt = style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    for i, metric in enumerate(("safe_task_completion", "collider_clearance_lower_m")):
        labels = ["F2", "A", "A1", "A\nret. 5", "A\nret. 0"]
        for x, method in enumerate(("F2", "A", "A1")):
            subset = [
                r
                for r in rows
                if r["panel"] == "A1_validation" and r["method"] == method and r["metric"] == metric
            ]
            require(len(subset) == 2, "A1 subset requires two validation worlds")
            for j, row in enumerate(sorted(subset, key=lambda r: r["physical_world_id"])):
                axes[i].scatter(
                    x + (j - 0.5) * 0.12,
                    row["value"],
                    color=COLORS[method],
                    marker="o" if "structured" in row["cell"] else "^",
                    s=43,
                )
        subset = [r for r in retention if r["metric"] == metric]
        require(len(subset) == 2, "retention subset requires two validation worlds")
        for j, row in enumerate(sorted(subset, key=lambda r: r["physical_world_id"])):
            axes[i].plot(
                [3 + (j - 0.5) * 0.12, 4 + (j - 0.5) * 0.12],
                [row["baseline_value"], row["value"]],
                color=COLORS["A"],
                alpha=0.4,
            )
            axes[i].scatter(
                [3 + (j - 0.5) * 0.12, 4 + (j - 0.5) * 0.12],
                [row["baseline_value"], row["value"]],
                color=COLORS["A"],
                marker="o" if "structured" in row["cell"] else "^",
                s=43,
            )
        axes[i].axvline(2.5, color=".75", linewidth=0.7)
        axes[i].set(
            xticks=range(5),
            xticklabels=labels,
            ylabel="Safe task completion indicator"
            if i == 0
            else "Minimum collider clearance lower bound (m)",
            title=f"{'AB'[i]}. Two validation worlds · seed 11",
        )
        if i == 0:
            axes[i].set_ylim(-0.06, 1.06)
        else:
            axes[i].axhline(0, color=".5", linestyle=":", linewidth=0.8)
        axes[i].grid(axis="y", alpha=0.18)
    for component, color in (("controller", COLORS["F2"]), ("learner", COLORS["A"])):
        stats = []
        for count in (4, 8, 16, 32):
            by_observation = [
                [
                    r["seconds"] * 1000
                    for r in cost
                    if r["K"] == count and r["component"] == component and r["observation"] == obs
                ]
                for obs in sorted({r["observation"] for r in cost})
            ]
            means = [np.mean(v) for v in by_observation]
            stats.append((np.mean(means), min(means), max(means)))
            rows.append(
                {
                    "panel": "K_cost_summary",
                    "K": count,
                    "component": component,
                    "mean_ms": float(np.mean(means)),
                    "minimum_observation_mean_ms": float(min(means)),
                    "maximum_observation_mean_ms": float(max(means)),
                    "changed_k_competence": "unavailable"
                    if count != 16
                    else "source checkpoint only",
                }
            )
        values = np.asarray(stats)
        axes[2].fill_between([4, 8, 16, 32], values[:, 1], values[:, 2], color=color, alpha=0.15)
        axes[2].plot(
            [4, 8, 16, 32], values[:, 0], color=color, marker="o", label=component.capitalize()
        )
    axes[2].set(
        xticks=[4, 8, 16, 32],
        xlabel="Library size K",
        ylabel="Isolated warm service (ms)",
        title="C. Cost only · changed-K competence unavailable",
    )
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.18)
    write_csv(output / "04_k_cost_samples.csv", cost)
    return save_figure(
        plt,
        fig,
        output,
        "04_library_and_retention_ablations",
        rows,
        "A1 uses its separately prepared competent single-recovery checkpoint. Retention-off "
        "preserves the original K=16 actor and persistent optimizer while changing only the "
        "declared retention objective; points are paired by physical world. Each validation "
        "condition contains two worlds and one library seed, so results are descriptive, with "
        "no crossed confidence intervals. Circles are structured and triangles navigation "
        "worlds. K costs are synchronized isolated services, averaged equally over four fixed "
        "observations; shading spans the four observation means, not uncertainty. Changed K "
        "also changes initialization/optimizer or policy subset and possibly controller "
        "branches; competence is unavailable and no pure causal K effect or serialized "
        "real-time budget is claimed. [All measured cost samples](04_k_cost_samples.csv).",
    )


def figure_mismatch(e: Evidence, output: Path) -> dict:
    conditions = (
        ("fine_p0", "oracle", {"plant_step_seconds"}, "P0 Δt=.002"),
        ("p1", "fine_p0", {"plant_level"}, "P1 Δt=.002"),
        ("p2", "fine_p0", {"plant_level"}, "P2 Δt=.002"),
        ("effectiveness_bias", "oracle", {"observation_config.effectiveness_bias"}, "η̂ = η + .10"),
        ("lag_scale", "oracle", {"observation_config.lag_scale"}, "τ̂ = 1.2 τ"),
        (
            "parameter_delay",
            "oracle",
            {"observation_config.parameter_delay_seconds"},
            "Parameter age .08 s",
        ),
        ("motor_noise", "oracle", {"observation_config.motor_noise_N"}, "Motor noise .002 N"),
    )
    supplements = (
        (
            "position_noise",
            "oracle",
            {"observation_config.position_noise_m"},
            "Position noise σ=.01 m / axis",
        ),
        (
            "obstacle_bias",
            "oracle",
            {"observation_config.obstacle_position_bias_m"},
            "Obstacle bias +.03 m / axis",
        ),
    )
    conditions += tuple(item for item in supplements if item[0] in e.roles)
    rows = []
    for role, baseline, allowed, _ in conditions:
        rows.extend(paired_secondary(e, role, baseline, allowed))
    plt = style()
    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.6 + 0.35 * (len(conditions) - 7)), layout="constrained"
    )
    methods = [m for m in METHODS if m in {r["method"] for r in rows}]
    for ax, metric, title in zip(
        axes,
        ("safe_task_completion", "collider_clearance_lower_m"),
        ("A. Safe completion difference", "B. Collider clearance difference"),
        strict=True,
    ):
        for i, (role, _, _, _) in enumerate(conditions):
            for mi, method in enumerate(methods):
                subset = [
                    r
                    for r in rows
                    if r["role"] == role and r["method"] == method and r["metric"] == metric
                ]
                require(
                    len(subset) == 2 and {r["library_seed"] for r in subset} == {11},
                    "mismatch figure requires exactly two validation worlds and seed 11 per method",
                )
                for row in subset:
                    ax.scatter(
                        row["difference"],
                        i + (mi - (len(methods) - 1) / 2) * 0.17,
                        color=COLORS[method],
                        marker="o" if "structured" in row["cell"] else "^",
                        s=30,
                        label=method if i == 0 and row is subset[0] else None,
                    )
        ax.axvline(0, color=".5", linewidth=0.8)
        ax.set(
            yticks=np.arange(len(conditions)),
            yticklabels=[c[3] for c in conditions],
            xlabel="Alternative − paired baseline"
            + (" (m)" if metric.endswith("_m") else " (indicator)"),
            title=title,
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)
    axes[0].legend(frameon=False, loc="best")
    return save_figure(
        plt,
        fig,
        output,
        "05_one_factor_mismatch",
        rows,
        "One-factor secondary validation differences on the same two physical worlds and "
        "library seed 11. P0 step .002 s is compared with the oracle P0 step .005 s; P1 and P2 "
        "at .002 s are compared with fine P0 to isolate plant model. Observation perturbations "
        "compare with the oracle baseline. Each point is one physical-world pair, with no "
        "crossed interval or population claim. Circles are structured and triangles navigation "
        "worlds. Positive values favor the alternative for both displayed metrics. Pacing and "
        "command delay are analyzed separately in figure 06 when explicitly supplied. "
        + (
            "Position noise is zero-mean Gaussian with .01 m standard deviation per coordinate. "
            if "position_noise" in e.roles
            else ""
        )
        + (
            "Obstacle bias adds (.03,.03,.03) m to observed/predicted centers, "
            "a .05196 m vector norm; physical obstacle paths stay unchanged. "
            if "obstacle_bias" in e.roles
            else ""
        ),
    )


def figure_timing(e: Evidence, output: Path) -> dict:
    # Both are explicit inputs: selecting this optional figure never guesses a timing campaign.
    rows = []
    for role in ("paced", "delayed"):
        rows.extend(timing_rows(e, e.role(role), role))
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4), layout="constrained")
    for ax, role in zip(axes, ("paced", "delayed"), strict=True):
        subset = [r for r in rows if r["role"] == role]
        methods = [m for m in (*METHODS, "A1") if m in {r["method"] for r in subset}]
        for i, method in enumerate(methods):
            group = [r for r in subset if r["method"] == method]
            for j, row in enumerate(group):
                ax.scatter(
                    i + (j - (len(group) - 1) / 2) * 0.08,
                    row["controller_deadline_miss_rate"],
                    color=COLORS[method],
                    marker="o" if "structured" in row["cell"] else "^",
                    s=35,
                )
        ax.set(
            xticks=range(len(methods)),
            xticklabels=methods,
            ylim=(-0.04, 1.04),
            ylabel="Controller deadline misses / attempted calls",
            title=f"{'A' if role == 'paced' else 'B'}. {role.capitalize()} execution",
        )
        ax.grid(axis="y", alpha=0.18)
    return save_figure(
        plt,
        fig,
        output,
        "06_paced_and_delayed_timing",
        rows,
        "Paced and delayed execution are distinct descriptive panels. Each point is an actual "
        "completed episode; controller deadline rates use attempted calls as the denominator "
        "and do not certify the full controller-plus-learner schedule. Full execution "
        "wall/physical exposure and learner/simulation accounting are retained in the CSV. "
        "Timing may be affected by workload and hardware; these results are not pooled into "
        "deterministic main inference.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--main-campaign", type=Path)
    parser.add_argument("--sealed-analysis", type=Path)
    parser.add_argument(
        "--analysis-sha256", help="Optional independently retained digest of analysis JSON"
    )
    parser.add_argument("--secondary-map", type=Path)
    parser.add_argument("--k-profile", type=Path)
    parser.add_argument("--only", nargs="+", choices=FIGURES, default=list(FIGURES[:-1]))
    parser.add_argument("--missing", choices=("refuse", "skip"), default="refuse")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Matplotlib only; never initialize a GPU or import the executable experiment runner.
    os.environ.setdefault("MPLBACKEND", "Agg")
    require(len(args.only) == len(set(args.only)), "duplicate requested figures")
    args.study_root = args.study_root.resolve()
    evidence = Evidence(args)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source_hash = evidence.sources.bind(Path(__file__))
    renderers = {
        "main": figure_main,
        "recovery": figure_recovery,
        "compute": figure_compute,
        "ablations": figure_ablations,
        "mismatch": figure_mismatch,
        "timing": figure_timing,
    }
    figures, missing = [], []
    try:
        for name in args.only:
            try:
                figures.append(renderers[name](evidence, args.output))
                print(json.dumps({"figure": name, "status": "completed"}), flush=True)
            except MissingEvidence as error:
                missing.append({"figure": name, "reason": str(error)})
                if args.missing == "refuse":
                    raise
    except Exception as error:
        write_json(
            args.output / "FAILED.json",
            {
                "error_type": type(error).__name__,
                "reason": str(error),
                "completed_figures": figures,
                "missing": missing,
                "input_sha256": evidence.sources.hashes,
            },
        )
        raise
    require(bool(figures) or bool(missing), "no figure work was requested")
    report = [
        "# Actuator study figures",
        "",
        "Figures use completed retained data only. Relative links below point to PNG, PDF "
        "and exact plotted CSV data.",
        "",
    ]
    for figure in figures:
        report += [
            f"## {figure['name']}",
            "",
            f"![{figure['name']}]({figure['png']})",
            "",
            f"[PDF]({figure['pdf']}) · [Plot data]({figure['data']})",
            "",
            figure["caption"],
            "",
        ]
    if missing:
        report += (
            ["## Omitted figures", ""]
            + [f"- {item['figure']}: {item['reason']}" for item in missing]
            + [""]
        )
    report += [
        "All consumed inputs and emitted figure/data files are SHA-256 bound in manifest.json. "
        "Missing inputs can omit a figure only with --missing skip; corrupt or contradictory "
        "supplied evidence is always fatal.",
        "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(report))
    outputs = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.iterdir())
        if p.is_file()
    }
    write_json(
        args.output / "manifest.json",
        {
            "schema": "da_plcbf_actuator_publication_figures_v1",
            "status": "completed_with_omissions" if missing else "completed",
            "figure_source_sha256": source_hash,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "figures": figures,
            "missing": missing,
            "input_sha256": evidence.sources.hashes,
            "output_sha256": outputs,
            "inference_policy": (
                "main reuses saved intervals from analysis with authenticated sealed inputs; "
                "secondary and three-seed recovery are descriptive; no simulation or "
                "resampling by renderer"
            ),
        },
    )


if __name__ == "__main__":
    main()
