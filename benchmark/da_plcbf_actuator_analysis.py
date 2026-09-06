"""Analyze retained actuator campaigns or sealed ledgers without inventing missing outcomes.

Examples (run with the repository environment)::

    JAX_PLATFORMS=cpu python benchmark/da_plcbf_actuator_analysis.py campaign \
        --input artifacts/example-campaign --compare A:F2 --output artifacts/example-analysis
    JAX_PLATFORMS=cpu python benchmark/da_plcbf_actuator_analysis.py ledger \
        --protocol protocol.json --ledger ledger.json --output artifacts/sealed-analysis
    JAX_PLATFORMS=cpu python benchmark/da_plcbf_actuator_analysis.py figures \
        --analysis artifacts/example-analysis/analysis.json --output artifacts/example-figures

Campaign comparisons are exploratory. Only the ledger route labels comparisons as
prespecified and uses the sealed multiplicity policy. A complete crossed matrix is
required for every aggregate and interval; incomplete attempts remain in source data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from crazyflow.safety.da_plcbf.actuator_protocol import (
    TrialLedger,
    content_sha256,
    crossed_paired_bootstrap,
    load_sealed_protocol,
    paired_protocol_bootstrap,
    world_binary_wilson,
)

SCHEMA = "da_plcbf_actuator_analysis_v1"
METRICS = {
    "actual_collision": {
        "kind": "binary",
        "units": "indicator",
        "definition": (
            "Modeled XML collider intersection on the executed physical trajectory; "
            "not measured MuJoCo contact"
        ),
    },
    "safe_task_completion": {
        "kind": "binary",
        "units": "indicator",
        "definition": (
            "All waypoints complete, no modeled collider collision and all operational "
            "nodes pass through full declared duration"
        ),
    },
    "operational_violation": {
        "kind": "binary",
        "units": "indicator",
        "definition": (
            "Any audited operational constraint below -1e-7 on the executed physical prefix"
        ),
    },
    "collider_clearance_lower_m": {
        "kind": "continuous",
        "units": "m",
        "definition": (
            "Minimum interpolation-error-adjusted modeled collider clearance lower bound "
            "to obstacles or ground on executed prefix"
        ),
    },
    "enclosure_clearance_lower_m": {
        "kind": "continuous",
        "units": "m",
        "definition": (
            "Minimum body-origin enclosure clearance lower bound to obstacles on executed "
            "prefix; distinct from collider contact"
        ),
    },
    "controller_deadline_any": {
        "kind": "binary",
        "units": "indicator",
        "definition": "At least one complete controller service exceeds the command period",
    },
    "controller_deadline_miss_rate": {
        "kind": "continuous",
        "units": "fraction",
        "definition": (
            "Controller deadline misses divided by attempted control calls within each "
            "episode; equal episode weighting"
        ),
    },
    "controller_p95_seconds": {
        "kind": "continuous",
        "units": "s",
        "definition": (
            "Per-episode p95 complete controller service; aggregate is the mean of episode "
            "p95 values, not a pooled p95"
        ),
    },
    "controller_mean_seconds": {
        "kind": "continuous",
        "units": "s",
        "definition": "Per-episode mean complete controller service; equal episode weighting",
    },
}


def _finite(value: Any) -> float | None:
    if isinstance(value, (bool, int, float)) and math.isfinite(value):
        return float(value)
    return None


@dataclass
class Sources:
    """Bind each input actually read, without requiring mutable original checkpoints."""

    hashes: dict[str, str] = field(default_factory=dict)

    def bind(self, path: Path) -> str:
        """Retain a binary file's exact digest without interpreting it."""
        path = path.resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.hashes[str(path)] = digest
        return digest

    def read(self, path: Path) -> Any:
        """Read strict finite JSON and retain its content digest."""
        path = path.resolve()
        data = path.read_bytes()
        self.hashes[str(path)] = hashlib.sha256(data).hexdigest()
        value = json.loads(data)
        _validate_finite_json(value)
        return value


def _validate_finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("input JSON contains a nonfinite number")
    if isinstance(value, dict):
        for item in value.values():
            _validate_finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite_json(item)


def _local_path(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("artifact path escapes its campaign directory")
    return path


def _episode_metrics(summary: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    # Completed collision-terminated episodes are valid outcomes; interruption is never safe.
    if summary.get("status") != "completed":
        return {metric: None for metric in METRICS}
    collision = summary.get("modeled_collider_collision")
    operational = summary.get("actual_operational_all_nodes_pass")
    controls = summary.get("control_count", 0)
    misses = summary.get("controller_deadline_misses")
    geometry = audit.get("audit", audit)
    clearance = [
        _finite(geometry.get(name, {}).get("minimum_clearance_lower_bound_m"))
        for name in ("actual_xml_sphere_geometry", "actual_xml_ground_geometry")
    ]
    # No-obstacle sphere clearance is null; ground remains a meaningful finite bound.
    clearance = [value for value in clearance if value is not None]
    return {
        "actual_collision": int(collision) if isinstance(collision, bool) else None,
        "safe_task_completion": _finite(summary.get("successful_full_episode")),
        "operational_violation": int(not operational) if isinstance(operational, bool) else None,
        "collider_clearance_lower_m": min(clearance) if clearance else None,
        "enclosure_clearance_lower_m": _finite(
            geometry.get("body_origin_envelope", {}).get("minimum_clearance_lower_bound_m")
        ),
        "controller_deadline_any": int(misses > 0)
        if _finite(misses) is not None and controls > 0
        else None,
        "controller_deadline_miss_rate": misses / controls
        if _finite(misses) is not None and controls > 0
        else None,
        "controller_p95_seconds": _finite(summary.get("controller_seconds", {}).get("p95")),
        "controller_mean_seconds": _finite(summary.get("controller_seconds", {}).get("mean")),
    }


def read_campaign(directory: Path, sources: Sources) -> dict[str, Any]:
    """Authenticate the campaign envelope and retain all available attempt summaries."""
    directory = directory.resolve()
    binding = sources.read(directory / "campaign_binding.json")
    digest = binding.get("sha256")
    if content_sha256({key: value for key, value in binding.items() if key != "sha256"}) != digest:
        raise ValueError(f"campaign binding hash mismatch: {directory}")
    result = sources.read(directory / "campaign_result.json")
    if result.get("campaign_sha256") != digest:
        raise ValueError("campaign result does not reference its authenticated binding")
    for relative, expected in binding["source_sha256"].items():
        snapshot = _local_path(directory, "source/" + relative)
        if sources.bind(snapshot) != expected:
            raise ValueError("retained campaign source snapshot hash mismatch")
    spec = binding["specification"]
    methods, seeds = spec["methods"], spec["library_seeds"]
    if (
        not methods
        or not seeds
        or len(set(methods)) != len(methods)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("campaign methods and library seeds must be nonempty and unique")
    planned = len(spec["worlds"]) * len(seeds) * len(methods)
    if result.get("planned") != planned:
        raise ValueError("campaign planned count differs from the bound matrix")
    index = {}
    for record in result["records"]:
        key = (record["world_index"], record["library_seed"], record["method"])
        if key in index or not (
            0 <= key[0] < len(spec["worlds"]) and key[1] in seeds and key[2] in methods
        ):
            raise ValueError("duplicate or out-of-support campaign result")
        summary_path = _local_path(directory, record["summary_path"])
        if sources.read(summary_path) != record["summary"]:
            raise ValueError("embedded campaign result differs from its retained summary")
        index[key] = record
    if result.get("completed") != sum(
        row["summary"].get("status") == "completed" for row in index.values()
    ):
        raise ValueError("campaign completed count differs from its records")
    rows, attempts, groups = [], [], {}
    world_identity = {}
    for world_index, world in enumerate(spec["worlds"]):
        cell = world.get("cell") or "/".join(
            str(world.get(key, default))
            for key, default in (
                ("family", "unspecified"),
                ("dynamics_cell", "unspecified"),
                ("obstacle_mode", "prescribed"),
            )
        )
        for seed, method in itertools.product(seeds, methods):
            key = (world_index, seed, method)
            record = index.get(key)
            summary = record["summary"] if record else {}
            trial = f"world-{world_index:04d}-seed-{seed}-{method}"
            if record and record["trial"] != trial:
                raise ValueError("campaign trial name differs from its planned identity")
            attempt_paths = []
            for attempt_directory in sorted((directory / trial).glob("attempt-*")):
                path = attempt_directory / "summary.json"
                if path.exists():
                    attempt_paths.append(path)
                else:
                    attempts.append(
                        {
                            "dataset": digest,
                            "trial": trial,
                            "path": str(attempt_directory),
                            "status": "attempt_without_summary",
                            "reason": "Retained attempt directory has no terminal summary",
                        }
                    )
            completed = []
            selected = None
            for path in attempt_paths:
                attempt = sources.read(path)
                attempts.append(
                    {"dataset": digest, "trial": trial, "path": str(path), "summary": attempt}
                )
                if attempt.get("status") == "completed":
                    completed.append(path)
                if record and path.resolve() == _local_path(directory, record["summary_path"]):
                    selected = path
            if len(completed) > 1 or (completed and (selected is None or completed[0] != selected)):
                raise ValueError("campaign does not select its unique completed retained attempt")
            if record and selected is None:
                raise ValueError("selected result is absent from the retained attempt directory")
            config = dict(spec["episode_config"])
            config.update(world.get("episode_config", {}))
            config.update(spec.get("method_config", {}).get(method, {}))
            config.setdefault("execution_mode", "deterministic")
            config.setdefault("plant_level", "P0")
            physical_id = summary.get("physical_world_id")
            if physical_id is not None:
                previous = world_identity.setdefault(world_index, physical_id)
                if previous != physical_id:
                    raise ValueError("paired methods/seeds do not share the same physical world")
            audit = {}
            if selected is not None:
                trial_binding = sources.read(selected.parent / "binding.json")
                if trial_binding["config"]["method"] != method:
                    raise ValueError("trial binding method differs from its campaign identity")
                if trial_binding["scene"].get("physical_world_id") != physical_id:
                    raise ValueError("trial summary physical identity differs from its binding")
                if summary.get("status") == "completed" and not physical_id:
                    raise ValueError("completed trial has no physical world identity")
                for name, expected in trial_binding.get("source_sha256", {}).items():
                    matches = [
                        value
                        for relative, value in binding["source_sha256"].items()
                        if Path(relative).name == name
                    ]
                    if len(matches) != 1 or expected != matches[0]:
                        raise ValueError("trial source differs from the campaign source binding")
                audit_path = selected.parent / "collision_audit.json"
                if audit_path.exists():
                    audit = sources.read(audit_path)
            duration = summary.get(
                "duration_seconds", world.get("world_config", {}).get("duration_seconds")
            )
            group_key = content_sha256(
                {
                    "dataset": digest,
                    "cell": cell,
                    "plant_level": config["plant_level"],
                    "execution_mode": config["execution_mode"],
                }
            )
            group = groups.setdefault(
                group_key,
                {
                    "group": group_key,
                    "dataset": digest,
                    "cell": cell,
                    "plant_level": config["plant_level"],
                    "execution_mode": config["execution_mode"],
                    "duration_seconds": duration,
                    "worlds": [],
                    "library_seeds": list(seeds),
                    "methods": [],
                },
            )
            if duration is not None:
                if group["duration_seconds"] not in (None, duration):
                    raise ValueError("unequal declared durations require distinct explicit cells")
                group["duration_seconds"] = duration
            if world_index not in group["worlds"]:
                group["worlds"].append(world_index)
            if method not in group["methods"]:
                group["methods"].append(method)
            rows.append(
                {
                    "dataset": digest,
                    "group": group_key,
                    "cell": cell,
                    "world": world_index,
                    "physical_world_id": physical_id,
                    "library_seed": seed,
                    "method": method,
                    "status": summary.get("status", "missing"),
                    "termination": summary.get("termination"),
                    "summary_path": str(selected) if selected else None,
                    "metrics": _episode_metrics(summary, audit),
                }
            )
    identities = list(world_identity.values())
    if len(set(identities)) != len(identities):
        raise ValueError(
            "duplicate physical worlds in one campaign cannot count as independent clusters"
        )
    return {
        "id": digest,
        "source": str(directory),
        "purpose": spec["purpose"],
        "inference": "exploratory; campaign binding alone is not a sealed analysis protocol",
        "planned_trials": planned,
        "completed_trials": result["completed"],
        "rows": rows,
        "attempts": attempts,
        "groups": list(groups.values()),
    }


def _matrix(
    rows: list[dict[str, Any]], group: dict[str, Any], method: str, metric: str
) -> np.ndarray:
    index = {
        (row["world"], row["library_seed"]): row
        for row in rows
        if row["group"] == group["group"] and row["method"] == method
    }
    values = np.full((len(group["worlds"]), len(group["library_seeds"])), np.nan)
    for i, world in enumerate(group["worlds"]):
        for j, seed in enumerate(group["library_seeds"]):
            row = index.get((world, seed), {})
            value = _finite(row.get("metrics", {}).get(metric))
            if row.get("status") == "completed" and value is not None:
                values[i, j] = value
    return values


def aggregate_groups(
    rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    metrics: dict[str, Any],
    confidence: float,
) -> list[dict[str, Any]]:
    """Compute balanced episode means and separately labeled world-any-event diagnostics."""
    output = []
    for group in groups:
        for method, metric in itertools.product(group["methods"], metrics):
            values = _matrix(rows, group, method, metric)
            complete = bool(np.isfinite(values).all())
            row = {
                **group,
                "method": method,
                "metric": metric,
                "complete": complete,
                "n_expected": values.size,
                "n_observed": int(np.isfinite(values).sum()),
                "estimate": None,
                "per_seed_means": None,
                "world_any_event_wilson": None,
            }
            if complete:
                if metrics[metric]["kind"] == "binary" and not np.all(
                    (values == 0) | (values == 1)
                ):
                    raise ValueError(f"binary metric {metric} has values outside zero/one")
                row["estimate"] = float(values.mean())
                row["per_seed_means"] = values.mean(axis=0).tolist()
                if metrics[metric]["kind"] == "binary":
                    row["world_any_event_wilson"] = world_binary_wilson(
                        values, confidence_level=confidence
                    )
            else:
                row["reason"] = (
                    "incomplete crossed matrix; no complete-case deletion or zero imputation"
                )
            output.append(row)
    return output


def _pair(
    rows: list[dict[str, Any]],
    group: dict[str, Any],
    method_a: str,
    method_b: str,
    metric: str,
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    a, b = (_matrix(rows, group, method, metric) for method in (method_a, method_b))
    result = {
        **group,
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "effect_direction": "method_a_minus_method_b",
        "estimate": None,
        "interval": None,
    }
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        result["reason"] = "incomplete paired matrix; interval refused"
    else:
        result["estimate"] = float((a - b).mean())
        if min(a.shape) < 2:
            result["reason"] = (
                "two independent worlds and two library seeds required for crossed interval"
            )
        else:
            result["interval"] = asdict(
                crossed_paired_bootstrap(
                    a, b, confidence_level=confidence, n_resamples=n_resamples, resampling_seed=seed
                )
            )
    return result


def analyze_campaigns(
    directories: list[Path],
    *,
    comparisons: list[str],
    confidence: float,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Analyze campaigns separately; repeated campaigns never inflate world counts."""
    sources = Sources()
    datasets = [read_campaign(directory, sources) for directory in directories]
    if len({dataset["id"] for dataset in datasets}) != len(datasets):
        raise ValueError("duplicate campaign bindings cannot be treated as independent evidence")
    rows = [row for dataset in datasets for row in dataset["rows"]]
    groups = [group for dataset in datasets for group in dataset["groups"]]
    pairs = []
    for group in groups:
        requested = (
            [tuple(value.split(":")) for value in comparisons]
            if comparisons
            else list(itertools.combinations(group["methods"], 2))
        )
        for pair in requested:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError("comparisons must name distinct methods as A:F2")
            if not set(pair).issubset(group["methods"]):
                continue
            for metric in METRICS:
                row = _pair(
                    rows,
                    group,
                    *pair,
                    metric,
                    confidence=confidence,
                    n_resamples=n_resamples,
                    seed=seed,
                )
                row["inference"] = (
                    "exploratory unadjusted descriptive interval; no confirmatory claim"
                )
                pairs.append(row)
    return {
        "schema": SCHEMA,
        "input_kind": "campaign",
        "datasets": [
            {
                key: value
                for key, value in dataset.items()
                if key not in {"rows", "groups", "attempts"}
            }
            for dataset in datasets
        ],
        "metrics": METRICS,
        "groups": groups,
        "rows": rows,
        "attempts": [attempt for dataset in datasets for attempt in dataset["attempts"]],
        "aggregates": aggregate_groups(rows, groups, METRICS, confidence),
        "paired_comparisons": pairs,
        "settings": {
            "confidence_level": confidence,
            "n_resamples": n_resamples,
            "resampling_seed": seed,
            "comparisons": comparisons or "all unordered pairs within each campaign/stratum",
            "multiplicity": "exploratory_descriptive_only",
        },
        "input_sha256": sources.hashes,
    }


def analyze_ledger(protocol_path: Path, ledger_path: Path) -> dict[str, Any]:
    """Honor the immutable protocol's metric, cell, bootstrap, and multiplicity choices."""
    sources = Sources()
    sources.read(protocol_path)
    protocol = load_sealed_protocol(protocol_path)
    ledger = TrialLedger.from_mapping(protocol, sources.read(ledger_path))
    manifest = protocol.manifest
    spec = manifest["splits"]["test"]
    methods = list(manifest["methods"])
    groups = []
    for cell in sorted({world["cell"] for world in spec["worlds"]}):
        groups.append(
            {
                "group": content_sha256({"protocol": protocol.sha256, "cell": cell}),
                "dataset": protocol.sha256,
                "cell": cell,
                "worlds": [world["world_id"] for world in spec["worlds"] if world["cell"] == cell],
                "library_seeds": spec["library_seeds"],
                "methods": methods,
            }
        )
    attempts = ledger.as_mapping()["records"]
    completed = {
        (row["world_id"], row["library_seed"], row["method"]): row
        for row in attempts
        if row["split"] == "test" and row["status"] == "completed"
    }
    rows = []
    for group in groups:
        for world, seed, method in itertools.product(
            group["worlds"], group["library_seeds"], methods
        ):
            record = completed.get((world, seed, method), {})
            rows.append(
                {
                    "dataset": protocol.sha256,
                    "group": group["group"],
                    "cell": group["cell"],
                    "world": world,
                    "physical_world_id": world,
                    "library_seed": seed,
                    "method": method,
                    "status": record.get("status", "missing_or_incomplete"),
                    "metrics": record.get("metrics", {}),
                }
            )
    settings = manifest["analysis"]
    pairs = []
    for comparison_id, comparison in settings["primary_comparisons"].items():
        for cell, metric in itertools.product(comparison["cells"], comparison["metrics"]):
            group = next(group for group in groups if group["cell"] == cell)
            pair = {
                **group,
                "comparison_id": comparison_id,
                "method_a": comparison["method_a"],
                "method_b": comparison["method_b"],
                "metric": metric,
                "effect_direction": "method_a_minus_method_b",
                "inference": "prespecified primary; sealed Bonferroni multiplicity",
                "estimate": None,
                "interval": None,
            }
            try:
                interval = paired_protocol_bootstrap(ledger, comparison_id, metric, cell=cell)
                pair.update(estimate=interval.estimate, interval=asdict(interval))
            except ValueError as error:
                pair["reason"] = str(error)
            pairs.append(pair)
    return {
        "schema": SCHEMA,
        "input_kind": "sealed_ledger",
        "protocol_sha256": protocol.sha256,
        "coverage": ledger.coverage(),
        "metrics": manifest["outcomes"],
        "settings": settings,
        "groups": groups,
        "rows": rows,
        "attempts": attempts,
        "aggregates": aggregate_groups(
            rows, groups, manifest["outcomes"], settings["confidence_level"]
        ),
        "paired_comparisons": pairs,
        "input_sha256": sources.hashes,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def write_analysis(output: Path, analysis: dict[str, Any]) -> None:
    """Write exclusive JSON/CSV source data and record executable figure commands."""
    output.mkdir(parents=True, exist_ok=False)
    analysis["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    analysis["qualifications"] = [
        "Episode rates require a complete world by library-seed matrix; "
        "W*S is never an independent Wilson denominator.",
        "World-any-event Wilson intervals change the estimand and condition on tested libraries.",
        "Crossed percentile intervals are approximate; few library seeds weaken coverage. "
        "Per-seed effects are retained.",
        "Clearance and operational violation refer to the executed prefix; "
        "early collision stops later exposure.",
        "Deterministic update mode supplies measured service diagnostics "
        "but makes no real-time execution claim.",
        "This script never selects campaigns, methods, cells, or metrics by observed performance.",
    ]
    with (output / "analysis.json").open("x") as stream:
        json.dump(analysis, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    _write_csv(
        output / "episodes.csv",
        [
            {**{key: value for key, value in row.items() if key != "metrics"}, **row["metrics"]}
            for row in analysis["rows"]
        ],
    )
    _write_csv(output / "attempts.csv", analysis["attempts"])
    _write_csv(output / "aggregate_source.csv", analysis["aggregates"])
    _write_csv(output / "paired_source.csv", analysis["paired_comparisons"])
    with (output / "figure_command.json").open("x") as stream:
        json.dump(
            {
                "environment": {"JAX_PLATFORMS": "cpu"},
                "argv": [
                    "python",
                    str(Path(__file__).resolve()),
                    "figures",
                    "--analysis",
                    str((output / "analysis.json").resolve()),
                    "--output",
                    str((output / "figures").resolve()),
                ],
            },
            stream,
            indent=2,
        )
        stream.write("\n")


def make_figures(analysis_path: Path, output: Path) -> dict[str, Any]:
    """Render only supplied results, with exact figure rows saved next to every panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = Sources()
    analysis = sources.read(analysis_path)
    if analysis.get("schema") != SCHEMA:
        raise ValueError("unsupported analysis schema")
    output.mkdir(parents=True, exist_ok=False)
    produced, skipped = [], []
    metadata = {group["group"]: group for group in analysis["groups"]}

    def label(row: dict[str, Any]) -> str:
        group = metadata[row["group"]]
        extra = " / ".join(
            str(group[key]) for key in ("plant_level", "execution_mode") if key in group
        )
        return f"{row['cell']} / {extra} / {row['dataset'][:8]}".replace(" /  / ", " / ")

    def save(fig: Any, name: str, rows: list[dict[str, Any]]) -> None:
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)
        _write_csv(output / f"{name}_source.csv", rows)
        produced.append(name)

    aggregates = [row for row in analysis["aggregates"] if row["complete"]]
    aliases = [
        name
        for name in (
            "actual_collision",
            "modeled_collision",
            "safe_task_completion",
            "operational_violation",
        )
        if name in analysis["metrics"]
    ]
    data = [row for row in aggregates if row["metric"] in aliases]
    if data:
        fig, axes = plt.subplots(
            1,
            len(aliases),
            figsize=(6 * len(aliases), max(4, len(analysis["groups"]) * 1.2)),
            squeeze=False,
        )
        for axis, metric in zip(axes[0], aliases, strict=True):
            selected = [row for row in data if row["metric"] == metric]
            for i, row in enumerate(selected):
                axis.plot(row["estimate"], i, "o", color="C0")
            axis.set(
                yticks=range(len(selected)),
                yticklabels=[f"{row['method']} | {label(row)}" for row in selected],
                xlim=(-0.03, 1.03),
                xlabel="Balanced episode fraction (descriptive)",
                title=metric.replace("_", " "),
            )
            axis.grid(axis="x", alpha=0.25)
        save(fig, "main_safety_task", data)
    else:
        skipped.append(
            {"figure": "main_safety_task", "reason": "no complete safety/task outcome matrices"}
        )
    data = [
        row
        for row in aggregates
        if row.get("world_any_event_wilson") is not None and row["metric"] in aliases
    ]
    if data:
        fig, axis = plt.subplots(figsize=(10, max(4, 0.38 * len(data))))
        for i, row in enumerate(data):
            interval = row["world_any_event_wilson"]
            axis.errorbar(
                interval["estimate"],
                i,
                xerr=[
                    [interval["estimate"] - interval["lower"]],
                    [interval["upper"] - interval["estimate"]],
                ],
                fmt="o",
                color="C1",
            )
        axis.set(
            yticks=range(len(data)),
            yticklabels=[f"{row['method']} {row['metric']} | {label(row)}" for row in data],
            xlim=(-0.03, 1.03),
            xlabel=(
                "World with any event across tested libraries: "
                "Wilson interval (denominator = worlds)"
            ),
            title="Conditional on tested libraries; event means metric = 1",
        )
        save(fig, "world_any_event", data)

    def forest(name: str, selected: list[dict[str, Any]]) -> None:
        if not selected:
            skipped.append(
                {
                    "figure": name,
                    "reason": "no complete paired intervals for requested methods/metrics",
                }
            )
            return
        metrics = list(dict.fromkeys(row["metric"] for row in selected))
        fig, axes = plt.subplots(
            len(metrics),
            1,
            figsize=(
                11,
                max(
                    3,
                    sum(
                        max(2, sum(row["metric"] == metric for row in selected) * 0.38)
                        for metric in metrics
                    ),
                ),
            ),
            squeeze=False,
        )
        for axis, metric in zip(axes[:, 0], metrics, strict=True):
            rows = [row for row in selected if row["metric"] == metric]
            for i, row in enumerate(rows):
                interval = row["interval"]
                axis.plot([interval["lower"], interval["upper"]], [i, i], color="C0")
                axis.plot(row["estimate"], i, "o", color="C0")
            axis.axvline(0, color="0.6", linewidth=0.8)
            axis.set(
                yticks=range(len(rows)),
                yticklabels=[
                    f"{row['method_a']} − {row['method_b']} | {label(row)}" for row in rows
                ],
                xlabel=f"{metric} difference ({analysis['metrics'][metric]['units']})",
                title="Paired crossed world × library-seed percentile intervals",
            )
        save(fig, name, selected)

    pairs = [row for row in analysis["paired_comparisons"] if row.get("interval") is not None]
    forest("paired_safety_task", [row for row in pairs if row["metric"] in aliases])
    forest(
        "ablation",
        [
            row
            for row in pairs
            if row["metric"] in aliases
            and {row["method_a"], row["method_b"]}.intersection({"F0", "F1", "A1"})
        ],
    )
    forest(
        "mismatch",
        [
            row
            for row in pairs
            if row["metric"] in aliases
            and metadata[row["group"]].get("plant_level") in {"P1", "P2"}
        ],
    )
    index = {(row["group"], row["method"], row["metric"]): row for row in aggregates}
    tradeoff = []
    for row in aggregates:
        if row["metric"] != "safe_task_completion":
            continue
        timing = index.get((row["group"], row["method"], "controller_p95_seconds"))
        if timing:
            tradeoff.append({**row, "controller_p95_seconds": timing["estimate"]})
    if tradeoff:
        fig, axis = plt.subplots(figsize=(10, 5))
        for row in tradeoff:
            axis.plot(1000 * row["controller_p95_seconds"], row["estimate"], "o")
            axis.annotate(
                f"{row['method']} | {label(row)}",
                (1000 * row["controller_p95_seconds"], row["estimate"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        axis.set(
            xlabel="Mean across episode p95 complete controller service (ms)",
            ylabel="Safe task completion: balanced episode fraction",
            ylim=(-0.03, 1.03),
            title="Computation versus observed safe task completion; timing mode shown per point",
        )
        axis.grid(alpha=0.25)
        save(fig, "safety_computation", tradeoff)
    else:
        skipped.append(
            {"figure": "safety_computation", "reason": "no complete paired timing/task aggregates"}
        )
    manifest = {
        "schema": "da_plcbf_actuator_figures_v1",
        "analysis_sha256": sources.hashes,
        "produced": produced,
        "skipped": skipped,
        "qualifications": analysis["qualifications"],
        "inference": analysis["input_kind"],
    }
    with (output / "figure_manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    campaign = commands.add_parser(
        "campaign", help="Exploratory analysis of bound campaign result directories"
    )
    campaign.add_argument("--input", type=Path, action="append", required=True)
    campaign.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Ordered method pair A:F2; default all unordered pairs",
    )
    campaign.add_argument("--confidence", type=float, default=0.95)
    campaign.add_argument("--resamples", type=int, default=10000)
    campaign.add_argument("--seed", type=int, default=0)
    ledger = commands.add_parser(
        "ledger", help="Prespecified primary analysis from a sealed protocol and retained ledger"
    )
    ledger.add_argument("--protocol", type=Path, required=True)
    ledger.add_argument("--ledger", type=Path, required=True)
    figures = commands.add_parser(
        "figures", help="Render only outcome-backed figures and exact source CSVs"
    )
    figures.add_argument("--analysis", type=Path, required=True)
    for command in (campaign, ledger, figures):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "figures":
        print(json.dumps(make_figures(args.analysis, args.output), indent=2))
        return
    if args.command == "campaign":
        if not 0 < args.confidence < 1 or args.resamples < 1000 or args.seed < 0:
            parser.error("confidence must be in (0,1), resamples >=1000, and seed >=0")
        analysis = analyze_campaigns(
            args.input,
            comparisons=args.compare,
            confidence=args.confidence,
            n_resamples=args.resamples,
            seed=args.seed,
        )
    else:
        analysis = analyze_ledger(args.protocol, args.ledger)
    write_analysis(args.output, analysis)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "input_kind": analysis["input_kind"],
                "episodes": len(analysis["rows"]),
                "retained_attempts": len(analysis["attempts"]),
                "paired_intervals": sum(
                    row["interval"] is not None for row in analysis["paired_comparisons"]
                ),
                "refused_intervals": sum(
                    row["interval"] is None for row in analysis["paired_comparisons"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
