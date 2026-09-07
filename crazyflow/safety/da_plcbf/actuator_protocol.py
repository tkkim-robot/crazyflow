"""Frozen actuator-study manifests, retained attempts, and crossed paired analysis.

This module supplies no world generator, tuning callback, or default test budget.
The caller must finish development/validation and resolve all settings before
sealing. Physical world identity hashes the complete physical specification,
independently of split labels, method names, and library seeds. See
``docs/da_plcbf_actuator_theory.md`` for statistical and theoretical limitations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike, NDArray


PROTOCOL_SCHEMA = "da_plcbf_actuator_protocol_v1"
SPLITS = ("development", "validation", "test")
ATTEMPT_STATUSES = ("completed", "interrupted", "simulator_error", "censored", "inadmissible")
_PLACEHOLDER = re.compile(r"\b(TBD|TODO|TBA|PLACEHOLDER)\b", re.IGNORECASE)
_PERSISTENT_FAULT_RECOVERY = re.compile(
    r"^manifest\.splits\.(development|validation|test)\.worlds\[\d+\]"
    r"\.physical_spec\.actuator_events\[\d+\]\.recovery_time_seconds$"
)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not permit nonfinite numbers")
        return int(value) if value.is_integer() else value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("canonical JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise ValueError(
        f"unsupported JSON value type {type(value).__name__}; resolve it before sealing"
    )


def canonical_json(value: Any) -> str:
    """Serialize finite plain JSON deterministically, normalizing 1.0/1 and -0.0/0."""
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_sha256(value: Any) -> str:
    """Hash canonical JSON content, independent of dictionary insertion order."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a nonempty JSON object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _resolved(value: Any, path: str = "manifest") -> None:
    # ActuatorScene uses this exact nullable physical field to mean a persistent
    # fault. Preserve it verbatim so sealing cannot change physical world identity.
    if value is None and _PERSISTENT_FAULT_RECOVERY.fullmatch(path):
        return
    if value is None or (
        isinstance(value, str) and (not value.strip() or _PLACEHOLDER.search(value))
    ):
        raise ValueError(f"{path} contains an unresolved placeholder")
    if isinstance(value, dict):
        for key, item in value.items():
            _resolved(item, f"{path}.{key}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _resolved(item, f"{path}[{index}]")


def physical_world_id(physical_spec: Mapping[str, Any]) -> str:
    """Identify physical content; callers must include all initial/model/event geometry.

    Administrative labels at the top level are rejected, rather than included in
    a hash that would let identical physics masquerade as independent worlds.
    Generation seeds are provenance in the split manifest, not physical identity.
    This generic utility cannot establish that a supplied physical spec is complete.
    """
    physical = _object(dict(physical_spec), "physical_spec")
    forbidden = {
        "split",
        "method",
        "library_seed",
        "world_id",
        "attempt_id",
        "trial_id",
        "generation_seed",
        "world_seed",
        "seed",
        "cell",
    }
    if forbidden.intersection(physical):
        raise ValueError("physical_spec must omit administrative identity/seed fields")
    return "world-" + content_sha256(physical)


def validate_protocol(manifest: Mapping[str, Any]) -> None:
    """Validate resolved settings, content identity, disjoint splits, and analysis rules."""
    value = json.loads(canonical_json(dict(manifest)))
    _resolved(value)
    required = {
        "schema",
        "protocol_id",
        "source",
        "models",
        "control",
        "numeric_ranges",
        "methods",
        "training_support",
        "splits",
        "outcomes",
        "analysis",
        "budget_rationale",
    }
    if not required.issubset(value) or value["schema"] != PROTOCOL_SCHEMA:
        raise ValueError("manifest has missing fields or an unsupported protocol schema")
    _text(value["protocol_id"], "protocol_id")
    _text(value["budget_rationale"], "budget_rationale")
    source = _object(value["source"], "source")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit", ""))):
        raise ValueError("source.commit must be a full 40-character Git commit")
    files = _object(source.get("files_sha256"), "source.files_sha256")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(digest)) for digest in files.values()):
        raise ValueError("source file hashes must be SHA-256 hex strings")
    for name, model in _object(value["models"], "models").items():
        model = _object(model, f"models.{name}")
        _text(model.get("definition"), f"models.{name}.definition")
        _object(model.get("parameters"), f"models.{name}.parameters")
    control = _object(value["control"], "control")
    clocks = {
        key: _number(control.get(key), key)
        for key in ("command_period_s", "prediction_horizon_s", "integration_step_s")
    }
    if (
        not 0
        < clocks["integration_step_s"]
        <= clocks["command_period_s"]
        <= clocks["prediction_horizon_s"]
    ):
        raise ValueError("clocks must satisfy 0 < integration step <= command period <= horizon")
    for name, bounds in _object(value["numeric_ranges"], "numeric_ranges").items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"numeric range {name} must be [lower, upper]")
        if _number(bounds[0], name) > _number(bounds[1], name):
            raise ValueError(f"numeric range {name} has reversed bounds")
    methods = _object(value["methods"], "methods")
    for name, method in methods.items():
        method = _object(method, f"methods.{name}")
        _text(method.get("definition"), f"methods.{name}.definition")
        _object(method.get("configuration"), f"methods.{name}.configuration")
    support = _object(value["training_support"], "training_support")
    if set(support) != set(methods):
        raise ValueError(
            "training_support must explicitly cover every method, including no-training methods"
        )
    for name, item in support.items():
        _object(item, f"training_support.{name}")
    splits = _object(value["splits"], "splits")
    if set(splits) != set(SPLITS):
        raise ValueError("exactly development, validation, and test splits are required")
    used_worlds, used_generator_seeds = set(), set()
    for split in SPLITS:
        entry = _object(splits[split], split)
        for name in ("world_seeds", "library_seeds"):
            seeds = entry.get(name)
            if not isinstance(seeds, list) or not seeds:
                raise ValueError(f"{split}.{name} must be a nonempty list")
            for seed in seeds:
                _integer(seed, name)
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"{split}.{name} contains duplicate seeds")
        if used_generator_seeds.intersection(entry["world_seeds"]):
            raise ValueError("development/validation/test generation seeds must be disjoint")
        used_generator_seeds.update(entry["world_seeds"])
        worlds = entry.get("worlds")
        if not isinstance(worlds, list) or not worlds:
            raise ValueError(f"{split}.worlds must contain the resolved physical worlds")
        for world in worlds:
            world = _object(world, "world")
            _text(world.get("cell"), "world.cell")
            identity = physical_world_id(_object(world.get("physical_spec"), "physical_spec"))
            if world.get("world_id") != identity:
                raise ValueError("world_id does not match its physical specification")
            if identity in used_worlds:
                raise ValueError("physical worlds must be distinct within and across all splits")
            used_worlds.add(identity)
    outcomes = _object(value["outcomes"], "outcomes")
    for metric, definition in outcomes.items():
        definition = _object(definition, f"outcomes.{metric}")
        _text(definition.get("definition"), f"outcomes.{metric}.definition")
        _text(definition.get("units"), f"outcomes.{metric}.units")
        if definition.get("kind") not in ("binary", "continuous"):
            raise ValueError("outcome kind must be binary or continuous")
    analysis = _object(value["analysis"], "analysis")
    confidence = _number(analysis.get("confidence_level"), "confidence_level")
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    _integer(analysis.get("n_resamples"), "n_resamples", 1000)
    _integer(analysis.get("resampling_seed"), "resampling_seed")
    if (
        analysis.get("cluster_axes") != ["world", "library_seed"]
        or analysis.get("interval") != "percentile"
    ):
        raise ValueError(
            "primary analysis must use crossed world/library-seed percentile resampling"
        )
    if analysis.get("primary_multiplicity") != "bonferroni":
        raise ValueError(
            "primary_multiplicity must explicitly use bonferroni (also valid for one test)"
        )
    if analysis.get("secondary_multiplicity") not in ("bonferroni", "descriptive_only"):
        raise ValueError("secondary multiplicity policy must be explicit")
    if analysis.get("missing_results") != "refuse_incomplete_matrix":
        raise ValueError("missing paired cells cannot be silently excluded")
    test_cells = {world["cell"] for world in splits["test"]["worlds"]}
    for name, comparison in _object(
        analysis.get("primary_comparisons"), "primary_comparisons"
    ).items():
        comparison = _object(comparison, name)
        if comparison.get("method_a") not in methods or comparison.get("method_b") not in methods:
            raise ValueError("primary comparison references an unknown method")
        if comparison["method_a"] == comparison["method_b"]:
            raise ValueError("primary comparison must pair different methods")
        for key, permitted in (("metrics", set(outcomes)), ("cells", test_cells)):
            requested = comparison.get(key)
            if (
                not isinstance(requested, list)
                or not requested
                or len(set(requested)) != len(requested)
            ):
                raise ValueError(f"comparison {key} must be a nonempty list without duplicates")
            if not set(requested).issubset(permitted):
                raise ValueError(f"comparison references undeclared {key}")


@dataclass(frozen=True, slots=True)
class FrozenProtocol:
    """Immutable canonical content; ``manifest`` returns a fresh detached object."""

    canonical: str
    sha256: str

    def __post_init__(self) -> None:
        """Reject noncanonical, invalid, or incorrectly bound direct construction."""
        manifest = json.loads(self.canonical)
        validate_protocol(manifest)
        if canonical_json(manifest) != self.canonical or content_sha256(manifest) != self.sha256:
            raise ValueError("frozen protocol canonical content/hash mismatch")

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self.canonical)

    def require_section(self, name: str, current_configuration: Any) -> None:
        """Refuse changed runtime settings rather than updating a sealed definition."""
        if name not in self.manifest or canonical_json(self.manifest[name]) != canonical_json(
            current_configuration
        ):
            raise ValueError(f"runtime section {name} differs from the frozen protocol")


def freeze_protocol(manifest: Mapping[str, Any]) -> FrozenProtocol:
    """Validate and freeze in memory; this does not publish or seal an experiment file."""
    canonical = canonical_json(dict(manifest))
    return FrozenProtocol(canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest())


def seal_protocol(path: str | Path, manifest: Mapping[str, Any] | FrozenProtocol) -> FrozenProtocol:
    """Create a sealed manifest exclusively; an existing path always raises FileExistsError."""
    frozen = manifest if isinstance(manifest, FrozenProtocol) else freeze_protocol(manifest)
    envelope = {
        "schema": "sealed_" + PROTOCOL_SCHEMA,
        "sha256": frozen.sha256,
        "manifest": frozen.manifest,
    }
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(envelope) + "\n")
    return frozen


def load_sealed_protocol(path: str | Path) -> FrozenProtocol:
    """Read, validate, and authenticate a sealed manifest without changing it."""
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        set(envelope) != {"schema", "sha256", "manifest"}
        or envelope["schema"] != "sealed_" + PROTOCOL_SCHEMA
    ):
        raise ValueError("invalid sealed protocol envelope")
    frozen = freeze_protocol(envelope["manifest"])
    if frozen.sha256 != envelope["sha256"]:
        raise ValueError("sealed protocol hash mismatch")
    return frozen


class TrialLedger:
    """Retain every attempt; only the first completed result can occupy a planned trial."""

    def __init__(self, protocol: FrozenProtocol) -> None:
        self._protocol = protocol
        self._records: list[str] = []
        self._attempt_ids: set[str] = set()
        self._completed: dict[tuple[str, str, int, str], dict[str, Any]] = {}

    @property
    def protocol(self) -> FrozenProtocol:
        return self._protocol

    def record_attempt(
        self,
        *,
        attempt_id: str,
        split: str,
        world_id: str,
        library_seed: int,
        method: str,
        status: str,
        metrics: Mapping[str, Any] | None = None,
        reason: str = "",
    ) -> None:
        """Append resolved evidence without treating an interrupted/censored attempt as complete."""
        manifest = self.protocol.manifest
        _text(attempt_id, "attempt_id")
        _integer(library_seed, "library_seed")
        if attempt_id in self._attempt_ids:
            raise ValueError("attempt_id already exists; retained attempts cannot be overwritten")
        if split not in SPLITS or method not in manifest["methods"]:
            raise ValueError("unknown split or method")
        split_spec = manifest["splits"][split]
        if library_seed not in split_spec["library_seeds"] or world_id not in {
            world["world_id"] for world in split_spec["worlds"]
        }:
            raise ValueError("trial does not belong to the frozen split/seed/world support")
        if status not in ATTEMPT_STATUSES:
            raise ValueError("unknown attempt status")
        key = (split, world_id, library_seed, method)
        if key in self._completed:
            raise ValueError("a completed trial cannot be rerun/replaced inside this protocol")
        if status != "completed":
            _text(reason, "incomplete attempt reason")
        if status == "inadmissible" and any(item[1] == world_id for item in self._completed):
            raise ValueError("world screening cannot reject a scene after comparative outcomes")
        payload = json.loads(canonical_json(dict(metrics or {})))
        required = {
            metric
            for comparison in manifest["analysis"]["primary_comparisons"].values()
            for metric in comparison["metrics"]
        }
        if status == "completed" and not required.issubset(payload):
            raise ValueError("completed trial lacks primary outcome metrics")
        for metric, result in payload.items():
            if metric in manifest["outcomes"]:
                if isinstance(result, bool) and manifest["outcomes"][metric]["kind"] == "binary":
                    result = int(result)
                number = _number(result, metric)
                if manifest["outcomes"][metric]["kind"] == "binary" and number not in (0, 1):
                    raise ValueError("binary outcome values must be zero or one")
        record = {
            "trial_id": "trial-"
            + content_sha256({"protocol_sha256": self.protocol.sha256, "key": key}),
            "attempt_id": attempt_id,
            "split": split,
            "world_id": world_id,
            "library_seed": library_seed,
            "method": method,
            "status": status,
            "metrics": payload,
            "reason": reason,
        }
        self._records.append(canonical_json(record))
        self._attempt_ids.add(attempt_id)
        if status == "completed":
            self._completed[key] = record

    def coverage(self, split: str = "test") -> dict[str, Any]:
        """Count planned trials separately from completed trials and retained attempts."""
        manifest = self.protocol.manifest
        if split not in SPLITS:
            raise ValueError("unknown split")
        spec = manifest["splits"][split]
        planned = len(spec["worlds"]) * len(spec["library_seeds"]) * len(manifest["methods"])
        records = [
            json.loads(record) for record in self._records if json.loads(record)["split"] == split
        ]
        completed = sum(key[0] == split for key in self._completed)
        return {
            "planned_trials": planned,
            "completed_trials": completed,
            "missing_trials": planned - completed,
            "attempts": len(records),
            "incomplete_attempts": sum(record["status"] != "completed" for record in records),
            "by_status": {
                status: sum(record["status"] == status for record in records)
                for status in ATTEMPT_STATUSES
            },
        }

    def paired_metric(
        self, method_a: str, method_b: str, metric: str, *, cell: str, split: str = "test"
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Require every paired world/seed cell; never silently select complete cases."""
        manifest = self.protocol.manifest
        if split not in SPLITS or any(
            method not in manifest["methods"] for method in (method_a, method_b)
        ):
            raise ValueError("unknown split or method")
        if metric not in manifest["outcomes"]:
            raise ValueError("metric was not defined in the frozen protocol")
        spec = manifest["splits"][split]
        worlds = [world["world_id"] for world in spec["worlds"] if world["cell"] == cell]
        if not worlds:
            raise ValueError("cell has no declared worlds")
        output = []
        for method in (method_a, method_b):
            values = np.empty((len(worlds), len(spec["library_seeds"])), dtype=float)
            for i, world in enumerate(worlds):
                for j, seed in enumerate(spec["library_seeds"]):
                    record = self._completed.get((split, world, seed, method))
                    if record is None or metric not in record["metrics"]:
                        raise ValueError(
                            "incomplete paired matrix; missing/censored trials remain in ledger"
                        )
                    values[i, j] = record["metrics"][metric]
            output.append(values)
        return output[0], output[1]

    def as_mapping(self) -> dict[str, Any]:
        """Return detached serializable evidence, including unsuccessful attempts."""
        return {
            "schema": "da_plcbf_actuator_trial_ledger_v1",
            "protocol_sha256": self.protocol.sha256,
            "records": [json.loads(record) for record in self._records],
        }

    @classmethod
    def from_mapping(cls, protocol: FrozenProtocol, mapping: Mapping[str, Any]) -> TrialLedger:
        """Restore with the same duplicate, support, and completeness validation."""
        if (
            mapping.get("schema") != "da_plcbf_actuator_trial_ledger_v1"
            or mapping.get("protocol_sha256") != protocol.sha256
        ):
            raise ValueError("ledger protocol/schema binding mismatch")
        ledger = cls(protocol)
        for record in mapping["records"]:
            fields = dict(record)
            identity = fields.pop("trial_id", None)
            key = tuple(fields[name] for name in ("split", "world_id", "library_seed", "method"))
            expected = "trial-" + content_sha256({"protocol_sha256": protocol.sha256, "key": key})
            if identity != expected:
                raise ValueError("trial ID differs from its frozen physical/configuration identity")
            ledger.record_attempt(**fields)
        return ledger


@dataclass(frozen=True, slots=True)
class CrossedBootstrapResult:
    """Paired effect A-minus-B and approximate crossed-cluster percentile intervals."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    world_only_lower: float
    world_only_upper: float
    per_seed_effects: tuple[float, ...]
    n_worlds: int
    n_library_seeds: int
    n_pairs: int
    n_resamples: int
    resampling_seed: int
    qualification: str


def crossed_paired_bootstrap(
    method_a: ArrayLike,
    method_b: ArrayLike,
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    resampling_seed: int = 0,
) -> CrossedBootstrapResult:
    """Resample worlds and reused library seeds independently, preserving method pairs.

    Arrays must contain one complete, balanced scenario/dynamics cell, with axes
    (world, library_seed). The companion world-only interval conditions on the
    observed libraries. Neither interval treats W*S observations as independent.
    """
    a, b = np.asarray(method_a, dtype=float), np.asarray(method_b, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or min(a.shape) < 2:
        raise ValueError(
            "paired arrays must share (world, seed) shape with >=2 clusters on each axis"
        )
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("paired matrix must be finite and complete; no missing-case deletion")
    if not 0 < _number(confidence_level, "confidence_level") < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    _integer(n_resamples, "n_resamples", 1000)
    _integer(resampling_seed, "resampling_seed")
    differences = a - b
    if not np.all(np.isfinite(differences)):
        raise ValueError("paired differences overflowed; use appropriately scaled metrics")
    worlds, seeds = differences.shape
    rng = np.random.default_rng(resampling_seed)
    crossed, conditional = np.empty(n_resamples), np.empty(n_resamples)
    for start in range(0, n_resamples, 256):
        count = min(256, n_resamples - start)
        rows = rng.integers(worlds, size=(count, worlds))
        columns = rng.integers(seeds, size=(count, seeds))
        crossed[start : start + count] = differences[rows[:, :, None], columns[:, None, :]].mean(
            axis=(1, 2)
        )
        conditional[start : start + count] = differences[rows].mean(axis=(1, 2))
    quantiles = [(1 - confidence_level) / 2, (1 + confidence_level) / 2]
    lower, upper = np.quantile(crossed, quantiles)
    world_lower, world_upper = np.quantile(conditional, quantiles)
    qualification = "Approximate crossed-cluster percentile interval; not a zero-risk guarantee."
    if seeds <= 5:
        qualification += (
            " Few library seeds: report per-seed effects; seed-population coverage is weak."
        )
    return CrossedBootstrapResult(
        float(differences.mean()),
        float(lower),
        float(upper),
        confidence_level,
        float(world_lower),
        float(world_upper),
        tuple(map(float, differences.mean(axis=0))),
        worlds,
        seeds,
        worlds * seeds,
        n_resamples,
        resampling_seed,
        qualification,
    )


def paired_protocol_bootstrap(
    ledger: TrialLedger, comparison_id: str, metric: str, *, cell: str
) -> CrossedBootstrapResult:
    """Run a prespecified primary test with Bonferroni confidence across metric/cell tests."""
    analysis = ledger.protocol.manifest["analysis"]
    comparisons = analysis["primary_comparisons"]
    if comparison_id not in comparisons:
        raise ValueError("comparison was not declared before test evaluation")
    comparison = comparisons[comparison_id]
    if metric not in comparison["metrics"] or cell not in comparison["cells"]:
        raise ValueError("metric/cell is outside the prespecified primary comparison")
    count = sum(len(item["metrics"]) * len(item["cells"]) for item in comparisons.values())
    confidence = 1 - (1 - analysis["confidence_level"]) / count
    a, b = ledger.paired_metric(comparison["method_a"], comparison["method_b"], metric, cell=cell)
    return crossed_paired_bootstrap(
        a,
        b,
        confidence_level=confidence,
        n_resamples=analysis["n_resamples"],
        resampling_seed=analysis["resampling_seed"],
    )


def world_binary_wilson(values: ArrayLike, *, confidence_level: float = 0.95) -> dict[str, Any]:
    """Wilson diagnostic for any event across fixed seeds in an independent world.

    This changes the estimand: the denominator is W worlds, never W*S episodes.
    It is conditional on the tested libraries and does not estimate fresh-seed risk.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or not np.all((values == 0) | (values == 1)):
        raise ValueError("binary diagnostic requires a complete (world, seed) array of zeros/ones")
    if not 0 < _number(confidence_level, "confidence_level") < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    events = np.any(values == 1, axis=1)
    count, n_worlds = int(np.sum(events)), len(events)
    p = count / n_worlds
    z = NormalDist().inv_cdf((1 + confidence_level) / 2)
    denominator = 1 + z * z / n_worlds
    center = (p + z * z / (2 * n_worlds)) / denominator
    half = z * math.sqrt(p * (1 - p) / n_worlds + z * z / (4 * n_worlds**2)) / denominator
    return {
        "estimand": "any_event_across_fixed_library_seeds_in_a_world",
        "events": count,
        "worlds": n_worlds,
        "library_seeds": values.shape[1],
        "estimate": p,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
        "confidence_level": confidence_level,
        "conditional_on_tested_libraries": True,
    }


__all__ = [
    "ATTEMPT_STATUSES",
    "PROTOCOL_SCHEMA",
    "SPLITS",
    "CrossedBootstrapResult",
    "FrozenProtocol",
    "TrialLedger",
    "canonical_json",
    "content_sha256",
    "crossed_paired_bootstrap",
    "freeze_protocol",
    "load_sealed_protocol",
    "paired_protocol_bootstrap",
    "physical_world_id",
    "seal_protocol",
    "validate_protocol",
    "world_binary_wilson",
]
