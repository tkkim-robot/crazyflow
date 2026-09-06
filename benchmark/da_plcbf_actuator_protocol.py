"""Prepare an auditable actuator protocol, explicitly seal it, or import retained trials.

``prepare`` writes a validated draft, never a sealed protocol. Inputs are an actual
test campaign, explicit development/validation split objects with ``worlds`` and
``library_seeds``, and a decision object containing ``protocol_id``, ``numeric_ranges``,
``training_support``, ``uncertainty``, ``analysis`` settings and ``budget_rationale``.
There are no generated test seeds, default budgets, or automatic test executions.

``seal`` is a separate explicit command and rechecks committed source and checkpoint
bytes. ``import`` authenticates a completed campaign envelope, retains every attempt,
and leaves interrupted, erroneous, unresolved or inadmissible trials incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.da_plcbf_actuator_analysis import METRICS, Sources, _episode_metrics, read_campaign
from benchmark.da_plcbf_actuator_study import resolve_scene, runner_source_files
from crazyflow.safety.da_plcbf.actuator_experiment import ActuatorEpisodeConfig
from crazyflow.safety.da_plcbf.actuator_independent import NativeRotorParameters
from crazyflow.safety.da_plcbf.actuator_learning import (
    actuator_reference_fingerprint,
    load_actuator_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.actuator_opt import ActuatorOPTConfig
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_protocol import (
    ATTEMPT_STATUSES,
    PROTOCOL_SCHEMA,
    TrialLedger,
    content_sha256,
    freeze_protocol,
    load_sealed_protocol,
    seal_protocol,
)
from crazyflow.safety.da_plcbf.actuator_study import ActuatorObservationConfig

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_METRICS = (
    "actual_collision",
    "safe_task_completion",
    "operational_violation",
    "collider_clearance_lower_m",
    "controller_deadline_miss_rate",
)
DEFINITIONS = {
    "F0": "Frozen library with nominal instantaneous wrench-to-effort allocation",
    "F1": "Frozen library with current-effectiveness instantaneous allocation",
    "F2": "Frozen library with current-effectiveness and lag-aware endpoint allocation",
    "A": "Persistent adaptive library with the same current-parameter F2 allocation",
    "DR": "Frozen domain-randomized library with current-parameter F2 allocation",
    "OPT": "Bounded direct-command horizon optimization with measured completion availability",
    "A1": "Persistent adaptive single-recovery library with current-parameter F2 allocation",
}


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_asdict"):
        return _plain(value._asdict())
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return _plain(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    return value


def _omit_inapplicable(value: Any) -> Any:
    # Optional runtime configuration fields have explicit disabled semantics.
    # Their original complete JSON is bound by the campaign specification digest.
    if isinstance(value, dict):
        return {key: _omit_inapplicable(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_omit_inapplicable(item) for item in value]
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(_plain(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def committed_sources(root: Path = ROOT) -> dict[str, Any]:
    """Require every bound implementation file to equal its bytes at the current commit."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    files = runner_source_files(root)
    hashes = {}
    for file in files:
        relative = str(file.relative_to(root))
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=root, capture_output=True, check=False
        )
        if result.returncode or result.stdout != file.read_bytes():
            raise ValueError(
                f"bound source must be committed and clean before preparation: {relative}"
            )
        hashes[relative] = hashlib.sha256(result.stdout).hexdigest()
    # The builder and metric adapter are separately recorded because the runner's
    # envelope intentionally hashes only its own implementation files.
    auxiliary = {}
    for relative in (
        "benchmark/da_plcbf_actuator_protocol.py",
        "benchmark/da_plcbf_actuator_analysis.py",
    ):
        result = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=root, capture_output=True, check=False
        )
        file = root / relative
        if result.returncode or result.stdout != file.read_bytes():
            raise ValueError(
                f"protocol/analysis implementation must be committed and clean: {relative}"
            )
        auxiliary[relative] = hashlib.sha256(result.stdout).hexdigest()
    return {"commit": commit, "files_sha256": hashes, "analysis_files_sha256": auxiliary}


def _configuration(
    raw: dict[str, Any], method: str
) -> tuple[ActuatorEpisodeConfig, ActuatorOPTConfig]:
    values = dict(raw)
    values.pop("method", None)
    filters = ActuatorFilterConfig(**values.pop("filter_config", {}))
    observation = ActuatorObservationConfig(**values.pop("observation_config", {}))
    optimizer = ActuatorOPTConfig(**values.pop("opt_config", {}))
    return ActuatorEpisodeConfig(
        method=method, filter_config=filters, observation_config=observation, **values
    ), optimizer


def _split_support(
    specification: dict[str, Any], common: ActuatorFilterConfig, name: str
) -> dict[str, Any]:
    if not specification.get("worlds") or not specification.get("library_seeds"):
        raise ValueError(f"{name} requires explicit nonempty worlds and library_seeds")
    worlds, seeds = [], []
    for world in specification["worlds"]:
        seed = world.get("scene_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("each world must explicitly declare its nonnegative scene_seed")
        override = world.get("episode_config", {}).get("filter_config")
        if override is not None and asdict(ActuatorFilterConfig(**override)) != asdict(common):
            raise ValueError("world-specific filter clocks must equal the campaign contract")
        scene = resolve_scene(world, common)
        metadata = _plain(scene.metadata())
        worlds.append(
            {
                "world_id": metadata["physical_world_id"],
                "physical_spec": metadata["physical_spec"],
                "cell": world.get("cell", f"{scene.family}_{scene.dynamics_cell}"),
            }
        )
        if seed not in seeds:
            seeds.append(seed)
    return {"world_seeds": seeds, "library_seeds": specification["library_seeds"], "worlds": worlds}


def prepare_protocol(
    campaign_path: Path, development_path: Path, validation_path: Path, decisions_path: Path
) -> dict[str, Any]:
    """Build only a validated draft from explicit decisions and resolved physical support."""
    source = committed_sources(ROOT)
    sources = Sources()
    campaign = sources.read(campaign_path)
    development = sources.read(development_path)
    validation = sources.read(validation_path)
    decisions = sources.read(decisions_path)
    required = {
        "protocol_id",
        "numeric_ranges",
        "training_support",
        "uncertainty",
        "analysis",
        "budget_rationale",
    }
    if not required.issubset(decisions):
        raise ValueError(
            f"explicit protocol decisions missing {sorted(required - decisions.keys())}"
        )
    if campaign.get("split") != "test":
        raise ValueError("the final campaign must explicitly declare split=test")
    methods, seeds = campaign["methods"], campaign["library_seeds"]
    if not {"A", "F2", "DR", "OPT"}.issubset(methods):
        raise ValueError("the primary campaign must include A, F2, DR and OPT")
    if len(set(methods)) != len(methods) or len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("methods must be unique and at least two distinct library seeds declared")
    if set(decisions["training_support"]) != set(methods):
        raise ValueError("training_support must explicitly cover every declared method")
    if not isinstance(decisions["uncertainty"], dict) or not decisions["uncertainty"]:
        raise ValueError("an explicit nonempty uncertainty contract is required")
    common = ActuatorFilterConfig(**campaign["episode_config"].get("filter_config", {}))
    common.validate()
    splits = {
        "development": _split_support(development, common, "development"),
        "validation": _split_support(validation, common, "validation"),
        "test": _split_support(campaign, common, "test"),
    }
    methods_output, checkpoints, support, bundles = {}, {}, {}, {}
    nominal_model = None
    for method in methods:
        if method not in DEFINITIONS:
            raise ValueError(f"unknown method {method}")
        configuration = dict(campaign["episode_config"])
        configuration.update(campaign.get("method_config", {}).get(method, {}))
        episode, optimizer = _configuration(configuration, method)
        if episode.filter_config != common:
            raise ValueError("primary methods must share the complete filter configuration")
        checkpoint_records = {}
        for seed in seeds:
            try:
                stem = Path(campaign["checkpoints"][method][str(seed)])
            except KeyError as error:
                raise ValueError(
                    f"missing explicit checkpoint for {method}, seed {seed}"
                ) from error
            stem = stem if stem.is_absolute() else ROOT / stem
            if not stem.resolve().is_relative_to(ROOT):
                raise ValueError(
                    "checkpoint paths must be within the repository, matching the runner"
                )
            bundle = load_actuator_learner_checkpoint(stem)
            if bundle.metadata.get("seed") != seed:
                raise ValueError(
                    "checkpoint seed provenance differs from its declared library seed"
                )
            bundles[method, seed] = bundle
            current_model = _plain(bundle.contract.model)
            if nominal_model is None:
                nominal_model = current_model
            elif content_sha256(current_model) != content_sha256(nominal_model):
                raise ValueError(
                    "all primary checkpoints must share the same nominal physical model"
                )
            scene = resolve_scene(campaign["worlds"][0], common)
            episode.validate(scene, bundle)
            pair_hashes = {}
            for suffix in (".npz", ".json"):
                path = stem.with_suffix(suffix)
                digest = sources.bind(path)
                checkpoints[str(path.relative_to(ROOT))] = digest
                pair_hashes[suffix] = digest
            checkpoint_records[str(seed)] = {
                "stem": str(stem.relative_to(ROOT)),
                "files_sha256": pair_hashes,
                "reference_sha256": actuator_reference_fingerprint(bundle.contract),
                "initial_library_version": int(bundle.state.library_version),
                "cumulative_gradient_steps": int(bundle.state.cumulative_gradient_steps),
                "policy_count": len(bundle.contract.spec.latent_codes),
            }
        first = bundles[method, seeds[0]]
        actor = _plain(asdict(first.config))
        if any(_plain(asdict(bundles[method, seed].config)) != actor for seed in seeds[1:]):
            raise ValueError(
                "one method must use identical actor configuration across library seeds"
            )
        if method in {"F0", "F1"}:
            actor["adapter_mode"] = method
        methods_output[method] = {
            "definition": DEFINITIONS[method],
            "configuration": _omit_inapplicable(
                {
                    "episode": _plain(asdict(episode)),
                    "actor": actor,
                    "optimizer": asdict(optimizer) if method == "OPT" else None,
                }
            ),
        }
        support[method] = {
            **decisions["training_support"][method],
            "resolved_checkpoints": checkpoint_records,
        }
    for seed in seeds:
        a, frozen = (
            support["A"]["resolved_checkpoints"][str(seed)],
            support["F2"]["resolved_checkpoints"][str(seed)],
        )
        if a["files_sha256"] != frozen["files_sha256"]:
            raise ValueError("A and F2 must start from identical complete checkpoint bytes")
        for method in ("DR", "OPT"):
            if (
                support[method]["resolved_checkpoints"][str(seed)]["reference_sha256"]
                != a["reference_sha256"]
            ):
                raise ValueError("primary methods must share an immutable nominal teacher per seed")
    for field in ("plant_level", "plant_step_seconds", "execution_mode", "observation_config"):
        values = [
            methods_output[method]["configuration"]["episode"][field]
            for method in ("F2", "DR", "A", "OPT")
        ]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"all primary methods must share {field}")
    episode = methods_output["A"]["configuration"]["episode"]
    analysis = dict(decisions["analysis"])
    if "primary_comparisons" in analysis:
        raise ValueError(
            "primary comparisons are fixed here to A versus F2/DR/OPT for all five primary outcomes"
        )
    cells = sorted({world["cell"] for world in splits["test"]["worlds"]})
    if any(sum(world["cell"] == cell for world in splits["test"]["worlds"]) < 2 for cell in cells):
        raise ValueError("each primary cell needs at least two independent physical worlds")
    analysis["primary_comparisons"] = {
        f"A_minus_{method}": {
            "method_a": "A",
            "method_b": method,
            "metrics": list(PRIMARY_METRICS),
            "cells": cells,
        }
        for method in ("F2", "DR", "OPT")
    }
    native = _plain(asdict(NativeRotorParameters.from_drone()))
    unsigned = {key: value for key, value in campaign.items() if key != "sealed_protocol_path"}
    manifest = {
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": decisions["protocol_id"],
        "source": source,
        "campaign_specification_sha256": content_sha256(unsigned),
        "checkpoint_sha256": checkpoints,
        "models": {
            "P0": {
                "definition": (
                    "Matched 17-state effort-lag surrogate with exact evolving motor effort "
                    "and RK4 body integration"
                ),
                "parameters": nominal_model,
            },
            "P1": {
                "definition": "Independent NumPy 17-state effort-lag model with RK4 integration",
                "parameters": nominal_model,
            },
            "P2": {
                "definition": (
                    "Independent native-RPM model with asymmetric spin dynamics "
                    "and nonlinear thrust mapping"
                ),
                "parameters": {
                    "nominal_body_and_effort_interface": nominal_model,
                    "native_rotor_parameters": native,
                },
            },
        },
        "control": {
            "command_period_s": common.command_period,
            "prediction_horizon_s": common.horizon * common.dt,
            "integration_step_s": episode["plant_step_seconds"],
            "prediction_step_s": common.dt,
            "filter": _plain(asdict(common)),
            "execution_mode": episode["execution_mode"],
            "timing_contract": (
                "Complete synchronized observation/transfer/controller/mandatory-host service; "
                "availability governs physical command timing according to declared execution mode"
            ),
        },
        "numeric_ranges": decisions["numeric_ranges"],
        "methods": methods_output,
        "training_support": support,
        "splits": splits,
        "outcomes": {metric: METRICS[metric] for metric in PRIMARY_METRICS},
        "analysis": analysis,
        "uncertainty": {
            **decisions["uncertainty"],
            "resolved_primary_observation_config": episode["observation_config"],
        },
        "budget_rationale": decisions["budget_rationale"],
        "builder_provenance": {
            "command": "prepare (draft only)",
            "campaign_source_path": str(campaign_path.resolve()),
            "input_sha256": sources.hashes,
            "optional_runtime_null_fields": (
                "Omitted as disabled/inapplicable in method configuration; "
                "original campaign remains fully digest-bound"
            ),
        },
    }
    return freeze_protocol(_plain(manifest)).manifest


def seal_draft(draft_path: Path, output: Path) -> dict[str, Any]:
    """Explicit sealing rechecks the committed implementation, checkpoints and campaign."""
    manifest = json.loads(draft_path.read_text())
    frozen = freeze_protocol(manifest)
    if committed_sources(ROOT) != manifest["source"]:
        raise ValueError("source or commit changed after draft preparation")
    for relative, expected in manifest["checkpoint_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("checkpoint changed after draft preparation")
    campaign = json.loads(Path(manifest["builder_provenance"]["campaign_source_path"]).read_text())
    unsigned = {key: value for key, value in campaign.items() if key != "sealed_protocol_path"}
    if content_sha256(unsigned) != manifest["campaign_specification_sha256"]:
        raise ValueError("campaign changed after draft preparation")
    seal_protocol(output, frozen)
    return {"sealed_protocol": str(output.resolve()), "sha256": frozen.sha256}


def import_campaign(protocol_path: Path, directory: Path) -> tuple[TrialLedger, dict[str, Any]]:
    """Retain all authenticated attempts, censoring unknown primary outcomes explicitly."""
    sources = Sources()
    sources.read(protocol_path)
    protocol = load_sealed_protocol(protocol_path)
    manifest = protocol.manifest
    dataset = read_campaign(directory, sources)
    binding = sources.read(directory / "campaign_binding.json")
    campaign = binding["specification"]
    unsigned = {key: value for key, value in campaign.items() if key != "sealed_protocol_path"}
    if (
        campaign.get("split") != "test"
        or content_sha256(unsigned) != manifest["campaign_specification_sha256"]
    ):
        raise ValueError("campaign is outside the frozen test specification")
    if binding.get("sealed_protocol_sha256") != protocol.sha256:
        raise ValueError("campaign does not bind this sealed protocol")
    if (
        binding["source_sha256"] != manifest["source"]["files_sha256"]
        or binding["checkpoint_sha256"] != manifest["checkpoint_sha256"]
    ):
        raise ValueError("campaign source/checkpoint bytes differ from the sealed protocol")
    spec = manifest["splits"]["test"]
    if campaign["library_seeds"] != spec["library_seeds"] or set(campaign["methods"]) != set(
        manifest["methods"]
    ):
        raise ValueError("campaign methods or seeds differ from the frozen matrix")
    worlds = spec["worlds"]
    if len(worlds) != len(campaign["worlds"]):
        raise ValueError("campaign world count differs from the frozen matrix")
    ledger = TrialLedger(protocol)
    primary = {
        metric
        for comparison in manifest["analysis"]["primary_comparisons"].values()
        for metric in comparison["metrics"]
    }
    selected = {row["summary_path"]: row for row in dataset["rows"] if row["summary_path"]}
    lookup = {
        f"world-{index:04d}-seed-{seed}-{method}": (world["world_id"], seed, method)
        for index, world in enumerate(worlds)
        for seed in spec["library_seeds"]
        for method in manifest["methods"]
    }
    for attempt in dataset["attempts"]:
        if attempt["trial"] not in lookup:
            raise ValueError("retained attempt is outside the frozen trial matrix")
        world_id, seed, method = lookup[attempt["trial"]]
        summary = attempt.get("summary", {})
        if summary.get("physical_world_id") not in (None, world_id):
            raise ValueError("retained attempt has a different physical world")
        path = Path(attempt["path"])
        audit_path = path.parent / "collision_audit.json"
        audit = (
            sources.read(audit_path) if path.name == "summary.json" and audit_path.exists() else {}
        )
        metrics = _episode_metrics(summary, audit)
        metrics = {
            key: value
            for key, value in metrics.items()
            if key in manifest["outcomes"] and value is not None
        }
        status = summary.get("status", "interrupted")
        reason = ""
        if status == "completed":
            if str(path) not in selected:
                raise ValueError("completed attempt was not selected by its campaign")
            if not primary.issubset(metrics):
                status = "censored"
                reason = (
                    f"Primary outcomes unresolved or missing: {sorted(primary - metrics.keys())}"
                )
        elif status not in ATTEMPT_STATUSES:
            error = summary.get("error")
            status = (
                "interrupted"
                if error and error.get("type") in {"KeyboardInterrupt", "SystemExit"}
                else "simulator_error"
                if error
                else "censored"
            )
        if status != "completed" and not reason:
            reason = str(
                summary.get("reason")
                or attempt.get("reason")
                or summary.get("error")
                or f"Runtime termination={summary.get('termination', 'no terminal summary')}"
            )
        attempt_name = path.parent.name if path.name == "summary.json" else path.name
        ledger.record_attempt(
            attempt_id=f"{dataset['id']}:{attempt_name}:{attempt['trial']}",
            split="test",
            world_id=world_id,
            library_seed=seed,
            method=method,
            status=status,
            metrics=metrics,
            reason=reason,
        )
    coverage = {
        "schema": "da_plcbf_actuator_import_coverage_v1",
        "protocol_sha256": protocol.sha256,
        "campaign_sha256": dataset["id"],
        **ledger.coverage(),
        "input_sha256": sources.hashes,
        "retained_attempts": dataset["attempts"],
        "missing_results_policy": "refuse_incomplete_matrix",
    }
    return ledger, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Build a validated draft without sealing or running test trials"
    )
    prepare.add_argument("--campaign", type=Path, required=True)
    prepare.add_argument("--development", type=Path, required=True)
    prepare.add_argument("--validation", type=Path, required=True)
    prepare.add_argument("--decisions", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    seal = commands.add_parser(
        "seal",
        help="Explicitly seal a reviewed draft after rechecking committed source/checkpoints",
    )
    seal.add_argument("--draft", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    importer = commands.add_parser(
        "import", help="Import all retained attempts into a sealed-protocol ledger"
    )
    importer.add_argument("--protocol", type=Path, required=True)
    importer.add_argument("--campaign", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare_protocol(
            args.campaign, args.development, args.validation, args.decisions
        )
        _write(args.output, manifest)
        print(
            json.dumps(
                {
                    "draft": str(args.output.resolve()),
                    "sha256": content_sha256(manifest),
                    "sealed": False,
                },
                indent=2,
            )
        )
    elif args.command == "seal":
        print(json.dumps(seal_draft(args.draft, args.output), indent=2))
    else:
        ledger, coverage = import_campaign(args.protocol, args.campaign)
        args.output.mkdir(parents=True, exist_ok=False)
        _write(args.output / "ledger.json", ledger.as_mapping())
        _write(args.output / "coverage.json", coverage)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in coverage.items()
                    if key not in {"input_sha256", "retained_attempts"}
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
