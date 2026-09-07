"""Separately sealed forty-flight validation of the unchanged UNION refresh wrapper.

The seed/identity rule is sealed before geometry opening. Thirty-two new-world
flights and eight old-harm numerical checks have separate result denominators.
An exclusive per-trial claim prevents accidental reruns or an enlarged budget.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.da_plcbf_actuator_diagnostic_protocol import (  # noqa: E402
    SCHEMA,
    _resolve_geometry,
    digest,
    file_digest,
    load_sealed,
)
from benchmark.da_plcbf_actuator_incumbent_refresh import (  # noqa: E402
    IncumbentRefreshController,
    _write,
)
from crazyflow.safety.da_plcbf.actuator_protocol import content_sha256  # noqa: E402

PROTOCOL_ID = "secondary-union-refresh-validation-v1"
NEW_WORLDS = tuple(("structured", 63001 + i) for i in range(4)) + tuple(
    ("navigation", 63101 + i) for i in range(4)
)
REALIZATIONS = ("fresh_build_1", "fresh_build_2", "perturb_plus", "perturb_minus")
ARMS = ("UNION_REFRESH", "UNION_REFRESH_FREEZE_AT_FAULT")
CELLS = ("eta1_lag1_extra0", "eta0.7_lag1_extra0")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_identity_rule(path: Path) -> dict:
    envelope = json.loads(path.read_text())
    rule = envelope["rule"]
    require(envelope["sha256"] == digest(rule), "secondary identity-rule checksum changed")
    require(rule["protocol_id"] == PROTOCOL_ID, "wrong secondary identity rule")
    require(
        [(row["family"], row["scene_seed"]) for row in rule["new_worlds"]] == list(NEW_WORLDS),
        "secondary new-world identities changed",
    )
    require(
        tuple(rule["arms"]) == ARMS and tuple(rule["cells"]) == CELLS, "arm/cell design changed"
    )
    require(rule["maximum_new_physical_flights"] == 40, "secondary flight cap changed")
    require(rule["new_outcomes_consulted"] is False, "identity rule opened after new outcomes")
    require(
        rule["immutable_core_count"] == rule["adaptive_policy_count"] == 16, "library sizes changed"
    )
    return rule


def make_proposal(
    identity: Path,
    original_protocol: Path,
    original_campaign: Path,
    *,
    geometry_resolver: Callable[[str, int], dict] = _resolve_geometry,
) -> dict:
    """Generate only already-sealed identities, then bind every physical specification."""
    rule = load_identity_rule(identity)
    original = load_sealed(original_protocol)
    old_binding_path = original_campaign / "diagnostic_runtime_binding.json"
    old_binding = json.loads(old_binding_path.read_text())
    require(
        content_sha256({k: v for k, v in old_binding.items() if k != "sha256"})
        == old_binding["sha256"],
        "original numerical binding checksum changed",
    )
    worlds = {"structured_30101": deepcopy(original["worlds"]["structured_30101"])}
    old_geometry = {world["geometry_sha256"] for world in original["worlds"].values()}
    for family, seed in NEW_WORLDS:
        geometry = geometry_resolver(family, seed)
        geometry_hash = digest(geometry)
        require(
            geometry_hash not in old_geometry, "new geometry duplicates an opened original world"
        )
        require(
            geometry_hash not in {w["geometry_sha256"] for w in worlds.values()},
            "duplicate secondary geometry",
        )
        key = f"{family}_{seed}"
        worlds[key] = {
            "world_key": key,
            "family": family,
            "scene_seed": seed,
            "split": "secondary_validation",
            "geometry": geometry,
            "geometry_sha256": geometry_hash,
            "selection": "fixed in the identity rule before geometry opening",
        }
    cells = [
        deepcopy(next(c for c in original["factorial_cells"] if c["cell_id"] == key))
        for key in CELLS
    ]
    trials, stages = [], {"fresh_validation": [], "numerical_robustness": []}

    def add(
        stage: str, world_key: str, cell: dict, label: str, realization: str = "primary"
    ) -> None:
        world = worlds[world_key]
        delta = {"perturb_plus": 1e-5, "perturb_minus": -1e-5}.get(realization, 0.0)
        arm = {
            "arm": label,
            "runtime_method": "UNION",
            "freeze_learning_at": 2.0 if label.endswith("_FREEZE_AT_FAULT") else None,
            "checkpoint_phase": "identical startup and complete refresh-wrapper pre-event history",
        }
        identity_value = {
            "protocol_id": PROTOCOL_ID,
            "world_key": world_key,
            "cell_id": cell["cell_id"],
            "library_seed": 11,
            "realization": realization,
            "arm": arm,
        }
        physical = deepcopy(world["geometry"])
        physical["initial_state"][0] += delta
        physical["initial_state"][7] += delta
        physical["actuator_events"] = (
            []
            if cell["no_change_control"]
            else [
                {
                    "time_seconds": 2.0,
                    "effectiveness": cell["effectiveness_after"],
                    "lag_multipliers": cell["lag_multipliers_after"],
                    "recovery_time_seconds": None,
                }
            ]
        )
        trial_id = digest(identity_value)
        trials.append(
            {
                "trial_id": trial_id,
                **identity_value,
                "split": world["split"],
                "geometry_sha256": world["geometry_sha256"],
                "physical_spec_sha256": digest(physical),
                "initial_position_x_delta_m": delta,
                "initial_velocity_x_delta_mps": delta,
            }
        )
        stages[stage].append(trial_id)

    for family, seed in NEW_WORLDS:
        for cell in cells:
            for arm in ARMS:
                add("fresh_validation", f"{family}_{seed}", cell, arm)
    fault = next(cell for cell in cells if not cell["no_change_control"])
    for realization in REALIZATIONS:
        for arm in ARMS:
            add("numerical_robustness", "structured_30101", fault, arm, realization)
    require(
        len(trials) == len({t["trial_id"] for t in trials}) == 40, "secondary budget/identity drift"
    )
    source_files = dict(old_binding["source_sha256"])
    additional = (
        Path(__file__),
        ROOT / "benchmark/da_plcbf_actuator_incumbent_refresh.py",
        ROOT / "tests/test_da_plcbf_union_refresh_secondary.py",
    )
    source_files.update({str(path.relative_to(ROOT)): file_digest(path) for path in additional[:2]})
    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "sealed_secondary_design_before_all_forty_outcomes",
        "identity_rule": {
            "path": str(identity.resolve()),
            "file_sha256": file_digest(identity),
            "rule": rule,
        },
        "original_protocol": {
            "path": str(original_protocol.resolve()),
            "file_sha256": file_digest(original_protocol),
        },
        "original_runtime_binding": {
            "path": str(old_binding_path.resolve()),
            "file_sha256": file_digest(old_binding_path),
            "value": old_binding,
        },
        "worlds": worlds,
        "factorial_cells": cells,
        "trials": trials,
        "stage_trial_ids": stages,
        "shared_control": deepcopy(original["shared_control"]),
        "controller_override": {
            "class": "IncumbentRefreshController",
            "module": "benchmark.da_plcbf_actuator_incumbent_refresh",
            "immutable_core_count": 16,
            "adaptive_policy_count": 16,
            "rule": rule["refresh_rule"],
            "active_from_time_zero_in_both_arms": True,
            "all_finite_updates_publish": True,
            "unchanged_policy_hysteresis": 0.02,
        },
        "execution": {
            "claim_directory": str((identity.parent / "flight-claims-v1").resolve()),
            "maximum_unique_new_flights": 40,
            "automatic_physical_retries": False,
            "capture_times_seconds": [2.0, 2.4, 2.8],
            "retain_rollouts": "oldharm numerical robustness only",
            "validation_order": "world/cell/declared arm lexical; all32 retained",
            "robustness_order": list(REALIZATIONS),
            "fresh_process_per_realization": True,
            "fresh_build_cache_contract": (
                "independent new empty cache for each fresh_build; "
                "perturbations reuse fresh_build_1 cache"
            ),
        },
        "budget": {
            "new_world_validation": 32,
            "oldharm_numerical_robustness": 8,
            "maximum_new_flights": 40,
        },
        "analysis_contract": {
            "all_declared_pair_slots": 20,
            "fresh_validation_pair_count": 16,
            "oldharm_robustness_pair_count": 4,
            "pre_event_identity": (
                "complete original physical/learner/Adam/reference/model/update prefix plus "
                "raw wrapper-call history and preceding parameter fingerprint at the event"
            ),
            "common_states": {
                "times_seconds": [2.0, 2.4, 2.8],
                "physical_branches": ["frozen", "adaptive"],
                "libraries": [
                    "immutable F2 core plus event-frozen adaptive component",
                    "same immutable F2 core plus current adaptive component",
                ],
                "fallback_count_each": 32,
                "indices": "nominal0, immutable core1..16, adaptive component17..32",
                "metrics": [
                    "per-policy hard/smooth collision values and rollout validity",
                    "0.4s displacement",
                    "1.2s terminal speed",
                    "first policy command",
                    "actual applied-command divergence",
                    "immutable core equality",
                ],
            },
            "incomplete_policy": (
                "retain all missing/failed/mismatched records and deny outcome credit "
                "until complete original and wrapper prefix authentication passes"
            ),
        },
        "source_sha256": source_files,
        "checkpoint_files_sha256": old_binding["checkpoint_files_sha256"],
        "planner_files": {str(path.resolve()): file_digest(path) for path in additional},
        "interpretation": (
            "Separate secondary validation, preserving the original144 and earlier8-refresh "
            "outcomes. Freeze/continue tests post-event learning plus its policy-version "
            "selection consequence. "
            "Benefit may come from selection refresh without better individual adapted policies. "
            "No new unrefreshed UNION or F2_2K comparison, no population superiority "
            "or recursive safety claim."
        ),
    }


def verify_sources(proposal: dict) -> None:
    for name, expected in proposal["source_sha256"].items():
        require(file_digest(ROOT / name) == expected, f"sealed source changed: {name}")
    for name, expected in proposal["checkpoint_files_sha256"].items():
        require(file_digest(Path(name)) == expected, f"sealed checkpoint changed: {name}")


def seal(identity: Path, original_protocol: Path, original_campaign: Path, output: Path) -> dict:
    require(not output.exists(), "secondary protocol already exists")
    proposal = make_proposal(identity, original_protocol, original_campaign)
    verify_sources(proposal)
    envelope = {"schema": SCHEMA, "sha256": digest(proposal), "proposal": proposal}
    _write(output, envelope)
    archived_names = set(proposal["source_sha256"])
    archived_names.update(str(Path(name).relative_to(ROOT)) for name in proposal["planner_files"])
    for name in sorted(archived_names):
        target = output.parent / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), "sealed source copy already exists")
        shutil.copy2(ROOT / name, target)
    return envelope


def load_secondary(path: Path) -> dict:
    proposal = load_sealed(path)
    require(proposal["protocol_id"] == PROTOCOL_ID, "not the secondary UNION refresh protocol")
    require(len(proposal["trials"]) == 40, "secondary trial count changed")
    identity = proposal["identity_rule"]
    require(
        file_digest(Path(identity["path"])) == identity["file_sha256"], "identity rule file changed"
    )
    require(load_identity_rule(Path(identity["path"])) == identity["rule"], "identity rule differs")
    verify_sources(proposal)
    return proposal


def claim_trial(proposal: dict, trial: dict, episode: Path) -> Path:
    """Reserve each planned physical attempt once across all launch processes/outputs."""
    require(trial in proposal["trials"], "cannot claim an unplanned trial")
    directory = Path(proposal["execution"]["claim_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{trial['trial_id']}.json"
    _write(
        path,
        {
            "trial_id": trial["trial_id"],
            "episode_directory": str(episode.resolve()),
            "reserved_wall_time": time.time(),
            "physical_reexecution_permitted": False,
        },
    )
    return path


def audit_calls(calls: list[dict], controls: dict[str, np.ndarray]) -> list[dict]:
    previous_hash = None
    for index, row in enumerate(calls):
        require(row["call_index"] == index, "wrapper call indices are not complete and sequential")
        require(
            row["preceding_call_parameter_sha256"] == previous_hash,
            "wrapper fingerprint memory differs",
        )
        require(
            row["parameters_changed"]
            == (previous_hash is not None and row["parameter_sha256"] != previous_hash),
            "wrapper parameter-change classification differs",
        )
        previous_hash = row["parameter_sha256"]
    physical = [row for row in calls if not row["warmup"]]
    require(
        len(physical) == len(controls["time"]), "wrapper physical-call count differs from controls"
    )
    for index, row in enumerate(physical):
        require(
            row["state_sha256"] == str(controls["controller_input_state_sha256"][index]),
            "wrapper state differs",
        )
        require(
            row["parameter_sha256"] == str(controls["control_params_sha256"][index]),
            "wrapper parameters differ",
        )
        require(
            np.array_equal(row["action"], controls["planned_command"][index]),
            "wrapper command differs",
        )
        require(
            row["incumbent_in_adaptive_component"] == (16 < row["requested_previous_index"] <= 32),
            "wrapper component classification differs",
        )
        require(
            row["refreshed"]
            == (row["parameters_changed"] and row["incumbent_in_adaptive_component"]),
            "wrapper refresh rule differs",
        )
        require(
            row["effective_previous_index"]
            == (-1 if row["refreshed"] else row["requested_previous_index"]),
            "wrapper effective index differs",
        )
        row.update(
            control_index=index,
            time_seconds=float(controls["time"][index]),
            command_applied=bool(controls["command_applied"][index]),
        )
    return physical


def run_part(protocol: Path, stage: str, output: Path, *, realization: str | None = None) -> dict:
    import jax

    from benchmark.da_plcbf_actuator_diagnostics import DiagnosticResources, resolve_trial_scene
    from crazyflow.safety.da_plcbf.actuator_experiment import (
        ActuatorEpisodeConfig,
        run_actuator_episode,
    )
    from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
    from crazyflow.safety.da_plcbf.actuator_study import ActuatorObservationConfig

    proposal = load_secondary(protocol)
    ids = set(proposal["stage_trial_ids"][stage])
    trials = [
        row
        for row in proposal["trials"]
        if row["trial_id"] in ids and (realization is None or row["realization"] == realization)
    ]
    require(bool(trials), "no declared trials in this execution part")
    if stage == "numerical_robustness":
        require(
            realization in REALIZATIONS and len(trials) == 2,
            "robustness requires one new process per declared realization",
        )
    trials.sort(key=lambda row: (row["world_key"], row["cell_id"], row["arm"]["arm"]))
    output.mkdir(parents=True, exist_ok=False)
    binding = {
        "protocol": str(protocol.resolve()),
        "protocol_file_sha256": file_digest(protocol),
        "source_sha256": proposal["source_sha256"],
        "checkpoint_files_sha256": proposal["checkpoint_files_sha256"],
        "scientific_runtime_sha256": content_sha256(
            {
                "source": proposal["source_sha256"],
                "checkpoints": proposal["checkpoint_files_sha256"],
            }
        ),
        "stage": stage,
        "resolved_trial_ids": [row["trial_id"] for row in trials],
        "controller_override": proposal["controller_override"],
        "secondary_protocol_id": PROTOCOL_ID,
        "execution_mode_override": None,
    }
    binding["sha256"] = content_sha256(binding)
    _write(output / "diagnostic_runtime_binding.json", binding)
    for name in binding["source_sha256"]:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    _write(
        output / "environment.json",
        {
            "devices": [str(device) for device in jax.devices()],
            "jax_version": jax.__version__,
            "pid": os.getpid(),
            "compilation_cache": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
            "execution_mode": "deterministic",
        },
    )
    resources, records = DiagnosticResources(), []
    for trial in trials:
        verify_sources(proposal)
        directory = output / trial["trial_id"]
        directory.mkdir()
        episode = directory / "attempt-00"
        configuration = deepcopy(proposal["shared_control"])
        filters = ActuatorFilterConfig(**configuration.pop("filter_config"))
        observation = ActuatorObservationConfig(**configuration.pop("observation_config", {}))
        scene = resolve_trial_scene(proposal, trial, filters)
        configuration.update(
            method="UNION",
            freeze_learning_at=trial["arm"]["freeze_learning_at"],
            capture_times=(2.0, 2.4, 2.8),
            save_checkpoints=True,
            retain_rollouts=stage == "numerical_robustness",
        )
        config = ActuatorEpisodeConfig(
            filter_config=filters, observation_config=observation, **configuration
        )
        bundle, functions, learner, metadata = resources.resolve("UNION", filters)
        require(
            metadata["immutable_core"]["fallback_policy_count"]
            == int(bundle.contract.spec.latent_codes.shape[0])
            == 16,
            "sealed library sizes differ",
        )
        wrapper = IncumbentRefreshController(
            functions.controller,
            immutable_core_count=16,
            adaptive_policy_count=16,
            warmup_calls=config.warmup_calls,
        )
        claim = claim_trial(proposal, trial, episode)
        print(
            json.dumps(
                {
                    "starting": trial["trial_id"],
                    "world": trial["world_key"],
                    "cell": trial["cell_id"],
                    "arm": trial["arm"]["arm"],
                    "realization": trial["realization"],
                }
            ),
            flush=True,
        )
        caught = None
        try:
            run_actuator_episode(
                scene,
                bundle,
                config,
                episode,
                controllerfunctions=functions._replace(controller=wrapper),
                learner_functions=learner,
            )
        except Exception as exc:
            caught = {"type": type(exc).__name__, "message": str(exc)}
        _write(directory / "incumbent_audit.json", wrapper.records)
        summary_path = episode / "summary.json"
        summary = (
            json.loads(summary_path.read_text())
            if summary_path.exists()
            else {
                "status": "interrupted",
                "termination": "missing_complete_summary",
                "error": caught,
            }
        )
        verified = False
        audit_error = None
        try:
            with np.load(episode / "controls.npz", allow_pickle=False) as source:
                physical = audit_calls(
                    wrapper.records, {name: source[name] for name in source.files}
                )
            verified = True
        except Exception as exc:
            physical = [row for row in wrapper.records if not row["warmup"]]
            audit_error = {"type": type(exc).__name__, "message": str(exc)}
        # Keep the original raw audit and separately retain the authenticated indexed history.
        _write(directory / "incumbent_audit_indexed.json", wrapper.records)
        record = {
            "trial": trial,
            "method_metadata": metadata,
            "summary": summary,
            "episode_directory": str(episode.resolve()),
            "campaign_binding_sha256": binding["sha256"],
            "runtime_binding_sha256": binding["scientific_runtime_sha256"],
            "controller_override": proposal["controller_override"],
            "refresh_count": sum(row["refreshed"] for row in physical),
            "actual_input_and_command_verification": verified,
            "wrapper_audit_error": audit_error,
            "execution_exception": caught,
            "exclusive_flight_claim": str(claim),
            "exclusive_flight_claim_sha256": file_digest(claim),
            "evidence_sha256": {
                str(path.resolve()): file_digest(path)
                for path in directory.rglob("*")
                if path.is_file()
            },
        }
        _write(directory / "record.json", record)
        records.append(record)
        _write(
            output / f"progress-{len(records):02d}.json",
            {
                "planned": len(trials),
                "retained": len(records),
                "record_file": str(directory / "record.json"),
            },
        )
        print(
            json.dumps(
                {
                    "finished": trial["arm"]["arm"],
                    "world": trial["world_key"],
                    "cell": trial["cell_id"],
                    "status": summary["status"],
                    "termination": summary["termination"],
                    "waypoints": summary.get("waypoints_completed"),
                    "refresh_count": record["refresh_count"],
                    "wrapper_verified": verified,
                }
            ),
            flush=True,
        )
    result = {
        "schema": "secondary_union_refresh_results_v1",
        "protocol_id": PROTOCOL_ID,
        "stage": stage,
        "planned": len(trials),
        "retained": len(records),
        "completed": sum(row["summary"]["status"] == "completed" for row in records),
        "records": records,
    }
    _write(output / "results.json", result)
    return result


def launch(protocol: Path, output: Path) -> dict:
    """Execute validation then four independent robustness processes, at most forty claims."""
    proposal = load_secondary(protocol)
    output.mkdir(parents=True, exist_ok=False)
    first_cache = Path(tempfile.mkdtemp(prefix="crazyflow-union-refresh-secondary-fresh1-"))
    second_cache = Path(tempfile.mkdtemp(prefix="crazyflow-union-refresh-secondary-fresh2-"))
    require(
        not list(first_cache.iterdir()) and not list(second_cache.iterdir()),
        "new robustness caches must be empty",
    )
    canonical_cache = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR", "/tmp/crazyflow-actuator-diagnostic-jax-v1"
    )
    parts = [
        {
            "stage": "fresh_validation",
            "realization": None,
            "output": str(output / "fresh_validation"),
            "cache": canonical_cache,
        }
    ]
    parts += [
        {
            "stage": "numerical_robustness",
            "realization": realization,
            "output": str(output / "numerical_robustness" / realization),
            "cache": str(second_cache if realization == "fresh_build_2" else first_cache),
        }
        for realization in REALIZATIONS
    ]
    manifest = {
        "protocol": str(protocol.resolve()),
        "protocol_sha256": file_digest(protocol),
        "driver_sha256": file_digest(Path(__file__)),
        "maximum_unique_new_flights": 40,
        "parts": parts,
        "fresh_empty_caches_before_launch": [str(first_cache), str(second_cache)],
    }
    _write(output / "launcher.json", manifest)
    records, completions = [], []
    for part in parts:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run",
            "--protocol",
            str(protocol.resolve()),
            "--stage",
            part["stage"],
            "--output",
            part["output"],
        ]
        if part["realization"] is not None:
            command += ["--realization", part["realization"]]
        environment = dict(
            os.environ,
            JAX_PLATFORMS="cuda",
            JAX_COMPILATION_CACHE_DIR=part["cache"],
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
        )
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        result_path = Path(part["output"]) / "results.json"
        part_record_paths = sorted(Path(part["output"]).glob("*/record.json"))
        records.extend(json.loads(path.read_text()) for path in part_record_paths)
        completions.append(
            {
                "part": part,
                "exit_code": completed.returncode,
                "result_file": str(result_path) if result_path.exists() else None,
            }
        )
        _write(output / f"launcher-completion-{len(completions)}.json", completions[-1])
    require(
        len({row["trial"]["trial_id"] for row in records}) == len(records),
        "duplicate secondary records",
    )
    result = {
        "schema": "secondary_union_refresh_results_v1",
        "protocol_id": PROTOCOL_ID,
        "planned": 40,
        "retained": len(records),
        "completed": sum(row["summary"]["status"] == "completed" for row in records),
        "records": records,
        "parts": completions,
        "claimed_trial_count": len(
            list(Path(proposal["execution"]["claim_directory"]).glob("*.json"))
        ),
        "status": "completed"
        if len(records) == 40
        and all(
            row["summary"]["status"] == "completed" and row["actual_input_and_command_verification"]
            for row in records
        )
        else "incomplete_all_attempts_retained",
    }
    _write(output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("seal")
    prepare.add_argument("--identity", type=Path, required=True)
    prepare.add_argument("--original-protocol", type=Path, required=True)
    prepare.add_argument("--original-campaign", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--stage", choices=("fresh_validation", "numerical_robustness"), required=True)
    run.add_argument("--realization", choices=REALIZATIONS)
    run.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("launch")
    execute.add_argument("--protocol", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        result = seal(args.identity, args.original_protocol, args.original_campaign, args.output)
        print(
            json.dumps(
                {
                    "protocol": str(args.output),
                    "sha256": result["sha256"],
                    "budget": result["proposal"]["budget"],
                }
            )
        )
    elif args.command == "run":
        run_part(args.protocol, args.stage, args.output, realization=args.realization)
    else:
        result = launch(args.protocol, args.output)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "planned": result["planned"],
                    "retained": result["retained"],
                }
            )
        )


if __name__ == "__main__":
    main()
