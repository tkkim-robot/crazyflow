"""Prospective eight-flight diagnostic of hysteresis across changed policy functions.

The existing controller receives previous_index=-1 only when parameters changed
and its incumbent belongs to the adaptive part of the library. Publication,
eligibility, hysteresis margin, QP, held checks and physical clocks stay governed
by the original runtime. This is a new benchmark wrapper, never an old-run edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "actuator_incumbent_refresh_diagnostic_v1"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    def plain(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): plain(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(child) for child in item]
        if isinstance(item, (np.ndarray, np.generic)):
            return plain(item.tolist())
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    encoded = json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(encoded)


def parameter_fingerprint(value: Any) -> str:
    """Use the same dtype/shape/parameter-byte digest recorded by the episode runner."""
    from crazyflow.safety.da_plcbf.actuator_experiment import _hash_tree

    return _hash_tree(value)


class IncumbentRefreshController:
    """Conservatively invalidate adaptive incumbency when library parameter bytes change."""

    def __init__(
        self,
        controller: Callable,
        *,
        immutable_core_count: int,
        adaptive_policy_count: int = 16,
        fingerprint: Callable[[Any], str] = parameter_fingerprint,
        warmup_calls: int = 0,
    ) -> None:
        """Create one episode's wrapper around a reusable packed controller."""
        if (
            isinstance(immutable_core_count, bool)
            or not isinstance(immutable_core_count, int)
            or immutable_core_count < 0
        ):
            raise ValueError("immutable core count must be a nonnegative integer")
        if isinstance(warmup_calls, bool) or not isinstance(warmup_calls, int) or warmup_calls < 0:
            raise ValueError("warmup call count must be a nonnegative integer")
        if (
            isinstance(adaptive_policy_count, bool)
            or not isinstance(adaptive_policy_count, int)
            or adaptive_policy_count < 1
        ):
            raise ValueError("adaptive policy count must be a positive integer")
        self.controller = controller
        self.immutable_core_count = immutable_core_count
        self.adaptive_policy_count = adaptive_policy_count
        self.fingerprint = fingerprint
        self.warmup_calls = warmup_calls
        self.previous_parameter_sha256: str | None = None
        self.records: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Change only the incumbent index and retain an audit of every actual call."""
        if kwargs or len(args) != 7:
            raise ValueError(
                "refresh wrapper requires the original seven positional controller inputs"
            )
        began = time.perf_counter()
        previous = np.asarray(args[5])
        if previous.shape != () or previous.dtype.kind != "i":
            raise ValueError("previous policy index must be one signed integer scalar")
        requested = int(previous)
        current_hash = self.fingerprint(args[1])
        changed = (
            self.previous_parameter_sha256 is not None
            and current_hash != self.previous_parameter_sha256
        )
        adaptive_incumbent = (
            self.immutable_core_count
            < requested
            <= self.immutable_core_count + self.adaptive_policy_count
        )
        refresh = changed and adaptive_incumbent
        effective = -1 if refresh else requested
        # Preserve the original scalar dtype and JAX weak_type. The selector
        # accepts -1 and clips before indexing; no array out-of-bounds occurs.
        effective_index = args[5] - (requested + 1) if refresh else args[5]
        modified = (*args[:5], effective_index, args[6])
        fingerprint_done = time.perf_counter()
        result = self.controller(*modified)
        completed = time.perf_counter()
        self.records.append(
            {
                "call_index": len(self.records),
                "warmup": len(self.records) < self.warmup_calls,
                "state_sha256": self.fingerprint(args[0]),
                "parameter_sha256": current_hash,
                "preceding_call_parameter_sha256": self.previous_parameter_sha256,
                "parameters_changed": changed,
                "requested_previous_index": requested,
                "effective_previous_index": effective,
                "incumbent_in_adaptive_component": adaptive_incumbent,
                "refreshed": refresh,
                "selected_index": int(np.asarray(result.selected_index)),
                "action": np.asarray(result.action).tolist(),
                "fingerprint_and_dispatch_seconds": fingerprint_done - began,
                "delegated_controller_seconds": completed - fingerprint_done,
                "wrapper_through_delegate_seconds": completed - began,
            }
        )
        self.previous_parameter_sha256 = current_hash
        return result


def build_amendment(protocol: Path, baseline: Path) -> dict[str, Any]:
    """Fix all eight new arms before inspecting any refresh outcomes."""
    original = json.loads(protocol.read_text())
    proposal = original["proposal"]
    if original["sha256"] != digest(proposal):
        raise ValueError("original prospective protocol digest changed")
    binding_path = baseline / "diagnostic_runtime_binding.json"
    binding = json.loads(binding_path.read_text())
    selected = [
        trial
        for trial in proposal["trials"]
        if trial["world_key"] in {"structured_30101", "structured_61001"}
        and trial["cell_id"] in {"eta0.7_lag1_extra0", "eta1_lag1_extra0"}
        and trial["arm"]["arm"] in {"A_BAL", "UNION"}
        and trial["realization"] == "primary"
    ]
    if len(selected) != 8:
        raise ValueError(
            "expected exactly two development geometries by two dynamics by two parents"
        )
    trials = []
    for parent in sorted(
        selected, key=lambda row: (row["world_key"], row["cell_id"], row["arm"]["arm"])
    ):
        parent_record = baseline / parent["trial_id"] / "record.json"
        saved = json.loads(parent_record.read_text())
        if saved["trial"] != parent or saved["summary"]["status"] != "completed":
            raise ValueError("required original comparator is not the exact completed sealed trial")
        identity = {"parent_trial_id": parent["trial_id"], "arm": f"{parent['arm']['arm']}_REFRESH"}
        trials.append(
            {
                **identity,
                "trial_id": digest(identity),
                "parent": parent,
                "immutable_core_count": 16 if parent["arm"]["arm"] == "UNION" else 0,
                "original_record": str(parent_record.resolve()),
                "original_record_sha256": sha(parent_record),
            }
        )
    return {
        "schema": SCHEMA,
        "status": "sealed_before_refresh_outcomes",
        "selection": (
            "diagnostic follow-up to the observed historical-anchor union hysteresis; "
            "all four fixed development world/cells retained; no validation outcomes "
            "used"
        ),
        "new_refresh_outcomes_consulted": False,
        "original_protocol": {
            "path": str(protocol.resolve()),
            "sha256": sha(protocol),
            "proposal_sha256": original["sha256"],
        },
        "original_runtime_binding": {
            "path": str(binding_path.resolve()),
            "sha256": sha(binding_path),
            "value": binding,
        },
        "intervention": {
            "rule": (
                "if complete parameter-byte fingerprint changed and the requested index is "
                "in the adaptive component, pass -1; otherwise preserve the requested index"
            ),
            "parameter_identity": (
                "same dtype/shape/bytes hash as actual episode control_params_sha256, never "
                "object identity or update counter"
            ),
            "conservative_scope": (
                "a change anywhere in the adaptive library invalidates adaptive incumbency; "
                "even a change affecting another skill alone may refresh it"
            ),
            "immutable_indices": (
                "nominal index 0; UNION frozen core 1..16; A_BAL has no frozen core"
            ),
            "unchanged": [
                "all finite updates published",
                "teacher and Adam history",
                "eligibility tests",
                "0.02 score hysteresis when incumbent function is unchanged",
                "physical bounds",
                "QP and held checks",
                "nominal controller",
                "40ms clock",
                "compute reserve",
            ],
            "not_a_claim": (
                "this tests selector continuity across policy changes; it is not recursive "
                "safety or a policy-quality gate"
            ),
        },
        "comparisons": (
            "each refreshed arm versus its exact already retained original parent; eight"
            " new episodes, no rerun or overwrite of original32"
        ),
        "budget": {"new_episode_count": 8, "library_seed": 11, "development_world_cell_count": 4},
        "trials": trials,
        "logging": [
            "requested/effective previous index each decision",
            "parameter and state fingerprints",
            "warmup versus physical call",
            "actual command verification",
            "all QP/held-check outputs",
            "full checkpoint at event,event+.4,event+.8",
            "wrapper overhead separately recorded",
        ],
        "outcomes": (
            "collision, operational violation, full-duration task success and timeout "
            "retained for every arm; no favorable-case selection"
        ),
        "execution": (
            "deterministic diagnostic only; this wrapper's extra host hash cost is "
            "measured, with no paced speed claim"
        ),
        "source_files_sha256": {
            str(Path(__file__).resolve()): sha(Path(__file__)),
            str(ROOT / "tests/test_da_plcbf_actuator_incumbent_refresh.py"): sha(
                ROOT / "tests/test_da_plcbf_actuator_incumbent_refresh.py"
            ),
        },
    }


def load_amendment(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != SCHEMA or value.get("sha256") != digest(value["manifest"]):
        raise ValueError("refresh amendment digest/schema mismatch")
    manifest = value["manifest"]
    for name, expected in manifest["source_files_sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"refresh wrapper/test bytes changed: {name}")
    return manifest


def run(amendment: Path, output: Path, *, reuse_completed: Path | None = None) -> dict[str, Any]:
    """Execute only the separately sealed eight-trial wrapper intervention."""
    import jax

    from benchmark.da_plcbf_actuator_diagnostics import DiagnosticResources, resolve_trial_scene
    from crazyflow.safety.da_plcbf.actuator_experiment import (
        ActuatorEpisodeConfig,
        run_actuator_episode,
    )
    from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
    from crazyflow.safety.da_plcbf.actuator_study import ActuatorObservationConfig

    manifest = load_amendment(amendment)
    original = manifest["original_runtime_binding"]["value"]
    for relative, expected in original["source_sha256"].items():
        actual = sha(ROOT / relative)
        if actual != expected:
            raise ValueError(f"original runtime source changed: {relative}")
    for name, expected in original["checkpoint_files_sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"original full checkpoint changed: {name}")
    protocol_path = Path(manifest["original_protocol"]["path"])
    if sha(protocol_path) != manifest["original_protocol"]["sha256"]:
        raise ValueError("original design file changed")
    proposal = json.loads(protocol_path.read_text())["proposal"]
    output.mkdir(parents=True, exist_ok=False)
    runtime = {
        "amendment": str(amendment.resolve()),
        "amendment_sha256": sha(amendment),
        "original_scientific_runtime_sha256": original["scientific_runtime_sha256"],
        "devices": [str(device) for device in jax.devices()],
        "override": manifest["intervention"],
    }
    _write(output / "refresh_runtime_binding.json", runtime)
    resources, results = DiagnosticResources(), []
    for trial in manifest["trials"]:
        parent = trial["parent"]
        original_record = Path(trial["original_record"])
        if sha(original_record) != trial["original_record_sha256"]:
            raise ValueError("retained comparator record changed")
        old = json.loads(original_record.read_text())
        if reuse_completed is not None:
            prior_runtime = json.loads(
                (reuse_completed / "refresh_runtime_binding.json").read_text()
            )
            prior_amendment = Path(prior_runtime["amendment"])
            previous = json.loads(prior_amendment.read_text())
            if sha(prior_amendment) != prior_runtime["amendment_sha256"]:
                raise ValueError("prior refresh runtime amendment changed")
            if (
                previous["manifest"]["trials"] != manifest["trials"]
                or previous["manifest"]["intervention"] != manifest["intervention"]
            ):
                raise ValueError(
                    "reused refresh requires identical trial identities and intervention"
                )
            prior_directory = reuse_completed / trial["trial_id"]
            prior_record_path = prior_directory / "record.json"
            prior_record = (
                json.loads(prior_record_path.read_text()) if prior_record_path.is_file() else {}
            )
            if prior_record.get("reused_completed_episode"):
                prior_directory = Path(prior_record["episode_directory"]).parent
            summary_path = prior_directory / "episode/summary.json"
            if summary_path.is_file():
                prior_summary = json.loads(summary_path.read_text())
                if prior_summary["status"] != "completed":
                    raise ValueError(
                        "prior interrupted attempt must be retained, not silently rerun"
                    )
                audit_path = prior_directory / "incumbent_audit.json"
                audit = json.loads(audit_path.read_text())
                physical_calls = [item for item in audit if not item["warmup"]]
                with np.load(
                    prior_directory / "episode/controls.npz", allow_pickle=False
                ) as controls:
                    if len(physical_calls) != len(controls["time"]):
                        raise ValueError("reused refresh call/control count mismatch")
                    for index, item in enumerate(physical_calls):
                        if (
                            item["parameter_sha256"]
                            != str(controls["control_params_sha256"][index])
                            or item["state_sha256"]
                            != str(controls["controller_input_state_sha256"][index])
                            or not np.array_equal(
                                item["action"], controls["planned_command"][index]
                            )
                        ):
                            raise ValueError("reused refresh input/command audit mismatch")
                directory = output / trial["trial_id"]
                directory.mkdir()
                record = {
                    "trial": trial,
                    "summary": prior_summary,
                    "episode_directory": str((prior_directory / "episode").resolve()),
                    "refresh_count": sum(item["refreshed"] for item in physical_calls),
                    "actual_input_and_command_verification": True,
                    "reused_completed_episode": True,
                    "original_refresh_amendment_sha256": prior_record.get(
                        "original_refresh_amendment_sha256", prior_runtime["amendment_sha256"]
                    ),
                    "evidence_sha256": {
                        str(path.resolve()): sha(path)
                        for path in prior_directory.rglob("*")
                        if path.is_file()
                    },
                }
                _write(directory / "record.json", record)
                results.append(record)
                print(
                    json.dumps(
                        {
                            "reused_completed": trial["arm"],
                            "termination": prior_summary["termination"],
                        }
                    ),
                    flush=True,
                )
                continue
        binding = json.loads((Path(old["episode_directory"]) / "binding.json").read_text())
        configuration = dict(binding["config"])
        filters = ActuatorFilterConfig(**configuration.pop("filter_config"))
        observation = ActuatorObservationConfig(**configuration.pop("observation_config"))
        configuration.pop("opt_config", None)
        if configuration["execution_mode"] != "deterministic":
            raise ValueError("refresh diagnostic requires the original deterministic contract")
        config = ActuatorEpisodeConfig(
            filter_config=filters, observation_config=observation, **configuration
        )
        scene = resolve_trial_scene(proposal, parent, filters)
        bundle, functions, learner, metadata = resources.resolve(
            parent["arm"]["runtime_method"], filters
        )
        if int(bundle.contract.spec.latent_codes.shape[0]) != 16:
            raise ValueError("sealed refresh comparison requires sixteen adaptive policies")
        if (
            trial["immutable_core_count"]
            and metadata["immutable_core"]["fallback_policy_count"] != 16
        ):
            raise ValueError("sealed union comparison requires sixteen immutable core policies")
        wrapper = IncumbentRefreshController(
            functions.controller,
            immutable_core_count=trial["immutable_core_count"],
            warmup_calls=config.warmup_calls,
        )
        wrapped = functions._replace(controller=wrapper)
        directory = output / trial["trial_id"]
        directory.mkdir()
        print(
            json.dumps(
                {"starting": trial["arm"], "world": parent["world_key"], "cell": parent["cell_id"]}
            ),
            flush=True,
        )
        result = run_actuator_episode(
            scene,
            bundle,
            config,
            directory / "episode",
            controllerfunctions=wrapped,
            learner_functions=learner,
        )
        physical_calls = [record for record in wrapper.records if not record["warmup"]]
        with np.load(directory / "episode/controls.npz", allow_pickle=False) as controls:
            if len(physical_calls) != len(controls["time"]):
                raise ValueError("wrapper physical calls do not match saved controls")
            for index, record in enumerate(physical_calls):
                if record["state_sha256"] != str(
                    controls["controller_input_state_sha256"][index]
                ) or record["parameter_sha256"] != str(controls["control_params_sha256"][index]):
                    raise ValueError("wrapper state/parameter identity differs from actual control")
                if not np.array_equal(
                    np.asarray(record["action"]), controls["planned_command"][index]
                ):
                    raise ValueError("wrapper returned command differs from retained control")
                record.update(
                    control_index=index,
                    time_seconds=float(controls["time"][index]),
                    command_applied=bool(controls["command_applied"][index]),
                )
        _write(directory / "incumbent_audit.json", wrapper.records)
        record = {
            "trial": trial,
            "summary": json.loads((directory / "episode/summary.json").read_text()),
            "method_metadata": metadata,
            "episode_directory": str((directory / "episode").resolve()),
            "refresh_count": sum(item["refreshed"] for item in physical_calls),
            "actual_input_and_command_verification": True,
            "evidence_sha256": {
                str(path.resolve()): sha(path) for path in directory.rglob("*") if path.is_file()
            },
        }
        _write(directory / "record.json", record)
        results.append(record)
        print(
            json.dumps(
                {
                    "finished": trial["arm"],
                    "termination": result.summary["termination"],
                    "waypoints": result.summary["waypoints_completed"],
                    "refresh_count": record["refresh_count"],
                }
            ),
            flush=True,
        )
    summary = {
        "status": "completed"
        if all(r["summary"]["status"] == "completed" for r in results)
        else "incomplete",
        "planned": 8,
        "retained": len(results),
        "records": results,
    }
    _write(output / "results.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--protocol", required=True, type=Path)
    seal.add_argument("--baseline", required=True, type=Path)
    seal.add_argument("--output", required=True, type=Path)
    seal.add_argument("--serialization-revision-of", type=Path)
    execute = commands.add_parser("run")
    execute.add_argument("--amendment", required=True, type=Path)
    execute.add_argument("--output", required=True, type=Path)
    execute.add_argument("--reuse-completed", type=Path)
    args = parser.parse_args()
    if args.command == "seal":
        manifest = build_amendment(args.protocol, args.baseline)
        if args.serialization_revision_of is not None:
            previous = json.loads(args.serialization_revision_of.read_text())["manifest"]
            if (
                previous["trials"] != manifest["trials"]
                or previous["intervention"] != manifest["intervention"]
            ):
                raise ValueError(
                    "serialization revision must preserve the entire trial plan and intervention"
                )
            manifest["serialization_revision"] = {
                "parent_amendment": str(args.serialization_revision_of.resolve()),
                "parent_amendment_sha256": sha(args.serialization_revision_of),
                "reason": (
                    "record writer rejected ndarray after first completed flight; fix only "
                    "serialization and reuse its retained evidence; same eight trials"
                ),
                "new_refresh_outcomes_consulted_for_revision": False,
            }
            manifest["new_refresh_outcomes_consulted"] = True
            manifest["new_refresh_outcomes_used_to_change_design"] = False
            manifest["status"] = "original_eight_trial_design_preserved_serialization_revision"
        envelope = {"schema": SCHEMA, "manifest": manifest, "sha256": digest(manifest)}
        _write(args.output, envelope)
        print(
            json.dumps(
                {
                    "path": str(args.output),
                    "sha256": envelope["sha256"],
                    "budget": manifest["budget"],
                }
            )
        )
    else:
        result = run(args.amendment, args.output, reuse_completed=args.reuse_completed)
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
