"""Fixed numerical repair probes and a 36-flight development factorial.

All worlds are previously inspected development cases. This driver never opens new worlds.
Old result files remain immutable; legacy modes are new executions, not historical replays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.safety.da_plcbf.actuator_observation import ActuatorObservationConfig

from benchmark.da_plcbf_actuator_diagnostics import (
    DiagnosticResources,
    resolve_trial_scene,
    source_binding,
)
from benchmark.da_plcbf_actuator_pd_loss_origin import EPISODE, _saved_case
from crazyflow.safety.da_plcbf.actuator_experiment import (
    ActuatorEpisodeConfig,
    run_actuator_episode,
)
from crazyflow.safety.da_plcbf.actuator_learning import build_actuator_skill_learner
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig, _normalized_qp_with_audit

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/da_plcbf/numerical-stabilization-20260906/v1"
OLD_PROTOCOL = ROOT / "artifacts/da_plcbf/actuator-diagnostic-20260906/v1/protocol-sealed-v1.json"
CASES = (
    ("harm", "structured_30101", "eta0.7_lag1_extra0"),
    ("gain", "navigation_62104", "eta0.7_lag1_extra0"),
    ("no_change_regression", "navigation_62104", "eta1_lag1_extra0"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sources() -> dict[str, str]:
    return {
        **source_binding(),
        str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        "benchmark/da_plcbf_actuator_pd_loss_origin.py": sha(
            ROOT / "benchmark/da_plcbf_actuator_pd_loss_origin.py"
        ),
    }


def seal(path: Path) -> None:
    if path.exists():
        raise ValueError("protocol already exists")
    old = json.loads(OLD_PROTOCOL.read_text())["proposal"]
    trials = []
    for case, world, cell in CASES:
        original = next(
            t
            for t in old["trials"]
            if t["world_key"] == world
            and t["cell_id"] == cell
            and t["arm"]["arm"] == "A_BAL"
            and t["realization"] == "primary"
        )
        for qp in ("legacy", "inward"):
            for method in ("PD_F", "F2", "A_BAL", "ONSET_FREEZE"):
                for learner in (
                    ("legacy", "matched") if method in {"A_BAL", "ONSET_FREEZE"} else ("matched",)
                ):
                    identity = f"{case}__{method}__{learner}__{qp}"
                    trials.append(
                        {
                            "id": identity,
                            "case": case,
                            "method": method,
                            "learner_numerics": learner,
                            "qp_numerics": qp,
                            "original_trial": original,
                        }
                    )
    assert len(trials) == 36
    resources = DiagnosticResources()
    files = {str(OLD_PROTOCOL): sha(OLD_PROTOCOL)}
    for method in ("PD_F", "F2", "A_BAL"):
        bundle = resources.bundle(method)
        for p in (bundle.npz_path, bundle.json_path):
            files[str(p)] = sha(p)
    proposal = {
        "schema": "numerical_stabilization_v1",
        "trials": trials,
        "source_sha256": sources(),
        "file_sha256": files,
        "old_proposal": old,
        "maximum_flights": 36,
        "probe_builds": 2,
        "probe_updates_per_history": 16,
        "scope": "Previously observed development cases; no held-out evidence or objective tuning.",
        "primary_methods": ["Handcrafted frozen", "Learned frozen", "Learned adaptive"],
        "auxiliary": "Onset freeze for causal attribution; PD tuning only in numerical probes",
        "all_finite_updates_publish": True,
        "tolerance_weakening": False,
    }
    write(path, proposal)
    snapshot = path.parent / "source"
    for name in proposal["source_sha256"]:
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    print(json.dumps({"protocol": str(path), "sha256": sha(path), "flights": 36}), flush=True)


def load(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    assert sources() == protocol["source_sha256"], "numerical source changed after sealing"
    for name, digest in protocol["file_sha256"].items():
        assert sha(Path(name)) == digest, name
    return protocol


def numeric(value: Any) -> Any:
    if hasattr(value, "_asdict"):
        return {k: numeric(v) for k, v in value._asdict().items()}
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        return None
    return array.item() if array.ndim == 0 else array.tolist()


def probe(protocol: Path, output: Path) -> None:
    plan = load(protocol)
    output.mkdir(parents=True, exist_ok=False)
    bundle, inputs, metadata = _saved_case(EPISODE)
    state_input = inputs["runtime"]["initial_state"]
    resources = DiagnosticResources()
    rows = []
    for family, b in [("handcrafted", bundle), ("learned", resources.bundle("A_BAL"))]:
        for mode in ("legacy", "matched"):
            learner = build_actuator_skill_learner(b.contract, b.config, reference_numerics=mode)
            for history in ("zero_momentum", "persistent"):
                state = (
                    learner.initialize(b.contract.params, b.contract.model)
                    if history == "zero_momentum"
                    else b.state
                )
                initial_params = state.params
                for index in range(plan["probe_updates_per_history"]):
                    before = time.perf_counter()
                    state, metric = learner.step(state, state_input, b.contract.model)
                    jax.block_until_ready(state)
                    rows.append(
                        {
                            "family": family,
                            "numerics": mode,
                            "history": history,
                            "iteration": index,
                            "metrics": numeric(metric),
                            "service_seconds": time.perf_counter() - before,
                            "max_parameter_displacement": max(
                                float(np.max(np.abs(np.asarray(a) - np.asarray(z))))
                                for a, z in zip(
                                    jax.tree.leaves(state.params),
                                    jax.tree.leaves(initial_params),
                                    strict=True,
                                )
                            ),
                        }
                    )
    model = bundle.contract.model
    row = jnp.array(
        [156.4965362548828, 218.12234497070312, -108.88327026367188, -147.86111450195312],
        jnp.float32,
    )
    bounds = [
        jnp.nextafter(jnp.float32(40.992610931396484), jnp.float32(direction))
        for direction in (-np.inf, np.inf)
    ]
    bounds.insert(1, jnp.float32(40.992610931396484))
    qp_rows = []
    for mode in ("legacy", "inward"):
        solve = jax.jit(
            lambda command, b: _normalized_qp_with_audit(
                command, row, b, model, ActuatorFilterConfig(qp_numerics=mode)
            )
        )
        for delta in np.linspace(-1e-5, 1e-5, 11):
            for bound in bounds:
                command = jnp.array([0.19, 0.19, 0.05, 0.05], jnp.float32) + delta
                result, audit = solve(command, bound)
                jax.block_until_ready(result)
                qp_rows.append(
                    {
                        "mode": mode,
                        "perturbation": float(delta),
                        "bound": float(bound),
                        "result": numeric(result),
                        "audit": numeric(audit),
                        "raw_policy_residual": numeric(bound - row @ result.action),
                    }
                )
    write(
        output / "report.json",
        {
            "protocol_sha256": sha(protocol),
            "pid": os.getpid(),
            "devices": [str(d) for d in jax.devices()],
            "jax_version": jax.__version__,
            "compilation_cache": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
            "original_pd_input_metadata": metadata,
            "fused_updates": rows,
            "qp_probes": qp_rows,
            "qualification": (
                "Fused production updates; immutable teacher. "
                "Compilation calls excluded from speed claims."
            ),
        },
    )
    assert all(
        r["metrics"]["gradient_norm"] == 0 and r["max_parameter_displacement"] == 0
        for r in rows
        if r["numerics"] == "matched" and r["history"] == "zero_momentum"
    )
    assert all(
        r["result"]["feasible"]
        and r["audit"]["executed_rows_passed"]
        and r["raw_policy_residual"] >= -2e-6
        for r in qp_rows
        if r["mode"] == "inward"
    )
    load(protocol)
    print(
        json.dumps({"status": "verified", "fused_updates": len(rows), "qp_probes": len(qp_rows)}),
        flush=True,
    )


def flights(protocol: Path, output: Path) -> None:
    plan = load(protocol)
    output.mkdir(parents=True, exist_ok=False)
    resources = DiagnosticResources()
    learners = {}
    records = []
    write(
        output / "environment.json",
        {
            "devices": [str(d) for d in jax.devices()],
            "pid": os.getpid(),
            "jax_version": jax.__version__,
            "compilation_cache": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
        },
    )
    for trial in plan["trials"]:
        load(protocol)
        directory = output / trial["id"]
        directory.mkdir(exist_ok=False)
        write(directory / "claim.json", trial)
        configuration = dict(plan["old_proposal"]["shared_control"])
        filters = ActuatorFilterConfig(
            **{**configuration.pop("filter_config"), "qp_numerics": trial["qp_numerics"]}
        )
        observation = ActuatorObservationConfig(**configuration.pop("observation_config", {}))
        scene = resolve_trial_scene(plan["old_proposal"], trial["original_trial"], filters)
        method = "A_BAL" if trial["method"] == "ONSET_FREEZE" else trial["method"]
        configuration.update(
            method=method,
            execution_mode="deterministic",
            freeze_learning_at=2.0 if trial["method"] == "ONSET_FREEZE" else None,
            reference_numerics=trial["learner_numerics"],
            capture_times=(2.0, 2.4, 2.8),
            save_checkpoints=True,
            retain_rollouts=trial["case"] == "harm"
            and trial["qp_numerics"] == "inward"
            and trial["learner_numerics"] == "matched"
            and trial["method"] != "ONSET_FREEZE",
        )
        config = ActuatorEpisodeConfig(
            filter_config=filters, observation_config=observation, **configuration
        )
        bundle, controller, _, method_metadata = resources.resolve(method, filters)
        learner = None
        if method == "A_BAL":
            key = (method, trial["learner_numerics"])
            if key not in learners:
                learners[key] = build_actuator_skill_learner(
                    bundle.contract, bundle.config, reference_numerics=trial["learner_numerics"]
                )
            learner = learners[key]
        print(json.dumps({"starting": trial["id"], "completed": len(records)}), flush=True)
        result = run_actuator_episode(
            scene,
            bundle,
            config,
            directory / "attempt-00",
            controllerfunctions=controller,
            learner_functions=learner,
        )
        episode = directory / "attempt-00"
        record = {
            "trial": trial,
            "summary": result.summary,
            "method_metadata": method_metadata,
            "episode_directory": str(episode),
            "protocol_sha256": sha(protocol),
            "evidence_sha256": {str(p): sha(p) for p in episode.rglob("*") if p.is_file()},
        }
        write(directory / "record.json", record)
        records.append(record)
        write(output / "results.json", {"planned": 36, "records": records})
        print(
            json.dumps(
                {
                    "finished": trial["id"],
                    "termination": result.summary["termination"],
                    "safe_task": result.summary["successful_full_episode"],
                }
            ),
            flush=True,
        )
    load(protocol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "probe", "flights"))
    parser.add_argument("--protocol", type=Path, default=BASE / "protocol.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "seal":
        seal(args.protocol)
    else:
        if args.output is None:
            parser.error("--output is required")
        (probe if args.command == "probe" else flights)(args.protocol, args.output)


if __name__ == "__main__":
    main()
