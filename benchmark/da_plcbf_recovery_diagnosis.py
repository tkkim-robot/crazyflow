"""Targeted recovery-loss diagnosis using the immutable corrected development runs.

Saved-state reconstruction must match every published parameter hash before any
counterfactual is evaluated. Diagnostic interventions are not online update gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.da_plcbf_actuator_confirmation import _recorded_inputs, load_saved_run
from benchmark.da_plcbf_actuator_diagnostics import DiagnosticResources, source_binding
from crazyflow.safety.da_plcbf.actuator_experiment import _filter_record, _hash_tree, _jsonable
from crazyflow.safety.da_plcbf.actuator_learning import (
    build_actuator_skill_learner,
    load_actuator_learner_checkpoint,
    save_actuator_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import (
    ActuatorFilterConfig,
    actuator_plcbf_step,
    check_actuator_hold,
)

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts/da_plcbf/numerical-stabilization-20260906/v1"
BASE = ROOT / "artifacts/da_plcbf/recovery-interaction-20260907/v1"
HARM = OLD / "development-36-v3/harm__A_BAL__matched__inward/attempt-00"
REPLAY = BASE / "reconstruct-v1"
DIAGNOSTIC_HARM = BASE / "instrumented-harm-v1/attempt-00"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def begin(output: Path, scope: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    sources = source_binding() | {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    for name in sources:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    write(
        output / "execution.json",
        {
            "scope": scope,
            "source_sha256": sources,
            "pid": os.getpid(),
            "devices": [str(d) for d in jax.devices()],
            "jax_version": jax.__version__,
            "compilation_cache": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
        },
    )


def reconstruct(output: Path) -> None:
    begin(output, "Replay saved learner inputs; no physical flight or parameter interpolation.")
    run = load_saved_run(HARM)
    bundle = load_actuator_learner_checkpoint(HARM / "snapshots/initial")
    learner = build_actuator_skill_learner(bundle.contract, bundle.config)
    state = bundle.state
    assert _hash_tree(state) == run.binding["initial_learner_sha256"]
    archive = output / "snapshots"
    archive.mkdir()
    rows = []
    for index, when in enumerate(run.controls["time"]):
        params_hash = _hash_tree(state.params)
        assert params_hash == str(run.controls["control_params_sha256"][index]), (index, when)
        assert int(state.library_version) == int(run.controls["control_library_version"][index])
        if when >= 2.0 - 1e-10:
            save_actuator_learner_checkpoint(
                state,
                bundle.contract,
                run.controls["actual_state"][index],
                archive / f"control-{index:04d}",
                config=bundle.config,
                metadata={
                    "source_episode": str(HARM),
                    "control_index": index,
                    "time": float(when),
                    "published_params_authenticated": True,
                },
            )
        rows.append(
            {
                "time": float(when),
                "control_index": index,
                "parameters_sha256": params_hash,
                "learner_sha256": _hash_tree(state),
                "published_version": int(state.library_version),
            }
        )
        if run.controls["learner_completion_credited"][index]:
            y, model, *_ = _recorded_inputs(run, index, float(when))
            state, metric = jax.block_until_ready(learner.step(state, y, model))
            assert bool(metric.finite_update_applied) == bool(
                run.controls["learner_finite_update_applied"][index]
            )
    final = load_actuator_learner_checkpoint(HARM / "snapshots/final")
    assert _hash_tree(state) == _hash_tree(final.state)
    write(
        output / "report.json",
        {
            "rows": rows,
            "all_published_parameters_exact": True,
            "final_complete_learner_exact": True,
            "source_evidence_sha256": run.source_sha256,
        },
    )
    print(
        json.dumps({"reconstructed_controls": len(rows), "final_learner_exact": True}), flush=True
    )


def snapshot(index: int) -> Any:
    return load_actuator_learner_checkpoint(
        DIAGNOSTIC_HARM / f"snapshots/critical-{index * 0.04:.9f}"
    )


def common_states(output: Path) -> None:
    begin(
        output, "Current, previous-publication and onset libraries at identical saved harm states."
    )
    run = load_saved_run(DIAGNOSTIC_HARM)
    resources = DiagnosticResources()
    fc = ActuatorFilterConfig(**run.binding["config"]["filter_config"])
    bundle, functions, _, _ = resources.resolve("A_BAL", fc)
    onset = snapshot(run.boundary_index(2.0))

    def forced(
        y: Any,
        params: Any,
        model: Any,
        obstacles: Any,
        safety: Any,
        previous: Any,
        goal: Any,
        target: Any,
    ) -> Any:
        def masked(state: Any, dynamics: Any) -> Any:
            batch = functions.candidates(state, params, goal, dynamics)
            return batch._replace(valid=batch.valid & (jnp.arange(len(batch.valid)) == target))

        return actuator_plcbf_step(
            y, masked, model, obstacles, safety, functions.emergency(y, model), previous, fc
        )

    force = jax.jit(forced)
    held = jax.jit(
        lambda y, command, model, obstacles, safety: check_actuator_hold(
            y, command, model, obstacles, safety, fc
        )
    )
    rows = []
    for index, when in enumerate(run.controls["time"]):
        if when < 2.4 - 1e-10:
            continue
        current, previous_publication = snapshot(index), snapshot(index - 1)
        y, model, obstacles, safety, previous, goal, _ = _recorded_inputs(run, index, float(when))
        row = {"time": float(when), "control_index": index, "libraries": {}}
        for label, selected in (
            ("adaptive", current),
            ("previous_publication", previous_publication),
            ("onset", onset),
        ):
            step = functions.controller(
                y, selected.state.params, model, obstacles, safety, previous, goal
            )
            record = _filter_record(step, False)
            record["action"] = np.asarray(step.action)
            cert = step.certificates
            eligible = np.flatnonzero(cert.eligible).tolist()
            if label == "adaptive":
                # Require exact control reproduction before interpreting counterfactuals.
                record["recorded_action_exact"] = np.array_equal(
                    step.action, run.controls["planned_command"][index]
                )
                record["recorded_eligibility_exact"] = np.array_equal(
                    cert.eligible, run.controls["eligible"][index]
                )
                record["recorded_row_max_delta"] = np.max(
                    np.abs(
                        np.asarray(cert.rows[int(step.selected_index)])
                        - run.controls["selected_row"][index]
                    )
                )
                assert record["recorded_action_exact"] and record["recorded_eligibility_exact"], (
                    when,
                    label,
                )
            branches = []
            for target in eligible:
                candidate_hold = jax.block_until_ready(
                    held(y, cert.rollouts.commands[target, 0], model, obstacles, safety)
                )
                # Evaluate all candidates through unchanged QP/SQP and nonlinear checks.
                branch = jax.block_until_ready(
                    force(
                        y,
                        selected.state.params,
                        model,
                        obstacles,
                        safety,
                        previous,
                        goal,
                        jnp.asarray(target),
                    )
                )
                branches.append(
                    {
                        "index": target,
                        "selected_as_requested": int(branch.selected_index) == target,
                        "qp_valid": bool(branch.qp_valid),
                        "fallback_valid": bool(branch.fallback_valid),
                        "direct_hold_passed": bool(candidate_hold.passed),
                        "direct_hold_collision_margin": float(candidate_hold.collision_margin),
                        "direct_hold_operational_min": float(
                            jnp.min(candidate_hold.operational_margins)
                        ),
                        "rejection_flags": np.asarray(branch.qp_rejection_flags),
                    }
                )
            record["branches"] = branches
            record["eligible_qp_count"] = sum(b["qp_valid"] for b in branches)
            record["eligible_direct_hold_count"] = sum(b["direct_hold_passed"] for b in branches)
            record["params_sha256"] = _hash_tree(selected.state.params)
            row["libraries"][label] = record
        rows.append(row)
        write(output / f"state-{index:04d}.json", row)
        print(
            json.dumps(
                {
                    "time": float(when),
                    "eligible": {
                        k: v["eligible_policy_count"] for k, v in row["libraries"].items()
                    },
                }
            ),
            flush=True,
        )
    write(
        output / "report.json",
        {
            "rows": rows,
            "source_evidence_sha256": run.source_sha256,
            "instrumented_execution_comparison_sha256": sha(
                DIAGNOSTIC_HARM.parent / "comparison.json"
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reconstruct", "common"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (reconstruct if args.stage == "reconstruct" else common_states)(args.output)


if __name__ == "__main__":
    main()
