"""Bounded compile-only scan-unroll parity and paired warmed learner timing.

No timing proposal is published. All configurations use identical checkpoint parameters,
optimizer history, physical models, initial states and immutable nominal teacher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import numpy as np

from benchmark.da_plcbf_actuator_behavior import validation_state_bank
from benchmark.da_plcbf_actuator_recovery import changed_model
from crazyflow.safety.da_plcbf.actuator_learning import (
    actuator_reference_fingerprint,
    build_actuator_skill_learner,
    load_actuator_learner_checkpoint,
    rollout_actuator_skill_library,
)


def _write_json(path: Path, values: Any) -> None:
    with path.open("x") as stream:
        json.dump(values, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _tree_arrays(tree: Any) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree.leaves(tree)]


def _parity(actual: Any, reference: Any, *, rtol: float, atol: float) -> dict[str, Any]:
    first, second = _tree_arrays(actual), _tree_arrays(reference)
    if jax.tree.structure(actual) != jax.tree.structure(reference):
        raise ValueError("parity tree structures differ")
    result = []
    for left, right in zip(first, second, strict=True):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("parity array shapes/dtypes differ")
        numeric = np.issubdtype(left.dtype, np.inexact)
        if numeric:
            delta = left.astype(np.float64) - right.astype(np.float64)
            difference = float(np.max(np.abs(delta))) if delta.size else 0.0
            rel = float(np.linalg.norm(delta) / max(1e-12, np.linalg.norm(right)))
            passed = np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False)
        else:
            difference, rel = 0.0, 0.0
            passed = np.array_equal(left, right)
        result.append(
            {
                "shape": list(left.shape),
                "max_abs": difference,
                "relative_l2": rel,
                "pass": bool(passed),
            }
        )
    return {
        "pass": all(row["pass"] for row in result),
        "rtol": rtol,
        "atol": atol,
        "leaves": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", type=int, choices=(2, 4), default=2)
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--braking-huber-delta", type=float, default=0.0)
    args = parser.parse_args()
    if args.pairs < 10:
        parser.error("at least ten timing pairs are required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = load_actuator_learner_checkpoint(args.checkpoint)
    contract = replace(
        checkpoint.contract,
        learning_config=replace(
            checkpoint.contract.learning_config,
            reference_braking_huber_delta=args.braking_huber_delta,
        ),
    )
    reference_hash = actuator_reference_fingerprint(contract)
    validation, _ = validation_state_bank(contract.model)
    probes = (
        ("nominal_development26", contract.anchors[26], contract.model),
        ("combined_development26", contract.anchors[26], changed_model(contract.model, "combined")),
        ("combined_validation7", validation[7], changed_model(contract.model, "combined")),
    )
    root = Path(__file__).resolve().parents[1]
    source_hashes = {}
    for name in (
        "benchmark/da_plcbf_actuator_unroll.py",
        "crazyflow/safety/da_plcbf/actuator_learning.py",
        "crazyflow/safety/da_plcbf/actuator_dynamics.py",
    ):
        payload = (root / name).read_bytes()
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        source_hashes[name] = hashlib.sha256(payload).hexdigest()
    runtimes, numerical, compiled_seconds = {}, {}, {}
    configurations = (1, args.candidate)
    for unroll in configurations:
        config = replace(checkpoint.config, rollout_scan_unroll=unroll)
        begin = time.perf_counter()
        learner = build_actuator_skill_learner(contract, config)
        forward = jax.jit(
            lambda params, state, model, config=config: rollout_actuator_skill_library(
                params, contract.spec, state, model, config
            )
        )
        gradient = jax.jit(jax.value_and_grad(learner.loss, has_aux=True))
        records = {}
        for label, state, model in probes:
            trajectory = forward(checkpoint.state.params, state, model)
            loss_gradient = gradient(
                checkpoint.state.params,
                state,
                model,
                checkpoint.state.previous_params,
                checkpoint.state.library_version,
            )
            proposal = learner.step(checkpoint.state, state, model)
            jax.block_until_ready((trajectory, loss_gradient, proposal))
            records[label] = (trajectory, loss_gradient, proposal)
            arrays = {}
            for category, tree in zip(
                ("rollout", "loss_gradient", "next_state_and_metrics"), records[label], strict=True
            ):
                arrays.update(
                    {
                        f"{category}_{index:03d}": value
                        for index, value in enumerate(_tree_arrays(tree))
                    }
                )
            np.savez_compressed(output / f"unroll{unroll}_{label}.npz", **arrays)
        compiled_seconds[str(unroll)] = time.perf_counter() - begin
        numerical[unroll] = records
        runtimes[unroll] = learner
        print(json.dumps({"event": "unroll_compiled", "unroll": unroll}), flush=True)
    comparisons = {}
    for label, _, _ in probes:
        baseline = numerical[1][label]
        candidate = numerical[args.candidate][label]
        comparisons[label] = {
            "forward": _parity(candidate[0], baseline[0], rtol=2e-5, atol=2e-5),
            "loss_and_raw_gradients": _parity(candidate[1], baseline[1], rtol=5e-4, atol=2e-5),
            "next_adam_state": _parity(candidate[2][0], baseline[2][0], rtol=5e-4, atol=5e-6),
            "step_metrics": _parity(candidate[2][1], baseline[2][1], rtol=5e-4, atol=2e-5),
        }
    _write_json(output / "parity.json", comparisons)
    timing_rows = []
    state, model = probes[1][1:]
    # Warm all variants before paired alternating timing; every call starts from the same state.
    for unroll in configurations:
        for _ in range(4):
            jax.block_until_ready(runtimes[unroll].step(checkpoint.state, state, model))
    for pair in range(args.pairs):
        order = configurations if pair % 2 == 0 else tuple(reversed(configurations))
        for unroll in order:
            begin = time.perf_counter()
            proposal = runtimes[unroll].step(checkpoint.state, state, model)
            jax.block_until_ready(proposal)
            elapsed = time.perf_counter() - begin
            timing_rows.append({"pair": pair, "unroll": unroll, "seconds": elapsed})
    _write_json(output / "timing_trace.json", timing_rows)
    timings = {}
    for unroll in configurations:
        values = [row["seconds"] for row in timing_rows if row["unroll"] == unroll]
        timings[str(unroll)] = {
            "mean_seconds": float(np.mean(values)),
            "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.percentile(values, 95)),
            "maximum_seconds": max(values),
        }
    baseline, candidate = timings["1"], timings[str(args.candidate)]
    parity_pass = all(v["pass"] for probe in comparisons.values() for v in probe.values())
    reduction = 1 - candidate["mean_seconds"] / baseline["mean_seconds"]
    selected = (
        parity_pass and reduction >= 0.10 and candidate["p95_seconds"] <= baseline["p95_seconds"]
    )
    assert actuator_reference_fingerprint(contract) == reference_hash
    summary = {
        "candidate": args.candidate,
        "selected_candidate": selected,
        "selection_rule": (
            "all parity checks pass, >=10% paired mean step reduction, no p95 regression"
        ),
        "numerical_parity_pass": parity_pass,
        "mean_learner_time_reduction": reduction,
        "timings": timings,
        "compile_and_probe_seconds": compiled_seconds,
        "checkpoint_stem": str(Path(args.checkpoint).resolve()),
        "checkpoint_npz_sha256": checkpoint.sha256,
        "reference_sha256_before_and_after": reference_hash,
        "original_reference_sha256": actuator_reference_fingerprint(checkpoint.contract),
        "objective": asdict(contract.learning_config),
        "teacher_actor_config": asdict(contract.actor_config),
        "current_actor_configs": {
            str(unroll): asdict(replace(checkpoint.config, rollout_scan_unroll=unroll))
            for unroll in configurations
        },
        "proposals_published": 0,
        "timing_scope": "synchronized isolated full learner step, same immutable input each time",
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "source_sha256_by_path": source_hashes,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
