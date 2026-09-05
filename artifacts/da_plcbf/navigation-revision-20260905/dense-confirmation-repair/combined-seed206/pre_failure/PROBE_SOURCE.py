"""Reconstruct adaptive updates and compare libraries at either recorded method's states."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.continuous_version_a import QP_REJECTION_REASONS
from crazyflow.safety.da_plcbf.learner_checkpoint import (
    load_learner_checkpoint,
    save_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.navigation_experiment import (
    NavigationExperimentConfig,
    build_navigation_controller,
)
from crazyflow.safety.da_plcbf.navigation_world import build_navigation_world
from crazyflow.safety.da_plcbf.state_conditioned_learning import (
    build_reference_skill_learner_from_checkpoint,
)
from examples.da_plcbf.navigation_demo import load_world_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=100)
    parser.add_argument("--verification-index", type=int, default=181)
    parser.add_argument("--indices", type=int, nargs="+", default=(165, 176))
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--expected-package-root", type=Path)
    parser.add_argument("--anchor", choices=("fixed", "adaptive"), default="fixed")
    args = parser.parse_args()
    package_root = Path(sys.modules["crazyflow"].__file__).resolve().parent.parent
    if args.expected_package_root and package_root != args.expected_package_root.resolve():
        raise ValueError(f"numerical package loaded from unexpected source: {package_root}")
    if not all(args.start_index <= index <= args.verification_index for index in args.indices):
        raise ValueError("all probe indices must lie in the reconstructed snapshot interval")
    args.output.mkdir(parents=True, exist_ok=False)
    script_path = Path(__file__).resolve()
    inputs = [
        args.run / name
        for name in (
            "navigation_comparison.json",
            "navigation_comparison.npz",
            "raw_diagnostics.npz",
            "config.json",
            "world.json",
            "schedule.json",
        )
    ]
    for prefix in (
        args.checkpoint,
        args.run / f"snapshots/{args.start_index:04d}-published",
        args.run / f"snapshots/{args.verification_index:04d}-published",
    ):
        prefix = prefix.with_suffix("") if prefix.suffix == ".npz" else prefix
        inputs.extend(Path(f"{prefix}.{suffix}") for suffix in ("npz", "json"))
    provenance = {
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "command": shlex.join([sys.executable, str(script_path), *sys.argv[1:]]),
        "numerical_package_root": str(package_root),
        "package_source_sha256": {
            str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((package_root / "crazyflow").rglob("*.py"))
        },
        "inputs_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
        },
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
    }
    (args.output / "PROBE_SOURCE.py").write_bytes(script_path.read_bytes())
    (args.output / "PROBE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    device = jax.devices(args.device)[0]
    metadata = json.loads((args.run / "navigation_comparison.json").read_text())
    config = NavigationExperimentConfig(**json.loads((args.run / "config.json").read_text()))
    world = build_navigation_world(load_world_config(args.run / "world.json"))
    initial = load_learner_checkpoint(args.checkpoint, device=device)
    start, _, learner = build_reference_skill_learner_from_checkpoint(
        args.run / f"snapshots/{args.start_index:04d}-published", device=device
    )
    verification = load_learner_checkpoint(
        args.run / f"snapshots/{args.verification_index:04d}-published", device=device
    )
    schedule = json.loads((args.run / "schedule.json").read_text())["opportunities"]
    with np.load(args.run / "navigation_comparison.npz") as archive:
        trace = {name: archive[name] for name in archive.files}
    with np.load(args.run / "raw_diagnostics.npz") as archive:
        raw = {name: archive[name] for name in archive.files}
    state, pending, captured, updates = start.state, None, {}, 0
    for index in range(args.start_index, args.verification_index + 1):
        if pending is not None:
            state, pending = pending, None
        if index in args.indices:
            captured[index] = state
        if index == args.verification_index:
            break
        if schedule[index] and trace["adaptive_recorded_control_valid"][index]:
            point = jax.device_put(
                world.dynamics_at(float(trace["time_seconds"][index]), initial.point_model).model,
                device,
            )
            pending, metrics = jax.block_until_ready(
                learner.step(
                    state, jax.device_put(trace["adaptive_full_state"][index], device), point
                )
            )
            if bool(metrics.finite_update_applied) != bool(raw["adaptive_finite_update"][index]):
                raise AssertionError(
                    "reconstructed finite-update flag differs from recorded schedule"
                )
            updates += 1
    differences = [
        float(np.max(np.abs(np.asarray(a).astype(float) - np.asarray(b).astype(float))))
        for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(verification.state), strict=True)
    ]
    exact = all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(verification.state), strict=True)
    )
    parity = {
        "start_index": args.start_index,
        "verification_index": args.verification_index,
        "updates_replayed": updates,
        "full_parameter_optimizer_model_state_bitwise_equal": exact,
        "maximum_leaf_abs_error": max(differences),
        "device": str(device),
    }
    (args.output / "reconstruction_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    if not exact:
        raise AssertionError(f"full continuation state mismatch: {parity}")
    print(json.dumps(parity), flush=True)
    controller = build_navigation_controller(world, initial, config)
    arrays, rows = {}, []
    names = metadata["held_operational_constraint_names"]
    for index in args.indices:
        when = float(trace["time_seconds"][index])
        physical = jax.device_put(trace[f"{args.anchor}_full_state"][index], device)
        goal = jax.device_put(trace[f"{args.anchor}_goal_position"][index], device)
        previous = jax.device_put(
            jnp.asarray(raw[f"{args.anchor}_selected_index"][index - 1], dtype=jnp.int32), device
        )
        model = jax.device_put(world.dynamics_at(when, initial.point_model).model, device)
        obstacles = world.obstacle_prediction(when, horizon=initial.config.horizon)
        arrays[f"{index}_{args.anchor}_physical_state"] = np.asarray(physical)
        arrays[f"{index}_adaptive_training_path_state"] = trace["adaptive_full_state"][index]
        arrays[f"{index}_obstacle_centers"] = np.asarray(obstacles.centers)
        arrays[f"{index}_obstacle_velocities"] = np.asarray(obstacles.velocities)
        for name, value in zip(model._fields, model, strict=True):
            arrays[f"{index}_point_model_{name}"] = np.asarray(value)
        for label, snapshot in (("fixed", initial.state), ("adaptive", captured[index])):
            save_learner_checkpoint(
                snapshot,
                initial.spec,
                initial.config,
                initial.actuator,
                physical,
                args.output / f"{index:04d}-{label}-at-{args.anchor}-state",
                metadata={
                    "simulation_time": when,
                    "source_run": str(args.run),
                    "scope": (
                        f"Recorded {args.anchor} physical anchor; adaptive parameters "
                        "reconstructed only from its original proprioceptive path"
                    ),
                    "previous_policy_index": int(previous),
                    "goal_for_external_replay_only": np.asarray(goal).tolist(),
                },
            )
            decision = jax.block_until_ready(
                controller(physical, snapshot.params, model, obstacles, previous, goal)
            )
            held = np.asarray(decision.qp_held_operational_residuals)
            finite_held = np.isfinite(held)
            substep, constraint = np.unravel_index(
                np.argmin(np.where(finite_held, held, np.inf)), held.shape
            )
            row = {
                "control_index": index,
                "simulation_time": when,
                "library": label,
                "library_version": int(snapshot.library_version),
                "anchor": f"{args.anchor} physical state/model/goal/selection history",
                "fallback_max_hard": float(np.max(decision.values.values[1:])),
                "augmented_max_hard": float(np.max(decision.values.values)),
                "eligible_count": int(decision.eligible_candidate_count),
                "accepted_qp": bool(decision.qp_valid),
                "used_fallback": bool(decision.used_fallback),
                "degraded": bool(decision.degraded),
                "applied_wrench": np.asarray(decision.action).tolist(),
                "qp_rejection_reasons": [
                    name
                    for name, flag in zip(
                        QP_REJECTION_REASONS, np.asarray(decision.qp_rejection_flags), strict=True
                    )
                    if flag
                ],
                "minimum_qp_held_operational_residual": float(held[substep, constraint])
                if np.any(finite_held)
                else None,
                "minimum_held_operational_constraint": names[constraint]
                if np.any(finite_held)
                else None,
                "minimum_held_operational_substep": int(substep) if np.any(finite_held) else None,
                "nonfinite_qp_held_operational_entries": int(np.sum(~finite_held)),
                "qp_minimum_scope": "finite proposal entries only; raw NaN/inf retained in NPZ",
                "held_operational_constraint_caused_rejection": bool(
                    decision.qp_rejection_flags[-1]
                ),
                "applied_held_operational_residual": float(
                    decision.applied_held_operational_residual
                ),
                "recorded_fixed_wrench_max_abs_error": (
                    float(
                        np.max(
                            np.abs(
                                np.asarray(decision.action) - trace["fixed_applied_wrench"][index]
                            )
                        )
                    )
                    if label == "fixed" and args.anchor == "fixed"
                    else None
                ),
                "recorded_anchor_wrench_max_abs_error": (
                    float(
                        np.max(
                            np.abs(
                                np.asarray(decision.action)
                                - trace[f"{args.anchor}_applied_wrench"][index]
                            )
                        )
                    )
                    if label == args.anchor
                    else None
                ),
            }
            rows.append(row)
            for field in ("states", "wrenches", "valid"):
                arrays[f"{index}_{label}_candidate_{field}"] = np.asarray(
                    getattr(decision.candidates, field)
                )
            arrays[f"{index}_{label}_hard_values"] = np.asarray(decision.values.values)
            arrays[f"{index}_{label}_smooth_values"] = np.asarray(decision.smooth_values)
            arrays[f"{index}_{label}_qp_held_operational_residuals"] = held
            for field in (
                "fallback_held_operational_residuals",
                "applied_held_operational_residuals",
                "applied_held_physical_margins",
            ):
                arrays[f"{index}_{label}_{field}"] = np.asarray(getattr(decision, field))
            print(json.dumps(row, allow_nan=False), flush=True)
    np.savez_compressed(args.output / "same_state_full_fixtures.npz", **arrays)
    (args.output / "same_state_comparison.json").write_text(
        json.dumps(
            {
                "reconstruction": parity,
                "chronology": (
                    f"Libraries are queried at the same recorded {args.anchor}-controller states. "
                    "Adaptive parameters are reconstructed from their original training path. "
                    "Event order alone does not establish disturbance-recovery necessity."
                ),
                "world_events": {
                    kind: metadata["summary"]["world"]["config"][kind]
                    for kind in ("wind_events", "payload_events")
                },
                "queried_time_seconds": [float(trace["time_seconds"][i]) for i in args.indices],
                "rows": rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
