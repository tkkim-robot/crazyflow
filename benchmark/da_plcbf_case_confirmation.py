"""Frozen case-family confirmation and causal learner branches, preserving all outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import numpy as np

from benchmark.da_plcbf_case_attribution import (
    encounter_from_row,
    persistent_provenance,
    run_branch,
)
from benchmark.da_plcbf_case_discovery import (
    CASE_CHECKPOINTS,
    validate_atlas_branch_snapshot,
    write_json,
)
from crazyflow.safety.da_plcbf.case_study_world import build_hover_encounter_world
from crazyflow.safety.da_plcbf.learner_checkpoint import load_learner_checkpoint
from crazyflow.safety.da_plcbf.navigation_experiment import (
    NavigationExperimentConfig,
    build_navigation_controller,
)
from crazyflow.safety.da_plcbf.state_conditioned_learning import (
    build_reference_skill_learner_from_checkpoint,
)


def critical_skill_from_controls(rows: list[dict]) -> int:
    """Use an executed positive-dual learned constraint, rather than the maximum cached H."""
    for row in rows:
        if row["executed"] and row["qp"] and row["selected"] > 0 and row["dual"] > 1e-7:
            return int(row["selected"]) - 1
    raise ValueError("critical-skill ablation requires a recorded learned positive-dual QP")


def confirm_case(
    selected_directory: Path, atlas: Path, output: Path, *, neighbors: int, seed: int, device: Any
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    row = json.loads((selected_directory / "selection.json").read_text())
    case, anchor = row["case"], row["anchor"]
    source = load_learner_checkpoint(CASE_CHECKPOINTS[case], device=device)
    bundle, contract, learner = build_reference_skill_learner_from_checkpoint(
        atlas / case / anchor, device=device
    )
    available = validate_atlas_branch_snapshot(bundle, row["time_seconds"])
    controls_path = selected_directory / "adaptive_held_controls.json"
    critical_skill = critical_skill_from_controls(json.loads(controls_path.read_text()))
    cfg = encounter_from_row(row)
    # Frozen BEFORE confirmation draws/outcomes. Perturbed states/wind are branch diagnostics,
    # not mislabeled reexecutions of a prefix in different dynamics.
    protocol = {
        "selected_development_case": str(selected_directory),
        "selection": row,
        "seed": seed,
        "neighbor_count": neighbors,
        "geometry_offset_uniform_m": [-0.015, 0.015],
        "arrival_shift_uniform_s": [-0.04, 0.04],
        "radius_shift_uniform_m": [-0.005, 0.005],
        "initial_position_uniform_m": [-0.005, 0.005],
        "initial_velocity_uniform_mps": [-0.01, 0.01],
        "wind_shift_uniform_mps": [-0.05, 0.05],
        "wind_perturbation_scope": (
            "new common point-model change at branch; saved prefix unchanged"
        ),
        "critical_skill_zero_based": critical_skill,
        "critical_skill_selection_rule": (
            "first executed learned positive-dual QP in selected branch"
        ),
        "critical_skill_source_controls": str(controls_path),
        "critical_skill_source_sha256": hashlib.sha256(controls_path.read_bytes()).hexdigest(),
        "snapshot": str(atlas / case / anchor),
        "snapshot_sha256": bundle.sha256,
        "frozen_checkpoint_sha256": source.sha256,
        "snapshot_available_at": available,
        "controller": "production full filter, unchanged emergency, same .04s hold",
        "continue_after_envelope_breach": True,
        "stop_event": "oriented asset geometry intersection; not an artificial motor cut",
    }
    write_json(output / "protocol.json", protocol)
    world = build_hover_encounter_world(
        cfg,
        initial_state=np.asarray(bundle.physical_state),
        initial_time_seconds=row["time_seconds"],
    )
    runconfig = NavigationExperimentConfig(
        navigation_start_seconds=cfg.navigation_start_seconds,
        fallback_mapping="compensated"
        if source.config.model_compensation
        else "matched_uncompensated",
    )
    controller = build_navigation_controller(world, source, runconfig)
    mixed = build_navigation_controller(
        world,
        source,
        runconfig,
        frozen_replacement=(source.state.params, protocol["critical_skill_zero_based"]),
    )
    results, arrays = {}, {}
    variants = (
        ("frozen", source.state, controller, None, 1),
        ("adapted_held", bundle.state, controller, None, 1),
        ("continued_learning", bundle.state, controller, learner, 1),
        ("critical_skill_original", bundle.state, mixed, None, 1),
        ("frozen_plant_dt_010", source.state, controller, None, 2),
        ("adapted_plant_dt_010", bundle.state, controller, None, 2),
    )
    for name, persistent, control, learning, substeps in variants:
        summary, rows, dense, times, _ = run_branch(
            world,
            source,
            persistent,
            control,
            learner=learning,
            snapshot_available=0 if name.startswith("frozen") else available,
            end=row["time_seconds"] + 3,
            plant_substeps=substeps,
            provenance_directory=output / "continued_learner" if learning is not None else None,
            reference_contract=contract if learning is not None else None,
        )
        results[name] = summary
        write_json(output / f"{name}_controls.json", rows)
        arrays[f"{name}_states"], arrays[f"{name}_times"] = dense, times
        print(
            name,
            "shell",
            summary["geometry_audit"]["safety_shell"]["minimum_clearance_m"],
            "degraded",
            summary["degraded_controls"],
            flush=True,
        )
    write_json(output / "causal_branches.json", results)
    np.savez_compressed(output / "causal_dense_states.npz", **arrays)
    rng = np.random.default_rng(seed)
    ledger = []
    for index in range(neighbors):
        offset = np.asarray(cfg.incoming.crossing_offset) + rng.uniform(-0.015, 0.015, 3)
        initial = np.asarray(bundle.physical_state).copy()
        initial[:3] += rng.uniform(-0.005, 0.005, 3)
        initial[7:10] += rng.uniform(-0.01, 0.01, 3)
        changed = replace(
            cfg,
            incoming=replace(
                cfg.incoming,
                crossing_offset=tuple(offset),
                arrival_time_seconds=cfg.incoming.arrival_time_seconds + rng.uniform(-0.04, 0.04),
                radius_m=cfg.incoming.radius_m + rng.uniform(-0.005, 0.005),
            ),
            wind_velocity=tuple(np.asarray(cfg.wind_velocity) + rng.uniform(-0.05, 0.05, 3)),
        )
        neighbor = build_hover_encounter_world(
            changed, initial_state=initial, initial_time_seconds=row["time_seconds"]
        )
        record = {"index": index, "world": neighbor.metadata(), "methods": {}}
        for name, persistent in (("frozen", source.state), ("adapted_held", bundle.state)):
            summary, rows, dense, times, _ = run_branch(
                neighbor,
                source,
                persistent,
                controller,
                snapshot_available=0 if name == "frozen" else available,
                end=row["time_seconds"] + 3,
            )
            record["methods"][name] = summary
            # Compact complete decisive interval, no large rollout tensor.
            write_json(output / f"neighbor-{index:03d}-{name}-controls.json", rows)
            np.savez_compressed(
                output / f"neighbor-{index:03d}-{name}-dense.npz", states=dense, times=times
            )
        ledger.append(record)
        write_json(output / "neighbor_ledger.json", ledger)
        print(
            "neighbor",
            index,
            {
                n: round(s["geometry_audit"]["safety_shell"]["minimum_clearance_m"], 5)
                for n, s in record["methods"].items()
            },
            flush=True,
        )


def rerun_continuation_with_provenance(
    confirmation_directory: Path, output: Path, *, device: Any
) -> None:
    """Reproduce only a prior continued-learning branch and compare its observable history.

    Original confirmation protocols and outcomes are never overwritten. The new run retains
    every publication/update hash and complete initial/final Adam checkpoints with their exact
    nominal-reference binding. It is a deterministic reproduction, not a paced availability test.
    """
    protocol_path = confirmation_directory / "protocol.json"
    original_protocol = json.loads(protocol_path.read_text())
    row = original_protocol["selection"]
    source = load_learner_checkpoint(CASE_CHECKPOINTS[row["case"]], device=device)
    bundle, contract, learner = build_reference_skill_learner_from_checkpoint(
        original_protocol["snapshot"], device=device
    )
    available = validate_atlas_branch_snapshot(bundle, row["time_seconds"])
    if (
        source.sha256 != original_protocol["frozen_checkpoint_sha256"]
        or bundle.sha256 != original_protocol["snapshot_sha256"]
    ):
        raise ValueError("continuation must use the original confirmation's exact checkpoints")
    output.mkdir(parents=True, exist_ok=False)
    inputs = [
        protocol_path,
        confirmation_directory / "continued_learning_controls.json",
        confirmation_directory / "causal_dense_states.npz",
        confirmation_directory / "causal_branches.json",
        Path(__file__),
        Path("benchmark/da_plcbf_case_attribution.py"),
    ]
    write_json(
        output / "protocol.json",
        {
            "scope": "deterministic continued-learning provenance reproduction only",
            "original_confirmation_directory": str(confirmation_directory),
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
            },
            "original_protocol_unchanged": True,
            "snapshot_source": original_protocol["snapshot"],
            "snapshot_source_sha256": bundle.sha256,
            "frozen_source_sha256": source.sha256,
            "initial_persistent_identity": persistent_provenance(bundle.state),
            "initial_snapshot_available_seconds": available,
            "publication_schedule": (
                "completed finite updates become available at the next .04s boundary"
            ),
            "wall_clock_scope": "synchronized service observations; not a paced deployment claim",
        },
    )
    cfg = encounter_from_row(row)
    world = build_hover_encounter_world(
        cfg,
        initial_state=np.asarray(bundle.physical_state),
        initial_time_seconds=row["time_seconds"],
    )
    runconfig = NavigationExperimentConfig(
        navigation_start_seconds=cfg.navigation_start_seconds,
        fallback_mapping="compensated"
        if source.config.model_compensation
        else "matched_uncompensated",
    )
    controller = build_navigation_controller(world, source, runconfig)
    summary, rows, dense, times, _ = run_branch(
        world,
        source,
        bundle.state,
        controller,
        end=row["time_seconds"] + 3,
        learner=learner,
        snapshot_available=available,
        provenance_directory=output / "learner",
        reference_contract=contract,
    )
    write_json(output / "summary.json", summary)
    write_json(output / "controls.json", rows)
    np.savez_compressed(output / "dense_states.npz", states=dense, times=times)
    original_rows = json.loads(
        (confirmation_directory / "continued_learning_controls.json").read_text()
    )
    with np.load(confirmation_directory / "causal_dense_states.npz") as original:
        original_states = original["continued_learning_states"]
        original_times = original["continued_learning_times"]
    numeric = {}
    for name, before, after in (
        ("dense_states", original_states, dense),
        ("dense_times", original_times, times),
        *(
            (
                name,
                np.asarray([r[name] for r in original_rows]),
                np.asarray([r[name] for r in rows]),
            )
            for name in ("time", "state", "action", "hard", "smooth", "dual")
        ),
    ):
        matching_shape = before.shape == after.shape
        numeric[name] = {
            "same_shape": matching_shape,
            "array_equal": bool(matching_shape and np.array_equal(before, after)),
            "maximum_absolute_difference": float(np.max(np.abs(before - after)))
            if matching_shape
            else None,
        }
    discrete = {
        name: [r.get(name) for r in original_rows] == [r.get(name) for r in rows]
        for name in (
            "executed",
            "version",
            "snapshot_available_time",
            "selected",
            "mode",
            "qp",
            "degraded",
            "eligible",
            "finite_update",
        )
    }
    updates = [r["completed_update"] for r in rows if "completed_update" in r]
    for index, observed in enumerate(rows):
        if "completed_update" not in observed:
            continue
        update = observed["completed_update"]
        assert update["before"] == observed["published_learner_state"]
        if index + 1 < len(rows):
            assert update["after"] == rows[index + 1]["published_learner_state"]
            assert (
                update["completed_perf_counter_ns"]
                <= rows[index + 1]["controller_started_perf_counter_ns"]
            )
            if update["finite_update_applied"]:
                assert (
                    update["publication_time_seconds"] == rows[index + 1]["snapshot_available_time"]
                )
    initial = load_learner_checkpoint(output / "learner/initial_checkpoint", device=device)
    final = load_learner_checkpoint(output / "learner/final_checkpoint", device=device)
    assert persistent_provenance(initial.state) == persistent_provenance(bundle.state)
    assert persistent_provenance(final.state) == summary["continuation_provenance"]["final"]
    comparison = {
        "original_observable_history": str(confirmation_directory),
        "numeric_history_comparison": numeric,
        "discrete_history_equal": discrete,
        "all_observed_history_equal": all(x["array_equal"] for x in numeric.values())
        and all(discrete.values()),
        "initial_checkpoint_persistent_state_exact": True,
        "final_checkpoint_matches_final_published_state": True,
        "each_completed_state_hash_matches_next_publication": True,
        "each_completed_update_precedes_next_control_wall_clock": True,
        "update_count": len(updates),
        "finite_update_count": sum(u["finite_update_applied"] for u in updates),
        "initial_library_version": int(initial.state.library_version),
        "final_library_version": int(final.state.library_version),
        "last_executed_control_version": summary["last_used_version"],
        "initial_checkpoint_npz_sha256": initial.sha256,
        "final_checkpoint_npz_sha256": final.sha256,
        "limitation": (
            "Original run had no per-update parameter hashes; observable equality is checked, "
            "not nonexistent original hashes."
        ),
    }
    write_json(output / "REPRODUCTION_COMPARISON.json", comparison)
    print(
        row["case"],
        "continued provenance",
        comparison["all_observed_history_equal"],
        "updates",
        len(updates),
        "versions",
        comparison["initial_library_version"],
        comparison["final_library_version"],
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-directory", type=Path)
    parser.add_argument("--atlas", type=Path)
    parser.add_argument("--continue-provenance-only", type=Path, metavar="CONFIRMATION_DIRECTORY")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--seed", type=int, default=27581)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    if args.continue_provenance_only is not None:
        rerun_continuation_with_provenance(
            args.continue_provenance_only, args.output_dir, device=jax.devices(args.device)[0]
        )
        return
    if args.selected_directory is None or args.atlas is None:
        parser.error("confirmation requires --selected-directory and --atlas")
    confirm_case(
        args.selected_directory,
        args.atlas,
        args.output_dir,
        neighbors=args.neighbors,
        seed=args.seed,
        device=jax.devices(args.device)[0],
    )


if __name__ == "__main__":
    main()
