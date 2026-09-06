"""Isolated same-observation K-size cost probes; no changed-K competence claim.

Example (run only when the selected device is free):
    python -m benchmark.da_plcbf_actuator_ablation_profile --platform cpu \
        --checkpoint /absolute/path/to/nominal128-seed11/deployment \
        --output /absolute/path/to/fresh/k-cost-cpu-v1

The source K retains its exact checkpoint and Adam state. Smaller K deterministically
slices policy identities and offsets, retaining the shared network, and starts fresh
Adam. Larger K constructs a fresh Fibonacci actor and its own nominal teacher with
fresh Adam. These changed libraries are untrained cost inputs, not deployments.
All repeated learner proposals start from the same state and are discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


def _write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _tree_arrays(tree: Any) -> dict[str, Any]:
    import jax
    import numpy as np

    return {f"leaf_{i:04d}": np.asarray(leaf) for i, leaf in enumerate(jax.tree.leaves(tree))}


def _tree_hash(tree: Any) -> str:
    digest = hashlib.sha256()
    for name, array in _tree_arrays(tree).items():
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _prepare_library(checkpoint: Any, count: int, seed: int) -> tuple[Any, ...]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from crazyflow.safety.da_plcbf.actuator_learning import (
        actuator_reference_fingerprint,
        build_actuator_skill_learner,
        initialize_actuator_skill_actor,
    )
    from crazyflow.safety.da_plcbf.persistent_skill_learner import build_fibonacci_skill_spec

    source_count = int(checkpoint.contract.spec.latent_codes.shape[0])
    contract = checkpoint.contract
    indices = None
    if count == source_count:
        params = checkpoint.state.params
        initialization = "exact_source_checkpoint_including_persistent_adam"
    elif count < source_count:
        # Nested even slices for K=4/8 of K=16; this is an explicit identity subset.
        indices = (np.arange(count) * source_count // count).tolist()
        selected = jnp.asarray(indices, dtype=jnp.int32)
        spec = jax.tree.map(lambda leaf: leaf[selected], contract.spec)

        def subset(actor: Any) -> Any:
            return actor.replace(
                velocity_offsets=actor.velocity_offsets[selected],
                duration_offsets=actor.duration_offsets[selected],
            )

        params = subset(checkpoint.state.params)
        contract = replace(contract, params=subset(contract.params), spec=spec)
        initialization = "even_policy_subset_and_shared_source_network_with_fresh_adam"
    else:
        velocities = np.asarray(contract.spec.base_desired_velocities)
        speeds = np.linalg.norm(velocities, axis=1)
        durations = np.asarray(contract.spec.base_durations)
        spec = build_fibonacci_skill_spec(
            policy_count=count,
            latent_size=int(contract.spec.latent_codes.shape[1]),
            minimum_speed=float(speeds.min()),
            maximum_speed=float(speeds.max()),
            minimum_duration=float(durations.min()),
            maximum_duration=float(durations.max()),
            horizon_duration=checkpoint.config.horizon * checkpoint.config.dt,
            dtype=contract.spec.latent_codes.dtype,
        )
        params = initialize_actuator_skill_actor(jax.random.key(seed), spec, checkpoint.config)
        contract = replace(contract, params=params, spec=spec)
        initialization = "fresh_fibonacci_actor_and_nominal_teacher_with_fresh_adam"
    learner = build_actuator_skill_learner(contract, checkpoint.config)
    state = (
        checkpoint.state if count == source_count else learner.initialize(params, contract.model)
    )
    metadata = {
        "policy_count": count,
        "controller_candidate_count_including_nominal": count + 1,
        "source_policy_count": source_count,
        "initialization": initialization,
        "source_policy_indices": indices,
        "fresh_initialization_seed": seed if count > source_count else None,
        "actor_parameters_equal_source": count == source_count,
        "competence_scope": "source_checkpoint_evidence_only_not_retested"
        if count == source_count
        else "unavailable_for_changed_K",
        "changed_k_competence": "unavailable" if count != source_count else "not_changed",
        "performance_evaluation_performed": False,
        "initial_gradient_steps": int(state.cumulative_gradient_steps),
        "initial_library_version": int(state.library_version),
        "reference_sha256": actuator_reference_fingerprint(contract),
        "initial_state_sha256": _tree_hash(state),
        "initial_params_sha256": _tree_hash(state.params),
        "initial_optimizer_sha256": _tree_hash(state.optimizer_state),
    }
    return contract, learner, state, metadata


def _observations(checkpoint: Any, filter_config: Any) -> list[dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from crazyflow.safety.da_plcbf.actuator_study import make_actuator_scene

    probes = []
    for family, seed in (("structured", 2003), ("navigation", 2103)):
        scene = make_actuator_scene(
            seed,
            family,
            "combined",
            dt=filter_config.dt,
            control_period=filter_config.command_period,
        )
        for phase, when in (("nominal", 0.0), ("combined", scene.event_time + 0.24)):
            # A fixed development-bank body/motor transient, translated to the scene origin.
            # It is deliberately not represented as an executed flight trajectory.
            state = np.asarray(checkpoint.contract.anchors[26]).copy()
            state[:3] = scene.world.initial_state[:3]
            model = scene.model_at(when, checkpoint.contract.model)
            prediction = scene.world.obstacle_prediction(
                when, dt=filter_config.dt, horizon=filter_config.horizon
            )
            safety = scene.world.safety_limits(when)
            goal = scene.world.waypoint_positions[0]
            inputs = (
                jnp.asarray(state),
                model,
                prediction,
                safety,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(goal, dtype=checkpoint.contract.spec.latent_codes.dtype),
            )
            jax.block_until_ready(inputs)
            probes.append(
                {
                    "label": f"{family}_{phase}",
                    "inputs": inputs,
                    "metadata": {
                        "family": family,
                        "scene_seed": seed,
                        "split": "validation_cost_only",
                        "phase": phase,
                        "observation_time_seconds": float(when),
                        "state_source": "development_anchor26_translated_to_scene_initial_position",
                        "physical_trajectory_executed": False,
                        "input_sha256": _tree_hash(inputs),
                    },
                }
            )
    return probes


def _plot(output: Path, counts: list[int], probes: list[dict[str, Any]], stats: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for axis, component in zip(axes, ("controller", "learner"), strict=True):
        for probe in probes:
            label = probe["label"]
            rows = [stats[str(count)][label][component] for count in counts]
            (line,) = axis.plot(
                counts, [1000 * row["mean_seconds"] for row in rows], marker="o", label=label
            )
            axis.plot(
                counts,
                [1000 * row["p95_seconds"] for row in rows],
                linestyle="--",
                color=line.get_color(),
                alpha=0.7,
            )
        axis.set(title=f"Isolated {component}: mean / dashed p95", xlabel="Fallback policy count K")
        axis.set_ylabel("Synchronized service (ms)")
        axis.set_xticks(counts)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("K-size cost probes; changed-K competence unavailable; no flight claim")
    fig.tight_layout()
    fig.savefig(output / "k_cost.png", dpi=180)
    fig.savefig(output / "k_cost.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--platform", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--counts", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--repetitions", type=int, default=40)
    parser.add_argument("--warmup-calls", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 10 or args.warmup_calls < 2:
        parser.error("require at least ten repetitions and two disposable warmups")
    if args.counts != sorted(set(args.counts)) or any(count < 2 for count in args.counts):
        parser.error("counts must be distinct increasing integers of at least two")
    # Set platform before importing any numerical runtime; a CPU request cannot initialize GPU.
    os.environ["JAX_PLATFORMS"] = args.platform
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import numpy as np

    from benchmark.da_plcbf_actuator_study import runner_source_files, timing_summary
    from crazyflow.safety.da_plcbf.actuator_compute import PackedActuatorController
    from crazyflow.safety.da_plcbf.actuator_learning import load_actuator_learner_checkpoint
    from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
    from crazyflow.safety.da_plcbf.actuator_study import build_actuator_controller

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = load_actuator_learner_checkpoint(args.checkpoint)
    source_count = int(checkpoint.contract.spec.latent_codes.shape[0])
    if source_count not in args.counts:
        parser.error("counts must include the actual source checkpoint policy count")
    if checkpoint.contract.anchors.shape[0] <= 26:
        parser.error("the source checkpoint must retain development anchor26")
    filter_config = ActuatorFilterConfig(
        dt=checkpoint.config.dt,
        horizon=checkpoint.config.horizon,
        command_hold_steps=checkpoint.config.control_interval_steps,
        gradient_mode="forward",
    )
    root = Path(__file__).resolve().parents[1]
    source_hashes = {}
    for path in [*runner_source_files(root), Path(__file__).resolve()]:
        payload = path.read_bytes()
        relative = path.relative_to(root)
        saved = output / "source_snapshot" / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(payload)
        source_hashes[str(relative)] = hashlib.sha256(payload).hexdigest()
    binding = {
        "scope": "isolated_same_observation_cost_only",
        "checkpoint_stem": str(args.checkpoint.resolve()),
        "checkpoint_npz_sha256": checkpoint.sha256,
        "checkpoint_json_sha256": hashlib.sha256(checkpoint.json_path.read_bytes()).hexdigest(),
        "source_policy_count": source_count,
        "counts": args.counts,
        "platform_requested": args.platform,
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "actor_config": asdict(checkpoint.config),
        "reference_learning_config": asdict(checkpoint.contract.learning_config),
        "filter_config": asdict(filter_config),
        "warmup_calls": args.warmup_calls,
        "repetitions": args.repetitions,
        "source_sha256": source_hashes,
        "controller_scope": (
            "full 17-state forward-AD filter plus packed device-to-host result; "
            "inputs pre-resident; excludes observation, obstacle prediction, "
            "nominal diagnostics, simulation and auditing"
        ),
        "learner_scope": (
            "synchronized full learner step from one fixed persistent input; no proposals installed"
        ),
        "limitations": [
            "changed-K initialization and filter branches differ; no pure K causal effect",
            "no changed-K competence validation, retraining, flight evaluation, or real-time claim",
            "isolated durations do not measure a serialized controller-plus-learner schedule",
        ],
    }
    _write_json(output / "binding.json", binding)
    probes = _observations(checkpoint, filter_config)
    _write_json(output / "observations.json", [probe["metadata"] for probe in probes])
    for probe in probes:
        np.savez_compressed(
            output / f"observation_{probe['label']}.npz", **_tree_arrays(probe["inputs"])
        )
    runtimes, initializations, cold, branches = {}, {}, {}, {}
    with (output / "progress.jsonl").open("x") as progress:
        for count in args.counts:
            began = time.perf_counter()
            contract, learner, state, metadata = _prepare_library(checkpoint, count, args.seed)
            functions = build_actuator_controller(contract.spec, checkpoint.config, filter_config)
            controller = PackedActuatorController(functions.controller)
            runtimes[count] = (controller, learner, state)
            initializations[str(count)] = metadata
            np.savez_compressed(
                output / f"K{count}_cost_inputs.npz",
                **_tree_arrays((state, contract.spec, contract.params)),
            )
            cold[str(count)] = {
                "library_teacher_setup_and_input_serialization_seconds": time.perf_counter() - began
            }
            branches[str(count)] = {}
            for probe in probes:
                y, model, prediction, safety, previous, goal = probe["inputs"]
                began = time.perf_counter()
                result = controller(y, state.params, model, prediction, safety, previous, goal)
                first_controller = time.perf_counter() - began
                began = time.perf_counter()
                proposal = learner.step(state, y, model)
                jax.block_until_ready(proposal)
                first_learner = time.perf_counter() - began
                cold[str(count)][probe["label"]] = {
                    "first_controller_call_seconds": first_controller,
                    "first_learner_call_seconds": first_learner,
                    "includes_any_required_tracing_compilation_and_execution": True,
                }
                branches[str(count)][probe["label"]] = {
                    "mode_index": int(result.execution_mode),
                    "selected_index": int(result.selected_index),
                    "input_valid": bool(result.certificates.input_valid),
                    "eligible_policy_count": int(np.sum(result.certificates.eligible)),
                    "qp_valid": bool(result.qp_valid),
                    "sqp_iterations": int(result.sqp_iterations),
                    "learner_finite_update": bool(proposal[1].finite_update_applied),
                    "packed_leaf_count": controller.last_leaf_count,
                    "packed_buffer_count": controller.last_buffer_count,
                }
                for _ in range(args.warmup_calls):
                    controller(y, state.params, model, prediction, safety, previous, goal)
                    jax.block_until_ready(learner.step(state, y, model))
            event = {"event": "K_prepared", "count": count, **metadata}
            progress.write(json.dumps(event, sort_keys=True) + "\n")
            progress.flush()
            print(json.dumps({"event": "K_prepared", "count": count}), flush=True)
    _write_json(output / "initializations.json", initializations)
    _write_json(output / "cold_calls.json", cold)
    _write_json(output / "controller_branches.json", branches)
    rows = []
    with (output / "timing_trace.jsonl").open("x") as stream:
        for repetition in range(args.repetitions):
            offset = repetition % len(args.counts)
            order = args.counts[offset:] + args.counts[:offset]
            if (repetition // len(args.counts)) % 2:
                order = list(reversed(order))
            for count in order:
                controller, learner, state = runtimes[count]
                for probe in probes:
                    y, model, prediction, safety, previous, goal = probe["inputs"]
                    operations = {
                        "controller": lambda: controller(
                            y, state.params, model, prediction, safety, previous, goal
                        ),
                        "learner": lambda: jax.block_until_ready(learner.step(state, y, model)),
                    }
                    for component in (
                        ("controller", "learner")
                        if repetition % 2 == 0
                        else ("learner", "controller")
                    ):
                        began = time.perf_counter()
                        operations[component]()
                        seconds = time.perf_counter() - began
                        row = {
                            "repetition": repetition,
                            "K": count,
                            "observation": probe["label"],
                            "component": component,
                            "seconds": seconds,
                        }
                        rows.append(row)
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        stream.flush()
    stats = {
        str(count): {
            probe["label"]: {
                component: timing_summary(
                    [
                        row["seconds"]
                        for row in rows
                        if row["K"] == count
                        and row["observation"] == probe["label"]
                        and row["component"] == component
                    ]
                )
                for component in ("controller", "learner")
            }
            for probe in probes
        }
        for count in args.counts
    }
    for count, (_, _, state) in runtimes.items():
        if _tree_hash(state) != initializations[str(count)]["initial_state_sha256"]:
            raise RuntimeError("a cost probe modified its immutable initial learner state")
    summary = {
        **binding,
        "status": "completed_cost_profile",
        "proposals_published": 0,
        "timings": stats,
        "initializations": initializations,
        "controller_branches": branches,
    }
    _write_json(output / "summary.json", summary)
    _plot(output, args.counts, probes, stats)
    print(json.dumps({"event": "completed_cost_profile", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
