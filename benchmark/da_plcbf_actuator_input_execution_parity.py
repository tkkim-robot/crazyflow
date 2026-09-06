"""Diagnose exact cached/reference inputs and repeated GPU executions, without episodes."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.da_plcbf_actuator_confirmation import (
    _sha,
    _world_from_binding,
    _write_json,
    load_saved_run,
)
from crazyflow.safety.da_plcbf.actuator_compute import PackedActuatorController
from crazyflow.safety.da_plcbf.actuator_experiment import _filter_record, _hash_tree
from crazyflow.safety.da_plcbf.actuator_inputs import CausalObservationInputCache, reference_inputs
from crazyflow.safety.da_plcbf.actuator_learning import (
    build_actuator_skill_learner,
    load_actuator_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_study import (
    ActuatorObservationConfig,
    ActuatorScene,
    build_actuator_controller,
    nominal_actuator_model,
)


def leaf_description(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    abstract = jax.typeof(value)
    return {
        "shape": array.shape,
        "dtype": str(array.dtype),
        "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "weak_type": bool(getattr(abstract, "weak_type", False)),
        "sharding": str(getattr(value, "sharding", None)),
        "layout": str(getattr(value, "format", getattr(value, "layout", None))),
    }


def compare(left: Any, right: Any) -> dict[str, Any]:
    lhs, lt = jax.tree.flatten_with_path(left)
    rhs, rt = jax.tree.flatten_with_path(right)
    rows = []
    for (path, a), (_, b) in zip(lhs, rhs, strict=True):
        ad, bd = leaf_description(a), leaf_description(b)
        av, bv = np.asarray(a), np.asarray(b)
        finite = np.isfinite(av) & np.isfinite(bv)
        maximum = (
            float(np.max(np.abs(av[finite].astype(np.float64) - bv[finite].astype(np.float64))))
            if np.any(finite)
            else None
        )
        rows.append(
            {
                "path": jax.tree_util.keystr(path),
                "left": ad,
                "right": bd,
                "max_absolute_difference": maximum,
                "bytes_equal": all(ad[k] == bd[k] for k in ("shape", "dtype", "bytes_sha256")),
                "abstract_and_device_equal": all(
                    ad[k] == bd[k] for k in ("weak_type", "sharding", "layout")
                ),
            }
        )
    return {
        "tree_equal": lt == rt,
        "leaf_count": len(rows),
        "all_bytes_equal": lt == rt and all(r["bytes_equal"] for r in rows),
        "all_abstract_and_device_equal": lt == rt
        and all(r["abstract_and_device_equal"] for r in rows),
        "leaves": rows,
    }


def executable(function: Any, args: tuple[Any, ...], output: Path, name: str) -> Any:
    lower = function.lower(*args)
    stable = str(lower.compiler_ir(dialect="stablehlo"))
    (output / f"{name}.stablehlo.txt").write_text(stable)
    compiled = lower.compile()
    (output / f"{name}.optimized_hlo.txt").write_text(compiled.as_text())
    return compiled


def diagnose(episode: Path, checkpoint: Path, output: Path, repetitions: int, copies: int) -> dict:
    run = load_saved_run(episode)
    bundle = load_actuator_learner_checkpoint(checkpoint)
    if (
        bundle.sha256 != run.binding["checkpoint"]["checkpoint_sha256"]
        or _hash_tree(bundle.state) != run.binding["initial_learner_sha256"]
    ):
        raise ValueError("checkpoint must be the episode's exact initial full learner")
    scene = ActuatorScene(
        world=_world_from_binding(run),
        **{
            key: value
            for key, value in run.binding["scene"].items()
            if key not in {"world", "physical_spec", "physical_world_id"}
        },
    )
    cfg = ActuatorFilterConfig(**run.binding["config"]["filter_config"])
    observation = ActuatorObservationConfig(**run.binding["config"]["observation_config"])
    nominal = nominal_actuator_model()
    actual = run.dense["state"][0]
    reference = reference_inputs(actual, 0.0, scene, nominal, observation, cfg)
    cache = CausalObservationInputCache(scene, nominal, observation, cfg)
    cache.at(actual, 0.0)  # The episode prepares its first templates during warmup.
    cached = cache.at(actual, 0.0)
    inputs = compare(reference, cached)
    if not inputs["all_bytes_equal"] or not inputs["all_abstract_and_device_equal"]:
        return {"status": "completed", "input_parity_passed": False, "inputs": inputs}
    goal = jnp.asarray(run.controls["goal"][0])
    previous = jnp.asarray(0, jnp.int32)

    def controller_args(values: Any) -> tuple[Any, ...]:
        return (
            jnp.asarray(values.observed_state),
            bundle.state.params,
            values.model,
            values.prediction,
            values.safety,
            previous,
            goal,
        )

    def learner_args(values: Any) -> tuple[Any, ...]:
        return bundle.state, jnp.asarray(values.observed_state), values.model

    control_inputs = (controller_args(reference), controller_args(cached))
    learn_inputs = (learner_args(reference), learner_args(cached))
    reports, prior_control, prior_learn = [], None, None
    for copy in range(copies):
        functions = build_actuator_controller(bundle.contract.spec, bundle.config, cfg)
        packed = PackedActuatorController(functions.controller)
        entry = packed._entry(control_inputs[0], {})
        control = executable(entry.function, control_inputs[0], output, f"copy-{copy}.controller")
        learner = build_actuator_skill_learner(bundle.contract, bundle.config)
        learn = executable(learner.step, learn_inputs[0], output, f"copy-{copy}.learner")
        control_results, learn_results = [], []
        for repeat in range(repetitions):
            for kind, args, largs in zip(
                ("reference", "cached"), control_inputs, learn_inputs, strict=True
            ):
                c = entry.schema.unpack(jax.device_get(control(*args)))
                learned = jax.device_get(learn(*largs))
                control_results.append(c)
                learn_results.append(learned)
                _write_json(
                    output / f"copy-{copy}.repeat-{repeat}.{kind}.decision.json",
                    _filter_record(c, retain_rollouts=False),
                )
                _write_json(
                    output / f"copy-{copy}.repeat-{repeat}.{kind}.update.json",
                    {
                        "full_learner_sha256": _hash_tree(learned[0]),
                        "params_sha256": _hash_tree(learned[0].params),
                        "gradient_norm": float(learned[1].gradient_norm),
                        "parameter_update_norm": float(learned[1].parameter_update_norm),
                    },
                )
        reports.append(
            {
                "copy": copy,
                "controller_repeated_comparisons": [
                    compare(control_results[0], value) for value in control_results[1:]
                ],
                "learner_repeated_comparisons": [
                    compare(learn_results[0], value) for value in learn_results[1:]
                ],
                "controller_previous_copy_comparison": None
                if prior_control is None
                else compare(prior_control, control_results[0]),
                "learner_previous_copy_comparison": None
                if prior_learn is None
                else compare(prior_learn, learn_results[0]),
            }
        )
        prior_control, prior_learn = control_results[0], learn_results[0]
    return {
        "status": "completed",
        "input_parity_passed": True,
        "inputs": inputs,
        "controller_inputs": compare(*control_inputs),
        "learner_inputs": compare(*learn_inputs),
        "copies": reports,
        "source_sha256": run.source_sha256,
        "scope": "Initial-state offline calls only; one fixed executable per copy; "
        "each learner call restarts the exact original full learner; no episode claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", default=3, type=int)
    parser.add_argument("--copies", default=2, type=int)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    _write_json(
        args.output / "request.json",
        {
            "arguments": vars(args),
            "script_sha256": _sha(Path(__file__)),
            "environment": {
                key: os.environ.get(key)
                for key in ("XLA_FLAGS", "JAX_ENABLE_X64", "JAX_COMPILATION_CACHE_DIR")
            },
        },
    )
    started = time.time()
    try:
        if args.repetitions < 2 or args.copies < 1:
            raise ValueError("at least two repetitions and one compiled copy are required")
        if args.platform == "cpu":
            jax.config.update("jax_platforms", "cpu")
        with jax.default_device(jax.devices(args.platform)[0]):
            result = diagnose(
                args.episode, args.checkpoint, args.output, args.repetitions, args.copies
            )
        result["elapsed_wall_seconds"] = time.time() - started
        result["hlo_source_sha256"] = {path.name: _sha(path) for path in args.output.glob("*.txt")}
        _write_json(args.output / "results.json", result)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "status": result["status"],
                    "input_parity_passed": result["input_parity_passed"],
                }
            )
        )
        return 0 if result["input_parity_passed"] else 2
    except Exception as error:
        _write_json(
            args.output / "error.json",
            {"status": "incomplete", "error": str(error), "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
