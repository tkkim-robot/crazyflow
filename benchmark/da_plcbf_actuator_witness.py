"""CPU direct-command feasibility witnesses for a saved body-trajectory recovery target.

This is an offline local optimization diagnostic with oracle current parameters,
30 independent held motor commands and a fine replay. It is not a learned policy,
an obstacle-safety controller, a deadline result, or an infeasibility certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.actuator_dynamics import effort_lag_step, hover_authority
from crazyflow.safety.da_plcbf.actuator_learning import load_actuator_learner_checkpoint


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(_json(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class _Finished(Exception):
    def __init__(self, reason: str, command: np.ndarray) -> None:
        self.reason = reason
        self.command = command.copy()


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Seek one fine-replayed feasible target, retaining all attempted initializations."""
    from scipy.optimize import minimize

    if jax.default_backend() != "cpu":
        raise ValueError("run this offline witness with JAX_PLATFORMS=cpu")
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = load_actuator_learner_checkpoint(args.checkpoint)
    config = bundle.config
    if config.horizon != 60 or config.dt != 0.02 or config.control_interval_steps != 2:
        raise ValueError("this witness requires the declared H60/.02/hold2 recovery contract")
    manifest_path = args.evaluation.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model = bundle.contract.model._replace(
        effectiveness=jnp.asarray(manifest["current_effectiveness"]),
        time_constants=jnp.asarray(manifest["current_time_constants_s"]),
    )
    with np.load(args.evaluation) as saved:
        initial = np.asarray(saved["initial_states"][args.state_index])
        reference = np.asarray(saved["reference_states"][args.state_index, args.skill])
        initializers = {
            "F2": np.asarray(saved["commands"][args.state_index, args.skill]),
            "nominal_teacher_commands": np.asarray(
                saved["reference_commands"][args.state_index, args.skill]
            ),
        }
    if not np.array_equal(reference[0], initial):
        raise ValueError("saved reference does not begin at the exact supplied augmented state")
    for file in args.initializer:
        with np.load(file) as saved:
            if not np.array_equal(saved["initial_states"][args.state_index], initial):
                raise ValueError("initializer does not share the exact augmented initial state")
            initializers[file.stem] = np.asarray(saved["commands"][args.state_index, args.skill])
    initializers["current_hover_trim"] = np.broadcast_to(
        hover_authority(model)["required_command_N"], (60, 4)
    ).copy()
    lower, upper = np.asarray(model.command_lower), np.asarray(model.command_upper)
    width = upper - lower
    source_files = [
        Path(__file__),
        Path("crazyflow/safety/da_plcbf/actuator_dynamics.py"),
        Path("crazyflow/safety/da_plcbf/direct_wrench.py"),
    ]
    inputs = [args.evaluation, manifest_path, *args.initializer]
    for suffix in (".json", ".npz"):
        inputs.append(args.checkpoint.with_suffix(suffix))
    provenance = {
        "scope": __doc__,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "initializer"
        },
        "input_sha256": {
            str(file.resolve()): hashlib.sha256(file.read_bytes()).hexdigest() for file in inputs
        },
        "source_sha256": {
            str(file.resolve()): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in source_files
        },
        "reference_sha256": manifest["reference_sha256"],
        "device": str(jax.devices()[0]),
        "model": jax.tree.map(np.asarray, model)._asdict(),
        "initial_state": initial,
        "prefix_nodes": [6, 15, 30, 45, 60],
        "fine_step_seconds": 0.002,
        "thresholds": {
            "position_coordinate_rmse_m": 0.3,
            "velocity_coordinate_rmse_mps": 0.6,
            "terminal_speed_mps": 0.8,
            "tilt_rad": 0.9,
            "body_rate_rps": 12,
            "speed_mps": 3.5,
            "arena_lower_m": [-4.92, -3.92, 0.23],
            "arena_upper_m": [4.92, 3.92, 3.92],
        },
    }
    # NamedTuple body fields need named conversion for a reviewable serialized model.
    provenance["model"]["body"] = jax.tree.map(np.asarray, model.body)._asdict()
    _write(args.output / "binding.json", provenance)
    for file in source_files:
        target = args.output / "source" / file.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(file.read_bytes())
    nodes = jnp.asarray([6, 15, 30, 45, 60])
    y0, target = jnp.asarray(initial), jnp.asarray(reference)
    arena_lower = jnp.asarray([-4.92, -3.92, 0.23])
    arena_upper = jnp.asarray([4.92, 3.92, 3.92])

    def rollout(normalized: jax.Array, fine: bool = False) -> jax.Array:
        commands = model.command_lower + jnp.reshape(normalized, (30, 4)) * width
        repeated = jnp.repeat(commands, 20 if fine else 2, axis=0)

        def step(state: jax.Array, command: jax.Array) -> tuple[jax.Array, jax.Array]:
            updated = effort_lag_step(state, command, model, 0.002 if fine else 0.02)
            return updated, updated

        _, states = jax.lax.scan(step, y0, repeated)
        return jnp.concatenate((y0[None], states), axis=0)

    def measures(states: jax.Array) -> tuple[jax.Array, jax.Array]:
        position_mse = jnp.mean((states[nodes, :3] - target[nodes, :3]) ** 2)
        velocity_mse = jnp.mean((states[nodes, 7:10] - target[nodes, 7:10]) ** 2)
        terminal_squared = jnp.sum(states[-1, 7:10] ** 2)
        margins = jnp.concatenate(
            (
                jnp.min(states[:, :3] - arena_lower, axis=0),
                jnp.min(arena_upper - states[:, :3], axis=0),
                jnp.asarray(
                    [
                        jnp.min(1 - 2 * jnp.sum(states[:, 3:5] ** 2, axis=-1)) - jnp.cos(0.9),
                        1 - jnp.max(jnp.sum(states[:, 10:13] ** 2, axis=-1)) / 144,
                        1 - jnp.max(jnp.sum(states[:, 7:10] ** 2, axis=-1)) / 12.25,
                        1 - position_mse / 0.09,
                        1 - velocity_mse / 0.36,
                        1 - terminal_squared / 0.64,
                    ]
                ),
            )
        )
        objective = position_mse / 0.09 + velocity_mse / 0.36 + 0.1 * terminal_squared / 0.64
        return objective, margins

    def objective(normalized: jax.Array) -> jax.Array:
        value, _ = measures(rollout(normalized))
        commands = normalized.reshape((30, 4))
        return value + 0.0001 * jnp.mean((commands[1:] - commands[:-1]) ** 2)

    objective_gradient = jax.jit(jax.value_and_grad(objective))
    constraints = jax.jit(lambda normalized: measures(rollout(normalized))[1])
    constraint_jacobian = jax.jit(jax.jacfwd(lambda normalized: measures(rollout(normalized))[1]))
    coarse_rollout = jax.jit(rollout)
    fine_rollout = jax.jit(lambda normalized: rollout(normalized, fine=True))

    def check(normalized: np.ndarray, *, fine: bool) -> tuple[dict[str, Any], np.ndarray]:
        states = np.asarray((fine_rollout if fine else coarse_rollout)(jnp.asarray(normalized)))
        stride = 10 if fine else 1
        sample = states[::stride]
        prefix = np.asarray([6, 15, 30, 45, 60])
        position = np.sqrt(np.mean((sample[prefix, :3] - reference[prefix, :3]) ** 2))
        velocity = np.sqrt(np.mean((sample[prefix, 7:10] - reference[prefix, 7:10]) ** 2))
        terminal = np.linalg.norm(states[-1, 7:10])
        tilt = np.arccos(np.clip(1 - 2 * np.sum(states[:, 3:5] ** 2, axis=1), -1, 1)).max()
        rate = np.linalg.norm(states[:, 10:13], axis=1).max()
        speed = np.linalg.norm(states[:, 7:10], axis=1).max()
        arena = np.minimum(
            states[:, :3] - np.asarray(arena_lower), np.asarray(arena_upper) - states[:, :3]
        ).min()
        commands = lower + normalized.reshape((30, 4)) * width
        motor_margin = min(np.min(states[:, 13:] - lower), np.min(upper - states[:, 13:]))
        command_margin = min(np.min(commands - lower), np.min(upper - commands))
        passed = bool(
            np.all(np.isfinite(states))
            and position <= 0.3
            and velocity <= 0.6
            and terminal <= 0.8
            and tilt <= 0.9
            and rate <= 12
            and speed <= 3.5
            and arena >= 0
            and motor_margin >= -1e-7
            and command_margin >= -1e-7
        )
        return {
            "fine": fine,
            "step_seconds": 0.002 if fine else 0.02,
            "position_coordinate_rmse_m": float(position),
            "velocity_coordinate_rmse_mps": float(velocity),
            "terminal_speed_mps": float(terminal),
            "maximum_tilt_rad": float(tilt),
            "maximum_body_rate_rps": float(rate),
            "maximum_speed_mps": float(speed),
            "arena_minimum_margin_m": float(arena),
            "motor_state_minimum_margin_N": float(motor_margin),
            "command_minimum_margin_N": float(command_margin),
            "passes_declared_thresholds": passed,
        }, states

    records = []
    warm_begin = time.perf_counter()
    first = ((initializers["F2"][::2] - lower) / width).reshape(-1).astype(float)
    jax.block_until_ready(
        (
            objective_gradient(jnp.asarray(first)),
            constraints(jnp.asarray(first)),
            constraint_jacobian(jnp.asarray(first)),
            fine_rollout(jnp.asarray(first)),
        )
    )
    warm_seconds = time.perf_counter() - warm_begin
    witness = None
    for index, (name, commands) in enumerate(initializers.items()):
        if not np.array_equal(commands[::2], commands[1::2]):
            raise ValueError("initializer violates the common two-node command hold")
        normalized = ((commands[::2] - lower) / width).reshape(-1).astype(float)
        if np.any(normalized < -1e-7) or np.any(normalized > 1 + 1e-7):
            raise ValueError("initializer exceeds physical command bounds")
        initial_check, _ = check(normalized, fine=True)
        began = time.perf_counter()
        evaluations = 0
        selected = normalized.copy()
        status, message, iterations = "initialization", "initial fine replay", 0

        def function(x: np.ndarray) -> tuple[float, np.ndarray]:
            value, gradient = objective_gradient(jnp.asarray(x))
            return float(value), np.asarray(gradient, dtype=float)

        def callback(x: np.ndarray) -> None:
            nonlocal evaluations, selected
            evaluations += 1
            selected = x.copy()
            if np.min(np.asarray(constraints(jnp.asarray(x)))) >= 0:
                candidate, _ = check(x, fine=True)
                if candidate["passes_declared_thresholds"]:
                    raise _Finished("fine_replayed_witness_found", x)
            if time.perf_counter() - began >= args.seconds_per_start:
                raise _Finished("local_optimization_time_budget", x)

        if not initial_check["passes_declared_thresholds"]:
            try:
                solution = minimize(
                    function,
                    normalized,
                    jac=True,
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * 120,
                    constraints=[
                        {
                            "type": "ineq",
                            "fun": lambda x: np.asarray(constraints(jnp.asarray(x)), dtype=float),
                            "jac": lambda x: np.asarray(
                                constraint_jacobian(jnp.asarray(x)), dtype=float
                            ),
                        }
                    ],
                    callback=callback,
                    options={"maxiter": args.maxiter, "ftol": 1e-7, "disp": False},
                )
                selected = solution.x
                status, message, iterations = (
                    "optimizer_completed",
                    str(solution.message),
                    int(solution.nit),
                )
            except _Finished as finished:
                selected = finished.command
                status, message, iterations = finished.reason, finished.reason, evaluations
        coarse, coarse_states = check(selected, fine=False)
        fine, fine_states = check(selected, fine=True)
        record = {
            "initializer": name,
            "status": status,
            "message": message,
            "iterations": iterations,
            "optimization_seconds": time.perf_counter() - began,
            "initial_fine": initial_check,
            "coarse": coarse,
            "fine": fine,
            "objective": function(selected)[0],
        }
        records.append(record)
        stem = args.output / f"attempt-{index:02d}-{name}"
        _write(stem.with_suffix(".json"), record)
        with stem.with_suffix(".npz").open("xb") as stream:
            np.savez_compressed(
                stream,
                normalized_commands=selected.reshape((30, 4)),
                commands_N=lower + selected.reshape((30, 4)) * width,
                initial_commands_N=commands[::2],
                initial_state=initial,
                reference_states=reference,
                coarse_states=coarse_states,
                fine_states=fine_states,
                fine_times_seconds=np.arange(601) * 0.002,
            )
        print(json.dumps(_json(record)), flush=True)
        if fine["passes_declared_thresholds"]:
            witness = str(stem)
            break
    summary = {
        "scope": __doc__,
        "witness_found": witness is not None,
        "witness_stem": witness,
        "compile_and_warm_seconds": warm_seconds,
        "attempts": records,
        "initializers_requested": list(initializers),
        "unstarted_initializers": list(initializers)[len(records) :],
        "failure_interpretation": (
            "Local optimization failure is not evidence of physical infeasibility"
        ),
        "coarse_fine_scope": (
            "Same held commands replayed at .02 s and .002 s; "
            "fine replay checks every operational and motor node"
        ),
    }
    _write(args.output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--state-index", type=int, required=True)
    parser.add_argument("--skill", type=int, default=2)
    parser.add_argument("--initializer", type=Path, action="append", default=[])
    parser.add_argument("--seconds-per-start", type=float, default=30)
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds_per_start <= 0 or args.maxiter < 1:
        parser.error("optimization budgets must be positive")
    run(args)


if __name__ == "__main__":
    main()
