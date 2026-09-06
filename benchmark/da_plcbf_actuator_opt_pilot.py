"""Reproducible CPU/GPU OPT service microbenchmarks with preserved numerical traces.

Example (CPU, fresh output required)::

    SCIPY_ARRAY_API=1 JAX_PLATFORMS=cpu .pixi/envs/gpu-tests/bin/python \
        benchmark/da_plcbf_actuator_opt_pilot.py --output artifacts/da_plcbf_actuator/opt_cpu_v1

This is a numerical one-decision development microbenchmark. The 0.025 m ego
radius is a deliberately small test geometry, not the physical aircraft collider
or an obstacle-performance episode. It must not supply safety/task claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.actuator_dynamics import (
    fit_native_time_constant,
    make_actuator_model,
)
from crazyflow.safety.da_plcbf.actuator_learning import ActuatorSkillConfig
from crazyflow.safety.da_plcbf.actuator_opt import ActuatorOPTConfig, build_actuator_opt
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.continuous_version_a import RuntimeObstacleTrajectories
from crazyflow.safety.da_plcbf.version_a_barriers import RigidBodySafetySet, VersionAModel


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (jax.Array, np.ndarray)):
        return _jsonable(np.asarray(value).tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def run(output: Path, budgets: list[float]) -> dict[str, Any]:
    """Measure two declared one-decision cases per budget without replacing any run."""
    output.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[1]
    source_paths = (
        "benchmark/da_plcbf_actuator_opt_pilot.py",
        "crazyflow/safety/da_plcbf/actuator_opt.py",
        "crazyflow/safety/da_plcbf/actuator_dynamics.py",
        "crazyflow/safety/da_plcbf/actuator_learning.py",
        "crazyflow/safety/da_plcbf/actuator_plcbf.py",
        "crazyflow/safety/da_plcbf/continuous_version_a.py",
        "crazyflow/drones/params.toml",
    )
    params = load_params("cf21B_500")
    body = VersionAModel(
        jnp.asarray(params["mass"]),
        jnp.asarray(params["gravity_vec"]),
        jnp.asarray(params["J"]),
        jnp.asarray(np.linalg.inv(params["J"])),
        jnp.asarray(params["drag_matrix"]),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros(3),
    )
    model = make_actuator_model(
        body,
        L=params["L"],
        thrust2torque=params["thrust2torque"],
        mixing_matrix=params["mixing_matrix"],
        thrust_min=params["thrust_min"],
        thrust_max=params["thrust_max"],
        time_constants=fit_native_time_constant()["nominal_tau"],
    )
    hover = float(-body.mass * body.gravity_vec[2] / 4)
    state = jnp.zeros(17).at[2].set(1).at[6].set(1).at[13:].set(hover)
    config = ActuatorFilterConfig(
        dt=0.02,
        horizon=60,
        command_hold_steps=2,
        held_substeps=2,
        obstacle_clearance=0,
        ego_radius=0.025,
    )
    actor = ActuatorSkillConfig(dt=0.02, horizon=60, control_interval_steps=2)
    safety = RigidBodySafetySet(
        jnp.array([[0.0, 0.0, 3.0]]),
        jnp.array([0.015]),
        jnp.array([False]),
        jnp.array([-3.0, -3.0, 0.1]),
        jnp.array([3.0, 3.0, 3.0]),
        jnp.asarray(4.0),
        jnp.asarray(12.0),
        jnp.asarray(1.0),
    )
    clear = RuntimeObstacleTrajectories(
        jnp.tile(jnp.array([[[0.0, 0.0, 3.0]]]), (61, 1, 1)),
        jnp.array([0.015]),
        jnp.ones((61, 1), dtype=bool),
    )
    blocked = clear._replace(centers=clear.centers.at[:, 0, 2].set(1.07))
    nominal = jnp.full((30, 4), hover)
    emergency = jnp.full(4, hover)
    manifest: dict[str, Any] = {
        "schema": "da_plcbf_actuator_opt_service_microbenchmark_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip(),
        "source_sha256": {
            path: hashlib.sha256((source_root / path).read_bytes()).hexdigest()
            for path in source_paths
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "backend": jax.default_backend(),
            "devices": [str(x) for x in jax.devices()],
        },
        "scope": "One-decision numerical microbenchmark; not physical-collider episodes",
        "timing_scope": (
            "Every operation inside solve; input generation and trace serialization "
            "are outside service timing"
        ),
        "filter_config": asdict(config),
        "actor_config": asdict(actor),
        "model": {**model._asdict(), "body": model.body._asdict()},
        "safety": safety._asdict(),
        "initial_state": state,
        "runs": [],
    }
    for index, budget in enumerate(budgets):
        opt_config = ActuatorOPTConfig(wall_time_budget=budget)
        controller = build_actuator_opt(config, actor, opt_config)
        startup = controller.warmup(state, model, clear, safety, nominal, emergency)
        for label, obstacles, commands in (
            ("clear_hover", clear, nominal),
            ("future_vertical_collision", blocked, jnp.full((30, 4), 0.15)),
        ):
            result = controller.solve(state, model, obstacles, safety, commands, emergency)
            stem = f"budget_{index:02d}_{label}"
            trace = {
                "source_state": np.asarray(state),
                "nominal_commands": np.asarray(commands),
                "emergency_command": np.asarray(emergency),
                "action": result.action,
                "obstacle_centers": np.asarray(obstacles.centers),
                "obstacle_radii": np.asarray(obstacles.radii),
                "obstacle_mask": np.asarray(obstacles.mask),
                "held_nodes": np.asarray(result.held_check.nodes),
                "held_forces": np.asarray(result.held_check.actual_forces),
                "held_wrenches": np.asarray(result.held_check.applied_wrenches),
            }
            if result.plan is not None:
                trace.update(plan_commands=result.plan.commands, plan_states=result.plan.states)
            np.savez_compressed(output / f"{stem}.npz", **trace)
            row = {
                "case": label,
                "opt_config": asdict(opt_config),
                "startup_seconds": startup,
                "service_seconds": result.service_seconds,
                "deadline_met": result.deadline_met,
                "available_at": result.available_at,
                "available_action": result.available_action,
                "action": result.action,
                "mode": result.mode,
                "plan_feasible": result.feasible,
                "compilation_in_solve": result.compilation_in_solve,
                "input_valid": result.input_valid,
                "budget_exhausted": result.budget_exhausted,
                "evaluations": result.evaluations,
                "attempts": [asdict(a) for a in result.attempts],
                "held_collision_margin": result.held_check.collision_margin,
                "held_operational_minimum": np.min(result.held_check.operational_margins),
                "held_command_motor_margin": result.held_check.command_motor_margin,
                "held_passed": result.held_check.passed,
                "trace_file": f"{stem}.npz",
                "plan_objective": None if result.plan is None else result.plan.objective,
                "plan_collision_margin": None
                if result.plan is None
                else result.plan.minimum_collision_margin,
            }
            manifest["runs"].append(row)
            (output / "summary.json").write_text(
                json.dumps(_jsonable(manifest), indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    _jsonable(
                        {
                            key: row[key]
                            for key in (
                                "case",
                                "service_seconds",
                                "deadline_met",
                                "mode",
                                "plan_feasible",
                                "evaluations",
                            )
                        }
                    )
                ),
                flush=True,
            )
    (output / "README.md").write_text(
        "# OPT service microbenchmark\n\n"
        "Two one-decision CPU/GPU development cases per declared wall-clock budget. "
        "These are numerical microbenchmarks with a 0.025 m ego sphere and do not constitute "
        "complete physical-collider episodes, tracking validation, "
        "or comparative safety results.\n\n"
        "`summary.json` records source hashes, settings, initial state, startup and service costs, "
        "every initialization/solver attempt and deadline availability. NPZ files retain complete "
        "candidate-plan and accepted-first-hold states, commands, forces and obstacle predictions. "
        "Serialization runs after service timing. No previous plan is installed by warmup.\n"
    )
    files = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in files)
    )
    return _jsonable(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.04, 0.2])
    arguments = parser.parse_args()
    run(arguments.output, arguments.budgets)


if __name__ == "__main__":
    main()
