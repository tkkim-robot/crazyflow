"""Coupled actuator authority and independently replayed prescribed-command witnesses.

This CPU diagnostic covers eta={1,.85,.7}, lag={1,2} on motors 0 and 1. Static
hover feasibility and individual-axis wrench margins are not navigation or safety
certificates. A failed trim or failed prescribed trajectory proves no general
infeasibility. The fault-transient replay preserves nominal motor memory; separate
settled-trim and maneuver replays explicitly use a different initial condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from crazyflow.safety.da_plcbf.actuator_independent import ActuatorEvent, NumpyEffortPlant
from crazyflow.safety.da_plcbf.actuator_learning import load_actuator_learner_checkpoint

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.actuator_dynamics import ActuatorModel


def write_json(path: Path, value: Any) -> None:
    """Preserve earlier attempts and reject nonfinite summary values."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def model_payload(model: ActuatorModel) -> dict[str, Any]:
    """Store every current physical parameter, not just its cell identifier."""
    return {
        "body": {
            name: np.asarray(value).tolist()
            for name, value in zip(model.body._fields, model.body, strict=True)
        },
        **{
            name: np.asarray(getattr(model, name)).tolist()
            for name in model._fields
            if name != "body"
        },
    }


def coupled_hover_authority(model: ActuatorModel) -> dict[str, Any]:
    """Solve all four wrench equations and bounded individual-axis sections exactly.

    The independent-axis margins hold the other three wrench coordinates fixed.
    Their extrema cannot in general be requested simultaneously.
    """
    gravity = np.asarray(model.body.gravity_vec)
    if (
        np.any(gravity[:2] != 0)
        or np.any(np.asarray(model.body.external_force) != 0)
        or np.any(np.asarray(model.body.external_torque) != 0)
        or np.any(np.asarray(model.body.wind_velocity) != 0)
    ):
        raise ValueError(
            "this upright static-trim diagnostic requires vertical gravity "
            "and no exogenous wrench/wind"
        )
    mapping = (
        np.asarray(model.force_to_wrench, dtype=np.float64)
        * np.asarray(model.effectiveness)[None, :]
    )
    lower, upper = np.asarray(model.command_lower), np.asarray(model.command_upper)
    weight = -float(model.body.mass) * float(gravity[2])
    target = np.array([weight, 0.0, 0.0, 0.0])
    inverse = np.linalg.inv(mapping)
    command = inverse @ target
    feasible = bool(np.all(command >= lower) and np.all(command <= upper))

    def interval(origin: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
        low, high = -math.inf, math.inf
        for point, slope, bound_low, bound_high in zip(
            origin, direction, lower, upper, strict=True
        ):
            if abs(slope) < 1e-15:
                if not bound_low <= point <= bound_high:
                    return math.inf, -math.inf
            else:
                ends = sorted(((bound_low - point) / slope, (bound_high - point) / slope))
                low, high = max(low, ends[0]), min(high, ends[1])
        return float(low), float(high)

    collective_low, collective_high = interval(np.zeros(4), inverse[:, 0])
    axes = {}
    if feasible:
        for index, name in enumerate(("collective_N", "roll_Nm", "pitch_Nm", "yaw_Nm")):
            low, high = interval(command, inverse[:, index])
            axes[name] = {
                "negative_delta": low,
                "positive_delta": high,
                "command_at_negative_endpoint_N": (command + low * inverse[:, index]).tolist(),
                "command_at_positive_endpoint_N": (command + high * inverse[:, index]).tolist(),
            }
    return {
        "coupled_mapping": mapping.tolist(),
        "rank": int(np.linalg.matrix_rank(mapping)),
        "hover_target_wrench": target.tolist(),
        "required_command_N": command.tolist(),
        "actual_motor_forces_at_trim_N": (np.asarray(model.effectiveness) * command).tolist(),
        "wrench_residual": (mapping @ command - target).tolist(),
        "feasible_hover_trim": feasible,
        "lower_command_margin_N": (command - lower).tolist(),
        "upper_command_margin_N": (upper - command).tolist(),
        "minimum_command_margin_N": float(np.min(np.minimum(command - lower, upper - command))),
        "torque_free_collective_interval_N": [collective_low, collective_high],
        "torque_free_collective_upper_over_weight": collective_high / weight,
        "torque_free_upward_acceleration_margin_mps2": (collective_high - weight)
        / float(model.body.mass),
        "individual_wrench_axis_sections": axes,
        "axis_scope": (
            "one axis at a time, other three coordinates fixed at hover; "
            "extrema are not jointly composable"
        ),
        "failure_scope": (
            "a failed hover trim is not proof that all finite-horizon maneuvers are impossible"
        ),
    }


def rest_state(model: ActuatorModel, command: np.ndarray) -> np.ndarray:
    """Static upright condition at 1.2 m, with explicit motor efforts."""
    state = np.zeros(17, dtype=np.float64)
    state[2], state[6] = 1.2, 1.0
    state[13:] = command
    return state


def prescribed_vertical_commands(
    model: ActuatorModel, *, duration: float = 1.2, hold: float = 0.04, acceleration: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute held commands for a sine vertical acceleration, without optimization.

    The ideal upright target includes linear vertical drag. Inverse lag matches
    each desired motor endpoint; differing time constants need not preserve zero
    instantaneous torque inside a hold. Independent replay measures that error.
    """
    if duration <= 0 or hold <= 0 or not math.isclose(duration / hold, round(duration / hold)):
        raise ValueError("duration must be a positive integer multiple of the positive hold")
    coupled_hover_authority(model)
    drag = np.asarray(model.body.drag_matrix)
    if np.any(drag - np.diag(np.diag(drag)) != 0):
        raise ValueError("upright sine target requires diagonal drag")
    times = np.linspace(0.0, duration, round(duration / hold) + 1)
    frequency = 2 * math.pi / duration
    acc = acceleration * np.sin(frequency * times)
    speed = acceleration / frequency * (1 - np.cos(frequency * times))
    height = 1.2 + acceleration / frequency * (times - np.sin(frequency * times) / frequency)
    collective = (
        float(model.body.mass) * (acc - float(model.body.gravity_vec[2])) - drag[2, 2] * speed
    )
    target = np.stack((height, speed, acc, collective), axis=1)
    effort = np.linalg.solve(
        np.asarray(model.force_to_wrench, dtype=np.float64),
        np.stack(
            (
                collective,
                np.zeros_like(collective),
                np.zeros_like(collective),
                np.zeros_like(collective),
            )
        ),
    )
    effort = effort.T / np.asarray(model.effectiveness)[None, :]
    beta = -np.expm1(-hold / np.asarray(model.time_constants, dtype=np.float64))
    commands = effort[:-1] + (effort[1:] - effort[:-1]) / beta
    if np.any(commands < np.asarray(model.command_lower)) or np.any(
        commands > np.asarray(model.command_upper)
    ):
        raise ValueError(
            "prescribed maneuver exceeds command bounds; no clipping or feasibility inference"
        )
    return times, commands, target


def replay(
    model: ActuatorModel,
    initial: np.ndarray,
    commands: np.ndarray,
    hold: float,
    step: float,
    *,
    fault_at_zero_from: ActuatorModel | None = None,
) -> dict[str, np.ndarray]:
    """Independent P1 body/motor integration with no motor reset at a fault event."""
    plant = NumpyEffortPlant(
        initial,
        model if fault_at_zero_from is None else fault_at_zero_from,
        max_step=step,
        events=() if fault_at_zero_from is None else (ActuatorEvent(0.0, model),),
    )
    traces = [plant.advance(command, hold) for command in commands]
    result = {}
    for name in ("times", "states", "actual_forces", "wrenches"):
        result[name] = np.concatenate(
            [getattr(traces[0], name), *(getattr(trace, name)[1:] for trace in traces[1:])]
        )
    result["commands"] = commands
    result["command_times"] = np.arange(len(commands)) * hold
    return result


def replay_metrics(trace: dict[str, np.ndarray], initial: np.ndarray) -> dict[str, Any]:
    """Describe the whole continuous integration trace; no collision checks are present."""
    states = trace["states"]
    tilt = np.arccos(np.clip(1 - 2 * np.sum(states[:, 3:5] ** 2, axis=1), -1, 1))
    return {
        "finite_trace": bool(np.all(np.isfinite(states))),
        "initial_state_exactly_preserved": bool(np.array_equal(states[0], initial)),
        "initial_state": initial.tolist(),
        "terminal_state": states[-1].tolist(),
        "maximum_speed_mps": float(np.max(np.linalg.norm(states[:, 7:10], axis=1))),
        "terminal_speed_mps": float(np.linalg.norm(states[-1, 7:10])),
        "maximum_body_rate_rps": float(np.max(np.linalg.norm(states[:, 10:13], axis=1))),
        "maximum_tilt_rad": float(np.max(tilt)),
        "minimum_height_m": float(np.min(states[:, 2])),
        "height_displacement_m": float(states[-1, 2] - initial[2]),
        "maximum_position_displacement_m": float(
            np.max(np.linalg.norm(states[:, :3] - initial[:3], axis=1))
        ),
        "maximum_lateral_displacement_m": float(
            np.max(np.linalg.norm(states[:, :2] - initial[:2], axis=1))
        ),
        "integration_nodes": len(states),
        "scope": (
            "P1 open-loop independent numerical trajectory; "
            "no ground contact, obstacle or navigation feasibility check"
        ),
    }


def archived_witness_inventory(root: Path, models: list[ActuatorModel]) -> list[dict[str, Any]]:
    """Bind actual archived numerical witnesses and mark exact-model applicability."""
    entries = []
    for binding_path in sorted(root.glob("body-witness-*/binding.json")):
        summary_path = binding_path.with_name("summary.json")
        binding, summary = (
            json.loads(binding_path.read_text()),
            json.loads(summary_path.read_text()),
        )
        payload = binding["model"]
        matches = [index for index, model in enumerate(models) if model_payload(model) == payload]
        entries.append(
            {
                "binding_path": str(binding_path.resolve()),
                "summary_path": str(summary_path.resolve()),
                "binding_sha256": hashlib.sha256(binding_path.read_bytes()).hexdigest(),
                "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                "archived_witness_found": summary["witness_found"],
                "archived_effectiveness": payload["effectiveness"],
                "archived_time_constants_s": payload["time_constants"],
                "exact_model_matching_cell_indices": matches,
                "applicability": (
                    "only exact-model entries could support these cells; "
                    "differing faults are not substituted"
                ),
            }
        )
    return entries


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute six fixed physical cells on CPU and retain source, command and state arrays."""
    import jax

    if jax.default_backend() != "cpu":
        raise ValueError("run this read-only physical diagnostic with JAX_PLATFORMS=cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = load_actuator_learner_checkpoint(args.checkpoint)
    nominal = checkpoint.contract.model
    nominal_command = np.asarray(coupled_hover_authority(nominal)["required_command_N"])
    cells, models = [], []
    for eta in (1.0, 0.85, 0.7):
        for lag in (1.0, 2.0):
            model = nominal._replace(
                effectiveness=np.asarray(
                    [eta, eta, 1.0, 1.0], dtype=np.asarray(nominal.effectiveness).dtype
                ),
                time_constants=np.asarray(nominal.time_constants)
                * np.asarray([lag, lag, 1.0, 1.0]),
            )
            models.append(model)
            cells.append({"eta": eta, "lag": lag, "affected_motors": [0, 1]})
    sources = {}
    repo = Path(__file__).resolve().parents[1]
    for name in (
        "benchmark/da_plcbf_actuator_diagnostic_authority.py",
        "crazyflow/safety/da_plcbf/actuator_independent.py",
        "crazyflow/safety/da_plcbf/actuator_dynamics.py",
    ):
        payload = (repo / name).read_bytes()
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        sources[name] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "scope": __doc__,
        "checkpoint": str(checkpoint.json_path.resolve()),
        "checkpoint_sha256": checkpoint.sha256,
        "sources": sources,
        "cells": cells,
        "duration_seconds": 1.2,
        "command_hold_seconds": 0.04,
        "independent_steps_seconds": [0.002, 0.001],
        "sine_acceleration_amplitude_mps2": 0.5,
        "maneuver_acceptance": {
            "terminal_speed_mps": 0.05,
            "height_target_error_m": 0.03,
            "tilt_rad": 0.1,
            "body_rate_rps": 0.5,
        },
        "timing_scope": (
            "CPU diagnostics run alongside unrelated work; wall times are not efficiency evidence"
        ),
        "archive_witness_inventory": archived_witness_inventory(Path(args.archive_root), models),
    }
    write_json(output / "manifest.json", manifest)
    rows = []
    for index, (cell, model) in enumerate(zip(cells, models, strict=True)):
        authority = coupled_hover_authority(model)
        row = {**cell, "model": model_payload(model), "authority": authority}
        if not authority["feasible_hover_trim"]:
            row["replays_skipped"] = "no bounded hover command; no general infeasibility claim"
            rows.append(row)
            continue
        trim = np.asarray(authority["required_command_N"])
        static = rest_state(model, trim)
        preserved = rest_state(nominal, nominal_command)
        times, maneuver, target = prescribed_vertical_commands(model)
        constant = np.tile(trim, (30, 1))
        cases = {
            "fault_preserving_nominal_motors": (preserved, constant, nominal),
            "already_settled_postfault_trim": (static, constant, None),
            "prescribed_vertical_from_settled_trim": (static, maneuver, None),
        }
        row["replays"] = {}
        for name, (initial, commands, before_fault) in cases.items():
            reports, endpoints = {}, []
            for step in (0.002, 0.001):
                trace = replay(
                    model, initial, commands, 0.04, step, fault_at_zero_from=before_fault
                )
                report = replay_metrics(trace, initial)
                report["minimum_command_margin_N"] = float(
                    np.min(
                        np.minimum(
                            commands - np.asarray(model.command_lower),
                            np.asarray(model.command_upper) - commands,
                        )
                    )
                )
                if name.startswith("prescribed_vertical"):
                    report["target_terminal_height_m"] = float(target[-1, 0])
                    report["terminal_height_target_error_m"] = float(
                        abs(trace["states"][-1, 2] - target[-1, 0])
                    )
                    criteria = manifest["maneuver_acceptance"]
                    report["bounded_maneuver_witness"] = bool(
                        report["finite_trace"]
                        and report["minimum_command_margin_N"] >= 0
                        and report["terminal_speed_mps"] <= criteria["terminal_speed_mps"]
                        and report["terminal_height_target_error_m"]
                        <= criteria["height_target_error_m"]
                        and report["maximum_tilt_rad"] <= criteria["tilt_rad"]
                        and report["maximum_body_rate_rps"] <= criteria["body_rate_rps"]
                    )
                endpoints.append(trace["states"][-1])
                reports[str(step)] = report
                path = output / f"cell{index}_{name}_step{step:g}.npz"
                with path.open("xb") as stream:
                    np.savez_compressed(
                        stream, **trace, ideal_target_times=times, ideal_target=target
                    )
            row["replays"][name] = {
                "steps": reports,
                "coarse_fine_terminal_state_max_difference": float(
                    np.max(np.abs(endpoints[0] - endpoints[1]))
                ),
            }
        rows.append(row)
        write_json(output / f"cell{index}.json", row)
        print(
            json.dumps(
                {
                    "event": "authority_cell_complete",
                    **cell,
                    "feasible_hover_trim": authority["feasible_hover_trim"],
                    "minimum_trim_margin_N": authority["minimum_command_margin_N"],
                }
            ),
            flush=True,
        )
    summary = {"manifest": manifest, "cells": rows}
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    """Require explicit numerical checkpoint and a fresh artifact directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--archive-root", default="artifacts/da_plcbf/actuator-study-20260906/v1")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
