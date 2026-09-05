"""One bounded closed-loop transfer check of the declared parameter-step cap.

Fresh common calm prefix; frozen versus unrestricted versus capped reference adaptation;
oracle and independent-estimator branches. Stops each branch at its first physical collision.
This is deterministic mechanism evidence, not a paced timing or success-rate experiment.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.da_plcbf_reference_ablation import _write
from crazyflow.safety.da_plcbf.continuous_demo_scenarios import (
    constant_wind_scenario,
    scenario_obstacle_window,
)
from crazyflow.safety.da_plcbf.learner_checkpoint import save_learner_checkpoint
from crazyflow.safety.da_plcbf.online_constant_wind import VersionAResources, _make_controller
from crazyflow.safety.da_plcbf.point_wind_estimator import (
    PointWindEstimatorConfig,
    initialize_point_wind_estimator,
    model_with_point_wind,
    update_point_wind_estimator,
)
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.state_conditioned_learning import (
    build_reference_skill_learner,
    build_reference_skill_learner_from_checkpoint,
    reference_contract_checkpoint_metadata,
    save_reference_contract,
)


def _array_tree(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Preserve every numerical decision field, including nested QP and held postchecks."""
    if hasattr(value, "_fields"):
        names = value._fields
    elif is_dataclass(value):
        names = [field.name for field in fields(value)]
    else:
        return {prefix: np.asarray(value)}
    result = {}
    for name in names:
        result.update(_array_tree(getattr(value, name), f"{prefix}__{name}" if prefix else name))
    return result


def _segment_clearance(before: np.ndarray, after: np.ndarray, scenario: Any) -> float:
    centers = np.asarray(scenario.obstacle_initial_centers)
    relative = before[:3] - centers
    delta = after[:3] - before[:3]
    fraction = np.clip(-np.sum(relative * delta, axis=-1) / max(float(delta @ delta), 1e-30), 0, 1)
    distance = np.linalg.norm(relative + fraction[:, None] * delta, axis=-1)
    return float(np.min(distance - np.asarray(scenario.obstacle_radii) - scenario.ego_radius))


def _physical_metrics(states: np.ndarray, clearances: list[float], scenario: Any) -> dict[str, Any]:
    quaternion = states[:, 3:7]
    rzz = 1 - 2 * np.sum(quaternion[:, :2] ** 2, axis=-1) / np.sum(quaternion**2, axis=-1)
    arena = np.minimum(
        states[:, :3] - np.asarray(scenario.arena_lower) - 0.08,
        np.asarray(scenario.arena_upper) - states[:, :3] - 0.08,
    )
    return {
        "minimum_physical_clearance_m": min(clearances),
        "minimum_shell_clearance_m": min(clearances) - scenario.obstacle_clearance,
        "maximum_speed_mps": float(np.max(np.linalg.norm(states[:, 7:10], axis=-1))),
        "maximum_angular_rate_rps": float(np.max(np.linalg.norm(states[:, 10:13], axis=-1))),
        "maximum_tilt_rad": float(np.max(np.arccos(np.clip(rzz, -1, 1)))),
        "minimum_arena_margin_m": float(np.min(arena)),
        "goal_distance_at_censor_or_end_m": float(
            np.linalg.norm(states[-1, :3] - scenario.goal_position)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = jax.devices(args.device)[0]
    bundle, contract, _ = build_reference_skill_learner_from_checkpoint(
        args.checkpoint, device=device
    )
    if (bundle.config.dt, bundle.config.horizon, bundle.config.control_interval_steps) != (
        0.02,
        60,
        2,
    ):
        raise ValueError("this declared diagnostic requires dt=.02, H60 and hold2")
    if not bundle.config.model_compensation:
        raise ValueError("model compensation must already be enabled in the nominal checkpoint")
    config = replace(bundle.config, learning_rate=0.001, max_parameter_update_norm=None)
    capped_config = replace(config, max_parameter_update_norm=0.002)
    learners = {
        "unrestricted": build_reference_skill_learner(contract, config, device=device),
        "capped": build_reference_skill_learner(contract, capped_config, device=device),
    }
    scenario = replace(
        constant_wind_scenario(),
        steps=400,
        wind_after=jnp.asarray([4.0, 1.6, 0.0]),
        name="declared_stabilization_closed_loop",
    )
    scenario.validate()
    resources = VersionAResources(contract.model, bundle.actuator)
    controller = _make_controller(
        scenario,
        resources,
        bundle.spec,
        config,
        nominal_acceleration_limit=1.2,
        waypoint_position_gain=2.0,
        waypoint_velocity_gain=2.8,
        device=device,
        nominal_model_compensation=True,
        control_interval_steps=2,
    )
    nominal_model = contract.model
    changed_model = nominal_model._replace(wind_velocity=scenario.wind_after)
    plant = jax.jit(lambda x, u, model: direct_wrench_symplectic_step(x, u, model, scenario.dt))
    estimator_config = PointWindEstimatorConfig(response_rate=2.4)
    estimate = jax.jit(
        lambda estimator, x, following, u: (
            update_point_wind_estimator(
                estimator, x, following, u, nominal_model, dt=scenario.dt, config=estimator_config
            ).state
        )
    )
    obstacle_windows = [scenario_obstacle_window(scenario, step) for step in range(0, 400, 2)]
    save_reference_contract(contract, args.output_dir / "nominal_reference")
    reference_metadata = reference_contract_checkpoint_metadata(
        args.output_dir / "nominal_reference"
    )
    _write(
        args.output_dir / "protocol.json",
        {
            "source_checkpoint": str(args.checkpoint),
            "source_checkpoint_sha256": bundle.sha256,
            "source_binding_status": bundle.metadata["reference_contract_binding"],
            "reference_binding": reference_metadata,
            "config": asdict(config),
            "capped_config": asdict(capped_config),
            "scenario": {k: np.asarray(v).tolist() for k, v in asdict(scenario).items()},
            "estimator_config": asdict(estimator_config),
            "nominal_controller": {
                "acceleration_limit": 1.2,
                "position_gain": 2.0,
                "velocity_gain": 2.8,
            },
            "schedule": (
                "post-event opportunities 0,2,4,...; update uses pre-action state/model "
                "and publishes next boundary"
            ),
            "prefix": (
                "one calm known-nominal-model physical prefix; "
                "estimator observes its transitions and is cloned"
            ),
            "censor": (
                "stop at first integration segment with physical swept clearance <=0; "
                "no subsequent commands or updates"
            ),
            "scope": (
                "one bounded deterministic mechanism experiment; "
                "no timing or guaranteed-winner claim"
            ),
            "source_sha256": {
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in (
                    __file__,
                    "crazyflow/safety/da_plcbf/persistent_skill_learner.py",
                    "crazyflow/safety/da_plcbf/state_conditioned_learning.py",
                    "crazyflow/safety/da_plcbf/online_constant_wind.py",
                    "crazyflow/safety/da_plcbf/continuous_version_a.py",
                )
            },
        },
    )
    state = jax.device_put(scenario.initial_state, device)
    previous = jnp.asarray(0, dtype=jnp.int32)
    estimator = initialize_point_wind_estimator()
    prefix_states, prefix_actions, prefix_decisions, prefix_clearances = (
        [np.asarray(state)],
        [],
        [],
        [],
    )
    for control in range(100):
        decision = jax.block_until_ready(
            controller(
                state, bundle.state.params, nominal_model, obstacle_windows[control], previous
            )
        )
        prefix_decisions.append(_array_tree(decision))
        prefix_actions.append(np.asarray(decision.action))
        for _ in range(2):
            following = plant(state, decision.action, nominal_model)
            estimator = estimate(estimator, state, following, decision.action)
            jax.block_until_ready((following, estimator))
            clearance = _segment_clearance(np.asarray(state), np.asarray(following), scenario)
            prefix_clearances.append(clearance)
            state = following
            prefix_states.append(np.asarray(state))
            if clearance <= 0:
                raise RuntimeError("shared fresh prefix collided before the declared branch event")
        previous = decision.selected_index
    prefix_arrays = {
        "states": np.asarray(prefix_states),
        "actions": np.asarray(prefix_actions),
        "physical_segment_clearance": np.asarray(prefix_clearances),
        **{f"estimator__{k}": v for k, v in _array_tree(estimator).items()},
        **{
            f"decision__{key}": np.stack([row[key] for row in prefix_decisions])
            for key in prefix_decisions[0]
        },
    }
    np.savez_compressed(args.output_dir / "shared_prefix.npz", **prefix_arrays)
    save_learner_checkpoint(
        bundle.state,
        bundle.spec,
        config,
        bundle.actuator,
        state,
        args.output_dir / "shared_prefix_checkpoint",
        metadata={
            "simulation_time": 4.0,
            "previous_policy_index": int(previous),
            **reference_metadata,
        },
    )
    shared_state, shared_previous, shared_estimator = state, previous, estimator
    print(f"shared prefix complete: state={np.asarray(shared_state).tolist()}", flush=True)
    outcomes, first_actions = {}, {}
    for information in ("oracle", "estimated"):
        for method in ("frozen", "unrestricted", "capped"):
            name = f"{information}_{method}"
            directory = args.output_dir / name
            directory.mkdir()
            persistent = bundle.state
            state, previous, estimator = shared_state, shared_previous, shared_estimator
            states, decisions, rows, clearances = [np.asarray(state)], [], [], []
            boundary_states, estimates = [], []
            used_parameters: list[dict[str, np.ndarray]] = []
            collided, collision_interval = False, None
            trial_config = capped_config if method == "capped" else config
            for control in range(100):
                when = 4.0 + 0.04 * control
                model = (
                    changed_model
                    if information == "oracle"
                    else model_with_point_wind(nominal_model, estimator)
                )
                before_state, before_estimator = state, estimator
                used_version = int(persistent.library_version)
                decision = jax.block_until_ready(
                    controller(
                        state, persistent.params, model, obstacle_windows[100 + control], previous
                    )
                )
                if control == 0:
                    if information not in first_actions:
                        first_actions[information] = np.asarray(decision.action)
                    np.testing.assert_array_equal(decision.action, first_actions[information])
                boundary_states.append(np.asarray(state))
                estimates.append(np.asarray(model.wind_velocity))
                used_parameters.append(_array_tree(persistent.params))
                decisions.append(_array_tree(decision))
                row = {
                    "time_seconds": when,
                    "used_library_version": used_version,
                    "previous_policy_index": int(previous),
                    "fallback_max_hard": float(np.max(decision.values.values[1:])),
                    "fallback_max_smooth": float(np.max(decision.smooth_values[1:])),
                    "eligible_count": int(decision.eligible_candidate_count),
                    "qp_accepted": bool(decision.qp_valid),
                    "execution_mode": int(decision.execution_mode),
                    "degraded": bool(decision.degraded),
                    "update_opportunity": control % 2 == 0 and method != "frozen",
                    "update_completed": False,
                    "estimator_before": {
                        k: v.tolist() for k, v in _array_tree(before_estimator).items()
                    },
                }
                for substep in range(2):
                    following = plant(state, decision.action, changed_model)
                    if information == "estimated":
                        estimator = estimate(estimator, state, following, decision.action)
                    jax.block_until_ready((following, estimator))
                    clearance = _segment_clearance(
                        np.asarray(state), np.asarray(following), scenario
                    )
                    clearances.append(clearance)
                    state = following
                    states.append(np.asarray(state))
                    if clearance <= 0:
                        collided = True
                        collision_interval = [when + substep * 0.02, when + (substep + 1) * 0.02]
                        break
                if row["update_opportunity"] and not collided:
                    persistent, metrics = jax.block_until_ready(
                        learners[method].step(persistent, before_state, model)
                    )
                    row.update(
                        update_completed=True,
                        finite_update=bool(metrics.finite_update_applied),
                        completed_library_version=int(persistent.library_version),
                        gradient_norm=float(metrics.gradient_norm),
                        parameter_update_norm=float(metrics.parameter_update_norm),
                        loss=float(metrics.loss.total),
                    )
                    save_learner_checkpoint(
                        persistent,
                        bundle.spec,
                        trial_config,
                        bundle.actuator,
                        state,
                        directory / f"completed-{control:03d}",
                        metadata={
                            "completed_after_control": control,
                            "next_publication_time": when + 0.04,
                            "previous_policy_index": int(decision.selected_index),
                            **reference_metadata,
                        },
                    )
                rows.append(row)
                previous = decision.selected_index
                if collided:
                    break
            arrays = {
                "dense_states": np.asarray(states),
                "boundary_states": np.asarray(boundary_states),
                "estimated_wind": np.asarray(estimates),
                "physical_segment_clearance": np.asarray(clearances),
                "boundary_time_seconds": np.asarray([row["time_seconds"] for row in rows]),
                **{
                    f"decision__{key}": np.stack([row[key] for row in decisions])
                    for key in decisions[0]
                },
                **{
                    f"used_params__{key}": np.stack([row[key] for row in used_parameters])
                    for key in used_parameters[0]
                },
            }
            np.savez_compressed(directory / "trace.npz", **arrays)
            outcome = {
                "method": method,
                "model_information": information,
                "physical_collision": collided,
                "first_collision_integration_interval": collision_interval,
                "censor_time_seconds": 4.0 + 0.02 * (len(states) - 1),
                "controls": len(rows),
                "finite_updates": sum(row.get("finite_update", False) for row in rows),
                "degraded_controls": sum(row["degraded"] for row in rows),
                "negative_fallback_hard_controls": sum(
                    row["fallback_max_hard"] < 0 for row in rows
                ),
                "initial_state_exactly_shared": bool(np.array_equal(states[0], shared_state)),
                **_physical_metrics(np.asarray(states), clearances, scenario),
            }
            _write(directory / "steps.json", rows)
            _write(directory / "result.json", outcome)
            save_learner_checkpoint(
                persistent,
                bundle.spec,
                trial_config,
                bundle.actuator,
                state,
                directory / "final_checkpoint",
                metadata={**outcome, **reference_metadata},
            )
            outcomes[name] = outcome
            print(f"{name}: {outcome}", flush=True)
    _write(args.output_dir / "summary.json", outcomes)


if __name__ == "__main__":
    main()
