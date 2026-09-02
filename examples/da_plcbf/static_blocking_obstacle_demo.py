"""Render the corrected continuous Version-A PL-CBF static-obstacle review case.

The left vehicle executes the unfiltered goal controller and collides with the sphere deliberately
placed on its path.  The right vehicle uses the same nominal controller and the same fixed,
obstacle-agnostic fallback library, but executes the continuous direct-wrench PL-CBF result.  The
recorded trace is rendered afterward, so video generation cannot alter controller timing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.continuous_demo_scenarios import (
    blocking_static_scenario,
    scenario_obstacle_window,
    scenario_safety_limits,
)
from crazyflow.safety.da_plcbf.continuous_version_a import (
    ContinuousVersionAConfig,
    augmented_policy_rollouts,
    continuous_version_a_step,
    obstacle_agnostic_waypoint_callbacks,
    runtime_policy_values,
)
from crazyflow.safety.da_plcbf.direct_wrench import quaternion_to_rotation_matrix
from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
    ComparisonRenderConfig,
    ComparisonVideoTrace,
    MethodVideoTrace,
    ObstacleTrack,
    render_comparison_video,
)
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig, VersionAModel
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator, VersionAFilterConfig

_VIDEO_NAME = "static_nominal_collision_vs_fixed_plcbf_avoidance.mp4"
_METRICS_NAME = "static_nominal_collision_vs_fixed_plcbf_avoidance_metrics.json"


def _resources() -> tuple[VersionAModel, VersionAActuator]:
    raw: dict[str, Any] = load_params("cf21B_500")
    dtype = jnp.float32
    inertia = jnp.asarray(raw["J"], dtype=dtype)
    model = VersionAModel(
        mass=jnp.asarray(raw["mass"], dtype=dtype),
        gravity_vec=jnp.asarray(raw["gravity_vec"], dtype=dtype),
        inertia=inertia,
        inertia_inv=jnp.linalg.inv(inertia),
        drag_matrix=jnp.asarray(raw["drag_matrix"], dtype=dtype),
        wind_velocity=jnp.zeros(3, dtype=dtype),
        external_force=jnp.zeros(3, dtype=dtype),
        external_torque=jnp.zeros(3, dtype=dtype),
    )
    actuator = VersionAActuator(
        arm_length=jnp.asarray(raw["L"], dtype=dtype),
        thrust_to_torque=jnp.asarray(raw["thrust2torque"], dtype=dtype),
        mixing_matrix=jnp.asarray(raw["mixing_matrix"], dtype=dtype),
        thrust_min=jnp.asarray(raw["thrust_min"], dtype=dtype),
        thrust_max=jnp.asarray(raw["thrust_max"], dtype=dtype),
    )
    return model, actuator


def _world_force_intervention(state: jax.Array, wrench_delta: jax.Array) -> np.ndarray:
    body_force = jnp.asarray([0.0, 0.0, wrench_delta[0]], dtype=state.dtype)
    rotation = quaternion_to_rotation_matrix(state[3:7])
    return np.asarray(rotation @ body_force, dtype=np.float64)


def _method_trace(records: dict[str, list[np.ndarray | float | int]]) -> MethodVideoTrace:
    float_fields = (
        "position",
        "quaternion_xyzw",
        "nominal_rollout",
        "fallback_rollouts",
        "selected_rollout",
        "intervention_world",
        "intervention_norm",
        "descriptors",
        "diversity_loss",
        "descriptor_target_loss",
        "gradient_norm",
        "parameter_update_norm",
        "minimum_library_value",
    )
    arrays = {name: np.asarray(records[name], dtype=np.float64) for name in float_fields}
    return MethodVideoTrace(
        position=arrays["position"],
        quaternion_xyzw=arrays["quaternion_xyzw"],
        nominal_rollout=arrays["nominal_rollout"],
        fallback_rollouts=arrays["fallback_rollouts"],
        fallback_safe=np.asarray(records["fallback_safe"], dtype=np.bool_),
        selected_policy=np.asarray(records["selected_policy"], dtype=np.int32),
        selected_rollout=arrays["selected_rollout"],
        intervention_world=arrays["intervention_world"],
        intervention_norm=arrays["intervention_norm"],
        descriptors=arrays["descriptors"],
        library_version=np.asarray(records["library_version"], dtype=np.int64),
        cumulative_gradient_steps=np.asarray(records["cumulative_gradient_steps"], dtype=np.int64),
        diversity_loss=arrays["diversity_loss"],
        descriptor_target_loss=arrays["descriptor_target_loss"],
        gradient_norm=arrays["gradient_norm"],
        parameter_update_norm=arrays["parameter_update_norm"],
        minimum_library_value=arrays["minimum_library_value"],
    )


def _empty_records() -> dict[str, list[np.ndarray | float | int]]:
    return {
        name: []
        for name in (
            "position",
            "quaternion_xyzw",
            "nominal_rollout",
            "fallback_rollouts",
            "fallback_safe",
            "selected_policy",
            "selected_rollout",
            "intervention_world",
            "intervention_norm",
            "descriptors",
            "library_version",
            "cumulative_gradient_steps",
            "diversity_loss",
            "descriptor_target_loss",
            "gradient_norm",
            "parameter_update_norm",
            "minimum_library_value",
        )
    }


def _append_context(
    records: dict[str, list[np.ndarray | float | int]],
    *,
    state: jax.Array,
    candidate_states: jax.Array,
    values: jax.Array,
    selected_augmented_index: int,
    intervention_world: np.ndarray,
    intervention_norm: float,
    descriptor_targets: np.ndarray,
) -> None:
    states = np.asarray(candidate_states, dtype=np.float64)
    hard_values = np.asarray(values, dtype=np.float64)
    fallbacks = states[1:, 1:, :3]
    descriptors = states[1:, -1, :3] - np.asarray(state[:3], dtype=np.float64)
    selected_policy = selected_augmented_index - 1
    records["position"].append(np.asarray(state[:3], dtype=np.float64))
    records["quaternion_xyzw"].append(np.asarray(state[3:7], dtype=np.float64))
    records["nominal_rollout"].append(states[0, 1:, :3])
    records["fallback_rollouts"].append(fallbacks)
    records["fallback_safe"].append(hard_values[1:] >= 0.0)
    records["selected_policy"].append(selected_policy)
    records["selected_rollout"].append(states[selected_augmented_index, 1:, :3])
    records["intervention_world"].append(intervention_world)
    records["intervention_norm"].append(intervention_norm)
    records["descriptors"].append(descriptors)
    records["library_version"].append(0)
    records["cumulative_gradient_steps"].append(0)
    records["diversity_loss"].append(0.0)
    records["descriptor_target_loss"].append(
        float(np.mean(np.square(descriptors - descriptor_targets)))
    )
    records["gradient_norm"].append(0.0)
    records["parameter_update_norm"].append(0.0)
    records["minimum_library_value"].append(float(np.min(hard_values[1:])))


def run_static_demo() -> tuple[ComparisonVideoTrace, dict[str, object]]:
    """Run and validate the one blocking-obstacle comparison used for review."""
    scenario = blocking_static_scenario()
    model, actuator = _resources()
    quad_config = QuadPolicyConfig()
    controller_config = ContinuousVersionAConfig(
        dt=scenario.dt, horizon=60, obstacle_clearance=0.15
    )
    if scenario.horizon != controller_config.horizon:
        raise RuntimeError("the review scenario must use the exact H=60 certificate horizon")
    nominal_rollout, fallback_rollouts = obstacle_agnostic_waypoint_callbacks(
        scenario.goal_position,
        scenario.goal_velocity,
        scenario.skill_displacements,
        actuator,
        quad_config,
        dt=scenario.dt,
        horizon=controller_config.horizon,
    )
    obstacles = scenario_obstacle_window(scenario, 0)
    safety_limits = scenario_safety_limits(scenario)
    barrier_config = VersionABarrierConfig(obstacle_clearance=0.15)
    filter_config = VersionAFilterConfig(policy_alpha=2.0)

    def nominal_context(state: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        candidates = augmented_policy_rollouts(
            state, nominal_rollout, fallback_rollouts, model, horizon=controller_config.horizon
        )
        values = runtime_policy_values(
            candidates.states, obstacles, obstacle_clearance=controller_config.obstacle_clearance
        )
        return candidates.states, candidates.wrenches, values.values

    context_step = jax.jit(nominal_context)
    filtered_step = jax.jit(
        lambda state, previous_index: continuous_version_a_step(
            state,
            nominal_rollout,
            fallback_rollouts,
            obstacles,
            model,
            actuator,
            safety_limits,
            barrier_config,
            filter_config,
            controller_config,
            previous_policy_index=previous_index,
        )
    )
    plant_step = jax.jit(
        lambda state, wrench: direct_wrench_symplectic_step(state, wrench, model, scenario.dt)
    )

    nominal_state = scenario.initial_state
    filtered_state = scenario.initial_state
    previous_index = jnp.asarray(-1, dtype=jnp.int32)
    nominal_records = _empty_records()
    filtered_records = _empty_records()
    obstacle_center = np.asarray(scenario.obstacle_initial_centers[0], dtype=np.float64)
    nominal_minimum = float("inf")
    filtered_minimum = float("inf")
    maximum_intervention = 0.0
    intervention_steps = 0
    fallback_execution_steps = 0
    degraded_steps = 0
    descriptor_targets = np.asarray(scenario.skill_displacements, dtype=np.float64)

    for step_index in range(scenario.steps + 1):
        left_states, left_wrenches, left_values = context_step(nominal_state)
        decision = filtered_step(filtered_state, previous_index)
        selected_index = int(np.asarray(decision.selected_index))
        intervention = float(np.asarray(decision.qp_intervention_norm))
        delta_wrench = decision.action - decision.nominal_action

        _append_context(
            nominal_records,
            state=nominal_state,
            candidate_states=left_states,
            values=left_values,
            selected_augmented_index=0,
            intervention_world=np.zeros(3, dtype=np.float64),
            intervention_norm=0.0,
            descriptor_targets=descriptor_targets,
        )
        _append_context(
            filtered_records,
            state=filtered_state,
            candidate_states=decision.candidates.states,
            values=decision.values.values,
            selected_augmented_index=selected_index,
            intervention_world=_world_force_intervention(filtered_state, delta_wrench),
            intervention_norm=intervention,
            descriptor_targets=descriptor_targets,
        )
        nominal_minimum = min(
            nominal_minimum, float(np.linalg.norm(np.asarray(nominal_state[:3]) - obstacle_center))
        )
        filtered_minimum = min(
            filtered_minimum,
            float(np.linalg.norm(np.asarray(filtered_state[:3]) - obstacle_center)),
        )
        maximum_intervention = max(maximum_intervention, intervention)
        intervention_steps += int(intervention > 1e-3)
        fallback_execution_steps += int(np.asarray(decision.used_fallback))
        degraded_steps += int(np.asarray(decision.degraded))

        if step_index < scenario.steps:
            nominal_state = plant_step(nominal_state, left_wrenches[0, 0])
            filtered_state = plant_step(filtered_state, decision.action)
            previous_index = decision.selected_index

    physical_radius = float(np.asarray(scenario.obstacle_radii[0]))
    inflated_radius = physical_radius + controller_config.obstacle_clearance
    final_goal_distance = float(
        np.linalg.norm(np.asarray(filtered_state[:3] - scenario.goal_position))
    )
    checks = {
        "nominal_collides_with_physical_obstacle": nominal_minimum < physical_radius,
        "plcbf_stays_outside_inflated_shell": filtered_minimum > inflated_radius,
        "plcbf_intervenes": maximum_intervention > 1e-3,
        "no_degraded_controller_steps": degraded_steps == 0,
        "plcbf_resumes_and_reaches_goal": final_goal_distance < 0.1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("static review gate failed: " + ", ".join(failed))

    sample_count = scenario.steps + 1
    time_seconds = np.arange(sample_count, dtype=np.float64) * scenario.dt
    centers = np.broadcast_to(obstacle_center, (sample_count, 3)).copy()
    trace = ComparisonVideoTrace(
        time_seconds=time_seconds,
        goal_position=np.asarray(scenario.goal_position, dtype=np.float64),
        obstacles=(
            ObstacleTrack(
                centers=centers,
                physical_radius=physical_radius,
                inflated_radius=inflated_radius,
                label="blocking sphere",
            ),
        ),
        true_wind=np.zeros((sample_count, 3), dtype=np.float64),
        estimated_wind=np.zeros((sample_count, 3), dtype=np.float64),
        wind_change_time=float(time_seconds[-1]),
        descriptor_targets=descriptor_targets,
        fixed=_method_trace(nominal_records),
        adaptive=_method_trace(filtered_records),
        title=(
            "Static blocking obstacle: nominal collision vs continuous fixed-library "
            "PL-CBF avoidance"
        ),
        left_label="NOMINAL ONLY — COLLIDES",
        right_label="FIXED-LIBRARY CONTINUOUS PL-CBF — AVOIDS",
        show_wind_change_banner=False,
    )
    trace.validate()
    metrics: dict[str, object] = {
        "scenario": scenario.name,
        "purpose": "nominal collision versus fixed-library continuous PL-CBF avoidance",
        "controller": {
            "dt_seconds": scenario.dt,
            "control_hz": 1.0 / scenario.dt,
            "steps": scenario.steps,
            "horizon": controller_config.horizon,
            "obstacle_clearance": controller_config.obstacle_clearance,
            "policy_alpha": filter_config.policy_alpha,
        },
        "geometry": {
            "physical_obstacle_radius": physical_radius,
            "inflated_obstacle_radius": inflated_radius,
        },
        "results": {
            "nominal_minimum_obstacle_distance": nominal_minimum,
            "filtered_minimum_obstacle_distance": filtered_minimum,
            "filtered_minimum_shell_margin": filtered_minimum - inflated_radius,
            "maximum_qp_intervention_norm": maximum_intervention,
            "qp_intervention_steps_above_1e-3": intervention_steps,
            "fallback_execution_steps": fallback_execution_steps,
            "degraded_steps": degraded_steps,
            "filtered_final_goal_distance": final_goal_distance,
        },
        "checks": checks,
        "all_checks_passed": True,
    }
    return trace, metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for the metrics JSON and optional MuJoCo MP4",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="rendered video frame rate")
    parser.add_argument(
        "--no-render", action="store_true", help="run the numerical gate and write metrics only"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / _METRICS_NAME
    video_path = args.output_dir / _VIDEO_NAME
    if metrics_path.exists() or metrics_path.is_symlink():
        raise FileExistsError(metrics_path)
    if not args.no_render and (video_path.exists() or video_path.is_symlink()):
        raise FileExistsError(video_path)

    trace, metrics = run_static_demo()
    video_result = None
    if not args.no_render:
        video_result = render_comparison_video(
            trace, video_path, ComparisonRenderConfig(fps=args.fps)
        )
        metrics["video"] = {**asdict(video_result), "path": str(video_result.path)}
    with metrics_path.open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {"metrics": str(metrics_path), "video": str(video_path) if video_result else None}
        )
    )


if __name__ == "__main__":
    main()
