"""Bounded four-second oracle route witnesses from recorded navigation failure boundaries.

Routes remain outside actor training and runtime filtering. A failed bounded route set does not
establish impossibility. Full diagnostic trajectories are retained, including failed continuations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.direct_wrench import wrench_to_motor_forces
from crazyflow.safety.da_plcbf.feasibility_reference import _collision_clearance
from crazyflow.safety.da_plcbf.learner_checkpoint import load_learner_checkpoint
from crazyflow.safety.da_plcbf.navigation_world import build_navigation_world
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig, waypoint_nominal_wrench
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from examples.da_plcbf.navigation_demo import load_world_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = jax.devices("cpu")[0]
    bundle = load_learner_checkpoint(args.checkpoint, device=device)
    world = build_navigation_world(load_world_config(args.run / "world.json"))
    with np.load(args.run / "navigation_comparison.npz") as archive:
        trace = {name: archive[name] for name in archive.files}
    dt, hold = world.config.dt, world.config.control_interval_steps
    controls = round(4.0 / world.config.control_period)
    actuator = bundle.actuator
    authority = QuadPolicyConfig(acceleration_limit=1.2)
    offsets = [("direct", 0.0, 0.0, 0.0)]
    offsets += [
        (f"lateral_{side:+g}_{forward:g}", forward, side, 0.0)
        for forward in (0.2, 0.6)
        for side in (-1.0, -0.6, 0.6, 1.0)
    ]
    offsets += [(f"vertical_{height:+g}", 0.6, 0.0, height) for height in (-0.6, 0.6)]
    offsets += [
        (f"combined_{side:+g}_{height:+g}", 0.4, side, height)
        for side in (-0.7, 0.7)
        for height in (-0.5, 0.5)
    ]
    protocol = {
        "indices": [165, 176],
        "maximum_local_seconds": 4.0,
        "route_offsets": offsets,
        "frame": "horizontal forward velocity and perpendicular lateral axis",
        "acceleration_limit_mps2": 1.2,
        "position_gain": 2.0,
        "velocity_gain": 2.8,
        "intermediate_arrival_radius_m": 0.35,
        "final_goal_radius_m": 0.25,
        "final_speed_limit_mps": 0.3,
        "command_hold_steps": hold,
        "integration_dt": dt,
        "scope": "Oracle numerical authority diagnostic only; no PL-CBF certificate or impossibility claim",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    @jax.jit
    def rollout(initial: jax.Array, routes: jax.Array, models: object) -> object:
        def single(route: jax.Array) -> object:
            def step(carry: object, model: object) -> object:
                current, waypoint = carry
                waypoint = jnp.where(
                    (waypoint == 0) & (jnp.linalg.norm(current[:3] - route[0]) <= 0.35), 1, waypoint
                )
                command = waypoint_nominal_wrench(
                    current,
                    route[waypoint],
                    jnp.zeros(3),
                    model,
                    actuator,
                    authority,
                    position_gain=2.0,
                    velocity_gain=2.8,
                    model_compensation=True,
                )

                def integrate(state: jax.Array, _: object) -> object:
                    following = direct_wrench_symplectic_step(state, command.wrench, model, dt)
                    return following, following

                following, nodes = jax.lax.scan(integrate, current, None, length=hold)
                motors = wrench_to_motor_forces(
                    command.wrench,
                    L=actuator.arm_length,
                    thrust2torque=actuator.thrust_to_torque,
                    mixing_matrix=actuator.mixing_matrix,
                )
                return (following, waypoint), (
                    nodes,
                    command.wrench,
                    motors,
                    waypoint,
                    command.input_valid,
                )

            _, history = jax.lax.scan(step, (initial, jnp.asarray(0)), models)
            return history

        return jax.vmap(single)(routes)

    arrays, summaries = {}, []
    for index in protocol["indices"]:
        initial = np.asarray(trace["fixed_full_state"][index])
        goal = np.asarray(trace["fixed_goal_position"][index])
        forward = initial[7:10].copy()
        forward[2] = 0
        if np.linalg.norm(forward) < 0.1:
            forward = goal - initial[:3]
            forward[2] = 0
        forward /= np.linalg.norm(forward)
        lateral = np.asarray((-forward[1], forward[0], 0))
        routes = np.asarray(
            [
                [
                    goal
                    if label == "direct"
                    else initial[:3] + ahead * forward + side * lateral + np.asarray((0, 0, up)),
                    goal,
                ]
                for label, ahead, side, up in offsets
            ],
            dtype=np.float32,
        )
        when = float(trace["time_seconds"][index])
        boundary_times = when + np.arange(controls) * world.config.control_period
        model_list = [world.dynamics_at(float(t), bundle.point_model).model for t in boundary_times]
        models = jax.tree.map(lambda *items: jnp.stack(items), *model_list)
        history = jax.block_until_ready(rollout(jnp.asarray(initial), jnp.asarray(routes), models))
        nodes, actions, motors, waypoint_indices, valid = map(np.asarray, history)
        states = np.concatenate(
            (
                np.tile(initial, (len(routes), 1, 1)),
                nodes.reshape(len(routes), controls * hold, 13),
            ),
            axis=1,
        )
        times = when + np.arange(states.shape[1]) * dt
        centers, velocities = world.obstacle_kinematics(times)
        arrays.update(
            {
                f"{index}_states": states,
                f"{index}_actions": actions,
                f"{index}_motor_forces": motors,
                f"{index}_waypoint_indices": waypoint_indices,
                f"{index}_routes": routes,
                f"{index}_time_seconds": times,
                f"{index}_obstacle_centers": centers,
                f"{index}_obstacle_velocities": velocities,
            }
        )
        for name, values in zip(models._fields, models, strict=True):
            arrays[f"{index}_point_model_{name}"] = np.asarray(values)
        for candidate, (label, *_offset) in enumerate(offsets):
            trajectory = states[candidate]
            clearance, closest = _collision_clearance(
                trajectory[:, :3],
                centers,
                world.obstacle_radii,
                np.full(len(times), world.config.ego_radius),
            )
            quaternion = trajectory[:, 3:7]
            tilt = np.arccos(np.clip(1 - 2 * np.sum(quaternion[:, :2] ** 2, axis=1), -1, 1))
            arena = min(
                np.min(
                    trajectory[:, :3] - world.config.arena_lower - np.asarray((0.08, 0.08, 0.08))
                ),
                np.min(
                    world.config.arena_upper - trajectory[:, :3] - np.asarray((0.08, 0.08, 0.08))
                ),
            )
            motor_margin = min(
                np.min(motors[candidate] - np.asarray(actuator.thrust_min)),
                np.min(np.asarray(actuator.thrust_max) - motors[candidate]),
            )
            speed = np.linalg.norm(trajectory[:, 7:10], axis=1)
            rate = np.linalg.norm(trajectory[:, 10:13], axis=1)
            goal_distance = float(np.linalg.norm(trajectory[-1, :3] - goal))
            checks = {
                "finite_valid_execution": bool(
                    np.all(valid[candidate]) and np.isfinite(trajectory).all()
                ),
                "physical_collision_clear": clearance > 0,
                "inflated_shell_clear": clearance > world.config.obstacle_clearance,
                "motor_limits": motor_margin >= -3e-6,
                "arena_limits": arena >= -2e-5,
                "speed_limit": float(max(speed)) <= world.config.speed_max + 2e-5,
                "tilt_limit": float(max(tilt)) <= world.config.tilt_max_radians + 2e-5,
                "angular_rate_limit": float(max(rate)) <= world.config.angular_rate_max + 2e-5,
                "goal_reached_and_slow": goal_distance <= 0.25 and speed[-1] <= 0.3,
            }
            checks = {key: bool(value) for key, value in checks.items()}
            row = {
                "control_index": index,
                "start_time_seconds": when,
                "route": label,
                "checks": checks,
                "feasible_witness_found": all(checks.values()),
                "minimum_physical_clearance_m": clearance,
                "minimum_inflated_clearance_m": clearance - world.config.obstacle_clearance,
                "closest_interval_start_seconds": float(times[closest]),
                "minimum_motor_margin_N": float(motor_margin),
                "minimum_arena_margin_m": float(arena),
                "maximum_speed_mps": float(max(speed)),
                "maximum_tilt_radians": float(max(tilt)),
                "maximum_angular_rate_rps": float(max(rate)),
                "final_goal_distance_m": goal_distance,
                "final_speed_mps": float(speed[-1]),
            }
            summaries.append(row)
        print(
            json.dumps(
                {
                    "index": index,
                    "successful_routes": [
                        row["route"]
                        for row in summaries
                        if row["control_index"] == index and row["feasible_witness_found"]
                    ],
                }
            ),
            flush=True,
        )
    arrays["obstacle_radii"] = world.obstacle_radii
    arrays["ego_radius"] = np.asarray(world.config.ego_radius)
    for name, value in zip(actuator._fields, actuator, strict=True):
        arrays[f"actuator_{name}"] = np.asarray(value)
    np.savez_compressed(args.output / "full_route_records.npz", **arrays)
    (args.output / "results.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "world": world.metadata(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "rows": summaries,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
