"""Measure complete corrected controllers on explicitly verified execution branches.

These prescribed computational probes are not episodes or safety-success evidence. A bounded
state search finds a naturally accepted full active-set solve without disabling the exact QP
shortcut. Each requested branch must be observed before its timing is labelled as that branch.
The same immutable actor, model, controller configuration, and two-obstacle shape are retained
throughout; only the recorded physical state and obstacle positions vary. No learner updates,
host-side simulation, or serialization are included in the synchronized controller-call samples.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.continuous_demo_scenarios import (
    constant_wind_scenario,
    model_with_wind,
    scenario_obstacle_window,
)
from crazyflow.safety.da_plcbf.continuous_version_a import EXECUTION_MODES, QP_REJECTION_REASONS
from crazyflow.safety.da_plcbf.learner_checkpoint import load_learner_checkpoint
from crazyflow.safety.da_plcbf.online_constant_wind import (
    OnlineConstantWindConfig,
    _make_controller,
    build_cf21b_version_a_resources,
)
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    build_fibonacci_skill_spec,
    initialize_skill_actor,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from crazyflow.safety.da_plcbf.continuous_version_a import (
        ContinuousVersionAStep,
        RuntimeObstacleTrajectories,
    )


BRANCHES = ("accepted_fast_qp", "accepted_full_qp", "certified_fallback", "emergency")


def _plain(value: Any) -> Any:
    """Produce strict JSON; mathematical nonfinite diagnostics have explicit null slots."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (jax.Array, np.ndarray)):
        return _plain(np.asarray(value).tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _diagnostics(decision: ContinuousVersionAStep) -> dict[str, Any]:
    filtered = decision.continuous_filter
    return _plain(
        {
            "execution_mode": EXECUTION_MODES[int(decision.execution_mode)],
            "qp_valid": bool(decision.qp_valid),
            "qp_fast_path_used": bool(filtered.qp_fast_path_used),
            "qp_solver_feasible": bool(filtered.qp.feasible),
            "qp_kkt_valid": bool(filtered.qp_kkt_valid),
            "used_fallback": bool(decision.used_fallback),
            "fallback_valid": bool(decision.fallback_valid),
            "used_emergency": bool(decision.used_emergency),
            "used_midpoint": bool(decision.used_midpoint),
            "degraded": bool(decision.degraded),
            "selected_index": int(decision.selected_index),
            "selected_hard_value": decision.values.values[decision.selected_index],
            "selected_smooth_value": decision.selected_smooth_value,
            "selected_policy_dual": decision.selected_policy_dual,
            "executed_policy_dual": decision.executed_policy_dual,
            "selectable_candidate_count": int(decision.eligible_candidate_count),
            "hard_values": decision.values.values,
            "smooth_values": decision.smooth_values,
            "gradient_valid": decision.gradient_valid,
            "action": decision.action,
            "nominal_action": decision.nominal_action,
            "applied_held_margin": decision.applied_held_margin,
            "applied_held_operational_margin": decision.applied_held_operational_margin,
            "applied_held_operational_residual": decision.applied_held_operational_residual,
            "applied_held_operational_passed": bool(decision.applied_held_operational_passed),
            "qp_rejection_reasons": [
                reason
                for reason, rejected in zip(
                    QP_REJECTION_REASONS, np.asarray(decision.qp_rejection_flags), strict=True
                )
                if rejected
            ],
        }
    )


def _matches(branch: str, diagnostic: dict[str, Any]) -> bool:
    if branch == "accepted_fast_qp":
        return diagnostic["qp_valid"] and diagnostic["qp_fast_path_used"]
    if branch == "accepted_full_qp":
        return diagnostic["qp_valid"] and not diagnostic["qp_fast_path_used"]
    if branch == "certified_fallback":
        return (
            diagnostic["used_fallback"]
            and diagnostic["fallback_valid"]
            and not diagnostic["degraded"]
        )
    return diagnostic["used_emergency"] and not diagnostic["used_midpoint"]


def _measure(call: Callable[[], Any], samples: int, interval: float) -> dict[str, Any]:
    durations = []
    for _ in range(samples):
        started = time.perf_counter()
        jax.block_until_ready(call())
        durations.append(time.perf_counter() - started)
    return {
        "samples": samples,
        "median_seconds": float(np.median(durations)),
        "p95_seconds": float(np.percentile(durations, 95)),
        "maximum_seconds": float(np.max(durations)),
        "minimum_seconds": float(np.min(durations)),
        "calls_above_control_interval": int(np.count_nonzero(np.asarray(durations) > interval)),
        "raw_seconds": durations,
    }


def _fixtures(
    branch: str,
    base_state: jax.Array,
    base_obstacles: RuntimeObstacleTrajectories,
    inflated_radius: float,
) -> Iterator[tuple[str, jax.Array, RuntimeObstacleTrajectories]]:
    """A fixed, bounded discovery list; no runtime-controller changes or actor adaptation."""
    state = base_state.at[:3].set(jnp.asarray([0.0, 0.0, 1.4], dtype=base_state.dtype))

    def obstacle_at(center: tuple[float, float, float]) -> RuntimeObstacleTrajectories:
        return base_obstacles._replace(
            centers=base_obstacles.centers.at[:, 0].set(jnp.asarray(center, state.dtype))
        )

    if branch == "accepted_fast_qp":
        yield "clear_hover_start", state, base_obstacles
        return
    if branch == "certified_fallback":
        for hard_margin in (0.001, 0.0002, 0.003):
            distance = math.sqrt(inflated_radius**2 + hard_margin)
            yield (
                f"positive_hard_margin_{hard_margin}_obstacle_behind",
                state,
                obstacle_at((-distance, 0.0, 1.4)),
            )
        return
    if branch == "emergency":
        # This begins inside the requested clearance shell; it intentionally tests degraded
        # execution cost and must never be counted as a successful avoidance episode.
        yield (
            "already_inside_requested_clearance_shell",
            state,
            obstacle_at((-(inflated_radius - 0.08), 0.0, 1.4)),
        )
        return
    # Near the speed limit, tilt can make the nominal action violate the speed face. The
    # projection onto only the selected policy face then fails the operational-row check.
    # A braking pitch rate provides positive held-interval slack after the instantaneous speed
    # face is enforced; zero-rate versions may correctly fail the second-substep postcheck.
    for speed, pitch in ((3.3, 0.3), (3.0, 0.3), (2.5, 0.45)):
        for pitch_rate in (-2.0, -1.0, -4.0):
            pitched = state.at[3:7].set(
                jnp.asarray([0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2)], state.dtype)
            )
            pitched = pitched.at[7].set(speed).at[11].set(pitch_rate)
            yield f"braking_speed_{speed}_pitch_{pitch}_rate_{pitch_rate}", pitched, base_obstacles
    for speed in (3.48, 3.3, 3.0, 2.5):
        for pitch in (0.3, 0.45, 0.6, 0.75):
            pitched = state.at[3:7].set(
                jnp.asarray([0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2)], state.dtype)
            )
            pitched = pitched.at[7].set(speed)
            yield f"operational_speed_{speed}_pitch_{pitch}", pitched, base_obstacles
    # A second route is an active policy face together with an actuator/operational face.
    for position in (4.65, 4.8, 4.95):
        for speed in (0.5, 1.0, 1.5):
            for pitch in (0.0, 0.25):
                candidate = state.at[0].set(position).at[7].set(speed)
                candidate = candidate.at[3:7].set(
                    jnp.asarray([0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2)], state.dtype)
                )
                name = f"obstacle_x_{position}_speed_{speed}_pitch_{pitch}"
                yield name, candidate, base_obstacles


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--policy-count", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--control-interval-steps", type=int, default=2)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wind", type=float, nargs=3, default=[0.9, 0.55, 0.0])
    parser.add_argument("--nominal-model-compensation", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.samples < 1 or not np.all(np.isfinite(args.wind)):
        raise ValueError("samples must be positive and wind finite")
    device = jax.devices(args.device)[0]
    resources = jax.device_put(build_cf21b_version_a_resources(), device)
    if args.checkpoint is not None:
        checkpoint = load_learner_checkpoint(args.checkpoint, device=device)
        params, spec, learner_config = checkpoint.state.params, checkpoint.spec, checkpoint.config
        if args.horizon is not None and args.horizon != learner_config.horizon:
            raise ValueError("--horizon must match the saved checkpoint")
        if args.policy_count is not None and args.policy_count != spec.latent_codes.shape[0]:
            raise ValueError("--policy-count must match the saved checkpoint")
        if not all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(resources.actuator, checkpoint.actuator, strict=True)
        ):
            raise ValueError("checkpoint actuator differs from the benchmark cf21B actuator")
        source = {
            "kind": "immutable_restored_checkpoint",
            "npz_path": str(checkpoint.npz_path.resolve()),
            "sha256": checkpoint.sha256,
            "library_version": int(checkpoint.state.library_version),
            "cumulative_gradient_steps": int(checkpoint.state.cumulative_gradient_steps),
        }
    else:
        policy_count = 16 if args.policy_count is None else args.policy_count
        horizon = 60 if args.horizon is None else args.horizon
        learner_config = PersistentSkillConfig(horizon=horizon, smooth_motor_bounds=False)
        learner_config.validate()
        spec = jax.device_put(
            build_fibonacci_skill_spec(
                policy_count=policy_count, horizon_duration=learner_config.dt * horizon
            ),
            device,
        )
        with jax.default_device(device):
            params = initialize_skill_actor(jax.random.key(args.seed), spec, learner_config)
        source = {"kind": "untrained_deterministic_initialization", "seed": args.seed}
    if not 1 <= args.control_interval_steps <= learner_config.horizon:
        raise ValueError("control interval steps must lie between one and the rollout horizon")
    scenario = replace(
        constant_wind_scenario(), dt=learner_config.dt, horizon=learner_config.horizon
    )
    point_model = model_with_wind(
        resources.model, jax.device_put(jnp.asarray(args.wind, dtype=jnp.float32), device)
    )
    config = OnlineConstantWindConfig(nominal_model_compensation=args.nominal_model_compensation)
    controller = _make_controller(
        scenario,
        resources,
        spec,
        learner_config,
        nominal_acceleration_limit=config.nominal_acceleration_limit,
        waypoint_position_gain=config.waypoint_position_gain,
        waypoint_velocity_gain=config.waypoint_velocity_gain,
        device=device,
        policy_alpha=config.policy_alpha,
        smooth_min_temperature=config.smooth_min_temperature,
        nominal_model_compensation=config.nominal_model_compensation,
        control_interval_steps=args.control_interval_steps,
    )
    params = jax.device_put(params, device)
    obstacles = jax.device_put(scenario_obstacle_window(scenario, 0), device)
    initial_state = jax.device_put(scenario.initial_state, device)
    previous = jax.device_put(jnp.asarray(-1, dtype=jnp.int32), device)
    control_interval = scenario.dt * args.control_interval_steps
    report: dict[str, Any] = {
        "benchmark": "verified complete-controller execution branches",
        "device": str(device),
        "device_kind": device.device_kind,
        "jax_version": jax.__version__,
        "dtype": str(initial_state.dtype),
        "actor_source": source,
        "fallback_policy_count": spec.latent_codes.shape[0],
        "runtime_candidate_count_including_nominal": spec.latent_codes.shape[0] + 1,
        "learner_config": asdict(learner_config),
        "nominal_controller_config": asdict(config),
        "point_model": point_model._asdict(),
        "scenario": asdict(scenario),
        "integration_dt": scenario.dt,
        "control_interval_steps": args.control_interval_steps,
        "control_interval_seconds": control_interval,
        "horizon_steps": scenario.horizon,
        "horizon_seconds": scenario.dt * scenario.horizon,
        "samples_per_branch": args.samples,
        "methodology": (
            "one immutable actor snapshot; bounded prescribed-state discovery; branch verified "
            "using controller flags; complete outputs synchronized; compilation and discovery "
            "excluded; each branch additionally warmed twice; no fabricated timing subtraction"
        ),
        "scope": (
            "isolated computational probes, not safety episodes or worst-case execution-time "
            "bounds; requested branch absent means no matching-state timing is reported; null "
            "diagnostic numbers denote nonfinite values"
        ),
        "results": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for branch in BRANCHES:
        attempts = []
        branch_result: dict[str, Any] = {"found": False, "attempts": attempts}
        report["results"][branch] = branch_result
        inflated_radius = (
            float(scenario.obstacle_radii[0]) + scenario.obstacle_clearance + scenario.ego_radius
        )
        for name, state, obstacle_window in _fixtures(
            branch, initial_state, obstacles, inflated_radius
        ):
            state, obstacle_window = jax.device_put((state, obstacle_window), device)

            def call() -> Any:
                return controller(state, params, point_model, obstacle_window, previous)

            decision = jax.block_until_ready(call())
            diagnostic = _diagnostics(decision)
            attempts.append({"name": name, "diagnostics": diagnostic})
            if not _matches(branch, diagnostic):
                continue
            jax.block_until_ready(call())
            jax.block_until_ready(call())
            branch_result.update(
                found=True,
                fixture_name=name,
                state=state,
                obstacle_window=obstacle_window._asdict(),
                diagnostics=diagnostic,
                full_controller=_measure(call, args.samples, control_interval),
            )
            break
        args.output.write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "branch": branch,
                    "found": branch_result["found"],
                    "attempts": len(attempts),
                    "median_seconds": branch_result.get("full_controller", {}).get(
                        "median_seconds"
                    ),
                }
            ),
            flush=True,
        )
    report["all_requested_branches_found"] = all(
        result["found"] for result in report["results"].values()
    )
    args.output.write_text(json.dumps(_plain(report), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
