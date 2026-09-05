"""Measure corrected point-model DA-PLCBF kernels as the fallback library grows.

This is an isolated timing utility, not an episode or safety/generalization campaign. It uses the
same near-obstacle state, point wind model, nominal controller, and obstacle geometry for every K.
Only directly callable kernels are timed; full-controller time includes rollout, differentiation,
selection, QP, and postchecks and is not split into fabricated component estimates.
"""

from __future__ import annotations

import argparse
import json
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
from crazyflow.safety.da_plcbf.continuous_version_a import runtime_policy_values
from crazyflow.safety.da_plcbf.online_constant_wind import (
    OnlineConstantWindConfig,
    _make_controller,
    build_cf21b_version_a_resources,
)
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    build_fibonacci_skill_spec,
    build_persistent_skill_learner,
    initialize_skill_actor,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _measure(call: Callable[[], Any], samples: int) -> dict[str, Any]:
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
        "calls_above_20ms": int(np.count_nonzero(np.asarray(durations) > 0.02)),
        "calls_above_50ms": int(np.count_nonzero(np.asarray(durations) > 0.05)),
        "raw_seconds": durations,
    }


def _json_dataclass(instance: Any) -> dict[str, Any]:
    return {
        key: np.asarray(value).tolist() if isinstance(value, (jax.Array, np.ndarray)) else value
        for key, value in asdict(instance).items()
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--policy-counts", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wind", type=float, nargs=3, default=[0.9, 0.55, 0.0])
    parser.add_argument("--position", type=float, nargs=3, default=[4.65, 0.0, 1.4])
    parser.add_argument("--velocity", type=float, nargs=3, default=[0.5, 0.0, 0.0])
    parser.add_argument("--nominal-model-compensation", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.samples < 1 or args.horizon < 1 or any(k < 2 for k in args.policy_counts):
        raise ValueError("samples/horizon must be positive and every policy count at least two")
    if len(set(args.policy_counts)) != len(args.policy_counts):
        raise ValueError("policy counts must be distinct")
    if not np.all(np.isfinite([*args.wind, *args.position, *args.velocity])):
        raise ValueError("wind, position, and velocity must be finite")
    device = jax.devices(args.device)[0]
    resources = jax.device_put(build_cf21b_version_a_resources(), device)
    scenario = replace(constant_wind_scenario(), horizon=args.horizon)
    state = jax.device_put(
        scenario.initial_state.at[:3]
        .set(jnp.asarray(args.position, dtype=jnp.float32))
        .at[7:10]
        .set(jnp.asarray(args.velocity, dtype=jnp.float32)),
        device,
    )
    point_model = model_with_wind(
        resources.model, jax.device_put(jnp.asarray(args.wind, dtype=jnp.float32), device)
    )
    obstacles = jax.device_put(
        scenario_obstacle_window(scenario, scenario.wind_change_step), device
    )
    previous = jax.device_put(jnp.asarray(-1, dtype=jnp.int32), device)
    base_config = OnlineConstantWindConfig(
        seed=args.seed, nominal_model_compensation=args.nominal_model_compensation
    )
    base_config.validate()
    learner_config = PersistentSkillConfig(
        dt=scenario.dt,
        horizon=scenario.horizon,
        acceleration_limit=base_config.fallback_acceleration_limit,
        learning_rate=base_config.learning_rate,
        target_weight=10.0,
        diversity_weight=0.001,
        pairwise_weight=0.005,
        trust_weight=1.0e-3,
        initial_residual_scale=base_config.initial_residual_scale,
        initial_skill_scale=base_config.initial_skill_scale,
        residual_scale=base_config.residual_scale,
        policy_gain=base_config.policy_gain,
        smooth_motor_bounds=base_config.smooth_motor_bounds,
    )
    report: dict[str, Any] = {
        "benchmark": "corrected isolated point-model DA-PLCBF scaling",
        "device": str(device),
        "device_kind": device.device_kind,
        "jax_version": jax.__version__,
        "dtype": str(state.dtype),
        "samples_per_kernel": args.samples,
        "policy_counts": args.policy_counts,
        "horizon": args.horizon,
        "state": np.asarray(state).tolist(),
        "point_model": {
            name: np.asarray(value).tolist() for name, value in point_model._asdict().items()
        },
        "scenario": _json_dataclass(scenario),
        "controller_config": _json_dataclass(base_config),
        "learner_config": _json_dataclass(learner_config),
        "warmup": (
            "discarded learner updates and controller calls with both initial/updated parameters"
        ),
        "methodology": (
            "each callable synchronized on its complete output; compilation excluded by explicit "
            "warmup; full-controller output retains diagnostic arrays; no subtraction of timings"
        ),
        "scope": "isolated kernels at one recorded state/model; no end-to-end deadline guarantee",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for policy_count in args.policy_counts:
        spec = jax.device_put(
            build_fibonacci_skill_spec(
                policy_count=policy_count, horizon_duration=scenario.dt * scenario.horizon
            ),
            device,
        )
        with jax.default_device(device):
            initial_params = initialize_skill_actor(jax.random.key(args.seed), spec, learner_config)
        initial_params = jax.device_put(initial_params, device)
        learner = build_persistent_skill_learner(
            spec, resources.actuator, learner_config, device=device
        )
        initial_learner_state = jax.device_put(
            learner.initialize(initial_params, point_model), device
        )
        controller = _make_controller(
            scenario,
            resources,
            spec,
            learner_config,
            nominal_acceleration_limit=base_config.nominal_acceleration_limit,
            waypoint_position_gain=base_config.waypoint_position_gain,
            waypoint_velocity_gain=base_config.waypoint_velocity_gain,
            device=device,
            policy_alpha=base_config.policy_alpha,
            smooth_min_temperature=base_config.smooth_min_temperature,
            nominal_model_compensation=base_config.nominal_model_compensation,
        )

        @jax.jit
        def hard_values(rollout_states: jax.Array) -> Any:
            return runtime_policy_values(
                rollout_states,
                obstacles,
                obstacle_clearance=scenario.obstacle_clearance,
                ego_radius=scenario.ego_radius,
            )

        started = time.perf_counter()
        jax.block_until_ready(controller(state, initial_params, point_model, obstacles, previous))
        warm_state, warm_metrics = learner.step(initial_learner_state, state, point_model)
        jax.block_until_ready((warm_state, warm_metrics))
        if not bool(np.asarray(warm_metrics.finite_update_applied)):
            raise RuntimeError(f"K={policy_count}: warm-up BPTT update is nonfinite")
        jax.block_until_ready(learner.step(warm_state, state, point_model))
        rollouts = learner.rollout(warm_state.params, state, point_model)
        jax.block_until_ready(rollouts)
        jax.block_until_ready(hard_values(rollouts.states))
        decision = controller(state, warm_state.params, point_model, obstacles, previous)
        jax.block_until_ready(decision)
        warmup_seconds = time.perf_counter() - started
        maximum_value = float(np.max(np.asarray(decision.values.values)))
        metrics = {
            "fallback_policy_count": policy_count,
            "effective_controller_config": _json_dataclass(
                replace(base_config, policy_count=policy_count)
            ),
            "runtime_candidate_count_including_nominal": policy_count + 1,
            "warmup_seconds": warmup_seconds,
            "library_maximum_hard_value": maximum_value,
            "positive_library_H": maximum_value > 0.0,
            "nominal_hard_value": float(np.asarray(decision.values.values)[0]),
            "safe_fallback_count": int(
                np.count_nonzero(np.asarray(decision.values.values)[1:] >= 0)
            ),
            "selected_policy_index": int(np.asarray(decision.selected_index)),
            "selected_policy_dual": float(np.asarray(decision.selected_policy_dual)),
            "qp_valid": bool(np.asarray(decision.qp_valid)),
            "degraded": bool(np.asarray(decision.degraded)),
            "warmed_snapshot_library_version": int(np.asarray(warm_state.library_version)),
            "fallback_forward_rollout": _measure(
                lambda: learner.rollout(warm_state.params, state, point_model), args.samples
            ),
            "hard_value_reduction_on_recorded_fallback_rollouts": _measure(
                lambda: hard_values(rollouts.states), args.samples
            ),
            "full_controller_including_gradient_selection_qp_postchecks": _measure(
                lambda: controller(state, warm_state.params, point_model, obstacles, previous),
                args.samples,
            ),
            "single_persistent_bptt_update": _measure(
                lambda: learner.step(warm_state, state, point_model), args.samples
            ),
        }
        report["results"].append(metrics)
        # Checkpoint each completed K so a later capacity failure does not erase useful evidence.
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "policy_count": policy_count,
                    "positive_H": metrics["positive_library_H"],
                    "controller_median_seconds": metrics[
                        "full_controller_including_gradient_selection_qp_postchecks"
                    ]["median_seconds"],
                    "bptt_median_seconds": metrics["single_persistent_bptt_update"][
                        "median_seconds"
                    ],
                }
            ),
            flush=True,
        )
    print(json.dumps({"output": str(args.output), "completed_counts": args.policy_counts}))


if __name__ == "__main__":
    main()
