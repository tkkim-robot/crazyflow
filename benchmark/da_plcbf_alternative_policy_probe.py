"""Bounded same-state alternative-certificate diagnosis for a saved frozen branch.

The ordinary saved decision is reproduced first. Diagnostic hysteresis then retains each already
eligible candidate in turn; eligibility, nominal command, QP, and all held-action checks remain
unchanged. No policy, dynamics, or experiment artifact is modified. This is a local diagnostic,
not a revised controller or evidence that the resulting alternative would complete an episode.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.competent_library_experiment import (
    CompetentExperimentConfig,
    _controller,
    _scenario,
)
from crazyflow.safety.da_plcbf.continuous_demo_scenarios import (
    model_with_wind,
    scenario_obstacle_window,
    scenario_safety_limits,
)
from crazyflow.safety.da_plcbf.continuous_version_a import (
    EXECUTION_MODES,
    QP_REJECTION_REASONS,
    ContinuousVersionAConfig,
    PolicyRollouts,
    _held_action_check,
    continuous_version_a_step,
    rollout_waypoint_library,
)
from crazyflow.safety.da_plcbf.learner_checkpoint import load_learner_checkpoint
from crazyflow.safety.da_plcbf.online_constant_wind import VersionAResources
from crazyflow.safety.da_plcbf.persistent_skill_learner import rollout_skill_library
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.selector import SelectionConfig
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig
from crazyflow.safety.da_plcbf.version_a_filter import VersionAFilterConfig


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (jax.Array, np.ndarray)):
        return _plain(np.asarray(value).tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--method", choices=("fixed", "compensated"), default="compensated")
    parser.add_argument("--time", type=float, default=6.2)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    output = args.directory / "alternative_policy_probe.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    metadata = json.loads((args.directory / "competent_comparison.json").read_text())
    summary = metadata["summary"]
    config = CompetentExperimentConfig(**summary["config"])
    if args.time < config.event_time_seconds:
        raise ValueError("this diagnostic expects a post-event frozen-branch sample")
    if config.disturbance == "payload":
        raise ValueError("payload reconstruction is outside this bounded diagnostic")
    device = jax.devices(args.device)[0]
    bundle = load_learner_checkpoint(Path(summary["checkpoint"]), device=device)
    if bundle.sha256 != summary["checkpoint_npz_sha256"]:
        raise ValueError("checkpoint checksum differs from the saved experiment")
    if bundle.spec.latent_codes.shape[0] > 16:
        raise ValueError("this diagnostic is bounded to 17 augmented candidates")
    with np.load(args.directory / "competent_comparison.npz", allow_pickle=False) as trace:
        times = trace["time_seconds"]
        matching = np.flatnonzero(np.isclose(times, args.time, atol=1e-7, rtol=0))
        if matching.size != 1 or matching[0] == 0:
            raise ValueError("requested timestamp must match one noninitial control boundary")
        index = int(matching[0])
        state = jax.device_put(trace[f"{args.method}_full_state"][index], device)
        wind = jax.device_put(trace[f"{args.method}_estimated_wind"][index], device)
        previous = jax.device_put(
            jnp.asarray(trace[f"{args.method}_selected_policy"][index - 1] + 1, dtype=jnp.int32),
            device,
        )
        saved = {
            key: trace[f"{args.method}_{key}"][index]
            for key in (
                "applied_wrench",
                "selected_policy",
                "selected_smooth_value",
                "qp_valid",
                "qp_rejection_flags",
                "execution_mode",
            )
        }
    resources = VersionAResources(bundle.point_model, bundle.actuator)
    point_model = model_with_wind(bundle.point_model, wind)
    scenario = _scenario(config)
    scenario_record = json.loads((args.directory / "feasibility_reference.json").read_text())[
        "scenario"
    ]
    for field in (
        "obstacle_initial_centers",
        "obstacle_velocities",
        "obstacle_radii",
        "arena_lower",
        "arena_upper",
        "goal_position",
        "speed_max",
        "angular_rate_max",
        "tilt_max_radians",
    ):
        np.testing.assert_array_equal(np.asarray(getattr(scenario, field)), scenario_record[field])
    learner_config = replace(bundle.config, model_compensation=args.method == "compensated")
    obstacles = scenario_obstacle_window(scenario, round(args.time / config.dt))
    ordinary = _controller(scenario, resources, bundle.spec, learner_config, config, device)
    original = jax.block_until_ready(
        ordinary(state, bundle.state.params, point_model, obstacles, previous)
    )
    np.testing.assert_allclose(original.action, saved["applied_wrench"], atol=2e-7, rtol=2e-5)
    np.testing.assert_allclose(
        original.selected_smooth_value, saved["selected_smooth_value"], atol=2e-6
    )
    assert int(original.selected_index) == int(saved["selected_policy"]) + 1
    assert bool(original.qp_valid) == bool(saved["qp_valid"])
    np.testing.assert_array_equal(original.qp_rejection_flags, saved["qp_rejection_flags"])
    assert int(original.execution_mode) == int(saved["execution_mode"])

    safety = scenario_safety_limits(scenario)
    barrier_config = VersionABarrierConfig(
        obstacle_clearance=scenario.obstacle_clearance,
        arena_clearance=0.08,
        ego_radius=scenario.ego_radius,
        include_obstacle_hocbf=False,
    )
    filter_config = VersionAFilterConfig()
    continuous_config = ContinuousVersionAConfig(
        dt=config.dt,
        horizon=config.horizon,
        obstacle_clearance=scenario.obstacle_clearance,
        ego_radius=scenario.ego_radius,
        prefer_nominal_when_safe=False,
        control_interval_steps=config.control_interval_steps,
    )

    def nominal(candidate: Any, model: Any) -> PolicyRollouts:
        return rollout_waypoint_library(
            candidate,
            scenario.goal_position[None],
            scenario.goal_velocity[None],
            model,
            bundle.actuator,
            QuadPolicyConfig(acceleration_limit=config.nominal_acceleration_limit),
            dt=config.dt,
            horizon=config.horizon,
            position_gain=2.0,
            velocity_gain=2.8,
            model_compensation=True,
        )

    def fallbacks(candidate: Any, model: Any, params: Any) -> PolicyRollouts:
        rollout = rollout_skill_library(
            params, bundle.spec, candidate, model, bundle.actuator, learner_config
        )
        valid = jnp.all(rollout.policy_valid, axis=1) & jnp.all(
            jnp.isfinite(rollout.states), axis=(1, 2)
        )
        return PolicyRollouts(rollout.states, rollout.wrenches, valid)

    @jax.jit
    def force_eligible_incumbent(
        candidate_state: Any, params: Any, model: Any, prediction: Any, incumbent: Any
    ) -> Any:
        return continuous_version_a_step(
            candidate_state,
            nominal,
            lambda x, point: fallbacks(x, point, params),
            prediction,
            model,
            bundle.actuator,
            safety,
            barrier_config,
            filter_config,
            continuous_config,
            previous_policy_index=incumbent,
            selection_config=SelectionConfig(switch_score_margin=2.0),
        )

    @jax.jit
    def held(wrench: Any) -> Any:
        return _held_action_check(
            state,
            wrench,
            point_model,
            obstacles,
            safety,
            barrier_config,
            filter_config,
            continuous_config,
        )

    def diagnose(decision: Any) -> dict[str, Any]:
        filtered = decision.continuous_filter
        qp_hold = jax.block_until_ready(held(filtered.qp.action))
        fallback_hold = jax.block_until_ready(
            held(decision.candidates.wrenches[decision.selected_index, 0])
        )
        return _plain(
            {
                "selected_index": decision.selected_index,
                "selected_hard_value": decision.values.values[decision.selected_index],
                "selected_smooth_value": decision.selected_smooth_value,
                "admissible_fraction": filtered.policy_admissible_fractions[
                    decision.selected_index
                ],
                "initial_qp_accepted": filtered.qp_accepted,
                "qp_solver_feasible": filtered.qp.feasible,
                "qp_kkt_valid": filtered.qp_kkt_valid,
                "qp_fast_path_used": filtered.qp_fast_path_used,
                "qp_accepted_including_hold": decision.qp_valid,
                "qp_action": filtered.qp.action,
                "qp_initial_operational_residual": (
                    filtered.qp_postcheck.minimum_analytic_barrier_residual
                ),
                "qp_hold": qp_hold._asdict(),
                "fallback_hold": fallback_hold._asdict(),
                "execution_mode": EXECUTION_MODES[int(decision.execution_mode)],
                "degraded": decision.degraded,
                "rejection_reasons": [
                    reason
                    for reason, rejected in zip(
                        QP_REJECTION_REASONS, np.asarray(decision.qp_rejection_flags), strict=True
                    )
                    if rejected
                ],
            }
        )

    eligibility = np.asarray(original.continuous_filter.policy_eligible)
    fractions = np.asarray(original.continuous_filter.policy_admissible_fractions)
    assert np.all((fractions[eligibility] >= 0) & (fractions[eligibility] <= 1))
    alternatives = []
    for candidate in np.flatnonzero(eligibility):
        decision = jax.block_until_ready(
            force_eligible_incumbent(
                state,
                bundle.state.params,
                point_model,
                obstacles,
                jnp.asarray(candidate, dtype=jnp.int32),
            )
        )
        assert int(decision.selected_index) == int(candidate)
        np.testing.assert_array_equal(decision.continuous_filter.policy_eligible, eligibility)
        np.testing.assert_allclose(decision.nominal_action, original.nominal_action, atol=2e-7)
        np.testing.assert_allclose(decision.values.values, original.values.values, atol=2e-5)
        alternatives.append(diagnose(decision))
    accepted = [row["selected_index"] for row in alternatives if row["qp_accepted_including_hold"]]
    report = {
        "diagnostic": "same-state eligible alternative certificate probe",
        "method": args.method,
        "time_seconds": args.time,
        "state": state,
        "point_model": point_model._asdict(),
        "checkpoint_sha256": bundle.sha256,
        "original_saved_decision_reproduced": True,
        "original": diagnose(original),
        "original_eligible_mask": eligibility,
        "alternative_count": len(alternatives),
        "accepted_alternative_indices": accepted,
        "alternatives": alternatives,
        "forcing_method": (
            "Diagnostic switch-score margin 2 retains each eligible incumbent because admissible "
            "fractions lie in [0,1]. All eligibility thresholds, nominal actions, dynamics, QP "
            "constraints, and held-action checks remain unchanged. Production code is unmodified."
        ),
        "interpretation": (
            "An accepted alternative exists at this one state; this does not prove that switching "
            "to it would complete the episode or provide invariant safety."
            if accepted
            else "No eligible alternative passes the full implemented acceptance checks at this "
            "one state. This does not prove that every physically safe action is impossible."
        ),
    }
    with output.open("x") as stream:
        json.dump(_plain(report), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "accepted_alternatives": accepted}), flush=True)


if __name__ == "__main__":
    main()
