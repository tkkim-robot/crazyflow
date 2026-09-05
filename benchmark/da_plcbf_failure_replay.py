"""Reconstruct published learner states from an archived opportunity schedule.

Run with --source-tree pointing to a git archive of the reviewed implementation to reproduce
legacy results. All updates use the recorded pre-action state and point model. The controller
is independently re-evaluated at every saved state; saved commands are never fed back as results.
This is deterministic mechanism evidence, not a measured real-time deployment experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-tree", type=Path)
    parser.add_argument("--end", type=float, default=5.2)
    parser.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    args = parser.parse_args()
    if args.source_tree:
        sys.path.insert(0, str(args.source_tree.resolve()))

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
    )
    from crazyflow.safety.da_plcbf.learner_checkpoint import (
        load_learner_checkpoint,
        save_learner_checkpoint,
    )
    from crazyflow.safety.da_plcbf.online_constant_wind import VersionAResources
    from crazyflow.safety.da_plcbf.persistent_skill_learner import build_persistent_skill_learner
    from crazyflow.safety.da_plcbf.point_wind_estimator import (
        PointWindEstimatorConfig,
        initialize_point_wind_estimator,
        model_with_point_wind,
        update_point_wind_estimator,
    )
    from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step

    args.output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.directory / "competent_comparison.json").read_text())["summary"]
    config = CompetentExperimentConfig(**meta["config"])
    if config.disturbance != "wind":
        raise ValueError("this archived failure fixture expects the wind experiment")
    device = jax.devices(args.device)[0]
    bundle = load_learner_checkpoint(Path(meta["checkpoint"]), device=device)
    if bundle.sha256 != meta["checkpoint_npz_sha256"]:
        raise ValueError("archived initial checkpoint checksum differs")
    learning_config = replace(bundle.config, model_compensation=True)
    learner = build_persistent_skill_learner(
        bundle.spec, bundle.actuator, learning_config, device=device
    )
    resources = VersionAResources(bundle.point_model, bundle.actuator)
    scenario = _scenario(config)
    controller = _controller(scenario, resources, bundle.spec, learning_config, config, device)
    actual_model = model_with_wind(bundle.point_model, scenario.wind_after)
    with np.load(args.directory / "competent_comparison.npz", allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    indices = np.flatnonzero(
        (trace["time_seconds"] >= config.event_time_seconds - 1e-6)
        & (trace["time_seconds"] <= args.end + 1e-6)
    )
    initial_version = int(bundle.state.library_version)
    versions = {initial_version: bundle.state}
    rows, arrays = [], {}
    for index in indices:
        when = float(trace["time_seconds"][index])
        version = int(trace["adaptive_library_version"][index])
        if version not in versions:
            raise AssertionError(f"snapshot {version} has not completed before publication")
        published = versions[version]
        state = jax.device_put(trace["adaptive_full_state"][index], device)
        model = model_with_wind(
            bundle.point_model, jax.device_put(trace["adaptive_estimated_wind"][index], device)
        )
        previous = jax.device_put(
            jnp.asarray(trace["adaptive_selected_policy"][index - 1] + 1, dtype=jnp.int32), device
        )
        obstacles = scenario_obstacle_window(scenario, round(when / config.dt))
        decision = jax.block_until_ready(
            controller(state, published.params, model, obstacles, previous)
        )
        error = float(
            np.max(np.abs(np.asarray(decision.action) - trace["adaptive_applied_wrench"][index]))
        )
        np.testing.assert_allclose(
            decision.action, trace["adaptive_applied_wrench"][index], atol=3e-7, rtol=3e-5
        )
        np.testing.assert_allclose(
            decision.smooth_values[decision.selected_index],
            trace["adaptive_selected_smooth_value"][index],
            atol=3e-5,
            rtol=3e-5,
        )
        assert int(decision.execution_mode) == int(trace["adaptive_execution_mode"][index])
        np.testing.assert_array_equal(
            decision.qp_rejection_flags, trace["adaptive_qp_rejection_flags"][index]
        )
        stem = args.output / f"snapshot-{index:04d}"
        save_learner_checkpoint(
            published,
            bundle.spec,
            learning_config,
            bundle.actuator,
            state,
            stem,
            metadata={
                "simulation_time": when,
                "source_index": int(index),
                "previous_policy_index": int(previous),
                "source_checkpoint_sha256": bundle.sha256,
            },
        )
        launched = bool(trace["adaptive_learner_seconds"][index] > 0)
        row = {
            "index": int(index),
            "time": when,
            "version": version,
            "update_opportunity": launched,
            "action_max_abs_replay_error": error,
            "fallback_max_hard": float(np.max(decision.values.values[1:])),
            "augmented_max_hard": float(np.max(decision.values.values)),
            "eligible_count": int(decision.eligible_candidate_count),
            "mode": int(decision.execution_mode),
            "wind_error": np.asarray(actual_model.wind_velocity - model.wind_velocity).tolist(),
            "snapshot": str(stem.relative_to(args.output)),
        }
        for key, value in {
            "state": state,
            "estimated_wind": model.wind_velocity,
            "actual_wind": actual_model.wind_velocity,
            "candidate_states": decision.candidates.states,
            "hard": decision.values.values,
            "smooth": decision.smooth_values,
            "action": decision.action,
            "rejection_flags": decision.qp_rejection_flags,
        }.items():
            arrays[f"boundary_{index}_{key}"] = np.asarray(value)
        if launched:
            changed, metrics = jax.block_until_ready(learner.step(published, state, model))
            versions[int(changed.library_version)] = changed
            row.update(
                gradient_norm=float(metrics.gradient_norm),
                parameter_update_norm=float(metrics.parameter_update_norm),
                finite=bool(metrics.finite_update_applied),
                completed_version=int(changed.library_version),
            )
        rows.append(row)
        print(json.dumps(row), flush=True)

    # Closed-loop branches begin before the recorded first loss at 4.16 s. The opportunity mask
    # is exogenous and identical; the estimated branches infer wind from their own transitions.
    start_time = 4.12
    start_index = int(np.flatnonzero(np.isclose(trace["time_seconds"], start_time))[0])
    start_version = int(trace["adaptive_library_version"][start_index])
    plant = jax.jit(lambda x, u: direct_wrench_symplectic_step(x, u, actual_model, config.dt))
    estimator_config = PointWindEstimatorConfig(response_rate=2.4)
    estimate = jax.jit(
        lambda e, x, y, u: (
            update_point_wind_estimator(
                e, x, y, u, bundle.point_model, dt=config.dt, config=estimator_config
            ).state
        )
    )
    branches = {}
    for information in ("oracle", "estimated"):
        for learning in (False, True):
            name = f"{information}_{'learning' if learning else 'frozen'}"
            persistent = versions[start_version]
            state = jax.device_put(trace["adaptive_full_state"][start_index], device)
            previous = jax.device_put(
                jnp.asarray(
                    trace["adaptive_selected_policy"][start_index - 1] + 1, dtype=jnp.int32
                ),
                device,
            )
            estimator = initialize_point_wind_estimator()
            # Restore all estimator state by replaying the archived prefix of observations.
            if information == "estimated":
                dense = np.load(args.directory / "dense_plant_states.npz", allow_pickle=False)
                dense_states = dense["adaptive"]
                for k in range(
                    round(config.event_time_seconds / config.dt), round(start_time / config.dt)
                ):
                    control_index = k // config.control_interval_steps
                    estimator = estimate(
                        estimator,
                        jnp.asarray(dense_states[k]),
                        jnp.asarray(dense_states[k + 1]),
                        jnp.asarray(trace["adaptive_applied_wrench"][control_index]),
                    )
                if config.model_mode == "estimated":
                    np.testing.assert_allclose(
                        estimator.wind_velocity,
                        trace["adaptive_estimated_wind"][start_index],
                        atol=2e-5,
                    )
                dense.close()
            states, actions, values, modes, winds, versions_used = (
                [np.asarray(state)],
                [],
                [],
                [],
                [],
                [],
            )
            for index in indices[indices >= start_index]:
                when = float(trace["time_seconds"][index])
                model = (
                    actual_model
                    if information == "oracle"
                    else model_with_point_wind(bundle.point_model, estimator)
                )
                decision = jax.block_until_ready(
                    controller(
                        state,
                        persistent.params,
                        model,
                        scenario_obstacle_window(scenario, round(when / config.dt)),
                        previous,
                    )
                )
                actions.append(np.asarray(decision.action))
                values.append(np.asarray(decision.values.values))
                modes.append(int(decision.execution_mode))
                winds.append(np.asarray(model.wind_velocity))
                versions_used.append(int(persistent.library_version))
                training_state = state
                for _ in range(config.control_interval_steps):
                    following = plant(state, decision.action)
                    if information == "estimated":
                        estimator = estimate(estimator, state, following, decision.action)
                    state = following
                    states.append(np.asarray(state))
                previous = decision.selected_index
                if learning and trace["adaptive_learner_seconds"][index] > 0:
                    persistent, _ = jax.block_until_ready(
                        learner.step(persistent, training_state, model)
                    )
            for key, value in {
                "states": states,
                "actions": actions,
                "hard": values,
                "modes": modes,
                "winds": winds,
                "versions": versions_used,
            }.items():
                arrays[f"branch_{name}_{key}"] = np.asarray(value)
            branches[name] = {
                "start_time": start_time,
                "start_version": start_version,
                "minimum_augmented_hard": float(np.min(np.max(values, axis=1))),
                "negative_hard_controls": int(np.sum(np.max(values, axis=1) < 0)),
                "emergency_controls": modes.count(2),
                "final_position": states[-1][:3].tolist(),
            }
            print(json.dumps({name: branches[name]}), flush=True)
    np.savez_compressed(args.output / "replay.npz", **arrays)
    report = {
        "source_directory": str(args.directory),
        "reviewed_commit": "00e89a742a1271b93655bf1bb4581a667dc13a14",
        "source_tree": str(args.source_tree),
        "source_trace_sha256": hashlib.sha256(
            (args.directory / "competent_comparison.npz").read_bytes()
        ).hexdigest(),
        "initial_checkpoint_sha256": bundle.sha256,
        "rows": rows,
        "pre_failure_closed_loop_branches": branches,
        "scope": (
            "Archived 20 ms prediction / 40 ms execution; recorded update opportunities, "
            "no wall-clock deployment claim"
        ),
    }
    (args.output / "replay.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
