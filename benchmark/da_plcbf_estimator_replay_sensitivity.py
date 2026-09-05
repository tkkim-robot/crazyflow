"""CPU-only estimator rate/noise replay of one fixed recorded closed-loop trajectory.

Commands, positions, attitude and rates stay exact. Shared seeded iid velocity-observation
noise is added once per recorded sample, so adjacent finite differences reuse the same noise.
This never re-executes a controller or learner and cannot establish a closed-loop benefit.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.da_plcbf_reference_ablation import _write
from crazyflow.safety.da_plcbf.point_wind_estimator import (
    PointWindEstimatorConfig,
    initialize_point_wind_estimator,
    update_point_wind_estimator,
)
from crazyflow.safety.da_plcbf.state_conditioned_learning import load_reference_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed-loop-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = jax.devices("cpu")[0]
    contract = load_reference_contract(args.closed_loop_dir / "nominal_reference", device=device)
    with np.load(args.closed_loop_dir / "shared_prefix.npz", allow_pickle=False) as data:
        prefix_states, prefix_actions = data["states"], data["actions"]
    with np.load(args.closed_loop_dir / "estimated_frozen/trace.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(prefix_states[-1], data["dense_states"][0])
        states = np.concatenate((prefix_states, data["dense_states"][1:]))
        control_actions = np.concatenate((prefix_actions, data["decision__action"]))
        recorded_wind = data["estimated_wind"]
    actions = np.repeat(control_actions, 2, axis=0)[: len(states) - 1]
    if (states.shape, actions.shape) != ((401, 13), (400, 4)):
        raise ValueError("this replay requires the complete 0–8 s recorded frozen branch")
    dt, event = 0.02, 4.0
    times = np.arange(len(states)) * dt
    truth = np.where(times[:, None] >= event, np.asarray([4.0, 1.6, 0.0]), 0.0)
    noise_seed = 20260905
    unit_noise = np.random.default_rng(noise_seed).normal(size=(len(states), 3)).astype(np.float32)
    arrays = {
        "time_seconds": times,
        "exact_states": states,
        "actions": actions,
        "true_wind": truth,
        "unit_velocity_observation_noise": unit_noise,
    }
    rows = []
    for rate in (1.2, 2.4, 6.0):
        config = PointWindEstimatorConfig(response_rate=rate)

        @jax.jit
        def replay(observations: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            def step(
                estimator: Any, values: tuple[jax.Array, jax.Array, jax.Array]
            ) -> tuple[Any, tuple[jax.Array, jax.Array, jax.Array]]:
                before, after, action = values
                result = update_point_wind_estimator(
                    estimator, before, after, action, contract.model, dt=dt, config=config
                )
                return result.state, (
                    result.state.wind_velocity,
                    result.instantaneous_wind,
                    result.measurement_valid,
                )

            _, values = jax.lax.scan(
                step,
                initialize_point_wind_estimator(),
                (observations[:-1], observations[1:], jnp.asarray(actions)),
            )
            return values

        for noise_std in (0.0, 0.01):
            observations = states.copy()
            observations[:, 7:10] += noise_std * unit_noise
            estimates, inferred, valid = jax.block_until_ready(replay(jnp.asarray(observations)))
            estimates = np.concatenate((np.zeros((1, 3)), np.asarray(estimates)))
            error = estimates - truth
            norms = np.linalg.norm(error, axis=1)
            post_indices = np.flatnonzero(times >= event)
            below = norms[post_indices] < 0.1
            starts = np.flatnonzero(below[:-2] & below[1:-1] & below[2:])
            first = int(post_indices[starts[0]]) if len(starts) else None
            late = times >= 7.0
            key = f"rate_{rate}__velocity_std_{noise_std}"
            arrays[f"{key}__observations"] = observations
            arrays[f"{key}__estimated_wind"] = estimates
            arrays[f"{key}__error"] = error
            arrays[f"{key}__instantaneous_wind_clipped"] = np.asarray(inferred)
            arrays[f"{key}__valid"] = np.asarray(valid)
            baseline_error = None
            if rate == 2.4 and noise_std == 0.0:
                baseline_error = float(np.max(np.abs(estimates[200::2][:100] - recorded_wind)))
                np.testing.assert_allclose(
                    estimates[200::2][:100], recorded_wind, atol=1e-5, rtol=1e-5
                )
            row = {
                "key": key,
                "config": asdict(config),
                "velocity_noise_std_mps": noise_std,
                "noise_seed": noise_seed,
                "finite_measurements": int(np.sum(valid)),
                "first_below_0_1_sustained_three_start_seconds": None
                if first is None
                else float(times[first]),
                "first_below_0_1_confirmation_seconds": None
                if first is None
                else float(times[first + 2]),
                "settling_delay_seconds": None if first is None else float(times[first] - event),
                "late_vector_rmse_mps": float(np.sqrt(np.mean(np.sum(error[late] ** 2, axis=1)))),
                "late_component_rmse_mps": np.sqrt(np.mean(error[late] ** 2, axis=0)).tolist(),
                "post_event_peak_error_mps": float(np.max(norms[post_indices])),
                "post_event_clipped_measurement_fraction": float(
                    np.mean(
                        np.any(
                            np.abs(np.asarray(inferred)[200:]) >= config.component_limit - 1e-6,
                            axis=1,
                        )
                    )
                ),
                "recorded_rate2_4_noiseless_max_abs_replay_error": baseline_error,
            }
            rows.append(row)
            print(row, flush=True)
    np.savez_compressed(args.output_dir / "estimator_replays.npz", **arrays)
    _write(
        args.output_dir / "summary.json",
        {
            "scope": (
                "estimator-only offline replay; states/actions fixed; no controller or BPTT rerun"
            ),
            "source_directory": str(args.closed_loop_dir),
            "source_trace_sha256": hashlib.sha256(
                (args.closed_loop_dir / "estimated_frozen/trace.npz").read_bytes()
            ).hexdigest(),
            "observation_model": (
                "iid Gaussian velocity error per sample, shared across response rates; "
                "all other state coordinates and actions exact; "
                "identical noisy sample reused in adjacent transitions"
            ),
            "late_window_seconds": [7.0, 8.0],
            "threshold_mps": 0.1,
            "consecutive_samples": 3,
            "results": rows,
        },
    )


if __name__ == "__main__":
    main()
