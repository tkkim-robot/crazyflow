"""Benchmark steady-state forward simulation throughput.

Paper-comparable measurements use 50 timed executions with 50 simulator steps fused into each
execution, after a separate warm-up/compile call. For example::

    pixi run -e benchmark python benchmark/main.py --device=gpu --worlds=262144 \
        --n_steps=50 --rollout_steps=50 --include_gym=False

The reported world-step and drone-update rates count every fused simulator step; compilation time
is excluded. Raw execution timings, their standard deviation, and software/hardware provenance are
retained in the output CSV. For Gymnasium rows, the legacy ``fps`` column remains environment
actions/s while the explicit world-step and drone-update columns include all simulator substeps.
"""

import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import fire
import gymnasium
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax.errors import JaxRuntimeError
from ml_collections import config_dict

import crazyflow  # Ensure Gymnasium environments are registered.
from crazyflow.sim import Sim


def _git_revision() -> str:
    """Return the checked-out revision and mark benchmark runs from a dirty tree."""
    repository = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{revision}-dirty" if status.strip() else revision


def analyze_timings(
    times: list[float],
    n_executions: int,
    n_worlds: int,
    freq: float,
    *,
    steps_per_execution: int = 1,
    n_drones: int = 1,
) -> None:
    """Analyze timing results and print performance metrics."""
    if not times:
        raise ValueError("The list of timing results is empty.")

    tmin, idx_tmin = np.min(times), np.argmin(times)
    tmax, idx_tmax = np.max(times), np.argmax(times)

    # Check for significant variance
    if tmax / tmin > 10:
        print("Warning: Fn time varies by more than 10x. Is JIT compiling during the benchmark?")
        print(f"Times: max {tmax:.2e} @ {idx_tmax}, min {tmin:.2e} @ {idx_tmin}")

    # Performance metrics
    n_world_steps = n_executions * steps_per_execution * n_worlds
    total_time = np.sum(times)
    avg_step_time = np.mean(times)
    step_time_std = np.std(times)
    world_steps_per_second = n_world_steps / total_time
    drone_updates_per_second = world_steps_per_second * n_drones
    real_time_factor = (n_executions * steps_per_execution / freq) * n_worlds / total_time

    print(
        f"Avg fn time: {avg_step_time:.2e}s, std: {step_time_std:.2e}"
        f"\nWorld steps/s: {world_steps_per_second:.3e}, "
        f"Drone updates/s: {drone_updates_per_second:.3e}, "
        f"Real time factor: {real_time_factor:.2e}\n"
    )


def profile_gym_env_step(
    sim_config: config_dict.ConfigDict, n_steps: int, device: str, print_summary: bool = True
) -> tuple[list[float], int, int]:
    """Profile the Crazyflow gym environment step performance."""
    times = []
    device = jax.devices(device)[0]

    envs = gymnasium.make_vec(
        "DroneReachPos-v0",
        max_episode_time=3,
        num_envs=sim_config.n_worlds,
        device=sim_config.device,
        freq=sim_config.freq,
        dynamics=sim_config.dynamics,
        drone=sim_config.drone,
    )

    # Attitude commands are [roll, pitch, yaw, collective thrust].
    action = np.zeros((sim_config.n_worlds, 4), dtype=np.float32)
    action[..., 3] = np.asarray(envs.unwrapped.sim.data.params.mass[:, 0, 0]) * 9.81
    # Step through env once to ensure JIT compilation
    envs.reset()
    envs.step(action)

    jax.block_until_ready(envs.unwrapped.sim.data)  # Ensure JIT compiled dynamics

    # Step through the environment
    for _ in range(n_steps):
        tstart = time.perf_counter()
        envs.step(action)
        jax.block_until_ready(envs.unwrapped.sim.data)
        times.append(time.perf_counter() - tstart)

    n_worlds = envs.unwrapped.sim.n_worlds
    n_substeps = envs.unwrapped.n_substeps
    sim_freq = envs.unwrapped.sim.freq
    envs.close()
    if print_summary:
        print("Gym env step performance:")
        analyze_timings(
            times, n_steps, n_worlds, sim_freq, steps_per_execution=n_substeps, n_drones=1
        )
    return times, n_substeps, sim_freq


def profile_step(
    sim_config: config_dict.ConfigDict,
    n_steps: int,
    device: str,
    print_summary: bool = True,
    rollout_steps: int = 1,
) -> list[float]:
    """Profile the Crazyflow simulator step performance."""
    sim = Sim(**sim_config)
    times = []
    device = jax.devices(device)[0]

    cmd = jnp.zeros((sim.n_worlds, sim.n_drones, 4), device=device)
    cmd = cmd.at[..., 3].set(sim.data.params.mass[..., 0] * 9.81)

    sim.reset()
    sim.attitude_control(cmd)
    sim.step(rollout_steps)
    jax.block_until_ready(sim.data)  # Ensure JIT compiled dynamics

    for _ in range(n_steps):
        tstart = time.perf_counter()
        sim.attitude_control(cmd)
        sim.step(rollout_steps)
        jax.block_until_ready(sim.data)
        times.append(time.perf_counter() - tstart)

    if print_summary:
        print("Sim step performance:")
        analyze_timings(
            times,
            n_steps,
            sim.n_worlds,
            sim.freq,
            steps_per_execution=rollout_steps,
            n_drones=sim.n_drones,
        )
    return times


def profile_reset(sim_config: config_dict.ConfigDict, n_steps: int, device: str):
    """Profile the Crazyflow simulator reset performance."""
    sim = Sim(**sim_config)
    times = []
    times_masked = []
    device = jax.devices(device)[0]

    # Ensure JIT compiled reset
    sim.reset()
    jax.block_until_ready(sim.data)

    # Test full reset
    for _ in range(n_steps):
        tstart = time.perf_counter()
        sim.reset()
        jax.block_until_ready(sim.data)
        times.append(time.perf_counter() - tstart)

    # Test masked reset (only reset first world)
    mask = jnp.zeros(sim.n_worlds, dtype=bool, device=device)
    mask = mask.at[0].set(True)
    sim.reset(mask)
    jax.block_until_ready(sim.data)

    for _ in range(n_steps):
        tstart = time.perf_counter()
        sim.reset(mask)
        jax.block_until_ready(sim.data)
        times_masked.append(time.perf_counter() - tstart)

    print("Sim reset performance:")
    analyze_timings(times, n_steps, sim.n_worlds, sim.freq)
    print("Sim masked reset performance:")
    analyze_timings(times_masked, n_steps, sim.n_worlds, sim.freq)


def main(
    device: str = "cpu",
    n_worlds_exp: int = 6,
    n_drones: int = 1,
    n_steps: int = 1000,
    worlds: str | int | None = None,
    drone: str = "cf2x_L250",
    rollout_steps: int = 1,
    include_gym: bool = True,
):
    """Profile simulator throughput for configurable world and drone counts.

    Args:
        device: JAX platform to benchmark.
        n_worlds_exp: Largest base-10 world-count exponent when ``worlds`` is omitted.
        n_drones: Drones per simulated world. Gymnasium measurements are only available for one.
        n_steps: Timed executions per configuration. The paper uses 50.
        worlds: Optional exact world count or comma-separated counts, for example ``"16,1024"``.
        drone: Drone parameters used by the simulator and Gymnasium benchmarks.
        rollout_steps: Simulator steps fused into each timed execution. The paper uses 50.
        include_gym: Also run the single-drone Gymnasium benchmark. Disable this for
            forward-simulator-only paper comparisons.
    """
    if n_drones <= 0:
        raise ValueError("n_drones must be positive")
    sim_config = config_dict.ConfigDict()
    sim_config.n_worlds = 1
    sim_config.n_drones = n_drones
    sim_config.dynamics = "first_principles"
    sim_config.control = "attitude"
    sim_config.drone = drone
    sim_config.attitude_freq = 500
    sim_config.device = device
    sim_config.freq = 500
    # Throughput measurements do not use geometry or contacts. Avoid allocating the quadratic
    # MuJoCo collision model, especially for the paper's large swarm configurations.
    sim_config.enable_mjx = False

    max_seconds_per_run = 60.0

    print("\nRunning benchmarks for increasing number of parallel environments...")

    # Create a CSV file to store results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = Path(__file__).parent / "data" / f"benchmark_results_{timestamp}.csv"
    csv_file.parent.mkdir(exist_ok=True)

    # Create CSV writer and write header
    with open(csv_file, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            [
                "test_type",
                "n_drones",
                "n_worlds",
                "n_executions",
                "steps_per_execution",
                "total_sim_steps",
                "total_time_s",
                "mean_execution_time_s",
                "std_execution_time_s",
                "execution_times_s",
                "world_steps_per_s",
                "fps",
                "drone_updates_per_s",
                "real_time_factor",
                "device",
                "device_kind",
                "jax_backend",
                "dynamics",
                "control",
                "drone",
                "sim_frequency_hz",
                "environment_frequency_hz",
                "mjx_enabled",
                "crazyflow_version",
                "git_commit",
                "jax_version",
                "jaxlib_version",
            ]
        )

    # Reopen the file in append mode for each result

    skip_sim, skip_gym = False, n_drones != 1 or not include_gym
    if skip_gym:
        reason = (
            "built-in environments support one drone per world"
            if n_drones != 1
            else "include_gym=False"
        )
        print(f"Gymnasium benchmark skipped: {reason}.")
    if worlds is None:
        world_counts = [10**i for i in range(n_worlds_exp + 1)]
    elif isinstance(worlds, int):
        world_counts = [worlds]
    else:
        # The early-stop logic below assumes an increasing resource sweep. Sorting also avoids a
        # failed large explicit count suppressing a viable smaller count supplied after it.
        world_counts = sorted({int(value.strip()) for value in worlds.split(",")})
    if not world_counts or any(value <= 0 for value in world_counts):
        raise ValueError("world counts must be positive integers")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    selected_device = jax.devices(device)[0]
    provenance = [
        selected_device.device_kind,
        selected_device.platform,
        sim_config.dynamics,
        sim_config.control,
        sim_config.drone,
    ]
    software = [crazyflow.__version__, _git_revision(), jax.__version__, jaxlib.__version__]
    # Test with increasing number of parallel environments (worlds)
    for n_worlds in world_counts:
        sim_config.n_worlds = n_worlds
        print("-" * 80)
        if not skip_sim:
            # Test with a single step first to see if we should continue
            sim_config.freq = 500  # Test sim at 500 hz
            try:
                single_step_time = profile_step(
                    sim_config, 2, device, print_summary=False, rollout_steps=rollout_steps
                )[1]
            except JaxRuntimeError:
                print(f"  Skipping benchmark for {n_worlds} and higher - resource exhausted")
                skip_sim = True
                continue

            # If single step takes too long, skip this and remaining tests
            if single_step_time > max_seconds_per_run / n_steps:  # threshold for the tests
                print(
                    f"  Skipping benchmark for {n_worlds} and higher - projected time "
                    f"{single_step_time * n_steps:.2f}s (> 1m)"
                )
                skip_sim = True

        if not skip_sim:
            # Configure simulator
            print(f"Running simulator benchmark ({n_worlds} worlds)...")
            # Run simulator benchmark using existing function
            try:
                times_sim = profile_step(sim_config, n_steps, device, rollout_steps=rollout_steps)
            except JaxRuntimeError:
                print(f"  Skipping benchmark for {n_worlds} and higher - resource exhausted")
                skip_sim = True
                continue

            # Calculate metrics for CSV
            total_time = sum(times_sim)
            mean_execution_time = np.mean(times_sim)
            std_execution_time = np.std(times_sim)
            total_sim_steps = n_steps * rollout_steps
            n_frames = total_sim_steps * n_worlds
            fps = n_frames / total_time
            real_time_factor = (total_sim_steps / sim_config.freq) * n_worlds / total_time

            # Save simulator results
            # Reopen CSV writer in append mode
            with open(csv_file, "a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(
                    [
                        "simulator",
                        sim_config.n_drones,
                        n_worlds,
                        n_steps,
                        rollout_steps,
                        total_sim_steps,
                        total_time,
                        mean_execution_time,
                        std_execution_time,
                        json.dumps(times_sim),
                        fps,
                        fps,
                        fps * sim_config.n_drones,
                        real_time_factor,
                        sim_config.device,
                        *provenance,
                        sim_config.freq,
                        "",
                        sim_config.enable_mjx,
                        *software,
                    ]
                )
                f.flush()

        if not skip_gym:
            print(f"Running gym environment benchmark ({n_worlds} worlds)...")
            # Run gym environment benchmark using existing function
            sim_config.freq = 50  # Test gym at 50 hz
            try:
                step_times, gym_substeps, gym_sim_freq = profile_gym_env_step(
                    sim_config, 2, device, print_summary=False
                )
                single_step_time = step_times[1]
                # If single step takes too long, skip this test only
                if single_step_time > max_seconds_per_run / n_steps:  # threshold for the tests
                    print(
                        f"  Skipping benchmark for {n_worlds} - projected time "
                        f"{single_step_time * n_steps:.2f}s (> 1m)"
                    )
                    skip_gym = True
            except JaxRuntimeError:
                print(f"  Skipping benchmark for {n_worlds} - resource exhausted")
                skip_gym = True

        if not skip_gym:
            try:
                times_gym, gym_substeps, gym_sim_freq = profile_gym_env_step(
                    sim_config, n_steps, device
                )
            except JaxRuntimeError:
                print(f"  Skipping benchmark for {n_worlds} - resource exhausted")
                skip_gym = True
                continue

            # Calculate metrics for CSV
            total_time = sum(times_gym)
            mean_execution_time = np.mean(times_gym)
            std_execution_time = np.std(times_gym)
            env_frames = n_steps * n_worlds
            fps = env_frames / total_time
            total_sim_steps = n_steps * gym_substeps
            world_steps_per_second = total_sim_steps * n_worlds / total_time
            real_time_factor = total_sim_steps / gym_sim_freq * sim_config.n_worlds / total_time

            # Save gym environment results
            with open(csv_file, "a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(
                    [
                        "gym_env",
                        sim_config.n_drones,
                        sim_config.n_worlds,
                        n_steps,
                        gym_substeps,
                        total_sim_steps,
                        total_time,
                        mean_execution_time,
                        std_execution_time,
                        json.dumps(times_gym),
                        world_steps_per_second,
                        fps,
                        world_steps_per_second,
                        real_time_factor,
                        sim_config.device,
                        *provenance,
                        gym_sim_freq,
                        sim_config.freq,
                        True,
                        *software,
                    ]
                )
                f.flush()

    print(f"\nBenchmark results saved to {csv_file}")


if __name__ == "__main__":
    fire.Fire(main)
