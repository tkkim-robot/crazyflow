"""Backpropagation-through-time benchmark for Crazyflow trajectory tracking.

This module has two deliberately separate protocols:

``public``
    A current-Crazyflow adaptation of the closest public author artifact,
    ``rl_diffsim@068239d``. It uses a deterministic two-layer, six-unit tanh actor and the
    artifact's 25,600 actual training steps. That artifact differs from the paper: it omits the
    attitude penalty and log-standard-deviation policy, adds a termination penalty, and measures
    action cost around zero in normalized action coordinates rather than physical hover.

``paper``
    A paper-informed reconstruction using the published 100,000 nominal timesteps, attitude
    penalty, and initial log standard deviation. The exact supplementary trainer/configuration
    commit behind the paper result is unavailable, so this protocol borrows the closest public
    artifact's architecture, action normalization, termination rule, and AdamW settings. Unlike
    that artifact, its action penalty is centered on physical hover after conversion to normalized
    coordinates. Stochastic actions are penalized before clipping, matching the public wrapper
    order. This protocol must not be treated as an exact reproduction.

Both protocols differentiate the complete 40-action rollout through ten 500 Hz simulator steps
per 50 Hz action. The optimizer loop is also fused in one JAX ``scan``. MuJoCo/MJX geometry is
disabled because BPTT only needs the differentiable flight dynamics. Compilation and one complete
warm execution are excluded from each reported timing; JSON output includes the raw repetitions
and software/hardware provenance.

Sources:
    Paper: https://arxiv.org/html/2606.01478v1#Sx4.SSx9
    Nearest public trainer: https://github.com/yufei4hua/rl_diffsim/blob/068239d165d8a12b6e32ba0dc25f627cdcf4ecf2/rl_diffsim/bptt/train_bptt_figure8.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, NamedTuple

import jax
import jax.numpy as jnp
import jaxlib
import optax
from flax import linen as nn
from jax import Array

from crazyflow import __version__ as crazyflow_version
from crazyflow.control import Control
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.dynamics import Dynamics
from crazyflow.envs.drone_env import action_space
from crazyflow.sim import Sim

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData

Protocol = Literal["public", "paper"]
PyTree = Any


@dataclass(frozen=True)
class BPTTConfig:
    """Static settings for one fully fused BPTT training run."""

    protocol: Protocol = "public"
    device: str = "cpu"
    seed: int = 42
    n_envs: int = 16
    rollout_steps: int = 40
    total_timesteps: int = 26_000
    sim_freq: int = 500
    env_freq: int = 50
    trajectory_time: float = 10.0
    episode_time: float = 20.0
    n_samples: int = 11
    samples_dt: float = 0.06
    hidden_size: int = 6
    learning_rate: float = 4.6e-2
    gamma: float = 1.0
    angle_weight: float = 0.0
    action_weights: tuple[float, float, float, float] = (0.12, 0.12, 0.0, 0.02)
    delta_action_weights: tuple[float, float, float, float] = (1.4, 1.4, 0.0, 0.8)
    stochastic: bool = False
    initial_log_std: float = -1.0
    adam_epsilon: float = 1e-5
    weight_decay: float = 1e-4
    drone: str = "cf21B_500"

    @property
    def batch_size(self) -> int:
        """Number of simulator actions contributing to one update."""
        return self.n_envs * self.rollout_steps

    @property
    def n_updates(self) -> int:
        """Whole BPTT updates that fit in the requested timestep budget."""
        return self.total_timesteps // self.batch_size

    @property
    def actual_timesteps(self) -> int:
        """Executed timesteps after matching the public trainer's floor division."""
        return self.n_updates * self.batch_size

    @property
    def sim_steps_per_action(self) -> int:
        """500 Hz dynamics steps taken for each 50 Hz policy action."""
        return self.sim_freq // self.env_freq

    @property
    def episode_steps(self) -> int:
        """Policy actions in one episode."""
        return round(self.episode_time * self.env_freq)

    def validate(self) -> None:
        """Reject configurations that would silently change the benchmark semantics."""
        if self.protocol not in ("public", "paper"):
            raise ValueError(f"unknown BPTT protocol: {self.protocol}")
        if (
            min(
                self.n_envs,
                self.rollout_steps,
                self.total_timesteps,
                self.hidden_size,
                self.n_samples,
                self.sim_freq,
                self.env_freq,
            )
            <= 0
        ):
            raise ValueError(
                "environment, rollout, timestep, sample, and frequency values must be positive"
            )
        if min(self.trajectory_time, self.episode_time, self.samples_dt) <= 0:
            raise ValueError("trajectory, episode, and sample times must be positive")
        if self.n_updates <= 0:
            raise ValueError("total_timesteps must contain at least one complete BPTT batch")
        if self.sim_freq % self.env_freq:
            raise ValueError("sim_freq must be an integer multiple of env_freq")
        if self.episode_steps <= self.rollout_steps:
            raise ValueError("episode_time must exceed one BPTT rollout")
        if round(self.samples_dt * self.env_freq) <= 0:
            raise ValueError("samples_dt must span at least one policy step")


def public_config(**overrides: Any) -> BPTTConfig:
    """Return settings from the closest public author artifact."""
    return replace(BPTTConfig(), **overrides)


def paper_config(**overrides: Any) -> BPTTConfig:
    """Return the explicitly labeled paper-informed reconstruction settings."""
    base = BPTTConfig(
        protocol="paper",
        total_timesteps=100_000,
        angle_weight=0.1,
        stochastic=True,
        initial_log_std=-1.0,
    )
    return replace(base, **overrides)


class TrainCarry(NamedTuple):
    """State threaded through fused optimizer iterations."""

    params: PyTree
    optimizer: optax.OptState
    data: SimData
    episode_steps: Array
    reset_next: Array
    last_actions: Array
    key: Array


class TrainMetrics(NamedTuple):
    """Per-update quantities returned by the fused training loop."""

    loss: Array
    gradient_norm: Array
    position_rmse: Array


@dataclass(frozen=True)
class BPTTResult:
    """Host-side benchmark result."""

    protocol: str
    device: str
    requested_timesteps: int
    actual_timesteps: int
    compile_seconds: float
    execution_seconds: tuple[float, ...]
    mean_execution_seconds: float
    best_execution_seconds: float
    steps_per_second: float
    first_loss: float
    final_loss: float
    first_gradient_norm: float
    final_gradient_norm: float
    parameter_delta_norm: float
    evaluation_steps: int
    evaluation_rmse_mm: float | None
    device_kind: str
    jax_version: str
    jaxlib_version: str
    crazyflow_version: str
    git_commit: str


@dataclass(frozen=True)
class BPTTTrainer:
    """Constructed simulator, initial state, and compiled-function inputs."""

    config: BPTTConfig
    initial_carry: TrainCarry
    train_fn: Callable[[TrainCarry], tuple[TrainCarry, TrainMetrics]]
    evaluate_fn: Callable[[PyTree, int], Array]


def _reference(phase_offsets: Array, steps: Array, config: BPTTConfig) -> tuple[Array, Array]:
    """Evaluate the public artifact's x-z Lissajous position and velocity."""
    extra_dims = (1,) * (steps.ndim - 1)
    offsets = phase_offsets.reshape((phase_offsets.shape[0], *extra_dims))
    omega = 2.0 * jnp.pi / config.trajectory_time
    if config.protocol == "public":
        # rl_diffsim@068239d materializes 1000 endpoint-inclusive samples over two loops. Its
        # sampling interval is therefore 20 / 999 seconds rather than exactly 1 / 50 seconds.
        sample_index = steps % config.episode_steps
        n_loops = config.episode_time / config.trajectory_time
        phase = offsets + 2.0 * jnp.pi * n_loops * sample_index / (config.episode_steps - 1)
    else:
        phase = offsets + omega * steps / config.env_freq
    x = jnp.sin(phase)
    y = jnp.zeros_like(phase)
    z = 0.5 * jnp.sin(2.0 * phase) + 1.25
    dx = omega * jnp.cos(phase)
    dy = jnp.zeros_like(phase)
    dz = omega * jnp.cos(2.0 * phase)
    return jnp.stack((x, y, z), axis=-1), jnp.stack((dx, dy, dz), axis=-1)


def _observations(
    data: SimData,
    episode_steps: Array,
    last_actions: Array,
    phase_offsets: Array,
    sample_offsets: Array,
    config: BPTTConfig,
) -> Array:
    """Build the public trainer's 50-element flattened observation."""
    future_steps = episode_steps[:, None] + sample_offsets[None, :]
    future_pos, _ = _reference(phase_offsets, future_steps, config)
    local_samples = future_pos - data.states.pos[:, 0, None, :]
    return jnp.concatenate(
        (
            data.states.pos[:, 0],
            data.states.quat[:, 0],
            data.states.vel[:, 0],
            data.states.ang_vel[:, 0],
            local_samples.reshape(config.n_envs, -1),
            last_actions,
        ),
        axis=-1,
    )


class ActorNet(nn.Module):
    """The exact two-layer Linen actor used by the pinned public artifact."""

    hidden_size: int = 64
    act_dim: int = 4
    num_layers: int = 2

    @nn.compact
    def __call__(self, observations: Array) -> Array:
        """Map flattened observations to bounded normalized action means."""
        value = observations
        for _ in range(self.num_layers):
            value = nn.Dense(
                self.hidden_size,
                kernel_init=nn.initializers.orthogonal(),
                bias_init=nn.initializers.zeros,
            )(value)
            value = nn.tanh(value)
        value = nn.Dense(
            self.act_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(value)
        return nn.tanh(value)


def _init_actor(config: BPTTConfig, obs_dim: int, key: Array, device: jax.Device) -> PyTree:
    """Initialize with Linen so seed-to-parameter mapping matches the public artifact."""
    actor = ActorNet(hidden_size=config.hidden_size)
    observations = jnp.zeros((1, obs_dim), dtype=jnp.float32, device=device)
    params = actor.init(key, observations)
    if config.stochastic:
        params["log_std"] = jnp.full(
            (1, 4), config.initial_log_std, dtype=jnp.float32, device=device
        )
    return params


def _actor_mean(params: PyTree, observations: Array) -> Array:
    """Apply the Linen actor while leaving an optional log-standard-deviation leaf separate."""
    network_params = {"params": params["params"]}
    hidden_size = network_params["params"]["Dense_0"]["bias"].shape[0]
    return ActorNet(hidden_size=hidden_size).apply(network_params, observations)


def _tree_l2_norm(tree: PyTree) -> Array:
    """Compute a stable global Euclidean norm for a pytree."""
    leaves = jax.tree.leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def _git_revision() -> str:
    """Return the checked-out revision and flag uncommitted benchmark code."""
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


def _make_optimizer(config: BPTTConfig) -> optax.GradientTransformation:
    """Construct the same AdamW transform and epsilon used by the public artifact."""
    return optax.adamw(
        learning_rate=config.learning_rate,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )


def _optimizer_update(
    optimizer: optax.GradientTransformation,
    params: PyTree,
    gradients: PyTree,
    state: optax.OptState,
) -> tuple[PyTree, optax.OptState]:
    """Apply one Optax update in the same order as Flax ``TrainState``."""
    updates, next_state = optimizer.update(gradients, state, params)
    return optax.apply_updates(params, updates), next_state


def build_trainer(config: BPTTConfig) -> BPTTTrainer:
    """Construct the current-API differentiable trainer without compiling it."""
    config.validate()
    sim = Sim(
        n_worlds=config.n_envs,
        n_drones=1,
        drone=config.drone,
        dynamics=Dynamics.first_principles,
        control=Control.attitude,
        freq=config.sim_freq,
        attitude_freq=config.sim_freq,
        device=config.device,
        rng_key=config.seed,
        enable_mjx=False,
    )
    device = sim.device
    phase_offsets = jax.device_put(
        jnp.linspace(0.0, 2.0 * jnp.pi, config.n_envs, endpoint=False), device
    )
    initial_steps = jnp.zeros((config.n_envs,), dtype=jnp.int32, device=device)
    initial_pos, initial_vel = _reference(phase_offsets, initial_steps, config)

    if config.protocol == "public":
        initial_rotor_vel = jnp.full_like(sim.data.states.rotor_vel, 15_900.0)
    else:
        per_motor_force = sim.data.params.mass * 9.81 / 4.0
        initial_rotor_vel = motor_force2rotor_vel(
            jnp.broadcast_to(per_motor_force, sim.data.states.rotor_vel.shape),
            sim.data.params.rpm2thrust,
        )
    data = sim.data.replace(
        states=sim.data.states.replace(
            pos=initial_pos[:, None, :], vel=initial_vel[:, None, :], rotor_vel=initial_rotor_vel
        )
    )
    default_data = data.replace()

    sample_stride = round(config.samples_dt * config.env_freq)
    sample_offsets = jax.device_put(
        jnp.arange(config.n_samples, dtype=jnp.int32) * sample_stride, device
    )
    last_actions = jnp.zeros((config.n_envs, 4), dtype=jnp.float32, device=device)
    reset_next = jnp.zeros((config.n_envs,), dtype=jnp.bool_, device=device)
    observations = _observations(
        data, initial_steps, last_actions, phase_offsets, sample_offsets, config
    )

    key = jax.device_put(jax.random.key(config.seed), device)
    init_key, train_key = jax.random.split(key)
    params = _init_actor(config, observations.shape[-1], init_key, device)
    optimizer_transform = _make_optimizer(config)
    optimizer = optimizer_transform.init(params)

    space = action_space(Control.attitude, config.drone)
    action_low = jax.device_put(jnp.asarray(space.low), device)
    action_high = jax.device_put(jnp.asarray(space.high), device)
    action_scale = (action_high - action_low) / 2.0
    action_mean = (action_high + action_low) / 2.0
    if config.protocol == "public":
        hover_action = jnp.zeros((4,), dtype=jnp.float32, device=device)
    else:
        physical_hover = jnp.zeros((4,), dtype=jnp.float32, device=device)
        physical_hover = physical_hover.at[3].set(sim.data.params.mass[0, 0, 0] * 9.81)
        hover_action = (physical_hover - action_mean) / action_scale
    action_weights = jax.device_put(jnp.asarray(config.action_weights), device)
    delta_weights = jax.device_put(jnp.asarray(config.delta_action_weights), device)
    discounts = jax.device_put(
        jnp.power(config.gamma, jnp.arange(config.rollout_steps, dtype=jnp.float32)), device
    )
    sim_step = sim.build_step_fn()
    sim_reset = sim.build_reset_fn()

    def rollout_loss(
        actor_params: PyTree,
        rollout_data: SimData,
        rollout_episode_steps: Array,
        rollout_reset_next: Array,
        rollout_last_actions: Array,
        rollout_key: Array,
    ) -> tuple[Array, tuple[SimData, Array, Array, Array, Array, Array]]:
        """Differentiate rewards through one truncated simulator rollout."""

        def policy_step(
            carry: tuple[SimData, Array, Array, Array, Array], discount: Array
        ) -> tuple[tuple[SimData, Array, Array, Array, Array], tuple[Array, Array]]:
            current_data, episode_step, reset_mask, previous_action, rng_key = carry
            if config.protocol == "paper":
                current_data = sim_reset(current_data, default_data, reset_mask)
                episode_step = jnp.where(reset_mask, 0, episode_step)
                previous_action = jnp.where(reset_mask[:, None], 0.0, previous_action)

            obs = _observations(
                current_data, episode_step, previous_action, phase_offsets, sample_offsets, config
            )
            policy_action = _actor_mean(actor_params, obs)
            if config.stochastic:
                rng_key, sample_key = jax.random.split(rng_key)
                noise = jax.random.normal(sample_key, policy_action.shape)
                policy_action = policy_action + jnp.exp(actor_params["log_std"]) * noise
            # The public wrapper records the policy's yaw in action history but zeros yaw before
            # passing the command to the normalized simulator action interface.
            executed_action = policy_action.at[:, 2].set(0.0)
            normalized_action = jnp.clip(executed_action, -1.0, 1.0)
            physical_action = normalized_action * action_scale + action_mean
            current_data = current_data.replace(
                controls=current_data.controls.replace(
                    attitude=current_data.controls.attitude.replace(
                        staged_cmd=physical_action[:, None, :]
                    )
                )
            )
            current_data = sim_step(current_data, config.sim_steps_per_action)
            episode_step = episode_step + 1
            if config.protocol == "public":
                # The pinned jittable environment applies a pending autoreset after stepping. The
                # action chosen for that transition is retained in the outer action-history wrapper.
                current_data = sim_reset(current_data, default_data, reset_mask)
                episode_step = jnp.where(reset_mask, 0, episode_step)

            goal, _ = _reference(phase_offsets, episode_step, config)
            position_error = current_data.states.pos[:, 0] - goal
            position_distance = jnp.linalg.norm(position_error, axis=-1)
            reward = jnp.exp(-2.0 * position_distance)

            quaternion = current_data.states.quat[:, 0]
            attitude_angle = 2.0 * jnp.arctan2(
                jnp.linalg.norm(quaternion[:, :3], axis=-1),
                jnp.maximum(jnp.abs(quaternion[:, 3]), 1e-7),
            )
            reward = reward - config.angle_weight * attitude_angle
            reward = reward - jnp.sum(
                action_weights * jnp.square(policy_action - hover_action), axis=-1
            )
            reward = reward - jnp.sum(
                delta_weights * jnp.square(policy_action - previous_action), axis=-1
            )

            pos = current_data.states.pos[:, 0]
            out_of_bounds = jnp.any(
                (pos < jnp.array([-4.0, -4.0, 0.0])) | (pos > jnp.array([4.0, 4.0, 4.0])), axis=-1
            )
            truncated = episode_step >= config.episode_steps
            reset_mask = out_of_bounds | truncated
            if config.protocol == "public":
                reward = reward - out_of_bounds.astype(reward.dtype)
            loss = -discount * reward
            return (current_data, episode_step, reset_mask, policy_action, rng_key), (
                loss,
                jnp.square(position_distance),
            )

        final, (losses, squared_errors) = jax.lax.scan(
            policy_step,
            (
                rollout_data,
                rollout_episode_steps,
                rollout_reset_next,
                rollout_last_actions,
                rollout_key,
            ),
            discounts,
        )
        final_data, final_steps, final_reset, final_actions, final_key = final
        return jnp.mean(losses), (
            final_data,
            final_steps,
            final_reset,
            final_actions,
            final_key,
            jnp.sqrt(jnp.mean(squared_errors)),
        )

    loss_and_grad = jax.value_and_grad(rollout_loss, has_aux=True)

    def update(carry: TrainCarry, _: None) -> tuple[TrainCarry, TrainMetrics]:
        (loss, rollout_aux), gradients = loss_and_grad(
            carry.params,
            carry.data,
            carry.episode_steps,
            carry.reset_next,
            carry.last_actions,
            carry.key,
        )
        next_data, next_steps, next_reset, next_actions, next_key, rmse = rollout_aux
        next_params, next_optimizer = _optimizer_update(
            optimizer_transform, carry.params, gradients, carry.optimizer
        )
        next_carry = TrainCarry(
            next_params, next_optimizer, next_data, next_steps, next_reset, next_actions, next_key
        )
        return next_carry, TrainMetrics(loss, _tree_l2_norm(gradients), rmse)

    def train_fn(carry: TrainCarry) -> tuple[TrainCarry, TrainMetrics]:
        return jax.lax.scan(update, carry, xs=None, length=config.n_updates)

    def evaluate_fn(actor_params: PyTree, n_steps: int) -> Array:
        """Run a deterministic rollout from the phase-distributed initial states."""

        def evaluate_step(
            carry: tuple[SimData, Array, Array], _: None
        ) -> tuple[tuple[SimData, Array, Array], Array]:
            current_data, episode_step, previous_action = carry
            obs = _observations(
                current_data, episode_step, previous_action, phase_offsets, sample_offsets, config
            )
            policy_action = _actor_mean(actor_params, obs)
            normalized_action = jnp.clip(policy_action.at[:, 2].set(0.0), -1.0, 1.0)
            physical_action = normalized_action * action_scale + action_mean
            current_data = current_data.replace(
                controls=current_data.controls.replace(
                    attitude=current_data.controls.attitude.replace(
                        staged_cmd=physical_action[:, None, :]
                    )
                )
            )
            current_data = sim_step(current_data, config.sim_steps_per_action)
            episode_step = episode_step + 1
            goal, _ = _reference(phase_offsets, episode_step, config)
            squared_error = jnp.sum(jnp.square(current_data.states.pos[:, 0] - goal), axis=-1)
            return (current_data, episode_step, policy_action), squared_error

        _, squared_errors = jax.lax.scan(
            evaluate_step, (default_data, initial_steps, last_actions), xs=None, length=n_steps
        )
        # World 0 starts at phase zero. Reporting only that world matches the public artifact's
        # single-environment evaluation instead of averaging phase-distributed training worlds.
        return jnp.sqrt(jnp.mean(squared_errors[:, 0]))

    initial_carry = TrainCarry(
        params, optimizer, data, initial_steps, reset_next, last_actions, train_key
    )
    return BPTTTrainer(
        config, initial_carry, jax.jit(train_fn), jax.jit(evaluate_fn, static_argnums=1)
    )


def run_bptt(
    config: BPTTConfig, *, repeats: int = 1, evaluation_steps: int | None = None
) -> BPTTResult:
    """Compile, execute, and optionally evaluate a fused BPTT training run."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if evaluation_steps is None:
        evaluation_steps = (
            config.episode_steps
            if config.protocol == "public"
            else round(config.trajectory_time * config.env_freq)
        )
    if evaluation_steps < 0:
        raise ValueError("evaluation_steps must be nonnegative")

    trainer = build_trainer(config)
    compile_start = time.perf_counter()
    executable = trainer.train_fn.lower(trainer.initial_carry).compile()
    compile_seconds = time.perf_counter() - compile_start

    # Compilation and device executable initialization can be separate on some backends. Keep one
    # complete training execution out of the measurements, matching the paper's warm-run protocol.
    warm_carry, warm_metrics = executable(trainer.initial_carry)
    jax.block_until_ready((warm_carry, warm_metrics))
    del warm_carry, warm_metrics

    durations = []
    final_carry = trainer.initial_carry
    metrics = None
    for _ in range(repeats):
        start = time.perf_counter()
        final_carry, metrics = executable(trainer.initial_carry)
        jax.block_until_ready((final_carry, metrics))
        durations.append(time.perf_counter() - start)
    assert metrics is not None

    evaluation_rmse_mm = None
    if evaluation_steps:
        evaluation_rmse = trainer.evaluate_fn(final_carry.params, evaluation_steps)
        evaluation_rmse_mm = float(evaluation_rmse.block_until_ready()) * 1000.0

    parameter_delta = jax.tree.map(
        lambda final, initial: final - initial, final_carry.params, trainer.initial_carry.params
    )
    parameter_delta_norm = float(_tree_l2_norm(parameter_delta))
    mean_execution = sum(durations) / len(durations)
    best = min(durations)
    selected_device = jax.devices(config.device)[0]
    return BPTTResult(
        protocol=config.protocol,
        device=str(selected_device),
        requested_timesteps=config.total_timesteps,
        actual_timesteps=config.actual_timesteps,
        compile_seconds=compile_seconds,
        execution_seconds=tuple(durations),
        mean_execution_seconds=mean_execution,
        best_execution_seconds=best,
        steps_per_second=config.actual_timesteps / mean_execution,
        first_loss=float(metrics.loss[0]),
        final_loss=float(metrics.loss[-1]),
        first_gradient_norm=float(metrics.gradient_norm[0]),
        final_gradient_norm=float(metrics.gradient_norm[-1]),
        parameter_delta_norm=parameter_delta_norm,
        evaluation_steps=evaluation_steps,
        evaluation_rmse_mm=evaluation_rmse_mm,
        device_kind=selected_device.device_kind,
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        crazyflow_version=crazyflow_version,
        git_commit=_git_revision(),
    )


def _parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("public", "paper"), default="public")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--evaluation-steps",
        type=int,
        help="Override evaluation length (public default: 1000; paper default: 500).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use two environments, a four-action rollout, and two optimizer updates.",
    )
    return parser


def main() -> None:
    """Run the selected BPTT protocol and print machine-readable settings/results."""
    args = _parser().parse_args()
    config = public_config() if args.protocol == "public" else paper_config()
    overrides = {"device": args.device}
    for argument, field_name in (
        (args.total_timesteps, "total_timesteps"),
        (args.n_envs, "n_envs"),
        (args.rollout_steps, "rollout_steps"),
        (args.hidden_size, "hidden_size"),
    ):
        if argument is not None:
            overrides[field_name] = argument
    config = replace(config, **overrides)
    evaluation_steps = args.evaluation_steps
    if args.smoke:
        config = replace(config, n_envs=2, rollout_steps=4, total_timesteps=16)
        evaluation_steps = 8 if evaluation_steps is None else min(evaluation_steps, 8)

    caveat = (
        "current-API adaptation of public rl_diffsim@068239d"
        if config.protocol == "public"
        else "paper-informed reconstruction; exact paper trainer/config commit is unavailable"
    )
    print(json.dumps({"caveat": caveat, "config": asdict(config)}, sort_keys=True))
    result = run_bptt(config, repeats=args.repeats, evaluation_steps=evaluation_steps)
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
