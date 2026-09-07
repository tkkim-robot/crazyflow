"""Obstacle-free fallback learning with observed motor memory and persistent Adam history.

The actor observes local displacement, attitude, velocity, body rate, phase, four normalized
nominal-equivalent motor efforts, and a fixed skill identity. Dynamics parameters enter the
shared low-level adapter and plant, never the neural input. Reference trajectories use an
immutable teacher and nominal actuator model from the *same full 17-component initial state*.
Checkpoint files are deliberately incompatible with the accepted 13-state learner format.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.struct import dataclass as struct_dataclass

from crazyflow.safety.da_plcbf.actuator_dynamics import (
    ActuatorModel,
    allocate_wrench,
    effort_lag_step,
    validate_actuator_model,
)
from crazyflow.safety.da_plcbf.bptt import tree_all_finite
from crazyflow.safety.da_plcbf.direct_wrench import quaternion_to_rotation_matrix
from crazyflow.safety.da_plcbf.learner_checkpoint import _decode_arrays, _encode_arrays
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    SkillActorParams,
    SkillLibrarySpec,
    _duration_gate,
    _parameter_distance,
    _trainable_skill_tree,
    model_compensation_acceleration,
    spatial_descriptor_losses,
)
from crazyflow.safety.da_plcbf.state_conditioned_learning import proprioceptive_state_bank
from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel

_FEATURE_COUNT = 18
_CHECKPOINT_FORMAT = "crazyflow.actuator_skill_checkpoint"
_CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ActuatorSkillConfig(PersistentSkillConfig):
    """Accepted actor settings plus fixed observation scaling and shared actuator control."""

    model_compensation: bool = True
    wind_feedforward: bool = True
    smooth_motor_bounds: bool = False
    motor_state_scale: float = 0.10
    plant_substeps: int = 1
    adapter_mode: str = "F2"
    attitude_gain: float = 8e-4
    angular_rate_gain: float = 2e-4
    allow_reference_gain_mismatch: bool = False
    rollout_scan_unroll: int = 1

    def validate(self) -> None:
        PersistentSkillConfig.validate(self)
        if type(self.wind_feedforward) is not bool:
            raise TypeError("wind_feedforward must be boolean")
        if not isinstance(self.allow_reference_gain_mismatch, bool):
            raise TypeError("allow_reference_gain_mismatch must be boolean")
        if (
            isinstance(self.rollout_scan_unroll, bool)
            or not isinstance(self.rollout_scan_unroll, int)
            or self.rollout_scan_unroll < 1
        ):
            raise ValueError("rollout_scan_unroll must be a positive integer")
        if not all(
            math.isfinite(v) and v > 0
            for v in (self.motor_state_scale, self.attitude_gain, self.angular_rate_gain)
        ):
            raise ValueError("motor normalization and feedback gains must be positive finite")
        if (
            isinstance(self.plant_substeps, bool)
            or not isinstance(self.plant_substeps, int)
            or self.plant_substeps < 1
        ):
            raise ValueError("plant_substeps must be a positive integer")
        if self.adapter_mode not in {"F0", "F1", "F2"}:
            raise ValueError("adapter_mode must be F0, F1, or F2")
        if self.smooth_motor_bounds:
            raise ValueError(
                "actuator study uses the shared bounded allocator; use smooth_motor_bounds=False"
            )


@dataclass(frozen=True, slots=True)
class ActuatorReferenceConfig:
    """Independent body-motion targets and motor/control penalties, with early prefixes."""

    anchor_batch_size: int = 2
    trajectory_weight: float = 5.0
    velocity_weight: float = 1.0
    retention_weight: float = 5.0
    motor_state_target_weight: float = 0.0
    reference_braking_excess: bool = True
    # C1 positive-part width in squared-speed units (m/s)^2; zero preserves the original ReLU.
    reference_braking_huber_delta: float = 0.0
    trajectory_fractions: tuple[float, ...] = (0.10, 0.25, 0.5, 0.75, 1.0)
    # The legacy branch and its checkpoint fingerprint remain unchanged by default.
    objective_mode: str = "legacy"
    recovery_prefix_fraction: float = 0.25
    recovery_prefix_weight: float = 1.0
    recovery_balance_temperature: float = 0.25
    recovery_position_scale_m: float = 0.3
    recovery_velocity_scale_mps: float = 0.6
    recovery_braking_scale_mps: float = 0.8
    recovery_numerical_tolerance: float = 1e-6
    # A separate, explicitly labeled development-gradient-calibrated ablation.
    recovery_braking_priority: float = 10.0

    def validate(self) -> None:
        if self.objective_mode not in {
            "legacy",
            "balanced_reference",
            "balanced_reference_braking",
        }:
            raise ValueError("unknown reference objective_mode")
        if not 0 < self.recovery_prefix_fraction <= 1:
            raise ValueError("recovery_prefix_fraction must be within (0,1]")
        if not all(
            not isinstance(value, bool) and math.isfinite(value) and value > 0
            for value in (
                self.recovery_balance_temperature,
                self.recovery_position_scale_m,
                self.recovery_velocity_scale_mps,
                self.recovery_braking_scale_mps,
                self.recovery_braking_priority,
            )
        ):
            raise ValueError("recovery balance temperature and scales must be positive finite")
        for name in ("recovery_prefix_weight", "recovery_numerical_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative finite")
        if not isinstance(self.reference_braking_excess, bool):
            raise TypeError("reference_braking_excess must be boolean")
        if (
            isinstance(self.reference_braking_huber_delta, bool)
            or not math.isfinite(self.reference_braking_huber_delta)
            or self.reference_braking_huber_delta < 0
        ):
            raise ValueError("reference_braking_huber_delta must be nonnegative and finite")
        if self.reference_braking_huber_delta and not self.reference_braking_excess:
            raise ValueError("braking Huber smoothing requires reference_braking_excess=True")
        if (
            isinstance(self.anchor_batch_size, bool)
            or not isinstance(self.anchor_batch_size, int)
            or self.anchor_batch_size < 0
        ):
            raise ValueError("anchor_batch_size must be a nonnegative integer")
        if not all(
            math.isfinite(v) and v >= 0
            for v in (
                self.trajectory_weight,
                self.velocity_weight,
                self.retention_weight,
                self.motor_state_target_weight,
            )
        ):
            raise ValueError("reference weights must be finite and nonnegative")
        fractions = self.trajectory_fractions
        if (
            not fractions
            or not all(math.isfinite(v) and 0 < v <= 1 for v in fractions)
            or tuple(sorted(set(fractions))) != fractions
        ):
            raise ValueError("trajectory_fractions must increase strictly within (0,1]")


@dataclass(frozen=True, slots=True)
class ActuatorReferenceContract:
    """Immutable nominal teacher, nominal model, and physical/motor retention anchors."""

    params: SkillActorParams
    model: ActuatorModel
    anchors: jax.Array
    spec: SkillLibrarySpec
    actor_config: ActuatorSkillConfig
    learning_config: ActuatorReferenceConfig = ActuatorReferenceConfig()


@struct_dataclass
class ActuatorLearnerState:
    """Complete continuation state; a finite step always advances both public counters."""

    params: SkillActorParams
    previous_params: SkillActorParams
    optimizer_state: Any
    cumulative_gradient_steps: jax.Array
    latest_dynamics_estimate: ActuatorModel
    library_version: jax.Array


class ActuatorSkillRollout(NamedTuple):
    states: jax.Array
    commands: jax.Array
    requested_commands: jax.Array
    desired_accelerations: jax.Array
    desired_wrenches: jax.Array
    achieved_endpoint_wrenches: jax.Array
    policy_valid: jax.Array
    descriptors: jax.Array

    @property
    def valid(self) -> jax.Array:
        return jnp.all(self.policy_valid, axis=-1) & jnp.all(
            jnp.isfinite(self.states), axis=(-1, -2)
        )


class ActuatorLossMetrics(NamedTuple):
    total: jax.Array
    trajectory_tracking: jax.Array
    velocity_tracking: jax.Array
    attitude: jax.Array
    angular_rate: jax.Array
    motor_effort: jax.Array
    command_change: jax.Array
    saturation: jax.Array
    reference_retention: jax.Array
    motor_state_tracking: jax.Array
    terminal_braking: jax.Array
    trust: jax.Array
    descriptor_target: jax.Array
    diversity: jax.Array
    pairwise: jax.Array
    per_skill_position_error: jax.Array
    per_skill_velocity_error: jax.Array
    rollout_valid_fraction: jax.Array


class ActuatorBalancedLossMetrics(NamedTuple):
    """Additive reference-restoring terms; absolute auxiliary costs are absent."""

    total: jax.Array
    trajectory_tracking: jax.Array
    velocity_tracking: jax.Array
    prefix_tracking: jax.Array
    terminal_braking: jax.Array
    reference_retention: jax.Array
    trust: jax.Array
    per_skill_position_error: jax.Array
    per_skill_velocity_error: jax.Array
    per_skill_prefix_error: jax.Array
    per_skill_braking_error: jax.Array
    rollout_valid_fraction: jax.Array


class ActuatorStepMetrics(NamedTuple):
    loss: ActuatorLossMetrics | ActuatorBalancedLossMetrics
    gradient_norm: jax.Array
    parameter_update_norm: jax.Array
    finite_update_applied: jax.Array
    cumulative_gradient_steps: jax.Array
    library_version: jax.Array


class ActuatorLearnerFunctions(NamedTuple):
    initialize: Callable[..., ActuatorLearnerState]
    rollout: Callable[..., ActuatorSkillRollout]
    loss: Callable[..., tuple[jax.Array, ActuatorLossMetrics]]
    step: Callable[..., tuple[ActuatorLearnerState, ActuatorStepMetrics]]


def _validate_actor(
    params: SkillActorParams | None, spec: SkillLibrarySpec, config: ActuatorSkillConfig
) -> tuple[int, int]:
    config.validate()
    if spec.latent_codes.ndim != 2:
        raise ValueError("latent_codes must have shape (K,Z)")
    count, latent = spec.latent_codes.shape
    if count < 1 or latent < 1:
        raise ValueError("at least one skill and one latent feature are required")
    for name, shape in {
        "base_desired_velocities": (count, 3),
        "base_durations": (count,),
        "target_descriptors": (count, 9),
    }.items():
        if getattr(spec, name).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if params is not None:
        width = config.hidden_width
        for name, shape in {
            "velocity_offsets": (count, 3),
            "duration_offsets": (count,),
            "input_kernel": (_FEATURE_COUNT + latent, width),
            "input_bias": (width,),
            "hidden_kernel": (width, width),
            "hidden_bias": (width,),
            "output_kernel": (width, 3),
            "output_bias": (3,),
        }.items():
            if getattr(params, name).shape != shape:
                raise ValueError(f"17-state actor {name} must have shape {shape}")
    return count, latent


def initialize_actuator_skill_actor(
    key: jax.Array, spec: SkillLibrarySpec, config: ActuatorSkillConfig
) -> SkillActorParams:
    """Fresh motor-aware initialization; no zero padding or conversion of old checkpoints."""
    count, latent = _validate_actor(None, spec, config)
    keys = jax.random.split(key, 3)

    def glorot(random_key: jax.Array, inputs: int, outputs: int) -> jax.Array:
        limit = jnp.sqrt(6.0 / (inputs + outputs))
        return jax.random.uniform(
            random_key,
            (inputs, outputs),
            dtype=spec.latent_codes.dtype,
            minval=-limit,
            maxval=limit,
        )

    def zeros(shape: tuple[int, ...]) -> jax.Array:
        return jnp.zeros(shape, dtype=spec.latent_codes.dtype)

    width = config.hidden_width
    return SkillActorParams(
        (config.initial_skill_scale - 1.0) * spec.base_desired_velocities,
        zeros((count,)),
        glorot(keys[0], _FEATURE_COUNT + latent, width),
        zeros((width,)),
        glorot(keys[1], width, width),
        zeros((width,)),
        config.initial_residual_scale * glorot(keys[2], width, 3),
        zeros((3,)),
    )


def build_single_recovery_spec(
    *, latent_size: int = 8, dtype: Any = jnp.float32
) -> SkillLibrarySpec:
    """An explicit velocity-braking recovery policy to train on the complete state bank."""
    if isinstance(latent_size, bool) or not isinstance(latent_size, int) or latent_size < 1:
        raise ValueError("latent_size must be a positive integer")
    return SkillLibrarySpec(
        jnp.ones((1, latent_size), dtype=dtype),
        jnp.zeros((1, 3), dtype=dtype),
        jnp.ones((1,), dtype=dtype),
        jnp.zeros((1, 9), dtype=dtype),
    )


def actuator_skill_actions(
    params: SkillActorParams,
    spec: SkillLibrarySpec,
    states: jax.Array,
    skill_start_position: jax.Array,
    phase: jax.Array,
    config: ActuatorSkillConfig,
) -> jax.Array:
    """Behavior acceleration from proprioception only; no model, obstacle, or goal argument."""
    count, _ = _validate_actor(params, spec, config)
    if states.shape != (count, 17):
        raise ValueError("states must have shape (K,17)")
    if skill_start_position.shape != (3,) or jnp.ndim(phase) != 0:
        raise ValueError("skill anchor and phase must have shapes (3,) and ()")
    quaternion = states[:, 3:7] / jnp.linalg.norm(states[:, 3:7], axis=-1, keepdims=True)
    features = jnp.concatenate(
        (
            (states[:, :3] - skill_start_position) / config.position_scale,
            quaternion,
            states[:, 7:10] / config.velocity_scale,
            states[:, 10:13] / config.angular_velocity_scale,
            jnp.broadcast_to(phase, (count, 1)),
            states[:, 13:17] / config.motor_state_scale,
            spec.latent_codes,
        ),
        axis=-1,
    )
    hidden = jnp.tanh(features @ params.input_kernel + params.input_bias)
    hidden = jnp.tanh(hidden @ params.hidden_kernel + params.hidden_bias)
    residual = config.residual_scale * jnp.tanh(hidden @ params.output_kernel + params.output_bias)
    elapsed = phase * config.horizon * config.dt
    duration = jnp.clip(
        spec.base_durations + params.duration_offsets,
        config.duration_transition,
        config.horizon * config.dt,
    )
    gate = _duration_gate(elapsed, duration, config.duration_transition)
    offsets = params.velocity_offsets
    if config.velocity_offset_limit is not None:
        offsets = jnp.clip(offsets, -config.velocity_offset_limit, config.velocity_offset_limit)
    structured = config.policy_gain * (
        gate[:, None] * (spec.base_desired_velocities + offsets) - states[:, 7:10]
    )
    residual_gate = gate[:, None] if config.gate_residual_with_skill_duration else 1.0
    action = config.acceleration_limit * jnp.tanh(
        (structured + residual_gate * residual) / config.acceleration_limit
    )
    finite = tree_all_finite((params, spec, states, skill_start_position, phase))
    return jnp.where(finite, action, jnp.full_like(action, jnp.nan))


def acceleration_to_actuator_command(
    desired_acceleration: jax.Array,
    states: jax.Array,
    model: ActuatorModel,
    config: ActuatorSkillConfig,
) -> Any:
    """Shared causal geometric controller and F0/F1/F2 map at the declared control period.

    Known external-force compensation is applied here, identically for nominal, fallback and
    emergency accelerations. The returned achieved wrench denotes the predicted motor endpoint,
    not a wrench that may be applied directly or held throughout the lagged plant integration.
    """
    config.validate()
    if states.shape[-1:] != (17,) or desired_acceleration.shape != (*states.shape[:-1], 3):
        raise ValueError("states and acceleration must have shapes (...,17) and (...,3)")
    acceleration = desired_acceleration
    if config.model_compensation:
        compensation_model = model.body
        if not config.wind_feedforward:
            # Remove only direct wind cancellation. The physical/prediction model
            # retains its actual wind; calm-air drag compensation is unchanged.
            compensation_model = compensation_model._replace(
                wind_velocity=jnp.zeros_like(compensation_model.wind_velocity)
            )
        acceleration = acceleration + model_compensation_acceleration(
            states[..., :13], compensation_model
        )
    force = jnp.reshape(model.body.mass, ()) * (acceleration - model.body.gravity_vec)
    force_norm = jnp.linalg.norm(force, axis=-1, keepdims=True)
    body_z = force / jnp.where(force_norm > 1e-6, force_norm, 1.0)
    heading = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0], dtype=states.dtype), force.shape)
    body_y_raw = jnp.cross(body_z, heading)
    heading_norm = jnp.linalg.norm(body_y_raw, axis=-1, keepdims=True)
    body_y = body_y_raw / jnp.where(heading_norm > 1e-6, heading_norm, 1.0)
    desired_rotation = jnp.stack((jnp.cross(body_y, body_z), body_y, body_z), axis=-1)
    rotation = quaternion_to_rotation_matrix(states[..., 3:7])
    skew = (
        jnp.swapaxes(desired_rotation, -1, -2) @ rotation
        - jnp.swapaxes(rotation, -1, -2) @ desired_rotation
    )
    attitude_error = 0.5 * jnp.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), axis=-1)
    rates = states[..., 10:13]
    momentum = (model.body.inertia @ rates[..., None])[..., 0]
    torque = (
        -config.attitude_gain * attitude_error
        - config.angular_rate_gain * rates
        + jnp.cross(rates, momentum)
    )
    collective = jnp.sum(force * rotation[..., :, 2], axis=-1, keepdims=True)
    desired_wrench = jnp.concatenate((collective, torque), axis=-1)
    valid = (force_norm[..., 0] > 1e-6) & (heading_norm[..., 0] > 1e-6)
    valid = (
        valid
        & jnp.all(jnp.isfinite(states), axis=-1)
        & jnp.all(jnp.isfinite(desired_wrench), axis=-1)
    )
    desired_wrench = jnp.where(valid[..., None], desired_wrench, jnp.nan)
    return allocate_wrench(
        desired_wrench,
        states[..., 13:17],
        model,
        config.dt * config.control_interval_steps,
        mode=config.adapter_mode,
    )


def rollout_actuator_skill_library(
    params: SkillActorParams,
    spec: SkillLibrarySpec,
    initial_state: jax.Array,
    model: ActuatorModel,
    config: ActuatorSkillConfig,
) -> ActuatorSkillRollout:
    """Predict every skill from the full motor/body state with exactly held deployment commands."""
    count, _ = _validate_actor(params, spec, config)
    if initial_state.shape != (17,):
        raise ValueError("initial_state must have shape (17,)")
    current = jnp.broadcast_to(initial_state, (count, 17))
    hold_steps = config.control_interval_steps

    def hold(state: jax.Array, boundary: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        phase = boundary * hold_steps / config.horizon
        acceleration = actuator_skill_actions(params, spec, state, initial_state[:3], phase, config)
        allocation = acceleration_to_actuator_command(acceleration, state, model, config)

        def integrate(value: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
            following = effort_lag_step(
                value, allocation.command, model, config.dt, substeps=config.plant_substeps
            )
            return following, following

        final, future = jax.lax.scan(
            integrate, state, None, length=hold_steps, unroll=min(hold_steps, 2)
        )
        diagnostics = (
            allocation.command,
            allocation.requested_command,
            acceleration,
            allocation.achieved_wrench - allocation.wrench_residual,
            allocation.achieved_wrench,
            allocation.valid,
        )
        repeated = tuple(jnp.broadcast_to(v, (hold_steps, *v.shape)) for v in diagnostics)
        return final, (future, *repeated)

    _, outputs = jax.lax.scan(
        hold,
        current,
        jnp.arange(math.ceil(config.horizon / hold_steps)),
        unroll=config.rollout_scan_unroll,
    )

    def time_to_policy(value: jax.Array) -> jax.Array:
        flattened = value.reshape((-1, *value.shape[2:]))[: config.horizon]
        return jnp.swapaxes(flattened, 0, 1)

    future, commands, requested, acceleration, desired_wrench, achieved, valid = map(
        time_to_policy, outputs
    )
    states = jnp.concatenate((current[:, None], future), axis=1)
    descriptors = jnp.concatenate(
        (
            states[:, -1, :3] - initial_state[:3],
            jnp.mean(states[:, 1:, 7:10], axis=1),
            states[:, -1, 7:10],
        ),
        axis=-1,
    )
    return ActuatorSkillRollout(
        states, commands, requested, acceleration, desired_wrench, achieved, valid, descriptors
    )


def actuator_proprioceptive_state_bank(
    model: ActuatorModel, *, dtype: Any = jnp.float32
) -> tuple[jax.Array, tuple[str, ...]]:
    """Rest, mission motion, attitude/rate perturbations and dynamically consistent transients.

    Transient entries are advanced through the actual lagged plant from known bounded motor
    states and commands, retaining the coupled body motion. They are never created by resetting
    motors after changing the body state or fault model.
    """
    body_states, body_labels = proprioceptive_state_bank(dtype=dtype)
    wrench = np.asarray(
        [
            -float(np.asarray(model.body.mass)) * float(np.asarray(model.body.gravity_vec)[2]),
            0,
            0,
            0,
        ]
    )
    force = np.linalg.solve(np.asarray(model.force_to_wrench, dtype=np.float64), wrench)
    trim = force / np.asarray(model.effectiveness)
    lower, upper = np.asarray(model.command_lower), np.asarray(model.command_upper)
    if not np.all(np.isfinite(trim)) or np.any(trim < lower) or np.any(trim > upper):
        raise ValueError("state bank requires a feasible nominal hover trim")
    motor = jnp.asarray(trim, dtype=dtype)
    states = [jnp.concatenate((state, motor)) for state in body_states]
    labels = list(body_labels)
    patterns = (
        (1.25, 1.25, 1.25, 1.25),
        (0.75, 0.75, 0.75, 0.75),
        (1.12, 0.88, 1.0, 1.0),
        (1.0, 1.0, 0.88, 1.12),
    )
    for initial_index in (0, 5, 6):
        for pattern_index, pattern in enumerate(patterns):
            command = jnp.clip(
                motor * jnp.asarray(pattern, dtype=dtype), model.command_lower, model.command_upper
            )
            transient = effort_lag_step(states[initial_index], command, model, 0.02, substeps=4)
            states.append(transient)
            labels.append(f"{body_labels[initial_index]}_motor_transient_{pattern_index}")
    return jnp.stack(states), tuple(labels)


def _validate_contract(contract: ActuatorReferenceContract) -> None:
    _validate_actor(contract.params, contract.spec, contract.actor_config)
    validate_actuator_model(contract.model)
    contract.learning_config.validate()
    if (
        contract.anchors.ndim != 2
        or contract.anchors.shape[1:] != (17,)
        or not len(contract.anchors)
    ):
        raise ValueError("reference anchors must have shape (B,17), B >= 1")
    if contract.learning_config.anchor_batch_size > len(contract.anchors):
        raise ValueError("anchor batch cannot exceed the full fixed bank")
    if not all(
        np.all(np.isfinite(np.asarray(v)))
        for v in jax.tree.leaves((contract.params, contract.model, contract.anchors, contract.spec))
    ):
        raise ValueError("reference contract must contain finite numeric inputs")


def _braking_positive_part(excess: jax.Array, delta: float) -> jax.Array:
    """Original ReLU or a declared C1 Huber positive part in squared-speed coordinates."""
    if delta == 0.0:
        return jax.nn.relu(excess)
    return jnp.where(excess < delta, jax.nn.relu(excess) ** 2 / (2 * delta), excess - delta / 2)


def _smooth_worst_skill(values: jax.Array, temperature: float) -> jax.Array:
    """Log-mean-exp over skills, exactly zero at zero without a norm singularity."""
    maximum = jnp.max(values, axis=-1)
    return maximum + temperature * jnp.log(
        jnp.mean(jnp.exp((values - maximum[..., None]) / temperature), axis=-1)
    )


def balanced_reference_terms(
    states: jax.Array,
    references: jax.Array,
    config: ActuatorSkillConfig,
    settings: ActuatorReferenceConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-state/per-skill motion errors in declared physical units; no scene inputs.

    Prefix loss covers every node through the first quarter of the maneuver. Braking
    restores the terminal velocity vector and penalizes squared excess speed-squared.
    Both derivatives vanish at identical teacher motion, including exact speed equality.
    Motor matching and absolute effort/attitude/rate/saturation/diversity are disabled in
    this objective: a changed effectiveness may require more motor effort for the same force.
    """
    nodes = jnp.asarray(
        sorted({max(1, round(config.horizon * f)) for f in settings.trajectory_fractions})
    )
    prefix_end = max(1, round(config.horizon * settings.recovery_prefix_fraction))

    def resolved_error(error: jax.Array) -> jax.Array:
        # Teacher anchor/current rollouts use separate float32 executables. A microscopic
        # declared dead zone prevents Adam from amplifying their roundoff disagreement.
        return jax.nn.relu(jnp.abs(error) - settings.recovery_numerical_tolerance)

    position = resolved_error(
        (states[..., :3] - references[..., :3]) / settings.recovery_position_scale_m
    )
    velocity = resolved_error(
        (states[..., 7:10] - references[..., 7:10]) / settings.recovery_velocity_scale_mps
    )
    trajectory = jnp.mean(position[..., nodes, :] ** 2, axis=(-1, -2))
    tracking = jnp.mean(velocity[..., nodes, :] ** 2, axis=(-1, -2))
    prefix = jnp.mean(position[..., 1 : prefix_end + 1, :] ** 2, axis=(-1, -2)) + jnp.mean(
        velocity[..., 1 : prefix_end + 1, :] ** 2, axis=(-1, -2)
    )
    terminal = states[..., -1, 7:10]
    reference_terminal = references[..., -1, 7:10]
    braking = (
        jnp.mean(
            resolved_error((terminal - reference_terminal) / settings.recovery_braking_scale_mps)
            ** 2,
            axis=-1,
        )
        + (
            jax.nn.relu(
                jnp.sum(terminal**2 - reference_terminal**2, axis=-1)
                / settings.recovery_braking_scale_mps**2
                - settings.recovery_numerical_tolerance
            )
        )
        ** 2
    )
    return trajectory, tracking, prefix, braking


def _balanced_reference_loss(
    params: SkillActorParams,
    previous_params: SkillActorParams,
    actual: ActuatorSkillRollout,
    references: jax.Array,
    model: ActuatorModel,
    config: ActuatorSkillConfig,
    settings: ActuatorReferenceConfig,
) -> tuple[jax.Array, ActuatorBalancedLossMetrics]:
    terms = balanced_reference_terms(actual.states, references, config, settings)
    reduced = tuple(
        _smooth_worst_skill(term, settings.recovery_balance_temperature) for term in terms
    )
    trajectory, velocity, prefix, braking = (value[0] for value in reduced)
    braking_weight = config.terminal_braking_weight * (
        settings.recovery_braking_priority
        if settings.objective_mode == "balanced_reference_braking"
        else 1.0
    )
    per_state_total = (
        settings.trajectory_weight * reduced[0]
        + settings.velocity_weight * reduced[1]
        + settings.recovery_prefix_weight * reduced[2]
        + braking_weight * reduced[3]
    )
    retention = (
        jnp.mean(per_state_total[1:])
        if settings.anchor_batch_size
        else jnp.zeros((), dtype=actual.states.dtype)
    )
    trust = _parameter_distance(params, previous_params)
    total = per_state_total[0] + settings.retention_weight * retention + config.trust_weight * trust
    finite = tree_all_finite((params, previous_params, model, actual.states, references)) & jnp.all(
        actual.policy_valid
    )
    total = jnp.where(finite & jnp.isfinite(total), total, jnp.inf)
    return total, ActuatorBalancedLossMetrics(
        total,
        trajectory,
        velocity,
        prefix,
        braking,
        retention,
        trust,
        *(term[0] for term in terms),
        jnp.mean(actual.policy_valid),
    )


def actuator_reference_loss(
    params: SkillActorParams,
    initial_state: jax.Array,
    model: ActuatorModel,
    previous_params: SkillActorParams,
    iteration: jax.Array,
    *,
    contract: ActuatorReferenceContract,
    config: ActuatorSkillConfig,
    anchor_reference_states: jax.Array,
    matched_reference: bool = False,
) -> tuple[jax.Array, ActuatorLossMetrics | ActuatorBalancedLossMetrics]:
    """Multi-resolution same-state motion recovery with separate motor and retention terms."""
    settings = contract.learning_config
    count = settings.anchor_batch_size
    indices = (iteration * max(count, 1) + jnp.arange(count)) % len(contract.anchors)
    initial = jnp.concatenate((initial_state[None], contract.anchors[indices]), axis=0)
    behavior_fields = (
        "dt",
        "horizon",
        "control_interval_steps",
        "plant_substeps",
        "rollout_scan_unroll",
        "position_scale",
        "velocity_scale",
        "angular_velocity_scale",
        "motor_state_scale",
        "residual_scale",
        "duration_transition",
        "velocity_offset_limit",
        "policy_gain",
        "gate_residual_with_skill_duration",
        "acceleration_limit",
        "model_compensation",
        "wind_feedforward",
        "attitude_gain",
        "angular_rate_gain",
        "adapter_mode",
    )
    same_behavior_config = all(
        getattr(config, name) == getattr(contract.actor_config, name) for name in behavior_fields
    )
    if matched_reference and same_behavior_config:
        # A single compiled loop body evaluates both parameter/model pairs. Separate
        # nominally identical graphs can disagree after fused reverse-mode compilation.
        # This is shared arithmetic, not an equality test or a conditional zero loss.
        paired_params, paired_models = jax.tree.map(
            lambda a, b: jnp.stack((a, b)), (params, model), (contract.params, contract.model)
        )

        def evaluate(pair: tuple) -> ActuatorSkillRollout:
            p, dynamics = pair
            return jax.vmap(
                lambda y: rollout_actuator_skill_library(p, contract.spec, y, dynamics, config)
            )(initial)

        paired = jax.lax.map(evaluate, (paired_params, paired_models))
        actual = jax.tree.map(lambda value: value[0], paired)
        teacher_batch = jax.tree.map(lambda value: value[1], paired)
    else:
        actual = jax.vmap(
            lambda y: rollout_actuator_skill_library(params, contract.spec, y, model, config)
        )(initial)
        if matched_reference:
            # Explicitly opted-in unequal feedback gains describe different behaviors;
            # preserve the fixed teacher rather than silently adopting the student gains.
            teacher_batch = jax.vmap(
                lambda y: rollout_actuator_skill_library(
                    contract.params, contract.spec, y, contract.model, contract.actor_config
                )
            )(initial)
    if matched_reference:
        teacher = jax.tree.map(lambda x: x[0], teacher_batch)
        references = jax.lax.stop_gradient(teacher_batch.states)
    else:
        teacher = rollout_actuator_skill_library(
            contract.params, contract.spec, initial_state, contract.model, contract.actor_config
        )
        references = jax.lax.stop_gradient(
            jnp.concatenate((teacher.states[None], anchor_reference_states[indices]), axis=0)
        )
    if settings.objective_mode in {"balanced_reference", "balanced_reference_braking"}:
        return _balanced_reference_loss(
            params, previous_params, actual, references, model, config, settings
        )
    nodes = jnp.asarray(
        sorted({max(1, round(config.horizon * f)) for f in settings.trajectory_fractions})
    )
    position_error = (
        actual.states[:, :, nodes, :3] - references[:, :, nodes, :3]
    ) / config.position_scale
    velocity_error = (
        actual.states[:, :, nodes, 7:10] - references[:, :, nodes, 7:10]
    ) / config.velocity_scale
    trajectory = jnp.mean(position_error[0] ** 2)
    velocity = jnp.mean(velocity_error[0] ** 2)
    retention = (
        jnp.mean(position_error[1:] ** 2) + jnp.mean(velocity_error[1:] ** 2)
        if count
        else jnp.asarray(0.0, dtype=initial_state.dtype)
    )
    rotation = quaternion_to_rotation_matrix(actual.states[..., 3:7])
    attitude = jnp.mean(1.0 - rotation[..., 2, 2])
    rate = jnp.mean((actual.states[..., 10:13] / config.angular_velocity_scale) ** 2)
    effort = jnp.mean((actual.states[..., 13:17] / config.motor_state_scale) ** 2)
    motor_tracking = jnp.mean(
        ((actual.states[..., 13:17] - references[..., 13:17]) / config.motor_state_scale) ** 2
    )
    commands_at_boundaries = actual.commands[:, :, :: config.control_interval_steps]
    command_change = (
        jnp.mean((jnp.diff(commands_at_boundaries, axis=2) / config.motor_state_scale) ** 2)
        if commands_at_boundaries.shape[2] > 1
        else jnp.asarray(0.0, dtype=initial_state.dtype)
    )
    center = (model.command_lower + model.command_upper) / 2
    half_width = (model.command_upper - model.command_lower) / 2
    coordinate = (actual.requested_commands - center) / half_width
    excess = (
        jax.nn.softplus((jnp.abs(coordinate) - 1.0) / config.saturation_temperature)
        * config.saturation_temperature
    )
    saturation = jnp.mean(excess**2)
    terminal_squared = jnp.sum(actual.states[:, :, -1, 7:10] ** 2, axis=-1)
    if settings.reference_braking_excess:
        terminal_squared = _braking_positive_part(
            terminal_squared - jnp.sum(references[:, :, -1, 7:10] ** 2, axis=-1),
            settings.reference_braking_huber_delta,
        )
    braking = jnp.mean(terminal_squared) / (3 * config.velocity_scale**2)
    trust = _parameter_distance(params, previous_params)
    if contract.spec.latent_codes.shape[0] > 1:
        spatial = spatial_descriptor_losses(
            actual.descriptors[0], jax.lax.stop_gradient(teacher.descriptors), config
        )
        descriptor_target, diversity, pairwise = spatial
    else:
        descriptor_target = jnp.mean(
            (
                (actual.descriptors[0, :, :3] - jax.lax.stop_gradient(teacher.descriptors[:, :3]))
                / config.position_scale
            )
            ** 2
        )
        diversity = pairwise = jnp.asarray(0.0, dtype=initial_state.dtype)
    total = (
        settings.trajectory_weight * trajectory
        + settings.velocity_weight * velocity
        + settings.retention_weight * retention
        + settings.motor_state_target_weight * motor_tracking
        + config.attitude_weight * attitude
        + config.angular_rate_weight * rate
        + config.action_weight * effort
        + config.action_rate_weight * command_change
        + config.saturation_weight * saturation
        + config.terminal_braking_weight * braking
        + config.trust_weight * trust
        + config.target_weight * descriptor_target
        + config.diversity_weight * diversity
        + config.pairwise_weight * pairwise
    )
    finite = tree_all_finite((params, previous_params, model, actual.states, references)) & jnp.all(
        actual.policy_valid
    )
    total = jnp.where(finite & jnp.isfinite(total), total, jnp.inf)
    metrics = ActuatorLossMetrics(
        total,
        trajectory,
        velocity,
        attitude,
        rate,
        effort,
        command_change,
        saturation,
        retention,
        motor_tracking,
        braking,
        trust,
        descriptor_target,
        diversity,
        pairwise,
        jnp.mean(position_error[0] ** 2, axis=(1, 2)),
        jnp.mean(velocity_error[0] ** 2, axis=(1, 2)),
        jnp.mean(actual.policy_valid),
    )
    return total, metrics


def _optimizer(config: ActuatorSkillConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
    )


def _initialize_state(
    params: SkillActorParams,
    model: ActuatorModel,
    spec: SkillLibrarySpec,
    config: ActuatorSkillConfig,
) -> ActuatorLearnerState:
    _validate_actor(params, spec, config)
    return ActuatorLearnerState(
        params,
        params,
        _optimizer(config).init(params),
        jnp.asarray(0, dtype=jnp.int32),
        model,
        jnp.asarray(0, dtype=jnp.int32),
    )


def build_actuator_skill_learner(
    contract: ActuatorReferenceContract,
    config: ActuatorSkillConfig | None = None,
    *,
    reference_numerics: str = "matched",
) -> ActuatorLearnerFunctions:
    """One persistent optimizer; all finite proposals publish without any quality gate."""
    _validate_contract(contract)
    if reference_numerics not in {"matched", "legacy"}:
        raise ValueError("reference_numerics must be matched or legacy")
    config = contract.actor_config if config is None else config
    config.validate()
    # These define the actual actor/plant observation and execution contract. Learning-rate and
    # regularization tuning may differ; phase, motors and low-level behavior cannot change silently.
    # rollout_scan_unroll is compile-only. The matched path uses the same unroll on
    # both sides while preserving the immutable teacher's physical behavior settings.
    for field in (
        "dt",
        "horizon",
        "control_interval_steps",
        "plant_substeps",
        "motor_state_scale",
        "adapter_mode",
        "model_compensation",
        "wind_feedforward",
        "gate_residual_with_skill_duration",
    ):
        if getattr(config, field) != getattr(contract.actor_config, field):
            raise ValueError(f"teacher and adapted actor must preserve {field}")
    for field in ("attitude_gain", "angular_rate_gain"):
        if (
            getattr(config, field) != getattr(contract.actor_config, field)
            and not config.allow_reference_gain_mismatch
        ):
            raise ValueError(
                f"teacher/current {field} mismatch requires allow_reference_gain_mismatch=True"
            )
    optimizer = _optimizer(config)
    anchor_reference = (
        jax.jit(
            jax.vmap(
                lambda y: (
                    rollout_actuator_skill_library(
                        contract.params, contract.spec, y, contract.model, contract.actor_config
                    ).states
                )
            )
        )(contract.anchors)
        if reference_numerics == "legacy"
        else jnp.empty((0,), jnp.float32)
    )
    jax.block_until_ready(anchor_reference)

    def loss(
        params: SkillActorParams,
        initial_state: jax.Array,
        model: ActuatorModel,
        previous: SkillActorParams,
        iteration: jax.Array | None = None,
        teacher_params: SkillActorParams | None = None,
        teacher_model: ActuatorModel | None = None,
    ) -> tuple[jax.Array, ActuatorLossMetrics]:
        return actuator_reference_loss(
            params,
            initial_state,
            model,
            previous,
            jnp.asarray(0, dtype=jnp.int32) if iteration is None else iteration,
            contract=contract
            if teacher_params is None
            else replace(
                contract,
                params=teacher_params,
                model=teacher_model,
                actor_config=replace(
                    contract.actor_config, rollout_scan_unroll=config.rollout_scan_unroll
                ),
            ),
            config=config,
            anchor_reference_states=anchor_reference,
            matched_reference=reference_numerics == "matched",
        )

    def step(
        state: ActuatorLearnerState,
        initial_state: jax.Array,
        model: ActuatorModel,
        teacher_params: SkillActorParams | None = None,
        teacher_model: ActuatorModel | None = None,
    ) -> tuple[ActuatorLearnerState, ActuatorStepMetrics]:
        (_, metrics), gradients = jax.value_and_grad(
            lambda p: loss(
                p,
                initial_state,
                model,
                state.previous_params,
                state.library_version,
                teacher_params,
                teacher_model,
            ),
            has_aux=True,
        )(state.params)
        updates, proposed_optimizer = optimizer.update(
            _trainable_skill_tree(gradients, config), state.optimizer_state, params=state.params
        )
        updates = _trainable_skill_tree(updates, config)
        if config.max_parameter_update_norm is not None:
            scale = jnp.minimum(
                1.0, config.max_parameter_update_norm / jnp.maximum(optax.tree.norm(updates), 1e-30)
            )
            updates = jax.tree.map(lambda v: v * scale, updates)
        proposed = optax.apply_updates(state.params, updates)
        if config.velocity_offset_limit is not None:
            proposed = proposed.replace(
                velocity_offsets=jnp.clip(
                    proposed.velocity_offsets,
                    -config.velocity_offset_limit,
                    config.velocity_offset_limit,
                )
            )
        finite = tree_all_finite((metrics, gradients, proposed, proposed_optimizer))

        def select(a: jax.Array, b: jax.Array) -> jax.Array:
            return jnp.where(finite, a, b)

        params = jax.tree.map(select, proposed, state.params)
        optimizer_state = jax.tree.map(select, proposed_optimizer, state.optimizer_state)
        previous = jax.tree.map(select, state.params, state.previous_params)
        increment = finite.astype(jnp.int32)
        latest = jax.tree.map(
            lambda a, b: jnp.where(tree_all_finite(model), a, b),
            model,
            state.latest_dynamics_estimate,
        )
        following = state.replace(
            params=params,
            previous_params=previous,
            optimizer_state=optimizer_state,
            cumulative_gradient_steps=state.cumulative_gradient_steps + increment,
            latest_dynamics_estimate=latest,
            library_version=state.library_version + increment,
        )
        update_norm = optax.tree.norm(jax.tree.map(lambda a, b: a - b, params, state.params))
        return following, ActuatorStepMetrics(
            metrics,
            optax.tree.norm(gradients),
            update_norm,
            finite,
            following.cumulative_gradient_steps,
            following.library_version,
        )

    compiled_loss, compiled_step = jax.jit(loss), jax.jit(step)

    def matched_loss(
        params: SkillActorParams,
        initial_state: jax.Array,
        model: ActuatorModel,
        previous: SkillActorParams,
        iteration: jax.Array | None = None,
    ) -> tuple[jax.Array, ActuatorLossMetrics]:
        return compiled_loss(
            params, initial_state, model, previous, iteration, contract.params, contract.model
        )

    def matched_step(
        state: ActuatorLearnerState, initial_state: jax.Array, model: ActuatorModel
    ) -> tuple[ActuatorLearnerState, ActuatorStepMetrics]:
        return compiled_step(state, initial_state, model, contract.params, contract.model)

    return ActuatorLearnerFunctions(
        lambda params, model: _initialize_state(params, model, contract.spec, config),
        jax.jit(
            lambda params, state, model: rollout_actuator_skill_library(
                params, contract.spec, state, model, config
            )
        ),
        matched_loss if reference_numerics == "matched" else compiled_loss,
        matched_step if reference_numerics == "matched" else compiled_step,
    )


@dataclass(frozen=True, slots=True)
class ActuatorLearnerCheckpoint:
    """Versioned full-state continuation, including the immutable reference contract."""

    state: ActuatorLearnerState
    contract: ActuatorReferenceContract
    config: ActuatorSkillConfig
    physical_state: np.ndarray
    metadata: dict[str, Any]
    npz_path: Path
    json_path: Path
    sha256: str


def _paths(path_stem: str | Path) -> tuple[Path, Path]:
    stem = Path(path_stem)
    if stem.suffix in {".npz", ".json"}:
        stem = stem.with_suffix("")
    return Path(f"{stem}.npz"), Path(f"{stem}.json")


def _model_dictionary(model: ActuatorModel) -> dict[str, Any]:
    return {
        "body": dict(model.body._asdict()),
        **{name: getattr(model, name) for name in model._fields if name != "body"},
    }


def _restore_model(values: dict[str, Any]) -> ActuatorModel:
    return ActuatorModel(
        body=VersionAModel(**values["body"]), **{k: v for k, v in values.items() if k != "body"}
    )


def _restore_config(values: dict[str, Any]) -> ActuatorSkillConfig:
    values = {**values, "descriptor_scales": tuple(values["descriptor_scales"])}
    config = ActuatorSkillConfig(**values)
    config.validate()
    return config


def _reference_payload(contract: ActuatorReferenceContract) -> dict[str, Any]:
    return {
        "params": serialization.to_state_dict(contract.params),
        "model": _model_dictionary(contract.model),
        "anchors": contract.anchors,
        "spec": serialization.to_state_dict(contract.spec),
    }


def actuator_reference_fingerprint(contract: ActuatorReferenceContract) -> str:
    """Bind nominal parameters/model/anchors, identities, input scales and objective settings."""
    arrays: dict[str, np.ndarray] = {}
    structure = _encode_arrays(_reference_payload(contract), arrays)
    learning_options = asdict(contract.learning_config)
    # The original v1 contract implied excess braking. Canonicalizing that default keeps
    # already-written checksummed pilot checkpoints readable while binding the explicit ablation.
    if learning_options["reference_braking_excess"]:
        del learning_options["reference_braking_excess"]
    if learning_options["reference_braking_huber_delta"] == 0.0:
        del learning_options["reference_braking_huber_delta"]
    # New opt-in recovery settings must not invalidate historical legacy checkpoints.
    defaults = asdict(ActuatorReferenceConfig())
    for name in (
        "objective_mode",
        "recovery_prefix_fraction",
        "recovery_prefix_weight",
        "recovery_balance_temperature",
        "recovery_position_scale_m",
        "recovery_velocity_scale_mps",
        "recovery_braking_scale_mps",
        "recovery_numerical_tolerance",
        "recovery_braking_priority",
    ):
        if learning_options[name] == defaults[name]:
            del learning_options[name]
    actor_options = asdict(contract.actor_config)
    # Existing checkpoints imply enabled wind feedforward. Bind the opt-out
    # without changing their established nominal-reference fingerprints.
    if actor_options["wind_feedforward"]:
        del actor_options["wind_feedforward"]
    if not actor_options["allow_reference_gain_mismatch"]:
        del actor_options["allow_reference_gain_mismatch"]
    # v1 checkpoints predate this compile-only option and implied scan unroll=1.
    if actor_options["rollout_scan_unroll"] == 1:
        del actor_options["rollout_scan_unroll"]
    digest = hashlib.sha256(
        json.dumps(
            {
                "structure": structure,
                "actor_config": actor_options,
                "learning_config": learning_options,
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
    )
    for key, array in arrays.items():
        digest.update(key.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def save_actuator_learner_checkpoint(
    state: ActuatorLearnerState,
    contract: ActuatorReferenceContract,
    physical_state: jax.Array | np.ndarray,
    path_stem: str | Path,
    *,
    config: ActuatorSkillConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save numeric arrays and JSON without pickle, overwrites, or 13-state compatibility."""
    _validate_contract(contract)
    config = contract.actor_config if config is None else config
    _validate_actor(state.params, contract.spec, config)
    _validate_actor(state.previous_params, contract.spec, config)
    physical = np.array(physical_state, copy=True)
    if physical.shape != (17,) or not np.all(np.isfinite(physical)):
        raise ValueError("physical_state must be a finite 17-component body/motor state")
    counters = (
        int(np.asarray(state.library_version)),
        int(np.asarray(state.cumulative_gradient_steps)),
    )
    if min(counters) < 0 or counters[0] != counters[1]:
        raise ValueError("published version and cumulative finite gradient steps must agree")
    template = _initialize_state(
        state.params, state.latest_dynamics_estimate, contract.spec, config
    )
    state_dictionary = serialization.to_state_dict(state)
    serialization.from_state_dict(template, state_dictionary)
    arrays: dict[str, np.ndarray] = {}
    structure = _encode_arrays(
        {
            "state": state_dictionary,
            "reference": _reference_payload(contract),
            "physical_state": physical,
        },
        arrays,
    )
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    manifest = {
        "format": _CHECKPOINT_FORMAT,
        "format_version": _CHECKPOINT_VERSION,
        "state_size": 17,
        "proprioceptive_features": _FEATURE_COUNT,
        "npz_sha256": hashlib.sha256(payload).hexdigest(),
        "reference_sha256": actuator_reference_fingerprint(contract),
        "config": asdict(config),
        "reference_actor_config": asdict(contract.actor_config),
        "reference_learning_config": asdict(contract.learning_config),
        "library_version": counters[0],
        "cumulative_gradient_steps": counters[1],
        "structure": structure,
        "arrays": {
            key: {"shape": list(value.shape), "dtype": value.dtype.str}
            for key, value in arrays.items()
        },
        "metadata": {} if metadata is None else metadata,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    paths = _paths(path_stem)
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite an existing actuator checkpoint")
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        for path, data in zip(paths, (payload, manifest_bytes), strict=True):
            with path.open("xb") as stream:
                created.append(path)
                stream.write(data)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return paths


def load_actuator_learner_checkpoint(
    path_stem: str | Path, *, device: jax.Device | None = None
) -> ActuatorLearnerCheckpoint:
    """Restore exact Adam continuation and immutable reference; reject old state dimensions."""
    npz_path, json_path = _paths(path_stem)
    manifest = json.loads(json_path.read_text())
    if (
        manifest.get("format") != _CHECKPOINT_FORMAT
        or manifest.get("format_version") != _CHECKPOINT_VERSION
        or manifest.get("state_size") != 17
        or manifest.get("proprioceptive_features") != _FEATURE_COUNT
    ):
        raise ValueError("unsupported actuator checkpoint format/version/observation schema")
    payload_bytes = npz_path.read_bytes()
    digest = hashlib.sha256(payload_bytes).hexdigest()
    if digest != manifest["npz_sha256"]:
        raise ValueError("actuator checkpoint checksum mismatch")
    with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as archive:
        if set(archive.files) != set(manifest["arrays"]):
            raise ValueError("checkpoint array keys do not match manifest")
        arrays = {key: archive[key] for key in archive.files}
    for key, value in arrays.items():
        expected = manifest["arrays"][key]
        if list(value.shape) != expected["shape"] or value.dtype.str != expected["dtype"]:
            raise ValueError(f"checkpoint array shape/dtype mismatch: {key}")
        if value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)):
            raise ValueError(f"checkpoint array nonnumeric/nonfinite: {key}")
    payload = _decode_arrays(manifest["structure"], arrays)
    if set(payload) != {"state", "reference", "physical_state"}:
        raise ValueError("unsupported actuator checkpoint payload fields")
    config = _restore_config(manifest["config"])
    reference_actor = _restore_config(manifest["reference_actor_config"])
    reference_learning = dict(manifest["reference_learning_config"])
    reference_learning["trajectory_fractions"] = tuple(reference_learning["trajectory_fractions"])
    reference = payload["reference"]
    contract = ActuatorReferenceContract(
        SkillActorParams(**reference["params"]),
        _restore_model(reference["model"]),
        reference["anchors"],
        SkillLibrarySpec(**reference["spec"]),
        reference_actor,
        ActuatorReferenceConfig(**reference_learning),
    )
    _validate_contract(contract)
    if actuator_reference_fingerprint(contract) != manifest["reference_sha256"]:
        raise ValueError("immutable actuator reference fingerprint mismatch")
    state_dictionary = payload["state"]
    # NamedTuple serialization uses ordinal keys, while the reference manifest uses named fields.
    model_template = contract.model
    saved_model = serialization.from_state_dict(
        model_template, state_dictionary["latest_dynamics_estimate"]
    )
    params = SkillActorParams(**state_dictionary["params"])
    template = _initialize_state(params, saved_model, contract.spec, config)
    restored = serialization.from_state_dict(template, state_dictionary)
    physical = np.array(payload["physical_state"], copy=True)
    if physical.shape != (17,):
        raise ValueError("checkpoint physical state must have 17 components")
    version, steps = (
        int(np.asarray(restored.library_version)),
        int(np.asarray(restored.cumulative_gradient_steps)),
    )
    if (
        version < 0
        or version != steps
        or version != manifest["library_version"]
        or steps != manifest["cumulative_gradient_steps"]
    ):
        raise ValueError("checkpoint learner counters do not match manifest")

    def place(value: Any) -> jax.Array:
        array = np.asarray(value)
        result = jax.device_put(jnp.asarray(array), device)
        if np.dtype(result.dtype) != array.dtype:
            raise ValueError("checkpoint dtype cannot be restored exactly; check JAX x64 settings")
        return result

    # Physical plant snapshots retain their stored NumPy dtype independently of JAX x64 mode.
    # A caller explicitly casts an observation when passing it to a controller or learner.
    physical.setflags(write=False)
    restored, teacher_params, teacher_model, anchors, spec = jax.tree.map(
        place, (restored, contract.params, contract.model, contract.anchors, contract.spec)
    )
    contract = replace(
        contract, params=teacher_params, model=teacher_model, anchors=anchors, spec=spec
    )
    return ActuatorLearnerCheckpoint(
        restored,
        contract,
        config,
        physical,
        manifest.get("metadata", {}),
        npz_path,
        json_path,
        digest,
    )


@dataclass(frozen=True, slots=True)
class ActuatorCompetenceThresholds:
    """Obstacle-free diagnostics; measured competence never gates online publication."""

    maximum_position_tracking_rmse_m: float = 0.30
    maximum_velocity_tracking_rmse_mps: float = 0.60
    maximum_terminal_speed_mps: float = 0.80
    maximum_tilt_radians: float = 0.90
    maximum_angular_rate_rps: float = 12.0
    minimum_pairwise_trajectory_spread_m: float = 0.08

    def validate(self) -> None:
        if not all(math.isfinite(value) and value > 0 for value in asdict(self).values()):
            raise ValueError("competence thresholds must be positive finite")


def probe_actuator_library(
    params: SkillActorParams,
    contract: ActuatorReferenceContract,
    model: ActuatorModel,
    *,
    states: jax.Array | None = None,
    labels: tuple[str, ...] | None = None,
    config: ActuatorSkillConfig | None = None,
    thresholds: ActuatorCompetenceThresholds = ActuatorCompetenceThresholds(),
) -> dict[str, Any]:
    """Measure each moving/motor state and skill; never require impossible isotropic coverage."""
    _validate_contract(contract)
    validate_actuator_model(model)
    thresholds.validate()
    config = contract.actor_config if config is None else config
    states = contract.anchors if states is None else states
    if states.ndim != 2 or states.shape[-1] != 17 or not len(states):
        raise ValueError("probe states must have shape (B,17), B >= 1")
    labels = tuple(f"state_{index}" for index in range(len(states))) if labels is None else labels
    if len(labels) != len(states):
        raise ValueError("probe labels must match the state bank")
    begin = time.perf_counter()
    rollout = jax.jit(
        jax.vmap(lambda y: rollout_actuator_skill_library(params, contract.spec, y, model, config))
    )(states)
    references = jax.jit(
        jax.vmap(
            lambda y: (
                rollout_actuator_skill_library(
                    contract.params, contract.spec, y, contract.model, contract.actor_config
                ).states
            )
        )
    )(states)
    jax.block_until_ready((rollout, references))
    elapsed = time.perf_counter() - begin
    actual, reference = np.asarray(rollout.states), np.asarray(references)
    nodes = sorted(
        {max(1, round(config.horizon * f)) for f in contract.learning_config.trajectory_fractions}
    )
    position = np.sqrt(
        np.mean((actual[:, :, nodes, :3] - reference[:, :, nodes, :3]) ** 2, axis=(-1, -2))
    )
    velocity = np.sqrt(
        np.mean((actual[:, :, nodes, 7:10] - reference[:, :, nodes, 7:10]) ** 2, axis=(-1, -2))
    )
    terminal = np.linalg.norm(actual[:, :, -1, 7:10], axis=-1)
    tilt = np.arccos(np.clip(1.0 - 2.0 * (actual[..., 3] ** 2 + actual[..., 4] ** 2), -1.0, 1.0))
    rate = np.linalg.norm(actual[..., 10:13], axis=-1)
    valid = np.all(np.asarray(rollout.policy_valid), axis=-1) & np.all(
        np.isfinite(actual), axis=(-1, -2)
    )
    requested = np.asarray(rollout.requested_commands)
    saturation = np.mean(
        (requested < np.asarray(model.command_lower))
        | (requested > np.asarray(model.command_upper)),
        axis=(-1, -2),
    )
    checks = {
        "all_rollouts_finite_and_valid": bool(np.all(valid)),
        "same_state_position_tracking": bool(
            np.all(position <= thresholds.maximum_position_tracking_rmse_m)
        ),
        "same_state_velocity_tracking": bool(
            np.all(velocity <= thresholds.maximum_velocity_tracking_rmse_mps)
        ),
        "terminal_braking": bool(np.all(terminal <= thresholds.maximum_terminal_speed_mps)),
        "tilt": bool(np.all(tilt <= thresholds.maximum_tilt_radians)),
        "angular_rate": bool(np.all(rate <= thresholds.maximum_angular_rate_rps)),
    }
    relative = actual[..., :3] - actual[:, :, :1, :3]
    pairwise = np.sqrt(
        np.mean(np.sum((relative[:, :, None] - relative[:, None, :]) ** 2, axis=-1), axis=-1)
    )
    pair_mask = ~np.eye(actual.shape[1], dtype=bool)
    spread = (
        np.mean(pairwise[:, pair_mask], axis=-1) if np.any(pair_mask) else np.zeros(len(states))
    )
    checks["nontrivial_repertoire_spread"] = bool(
        actual.shape[1] == 1 or np.all(spread >= thresholds.minimum_pairwise_trajectory_spread_m)
    )
    scaffold_targets = np.asarray(contract.spec.target_descriptors[:, :3])
    scaffold_error = np.sqrt(np.mean((relative[:, :, -1] - scaffold_targets) ** 2, axis=-1))
    effort_integral = config.dt * np.sum(actual[:, :, 1:, 13:17] ** 2, axis=(-1, -2))
    per_state = []
    for index, label in enumerate(labels):
        per_state.append(
            {
                "label": label,
                "initial_state": np.asarray(states[index]).tolist(),
                "prefix_displacement_m": relative[index][:, nodes].tolist(),
                "scaffold_displacement_target_m": scaffold_targets.tolist(),
                "scaffold_displacement_rmse_m": scaffold_error[index].tolist(),
                "motor_effort_squared_integral_n2_s": effort_integral[index].tolist(),
                "position_tracking_rmse_m": position[index].tolist(),
                "velocity_tracking_rmse_mps": velocity[index].tolist(),
                "terminal_speed_mps": terminal[index].tolist(),
                "maximum_tilt_radians": np.max(tilt[index], axis=-1).tolist(),
                "maximum_angular_rate_rps": np.max(rate[index], axis=-1).tolist(),
                "command_saturation_fraction": saturation[index].tolist(),
                "trajectory_pairwise_rms_mean_m": float(spread[index]),
                "valid": valid[index].tolist(),
            }
        )
    return {
        "scope": "obstacle-free state-conditioned body motion; no obstacle safety guarantee",
        "tracking_interpretation": (
            "teacher tracking is a recovery diagnostic and vanishes for the unchanged nominal "
            "teacher; absolute braking, attitude/rate and trajectory spread are checked "
            "separately. "
            "Scaffold displacement error is descriptive because moving-state targets can differ."
        ),
        "reference_sha256": actuator_reference_fingerprint(contract),
        "policy_count": int(actual.shape[1]),
        "state_count": int(len(states)),
        "prefix_times_s": [node * config.dt for node in nodes],
        "thresholds": asdict(thresholds),
        "competence_checks": checks,
        "competent_under_declared_criteria": all(checks.values()),
        "maximum_position_tracking_rmse_m": float(np.max(position)),
        "maximum_velocity_tracking_rmse_mps": float(np.max(velocity)),
        "maximum_terminal_speed_mps": float(np.max(terminal)),
        "maximum_tilt_radians": float(np.max(tilt)),
        "maximum_angular_rate_rps": float(np.max(rate)),
        "mean_command_saturation_fraction": float(np.mean(saturation)),
        "probe_wall_seconds_including_compile": elapsed,
        "per_state": per_state,
    }


def actuator_loss_gradient_contributions(
    learner: ActuatorLearnerFunctions,
    state: ActuatorLearnerState,
    initial_state: jax.Array,
    model: ActuatorModel,
) -> dict[str, Any]:
    """Raw per-term gradient norms and alignment; diagnostic only, never an acceptance gate."""
    sample = learner.loss(
        state.params, initial_state, model, state.previous_params, state.library_version
    )[1]
    names = tuple(
        name
        for name in sample._fields
        if name not in {"total", "rollout_valid_fraction"} and getattr(sample, name).ndim == 0
    )

    def components(params: SkillActorParams) -> jax.Array:
        _, metrics = learner.loss(
            params, initial_state, model, state.previous_params, state.library_version
        )
        return jnp.stack((metrics.total, *(getattr(metrics, name) for name in names)))

    begin = time.perf_counter()
    values = components(state.params)
    gradients = jax.jit(jax.jacrev(components))(state.params)
    jax.block_until_ready((values, gradients))
    leaves = [
        np.asarray(value).reshape((len(names) + 1, -1)) for value in jax.tree.leaves(gradients)
    ]
    flattened = np.concatenate(leaves, axis=1)
    norms = np.linalg.norm(flattened, axis=1)
    entries = {}
    for index, name in enumerate(("total", *names)):
        denominator = norms[index] * norms[0]
        entries[name] = {
            "value": float(values[index]),
            "gradient_norm": float(norms[index]),
            "cosine_with_total": float(np.dot(flattened[index], flattened[0]) / denominator)
            if denominator > 0
            else None,
        }
    return {
        "gradient_components": entries,
        "wall_seconds_including_compile": time.perf_counter() - begin,
        "scope": "raw component gradients; total includes configured weights",
    }


@dataclass(frozen=True, slots=True)
class ActuatorDynamicsSupport:
    """Declared independent per-motor support for the separately trained DR comparator."""

    effectiveness_range: tuple[float, float] = (0.65, 1.0)
    time_constant_range_s: tuple[float, float] = (0.01, 0.08)

    def validate(self) -> None:
        for values in (self.effectiveness_range, self.time_constant_range_s):
            if (
                len(values) != 2
                or not all(math.isfinite(v) and v > 0 for v in values)
                or values[0] > values[1]
            ):
                raise ValueError("DR support must contain ordered positive finite intervals")
        if self.effectiveness_range[1] > 1:
            raise ValueError("effectiveness support must not exceed nominal effectiveness one")


def sample_actuator_training_models(
    nominal_model: ActuatorModel,
    *,
    seed: int,
    count: int,
    support: ActuatorDynamicsSupport = ActuatorDynamicsSupport(),
) -> tuple[ActuatorModel, ...]:
    """Deterministic documented development dynamics, with unchanged physical command limits."""
    support.validate()
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    rng = np.random.default_rng(seed)
    dtype = nominal_model.effectiveness.dtype
    models = []
    for _ in range(count):
        models.append(
            nominal_model._replace(
                effectiveness=jnp.asarray(
                    rng.uniform(*support.effectiveness_range, size=4), dtype=dtype
                ),
                time_constants=jnp.asarray(
                    rng.uniform(*support.time_constant_range_s, size=4), dtype=dtype
                ),
            )
        )
    return tuple(models)


def train_actuator_library(
    contract: ActuatorReferenceContract,
    *,
    seed: int,
    steps: int,
    models: tuple[ActuatorModel, ...] | None = None,
    initial_params: SkillActorParams | None = None,
    config: ActuatorSkillConfig | None = None,
) -> tuple[ActuatorLearnerState, dict[str, Any]]:
    """Bounded offline preparation on the full proprioceptive bank with measured work.

    This routine performs exactly the requested gradient evaluations. It does not select a
    checkpoint by test performance or declare competence. Call ``probe_actuator_library`` on
    development and validation state banks, then explicitly choose the frozen reference.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    config = contract.actor_config if config is None else config
    models = (contract.model,) if models is None else models
    if not models:
        raise ValueError("at least one training model is required")
    for model in models:
        validate_actuator_model(model)
    prepare_begin = time.perf_counter()
    learner = build_actuator_skill_learner(contract, config)
    params = (
        initialize_actuator_skill_actor(jax.random.key(seed), contract.spec, config)
        if initial_params is None
        else initial_params
    )
    state = learner.initialize(params, models[0])
    preparation_seconds = time.perf_counter() - prepare_begin
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(contract.anchors))
    trace = []
    durations = []
    for iteration in range(steps):
        if iteration and iteration % len(order) == 0:
            order = rng.permutation(len(contract.anchors))
        state_index = int(order[iteration % len(order)])
        model_index = int(rng.integers(len(models)))
        begin = time.perf_counter()
        state, metrics = learner.step(state, contract.anchors[state_index], models[model_index])
        jax.block_until_ready((state, metrics))
        elapsed = time.perf_counter() - begin
        durations.append(elapsed)
        trace.append(
            {
                "gradient_evaluation": iteration + 1,
                "state_index": state_index,
                "model_index": model_index,
                "finite_update_published": bool(metrics.finite_update_applied),
                "library_version": int(state.library_version),
                "loss_before_update": float(metrics.loss.total),
                "per_skill_position_squared_error_normalized": np.asarray(
                    metrics.loss.per_skill_position_error
                ).tolist(),
                "per_skill_velocity_squared_error_normalized": np.asarray(
                    metrics.loss.per_skill_velocity_error
                ).tolist(),
                "gradient_norm": float(metrics.gradient_norm),
                "parameter_update_norm": float(metrics.parameter_update_norm),
                "wall_seconds": elapsed,
            }
        )
    count = contract.spec.latent_codes.shape[0]
    return state, {
        "seed": seed,
        "reference_sha256": actuator_reference_fingerprint(contract),
        "gradient_evaluations": steps,
        "finite_updates_published": int(state.library_version),
        "state_bank_count": len(contract.anchors),
        "training_model_count": len(models),
        "training_dynamics": [
            {
                "effectiveness": np.asarray(m.effectiveness).tolist(),
                "time_constants_s": np.asarray(m.time_constants).tolist(),
            }
            for m in models
        ],
        "student_integration_steps": steps
        * (1 + contract.learning_config.anchor_batch_size)
        * count
        * config.horizon,
        "current_teacher_integration_steps": steps * count * config.horizon,
        "cached_anchor_teacher_integration_steps": len(contract.anchors) * count * config.horizon,
        "body_rk_stages_per_integration_step": 4 * config.plant_substeps,
        "preparation_seconds_including_anchor_compile": preparation_seconds,
        "training_seconds_including_first_step_compile": sum(durations),
        "first_step_seconds_including_compile": durations[0] if durations else None,
        "steady_step_mean_seconds": float(np.mean(durations[1:])) if len(durations) > 1 else None,
        "trace": trace,
        "competence_status": "requires separately measured development and validation probes",
    }


def prepare_actuator_seed_libraries(
    spec: SkillLibrarySpec,
    model: ActuatorModel,
    config: ActuatorSkillConfig,
    *,
    seeds: tuple[int, ...] = (11, 23, 37),
    steps_per_seed: int = 0,
    learning_config: ActuatorReferenceConfig = ActuatorReferenceConfig(),
) -> dict[int, tuple[ActuatorReferenceContract, ActuatorLearnerState, dict[str, Any]]]:
    """Prepare at least three independent libraries; measure competence before using baselines.

    The initial structured actor defines an immutable obstacle-free bootstrap reference. Once
    the requested nominal preparation finishes, each seed's resulting parameters define the
    immutable nominal teacher for subsequent DR/adaptive experiments. Adam history is retained
    so primary frozen/adaptive copies can start from exactly the same continuation state.
    """
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("main study preparation requires at least three distinct library seeds")
    anchors, labels = actuator_proprioceptive_state_bank(model, dtype=spec.latent_codes.dtype)
    prepared = {}
    for seed in seeds:
        params = initialize_actuator_skill_actor(jax.random.key(seed), spec, config)
        initial_contract = ActuatorReferenceContract(
            params, model, anchors, spec, config, learning_config
        )
        state, report = train_actuator_library(
            initial_contract, seed=seed, steps=steps_per_seed, initial_params=params
        )
        contract = replace(
            initial_contract,
            params=state.params,
            learning_config=replace(learning_config, reference_braking_excess=True),
        )
        report.update(
            {
                "state_bank_labels": labels,
                "bootstrap_reference_sha256": actuator_reference_fingerprint(initial_contract),
                "deployment_reference_sha256": actuator_reference_fingerprint(contract),
                "library_kind": "nominal_prepared",
            }
        )
        prepared[seed] = contract, state, report
    return prepared


__all__ = [
    "ActuatorCompetenceThresholds",
    "ActuatorDynamicsSupport",
    "ActuatorLearnerCheckpoint",
    "ActuatorLearnerFunctions",
    "ActuatorLearnerState",
    "ActuatorLossMetrics",
    "ActuatorReferenceConfig",
    "ActuatorReferenceContract",
    "ActuatorSkillConfig",
    "ActuatorSkillRollout",
    "ActuatorStepMetrics",
    "acceleration_to_actuator_command",
    "actuator_loss_gradient_contributions",
    "actuator_proprioceptive_state_bank",
    "actuator_reference_fingerprint",
    "actuator_reference_loss",
    "actuator_skill_actions",
    "build_actuator_skill_learner",
    "build_single_recovery_spec",
    "initialize_actuator_skill_actor",
    "load_actuator_learner_checkpoint",
    "prepare_actuator_seed_libraries",
    "probe_actuator_library",
    "rollout_actuator_skill_library",
    "sample_actuator_training_models",
    "save_actuator_learner_checkpoint",
    "train_actuator_library",
]
