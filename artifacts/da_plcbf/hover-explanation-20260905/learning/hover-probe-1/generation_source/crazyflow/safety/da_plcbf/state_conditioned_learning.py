"""Persistent obstacle-free adaptation to immutable nominal-model maneuver references.

Targets depend only on the initial proprioceptive state, fixed skill identity, a saved teacher,
and a fixed nominal model. They never use obstacles, goals, runtime certificates, or the current
model's frozen rollout as a target. A small rotating anchor batch regularizes behavior retention;
the persisted library version determines its indices, so checkpoint resume needs no hidden RNG.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.nn import softplus

from crazyflow.safety.da_plcbf.bptt import tree_all_finite
from crazyflow.safety.da_plcbf.direct_wrench import quaternion_to_rotation_matrix
from crazyflow.safety.da_plcbf.learner_checkpoint import LearnerCheckpoint, load_learner_checkpoint
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    PersistentSkillFunctions,
    SkillActorParams,
    SkillLibrarySpec,
    SkillLossMetrics,
    _parameter_distance,
    _trainable_skill_tree,
    build_persistent_skill_learner,
    rollout_skill_library,
    spatial_descriptor_losses,
)
from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


@dataclass(frozen=True, slots=True)
class ReferenceLearningConfig:
    """Static reference-tracking and obstacle-free retention settings."""

    anchor_batch_size: int = 2
    trajectory_weight: float = 5.0
    velocity_weight: float = 1.0
    retention_weight: float = 5.0
    reference_braking_excess: bool = True
    trajectory_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

    def validate(self) -> None:
        if (
            isinstance(self.anchor_batch_size, bool)
            or not isinstance(self.anchor_batch_size, int)
            or self.anchor_batch_size < 0
        ):
            raise ValueError("anchor_batch_size must be a nonnegative integer")
        if not all(
            math.isfinite(value) and value >= 0
            for value in (self.trajectory_weight, self.velocity_weight, self.retention_weight)
        ):
            raise ValueError("reference weights must be finite and nonnegative")
        if not self.trajectory_fractions or not all(
            math.isfinite(x) and 0 < x <= 1 for x in self.trajectory_fractions
        ):
            raise ValueError("trajectory fractions must lie in (0, 1]")
        if tuple(sorted(set(self.trajectory_fractions))) != self.trajectory_fractions:
            raise ValueError("trajectory fractions must be strictly increasing")
        if not isinstance(self.reference_braking_excess, bool):
            raise TypeError("reference_braking_excess must be boolean")


@dataclass(frozen=True, slots=True)
class ReferenceContract:
    """Immutable teacher/model/anchor inputs that accompany a full persistent checkpoint."""

    params: SkillActorParams
    model: VersionAModel
    anchors: jax.Array
    actor_config: PersistentSkillConfig
    learning_config: ReferenceLearningConfig
    spec: SkillLibrarySpec
    actuator: VersionAActuator


def proprioceptive_state_bank(*, dtype: Any = jnp.float32) -> tuple[jax.Array, tuple[str, ...]]:
    """Return explicit rest, signed velocity, attitude, and rate states with no scene metadata."""
    values = []
    labels = []

    def append(
        label: str,
        velocity: tuple[float, float, float],
        pitch: float = 0.0,
        roll: float = 0.0,
        rate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        # Compose roll then pitch in the scalar-last convention.
        sr, cr = math.sin(roll / 2), math.cos(roll / 2)
        sp, cp = math.sin(pitch / 2), math.cos(pitch / 2)
        values.append((0.0, 0.0, 1.4, sr * cp, cr * sp, -sr * sp, cr * cp, *velocity, *rate))
        labels.append(label)

    append("rest", (0.0, 0.0, 0.0))
    for speed in (0.5, 1.5, 2.06):
        for sign in (-1, 1):
            append(f"vx_{sign * speed:+.2f}", (sign * speed, 0.0, 0.0))
    for sign in (-1, 1):
        append(f"vy_{sign:+d}", (0.0, float(sign), 0.0))
        append(f"vz_{0.5 * sign:+.1f}", (0.0, 0.0, 0.5 * sign))
        append(f"pitch_{0.2 * sign:+.1f}", (0.5 * sign, 0.0, 0.0), pitch=0.2 * sign)
        append(
            f"roll_rate_{sign:+d}",
            (0.0, 0.5 * sign, 0.0),
            roll=0.15 * sign,
            rate=(0.5 * sign, 0.0, 0.0),
        )
    return jnp.asarray(values, dtype=dtype), tuple(labels)


def _validate_contract(contract: ReferenceContract) -> None:
    contract.actor_config.validate()
    contract.learning_config.validate()
    if contract.anchors.ndim != 2 or contract.anchors.shape[-1] != 13 or not len(contract.anchors):
        raise ValueError("reference anchors must have shape (B,13), B >= 1")
    if contract.learning_config.anchor_batch_size > len(contract.anchors):
        raise ValueError("anchor batch cannot exceed the fixed bank")
    if not all(
        np.all(np.isfinite(np.asarray(x)))
        for x in jax.tree.leaves(
            (contract.params, contract.model, contract.anchors, contract.spec, contract.actuator)
        )
    ):
        raise ValueError("reference contract must contain finite numeric inputs")


def reference_trajectory_loss(
    params: SkillActorParams,
    initial_state: jax.Array,
    point_model: VersionAModel,
    previous_params: SkillActorParams,
    iteration: jax.Array,
    *,
    contract: ReferenceContract,
    config: PersistentSkillConfig,
    anchor_reference_states: jax.Array,
) -> tuple[jax.Array, SkillLossMetrics]:
    """Track a same-state nominal teacher and retain its maneuvers on rotating fixed anchors."""
    settings = contract.learning_config
    count = settings.anchor_batch_size
    indices = (iteration * max(count, 1) + jnp.arange(count)) % contract.anchors.shape[0]
    initial_states = jnp.concatenate((initial_state[None], contract.anchors[indices]), axis=0)
    actual = jax.vmap(
        lambda state: rollout_skill_library(
            params, contract.spec, state, point_model, contract.actuator, config
        )
    )(initial_states)
    current_reference = rollout_skill_library(
        contract.params,
        contract.spec,
        initial_state,
        contract.model,
        contract.actuator,
        contract.actor_config,
    )
    references = jax.lax.stop_gradient(
        jnp.concatenate((current_reference.states[None], anchor_reference_states[indices]), axis=0)
    )
    nodes = jnp.asarray(
        [max(1, round(config.horizon * f)) for f in settings.trajectory_fractions], dtype=jnp.int32
    )
    relative = actual.states[..., :3] - initial_states[:, None, None, :3]
    reference_relative = references[..., :3] - initial_states[:, None, None, :3]
    position_error = (
        relative[:, :, nodes] - reference_relative[:, :, nodes]
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
    spatial = spatial_descriptor_losses(
        actual.descriptors[0], current_reference.descriptors, config
    )
    terminal_squared = jnp.sum(actual.states[:, :, -1, 7:10] ** 2, axis=-1)
    if settings.reference_braking_excess:
        terminal_squared = jax.nn.relu(
            terminal_squared - jnp.sum(references[:, :, -1, 7:10] ** 2, axis=-1)
        )
    braking = jnp.mean(terminal_squared) / (3 * config.velocity_scale**2)
    rotation = quaternion_to_rotation_matrix(actual.states[..., 3:7])
    attitude = jnp.mean(1.0 - rotation[..., 2, 2])
    angular_rate = jnp.mean((actual.states[..., 10:13] / config.angular_velocity_scale) ** 2)
    # Penalize behavioral acceleration, not the force needed to cancel known wind/gravity bias.
    behavior = actual.behavior_accelerations
    action = jnp.mean((behavior / config.acceleration_limit) ** 2)
    action_rate = (
        jnp.mean((jnp.diff(behavior, axis=2) / config.acceleration_limit) ** 2)
        if config.horizon > 1
        else jnp.asarray(0.0)
    )
    lower = jnp.broadcast_to(contract.actuator.thrust_min, (4,))
    upper = jnp.broadcast_to(contract.actuator.thrust_max, (4,))
    coordinate = (actual.raw_motor_forces - (lower + upper) / 2) / ((upper - lower) / 2)
    excess = softplus((jnp.abs(coordinate) - 1) / config.saturation_temperature)
    saturation = jnp.mean((excess * config.saturation_temperature) ** 2)
    trust = _parameter_distance(params, previous_params)
    total = (
        config.target_weight * spatial.target
        + config.diversity_weight * spatial.diversity
        + config.pairwise_weight * spatial.pairwise
        + config.terminal_braking_weight * braking
        + config.attitude_weight * attitude
        + config.angular_rate_weight * angular_rate
        + config.action_weight * action
        + config.action_rate_weight * action_rate
        + config.saturation_weight * saturation
        + config.trust_weight * trust
        + settings.trajectory_weight * trajectory
        + settings.velocity_weight * velocity
        + settings.retention_weight * retention
    )
    valid = jnp.all(actual.policy_valid) & tree_all_finite(
        (actual.states, references, params, previous_params, point_model)
    )
    total = jnp.where(valid & jnp.isfinite(total), total, jnp.inf)
    metrics = SkillLossMetrics(
        total,
        spatial.target,
        spatial.diversity,
        spatial.pairwise,
        action,
        action_rate,
        saturation,
        trust,
        jnp.mean(actual.policy_valid),
        actual.descriptors[0],
        braking,
        attitude,
        angular_rate,
        trajectory,
        velocity,
        retention,
    )
    return total, metrics


def build_reference_skill_learner(
    contract: ReferenceContract,
    config: PersistentSkillConfig | None = None,
    *,
    device: jax.Device | None = None,
) -> PersistentSkillFunctions:
    """Build the standard persistent learner interface with immutable reference targets."""
    _validate_contract(contract)
    config = contract.actor_config if config is None else config
    for name in (
        "dt",
        "horizon",
        "control_interval_steps",
        "model_compensation",
        "gate_residual_with_skill_duration",
    ):
        if getattr(config, name) != getattr(contract.actor_config, name):
            raise ValueError(f"reference and adapted actor must preserve {name}")
    anchor_reference = jax.jit(
        jax.vmap(
            lambda state: (
                rollout_skill_library(
                    contract.params,
                    contract.spec,
                    state,
                    contract.model,
                    contract.actuator,
                    contract.actor_config,
                ).states
            )
        )
    )(contract.anchors)
    jax.block_until_ready(anchor_reference)

    def objective(
        params: SkillActorParams,
        state: jax.Array,
        model: VersionAModel,
        previous: SkillActorParams,
        iteration: jax.Array,
    ) -> tuple[jax.Array, SkillLossMetrics]:
        return reference_trajectory_loss(
            params,
            state,
            model,
            previous,
            iteration,
            contract=contract,
            config=config,
            anchor_reference_states=anchor_reference,
        )

    return build_persistent_skill_learner(
        contract.spec, contract.actuator, config, device=device, loss_function=objective
    )


def save_reference_contract(
    contract: ReferenceContract, path_stem: str | Path
) -> tuple[Path, Path]:
    """Save immutable nominal targets and their explicit proprioceptive bank without pickle."""
    _validate_contract(contract)
    stem = Path(path_stem)
    arrays = {"anchors": np.asarray(contract.anchors)}
    for group, values in (
        ("params", contract.params),
        ("model", contract.model),
        ("spec", contract.spec),
        ("actuator", contract.actuator),
    ):
        names = values._fields if hasattr(values, "_fields") else values.__dataclass_fields__
        for name in names:
            arrays[f"{group}_{name}"] = np.asarray(getattr(values, name))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    manifest = {
        "format": "crazyflow.nominal_reference_contract.v1",
        "npz_sha256": hashlib.sha256(payload).hexdigest(),
        "actor_config": asdict(contract.actor_config),
        "learning_config": asdict(contract.learning_config),
        "scope": "fixed nominal model/teacher and proprioceptive anchors; no scene or goal inputs",
    }
    paths = Path(f"{stem}.npz"), Path(f"{stem}.json")
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite reference contract")
    stem.parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open("xb") as stream:
        stream.write(payload)
    with paths[1].open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return paths


def load_reference_contract(
    path_stem: str | Path, *, device: jax.Device | None = None
) -> ReferenceContract:
    """Restore a checksummed fixed-reference contract alongside the persistent optimizer state."""
    stem = Path(path_stem)
    metadata = json.loads(Path(f"{stem}.json").read_text())
    if metadata.get("format") != "crazyflow.nominal_reference_contract.v1":
        raise ValueError("unsupported reference contract format")
    payload = Path(f"{stem}.npz").read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata["npz_sha256"]:
        raise ValueError("reference contract checksum mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: jax.device_put(archive[name], device) for name in archive.files}

    def restore(prefix: str, cls: Any) -> Any:
        return cls(
            **{
                name.removeprefix(f"{prefix}_"): value
                for name, value in arrays.items()
                if name.startswith(f"{prefix}_")
            }
        )

    actor = dict(metadata["actor_config"])
    actor["descriptor_scales"] = tuple(actor["descriptor_scales"])
    learning = dict(metadata["learning_config"])
    learning["trajectory_fractions"] = tuple(learning["trajectory_fractions"])
    contract = ReferenceContract(
        restore("params", SkillActorParams),
        restore("model", VersionAModel),
        arrays["anchors"],
        PersistentSkillConfig(**actor),
        ReferenceLearningConfig(**learning),
        restore("spec", SkillLibrarySpec),
        restore("actuator", VersionAActuator),
    )
    _validate_contract(contract)
    return contract


def reference_contract_checkpoint_metadata(path_stem: str | Path) -> dict[str, str]:
    """Bind a checkpoint to both numeric targets and the exact configuration manifest."""
    stem = Path(path_stem)
    digest = hashlib.sha256(Path(f"{stem}.npz").read_bytes()).hexdigest()
    manifest_payload = Path(f"{stem}.json").read_bytes()
    manifest = json.loads(manifest_payload)
    if manifest.get("npz_sha256") != digest:
        raise ValueError("reference contract checksum mismatch")
    return {
        "reference_contract_npz_sha256": digest,
        "reference_contract_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }


def build_reference_skill_learner_from_checkpoint(
    checkpoint_stem: str | Path,
    contract_stem: str | Path | None = None,
    *,
    device: jax.Device | None = None,
) -> tuple[LearnerCheckpoint, ReferenceContract, PersistentSkillFunctions]:
    """Load full Adam continuation plus its sibling nominal_reference contract for deployment."""
    checkpoint = load_learner_checkpoint(checkpoint_stem, device=device)
    stem = (
        Path(checkpoint_stem).parent / "nominal_reference"
        if contract_stem is None
        else contract_stem
    )
    contract = load_reference_contract(stem, device=device)
    expected_hash = checkpoint.metadata.get("reference_contract_npz_sha256")
    expected_manifest = checkpoint.metadata.get("reference_contract_manifest_sha256")
    actual_binding = reference_contract_checkpoint_metadata(stem)
    for key, expected in (
        ("reference_contract_npz_sha256", expected_hash),
        ("reference_contract_manifest_sha256", expected_manifest),
    ):
        if expected is not None and expected != actual_binding[key]:
            raise ValueError(f"reference contract does not match checkpoint reference hash ({key})")
    binding = "legacy_unbound"
    if expected_hash is not None and expected_manifest is not None:
        binding = "verified_npz_and_manifest_sha256"
    elif expected_hash is not None or expected_manifest is not None:
        binding = "partial_npz_only" if expected_hash is not None else "partial_manifest_only"
    checkpoint = replace(
        checkpoint, metadata={**checkpoint.metadata, "reference_contract_binding": binding}
    )
    for expected, actual in zip(
        jax.tree.leaves((contract.spec, contract.actuator)),
        jax.tree.leaves((checkpoint.spec, checkpoint.actuator)),
        strict=True,
    ):
        if not np.array_equal(np.asarray(expected), np.asarray(actual)):
            raise ValueError("reference contract does not match checkpoint spec/actuator")
    learner = build_reference_skill_learner(contract, checkpoint.config, device=device)
    return checkpoint, contract, learner


def loss_gradient_contributions(
    learner: PersistentSkillFunctions,
    params: SkillActorParams,
    state: jax.Array,
    model: VersionAModel,
    previous: SkillActorParams,
    *,
    component_weights: dict[str, float] | None = None,
    iteration: jax.Array | None = None,
    trainable_config: PersistentSkillConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Measure raw component gradient magnitudes/alignment; not an update acceptance rule."""
    fields = (
        "descriptor_target",
        "terminal_braking",
        "diversity",
        "pairwise",
        "action",
        "action_rate",
        "saturation",
        "trust",
        "trajectory_tracking",
        "velocity_tracking",
        "reference_retention",
    )

    def losses(candidate: SkillActorParams) -> tuple[jax.Array, SkillLossMetrics]:
        return learner.loss(candidate, state, model, previous, iteration)

    total_gradient = jax.grad(lambda p: losses(p)[0])(params)
    total_norm = float(optax.tree.norm(total_gradient))
    trainable_total = (
        _trainable_skill_tree(total_gradient, trainable_config)
        if trainable_config is not None
        else total_gradient
    )
    trainable_total_norm = float(optax.tree.norm(trainable_total))
    values = losses(params)[1]
    result = {}
    for name in fields:
        gradient = jax.grad(lambda p: getattr(losses(p)[1], name))(params)
        norm = float(optax.tree.norm(gradient))
        dot = sum(
            float(jnp.vdot(a, b))
            for a, b in zip(jax.tree.leaves(gradient), jax.tree.leaves(total_gradient), strict=True)
        )
        weight = (component_weights or {}).get(name, 1.0)
        trainable = (
            _trainable_skill_tree(gradient, trainable_config)
            if trainable_config is not None
            else gradient
        )
        trainable_norm = float(optax.tree.norm(trainable))
        trainable_dot = sum(
            float(jnp.vdot(a, b))
            for a, b in zip(
                jax.tree.leaves(trainable), jax.tree.leaves(trainable_total), strict=True
            )
        )
        result[name] = {
            "raw_loss": float(getattr(values, name)),
            "raw_gradient_norm": norm,
            "weight": weight,
            "weighted_gradient_norm": abs(weight) * norm,
            "cosine_with_total_gradient": dot / max(norm * total_norm, 1e-30),
            "trainable_gradient_norm": trainable_norm,
            "weighted_trainable_gradient_norm": abs(weight) * trainable_norm,
            "cosine_with_trainable_total_gradient": trainable_dot
            / max(trainable_norm * trainable_total_norm, 1e-30),
        }
    result["total"] = {
        "raw_loss": float(values.total),
        "raw_gradient_norm": total_norm,
        "trainable_gradient_norm": trainable_total_norm,
        "cosine_with_total_gradient": 1.0,
    }
    return result
