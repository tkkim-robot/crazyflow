"""Handcrafted maneuver library for matched actuator adaptation diagnostics.

These primitives use directional velocity feedback and the common attitude/rate PD
controller. Their neural residual is exactly zero. Online learning can change only
the bounded desired-velocity offsets and maneuver durations; the frozen comparator
starts from the same complete checkpoint and uses the same F2 actuator adapter.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.actuator_learning import (
    ActuatorReferenceConfig,
    ActuatorReferenceContract,
    ActuatorSkillConfig,
    actuator_proprioceptive_state_bank,
    initialize_actuator_skill_actor,
)
from crazyflow.safety.da_plcbf.persistent_skill_learner import SkillLibrarySpec

PD_PRIMITIVE_NAMES = (
    "hover_brake",
    "east",
    "west",
    "north",
    "south",
    "up",
    "down",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
    "east_up",
    "west_up",
    "north_up",
    "south_up",
    "east_down",
)


def handcrafted_pd_spec(
    *,
    speed: float = 1.0,
    duration: float = 0.45,
    horizon_seconds: float = 1.2,
    dtype: Any = jnp.float32,
) -> SkillLibrarySpec:
    """Sixteen declared feedback primitives, independent of obstacles and task goals.

    Hover brakes initial velocity; the remaining fifteen primitives first move in
    their declared direction and then brake within the shared prediction horizon.
    """
    if not all(math.isfinite(v) and v > 0 for v in (speed, duration, horizon_seconds)):
        raise ValueError("PD maneuver scales must be finite and positive")
    if duration > horizon_seconds:
        raise ValueError("maneuver duration must not exceed the prediction horizon")
    directions = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [1, 0, 1],
            [-1, 0, 1],
            [0, 1, 1],
            [0, -1, 1],
            [1, 0, -1],
        ],
        dtype=float,
    )
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1)
    velocities = speed * directions
    durations = np.full(16, duration)
    displacement = velocities * durations[:, None]
    targets = np.concatenate(
        (displacement, displacement / horizon_seconds, np.zeros_like(displacement)), axis=1
    )
    return SkillLibrarySpec(
        jnp.eye(16, dtype=dtype),
        jnp.asarray(velocities, dtype=dtype),
        jnp.asarray(durations, dtype=dtype),
        jnp.asarray(targets, dtype=dtype),
    )


def handcrafted_pd_contract(
    model: Any,
    *,
    policy_gain: float = 2.4,
    duration: float = 0.45,
    speed: float = 1.0,
    objective: ActuatorReferenceConfig | None = None,
) -> ActuatorReferenceContract:
    """Build an untrained teacher and a 64-coordinate maneuver adaptation contract."""
    config = ActuatorSkillConfig(
        horizon=60,
        control_interval_steps=2,
        hidden_width=1,
        policy_gain=policy_gain,
        residual_scale=0.0,
        initial_residual_scale=0.0,
        trainable_parameters="offsets",
        learn_durations=True,
        velocity_offset_limit=0.5,
        max_parameter_update_norm=0.025,
    )
    spec = handcrafted_pd_spec(speed=speed, duration=duration)
    params = initialize_actuator_skill_actor(jax.random.key(0), spec, config)
    # Zero all network leaves, so the checkpoint itself records the absent residual.
    params = params.replace(
        **{
            name: jnp.zeros_like(getattr(params, name))
            for name in params.__dataclass_fields__
            if not name.endswith("_offsets")
        }
    )
    anchors, _ = actuator_proprioceptive_state_bank(model)
    return ActuatorReferenceContract(
        params,
        model,
        anchors,
        spec,
        config,
        ActuatorReferenceConfig() if objective is None else objective,
    )


def with_learning_objective(bundle: Any, objective: ActuatorReferenceConfig) -> Any:
    """Change the declared loss while retaining every parameter and Adam-history leaf."""
    objective.validate()
    return replace(bundle, contract=replace(bundle.contract, learning_config=objective))


def library_size_metadata(bundle: Any) -> dict[str, int | str]:
    """Count actual controller candidates and trainable coordinates, not optimizer slots."""
    from crazyflow.safety.da_plcbf.persistent_skill_learner import _trainable_skill_tree

    ones = jax.tree.map(jnp.ones_like, bundle.state.params)
    mask = _trainable_skill_tree(ones, bundle.config)
    return {
        "fallback_policy_count": int(bundle.contract.spec.latent_codes.shape[0]),
        "trainable_parameter_count": int(sum(float(jnp.sum(v)) for v in jax.tree.leaves(mask))),
        "stored_parameter_count": int(sum(v.size for v in jax.tree.leaves(ones))),
        "adapter": bundle.config.adapter_mode,
    }
