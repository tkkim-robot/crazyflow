"""Exact hard shared-library certificates produced from Version-A quadrotor rollouts."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.capsules import (
    CapsuleBarrierConfig,
    CapsuleObstacleSet,
    quad_capsule_trajectory_values,
)
from crazyflow.safety.da_plcbf.quad_actor_losses import quad_safety_values
from crazyflow.safety.da_plcbf.quad_rollouts import rollout_shared_quad_library
from crazyflow.safety.da_plcbf.version_a_filter import PolicyLibraryCertificates

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.actor import (
        SharedActorConfig,
        SharedActorParams,
        SharedActorSpec,
    )
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
    from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
    from crazyflow.safety.da_plcbf.version_a_barriers import (
        RigidBodySafetySet,
        VersionABarrierConfig,
        VersionAModel,
    )
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


class LibraryCertificateDiagnostics(NamedTuple):
    """Runtime certificates plus hard-min identity and rollout audit fields."""

    certificates: PolicyLibraryCertificates
    active_indices: Array
    second_value_gaps: Array
    rollout_valid: Array
    includes_current_state: Array
    constraint_values: Array
    rollout_states: Array


class LibraryValueDiagnostics(NamedTuple):
    """Value-only hard-library evidence without the continuous-gradient construction."""

    values: Array
    rollout_valid: Array
    includes_current_state: Array
    constraint_values: Array
    fallback_wrenches: Array


def _validate_runtime_inputs(
    state: Array,
    scenarios: CircleScenarioBatch,
    safety: RigidBodySafetySet,
    barrier_config: VersionABarrierConfig,
    capsule_config: CapsuleBarrierConfig,
    *,
    dt: float,
    horizon: int,
    policy_gain: float,
) -> None:
    barrier_config.validate()
    capsule_config.validate()
    if state.shape != (13,):
        raise ValueError("state must have shape (13,)")
    if scenarios.obstacle_centers.shape[0] != 1:
        raise ValueError("runtime certificate generation requires exactly one scenario")
    if safety.obstacle_centers.shape[0] != 1:
        raise ValueError("runtime safety data must have exactly one scenario")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not math.isfinite(policy_gain) or policy_gain <= 0:
        raise ValueError("policy_gain must be finite and positive")


def _rollout_and_values(
    state: Array,
    params: SharedActorParams,
    spec: SharedActorSpec,
    scenarios: CircleScenarioBatch,
    safety: RigidBodySafetySet,
    model: VersionAModel,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    barrier_config: VersionABarrierConfig,
    *,
    dt: float,
    horizon: int,
    policy_gain: float,
    capsules: CapsuleObstacleSet | None,
    capsule_config: CapsuleBarrierConfig,
) -> tuple[Array, tuple[Array, ...]]:
    rollout = rollout_shared_quad_library(
        params,
        spec,
        state[None, :],
        scenarios,
        model,
        actuator,
        dt=dt,
        horizon=horizon,
        policy_gain=policy_gain,
        actor_config=actor_config,
        quad_config=quad_config,
    )
    values = quad_safety_values(rollout.states, safety, barrier_config, softmin_beta=1.0)
    masked_nodes = jnp.where(values.node_enabled, values.node_values, jnp.inf).reshape(
        spec.base_codes.shape[0], -1
    )
    masked_segments = jnp.where(
        values.segment_obstacle_enabled, values.segment_obstacle_values, jnp.inf
    ).reshape(spec.base_codes.shape[0], -1)
    flattened = jnp.concatenate((masked_nodes, masked_segments), axis=-1)
    rollout_valid = values.input_valid[:, 0] & jnp.all(rollout.policy_valid[:, 0], axis=-1)
    if capsules is not None:
        capsule_values = quad_capsule_trajectory_values(
            rollout.states, capsules, clearance=capsule_config.clearance, softmin_beta=1.0
        )
        capsule_nodes = jnp.where(
            capsule_values.node_enabled, capsule_values.node_values, jnp.inf
        ).reshape(spec.base_codes.shape[0], -1)
        capsule_segments = jnp.where(
            capsule_values.segment_enabled, capsule_values.segment_values, jnp.inf
        ).reshape(spec.base_codes.shape[0], -1)
        flattened = jnp.concatenate((flattened, capsule_nodes, capsule_segments), axis=-1)
        rollout_valid = rollout_valid & capsule_values.input_valid[:, 0]
    hard = jnp.min(flattened, axis=-1)
    current_scale = jnp.maximum(jnp.max(jnp.abs(state)), 1.0)
    current_tolerance = 32.0 * jnp.finfo(state.dtype).eps * current_scale
    includes_current = (
        jnp.max(jnp.abs(rollout.states[:, 0, 0] - state), axis=-1) <= current_tolerance
    )
    hard = jnp.where(rollout_valid & includes_current, hard, -jnp.inf)
    return hard, (
        flattened,
        rollout_valid,
        includes_current,
        rollout.wrenches[:, 0, 0],
        rollout.states[:, 0],
    )


def version_a_shared_library_values(
    state: Array,
    params: SharedActorParams,
    spec: SharedActorSpec,
    scenarios: CircleScenarioBatch,
    safety: RigidBodySafetySet,
    model: VersionAModel,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    barrier_config: VersionABarrierConfig,
    *,
    dt: float,
    horizon: int,
    policy_gain: float,
    capsules: CapsuleObstacleSet | None = None,
    capsule_config: CapsuleBarrierConfig = CapsuleBarrierConfig(),
) -> LibraryValueDiagnostics:
    """Evaluate exact hard values without constructing continuous PL-CBF gradients.

    This value-only path is used for held-step discrete postchecks.  It uses the same policies,
    horizon, physical node constraints, and swept geometry as the gradient-bearing certificates.
    """
    _validate_runtime_inputs(
        state,
        scenarios,
        safety,
        barrier_config,
        capsule_config,
        dt=dt,
        horizon=horizon,
        policy_gain=policy_gain,
    )
    values, auxiliary = _rollout_and_values(
        state,
        params,
        spec,
        scenarios,
        safety,
        model,
        actuator,
        actor_config,
        quad_config,
        barrier_config,
        dt=dt,
        horizon=horizon,
        policy_gain=policy_gain,
        capsules=capsules,
        capsule_config=capsule_config,
    )
    flattened, rollout_valid, includes_current, first_wrenches, _ = auxiliary
    return LibraryValueDiagnostics(
        values, rollout_valid, includes_current, flattened, first_wrenches
    )


def version_a_shared_library_certificates(
    state: Array,
    params: SharedActorParams,
    spec: SharedActorSpec,
    scenarios: CircleScenarioBatch,
    safety: RigidBodySafetySet,
    model: VersionAModel,
    actuator: VersionAActuator,
    actor_config: SharedActorConfig,
    quad_config: QuadPolicyConfig,
    barrier_config: VersionABarrierConfig,
    *,
    dt: float,
    horizon: int,
    policy_gain: float,
    capsules: CapsuleObstacleSet | None = None,
    capsule_config: CapsuleBarrierConfig = CapsuleBarrierConfig(),
) -> LibraryCertificateDiagnostics:
    """Differentiate every exact hard policy value with respect to one current state.

    The scenario batch must have ``B=1`` because this function produces one runtime filter
    decision.  The hard value includes current/future nodes and swept sphere segments.  A tied
    active minimum remains reportable but is not eligible as one differentiable continuous
    halfspace.
    """
    _validate_runtime_inputs(
        state,
        scenarios,
        safety,
        barrier_config,
        capsule_config,
        dt=dt,
        horizon=horizon,
        policy_gain=policy_gain,
    )

    def rollout_and_values(candidate_state: Array) -> tuple[Array, tuple[Array, ...]]:
        return _rollout_and_values(
            candidate_state,
            params,
            spec,
            scenarios,
            safety,
            model,
            actuator,
            actor_config,
            quad_config,
            barrier_config,
            dt=dt,
            horizon=horizon,
            policy_gain=policy_gain,
            capsules=capsules,
            capsule_config=capsule_config,
        )

    values, auxiliary = rollout_and_values(state)
    gradients = jax.jacfwd(lambda candidate_state: rollout_and_values(candidate_state)[0])(state)
    flattened, rollout_valid, includes_current, first_wrenches, rollout_states = auxiliary
    finite_constraints = jnp.where(jnp.isfinite(flattened), flattened, jnp.inf)
    ordered = jnp.sort(finite_constraints, axis=-1)
    indices = jnp.argmin(finite_constraints, axis=-1)
    finite_count = jnp.sum(jnp.isfinite(flattened), axis=-1)
    second = jnp.where(finite_count > 1, ordered[:, 1], jnp.inf)
    gaps = second - ordered[:, 0]
    gradient_valid = (
        rollout_valid
        & includes_current
        & jnp.isfinite(values)
        & jnp.all(jnp.isfinite(gradients), axis=-1)
        & ((finite_count == 1) | (gaps > barrier_config.minimum_tie_tolerance))
    )
    certificates = PolicyLibraryCertificates(
        values=values,
        gradients=gradients,
        gradient_valid=gradient_valid,
        fallback_wrenches=first_wrenches,
    )
    return LibraryCertificateDiagnostics(
        certificates, indices, gaps, rollout_valid, includes_current, flattened, rollout_states
    )


__all__ = [
    "LibraryCertificateDiagnostics",
    "LibraryValueDiagnostics",
    "version_a_shared_library_certificates",
    "version_a_shared_library_values",
]
