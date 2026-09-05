"""Offline counterfactual QPs and swept-geometry diagnostics for case discovery.

These diagnostics never alter a deployed controller or a learner. Each counterfactual reruns
the complete production control computation with one originally eligible policy retained as
the incumbent. Only selection preference changes: exact admissible fractions lie in [0, 1],
so a switch margin of one retains any eligible incumbent. All certificate thresholds, motor
and operational faces, predictive refinements, held checks, and emergency logic are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from crazyflow.safety.da_plcbf.continuous_version_a import (
    EXECUTION_MODES,
    QP_REJECTION_REASONS,
    ContinuousVersionAConfig,
    ContinuousVersionAStep,
    PolicyRollouts,
    RolloutFunction,
    RuntimeObstacleTrajectories,
    continuous_version_a_step,
    obstacle_agnostic_emergency_wrench,
)
from crazyflow.safety.da_plcbf.quad_rollouts import zero_order_hold_rollout
from crazyflow.safety.da_plcbf.selector import SelectionConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from crazyflow.safety.da_plcbf.navigation_experiment import NavigationExperimentConfig
    from crazyflow.safety.da_plcbf.navigation_world import NavigationWorld
    from crazyflow.safety.da_plcbf.version_a_barriers import (
        RigidBodySafetySet,
        VersionABarrierConfig,
        VersionAModel,
    )
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator, VersionAFilterConfig


@dataclass(frozen=True)
class AllPolicyQPAudit:
    """The normal decision and complete decisions for every originally eligible policy."""

    runtime: ContinuousVersionAStep
    counterfactuals: dict[int, ContinuousVersionAStep]


def _audit_eligible_policies(
    runtime: ContinuousVersionAStep, forced: Callable[[int], ContinuousVersionAStep]
) -> AllPolicyQPAudit:
    eligible = np.asarray(runtime.continuous_filter.policy_eligible)
    results = {}
    for index in np.flatnonzero(eligible):
        result = forced(int(index))
        if int(result.selected_index) != index:
            raise RuntimeError("eligible incumbent was not retained in offline diagnostic")
        if not np.array_equal(np.asarray(result.continuous_filter.policy_eligible), eligible):
            raise RuntimeError("offline selection override changed policy eligibility")
        results[int(index)] = result
    return AllPolicyQPAudit(runtime, results)


def make_navigation_policy_qp_auditor(
    world: NavigationWorld,
    bundle: Any,
    config: NavigationExperimentConfig,
    *,
    frozen_replacement: tuple[Any, int] | None = None,
) -> Callable[..., AllPolicyQPAudit]:
    """Build an all-eligible-policy auditor with the exact navigation controller settings.

    Its call signature is ``audit(state, params, model, obstacles, previous, goal)``. Parameters,
    state, model, obstacle prediction and goal are dynamic JIT inputs; reuse the returned callable
    across geometry candidates and snapshots of the same shape. An optional original-skill
    replacement uses the same explicit mixed evaluator as ``build_navigation_controller``.
    """
    from crazyflow.safety.da_plcbf.navigation_experiment import build_navigation_controller

    normal = build_navigation_controller(
        world, bundle, config, frozen_replacement=frozen_replacement
    )
    forced = build_navigation_controller(
        world,
        bundle,
        config,
        selection_config=SelectionConfig(switch_score_margin=1.0),
        frozen_replacement=frozen_replacement,
    )

    def audit(
        state: Array,
        params: Any,
        model: VersionAModel,
        obstacles: RuntimeObstacleTrajectories,
        previous: Array | int,
        goal: Array,
    ) -> AllPolicyQPAudit:
        runtime = normal(state, params, model, obstacles, jnp.asarray(previous, jnp.int32), goal)
        return _audit_eligible_policies(
            runtime,
            lambda index: forced(
                state, params, model, obstacles, jnp.asarray(index, jnp.int32), goal
            ),
        )

    return audit


def make_policy_qp_auditor(
    nominal_rollout: RolloutFunction,
    fallback_rollouts: RolloutFunction,
    actuator: VersionAActuator,
    safety_limits: RigidBodySafetySet,
    barrier_config: VersionABarrierConfig,
    filter_config: VersionAFilterConfig,
    config: ContinuousVersionAConfig,
    *,
    wrench_weight: Array | None = None,
    selection_config: SelectionConfig = SelectionConfig(),
) -> Callable[..., AllPolicyQPAudit]:
    """Compile reusable normal/forced-selection evaluators for one immutable snapshot.

    The returned host-side callable accepts ``state, obstacles, model, previous_policy_index``.
    It solves only originally eligible counterfactuals, serially, retaining the complete returned
    decision. It is intentionally not a deployment schedule or an alternative policy selector.
    Empty counterfactuals mean no candidate was eligible, not that all QPs were infeasible.
    """
    forced_config = replace(config, prefer_nominal_when_safe=False)
    forced_selection = replace(
        selection_config, switch_score_margin=1.0, prefer_first_eligible=False
    )

    def evaluate(
        state: Array,
        obstacles: RuntimeObstacleTrajectories,
        model: VersionAModel,
        previous: Array,
        *,
        forced: bool,
    ) -> ContinuousVersionAStep:
        return continuous_version_a_step(
            state,
            nominal_rollout,
            fallback_rollouts,
            obstacles,
            model,
            actuator,
            safety_limits,
            barrier_config,
            filter_config,
            forced_config if forced else config,
            wrench_weight=wrench_weight,
            previous_policy_index=previous,
            selection_config=forced_selection if forced else selection_config,
        )

    normal = jax.jit(lambda x, o, m, p: evaluate(x, o, m, p, forced=False))
    forced = jax.jit(lambda x, o, m, p: evaluate(x, o, m, p, forced=True))

    def audit(
        state: Array,
        obstacles: RuntimeObstacleTrajectories,
        model: VersionAModel,
        previous_policy_index: Array | int = -1,
    ) -> AllPolicyQPAudit:
        runtime = normal(state, obstacles, model, jnp.asarray(previous_policy_index, jnp.int32))
        return _audit_eligible_policies(
            runtime, lambda index: forced(state, obstacles, model, jnp.asarray(index, jnp.int32))
        )

    return audit


def rollout_emergency_brake(
    state: Array, model: VersionAModel, actuator: VersionAActuator, config: ContinuousVersionAConfig
) -> PolicyRollouts:
    """Roll out the unchanged feedback brake with the controller's physical command hold."""

    def command(current: Array, _: Array) -> tuple[Array, Array]:
        wrench, valid = obstacle_agnostic_emergency_wrench(current, model, actuator, config)
        return wrench, valid

    future, (wrenches, valid) = zero_order_hold_rollout(
        state,
        command,
        model,
        dt=config.dt,
        horizon=config.horizon,
        command_hold_steps=config.control_interval_steps,
    )
    states = jnp.concatenate((state[None], future), axis=0)[None]
    return PolicyRollouts(
        states, wrenches[None], (jnp.all(valid) & jnp.all(jnp.isfinite(states)))[None]
    )


def policy_geometry_diagnostics(
    rollout_states: Array | np.ndarray,
    obstacles: RuntimeObstacleTrajectories,
    config: ContinuousVersionAConfig,
) -> list[dict[str, Any]]:
    """Locate the exact node/swept hard minimum and independently measure metre clearance.

    The hard-value minimum need not minimize geometric clearance when obstacle radii differ.
    ``active_time_seconds`` is relative to this prediction's start and includes the closest
    fraction within a swept interval. It is not rounded to a control boundary.
    """
    config.validate()
    positions = np.asarray(rollout_states)[..., :3]
    centers, radii, mask = map(np.asarray, (obstacles.centers, obstacles.radii, obstacles.mask))
    if positions.ndim != 3 or positions.shape[1:] != (config.horizon + 1, 3):
        raise ValueError("rollout states must match the configured prediction horizon")
    if centers.shape != (config.horizon + 1, len(radii), 3) or mask.shape != centers.shape[:2]:
        raise ValueError("obstacle prediction shapes must match the configured horizon")
    if not np.isfinite(positions).all() or not np.isfinite(centers[mask]).all():
        raise ValueError("active rollout and obstacle positions must be finite")
    if not np.all(np.isfinite(radii[np.any(mask, axis=0)])) or np.any(
        radii[np.any(mask, axis=0)] <= 0
    ):
        raise ValueError("active obstacle radii must be positive finite")
    if not len(radii) or not mask.any():
        return [
            {"collision_constraints_active": False, "active_index": -1}
            for _ in range(len(positions))
        ]
    relative = positions[:, :, None, :] - np.where(mask[..., None], centers, 0)[None]
    start = relative[:, :-1]
    delta = relative[:, 1:] - start
    denominator = np.sum(delta**2, axis=-1)
    fraction = np.clip(
        -np.sum(start * delta, axis=-1) / np.where(denominator > 0, denominator, 1), 0, 1
    )
    distances = np.concatenate(
        (
            np.linalg.norm(relative, axis=-1).reshape(len(positions), -1),
            np.linalg.norm(start + fraction[..., None] * delta, axis=-1).reshape(
                len(positions), -1
            ),
        ),
        axis=-1,
    )
    enabled = np.concatenate((mask.reshape(-1), (mask[:-1] & mask[1:]).reshape(-1)))
    effective_radii = np.tile(
        radii + config.ego_radius + config.obstacle_clearance, 2 * config.horizon + 1
    )
    hard = np.where(enabled, distances**2 - effective_radii**2, np.inf)
    clearances = np.where(enabled, distances - effective_radii, np.inf)
    output = []
    node_count = (config.horizon + 1) * len(radii)
    for policy, active in enumerate(np.argmin(hard, axis=1)):
        is_node = active < node_count
        local = int(active if is_node else active - node_count)
        step, obstacle = divmod(local, len(radii))
        within = 0.0 if is_node else float(fraction[policy, step, obstacle])
        output.append(
            {
                "collision_constraints_active": True,
                "active_index": int(active),
                "active_kind": "node" if is_node else "swept_interval",
                "active_obstacle": obstacle,
                "active_time_seconds": (step + within) * config.dt,
                "active_interval_fraction": within,
                "active_hard_value_m2": float(hard[policy, active]),
                "active_clearance_m": float(clearances[policy, active]),
                "minimum_clearance_m": float(np.min(clearances[policy])),
            }
        )
    return output


def _json_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim:
        return [_json_value(item) for item in array]
    item = array.item()
    return None if isinstance(item, float) and not np.isfinite(item) else item


def summarize_qp_decision(decision: ContinuousVersionAStep) -> dict[str, Any]:
    """Compact JSON-compatible execution diagnostics; unavailable nonfinite scalars are null."""
    filtered = decision.continuous_filter
    names = (
        "action",
        "nominal_action",
        "selected_index",
        "selected_is_nominal",
        "qp_valid",
        "fallback_valid",
        "degraded",
        "executed_policy_dual",
        "selected_policy_dual",
        "qp_held_margin",
        "fallback_held_margin",
        "applied_held_margin",
        "applied_held_operational_margin",
        "applied_held_operational_residual",
        "applied_held_operational_passed",
        "qp_held_operational_residuals",
        "fallback_held_operational_residuals",
        "applied_held_operational_residuals",
        "applied_held_physical_margins",
        "predictive_operational_iterations",
        "initial_qp_held_operational_residual",
    )
    output = {name: _json_value(getattr(decision, name)) for name in names}
    output.update(
        execution_mode=EXECUTION_MODES[int(decision.execution_mode)],
        qp_rejections=[
            name
            for name, flag in zip(QP_REJECTION_REASONS, decision.qp_rejection_flags, strict=True)
            if bool(flag)
        ],
        proposed_qp_wrench=_json_value(filtered.qp.action),
        qp_kkt_valid=bool(filtered.qp_kkt_valid),
        qp_postcheck={
            name: _json_value(value) for name, value in filtered.qp_postcheck._asdict().items()
        },
        qp={
            name: _json_value(value)
            for name, value in filtered.qp._asdict().items()
            if name != "action"
        },
        selected_policy_row=_json_value(filtered.selected_policy_row),
        selected_policy_bound=_json_value(filtered.selected_policy_bound),
        motor_constraint_matrix=_json_value(filtered.motor_polytope.matrix),
        motor_constraint_upper_bound=_json_value(filtered.motor_polytope.upper_bound),
        analytic_constraint_matrix=_json_value(filtered.analytic_barriers.matrix),
        analytic_constraint_upper_bound=_json_value(filtered.analytic_barriers.upper_bound),
        analytic_constraint_enabled=_json_value(filtered.analytic_barriers.enabled),
    )
    return output


def summarize_policy_qp_audit(
    audit: AllPolicyQPAudit,
    obstacles: RuntimeObstacleTrajectories,
    config: ContinuousVersionAConfig,
) -> dict[str, Any]:
    """Expose the complete causal distinction between eligibility and executable full QPs."""
    runtime = audit.runtime
    return {
        "protocol": "offline forced eligible incumbent; unchanged full QP and execution checks",
        "runtime": summarize_qp_decision(runtime),
        "hard_values_m2": _json_value(runtime.values.values),
        "smooth_values_m2": _json_value(runtime.smooth_values),
        "smooth_gradients": _json_value(runtime.gradients),
        "smooth_time_derivatives_m2_per_s": _json_value(runtime.time_derivatives),
        "gradient_valid": _json_value(runtime.gradient_valid),
        "eligible": _json_value(runtime.continuous_filter.policy_eligible),
        "motor_box_admissible_fractions": _json_value(
            runtime.continuous_filter.policy_admissible_fractions
        ),
        "effective_smooth_temperature": _json_value(runtime.effective_smooth_temperature),
        "smooth_gap_bound": _json_value(runtime.smooth_gap_bound),
        "geometry": policy_geometry_diagnostics(runtime.candidates.states, obstacles, config),
        "eligible_full_qp_count": len(audit.counterfactuals),
        "eligible_accepted_held_qp_count": sum(
            bool(d.qp_valid) for d in audit.counterfactuals.values()
        ),
        "counterfactuals": {
            str(index): summarize_qp_decision(decision)
            for index, decision in audit.counterfactuals.items()
        },
    }
