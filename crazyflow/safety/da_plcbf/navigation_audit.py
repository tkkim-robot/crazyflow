"""Post-execution physical and operational audits of recorded navigation episodes.

This module uses executed plant nodes and explicitly valid control rows. Prediction residuals
remain separate from measured physical margins. It has no learner or policy-publication role.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from crazyflow.safety.da_plcbf.version_a_barriers import safety_constraint_names

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.mujoco_comparison_video import MethodVideoTrace
    from crazyflow.safety.da_plcbf.navigation_world import NavigationWorld


def audit_navigation_execution(
    states: np.ndarray,
    trace: MethodVideoTrace,
    world: NavigationWorld,
    *,
    arena_clearance: float = 0.08,
    domain_tolerance: float = 1e-7,
    barrier_tolerance: float = 3e-6,
    motor_tolerance: float = 3e-6,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return named physical/derivative/motor checks and full actual-node physical margins.

    Dimensionless physical values match ``dimensionless_safety_values`` for the nine operational
    columns. Physical limits do not use the estimated dynamics. Padding is excluded by the
    control-valid mask and cannot inflate counts or hide the final executed plant node.
    """
    states = np.asarray(states)
    active = np.asarray(trace.recorded_control_valid, dtype=bool)
    if states.ndim != 2 or states.shape[1] != 13 or not np.all(np.isfinite(states)):
        raise ValueError("dense states must be finite shape [nodes,13]")
    if active.shape != (len(trace.position),) or not np.any(active):
        raise ValueError("execution audit requires recorded valid control rows")
    if len(states) != int(np.sum(active)) * world.config.control_interval_steps + 1:
        raise ValueError("dense plant nodes must span exactly the executed control holds")
    quaternion = states[:, 3:7]
    norm = np.linalg.norm(quaternion, axis=1)
    if np.any(np.abs(norm - 1) > 2e-4):
        raise ValueError("recorded plant quaternions violate the model normalization tolerance")
    quaternion = quaternion / norm[:, None]
    body_z_vertical = 1 - 2 * (quaternion[:, 0] ** 2 + quaternion[:, 1] ** 2)
    position, velocity, rate = states[:, :3], states[:, 7:10], states[:, 10:13]
    cfg = world.config
    lower, upper = np.asarray(cfg.arena_lower), np.asarray(cfg.arena_upper)
    lower_margin, upper_margin = (
        position - lower - arena_clearance,
        upper - arena_clearance - position,
    )
    speed, angular_rate = np.linalg.norm(velocity, axis=1), np.linalg.norm(rate, axis=1)
    cosine = np.cos(cfg.tilt_max_radians)
    values = np.column_stack(
        (
            lower_margin / (upper - lower),
            upper_margin / (upper - lower),
            1 - speed**2 / cfg.speed_max**2,
            1 - angular_rate**2 / cfg.angular_rate_max**2,
            (body_z_vertical - cosine) / (1 - cosine),
        )
    )
    names = safety_constraint_names(0)
    minima = np.min(values, axis=0)
    motors = np.asarray(trace.actuator_margins)[active]
    if motors.shape != (int(np.sum(active)), 8) or not np.all(np.isfinite(motors)):
        raise ValueError("actuator audit requires eight finite margins per executed command")
    summary: dict[str, Any] = {
        "scope": (
            "actual executed integration nodes and valid applied commands; "
            "no continuous-time guarantee"
        ),
        "actual_node_count": len(states),
        "actual_physical_minimum_dimensionless_by_constraint": dict(
            zip(names, minima.tolist(), strict=True)
        ),
        "actual_physical_violating_nodes_by_constraint": dict(
            zip(names, np.sum(values < -domain_tolerance, axis=0).tolist(), strict=True)
        ),
        "all_actual_physical_nodes_pass": bool(np.all(values >= -domain_tolerance)),
        "minimum_arena_margin_m": float(np.min(np.column_stack((lower_margin, upper_margin)))),
        "minimum_speed_margin_m_per_s": float(cfg.speed_max - np.max(speed)),
        "minimum_angular_rate_margin_rad_per_s": float(cfg.angular_rate_max - np.max(angular_rate)),
        "minimum_tilt_margin_rad": float(
            cfg.tilt_max_radians - np.max(np.arccos(np.clip(body_z_vertical, -1, 1)))
        ),
        "minimum_applied_motor_margin_N": float(np.min(motors)),
        "applied_motor_limit_violating_controls": int(
            np.sum(np.any(motors < -motor_tolerance, axis=1))
        ),
        "predicted_derivative_scope": (
            "qp and fallback describe all evaluated proposals, including rejected or unused "
            "ones; applied describes executed commands. Minima and violation counts use "
            "finite entries only; missing finite evidence is null and nonfinite proposal "
            "counts are separate. Accepted or executed residuals must be finite."
        ),
    }
    for prefix in ("qp", "fallback", "applied"):
        residuals = np.asarray(getattr(trace, f"{prefix}_held_operational_residuals"))[active]
        if residuals.shape != (int(np.sum(active)), cfg.control_interval_steps, 9):
            raise ValueError("held residuals must cover the same valid controls and substeps")
        finite = np.isfinite(residuals)
        nonfinite_controls = ~np.all(finite, axis=(1, 2))
        if prefix == "applied":
            protected = np.ones(int(np.sum(active)), dtype=bool)
        else:
            mask_name = "qp_valid" if prefix == "qp" else "used_fallback"
            mask = np.asarray(getattr(trace, mask_name))
            if mask.shape != active.shape or mask.dtype != np.dtype(bool):
                raise ValueError(f"{mask_name} must identify accepted or executed proposal rows")
            protected = mask[active]
        if np.any(nonfinite_controls & protected):
            raise ValueError(f"accepted or executed {prefix} held residuals must be finite")
        columns = residuals.reshape(-1, 9)
        finite_minima = [
            float(np.min(column[np.isfinite(column)])) if np.any(np.isfinite(column)) else None
            for column in columns.T
        ]
        summary[f"{prefix}_predicted_derivative_minimum_by_constraint"] = dict(
            zip(names, finite_minima, strict=True)
        )
        summary[f"{prefix}_predicted_derivative_finite_entries_by_constraint"] = dict(
            zip(names, np.sum(finite, axis=(0, 1)).tolist(), strict=True)
        )
        summary[f"{prefix}_predicted_derivative_nonfinite_controls"] = int(
            np.sum(nonfinite_controls)
        )
        summary[f"{prefix}_predicted_derivative_violating_controls"] = int(
            np.sum(np.any(finite & (residuals < -barrier_tolerance), axis=(1, 2)))
        )
        if prefix != "applied":
            proposal_status = "rejected" if prefix == "qp" else "unexecuted"
            summary[f"{prefix}_{proposal_status}_nonfinite_proposal_controls"] = int(
                np.sum(nonfinite_controls & ~protected)
            )
    return summary, values
