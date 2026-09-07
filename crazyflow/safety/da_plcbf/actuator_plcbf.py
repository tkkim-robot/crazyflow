"""Actuator-state PL-CBF with four bounded effort commands and nonlinear hold checks.

The 17-state derivative has command dependence only in the motor rows. The collision value
uses the existing audited swept-sphere geometry on the body projection of a full augmented
rollout. Operational proposals use bounded sequential QPs through the actual held predictor;
no direct-wrench CBF row is padded or reused. Accepted checks are finite-horizon numerical
claims, not recursive feasibility across restarted skills, faults, or learner publications.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.actuator_dynamics import (
    ActuatorModel,
    augmented_control_affine_terms,
    effort_lag_step,
)
from crazyflow.safety.da_plcbf.continuous_version_a import (
    RuntimeObstacleTrajectories,
    RuntimePolicyValues,
    conservative_smooth_policy_values,
    runtime_policy_values,
    shift_obstacle_prediction,
    smooth_min_conservatism,
)
from crazyflow.safety.da_plcbf.selector import SelectionConfig, select_hard_policy
from crazyflow.safety.da_plcbf.version_a_barriers import (
    RigidBodySafetySet,
    VersionABarrierConfig,
    dimensionless_safety_values,
    validated_control_affine_terms,
)
from crazyflow.safety.da_plcbf.version_a_filter import (
    ValidatedMotorPolytope,
    VersionAFilterConfig,
    _project_with_exact_fast_path,
    motor_box_halfspace_fraction,
)

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.polytope_qp import PolytopeQPResult

ACTUATOR_EXECUTION_MODES = ("qp", "fallback", "emergency", "degraded", "invalid_input")
ACTUATOR_REJECTION_REASONS = (
    "invalid_input",
    "no_eligible_policy",
    "qp_infeasible",
    "kkt_failed",
    "command_or_motor_invalid",
    "plcbf_failed",
    "held_collision_failed",
    "held_operational_failed",
)


class ActuatorRollouts(NamedTuple):
    """Augmented trajectories with one held effort command per predictor node.

    Shapes are ``(K,H+1,17)``, ``(K,H,4)``, and ``(K,)``. Node zero includes the
    actual current motor state. The first candidate supplied to the filter is nominal.
    """

    states: Array
    commands: Array
    valid: Array


@dataclass(frozen=True, slots=True)
class ActuatorFilterConfig:
    """Fixed predictor, command-hold, and bounded refinement conventions.

    Operational refinement linearizes the minimum of each of nine physical margins over
    the held nodes, at the preceding proposal. A command trust box intersects the original
    bounds; at most ``sqp_iterations`` proposals are solved. Each proposal is independently
    checked with the full nonlinear hold. No constraint is relaxed if the budget is exhausted.
    """

    dt: float = 0.02
    horizon: int = 60
    command_hold_steps: int = 2
    held_substeps: int = 4
    obstacle_clearance: float = 0.15
    arena_clearance: float = 0.08
    ego_radius: float = 0.106
    smooth_temperature: float = 0.005
    smooth_gap_budget: float = 0.03
    policy_alpha: float = 2.0
    tolerance: float = 2e-6
    operational_tolerance: float = 1e-6
    kkt_tolerance: float = 5e-5
    sqp_iterations: int = 2
    command_trust_fraction: float = 0.5
    switch_score_margin: float = 0.02
    prefer_nominal_when_safe: bool = False
    gradient_mode: str = "forward"
    qp_numerics: str = "inward"

    @property
    def command_period(self) -> float:
        """Physical zero-order command hold in seconds."""
        return self.dt * self.command_hold_steps

    def validate(self) -> None:
        """Reject invalid physical or fixed-shape numerical settings."""
        for name in (
            "dt",
            "smooth_temperature",
            "smooth_gap_budget",
            "policy_alpha",
            "command_trust_fraction",
            "kkt_tolerance",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive finite")
        for name in (
            "obstacle_clearance",
            "arena_clearance",
            "ego_radius",
            "tolerance",
            "operational_tolerance",
            "switch_score_margin",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative finite")
        for name in ("horizon", "command_hold_steps", "held_substeps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.command_hold_steps > self.horizon:
            raise ValueError("command hold exceeds horizon")
        if type(self.sqp_iterations) is not int or not 0 <= self.sqp_iterations <= 4:
            raise ValueError("sqp_iterations must be an integer in [0,4]")
        if self.command_trust_fraction > 1:
            raise ValueError("command trust fraction cannot exceed the command range")
        if self.gradient_mode not in {"forward", "reverse", "directional"}:
            raise ValueError("gradient_mode must be forward, reverse, or directional")
        if self.qp_numerics not in {"legacy", "inward"}:
            raise ValueError("qp_numerics must be legacy or inward")


class ActuatorCertificates(NamedTuple):
    rollouts: ActuatorRollouts
    hard: RuntimePolicyValues
    smooth_values: Array
    gradients: Array
    gradient_components_computed: Array
    drift_derivatives: Array
    time_derivatives: Array
    rows: Array
    bounds: Array
    gradient_valid: Array
    eligible: Array
    command_volume_fractions: Array
    effective_temperature: Array
    smooth_gap_bound: Array
    input_valid: Array


class ActuatorHeldCheck(NamedTuple):
    nodes: Array
    actual_forces: Array
    applied_wrenches: Array
    collision_margin: Array
    operational_margins: Array
    command_motor_margin: Array
    finite: Array
    command_motor_passed: Array
    collision_passed: Array
    operational_passed: Array
    passed: Array


class ActuatorQPNumerics(NamedTuple):
    """Solver coordinates and executed-precision checks, including rejected proposals."""

    normalized_action: Array
    executed_command: Array
    row_scales: Array
    inward_margins: Array
    normalized_solver_tolerance: Array
    solver_feasible: Array
    fast_path_used: Array
    solver_primal_residual: Array
    solver_stationarity_residual: Array
    solver_complementarity_residual: Array
    executed_residuals: Array
    executed_rows_passed: Array


class ActuatorFilterStep(NamedTuple):
    action: Array
    nominal_action: Array
    next_estimated_state: Array
    certificates: ActuatorCertificates
    selected_index: Array
    qp: PolytopeQPResult
    qp_valid: Array
    fallback_valid: Array
    emergency_valid: Array
    execution_mode: Array
    degraded: Array
    applied: ActuatorHeldCheck
    qp_check: ActuatorHeldCheck
    fallback_check: ActuatorHeldCheck
    emergency_check: ActuatorHeldCheck
    executed_policy_dual: Array
    selected_policy_dual: Array
    policy_residual: Array
    qp_rejection_flags: Array
    sqp_iterations: Array
    intervention_norm: Array
    qp_numerics: ActuatorQPNumerics


ActuatorRolloutFunction = Callable[[Array, ActuatorModel], ActuatorRollouts]


def actuator_model_state_valid(state: Array, model: ActuatorModel) -> Array:
    """Audit state/model data without changing the model used in any calculation."""
    body_valid = validated_control_affine_terms(state[:13], model.body).input_valid
    scales = jnp.abs(model.force_to_wrench[:, 0])
    normalized = model.force_to_wrench / jnp.where(scales > 0, scales, 1.0)[:, None]
    gram = jnp.sum(normalized[:, :, None] * normalized[:, None, :], axis=0)
    mixer_valid = (
        jnp.all(scales > 0)
        & jnp.all(jnp.abs(model.force_to_wrench[0] - 1) < 2e-6)
        & jnp.all(jnp.abs(gram - 4 * jnp.eye(4, dtype=state.dtype)) < 2e-5)
    )
    return (
        body_valid
        & mixer_valid
        & jnp.all(jnp.isfinite(state))
        & jnp.all(jnp.isfinite(model.force_to_wrench))
        & jnp.all(jnp.isfinite(model.effectiveness))
        & jnp.all((model.effectiveness > 0) & (model.effectiveness <= 1))
        & jnp.all(jnp.isfinite(model.time_constants) & (model.time_constants > 0))
        & jnp.all(jnp.isfinite(model.command_lower))
        & jnp.all(jnp.isfinite(model.command_upper))
        & jnp.all((model.command_lower >= 0) & (model.command_upper > model.command_lower))
        & jnp.all((state[13:] >= 0) & (state[13:] <= model.command_upper + 2e-7))
    )


def command_box_fraction(row: Array, bound: Array, model: ActuatorModel) -> Array:
    """Exact volume fraction in effort-command coordinates using an identity map.

    The reused inclusion-exclusion utility is coordinate independent. Its identity map here
    is essential: no wrench-to-force transformation enters this command-space calculation.
    """
    identity = jnp.eye(4, dtype=row.dtype)
    box = ValidatedMotorPolytope(
        jnp.concatenate((identity, -identity)),
        jnp.concatenate((model.command_upper, -model.command_lower)),
        identity,
        identity,
        model.command_lower,
        model.command_upper,
        (model.command_lower + model.command_upper) / 2,
        jnp.asarray(0.0, row.dtype),
        jnp.asarray(True),
    )
    return motor_box_halfspace_fraction(row, bound, box)


def actuator_policy_certificates(
    state: Array,
    rollouts: ActuatorRolloutFunction,
    model: ActuatorModel,
    obstacles: RuntimeObstacleTrajectories,
    config: ActuatorFilterConfig,
) -> ActuatorCertificates:
    """Differentiate the same frozen full-state rollout and absolute obstacle clock.

    Policy phase restarts at zero and its anchor is recomputed from the perturbed initial
    state inside the callback. The time partial shifts obstacle predictions with the current
    waypoint/skill phase/model held fixed. Switching those objects is a separate value jump.
    """
    config.validate()
    if state.shape != (17,):
        raise ValueError("actuator PL-CBF requires an explicit 17-state effort vector")

    def smooth(y: Array, time_shift: Array) -> tuple[Array, tuple]:
        batch = rollouts(y, model)
        if batch.states.shape[1:] != (config.horizon + 1, 17):
            raise ValueError("rollout states must be (K,horizon+1,17)")
        if batch.commands.shape != (batch.states.shape[0], config.horizon, 4):
            raise ValueError("rollout commands must be (K,horizon,4)")
        if batch.valid.shape != (batch.states.shape[0],):
            raise ValueError("rollout validity must have one entry per candidate")
        shifted = shift_obstacle_prediction(obstacles, time_shift, dt=config.dt)
        hard = runtime_policy_values(
            batch.states[..., :13],
            shifted,
            obstacle_clearance=config.obstacle_clearance,
            ego_radius=config.ego_radius,
            envelope_derivative=True,
        )
        values = conservative_smooth_policy_values(
            hard, temperature=config.smooth_temperature, max_gap_budget=config.smooth_gap_budget
        )
        # A vacuous horizon has no barrier row. A finite placeholder keeps its Jacobian finite;
        # the independently returned infinity remains the geometric diagnostic.
        differentiable = jnp.where(jnp.isfinite(values), values, 0.0)
        return differentiable, (batch, hard, values)

    affine = augmented_control_affine_terms(state, model)
    zero_time = jnp.asarray(0.0, state.dtype)
    if config.gradient_mode == "directional":
        # The row needs four motor partials, DH*f_a, and partial_t(H). Forward
        # propagation of those six exact tangent directions avoids materializing
        # the thirteen unused body-coordinate partials. No finite difference or
        # numerical gradient approximation enters the controller.
        state_directions = jnp.concatenate(
            (
                jnp.eye(17, dtype=state.dtype)[13:],
                affine.drift[None],
                jnp.zeros((1, 17), dtype=state.dtype),
            )
        )
        time_directions = jnp.asarray([0, 0, 0, 0, 0, 1], dtype=state.dtype)

        def pushforward(dy: Array, dtime: Array) -> tuple:
            return jax.jvp(smooth, (state, zero_time), (dy, dtime), has_aux=True)

        _, derivative, (batch, hard, smooth_values) = jax.vmap(
            pushforward, out_axes=(None, -1, None)
        )(state_directions, time_directions)
        motor_gradients = derivative[:, :4]
        drift_derivatives, time_derivatives = derivative[:, 4], derivative[:, 5]
        # Missing diagnostic partials are explicit NaNs with a component mask;
        # they are never presented as zeros or used to form the physical row.
        gradients = jnp.concatenate(
            (jnp.full((len(derivative), 13), jnp.nan, dtype=state.dtype), motor_gradients), axis=-1
        )
        gradient_components_computed = jnp.arange(17) >= 13
        derivative_valid = jnp.all(jnp.isfinite(derivative), axis=-1)
        rows = -motor_gradients / model.time_constants
    else:
        differentiate = jax.jacfwd if config.gradient_mode == "forward" else jax.jacrev
        (gradients, time_derivatives), (batch, hard, smooth_values) = differentiate(
            smooth, argnums=(0, 1), has_aux=True
        )(state, zero_time)
        gradient_components_computed = jnp.ones((17,), dtype=bool)
        drift_derivatives = jnp.sum(gradients * affine.drift[None, :], axis=-1)
        derivative_valid = jnp.all(jnp.isfinite(gradients), axis=-1)
        rows = -jnp.matmul(gradients, affine.input_matrix, precision=jax.lax.Precision.HIGHEST)
    active = jnp.any(obstacles.mask)
    bounds = time_derivatives + drift_derivatives
    bounds = bounds + config.policy_alpha * jnp.where(active, smooth_values, 0.0)
    rows = jnp.where(active, rows, 0.0)
    bounds = jnp.where(active, bounds, 1.0)
    input_valid = actuator_model_state_valid(state, model) & jnp.any(hard.input_valid)
    gradient_valid = (
        input_valid
        & hard.input_valid
        & derivative_valid
        & jnp.isfinite(time_derivatives)
        & jnp.all(jnp.isfinite(rows), axis=-1)
        & jnp.isfinite(bounds)
        & batch.valid
    )
    fractions = jax.vmap(command_box_fraction, in_axes=(0, 0, None))(rows, bounds, model)
    eligible = gradient_valid & ((smooth_values >= 0) | ~active) & (fractions > 0)
    temperature, gap = smooth_min_conservatism(
        hard, temperature=config.smooth_temperature, max_gap_budget=config.smooth_gap_budget
    )
    return ActuatorCertificates(
        batch,
        hard,
        smooth_values,
        gradients,
        gradient_components_computed,
        drift_derivatives,
        time_derivatives,
        rows,
        bounds,
        gradient_valid,
        eligible,
        fractions,
        temperature,
        gap,
        input_valid,
    )


def held_actuator_nodes(
    state: Array, command: Array, model: ActuatorModel, config: ActuatorFilterConfig
) -> Array:
    """Integrate one unchanged command over all fine held-check nodes."""
    count = config.command_hold_steps * config.held_substeps
    step = config.dt / config.held_substeps

    def advance(y: Array, _: None) -> tuple[Array, Array]:
        following = effort_lag_step(y, command, model, step)
        return following, following

    _, future = jax.lax.scan(advance, state, None, length=count)
    return jnp.concatenate((state[None], future), axis=0)


def operational_node_margins(
    nodes: Array, safety: RigidBodySafetySet, *, arena_clearance: float = 0.08
) -> Array:
    """Nine physical margins per augmented node, with no old analytic input rows."""
    operational = safety._replace(obstacle_mask=jnp.zeros_like(safety.obstacle_mask))
    values = jax.vmap(
        lambda y: dimensionless_safety_values(
            y[:13],
            operational,
            VersionABarrierConfig(include_obstacle_hocbf=False, arena_clearance=arena_clearance),
        ).values[-9:]
    )(nodes)
    return values


def _held_obstacles(
    obstacles: RuntimeObstacleTrajectories, config: ActuatorFilterConfig
) -> RuntimeObstacleTrajectories:
    coordinates = jnp.arange(config.command_hold_steps * config.held_substeps + 1)
    coordinates = coordinates / config.held_substeps
    left = jnp.minimum(jnp.floor(coordinates).astype(jnp.int32), config.command_hold_steps - 1)
    fraction = coordinates - left
    centers = (1 - fraction[:, None, None]) * obstacles.centers[left]
    centers = centers + fraction[:, None, None] * obstacles.centers[left + 1]
    mask = obstacles.mask[left] & obstacles.mask[left + 1]
    return RuntimeObstacleTrajectories(centers, obstacles.radii, mask)


def check_actuator_hold(
    state: Array,
    command: Array,
    model: ActuatorModel,
    obstacles: RuntimeObstacleTrajectories,
    safety: RigidBodySafetySet,
    config: ActuatorFilterConfig,
) -> ActuatorHeldCheck:
    """Check the full nonlinear numerical hold, command bounds, motors, and geometry.

    Swept collision values are exact for the node interpolation. Continuous curved body
    motion and operational extrema between nodes require the separately reported refinement
    study; these checks alone are not a global continuous-time guarantee.
    """
    nodes = held_actuator_nodes(state, command, model, config)
    forces = nodes[:, 13:] * model.effectiveness
    wrenches = jnp.matmul(forces, model.force_to_wrench.T, precision=jax.lax.Precision.HIGHEST)
    collision = runtime_policy_values(
        nodes[None, :, :13],
        _held_obstacles(obstacles, config),
        obstacle_clearance=config.obstacle_clearance,
        ego_radius=config.ego_radius,
    )
    operational = operational_node_margins(nodes, safety, arena_clearance=config.arena_clearance)
    safety_valid = dimensionless_safety_values(
        state[:13], safety, VersionABarrierConfig(include_obstacle_hocbf=False)
    ).input_valid
    margin = jnp.min(
        jnp.concatenate(
            (
                command - model.command_lower,
                model.command_upper - command,
                nodes[:, 13:].reshape(-1),
                (model.command_upper - nodes[:, 13:]).reshape(-1),
            )
        )
    )
    finite = (
        actuator_model_state_valid(state, model)
        & jnp.all(jnp.isfinite(nodes))
        & jnp.all(jnp.isfinite(command))
        & jnp.all(jnp.isfinite(wrenches))
        & jnp.all(jnp.isfinite(operational))
        & collision.input_valid[0]
        & safety_valid
    )
    motor_pass = finite & (margin >= -config.tolerance)
    collision_pass = collision.values[0] >= -config.tolerance
    operation_pass = jnp.min(operational) >= -config.operational_tolerance
    return ActuatorHeldCheck(
        nodes,
        forces,
        wrenches,
        collision.values[0],
        operational,
        margin,
        finite,
        motor_pass,
        collision_pass,
        operation_pass,
        motor_pass & collision_pass & operation_pass,
    )


def predictive_operational_rows(
    state: Array,
    reference_command: Array,
    model: ActuatorModel,
    safety: RigidBodySafetySet,
    config: ActuatorFilterConfig,
) -> tuple[Array, Array]:
    """Linearize held physical margins at the declared command proposal.

    For ``r_j(u)>=0``, the row is ``-Dr_j(u0)u <= r_j(u0)-Dr_j(u0)u0``.
    Node zero is checked separately and omitted from the controllable proposal rows.
    Hard node minima are piecewise differentiable; finite-difference tests avoid ties.
    """

    def margins(command: Array) -> Array:
        nodes = held_actuator_nodes(state, command, model, config)
        return jnp.min(
            operational_node_margins(nodes[1:], safety, arena_clearance=config.arena_clearance),
            axis=0,
        )

    residual = margins(reference_command)
    derivative = jax.jacfwd(margins)(reference_command)
    return -derivative, residual - derivative @ reference_command


def _normalized_qp_with_audit(
    nominal: Array,
    row: Array,
    bound: Array,
    model: ActuatorModel,
    config: ActuatorFilterConfig,
    operational_rows: Array | None = None,
    operational_bounds: Array | None = None,
    reference: Array | None = None,
) -> tuple[PolytopeQPResult, ActuatorQPNumerics]:
    span = model.command_upper - model.command_lower
    z_nominal = (nominal - model.command_lower) / span
    lower, upper = jnp.zeros(4, nominal.dtype), jnp.ones(4, nominal.dtype)
    if reference is not None:
        z_reference = (reference - model.command_lower) / span
        lower = jnp.maximum(lower, z_reference - config.command_trust_fraction)
        upper = jnp.minimum(upper, z_reference + config.command_trust_fraction)
    identity = jnp.eye(4, dtype=nominal.dtype)
    matrix = jnp.concatenate((identity, -identity), axis=0)
    bounds = jnp.concatenate((upper, -lower))
    if operational_rows is not None:
        matrix = jnp.concatenate((matrix, operational_rows * span), axis=0)
        bounds = jnp.concatenate(
            (bounds, operational_bounds - operational_rows @ model.command_lower)
        )
    matrix = jnp.concatenate((matrix, (row * span)[None]), axis=0)
    bounds = jnp.concatenate((bounds, (bound - row @ model.command_lower)[None]))
    scales = jnp.linalg.norm(matrix, axis=-1)
    tolerance = jnp.asarray(config.tolerance, nominal.dtype)
    margins = jnp.zeros_like(bounds)
    if config.qp_numerics == "inward":
        # The solver divides row residuals by their norm. Bound its normalized tolerance
        # so no original affine row receives more than the declared raw-unit tolerance.
        tolerance = tolerance / jnp.maximum(1.0, jnp.max(scales))
        # Reserve roundoff room for dot products and conversion back to float32 Newtons.
        # This tightens the physical halfspaces; it never expands their accepted region.
        margins = (
            8 * jnp.finfo(nominal.dtype).eps * (jnp.abs(bounds) + jnp.sum(jnp.abs(matrix), axis=-1))
        )
    result, fast_path = _project_with_exact_fast_path(
        z_nominal,
        jnp.ones(4, nominal.dtype),
        matrix,
        bounds - margins,
        VersionAFilterConfig(qp_tolerance=tolerance, kkt_tolerance=config.kkt_tolerance),
    )
    # QP residuals are audited in normalized command coordinates; multiplier units correspond
    # to the explicitly normalized objective. Only the executable action is mapped back to N.
    command = model.command_lower + span * result.action
    # Re-evaluate ALL affine rows using the actual executable representation. Avoid a
    # normalize/map-back round trip when checking the original policy/operational rows.
    raw_residuals = jnp.concatenate(
        (
            model.command_lower + span * upper - command,
            command - (model.command_lower + span * lower),
        )
    )
    if operational_rows is not None:
        raw_residuals = jnp.concatenate(
            (raw_residuals, operational_bounds - operational_rows @ command)
        )
    raw_residuals = jnp.concatenate((raw_residuals, (bound - row @ command)[None]))
    executed_pass = jnp.all(jnp.isfinite(command)) & jnp.all(raw_residuals >= -config.tolerance)
    audit = ActuatorQPNumerics(
        result.action,
        command,
        scales,
        margins,
        tolerance,
        result.feasible,
        fast_path,
        result.primal_residual,
        result.stationarity_residual,
        result.complementarity_residual,
        raw_residuals,
        executed_pass,
    )
    feasible = (
        result.feasible & executed_pass if config.qp_numerics == "inward" else result.feasible
    )
    if config.qp_numerics == "inward":
        executed_z = (command - model.command_lower) / span
        delta = executed_z - z_nominal
        residuals = matrix @ executed_z - (bounds - margins)
        result = result._replace(
            objective=0.5 * jnp.dot(delta, delta),
            primal_residual=jnp.maximum(0.0, jnp.max(residuals)),
            stationarity_residual=jnp.max(jnp.abs(delta + matrix.T @ result.multipliers)),
            complementarity_residual=jnp.max(jnp.abs(result.multipliers * residuals)),
        )
    return result._replace(action=command, feasible=feasible), audit


def _normalized_qp(
    nominal: Array,
    row: Array,
    bound: Array,
    model: ActuatorModel,
    config: ActuatorFilterConfig,
    operational_rows: Array | None = None,
    operational_bounds: Array | None = None,
    reference: Array | None = None,
) -> PolytopeQPResult:
    """Compatibility entry point; runtime retains the accompanying numerical audit."""
    return _normalized_qp_with_audit(
        nominal, row, bound, model, config, operational_rows, operational_bounds, reference
    )[0]


def _pad_qp_audit(audit: ActuatorQPNumerics) -> ActuatorQPNumerics:
    def pad(value: Array) -> Array:
        return jnp.concatenate((value[:8], jnp.zeros(9, value.dtype), value[-1:]))

    return audit._replace(
        row_scales=pad(audit.row_scales),
        inward_margins=pad(audit.inward_margins),
        executed_residuals=pad(audit.executed_residuals),
    )


def _pad_operational_qp(result: PolytopeQPResult) -> PolytopeQPResult:
    """Retain a fixed diagnostic shape across initial and refined QP branches."""

    def pad(values: Array) -> Array:
        return jnp.concatenate((values[:8], jnp.zeros(9, values.dtype), values[-1:]))

    return result._replace(active_mask=pad(result.active_mask), multipliers=pad(result.multipliers))


def actuator_plcbf_step(
    state: Array,
    rollouts: ActuatorRolloutFunction,
    model: ActuatorModel,
    obstacles: RuntimeObstacleTrajectories,
    safety: RigidBodySafetySet,
    emergency_command: Array,
    previous_index: Array,
    config: ActuatorFilterConfig = ActuatorFilterConfig(),
    *,
    force_candidate_index: Array | None = None,
) -> ActuatorFilterStep:
    """Select a command-space certificate, solve, postcheck, and retain all rescue paths.

    ``force_candidate_index`` is an offline audit hook; benchmark runtime leaves it unset.
    Emergency commands are supplied by the common obstacle-free brake and shared F2 adapter.
    An emergency may pass the immediate physical checks without a policy certificate; it
    never receives a fictitious PL-CBF dual or a recursive-safety claim.
    """
    cert = actuator_policy_certificates(state, rollouts, model, obstacles, config)
    active = jnp.any(obstacles.mask)
    selection_values = jnp.where(active, cert.smooth_values, 1.0)
    selection = select_hard_policy(
        jnp.where(cert.eligible, selection_values, -jnp.inf),
        jnp.where(cert.eligible, cert.command_volume_fractions, 0.0),
        previous_index,
        SelectionConfig(
            switch_score_margin=config.switch_score_margin,
            prefer_first_eligible=config.prefer_nominal_when_safe,
        ),
    )
    # If every smooth certificate fails, preserve the strongest hard candidate as an explicit
    # best-effort diagnostic; no certificate is invented for it.
    strongest = jnp.argmax(jnp.where(cert.rollouts.valid, cert.hard.values, -jnp.inf))
    selected = jnp.where(jnp.any(cert.eligible), selection.selected_index, strongest)
    if force_candidate_index is not None:
        selected = jnp.clip(force_candidate_index, 0, cert.rows.shape[0] - 1)
    nominal = cert.rollouts.commands[0, 0]
    fallback = cert.rollouts.commands[selected, 0]
    row, bound = cert.rows[selected], cert.bounds[selected]
    has_certificate = cert.eligible[selected]
    qp, qp_audit = _normalized_qp_with_audit(nominal, row, bound, model, config)
    qp, qp_audit = _pad_operational_qp(qp), _pad_qp_audit(qp_audit)

    def held(command: Array) -> ActuatorHeldCheck:
        return check_actuator_hold(state, command, model, obstacles, safety, config)

    def qp_pass(result: PolytopeQPResult, check: ActuatorHeldCheck) -> Array:
        kkt = jnp.maximum(result.stationarity_residual, result.complementarity_residual)
        return (
            has_certificate
            & result.feasible
            & check.passed
            & (kkt <= config.kkt_tolerance)
            & (bound - row @ result.action >= -config.tolerance)
        )

    qp_check = held(qp.action)
    used_iterations = jnp.asarray(0, jnp.int32)
    if config.sqp_iterations:

        def refine(_: int, carry: tuple) -> tuple:
            result, check, count, audit = carry
            needs_refinement = has_certificate & ~qp_pass(result, check) & ~check.operational_passed

            def propose(current: tuple) -> tuple:
                previous, _, iterations, _ = current
                reference = jnp.where(
                    jnp.all(jnp.isfinite(previous.action)), previous.action, fallback
                )
                reference = jnp.clip(reference, model.command_lower, model.command_upper)
                op_rows, op_bounds = predictive_operational_rows(
                    state, reference, model, safety, config
                )
                proposal, proposal_audit = _normalized_qp_with_audit(
                    nominal, row, bound, model, config, op_rows, op_bounds, reference
                )
                return proposal, held(proposal.action), iterations + 1, proposal_audit

            return jax.lax.cond(needs_refinement, propose, lambda x: x, carry)

        qp, qp_check, used_iterations, qp_audit = jax.lax.fori_loop(
            0, config.sqp_iterations, refine, (qp, qp_check, used_iterations, qp_audit)
        )
    qp_valid = qp_pass(qp, qp_check)
    fallback_check = held(fallback)
    fallback_valid = (
        has_certificate & fallback_check.passed & (bound - row @ fallback >= -config.tolerance)
    )
    emergency_check = held(emergency_command)
    emergency_valid = emergency_check.passed
    action = jnp.where(qp_valid, qp.action, jnp.where(fallback_valid, fallback, emergency_command))
    safety_valid = dimensionless_safety_values(
        state[:13], safety, VersionABarrierConfig(include_obstacle_hocbf=False)
    ).input_valid
    emergency_input_valid = (
        jnp.all(jnp.isfinite(emergency_command))
        & jnp.all(emergency_command >= model.command_lower - config.tolerance)
        & jnp.all(emergency_command <= model.command_upper + config.tolerance)
    )
    input_valid = (
        cert.input_valid & safety_valid & emergency_input_valid & jnp.all(jnp.isfinite(nominal))
    )
    action = jnp.where(input_valid, action, jnp.full_like(action, jnp.nan))
    mode = jnp.where(qp_valid, 0, jnp.where(fallback_valid, 1, jnp.where(emergency_valid, 2, 3)))
    mode = jnp.where(input_valid, mode, 4)
    applied = jax.tree.map(
        lambda q, f, e: jnp.where(qp_valid, q, jnp.where(fallback_valid, f, e)),
        qp_check,
        fallback_check,
        emergency_check,
    )
    applied = jax.lax.cond(input_valid, lambda _: applied, lambda _: held(action), operand=None)
    residual = bound - row @ action
    dual = qp.multipliers[-1]
    flags = jnp.stack(
        (
            ~input_valid,
            ~has_certificate,
            ~qp.feasible,
            jnp.maximum(qp.stationarity_residual, qp.complementarity_residual)
            > config.kkt_tolerance,
            ~qp_check.command_motor_passed,
            ~(bound - row @ qp.action >= -config.tolerance),
            ~qp_check.collision_passed,
            ~qp_check.operational_passed,
        )
    )
    return ActuatorFilterStep(
        action,
        nominal,
        applied.nodes[-1],
        cert,
        selected,
        qp,
        qp_valid,
        fallback_valid,
        emergency_valid,
        mode,
        ~qp_valid & ~fallback_valid,
        applied,
        qp_check,
        fallback_check,
        emergency_check,
        jnp.where(qp_valid & active, dual, 0.0),
        dual,
        residual,
        flags,
        used_iterations,
        jnp.linalg.norm((action - nominal) / (model.command_upper - model.command_lower)),
        qp_audit,
    )
