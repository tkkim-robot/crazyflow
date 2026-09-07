"""Warm-started MPC-based finite-horizon safety filter for the actuator study.

This nonlinear trajectory comparator uses the same 17-state effort plant, command
holds, obstacle prediction and first-hold postcheck as the library filter. It has
no terminal invariant set and makes no recursive-feasibility claim. The predictive
safety-filter framework (Wabersich and Zeilinger, https://arxiv.org/abs/1812.05506)
is context, not a claim that this finite-horizon implementation reproduces its
assumptions or guarantees.

Every mandatory operation inside ``solve`` is timed, including plan revalidation,
initializations, transfers, optimization and final nonlinear checks. JAX calls and
SLSQP cannot be interrupted inside an individual evaluation; overruns remain in
the measured service time. A late result never appears as ``available_action`` at
the requested deadline. Explicit predeployment ``warmup`` measures compilation
separately and never alters a saved plan or solver state.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from crazyflow.safety.da_plcbf.actuator_dynamics import ActuatorModel, effort_lag_step
from crazyflow.safety.da_plcbf.actuator_learning import (
    ActuatorSkillConfig,
    acceleration_to_actuator_command,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import (
    ActuatorFilterConfig,
    ActuatorHeldCheck,
    ActuatorRollouts,
    actuator_model_state_valid,
    check_actuator_hold,
    operational_node_margins,
)
from crazyflow.safety.da_plcbf.continuous_version_a import (
    RuntimeObstacleTrajectories,
    runtime_policy_values,
)
from crazyflow.safety.da_plcbf.version_a_barriers import (
    RigidBodySafetySet,
    VersionABarrierConfig,
    dimensionless_safety_values,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class ActuatorOPTConfig:
    """Development-frozen control-knot, objective and actual service-budget settings."""

    control_knots: int = 6
    wall_time_budget: float = 0.040
    finalization_reserve_fraction: float = 0.10
    max_iterations: int = 80
    function_tolerance: float = 1e-7
    command_change_weight: float = 0.005
    effort_weight: float = 0.0001
    evasion_acceleration: float = 2.0
    brake_gain: float = 2.0
    initializations: tuple[str, ...] = ("nominal", "brake", "left", "right", "up")

    def validate(self, holds: int) -> None:
        """Reject invalid budgets and undeclared initialization families."""
        if type(self.control_knots) is not int or not 1 <= self.control_knots <= holds:
            raise ValueError("control_knots must be an integer in [1, command holds]")
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        for name in (
            "wall_time_budget",
            "function_tolerance",
            "evasion_acceleration",
            "brake_gain",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive finite")
        for name in ("command_change_weight", "effort_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative finite")
        if not math.isfinite(self.finalization_reserve_fraction) or not (
            0 <= self.finalization_reserve_fraction < 1
        ):
            raise ValueError("finalization_reserve_fraction must be in [0,1)")
        allowed = {"nominal", "brake", "left", "right", "up"}
        if not self.initializations or not set(self.initializations) <= allowed:
            raise ValueError("initializations must name nominal/brake/left/right/up")
        if len(set(self.initializations)) != len(self.initializations):
            raise ValueError("initializations must be unique")


@dataclass(frozen=True, slots=True)
class ActuatorOPTPlan:
    """Immutable completed plan at a recorded observation clock, in physical units."""

    commands: np.ndarray  # One physical command per actual hold, not integration node.
    states: np.ndarray  # H+1 augmented states, including the source state.
    observation_time: float
    objective: float
    minimum_collision_margin: float
    minimum_operational_margin: float
    minimum_motor_margin: float


@dataclass(frozen=True, slots=True)
class ActuatorOPTAttempt:
    initialization: str
    initial_feasible: bool
    initial_objective: float
    iterations: int
    evaluations: int
    success: bool
    status: str
    service_seconds: float


@dataclass(frozen=True, slots=True)
class ActuatorOPTResult:
    """Planned action and independently timed availability; no implicit late execution."""

    action: np.ndarray
    available_action: np.ndarray | None
    plan: ActuatorOPTPlan | None
    held_check: ActuatorHeldCheck
    mode: str
    feasible: bool
    service_seconds: float
    deadline_met: bool
    available_at: float
    attempts: tuple[ActuatorOPTAttempt, ...]
    warm_start_revalidated: bool
    warm_start_feasible: bool
    warm_start_shift: int
    budget_exhausted: bool
    input_valid: bool
    compilation_in_solve: bool  # Conservative flag: this signature was not explicitly prewarmed.
    evaluations: int


class _PlanEvaluation(NamedTuple):
    objective: jax.Array
    constraints: jax.Array
    states: jax.Array
    collision_margin: jax.Array
    operational_margin: jax.Array
    motor_margin: jax.Array
    valid: jax.Array


class _BudgetExpired(RuntimeError):
    pass


def _readonly(value: Any) -> np.ndarray:
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _synchronize(value: Any) -> Any:
    return jax.device_get(value)


class ActuatorOPTController:
    """State-explicit SLSQP controller with JAX trajectory values and exact AD Jacobians.

    The caller retains the prior completed plan. Nominal commands that pass the full
    horizon and fine first-hold checks use a zero-intervention fast path. Otherwise
    SLSQP minimizes normalized nominal deviation plus command-change and effort
    regularization. Previous feasible plans and feasible solver iterates may be
    retained after current-state/model/obstacle revalidation. No solver success flag
    substitutes for the nonlinear physical checks.
    """

    def __init__(
        self,
        filter_config: ActuatorFilterConfig,
        actor_config: ActuatorSkillConfig,
        opt_config: ActuatorOPTConfig = ActuatorOPTConfig(),
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        filter_config.validate()
        actor_config.validate()
        if filter_config.horizon % filter_config.command_hold_steps:
            raise ValueError("OPT horizon must contain an integer number of command holds")
        if not math.isclose(actor_config.dt, filter_config.dt) or (
            actor_config.control_interval_steps != filter_config.command_hold_steps
        ):
            raise ValueError("actor and OPT must share exactly the same command clock")
        self.filter_config = filter_config
        self.actor_config = actor_config
        self.opt_config = opt_config
        self.holds = filter_config.horizon // filter_config.command_hold_steps
        opt_config.validate(self.holds)
        self.clock = clock
        self._warmed_signatures: set[tuple] = set()
        self._knot_indices = jnp.asarray(
            np.arange(self.holds) * opt_config.control_knots // self.holds, dtype=jnp.int32
        )
        self._knot_starts = np.array(
            [
                np.flatnonzero(np.asarray(self._knot_indices) == k)[0]
                for k in range(opt_config.control_knots)
            ]
        )
        self._evaluate = jax.jit(self._evaluate_plan)
        self._differentiate = jax.jit(jax.jacfwd(self._knot_problem, has_aux=True))
        self._held = jax.jit(
            lambda y, u, m, o, s: check_actuator_hold(y, u, m, o, s, self.filter_config)
        )
        self._seeds = jax.jit(self._make_seeds)

    def _rollout(self, commands: jax.Array, state: jax.Array, model: ActuatorModel) -> jax.Array:
        repeated = jnp.repeat(commands, self.filter_config.command_hold_steps, axis=0)

        def advance(y: jax.Array, command: jax.Array) -> tuple[jax.Array, jax.Array]:
            following = effort_lag_step(
                y, command, model, self.filter_config.dt, substeps=self.actor_config.plant_substeps
            )
            return following, following

        _, future = jax.lax.scan(advance, state, repeated)
        return jnp.concatenate((state[None], future), axis=0)

    def _evaluate_plan(
        self,
        commands: jax.Array,
        state: jax.Array,
        model: ActuatorModel,
        obstacles: RuntimeObstacleTrajectories,
        safety: RigidBodySafetySet,
        nominal: jax.Array,
        previous_command: jax.Array,
    ) -> _PlanEvaluation:
        nodes = self._rollout(commands, state, model)
        collision = runtime_policy_values(
            nodes[None, :, :13],
            obstacles,
            obstacle_clearance=self.filter_config.obstacle_clearance,
            ego_radius=self.filter_config.ego_radius,
            envelope_derivative=True,
        )
        operational = operational_node_margins(nodes, safety)
        span = model.command_upper - model.command_lower
        z = (commands - model.command_lower) / span
        z_previous = (previous_command - model.command_lower) / span
        difference = jnp.diff(jnp.concatenate((z_previous[None], z), axis=0), axis=0)
        objective = jnp.mean(((commands - nominal) / span) ** 2)
        objective += self.opt_config.command_change_weight * jnp.mean(difference**2)
        objective += self.opt_config.effort_weight * jnp.mean(z**2)
        motors = jnp.concatenate((nodes[:, 13:], model.command_upper - nodes[:, 13:]), axis=-1)
        motor_scales = jnp.concatenate((span, span))
        motor_constraints = (motors + self.filter_config.tolerance) / motor_scales
        # Masked obstacles give +inf by contract; solver placeholders are constant,
        # while the independent collision validity bit retains invalid-input failures.
        collision_rows = collision.constraint_values[0]
        collision_constraints = jnp.where(
            jnp.isfinite(collision_rows), collision_rows + self.filter_config.tolerance, 1.0
        )
        constraints = jnp.concatenate(
            (
                collision_constraints,
                (operational + self.filter_config.operational_tolerance).reshape(-1),
                motor_constraints.reshape(-1),
                z.reshape(-1),
                (1 - z).reshape(-1),
            )
        )
        safety_valid = dimensionless_safety_values(
            state[:13], safety, VersionABarrierConfig(include_obstacle_hocbf=False)
        ).input_valid
        valid = (
            actuator_model_state_valid(state, model)
            & safety_valid
            & collision.input_valid[0]
            & jnp.all(jnp.isfinite(nodes))
            & jnp.all(jnp.isfinite(constraints))
            & jnp.isfinite(objective)
        )
        return _PlanEvaluation(
            objective,
            constraints,
            nodes,
            collision.values[0],
            jnp.min(operational),
            jnp.min(motors),
            valid,
        )

    def _knot_problem(self, vector: jax.Array, *args: Any) -> tuple[jax.Array, _PlanEvaluation]:
        model = args[1]
        knots = vector.reshape(self.opt_config.control_knots, 4)
        commands = model.command_lower + (model.command_upper - model.command_lower) * knots
        evaluation = self._evaluate_plan(commands[self._knot_indices], *args)
        values = jnp.concatenate((evaluation.objective[None], evaluation.constraints))
        return values, evaluation

    def _make_seeds(self, state: jax.Array, model: ActuatorModel) -> jax.Array:
        velocity = state[7:10]
        xy_norm = jnp.sqrt(jnp.sum(velocity[:2] ** 2))
        forward = jnp.where(
            xy_norm > 1e-6,
            velocity[:2] / jnp.maximum(xy_norm, 1e-6),
            jnp.array([1.0, 0.0], state.dtype),
        )
        left = jnp.array([-forward[1], forward[0], 0.0])
        directions = jnp.stack((jnp.zeros(3), left, -left, jnp.array([0.0, 0.0, 1.0])))

        def one(direction: jax.Array) -> jax.Array:
            def hold(y: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array]:
                pulse = jnp.where(index < max(1, self.holds // 2), 1.0, 0.0)
                acceleration = -self.opt_config.brake_gain * y[7:10]
                acceleration += pulse * self.opt_config.evasion_acceleration * direction
                acceleration = jnp.clip(
                    acceleration,
                    -self.actor_config.acceleration_limit,
                    self.actor_config.acceleration_limit,
                )
                command = acceleration_to_actuator_command(
                    acceleration, y, model, self.actor_config
                ).command
                following = effort_lag_step(
                    y,
                    command,
                    model,
                    self.filter_config.command_period,
                    substeps=self.filter_config.command_hold_steps
                    * self.actor_config.plant_substeps,
                )
                return following, command

            _, commands = jax.lax.scan(hold, state, jnp.arange(self.holds))
            return commands

        return jax.vmap(one)(directions)

    def _arguments(
        self,
        state: Any,
        model: ActuatorModel,
        obstacles: RuntimeObstacleTrajectories,
        safety: RigidBodySafetySet,
        nominal_commands: Any,
        emergency_command: Any,
        previous_command: Any | None,
    ) -> tuple:
        state = jnp.asarray(state)
        if state.shape != (17,):
            raise ValueError("OPT requires the full 17-state effort model")
        nominal = np.asarray(nominal_commands)
        if nominal.shape == (self.filter_config.horizon, 4):
            reshaped = nominal.reshape(self.holds, self.filter_config.command_hold_steps, 4)
            if not np.all(reshaped == reshaped[:, :1]):
                raise ValueError("nominal commands must be unchanged inside each declared hold")
            nominal = reshaped[:, 0]
        if nominal.shape != (self.holds, 4):
            raise ValueError("nominal_commands must be (H,4) or (command_holds,4)")
        if np.shape(emergency_command) != (4,):
            raise ValueError("emergency_command must have shape (4,)")
        previous = nominal[0] if previous_command is None else previous_command
        if np.shape(previous) != (4,):
            raise ValueError("previous_command must have shape (4,)")
        return (
            state,
            model,
            obstacles,
            safety,
            jnp.asarray(nominal, dtype=state.dtype),
            jnp.asarray(previous, dtype=state.dtype),
        )

    @staticmethod
    def _signature(arguments: tuple) -> tuple:
        return tuple(
            (
                tuple(np.shape(x)),
                str(getattr(x, "dtype", type(x))),
                bool(getattr(x, "weak_type", False)),
            )
            for x in jax.tree.leaves(arguments)
        )

    def warmup(
        self,
        state: Any,
        model: ActuatorModel,
        obstacles: RuntimeObstacleTrajectories,
        safety: RigidBodySafetySet,
        nominal_commands: Any,
        emergency_command: Any,
    ) -> float:
        """Compile every numerical kernel and synchronize; never install a previous plan."""
        started = self.clock()
        args = self._arguments(
            state, model, obstacles, safety, nominal_commands, emergency_command, None
        )
        span = model.command_upper - model.command_lower
        vector = ((args[4][self._knot_starts] - model.command_lower) / span).reshape(-1)
        _synchronize(self._evaluate(args[4], *args))
        _synchronize(self._differentiate(vector, *args))
        _synchronize(
            self._held(
                args[0],
                jnp.asarray(emergency_command, dtype=args[0].dtype),
                model,
                obstacles,
                safety,
            )
        )
        _synchronize(self._seeds(args[0], model))
        self._warmed_signatures.add(self._signature(args))
        return self.clock() - started

    def solve(
        self,
        state: Any,
        model: ActuatorModel,
        obstacles: RuntimeObstacleTrajectories,
        safety: RigidBodySafetySet,
        nominal_commands: Any,
        emergency_command: Any,
        *,
        previous_plan: ActuatorOPTPlan | None = None,
        previous_command: Any | None = None,
        observation_time: float = 0.0,
    ) -> ActuatorOPTResult:
        """Solve one causal current-snapshot problem and report actual result availability."""
        started = self.clock()
        if not math.isfinite(observation_time):
            raise ValueError("observation_time must be finite")
        args = self._arguments(
            state, model, obstacles, safety, nominal_commands, emergency_command, previous_command
        )
        compilation_in_solve = self._signature(args) not in self._warmed_signatures
        state, model, obstacles, safety, nominal, previous = args
        lower, span = (
            np.asarray(model.command_lower),
            np.asarray(model.command_upper - model.command_lower),
        )
        attempts: list[ActuatorOPTAttempt] = []
        best: tuple[np.ndarray, _PlanEvaluation, ActuatorHeldCheck, str] | None = None
        warm_revalidated, warm_feasible, warm_shift = False, False, 0
        evaluations = 0
        exhausted = False

        def expired() -> bool:
            solve_budget = self.opt_config.wall_time_budget * (
                1 - self.opt_config.finalization_reserve_fraction
            )
            return self.clock() - started >= solve_budget

        def check_budget() -> None:
            if expired():
                raise _BudgetExpired("wall-clock solve allowance exhausted; finalization reserved")

        def consider(commands: np.ndarray, evaluation: _PlanEvaluation, mode: str) -> bool:
            nonlocal best
            commands = np.asarray(commands, dtype=state.dtype)
            feasible = bool(evaluation.valid) and bool(np.min(evaluation.constraints) >= 0.0)
            if not feasible:
                return False
            held = _synchronize(
                self._held(
                    state, jnp.asarray(commands[0], dtype=state.dtype), model, obstacles, safety
                )
            )
            if not bool(held.passed):
                return False
            if best is None or float(evaluation.objective) < float(best[1].objective):
                best = (np.asarray(commands), evaluation, held, mode)
            return True

        nominal_evaluation = _synchronize(self._evaluate(nominal, *args))
        evaluations += 1
        emergency = np.asarray(emergency_command)
        input_valid = (
            bool(nominal_evaluation.valid)
            and bool(np.all(np.isfinite(emergency)))
            and bool(np.all(emergency >= lower))
            and bool(np.all(emergency <= lower + span))
            and bool(np.all(np.isfinite(previous)))
        )
        nominal_safe = input_valid and consider(np.asarray(nominal), nominal_evaluation, "nominal")
        warm_commands = None
        if not nominal_safe and previous_plan is not None and input_valid:
            age = observation_time - previous_plan.observation_time
            if age >= 0 and previous_plan.commands.shape == (self.holds, 4):
                warm_shift = int(math.floor(age / self.filter_config.command_period + 1e-7))
                if warm_shift < self.holds:
                    warm_commands = np.concatenate(
                        (
                            previous_plan.commands[warm_shift:],
                            np.repeat(previous_plan.commands[-1:], warm_shift, axis=0),
                        ),
                        axis=0,
                    )
                    evaluation = _synchronize(self._evaluate(jnp.asarray(warm_commands), *args))
                    evaluations += 1
                    warm_revalidated = True
                    warm_feasible = consider(warm_commands, evaluation, "retained")

        if not nominal_safe and input_valid:
            seeds: dict[str, np.ndarray] = {"nominal": np.asarray(nominal)}
            if warm_commands is not None:
                seeds = {"warm_start": warm_commands, **seeds}
            try:
                check_budget()
                if any(name != "nominal" for name in self.opt_config.initializations):
                    generated = _synchronize(self._seeds(state, model))
                    seeds.update(zip(("brake", "left", "right", "up"), generated, strict=True))
                order = ("warm_start",) if warm_commands is not None else ()
                order += self.opt_config.initializations
                # Establish checked recovery initializations before spending the
                # remaining service budget improving any one local solution.
                # Only completed, bounded knot plans pass this admission check.
                ranked: list[tuple[bool, float, float, str]] = []
                for label in order:
                    check_budget()
                    seed_started = self.clock()
                    knots = np.clip(seeds[label][self._knot_starts], lower, lower + span)
                    commands = knots[np.asarray(self._knot_indices)]
                    evaluation = _synchronize(
                        self._evaluate(jnp.asarray(commands, dtype=state.dtype), *args)
                    )
                    evaluations += 1
                    feasible = consider(commands, evaluation, "initialization")
                    objective = float(evaluation.objective)
                    violation = max(0.0, -float(np.min(evaluation.constraints)))
                    if not bool(evaluation.valid):
                        violation = math.inf
                    ranked.append((not feasible, 0.0 if feasible else violation, objective, label))
                    attempts.append(
                        ActuatorOPTAttempt(
                            label,
                            feasible,
                            objective,
                            0,
                            1,
                            False,
                            "initialization_postcheck_passed"
                            if feasible
                            else "initialization_postcheck_failed",
                            self.clock() - seed_started,
                        )
                    )
                # Feasible starts precede infeasible starts; objective ranks within
                # each group. Stable ordering preserves declared ties and warm starts.
                order = tuple(item[3] for item in sorted(ranked, key=lambda item: item[:3]))
                for label in order:
                    check_budget()
                    attempt_start = self.clock()
                    seed = seeds[label]
                    vector = (
                        np.clip((seed[self._knot_starts] - lower) / span, 0, 1)
                        .astype(float)
                        .reshape(-1)
                    )
                    cached_x = None
                    cached_jac = None
                    cached_eval = None
                    cached_feasible = False
                    count = 0
                    iterations = 0
                    initial_feasible = False
                    initial_objective = math.inf

                    def evaluate(z: np.ndarray) -> tuple[np.ndarray, _PlanEvaluation]:
                        nonlocal \
                            cached_x, \
                            cached_jac, \
                            cached_eval, \
                            cached_feasible, \
                            count, \
                            evaluations
                        check_budget()
                        if cached_x is None or not np.array_equal(z, cached_x):
                            jacobian, evaluation = _synchronize(
                                self._differentiate(jnp.asarray(z, dtype=state.dtype), *args)
                            )
                            count += 1
                            evaluations += 1
                            if not np.all(np.isfinite(jacobian)) or not bool(evaluation.valid):
                                raise FloatingPointError("nonfinite trajectory or AD Jacobian")
                            cached_x, cached_jac, cached_eval = (
                                z.copy(),
                                np.asarray(jacobian),
                                evaluation,
                            )
                            commands = lower + span * z.reshape(self.opt_config.control_knots, 4)
                            cached_feasible = consider(
                                commands[np.asarray(self._knot_indices)], evaluation, "optimized"
                            )
                        return cached_jac, cached_eval

                    def callback(_: np.ndarray) -> None:
                        nonlocal iterations
                        iterations += 1
                        check_budget()

                    success, status = False, "not_started"
                    try:
                        _, first = evaluate(vector)
                        initial_objective = float(first.objective)
                        initial_feasible = cached_feasible
                        result = minimize(
                            lambda z: float(evaluate(z)[1].objective),
                            vector,
                            jac=lambda z: np.asarray(evaluate(z)[0][0], dtype=float),
                            method="SLSQP",
                            bounds=[(0.0, 1.0)] * vector.size,
                            constraints={
                                "type": "ineq",
                                "fun": lambda z: np.asarray(
                                    evaluate(z)[1].constraints, dtype=float
                                ),
                                "jac": lambda z: np.asarray(evaluate(z)[0][1:], dtype=float),
                            },
                            callback=callback,
                            options={
                                "maxiter": self.opt_config.max_iterations,
                                "ftol": self.opt_config.function_tolerance,
                                "disp": False,
                            },
                        )
                        iterations = int(result.nit)
                        success, status = bool(result.success), str(result.message)
                        evaluate(np.asarray(result.x))
                    except _BudgetExpired as exc:
                        exhausted, status = True, str(exc)
                    except Exception as exc:
                        # Native SLSQP has its own Exception subclass. Preserve
                        # its type/message and checked rescue in the episode trace.
                        status = f"{type(exc).__name__}: {exc}"
                    attempts.append(
                        ActuatorOPTAttempt(
                            label,
                            initial_feasible,
                            initial_objective,
                            iterations,
                            count,
                            success,
                            status,
                            self.clock() - attempt_start,
                        )
                    )
                    if exhausted:
                        break
            except _BudgetExpired:
                exhausted = True

        if best is None:
            action = emergency if input_valid else np.full(4, np.nan, dtype=state.dtype)
            held = _synchronize(
                self._held(state, jnp.asarray(action, dtype=state.dtype), model, obstacles, safety)
            )
            mode = "emergency" if bool(held.passed) else "degraded"
            if not input_valid:
                mode = "invalid_input"
            plan = None
        else:
            commands, evaluation, held, mode = best
            action = commands[0]
            plan = ActuatorOPTPlan(
                _readonly(commands),
                _readonly(evaluation.states),
                observation_time,
                float(evaluation.objective),
                float(evaluation.collision_margin),
                float(evaluation.operational_margin),
                float(evaluation.motor_margin),
            )
        # Include mandatory result copies and all final postchecks in service time.
        action = _readonly(action)
        service = self.clock() - started
        deadline_met = service <= self.opt_config.wall_time_budget
        return ActuatorOPTResult(
            action,
            action if deadline_met and input_valid else None,
            plan,
            held,
            mode,
            best is not None,
            service,
            deadline_met,
            observation_time + service,
            tuple(attempts),
            warm_revalidated,
            warm_feasible,
            warm_shift,
            exhausted or not deadline_met,
            input_valid,
            compilation_in_solve,
            evaluations,
        )


def build_actuator_opt(
    filter_config: ActuatorFilterConfig,
    actor_config: ActuatorSkillConfig,
    opt_config: ActuatorOPTConfig = ActuatorOPTConfig(),
) -> ActuatorOPTController:
    """Build a state-explicit MPC-based finite-horizon safety filter; no implicit warmup."""
    return ActuatorOPTController(filter_config, actor_config, opt_config)


def opt_plan_rollouts(plan: ActuatorOPTPlan, config: ActuatorFilterConfig) -> ActuatorRollouts:
    """Expose a completed plan in the common augmented trajectory trace format."""
    return ActuatorRollouts(
        jnp.asarray(plan.states)[None],
        jnp.repeat(jnp.asarray(plan.commands), config.command_hold_steps, axis=0)[None],
        jnp.array([True]),
    )
