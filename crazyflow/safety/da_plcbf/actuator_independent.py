"""Independent airborne evaluation plants for the actuator study.

P1 integrates independently written NumPy Newton--Euler equations with RK4 and
analytically evolving effort at all RK stages. It imports no predictor transition,
body derivative, quaternion helper, or allocator. P2 integrates the native
``first_principles`` RPM differential equations, including separate up/down
coefficients, both polynomial curves, and propeller inertia. Its internal state
ends in RPM; observations end in *nominal-equivalent effort*, before effectiveness.
Those P2 observations use the actual plant thrust curve and true RPM: they are an
explicit motor-state/curve oracle, not an implemented sensor or estimator. The
returned model snapshot holds only the surrogate parameters active now.

This is numerical/model-mismatch validation, not hardware validation. Native
parameters contain TODOs (including inertia and some curve coefficients), and the
native drag matrix was fitted for a different aggregate model. We preserve native
motor order, inertial-torque signs, and its use of L directly as the roll/pitch
lever, despite its docstring describing L as radial distance. We do not add a
sqrt(2) correction. Effectiveness scales thrust and aerodynamic reaction torque
once; rotor momentum/inertial torque remains present. Contact is not modeled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from crazyflow.drones import load_params

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from crazyflow.safety.da_plcbf.actuator_dynamics import ActuatorModel
    from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel


def _array(value: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite ({size},) array")
    return result


def _readonly(value: ArrayLike) -> NDArray[np.float64]:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _snapshot(model: ActuatorModel) -> ActuatorModel:
    body = model.body._make(_readonly(value) for value in model.body)
    return model._replace(
        body=body,
        **{name: _readonly(getattr(model, name)) for name in model._fields if name != "body"},
    )


def _validate_step(
    state: ArrayLike, command: ArrayLike, model: ActuatorModel, dt: float, substeps: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    state = _array(state, 17, "state").copy()
    command = _array(command, 4, "command")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    tau = _array(model.time_constants, 4, "time_constants")
    eta = _array(model.effectiveness, 4, "effectiveness")
    lower = _array(model.command_lower, 4, "command_lower")
    upper = _array(model.command_upper, 4, "command_upper")
    if np.any(tau <= 0) or np.any(eta <= 0) or np.any(eta > 1):
        raise ValueError("positive lag and partial effectiveness in (0, 1] are required")
    if np.any(lower < 0) or np.any(upper <= lower):
        raise ValueError("invalid effort command bounds")
    if np.any(command < lower) or np.any(command > upper):
        raise ValueError("effort command violates the actual motor bounds; no clipping is applied")
    norm = np.linalg.norm(state[3:7])
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("state quaternion must have nonzero norm")
    state[3:7] /= norm
    return state, command


def _body_rhs(
    state: NDArray[np.float64], wrench: NDArray[np.float64], body: VersionAModel
) -> NDArray[np.float64]:
    """Independent Newton--Euler derivative; SciPy defines the quaternion rotation."""
    quat = state[3:7] / np.linalg.norm(state[3:7])
    rotation = Rotation.from_quat(quat).as_matrix()
    velocity, omega = state[7:10], state[10:13]
    relative_body = rotation.T @ (velocity - np.asarray(body.wind_velocity))
    world_force = (
        rotation[:, 2] * wrench[0]
        + rotation @ (np.asarray(body.drag_matrix) @ relative_body)
        + np.asarray(body.external_force)
    )
    acceleration = np.asarray(body.gravity_vec) + world_force / float(np.asarray(body.mass))
    inertia = np.asarray(body.inertia)
    torque = wrench[1:] + rotation.T @ np.asarray(body.external_torque)
    omega_dot = np.linalg.solve(inertia, torque - np.cross(omega, inertia @ omega))
    # Explicit right quaternion multiplication, q_dot = q * [omega, 0] / 2.
    x, y, z, w = quat
    ox, oy, oz = omega
    quat_dot = 0.5 * np.array(
        [
            w * ox + y * oz - z * oy,
            w * oy + z * ox - x * oz,
            w * oz + x * oy - y * ox,
            -x * ox - y * oy - z * oz,
        ]
    )
    return np.concatenate((velocity, quat_dot, acceleration, omega_dot))


def independent_effort_step(
    state: ArrayLike, command: ArrayLike, model: ActuatorModel, dt: float, *, substeps: int = 1
) -> NDArray[np.float64]:
    """Advance P1 with RK4 body stages and exact held effort, independently of P0."""
    current, command = _validate_step(state, command, model, dt, substeps)
    if np.any(current[13:] < 0) or np.any(current[13:] > np.asarray(model.command_upper)):
        raise ValueError("initial effort state is outside [0, command_upper]")
    tau, eta, mapping = map(
        np.asarray, (model.time_constants, model.effectiveness, model.force_to_wrench)
    )
    h = dt / substeps
    for _ in range(substeps):
        initial = current[13:].copy()

        def rhs(body_state: NDArray[np.float64], elapsed: float) -> NDArray[np.float64]:
            effort = initial + (-np.expm1(-elapsed / tau)) * (command - initial)
            return _body_rhs(body_state, mapping @ (eta * effort), model.body)

        body = current[:13]
        k1 = rhs(body, 0.0)
        k2 = rhs(body + 0.5 * h * k1, 0.5 * h)
        k3 = rhs(body + 0.5 * h * k2, 0.5 * h)
        k4 = rhs(body + h * k3, h)
        current[:13] = body + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        current[3:7] /= np.linalg.norm(current[3:7])
        current[13:] = initial + (-np.expm1(-h / tau)) * (command - initial)
    if not np.all(np.isfinite(current)):
        raise FloatingPointError("nonfinite P1 integration")
    return current


class RPMConversion(NamedTuple):
    """Audited effort-to-RPM conversion with no bound expansion or clipping."""

    rpm: NDArray[np.float64]
    reconstructed_effort: NDArray[np.float64]
    residual: float
    rpm_lower: NDArray[np.float64]
    rpm_upper: NDArray[np.float64]
    command_bound_residual: float


def rpm_to_effort(rpm: ArrayLike, coefficients: ArrayLike) -> NDArray[np.float64]:
    """Evaluate the complete thrust polynomial [offset, linear, quadratic] in N."""
    a, b, c = _array(coefficients, 3, "thrust coefficients")
    rotor = np.asarray(rpm, dtype=np.float64)
    if not np.all(np.isfinite(rotor)) or np.any(rotor < 0):
        raise ValueError("RPM must be finite and nonnegative")
    return a + rotor * (b + c * rotor)


def effort_to_rpm(
    effort: ArrayLike, coefficients: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> RPMConversion:
    """Invert the nonnegative strictly increasing physical polynomial branch.

    The upper quadratic root is selected; a negative linear coefficient therefore
    does not select the negative-thrust low-speed branch. Offsets are retained.
    RPM bounds are the images of the declared effort bounds, not new hardware
    speed specifications. Out-of-range or noninvertible requests raise an error.
    For negative linear coefficients, zero effort maps to the upper zero-thrust
    root at nonzero RPM. This airborne branch is not a stopped-rotor/idle model.
    """
    effort = np.asarray(effort, dtype=np.float64)
    lower, upper = np.broadcast_arrays(np.asarray(lower, float), np.asarray(upper, float))
    lower, upper, effort = np.broadcast_arrays(lower, upper, effort)
    a, b, c = _array(coefficients, 3, "thrust coefficients")
    if (
        not np.all(np.isfinite([lower, upper, effort]))
        or np.any(lower < 0)
        or np.any(lower >= upper)
        or c < 0
        or (c == 0 and b <= 0)
    ):
        raise ValueError("invalid bounds or non-increasing thrust curve")
    bound_residual = float(np.max(np.maximum(lower - effort, effort - upper)))
    if bound_residual > 0:
        raise ValueError("effort lies outside the declared motor command bounds")

    def invert(force: NDArray[np.float64]) -> NDArray[np.float64]:
        if c == 0:
            rpm = (force - a) / b
        else:
            discriminant = b * b + 4 * c * (force - a)
            if np.any(discriminant <= 0):
                raise ValueError("force does not have a strictly increasing physical RPM branch")
            root = np.sqrt(discriminant)
            rpm = (-b + root) / (2 * c) if b <= 0 else 2 * (force - a) / (b + root)
        if np.any(rpm < 0) or np.any(b + 2 * c * rpm <= 0) or not np.all(np.isfinite(rpm)):
            raise ValueError("force has no nonnegative strictly increasing physical RPM branch")
        return rpm

    rpm_lower, rpm_upper, rpm = invert(lower), invert(upper), invert(effort)
    reconstructed = rpm_to_effort(rpm, coefficients)
    residual = float(np.max(np.abs(reconstructed - effort)))
    if residual > 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(effort)))):
        raise FloatingPointError("RPM conversion exceeds the double precision residual tolerance")
    return RPMConversion(rpm, reconstructed, residual, rpm_lower, rpm_upper, bound_residual)


@dataclass(frozen=True, slots=True)
class NativeRotorParameters:
    """Native polynomial/dynamic parameters and an explicit local lag reference."""

    thrust_coefficients: NDArray[np.float64]
    torque_coefficients: NDArray[np.float64]
    rotor_coefficients: NDArray[np.float64]
    mixing_matrix: NDArray[np.float64]
    arm_length: float
    propeller_inertia: float
    nominal_time_constant: float
    drone: str = "custom"

    def __post_init__(self) -> None:
        """Freeze arrays and reject invalid native physical parameters."""
        for name, size in (
            ("thrust_coefficients", 3),
            ("torque_coefficients", 3),
            ("rotor_coefficients", 4),
        ):
            object.__setattr__(self, name, _readonly(_array(getattr(self, name), size, name)))
        mixing = np.asarray(self.mixing_matrix, dtype=float)
        if mixing.shape != (3, 4) or not np.all(np.isfinite(mixing)):
            raise ValueError("mixing_matrix must be finite and (3, 4)")
        object.__setattr__(self, "mixing_matrix", _readonly(mixing))
        if not all(
            math.isfinite(value) and value > 0
            for value in (self.arm_length, self.nominal_time_constant)
        ):
            raise ValueError("arm length and nominal time constant must be positive")
        if not math.isfinite(self.propeller_inertia) or self.propeller_inertia < 0:
            raise ValueError("propeller inertia must be nonnegative")
        if np.any(self.rotor_coefficients < 0):
            raise ValueError("native rotor coefficients must be nonnegative")

    @classmethod
    def from_drone(cls, drone: str = "cf21B_500") -> NativeRotorParameters:
        """Load repository values and invert the mean of local hover response rates."""
        params = load_params(drone)
        hover = params["mass"] * -params["gravity_vec"][2] / 4
        rpm = float(
            effort_to_rpm(
                hover, params["rpm2thrust"], params["thrust_min"], params["thrust_max"]
            ).rpm
        )
        c0, c1, c2, c3 = params["rotor_dyn_coef"]
        tau = 2 / (c0 + 2 * c1 * rpm + c2 + 2 * c3 * rpm)
        return cls(
            params["rpm2thrust"],
            params["rpm2torque"],
            params["rotor_dyn_coef"],
            params["mixing_matrix"],
            params["L"],
            params["prop_inertia"],
            tau,
            drone,
        )


def native_rotor_wrench(
    native_state: ArrayLike,
    command_rpm: ArrayLike,
    model: ActuatorModel,
    parameters: NativeRotorParameters,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return native applied wrench, RPM derivative, and actual motor forces.

    Lag changes scale both native response rates by nominal_tau/current_tau.
    This preserves native up/down asymmetry; it does not turn RPM into effort lag.
    """
    state = _array(native_state, 17, "native_state")
    rotor, command = state[13:], _array(command_rpm, 4, "command_rpm")
    c0, c1, c2, c3 = parameters.rotor_coefficients
    rotor_dot = np.where(
        command > rotor,
        c0 * (command - rotor) + c1 * (command**2 - rotor**2),
        c2 * (command - rotor) + c3 * (command**2 - rotor**2),
    ) * (parameters.nominal_time_constant / np.asarray(model.time_constants))
    eta = np.asarray(model.effectiveness)
    forces = eta * rpm_to_effort(rotor, parameters.thrust_coefficients)
    reaction = eta * rpm_to_effort(rotor, parameters.torque_coefficients)
    mixing = parameters.mixing_matrix
    torques = parameters.arm_length * (mixing @ forces) * np.array([1.0, 1.0, 0.0])
    torques[2] = mixing[2] @ reaction
    signed_speed = mixing[2] @ rotor * (2 * np.pi / 60)
    signed_acceleration = mixing[2] @ rotor_dot * (2 * np.pi / 60)
    torques += parameters.propeller_inertia * np.array(
        [state[11] * signed_speed, -state[10] * signed_speed, signed_acceleration]
    )
    return np.concatenate(([np.sum(forces)], torques)), rotor_dot, forces


def native_rotor_step(
    native_state: ArrayLike,
    command: ArrayLike,
    model: ActuatorModel,
    parameters: NativeRotorParameters,
    dt: float,
    *,
    substeps: int = 1,
) -> NDArray[np.float64]:
    """Advance native body-plus-RPM equations jointly with NumPy RK4."""
    current, command = _validate_step(native_state, command, model, dt, substeps)
    effort = rpm_to_effort(current[13:], parameters.thrust_coefficients)
    slope = parameters.thrust_coefficients[1] + 2 * parameters.thrust_coefficients[2] * current[13:]
    if (
        np.any(slope <= 0)
        or np.any(effort < -1e-12)
        or np.any(effort > np.asarray(model.command_upper) + 1e-12)
    ):
        raise ValueError(
            "native RPM state is outside the nonnegative-effort physical branch/bounds"
        )
    conversion = effort_to_rpm(
        command, parameters.thrust_coefficients, model.command_lower, model.command_upper
    )
    h = dt / substeps

    def rhs(state: NDArray[np.float64]) -> NDArray[np.float64]:
        wrench, rotor_dot, _ = native_rotor_wrench(state, conversion.rpm, model, parameters)
        return np.concatenate((_body_rhs(state[:13], wrench, model.body), rotor_dot))

    for _ in range(substeps):
        previous_rpm = current[13:].copy()
        k1 = rhs(current)
        k2 = rhs(current + 0.5 * h * k1)
        k3 = rhs(current + 0.5 * h * k2)
        k4 = rhs(current + h * k3)
        current += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        current[3:7] /= np.linalg.norm(current[3:7])
        if (
            not np.all(np.isfinite(current))
            or np.any(current[13:] < np.minimum(previous_rpm, conversion.rpm) - 1e-8)
            or np.any(current[13:] > np.maximum(previous_rpm, conversion.rpm) + 1e-8)
        ):
            raise FloatingPointError("native RK4 lost monotone motor response; refine substeps")
    return current


@dataclass(frozen=True, slots=True)
class ActuatorEvent:
    """Exogenous parameter change on the absolute plant clock; no state reset."""

    time: float
    model: ActuatorModel


class IndependentPlantTrace(NamedTuple):
    """Actual hold trace, including both endpoints and every integration/event boundary."""

    times: NDArray[np.float64]
    states: NDArray[np.float64]
    native_states: NDArray[np.float64]
    actual_forces: NDArray[np.float64]
    wrenches: NDArray[np.float64]
    conversion_residual: float


class NumpyEffortPlant:
    """Causal P1 evaluator; model snapshots expose only parameters active now."""

    def __init__(
        self,
        initial_state: ArrayLike,
        model: ActuatorModel,
        *,
        events: Sequence[ActuatorEvent] = (),
        time: float = 0.0,
        max_step: float = 0.002,
    ) -> None:
        if not math.isfinite(time) or time < 0 or not math.isfinite(max_step) or max_step <= 0:
            raise ValueError("initial time must be nonnegative and max_step positive")
        self.time, self.max_step = float(time), float(max_step)
        self._state = _array(initial_state, 17, "initial_state").copy()
        self._model = _snapshot(model)
        self._events = tuple(ActuatorEvent(event.time, _snapshot(event.model)) for event in events)
        event_times = [event.time for event in self._events]
        if any(not math.isfinite(t) or t < 0 for t in event_times) or any(
            later <= earlier
            for earlier, later in zip(event_times[:-1], event_times[1:], strict=True)
        ):
            raise ValueError("events must have finite, nonnegative, strictly increasing times")
        self._event_index = 0
        self._apply_events()

    def _apply_events(self) -> None:
        while self._event_index < len(self._events):
            event = self._events[self._event_index]
            if event.time > self.time:
                break
            self._model = event.model
            self._event_index += 1

    def model_snapshot(self) -> ActuatorModel:
        """Copy the parameters active at the current time, without future event data."""
        return _snapshot(self._model)

    def observe(self) -> NDArray[np.float64]:
        """Return 17 body/nominal-equivalent-effort coordinates."""
        return self._state.copy()

    @property
    def native_state(self) -> NDArray[np.float64]:
        """Return a copy of the actual plant state (effort for P1, RPM for P2)."""
        return self._state.copy()

    def _step(self, command: NDArray[np.float64], dt: float) -> None:
        self._state = independent_effort_step(self._state, command, self._model, dt)

    def _audit(
        self, command: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        forces = np.asarray(self._model.effectiveness) * self._state[13:]
        return forces, np.asarray(self._model.force_to_wrench) @ forces, 0.0

    def advance(self, command: ArrayLike, duration: float) -> IndependentPlantTrace:
        """Hold a command, splitting at parameter events without revealing them to control."""
        command = _array(command, 4, "command")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        end = self.time + duration
        times, states, native_states, forces, wrenches = [], [], [], [], []
        max_residual = 0.0

        def record() -> None:
            nonlocal max_residual
            actual_force, wrench, residual = self._audit(command)
            times.append(self.time)
            states.append(self.observe())
            native_states.append(self.native_state)
            forces.append(actual_force)
            wrenches.append(wrench)
            max_residual = max(max_residual, residual)

        record()
        while self.time < end:
            boundary = min(end, self.time + self.max_step)
            if self._event_index < len(self._events):
                boundary = min(boundary, self._events[self._event_index].time)
            self._step(command, boundary - self.time)
            self.time = boundary
            self._apply_events()
            record()
        return IndependentPlantTrace(
            np.asarray(times),
            np.asarray(states),
            np.asarray(native_states),
            np.asarray(forces),
            np.asarray(wrenches),
            max_residual,
        )


class NativeRotorPlant(NumpyEffortPlant):
    """P2 evaluator with persistent native RPM, sharing the controller's effort interface."""

    def __init__(
        self,
        initial_state: ArrayLike,
        model: ActuatorModel,
        parameters: NativeRotorParameters | None = None,
        *,
        events: Sequence[ActuatorEvent] = (),
        time: float = 0.0,
        max_step: float = 0.002,
    ) -> None:
        super().__init__(initial_state, model, events=events, time=time, max_step=max_step)
        self.parameters = NativeRotorParameters.from_drone() if parameters is None else parameters
        # Initial states may be below the airborne command minimum, but never above its maximum.
        conversion = effort_to_rpm(
            self._state[13:],
            self.parameters.thrust_coefficients,
            np.minimum(self._model.command_lower, self._state[13:]),
            self._model.command_upper,
        )
        self._state[13:] = conversion.rpm

    def observe(self) -> NDArray[np.float64]:
        observed = self._state.copy()
        observed[13:] = rpm_to_effort(observed[13:], self.parameters.thrust_coefficients)
        return observed

    def _step(self, command: NDArray[np.float64], dt: float) -> None:
        self._state = native_rotor_step(self._state, command, self._model, self.parameters, dt)

    def _audit(
        self, command: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        conversion = effort_to_rpm(
            command,
            self.parameters.thrust_coefficients,
            self._model.command_lower,
            self._model.command_upper,
        )
        wrench, _, forces = native_rotor_wrench(
            self._state, conversion.rpm, self._model, self.parameters
        )
        return forces, wrench, conversion.residual


__all__ = [
    "ActuatorEvent",
    "IndependentPlantTrace",
    "NativeRotorParameters",
    "NativeRotorPlant",
    "NumpyEffortPlant",
    "RPMConversion",
    "effort_to_rpm",
    "independent_effort_step",
    "native_rotor_step",
    "native_rotor_wrench",
    "rpm_to_effort",
]
