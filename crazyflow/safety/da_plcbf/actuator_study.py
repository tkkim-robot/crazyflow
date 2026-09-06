"""Complete actuator-study episodes, exogenous worlds, and causal execution clocks.

Actor callbacks receive only augmented proprioception, immutable skill identity and the
current point model through their shared low-level adapter. Goals and obstacle predictions
enter the nominal controller and safety filter here. Every method executes its own complete
episode, including ordinary rescue and post-task exposure to the remaining prescribed threats.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.actuator_compute import MatchedEffortPlant
from crazyflow.safety.da_plcbf.actuator_dynamics import (
    ActuatorModel,
    fit_native_time_constant,
    make_actuator_model,
)
from crazyflow.safety.da_plcbf.actuator_independent import (
    ActuatorEvent,
    NativeRotorParameters,
    NativeRotorPlant,
    NumpyEffortPlant,
)
from crazyflow.safety.da_plcbf.actuator_learning import (
    ActuatorSkillConfig,
    acceleration_to_actuator_command,
    rollout_actuator_skill_library,
)
from crazyflow.safety.da_plcbf.actuator_plcbf import (
    ActuatorFilterConfig,
    ActuatorRollouts,
    actuator_plcbf_step,
)
from crazyflow.safety.da_plcbf.actuator_protocol import physical_world_id
from crazyflow.safety.da_plcbf.case_study_world import (
    GuardSphere,
    HoverEncounterConfig,
    IncomingSphere,
    build_hover_encounter_world,
)
from crazyflow.safety.da_plcbf.navigation_world import NavigationWorld, NavigationWorldConfig
from crazyflow.safety.da_plcbf.online_constant_wind import build_cf21b_version_a_resources

if TYPE_CHECKING:
    from collections.abc import Callable

    from crazyflow.safety.da_plcbf.continuous_version_a import RuntimeObstacleTrajectories
    from crazyflow.safety.da_plcbf.persistent_skill_learner import (
        SkillActorParams,
        SkillLibrarySpec,
    )
    from crazyflow.safety.da_plcbf.version_a_barriers import RigidBodySafetySet


DYNAMICS_CELLS = ("nominal", "effectiveness", "lag", "combined")
PRIMARY_METHODS = ("F2", "DR", "A", "OPT")
ALL_METHODS = ("F0", "F1", "F2", "DR", "A", "A1", "OPT")
# Explicitly named diagnostics supplement the immutable original-study registry.
DIAGNOSTIC_METHODS = ("PD_F", "PD_A", "A_BAL", "UNION", "F2_2K", "PD_UNION")
ALL_METHODS += DIAGNOSTIC_METHODS
ADAPTIVE_METHODS = ("A", "A1", "PD_A", "A_BAL", "UNION", "PD_UNION")


def nominal_actuator_model(*, dtype: Any = jnp.float32) -> ActuatorModel:
    """Build the documented nominal model from physical, not firmware, parameters."""
    raw = load_params("cf21B_500")
    body, _ = build_cf21b_version_a_resources(dtype=dtype)
    model = make_actuator_model(
        body,
        L=raw["L"],
        thrust2torque=raw["thrust2torque"],
        mixing_matrix=raw["mixing_matrix"],
        thrust_min=raw["thrust_min"],
        thrust_max=raw["thrust_max"],
        time_constants=fit_native_time_constant()["nominal_tau"],
    )
    return jax.tree.map(lambda value: jnp.asarray(value, dtype=dtype), model)


def initial_augmented_state(body: np.ndarray, model: ActuatorModel) -> np.ndarray:
    """Use the initial nominal level-hover motor effort without any event-time reset."""
    if np.asarray(body).shape != (13,):
        raise ValueError("body state must have 13 entries before adding motor state")
    hover = float(-model.body.mass * model.body.gravity_vec[2] / 4)
    return np.concatenate((np.asarray(body, dtype=float), np.full(4, hover)))


@dataclass(frozen=True, slots=True)
class ActuatorScene:
    """Immutable physical identity and fault schedule, unavailable to the fallback actor."""

    world: NavigationWorld
    family: str
    scene_seed: int
    dynamics_cell: str
    event_time: float
    effectiveness_after: tuple[float, float, float, float]
    lag_multipliers_after: tuple[float, float, float, float]
    navigation_start: float
    recovery_time: float | None = None

    def model_at(self, when: float, nominal: ActuatorModel) -> ActuatorModel:
        """Return parameters active at one queried time; prediction receives only this result."""
        changed = when >= self.event_time - 1e-10
        if self.recovery_time is not None and when >= self.recovery_time - 1e-10:
            changed = False
        if not changed:
            return nominal
        return nominal._replace(
            effectiveness=jnp.asarray(self.effectiveness_after, dtype=nominal.effectiveness.dtype),
            time_constants=nominal.time_constants
            * jnp.asarray(self.lag_multipliers_after, dtype=nominal.time_constants.dtype),
        )

    def events(self, nominal: ActuatorModel) -> tuple[ActuatorEvent, ...]:
        """Private plant schedule; never passed into controller or learner calls."""
        events = [ActuatorEvent(self.event_time, self.model_at(self.event_time, nominal))]
        if self.recovery_time is not None:
            events.append(ActuatorEvent(self.recovery_time, nominal))
        return tuple(events)

    def metadata(self) -> dict[str, Any]:
        """Complete physical scene, with an identity independent of method/library seed."""
        values = {
            "world": self.world.metadata(),
            "family": self.family,
            "scene_seed": self.scene_seed,
            "dynamics_cell": self.dynamics_cell,
            "event_time": self.event_time,
            "effectiveness_after": self.effectiveness_after,
            "lag_multipliers_after": self.lag_multipliers_after,
            "navigation_start": self.navigation_start,
            "recovery_time": self.recovery_time,
        }
        physical = self.physical_spec()
        return {
            **values,
            "physical_spec": physical,
            "physical_world_id": physical_world_id(physical),
        }

    def physical_spec(self) -> dict[str, Any]:
        """Hash geometry, task, initial body state and faults, excluding seed labels."""
        world = self.world
        physical = {
            "robot_model": "cf21B_500_actuator_contract_v1",
            "duration_seconds": world.config.duration_seconds,
            "navigation_start_seconds": self.navigation_start,
            "reach_radius_m": world.config.reach_radius,
            "initial_motor_rule": "nominal level-hover effort, retained at every fault",
            "obstacle_motion": "mean+amplitude*sin(frequency*t+phase)",
            "collider_radius_m": 0.086,
            "collider_offset_body_m": [0.0, 0.0, 0.02],
            "ground_height_m": 0.0,
            "operational_limits": {
                "arena_lower_m": world.config.arena_lower,
                "arena_upper_m": world.config.arena_upper,
                "speed_max_mps": world.config.speed_max,
                "angular_rate_max_rps": world.config.angular_rate_max,
                "tilt_max_rad": world.config.tilt_max_radians,
            },
            **{
                name: np.asarray(getattr(world, name)).tolist()
                for name in (
                    "initial_state",
                    "waypoint_positions",
                    "obstacle_mean_centers",
                    "obstacle_amplitudes",
                    "obstacle_angular_frequencies",
                    "obstacle_phases",
                    "obstacle_radii",
                )
            },
        }
        changed = not (
            np.allclose(self.effectiveness_after, 1, rtol=0, atol=0)
            and np.allclose(self.lag_multipliers_after, 1, rtol=0, atol=0)
        )
        physical["actuator_events"] = (
            [
                {
                    "time_seconds": self.event_time,
                    "effectiveness": self.effectiveness_after,
                    "lag_multipliers": self.lag_multipliers_after,
                    "recovery_time_seconds": self.recovery_time,
                }
            ]
            if changed
            else []
        )
        return physical


def make_actuator_scene(
    seed: int, family: str, dynamics_cell: str, *, control_period: float = 0.04, dt: float = 0.02
) -> ActuatorScene:
    """Generate prespecified structured encounters or sustained waypoint navigation.

    The distribution is a development proposal until its numeric support and seed lists are
    sealed. No adaptive/frozen outcomes or initial certificate gaps enter scene generation.
    """
    if family not in {"structured", "navigation"} or dynamics_cell not in DYNAMICS_CELLS:
        raise ValueError("unknown actuator scenario family or dynamics cell")
    if not np.isclose(control_period / dt, round(control_period / dt)):
        raise ValueError("control period must be an integer number of predictor steps")
    rng = np.random.default_rng(seed)
    hold_steps = round(control_period / dt)
    event = round(2.0 / control_period) * control_period
    # Draw all factors in every cell so paired cells preserve the same geometric random stream.
    eta = float(rng.choice([0.70, 0.85]))
    lag = float(rng.choice([1.5, 2.0, 3.0]))
    pattern = int(rng.integers(4))
    affected = ((0,), (0, 1), (0, 2), (0, 1, 2, 3))[pattern]
    effectiveness = np.ones(4)
    multipliers = np.ones(4)
    if dynamics_cell in {"effectiveness", "combined"}:
        effectiveness[list(affected)] = eta
    if dynamics_cell in {"lag", "combined"}:
        multipliers[list(affected)] = lag
    if family == "structured":
        angle = rng.uniform(-np.pi, np.pi)
        direction = np.asarray([np.cos(angle), np.sin(angle), rng.uniform(-0.16, 0.16)])
        arrival = event + rng.uniform(1.6, 4.0)
        radius = rng.uniform(0.38, 0.67)
        offset = rng.uniform(-0.12, 0.12, 3)
        offset[2] *= 0.5
        sphere = IncomingSphere(
            arrival_time_seconds=float(arrival),
            direction=tuple(direction),
            speed_m_s=float(rng.uniform(1.2, 2.8)),
            radius_m=float(radius),
            crossing_offset=tuple(offset),
        )
        transverse = np.asarray([-np.sin(angle), np.cos(angle), rng.uniform(-0.12, 0.12)])
        second = IncomingSphere(
            arrival_time_seconds=float(arrival + rng.uniform(0.4, 1.4)),
            direction=tuple(transverse),
            speed_m_s=float(rng.uniform(1.2, 2.8)),
            radius_m=float(rng.uniform(0.32, 0.58)),
            crossing_offset=tuple(rng.uniform(-0.20, 0.20, 3)),
        )
        guard_distance = rng.uniform(1.0, 1.5)
        guards = (
            GuardSphere(tuple(guard_distance * transverse), float(rng.uniform(0.22, 0.34))),
            GuardSphere(tuple(-guard_distance * transverse), float(rng.uniform(0.22, 0.34))),
        )
        navigation_start = (
            math.ceil((second.arrival_time_seconds + 1.2) / control_period) * control_period
        )
        configuration = HoverEncounterConfig(
            incoming=sphere,
            additional_incoming=(second,),
            guards=guards,
            seed=seed,
            wind_onset_seconds=event,
            wind_velocity=(0.0, 0.0, 0.0),
            duration_seconds=14.0,
            navigation_start_seconds=navigation_start,
            dt=dt,
            control_interval_steps=hold_steps,
            initial_velocity=tuple(rng.uniform(-0.35, 0.35, 3)),
            waypoint_offsets=((-1.5, 0.9, 0.4), (-2.0, -0.8, 0.0)),
        )
        world = build_hover_encounter_world(configuration)
    else:
        navigation_start = 0.0
        configuration = NavigationWorldConfig(
            seed=seed,
            obstacle_count=6,
            dt=dt,
            control_interval_steps=hold_steps,
            duration_seconds=24.0,
            waypoint_count=4,
        )
        configuration.validate()
        start = np.asarray([-2.5, -0.7, 1.4])
        body = np.concatenate(
            (start, [0.0, 0.0, 0.0, 1.0], rng.uniform(-0.25, 0.25, 3), np.zeros(3))
        )
        waypoints = np.asarray(
            [[1.8, -0.8, 1.6], [2.0, 1.3, 2.2], [-1.6, 1.3, 2.7], [-2.1, -0.8, 1.4]]
        )
        means = np.asarray(
            [
                [-0.5, -0.8, 1.5],
                [1.9, 0.3, 1.9],
                [0.5, 1.3, 2.4],
                [-1.7, 0.2, 2.0],
                [0.0, -0.5, 2.1],
                [-1.4, 0.9, 1.8],
            ]
        )
        means += rng.uniform(-0.12, 0.12, means.shape)
        amplitudes = np.asarray(
            [
                [0.0, 2.8, 0.25],
                [2.2, 0.0, 0.35],
                [0.0, 2.6, 0.35],
                [2.5, 0.0, 0.25],
                [0.0, 2.8, 0.45],
                [2.3, 0.0, 0.4],
            ]
        )
        frequencies = 2 * np.pi / rng.uniform(6.0, 10.0, 6)
        arrivals = np.asarray([2.8, 6.0, 10.0, 15.0, 19.0, 21.0]) + rng.uniform(-0.6, 0.6, 6)
        phases = -frequencies * arrivals
        world = NavigationWorld(
            configuration,
            body,
            waypoints,
            means,
            amplitudes,
            frequencies,
            phases,
            rng.uniform(0.30, 0.50, 6),
        )
        for array in (
            world.initial_state,
            world.waypoint_positions,
            world.obstacle_mean_centers,
            world.obstacle_amplitudes,
            world.obstacle_angular_frequencies,
            world.obstacle_phases,
            world.obstacle_radii,
        ):
            array.setflags(write=False)
    return ActuatorScene(
        world,
        family,
        seed,
        dynamics_cell,
        event,
        tuple(effectiveness),
        tuple(multipliers),
        navigation_start,
    )


def make_actuator_plant(
    scene: ActuatorScene,
    nominal: ActuatorModel,
    *,
    level: str = "P0",
    max_step: float = 0.005,
    initial_state: np.ndarray | None = None,
    native_parameters: NativeRotorParameters | None = None,
) -> NumpyEffortPlant:
    """Create each method's complete physical plant from the common time-zero state."""
    state = (
        initial_augmented_state(scene.world.initial_state, nominal)
        if initial_state is None
        else initial_state
    )
    kwargs = {"events": scene.events(nominal), "max_step": max_step}
    if level == "P0":
        return MatchedEffortPlant(state, nominal, **kwargs)
    if level == "P1":
        return NumpyEffortPlant(state, nominal, **kwargs)
    if level == "P2":
        return NativeRotorPlant(state, nominal, native_parameters, **kwargs)
    raise ValueError("plant level must be P0, P1, or P2")


def nominal_actuator_rollout(
    state: jax.Array,
    goal: jax.Array,
    model: ActuatorModel,
    actor_config: ActuatorSkillConfig,
    *,
    acceleration_limit: float = 1.2,
    position_gain: float = 2.0,
    velocity_gain: float = 2.8,
) -> ActuatorRollouts:
    """Shared goal controller, held at the same cadence as every fallback command."""
    from crazyflow.safety.da_plcbf.actuator_dynamics import effort_lag_step

    cfg = actor_config

    def boundary(carry: tuple[jax.Array, jax.Array], index: jax.Array) -> tuple[tuple, tuple]:
        y, old_command = carry

        def command_at_boundary(_: None) -> jax.Array:
            acceleration = position_gain * (goal - y[:3]) - velocity_gain * y[7:10]
            norm = jnp.sqrt(jnp.sum(acceleration * acceleration) + 1e-12)
            acceleration = acceleration * jnp.minimum(1.0, acceleration_limit / norm)
            return acceleration_to_actuator_command(acceleration, y, model, cfg).command

        command = jax.lax.cond(
            index % cfg.control_interval_steps == 0,
            command_at_boundary,
            lambda _: old_command,
            operand=None,
        )
        following = effort_lag_step(y, command, model, cfg.dt, substeps=cfg.plant_substeps)
        return (following, command), (following, command)

    _, (future, commands) = jax.lax.scan(boundary, (state, state[13:]), jnp.arange(cfg.horizon))
    states = jnp.concatenate((state[None], future))
    valid = jnp.all(jnp.isfinite(states)) & jnp.all(jnp.isfinite(commands))
    return ActuatorRollouts(states[None], commands[None], valid[None])


def emergency_actuator_command(
    state: jax.Array, model: ActuatorModel, actor_config: ActuatorSkillConfig
) -> jax.Array:
    """Common obstacle/goal-free velocity brake with the same physical adapter."""
    acceleration = -2.0 * state[7:10]
    norm = jnp.sqrt(jnp.sum(acceleration * acceleration) + 1e-12)
    acceleration = acceleration * jnp.minimum(1.0, 4.0 / norm)
    return acceleration_to_actuator_command(acceleration, state, model, actor_config).command


class ActuatorControllerFunctions(NamedTuple):
    controller: Callable
    nominal: Callable
    emergency: Callable
    candidates: Callable


def build_actuator_controller(
    spec: SkillLibrarySpec,
    actor_config: ActuatorSkillConfig,
    filter_config: ActuatorFilterConfig = ActuatorFilterConfig(),
    *,
    nominal_acceleration_limit: float = 1.2,
    frozen_library: tuple[SkillActorParams, SkillLibrarySpec, ActuatorSkillConfig] | None = None,
) -> ActuatorControllerFunctions:
    """Build once and pass model/parameters/state/obstacle clock as explicit immutable inputs."""
    actor_config.validate()
    filter_config.validate()
    if (
        actor_config.dt != filter_config.dt
        or actor_config.horizon != filter_config.horizon
        or actor_config.control_interval_steps != filter_config.command_hold_steps
    ):
        raise ValueError("actor, prediction, and physical command-hold clocks must match")
    # F0/F1 alter only the diagnostic fallback mapping. Nominal and emergency use the
    # same strongest causal adapter in every method, as in the accepted wind comparison.
    common_config = replace(actor_config, adapter_mode="F2", model_compensation=True)
    if frozen_library is not None:
        _, _, frozen_config = frozen_library
        frozen_config.validate()
        for name in ("dt", "horizon", "control_interval_steps", "plant_substeps", "adapter_mode"):
            if getattr(frozen_config, name) != getattr(actor_config, name):
                raise ValueError(f"frozen and current library must share {name}")

    def nominal(y: jax.Array, goal: jax.Array, model: ActuatorModel) -> ActuatorRollouts:
        return nominal_actuator_rollout(
            y, goal, model, common_config, acceleration_limit=nominal_acceleration_limit
        )

    def candidates(
        y: jax.Array, params: SkillActorParams, goal: jax.Array, model: ActuatorModel
    ) -> ActuatorRollouts:
        mission = nominal(y, goal, model)
        library = rollout_actuator_skill_library(params, spec, y, model, actor_config)
        if frozen_library is not None:
            frozen_params, frozen_spec, frozen_config = frozen_library
            core = rollout_actuator_skill_library(
                frozen_params, frozen_spec, y, model, frozen_config
            )
            return ActuatorRollouts(
                jnp.concatenate((mission.states, core.states, library.states)),
                jnp.concatenate((mission.commands, core.commands, library.commands)),
                jnp.concatenate((mission.valid, core.valid, library.valid)),
            )
        return ActuatorRollouts(
            jnp.concatenate((mission.states, library.states)),
            jnp.concatenate((mission.commands, library.commands)),
            jnp.concatenate((mission.valid, library.valid)),
        )

    def emergency(y: jax.Array, model: ActuatorModel) -> jax.Array:
        return emergency_actuator_command(y, model, common_config)

    def controller(
        y: jax.Array,
        params: SkillActorParams,
        model: ActuatorModel,
        obstacles: RuntimeObstacleTrajectories,
        safety: RigidBodySafetySet,
        previous: jax.Array,
        goal: jax.Array,
    ) -> Any:
        return actuator_plcbf_step(
            y,
            lambda initial, point: candidates(initial, params, goal, point),
            model,
            obstacles,
            safety,
            emergency(y, model),
            previous,
            filter_config,
        )

    return ActuatorControllerFunctions(
        jax.jit(controller), jax.jit(nominal), jax.jit(emergency), jax.jit(candidates)
    )


@dataclass(frozen=True, slots=True)
class ActuatorObservationConfig:
    """Controlled mismatch, with one common absolute-time noise tape per physical world."""

    parameter_delay_seconds: float = 0.0
    effectiveness_bias: float = 0.0
    lag_scale: float = 1.0
    position_noise_m: float = 0.0
    velocity_noise_mps: float = 0.0
    attitude_noise_rad: float = 0.0
    rate_noise_rps: float = 0.0
    motor_noise_N: float = 0.0
    obstacle_position_bias_m: float = 0.0
    noise_sample_seconds: float = 0.001

    def validate(self) -> None:
        """Reject invalid noise, delay, or estimator settings."""
        for key, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.lag_scale <= 0 or self.noise_sample_seconds <= 0:
            raise ValueError("lag scale and noise sample period must be positive")
        if any(
            getattr(self, key) < 0
            for key in (
                "parameter_delay_seconds",
                "position_noise_m",
                "velocity_noise_mps",
                "attitude_noise_rad",
                "rate_noise_rps",
                "motor_noise_N",
            )
        ):
            raise ValueError("delay and standard deviations must be nonnegative")


def observe_actuator_state(
    actual: np.ndarray, time_seconds: float, scene_seed: int, config: ActuatorObservationConfig
) -> np.ndarray:
    """Indexed noise depends on absolute time/channel, never on method or query order."""
    config.validate()
    index = int(math.floor((time_seconds + 1e-10) / config.noise_sample_seconds))
    rng = np.random.default_rng(np.random.SeedSequence([scene_seed, index, 918273]))
    tape = rng.standard_normal(16)
    observed = np.asarray(actual, dtype=float).copy()
    observed[:3] += config.position_noise_m * tape[:3]
    observed[7:10] += config.velocity_noise_mps * tape[3:6]
    observed[10:13] += config.rate_noise_rps * tape[6:9]
    delta = Rotation.from_rotvec(config.attitude_noise_rad * tape[9:12])
    observed[3:7] = (Rotation.from_quat(observed[3:7]) * delta).as_quat()
    observed[13:] += config.motor_noise_N * tape[12:16]
    return observed


def observed_actuator_model(
    scene: ActuatorScene, when: float, nominal: ActuatorModel, config: ActuatorObservationConfig
) -> ActuatorModel:
    """Controlled current/past parameter observation; no future fault information."""
    config.validate()
    model = scene.model_at(max(0.0, when - config.parameter_delay_seconds), nominal)
    return model._replace(
        effectiveness=jnp.clip(model.effectiveness + config.effectiveness_bias, 1e-3, 1.0),
        time_constants=model.time_constants * config.lag_scale,
    )
