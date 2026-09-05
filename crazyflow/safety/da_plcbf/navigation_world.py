"""Deterministic exogenous 3D navigation worlds and separately tracked task progress.

Only task control and safety evaluation consume this module. The waypoint queue, obstacle
trajectories, and encounter metrics are never learner inputs or training targets. Periodic
obstacles have analytic positions and time derivatives; they do not react to a method's route.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.continuous_version_a import RuntimeObstacleTrajectories
from crazyflow.safety.da_plcbf.rigid_payload import CenteredRigidPayload
from crazyflow.safety.da_plcbf.version_a_barriers import RigidBodySafetySet, VersionAModel

# cf21B_500.xml: col_sphere has radius .086 m and center (0, 0, .02) in the body.
# A sphere at the modeled CoM must enclose that offset collider at every attitude.
CF21B_BODY_ORIGIN_ENCLOSURE_M = 0.106


@dataclass(frozen=True, slots=True)
class WindEvent:
    """Set the absolute world wind at a control boundary, including subsequent holds."""

    time_seconds: float
    velocity: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PayloadEvent:
    """Prescribed centered attachment; mass_fraction refers to the original bare vehicle."""

    time_seconds: float
    mass_fraction: float = 0.25
    half_extents: tuple[float, float, float] = (0.025, 0.025, 0.025)


@dataclass(frozen=True, slots=True)
class NavigationWorldConfig:
    seed: int = 0
    obstacle_count: int = 8
    dt: float = 0.02
    control_interval_steps: int = 2
    duration_seconds: float = 40.0
    waypoint_count: int = 8
    reach_radius: float = 0.4
    moving_obstacles: bool = True
    wind_events: tuple[WindEvent, ...] = ()
    payload_events: tuple[PayloadEvent, ...] = ()
    ego_radius: float = CF21B_BODY_ORIGIN_ENCLOSURE_M
    obstacle_clearance: float = 0.15
    arena_lower: tuple[float, float, float] = (-5.0, -4.0, 0.15)
    arena_upper: tuple[float, float, float] = (5.0, 4.0, 4.0)
    speed_max: float = 3.5
    angular_rate_max: float = 12.0
    tilt_max_radians: float = 0.9

    @property
    def control_period(self) -> float:
        return self.dt * self.control_interval_steps

    def validate(self) -> None:
        for name in (
            "dt",
            "duration_seconds",
            "reach_radius",
            "ego_radius",
            "speed_max",
            "angular_rate_max",
            "tilt_max_radians",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite")
        for name, minimum in (
            ("obstacle_count", 0),
            ("waypoint_count", 2),
            ("control_interval_steps", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.moving_obstacles, bool):
            raise ValueError("moving_obstacles must be boolean")
        lower, upper = np.asarray(self.arena_lower), np.asarray(self.arena_upper)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.all(np.isfinite([lower, upper]))
            or np.any(upper - lower < (8.0, 6.0, 3.4))
        ):
            raise ValueError("arena must provide at least 8 by 6 by 3.4 metres")
        if not math.isfinite(self.obstacle_clearance) or self.obstacle_clearance < 0:
            raise ValueError("obstacle_clearance must be nonnegative finite")
        _control_boundary(self.duration_seconds, self.control_period)
        for events in (self.wind_events, self.payload_events):
            previous = -math.inf
            for event in events:
                _control_boundary(event.time_seconds, self.control_period)
                if not previous < event.time_seconds < self.duration_seconds:
                    raise ValueError("events must be strictly ordered within the episode")
                previous = event.time_seconds
                if isinstance(event, WindEvent):
                    value = np.asarray(event.velocity)
                    if value.shape != (3,) or not np.all(np.isfinite(value)):
                        raise ValueError("wind event requires a finite 3-vector")
                else:
                    CenteredRigidPayload(event.mass_fraction, event.half_extents).validate()


def _control_boundary(time: float, period: float) -> None:
    if (
        not math.isfinite(time)
        or time < 0
        or not math.isclose(time / period, round(time / period), abs_tol=1e-8)
    ):
        raise ValueError("event and duration times must lie on declared control boundaries")


class NavigationDynamics(NamedTuple):
    model: VersionAModel
    ego_radius: float
    payload_mass_kg: float
    payload_half_extents: tuple[float, float, float] | None


@dataclass(frozen=True, slots=True)
class NavigationWorld:
    """Read-only generated world. Query order cannot change any subsequent world sample."""

    config: NavigationWorldConfig
    initial_state: np.ndarray
    waypoint_positions: np.ndarray
    obstacle_mean_centers: np.ndarray
    obstacle_amplitudes: np.ndarray
    obstacle_angular_frequencies: np.ndarray
    obstacle_phases: np.ndarray
    obstacle_radii: np.ndarray

    def obstacle_kinematics(self, times: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (..., O, 3) analytic positions and d(position)/d(absolute time)."""
        times = np.asarray(times, dtype=float)
        if not np.all(np.isfinite(times)):
            raise ValueError("obstacle query times must be finite")
        angle = (times[..., None] * self.obstacle_angular_frequencies + self.obstacle_phases)[
            ..., None
        ]
        centers = self.obstacle_mean_centers + self.obstacle_amplitudes * np.sin(angle)
        velocities = (
            self.obstacle_amplitudes * self.obstacle_angular_frequencies[:, None] * np.cos(angle)
        )
        return centers, velocities

    def obstacle_prediction(
        self,
        time_seconds: float,
        dt: float | None = None,
        horizon: int = 60,
        *,
        dtype: jnp.dtype = jnp.float32,
    ) -> RuntimeObstacleTrajectories:
        """Integrating at dt retains every node; commands may use a longer held period."""
        dt = self.config.dt if dt is None else dt
        if not math.isfinite(dt) or dt <= 0 or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("prediction needs positive dt and integer horizon >= 1")
        centers, velocities = self.obstacle_kinematics(time_seconds + dt * np.arange(horizon + 1))
        return RuntimeObstacleTrajectories(
            jnp.asarray(centers, dtype=dtype),
            jnp.asarray(self.obstacle_radii, dtype=dtype),
            jnp.ones(centers.shape[:-1], dtype=bool),
            jnp.asarray(velocities, dtype=dtype),
        )

    def wind_at(self, time_seconds: float) -> np.ndarray:
        wind = np.zeros(3)
        for event in self.config.wind_events:
            if time_seconds >= event.time_seconds - 1e-10:
                wind = np.asarray(event.velocity, dtype=float)
        return wind

    def dynamics_at(self, time_seconds: float, base_model: VersionAModel) -> NavigationDynamics:
        """Right-continuous events, shared by plant/predictor/allocator on the same boundary.

        Only scheduled model parameters change. Existing allocation geometry is unchanged for
        these centered attachments. Repeated calls always start from the bare base_model.
        """
        model = base_model._replace(
            wind_velocity=jnp.asarray(self.wind_at(time_seconds), dtype=base_model.mass.dtype)
        )
        radius, added_mass, extents = self.config.ego_radius, 0.0, None
        base_mass = float(np.asarray(base_model.mass))
        for event in self.config.payload_events:
            if time_seconds >= event.time_seconds - 1e-10:
                payload = CenteredRigidPayload(base_mass * event.mass_fraction, event.half_extents)
                model = payload.apply(model)
                radius = payload.enclosing_radius(radius)
                added_mass += payload.mass
                extents = (
                    event.half_extents
                    if extents is None
                    else tuple(np.maximum(extents, event.half_extents))
                )
        return NavigationDynamics(model, radius, added_mass, extents)

    def safety_limits(
        self, time_seconds: float = 0.0, *, dtype: jnp.dtype = jnp.float32
    ) -> RigidBodySafetySet:
        centers, _ = self.obstacle_kinematics(time_seconds)
        cfg = self.config
        return RigidBodySafetySet(
            jnp.asarray(centers, dtype=dtype),
            jnp.asarray(self.obstacle_radii, dtype=dtype),
            jnp.ones(cfg.obstacle_count, dtype=bool),
            jnp.asarray(cfg.arena_lower, dtype=dtype),
            jnp.asarray(cfg.arena_upper, dtype=dtype),
            jnp.asarray(cfg.speed_max, dtype=dtype),
            jnp.asarray(cfg.angular_rate_max, dtype=dtype),
            jnp.asarray(cfg.tilt_max_radians, dtype=dtype),
        )

    def metadata(self) -> dict[str, object]:
        return {
            "schema": "exogenous_navigation_world_v1",
            "config": asdict(self.config),
            "trajectory": "center(t) = mean + amplitude * sin(omega*t + phase)",
            "event_semantics": "right-continuous prescribed changes at command boundaries",
            "obstacle_masks": "all obstacles active at all prediction nodes; no teleports",
            "ego_enclosure": {
                "radius_m": self.config.ego_radius,
                "cf21b_asset_collider_radius_m": 0.086,
                "cf21b_asset_collider_body_offset_m": [0.0, 0.0, 0.02],
                "encloses_asset_collider": self.config.ego_radius
                >= CF21B_BODY_ORIGIN_ENCLOSURE_M - 1e-9,
                "scope": (
                    "body-origin spherical safety envelope; envelope intersection is distinct "
                    "from a measured MuJoCo collider contact"
                ),
            },
            **{
                name: getattr(self, name).tolist()
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


def build_navigation_world(
    config: NavigationWorldConfig = NavigationWorldConfig(),
) -> NavigationWorld:
    """Generate a room-scale alternating-height route with lateral/vertical crossings.

    Obstacles oscillate across route segments with sufficient surrounding free space for
    deviations. This construction does not prove feasibility; encounter metrics and independent
    route witnesses must establish the severity/authority of a particular evaluated scene.
    """
    config.validate()
    rng = np.random.default_rng(config.seed)
    lower, upper = np.asarray(config.arena_lower), np.asarray(config.arena_upper)
    center = (lower + upper) / 2
    halfspan = (upper - lower) / 2
    initial = center + halfspan * np.asarray((-0.74, -0.67, -0.43))
    route_shape = np.asarray(
        ((0.70, -0.60, 0.38), (0.63, 0.62, -0.37), (-0.65, 0.57, 0.35), (-0.69, -0.63, -0.38))
    )
    waypoints = center + halfspan * route_shape[np.arange(config.waypoint_count) % 4]
    waypoints += rng.uniform(-0.16, 0.16, waypoints.shape)
    starts = np.vstack((initial, waypoints[:-1]))
    means, amplitudes, frequencies, phases, radii = [], [], [], [], []
    for index in range(config.obstacle_count):
        segment = index % config.waypoint_count
        lane = index // config.waypoint_count
        fraction = (0.46, 0.70, 0.25, 0.85)[lane % 4] + rng.uniform(-0.04, 0.04)
        mean = starts[segment] + fraction * (waypoints[segment] - starts[segment])
        direction = waypoints[segment] - starts[segment]
        transverse = np.asarray((-direction[1], direction[0], 0.0))
        transverse /= np.linalg.norm(transverse)
        if index % 3 == 1:
            amplitude = np.asarray((0.0, 0.0, rng.uniform(0.55, 0.8)))
        else:
            amplitude = transverse * rng.uniform(0.75, 1.25)
        radius = rng.uniform(0.20, 0.30)
        available = np.minimum(mean - lower, upper - mean) - radius - 0.12
        amplitude = np.sign(amplitude) * np.minimum(np.abs(amplitude), available)
        if not config.moving_obstacles:
            # Keep the same seed/world means and static phases, only remove motion.
            amplitude *= 0
        means.append(mean)
        amplitudes.append(amplitude)
        frequencies.append(2 * math.pi / rng.uniform(5.0, 8.0))
        phases.append(rng.uniform(-math.pi, math.pi))
        radii.append(radius)
    arrays = [
        np.asarray((*initial, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0), dtype=float),
        waypoints,
        np.asarray(means).reshape(-1, 3),
        np.asarray(amplitudes).reshape(-1, 3),
        np.asarray(frequencies),
        np.asarray(phases),
        np.asarray(radii),
    ]
    for array in arrays:
        array.setflags(write=False)
    world = NavigationWorld(config, *arrays)
    centers, _ = world.obstacle_kinematics(0.0)
    if np.any(
        np.linalg.norm(centers - initial, axis=-1)
        <= world.obstacle_radii + config.ego_radius + config.obstacle_clearance
    ):
        raise ValueError("generated world begins inside an inflated obstacle")
    return world


@dataclass(frozen=True, slots=True)
class WaypointProgress:
    """Per-method progress over a shared immutable queue, censored on termination."""

    completed: int = 0
    termination: str | None = None
    termination_time_seconds: float | None = None
    arrival_times_seconds: tuple[float, ...] = ()

    def active_goal(self, world: NavigationWorld) -> np.ndarray:
        return world.waypoint_positions[min(self.completed, len(world.waypoint_positions) - 1)]


def advance_waypoints(
    world: NavigationWorld,
    progress: WaypointProgress,
    position: np.ndarray,
    time_seconds: float,
    *,
    physical_collision: bool = False,
    navigation_enabled: bool = True,
) -> WaypointProgress:
    """Update once at a control boundary; collision takes precedence over arrival/timeout.

    The caller must detect physical collision over the entire preceding integration interval.
    No post-collision or post-timeout arrival is ever credited. Reaching the radius is the
    declared rule; settling is intentionally not required at intermediate waypoints.
    """
    if progress.termination is not None:
        return progress
    if physical_collision:
        return WaypointProgress(
            progress.completed, "physical_collision", time_seconds, progress.arrival_times_seconds
        )
    if time_seconds >= world.config.duration_seconds - 1e-10:
        return WaypointProgress(
            progress.completed, "timeout", time_seconds, progress.arrival_times_seconds
        )
    if not navigation_enabled:
        return progress
    if (
        np.linalg.norm(np.asarray(position) - progress.active_goal(world))
        <= world.config.reach_radius
    ):
        completed = progress.completed + 1
        done = completed == len(world.waypoint_positions)
        return WaypointProgress(
            completed,
            "completed" if done else None,
            time_seconds if done else None,
            (*progress.arrival_times_seconds, float(time_seconds)),
        )
    return progress


def nominal_encounter_metrics(
    nominal_positions: np.ndarray,
    obstacles: RuntimeObstacleTrajectories,
    *,
    dt: float,
    ego_radius: float,
    obstacle_clearance: float,
    close_approach_margin: float = 0.2,
) -> dict[str, float | int | bool | None]:
    """Measured nominal threat, using relative swept chords between integration nodes.

    TTC is the first crossing of the requested spherical clearance on these recorded chords.
    Accelerated obstacle arcs between nodes are not certified by the chord approximation.
    ``threatening_obstacle_count`` counts distinct obstacles crossing the requested shell;
    ``peak_simultaneous_threats`` counts shells breached at the same prediction interval.
    """
    positions, centers = np.asarray(nominal_positions), np.asarray(obstacles.centers)
    if positions.shape != (len(centers), 3):
        raise ValueError("nominal positions must match obstacle prediction nodes, including t=0")
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError("dt must be positive finite")
    active = np.asarray(obstacles.mask, dtype=bool)
    radius = np.asarray(obstacles.radii) + ego_radius + obstacle_clearance
    if centers.shape[1] == 0:
        return {
            "minimum_predicted_clearance_m": None,
            "closest_time_to_contact_seconds": None,
            "threatening_obstacle_count": 0,
            "close_approach_obstacle_count": 0,
            "peak_simultaneous_threats": 0,
            "nominal_blocked": False,
        }
    relative = positions[:, None] - centers
    node_clearance = np.where(active, np.linalg.norm(relative, axis=-1) - radius, np.inf)
    node_threats = node_clearance <= 0
    starts, delta = relative[:-1], np.diff(relative, axis=0)
    length2 = np.sum(delta * delta, axis=-1)
    dot = np.sum(starts * delta, axis=-1)
    fraction = np.clip(-dot / np.maximum(length2, 1e-30), 0, 1)
    distances = np.linalg.norm(starts + fraction[..., None] * delta, axis=-1)
    interval_active = active[:-1] & active[1:]
    clearance = np.where(interval_active, distances - radius, np.inf)
    threats = clearance <= 0
    c = np.sum(starts * starts, axis=-1) - radius**2
    discriminant = dot**2 - length2 * c
    first_fraction = (-dot - np.sqrt(np.maximum(discriminant, 0))) / np.maximum(length2, 1e-30)
    first_fraction = np.where(c <= 0, 0, first_fraction)
    contact = np.where(
        threats, (np.arange(len(starts))[:, None] + np.clip(first_fraction, 0, 1)) * dt, np.inf
    )
    node_contact = np.where(node_threats, np.arange(len(positions))[:, None] * dt, np.inf)
    minimum = float(min(np.min(clearance), np.min(node_clearance)))
    ttc = float(min(np.min(contact), np.min(node_contact)))
    threatening_obstacles = np.any(threats, axis=0) | np.any(node_threats, axis=0)
    close_obstacles = np.any(clearance <= close_approach_margin, axis=0) | np.any(
        node_clearance <= close_approach_margin, axis=0
    )
    return {
        "minimum_predicted_clearance_m": minimum if math.isfinite(minimum) else None,
        "closest_time_to_contact_seconds": ttc if math.isfinite(ttc) else None,
        "threatening_obstacle_count": int(np.count_nonzero(threatening_obstacles)),
        "close_approach_obstacle_count": int(np.count_nonzero(close_obstacles)),
        "peak_simultaneous_threats": int(
            max(
                np.max(np.count_nonzero(threats, axis=1)),
                np.max(np.count_nonzero(node_threats, axis=1)),
            )
        ),
        "nominal_blocked": bool(np.any(threatening_obstacles)),
    }
