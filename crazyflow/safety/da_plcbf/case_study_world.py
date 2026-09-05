"""Prescribed hover encounters and independent audits of recorded collision geometry.

These helpers only define experiments and evaluate recorded trajectories. Neither obstacle
geometry nor an audit outcome is a fallback-learner input. The encounter is an ordinary
``NavigationWorld`` with analytic absolute-time motion, so prediction and execution use the
same obstacle function. A controlled branch records the absolute time of its initial state;
the caller must retain that time instead of restarting the obstacle clock.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from crazyflow.safety.da_plcbf.navigation_world import (
    CF21B_BODY_ORIGIN_ENCLOSURE_M,
    NavigationWorld,
    NavigationWorldConfig,
    WindEvent,
)

CF21B_XML_COLLIDER_RADIUS_M = 0.086
CF21B_XML_COLLIDER_OFFSET_BODY_M = (0.0, 0.0, 0.02)


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite three-vector")
    return result


@dataclass(frozen=True, slots=True)
class IncomingSphere:
    """One smooth passage through hover + offset at the specified absolute arrival time.

    ``direction`` is the direction of motion at arrival, normalized by the builder. The
    analytic sinusoid has the requested arrival speed and zero acceleration at arrival. Its
    large, finite amplitude approximates a linear crossing over a short encounter, without
    introducing a new motion law into the existing prediction/recording pipeline. The
    obstacle may originate outside the ego flight arena; it is not clipped at that boundary.
    """

    arrival_time_seconds: float = 8.0
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    speed_m_s: float = 2.0
    radius_m: float = 0.5
    crossing_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude_m: float = 20.0

    def validate(self, duration_seconds: float) -> None:
        """Reject invalid geometry and repeated center passages within the episode."""
        for name in ("arrival_time_seconds", "speed_m_s", "radius_m", "amplitude_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"incoming {name} must be positive finite")
        if not 0 < self.arrival_time_seconds < duration_seconds:
            raise ValueError("incoming arrival must lie strictly inside the episode")
        if np.linalg.norm(_vector(self.direction, "incoming direction")) <= 1e-12:
            raise ValueError("incoming direction must be nonzero")
        _vector(self.crossing_offset, "incoming crossing offset")
        half_period = math.pi * self.amplitude_m / self.speed_m_s
        if (
            self.arrival_time_seconds - half_period >= 0
            or self.arrival_time_seconds + half_period <= duration_seconds
        ):
            raise ValueError("incoming amplitude permits an additional passage in this episode")


@dataclass(frozen=True, slots=True)
class GuardSphere:
    """A static, always-present obstacle at hover position + offset."""

    offset: tuple[float, float, float]
    radius_m: float = 0.3

    def validate(self) -> None:
        """Validate the fixed obstacle geometry."""
        _vector(self.offset, "guard offset")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("guard radius must be positive finite")


@dataclass(frozen=True, slots=True)
class HoverEncounterConfig:
    """A small persistent-wind world with explicit absolute timing and a short final task."""

    incoming: IncomingSphere = IncomingSphere()
    guards: tuple[GuardSphere, ...] = ()
    seed: int = 0
    wind_onset_seconds: float = 3.0
    wind_velocity: tuple[float, float, float] = (1.6, 0.8, 0.0)
    duration_seconds: float = 16.0
    navigation_start_seconds: float = 11.0
    hover_position: tuple[float, float, float] = (0.0, 0.0, 1.4)
    waypoint_offsets: tuple[tuple[float, float, float], ...] = ((-1.5, 1.0, 0.3), (-2.0, -1.0, 0.0))
    arena_lower: tuple[float, float, float] = (-5.0, -4.0, 0.15)
    arena_upper: tuple[float, float, float] = (5.0, 4.0, 4.0)
    dt: float = 0.02
    control_interval_steps: int = 2
    ego_radius: float = CF21B_BODY_ORIGIN_ENCLOSURE_M
    obstacle_clearance: float = 0.15
    reach_radius: float = 0.4

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HoverEncounterConfig:
        """Reconstruct typed immutable specifications from their JSON representation."""
        values = dict(data)
        if "incoming" in values:
            incoming = dict(values["incoming"])
            for key in ("direction", "crossing_offset"):
                if key in incoming:
                    incoming[key] = tuple(incoming[key])
            values["incoming"] = IncomingSphere(**incoming)
        if "guards" in values:
            values["guards"] = tuple(
                GuardSphere(tuple(guard["offset"]), guard.get("radius_m", 0.3))
                for guard in values["guards"]
            )
        for key in ("wind_velocity", "hover_position", "arena_lower", "arena_upper"):
            if key in values:
                values[key] = tuple(values[key])
        if "waypoint_offsets" in values:
            values["waypoint_offsets"] = tuple(tuple(row) for row in values["waypoint_offsets"])
        return cls(**values)

    def world_config(self) -> NavigationWorldConfig:
        """Build the unchanged controller/plant world configuration and validate it."""
        config = NavigationWorldConfig(
            seed=self.seed,
            obstacle_count=1 + len(self.guards),
            dt=self.dt,
            control_interval_steps=self.control_interval_steps,
            duration_seconds=self.duration_seconds,
            waypoint_count=len(self.waypoint_offsets),
            reach_radius=self.reach_radius,
            wind_events=(WindEvent(self.wind_onset_seconds, self.wind_velocity),),
            ego_radius=self.ego_radius,
            obstacle_clearance=self.obstacle_clearance,
            arena_lower=self.arena_lower,
            arena_upper=self.arena_upper,
        )
        config.validate()
        self.incoming.validate(self.duration_seconds)
        for guard in self.guards:
            guard.validate()
        if self.ego_radius < CF21B_BODY_ORIGIN_ENCLOSURE_M - 1e-9:
            raise ValueError("case study must enclose the offset cf21B XML collider")
        if (
            not math.isfinite(self.navigation_start_seconds)
            or not self.wind_onset_seconds <= self.navigation_start_seconds < self.duration_seconds
            or not math.isclose(
                self.navigation_start_seconds / config.control_period,
                round(self.navigation_start_seconds / config.control_period),
                abs_tol=1e-8,
            )
        ):
            raise ValueError("navigation start must be a control boundary after wind onset")
        hover = _vector(self.hover_position, "hover position")
        for point in (
            hover,
            *(hover + _vector(v, "waypoint offset") for v in self.waypoint_offsets),
        ):
            if np.any(point <= np.asarray(self.arena_lower) + self.ego_radius) or np.any(
                point >= np.asarray(self.arena_upper) - self.ego_radius
            ):
                raise ValueError("hover and task waypoints must lie inside the ego flight arena")
        return config


@dataclass(frozen=True, slots=True)
class HoverEncounterWorld(NavigationWorld):
    """Ordinary analytic world with provenance for a common-state controlled branch."""

    case_study_config: HoverEncounterConfig
    initial_state_time_seconds: float = 0.0

    def metadata(self) -> dict[str, object]:
        """Retain full legacy motion metadata and explicitly record the branch clock."""
        return {
            **NavigationWorld.metadata(self),
            "case_study_config": asdict(self.case_study_config),
            "initial_state_time_seconds": self.initial_state_time_seconds,
            "case_study_time_convention": "all obstacle and dynamics queries use absolute time",
            "incoming_motion_scope": (
                "globally bounded analytic passage; obstacle may originate outside ego arena"
            ),
        }


def build_hover_encounter_world(
    config: HoverEncounterConfig = HoverEncounterConfig(),
    *,
    initial_state: np.ndarray | None = None,
    initial_time_seconds: float = 0.0,
) -> HoverEncounterWorld:
    """Build a reproducible encounter; never reset obstacle phases for a controlled branch.

    The existing navigation runner starts at zero and therefore supports the continuous
    ``initial_time_seconds=0`` experiment. A branch runner must explicitly start its clock
    at ``world.initial_state_time_seconds``. Its supplied state may include the measured
    wind-compensating attitude and velocity of an authenticated obstacle-free prefix.
    """
    world_config = config.world_config()
    if (
        not math.isfinite(initial_time_seconds)
        or not 0 <= initial_time_seconds < config.duration_seconds
        or not math.isclose(
            initial_time_seconds / world_config.control_period,
            round(initial_time_seconds / world_config.control_period),
            abs_tol=1e-8,
        )
    ):
        raise ValueError("initial state time must be a control boundary inside the episode")
    hover = np.asarray(config.hover_position, dtype=float)
    state = (
        np.asarray((*hover, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0), dtype=float)
        if initial_state is None
        else np.array(initial_state, dtype=float, copy=True)
    )
    if state.shape != (13,) or not np.isfinite(state).all():
        raise ValueError("initial state must be a finite 13-vector")
    if abs(np.linalg.norm(state[3:7]) - 1) > 2e-4:
        raise ValueError("initial state quaternion must be normalized")
    sphere = config.incoming
    direction = np.asarray(sphere.direction) / np.linalg.norm(sphere.direction)
    omega = sphere.speed_m_s / sphere.amplitude_m
    count = world_config.obstacle_count
    means = np.asarray(
        [hover + np.asarray(sphere.crossing_offset)]
        + [hover + np.asarray(guard.offset) for guard in config.guards]
    )
    amplitudes = np.zeros((count, 3))
    amplitudes[0] = direction * sphere.amplitude_m
    frequencies, phases = np.zeros(count), np.zeros(count)
    frequencies[0], phases[0] = omega, -omega * sphere.arrival_time_seconds
    radii = np.asarray([sphere.radius_m, *(guard.radius_m for guard in config.guards)])
    arrays = (
        state,
        hover + np.asarray(config.waypoint_offsets),
        means,
        amplitudes,
        frequencies,
        phases,
        radii,
    )
    for array in arrays:
        array.setflags(write=False)
    world = HoverEncounterWorld(world_config, *arrays, config, float(initial_time_seconds))
    centers, _ = world.obstacle_kinematics(initial_time_seconds)
    if np.any(
        np.linalg.norm(centers - state[:3], axis=-1)
        <= radii + config.ego_radius + config.obstacle_clearance
    ):
        raise ValueError("case study initial state must be outside every inflated obstacle")
    return world


def _sphere_sweep(
    times: np.ndarray,
    positions: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    added_radius: float,
    curvature_error: np.ndarray,
) -> dict[str, Any]:
    if len(radii) == 0:
        return {
            "minimum_clearance_m": None,
            "minimum_clearance_lower_bound_m": None,
            "minimum_clearance_upper_bound_m": None,
            "first_chord_intersection_time_seconds": None,
            "first_intersection_obstacle": None,
            "intersection_classification": "no_obstacles",
        }
    relative = positions[:, None] - centers
    start, delta = relative[:-1], np.diff(relative, axis=0)
    length2 = np.sum(delta * delta, axis=-1)
    dot = np.sum(start * delta, axis=-1)
    fractions = np.clip(-dot / np.maximum(length2, 1e-30), 0, 1)
    clearance = np.linalg.norm(start + fractions[..., None] * delta, axis=-1) - (
        radii + added_radius
    )
    c = np.sum(start * start, axis=-1) - (radii + added_radius) ** 2
    discriminant = dot**2 - length2 * c
    root = (-dot - np.sqrt(np.maximum(discriminant, 0))) / np.maximum(length2, 1e-30)
    root = np.where(c <= 0, 0, np.where((length2 > 1e-24) & (discriminant >= 0), root, np.inf))
    root = np.where((root >= 0) & (root <= 1), root, np.inf)
    crossing_times = times[:-1, None] + root * np.diff(times)[:, None]
    first = None
    obstacle = None
    if np.isfinite(crossing_times).any():
        interval, obstacle = np.unravel_index(np.argmin(crossing_times), crossing_times.shape)
        first, obstacle = float(crossing_times[interval, obstacle]), int(obstacle)
    lower, upper = (
        float(np.min(clearance - curvature_error)),
        float(np.min(clearance + curvature_error)),
    )
    classification = (
        "separated_under_recorded_state_interpolation"
        if lower > 0
        else "intersecting_under_recorded_state_interpolation"
        if upper < 0
        else "unresolved_at_interpolation_error_bound"
    )
    return {
        "minimum_clearance_m": float(np.min(clearance)),
        "minimum_clearance_lower_bound_m": lower,
        "minimum_clearance_upper_bound_m": upper,
        "maximum_chord_curvature_error_bound_m": float(np.max(curvature_error)),
        "first_chord_intersection_time_seconds": first,
        "first_intersection_obstacle": obstacle,
        "intersection_classification": classification,
    }


def audit_recorded_collider_clearance(
    world: NavigationWorld,
    times: np.ndarray,
    full_states: np.ndarray,
    *,
    max_substep_seconds: float = 0.001,
    ground_height: float = 0.0,
) -> dict[str, Any]:
    """Separate safety-shell, body-envelope and rotated XML-sphere intersections.

    Body positions use linear interpolation between recorded integration states; attitude
    uses shortest-arc SLERP. Obstacles are queried analytically on the same absolute clock.
    Swept chords at <=1 ms include an explicit curvature bound for the rotating collider
    offset and obstacle arcs. Bounds apply to this recorded-state interpolation, not to
    unrecorded plant motion. A geometric intersection is not a measured MuJoCo contact.
    """
    times, states = np.asarray(times, dtype=float), np.asarray(full_states, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError("audit times must be finite and strictly increasing with >=2 nodes")
    if states.shape != (len(times), 13) or not np.isfinite(states).all():
        raise ValueError("audit states must be finite [time,13]")
    if np.any(np.abs(np.linalg.norm(states[:, 3:7], axis=1) - 1) > 2e-4):
        raise ValueError("audit quaternions must be normalized")
    if not math.isfinite(max_substep_seconds) or not 0 < max_substep_seconds <= 0.001:
        raise ValueError("audit substeps must be positive and no larger than 1 ms")
    if not math.isfinite(ground_height):
        raise ValueError("ground height must be finite")
    subdivisions = np.ceil(np.diff(times) / max_substep_seconds).astype(int)
    fine_times = np.concatenate(
        [
            np.linspace(a, b, int(n) + 1)[:-1]
            for a, b, n in zip(times[:-1], times[1:], subdivisions, strict=True)
        ]
        + [times[-1:]]
    )
    positions = np.column_stack([np.interp(fine_times, times, states[:, i]) for i in range(3)])
    rotations = Slerp(times, Rotation.from_quat(states[:, 3:7]))(fine_times)
    collider_centers = positions + rotations.apply(CF21B_XML_COLLIDER_OFFSET_BODY_M)
    obstacle_centers, _ = world.obstacle_kinematics(fine_times)
    substep_dt = np.diff(fine_times)
    rotation_angle = (rotations[:-1].inv() * rotations[1:]).magnitude()
    offset_error = np.linalg.norm(CF21B_XML_COLLIDER_OFFSET_BODY_M) * rotation_angle**2 / 8
    obstacle_accel_bound = np.linalg.norm(world.obstacle_amplitudes, axis=-1) * (
        world.obstacle_angular_frequencies**2
    )
    obstacle_error = substep_dt[:, None] ** 2 * obstacle_accel_bound[None, :] / 8
    output: dict[str, Any] = {
        "schema": "recorded_cf21b_collision_geometry_audit_v1",
        "scope": (
            "geometry of interpolated recorded states; no contact dynamics or measured "
            "MuJoCo contact event is inferred"
        ),
        "source_node_count": len(times),
        "interpolated_node_count": len(fine_times),
        "time_support_seconds": [float(times[0]), float(times[-1])],
        "maximum_substep_seconds": float(np.max(substep_dt)),
        "interpolation": "linear body position; shortest-arc quaternion SLERP; analytic obstacles",
        "xml_collider_radius_m": CF21B_XML_COLLIDER_RADIUS_M,
        "xml_collider_body_offset_m": list(CF21B_XML_COLLIDER_OFFSET_BODY_M),
        "safety_shell": _sphere_sweep(
            fine_times,
            positions,
            obstacle_centers,
            world.obstacle_radii,
            world.config.ego_radius + world.config.obstacle_clearance,
            obstacle_error,
        ),
        "body_origin_envelope": _sphere_sweep(
            fine_times,
            positions,
            obstacle_centers,
            world.obstacle_radii,
            world.config.ego_radius,
            obstacle_error,
        ),
        "actual_xml_sphere_geometry": _sphere_sweep(
            fine_times,
            collider_centers,
            obstacle_centers,
            world.obstacle_radii,
            CF21B_XML_COLLIDER_RADIUS_M,
            obstacle_error + offset_error[:, None],
        ),
    }
    ground_clearance = collider_centers[:, 2] - CF21B_XML_COLLIDER_RADIUS_M - ground_height
    interval_min = np.minimum(ground_clearance[:-1], ground_clearance[1:])
    contact = np.flatnonzero(ground_clearance <= 0)
    first_ground = None
    if len(contact):
        node = int(contact[0])
        if node == 0:
            first_ground = float(fine_times[0])
        else:
            fraction = ground_clearance[node - 1] / (
                ground_clearance[node - 1] - ground_clearance[node]
            )
            first_ground = float(fine_times[node - 1] + fraction * substep_dt[node - 1])
    output["actual_xml_ground_geometry"] = {
        "ground_height_m": ground_height,
        "minimum_clearance_m": float(np.min(ground_clearance)),
        "minimum_clearance_lower_bound_m": float(np.min(interval_min - offset_error)),
        "minimum_clearance_upper_bound_m": float(np.min(interval_min + offset_error)),
        "first_chord_intersection_time_seconds": first_ground,
    }
    return output
