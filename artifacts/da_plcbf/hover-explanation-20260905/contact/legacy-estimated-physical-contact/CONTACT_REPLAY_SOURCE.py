"""Separate, motor-off MuJoCo contact continuations of recorded airborne trajectories.

The airborne controller is never rerun here. A declared safety abort or contact trigger transfers
the recorded pose and velocity to a free rigid body; subsequent robot motion comes from mj_step.
Obstacle motion remains prescribed by the recorded world. This is a crash-presentation policy,
not a claim about the original controller's response or a contact-safety certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class ContactReplayConfig:
    duration_seconds: float = 3.0
    timestep: float = 0.001
    ground_height: float = 0.0
    collision_radius: float = 0.086
    collision_offset_body: tuple[float, float, float] = (0.0, 0.0, 0.02)
    friction: tuple[float, float, float] = (0.7, 0.01, 0.001)
    contact_time_constant: float = 0.006

    def validate(self) -> None:
        positive = (self.duration_seconds, self.timestep, self.collision_radius)
        if not all(math.isfinite(x) and x > 0 for x in positive):
            raise ValueError("duration, timestep and collider radius must be positive finite")
        if self.timestep > 0.002 or self.duration_seconds > 20:
            raise ValueError("contact replay supports at most 2 ms steps and 20 s continuations")
        if not math.isfinite(self.ground_height):
            raise ValueError("ground height must be finite")
        if (
            len(self.collision_offset_body) != 3
            or not np.isfinite(self.collision_offset_body).all()
        ):
            raise ValueError("collider offset must be a finite 3-vector")
        if len(self.friction) != 3 or not all(math.isfinite(x) and x >= 0 for x in self.friction):
            raise ValueError("friction coefficients must be three nonnegative finite values")
        if (
            not math.isfinite(self.contact_time_constant)
            or self.contact_time_constant < 2 * self.timestep
        ):
            raise ValueError("contact time constant must be at least two integration steps")


@dataclass(frozen=True)
class ContactBody:
    mass: float
    inertia: np.ndarray
    gravity: np.ndarray
    drag_matrix_body: np.ndarray
    external_force_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    external_torque_world: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def validate(self) -> None:
        if not math.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("contact body mass must be positive finite")
        inertia = np.asarray(self.inertia)
        if (
            inertia.shape != (3, 3)
            or not np.isfinite(inertia).all()
            or not np.allclose(inertia, inertia.T, rtol=1e-10, atol=1e-14)
            or np.min(np.linalg.eigvalsh(inertia)) <= 0
        ):
            raise ValueError("contact inertia must be symmetric positive definite")
        if np.max(np.linalg.eigvalsh(inertia)) > np.trace(inertia) / 2 + 1e-12:
            raise ValueError("inertia eigenvalues must satisfy the rigid-body triangle inequality")
        for name in ("gravity", "external_force_world", "external_torque_world"):
            value = np.asarray(getattr(self, name))
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite 3-vector")
        if (
            np.shape(self.drag_matrix_body) != (3, 3)
            or not np.isfinite(self.drag_matrix_body).all()
        ):
            raise ValueError("body drag matrix must be finite shape (3,3)")


@dataclass(frozen=True)
class ObstacleMotion:
    """Absolute-time piecewise-linear measured centers, with no extrapolation."""

    time_seconds: np.ndarray
    centers: np.ndarray
    radii: np.ndarray

    def validate(self) -> None:
        times, centers, radii = map(np.asarray, (self.time_seconds, self.centers, self.radii))
        if (
            times.ndim != 1
            or len(times) < 2
            or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)
        ):
            raise ValueError("obstacle times must be a finite strictly increasing vector")
        if centers.shape != (len(times), len(radii), 3) or not np.isfinite(centers).all():
            raise ValueError("obstacle centers must be finite [time,obstacle,3]")
        if radii.ndim != 1 or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError("obstacle radii must be a positive finite vector")

    def sample(self, time_seconds: np.ndarray | float) -> np.ndarray:
        query = np.asarray(time_seconds, dtype=float)
        if (
            not np.isfinite(query).all()
            or np.any(query < self.time_seconds[0] - 1e-9)
            or np.any(query > self.time_seconds[-1] + 1e-9)
        ):
            raise ValueError(
                "contact continuation must remain inside recorded obstacle time support"
            )
        output = np.empty((*query.shape, len(self.radii), 3))
        for obstacle in range(len(self.radii)):
            for axis in range(3):
                output[..., obstacle, axis] = np.interp(
                    query, self.time_seconds, self.centers[:, obstacle, axis]
                )
        return output


@dataclass(frozen=True)
class ContactTrigger:
    time_seconds: float
    full_state: np.ndarray
    kind: str
    source_interval: int
    obstacle_index: int | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContactReplay:
    time_seconds: np.ndarray
    full_state: np.ndarray
    obstacle_centers: np.ndarray
    obstacle_contact: np.ndarray
    ground_contact: np.ndarray
    contact_force_norm_N: np.ndarray
    events: tuple[dict[str, Any], ...]
    model_xml: str
    metadata: dict[str, Any]


def cf21b_contact_body() -> ContactBody:
    """Use the actual Version-A parameter file, whose inertia differs from the visual XML."""
    params = Path(__file__).resolve().parents[2] / "drones/params.toml"
    raw = tomllib.loads(params.read_text())["cf21B_500"]
    return ContactBody(
        raw["mass"],
        np.asarray(raw["J"]),
        np.asarray(raw["gravity_vec"]),
        np.asarray(raw["drag_matrix"]),
    )


def _interpolate_state(times: np.ndarray, states: np.ndarray, when: float) -> np.ndarray:
    state = np.asarray([np.interp(when, times, states[:, i]) for i in range(13)])
    state[3:7] = Slerp(times, Rotation.from_quat(states[:, 3:7]))(when).as_quat()
    return state


def _first_swept_crossing(
    times: np.ndarray, positions: np.ndarray, obstacles: ObstacleMotion, added_radius: float
) -> tuple[float, int, int] | None:
    if not len(obstacles.radii):
        return None
    relative = positions[:, None] - obstacles.sample(times)
    start, delta = relative[:-1], np.diff(relative, axis=0)
    radius = obstacles.radii + added_radius
    a = np.sum(delta * delta, axis=-1)
    b = 2 * np.sum(start * delta, axis=-1)
    c = np.sum(start * start, axis=-1) - radius**2
    discriminant = b * b - 4 * a * c
    root = (-b - np.sqrt(np.maximum(discriminant, 0))) / np.maximum(2 * a, 1e-30)
    root = np.where(c <= 0, 0, np.where((a > 1e-24) & (discriminant >= 0), root, np.inf))
    root = np.where((root >= 0) & (root <= 1), root, np.inf)
    if not np.isfinite(root).any():
        return None
    absolute = times[:-1, None] + root * np.diff(times)[:, None]
    interval, obstacle = np.unravel_index(np.argmin(absolute), absolute.shape)
    return float(absolute[interval, obstacle]), int(interval), int(obstacle)


def find_contact_trigger(
    times: np.ndarray,
    states: np.ndarray,
    obstacles: ObstacleMotion,
    config: ContactReplayConfig,
    *,
    kind: str = "physical_contact",
    shell_ego_radius: float | None = None,
    shell_clearance: float = 0.0,
    degraded_times: np.ndarray | None = None,
) -> ContactTrigger:
    """Find a swept contact or explicitly requested shell/degraded safety-abort trigger.

    Swept roots use chords between recorded collider-center nodes. Quaternion interpolation
    preserves orientation; a physical handoff is backed up if its curved offset would begin
    inside the collider. Actual MuJoCo contact events are always measured independently.
    """
    config.validate()
    obstacles.validate()
    times, states = np.asarray(times), np.asarray(states)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.all(np.diff(times) > 0)
        or not np.isfinite(times).all()
    ):
        raise ValueError("state times must be finite and strictly increasing")
    if states.shape != (len(times), 13) or not np.isfinite(states).all():
        raise ValueError("source states must be finite [time,13]")
    if np.any(np.abs(np.linalg.norm(states[:, 3:7], axis=1) - 1) > 2e-4):
        raise ValueError("source quaternions must be normalized")
    if kind not in ("physical_contact", "unsafe_shell", "degraded", "unsafe"):
        raise ValueError("trigger must be physical_contact, unsafe_shell, degraded or unsafe")
    candidates = []
    if kind == "physical_contact":
        offset = Rotation.from_quat(states[:, 3:7]).apply(config.collision_offset_body)
        crossing = _first_swept_crossing(
            times, states[:, :3] + offset, obstacles, config.collision_radius
        )
        if crossing:
            candidates.append((*crossing, "swept_collider_contact"))
        ground_gap = states[:, 2] + offset[:, 2] - config.collision_radius - config.ground_height
        hits = np.flatnonzero(ground_gap <= 0)
        if len(hits):
            node = int(hits[0])
            interval = max(node - 1, 0)
            fraction = (
                0 if node == 0 else ground_gap[interval] / (ground_gap[interval] - ground_gap[node])
            )
            candidates.append(
                (
                    float(
                        times[interval]
                        + fraction * (times[min(interval + 1, len(times) - 1)] - times[interval])
                    ),
                    interval,
                    None,
                    "swept_ground_contact",
                )
            )
    if kind in ("unsafe_shell", "unsafe"):
        if (
            shell_ego_radius is None
            or not math.isfinite(shell_ego_radius)
            or shell_ego_radius <= 0
            or not math.isfinite(shell_clearance)
            or shell_clearance < 0
        ):
            raise ValueError(
                "shell trigger requires explicit positive ego radius and nonnegative clearance"
            )
        crossing = _first_swept_crossing(
            times, states[:, :3], obstacles, shell_ego_radius + shell_clearance
        )
        if crossing:
            candidates.append((*crossing, "safety_abort_unsafe_shell"))
    if kind in ("degraded", "unsafe") and degraded_times is not None:
        valid_times = np.asarray(degraded_times)
        valid_times = valid_times[(valid_times >= times[0]) & (valid_times <= times[-1])]
        if len(valid_times):
            when = float(np.min(valid_times))
            interval = max(
                0, min(len(times) - 2, int(np.searchsorted(times, when, side="right") - 1))
            )
            candidates.append((when, interval, None, "safety_abort_degraded_control"))
    if not candidates:
        raise ValueError(f"recorded trajectory has no {kind} trigger")
    when, interval, obstacle, label = min(candidates, key=lambda item: item[0])
    state = _interpolate_state(times, states, when)
    actual_start = when
    if kind == "physical_contact":
        # Offset-sphere chords can slightly cut the rotational arc. Back up by at most one
        # recorded interval until the transferred free body has no deep initial penetration.
        for _ in range(100):
            center = state[:3] + Rotation.from_quat(state[3:7]).apply(config.collision_offset_body)
            gaps = (
                np.linalg.norm(center - obstacles.sample(actual_start), axis=-1)
                - obstacles.radii
                - config.collision_radius
            )
            ground_gap = center[2] - config.collision_radius - config.ground_height
            if min(float(np.min(gaps, initial=np.inf)), ground_gap) >= -1e-7:
                break
            if actual_start <= times[interval] + 1e-12:
                raise ValueError("physical contact source interval begins in penetration")
            actual_start = max(float(times[interval]), actual_start - config.timestep)
            state = _interpolate_state(times, states, actual_start)
    return ContactTrigger(
        actual_start,
        state,
        label,
        interval,
        obstacle,
        {
            "source_swept_event_time_seconds": when,
            "handoff_backoff_seconds": when - actual_start,
            "shell_ego_radius_m": shell_ego_radius,
            "shell_clearance_m": shell_clearance,
        },
    )


def _numbers(values: Any) -> str:
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).reshape(-1))


def build_contact_model(
    body: ContactBody, obstacles: ObstacleMotion, config: ContactReplayConfig
) -> tuple[mujoco.MjModel, str]:
    """Compile a free cf21B body, physical sphere colliders, and a ground plane."""
    body.validate()
    obstacles.validate()
    config.validate()
    root = ET.Element("mujoco", model="cf21B motor-off contact continuation")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    ET.SubElement(
        root,
        "option",
        timestep=str(config.timestep),
        gravity=_numbers(body.gravity),
        integrator="implicitfast",
        solver="Newton",
        iterations="100",
        tolerance="1e-10",
    )
    defaults = ET.SubElement(root, "default")
    ET.SubElement(
        defaults,
        "geom",
        condim="6",
        friction=_numbers(config.friction),
        solref=f"{config.contact_time_constant} 1",
        solimp="0.95 0.99 0.001",
        margin="0",
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 5", dir="0 0 -1")
    ET.SubElement(
        world,
        "geom",
        name="ground",
        type="plane",
        size="20 20 0.1",
        pos=f"0 0 {config.ground_height}",
        rgba="0.18 0.23 0.29 1",
        contype="2",
        conaffinity="1",
    )
    drone = ET.SubElement(world, "body", name="drone")
    ET.SubElement(drone, "freejoint", name="drone_free", align="false")
    inertia = body.inertia
    ET.SubElement(
        drone,
        "inertial",
        pos="0 0 0",
        mass=str(body.mass),
        fullinertia=_numbers(
            [
                inertia[0, 0],
                inertia[1, 1],
                inertia[2, 2],
                inertia[0, 1],
                inertia[0, 2],
                inertia[1, 2],
            ]
        ),
    )
    ET.SubElement(
        drone,
        "geom",
        name="drone_collider",
        type="sphere",
        size=str(config.collision_radius),
        pos=_numbers(config.collision_offset_body),
        rgba="0.15 0.7 0.95 0.45",
        contype="1",
        conaffinity="2",
    )
    for index, radius in enumerate(obstacles.radii):
        obstacle = ET.SubElement(world, "body", name=f"obstacle_{index}", mocap="true")
        ET.SubElement(
            obstacle,
            "geom",
            name=f"obstacle_geom_{index}",
            type="sphere",
            size=str(float(radius)),
            rgba="0.8 0.12 0.1 0.8",
            contype="2",
            conaffinity="1",
        )
    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml), xml


def run_contact_replay(
    trigger: ContactTrigger,
    body: ContactBody,
    obstacles: ObstacleMotion,
    config: ContactReplayConfig = ContactReplayConfig(),
    *,
    wind_velocity_world: np.ndarray | None = None,
) -> ContactReplay:
    """Advance a free body with zero rotor thrust and actual MuJoCo contact constraints."""
    model, xml = build_contact_model(body, obstacles, config)
    state = np.asarray(trigger.full_state, dtype=float)
    if (
        state.shape != (13,)
        or not np.isfinite(state).all()
        or abs(np.linalg.norm(state[3:7]) - 1) > 2e-4
    ):
        raise ValueError("handoff state must be finite [13] with a normalized xyzw quaternion")
    count = math.ceil(config.duration_seconds / config.timestep)
    times = trigger.time_seconds + np.arange(count + 1) * config.timestep
    centers = obstacles.sample(times)
    wind = (
        np.zeros((len(times), 3))
        if wind_velocity_world is None
        else np.broadcast_to(np.asarray(wind_velocity_world), (len(times), 3))
    )
    if not np.isfinite(wind).all():
        raise ValueError("replay wind must be finite")
    data = mujoco.MjData(model)
    data.time = trigger.time_seconds
    data.qpos[:3] = state[:3]
    data.qpos[3:7] = state[[6, 3, 4, 5]]
    data.qvel[:3], data.qvel[3:6] = state[7:10], state[10:13]
    drone_id = model.body("drone").id
    drone_geom = model.geom("drone_collider").id
    ground_geom = model.geom("ground").id
    obstacle_geoms = {model.geom(f"obstacle_geom_{i}").id: i for i in range(len(obstacles.radii))}
    states = np.empty((len(times), 13))
    contacts = np.zeros((len(times), len(obstacles.radii)), dtype=bool)
    ground = np.zeros(len(times), dtype=bool)
    forces = np.zeros(len(times))
    events = []
    previous_pairs: set[str] = set()
    minimum_distance = np.inf
    for index, when in enumerate(times):
        data.mocap_pos[:] = centers[index]
        data.xfrc_applied[:] = 0
        rotation = Rotation.from_quat(data.qpos[[4, 5, 6, 3]]).as_matrix()
        air_velocity_body = rotation.T @ (data.qvel[:3] - wind[index])
        data.xfrc_applied[drone_id, :3] = (
            rotation @ body.drag_matrix_body @ air_velocity_body + body.external_force_world
        )
        data.xfrc_applied[drone_id, 3:] = body.external_torque_world
        mujoco.mj_forward(model, data)
        states[index] = np.r_[data.qpos[:3], data.qpos[[4, 5, 6, 3]], data.qvel[:6]]
        current_pairs = set()
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = (int(contact.geom1), int(contact.geom2))
            if drone_geom not in pair or contact.efc_address < 0:
                continue
            other = pair[1] if pair[0] == drone_geom else pair[0]
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_index, force)
            force_norm = float(np.linalg.norm(force[:3]))
            forces[index] += force_norm
            minimum_distance = min(minimum_distance, float(contact.dist))
            label = "ground" if other == ground_geom else f"obstacle_{obstacle_geoms[other]}"
            current_pairs.add(label)
            if other == ground_geom:
                ground[index] = True
            else:
                contacts[index, obstacle_geoms[other]] = True
            if label not in previous_pairs:
                events.append(
                    {
                        "time_seconds": float(when),
                        "kind": "ground_contact" if other == ground_geom else "obstacle_contact",
                        "other": label,
                        "point_world": contact.pos.tolist(),
                        "normal_world": contact.frame[:3].tolist(),
                        "contact_distance_m": float(contact.dist),
                        "force_norm_N": force_norm,
                    }
                )
        previous_pairs = current_pairs
        if not np.isfinite(states[index]).all():
            raise RuntimeError("MuJoCo contact continuation became nonfinite")
        if index < count:
            mujoco.mj_step(model, data)
    warnings = {
        str(i): int(warning.number) for i, warning in enumerate(data.warning) if warning.number
    }
    if warnings:
        raise RuntimeError(f"MuJoCo emitted simulation warnings: {warnings}")
    metadata = {
        "schema": "crazyflow.mujoco_contact_replay.v1",
        "scope": (
            "Separate motor-off crash/abort presentation; no original controller commands "
            "execute after handoff and no controller safety guarantee is transferred."
        ),
        "trigger": {
            "time_seconds": trigger.time_seconds,
            "kind": trigger.kind,
            "source_interval": trigger.source_interval,
            "obstacle_index": trigger.obstacle_index,
            **trigger.details,
        },
        "config": asdict(config),
        "body": {
            "mass_kg": body.mass,
            "inertia_kg_m2": body.inertia.tolist(),
            "inertial_origin_body_m": [0, 0, 0],
            "gravity_m_s2": body.gravity.tolist(),
            "drag_matrix_body": body.drag_matrix_body.tolist(),
        },
        "model_semantics": (
            "Recorded mass/inertia and aerodynamic parameters are frozen at handoff; "
            "prescribed wind and obstacle positions may vary. Obstacle mocap bodies are "
            "fixed to the prescribed trajectory and do not react dynamically to impact."
        ),
        "collider_semantics": (
            "cf21B XML sphere geometry by default; separate from any older centered "
            "point-model ego envelope."
        ),
        "state_semantics": (
            "position/world velocity; xyzw body-to-world quaternion; body-frame angular velocity"
        ),
        "integration": (
            "MuJoCo implicitfast, explicit 1-step mj_step calls; body qpos/qvel assigned "
            "only once at handoff"
        ),
        "mujoco_version": mujoco.__version__,
        "obstacle_contact_steps": int(np.sum(np.any(contacts, axis=1))),
        "ground_contact_steps": int(np.sum(ground)),
        "maximum_contact_force_norm_N": float(np.max(forces)),
        "minimum_contact_distance_m": float(minimum_distance)
        if np.isfinite(minimum_distance)
        else None,
        "warning_counts": warnings,
        "official_references": [
            "https://mujoco.readthedocs.io/en/latest/overview.html#floating-objects",
            "https://mujoco.readthedocs.io/en/latest/XMLreference.html#body-mocap",
        ],
    }
    return ContactReplay(
        times, states, centers, contacts, ground, forces, tuple(events), xml, metadata
    )


def save_contact_replay(result: ContactReplay, directory: str | Path) -> Path:
    """Write a separate replay, standalone contact XML, and integrity-bound metadata."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arrays = directory / "contact_replay.npz"
    np.savez_compressed(
        arrays,
        time_seconds=result.time_seconds,
        full_state=result.full_state,
        obstacle_centers=result.obstacle_centers,
        obstacle_contact=result.obstacle_contact,
        ground_contact=result.ground_contact,
        contact_force_norm_N=result.contact_force_norm_N,
    )
    xml = directory / "contact_model.xml"
    xml.write_text(result.model_xml)
    metadata = {
        **result.metadata,
        "contact_events": result.events,
        "npz_sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
        "xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (directory / "contact_replay.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    return directory


def navigation_contact_replay(
    directory: str | Path,
    *,
    method: str = "adaptive",
    trigger_kind: str = "unsafe",
    config: ContactReplayConfig = ContactReplayConfig(),
) -> ContactReplay:
    """Create a motor-off continuation from an existing navigation episode, without control."""
    if method not in ("fixed", "adaptive"):
        raise ValueError("method must be fixed or adaptive")
    directory = Path(directory)
    metadata_path = directory / "navigation_comparison.json"
    metadata = json.loads(metadata_path.read_text())
    world = metadata["summary"]["world"]["config"]
    with np.load(directory / "navigation_comparison.npz", allow_pickle=False) as trace:
        times = trace["time_seconds"]
        obstacles = ObstacleMotion(
            times, trace["obstacle_centers"], trace["obstacle_physical_radii"]
        )
        active = trace[f"{method}_recorded_control_valid"]
        degraded = times[active & trace[f"{method}_degraded"]]
    with np.load(directory / "dense_plant_states.npz", allow_pickle=False) as dense:
        states = dense[method]
    source_times = np.arange(len(states)) * world["dt"]
    trigger = find_contact_trigger(
        source_times,
        states,
        obstacles,
        config,
        kind=trigger_kind,
        shell_ego_radius=world["ego_radius"],
        shell_clearance=world["obstacle_clearance"],
        degraded_times=degraded,
    )
    control_index = min(
        int(np.searchsorted(times, trigger.time_seconds, side="right") - 1), int(np.sum(active)) - 1
    )
    base = cf21b_contact_body()
    with np.load(directory / "raw_diagnostics.npz", allow_pickle=False) as raw:
        body = ContactBody(
            float(raw[f"{method}_actual_mass"][control_index]),
            raw[f"{method}_actual_inertia"][control_index],
            base.gravity,
            base.drag_matrix_body,
        )
    replay_times = (
        trigger.time_seconds
        + np.arange(math.ceil(config.duration_seconds / config.timestep) + 1) * config.timestep
    )
    wind = np.zeros((len(replay_times), 3))
    for event in world["wind_events"]:
        wind[replay_times >= event["time_seconds"] - 1e-10] = event["velocity"]
    replay = run_contact_replay(trigger, body, obstacles, config, wind_velocity_world=wind)
    replay.metadata["source"] = {
        "directory": str(directory.resolve()),
        "method": method,
        "recorded_point_model_ego_radius_m": world["ego_radius"],
        "recorded_controller_termination": metadata["summary"]["methods"][method]["termination"],
        "recorded_controller_physical_collision": metadata["summary"]["methods"][method][
            "physical_collision"
        ],
        "input_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in (
                "navigation_comparison.json",
                "navigation_comparison.npz",
                "dense_plant_states.npz",
                "raw_diagnostics.npz",
            )
        },
    }
    replay.metadata["obstacle_motion"] = (
        "Piecewise-linear interpolation of archived absolute-time obstacle-center samples; "
        "no trajectory is extrapolated beyond the source recording."
    )
    return replay
