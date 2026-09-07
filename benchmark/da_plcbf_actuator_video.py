"""Render recorded F2/A actuator episodes without executing a controller or plant.

The existing MuJoCo comparison renderer supplies the body-pose and geometry
primitives. This presentation consumes the new complete 17-state episode schema,
retained policy predictions and actual command application clocks. It never adds
contact dynamics, changes vehicle attitude, or splices independent experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BODY_XML = ROOT / "crazyflow/drones/cf21B_500.xml"
LEGACY_RENDERER = ROOT / "crazyflow/safety/da_plcbf/mujoco_comparison_video.py"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _array_sha(value: np.ndarray) -> str:
    array = np.asarray(value)
    return hashlib.sha256(
        str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True, slots=True)
class ActuatorVideoConfig:
    """A continuous real-time replay with one shared camera and visual scale."""

    fps: int = 20
    width: int = 1600
    height: int = 900
    camera_azimuth: float = 135.0
    camera_elevation: float = -24.0
    camera_distance: float = 3.4
    trail_seconds: float = 5.0
    synthetic_fixture: bool = False
    save_frame_every_seconds: float = 5.0
    left_method: str = "F2"
    right_method: str = "A"
    left_label: str = "Frozen · lag-aware"
    right_label: str = "Adaptive · lag-aware"
    allow_different_learning_contract: bool = False
    allow_different_libraries: bool = False

    def validate(self) -> None:
        """Require an ordinary fixed-rate video without a title or pause interval."""
        if type(self.fps) is not int or self.fps < 1:
            raise ValueError("fps must be a positive integer")
        for name in ("left_method", "right_method", "left_label", "right_label"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if type(self.allow_different_learning_contract) is not bool:
            raise ValueError("allow_different_learning_contract must be boolean")
        if type(self.allow_different_libraries) is not bool:
            raise ValueError("allow_different_libraries must be boolean")
        if self.width < 960 or self.height < 540 or self.width % 2 or self.height % 2:
            raise ValueError("video dimensions must be even and at least 960 by 540")
        for name in ("camera_distance", "trail_seconds", "save_frame_every_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite")
        for name in ("camera_azimuth", "camera_elevation"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class ReplaySample:
    """One auditable pose and available prediction, with no unrecorded suffix."""

    global_time: float
    display_time: float
    state: np.ndarray
    command: np.ndarray
    actual_forces: np.ndarray
    effectiveness: np.ndarray
    time_constants: np.ndarray
    dense_left: int
    dense_right: int
    interpolation_fraction: float
    control_index: int | None
    contact_stopped: bool
    terminal_stopped: bool
    goal: np.ndarray


@dataclass(frozen=True, slots=True)
class ReplayEpisode:
    """Validated, immutable arrays from one complete physical episode."""

    directory: Path
    binding: dict[str, Any]
    summary: dict[str, Any]
    dense: dict[str, np.ndarray]
    controls: dict[str, np.ndarray]
    applications: dict[str, np.ndarray]
    source_sha256: dict[str, str]
    stop_time: float
    contact_time: float | None
    force_boundary_audit: tuple[dict[str, Any], ...] = ()

    @property
    def world(self) -> dict[str, Any]:
        """Return the saved analytic world rather than regenerating a random seed."""
        return self.binding["scene"]["world"]

    def obstacle_centers(self, when: float) -> np.ndarray:
        """Evaluate only the recorded motion on its absolute clock."""
        world = self.world
        mean = np.asarray(world["obstacle_mean_centers"], dtype=float).reshape(-1, 3)
        amplitude = np.asarray(world["obstacle_amplitudes"], dtype=float).reshape(-1, 3)
        frequency = np.asarray(world["obstacle_angular_frequencies"], dtype=float)
        phase = np.asarray(world["obstacle_phases"], dtype=float)
        when = max(0.0, when - world.get("config", {}).get("obstacle_time_offset_seconds", 0.0))
        return mean + amplitude * np.sin(frequency * when + phase)[:, None]

    def wind_at(self, when: float) -> np.ndarray:
        """Read the recorded uniform wind schedule; never invent a visual disturbance."""
        wind = np.zeros(3)
        for event in self.world.get("config", {}).get("wind_events", []):
            if event["time_seconds"] <= when + 1e-10:
                wind = np.asarray(event["velocity"], dtype=float)
        return wind

    def wind_displacement(self, when: float) -> np.ndarray:
        """Integrate the recorded piecewise-constant field for visual tracer advection."""
        displacement, wind, previous = np.zeros(3), np.zeros(3), 0.0
        for event in self.world.get("config", {}).get("wind_events", []):
            boundary = min(when, event["time_seconds"])
            displacement += max(0.0, boundary - previous) * wind
            previous = boundary
            if event["time_seconds"] > when:
                break
            wind = np.asarray(event["velocity"], dtype=float)
        return displacement + max(0.0, when - previous) * wind

    def sample(self, when: float) -> ReplaySample:
        """Interpolate recorded position/motors and SLERP attitude, never extrapolating."""
        display = min(max(float(when), 0.0), self.stop_time)
        times = self.dense["time"]
        left = int(np.clip(np.searchsorted(times, display, side="right") - 1, 0, len(times) - 1))
        right = min(left + 1, len(times) - 1)
        fraction = (
            0.0 if left == right else float((display - times[left]) / (times[right] - times[left]))
        )
        state = (1 - fraction) * self.dense["state"][left] + fraction * self.dense["state"][right]
        state = np.asarray(state, dtype=float).copy()
        if right != left and fraction > 0:
            state[3:7] = Slerp(
                times[[left, right]], Rotation.from_quat(self.dense["state"][[left, right], 3:7])
            )(display).as_quat()
        else:
            state[3:7] = self.dense["state"][left, 3:7]
        effectiveness = self.dense["actual_effectiveness"][left].copy()
        tau = self.dense["actual_time_constants"][left].copy()
        for boundary in self.force_boundary_audit:
            if left == boundary["left_index"] and display < boundary["event_time_seconds"]:
                effectiveness = self.dense["actual_effectiveness"][left - 1].copy()
                tau = self.dense["actual_time_constants"][left - 1].copy()
        apply_times = self.applications["time"]
        application = int(
            np.clip(
                np.searchsorted(apply_times, display + 1e-10, side="right") - 1,
                0,
                len(apply_times) - 1,
            )
        )
        command = self.applications["command"][application].copy()
        applied = self.controls["command_applied"] & (
            self.controls["command_applied_at"] <= display + 1e-10
        )
        available = np.flatnonzero(applied)
        control_index = int(available[-1]) if len(available) else None
        # Before the first computed action is available, no future skill paths are shown.
        if control_index is None:
            navigation_start = self.binding["scene"]["navigation_start"]
            goal = np.asarray(
                self.world["initial_state"][:3]
                if display < navigation_start
                else self.world["waypoint_positions"][0]
            )
        else:
            goal = self.controls["goal"][control_index].copy()
        return ReplaySample(
            float(when),
            display,
            state,
            command,
            effectiveness * state[13:],
            effectiveness,
            tau,
            left,
            right,
            fraction,
            control_index,
            self.contact_time is not None and when >= self.contact_time - 1e-10,
            when > self.stop_time + 1e-10,
            goal,
        )


def _event_state_agrees(left: np.ndarray, right: np.ndarray, dt: float) -> bool:
    """Permit only quaternion normalization roundoff across a near-zero event step."""
    indices = np.r_[0:3, 7:17]
    if not np.array_equal(left[indices], right[indices]):
        return False
    quaternions = np.asarray([left[3:7], right[3:7]])
    precision = (
        np.float32
        if np.array_equal(quaternions, quaternions.astype(np.float32).astype(quaternions.dtype))
        else np.float64
    )
    epsilon = np.finfo(precision).eps
    norms = np.linalg.norm(quaternions, axis=1)
    if np.max(np.abs(norms - 1)) > 8 * epsilon:
        return False
    normalized = quaternions / norms[:, None]
    difference = min(
        np.linalg.norm(normalized[0] - normalized[1]), np.linalg.norm(normalized[0] + normalized[1])
    )
    # One short integration step may normalize the stored quaternion again. Its
    # orientation tolerance is tied to the recoverable storage precision, plus
    # the maximum observed rotation over the actual sub-ulp physical duration.
    allowance = (
        8 * epsilon + 0.5 * max(np.linalg.norm(left[10:13]), np.linalg.norm(right[10:13])) * dt
    )
    return bool(difference <= allowance)


def _validate_force_event_limits(
    dense: dict[str, np.ndarray], binding: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Authenticate the retained legacy left-limit label at a declared event only.

    The sealed runtime used sensing's time tolerance for telemetry labels. A
    pre-event endpoint a few ulps below an event can therefore carry the next
    effectiveness label while its force is the physical left limit. Accept this
    only when the immediately following exact-event row preserves the physical
    state up to quaternion normalization and supplies the correct right-limit
    force; never repair a general mismatch.
    """
    state, times = dense["state"], dense["time"]
    eta, forces = dense["actual_effectiveness"], dense["actual_forces"]
    expected = eta * state[:, 13:]
    bad = np.flatnonzero(np.any(~np.isclose(forces, expected, rtol=2e-5, atol=2e-7), axis=1))
    audit = []
    scene = binding["scene"]
    events = []
    if "event_time" in scene and "effectiveness_after" in scene:
        changed = np.asarray(scene["effectiveness_after"])
        events.append((float(scene["event_time"]), np.ones(4), changed))
        if scene.get("recovery_time") is not None:
            events.append((float(scene["recovery_time"]), changed, np.ones(4)))
    for index in bad:
        accepted = False
        for event, before, after in events:
            tolerance = 8 * abs(np.spacing(event))
            if not (
                0 < index < len(times) - 1
                and 0 < event - times[index] <= tolerance
                and times[index + 1] == event
                and _event_state_agrees(state[index], state[index + 1], event - times[index])
                and np.array_equal(eta[index], eta[index + 1])
                and np.allclose(eta[index - 1], before, rtol=0, atol=1e-7)
                and np.allclose(eta[index], after, rtol=0, atol=1e-7)
                and np.allclose(
                    forces[index], eta[index - 1] * state[index, 13:], rtol=2e-5, atol=2e-7
                )
                and np.allclose(forces[index + 1], expected[index + 1], rtol=2e-5, atol=2e-7)
            ):
                continue
            audit.append(
                {
                    "left_index": int(index),
                    "right_index": int(index + 1),
                    "left_time_seconds": float(times[index]),
                    "event_time_seconds": event,
                    "recorded_left_force_N": forces[index].tolist(),
                    "recorded_right_force_N": forces[index + 1].tolist(),
                    "non_attitude_coordinates_unchanged": True,
                    "quaternion_handling": (
                        "equivalent orientation within storage-precision normalization tolerance"
                    ),
                    "interpretation": (
                        "legacy left-limit force with right-limit effectiveness label"
                    ),
                }
            )
            accepted = True
            break
        if not accepted:
            raise ValueError(
                f"unexplained recorded force/effectiveness mismatch at dense row {index}"
            )
    return tuple(audit)


def load_episode(directory: str | Path, *, expected_method: str) -> ReplayEpisode:
    """Reject partial evidence, invented rollouts, and an inconsistent motor mapping."""
    directory = Path(directory).resolve()
    names = ("binding.json", "summary.json", "dense.npz", "controls.npz", "applications.npz")
    paths = [directory / name for name in names]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    binding = json.loads(paths[0].read_text())
    summary = json.loads(paths[1].read_text())
    if summary["status"] != "completed" or summary["method"] != expected_method:
        raise ValueError(f"the {expected_method} panel needs a completed {expected_method} episode")
    if summary["termination"] not in {"duration_complete", "physical_collision"}:
        raise ValueError("the renderer cannot promote an incomplete numerical episode")
    arrays = []
    for path in paths[2:]:
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        for value in values.values():
            value.setflags(write=False)
        arrays.append(values)
    dense, controls, applications = arrays
    times, state = dense["time"], dense["state"]
    if (
        len(times) < 2
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
        or state.shape != (len(times), 17)
        or not np.all(np.isfinite(state))
    ):
        raise ValueError("dense evidence must start at zero with finite increasing 17-state nodes")
    if np.max(np.abs(np.linalg.norm(state[:, 3:7], axis=1) - 1)) > 2e-4:
        raise ValueError("recorded attitudes are not normalized")
    if not math.isclose(float(times[-1]), float(summary["physical_time_seconds"]), abs_tol=1e-9):
        raise ValueError("summary stop time disagrees with the dense trace")
    if "candidate_states" not in controls:
        raise ValueError(
            "the promoted episode must retain actual candidate_states; paths are never fabricated"
        )
    candidates = controls["candidate_states"]
    if (
        candidates.ndim != 4
        or candidates.shape[0] != len(controls["time"])
        or candidates.shape[-1] != 17
    ):
        raise ValueError("candidate_states must have shape [controls,policies,nodes,17]")
    for name in ("candidate_valid", "eligible", "hard"):
        if controls[name].shape != candidates.shape[:2]:
            raise ValueError(f"{name} does not identify the retained candidate library")
    if controls["command_applied"].dtype != bool:
        raise ValueError("command_applied must be an explicit boolean mask")
    if not np.all(np.isfinite(controls["command_applied_at"][controls["command_applied"]])):
        raise ValueError("applied predictions require finite physical availability times")
    if applications["command"].shape != (len(applications["time"]), 4) or np.any(
        np.diff(applications["time"]) < 0
    ):
        raise ValueError("actual command application records are malformed")
    force_boundary_audit = _validate_force_event_limits(dense, binding)
    contact = None
    if summary["termination"] == "physical_collision":
        if summary["modeled_collider_collision"] is not True:
            raise ValueError("a contact ending requires an observed geometric intersection")
        contact = summary["collision"].get("first_intersection_time_seconds")
        contact = float(times[-1]) if contact is None else float(contact)
        if not 0 <= contact <= times[-1] + 1e-9:
            raise ValueError("contact time is outside the physical trace")
    stop_time = float(times[-1]) if contact is None else contact
    return ReplayEpisode(
        directory,
        binding,
        summary,
        dense,
        controls,
        applications,
        {str(path): _sha(path) for path in paths},
        stop_time,
        contact,
        force_boundary_audit,
    )


def validate_pair(
    left: ReplayEpisode,
    right: ReplayEpisode,
    *,
    allow_different_learning_contract: bool = False,
    allow_different_libraries: bool = False,
) -> None:
    """Bind the movie to one shared physical world, initial library and timing contract."""
    if left.summary["physical_world_id"] != right.summary["physical_world_id"]:
        raise ValueError("the panels must share the same complete physical world")
    if (
        left.binding["checkpoint"]["checkpoint_sha256"]
        != right.binding["checkpoint"]["checkpoint_sha256"]
    ) and not (allow_different_learning_contract or allow_different_libraries):
        raise ValueError("F2 and A must start from the same nominal checkpoint")
    for key in ("initial_state_sha256", "initial_learner_sha256"):
        if key == "initial_learner_sha256" and allow_different_libraries:
            continue
        if left.binding[key] != right.binding[key]:
            raise ValueError(f"paired {key} differs")
    for key in (
        "execution_mode",
        "plant_level",
        "plant_step_seconds",
        "filter_config",
        "observation_config",
    ):
        if left.binding["config"][key] != right.binding["config"][key]:
            raise ValueError(f"the panel {key} differs")
    np.testing.assert_array_equal(left.dense["state"][0], right.dense["state"][0])
    for key in (
        "obstacle_mean_centers",
        "obstacle_amplitudes",
        "obstacle_angular_frequencies",
        "obstacle_phases",
        "obstacle_radii",
        "waypoint_positions",
    ):
        np.testing.assert_array_equal(left.world[key], right.world[key])
    if left.controls["candidate_states"].shape[1:] != right.controls["candidate_states"].shape[1:]:
        raise ValueError("the paired candidate library and prediction horizon must match")


def motor_site_positions() -> np.ndarray:
    """Read the actual XML motor0..motor3 locations in the allocator's index order."""
    xml = ET.parse(BODY_XML).getroot()
    positions = []
    for index in range(4):
        site = xml.find(f".//site[@name='motor{index}']")
        if site is None:
            raise ValueError(f"missing motor{index} site")
        position = np.fromstring(site.attrib["pos"], sep=" ")
        if position.shape != (3,):
            raise ValueError("malformed rotor site position")
        positions.append(position)
    return np.asarray(positions)


def frame_times(left: ReplayEpisode, right: ReplayEpisode, fps: int) -> np.ndarray:
    """Keep one continuous absolute timeline from zero to the last physical ending."""
    duration = max(left.stop_time, right.stop_time)
    count = int(math.ceil(duration * fps - 1e-10)) + 1
    return np.minimum(np.arange(count, dtype=float) / fps, duration)


def _camera(sample: ReplaySample, config: ActuatorVideoConfig) -> dict[str, Any]:
    direction = sample.goal - sample.state[:3]
    lookahead = direction * min(0.25, 0.6 / max(np.linalg.norm(direction), 1e-12))
    lookat = sample.state[:3] + lookahead
    lookat[2] = max(0.45, lookat[2] - 0.1)
    return {
        "lookat": lookat,
        "distance": config.camera_distance,
        "azimuth": config.camera_azimuth,
        "elevation": config.camera_elevation,
    }


def _prediction_affects_control(episode: ReplayEpisode, sample: ReplaySample) -> bool:
    """Highlight only an applied, eligible prediction used by an accepted control branch.

    A QP must be accepted with a positive actually executed policy multiplier;
    a fallback must be accepted and executed. Emergency, invalid, degraded and
    zero-policy-dual QP decisions have no highlighted prediction. The
    ring never means that the complete predicted trajectory was flown.
    """
    index, controls = sample.control_index, episode.controls
    required = {"mode", "qp_valid", "fallback_valid", "executed_policy_dual"}
    if index is None or not required.issubset(controls):
        return False
    selected = int(controls["selected_index"][index])
    if (
        not 0 <= selected < controls["candidate_states"].shape[1]
        or not bool(controls["command_applied"][index])
        or not bool(controls["candidate_valid"][index, selected])
        or not bool(controls["eligible"][index, selected])
        or not np.all(np.isfinite(controls["candidate_states"][index, selected, :, :3]))
    ):
        return False
    mode = str(controls["mode"][index])
    return (
        mode == "qp"
        and bool(controls["qp_valid"][index])
        and float(controls["executed_policy_dual"][index]) > 0
    ) or (mode == "fallback" and bool(controls["fallback_valid"][index]))


def _thrust_caption(effectiveness: np.ndarray) -> str:
    """Use one plain percentage for a uniform loss, preserving motor order otherwise."""
    if np.allclose(effectiveness, effectiveness[0], rtol=0, atol=1e-7):
        return f"Thrust {effectiveness[0]:.0%}"
    return "Thrust " + " / ".join(f"{value:.0%}" for value in effectiveness)


def _markers(
    sim: Any,
    episode: ReplayEpisode,
    sample: ReplaySample,
    config: ActuatorVideoConfig,
    sites: np.ndarray,
) -> None:
    import mujoco

    from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
        _add_arrow,
        _add_endpoint_ring,
        _add_polyline,
        _skill_color,
    )

    viewer = sim.viewer.viewer
    wind = episode.wind_at(sample.display_time)
    if np.linalg.norm(wind) > 1e-9:
        displacement = episode.wind_displacement(sample.display_time)
        spacing = 0.8
        cell = np.floor((sample.state[:3] - displacement) / spacing).astype(int)
        for ix in range(cell[0] - 3, cell[0] + 4):
            for iy in range(cell[1] - 3, cell[1] + 4):
                for iz in range(cell[2] - 1, cell[2] + 2):
                    if (ix + 2 * iy + iz) % 3:
                        continue
                    start = np.asarray([ix, iy, iz]) * spacing + displacement + [0, 0, 0.45]
                    if start[2] > 0.15:
                        _add_polyline(
                            sim,
                            np.asarray([start, start + 0.18 * wind]),
                            np.asarray((0.28, 0.85, 1.0, 0.34)),
                            radius=0.003,
                        )
        _add_arrow(
            sim,
            sample.state[:3] + [0, 0, 0.42],
            0.22 * wind,
            np.asarray((0.25, 0.85, 1.0, 0.9)),
            radius=0.009,
        )
    centers = episode.obstacle_centers(sample.display_time)
    radii = np.asarray(episode.world["obstacle_radii"])
    clearance = episode.binding["config"]["filter_config"]["obstacle_clearance"]
    ego = episode.binding["config"]["filter_config"]["ego_radius"]
    for center, radius in zip(centers, radii, strict=True):
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=center,
            size=np.full(3, radius + ego + clearance),
            rgba=np.asarray((1.0, 0.40, 0.12, 0.09)),
            label="",
        )
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=center,
            size=np.full(3, radius),
            rgba=np.asarray((0.75, 0.13, 0.09, 0.98)),
            label="",
            category=int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
        )
    viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=sample.goal,
        size=np.full(3, 0.045),
        rgba=np.asarray((0.20, 0.95, 0.40, 0.85)),
        label="",
    )
    trail_mask = (episode.dense["time"] >= max(0.0, sample.display_time - config.trail_seconds)) & (
        episode.dense["time"] < sample.display_time
    )
    trail = np.vstack((episode.dense["state"][trail_mask, :3], sample.state[:3]))
    # Dense physical history may have thousands of points; decimation changes only the drawing.
    if len(trail) > 300:
        trail = trail[
            np.unique(np.r_[np.linspace(0, len(trail) - 1, 300).astype(int), len(trail) - 1])
        ]
    _add_polyline(sim, trail, np.asarray((0.95, 0.99, 1.0, 0.95)), radius=0.009)
    index = sample.control_index
    if index is not None:
        controls = episode.controls
        candidates = controls["candidate_states"][index, :, :, :3]
        for policy, points in enumerate(candidates):
            finite = bool(np.all(np.isfinite(points)))
            if not finite:
                continue
            clear = bool(controls["eligible"][index, policy])
            color = (
                np.asarray((1.0, 0.82, 0.15, 0.65))
                if policy == 0
                else _skill_color(policy - 1, clear)
            )
            color[3] = 0.55 if clear else 0.23
            _add_polyline(sim, points, color, radius=0.0045, dashed=not clear)
        selected = int(controls["selected_index"][index])
        if _prediction_affects_control(episode, sample):
            _add_endpoint_ring(
                sim, candidates[selected, -1], np.asarray((1.0, 1.0, 1.0, 0.95)), radius=0.047
            )
        if "backup_executed_backup" in controls and controls["backup_executed_backup"][index]:
            nodes = int(controls["backup_checked_trajectory_nodes"][index])
            points = controls["backup_checked_trajectory"][index, :nodes, :3]
            if len(points) > 1 and np.all(np.isfinite(points)):
                _add_polyline(sim, points, np.asarray((1.0, 0.71, 0.12, 0.95)), radius=0.009)
                _add_endpoint_ring(
                    sim, points[-1], np.asarray((1.0, 0.71, 0.12, 0.95)), radius=0.047
                )
    rotation = Rotation.from_quat(sample.state[3:7])
    # These small annotations follow actual motor sites; vehicle geometry/attitude never changes.
    for site, eta in zip(sites, sample.effectiveness, strict=True):
        severity = float(np.clip((1 - eta) / 0.5, 0, 1))
        rgba = (1 - severity) * np.asarray((0.10, 0.83, 0.93, 0.55)) + severity * np.asarray(
            (1.0, 0.35, 0.05, 0.95)
        )
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=sample.state[:3] + rotation.apply(site + np.asarray([0, 0, 0.018])),
            size=np.full(3, 0.008),
            rgba=rgba,
            label="",
        )


def _font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(str(path), size)


def _online_updates_used(episode: ReplayEpisode, sample: ReplaySample) -> int:
    """Count finite online versions in the last physically applied decision only."""
    initial = int(episode.summary["initial_library_version"])
    version = (
        initial
        if sample.control_index is None
        else int(episode.controls["control_library_version"][sample.control_index])
    )
    return max(0, version - initial)


def _compose(
    images: tuple[np.ndarray, np.ndarray],
    samples: tuple[ReplaySample, ReplaySample],
    episodes: tuple[ReplayEpisode, ReplayEpisode],
    config: ActuatorVideoConfig,
    force_scale: float,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    frame = Image.new("RGB", (config.width, config.height), "#071018")
    draw = ImageDraw.Draw(frame)
    scale = config.width / 1600
    header = round(66 * scale)
    footer = round(140 * scale)
    panel_width = config.width // 2
    title_font = _font(round(24 * scale), bold=True)
    small_font = _font(round(15 * scale))
    tiny_font = _font(round(13 * scale))
    draw.line([(panel_width, 0), (panel_width, config.height)], fill="#33434c", width=1)
    for side, (image, sample, episode, label) in enumerate(
        zip(images, samples, episodes, (config.left_label, config.right_label), strict=True)
    ):
        x = side * panel_width
        frame.paste(Image.fromarray(image), (x, header))
        draw.text((x + 24 * scale, 19 * scale), label, font=title_font, fill="#f2f6fa")
        local_time = f"{sample.display_time:05.2f} s"
        draw.text(
            (x + panel_width - 108 * scale, 24 * scale), local_time, font=small_font, fill="#b1c2ce"
        )
        if sample.contact_stopped:
            text = f"Geometric contact · stopped at {sample.display_time:.3f} s"
            draw.rounded_rectangle(
                (x + 20 * scale, header + 18 * scale, x + 404 * scale, header + 53 * scale),
                radius=5,
                fill="#591b18",
            )
            draw.text((x + 32 * scale, header + 27 * scale), text, font=small_font, fill="#ffe2d7")
        if any(
            np.any(event["velocity"])
            for event in episode.world.get("config", {}).get("wind_events", [])
        ):
            wind = episode.wind_at(sample.display_time)
            stage = (
                "Preflight"
                if sample.display_time < episode.binding["scene"]["navigation_start"]
                else "Navigation"
            )
            wind_text = (
                "calm"
                if np.linalg.norm(wind) < 1e-9
                else f"wind ({wind[0]:+.1f}, {wind[1]:+.1f}, {wind[2]:+.1f}) m/s"
            )
            caption = f"{stage} · {wind_text}"
            by = header + (64 if sample.contact_stopped else 16) * scale
            draw.rounded_rectangle(
                (x + 20 * scale, by, x + 495 * scale, by + 34 * scale), radius=5, fill="#102936"
            )
            draw.text((x + 30 * scale, by + 8 * scale), caption, font=small_font, fill="#9fe8fa")
        y0 = config.height - footer
        draw.text(
            (x + 24 * scale, y0 + 8 * scale),
            "Motor 0–3 · command / thrust (N)",
            font=small_font,
            fill="#d9e4eb",
        )
        bar_left, bar_top = x + 40 * scale, y0 + 40 * scale
        bar_height, bar_width, spacing = 55 * scale, 20 * scale, 55 * scale
        for motor in range(4):
            bx = bar_left + motor * spacing
            baseline = bar_top + bar_height
            command_height = bar_height * sample.command[motor] / force_scale
            actual_height = bar_height * sample.actual_forces[motor] / force_scale
            color = "#ff9d47" if sample.effectiveness[motor] < 0.999 else "#46bdd0"
            draw.rectangle((bx, baseline - actual_height, bx + bar_width, baseline), fill=color)
            draw.rectangle(
                (bx - 2 * scale, baseline - command_height, bx + bar_width + 2 * scale, baseline),
                outline="#ecf4f8",
                width=max(1, round(scale)),
            )
            draw.text(
                (bx + 4 * scale, baseline + 6 * scale), str(motor), font=tiny_font, fill="#aabac5"
            )
        draw.text(
            (x + 276 * scale, y0 + 38 * scale),
            f"0 — {force_scale:.2f} N",
            font=tiny_font,
            fill="#aabac5",
        )
        draw.text(
            (x + 276 * scale, y0 + 60 * scale), "Outline: command", font=tiny_font, fill="#d9e4eb"
        )
        draw.text(
            (x + 276 * scale, y0 + 80 * scale),
            "Fill: actual thrust",
            font=tiny_font,
            fill="#d9e4eb",
        )
        changed = np.flatnonzero(sample.effectiveness < 0.999)
        lag_ms = sample.time_constants * 1000
        lag_caption = (
            f"Lag {lag_ms.min():.0f}–{lag_ms.max():.0f} ms"
            if np.ptp(lag_ms) > 0.1
            else f"Lag {lag_ms[0]:.0f} ms"
        )
        draw.text(
            (x + 446 * scale, y0 + 38 * scale),
            _thrust_caption(sample.effectiveness),
            font=tiny_font,
            fill="#ffb573" if len(changed) else "#aabac5",
        )
        draw.text((x + 446 * scale, y0 + 60 * scale), lag_caption, font=tiny_font, fill="#aabac5")
        arrivals = episode.summary.get("waypoint_arrival_times_seconds", [])
        completed = sum(when <= sample.display_time + 1e-10 for when in arrivals)
        draw.text(
            (x + 446 * scale, y0 + 82 * scale),
            f"Waypoints {completed}/{episode.summary['waypoints_total']}",
            font=tiny_font,
            fill="#aabac5",
        )
        if side == 1 or _online_updates_used(episode, sample) > 0:
            draw.text(
                (x + 620 * scale, y0 + 82 * scale),
                f"Online updates: {_online_updates_used(episode, sample)}",
                font=tiny_font,
                fill="#aabac5",
            )
    mode = episodes[0].binding["config"]["execution_mode"]
    legend = (
        f"Selected {mode} simulation    Colored: predictions    ○ affects control    ━ white: flown"
    )
    if any(
        np.any(event["velocity"])
        for event in episodes[0].world.get("config", {}).get("wind_events", [])
    ):
        legend = (
            "Colored: current predictions    Cyan: wind    White: flown    "
            "Gold: executing checked backup"
        )
    if config.synthetic_fixture:
        legend = "SYNTHETIC FIXTURE · " + legend
    box = draw.textbbox((0, 0), legend, font=tiny_font)
    draw.text(
        ((config.width - box[2]) / 2, config.height - 24 * scale),
        legend,
        font=tiny_font,
        fill="#aabac5",
    )
    return np.asarray(frame).copy()


def _sample_audit(episode: ReplayEpisode, sample: ReplaySample) -> dict[str, Any]:
    index = sample.control_index
    return {
        "display_time_seconds": sample.display_time,
        "dense_bracket_indices": [sample.dense_left, sample.dense_right],
        "interpolation_fraction": sample.interpolation_fraction,
        "state17_sha256": _array_sha(sample.state),
        "state17": sample.state,
        "available_control_index": index,
        "control_application_time": float(episode.controls["command_applied_at"][index])
        if index is not None
        else None,
        "library_version": int(episode.controls["control_library_version"][index])
        if index is not None
        else None,
        "selected_policy_index": int(episode.controls["selected_index"][index])
        if index is not None
        else None,
        "selected_prediction_highlighted": _prediction_affects_control(episode, sample),
        "online_updates_used": _online_updates_used(episode, sample),
        "command_N": sample.command,
        "actual_force_N": sample.actual_forces,
        "effectiveness": sample.effectiveness,
        "time_constants_seconds": sample.time_constants,
        "contact_stopped": sample.contact_stopped,
        "terminal_stopped": sample.terminal_stopped,
        "wind_velocity_mps": episode.wind_at(sample.display_time),
        "preflight": sample.display_time < episode.binding["scene"]["navigation_start"],
        "executing_committed_backup": bool(episode.controls["backup_executed_backup"][index])
        if index is not None and "backup_executed_backup" in episode.controls
        else False,
    }


def render_pair(
    left_directory: str | Path,
    right_directory: str | Path,
    output: str | Path,
    config: ActuatorVideoConfig = ActuatorVideoConfig(),
) -> Path:
    """Encode an exclusive MP4 and full source/frame audit, retaining render failures."""
    config.validate()
    left = load_episode(left_directory, expected_method=config.left_method)
    right = load_episode(right_directory, expected_method=config.right_method)
    validate_pair(
        left,
        right,
        allow_different_learning_contract=config.allow_different_learning_contract,
        allow_different_libraries=config.allow_different_libraries,
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Read-only validation above needs neither a GL context nor a JAX device.
    import imageio_ffmpeg
    from PIL import Image

    from crazyflow import Sim
    from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
        _install_marker_shadow_categories,
        _set_two_world_poses,
    )

    sites = motor_site_positions()
    sources = {
        **left.source_sha256,
        **right.source_sha256,
        str(Path(__file__).resolve()): _sha(Path(__file__)),
        str(LEGACY_RENDERER): _sha(LEGACY_RENDERER),
        str(BODY_XML): _sha(BODY_XML),
    }
    proof = {
        "status": "rendering",
        "synthetic_fixture": config.synthetic_fixture,
        "config": asdict(config),
        "source_sha256": sources,
        "physical_world_id": left.summary["physical_world_id"],
        "panel_methods": {"left": config.left_method, "right": config.right_method},
        "same_complete_initial_learner_state": (
            left.binding["initial_learner_sha256"] == right.binding["initial_learner_sha256"]
        ),
        "same_checkpoint_bytes": (
            left.binding["checkpoint"]["checkpoint_sha256"]
            == right.binding["checkpoint"]["checkpoint_sha256"]
        ),
        "learning_contract_comparison": config.allow_different_learning_contract,
        "different_initial_libraries_comparison": config.allow_different_libraries,
        "motor_indices": list(range(4)),
        "motor_site_positions_body_m": sites,
        "pose_interpolation": (
            "recorded position/effort linear; recorded quaternion shortest-arc SLERP"
        ),
        "command_interpolation": "exact right-continuous application events",
        "force_display": "current recorded effectiveness times interpolated internal effort, in N",
        "force_boundary_audit": {"F2": left.force_boundary_audit, "A": right.force_boundary_audit},
        "force_boundary_semantics": (
            "original arrays are unchanged; any accepted legacy pre-event endpoint must be "
            "within eight ulps, immediately precede the exact declared event, retain the "
            "same position/velocity/rates/motor states and equivalent normalized attitude, "
            "and match both physical force limits; display uses the "
            "pre-event model strictly before that event and the post-event model at it"
        ),
        "obstacle_clock": "saved analytic absolute time; each panel freezes with its contact pose",
        "policy_paths": (
            "only retained candidate_states from the last physically applied available decision"
        ),
        "selection_semantics": (
            "endpoint ring only for a physically applied eligible selected prediction: "
            "accepted QP with positive executed_policy_dual, or accepted executed fallback; "
            "no ring for emergency/degraded/invalid or a QP with zero executed policy dual; "
            "the ring never denotes a flown future trajectory; only white trail is flown"
        ),
        "contact_semantics": "first audited geometric intersection; no MuJoCo contact continuation",
        "no_controller_learner_or_plant_execution": True,
        "no_experimental_splicing": True,
        "no_vehicle_scaling_or_attitude_edits": True,
    }
    _write_json(output / "render_binding.json", proof)
    times = frame_times(left, right, config.fps)
    sample_rows = ([left.sample(when) for when in times], [right.sample(when) for when in times])
    pose_trace = SimpleNamespace(
        fixed=SimpleNamespace(
            position=np.asarray([sample.state[:3] for sample in sample_rows[0]]),
            quaternion_xyzw=np.asarray([sample.state[3:7] for sample in sample_rows[0]]),
        ),
        adaptive=SimpleNamespace(
            position=np.asarray([sample.state[:3] for sample in sample_rows[1]]),
            quaternion_xyzw=np.asarray([sample.state[3:7] for sample in sample_rows[1]]),
        ),
    )
    scale = config.width / 1600
    panel_width = config.width // 2
    panel_height = config.height - round(66 * scale) - round(140 * scale)
    maximum = max(
        float(np.max(episode.dense[name]))
        for episode in (left, right)
        for name in ("command", "actual_forces")
    )
    force_scale = max(0.05, math.ceil(maximum * 20) / 20)
    temporary = output / ".comparison.encoding.mp4"
    destination = output / "comparison.mp4"
    audit_path = output / "frame_audit.jsonl"
    writer = None
    sim = None
    frame_count = 0
    failure = None
    cached_images: list[np.ndarray | None] = [None, None]
    cached_times: list[float | None] = [None, None]
    try:
        sim = Sim(
            n_worlds=2,
            n_drones=1,
            drone="cf21B_500",
            device="cpu",
            fused_mjx_model=False,
            enable_contacts=False,
        )
        sim.max_visual_geom = 6000
        # Set the actual recorded room extent before MuJoCo allocates shadow textures.
        room_extent = (
            max(
                float(np.max(np.linalg.norm(episode.dense["state"][:, :3], axis=1)))
                for episode in (left, right)
            )
            + config.camera_distance
            + 2
        )
        sim.mj_model.vis.map.shadowclip = max(
            sim.mj_model.vis.map.shadowclip, room_extent / sim.mj_model.stat.extent
        )
        sim.mj_model.vis.quality.shadowsize = max(sim.mj_model.vis.quality.shadowsize, 4096)
        _set_two_world_poses(sim, pose_trace, 0)
        initial_camera = _camera(sample_rows[0][0], config)
        sim.render(
            mode="rgb_array",
            world=0,
            camera=-1,
            cam_config=initial_camera,
            width=panel_width,
            height=panel_height,
        )
        _install_marker_shadow_categories(sim)
        writer = imageio_ffmpeg.write_frames(
            str(temporary),
            (config.width, config.height),
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=config.fps,
            codec="libx264",
            quality=None,
            macro_block_size=2,
            ffmpeg_log_level="error",
            output_params=[
                "-preset",
                "medium",
                "-crf",
                "18",
                "-threads",
                "1",
                "-movflags",
                "+faststart",
            ],
        )
        writer.send(None)
        with audit_path.open("x") as audit:
            for index, when in enumerate(times):
                _set_two_world_poses(sim, pose_trace, index)
                samples = (sample_rows[0][index], sample_rows[1][index])
                for side, (episode, sample) in enumerate(zip((left, right), samples, strict=True)):
                    if cached_times[side] == sample.display_time:
                        continue
                    camera = _camera(sample, config)
                    sim.viewer.viewer.cam.lookat[:] = camera["lookat"]
                    sim.viewer.viewer.cam.distance = camera["distance"]
                    _markers(sim, episode, sample, config, sites)
                    pixels = sim.render(
                        mode="rgb_array",
                        world=side,
                        camera=-1,
                        # Sim's persistent renderer keeps its initialization contract;
                        # the live viewer camera above follows each recorded pose.
                        cam_config=initial_camera,
                        width=panel_width,
                        height=panel_height,
                    )
                    if pixels is None:
                        raise RuntimeError("MuJoCo returned no RGB frame")
                    cached_images[side] = np.asarray(pixels, dtype=np.uint8).copy()
                    cached_times[side] = sample.display_time
                frame = _compose(tuple(cached_images), samples, (left, right), config, force_scale)
                writer.send(frame)
                audit.write(
                    json.dumps(
                        _jsonable(
                            {
                                "frame": index,
                                "video_time_seconds": index / config.fps,
                                "physical_time_seconds": float(when),
                                "raw_rgb_sha256": _array_sha(frame),
                                "F2": _sample_audit(left, samples[0]),
                                "A": _sample_audit(right, samples[1]),
                            }
                        ),
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                audit.flush()
                if (
                    index % max(1, round(config.save_frame_every_seconds * config.fps)) == 0
                    or index == len(times) - 1
                ):
                    Image.fromarray(frame).save(output / f"frame-{index:06d}.png")
                frame_count += 1
        writer.close()
        writer = None
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(temporary, destination)
        for path, digest in sources.items():
            if _sha(Path(path)) != digest:
                raise RuntimeError(f"source changed during rendering: {path}")
    except BaseException as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        if writer is not None:
            writer.close()
        if sim is not None:
            sim.close()
        _write_json(
            output / "render_summary.json",
            {
                "status": "incomplete" if failure else "completed",
                "error": failure,
                "frame_count": frame_count,
                "physical_duration_seconds": float(times[-1]),
                "video_duration_seconds": frame_count / config.fps,
                "fps": config.fps,
                "width": config.width,
                "height": config.height,
                "video": str(destination) if destination.exists() else None,
                "video_sha256": _sha(destination) if destination.exists() else None,
                "frame_audit_sha256": _sha(audit_path) if audit_path.exists() else None,
                "force_scale_N": force_scale,
                "preserved_partial_video": str(temporary) if temporary.exists() else None,
            },
        )
    return destination


def main() -> None:
    """Render a promoted pair, or validate its complete evidence without a GL context."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--adaptive", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--camera-distance", type=float, default=3.4)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-24.0)
    parser.add_argument("--synthetic-fixture", action="store_true")
    parser.add_argument("--left-method", default="F2")
    parser.add_argument("--right-method", default="A")
    parser.add_argument("--left-label", default="Frozen · lag-aware")
    parser.add_argument("--right-label", default="Adaptive · lag-aware")
    parser.add_argument("--allow-different-learning-contract", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        left = load_episode(args.frozen, expected_method=args.left_method)
        right = load_episode(args.adaptive, expected_method=args.right_method)
        validate_pair(
            left, right, allow_different_learning_contract=args.allow_different_learning_contract
        )
        print(
            json.dumps(
                {
                    "status": "validated",
                    "physical_world_id": left.summary["physical_world_id"],
                    "frame_count": len(frame_times(left, right, args.fps)),
                }
            )
        )
        return
    if args.output is None:
        parser.error("--output is required for rendering")
    result = render_pair(
        args.frozen,
        args.adaptive,
        args.output,
        ActuatorVideoConfig(
            fps=args.fps,
            width=args.width,
            height=args.height,
            camera_distance=args.camera_distance,
            camera_azimuth=args.camera_azimuth,
            camera_elevation=args.camera_elevation,
            synthetic_fixture=args.synthetic_fixture,
            left_method=args.left_method,
            right_method=args.right_method,
            left_label=args.left_label,
            right_label=args.right_label,
            allow_different_learning_contract=args.allow_different_learning_contract,
        ),
    )
    print(result)


if __name__ == "__main__":
    main()
