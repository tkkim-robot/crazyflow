"""Synchronized MuJoCo-rendered replay of recorded Version-A comparisons.

The numerical demo owns simulation and learning.  This module only consumes a fully recorded,
validated trace and renders it.  Keeping that boundary explicit prevents the video path from
silently re-running controllers, inventing rollouts, or changing the timing of the online learner.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import imageio_ffmpeg
import jax.numpy as jnp
import mujoco
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from scipy.spatial.transform import Rotation

from crazyflow import Sim
from crazyflow.safety.da_plcbf.continuous_version_a import QP_REJECTION_REASONS
from crazyflow.sim.visualize import change_material, draw_capsule

if TYPE_CHECKING:
    from collections.abc import Iterator


_BACKGROUND = "#071018"
_PANEL = "#0e1b25"
_TEXT = "#f5f7fa"
_MUTED = "#a9bac7"
_YELLOW = np.asarray((1.0, 0.82, 0.15, 0.95))
_SAFE = np.asarray((0.10, 0.95, 0.82, 0.34))
_UNSAFE = np.asarray((1.0, 0.18, 0.20, 0.27))
_SELECTED = np.asarray((0.12, 0.45, 1.0, 0.98))
_HISTORY = np.asarray((0.94, 1.0, 1.0, 0.96))
_PHYSICAL_OBSTACLE = np.asarray((0.72, 0.08, 0.06, 0.95))
_INFLATED_OBSTACLE = np.asarray((1.0, 0.34, 0.05, 0.16))
_TRUE_WIND = np.asarray((0.96, 0.22, 0.86, 0.98))
_ESTIMATED_WIND = np.asarray((0.22, 1.0, 0.38, 0.98))
_INTERVENTION = np.asarray((1.0, 0.47, 0.07, 0.98))
_GOAL = np.asarray((0.24, 1.0, 0.30, 0.96))
_CERTIFICATE_SENTENCE = (
    "Collision certification requires nonnegative H; executed-command status is shown above."
)


@dataclass(frozen=True, slots=True)
class ObstacleTrack:
    """One recorded spherical obstacle and its inflated safety shell.

    ``centers`` is ``[T,3]``.  A static obstacle simply repeats the same center at every sample.
    """

    centers: np.ndarray
    physical_radius: float
    inflated_radius: float
    label: str = "obstacle"


@dataclass(frozen=True, slots=True)
class MethodVideoTrace:
    """Recorded state, rollouts, and continuous-learning telemetry for one method.

    Rollout arrays use ``[T,H,3]`` for the nominal/selected trajectory and ``[T,K,H,3]`` for the
    fallback library.  ``selected_policy == -1`` denotes the nominal policy; nonnegative values
    index the fallback library.  Quaternions use Crazyflow's scalar-last ``xyzw`` convention.
    """

    position: np.ndarray
    quaternion_xyzw: np.ndarray
    nominal_rollout: np.ndarray
    fallback_rollouts: np.ndarray
    fallback_safe: np.ndarray
    selected_policy: np.ndarray
    selected_rollout: np.ndarray
    intervention_world: np.ndarray
    intervention_norm: np.ndarray
    descriptors: np.ndarray
    library_version: np.ndarray
    cumulative_gradient_steps: np.ndarray
    diversity_loss: np.ndarray
    descriptor_target_loss: np.ndarray
    gradient_norm: np.ndarray
    parameter_update_norm: np.ndarray
    minimum_library_value: np.ndarray
    # Old traces contain only the worst-policy value.  Never reinterpret it as the library max.
    maximum_library_value: np.ndarray | None = None
    selected_policy_value: np.ndarray | None = None
    selected_policy_dual: np.ndarray | None = None
    qp_valid: np.ndarray | None = None
    used_fallback: np.ndarray | None = None
    degraded: np.ndarray | None = None
    qp_rejection_flags: np.ndarray | None = None
    estimated_wind: np.ndarray | None = None
    control_mode: str = "plcbf"


@dataclass(frozen=True, slots=True)
class ComparisonVideoTrace:
    """Complete renderer input for a synchronized fixed/adaptive comparison."""

    time_seconds: np.ndarray
    goal_position: np.ndarray
    obstacles: tuple[ObstacleTrack, ...]
    true_wind: np.ndarray
    estimated_wind: np.ndarray
    wind_change_time: float
    descriptor_targets: np.ndarray
    fixed: MethodVideoTrace
    adaptive: MethodVideoTrace
    title: str = "Constant-wind adaptation: fixed PL-CBF vs continuously adaptive DA-PLCBF"
    left_label: str = "FIXED-LIBRARY PL-CBF"
    right_label: str = "CONTINUOUSLY ADAPTIVE DA-PLCBF"
    show_wind_change_banner: bool = True
    drone_radius: float = 0.0
    coverage_probes: dict[str, object] | None = None

    def validate(self) -> None:
        """Reject malformed records before opening a MuJoCo render context."""
        time = _finite_array(self.time_seconds, "time_seconds", ndim=1)
        if time.size < 2 or not np.all(np.diff(time) > 0.0):
            raise ValueError("time_seconds must contain at least two strictly increasing samples")
        steps = int(time.size)
        if not float(time[0]) <= float(self.wind_change_time) <= float(time[-1]):
            raise ValueError("wind_change_time must lie inside the recorded time interval")
        _shape(_finite_array(self.goal_position, "goal_position"), (3,), "goal_position")
        true_wind = _shape(_finite_array(self.true_wind, "true_wind"), (steps, 3), "true_wind")
        _shape(_finite_array(self.estimated_wind, "estimated_wind"), (steps, 3), "estimated_wind")

        before = time < float(self.wind_change_time) - 1e-12
        after = ~before
        if np.any(np.abs(true_wind[before]) > 1e-10):
            raise ValueError("true_wind must be exactly zero before the single wind change")
        if not np.any(after) or not np.allclose(true_wind[after], true_wind[after][0], atol=1e-10):
            raise ValueError("true_wind must remain one constant vector after the wind change")

        fixed_shape = _validate_method(self.fixed, steps, "fixed")
        adaptive_shape = _validate_method(self.adaptive, steps, "adaptive")
        if fixed_shape != adaptive_shape:
            raise ValueError("fixed and adaptive traces must use identical K and H rollout shapes")
        policy_count, _ = fixed_shape
        targets = _finite_array(self.descriptor_targets, "descriptor_targets", ndim=2)
        if targets.shape[0] != policy_count or targets.shape[1] < 2:
            raise ValueError("descriptor_targets must have shape [K,D] with D >= 2")
        if self.fixed.descriptors.shape[2] != targets.shape[1]:
            raise ValueError("fixed descriptors must match descriptor_targets dimensions")
        if self.adaptive.descriptors.shape[2] != targets.shape[1]:
            raise ValueError("adaptive descriptors must match descriptor_targets dimensions")
        for name in ("title", "left_label", "right_label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        if not isinstance(self.show_wind_change_banner, bool):
            raise TypeError("show_wind_change_banner must be boolean")
        if not np.isfinite(self.drone_radius) or self.drone_radius < 0.0:
            raise ValueError("drone_radius must be nonnegative and finite")
        _validate_coverage_probes(self.coverage_probes, time, policy_count)

        for index, obstacle in enumerate(self.obstacles):
            prefix = f"obstacles[{index}]"
            _shape(
                _finite_array(obstacle.centers, f"{prefix}.centers"),
                (steps, 3),
                f"{prefix}.centers",
            )
            physical = _positive_finite(obstacle.physical_radius, f"{prefix}.physical_radius")
            inflated = _positive_finite(obstacle.inflated_radius, f"{prefix}.inflated_radius")
            if inflated < physical + self.drone_radius:
                raise ValueError(f"{prefix}.inflated_radius must include physical and drone radii")
            if not isinstance(obstacle.label, str) or not obstacle.label.strip():
                raise ValueError(f"{prefix}.label must be nonempty")
            for name, method in (("fixed", self.fixed), ("adaptive", self.adaptive)):
                initial_distance = np.linalg.norm(method.position[0] - obstacle.centers[0])
                if initial_distance <= physical + self.drone_radius:
                    raise ValueError(f"{name} trace begins inside physical {prefix}")


@dataclass(frozen=True, slots=True)
class ComparisonRenderConfig:
    """Camera, timing, and output settings for the comparison renderer."""

    fps: float = 20.0
    width: int = 1600
    height: int = 900
    camera_azimuth: float = 135.0
    camera_elevation: float = -22.0
    camera_distance: float | None = None
    wind_arrow_scale: float = 0.18
    intervention_arrow_scale: float = 0.18
    probe_pause_time: float | None = None
    probe_pause_seconds: float = 0.0

    def validate(self) -> None:
        """Validate render settings."""
        _positive_finite(self.fps, "fps")
        if self.width < 960 or self.height < 540 or self.width % 2 or self.height % 2:
            raise ValueError("comparison video must use even dimensions of at least 960x540")
        for value, name in (
            (self.camera_azimuth, "camera_azimuth"),
            (self.camera_elevation, "camera_elevation"),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.camera_distance is not None:
            _positive_finite(self.camera_distance, "camera_distance")
        _positive_finite(self.wind_arrow_scale, "wind_arrow_scale")
        _positive_finite(self.intervention_arrow_scale, "intervention_arrow_scale")
        if not np.isfinite(self.probe_pause_seconds) or self.probe_pause_seconds < 0.0:
            raise ValueError("probe_pause_seconds must be nonnegative and finite")
        if self.probe_pause_time is not None:
            if not np.isfinite(self.probe_pause_time) or self.probe_pause_seconds <= 0.0:
                raise ValueError("probe_pause_time needs a finite time and positive pause duration")
        elif self.probe_pause_seconds != 0.0:
            raise ValueError("probe_pause_seconds requires an explicit probe_pause_time")


@dataclass(frozen=True, slots=True)
class ComparisonVideoResult:
    """Basic properties of an encoded comparison video."""

    path: Path
    frame_count: int
    width: int
    height: int
    fps: float


def comparison_video_frames(
    trace: ComparisonVideoTrace, config: ComparisonRenderConfig = ComparisonRenderConfig()
) -> Iterator[np.ndarray]:
    """Yield synchronized RGB frames using the actual Crazyflow quadrotor mesh."""
    if not isinstance(trace, ComparisonVideoTrace):
        raise TypeError("trace must be a ComparisonVideoTrace")
    if not isinstance(config, ComparisonRenderConfig):
        raise TypeError("config must be a ComparisonRenderConfig")
    trace.validate()
    config.validate()
    frame_indices = _frame_indices(trace.time_seconds, config.fps)
    pause_index = _probe_pause_index(trace, config, frame_indices)
    lookat = _method_camera_lookat(trace, trace.fixed, int(frame_indices[0]))
    distance = _camera_distance(trace, config)
    panel_width = max(480, config.width // 2)
    panel_height = max(270, int(config.height * 0.49))
    camera = {
        "lookat": lookat,
        "distance": distance,
        "azimuth": float(config.camera_azimuth),
        "elevation": float(config.camera_elevation),
    }
    sim = Sim(
        n_worlds=2,
        n_drones=1,
        drone="cf2x_T350",
        device="cpu",
        fused_mjx_model=False,
        enable_contacts=False,
    )
    sim.max_visual_geom = 5000
    # The physical CF mesh is only ~10 cm across.  Illuminated deck LEDs keep the real body and
    # attitude readable without replacing or enlarging the vehicle geometry.
    change_material(
        sim,
        "led_top",
        np.asarray((0,), dtype=np.int32),
        rgba=np.asarray((0.10, 0.86, 1.0, 1.0)),
        emission=np.asarray(1.0),
    )
    change_material(
        sim,
        "led_bot",
        np.asarray((0,), dtype=np.int32),
        rgba=np.asarray((0.10, 0.86, 1.0, 1.0)),
        emission=np.asarray(0.8),
    )
    try:
        initialized = False
        pause_inserted = False
        for index in frame_indices:
            _set_two_world_poses(sim, trace, index)
            if not initialized:
                sim.render(
                    mode="rgb_array",
                    world=0,
                    camera=-1,
                    cam_config=camera,
                    width=panel_width,
                    height=panel_height,
                )
                initialized = True
            left = _render_world(
                sim, trace, trace.fixed, index, 0, camera, panel_width, panel_height, config
            )
            right = _render_world(
                sim, trace, trace.adaptive, index, 1, camera, panel_width, panel_height, config
            )
            yield _compose_frame(trace, config, index, left, right)
            if not pause_inserted and pause_index is not None and index == pause_index:
                pause_inserted = True
                paused_frame = _compose_frame(trace, config, index, left, right, probe_pause=True)
                for _ in range(max(1, round(config.probe_pause_seconds * config.fps))):
                    yield paused_frame
    finally:
        sim.close()


def render_comparison_video(
    trace: ComparisonVideoTrace,
    path: str | os.PathLike[str],
    config: ComparisonRenderConfig = ComparisonRenderConfig(),
) -> ComparisonVideoResult:
    """Atomically encode the synchronized MuJoCo comparison as H.264."""
    trace.validate()
    config.validate()
    destination = Path(path)
    if destination.suffix.lower() != ".mp4":
        raise ValueError("comparison video path must end in .mp4")
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    temporary = destination.parent / f".{destination.name}.encoding.tmp.mp4"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    writer = imageio_ffmpeg.write_frames(
        str(temporary),
        (config.width, config.height),
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        fps=float(config.fps),
        quality=None,
        bitrate=None,
        codec="libx264",
        macro_block_size=2,
        ffmpeg_log_level="error",
        ffmpeg_timeout=60,
        output_params=[
            "-preset",
            "medium",
            "-crf",
            "18",
            "-threads",
            "1",
            "-movflags",
            "+faststart",
            "-metadata",
            "creation_time=1970-01-01T00:00:00Z",
        ],
    )
    frame_count = 0
    try:
        writer.send(None)
        for frame in comparison_video_frames(trace, config):
            writer.send(frame)
            frame_count += 1
        writer.close()
        writer = None
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
    return ComparisonVideoResult(
        path=destination,
        frame_count=frame_count,
        width=config.width,
        height=config.height,
        fps=float(config.fps),
    )


def _validate_method(method: MethodVideoTrace, steps: int, prefix: str) -> tuple[int, int]:
    if not isinstance(method, MethodVideoTrace):
        raise TypeError(f"{prefix} must be a MethodVideoTrace")
    _shape(_finite_array(method.position, f"{prefix}.position"), (steps, 3), f"{prefix}.position")
    quaternion = _shape(
        _finite_array(method.quaternion_xyzw, f"{prefix}.quaternion_xyzw"),
        (steps, 4),
        f"{prefix}.quaternion_xyzw",
    )
    if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1.0, atol=2e-3):
        raise ValueError(f"{prefix}.quaternion_xyzw must contain unit quaternions")
    fallback = _finite_array(method.fallback_rollouts, f"{prefix}.fallback_rollouts", ndim=4)
    if fallback.shape[0] != steps or fallback.shape[-1] != 3:
        raise ValueError(f"{prefix}.fallback_rollouts must have shape [T,K,H,3]")
    policy_count, horizon = int(fallback.shape[1]), int(fallback.shape[2])
    if policy_count < 1 or horizon < 2:
        raise ValueError(f"{prefix}.fallback_rollouts requires K >= 1 and H >= 2")
    _shape(
        _finite_array(method.nominal_rollout, f"{prefix}.nominal_rollout"),
        (steps, horizon, 3),
        f"{prefix}.nominal_rollout",
    )
    _shape(
        _finite_array(method.selected_rollout, f"{prefix}.selected_rollout"),
        (steps, horizon, 3),
        f"{prefix}.selected_rollout",
    )
    safe = np.asarray(method.fallback_safe)
    if safe.dtype != np.bool_ or safe.shape != (steps, policy_count):
        raise ValueError(f"{prefix}.fallback_safe must be boolean with shape [T,K]")
    selected = _integer_vector(method.selected_policy, steps, f"{prefix}.selected_policy")
    if np.any(selected < -1) or np.any(selected >= policy_count):
        raise ValueError(f"{prefix}.selected_policy must lie in [-1,K-1]")
    _shape(
        _finite_array(method.intervention_world, f"{prefix}.intervention_world"),
        (steps, 3),
        f"{prefix}.intervention_world",
    )
    intervention_norm = _shape(
        _finite_array(method.intervention_norm, f"{prefix}.intervention_norm"),
        (steps,),
        f"{prefix}.intervention_norm",
    )
    if np.any(intervention_norm < 0.0):
        raise ValueError(f"{prefix}.intervention_norm must be nonnegative")
    descriptors = _finite_array(method.descriptors, f"{prefix}.descriptors", ndim=3)
    if descriptors.shape[:2] != (steps, policy_count) or descriptors.shape[2] < 2:
        raise ValueError(f"{prefix}.descriptors must have shape [T,K,D] with D >= 2")
    for name in (
        "diversity_loss",
        "descriptor_target_loss",
        "gradient_norm",
        "parameter_update_norm",
        "minimum_library_value",
    ):
        _shape(
            _finite_array(getattr(method, name), f"{prefix}.{name}"), (steps,), f"{prefix}.{name}"
        )
    for name in ("library_version", "cumulative_gradient_steps"):
        values = _integer_vector(getattr(method, name), steps, f"{prefix}.{name}")
        if np.any(values < 0) or np.any(np.diff(values) < 0):
            raise ValueError(f"{prefix}.{name} must be nonnegative and monotone")
    for name in ("maximum_library_value", "selected_policy_value", "selected_policy_dual"):
        value = getattr(method, name)
        if value is not None:
            _shape(_finite_array(value, f"{prefix}.{name}"), (steps,), f"{prefix}.{name}")
    for name in ("qp_valid", "used_fallback", "degraded"):
        value = getattr(method, name)
        if value is not None:
            value = np.asarray(value)
            if value.dtype != np.bool_ or value.shape != (steps,):
                raise ValueError(f"{prefix}.{name} must be boolean with shape [T]")
    if method.qp_rejection_flags is not None:
        flags = np.asarray(method.qp_rejection_flags)
        if flags.dtype != np.bool_ or flags.shape != (steps, len(QP_REJECTION_REASONS)):
            raise ValueError(
                f"{prefix}.qp_rejection_flags must be boolean [T,{len(QP_REJECTION_REASONS)}]"
            )
    if method.estimated_wind is not None:
        _shape(
            _finite_array(method.estimated_wind, f"{prefix}.estimated_wind"),
            (steps, 3),
            f"{prefix}.estimated_wind",
        )
    if method.control_mode not in ("nominal", "analytic", "plcbf"):
        raise ValueError(f"{prefix}.control_mode must be nominal, analytic, or plcbf")
    if method.qp_valid is not None:
        for name in ("used_fallback", "degraded"):
            value = getattr(method, name)
            if value is not None and np.any(np.asarray(value) & np.asarray(method.qp_valid)):
                raise ValueError(f"{prefix}: valid QP cannot also use {name}")
    return policy_count, horizon


def _frame_indices(time_seconds: np.ndarray, fps: float) -> np.ndarray:
    start, stop = float(time_seconds[0]), float(time_seconds[-1])
    video_times = start + np.arange(int(np.floor((stop - start) * fps)) + 1) / fps
    if video_times[-1] < stop - 0.25 / fps:
        video_times = np.append(video_times, stop)
    right = np.searchsorted(time_seconds, video_times, side="left")
    right = np.clip(right, 0, len(time_seconds) - 1)
    left = np.clip(right - 1, 0, len(time_seconds) - 1)
    choose_left = np.abs(time_seconds[left] - video_times) <= np.abs(
        time_seconds[right] - video_times
    )
    return np.where(choose_left, left, right).astype(np.int64)


def _validate_coverage_probes(
    probes: dict[str, object] | None, time: np.ndarray, policy_count: int
) -> None:
    if probes is None:
        return
    if not isinstance(probes, dict):
        raise TypeError("coverage_probes must be a dictionary")
    times = _finite_array(probes.get("time_seconds"), "coverage_probes.time_seconds", ndim=1)
    if len(times) == 0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("coverage probe times must be nonempty and strictly increasing")
    if times[0] < time[0] - 1e-9 or times[-1] > time[-1] + 1e-9:
        raise ValueError("coverage probe times must lie in the recorded interval")
    if not isinstance(probes.get("source"), str) or not probes["source"].strip():
        raise ValueError("coverage_probes.source must identify the shared probe state/model")
    for method in ("fixed", "compensated", "adaptive"):
        if method == "compensated" and not any(
            key in probes for key in (f"{method}_h", f"{method}_safe_count")
        ):
            continue
        _shape(
            _finite_array(probes.get(f"{method}_h"), f"coverage_probes.{method}_h"),
            times.shape,
            f"coverage_probes.{method}_h",
        )
        counts = _integer_vector(
            probes.get(f"{method}_safe_count"), len(times), f"coverage_probes.{method}_safe_count"
        )
        if np.any(counts < 0) or np.any(counts > policy_count):
            raise ValueError("coverage probe safe counts must lie in [0,K]")


def _latest_coverage_probe(trace: ComparisonVideoTrace, index: int) -> int | None:
    """Return the last measured probe, never a future or interpolated value."""
    if trace.coverage_probes is None:
        return None
    times = np.asarray(trace.coverage_probes["time_seconds"])
    now = float(trace.time_seconds[index])
    if now < times[0] - 1e-9 or now > times[-1] + 0.25:
        return None
    latest = int(np.searchsorted(times, now + 1e-9, side="right")) - 1
    return latest if latest >= 0 else None


def _probe_pause_index(
    trace: ComparisonVideoTrace, config: ComparisonRenderConfig, frame_indices: np.ndarray
) -> int | None:
    if config.probe_pause_time is None:
        return None
    if trace.coverage_probes is None or not np.any(
        np.isclose(
            trace.coverage_probes["time_seconds"], config.probe_pause_time, atol=1e-9, rtol=0
        )
    ):
        raise ValueError("probe_pause_time must match a recorded coverage probe")
    matches = np.flatnonzero(
        np.isclose(trace.time_seconds[frame_indices], config.probe_pause_time, atol=1e-9, rtol=0)
    )
    if len(matches) == 0:
        raise ValueError("probe_pause_time must coincide with a sampled video frame")
    return int(frame_indices[matches[0]])


def _camera_distance(trace: ComparisonVideoTrace, config: ComparisonRenderConfig) -> float:
    if config.camera_distance is not None:
        return float(config.camera_distance)
    largest_shell = max((obstacle.inflated_radius for obstacle in trace.obstacles), default=0.0)
    return max(3.8, 5.0 * float(largest_shell))


def _method_camera_lookat(
    trace: ComparisonVideoTrace, method: MethodVideoTrace, index: int
) -> np.ndarray:
    """Use equal fixed-window ego cameras while looking slightly toward the shared goal."""
    position = np.asarray(method.position[index], dtype=np.float64)
    goal_delta = np.asarray(trace.goal_position, dtype=np.float64) - position
    horizontal = goal_delta.copy()
    horizontal[2] = 0.0
    norm = float(np.linalg.norm(horizontal))
    lookahead = np.zeros(3) if norm <= 1e-9 else 0.45 * horizontal / norm
    lookat = position + lookahead
    lookat[2] = max(0.55, position[2])
    return lookat


def _set_two_world_poses(sim: Sim, trace: ComparisonVideoTrace, index: int) -> None:
    position = jnp.asarray(
        np.stack((trace.fixed.position[index], trace.adaptive.position[index]))[:, None, :]
    )
    quaternion = jnp.asarray(
        np.stack((trace.fixed.quaternion_xyzw[index], trace.adaptive.quaternion_xyzw[index]))[
            :, None, :
        ]
    )
    states = sim.data.states.replace(pos=position, quat=quaternion)
    core = sim.data.core.replace(
        mjx_synced=jnp.asarray(False), mjx_collision_synced=jnp.asarray(False)
    )
    sim.data = sim.data.replace(states=states, core=core)


def _render_world(
    sim: Sim,
    trace: ComparisonVideoTrace,
    method: MethodVideoTrace,
    index: int,
    world: int,
    camera: dict[str, object],
    width: int,
    height: int,
    config: ComparisonRenderConfig,
) -> np.ndarray:
    # The two panels use the same azimuth/elevation/distance and the same goal-relative lookahead;
    # only each method's ego position translates its camera window.
    sim.viewer.viewer.cam.lookat[:] = _method_camera_lookat(trace, method, index)
    _draw_scene_markers(sim, trace, method, index, config)
    frame = sim.render(
        mode="rgb_array", world=world, camera=-1, cam_config=camera, width=width, height=height
    )
    if frame is None:
        raise RuntimeError("MuJoCo offscreen renderer returned no RGB frame")
    return np.asarray(frame, dtype=np.uint8)


def _draw_scene_markers(
    sim: Sim,
    trace: ComparisonVideoTrace,
    method: MethodVideoTrace,
    index: int,
    config: ComparisonRenderConfig,
) -> None:
    viewer = sim.viewer.viewer
    for obstacle in trace.obstacles:
        center = np.asarray(obstacle.centers[index], dtype=np.float64)
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=center,
            size=np.full(3, obstacle.inflated_radius),
            rgba=_INFLATED_OBSTACLE,
            label="",
        )
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=center,
            size=np.full(3, obstacle.physical_radius),
            rgba=_PHYSICAL_OBSTACLE,
            label="",
        )
    viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=np.asarray(trace.goal_position, dtype=np.float64),
        size=np.full(3, 0.09),
        rgba=_GOAL,
        label="",
    )
    _add_polyline(sim, method.position[: index + 1], _HISTORY, radius=0.012)
    _add_polyline(sim, method.nominal_rollout[index], _YELLOW, radius=0.009, dashed=True)
    for policy, rollout in enumerate(method.fallback_rollouts[index]):
        color = _SAFE if bool(method.fallback_safe[index, policy]) else _UNSAFE
        _add_polyline(sim, rollout, color, radius=0.006)
    _add_polyline(sim, method.selected_rollout[index], _SELECTED, radius=0.016)

    position = np.asarray(method.position[index], dtype=np.float64)
    if trace.drone_radius > 0.0:
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=position,
            size=np.full(3, trace.drone_radius),
            rgba=np.asarray((0.72, 0.93, 1.0, 0.15)),
            label="",
        )
    rotation = Rotation.from_quat(method.quaternion_xyzw[index]).as_matrix()
    for axis, color in zip(
        rotation.T,
        (
            np.asarray((1.0, 0.12, 0.12, 0.95)),
            np.asarray((0.12, 1.0, 0.18, 0.95)),
            np.asarray((0.18, 0.38, 1.0, 0.95)),
        ),
        strict=True,
    ):
        _add_arrow(sim, position, 0.16 * axis, color, radius=0.008)

    wind_base = position + np.asarray((0.0, 0.0, 0.42))
    _add_arrow(
        sim,
        wind_base + np.asarray((0.0, 0.0, 0.06)),
        config.wind_arrow_scale * trace.true_wind[index],
        _TRUE_WIND,
        radius=0.018,
    )
    _add_arrow(
        sim,
        wind_base - np.asarray((0.0, 0.0, 0.06)),
        config.wind_arrow_scale * _estimated_wind(trace, method, index),
        _ESTIMATED_WIND,
        radius=0.014,
    )
    _add_arrow(
        sim,
        position,
        config.intervention_arrow_scale * method.intervention_world[index],
        _INTERVENTION,
        radius=0.015,
    )


def _add_polyline(
    sim: Sim, points: np.ndarray, rgba: np.ndarray, *, radius: float, dashed: bool = False
) -> None:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 2:
        return
    for segment, (start, stop) in enumerate(zip(values[:-1], values[1:], strict=True)):
        if dashed and segment % 2:
            continue
        if np.linalg.norm(stop - start) <= 1e-10:
            continue
        draw_capsule(sim, start, stop, radius=radius, rgba=rgba)


def _add_arrow(
    sim: Sim, start: np.ndarray, vector: np.ndarray, rgba: np.ndarray, *, radius: float
) -> None:
    start = np.asarray(start, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return
    stop = start + vector
    draw_capsule(sim, start, stop, radius=radius, rgba=rgba)
    viewer = sim.viewer.viewer
    viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=stop,
        size=np.full(3, 1.8 * radius),
        rgba=rgba,
        label="",
    )


def _compose_frame(
    trace: ComparisonVideoTrace,
    config: ComparisonRenderConfig,
    index: int,
    left: np.ndarray,
    right: np.ndarray,
    *,
    probe_pause: bool = False,
) -> np.ndarray:
    figure = Figure(
        figsize=(config.width / 100.0, config.height / 100.0), dpi=100, facecolor=_BACKGROUND
    )
    canvas = FigureCanvasAgg(figure)
    scale = config.width / 1600.0
    for x, image, method, label in zip(
        (0.015, 0.510),
        (left, right),
        (trace.fixed, trace.adaptive),
        (trace.left_label, trace.right_label),
        strict=True,
    ):
        axis = figure.add_axes((x, 0.315, 0.475, 0.49))
        axis.imshow(image, aspect="auto")
        axis.set_axis_off()
        status, status_color = _execution_status(method, index)
        physical_margin, shell_margin = _clearances(trace, method, index)
        if physical_margin < 0.0:
            clearance = f"COLLISION  |  body clearance {physical_margin:+.2f} m"
            status_color = "#ff6470"
        elif _has_recorded_collision(trace, method, index):
            clearance = f"COLLISION EARLIER  |  shell {shell_margin:+.2f} m now"
            status_color = "#ff6470"
        else:
            clearance = f"body clearance {physical_margin:+.2f} m  |  shell {shell_margin:+.2f} m"
        figure.text(
            x + 0.2375, 0.865, label, color=_TEXT, fontsize=13 * scale, weight="bold", ha="center"
        )
        figure.text(
            x + 0.2375,
            0.831,
            f"{status}  |  {clearance}",
            color=status_color,
            fontsize=9.2 * scale,
            weight="bold",
            ha="center",
        )
        figure.text(
            x + 0.008,
            0.293,
            _hud(method, index),
            color=_TEXT,
            fontsize=10 * scale,
            va="top",
            linespacing=1.45,
        )
        estimate = _estimated_wind(trace, method, index)
        axis.text(
            0.016,
            0.968,
            f"wind estimate {_vector_label(estimate)} m/s\n"
            f"library v{int(method.library_version[index])}  |  "
            f"{int(method.cumulative_gradient_steps[index])} learning updates\n"
            f"skill target loss {float(method.descriptor_target_loss[index]):.3f}",
            transform=axis.transAxes,
            color=_TEXT,
            fontsize=9 * scale,
            va="top",
            linespacing=1.5,
            bbox={"facecolor": _PANEL, "edgecolor": "none", "alpha": 0.82, "pad": 5},
        )
        reason = _qp_rejection_label(method, index)
        if reason:
            axis.text(
                0.016,
                0.025,
                f"QP rejected: {reason}",
                transform=axis.transAxes,
                color="#ffd0a0",
                fontsize=8.5 * scale,
                bbox={"facecolor": _PANEL, "edgecolor": "none", "alpha": 0.82, "pad": 4},
            )
        graph_y = 0.129 if trace.coverage_probes is not None else 0.093
        graph_height = 0.216 - graph_y
        _draw_value_history(figure, (x + 0.031, graph_y, 0.277, graph_height), trace, method, index)
        _draw_route_overview(
            figure, (x + 0.340, graph_y, 0.122, graph_height), trace, method, index
        )

    time = float(trace.time_seconds[index])
    figure.text(
        0.5,
        0.981,
        trace.title,
        color=_TEXT,
        fontsize=16 * scale,
        weight="bold",
        va="top",
        ha="center",
    )
    figure.text(
        0.5,
        0.944,
        "MuJoCo-rendered replay of the differentiable Version-A simulation  |  "
        "finite-horizon collision values under each estimated model",
        color=_MUTED,
        fontsize=9 * scale,
        ha="center",
    )
    if trace.show_wind_change_banner:
        phase = "CALM" if time < trace.wind_change_time else "CONSTANT WIND"
        banner = (
            f"t = {time:5.2f} s   |   {phase}   |   "
            f"true wind {_vector_label(trace.true_wind[index])} m/s   |   "
            f"wind step at {trace.wind_change_time:g} s"
        )
    else:
        banner = f"t = {time:5.2f} s   |   same start, goal, obstacle geometry and actuator limits"
    if probe_pause:
        banner = (
            f"PROBE PAUSE  |  simulation t = {time:.2f} s  |  recorded state held for inspection"
        )
    figure.text(
        0.5,
        0.906,
        banner,
        color="#f4d6ed" if time >= trace.wind_change_time else _MUTED,
        fontsize=10 * scale,
        ha="center",
        weight="bold",
    )
    probe_index = _latest_coverage_probe(trace, index)
    if probe_index is not None:
        _draw_coverage_probe(figure, trace, probe_index, scale)
    else:
        figure.text(
            0.5,
            0.048,
            "white: executed path  |  yellow dashed: nominal prediction  |  "
            "teal / red: collision-clear / blocked skills  |  blue: selected rollout  |  "
            "orange: command change",
            color=_MUTED,
            fontsize=8.8 * scale,
            ha="center",
        )
    radius = (
        f"Drone collision radius {trace.drone_radius:.2f} m; "
        "shell radius = obstacle + drone + clearance."
    )
    if trace.drone_radius == 0.0:
        radius = "Legacy point-drone trace (radius 0 m); translucent surfaces are clearance shells."
    figure.text(
        0.5,
        0.020,
        radius + "  " + _CERTIFICATE_SENTENCE,
        color=_MUTED,
        fontsize=8.1 * scale,
        ha="center",
    )
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    if frame.shape != (config.height, config.width, 3):
        raise RuntimeError("Agg canvas produced unexpected comparison-frame dimensions")
    return frame


def _estimated_wind(
    trace: ComparisonVideoTrace, method: MethodVideoTrace, index: int
) -> np.ndarray:
    values = trace.estimated_wind if method.estimated_wind is None else method.estimated_wind
    return np.asarray(values[index])


def _draw_coverage_probe(
    figure: Figure, trace: ComparisonVideoTrace, probe_index: int, scale: float
) -> None:
    probes = trace.coverage_probes
    if probes is None:
        return
    time = float(np.asarray(probes["time_seconds"])[probe_index])
    figure.text(
        0.5,
        0.087,
        f"Same-state coverage probe  |  measured at t = {time:.2f} s",
        color="#f5e3ab",
        fontsize=10.3 * scale,
        weight="bold",
        ha="center",
    )
    names = [name for name in ("fixed", "compensated", "adaptive") if f"{name}_h" in probes]
    centers = (0.18, 0.50, 0.82) if len(names) == 3 else (0.27, 0.73)
    labels = {"fixed": "Frozen", "compensated": "Compensated", "adaptive": "Adaptive"}
    policy_count = trace.fixed.fallback_safe.shape[1]
    for center, name in zip(centers, names, strict=True):
        value = float(np.asarray(probes[f"{name}_h"])[probe_index])
        count = int(np.asarray(probes[f"{name}_safe_count"])[probe_index])
        figure.text(
            center,
            0.062,
            f"{labels[name]} H {value:+.6f}  |  {count}/{policy_count} collision-clear skills",
            color="#ff8f96" if value < 0.0 else "#58ebc4",
            fontsize=10.8 * scale,
            weight="bold",
            ha="center",
        )
    figure.text(
        0.5,
        0.041,
        f"Counterfactual evaluation at {probes['source']}. "
        "Main panels show each method's own executed trajectory.",
        color=_MUTED,
        fontsize=8.7 * scale,
        ha="center",
    )


def _vector_label(vector: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):+.2f}" for value in vector) + "]"


def _execution_status(method: MethodVideoTrace, index: int) -> tuple[str, str]:
    if method.control_mode == "nominal":
        return "EXECUTING NOMINAL", "#ffe085"
    if method.degraded is not None and bool(method.degraded[index]):
        return "UNCERTIFIED BEST EFFORT", "#ff6470"
    if (
        method.control_mode == "plcbf"
        and method.maximum_library_value is not None
        and float(method.maximum_library_value[index]) < 0.0
    ):
        return "NO COLLISION CERTIFICATE", "#ff6470"
    if method.used_fallback is not None and bool(method.used_fallback[index]):
        return "EXECUTING FALLBACK", "#ffba78"
    if method.qp_valid is not None and bool(method.qp_valid[index]):
        active = float(method.intervention_norm[index]) > 1e-4
        return ("QP INTERVENING", "#ffba78") if active else ("QP ACCEPTED", "#94efc4")
    return "COMMAND STATUS NOT RECORDED", _MUTED


def _qp_rejection_label(method: MethodVideoTrace, index: int) -> str:
    if method.qp_rejection_flags is None or method.control_mode == "nominal":
        return ""
    labels = []
    for reason, active in zip(QP_REJECTION_REASONS, method.qp_rejection_flags[index], strict=True):
        if active:
            labels.append(reason.replace("_", " ").removesuffix(" failed"))
    return ", ".join(labels)


def _clearances(
    trace: ComparisonVideoTrace, method: MethodVideoTrace, index: int
) -> tuple[float, float]:
    physical, inflated = [], []
    for obstacle in trace.obstacles:
        distance = float(np.linalg.norm(method.position[index] - obstacle.centers[index]))
        physical.append(distance - obstacle.physical_radius - trace.drone_radius)
        inflated.append(distance - obstacle.inflated_radius)
    return min(physical, default=float("inf")), min(inflated, default=float("inf"))


def _has_recorded_collision(
    trace: ComparisonVideoTrace, method: MethodVideoTrace, index: int
) -> bool:
    # Interpolate relative segments just as the recorded position history does. This also keeps
    # a passed-through obstacle visible as a failure after the nominal vehicle exits its body.
    for obstacle in trace.obstacles:
        relative = method.position[: index + 1] - obstacle.centers[: index + 1]
        if len(relative) > 1:
            start, delta = relative[:-1], np.diff(relative, axis=0)
            squared = np.sum(delta * delta, axis=1)
            fraction = np.clip(
                -np.sum(start * delta, axis=1) / np.maximum(squared, 1e-12), 0.0, 1.0
            )
            relative = np.concatenate((relative, start + fraction[:, None] * delta))
        if np.any(np.linalg.norm(relative, axis=1) < obstacle.physical_radius + trace.drone_radius):
            return True
    return False


def _value_label(values: np.ndarray | None, index: int, precision: int = 3) -> str:
    return "unrecorded" if values is None else f"{float(values[index]):+.{precision}f}"


def _hud(method: MethodVideoTrace, index: int) -> str:
    safe_count = int(np.count_nonzero(method.fallback_safe[index]))
    maximum = _value_label(method.maximum_library_value, index)
    selected = _value_label(method.selected_policy_value, index)
    if method.control_mode == "plcbf":
        policy = int(method.selected_policy[index])
        policy_name = "nominal" if policy < 0 else f"skill {policy}"
        dual = (
            "unrecorded"
            if method.selected_policy_dual is None
            else f"{float(method.selected_policy_dual[index]):.2e}"
        )
        certificate = f"selected H {selected} ({policy_name})   |   PL-CBF dual {dual}"
    else:
        certificate = "PL-CBF constraint disabled; library values shown for comparison"
    fallbacks = (
        "?"
        if method.used_fallback is None
        else str(np.count_nonzero(method.used_fallback[: index + 1]))
    )
    degraded = (
        "?" if method.degraded is None else str(np.count_nonzero(method.degraded[: index + 1]))
    )
    return (
        f"library H = max {maximum}   |   "
        f"collision-clear skills {safe_count}/{method.fallback_safe.shape[1]}\n"
        f"{certificate}\n"
        f"command change {float(method.intervention_norm[index]):.3f}   |   "
        f"fallback / degraded steps {fallbacks} / {degraded}"
    )


def _draw_value_history(
    figure: Figure,
    bounds: tuple[float, float, float, float],
    trace: ComparisonVideoTrace,
    method: MethodVideoTrace,
    index: int,
) -> None:
    axis = figure.add_axes(bounds, facecolor=_PANEL)
    time = trace.time_seconds[: index + 1]
    scale = figure.get_figwidth() / 16.0
    if method.maximum_library_value is None:
        axis.text(
            0.5,
            0.5,
            "Library max H was not recorded",
            color=_MUTED,
            transform=axis.transAxes,
            ha="center",
            fontsize=8 * scale,
        )
    else:
        axis.plot(
            time,
            method.maximum_library_value[: index + 1],
            color="#38ebc4",
            linewidth=1.5,
            label="library max H",
        )
        if method.selected_policy_value is not None and method.control_mode == "plcbf":
            axis.plot(
                time,
                method.selected_policy_value[: index + 1],
                color="#6199ff",
                linewidth=1.0,
                alpha=0.9,
                label="selected H",
            )
        axis.legend(
            loc="upper left",
            fontsize=6.6 * scale,
            frameon=False,
            labelcolor=_TEXT,
            ncol=2,
            handlelength=1.4,
            borderaxespad=0.3,
        )
    axis.axhline(0.0, color="#f7b7b9", linewidth=0.7, linestyle="--")
    if trace.show_wind_change_banner:
        axis.axvline(trace.wind_change_time, color="#d793c9", linewidth=0.7, linestyle=":")
    axis.axvline(trace.time_seconds[index], color=_TEXT, alpha=0.5, linewidth=0.6)
    axis.set_xlim(float(trace.time_seconds[0]), float(trace.time_seconds[-1]))
    # One common vertical scale for both views, fixed over time.
    values = [
        np.asarray(value)
        for item in (trace.fixed, trace.adaptive)
        for value in (item.maximum_library_value, item.selected_policy_value)
        if value is not None
    ]
    if values:
        joined = np.concatenate(values)
        low, high = min(-0.03, float(joined.min())), max(0.03, float(joined.max()))
        if high - low > 5.0:
            # Distant obstacles can produce H values tens of square metres above zero. A linear
            # axis would hide the meaningful sign changes during the encounter entirely.
            axis.set_yscale("symlog", linthresh=0.1, linscale=1.0)
            axis.set_ylim(1.12 * low, 1.12 * high)
        else:
            pad = 0.08 * (high - low)
            axis.set_ylim(low - pad, high + pad)
    axis.set_title(
        "Collision H history" + (" (symmetric log scale)" if axis.get_yscale() == "symlog" else ""),
        color=_TEXT,
        fontsize=8 * scale,
        pad=3,
    )
    axis.tick_params(colors=_MUTED, labelsize=7 * scale, length=2)
    axis.grid(color="#617080", alpha=0.18, linewidth=0.5)
    for spine in axis.spines.values():
        spine.set_color("#405564")


def _draw_route_overview(
    figure: Figure,
    bounds: tuple[float, float, float, float],
    trace: ComparisonVideoTrace,
    method: MethodVideoTrace,
    index: int,
) -> None:
    axis = figure.add_axes(bounds, facecolor=_PANEL)
    scale = figure.get_figwidth() / 16.0
    points = [
        trace.fixed.position[:, :2],
        trace.adaptive.position[:, :2],
        trace.goal_position[None, :2],
    ]
    for obstacle in trace.obstacles:
        center = obstacle.centers[index, :2]
        axis.add_patch(
            Circle(
                center,
                obstacle.inflated_radius,
                facecolor="#f77732",
                edgecolor="#f7a574",
                alpha=0.20,
                linewidth=0.6,
            )
        )
        axis.add_patch(
            Circle(
                center, obstacle.physical_radius, facecolor="#d63b39", edgecolor="none", alpha=0.75
            )
        )
        points.extend(
            [
                obstacle.centers[:, :2] - obstacle.inflated_radius,
                obstacle.centers[:, :2] + obstacle.inflated_radius,
            ]
        )
    history = method.position[: index + 1]
    axis.plot(history[:, 0], history[:, 1], color=_TEXT, linewidth=1.5)
    axis.scatter(*method.position[index, :2], s=17, color="#9df7ff", zorder=4)
    axis.scatter(*trace.goal_position[:2], s=30, marker="*", color="#7bff9f", zorder=4)
    all_points = np.concatenate(points)
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    center = (low + high) / 2
    half = max(0.5, float(np.max(high - low)) / 2 + 0.3)
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Executed path: top view", color=_TEXT, fontsize=7.7 * scale, pad=3)
    axis.tick_params(colors=_MUTED, labelsize=6 * scale, length=2)
    for spine in axis.spines.values():
        spine.set_color("#405564")


def _finite_array(value: object, name: str, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must use a floating dtype")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _integer_vector(value: object, steps: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (steps,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be an integer vector with shape [T]")
    return array


def _shape(array: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


__all__ = [
    "ComparisonRenderConfig",
    "ComparisonVideoResult",
    "ComparisonVideoTrace",
    "MethodVideoTrace",
    "ObstacleTrack",
    "comparison_video_frames",
    "render_comparison_video",
]
