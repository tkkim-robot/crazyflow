"""Simple synchronized 3D video for the fixed-versus-adaptive DA-PLCBF demo.

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
from scipy.spatial.transform import Rotation

from crazyflow import Sim
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
    "The selected fallback defines the safety certificate; the QP command is executed."
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

        for index, obstacle in enumerate(self.obstacles):
            prefix = f"obstacles[{index}]"
            _shape(
                _finite_array(obstacle.centers, f"{prefix}.centers"),
                (steps, 3),
                f"{prefix}.centers",
            )
            physical = _positive_finite(obstacle.physical_radius, f"{prefix}.physical_radius")
            inflated = _positive_finite(obstacle.inflated_radius, f"{prefix}.inflated_radius")
            if inflated < physical:
                raise ValueError(f"{prefix}.inflated_radius cannot be smaller than physical_radius")
            if not isinstance(obstacle.label, str) or not obstacle.label.strip():
                raise ValueError(f"{prefix}.label must be nonempty")
            for name, method in (("fixed", self.fixed), ("adaptive", self.adaptive)):
                initial_distance = np.linalg.norm(method.position[0] - obstacle.centers[0])
                if initial_distance <= physical:
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
    lookat = _method_camera_lookat(trace, trace.fixed, int(frame_indices[0]))
    distance = _camera_distance(trace, config)
    panel_width = max(480, config.width // 2)
    panel_height = max(420, int(config.height * 0.72))
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
        descriptor_limit = _descriptor_limit(trace)
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
            yield _compose_frame(trace, config, index, left, right, descriptor_limit)
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
        config.wind_arrow_scale * trace.estimated_wind[index],
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
    descriptor_limit: float,
) -> np.ndarray:
    figure = Figure(
        figsize=(config.width / 100.0, config.height / 100.0), dpi=100, facecolor=_BACKGROUND
    )
    canvas = FigureCanvasAgg(figure)
    axes = (
        figure.add_axes((0.015, 0.115, 0.475, 0.755)),
        figure.add_axes((0.510, 0.115, 0.475, 0.755)),
    )
    methods = (trace.fixed, trace.adaptive)
    images = (left, right)
    labels = (trace.left_label, trace.right_label)
    for axis, image, method, label in zip(axes, images, methods, labels, strict=True):
        axis.imshow(image)
        axis.set_axis_off()
        axis.set_title(label, color=_TEXT, fontsize=13, weight="bold", pad=7)
        intervention = float(method.intervention_norm[index])
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3.0 if intervention > 1e-4 else 1.0)
            spine.set_color("#ff781f" if intervention > 1e-4 else "#5a7180")
        axis.text(
            0.018,
            0.975,
            _hud(method, index),
            transform=axis.transAxes,
            va="top",
            ha="left",
            color=_TEXT,
            fontsize=8.4,
            family="monospace",
            linespacing=1.32,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": _PANEL,
                "edgecolor": "#6b8290",
                "alpha": 0.90,
            },
        )

    _draw_descriptor_inset(
        figure, (0.315, 0.145, 0.155, 0.205), trace, trace.fixed, index, descriptor_limit
    )
    _draw_descriptor_inset(
        figure, (0.810, 0.145, 0.155, 0.205), trace, trace.adaptive, index, descriptor_limit
    )
    time = float(trace.time_seconds[index])
    true_wind = np.asarray(trace.true_wind[index])
    estimated = np.asarray(trace.estimated_wind[index])
    title_size = 16 if config.width >= 1200 else 13
    figure.text(
        0.5,
        0.982,
        trace.title,
        color=_TEXT,
        fontsize=title_size,
        weight="bold",
        va="top",
        ha="center",
    )
    if trace.show_wind_change_banner and time >= trace.wind_change_time:
        status = (
            f"t = {time:5.2f} s     WIND CHANGE: [0, 0, 0] -> "
            + np.array2string(true_wind, precision=2)
            + "     estimated "
            + np.array2string(estimated, precision=2)
        )
        status_color = "#ffffff"
        status_box = {"boxstyle": "round,pad=0.35", "facecolor": "#a91788", "edgecolor": "#ffb3ef"}
    else:
        status = (
            f"t = {time:5.2f} s     true wind "
            + np.array2string(true_wind, precision=2)
            + "     estimated wind "
            + np.array2string(estimated, precision=2)
        )
        status_color = _MUTED
        status_box = None
    figure.text(
        0.5,
        0.930,
        status,
        color=status_color,
        fontsize=9.5,
        weight="bold" if time >= trace.wind_change_time else "normal",
        ha="center",
        bbox=status_box,
    )
    figure.text(
        0.5, 0.060, _CERTIFICATE_SENTENCE, color=_TEXT, fontsize=11, weight="bold", ha="center"
    )
    figure.text(
        0.5,
        0.027,
        "yellow dashed: nominal  |  cyan: safe fallback  |  red: unsafe fallback  |  "
        "blue: selected certificate  |  white: executed  |  orange arrow/border: QP intervention",
        color=_MUTED,
        fontsize=8.4,
        ha="center",
    )
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    if frame.shape != (config.height, config.width, 3):
        raise RuntimeError("Agg canvas produced unexpected comparison-frame dimensions")
    return frame


def _hud(method: MethodVideoTrace, index: int) -> str:
    selected = int(method.selected_policy[index])
    selected_label = "nominal" if selected < 0 else f"fallback {selected}"
    safe_count = int(np.count_nonzero(method.fallback_safe[index]))
    intervention = float(method.intervention_norm[index])
    return (
        f"library v{int(method.library_version[index])}  |  "
        f"BPTT steps {int(method.cumulative_gradient_steps[index])}\n"
        f"target loss {float(method.descriptor_target_loss[index]):8.4f}\n"
        f"diversity loss {float(method.diversity_loss[index]):8.4f}\n"
        f"||grad|| {float(method.gradient_norm[index]):8.3e}  "
        f"||dtheta|| {float(method.parameter_update_norm[index]):8.3e}\n"
        f"safe fallbacks {safe_count:2d}/{method.fallback_safe.shape[1]:2d}  "
        f"selected {selected_label}\n"
        f"min library H {float(method.minimum_library_value[index]):+8.3f}  "
        f"||u_QP-u_nom|| {intervention:7.3f}"
    )


def _draw_descriptor_inset(
    figure: Figure,
    bounds: tuple[float, float, float, float],
    trace: ComparisonVideoTrace,
    method: MethodVideoTrace,
    index: int,
    limit: float,
) -> None:
    axis = figure.add_axes(bounds, facecolor=_PANEL)
    targets = np.asarray(trace.descriptor_targets)
    descriptors = np.asarray(method.descriptors[index])
    safe = np.asarray(method.fallback_safe[index])
    axis.scatter(targets[:, 0], targets[:, 1], marker="x", s=17, c="#8fa2af", linewidths=0.8)
    colors = np.where(safe[:, None], np.asarray((0.15, 0.95, 0.82)), np.asarray((1.0, 0.22, 0.23)))
    axis.scatter(descriptors[:, 0], descriptors[:, 1], s=22, c=colors, edgecolors="none")
    selected = int(method.selected_policy[index])
    if selected >= 0:
        axis.scatter(
            descriptors[selected, 0],
            descriptors[selected, 1],
            s=80,
            facecolors="none",
            edgecolors="#4c8dff",
            linewidths=2.0,
        )
    axis.axhline(0.0, color="#50616d", linewidth=0.6)
    axis.axvline(0.0, color="#50616d", linewidth=0.6)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(colors=_MUTED, labelsize=6, length=2)
    for spine in axis.spines.values():
        spine.set_color("#6b8290")
    axis.set_title("realized skill displacement (x,y)", color=_TEXT, fontsize=7, pad=2)


def _descriptor_limit(trace: ComparisonVideoTrace) -> float:
    values = np.concatenate(
        (
            np.asarray(trace.descriptor_targets)[..., :2].reshape(-1, 2),
            np.asarray(trace.fixed.descriptors)[..., :2].reshape(-1, 2),
            np.asarray(trace.adaptive.descriptors)[..., :2].reshape(-1, 2),
        ),
        axis=0,
    )
    return max(0.5, 1.15 * float(np.max(np.abs(values))))


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
