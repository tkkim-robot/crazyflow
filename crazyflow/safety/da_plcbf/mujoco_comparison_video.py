"""Synchronized MuJoCo-rendered replay of recorded Version-A comparisons.

The numerical demo owns simulation and learning.  This module only consumes a fully recorded,
validated trace and renders it.  Keeping that boundary explicit prevents the video path from
silently re-running controllers, inventing rollouts, or changing the timing of the online learner.
"""

from __future__ import annotations

import os
from colorsys import hsv_to_rgb
from dataclasses import dataclass
from itertools import product
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
    full_state: np.ndarray | None = None
    applied_wrench: np.ndarray | None = None
    nominal_wrench: np.ndarray | None = None
    used_midpoint: np.ndarray | None = None
    used_emergency: np.ndarray | None = None
    execution_mode: np.ndarray | None = None
    selected_smooth_value: np.ndarray | None = None
    eligible_candidate_count: np.ndarray | None = None
    executed_policy_dual: np.ndarray | None = None
    actuator_margins: np.ndarray | None = None
    operational_residuals: np.ndarray | None = None
    snapshot_age_seconds: np.ndarray | None = None
    controller_seconds: np.ndarray | None = None
    learner_seconds: np.ndarray | None = None
    missed_deadline: np.ndarray | None = None
    collision_constraint_active: np.ndarray | None = None
    goal_position: np.ndarray | None = None
    waypoint_index: np.ndarray | None = None
    recorded_control_valid: np.ndarray | None = None
    physical_collision_recorded: np.ndarray | None = None
    contact_replay: np.ndarray | None = None
    # Nine operational columns follow version_a_barriers.safety_constraint_names(0).
    # Residuals are [T,N,9] at integration-substep starts; physical margins are [T,N+1,9].
    # Rejected/nonexecuted proposal diagnostics retain NaN/inf. Accepted/executed rows must
    # remain finite; qp_valid / used_fallback / recorded_control_valid supply explicit masks.
    qp_held_operational_residuals: np.ndarray | None = None
    fallback_held_operational_residuals: np.ndarray | None = None
    applied_held_operational_residuals: np.ndarray | None = None
    applied_held_physical_margins: np.ndarray | None = None
    predictive_operational_iterations: np.ndarray | None = None
    initial_qp_held_operational_residual: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class ComparisonVideoTrace:
    """Complete renderer input for a synchronized fixed/adaptive comparison.

    Optional ``repertoire_probes`` records ``time_seconds[P]``, ``reference_position[P,3]``,
    ``left_rollouts/right_rollouts[P,K,H,3]`` (absolute world positions), boolean
    ``left_safe/right_safe[P,K]``, and a descriptive ``source``. Both methods must be evaluated at
    the same recorded reference state/model. Insets subtract only the reference position and use
    fixed metre limits; they never extrapolate, recenter on an endpoint, or separate coincident
    paths. Additional named-method arrays may be retained for selecting a comparison pair.
    """

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
    repertoire_probes: dict[str, object] | None = None
    drone_model: str = "cf2x_T350"
    physical_model_name: str | None = None
    payload_attachment_time_seconds: float | None = None
    payload_half_extents: np.ndarray | None = None
    payload_mass_delta_kg: float | None = None
    payload_base_mass_kg: float | None = None
    wind_event_times_seconds: np.ndarray | None = None
    task_phase: np.ndarray | None = None
    phase_caption: np.ndarray | None = None

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
        if (self.task_phase is None) != (self.phase_caption is None):
            raise ValueError("task_phase and phase_caption must be recorded together")
        if self.task_phase is not None:
            for name in ("task_phase", "phase_caption"):
                values = np.asarray(getattr(self, name))
                if (
                    values.shape != (steps,)
                    or values.dtype.kind not in "US"
                    or any(not str(value).strip() for value in values)
                ):
                    raise ValueError(f"{name} must contain nonempty strings with shape [T]")
            if not np.all(np.isin(self.task_phase, ("hover", "navigation", "contact"))):
                raise ValueError("task_phase must be hover, navigation, or contact")

        if self.wind_event_times_seconds is None:
            before = time < float(self.wind_change_time) - 1e-12
            after = ~before
            if np.any(np.abs(true_wind[before]) > 1e-10):
                raise ValueError("true_wind must be exactly zero before the single wind change")
            if not np.any(after) or not np.allclose(
                true_wind[after], true_wind[after][0], atol=1e-10
            ):
                raise ValueError("true_wind must remain one constant vector after the wind change")
        else:
            events = _finite_array(
                self.wind_event_times_seconds, "wind_event_times_seconds", ndim=1
            )
            if (
                np.any(np.diff(events) <= 0)
                or np.any(events < time[0])
                or np.any(events > time[-1])
            ):
                raise ValueError("wind events must be strictly ordered inside the trace")
            event_indices = np.searchsorted(events, time + 1e-10, side="right")
            if np.any(np.abs(true_wind[event_indices == 0]) > 1e-10):
                raise ValueError("true_wind must be zero before the first declared wind event")
            for interval in np.unique(event_indices):
                values = true_wind[event_indices == interval]
                if not np.allclose(values, values[0], atol=1e-10):
                    raise ValueError("wind may change only at declared event boundaries")

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
        _validate_repertoire_probes(self.repertoire_probes, time, policy_count)
        if not isinstance(self.drone_model, str) or not self.drone_model.strip():
            raise ValueError("drone_model must identify the rendered drone asset")
        if self.physical_model_name is not None and (
            not isinstance(self.physical_model_name, str) or not self.physical_model_name.strip()
        ):
            raise ValueError("physical_model_name must be a nonempty string when recorded")
        payload = (
            self.payload_attachment_time_seconds,
            self.payload_half_extents,
            self.payload_mass_delta_kg,
            self.payload_base_mass_kg,
        )
        if any(value is not None for value in payload):
            if any(value is None for value in payload):
                raise ValueError(
                    "payload replay requires attachment time, extents, added mass, and base mass"
                )
            when = float(self.payload_attachment_time_seconds)
            if not np.isfinite(when) or not time[0] <= when <= time[-1]:
                raise ValueError("payload attachment time must lie inside the recorded interval")
            extents = _shape(
                _finite_array(self.payload_half_extents, "payload_half_extents"),
                (3,),
                "payload_half_extents",
            )
            if np.any(extents <= 0.0) or np.linalg.norm(extents) > self.drone_radius + 1e-9:
                raise ValueError(
                    "positive payload half extents must fit inside the drone collision enclosure"
                )
            _positive_finite(self.payload_mass_delta_kg, "payload_mass_delta_kg")
            _positive_finite(self.payload_base_mass_kg, "payload_base_mass_kg")

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
    mode: str = "diagnostic"
    repertoire_extent_m: float = 1.5
    wind_streak_spacing_m: float = 1.2
    wind_streak_exposure_seconds: float = 0.25
    freeze_after_collision: bool = True
    freeze_after_termination: bool = True
    comparison_note: str = ""
    hover_camera_distance: float | None = None
    camera_transition_seconds: float = 2.0

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
        if self.hover_camera_distance is not None:
            _positive_finite(self.hover_camera_distance, "hover_camera_distance")
        _positive_finite(self.camera_transition_seconds, "camera_transition_seconds")
        if not isinstance(self.comparison_note, str):
            raise TypeError("comparison_note must be a string")
        _positive_finite(self.wind_arrow_scale, "wind_arrow_scale")
        _positive_finite(self.intervention_arrow_scale, "intervention_arrow_scale")
        if not np.isfinite(self.probe_pause_seconds) or self.probe_pause_seconds < 0.0:
            raise ValueError("probe_pause_seconds must be nonnegative and finite")
        if self.probe_pause_time is not None:
            if not np.isfinite(self.probe_pause_time) or self.probe_pause_seconds <= 0.0:
                raise ValueError("probe_pause_time needs a finite time and positive pause duration")
        elif self.probe_pause_seconds != 0.0:
            raise ValueError("probe_pause_seconds requires an explicit probe_pause_time")
        if self.mode not in ("demo", "diagnostic"):
            raise ValueError("mode must be demo or diagnostic")
        for name in ("freeze_after_collision", "freeze_after_termination"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        for name in (
            "repertoire_extent_m",
            "wind_streak_spacing_m",
            "wind_streak_exposure_seconds",
        ):
            _positive_finite(getattr(self, name), name)


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
    collision_stops = (
        tuple(
            _first_recorded_terminal_index(
                trace, method, include_collision=config.freeze_after_collision
            )
            for method in (trace.fixed, trace.adaptive)
        )
        if config.mode == "demo" and config.freeze_after_termination
        else (None, None)
    )
    lookat = _method_camera_lookat(trace, trace.fixed, int(frame_indices[0]), config)
    distance = _camera_distance(trace, config)
    panel_width = max(480, config.width // 2)
    panel_height = max(270, int(config.height * (0.788 if config.mode == "demo" else 0.49)))
    camera = {
        "lookat": lookat,
        "distance": distance,
        "azimuth": float(config.camera_azimuth),
        "elevation": float(config.camera_elevation),
    }
    sim = Sim(
        n_worlds=2,
        n_drones=1,
        drone=trace.drone_model,
        device="cpu",
        fused_mjx_model=False,
        enable_contacts=False,
    )
    sim.max_visual_geom = 5000
    _configure_scene_shadows(sim, trace, config)
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
        previous_indices = (None, None)
        left, right = None, None
        for index in frame_indices:
            display_indices = tuple(
                int(index) if stop is None else min(int(index), stop) for stop in collision_stops
            )
            _set_two_world_poses(sim, trace, index, display_indices=display_indices)
            if not initialized:
                sim.render(
                    mode="rgb_array",
                    world=0,
                    camera=-1,
                    cam_config=camera,
                    width=panel_width,
                    height=panel_height,
                )
                _install_marker_shadow_categories(sim)
                initialized = True
            if previous_indices[0] != display_indices[0]:
                left = _render_world(
                    sim,
                    trace,
                    trace.fixed,
                    display_indices[0],
                    0,
                    camera,
                    panel_width,
                    panel_height,
                    config,
                )
            if previous_indices[1] != display_indices[1]:
                right = _render_world(
                    sim,
                    trace,
                    trace.adaptive,
                    display_indices[1],
                    1,
                    camera,
                    panel_width,
                    panel_height,
                    config,
                )
            previous_indices = display_indices
            yield _compose_frame(trace, config, index, left, right, display_indices=display_indices)
            if not pause_inserted and pause_index is not None and index == pause_index:
                pause_inserted = True
                paused_frame = _compose_frame(
                    trace,
                    config,
                    index,
                    left,
                    right,
                    probe_pause=True,
                    display_indices=display_indices,
                )
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
    ):
        _shape(
            _finite_array(getattr(method, name), f"{prefix}.{name}"), (steps,), f"{prefix}.{name}"
        )
    for name in ("library_version", "cumulative_gradient_steps"):
        values = _integer_vector(getattr(method, name), steps, f"{prefix}.{name}")
        if np.any(values < 0) or np.any(np.diff(values) < 0):
            raise ValueError(f"{prefix}.{name} must be nonnegative and monotone")
    for name in (
        "minimum_library_value",
        "maximum_library_value",
        "selected_policy_value",
        "selected_smooth_value",
    ):
        value = getattr(method, name)
        if value is not None:
            _collision_value_array(
                value, method.collision_constraint_active, steps, f"{prefix}.{name}"
            )
    for name in (
        "selected_policy_dual",
        "executed_policy_dual",
        "snapshot_age_seconds",
        "controller_seconds",
        "learner_seconds",
    ):
        value = getattr(method, name)
        if value is not None:
            values = _shape(_finite_array(value, f"{prefix}.{name}"), (steps,), f"{prefix}.{name}")
            if np.any(values < 0.0):
                raise ValueError(f"{prefix}.{name} must be nonnegative")
    for name in (
        "qp_valid",
        "used_fallback",
        "degraded",
        "used_midpoint",
        "used_emergency",
        "missed_deadline",
        "collision_constraint_active",
        "recorded_control_valid",
        "physical_collision_recorded",
        "contact_replay",
    ):
        value = getattr(method, name)
        if value is not None:
            value = np.asarray(value)
            if value.dtype != np.bool_ or value.shape != (steps,):
                raise ValueError(f"{prefix}.{name} must be boolean with shape [T]")
    if method.contact_replay is not None and np.any(method.contact_replay):
        if method.recorded_control_valid is None or np.any(
            method.contact_replay & method.recorded_control_valid
        ):
            raise ValueError(f"{prefix}: contact replay must have recorded controls inactive")
    if method.recorded_control_valid is not None and np.any(
        np.diff(np.asarray(method.recorded_control_valid, dtype=int)) > 0
    ):
        raise ValueError(f"{prefix}.recorded_control_valid cannot resume after termination")
    for name, width in (
        ("full_state", 13),
        ("applied_wrench", 4),
        ("nominal_wrench", 4),
        ("actuator_margins", 8),
        ("goal_position", 3),
    ):
        value = getattr(method, name)
        if value is not None:
            _shape(_finite_array(value, f"{prefix}.{name}"), (steps, width), f"{prefix}.{name}")
    if method.waypoint_index is not None:
        waypoints = _integer_vector(method.waypoint_index, steps, f"{prefix}.waypoint_index")
        if np.any(waypoints < 0) or np.any(np.diff(waypoints) < 0):
            raise ValueError(f"{prefix}.waypoint_index must be nonnegative and monotone")
        if method.goal_position is None:
            raise ValueError(f"{prefix}.waypoint_index requires recorded goal_position")
    if method.operational_residuals is not None:
        residuals = _finite_array(
            method.operational_residuals, f"{prefix}.operational_residuals", ndim=2
        )
        if residuals.shape[0] != steps:
            raise ValueError(f"{prefix}.operational_residuals must have shape [T,B]")
    hold_steps = None
    qp_required = np.ones(steps, dtype=bool) if method.qp_valid is None else method.qp_valid
    fallback_required = (
        np.ones(steps, dtype=bool) if method.used_fallback is None else method.used_fallback
    )
    applied_required = (
        np.ones(steps, dtype=bool)
        if method.recorded_control_valid is None
        else method.recorded_control_valid
    )
    required_rows = {
        "qp_held_operational_residuals": qp_required,
        "fallback_held_operational_residuals": fallback_required,
        "applied_held_operational_residuals": applied_required,
    }
    for name in (
        "qp_held_operational_residuals",
        "fallback_held_operational_residuals",
        "applied_held_operational_residuals",
    ):
        value = getattr(method, name)
        if value is not None:
            residuals = _masked_finite_diagnostic(
                value, required_rows[name], f"{prefix}.{name}", ndim=3
            )
            if residuals.shape[0] != steps or residuals.shape[2] != 9 or residuals.shape[1] < 1:
                raise ValueError(f"{prefix}.{name} must have shape [T,N,9] with N positive")
            if hold_steps is not None and hold_steps != residuals.shape[1]:
                raise ValueError(f"{prefix} held operational traces must share N")
            hold_steps = residuals.shape[1]
    if method.applied_held_physical_margins is not None:
        margins = _masked_finite_diagnostic(
            method.applied_held_physical_margins,
            applied_required,
            f"{prefix}.applied_held_physical_margins",
            ndim=3,
        )
        if margins.shape[0] != steps or margins.shape[2] != 9 or margins.shape[1] < 2:
            raise ValueError(f"{prefix}.applied_held_physical_margins must have shape [T,N+1,9]")
        if hold_steps is not None and margins.shape[1] != hold_steps + 1:
            raise ValueError(f"{prefix} held physical margins must have N+1 nodes")
    if method.predictive_operational_iterations is not None:
        iterations = _integer_vector(
            method.predictive_operational_iterations,
            steps,
            f"{prefix}.predictive_operational_iterations",
        )
        if np.any((iterations < 0) | (iterations > 4)):
            raise ValueError(f"{prefix}.predictive_operational_iterations must lie in [0,4]")
    if method.initial_qp_held_operational_residual is not None:
        _shape(
            _masked_finite_diagnostic(
                method.initial_qp_held_operational_residual,
                qp_required,
                f"{prefix}.initial_qp_held_operational_residual",
                ndim=1,
            ),
            (steps,),
            f"{prefix}.initial_qp_held_operational_residual",
        )
    if method.eligible_candidate_count is not None:
        eligible = _integer_vector(
            method.eligible_candidate_count, steps, f"{prefix}.eligible_candidate_count"
        )
        if np.any(eligible < 0) or np.any(eligible > policy_count + 1):
            raise ValueError(f"{prefix}.eligible_candidate_count must lie in [0,K+1]")
    if method.execution_mode is not None:
        modes = _integer_vector(method.execution_mode, steps, f"{prefix}.execution_mode")
        if np.any(modes < 0) or np.any(modes > 3):
            raise ValueError(f"{prefix}.execution_mode must lie in [0,3]")
        # Contact rows retain the last airborne mode as archived telemetry, but execute no
        # controller command. Their explicit replay mask already requires control inactivity.
        mode_recorded = (
            np.ones(steps, dtype=bool)
            if method.contact_replay is None
            else ~np.asarray(method.contact_replay)
        )
        for name, mode in (
            ("used_fallback", 1),
            ("used_emergency", 2),
            ("used_midpoint", 3),
            ("qp_valid", 0),
        ):
            values = getattr(method, name)
            if values is not None and np.any(
                mode_recorded & (np.asarray(values) != (modes == mode))
            ):
                raise ValueError(f"{prefix}.{name} must agree with execution_mode")
    if method.qp_rejection_flags is not None:
        flags = np.asarray(method.qp_rejection_flags)
        if flags.dtype != np.bool_ or flags.shape not in (
            (steps, len(QP_REJECTION_REASONS)),
            (steps, 8),
        ):
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
        for name in ("used_fallback", "degraded", "used_emergency", "used_midpoint"):
            value = getattr(method, name)
            if value is not None and np.any(np.asarray(value) & np.asarray(method.qp_valid)):
                raise ValueError(f"{prefix}: valid QP cannot also use {name}")
    return policy_count, horizon


def _masked_finite_diagnostic(
    value: object, required_rows: np.ndarray, name: str, *, ndim: int
) -> np.ndarray:
    """Preserve rejected proposals without admitting a nonfinite executed diagnostic row."""
    values = np.asarray(value)
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{name} must use a floating dtype")
    if values.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}")
    required = np.asarray(required_rows, dtype=bool)
    if values.shape[0] != len(required):
        raise ValueError(f"{name} must match recorded control rows")
    if not np.all(np.isfinite(values[required])):
        raise ValueError(f"{name} must be finite on accepted/executed control rows")
    return values


def held_diagnostic_validity(method: MethodVideoTrace) -> dict[str, object]:
    """Export explicit finite masks/counts alongside untouched proposed and applied arrays."""
    steps = len(method.position)
    masks = {
        "qp_held_operational_residuals": method.qp_valid,
        "initial_qp_held_operational_residual": method.qp_valid,
        "fallback_held_operational_residuals": method.used_fallback,
        "applied_held_operational_residuals": method.recorded_control_valid,
        "applied_held_physical_margins": method.recorded_control_valid,
    }
    result = {}
    for name, mask in masks.items():
        value = getattr(method, name)
        if value is None:
            continue
        finite = np.isfinite(value)
        rows = finite.reshape(steps, -1).all(axis=1)
        required = np.ones(steps, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        result[name] = {
            "finite_rows": rows.tolist(),
            "required_finite_rows": required.tolist(),
            "nonfinite_element_count": int(np.count_nonzero(~finite)),
            "nonfinite_row_count": int(np.count_nonzero(~rows)),
            "nonfinite_required_row_count": int(np.count_nonzero(required & ~rows)),
        }
    return result


def _collision_value_array(value: object, active: object, steps: int, name: str) -> np.ndarray:
    """An empty collision set has +inf H, but no active constraint may contain a nonfinite H."""
    values = np.asarray(value)
    _shape(values, (steps,), name)
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{name} must use a floating dtype")
    inactive = np.zeros(steps, dtype=bool) if active is None else ~np.asarray(active, dtype=bool)
    if inactive.shape != (steps,) or np.any(
        ~np.isfinite(values) & ~(np.isposinf(values) & inactive)
    ):
        raise ValueError(f"{name}: only inactive collision constraints may have +inf values")
    return values


def _validate_repertoire_probes(
    probes: dict[str, object] | None, time: np.ndarray, policy_count: int
) -> None:
    """Validate recorded common-reference paths; the renderer never synthesizes a skill fan."""
    if probes is None:
        return
    if not isinstance(probes, dict):
        raise TypeError("repertoire_probes must be a dictionary")
    times = _finite_array(probes.get("time_seconds"), "repertoire_probes.time_seconds", ndim=1)
    if len(times) == 0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("repertoire probe times must be nonempty and strictly increasing")
    if times[0] < time[0] - 1e-9 or times[-1] > time[-1] + 1e-9:
        raise ValueError("repertoire probe times must lie in the recorded interval")
    _shape(
        _finite_array(probes.get("reference_position"), "repertoire_probes.reference_position"),
        (len(times), 3),
        "repertoire_probes.reference_position",
    )
    if not isinstance(probes.get("source"), str) or not probes["source"].strip():
        raise ValueError("repertoire_probes.source must identify the common reference")
    shape = None
    for side in ("left", "right"):
        rollouts = _finite_array(
            probes.get(f"{side}_rollouts"), f"repertoire_probes.{side}_rollouts", ndim=4
        )
        if (
            rollouts.shape[:2] != (len(times), policy_count)
            or rollouts.shape[2] < 2
            or rollouts.shape[3] != 3
        ):
            raise ValueError("repertoire probe rollouts must have shape [P,K,H,3] with H >= 2")
        if shape is not None and shape != rollouts.shape:
            raise ValueError("left and right repertoire probes must have identical shapes")
        shape = rollouts.shape
        safe = np.asarray(probes.get(f"{side}_safe"))
        if safe.dtype != np.bool_ or safe.shape != (len(times), policy_count):
            raise ValueError("repertoire probe safe masks must be boolean [P,K]")


def _latest_repertoire_probe(trace: ComparisonVideoTrace, index: int) -> int | None:
    if trace.repertoire_probes is None:
        return None
    times = np.asarray(trace.repertoire_probes["time_seconds"])
    latest = int(np.searchsorted(times, trace.time_seconds[index] + 1e-9, side="right")) - 1
    return latest if latest >= 0 else None


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


def _configure_scene_shadows(
    sim: Sim, trace: ComparisonVideoTrace, config: ComparisonRenderConfig
) -> None:
    """Cover the recorded room instead of clipping shadows to the asset's 2.5 m square.

    MuJoCo directional-light shadow half-width is ``stat.extent * vis.map.shadowclip``.
    Playback markers are absent when the model is compiled, so its default extent cannot cover
    their room-scale motion. A sphere around all physical recorded positions, plus the camera
    window, bounds their projections for any directional-light orientation. Configure before
    the render context allocates its shadow texture; leave lights and geometry unchanged.
    """
    model = sim.mj_model
    center = np.asarray(model.stat.center)
    radius = max(
        float(np.max(np.linalg.norm(method.position - center, axis=1))) + trace.drone_radius
        for method in (trace.fixed, trace.adaptive)
    )
    for obstacle in trace.obstacles:
        radius = max(
            radius,
            float(np.max(np.linalg.norm(obstacle.centers - center, axis=1)))
            + obstacle.inflated_radius,
        )
    half_width = radius + max(_camera_distance(trace, config), config.hover_camera_distance or 0.0)
    model.vis.map.shadowclip = max(model.vis.map.shadowclip, half_width / model.stat.extent)
    model.vis.quality.shadowsize = max(model.vis.quality.shadowsize, 4096)


def _install_marker_shadow_categories(sim: Sim) -> None:
    """Keep physical shadows while preventing annotation paths from casting fake shadows.

    MuJoCo excludes ``mjCAT_DECOR`` from the shadow pass. Gymnasium's current marker adapter
    calls mjv_initGeom but drops the supplied category; its legacy adapter preserves it. Apply
    the category after either adapter, without changing lights, depth or material illumination.
    """
    viewer = sim.viewer.viewer
    add_marker = viewer._add_marker_to_scene

    def add_with_category(marker: dict[str, object]) -> None:
        first = viewer.scn.ngeom
        add_marker(marker)
        viewer.scn.geoms[first].category = int(marker.get("category", mujoco.mjtCatBit.mjCAT_DECOR))

    viewer._add_marker_to_scene = add_with_category


def _is_contact_replay(method: MethodVideoTrace, index: int) -> bool:
    return method.contact_replay is not None and bool(method.contact_replay[index])


def _task_phase(trace: ComparisonVideoTrace, method: MethodVideoTrace, index: int) -> str:
    if _is_contact_replay(method, index):
        return "contact"
    if trace.task_phase is not None:
        return str(trace.task_phase[index])
    return "navigation" if method.waypoint_index is not None else "target tracking"


def _task_label(trace: ComparisonVideoTrace, method: MethodVideoTrace, index: int) -> str:
    phase = _task_phase(trace, method, index)
    if phase == "contact":
        return "CONTACT · motors off"
    if phase == "hover":
        return "HOVER · hold the green target"
    if method.waypoint_index is not None:
        return f"NAVIGATE · waypoint {method.waypoint_index[index] + 1}"
    return "TRACK THE GREEN TARGET"


def _wind_caption(trace: ComparisonVideoTrace, index: int) -> str:
    wind = np.asarray(trace.true_wind[index])
    speed = float(np.linalg.norm(wind))
    if speed <= 1e-9:
        return "WIND OFF · still air"
    return f"WIND ON · {speed:.2f} m/s · velocity {_vector_label(wind)} m/s"


def _predicted_direction_count(
    trace: ComparisonVideoTrace, method: MethodVideoTrace, index: int
) -> tuple[int, int]:
    """Count recorded endpoint-direction bins; this geometric diagnostic is not a safety mask.

    Match the competency direction rule: at least 0.10 m displacement and nearest nonzero
    target-displacement direction on the unit sphere. Pure policy validity is not stored in the
    video schema, so this counts finite recorded positions without substituting collision safety.
    """
    paths = np.asarray(method.fallback_rollouts[index], dtype=float)
    targets = np.asarray(trace.descriptor_targets[:, :3], dtype=float)
    if targets.shape[1] != 3:
        raise ValueError("predicted 3D direction count requires three target displacement columns")
    displacement = paths[:, -1] - paths[:, 0]
    norm = np.linalg.norm(displacement, axis=1)
    target_norm = np.linalg.norm(targets, axis=1)
    active = np.all(np.isfinite(paths), axis=(1, 2)) & (norm >= 0.10)
    if not np.any(target_norm > 1e-12):
        return 0, len(targets)
    similarity = (displacement / np.maximum(norm[:, None], 1e-12)) @ (
        targets / np.maximum(target_norm[:, None], 1e-12)
    ).T
    similarity[:, target_norm <= 1e-12] = -np.inf
    occupied = np.unique(np.argmax(similarity, axis=1)[active])
    return len(occupied), len(targets)


def _body_tilt_degrees(method: MethodVideoTrace, index: int) -> float:
    vertical = Rotation.from_quat(method.quaternion_xyzw[index]).as_matrix()[2, 2]
    return float(np.rad2deg(np.arccos(np.clip(vertical, -1, 1))))


def _camera_distance_at(
    trace: ComparisonVideoTrace, config: ComparisonRenderConfig, index: int
) -> float:
    """Use the same metre scale in both panes, widening smoothly at the recorded route release."""
    navigation_distance = _camera_distance(trace, config)
    if config.hover_camera_distance is None or trace.task_phase is None:
        return navigation_distance
    navigation = np.flatnonzero(np.asarray(trace.task_phase[: index + 1]) == "navigation")
    if not len(navigation):
        return config.hover_camera_distance
    elapsed = trace.time_seconds[index] - trace.time_seconds[navigation[0]]
    alpha = np.clip(elapsed / config.camera_transition_seconds, 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)
    return float(
        config.hover_camera_distance + alpha * (navigation_distance - config.hover_camera_distance)
    )


def _method_camera_lookat(
    trace: ComparisonVideoTrace,
    method: MethodVideoTrace,
    index: int,
    config: ComparisonRenderConfig | None = None,
) -> np.ndarray:
    """Anchor hover views to the recorded target, then smoothly follow navigation or a fall."""
    phase = _task_phase(trace, method, index)
    if phase == "hover":
        return _active_goal(trace, method, index).copy()
    position = np.asarray(method.position[index], dtype=np.float64)
    goal_delta = _active_goal(trace, method, index) - position
    horizontal = goal_delta.copy()
    horizontal[2] = 0.0
    norm = float(np.linalg.norm(horizontal))
    lookahead = np.zeros(3) if norm <= 1e-9 else 0.45 * horizontal / norm
    lookat = position + lookahead
    lookat[2] = max(0.55, position[2])
    if phase == "navigation" and trace.task_phase is not None and config is not None:
        first = np.flatnonzero(np.asarray(trace.task_phase[: index + 1]) == "navigation")[0]
        if first > 0 and trace.task_phase[first - 1] == "hover":
            alpha = np.clip(
                (trace.time_seconds[index] - trace.time_seconds[first])
                / config.camera_transition_seconds,
                0,
                1,
            )
            alpha = alpha * alpha * (3 - 2 * alpha)
            held_target = _active_goal(trace, method, first - 1)
            lookat = held_target + alpha * (lookat - held_target)
    return lookat


def _active_goal(trace: ComparisonVideoTrace, method: MethodVideoTrace, index: int) -> np.ndarray:
    """Replay each method's saved active queue entry; never infer arrival from video geometry."""
    return np.asarray(
        trace.goal_position if method.goal_position is None else method.goal_position[index],
        dtype=np.float64,
    )


def _set_two_world_poses(
    sim: Sim,
    trace: ComparisonVideoTrace,
    index: int,
    *,
    display_indices: tuple[int, int] | None = None,
) -> None:
    left_index, right_index = display_indices or (index, index)
    position = jnp.asarray(
        np.stack((trace.fixed.position[left_index], trace.adaptive.position[right_index]))[
            :, None, :
        ]
    )
    quaternion = jnp.asarray(
        np.stack(
            (trace.fixed.quaternion_xyzw[left_index], trace.adaptive.quaternion_xyzw[right_index])
        )[:, None, :]
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
    sim.viewer.viewer.cam.lookat[:] = _method_camera_lookat(trace, method, index, config)
    sim.viewer.viewer.cam.distance = _camera_distance_at(trace, config, index)
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
            category=int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
        )
    _draw_goal_marker(sim, _active_goal(trace, method, index), config)
    _add_polyline(sim, method.position[: index + 1], _HISTORY, radius=0.016)
    streak_origins, streak_vector = _wind_streak_geometry(trace, index, config)
    for origin in streak_origins:
        _add_arrow(sim, origin, streak_vector, np.asarray((0.92, 0.50, 0.89, 0.42)), radius=0.004)
    if _is_contact_replay(method, index):
        # The splice contains actual contact dynamics, with motors off and no new predictions.
        _draw_payload_marker(sim, trace, method, index)
        return
    _add_polyline(sim, method.nominal_rollout[index], _YELLOW, radius=0.009, dashed=True)
    for policy, rollout in enumerate(method.fallback_rollouts[index]):
        clear = bool(method.fallback_safe[index, policy])
        color = _skill_color(policy, clear)
        _add_polyline(sim, rollout, color, radius=0.008, dashed=not clear)
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=np.asarray(rollout[-1], dtype=np.float64),
            size=np.full(3, 0.025),
            rgba=color,
            label="",
        )
    # A white endpoint ring identifies selection without painting over any library trajectory.
    # In particular, identical paths remain identical: no artificial jitter or fan-out is added.
    _add_endpoint_ring(sim, method.selected_rollout[index, -1], _HISTORY, radius=0.052)

    position = np.asarray(method.position[index], dtype=np.float64)
    _draw_payload_marker(sim, trace, method, index)
    if trace.drone_radius > 0.0:
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=position,
            size=np.full(3, trace.drone_radius),
            rgba=np.asarray((0.72, 0.93, 1.0, 0.15)),
            label="",
        )
    if config.mode == "diagnostic":
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
            wind_base,
            config.wind_arrow_scale * trace.true_wind[index],
            _TRUE_WIND,
            radius=0.018,
        )
        _add_arrow(
            sim,
            wind_base - np.asarray((0.0, 0.0, 0.10)),
            config.wind_arrow_scale * _estimated_wind(trace, method, index),
            _ESTIMATED_WIND,
            radius=0.014,
        )
        # This vector records only collective-thrust change along body z; torque is not a vector
        # displacement and is deliberately absent. The diagnostic legend says this explicitly.
        _add_arrow(
            sim,
            position,
            config.intervention_arrow_scale * method.intervention_world[index],
            _INTERVENTION,
            radius=0.015,
        )


def _skill_color(policy: int, clear: bool = True) -> np.ndarray:
    """Identity hue is stable across time, method, library size, and collision status."""
    rgb = hsv_to_rgb((0.07 + policy * 0.6180339887498949) % 1.0, 0.72, 1.0 if clear else 0.68)
    return np.asarray((*rgb, 0.95 if clear else 0.65))


def _draw_goal_marker(sim: Sim, goal_position: np.ndarray, config: ComparisonRenderConfig) -> None:
    """Keep the demo goal center open so the actual drone remains visible on arrival."""
    goal = np.asarray(goal_position, dtype=np.float64)
    if config.mode == "demo":
        phase = np.linspace(0.0, 2.0 * np.pi, 49)
        ring = np.tile(goal, (len(phase), 1))
        ring[:, 0] += 0.14 * np.cos(phase)
        ring[:, 1] += 0.14 * np.sin(phase)
        _add_polyline(sim, ring, _GOAL, radius=0.006)
    else:
        sim.viewer.viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=goal, size=np.full(3, 0.09), rgba=_GOAL, label=""
        )


def _payload_is_attached(trace: ComparisonVideoTrace, index: int) -> bool:
    return (
        trace.payload_attachment_time_seconds is not None
        and trace.time_seconds[index] >= trace.payload_attachment_time_seconds - 1e-9
    )


def _payload_caption(trace: ComparisonVideoTrace, index: int) -> str:
    if trace.payload_attachment_time_seconds is None:
        return ""
    before_g = 1000 * trace.payload_base_mass_kg
    after_g = before_g + 1000 * trace.payload_mass_delta_kg
    percent = 100.0 * trace.payload_mass_delta_kg / trace.payload_base_mass_kg
    if _payload_is_attached(trace, index):
        return (
            f"PAYLOAD ON · mass {before_g:.2f} → {after_g:.2f} g (+{percent:g}%) · "
            "centered load: center of mass unchanged"
        )
    return (
        f"PAYLOAD OFF · mass {before_g:.2f} g · "
        f"centered load attaches at t = {trace.payload_attachment_time_seconds:g} s"
    )


def _draw_payload_marker(
    sim: Sim, trace: ComparisonVideoTrace, method: MethodVideoTrace, index: int
) -> None:
    """Replay only the supplied centered rigid box; no pickup/contact dynamics are invented."""
    if not _payload_is_attached(trace, index):
        return
    sim.viewer.viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=np.asarray(method.position[index], dtype=np.float64),
        size=np.asarray(trace.payload_half_extents, dtype=np.float64),
        mat=Rotation.from_quat(method.quaternion_xyzw[index]).as_matrix().flatten(),
        rgba=np.asarray((0.96, 0.65, 0.12, 1.0)),
        label="",
        category=int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
    )
    # These edges mark the actual box corners at the actual COM. Their thin stroke is a visual
    # outline, not an enlarged simulated package or a change in the collision enclosure.
    corners, edges = _payload_edges(trace, method, index)
    for start, end in edges:
        _add_polyline(sim, corners[[start, end]], np.asarray((1.0, 0.96, 0.42, 1.0)), radius=0.0008)


def _payload_edges(
    trace: ComparisonVideoTrace, method: MethodVideoTrace, index: int
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    signs = np.asarray(tuple(product((-1, 1), repeat=3)))
    corners = signs * np.asarray(trace.payload_half_extents)
    rotation = Rotation.from_quat(method.quaternion_xyzw[index]).as_matrix()
    corners = corners @ rotation.T + method.position[index]
    edges = tuple(
        (a, b)
        for a in range(8)
        for b in range(a + 1, 8)
        if np.count_nonzero(signs[a] != signs[b]) == 1
    )
    return corners, edges


def _draw_payload_detail(
    figure: Figure,
    bounds: tuple[float, float, float, float],
    trace: ComparisonVideoTrace,
    method: MethodVideoTrace,
    index: int,
    *,
    label_size: float,
) -> None:
    """Brief metric projection of the exact centered box and its enclosing body sphere."""
    axis = figure.add_axes(bounds, facecolor=_PANEL)
    corners, edges = _payload_edges(trace, method, index)
    projection = Rotation.from_euler("xz", (25, -35), degrees=True).as_matrix()
    projected = (corners - method.position[index]) @ projection.T
    for start, end in edges:
        axis.plot(
            projected[[start, end], 0] * 100,
            projected[[start, end], 2] * 100,
            color="#fff56b",
            linewidth=1.5,
        )
    axis.add_patch(
        Circle(
            (0, 0),
            trace.drone_radius * 100,
            fill=False,
            edgecolor="#8ac9da",
            linewidth=0.8,
            linestyle="--",
        )
    )
    axis.scatter(0, 0, marker="+", s=14, color=_TEXT)
    extent = 100 * max(0.065, trace.drone_radius * 1.3)
    axis.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
    axis.set_xticks((-5, 0, 5), labels=("−5", "0", "+5 cm"))
    axis.set_yticks(())
    axis.tick_params(colors=_MUTED, labelsize=label_size * 0.82, length=2)
    dimensions = " × ".join(f"{200 * x:g}" for x in trace.payload_half_extents)
    axis.set_title(f"Centered payload · {dimensions} cm", color=_TEXT, fontsize=label_size, pad=3)
    for spine in axis.spines.values():
        spine.set_color("#637f8e")


def _add_endpoint_ring(sim: Sim, endpoint: np.ndarray, rgba: np.ndarray, *, radius: float) -> None:
    phase = np.linspace(0, 2 * np.pi, 17)
    for axes in ((0, 1), (0, 2)):
        ring = np.tile(np.asarray(endpoint, dtype=float), (len(phase), 1))
        ring[:, axes[0]] += radius * np.cos(phase)
        ring[:, axes[1]] += radius * np.sin(phase)
        _add_polyline(sim, ring, rgba, radius=0.003)


def _wind_streak_geometry(
    trace: ComparisonVideoTrace, index: int, config: ComparisonRenderConfig
) -> tuple[np.ndarray, np.ndarray]:
    """One prescribed uniform field, advected in world coordinates for both panels.

    Seeds lie on a deterministic world grid. Their displacement is the time integral of recorded
    true wind, with no body-following motion, estimated-wind feedback, or invented turbulence.
    The union of both camera neighborhoods is used only to cull invisible geometry.
    """
    wind = np.asarray(trace.true_wind[index], dtype=float)
    vector = wind * config.wind_streak_exposure_seconds
    if np.linalg.norm(vector) < 1e-9:
        return np.empty((0, 3)), vector
    displacement = np.sum(
        np.diff(trace.time_seconds[: index + 1])[:, None] * trace.true_wind[:index], axis=0
    )
    spacing = config.wind_streak_spacing_m
    half_cells = max(2, int(np.ceil(_camera_distance(trace, config) / spacing / 2)))
    cells: set[tuple[int, int, int]] = set()
    for method in (trace.fixed, trace.adaptive):
        center = np.floor((method.position[index] - displacement) / spacing).astype(int)
        for x in range(center[0] - half_cells, center[0] + half_cells + 1):
            for y in range(center[1] - half_cells, center[1] + half_cells + 1):
                for z in range(center[2] - 1, center[2] + 2):
                    if (x + 2 * y + z) % 3 == 0:
                        cells.add((x, y, z))
    points = (
        np.asarray(sorted(cells), dtype=float) * spacing
        + displacement
        + np.asarray((0.0, 0.0, 0.45))
    )
    # Do not put atmospheric tracers underneath the floor.
    points = points[points[:, 2] > 0.15]
    return points, vector


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
    direction = vector / length
    transverse = np.cross(direction, np.asarray((0.0, 0.0, 1.0)))
    if np.linalg.norm(transverse) < 1e-8:
        transverse = np.cross(direction, np.asarray((0.0, 1.0, 0.0)))
    transverse /= np.linalg.norm(transverse)
    head_length = min(length * 0.3, max(0.07, radius * 6))
    for sign in (-1, 1):
        wing = stop - head_length * direction + sign * 0.5 * head_length * transverse
        draw_capsule(sim, stop, wing, radius=radius, rgba=rgba)


def _compose_frame(
    trace: ComparisonVideoTrace,
    config: ComparisonRenderConfig,
    index: int,
    left: np.ndarray,
    right: np.ndarray,
    *,
    probe_pause: bool = False,
    display_indices: tuple[int, int] | None = None,
) -> np.ndarray:
    if config.mode == "demo":
        return _compose_demo_frame(
            trace,
            config,
            index,
            left,
            right,
            probe_pause=probe_pause,
            display_indices=display_indices,
        )
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
        elif _has_recorded_margin_violation(trace, method, index):
            clearance = f"MARGIN VIOLATION EARLIER  |  shell {shell_margin:+.2f} m now"
            status_color = "#ffc17b"
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
            f"point-model wind {_vector_label(estimate)} m/s\n"
            f"library v{int(method.library_version[index])}  |  "
            f"{int(method.cumulative_gradient_steps[index])} learning updates\n"
            f"skill target loss {float(method.descriptor_target_loss[index]):.3f}\n"
            "orange arrow: collective thrust only; torque omitted",
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
        _model_caption(trace)
        + "  |  finite-horizon collision values under each recorded point model",
        color=_MUTED,
        fontsize=9 * scale,
        ha="center",
    )
    banner = f"t = {time:5.2f} s   |   {_wind_caption(trace, index)}"
    if payload_caption := _payload_caption(trace, index):
        banner += f"   |   {payload_caption}"
    if probe_pause:
        banner = (
            f"PROBE PAUSE  |  simulation t = {time:.2f} s  |  recorded state held for inspection"
        )
    figure.text(
        0.5,
        0.906,
        banner,
        color="#f4d6ed" if time >= trace.wind_change_time else _MUTED,
        fontsize=8.7 * scale,
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
            "skill hues: identity; dashed/dim: blocked or invalid  |  ring: selected endpoint  |  "
            "orange: collective thrust change only (no torque)",
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


def _model_caption(trace: ComparisonVideoTrace) -> str:
    if trace.physical_model_name == trace.drone_model:
        return f"MuJoCo-rendered Version-A replay · {trace.drone_model}"
    physical = trace.physical_model_name or "unrecorded physical model"
    return f"Version-A replay · {trace.drone_model} visual proxy for {physical}"


def _demo_status(
    trace: ComparisonVideoTrace, method: MethodVideoTrace, index: int
) -> tuple[str, str]:
    """Keep both body collision and shell breach visible after an encounter has passed."""
    if _has_recorded_collision(trace, method, index):
        return "COLLISION RECORDED", "#ff7882"
    if _has_recorded_margin_violation(trace, method, index):
        return "SAFETY MARGIN VIOLATED", "#ffc17b"
    return _demo_execution_status(method, index)


def _demo_execution_status(method: MethodVideoTrace, index: int) -> tuple[str, str]:
    if _is_contact_replay(method, index):
        return "CONTACT REPLAY · MOTORS OFF", "#ffbdc1"
    if method.recorded_control_valid is not None and not method.recorded_control_valid[index]:
        return "TERMINATED · NO FURTHER CONTROL", _MUTED
    if method.used_midpoint is not None and bool(method.used_midpoint[index]):
        return "MIDPOINT EMERGENCY · UNCERTIFIED", "#ff7882"
    if method.used_emergency is not None and bool(method.used_emergency[index]):
        return "EMERGENCY POLICY · UNCERTIFIED", "#ff7882"
    if method.degraded is not None and bool(method.degraded[index]):
        return "UNCERTIFIED COMMAND", "#ff7882"
    if method.control_mode == "nominal":
        return "NOMINAL CONTROL", "#ffe085"
    if method.maximum_library_value is not None and method.maximum_library_value[index] < 0:
        return "NO COLLISION CERTIFICATE", "#ff7882"
    if method.used_fallback is not None and bool(method.used_fallback[index]):
        return "FALLBACK EXECUTING", "#ffc17b"
    if method.qp_valid is not None and bool(method.qp_valid[index]):
        return "QP CONTROL", "#94efc4"
    return "EXECUTION UNRECORDED", _MUTED


def _demo_coverage(method: MethodVideoTrace, index: int) -> str:
    if _is_contact_replay(method, index):
        return "MuJoCo contacts · predictions stopped"
    if method.recorded_control_valid is not None and not method.recorded_control_valid[index]:
        return "last recorded skill predictions"
    if (
        method.collision_constraint_active is not None
        and not method.collision_constraint_active[index]
    ):
        return "No active collision constraint"
    count, total = int(np.count_nonzero(method.fallback_safe[index])), method.fallback_safe.shape[1]
    return f"{count}/{total} valid collision-clear skills"


def _compose_demo_frame(
    trace: ComparisonVideoTrace,
    config: ComparisonRenderConfig,
    index: int,
    left: np.ndarray,
    right: np.ndarray,
    *,
    probe_pause: bool = False,
    display_indices: tuple[int, int] | None = None,
) -> np.ndarray:
    """Two dominant scenes with explicit task, disturbance and centered-payload explanations."""
    figure = Figure(
        figsize=(config.width / 100.0, config.height / 100.0), dpi=100, facecolor=_BACKGROUND
    )
    canvas = FigureCanvasAgg(figure)
    scale = config.width / 1600.0
    display_indices = display_indices or (index, index)
    for x, image, method, label, side in zip(
        (0.006, 0.503),
        (left, right),
        (trace.fixed, trace.adaptive),
        (trace.left_label, trace.right_label),
        ("left", "right"),
        strict=True,
    ):
        axis = figure.add_axes((x, 0.128, 0.491, 0.788))
        axis.imshow(image, aspect="auto")
        axis.set_axis_off()
        method_index = display_indices[0 if side == "left" else 1]
        latest = _latest_repertoire_probe(trace, method_index)
        hovering = _task_phase(trace, method, method_index) == "hover"
        status, color = _demo_execution_status(method, method_index)
        axis.text(
            0.022,
            0.972,
            label,
            color=_TEXT,
            fontsize=12 * scale,
            weight="bold",
            va="top",
            transform=axis.transAxes,
            bbox={"facecolor": _PANEL, "edgecolor": "none", "alpha": 0.80, "pad": 5},
        )
        axis.text(
            0.022,
            0.827 if hovering else 0.872,
            f"{status}  ·  {_demo_coverage(method, method_index)}",
            color=color,
            fontsize=(8.3 if hovering else 9.5) * scale,
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": _PANEL, "edgecolor": "none", "alpha": 0.80, "pad": 4},
        )
        history, history_color = _demo_status(trace, method, method_index)
        if history != status:
            axis.text(
                0.022,
                0.782 if hovering else 0.823,
                history,
                color=history_color,
                fontsize=9.5 * scale,
                transform=axis.transAxes,
                va="top",
                weight="bold",
                bbox={"facecolor": _PANEL, "edgecolor": "none", "alpha": 0.85, "pad": 4},
            )
        if method_index < index:
            axis.text(
                0.022,
                0.737 if hovering else 0.776,
                f"REPLAY FROZEN · recorded t={trace.time_seconds[method_index]:.2f} s",
                color="#ffbdc1",
                fontsize=9 * scale,
                transform=axis.transAxes,
                va="top",
                bbox={"facecolor": _PANEL, "edgecolor": "none", "alpha": 0.85, "pad": 4},
            )
        axis.text(
            0.022,
            0.922,
            _task_label(trace, method, method_index),
            color="#a7ffb8" if not _is_contact_replay(method, method_index) else "#ffbdc1",
            fontsize=10.5 * scale,
            va="top",
            transform=axis.transAxes,
            bbox={"facecolor": _PANEL, "edgecolor": "none", "alpha": 0.82, "pad": 4},
        )
        if hovering:
            occupied, total = _predicted_direction_count(trace, method, method_index)
            axis.text(
                0.022,
                0.872,
                f"Predicted directions: {occupied}/{total} · "
                f"Body tilt {_body_tilt_degrees(method, method_index):.1f}°",
                color=_TEXT,
                fontsize=10.5 * scale,
                va="top",
                transform=axis.transAxes,
                bbox={"facecolor": _PANEL, "edgecolor": "none", "alpha": 0.82, "pad": 4},
            )
        online_updates = int(
            method.cumulative_gradient_steps[method_index] - method.cumulative_gradient_steps[0]
        )
        axis.text(
            0.973,
            0.922,
            f"Online updates: {online_updates}",
            color=_TEXT,
            fontsize=9.5 * scale,
            va="top",
            ha="right",
            transform=axis.transAxes,
            bbox={"facecolor": _PANEL, "edgecolor": "none", "alpha": 0.82, "pad": 4},
        )
        if latest is not None and probe_pause and not _is_contact_replay(method, method_index):
            # Reference views appear only during an explicitly requested measurement pause.
            # The normal scene has no unexplained payload sketch or miniature reference plots.
            width = 0.126
            height = width * config.width / config.height
            base_x = x + 0.013
            base_y = 0.155
            for projection, offset in (((0, 1), 0), ((0, 2), 1)):
                _draw_repertoire_projection(
                    figure,
                    (base_x + offset * (width + 0.016), base_y, width, height),
                    trace,
                    side,
                    latest,
                    projection,
                    config.repertoire_extent_m,
                    label_size=8 * scale,
                )
            measured = float(np.asarray(trace.repertoire_probes["time_seconds"])[latest])
            reference_label = (
                "Neutral reference"
                if "neutral" in str(trace.repertoire_probes["source"]).lower()
                else "Common reference"
            )
            figure.text(
                base_x,
                base_y + height + 0.021,
                f"{reference_label} · same recorded state · measured {measured:.2f} s",
                color=_TEXT,
                fontsize=8.8 * scale,
                bbox={"facecolor": _PANEL, "edgecolor": "none", "alpha": 0.9, "pad": 3},
            )
    time = float(trace.time_seconds[index])
    time_label = f"t = {time:.2f} s"
    if probe_pause:
        time_label = f"PROBE PAUSE · simulation {time_label}"
    figure.text(
        0.018, 0.980, trace.title, color=_TEXT, fontsize=13 * scale, weight="bold", va="top"
    )
    figure.text(
        0.985,
        0.978,
        time_label,
        color="#f5e3ab" if probe_pause else _TEXT,
        fontsize=9.5 * scale,
        ha="right",
        va="top",
    )
    phase_caption = (
        str(trace.phase_caption[index])
        if trace.phase_caption is not None
        else "Recorded flight · both methods share the prescribed environment"
    )
    figure.text(0.018, 0.943, phase_caption, color=_TEXT, fontsize=11.5 * scale, va="center")
    figure.text(
        0.5,
        0.103,
        _wind_caption(trace, index),
        color="#f5a4e2" if np.linalg.norm(trace.true_wind[index]) > 1e-9 else _TEXT,
        fontsize=10.5 * scale,
        ha="center",
    )
    figure.text(
        0.5,
        0.075,
        _payload_caption(trace, index) or "NO PAYLOAD CHANGE",
        color="#ffe59a" if _payload_is_attached(trace, index) else _MUTED,
        fontsize=10.5 * scale,
        ha="center",
    )
    figure.text(
        0.5,
        0.046,
        "White: flown path · colors: skill predictions · yellow dashed: nominal · "
        "green ring: target · white ring: selected skill",
        color=_TEXT,
        fontsize=8.2 * scale,
        ha="center",
    )
    figure.text(
        0.5,
        0.021,
        config.comparison_note
        or (
            _model_caption(trace)
            + f" · body radius {trace.drone_radius:.3f} m · orange shell: required clearance"
        ),
        color=_MUTED,
        fontsize=(9.3 if config.comparison_note else 7.7) * scale,
        ha="center",
    )
    canvas.draw()
    return np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()


def _draw_repertoire_projection(
    figure: Figure,
    bounds: tuple[float, float, float, float],
    trace: ComparisonVideoTrace,
    side: str,
    probe_index: int,
    projection: tuple[int, int],
    extent_m: float,
    *,
    label_size: float,
) -> None:
    probes = trace.repertoire_probes
    if probes is None:
        return
    axis = figure.add_axes(bounds, facecolor=_PANEL)
    reference = np.asarray(probes["reference_position"])[probe_index]
    paths = np.asarray(probes[f"{side}_rollouts"])[probe_index] - reference
    safe = np.asarray(probes[f"{side}_safe"])[probe_index]
    clipped = False
    for policy, (path, clear) in enumerate(zip(paths, safe, strict=True)):
        color = _skill_color(policy, bool(clear))
        xy = path[:, projection]
        axis.plot(xy[:, 0], xy[:, 1], color=color, linewidth=0.9, linestyle="-" if clear else "--")
        axis.scatter(*xy[-1], color=color, s=5, zorder=3)
        clipped |= bool(np.any(np.abs(xy) > extent_m))
    axis.scatter(0, 0, s=7, color=_TEXT, marker="+", zorder=4)
    axis.set_xlim(-extent_m, extent_m)
    axis.set_ylim(-extent_m, extent_m)
    axis.set_aspect("equal", adjustable="box")
    plane = "XY" if projection == (0, 1) else "XZ"
    axis.set_title(
        f"{plane} · metres" + (" (clipped)" if clipped else ""),
        color=_TEXT,
        fontsize=label_size,
        pad=2,
    )
    axis.set_xticks((-extent_m, 0, extent_m))
    axis.set_yticks((-extent_m, 0, extent_m))
    axis.tick_params(colors=_MUTED, labelsize=label_size * 0.8, length=2, pad=1)
    axis.grid(color="#748496", alpha=0.25, linewidth=0.5)
    for spine in axis.spines.values():
        spine.set_color("#405564")


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
    if method.recorded_control_valid is not None and not method.recorded_control_valid[index]:
        return "TERMINATED · NO FURTHER CONTROL", _MUTED
    if method.control_mode == "nominal":
        return "EXECUTING NOMINAL", "#ffe085"
    if method.used_midpoint is not None and bool(method.used_midpoint[index]):
        return "MIDPOINT EMERGENCY · UNCERTIFIED", "#ff6470"
    if method.used_emergency is not None and bool(method.used_emergency[index]):
        return "EMERGENCY POLICY · UNCERTIFIED", "#ff6470"
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
    flags = method.qp_rejection_flags[index]
    for reason, active in zip(QP_REJECTION_REASONS[: len(flags)], flags, strict=True):
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
    if method.physical_collision_recorded is not None:
        return bool(np.any(method.physical_collision_recorded[: index + 1]))
    return _has_recorded_clearance_violation(trace, method, index, shell=False)


def _first_recorded_collision_index(
    trace: ComparisonVideoTrace, method: MethodVideoTrace
) -> int | None:
    """End sample of the first colliding recorded segment; no synthetic impact pose is made."""
    if method.physical_collision_recorded is not None:
        indices = np.flatnonzero(method.physical_collision_recorded)
        return int(indices[0]) if len(indices) else None
    hits = np.zeros(len(trace.time_seconds), dtype=bool)
    for obstacle in trace.obstacles:
        relative = method.position - obstacle.centers
        radius = obstacle.physical_radius + trace.drone_radius
        hits |= np.linalg.norm(relative, axis=1) < radius
        start, delta = relative[:-1], np.diff(relative, axis=0)
        squared = np.sum(delta * delta, axis=1)
        fraction = np.clip(-np.sum(start * delta, axis=1) / np.maximum(squared, 1e-12), 0.0, 1.0)
        hits[1:] |= np.linalg.norm(start + fraction[:, None] * delta, axis=1) < radius
    if method.recorded_control_valid is not None:
        terminal = np.flatnonzero(~method.recorded_control_valid)
        if len(terminal):
            hits[terminal[0] + 1 :] = False
    indices = np.flatnonzero(hits)
    return int(indices[0]) if len(indices) else None


def _first_recorded_terminal_index(
    trace: ComparisonVideoTrace, method: MethodVideoTrace, *, include_collision: bool = True
) -> int | None:
    """Freeze each terminated replay before later exogenous events affect padded poses."""
    terminal = []
    if method.recorded_control_valid is not None:
        terminal = np.flatnonzero(~method.recorded_control_valid).tolist()
    if include_collision:
        collision = _first_recorded_collision_index(trace, method)
        if collision is not None:
            terminal.append(collision)
    return min(terminal) if terminal else None


def _has_recorded_margin_violation(
    trace: ComparisonVideoTrace, method: MethodVideoTrace, index: int
) -> bool:
    return _has_recorded_clearance_violation(trace, method, index, shell=True)


def _has_recorded_clearance_violation(
    trace: ComparisonVideoTrace, method: MethodVideoTrace, index: int, *, shell: bool
) -> bool:
    if method.recorded_control_valid is not None:
        terminal = np.flatnonzero(~method.recorded_control_valid)
        if len(terminal):
            index = min(index, int(terminal[0]))
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
        radius = (
            obstacle.inflated_radius if shell else obstacle.physical_radius + trace.drone_radius
        )
        if np.any(np.linalg.norm(relative, axis=1) < radius):
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
        if method.executed_policy_dual is None:
            certificate = (
                f"selected H {selected} ({policy_name})   |   PL-CBF dual {dual} (proposal)"
            )
        else:
            certificate = (
                f"selected H {selected} ({policy_name})   |   "
                f"executed PL-CBF dual {float(method.executed_policy_dual[index]):.2e}"
            )
        if method.selected_smooth_value is not None:
            certificate += f"   |   smooth {_value_label(method.selected_smooth_value, index)}"
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
    coverage = (
        "No active collision constraint"
        if method.collision_constraint_active is not None
        and not method.collision_constraint_active[index]
        else f"library H = max {maximum}   |   "
        f"valid collision-clear skills {safe_count}/{method.fallback_safe.shape[1]}"
    )
    if method.eligible_candidate_count is not None:
        coverage += f"   |   eligible {int(method.eligible_candidate_count[index])}"
    return (
        f"{coverage}\n"
        f"{certificate}\n"
        f"wrench change norm {float(method.intervention_norm[index]):.3f} (mixed units)   |   "
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
        joined = joined[np.isfinite(joined)]
        if joined.size == 0:
            joined = np.asarray((-0.03, 0.03))
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
        _active_goal(trace, method, index)[None, :2],
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
    axis.scatter(
        *_active_goal(trace, method, index)[:2], s=30, marker="*", color="#7bff9f", zorder=4
    )
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
