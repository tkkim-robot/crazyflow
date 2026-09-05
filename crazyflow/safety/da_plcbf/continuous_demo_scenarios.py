"""Two deterministic scenarios for the minimal continuous Version-A demonstration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from crazyflow.safety.da_plcbf.continuous_version_a import RuntimeObstacleTrajectories
from crazyflow.safety.da_plcbf.version_a_barriers import RigidBodySafetySet, VersionAModel


@dataclass(frozen=True, slots=True)
class ContinuousDemoScenario:
    """Fixed point-navigation geometry and exactly one optional constant wind step."""

    name: str
    dt: float
    steps: int
    horizon: int
    initial_state: Array
    goal_position: Array
    goal_velocity: Array
    obstacle_initial_centers: Array
    obstacle_velocities: Array
    obstacle_radii: Array
    obstacle_mask: Array
    obstacle_clearance: float
    arena_lower: Array
    arena_upper: Array
    speed_max: Array
    angular_rate_max: Array
    tilt_max_radians: Array
    wind_before: Array
    wind_after: Array
    wind_change_step: int
    skill_displacements: Array
    ego_radius: float = 0.05

    def validate(self) -> None:
        """Validate fixed shapes, initial clearance, and one zero-to-constant wind contract."""
        if not self.name:
            raise ValueError("scenario name must be nonempty")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps <= 1:
            raise ValueError("steps must be an integer greater than one")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if self.initial_state.shape != (13,):
            raise ValueError("initial_state must have shape (13,)")
        if self.goal_position.shape != (3,) or self.goal_velocity.shape != (3,):
            raise ValueError("goal position and velocity must have shape (3,)")
        if self.obstacle_initial_centers.ndim != 2 or self.obstacle_initial_centers.shape[-1] != 3:
            raise ValueError("obstacle_initial_centers must have shape (obstacles, 3)")
        obstacle_count = self.obstacle_initial_centers.shape[0]
        if self.obstacle_velocities.shape != (obstacle_count, 3):
            raise ValueError("obstacle_velocities must match obstacle centers")
        if self.obstacle_radii.shape != (obstacle_count,):
            raise ValueError("obstacle_radii must match obstacle count")
        if self.obstacle_mask.shape != (obstacle_count,) or self.obstacle_mask.dtype != jnp.bool_:
            raise ValueError("obstacle_mask must be boolean shape (obstacles,)")
        if not math.isfinite(self.obstacle_clearance) or self.obstacle_clearance < 0:
            raise ValueError("obstacle_clearance must be finite and nonnegative")
        if not math.isfinite(self.ego_radius) or self.ego_radius < 0:
            raise ValueError("ego_radius must be finite and nonnegative")
        if self.arena_lower.shape != (3,) or self.arena_upper.shape != (3,):
            raise ValueError("arena bounds must have shape (3,)")
        if not bool(jnp.all(self.arena_lower < self.arena_upper)):
            raise ValueError("arena lower bounds must be below upper bounds")
        if self.wind_before.shape != (3,) or self.wind_after.shape != (3,):
            raise ValueError("wind vectors must have shape (3,)")
        if not bool(jnp.all(self.wind_before == 0.0)):
            raise ValueError("the demo wind must begin at exactly zero")
        if self.wind_change_step < 0 or self.wind_change_step > self.steps:
            raise ValueError("wind_change_step must lie in [0, steps]")
        if self.skill_displacements.ndim != 2 or self.skill_displacements.shape[-1] != 3:
            raise ValueError("skill_displacements must have shape (skills, 3)")
        if not bool(jnp.all(jnp.isfinite(self.initial_state))):
            raise ValueError("initial_state must be finite")
        active = self.obstacle_mask
        distances = jnp.linalg.norm(
            self.initial_state[None, :3] - self.obstacle_initial_centers, axis=-1
        )
        if bool(
            jnp.any(
                active
                & (distances <= self.obstacle_radii + self.ego_radius + self.obstacle_clearance)
            )
        ):
            raise ValueError("the scenario must not begin inside an inflated obstacle")


def _state(position: tuple[float, float, float], dtype: jnp.dtype) -> Array:
    return jnp.asarray((*position, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=dtype)


def _skill_displacements(dtype: jnp.dtype) -> Array:
    """Obstacle-independent local maneuver descriptors shared by both scenarios."""
    return jnp.asarray(
        [
            [1.2, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.4, 0.0],
            [0.0, -1.4, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -0.7],
            [0.8, 1.0, 0.5],
            [0.8, -1.0, 0.5],
        ],
        dtype=dtype,
    )


def blocking_static_scenario(*, dtype: jnp.dtype = jnp.float32) -> ContinuousDemoScenario:
    """Return a sphere blocking the nominal path with a small lateral offset.

    The offset avoids an exactly symmetric goal-controller deadlock; the nominal centerline still
    intersects the physical sphere. The centered safety-with-stall case remains a development
    counterexample, since minimum intervention supplies no navigation progress guarantee.
    """
    scenario = ContinuousDemoScenario(
        name="blocking_static",
        dt=0.02,
        steps=400,
        horizon=60,
        initial_state=_state((-2.0, 0.0, 1.4), dtype),
        goal_position=jnp.asarray([2.0, 0.0, 1.4], dtype=dtype),
        goal_velocity=jnp.zeros(3, dtype=dtype),
        obstacle_initial_centers=jnp.asarray([[0.0, 0.15, 1.4]], dtype=dtype),
        obstacle_velocities=jnp.zeros((1, 3), dtype=dtype),
        obstacle_radii=jnp.asarray([0.48], dtype=dtype),
        obstacle_mask=jnp.asarray([True]),
        obstacle_clearance=0.15,
        arena_lower=jnp.asarray([-4.0, -3.0, 0.15], dtype=dtype),
        arena_upper=jnp.asarray([4.0, 3.0, 3.2], dtype=dtype),
        speed_max=jnp.asarray(3.5, dtype=dtype),
        angular_rate_max=jnp.asarray(12.0, dtype=dtype),
        tilt_max_radians=jnp.asarray(0.9, dtype=dtype),
        wind_before=jnp.zeros(3, dtype=dtype),
        wind_after=jnp.zeros(3, dtype=dtype),
        wind_change_step=400,
        skill_displacements=_skill_displacements(dtype),
    )
    scenario.validate()
    return scenario


def constant_wind_scenario(*, dtype: jnp.dtype = jnp.float32) -> ContinuousDemoScenario:
    """Return the one allowed dynamics change: zero wind to one persistent vector at 4 s."""
    scenario = ContinuousDemoScenario(
        name="constant_wind",
        dt=0.02,
        steps=600,
        horizon=60,
        initial_state=_state((-3.0, 0.0, 1.4), dtype),
        goal_position=jnp.asarray([10.0, 0.0, 1.4], dtype=dtype),
        goal_velocity=jnp.zeros(3, dtype=dtype),
        obstacle_initial_centers=jnp.asarray([[5.8, 0.0, 1.2], [7.3, 0.15, 2.65]], dtype=dtype),
        obstacle_velocities=jnp.zeros((2, 3), dtype=dtype),
        obstacle_radii=jnp.asarray([0.55, 0.55], dtype=dtype),
        obstacle_mask=jnp.asarray([True, True]),
        obstacle_clearance=0.15,
        arena_lower=jnp.asarray([-5.0, -3.5, 0.15], dtype=dtype),
        arena_upper=jnp.asarray([12.0, 4.0, 4.0], dtype=dtype),
        speed_max=jnp.asarray(3.5, dtype=dtype),
        angular_rate_max=jnp.asarray(12.0, dtype=dtype),
        tilt_max_radians=jnp.asarray(0.9, dtype=dtype),
        wind_before=jnp.zeros(3, dtype=dtype),
        wind_after=jnp.asarray([0.9, 0.55, 0.0], dtype=dtype),
        wind_change_step=200,
        skill_displacements=_skill_displacements(dtype),
    )
    scenario.validate()
    return scenario


def scenario_obstacle_window(
    scenario: ContinuousDemoScenario, step_index: Array | int
) -> RuntimeObstacleTrajectories:
    """Return the single deterministic obstacle prediction at one controller boundary."""
    step = jnp.asarray(step_index, dtype=scenario.initial_state.dtype)
    offsets = step + jnp.arange(scenario.horizon + 1, dtype=scenario.initial_state.dtype)
    times = offsets * scenario.dt
    centers = (
        scenario.obstacle_initial_centers[None, ...]
        + times[:, None, None] * scenario.obstacle_velocities[None, ...]
    )
    mask = jnp.broadcast_to(
        scenario.obstacle_mask[None, :], (scenario.horizon + 1, scenario.obstacle_mask.size)
    )
    return RuntimeObstacleTrajectories(centers, scenario.obstacle_radii, mask)


def scenario_safety_limits(scenario: ContinuousDemoScenario) -> RigidBodySafetySet:
    """Return the current-limit template; controller code replaces obstacle centres at runtime."""
    return RigidBodySafetySet(
        obstacle_centers=scenario.obstacle_initial_centers,
        obstacle_radii=scenario.obstacle_radii,
        obstacle_mask=scenario.obstacle_mask,
        arena_lower=scenario.arena_lower,
        arena_upper=scenario.arena_upper,
        speed_max=scenario.speed_max,
        angular_rate_max=scenario.angular_rate_max,
        tilt_max_radians=scenario.tilt_max_radians,
    )


def scenario_true_wind(scenario: ContinuousDemoScenario, step_index: Array | int) -> Array:
    """Return zero before the one transition and the same constant vector forever afterward."""
    return jnp.where(
        jnp.asarray(step_index) >= scenario.wind_change_step,
        scenario.wind_after,
        scenario.wind_before,
    )


def model_with_wind(model: VersionAModel, wind_velocity: Array) -> VersionAModel:
    """Bind one true or estimated point wind without changing any other physical parameter."""
    if wind_velocity.shape != (3,):
        raise ValueError("wind_velocity must have shape (3,)")
    return model._replace(wind_velocity=jnp.asarray(wind_velocity, dtype=model.wind_velocity.dtype))


__all__ = [
    "ContinuousDemoScenario",
    "blocking_static_scenario",
    "constant_wind_scenario",
    "model_with_wind",
    "scenario_obstacle_window",
    "scenario_safety_limits",
    "scenario_true_wind",
]
