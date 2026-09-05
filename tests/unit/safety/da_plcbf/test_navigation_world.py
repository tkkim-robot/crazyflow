from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.continuous_version_a import RuntimeObstacleTrajectories
from crazyflow.safety.da_plcbf.navigation_world import (
    CF21B_BODY_ORIGIN_ENCLOSURE_M,
    NavigationWorldConfig,
    PayloadEvent,
    WaypointProgress,
    WindEvent,
    advance_waypoints,
    build_navigation_world,
    nominal_encounter_metrics,
)
from crazyflow.safety.da_plcbf.online_constant_wind import build_cf21b_version_a_resources


@pytest.mark.parametrize("count", (0, 8, 16, 32))
def test_world_is_exogenous_bounded_and_has_analytic_prediction_velocities(count: int) -> None:
    config = NavigationWorldConfig(seed=13, obstacle_count=count)
    world = build_navigation_world(config)
    duplicate = build_navigation_world(config)
    before = world.obstacle_kinematics(np.asarray((0, 4.0, 20.0)))
    # A method completing a different number of waypoints cannot consume world randomness.
    progress = advance_waypoints(world, WaypointProgress(), world.waypoint_positions[0], 2.0)
    assert progress.completed == 1
    world.obstacle_kinematics(np.linspace(30, 5, 91))
    after = world.obstacle_kinematics(np.asarray((0, 4.0, 20.0)))
    for actual, repeated, independently_generated in zip(
        before, after, duplicate.obstacle_kinematics(np.asarray((0, 4.0, 20.0))), strict=True
    ):
        np.testing.assert_array_equal(actual, repeated)
        np.testing.assert_array_equal(actual, independently_generated)
    centers, velocities = world.obstacle_kinematics(np.linspace(0, 40, 1001))
    assert np.all(centers - world.obstacle_radii[None, :, None] >= config.arena_lower)
    assert np.all(centers + world.obstacle_radii[None, :, None] <= config.arena_upper)
    t, epsilon = 3.127, 1e-5
    numerical = (
        world.obstacle_kinematics(t + epsilon)[0] - world.obstacle_kinematics(t - epsilon)[0]
    ) / (2 * epsilon)
    np.testing.assert_allclose(world.obstacle_kinematics(t)[1], numerical, atol=1e-9)
    first = world.obstacle_prediction(2.0, horizon=60)
    second = world.obstacle_prediction(2.04, horizon=60)
    np.testing.assert_array_equal(first.centers[2:], second.centers[:-2])
    np.testing.assert_array_equal(first.velocities[2:], second.velocities[:-2])
    assert np.all(first.mask)
    assert np.all(np.abs(np.diff(world.waypoint_positions[:, 2])) > 0.8)
    assert np.array_equal(
        np.asarray(json.loads(json.dumps(world.metadata()))["waypoint_positions"]),
        world.waypoint_positions,
    )
    with pytest.raises(ValueError, match="read-only"):
        world.waypoint_positions[0, 0] = 0


def test_composable_events_are_synchronized_on_command_boundaries() -> None:
    config = NavigationWorldConfig(
        wind_events=(WindEvent(4.0, (2.0, 0.0, 0.0)), WindEvent(8.0, (0.0, -1.0, 0.0))),
        payload_events=(PayloadEvent(4.0, 0.25), PayloadEvent(12.0, 0.10)),
    )
    world = build_navigation_world(config)
    resources = build_cf21b_version_a_resources()
    base = resources.model
    mass = float(base.mass)
    for t, expected_wind, fraction in (
        (3.98, (0, 0, 0), 1),
        (4.0, (2, 0, 0), 1.25),
        (4.02, (2, 0, 0), 1.25),
        (8.0, (0, -1, 0), 1.25),
        (12.0, (0, -1, 0), 1.35),
    ):
        result = world.dynamics_at(t, base)
        np.testing.assert_allclose(result.model.mass, mass * fraction, rtol=1e-6)
        np.testing.assert_array_equal(result.model.wind_velocity, expected_wind)
        expected_inertia = np.asarray(base.inertia) + np.eye(3) * (
            mass * (fraction - 1) * 2 * 0.025**2 / 3
        )
        np.testing.assert_allclose(result.model.inertia, expected_inertia, rtol=1e-6)
        np.testing.assert_allclose(
            result.model.inertia @ result.model.inertia_inv, np.eye(3), atol=1e-6
        )
        assert result.ego_radius == config.ego_radius
    # Model queries are idempotent; a second method never attaches the payload a second time.
    np.testing.assert_array_equal(
        world.dynamics_at(12, base).model.mass, world.dynamics_at(12, base).model.mass
    )
    with pytest.raises(ValueError, match="control boundaries"):
        build_navigation_world(replace(config, payload_events=(PayloadEvent(4.01),)))
    with pytest.raises(ValueError, match="strictly ordered"):
        build_navigation_world(replace(config, wind_events=tuple(reversed(config.wind_events))))


def test_shared_queue_arrival_rule_and_collision_censoring() -> None:
    world = build_navigation_world(NavigationWorldConfig(waypoint_count=4))
    left, right = WaypointProgress(), WaypointProgress()
    left = advance_waypoints(world, left, world.waypoint_positions[0], 1.0)
    np.testing.assert_array_equal(left.active_goal(world), world.waypoint_positions[1])
    np.testing.assert_array_equal(right.active_goal(world), world.waypoint_positions[0])
    # A swept collision in the incoming interval takes precedence even at the goal center.
    left = advance_waypoints(world, left, world.waypoint_positions[1], 2.0, physical_collision=True)
    assert left.completed == 1 and left.termination == "physical_collision"
    assert advance_waypoints(world, left, world.waypoint_positions[1], 3.0) is left
    for index, goal in enumerate(world.waypoint_positions):
        right = advance_waypoints(world, right, goal, 2.0 + index)
    assert right.completed == 4 and right.termination == "completed"
    assert right.arrival_times_seconds == (2.0, 3.0, 4.0, 5.0)
    timeout = advance_waypoints(world, WaypointProgress(), world.waypoint_positions[0], 40.0)
    assert timeout.completed == 0 and timeout.termination == "timeout"


def test_hover_phase_does_not_credit_waypoints_but_still_censors_collision() -> None:
    world = build_navigation_world()
    progress = WaypointProgress()
    assert (
        advance_waypoints(
            world, progress, world.waypoint_positions[0], 3.0, navigation_enabled=False
        )
        is progress
    )
    crashed = advance_waypoints(
        world,
        progress,
        world.waypoint_positions[0],
        3.0,
        navigation_enabled=False,
        physical_collision=True,
    )
    assert crashed.termination == "physical_collision" and crashed.completed == 0
    assert (
        advance_waypoints(
            world, progress, world.waypoint_positions[0], 4.0, navigation_enabled=True
        ).completed
        == 1
    )


def test_default_envelope_encloses_the_actual_offset_asset_collider() -> None:
    import crazyflow.safety.da_plcbf.navigation_world as module

    path = Path(module.__file__).parents[2] / "drones/cf21B_500.xml"
    tree = ElementTree.parse(path)
    collider = tree.find(".//geom[@name='col_sphere']")
    radius = float(collider.attrib["size"])
    offset = np.fromstring(collider.attrib["pos"], sep=" ")
    np.testing.assert_allclose(CF21B_BODY_ORIGIN_ENCLOSURE_M, radius + np.linalg.norm(offset))
    world = build_navigation_world()
    assert world.metadata()["ego_enclosure"]["encloses_asset_collider"]
    assert NavigationWorldConfig().ego_radius >= radius + np.linalg.norm(offset) - 1e-9


def test_encounter_metrics_detect_swept_threats_and_masked_padding() -> None:
    positions = np.asarray(((-1.0, 0, 0), (1.0, 0, 0), (2.0, 0, 0)))
    centers = np.zeros((3, 2, 3))
    centers[:, 1] = np.nan
    obstacles = RuntimeObstacleTrajectories(
        jnp.asarray(centers),
        jnp.asarray((0.2, 0.2)),
        jnp.asarray(((True, False), (True, False), (True, False))),
    )
    result = nominal_encounter_metrics(
        positions, obstacles, dt=1.0, ego_radius=0.05, obstacle_clearance=0.15
    )
    assert result["nominal_blocked"]
    assert result["threatening_obstacle_count"] == 1
    assert result["peak_simultaneous_threats"] == 1
    np.testing.assert_allclose(result["minimum_predicted_clearance_m"], -0.4, atol=1e-7)
    np.testing.assert_allclose(result["closest_time_to_contact_seconds"], 0.3, atol=1e-7)
    inactive = obstacles._replace(mask=jnp.zeros_like(obstacles.mask))
    assert (
        nominal_encounter_metrics(
            positions, inactive, dt=1, ego_radius=0.05, obstacle_clearance=0.15
        )["closest_time_to_contact_seconds"]
        is None
    )
    # A mask entering and leaving at one prediction node still contributes its active value;
    # no segment is inferred through the neighboring masked observations.
    entering_centers = np.asarray(obstacles.centers).copy()
    entering_centers[1, 0] = positions[1]
    entering = obstacles._replace(
        centers=jnp.asarray(entering_centers),
        mask=jnp.asarray(((False, False), (True, False), (False, False))),
    )
    entered = nominal_encounter_metrics(
        positions, entering, dt=1, ego_radius=0.05, obstacle_clearance=0.15
    )
    assert entered["nominal_blocked"]
    assert entered["closest_time_to_contact_seconds"] == 1.0
