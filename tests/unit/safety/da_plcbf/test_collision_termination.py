"""Regression tests for conservative-envelope censoring and actual geometry termination."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from crazyflow.safety.da_plcbf.navigation_experiment import (
    NavigationExperimentConfig,
    evaluate_collision_termination,
    summarize_collision_observation,
)
from crazyflow.safety.da_plcbf.navigation_world import (
    NavigationWorld,
    NavigationWorldConfig,
    WaypointProgress,
    advance_waypoints,
    build_navigation_world,
)


def _world() -> NavigationWorld:
    world = build_navigation_world(
        NavigationWorldConfig(obstacle_count=0, waypoint_count=2, duration_seconds=0.12)
    )
    initial = np.array([0, 0, 1.4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    return replace(
        world,
        config=replace(world.config, obstacle_count=1),
        initial_state=initial,
        obstacle_mean_centers=np.array([[0.5, 0, 1.4]]),
        obstacle_amplitudes=np.zeros((1, 3)),
        obstacle_angular_frequencies=np.zeros(1),
        obstacle_phases=np.zeros(1),
        obstacle_radii=np.array([0.1]),
    )


def test_enclosure_intersection_does_not_censor_later_modeled_collision() -> None:
    world = _world()
    states = np.tile(world.initial_state, (5, 1))
    states[:, 0] = [0, 0.15, 0.3, 0.45, 0.6]
    times = np.arange(5) * 0.02
    legacy = evaluate_collision_termination(
        world, times[:3], states[:3], termination_geometry="body_origin_enclosure"
    )
    first = evaluate_collision_termination(world, times[:3], states[:3])
    assert legacy["terminate"]
    assert first["body_origin_enclosure_breach"]
    assert first["requested_shell_breach"]
    assert not first["terminate"]
    assert first["audit"]["actual_xml_sphere_geometry"]["minimum_clearance_lower_bound_m"] > 0
    observation = summarize_collision_observation(
        legacy["audit"],
        termination_geometry="body_origin_enclosure",
        termination="physical_collision",
    )
    assert observation["modeled_collider_collision"] is None
    assert observation["enclosure_termination_censors_later_collider_outcome"]
    second = evaluate_collision_termination(world, times[2:], states[2:])
    assert second["terminate"]
    assert second["collision_kind"] == "modeled_collider_obstacle"
    assert 0.04 < second["first_intersection_time_seconds"] < 0.08
    progress = advance_waypoints(
        world,
        WaypointProgress(),
        world.waypoint_positions[0],
        0.08,
        physical_collision=second["terminate"],
    )
    assert progress.termination == "physical_collision"
    assert progress.completed == 0
    assert progress.arrival_times_seconds == ()


def test_floor_uses_the_same_rotated_sphere_and_collision_precedes_goal() -> None:
    world = _world()
    states = np.tile(world.initial_state, (3, 1))
    states[:, 2] = [0.2, 0.1, 0.02]
    event = evaluate_collision_termination(world, np.array([0.0, 0.02, 0.04]), states)
    assert event["terminate"]
    assert event["collision_kind"] == "modeled_collider_floor"
    assert event["audit"]["actual_xml_ground_geometry"]["minimum_clearance_upper_bound_m"] < 0
    assert not event["body_origin_enclosure_breach"]


def test_zero_straddling_geometry_is_unknown_not_an_asserted_collision() -> None:
    world = _world()
    states = np.tile(world.initial_state, (2, 1))
    event = evaluate_collision_termination(world, np.array([0.0, 0.02]), states)
    sphere = event["audit"]["actual_xml_sphere_geometry"]
    sphere["minimum_clearance_lower_bound_m"] = -1e-7
    sphere["minimum_clearance_upper_bound_m"] = 1e-7
    observation = summarize_collision_observation(
        event["audit"], termination_geometry="modeled_collider", termination="timeout"
    )
    assert observation["modeled_collider_collision"] is None
    assert observation["modeled_collision_observation"] == "unresolved_at_interpolation_error_bound"
    assert not observation["enclosure_termination_censors_later_collider_outcome"]


def test_legacy_default_and_explicit_geometry_validation() -> None:
    world = _world()
    assert NavigationExperimentConfig().termination_geometry == "body_origin_enclosure"
    NavigationExperimentConfig(termination_geometry="modeled_collider").validate(world)
    with pytest.raises(ValueError, match="termination geometry"):
        NavigationExperimentConfig(termination_geometry="shell").validate(world)
