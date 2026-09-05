from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from crazyflow.safety.da_plcbf.case_study_world import (
    GuardSphere,
    HoverEncounterConfig,
    IncomingSphere,
    audit_recorded_collider_clearance,
    build_hover_encounter_world,
)


def test_prescribed_crossing_and_branch_share_absolute_clock_and_roundtrip() -> None:
    config = HoverEncounterConfig(
        incoming=IncomingSphere(
            arrival_time_seconds=8.68,
            direction=(1, 2, 0),
            speed_m_s=2.7,
            crossing_offset=(0.1, -0.2, 0.25),
        ),
        guards=(GuardSphere((0.0, 1.1, 0.0), 0.3),),
    )
    world = build_hover_encounter_world(config)
    duplicate = build_hover_encounter_world(
        HoverEncounterConfig.from_dict(json.loads(json.dumps(asdict(config))))
    )
    times = np.array((0.0, 3.0, 8.0, 8.68, 9.0, 16.0))
    for actual, expected in zip(
        world.obstacle_kinematics(times), duplicate.obstacle_kinematics(times), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    centers, velocities = world.obstacle_kinematics(config.incoming.arrival_time_seconds)
    np.testing.assert_allclose(centers[0], np.array(config.hover_position) + (0.1, -0.2, 0.25))
    np.testing.assert_allclose(velocities[0], np.array((1, 2, 0)) * 2.7 / np.sqrt(5))
    np.testing.assert_array_equal(velocities[1], 0)
    epsilon = 1e-5
    np.testing.assert_allclose(
        velocities,
        (
            world.obstacle_kinematics(8.68 + epsilon)[0]
            - world.obstacle_kinematics(8.68 - epsilon)[0]
        )
        / (2 * epsilon),
        atol=1e-9,
    )
    initial = world.initial_state.copy()
    initial[3:7] = Rotation.from_euler("x", 0.1).as_quat()
    initial[7] = 0.02
    branch = build_hover_encounter_world(config, initial_state=initial, initial_time_seconds=8.0)
    initial[:] = 0  # A mutable caller buffer cannot later change the authenticated branch.
    np.testing.assert_array_equal(
        branch.obstacle_kinematics(times)[0], world.obstacle_kinematics(times)[0]
    )
    np.testing.assert_allclose(branch.initial_state[7], 0.02)
    assert branch.metadata()["initial_state_time_seconds"] == 8.0
    assert branch.metadata()["case_study_config"] == asdict(config)
    np.testing.assert_array_equal(branch.wind_at(2.999), (0, 0, 0))
    np.testing.assert_array_equal(branch.wind_at(3.0), config.wind_velocity)
    np.testing.assert_array_equal(branch.wind_at(15.99), config.wind_velocity)
    with pytest.raises(ValueError, match="read-only"):
        branch.initial_state[0] = 2


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"incoming": IncomingSphere(direction=(0, 0, 0))}, "nonzero"),
        ({"incoming": IncomingSphere(amplitude_m=1)}, "additional passage"),
        ({"ego_radius": 0.05}, "enclose"),
        ({"wind_onset_seconds": 3.01}, "control boundaries"),
        ({"navigation_start_seconds": 11.01}, "control boundary"),
        ({"guards": (GuardSphere((0, 0, 0)),)}, "outside every inflated"),
    ),
)
def test_case_rejects_invalid_cadence_geometry_and_undersized_envelope(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_hover_encounter_world(replace(HoverEncounterConfig(), **changes))


def _stationary_states(
    count: int, position: tuple[float, float, float] = (0.0, 0.0, 1.4)
) -> np.ndarray:
    states = np.zeros((count, 13))
    states[:, :3] = position
    states[:, 6] = 1
    return states


def test_xml_audit_does_not_call_conservative_envelope_breach_actual_contact() -> None:
    world = build_hover_encounter_world(
        HoverEncounterConfig(guards=(GuardSphere((0.31, 0, 0), radius_m=0.05),))
    )
    states = _stationary_states(2, position=(0.16, 0.0, 1.4))
    report = audit_recorded_collider_clearance(world, np.array((0.0, 0.02)), states)
    envelope = report["body_origin_envelope"]
    collider = report["actual_xml_sphere_geometry"]
    assert envelope["minimum_clearance_m"] == pytest.approx(-0.006)
    assert envelope["intersection_classification"].startswith("intersecting_")
    assert collider["minimum_clearance_lower_bound_m"] > 0.015
    assert collider["first_chord_intersection_time_seconds"] is None
    assert "no contact dynamics" in report["scope"]


def test_rotating_offset_audit_detects_contact_between_nonintersecting_recorded_nodes() -> None:
    # The collider swings toward the guard halfway through a 120-degree pitch rotation;
    # body position is constant and both recorded endpoint collider spheres miss it.
    world = build_hover_encounter_world(
        HoverEncounterConfig(guards=(GuardSphere((0.31, 0, 0), radius_m=0.05),))
    )
    states = _stationary_states(2, position=(0.16, 0.0, 1.4))
    states[:, 3:7] = Rotation.from_euler("y", ((30,), (150,)), degrees=True).as_quat()
    times = np.array((0.0, 0.02))
    endpoint_centers = states[:, :3] + Rotation.from_quat(states[:, 3:7]).apply((0, 0, 0.02))
    guard_center = world.obstacle_mean_centers[1]
    assert np.all(np.linalg.norm(endpoint_centers - guard_center, axis=-1) > 0.136)
    report = audit_recorded_collider_clearance(world, times, states)
    collider = report["actual_xml_sphere_geometry"]
    assert collider["minimum_clearance_m"] == pytest.approx(-0.006, abs=1e-6)
    assert collider["minimum_clearance_upper_bound_m"] < -0.0059
    assert 0 < collider["first_chord_intersection_time_seconds"] < 0.01
    # Halving substeps retains the result while tightening the declared arc error.
    refined = audit_recorded_collider_clearance(world, times, states, max_substep_seconds=0.0005)
    assert refined["actual_xml_sphere_geometry"]["maximum_chord_curvature_error_bound_m"] < (
        collider["maximum_chord_curvature_error_bound_m"] / 3.9
    )
    assert report["actual_xml_ground_geometry"]["minimum_clearance_m"] > 1


def test_fast_obstacle_sweep_and_ground_are_detected_between_recorded_states() -> None:
    config = HoverEncounterConfig(incoming=IncomingSphere(arrival_time_seconds=8.0, speed_m_s=3.0))
    world = build_hover_encounter_world(config)
    states = _stationary_states(2)
    report = audit_recorded_collider_clearance(world, np.array((7.0, 9.0)), states)
    collider = report["actual_xml_sphere_geometry"]
    assert collider["minimum_clearance_m"] == pytest.approx(-0.566, abs=1e-6)
    assert 7.7 < collider["first_chord_intersection_time_seconds"] < 7.9
    assert report["maximum_substep_seconds"] <= 0.001 + 1e-12
    # Floor uses the offset XML sphere rather than the more conservative .106 m body sphere.
    fall = _stationary_states(2)
    fall[:, 2] = (0.1, 0.0)
    ground = audit_recorded_collider_clearance(world, np.array((0.0, 0.02)), fall)[
        "actual_xml_ground_geometry"
    ]
    assert ground["minimum_clearance_m"] == pytest.approx(-0.066)
    assert ground["first_chord_intersection_time_seconds"] == pytest.approx(0.0068)
