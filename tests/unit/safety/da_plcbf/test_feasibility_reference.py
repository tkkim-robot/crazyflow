from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.continuous_demo_scenarios import blocking_static_scenario
from crazyflow.safety.da_plcbf.feasibility_reference import (
    FeasibilityReferenceConfig,
    _collision_clearance,
    run_feasibility_reference,
    save_feasibility_reference,
)
from crazyflow.safety.da_plcbf.online_constant_wind import build_cf21b_version_a_resources
from crazyflow.safety.da_plcbf.rigid_payload import CenteredRigidPayload

if TYPE_CHECKING:
    from pathlib import Path

    from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel


def test_reference_uses_actual_parameter_switch_and_preserves_full_hover_trace(
    tmp_path: Path,
) -> None:
    resources = build_cf21b_version_a_resources()
    scenario = blocking_static_scenario()
    scenario = replace(
        scenario,
        goal_position=scenario.initial_state[:3],
        obstacle_mask=jnp.zeros_like(scenario.obstacle_mask),
    )
    combined = CenteredRigidPayload(0.006).apply(resources.model)

    def model(index: int) -> VersionAModel:
        base = resources.model if index < 5 else combined
        return base._replace(wind_velocity=jnp.asarray([0.0, 0.0, 0.4]))

    result = run_feasibility_reference(
        scenario,
        resources,
        scenario.initial_state,
        start_step=3,
        waypoints=[],
        max_steps=8,
        model_at_step=model,
        ego_radius_at_step=lambda index: 0.05 if index < 5 else 0.08,
        config=FeasibilityReferenceConfig(goal_hold_steps=4),
    )
    assert result.summary["feasible_witness_found"] is True
    assert result.states.shape == (5, 13)
    assert result.actions.shape == (4, 4)
    np.testing.assert_allclose(
        result.states, np.broadcast_to(scenario.initial_state, (5, 13)), atol=2e-7
    )
    np.testing.assert_allclose(
        result.model_parameters["mass"],
        [resources.model.mass, resources.model.mass, combined.mass, combined.mass],
    )
    np.testing.assert_allclose(result.ego_radii, [0.05, 0.05, 0.08, 0.08, 0.08])
    np.testing.assert_allclose(result.time_seconds, np.arange(3, 8) * scenario.dt)
    npz_path, _ = save_feasibility_reference(result, tmp_path / "reference")
    with np.load(npz_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["states"], result.states)
        np.testing.assert_array_equal(archive["model_mass"], result.model_parameters["mass"])


def test_reference_does_not_call_an_unreachable_route_a_feasible_witness() -> None:
    resources = build_cf21b_version_a_resources()
    scenario = blocking_static_scenario()
    scenario = replace(
        scenario,
        goal_position=scenario.initial_state[:3] + jnp.asarray([0.0, 0.0, 0.5]),
        obstacle_mask=jnp.zeros_like(scenario.obstacle_mask),
    )
    heavy = CenteredRigidPayload(0.2).apply(resources.model)
    result = run_feasibility_reference(
        scenario,
        resources,
        scenario.initial_state,
        start_step=0,
        waypoints=[],
        max_steps=20,
        model_at_step=lambda _: heavy,
    )
    assert result.summary["checks"]["motor_limits"] is True
    assert result.summary["checks"]["goal_reached_and_settled"] is False
    assert result.summary["feasible_witness_found"] is False
    assert result.states[-1, 2] < result.states[0, 2]


def test_reference_swept_clearance_detects_crossing_obstacle_and_radius_switch() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    crossing = np.asarray([[[-1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    clearance, _ = _collision_clearance(
        positions, crossing, np.asarray([0.1]), np.asarray([0.05, 0.05])
    )
    np.testing.assert_allclose(clearance, -0.15, atol=1e-12)
    static = np.asarray([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
    clearance, _ = _collision_clearance(
        positions, static, np.asarray([0.4]), np.asarray([0.05, 0.8])
    )
    np.testing.assert_allclose(clearance, -0.2, atol=1e-12)
