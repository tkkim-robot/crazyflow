from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.continuous_version_a import (
    ContinuousVersionAConfig,
    PolicyRollouts,
    RuntimeObstacleTrajectories,
    obstacle_agnostic_emergency_wrench,
    rollout_waypoint_library,
    runtime_policy_values,
)
from crazyflow.safety.da_plcbf.policy_qp_audit import (
    make_navigation_policy_qp_auditor,
    make_policy_qp_auditor,
    policy_geometry_diagnostics,
    rollout_emergency_brake,
    summarize_policy_qp_audit,
)
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.version_a_barriers import (
    RigidBodySafetySet,
    VersionABarrierConfig,
    VersionAModel,
)
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator, VersionAFilterConfig


def resources() -> tuple[VersionAModel, VersionAActuator]:
    raw = load_params("cf21B_500")

    def array(value: Any) -> jax.Array:
        return jnp.asarray(value, jnp.float32)

    model = VersionAModel(
        array(raw["mass"]),
        array(raw["gravity_vec"]),
        array(raw["J"]),
        jnp.linalg.inv(array(raw["J"])),
        array(raw["drag_matrix"]),
        array([1.6, 0.8, 0]),
        array([0, 0, 0]),
        array([0, 0, 0]),
    )
    actuator = VersionAActuator(
        array(raw["L"]),
        array(raw["thrust2torque"]),
        array(raw["mixing_matrix"]),
        array(raw["thrust_min"]),
        array(raw["thrust_max"]),
    )
    return model, actuator


def test_offline_qps_keep_predictive_repair_and_held_checks_for_every_eligible_policy():
    """The recorded upper-arena held violation must be repaired in each forced full QP."""
    model, actuator = resources()
    model = model._replace(wind_velocity=jnp.zeros(3))
    state = jnp.asarray(
        [
            10.474519729614258,
            -0.001515018637292087,
            1.3911974430084229,
            0.0015002323780208826,
            -0.010995380580425262,
            4.380210884846747e-5,
            0.9999384880065918,
            1.7187553644180298,
            0.0025914725847542286,
            0.007324578240513802,
            -0.010574331507086754,
            -0.05424872413277626,
            -0.00037901222822256386,
        ]
    )
    safety = RigidBodySafetySet(
        jnp.asarray([[0.0, 3.0, 1.4]]),
        jnp.asarray([0.1]),
        jnp.ones(1, dtype=bool),
        jnp.asarray([-5.0, -3.5, 0.15]),
        jnp.asarray([12.0, 4.0, 4.0]),
        jnp.asarray(3.5),
        jnp.asarray(12.0),
        jnp.asarray(0.9),
    )
    obstacles = RuntimeObstacleTrajectories(
        jnp.broadcast_to(safety.obstacle_centers, (5, 1, 3)),
        safety.obstacle_radii,
        jnp.ones((5, 1), dtype=bool),
        jnp.zeros((1, 3)),
    )
    config = ContinuousVersionAConfig(horizon=4, control_interval_steps=2)

    def rollout(x: jax.Array, point: VersionAModel) -> PolicyRollouts:
        return rollout_waypoint_library(
            x,
            jnp.asarray([[10.0, 0.0, 1.4]]),
            jnp.zeros((1, 3)),
            point,
            actuator,
            QuadPolicyConfig(acceleration_limit=1.2),
            dt=0.02,
            horizon=4,
            position_gain=2.0,
            velocity_gain=2.8,
            model_compensation=True,
            command_hold_steps=2,
        )

    audit = make_policy_qp_auditor(
        rollout,
        rollout,
        actuator,
        safety,
        VersionABarrierConfig(arena_clearance=0.08),
        VersionAFilterConfig(),
        config,
    )
    result = audit(state, obstacles, model)
    assert set(result.counterfactuals) == {0, 1}
    assert int(result.runtime.selected_index) == 0
    for index, decision in result.counterfactuals.items():
        assert int(decision.selected_index) == index
        assert int(decision.predictive_operational_iterations) > 0
        assert float(decision.initial_qp_held_operational_residual) < -0.04
        assert bool(decision.qp_valid)
        assert bool(decision.applied_held_operational_passed)
        assert float(jnp.min(decision.qp_held_operational_residuals)) >= -3e-6
        np.testing.assert_allclose(decision.action, result.runtime.action, atol=2e-6)
        np.testing.assert_array_equal(decision.gradients, result.runtime.gradients)
        np.testing.assert_array_equal(decision.smooth_values, result.runtime.smooth_values)
    summary = summarize_policy_qp_audit(result, obstacles, config)
    assert summary["eligible_accepted_held_qp_count"] == 2
    json.dumps(summary, allow_nan=False)
    # With the declared refinement disabled, the motor-box scores and instantaneous QPs remain
    # favorable but no candidate passes the held operational check. Report the distinction.
    unrefined_config = replace(config, predictive_operational_iterations=0)
    unrefined_audit = make_policy_qp_auditor(
        rollout,
        rollout,
        actuator,
        safety,
        VersionABarrierConfig(arena_clearance=0.08),
        VersionAFilterConfig(),
        unrefined_config,
    )
    unrefined = unrefined_audit(state, obstacles, model)
    unrefined_summary = summarize_policy_qp_audit(unrefined, obstacles, unrefined_config)
    assert unrefined_summary["eligible_full_qp_count"] == 2
    assert unrefined_summary["eligible_accepted_held_qp_count"] == 0
    assert all(
        "held_operational_failed" in item["qp_rejections"]
        for item in unrefined_summary["counterfactuals"].values()
    )
    # A shell overlap does not become an eligible diagnostic candidate merely because it has
    # a large motor-box score. The empty result explicitly differs from QP infeasibility.
    overlap = obstacles._replace(centers=jnp.broadcast_to(state[:3], (5, 1, 3)))
    no_certificate = audit(state, overlap, model)
    assert no_certificate.counterfactuals == {}
    assert not bool(no_certificate.runtime.qp_valid)


def test_swept_active_time_and_clearance_have_independent_units_and_minima():
    config = ContinuousVersionAConfig(horizon=1, dt=1.0, obstacle_clearance=0)
    states = np.zeros((1, 2, 13))
    states[0, 0, :3] = (-1, 0, 0)
    states[0, 1, :3] = (1, 0, 0)
    # Small sphere determines minimum squared value; large sphere determines metre clearance.
    obstacles = RuntimeObstacleTrajectories(
        jnp.asarray([[[0, 0.4, 0], [0, 2.05, 0]]] * 2),
        jnp.asarray([0.1, 2.0]),
        jnp.ones((2, 2), dtype=bool),
    )
    details = policy_geometry_diagnostics(states, obstacles, config)[0]
    expected = runtime_policy_values(jnp.asarray(states), obstacles, obstacle_clearance=0)
    assert details["active_obstacle"] == 0
    assert details["active_kind"] == "swept_interval"
    assert details["active_time_seconds"] == pytest.approx(0.5)
    assert details["active_index"] == int(expected.active_indices[0])
    assert details["active_hard_value_m2"] == pytest.approx(float(expected.values[0]))
    assert details["active_clearance_m"] == pytest.approx(0.3)
    assert details["minimum_clearance_m"] == pytest.approx(0.05, abs=1e-6)


def test_emergency_diagnostic_uses_real_feedback_with_the_declared_command_hold():
    model, actuator = resources()
    config = ContinuousVersionAConfig(horizon=6, control_interval_steps=2)
    state = jnp.asarray([0, 0, 1.4, 0, 0, 0, 1, 0.7, -0.2, 0.1, 0, 0, 0], jnp.float32)
    result = jax.jit(lambda x: rollout_emergency_brake(x, model, actuator, config))(state)
    expected_states, expected_wrenches = [np.asarray(state)], []
    current = state
    for step in range(config.horizon):
        if step % 2 == 0:
            command, valid = obstacle_agnostic_emergency_wrench(current, model, actuator, config)
            assert bool(valid)
        expected_wrenches.append(np.asarray(command))
        current = direct_wrench_symplectic_step(current, command, model, config.dt)
        expected_states.append(np.asarray(current))
    np.testing.assert_allclose(result.states[0], expected_states, atol=2e-7)
    np.testing.assert_allclose(result.wrenches[0], expected_wrenches, atol=2e-7)
    assert bool(result.valid[0])


def test_vacuous_geometry_does_not_invent_an_active_constraint():
    config = ContinuousVersionAConfig(horizon=2)
    states = np.zeros((2, 3, 13))
    obstacles = RuntimeObstacleTrajectories(
        np.full((3, 1, 3), np.nan), np.asarray([np.nan]), np.zeros((3, 1), dtype=bool)
    )
    assert policy_geometry_diagnostics(states, obstacles, config) == [
        {"collision_constraints_active": False, "active_index": -1},
        {"collision_constraints_active": False, "active_index": -1},
    ]


def test_navigation_auditor_dynamic_snapshots_and_single_original_skill_ablation():
    """The diagnostic shares deployment settings and differentiates the actual mixed evaluator."""
    from crazyflow.safety.da_plcbf.navigation_experiment import (
        NavigationExperimentConfig,
        build_navigation_controller,
    )
    from crazyflow.safety.da_plcbf.navigation_world import (
        NavigationWorldConfig,
        build_navigation_world,
    )
    from crazyflow.safety.da_plcbf.persistent_skill_learner import (
        PersistentSkillConfig,
        build_fibonacci_skill_spec,
        initialize_skill_actor,
    )

    model, actuator = resources()
    world = build_navigation_world(NavigationWorldConfig(obstacle_count=1, duration_seconds=0.12))
    config = PersistentSkillConfig(horizon=4, hidden_width=8, control_interval_steps=2)
    spec = build_fibonacci_skill_spec(
        policy_count=2,
        latent_size=3,
        minimum_duration=0.04,
        maximum_duration=0.06,
        horizon_duration=0.08,
    )
    original = initialize_skill_actor(jax.random.key(15), spec, config)
    learned = original.replace(output_bias=original.output_bias + jnp.asarray([0.8, -0.4, 0.3]))
    bundle = SimpleNamespace(config=config, spec=spec, actuator=actuator)
    run_config = NavigationExperimentConfig(fallback_mapping="matched_uncompensated")
    state = jnp.asarray(world.initial_state, jnp.float32)
    goal = state[:3] + jnp.asarray([0.2, 0.0, 0.05])
    obstacles = RuntimeObstacleTrajectories(
        jnp.broadcast_to(state[:3] + jnp.asarray([1.0, 1.0, 1.0]), (5, 1, 3)),
        jnp.asarray([0.1]),
        jnp.ones((5, 1), dtype=bool),
        jnp.zeros((1, 3)),
    )
    audit = make_navigation_policy_qp_auditor(world, bundle, run_config)
    controller = build_navigation_controller(world, bundle, run_config)
    results = {}
    for name, params in (("original", original), ("learned", learned)):
        direct = controller(state, params, model, obstacles, jnp.asarray(-1), goal)
        result = audit(state, params, model, obstacles, -1, goal)
        for expected, actual in zip(
            jax.tree.leaves(direct), jax.tree.leaves(result.runtime), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
        assert set(result.counterfactuals) == set(
            np.flatnonzero(direct.continuous_filter.policy_eligible)
        )
        assert result.counterfactuals
        results[name] = direct
    assert (
        np.max(np.abs(results["learned"].candidates.states - results["original"].candidates.states))
        > 1e-5
    )

    mixed_controller = build_navigation_controller(
        world, bundle, run_config, frozen_replacement=(original, 1)
    )
    mixed = mixed_controller(state, learned, model, obstacles, jnp.asarray(-1), goal)
    # The original fallback skill at index one occupies augmented index two. Everything else,
    # including shared nominal, still comes from the learned controller's evaluator.
    for policy in range(3):
        source = results["original"] if policy == 2 else results["learned"]
        np.testing.assert_array_equal(
            mixed.candidates.states[policy], source.candidates.states[policy]
        )
        np.testing.assert_array_equal(
            mixed.candidates.wrenches[policy], source.candidates.wrenches[policy]
        )
        np.testing.assert_array_equal(
            mixed.candidates.valid[policy], source.candidates.valid[policy]
        )
        np.testing.assert_allclose(mixed.gradients[policy], source.gradients[policy], atol=3e-6)
        np.testing.assert_array_equal(mixed.smooth_values[policy], source.smooth_values[policy])
        np.testing.assert_array_equal(
            mixed.time_derivatives[policy], source.time_derivatives[policy]
        )
    assert np.max(np.abs(mixed.gradients[2] - results["learned"].gradients[2])) > 1e-5
    for invalid in (-1, 2, True):
        with pytest.raises(ValueError, match="replacement skill index"):
            build_navigation_controller(
                world, bundle, run_config, frozen_replacement=(original, invalid)
            )
