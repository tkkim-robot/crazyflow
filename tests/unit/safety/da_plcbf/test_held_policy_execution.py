"""Executable feedback cadence and predictive held operational regressions."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.continuous_version_a import (
    ContinuousVersionAConfig,
    ContinuousVersionAStep,
    PolicyRollouts,
    RuntimeObstacleTrajectories,
    continuous_version_a_step,
    rollout_waypoint_library,
)
from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentSkillConfig,
    build_fibonacci_skill_spec,
    initialize_skill_actor,
    obstacle_agnostic_skill_actions,
    rollout_skill_library,
)
from crazyflow.safety.da_plcbf.quad_policy import (
    QuadPolicyConfig,
    acceleration_to_feasible_wrench,
    waypoint_nominal_wrench,
)
from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
from crazyflow.safety.da_plcbf.version_a_barriers import (
    RigidBodySafetySet,
    VersionABarrierConfig,
    VersionAModel,
    safety_constraint_names,
)
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator, VersionAFilterConfig


def _resources() -> tuple[VersionAModel, VersionAActuator]:
    raw = load_params("cf21B_500")
    model = VersionAModel(
        jnp.asarray(raw["mass"]),
        jnp.asarray(raw["gravity_vec"]),
        jnp.asarray(raw["J"]),
        jnp.linalg.inv(jnp.asarray(raw["J"])),
        jnp.asarray(raw["drag_matrix"]),
        jnp.asarray([0.8, -0.4, 0.2]),
        jnp.zeros(3),
        jnp.zeros(3),
    )
    actuator = VersionAActuator(
        *[
            jnp.asarray(raw[k])
            for k in ("L", "thrust2torque", "mixing_matrix", "thrust_min", "thrust_max")
        ]
    )
    return model, actuator


@pytest.mark.parametrize("hold_steps", [1, 2])
def test_full_committed_skill_matches_independent_command_hold_replay(hold_steps: int) -> None:
    """The complete 1.2 s fallback and its first hold must match advancing-phase replay."""
    model, actuator = _resources()
    config = PersistentSkillConfig(
        horizon=60, hidden_width=8, control_interval_steps=hold_steps, model_compensation=True
    )
    spec = build_fibonacci_skill_spec(
        policy_count=4, latent_size=3, minimum_duration=0.4, maximum_duration=0.7
    )
    params = initialize_skill_actor(jax.random.key(19), spec, config)
    state = jnp.asarray([2.0, -1.0, 1.4, 0.0, 0.0, 0.0, 1.0, 1.5, -0.3, 0.2, 0.04, -0.08, 0.01])
    rollout = jax.jit(lambda x: rollout_skill_library(params, spec, x, model, actuator, config))(
        state
    )
    current = jnp.broadcast_to(state, (4, 13))
    nodes = [current]
    commands = []
    command_at = jax.jit(
        lambda x, phase: (
            acceleration_to_feasible_wrench(
                obstacle_agnostic_skill_actions(
                    params, spec, x, state[:3], phase, config, point_model=model
                ),
                x[:, 3:7],
                x[:, 10:13],
                model,
                actuator,
                QuadPolicyConfig(acceleration_limit=config.acceleration_limit),
                smooth_motor_bounds=config.smooth_motor_bounds,
            ).wrench
        )
    )
    plant = jax.jit(lambda x, u: direct_wrench_symplectic_step(x, u, model, config.dt))
    for index in range(config.horizon):
        if index % hold_steps == 0:
            wrench = command_at(current, jnp.asarray(index / config.horizon))
        current = plant(current, wrench)
        nodes.append(current)
        commands.append(wrench)
    np.testing.assert_allclose(rollout.states, jnp.stack(nodes, axis=1), atol=3e-6, rtol=2e-5)
    np.testing.assert_allclose(rollout.wrenches, jnp.stack(commands, axis=1), atol=3e-7, rtol=2e-5)
    assert rollout.states.shape[1] == 61
    if hold_steps == 2:
        np.testing.assert_array_equal(rollout.wrenches[:, ::2], rollout.wrenches[:, 1::2])


def test_waypoint_partial_hold_retains_integration_horizon_and_feedback_cadence():
    model, actuator = _resources()
    initial = jnp.asarray([0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 1.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0])
    target = jnp.asarray([1.0, 0.6, 2.0])
    qconfig = QuadPolicyConfig(acceleration_limit=1.2)
    rollout = rollout_waypoint_library(
        initial,
        target[None],
        jnp.zeros((1, 3)),
        model,
        actuator,
        qconfig,
        dt=0.02,
        horizon=5,
        command_hold_steps=2,
    )
    state = initial
    nodes = [state]
    commands = []
    for index in range(5):
        if index % 2 == 0:
            wrench = waypoint_nominal_wrench(
                state, target, jnp.zeros(3), model, actuator, qconfig
            ).wrench
        state = direct_wrench_symplectic_step(state, wrench, model, 0.02)
        nodes.append(state)
        commands.append(wrench)
    assert rollout.states.shape == (1, 6, 13)
    np.testing.assert_allclose(rollout.states[0], jnp.stack(nodes), atol=2e-6)
    np.testing.assert_allclose(rollout.wrenches[0], jnp.stack(commands), atol=2e-7)


def test_predictive_operational_repair_preserves_original_held_postcheck():
    """Crossing 6.2 s failure is the upper-x HOCBF at +20 ms, not obstacle coverage."""
    model, actuator = _resources()
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
        jnp.empty((0, 3)),
        jnp.empty(0),
        jnp.empty(0, dtype=bool),
        jnp.asarray([-5.0, -3.5, 0.15]),
        jnp.asarray([12.0, 4.0, 4.0]),
        jnp.asarray(3.5),
        jnp.asarray(12.0),
        jnp.asarray(0.9),
    )
    prediction = RuntimeObstacleTrajectories(
        jnp.empty((5, 0, 3)), jnp.empty(0), jnp.empty((5, 0), dtype=bool)
    )
    config = ContinuousVersionAConfig(
        horizon=4, control_interval_steps=2, predictive_operational_iterations=0
    )

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

    def evaluate(c: ContinuousVersionAConfig) -> ContinuousVersionAStep:
        return jax.jit(
            lambda x: continuous_version_a_step(
                x,
                rollout,
                rollout,
                prediction,
                model,
                actuator,
                safety,
                VersionABarrierConfig(arena_clearance=0.08),
                VersionAFilterConfig(),
                c,
            )
        )(state)

    original = evaluate(config)
    revised = evaluate(replace(config, predictive_operational_iterations=3))
    index = np.unravel_index(np.argmin(original.qp_held_operational_residuals), (2, 9))
    assert index == (1, 3)
    assert safety_constraint_names(0)[index[1]] == "arena_x_upper"
    np.testing.assert_allclose(original.qp_held_operational_residuals[index], -0.047247, atol=2e-6)
    assert not bool(original.qp_valid)
    assert bool(revised.qp_valid)
    assert int(revised.predictive_operational_iterations) > 0
    assert (
        float(jnp.min(revised.qp_held_operational_residuals))
        >= -VersionAFilterConfig().barrier_tolerance
    )
    assert bool(revised.applied_held_operational_passed)
    assert bool(revised.applied_postcheck.passed)
    assert float(revised.applied_held_operational_margin) > 0


def test_infeasible_predictive_refinement_preserves_explicit_fallback_proposal() -> None:
    """An infeasible refinement must not relabel its old rejected QP action as fallback."""
    from crazyflow.safety.da_plcbf.version_a_filter import (
        PolicyLibraryCertificates,
        reproject_with_predictive_operational_faces,
        version_a_plcbf_filter,
    )

    model, actuator = _resources()
    state = jnp.zeros(13).at[2].set(1.4).at[6].set(1.0)
    safety = RigidBodySafetySet(
        jnp.empty((0, 3)),
        jnp.empty(0),
        jnp.empty(0, dtype=bool),
        jnp.asarray([-5.0, -3.5, 0.15]),
        jnp.asarray([12.0, 4.0, 4.0]),
        jnp.asarray(3.5),
        jnp.asarray(12.0),
        jnp.asarray(0.9),
    )
    nominal = waypoint_nominal_wrench(
        state,
        jnp.asarray([1.0, 0.0, 1.4]),
        jnp.zeros(3),
        model,
        actuator,
        QuadPolicyConfig(acceleration_limit=1.2),
    ).wrench
    fallback = waypoint_nominal_wrench(
        state, state[:3], jnp.zeros(3), model, actuator, QuadPolicyConfig()
    ).wrench
    library = PolicyLibraryCertificates(
        jnp.ones(1), jnp.zeros((1, 13)), jnp.ones(1, dtype=bool), fallback[None]
    )
    config = VersionAFilterConfig(selection_requires_certified_fallback=False)
    weight = jnp.asarray([1.0, 2e4, 2e4, 2e4])
    original = version_a_plcbf_filter(
        state, nominal, weight, library, model, actuator, safety, VersionABarrierConfig(), config
    )
    assert bool(original.qp_accepted)
    revised = reproject_with_predictive_operational_faces(
        original,
        nominal,
        weight,
        actuator,
        config,
        jnp.zeros((1, 4)),
        jnp.asarray([-1.0]),
        omitted_obstacle_rows=0,
        selected_fallback_wrench=fallback,
    )
    assert not bool(revised.qp_accepted)
    assert bool(revised.used_fallback)
    assert not bool(revised.used_midpoint)
    np.testing.assert_array_equal(revised.action, fallback)


def test_longer_holds_require_explicitly_disabling_bounded_predictive_refinement() -> None:
    """Prevent an unbounded expansion of enumerated QP faces beyond the validated N=2 case."""
    with pytest.raises(ValueError, match="predictive_operational_iterations=0"):
        ContinuousVersionAConfig(horizon=6, control_interval_steps=3).validate()
    ContinuousVersionAConfig(
        horizon=6, control_interval_steps=3, predictive_operational_iterations=0
    ).validate()
    ContinuousVersionAConfig(horizon=6, control_interval_steps=2).validate()
