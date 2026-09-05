"""Independent scientific-isolation checks for the corrected continuous controller."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.continuous_version_a import (
    PolicyRollouts,
    RuntimeObstacleTrajectories,
    conservative_smooth_policy_values,
    runtime_policy_values,
)
from crazyflow.safety.da_plcbf.version_a_barriers import (
    RigidBodySafetySet,
    VersionABarrierConfig,
    VersionAModel,
    continuous_safety_halfspaces,
)
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


def test_smooth_min_is_conservative_and_differentiable_at_duplicate_and_competing_minima() -> None:
    """Tied obstacle branches and duplicate node/segment endpoints must remain usable."""
    obstacles = RuntimeObstacleTrajectories(
        centers=jnp.asarray([[[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]] * 3),
        radii=jnp.asarray([0.2, 0.2]),
        mask=jnp.ones((3, 2), dtype=bool),
    )
    temperature = 0.02

    def evaluate(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        states = jnp.zeros((1, 3, 13)).at[:, :, 6].set(1).at[:, :, 2].set(1)
        states = states.at[:, :, 0].set(x)
        hard = runtime_policy_values(states, obstacles, obstacle_clearance=0.1, ego_radius=0.05)
        smooth = conservative_smooth_policy_values(hard, temperature=temperature)
        return smooth[0], hard.values[0]

    value, gradient = jax.value_and_grad(lambda x: evaluate(x)[0])(jnp.asarray(0.0))
    hard = evaluate(jnp.asarray(0.0))[1]
    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, 0.0, atol=1e-6)
    assert value <= hard
    assert hard - value <= temperature * np.log(10) + 1e-6
    for x in (-0.04, 0.04):
        point = jnp.asarray(x)
        derivative = jax.grad(lambda t: evaluate(t)[0])(point)
        finite_difference = (evaluate(point + 1e-3)[0] - evaluate(point - 1e-3)[0]) / 2e-3
        np.testing.assert_allclose(derivative, finite_difference, atol=2e-3)


def test_ego_radius_and_masked_padding_preserve_value_and_gradient() -> None:
    states = jnp.zeros((1, 2, 13)).at[:, :, 6].set(1).at[:, :, 0].set(1)
    obstacles = RuntimeObstacleTrajectories(
        centers=jnp.asarray([[[0.0, 0.0, 0.0], [jnp.nan, jnp.nan, jnp.nan]]] * 2),
        radii=jnp.asarray([0.2, jnp.nan]),
        mask=jnp.asarray([[True, False]] * 2),
    )

    def smooth(rollout: jax.Array) -> jax.Array:
        hard = runtime_policy_values(rollout, obstacles, obstacle_clearance=0.1, ego_radius=0.05)
        return conservative_smooth_policy_values(hard, temperature=0.005)[0]

    hard = runtime_policy_values(states, obstacles, obstacle_clearance=0.1, ego_radius=0.05)
    np.testing.assert_allclose(hard.values, [1.0 - 0.35**2], atol=1e-6)
    assert bool(hard.input_valid[0])
    assert np.all(np.isfinite(jax.grad(smooth)(states)))


def test_disabling_obstacle_hocbf_preserves_all_operational_faces() -> None:
    model = VersionAModel(
        jnp.asarray(1.0),
        jnp.asarray([0.0, 0.0, -9.81]),
        jnp.eye(3),
        jnp.eye(3),
        jnp.zeros((3, 3)),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros(3),
    )
    state = jnp.zeros(13).at[2].set(1).at[6].set(1)
    safety = RigidBodySafetySet(
        jnp.asarray([[0.0, 0.0, 1.0]]),
        jnp.asarray([0.2]),
        jnp.asarray([True]),
        jnp.asarray([-5.0, -5.0, 0.0]),
        jnp.asarray([5.0, 5.0, 5.0]),
        jnp.asarray(5.0),
        jnp.asarray(10.0),
        jnp.asarray(0.7),
    )
    full = continuous_safety_halfspaces(state, model, safety, VersionABarrierConfig())
    operational = continuous_safety_halfspaces(
        state, model, safety, VersionABarrierConfig(include_obstacle_hocbf=False)
    )
    assert not bool(full.domain_valid)
    assert bool(operational.domain_valid)
    np.testing.assert_array_equal(operational.enabled, [False] + [True] * 9)
    np.testing.assert_allclose(operational.matrix[1:], full.matrix[1:])
    np.testing.assert_allclose(operational.upper_bound[1:], full.upper_bound[1:])
    np.testing.assert_array_equal(operational.matrix[0], np.zeros(4))


def test_submillimetre_swept_segment_does_not_skip_interior_collision() -> None:
    states = jnp.zeros((1, 2, 13)).at[:, :, 6].set(1)
    states = states.at[0, :, 0].set(jnp.asarray([-1e-4, 1e-4]))
    obstacles = RuntimeObstacleTrajectories(
        centers=jnp.zeros((2, 1, 3)), radii=jnp.asarray([5e-5]), mask=jnp.ones((2, 1), dtype=bool)
    )
    values = runtime_policy_values(states, obstacles, obstacle_clearance=0.0)
    assert bool(values.input_valid[0])
    assert float(values.values[0]) < 0.0
    np.testing.assert_allclose(values.values[0], -((5e-5) ** 2), rtol=1e-5, atol=1e-12)


def test_softmin_duplicate_penalty_and_resolution_budget_are_explicit() -> None:
    from crazyflow.safety.da_plcbf.continuous_version_a import (
        RuntimePolicyValues,
        smooth_min_conservatism,
    )

    def data(constraints: jax.Array) -> RuntimePolicyValues:
        return RuntimePolicyValues(
            jnp.min(constraints, axis=1),
            constraints,
            jnp.asarray([0]),
            jnp.asarray([0.0]),
            jnp.asarray([True]),
        )

    base = data(jnp.asarray([[0.2, 0.5, 0.9]]))
    repeated = data(jnp.tile(base.constraint_values, (1, 7)))
    temperature = 0.01
    ordinary = conservative_smooth_policy_values(base, temperature=temperature)
    duplicate = conservative_smooth_policy_values(repeated, temperature=temperature)
    np.testing.assert_allclose(duplicate, ordinary - temperature * np.log(7), atol=1e-6)
    for count in (3, 21, 243, 1201):
        values = data(jnp.full((1, count), 0.2))
        smooth = conservative_smooth_policy_values(
            values, temperature=temperature, max_gap_budget=0.02
        )
        _, bound = smooth_min_conservatism(values, temperature=temperature, max_gap_budget=0.02)
        assert float(smooth[0]) <= float(values.values[0])
        assert float(values.values[0] - smooth[0]) <= 0.020001
        np.testing.assert_allclose(values.values - smooth, bound, atol=1e-6)


def test_time_partial_matches_absolute_time_difference_for_approaching_and_crossing_obstacles() -> (
    None
):
    from crazyflow.safety.da_plcbf.continuous_version_a import shift_obstacle_prediction

    dt, horizon = 0.1, 5
    states = jnp.zeros((1, horizon + 1, 13)).at[:, :, 6].set(1).at[:, :, 2].set(1)
    for start, velocity in (
        (jnp.asarray([2.0, 0.0, 1.0]), jnp.asarray([-0.7, 0.0, 0.0])),
        (jnp.asarray([0.45, 0.45, 1.0]), jnp.asarray([-1.0, 0.0, 0.0])),
    ):

        def prediction(
            absolute_time: jax.Array, explicit: bool = True
        ) -> RuntimeObstacleTrajectories:
            times = absolute_time + dt * jnp.arange(horizon + 1)
            return RuntimeObstacleTrajectories(
                (start + times[:, None] * velocity)[:, None, :],
                jnp.asarray([0.2]),
                jnp.ones((horizon + 1, 1), dtype=bool),
                velocity[None, :] if explicit else None,
            )

        def value(obstacles: RuntimeObstacleTrajectories) -> jax.Array:
            return conservative_smooth_policy_values(
                runtime_policy_values(states, obstacles, obstacle_clearance=0.1), temperature=0.005
            )[0]

        time = jnp.asarray(0.1)
        temporal = jax.grad(
            lambda shift: value(shift_obstacle_prediction(prediction(time), shift, dt=dt))
        )(jnp.asarray(0.0))
        inferred = jax.grad(
            lambda shift: value(shift_obstacle_prediction(prediction(time, False), shift, dt=dt))
        )(jnp.asarray(0.0))
        difference = (value(prediction(time + 1e-3)) - value(prediction(time - 1e-3))) / 2e-3
        np.testing.assert_allclose(temporal, difference, atol=4e-4, rtol=3e-3)
        np.testing.assert_allclose(inferred, temporal, atol=1e-5)
        if float(start[0]) > 1:
            assert float(temporal) < -1.0


def _small_controller_resources() -> tuple[
    VersionAModel, VersionAActuator, jax.Array, RigidBodySafetySet
]:
    from crazyflow.drones import load_params

    raw = load_params("cf21B_500")
    inertia = jnp.asarray(raw["J"], dtype=jnp.float32)
    model = VersionAModel(
        jnp.asarray(raw["mass"], dtype=jnp.float32),
        jnp.asarray(raw["gravity_vec"]),
        inertia,
        jnp.linalg.inv(inertia),
        jnp.asarray(raw["drag_matrix"]),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros(3),
    )
    actuator = VersionAActuator(
        *[
            jnp.asarray(raw[name])
            for name in ("L", "thrust2torque", "mixing_matrix", "thrust_min", "thrust_max")
        ]
    )
    state = jnp.zeros(13).at[2].set(1.4).at[6].set(1)
    safety = RigidBodySafetySet(
        jnp.asarray([[4.0, 0.0, 1.4]]),
        jnp.asarray([0.2]),
        jnp.asarray([True]),
        jnp.asarray([-5.0, -5.0, 0.1]),
        jnp.asarray([5.0, 5.0, 5.0]),
        jnp.asarray(5.0),
        jnp.asarray(12.0),
        jnp.asarray(0.8),
    )
    return model, actuator, state, safety


def test_empty_windows_omit_collision_row_and_survive_appearance_transitions() -> None:
    from crazyflow.safety.da_plcbf.continuous_version_a import (
        ContinuousVersionAConfig,
        continuous_version_a_step,
        rollout_waypoint_library,
    )
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAFilterConfig

    model, actuator, state, safety = _small_controller_resources()
    config = ContinuousVersionAConfig(horizon=4, control_interval_steps=2)

    def rollout(x: jax.Array, model: VersionAModel) -> PolicyRollouts:
        return rollout_waypoint_library(
            x,
            state[None, :3],
            jnp.zeros((1, 3)),
            model,
            actuator,
            QuadPolicyConfig(),
            dt=config.dt,
            horizon=config.horizon,
        )

    for slots in (1, 0):
        centers = jnp.broadcast_to(safety.obstacle_centers[:slots], (5, slots, 3))
        radii = safety.obstacle_radii[:slots]
        local_safety = safety._replace(
            obstacle_centers=safety.obstacle_centers[:slots],
            obstacle_radii=radii,
            obstacle_mask=safety.obstacle_mask[:slots],
        )
        step = jax.jit(
            lambda mask: continuous_version_a_step(
                state,
                rollout,
                rollout,
                RuntimeObstacleTrajectories(centers, radii, mask),
                model,
                actuator,
                local_safety,
                VersionABarrierConfig(),
                VersionAFilterConfig(),
                config,
            )
        )
        for active in (False, True, False):
            result = step(jnp.full((5, slots), active, dtype=bool))
            assert bool(result.qp_valid)
            assert not bool(result.used_emergency | result.used_midpoint | result.degraded)
            assert bool(result.collision_constraint_active) == (active and slots > 0)
            assert bool(result.continuous_filter.policy_constraint_active) == (active and slots > 0)
            if not active or slots == 0:
                assert np.all(np.isposinf(result.values.values))
                assert float(result.selected_policy_dual) == 0.0
                assert float(result.continuous_filter.qp.multipliers[-1]) == 0.0
            assert bool(result.applied_held_operational_passed)


def test_invalid_future_obstacle_velocity_cannot_supply_a_temporal_certificate() -> None:
    from crazyflow.safety.da_plcbf.continuous_version_a import (
        ContinuousVersionAConfig,
        ContinuousVersionAStep,
        continuous_version_a_step,
        rollout_waypoint_library,
    )
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAFilterConfig

    model, actuator, state, safety = _small_controller_resources()
    config = ContinuousVersionAConfig(horizon=4)

    def rollout(x: jax.Array, point_model: VersionAModel) -> PolicyRollouts:
        return rollout_waypoint_library(
            x,
            state[None, :3],
            jnp.zeros((1, 3)),
            point_model,
            actuator,
            QuadPolicyConfig(),
            dt=config.dt,
            horizon=config.horizon,
        )

    @jax.jit
    def step(velocities: jax.Array, mask: jax.Array) -> ContinuousVersionAStep:
        return continuous_version_a_step(
            state,
            rollout,
            rollout,
            RuntimeObstacleTrajectories(
                jnp.broadcast_to(safety.obstacle_centers, (5, 1, 3)),
                safety.obstacle_radii,
                mask,
                velocities,
            ),
            model,
            actuator,
            safety,
            VersionABarrierConfig(),
            VersionAFilterConfig(),
            config,
        )

    active = jnp.ones((5, 1), dtype=bool)
    finite_motion = jnp.zeros((5, 1, 3))
    assert bool(step(finite_motion, active).qp_valid)
    invalid_future_motion = finite_motion.at[-1, 0, 0].set(jnp.nan)
    invalid = step(invalid_future_motion, active)
    assert not bool(invalid.qp_valid)
    assert not np.any(invalid.gradient_valid)
    assert not bool(invalid.continuous_filter.has_certificate)
    masked = step(invalid_future_motion, active.at[-1, 0].set(False))
    assert bool(masked.qp_valid)
    assert np.all(masked.gradient_valid)


def test_no_executable_policy_uses_same_wind_aware_brake_not_midpoint() -> None:
    from crazyflow.safety.da_plcbf.continuous_version_a import (
        ContinuousVersionAConfig,
        continuous_version_a_step,
        rollout_waypoint_library,
    )
    from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAFilterConfig

    model, actuator, state, safety = _small_controller_resources()
    model = model._replace(wind_velocity=jnp.asarray([0.6, 0.3, 0.0]))
    config = ContinuousVersionAConfig(horizon=4)

    def invalid_rollout(x: jax.Array, point_model: VersionAModel) -> PolicyRollouts:
        result = rollout_waypoint_library(
            x,
            state[None, :3],
            jnp.zeros((1, 3)),
            point_model,
            actuator,
            QuadPolicyConfig(),
            dt=config.dt,
            horizon=config.horizon,
        )
        return result._replace(valid=jnp.asarray([False]))

    obstacles = RuntimeObstacleTrajectories(
        jnp.broadcast_to(safety.obstacle_centers, (5, 1, 3)),
        safety.obstacle_radii,
        jnp.ones((5, 1), dtype=bool),
    )
    decision = jax.jit(
        lambda x: continuous_version_a_step(
            x,
            invalid_rollout,
            invalid_rollout,
            obstacles,
            model,
            actuator,
            safety,
            VersionABarrierConfig(),
            VersionAFilterConfig(),
            config,
        )
    )(state)
    assert bool(decision.used_emergency)
    assert not bool(decision.used_midpoint)
    assert int(decision.execution_mode) == 2
    assert bool(decision.degraded)
    assert bool(decision.applied_postcheck.actuator_passed)
    assert abs(float(decision.action[0]) - float(model.mass * 9.81)) < 1e-4
    assert float(jnp.linalg.norm(decision.action[1:])) > 1e-6
    assert float(decision.executed_policy_dual) == 0.0


def test_control_hold_checks_operational_limits_at_each_integration_node() -> None:
    from crazyflow.safety.da_plcbf.continuous_version_a import (
        ContinuousVersionAConfig,
        _held_action_check,
    )
    from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step
    from crazyflow.safety.da_plcbf.version_a_filter import VersionAFilterConfig

    model, _, state, safety = _small_controller_resources()
    safety = safety._replace(speed_max=jnp.asarray(0.06))
    obstacles = RuntimeObstacleTrajectories(
        jnp.broadcast_to(safety.obstacle_centers, (5, 1, 3)),
        safety.obstacle_radii,
        jnp.zeros((5, 1), dtype=bool),
    )
    wrench = jnp.asarray([model.mass * (9.81 + 2.0), 0.0, 0.0, 0.0])
    single = _held_action_check(
        state,
        wrench,
        model,
        obstacles,
        safety,
        VersionABarrierConfig(),
        VersionAFilterConfig(),
        ContinuousVersionAConfig(horizon=4),
    )
    double = _held_action_check(
        state,
        wrench,
        model,
        obstacles,
        safety,
        VersionABarrierConfig(),
        VersionAFilterConfig(),
        ContinuousVersionAConfig(horizon=4, control_interval_steps=2),
    )
    expected = direct_wrench_symplectic_step(
        direct_wrench_symplectic_step(state, wrench, model, 0.02), wrench, model, 0.02
    )
    np.testing.assert_allclose(double.next_state, expected, atol=1e-7)
    assert bool(single.operational_passed)
    assert not bool(double.operational_passed)
    assert float(double.operational_margin) < 0
    assert np.isposinf(double.collision_margin)


def test_hover_quaternion_derivative_has_analytic_zero_rate_limit() -> None:
    from crazyflow.safety.da_plcbf.quad_rollouts import direct_wrench_symplectic_step

    model, _, state, _ = _small_controller_resources()
    wrench = jnp.asarray([model.mass * 9.81, 0.0, 0.0, 0.0])

    def function(u: jax.Array) -> jax.Array:
        return direct_wrench_symplectic_step(state, u, model, 0.02)[3:7]

    forward = jax.jacfwd(function)(wrench)
    reverse = jax.jacrev(function)(wrench)
    assert np.all(np.isfinite(forward))
    assert np.all(np.isfinite(reverse))
    np.testing.assert_allclose(forward, reverse, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(forward[:3, 1:], 0.5 * 0.02**2 * model.inertia_inv, rtol=2e-6)
