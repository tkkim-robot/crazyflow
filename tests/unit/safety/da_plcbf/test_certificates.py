from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.actor import (
    SharedActorConfig,
    SharedActorSpec,
    initialize_shared_actor,
)
from crazyflow.safety.da_plcbf.capsules import CapsuleObstacleSet
from crazyflow.safety.da_plcbf.certificates import version_a_shared_library_certificates
from crazyflow.safety.da_plcbf.quad_actor_losses import rigid_body_safety_batch_from_circles
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig, VersionAModel
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator


def _physical() -> tuple[VersionAModel, VersionAActuator]:
    params: dict[str, Any] = load_params("cf21B_500")
    model = VersionAModel(
        mass=jnp.asarray(params["mass"]),
        gravity_vec=jnp.asarray(params["gravity_vec"]),
        inertia=jnp.asarray(params["J"]),
        inertia_inv=jnp.linalg.inv(jnp.asarray(params["J"])),
        drag_matrix=jnp.asarray(params["drag_matrix"]),
        wind_velocity=jnp.zeros(3),
        external_force=jnp.zeros(3),
        external_torque=jnp.zeros(3),
    )
    actuator = VersionAActuator(
        arm_length=jnp.asarray(params["L"]),
        thrust_to_torque=jnp.asarray(params["thrust2torque"]),
        mixing_matrix=jnp.asarray(params["mixing_matrix"]),
        thrust_min=jnp.asarray(params["thrust_min"]),
        thrust_max=jnp.asarray(params["thrust_max"]),
    )
    return model, actuator


def _problem() -> tuple[object, ...]:
    model, actuator = _physical()
    spec = SharedActorSpec(
        base_codes=jnp.eye(3),
        base_desired_velocities=jnp.array([[0.0, 0.0, 0.0], [0.4, 0.1, 0.0], [-0.3, 0.2, 0.0]]),
        base_durations=jnp.array([0.3, 0.35, 0.4]),
        adaptive_mask=jnp.array([False, True, True]),
    )
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.array([[[2.2, 1.5, 1.2]]]),
        obstacle_radii=jnp.array([[0.2]]),
        obstacle_mask=jnp.ones((1, 1), dtype=bool),
        arena_lower=jnp.array([[-2.0, -2.0, 0.1]]),
        arena_upper=jnp.array([[3.0, 2.0, 2.4]]),
        speed_limit=jnp.array([3.0]),
    )
    safety = rigid_body_safety_batch_from_circles(
        scenarios, angular_rate_max=8.0, tilt_max_radians=1.1
    )
    actor_config = SharedActorConfig(hidden_width=8, max_duration=0.8)
    params = initialize_shared_actor(
        jax.random.key(6), spec, dimension=3, n_obstacles=1, config=actor_config
    )
    state = jnp.array([0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.05, -0.02, 0.1, 0.01, 0.02, -0.01])
    return model, actuator, spec, scenarios, safety, actor_config, params, state


def test_shared_library_certificates_have_exact_shapes_and_directional_gradient() -> None:
    model, actuator, spec, scenarios, safety, actor_config, params, state = _problem()
    barrier = VersionABarrierConfig(minimum_tie_tolerance=1e-7)

    def certificates(candidate_state: jax.Array) -> object:
        return version_a_shared_library_certificates(
            candidate_state,
            params,
            spec,
            scenarios,
            safety,
            model,
            actuator,
            actor_config,
            QuadPolicyConfig(),
            barrier,
            dt=0.02,
            horizon=4,
            policy_gain=1.5,
        )

    result = jax.jit(certificates)(state)
    library = result.certificates
    assert library.values.shape == (3,)
    assert library.gradients.shape == (3, 13)
    assert library.gradient_valid.shape == (3,)
    assert library.fallback_wrenches.shape == (3, 4)
    assert np.all(np.asarray(result.rollout_valid))
    assert np.all(np.asarray(result.includes_current_state))
    assert np.any(np.asarray(library.gradient_valid))

    policy = int(np.flatnonzero(np.asarray(library.gradient_valid))[0])
    direction = jnp.array([0.2, -0.1, 0.15, 0.0, 0.0, 0.0, 0.0, -0.05, 0.1, 0.0, 0.0, 0.0, 0.0])
    epsilon = 2e-3
    finite_difference = (
        certificates(state + epsilon * direction).certificates.values[policy]
        - certificates(state - epsilon * direction).certificates.values[policy]
    ) / (2 * epsilon)
    autodiff = jnp.dot(library.gradients[policy], direction)
    np.testing.assert_allclose(autodiff, finite_difference, rtol=2e-2, atol=2e-3)


def test_tied_hard_minimum_is_reported_but_not_eligible_as_one_gradient_face() -> None:
    model, actuator = _physical()
    spec = SharedActorSpec(
        base_codes=jnp.zeros((1, 1)),
        base_desired_velocities=jnp.zeros((1, 3)),
        base_durations=jnp.array([0.2]),
        adaptive_mask=jnp.array([False]),
    )
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.zeros((1, 1, 3)),
        obstacle_radii=jnp.ones((1, 1)),
        obstacle_mask=jnp.zeros((1, 1), dtype=bool),
        arena_lower=jnp.array([[-1.0, -1.0, 0.0]]),
        arena_upper=jnp.array([[1.0, 1.0, 2.0]]),
        speed_limit=jnp.array([3.0]),
    )
    safety = rigid_body_safety_batch_from_circles(
        scenarios, angular_rate_max=8.0, tilt_max_radians=1.0
    )
    actor_config = SharedActorConfig(hidden_width=4, max_duration=0.5)
    params = initialize_shared_actor(
        jax.random.key(1), spec, dimension=3, n_obstacles=1, config=actor_config
    )
    state = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = version_a_shared_library_certificates(
        state,
        params,
        spec,
        scenarios,
        safety,
        model,
        actuator,
        actor_config,
        QuadPolicyConfig(),
        VersionABarrierConfig(minimum_tie_tolerance=1e-3),
        dt=1e-6,
        horizon=1,
        policy_gain=1.5,
    )

    assert np.isfinite(float(result.certificates.values[0]))
    assert result.second_value_gaps[0] <= 1e-3
    assert not bool(result.certificates.gradient_valid[0])


def test_nonfinite_state_fails_closed_without_a_runtime_certificate() -> None:
    model, actuator, spec, scenarios, safety, actor_config, params, state = _problem()
    result = version_a_shared_library_certificates(
        state.at[0].set(jnp.nan),
        params,
        spec,
        scenarios,
        safety,
        model,
        actuator,
        actor_config,
        QuadPolicyConfig(),
        VersionABarrierConfig(),
        dt=0.02,
        horizon=2,
        policy_gain=1.5,
    )

    assert np.all(np.isneginf(np.asarray(result.certificates.values)))
    assert not np.any(np.asarray(result.certificates.gradient_valid))


def test_capsule_node_and_exact_swept_values_are_part_of_the_hard_certificate() -> None:
    model, actuator, spec, scenarios, safety, actor_config, params, state = _problem()
    kwargs = dict(dt=0.02, horizon=2, policy_gain=1.5)
    without = version_a_shared_library_certificates(
        state,
        params,
        spec,
        scenarios,
        safety,
        model,
        actuator,
        actor_config,
        QuadPolicyConfig(),
        VersionABarrierConfig(),
        **kwargs,
    )
    capsule = CapsuleObstacleSet(
        segment_start=jnp.array([[[0.1, 0.2, 0.7]]]),
        segment_end=jnp.array([[[0.1, 0.2, 1.3]]]),
        radii=jnp.array([[0.15]]),
        mask=jnp.array([[True]]),
    )
    with_capsule = version_a_shared_library_certificates(
        state,
        params,
        spec,
        scenarios,
        safety,
        model,
        actuator,
        actor_config,
        QuadPolicyConfig(),
        VersionABarrierConfig(),
        capsules=capsule,
        **kwargs,
    )

    # H=2 gives three node values and two exact swept-segment values per capsule.
    assert with_capsule.constraint_values.shape[1] == without.constraint_values.shape[1] + 5
    assert np.all(np.asarray(with_capsule.certificates.values) < 0)
    assert np.all(np.asarray(with_capsule.active_indices) >= without.constraint_values.shape[1])
