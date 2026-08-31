from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.control import Control
from crazyflow.dynamics import Dynamics
from crazyflow.safety.da_plcbf.actor import (
    SharedActorConfig,
    SharedActorSpec,
    initialize_shared_actor,
)
from crazyflow.safety.da_plcbf.full_stack import build_unclipped_full_stack_step
from crazyflow.safety.da_plcbf.quad_actor_losses import rigid_body_safety_batch_from_circles
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.selector import SelectionConfig
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig, VersionAModel
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator
from crazyflow.safety.da_plcbf.version_b_runtime import (
    VersionBRuntimeConfig,
    execute_version_b_held_command,
    replace_version_b_state,
    sim_data_to_version_b_state,
    version_b_action_evidence,
    version_b_runtime_step,
    version_b_shared_library_certificates,
)
from crazyflow.sim import Sim
from crazyflow.sim import functional as sim_functional
from crazyflow.sim.integration import Integrator


def _problem(device: str = "cpu") -> dict[str, Any]:
    sim = Sim(
        dynamics=Dynamics.first_principles,
        control=Control.force_torque,
        integrator=Integrator.symplectic_euler,
        freq=500,
        force_torque_freq=500,
        device=device,
        enable_mjx=False,
    )
    controller = sim.data.controls.force_torque.params
    physical = sim.data.params
    mass = physical.mass[0, 0, 0]
    gravity = physical.gravity_vec
    hover = jnp.array([mass * -gravity[2], 0.0, 0.0, 0.0])
    hover_motor_force = hover[0] / 4
    rpm2thrust = physical.rpm2thrust
    hover_rpm = (
        -rpm2thrust[1]
        + jnp.sqrt(rpm2thrust[1] ** 2 - 4 * rpm2thrust[2] * (rpm2thrust[0] - hover_motor_force))
    ) / (2 * rpm2thrust[2])
    data = sim.data.replace(
        states=sim.data.states.replace(
            pos=sim.data.states.pos.at[0, 0, 2].set(1.0),
            rotor_vel=jnp.full_like(sim.data.states.rotor_vel, hover_rpm),
        )
    )
    model = VersionAModel(
        mass=mass,
        gravity_vec=gravity,
        inertia=physical.J[0, 0],
        inertia_inv=physical.J_inv[0, 0],
        drag_matrix=physical.drag_matrix,
        wind_velocity=jnp.zeros(3),
        external_force=jnp.zeros(3),
        external_torque=jnp.zeros(3),
    )
    actuator = VersionAActuator(
        arm_length=controller["L"],
        thrust_to_torque=controller["thrust2torque"],
        mixing_matrix=controller["mixing_matrix"],
        thrust_min=controller["thrust_min"],
        thrust_max=controller["thrust_max"],
    )
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.zeros((1, 1, 3)),
        obstacle_radii=jnp.ones((1, 1)),
        obstacle_mask=jnp.zeros((1, 1), dtype=bool),
        arena_lower=jnp.array([[-4.0, -4.0, 0.1]]),
        arena_upper=jnp.array([[4.0, 4.0, 4.1]]),
        speed_limit=jnp.array([8.0]),
    )
    safety = rigid_body_safety_batch_from_circles(
        scenarios, angular_rate_max=20.0, tilt_max_radians=1.4
    )
    spec = SharedActorSpec(
        base_codes=jnp.zeros((1, 2)),
        base_desired_velocities=jnp.zeros((1, 3)),
        base_durations=jnp.array([0.2]),
        adaptive_mask=jnp.array([False]),
    )
    actor_config = SharedActorConfig(hidden_width=4, max_duration=0.5)
    actor_params = initialize_shared_actor(
        jax.random.key(3), spec, dimension=3, n_obstacles=1, config=actor_config
    )
    return {
        "sim": sim,
        "data": data,
        "hover": hover,
        "model": model,
        "actuator": actuator,
        "scenarios": scenarios,
        "safety": safety,
        "spec": spec,
        "actor_config": actor_config,
        "actor_params": actor_params,
        "quad_config": QuadPolicyConfig(),
        "barrier_config": VersionABarrierConfig(),
        "one_step": build_unclipped_full_stack_step(sim),
    }


def _runtime(problem: dict[str, Any], nominal: jax.Array, **config: Any) -> Any:
    previous_policy_index = config.pop("previous_policy_index", None)
    selection_config = config.pop("selection_config", None)
    return version_b_runtime_step(
        problem["data"],
        nominal,
        problem["actor_params"],
        problem["spec"],
        problem["scenarios"],
        problem["safety"],
        problem["model"],
        problem["actuator"],
        problem["actor_config"],
        problem["quad_config"],
        problem["barrier_config"],
        problem["one_step"],
        jnp.array([0.0, -1.0, -1.0, -1.0]),
        jnp.array([10.0, 1.0, 1.0, 1.0]),
        jnp.ones(4),
        jnp.array([10.0, 1.0, 1.0, 1.0]),
        VersionBRuntimeConfig(
            n_substeps=2,
            certificate_horizon=1,
            policy_gain=1.5,
            decay=0.99,
            tolerance=2e-5,
            qp_iterations=32,
            **config,
        ),
        previous_policy_index=previous_policy_index,
        selection_config=selection_config,
    )


def _two_policy_problem() -> dict[str, Any]:
    problem = _problem()
    spec = SharedActorSpec(
        base_codes=jnp.zeros((2, 2)),
        base_desired_velocities=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        base_durations=jnp.array([0.2, 0.2]),
        adaptive_mask=jnp.array([False, False]),
    )
    problem["spec"] = spec
    problem["actor_params"] = initialize_shared_actor(
        jax.random.key(4), spec, dimension=3, n_obstacles=1, config=problem["actor_config"]
    )
    return problem


def test_state_conversion_is_exact_xyzw_and_preserves_hidden_plant_state() -> None:
    problem = _problem()
    data = problem["data"]
    quaternion = jnp.array([0.10259784, -0.20519568, 0.30779353, 0.9233805])
    state = jnp.concatenate(
        (
            jnp.array([1.0, 2.0, 3.0]),
            quaternion,
            jnp.array([0.4, -0.5, 0.6]),
            jnp.array([-0.7, 0.8, -0.9]),
        )
    )
    replaced = replace_version_b_state(data, state)

    np.testing.assert_array_equal(sim_data_to_version_b_state(replaced), state)
    np.testing.assert_array_equal(replaced.states.quat[0, 0], quaternion)
    np.testing.assert_array_equal(replaced.states.rotor_vel, data.states.rotor_vel)
    np.testing.assert_array_equal(replaced.controls.rotor_vel, data.controls.rotor_vel)


def test_full_stack_library_certificate_uses_common_horizon_and_exact_trace() -> None:
    problem = _problem()
    result = version_b_shared_library_certificates(
        problem["data"],
        problem["actor_params"],
        problem["spec"],
        problem["scenarios"],
        problem["safety"],
        problem["model"],
        problem["actuator"],
        problem["actor_config"],
        problem["quad_config"],
        problem["barrier_config"],
        problem["one_step"],
        n_substeps=2,
        horizon=2,
        policy_gain=1.5,
        tolerance=2e-5,
    )

    assert result.values.shape == (1,)
    assert result.state_traces.shape == (1, 5, 13)
    assert result.held_interval_margins.shape == (1, 2)
    assert bool(result.rollout_valid[0])
    assert np.isfinite(float(result.values[0]))
    assert np.max(np.asarray(result.actuator_residuals)) <= 2e-5
    assert np.max(np.asarray(result.replay_state_errors)) <= 1e-7


def test_held_interval_uses_swept_static_sphere_not_only_safe_endpoint_nodes() -> None:
    problem = _problem()
    scenario = problem["scenarios"].replace(
        obstacle_centers=jnp.array([[[0.0, 0.0, 1.0]]]),
        obstacle_radii=jnp.array([[0.005]]),
        obstacle_mask=jnp.ones((1, 1), dtype=bool),
        speed_limit=jnp.array([100.0]),
    )
    safety = rigid_body_safety_batch_from_circles(
        scenario, angular_rate_max=20.0, tilt_max_radians=1.4
    )
    state = sim_data_to_version_b_state(problem["data"])
    crossing = replace_version_b_state(problem["data"], state.at[0].set(-0.02).at[7].set(20.0))
    held = execute_version_b_held_command(
        crossing,
        problem["hover"],
        problem["one_step"],
        safety,
        problem["barrier_config"],
        n_substeps=1,
        tolerance=2e-5,
    )

    assert held.node_interval_margin > 0
    assert held.interval_margin < 0
    assert held.state_trace[0, 0] < 0 < held.state_trace[-1, 0]


def test_certificate_depends_on_hidden_rotor_state_not_version_a_state_alone() -> None:
    problem = _problem()
    zero_rotors = problem["data"].replace(
        states=problem["data"].states.replace(
            rotor_vel=jnp.zeros_like(problem["data"].states.rotor_vel)
        )
    )
    np.testing.assert_array_equal(
        sim_data_to_version_b_state(zero_rotors), sim_data_to_version_b_state(problem["data"])
    )

    def certificate(data: Any) -> Any:
        return version_b_shared_library_certificates(
            data,
            problem["actor_params"],
            problem["spec"],
            problem["scenarios"],
            problem["safety"],
            problem["model"],
            problem["actuator"],
            problem["actor_config"],
            problem["quad_config"],
            problem["barrier_config"],
            problem["one_step"],
            n_substeps=4,
            horizon=2,
            policy_gain=1.5,
            tolerance=2e-5,
        )

    at_hover = certificate(problem["data"])
    from_zero = certificate(zero_rotors)

    assert bool(at_hover.rollout_valid[0])
    assert bool(from_zero.rollout_valid[0])
    assert not np.allclose(at_hover.state_traces, from_zero.state_traces, atol=1e-7)
    assert not np.allclose(at_hover.values, from_zero.values, atol=1e-7)


def test_nonphysical_negative_hidden_rotor_state_invalidates_certificate() -> None:
    problem = _problem()
    negative_rotors = problem["data"].replace(
        states=problem["data"].states.replace(
            rotor_vel=-jnp.ones_like(problem["data"].states.rotor_vel)
        )
    )
    result = version_b_shared_library_certificates(
        negative_rotors,
        problem["actor_params"],
        problem["spec"],
        problem["scenarios"],
        problem["safety"],
        problem["model"],
        problem["actuator"],
        problem["actor_config"],
        problem["quad_config"],
        problem["barrier_config"],
        problem["one_step"],
        n_substeps=1,
        horizon=1,
        policy_gain=1.5,
        tolerance=2e-5,
    )

    assert result.actuator_residuals[0, 0] > 0
    assert not bool(result.rollout_valid[0])
    assert np.isneginf(float(result.values[0]))


def test_accepted_result_passes_recomputed_exact_residual_and_final_data_crosscheck() -> None:
    problem = _problem()
    result = _runtime(problem, problem["hover"])
    compiled = jax.jit(lambda nominal: _runtime(problem, nominal))(problem["hover"])

    assert bool(result.has_certificate)
    assert bool(result.discrete_filter.proposal_accepted)
    assert not bool(result.discrete_filter.used_fallback)
    assert bool(result.applied_accepted)
    assert not bool(result.degraded)
    assert result.applied_exact_residual >= -2e-5
    assert result.postcheck_replay_error <= 2e-5
    np.testing.assert_allclose(
        result.applied_exact_residual, result.discrete_filter.proposal_exact_residual, atol=2e-6
    )
    np.testing.assert_allclose(compiled.action, result.action, atol=1e-7)
    assert bool(compiled.applied_accepted)

    independent = sim_functional.force_torque_control(problem["data"], result.action[None, None, :])
    for _ in range(2):
        independent = problem["one_step"](independent)
    for name in ("pos", "quat", "vel", "ang_vel", "rotor_vel"):
        np.testing.assert_allclose(
            getattr(result.next_data.states, name), getattr(independent.states, name), atol=1e-7
        )
    np.testing.assert_array_equal(
        result.next_data.controls.force_torque.cmd, independent.controls.force_torque.cmd
    )
    np.testing.assert_array_equal(result.next_data.core.steps, independent.core.steps)


def test_internal_allocation_clipping_rejects_proposal_and_uses_certified_fallback() -> None:
    problem = _problem()
    result = _runtime(problem, jnp.array([10.0, 0.0, 0.0, 0.0]))

    assert result.discrete_filter.proposal_actuator_residual > 2e-5
    assert not bool(result.discrete_filter.proposal_accepted)
    assert bool(result.discrete_filter.fallback_accepted)
    assert bool(result.discrete_filter.used_fallback)
    assert bool(result.applied_accepted)
    assert not bool(result.degraded)
    np.testing.assert_allclose(result.action, result.selected_fallback, atol=1e-7)


def test_full_stack_admissible_score_selection_logs_switch_and_hysteresis() -> None:
    problem = _two_policy_problem()
    certificates = version_b_shared_library_certificates(
        problem["data"],
        problem["actor_params"],
        problem["spec"],
        problem["scenarios"],
        problem["safety"],
        problem["model"],
        problem["actuator"],
        problem["actor_config"],
        problem["quad_config"],
        problem["barrier_config"],
        problem["one_step"],
        n_substeps=2,
        horizon=1,
        policy_gain=1.5,
        tolerance=2e-5,
    )
    scores = np.asarray(certificates.admissible_scores)
    assert np.all(np.asarray(certificates.values) >= 0)
    assert np.all(np.isfinite(scores) & (scores > 0))
    challenger = int(np.argmax(scores))
    incumbent = int(np.argmin(scores))
    gap = float(scores[challenger] - scores[incumbent])
    assert challenger != incumbent
    assert gap > 1e-4

    retained = _runtime(
        problem,
        problem["hover"],
        previous_policy_index=jnp.asarray(incumbent),
        selection_config=SelectionConfig(switch_score_margin=gap + 1e-3),
    )
    switched = _runtime(
        problem,
        problem["hover"],
        previous_policy_index=jnp.asarray(incumbent),
        selection_config=SelectionConfig(switch_score_margin=0.0),
    )

    assert bool(retained.selection.previous_eligible)
    assert bool(retained.selection.retained_by_hysteresis)
    assert not bool(retained.selection.switched)
    assert int(retained.selected_index) == incumbent
    assert not bool(switched.selection.retained_by_hysteresis)
    assert bool(switched.selection.switched)
    assert int(switched.selected_index) == challenger
    np.testing.assert_allclose(switched.admissible_scores, certificates.admissible_scores)


def test_no_current_certificate_is_explicitly_degraded() -> None:
    problem = _problem()
    outside = replace_version_b_state(
        problem["data"], sim_data_to_version_b_state(problem["data"]).at[2].set(-0.2)
    )
    problem["data"] = outside
    result = _runtime(problem, problem["hover"])

    assert not bool(result.has_certificate)
    assert not bool(result.discrete_filter.proposal_accepted)
    assert not bool(result.discrete_filter.fallback_accepted)
    assert not bool(result.applied_accepted)
    assert bool(result.degraded)
    assert result.certificates.values[0] < 0
    assert result.next_data.states.pos[0, 0, 2] < 0


def test_action_evidence_is_jittable_and_differentiates_full_stack_next_value() -> None:
    problem = _problem()

    def evaluate(command: jax.Array) -> Any:
        return version_b_action_evidence(
            problem["data"],
            command,
            jnp.asarray(0),
            problem["actor_params"],
            problem["spec"],
            problem["scenarios"],
            problem["safety"],
            problem["model"],
            problem["actuator"],
            problem["actor_config"],
            problem["quad_config"],
            problem["barrier_config"],
            problem["one_step"],
            n_substeps=2,
            horizon=1,
            policy_gain=1.5,
            tolerance=2e-5,
        )

    eager = evaluate(problem["hover"])
    compiled = jax.jit(evaluate)(problem["hover"])
    gradient = jax.grad(lambda command: evaluate(command).evaluation.next_value)(problem["hover"])

    np.testing.assert_allclose(
        compiled.evaluation.next_value, eager.evaluation.next_value, atol=1e-7
    )
    assert compiled.next_certificate.state_traces.shape == (1, 3, 13)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.linalg.norm(np.asarray(gradient)) > 0


def test_complete_runtime_executes_on_actual_gpu_when_available() -> None:
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("an actual GPU is not available in this environment")
    problem = _problem("gpu")
    result = jax.jit(lambda nominal: _runtime(problem, nominal))(problem["hover"])
    result.action.block_until_ready()

    assert next(iter(result.action.devices())).platform == "gpu"
    assert bool(result.has_certificate)
    assert bool(result.applied_accepted)
    assert not bool(result.degraded)
