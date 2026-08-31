from __future__ import annotations

import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.dynamic_filter import DynamicFilterConfig
from crazyflow.safety.da_plcbf.dynamic_rollouts import evaluate_dynamic_quad_library
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig
from crazyflow.safety.da_plcbf.quad_uncertainty import VersionAModelSamples
from crazyflow.safety.da_plcbf.uncertain_dynamic_filter import (
    evaluate_uncertain_dynamic_quad_library,
    uncertain_dynamic_discrete_runtime_step,
)
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig


def _helpers() -> dict[str, object]:
    return runpy.run_path(
        str(Path(__file__).with_name("test_dynamic_rollouts.py")),
        run_name="uncertain_dynamic_filter_test_helpers",
    )


def _problem() -> tuple[object, ...]:
    helper = _helpers()
    model, actuator = helper["_physical"]()
    scenarios = helper["_dynamic_scenarios"](predictions=2, nodes=5)
    spec, params, actor_config = helper["_actor"](1)
    state = helper["_initial"]()[0]
    return model, actuator, scenarios, spec, params, actor_config, state


def _samples(model: object, *, efficiencies: jax.Array | None = None) -> VersionAModelSamples:
    models = jax.tree.map(lambda value: jnp.broadcast_to(value, (4, *jnp.shape(value))), model)
    return VersionAModelSamples(
        models=models,
        rotor_efficiency=(jnp.ones((4, 4)) if efficiencies is None else efficiencies),
        weights=jnp.full((4,), 0.25),
        sample_valid=jnp.ones((4,), dtype=bool),
        retained_variance_fraction=jnp.array(1.0),
        model_version=jnp.array(0, dtype=jnp.int32),
    )


def _evaluate(problem: tuple[object, ...], samples: VersionAModelSamples) -> object:
    model, actuator, scenarios, spec, params, actor_config, state = problem
    return evaluate_uncertain_dynamic_quad_library(
        params,
        spec,
        state[None],
        scenarios,
        model,
        samples,
        actuator,
        actor_config,
        QuadPolicyConfig(),
        VersionABarrierConfig(),
        dt=0.02,
        policy_gain=1.5,
        softmin_beta=40.0,
    )


def test_cartesian_axes_are_explicit_and_model_permutation_invariant() -> None:
    problem = _problem()
    samples = _samples(problem[0])
    original = jax.jit(lambda: _evaluate(problem, samples))()
    permutation = jnp.array([2, 0, 3, 1])
    permuted_samples = samples._replace(
        models=jax.tree.map(lambda value: value[permutation], samples.models),
        rotor_efficiency=samples.rotor_efficiency[permutation],
        weights=samples.weights[permutation],
        sample_valid=samples.sample_valid[permutation],
    )
    permuted = jax.jit(lambda: _evaluate(problem, permuted_samples))()

    assert original.rollouts.states.shape[:5] == (3, 1, 2, 4, 5)
    np.testing.assert_allclose(original.hard_values, permuted.hard_values, atol=2e-6)
    np.testing.assert_allclose(original.first_motor_forces, permuted.first_motor_forces, atol=2e-6)


def test_hard_value_is_exact_worst_cartesian_sample_not_weighted_average() -> None:
    problem = _problem()
    efficiencies = jnp.ones((4, 4)).at[3].set(0.35)
    result = _evaluate(problem, _samples(problem[0], efficiencies=efficiencies))
    per_combination = result.safety_values.prediction_hard_margins

    np.testing.assert_allclose(result.hard_values, jnp.min(per_combination, axis=-1), atol=1e-6)
    assert np.any(
        np.abs(np.asarray(result.hard_values - jnp.mean(per_combination, axis=-1))) > 1e-5
    )


def test_first_command_is_sample_independent_and_realized_efficiency_is_used() -> None:
    problem = _problem()
    efficiencies = jnp.asarray([[1.0, 1.0, 1.0, 1.0], [0.9, 0.8, 0.7, 0.6], [0.8] * 4, [0.5] * 4])
    result = _evaluate(problem, _samples(problem[0], efficiencies=efficiencies))
    first = result.rollouts.commanded_motor_forces[..., 0, :]
    expected = result.first_motor_forces[:, :, None, None, :]

    assert np.all(np.asarray(result.first_action_consistent))
    np.testing.assert_allclose(first, jnp.broadcast_to(expected, first.shape), atol=2e-6)
    np.testing.assert_allclose(
        result.rollouts.realized_motor_forces,
        result.rollouts.commanded_motor_forces * efficiencies[None, None, None, :, None, :],
        atol=1e-7,
    )


def test_invalid_dynamics_sample_fails_closed_in_filter() -> None:
    problem = _problem()
    model, actuator, scenarios, spec, params, actor_config, state = problem
    samples = _samples(model)._replace(sample_valid=jnp.array([True, True, False, True]))
    result = uncertain_dynamic_discrete_runtime_step(
        state,
        jnp.array([0.8, 0.0, 1.0]),
        jnp.zeros(3),
        jnp.array(-1),
        params,
        spec,
        scenarios,
        model,
        samples,
        actuator,
        actor_config,
        QuadPolicyConfig(),
        VersionABarrierConfig(),
        DynamicFilterConfig(),
        dt=0.02,
        policy_gain=1.5,
    )

    assert np.all(np.isneginf(np.asarray(result.library.hard_values)))
    assert not bool(result.selection.has_certificate)
    assert not bool(result.filter.proposal_accepted)
    assert bool(result.degraded)


def test_identical_nominal_samples_reduce_to_single_model_dynamic_library() -> None:
    problem = _problem()
    model, actuator, scenarios, spec, params, actor_config, state = problem
    uncertain = _evaluate(problem, _samples(model))
    nominal = evaluate_dynamic_quad_library(
        params,
        spec,
        state,
        scenarios,
        model,
        actuator,
        actor_config,
        QuadPolicyConfig(),
        VersionABarrierConfig(),
        dt=0.02,
        policy_gain=1.5,
        softmin_beta=40.0,
        current_action_tolerance=2e-5,
    )

    np.testing.assert_allclose(uncertain.hard_values[:, 0], nominal.hard_values, atol=2e-6)
    # The uncertain path evaluates an explicitly repeated R_m batch, so CUDA may choose a
    # different float32 GEMM kernel than the single-model path.  The commands remain well inside
    # the runtime's independently enforced 2e-5 consistency tolerance.
    np.testing.assert_allclose(
        uncertain.first_motor_forces[:, 0], nominal.first_motor_forces, atol=1e-5
    )


def test_exact_postcheck_returns_one_command_and_all_sampled_successors() -> None:
    problem = _problem()
    model, actuator, scenarios, spec, params, actor_config, state = problem
    samples = _samples(model, efficiencies=jnp.full((4, 4), 0.9))

    def run(candidate_state: jax.Array) -> object:
        return uncertain_dynamic_discrete_runtime_step(
            candidate_state,
            jnp.array([0.8, 0.0, 1.0]),
            jnp.zeros(3),
            jnp.array(-1),
            params,
            spec,
            scenarios,
            model,
            samples,
            actuator,
            actor_config,
            QuadPolicyConfig(),
            VersionABarrierConfig(),
            DynamicFilterConfig(),
            dt=0.02,
            policy_gain=1.5,
        )

    result = jax.jit(run)(state)
    assert result.motor_forces.shape == (4,)
    assert result.sampled_next_states.shape == (4, 13)
    assert np.all(np.isfinite(np.asarray(result.sampled_next_states)))
    if bool(result.filter.proposal_accepted):
        assert result.filter.proposal_exact_residual >= -2e-6
        assert result.filter.proposal_actuator_residual <= 2e-6
        assert result.filter.proposal_interval_margin >= -2e-6
