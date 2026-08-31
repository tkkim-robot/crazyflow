from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.discrete_filter import (
    DiscreteActionEvaluation,
    discrete_nonlinear_plcbf_filter,
)
from crazyflow.safety.da_plcbf.experiments import _discrete_kkt


def _evaluation(
    residual_fn: object,
    *,
    current_value: float = 1.0,
    decay: float = 0.9,
    interval_fn: object | None = None,
    actuator_fn: object | None = None,
) -> object:
    interval = (lambda _action: jnp.asarray(1.0)) if interval_fn is None else interval_fn
    actuator = (lambda _action: jnp.asarray(0.0)) if actuator_fn is None else actuator_fn

    def evaluate(action: jax.Array) -> DiscreteActionEvaluation:
        return DiscreteActionEvaluation(
            next_value=decay * current_value + residual_fn(action),
            interval_margin=interval(action),
            actuator_residual=actuator(action),
            applied_action=action,
        )

    return evaluate


def _run(evaluate: object, **replacements: object) -> object:
    arguments: dict[str, object] = {
        "nominal_action": jnp.array([1.0]),
        "fallback_action": jnp.array([0.0]),
        "action_lower": jnp.array([-2.0]),
        "action_upper": jnp.array([2.0]),
        "weight": jnp.ones(1),
        "trust_radius": jnp.array([2.0]),
        "current_value": jnp.array(1.0),
        "has_certificate": jnp.array(True),
        "evaluate_action": evaluate,
        "decay": 0.9,
        "tolerance": 1e-6,
    }
    arguments.update(replacements)
    return discrete_nonlinear_plcbf_filter(**arguments)


def test_linearized_proposal_requires_exact_nonlinear_acceptance() -> None:
    # At u=1, the tangent of 0.1-u^2 proposes u=0.55. The exact residual there is negative,
    # demonstrating why QP feasibility alone cannot authorize a nonlinear plant command.
    evaluate = _evaluation(lambda action: 0.1 - action[0] ** 2)
    result = _run(evaluate)

    assert bool(result.proposal_feasible)
    np.testing.assert_allclose(result.action, np.array([0.0]), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result.residual_gradient, np.array([-2.0]), atol=1e-6)
    assert result.proposal_exact_residual < 0
    assert not bool(result.proposal_accepted)
    assert bool(result.fallback_accepted)
    assert bool(result.used_fallback)
    assert not bool(result.degraded)


def test_fallback_kkt_diagnostics_remain_bound_to_the_qp_proposal() -> None:
    """A nonlinear rejection must not substitute the fallback into the QP stationarity test."""
    evaluate = _evaluation(lambda action: 0.1 - action[0] ** 2)
    result = _run(evaluate, weight=jnp.asarray([1.0 / 16.0]))

    assert bool(result.used_fallback)
    assert not np.allclose(np.asarray(result.qp_action), np.asarray(result.action))
    residual = _discrete_kkt(
        SimpleNamespace(filter=result), jnp.asarray([-2.0]), jnp.asarray([2.0])
    )
    assert np.isfinite(float(residual))
    assert float(residual) < 1e-5


def test_safe_nonlinear_nominal_passes_without_intervention() -> None:
    evaluate = _evaluation(lambda action: 0.5 - 0.1 * action[0] ** 2)
    result = _run(evaluate, nominal_action=jnp.array([0.4]))

    assert bool(result.proposal_accepted)
    assert not bool(result.used_fallback)
    np.testing.assert_allclose(result.action, np.array([0.4]), rtol=0.0, atol=1e-6)
    assert result.proposal_exact_residual > 0


@pytest.mark.parametrize(("interval", "actuator"), [(-1e-3, 0.0), (1.0, 1e-3)])
def test_exact_interval_and_actuator_checks_can_reject_discrete_safe_proposal(
    interval: float, actuator: float
) -> None:
    evaluate = _evaluation(
        lambda _action: jnp.asarray(0.5),
        interval_fn=lambda action: jnp.where(action[0] > 0.5, interval, 1.0),
        actuator_fn=lambda action: jnp.where(action[0] > 0.5, actuator, 0.0),
    )
    result = _run(evaluate)

    assert result.proposal_exact_residual > 0
    assert not bool(result.proposal_accepted)
    assert bool(result.fallback_accepted)
    np.testing.assert_array_equal(result.action, np.array([0.0]))


def test_trust_region_is_part_of_the_qp_not_a_posthoc_clip() -> None:
    evaluate = _evaluation(lambda action: action[0] - 0.8)
    result = _run(
        evaluate,
        nominal_action=jnp.array([0.0]),
        fallback_action=jnp.array([1.0]),
        trust_radius=jnp.array([0.25]),
    )

    assert not bool(result.proposal_feasible)
    assert bool(result.fallback_accepted)
    np.testing.assert_array_equal(result.trust_lower, np.array([-0.25]))
    np.testing.assert_array_equal(result.trust_upper, np.array([0.25]))
    np.testing.assert_array_equal(result.action, np.array([1.0]))


def test_no_current_certificate_is_explicitly_degraded_even_if_next_residual_is_positive() -> None:
    evaluate = _evaluation(lambda _action: jnp.asarray(10.0))
    result = _run(evaluate, has_certificate=jnp.array(False), current_value=jnp.array(-0.1))

    assert not bool(result.proposal_accepted)
    assert not bool(result.fallback_accepted)
    assert bool(result.degraded)
    assert bool(result.used_fallback)


@pytest.mark.parametrize("fallback", [jnp.array([jnp.nan]), jnp.array([3.0])])
def test_invalid_fallback_is_not_silently_clipped_and_is_reported(fallback: jax.Array) -> None:
    evaluate = _evaluation(lambda _action: jnp.asarray(-1.0))
    result = _run(evaluate, fallback_action=fallback)

    assert not bool(result.fallback_input_valid)
    assert bool(result.fallback_substituted)
    assert bool(result.degraded)
    np.testing.assert_array_equal(result.action, np.array([0.0]))


def test_filter_is_jittable_and_differentiates_the_exact_transition() -> None:
    def call(nominal: jax.Array) -> object:
        evaluate = _evaluation(lambda action: 0.2 - jnp.sin(action[0]) ** 2)
        return _run(evaluate, nominal_action=nominal)

    eager = call(jnp.array([0.7]))
    compiled = jax.jit(call)(jnp.array([0.7]))

    np.testing.assert_allclose(compiled.action, eager.action, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(compiled.residual_gradient, eager.residual_gradient, atol=1e-6)
    np.testing.assert_allclose(eager.residual_gradient, -jnp.sin(1.4), atol=1e-6)


@pytest.mark.parametrize("decay", [0.0, -1.0, 1.1, float("nan"), float("inf")])
def test_filter_rejects_invalid_decay(decay: float) -> None:
    with pytest.raises(ValueError, match="decay"):
        _run(_evaluation(lambda action: action[0]), decay=decay)


@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf")])
def test_filter_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        _run(_evaluation(lambda action: action[0]), tolerance=tolerance)
