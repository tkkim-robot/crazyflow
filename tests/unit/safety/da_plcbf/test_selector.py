from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.selector import SelectionConfig, select_hard_policy


def test_selector_chooses_largest_admissible_score_among_hard_safe_policies() -> None:
    result = select_hard_policy(
        hard_values=jnp.array([-0.1, 0.2, 0.5, 0.3]),
        admissible_scores=jnp.array([0.99, 0.3, 0.8, 0.6]),
        previous_index=jnp.array(-1),
        config=SelectionConfig(switch_score_margin=0.05),
    )

    assert bool(result.has_certificate)
    assert int(result.selected_index) == 2
    np.testing.assert_array_equal(result.eligible, np.array([False, True, True, True]))
    assert not bool(result.retained_by_hysteresis)


def test_score_hysteresis_retains_close_incumbent_and_switches_on_strict_gap() -> None:
    config = SelectionConfig(switch_score_margin=0.1)
    retained = select_hard_policy(jnp.ones(3), jnp.array([0.2, 0.7, 0.8]), jnp.array(1), config)
    switched = select_hard_policy(jnp.ones(3), jnp.array([0.2, 0.7, 0.8001]), jnp.array(1), config)

    assert int(retained.selected_index) == 1
    assert bool(retained.retained_by_hysteresis)
    assert not bool(retained.switched)
    assert int(switched.selected_index) == 2
    assert not bool(switched.retained_by_hysteresis)
    assert bool(switched.switched)


def test_unsafe_incumbent_is_never_retained_by_hysteresis() -> None:
    result = select_hard_policy(
        hard_values=jnp.array([0.4, -1e-4]),
        admissible_scores=jnp.array([0.4, 1.0]),
        previous_index=jnp.array(1),
        config=SelectionConfig(switch_score_margin=100.0),
    )

    assert int(result.selected_index) == 0
    assert not bool(result.previous_eligible)
    assert not bool(result.retained_by_hysteresis)
    assert bool(result.switched)


def test_no_certificate_uses_largest_finite_hard_value_as_explicit_best_effort() -> None:
    result = select_hard_policy(
        hard_values=jnp.array([-0.3, jnp.nan, -0.1]),
        admissible_scores=jnp.array([0.9, 0.8, 0.0]),
        previous_index=jnp.array(0),
        config=SelectionConfig(),
    )

    assert not bool(result.has_certificate)
    assert int(result.selected_index) == 2
    assert result.selected_hard_value == pytest.approx(-0.1)


def test_selector_is_jittable_and_ties_are_deterministic() -> None:
    function = jax.jit(
        lambda values, scores, previous: select_hard_policy(
            values, scores, previous, SelectionConfig(switch_score_margin=0.0)
        )
    )
    result = function(jnp.ones(3), jnp.full(3, 0.5), jnp.array(-1))

    assert int(result.selected_index) == 0
    assert int(result.best_eligible_index) == 0


@pytest.mark.parametrize(
    "config",
    [
        SelectionConfig(minimum_hard_value=float("nan")),
        SelectionConfig(minimum_admissible_score=float("inf")),
        SelectionConfig(switch_score_margin=-1.0),
    ],
)
def test_selector_rejects_invalid_configuration(config: SelectionConfig) -> None:
    with pytest.raises(ValueError):
        select_hard_policy(jnp.ones(2), jnp.ones(2), jnp.array(0), config)
