"""Mathematically explicit hard-certificate policy selection with score hysteresis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class SelectionConfig:
    """Hard eligibility thresholds and the documented admissible-score switch gap."""

    minimum_hard_value: float = 0.0
    minimum_admissible_score: float = 0.0
    switch_score_margin: float = 0.02
    prefer_first_eligible: bool = False

    def validate(self) -> None:
        """Reject nonfinite thresholds and negative hysteresis."""
        values = (self.minimum_hard_value, self.minimum_admissible_score, self.switch_score_margin)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("selection thresholds must be finite")
        if self.switch_score_margin < 0:
            raise ValueError("switch_score_margin must be nonnegative")
        if not isinstance(self.prefer_first_eligible, bool):
            raise TypeError("prefer_first_eligible must be boolean")


class PolicySelection(NamedTuple):
    """Selected policy and complete eligibility/switch audit record."""

    selected_index: Array
    best_eligible_index: Array
    previous_index_valid: Array
    previous_eligible: Array
    has_certificate: Array
    retained_by_hysteresis: Array
    switched: Array
    selected_hard_value: Array
    selected_admissible_score: Array
    eligible: Array
    finite: Array


def select_hard_policy(
    hard_values: Array, admissible_scores: Array, previous_index: Array, config: SelectionConfig
) -> PolicySelection:
    """Select by positive hard value and admissible-set score, retaining close incumbents.

    Eligibility is a conjunction of finite values, ``hard_value >= minimum_hard_value``, and
    ``admissible_score > minimum_admissible_score``. Among eligible policies the exact maximum
    admissible score is the challenger (``argmax`` gives a deterministic lowest-index tie break).
    An eligible incumbent is retained unless the challenger exceeds its score by *strictly more*
    than ``switch_score_margin``. This is score hysteresis, not maneuver-state-machine logic.

    If no certificate exists, the finite policy with the largest hard value is returned only as an
    explicitly uncertified best effort. If all values are nonfinite, index zero is the deterministic
    sentinel and its selected diagnostics remain nonfinite.
    """
    config.validate()
    if hard_values.ndim != 1 or admissible_scores.shape != hard_values.shape:
        raise ValueError("hard_values and admissible_scores must be matching 1-D arrays")
    if hard_values.size == 0:
        raise ValueError("the policy library must not be empty")
    if jnp.ndim(previous_index) != 0:
        raise ValueError("previous_index must be a scalar")

    finite = jnp.isfinite(hard_values) & jnp.isfinite(admissible_scores)
    eligible = (
        finite
        & (hard_values >= config.minimum_hard_value)
        & (admissible_scores > config.minimum_admissible_score)
    )
    has_certificate = jnp.any(eligible)
    best_eligible_index = jnp.argmax(jnp.where(eligible, admissible_scores, -jnp.inf))

    previous_index = jnp.asarray(previous_index, dtype=jnp.int32)
    previous_index_valid = (previous_index >= 0) & (previous_index < hard_values.size)
    safe_previous_index = jnp.clip(previous_index, 0, hard_values.size - 1)
    previous_eligible = previous_index_valid & eligible[safe_previous_index]
    incumbent_score = admissible_scores[safe_previous_index]
    challenger_score = admissible_scores[best_eligible_index]
    first_preferred = jnp.asarray(config.prefer_first_eligible) & eligible[0]
    retained = (
        has_certificate
        & previous_eligible
        & (challenger_score <= incumbent_score + config.switch_score_margin)
        & ~first_preferred
    )
    certified_index = jnp.where(
        first_preferred,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.where(retained, safe_previous_index, best_eligible_index),
    )

    any_finite_hard = jnp.any(jnp.isfinite(hard_values))
    best_effort_index = jnp.argmax(jnp.where(jnp.isfinite(hard_values), hard_values, -jnp.inf))
    best_effort_index = jnp.where(any_finite_hard, best_effort_index, 0)
    selected_index = jnp.where(has_certificate, certified_index, best_effort_index)
    switched = has_certificate & previous_index_valid & (selected_index != safe_previous_index)
    return PolicySelection(
        selected_index=selected_index,
        best_eligible_index=best_eligible_index,
        previous_index_valid=previous_index_valid,
        previous_eligible=previous_eligible,
        has_certificate=has_certificate,
        retained_by_hysteresis=retained,
        switched=switched,
        selected_hard_value=hard_values[selected_index],
        selected_admissible_score=admissible_scores[selected_index],
        eligible=eligible,
        finite=finite,
    )


__all__ = ["PolicySelection", "SelectionConfig", "select_hard_policy"]
