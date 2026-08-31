"""Static configuration for the DA-PLCBF reference implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class RolloutConfig:
    """Fixed-shape planar reference rollout settings.

    Attributes:
        dt: Duration of one simulator step in seconds.
        horizon: Common certificate horizon in simulator steps.
        policy_gain: Feedback gain from velocity error to acceleration.
        action_limit: Symmetric acceleration limit in m/s².
        safety_margin: Extra geometric clearance around each obstacle in metres.
        softmin_beta: Inverse temperature for conservative training minima.
    """

    dt: float = 0.05
    horizon: int = 40
    policy_gain: float = 2.0
    action_limit: float = 3.0
    safety_margin: float = 0.05
    softmin_beta: float = 20.0

    def validate(self) -> None:
        """Reject settings that change the mathematical contract silently."""
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if (
            isinstance(self.horizon, bool)
            or not isinstance(self.horizon, Integral)
            or self.horizon <= 0
        ):
            raise ValueError("horizon must be a positive integer")
        if not math.isfinite(self.policy_gain) or self.policy_gain <= 0:
            raise ValueError("policy_gain must be finite and positive")
        if not math.isfinite(self.action_limit) or self.action_limit <= 0:
            raise ValueError("action_limit must be finite and positive")
        if not math.isfinite(self.safety_margin) or self.safety_margin < 0:
            raise ValueError("safety_margin must be finite and nonnegative")
        if not math.isfinite(self.softmin_beta) or self.softmin_beta <= 0:
            raise ValueError("softmin_beta must be finite and positive")


@dataclass(frozen=True)
class LibraryLossConfig:
    """Weights and temperatures for the PL-CBF-aligned library objective."""

    target_margin: float = 0.02
    coverage_softplus_temperature: float = 0.05
    safe_count_temperature: float = 0.05
    covariance_regularizer: float = 1e-4
    log_epsilon: float = 1e-6
    coverage_weight: float = 1.0
    redundancy_weight: float = 0.1
    diversity_weight: float = 0.01
    code_weight: float = 0.01
    action_weight: float = 1e-3
    action_rate_weight: float = 1e-3
    terminal_weight: float = 1e-3
    trust_weight: float = 1e-2

    def validate(self) -> None:
        """Validate temperatures, stabilizers, and nonnegative loss weights."""
        positive = (
            self.coverage_softplus_temperature,
            self.safe_count_temperature,
            self.covariance_regularizer,
            self.log_epsilon,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("temperatures and numerical stabilizers must be finite and positive")
        weights = (
            self.coverage_weight,
            self.redundancy_weight,
            self.diversity_weight,
            self.code_weight,
            self.action_weight,
            self.action_rate_weight,
            self.terminal_weight,
            self.trust_weight,
        )
        if not all(math.isfinite(value) and value >= 0 for value in weights):
            raise ValueError("loss weights must be finite and nonnegative")
        if not math.isfinite(self.target_margin) or self.target_margin < 0:
            raise ValueError("target_margin must be finite and nonnegative")
