from dataclasses import replace

import jax.numpy as jnp
import pytest

from crazyflow.safety.da_plcbf.config import LibraryLossConfig, RolloutConfig


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dt", 0.0),
        ("dt", -0.1),
        ("horizon", 0),
        ("policy_gain", 0.0),
        ("action_limit", 0.0),
        ("safety_margin", -0.01),
        ("softmin_beta", 0.0),
    ],
)
def test_rollout_config_rejects_values_outside_its_contract(field: str, value: float) -> None:
    config = replace(RolloutConfig(), **{field: value})

    with pytest.raises(ValueError):
        config.validate()


def test_rollout_config_accepts_boundary_and_default_values() -> None:
    RolloutConfig().validate()
    replace(RolloutConfig(), horizon=1, safety_margin=0.0).validate()


@pytest.mark.parametrize(
    "field",
    [
        "coverage_softplus_temperature",
        "safe_count_temperature",
        "covariance_regularizer",
        "log_epsilon",
    ],
)
def test_loss_config_rejects_nonpositive_temperatures_and_stabilizers(field: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        replace(LibraryLossConfig(), **{field: 0.0}).validate()


@pytest.mark.parametrize(
    "field",
    [
        "coverage_weight",
        "redundancy_weight",
        "diversity_weight",
        "code_weight",
        "action_weight",
        "action_rate_weight",
        "terminal_weight",
        "trust_weight",
    ],
)
def test_loss_config_rejects_negative_weights(field: str) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        replace(LibraryLossConfig(), **{field: -jnp.finfo(jnp.float32).eps}).validate()


def test_loss_config_accepts_zero_weights_and_default_values() -> None:
    LibraryLossConfig().validate()
    replace(
        LibraryLossConfig(),
        coverage_weight=0.0,
        redundancy_weight=0.0,
        diversity_weight=0.0,
        code_weight=0.0,
        action_weight=0.0,
        action_rate_weight=0.0,
        terminal_weight=0.0,
        trust_weight=0.0,
    ).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dt", float("nan")),
        ("dt", float("inf")),
        ("policy_gain", float("nan")),
        ("action_limit", float("inf")),
        ("safety_margin", float("nan")),
        ("softmin_beta", float("inf")),
        ("horizon", 1.5),
        ("horizon", True),
    ],
)
def test_rollout_config_rejects_nonfinite_and_noninteger_contract_values(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError):
        replace(RolloutConfig(), **{field: value}).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_margin", float("nan")),
        ("target_margin", float("inf")),
        ("coverage_softplus_temperature", float("nan")),
        ("safe_count_temperature", float("inf")),
        ("covariance_regularizer", float("inf")),
        ("coverage_weight", float("nan")),
        ("trust_weight", float("inf")),
    ],
)
def test_loss_config_rejects_every_nonfinite_scalar(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        replace(LibraryLossConfig(), **{field: value}).validate()
