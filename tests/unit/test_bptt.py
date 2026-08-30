from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn

if TYPE_CHECKING:
    from types import ModuleType


def _load_bptt_benchmark() -> ModuleType:
    """Load the standalone benchmark without making benchmark/ an installed package."""
    path = Path(__file__).parents[2] / "benchmark" / "bptt.py"
    spec = importlib.util.spec_from_file_location("crazyflow_bptt_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bptt = _load_bptt_benchmark()


class PublicArtifactActor(nn.Module):
    """Direct copy of the pinned artifact's actor for initialization/output parity checks."""

    hidden_size: int = 64
    act_dim: int = 4
    num_layers: int = 2

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        """Apply the artifact's two hidden layers and bounded output layer."""
        x = obs
        for _ in range(self.num_layers):
            x = nn.Dense(
                self.hidden_size,
                kernel_init=nn.initializers.orthogonal(),
                bias_init=nn.initializers.zeros,
            )(x)
            x = nn.tanh(x)
        mean = nn.Dense(
            self.act_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(x)
        return nn.tanh(mean)


def _assert_trees_allclose(
    actual: Any, expected: Any, *, atol: float = 0.0, rtol: float = 0.0
) -> None:
    """Compare matching PyTrees leaf by leaf."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(actual_leaf, expected_leaf, atol=atol, rtol=rtol)


@pytest.mark.unit
def test_actor_initialization_and_output_match_public_linen_artifact():
    """Seed 42 must produce the same kernels and outputs as the public Linen actor."""
    config = bptt.public_config(device="cpu")
    device = jax.devices("cpu")[0]
    init_key, _ = jax.random.split(jax.device_put(jax.random.key(config.seed), device))
    dummy_obs = jnp.zeros((1, 50), dtype=jnp.float32, device=device)

    actual_params = bptt._init_actor(config, dummy_obs.shape[-1], init_key, device)
    reference_actor = PublicArtifactActor(hidden_size=config.hidden_size)
    expected_params = reference_actor.init(init_key, dummy_obs)

    _assert_trees_allclose(actual_params, expected_params)
    observations = jax.random.normal(jax.random.key(7), (3, dummy_obs.shape[-1]))
    actual_output = bptt._actor_mean(actual_params, observations)
    expected_output = reference_actor.apply(expected_params, observations)
    np.testing.assert_allclose(actual_output, expected_output, atol=0.0, rtol=0.0)


@pytest.mark.unit
def test_optimizer_one_update_matches_direct_optax_adamw():
    """The benchmark update must retain the artifact's exact Optax AdamW configuration."""
    config = bptt.public_config(device="cpu")
    params = {
        "kernel": jnp.array([[0.25, -0.5], [1.0, -2.0]], dtype=jnp.float32),
        "bias": jnp.array([0.1, -0.2], dtype=jnp.float32),
    }
    gradients = jax.tree.map(
        lambda value: jnp.arange(value.size).reshape(value.shape) + 0.25, params
    )

    benchmark_optimizer = bptt._make_optimizer(config)
    actual_params, actual_state = bptt._optimizer_update(
        benchmark_optimizer, params, gradients, benchmark_optimizer.init(params)
    )

    reference_optimizer = optax.adamw(
        learning_rate=config.learning_rate,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    reference_state = reference_optimizer.init(params)
    expected_updates, expected_state = reference_optimizer.update(
        gradients, reference_state, params
    )
    expected_params = optax.apply_updates(params, expected_updates)

    _assert_trees_allclose(actual_params, expected_params)
    _assert_trees_allclose(actual_state, expected_state)


@pytest.mark.unit
def test_public_reference_matches_endpoint_inclusive_artifact():
    """The public protocol uses 1000 endpoint-inclusive samples over its 20-second episode."""
    config = bptt.public_config(n_envs=1)
    steps = jnp.array([0, 500, 999], dtype=jnp.int32)

    position, _ = bptt._reference(jnp.array([0.0]), steps[None, :], config)

    phase = 4.0 * np.pi * np.asarray(steps) / 999.0
    expected = np.stack((np.sin(phase), np.zeros(3), 0.5 * np.sin(2.0 * phase) + 1.25), axis=-1)
    np.testing.assert_allclose(position[0], expected, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides", ({"env_freq": 0}, {"sim_freq": 0}, {"trajectory_time": 0.0}, {"samples_dt": 0.0})
)
def test_invalid_nonpositive_protocol_values_are_rejected(overrides: dict[str, Any]):
    """Invalid frequencies and times should raise a useful error, never arithmetic exceptions."""
    config = replace(bptt.public_config(), **overrides)

    with pytest.raises(ValueError, match="positive"):
        config.validate()


@pytest.mark.unit
def test_bptt_smoke_updates_policy_with_finite_gradients():
    """A differentiated rollout must produce and apply a finite, nonzero policy gradient."""
    config = replace(
        bptt.public_config(device="cpu"),
        n_envs=2,
        rollout_steps=4,
        total_timesteps=16,
        n_samples=3,
        hidden_size=4,
        action_weights=(0.0, 0.0, 0.0, 0.0),
        delta_action_weights=(0.0, 0.0, 0.0, 0.0),
        weight_decay=0.0,
    )

    result = bptt.run_bptt(config, repeats=1, evaluation_steps=0)

    assert result.actual_timesteps == 16
    assert np.isfinite(result.first_loss)
    assert np.isfinite(result.final_loss)
    assert np.isfinite(result.first_gradient_norm)
    assert np.isfinite(result.final_gradient_norm)
    assert result.first_gradient_norm > 0.0
    assert result.parameter_delta_norm > 0.0
    assert result.device_kind
    assert result.jax_version
    assert result.git_commit
