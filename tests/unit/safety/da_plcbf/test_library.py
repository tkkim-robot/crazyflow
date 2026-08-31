import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.actor import SharedActorConfig, initialize_shared_actor
from crazyflow.safety.da_plcbf.library import (
    build_shared_quad_library_spec,
    descriptor_targets_from_spec,
    slice_shared_actor_policy,
)


def test_default_library_has_exact_k64_shared_actor_contract() -> None:
    first = build_shared_quad_library_spec()
    second = build_shared_quad_library_spec()

    assert first.base_codes.shape == (64, 8)
    assert first.base_desired_velocities.shape == (64, 3)
    assert first.base_durations.shape == (64,)
    assert np.count_nonzero(~np.asarray(first.adaptive_mask)) == 8
    assert np.all(np.asarray(first.base_desired_velocities[0]) == 0)
    assert np.all(np.asarray(first.base_durations) >= 0.35)
    assert np.all(np.asarray(first.base_durations) <= 1.5)
    assert np.max(np.linalg.norm(np.asarray(first.base_desired_velocities), axis=-1)) <= 1.25 + 1e-6
    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(left, right)


def test_descriptor_targets_have_physical_displacement_and_hover_tail() -> None:
    spec = build_shared_quad_library_spec(policy_count=16, code_size=4)
    targets = descriptor_targets_from_spec(spec)

    assert targets.shape == (16, 9)
    np.testing.assert_allclose(
        targets[:, :3], spec.base_desired_velocities * spec.base_durations[:, None]
    )
    np.testing.assert_array_equal(targets[:, 3:6], spec.base_desired_velocities)
    np.testing.assert_array_equal(targets[:, 6:9], jnp.zeros((16, 3)))


def test_dynamic_policy_slice_keeps_shared_network_and_selects_exact_slot() -> None:
    spec = build_shared_quad_library_spec(policy_count=16, code_size=4)
    config = SharedActorConfig(hidden_width=8)
    params = initialize_shared_actor(
        jax.random.key(2), spec, dimension=3, n_obstacles=2, config=config
    )
    selected_params, selected_spec = slice_shared_actor_policy(params, spec, jnp.array(11))

    np.testing.assert_array_equal(selected_spec.base_codes[0], spec.base_codes[11])
    np.testing.assert_array_equal(
        selected_spec.base_desired_velocities[0], spec.base_desired_velocities[11]
    )
    np.testing.assert_array_equal(selected_params.code_offsets[0], params.code_offsets[11])
    np.testing.assert_array_equal(selected_params.input_kernel, params.input_kernel)
    np.testing.assert_array_equal(selected_params.output_kernel, params.output_kernel)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"policy_count": 7}, "at least eight"),
        ({"structural_policy_count": 7}, "lie in"),
        ({"structural_policy_count": 65}, "lie in"),
        ({"minimum_speed": 2.0, "maximum_speed": 1.0}, "must not exceed"),
        ({"minimum_duration": 2.0, "maximum_duration": 1.0}, "must not exceed"),
    ],
)
def test_invalid_library_contract_is_rejected(arguments: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_shared_quad_library_spec(**arguments)
