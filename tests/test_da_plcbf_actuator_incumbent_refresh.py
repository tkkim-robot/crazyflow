from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmark.da_plcbf_actuator_incumbent_refresh import IncumbentRefreshController, _write

if TYPE_CHECKING:
    from pathlib import Path


def _fingerprint(value: object) -> str:
    array = np.asarray(value)
    return hashlib.sha256(array.dtype.str.encode() + array.tobytes()).hexdigest()


def _delegate(*args: object) -> SimpleNamespace:
    return SimpleNamespace(selected_index=np.asarray(args[5]), action=np.zeros(4))


def _args(params: np.ndarray, previous: int) -> tuple:
    return (np.zeros(17), params, None, None, None, np.asarray(previous, dtype=np.int32), None)


@pytest.mark.parametrize(
    ("core", "previous", "effective"),
    [
        (16, 0, 0),
        (16, 1, 1),
        (16, 16, 16),
        (16, 17, -1),
        (16, 32, -1),
        (0, 0, 0),
        (0, 1, -1),
        (0, 16, -1),
        (16, -1, -1),
        (16, 33, 33),
    ],
)
def test_only_existing_adaptive_incumbents_refresh(
    core: int, previous: int, effective: int
) -> None:
    wrapper = IncumbentRefreshController(
        _delegate, immutable_core_count=core, fingerprint=_fingerprint
    )
    wrapper(*_args(np.array([0.0]), 0))
    result = wrapper(*_args(np.array([1.0]), previous))
    assert int(result.selected_index) == effective
    assert wrapper.records[-1]["refreshed"] is (effective != previous)


def test_new_object_identical_bytes_does_not_refresh_and_changed_bytes_do() -> None:
    wrapper = IncumbentRefreshController(
        _delegate, immutable_core_count=0, fingerprint=_fingerprint
    )
    first = np.array([1.0], dtype=np.float32)
    wrapper(*_args(first, 1))
    wrapper(*_args(first.copy(), 1))
    assert wrapper.records[-1]["refreshed"] is False
    following = np.nextafter(first, np.float32(2.0))
    wrapper(*_args(following, 1))
    assert wrapper.records[-1]["refreshed"] is True
    wrapper(*_args(following, 1))
    assert wrapper.records[-1]["refreshed"] is False


def test_core_observation_consumes_changed_fingerprint_without_future_false_reset() -> None:
    wrapper = IncumbentRefreshController(
        _delegate, immutable_core_count=16, fingerprint=_fingerprint
    )
    wrapper(*_args(np.array([0.0]), 0))
    wrapper(*_args(np.array([1.0]), 12))
    assert wrapper.records[-1]["parameters_changed"] is True
    assert wrapper.records[-1]["refreshed"] is False
    wrapper(*_args(np.array([1.0]), 28))
    assert wrapper.records[-1]["refreshed"] is False


def test_wrapper_state_does_not_leak_between_episodes_sharing_the_delegate() -> None:
    first = IncumbentRefreshController(
        _delegate, immutable_core_count=0, fingerprint=_fingerprint, warmup_calls=1
    )
    second = IncumbentRefreshController(
        _delegate, immutable_core_count=0, fingerprint=_fingerprint, warmup_calls=1
    )
    first(*_args(np.array([0.0]), 0))
    first(*_args(np.array([1.0]), 1))
    second(*_args(np.array([1.0]), 0))
    assert len(first.records) == 2 and len(second.records) == 1
    assert first.records[0]["warmup"] is True
    assert first.records[1]["warmup"] is False
    assert second.records[0]["preceding_call_parameter_sha256"] is None


def test_wrapper_preserves_all_other_inputs_and_original_unmodified_index_object() -> None:
    seen = []

    def delegate(*args: object) -> SimpleNamespace:
        seen.append(args)
        return _delegate(*args)

    wrapper = IncumbentRefreshController(
        delegate, immutable_core_count=16, fingerprint=_fingerprint
    )
    args = _args(np.array([0.0]), 12)
    wrapper(*args)
    assert all(actual is expected for actual, expected in zip(seen[0], args, strict=True))
    changed = _args(np.array([1.0]), 28)
    wrapper(*changed)
    assert all(seen[1][i] is changed[i] for i in (0, 1, 2, 3, 4, 6))
    assert seen[1][5].dtype == changed[5].dtype


def test_negative_previous_index_disables_only_incumbency_under_actual_selector() -> None:
    import jax.numpy as jnp

    from crazyflow.safety.da_plcbf.selector import SelectionConfig, select_hard_policy

    values = jnp.array([0.1, 0.1, 0.1])
    scores = jnp.array([0.2, 0.75, 0.745])
    config = SelectionConfig(switch_score_margin=0.02)
    retained = select_hard_policy(values, scores, jnp.asarray(2), config)
    refreshed = select_hard_policy(values, scores, jnp.asarray(-1), config)
    assert int(retained.selected_index) == 2
    assert bool(retained.retained_by_hysteresis)
    assert int(refreshed.selected_index) == 1
    assert not bool(refreshed.previous_index_valid)
    preferred = select_hard_policy(
        values, scores, jnp.asarray(-1), SelectionConfig(prefer_first_eligible=True)
    )
    assert int(preferred.selected_index) == 0


def test_refresh_preserves_jax_index_dtype_and_weak_type() -> None:
    import jax.numpy as jnp

    received = []

    def delegate(*args: object) -> SimpleNamespace:
        received.append(args[5])
        return _delegate(*args)

    wrapper = IncumbentRefreshController(delegate, immutable_core_count=0, fingerprint=_fingerprint)
    wrapper(*_args(np.array([0.0]), 0))
    args = list(_args(np.array([1.0]), 1))
    args[5] = jnp.asarray(1)
    wrapper(*args)
    assert received[-1].dtype == args[5].dtype
    assert received[-1].weak_type == args[5].weak_type
    assert int(received[-1]) == -1


@pytest.mark.parametrize("invalid", [np.uint32(1), np.array([1]), np.array(1.0)])
def test_rejects_unrepresentable_or_nonscalar_previous_indices(invalid: object) -> None:
    wrapper = IncumbentRefreshController(
        _delegate, immutable_core_count=0, fingerprint=_fingerprint
    )
    args = list(_args(np.array([0.0]), 0))
    args[5] = invalid
    with pytest.raises(ValueError, match="signed integer scalar"):
        wrapper(*args)


def test_record_serialization_accepts_real_summary_arrays_and_refuses_partial_invalid_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "record.json"
    _write(
        path,
        {
            "summary": {
                "margins": np.array([0.1, 0.2]),
                "count": np.int32(3),
                "artifact": tmp_path / "application.npz",
                "unavailable": float("nan"),
            }
        },
    )
    assert json.loads(path.read_text()) == {
        "summary": {
            "margins": [0.1, 0.2],
            "count": 3,
            "artifact": str(tmp_path / "application.npz"),
            "unavailable": None,
        }
    }
    invalid = tmp_path / "invalid.json"
    with pytest.raises(TypeError):
        _write(invalid, {"unserializable": object()})
    assert not invalid.exists()
