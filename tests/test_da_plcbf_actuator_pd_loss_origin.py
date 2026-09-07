"""NumPy-only checks for exact representation and temporal rollout comparisons."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark import da_plcbf_actuator_pd_loss_origin as origin


def test_signed_zero_changes_bytes_without_a_numeric_delta() -> None:
    result = origin.compare_arrays(
        np.asarray([1.0, -0.0], dtype=np.float32), np.asarray([1.0, 0.0], dtype=np.float32)
    )
    assert result["dtype_equal"] and result["finite"]
    assert not result["byte_equal"]
    assert result["first_differing_index"] == [1]
    assert result["l2_delta"] == result["max_absolute_delta"] == 0.0
    scalar = origin.compare_arrays(np.float32(-0.0), np.float32(0.0))
    assert scalar["shape"] == []
    assert scalar["first_differing_index"] == []
    assert not scalar["byte_equal"]


def test_equal_values_with_different_dtypes_are_representation_changes() -> None:
    result = origin.compare_arrays(
        np.asarray([1.0, 2.0], dtype=np.float32), np.asarray([1.0, 2.0], dtype=np.float64)
    )
    assert result["shape"] == [2]
    assert (result["left_dtype"], result["right_dtype"]) == ("float32", "float64")
    assert not result["dtype_equal"] and not result["byte_equal"]
    assert result["first_differing_index"] == [0]
    assert result["l2_delta"] == result["max_absolute_delta"] == 0.0


def test_comparisons_reject_mismatched_or_invalid_rollout_shapes() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        origin.compare_arrays(np.zeros(2), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="matching.*shapes"):
        origin.compare_rollout_states(np.zeros((3, 2, 4, 17)), np.zeros((3, 2, 5, 17)))
    with pytest.raises(ValueError, match="matching.*shapes"):
        origin.compare_rollout_states(np.zeros((2, 2, 4, 17)), np.zeros((2, 2, 4, 17)))


def test_first_differing_node_uses_time_across_skills_and_counts_signed_zero() -> None:
    reference = np.zeros((3, 2, 5, 17), dtype=np.float32)
    student = reference.copy()
    student[0, 0, 4, 0] = 0.25
    student[0, 1, 1, 16] = -0.5
    student[1, 1, 0, 3] = -0.0
    current, anchor0, anchor1 = origin.compare_rollout_states(student, reference)

    assert current["case"] == "current"
    assert current["comparison"]["first_differing_index"] == [0, 4, 0]
    assert current["first_differing_node"] == 1
    expected = np.zeros(17)
    expected[0], expected[16] = 0.25, 0.5
    np.testing.assert_array_equal(current["per_component_max_absolute_delta"], expected)
    assert anchor0["case"] == "anchor0"
    assert anchor0["first_differing_node"] == 0
    assert not anchor0["comparison"]["byte_equal"]
    np.testing.assert_array_equal(anchor0["per_component_max_absolute_delta"], np.zeros(17))
    assert anchor1["case"] == "anchor1"
    assert anchor1["first_differing_node"] is None


def test_identical_rollouts_report_no_difference_for_each_case() -> None:
    states = np.arange(3 * 2 * 4 * 17, dtype=np.float32).reshape(3, 2, 4, 17)
    rows = origin.compare_rollout_states(states, states.copy())
    assert [row["case"] for row in rows] == ["current", "anchor0", "anchor1"]
    for row in rows:
        comparison = row["comparison"]
        assert comparison["shape"] == [2, 4, 17]
        assert comparison["finite"] and comparison["byte_equal"] and comparison["dtype_equal"]
        assert comparison["first_differing_index"] is None
        assert comparison["l2_delta"] == comparison["max_absolute_delta"] == 0.0
        assert row["first_differing_node"] is None
        np.testing.assert_array_equal(row["per_component_max_absolute_delta"], np.zeros(17))
