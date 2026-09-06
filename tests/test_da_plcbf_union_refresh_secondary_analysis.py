from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from benchmark.da_plcbf_actuator_diagnostic_protocol import digest
from benchmark.da_plcbf_union_refresh_secondary_analysis import (
    verify_previous_indices,
    wrapper_prefix,
)


def history() -> list[dict]:
    return [
        {
            "warmup": True,
            "call_index": 0,
            "parameter_sha256": "initial",
            "wrapper_through_delegate_seconds": 0.1,
        },
        {
            "warmup": False,
            "call_index": 1,
            "time_seconds": 0.0,
            "preceding_call_parameter_sha256": "initial",
            "requested_previous_index": 0,
            "selected_index": 17,
            "effective_previous_index": 0,
            "parameter_sha256": "first",
            "fingerprint_and_dispatch_seconds": 0.01,
        },
        {
            "warmup": False,
            "call_index": 2,
            "time_seconds": 2.0,
            "preceding_call_parameter_sha256": "first",
            "requested_previous_index": 17,
            "selected_index": 12,
            "effective_previous_index": -1,
            "parameter_sha256": "second",
            "delegated_controller_seconds": 0.02,
        },
        {
            "warmup": False,
            "call_index": 3,
            "time_seconds": 2.04,
            "preceding_call_parameter_sha256": "second",
            "requested_previous_index": 12,
            "selected_index": 12,
            "effective_previous_index": 12,
            "parameter_sha256": "third",
            "delegated_controller_seconds": 0.02,
        },
    ]


def test_wrapper_prefix_ignores_service_cost_and_retains_event_input_memory():
    first, second = history(), history()
    second[0]["wrapper_through_delegate_seconds"] = 1000
    second[1]["fingerprint_and_dispatch_seconds"] = 2000
    second[3]["selected_index"] = 31
    assert digest(wrapper_prefix(first, 2.0)) == digest(wrapper_prefix(second, 2.0))
    second[2]["preceding_call_parameter_sha256"] = "unmatched"
    assert digest(wrapper_prefix(first, 2.0)) != digest(wrapper_prefix(second, 2.0))


def test_identical_actual_actions_cannot_hide_a_different_effective_incumbent_history():
    first, second = history(), history()
    second[1]["effective_previous_index"] = -1
    assert digest(wrapper_prefix(first, 2.0)) != digest(wrapper_prefix(second, 2.0))


def test_wrapper_prefix_requires_the_actual_event_boundary():
    calls = history()
    calls[2]["time_seconds"] = 2.0001
    with pytest.raises(ValueError, match="exact actual event"):
        wrapper_prefix(calls, 2.0)


def test_raw_incumbency_follows_only_actually_applied_selections():
    calls = history()
    selected = np.array([17, 12, 12])
    verify_previous_indices(calls, selected, np.array([True, True, True]))
    changed = deepcopy(calls)
    changed[3]["requested_previous_index"] = 17
    verify_previous_indices(changed, selected, np.array([True, False, True]))
    with pytest.raises(ValueError, match="applied selection memory"):
        verify_previous_indices(changed, selected, np.array([True, True, True]))


def test_wrapper_output_index_must_equal_retained_control_index():
    with pytest.raises(ValueError, match="output index"):
        verify_previous_indices(history(), np.array([18, 12, 12]), np.array([True, True, True]))
