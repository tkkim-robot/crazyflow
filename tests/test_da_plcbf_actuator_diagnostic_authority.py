"""Coupled trim/axis bounds and independent memory-preserving authority diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.da_plcbf_actuator_diagnostic_authority import (
    coupled_hover_authority,
    prescribed_vertical_commands,
    replay,
    rest_state,
)
from crazyflow.safety.da_plcbf.actuator_dynamics import ActuatorModel
from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel


def model() -> ActuatorModel:
    """A complete dimensional quadrotor with physical motor bounds and two lag scales."""
    inertia = np.diag([2.5e-5, 2.8e-5, 4.9e-5])
    body = VersionAModel(
        np.asarray(0.04),
        np.array([0.0, 0.0, -9.81]),
        inertia,
        np.linalg.inv(inertia),
        np.diag([-0.02, -0.02, -0.024]),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
    )
    mapping = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [-0.035, -0.035, 0.035, 0.035],
            [-0.035, 0.035, 0.035, -0.035],
            [-0.006, 0.006, -0.006, 0.006],
        ]
    )
    return ActuatorModel(
        body, np.ones(4), np.full(4, 0.06), np.full(4, 0.01), np.full(4, 0.2), mapping
    )


@pytest.mark.parametrize("eta", [1.0, 0.85, 0.7])
def test_trim_and_each_axis_section_satisfy_coupled_wrench_and_motor_limits(eta: float) -> None:
    current = model()._replace(effectiveness=np.array([eta, eta, 1.0, 1.0]))
    result = coupled_hover_authority(current)
    assert result["feasible_hover_trim"]
    trim = np.asarray(result["required_command_N"])
    mapping = np.asarray(result["coupled_mapping"])
    target = np.asarray(result["hover_target_wrench"])
    np.testing.assert_allclose(mapping @ trim, target, atol=1e-15)
    np.testing.assert_allclose(current.effectiveness * trim, target[0] / 4, atol=1e-15)
    assert result["torque_free_collective_upper_over_weight"] == pytest.approx(
        4 * 0.2 * eta / target[0]
    )
    for axis, section in enumerate(result["individual_wrench_axis_sections"].values()):
        for side in ("negative", "positive"):
            command = np.asarray(section[f"command_at_{side}_endpoint_N"])
            assert np.all(command >= current.command_lower - 1e-14)
            assert np.all(command <= current.command_upper + 1e-14)
            assert np.min(
                np.minimum(command - current.command_lower, current.command_upper - command)
            ) == pytest.approx(0, abs=1e-14)
            expected = target.copy()
            expected[axis] += section[f"{side}_delta"]
            np.testing.assert_allclose(mapping @ command, expected, atol=1e-14)


def test_fault_replay_preserves_motor_memory_and_static_trim_is_separate() -> None:
    nominal = model()
    current = nominal._replace(effectiveness=np.array([0.7, 0.7, 1.0, 1.0]))
    initial = rest_state(
        nominal, np.asarray(coupled_hover_authority(nominal)["required_command_N"])
    )
    trim = np.asarray(coupled_hover_authority(current)["required_command_N"])
    fault = replay(current, initial, trim[None], 0.04, 0.002, fault_at_zero_from=nominal)
    np.testing.assert_array_equal(fault["states"][0], initial)
    np.testing.assert_array_equal(fault["actual_forces"][0], current.effectiveness * initial[13:])
    assert np.linalg.norm(fault["states"][-1, 10:13]) > 0.1
    settled = rest_state(current, trim)
    static = replay(current, settled, trim[None], 0.04, 0.002)
    np.testing.assert_allclose(
        static["states"], np.broadcast_to(settled, static["states"].shape), atol=1e-12
    )


def test_prescribed_profile_has_physical_held_commands_and_exact_motor_endpoints() -> None:
    current = model()._replace(
        effectiveness=np.array([0.7, 0.7, 1.0, 1.0]),
        time_constants=np.array([0.12, 0.12, 0.06, 0.06]),
    )
    times, commands, target = prescribed_vertical_commands(current)
    assert commands.shape == (30, 4)
    np.testing.assert_allclose(np.diff(times), 0.04, atol=1e-15)
    assert np.all(commands >= current.command_lower)
    assert np.all(commands <= current.command_upper)
    motor = np.asarray(coupled_hover_authority(current)["required_command_N"])
    decay = np.exp(-0.04 / current.time_constants)
    for index, command in enumerate(commands):
        motor = command + (motor - command) * decay
        np.testing.assert_allclose(
            current.force_to_wrench @ (current.effectiveness * motor),
            [target[index + 1, 3], 0, 0, 0],
            atol=1e-14,
        )
    assert target[-1, 1] == pytest.approx(0.0, abs=1e-15)
    assert target[-1, 0] - target[0, 0] == pytest.approx(0.5 * 1.2**2 / (2 * np.pi))


def test_failed_trim_does_not_create_a_general_infeasibility_result() -> None:
    current = model()._replace(effectiveness=np.array([0.1, 0.1, 1.0, 1.0]))
    result = coupled_hover_authority(current)
    assert not result["feasible_hover_trim"]
    assert result["minimum_command_margin_N"] < 0
    assert result["individual_wrench_axis_sections"] == {}
    assert "not proof" in result["failure_scope"]
