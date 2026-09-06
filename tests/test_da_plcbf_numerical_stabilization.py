"""Executed-precision feasibility for the recorded large policy row."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.actuator_dynamics import ActuatorModel, make_actuator_model
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig, _normalized_qp_with_audit
from crazyflow.safety.da_plcbf.online_constant_wind import build_cf21b_version_a_resources


def _model() -> ActuatorModel:
    resources = build_cf21b_version_a_resources()
    actuator = resources.actuator
    return make_actuator_model(
        resources.model,
        L=float(actuator.arm_length),
        thrust2torque=float(actuator.thrust_to_torque),
        mixing_matrix=actuator.mixing_matrix,
        thrust_min=actuator.thrust_min,
        thrust_max=actuator.thrust_max,
        time_constants=(0.06, 0.06, 0.06, 0.06),
    )


@pytest.mark.parametrize("perturbation", [-1e-5, 0.0, 1e-5])
def test_recorded_large_row_rechecks_executable_precision(perturbation: float) -> None:
    model = _model()
    row = jnp.array(
        [156.4965362548828, 218.12234497070312, -108.88327026367188, -147.86111450195312],
        dtype=jnp.float32,
    )
    bound = jnp.float32(40.992610931396484)
    nominal = jnp.array([0.19, 0.19, 0.05, 0.05], dtype=jnp.float32) + perturbation
    solve = jax.jit(
        lambda command: _normalized_qp_with_audit(
            command, row, bound, model, ActuatorFilterConfig()
        )
    )
    result, audit = solve(nominal)
    assert bool(result.feasible)
    assert bool(audit.executed_rows_passed)
    assert float(bound - row @ result.action) >= -2e-6
    assert np.all(np.asarray(result.action) >= np.asarray(model.command_lower))
    assert np.all(np.asarray(result.action) <= np.asarray(model.command_upper))
    assert float(audit.normalized_solver_tolerance * jnp.max(audit.row_scales)) <= 2.000001e-6
    assert float(audit.inward_margins[-1]) > 0
    np.testing.assert_array_equal(result.action, audit.executed_command)


def test_inward_solve_preserves_operational_and_trust_faces():
    model = _model()
    row = jnp.array([156.4965, 218.1223, -108.8833, -147.8611])
    nominal = jnp.array([0.19, 0.19, 0.05, 0.05])
    operational = jnp.eye(4)[:1]
    config = replace(ActuatorFilterConfig(), command_trust_fraction=0.4)
    reference = jnp.full(4, 0.12)
    result, audit = jax.jit(
        lambda: _normalized_qp_with_audit(
            nominal,
            row,
            jnp.float32(40.99261),
            model,
            config,
            operational,
            jnp.array([0.1]),
            reference,
        )
    )()
    assert bool(result.feasible)
    assert bool(audit.executed_rows_passed)
    assert float(result.action[0]) <= 0.1 + 2e-6
    assert np.all(
        np.abs(np.asarray(result.action - reference))
        <= np.asarray(model.command_upper - model.command_lower) * 0.4 + 2e-6
    )


def test_infeasible_physical_row_is_not_relaxed():
    model = _model()
    result, audit = jax.jit(
        lambda: _normalized_qp_with_audit(
            jnp.full(4, 0.1), jnp.ones(4), jnp.float32(-1), model, ActuatorFilterConfig()
        )
    )()
    assert not bool(result.feasible)
    assert not bool(audit.executed_rows_passed)


def test_hold_timing_does_not_confuse_service_deadline_with_certified_interval() -> None:
    from crazyflow.safety.da_plcbf.actuator_experiment import audit_applied_hold_timing

    controls = [{"control_index": 0, "time": 0.0}, {"control_index": 1, "time": 0.04}]
    aligned = [{"time": 0.0, "control_index": 0}, {"time": 0.04, "control_index": 1}]
    result = audit_applied_hold_timing(aligned, controls, 0.08, 0.04)
    assert result["online_held_check_timing_all_covered"]
    delayed = [{"time": 0.02, "control_index": 0}, {"time": 0.075, "control_index": 1}]
    result = audit_applied_hold_timing(delayed, controls, 0.10, 0.04)
    assert result["online_held_check_uncovered_intervals"] == 2
    assert result["maximum_actual_command_hold_seconds"] == pytest.approx(0.055)
    assert not result["online_held_check_timing_all_covered"]
