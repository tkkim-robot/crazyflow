from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.drones import load_params
from crazyflow.safety.da_plcbf.actuator_compute import (
    MatchedEffortPlant,
    PackedActuatorController,
    PackedTreeSchema,
    _variable_effort_step,
)
from crazyflow.safety.da_plcbf.actuator_dynamics import (
    ActuatorModel,
    effort_lag_step,
    fit_native_time_constant,
    make_actuator_model,
)
from crazyflow.safety.da_plcbf.actuator_independent import ActuatorEvent, NumpyEffortPlant
from crazyflow.safety.da_plcbf.actuator_plcbf import (
    ActuatorFilterConfig,
    ActuatorRollouts,
    actuator_plcbf_step,
)
from crazyflow.safety.da_plcbf.continuous_version_a import RuntimeObstacleTrajectories
from crazyflow.safety.da_plcbf.version_a_barriers import RigidBodySafetySet, VersionAModel


class _NumericalAudit(NamedTuple):
    values: object
    flags: object
    counters: object
    nested: object


def _assert_bitwise_tree(actual: object, expected: object) -> None:
    actual_leaves, actual_tree = jax.tree.flatten(actual)
    expected_leaves, expected_tree = jax.tree.flatten(expected)
    assert actual_tree == expected_tree
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        left, right = np.asarray(actual_leaf), np.asarray(expected_leaf)
        assert left.shape == right.shape and left.dtype == right.dtype
        assert left.tobytes() == right.tobytes()


@pytest.mark.unit
def test_packed_mock_tree_preserves_bits_types_nonfinite_payloads_and_one_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nan_bits = jnp.array([0x7FC12345, 0xFF800000, 0x7F800000], dtype=jnp.uint32)
    payload = jax.lax.bitcast_convert_type(nan_bits, jnp.float32)

    @jax.jit
    def controller(x: jax.Array) -> _NumericalAudit:
        return _NumericalAudit(
            {"action": x, "nonfinite": payload, "empty": jnp.empty((0, 4), jnp.float32)},
            [jnp.array([True, False]), jnp.asarray(False)],
            (jnp.array([2**30 + 17, -(2**29)], jnp.int32), jnp.array([2**32 - 1], jnp.uint32)),
            {"omitted": None, "matrix": x.reshape(2, 2), "signed_zero": jnp.array([-0.0])},
        )

    command = jnp.array([0.03, 0.09, 0.13, 0.19])
    expected = jax.device_get(controller(command))
    packed = PackedActuatorController(controller)
    assert packed.warmup(command) > 0
    calls = []
    original = jax.device_get

    def counted(value: object) -> object:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(jax, "device_get", counted)
    actual = packed(command)
    assert len(calls) == 1
    assert len(calls[0]) == 4
    assert packed.last_buffer_count == 4
    assert packed.last_leaf_count == 9
    assert isinstance(actual, _NumericalAudit)
    assert isinstance(actual.flags, list) and isinstance(actual.counters, tuple)
    assert actual.nested["omitted"] is None
    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(actual))
    _assert_bitwise_tree(actual, expected)
    schema = packed.schemas[0]
    host_buffers = original(calls[0])
    unpacked = schema.unpack(host_buffers)
    assert np.shares_memory(unpacked.values["action"], host_buffers[0])


@pytest.mark.unit
def test_packing_keeps_int64_uint64_float64_complex_and_boolean_separate() -> None:
    with jax.enable_x64():

        @jax.jit
        def controller(x: jax.Array) -> dict:
            return {
                "large_ints": jnp.array([2**62 + 19, -(2**62) + 7], dtype=jnp.int64),
                "large_unsigned": jnp.array([2**63 + 17, 2**64 - 1], dtype=jnp.uint64),
                "precise": x,
                "complex": jnp.asarray([1.25 + 2.5j], dtype=jnp.complex128),
                "flag": jnp.asarray(True),
            }

        x = jnp.array([1 + 2**-45, -np.inf, np.nan], dtype=jnp.float64)
        expected = jax.device_get(controller(x))
        packed = PackedActuatorController(controller)
        actual = packed(x)
        assert packed.last_buffer_count == 5
        _assert_bitwise_tree(actual, expected)
        assert int(actual["large_ints"][0]) == 2**62 + 19
        assert int(actual["large_unsigned"][1]) == 2**64 - 1
        assert float(actual["precise"][0]) != 1.0


@pytest.mark.unit
def test_changed_input_shapes_and_dtypes_get_correct_cached_layouts() -> None:
    @jax.jit
    def controller(x: jax.Array) -> dict:
        return {"same": x, "transpose": x.T, "count": jnp.asarray(x.size, jnp.int32)}

    packed = PackedActuatorController(controller)
    for shape, dtype in (((2, 3), jnp.float32), ((4, 1), jnp.float32), ((2, 3), jnp.int32)):
        x = jnp.arange(np.prod(shape), dtype=dtype).reshape(shape)
        _assert_bitwise_tree(packed(x), jax.device_get(controller(x)))
    assert packed.cache_size == 3
    x = jnp.zeros((2, 3), jnp.float32)
    packed(x)
    assert packed.cache_size == 3
    packed.clear_cache()
    assert packed.cache_size == 0
    _assert_bitwise_tree(packed(x), jax.device_get(controller(x)))


@pytest.mark.unit
def test_packed_schema_rejects_wrong_transport_dtype_without_casting() -> None:
    schema = PackedTreeSchema.from_abstract_tree({"flag": jax.ShapeDtypeStruct((2,), jnp.bool_)})
    with pytest.raises(ValueError, match="dtype"):
        schema.unpack((np.array([1.0, 0.0], dtype=np.float32),))
    with pytest.raises(ValueError, match="count"):
        schema.unpack(())


@pytest.fixture(scope="module")
def model() -> ActuatorModel:
    params = load_params("cf21B_500")
    body = VersionAModel(
        jnp.asarray(params["mass"]),
        jnp.asarray(params["gravity_vec"]),
        jnp.asarray(params["J"]),
        jnp.asarray(np.linalg.inv(params["J"])),
        jnp.asarray(params["drag_matrix"]),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros(3),
    )
    return make_actuator_model(
        body,
        L=params["L"],
        thrust2torque=params["thrust2torque"],
        mixing_matrix=params["mixing_matrix"],
        thrust_min=params["thrust_min"],
        thrust_max=params["thrust_max"],
        time_constants=fit_native_time_constant()["nominal_tau"],
    )


def _state(model: ActuatorModel) -> np.ndarray:
    state = np.zeros(17)
    state[2], state[6] = 1.0, 1.0
    state[13:] = float(-model.body.mass * model.body.gravity_vec[2] / 4)
    return state


class _PreviousPerStepP0(NumpyEffortPlant):
    """Frozen pre-batching P0 launch/model-transfer behavior as a numerical comparison."""

    _compiled = staticmethod(jax.jit(_variable_effort_step))

    def _step(self, command: np.ndarray, dt: float) -> None:
        model = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), self._model)
        self._state = np.asarray(
            self._compiled(
                jnp.asarray(self._state, jnp.float32),
                jnp.asarray(command, jnp.float32),
                model,
                jnp.asarray(dt, jnp.float32),
            ),
            dtype=float,
        )


@pytest.mark.unit
def test_complete_filter_tree_has_bitwise_packed_parity(model: ActuatorModel) -> None:
    state = jnp.asarray(_state(model), dtype=jnp.float32)
    command = state[13:]
    config = ActuatorFilterConfig(horizon=4, command_hold_steps=2, held_substeps=1)
    obstacles = RuntimeObstacleTrajectories(
        jnp.zeros((5, 0, 3)), jnp.zeros(0), jnp.zeros((5, 0), dtype=bool)
    )
    safety = RigidBodySafetySet(
        jnp.zeros((0, 3)),
        jnp.zeros(0),
        jnp.zeros(0, dtype=bool),
        jnp.array([-5.0, -5.0, 0.1]),
        jnp.array([5.0, 5.0, 5.0]),
        jnp.asarray(4.0),
        jnp.asarray(10.0),
        jnp.asarray(1.0),
    )

    def controller(y: jax.Array, point_model: ActuatorModel) -> object:
        def rollouts(initial: jax.Array, current: ActuatorModel) -> ActuatorRollouts:
            def step(x: jax.Array, _: None) -> tuple:
                following = effort_lag_step(x, command, current, config.dt)
                return following, following

            _, future = jax.lax.scan(step, initial, None, length=config.horizon)
            return ActuatorRollouts(
                jnp.concatenate((initial[None], future))[None],
                jnp.repeat(command[None, None], config.horizon, axis=1),
                jnp.array([True]),
            )

        return actuator_plcbf_step(
            y, rollouts, point_model, obstacles, safety, command, jnp.asarray(0), config
        )

    compiled = jax.jit(controller)
    expected = jax.device_get(compiled(state, model))
    packed = PackedActuatorController(compiled)
    packed.warmup(state, model)
    actual = packed(state, model)
    _assert_bitwise_tree(actual, expected)
    assert actual.qp_valid
    assert packed.last_leaf_count > 70
    assert packed.last_buffer_count <= 4
    assert type(actual) is type(expected)
    assert type(actual.certificates) is type(expected.certificates)


@pytest.mark.unit
@pytest.mark.parametrize(
    "max_step,durations", [(0.005, [0.04, 0.04, 0.017]), (0.003, [0.017, 0.043])]
)
def test_batched_p0_matches_old_node_grid_and_numerical_order(
    model: ActuatorModel, max_step: float, durations: list[float]
) -> None:
    initial = _state(model)
    initial[7:10] = [0.2, -0.1, 0.04]
    initial[10:13] = [0.1, -0.05, 0.02]
    old = _PreviousPerStepP0(initial, model, max_step=max_step)
    batched = MatchedEffortPlant(initial, model, max_step=max_step)
    for index, duration in enumerate(durations):
        command = np.array([0.12, 0.11, 0.13, 0.10]) + 0.003 * index
        expected, actual = old.advance(command, duration), batched.advance(command, duration)
        np.testing.assert_array_equal(actual.times, expected.times)
        np.testing.assert_allclose(actual.states, expected.states, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(
            actual.actual_forces, expected.actual_forces, atol=2e-8, rtol=2e-6
        )
        np.testing.assert_allclose(actual.wrenches, expected.wrenches, atol=1e-7, rtol=2e-6)
        np.testing.assert_array_equal(actual.states, actual.native_states)
        assert batched.last_advance_segments == 1
        assert batched.last_advance_steps == len(actual.times) - 1
    assert batched.model_transfer_count == 1


@pytest.mark.unit
def test_p0_events_split_scans_preserve_state_and_report_postevent_forces(
    model: ActuatorModel,
) -> None:
    damaged = model._replace(effectiveness=jnp.array([0.7, 0.85, 1.0, 0.85]))
    slow = damaged._replace(time_constants=3 * model.time_constants)
    events = (
        ActuatorEvent(0.013, damaged),
        ActuatorEvent(0.037, slow),
        ActuatorEvent(0.073, model),
    )
    old = _PreviousPerStepP0(_state(model), model, events=events, max_step=0.005)
    batched = MatchedEffortPlant(_state(model), model, events=events, max_step=0.005)
    before = batched.model_snapshot()
    np.testing.assert_array_equal(before.effectiveness, model.effectiveness)
    expected = old.advance(np.full(4, 0.12), 0.081)
    actual = batched.advance(np.full(4, 0.12), 0.081)
    np.testing.assert_array_equal(actual.times, expected.times)
    np.testing.assert_allclose(actual.states, expected.states, atol=3e-6, rtol=3e-6)
    for event in events:
        index = np.flatnonzero(actual.times == event.time).item()
        np.testing.assert_allclose(
            actual.actual_forces[index],
            actual.states[index, 13:] * event.model.effectiveness,
            atol=2e-8,
        )
    assert batched.model_transfer_count == 4
    assert batched.last_advance_segments == 4
    np.testing.assert_array_equal(batched.model_snapshot().effectiveness, model.effectiveness)
    np.testing.assert_array_equal(before.effectiveness, model.effectiveness)


@pytest.mark.unit
def test_p0_warmup_is_disposable_and_shapes_compile_without_reset(model: ActuatorModel) -> None:
    damaged = model._replace(effectiveness=jnp.array([0.7, 1.0, 1.0, 1.0]))
    plant = MatchedEffortPlant(
        _state(model), model, events=(ActuatorEvent(0.013, damaged),), max_step=0.005
    )
    before = plant.observe()
    assert plant.warmup(np.full(4, 0.12), durations=(0.04,), substep_counts=(1, 2, 3)) > 0
    np.testing.assert_array_equal(plant.observe(), before)
    np.testing.assert_array_equal(plant.native_state, before)
    assert plant.time == 0 and plant.model_transfer_count == 1
    assert {1, 2, 3}.issubset(plant.seen_step_counts)
    first = plant.advance(np.full(4, 0.12), 0.013)
    np.testing.assert_array_equal(first.states[0], before)
    assert plant.time == 0.013 and plant.model_transfer_count == 2
    second = plant.advance(np.full(4, 0.11), 0.004)
    np.testing.assert_array_equal(second.states[0], first.states[-1])
    assert plant.time == 0.017


@pytest.mark.unit
def test_p0_strict_command_bounds_and_fault_initialization_have_no_hidden_clipping(
    model: ActuatorModel,
) -> None:
    plant = MatchedEffortPlant(_state(model), model, max_step=0.005)
    for command in (np.full(4, 0.3), np.full(4, -0.1), np.full(4, np.nan)):
        with pytest.raises(ValueError):
            plant.advance(command, 0.04)
        assert plant.time == 0
    with pytest.raises(ValueError, match="positive"):
        plant.advance(np.full(4, 0.12), 0)
    damaged = model._replace(effectiveness=jnp.array([0.7, 1.0, 1.0, 1.0]))
    damaged_initial = MatchedEffortPlant(
        _state(model), model, events=(ActuatorEvent(0.0, damaged),), max_step=0.005
    )
    np.testing.assert_array_equal(damaged_initial.observe(), _state(model))
    trace = damaged_initial.advance(np.full(4, 0.12), 0.005)
    np.testing.assert_allclose(trace.actual_forces[0], _state(model)[13:] * damaged.effectiveness)


@pytest.mark.unit
@pytest.mark.parametrize("use_precheck", [False, True])
def test_audited_batch_commits_contact_prefix_and_discards_later_events(
    model: ActuatorModel, use_precheck: bool
) -> None:
    damaged = model._replace(effectiveness=jnp.array([0.7, 0.85, 1.0, 1.0]))
    events = (ActuatorEvent(0.013, damaged), ActuatorEvent(0.027, model))
    plant = MatchedEffortPlant(_state(model), model, events=events, max_step=0.005)
    calls = []

    def stop(times: np.ndarray, states: np.ndarray) -> bool:
        calls.append(times.copy())
        assert states.shape == (2, 17)
        return bool(times[-1] >= 0.018 - 1e-12)

    trace = plant.advance_audited(
        np.full(4, 0.12),
        0.04,
        stop,
        separation_precheck=(lambda times, states: False) if use_precheck else None,
    )
    assert trace.times[-1] == plant.time
    assert abs(plant.time - 0.018) < 1e-12
    assert all(times[-1] <= 0.018 + 1e-12 for times in calls)
    np.testing.assert_array_equal(plant.observe(), trace.states[-1])
    np.testing.assert_array_equal(plant.model_snapshot().effectiveness, damaged.effectiveness)
    assert plant._event_index == 1
    # A continued test call proves that both the device state and future-event
    # cursor were restored, rather than only changing the reported host pose.
    continuation = plant.advance(np.full(4, 0.12), 0.04 - plant.time)
    original = _PreviousPerStepP0(_state(model), model, events=events, max_step=0.005)
    expected = original.advance(np.full(4, 0.12), 0.04)
    np.testing.assert_allclose(continuation.states[-1], expected.states[-1], atol=3e-6, rtol=3e-6)
    np.testing.assert_array_equal(plant.model_snapshot().effectiveness, model.effectiveness)


@pytest.mark.unit
@pytest.mark.parametrize("precheck_failure", [False, True])
def test_audit_exception_restores_physical_state_and_cache(
    model: ActuatorModel, precheck_failure: bool
) -> None:
    plant = MatchedEffortPlant(_state(model), model, max_step=0.005)
    before = plant.observe()

    def fail(times: np.ndarray, states: np.ndarray) -> bool:
        raise RuntimeError("injected collider audit failure")

    with pytest.raises(RuntimeError, match="injected collider"):
        plant.advance_audited(
            np.full(4, 0.12), 0.04, fail, separation_precheck=fail if precheck_failure else None
        )
    assert plant.time == 0
    np.testing.assert_array_equal(plant.observe(), before)
    correct = MatchedEffortPlant(before, model, max_step=0.005)
    np.testing.assert_array_equal(
        plant.advance(np.full(4, 0.12), 0.01).states, correct.advance(np.full(4, 0.12), 0.01).states
    )


@pytest.mark.unit
def test_separation_precheck_commits_complete_trace_without_adjacent_audits(
    model: ActuatorModel,
) -> None:
    damaged = model._replace(effectiveness=jnp.array([0.7, 0.85, 1.0, 1.0]))
    events = (ActuatorEvent(0.013, damaged), ActuatorEvent(0.027, model))
    plant = MatchedEffortPlant(_state(model), model, events=events, max_step=0.005)
    reference = MatchedEffortPlant(_state(model), model, events=events, max_step=0.005)
    expected = reference.advance(np.full(4, 0.12), 0.04)
    prechecks = []

    def separated(times: np.ndarray, states: np.ndarray) -> bool:
        prechecks.append(times.copy())
        np.testing.assert_array_equal(times, expected.times)
        np.testing.assert_array_equal(states, expected.states)
        return True

    def should_skip(times: np.ndarray, states: np.ndarray) -> bool:
        raise AssertionError("a certified separated hold needs no adjacent-node audits")

    actual = plant.advance_audited(
        np.full(4, 0.12), 0.04, should_skip, separation_precheck=separated
    )
    assert len(prechecks) == 1 and plant.time == 0.04
    for observed, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed, wanted)
    np.testing.assert_array_equal(plant.observe(), expected.states[-1])
    assert plant._event_index == 2
