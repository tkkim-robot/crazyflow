"""Actuator-study execution transport and matched-plant batching without new physics.

The packed controller retains every leaf, dtype, flag and nonfinite payload from
its supplied controller. P0 batches the existing exact-effort/RK4 predictor stages
between physical events. Neither optimization changes a command, a safety row,
an integration node, or the independent P1/P2 models.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.actuator_dynamics import (
    ActuatorModel,
    _body_rk4,
    applied_wrench,
    exact_effort_update,
)
from crazyflow.safety.da_plcbf.actuator_independent import (
    IndependentPlantTrace,
    NumpyEffortPlant,
    _array,
    _validate_step,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike

    from crazyflow.safety.da_plcbf.actuator_independent import ActuatorEvent


@dataclass(frozen=True, slots=True)
class PackedLeaf:
    """Exact output-leaf location within one homogeneous transport buffer."""

    shape: tuple[int, ...]
    dtype: np.dtype
    buffer_index: int
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class PackedTreeSchema:
    """Original pytree topology and exact homogeneous-buffer layout."""

    tree_definition: Any
    leaves: tuple[PackedLeaf, ...]
    buffer_dtypes: tuple[np.dtype, ...]
    buffer_sizes: tuple[int, ...]

    @classmethod
    def from_abstract_tree(cls, tree: Any) -> PackedTreeSchema:
        """Describe JAX's abstract output without evaluating or inspecting its quality."""
        abstract_leaves, definition = jax.tree.flatten(tree)
        groups: dict[np.dtype, int] = {}
        dtypes: list[np.dtype] = []
        sizes: list[int] = []
        descriptors = []
        for leaf in abstract_leaves:
            dtype = np.dtype(leaf.dtype)
            if dtype not in groups:
                groups[dtype] = len(groups)
                dtypes.append(dtype)
                sizes.append(0)
            group = groups[dtype]
            shape = tuple(leaf.shape)
            size = math.prod(shape)
            descriptors.append(PackedLeaf(shape, dtype, group, sizes[group], size))
            sizes[group] += size
        return cls(definition, tuple(descriptors), tuple(dtypes), tuple(sizes))

    def pack(self, tree: Any) -> tuple[jax.Array, ...]:
        """Concatenate leaves by exact dtype without a cross-dtype conversion."""
        leaves, definition = jax.tree.flatten(tree)
        if definition != self.tree_definition or len(leaves) != len(self.leaves):
            raise ValueError("controller output topology changed for a cached input signature")
        groups: list[list[jax.Array]] = [[] for _ in self.buffer_dtypes]
        for leaf, descriptor in zip(leaves, self.leaves, strict=True):
            if tuple(leaf.shape) != descriptor.shape or np.dtype(leaf.dtype) != descriptor.dtype:
                raise ValueError("controller output shape or dtype changed for a cached signature")
            groups[descriptor.buffer_index].append(jnp.reshape(leaf, (-1,)))
        return tuple(jnp.concatenate(values) for values in groups)

    def unpack(self, buffers: tuple[np.ndarray, ...]) -> Any:
        """Rebuild the original NamedTuple/dict/list structure using NumPy views."""
        if len(buffers) != len(self.buffer_dtypes):
            raise ValueError("packed transport buffer count does not match the cached schema")
        arrays = tuple(np.asarray(buffer) for buffer in buffers)
        for buffer, dtype, size in zip(arrays, self.buffer_dtypes, self.buffer_sizes, strict=True):
            if buffer.dtype != dtype or buffer.shape != (size,):
                raise ValueError("packed transport buffer shape or dtype does not match schema")
        leaves = [
            arrays[descriptor.buffer_index][
                descriptor.offset : descriptor.offset + descriptor.size
            ].reshape(descriptor.shape)
            for descriptor in self.leaves
        ]
        return jax.tree.unflatten(self.tree_definition, leaves)


@dataclass(frozen=True, slots=True)
class _PackedEntry:
    schema: PackedTreeSchema
    function: Any


class PackedActuatorController:
    """Callable wrapper returning the complete original controller tree on the host.

    One JIT call packs its numerical result and one ``jax.device_get`` transfers
    the tuple of homogeneous buffers. The host performs only slicing, reshaping
    and pytree reconstruction. ``warmup`` is explicit and includes schema tracing,
    compilation, transport and unpacking; normal first calls also work and incur
    those same costs honestly. Different input shapes/dtypes get separate schemas.
    No plan, policy, optimizer state or acceptance decision is retained here.
    """

    def __init__(self, controller: Callable) -> None:
        self.controller = controller
        self._entries: dict[tuple, _PackedEntry] = {}
        self.last_buffer_count = 0
        self.last_leaf_count = 0

    @property
    def cache_size(self) -> int:
        """Number of input signatures with a resolved output layout."""
        return len(self._entries)

    @property
    def schemas(self) -> tuple[PackedTreeSchema, ...]:
        """Inspectable resolved layouts, in their construction order."""
        return tuple(entry.schema for entry in self._entries.values())

    @staticmethod
    def _signature(args: tuple, kwargs: dict) -> tuple:
        leaves, definition = jax.tree.flatten((args, kwargs))
        specifications = []
        for leaf in leaves:
            abstract = jax.typeof(leaf)
            specifications.append(
                (
                    tuple(abstract.shape),
                    np.dtype(abstract.dtype),
                    bool(getattr(abstract, "weak_type", False)),
                )
            )
        return definition, tuple(specifications), bool(jax.config.x64_enabled)

    def _entry(self, args: tuple, kwargs: dict) -> _PackedEntry:
        signature = self._signature(args, kwargs)
        if signature not in self._entries:
            abstract = jax.eval_shape(self.controller, *args, **kwargs)
            schema = PackedTreeSchema.from_abstract_tree(abstract)

            def packed(*call_args: Any, **call_kwargs: Any) -> tuple[jax.Array, ...]:
                result = self.controller(*call_args, **call_kwargs)
                # Keep transport concatenation outside producer arithmetic fusions.
                # This is a compiler barrier, not a value or precision conversion.
                result = jax.lax.optimization_barrier(result)
                return schema.pack(result)

            self._entries[signature] = _PackedEntry(schema, jax.jit(packed))
        return self._entries[signature]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Return every original output field as its original host pytree type."""
        entry = self._entry(args, kwargs)
        buffers = jax.device_get(entry.function(*args, **kwargs))
        result = entry.schema.unpack(buffers)
        self.last_buffer_count = len(buffers)
        self.last_leaf_count = len(entry.schema.leaves)
        return result

    def warmup(self, *args: Any, **kwargs: Any) -> float:
        """Measure a disposable full call; do not install a controller/policy snapshot."""
        started = time.perf_counter()
        self(*args, **kwargs)
        return time.perf_counter() - started

    def clear_cache(self) -> None:
        """Discard transport schemas; the supplied controller itself remains unchanged."""
        self._entries.clear()
        self.last_buffer_count = 0
        self.last_leaf_count = 0


def _variable_effort_step(
    state: jax.Array, command: jax.Array, model: ActuatorModel, dt: jax.Array
) -> jax.Array:
    """Exactly the prior P0 variable-duration predictor RK stage ordering."""
    effort = state[13:]
    half = exact_effort_update(effort, command, model.time_constants, dt / 2)
    end = exact_effort_update(effort, command, model.time_constants, dt)
    body = _body_rk4(
        state[:13],
        applied_wrench(effort, model),
        applied_wrench(half, model),
        applied_wrench(end, model),
        model,
        dt,
    )
    return jnp.concatenate((body, end))


@jax.jit
def _matched_segment(
    state: jax.Array, command: jax.Array, model: ActuatorModel, steps: jax.Array
) -> tuple[jax.Array, jax.Array]:
    def advance(current: jax.Array, dt: jax.Array) -> tuple[jax.Array, jax.Array]:
        following = _variable_effort_step(current, command, model, dt)
        return following, following

    return jax.lax.scan(advance, state, steps)


class MatchedEffortPlant(NumpyEffortPlant):
    """P0 predictor physics with one device scan/copy per event-separated segment.

    The host grid is identical to the previous per-step ``min(t+max_step,end)``
    clock, including its final remainder and event node. Model parameters are
    transferred only when the active immutable model changes. Event scheduling
    belongs to the physical plant and never appears in ``model_snapshot``.
    P1/P2 remain their independent NumPy implementations.
    """

    def __init__(
        self,
        initial_state: ArrayLike,
        model: ActuatorModel,
        *,
        events: Sequence[ActuatorEvent] = (),
        time: float = 0.0,
        max_step: float = 0.002,
    ) -> None:
        self._cached_model_source = None
        self._jax_model: ActuatorModel | None = None
        self.model_transfer_count = 0
        self.last_advance_segments = 0
        self.last_advance_steps = 0
        self._seen_step_counts: set[int] = set()
        super().__init__(initial_state, model, events=events, time=time, max_step=max_step)
        self._jax_state = jnp.asarray(self._state, dtype=jnp.float32)

    def _apply_events(self) -> None:
        super()._apply_events()
        if self._model is not self._cached_model_source:
            self._jax_model = jax.tree.map(
                lambda value: jnp.asarray(value, dtype=jnp.float32), self._model
            )
            self._cached_model_source = self._model
            self.model_transfer_count += 1

    @property
    def seen_step_counts(self) -> tuple[int, ...]:
        """Scan lengths encountered or explicitly warmed; new lengths can compile."""
        return tuple(sorted(self._seen_step_counts))

    def _grid(self, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
        times, steps = [], []
        current = start
        while current < end:
            following = min(end, current + self.max_step)
            if following <= current:
                raise ValueError("max_step cannot advance the floating-point physical clock")
            steps.append(following - current)
            times.append(following)
            current = following
        return np.asarray(times, dtype=float), np.asarray(steps, dtype=np.float32)

    def _validate_command(self, command: np.ndarray, duration: float) -> None:
        # Use only independent argument validation, never its body/actuator equations.
        _validate_step(self._state, command, self._model, duration, 1)
        if np.any(self._state[13:] < 0) or np.any(self._state[13:] > self._model.command_upper):
            raise ValueError("initial effort state is outside [0, command_upper]")

    def warmup(
        self,
        command: ArrayLike,
        *,
        durations: Sequence[float] = (0.04,),
        substep_counts: Sequence[int] = (),
    ) -> float:
        """Compile disposable segment shapes without moving time/state or applying events.

        Explicit counts can cover short event/latency segments before an episode.
        Counts alone reveal no future model value or fault schedule to a controller.
        Unwarmed lengths still work, and their compilation remains normal runtime cost.
        """
        started = time.perf_counter()
        command = _array(command, 4, "command")
        shapes: list[np.ndarray] = []
        for duration in durations:
            self._validate_command(command, duration)
            _, steps = self._grid(self.time, self.time + duration)
            shapes.append(steps)
        for count in substep_counts:
            if type(count) is not int or count < 1:
                raise ValueError("substep_counts must contain positive integers")
            self._validate_command(command, self.max_step * count)
            shapes.append(np.full(count, self.max_step, dtype=np.float32))
        for steps in shapes:
            jax.device_get(
                _matched_segment(
                    self._jax_state,
                    jnp.asarray(command, dtype=jnp.float32),
                    self._jax_model,
                    jnp.asarray(steps, dtype=jnp.float32),
                )
            )
            self._seen_step_counts.add(len(steps))
        return time.perf_counter() - started

    def advance(self, command: ArrayLike, duration: float) -> IndependentPlantTrace:
        """Hold one validated command through all current and intervening plant events."""
        command = _array(command, 4, "command")
        self._validate_command(command, duration)
        end = self.time + duration
        if not math.isfinite(end) or end <= self.time:
            raise ValueError("duration cannot advance the finite physical clock")
        force, wrench, _ = self._audit(command)
        time_blocks = [np.array([self.time])]
        state_blocks = [self._state[None].copy()]
        force_blocks, wrench_blocks = [force[None]], [wrench[None]]
        self.last_advance_segments = self.last_advance_steps = 0
        device_command = jnp.asarray(command, dtype=jnp.float32)
        while self.time < end:
            self._validate_command(command, end - self.time)
            boundary = end
            if self._event_index < len(self._events):
                boundary = min(boundary, self._events[self._event_index].time)
            times, steps = self._grid(self.time, boundary)
            following, future = _matched_segment(
                self._jax_state, device_command, self._jax_model, jnp.asarray(steps)
            )
            future_host = np.asarray(jax.device_get(future), dtype=float)
            if not np.all(np.isfinite(future_host)):
                raise FloatingPointError("nonfinite P0 integration")
            self._jax_state = following
            self._state = future_host[-1].copy()
            self.time = boundary
            forces = future_host[:, 13:] * np.asarray(self._model.effectiveness)
            wrenches = forces @ np.asarray(self._model.force_to_wrench).T
            self._apply_events()
            # Event endpoints retain x,s and report the force after eta changes,
            # exactly as NumpyEffortPlant's previous record-after-event convention.
            forces[-1], wrenches[-1], _ = self._audit(command)
            time_blocks.append(times)
            state_blocks.append(future_host)
            force_blocks.append(forces)
            wrench_blocks.append(wrenches)
            self._seen_step_counts.add(len(steps))
            self.last_advance_segments += 1
            self.last_advance_steps += len(steps)
        states = np.concatenate(state_blocks)
        return IndependentPlantTrace(
            np.concatenate(time_blocks),
            states,
            states.copy(),
            np.concatenate(force_blocks),
            np.concatenate(wrench_blocks),
            0.0,
        )

    def advance_audited(
        self,
        command: ArrayLike,
        duration: float,
        audit_callback: Callable[[np.ndarray, np.ndarray], bool],
        *,
        separation_precheck: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    ) -> IndependentPlantTrace:
        """Batch speculative stages, then commit only the chronologically audited prefix.

        The callback receives each pair of adjacent times and 17-state nodes. Its
        first True result commits the right node, exactly matching the previous
        per-step detection convention. A returned trace never contains a later
        speculative state. Model/event/device caches also return to that prefix;
        no command or event after it becomes part of the physical episode.
        A caller-supplied separation precheck may certify the complete trace and
        skip all adjacent-node callbacks by returning True. False retains the
        original first-stop scan. Either callback raising restores the entire
        starting physical state.
        """
        initial = (
            self.time,
            self._state.copy(),
            self._jax_state,
            self._model,
            self._event_index,
            self._jax_model,
            self._cached_model_source,
        )

        def restore_start() -> None:
            (
                self.time,
                self._state,
                self._jax_state,
                self._model,
                self._event_index,
                self._jax_model,
                self._cached_model_source,
            ) = initial

        try:
            trace = self.advance(command, duration)
            if separation_precheck is not None and separation_precheck(trace.times, trace.states):
                return trace
            for index in range(1, len(trace.times)):
                if not audit_callback(
                    trace.times[index - 1 : index + 1], trace.states[index - 1 : index + 1]
                ):
                    continue
                # Return all physical caches to the source before applying only
                # the events actually reached by this committed prefix.
                restore_start()
                self.time = float(trace.times[index])
                self._state = trace.states[index].copy()
                self._jax_state = jnp.asarray(self._state, dtype=jnp.float32)
                self._apply_events()
                return IndependentPlantTrace(
                    trace.times[: index + 1],
                    trace.states[: index + 1],
                    trace.native_states[: index + 1],
                    trace.actual_forces[: index + 1],
                    trace.wrenches[: index + 1],
                    trace.conversion_residual,
                )
            return trace
        except Exception:
            restore_start()
            raise
