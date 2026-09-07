"""Unbound exact-input cache prototype; never imported by the episode runtime.

Parity precedes profiling. No controller, policy, learner, or physical episode is
executed. Model values are cached only after a current/past observation query;
future obstacle positions retain the original public analytic prediction clock.
Runtime integration requires independent exact GPU parity and measured service gain.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crazyflow.safety.da_plcbf.actuator_experiment import _jsonable
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_study import (
    ActuatorObservationConfig,
    initial_augmented_state,
    make_actuator_scene,
    nominal_actuator_model,
    observe_actuator_state,
    observed_actuator_model,
)


class ObservationInputs(NamedTuple):
    """The unchanged state observation, model estimate and absolute-clock obstacle inputs."""

    observed_state: np.ndarray
    model: Any
    prediction: Any
    safety: Any


def reference_prediction_inputs(
    scene: Any, when: float, observation: ActuatorObservationConfig, config: ActuatorFilterConfig
) -> tuple[Any, Any]:
    """Reproduce the current runtime's observation-input expressions exactly."""
    prediction = scene.world.obstacle_prediction(when, dt=config.dt, horizon=config.horizon)
    safety = scene.world.safety_limits(when)
    bias = jnp.asarray(observation.obstacle_position_bias_m)
    return (
        prediction._replace(centers=prediction.centers + bias),
        safety._replace(obstacle_centers=safety.obstacle_centers + bias),
    )


def reference_inputs(
    actual: np.ndarray,
    when: float,
    scene: Any,
    nominal: Any,
    observation: ActuatorObservationConfig,
    config: ActuatorFilterConfig,
) -> ObservationInputs:
    """Run only original observation preparation, without invoking a controller."""
    state = observe_actuator_state(actual, when, scene.scene_seed, observation)
    model = observed_actuator_model(scene, when, nominal, observation)
    prediction, safety = reference_prediction_inputs(scene, when, observation, config)
    return ObservationInputs(state, model, prediction, safety)


class CausalObservationInputCache:
    """Lazy immutable-phase model and static-leaf cache with exact host bias ordering.

    The scene/configuration/nominal model and JAX precision/device context must
    remain fixed for the cache's lifetime. State noise and quaternion processing
    use the original observer every time. Safety centers keep their own scalar
    kinematics call: prediction node zero may have different signed-zero clock
    arithmetic from a direct scalar query, even though both describe time zero.
    """

    def __init__(
        self,
        scene: Any,
        nominal: Any,
        observation: ActuatorObservationConfig,
        config: ActuatorFilterConfig,
    ) -> None:
        """Store immutable inputs without querying any current or future model phase."""
        observation.validate()
        config.validate()
        self.scene, self.nominal = scene, nominal
        self.observation, self.config = observation, config
        self._models: dict[bool, Any] = {}
        self._templates: tuple[Any, Any] | None = None
        # This is the same original NumPy multiplication, including rounding.
        self._offsets = config.dt * np.arange(config.horizon + 1)
        self._bias = np.asarray(observation.obstacle_position_bias_m, dtype=np.float32)
        self._x64_enabled = jax.config.x64_enabled
        self.model_materializations: list[dict[str, float | bool]] = []

    def model_at(self, when: float) -> Any:
        """Materialize a phase only after its current/past estimate is actually requested."""
        observed_when = max(0.0, when - self.observation.parameter_delay_seconds)
        active = observed_when >= self.scene.event_time - 1e-10
        if self.scene.recovery_time is not None:
            active = active and observed_when < self.scene.recovery_time - 1e-10
        if active not in self._models:
            # Crucially, construction never evaluates the other/future phase.
            self._models[active] = observed_actuator_model(
                self.scene, when, self.nominal, self.observation
            )
            self.model_materializations.append(
                {
                    "requested_time_seconds": when,
                    "observed_time_seconds": observed_when,
                    "active_fault_phase": active,
                }
            )
        return self._models[active]

    def prediction_inputs(self, when: float) -> tuple[Any, Any]:
        """Preserve NumPy motion/cast/add order while batching three dynamic transfers."""
        if self._templates is None:
            self._templates = reference_prediction_inputs(
                self.scene, when, self.observation, self.config
            )
            return self._templates
        centers, velocities = self.scene.world.obstacle_kinematics(when + self._offsets)
        current_centers, _ = self.scene.world.obstacle_kinematics(when)
        # The original device expression casts centers to float32 first, then
        # adds the weak scalar bias in float32. Always add even a zero bias:
        # skipping it changes the sign bit of some -0.0 inputs.
        centers = np.add(np.asarray(centers, dtype=np.float32), self._bias, dtype=np.float32)
        current_centers = np.add(
            np.asarray(current_centers, dtype=np.float32), self._bias, dtype=np.float32
        )
        device_centers, device_velocities, device_current = jax.device_put(
            (centers, np.asarray(velocities, dtype=np.float32), current_centers)
        )
        prediction, safety = self._templates
        return (
            prediction._replace(centers=device_centers, velocities=device_velocities),
            safety._replace(obstacle_centers=device_current),
        )

    def at(self, actual: np.ndarray, when: float) -> ObservationInputs:
        """Return unchanged current-state observations plus cached immutable input leaves."""
        if jax.config.x64_enabled != self._x64_enabled:
            raise ValueError("the cache requires a fixed JAX precision context")
        observed = observe_actuator_state(actual, when, self.scene.scene_seed, self.observation)
        model = self.model_at(when)
        prediction, safety = self.prediction_inputs(when)
        return ObservationInputs(observed, model, prediction, safety)


def _compare(reference: Any, candidate: Any) -> dict[str, Any]:
    a, tree_a = jax.tree_util.tree_flatten(reference)
    b, tree_b = jax.tree_util.tree_flatten(candidate)
    differences = []
    if tree_a != tree_b:
        return {"all_leaves_exact": False, "structure_equal": False, "differences": []}
    for index, (first, second) in enumerate(zip(a, b, strict=True)):
        first, second = np.asarray(first), np.asarray(second)
        equal = (
            first.dtype == second.dtype
            and first.shape == second.shape
            and first.tobytes() == second.tobytes()
        )
        if not equal:
            differences.append(
                {
                    "leaf_index": index,
                    "reference_dtype": str(first.dtype),
                    "candidate_dtype": str(second.dtype),
                    "reference_shape": list(first.shape),
                    "candidate_shape": list(second.shape),
                    "reference_bytes_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
                    "candidate_bytes_sha256": hashlib.sha256(second.tobytes()).hexdigest(),
                }
            )
    return {
        "all_leaves_exact": not differences,
        "structure_equal": True,
        "leaf_count": len(a),
        "differences": differences,
    }


PARITY_CONTRACT = {
    "comparison": (
        "every leaf: identical tree structure, shape, dtype and raw bytes; includes signed zero"
    ),
    "physical_state_dtypes": ["float32 (P0)", "float64 with non-float32-representable values (P1)"],
    "nominal_model_dtypes": ["float32", "float64 under explicitly enabled x64"],
    "observations": [
        "zero noise/bias/delay",
        "nonzero all noise channels, parameter delay, effectiveness bias, "
        "lag scale and obstacle bias",
        "negative-zero obstacle bias",
    ],
    "event_queries": (
        "onset/recovery, neighboring floats, tolerance-edge neighbors, delayed event edges, "
        "repeated and out-of-order past queries"
    ),
    "signed_zero_fixture": (
        "negative-zero motion phases/means and both signed zero query times; "
        "scalar safety motion remains separately evaluated"
    ),
    "future_model_guard": (
        "cold cache is empty; querying before event leaves only the nominal phase materialized; "
        "only the actually queried current/past observed time enters source model_at"
    ),
    "source_scope": (
        "no runtime edits, controller/learner calls, integration, episode execution "
        "or 40ms clock changes"
    ),
}


def _query_times(scene: Any, observation: ActuatorObservationConfig) -> list[float]:
    values = [0.0, -0.0, 0.04, 0.0]
    for event in (scene.event_time, scene.recovery_time):
        if event is None:
            continue
        for center in (
            event,
            event + observation.parameter_delay_seconds,
            event + observation.parameter_delay_seconds - 1e-10,
        ):
            values.extend(
                [float(np.nextafter(center, -np.inf)), center, float(np.nextafter(center, np.inf))]
            )
    # Reverse queries prove that lookup order does not alter values or publish
    # the active estimate into a past observation of nominal dynamics.
    return values + list(reversed(values))


def verify_parity() -> dict[str, Any]:
    """Run the predeclared state/noise/clock/precision matrix without any controller call."""
    checks = []
    observations = [
        ActuatorObservationConfig(),
        ActuatorObservationConfig(
            parameter_delay_seconds=0.08,
            effectiveness_bias=-0.02,
            lag_scale=1.1,
            position_noise_m=0.002,
            velocity_noise_mps=0.003,
            attitude_noise_rad=0.001,
            rate_noise_rps=0.002,
            motor_noise_N=0.0001,
            obstacle_position_bias_m=0.003,
        ),
        ActuatorObservationConfig(obstacle_position_bias_m=-0.0),
    ]
    config = ActuatorFilterConfig(horizon=60)
    for dtype_name in ("float32", "float64"):
        with jax.enable_x64(dtype_name == "float64"):
            nominal = nominal_actuator_model(dtype=getattr(jnp, dtype_name))
            base = replace(make_actuator_scene(1002, "structured", "combined"), recovery_time=4.0)
            zeros = replace(
                base,
                world=replace(
                    base.world,
                    obstacle_mean_centers=np.full_like(base.world.obstacle_mean_centers, -0.0),
                    obstacle_amplitudes=np.zeros_like(base.world.obstacle_amplitudes),
                    obstacle_phases=np.full_like(base.world.obstacle_phases, -0.0),
                ),
            )
            for scene_label, scene in (("structured", base), ("signed_zero", zeros)):
                for physical_dtype in (np.float32, np.float64):
                    actual = initial_augmented_state(scene.world.initial_state, nominal).astype(
                        physical_dtype
                    )
                    actual[0] = physical_dtype(0.123456789012345)
                    for observation in observations:
                        cache = CausalObservationInputCache(scene, nominal, observation, config)
                        if cache._models or cache._templates is not None:
                            raise AssertionError("cache eagerly materialized unqueried inputs")
                        cold = cache.at(actual, 0.0)
                        if set(cache._models) != {False}:
                            raise AssertionError(
                                "a time-zero query materialized a future fault model"
                            )
                        jax.block_until_ready(cold)
                        for when in _query_times(scene, observation):
                            expected = jax.block_until_ready(
                                reference_inputs(actual, when, scene, nominal, observation, config)
                            )
                            proposed = jax.block_until_ready(cache.at(actual, when))
                            comparison = _compare(expected, proposed)
                            checks.append(
                                {
                                    "nominal_model_dtype": dtype_name,
                                    "physical_state_dtype": np.dtype(physical_dtype).name,
                                    "scene": scene_label,
                                    "observation": asdict(observation),
                                    "when": when,
                                    "when_signbit": bool(np.signbit(when)),
                                    **comparison,
                                }
                            )
    return {
        "passed": all(check["all_leaves_exact"] for check in checks),
        "comparison_count": len(checks),
        "contract": PARITY_CONTRACT,
        "checks": checks,
    }


def profile_inputs(count: int) -> dict[str, Any]:
    """Measure complete synchronized input preparation only after parity passes."""
    if count < 20:
        raise ValueError("profile count must be at least 20")
    nominal = nominal_actuator_model()
    scene = replace(make_actuator_scene(1002, "structured", "combined"), recovery_time=4.0)
    observation = ActuatorObservationConfig()
    config = ActuatorFilterConfig(horizon=60)
    actual = initial_augmented_state(scene.world.initial_state, nominal)
    cache = CausalObservationInputCache(scene, nominal, observation, config)
    # The warmup only queries current/past phases explicitly. All profile times
    # below are subsequently queried in increasing order; future faults are not
    # pre-materialized to improve a cold transition measurement.
    for when in (0.0, 0.04, 0.08):
        jax.block_until_ready(reference_inputs(actual, when, scene, nominal, observation, config))
        jax.block_until_ready(cache.at(actual, when))
    timings = {"reference_seconds": [], "prototype_seconds": []}
    differences = []
    query_times = np.arange(count, dtype=float) * config.command_period
    for index, when in enumerate(query_times):
        outputs = {}
        ordering = ("reference", "prototype") if index % 2 == 0 else ("prototype", "reference")
        for name in ordering:
            started = time.perf_counter()
            value = (
                reference_inputs(actual, float(when), scene, nominal, observation, config)
                if name == "reference"
                else cache.at(actual, float(when))
            )
            outputs[name] = jax.block_until_ready(value)
            timings[f"{name}_seconds"].append(time.perf_counter() - started)
        check = _compare(outputs["reference"], outputs["prototype"])
        if not check["all_leaves_exact"]:
            differences.append({"time_seconds": float(when), **check})
    return {
        "scope": (
            "synchronized input-preparation wall time; not complete controller service, "
            "paced viability, or an episode outcome"
        ),
        "count": count,
        "query_times_seconds": query_times,
        "cold_model_materializations": cache.model_materializations,
        "all_profile_inputs_exact": not differences,
        "differences": differences,
        "raw_timings": timings,
        "statistics": {
            name: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
            }
            for name, values in timings.items()
        },
    }


def _save(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("parity", "profile"))
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=160)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *(ROOT / "crazyflow/safety/da_plcbf").glob("*.py")]
    source_dir = output / "source"
    source_dir.mkdir()
    source_hashes = {}
    for path in sources:
        payload = path.read_bytes()
        (source_dir / path.name).write_bytes(payload)
        source_hashes[str(path)] = hashlib.sha256(payload).hexdigest()
    _save(
        output / "request.json",
        {
            "arguments": vars(args),
            "parity_contract": PARITY_CONTRACT,
            "source_sha256": source_hashes,
        },
    )
    try:
        if args.platform == "cpu":
            os.environ["JAX_PLATFORMS"] = "cpu"
            jax.config.update("jax_platforms", "cpu")
        device = jax.devices(args.platform)[0]
        with jax.default_device(device):
            parity = verify_parity()
            _save(output / "parity.json", parity)
            if not parity["passed"]:
                print(json.dumps({"output": str(output), "parity_passed": False}))
                return 2
            profile = profile_inputs(args.count) if args.operation == "profile" else None
            if profile is not None:
                _save(output / "profile.json", profile)
        _save(
            output / "result.json",
            {
                "status": "completed",
                "parity_passed": True,
                "device": str(device),
                "profile": profile,
                "runtime_integration_authorized_by_this_tool": False,
            },
        )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "parity_passed": True,
                    "statistics": None if profile is None else profile["statistics"],
                }
            )
        )
        return 0 if profile is None or profile["all_profile_inputs_exact"] else 2
    except Exception as error:
        _save(
            output / "error.json",
            {
                "status": "incomplete",
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        print(json.dumps({"output": str(output), "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
