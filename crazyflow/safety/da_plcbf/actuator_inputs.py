"""Exact observation preparation with lazy current/past model caching.

The implementation preserves the byte-verified input-cache prototype: current
state observation remains unchanged, static device leaves are reused, and only
already queried model phases can be cached. Future obstacle predictions use the
same declared analytic absolute clock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.actuator_study import observe_actuator_state, observed_actuator_model

if TYPE_CHECKING:
    from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
    from crazyflow.safety.da_plcbf.actuator_study import ActuatorObservationConfig


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
