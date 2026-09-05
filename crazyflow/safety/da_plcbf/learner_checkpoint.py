"""Portable, integrity-checked checkpoints for the persistent obstacle-free learner.

The NPZ contains numeric arrays only; JSON contains the configuration, array shapes/dtypes, and
nested state-dictionary structure. Loading reconstructs the optimizer with the saved configuration
and restores its complete history through Flax serialization. No Python pickle or executable
object serialization is used. A checkpoint does not certify obstacle safety or admit a policy.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from crazyflow.safety.da_plcbf.persistent_skill_learner import (
    PersistentLearnerState,
    PersistentSkillConfig,
    SkillActorParams,
    SkillLibrarySpec,
    build_persistent_skill_learner,
)
from crazyflow.safety.da_plcbf.version_a_barriers import VersionAModel
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator

_FORMAT = "crazyflow.persistent_skill_checkpoint"
_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class LearnerCheckpoint:
    """One complete immutable episode continuation point and its physical comparison state."""

    state: PersistentLearnerState
    spec: SkillLibrarySpec
    config: PersistentSkillConfig
    actuator: VersionAActuator
    physical_state: jax.Array
    metadata: dict[str, Any]
    npz_path: Path
    json_path: Path
    sha256: str

    @property
    def point_model(self) -> VersionAModel:
        """Return the dynamics estimate saved with the optimizer snapshot."""
        return self.state.latest_dynamics_estimate


def _checkpoint_paths(path_stem: str | Path) -> tuple[Path, Path]:
    stem = Path(path_stem)
    if stem.suffix in (".npz", ".json"):
        stem = stem.with_suffix("")
    return Path(f"{stem}.npz"), Path(f"{stem}.json")


def _encode_arrays(value: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": {str(key): _encode_arrays(item, arrays) for key, item in value.items()},
        }
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"unsupported checkpoint array dtype {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError("only a finite learner continuation state can be checkpointed")
    key = f"array_{len(arrays):05d}"
    arrays[key] = array
    return {"kind": "array", "key": key}


def _decode_arrays(node: dict[str, Any], arrays: dict[str, np.ndarray]) -> Any:
    if not isinstance(node, dict):
        raise ValueError("invalid checkpoint structure node")
    if node.get("kind") == "dict" and set(node) == {"kind", "items"}:
        return {key: _decode_arrays(value, arrays) for key, value in node["items"].items()}
    if node.get("kind") == "array" and set(node) == {"kind", "key"}:
        if node["key"] not in arrays:
            raise ValueError("checkpoint structure references a missing array")
        return arrays[node["key"]]
    raise ValueError("unknown checkpoint structure node")


def save_learner_checkpoint(
    state: PersistentLearnerState,
    spec: SkillLibrarySpec,
    config: PersistentSkillConfig,
    actuator: VersionAActuator,
    physical_state: jax.Array,
    path_stem: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save parameters, previous parameters, full Adam history, model/spec, and physical state.

    ``path_stem`` names a paired ``.npz``/``.json`` checkpoint. Existing checkpoints are never
    overwritten. Optional metadata may contain measured competency diagnostics; it is descriptive
    and is never interpreted as a learner-update acceptance decision.
    """
    config.validate()
    if physical_state.shape != (13,) or not np.all(np.isfinite(np.asarray(physical_state))):
        raise ValueError("physical_state must be a finite 13-component state")
    # Building/initializing validates the actor/spec shapes and reconstructible optimizer config.
    functions = build_persistent_skill_learner(spec, actuator, config)
    template = functions.initialize(state.params, state.latest_dynamics_estimate)
    state_dictionary = serialization.to_state_dict(state)
    serialization.from_state_dict(template, state_dictionary)
    if (
        int(np.asarray(state.library_version)) < 0
        or int(np.asarray(state.cumulative_gradient_steps)) < 0
    ):
        raise ValueError("learner version and cumulative step count must be nonnegative")
    payload = {
        "learner_state": state_dictionary,
        "spec": serialization.to_state_dict(spec),
        "actuator": serialization.to_state_dict(actuator),
        "physical_state": physical_state,
    }
    arrays: dict[str, np.ndarray] = {}
    structure = _encode_arrays(payload, arrays)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    npz_bytes = buffer.getvalue()
    digest = hashlib.sha256(npz_bytes).hexdigest()
    manifest = {
        "format": _FORMAT,
        "format_version": _FORMAT_VERSION,
        "npz_sha256": digest,
        "config": asdict(config),
        "objective": "3D displacement diversity; separate terminal braking/attitude/rate losses",
        "library_version": int(np.asarray(state.library_version)),
        "cumulative_gradient_steps": int(np.asarray(state.cumulative_gradient_steps)),
        "structure": structure,
        "arrays": {
            key: {"shape": list(value.shape), "dtype": value.dtype.str}
            for key, value in arrays.items()
        },
        "libraries": {"jax": jax.__version__, "flax": flax.__version__, "optax": optax.__version__},
        "metadata": {} if metadata is None else metadata,
    }
    json_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    npz_path, json_path = _checkpoint_paths(path_stem)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    if npz_path.exists() or json_path.exists():
        raise FileExistsError("refusing to overwrite an existing learner checkpoint")
    created = []
    try:
        for path, data in ((npz_path, npz_bytes), (json_path, json_bytes)):
            with path.open("xb") as stream:
                created.append(path)
                stream.write(data)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return npz_path, json_path


def load_learner_checkpoint(
    path_stem: str | Path, *, device: jax.Device | None = None
) -> LearnerCheckpoint:
    """Restore a checked numeric checkpoint without resetting any optimizer history."""
    npz_path, json_path = _checkpoint_paths(path_stem)
    manifest = json.loads(json_path.read_text())
    if manifest.get("format") != _FORMAT or manifest.get("format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported learner checkpoint format/version")
    npz_bytes = npz_path.read_bytes()
    digest = hashlib.sha256(npz_bytes).hexdigest()
    if digest != manifest.get("npz_sha256"):
        raise ValueError("learner checkpoint NPZ checksum mismatch")
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
        if set(archive.files) != set(manifest["arrays"]):
            raise ValueError("checkpoint array keys do not match the manifest")
        arrays = {key: archive[key] for key in archive.files}
    for key, value in arrays.items():
        expected = manifest["arrays"][key]
        if list(value.shape) != expected["shape"] or value.dtype.str != expected["dtype"]:
            raise ValueError(f"checkpoint array shape/dtype mismatch: {key}")
        if value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)):
            raise ValueError(f"checkpoint array is nonnumeric/nonfinite: {key}")
    payload = _decode_arrays(manifest["structure"], arrays)
    if set(payload) != {"learner_state", "spec", "actuator", "physical_state"}:
        raise ValueError("checkpoint payload fields do not match the supported schema")
    options = dict(manifest["config"])
    options["descriptor_scales"] = tuple(options["descriptor_scales"])
    config = PersistentSkillConfig(**options)
    config.validate()
    spec = SkillLibrarySpec(**payload["spec"])
    actuator = VersionAActuator(**payload["actuator"])
    state_dictionary = payload["learner_state"]
    params = SkillActorParams(**state_dictionary["params"])
    model = VersionAModel(**state_dictionary["latest_dynamics_estimate"])
    functions = build_persistent_skill_learner(spec, actuator, config)
    template = functions.initialize(params, model)
    restored = serialization.from_state_dict(template, state_dictionary)
    physical_state = payload["physical_state"]
    if physical_state.shape != (13,):
        raise ValueError("checkpoint physical state must contain 13 components")
    if (
        int(np.asarray(restored.library_version)) != manifest["library_version"]
        or int(np.asarray(restored.cumulative_gradient_steps))
        != manifest["cumulative_gradient_steps"]
    ):
        raise ValueError("checkpoint learner counters do not match the manifest")
    selected_device = jax.devices()[0] if device is None else device

    def place(value: Any) -> jax.Array:
        array = np.asarray(value)
        converted = jax.device_put(jnp.asarray(array), selected_device)
        if np.dtype(converted.dtype) != array.dtype:
            raise ValueError("checkpoint dtype cannot be restored exactly; check JAX x64 settings")
        return converted

    restored, spec, actuator, physical_state = jax.tree.map(
        place, (restored, spec, actuator, physical_state)
    )
    return LearnerCheckpoint(
        state=restored,
        spec=spec,
        config=config,
        actuator=actuator,
        physical_state=physical_state,
        metadata=manifest.get("metadata", {}),
        npz_path=npz_path,
        json_path=json_path,
        sha256=digest,
    )


__all__ = ["LearnerCheckpoint", "load_learner_checkpoint", "save_learner_checkpoint"]
