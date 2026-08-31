"""Immutable policy snapshots and atomic active-snapshot publication.

Snapshots keep array payloads as immutable byte strings.  Accessors reconstruct a fresh PyTree
whose NumPy leaves view those byte strings, so callers cannot mutate either the leaf storage or the
container structure held by the snapshot.  Publication is deliberately separate from candidate
creation: only :class:`ActiveSnapshotStore` assigns active versions.
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import jax
import numpy as np

if TYPE_CHECKING:
    from jax.tree_util import PyTreeDef

    from crazyflow.safety.da_plcbf.validation import ValidationReport


SnapshotKind = Literal["active", "candidate"]
_ALLOWED_ARRAY_KINDS = frozenset("biufc")


@dataclass(frozen=True, slots=True)
class _FrozenLeaf:
    """Canonical immutable representation of one numeric PyTree leaf."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def array(self) -> np.ndarray:
        """Reconstruct a read-only array backed by an immutable bytes object."""
        return np.frombuffer(self.data, dtype=np.dtype(self.dtype)).reshape(self.shape)


@dataclass(frozen=True, slots=True, eq=False)
class PolicySnapshot:
    """Content-addressed, immutable policy-library state.

    Use :func:`create_active_snapshot` or :func:`create_candidate_snapshot`; direct construction is
    intentionally unsupported.  ``params`` and ``structural_core`` return defensive PyTree views.
    Mutating a returned container cannot change this object, and returned arrays are read-only.
    """

    kind: SnapshotKind
    version: int
    base_active_version: int
    base_active_digest: str
    model_version: int
    digest: str
    _params_treedef: PyTreeDef = field(repr=False)
    _params_leaves: tuple[_FrozenLeaf, ...] = field(repr=False)
    _core_treedef: PyTreeDef = field(repr=False)
    _core_leaves: tuple[_FrozenLeaf, ...] = field(repr=False)
    _metadata_json: str = field(repr=False)

    @property
    def params(self) -> Any:
        """Return a defensive, read-only reconstruction of the policy parameter PyTree."""
        return self._params_treedef.unflatten([leaf.array() for leaf in self._params_leaves])

    @property
    def structural_core(self) -> Any:
        """Return the immutable structural policy core as a defensive PyTree reconstruction."""
        return self._core_treedef.unflatten([leaf.array() for leaf in self._core_leaves])

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return a recursively immutable copy of canonical snapshot metadata."""
        return _freeze_json_value(json.loads(self._metadata_json))

    @property
    def params_digest(self) -> str:
        """Return the content digest of only the trainable parameter PyTree."""
        return _tree_digest(self._params_treedef, self._params_leaves)

    @property
    def params_schema_digest(self) -> str:
        """Return the digest of parameter structure, dtypes, and shapes, excluding values."""
        return _tree_schema_digest(self._params_treedef, self._params_leaves)

    @property
    def structural_core_digest(self) -> str:
        """Return the content digest of only the structural-core PyTree."""
        return _tree_digest(self._core_treedef, self._core_leaves)

    def all_finite(self) -> bool:
        """Return whether all stored numeric leaves are finite."""
        return all(
            bool(np.all(np.isfinite(leaf.array())))
            for leaf in (*self._params_leaves, *self._core_leaves)
        )

    def verify_integrity(self) -> bool:
        """Recompute and compare the complete content-addressed snapshot digest."""
        try:
            expected = _snapshot_digest(
                kind=self.kind,
                version=self.version,
                base_active_version=self.base_active_version,
                base_active_digest=self.base_active_digest,
                model_version=self.model_version,
                params_treedef=self._params_treedef,
                params_leaves=self._params_leaves,
                core_treedef=self._core_treedef,
                core_leaves=self._core_leaves,
                metadata_json=self._metadata_json,
            )
        except (TypeError, ValueError):
            return False
        return self.digest == expected


@dataclass(frozen=True, slots=True)
class PublicationResult:
    """Outcome of an admission or rollback publication attempt."""

    accepted: bool
    reason: str
    active: PolicySnapshot


class ActiveSnapshotStore:
    """Thread-safe owner of the active and previous validated policy snapshots.

    The lock spans validation-report binding checks and publication, making the compare-and-swap
    atomic.  A candidate based on an older active or model version is rejected after another thread
    publishes or advances the model version.
    """

    def __init__(self, initial: PolicySnapshot, *, model_version: int | None = None) -> None:
        if initial.kind != "active":
            raise ValueError("initial snapshot must be active")
        if not initial.verify_integrity():
            raise ValueError("initial snapshot failed integrity verification")
        if not initial.all_finite():
            raise ValueError("initial snapshot contains nonfinite values")
        resolved_model_version = initial.model_version if model_version is None else model_version
        _validate_nonnegative_version(resolved_model_version, "model_version")
        if resolved_model_version != initial.model_version:
            raise ValueError("initial snapshot and store model versions must match")
        self._lock = threading.RLock()
        self._active = initial
        self._previous: PolicySnapshot | None = None
        self._model_version = resolved_model_version

    @property
    def active(self) -> PolicySnapshot:
        """Return the current immutable active snapshot."""
        with self._lock:
            return self._active

    @property
    def previous(self) -> PolicySnapshot | None:
        """Return the immediately previous validated snapshot, if one has been published."""
        with self._lock:
            return self._previous

    @property
    def model_version(self) -> int:
        """Return the dynamics-model version against which candidates must be validated."""
        with self._lock:
            return self._model_version

    def advance_model_version(self, new_version: int) -> None:
        """Atomically advance the model version and invalidate in-flight candidates.

        Versions are logical counters, not wall-clock values.  Requiring a strict increase avoids
        time-based freshness heuristics and ambiguous reuse.
        """
        _validate_nonnegative_version(new_version, "new_version")
        with self._lock:
            if new_version <= self._model_version:
                raise ValueError("new model version must be strictly greater than the current one")
            self._model_version = new_version

    def admit(self, candidate: PolicySnapshot, report: ValidationReport) -> PublicationResult:
        """Atomically publish a fresh candidate only when its bound hard report passes."""
        with self._lock:
            rejection = self._admission_rejection(candidate, report)
            if rejection is not None:
                return PublicationResult(False, rejection, self._active)

            previous = self._active
            metadata = json.loads(candidate._metadata_json)
            metadata["publication"] = {
                "candidate_digest": candidate.digest,
                "candidate_version": candidate.version,
                "report_digest": report.digest,
                "type": "admission",
            }
            published = create_active_snapshot(
                candidate.params,
                version=previous.version + 1,
                model_version=self._model_version,
                structural_core=candidate.structural_core,
                metadata=metadata,
                base_active_version=previous.version,
                base_active_digest=previous.digest,
            )
            self._previous = previous
            self._active = published
            return PublicationResult(True, "admitted", published)

    def publish_rollback(self, *, expected_active_version: int | None = None) -> PublicationResult:
        """Publish the previous validated payload under a new monotonically increasing version.

        Rollback never reinstates or mutates the old snapshot object.  It republishes a byte-exact
        copy of its policy/core payload with explicit rollback provenance.  A previous snapshot
        validated under another model version is rejected as stale.
        """
        with self._lock:
            current = self._active
            previous = self._previous
            if expected_active_version is not None and expected_active_version != current.version:
                return PublicationResult(False, "active_version_mismatch", current)
            if previous is None:
                return PublicationResult(False, "no_previous_snapshot", current)
            if not current.verify_integrity() or not previous.verify_integrity():
                return PublicationResult(False, "snapshot_integrity_failed", current)
            if not previous.all_finite():
                return PublicationResult(False, "previous_snapshot_nonfinite", current)
            if previous.model_version != self._model_version:
                return PublicationResult(False, "previous_model_version_stale", current)

            metadata = json.loads(previous._metadata_json)
            metadata["publication"] = {
                "rolled_back_from_digest": current.digest,
                "rolled_back_from_version": current.version,
                "source_digest": previous.digest,
                "source_version": previous.version,
                "type": "rollback",
            }
            published = create_active_snapshot(
                previous.params,
                version=current.version + 1,
                model_version=self._model_version,
                structural_core=previous.structural_core,
                metadata=metadata,
                base_active_version=current.version,
                base_active_digest=current.digest,
            )
            self._previous = current
            self._active = published
            return PublicationResult(True, "rollback_published", published)

    def _admission_rejection(
        self, candidate: PolicySnapshot, report: ValidationReport
    ) -> str | None:
        """Return a stable rejection code, or ``None`` when publication is allowed."""
        active = self._active
        if candidate.kind != "candidate":
            return "not_a_candidate"
        if not active.verify_integrity() or not candidate.verify_integrity():
            return "snapshot_integrity_failed"
        if not candidate.all_finite():
            return "candidate_nonfinite"
        if candidate.base_active_version != active.version:
            return "stale_base_active_version"
        if candidate.base_active_digest != active.digest:
            return "stale_base_active_digest"
        if candidate.model_version != self._model_version:
            return "stale_model_version"
        if not report.verify_integrity():
            return "report_integrity_failed"
        if report.active_digest != active.digest or report.active_version != active.version:
            return "report_active_mismatch"
        if (
            report.candidate_digest != candidate.digest
            or report.candidate_version != candidate.version
        ):
            return "report_candidate_mismatch"
        if report.model_version != self._model_version:
            return "report_model_version_stale"
        if not report.passed:
            return "hard_validation_failed"
        return None


def create_active_snapshot(
    params: Any,
    *,
    version: int = 0,
    model_version: int = 0,
    structural_core: Any = None,
    metadata: Mapping[str, Any] | None = None,
    base_active_version: int = -1,
    base_active_digest: str = "",
) -> PolicySnapshot:
    """Create an immutable active snapshot, normally for initialization or store publication."""
    return _create_snapshot(
        kind="active",
        params=params,
        version=version,
        base_active_version=base_active_version,
        base_active_digest=base_active_digest,
        model_version=model_version,
        structural_core=structural_core,
        metadata=metadata,
    )


def create_candidate_snapshot(
    params: Any,
    *,
    version: int,
    base_active: PolicySnapshot,
    model_version: int | None = None,
    structural_core: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PolicySnapshot:
    """Create an immutable candidate tied to an exact active snapshot and model version."""
    if base_active.kind != "active":
        raise ValueError("base_active must be an active snapshot")
    if not base_active.verify_integrity():
        raise ValueError("base_active failed integrity verification")
    resolved_core = base_active.structural_core if structural_core is None else structural_core
    resolved_model_version = base_active.model_version if model_version is None else model_version
    return _create_snapshot(
        kind="candidate",
        params=params,
        version=version,
        base_active_version=base_active.version,
        base_active_digest=base_active.digest,
        model_version=resolved_model_version,
        structural_core=resolved_core,
        metadata=metadata,
    )


def tree_content_digest(tree: Any) -> str:
    """Compute the canonical SHA-256 digest of a numeric PyTree's structure and leaves."""
    treedef, leaves = _freeze_tree(tree)
    return _tree_digest(treedef, leaves)


def _create_snapshot(
    *,
    kind: SnapshotKind,
    params: Any,
    version: int,
    base_active_version: int,
    base_active_digest: str,
    model_version: int,
    structural_core: Any,
    metadata: Mapping[str, Any] | None,
) -> PolicySnapshot:
    if kind not in ("active", "candidate"):
        raise ValueError(f"unsupported snapshot kind: {kind!r}")
    _validate_nonnegative_version(version, "version")
    _validate_nonnegative_version(model_version, "model_version")
    if base_active_version < -1:
        raise ValueError("base_active_version must be at least -1")
    if not isinstance(base_active_digest, str):
        raise TypeError("base_active_digest must be a string")
    if kind == "candidate" and (base_active_version < 0 or not base_active_digest):
        raise ValueError("candidate snapshots require an exact base-active identity")

    params_treedef, params_leaves = _freeze_tree(params)
    core_treedef, core_leaves = _freeze_tree(structural_core)
    metadata_json = _canonical_metadata(metadata)
    digest = _snapshot_digest(
        kind=kind,
        version=version,
        base_active_version=base_active_version,
        base_active_digest=base_active_digest,
        model_version=model_version,
        params_treedef=params_treedef,
        params_leaves=params_leaves,
        core_treedef=core_treedef,
        core_leaves=core_leaves,
        metadata_json=metadata_json,
    )
    return PolicySnapshot(
        kind=kind,
        version=version,
        base_active_version=base_active_version,
        base_active_digest=base_active_digest,
        model_version=model_version,
        digest=digest,
        _params_treedef=params_treedef,
        _params_leaves=params_leaves,
        _core_treedef=core_treedef,
        _core_leaves=core_leaves,
        _metadata_json=metadata_json,
    )


def _freeze_tree(tree: Any) -> tuple[PyTreeDef, tuple[_FrozenLeaf, ...]]:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    frozen: list[_FrozenLeaf] = []
    for index, value in enumerate(leaves):
        array = np.asarray(value)
        if array.dtype.kind not in _ALLOWED_ARRAY_KINDS:
            raise TypeError(
                f"PyTree leaf {index} has unsupported dtype {array.dtype}; numeric leaves required"
            )
        contiguous = np.ascontiguousarray(array)
        frozen.append(
            _FrozenLeaf(
                dtype=contiguous.dtype.str,
                shape=tuple(int(size) for size in contiguous.shape),
                data=contiguous.tobytes(order="C"),
            )
        )
    return treedef, tuple(frozen)


def _tree_digest(treedef: PyTreeDef, leaves: tuple[_FrozenLeaf, ...]) -> str:
    hasher = hashlib.sha256()
    _hash_component(hasher, b"crazyflow.da_plcbf.pytree.v1")
    _hash_component(hasher, str(treedef).encode("utf-8"))
    _hash_component(hasher, struct.pack(">Q", len(leaves)))
    for leaf in leaves:
        _hash_component(hasher, leaf.dtype.encode("ascii"))
        _hash_component(hasher, json.dumps(leaf.shape, separators=(",", ":")).encode("ascii"))
        _hash_component(hasher, leaf.data)
    return hasher.hexdigest()


def _tree_schema_digest(treedef: PyTreeDef, leaves: tuple[_FrozenLeaf, ...]) -> str:
    hasher = hashlib.sha256()
    _hash_component(hasher, b"crazyflow.da_plcbf.pytree_schema.v1")
    _hash_component(hasher, str(treedef).encode("utf-8"))
    _hash_component(hasher, struct.pack(">Q", len(leaves)))
    for leaf in leaves:
        _hash_component(hasher, leaf.dtype.encode("ascii"))
        _hash_component(hasher, json.dumps(leaf.shape, separators=(",", ":")).encode("ascii"))
    return hasher.hexdigest()


def _snapshot_digest(
    *,
    kind: SnapshotKind,
    version: int,
    base_active_version: int,
    base_active_digest: str,
    model_version: int,
    params_treedef: PyTreeDef,
    params_leaves: tuple[_FrozenLeaf, ...],
    core_treedef: PyTreeDef,
    core_leaves: tuple[_FrozenLeaf, ...],
    metadata_json: str,
) -> str:
    hasher = hashlib.sha256()
    _hash_component(hasher, b"crazyflow.da_plcbf.snapshot.v1")
    identity = {
        "base_active_digest": base_active_digest,
        "base_active_version": base_active_version,
        "kind": kind,
        "model_version": model_version,
        "version": version,
    }
    _hash_component(
        hasher, json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    _hash_component(hasher, _tree_digest(params_treedef, params_leaves).encode("ascii"))
    _hash_component(hasher, _tree_digest(core_treedef, core_leaves).encode("ascii"))
    _hash_component(hasher, metadata_json.encode("utf-8"))
    return hasher.hexdigest()


def _hash_component(hasher: Any, value: bytes) -> None:
    hasher.update(struct.pack(">Q", len(value)))
    hasher.update(value)


def _canonical_metadata(metadata: Mapping[str, Any] | None) -> str:
    value: Mapping[str, Any] = {} if metadata is None else metadata
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    try:
        return json.dumps(
            _normalize_json_value(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise TypeError("metadata must contain finite JSON-compatible values") from error


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("metadata mapping keys must be strings")
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _normalize_json_value(value.item())
    raise TypeError(f"unsupported metadata value type: {type(value).__name__}")


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _validate_nonnegative_version(version: int, name: str) -> None:
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError(f"{name} must be an integer")
    if version < 0:
        raise ValueError(f"{name} must be nonnegative")
