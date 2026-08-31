"""Content-addressed evidence for online BPTT proposal and admission decisions.

An admission event is only a convenient index.  It is not scientific evidence by itself.  This
module persists the immutable active/candidate payloads, the held-out numerical evidence, the
hard-gate thresholds/report, and the active snapshot visible at the publication boundary.  A
validator then recomputes the report and replays :class:`ActiveSnapshotStore.admit` before an
online run may be treated as claim eligible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np

from crazyflow.safety.da_plcbf.artifacts import (
    ArtifactEvent,
    ImmutableTrace,
    _atomic_write_bytes,
    _canonical_array_digest,
    _deterministic_npz_bytes,
    _load_strict_npz,
)
from crazyflow.safety.da_plcbf.snapshots import ActiveSnapshotStore, PolicySnapshot, _FrozenLeaf
from crazyflow.safety.da_plcbf.validation import (
    GATE_NAMES,
    GateResult,
    HardValidationEvidence,
    HardValidationThresholds,
    ValidationReport,
    hard_validate_candidate,
)

ADAPTATION_EVIDENCE_SCHEMA_VERSION = 1
_PREFIX = b"crazyflow.da_plcbf.adaptation-evidence.v1\0"
_SNAPSHOT_ROLES = ("proposal_active", "decision_active", "candidate", "publication_active")
_STATUSES = frozenset({"admitted", "rejected", "expired"})


@dataclass(frozen=True, slots=True)
class CandidateValidationMaterial:
    """Exact proposal-time material retained until its publication boundary."""

    proposal_active: PolicySnapshot
    context_step: int
    candidate: PolicySnapshot
    evidence: HardValidationEvidence
    thresholds: HardValidationThresholds
    report: ValidationReport


@dataclass(frozen=True, slots=True)
class AdaptationDecisionProof:
    """One proposal/report and its boundary-synchronized publication outcome."""

    phase: Literal["cold_start", "online"]
    job_id: int
    context_step: int
    boundary_step: int
    status: Literal["admitted", "rejected", "expired"]
    decision_model_version: int
    publication_reason: str
    used_by_executed_control: bool
    proposal_active: PolicySnapshot
    decision_active: PolicySnapshot
    candidate: PolicySnapshot
    publication_active: PolicySnapshot
    evidence: HardValidationEvidence
    thresholds: HardValidationThresholds
    report: ValidationReport


@dataclass(frozen=True, slots=True)
class AdaptationEvidence:
    """Ordered online decision proofs for one trace."""

    trace_content_sha256: str
    decisions: tuple[AdaptationDecisionProof, ...]

    @property
    def content_sha256(self) -> str:
        """Return the canonical digest of the serialized semantic arrays."""
        return _canonical_array_digest(_PREFIX, _evidence_arrays(self))


def validate_adaptation_evidence_binding(
    evidence: AdaptationEvidence,
    trace: ImmutableTrace,
    events: tuple[ArtifactEvent, ...],
    *,
    tape: Any | None = None,
    condition: Any | None = None,
    method: Any | None = None,
    config: Any | None = None,
) -> None:
    """Recompute every report/admission and bind each proof to the trace and event stream."""
    if not isinstance(evidence, AdaptationEvidence):
        raise TypeError("evidence must be AdaptationEvidence")
    if not isinstance(trace, ImmutableTrace):
        raise TypeError("trace must be ImmutableTrace")
    trace.validate()
    if evidence.trace_content_sha256 != trace.content_sha256:
        raise ValueError("adaptation evidence trace digest does not match the trace")
    if not evidence.decisions:
        raise ValueError("online adaptation evidence must contain at least one decision")
    numerical_context = (tape, condition, method, config)
    if any(item is None for item in numerical_context) and not all(
        item is None for item in numerical_context
    ):
        raise ValueError("adaptation numerical replay requires tape, condition, method, and config")

    decision_events = tuple(
        event
        for event in events
        if (
            event.category == "cold_start"
            and event.name in {"candidate_admitted", "candidate_rejected"}
        )
        or (
            event.category == "adaptation"
            and event.name in {"candidate_admitted", "candidate_rejected", "candidate_expired"}
        )
    )
    if len(decision_events) != len(evidence.decisions):
        raise ValueError("adaptation proof count does not match completed decision events")

    previous_boundary = -1
    for index, (proof, event) in enumerate(zip(evidence.decisions, decision_events, strict=True)):
        _validate_decision_proof(proof, trace)
        if proof.boundary_step < previous_boundary:
            raise ValueError("adaptation decision proofs are not in boundary order")
        previous_boundary = proof.boundary_step
        expected_phase = "cold_start" if event.category == "cold_start" else "online"
        if proof.phase != expected_phase:
            raise ValueError("adaptation proof phase does not match its event")
        if proof.boundary_step != event.step:
            raise ValueError("adaptation proof boundary does not match its event")
        if event.name != f"candidate_{proof.status}":
            raise ValueError("adaptation proof status does not match its event")
        if proof.phase == "cold_start":
            if proof.job_id != -1:
                raise ValueError("cold-start proof must use the reserved job id -1")
        elif event.details.get("job_id") != proof.job_id or proof.job_id < 0:
            raise ValueError("online adaptation proof job id does not match its event")
        if proof.phase == "online":
            submissions = tuple(
                item
                for item in events
                if item.category == "adaptation"
                and item.name == "candidate_submitted"
                and item.details.get("reason") == "submitted"
                and item.details.get("job_id") == proof.job_id
            )
            if len(submissions) != 1 or submissions[0].step != proof.context_step:
                raise ValueError("online adaptation proof is not bound to its submission context")
        if event.details.get("candidate_digest") != proof.candidate.digest:
            raise ValueError("adaptation event candidate digest is not proof-bound")
        if event.details.get("report_digest") != proof.report.digest:
            raise ValueError("adaptation event report digest is not proof-bound")
        if event.details.get("reason") != proof.publication_reason:
            raise ValueError("adaptation event reason is not proof-bound")
        published_version = proof.publication_active.version if proof.status == "admitted" else None
        if event.details.get("published_snapshot_version") != published_version:
            raise ValueError("adaptation event publication version is not proof-bound")
        if int(event.snapshot_version) != proof.publication_active.version:
            raise ValueError("adaptation event snapshot version is not proof-bound")
        if proof.phase == "online" and event.details.get("publication_boundary") != event.step:
            raise ValueError("online adaptation event has an invalid publication boundary")
        if index == 0 and proof.phase != "cold_start":
            raise ValueError("the first online-library decision must be cold-start admission")
    if all(item is not None for item in numerical_context):
        _validate_numerical_inputs(
            evidence.decisions,
            trace,
            events=events,
            tape=tape,
            condition=condition,
            method=method,
            config=config,
        )


def _validate_decision_proof(proof: AdaptationDecisionProof, trace: ImmutableTrace) -> None:
    if proof.phase not in {"cold_start", "online"} or proof.status not in _STATUSES:
        raise ValueError("adaptation proof phase/status is invalid")
    if (
        isinstance(proof.job_id, bool)
        or not isinstance(proof.job_id, int)
        or isinstance(proof.boundary_step, bool)
        or not isinstance(proof.boundary_step, int)
        or isinstance(proof.context_step, bool)
        or not isinstance(proof.context_step, int)
        or not 0 <= proof.context_step <= proof.boundary_step
        or not 0 <= proof.boundary_step < trace.steps
    ):
        raise ValueError("adaptation proof job/boundary is invalid")
    if proof.phase == "cold_start" and proof.context_step != 0:
        raise ValueError("cold-start adaptation proof must use context step zero")
    if (
        isinstance(proof.decision_model_version, bool)
        or not isinstance(proof.decision_model_version, int)
        or proof.decision_model_version < 0
    ):
        raise ValueError("adaptation proof decision model version is invalid")
    if not proof.publication_reason:
        raise ValueError("adaptation proof publication reason is empty")

    snapshots = (
        proof.proposal_active,
        proof.decision_active,
        proof.candidate,
        proof.publication_active,
    )
    if any(not item.verify_integrity() or not item.all_finite() for item in snapshots):
        raise ValueError("adaptation proof contains an invalid snapshot")
    if (
        proof.proposal_active.kind != "active"
        or proof.decision_active.kind != "active"
        or proof.candidate.kind != "candidate"
        or proof.publication_active.kind != "active"
    ):
        raise ValueError("adaptation proof snapshot roles are invalid")
    if proof.candidate.params_digest == proof.proposal_active.params_digest:
        raise ValueError("adaptation candidate contains no BPTT parameter change")
    if proof.candidate.base_active_version != proof.proposal_active.version or (
        proof.candidate.base_active_digest != proof.proposal_active.digest
    ):
        raise ValueError("adaptation candidate is not bound to its proposal active snapshot")
    proof.thresholds.validate()
    recomputed = hard_validate_candidate(
        proof.proposal_active,
        proof.candidate,
        proof.evidence,
        proof.thresholds,
        current_model_version=proof.report.model_version,
    )
    if recomputed != proof.report or not proof.report.verify_integrity():
        raise ValueError("adaptation hard-validation report does not recompute")

    if proof.decision_model_version < proof.decision_active.model_version:
        raise ValueError("adaptation decision model version predates its active snapshot")
    store = ActiveSnapshotStore(proof.decision_active)
    if proof.decision_model_version > store.model_version:
        store.advance_model_version(proof.decision_model_version)

    if proof.status == "expired":
        if proof.publication_active.digest != proof.decision_active.digest:
            raise ValueError("expired candidate changed the active snapshot")
        if proof.used_by_executed_control:
            raise ValueError("expired candidate cannot drive an executed control")
    else:
        replayed = store.admit(proof.candidate, proof.report)
        expected_accepted = proof.status == "admitted"
        if replayed.accepted is not expected_accepted:
            raise ValueError("adaptation publication status does not replay")
        if replayed.reason != proof.publication_reason:
            raise ValueError("adaptation publication reason does not replay")
        if replayed.active.digest != proof.publication_active.digest:
            raise ValueError("adaptation publication snapshot does not replay")

    used = bool(
        proof.status == "admitted"
        and np.any(
            trace.executed_control
            & (np.arange(trace.steps, dtype=np.int64) >= proof.boundary_step)
            & (trace.snapshot_version == proof.publication_active.version)
        )
    )
    if proof.used_by_executed_control is not used:
        raise ValueError("adaptation proof executed-control lineage does not match the trace")
    if int(trace.snapshot_version[proof.boundary_step]) != proof.publication_active.version:
        raise ValueError("adaptation proof active snapshot does not match its boundary trace row")


def _validate_numerical_inputs(
    decisions: tuple[AdaptationDecisionProof, ...],
    trace: ImmutableTrace,
    *,
    events: tuple[ArtifactEvent, ...],
    tape: Any,
    condition: Any,
    method: Any,
    config: Any,
) -> None:
    """Reconstruct proposal/held-out batches and hard evidence from the scheduled tape."""
    import jax.numpy as jnp

    from crazyflow.safety.da_plcbf.dynamic_rollouts import dynamic_sphere_window_from_tape
    from crazyflow.safety.da_plcbf.experiments import (
        _VALIDATION_FOLD_NAMES,
        ConditionID,
        _auxiliary_tape,
        _candidate_evidence_device,
        _hard_validation_batch,
        _numeric_digest,
        _replay_dashboard_dynamics_and_contexts,
        _resources_for_tape,
        _scenario_digest,
        _training_batch,
        build_experiment_resources,
    )
    from crazyflow.safety.da_plcbf.library import descriptor_targets_from_spec
    from crazyflow.safety.da_plcbf.quad_actor_bptt import (
        build_dynamic_model_quad_actor_bptt_functions,
    )
    from crazyflow.safety.da_plcbf.quad_actor_losses import QuadLearningConfig
    from crazyflow.safety.da_plcbf.snapshots import create_active_snapshot

    obstacle_count = tape.static_positions.shape[0] + tape.dynamic_positions.shape[1]
    selected_condition = ConditionID(condition)
    resources = _resources_for_tape(
        build_experiment_resources(config, obstacle_count=obstacle_count, initialization_seed=0),
        tape,
        config,
    )
    schema_reference = create_active_snapshot(
        resources.initial_params, structural_core=resources.spec, model_version=0
    )
    replay = _replay_dashboard_dynamics_and_contexts(
        trace, tape, selected_condition, method, config
    )
    controller_models = replay[4]
    model_samples = replay[5]
    heldout = _auxiliary_tape(tape, selected_condition, config, purpose="hard-validation")
    learning = QuadLearningConfig(
        dt=config.dt, horizon=config.certificate_horizon, policy_gain=config.policy_gain
    )
    bptt_by_device: dict[tuple[str, int], Any] = {}
    resources_by_device: dict[tuple[str, int], Any] = {}
    for proof in decisions:
        if proof.proposal_active.params_schema_digest != schema_reference.params_schema_digest:
            raise ValueError("adaptation proposal parameter schema is not the configured actor")
        if proof.proposal_active.structural_core_digest != schema_reference.structural_core_digest:
            raise ValueError("adaptation proposal structural core is not the configured library")
        if proof.candidate.model_version != int(trace.model_version[proof.context_step]):
            raise ValueError("adaptation candidate model version does not match its causal context")
        metadata = proof.candidate.metadata
        expected_metadata = {
            "algorithm": "fixed_budget_truncated_bptt",
            "burst_steps": config.bptt_burst_steps,
            "objective": "plcbf_aligned_coverage_diversity",
            "bptt_execution_scope": "compiled_burst_only",
            "bptt_compilation_excluded_from_execution_timing": True,
        }
        if any(metadata.get(name) != value for name, value in expected_metadata.items()):
            raise ValueError("adaptation candidate BPTT metadata does not match the configuration")
        matching_events = tuple(
            event
            for event in events
            if event.details.get("candidate_digest") == proof.candidate.digest
            and event.name == f"candidate_{proof.status}"
        )
        if len(matching_events) != 1:
            raise ValueError("adaptation BPTT replay has no unique decision event")
        decision_details = matching_events[0].details
        backend = metadata.get("bptt_execution_backend")
        device_id = metadata.get("bptt_execution_device_id")
        if (
            not isinstance(backend, str)
            or not backend
            or isinstance(device_id, bool)
            or not isinstance(device_id, int)
            or device_id < 0
        ):
            raise ValueError("adaptation BPTT execution backend metadata is invalid")
        try:
            matching_devices = tuple(
                device for device in jax.devices(backend) if int(device.id) == device_id
            )
        except (RuntimeError, ValueError) as error:
            raise ValueError("adaptation BPTT execution backend is unavailable") from error
        if len(matching_devices) != 1:
            raise ValueError("adaptation BPTT execution device is unavailable or ambiguous")
        proof_device = matching_devices[0]
        isolated_cpu = decision_details.get("execution_device_is_cpu")
        if not isinstance(isolated_cpu, bool):
            raise ValueError("adaptation event omits the BPTT device-isolation evidence")
        cache_key = metadata.get("bptt_cache_key")
        expected_cache_key = f"online:{backend}:{device_id}" if isolated_cpu else "startup:default"
        if (
            cache_key != expected_cache_key
            or (isolated_cpu and (proof.phase != "online" or backend != "cpu"))
            or decision_details.get("bptt_cache_key") != cache_key
            or decision_details.get("bptt_execution_backend") != backend
            or decision_details.get("bptt_execution_device_id") != device_id
        ):
            raise ValueError("adaptation BPTT execution backend is not proof-bound")

        device_key = (backend, device_id)

        def on_device(tree: Any) -> Any:
            return jax.device_put(tree, proof_device)

        proof_resources = resources_by_device.get(device_key)
        if proof_resources is None:
            proof_resources = replace(
                resources,
                model=on_device(resources.model),
                actuator=on_device(resources.actuator),
                spec=on_device(resources.spec),
                initial_params=on_device(resources.initial_params),
            )
            resources_by_device[device_key] = proof_resources
        bptt = bptt_by_device.get(device_key)
        if bptt is None:
            bptt = build_dynamic_model_quad_actor_bptt_functions(
                proof_resources.spec,
                proof_resources.actuator,
                proof_resources.actor_config,
                proof_resources.quad_config,
                proof_resources.barrier_config,
                learning,
                proof_resources.loss_config,
                burst_steps=config.bptt_burst_steps,
                device=proof_device,
            )
            bptt_by_device[device_key] = bptt

        with jax.default_device(proof_device):
            state = on_device(jnp.asarray(trace.true_state[proof.context_step], dtype=jnp.float32))
            controller_model = on_device(controller_models[proof.context_step])
            samples = on_device(model_samples[proof.context_step])
            initial_states, circles, safety = _training_batch(
                tape, state, proof.context_step, config
            )
            initial_states = on_device(initial_states)
            circles = on_device(circles)
            safety = on_device(safety)
            training_digest = _numeric_digest(
                "scenario-content",
                initial_states,
                circles.obstacle_centers,
                circles.obstacle_radii,
                circles.obstacle_mask,
            )
            if metadata.get("proposal_training_digest") != training_digest:
                raise ValueError(
                    "adaptation proposal-training digest does not replay from the tape"
                )
            active_params = on_device(jax.tree.map(jnp.asarray, proof.proposal_active.params))
            optimizer = on_device(bptt.initialize(active_params))
            descriptor_targets = on_device(descriptor_targets_from_spec(proof_resources.spec))
            descriptor_scales_device = on_device(
                jnp.asarray([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=state.dtype)
            )
            trained, metrics = bptt.burst(
                optimizer,
                initial_states,
                circles,
                safety,
                descriptor_targets,
                active_params,
                descriptor_scales_device,
                controller_model,
            )
        replayed_leaves, replayed_tree = jax.tree_util.tree_flatten(trained.params)
        candidate_leaves, candidate_tree = jax.tree_util.tree_flatten(proof.candidate.params)
        if replayed_tree != candidate_tree or any(
            not np.array_equal(np.asarray(actual), np.asarray(expected))
            for actual, expected in zip(replayed_leaves, candidate_leaves, strict=True)
        ):
            raise ValueError("adaptation candidate parameters do not match deterministic BPTT")
        accepted_updates = np.asarray(metrics.update_accepted, dtype=np.bool_)
        parameter_deltas = np.asarray(metrics.parameter_delta_norm, dtype=np.float64)
        gradient_norms = np.asarray(metrics.gradient_norm, dtype=np.float64)
        if (
            not np.all(accepted_updates)
            or not np.all(np.isfinite(parameter_deltas))
            or not np.all(np.isfinite(gradient_norms))
            or not np.any(parameter_deltas > 0.0)
        ):
            raise ValueError("adaptation deterministic BPTT replay did not produce valid updates")
        replayed_gradient = float(gradient_norms[-1])
        replayed_loss = float(np.asarray(metrics.loss.total[-1]))
        if (
            decision_details.get("update_accepted") != bool(accepted_updates[-1])
            or not np.isclose(
                decision_details.get("gradient_norm", np.nan),
                replayed_gradient,
                rtol=1e-7,
                atol=1e-8,
            )
            or not np.isclose(
                decision_details.get("loss", np.nan), replayed_loss, rtol=1e-7, atol=1e-8
            )
        ):
            raise ValueError("adaptation event BPTT gradient/loss evidence does not replay")
        with jax.default_device(proof_device):
            validation = _hard_validation_batch(
                tape, heldout, state, proof.context_step, controller_model, proof_resources, config
            )
            validation = validation._replace(
                initial_states=on_device(validation.initial_states),
                scenarios=on_device(validation.scenarios),
            )
        if metadata.get("hard_validation_digest") != validation.digest:
            raise ValueError("adaptation hard-validation batch digest does not replay")
        expected_validation_digest = _scenario_digest(
            tape.sha256, proof.candidate.model_version, validation.digest, *_VALIDATION_FOLD_NAMES
        )
        if proof.evidence.validation_set_digest != expected_validation_digest:
            raise ValueError("adaptation validation-set digest is not tape/config bound")
        with jax.default_device(proof_device):
            window = on_device(
                dynamic_sphere_window_from_tape(
                    tape,
                    start_index=proof.context_step,
                    horizon=config.certificate_horizon + 1,
                    speed_limit=config.speed_limit,
                    angular_rate_max=config.angular_rate_max,
                    tilt_max_radians=config.tilt_max_radians,
                )
            )
            candidate_params = on_device(jax.tree.map(jnp.asarray, proof.candidate.params))
        current, candidate_local, active_local, descriptors, feasibility = (
            np.asarray(value)
            for value in _candidate_evidence_device(
                candidate_params,
                active_params,
                proof_resources.spec,
                validation.initial_states,
                validation.scenarios,
                state,
                window,
                controller_model,
                samples,
                proof_resources,
                config,
            )
        )
        descriptor_scales = np.asarray(
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=np.asarray(descriptors).dtype
        )
        observed = (
            proof.evidence.current_policy_margins,
            proof.evidence.candidate_local_policy_margins,
            proof.evidence.active_local_policy_margins,
            proof.evidence.candidate_descriptors,
            proof.evidence.feasibility_margins,
            proof.evidence.descriptor_scales,
        )
        expected = (
            current,
            candidate_local,
            active_local,
            descriptors,
            feasibility,
            descriptor_scales,
        )
        if any(
            np.asarray(actual).shape != np.asarray(replayed).shape
            or not np.allclose(actual, replayed, rtol=3e-6, atol=3e-6)
            for actual, replayed in zip(observed, expected, strict=True)
        ):
            raise ValueError("adaptation held-out hard evidence does not numerically replay")


def save_adaptation_evidence(
    evidence: AdaptationEvidence, path: str | Path, *, overwrite: bool = False
) -> str:
    """Atomically save deterministic proof evidence and return its semantic digest."""
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("adaptation evidence path must end in .npz")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    arrays = _evidence_arrays(evidence)
    arrays["content_sha256"] = np.asarray(_canonical_array_digest(_PREFIX, arrays), dtype="<U64")
    _atomic_write_bytes(destination, _deterministic_npz_bytes(arrays), overwrite=overwrite)
    return str(arrays["content_sha256"])


def load_adaptation_evidence(path: str | Path) -> AdaptationEvidence:
    """Strictly load, reconstruct, and digest-check online adaptation evidence."""
    source = Path(path)
    loaded = _load_strict_npz(source, _EXPECTED_ARRAYS)
    recorded = _scalar_text(loaded.pop("content_sha256"), "content_sha256")
    actual = _canonical_array_digest(_PREFIX, loaded)
    if recorded != actual:
        raise ValueError("adaptation evidence content digest mismatch")
    evidence = _evidence_from_arrays(loaded)
    if evidence.content_sha256 != recorded:
        raise ValueError("adaptation evidence semantic reconstruction mismatch")
    return evidence


def _snapshot_schema(snapshots: tuple[PolicySnapshot, ...]) -> dict[str, Any]:
    first = snapshots[0]
    params_proto = first._params_treedef.serialize_using_proto().hex()
    core_proto = first._core_treedef.serialize_using_proto().hex()
    result = {
        "params_treedef_proto": params_proto,
        "params_leaves": [
            {"dtype": leaf.dtype, "shape": list(leaf.shape)} for leaf in first._params_leaves
        ],
        "core_treedef_proto": core_proto,
        "core_leaves": [
            {"dtype": leaf.dtype, "shape": list(leaf.shape)} for leaf in first._core_leaves
        ],
    }
    for snapshot in snapshots[1:]:
        other = {
            "params_treedef_proto": snapshot._params_treedef.serialize_using_proto().hex(),
            "params_leaves": [
                {"dtype": leaf.dtype, "shape": list(leaf.shape)} for leaf in snapshot._params_leaves
            ],
            "core_treedef_proto": snapshot._core_treedef.serialize_using_proto().hex(),
            "core_leaves": [
                {"dtype": leaf.dtype, "shape": list(leaf.shape)} for leaf in snapshot._core_leaves
            ],
        }
        if other != result:
            raise ValueError("adaptation proof snapshots do not share one exact numeric schema")
    return result


def _flatten_snapshot_bytes(snapshot: PolicySnapshot, attribute: str) -> np.ndarray:
    leaves = getattr(snapshot, attribute)
    return np.frombuffer(b"".join(leaf.data for leaf in leaves), dtype=np.uint8).copy()


def _evidence_arrays(evidence: AdaptationEvidence) -> dict[str, np.ndarray]:
    decisions = evidence.decisions
    if not decisions:
        raise ValueError("online adaptation evidence must contain decisions")
    rows = len(decisions)
    snapshots = tuple(
        snapshot
        for decision in decisions
        for snapshot in (
            decision.proposal_active,
            decision.decision_active,
            decision.candidate,
            decision.publication_active,
        )
    )
    schema = _snapshot_schema(snapshots)
    grouped = [snapshots[index * 4 : (index + 1) * 4] for index in range(rows)]
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(ADAPTATION_EVIDENCE_SCHEMA_VERSION, dtype=np.uint16),
        "trace_content_sha256": np.asarray(evidence.trace_content_sha256, dtype="<U64"),
        "snapshot_schema_json": np.asarray(
            json.dumps(schema, sort_keys=True, separators=(",", ":"))
        ),
        "phase": np.asarray([item.phase for item in decisions], dtype="<U16"),
        "job_id": np.asarray([item.job_id for item in decisions], dtype=np.int64),
        "context_step": np.asarray([item.context_step for item in decisions], dtype=np.int64),
        "boundary_step": np.asarray([item.boundary_step for item in decisions], dtype=np.int64),
        "status": np.asarray([item.status for item in decisions], dtype="<U16"),
        "decision_model_version": np.asarray(
            [item.decision_model_version for item in decisions], dtype=np.int64
        ),
        "publication_reason": np.asarray(
            [item.publication_reason for item in decisions], dtype="<U256"
        ),
        "used_by_executed_control": np.asarray(
            [item.used_by_executed_control for item in decisions], dtype=np.bool_
        ),
        "snapshot_kind": np.asarray(
            [[snapshot.kind for snapshot in group] for group in grouped], dtype="<U16"
        ),
        "snapshot_version": np.asarray(
            [[snapshot.version for snapshot in group] for group in grouped], dtype=np.int64
        ),
        "snapshot_base_active_version": np.asarray(
            [[snapshot.base_active_version for snapshot in group] for group in grouped],
            dtype=np.int64,
        ),
        "snapshot_base_active_digest": np.asarray(
            [[snapshot.base_active_digest for snapshot in group] for group in grouped], dtype="<U64"
        ),
        "snapshot_model_version": np.asarray(
            [[snapshot.model_version for snapshot in group] for group in grouped], dtype=np.int64
        ),
        "snapshot_digest": np.asarray(
            [[snapshot.digest for snapshot in group] for group in grouped], dtype="<U64"
        ),
        "snapshot_metadata_json": np.asarray(
            [[snapshot._metadata_json for snapshot in group] for group in grouped]
        ),
        "snapshot_params_bytes": np.stack(
            [
                np.stack(
                    [_flatten_snapshot_bytes(snapshot, "_params_leaves") for snapshot in group]
                )
                for group in grouped
            ]
        ),
        "snapshot_core_bytes": np.stack(
            [
                np.stack([_flatten_snapshot_bytes(snapshot, "_core_leaves") for snapshot in group])
                for group in grouped
            ]
        ),
        "current_policy_margins": _stack_evidence(decisions, "current_policy_margins"),
        "candidate_local_policy_margins": _stack_evidence(
            decisions, "candidate_local_policy_margins"
        ),
        "active_local_policy_margins": _stack_evidence(decisions, "active_local_policy_margins"),
        "candidate_descriptors": _stack_evidence(decisions, "candidate_descriptors"),
        "descriptor_scales": _stack_evidence(decisions, "descriptor_scales"),
        "feasibility_margins": _stack_evidence(decisions, "feasibility_margins"),
        "runtime_seconds": _stack_evidence(decisions, "runtime_seconds"),
        "validation_set_digest": np.asarray(
            [item.evidence.validation_set_digest for item in decisions], dtype="<U64"
        ),
        "threshold_floats": np.asarray(
            [
                (
                    item.thresholds.minimum_current_margin,
                    item.thresholds.safe_policy_margin,
                    item.thresholds.local_non_regression_tolerance,
                    item.thresholds.minimum_coverage,
                    item.thresholds.minimum_diversity,
                    item.thresholds.minimum_feasible_fraction,
                    item.thresholds.maximum_runtime_seconds,
                )
                for item in decisions
            ],
            dtype=np.float64,
        ),
        "threshold_minimum_redundancy": np.asarray(
            [item.thresholds.minimum_redundancy for item in decisions], dtype=np.int64
        ),
        "report_active_digest": np.asarray(
            [item.report.active_digest for item in decisions], dtype="<U64"
        ),
        "report_active_version": np.asarray(
            [item.report.active_version for item in decisions], dtype=np.int64
        ),
        "report_candidate_digest": np.asarray(
            [item.report.candidate_digest for item in decisions], dtype="<U64"
        ),
        "report_candidate_version": np.asarray(
            [item.report.candidate_version for item in decisions], dtype=np.int64
        ),
        "report_model_version": np.asarray(
            [item.report.model_version for item in decisions], dtype=np.int64
        ),
        "report_validation_set_digest": np.asarray(
            [item.report.validation_set_digest for item in decisions], dtype="<U64"
        ),
        "report_gate_passed": np.asarray(
            [[gate.passed for gate in item.report.gates] for item in decisions], dtype=np.bool_
        ),
        "report_gate_observed": np.asarray(
            [[gate.observed for gate in item.report.gates] for item in decisions], dtype="<U512"
        ),
        "report_gate_requirement": np.asarray(
            [[gate.requirement for gate in item.report.gates] for item in decisions], dtype="<U512"
        ),
        "report_gate_detail": np.asarray(
            [[gate.detail for gate in item.report.gates] for item in decisions], dtype="<U512"
        ),
        "report_candidate_local_best": np.asarray(
            [item.report.candidate_local_best for item in decisions], dtype=np.float64
        ),
        "report_active_local_best": np.asarray(
            [item.report.active_local_best for item in decisions], dtype=np.float64
        ),
        "report_local_non_regression_passes": np.asarray(
            [item.report.local_non_regression_passes for item in decisions], dtype=np.bool_
        ),
        "report_digest": np.asarray([item.report.digest for item in decisions], dtype="<U64"),
    }
    return arrays


def _stack_evidence(decisions: tuple[AdaptationDecisionProof, ...], attribute: str) -> np.ndarray:
    values = [np.asarray(getattr(item.evidence, attribute)) for item in decisions]
    if any(value.shape != values[0].shape or value.dtype != values[0].dtype for value in values):
        raise ValueError(f"adaptation evidence field {attribute} changed schema within one trial")
    return np.stack(values)


def _evidence_from_arrays(arrays: dict[str, np.ndarray]) -> AdaptationEvidence:
    schema_version = arrays["schema_version"]
    if schema_version.shape != () or schema_version.dtype != np.uint16 or int(schema_version) != 1:
        raise ValueError("adaptation evidence schema version is invalid")
    trace_digest = _scalar_text(arrays["trace_content_sha256"], "trace_content_sha256")
    schema = _parse_schema(_scalar_text(arrays["snapshot_schema_json"], "snapshot_schema_json"))
    rows = arrays["phase"].shape[0]
    if rows < 1:
        raise ValueError("adaptation evidence contains no decisions")
    decisions: list[AdaptationDecisionProof] = []
    for row in range(rows):
        snapshots = tuple(_snapshot_from_arrays(arrays, schema, row, role) for role in range(4))
        raw_thresholds = arrays["threshold_floats"][row]
        if raw_thresholds.shape != (7,):
            raise ValueError("adaptation threshold evidence has an invalid shape")
        thresholds = HardValidationThresholds(
            minimum_current_margin=float(raw_thresholds[0]),
            safe_policy_margin=float(raw_thresholds[1]),
            local_non_regression_tolerance=float(raw_thresholds[2]),
            minimum_coverage=float(raw_thresholds[3]),
            minimum_redundancy=int(arrays["threshold_minimum_redundancy"][row]),
            minimum_diversity=float(raw_thresholds[4]),
            minimum_feasible_fraction=float(raw_thresholds[5]),
            maximum_runtime_seconds=float(raw_thresholds[6]),
        )
        raw_evidence = HardValidationEvidence(
            current_policy_margins=arrays["current_policy_margins"][row],
            candidate_local_policy_margins=arrays["candidate_local_policy_margins"][row],
            active_local_policy_margins=arrays["active_local_policy_margins"][row],
            candidate_descriptors=arrays["candidate_descriptors"][row],
            descriptor_scales=arrays["descriptor_scales"][row],
            feasibility_margins=arrays["feasibility_margins"][row],
            runtime_seconds=arrays["runtime_seconds"][row],
            validation_set_digest=str(arrays["validation_set_digest"][row]),
        )
        gates = tuple(
            GateResult(
                name=name,
                passed=bool(arrays["report_gate_passed"][row, gate_index]),
                observed=str(arrays["report_gate_observed"][row, gate_index]),
                requirement=str(arrays["report_gate_requirement"][row, gate_index]),
                detail=str(arrays["report_gate_detail"][row, gate_index]),
            )
            for gate_index, name in enumerate(GATE_NAMES)
        )
        report = ValidationReport(
            active_digest=str(arrays["report_active_digest"][row]),
            active_version=int(arrays["report_active_version"][row]),
            candidate_digest=str(arrays["report_candidate_digest"][row]),
            candidate_version=int(arrays["report_candidate_version"][row]),
            model_version=int(arrays["report_model_version"][row]),
            validation_set_digest=str(arrays["report_validation_set_digest"][row]),
            gates=gates,
            candidate_local_best=tuple(
                float(value) for value in arrays["report_candidate_local_best"][row]
            ),
            active_local_best=tuple(
                float(value) for value in arrays["report_active_local_best"][row]
            ),
            local_non_regression_passes=tuple(
                bool(value) for value in arrays["report_local_non_regression_passes"][row]
            ),
            digest=str(arrays["report_digest"][row]),
        )
        decisions.append(
            AdaptationDecisionProof(
                phase=str(arrays["phase"][row]),  # type: ignore[arg-type]
                job_id=int(arrays["job_id"][row]),
                context_step=int(arrays["context_step"][row]),
                boundary_step=int(arrays["boundary_step"][row]),
                status=str(arrays["status"][row]),  # type: ignore[arg-type]
                decision_model_version=int(arrays["decision_model_version"][row]),
                publication_reason=str(arrays["publication_reason"][row]),
                used_by_executed_control=bool(arrays["used_by_executed_control"][row]),
                proposal_active=snapshots[0],
                decision_active=snapshots[1],
                candidate=snapshots[2],
                publication_active=snapshots[3],
                evidence=raw_evidence,
                thresholds=thresholds,
                report=report,
            )
        )
    return AdaptationEvidence(trace_content_sha256=trace_digest, decisions=tuple(decisions))


def _parse_schema(raw: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key in adaptation snapshot schema")
        return dict(pairs)

    try:
        value = json.loads(raw, object_pairs_hook=pairs_hook)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("adaptation snapshot schema is invalid JSON") from error
    if not isinstance(value, dict) or set(value) != {
        "params_treedef_proto",
        "params_leaves",
        "core_treedef_proto",
        "core_leaves",
    }:
        raise ValueError("adaptation snapshot schema fields are invalid")
    return value


def _snapshot_from_arrays(
    arrays: dict[str, np.ndarray], schema: dict[str, Any], row: int, role: int
) -> PolicySnapshot:
    params = _unpack_leaves(arrays["snapshot_params_bytes"][row, role], schema["params_leaves"])
    core = _unpack_leaves(arrays["snapshot_core_bytes"][row, role], schema["core_leaves"])
    try:
        params_treedef = jax.tree_util.PyTreeDef.deserialize_using_proto(
            jax.tree_util.default_registry, bytes.fromhex(schema["params_treedef_proto"])
        )
        core_treedef = jax.tree_util.PyTreeDef.deserialize_using_proto(
            jax.tree_util.default_registry, bytes.fromhex(schema["core_treedef_proto"])
        )
    except (TypeError, ValueError) as error:
        raise ValueError("adaptation snapshot PyTree schema is invalid") from error
    snapshot = PolicySnapshot(
        kind=str(arrays["snapshot_kind"][row, role]),  # type: ignore[arg-type]
        version=int(arrays["snapshot_version"][row, role]),
        base_active_version=int(arrays["snapshot_base_active_version"][row, role]),
        base_active_digest=str(arrays["snapshot_base_active_digest"][row, role]),
        model_version=int(arrays["snapshot_model_version"][row, role]),
        digest=str(arrays["snapshot_digest"][row, role]),
        _params_treedef=params_treedef,
        _params_leaves=params,
        _core_treedef=core_treedef,
        _core_leaves=core,
        _metadata_json=str(arrays["snapshot_metadata_json"][row, role]),
    )
    if not snapshot.verify_integrity():
        raise ValueError(f"adaptation snapshot role {_SNAPSHOT_ROLES[role]} failed integrity")
    return snapshot


def _unpack_leaves(raw: np.ndarray, schema: Any) -> tuple[_FrozenLeaf, ...]:
    if raw.dtype != np.uint8 or raw.ndim != 1 or not isinstance(schema, list):
        raise ValueError("adaptation snapshot byte payload/schema is invalid")
    offset = 0
    leaves: list[_FrozenLeaf] = []
    payload = raw.tobytes()
    for item in schema:
        if not isinstance(item, dict) or set(item) != {"dtype", "shape"}:
            raise ValueError("adaptation snapshot leaf schema is invalid")
        dtype = np.dtype(item["dtype"])
        shape = tuple(int(value) for value in item["shape"])
        if dtype.kind not in "biufc" or any(value < 0 for value in shape):
            raise ValueError("adaptation snapshot leaf dtype/shape is invalid")
        size = int(dtype.itemsize * np.prod(shape, dtype=np.int64))
        leaves.append(_FrozenLeaf(dtype.str, shape, payload[offset : offset + size]))
        offset += size
    if offset != len(payload):
        raise ValueError("adaptation snapshot byte payload length is invalid")
    return tuple(leaves)


def _scalar_text(value: np.ndarray, name: str) -> str:
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar Unicode string")
    return str(value)


_EXPECTED_ARRAYS = {
    "schema_version",
    "trace_content_sha256",
    "snapshot_schema_json",
    "phase",
    "job_id",
    "context_step",
    "boundary_step",
    "status",
    "decision_model_version",
    "publication_reason",
    "used_by_executed_control",
    "snapshot_kind",
    "snapshot_version",
    "snapshot_base_active_version",
    "snapshot_base_active_digest",
    "snapshot_model_version",
    "snapshot_digest",
    "snapshot_metadata_json",
    "snapshot_params_bytes",
    "snapshot_core_bytes",
    "current_policy_margins",
    "candidate_local_policy_margins",
    "active_local_policy_margins",
    "candidate_descriptors",
    "descriptor_scales",
    "feasibility_margins",
    "runtime_seconds",
    "validation_set_digest",
    "threshold_floats",
    "threshold_minimum_redundancy",
    "report_active_digest",
    "report_active_version",
    "report_candidate_digest",
    "report_candidate_version",
    "report_model_version",
    "report_validation_set_digest",
    "report_gate_passed",
    "report_gate_observed",
    "report_gate_requirement",
    "report_gate_detail",
    "report_candidate_local_best",
    "report_active_local_best",
    "report_local_non_regression_passes",
    "report_digest",
    "content_sha256",
}


__all__ = [
    "ADAPTATION_EVIDENCE_SCHEMA_VERSION",
    "AdaptationDecisionProof",
    "AdaptationEvidence",
    "CandidateValidationMaterial",
    "load_adaptation_evidence",
    "save_adaptation_evidence",
    "validate_adaptation_evidence_binding",
]
