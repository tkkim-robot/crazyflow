from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from crazyflow.safety.da_plcbf.artifact_smoke import synthetic_trace
from crazyflow.safety.da_plcbf.dashboard_evidence import (
    DASHBOARD_EVIDENCE_SCHEMA_VERSION,
    DashboardEvidence,
    _prediction_evidence_from_tape,
    load_dashboard_evidence,
    save_dashboard_evidence,
    validate_dashboard_evidence_binding,
)
from crazyflow.safety.da_plcbf.scenarios import ScenarioTapeConfig, generate_scenario_tape
from crazyflow.safety.da_plcbf.scientific_dashboard import (
    evidence_inventory,
    render_scientific_dashboard,
    scientific_dashboard_frames,
    verify_scientific_dashboard_replay,
)

if TYPE_CHECKING:
    from pathlib import Path

    from crazyflow.safety.da_plcbf.artifacts import ImmutableTrace
    from crazyflow.safety.da_plcbf.scenarios import ScenarioTape


def _evidence() -> tuple[DashboardEvidence, ImmutableTrace, ScenarioTape]:
    tape = generate_scenario_tape(913, ScenarioTapeConfig(steps=8, dt=0.1), fold=8)
    trace = synthetic_trace(tape.sha256, steps=6, dt=0.1)
    steps = trace.steps
    policies = len(trace.policy_names)
    rollout_time = np.linspace(0.0, 0.3, 4)
    base = (
        trace.true_state[:, None, :3]
        + np.stack(
            (np.linspace(0.0, 0.15, 4), np.linspace(0.0, -0.08, 4), np.linspace(0.0, 0.05, 4)),
            axis=1,
        )[None, :, :]
    )
    nominal = np.array(base, copy=True)
    fallback = np.broadcast_to(base[:, None, :, :], (steps, policies, 4, 3)).copy()
    fallback += np.linspace(-0.06, 0.06, policies)[None, :, None, None]
    selected = np.stack(
        [fallback[step, trace.selected_policy[step]] for step in range(steps)], axis=0
    )
    ghost_names = np.asarray(("pre_change", "post_change", "adapted"))
    ghosts = np.broadcast_to(base[:, None, :, :], (steps, 3, 4, 3)).copy()
    ghosts += np.asarray((-0.04, 0.04, 0.0))[None, :, None, None]

    prediction, prediction_available, prediction_time = _prediction_evidence_from_tape(
        tape, steps=steps, prediction_nodes=3
    )

    descriptors = np.linspace(-1.0, 1.0, steps * policies * 3).reshape(steps, policies, 3)
    dynamics_true = np.stack(
        (
            np.linspace(1.0, 1.3, steps),
            np.linspace(1.0, 0.8, steps),
            np.linspace(0.0, 0.7, steps),
            np.linspace(1.0, 0.65, steps),
        ),
        axis=1,
    )
    dynamics_estimated = dynamics_true + 0.03
    uncertainty = dynamics_estimated[:, None, :] + np.asarray((-0.04, 0.0, 0.04))[None, :, None]
    candidate_present = np.zeros(steps, dtype=np.bool_)
    candidate_admitted = np.zeros(steps, dtype=np.bool_)
    candidate_rejected = np.zeros(steps, dtype=np.bool_)
    candidate_present[[3, 5]] = True
    candidate_admitted[3] = True
    candidate_rejected[5] = True
    admission_margin = np.zeros(steps)
    admission_margin[[3, 5]] = (0.08, -0.03)
    reason_index = np.full(steps, -1, dtype=np.int16)
    reason_index[[3, 5]] = (0, 1)
    bptt = np.stack(
        (
            np.linspace(0.001, 0.0014, steps),
            np.linspace(0.002, 0.0025, steps),
            np.linspace(0.0005, 0.0008, steps),
        ),
        axis=1,
    )
    evidence = DashboardEvidence(
        schema_version=np.asarray(DASHBOARD_EVIDENCE_SCHEMA_VERSION, dtype=np.uint16),
        trace_content_sha256=np.asarray(trace.content_sha256),
        scenario_tape_sha256=np.asarray(tape.sha256),
        policy_names=np.asarray(trace.policy_names),
        rollout_time=rollout_time,
        nominal_rollout_positions=nominal,
        nominal_rollout_available=np.ones(steps, dtype=np.bool_),
        fallback_rollout_positions=fallback,
        fallback_rollout_available=np.ones((steps, policies), dtype=np.bool_),
        selected_rollout_positions=selected,
        selected_rollout_available=np.ones(steps, dtype=np.bool_),
        ghost_rollout_names=ghost_names,
        ghost_rollout_positions=ghosts,
        ghost_rollout_available=np.ones((steps, len(ghost_names)), dtype=np.bool_),
        prediction_time=prediction_time,
        prediction_positions=prediction,
        prediction_available=prediction_available,
        descriptor_names=np.asarray(("endpoint", "path_length", "turning")),
        normalized_descriptors=descriptors,
        descriptor_available=np.ones((steps, policies), dtype=np.bool_),
        dynamics_parameter_names=np.asarray(
            ("mass_scale", "drag_scale", "wind_norm", "minimum_rotor_efficiency")
        ),
        dynamics_true=dynamics_true,
        dynamics_true_available=np.ones_like(dynamics_true, dtype=np.bool_),
        dynamics_estimated=dynamics_estimated,
        dynamics_estimated_available=np.ones_like(dynamics_estimated, dtype=np.bool_),
        dynamics_uncertainty_samples=uncertainty,
        dynamics_uncertainty_available=np.ones((steps, 3), dtype=np.bool_),
        admission_recorded=np.asarray(True),
        candidate_present=candidate_present,
        candidate_admitted=candidate_admitted,
        candidate_rejected=candidate_rejected,
        admission_margin=admission_margin,
        admission_reason_names=np.asarray(("hard-gates-passed", "non-regression-failed")),
        admission_reason_index=reason_index,
        bptt_timing_names=np.asarray(("forward", "backward", "validation")),
        bptt_timing_seconds=bptt,
        bptt_timing_available=np.ones_like(bptt, dtype=np.bool_),
    )
    return evidence, trace, tape


def _archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def test_dashboard_evidence_is_deterministic_immutable_and_strictly_bound(tmp_path: Path) -> None:
    evidence, trace, tape = _evidence()
    validate_dashboard_evidence_binding(evidence, trace, tape)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    assert save_dashboard_evidence(evidence, first) == evidence.content_sha256
    assert save_dashboard_evidence(evidence, second) == evidence.content_sha256
    assert first.read_bytes() == second.read_bytes()
    loaded = load_dashboard_evidence(first)
    assert loaded.content_sha256 == evidence.content_sha256
    for item in fields(evidence):
        np.testing.assert_array_equal(getattr(loaded, item.name), getattr(evidence, item.name))
        assert not getattr(loaded, item.name).flags.writeable

    wrong_trace = synthetic_trace("a" * 64, steps=trace.steps, dt=0.1)
    with pytest.raises(ValueError, match="trace digest"):
        validate_dashboard_evidence_binding(evidence, wrong_trace)
    wrong_tape = generate_scenario_tape(914, ScenarioTapeConfig(steps=8, dt=0.1), fold=8)
    with pytest.raises(ValueError, match="tape"):
        validate_dashboard_evidence_binding(evidence, trace, wrong_tape)


def test_dashboard_evidence_rejects_tamper_shapes_and_hidden_unavailable_values(
    tmp_path: Path,
) -> None:
    evidence, _, _ = _evidence()
    path = tmp_path / "valid.npz"
    save_dashboard_evidence(evidence, path)
    tampered = _archive(path)
    tampered["normalized_descriptors"][0, 0, 0] += 0.5
    tamper_path = tmp_path / "tampered.npz"
    np.savez(tamper_path, **tampered)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_dashboard_evidence(tamper_path)

    with pytest.raises(ValueError, match="nominal_rollout_positions"):
        replace(evidence, nominal_rollout_positions=evidence.nominal_rollout_positions[:, :-1])

    available = np.array(evidence.nominal_rollout_available, copy=True)
    available[0] = False
    with pytest.raises(ValueError, match="zero behind"):
        replace(evidence, nominal_rollout_available=available)

    rejected = np.array(evidence.candidate_rejected, copy=True)
    rejected[4] = True
    with pytest.raises(ValueError, match="candidate_present"):
        replace(evidence, candidate_rejected=rejected)


def test_dashboard_evidence_requires_exact_members_and_digest_binding(tmp_path: Path) -> None:
    evidence, _, _ = _evidence()
    path = tmp_path / "valid.npz"
    save_dashboard_evidence(evidence, path)
    missing = _archive(path)
    del missing["prediction_positions"]
    missing_path = tmp_path / "missing.npz"
    np.savez(missing_path, **missing)
    with pytest.raises(ValueError, match="missing, duplicate, or unexpected"):
        load_dashboard_evidence(missing_path)

    wrong_digest = replace(evidence, trace_content_sha256=np.asarray("f" * 64))
    with pytest.raises(ValueError, match="trace digest"):
        validate_dashboard_evidence_binding(wrong_digest, _evidence()[1])


def test_dashboard_evidence_recomputes_prediction_positions_and_masks_from_tape() -> None:
    evidence, trace, tape = _evidence()
    position = np.array(evidence.prediction_positions, copy=True)
    available_index = tuple(np.argwhere(evidence.prediction_available)[0])
    position[(*available_index, 0)] += 0.01
    with pytest.raises(ValueError, match="prediction positions"):
        validate_dashboard_evidence_binding(
            replace(evidence, prediction_positions=position), trace, tape
        )

    available = np.array(evidence.prediction_available, copy=True)
    available[available_index] = False
    position = np.array(evidence.prediction_positions, copy=True)
    position[available_index] = 0.0
    with pytest.raises(ValueError, match="prediction availability"):
        validate_dashboard_evidence_binding(
            replace(evidence, prediction_positions=position, prediction_available=available),
            trace,
            tape,
        )


def test_scientific_dashboard_uses_recorded_sidecar_instead_of_unavailable_labels() -> None:
    evidence, trace, tape = _evidence()
    inventory = {item.key: item for item in evidence_inventory(trace, tape, sidecar=evidence)}
    for key in (
        "nominal_rollout",
        "fallback_rollouts",
        "selected_rollout",
        "prediction_tubes",
        "descriptors",
        "dynamics_truth",
        "dynamics_estimate",
        "dynamics_uncertainty",
        "candidate_admission",
        "bptt_time",
        "ghost_rollouts",
    ):
        assert inventory[key].status == "recorded"
    frame = next(scientific_dashboard_frames(trace, tape=tape, sidecar=evidence, size=(1280, 720)))
    assert frame.shape == (720, 1280, 3)
    assert np.unique(frame.reshape(-1, 3), axis=0).shape[0] > 100


@pytest.mark.render
def test_sidecar_dashboard_h264_replay_is_deterministic(tmp_path: Path) -> None:
    evidence, trace, tape = _evidence()
    video = tmp_path / "sidecar-dashboard.mp4"
    rendered = render_scientific_dashboard(
        trace, video, tape=tape, sidecar=evidence, fps=5.0, size=(1280, 720)
    )
    replay = verify_scientific_dashboard_replay(
        trace, video, tape=tape, sidecar=evidence, fps=5.0, size=(1280, 720)
    )
    assert replay.validation.decoded_frames_sha256 == rendered.validation.decoded_frames_sha256
