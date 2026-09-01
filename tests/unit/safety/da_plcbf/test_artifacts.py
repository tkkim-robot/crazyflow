from __future__ import annotations

import json
import struct
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from crazyflow.safety.da_plcbf.artifact_smoke import synthetic_trace
from crazyflow.safety.da_plcbf.artifacts import (
    ArtifactEvent,
    ImmutableTrace,
    _artifact_role,
    _validate_final_core_video_coverage,
    _validate_review_frame_artifacts,
    aggregate_row,
    derive_metrics,
    derive_timing,
    file_sha256,
    load_events,
    load_metrics,
    load_paired_metrics_csv,
    load_timing,
    load_trace,
    review_contact_sheet_title,
    save_trace,
    validate_campaign_visual_reviews,
    validate_metrics,
    validate_provenance,
    validate_run_config,
    validate_seeds,
    validate_timing,
    validate_trace_scenario_binding,
    verify_sha256sums,
    write_confidence_intervals,
    write_events,
    write_metrics,
    write_paired_metrics_csv,
    write_seeds,
    write_sha256sums,
    write_timing,
)
from crazyflow.safety.da_plcbf.dashboard import render_dashboard
from crazyflow.safety.da_plcbf.scenarios import (
    ScenarioTapeConfig,
    generate_scenario_tape,
    save_scenario_tape,
)
from crazyflow.safety.da_plcbf.scientific_dashboard import (
    VisualReviewCheck,
    VisualReviewRecord,
    extract_keyframes,
    render_contact_sheet,
    write_visual_review_record,
)

if TYPE_CHECKING:
    from pathlib import Path


def _trace() -> ImmutableTrace:
    return synthetic_trace("a" * 64, steps=12, dt=0.05)


def _archive_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _provenance() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "git": {"commit": "1" * 40, "branch": "plcbf", "dirty": True},
        "runtime": {
            "python": "3.14.0",
            "implementation": "CPython",
            "platform": "test-linux",
            "machine": "x86_64",
        },
        "hardware": {
            "cpu": "test-cpu",
            "gpus": [
                {
                    "index": 0,
                    "name": "RTX 4090",
                    "driver_version": "test-driver",
                    "memory_total_bytes": 24_000_000_000,
                    "uuid": "GPU-test",
                }
            ],
        },
        "jax": {
            "version": "0.11.1",
            "jaxlib_version": "0.11.1",
            "backend": "gpu",
            "jax_enable_x64": False,
            "devices": ["CUDA:0", "TFRT_CPU_0"],
            "cpu_devices": ["TFRT_CPU_0"],
            "role_devices": {
                "controller": "CUDA:0",
                "plant": "CUDA:0",
                "estimator": "TFRT_CPU_0",
                "bptt": "CUDA:0",
                "validation": "CUDA:0",
            },
        },
        "packages": {
            "crazyflow": "0.0.1",
            "numpy": "2.4.2",
            "scipy": "1.17.0",
            "jax": "0.11.1",
            "jaxlib": "0.11.1",
            "flax": "0.12.2",
            "optax": "0.2.6",
            "imageio": "2.37.2",
            "imageio-ffmpeg": "0.6.0",
        },
        "video": {
            "backend": "imageio-ffmpeg",
            "package_version": "0.6.0",
            "encoder_executable": "/pinned/ffmpeg",
            "encoder_sha256": "2" * 64,
            "encoder_version": "ffmpeg version 7.0.2",
            "codec": "libx264",
            "codec_library_version": "libavcodec 61.3.100",
        },
    }


def test_trace_round_trip_is_deterministic_content_addressed_and_immutable(tmp_path: Path) -> None:
    trace = _trace()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    assert save_trace(trace, first) == trace.content_sha256
    assert save_trace(trace, second) == trace.content_sha256
    assert first.read_bytes() == second.read_bytes()

    restored = load_trace(first)
    assert restored.content_sha256 == trace.content_sha256
    for item in fields(trace):
        np.testing.assert_array_equal(getattr(restored, item.name), getattr(trace, item.name))
        assert not getattr(restored, item.name).flags.writeable
    with pytest.raises(ValueError):
        restored.true_state[0, 0] = 10.0
    with pytest.raises(FileExistsError):
        save_trace(trace, first)


def test_trace_loader_rejects_digest_tampering_missing_and_nonfinite_members(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.npz"
    save_trace(_trace(), valid)

    bad_digest = _archive_payload(valid)
    bad_digest["content_sha256"] = np.asarray("0" * 64)
    digest_path = tmp_path / "digest.npz"
    np.savez(digest_path, **bad_digest)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_trace(digest_path)

    nonfinite = _archive_payload(valid)
    nonfinite["hard_barriers"][0, 0] = np.nan
    nonfinite_path = tmp_path / "nonfinite.npz"
    np.savez(nonfinite_path, **nonfinite)
    with pytest.raises(ValueError, match="schema validation"):
        load_trace(nonfinite_path)

    missing = _archive_payload(valid)
    del missing["failure"]
    missing_path = tmp_path / "missing.npz"
    np.savez(missing_path, **missing)
    with pytest.raises(ValueError, match="missing, duplicate, or unexpected"):
        load_trace(missing_path)


def test_trace_schema_forces_constraint_violations_and_contact_into_failure() -> None:
    trace = _trace()
    barriers = np.array(trace.hard_barriers, copy=True)
    barriers[2, 0] = -1e-9
    with pytest.raises(ValueError, match="negative hard barrier"):
        replace(trace, hard_barriers=barriers)

    contact = np.array(trace.contact, copy=True)
    contact[3] = True
    with pytest.raises(ValueError, match="contact is a hard failure"):
        replace(trace, contact=contact)

    failure = np.array(trace.failure, copy=True)
    failure[2:4] = True
    repaired = replace(trace, hard_barriers=barriers, contact=contact, failure=failure)
    repaired.validate()


def test_config_seed_and_provenance_schemas_reject_extra_nonfinite_or_unpinned_values() -> None:
    config = {
        "schema_version": 1,
        "experiment_id": "test",
        "description": "strict test",
        "control_dt_seconds": 0.01,
        "horizon_steps": 20,
        "paired_trials": True,
        "trials_per_condition": 100,
        "methods": ["nominal", "da-plcbf"],
        "conditions": ["static"],
        "parameters": {"alpha": 2.0},
    }
    assert validate_run_config(config) == config
    with pytest.raises(ValueError, match="extra keys"):
        validate_run_config({**config, "silent": True})
    with pytest.raises(ValueError, match="non-finite"):
        validate_run_config({**config, "parameters": {"alpha": float("nan")}})

    seeds = {
        "schema_version": 1,
        "root_seed": 3,
        "folds": [0, 1],
        "named_streams": {"scenario": 42},
        "scenario_tapes": [
            {
                "condition": "static",
                "fold": 0,
                "path": "scenario_tapes/static/0.npz",
                "content_sha256": "a" * 64,
            },
            {
                "condition": "static",
                "fold": 1,
                "path": "scenario_tapes/static/1.npz",
                "content_sha256": "b" * 64,
            },
        ],
        "pairing_id": "paired-v1",
    }
    assert validate_seeds(seeds) == seeds
    assert (
        validate_trace_scenario_binding(_trace(), condition="static", fold=0, seeds=seeds)
        == seeds["scenario_tapes"][0]
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_trace_scenario_binding(_trace(), condition="static", fold=1, seeds=seeds)
    with pytest.raises(ValueError, match="unique"):
        validate_seeds({**seeds, "folds": [0, 0]})
    shared = {
        **seeds,
        "scenario_tapes": [
            {
                "condition": condition,
                "fold": fold,
                "path": f"scenario_tapes/{fold}.npz",
                "content_sha256": "c" * 64,
            }
            for condition in ("static", "wind")
            for fold in (0, 1)
        ],
    }
    assert validate_seeds(shared) == shared
    ambiguous = json.loads(json.dumps(shared))
    ambiguous["scenario_tapes"][2]["content_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="ambiguous"):
        validate_seeds(ambiguous)
    bad_path = json.loads(json.dumps(seeds))
    bad_path["scenario_tapes"][0]["path"] = "scenario_tapes/unbound.npz"
    with pytest.raises(ValueError, match="canonical"):
        validate_seeds(bad_path)

    provenance = _provenance()
    assert validate_provenance(provenance) == provenance
    bad_video = {**provenance["video"], "package_version": "0.5.1"}
    with pytest.raises(ValueError, match="pinned 0.6.0"):
        validate_provenance({**provenance, "video": bad_video})
    invented_cpu = json.loads(json.dumps(provenance))
    invented_cpu["jax"]["cpu_devices"].append("TFRT_CPU_99")
    with pytest.raises(ValueError, match="CPU inventory.*subset"):
        validate_provenance(invented_cpu)
    missing_package = json.loads(json.dumps(provenance))
    missing_package["packages"].pop("optax")
    with pytest.raises(ValueError, match="exactly the collected package set"):
        validate_provenance(missing_package)
    extra_package = json.loads(json.dumps(provenance))
    extra_package["packages"]["untracked"] = "1.0"
    with pytest.raises(ValueError, match="exactly the collected package set"):
        validate_provenance(extra_package)
    partial_unavailable = json.loads(json.dumps(provenance))
    partial_unavailable["jax"]["jax_enable_x64"] = "unavailable"
    with pytest.raises(ValueError, match="available JAX runtime identity is incomplete"):
        validate_provenance(partial_unavailable)


def test_events_are_canonical_ordered_and_bound_to_trace(tmp_path: Path) -> None:
    trace = _trace()
    events = (
        ArtifactEvent(0, 0, 0.0, "runtime", "start", "info", 0, 0, {}),
        ArtifactEvent(1, 5, float(trace.time[5]), "solver", "fallback", "warning", 0, 0, {}),
    )
    path = tmp_path / "events.jsonl"
    write_events(events, path, trace=trace)
    assert load_events(path, trace=trace) == events
    assert all(json.loads(line)["schema_version"] == 1 for line in path.read_text().splitlines())

    with pytest.raises(ValueError, match="contiguous"):
        write_events((replace(events[0], sequence=1),), tmp_path / "bad.jsonl", trace=trace)
    with pytest.raises(ValueError, match="does not match"):
        write_events(
            (replace(events[0], time_seconds=0.1),), tmp_path / "bad-time.jsonl", trace=trace
        )


def test_metrics_and_timing_are_recomputed_from_raw_trace_and_detect_tampering(
    tmp_path: Path,
) -> None:
    trace = _trace()
    metrics_path = tmp_path / "metrics.json"
    timing_path = tmp_path / "timing.json"
    write_metrics(trace, metrics_path)
    metrics = load_metrics(metrics_path, trace=trace)
    assert metrics == derive_metrics(trace)
    tampered_metrics = {**metrics, "minimum_hard_margin": metrics["minimum_hard_margin"] + 1.0}
    with pytest.raises(ValueError, match="does not agree"):
        validate_metrics(tampered_metrics, trace=trace)

    compile_times = {name: 0.2 for name in trace.latency_names.tolist()}
    deadlines = {name: 0.01 for name in trace.latency_names.tolist()}
    write_timing(trace, timing_path, compile_seconds=compile_times, deadline_seconds=deadlines)
    timing = load_timing(timing_path, trace=trace)
    assert timing == derive_timing(trace, compile_seconds=compile_times, deadline_seconds=deadlines)
    tampered_timing = json.loads(json.dumps(timing))
    first_name = next(iter(tampered_timing["components"]))
    tampered_timing["components"][first_name]["p99_seconds"] += 1.0
    with pytest.raises(ValueError, match="does not agree"):
        validate_timing(tampered_timing, trace=trace)


def test_aggregate_csv_is_deterministic_strict_and_confidence_limits_are_labeled(
    tmp_path: Path,
) -> None:
    metrics = derive_metrics(_trace())
    rows = (
        aggregate_row("method-b", "condition", 1, metrics),
        aggregate_row("method-a", "condition", 1, metrics),
    )
    csv_path = tmp_path / "paired_metrics.csv"
    write_paired_metrics_csv(rows, csv_path)
    loaded = load_paired_metrics_csv(csv_path)
    assert [row["method"] for row in loaded] == ["method-a", "method-b"]

    intervals = tmp_path / "confidence_intervals.json"
    write_confidence_intervals(rows, intervals)
    data = json.loads(intervals.read_text())
    assert data["interval_method"] == "descriptive-normal-not-a-paired-superiority-test"
    assert all(not summary["interval_available"] for summary in data["summaries"])

    duplicate = rows + (rows[0],)
    with pytest.raises(ValueError, match="unique"):
        write_paired_metrics_csv(duplicate, tmp_path / "duplicate.csv")


def test_sha256sums_detects_tampering_extra_files_and_unsafe_entries(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}\n")
    (tmp_path / "trace.npz").write_bytes(b"trace")
    write_sha256sums(tmp_path)
    verify_sha256sums(tmp_path)

    (tmp_path / "trace.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        verify_sha256sums(tmp_path)

    checksum = tmp_path / "SHA256SUMS"
    checksum.write_text(f"{'0' * 64}  ../escape\n")
    with pytest.raises(ValueError, match="traversal"):
        verify_sha256sums(tmp_path)


def test_manifest_inventory_recognizes_adaptation_evidence_sidecars() -> None:
    assert (
        _artifact_role("methods/da_plcbf_full/static/0/adaptation_evidence.npz")
        == "adaptation-evidence"
    )


def _video_record(condition: str, *, method: str = "da_plcbf_full") -> dict[str, Any]:
    return {
        "renderer": "scientific-dashboard-v1",
        "path": f"videos/{method}--{condition}--fold-0000.mp4",
        "source_trace_path": f"methods/{method}/{condition}/0/trace.npz",
        "sha256": "a" * 64,
        "codec": "h264",
        "width": 1280,
        "height": 720,
        "fps": 10.0,
        "frame_count": 10,
        "duration_seconds": 1.0,
        "decoded_frames_sha256": "b" * 64,
    }


def test_final_video_coverage_requires_one_full_method_video_per_core_condition() -> None:
    records = tuple(
        _video_record(condition)
        for condition in ("static", "dynamics_change", "ballistic_ball", "interceptor_drone")
    )
    _validate_final_core_video_coverage(records)

    with pytest.raises(ValueError, match="exactly four"):
        _validate_final_core_video_coverage(records[:-1])
    with pytest.raises(ValueError, match="da_plcbf_full"):
        _validate_final_core_video_coverage(
            (_video_record("static", method="nominal_only"), *records[1:])
        )
    with pytest.raises(ValueError, match="every and only"):
        _validate_final_core_video_coverage((records[0], records[0], *records[2:]))


def test_final_review_uses_each_video_stem_for_its_exact_frame_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conditions = ("static", "dynamics_change", "ballistic_ball", "interceptor_drone")
    run = tmp_path / "review-run"
    video_directory = run / "videos"
    review_directory = run / "visual_reviews"
    video_directory.mkdir(parents=True)
    review_directory.mkdir()

    tape = generate_scenario_tape(4, ScenarioTapeConfig(steps=12, dt=0.05), fold=0)
    trace = synthetic_trace(tape.sha256, steps=12, dt=0.05)
    sidecar_digest = "d" * 64
    tape_records = []
    videos = []
    expected_calls = []
    check_names = (
        "original_resolution_inspected",
        "labels_legible_without_console",
        "unsafe_and_degraded_visibly_distinct",
        "overlays_agree_with_trace",
        "event_annotations_agree_with_trace",
        "camera_and_occlusion_acceptable",
        "scales_units_and_timing_clear",
        "unavailable_evidence_explicit",
    )
    for offset, condition in enumerate(conditions, start=1):
        method_directory = run / "methods" / "da_plcbf_full" / condition / "0"
        tape_directory = run / "scenario_tapes" / condition
        method_directory.mkdir(parents=True)
        tape_directory.mkdir(parents=True)
        save_trace(trace, method_directory / "trace.npz")
        save_scenario_tape(tape, tape_directory / "0.npz")
        (method_directory / "dashboard_evidence.npz").write_bytes(b"bound-sidecar")
        tape_records.append(
            {
                "condition": condition,
                "fold": 0,
                "path": f"scenario_tapes/{condition}/0.npz",
                "content_sha256": tape.sha256,
            }
        )

        stem = f"da_plcbf_full--{condition}--fold-0000"
        video_path = video_directory / f"{stem}.mp4"
        video_path.write_bytes(f"video-{condition}".encode())
        video = {
            **_video_record(condition),
            "sha256": file_sha256(video_path),
            "frame_count": trace.steps,
            "duration_seconds": trace.steps / 10.0,
        }
        videos.append(video)
        indices = (0, offset)
        expected_calls.append(
            (stem, indices, review_contact_sheet_title("da_plcbf_full", condition, 0))
        )
        review = VisualReviewRecord(
            schema_version=1,
            reviewer="audit agent",
            reviewer_kind="agent",
            reviewed_utc="2026-08-31T00:00:00Z",
            disposition="pass",
            trace_content_sha256=trace.content_sha256,
            scenario_tape_sha256=tape.sha256,
            dashboard_evidence_sha256=sidecar_digest,
            video_file_sha256=video["sha256"],
            decoded_frames_sha256=video["decoded_frames_sha256"],
            frame_width=video["width"],
            frame_height=video["height"],
            keyframe_indices=indices,
            checks=tuple(
                VisualReviewCheck(name=name, status="pass", note="Inspected exact evidence.")
                for name in check_names
            ),
            notes=("Inspected exact condition-bound full-resolution frames.",),
            revisions=(),
        )
        write_visual_review_record(review, review_directory / f"{stem}.md")

    write_seeds(
        {
            "schema_version": 1,
            "root_seed": 4,
            "folds": [0],
            "named_streams": {"scenario": 1},
            "scenario_tapes": sorted(
                tape_records, key=lambda record: (record["condition"], record["fold"])
            ),
            "pairing_id": "review-binding",
        },
        run / "seeds.json",
    )

    class _Sidecar:
        content_sha256 = sidecar_digest

    monkeypatch.setattr(
        "crazyflow.safety.da_plcbf.dashboard_evidence.load_dashboard_evidence",
        lambda _path: _Sidecar(),
    )
    observed_calls = []

    def capture_frame_validation(
        _root: Path,
        *,
        stem: str,
        keyframe_indices: tuple[int, ...],
        contact_sheet_title: str,
        **_kwargs: Any,
    ) -> None:
        observed_calls.append((stem, tuple(keyframe_indices), contact_sheet_title))

    monkeypatch.setattr(
        "crazyflow.safety.da_plcbf.artifacts._validate_review_frame_artifacts",
        capture_frame_validation,
    )
    reviewed = validate_campaign_visual_reviews(
        run, tuple(videos), require_all=True, require_final_core=True
    )

    assert observed_calls == expected_calls
    assert reviewed == tuple(f"visual_reviews/{stem}.md" for stem, _, _ in expected_calls)


def _write_test_png_header(path: Path, *, width: int, height: int) -> None:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)
    )


def test_final_review_requires_exact_full_resolution_keyframes_and_contact_sheet(
    tmp_path: Path,
) -> None:
    stem = "da_plcbf_full--static--fold-0000"
    trace = _trace()
    videos = tmp_path / "videos"
    keyframes = tmp_path / "keyframes" / stem
    sheets = tmp_path / "contact_sheets"
    videos.mkdir()
    (tmp_path / "keyframes").mkdir()
    sheets.mkdir()
    video = videos / f"{stem}.mp4"
    render_dashboard(trace, video, fps=10.0, size=(640, 360))
    records = extract_keyframes(video, trace, keyframes, count=8)
    title = review_contact_sheet_title("da_plcbf_full", "static", 0)
    assert title == (
        "da_plcbf_full · static · fold 0 · inspect ego-centric fallback selection, "
        "evasive response, hazards, and BPTT updates"
    )
    sheet = sheets / f"{stem}.png"
    render_contact_sheet(records, sheet, title=title)
    indices = tuple(record.step for record in records)

    _validate_review_frame_artifacts(
        tmp_path,
        stem=stem,
        keyframe_indices=indices,
        expected_size=(640, 360),
        trace=trace,
        tape=None,
        sidecar=None,
        video_path=video,
        contact_sheet_title=title,
    )
    first = keyframes / f"keyframe-{indices[0]:06d}.png"
    original = first.read_bytes()
    _write_test_png_header(first, width=640, height=360)
    with pytest.raises(ValueError, match="decodable PNG"):
        _validate_review_frame_artifacts(
            tmp_path,
            stem=stem,
            keyframe_indices=indices,
            expected_size=(640, 360),
            trace=trace,
            tape=None,
            sidecar=None,
            video_path=video,
            contact_sheet_title=title,
        )
    first.write_bytes(original)
    sheet.unlink()
    with pytest.raises(ValueError, match="missing regular PNG"):
        _validate_review_frame_artifacts(
            tmp_path,
            stem=stem,
            keyframe_indices=indices,
            expected_size=(640, 360),
            trace=trace,
            tape=None,
            sidecar=None,
            video_path=video,
            contact_sheet_title=title,
        )
    render_contact_sheet(records, sheet, title=title)
    _write_test_png_header(keyframes / "keyframe-999999.png", width=1280, height=720)
    with pytest.raises(ValueError, match="do not exactly match"):
        _validate_review_frame_artifacts(
            tmp_path,
            stem=stem,
            keyframe_indices=indices,
            expected_size=(640, 360),
            trace=trace,
            tape=None,
            sidecar=None,
            video_path=video,
            contact_sheet_title=title,
        )


def test_visual_review_validator_parses_and_binds_canonical_record(tmp_path: Path) -> None:
    run = tmp_path / "review-run"
    method_directory = run / "methods" / "method" / "static" / "0"
    video_directory = run / "videos"
    review_directory = run / "visual_reviews"
    tape_directory = run / "scenario_tapes" / "static"
    for directory in (method_directory, video_directory, review_directory, tape_directory):
        directory.mkdir(parents=True)
    tape = generate_scenario_tape(4, ScenarioTapeConfig(steps=12, dt=0.05), fold=0)
    save_scenario_tape(tape, tape_directory / "0.npz")
    trace = synthetic_trace(tape.sha256, steps=12, dt=0.05)
    save_trace(trace, method_directory / "trace.npz")
    write_seeds(
        {
            "schema_version": 1,
            "root_seed": 4,
            "folds": [0],
            "named_streams": {"scenario": 1},
            "scenario_tapes": [
                {
                    "condition": "static",
                    "fold": 0,
                    "path": "scenario_tapes/static/0.npz",
                    "content_sha256": tape.sha256,
                }
            ],
            "pairing_id": "review-binding",
        },
        run / "seeds.json",
    )
    video_path = video_directory / "dashboard.mp4"
    video_path.write_bytes(b"review-binding-video")
    video = {
        "path": "videos/dashboard.mp4",
        "source_trace_path": "methods/method/static/0/trace.npz",
        "sha256": file_sha256(video_path),
        "codec": "h264",
        "width": 1280,
        "height": 720,
        "fps": 10.0,
        "frame_count": 10,
        "duration_seconds": 1.0,
        "decoded_frames_sha256": "b" * 64,
    }
    check_names = (
        "original_resolution_inspected",
        "labels_legible_without_console",
        "unsafe_and_degraded_visibly_distinct",
        "overlays_agree_with_trace",
        "event_annotations_agree_with_trace",
        "camera_and_occlusion_acceptable",
        "scales_units_and_timing_clear",
        "unavailable_evidence_explicit",
    )
    review = VisualReviewRecord(
        schema_version=1,
        reviewer="audit agent",
        reviewer_kind="agent",
        reviewed_utc="2026-08-31T00:00:00Z",
        disposition="pass",
        trace_content_sha256=trace.content_sha256,
        scenario_tape_sha256=tape.sha256,
        dashboard_evidence_sha256=None,
        video_file_sha256=video["sha256"],
        decoded_frames_sha256=video["decoded_frames_sha256"],
        frame_width=video["width"],
        frame_height=video["height"],
        keyframe_indices=(0, 5, 11),
        checks=tuple(
            VisualReviewCheck(name=name, status="pass", note="Inspected exact bound evidence.")
            for name in check_names
        ),
        notes=("Inspected the exact trace-bound dashboard at its recorded resolution.",),
        revisions=(),
    )
    write_visual_review_record(review, review_directory / "dashboard.md")

    assert validate_campaign_visual_reviews(
        run, (video,), require_all=True, require_final_core=False
    ) == ("visual_reviews/dashboard.md",)
    with pytest.raises(ValueError, match="does not bind exactly"):
        validate_campaign_visual_reviews(
            run,
            ({**video, "decoded_frames_sha256": "c" * 64},),
            require_all=True,
            require_final_core=False,
        )
