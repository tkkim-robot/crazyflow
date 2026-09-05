"""Publication must preserve failures, physical identity, and exact control diagnostics."""

from __future__ import annotations

import gzip
import json
import subprocess
import tarfile
from typing import TYPE_CHECKING

import numpy as np
import pytest

import benchmark.da_plcbf_closed_loop_publication as publication

if TYPE_CHECKING:
    from pathlib import Path


def _batch(root: Path, name: str, scenes: list[dict]) -> Path:
    batch = root / name
    archived = batch / "source/benchmark/driver.py"
    archived.parent.mkdir(parents=True)
    archived.write_text("# actual execution source\n")
    publication.write_json(
        batch / "protocol.json",
        {
            "discovery_only": True,
            "source_hashes": {"benchmark/driver.py": publication.digest(archived)},
        },
    )
    rows = []
    for index, scene in enumerate(scenes):
        trial_id = f"{name}-{index:04d}"
        result = {
            "scene": scene,
            "mapping": "uncompensated",
            "methods": {
                "fixed": {"termination": "physical_collision", "all_operational_nodes_pass": True},
                "adaptive": {"termination": "completed", "all_operational_nodes_pass": True},
            },
            "outcome_class": "fixed_only_collision",
            "promotion_candidate": True,
        }
        publication.write_json(batch / trial_id / "result.json", result)
        np.savez_compressed(batch / trial_id / "traces.npz", fixed_state=np.zeros((3, 13)))
        rows.append({**result, "trial_id": trial_id, "family": "staggered"})
    (batch / "trials.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return batch


def test_completed_ledger_preserves_duplicate_executions_without_inflating_distinct_count(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    batch = _batch(
        root,
        "staggered-test",
        [
            {"seed": 1, "wind": [1, 0, 0]},
            {"seed": 2, "wind": [1, 0, 0]},
            {"seed": 3, "wind": [2, 0, 0]},
        ],
    )
    _batch(root, "smoke-test", [{"seed": 1, "wind": [1, 0, 0]}])
    (batch / "staggered-test-0003").mkdir()  # An interrupted/unledgered attempt is not a pair.
    with (batch / "trials.jsonl").open("a") as target:
        target.write(json.dumps({"trial_id": "partial", "methods": {"fixed": {}}}) + "\n")
    summary = publication.prepare(root, tmp_path)
    assert summary["discovery"]["completed_pairs"] == 3
    assert summary["discovery"]["distinct_scene_mapping_pairs_ignoring_seed"] == 2
    assert summary["discovery"]["outcomes"] == {"fixed_only_collision": 3}
    assert summary["smoke"]["completed_pairs"] == 1
    with gzip.open(root / "publication/all_completed_discovery.jsonl.gz", "rt") as source:
        decoded = [json.loads(line)["record"] for line in source]
    assert decoded == [
        json.loads(line) for line in (batch / "trials.jsonl").read_text().splitlines()[:3]
    ]
    counts = json.loads((root / "publication/COMPLETED_TRIAL_COUNTS.json").read_text())
    assert counts["unledgered_attempt_directories_not_counted"] == [
        "staggered-test/staggered-test-0003"
    ]
    assert counts["ledger_sources"]["staggered-test/trials.jsonl"]["incomplete_ledger_lines"] == [4]


def test_result_disagreement_blocks_publication(tmp_path: Path) -> None:
    batch = _batch(tmp_path, "staggered-test", [{"seed": 1}])
    result_path = batch / "staggered-test-0000/result.json"
    result = json.loads(result_path.read_text())
    result["outcome_class"] = "both_separated"
    publication.write_json(result_path, result)
    with pytest.raises(ValueError, match="Ledger differs"):
        publication.completed_ledgers(tmp_path, tmp_path)


def test_archived_source_disagreement_blocks_publication(tmp_path: Path) -> None:
    batch = _batch(tmp_path, "staggered-test", [{"seed": 1}])
    (batch / "source/benchmark/driver.py").write_text("# changed after execution\n")
    with pytest.raises(ValueError, match="Archived execution source differs"):
        publication.completed_ledgers(tmp_path, tmp_path)


def test_confirmation_reused_episode_and_incomplete_rows_are_counted_separately(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "confirmation-test"
    directory.mkdir()
    methods = {"fixed": {"termination": "completed"}, "adaptive": {"termination": "completed"}}
    rows = [
        {"variant": "original", "methods": methods},
        {"variant": "freeze_at_wind_onset", "methods": methods},
        {"variant": "interrupted", "methods": {"fixed": {}}},
    ]
    (directory / "trials.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    publication.write_json(
        tmp_path / "targeted-confirmation-test/SUMMARY.json",
        {"completed": {"methods": methods}, "unexecuted": {"scene": {}}},
    )
    discovery, smoke, evidence = publication.completed_ledgers(tmp_path, tmp_path)
    assert not discovery and not smoke
    counts = evidence["other_ledgers_not_counted_as_discovery"]
    assert counts["confirmation-test"]["paired_comparisons"] == 2
    assert counts["confirmation-test"]["new_method_episodes"] == 3
    assert counts["confirmation-test"]["reused_adaptive_comparisons"] == 1
    assert counts["targeted-confirmation-test"]["paired_comparisons"] == 1


def test_compact_trace_keeps_exact_dtype_nan_padding_and_control_evidence(tmp_path: Path) -> None:
    source = tmp_path / "navigation_comparison.npz"
    values = np.asarray([[0.1, np.nan], [0.2, 0.3]], dtype=np.float32)
    expected = {
        "time_seconds": np.asarray([0, 0.04]),
        "fixed_full_state": np.arange(26, dtype=np.float32).reshape(2, 13),
        "fixed_recorded_control_valid": np.asarray([True, False]),
        "adaptive_library_version": np.asarray([715, 716], dtype=np.int32),
        "adaptive_controller_seconds": np.asarray([0.01, 0.02], dtype=np.float32),
    }
    np.savez_compressed(source, **expected, adaptive_fallback_rollouts=np.ones((2, 16, 61, 3)))
    np.savez_compressed(
        tmp_path / "raw_diagnostics.npz",
        adaptive_hard=values,
        adaptive_candidate_valid=np.isfinite(values),
        adaptive_candidate_wrenches=np.ones((2, 17, 60, 4)),
    )
    record = publication.compact_trace(source, tmp_path)
    with np.load(tmp_path / "compact_control_trace.npz") as saved:
        assert set(saved.files) == set(expected) | {"adaptive_hard", "adaptive_candidate_valid"}
        for key, value in expected.items():
            np.testing.assert_array_equal(saved[key], value, strict=True)
        np.testing.assert_array_equal(saved["adaptive_hard"], values, strict=True)
    assert record["all_saved_arrays_verified_exact"]
    assert record["sources"]["navigation_comparison.npz"]["sha256"] == publication.digest(source)
    assert (
        "adaptive_fallback_rollouts"
        in record["sources"]["navigation_comparison.npz"]["excluded_arrays"]
    )


def test_conflicting_raw_diagnostic_never_silently_overwrites_control_trace(tmp_path: Path) -> None:
    source = tmp_path / "navigation_comparison.npz"
    np.savez_compressed(source, adaptive_library_version=np.array([715]))
    np.savez_compressed(tmp_path / "raw_diagnostics.npz", adaptive_library_version=np.array([716]))
    with pytest.raises(AssertionError):
        publication.compact_trace(source, tmp_path)


@pytest.mark.parametrize("name", ["video.MP4", "nested/file.webm", "nested/file.mov"])
def test_video_is_excluded_regardless_of_directory_or_extension_case(
    tmp_path: Path, name: str
) -> None:
    assert not publication.include(tmp_path / name, tmp_path)[0]


def test_final_figures_and_physical_contact_evidence_are_kept(tmp_path: Path) -> None:
    for name in (
        "figures-v2/clearance.png",
        "contact/contact_replay.npz",
        "trial/traces.npz",
        "videos/paced-collision-v2/frame.png",
        "videos/VIDEO_REVIEW.json",
        "videos/SUPERSEDED_V1.json",
    ):
        assert publication.include(tmp_path / name, tmp_path)[0]
    for name in (
        "figures-v1/clearance.png",
        "trial.partial-1/traces.npz",
        "run/raw_diagnostics.npz",
        "videos/paced-collision-v1/frame.png",
        "videos/camera-preview/4.70.png",
    ):
        assert not publication.include(tmp_path / name, tmp_path)[0]


def test_source_delta_includes_untracked_nested_source_and_records_deleted_files(
    tmp_path: Path,
) -> None:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Publication Test")
    (tmp_path / "old.py").write_text("old = 1\n")
    git("add", "old.py")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "old.py").unlink()
    (tmp_path / "benchmark").mkdir()
    (tmp_path / "benchmark/new.py").write_text("new = 2\n")
    (tmp_path / "video.MP4").write_bytes(b"local video")
    output = tmp_path / "artifacts/publication"
    output.mkdir(parents=True)
    record = publication.source_delta(output, tmp_path, base)
    assert record["deleted_files"] == ["old.py"]
    assert set(record["files"]) == {"benchmark/new.py"}
    assert record["excluded_media_and_bulk_files"] == ["video.MP4"]
    with tarfile.open(output / "source_delta.tar.gz") as archive:
        assert archive.getnames() == ["benchmark/new.py"]
        assert archive.extractfile("benchmark/new.py").read() == b"new = 2\n"
