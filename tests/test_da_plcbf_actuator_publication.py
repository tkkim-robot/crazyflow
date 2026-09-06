"""Exercise real archive/inventory paths with tiny format fixtures and no numerical runtime.

The media and episode payloads below are integrity fixtures, not scientific data.
Only the repository location is redirected; collection and verification run unmocked.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import lzma
import shutil
import struct
import tarfile
import zipfile
from pathlib import Path

import pytest

from benchmark import da_plcbf_actuator_publication as publication

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aKZkAAAAASUVORK5CYII="
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_fixture() -> bytes:
    header = b"{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }"
    header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    array = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack("<d", 0)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(zipfile.ZipInfo("value.npy", date_time=(1980, 1, 1, 0, 0, 0)), array)
    return stream.getvalue()


@pytest.fixture
def packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    root = tmp_path / "repository"
    collector = root / "benchmark/da_plcbf_actuator_publication.py"
    collector.parent.mkdir(parents=True)
    collector.write_bytes(Path(publication.__file__).read_bytes())
    figure_source = collector.with_name("da_plcbf_actuator_figures.py")
    figure_source.write_text("# Figure source integrity fixture.\n")
    monkeypatch.setattr(publication, "ROOT", root)
    monkeypatch.setattr(publication, "__file__", str(collector))
    study = root / "artifacts/study"
    main = study / "main"
    figures, video = study / "figures", study / "video"
    figures.mkdir(parents=True)
    video.mkdir()
    long_source = study / ("long-directory-" * 10) / ("long-filename-" * 10 + ".json")
    write_json(long_source, {"fixture": "long PAX source path"})
    spec = {
        "worlds": [{"scene_seed": 1}],
        "library_seeds": [11],
        "methods": ["F2", "A"],
        "split": "test",
    }
    protocol_manifest = {
        "campaign_specification_sha256": publication.content_sha256(spec),
        "source": {"commit": "1" * 40},
    }
    seal = publication.content_sha256(protocol_manifest)
    protocol_path = study / "protocol.json"
    write_json(
        protocol_path,
        {
            "schema": "sealed_da_plcbf_actuator_protocol_v1",
            "manifest": protocol_manifest,
            "sha256": seal,
        },
    )
    spec["sealed_protocol_path"] = protocol_path.relative_to(root).as_posix()
    binding = {"specification": spec, "sealed_protocol_sha256": seal}
    binding["sha256"] = publication.content_sha256(binding)
    write_json(main / "campaign_binding.json", binding)
    records, video_sources = [], {}
    for method in spec["methods"]:
        episode = main / f"world-0000-seed-11-{method}/attempt-00"
        config = {"method": method, "execution_mode": "deterministic", "plant_level": "P0"}
        summary = {
            **config,
            "status": "completed",
            "termination": "duration_complete",
            "physical_world_id": "fixture-world",
        }
        write_json(
            episode / "binding.json",
            {"config": config, "scene": {"physical_world_id": "fixture-world"}},
        )
        write_json(episode / "summary.json", summary)
        for name in ("dense.npz", "controls.npz", "applications.npz"):
            (episode / name).write_bytes(numerical_fixture())
        records.append(
            {
                "world_index": 0,
                "library_seed": 11,
                "method": method,
                "summary_path": (episode / "summary.json").relative_to(main).as_posix(),
                "summary": summary,
            }
        )
        video_sources.update({str(path): digest(path) for path in episode.iterdir()})
    write_json(
        main / "campaign_result.json",
        {"campaign_sha256": binding["sha256"], "planned": 2, "completed": 2, "records": records},
    )
    analysis = study / "analysis/analysis.json"
    write_json(
        analysis,
        {
            "schema": "da_plcbf_actuator_analysis_v1",
            "input_kind": "sealed_ledger",
            "protocol_sha256": seal,
            "rows": [{"method": "F2"}, {"method": "A"}],
        },
    )
    figure_records, output_hashes = [], {}
    for stem in publication.FIGURE_STEMS:
        record = {"name": stem, "png": stem + ".png", "pdf": stem + ".pdf", "data": stem + ".csv"}
        for name, payload in (
            (record["png"], PNG),
            (record["pdf"], b"%PDF-1.4\n%%EOF\n"),
            (record["data"], b"method,value\nF2,0\nA,0\n"),
        ):
            path = figures / name
            path.write_bytes(payload)
            output_hashes[name] = digest(path)
        figure_records.append(record)
    write_json(
        figures / "manifest.json",
        {
            "schema": "da_plcbf_actuator_publication_figures_v1",
            "status": "completed",
            "figures": figure_records,
            "output_sha256": output_hashes,
            "figure_source_sha256": digest(figure_source),
            "input_sha256": {
                str(analysis): digest(analysis),
                str(long_source): digest(long_source),
            },
        },
    )
    (video / "poster.png").write_bytes(PNG)
    (video / "frame_audit.jsonl").write_text('{"frame": 0}\n')
    (video / "comparison.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    write_json(
        video / "render_binding.json",
        {
            "synthetic_fixture": False,
            "config": {"synthetic_fixture": False},
            "source_sha256": video_sources,
            "physical_world_id": "fixture-world",
        },
    )
    write_json(
        video / "render_summary.json",
        {
            "status": "completed",
            "video": str(video / "comparison.mp4"),
            "video_sha256": digest(video / "comparison.mp4"),
            "frame_audit_sha256": digest(video / "frame_audit.jsonl"),
        },
    )
    report, technical = root / "docs/report.md", root / "docs/theory.md"
    report.parent.mkdir()
    report.write_text("# Integrity test report\nNo scientific outcomes are asserted.\n")
    technical.write_text("# Integrity test note\n")
    return argparse.Namespace(
        study_root=study,
        main_campaign=main,
        figure_dir=figures,
        video_dir=video,
        video_poster="poster.png",
        report=[report],
        technical_note=technical,
        main_analysis=analysis,
        reporting_revision="2" * 40,
        promoted_case_campaign=[],
        output=root / "publication-a",
        long_source=long_source,
    )


def test_valid_packet_has_deterministic_archive_and_long_pax_path(
    packet: argparse.Namespace,
) -> None:
    original = packet.long_source.read_bytes()
    result = publication.collect(packet)
    assert result["status"] == "verified"
    assert publication.verify(packet.output)["all_declared_input_output_digests_match_inventory"]
    archive = packet.output / publication.ARCHIVE_NAME
    first_bytes = archive.read_bytes()
    with tarfile.open(archive, "r:xz") as stream:
        members = stream.getmembers()
    long_name = packet.long_source.relative_to(publication.ROOT).as_posix()
    assert (
        next(member for member in members if member.name == long_name).pax_headers["path"]
        == long_name
    )
    assert not any(member.name.endswith(".mp4") for member in members)
    readme = (packet.output / "README.md").read_text()
    assert "recorded deterministic/P0 F2/A case" in readme
    assert digest(packet.video_dir / "comparison.mp4") in readme
    packet.output = publication.ROOT / "publication-b"
    publication.collect(packet)
    assert (packet.output / publication.ARCHIVE_NAME).read_bytes() == first_bytes
    assert packet.long_source.read_bytes() == original


@pytest.mark.parametrize("corrupt_movie", [False, True])
def test_companion_has_independent_scope_and_authenticated_media(
    packet: argparse.Namespace, corrupt_movie: bool
) -> None:
    companion = packet.study_root / "paced-video"
    shutil.copytree(packet.video_dir, companion)
    sources = {}
    for method in ("F2", "A"):
        episode = packet.study_root / "paced-episodes" / method
        shutil.copytree(packet.main_campaign / f"world-0000-seed-11-{method}/attempt-00", episode)
        for filename in ("binding.json", "summary.json"):
            path = episode / filename
            value = json.loads(path.read_text())
            (value["config"] if filename == "binding.json" else value)["execution_mode"] = "paced"
            write_json(path, value)
        sources.update({str(path): digest(path) for path in episode.iterdir()})
    binding = json.loads((companion / "render_binding.json").read_text())
    binding["source_sha256"] = sources
    write_json(companion / "render_binding.json", binding)
    summary = json.loads((companion / "render_summary.json").read_text())
    summary["video"] = str(companion / "comparison.mp4")
    write_json(companion / "render_summary.json", summary)
    packet.companion_video = [(str(companion), "poster.png")]
    if corrupt_movie:
        (companion / "comparison.mp4").write_bytes(b"changed-media-bytes")
        with pytest.raises(ValueError, match="declared input/output digest"):
            publication.collect(packet)
        return
    assert publication.collect(packet)["status"] == "verified"
    manifest = json.loads((packet.output / "manifest.json").read_text())
    scope = manifest["reviewer_inputs"]["companion_videos"][0]
    assert scope["execution_mode"] == "paced" and scope["plant_level"] == "P0"
    assert "Separate companion (paced/P0)" in (packet.output / "README.md").read_text()
    records = {row["original_repo_path"]: row for row in manifest["all_original_files"]}
    assert records[scope["video_path"]]["archive_path"] is None
    assert records[scope["video_poster"]]["archive_path"] == scope["video_poster"]
    audit = scope["video_directory"] + "/frame_audit.jsonl"
    assert records[audit]["archive_path"] == audit


@pytest.mark.parametrize("corruption", ["count", "duplicate", "out_of_support", "summary"])
def test_main_completion_rejects_inconsistent_bound_trials(
    packet: argparse.Namespace, corruption: str
) -> None:
    path = packet.main_campaign / "campaign_result.json"
    result = json.loads(path.read_text())
    if corruption == "count":
        result.update(planned=1, completed=1, records=result["records"][:1])
    elif corruption == "duplicate":
        result["records"][1] = copy.deepcopy(result["records"][0])
    elif corruption == "out_of_support":
        result["records"][0]["world_index"] = 1
    else:
        result["records"][0]["summary"]["termination"] = "changed-only-in-result"
    write_json(path, result)
    with pytest.raises(
        ValueError, match="bound matrix|duplicate or out-of-support|retained summary"
    ):
        publication.collect(packet)
    assert not packet.output.exists()


@pytest.mark.parametrize("source", ["figure_input", "episode_binding", "episode_array"])
def test_collection_rejects_stale_declared_sources(packet: argparse.Namespace, source: str) -> None:
    if source == "figure_input":
        path = packet.long_source
    else:
        filename = "binding.json" if source == "episode_binding" else "dense.npz"
        path = packet.main_campaign / "world-0000-seed-11-F2/attempt-00" / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="conflicting source digests|declared input/output digest"):
        publication.collect(packet)


def test_synthetic_video_is_not_accepted_as_real_study_evidence(packet: argparse.Namespace) -> None:
    path = packet.video_dir / "render_binding.json"
    binding = json.loads(path.read_text())
    binding["synthetic_fixture"] = binding["config"]["synthetic_fixture"] = True
    write_json(path, binding)
    with pytest.raises(ValueError, match="synthetic or unclassified fixture"):
        publication.collect(packet)
    assert not packet.output.exists()


@pytest.mark.parametrize("tampering", ["compressed_bytes", "member_with_updated_outer_digest"])
def test_verifier_rejects_archive_tampering(packet: argparse.Namespace, tampering: str) -> None:
    publication.collect(packet)
    archive = packet.output / publication.ARCHIVE_NAME
    if tampering == "compressed_bytes":
        payload = bytearray(archive.read_bytes())
        payload[len(payload) // 2] ^= 1
        archive.write_bytes(payload)
        expected_error = "archive SHA-256 or size mismatch"
    else:
        long_name = packet.long_source.relative_to(publication.ROOT).as_posix()
        with tarfile.open(archive, "r:xz") as stream:
            member = stream.getmember(long_name)
        payload = bytearray(lzma.decompress(archive.read_bytes()))
        payload[member.offset_data] ^= 1
        archive.write_bytes(lzma.compress(payload, check=lzma.CHECK_CRC64, preset=6))
        manifest_path = packet.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["archive"].update(sha256=digest(archive), size_bytes=archive.stat().st_size)
        write_json(manifest_path, manifest)
        expected_error = "archive member SHA-256 mismatch"
    with pytest.raises(ValueError, match=expected_error):
        publication.verify(packet.output)
