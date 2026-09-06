"""Collect a deterministic review archive and a complete local-only file inventory.

Collection must wait until the sealed main campaign and all intended numerics finish::

    python -m benchmark.da_plcbf_actuator_publication collect \
        --study-root artifacts/da_plcbf/actuator-study-20260906/v1 \
        --main-campaign artifacts/da_plcbf/actuator-study-20260906/v1/main384-sealed-results-v1 \
        --figure-dir /absolute/completed-figures --video-dir /absolute/completed-video \
        --video-poster frame-000080.png --report docs/da_plcbf_actuator_report.md \
        --technical-note docs/da_plcbf_actuator_theory.md \
        --main-analysis /absolute/completed-analysis/analysis.json \
        --reporting-revision FINAL_40_CHARACTER_GIT_REVISION \
        --output artifacts/da_plcbf/actuator-study-20260906/v1/publication-v1

    python -m benchmark.da_plcbf_actuator_publication verify \
        --publication-dir artifacts/da_plcbf/actuator-study-20260906/v1/publication-v1

This standard-library-only tool does not run models, tests, Git, or extraction.
Every original study file is hashed and sized, including omitted videos and cache
binaries; only the fresh output and __pycache__ directories are outside inventory.
Original included members keep their repository-relative paths and exact bytes.
Generated review/checksum members use fixed __publication__/ paths. The detached
manifest carries the final archive hash, avoiding a circular checksum dependency.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import lzma
import math
import os
import re
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "da_plcbf_actuator_publication_v1"
ARCHIVE_NAME = "review-only.tar.xz"
CHECKSUM_MEMBER = "__publication__/SHA256SUMS"
REVIEW_MEMBER = "__publication__/REVIEW_ONLY.md"
README_MEMBER = "__publication__/README.md"
METADATA_SUFFIXES = {".json", ".md", ".csv", ".py", ".toml", ".xml", ".sh"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif"}
FIGURE_STEMS = (
    "01_main_safety",
    "02_fixed_bank_recovery",
    "03_safety_and_execution_cost",
    "04_library_and_retention_ablations",
    "05_one_factor_mismatch",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, value in pairs:
        require(name not in result, f"duplicate JSON member: {name}")
        result[name] = value
    return result


def read_json(path: Path, expected_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    before = regular_file(path)
    payload = path.read_bytes()
    require(
        stable_identity(before) == stable_identity(regular_file(path)),
        f"JSON source changed while reading: {path}",
    )

    def reject_constant(value: str) -> Any:
        raise ValueError(f"nonfinite JSON number: {value}")

    result = json.loads(payload, object_pairs_hook=strict_object, parse_constant=reject_constant)
    require(isinstance(result, dict), f"expected a JSON object: {path}")
    if expected_hashes is not None:
        add_expected_digest(expected_hashes, path, hashlib.sha256(payload).hexdigest())
    return result


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def canonical(value: Any) -> Any:
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite canonical JSON value")
        return int(value) if value.is_integer() else value
    if isinstance(value, dict):
        return {name: canonical(item) for name, item in value.items()}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def content_sha256(value: Any) -> str:
    payload = json.dumps(
        canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def safe_member(name: str) -> str:
    require(isinstance(name, str) and bool(name), "empty or nonstring archive member")
    require(
        not any(character in name for character in ("\\", "\x00", "\n", "\r")),
        f"unsafe archive member characters: {name!r}",
    )
    path = PurePosixPath(name)
    require(
        not path.is_absolute()
        and not re.match(r"^[A-Za-z]:", name)
        and all(part not in {"", ".", ".."} for part in name.split("/"))
        and path.as_posix() == name,
        f"unsafe or noncanonical archive path: {name!r}",
    )
    return name


def regular_file(path: Path) -> os.stat_result:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), f"regular nonsymlink file required: {path}")
    return info


def stable_identity(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def hash_file(path: Path) -> tuple[str, int, tuple[int, ...]]:
    before = regular_file(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = regular_file(path)
    require(
        stable_identity(before) == stable_identity(after), f"source changed while hashing: {path}"
    )
    return digest, before.st_size, stable_identity(before)


def source_path(path: Path) -> Path:
    require(not path.is_symlink(), f"source path cannot be a symlink: {path}")
    path = path.resolve(strict=True)
    require(path.is_relative_to(ROOT), f"source must be inside the repository: {path}")
    return path


def relative(path: Path) -> str:
    return safe_member(path.relative_to(ROOT).as_posix())


def add_expected_digest(expected: dict[str, str], path: Path, digest: str) -> None:
    require(
        isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
        f"missing or malformed declared source digest: {path}",
    )
    name = relative(source_path(path))
    require(name not in expected or expected[name] == digest, f"conflicting source digests: {name}")
    expected[name] = digest


def add_declared_hashes(expected: dict[str, str], declared: Any, label: str) -> None:
    require(isinstance(declared, dict) and bool(declared), f"{label} has no declared source hashes")
    for name, digest in declared.items():
        require(isinstance(name, str) and bool(name), f"{label} has an invalid source path")
        path = Path(name)
        path = source_path(path if path.is_absolute() else ROOT / path)
        regular_file(path)
        add_expected_digest(expected, path, digest)


def walk_files(directory: Path, output: Path) -> list[Path]:
    require(
        directory.is_dir() and not directory.is_symlink(), f"source directory required: {directory}"
    )
    found = []
    for current, directories, files in os.walk(directory, topdown=True, followlinks=False):
        parent = Path(current)
        kept = []
        for name in sorted(directories):
            path = parent / name
            if name == "__pycache__" or path == output:
                continue
            require(not path.is_symlink(), f"symlink directory is not collectable: {path}")
            kept.append(name)
        directories[:] = kept
        for name in sorted(files):
            path = parent / name
            if path.is_relative_to(output):
                continue
            regular_file(path)
            found.append(path)
    return found


def complete_main(directory: Path, expected_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    expected_hashes = {} if expected_hashes is None else expected_hashes
    result_path = directory / "campaign_result.json"
    require(
        result_path.is_file(),
        "main campaign is active or incomplete: campaign_result.json is absent",
    )
    result = read_json(result_path, expected_hashes)
    planned, completed = result.get("planned"), result.get("completed")
    require(
        type(planned) is int and planned > 0 and type(completed) is int and completed == planned,
        "main campaign is active or incomplete: completed does not equal planned",
    )
    records = result.get("records")
    require(
        isinstance(records, list)
        and len(records) == planned
        and all(record.get("summary", {}).get("status") == "completed" for record in records),
        "main campaign completion counts contradict its terminal records",
    )
    binding_path = directory / "campaign_binding.json"
    binding = read_json(binding_path, expected_hashes)
    require(
        content_sha256({name: value for name, value in binding.items() if name != "sha256"})
        == binding.get("sha256")
        == result.get("campaign_sha256"),
        "main campaign result/binding checksum mismatch",
    )
    specification = binding["specification"]
    worlds, seeds, methods = (
        specification["worlds"],
        specification["library_seeds"],
        specification["methods"],
    )
    require(
        isinstance(worlds, list)
        and bool(worlds)
        and isinstance(seeds, list)
        and bool(seeds)
        and all(type(seed) is int for seed in seeds)
        and len(set(seeds)) == len(seeds)
        and isinstance(methods, list)
        and bool(methods)
        and all(isinstance(method, str) and bool(method) for method in methods)
        and len(set(methods)) == len(methods),
        "main bound worlds, library seeds or methods are empty, duplicated or invalid",
    )
    expected_trials = {
        (world_index, seed, method)
        for world_index in range(len(worlds))
        for seed in seeds
        for method in methods
    }
    require(planned == len(expected_trials), "main planned count differs from its bound matrix")
    seen_trials, seen_summaries = set(), set()
    for record in records:
        identity = record["world_index"], record["library_seed"], record["method"]
        require(
            type(identity[0]) is int
            and type(identity[1]) is int
            and isinstance(identity[2], str)
            and identity in expected_trials
            and identity not in seen_trials,
            "duplicate or out-of-support main result record",
        )
        summary_name = safe_member(record["summary_path"])
        summary_path = source_path(directory / summary_name)
        require(
            summary_path.is_relative_to(directory)
            and summary_path.name == "summary.json"
            and summary_path not in seen_summaries,
            "main result has an invalid or reused retained-summary path",
        )
        require(
            read_json(summary_path, expected_hashes) == record["summary"],
            f"embedded main result differs from its retained summary: {summary_name}",
        )
        require(
            record["summary"].get("method") == identity[2], "main record/summary method mismatch"
        )
        seen_trials.add(identity)
        seen_summaries.add(summary_path)
    require(seen_trials == expected_trials, "main result omits a bound trial")
    seal = binding.get("sealed_protocol_sha256")
    require(
        specification.get("split") == "test"
        and isinstance(seal, str)
        and re.fullmatch(r"[0-9a-f]{64}", seal) is not None,
        "main campaign does not identify a sealed test protocol",
    )
    protocol_path = Path(specification["sealed_protocol_path"])
    protocol_path = source_path(
        protocol_path if protocol_path.is_absolute() else ROOT / protocol_path
    )
    protocol = read_json(protocol_path, expected_hashes)
    require(
        protocol.get("schema") == "sealed_da_plcbf_actuator_protocol_v1"
        and protocol.get("sha256") == seal == content_sha256(protocol.get("manifest")),
        "sealed protocol envelope checksum mismatch",
    )
    unsigned = {
        name: value for name, value in specification.items() if name != "sealed_protocol_path"
    }
    require(
        content_sha256(unsigned) == protocol["manifest"]["campaign_specification_sha256"],
        "completed main specification differs from its sealed protocol",
    )
    return {
        "campaign_directory": relative(directory),
        "completed": completed,
        "planned": planned,
        "campaign_sha256": binding["sha256"],
        "protocol_sha256": seal,
        "protocol_path": relative(protocol_path),
        "sealed_numerical_revision": protocol["manifest"]["source"]["commit"],
        "campaign_result_sha256": hash_file(result_path)[0],
        "campaign_binding_sha256": hash_file(binding_path)[0],
        "protocol_file_sha256": hash_file(protocol_path)[0],
    }


def figure_files(directory: Path) -> tuple[set[Path], dict[str, str]]:
    selected, hashes = set(), {}
    manifest = read_json(directory / "manifest.json", hashes)
    require(
        manifest.get("schema") == "da_plcbf_actuator_publication_figures_v1"
        and manifest.get("status") in {"completed", "completed_with_omissions"},
        "figure directory has no completed publication-figure manifest",
    )
    records = {record["name"]: record for record in manifest.get("figures", [])}
    require(len(records) == len(manifest.get("figures", [])), "duplicate figure manifest entries")
    add_declared_hashes(hashes, manifest.get("input_sha256"), "figure manifest")
    add_expected_digest(
        hashes,
        Path(__file__).with_name("da_plcbf_actuator_figures.py"),
        manifest.get("figure_source_sha256"),
    )
    require(isinstance(manifest.get("output_sha256"), dict), "figure output hashes are absent")
    for name, digest in manifest["output_sha256"].items():
        path = directory / safe_member(name)
        regular_file(path)
        add_expected_digest(hashes, path, digest)
    for stem in FIGURE_STEMS:
        require(stem in records, f"required final figure is missing: {stem}")
        for field, suffix in (("png", ".png"), ("pdf", ".pdf"), ("data", ".csv")):
            name = safe_member(records[stem][field])
            require(name == stem + suffix, f"unexpected final figure path: {name}")
            path = directory / name
            regular_file(path)
            expected = manifest["output_sha256"].get(name)
            require(isinstance(expected, str), f"figure output digest missing: {name}")
            selected.add(path)
            add_expected_digest(hashes, path, expected)
    return selected, hashes


def video_files(directory: Path, poster: str) -> tuple[set[Path], dict[str, str], dict[str, Any]]:
    hashes = {}
    name = safe_member(poster)
    path = directory / name
    require(path.suffix.lower() == ".png", "selected video poster must be a PNG")
    regular_file(path)
    binding = read_json(directory / "render_binding.json", hashes)
    summary = read_json(directory / "render_summary.json", hashes)
    require(summary.get("status") == "completed", "video rendering has not completed successfully")
    require(
        binding.get("synthetic_fixture") is False
        and binding.get("config", {}).get("synthetic_fixture") is False,
        "a synthetic or unclassified fixture video cannot enter the real-study packet",
    )
    add_declared_hashes(hashes, binding.get("source_sha256"), "video render binding")
    episode_directories = {
        (ROOT / name).parent for name in hashes if Path(name).name == "binding.json"
    }
    require(len(episode_directories) == 2, "video must bind exactly two source episodes")
    episodes = []
    for episode in sorted(episode_directories):
        require(
            all(
                relative(episode / filename) in hashes
                for filename in (
                    "binding.json",
                    "summary.json",
                    "dense.npz",
                    "controls.npz",
                    "applications.npz",
                )
            ),
            "video source binding omits a required episode file",
        )
        configuration = read_json(episode / "binding.json", hashes)
        episode_summary = read_json(episode / "summary.json", hashes)
        require(episode_summary.get("status") == "completed", "video source episode is incomplete")
        require(
            episode_summary.get("physical_world_id")
            == binding.get("physical_world_id")
            == configuration["scene"]["physical_world_id"],
            "video source episode world differs from its render binding",
        )
        mode, plant = (
            configuration["config"]["execution_mode"],
            configuration["config"]["plant_level"],
        )
        require(
            mode in {"deterministic", "paced", "delayed"} and plant in {"P0", "P1", "P2"},
            "video source has an unsupported execution or plant label",
        )
        require(
            episode_summary.get("execution_mode") == mode
            and episode_summary.get("plant_level") == plant
            and episode_summary.get("method") == configuration["config"]["method"]
            and episode_summary.get("termination") in {"duration_complete", "physical_collision"},
            "video source summary contradicts its configuration or terminal status",
        )
        episodes.append(
            {
                "directory": relative(episode),
                "method": episode_summary["method"],
                "execution_mode": mode,
                "plant_level": plant,
            }
        )
    require(
        {episode["method"] for episode in episodes} == {"F2", "A"}, "video must bind F2/A episodes"
    )
    require(
        len({(episode["execution_mode"], episode["plant_level"]) for episode in episodes}) == 1,
        "video source episodes disagree on execution mode or plant level",
    )
    selected = {path}
    audit = directory / "frame_audit.jsonl"
    regular_file(audit)
    selected.add(audit)
    movie = Path(summary["video"])
    movie = source_path(movie if movie.is_absolute() else ROOT / movie)
    require(movie.is_relative_to(directory), "finished video lies outside its declared directory")
    require(movie.suffix.lower() in VIDEO_SUFFIXES, "finished video has an unexpected extension")
    add_expected_digest(hashes, movie, summary.get("video_sha256"))
    add_expected_digest(hashes, audit, summary.get("frame_audit_sha256"))
    return (
        selected,
        hashes,
        {
            "video_path": relative(movie),
            "source_episodes": episodes,
            "execution_mode": episodes[0]["execution_mode"],
            "plant_level": episodes[0]["plant_level"],
            "physical_world_id": binding["physical_world_id"],
            "synthetic_fixture": False,
        },
    )


def critical_npz(path: Path, study_root: Path) -> str | None:
    if path.suffix.lower() != ".npz" or not path.is_relative_to(study_root):
        return None
    parts = path.relative_to(study_root).parts
    if path.name.startswith("deployment") and any(
        re.fullmatch(r"behavior-(?:nominal128|dr512|single128)-seed\d+-v\d+", part)
        or re.fullmatch(r"retention-off-initial-seed\d+-v\d+", part)
        for part in parts[:-1]
    ):
        return "selected initial deployment checkpoint"
    if path.name == "numerical.npz" and "shared" in path.parent.name:
        return "selected shared-state numerical probe"
    if (
        path.name in {"dense.npz", "applications.npz"}
        and any(re.fullmatch(r"candidate-\d+-original-v\d+", part) for part in parts[:-1])
        and any(re.fullmatch(r"world-\d+-seed-\d+-(?:F2|A)", part) for part in parts[:-1])
    ):
        return "selected candidate original F2/A physical trace subset"
    return None


def inclusion(path: Path, study_root: Path, selected_media: set[Path]) -> tuple[bool, str]:
    parts = path.parts
    if any(
        re.fullmatch(r"canonical-compilation-cache-v\d+", parts[index])
        and parts[index + 1] == "initial-cache"
        for index in range(len(parts) - 1)
    ):
        return False, "optimized compilation-cache binary subtree; retained locally"
    if path.name == "events.jsonl":
        return False, "bulk experiment event stream; retained locally"
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return False, "video/animation media; retained locally"
    if path in selected_media:
        return True, "explicit final figure or selected video audit/poster"
    if path.suffix.lower() == ".txt" and "hlo" in path.name.lower():
        return False, "large compiler HLO text; digest-bearing metadata is included"
    selected = critical_npz(path, study_root)
    if selected is not None:
        return True, selected
    if path.suffix.lower() in METADATA_SUFFIXES or path.name == "SHA256SUMS":
        return True, "metadata/source text; status and outcome do not affect inclusion"
    if path.suffix.lower() in {".npz", ".npy"}:
        return False, "unselected numerical rollout/control/snapshot array; retained locally"
    if path.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg", ".svg", ".webp"}:
        return False, "unselected figure/image; retained locally"
    if path.suffix.lower() == ".jsonl":
        return False, "unselected line-oriented audit/log; retained locally"
    return False, "outside the declared compact-review format allowlist; retained locally"


class HashingReader:
    """Hash the exact byte stream handed to tar without retaining it in memory."""

    def __init__(self, source: BinaryIO):
        """Wrap a read-only source stream."""
        self.source = source
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        """Read and account for the bytes tar actually consumes."""
        data = self.source.read(size)
        self.digest.update(data)
        self.count += len(data)
        return data


def tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(safe_member(name))
    info.size, info.mtime, info.mode = size, 0, 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.type = tarfile.REGTYPE
    info.pax_headers = {}
    return info


def write_archive(
    destination: Path,
    originals: dict[str, tuple[Path, dict, tuple[int, ...]]],
    generated: dict[str, bytes],
) -> list[dict[str, Any]]:
    members = []
    require(not set(originals).intersection(generated), "original/generated archive path collision")
    with (
        destination.open("xb") as raw,
        lzma.LZMAFile(
            raw, mode="w", format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=9
        ) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for name in sorted(set(originals) | set(generated)):
            if name in generated:
                payload = generated[name]
                archive.addfile(tar_info(name, len(payload)), io.BytesIO(payload))
                digest, size = hashlib.sha256(payload).hexdigest(), len(payload)
                origin = "generated"
            else:
                path, record, identity = originals[name]
                require(
                    stable_identity(regular_file(path)) == identity,
                    f"source changed before archiving: {path}",
                )
                with path.open("rb") as source:
                    stream = HashingReader(source)
                    archive.addfile(tar_info(name, record["size_bytes"]), stream)
                digest, size = stream.digest.hexdigest(), stream.count
                require(
                    digest == record["sha256"]
                    and size == record["size_bytes"]
                    and stable_identity(regular_file(path)) == identity,
                    f"source changed during archiving: {path}",
                )
                origin = "original"
            members.append(
                {"archive_path": name, "sha256": digest, "size_bytes": size, "origin": origin}
            )
    return members


def review_text(main: dict, inventory: list[dict]) -> bytes:
    selected = sum(row["archive_path"] is not None for row in inventory)
    local = len(inventory) - selected
    return (
        "# Review-only actuator study bundle\n\n"
        f"The sealed main campaign records {main['completed']}/{main['planned']} completed "
        f"method episodes. Protocol SHA-256: `{main['protocol_sha256']}`.\n\n"
        f"This compact archive includes {selected} original files; {local} inventoried files "
        "remain local-only. The detached manifest.json records the original repository-relative "
        "path, exact SHA-256, byte size, and inclusion or omission reason for every source file. "
        "Failed, interrupted, negative, and tuning metadata use the same inclusion rule as "
        "successful metadata. Collection does not rerun or validate the scientific results.\n\n"
        "This is an incomplete subset for full physical replay. Most rollout/control arrays, "
        "event streams, and saved learner snapshots are omitted. The selected candidate F2/A "
        "dense and command-application arrays do not include the controls needed by the full "
        "video replay validator. Shared-state probe arrays and selected initial deployments "
        "are included; their broader dependencies may remain local. Video files and optimized "
        "compilation-cache binaries are local-only. The selected PNG is an audit poster, not "
        "a new experiment. Exact float32 replay additionally depends on compatible preserved "
        "optimized executables and the recorded software/GPU environment.\n\n"
        "Original members preserve repository-relative paths and exact bytes. Generated "
        "review/checksum members are under __publication__/. Tar entries have mtime 0, uid/gid "
        "0, empty owner names, and regular-file mode 0644. Source files are never overwritten.\n\n"
        "Verify every included byte, without extraction or Git:\n\n"
        "```sh\npython -m benchmark.da_plcbf_actuator_publication verify "
        "--publication-dir /absolute/path/to/publication-directory\n```\n\n"
        "The verifier checks the detached archive digest, exact member set, member sizes and "
        "digests, canonical headers and SHA256SUMS. It rejects unexpected members, traversal, "
        "duplicate names, symlinks and other nonregular entries. Authenticity of the whole "
        "bundle still requires retaining the detached manifest digest through a trusted channel.\n"
    ).encode()


def reviewer_readme(main: dict, review: dict, inventory: list[dict]) -> bytes:
    def link(label: str, path: str) -> str:
        return f"[{label}](<../{safe_member(path)}>)"

    records = {row["original_repo_path"]: row for row in inventory}
    movie = records[review["video_path"]]
    lines = [
        "# Actuator study reviewer guide",
        "",
        f"The selected video illustrates one recorded {review['video_scope']['execution_mode']}/"
        f"{review['video_scope']['plant_level']} F2/A case. Broad comparisons "
        "come from the sealed main analysis; timed adaptation and transfer claims require "
        "their separately declared experiments. Candidate-case transfer results remain separate "
        "from validation sensitivity curves. The collector does not decide the scientific "
        "verdict; use the completed report's evidence and qualifications.",
        "",
        f"Sealed main: {main['completed']}/{main['planned']} method episodes. Numerical source "
        f"revision: `{main['sealed_numerical_revision']}`. Final reporting revision: "
        f"`{review['reporting_revision']}` (caller supplied; individual source bytes are hashed).",
        "",
        "Links resolve in the extracted archive layout. MP4 files remain local-only.",
        "",
    ]
    lines.extend(
        f"- {link('Research report' if index == 0 else 'Additional report', path)}"
        for index, path in enumerate(review["reports"])
    )
    lines.extend(
        [
            f"- {link('Technical note', review['technical_note'])}",
            f"- {link('Sealed protocol', main['protocol_path'])} · "
            f"{link('Main analysis', review['main_analysis'])}",
            "- "
            + link(
                "Main campaign and all method/seed summaries",
                main["campaign_directory"] + "/campaign_result.json",
            ),
        ]
    )
    for stem in FIGURE_STEMS:
        prefix = review["figure_directory"] + "/" + stem
        title = stem[3:].replace("_", " ").capitalize()
        lines.append(
            f"- {title}: {link('PNG', prefix + '.png')} · "
            f"{link('PDF', prefix + '.pdf')} · {link('Data', prefix + '.csv')}"
        )
    if review["promoted_case_campaigns"]:
        lines += ["", "Selected development-case timing/transfer evidence (descriptive):", ""]
        lines.extend(
            f"- {link(path.rsplit('/', 1)[-1], path + '/campaign_result.json')}"
            for path in review["promoted_case_campaigns"]
        )
    lines += [
        "",
        f"Video: `{review['video_path']}` ({movie['size_bytes']} bytes, omitted). "
        f"SHA-256: `{movie['sha256']}`. "
        f"{link('Selected poster', review['video_poster'])} · "
        f"{link('Frame audit', review['video_directory'] + '/frame_audit.jsonl')} · "
        f"{link('Render binding', review['video_directory'] + '/render_binding.json')}.",
    ]
    for companion in review.get("companion_videos", []):
        record = records[companion["video_path"]]
        lines += [
            "",
            f"Separate companion ({companion['execution_mode']}/{companion['plant_level']}): "
            f"`{companion['video_path']}` ({record['size_bytes']} bytes, omitted). "
            f"SHA-256: `{record['sha256']}`. "
            f"{link('Poster', companion['video_poster'])} · "
            f"{link('Frame audit', companion['video_directory'] + '/frame_audit.jsonl')} · "
            f"{link('Render binding', companion['video_directory'] + '/render_binding.json')}.",
        ]
    lines += [
        "",
        "Study reports, campaign summaries and tuning metadata are included without filtering "
        "by outcome, including negative results, interrupted attempts, tuning decisions "
        "and numerical-cache sensitivity. "
        "Most raw rollout arrays, event streams and optimized caches remain local-only. "
        "This compact bundle cannot support complete physical replay on its own; see "
        "[inclusion and verification scope](REVIEW_ONLY.md).",
        "",
    ]
    return "\n".join(lines).encode()


def collect(args: argparse.Namespace) -> dict[str, Any]:
    study = source_path(args.study_root)
    main_directory = source_path(args.main_campaign)
    require(main_directory.is_relative_to(study), "main campaign must be inside the study root")
    expected_hashes = {}
    main = complete_main(main_directory, expected_hashes)  # Refuse active main before output.
    figures, video = source_path(args.figure_dir), source_path(args.video_dir)
    selected_figures, figure_hashes = figure_files(figures)
    selected_video, video_hashes, video_scope = video_files(video, args.video_poster)
    add_declared_hashes(expected_hashes, figure_hashes, "figure input/output references")
    add_declared_hashes(expected_hashes, video_hashes, "video input/output references")
    companion_directories, companion_scopes = [], []
    for directory_argument, poster in getattr(args, "companion_video", []):
        directory = source_path(Path(directory_argument))
        require(
            directory != video and directory not in companion_directories,
            "companion video directories must be distinct",
        )
        selected, hashes, scope = video_files(directory, poster)
        add_declared_hashes(expected_hashes, hashes, "companion video input/output references")
        selected_video |= selected
        companion_directories.append(directory)
        companion_scopes.append(
            {
                **scope,
                "video_directory": relative(directory),
                "video_poster": relative(directory / safe_member(poster)),
            }
        )
    reports = [source_path(path) for path in args.report]
    require(
        all(path.suffix.lower() in METADATA_SUFFIXES for path in reports),
        "explicit reports must be metadata/source text",
    )
    technical_note = source_path(args.technical_note)
    require(
        technical_note.suffix.lower() in METADATA_SUFFIXES,
        "technical note must be metadata/source text",
    )
    analysis_path = source_path(args.main_analysis)
    analysis = read_json(analysis_path, expected_hashes)
    require(
        analysis.get("schema") == "da_plcbf_actuator_analysis_v1"
        and analysis.get("input_kind") == "sealed_ledger"
        and analysis.get("protocol_sha256") == main["protocol_sha256"]
        and len(analysis.get("rows", [])) == main["planned"],
        "reviewer main analysis does not match the completed sealed main",
    )
    require(
        relative(analysis_path) in figure_hashes
        and figure_hashes[relative(analysis_path)] == expected_hashes[relative(analysis_path)],
        "final figures do not bind the supplied main analysis",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", args.reporting_revision) is not None,
        "reporting revision must be an explicit full 40-character Git revision",
    )
    promoted_campaigns = [source_path(path) for path in args.promoted_case_campaign]
    for path in promoted_campaigns:
        require(path.is_relative_to(study), "promoted-case campaign must lie inside study root")
        result = read_json(path / "campaign_result.json", expected_hashes)
        require(
            type(result.get("planned")) is int
            and result["planned"] > 0
            and result.get("completed") == result["planned"]
            and len(result.get("records", [])) == result["planned"]
            and all(
                row.get("summary", {}).get("status") == "completed" for row in result["records"]
            ),
            "promoted-case campaign is incomplete",
        )
    review_inputs = {
        "reports": [relative(path) for path in reports],
        "technical_note": relative(technical_note),
        "main_analysis": relative(analysis_path),
        "reporting_revision": args.reporting_revision,
        "figure_directory": relative(figures),
        "video_directory": relative(video),
        "video_path": video_scope["video_path"],
        "video_scope": video_scope,
        "video_poster": relative(video / safe_member(args.video_poster)),
        "companion_videos": companion_scopes,
        "promoted_case_campaigns": [relative(path) for path in promoted_campaigns],
    }
    explicit_files = [*reports, technical_note, analysis_path, Path(__file__).resolve()]
    output = args.output.absolute()
    require(not output.exists() and not output.is_symlink(), "output must be a fresh directory")
    output = output.resolve()
    origins: dict[Path, set[str]] = collections.defaultdict(set)
    for directory, origin in (
        (study, "study_root"),
        (figures, "explicit_figure_directory"),
        (video, "explicit_video_directory"),
        *((directory, "explicit_companion_video_directory") for directory in companion_directories),
    ):
        for path in walk_files(directory, output):
            origins[path].add(origin)
    for path in explicit_files:
        regular_file(path)
        origins[path].add(
            "collector_source"
            if path == Path(__file__).resolve()
            else "explicit_report_or_reference"
        )
    referenced_files = {ROOT / name for name in expected_hashes}
    for path in referenced_files:
        regular_file(path)
        origins[path].add("declared_figure_video_or_main_input")
    protocol_path = ROOT / main["protocol_path"]
    origins[protocol_path].add("sealed_main_protocol")
    selected_media = selected_figures | selected_video
    inventory, originals, identities = [], {}, {}
    output.mkdir(parents=True, exist_ok=False)
    try:
        for path in sorted(origins, key=relative):
            digest, size, identity = hash_file(path)
            name = relative(path)
            expected_hash = expected_hashes.get(name)
            if expected_hash is not None:
                require(
                    digest == expected_hash,
                    f"retained source differs from a declared input/output digest: {name}",
                )
            included, reason = inclusion(path, study, selected_media)
            record = {
                "original_repo_path": name,
                "sha256": digest,
                "size_bytes": size,
                "inventory_origins": sorted(origins[path]),
                "archive_path": name if included else None,
                "included_reason": reason if included else None,
                "local_only_reason": None if included else reason,
            }
            inventory.append(record)
            identities[path] = identity
            if included:
                originals[name] = path, record, identity
        review = review_text(main, inventory)
        readme = reviewer_readme(main, review_inputs, inventory)
        checksums = {name: record[1]["sha256"] for name, record in originals.items()}
        checksums[REVIEW_MEMBER] = hashlib.sha256(review).hexdigest()
        checksums[README_MEMBER] = hashlib.sha256(readme).hexdigest()
        checksum_bytes = "".join(
            f"{checksums[name]}  {name}\n" for name in sorted(checksums)
        ).encode()
        generated = {REVIEW_MEMBER: review, README_MEMBER: readme, CHECKSUM_MEMBER: checksum_bytes}
        for filename, payload in (
            ("REVIEW_ONLY.md", review),
            ("README.md", readme),
            ("SHA256SUMS", checksum_bytes),
        ):
            with (output / filename).open("xb") as stream:
                stream.write(payload)
        archive = output / ARCHIVE_NAME
        members = write_archive(archive, originals, generated)
        for path, identity in identities.items():
            require(
                stable_identity(regular_file(path)) == identity,
                f"inventoried source changed during collection: {path}",
            )
        current_paths = set()
        for directory in (study, figures, video, *companion_directories):
            current_paths.update(walk_files(directory, output))
        current_paths.update([*explicit_files, protocol_path])
        current_paths.update(referenced_files)
        require(current_paths == set(origins), "source file set changed during collection")
        archive_sha, archive_size, _ = hash_file(archive)
        manifest = {
            "schema": SCHEMA,
            "status": "completed_review_only_collection",
            "study_root": relative(study),
            "main": main,
            "figure_directory": relative(figures),
            "video_directory": relative(video),
            "selected_video_poster": relative(video / safe_member(args.video_poster)),
            "explicit_reports": [relative(path) for path in reports],
            "reviewer_inputs": review_inputs,
            "declared_input_output_sha256": dict(sorted(expected_hashes.items())),
            "inventory_exclusions": ["current output directory", "__pycache__ directories"],
            "inventory_scope": (
                "Every study file plus explicit figure/video directories, "
                "reports/references, every declared figure/video/main input, "
                "protocol and collector source"
            ),
            "all_original_files": inventory,
            "archive": {
                "filename": ARCHIVE_NAME,
                "sha256": archive_sha,
                "size_bytes": archive_size,
            },
            "archive_members": members,
            "archive_policy": {
                "compression": "XZ CRC64 preset9; deterministic sorted POSIX members",
                "tar_format": "PAX with only necessary path metadata",
                "mtime": 0,
                "uid": 0,
                "gid": 0,
                "mode": "0644",
                "source_bytes_preserved": True,
                "source_files_overwritten": False,
                "generated_members": [CHECKSUM_MEMBER, README_MEMBER, REVIEW_MEMBER],
                "sha256sums_scope": "every archive member except SHA256SUMS itself",
                "metadata_selection_uses_outcomes": False,
            },
            "counts": {
                "original_files": len(inventory),
                "included_original_files": len(originals),
                "local_only_original_files": len(inventory) - len(originals),
                "original_bytes": sum(row["size_bytes"] for row in inventory),
                "included_original_bytes": sum(
                    row["size_bytes"] for row in inventory if row["archive_path"] is not None
                ),
                "archive_members": len(members),
            },
            "collector_source_sha256": next(
                row["sha256"]
                for row in inventory
                if row["original_repo_path"] == relative(Path(__file__).resolve())
            ),
        }
        with (output / "manifest.json").open("xb") as stream:
            stream.write(json_bytes(manifest))
        result = verify(output)
        with (output / "verification.json").open("xb") as stream:
            stream.write(json_bytes(result))
        return result
    except Exception as error:
        with (output / "FAILED.json").open("xb") as stream:
            stream.write(
                json_bytes(
                    {
                        "status": "failed_collection",
                        "error_type": type(error).__name__,
                        "reason": str(error),
                        "inventoried_original_files": inventory,
                    }
                )
            )
        raise


def verify(directory: Path) -> dict[str, Any]:
    """Authenticate every archived member without extracting or invoking external commands."""
    require(not directory.is_symlink(), "publication directory cannot be a symlink")
    directory = directory.resolve(strict=True)
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    require(
        manifest.get("schema") == SCHEMA
        and manifest.get("status") == "completed_review_only_collection",
        "unsupported or incomplete publication manifest",
    )
    archive_record = manifest["archive"]
    require(archive_record["filename"] == ARCHIVE_NAME, "unexpected archive filename")
    archive = directory / ARCHIVE_NAME
    archive_sha, archive_size, _ = hash_file(archive)
    require(
        archive_sha == archive_record["sha256"] and archive_size == archive_record["size_bytes"],
        "archive SHA-256 or size mismatch",
    )
    expected = {}
    for member in manifest["archive_members"]:
        name = safe_member(member["archive_path"])
        require(name not in expected, f"duplicate manifest archive member: {name}")
        require(
            type(member["size_bytes"]) is int
            and member["size_bytes"] >= 0
            and isinstance(member["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", member["sha256"]) is not None
            and member["origin"] in {"original", "generated"},
            f"invalid manifest archive member: {name}",
        )
        expected[name] = member
    original_names, included_originals = set(), set()
    for record in manifest["all_original_files"]:
        name = safe_member(record["original_repo_path"])
        require(name not in original_names, f"duplicate original source inventory entry: {name}")
        original_names.add(name)
        require(
            type(record["size_bytes"]) is int
            and record["size_bytes"] >= 0
            and isinstance(record["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
            f"invalid original source inventory entry: {name}",
        )
        if record["archive_path"] is None:
            require(
                bool(record["local_only_reason"]) and record["included_reason"] is None,
                "omitted source lacks its omission reason",
            )
        else:
            require(
                record["archive_path"] == name
                and bool(record["included_reason"])
                and record["local_only_reason"] is None,
                "original archive path or inclusion reason is inconsistent",
            )
            require(
                name in expected
                and expected[name]["origin"] == "original"
                and expected[name]["sha256"] == record["sha256"]
                and expected[name]["size_bytes"] == record["size_bytes"],
                "archive member differs from original inventory",
            )
            included_originals.add(name)
    require(
        included_originals
        == {name for name, record in expected.items() if record["origin"] == "original"},
        "original inventory and archive manifest do not match",
    )
    inventory_digests = {
        record["original_repo_path"]: record["sha256"] for record in manifest["all_original_files"]
    }
    declarations = manifest.get("declared_input_output_sha256")
    require(
        isinstance(declarations, dict) and bool(declarations), "declared input hashes are absent"
    )
    for name, digest in declarations.items():
        safe_member(name)
        require(
            name in inventory_digests and inventory_digests[name] == digest,
            f"declared figure/video/main input differs from its retained inventory: {name}",
        )
    generated_members = {README_MEMBER, REVIEW_MEMBER, CHECKSUM_MEMBER}
    require(
        {name for name, record in expected.items() if record["origin"] == "generated"}
        == generated_members,
        "unexpected generated archive members",
    )
    seen, generated_bytes = set(), {}
    member_end = 0
    with tarfile.open(archive, mode="r:xz") as stream:
        for member in stream:
            require(member.offset == member_end, "unexpected bytes between archive members")
            name = safe_member(member.name)
            require(name not in seen, f"duplicate archive member: {name}")
            require(name in expected, f"unexpected archive member: {name}")
            require(
                member.isreg()
                and not member.issym()
                and not member.islnk()
                and not member.linkname,
                f"nonregular or linked archive member: {name}",
            )
            require(
                member.mtime == 0
                and member.uid == 0
                and member.gid == 0
                and member.mode == 0o644
                and member.uname == ""
                and member.gname == "",
                f"noncanonical archive ownership/time/mode: {name}",
            )
            require(
                set(member.pax_headers).issubset({"path"})
                and all(value == name for value in member.pax_headers.values()),
                f"unexpected extended archive headers: {name}",
            )
            require(
                member.size == expected[name]["size_bytes"], f"archive member size mismatch: {name}"
            )
            source = stream.extractfile(member)
            require(source is not None, f"unreadable archive member: {name}")
            digest, count = hashlib.sha256(), 0
            pieces = [] if name in generated_members else None
            with source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    count += len(chunk)
                    if pieces is not None:
                        pieces.append(chunk)
            require(
                count == member.size and digest.hexdigest() == expected[name]["sha256"],
                f"archive member SHA-256 mismatch: {name}",
            )
            if pieces is not None:
                generated_bytes[name] = b"".join(pieces)
            seen.add(name)
            member_end = member.offset_data + ((member.size + 511) // 512) * 512
        stream.fileobj.seek(member_end)
        padding_size = 0
        while padding := stream.fileobj.read(1024 * 1024):
            require(not any(padding), "unexpected bytes after the final archive member")
            padding_size += len(padding)
        require(
            padding_size >= 1024 and (member_end + padding_size) % tarfile.RECORDSIZE == 0,
            "missing or noncanonical archive end padding",
        )
    require(seen == set(expected), f"archive is missing members: {sorted(set(expected) - seen)}")
    checksum_entries = {}
    for line in generated_bytes[CHECKSUM_MEMBER].decode("utf-8").splitlines():
        require(len(line) > 66 and line[64:66] == "  ", "malformed SHA256SUMS line")
        digest, name = line[:64], safe_member(line[66:])
        require(
            name not in checksum_entries and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            "duplicate or malformed SHA256SUMS member",
        )
        checksum_entries[name] = digest
    require(
        checksum_entries
        == {name: record["sha256"] for name, record in expected.items() if name != CHECKSUM_MEMBER},
        "SHA256SUMS differs from the exact archive inventory",
    )
    for filename, member in (
        ("README.md", README_MEMBER),
        ("REVIEW_ONLY.md", REVIEW_MEMBER),
        ("SHA256SUMS", CHECKSUM_MEMBER),
    ):
        regular_file(directory / filename)
        require(
            (directory / filename).read_bytes() == generated_bytes[member],
            f"detached {filename} differs from its archived copy",
        )
    counts = manifest["counts"]
    require(
        counts["original_files"] == len(original_names)
        and counts["included_original_files"] == len(included_originals)
        and counts["local_only_original_files"] == len(original_names) - len(included_originals)
        and counts["archive_members"] == len(expected),
        "manifest counts do not match inventories",
    )
    return {
        "schema": "da_plcbf_actuator_publication_verification_v1",
        "status": "verified",
        "manifest_sha256": hash_file(manifest_path)[0],
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "verified_archive_members": len(seen),
        "included_original_files": len(included_originals),
        "inventoried_local_only_files": len(original_names) - len(included_originals),
        "all_included_bytes_match_manifest": True,
        "all_declared_input_output_digests_match_inventory": True,
        "extraction_performed": False,
        "local_only_payloads_rehashed_by_verify": False,
        "scope": (
            "Archive and detached manifest consistency; omitted local payloads "
            "are not contained or independently authenticated by this command"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("collect", help="Collect only after the sealed main is complete")
    build.add_argument("--study-root", type=Path, required=True)
    build.add_argument("--main-campaign", type=Path, required=True)
    build.add_argument("--figure-dir", type=Path, required=True)
    build.add_argument("--video-dir", type=Path, required=True)
    build.add_argument(
        "--video-poster", required=True, help="Explicit PNG path relative to --video-dir"
    )
    build.add_argument(
        "--companion-video",
        nargs=2,
        action="append",
        default=[],
        metavar=("DIRECTORY", "POSTER"),
        help="Separate completed video and its relative PNG poster; repeat as needed",
    )
    build.add_argument("--report", action="append", type=Path, required=True)
    build.add_argument("--technical-note", type=Path, required=True)
    build.add_argument(
        "--main-analysis",
        type=Path,
        required=True,
        help="Completed sealed analysis.json used by the figures",
    )
    build.add_argument(
        "--reporting-revision",
        required=True,
        help="Full final Git revision, recorded separately from sealed numerical source",
    )
    build.add_argument(
        "--promoted-case-campaign",
        action="append",
        type=Path,
        default=[],
        help="Completed descriptive candidate timing/transfer campaign; repeat as needed",
    )
    build.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify", help="Authenticate archive members without extraction")
    check.add_argument("--publication-dir", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args) if args.command == "collect" else verify(args.publication_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
