"""Publish complete closed-loop evidence with videos and candidate trajectories kept locally.

``prepare`` derives lossless completed-trial ledgers and exact compact control traces.
``inventory`` must run last, after figures, reviews, tests, and video audits finish.
It records every local artifact and writes a NUL-delimited list for explicit Git staging.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

BASE_COMMIT = "0bcd4a17b03d0fc99f4bdcc024b866090072fa43"
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v", ".mpeg", ".mpg"}
BULK_NAMES = {
    "navigation_comparison.npz",
    "raw_diagnostics.npz",
    "same_state_probe_trajectories.npz",
}
NAVIGATION_BULK_SUFFIXES = ("_nominal_rollout", "_fallback_rollouts", "_selected_rollout")


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identity(scene: dict, mapping: str) -> str:
    value = {"scene": {k: v for k, v in scene.items() if k != "seed"}, "mapping": mapping}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def archived_sources(directory: Path, protocol: dict, row: dict, repository: Path) -> dict:
    """Verify execution copies, including early protocols with absolute source names."""
    execution_id = row.get("source_execution_id") or protocol.get("source_execution_id")
    archive = directory / "source_versions" / execution_id if execution_id else directory / "source"
    source_manifest = archive / "SOURCE.json"
    hashes = (
        json.loads(source_manifest.read_text())
        if source_manifest.exists()
        else protocol["source_hashes"]
    )
    verified = {}
    for filename, expected in hashes.items():
        relative = Path(filename)
        if relative.is_absolute():
            relative = relative.relative_to(repository)
        candidate = archive / relative
        if digest(candidate) != expected:
            raise ValueError(f"Archived execution source differs: {candidate}")
        verified[str(relative)] = expected
    if execution_id:
        actual = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        if actual != execution_id:
            raise ValueError(f"Execution source identity differs: {archive}")
    return {"archive": str(archive.relative_to(directory.parent)), "files": verified}


def completed_ledgers(root: Path, repository: Path) -> tuple[list[dict], list[dict], dict]:
    """Count only ledger-backed completed physical pairs; preserve every row, including failures."""
    discovery, smoke, provenance, unledgered, other = [], [], {}, [], {}
    for ledger in sorted(root.glob("*/trials.jsonl")):
        directory = ledger.parent
        protocol_path = directory / "protocol.json"
        protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
        rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        if not protocol.get("discovery_only"):
            pairs = [row for row in rows if set(row.get("methods", {})) == {"fixed", "adaptive"}]
            other[directory.name] = {
                "ledger_rows": len(rows),
                "paired_comparisons": len(pairs),
                "new_method_episodes": sum(
                    1 if row.get("variant") == "freeze_at_wind_onset" else 2 for row in pairs
                ),
                "reused_adaptive_comparisons": sum(
                    row.get("variant") == "freeze_at_wind_onset" for row in pairs
                ),
                "sha256": digest(ledger),
            }
            continue
        retained, incomplete, bindings, source_archives = [], [], {}, {}
        for line, row in enumerate(rows, 1):
            if set(row.get("methods", {})) != {"fixed", "adaptive"}:
                incomplete.append(line)
                continue
            trial_id = row["trial_id"]
            if Path(trial_id).name != trial_id:
                raise ValueError(f"Invalid trial identifier: {trial_id}")
            result_path, trace_path = (
                directory / trial_id / "result.json",
                directory / trial_id / "traces.npz",
            )
            result = json.loads(result_path.read_text())
            if any(row.get(key) != value for key, value in result.items()):
                raise ValueError(f"Ledger differs from completed result: {result_path}")
            physical_id = identity(row["scene"], row["mapping"])
            if row.get("physical_scene_identity", physical_id) != physical_id:
                raise ValueError(f"Physical identity differs: {result_path}")
            source = archived_sources(directory, protocol, row, repository)
            source_archives[source["archive"]] = source["files"]
            bindings[trial_id] = {
                "result_sha256": digest(result_path),
                "trace_sha256": digest(trace_path),
                "physical_scene_identity": physical_id,
                "execution_source_archive": source["archive"],
            }
            retained.append(trial_id)
            item = {"batch": directory.name, "source_line": line, "record": row}
            (smoke if directory.name.startswith("smoke-") else discovery).append(item)
        unledgered.extend(
            str(path.relative_to(root))
            for path in sorted(directory.glob(f"{directory.name}-*"))
            if path.is_dir() and path.name not in retained
        )
        provenance[str(ledger.relative_to(root))] = {
            "sha256": digest(ledger),
            "protocol_sha256": digest(protocol_path),
            "completed_pairs": len(retained),
            "incomplete_ledger_lines": incomplete,
            "trial_bindings": bindings,
            "verified_execution_sources": source_archives,
        }
    for summary_path in sorted(root.glob("targeted-confirmation-*/SUMMARY.json")):
        summary = json.loads(summary_path.read_text())
        pairs = [
            row for row in summary.values() if set(row.get("methods", {})) == {"fixed", "adaptive"}
        ]
        other[summary_path.parent.name] = {
            "summary": str(summary_path.relative_to(root)),
            "sha256": digest(summary_path),
            "paired_comparisons": len(pairs),
            "new_method_episodes": 2 * len(pairs),
        }
    return (
        discovery,
        smoke,
        {
            "ledger_sources": provenance,
            "other_ledgers_not_counted_as_discovery": other,
            "unledgered_attempt_directories_not_counted": unledgered,
        },
    )


def aggregate(items: list[dict]) -> dict:
    rows = [item["record"] for item in items]
    unique = {identity(row["scene"], row["mapping"]) for row in rows}
    return {
        "completed_pairs": len(rows),
        "distinct_scene_mapping_pairs_ignoring_seed": len(unique),
        "duplicate_execution_pairs": len(rows) - len(unique),
        "by_batch": dict(sorted(Counter(item["batch"] for item in items).items())),
        "by_family": dict(sorted(Counter(row["family"] for row in rows).items())),
        "by_mapping": dict(sorted(Counter(row["mapping"] for row in rows).items())),
        "outcomes": dict(sorted(Counter(row["outcome_class"] for row in rows).items())),
        "method_terminations": {
            method: dict(
                sorted(Counter(row["methods"][method]["termination"] for row in rows).items())
            )
            for method in ("fixed", "adaptive")
        },
        "all_operational_nodes_pass": all(
            row["methods"][method]["all_operational_nodes_pass"]
            for row in rows
            for method in ("fixed", "adaptive")
        ),
        "promotion_candidates": [
            {"batch": item["batch"], "trial_id": item["record"]["trial_id"]}
            for item in items
            if item["record"]["promotion_candidate"]
        ],
    }


def compact_trace(path: Path, root: Path) -> dict:
    """Retain every non-trajectory array exactly; reject conflicting duplicate keys."""
    arrays, sources = {}, {}
    for source in (path, path.parent / "raw_diagnostics.npz"):
        if not source.exists():
            raise FileNotFoundError(source)
        before = digest(source)
        kept, excluded = [], {}
        with np.load(source, allow_pickle=False) as archive:
            for key in archive.files:
                value = archive[key]
                bulk = (
                    key.endswith(NAVIGATION_BULK_SUFFIXES)
                    if source == path
                    else key.endswith("_candidate_wrenches")
                )
                if bulk:
                    excluded[key] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "bytes": value.nbytes,
                    }
                    continue
                if key in arrays:
                    np.testing.assert_array_equal(arrays[key], value, strict=True)
                arrays[key] = value
                kept.append(key)
        if digest(source) != before:
            raise ValueError(f"Source changed during derivation: {source}")
        sources[str(source.relative_to(root))] = {
            "sha256": before,
            "retained_keys": kept,
            "excluded_arrays": excluded,
        }
    target = path.parent / "compact_control_trace.npz"
    np.savez_compressed(target, **arrays)
    with np.load(target, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError(f"Saved keys differ: {target}")
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value, strict=True)
    probe = path.parent / "same_state_probe_trajectories.npz"
    return {
        "sha256": digest(target),
        "arrays": len(arrays),
        "sources": sources,
        "all_saved_arrays_verified_exact": True,
        "probe_trajectory_source_sha256": digest(probe) if probe.exists() else None,
        "scope": (
            "All non-trajectory control and raw diagnostic arrays, preserving dtype and padding "
            "masks. Candidate trajectories remain local; probe scalar summaries remain in "
            "navigation_comparison.json. Dense physical states remain separately published."
        ),
    }


def prepare(root: Path, repository: Path) -> dict:
    output = root / "publication"
    output.mkdir(parents=True, exist_ok=True)
    discovery, smoke, provenance = completed_ledgers(root, repository)
    for name, rows in (("all_completed_discovery", discovery), ("all_completed_smoke", smoke)):
        encoded = "".join(
            json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows
        ).encode()
        compressed = gzip.compress(encoded, mtime=0)
        target = output / f"{name}.jsonl.gz"
        target.write_bytes(compressed)
        decoded = [json.loads(line) for line in gzip.decompress(compressed).splitlines()]
        if decoded != rows:
            raise ValueError(f"Merged ledger roundtrip differs: {target}")
    counts = {
        "scope": (
            "Completed physical discovery pairs only. Smoke, confirmation, paced replay, "
            "unexecuted proposals, and historical cached geometry are separate. "
            "No initial-H gate was used."
        ),
        "discovery": aggregate(discovery),
        "smoke": aggregate(smoke),
        "historical_cached_geometry_not_counted_as_new_trials": True,
        **provenance,
    }
    write_json(output / "COMPLETED_TRIAL_COUNTS.json", counts)
    traces = {
        str(path.parent.relative_to(root) / "compact_control_trace.npz"): compact_trace(path, root)
        for path in sorted(root.rglob("navigation_comparison.npz"))
    }
    derivation = {"completed_ledgers": provenance["ledger_sources"], "traces": traces}
    write_json(output / "DERIVATION.json", derivation)
    return {
        "discovery": counts["discovery"],
        "smoke": counts["smoke"],
        "compact_traces": len(traces),
    }


def include(path: Path, root: Path) -> tuple[bool, str]:
    relative = path.relative_to(root)
    if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
        return False, "generated interpreter cache stays local"
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return False, "video stays local; metadata and final stills are published"
    if path.name in BULK_NAMES:
        return (
            False,
            "candidate trajectory tensor stays local; source hash and exact compact "
            "control derivation are published",
        )
    if relative.parts[0] == "figures-v1":
        return False, "superseded figure layout; final figures-v2 are published"
    if relative.parts[0] == "videos" and len(relative.parts) > 2:
        if relative.parts[1].endswith("-v1") or relative.parts[1] == "camera-preview":
            return False, "superseded or preview camera; final v2 stills and metadata are published"
        review_frames = {
            "paced-collision-v2": {0, 70, 93, 110, 214},
            "paced-compensated-v2": {0, 94, 239},
        }
        if relative.parts[1] in review_frames and path.name.startswith("frame_"):
            selected = int(path.name.split("_")[1]) in review_frames[relative.parts[1]]
            return selected, (
                "representative calm/wind/contact/completion still"
                if selected
                else "additional decoded frame stays local; full video review and hashes retained"
            )
    if any(part.startswith("partial_attempt_") or ".partial-" in part for part in relative.parts):
        return False, "unledgered partial attempt is excluded from completed trial counts"
    return True, "compact numerical, source, contact, figure, or review evidence"


def source_delta(output: Path, repository: Path, base_commit: str) -> dict:
    tracked = subprocess.check_output(
        ["git", "diff", "--name-only", "-z", base_commit, "--"], cwd=repository
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repository
    )
    names = sorted({name.decode() for name in (tracked + untracked).split(b"\0") if name})
    source_hashes, deleted, excluded_media = {}, [], []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in names:
            relative = Path(name)
            if relative.parts[0] == "artifacts":
                continue
            if relative.suffix.lower() in VIDEO_SUFFIXES or relative.name in BULK_NAMES:
                excluded_media.append(name)
                continue
            path = repository / relative
            if not path.exists():
                deleted.append(name)
                continue
            if not path.is_file():
                continue
            data = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            archive.addfile(info, io.BytesIO(data))
            source_hashes[name] = hashlib.sha256(data).hexdigest()
    target = output / "source_delta.tar.gz"
    target.write_bytes(gzip.compress(buffer.getvalue(), mtime=0))
    record = {
        "base_commit": base_commit,
        "scope": (
            "Changed/new non-artifact workspace files over the named base, collected before "
            "implementation commit; deletions listed explicitly."
        ),
        "archive_sha256": digest(target),
        "files": source_hashes,
        "deleted_files": deleted,
        "excluded_media_and_bulk_files": excluded_media,
    }
    write_json(output / "SOURCE.json", record)
    return record


def inventory(root: Path, repository: Path, base_commit: str = BASE_COMMIT) -> dict:
    """Refresh derivations and inventory only after the artifact producers have stopped."""
    prepare(root, repository)
    output = root / "publication"
    source_delta(output, repository, base_commit)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"PUBLICATION_MANIFEST.json", "git-paths.nul"}:
            continue
        included, reason = include(path, root)
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
                "included": included,
                "reason": reason,
            }
        )
    summary = {
        "included_files": sum(row["included"] for row in files),
        "included_bytes": sum(row["bytes"] for row in files if row["included"]),
        "local_only_files": sum(not row["included"] for row in files),
        "local_only_bytes": sum(row["bytes"] for row in files if not row["included"]),
        "included_video_files": sum(
            row["included"] and Path(row["path"]).suffix.lower() in VIDEO_SUFFIXES for row in files
        ),
    }
    if summary["included_video_files"]:
        raise ValueError("Videos may not be published")
    manifest = output / "PUBLICATION_MANIFEST.json"
    write_json(
        manifest,
        {
            "scope": "Complete compact closed-loop evidence; videos and candidate "
            "trajectories remain local",
            "files": files,
            **summary,
        },
    )
    paths = [root / row["path"] for row in files if row["included"]] + [manifest]
    (output / "git-paths.nul").write_bytes(
        b"\0".join(str(path.relative_to(repository)).encode() for path in paths) + b"\0"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "inventory"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-commit", default=BASE_COMMIT)
    args = parser.parse_args()
    repository = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    ).resolve()
    root = args.root.resolve()
    result = (
        prepare(root, repository)
        if args.stage == "prepare"
        else inventory(root, repository, args.base_commit)
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
