"""Publish compact, auditable case-study evidence while retaining videos and bulk locally."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path) -> None:
    output = root / "publication"
    output.mkdir(exist_ok=False)
    ledger, provenance = [], {}
    for directory in sorted(root.glob("geometry-v*")):
        path = directory / "ledger.json"
        rows = json.loads(path.read_text())
        provenance[str(path.relative_to(root))] = {"sha256": digest(path), "rows": len(rows)}
        ledger.extend({"family": directory.name, **row} for row in rows)
    text = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in ledger)
    compressed = gzip.compress(text.encode(), mtime=0)
    (output / "all_candidates.jsonl.gz").write_bytes(compressed)
    assert len(gzip.decompress(compressed).splitlines()) == len(ledger)
    traces = {}
    names = (
        "full_state",
        "applied_wrench",
        "nominal_wrench",
        "selected_policy",
        "qp_valid",
        "used_fallback",
        "used_emergency",
        "degraded",
        "library_version",
        "cumulative_gradient_steps",
        "snapshot_age_seconds",
        "controller_seconds",
        "learner_seconds",
        "missed_deadline",
        "recorded_control_valid",
        "execution_mode",
        "eligible_candidate_count",
        "executed_policy_dual",
        "actuator_margins",
        "operational_residuals",
        "applied_held_operational_residuals",
        "applied_held_physical_margins",
        "predictive_operational_iterations",
        "qp_rejection_flags",
        "goal_position",
        "estimated_wind",
    )
    for path in sorted(root.rglob("*.npz")):
        if path.name not in {"comparison.npz", "navigation_comparison.npz"}:
            continue
        arrays = {}
        with np.load(path, allow_pickle=False) as archive:
            for key in (
                "time_seconds",
                "true_wind",
                "obstacle_centers",
                "obstacle_physical_radii",
                "obstacle_inflated_radii",
            ):
                arrays[key] = archive[key]
            for method in ("fixed", "adaptive"):
                for name in names:
                    key = f"{method}_{name}"
                    if key in archive:
                        arrays[key] = archive[key]
        raw = path.parent / "raw_diagnostics.npz"
        if raw.exists():
            with np.load(raw, allow_pickle=False) as archive:
                for key in archive.files:
                    if "candidate_wrenches" not in key:
                        arrays[key] = archive[key]
        else:
            for method, filename in (
                ("fixed", "fixed_controls.json"),
                ("adaptive", "adaptive_held_controls.json"),
            ):
                rows = json.loads((path.parent / filename).read_text())
                for name in ("hard", "smooth", "eligible"):
                    arrays[f"{method}_{name}"] = np.asarray([row[name] for row in rows])
        target = path.parent / "compact_control_trace.npz"
        if target.exists():
            raise FileExistsError(target)
        np.savez_compressed(target, **arrays)
        with np.load(target, allow_pickle=False) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
        traces[str(target.relative_to(root))] = {
            "source": str(path.relative_to(root)),
            "source_sha256": digest(path),
            "sha256": digest(target),
            "all_saved_arrays_verified_exact": True,
            "raw_source_sha256": digest(raw) if raw.exists() else None,
            "scope": "complete control interval, no large candidate rollout tensor",
        }
    (output / "DERIVATION.json").write_text(
        json.dumps(
            {
                "candidate_ledger_sources": provenance,
                "candidate_count": len(ledger),
                "traces": traces,
            },
            indent=2,
        )
        + "\n"
    )


def include(path: Path, root: Path) -> tuple[bool, str]:
    relative = path.relative_to(root)
    if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
        return False, "generated interpreter cache stays local"
    if path.suffix.lower() in {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}:
        return False, "generated video stays local"
    if relative.parts[0] in {"closed-loop-v1", "qp-v1", "mechanism-v1", "figures-v1"}:
        return False, "superseded or incomplete development output"
    if path.name in {
        "comparison.npz",
        "navigation_comparison.npz",
        "same_state_probe_trajectories.npz",
        "raw_diagnostics.npz",
    }:
        return False, "bulk tensor; source hashes and exact compact control trace published"
    if relative.parts[0].startswith("geometry-v") and path.name == "ledger.json":
        return False, "every row published in lossless merged all_candidates.jsonl.gz"
    if (
        relative.parts[0].startswith("runtime-")
        and ("snapshots" in relative.parts or "completed_updates" in relative.parts)
        and path.stem not in {"u0001", "u0002", "nominal_reference"}
    ):
        return (
            False,
            "all-update profiling snapshots stay local; "
            "bound initial/final and update ledger published",
        )
    if relative.parts[0] == "videos":
        keep = (
            path.name in {"VIDEO_REVIEW.json", "render_case_video_v2.py", "SUPERSEDED_V1.json"}
            or "v2" in path.name.lower()
        )
        return (
            keep,
            "final still/provenance" if keep else "preliminary renderer investigation stays local",
        )
    return True, "compact numerical, source, or review evidence"


def inventory(root: Path) -> None:
    output = root / "publication"
    source_files = subprocess.check_output(
        ["git", "status", "--porcelain", "-z"], text=False
    ).split(b"\0")
    changed = sorted(
        {
            Path(record[3:].decode())
            for record in source_files
            if record and not record[3:].decode().startswith("artifacts/")
        }
    )
    source_hashes = {}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in changed:
            if not path.is_file() or path.suffix not in {".py", ".md"}:
                continue
            data = path.read_bytes()
            info = tarfile.TarInfo(str(path))
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            archive.addfile(info, io.BytesIO(data))
            source_hashes[str(path)] = digest(path)
    source_archive = output / "source_delta.tar.gz"
    source_archive.write_bytes(gzip.compress(buffer.getvalue(), mtime=0))
    (output / "SOURCE.json").write_text(
        json.dumps(
            {
                "base_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "scope": (
                    "changed/new source and review files over base commit; "
                    "collected before implementation commit"
                ),
                "archive_sha256": digest(source_archive),
                "files": source_hashes,
            },
            indent=2,
        )
        + "\n"
    )
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
        "included_files": sum(r["included"] for r in files),
        "included_bytes": sum(r["bytes"] for r in files if r["included"]),
        "local_only_files": sum(not r["included"] for r in files),
        "local_only_bytes": sum(r["bytes"] for r in files if not r["included"]),
    }
    manifest = output / "PUBLICATION_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "scope": "compact engineering case-study review; videos/bulk remain local",
                "files": files,
                **summary,
            },
            indent=2,
        )
        + "\n"
    )
    paths = [root / record["path"] for record in files if record["included"]] + [manifest]
    (output / "git-paths.nul").write_bytes(b"\0".join(str(path).encode() for path in paths) + b"\0")
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "inventory"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else inventory)(args.root)


if __name__ == "__main__":
    main()
