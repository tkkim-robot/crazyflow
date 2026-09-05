"""Capture source and verify immutable artifact inventories for a local research revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(directory: Path, excluded: Path) -> dict[str, dict[str, int | str]]:
    return {
        str(path.relative_to(directory)): {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != excluded
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--source-output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    roots = [path.resolve() for path in args.artifact_roots]
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"missing artifact directory: {root}")
    if args.verify:
        results = []
        for root in roots:
            manifest = root / "ARTIFACT_SHA256.json"
            recorded = json.loads(manifest.read_text())["files"]
            current = _inventory(root, manifest)
            if recorded != current:
                missing = sorted(recorded.keys() - current.keys())
                added = sorted(current.keys() - recorded.keys())
                changed = sorted(
                    name
                    for name in recorded.keys() & current.keys()
                    if recorded[name] != current[name]
                )
                raise ValueError(
                    f"artifact mismatch in {root}: missing={missing}, added={added}, "
                    f"changed={changed}"
                )
            results.append({"directory": str(root), "verified_files": len(current)})
        print(json.dumps({"verified": True, "roots": results}, indent=2))
        return
    if args.source_output is None:
        raise ValueError("capture requires --source-output inside an artifact root")
    output = args.source_output.resolve()
    if not any(output.is_relative_to(root) for root in roots):
        raise ValueError("source output must belong to an inventoried artifact root")
    if output.exists() or any((root / "ARTIFACT_SHA256.json").exists() for root in roots):
        raise FileExistsError("refusing to overwrite prior source or artifact evidence")
    patterns = (
        "crazyflow/safety/da_plcbf/**/*.py",
        "benchmark/da_plcbf*.py",
        "examples/da_plcbf/**/*.py",
        "tests/unit/safety/da_plcbf/**/*.py",
        "docs/da_plcbf*.md",
        "DA_PLCBF*.md",
        "HANDOFF_DA_PLCBF.md",
    )
    sources = sorted(
        {path for pattern in patterns for path in repo.glob(pattern) if path.is_file()}
    )
    output.mkdir(parents=True)
    archive = output / "SOURCE.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in sources:
            stream.add(path, arcname=str(path.relative_to(repo)), recursive=False)
    source_manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo, text=True
        ).strip(),
        "scope": (
            "working source, not a new commit; extract over base commit. Historical numerical "
            "campaigns retain their own frozen source and explicitly declared reporting patches."
        ),
        "archive_sha256": _sha(archive),
        "files": {str(path.relative_to(repo)): _sha(path) for path in sources},
    }
    (output / "SOURCE_SHA256.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    counts = []
    for root in roots:
        manifest = root / "ARTIFACT_SHA256.json"
        inventory = _inventory(root, manifest)
        record = {
            "scope": "all existing files, including retained failures; only this manifest excluded",
            "file_count": len(inventory),
            "total_bytes": sum(row["bytes"] for row in inventory.values()),
            "files": inventory,
        }
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        counts.append({"directory": str(root), "files": len(inventory)})
    print(json.dumps({"source_files": len(sources), "artifacts": counts}, indent=2))


if __name__ == "__main__":
    main()
