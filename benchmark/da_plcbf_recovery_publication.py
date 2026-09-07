"""Package recovery evidence and authenticate all archived bytes without rerunning physics."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from benchmark.da_plcbf_stabilization_publication import sha, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/da_plcbf/recovery-interaction-20260907/v1"


def main() -> None:
    output = BASE / "publication-v1"
    output.mkdir(exist_ok=False)
    for src in Path("/tmp").glob("recovery-*.log"):
        if "publication" not in src.name:
            shutil.copy2(src, BASE / "logs" / src.name)
    stages = {}
    for name in ("guard-fixed-v1", "guard-fixed-v2", "causal-guard-v2", "wind-demo-v1"):
        binding = json.loads((BASE / name / "execution.json").read_text())
        for source, digest in binding["source_sha256"].items():
            assert sha(BASE / name / "source" / source) == digest, (name, source)
        stages[name] = {"retained_sources_verified": len(binding["source_sha256"])}
    wind_sources = json.loads((BASE / "wind-demo-v1/execution.json").read_text())["source_sha256"]
    for source, digest in wind_sources.items():
        assert sha(ROOT / source) == digest, source
    rows = json.loads((BASE / "analysis-v2/report.json").read_text())["rows"]
    assert len(rows) == 27
    assert all(r["held_timing_covered"] for r in rows)
    for row in rows:
        directory = ROOT / row["directory"]
        assert sha(directory / "controls.npz") == row["controls_sha256"]
        assert sha(directory / "binding.json") == row["binding_sha256"]
    for pair in ("libraries", "adaptation"):
        directory = BASE / "wind-video-v1" / pair
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["full_decode_passed"]
        assert sha(Path(manifest["video"])) == manifest["sha256"]
        assert json.loads((directory / "render_summary.json").read_text())["status"] == "completed"
    write(
        BASE / "execution-verification.json",
        {
            "stages": stages,
            "current_source_matches_wind_execution": True,
            "completed_governor_and_wind_flights": 27,
            "wind_vectors_authenticated": True,
            "scope": "Each version's sources are authenticated against its own execution.",
        },
    )
    inventory, selected = [], {}
    for path in sorted(BASE.rglob("*")):
        if not path.is_file() or output in path.parents:
            continue
        name = str(path.relative_to(ROOT))
        included = path.suffix in {".json", ".csv", ".md", ".py", ".log", ".txt"}
        if "instrumented-harm-v1" in path.parts and "snapshots" in path.parts:
            included = True
        inventory.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": sha(path), "included": included}
        )
        if included:
            selected[name] = path
    extra = set(wind_sources)
    extra.update(str(p.relative_to(ROOT)) for p in ROOT.glob("benchmark/da_plcbf_recovery_*.py"))
    extra.update(str(p.relative_to(ROOT)) for p in ROOT.glob("tests/test_da_plcbf_*.py"))
    extra.update(str(p.relative_to(ROOT)) for p in ROOT.glob("tests/unit/safety/da_plcbf/*.py"))
    extra.update(
        {
            "docs/da_plcbf_recovery_interaction_report.md",
            "benchmark/da_plcbf_actuator_video.py",
            "benchmark/da_plcbf_stabilization_publication.py",
        }
    )
    selected.update({name: ROOT / name for name in extra})
    write(output / "inventory.json", inventory)
    members = {name: {"sha256": sha(p), "bytes": p.stat().st_size} for name, p in selected.items()}
    archive = output / "review-only.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, path in sorted(selected.items()):
            tar.add(path, arcname=name, recursive=False)
    with tarfile.open(archive, "r:gz") as tar:
        assert set(tar.getnames()) == set(members)
        for member in tar:
            assert member.isfile()
            stream = tar.extractfile(member)
            assert stream is not None
            assert (
                hashlib.file_digest(stream, "sha256").hexdigest() == members[member.name]["sha256"]
            )
    write(
        output / "manifest.json",
        {
            "archive_sha256": sha(archive),
            "inventory_sha256": sha(output / "inventory.json"),
            "members": members,
            "scope": "Review-only; large excluded files are hash-indexed.",
        },
    )
    write(output / "verification.json", {"all_members_verified": True, "members": len(members)})
    (output / "README.md").write_text(
        "# Recovery interaction evidence\n\n"
        "The package preserves the common-state diagnosis, controlled continuations, "
        "80-digit QP reference, both 12-flight governor versions, the exact-state repair "
        "continuation, three wind flights, failed harnesses and their qualifications, "
        "executed sources, tests, logs and video metadata. This is development evidence.\n\n"
        "Archive member hashes were verified. The instrumented harmful flight's complete "
        "saved snapshots are included. Large rollout/dense/application arrays, JSONL streams "
        "and movies remain local and are hash-indexed in inventory.json. This is a review "
        "package, not a complete replay archive. Extract at the repository root. Absolute "
        "paths record the original execution host; prior checkpoint provenance is in the "
        "previously published packages.\n"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha(p)}  {p.name}\n" for p in sorted(output.iterdir()) if p.name != "SHA256SUMS")
    )
    print(json.dumps({"members": len(members), "archive_bytes": archive.stat().st_size}))


if __name__ == "__main__":
    main()
