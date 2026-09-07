"""Package the closed numerical iteration without importing or executing the controller."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/da_plcbf/numerical-stabilization-20260906/v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    output = BASE / "publication-v1"
    output.mkdir(exist_ok=False)
    protocol = json.loads((BASE / "sealed-v4/protocol.json").read_text())
    parent = json.loads((BASE / "sealed-v3/protocol.json").read_text())
    driver = "benchmark/da_plcbf_numerical_stabilization.py"
    changed = [k for k, v in parent["source_sha256"].items() if v != protocol["source_sha256"][k]]
    assert changed == [driver]
    assert parent["trials"] == protocol["trials"]
    assert parent["file_sha256"] == protocol["file_sha256"]
    for name, expected in protocol["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    campaign = BASE / "development-36-v3"
    records = json.loads((campaign / "results.json").read_text())["records"]
    assert len(records) == len(list(campaign.glob("*/claim.json"))) == 36
    assert len(list(campaign.glob("*/attempt-*/binding.json"))) == 36
    assert len(list(campaign.glob("*/record.json"))) == 36
    assert sum(r.get("completed_attempt_reused_without_flight", False) for r in records) == 1
    assert {r["trial"]["id"] for r in records} == {t["id"] for t in protocol["trials"]}
    write(
        BASE / "execution-v4-verification.json",
        {
            "protocol_sha256": sha(BASE / "sealed-v4/protocol.json"),
            "parent_protocol_sha256": sha(BASE / "sealed-v3/protocol.json"),
            "changed_source_files": changed,
            "completed_records": 36,
            "physical_attempt_bindings": 36,
            "claim_files": 36,
            "completed_attempts_reused_without_new_flight": 1,
            "remaining_attempts_executed_after_io_amendment": 35,
        },
    )
    inventory = []
    selected = {}
    for path in sorted(BASE.rglob("*")):
        if not path.is_file() or output in path.parents:
            continue
        rel = str(path.relative_to(ROOT))
        included = path.suffix in {".json", ".csv", ".md", ".py", ".log", ".txt"}
        inventory.append(
            {"path": rel, "bytes": path.stat().st_size, "sha256": sha(path), "included": included}
        )
        if included:
            selected[rel] = path
    extra = set(protocol["source_sha256"])
    extra.update(str(p.relative_to(ROOT)) for p in ROOT.glob("tests/test_da_plcbf_*.py"))
    extra.update(str(p.relative_to(ROOT)) for p in ROOT.glob("tests/unit/safety/da_plcbf/*.py"))
    extra.update(
        {
            "docs/da_plcbf_numerical_stabilization_report.md",
            "benchmark/da_plcbf_numerical_stabilization_analysis.py",
            "benchmark/da_plcbf_stabilized_video.py",
            "benchmark/da_plcbf_actuator_video.py",
            str(Path(__file__).relative_to(ROOT)),
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
            "scope": "Review-only; excluded large raw files are hash-indexed.",
        },
    )
    write(output / "verification.json", {"all_members_verified": True, "members": len(members)})
    (output / "README.md").write_text(
        "# Numerical stabilization evidence\n\n"
        "This is the closed 36-attempt development iteration, not fresh validation. "
        "The archive retains source, protocols, failed preliminary probes, final probes, "
        "36 flight summaries/bindings, 12 causal-pair analyses, historical timing audit, "
        "regression logs, and video manifests. Archive member hashes were verified.\n\n"
        "`inventory.json` hashes every local iteration artifact. Large NPZ/checkpoint arrays, "
        "event/frame JSONL streams, PNG previews and MP4 videos are excluded from transport; "
        "this is not a complete replay archive. Extract at the repository root to restore "
        "the review metadata. Recorded absolute paths describe the original execution host. "
        "Earlier checkpoint provenance is retained in the prior published evidence.\n"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha(p)}  {p.name}\n" for p in sorted(output.iterdir()) if p.name != "SHA256SUMS")
    )
    print(json.dumps({"members": len(members), "archive_bytes": archive.stat().st_size}))


if __name__ == "__main__":
    main()
