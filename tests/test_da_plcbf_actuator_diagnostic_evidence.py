"""Review transport must detect corrupted members independently of its outer hash."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from typing import TYPE_CHECKING

import pytest

from benchmark import da_plcbf_actuator_diagnostic_evidence as evidence

if TYPE_CHECKING:
    from pathlib import Path


def archive_fixture(tmp_path: Path, *, mutation: str | None = None) -> None:
    source = b"trial,outcome\nfixed,collision\n"
    inventory = {
        "schema": evidence.SCHEMA,
        "files": [
            {
                "path": "artifacts/review/table.csv",
                "sha256": hashlib.sha256(source).hexdigest(),
                "bytes": len(source),
                "included": True,
            },
            {
                "path": "artifacts/local-only/large-rollout.npz",
                "sha256": "0" * 64,
                "bytes": 1000,
                "included": False,
            },
        ],
    }
    inventory_bytes = evidence.json_bytes(inventory)
    generated = {
        "inventory.json": inventory_bytes,
        "README.md": b"Review transport only.\n",
        "SHA256SUMS": b"independently bound inventory entries\n",
    }
    for name, payload in generated.items():
        (tmp_path / name).write_bytes(payload)
    members = [("artifacts/review/table.csv", source)]
    members += [(f"__publication__/{name}", payload) for name, payload in generated.items()]
    if mutation == "changed_member":
        members[0] = (members[0][0], b"trial,outcome\nfixed,safe_task\n")
    elif mutation == "duplicate_member":
        members.append(members[0])
    elif mutation == "missing_member":
        members.pop(0)
    with tarfile.open(tmp_path / evidence.ARCHIVE, "w:gz") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    payload = (tmp_path / evidence.ARCHIVE).read_bytes()
    # Even a correctly recomputed outer checksum must not authenticate corrupted members.
    manifest = {
        "schema": evidence.SCHEMA,
        "archive": {
            "path": evidence.ARCHIVE,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        },
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "included_files": 1,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_transport_verifies_without_omitted_local_rollouts(tmp_path: Path) -> None:
    archive_fixture(tmp_path)
    result = evidence.verify(tmp_path, verify_local=False)
    assert result["archive_verified"]
    assert result["archive_members"] == 4
    assert not result["local_inventory_verified"]


@pytest.mark.parametrize("mutation", ["changed_member", "duplicate_member", "missing_member"])
def test_transport_rejects_invalid_members_despite_matching_outer_hash(
    tmp_path: Path, mutation: str
) -> None:
    archive_fixture(tmp_path, mutation=mutation)
    with pytest.raises(ValueError, match="member|membership"):
        evidence.verify(tmp_path, verify_local=False)
