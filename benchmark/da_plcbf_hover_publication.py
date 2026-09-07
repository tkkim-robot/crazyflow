"""Archive compact, hash-verified evidence without discarding negative hover experiments."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np

from benchmark.da_plcbf_hover_inventory import PHASES
from benchmark.da_plcbf_hover_return import BASE, ROOT
from benchmark.da_plcbf_recovery_diagnosis import sha, write

DETAILED = {
    "selected-replay-v2",
    "selected-freeze2-v1",
    "selected-freeze16-v1",
    "selected-delayed-v1",
    "selected-compensated-v1",
    "harm-regression-v1",
}


def main() -> None:
    assert json.loads((BASE / "inventory-v1/report.json").read_text())["complete"]
    assert json.loads((BASE / "video-v2/validation.json").read_text())["full_decode_passed"]
    assert (BASE / "selected-delayed-v1/timing-audit.json").exists()
    logs = BASE / "logs"
    logs.mkdir(exist_ok=True)
    for p in Path("/tmp").glob("hover-return-*.log"):
        shutil.copy2(p, logs / p.name)
    compact = BASE / "compact-review"
    compact.mkdir(exist_ok=False)
    for path in sorted(BASE.rglob("summary.json")):
        parts = path.relative_to(BASE).parts
        if parts[0] not in PHASES:
            continue
        summary = json.loads(path.read_text())
        omitted = ("snapshot_captures", "snapshot_publications", "applied_hold_timing_audit")
        derived = {k: v for k, v in summary.items() if k not in omitted}
        derived["raw_summary_sha256"] = sha(path)
        derived["omitted_verbose_fields"] = list(omitted)
        destination = compact / "outcomes" / path.relative_to(BASE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write(destination, derived)
    for stage in DETAILED:
        for path in sorted((BASE / stage).glob("*/attempt-00/controls.npz")):
            with np.load(path) as controls:
                data = {
                    k: controls[k]
                    for k in controls.files
                    if k not in ("candidate_states", "candidate_commands")
                }
                requested = np.array(
                    [
                        1.96,
                        2.0,
                        3.0,
                        4.0,
                        5.2,
                        5.4,
                        5.6,
                        5.8,
                        6.0,
                        8.0,
                        9.0,
                        10.0,
                        16.0,
                        18.0,
                        20.0,
                        22.0,
                        24.0,
                        25.2,
                        31.96,
                    ]
                )
                requested = requested[requested <= controls["time"][-1] + 1e-10]
                indices = np.unique([int(np.argmin(abs(controls["time"] - t))) for t in requested])
                data["candidate_sample_indices"] = indices
                for key in ("candidate_states", "candidate_commands"):
                    if key in controls and controls[key].size:
                        data[key + "_sampled"] = controls[key][indices]
                data["source_controls_sha256"] = np.asarray(sha(path))
                np.savez_compressed(
                    compact / f"{stage}-{path.parents[1].name}-controls.npz", **data
                )
    output = BASE / "publication-v1"
    output.mkdir(exist_ok=False)
    inventory, members = [], {}
    for path in sorted(BASE.rglob("*")):
        if not path.is_file() or output in path.parents:
            continue
        parts = path.relative_to(BASE).parts
        if parts[0].startswith("publication-") or parts[0].startswith("compact-review-incomplete"):
            continue
        detailed = parts[0] in DETAILED
        included = (
            path.suffix in (".json", ".csv", ".py", ".log", ".md", ".txt")
            and "snapshots" not in parts
        )
        if parts[0] in PHASES and path.name in ("report.json", "summary.json"):
            included = False
        if not detailed and path.name == "updates.json":
            included = False
        included |= detailed and path.suffix == ".npz" and path.name != "controls.npz"
        included |= detailed and "snapshots" in parts and path.suffix == ".json"
        included |= parts[0] == "compact-review"
        included |= parts[0] == "selected-mechanism-v1" and path.name == "rollouts.npz"
        included |= parts[0] == "video-v2" and "contact" in parts and path.suffix == ".npz"
        included |= parts[0] == "figures-v1" and path.suffix in (".png", ".pdf")
        name = str(path.relative_to(ROOT))
        digest = sha(path)
        inventory.append(dict(path=name, sha256=digest, included=bool(included)))
        if included:
            members[name] = (path, digest)
    for path in [
        *ROOT.glob("benchmark/da_plcbf_hover_*.py"),
        ROOT / "crazyflow/safety/da_plcbf/actuator_hover.py",
        ROOT / "crazyflow/safety/da_plcbf/actuator_backup.py",
        ROOT / "crazyflow/safety/da_plcbf/actuator_experiment.py",
        ROOT / "tests/test_da_plcbf_actuator_hover.py",
        ROOT / "tests/test_da_plcbf_actuator_backup.py",
        ROOT / "docs/da_plcbf_hover_return_report.md",
    ]:
        members[str(path.relative_to(ROOT))] = (path, sha(path))
    archive = output / "review-evidence.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        for name, (path, _) in members.items():
            tar.add(path, arcname=name, recursive=False)
    with tarfile.open(archive, "r:gz") as tar:
        assert set(tar.getnames()) == set(members)
        for member in tar:
            stream = tar.extractfile(member)
            assert stream is not None
            assert hashlib.file_digest(stream, "sha256").hexdigest() == members[member.name][1]
    write(output / "inventory.json", inventory)
    write(
        output / "manifest.json",
        dict(
            archive_sha256=sha(archive),
            archive_bytes=archive.stat().st_size,
            all_members_verified=True,
            members={k: v[1] for k, v in members.items()},
            scope="All development plans and complete outcome metrics, including negatives, "
            "are included. "
            "Development summary copies omit verbose snapshot and timing histories; raw originals "
            "remain hash-indexed. "
            "Detailed selected replay, freezes, timing, compensated baseline and harmful "
            "regression include dense states, learner/Adam snapshots, decisions and sampled "
            "prediction fans. Exact contact suffixes and video audits included. Full prediction "
            "arrays, verbose events, movies and other raw rollouts remain local and hash-indexed.",
        ),
    )
    print(
        json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, members=len(members))),
        flush=True,
    )


if __name__ == "__main__":
    main()
