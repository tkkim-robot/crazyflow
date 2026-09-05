"""Verify contact artifact bindings and derive contact timings from saved arrays only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name in (
        "legacy-estimated-physical-contact-v2",
        "static-impact-drop-fixture",
        "seed209-adaptive-safety-abort-drop",
    ):
        directory = args.root / name
        metadata = json.loads((directory / "contact_replay.json").read_text())
        for filename, key in (
            ("contact_replay.npz", "npz_sha256"),
            ("contact_model.xml", "xml_sha256"),
            ("CONTACT_REPLAY_SOURCE.py", "source_sha256"),
            ("CONTACT_CLI_SOURCE.py", "cli_sha256"),
        ):
            assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == metadata[key]
        source = metadata["source"]
        for filename, digest in source.get("input_sha256", {}).items():
            path = Path(filename)
            if not path.is_absolute():
                path = Path(source["directory"]) / path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        first_contact = {}
        with np.load(directory / "contact_replay.npz", allow_pickle=False) as raw:
            assert np.isfinite(raw["full_state"]).all()
            for kind in ("obstacle", "ground"):
                mask = raw[f"{kind}_contact"]
                if mask.ndim == 2:
                    mask = np.any(mask, axis=1)
                indices = np.flatnonzero(mask)
                assert len(indices) == metadata[f"{kind}_contact_steps"]
                time = float(raw["time_seconds"][indices[0]]) if len(indices) else None
                event_time = next(
                    (
                        event["time_seconds"]
                        for event in metadata["contact_events"]
                        if event["kind"] == f"{kind}_contact"
                    ),
                    None,
                )
                assert time == event_time
                first_contact[f"first_{kind}_contact_seconds"] = time
        rows.append(
            {
                "artifact": name,
                "trigger_seconds": metadata["trigger"]["time_seconds"],
                "trigger_kind": metadata["trigger"]["kind"],
                **first_contact,
                "minimum_contact_distance_m": metadata["minimum_contact_distance_m"],
                "maximum_contact_force_norm_N": metadata["maximum_contact_force_norm_N"],
                "all_bound_hashes_verified": True,
                "warning_counts": metadata["warning_counts"],
            }
        )
    result = {
        "scope": "Saved-array and hash audit; no simulation or controller rerun.",
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": rows,
    }
    output = args.root / "CONTACT_ARTIFACT_AUDIT.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
