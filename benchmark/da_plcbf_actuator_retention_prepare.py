"""Prepare six untrained retention-10 contract variants without changing source checkpoints.

This is preparation only: it performs no optimizer update, actor rollout, controller
call or plant simulation. The nominal128 and DR512 numeric NPZ payloads are copied
byte for byte. JSON changes only the active retention weight, its contract digest,
and explicit derivation metadata. Original DR training remains retention5.

    python -m benchmark.da_plcbf_actuator_retention_prepare \
        --study-root /abs/actuator-study/v1 --output /abs/fresh-retention10-candidate

The output is a candidate input bundle, not evidence of validation or a decision to
use retention10. F2/A/OPT share one exact nominal checkpoint per seed; DR preserves
its original frozen actor and optimizer. Corresponding active teacher contracts
must have identical fingerprints across the nominal and DR variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _numeric_identity(first: Any, second: Any) -> bool:
    import jax
    import numpy as np

    left, left_structure = jax.tree.flatten(first)
    right, right_structure = jax.tree.flatten(second)
    return left_structure == right_structure and all(
        np.asarray(a).dtype == np.asarray(b).dtype
        and np.asarray(a).shape == np.asarray(b).shape
        and np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left, right, strict=True)
    )


def _prepare_one(parent_stem: Path, destination: Path, seed: int, mode: str) -> dict:
    from crazyflow.safety.da_plcbf.actuator_learning import (
        actuator_reference_fingerprint,
        load_actuator_learner_checkpoint,
    )

    parent = load_actuator_learner_checkpoint(parent_stem)
    old_json = json.loads(parent.json_path.read_text())
    expected_version = 128 if mode == "nominal" else 512
    if (
        parent.metadata["mode"] != mode
        or parent.metadata["seed"] != seed
        or int(parent.state.library_version) != expected_version
        or parent.contract.spec.latent_codes.shape != (16, 8)
        or parent.contract.learning_config.retention_weight != 5
    ):
        raise ValueError(f"Source lacks the specified {mode} seed{seed} continuation")
    replacement = replace(parent.contract.learning_config, retention_weight=10.0)
    contract = replace(parent.contract, learning_config=replacement)
    reference_before = actuator_reference_fingerprint(parent.contract)
    reference_after = actuator_reference_fingerprint(contract)
    if reference_before == reference_after or old_json["reference_sha256"] != reference_before:
        raise ValueError("Original or derived reference fingerprint is inconsistent")
    learning_before, learning_after = asdict(parent.contract.learning_config), asdict(replacement)
    differences = {
        key: {"before": learning_before[key], "after": learning_after[key]}
        for key in learning_before
        if learning_before[key] != learning_after[key]
    }
    if differences != {"retention_weight": {"before": 5.0, "after": 10.0}}:
        raise ValueError("Candidate changes another objective field")
    provenance = {
        "kind": "candidate_active_retention_contract_variant",
        "selected_for_deployment": False,
        "parent_checkpoint_stem": str(parent_stem.resolve()),
        "parent_npz_sha256": _sha256(parent.npz_path),
        "parent_json_sha256": _sha256(parent.json_path),
        "parent_reference_sha256": reference_before,
        "active_reference_sha256": reference_after,
        "source_mode": mode,
        "source_library_seed": seed,
        "source_library_version": expected_version,
        "source_cumulative_gradient_steps": old_json["cumulative_gradient_steps"],
        "offline_training_retention_weight": 5.0,
        "active_runtime_retention_weight": 10.0,
        "training_updates_performed_by_derivation": 0,
        "numeric_teacher_model_anchors_and_spec": "unchanged source NPZ bytes",
        "numeric_actor_adam_counters_and_latest_model": "unchanged source NPZ bytes",
        "declared_difference": {
            "reference_learning_config.retention_weight": differences["retention_weight"]
        },
        "scope": (
            "Runtime contract candidate only. DR actor training remains the original "
            "retention5, 512-update, 64-model training protocol; this derivative is not "
            "a retention10-trained DR actor. No new competence or flight claim."
            if mode == "dr"
            else (
                "Runtime contract candidate only; unchanged nominal128 actor/Adam. "
                "No new validation claim."
            )
        ),
        "preparation_source_sha256": _sha256(Path(__file__)),
    }
    destination.parent.mkdir(parents=True, exist_ok=False)
    payload = json.loads(json.dumps(old_json))
    payload["reference_learning_config"]["retention_weight"] = 10.0
    payload["reference_sha256"] = reference_after
    payload["metadata"]["retention_contract_derivation"] = provenance
    shutil.copyfile(parent.npz_path, destination.with_suffix(".npz"))
    _write_json(destination.with_suffix(".json"), payload)
    restored = load_actuator_learner_checkpoint(destination)
    if (
        destination.with_suffix(".npz").read_bytes() != parent.npz_path.read_bytes()
        or not _numeric_identity(restored.state, parent.state)
        or not _numeric_identity(
            (
                restored.contract.params,
                restored.contract.model,
                restored.contract.anchors,
                restored.contract.spec,
            ),
            (
                parent.contract.params,
                parent.contract.model,
                parent.contract.anchors,
                parent.contract.spec,
            ),
        )
        or not _numeric_identity(restored.physical_state, parent.physical_state)
        or asdict(restored.config) != asdict(parent.config)
        or asdict(restored.contract.actor_config) != asdict(parent.contract.actor_config)
        or actuator_reference_fingerprint(restored.contract) != reference_after
    ):
        raise RuntimeError("Candidate failed exact checkpoint continuation verification")
    # Verify all parent files still match the digests captured before the write.
    if (
        _sha256(parent.npz_path) != provenance["parent_npz_sha256"]
        or _sha256(parent.json_path) != provenance["parent_json_sha256"]
    ):
        raise RuntimeError("A source checkpoint changed during candidate preparation")
    report = {
        **provenance,
        "status": "candidate_prepared_exact_numeric_continuation_verified",
        "checkpoint_stem": str(destination.resolve()),
        "checkpoint_npz_sha256": _sha256(destination.with_suffix(".npz")),
        "checkpoint_json_sha256": _sha256(destination.with_suffix(".json")),
        "numeric_npz_bytes_identical_to_parent": True,
    }
    _write_json(destination.parent / "preparation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Package imports may initialize JAX through Crazyflow registration. Preparation
    # is CPU-only and deliberately does not construct a learner or execute a step.
    os.environ["JAX_PLATFORMS"] = "cpu"
    args.study_root, args.output = args.study_root.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, checkpoint_map = [], {method: {} for method in ("F2", "A", "DR", "OPT")}
    try:
        for seed in (11, 23, 37):
            pair = []
            for mode, version in (("nominal", 128), ("dr", 512)):
                parent = args.study_root / f"behavior-{mode}{version}-seed{seed}-v1" / "deployment"
                destination = args.output / f"{mode}{version}-seed{seed}" / "deployment"
                report = _prepare_one(parent, destination, seed, mode)
                rows.append(report)
                pair.append(report)
                for method in ("DR",) if mode == "dr" else ("F2", "A", "OPT"):
                    checkpoint_map[method][str(seed)] = str(destination)
            if (
                pair[0]["parent_reference_sha256"] != pair[1]["parent_reference_sha256"]
                or pair[0]["active_reference_sha256"] != pair[1]["active_reference_sha256"]
            ):
                raise RuntimeError(f"Nominal/DR seed{seed} teachers or active contracts differ")
        _write_json(args.output / "checkpoint-map.json", checkpoint_map)
        _write_json(
            args.output / "manifest.json",
            {
                "schema": "da_plcbf_retention10_contract_candidates_v1",
                "status": "prepared_unselected_candidates",
                "selected_for_deployment": False,
                "training_updates_performed": 0,
                "existing_checkpoints_modified": False,
                "same_active_teacher_contract_across_nominal_and_dr_per_seed": True,
                "candidate_checkpoints": rows,
                "checkpoint_map": checkpoint_map,
                "fairness_scope": (
                    "F2/A/OPT use identical nominal128 initial checkpoint bytes per seed; "
                    "DR preserves its original DR512 actor/Adam and retention5 offline training. "
                    "All active contracts use retention10. Changing inactive F2/DR/OPT objective "
                    "metadata does not add actor training or alter frozen behavior. The choice "
                    "requires a separately recorded development/validation selection decision."
                ),
            },
        )
    except Exception as error:
        _write_json(
            args.output / "FAILED.json",
            {
                "type": type(error).__name__,
                "reason": str(error),
                "retained_completed_candidates": rows,
            },
        )
        raise
    print(
        json.dumps(
            {
                "status": "prepared_unselected_candidates",
                "output": str(args.output),
                "checkpoint_count": len(rows),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
