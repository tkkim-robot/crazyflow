"""Prepare an explicitly fingerprinted retention-off continuation without training.

Run after active measured work completes:
    python -m benchmark.da_plcbf_actuator_ablation_prepare

Only the reference learning configuration's retention_weight changes from 5 to 0.
Every numeric array, complete Adam state, counter, model, teacher, anchor, physical
snapshot and actor configuration must round-trip exactly. The original checkpoint
and its artifacts remain untouched. This prepares inputs; it runs no flight study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "artifacts/da_plcbf/actuator-study-20260906/v1"


def _write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-checkpoint", type=Path, default=STUDY / "behavior-nominal128-seed11-v1/deployment"
    )
    parser.add_argument(
        "--output-stem", type=Path, default=STUDY / "retention-off-initial-seed11-v1/deployment"
    )
    args = parser.parse_args()
    # This setup only restores and serializes arrays. It must never initialize a GPU.
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import numpy as np

    from benchmark.da_plcbf_actuator_ablation_profile import _tree_hash
    from crazyflow.safety.da_plcbf.actuator_learning import (
        actuator_reference_fingerprint,
        load_actuator_learner_checkpoint,
        save_actuator_learner_checkpoint,
    )

    parent = load_actuator_learner_checkpoint(args.parent_checkpoint)
    if (
        parent.metadata.get("mode") != "nominal"
        or parent.metadata.get("seed") != 11
        or int(parent.state.library_version) != 128
        or int(parent.contract.spec.latent_codes.shape[0]) != 16
    ):
        parser.error("this scoped ablation requires the nominal128 seed11 K16 parent")
    original_learning = asdict(parent.contract.learning_config)
    if original_learning["retention_weight"] != 5.0:
        parser.error("the original retention weight must equal 5")
    contract = replace(
        parent.contract,
        learning_config=replace(parent.contract.learning_config, retention_weight=0.0),
    )
    changed_learning = asdict(contract.learning_config)
    differences = {
        key: {"before": original_learning[key], "after": changed_learning[key]}
        for key in original_learning
        if original_learning[key] != changed_learning[key]
    }
    if differences != {"retention_weight": {"before": 5.0, "after": 0.0}}:
        raise RuntimeError("the declared retention-off contract changed another objective field")
    parent_reference = actuator_reference_fingerprint(parent.contract)
    new_reference = actuator_reference_fingerprint(contract)
    if parent_reference == new_reference:
        raise RuntimeError("retention-off must have a distinct declared reference fingerprint")
    inherited_reports = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name in ("final_training.json", "final_validation.json")
        if (path := parent.json_path.parent / name).is_file()
    }
    provenance = {
        "ablation": "retention_off",
        "parent_checkpoint_stem": str(args.parent_checkpoint.resolve()),
        "parent_npz_sha256": _sha256(parent.npz_path),
        "parent_json_sha256": _sha256(parent.json_path),
        "parent_reference_sha256": parent_reference,
        "derived_reference_sha256": new_reference,
        "declared_difference": {
            "contract.learning_config.retention_weight": differences["retention_weight"]
        },
        "numeric_continuation_sha256": _tree_hash(parent.state),
        "numeric_teacher_and_anchors_sha256": _tree_hash(
            (
                parent.contract.params,
                parent.contract.model,
                parent.contract.anchors,
                parent.contract.spec,
            )
        ),
        "physical_state_sha256": _tree_hash(parent.physical_state),
        "inherited_actor_competence_reports": inherited_reports,
        "competence_scope": "unchanged initial actor only; retention-off adaptation is unevaluated",
        "training_updates_performed": 0,
        "script_sha256": _sha256(Path(__file__)),
        "learning_source_sha256": _sha256(ROOT / "crazyflow/safety/da_plcbf/actuator_learning.py"),
    }
    destination = args.output_stem.resolve()
    destination.parent.mkdir(parents=True, exist_ok=False)
    paths = save_actuator_learner_checkpoint(
        parent.state,
        contract,
        parent.physical_state,
        destination,
        config=parent.config,
        metadata={**parent.metadata, "ablation_provenance": provenance},
    )
    derived = load_actuator_learner_checkpoint(destination)
    if _tree_hash(derived.state) != provenance["numeric_continuation_sha256"]:
        raise RuntimeError("derived checkpoint changed numeric learner/Adam continuation")
    if (
        _tree_hash(
            (
                derived.contract.params,
                derived.contract.model,
                derived.contract.anchors,
                derived.contract.spec,
            )
        )
        != provenance["numeric_teacher_and_anchors_sha256"]
    ):
        raise RuntimeError("derived checkpoint changed the nominal teacher/model/anchors/spec")
    if (
        derived.physical_state.dtype != parent.physical_state.dtype
        or not np.array_equal(derived.physical_state, parent.physical_state)
        or asdict(derived.config) != asdict(parent.config)
        or asdict(derived.contract.actor_config) != asdict(parent.contract.actor_config)
        or actuator_reference_fingerprint(derived.contract) != new_reference
    ):
        raise RuntimeError(
            "derived checkpoint did not preserve its declared physical/actor contract"
        )
    report = {
        **provenance,
        "status": "prepared_and_exact_roundtrip_verified",
        "derived_checkpoint_stem": str(destination),
        "derived_npz_sha256": _sha256(paths[0]),
        "derived_json_sha256": _sha256(paths[1]),
        "npz_bytes_equal_parent": paths[0].read_bytes() == parent.npz_path.read_bytes(),
        "npz_equality_note": (
            "array contents are intentionally identical; changed objective is bound "
            "by JSON and reference fingerprint"
        ),
        "numeric_array_equality_verified": True,
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "comparison_plan": (
            "A-only validation2003/2103 combined; descriptive comparison to secondary-oracle-v1 A"
        ),
    }
    _write_json(destination.parent / "preparation.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
