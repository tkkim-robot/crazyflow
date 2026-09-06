"""Clone the nominal128 seed11 continuation with retention weight 5 -> 10 only.

This bounded preparation performs no training, controller call, or physical
episode. Full params, previous params, Adam, counters, model, teacher, anchors,
skill specification, actor configuration and physical snapshot remain exact.
The nominal training provenance is inherited from the authenticated parent;
the changed adaptation objective receives a new reference fingerprint and tag.
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


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _save(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-checkpoint", type=Path, default=STUDY / "behavior-nominal128-seed11-v1/deployment"
    )
    parser.add_argument(
        "--output-stem", type=Path, default=STUDY / "retention10-initial-seed11-v1/deployment"
    )
    args = parser.parse_args(argv)
    # CPU-only deserialize/serialize setup; no GPU backend may be initialized.
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import numpy as np

    from crazyflow.safety.da_plcbf.actuator_experiment import _checkpoint_provenance, _hash_tree
    from crazyflow.safety.da_plcbf.actuator_learning import (
        actuator_reference_fingerprint,
        load_actuator_learner_checkpoint,
        save_actuator_learner_checkpoint,
    )

    jax.config.update("jax_platforms", "cpu")
    parent = load_actuator_learner_checkpoint(args.parent_checkpoint)
    if (
        parent.metadata.get("mode") != "nominal"
        or parent.metadata.get("seed") != 11
        or int(parent.state.library_version) != 128
        or int(parent.state.cumulative_gradient_steps) != 128
        or int(parent.contract.spec.latent_codes.shape[0]) != 16
    ):
        parser.error("this diagnostic requires the genuine nominal128 seed11 K16 parent")
    before = asdict(parent.contract.learning_config)
    if before["retention_weight"] != 5.0:
        parser.error("the authenticated parent must have retention_weight=5")
    contract = replace(
        parent.contract,
        learning_config=replace(parent.contract.learning_config, retention_weight=10.0),
    )
    after = asdict(contract.learning_config)
    differences = {
        key: {"before": before[key], "after": after[key]}
        for key in before
        if before[key] != after[key]
    }
    if differences != {"retention_weight": {"before": 5.0, "after": 10.0}}:
        raise AssertionError("another adaptation-objective field changed")
    original_reference = actuator_reference_fingerprint(parent.contract)
    derived_reference = actuator_reference_fingerprint(contract)
    if original_reference == derived_reference:
        raise AssertionError("the declared objective must receive a distinct reference fingerprint")
    numeric_teacher = (
        parent.contract.params,
        parent.contract.model,
        parent.contract.anchors,
        parent.contract.spec,
    )
    provenance = {
        "ablation": "retention_weight_10",
        "preparation_kind": "exact nominal continuation clone; no training",
        "parent_checkpoint_npz": str(parent.npz_path.resolve()),
        "parent_checkpoint_json": str(parent.json_path.resolve()),
        "parent_npz_sha256": _sha(parent.npz_path),
        "parent_json_sha256": _sha(parent.json_path),
        "parent_reference_sha256": original_reference,
        "derived_reference_sha256": derived_reference,
        "declared_difference": {
            "contract.learning_config.retention_weight": differences["retention_weight"]
        },
        "inherited_metadata_mode": parent.metadata["mode"],
        "inherited_metadata_seed": parent.metadata["seed"],
        "full_learner_continuation_sha256": _hash_tree(parent.state),
        "parameters_sha256": _hash_tree(parent.state.params),
        "previous_parameters_sha256": _hash_tree(parent.state.previous_params),
        "adam_state_sha256": _hash_tree(parent.state.optimizer_state),
        "numeric_teacher_model_anchors_spec_sha256": _hash_tree(numeric_teacher),
        "physical_state_sha256": _hash_tree(parent.physical_state),
        "physical_state_dtype": str(parent.physical_state.dtype),
        "training_updates_performed": 0,
        "source_sha256": {
            str(path): _sha(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "crazyflow/safety/da_plcbf/actuator_learning.py",
                ROOT / "crazyflow/safety/da_plcbf/actuator_experiment.py",
            )
        },
        "inherited_initial_actor_competence_reports": {
            name: {"path": str(path.resolve()), "sha256": _sha(path)}
            for name in ("final_training.json", "final_validation.json")
            if (path := parent.json_path.parent / name).is_file()
        },
        "competence_scope": (
            "unchanged initial actor only; retention10 online adaptation is unevaluated"
        ),
    }
    destination = args.output_stem.resolve()
    destination.parent.mkdir(parents=True, exist_ok=False)
    for path_text in provenance["source_sha256"]:
        path = Path(path_text)
        copied = destination.parent / "source" / path.name
        copied.parent.mkdir(exist_ok=True)
        with copied.open("xb") as stream:
            stream.write(path.read_bytes())
    metadata = {
        **parent.metadata,
        "ablation_tag": "retention_weight_10",
        "ablation_provenance": provenance,
    }
    paths = save_actuator_learner_checkpoint(
        parent.state,
        contract,
        parent.physical_state,
        destination,
        config=parent.config,
        metadata=metadata,
    )
    derived = load_actuator_learner_checkpoint(destination)
    checks = {
        "full_learner_continuation": _hash_tree(derived.state) == _hash_tree(parent.state),
        "teacher_model_anchors_spec": _hash_tree(
            (
                derived.contract.params,
                derived.contract.model,
                derived.contract.anchors,
                derived.contract.spec,
            )
        )
        == _hash_tree(numeric_teacher),
        "physical_state": (
            derived.physical_state.dtype == parent.physical_state.dtype
            and derived.physical_state.tobytes() == parent.physical_state.tobytes()
        ),
        "actor_config": asdict(derived.config) == asdict(parent.config),
        "reference_actor_config": asdict(derived.contract.actor_config)
        == asdict(parent.contract.actor_config),
        "reference_learning_config": asdict(derived.contract.learning_config) == after,
        "derived_reference_fingerprint": actuator_reference_fingerprint(derived.contract)
        == derived_reference,
        "nominal_training_metadata_inherited": all(
            derived.metadata[key] == value for key, value in parent.metadata.items()
        ),
    }
    with np.load(parent.npz_path, allow_pickle=False) as original_arrays:
        with np.load(derived.npz_path, allow_pickle=False) as derived_arrays:
            array_checks = {
                key: (
                    original_arrays[key].dtype == derived_arrays[key].dtype
                    and original_arrays[key].shape == derived_arrays[key].shape
                    and original_arrays[key].tobytes() == derived_arrays[key].tobytes()
                )
                for key in original_arrays.files
                if key in derived_arrays.files
            }
            checks["all_numeric_array_keys"] = set(original_arrays.files) == set(
                derived_arrays.files
            )
            checks["all_numeric_array_bytes"] = all(array_checks.values())
    if not all(checks.values()):
        _save(
            destination.parent / "error.json", {"status": "verification_failed", "checks": checks}
        )
        raise AssertionError(
            "derived checkpoint changed more than the declared retention objective"
        )
    # Exercise the existing provenance guard; do not change or bypass it.
    accepted = _checkpoint_provenance(derived, "A")
    if accepted["metadata"]["mode"] != "nominal":
        raise AssertionError("runtime nominal provenance was not preserved")
    report = {
        **provenance,
        "status": "prepared_and_exact_roundtrip_verified",
        "derived_checkpoint_stem": str(destination),
        "derived_npz_sha256": _sha(paths[0]),
        "derived_json_sha256": _sha(paths[1]),
        "npz_bytes_equal_parent": paths[0].read_bytes() == parent.npz_path.read_bytes(),
        "numeric_array_count": len(array_checks),
        "roundtrip_checks": checks,
        "runtime_A_original_provenance_guard_accepted": True,
        "diagnostic_scope": (
            "one bounded retention10 diagnostic on explicitly selected nominal1003 "
            "and combined1004 harm cases; no optimization sweep, promotion, or outcome inferred"
        ),
        "devices": [str(device) for device in jax.devices()],
    }
    _save(destination.parent / "preparation.json", report)
    with (destination.parent / "README.md").open("x") as stream:
        stream.write(
            "# Nominal128 seed11 retention-10 continuation\n\n"
            "Cloned directly from the genuine nominal128 seed11 deployment checkpoint. "
            "Only the declared reference learning retention weight changes from 5 to 10; "
            "the new reference fingerprint and parent lineage are explicit in metadata.\n\n"
            "Every numerical payload array, full params/previous params/Adam/counters/model, "
            "teacher/anchors/spec, actor configuration and physical snapshot passed exact "
            "round-trip verification. Metadata mode remains the inherited `nominal`, and "
            "the existing runtime provenance guard accepted the checkpoint unchanged. "
            "No recovery diagnostic checkpoint was relabeled or used as the source.\n\n"
            "Preparation performed zero training updates and no controller or episode run. "
            "This prepares one retention diagnostic; it establishes no adaptation outcome. "
            "See `preparation.json` for hashes and all checks.\n"
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checkpoint_stem": str(destination),
                "numeric_array_count": len(array_checks),
                "all_checks_passed": True,
                "parent_npz_sha256": provenance["parent_npz_sha256"],
                "derived_npz_sha256": report["derived_npz_sha256"],
                "derived_reference_sha256": derived_reference,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
