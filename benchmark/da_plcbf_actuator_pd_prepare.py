"""Nominal-only tuning of an explicit handcrafted feedback library, before fault trials."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from benchmark.da_plcbf_actuator_behavior import _compact_probe, validation_state_bank
from benchmark.da_plcbf_actuator_study import write_json
from crazyflow.safety.da_plcbf.actuator_learning import (
    ActuatorReferenceConfig,
    actuator_proprioceptive_state_bank,
    build_actuator_skill_learner,
    probe_actuator_library,
    save_actuator_learner_checkpoint,
)
from crazyflow.safety.da_plcbf.actuator_libraries import PD_PRIMITIVE_NAMES, handcrafted_pd_contract
from crazyflow.safety.da_plcbf.actuator_study import nominal_actuator_model


def prepare(output: Path) -> dict:
    """Choose solely on nominal development braking with fixed nontrivial spread checks."""
    output.mkdir(parents=True, exist_ok=False)
    model = nominal_actuator_model()
    training, labels = actuator_proprioceptive_state_bank(model)
    validation, validation_labels = validation_state_bank(model)
    # These choices are committed to the output before any rollout evaluation.
    grid = [
        {"policy_gain": g, "duration": d, "speed": 1.0}
        for g in (1.8, 2.4, 3.0)
        for d in (0.35, 0.50)
    ]
    write_json(
        output / "protocol.json",
        {
            "grid": grid,
            "initial_neural_residual": "identically zero",
            "selection": (
                "passing nominal development criteria, then minimum maximum terminal speed"
            ),
            "tie_break": "grid index",
            "validation_used_for_selection": False,
            "development_state_count": len(training),
            "validation_state_count": len(validation),
            "online_trainable_coordinates": "48 desired-velocity offsets + 16 duration offsets",
            "velocity_offset_bounds_mps": [-0.5, 0.5],
            "effective_duration_bounds_seconds": [0.1, 1.2],
            "every_finite_online_update_published": True,
            "primitive_names": PD_PRIMITIVE_NAMES,
        },
    )
    candidates = []
    contracts = []
    objective = ActuatorReferenceConfig(
        objective_mode="balanced_reference_braking", recovery_braking_priority=10.0
    )
    for index, parameters in enumerate(grid):
        contract = handcrafted_pd_contract(model, objective=objective, **parameters)
        contracts.append(contract)
        probe = probe_actuator_library(
            contract.params, contract, model, states=training, labels=labels
        )
        write_json(output / f"development-{index:02d}.json", probe)
        row = {"index": index, **parameters, **_compact_probe(probe)}
        candidates.append(row)
        print(json.dumps(row), flush=True)
    passing = [row for row in candidates if row["competent_under_declared_criteria"]]
    if not passing:
        write_json(
            output / "summary.json",
            {"status": "no_nominally_competent_PD_candidate", "candidates": candidates},
        )
        raise RuntimeError("PD library needs a separate declared repair; no checkpoint promoted")
    selected = min(passing, key=lambda row: (row["maximum_terminal_speed_mps"], row["index"]))
    contract = contracts[selected["index"]]
    validation_probe = probe_actuator_library(
        contract.params, contract, model, states=validation, labels=validation_labels
    )
    write_json(output / "validation.json", validation_probe)
    learner = build_actuator_skill_learner(contract)
    state = learner.initialize(contract.params, model)
    paths = save_actuator_learner_checkpoint(
        state,
        contract,
        np.asarray(training[0]),
        output / "deployment",
        config=contract.actor_config,
        metadata={
            "purpose": "handcrafted_PD_matched_initialization",
            "selection": selected,
            "offline_gradient_steps": 0,
            "online_trainable_parameters": 64,
            "nominal_validation_competent": validation_probe["competent_under_declared_criteria"],
        },
    )
    summary = {
        "status": "complete",
        "candidates": candidates,
        "selected": selected,
        "validation": _compact_probe(validation_probe),
        "checkpoint_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "actor_config": asdict(contract.actor_config),
        "objective": asdict(objective),
        "devices": [str(d) for d in jax.devices()],
    }
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output)
