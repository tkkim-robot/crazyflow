"""Confirm a promoted continuous collision case under prespecified paired ablations.

Every variant restarts its complete physical episode at time zero. Local scenes are drawn
and validated before any evaluation, with invalid draws retained and never replaced. This
driver uses the compact discovery evaluator for deterministic experiments and the canonical
publication scheduler for actual paced confirmation. It does not render or retune scenes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import jax
import numpy as np

from benchmark.da_plcbf_case_attribution import persistent_provenance
from benchmark.da_plcbf_case_discovery import CASE_CHECKPOINTS
from benchmark.da_plcbf_case_episode import run_case_episode
from benchmark.da_plcbf_closed_loop_search import (
    EpisodeEvaluator,
    checksum,
    classify_pair,
    write_json,
)
from crazyflow.safety.da_plcbf.case_study_world import (
    HoverEncounterConfig,
    IncomingSphere,
    build_hover_encounter_world,
)
from crazyflow.safety.da_plcbf.continuous_version_a import ContinuousVersionAConfig
from crazyflow.safety.da_plcbf.learner_checkpoint import save_learner_checkpoint
from crazyflow.safety.da_plcbf.navigation_experiment import NavigationExperimentConfig
from crazyflow.safety.da_plcbf.policy_qp_audit import (
    make_navigation_policy_qp_auditor,
    summarize_policy_qp_audit,
)
from crazyflow.safety.da_plcbf.state_conditioned_learning import (
    reference_contract_checkpoint_metadata,
    save_reference_contract,
)

MAPPINGS = ("uncompensated", "compensated")
OUTCOME_CLASSES = (
    "both_separated",
    "fixed_only_collision",
    "adaptive_only_collision",
    "both_collision",
    "censored_or_unresolved",
)


def confirmation_plan(
    scene: HoverEncounterConfig, *, seed: int = 48791, neighborhood_count: int = 12
) -> dict[str, Any]:
    """Freeze a local distribution and every draw before observing controller outcomes."""
    if type(seed) is not int or seed < 0:
        raise ValueError("confirmation seed must be a nonnegative integer")
    if type(neighborhood_count) is not int or neighborhood_count < 12:
        raise ValueError("confirmation requires at least twelve prespecified local scenes")
    build_hover_encounter_world(scene)
    if not np.isclose(scene.dt, 0.02) or scene.control_interval_steps != 2:
        raise ValueError("confirmation expects the validated .02 s prediction/.04 s hold")
    rng = np.random.default_rng(seed)
    neighbors = []
    for index in range(neighborhood_count):
        hover_delta = rng.uniform(-0.005, 0.005, 3)

        def move(sphere: IncomingSphere) -> IncomingSphere:
            return replace(
                sphere,
                crossing_offset=tuple(
                    np.asarray(sphere.crossing_offset)
                    - hover_delta
                    + rng.uniform(-0.015, 0.015, 3)
                ),
                arrival_time_seconds=sphere.arrival_time_seconds + rng.uniform(-0.04, 0.04),
                speed_m_s=sphere.speed_m_s + rng.uniform(-0.05, 0.05),
                radius_m=sphere.radius_m + rng.uniform(-0.005, 0.005),
            )

        neighbor = replace(
            scene,
            seed=seed + index,
            hover_position=tuple(np.asarray(scene.hover_position) + hover_delta),
            # Keep task waypoints absolute; each obstacle gets only its own declared draw.
            waypoint_offsets=tuple(
                tuple(np.asarray(point) - hover_delta) for point in scene.waypoint_offsets
            ),
            incoming=move(scene.incoming),
            additional_incoming=tuple(move(sphere) for sphere in scene.additional_incoming),
            guards=tuple(
                replace(
                    guard,
                    offset=tuple(
                        np.asarray(guard.offset) - hover_delta + rng.uniform(-0.015, 0.015, 3)
                    ),
                    radius_m=guard.radius_m + rng.uniform(-0.005, 0.005),
                )
                for guard in scene.guards
            ),
            initial_velocity=tuple(
                np.asarray(scene.initial_velocity) + rng.uniform(-0.01, 0.01, 3)
            ),
            wind_velocity=tuple(np.asarray(scene.wind_velocity) + rng.uniform(-0.05, 0.05, 3)),
            wind_onset_seconds=float(
                scene.wind_onset_seconds + int(rng.integers(-1, 2)) * 0.04
            ),
        )
        try:
            build_hover_encounter_world(neighbor)
        except ValueError as error:
            valid, reason = False, str(error)
        else:
            valid, reason = True, None
        data = asdict(neighbor)
        neighbors.append(
            {
                "index": index,
                "scene": data,
                "scene_sha256": checksum(data),
                "admissible": valid,
                "invalid_reason": reason,
            }
        )
    return {
        "schema": "da_plcbf_closed_loop_confirmation_plan_v1",
        "seed": seed,
        "scene": asdict(scene),
        "scene_sha256": checksum(asdict(scene)),
        "mappings": MAPPINGS,
        "fixed_penetration_threshold_m": -0.002,
        "adaptive_survival_threshold_m": 0.01,
        "threshold_scope": (
            "prespecified visible-separation targets; integration sensitivity is measured "
            "separately and subtracted from fine-grid separation before confirmation"
        ),
        "plant_dt_seconds": [0.02, 0.01, 0.005],
        "predictor_dt_seconds": scene.dt,
        "command_hold_seconds": 0.04,
        "termination_geometry": "modeled_collider",
        "controller_and_emergency_unchanged": True,
        "neighborhood_distribution": {
            "type": "independent bounded uniform draws, except discrete wind-onset shift",
            "initial_hover_position_and_hover_goal_half_width_m": 0.005,
            "absolute_waypoint_positions": "held fixed",
            "obstacle_absolute_center_offset_half_width_m": 0.015,
            "each_mover_arrival_half_width_seconds": 0.04,
            "each_mover_speed_half_width_m_s": 0.05,
            "each_obstacle_radius_half_width_m": 0.005,
            "initial_velocity_half_width_m_s": 0.01,
            "wind_velocity_half_width_m_s": 0.05,
            "wind_onset_shift_seconds": [-0.04, 0.0, 0.04],
            "physical_robot_radius_and_requested_clearance": "held fixed",
            "inadmissible_draws": "retained with reason, never replaced or retuned",
            "wind_training": "each complete perturbed episode trains on its own observations",
        },
        "neighborhood": neighbors,
    }


def compare_calm_prefix(
    continued: dict[str, np.ndarray], frozen: dict[str, np.ndarray], onset: float
) -> dict[str, Any]:
    """Check observed state, command, values, and publications through the onset input."""
    left = np.asarray(continued["time"]) <= onset + 1e-10
    right = np.asarray(frozen["time"]) <= onset + 1e-10
    fields = ("time", "state", "action", "hard", "smooth", "version_used")
    details = {}
    for field in fields:
        a, b = np.asarray(continued[field])[left], np.asarray(frozen[field])[right]
        same_shape = a.shape == b.shape
        maximum_difference = None
        if same_shape and a.size:
            delta = np.zeros(a.shape, dtype=float)
            np.subtract(a, b, out=delta, where=a != b)
            if np.isfinite(delta).all():
                maximum_difference = float(np.max(np.abs(delta)))
        details[field] = {
            "exact_match": bool(same_shape and np.array_equal(a, b)),
            "maximum_absolute_difference": maximum_difference,
        }
    reached_onset = bool(
        np.any(np.isclose(continued["time"], onset, rtol=0, atol=1e-9))
        and np.any(np.isclose(frozen["time"], onset, rtol=0, atol=1e-9))
    )
    return {
        "both_reached_wind_onset": reached_onset,
        "shared_calm_execution_verified": reached_onset
        and all(row["exact_match"] for row in details.values()),
        "fields": details,
        "scope": (
            "direct observable comparison through the wind-onset control; full parameter/Adam "
            "snapshot hashes are a separate provenance requirement"
        ),
    }


def integration_sensitivity(records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Apply observed .01-to-.005 margin sensitivity without calling it a rigorous error bound."""
    output = {}
    for method in ("fixed", "adaptive"):
        middle, fine = records[2]["methods"][method], records[4]["methods"][method]
        output[method] = {
            key: abs(fine[key] - middle[key])
            for key in ("collider_lower_m", "collider_upper_m", "ground_lower_m")
        }
    fine = records[4]["methods"]
    target = (
        fine["fixed"]["collider_upper_m"] + output["fixed"]["collider_upper_m"] < -0.002
        and fine["adaptive"]["collider_lower_m"] - output["adaptive"]["collider_lower_m"] > 0.01
        and fine["adaptive"]["ground_lower_m"] - output["adaptive"]["ground_lower_m"] > 0.01
        and all(row["promotion_candidate"] for row in records.values())
    )
    return {
        "observed_margin_changes_m": output,
        "collision_survival_target_survives_observed_integration_sensitivity": bool(target),
        "scope": (
            "empirical change between two refined held-plant integrations; predictor grid and "
            ".04 s commands unchanged; this is not a proven continuous-time integration bound"
        ),
    }


def attribution_boundary(scene: HoverEncounterConfig, fixed: dict[str, Any]) -> float | None:
    """Freeze a pre-encounter intervention clock; do not search it after branch outcomes."""
    event = fixed.get("first_collider_intersection_seconds")
    event = scene.incoming.arrival_time_seconds if event is None else event
    boundary = max(
        scene.wind_onset_seconds + 0.04, np.floor((event - 0.4) / 0.04 + 1e-9) * 0.04
    )
    return float(boundary) if boundary < event - 1e-9 else None


def parameter_reversion_case(
    engine: EpisodeEvaluator,
    scene: HoverEncounterConfig,
    mapping: str,
    fixed: dict[str, Any],
    directory: Path,
    *,
    boundary_seconds: float | None = None,
    boundary_reason: str | None = None,
) -> dict[str, Any]:
    """Intervene only on parameters after an identical observed adaptive episode prefix."""
    if boundary_seconds is not None:
        if (
            not np.isfinite(boundary_seconds)
            or not (
                scene.wind_onset_seconds + 0.04 - 1e-10 <= boundary_seconds < scene.duration_seconds
            )
            or not np.isclose(boundary_seconds / 0.04, round(boundary_seconds / 0.04))
        ):
            raise ValueError("explicit intervention must be a post-wind control boundary")
        if not isinstance(boundary_reason, str) or not boundary_reason.strip():
            raise ValueError("an explicit intervention boundary requires its selection reason")
    elif boundary_reason is not None:
        raise ValueError("a boundary reason requires an explicit boundary time")
    directory.mkdir(parents=True, exist_ok=False)
    when = (
        attribution_boundary(scene, fixed) if boundary_seconds is None else float(boundary_seconds)
    )
    protocol = {
        "boundary_seconds": when,
        "selection": (
            "fixed first collider event minus .4 s, or first mover arrival if separated"
            if boundary_seconds is None
            else "explicit follow-up boundary; original intervention results remain unchanged"
        ),
        "selection_reason": boundary_reason,
        "rounding": (
            "floor to .04 s; at least one wind update; no outcome-dependent retiming"
            if boundary_seconds is None
            else "explicit .04 s boundary after at least one wind update"
        ),
        "scope": "controlled parameter intervention after an actual full-scene adaptive prefix",
    }
    write_json(directory / "protocol.json", protocol)
    if when is None:
        return {**protocol, "available": False, "reason": "no pre-event wind-adapted boundary"}
    world = build_hover_encounter_world(scene)
    bundle, contract, _, _, _ = engine.resources(mapping, world)
    captures, traces, methods = {}, {}, {}
    for label in ("held_learned", "reverted_initial"):

        def capture(payload: dict[str, Any], *, capture_label: str = label) -> None:
            if np.isclose(payload["time"], when, rtol=0, atol=1e-9):
                captures[capture_label] = payload

        methods[label], traces[label] = engine.run_method(
            scene,
            mapping,
            "adaptive",
            freeze_at=when,
            revert_at=when if label == "reverted_initial" else None,
            snapshot_callback=capture,
        )
        np.savez_compressed(directory / f"{label}_traces.npz", **traces[label])
    if len(captures) != 2:
        result = {
            **protocol,
            "available": False,
            "reason": "one or both full episodes terminated before the prespecified boundary",
            "methods": methods,
        }
        write_json(directory / "result.json", result)
        return result
    save_reference_contract(contract, directory / "nominal_reference")
    binding = reference_contract_checkpoint_metadata(directory / "nominal_reference")
    provenance = {}
    for label, capture in captures.items():
        provenance[label] = {}
        for snapshot_name in ("snapshot", "used_snapshot"):
            snapshot = capture[snapshot_name]
            provenance[label][snapshot_name] = persistent_provenance(snapshot)
            publications = traces[label]["publication_time"]
            publications = publications[(publications >= 0) & (publications <= when + 1e-10)]
            available = float(publications.max()) if len(publications) else 0.0
            if label == "reverted_initial" and snapshot_name == "used_snapshot":
                available = 0.0
            save_learner_checkpoint(
                snapshot,
                bundle.spec,
                bundle.config,
                bundle.actuator,
                capture["state"],
                directory / f"{label}_{snapshot_name}",
                metadata={
                    **binding,
                    "available_time_seconds": available,
                    "training_before_seconds": available,
                    "physical_state_time_seconds": when,
                    "parameter_intervention": label,
                    "point_wind_for_external_replay": np.asarray(
                        capture["model"].wind_velocity
                    ).tolist(),
                    "previous_policy_index": int(capture["previous"]),
                    "goal_for_external_replay_only": np.asarray(capture["goal"]).tolist(),
                },
            )
        np.savez_compressed(
            directory / f"{label}_controller_input.npz",
            **{key: np.asarray(capture[key]) for key in ("state", "previous", "goal")},
            **{
                f"model_{key}": np.asarray(value)
                for key, value in capture["model"]._asdict().items()
            },
        )
    left, right = captures["held_learned"], captures["reverted_initial"]
    matching_inputs = {
        key: bool(np.array_equal(np.asarray(left[key]), np.asarray(right[key])))
        for key in ("state", "previous", "goal")
    }
    matching_inputs["point_model"] = all(
        np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(left["model"]), jax.tree.leaves(right["model"]), strict=True
        )
    )
    matching_inputs["published_full_learner_state"] = (
        provenance["held_learned"]["snapshot"]["persistent_state_sha256"]
        == provenance["reverted_initial"]["snapshot"]["persistent_state_sha256"]
    )
    prefix = compare_calm_prefix(traces["held_learned"], traces["reverted_initial"], when - 0.04)
    parameters_differ = (
        provenance["held_learned"]["used_snapshot"]["parameters_sha256"]
        != provenance["reverted_initial"]["used_snapshot"]["parameters_sha256"]
    )
    config = NavigationExperimentConfig(
        navigation_start_seconds=scene.navigation_start_seconds,
        fallback_mapping="compensated"
        if bundle.config.model_compensation
        else "matched_uncompensated",
        termination_geometry="modeled_collider",
    )
    auditor = make_navigation_policy_qp_auditor(world, bundle, config)
    prediction = jax.device_put(
        world.obstacle_prediction(when, horizon=bundle.config.horizon), engine.device
    )
    rollout_config = ContinuousVersionAConfig(
        dt=bundle.config.dt,
        horizon=bundle.config.horizon,
        control_interval_steps=bundle.config.control_interval_steps,
        obstacle_clearance=scene.obstacle_clearance,
        ego_radius=scene.ego_radius,
    )
    audits = {}
    for label, capture in captures.items():
        audit = auditor(
            capture["state"],
            capture["used_snapshot"].params,
            capture["model"],
            prediction,
            capture["previous"],
            capture["goal"],
        )
        audits[label] = summarize_policy_qp_audit(audit, prediction, rollout_config)
        index = int(np.flatnonzero(np.isclose(traces[label]["time"], when))[0])
        audits[label]["runtime_action_max_difference_vs_executed"] = float(
            np.max(np.abs(np.asarray(audit.runtime.action) - traces[label]["action"][index]))
        )
    write_json(directory / "FULL_QP_AUDIT.json", audits)
    paired = {"fixed": methods["reverted_initial"], "adaptive": methods["held_learned"]}
    result = {
        **protocol,
        "available": True,
        "same_controller_inputs_and_published_snapshot": matching_inputs,
        "shared_prefix_execution": {
            "verified": prefix["shared_calm_execution_verified"],
            "fields": prefix["fields"],
            "scope": "all executed controls strictly before the intervention boundary",
        },
        "parameters_differ_at_intervention": parameters_differ,
        "common_state_intervention_valid": (
            all(matching_inputs.values())
            and prefix["shared_calm_execution_verified"]
            and parameters_differ
        ),
        "provenance": provenance,
        "methods": paired,
        "fixed_label": "original parameters after common adaptive prefix",
        "adaptive_label": "available learned parameters held after the same prefix",
        **classify_pair(paired),
    }
    write_json(directory / "result.json", result)
    return result


def _paced_record(report: dict[str, Any], scene: HoverEncounterConfig) -> dict[str, Any]:
    methods = {}
    last_arrival = max(
        sphere.arrival_time_seconds for sphere in (scene.incoming, *scene.additional_incoming)
    )
    for name, source in report["methods"].items():
        sphere, floor = (
            source["xml_sphere_geometric_intersection"],
            source["xml_ground_geometric_intersection"],
        )
        physical = source["execution_audit"]
        end = source["termination_time_seconds"]
        methods[name] = {
            "termination": source["termination"],
            "censored": source["enclosure_termination_censors_later_collider_outcome"],
            "collider_lower_m": sphere["minimum_clearance_lower_bound_m"],
            "collider_upper_m": sphere["minimum_clearance_upper_bound_m"],
            "ground_lower_m": floor["minimum_clearance_lower_bound_m"],
            "ground_upper_m": floor["minimum_clearance_upper_bound_m"],
            "all_operational_nodes_pass": physical["all_actual_physical_nodes_pass"]
            and physical["applied_motor_limit_violating_controls"] == 0,
            "encounter_completed": bool(
                end is not None
                and end > last_arrival + 0.8
                and source["modeled_collider_collision"] is False
            ),
            "finite_updates": source["finite_updates"],
            "deadline_misses": source["service_exceeds_nominal_period_count"],
            "availability_at_arrival": source["availability_at_arrival"],
        }
    return {
        "methods": methods,
        **classify_pair(methods),
        "actual_positive_used_updates_zero_deadline_misses": report[
            "positive_used_updates_with_zero_paced_deadline_misses"
        ],
    }


def run_confirmation(
    selected: Path,
    output: Path,
    *,
    mode: str = "deterministic",
    seed: int = 48791,
    neighborhood_count: int = 12,
    device_name: str = "gpu",
) -> dict[str, Any]:
    """Persist the frozen plan first, then execute bounded full-episode confirmation."""
    if mode not in {"plan", "deterministic", "paced", "all"}:
        raise ValueError("unknown confirmation mode")
    original = json.loads(selected.read_text())
    scene = HoverEncounterConfig.from_dict(original["scene"])
    plan = confirmation_plan(scene, seed=seed, neighborhood_count=neighborhood_count)
    plan.update(
        selected_result=str(selected),
        selected_result_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
        requested_mode=mode,
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path("benchmark/da_plcbf_closed_loop_search.py"),
                Path("crazyflow/safety/da_plcbf/navigation_experiment.py"),
                Path("crazyflow/safety/da_plcbf/case_study_world.py"),
            )
        },
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    if mode == "plan":
        return plan
    device = jax.devices(device_name)[0]
    engines = {
        substeps: EpisodeEvaluator(device, plant_substeps=substeps) for substeps in (1, 2, 4)
    }
    ledger, reports = [], {}

    def retain(label: str, mapping: str, record: dict[str, Any]) -> None:
        row = {"variant": label, "mapping": mapping, **record}
        ledger.append(row)
        with (output / "trials.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        print(
            json.dumps({"variant": label, "mapping": mapping, "outcome": row["outcome_class"]}),
            flush=True,
        )

    for mapping in MAPPINGS:
        mapping_report = {}
        reports[mapping] = mapping_report
        if mode in {"deterministic", "all"}:
            baseline = engines[1].evaluate(scene, mapping, output / mapping / "original")
            retain("original", mapping, baseline)
            records = {1: baseline}
            for substeps in (2, 4):
                records[substeps] = engines[substeps].evaluate(
                    scene, mapping, output / mapping / f"plant_substeps_{substeps}"
                )
                retain(f"plant_substeps_{substeps}", mapping, records[substeps])
            mapping_report["integration_sensitivity"] = integration_sensitivity(records)
            frozen, arrays = engines[1].run_method(
                scene, mapping, "adaptive", freeze_at=scene.wind_onset_seconds
            )
            onset_dir = output / mapping / "freeze_at_wind_onset"
            onset_dir.mkdir()
            np.savez_compressed(onset_dir / "traces.npz", **arrays)
            with np.load(output / mapping / "original/traces.npz", allow_pickle=False) as saved:
                continued = {
                    key.removeprefix("adaptive_"): saved[key]
                    for key in saved.files
                    if key.startswith("adaptive_")
                }
            calm = compare_calm_prefix(continued, arrays, scene.wind_onset_seconds)
            onset_methods = {"fixed": frozen, "adaptive": baseline["methods"]["adaptive"]}
            onset = {
                "methods": onset_methods,
                **classify_pair(onset_methods),
                "fixed_label": "same calm learner prefix, frozen at actual wind-onset snapshot",
                "shared_calm_audit": calm,
                "adaptive_trace": "../original/traces.npz:adaptive_*",
            }
            write_json(onset_dir / "result.json", onset)
            retain("freeze_at_wind_onset", mapping, onset)
            mapping_report["freeze_at_wind_onset"] = onset
            attribution = parameter_reversion_case(
                engines[1],
                scene,
                mapping,
                baseline["methods"]["fixed"],
                output / mapping / "parameter_reversion",
            )
            if "outcome_class" not in attribution:
                attribution["outcome_class"] = "attribution_unavailable"
            retain("parameter_reversion", mapping, attribution)
            mapping_report["parameter_reversion"] = attribution
            no_wind = engines[1].evaluate(
                replace(scene, wind_velocity=(0.0, 0.0, 0.0)), mapping, output / mapping / "no_wind"
            )
            retain("no_wind", mapping, no_wind)
            counts = Counter({name: 0 for name in (*OUTCOME_CLASSES, "invalid_scene")})
            strict_targets = 0
            for neighbor in plan["neighborhood"]:
                label = f"neighborhood_{neighbor['index']:03d}"
                if not neighbor["admissible"]:
                    result = {**neighbor, "outcome_class": "invalid_scene"}
                else:
                    result = engines[1].evaluate(
                        HoverEncounterConfig.from_dict(neighbor["scene"]),
                        mapping,
                        output / mapping / label,
                    )
                retain(label, mapping, result)
                counts[result["outcome_class"]] += 1
                strict_targets += bool(result.get("promotion_candidate", False))
            mapping_report["neighborhood_outcomes"] = dict(counts)
            mapping_report["neighborhood_strict_target_count"] = strict_targets
            mapping_report["neighborhood_admissible_pairs_executed"] = (
                neighborhood_count - counts["invalid_scene"]
            )
        if mode in {"paced", "all"}:
            report = run_case_episode(
                scene,
                CASE_CHECKPOINTS[mapping],
                output / mapping / "canonical_paced",
                execution_mode="budgeted",
                termination_geometry="modeled_collider",
                device=device,
            )
            paced = _paced_record(report, scene)
            retain("canonical_paced", mapping, paced)
            mapping_report["canonical_paced"] = paced
    summary = {
        "schema": "da_plcbf_closed_loop_confirmation_result_v1",
        "mode": mode,
        "plan_sha256": hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest(),
        "mappings": reports,
        "outcomes_all_variants": dict(Counter(row["outcome_class"] for row in ledger)),
        "separation_definition": (
            "geometric outcome only; encounter completion, mission completion, operational "
            "limits, finite updates, and actual deadlines remain independent requirements"
        ),
        "attribution_scope": (
            "deterministic mode records actual continuous-prefix snapshots, original-parameter "
            "reversion, and every eligible complete QP at one prespecified boundary; positive "
            "mechanism attribution still depends on the recorded counterfactual outcomes"
        ),
        "paced_confirmation_executed": mode in {"paced", "all"},
        "deterministic_confirmation_executed": mode in {"deterministic", "all"},
    }
    write_json(output / "CONFIRMATION_SUMMARY.json", summary)
    return summary


def main() -> None:
    """Prepare or execute a selected case without modifying any search geometry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("plan", "deterministic", "paced", "all"), default="deterministic"
    )
    parser.add_argument("--seed", type=int, default=48791)
    parser.add_argument("--neighborhood-count", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    args = parser.parse_args()
    run_confirmation(
        args.selected,
        args.output,
        mode=args.mode,
        seed=args.seed,
        neighborhood_count=args.neighborhood_count,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
