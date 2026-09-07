"""Create CSV-derived figures and a compact, checksummed diagnostic review archive.

This reader never imports the model or launches numerical experiments. ``tables``
may describe an incomplete campaign; ``collect`` refuses an incomplete original
protocol. Collection inventories and hashes omitted raw arrays and movies as well
as included metadata. It must run after all numerical and rendering work stops.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.da_plcbf_actuator_publication import (
    hash_file,
    json_bytes,
    read_json,
    relative,
    require,
    safe_member,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/da_plcbf/actuator-diagnostics-20260906/v1"
PROTOCOL = ROOT / "artifacts/da_plcbf/actuator-diagnostic-20260906/v1/protocol-sealed-v1.json"
METHODS = ("F2", "A", "A_BAL", "PD_F", "PD_A", "DR", "UNION", "F2_2K")
SCHEMA = "da_plcbf_actuator_diagnostic_evidence_v1"
ARCHIVE = "review-only.tar.gz"


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(json_bytes(value))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"no rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def outcome(summary: dict[str, Any]) -> str:
    if summary.get("status") != "completed":
        return "incomplete"
    if summary["modeled_collider_collision"]:
        return "collision"
    if summary["successful_full_episode"]:
        return "safe_task"
    if not summary["actual_operational_all_nodes_pass"]:
        return "operational_failure"
    if summary["reached_full_duration"] and not summary["task_completed"]:
        return "timeout"
    return "other_failure"


def protocol_records(base: Path, protocol: Path) -> tuple[dict[str, Any], dict[str, Any], dict]:
    envelope = read_json(protocol)
    proposal = envelope["proposal"]
    expected = {trial["trial_id"]: trial for trial in proposal["trials"]}
    records: dict[str, Any] = {}
    sources = {relative(protocol): hash_file(protocol)[0]}
    for path in sorted(base.rglob("campaign_result.json")):
        campaign = read_json(path)
        sources[relative(path)] = hash_file(path)[0]
        for record in campaign["records"]:
            identity = record["trial"]["trial_id"]
            require(identity in expected, f"unplanned trial in original campaign: {identity}")
            require(record["trial"] == expected[identity], f"changed trial identity: {identity}")
            if identity in records:
                require(
                    record["summary"] == records[identity]["summary"]
                    and record["evidence_sha256"] == records[identity]["evidence_sha256"],
                    f"conflicting duplicate trial: {identity}",
                )
            else:
                records[identity] = record
    return proposal, records, sources


def secondary_records(base: Path) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, str]]:
    root = base / "secondary-refresh-validation-v1"
    protocol = root / "protocol-sealed-v1.json"
    if not protocol.exists():
        return None, {}, {}
    proposal = read_json(protocol)["proposal"]
    expected = {trial["trial_id"]: trial for trial in proposal["trials"]}
    require(len(expected) == 40, "changed secondary budget")
    sources = {relative(protocol): hash_file(protocol)[0]}
    records: dict[str, Any] = {}
    for path in sorted((root / "results-v1").rglob("results.json")):
        part = read_json(path)
        sources[relative(path)] = hash_file(path)[0]
        for record in part["records"]:
            identity = record["trial"]["trial_id"]
            require(identity in expected, f"unplanned secondary trial: {identity}")
            require(
                record["trial"] == expected[identity], f"changed secondary identity: {identity}"
            )
            require(
                identity not in records or records[identity] == record,
                f"conflicting secondary duplicate: {identity}",
            )
            records[identity] = record
    return proposal, records, sources


def secondary_analysis_tables(base: Path, table_dir: Path, sources: dict[str, str]) -> dict:
    root = base / "secondary-refresh-validation-v1/analysis-v1"
    index_path = root / "report.json"
    if not index_path.exists():
        return {"status": "pending"}
    index = read_json(index_path)
    sources[relative(index_path)] = hash_file(index_path)[0]
    pair_rows, common_rows = [], []
    for stage in ("fresh_validation", "numerical_robustness"):
        stage_path = root / stage / "report.json"
        report = read_json(stage_path)
        sources[relative(stage_path)] = hash_file(stage_path)[0]
        for pair in report["rows"]:
            command = pair.get("actual_command_comparison", {}).get("first_divergence")
            row = {
                "stage": stage,
                "pair_id": pair["pair_id"],
                "world": pair["world_key"],
                "cell": pair["cell_id"],
                "realization": pair["realization"],
                "declared_controller": pair["declared_controller"],
                "analysis_status": pair["status"],
                "admissible_pair": pair["admissible_pair"],
                "wrapper_prefix_exact_equal": pair.get("wrapper_prefix_exact_equal"),
                "common_state_completed": pair.get("common_state_completed", False),
                "task_outcome": pair.get("task_outcome"),
                "collision_free_outcome": pair.get("collision_free_outcome"),
                "first_command_divergence_time_s": command["time_seconds"] if command else None,
                "maximum_command_difference_N": (
                    command["maximum_motor_command_difference_n"] if command else None
                ),
                "source_report": relative(stage_path),
                "source_sha256": sources[relative(stage_path)],
            }
            for arm in ("frozen", "adaptive"):
                summary = pair.get("retained_summaries", {}).get(arm, {})
                row[f"{arm}_outcome"] = outcome(summary)
                row[f"{arm}_physical_time_s"] = summary.get("physical_time_seconds")
                row[f"{arm}_declared_duration_s"] = summary.get("duration_seconds")
                row[f"{arm}_publications"] = len(summary.get("snapshot_publications", []))
            pair_rows.append(row)
            if not pair.get("common_state_completed"):
                continue
            common_path = root / stage / pair["pair_id"] / "common_state_report.json"
            sources[relative(common_path)] = hash_file(common_path)[0]
            common = read_json(common_path)["common_state_evaluation"]
            for measurement in common["rows"]:
                common_rows.append(
                    {
                        "stage": stage,
                        "pair_id": pair["pair_id"],
                        "world": pair["world_key"],
                        "cell": pair["cell_id"],
                        "realization": pair["realization"],
                        "task_outcome": pair["task_outcome"],
                        **measurement,
                        "source_report": relative(common_path),
                        "source_sha256": sources[relative(common_path)],
                        "per_skill_values_sha256": common["per_skill_values_sha256"],
                    }
                )
    write_csv(table_dir / "secondary_union_refresh_pairs.csv", pair_rows)
    if common_rows:
        write_csv(table_dir / "secondary_union_refresh_common_states.csv", common_rows)
    return {
        "stages": index["stages"],
        "retained_pairs": len(pair_rows),
        "common_state_rows": len(common_rows),
    }


def tables(base: Path, protocol: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / ARCHIVE).exists(), "sealed review archive exists; use a new output")
    table_dir = output / "tables"
    table_dir.mkdir(exist_ok=True)
    proposal, records, sources = protocol_records(base, protocol)
    stages = proposal["stage_trial_ids"]
    cells = {cell["cell_id"]: cell for cell in proposal["factorial_cells"]}
    flight_rows = []
    for identity, record in sorted(records.items()):
        trial, summary = record["trial"], record["summary"]
        cell = cells[trial["cell_id"]]
        summary_path = Path(record["episode_directory"]) / "summary.json"
        require(
            read_json(summary_path) == summary,
            f"campaign summary differs from episode source: {summary_path}",
        )
        summary_digest = hash_file(summary_path)[0]
        require(
            record["evidence_sha256"][str(summary_path)] == summary_digest,
            f"summary digest mismatch: {summary_path}",
        )
        flight_rows.append(
            {
                "trial_id": identity,
                "stages": ";".join(stage for stage, ids in stages.items() if identity in ids),
                "split": trial["split"],
                "world": trial["world_key"],
                "cell": trial["cell_id"],
                "eta_motors01": cell["effectiveness_after"][0],
                "lag_motors01": cell["lag_multipliers_after"][0],
                "fault_time_s": cell["event_time_seconds"],
                "extra_lead_s": cell["extra_lead_seconds"],
                "arm": trial["arm"]["arm"],
                "realization": trial["realization"],
                "outcome": outcome(summary),
                "status": summary["status"],
                "safe_task_completion": summary["successful_full_episode"],
                "modeled_collider_collision": summary["modeled_collider_collision"],
                "operational_all_nodes_pass": summary["actual_operational_all_nodes_pass"],
                "waypoints_completed": summary["waypoints_completed"],
                "waypoints_total": summary["waypoints_total"],
                "physical_time_s": summary["physical_time_seconds"],
                "declared_duration_s": summary["duration_seconds"],
                "finite_credited_updates": summary["finite_credited_updates"],
                "publications": len(summary["snapshot_publications"]),
                "source_summary": relative(summary_path),
                "source_sha256": summary_digest,
                "runtime_binding_sha256": record["runtime_binding_sha256"],
            }
        )
    write_csv(table_dir / "flight_source.csv", flight_rows)
    secondary_proposal, secondary, secondary_sources = secondary_records(base)
    sources.update(secondary_sources)
    secondary_rows = []
    for identity, record in sorted(secondary.items()):
        trial, summary = record["trial"], record.get("summary") or {}
        record_path = Path(record["episode_directory"]).parent / "record.json"
        secondary_rows.append(
            {
                "trial_id": identity,
                "world": trial["world_key"],
                "cell": trial["cell_id"],
                "arm": trial["arm"]["arm"],
                "realization": trial["realization"],
                "split": trial["split"],
                "outcome": outcome(summary),
                "status": summary.get("status", "incomplete"),
                "declared_duration_s": summary.get("duration_seconds"),
                "physical_time_s": summary.get("physical_time_seconds"),
                "safe_task_completion": summary.get("successful_full_episode"),
                "collision": summary.get("modeled_collider_collision"),
                "operational_all_nodes_pass": summary.get("actual_operational_all_nodes_pass"),
                "waypoints_completed": summary.get("waypoints_completed"),
                "waypoints_total": summary.get("waypoints_total"),
                "finite_credited_updates": summary.get("finite_credited_updates"),
                "publications": len(summary.get("snapshot_publications", [])),
                "wrapper_commands_authenticated": record["actual_input_and_command_verification"],
                "source_record": relative(record_path),
                "source_sha256": hash_file(record_path)[0],
            }
        )
    if secondary_rows:
        write_csv(table_dir / "secondary_union_refresh_source.csv", secondary_rows)
    development = [row for row in flight_rows if "library_comparison" in row["stages"]]
    require(len(development) == 32, "the development figure requires all 32 declared trials")
    write_csv(table_dir / "development_source.csv", development)
    aggregate = []
    for method in METHODS:
        rows = [row for row in development if row["arm"] == method]
        counts = Counter(row["outcome"] for row in rows)
        aggregate.append(
            {
                "method": method,
                "episodes": len(rows),
                **{
                    key: counts[key]
                    for key in (
                        "safe_task",
                        "collision",
                        "timeout",
                        "operational_failure",
                        "other_failure",
                        "incomplete",
                    )
                },
                "operational_violations": sum(
                    not row["operational_all_nodes_pass"] for row in rows
                ),
            }
        )
    write_csv(table_dir / "development_aggregate.csv", aggregate)
    runtime_path = base / "runtime-measured-v2-audit-v1/report.json"
    runtime = read_json(runtime_path)
    sources[relative(runtime_path)] = hash_file(runtime_path)[0]
    runtime_rows = []
    for record in runtime["records"]:
        async_record = record["asynchronous_learner"] or {}
        runtime_rows.append(
            {
                "method": record["method"],
                "mode": record["execution_mode"],
                "status": record["status"],
                "safe_task_completion": record["safe_task_completion"],
                "collision": record["collision"],
                "physical_time_s": record["physical_time_seconds"],
                "controller_calls": record["controller_calls"],
                "controller_mean_ms": 1000 * record["controller_seconds"]["mean"],
                "controller_p95_ms": 1000 * record["controller_seconds"]["p95"],
                "controller_max_ms": 1000 * record["controller_seconds"]["maximum"],
                "controller_misses": record["controller_deadline_misses"],
                "skipped_sensing_ticks": record["skipped_sensing_ticks"],
                "learner_jobs_completed": async_record.get("completed_jobs", 0),
                "learner_service_mean_ms": (
                    1000 * record["learner_seconds"]["mean"]
                    if record["learner_seconds"]["mean"] is not None
                    else None
                ),
                "learner_service_p95_ms": (
                    1000 * record["learner_seconds"]["p95"]
                    if record["learner_seconds"]["p95"] is not None
                    else None
                ),
                "inflight_publications": record["published_online_updates"],
                "snapshot_age_p95_ms": 1000 * record["snapshot_age_seconds"]["p95"],
                "sensing_to_application_p95_ms": (
                    1000 * record["actual_sensing_to_command_application_seconds"]["p95"]
                ),
                "actual_command_hold_p95_ms": 1000 * record["actual_command_hold_seconds"]["p95"],
                "actual_command_hold_maximum_ms": (
                    1000 * record["actual_command_hold_seconds"]["maximum"]
                ),
                "controller_services_with_learner_host_overlap": async_record.get(
                    "controls_with_host_overlap", 0
                ),
                "learner_controller_host_overlap_ms": 1000
                * async_record.get("controller_host_overlap_seconds", 0),
                "source_report": relative(runtime_path),
                "source_sha256": sources[relative(runtime_path)],
                "source_episode": relative(Path(record["episode_directory"])),
            }
        )
    write_csv(table_dir / "runtime_source.csv", runtime_rows)
    cost_path = base / "controller-cost-measured-v1/report.json"
    if cost_path.exists():
        cost = read_json(cost_path)
        sources[relative(cost_path)] = hash_file(cost_path)[0]
        cost_rows = []
        for group in cost["groups"]:
            cost_rows.append(
                {
                    "case": group["case"],
                    "method": group["method"],
                    "fallback_count": group["total_fallback_count"],
                    "calls": group["service_seconds"]["count"],
                    "median_ms": 1000 * group["service_seconds"]["median"],
                    "mean_ms": 1000 * group["service_seconds"]["mean"],
                    "p95_ms": 1000 * group["service_seconds"]["p95"],
                    "maximum_ms": 1000 * group["service_seconds"]["maximum"],
                    "distinct_output_hashes": group["distinct_output_hashes"],
                    "source_report": relative(cost_path),
                    "source_sha256": sources[relative(cost_path)],
                }
            )
        write_csv(table_dir / "controller_cost_source.csv", cost_rows)
    for name in ("fixed_bank_source.csv", "gradient_source.csv"):
        source = base / "loss-diagnosis-summary-v1" / name
        (table_dir / name).write_bytes(source.read_bytes())
        sources[relative(source)] = hash_file(source)[0]
    for stage in ("matched_factorial", "fresh_validation", "numerical_robustness"):
        analysis_path = base / f"{stage.replace('_', '-')}-analysis-v1/report.json"
        completed_common = base / f"{stage.replace('_', '-')}-common-states-v1/report.json"
        if completed_common.exists():
            analysis_path = completed_common
        if not analysis_path.exists():
            continue
        analysis = read_json(analysis_path)
        sources[relative(analysis_path)] = hash_file(analysis_path)[0]
        pair_rows = []
        for pair in analysis["rows"]:
            require(pair["admissible_pair"], f"inadmissible pair: {pair['pair_id']}")
            command = pair["actual_command_comparison"]["first_divergence"]
            row = {
                "pair_id": pair["pair_id"],
                "world": pair["world_key"],
                "cell": pair["cell_id"],
                "runtime_method": pair["runtime_method"],
                "realization": pair["realization"],
                "event_time_s": pair["event_time_seconds"],
                "task_outcome": pair["task_outcome"],
                "collision_free_outcome": pair["collision_free_outcome"],
                "admissible_pair": pair["admissible_pair"],
                "analysis_status": pair["status"],
                "first_command_divergence_time_s": command["time_seconds"] if command else None,
                "maximum_command_difference_N": (
                    command["maximum_motor_command_difference_n"] if command else None
                ),
                "first_divergence_identical_physical_state": (
                    command["identical_physical_state"] if command else None
                ),
                "source_report": relative(analysis_path),
                "source_sha256": sources[relative(analysis_path)],
                "pair_outcome_report_sha256": pair["outcome_report_sha256"],
            }
            for arm in ("frozen", "adaptive"):
                summary = pair["retained_summaries"][arm]
                row[f"{arm}_outcome"] = outcome(summary)
                row[f"{arm}_waypoints_completed"] = summary["waypoints_completed"]
                row[f"{arm}_physical_time_s"] = summary["physical_time_seconds"]
                row[f"{arm}_publications"] = len(summary["snapshot_publications"])
                row[f"{arm}_source_episode"] = relative(
                    Path(pair["record_resolution"][arm]["actual_episode_directory"])
                )
            pair_rows.append(row)
        write_csv(table_dir / f"{stage}_pairs.csv", pair_rows)
    common_rows = []
    for stage in ("matched_factorial", "fresh_validation", "numerical_robustness"):
        common_dir = base / f"{stage.replace('_', '-')}-common-states-v1"
        common_index = common_dir / "report.json"
        if not common_index.exists():
            continue
        index = read_json(common_index)
        sources[relative(common_index)] = hash_file(common_index)[0]
        for pair in index["rows"]:
            require(
                pair["common_state_completed"], f"unfinished common-state pair: {pair['pair_id']}"
            )
            reports = list(common_dir.glob(f"**/{pair['pair_id']}/common_state_report.json"))
            require(len(reports) == 1, f"ambiguous common-state report: {pair['pair_id']}")
            report_path = reports[0]
            digest = hash_file(report_path)[0]
            require(digest == pair["common_state_report_sha256"], "common-state digest mismatch")
            report = read_json(report_path)["common_state_evaluation"]
            for row in report["rows"]:
                common_rows.append(
                    {
                        "stage": stage,
                        "pair_id": pair["pair_id"],
                        "world": pair["world_key"],
                        "cell": pair["cell_id"],
                        "runtime_method": pair["runtime_method"],
                        "realization": pair["realization"],
                        "time_s": row["time_seconds"],
                        "physical_branch": row["physical_branch"],
                        "task_outcome": pair["task_outcome"],
                        "frozen_max_fallback_hard_value": row["frozen_max_fallback_hard_value"],
                        "adaptive_max_fallback_hard_value": row["adaptive_max_fallback_hard_value"],
                        "adaptive_minus_frozen_hard_value": (
                            row["adaptive_max_fallback_hard_value"]
                            - row["frozen_max_fallback_hard_value"]
                        ),
                        "diagnostic_union_minus_fault_frozen": row[
                            "diagnostic_union_minus_fault_frozen"
                        ],
                        "shared_inputs_sha256": row["shared_inputs_sha256"],
                        "source_report": relative(report_path),
                        "source_sha256": digest,
                        "per_skill_values_sha256": report["per_skill_values_sha256"],
                    }
                )
    if common_rows:
        write_csv(table_dir / "common_state_source.csv", common_rows)
    boundary_source = base / "first-command-boundaries-v1/boundaries.csv"
    if boundary_source.exists():
        (table_dir / "first_command_boundaries.csv").write_bytes(boundary_source.read_bytes())
        sources[relative(boundary_source)] = hash_file(boundary_source)[0]
    secondary_analysis = secondary_analysis_tables(base, table_dir, sources)
    stage_status = {
        stage: {
            "planned_memberships": len(ids),
            "available_memberships": sum(identity in records for identity in ids),
            "completed_memberships": sum(
                records[identity]["summary"]["status"] == "completed"
                for identity in ids
                if identity in records
            ),
        }
        for stage, ids in stages.items()
    }
    manifest = {
        "schema": SCHEMA,
        "scope": "source-derived descriptive tables; repeated worlds are not independent samples",
        "generator_sha256": hash_file(Path(__file__).resolve())[0],
        "unique_original_trials_planned": len(proposal["trials"]),
        "unique_original_trials_available": len(records),
        "unique_original_trials_completed": sum(
            record["summary"]["status"] == "completed" for record in records.values()
        ),
        "stages": stage_status,
        "runtime_completed_attempts": runtime["completed_attempts"],
        "secondary_union_refresh": {
            "planned": len(secondary_proposal["trials"]) if secondary_proposal else 0,
            "retained": len(secondary),
            "completed": sum(
                (record.get("summary") or {}).get("status") == "completed"
                for record in secondary.values()
            ),
            "analysis": secondary_analysis,
        },
        "sources": sources,
    }
    write_json(output / "tables_manifest.json", manifest)
    return manifest


def figures(output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    rows = read_csv(output / "tables/development_source.csv")
    columns = [
        ("structured_30101", "eta1_lag1_extra0"),
        ("structured_30101", "eta0.7_lag1_extra0"),
        ("structured_61001", "eta1_lag1_extra0"),
        ("structured_61001", "eta0.7_lag1_extra0"),
    ]
    categories = {"safe_task": 0, "timeout": 1, "collision": 2}
    colors = ["#326e5c", "#d5a247", "#b24c49"]
    lookup = {(row["arm"], row["world"], row["cell"]): row for row in rows}
    matrix = [
        [categories[lookup[(method, *key)]["outcome"]] for key in columns] for method in METHODS
    ]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.imshow(matrix, cmap=ListedColormap(colors), vmin=-0.5, vmax=2.5, aspect="auto")
    ax.set_yticks(range(len(METHODS)), METHODS)
    ax.set_xticks(
        range(4),
        ["30101\nNominal", "30101\nEffectiveness .7", "61001\nNominal", "61001\nEffectiveness .7"],
    )
    for i, method in enumerate(METHODS):
        for j, key in enumerate(columns):
            row = lookup[(method, *key)]
            label = {"safe_task": "Complete", "timeout": "Timeout", "collision": "Collision"}[
                row["outcome"]
            ]
            ax.text(j, i, label, ha="center", va="center", color="white")
    ax.set_title("Development outcomes · 32 runs, two geometries, one library seed", pad=14)
    ax.set_xticks([0.5, 1.5, 2.5], minor=True)
    ax.set_yticks([i + 0.5 for i in range(7)], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(figure_dir / "01_development_outcomes.png", dpi=180)
    fig.savefig(figure_dir / "01_development_outcomes.pdf")
    plt.close(fig)

    bank = [
        row
        for row in read_csv(output / "tables/fixed_bank_source.csv")
        if row["bank"] == "fresh_validation16"
        and row["cell"] == "effectiveness"
        and row["library"] == "learned"
    ]
    order = [
        ("F2", "frozen", "Frozen F2"),
        ("legacy", "persistent", "Legacy · persistent"),
        ("legacy", "reset", "Legacy · reset"),
        ("balanced_reference", "persistent", "Balanced · persistent"),
        ("balanced_reference", "reset", "Balanced · reset"),
        ("balanced_reference_braking", "persistent", "Braking ×10 · persistent"),
        ("balanced_reference_braking", "reset", "Braking ×10 · reset"),
    ]
    indexed = {(row["objective"], row["optimizer"]): row for row in bank}
    require(all(key[:2] in indexed for key in order), "missing fresh-state endpoint")
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    values = [float(indexed[key[:2]]["terminal_speed_max_mps"]) for key in order]
    ax.barh(
        range(len(order)),
        values,
        color=["#547786"] + [colors[0] if value <= 0.8 else colors[2] for value in values[1:]],
    )
    ax.set_yticks(range(len(order)), [key[2] for key in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.axvline(0.8, color="#333333", linestyle="--", linewidth=1.3)
    for index, value in enumerate(values):
        ax.text(value + 0.008, index, f"{value:.3f}", va="center")
    ax.set_xlabel("Maximum terminal speed (m/s); declared limit = 0.8")
    ax.set_title("Fresh state bank · effectiveness-only, learned library", pad=12)
    fig.savefig(figure_dir / "02_braking_recovery.png", dpi=180)
    fig.savefig(figure_dir / "02_braking_recovery.pdf")
    plt.close(fig)

    runtime = read_csv(output / "tables/runtime_source.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    labels = [f"{row['method']} · {row['mode']}" for row in runtime]
    y = list(range(len(runtime)))
    runtime_colors = [
        colors[0] if row["safe_task_completion"] == "True" else colors[2] for row in runtime
    ]
    axes[0].barh(y, [int(row["inflight_publications"]) for row in runtime], color=runtime_colors)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Published online updates during physical flight")
    axes[0].set_xlim(0, 290)
    for i, row in enumerate(runtime):
        axes[0].text(
            int(row["inflight_publications"]) + 3, i, row["inflight_publications"], va="center"
        )
    axes[1].barh(y, [float(row["controller_p95_ms"]) for row in runtime], color="#547786")
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].axvline(40, color="#333333", linestyle="--")
    axes[1].set_xlim(0, 44)
    axes[1].set_xlabel("Controller service p95 (ms); period = 40 ms")
    fig.suptitle("Eight measured runs · different execution modes include different command delays")
    axes[0].legend(
        handles=[
            Patch(color=colors[0], label="Safe task"),
            Patch(color=colors[2], label="Collision"),
        ],
        loc="upper right",
    )
    fig.savefig(figure_dir / "03_runtime_availability.png", dpi=180)
    fig.savefig(figure_dir / "03_runtime_availability.pdf")
    plt.close(fig)

    factorial_csv = output / "tables/matched_factorial_pairs.csv"
    if factorial_csv.exists():
        factorial = read_csv(factorial_csv)
        cells = sorted({row["cell"] for row in factorial})
        worlds = sorted({row["world"] for row in factorial})
        require(len(cells) == 11 and len(worlds) == 2, "incomplete factorial figure")
        categories = {
            "both_succeed": 0,
            "adaptation_alone_succeeds": 1,
            "frozen_alone_succeeds": 2,
            "both_fail": 3,
        }
        cmap = ListedColormap(["#326e5c", "#547786", "#d5a247", "#b24c49"])
        lookup = {(row["world"], row["cell"]): row for row in factorial}
        fig, axes = plt.subplots(2, 1, figsize=(12, 5.3), layout="constrained")
        for ax, metric, title in zip(
            axes,
            ("task_outcome", "collision_free_outcome"),
            ("Safe task completion", "Collision-free physical prefix"),
            strict=True,
        ):
            matrix = [
                [categories[lookup[(world, cell)][metric]] for cell in cells] for world in worlds
            ]
            ax.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
            ax.set_yticks(range(2), [world.removeprefix("structured_") for world in worlds])
            ax.set_xticks(
                range(11),
                [
                    cell.replace("eta", "η").replace("_lag", "\nlag ").replace("_extra", "\nlead ")
                    for cell in cells
                ],
            )
            ax.set_xticks([i + 0.5 for i in range(10)], minor=True)
            ax.set_yticks([0.5], minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.set_title(title)
        fig.suptitle("Matched development factorial · continue learning versus freeze at fault")
        axes[1].legend(
            handles=[
                Patch(color=cmap(index), label=label)
                for index, label in enumerate(
                    ("Both succeed", "Adaptation alone", "Freeze alone", "Both fail")
                )
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.7),
            ncols=4,
        )
        fig.savefig(figure_dir / "04_matched_factorial.png", dpi=180, bbox_inches="tight")
        fig.savefig(figure_dir / "04_matched_factorial.pdf", bbox_inches="tight")
        plt.close(fig)


def included(path: Path, table_output: Path) -> bool:
    if path.is_relative_to(table_output):
        return True
    # Episode JSONL embeds full predicted arrays and can exceed 600 MiB per flight.
    # It is raw trajectory evidence even though its container is textual.
    if path.name == "events.jsonl":
        return False
    if path.suffix == ".jsonl":
        return path.name == "frame_audit.jsonl" or path.stat().st_size <= 1024 * 1024
    if "pd-loss-origin-measured-v1" in path.parts and (
        path.suffix == ".npz" or path.name.endswith(".hlo.txt")
    ):
        return True
    if path.suffix in {".json", ".csv", ".md", ".py", ".toml", ".xml", ".log", ".sh"}:
        return True
    if path.suffix == ".npz" and (
        path.stem == "deployment"
        or path.name
        in {"control_excerpt.npz", "common_states.npz", "common-inputs.npz", "inputs.npz"}
    ):
        return True
    return path.name in {"SHA256SUMS", "pixi.lock"}


def add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(safe_member(name))
    info.size, info.mode, info.mtime = len(payload), 0o644, 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(payload))


def collect(base: Path, protocol: Path, table_output: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), "archive output must be new")
    table_manifest = read_json(table_output / "tables_manifest.json")
    proposal, records, _ = protocol_records(base, protocol)
    require(len(records) == len(proposal["trials"]), "original protocol is incomplete")
    require(
        all(r["summary"]["status"] == "completed" for r in records.values()),
        "unfinished original attempt",
    )
    require(table_manifest["unique_original_trials_completed"] == len(records), "tables are stale")
    secondary_proposal, secondary, _ = secondary_records(base)
    if secondary_proposal:
        require(len(secondary) == 40, "secondary trial set is not fully retained")
        require(
            table_manifest["secondary_union_refresh"]["retained"] == 40,
            "secondary tables are stale",
        )
    declared_evidence: dict[str, str] = {}

    def add_declared(record: dict[str, Any]) -> None:
        for key in ("evidence_sha256", "environment_evidence_sha256", "file_sha256"):
            for name, digest in record.get(key, {}).items():
                source = Path(name)
                if not source.is_absolute():
                    source = ROOT / safe_member(name)
                name = relative(source.resolve())
                require(
                    name not in declared_evidence or declared_evidence[name] == digest,
                    f"conflicting evidence digest: {name}",
                )
                declared_evidence[name] = digest

    for record in records.values():
        add_declared(record)
    for record in secondary.values():
        add_declared(record)
    for report_path, expected_count in (
        (base / "runtime-measured-v2-audit-v1/report.json", 8),
        (base / "incumbent-refresh-results-v3/results.json", 8),
    ):
        extra_records = read_json(report_path)["records"]
        require(
            len(extra_records) == expected_count, f"incomplete supplementary set: {report_path}"
        )
        for record in extra_records:
            add_declared(record)
    episode_evidence_count = len(declared_evidence)
    for measured_audit in (
        base / "controller-cost-measured-v1-audit-v1/report.json",
        base / "pd-loss-origin-audit-v1/report.json",
    ):
        add_declared(read_json(measured_audit))
    sources = {
        path
        for top in (base, protocol.parent)
        for path in top.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    sources.update(
        {
            ROOT / "docs/da_plcbf_actuator_diagnostics_report.md",
            ROOT / "docs/da_plcbf_actuator_report.md",
            ROOT / "crazyflow/safety/da_plcbf/actuator_async.py",
            ROOT / "benchmark/da_plcbf_actuator_publication.py",
            ROOT / "benchmark/da_plcbf_actuator_incumbent_refresh.py",
            ROOT / "tests/test_da_plcbf_actuator_incumbent_refresh.py",
            ROOT / "benchmark/da_plcbf_actuator_controller_cost.py",
            ROOT / "tests/test_da_plcbf_actuator_controller_cost.py",
            ROOT / "benchmark/da_plcbf_actuator_pd_loss_origin.py",
            ROOT / "tests/test_da_plcbf_actuator_pd_loss_origin.py",
            ROOT / "benchmark/da_plcbf_union_refresh_secondary.py",
            ROOT / "tests/test_da_plcbf_union_refresh_secondary.py",
            ROOT / "benchmark/da_plcbf_union_refresh_secondary_analysis.py",
            ROOT / "tests/test_da_plcbf_union_refresh_secondary_analysis.py",
            Path(__file__).resolve(),
        }
    )
    sources.update(ROOT.glob("benchmark/da_plcbf_actuator_diagnostic*.py"))
    sources.update(ROOT.glob("tests/test_da_plcbf_actuator_diagnostic*.py"))
    # Include exact historical initial checkpoints referenced by the new runs.
    for record in records.values():
        metadata = record["method_metadata"]
        prefix = Path(metadata["current_checkpoint"])
        sources.update({prefix.with_suffix(".json"), prefix.with_suffix(".npz")})
        core = metadata.get("immutable_core")
        if core:
            prefix = Path(core["checkpoint"])
            sources.update({prefix.with_suffix(".json"), prefix.with_suffix(".npz")})
    for path in sources:
        require(path.is_relative_to(ROOT), f"source outside repository: {path}")
        require(not path.is_relative_to(output), "archive cannot include itself")
    inventory = []
    for path in sorted(sources):
        digest, size, _ = hash_file(path)
        inventory.append(
            {
                "path": relative(path),
                "sha256": digest,
                "bytes": size,
                "included": included(path, table_output),
            }
        )
    actual_hashes = {row["path"]: row["sha256"] for row in inventory}
    require(
        actual_hashes[relative(Path(__file__).resolve())] == table_manifest["generator_sha256"],
        "tables were produced by a different generator revision",
    )
    for name, digest in table_manifest["sources"].items():
        require(actual_hashes.get(name) == digest, f"table source changed: {name}")
    for name, expected_digest in declared_evidence.items():
        require(name in actual_hashes, f"declared evidence omitted from inventory: {name}")
        require(actual_hashes[name] == expected_digest, f"declared evidence mismatch: {name}")
    output.mkdir(parents=True)
    inventory_payload = json_bytes({"schema": SCHEMA, "files": inventory})
    (output / "inventory.json").write_bytes(inventory_payload)
    readme = (
        "# Diagnostic review archive\n\n"
        "Exact source metadata, protocols and amendments, initial checkpoints, evidence tables "
        "and figures are included. Large raw rollouts/event streams, learner arrays and movies "
        "remain local and are explicitly hashed in inventory.json. This is a review archive, "
        "not a self-contained full numerical replay bundle. All original protocol trial IDs "
        "must have a completed physical attempt before collection. Outcomes and validity are "
        "separate: completed failures remain failures.\n\n"
        "Archive members retain repository-relative paths. Verify without extraction using "
        "`python -m benchmark.da_plcbf_actuator_diagnostic_evidence verify "
        "--output <this-directory>`. `--verify-local` additionally checks the current raw files "
        "against the complete inventory.\n"
    ).encode()
    (output / "README.md").write_bytes(readme)
    checksums = "".join(
        f"{row['sha256']}  {row['path']}\n" for row in inventory if row["included"]
    ).encode()
    (output / "SHA256SUMS").write_bytes(checksums)
    with (output / ARCHIVE).open("wb") as raw:
        with gzip.GzipFile(
            fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=6
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for row in inventory:
                    if row["included"]:
                        payload = (ROOT / row["path"]).read_bytes()
                        require(
                            hashlib.sha256(payload).hexdigest() == row["sha256"],
                            f"source changed: {row['path']}",
                        )
                        add_tar_bytes(archive, row["path"], payload)
                add_tar_bytes(archive, "__publication__/inventory.json", inventory_payload)
                add_tar_bytes(archive, "__publication__/README.md", readme)
                add_tar_bytes(archive, "__publication__/SHA256SUMS", checksums)
    digest, size, _ = hash_file(output / ARCHIVE)
    manifest = {
        "schema": SCHEMA,
        "archive": {"path": ARCHIVE, "sha256": digest, "bytes": size},
        "inventory_sha256": hashlib.sha256(inventory_payload).hexdigest(),
        "included_files": sum(row["included"] for row in inventory),
        "omitted_files": sum(not row["included"] for row in inventory),
        "included_bytes": sum(row["bytes"] for row in inventory if row["included"]),
        "omitted_bytes": sum(row["bytes"] for row in inventory if not row["included"]),
        "original_protocol_unique_completed": len(records),
        "secondary_union_refresh_retained": len(secondary),
        "secondary_union_refresh_completed": sum(
            (record.get("summary") or {}).get("status") == "completed"
            for record in secondary.values()
        ),
        "declared_episode_evidence_files_authenticated": episode_evidence_count,
        "declared_episode_and_measured_evidence_files_authenticated": len(declared_evidence),
        "scope": (
            "P0 diagnostic flights; descriptive evidence; "
            "no recursive safety or identification claim"
        ),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def verify(output: Path, verify_local: bool) -> dict[str, Any]:
    manifest = read_json(output / "manifest.json")
    require(manifest["schema"] == SCHEMA, "unknown manifest schema")
    require(manifest["archive"]["path"] == ARCHIVE, "unexpected archive path")
    inventory_payload = (output / "inventory.json").read_bytes()
    require(
        hashlib.sha256(inventory_payload).hexdigest() == manifest["inventory_sha256"],
        "inventory digest mismatch",
    )
    inventory = json.loads(inventory_payload)["files"]
    expected = {safe_member(row["path"]): row for row in inventory if row["included"]}
    require(
        len(expected) == manifest["included_files"], "duplicate or missing included inventory path"
    )
    digest, size, _ = hash_file(output / manifest["archive"]["path"])
    require(
        digest == manifest["archive"]["sha256"] and size == manifest["archive"]["bytes"],
        "archive mismatch",
    )
    seen = set()
    generated = {
        "__publication__/inventory.json": inventory_payload,
        "__publication__/README.md": (output / "README.md").read_bytes(),
        "__publication__/SHA256SUMS": (output / "SHA256SUMS").read_bytes(),
    }
    with tarfile.open(output / manifest["archive"]["path"], "r:gz") as archive:
        for member in archive:
            name = safe_member(member.name)
            require(
                member.isfile() and name not in seen, f"unsafe or duplicate archive member: {name}"
            )
            seen.add(name)
            stream = archive.extractfile(member)
            require(stream is not None, f"unreadable archive member: {name}")
            payload = stream.read()
            if name in generated:
                require(payload == generated[name], f"generated member mismatch: {name}")
            else:
                require(name in expected, f"unexpected archive member: {name}")
                row = expected[name]
                require(
                    len(payload) == row["bytes"]
                    and hashlib.sha256(payload).hexdigest() == row["sha256"],
                    f"member mismatch: {name}",
                )
    require(seen == set(expected) | set(generated), "archive membership is incomplete")
    if verify_local:
        for row in inventory:
            digest, size, _ = hash_file(ROOT / safe_member(row["path"]))
            require(
                digest == row["sha256"] and size == row["bytes"],
                f"local source mismatch: {row['path']}",
            )
    result = {
        "schema": SCHEMA,
        "archive_verified": True,
        "archive_members": len(seen),
        "local_inventory_verified": verify_local,
    }
    write_json(output / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("tables", "collect", "verify"))
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--tables", type=Path, default=BASE / "report-evidence-v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-local", action="store_true")
    args = parser.parse_args()
    base, protocol, table_output = (
        args.base.resolve(),
        args.protocol.resolve(),
        args.tables.resolve(),
    )
    if args.command == "tables":
        result = tables(base, protocol, table_output)
        figures(table_output)
    else:
        require(args.output is not None, "--output is required for collect/verify")
        output = args.output.resolve()
        result = (
            collect(base, protocol, table_output, output)
            if args.command == "collect"
            else verify(output, args.verify_local)
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
