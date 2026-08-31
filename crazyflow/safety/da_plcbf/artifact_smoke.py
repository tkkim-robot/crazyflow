"""Deterministic non-scientific vertical smoke for artifact writing and dashboard replay."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from crazyflow.safety.da_plcbf.artifacts import (
    ArtifactEvent,
    ImmutableTrace,
    aggregate_row,
    collect_provenance,
    derive_metrics,
    save_trace,
    validate_run_artifacts,
    write_aggregate_report,
    write_confidence_intervals,
    write_events,
    write_manifest,
    write_metrics,
    write_paired_metrics_csv,
    write_provenance,
    write_run_config,
    write_seeds,
    write_sha256sums,
    write_timing,
)
from crazyflow.safety.da_plcbf.dashboard import render_dashboard, video_manifest_record

if TYPE_CHECKING:
    import os


def synthetic_trace(
    scenario_tape_sha256: str, *, steps: int = 24, dt: float = 0.1
) -> ImmutableTrace:
    """Create a small immutable trace whose values are explicitly not experimental results."""
    if steps < 2 or not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("steps and dt must define at least two finite positive-time nodes")
    time = np.arange(steps, dtype=np.float64) * dt
    phase = np.linspace(0.0, 1.5 * math.pi, steps)
    position = np.stack(
        (0.7 * np.cos(phase), 0.55 * np.sin(phase), 1.2 + 0.1 * np.sin(0.5 * phase)), axis=1
    )
    velocity = np.gradient(position, dt, axis=0)
    true_state = np.concatenate((position, velocity), axis=1)
    estimated_state = true_state.copy()
    estimated_state[:, :3] += 0.015 * np.stack(
        (np.sin(phase), np.cos(phase), np.sin(2.0 * phase)), axis=1
    )
    nominal = np.stack(
        (
            0.32 + 0.02 * np.sin(phase),
            0.015 * np.cos(phase),
            0.012 * np.sin(phase),
            0.008 * np.cos(2.0 * phase),
        ),
        axis=1,
    )
    correction = np.stack(
        (
            0.008 * np.maximum(0.0, np.sin(phase)),
            -0.006 * np.sin(phase),
            0.004 * np.cos(phase),
            np.zeros(steps),
        ),
        axis=1,
    )
    filtered = nominal + correction
    hard_barriers = np.stack((0.22 + 0.05 * np.cos(phase), 0.31 + 0.04 * np.sin(phase)), axis=1)
    policy_values = np.stack(
        tuple(
            0.08 + 0.11 * np.cos(phase + offset)
            for offset in np.linspace(0, 2 * math.pi, 4, endpoint=False)
        ),
        axis=1,
    )
    selected = np.argmax(policy_values, axis=1).astype(np.int64)
    latency = np.stack(
        (
            0.0010 + 0.0001 * (1.0 + np.sin(phase)),
            0.0030 + 0.0002 * (1.0 + np.cos(phase)),
            0.0004 + 0.00005 * (1.0 + np.sin(2.0 * phase)),
        ),
        axis=1,
    )
    return ImmutableTrace(
        schema_version=np.asarray(2, dtype=np.int16),
        scenario_tape_sha256=np.asarray(scenario_tape_sha256),
        time=time,
        state_names=np.asarray(
            ("position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z")
        ),
        control_names=np.asarray(("collective_force", "torque_x", "torque_y", "torque_z")),
        barrier_names=np.asarray(("synthetic_clearance", "synthetic_arena")),
        policy_names=np.asarray(tuple(f"policy_{index}" for index in range(4))),
        loss_term_names=np.asarray(("coverage", "diversity", "trust")),
        latency_names=np.asarray(("rollout_forward", "bptt_step", "active_filter")),
        true_state=true_state,
        estimated_state=estimated_state,
        nominal_control=nominal,
        filtered_control=filtered,
        applied_control=filtered,
        executed_control=np.ones(steps, dtype=np.bool_),
        hard_barriers=hard_barriers,
        training_values=policy_values - 0.02,
        policy_values=policy_values,
        selected_policy=selected,
        snapshot_version=np.zeros(steps, dtype=np.int64),
        model_version=np.zeros(steps, dtype=np.int64),
        solver_kkt_residual=np.linspace(1e-12, 2e-12, steps),
        postcheck_residual=np.min(hard_barriers, axis=1),
        clipped=np.zeros(steps, dtype=np.bool_),
        saturated=np.zeros(steps, dtype=np.bool_),
        degraded=np.zeros(steps, dtype=np.bool_),
        contact=np.zeros(steps, dtype=np.bool_),
        failure=np.zeros(steps, dtype=np.bool_),
        loss_terms=np.stack(
            (0.3 - 0.1 * time / time[-1], 0.2 + 0.02 * np.sin(phase), 0.01 * time), axis=1
        ),
        gradient_norm=0.1 + 0.02 * np.cos(phase),
        component_latency_seconds=latency,
    )


def create_synthetic_smoke_run(
    output_parent: str | os.PathLike[str], *, run_id: str = "synthetic-artifact-smoke-v1"
) -> tuple[Path, dict[str, object]]:
    """Create and fully validate one deterministic synthetic artifact tree with a tiny MP4."""
    parent = Path(output_parent).resolve()
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", run_id):
        raise ValueError("run_id must be a portable slug")
    run = parent / run_id
    run.mkdir()
    for relative in (
        "aggregate",
        "methods/synthetic-reference/schema-smoke/0",
        "scenario_tapes",
        "videos",
    ):
        (run / relative).mkdir(parents=True)

    repository = Path(__file__).resolve().parents[3]
    provenance = collect_provenance(repository)
    tape_path = "scenario_tapes/0.npz"
    tape_digest, rng_stream_ids = _generate_synthetic_tape(run / tape_path)
    trace = synthetic_trace(tape_digest, steps=24, dt=0.1)
    method_directory = run / "methods" / "synthetic-reference" / "schema-smoke" / "0"
    save_trace(trace, method_directory / "trace.npz")
    events = (
        ArtifactEvent(0, 0, 0.0, "runtime", "start", "info", 0, 0, {"synthetic": True}),
        ArtifactEvent(
            1,
            trace.steps - 1,
            float(trace.time[-1]),
            "runtime",
            "complete",
            "info",
            0,
            0,
            {"synthetic": True},
        ),
    )
    write_events(events, method_directory / "events.jsonl", trace=trace)
    write_metrics(trace, method_directory / "metrics.json")
    write_timing(
        trace,
        method_directory / "timing.json",
        compile_seconds={name: 0.0 for name in trace.latency_names.tolist()},
        deadline_seconds={"rollout_forward": 0.002, "bptt_step": 0.005, "active_filter": 0.001},
    )

    config = {
        "schema_version": 1,
        "experiment_id": "synthetic-artifact-smoke",
        "description": "Synthetic schema, hash, metric, video, and replay validation only.",
        "control_dt_seconds": 0.1,
        "horizon_steps": 8,
        "paired_trials": True,
        "trials_per_condition": 1,
        "methods": ["synthetic-reference"],
        "conditions": ["schema-smoke"],
        "parameters": {"synthetic": True, "scientific_evidence": False},
    }
    write_run_config(config, run / "config.json")
    write_provenance(provenance, run / "provenance.json")
    write_seeds(
        {
            "schema_version": 1,
            "root_seed": 17,
            "folds": [0],
            "named_streams": rng_stream_ids,
            "scenario_tapes": [
                {
                    "condition": "schema-smoke",
                    "fold": 0,
                    "path": tape_path,
                    "content_sha256": tape_digest,
                }
            ],
            "pairing_id": "synthetic-smoke-pairing",
        },
        run / "seeds.json",
    )
    metrics = derive_metrics(trace)
    row = aggregate_row("synthetic-reference", "schema-smoke", 0, metrics)
    write_paired_metrics_csv((row,), run / "aggregate" / "paired_metrics.csv")
    write_confidence_intervals((row,), run / "aggregate" / "confidence_intervals.json")
    write_aggregate_report((row,), run / "aggregate" / "report.md", scientific_evidence=False)
    (run / "visual_review.md").write_text(
        "# Synthetic dashboard check\n\n"
        "Programmatic codec, dimensions, timing, frame-count, non-static-content, and replay "
        "checks passed. This smoke is not a human visual review and is not scientific evidence.\n",
        encoding="utf-8",
    )
    video_path = run / "videos" / "synthetic-dashboard.mp4"
    video = render_dashboard(trace, video_path, fps=10.0, size=(640, 360))
    source_trace = "methods/synthetic-reference/schema-smoke/0/trace.npz"
    video_record = video_manifest_record(video_path, source_trace, video, run_directory=run)
    write_manifest(
        run,
        run_id=run_id,
        status="synthetic-smoke",
        scientific_evidence=False,
        replay_command=(
            f"python examples/da_plcbf/artifact_smoke.py validate {run.as_posix()} --verify-replay"
        ),
        video_records=(video_record,),
        created_utc="2000-01-01T00:00:00Z",
    )
    write_sha256sums(run)
    result = validate_run_artifacts(run, verify_replay=True)
    return run, result


def _generate_synthetic_tape(path: Path) -> tuple[str, dict[str, int]]:
    script = """
import json
import sys
from crazyflow.safety.da_plcbf.scenarios import (
    RNG_STREAM_IDS,
    ScenarioTapeConfig,
    generate_scenario_tape,
    save_scenario_tape,
)
config = ScenarioTapeConfig(
    steps=24,
    dt=0.1,
    prediction_samples=2,
    static_capacity=1,
    static_count=0,
    dynamic_capacity=1,
    ballistic_count=1,
    crossing_count=0,
    pursuit_count=0,
    interceptor_count=0,
    random_attacker_count=0,
)
tape = generate_scenario_tape(17, config, fold=0)
save_scenario_tape(tape, sys.argv[1])
print(json.dumps({"sha256": tape.sha256, "rng_stream_ids": dict(RNG_STREAM_IDS)}, sort_keys=True))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120.0,
        )
        payload = json.loads(result.stdout.splitlines()[-1])
        digest = payload["sha256"]
        streams = payload["rng_stream_ids"]
    except (
        OSError,
        subprocess.SubprocessError,
        IndexError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError("isolated synthetic scenario-tape generation failed") from error
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("isolated scenario generator returned an invalid digest")
    if not isinstance(streams, dict) or not streams:
        raise ValueError("isolated scenario generator returned invalid RNG stream IDs")
    return digest, {str(name): int(value) for name, value in streams.items()}
