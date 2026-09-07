"""Prespecified wind-only hover/avoid/return development experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from benchmark.da_plcbf_actuator_diagnostics import DiagnosticResources
from benchmark.da_plcbf_recovery_diagnosis import begin, write
from benchmark.da_plcbf_wind_learning_comparison import BASE as CHECKPOINTS
from crazyflow.safety.da_plcbf.actuator_experiment import (
    ActuatorEpisodeConfig,
    run_actuator_episode,
)
from crazyflow.safety.da_plcbf.actuator_hover import HoverReturnTask
from crazyflow.safety.da_plcbf.actuator_plcbf import ActuatorFilterConfig
from crazyflow.safety.da_plcbf.actuator_study import ActuatorScene
from crazyflow.safety.da_plcbf.navigation_world import (
    NavigationWorld,
    NavigationWorldConfig,
    WindEvent,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/da_plcbf/hover-return-20260907/v1"
METHODS = ("PD_F", "F2", "A_BAL")


@dataclass(frozen=True, slots=True)
class HoverSceneSpec:
    """Exogenous crossing geometry and wind schedule, identical across methods."""

    seed: int = 71000
    obstacle_count: int = 3
    wind_speed: float = 1.8
    home_tolerance_m: float = 0.7
    wind_angle: float = 0.463647609
    reversal_angle: float = 2.617993878
    first_encounter: float = 6.0
    stagger: float = 1.2
    half_period: float = 16.0
    amplitude: float = 3.2
    radius: float = 0.45
    path_rotation: float = 0.0
    lateral_offset: float = 0.0
    height_offset: float = 0.0
    duration: float = 32.0
    wind_onset: float = 2.0
    reversal_time: float = 14.0
    calm_time: float = 25.2


def make_scene(spec: HoverSceneSpec) -> ActuatorScene:
    home = np.asarray([0.0, 0.0, 1.7])
    directions = np.asarray(
        [
            [
                np.cos(spec.path_rotation + 2 * np.pi * i / max(spec.obstacle_count, 1)),
                np.sin(spec.path_rotation + 2 * np.pi * i / max(spec.obstacle_count, 1)),
                0.0,
            ]
            for i in range(spec.obstacle_count)
        ]
    ).reshape(-1, 3)
    lateral = np.column_stack((-directions[:, 1], directions[:, 0], directions[:, 2]))
    mean = home + spec.lateral_offset * lateral
    mean[:, 2] += spec.height_offset
    omega = np.full(spec.obstacle_count, np.pi / spec.half_period)
    crossings = spec.first_encounter + spec.stagger * np.arange(spec.obstacle_count)
    config = NavigationWorldConfig(
        seed=spec.seed,
        obstacle_count=spec.obstacle_count,
        waypoint_count=2,
        duration_seconds=spec.duration,
        wind_events=(
            WindEvent(
                spec.wind_onset,
                (
                    spec.wind_speed * np.cos(spec.wind_angle),
                    spec.wind_speed * np.sin(spec.wind_angle),
                    0.0,
                ),
            ),
            WindEvent(
                spec.reversal_time,
                (
                    spec.wind_speed * np.cos(spec.reversal_angle),
                    spec.wind_speed * np.sin(spec.reversal_angle),
                    0.0,
                ),
            ),
            WindEvent(spec.calm_time, (0.0, 0.0, 0.0)),
        ),
    )
    config.validate()
    initial = np.r_[home, [0, 0, 0, 1], np.zeros(6)]
    world = NavigationWorld(
        config,
        initial,
        np.tile(home, (2, 1)),
        mean,
        spec.amplitude * directions,
        omega,
        -omega * crossings,
        np.full(spec.obstacle_count, spec.radius),
    )
    return ActuatorScene(
        world, "hover_return", spec.seed, "wind_only", spec.wind_onset, (1.0,) * 4, (1.0,) * 4, 0.0
    )


def task_for(spec: HoverSceneSpec) -> HoverReturnTask:
    last = spec.first_encounter + spec.stagger * max(0, spec.obstacle_count - 1)
    first_start = last + 2.0
    first_end = spec.reversal_time - 0.4
    second_start = last + spec.half_period + 2.0
    windows = tuple(
        (a, b) for a, b in ((first_start, first_end), (second_start, spec.duration)) if b - a >= 0.8
    )
    task = HoverReturnTask(position_tolerance_m=spec.home_tolerance_m, return_windows=windows)
    task.validate(spec.duration)
    return task


def resources() -> DiagnosticResources:
    result = DiagnosticResources()
    for method in METHODS:
        result.paths[method] = CHECKPOINTS / "prepared" / method / "deployment"
    return result


def run_case(
    output: Path,
    spec: HoverSceneSpec,
    cache: DiagnosticResources,
    *,
    methods: tuple[str, ...] = METHODS,
    retain_rollouts: bool = False,
    freeze_learning_at: float | None = None,
    execution_mode: str = "deterministic",
) -> list[dict]:
    scene = make_scene(spec)
    task = task_for(spec)
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "scene-plan.json",
        {"spec": asdict(spec), "scene": scene.metadata(), "task": asdict(task), "methods": methods},
    )
    rows = []
    for method in methods:
        bundle, controller, learner, metadata = cache.resolve(method, ActuatorFilterConfig())
        assert bundle.config.wind_feedforward is False
        capture = sorted(
            set(
                t
                for t in (
                    spec.wind_onset - 0.04,
                    spec.wind_onset,
                    spec.wind_onset + 0.04,
                    spec.wind_onset + 1,
                    spec.wind_onset + 2,
                    spec.first_encounter,
                    spec.first_encounter + spec.half_period,
                    spec.first_encounter + spec.half_period + 2 * spec.stagger,
                    spec.reversal_time,
                    spec.reversal_time + 1,
                    spec.reversal_time + 2,
                    spec.calm_time,
                    spec.duration - 0.04,
                )
                if 0 <= t <= spec.duration
            )
        )
        config = ActuatorEpisodeConfig(
            method=method,
            command_governor="committed_backup",
            allow_wind=True,
            hover_task=task,
            retain_rollouts=retain_rollouts,
            save_checkpoints=True,
            capture_times=tuple(capture),
            freeze_learning_at=freeze_learning_at,
            execution_mode=execution_mode,
        )
        print(json.dumps({"starting": str(output), "method": method}), flush=True)
        result = run_actuator_episode(
            scene,
            bundle,
            config,
            output / method / "attempt-00",
            controllerfunctions=controller,
            learner_functions=learner,
        )
        summary = result.summary
        d, c = result.dense_traces, result.control_traces
        phase_errors = []
        for start, end in (
            (0, spec.wind_onset),
            (spec.wind_onset + 3, spec.reversal_time),
            (spec.reversal_time + 3, spec.calm_time),
            (spec.duration - 2, spec.duration),
        ):
            active = (d["time"] >= start) & (d["time"] <= end)
            error = np.linalg.norm(d["state"][active, :3] - scene.world.initial_state[:3], axis=1)
            phase_errors.append(
                {
                    "window": [start, end],
                    "mean_home_distance_m": float(np.mean(error)) if len(error) else None,
                    "max_home_distance_m": float(np.max(error)) if len(error) else None,
                }
            )
        row = {
            "method": method,
            "summary": summary,
            "metadata": metadata,
            "phase_home_error": phase_errors,
            "unchecked_controls": int(np.count_nonzero(~c["backup_has_checked_plan"]))
            if "backup_has_checked_plan" in c
            else None,
        }
        rows.append(row)
        write(output / "report.json", {"records": rows, "planned": len(methods)})
        print(
            json.dumps(
                {
                    "finished": str(output),
                    "method": method,
                    "termination": summary["termination"],
                    "safe_return": summary["successful_full_episode"],
                    "hover": summary["hover_return"],
                    "phase_home_error": phase_errors,
                }
            ),
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibration", "pilot"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or BASE / f"{args.stage}-v1"
    begin(output, "Wind-only hover/avoid/return; no change to governor or learner objective.")
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    (output / "hover_task_source.py").write_bytes(
        (ROOT / "crazyflow/safety/da_plcbf/actuator_hover.py").read_bytes()
    )
    cache = resources()
    if args.stage == "calibration":
        specs = [
            HoverSceneSpec(seed=71000 + i, obstacle_count=0, wind_speed=w)
            for i, w in enumerate((1.4, 1.8, 2.2))
        ]
    else:
        specs = [
            HoverSceneSpec(seed=71100 + i, wind_speed=w) for i, w in enumerate((1.4, 1.8, 2.2))
        ]
    write(
        output / "plan.json",
        {
            "stage": args.stage,
            "specs": [asdict(s) for s in specs],
            "methods": METHODS,
            "selection": "All complete triplets retained",
        },
    )
    for index, spec in enumerate(specs):
        run_case(output / f"case-{index:03d}", spec, cache)


if __name__ == "__main__":
    main()
