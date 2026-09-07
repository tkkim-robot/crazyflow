"""Bind the selected hover replay, common-history ablation and mechanism to source files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmark.da_plcbf_hover_followup import verify_common_history
from benchmark.da_plcbf_hover_return import BASE, METHODS
from benchmark.da_plcbf_recovery_diagnosis import sha, write
from crazyflow.safety.da_plcbf.actuator_experiment import _hash_tree
from crazyflow.safety.da_plcbf.actuator_learning import load_actuator_learner_checkpoint


def main() -> None:
    output = BASE / "analysis-v2"
    output.mkdir(parents=True, exist_ok=False)
    source = BASE / "direction-refinement-v1/case-003"
    selected = BASE / "selected-replay-v2"
    initial = [
        load_actuator_learner_checkpoint(selected / m / "attempt-00/snapshots/initial")
        for m in ("F2", "A_BAL")
    ]
    assert _hash_tree(initial[0].state) == _hash_tree(initial[1].state)
    initial_learner_sha256 = _hash_tree(initial[0].state)
    bindings = [json.loads((selected / m / "attempt-00/binding.json").read_text()) for m in METHODS]
    for binding in bindings:
        assert binding["checkpoint"]["runtime_actor_config"]["wind_feedforward"] is False
        assert binding["config"]["command_governor"] == "committed_backup"
        for key in ("filter_config", "observation_config", "hover_task"):
            assert binding["config"][key] == bindings[0]["config"][key]
        assert binding["scene"]["physical_world_id"] == bindings[0]["scene"]["physical_world_id"]
    rows = []
    for method in METHODS:
        a, b = (root / method / "attempt-00" for root in (source, selected))
        with np.load(a / "dense.npz") as x, np.load(b / "dense.npz") as y:
            assert set(x.files) == set(y.files)
            for k in x.files:
                np.testing.assert_array_equal(x[k], y[k], err_msg=k)
        with np.load(a / "controls.npz") as x, np.load(b / "controls.npz") as y:
            keys = [
                "time",
                "controller_input_state",
                "goal",
                "planned_command",
                "command_applied",
                "control_params_sha256",
                "control_library_version",
                "estimated_model_sha256",
                "selected_index",
                "eligible",
                "qp_valid",
                "backup_backup_params_sha256",
                "backup_backup_anchor",
                "backup_backup_started_at",
                "backup_backup_certified_until",
                "backup_checked_trajectory",
            ]
            for k in keys:
                np.testing.assert_array_equal(x[k], y[k], err_msg=k)
            controls = {
                k: y[k] for k in y.files if k not in ("candidate_states", "candidate_commands")
            }
        summary = json.loads((b / "summary.json").read_text())
        backup_hash = controls["backup_backup_params_sha256"]
        versions_by_hash = dict(
            zip(controls["control_params_sha256"], controls["control_library_version"], strict=True)
        )
        used = controls["backup_executed_backup"].astype(bool)
        initial = int(summary["initial_library_version"])
        executed_learned = used & np.array(
            [h in versions_by_hash and versions_by_hash[h] > initial for h in backup_hash]
        )
        checkpoints = {}
        for t in [4.0, 5.2, 5.4, 5.6, 5.8, 6.0, 20.0, 21.8, 22.0]:
            inds = np.flatnonzero(np.isclose(controls["time"], t, atol=1e-10, rtol=0))
            if not len(inds):
                continue
            i = int(inds[0])
            h = backup_hash[i]
            checkpoints[str(t)] = dict(
                current_updates=int(controls["control_library_version"][i]) - initial,
                backup_online_updates=None
                if h not in versions_by_hash
                else int(versions_by_hash[h]) - initial,
                backup_params_sha256=str(h),
                executed_backup=bool(used[i]),
                selected_policy_dual=float(controls["executed_policy_dual"][i]),
                remaining_checked_seconds=float(controls["backup_remaining_checked_seconds"][i]),
            )
        rows.append(
            dict(
                method=method,
                summary=summary,
                collision_audit=json.loads((b / "collision_audit.json").read_text()),
                exact_dense_replay=True,
                exact_control_fields=keys,
                executed_backup_controls=int(used.sum()),
                executed_learned_backup_controls=int(executed_learned.sum()),
                controls_without_checked_plan=int(
                    (~controls["backup_has_checked_plan"].astype(bool)).sum()
                ),
                backup_checkpoints=checkpoints,
                sources={
                    str(p): sha(p)
                    for p in [
                        a / "dense.npz",
                        b / "dense.npz",
                        a / "controls.npz",
                        b / "controls.npz",
                        b / "binding.json",
                    ]
                },
            )
        )
    freeze = BASE / "selected-freeze2-v1/A_BAL/attempt-00"
    common = verify_common_history(selected / "A_BAL/attempt-00", freeze, 2.0)
    write(
        output / "report.json",
        dict(
            records=rows,
            common_initial_full_learner_and_adam_sha256=initial_learner_sha256,
            common_history=common,
            freeze_summary=json.loads((freeze / "summary.json").read_text()),
            mechanism_report_sha256=sha(BASE / "selected-mechanism-v1/report.json"),
            decisions_report_sha256=sha(BASE / "selected-decisions-v3/report.json"),
            scope="Exact full replay in aligned simulated-time execution. Same-state probes "
            "and onset-freeze support this selected development case, not general robustness "
            "or real-time feasibility.",
        ),
    )
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"replays_verified": METHODS, "common_history": common}), flush=True)


if __name__ == "__main__":
    main()
