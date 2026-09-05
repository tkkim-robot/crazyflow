# Failure diagnosis of 00e89a7

The original evidence is retained unchanged under `competent-revision-20260904`. This directory adds deterministic reconstructions, complete learner/Adam snapshots around the first failures, full rollout factorials, and a tested controller repair. These are mechanism experiments: recorded update opportunities determine which updates run, independently of wall-clock jitter.

## Confirmed mechanisms

1. **Prediction cadence differed from execution.** The old skill and nominal predictors updated every 20 ms while the plant held each command for 40 ms. Shared ZOH rollouts now retain all 60 integration steps and the 1.2 s horizon while updating policy feedback every two steps. A committed skill preserves its original anchor and advancing phase and reproduces the candidate trajectory. Receding-horizon restart and QP execution remain distinct operations.
2. **The compensation switch changed behavior independently of changed physics.** `../learning-revision-20260905/reference-ablation-seed7/legacy_compensation_2x2.json` contains the old/new plant × compensation off/on cells. Its companion NPZ retains full trajectories and motors at rest and at the recorded moving event state. With zero wind, enabling compensation changes mean terminal speed from 0.13862 to 0.21842 m/s at rest, and from 0.13260 to 0.23349 m/s at the moving state. The new principal checkpoints and branches keep compensation enabled throughout.
3. **Early parameter updates damaged fallback coverage at a fixed state and model.** All 31 archived adaptive commands and rejection-flag vectors in the estimated run's 4.0–5.2 s window reproduce exactly; maximum absolute applied-wrench error is zero. At its recorded 4.16 s state, the original frozen library has positive fallback coverage under the recorded estimate. The three-update snapshot already has negative coverage there; the fourth update reduces it further. Correcting cadence alone does not remove this loss.
4. **The point estimate is optimistic at that same state.** Supplying oracle wind makes even the original frozen library's value negative there. This is a separate model-information effect, interacting with physical state and learned parameters; the point model also enters the declared compensation mapping.
5. **The crossing run's 6.2 s emergency is a held arena-constraint failure.** The failing row is `arena_x_upper` at substep 1, 20 ms into the hold. All 17 candidates were collision-clear. Cadence correction alone leaves this rejection. A predictive operational QP refinement restores acceptance with unchanged physical limits and postcheck tolerances.

## Fixed-state factorial at 4.16 s

The table reports fallback-only maximum hard collision values in square metres. Every row uses the same recorded adaptive state, compensation enabled, and the same absolute-time obstacle prediction. The task nominal is stored separately and excluded from these maxima. All 144 cells—three parameter snapshots × two models × two states × two compensation settings × two prediction cadences, at three boundaries—and their full trajectories are in `estimated-factorial`.

| Parameters | Point model | Old 20 ms prediction | Corrected 40 ms prediction |
| --- | --- | ---: | ---: |
| Original frozen | Recorded estimate | +0.05234 | +0.06063 |
| After 3 updates | Recorded estimate | −0.03125 | −0.02394 |
| After 4 updates | Recorded estimate | −0.05835 | −0.05141 |
| Original frozen | Oracle | −0.04962 | −0.04390 |
| After 4 updates | Oracle | −0.14354 | −0.14154 |

The fourth update alone reduces the value by 0.0270955 m² under the same estimate, state and old cadence. The complete parameter change from the initial checkpoint reduces it by 0.1106890 m². These conditional effects do not define a unique additive causal decomposition. A favorable estimate-based value or a finite update does not establish actual recoverability.

The original-code replay and revised-code factorial are distinct measurements. The former records −0.05832916 m² for the current snapshot at 4.16 s; the latter's old-cadence cell records −0.05834925 m², a difference of approximately 0.00002009 m². The table and parameter contrasts consistently use the factorial's own cells, rather than treating this small numerical implementation difference as an update effect.

## Closed-loop branches from the estimated run's pre-failure state

All branches in the table start at 4.12 s from the saved estimated-run state and its complete version 503 learner/Adam snapshot, before that run's first recorded coverage loss at 4.16 s. Learning-enabled branches use the recorded opportunity mask. Estimated branches replay earlier observations to restore their estimator, then infer wind independently from their own transitions. Oracle branches receive the actual model. No post-collision state is used as their starting point.

| Model information | Learning after 4.12 s | Raw emergency commands | Minimum physical clearance | Minimum shell clearance |
| --- | --- | ---: | ---: | ---: |
| Oracle | Frozen | 11 | +0.07149 m | −0.07851 m |
| Oracle | Enabled | 12 | +0.00768 m | −0.14232 m |
| Independent estimator | Frozen | 14 | +0.02403 m | −0.12597 m |
| Independent estimator | Enabled | 14 | −0.00914 m | −0.15914 m |

The diagnostic window ends at 5.24 s after applying the 5.2 s command. Clearances use the saved dense 20 ms states and relative swept segments. The table's raw counts and minima retain the full diagnostic window. The colliding estimated-learning branch contains seven additional control commands after the first colliding control interval; seven of its 14 emergency commands occur through that collision and seven afterward. Those later commands receive no task-success credit. `branch_clearance_audit.json` records both raw and censored counts, first violating intervals, exact clearances and source checksums.

Freezing at 4.12 s avoids the observed physical collision within this window but does not restore the shell: three earlier updates, model error and physical evolution have already occurred. This does not establish that the frozen branch would remain safe afterward.

`legacy-oracle-replay-2` contains additional branches from the oracle run's 4.12 s state. Its oracle-frozen branch retains +0.00729 m shell clearance in the recorded window. That oracle starting state already has a negative certificate at 4.12 s, unlike the estimated-run branch point. These are secondary diagnostics, not oracle branches started before the first certificate loss.

## Tested held-command controller repair

At the unchanged crossing fixture, the boundary arena residual is approximately zero; at +20 ms it is −0.04724696. The repair linearizes later operational residuals through the actual held-command integrator and adds predictive QP faces. It preserves the original collision face, objective, motor bounds and instantaneous operational faces. The complete nonlinear held check still determines whether the resulting command may execute.

One refinement changes pitch torque to approximately −0.00030105 N·m and yields a minimum held residual of −0.00000095, within the existing 3e−6 tolerance. The minimum physical dimensionless margin remains positive at 0.0810624. Thirty synchronized GPU calls have a median of 15.152 ms, p95 of 15.503 ms and maximum of 16.408 ms. This measures the repaired-controller branch, not the complete online learner/control-period budget.

Longer holds require explicitly disabling this bounded predictive solver. The implementation supports at most two integration steps per hold to avoid unbounded active-set enumeration; nonlinear held postchecks remain enabled for longer holds.

## What the first learner revision does and does not fix

The reference learner uses immutable nominal-model maneuvers from each same proprioceptive initial state, trajectory and velocity targets at multiple times, terminal braking excess, bounded skill offsets, frozen durations and a rotating anchor bank. Every finite Adam update still publishes. The bank contains rest, signed horizontal/vertical velocities, attitude/rate samples and the moving event state. No goal or obstacle quantity enters sampling, targets or loss.

Across development learner seeds 7, 9 and 11, the reference-bank variant at learning-rate multiplier 1.0 achieves final obstacle-free trajectory-position RMSEs of 0.09834, 0.09851 and 0.09904 m, respectively. The corresponding original current-state variants at multiplier 1.0 achieve 0.11330, 0.11178 and 0.11424 m. These are behavior-restoration measurements.

However, the saved seed-7 coverage probe shows that all seven initial variants lose the single eligible certificate after four updates at the exact event-state/oracle fixture. The reference-bank revision therefore improves behavior restoration but **does not establish stable safety coverage**. Subsequent parameter-group and update-norm ablations are retained separately; they do not retroactively change the frozen navigation-trial configuration.

## Reproduction and retained failures

- `benchmark/da_plcbf_failure_replay.py` accepts `--source-tree` pointing to a git archive of `00e89a7`, the archived run directory, and a fresh output directory. It verifies the checkpoint hash, reconstructs recorded update opportunities, and re-evaluates the archived commands.
- `benchmark/da_plcbf_failure_factorial.py` loads those snapshots under the revised code.
- `benchmark/da_plcbf_held_operational_probe.py` reproduces and repairs the crossing fixture.
- `benchmark/da_plcbf_branch_clearance_audit.py` derives the branch-clearance audit from existing replay NPZs and archived scenarios. It runs no controller, optimizer or simulation.

`legacy-oracle-replay` is an incomplete development output: an estimator assertion incorrectly compared inferred lag with the oracle field, after successfully reconstructing snapshots. It is retained and superseded by `legacy-oracle-replay-2`. No original experiment output was overwritten.

Finite-horizon point-model values and sampled postchecks do not establish invariance across model, parameter, perception or task switches. These fixtures support the specific conditional diagnoses above; they do not establish adaptation superiority.
