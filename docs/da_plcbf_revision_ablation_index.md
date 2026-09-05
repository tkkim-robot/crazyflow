# Revision ablation index: F1–F7 and measured runtime

This index links the findings in `DA_PLCBF_REVIEW_AND_NEXT_STEPS_00e89a7.md` to retained
experiments. A corrected implementation, better behavior loss, an offline certificate value,
a completed navigation task and measured online runtime are distinct results.

| Finding | Implemented comparison and retained evidence | Actual conclusion |
| --- | --- | --- |
| **F1: rollout/execution cadence mismatch** | Nominal, fallback and BPTT rollouts now use the executed zero-order hold (`dt=.02`, hold 2, `H=60`). [Held-policy regressions](../tests/unit/safety/da_plcbf/test_held_policy_execution.py) and the [144-cell same-state factorial](../artifacts/da_plcbf/navigation-revision-20260905/estimated-factorial/factorial.json) compare hold 1/2 explicitly. | Cadence consistency is repaired. It does not remove the parameter-induced failure: at recorded 4.16 s with the same estimate, initial/updated hard fallback values are `+.05234/−.05835` with hold 1 and `+.06063/−.05141` with hold 2. |
| **F2: compensation changes the actor at the event** | The [legacy compensation 2×2](../artifacts/da_plcbf/learning-revision-20260905/reference-ablation-seed7/legacy_compensation_2x2.json) holds parameters and rest/event states fixed while changing nominal/wind model and compensation off/on; sibling NPZ retains states and motors. New teacher, warmup, prefix and deployment use compensation throughout. | Even at zero wind, feedforward cancels velocity-dependent drag and changes behavior. The legacy moving-state mean terminal speed changes from about `.133` to `.233 m/s` when compensation is enabled. Compensation benefit cannot be credited to BPTT. |
| **F3: rest competence and incompatible moving-state targets** | [State-conditioned reference protocol](da_plcbf_state_conditioned_learning.md) replaces fixed absolute displacement targets with a same-state immutable nominal teacher. [Seed-7 competence](../artifacts/da_plcbf/learning-revision-20260905/reference-ablation-seed7/nominal_competency.json) and [three-seed ablation summary](../artifacts/da_plcbf/learning-revision-20260905/development_summary.json) retain rest and broad moving-state measurements. | The nominal checkpoint occupies 16/16 direction bins from rest but 2/16 at the fast event state. Reference-bank LR `.001` reduces bank trajectory RMSE to about `.098–.099 m` across seeds 7/9/11; braking trades off slightly. No universal moving-state competence or changed-model target attainability is established. |
| **F4: global interference from current-state updates** | [Seven parameter-group/cap variants](../artifacts/da_plcbf/learning-revision-20260905/stabilization-seed7/ablation_summary.json), [early same-state H/QP checks](../artifacts/da_plcbf/learning-revision-20260905/stabilization-coverage-seed7/coverage.json), and [six closed-loop branches](../artifacts/da_plcbf/learning-revision-20260905/stabilization-closed-loop/summary.json). All finite updates publish, with persistent Adam and no safety gate. | Shared-network updates dominate the first loss; freezing offsets does not solve it. A `.002` parameter-step cap retains oracle eligibility through sampled update 20, loses it by 80, and stays near frozen in the closed-loop estimated case. It limits deterioration without fixing the shared shell violation or demonstrating an advantage over frozen. |
| **F5: estimator lag versus parameter harm** | [Exact legacy replay audit](../artifacts/da_plcbf/navigation-revision-20260905/failure_diagnosis_audit.json), [same-state factorial](../artifacts/da_plcbf/navigation-revision-20260905/estimated-factorial/factorial.json), and [CPU rate/noise replay](../artifacts/da_plcbf/learning-revision-20260905/estimator-replay-sensitivity/summary.json). | Both effects matter. At 4.16 s the initial estimated-model H is positive while oracle H is negative; learned parameters worsen both. Noiseless estimator rates 1.2/2.4/6 settle within `.1 m/s` after 3.14/1.58/.64 s. Shared `.01 m/s` velocity noise prevents that sustained threshold by 8 s in all three noisy conditions. This replay does not establish a closed-loop benefit from a faster estimator. |
| **F6: held operational rejection** | The [recorded held-operational fixture](../artifacts/da_plcbf/navigation-revision-20260905/held-operational-fixture/held_operational_probe.json) compares old cadence, corrected cadence alone, and local predictive operational correction with the same nonlinear held postcheck. | Cadence alone still rejects the arena-upper constraint (`−.047247`). One predictive correction reaches about `−9.54e-7`, passes the unchanged tolerance/KKT checks and accepts the QP. This is a resolved local controller fixture, not a global sampled-data guarantee. |
| **F7: the model-aware actor already cancels much wind/payload change** | [Compensation/physics matrix](../artifacts/da_plcbf/learning-revision-20260905/reference-ablation-seed7/compensation_physics_matrix.json) includes nominal, wind and centered 25% payload, compensation off/on, exact states/actions and actuator/hover diagnostics. | With compensation, the two-state payload trajectory RMSE is only about `.00705 m`, versus `.12039 m` for wind; both have zero recorded saturation and positive hover authority. The fixed model-aware baseline already handles much of this change. These conditions do not prove that residual learning is necessary. |

The first seven learner variants were selected by a declared obstacle-free behavior rule. The
later mask/cap extensions and short closed-loop test are development diagnostics; they do not
retroactively change the already frozen held-out navigation campaign. Full commands, gradient
contributions, parameter bounds, checkpoints and observation assumptions are in the
[consolidated learner note](da_plcbf_state_conditioned_learning.md).

## Task success is not online runtime feasibility

The [runtime-feasibility audit](../artifacts/da_plcbf/navigation-revision-20260905/paced_runtime_audit_three_attempts.json)
checks original summary, trace, raw update flags and wall-clock timestamps without changing
the three saved runs. File SHA-256 values identify every audited input. All three runs complete the
eight-waypoint task with positive obstacle-shell clearance and no degraded controls, but none
demonstrates online-learning service:

| Retained paced run | Adaptive finite updates / publications / advanced versions used | Fixed / adaptive deadline misses | Online runtime conclusion |
| --- | ---: | ---: | --- |
| `paced-validation` | 0 / 0 / 0 | 13 / 1 | Failed; learner was never exercised online and deadlines were missed. Warmup service metadata was not saved. |
| `paced-validation2` | 0 / 0 / 0 | 4 / 1 | Failed; the declared budget again admitted no update and deadlines were missed. |
| `paced-validation3` | 0 / 0 / 0 | 0 / 0 | Startup timing repaired, but zero of 335 opportunities launch; online-learning runtime still fails. |

The second run records a warmup learner service estimate of `22.18 ms`, expanded by factor 1.25
to `27.72 ms`. Adaptive controller median service is `13.74 ms`; adding the declared `3 ms`
reserve gives `44.47 ms` before plant and telemetry work, exceeding the `40 ms` control period.
This is an illustrative serialized cost sum, not an inferred update launch. The first run has
no retained warmup estimate; one is not reconstructed from unrelated measurements.

[`runtime_feasibility.py`](../crazyflow/safety/da_plcbf/runtime_feasibility.py) now reports task
success and online runtime separately. Its minimum observed runtime criterion requires paced
execution, an active allowed opportunity, at least one finite completed update, an advanced
snapshot published after completion and actually used by a later controller, consistent
accounting, and zero active adaptive deadline misses. A separate paired criterion also rejects
fixed-method misses. A final snapshot never used by a controller does not count. Zero updates
cannot satisfy the criterion. Passing it would still promise no update frequency or hard
real-time operating-system behavior.

The second run's warmup stopwatch could include pending plant/estimator work. The final
implementation synchronizes those inputs and removes two first-use JAX telemetry slices.
The separately captured third run has zero deadline misses in both branches, confirming that
repair. Its 14.02 ms adaptive controller median plus 27.72 ms reserved learner service and 3 ms
reserve totals 44.75 ms before plant/telemetry, leaving no update launched. All three results
therefore remain failed online-learning runtime checks, with different measured timing outcomes.

Fourteen focused accounting/runner regressions passed; they cover zero updates, publication without use,
misses, deterministic-mode mislabeling, forbidden opportunities, early publication, and use of
an unavailable version, missing/tampered wall chronology and backward versions. The fake-clock
runner also verifies that the saved schedule identifies actual paced-boundary publication.
The retrospective audit independently recomputes misses from completion time versus scheduled
release plus the control period and checks nondecreasing used versions. Reproduce it with:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' PYTHONPATH=. pixi run -e gpu-tests python \
  benchmark/da_plcbf_runtime_feasibility_audit.py \
  artifacts/da_plcbf/navigation-revision-20260905/paced-validation/combined-seed100 \
  artifacts/da_plcbf/navigation-revision-20260905/paced-validation2/combined-seed100 \
  artifacts/da_plcbf/navigation-revision-20260905/paced-validation3/combined-seed100 \
  --output /tmp/paced_runtime_audit.json
```
