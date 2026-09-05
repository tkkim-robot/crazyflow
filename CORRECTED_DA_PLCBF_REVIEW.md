# Corrected DA-PLCBF review guide

> **Superseded checkpoint.** The current entry point is
> [DA_PLCBF_REVISION_REVIEW.md](DA_PLCBF_REVISION_REVIEW.md). This guide preserves the `938a3b2`
> mechanism review and its artifacts; its passing results are not validation of the subsequent
> isolation, learning, geometry, and renderer corrections.

## What this review covers

This is the review entry point for the corrected algorithmic target. It supersedes the older
candidate-admission, uncertainty-particle, robust-Cartesian, and large-campaign path documented as
historical material in `HANDOFF_DA_PLCBF.md` and `DA_PLCBF_PLAN.md`.

The corrected mechanism is:

```text
obstacle-agnostic latent skills
    -> persistent BPTT through one current point dynamics model
    -> every finite update becomes the next library version
    -> {nominal + latest fallback library} runtime rollouts
    -> current obstacle values and JAX policy-value gradient
    -> continuous direct-wrench PL-CBF minimum-intervention QP
    -> QP action, with selected fallback only when the QP is invalid
```

This is an engineering mechanism review. It is not a claim-grade experiment, a hardware result,
or a robust safety guarantee under model-estimation error.

## Recommended review order

### 1. Watch the passed static gate

- video:
  `artifacts/da_plcbf/corrected-static-review-20260901/static_nominal_collision_vs_fixed_plcbf_avoidance.mp4`
- metrics:
  `artifacts/da_plcbf/corrected-static-review-20260901/static_nominal_collision_vs_fixed_plcbf_avoidance_metrics.json`
- entry point: `examples/da_plcbf/static_blocking_obstacle_demo.py`

The left vehicle executes the nominal controller and collides with the deliberately blocking
sphere. The right vehicle starts from the same state and goal but executes the continuous
fixed-library PL-CBF output, visibly detours around the inflated shell, and resumes the goal.

Verified static metrics:

| Quantity | Result |
|---|---:|
| Control interval / horizon | `0.02 s` (50 Hz) / `H=60` |
| Physical / inflated radius | `0.480000 m` / `0.630000 m` |
| Nominal minimum center distance | `0.021334 m` (collision) |
| Filtered minimum center distance | `0.811760 m` |
| Filtered shell margin | `0.181760 m` |
| Maximum QP intervention norm | `0.207791` |
| Samples with intervention above `1e-3` | `84` |
| Direct selected-fallback execution samples | `20` |
| Degraded samples | `0` |
| Final filtered goal error | `0.041914 m` |

### 2. Watch the constant-wind comparison

- video:
  `artifacts/da_plcbf/corrected-online-wind-review-20260901-v5/constant_wind_fixed_vs_continuously_adaptive_da_plcbf.mp4`;
- numerical trace:
  `artifacts/da_plcbf/corrected-online-wind-review-20260901-v5/online_constant_wind.npz`;
- summary/metrics:
  `artifacts/da_plcbf/corrected-online-wind-review-20260901-v5/online_constant_wind.json`;
- run identity: final corrected v5 on `cuda:0`, 600 control samples at `dt=0.02 s`.

The numerical run passed every corrected mechanism gate. The synchronized left/right comparison
has the frozen and adaptive methods exactly identical before the single `t=4 s` wind step. Both use
the same continuously updated point wind estimate. Only the right library receives persistent BPTT
updates. Review the changing library version, descriptor inset, safe fallback count, selected
certificate, executed history, and QP intervention.

The encoded 1600x900, 20 fps, 12.05 s MP4 was visually inspected at the initial state, the 4 s
wind transition, the learning interval, the 7.8 s coverage-separation encounter, obstacle
clearance, and terminal goal recovery. The scene, HUD, rollout colors, descriptor insets, and
fixed/adaptive synchronization are legible and agree with the saved trace.

| Required final-wind evidence | Final value |
|---|---:|
| All corrected mechanism gates | `true`; all individual checks `true` |
| True wind and number of changes | `[0.9, 0.55, 0.0] m/s`; exactly `1` change |
| Maximum pre-wind full-state difference | `0.0` |
| Wind detection time / final estimator error | `4.04 s` / `6.703e-7 m/s` |
| Finite gradient steps / final library version | `398` / `v398` |
| Adaptive parameter delta norm | `3.196885` |
| Fixed vs adaptive shared-probe target loss | `0.119108` / `0.095936` |
| Fixed vs adaptive shared-probe pairwise spread | `0.532425` / `0.659458` |
| Maximum adaptive safe-count advantage | `3` common-state; `4` at actual encounter |
| Fixed / adaptive minimum inflated clearance | `0.180672 m` / `0.206096 m` |
| Adaptive maximum QP intervention | `0.211479` |
| Fixed / adaptive degraded samples | `0` / `0` |
| Adaptive final goal distance | `0.193219 m` |
| Warm controller median/p95 per method | `24.321 ms` / `24.517 ms` |
| Warm BPTT update median/p95 | `12.394 ms` / `12.512 ms` |

These values come only from the corrected v5 point-estimate/persistent-learner trace, not the
historical four-condition campaign. The frozen library's shared-probe target loss worsened from its
zero-wind value of `0.083920` to `0.119108`; adaptation reduced it to `0.095936` and increased
pairwise spread from the frozen `0.532425` to `0.659458`. At the actual obstacle encounter, the
adaptive library had as many as four additional safe fallback rollouts.

The measured controller plus one BPTT update is about `36.7 ms` at the medians and `37.0 ms` using
the two reported p95 values. That is compatible with a 50 ms (20 Hz) soft loop on this machine, but
not with the configured 20 ms (50 Hz) period and not a hard-real-time guarantee. An asynchronous
learner could overlap these operations; this deterministic evidence run executed them sequentially.

### 3. Inspect the obstacle-agnostic learner

Read `crazyflow/safety/da_plcbf/persistent_skill_learner.py`:

- `obstacle_agnostic_skill_actions` accepts full state, skill-start position, phase, and latent
  specification. It has no goal, waypoint, obstacle, margin, or safety input.
- `rollout_skill_library` differentiates through the full 13-state direct-wrench dynamics.
- `obstacle_agnostic_skill_loss` uses final displacement, mean velocity, and terminal velocity
  descriptors with fixed directional targets, log-determinant diversity, pairwise repulsion,
  effort/rate, actuator-saturation, and trust regularization.
- `PersistentLearnerState` owns parameters and AdamW history for the episode.
- one finite step increments both `cumulative_gradient_steps` and `library_version`; NaN/Inf keeps
  the last finite parameters and optimizer state. There is no policy-safety admission decision.

### 4. Inspect the point estimator and online integration

Read these files in order:

1. `crazyflow/safety/da_plcbf/point_wind_estimator.py`;
2. `crazyflow/safety/da_plcbf/continuous_demo_scenarios.py`;
3. `crazyflow/safety/da_plcbf/online_constant_wind.py`;
4. `examples/da_plcbf/online_constant_wind_demo.py`.

The estimator infers one wind vector from the measured state transition, applied wrench, and known
mass/drag model, then low-pass filters it. The true wind is supplied only to the simulated plant.
The same current estimated model is passed to BPTT, nominal/fallback rollouts, value gradient, and
QP construction. The frozen and adaptive methods share this estimate; only adaptive parameters
change.

### 5. Inspect the continuous PL-CBF path

Read `crazyflow/safety/da_plcbf/continuous_version_a.py`, followed by the reused QP/postcheck core in
`crazyflow/safety/da_plcbf/version_a_filter.py`.

Check these mechanics:

- nominal is explicit augmented candidate zero;
- fallback callbacks receive no obstacle data;
- static and moving obstacle predictions enter only `runtime_policy_values`;
- swept relative segments, not only integration nodes, contribute to the hard value;
- `jax.jacfwd` differentiates every candidate value with respect to the current 13-state vector;
- the selected value supplies the PL-CBF halfspace;
- the QP includes coupled per-motor wrench feasibility and is postchecked;
- the QP command executes when valid; direct selected-fallback execution occurs only on invalid QP;
- no valid QP/fallback becomes an explicit degraded midpoint result.

### 6. Inspect focused tests only

The corrected tests are intentionally small:

- `tests/unit/safety/da_plcbf/test_continuous_version_a.py`;
- `tests/unit/safety/da_plcbf/test_persistent_skill_learner.py`;
- `tests/unit/safety/da_plcbf/test_online_constant_wind.py`;
- `tests/unit/safety/da_plcbf/test_mujoco_comparison_video.py`.

They cover the blocking static gate, obstacle-free public actor input, persistent optimizer/version
accumulation, nonfinite-only skip, one telemetry-derived wind transition, final GPU integration,
and actual MuJoCo frame generation. A new broad repository/campaign run is deliberately outside
this review.

### 7. Inspect the renderer last

`crazyflow/safety/da_plcbf/mujoco_comparison_video.py` consumes a completed immutable trace. It uses
two synchronized Crazyflow MuJoCo worlds and the actual quadrotor mesh. The scene includes physical
obstacles, inflated shells, goal, true/estimated wind, nominal/fallback/selected rollouts, executed
history, intervention, descriptor inset, and continuous learner telemetry. It displays:

> The selected fallback defines the safety certificate; the QP command is executed.

The selected rollout is the certificate trajectory; it is not presented as the executed QP
trajectory.

## Acceptance checklist

- [x] Intentionally blocking nominal static case collides.
- [x] Fixed continuous PL-CBF avoids the inflated shell with nonzero intervention and reaches goal.
- [x] Actor public input and skill loss are obstacle-agnostic.
- [x] Persistent optimizer state, finite-only update guard, and monotone library version exist.
- [x] Runtime uses augmented nominal/fallback rollouts and one point model.
- [x] Actual-MuJoCo static review video is rendered and visually inspected.
- [x] Final wind run changes wind exactly once and preserves exact pre-wind equality.
- [x] All 398 attempted BPTT updates are finite and publish monotonically through `v398`.
- [x] Shared-probe descriptor/spread and common/encounter safe coverage improve adaptively.
- [x] Adaptive run avoids the inflated shell, intervenes, has zero degradation, and reaches goal.
- [x] Final synchronized wind MP4 is encoded and visually inspected.

## Conditional safety boundary

The continuous filter certifies against the current estimated model and the obstacle prediction
available to the controller. It does not take a robust minimum over dynamics uncertainty. During
estimator convergence, the true plant can differ from that model. Therefore, a positive policy
value and accepted QP support only the reported estimated-model, finite-horizon condition plus the
implemented actuator and held-interval postchecks.

The review does not establish robustness to estimation error, unmodeled aerodynamics, delayed or
incorrect obstacle tracking, active-minimum/selected-policy nonsmoothness, arbitrary future
disturbances, hardware effects, or safety beyond the finite horizon. One deterministic static run
and one constant-wind run demonstrate the intended mechanism; they do not establish statistical
superiority or a real-world safety guarantee.

## Historical material

The legacy admission/rejection, uncertainty particles, Version-B robust discrete filter, dynamic
adversaries, seven baselines, campaign producers, evidence schemas, sealed artifacts, and
claim-grade schedules remain in the repository. They may be revisited later, but they are bypassed
by this corrected review and are not evidence for the corrected v5 results above.
