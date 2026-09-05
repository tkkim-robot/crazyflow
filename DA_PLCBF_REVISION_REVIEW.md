# DA-PLCBF independent revision review — 2026-09-04

This is the current review entry point. It supersedes the implementation conclusions and
acceptance results for commit `938a3b2` in the earlier handoff, review guide, and plan. Those
documents retain the prior checkpoints and reproduction history; their passing numbers do not
validate this revision. The numerical results below come from this revision's saved traces.
This revision is on `plcbf`, based on handoff commit `e4c4f39`. The implementation, review,
and complete revision evidence are committed together. Artifacts are explicitly tracked under
the otherwise ignored `artifacts/da_plcbf/revision-20260904/` directory, including retained
development trials, numerical traces, all four MP4s, and decoded review frames.

## Intended outcome

The project should learn a diverse, reusable fallback-policy library from a dynamics model,
adapt that library online when the dynamics change, and use it to keep a goal-seeking quadrotor
away from obstacles. The learner must not receive obstacle geometry, goals, safety values,
safe-policy counts, or an obstacle-avoidance reward. Obstacles enter only when the runtime
evaluates the predicted trajectories and constructs its safety filter.

One safe policy can certify a state. Consequently the relevant library value is
`H_library = max_i H_i` over the augmented nominal-plus-fallback library. Showing more safe
policies or a better descriptor loss alone does not establish that learning is necessary for
safety. The stronger experiment needs a shared state/model at which the frozen library loses
coverage and the learned library recovers it, followed by an honestly reported closed-loop
comparison that includes simple model compensation.

## Implemented mechanism and corrections

- **Persistent obstacle-free learning.** The latent actor uses state, displacement from the skill
  start, phase, and a structured skill identity. Its loss uses motion descriptors, diversity,
  control effort/rate, saturation, and parameter trust. BPTT propagates through the 13-state
  Version-A dynamics under one point model. AdamW history persists, and every finite update
  advances the library version; there is no safety-admission gate or rollback protocol.
- **Online timing.** The experiment supports learning after wind detection (`learning_start=wind`)
  or from the initial control boundary (`learning_start=startup`), with a configurable number of
  micro-updates. Once active, learning continues at control boundaries. Each controller call
  consumes an immutable completed parameter snapshot. Execution remains sequential; actual
  asynchronous overlap and hard real-time behavior are not established. The main demonstration
  sets `initial_skill_scale=0`: the directional scaffold starts at zero, leaving a stabilizing
  braking controller and a small random residual. Motion targets and latent identities are still
  supplied; this is online construction from an undiversified stabilizing seed, not learning
  stabilization, dynamics, and skills from nothing.
- **Scientific isolation.** Analytic obstacle HOCBF rows and the PL-CBF row have independent
  switches. The PL-CBF comparisons disable analytic obstacle HOCBF rows while retaining arena,
  altitude, speed, angular-rate, tilt, and actuator checks. An analytic-only baseline disables
  the PL-CBF row and does not execute a library fallback. The selected PL-CBF dual multiplier
  records whether that row contributes to the QP proposal; executed-QP attribution additionally
  requires the recorded QP-valid flag.
- **Consistent smooth QP value.** The QP uses the conservative unnormalised log-sum-exp lower
  bound of the obstacle node/swept constraints and the derivative of that same value. Exact hard
  node/relative-segment minima remain separate collision diagnostics, certificate checks, and
  held-interval postchecks. The bound satisfies `soft <= hard <= soft + temperature * log(N)`.
  This is a smooth surrogate, not the deferred cubic-spline implementation, and does not remove
  policy switches or all piecewise rollout nonsmoothness.
- **Physical geometry and scope.** Collision geometry includes obstacle radius, explicit drone
  radius, and clearance: `r_effective = r_obstacle + r_drone + clearance`. Finite-horizon policy
  values cover collision clearance only. Other operational limits are checked by the
  instantaneous filter; a collision-clear rollout is not a certificate for every state limit
  throughout the horizon.
- **Matched dynamics information.** Every simulated method has an identical estimator design
  driven by its own measured transition and executed command. Each obtains one point wind
  estimate, used consistently by its predictions and filter. The model-compensated frozen
  baseline adds explicit drag/wind feedforward to its obstacle-free fallback actor and performs
  no gradient updates. It tests whether known-model compensation explains an apparent learning
  advantage. The main construction experiment also enables identical model feedforward in the
  **nominal controller of every method**; only the compensated baseline adds it to frozen skills.
- **Actuator and descriptor consistency.** The revised online demonstration defaults to hard
  motor clipping, which preserves already-feasible requested forces and avoids the interior
  force bias of the previous `tanh` motor mapping. Clipping remains piecewise differentiable.
  Mean-velocity descriptors use the post-step velocities from symplectic Euler, so displacement
  equals horizon duration times mean velocity. Targets obey the same identity and specify zero
  terminal velocity.
- **Truthful execution and visualization.** The trace records maximum and selected hard H,
  collision-clear counts, QP validity, selected-row dual, overlapping rejection reasons,
  fallback execution, and degradation. An executable fallback without a horizon certificate is
  explicitly degraded best effort; it is not labeled safe. Videos show actual command status,
  persistent executed paths, collision history, and certificate histories. They are labeled
  **MuJoCo-rendered replays of the differentiable Version-A simulation**. Contacts and full
  rotor/controller-stack dynamics are not being simulated by the renderer.
- **Exact QP acceleration.** Disabled obstacle rows are omitted from active-set enumeration.
  A closed-form projection onto the selected PL-CBF face is used only when it satisfies every
  actuator/operational constraint and the KKT checks. Otherwise the complete active-set solver
  runs. The shortcut changes computation, not the optimization problem. The derivative pass also
  reuses its primal rollouts.

## Comparisons and interpretation

| Method | Obstacle HOCBF | PL-CBF row | Fallback updates |
|---|---|---|---|
| Nominal controller, static gate | No | No | None |
| `analytic` | Yes | No | None |
| `fixed` | No | Yes | None |
| `compensated` | No | Yes | Explicit point-model compensation; frozen parameters |
| `adaptive` | No | Yes | Persistent BPTT |

Shared-state probes separate changes in the library from differences in the executed states.
Closed-loop trajectories answer the different question of what each controller actually does.
The wind-triggered fixed/adaptive runs should agree before adaptation starts; startup-learning
runs need not. Failed trials and counterexamples must remain visible alongside selected examples.

The centered static pilot safely stalled in front of the sphere rather than reaching its goal.
Its failed completion check is retained in
[`static-pilot/static_nominal_collision_vs_fixed_plcbf_avoidance_metrics.json`](artifacts/da_plcbf/revision-20260904/static-pilot/static_nominal_collision_vs_fixed_plcbf_avoidance_metrics.json).
This is a safety-versus-progress counterexample, not a successful navigation result. The current
pilot with a `0.15 m` lateral obstacle offset still causes nominal collision and lets the isolated
fixed PL-CBF reach the goal. That pilot is retained in
[`static-offset-0.15/`](artifacts/da_plcbf/revision-20260904/static-offset-0.15/);
the final current-code run is recorded below. The offset breaks the symmetric encounter;
it does not establish general recovery from symmetric deadlock.

## Numerical results

### Static PL-CBF isolation

Saved in [`review-static/`](artifacts/da_plcbf/revision-20260904/review-static/).
The nominal center passes within `0.15151 m` of a sphere with physical radius `0.48 m` and drone
radius `0.05 m`: collision. Fixed PL-CBF stays `0.01583 m` beyond the full `0.68 m` inflated shell,
reaches the goal within `0.00353 m`, and uses the QP throughout all 400 control intervals.
There are no degraded or direct-fallback intervals. The selected PL-CBF dual is positive on
75 recorded boundaries; analytic obstacle rows are disabled. The trace contains 401 boundaries,
including the terminal diagnostic frame.

### Online construction, permanent wind, and compensation baseline

Primary numerical evidence is
[`cold-start-shared-feedforward-3/`](artifacts/da_plcbf/revision-20260904/cold-start-shared-feedforward-3/):
[`config.json`](artifacts/da_plcbf/revision-20260904/cold-start-shared-feedforward-3/config.json),
[`numerical trace`](artifacts/da_plcbf/revision-20260904/cold-start-shared-feedforward-3/online_constant_wind.npz),
[`raw summary`](artifacts/da_plcbf/revision-20260904/cold-start-shared-feedforward-3/online_constant_wind.json),
and [`comparison figure`](artifacts/da_plcbf/revision-20260904/cold-start-shared-feedforward-3/comparison.png).

All methods start with the same eight undiversified braking skills. Adaptive BPTT begins at
switch-on and performs **800 finite, persistent updates in 800 control intervals**, ending at
library `v800`. The simulation is 16 seconds at `dt=0.02 s`; the prediction horizon contains
60 integration intervals and 61 state nodes, spanning 1.2 s.
Wind changes once at 4 seconds to `[1.5, 0.7, 0] m/s`. Obstacles, goal, physical parameters,
actuator limits, and nominal model compensation are matched. The learning rate is `0.002`.

| Method | Minimum physical body clearance | Minimum inflated-shell clearance | Negative library-H intervals | Degraded intervals | Successful terminal goal error |
|---|---:|---:|---:|---:|---:|
| Analytic obstacle HOCBF | −0.52070 m; collision | −0.67070 m | Not its control criterion | 655 | — |
| Frozen braking library | −0.09390 m; collision | −0.24390 m | 121 | 121 | — |
| Frozen + model-compensated skills | +0.04849 m | −0.10151 m | 197 | 197 | 0.00524 m, with safety-margin violation |
| Online-learned DA-PLCBF | **+0.15782 m** | **+0.00782 m** | **0** | **0** | **0.00214 m** |

The learned method executes 795 QP commands and five certified fallback commands. Its PL-CBF
dual is positive on 224 executed QPs and five QP proposals replaced by fallback. These five
intervals retain a positive hard-horizon certificate but lack the more conservative smooth QP
certificate. The frozen hard library H first becomes negative at 4.46 s; the compensated frozen
library does so at 4.56 s (its smooth QP certificate is already unavailable at 4.54 s).
This is a concrete example in which learning
preserves the requested safety margin and horizon coverage that the two frozen libraries lose.
The compensated method does **not** physically collide; its failure is the inflated margin and
finite-horizon certificate. Post-collision trajectories continue in the airborne numerical model;
their eventual goal positions are not credited as successful navigation.

At the **same adaptive state and point model at 5.40 s**, including the common nominal candidate:

| Library evaluated at that identical state | Maximum H | Collision-clear fallback count |
|---|---:|---:|
| Frozen | −0.319716 | 0 / 8 |
| Frozen with wind feedforward | −0.036768 | 0 / 8 |
| Learned | **+0.106960** | **5 / 8** |

These are measured shared-state probes, not comparisons between different executed positions.
There are 16 sampled adaptive states with negative frozen/positive learned H, including seven
where the compensated frozen library is also negative. Shared-reference descriptor loss improves
from frozen `0.186455` to adaptive `0.125697` after wind, with no obstacle term in training.

Because learning begins at switch-on, the trajectories can differ before the wind transition
(maximum full-state component difference `0.13849`). The result demonstrates online construction
and continued adaptation through a wind change; it alone does not isolate learning only after
wind detection. The separate wind-triggered ablation below isolates that timing.

The raw summary preserves older comparison thresholds that required *both* methods to avoid
degradation and required pre-wind equality even during startup learning. Those two flags are
expected to be false here. The current loader retains them under `legacy_comparison_diagnostics`
and evaluates separate mechanism/adaptive-success checks: all seven current checks pass. This
classification changes no recorded state, action, value, or artifact bytes.

### Controlled adaptation only after wind detection

[`wind-triggered-controlled-ablation/`](artifacts/da_plcbf/revision-20260904/wind-triggered-controlled-ablation/)
uses the same configuration as the construction experiment except `learning_start=wind`.
The fixed and adaptive full states agree **exactly before wind** (maximum component difference
`0.0`). Wind changes at 4.00 s, the estimator detects it at 4.04 s, and only then does learning
begin. All **598 attempted updates are finite**, ending at library `v598`. Frozen methods receive
the same nominal model compensation as before and reproduce their prior trajectories.

| Method | Minimum physical body clearance | Minimum inflated-shell clearance | Negative H / degraded intervals | Successful terminal goal error |
|---|---:|---:|---:|---:|
| Frozen braking library | −0.09390 m; collision | −0.24390 m | 121 / 121 | — |
| Frozen + model-compensated skills | +0.04849 m | −0.10151 m | 197 / 197 | 0.00524 m, with safety-margin violation |
| Adaptation after wind detection | **+0.15597 m** | **+0.00597 m** | **0 / 0** | **0.01731 m** |

The adaptive method uses 790 QP commands and ten certified fallback commands, with a positive
PL-CBF dual on 325 executed QPs plus ten proposals replaced by fallback. All eight current
mechanism and adaptive-success checks pass,
including pre-wind equality. Its minimum augmented-library H is positive (`0.008984`).

At the identical adaptive state and point model at **5.60 s**, the frozen library has
`H = −0.282512` and 0/8 collision-clear fallback policies; frozen skills with wind feedforward
have `H = −0.080393` and 0/8; the learned library has **`H = +0.072078` and 4/8**. These maxima
include the shared nominal candidate. Across sampled adaptive states, 25 probes have negative
frozen/positive learned H, including 19 with negative compensated/positive learned H.
The shared-reference descriptor loss is `0.133935`, versus frozen `0.186455`.

This strengthens the causal evidence: the adaptive benefit appears when learning is enabled
only in response to the changed dynamics. It is still one tuned simulated scenario, with small
positive margin, a braking seed, and supplied motion targets. It does not show that every
structured fixed library or every alternative compensation controller must fail. The analytic
baseline is retained in the saved four-method trace and has the same failure as above.

Exact reproduction inputs and evidence:
[`config`](artifacts/da_plcbf/revision-20260904/wind-triggered-controlled-ablation/config.json),
[`trace`](artifacts/da_plcbf/revision-20260904/wind-triggered-controlled-ablation/online_constant_wind.npz),
[`summary`](artifacts/da_plcbf/revision-20260904/wind-triggered-controlled-ablation/online_constant_wind.json),
[`figure`](artifacts/da_plcbf/revision-20260904/wind-triggered-controlled-ablation/comparison.png).
Its timing overlaps final graphics work and is not the primary performance evidence below.

### Retained development outcomes

These are tuned engineering examples, not a predeclared statistical campaign. The centered
static failure and both offset pilots remain available. `wind-pilot-0` has all three PL-CBF
methods safe, and `wind-compensated-nominal-1` shows the compensated baseline reaching the goal
sooner than the adaptive method. `cold-start-pilot-1` and `cold-start-encounter-2` have all three
PL-CBF methods safe; the latter already shows counterfactual coverage expansion over both frozen
libraries. Those outcomes are retained rather than presented as adaptive wins.

## Performance and verification

[`corrected_scaling.json`](artifacts/da_plcbf/revision-20260904/corrected_scaling.json) benchmarks
the actual corrected controller at K = 8, 16, 32, 64, 128, H = 60, with 20 synchronized warm
samples per component on the RTX 4090. Each case has a positive library H, an unsafe nominal
rollout, a positive selected-policy dual, and an accepted QP.

| Policies | Full controller median / p95 | BPTT update median |
|---:|---:|---:|
| 8 | 13.518 / 14.135 ms | 12.902 ms |
| 16 | 13.707 / 13.981 ms | 11.639 ms |
| 32 | 13.301 / 13.538 ms | 12.528 ms |
| 64 | 13.826 / 14.345 ms | 12.721 ms |
| 128 | 14.240 / 15.225 ms | 12.313 ms |

The main construction run separately measures 14.395 ms median / 14.996 ms p95 for the adaptive
controller and 12.536 / 12.785 ms for BPTT. Approximately **27 ms sequentially exceeds a 20 ms
50 Hz budget**. These measurements support a soft 20 Hz schedule on this machine; they are not
a hard-real-time or asynchronous execution result. The simulation's 50 Hz clock is distinct
from wall-clock throughput. The large-library study is an isolated benchmark, not 128-policy
closed-loop safety validation. An XLA autotuning warning occurred during benchmark compilation;
the retained synchronized host-clock samples are the timing evidence.

Current core verification: **53 CPU tests passed** covering shared filter/barriers, smooth-value
and geometry isolation, exact-QP branch parity, and static goal recovery. The learner/integration
GPU checks passed nine tests; two additional CPU checks cover undiversified initialization and
shared nominal compensation. The three focused renderer tests passed across the frame-smoke
and overlay/probe checks; encoded-video inspection is recorded with the videos.
Ruff lint and format checks passed for all 18 modified/new Python files;
compilation, CLI render-help, and `git diff --check` passed.

Conditional finite-horizon evidence under a point estimate does not establish robustness to
estimation error, hardware safety, infinite-horizon safety, or broad statistical superiority.

## Updated videos

All videos replay saved Version-A trajectories using MuJoCo graphics at 1600×900 and 20 fps.
Wind videos hold one measured coverage probe for two explicitly labeled seconds. This extends
playback time without adding simulated time or additional controller steps. Library H is the
maximum hard collision value; the selected dual describes the QP proposal, whose execution
status is shown separately. Persistent collision labels prevent a later goal position from
concealing an earlier physical collision.

| Video | Recorded comparison | Probe pause |
|---|---|---|
| [Static PL-CBF isolation](artifacts/da_plcbf/revision-20260904/review-videos/static_plcbf_only.mp4) | Nominal collision versus fixed PL-CBF avoidance and goal recovery | None |
| [Online skill construction](artifacts/da_plcbf/revision-20260904/review-videos/online_skill_construction.mp4) | Frozen braking versus learning from startup | 5.40 s |
| [Learning versus wind feedforward](artifacts/da_plcbf/revision-20260904/review-videos/learned_vs_wind_compensation.mp4) | Compensated frozen skills versus learning from startup | 5.40 s |
| [Adaptation after wind detection](artifacts/da_plcbf/revision-20260904/review-videos/wind_triggered_adaptation.mp4) | Identical pre-wind fixed/adaptive execution, followed by post-detection learning | 5.60 s |

The [video manifest](artifacts/da_plcbf/revision-20260904/review-videos/video_manifest.json)
records source files, encoded properties, checksums, and review artifacts. Contact sheets and
decoded keyframes are alongside each video. All four MP4s passed complete frame decoding,
sampled visual inspection, and final manifest checksum verification: 161 frames / 8.05 seconds
for static isolation, and 361 frames / 18.05 seconds for each wind comparison.
The wind-triggered video's general construction
title also applies to its initially braking library; the recorded update count remains zero
until the wind is detected.

## Reproduction

From the repository root, use a new output directory for each numerical run:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .pixi/envs/gpu-tests/bin/python examples/da_plcbf/online_constant_wind_demo.py run --config artifacts/da_plcbf/revision-20260904/cold-start-shared-feedforward-3/config.json --output-dir /tmp/crazyflow-new-construction --no-render

XLA_PYTHON_CLIENT_PREALLOCATE=false .pixi/envs/gpu-tests/bin/python examples/da_plcbf/online_constant_wind_demo.py run --config artifacts/da_plcbf/revision-20260904/wind-triggered-controlled-ablation/config.json --output-dir /tmp/crazyflow-new-wind-adaptation --no-render

MUJOCO_GL=egl .pixi/envs/gpu-tests/bin/python examples/da_plcbf/online_constant_wind_demo.py render --trace /tmp/crazyflow-new-construction/online_constant_wind.npz --summary /tmp/crazyflow-new-construction/online_constant_wind.json --video /tmp/crazyflow-new-construction/online_skill_construction.mp4 --left-method fixed --right-method adaptive --probe-pause-time 5.4 --probe-pause-seconds 2

XLA_PYTHON_CLIENT_PREALLOCATE=false .pixi/envs/gpu-tests/bin/python examples/da_plcbf/static_blocking_obstacle_demo.py --output-dir /tmp/crazyflow-new-static

XLA_PYTHON_CLIENT_PREALLOCATE=false .pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_corrected_scaling.py --device gpu --policy-counts 8 16 32 64 128 --horizon 60 --samples 20 --output /tmp/crazyflow-new-scaling.json
```

Use `--left-method compensated` to render the wind-compensated frozen baseline from the same
recorded run. Learning/configuration overrides and exact scenario arrays are saved in the summary.
For the controlled wind-triggered video, use its trace/summary and `--probe-pause-time 5.6`.

## Review entry points

For an independent review, start here and inspect the implementation diff from `e4c4f39` to
the revision commit. Treat the reported checks as claims to verify against the source and saved
traces. The strongest controlled comparison is `wind-triggered-controlled-ablation`; the
startup-construction comparison and retained counterexamples provide the surrounding evidence.
Review the smooth QP value and its derivative together, distinguish rejected QP proposals from
executed commands, and keep shared-state coverage separate from each method's actual trajectory.
If a connector cannot decode MP4 or NPZ files, the tracked JSON summaries, contact sheets, and
individual PNG frames provide directly inspectable evidence; do not claim video decoding without
access to its bytes.

Read `persistent_skill_learner.py`, `continuous_version_a.py`, and `online_constant_wind.py` under
[`crazyflow/safety/da_plcbf/`](crazyflow/safety/da_plcbf/), followed by the static and wind examples
under [`examples/da_plcbf/`](examples/da_plcbf/). The renderer consumes recorded traces; it does
not rerun controllers. Keep the corrected focused tests as the validation scope for this round.
