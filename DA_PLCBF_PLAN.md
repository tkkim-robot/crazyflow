# DA-PLCBF implementation and evidence plan

## Mission

Implement the shared-chat design, **Differentiable Adaptive Policy-Library Control Barrier
Functions (DA-PLCBF)**, in Crazyflow and produce reviewable evidence that the implementation is
mathematically consistent, reproducible, fast on the available RTX 4090, and safer than matched
baselines in the tested finite-horizon simulation conditions.

The implementation must not replace learned fallback-library adaptation with a hand-scripted
maneuver state machine or hidden safety heuristics. The structural seed library, snapshot lifecycle,
selection hysteresis, admission gate, and explicit degraded condition are retained because the
shared chat requires them. They will be visible in configuration, logs, tests, and ablations.

## Source of truth and traceability

Priority order:

1. The user's current request in this task.
2. The complete shared chat, **Online Safety Filter Learning**:
   <https://chatgpt.com/s/t_6a94247c0bb081918b7acd5c6baad95c>.
3. The PL-CBF paper and its official public implementation, used as independent references for the
   fixed-library baseline and equations.
4. The Crazyflow paper, this repository, and the official upstream repository, used for simulator
   semantics and performance methodology.
5. Other primary papers named in the chat, used only for their corresponding baselines or optional
   extensions.

The shared post contains the complete 2,721-line DA-PLCBF response and identifies two uploaded PDFs
(`2026_IROS__Policy_Library_CBF (9).pdf` and `2026_Exntension__TKCBF.pdf`). The public post exposes
their citation metadata and excerpts, but not a public file download. The shared response is
sufficient to specify DA-PLCBF; any claim of exact reproduction of the unpublished extension will
remain gated on access to that PDF or its source.

The official Crazyflow repository and all current branches/PRs contain no PL-CBF, SDCBF, CBF,
policy-library, QP-filter, or safety-filter implementation. Commit `4a008e02` on
`upstream/feat.thrust_limits` correctly fixes the symplectic vectorizer's excluded `dt` argument,
which is independently applied here. Its rotor clipping is intentionally not imported: it maps a
stopped rotor state to the calibrated airborne minimum after one step and therefore changes
powered-off semantics; filter-level motor feasibility and exact postchecks remain separate.
Crazyflow is MIT-licensed. The public PL-CBF paper is CC BY 4.0, while its code repository currently
has no license; implement the method clean-room from the paper/chat and do not copy unlicensed code.

A requirement-to-evidence table will be maintained as implementation proceeds:

| Requirement | Code | Tests | Experiment/metric | Status |
|---|---|---|---|---|
| Scenario-batched finite-horizon values | `values.py`, `quad_rollouts.py`, `quad_uncertainty.py`, `dynamic_rollouts.py` | value/sign/shape, swept, batch/single, and uncertainty regressions | hard margin traces | code/tests pass; final evidence pending |
| Shared structured + latent residual actor | `actor.py`, `library.py`, `quad_policy.py` | broadcasting, goal-exclusion, duration-tail, structural-slot, and gradient tests | normalized descriptor sidecar/dashboard | code/tests pass; final evidence pending |
| Truncated BPTT learner | `bptt.py`, `actor_bptt.py`, `quad_actor_bptt.py`, `quad_generic_diversity_bptt.py` | central-difference, finite/nonzero gradient, fixed-budget, cache, and learning tests | compile/warm/execution timing benchmark | code/tests pass; claim-grade timing pending |
| Coverage + redundancy + diversity objective | `losses.py`, `actor_losses.py`, `quad_actor_losses.py` | term-level, units, conservative surrogate, trust, and anti-collapse tests | matched one-factor ablations | code/tests pass; ablation evidence pending |
| Immutable active/candidate snapshots | `snapshots.py`, `runtime.py` | immutability, staleness, controller-boundary publication, rollback, and concurrency tests | content-addressed admit/reject events | code/tests pass; run artifacts pending |
| Hard candidate admission gate | `validation.py`, `experiments.py` | bad/stale/nonfinite/collapsed/regressing/runtime rejection tests | six-fold non-regression report | code/tests pass; run artifacts pending |
| PL-CBF filter with exact post-check | `polytope_qp.py`, `version_a_filter.py`, `dynamic_filter.py`, `version_b_runtime.py` | reference-solver KKT, allocation, held/swept, nonlinear replay, and rejection tests | feasibility/intervention/tail latency | code/tests pass; campaign evidence pending |
| Dynamics adaptation and uncertainty samples | `estimator.py`, `quad_uncertainty.py`, `uncertain_dynamic_filter.py` | identifiable/rank-deficient estimator, bounded particles, Cartesian robust rollouts | estimation error and coverage recovery | code/tests pass; R4/R8/oracle evidence pending |
| Static, ball, and interceptor scenarios | `scenarios.py`, `experiments.py` | deterministic streams, analytic motion, contact/swept, pairing, and causality tests | paired safety trials and falsification | interceptor ready; ballistic exposure redesign in progress |
| MP4 dashboard and replayable traces | `artifacts.py`, `dashboard_evidence.py`, `scientific_dashboard.py` | schema/hash/replay/codec/non-static/terminal-mask tests | visually inspected trace-bound MP4s | synthetic QA passes; real campaign videos pending |

## Exact method contract

### Roles

- The nominal controller performs task tracking and is not trained by DA-PLCBF.
- A task-agnostic fallback library supplies diverse feedback recovery behaviors.
- The runtime filter evaluates only an immutable, previously validated active snapshot.
- A learner copies the active snapshot, trains a candidate with truncated BPTT, validates it on a
  common hard scenario set, and atomically admits it only when every gate passes.
- A model estimator updates low-dimensional dynamics parameters; residual dynamics are deferred
  until the parametric path is correct.

### Policy library

Use one shared policy of the form

```text
pi_i(o) = bounded(pi_struct(o; c_i) + residual_scale * delta_pi_theta(o, z_i))
```

with a small, initially zero residual MLP, fixed base codes plus trainable code offsets, fixed-size
obstacle features and masks, skill phase, and a common rollout horizon. Variable maneuver durations
transition through a mask into a defined hover/brake tail; they never shorten the certificate
horizon or alter JIT shapes.

The initial proof of concept uses a documented structural core and `K=64` on Crazyflow, as required
by the chat. Structural policies remain unchanged during candidate training. Adaptive slots are
trained; candidates are admitted before old adaptive policies are considered for retirement.

### Values and training objective

For policy `i`, local state/environment scenario `b`, and dynamics/disturbance sample `r`, roll out
the full closed loop and compute every barrier over every time step. Maintain separate values:

- a smooth conservative soft minimum for training;
- a hard sampled minimum for selection and reported safety;
- a denser/exact hard check for candidate admission and post-filter acceptance.

The primary objective contains the chat's terms:

- best-policy scenario coverage;
- safe-policy redundancy;
- normalized trajectory-descriptor diversity;
- skill-code coverage;
- action magnitude and action-rate regularization;
- terminal recoverability bias;
- active/candidate trust-region retention.

The hard runtime certificate never uses a learned critic or soft training surrogate.

### Filtering

Implement and distinguish two paths:

1. **Version A:** direct collective-force/body-torque rigid-body model and continuous PL-CBF QP.
2. **Version B:** Crazyflow's full controller/actuator stack with the discrete nonlinear PL-CBF
   condition, a trust-region linearization, and an exact nonlinear acceptance check.

The current Crazyflow `Control.force_torque` interface converts desired wrench through bounded motor
allocation and rotor-speed dynamics. It is therefore not automatically the control-affine plant
assumed by Version A. Before making a continuous-QP claim, the implementation will either expose a
validated direct-wrench dynamics adapter or explicitly use the discrete condition for the existing
stack. Downstream clipping may not masquerade as QP feasibility.

For the continuous minimum-intervention QP, enforce the selected barrier halfspace together with
the affine per-motor thrust inequalities induced by wrench allocation. The physically valid airborne
wrench set is a coupled polytope, not a four-dimensional wrench box. Prefer a small exact
four-dimensional active-set/projection solver whose KKT conditions can be checked against an
independent reference. If a generic solver is added, its version and tolerances become part of every
manifest. Every proposed action is checked for finite values, motor-thrust bounds, allocation
round-trip consistency, and the exact applicable barrier condition before execution. Version A is
explicitly scoped to airborne flight; Crazyflow's special all-zero idle command is not folded into
the convex flight set through a mode switch.

### Safety claim boundary

Passing experiments supports only this statement:

> Under the logged model/scenario samples, constraints, numerical tolerances, and finite horizon,
> the hard rollout and filter checks observed the reported margins and violation rates.

It does **not** prove infinite-horizon, distribution-free, real-world, or hardware safety. No safe
fallback is an explicit degraded result, not a success. Candidate learning cannot retroactively
certify an initially unsafe state.

## Repository layout

Fit the chat's proposed architecture into the existing package:

```text
crazyflow/safety/da_plcbf/
├── config.py
├── types.py
├── dynamics.py
├── scenarios.py
├── obstacles.py
├── policies.py
├── rollouts.py
├── values.py
├── descriptors.py
├── losses.py
├── bptt.py
├── proposal.py
├── selector.py
├── qp.py
├── discrete_filter.py
├── snapshots.py
├── validation.py
├── estimator.py
├── runtime.py
├── metrics.py
└── artifacts.py

examples/da_plcbf/
benchmark/da_plcbf.py
tests/unit/safety/da_plcbf/
tests/integration/safety/da_plcbf/
```

Keep the numerical rollout independent from visualization. Authoritative runs save numeric traces;
video rendering replays those immutable traces offline.

Declare Optax directly rather than relying on its current transitive installation through Flax.
Add and pin an explicit video encoder/backend in the experiment dependency set.

## Work plan

### Phase 0 — clean baseline and source audit

- [x] Preserve `main` at pushed commit `7bb7aa4`.
- [x] Reset `plcbf` to `main` and remove discarded branch-only work.
- [x] Read the complete shared DA-PLCBF response.
- [x] Record exact upstream branches/commits and determine whether any compatible PL-CBF/SDCBF
  implementation exists.
- [x] Audit `upstream/feat.thrust_limits` commit `4a008e02` and reuse only changes whose semantics
  pass DA-PLCBF actuator and integration tests; do not inherit its known overshoot as a guarantee.
- [ ] Re-run the complete CPU test suite and the existing generic BPTT GPU smoke test.
- [ ] Record GPU, driver, JAX/XLA, Python, OS, and git provenance.
- [x] Fix the audited reproducibility hazards before scientific runs: seeded environment resets now
  reproduce goal sequences, visualization marker orientation uses a deterministic construction,
  and a persistent renderer rejects camera/resolution changes until it is explicitly closed.

Gate: a clean reproducible baseline and no mislabeled claim that Crazyflow's generic BPTT benchmark
already tests DA-PLCBF BPTT.

### Phase 1 — minimal mathematical reference system

Implement a double-integrator or planar reference problem with static circular obstacles, `K=16`,
structured feedback policies, fixed shapes, and direct BPTT over policy parameters.

- [ ] Barrier sign and units.
- [ ] Hard and conservative-soft rollout values.
- [ ] Scenario/policy/dynamics batching `[K, B, R, H, ...]`.
- [ ] Coverage, redundancy, descriptor diversity, regularization, and trust terms.
- [ ] Exact minimum-intervention QP over the applicable actuator polytope and fallback action path.
- [ ] BPTT gradient and optimization loop.
- [ ] Fixed-library PL-CBF and single-policy PCBF baselines.

Gate: finite-difference gradients agree, QP KKT residuals pass, hard margins never come from a soft
surrogate, and training improves held-out empirical certified coverage without collapsing the
library.

### Phase 2 — Crazyflow Version A and cold start

- [ ] Implement or validate a direct-wrench rigid-body adapter using Crazyflow state conventions and
  physical parameters.
- [ ] Cross-check one-step and batched dynamics against an independently equivalent calculation and
  the closest executable Crazyflow path under matched assumptions.
- [ ] Verify the control-affine identity numerically and analytically over randomized states and
  wrench pairs.
- [ ] Factor unclipped wrench-to-motor-force and motor-force-to-wrench maps, enforce motor bounds in
  the filter, and verify accepted commands survive allocation unchanged.
- [ ] Add static spherical/capsule barriers plus arena, altitude, speed, angular-rate, and tilt
  constraints.
- [ ] Implement a separate waypoint nominal controller.
- [ ] Implement the `K=64` structured library and shared 2x32/64 latent residual actor.
- [ ] Train skill-code offsets and residual weights with truncated BPTT.
- [ ] Run cold-start learning before motion and save every adaptation epoch.
- [ ] Implement the continuous PL-CBF filter only where the affine/direct-wrench contract is true.

Gate: hard held-out policy values and local recovered-safe-set coverage increase, the nominal goal is
absent from fallback observations, physical limits are respected, and no policy improvement is
created by hidden floor/contact clipping.

### Phase 3 — active/candidate runtime

- [ ] Immutable, versioned active and candidate snapshots.
- [ ] Fixed-budget adaptation worker that cannot block the filter.
- [ ] Candidate validation on current, perturbed, replay, reachable, dynamics, and obstacle samples.
- [ ] Current-state margin, local non-regression, core preservation, feasibility, diversity,
  freshness, finite-value, and runtime gates.
- [ ] Atomic swap, stale rejection, rollback, and explicit degraded outcome.
- [ ] Mathematically defined selection by positive hard value then admissible-set proxy, with the
  chat-specified switch hysteresis logged and ablated.

Gate: concurrency tests prove active parameters cannot mutate during learning; injected bad, stale,
nonfinite, collapsed, or slower-than-budget candidates never become active.

### Phase 4 — dynamics changes and estimation

In order:

- [ ] Oracle wind step.
- [ ] Oracle mass/payload change.
- [ ] Oracle drag change.
- [ ] Oracle symmetric and single-rotor efficiency changes.
- [ ] Smooth/time-varying gust.
- [ ] Low-dimensional online parameter estimator.
- [ ] Uncertainty/sigma-point scenario rollouts (`R=4`, then `R=8`).
- [ ] Residual model only after the parametric estimator and uncertainty path pass.

Gate: report coverage loss and recovery rather than reward alone; compare oracle, estimated, and
estimated-plus-uncertainty variants; reject stale candidates when the dynamics version changes.

### Phase 5 — nonlinear full stack

- [ ] Use Crazyflow force/torque or attitude commands through motor allocation, actuator clipping,
  rotor dynamics, and the original integrator.
- [ ] Implement the discrete nonlinear PL-CBF condition.
- [ ] Implement trust-region linearization and exact nonlinear post-check.
- [ ] Cross-check accepted actions against direct nonlinear evaluation and fall back on rejection.
- [ ] Compare Version A and Version B without transferring the affine guarantee to Version B.

Gate: no accepted action violates the configured exact residual tolerance in the validation suite;
all solver failure and fallback events are counted.

### Phase 6 — dynamic and adversarial obstacles

- [ ] Ballistic balls with uncertain release velocity, treated as predicted obstacles rather than
  differentiable contacts.
- [ ] Scripted crossing drone.
- [ ] Bounded pursuit controller.
- [ ] Predictive interceptor.
- [ ] Randomized attacker modes represented as finite trajectory scenarios.
- [ ] Combined wind/rotor-change stress tests.

Gate: matched-condition trials show where DA-PLCBF helps and where it fails; contact is always a hard
failure in these experiments and is never used as a training shortcut.

### Phase 7 — baselines, ablations, and falsification

Core baselines:

1. nominal only;
2. analytic distance CBF/HOCBF;
3. one fixed-fallback PCBF;
4. handcrafted fixed-library PL-CBF;
5. offline/frozen SDCBF-style learned library, labeled precisely to the available source;
6. DA-PLCBF without online model adaptation;
7. full DA-PLCBF.

Required ablations:

- BPTT versus sampling-only and hybrid proposal+BPTT;
- generic diversity versus PL-CBF-aligned coverage+diversity;
- no redundancy, diversity, trust, validation gate, or uncertainty sampling;
- fixed versus trainable skill codes and durations;
- policy count, horizon, scenario count, and adaptation budget;
- independent policies versus the shared actor on a smaller matched configuration;
- BPTT versus SHAC only after a faithful SHAC training-only implementation exists.

Run randomized boundary searches and optimization-based falsification around low-margin initial
states, obstacle timings, wind, mass, rotor efficiency, estimator error, and actuator saturation.
Counterexamples become regression scenarios; they are not discarded or silently retuned away.

Gate: at least 100 paired randomized trials per final reported condition, matched seeds across
methods, uncertainty intervals, all failures retained, and no claim of superiority where the paired
evidence does not support it.

### Phase 8 — performance and reproducible artifacts

Benchmark on the available RTX 4090 with compilation separated from warm execution:

- rollout forward pass;
- backward pass and optimizer step;
- active filter end-to-end;
- QP/nonlinear solve;
- candidate validation;
- video rendering separately from control timing;
- scaling over `K`, `B`, `R`, and `H`.

Report raw repetitions plus median, p95, p99, worst observed, deadline misses, memory use, and device
provenance. Compare paper numbers only with explicit hardware/protocol differences.

Every run writes:

```text
artifacts/da_plcbf/<run-id>/
├── manifest.json
├── config.json
├── provenance.json
├── seeds.json
├── scenario_tapes/
│   ├── <fold>.npz                 # shared across conditions only when explicitly mapped
│   └── <condition>/<fold>.npz     # condition-specific immutable tape
├── checkpoints/
├── methods/
│   └── <method>/<condition>/<seed>/
│       ├── trace.npz
│       ├── events.jsonl
│       ├── metrics.json
│       └── timing.json
├── aggregate/
│   ├── paired_metrics.csv
│   ├── confidence_intervals.json
│   └── report.md
├── plots/
├── videos/
├── keyframes/
├── contact_sheets/
├── SHA256SUMS
└── visual_review.md
```

The manifest records hashes for configuration, traces, checkpoints, plots, and MP4s. A replay
command regenerates plots and videos from the trace without re-running control.

Each immutable scenario tape contains initial states, obstacle trajectories, disturbance sequences,
dynamics changes, estimator noise, time grids, masks, and named/folded RNG streams. `seeds.json`
maps every condition/fold to an exact relative path and semantic digest. All paired methods for the
same condition/fold must consume that digest. A shared `scenario_tapes/<fold>.npz` is permitted only
through explicit, unambiguous mappings; condition-specific generation uses
`scenario_tapes/<condition>/<fold>.npz`. Traces include true/estimated states, nominal/filtered/
actually applied controls, every hard barrier, training values, policy values and selection,
snapshot/model versions, all solver/KKT/post-check residuals, clipping/saturation, degraded/failure
flags, loss terms, gradient norms, and component latencies. Safety clearance is evaluated at
dynamics substeps or with swept checks, never only at controller ticks.

The repository currently ignores JSON and CSV broadly. Add a deliberate artifact policy: either
scoped `.gitignore` exceptions for compact manifests/reports plus ignored bulk traces/videos, or an
ignored artifact tree with a committed index containing hashes and reproduction commands. Never let
glob ignores silently drop claimed evidence. Pin an explicit MP4 backend (rather than relying only
on the host's untracked `/usr/bin/ffmpeg`) and record codec/backend versions.

Gate: artifact validation checks schema, finiteness, frame count, duration, dimensions, codec,
non-static video content, metric/trace agreement, and replay determinism.

### Phase 9 — visual review and revision

Create synchronized MP4 dashboards with:

- 3D world, actual/nominal/fallback trajectories, selected rollout, prediction tubes, and margin;
- policy-value heat map;
- normalized descriptor view;
- true/estimated dynamics and uncertainty;
- loss, safe-policy count, candidate admission, and BPTT time;
- nominal/filtered action, intervention, solver status, and control latency;
- pre-change, post-change, and adapted ghost rollouts.

For each final video, extract representative frames and a contact sheet, inspect them at original
resolution, and revise occlusion, camera, colors, labels, scales, timing, and event annotations until
the safety-critical behavior is unambiguous. Visual appeal never substitutes for numeric evidence.

Gate: videos are legible without the console, unsafe trajectories and degraded periods are visibly
distinct, overlays agree with saved traces, and the visual review records what was checked and what
was revised.

## Test hierarchy

### Unit

- dynamics equivalence and batch/single agreement;
- policy/code/scenario broadcasting;
- duration-to-tail transition and common horizon;
- barrier signs and physical units;
- soft-min conservativeness and numeric stability;
- autodiff versus central finite difference for state, action, and parameters;
- QP KKT, actuator-polytope bounds, infeasibility, and exact residual;
- loss-term invariants and anti-collapse behavior;
- snapshot immutability, staleness, merge, rollback, and provenance;
- estimator recovery on synthetic identifiable data;
- deterministic scenario generation and artifact serialization.

### Integration

- learner disabled while filter runs;
- learner active while active snapshot remains byte/leaf-identical;
- swap only at a control boundary;
- no JIT retracing in steady fixed-shape operation;
- bad/stale candidate rejection;
- no-safe-policy degraded reporting;
- full-stack exact post-check and fallback;
- deterministic rerun from manifest;
- logging does not synchronize the timed device hot path.

### Scientific

- collision/constraint-violation rate and minimum clearance;
- fraction of time and states with at least one hard-certified policy;
- robust library value and safe-policy count;
- QP/NLP infeasibility and degraded duration;
- intervention norm, selections, and switch rate;
- descriptor coverage and collapse measures;
- coverage-recovery time after change;
- accepted/rejected candidate ratio;
- estimation error;
- forward, backward, validation, filter, and solver tail latency.

## Review loop

For every phase:

1. implement the smallest complete vertical slice;
2. run focused unit and integration tests;
3. independently cross-check equations, signs, shapes, units, and gradients;
4. run deterministic CPU smoke and GPU smoke;
5. run held-out and adversarial conditions;
6. inspect quantitative artifacts;
7. generate and visually inspect replay videos;
8. turn every discovered failure into a regression test or documented limitation;
9. rerun the complete relevant matrix before advancing.

## Ready-for-review definition

The branch is ready only when:

- all applicable phase gates above pass;
- the full repository test suite, lint, format, docs, and artifact validators pass;
- BPTT gradients and learning are directly tested for DA-PLCBF, not inferred from the generic
  Crazyflow benchmark;
- the active/candidate safety architecture survives injected failures and adversarial cases;
- final paired trials and raw results are saved and reproducible;
- performance claims include tail latency and compilation/provenance distinctions;
- at least the cold-start, wind/mass/rotor, ball, and interceptor demonstrations have clear MP4s;
- every video has a completed visual review;
- the report states counterexamples and finite-horizon limitations alongside improvements;
- no hidden heuristic, state-machine maneuver selector, silent clipping, or unsupported guarantee
  remains.

Hardware deployment is not part of this simulation completion gate because no physical platform or
hardware authority has been provided. It is a later phase only after simulation evidence passes.
