# Differentiable adaptive policy-library CBFs

Crazyflow's experimental DA-PLCBF package implements a finite-horizon, simulation-grade safety
filter for quadrotors. It combines a task-agnostic fallback-policy library, exact sampled rollout
values, candidate-only truncated backpropagation through time (BPTT), hard candidate admission,
and minimum-intervention filtering. It does not provide an infinite-horizon, distribution-free,
hardware, or real-world guarantee.

## Safety contract

All physical margins use the same sign convention: positive is safe, zero is the configured
boundary, and negative is a violation. Training may use a conservative smooth minimum, but policy
selection, admission, experiment metrics, and postchecks use exact hard sampled minima. Enabled
non-finite inputs fail closed. A rollout certificate applies only to its logged model samples,
obstacle predictions, constraints, numerical tolerances, and finite horizon.

The fallback actor never receives the waypoint or nominal-controller goal. Every library slot is
evaluated concurrently. The structural seed slots and adaptive slots share one residual network;
the runtime selector is an explicit hard-value/admissible-score optimization with score hysteresis,
not a maneuver state machine. A variable-duration skill transitions into a defined brake/hover
tail without shortening the common certificate horizon.

## Runtime paths

The package keeps three plant contracts distinct:

- Version A uses an airborne direct collective-force/body-torque rigid-body model. Its continuous
  QP includes analytic CBF/HOCBF faces, a unique-gradient hard policy-value face, and the exact
  coupled motor-force polytope. Accepted commands pass independent KKT, motor-bound, allocation,
  barrier, and held-step checks.
- Version B runs commands through Crazyflow's force/torque controller, motor allocation and
  clipping audit, rotor lag, first-principles dynamics, and an unclamped integrator. A trust-region
  linearization proposes a command, but only the exact nonlinear equal-horizon discrete residual
  can accept it. Version A's control-affine claim does not transfer to Version B.
- The moving-obstacle path evaluates ballistic/crossing/pursuit/interceptor predictions. These are
  predeclared exogenous oracle forecasts, not sensor-derived or learned forecasts. Only obstacle
  slots active at the current observation boundary are exposed across that boundary's horizon, so
  an unreleased ballistic obstacle cannot reveal its future release schedule. The robust variant
  forms the explicit Cartesian product of finite obstacle and dynamics hypotheses, applies one
  sample-independent motor command, and takes the exact worst sampled postcheck.

Contact, floor crossing, internal command clipping, a missing certified fallback, a failed solver,
or a failed exact postcheck is logged as failure or degraded operation. The experiments never use
Crazyflow's convenience floor clamp as a recovery mechanism.

## Online adaptation

The filter reads only an immutable active snapshot. Candidate work has two explicit execution
modes because wall-clock races must not determine a scientific safety result:

- `logical_simulation` runs a fixed BPTT and hard-validation job from the post-transition causal
  state at a declared logical boundary. Simulated time does not advance during the computation,
  and an accepted snapshot can first appear at the next control boundary. The complete GPU job is
  included in wall-step and release-lateness timing. This is the load-invariant mode used for
  paired algorithmic safety experiments; it is explicitly not a real-time implementation claim.
- `realtime_probe` uses the single-flight CPU worker to measure the hardware-specific asynchronous
  path. The worker only stages a result; the controller thread alone can publish it at a boundary.
  These host-load-dependent traces are feasibility evidence and are excluded from paired safety
  claims. Work that finishes at the terminal observation expires without publication because it
  cannot drive another control.

Both modes copy the active snapshot, perform a fixed number of BPTT updates, and validate the
candidate on current, local, reachable, replay, core, feasibility, diversity, model-version,
finite-value, and runtime gates. Atomic compare-and-swap publication keeps parameters, version,
and digest consistent. Stale, non-finite, collapsed, regressing, or over-budget candidates cannot
become active; the previous admitted payload can be republished as an explicit rollback under a
new version. The admission runtime gate covers the complete prepublication candidate job: context
setup, warm compiled BPTT execution, snapshot construction, evidence preparation/execution, and a
measured hard-validation report pass. Compilation and warmup remain separately recorded
pre-control costs. The atomic publication itself occurs only at a control boundary and is included
in measured orchestration rather than hidden inside the worker budget: startup publication is
pre-control, logical-simulation publication is part of postprocessing and wall-step, and
realtime-probe publication is part of command-preparation and wall-step.

The low-dimensional estimator separately identifies inverse mass, diagonal drag acceleration,
wind, and (when realized motor forces are independently observed) rotor efficiency. Rank-deficient
windows do not update the model. Every accepted estimate increments a logical model version, making
in-flight candidates stale. Bounded symmetric `R=4` or `R=8` particles provide finite-scenario
uncertainty rollouts; they are not a chance-constraint or posterior-coverage claim.

## Reproducible evidence

Authoritative experiments save numeric traces before rendering. Each condition/fold has one
content-addressed scenario tape shared by all paired methods. Run manifests bind configuration,
git/runtime/GPU provenance, named RNG streams, tape and trace digests, events, metrics, raw timing
samples, aggregate statistics, dashboards, and checksums. Compilation, warm control execution, and
offline video encoding are timed separately.

The scientific dashboard consumes only a trace, its exactly bound scenario tape, and an optional
trace-bound evidence sidecar containing recorded rollouts, descriptors, uncertainty, admissions,
and BPTT timing. Missing evidence is labelled `UNAVAILABLE`; the renderer never integrates controls
or invents a trajectory. Final MP4s are decoded and checked for codec, dimensions, frame count,
duration, non-static content, and deterministic replay. Keyframes and contact sheets support a
separate visual-review record.

The repository-root `DA_PLCBF_PLAN.md` maintains the implementation/evidence checklist and exact
ready-for-review gates. Bulk run products live below `artifacts/da_plcbf/`; only the reviewed
content-addressed index is committed.

### Confirmatory and exploratory inference

The final campaign predeclares three confirmatory safety endpoints: trial-level
`operational_failure`, trial-level `any_failure` (the union of physical contact and hard-constraint
violation), and `minimum_hard_margin`. The Bonferroni family contains all three endpoints for every
one of the four required conditions and every full-method-versus-baseline pairing:
`4 × 6 × 3 = 72` comparisons.
A metric-level superiority result requires at least 100 complete matched pairs, an adjusted paired
percentile-bootstrap interval above the predeclared oriented-effect threshold, and both the exact
paired sign interval and one-sided exact sign test to favor the full method. A missing continuous
metric after any retained execution failure blocks that comparison instead of dropping the pair.
For a method that declares a safety controller, `operational_failure` also counts every explicitly
degraded interval, including no-certificate, solver-fallback, and failed-postcheck outcomes, even
when its best-effort command remains physically lucky on that tape. `nominal_only` intentionally
has no safety controller, so its certificate-unavailable marker is excluded from that endpoint;
its physical failures are still counted by both failure endpoints.

Bootstrap count is derived from the confirmatory family size and rounded upward. It must put at
least 100 expected Monte Carlo draws beyond each adjusted two-sided interval endpoint; the
72-comparison family therefore uses 290,000 replicates, not the under-resolved 20,000-replicate
bootstrap that would put only about 6.9 draws in each tail (and only about 3.5 for the old
144-comparison family). An exactly constant paired-effect vector is a declared exception because
its resampled-mean distribution is analytically degenerate and its interval does not depend on
Monte Carlo tail sampling. This resolution gate does not make percentile bootstrap an exact
finite-sample procedure; it only prevents a claim from depending on a handful of simulated tail
draws.

Certification coverage, degraded duration, intervention, controller p99 latency, directly
recorded command-ready p99 latency, and complete wall-step p99 latency are exploratory diagnostics.
They receive unadjusted 95% intervals and are structurally prohibited from setting a superiority
flag, even on a final schedule. Reports and JSON artifacts preserve the analysis role and inference
configuration for every comparison. A supported comparison remains a metric-, baseline-,
condition-, and finite-horizon statement. The campaign-level global flag is true only if every
predeclared confirmatory endpoint supports superiority for every baseline and condition.

## Known boundaries

- Version A is airborne and continuous/local; rotor lag and controller memory belong to Version B.
- Classical HOCBF invariance requires the initial `h >= 0` and first-order `psi_1 >= 0` domain.
- Capsule constraints fail closed at exact projection-Hessian seams and otherwise use exact capsule
  and swept segment--capsule geometry, not sampled-sphere approximations.
- Dynamic contacts are failures, not differentiable training shortcuts.
- On the tested RTX 4090, the final K=64/B=64/H=50 workload must pass a separate command-ready and
  candidate-within-horizon preflight before any 50 Hz claim. Algorithmic safety evidence does not
  imply that preflight passed.
- Safety conclusions are paired empirical finite-horizon conclusions. Counterexamples remain in
  the results and become regression scenarios; they are not silently discarded.
