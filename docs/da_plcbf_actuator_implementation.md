# Actuator implementation and validation map

The study starts from clean `main` commit `8c0320617a6fae56d8fa7179349ed4c07311a04c`,
including the accepted minor video polish, and develops on `codex/actuator-study`.
`artifacts/da_plcbf/actuator-study-20260906/v1/STARTING_MANIFEST.json` records SHA-256
digests and sizes for 1,164 existing accepted-study files (486,440,662 bytes). All new
results use fresh directories. The accepted wind result and its stronger-comparator
limitation remain the regression anchor; no old checkpoint is relabeled as actuator-aware.

## Reuse and changed assumptions

The old `full_stack.py` and `discrete_filter.py` were inspected. Their complete native
simulation path is useful context for P2 but its wrench-command interface and older
discrete certification scheme do not implement the proposed effort-coordinate A2 filter.
The old `quad_rollouts.py` is explicitly a 13-state direct-wrench transition. The new
plant and learner do not pass a wrench to that integrator. `continuous_version_a` uses
the word “augmented” for adding a nominal *candidate*, not for adding motor states.

The actuator filter reuses only the body projection of the audited swept-sphere
geometry, its conservative smooth minimum, the generic SPD QP projector, and the
inclusion-exclusion volume formula with an **identity command map**. Their mathematical
assumptions were checked. None of the old speed/tilt/arena input rows is reused.
The existing analytic world and actual XML-sphere/floor audit are reused for independent
outcome accounting. The recorded-state interpolation bound remains distinct from a
bound on unrecorded plant integration error.

## Filter derivation and acceptance

For a frozen model/library, `H(y,t)` is the conservative smooth lower bound on the
minimum squared collision clearance over the full 17-state rollout. Its row is

`a = -grad_s(H)/tau`

`b = partial_t(H) + grad_x(H)·body_rhs(x,B diag(eta)s) - grad_s(H)·s/tau + alpha H`.

The QP solves `a u <= b` in normalized physical commands
`z=(u-u_min)/(u_max-u_min)` with identity SPD metric. Reported multipliers correspond
to this normalized objective; the selected collision multiplier is credited only when
that QP action executes. Hard values, smooth values, eligibility, QP feasibility,
nonlinear acceptance and actual physical outcomes remain separate fields.

Policy anchors restart from the current state and phase zero at each control call.
State differentiation recomputes that anchor from the perturbed initial state. The
absolute-time partial shifts obstacle predictions while keeping the active waypoint,
policy phase, parameters and model fixed. Waypoint switches, model changes and policy
publications can therefore jump the value function; no invariant-set conclusion is made.

The initial QP has eight command-box faces and the selected policy row. If its held
prediction violates operational constraints, at most two sequential QP refinements
linearize the nine minimum physical margins over the later held nodes. The reference
is the preceding proposal (or the bounded fallback if that solve was nonfinite), with
a trust box of half the original command range. Original bounds are never enlarged.
The nine margins cover six arena faces with the accepted 0.08 m arena inset, speed,
body angular rate, and tilt. Every proposal must then pass the full nonlinear holding
checks; exhausting the iteration budget does not authorize relaxation.

The reference hold is 0.040 s. Fine checks use evolving motors at 0.005 s nodes and
swept relative segments against the current obstacle prediction. Those geometric
checks are exact for their interpolation. Finer plant integration and the separate
rotating XML-collider audit bound the practical interpretation of curved motion;
sampled physical margins alone do not establish a continuous-time global guarantee.

The normal fallback and the common emergency brake use the same bounded actuator
adapter and evolving plant. Modes distinguish accepted QP, valid fallback, immediately
checked emergency, degraded best effort and invalid input. Emergency execution carries
no fabricated policy multiplier. Invalid physical/model data returns an explicit NaN
command sentinel for the experiment runner to terminate and report.

## Milestone record

This is a live implementation map, not the final research report. Physics/allocator
tests pass (22), independent-plant tests pass (28), and initial actuator-filter tests
pass (9). The learning module has a separate focused suite (10). Baseline competence,
full-controller sanity runs, development budgets, sealed evaluation, timing/transfer
checks and video promotion are still required before a comparative result is claimed.

### Full-controller sanity and computation checks

The three completed GPU Gate B episodes use the seed-11 nominal 128-update library:
8 seconds of obstacle-free tracking, 14 seconds with static obstacles, and 14 seconds
with the prescribed moving obstacles. All three finish both waypoints, remain exposed
through the declared duration, and have no recorded actual-collider or operational
violation. Their 900 controls have no measured controller-service deadline miss.
The moving case has 55 nominal predictions with negative hard collision value and
43 executed positive policy-row multipliers, so the obstacle case exercises the filter.
These development sanity episodes do not establish adaptive benefit.

The initial full reverse Jacobian costs about 45 ms on the RTX 4090. Full forward AD
reduces the same-state computation to about 13–14 ms; 10 CPU filter tests include
forward/reverse and finite-difference checks. Homogeneous packed transport retains
every result field and its exact dtype, and separately tested P0 batching preserves
the original integration nodes and first-contact prefix. Complete moving-case
controller service averages 16.37 ms including mandatory host work, with 17.25 ms
p95 and 22.39 ms maximum. Simulation and physical auditing average another 6.69 ms
per control. These numbers motivate measuring actual learner availability rather than
equating an 18 ms isolated update with schedulability in a 40 ms cycle.

A bounded directional-AD experiment computes the four motor partials, the derivative
along the full augmented drift, and the obstacle-time partial. This is sufficient for
the same mathematical row; missing body-coordinate diagnostic partials are NaNs with
an explicit mask. Full forward/reverse modes remain available. CPU row/action parity
passes, but `linearize` is slower on the GPU (about 40 ms). Direct batched JVP costs
about 13.3 ms and gives the same tested actions, without a material speed advantage;
full forward AD remains the chosen implementation pending further validated profiling.
Both negative profiles and their exact source versions are retained.

An initial CPU integration attempt failed before its first control during P0 warmup
with an LLVM materialization/allocation error; it is an incomplete attempt, not a
collision-free trial. Two orchestration serialization errors were also retained and
fixed. GPU P0 integration subsequently completed the three sanity episodes above.

The accepted wind case was replayed after the shared geometry utility change: all
shared non-timing arrays, including actual dense states and actions for both methods,
are bitwise identical to the accepted record. The legacy geometry derivative mode
remains the default on the old path.
