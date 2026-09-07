# Numerical stabilization and causal re-evaluation

The two numerical defects identified in the review are repaired and tested in the production computation paths. The primary comparison is **handcrafted frozen → learned frozen → learned adaptive**. The corrected implementation retains a conditional adaptation benefit and a conditional harm; it does not establish generally safer adaptive control. This report supplements the immutable [previous report](da_plcbf_actuator_diagnostics_report.md).

## Implemented repairs

The matched learner passes the fixed nominal teacher parameters and model as dynamic executable arguments. Student and teacher evaluate the same current-state/anchor batch through one shared `jax.lax.map` body, including inside the fused reverse-mode loss-and-update executable. The teacher remains stopped-gradient and physically fixed. Compile-only scan unrolling is aligned; explicitly permitted differences in physical feedback gains retain the original teacher behavior. There is no model-equality shortcut, hard-coded zero loss, larger dead zone, outcome gate, optimizer reset, or publication suppression. Persistent Adam remains part of the method.

Matching two separately compiled graphs was insufficient: an initial long probe still drifted after changing anchor batches. That failed probe is retained. The final shared-body implementation passed two fresh GPU processes with separate initially empty compilation caches. Each executed 128 fused updates: two library families × old/new numerics × zero/persistent history × 16 successive updates. Across the **64 corrected zero-history updates**, gradient norm and parameter displacement were exactly zero. The legacy handcrafted first step reproduced loss 1.90747e-8, gradient norm 1.18930e-4, and update norm 0.00790608. The persistent learned checkpoint still moved on its first corrected update (norm 0.0133639); stationarity is not asserted for an optimizer with history. A separate regression test induces momentum with fault updates, restores teacher parameters, and verifies that zero current gradient can coexist with a nonzero Adam update.

The QP now scales its normalized solver tolerance by the largest normalized row norm and tightens all affine faces with a float32 roundoff-aware inward margin. It converts the candidate back to actual command precision and rechecks the original policy, operational, command, and trust-region rows. The original raw physical tolerance remains **2e-6**, and nonlinear held-command checks remain required. A narrow polytope can fail closed under this conservative formulation; tightening is not guaranteed to improve flight outcomes.

The retained diagnostics include the final QP proposal even when rejected, normalized action, executed command, row scales, inward margins, solver branch and feasibility, solver primal/stationarity/complementarity residuals, executed residuals, and rejection flags. These describe the final QP proposal, not every intermediate sequential-refinement attempt.

Each fresh build also tested 66 QP points: old/new numerics × 11 command perturbations spanning ±1e-5 × three adjacent float32 bounds. Across both builds, **20/66 legacy points** violated the final raw tolerance despite solver feasibility (minimum −7.62939e-6). **All 66 corrected points** were feasible and passed the executed check; their raw policy residuals ranged from 1.37329e-4 to 1.52588e-4. These are targeted numerical regression probes, not a global feasibility proof.

## Closed development set

The sealed set contains exactly three already observed cases: the harmful structured scene (seed 30101, effectiveness 0.7), the prior learned fault-cell gain (navigation seed 62104, effectiveness 0.7), and its no-change regression (same navigation geometry, nominal dynamics). These are development re-evaluations, not fresh validation worlds. Geometry, initial physical state, observation/filter contract, low-level F2 adapter, 16 candidates, 60 × 20 ms rollout horizon, and 40 ms control clock are held fixed within comparisons. The exposure is 14 seconds for the structured scene and 24 seconds for navigation, with collisions recorded as early termination.

Each case has four frozen-baseline runs (two libraries × two QP modes) and eight continuation/onset-freeze runs (two learner modes × two QP modes × two adaptation schedules): **36 physical attempts total**. No union or handcrafted-adaptation flight is included. “BPTT-tuned handcrafted primitives” is the auxiliary name for historical PD adaptation: it changes velocity/duration offsets, not PD feedback gains.

With both numerical repairs enabled, the primary results are:

| Previously observed case | Handcrafted frozen | Learned frozen | Learned adaptive | Learned onset-freeze control |
|---|---|---|---|---|
| Harmful effectiveness case | Collision | Safe task | Collision | Safe task |
| Prior fault-cell gain | Collision | Collision | Safe task | Collision |
| No-change regression | Collision | Safe task | Safe task | Safe task |

“Safe task” requires all waypoints, no modeled collider contact, no actual operational violation, and completion of the common exposure. Startup-frozen deployment and onset-freeze attribution are different comparisons: onset-freeze shares all pre-intervention learning and optimizer history with continuation.

All **12 continuation/onset-freeze pairs** passed exact authentication of the initial full learner state, physical world, and pre-intervention state/command/update prefix, including boundary optimizer/teacher/model state. The complete factorial outcomes below show continuation/onset-freeze, with **S** = safe task and **C** = collision:

| Case | Old learner / old QP | New learner / old QP | Old learner / new QP | New learner / new QP |
|---|---|---|---|---|
| Harmful effectiveness | C / S | C / S | C / S | C / S |
| Prior fault-cell gain | S / C | S / S | S / C | S / C |
| No change | C / S | S / S | S / C | S / S |

Thus the nominal regression disappears with the learner repair under either QP formulation. The harmful adaptive flight persists in every cell. The fault benefit remains with both repairs, although the old-QP result changes to both-success after fixing the learner. These conditional results cannot be pooled into a general success rate: the cases were selected from known outcomes and share a library seed.

The QP change also affects the frozen baselines. Handcrafted frozen is safe in all three legacy-QP cases and collides in all three corrected-QP cases. Learned frozen changes from safe/safe/collision to safe/collision/safe across harm/gain/no-change. Those changes are disclosed rather than treated as learner effects. Legacy cells are new executions under the retained legacy numerical modes, not claims of bitwise reproduction of historical binaries.

Across the flight records, the legacy QP reports solver-feasible proposals whose executed policy row fails tolerance **84 times in 8,047 controller decisions**. The corrected QP has **zero such mismatches in 7,434 decisions**; its minimum raw policy residual among solver-feasible proposals is 1.78814e-7. Different trajectories and exposure lengths prevent interpreting these denominators as matched samples. The targeted common-input probes above establish the numerical comparison more directly.

## What changes the actual commands

At the first corrected harm-pair command divergence, **t = 2.48 s**, physical state, estimated model, previous command, goal, and nominal command are identical. Both select skill 12, but adaptation changes its certificate row and gradient. The nominal command's raw row residual is approximately **+0.063025** for onset-freeze and **−0.075499** for adaptation (host float64 calculation from logged float32 inputs). The frozen filter passes the nominal command; adaptation activates the policy constraint with dual 8.76242e-5 and executes a different feasible QP command. Both QPs pass; this initial divergence is not a roundoff-induced emergency switch. The adaptive trajectory subsequently collides. This local causal observation does not prove that this one decision alone determines the later collision.

At the corrected gain pair's first command divergence, **t = 3.84 s**, the same five inputs are again identical. Onset-freeze selects skill 14 and adaptation selects skill 3. The nominal row residual changes from approximately **+6.96123 to −1.30700**, and adaptation makes a feasible QP intervention. Adaptation subsequently completes all four waypoints while onset-freeze collides after three. At the nominal pair's divergence, **t = 3.92 s**, both select skill 3 and make slightly different feasible QP corrections; both complete the task. Full commands, rows, gradients, duals, and flags are retained in the analysis report.

This supports investigating how updates alter selected certificates and gradients. It does not support increasing diversity merely to widen a plotted fan. The current objective still restores nominal-teacher motion and braking; no larger objective revision was mixed into this numerical iteration.

## Timing, validation, and provenance

The runtime summary now audits each actual application-to-application interval against its sensing time and certified duration, separately from controller service deadlines. All 36 simulated development runs use the aligned zero-delay clock and pass this timing audit. Applying the same audit read-only to the eight historical runtime attempts finds all four paced attempts covered and all four asynchronous attempts uncovered. Their maximum holds are 50.109 ms (learned frozen), 55.322 ms (learned adaptive), 51.142 ms (handcrafted frozen), and 53.066 ms (BPTT-tuned handcrafted primitives), versus a 40 ms modeled check. Application latency also invalidates the assumed starting state. **The asynchronous scheduler/certified-hold contract is not repaired here, and deployable adaptive safety is not claimed.** Recomputing teacher rollouts changes learner cost; no new uncontended throughput or deadline claim is made from these probe timings.

Regression records contain separate passing suites: 49 learner/filter/video checks on the final shared-body implementation; 60 experiment/async/packed/numerical integration checks before the final learner-body adjustment; and seven final focused numerical/timing/serialization checks. Scopes overlap and are not added into a unique test count. The final learner suite includes explicit teacher gain-mismatch and scan-unroll cases. The comparison movie uses the corrected historical harm scene, first handcrafted versus learned frozen, then learned frozen versus learned adaptive, with the same physical setup and full recorded trajectory exposure. It shows the failure as well as the success.

The first physical run completed before a NumPy-valued summary exposed a JSON writer error. A sealed I/O-only amendment authenticated and reused that completed attempt without rerunning physics, then executed the remaining 35. Its numerical source hashes equal the preceding protocol. Failed preliminary probe reports, old/new protocol source snapshots, process/cache metadata, and the writer failure log are retained. Earlier published results and videos are untouched.

The [compact evidence package](../artifacts/da_plcbf/numerical-stabilization-20260906/v1/publication-v1/README.md) provides the reports, all 36 summaries/bindings, 12 causal-pair analyses, probe records, tests, sealed sources, and SHA-256 inventory. Large dense/event/rollout/checkpoint arrays and movies remain local and are hash-indexed; the package is for review, not a complete replay archive. The [flight table](../artifacts/da_plcbf/numerical-stabilization-20260906/v1/analysis-v1/flight_source.csv) and [analysis](../artifacts/da_plcbf/numerical-stabilization-20260906/v1/analysis-v1/report.json) expose every completed result.
