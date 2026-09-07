# Quadrotor actuator adaptation: diagnostic iteration

The review led to an objective diagnosis, a bounded braking repair, a handcrafted PD comparison, an immutable-library union, and a real asynchronous learner. **The available results do not establish consistently beneficial adaptation.** The braking repair restores declared competence on a fresh obstacle-free state bank, but still collides in the historical harmful scene. PD frozen and adaptive libraries both complete all four development cases; that tie does not show an adaptive benefit, and the frozen PD outcome changes in a separate measured build. Asynchronous execution publishes completed updates during flight, while its matched frozen/adaptive comparisons show no safety-category advantage for learning.

This is a separate follow-up to the [reviewed report](da_plcbf_actuator_report.md) at `c9a2b8e67f7a3908524942cb2c5bd802719d2610`. Its old studies, splits and conclusions remain intact. The working branch is **`actuator-learning-diagnostics`**, including the remote branch. The original diagnostic numerical source is `cc7847cd62cf00044c9317344ddf31ab1808108c`; each campaign additionally binds exact source and checkpoint bytes. The [supplied review][review] is retained verbatim.

The original sealed protocol has **144/144 unique completed trials**: 32 development comparisons, 40 additional factorial flights, 64 fresh-world trials and eight numerical-robustness flights. Four exact development reuses give 44 factorial stage memberships. The separate selector diagnostic and measured-runtime amendment each add eight completed physical attempts, and the secondary union-repair validation completes all 40, for **200 new physical flight attempts** across the explicitly separated protocols. **All 58 original continue/freeze pairs pass complete history, common-state repertoire and actual-command audits**, with no missing or rejected pairs. A completed attempt can be a collision or timeout; it is never automatically a successful flight.

## Experimental contract and selection

All new full-flight results use **P0**, the matched differentiable body/effort surrogate. The four commanded motor efforts and four evolving motor states are separate; effectiveness multiplies actual motor force once. The command period is 40 ms, predictions span 60 × 20 ms = 1.2 s, and all methods share command limits, the F2 endpoint adapter, the nominal task controller, the safety filter, and exact current-state/current-model information. Updating a policy under this model oracle is **not identifying unknown dynamics**. The primary outcome requires all waypoints, no modeled collider intersection and passing operational checks through each scene's declared exposure: **14 s for structured scenes and 24 s for navigation scenes**. Each pair shares its full exposure; a safe prefix or timeout is insufficient. Modeled collision is audited geometric intersection of the recorded XML collider, not a measured MuJoCo contact event.

The [sealed protocol][protocol] fixes two development geometries (`structured_30101`, `structured_61001`), eight fresh validation geometries, and library seed 11. The first geometry is explicitly **outcome-selected development**: the lowest indexed historical effectiveness-only case with harmful adaptation, using the lowest selected library seed. The second geometry and all eight validation seeds were fixed before new outcomes. Neither development geometry is held out. The nominal PD grid and the loss repair were chosen using development diagnostics before the fresh-world campaign was opened. The later incumbent-refresh selector diagnostic is a labeled **post-development amendment**, not a silently revised primary method.

| Original sealed stage | Design | Unique new flights | Status |
|---|---|---:|---|
| Library comparison | 2 development scenes × 2 cells × 8 methods | 32 | 32 complete |
| Matched factorial | 2 scenes × 11 distinct effectiveness/lag/lead cells × continue/freeze | 40, plus 4 exact reused comparisons | 44 complete memberships; 22 authenticated pairs |
| Fresh-world validation | 8 fresh scenes × 2 cells × 2 initialization families × continue/freeze | 64 | 64 complete; 32 authenticated pairs |
| Numerical robustness | Historical harmful scene × 2 fresh builds and 2 initial perturbations × continue/freeze | 8 | 8 complete; 4 authenticated pairs |
| Total original protocol | 148 stage memberships, with 4 duplicates | **144** | **144 unique completed trials** |

The fixed factorial varies effectiveness on motors 0 and 1 over `{1, .85, .7}`, their lag multiplier over `{1, 2}`, and fault onset between 2.0 and 0.4 s. Advancing onset adds 1.6 s of learning opportunity while retaining the same absolute obstacle paths, initial conditions and mission. Physical histories differ across onset cells; the **entire pre-fault history must match within each continue/freeze pair**. Nominal duplicate lead cells are collapsed. The freeze baseline uses the same learner, teacher, anchors, Adam moments, counters and published weights at fault onset; it is not startup-frozen F2. The eight validation worlds provide two prespecified within-world contrasts, one library seed and descriptive world-level evidence, without a broad population-superiority claim.

## What the learner diagnosis establishes

The [component-gradient and per-skill diagnosis][loss-report] begins at the complete nominal checkpoint, including persistent Adam. Eight 128-update arms cross nominal/effectiveness-only dynamics, legacy/initial-balanced loss and persistent/reset-at-onset Adam. Development and the original disjoint validation bank are evaluated at fixed states; unit-gradient probes are distinguished from actual Adam updates. These are obstacle-free recovery experiments.

At the competent nominal teacher, weighted diversity and pairwise gradient norms are **0.0330033 and 0.0109127**, versus **0.00008496** for motor effort. Teacher matching is therefore not an equilibrium of the legacy total objective, but the proposed motor-effort explanation is not the dominant measured term. Legacy effectiveness adaptation improves mean position RMSE on the original validation bank from **0.098023 to 0.047195 m** while worsening maximum terminal speed from **0.669050 to 0.911911 m/s**. The largest per-skill terminal-speed deterioration is **+0.252087 m/s cumulatively**, versus **+0.000159 m/s in the last update**. Resetting Adam once at fault onset gives **0.929121 m/s**, so momentum reset alone does not repair braking. These measurements identify objective interference; they do not individually establish the cause of a later obstacle collision. [Source rows][gradient-csv], [per-skill source artifacts][loss-summary].

The first balanced loss uses physical-unit normalization, smooth worst-skill aggregation, first-quarter prefix tracking and terminal recovery, and removes absolute effort, attitude, rate, saturation and diversity penalties. It retains trust and the existing retention weight of 5. At its effectiveness endpoint on development state 4, trajectory/braking gradients have cosine **−0.6280**, and total/braking gradients **−0.3234**: descending the total gradient locally worsens braking. The measured minimum total braking multiplier that makes this Euclidean dot product nonnegative is **8.771322**. A separately declared second objective fixes that multiplier at **10** before four further 128-update arms. There is no retention grid. This calibration is local; Adam preconditioning and other states can still change alignment.

The selected `balanced_reference_braking` mode is an objective-only fork: **75 numeric checkpoint leaves remain identical**, including parameters, optimizer history and references. A `1e-6` normalized-error dead zone suppresses teacher/current discrepancies within that threshold; the larger computation-path mismatch documented below still produces measured nominal PD optimizer drift. It is a loss numerical tolerance, not a relaxed controller safety tolerance. Fresh CPU H8 comparisons with the reviewed source match **392 numeric leaves across 12 rollout/loss/update cases**, and retain the old checkpoint fingerprint. That limited parity check does not imply full-flight or cross-build equivalence.

After the objective was fixed, all relevant endpoints were evaluated without further training on the new 16-state bank, seed 72606. Its changes include motion directions, attitude/rates and coupled motor transients. The table reports effectiveness-only learned-library terminal speed; the declared limit is 0.8 m/s.

| Endpoint | Persistent Adam (m/s) | Reset at onset (m/s) |
|---|---:|---:|
| Frozen F2 | 0.746960 | — |
| Legacy after 128 updates | 0.853384 | 0.870286 |
| Initial balanced loss | 0.802105 | 0.803119 |
| Selected braking ×10 loss | **0.790360** | **0.760543** |

Both selected-loss histories pass the declared tracking, braking and operational criteria on development and fresh states. Persistent mean position/velocity RMSE improves from **0.099673 m / 0.134901 m/s** to **0.052638 m / 0.089681 m/s**. Frozen F2 still brakes better, and the persistent repair's braking margin is only **0.00964 m/s**. This is bounded recovery of declared competence, not uniform behavior improvement or robust closed-loop safety. The earlier failed balanced endpoint remains in the record. [Exact fixed-bank rows][bank-csv].

![Fresh-bank braking results][braking-figure]

## Handcrafted versus learned libraries

The PD library contains 16 distinct velocity-and-braking primitives, including hover, axial/diagonal motion and vertical recovery. Its neural residual is identically zero. A nominal-only grid of six gain/duration choices selects gain **2.4** and duration **0.35 s** by the predeclared development score; all six pass nominal competence. The selected fixed library also passes the disjoint nominal validation bank, with maximum terminal speed **0.218180 m/s**. PD adaptation changes **64 trainable coordinates**: 48 velocity offsets clipped to ±0.5 m/s and 16 duration offsets. Effective durations are clipped to 0.1–1.2 s; the underlying duration offsets are not clipped. The parameter-update norm is capped at 0.025. The learned actor has 2,083 trainable parameters. [Nominal preparation and grid][pd-prepare].

PD and learned adaptive arms use the same selected braking objective and the same shared actuator/control contract. PD effectiveness adaptation improves fresh-state mean position RMSE **0.113665→0.084851 m**, mean velocity RMSE **0.155696→0.121886 m/s**, and maximum terminal speed **0.423074→0.328069 m/s**. Nevertheless, both fixed and adapted PD fail strict tracking in that bank; the adapted maximum position/velocity RMSE is **0.454837 m / 0.624030 m/s**. Each library tracks its own nominal teacher, so PD and learned tracking errors are not comparisons against identical maneuver targets. [PD recovery sources][pd-loss].

| Method | Initial library / online change | Fallback count |
|---|---|---:|
| F2 | Reviewed nominal learned library, frozen | 16 |
| A | Same complete initialization, legacy persistent adaptation | 16 |
| A_BAL | Same complete initialization, selected balanced braking adaptation | 16 |
| PD_F / PD_A | Matched handcrafted initialization, frozen / bounded adaptation | 16 |
| DR | Independently prepared domain-randomized learned library, frozen | 16 |
| UNION | Immutable original F2 core plus A_BAL current library | 32 |
| F2_2K | Immutable original F2 plus distinct immutable DR library | 32 |

The shared nominal candidate is additional to these counts. F2_2K is an equal-total-size frozen comparator using genuinely different policies; it does not duplicate F2 rows. Every finite adaptive update in the deterministic comparisons is published, including in the union, with no outcome rejection or rollback. Asynchronous completion and in-flight publication are accounted for separately below.

| Development method | Safe task completions / 4 | Collisions / 4 | Timeouts / 4 |
|---|---:|---:|---:|
| F2 | 3 | 1 | 0 |
| A | 1 | 3 | 0 |
| A_BAL | 2 | 1 | 1 |
| PD_F | 4 | 0 | 0 |
| PD_A | 4 | 0 | 0 |
| DR | 3 | 1 | 0 |
| UNION | 1 | 2 | 1 |
| F2_2K | 4 | 0 | 0 |

All 32 attempts complete their executed prefixes with operational checks passing. The matrix below retains all four cells. In the historical harmful effectiveness scene, F2 completes safely while legacy A, A_BAL and UNION collide. In the second scene's effectiveness cell, A_BAL times out after one of two waypoints; F2 and DR collide; both PD variants and F2_2K complete. **Four PD successes are promising development behavior, not evidence that online adaptation helped.** They also do not override the separate build-sensitive failure below. Development controller timing was collected with concurrent CPU regression activity and is not used as an uncontended latency measurement. [32 source rows][development-csv], [aggregate counts][development-aggregate].

![All development outcomes][development-figure]

## From changed skills to changed executed commands

The [authenticated harmful-flight audit][harm] finds that A, A_BAL and UNION have the same physical trajectory and applied commands as F2 through the 2.0 s fault, despite 50 calm updates. A's first applied motor-command divergence is at **2.40 s**, with maximum difference **0.000259168 N**, after 60 identical previous commands. A_BAL first diverges at **2.48 s**, **0.000333443 N**, after 62 identical previous commands. At each first divergence, the actual 17-state controller input and nominal command are identical and both methods select policy 12. The current library has changed the selected policy's CBF row: the nominal-command residual changes from F2 **+0.192141** to A **−0.060845**, or from F2 **+0.061242** to A_BAL **−0.076038**. The adaptive controller consequently clips a nominal command F2 can accept. QP and held checks pass in both cases. This is a direct command-level effect of different current policy functions at an identical input, without assuming a smaller training loss implies better recovery.

Union coverage does not guarantee union selection. At 2.48 s the immutable F2 skill 12 remains eligible with volume score **0.750772**; the adapted counterpart 28 scores **0.745952**. Its difference **0.004820** is smaller than the unchanged **0.02** incumbent hysteresis, so the selector retains the adapted skill. The pointwise maximum-value inequality for adding policies still holds, but the controller does not necessarily choose the old policy or reproduce its trajectory. The audit separates parameter movement before and after fault onset; the matched freeze/continue experiment below addresses the complete common pre-fault history.

The [three-library anchor audit][three-library] also separates startup F2, the complete fault-frozen library and the continued library at identical recorded states and current models. At 2.4 s their maximum valid fallback hard values are **0.475951 / 0.475911 / 0.483236**; terminal speeds of the corresponding best-value fallback are **0.631462 / 0.634832 / 0.605711 m/s**. All three best-value fallbacks have index 14, while the actual selected policy is 12. These are collision-geometry values of valid rollouts, not full QP/operational eligibility certificates. Improving the best unused policy's predicted clearance and braking therefore does not establish improvement of the executed trajectory. Six branch/time inputs and all per-skill motion prefixes are retained.

A bounded follow-up [selector amendment][refresh-protocol] resets the previous index only when an actually changed parameter snapshot had supplied the previous adapted policy. It leaves fixed-core incumbents, all current/frozen policies, scores, 0.02 hysteresis, QP checks and publication rules intact. It is a diagnostic heuristic, not a safety theorem or a guaranteed bug fix. Eight new development flights compare A_BAL_REFRESH and UNION_REFRESH in the same four cells:

| Scene / cell | A_BAL → A_BAL_REFRESH | UNION → UNION_REFRESH |
|---|---|---|
| 30101, effectiveness .7 | Collision → collision | Collision → safe task |
| 30101, nominal | Safe task → safe task | Safe task → safe task |
| 61001, effectiveness .7 | Timeout → safe task | Collision → collision |
| 61001, nominal | Safe task → **collision** | Timeout → safe task |

All eight physical/application prefixes through fault onset match their originals exactly, and the original physical/control bindings are identical. The fixed harmful union case is repaired, while nominal A_BAL gains a collision. These are distinct variants: **UNION_REFRESH preserves its original development success and adds two task successes, reaching 3/4**, whereas **A_BAL_REFRESH introduces nominal harm**. Neither replaces the original fresh-world hypotheses. Two metadata writer failures occurred after completed physical flights; two I/O-only amendments retained their results and reused those attempts. **Exactly eight unique physical flights ran, without rerunning the first two.** The original amendments and failed writer outputs remain inventoried. [Complete amended results][refresh-results], [byte and prefix audit][refresh-analysis].

The separately [sealed **40-flight secondary validation**][secondary-protocol] of UNION_REFRESH uses eight genuinely new scenes (`structured_63001`–`63004`, `navigation_63101`–`63104`) × nominal/effectiveness × continue/onset-freeze, plus eight fixed-anchor fresh-build/perturbation trials. Its purpose is to test the particular union repair that preserved development successes, with source, loss, checkpoints and methods fixed before opening these new worlds. It is an outcome-selected secondary hypothesis, separate from the completed original 144-trial protocol, with a fixed stop after these 40 flights irrespective of outcomes. No result from the earlier eight validation worlds is reclassified as new held-out evidence. The proposal digest is `41097523e67f53dbfbf3b8ce9344262b838fdc76148b76b7423c68189b04f458`.

All **32 fresh-scene flights are complete**, and all **16 pairs authenticate the complete physical, learner and selector-wrapper history** through the fault. Their paired full-task results are:

| Secondary fresh-world condition | Both succeed | Continuation alone | Onset-freeze alone | Both fail |
|---|---:|---:|---:|---:|
| Effectiveness .7 | 4 | 0 | 3 | 1 |
| Nominal | 5 | 0 | 1 | 2 |

Continuation adds **zero task gains and four losses** across these 16 cells. The three fault-cell losses occur in `navigation_63101`, `63103` and `63104`; the nominal loss occurs in `navigation_63103`. In the fault cell, continuation has four safe tasks, three collisions, one timeout and two operational violations, versus seven safe tasks and one collision for onset-freeze. Nominal continuation has five safe tasks and three collisions, versus six safe tasks, one collision and one timeout for onset-freeze. Collision and operational counts overlap. Collision-free paired categories are 10 both-pass, zero continuation-only, four freeze-only and two both-fail. The repeated nominal/fault cells share eight world geometries. Thus, the selected union repair's development gains do not translate into a continuation advantage on these new scenes. [Authenticated secondary fresh-world outcomes][secondary-fresh-analysis].

The remaining **eight robustness flights also complete**, reaching both waypoints safely through 14 s in both arms for each of two fresh builds and the positive/negative initial perturbations. These are four both-success ties between **refreshed continuation and refreshed onset-freeze**. They show that post-fault continuation is unnecessary for success in these realizations. They do not repeat the original UNION-versus-UNION_REFRESH comparison, because both secondary arms use the repaired selector. All 40 sealed identities have exactly one claimed and retained physical attempt, with no exceptions or outcome-driven extra trials. All **20/20 pairs pass the complete original and selector-memory history checks**, and all **120 common-state evaluations** include the full 32 fallback policies plus the nominal candidate, with the immutable core identical in both evaluations. The diagnostic maximum over both repertoires never decreases and increases in 33/120 rows (maximum 0.009693861), yet continuation adds no task successes on the new scenes. This pointwise coverage result does not guarantee useful selection or closed-loop improvement. [Final secondary analysis][secondary-analysis]. [Complete secondary trial ledger][secondary-results].

## Online updates that actually become available

The asynchronous implementation runs one real worker with persistent sequential Adam, immutable state copies and immutable completed snapshots. At most one submitted or unconsumed result exists; completion is timestamped after numerical work finishes. The controller applies the previously held command until its next result is actually available. Physical progression includes the measured service and propagation/audit costs. Late terminal learner work is recorded without in-flight publication or flight credit. The existing **3 ms paced reserve remains unchanged**. Asynchronous freeze means stopping new launches while already submitted work may publish; the exact fault-onset causal experiment therefore uses deterministic freeze semantics.

The [amended runtime protocol][runtime-protocol] runs four methods in paced and asynchronous modes on the same historical harmful world, one attempt per cell, using **NVIDIA GeForce RTX 4090, JAX 0.11.1 and CUDA**. It records an existing AnyDesk desktop context using 0 MiB GPU memory; no competing experimental GPU computation is observed in the retained process samples. This is a desktop timing experiment, not a hard real-time guarantee. The initial unsupported GPU-backend attempt and an overly strict desktop-context preflight failed before any flight and are retained separately. The final eight physical attempts bind protocol v3 even though their output directory is named `runtime-measured-v2`.

| Method / mode | Outcome | In-flight publications | Controller calls | Controller p95 (ms) | Learner jobs / p95 (ms) | Sensing→application p95 (ms) |
|---|---|---:|---:|---:|---:|---:|
| F2 paced | Safe task | 0 | 350 | 22.251 | 0 / — | 0.000 |
| F2 asynchronous | Collision, 4.351 s | 0 | 109 | 20.198 | 0 / — | 26.558 |
| A_BAL paced | Safe task | 0 | 350 | 22.115 | 0 / — | 0.000 |
| A_BAL asynchronous | Collision, 4.415 s | 59 | 111 | 19.604 | 60 / 16.349 | 31.399 |
| PD_F paced | Collision, 4.805 s | 0 | 121 | 21.125 | 0 / — | 0.000 |
| PD_F asynchronous | Safe task | 0 | 350 | 20.093 | 0 / — | 25.254 |
| PD_A paced | Collision, 4.805 s | 0 | 121 | 21.670 | 0 / — | 0.000 |
| PD_A asynchronous | Safe task | 257 | 350 | 19.728 | 258 / 15.355 | 30.355 |

All eight have zero recorded controller service deadline misses; asynchronous runs have zero skipped sensing ticks. Nevertheless, variable completion latency produces maximum actual command holds of **50.109–55.322 ms** across asynchronous runs, exceeding the 40 ms sensing period. The service-deadline result therefore does not establish a 40 ms maximum physical hold. In-flight publication differs from job completion: each adaptive asynchronous run has one completed result that was not published before termination. Before 2.0 / 2.40 / 2.48 s, A_BAL has **25 / 30 / 31** publications, of which **0 / 4 / 5** were trained after the fault; PD_A has **36 / 42 / 43**, of which **0 / 5 / 6** were trained after fault. These are strict-before counts at declared reference landmarks, not evidence of a universally necessary update deadline. The first obstacle-center crossing is 4.900453 s; contact can occur earlier, so its label is not “first threat.” [Source samples, counts and denominators][runtime-audit], [compact runtime rows][runtime-csv].

Controller/learner host service intervals overlap in **0/111 A_BAL calls**, and **3/350 PD_A calls**, totaling **29.64245 ms** for PD_A. Its controller mean is **27.313 ms** during those three overlapping calls versus **16.744 ms** across the remaining 347. This tiny observational subset neither isolates contention causally nor demonstrates concurrent GPU kernels. Published-snapshot age p95 is **120.008 ms** for A_BAL and **119.996 ms** for PD_A. Reduced parameter count alone does not make PD learning instantaneous: measured learner p95 remains **15.355 ms**, versus **16.349 ms** for the learned actor.

Real online updates are now available, but none of these four matched within-mode frozen/adaptive comparisons changes the success/collision category in favor of adaptation. Different modes also have different physical command-delay semantics: paced zero delay is part of that simulated execution contract, not an assertion of instant real actuation. Cross-mode success changes cannot establish scheduling superiority. [Runtime implementation and tests][async-code].

![Runtime publication and controller service measurements][runtime-figure]

The separate [controller-cost experiment][controller-cost] measures **300 calls**, 30 per method/input, after 30 retained disposable warmup calls. It uses F2's recorded physical state/model/goal/previous policy at 0 and 2.4 s, with obstacle/safety arrays reconstructed once from the bound geometry and shared as fixed bytes. At 2.4 s each method uses its own authenticated published parameters, while immutable cores remain the original F2 library. A sealed rotating/reversing order balances all five methods; no learner or flight runs during these calls.

| Method | Fallbacks | Median at 0 / 2.4 s (ms) | p95 at 0 / 2.4 s (ms) |
|---|---:|---:|---:|
| F2 | 16 | 13.494 / 13.477 | 14.082 / 13.818 |
| A_BAL | 16 | 13.502 / 13.484 | 13.731 / 13.957 |
| PD_F | 16 | 13.379 / 13.452 | 13.805 / 13.944 |
| UNION | 32 | 19.933 / 19.838 | 20.277 / 23.028 |
| F2_2K | 32 | 19.870 / 19.924 | 22.681 / 20.197 |

UNION/F2 mean service ratios are **1.471 and 1.513** at the two inputs; UNION/F2_2K ratios are **0.985 and 1.025**, with paired medians **0.997 and 1.002**. Enlarging the repertoire takes **47–51% more mean service time** in these warm calls, while the adapted union and equally large frozen library have similar costs. All 300 return valid QP decisions without sequential refinement/rescue, remain below **23.767 ms**, and reproduce the same output hash within each fixed cell. This is a narrow ordinary-QP path profile, not a worst-case controller bound. The measured window includes **0.44 CPU seconds** across matched nonself processes; 35 GPU observations show no other experimental computation. These preloaded-input costs exclude the full control-loop observation, scheduling, propagation and audit work, so the approximately 13.5 ms figures do not establish that serialized learning fits the paced budget. [Exact source groups][controller-cost-csv].

## Numerical fragility is a separate failure mechanism

The [frozen parity audit][parity-audit] compares the safe deterministic development PD_F flight to the collided paced PD_F flight. Physical scene, checkpoint, model, actor/filter, source and recorded device input dtypes match. The first internal nominal certificate difference is **5.96e-8 at 0 s**. The first nominal/applied motor-command difference is **1.49e-8 N at 0.16 s**, with exactly equal recorded controller inputs. Dense state first differs by **7.45e-9 at 0.165 s**; controller state input first differs at 0.20 s. The first capture-grid timestamp discrepancy occurs only at 2.8 s and cannot explain this onset. Obstacle-prediction and safety arrays were reconstructed from the bound scene rather than logged directly; byte equality is asserted only for the retained inputs, not for every original device argument.

At the 2.0 s fault both PD_F runs choose policy 8, but the paced run rejects the QP on the strict policy-row check and applies an emergency action; development accepts its QP. Other recorded feasibility, KKT, held-collision, operational and motor checks pass. The applied command difference becomes **0.013087347 N**, and the final outcome changes from safe task to collision at 4.805 s. F2 exhibits an analogous QP/emergency difference at 3.20 s, **0.005456164 N**, while both of its compared flights finish safely.

Development used a persistent compilation cache and measured runs compiled afresh. This evidence supports **unresolved numerical execution sensitivity with a discrete acceptance/selection effect**; it does not isolate a particular compiler transformation or kernel as its cause. The controller acceptance tolerance has not been weakened to remove the discrepancy. The completed fresh-build and ±1e-5 initial position/velocity checks below also change a baseline's outcome, and any favorable video retains this robustness limitation.

The [read-only QP roundoff audit][qp-roundoff] identifies a scale mismatch that can contribute to such rejection. At the PD fault decision, the normalized-coordinate row norm is **58.1016493**: the full solver's `2e-6` normalized primal tolerance corresponds to approximately **1.16203e-4** in raw row units, while the executed command must pass the stricter raw `bound − row·command ≥ −2e-6`. The bound's float32 spacing is **3.8146973e-6**, larger than the final tolerance; a single motor-coordinate ULP contributes about **3.2502762e-6** to that row. Stronger fast-path guards also exist. The rejected QP command, its exact residual and its solver branch were not saved, so the recorded **−6.411830902** residual of the executed emergency action is not mislabeled as the rejected QP residual. A future bounded inward projection could be tested only with all existing physical, feasibility, KKT and nonlinear held checks retained. No such repair has been implemented or credited in these results.

## Matched factorial and fresh-world results

The factorial completes **44/44 stage memberships**, comprising **40 new flights and four byte-authenticated development reuses**. All **22 pairs pass the complete pre-fault snapshot/array and actual applied-command audits**, with no missing or inadmissible pair. The two outcome definitions lead to different boundaries:

| Outcome over 22 matched development pairs | Both succeed | Adaptation alone | Onset-freeze alone | Both fail |
|---|---:|---:|---:|---:|
| Safe full-duration task completion | 7 | 1 | 8 | 6 |
| Collision-free observed physical prefix | 12 | 3 | 3 | 4 |

The only task-completion gain is `structured_61001`, effectiveness .7, lag ×2, with fault at **0.4 s** (1.6 s earlier than the 2.0 s condition). The other two collision-free gains still fail to complete their tasks. Continued adaptation yields eight task losses, including five cases whose frozen and adaptive trajectories both remain collision-free. The experiment does not support a monotone improvement from more severity or earlier learning. [All authenticated pair records][factorial-analysis], [compact pair source table][factorial-csv].

In the canonical execution of the historical harmful effectiveness scene, both continued A_BAL and its exact fault-onset freeze collide, whereas startup-frozen F2 had succeeded. Post-fault learning is unnecessary for that canonical failure category; the common pre-fault adaptation already loses the F2 outcome in this execution. The no-change twin succeeds with both continue and freeze. This result qualifies the first-command-divergence mechanism: seeing the first different command after a fault does not prove that post-fault optimizer steps were necessary for the eventual collision. **The fresh builds below change the onset-freeze outcome**, so calm-history sufficiency is not generalized across numerical executions.

![Matched factorial outcomes][factorial-figure]

All **22 factorial common-state comparisons are complete**, evaluating both published repertoires at the same saved physical inputs on both branches, at fault onset and 0.4/0.8 s later. No new flight or learner update is run. Saved per-skill arrays include 0.4 s motion prefixes, and actual applied commands are separately authenticated. The diagnostic maximum over continue/onset-freeze policies is a pointwise analysis quantity; it is not the deployed UNION method's immutable original-F2 core.

For the sole task-gain pair, adaptation raises the maximum fallback hard value by **0.000751257 at 0.8 s** and **0.000052094 at 1.2 s**; its first actual command diverges much later, at **3.12 s**, by **0.013429664 N**, with policies 12 versus 14. At that actual boundary the onset-frozen row rejects the nominal command (host-recomputed residual **−2.33694**, QP multiplier **0.00299997**) while the adaptive row accepts it (**+2.14344**, zero multiplier). Thus adaptation permits the unchanged nominal task command in this gain case.

The first lexicographic freeze-only task pair (`30101`, effectiveness .85, lag ×2, fault 2.0 s) has a **+0.005638599** hard-value change at 2.4 s and **−0.000102878** at 2.8 s. Its commands also first differ at **3.12 s**, by **0.005475618 N**, although both choose policy 10. Again the frozen row restricts the nominal command (**−1.74292**, multiplier **0.00072467**) while the adaptive row admits it (**+1.40360**, zero multiplier); here adaptation loses the task. Both actual QPs and held checks pass. These outcome-selected examples show that more permissive rows and early maximum-value gains can precede either task outcome; in the original harmful anchor, adaptation instead restricts a nominal command F2 admitted. [Complete common-state audit][factorial-common], [source values and per-skill hashes][common-csv].

The [first-command boundary audit][first-boundaries] covers **all 58 original pairs**, with exact retained state/observation/model hashes, prior command, goal and nominal command at their first divergence. There are **36 same-index and 22 changed-index decisions**. It authenticates actual application/control indices, QP mode and held flags separately from host float64 row-residual reconstruction. Host reconstructions are not relabeled as original GPU acceptance predicates, and missing full parameter snapshots at later first-divergence times are not synthesized. This bounds the command-level explanation without attributing a future outcome to one scalar alone.

All **64 fresh-world trials are complete**, with A_BAL/PD_A choices fixed before opening the validation outcomes. Four structured geometries have 14 s exposure and four navigation geometries have 24 s; paired arms share their scene's duration. The adverse refresh results were not substituted into this denominator. All **32 pairs pass complete common-history, common-state repertoire and actual-command authentication**, with no missing or inadmissible pairs. [Fresh common-state source][fresh-common].

| Fresh-world condition | Method | Safe task / 8 | Collisions / 8 | Timeouts / 8 | Operational violations / 8 |
|---|---|---:|---:|---:|---:|
| Effectiveness .7 | A_BAL onset-freeze | 5 | 3 | 0 | 1 |
| Effectiveness .7 | A_BAL continued | 6 | 1 | 1 | 0 |
| Effectiveness .7 | PD_A onset-freeze | 5 | 3 | 0 | 0 |
| Effectiveness .7 | PD_A continued | 2 | 5 | 1 | 0 |
| Nominal | A_BAL onset-freeze | 8 | 0 | 0 | 0 |
| Nominal | A_BAL continued | 6 | 1 | 1 | 0 |
| Nominal | PD_A onset-freeze | 5 | 3 | 0 | 1 |
| Nominal | PD_A continued | 8 | 0 | 0 | 0 |

Collision and operational-violation counts can overlap. The same eight world geometries appear in both cells and both initialization families; these rows are not independent sets of eight fresh draws. The paired full-task categories are:

| Fresh-world condition / initialization | Both succeed | Adaptation alone | Onset-freeze alone | Both fail |
|---|---:|---:|---:|---:|
| Effectiveness .7 / learned | 5 | 1 | 0 | 2 |
| Effectiveness .7 / PD | 1 | 1 | 4 | 2 |
| Nominal / learned | 6 | 0 | 2 | 0 |
| Nominal / PD | 5 | 3 | 0 | 0 |

Learned adaptation's one fault-cell task gain occurs in `navigation_62104`, with no fault-cell task losses across these eight worlds; nominal continuation adds two losses. PD adaptation has one fault-cell gain and four losses, despite tying its frozen baseline on all four development cells. Its nominal continuation gains three tasks. These conditional results do not establish population superiority or show that a stronger disturbance consistently makes adaptation useful. Across the 32 matched pairs there are 17 both-task-success, five adaptation-only, six freeze-only and four both-fail outcomes; the collision-free categories are 19 / 6 / 4 / 3. The within-condition tables retain the relevant geometry and initialization dependence. [All fresh-world summaries][fresh-campaign], [authenticated pair audit][fresh-analysis], [compact paired rows][fresh-csv].

The nominal PD gains require particular care. A [read-only drift audit][pd-nominal-drift] selects the lexically first of those three gains, `navigation_62104`, explicitly as an outcome-selected mechanism diagnostic. Its initial student and teacher parameters, current/teacher models and configurations match, with zero Adam steps. Nevertheless, its first logged loss is **1.90747e-8**, gradient norm **1.18930e-4**, and parameter-update norm **0.00790608**. Cumulative parameter displacement is **0.01634765** at 2.0 s and **0.01687790** at 2.8 s; onset-freeze holds the former value. The first applied command differs at **3.8 s** by **5.1409e-7 N**, at the same physical state and selected policy 2. This establishes nominal optimizer drift despite no model change; these nominal gains are not evidence of fault-specific recovery.

A [sealed, pure GPU loss-origin audit][pd-loss-origin-audit] localizes a source of that spurious nominal signal. Without an optimizer step or new flight, production `value_and_grad` reproduces the entire first recorded loss-metric tree exactly, including total **1.90746956008e-8**; its host-reduced gradient norm is **1.18929519e-4**. Teacher and student rollouts are **byte-identical across all three states and all eight rollout leaves** in separately compiled probes with matching batching and dynamic arguments. Closing over constant teacher inputs produces different numerical results. The original separately evaluated current-state teacher differs from the student by up to **3.40939e-5 m** in position, **6.80089e-5 m/s** in velocity and **1.09941e-4 rad/s** in angular rate. States first differ at prediction time **0.02 s** and commands at **0.04 s**. The instrumented full anchor teacher reproduces the production cached anchor states exactly, so the cache is checked directly. Feeding the student's identical, stopped-gradient rollout as reference gives **exactly zero loss and every gradient leaf zero**. [All graph outputs and raw measurements][pd-loss-origin].

The result identifies differing numerical computation paths as a source of nominal teacher/student mismatch, without a model, skill, parameter or hidden-penalty difference. It does not identify a particular compiler transformation or prove that this signal alone caused the task gain. The original gradient vector was not retained, and this audit does not recreate the optimizer-fused flight executable. Even plain loss and `value_and_grad` differ by **3.12284e-12** in total loss, reinforcing that instrumentation and compilation context matter. A future repair should match teacher/student batching and dynamic arguments, then revalidate controller cost and full flights; no such repair is credited here.

The [robustness launcher][robust-launcher] completes all eight trials in **four distinct processes**. Two fresh-cache directories were absent before launch; their before/after inventories and subprocess IDs are retained. The two perturbation processes reuse the first build's cache and change initial x position and x velocity together by **±1e-5 m and ±1e-5 m/s**, respectively. The summaries show:

| Harmful anchor realization | Continued A_BAL | Exact onset-freeze |
|---|---|---|
| Canonical development/factorial execution | Collision, stop 4.45 s | Collision |
| Fresh build 1 | Collision, stop 4.49 s | Safe full task |
| Fresh build 2 | Collision, stop 4.45 s | Safe full task |
| Positive initial perturbation | Collision, stop 4.46 s | Safe full task |
| Negative initial perturbation | Collision, stop 4.43 s | Safe full task |

The continued adaptive collision is stable across the four new realizations, while the baseline changes from collision to success. The paired task category consequently changes from canonical both-fail to freeze-only success. This supports a robust adverse adaptive outcome on these particular executions, **not numerical robustness of the full causal comparison**. It also prevents elevating the canonical calm-history explanation into a general mechanism. All **four new pairs pass complete common-history, common-state and actual-command authentication**; trial identities, process/cache provenance and original summaries are retained. [Robustness pair and common-state audit][robust-common].

## Authority, verification and scope

The [independent coupled-authority calculation][authority] finds an upright bounded hover trim for all six effectiveness/lag cells. At effectiveness .7, the minimum command margin is **0.048015 N** and maximum torque-free collective is **1.315920 × weight**. Lag changes transients rather than the static reachable wrench set. These results exclude “no hover authority” as an explanation for these cells; they do not prove recoverability from the actual fault state.

That distinction is tested explicitly. Preserving the nominal motor state at fault onset, then applying the new static trim command, produces large attitude excursions in independent P1 open-loop replays. A separately initialized, already settled post-fault trim remains at equilibrium. Six prescribed 40 ms held-command vertical maneuvers from the settled trim pass the declared witness checks: maximum terminal height-target error **0.000376 m**, tilt **0.007582 rad** and terminal speed **0.038848 m/s**; 2 ms versus 1 ms integration endpoints differ by less than **7.81e-10**. These are bounded maneuver witnesses from specified states, not obstacle-avoidance or fault-recovery proofs. Below-ground mathematical continuations in P1 have no contact model and are not counted as flights. The two older optimized witnesses use different per-motor faults and are explicitly excluded as witnesses for these cells.

A fresh focused CPU run at `cc7847c` passes **279 tests in 213.72 s**, selected from `tests/test_da_plcbf_actuator_*.py` and the actuator filter unit tests. The final combined run of twelve diagnostic-tooling test files passes **162 tests in 1.34 s**, with the exact command, source hashes and JUnit log retained. These counts overlap earlier development runs and are not summed into a new total. The original broad-suite failures and compiler abort were not rerun here and remain unresolved in the old report. [Original focused command, log and source revision][regression], [final diagnostic regression][final-regression].

The resulting contribution is a more explicit diagnosis and a functioning availability mechanism, with several negative controls. It is not yet a consistently beneficial online adaptation method. This iteration stays with the quadrotor. New full-flight evidence is confined to matched-model P0; the old P2 transfer failure remains unresolved. Finite-horizon QP/held checks do not establish recursive safety through parameter publication, policy restarts or model changes, and the learner does not estimate unknown dynamics.

## Reproducing and inspecting the evidence

The [table/figure generator][evidence-code] reads saved JSON/CSV without importing the controller or launching numerical runs. It verifies source-summary hashes, exact protocol identities and duplicate-trial equality. Figure values are read from exported CSV tables. Four report figures show all development outcomes, all fresh-state learned braking endpoints, all eight runtime attempts, and all matched-factorial pairs.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .pixi/envs/gpu-tests/bin/python -m benchmark.da_plcbf_actuator_diagnostic_evidence tables
```

The [compact review archive][archive] contains exact review text, protocols and amendments, initial checkpoints, JSON/CSV numerical sources, saved numerical source copies and these figures. Large rollout arrays, intermediate learner arrays and movies remain local with explicit sizes and hashes in the [complete inventory][inventory]. This is a review bundle, not a self-contained full numerical replay bundle. The [manifest][archive-manifest] gives the archive checksum and exact included/omitted sizes. Verification checks every archive member without extracting it; optional local verification also checks omitted raw files. Extract the archive at the repository root to resolve its repository-relative evidence links.

```bash
PYTHONPATH=. python -m benchmark.da_plcbf_actuator_diagnostic_evidence verify \
  --output artifacts/da_plcbf/actuator-diagnostics-20260906/v1/publication-v1
```

The [comparison video][comparison-video] shows full 14 s matched frozen/adapting learned and PD comparisons on the preselected harmful geometry, including the adaptive learned collision. Both native MuJoCo segments contain 281 audited frames from 0 through 14 s. The joined movie is **562 frames, 28.1 s, 20 fps, 1600 × 900**, and passes a full decode and frame-count check; its SHA-256 is `68714c0e781fbef74cf78bd5c28cd742495fe9ead3a7a30d6c96032dea7eca8e`. A terminated panel retains its last executed physical pose; it does not invent a post-contact trajectory. The PD development success is displayed with the numerical-build limitation above. [Frame provenance and video verification][video-verification].

[review]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/review.md
[protocol]: ../artifacts/da_plcbf/actuator-diagnostic-20260906/v1/protocol-sealed-v1.json
[loss-report]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/loss-diagnosis-summary-v1/diagnosis.md
[loss-summary]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/loss-diagnosis-v1/summary.json
[gradient-csv]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/report-evidence-v1/tables/gradient_source.csv
[bank-csv]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/report-evidence-v1/tables/fixed_bank_source.csv
[braking-figure]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/report-evidence-v1/figures/02_braking_recovery.png
[pd-prepare]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/pd-prepare-v1/summary.json
[pd-loss]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/pd-loss-braking-diagnosis-v1/summary.json
[development-csv]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/report-evidence-v1/tables/development_source.csv
[development-aggregate]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/report-evidence-v1/tables/development_aggregate.csv
[development-figure]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/report-evidence-v1/figures/01_development_outcomes.png
[harm]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/harm-flight-diagnosis-v1/report.json
[three-library]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/oldharm-three-library-common-states-v1/report.json
[refresh-protocol]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/incumbent-refresh-protocol-v3.json
[refresh-results]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/incumbent-refresh-results-v3/results.json
[refresh-analysis]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/incumbent-refresh-analysis-v1/report.json
[secondary-protocol]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/secondary-refresh-validation-v1/protocol-sealed-v1.json
[secondary-fresh-analysis]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/secondary-refresh-validation-v1/cpu-analysis-v1/fresh_validation/report.json
[secondary-results]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/secondary-refresh-validation-v1/results-v1/results.json
[runtime-protocol]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/runtime-protocol-v3/protocol.json
[runtime-audit]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/runtime-measured-v2-audit-v1/report.json
[runtime-csv]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/report-evidence-v1/tables/runtime_source.csv
[runtime-figure]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/report-evidence-v1/figures/03_runtime_availability.png
[controller-cost]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/controller-cost-measured-v1/report.json
[controller-cost-csv]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/report-evidence-v1/tables/controller_cost_source.csv
[async-code]: ../crazyflow/safety/da_plcbf/actuator_async.py
[parity-audit]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/runtime-frozen-parity-audit-v1/report.json
[qp-roundoff]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/runtime-qp-roundoff-audit-v1/report.json
[authority]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/authority-v1/summary.json
[regression]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/focused-regression-v1/summary.json
[final-regression]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/final-diagnostic-regression-v1/summary.json
[factorial-analysis]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/matched-factorial-analysis-v1/report.json
[factorial-csv]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/report-evidence-v1/tables/matched_factorial_pairs.csv
[factorial-figure]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/report-evidence-v1/figures/04_matched_factorial.png
[factorial-common]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/matched-factorial-common-states-v1/report.json
[common-csv]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/report-evidence-v1/tables/common_state_source.csv
[first-boundaries]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/first-command-boundaries-v1/report.json
[fresh-campaign]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/fresh-validation-v1/campaign_result.json
[fresh-analysis]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/fresh-validation-analysis-v1/report.json
[fresh-common]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/fresh-validation-common-states-v1/report.json
[fresh-csv]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/report-evidence-v1/tables/fresh_validation_pairs.csv
[robust-launcher]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/numerical-robustness-v1/launcher.json
[robust-common]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/numerical-robustness-common-states-v1/report.json
[pd-nominal-drift]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/pd-nominal-drift-diagnosis-v1/report.json
[pd-loss-origin]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/pd-loss-origin-measured-v1/report.json
[pd-loss-origin-audit]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/pd-loss-origin-audit-v1/report.json
[comparison-video]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/comparison-video-v3/learned-and-pd-comparison.mp4
[video-verification]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/comparison-video-v3/video-verification.json
[evidence-code]: ../benchmark/da_plcbf_actuator_diagnostic_evidence.py

[secondary-analysis]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/secondary-refresh-validation-v1/analysis-v1/report.json
[archive]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/publication-v1/review-only.tar.gz
[inventory]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/publication-v1/inventory.json
[archive-manifest]: ../artifacts/da_plcbf/actuator-diagnostics-20260906/v1/publication-v1/manifest.json
