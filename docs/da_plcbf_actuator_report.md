# Actuator-aware DA-PLCBF research report — working draft

This report covers saved development evidence inspected on 2026-09-06. The actuator-aware plant, filter, learner, comparison drivers, and provenance checks are implemented. The completed initial development budget has **128 method episodes on 64 paired physical worlds**, with mixed outcomes across the two scenario families. Fixed-state recovery improves combined-fault tracking in three library seeds, but fails the declared worst-case tracking and braking thresholds. The available paced experiments complete no post-fault learner update. These results support a bounded implementation and learning result; they do not yet establish comparative safety or real-time adaptive benefit.

The 384-episode main study is a prepared protocol input at this writing, not a completed or sealed result. The explicitly marked main, causal, secondary, figure, and video slots below require authenticated final artifacts. Earlier summaries remain useful provenance, but their statements that the initial development budget is unfinished are superseded by the completed budget reported here.

## 1. Starting and final code versions; exact modifications

The starting reference is clean `main` commit `8c0320617a6fae56d8fa7179349ed4c07311a04c`; development uses `codex/actuator-study`. The [starting manifest][start] inventories 1,164 accepted-study files totaling 486,440,662 bytes, including the accepted video polish. The working tree still has that HEAD and uncommitted actuator-study additions at this report checkpoint. There is therefore no final committed study revision to report yet. Each completed campaign instead binds its own exact retained source, specification, and checkpoint digests. Final protocol preparation must bind the committed source actually selected for the main study.

The [implementation map][implementation] documents reuse and changed assumptions. The main additions are:

| Area | Implementation | Resulting behavior |
|---|---|---|
| Physics and allocation | [actuator_dynamics.py][dynamics-code], [actuator_independent.py][independent-code] | Separate algebraic and positive-lag models, bounded motor-command allocation, continuous actuator-state events, and independent effort/RPM plants. |
| Safety and optimization | [actuator_plcbf.py][filter-code], [actuator_opt.py][opt-code] | Motor-coordinate finite-horizon collision row, held-command nonlinear checks, operational margins, rescue modes, and a bounded multistart OPT comparator. |
| Learning | [actuator_learning.py][learning-code] | Full augmented-state teacher/student rollouts, immutable teacher references, persistent Adam state, finite update publication, repertoire preparation, and DR preparation. |
| Execution and computation | [actuator_experiment.py][runtime-code], [actuator_inputs.py][inputs-code], [actuator_compute.py][compute-code] | Deterministic, paced, and delayed execution contracts; dense physical evidence; exact observation-input caching; packed controller transport; and completion/publication accounting. |
| Study and analysis | [study driver][study-driver], [protocol driver][protocol-driver], [analysis driver][analysis-driver] | Paired world/checkpoint bindings, fresh attempt directories, reviewed protocol preparation and sealing, retained-attempt ledgers, and paired analysis that refuses incomplete matrices. |
| Reporting | [figure renderer][figure-driver], [video renderer][video-driver], this report | Artifact-derived plots and source tables, validated paired physical replay, and explicit inclusion boundaries. |

The shared swept-geometry utility in `continuous_version_a.py` was extended for the new derivative/transport path while preserving its legacy default. The old 13-state direct-wrench rollout is not used as the new 17-state effort plant. A runtime accounting fix also credits arrival at the true final physical boundary, without issuing another command or learner call; physical collision retains precedence. Its [boundary regression cases][runtime-tests] cover arrival at the reach threshold and contact at the same boundary. A read-only audit of 13 then-existing runtime summaries found 12 completed episodes and no archived result affected by this fix; the five incomplete-task endpoints remained outside the 0.4 m reach radius.

The accepted [wind regression][wind] retains exact equality for all 44 shared non-timing array comparisons, including dense states and commands. The fixed trace reaches modeled collider intersection at 4.64 s; the adaptive trace completes both waypoints at 10.52 s with 0.1724472846 m lower collider clearance. The accepted stronger compensated frozen comparator also survives, and that qualification remains attached to the wind result. This is preservation of the existing anchor, not new actuator-study evidence.

Regression status is explicitly incomplete. The original non-render suite selected 1,813 tests and deselected 21 render tests, then exited 134 after 743 pass, 6 fail, and 7 skip outcomes during JAX compilation. Targeted reruns reproduced all six example/harness failures in immediate source paths byte-identical to the starting commit. Fresh-cache and cache-disabled competent-library runs each passed 16 of 17 tests and aborted during the estimated-controller compilation; the latter exposed LLVM allocation errors. Subsequent per-file execution was paused to avoid contention with timing experiments. Across retained attempts, deduplicated by test node, the current coverage is **1,092 pass, 7 fail, 9 skip**. The seventh failure concerns a resume source digest during concurrent benchmark-source writing; that is a candidate explanation, not a proven diagnosis. This report does not mark the full suite as passing.

## 2. Physics/control contract and independent validation

The [physical contract][contract] defines body state `x=[p_world,q_xyzw,v_world,omega_body]` and augmented state `y=[x,s]`, with four nominal-equivalent effort states in newtons. Motor command `u` and internal effort `s` are distinct. The positive-lag model is

```text
ds/dt = (u-s)/tau
actual motor force = eta*s
body wrench = B*(eta*s)
```

Effectiveness is applied once, to each motor's thrust and associated reaction torque. The unchanged repository motor order and mixing matrix define signs. All methods share the original physical command bounds; no extra command authority compensates for lost effectiveness. The 13-state algebraic effectiveness model is separate and never divides by a time constant. Its model label A1 is distinct from the benchmark's adaptive single-fallback method, also called A1.

Held motor effort uses the analytic exponential endpoint. Body RK4 stages evaluate evolving effort at the start, midpoint, and end; they do not apply end-of-hold force throughout the interval. Motor endpoints are analytic up to arithmetic, whereas the body trajectory retains integration error. Fault onset preserves every body and motor state. Effectiveness changes may change force instantly; lag changes derivatives. Neither event resets motors to a new trim or reveals a future fault schedule.

The [parameter derivation][parameters] gives nominal `tau=0.059498771217 s` from the mean native spin-up/down poles at hover. Symmetric operating-region values vary from about 0.076229437 to 0.051844801 s. These are reductions of repository coefficients, including historical TODOs, rather than hardware measurements. The [parameter atlas][parameter-json] contains 38 static trim cases: 36 proposed-support cases admit the specified coupled hover, with minimum per-motor headroom 0.048015066 N; two deep 0.40-effectiveness controls fail that hover. A missing local maneuver witness remains `witness_not_found`, not a proof that all flight is infeasible.

F2 allocation uses current effectiveness, lag, and motor state to target the desired motor endpoint over the shared 40 ms hold. With the declared natural wrench normalization and orthogonal mixing columns, independently clipping the inverse allocation is the exact minimizer of that specified bounded least-squares objective. This conclusion does not extend to arbitrary mixing or wrench priorities. Allocation residuals and saturation remain diagnostics, not certificates that an infeasible request was recovered.

The filter predicts 60 steps at 20 ms, or 1.2 s, with two integration steps per feedback hold. For a frozen model and policy snapshot, its motor-command collision row is

```text
a = -grad_s(H)/tau
b = partial_t(H) + grad_x(H)·body_rhs - grad_s(H)·s/tau + alpha*H
a·u <= b
```

The metric uses normalized physical commands. Full nonlinear held-command checks and at most two sequential QP refinements enforce six arena faces, speed, body angular rate, and tilt without enlarging bounds. Normal fallback, emergency brake, and nominal proposals all use the same actuator adapter. Hard clearance, smooth value, eligibility, QP feasibility, executed multiplier, and actual physical outcome remain separate fields.

The primary safety enclosure has radius 0.106 m with requested clearance 0.15 m. Actual collision uses the rotating XML sphere of radius 0.086 m, body offset `(0,0,0.02)` m, and the ground plane. Obstacles follow saved continuous analytic trajectories, including absolute phase and velocity. Interpolated swept checks, physical integration resolution, enclosure clearance, and actual-collider clearance are not interchangeable. The [theory note][theory] establishes a conditional finite-horizon geometric implication, including attitude-induced offset-collider error. Average training loss does not supply its uniform trajectory-error premise, and the note does not prove BPTT convergence, recursive safety through policy/model switches, or infinite-time safety.

P0 is the matched differentiable surrogate. P1 independently codes continuous body/effort dynamics and integration. P2 uses native RPM states, asymmetric motor response, thrust/torque curves, and rotor inertia. P2's exact RPM-to-effort mapping is a declared observation oracle. Historical focused validation records 22 physics/allocation tests and 28 independent-plant tests; these overlapping development counts are not added to the current regression total. Independent full-flight and model-mismatch conclusions require the secondary evidence in Section 7.

## 3. Baseline competence, tuning, and information fairness

The [behavior report][behavior] uses 27 development states and 16 disjoint validation states, including motor transients. The main repertoire has K=16 fallback policies, a separate nominal candidate, hidden width 32, and the shared F2 adapter.

| Prepared deployment | Seeds | Finite updates per seed | Student integration steps per seed | Competence evidence |
|---|---|---:|---:|---|
| Nominal repertoire | 11, 23, 37 | 128 total | 368,640 | All three pass nominal development and validation checks. |
| Independently trained DR repertoire | 11, 23, 37 | 512 | 1,474,560 | All three pass nominal validation and 3/4 DR validation models. |
| Single braking fallback | 11 | 128 | 23,040 | Passes nominal development and validation checks. |

Seed 11's nominal budget consists of the selected 32-update preparation plus 96 persistent updates, not 128 additional updates. The continuation preserves optimizer history. Seeds 23 and 37 run 128 updates from version zero. Student integration counts exclude teacher rollouts, anchor caching, validation, and RK body stages. The original 32-update bootstrap failed the unchanged 0.8 m/s terminal-speed threshold at 0.82570 m/s. The retained paired repair used absolute terminal-speed squared during preparation and reached 0.76371 m/s; deployed adaptation retains the original excess-braking objective and immutable competent teacher.

The [DR record][dr] uses 64 development dynamics samples, four validation models, independent per-motor effectiveness in `[0.7,1]`, and lag in `[1,3]` times nominal. Each seed has its matching nominal teacher and separately initialized actor/Adam state. The declared validation score selects step 512 from checkpoints 0/128/256/384/512 for all three seeds. DR still fails the hardest combined validation model; broad competence throughout its randomized support is not established.

In the main comparison, F2 and A must start from the identical complete nominal checkpoint; DR uses its separate dr512 checkpoint with the same teacher reference. OPT shares the same state/model information, geometry, command constraints, and task. The main model observation is current-state/current-parameter oracle information available equally to the methods. It contains no future fault schedule, and it does not demonstrate online identification. New test geometry seeds are fresh scenes within declared dynamics support; they are not automatically out-of-distribution dynamics.

The [Gate D campaign][gate-d] runs F2, DR, A, and OPT for a full 8 s on the same nominal empty-obstacle validation world. All four finish both waypoints without recorded collider or operational violation. A credits 199 finite updates under the deterministic schedule. This is a bounded competence check, not an obstacle-performance comparison. The [OPT microbenchmark][opt] contains four one-decision cases at 40 and 200 ms budgets; its explicit 0.025 m ego sphere prevents promotion to actual-collider episodes. OPT is a finite-horizon multistart controller without a terminal safe-set guarantee.

## 4. Completed development budgets and retained negative outcomes

The [initial budget summary][development-budget] records **128 completed method episodes**, **64 physical worlds paired across F2/A**, **16 geometric seeds**, and library seed 11. The original pilot completes 32/32 episodes; the expansion completes 96/96. Each family contributes eight geometry seeds crossed with four dynamics cells, yielding 32 physical worlds and 64 method episodes per family.

| Family | Method | Episodes | Actual collisions | Safe task completions | Operational violations | Waypoints completed |
|---|---|---:|---:|---:|---:|---:|
| Structured | F2 | 32 | 8 | 21 | 1 | 45 |
| Structured | A | 32 | 10 | 20 | 1 | 42 |
| Navigation | F2 | 32 | 12 | 20 | 3 | 109 |
| Navigation | A | 32 | 5 | 24 | 2 | 116 |

The result is mixed: A has fewer collisions and more safe task completions in navigation, while its structured outcomes are worse on those two counts. These are development observations from one library seed, after which methods and candidate cases were still being examined. They supply neither frozen-test inference nor a general superiority claim. Collision and operational-violation columns can overlap. Safe task completion requires full task and outcome conditions; a full-duration timeout is not a task success.

The uncontended 32-episode pilot takes 442.2412 s including warmup. The expansion takes 1,248.0928 s while CPU regression work is concurrent. Expansion timing is excluded from latency and efficiency inference. The proposed 384-episode budget is a separate main-study plan, not an enlargement of this completed development count.

The [recovery matrix][recovery] contains 12 obstacle-free fixed-bank runs: three seeds crossed with nominal, effectiveness-only, lag-only, and combined dynamics, each with 128 finite updates. These are 1,536 matrix updates, not flight episodes. Six current development states cycle with two rotating retention anchors; all 16 held-out states remain evaluation-only. The earlier seed-11 combined diagnostic and the separate retention10 repair each add a distinct 128-update diagnostic and are not additional independent seeds.

| Seed | Combined F2 max position RMSE (m) | DR (m) | A after 128 updates (m) | F2 / A max terminal speed (m/s) |
|---:|---:|---:|---:|---:|
| 11 | 0.7939 | 0.5217 | 0.3707 | 1.5300 / 0.9833 |
| 23 | 0.7972 | 0.4474 | 0.3662 | 1.5466 / 0.9762 |
| 37 | 0.7916 | 0.5360 | 0.3733 | 1.5433 / 0.9748 |

Combined tracking improves beyond F2 and DR for all three seeds but remains above the 0.30 m position-RMSE and 0.8 m/s braking limits. RMSE averages squared xyz coordinate errors at five declared prefix nodes; terminal speed is a Euclidean norm. Effectiveness-only adaptation worsens F2's passing 0.6648–0.6690 m/s terminal speed to 0.9119–0.9449 m/s. Lag-only F2 is already competent; adaptation slightly worsens tracking. No-change adaptation retains competence with small teacher drift.

The [four-pair gain screen][gains] retains the original attitude/rate gains: only that pair passes nominal competence, and its declared development score is 4.1766 versus 10.2340, 24.3864, and 54.0652. The [fixed-bank retention repair decision][retention] rejects weight 10 against weight 5: development score worsens from 0.19682161 to 0.23179055 and held-out terminal speed from 0.98330319 to 1.00735915 m/s, despite a small held-out position-RMSE improvement. This is retention-strength evidence, not a retention on/off ablation.

The later closed-loop repair check adds **13 completed method episodes** in separate campaigns: four [original-method negative-case replays][negative-original], two [weight-10 negative cases][negative-retention10], one [weight-10 positive control][positive-retention10], two [weight-10 validation cases][validation-retention10], and four [contemporaneous original-weight oracle validation cases][secondary-oracle]. These are targeted development/validation follow-ups and are not added to the initial 128-episode sample or the main-test denominator.

| Case | Original weight 5 | Weight 10 | Interpretation |
|---|---|---|---|
| Structured 1003, nominal | New A replay finishes two waypoints at 12.32 s; F2 also succeeds. | A finishes two at 13.60 s. | Earlier nominal contact does not reproduce; no repair credit. |
| Structured 1004, combined | A collision reproduces at about 5.27213 s; F2 completes safely. | A completes both waypoints at 12.84 s without collider or operational violation. | Positive, localized repair evidence. |
| Structured 1002 positive control | Retained original positive case remains separate. | A completes both at 11.56 s without collider or operational violation. | Repair preserves this positive-control outcome. |
| Structured validation 2003 | F2 and A each reach 1/2 waypoints; no collider or operational violation. | A reaches 0/2 and violates operational constraints; no collider contact. | Validation task and operational harm. |
| Navigation validation 2103 | F2 and A each reach 2/4 waypoints and violate operational constraints; no collider contact. | A also reaches 2/4 and violates operational constraints; no collider contact. | No validated improvement. |

The selected method remains **retention weight 5 with the original gains and objective**. The clean 1004 repair is retained, but the structured validation harm and worse fixed-bank selection score reject weight 10 for the main study. No new retention10 primary checkpoint or training budget is substituted for the nominal128 F2/A/OPT and dr512 deployments.

The [learner compute parity record][parity] also preserves rejected optimizations. Unroll 2 reduces mean update time from 18.179 to 15.652 ms for the original objective, but fails gradient/next-Adam and strict forward parity. A declared C1 Huber variant reduces 18.283 to 15.837 ms and passes three gradient/next-Adam probes, but still fails strict forward parity in one body-rate coordinate. Tolerances are not relaxed. Selected defaults remain unroll 1 and zero Huber delta; no rejected learner proposal is installed.

## 5. Frozen main results with paired uncertainty

**Pending main result:** no sealed protocol, completed main ledger, or primary confidence interval is represented by this draft. The [prepared inputs][protocol-inputs] declare eight family/dynamics cells, four distinct physical worlds per cell, three library seeds, and F2/DR/A/OPT: **384 method episodes**. The main plan uses matched P0, a 5 ms physical step, a 40 ms command period, and deterministic execution. It is a bounded mini benchmark, not the plan's larger default budget or a power guarantee.

The proposed primary metrics are actual collision, safe task completion, operational violation, lower actual-collider clearance, and controller deadline-miss rate. The builder creates A-minus-F2, A-minus-DR, and A-minus-OPT comparisons for all five metrics and eight cells: 120 primary comparisons. Its proposed 95% family confidence, Bonferroni correction, 20,000 percentile resamples, and seed 20260906 imply individual interval confidence `0.9995833333333333`. Physical worlds and library seeds are both cluster axes. Each method/cell contains only four independent worlds crossed with three library seeds; the 12 episodes must not be presented as 12 independent world draws.

This small world count produces limited precision. The draft budget rationale illustrates this with a two-sided 95% Wilson upper bound of about 0.49 after zero events in four worlds. That illustration is not the prespecified paired primary interval. Final reporting must retain per-seed effects, world-only sensitivity, crossing support, signed effect direction, multiplicity, and all attempts. The analysis refuses incomplete required matrices. It may not drop failed, missing, or censored attempts to manufacture a complete denominator.

Final insertion requires the sealed protocol digest, committed source identity, checkpoint digests, retained-attempt coverage, exact completed/failed/missing counts, every primary estimate and interval, and links to the resulting analysis and source tables. Development counts from Section 4 cannot fill this slot.

## 6. Causal mechanism and the role of the repertoire

The current fixed-bank evidence shows that persistent learning can reduce combined-fault tracking error from a shared full augmented-state initialization. It does not show that a policy update completes after a fault and changes a later executed action in time to avert collision. Repertoire size, teacher retention, optimizer persistence, and fault compensation are separate possible contributors.

The nominal structured-1003 replay is a material numerical limitation. The earlier expansion A trace collided at about 5.257 s, whereas the saved new replay completes both waypoints at 12.32 s with matching physical specification, checkpoint, controller source, and learner source. The traces already differ slightly at the first float32 step. The combined-1004 negative A contact is much more stable: 5.2721325 s in the replay versus 5.2721195 s earlier.

The [GPU execution parity investigation][execution-parity] finds identical raw bytes and abstract/device properties for all 26 observation, 36 controller-input, and 62 learner-input leaves, including weak types, layouts, and sharding. Two fresh function copies have identical StableHLO and optimized HLO. All 12 controller outputs and 12 complete learner outputs agree across reference/cache inputs, repetitions, and copies, and match the new replay.

The follow-up [cached-executable comparison][cache-comparison] identifies the source of the initial arithmetic drift: the original development compilation cache and the newer cache contain identical StableHLO but different optimized controller and learner HLO. The original cache exactly reproduces the original expansion's first controller record, first-update parameter digest, and gradient; the newer cache exactly reproduces the newer replay's first controller and learner outputs. Within each cache, reference/cached observation inputs and repeated outputs are identical. This directly attributes the initial divergence to optimized-executable/cache drift, rather than observation caching or unexplained within-executable nondeterminism. It does not establish the unique causal sequence from that first difference to the later collision/success outcome, and it is not a new physical episode.

The main execution decision preserves a copy of the original development compilation cache, matching the numerical realization used for the initial 128 development episodes; the canonical cache and environment manifest must accompany the final source/protocol provenance. Both executable variants and both physical outcomes remain retained as numerical-fragility evidence. The nominal case is not promoted as an authenticated common-prefix repair or causal result.

The [direct-command development][witness-dev] and [held-out witnesses][witness-val] help localize the learning limitation. Offline solves with 30 held commands find the two hard targets after 12 and 13 iterations from the first F2 initialization. Fine 2 ms replay gives position RMSE 0.242444/0.059421 m and terminal speed 0.509388/0.313035 m/s. These numerical witnesses show those two probed targets are achievable; they are not real-time controllers, general feasibility proofs, or evidence that the learned repertoire recovers them.

**Pending causal result:** the selected case must bind the same world, initial state, model, checkpoint, and actual published snapshot times across its original, freeze, hold, revert, no-fault, refinement, and neighboring-world records. Report the first outcome divergence and the relevant executed policy/action, hard and smooth clearance, eligibility, feasibility, multiplier, and physical margins. A retained negative or non-robust case supports a limitation and must remain visible. Isolated learner completion times cannot be substituted for snapshot availability in a full episode.

**Pending repertoire/retention result:** the K=1 deployment is competent at seed 11, but that alone does not establish the necessity or benefit of K=16. Fill this slot with matched A1/F2/A episodes and the separately fingerprinted zero-retention contract. The completed weight-5/weight-10 screen is not zero-retention evidence. Single-world or single-library-seed subsets are descriptive and cannot support the crossed main-study confidence interval.

## 7. Actual runtime and independent-plant evidence

The complete [GPU Gate B][gate-b] development cases run for 8 s without obstacles, 14 s with static obstacles, and 14 s with moving obstacles. All three finish two waypoints without recorded actual-collider or operational violation. Their 900 controls have zero controller-service deadline misses. The moving case records 55 negative nominal hard collision predictions and 43 executed positive policy-row multipliers, demonstrating that the filter is exercised. The earlier CPU warmup failure remains an incomplete time-zero attempt with no safety or task credit.

Full reverse AD controller probes cost about 45.2–45.6 ms; [full forward AD][forward] reduces isolated device service to 13.23–13.96 ms. These warmed same-state profiles exclude parts of complete scheduling. The [input-cache parity record][cache-parity] passes **1,056 exact comparisons**, requiring identical tree structure, shape, dtype, and raw bytes, including signed zero. Cases cover physical/model float32 and float64, faults and recovery, neighboring event times, delayed observations, biases/noise, and repeated/out-of-order queries. The cache only materializes model phases actually requested at the current or past observed time.

The [GPU input profile][cache-profile] has 160 exact paired queries. Mean synchronized input-preparation time falls from **1.286 ms to 0.281 ms**; p95 falls from **1.419 ms to 0.293 ms**. This measures input preparation, not complete controller service or adaptive safety. Runtime integration has separate deterministic equivalence tests for physical records, controller inputs, finite updates, and final learner state.

Three completed paced validation variants retain the same 14 s fault world and original objective; the fault begins at 2 s with effectiveness `(0.85,1,1,1)` and lag multipliers `(3,1,1,1)`.

| A paced variant | Finite published updates | Training / publication time (s) | Controller mean / p95 / max (ms) |
|---|---:|---|---|
| [Default guard][paced-default], factor 1.25 and 3 ms reserve | 0 | None | 21.546 / 26.626 / 31.047 |
| [Measured guard][paced-guard], factor 1.05 and 1 ms reserve | 1 | 1.04 / 1.08 | 21.518 / 26.280 / 34.842 |
| [Measured guard with exact input cache][paced-cache] | 1 | 0.08 / 0.12 | 19.519 / 22.553 / 27.653 |

Every variant completes both its F2 and A episodes through 14 s without recorded collider or operational violation, but each completes only one of two waypoints. All six have zero controller-service deadline misses. The cache-backed A update takes 23.426 ms; its summary records one learner deadline miss, one credited finite update, no uncredited finite updates, and final version 129. **None of these variants completes a post-fault learner update.** Their completed collision-free timeouts do not demonstrate real-time fault recovery. Paced serialized scheduling with nondelayed fixed physical holds and delayed-command plant evolution are distinct execution contracts.

The [CPU P1 sanity episode][cpu-p1] runs through 4 s without a recorded collider or operational violation and completes only one of two waypoints. Independently, the two saved body witnesses were replayed by a NumPy P1 implementation at 2 ms:

| Witness | P1 position RMSE (m) | P1 terminal speed (m/s) | Maximum position difference from fine JAX replay (m) |
|---|---:|---:|---:|
| [Development state 26, skill 2][witness-dev-p1] | 0.242442 | 0.509372 | 0.00000984 |
| [Validation state 7, skill 2][witness-val-p1] | 0.059419 | 0.313039 | 0.00002361 |

Both independent replays pass their declared witness thresholds. They validate two saved held-command body trajectories; they are not obstacle-safety episodes or learned-policy transfer results.

The completed four-episode [oracle validation baseline][secondary-oracle] is reported in Section 4: all trials are collider-free, all fail full task completion, and both navigation methods violate operational constraints. This is the matched baseline for the remaining secondary comparisons, not evidence of adaptive improvement.

**Pending secondary result:** insert the remaining fine-P0/P1/P2, observation-bias/noise/delay, paced, and delayed-command subsets with exact matched world/checkpoint support. P1/P2 comparisons must use the matching fine-P0 integration baseline; otherwise plant level and step size change together. Timing must separate controller computation, mandatory host work, observation/transfer, learner service, simulator/audit work, completion/publication/use, and held-command age. Mean episode p95 is not pooled controller p95. The small secondary subsets remain descriptive. No current result establishes P2 adaptive safety transfer or hardware validity.

## 8. Supported claims, unsupported claims, and unresolved limitations

The saved evidence supports an actuator-aware implementation with explicit force/command semantics, tested bounded allocation and independent dynamics, complete nominal controller sanity episodes, persistent recovery learning, and retained source/checkpoint provenance. It supports three-seed combined-fault tracking improvement on a fixed obstacle-free bank, a mixed closed-loop development result, exact observation-input equivalence with reduced preparation cost, and the failure of the tested paced schedules to deliver a post-fault update.

The evidence currently does not support broad collision or task superiority, efficiency superiority, successful online system identification, necessity of K=16, zero-retention benefit, real-time post-fault adaptation, broad native-plant transfer, hardware validation, or an unconditional/infinite-time safety guarantee. Static hover feasibility, two local maneuver witnesses, and prediction clearance cannot replace these missing results.

Remaining limitations include the development dependence of selected cases and tuning, only three learned library seeds, four planned main worlds per cell, incomplete DR competence at the hard combined model, degraded effectiveness-only braking, the fragile nominal replay under identified optimized-executable/cache drift, policy/model switch discontinuities, numerical integration and interpolation error, exact current-model/motor observation assumptions, incomplete regression coverage, and unmeasured causal benefit under the actual paced compute budget. The final report must retain negative development and repair outcomes even if a clearer video case is available.

## 9. Reproduction commands and artifact inclusion/exclusion manifest

Use the repository's `gpu-tests` environment and the exact source/checkpoint version bound by each campaign. The selected main numerical execution also preserves the original development compilation-cache copy and its environment manifest; source equality alone did not ensure identical optimized executables in the retained comparison. The following commands are documented entry points; they were not executed while writing this report. Choose a fresh output directory and apply the final bound execution environment. Source changes produce a different binding and must not overwrite or silently resume an old result.

```bash
ACTUATOR_STUDY=artifacts/da_plcbf/actuator-study-20260906/v1

# Reproduce a saved development specification in a fresh directory.
pixi run -e gpu-tests python -m benchmark.da_plcbf_actuator_study campaign \
  --specification "$ACTUATOR_STUDY/specifications/development-pilot-original-v1.json" \
  --output "$ACTUATOR_STUDY/reproduction-development-pilot-v1"

# Recreate review inputs; this does not seal or execute the main study.
pixi run -e gpu-tests python -m benchmark.da_plcbf_actuator_protocol_inputs \
  --output "$ACTUATOR_STUDY/reproduction-protocol-inputs-v1"

# Build a draft only after the relevant source is committed and selected.
pixi run -e gpu-tests python -m benchmark.da_plcbf_actuator_protocol prepare \
  --campaign "$ACTUATOR_STUDY/reproduction-protocol-inputs-v1/main-campaign.json" \
  --development "$ACTUATOR_STUDY/reproduction-protocol-inputs-v1/development.json" \
  --validation "$ACTUATOR_STUDY/reproduction-protocol-inputs-v1/validation.json" \
  --decisions "$ACTUATOR_STUDY/reproduction-protocol-inputs-v1/decisions.json" \
  --output "$ACTUATOR_STUDY/reproduction-protocol-draft-v1.json"
```

The [protocol CLI][protocol-driver] separates `prepare`, `seal`, and `import`. After the final reviewed protocol and campaign exist, `import --protocol <sealed.json> --campaign <campaign-directory> --output <new-ledger-directory>` authenticates all retained attempts. The [analysis CLI][analysis-driver] then uses `ledger --protocol <sealed.json> --ledger <ledger.json> --output <new-analysis-directory>`. These placeholders are intentional: this draft has no completed sealed main artifact to name. Exploratory `campaign` analysis must retain its development status.

The [figure renderer][figure-driver] accepts a study root, explicit main campaign/analysis, an explicit secondary role map, and a fresh output directory. It must bind source artifacts and save exact source tables and a figure manifest. CI endpoints copied from a saved analysis require that analysis file's trusted digest; structural checks alone do not re-run or authenticate bootstrap calculations. The [video renderer][video-driver] accepts paired frozen/adaptive artifact directories and provides `--validate-only` before rendering.

| Artifact class | Included evidence | Exclusion or qualification |
|---|---|---|
| Starting anchor | Starting file inventory; accepted wind replay and stronger-comparator qualification | No legacy wind result is relabeled as actuator-aware. |
| Physics/theory | Contract, parameter atlas, independent tests, theory note, two independently replayed body witnesses | Static trim and local solves are not broad feasibility or safety proofs. |
| Baseline preparation | Nominal, DR, and single-policy manifests; exact teacher/actor/optimizer identities; finite update counts | No duplicated seed-11 preparation count; no teacher/validation work hidden inside student-step counts. |
| Development | Original 32 and expansion 96 method episodes; all recovery/gain/retention records | One library seed for the initial closed-loop budget; expansion wall time excluded from latency inference. |
| Failed/rejected work | CPU warmup and regression aborts; harness failures; serialization attempts; failed competence and parity candidates | No incomplete attempt receives completed-episode or successful-task credit. |
| Timing | Separate synchronized profiles and complete paced records, including zero/one-update outcomes | No isolated microbenchmark or deterministic episode is presented as complete paced schedulability. |
| Numerical execution | Both optimized-executable probes, StableHLO/optimized-HLO hashes, physical outcome variants, and selected original-cache execution provenance | Within-cache repeatability does not imply identical arithmetic across compiled executables; the fragile nominal case earns no repair credit. |
| Main and secondary | Pending authenticated sealed ledger and explicit secondary mappings | Specifications alone are not results; incomplete main matrices produce no primary inference. |
| Figures and media | Pending five artifact-derived figures, source CSVs/manifests, and numerically promoted paired video | Synthetic fixtures, rendering/contact continuations, and selected scenes are not additional experiment trials. |

**Pending final figures:** (1) main safety/task outcome; (2) behavior recovery over time; (3) safety versus compute; (4) repertoire and retention ablations; (5) model/observation mismatch. Every figure needs exact input hashes, sample support, execution/plant labels, uncertainty scope, and source CSVs. Missing datasets must be explicitly skipped or refused, never replaced with synthetic values.

**Pending final video:** choose only after numeric promotion and preserve the complete paired physical timeline. Use a motor-specific fault cue, stable policy identities and trajectory colors, a clear selected/executed path distinction, physically recorded body attitude, and minimal text. Do not add artificial tilt, alter obstacle motion, or splice phases without marking them. A contact/rendering continuation is presentation after outcome termination and remains excluded from controller performance. The video explains one retained case and cannot replace the population results or their limitations.

[start]: ../artifacts/da_plcbf/actuator-study-20260906/v1/STARTING_MANIFEST.json
[implementation]: da_plcbf_actuator_implementation.md
[wind]: ../artifacts/da_plcbf/actuator-study-20260906/v1/legacy-wind-regression-v1/REGRESSION.json
[contract]: da_plcbf_actuator_contract.md
[parameters]: da_plcbf_actuator_parameters.md
[parameter-json]: da_plcbf_actuator_parameters.json
[theory]: da_plcbf_actuator_theory.md
[dynamics-code]: ../crazyflow/safety/da_plcbf/actuator_dynamics.py
[independent-code]: ../crazyflow/safety/da_plcbf/actuator_independent.py
[filter-code]: ../crazyflow/safety/da_plcbf/actuator_plcbf.py
[opt-code]: ../crazyflow/safety/da_plcbf/actuator_opt.py
[learning-code]: ../crazyflow/safety/da_plcbf/actuator_learning.py
[runtime-code]: ../crazyflow/safety/da_plcbf/actuator_experiment.py
[inputs-code]: ../crazyflow/safety/da_plcbf/actuator_inputs.py
[compute-code]: ../crazyflow/safety/da_plcbf/actuator_compute.py
[runtime-tests]: ../tests/test_da_plcbf_actuator_experiment.py
[behavior]: ../artifacts/da_plcbf/actuator-study-20260906/v1/M3_BEHAVIOR_SUMMARY.md
[dr]: ../artifacts/da_plcbf/actuator-study-20260906/v1/M3_DR_THREE_SEED_SUMMARY.json
[gate-d]: ../artifacts/da_plcbf/actuator-study-20260906/v1/gate-d-original-v1/campaign_result.json
[opt]: ../artifacts/da_plcbf_actuator/opt_cpu_development_v1/final_shared_geometry/README.md
[development-budget]: ../artifacts/da_plcbf/actuator-study-20260906/v1/M4_DEVELOPMENT_INITIAL_BUDGET_SUMMARY.json
[recovery]: ../artifacts/da_plcbf/actuator-study-20260906/v1/M4_RECOVERY_THREE_SEED_SUMMARY.json
[gains]: ../artifacts/da_plcbf/actuator-study-20260906/v1/recovery-gains-v1/summary.json
[retention]: ../artifacts/da_plcbf/actuator-study-20260906/v1/RECOVERY_REPAIR_DECISION.json
[negative-original]: ../artifacts/da_plcbf/actuator-study-20260906/v1/negative-first-original-v2/campaign_result.json
[negative-retention10]: ../artifacts/da_plcbf/actuator-study-20260906/v1/negative-first-retention10-v1/campaign_result.json
[positive-retention10]: ../artifacts/da_plcbf/actuator-study-20260906/v1/positive-control-retention10-v1/campaign_result.json
[validation-retention10]: ../artifacts/da_plcbf/actuator-study-20260906/v1/retention10-validation-subset-v1/campaign_result.json
[secondary-oracle]: ../artifacts/da_plcbf/actuator-study-20260906/v1/secondary-oracle-v1/campaign_result.json
[execution-parity]: ../artifacts/da_plcbf/actuator-study-20260906/v1/input-execution-parity-gpu-v1/results.json
[cache-comparison]: ../artifacts/da_plcbf/actuator-study-20260906/v1/input-execution-cache-comparison-v1/results.json
[witness-dev]: ../artifacts/da_plcbf/actuator-study-20260906/v1/body-witness-development26-skill2-v1/summary.json
[witness-val]: ../artifacts/da_plcbf/actuator-study-20260906/v1/body-witness-validation7-skill2-v1/summary.json
[witness-dev-p1]: ../artifacts/da_plcbf/actuator-study-20260906/v1/body-witness-development26-skill2-v1/independent_replay/report.json
[witness-val-p1]: ../artifacts/da_plcbf/actuator-study-20260906/v1/body-witness-validation7-skill2-v1/independent_replay/report.json
[parity]: ../artifacts/da_plcbf/actuator-study-20260906/v1/M5_COMPUTE_PARITY_SUMMARY.md
[protocol-inputs]: ../artifacts/da_plcbf/actuator-study-20260906/v1/protocol-inputs-main384-draft-v1/verification.json
[gate-b]: ../artifacts/da_plcbf/actuator-study-20260906/v1/gate-b-gpu-sanity-v1/campaign_result.json
[forward]: ../artifacts/da_plcbf/actuator-study-20260906/v1/controller-profile-forward-v3/profile.json
[cache-parity]: ../artifacts/da_plcbf/actuator-study-20260906/v1/input-cache-cpu-parity-v1/parity.json
[cache-profile]: ../artifacts/da_plcbf/actuator-study-20260906/v1/input-cache-gpu-profile-v1/profile.json
[paced-default]: ../artifacts/da_plcbf/actuator-study-20260906/v1/paced-validation-default-v1/campaign_result.json
[paced-guard]: ../artifacts/da_plcbf/actuator-study-20260906/v1/paced-validation-measured-guard-v1/campaign_result.json
[paced-cache]: ../artifacts/da_plcbf/actuator-study-20260906/v1/paced-validation-input-cache-v1/campaign_result.json
[cpu-p1]: ../artifacts/da_plcbf/actuator-study-20260906/v1/gate-b-cpu-p1-v1/campaign_result.json
[study-driver]: ../benchmark/da_plcbf_actuator_study.py
[protocol-driver]: ../benchmark/da_plcbf_actuator_protocol.py
[analysis-driver]: ../benchmark/da_plcbf_actuator_analysis.py
[figure-driver]: ../benchmark/da_plcbf_actuator_figures.py
[video-driver]: ../benchmark/da_plcbf_actuator_video.py
