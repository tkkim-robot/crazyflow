# Actuator-aware DA-PLCBF research report

The frozen main study is complete: **384/384 method episodes, 384 retained attempts, no missing or incomplete trials, and all 120 prespecified primary intervals available**. Across the 96 episodes per method, F2 records 15 collisions and 77 safe task completions; A records 17 and 69, DR 16 and 73, and OPT 36 and 59. These pooled counts describe the tested matrix, not 96 independent physical-world draws. None of the multiplicity-adjusted A-versus-F2 or A-versus-DR intervals excludes zero. The result does not establish broad safety or task superiority for adaptation; its improvement in structured combined-fault scenes coexists with harm in structured effectiveness-only scenes.

This report covers saved evidence inspected on 2026-09-06. The separate initial development budget has **128 method episodes on 64 paired physical worlds**, and all **88 declared post-main method episodes plus four common-prefix audits are complete**. Fixed-state recovery improves combined-fault tracking in three library seeds, but fails the declared worst-case tracking and braking thresholds. In the selected positive case, onset-freeze succeeds and early parameter reversion causes contact: accumulated adaptation contributes, but post-fault learning is unnecessary for that outcome. Its adaptive safety contrast survives P1 and two P0 step refinements, but disappears under paced/delayed execution and native P2. Five artifact-derived figures and separate deterministic/paced videos retain these findings and limitations. Earlier summaries describing numerical work as unfinished are superseded by these completed counts.

## 1. Starting and final code versions; exact modifications

The starting reference is clean `main` commit `8c0320617a6fae56d8fa7179349ed4c07311a04c`; development uses `codex/actuator-study`. The [starting manifest][start] inventories 1,164 accepted-study files totaling 486,440,662 bytes, including the accepted video polish. The numerical source revision selected and sealed for the main study is **`377f59f942cbe9a0eaf313b4c8bce40ceeb8b003`**. The [sealed protocol][sealed-protocol] has digest **`402b466f9fcf294e4c828cf391d04a72fe7b5e0bb6d0e54dfb2e3e2daab7631d`**. Each earlier completed campaign retains its own exact source, specification, and checkpoint bindings; its results are not relabeled with the sealed source. A subsequent telemetry correction is distinguished below from these preserved numerical results. The final source/report revision is recorded as **`reviewer_inputs.reporting_revision` in the [publication manifest][publication-manifest]**, avoiding a self-referential commit hash in this report.

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

After all study episodes, the dense recorder's model-label lookup was corrected to use the strict physical event clock rather than the observation path's `1e-10` tolerance. The original candidate's row 424 at `1.9999999999999991` s contains the correct pre-event force but prematurely labels effectiveness and lag with their post-event values; the adjacent row at exactly 2.0 s has the correct post-event force. The [separate corrected-source replay audit][telemetry-parity] passes **400 required exact array comparisons** across two full F2/A replays, including physical states, commands, forces/wrenches, predictions, application clocks, and final complete learner state. Only dense effectiveness/time-constant labels change, at row 424 in each replay; measured timing differs and is excluded because these verification replays were contended. The corrected runtime SHA-256 is `389e03f6a7df920b70fa4c1748ffe250f4af26004978873dc4bba5541f4352d0`. These two verification flights are separate from both the 384 main and 88 post-main denominators, and no sealed archive was edited.

The initial [telemetry decision][telemetry-decision] assumed the adjacent complete states were byte-identical. The full-array audit corrects that assumption: non-attitude coordinates are exact, while quaternion renormalization changes one raw coordinate by up to `1.1920929e-7`, with normalized quaternion L2 difference `4.5332774e-12`. The renderer's legacy reader recognizes only an authenticated adjacent declared event within eight floating-point ULPs, consistent non-attitude states and normalized attitude at storage precision, and correct force limits on both sides. This narrow interpretation of retained telemetry does not rewrite the original arrays or reclassify the final source as reporting-only changes.

The accepted [wind regression][wind] retains exact equality for all 44 shared non-timing array comparisons, including dense states and commands. The fixed trace reaches modeled collider intersection at 4.64 s; the adaptive trace completes both waypoints at 10.52 s with 0.1724472846 m lower collider clearance. The accepted stronger compensated frozen comparator also survives, and that qualification remains attached to the wind result. The [final preservation audit][wind-final] verifies all **1,164 accepted files and 486,440,662 bytes unchanged**. This is preservation of the existing anchor, not new actuator-study evidence.

The [final regression summary][regression-summary] and [deduplicated coverage ledger][regression-ledger] account for **1,858 selected tests: 1,806 pass, six fail, 45 skip, and one incomplete due to a compiler abort**, with 21 render tests deselected. The actuator subset has **217 selected and 217 passing test nodes** across the retained execution history; this is deduplicated coverage across revisions, not one fresh run of every test on the final checkout. The [diagnosis][regression-diagnosis] retains exact commands, environments, historical failures and exclusions. The six example/harness failures also reproduce on a clean detached starting commit `8c03206` using the current dependencies, with the baseline package import and clean worktree verified. That establishes their presence in the starting source under this environment, without claiming that all dependency versions match a historical baseline test run.

The one incomplete legacy estimated-controller case still aborts when selected alone with the compilation cache disabled: LLVM reports `Cannot allocate memory` during controller warmup at `competent_library_experiment.py:722`, after 116.03 s, with signal/exit status `-6` (shell 134). The relevant source digest is unchanged. Earlier fresh-cache and cache-disabled competent-library attempts each passed 16 of 17 tests before aborting at that compilation. The earlier resume-source-digest failure now passes with stable before/after source hashes; it remains in the historical record rather than the final six-failure count. The earlier broad run selected 1,813 tests and exited 134 after 743 pass, six fail and seven skip outcomes; the subsequent 1,092-pass/seven-fail/nine-skip rollup is superseded by the completed accounting above. The full suite is therefore **not passing**: baseline-reproduced failures and one unresolved environment/compiler abort remain explicit.

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

The uncontended 32-episode pilot takes 442.2412 s including warmup. The expansion takes 1,248.0928 s while CPU regression work is concurrent. Expansion timing is excluded from latency and efficiency inference. The completed sealed 384-episode main study is separate from this development count.

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
| Structured 1003, nominal | Fresh-cache A replay finishes two waypoints at 12.32 s; F2 also succeeds. | A finishes two at 13.60 s in the same fresh-cache comparison. | No repair credit; the later canonical-cache replay restores the earlier contact, as detailed in Section 6. |
| Structured 1004, combined | A collision reproduces at about 5.27213 s; F2 completes safely. | A completes both waypoints at 12.84 s without collider or operational violation. | Positive, localized repair evidence. |
| Structured 1002 positive control | Retained original positive case remains separate. | A completes both at 11.56 s without collider or operational violation. | Repair preserves this positive-control outcome. |
| Structured validation 2003 | F2 and A each reach 1/2 waypoints; no collider or operational violation. | A reaches 0/2 and violates operational constraints; no collider contact. | Validation task and operational harm. |
| Navigation validation 2103 | F2 and A each reach 2/4 waypoints and violate operational constraints; no collider contact. | A also reaches 2/4 and violates operational constraints; no collider contact. | No validated improvement. |

The [final retention decision][retention-final] keeps **retention weight 5 with the original gains and objective**. The clean 1004 repair is retained, but the structured validation harm and worse fixed-bank selection score reject weight 10 for the main study. No new retention10 primary checkpoint or training budget is substituted for the nominal128 F2/A/OPT and dr512 deployments.

The [learner compute parity record][parity] also preserves rejected optimizations. Unroll 2 reduces mean update time from 18.179 to 15.652 ms for the original objective, but fails gradient/next-Adam and strict forward parity. A declared C1 Huber variant reduces 18.283 to 15.837 ms and passes three gradient/next-Adam probes, but still fails strict forward parity in one body-rate coordinate. Tolerances are not relaxed. Selected defaults remain unroll 1 and zero Huber delta; no rejected learner proposal is installed.

## 5. Frozen main results with paired uncertainty

The [imported ledger][main-ledger] and [coverage record][main-coverage] contain **384 planned and completed trials, 384 retained attempts, zero missing trials, and zero incomplete attempts**. The analysis status counts are 384 completed and zero censored, inadmissible, interrupted, or simulator-error attempts. A completed trial can end in physical collision or an unsuccessful task; completion here means an accounted-for terminal experiment. The [sealed analysis][main-analysis] supplies all **120 primary intervals, with zero refused comparisons**.

The [final inputs][protocol-inputs] and [sealed campaign specification][sealed-campaign] bind eight family/dynamics cells, four distinct physical worlds per cell, library seeds **11, 23, and 37**, and F2/DR/A/OPT. The sealed support contains **74 development worlds, four validation worlds, and 32 test worlds**; support cardinality is not a completed-episode count. Main execution uses matched P0, a 5 ms physical step, a 40 ms command period, the original retention weight 5, and the unchanged source revision `377f59f942cbe9a0eaf313b4c8bce40ceeb8b003`. F2/A/OPT use their bound nominal128 deployments and DR its bound dr512 deployment. This is a bounded mini benchmark, not the plan's larger default budget or a power guarantee.

Each cell/method entry below is **actual collisions / safe task completions / operational violations out of 12 episodes**. Safe task completion requires all waypoints, no modeled collider collision, and passing operational nodes through the full declared duration. Collision and operational-violation counts can overlap. Actual collision denotes audited modeled XML-collider intersection, not a measured MuJoCo contact event. [Episode rows][main-episodes] and [aggregate source data][main-aggregates] retain every cell and metric.

| Cell | F2 | DR | A | OPT |
|---|---:|---:|---:|---:|
| Structured, nominal | 3 / 9 / 0 | 2 / 7 / 0 | 3 / 5 / 0 | 3 / 8 / 0 |
| Structured, effectiveness | 0 / 12 / 0 | 3 / 9 / 2 | 5 / 6 / 1 | 9 / 3 / 0 |
| Structured, lag | 0 / 9 / 0 | 1 / 8 / 0 | 0 / 9 / 0 | 12 / 0 / 0 |
| Structured, combined | 6 / 6 / 0 | 2 / 10 / 0 | 1 / 11 / 0 | 5 / 7 / 0 |
| Navigation, nominal | 2 / 10 / 0 | 2 / 9 / 2 | 1 / 11 / 0 | 0 / 12 / 0 |
| Navigation, effectiveness | 2 / 9 / 2 | 2 / 10 / 1 | 2 / 8 / 4 | 3 / 9 / 0 |
| Navigation, lag | 1 / 11 / 0 | 2 / 10 / 0 | 2 / 10 / 1 | 2 / 10 / 0 |
| Navigation, combined | 1 / 11 / 0 | 2 / 10 / 1 | 3 / 9 / 0 | 2 / 10 / 0 |
| Descriptive total, each out of 96 | **15 / 77 / 2** | **16 / 73 / 6** | **17 / 69 / 6** | **36 / 59 / 0** |

The sealed primary metrics are actual collision, safe task completion, operational violation, lower actual-collider clearance, and controller deadline-miss rate. A-minus-F2, A-minus-DR, and A-minus-OPT are evaluated for all five metrics and eight cells: **120 primary comparisons**. The declared 95% family confidence and Bonferroni correction give individual confidence **99.958333%**, implemented with 20,000 crossed-cluster percentile resamples and resampling seed 20260906. These are approximate intervals; the nominal family-confidence target is not an exact finite-sample coverage guarantee. Physical worlds and library seeds are both resampling axes. Each cell contains only **four independent worlds crossed with three library seeds**; its 12 episodes, or the pooled 96 per method, are not independent world draws. No unregistered pooled confidence interval is introduced.

The following A-minus-F2 effects and adjusted intervals are in **percentage points**. Negative collision effects and positive safe-task effects favor A. Seed triples list separate effects for **11 / 23 / 37**. Values are rounded here; the [complete 120-comparison source table][main-paired] retains exact estimates, endpoints, world-only sensitivity intervals, seed effects, and support for every metric and comparator.

| Cell | Collision effect [adjusted interval] | Collision effects by seed | Safe-task effect [adjusted interval] | Safe-task effects by seed |
|---|---:|---|---:|---|
| Structured, nominal | 0.00 [0.00, 0.00] | 0 / 0 / 0 | -33.33 [-100.00, 0.00] | -25 / -50 / -25 |
| Structured, effectiveness | +41.67 [0.00, 100.00] | +25 / +50 / +50 | -50.00 [-100.00, 0.00] | -50 / -50 / -50 |
| Structured, lag | 0.00 [0.00, 0.00] | 0 / 0 / 0 | 0.00 [0.00, 0.00] | 0 / 0 / 0 |
| Structured, combined | -41.67 [-100.00, 0.00] | -50 / -25 / -50 | +41.67 [0.00, 100.00] | +50 / +25 / +50 |
| Navigation, nominal | -8.33 [-100.00, 75.00] | -25 / -25 / +25 | +8.33 [-75.00, 100.00] | +25 / +25 / -25 |
| Navigation, effectiveness | 0.00 [-75.00, 75.00] | -25 / 0 / +25 | -8.33 [-100.00, 75.00] | 0 / 0 / -25 |
| Navigation, lag | +8.33 [-75.00, 100.00] | 0 / +50 / -25 | -8.33 [-100.00, 75.00] | 0 / -50 / +25 |
| Navigation, combined | +16.67 [0.00, 95.84] | +25 / +25 / 0 | -16.67 [-95.84, 0.00] | -25 / -25 / 0 |

**No A-versus-F2 or A-versus-DR primary interval excludes zero**, across all five metrics. This is limited evidence, not proof of equivalence or absence of effects. The structured combined-fault benefit and effectiveness-only harm occur in all three library-seed effects, while navigation effects often change sign by seed. World-only sensitivity conditions on the three tested libraries rather than treating them as a sampled population. For example, the navigation-combined collision difference has world-only interval [0, 33.33] percentage points, versus crossed interval [0, 95.84]; retaining both exposes the weak precision associated with three learned seeds.

Ten A-versus-OPT adjusted intervals exclude zero: seven controller deadline-miss-rate differences, two lower-clearance differences, and structured-lag collision. The lower-clearance differences are **+0.079176 m [0.004802, 0.184272]** in structured combined and **+0.178621 m [0.174617, 0.182144]** in structured lag. Clearance and operational violation concern only the executed physical prefix; early collision stops later exposure. Structured lag has A collision count 0/12 versus OPT 12/12, producing an empirical bootstrap difference and interval of **-1 [-1, -1]**. That degenerate interval reflects identical observed paired outcomes on four worlds and three tested libraries, not general certainty or a zero-risk guarantee. Likewise, [0, 0] intervals elsewhere reflect the observed resampling support. The two-sided 95% Wilson upper bound after zero events in four independent worlds is about 0.49; this illustrates limited world coverage and is not the paired primary interval. The analysis's world-any-event Wilson statistic changes the estimand and conditions on the tested libraries.

F2, DR, and A record zero controller-service deadline misses. OPT's equal-episode mean miss rate is **0.2081894**; the seven adjusted A-minus-OPT deadline differences exclude zero in every cell except navigation nominal. This is a controller-service diagnostic. Deterministic execution does not enforce an equal complete wall-time budget for adaptation and OPT: eligible A updates run regardless of measured remaining wall-clock slack, whereas OPT retains its explicit 40 ms optimizer service budget. Main physical outcomes characterize that declared algorithmic schedule. Computational value at equal available computation additionally requires measured paced/delayed safety, task, and total service; these main comparisons do not establish H5 or real-time adaptive benefit.

The protocol binds source, checkpoint digests, method selection, and numerical execution evidence. The [retained-attempt table][main-attempts] includes all 384 terminal attempts. No failed, missing, or censored attempt was removed to complete a denominator, and no development episode from Section 4 enters the main matrix. The sealed analysis identifies its analysis source digest as `501ca505c38b9cce487c233dea9fd645ff65c0b98044f165bc9388768f5841f3` and binds the exact protocol and ledger inputs.

## 6. Causal mechanism and the role of the repertoire

The fixed-bank evidence shows that persistent learning can reduce combined-fault tracking error from a shared full augmented-state initialization. The completed intervention evidence below also establishes an outcome-relevant contribution from cumulative adapted parameters in one selected full episode. It does not establish that post-fault learning is necessary: freezing the available snapshot at fault onset still succeeds. Repertoire size, teacher retention, optimizer persistence, and fault compensation remain distinct possible contributors.

The nominal structured-1003 replay is a material numerical limitation. The earlier expansion A trace collided at about 5.257 s, whereas the saved new replay completes both waypoints at 12.32 s with matching physical specification, checkpoint, controller source, and learner source. The traces already differ slightly at the first float32 step. The combined-1004 negative A contact is much more stable: 5.2721325 s in the replay versus 5.2721195 s earlier.

The [GPU execution parity investigation][execution-parity] finds identical raw bytes and abstract/device properties for all 26 observation, 36 controller-input, and 62 learner-input leaves, including weak types, layouts, and sharding. Two fresh function copies have identical StableHLO and optimized HLO. All 12 controller outputs and 12 complete learner outputs agree across reference/cache inputs, repetitions, and copies, and match the new replay.

The follow-up [cached-executable comparison][cache-comparison] identifies the source of the initial arithmetic drift: the original development compilation cache and the newer cache contain identical StableHLO but different optimized controller and learner HLO. The original cache exactly reproduces the original expansion's first controller record, first-update parameter digest, and gradient; the newer cache exactly reproduces the newer replay's first controller and learner outputs. Within each cache, reference/cached observation inputs and repeated outputs are identical. This directly attributes the initial divergence to optimized-executable/cache drift, rather than observation caching or unexplained within-executable nondeterminism. It does not establish the unique causal sequence from that first difference to the later collision/success outcome, and it is not a new physical episode.

The [canonical compilation-cache manifest][canonical-cache] preserves 4,197 files totaling 233,052,534 bytes from the original development cache, matching the numerical realization used for the initial 128 development episodes. Its [environment record][canonical-environment] specifies the RTX 4090, driver 555.42.06, JAX/jaxlib 0.11.1, Python 3.14.7, disabled x64, lockfile digest, and launch environment. The sealed protocol binds these evidence digests. A [full-episode parity comparison][canonical-parity], binding 48 input files, finds exact equality of all compared control/dense arrays and final complete learner/Adam state across four canonical negative-case replays and their original development episodes. This establishes the tested canonical replay path beyond the initial-step probes. Both executable variants and both physical outcomes remain retained as numerical-fragility evidence; compatibility with this execution environment remains part of exact replay.

The [canonical nominal-1003 shared-state audit][negative-shared] localizes a cumulative learning effect without attributing it to the last update. At 3.96 s, the previously available 3.92 s snapshot and current snapshot execute the exact same nominal command with zero policy-row multiplier. Replacing the current parameters with the original deployment parameters at that same state would apply a correction with multiplier 0.00146844 and maximum per-motor command difference 0.00837717 N. Thus cumulative adaptation suppresses that original correction, while the immediately preceding finite update does not change the executed action at the probed boundary. The full causal path to collision remains unproven.

The separate [four-method candidate-1002 campaign][candidate-original] completes four full physical episodes: F2 and OPT collide before completing a waypoint; A and DR both complete two waypoints without collider or operational violation, at 11.28 and 11.52 s respectively. The [authenticated shared-state evidence][candidate-shared] links this case to executed control differences:

| A trajectory boundary | Original deployment at the same state | Actually available adapted snapshot | Maximum command difference (N) |
|---|---|---|---:|
| 2.8 s | Executed policy-row multiplier 0.00220442 | Nominal command, multiplier zero | 0.00737757 |
| 3.0 s | Accepted QP, multiplier 0.00205007 | Accepted QP, multiplier 0.00182051 | 0.000476897 |
| 4.2 s | Zero eligible policies; emergency execution | Candidate 12 is the sole eligible policy; accepted QP with executed multiplier 0.00647239 | 0.00876194 |

The comparisons authenticate saved source, physical state/model inputs, snapshot availability, recorded normal selection, and equivalence of the separately forced candidate audit. They demonstrate an executed learned-row contribution at 3.0 and 4.2 s and reduced intervention at 2.8 s. Candidate 12 means one eligible policy with index 12, not 12 eligible policies. DR's successful full episode prevents treating this selected case as unique evidence for online adaptation. These offline shared-state comparisons are not extra flight episodes and do not replace full-prefix authentication or the separate intervention, refinement, and neighboring-world experiments.

Six declared candidate interventions have completed their numerical episodes. All six pass operational checks over their executed prefixes; five complete both waypoints without collider intersection. Times below are task completion times except for the explicitly labeled collision-detection boundary. The unchanged original A episode completes safely at **11.28 s**.

| Candidate intervention | Recorded intervention boundary | Outcome |
|---|---:|---|
| [Freeze learning at onset][candidate-onset-freeze] | 2.0 s | Safe task completion at 11.32 s; 50 credited updates, final published version 178. |
| [Hold the early adapted parameters][candidate-early-hold] | 3.0 s | Safe task completion at 11.32 s; 75 credited updates. |
| [Restore original control parameters early][candidate-early-revert] | 3.0 s | Collider intersection detected at 4.58 s; 0/2 waypoints; 75 credited updates before freezing. |
| [Hold the late adapted parameters][candidate-late-hold] | 4.2 s | Safe task completion at 11.72 s; 105 credited updates. |
| [Restore original control parameters late][candidate-late-revert] | 4.2 s | Safe task completion at 11.28 s; 105 credited updates before freezing. |
| [No fault, adaptation enabled][candidate-no-fault] | Fault removed | Safe task completion at 11.76 s; 349 credited updates. |

All four common-prefix audits pass: [onset freeze through 2.0 s][prefix-onset], [no-fault control through 2.0 s][prefix-no-fault], [early hold/revert through 3.0 s][prefix-early], and [late hold/revert through 4.2 s][prefix-late]. Each authenticates the full published learner, optimizer and parameter snapshot, exact body/motor samples through the cut, and commands, forces, control decisions, model inputs and used versions before the intervention boundary. Commands and force labels at the cut are right-continuous and are excluded when they can legitimately change there. The comparisons use exact samples without interpolation; early and late reversions are verified nontrivial control-parameter changes.

Successful onset-freeze execution therefore establishes that post-fault learning is unnecessary for this selected safe outcome. The authenticated 3.0 s hold/revert contrast supports a safety contribution from the cumulative adapted parameters available then; it combines pre-fault and post-fault learning and cannot isolate the latter. Both late interventions succeed, so they do not establish a safety need for the 4.2 s adapted parameters. No-fault success alone does not isolate the effect of adaptation, and these selected-world conclusions do not establish outcomes on other worlds.

The first executed-command divergence in the early intervention pair is exactly at 3.0 s, with maximum per-motor difference **0.000476897 N**. Both choose policy 12 from eight eligible candidates, solve an accepted QP, pass the nonlinear held check, and have positive held operational margins. The included [compact control extract][early-control-extract] retains the exact boundary states, observation/model inputs, command, selected row/bound, hard/smooth values, multipliers and held-check flags, with source dtypes, shapes and hashes. Its [NumPy-only extraction script][early-control-script] authenticates every command against the separate saved application log, covers all 115 exactly aligned common control times, and records all 75 equal applied commands before the first divergence. The [extract manifest][early-control-manifest] binds the script, output and all four source archives. Full [hold controls (local-only)][early-hold-controls] and [revert controls (local-only)][early-revert-controls] remain retained bulk evidence rather than part of the compact review archive. The extract distinguishes the changed certificate from the unchanged physical state:

| At the authenticated 3.0 s boundary | Hold adapted parameters | Restore original parameters |
|---|---:|---:|
| Selected hard collision value (m²) | 1.029786 | 0.927649 |
| Selected smooth collision value (m²) | 1.026423 | 0.924286 |
| Executed policy-row multiplier | 0.001820514 | 0.002050073 |
| Held-command collision margin (m²) | 1.753465 | 1.753465 |
| Full-episode physical outcome | Safe 2/2 completion at 11.32 s | First interpolated intersection at 4.57730780 s, detected at 4.58 s |

The positive prediction margins at the intervention boundary are not guarantees for the later switched closed loop. The full outcome contrast supplies the selected-case causal evidence; prefix authentication alone does not infer the outcome.

The completed P0 integration refinements retain the selected F2-collision/A-safe-task contrast in four additional method episodes. Both methods pass operational checks over their executed prefixes. The following clearances and intersection times come from the saved whole-prefix actual-collider audits; F2 reaches 0/2 waypoints in both runs, while A completes 2/2 and remains separated through 14 s.

| Physical step | A task completion | A minimum obstacle-collider clearance lower bound | F2 first interpolated intersection |
|---:|---:|---:|---:|
| [2.5 ms, half the main step][candidate-fine2] | 11.44 s | 0.173947 m | 4.58385662 s |
| [1.25 ms, quarter the main step][candidate-fine4] | 11.28 s | 0.169188 m | 4.58385676 s |

Both F2 contacts are detected at the 4.585 s integration boundary. This supports local outcome stability under these two refinements; it is not an integration-error bound, independent-plant validation, or evidence of broad robustness. The original and both refined outcomes remain separate from the main sample.

The [neighboring-world campaign][candidate-neighbors] completes **12/12 method episodes on all six declared local perturbations**, using the same structured geometry seed 1002 and library seed 11. The baseline event is at 2.0 s with effectiveness 0.85 and lag multiplier 2.0 on all four motors. Each row changes only its named factor. Every episode passes operational checks over its executed prefix.

| Local perturbation | F2 outcome | A outcome |
|---|---|---|
| Fault onset 1.92 s | Collider intersection detected at 4.585 s; 0/2 waypoints | Safe 2/2 completion at 11.32 s |
| Fault onset 2.08 s | Collider intersection detected at 4.585 s; 0/2 waypoints | Safe 2/2 completion at 11.44 s |
| All-motor effectiveness 0.83 | Collider intersection detected at 4.585 s; 0/2 waypoints | Safe 2/2 completion at 11.56 s |
| All-motor effectiveness 0.87 | Safe 2/2 completion at 11.56 s | Safe 2/2 completion at 11.64 s |
| All-motor lag multiplier 1.9 | Safe 2/2 completion at 11.48 s | Safe 2/2 completion at 11.72 s |
| All-motor lag multiplier 2.1 | Collider intersection detected at 4.585 s; 0/2 waypoints | Safe 2/2 completion at 11.64 s |

A therefore completes safely on all six neighbors; F2 completes safely on two and collides on four. On the two mutually successful neighbors, F2 finishes earlier. The selected safety contrast persists on four neighbors and disappears on two; these deterministic perturbations of one selected geometry and one library are descriptive local sensitivity evidence, not independent population draws or main-test trials. Together, the six interventions, four refinement episodes, and 12 neighbor episodes complete the **22 declared candidate-confirmation episodes**. The four completed common-prefix audits are separate checks, not extra episodes.

The [direct-command development][witness-dev] and [held-out witnesses][witness-val] help localize the learning limitation. Offline solves with 30 held commands find the two hard targets after 12 and 13 iterations from the first F2 initialization. Fine 2 ms replay gives position RMSE 0.242444/0.059421 m and terminal speed 0.509388/0.313035 m/s. These numerical witnesses show those two probed targets are achievable; they are not real-time controllers, general feasibility proofs, or evidence that the learned repertoire recovers them.

The [completed A1 comparison][a1-results] has six episodes on combined-fault validation worlds structured2003 and navigation2103, using library seed 11. F2 and A start from the identical full K=16 nominal128 checkpoint; A1 uses its separately trained and validated K=1 nominal128 checkpoint. The [zero-retention comparison][retention-off-results] adds two A episodes. Its [preparation record][retention-off-preparation] verifies identical numerical checkpoint arrays, full Adam state, teacher and anchors; only the fingerprinted learning-contract retention weight changes from 5 to 0. All use the canonical executable cache, so [canonical oracle A][secondary-canonical] supplies the retention-on reference.

| Validation world | F2, K=16 | A, K=16, retention 5 | A1, K=1 | A, K=16, retention 0 |
|---|---|---|---|---|
| Structured2003, 14 s | Collider-free, 1/2 waypoints, operational pass | Collider-free, 1/2, operational pass | Safe 2/2 completion at 13.20 s | Collider intersection at the 8.665 s detection boundary, 0/2, operational violation |
| Navigation2103, 24 s | Collider-free, 2/4, operational violation | Collider-free, 2/4, operational violation | Collider-free, 2/4, operational violation | Collider-free, 2/4, operational violation |

A1 receives 349/599 finite updates on structured/navigation, as does retention-on A; zero-retention A receives 216/599 before its respective termination. The single-policy result provides no evidence that K=16 is necessary here, and the separately trained checkpoint prevents attributing its success solely to policy count. Removing retention harms the structured validation case and improves neither recorded task count in navigation; this supports a local retention contribution, not a general optimum at weight 5. The completed weight-5/weight-10 screen remains a separate experiment. Two worlds and one library seed support descriptive ablation evidence, not the crossed main-study confidence interval or an optimizer-reset ablation.

## 7. Actual runtime and independent-plant evidence

The complete [GPU Gate B][gate-b] development cases run for 8 s without obstacles, 14 s with static obstacles, and 14 s with moving obstacles. All three finish two waypoints without recorded actual-collider or operational violation. Their 900 controls have zero controller-service deadline misses. The moving case records 55 negative nominal hard collision predictions and 43 executed positive policy-row multipliers, demonstrating that the filter is exercised. The earlier CPU warmup failure remains an incomplete time-zero attempt with no safety or task credit.

Full reverse AD controller probes cost about 45.2–45.6 ms; [full forward AD][forward] reduces isolated device service to 13.23–13.96 ms. These warmed same-state profiles exclude parts of complete scheduling. The [input-cache parity record][cache-parity] passes **1,056 exact comparisons**, requiring identical tree structure, shape, dtype, and raw bytes, including signed zero. Cases cover physical/model float32 and float64, faults and recovery, neighboring event times, delayed observations, biases/noise, and repeated/out-of-order queries. The cache only materializes model phases actually requested at the current or past observed time.

The [GPU input profile][cache-profile] has 160 exact paired queries. Mean synchronized input-preparation time falls from **1.286 ms to 0.281 ms**; p95 falls from **1.419 ms to 0.293 ms**. This measures input preparation, not complete controller service or adaptive safety. Runtime integration has separate deterministic equivalence tests for physical records, controller inputs, finite updates, and final learner state.

The completed [K-cost profile][k-cost] measures four fixed observations spanning both families and nominal/combined dynamics, with three warmup calls and 40 synchronized repetitions per observation and K. The ranges below span those four observation means; they are not uncertainty intervals or episode service times.

| Fallback count K | Isolated controller mean range (ms) | Isolated learner mean range (ms) |
|---:|---:|---:|
| 4 | 14.291–14.916 | 15.374–15.597 |
| 8 | 14.924–15.465 | 15.581–15.855 |
| 16 | 13.783–14.098 | 16.016–16.247 |
| 32 | 14.480–14.779 | 16.549–16.677 |

Controller timing includes the full forward-AD filter and packed device-to-host result with inputs already resident; it excludes observation, obstacle prediction, nominal diagnostics, simulation, and auditing. Learner timing repeats the full step from a fixed input without installing any proposal. K=16 preserves the complete nominal seed-11 checkpoint and Adam state; K=4/8 use policy subsets with fresh Adam, while K=32 uses a fresh actor and teacher. Changed-K competence and flight performance were not evaluated, and filter branches differ. Consequently this is cost evidence, not a pure causal effect of K, a K=16 necessity result, or proof that a serialized controller-plus-learner schedule fits 40 ms.

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

The earlier four-episode [oracle validation baseline][secondary-oracle] is reported in Section 4. The completed [canonical-cache oracle repeat][secondary-canonical] supplies the numerical baseline for the post-main secondary comparisons: all four trials are collider-free, all fail full task completion, and both navigation methods violate operational constraints. The earlier oracle remains attached to its contemporaneous retention-repair decision; it is not substituted for this canonical reference.

In delayed mode, stored controller held-check fields refer to the sensed state and recorded planning time. The old command advances the physical plant during computation; the new command is then applied at its recorded application time after finiteness and command-bound checks, without a fresh controller evaluation or certificate at that shifted state. A passed held check therefore does not recertify the delay-shifted interval. Dense physical states, actual-collider auditing, and operational outcomes remain authoritative for this latency-sensitivity experiment.

Three separate declarations cover the completed post-main numerical work without changing the sealed primary study: the [original post-main queue][post-main-queue] has **70/70 method episodes**, including 22 candidate confirmations, the four-episode canonical oracle repeat, and 44 other secondary/ablation episodes; the [promoted-case supplement][promoted-queue] has **10/10 F2/A episodes** on candidate1002 for paced, delayed, P1, P2, and motor-noise execution; the [observation supplement][observation-queue] has **8/8 F2/A episodes** on validation2003/2103 for position noise and obstacle-prediction bias. The [execution log][post-main-execution] and linked terminal campaigns support completion of all 88 episodes. The queues retain their original declaration-time status strings; specifications alone are not evidence of completion. Candidate P1/P2 use a 2.5 ms physical step; validation P1/P2 use 2.0 ms. Each matches its respective fine-P0 reference. Paced/delayed guards remain the original factor 1.25 and 3 ms reserve.

The validation sensitivity table uses the same combined-fault structured2003 (14 s, two waypoints) and navigation2103 (24 s, four waypoints), library seed 11, identical complete F2/A nominal128 checkpoints, and canonical executable cache. Each row changes only its named observation factor or declared plant level; P1/P2 are compared to fine P0. “Timeout” means collider-free execution through the full duration with incomplete task; “ops” denotes at least one physical operational violation. Every contact and timeout is retained.

| Validation condition | Structured F2 | Structured A | Navigation F2 | Navigation A |
|---|---|---|---|---|
| [Oracle P0, 5 ms][secondary-canonical] | Timeout, 1/2 | Timeout, 1/2 | Timeout, 2/4; ops | Timeout, 2/4; ops |
| [Oracle P0, 2.0 ms][secondary-fine] | Timeout, 1/2 | Timeout, 1/2 | Timeout, 2/4; ops | Timeout, 2/4; ops |
| [Independent P1, 2.0 ms][secondary-p1] | Timeout, 1/2 | Timeout, 1/2 | Timeout, 2/4; ops | Timeout, 2/4; ops |
| [Native P2, 2.0 ms][secondary-p2] | Timeout, 1/2 | Timeout, 1/2 | Timeout, 2/4; ops | Timeout, 2/4; ops |
| [Effectiveness estimate +0.10][secondary-eta] | Contact at 7.715 s, 0/2; ops | Contact at 7.820 s, 0/2; ops | Contact at 22.500 s, 3/4; ops | Contact at 5.440 s, 0/4; ops |
| [Lag estimate ×1.20][secondary-tau] | Timeout, 1/2 | Contact at 5.845 s, 0/2 | Timeout, 2/4; ops | Safe 4/4 at 17.20 s |
| [Parameter observation delay 80 ms][secondary-delay] | Contact at 9.075 s, 0/2; ops | Timeout, 1/2 | Timeout, 2/4; ops | Contact at 12.340 s, 2/4; ops |
| [Motor observation noise, SD 0.002 N][secondary-motor] | Timeout, 1/2 | Timeout, 1/2 | Timeout, 2/4; ops | Timeout, 2/4; ops |
| [Position observation noise, SD 0.01 m/coordinate][secondary-position] | Safe 2/2 at 13.60 s | Safe 2/2 at 13.16 s | Timeout, 2/4; ops | Timeout, 2/4; ops |
| [Predicted obstacle-center bias +0.03 m/coordinate][secondary-obstacle] | Timeout, 0/2 | Timeout, 1/2 | Timeout, 2/4; ops | Timeout, 2/4; ops |

Contact times in this table are physical detection boundaries; source collider audits retain the first interpolated intersection times separately. All safe task outcomes also pass physical operational checks and remain collider-free through the common final duration. The observation-bias effects are not monotone: the 20% lag overestimate harms adaptive structured execution but produces a safe adaptive navigation completion, and position noise changes both structured timeouts into successes. These are individual matched sensitivity outcomes, not evidence that deliberately biasing or adding noise improves a controller in general. The obstacle bias shifts predictions by a 3D magnitude of about 0.05196 m while true obstacle paths remain unchanged.

The promoted candidate has the following separately completed execution/transfer results. All ten episodes pass physical operational checks over their executed prefixes. Times for contacts are detection boundaries; safe task completions remain exposed through 14 s.

| Candidate1002 condition | F2 outcome | A outcome | A credited updates |
|---|---|---|---:|
| [Paced, default guard][candidate-paced] | Contact at 4.585 s, 0/2 | Contact at 4.585 s, 0/2 | 0 |
| [Delayed command, default guard][candidate-delayed] | Contact at 4.705 s, 0/2 | Contact at 4.705 s, 0/2 | 0 |
| [Independent P1, 2.5 ms][candidate-p1] | Contact at 4.6975 s, 0/2 | Safe 2/2 at 11.28 s | 349 |
| [Native P2, 2.5 ms][candidate-p2] | Contact at 4.585 s, 0/2 | Contact at 4.630 s, 0/2 | 115 |
| [Motor observation noise, SD 0.002 N][candidate-motor] | Safe 2/2 at 11.68 s | Safe 2/2 at 11.72 s | 349 |

Against the matching 2.5 ms P0 result in Section 6, the candidate safety contrast survives independently implemented P1 but fails under native P2 despite 115 finite adaptive updates. Both noisy-motor episodes succeed, with F2 finishing 0.04 s earlier, so that perturbation removes the selected adaptive advantage without causing adaptive contact. The zero-update paced/delayed outcomes establish failure to deliver the deterministic adaptive benefit under those measured execution contracts. Neither the P1 survival nor the four collider-free native validation timeouts establishes broad native-plant safety transfer.

The completed [paced validation][secondary-paced] and [delayed validation][secondary-delayed] campaigns each add four F2/A episodes. Both methods time out at 1/2 waypoints in structured2003 with operational checks passing, and at 2/4 in navigation2103 with operational violation. All eight are collider-free. Controller-service means/p95 below are calculated within each individual episode, not pooled quantiles or a mean of episode p95 values.

| World and execution | F2 controller mean / p95 (ms) | A controller mean / p95 (ms) | A finite updates; training → publication time |
|---|---:|---:|---|
| Structured2003, paced | 18.998 / 21.267 | 19.673 / 22.517 | 0; none |
| Navigation2103, paced | 21.017 / 24.021 | 21.096 / 23.691 | 0; none |
| Structured2003, delayed | 14.826 / 15.280 | 15.155 / 15.908 | 1; 0.60 → 0.64 s |
| Navigation2103, delayed | 17.971 / 21.170 | 17.893 / 21.079 | 1; 0.00 → 0.04 s |
| Candidate1002, paced | 27.420 / 28.080 | 28.241 / 29.342 | 0; none |
| Candidate1002, delayed | 27.504 / 28.141 | 27.809 / 28.908 | 0; none |

All 12 episodes in that table have zero controller-service deadline misses, zero learner deadline misses, zero uncredited finite updates, and zero skipped sensing ticks. Paced validation A publishes no update; delayed validation A publishes version 129 once in each world, before fault onset. The actual learner calls take **21.2395 ms** and **19.4582 ms**, respectively. The summaries' per-control learner mean includes no-call zeros and must not be read as update service cost. The first controller use of each new version is at its publication boundary, with source-state age 40 ms; no additional update refreshes those snapshots later in the episode. The original default guard thus delivers no post-fault learning in these validation episodes or in the promoted timing case.

Recorded controller service separates synchronized computation, mandatory host work, observation, and transfer. For A, the episode means are:

| World and execution | Controller compute (ms) | Mandatory host (ms) | Observation (ms) | Transfer (ms) | Total simulator/audit work (s) |
|---|---:|---:|---:|---:|---:|
| Structured2003, paced | 15.802 | 1.643 | 1.110 | 1.118 | 0.762 |
| Navigation2103, paced | 17.766 | 1.581 | 0.871 | 0.878 | 1.177 |
| Structured2003, delayed | 13.431 | 1.168 | 0.273 | 0.283 | 0.987 |
| Navigation2103, delayed | 16.121 | 1.207 | 0.287 | 0.278 | 1.691 |
| Candidate1002, paced | 14.627 | 12.741 | 0.322 | 0.551 | 0.257 |
| Candidate1002, delayed | 14.545 | 12.440 | 0.330 | 0.494 | 0.448 |

Simulator/audit totals are measured implementation work, not an added physical command delay. Paced application age is zero by its nondelayed-hold contract. In delayed validation, sensed-state-to-command-application p95 age equals the corresponding controller p95: F2/A 15.280/15.908 ms in structured and 21.170/21.079 ms in navigation. In the delayed candidate, the 117 actually applied commands per method have F2/A mean age **27.486/27.799 ms**, p95 **28.068/28.867 ms**, and maximum **29.746/36.273 ms**; the final computed command is not applied because collision terminates during the old-command interval. Each publication and application remains governed by its saved physical clock, without counting the same delay twice.

Startup is preserved separately in each episode's `warmup.json`; three disposable warmups precede these service measurements. For candidate paced A, the first controller/learner calls take **3.4137/1.9509 s** and plant compilation takes **2.1163 s**. These are startup costs, so zero warmed controller misses does not establish cold-start feasibility. Complete timing records retain controller, update, publication and application clocks; isolated warmed profiles cannot substitute for the observed lack of useful update availability.

Wind-error sensitivity from plan Section 10.2 is explicitly omitted: the frozen actuator runner rejects nonzero wind, and this study does not add a wind implementation or claim wind robustness. Position noise does not stand for velocity, attitude, or rate noise, which remain zero. P2 changes native motor response, curves, and rotor effects together; it tests the combined plant-model discrepancy and does not attribute an outcome to lag asymmetry or curve mismatch individually. The deterministic P0 video is therefore a labeled mechanism illustration, accompanied by the failed paced comparison; it cannot substitute for real-time adaptive benefit, P2 adaptive safety transfer, or hardware validity.

## 8. Supported claims, unsupported claims, and unresolved limitations

The saved evidence supports an actuator-aware implementation with explicit force/command semantics, tested bounded allocation and independent dynamics, complete nominal controller sanity episodes, persistent recovery learning, and retained source/checkpoint provenance. It supports three-seed combined-fault tracking improvement on a fixed obstacle-free bank, mixed closed-loop development and frozen-test results, exact observation-input equivalence with reduced preparation cost, a selected-case causal contribution from accumulated adapted parameters, and local harm from removing teacher retention. The completed 384-episode main study supplies all prespecified primary comparisons: none supports a nonzero A-versus-F2 or A-versus-DR effect at the declared adjusted interval level. The favorable structured-lag collision and two clearance contrasts against OPT are limited to the sampled cells and the deterministic compute contract.

The completed 88-episode follow-up budget bounds the selected result: onset-freeze succeeds; early cumulative-parameter reversion causes contact; A remains safe on all six local neighbors, both P0 refinements, and P1. F2 also succeeds on two neighbors, so the neighboring-world advantage persists on four of six. Paced and delayed candidate execution provide zero updates and both methods collide. Native P2 loses adaptive safety despite 115 updates, and motor noise makes both methods succeed. Validation model/observation perturbations show additional mixed outcomes and failures. These are completed negative and sensitivity findings, not pending demonstrations of robustness.

The evidence does not support broad collision or task superiority, efficiency superiority at equal available computation, successful online system identification, necessity of K=16, benefit from zero retention, a safety need for post-fault learning in the selected case, real-time post-fault adaptation, broad native-plant transfer, hardware validation, or an unconditional/infinite-time safety guarantee. Static hover feasibility, two local maneuver witnesses, and prediction clearance cannot replace these missing results.

Remaining limitations include the development dependence of selected cases and tuning, only three learned library seeds, four tested main worlds per cell, approximate and sometimes degenerate bootstrap intervals, incomplete DR competence at the hard combined model, degraded effectiveness-only braking and main effectiveness-only outcomes, the fragile nominal replay under identified optimized-executable/cache drift, policy/model switch discontinuities, numerical integration and interpolation error, exact current-model/motor observation assumptions in the primary comparison, incomplete regression coverage, and failed adaptive benefit under the measured candidate compute budget. Independent lag asymmetry and motor-curve effects are not isolated by the bundled native P2 comparison; wind, velocity/attitude/rate observation noise, and an optimizer-reset ablation remain outside the completed evidence. Negative development, main, repair, execution, and transfer outcomes remain part of the research packet.

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

The [protocol CLI][protocol-driver] separates `prepare`, `seal`, and `import`. The completed main import is saved in `main384-sealed-ledger-v1`, and its completed analysis in `main384-sealed-analysis-v1`. To reproduce the import and analysis in fresh directories, use the same sealed numerical source and environment:

```bash
pixi run -e gpu-tests python -m benchmark.da_plcbf_actuator_protocol import \
  --protocol "$ACTUATOR_STUDY/main384-protocol-sealed-v1.json" \
  --campaign "$ACTUATOR_STUDY/main384-sealed-results-v1" \
  --output "$ACTUATOR_STUDY/reproduction-main384-ledger-v1"

pixi run -e gpu-tests python -m benchmark.da_plcbf_actuator_analysis ledger \
  --protocol "$ACTUATOR_STUDY/main384-protocol-sealed-v1.json" \
  --ledger "$ACTUATOR_STUDY/reproduction-main384-ledger-v1/ledger.json" \
  --output "$ACTUATOR_STUDY/reproduction-main384-analysis-v1"
```

These commands reproduce reporting from retained main trials; they do not execute new physical episodes. The saved [analysis JSON][main-analysis], [episode CSV][main-episodes], [aggregate CSV][main-aggregates], [120-comparison CSV][main-paired], and [attempt CSV][main-attempts] are already available. Exploratory `campaign` analysis retains its development status. These report tables were reconciled to saved outputs without rerunning import, bootstrap analysis, or experiments. The final checkout includes the separately audited telemetry correction; exact sealed-source reexecution must use the bound numerical revision and canonical environment rather than assuming that final-checkout source hashes match the sealed protocol.

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
| Main | Authenticated complete sealed ledger, 384/384 coverage, all 120 primary intervals, exact episode/aggregate/paired/attempt tables | Four independent worlds per cell and three library seeds; pooled totals are descriptive; deterministic service is not equal-compute adaptive evidence. |
| Secondary | All 88 declared episodes, four completed prefix audits, matched fine-P0/P1/P2, ablations, observation factors, and execution checks | Descriptive selected/validation support; no extra main denominator or isolated native-factor causal attribution. |
| Telemetry correction | Separate corrected-source F2/A verification replays and the passing 400-array parity record | Original arrays preserved; two verification flights excluded from study denominators and their contended timing excluded from inference. |
| Figures and media | Five artifact-derived PNG/PDF figures, exact source CSVs/manifests, deterministic paired video, separate paced companion, and presentation audit | Selected scenes and rendering frames are not additional trials; deterministic illustration cannot imply real-time or native-plant adaptive safety. |

The [completed figure manifest][figure-manifest] binds the exact source inputs, output hashes, captions and five PNG/PDF/CSV sets. Visual inspection found the figures readable and their declared inference limits visible. The [final figure-source audit][figure-final-audit] verifies all 12 PNG/CSV files byte-identical to the preceding figure set after source-formatting changes and explicit binding of the analysis digest. The renderer reuses authenticated saved main intervals; it does not simulate episodes or regenerate bootstrap intervals.

| Figure | Plot | Exact source table | Interpretation |
|---|---|---|---|
| Main safety and task comparison | [PNG][figure-main], [PDF][figure-main-pdf] | [CSV][figure-main-csv] | Eight cells, four worlds/cell and three libraries; prespecified adjusted paired intervals. |
| Behavior recovery over completed updates | [PNG][figure-recovery], [PDF][figure-recovery-pdf] | [CSV][figure-recovery-csv] | Three-library mean/min–max, not confidence bands; unmet thresholds remain censored after 128 updates. |
| Safety and execution cost | [PNG][figure-cost], [PDF][figure-cost-pdf] | [CSV][figure-cost-csv] | Complete episode wall time per executed physical exposure; deterministic cost, excluding warmup/final serialization. |
| Library and retention ablations | [PNG][figure-ablation], [PDF][figure-ablation-pdf] | [CSV][figure-ablation-csv] | Two validation worlds and one library; changed-K isolated service has no demonstrated changed-K competence. |
| Model and observation sensitivity | [PNG][figure-mismatch], [PDF][figure-mismatch-pdf] | [CSV][figure-mismatch-csv] | Separate matched factors; validation P1/P2 compare to 2 ms P0, not the 5 ms default. |

The [deterministic paired video][video-main] renders the original recorded F2/A candidate trajectories with visible motor command/force bars, a thrust-85%/lag cue, persistent policy identities, selected/executed paths, actual controller mode and online updates used. It contains **421 frames at 1920×1080 and 30 fps**, encoding 14.0333 s for the 14 s physical timeline. Its [render summary][video-main-summary] records SHA-256 `78c0156cc2158a4fc38e58b8cbbb8e73351a293456e6ce8c18553076a7359096`. F2 freezes at its recorded contact termination; the A timeline continues through the common 14 s horizon. No invented physical contact response or further controller trial is shown.

The separate [paced companion][video-paced] makes the failed execution check visible: both methods collide and A uses zero online updates. Its **139 frames at 1920×1080 and 30 fps** encode 4.6333 s for the recorded 4.5838561075 s physical span; the frame-grid padding carries no further task, command or learner credit. Its [render summary][video-paced-summary] records SHA-256 `fd5c4e37759daf44ddc7a307858e6bcca03c9e46bb34d9bf0188be89a658d801`. These are separate labeled experiments, with saved posters, per-frame audits and render bindings.

The [presentation audit][presentation-audit] passes H.264 dimensions/rate/frame-count checks, full video decode without errors, command availability before display, and frozen post-contact state/control checks. The final adaptive used-update count is 349 in the deterministic video and zero in the paced companion. The final render metadata revision preserves every compared physical frame-audit field from the preceding render; some RGB frames differ, so no bitwise-rendering equality is claimed. These videos explain one selected case and its execution limit; neither substitutes for population results or native-plant transfer.

Included evidence links resolve in the current workspace or after extracting the compact [review-only archive][review-archive] at the repository root, after verifying the detached archive checksum. Links labeled local-only require the retained bulk files. Git includes the compact archive and figures rather than every loose source-result file. The [publication manifest][publication-manifest] records included reviewer inputs, hashes, and separately retained bulky artifacts; the video files remain separately checksummed media, while the review packet retains their posters and validation evidence.

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
[retention-final]: ../artifacts/da_plcbf/actuator-study-20260906/v1/M4_RETENTION_REPAIR_DECISION.json
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
[protocol-inputs]: ../artifacts/da_plcbf/actuator-study-20260906/v1/protocol-inputs-main384-final-v1/verification.json
[sealed-protocol]: ../artifacts/da_plcbf/actuator-study-20260906/v1/main384-protocol-sealed-v1.json
[sealed-campaign]: ../artifacts/da_plcbf/actuator-study-20260906/v1/main384-campaign-sealed-v1.json
[main-ledger]: ../artifacts/da_plcbf/actuator-study-20260906/v1/main384-sealed-ledger-v1/ledger.json
[main-coverage]: ../artifacts/da_plcbf/actuator-study-20260906/v1/main384-sealed-ledger-v1/coverage.json
[main-analysis]: ../artifacts/da_plcbf/actuator-study-20260906/v1/main384-sealed-analysis-v1/analysis.json
[main-episodes]: ../artifacts/da_plcbf/actuator-study-20260906/v1/main384-sealed-analysis-v1/episodes.csv
[main-aggregates]: ../artifacts/da_plcbf/actuator-study-20260906/v1/main384-sealed-analysis-v1/aggregate_source.csv
[main-paired]: ../artifacts/da_plcbf/actuator-study-20260906/v1/main384-sealed-analysis-v1/paired_source.csv
[main-attempts]: ../artifacts/da_plcbf/actuator-study-20260906/v1/main384-sealed-analysis-v1/attempts.csv
[canonical-cache]: ../artifacts/da_plcbf/actuator-study-20260906/v1/canonical-compilation-cache-v1/manifest.json
[canonical-environment]: ../artifacts/da_plcbf/actuator-study-20260906/v1/canonical-compilation-cache-v1/environment.json
[canonical-parity]: ../artifacts/da_plcbf/actuator-study-20260906/v1/canonical-episode-parity-v1/results.json
[negative-shared]: ../artifacts/da_plcbf/actuator-study-20260906/v1/negative-1003-canonical-shared-evidence-v1/results.json
[candidate-original]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-original-v1/campaign_result.json
[candidate-shared]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-shared-evidence-v1/results.json
[candidate-onset-freeze]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-onset-freeze-v1/campaign_result.json
[candidate-no-fault]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-no-fault-v1/campaign_result.json
[candidate-early-hold]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-early-hold-v1/campaign_result.json
[candidate-early-revert]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-early-revert-v1/campaign_result.json
[candidate-late-hold]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-late-hold-v1/campaign_result.json
[candidate-late-revert]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-late-revert-v1/campaign_result.json
[candidate-fine2]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-fine2-v1/campaign_result.json
[candidate-fine4]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-fine4-v1/campaign_result.json
[candidate-neighbors]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-neighbors-v1/campaign_result.json
[k-cost]: ../artifacts/da_plcbf/actuator-study-20260906/v1/k-cost-gpu-v1/summary.json
[post-main-queue]: ../artifacts/da_plcbf/actuator-study-20260906/v1/post-main-queue-v1.json
[promoted-queue]: ../artifacts/da_plcbf/actuator-study-20260906/v1/promoted-execution-transfer-queue-v1.json
[observation-queue]: ../artifacts/da_plcbf/actuator-study-20260906/v1/supplemental-observation-queue-v1.json
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

[publication-manifest]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-v1/manifest.json
[review-archive]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-v1/review-only.tar.xz
[telemetry-parity]: ../artifacts/da_plcbf/actuator-study-20260906/v1/telemetry-event-boundary-fix-v1/parity.json
[telemetry-decision]: ../artifacts/da_plcbf/actuator-study-20260906/v1/telemetry-event-boundary-fix-v1/decision.json
[wind-final]: ../artifacts/da_plcbf/actuator-study-20260906/v1/accepted-wind-final-preservation-v1/audit.json
[prefix-onset]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-prefix-onset_freeze-v1/results.json
[prefix-no-fault]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-prefix-no_fault_control-v1/results.json
[prefix-early]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-prefix-early_freeze_vs_revert-v1/results.json
[prefix-late]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-prefix-late_reversion-v1/results.json
[early-hold-controls]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-early-hold-v1/world-0000-seed-11-A/attempt-00/controls.npz
[early-revert-controls]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-early-revert-v1/world-0000-seed-11-A/attempt-00/controls.npz
[a1-results]: ../artifacts/da_plcbf/actuator-study-20260906/v1/a1-validation-subset-original-v1/campaign_result.json
[retention-off-results]: ../artifacts/da_plcbf/actuator-study-20260906/v1/retention-off-validation-subset-v1/campaign_result.json
[retention-off-preparation]: ../artifacts/da_plcbf/actuator-study-20260906/v1/retention-off-initial-seed11-v1/preparation.json
[secondary-canonical]: ../artifacts/da_plcbf/actuator-study-20260906/v1/secondary-oracle-canonical-v1/campaign_result.json
[secondary-fine]: ../artifacts/da_plcbf/actuator-study-20260906/v1/secondary-p0-fine-v1/campaign_result.json
[secondary-p1]: ../artifacts/da_plcbf/actuator-study-20260906/v1/secondary-p1-v1/campaign_result.json
[secondary-p2]: ../artifacts/da_plcbf/actuator-study-20260906/v1/secondary-p2-v1/campaign_result.json
[secondary-eta]: ../artifacts/da_plcbf/actuator-study-20260906/v1/secondary-effectiveness-bias-v1/campaign_result.json
[secondary-tau]: ../artifacts/da_plcbf/actuator-study-20260906/v1/secondary-lag-bias-v1/campaign_result.json
[secondary-delay]: ../artifacts/da_plcbf/actuator-study-20260906/v1/secondary-parameter-delay-v1/campaign_result.json
[secondary-motor]: ../artifacts/da_plcbf/actuator-study-20260906/v1/secondary-motor-noise-v1/campaign_result.json
[secondary-position]: ../artifacts/da_plcbf/actuator-study-20260906/v1/secondary-position-noise-v1/campaign_result.json
[secondary-obstacle]: ../artifacts/da_plcbf/actuator-study-20260906/v1/secondary-obstacle-bias-v1/campaign_result.json
[secondary-paced]: ../artifacts/da_plcbf/actuator-study-20260906/v1/secondary-paced-v1/campaign_result.json
[secondary-delayed]: ../artifacts/da_plcbf/actuator-study-20260906/v1/secondary-delayed-v1/campaign_result.json
[candidate-paced]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-paced-v1/campaign_result.json
[candidate-delayed]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-delayed-v1/campaign_result.json
[candidate-p1]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-p1-v1/campaign_result.json
[candidate-p2]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-p2-v1/campaign_result.json
[candidate-motor]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-motor-noise-v1/campaign_result.json
[post-main-execution]: ../artifacts/da_plcbf/actuator-study-20260906/v1/post-main-execution-v1.jsonl
[figure-manifest]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/manifest.json
[video-main]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-video-v2/comparison.mp4
[video-main-summary]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-video-v2/render_summary.json
[video-paced]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-paced-video-v2/comparison.mp4
[video-paced-summary]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-paced-video-v2/render_summary.json
[presentation-audit]: ../artifacts/da_plcbf/actuator-study-20260906/v1/presentation-audit-v1/audit.json
[figure-main]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/01_main_safety.png
[figure-main-pdf]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/01_main_safety.pdf
[figure-main-csv]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/01_main_safety.csv
[figure-recovery]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/02_fixed_bank_recovery.png
[figure-recovery-pdf]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/02_fixed_bank_recovery.pdf
[figure-recovery-csv]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/02_fixed_bank_recovery.csv
[figure-cost]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/03_safety_and_execution_cost.png
[figure-cost-pdf]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/03_safety_and_execution_cost.pdf
[figure-cost-csv]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/03_safety_and_execution_cost.csv
[figure-ablation]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/04_library_and_retention_ablations.png
[figure-ablation-pdf]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/04_library_and_retention_ablations.pdf
[figure-ablation-csv]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/04_library_and_retention_ablations.csv
[figure-mismatch]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/05_one_factor_mismatch.png
[figure-mismatch-pdf]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/05_one_factor_mismatch.pdf
[figure-mismatch-csv]: ../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/05_one_factor_mismatch.csv

[early-control-extract]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-early-control-extract-v1/extract.json
[early-control-script]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-early-control-extract-v1/extract.py
[early-control-manifest]: ../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-early-control-extract-v1/manifest.json
[figure-final-audit]: ../artifacts/da_plcbf/actuator-study-20260906/v1/figure-final-source-audit-v1/audit.json
[regression-summary]: ../artifacts/da_plcbf/actuator-study-20260906/v1/final-regression-v1/summary.md
[regression-ledger]: ../artifacts/da_plcbf/actuator-study-20260906/v1/final-regression-v1/coverage-ledger.json
[regression-diagnosis]: ../artifacts/da_plcbf/actuator-study-20260906/v1/final-regression-v1/diagnosis.md
