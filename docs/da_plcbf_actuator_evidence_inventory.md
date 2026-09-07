# Actuator study evidence inventory — draft

This inventory records completed, saved evidence available during development on
2026-09-06. It supports a scoped result: persistent adaptation improves combined-fault
body tracking on fixed obstacle-free probes, while braking deficits and actual
computation availability remain limiting. It does **not** establish an additional
closed-loop safety or computational benefit over F2, DR, or OPT.

M0–M3 have substantial implementation and validation evidence. M4 development is
unfinished, and M5–M7 are unfinished. There is no completed frozen main test, paired
test uncertainty analysis, or publication-ready research packet in this inventory.
The artifact named `M5_COMPUTE_PARITY_SUMMARY` contains development profiling; its
filename does not mean the publication plan's M5 frozen-study milestone is complete.
Campaigns still running or subsequently generated require their own updated accounting.

## M0: source authority and accepted reference

The [starting manifest][start] records clean `main` commit
`8c0320617a6fae56d8fa7179349ed4c07311a04c` and development branch
`codex/actuator-study`. This includes the accepted video polish after the plan's
earlier reference commit. The manifest inventories 1,164 accepted-study files;
the [implementation map][implementation] records their total size as 486,440,662
bytes. This is a starting reference inventory, not a declaration of a final source
version or a newly sealed protocol.

The [legacy wind regression][wind] reproduces the accepted fixed-only collision
outcome. All 44 shared non-timing array comparisons pass exact equality, including
the physical dense states and actions. The fixed trace terminates at 4.64 s after
modeled collider intersection; the adaptive trace completes both waypoints at
10.52 s with collider lower clearance 0.1724472846 m. The stronger compensated
frozen comparator's accepted survival remains part of the claim boundary. This
replay preserves the wind anchor; it does not supply an independent actuator-plant
experiment.

## M1: physical contract, parameter provenance, and independent implementation

The [actuator contract][contract] distinguishes commanded nominal-equivalent effort
`u` in N, internal effort state `s` in N, actual force `eta*s`, and the applied body
wrench `B*(eta*s)`. The lagged model has 17 states and applies effectiveness once.
Its motor endpoint uses the analytic exponential solution; body RK4 evaluates
evolving force during each substep. Events preserve body and motor states. The
algebraic effectiveness-only model is separate from the positive-lag model.
Here the model label **A1** means the algebraic limit; the benchmark method label
**A1** below means the adaptive single-fallback ablation. They must not be conflated.

The [parameter derivation][parameters] and [machine-readable atlas][parameter-json]
derive the nominal time constant as **0.059498771217 s**, from the mean of the native
spin-up/down poles at hover. The operating-region symmetric values are approximately
0.076229437, 0.059498771, and 0.051844801 s at 0.021362631, 0.106389450, and 0.2 N
per motor. These are reductions of repository coefficients, including historical
TODOs, rather than new hardware measurements.

The initial atlas contains **38 static trim cases**: 36 proposed-support cases
admit the specified coupled hover, with minimum per-motor headroom 0.048015066 N;
two deep 0.40-effectiveness controls fail that specified hover. The atlas preserves
`witness_not_found` for maneuver feasibility. Neither total thrust nor a failed
local maneuver solve is treated as a general flight-feasibility certificate.

Recorded validation in the parameter note covers 22 physics/allocation tests,
including independent bounded least squares, analytic response, integration
refinement, event continuity, and derivatives. The implementation map separately
records 28 independent-plant tests. These are historical reported suite counts;
this inventory did not rerun them, combine overlapping runs, or assert a current
unique-test total. Relevant suites are [physics tests][physics-tests] and
[independent-plant tests][independent-tests].

P0 is the matched differentiable effort surrogate. P1 independently implements
continuous body/effort dynamics. P2 uses native RPM states, asymmetric motor response,
thrust/torque curves, and rotor inertia; its exact RPM-to-effort observation is a
declared observation oracle. P2 is model mismatch, not hardware validation. The
saved [CPU P1 episode][cpu-p1] runs from time zero through 4 s without recorded
collider or operational violation but finishes only one of two waypoints. This
is a short independent integration sanity result, not a completed transfer study
of adaptive benefit.

## M2: actuator-aware filter and complete sanity episodes

The [implementation map][implementation] derives the collision row in motor-command
coordinates: `a=-grad_s(H)/tau`, with body drift, motor drift, and absolute obstacle
time derivative in the right-hand side. The filter uses normalized physical command
bounds, nonlinear held-command acceptance, nine operational margins, and the same
actuator adapter for nominal, fallback, and emergency execution. The reference
command period is 40 ms; prediction spans 60 steps at 20 ms, or 1.2 s. The selected
full forward AD implementation and checks retain the nominal candidate and normal
rescue paths.

The [GPU Gate B campaign][gate-b] completes three seed-11 F2 episodes: 8 s of empty-
obstacle tracking, 14 s with static obstacles, and 14 s with moving obstacles.
All finish two of two waypoints and retain full exposure, with no recorded collider
or operational violation. Across 900 controls there are zero controller-service
deadline misses. The moving case records 55 nominal predictions with negative hard
collision value and 43 executed positive policy-row multipliers; the filter is
exercised rather than merely present. These are development sanity outcomes.

The earlier [CPU warmup failure][cpu-failure] remains an incomplete attempt at time
zero. It supplies no collision-free or task-success credit. The implementation map
also describes retained orchestration serialization failures and their repairs.

The [theory note][theory] supplies a conditional finite-horizon geometric implication,
including offset-collider attitude error and squared-distance units. It does not
derive uniform trajectory-error bounds from average learning loss, prove BPTT
convergence, or establish recursive safety across policy/model changes.

## M3: behaviors, training budgets, and baseline competence

The [behavior report][behavior] uses 27 development states and 16 disjoint validation
states, including motor transients, with K=16 learned fallback policies plus the
separate nominal candidate. The shared actor has hidden width 32, F2 compensation,
60 integration steps, and two integration steps per command hold.

| Prepared baseline | Seeds | Finite training updates per seed | Student integration steps per seed | Saved nominal competence |
|---|---|---:|---:|---|
| Nominal repertoire | 11, 23, 37 | 128 | 368,640 | All three pass development and validation checks |
| Independently trained DR repertoire | 11, 23, 37 | 512 | 1,474,560 | All three pass nominal validation; each passes 3/4 DR validation models |
| Single braking fallback | 11 | 128 | 23,040 | Passes nominal development/validation checks |

The nominal budget is **128 total updates per deployment**. Seed 11 consists of
the selected 32-update preparation followed by **96 persistent updates**: its
[continuation manifest][nominal11-manifest] starts at version 32, and its
[continuation summary][nominal11-summary] records 276,480 additional student
integration steps. Seeds 23 and 37 each start at version zero and run 128 updates.
The table reports the cumulative 368,640-step nominal budget, without counting
seed 11's selected first 32 updates twice. These student counts exclude teacher
rollouts, anchor caching, validation, and RK body stages. Nominal maximum validation terminal
speeds are 0.71706, 0.71078, and 0.71378 m/s for seeds 11, 23, and 37. The single
brake policy's maximum is 0.30868 m/s. Its checkpoint is
[behavior-single128-seed11-v1/deployment.json][single-checkpoint], with `mode=single`,
seed 11, K=1, and version/finite-step count 128. A1 has no three-seed deployment or
full closed-loop ablation result in this inventory.

The original 32-update bootstrap failed the unchanged 0.8 m/s terminal-speed
criterion at 0.82570 m/s. A paired 32-update repair used absolute terminal-speed
squared during nominal preparation and reached 0.76371 m/s with the other declared
settings unchanged. Both records remain linked from the [behavior report][behavior].
Deployed adaptation retains the immutable nominal teacher and the original excess-
braking objective.

The [three-seed DR record][dr] uses 64 development dynamics samples, four validation
models, independent per-motor effectiveness in [0.7, 1.0], and lag in [1, 3] times
nominal. Every seed uses its corresponding competent nominal128 teacher and a
separately initialized actor/Adam state. Checkpoints 0/128/256/384/512 are evaluated
with the declared validation score; step 512 is selected in all three seeds. The
hard combined probe still has maximum position RMSE 0.5217/0.4474/0.5360 m and
maximum terminal speed 1.0014/0.9992/1.0193 m/s. Broad competence across the whole
randomized support has therefore not been established.

The [Gate D campaign][gate-d] checks F2, DR, A, and OPT on one common nominal,
empty-obstacle, seed-11 validation world. All four run for the full 8 s, finish both
waypoints, and have no recorded collider or operational violation. A credits 199
finite updates in this deterministic schedule. F2/A use the identical nominal128
checkpoint; DR has its separate dr512 checkpoint. This establishes a bounded
full-controller competence check, not comparative obstacle performance.

The [OPT development microbenchmark][opt] additionally contains four one-decision
cases across two service budgets. Its explicitly reduced 0.025 m ego sphere is a
microbenchmark geometry, so those cases cannot be promoted as actual-collider
episodes. OPT remains an MPC-based finite-horizon filter without the terminal-set
guarantees of the general predictive-safety-filter framework.

## M4: recovery diagnostics and retained negative revisions

The [three-seed recovery matrix][recovery] contains **12 fixed-bank runs**: three
library seeds crossed with nominal, effectiveness-only, lag-only, and combined
dynamics. Each starts from that seed's exact nominal128 parameters and Adam history
and publishes 128 finite updates, for **1,536 matrix updates**. Six current states
cycle with two rotating retention anchors; the 16 held-out states never drive an
update. These are obstacle-free virtual-state diagnostics, not 12 flight episodes.
The earlier [seed-11 combined diagnostic][combined] and the separate retention10
repair each add a distinct 128-update diagnostic; they are not independent seeds.

| Seed | Combined F2 max position RMSE (m) | DR (m) | A after 128 updates (m) | F2 / A terminal speed (m/s) |
|---:|---:|---:|---:|---:|
| 11 | 0.7939 | 0.5217 | 0.3707 | 1.5300 / 0.9833 |
| 23 | 0.7972 | 0.4474 | 0.3662 | 1.5466 / 0.9762 |
| 37 | 0.7916 | 0.5360 | 0.3733 | 1.5433 / 0.9748 |

Combined tracking improves beyond both frozen baselines in every seed, but the
0.30 m position-RMSE and 0.8 m/s braking limits remain unmet. Effectiveness-only
adaptation worsens terminal speed from F2's 0.6648–0.6690 m/s to 0.9119–0.9449 m/s.
Lag-only F2 is already competent; adaptation slightly increases its tracking error.
Nominal adaptation retains competence with small teacher drift. RMSE averages squared
xyz coordinate errors over the five declared prefixes; terminal speed is a Euclidean
norm. These distinctions matter when comparing the numbers with thresholds.

The [gain screen][gains] evaluates four attitude/rate gain pairs:
(0.0008,0.0002), (0.0008,0.0003), (0.0006,0.0003), and (0.0004,0.0003).
Selection first requires nominal competence, then uses the declared combined plus
nominal development score. Only the original pair passes nominal competence; the
original score is 4.1766 versus 10.2340, 24.3864, and 54.0652 for the alternatives.
The original gains remain selected.

The [retention decision][retention] compares weight 5 with weight 10 at seed 11
under the same combined fault and 128-update budget. Selection uses the unchanged
27-state development score; held-out results are corroborative only.

| Metric | Retention 5 | Retention 10 |
|---|---:|---:|
| Development selection score | 0.19682161 | 0.23179055 |
| Development maximum terminal speed (m/s) | 0.96314919 | 0.98814106 |
| Held-out maximum position RMSE (m) | 0.37065211 | 0.36928821 |
| Held-out maximum terminal speed (m/s) | 0.98330319 | 1.00735915 |

Weight 10 is rejected. This is a scoped retention-strength comparison, not retention
on/off evidence; no zero-retention run or deployment is present. The changed objective
has a distinct reference fingerprint. Its manifest retains the original reference
configuration and separately records `active_reference_sha256`; the
[128-update checkpoint][retention-checkpoint] explicitly records weight 10. A future
retention-off contract must be separately declared and cannot inherit the original
teacher/objective fingerprint.

Two offline local [development][witness-dev] and [held-out][witness-val] direct-command
witnesses show the selected hard targets are numerically achievable. Each solves
30 held commands and replays them with 2 ms integration. The first F2 initialization
finds a witness after 12 and 13 iterations, respectively; remaining declared
initializations are unstarted. Fine-replay position RMSE is 0.242444 and 0.059421 m,
and terminal speed is 0.509388 and 0.313035 m/s. These witnesses localize a limitation
of the present learned recovery; they are neither real-time controllers nor proofs
of broad maneuver feasibility or independent-plant transfer.

The [development pilot specification][pilot] declares 16 resolved worlds across two
scenario families and four dynamics cells, one library seed, and F2/A: 32 method
episodes. Its complete paired result is outside this inventory's closed campaigns.
The plan's proposed development batches and frozen main budget must not be reported
as completed work merely because their specifications exist.

## Preliminary computation and execution evidence

Recorded full reverse AD controller calls cost about 45.2–45.6 ms in the two
[same-state probes][reverse]. [Full forward AD][forward] reduces device service to
13.23–13.96 ms, with 16 warmed calls per probe; compilation and host transfer are
reported separately. Directional differentiation experiments and packed transport
are documented in the implementation map. They do not turn isolated service time
into a demonstrated complete 40 ms control/learning schedule.

The [learner parity record][parity] retains two paired unroll experiments:

| Objective | Unroll 1 mean / p95 (ms) | Unroll 2 mean / p95 (ms) | Outcome |
|---|---:|---:|---|
| Original ReLU braking | 18.179 / 18.471 | 15.652 / 16.032 | Gradient/next-Adam and strict forward parity fail |
| Declared C1 Huber, delta 0.01 (m/s)² | 18.283 / 18.582 | 15.837 / 16.092 | Gradient/next-Adam pass three probes; strict forward parity fails |

Both exceed the proposed 10% mean time-reduction requirement but fail its mandatory
parity condition, so both are rejected. The original discrepancy is attributed to
the braking ReLU active set: total gradient-difference norm 0.006310856, braking
difference 0.006310833. Under the changed Huber objective, one body-rate forward
coordinate differs by 2.7955e-5 rad/s against an allowed 2.7075e-5. The tolerance was
not relaxed. No candidate proposals were installed, no new deployment was selected,
and neither unroll 4 nor a smoothed 128-update recovery experiment was run. Selected
defaults remain `rollout_scan_unroll=1` and `reference_braking_huber_delta=0`.

The two paced validation variants each run a complete F2/A pair on the same 14 s
world, with a fault at 2 s: effectiveness (0.85,1,1,1), lag multipliers (3,1,1,1).
They retain the original actor, loss, and full current-model filter.

| Saved campaign | Slot safety factor / reserve | A learner calls / published updates | Publication timing | A controller mean / p95 / max (ms) |
|---|---|---|---|---|
| [Default guard][paced-default] | 1.25 / 3 ms | 0 / 0 | None; version remains 128 | 21.546 / 26.626 / 31.047 |
| [Measured guard][paced-guard] | 1.05 / 1 ms | 1 / 1 | Trained at 1.04 s, published at 1.08 s; version 129 | 21.518 / 26.280 / 34.842 |

**Neither variant completes a post-fault learner update.** The sole measured-guard
update takes 25.056 ms and precedes the fault. All four episodes reach 14 s without
recorded collider or operational violation, and all finish only one of two waypoints.
All four record zero controller-service deadline misses. These are completed,
collision-free task timeouts, not successful full tasks or evidence that adaptation
repairs the fault in time. Pacing uses serialized wall scheduling with nondelayed
fixed physical holds; an explicit delayed-command study is a separate requirement.

Across the closed CPU P1, GPU Gate B, Gate D, and two paced campaigns cited here,
there are **12 completed runtime episodes**: 1 + 3 + 4 + 2 + 2. Seven complete the
task and five time out; all reach their declared duration without recorded collider
or operational violation. Reused worlds and library seeds prevent treating these
12 episodes as 12 independent draws from a target safety distribution. Warmups,
fixed-state recovery probes, one-decision OPT cases, and incomplete attempts are
excluded from this episode count.

## Unfinished work and report use

M4 still needs completed full-controller development accounting, a localized causal
case or a supported closed-loop limitation, and methods frozen from development and
validation evidence. M5 still needs a sealed numeric protocol, the complete declared
main comparison and ablations, and paired uncertainty estimates. There is no main
test result to summarize here. M6 still needs consequential post-fault snapshot
availability, delayed-command evidence, and a prescribed independent-plant/mismatch
subset. M7 still needs the final nine-section report, reproducible main figures,
and artifact inclusion/exclusion manifest.

The strongest supported new finding is the fixed-bank combined-fault tracking
improvement together with its retained braking and timing limitations. Unsupported
claims include broad safety superiority, task superiority, efficiency superiority,
necessity of a repertoire, retention-off performance, successful online parameter
identification, hardware transfer, and infinite-time safety guarantees.

For reproduction, use each campaign's `campaign_binding.json` and each diagnostic's
manifest/source snapshot, checkpoint hashes, and exact saved configuration. The
entry points are [behavior preparation][behavior-driver], [recovery diagnostics][recovery-driver],
[learner unroll profiling][unroll-driver], and [episode campaigns][study-driver].
This inventory links compact summaries and representative bindings rather than
enumerating bulk tensors or media. It ran no tests, training, or benchmark jobs and
does not certify archived source against the later working tree.

[start]: ../artifacts/da_plcbf/actuator-study-20260906/v1/STARTING_MANIFEST.json
[implementation]: da_plcbf_actuator_implementation.md
[wind]: ../artifacts/da_plcbf/actuator-study-20260906/v1/legacy-wind-regression-v1/REGRESSION.json
[contract]: da_plcbf_actuator_contract.md
[parameters]: da_plcbf_actuator_parameters.md
[parameter-json]: da_plcbf_actuator_parameters.json
[physics-tests]: ../tests/test_da_plcbf_actuator_dynamics.py
[independent-tests]: ../tests/test_da_plcbf_actuator_independent.py
[cpu-p1]: ../artifacts/da_plcbf/actuator-study-20260906/v1/gate-b-cpu-p1-v1/campaign_result.json
[gate-b]: ../artifacts/da_plcbf/actuator-study-20260906/v1/gate-b-gpu-sanity-v1/campaign_result.json
[cpu-failure]: ../artifacts/da_plcbf/actuator-study-20260906/v1/gate-b-cpu-smoke-v2/world-0000-seed-11-F2/attempt-00/summary.json
[theory]: da_plcbf_actuator_theory.md
[behavior]: ../artifacts/da_plcbf/actuator-study-20260906/v1/M3_BEHAVIOR_SUMMARY.md
[nominal11-manifest]: ../artifacts/da_plcbf/actuator-study-20260906/v1/behavior-nominal128-seed11-v1/manifest.json
[nominal11-summary]: ../artifacts/da_plcbf/actuator-study-20260906/v1/behavior-nominal128-seed11-v1/summary.json
[single-checkpoint]: ../artifacts/da_plcbf/actuator-study-20260906/v1/behavior-single128-seed11-v1/deployment.json
[dr]: ../artifacts/da_plcbf/actuator-study-20260906/v1/M3_DR_THREE_SEED_SUMMARY.json
[gate-d]: ../artifacts/da_plcbf/actuator-study-20260906/v1/gate-d-original-v1/campaign_result.json
[opt]: ../artifacts/da_plcbf_actuator/opt_cpu_development_v1/final_shared_geometry/README.md
[recovery]: ../artifacts/da_plcbf/actuator-study-20260906/v1/M4_RECOVERY_THREE_SEED_SUMMARY.json
[combined]: ../artifacts/da_plcbf/actuator-study-20260906/v1/recovery-combined-seed11-v1/summary.json
[gains]: ../artifacts/da_plcbf/actuator-study-20260906/v1/recovery-gains-v1/summary.json
[retention]: ../artifacts/da_plcbf/actuator-study-20260906/v1/RECOVERY_REPAIR_DECISION.json
[retention-checkpoint]: ../artifacts/da_plcbf/actuator-study-20260906/v1/recovery-retention10-combined-seed11-v1/updates_0128.json
[witness-dev]: ../artifacts/da_plcbf/actuator-study-20260906/v1/body-witness-development26-skill2-v1/summary.json
[witness-val]: ../artifacts/da_plcbf/actuator-study-20260906/v1/body-witness-validation7-skill2-v1/summary.json
[pilot]: ../artifacts/da_plcbf/actuator-study-20260906/v1/specifications/development-pilot-original-v1.json
[reverse]: ../artifacts/da_plcbf/actuator-study-20260906/v1/controller-profile-v2/profile.json
[forward]: ../artifacts/da_plcbf/actuator-study-20260906/v1/controller-profile-forward-v3/profile.json
[parity]: ../artifacts/da_plcbf/actuator-study-20260906/v1/M5_COMPUTE_PARITY_SUMMARY.md
[paced-default]: ../artifacts/da_plcbf/actuator-study-20260906/v1/paced-validation-default-v1/campaign_result.json
[paced-guard]: ../artifacts/da_plcbf/actuator-study-20260906/v1/paced-validation-measured-guard-v1/campaign_result.json
[behavior-driver]: ../benchmark/da_plcbf_actuator_behavior.py
[recovery-driver]: ../benchmark/da_plcbf_actuator_recovery.py
[unroll-driver]: ../benchmark/da_plcbf_actuator_unroll.py
[study-driver]: ../benchmark/da_plcbf_actuator_study.py
