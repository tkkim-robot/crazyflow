# DA-PLCBF navigation and failure-mechanism revision

> **Publication scope:** `main` contains a [compact review subset](artifacts/da_plcbf/REVIEW_ONLY.md);
> generated videos and bulky traces remain local. The original complete-folder manifests are unchanged.
>
> **Later visual/physics correction:** start with `DA_PLCBF_HOVER_REVIEW.md` for the
> hover-first revision. The campaign below used a 0.05 m body-origin sphere. The rendered
> `cf21B_500.xml` asset actually has a 0.086 m collider offset 0.02 m above the body origin.
> Therefore the campaign's “physical collision-free” language means clear of its configured
> point-model sphere; it does **not** establish absence of asset contact. New missions use a
> conservative 0.106 m enclosing sphere, and measured MuJoCo contact continuations are separate
> labeled artifacts. The archived campaign files and manifests remain unchanged.

This is the review entry point for the working revision following exact base commit
`00e89a742a1271b93655bf1bb4581a667dc13a14` from branch `plcbf`, subsequently published on `main`. It implements the attached
review's execution repairs, causal replay, moving-state reference learning, bounded update
ablations, composable 3D missions and recorded videos. The principal architecture remains an
obstacle/goal-agnostic persistent skill learner with every finite update published. A frozen
controller receives the same model information, actuator limits, nominal task controller,
prediction horizon and total library size.

The concrete research objective is to restore useful reusable fallback behavior after physics
changes, then test whether it preserves executable safety and navigation progress. Behavior
restoration, collision coverage, eligible certificates, accepted QPs and executed outcomes are
measured separately. Improved average trajectory tracking is not treated as a safety guarantee.

## Implemented changes

- **Executable candidate predictions:** shared zero-order-held policy rollout distinguishes
  20 ms integration from 40 ms command updates, preserving the 60-step / 1.2 s horizon. Tests
  replay a committed skill with its original anchor and advancing phase through every node.
- **Predictive operational QP repair:** the crossing rejection is an upper-x arena HOCBF face
  at the later held substep. Additional predictive faces repair that exact fixture while keeping
  the original objective, collision face, limits, tolerances and nonlinear postcheck. The bounded
  refiner supports one- and two-step holds; longer holds retain postchecks but require explicitly
  disabling refinement to avoid combinatorial solver growth.
- **Consistent compensation:** new teacher warmup, physical prefixes and principal branches
  all enable the same compensation mapping. A saved old/new-plant × mapping-off/on matrix
  isolates the previous protocol confound at both rest and moving event states.
- **State-conditioned learning:** immutable nominal-model reference maneuvers start from the
  same proprioceptive state. A broad anchor bank, multi-time trajectory/velocity tracking,
  separate braking/effort terms, bounded offsets and frozen-duration options replace reliance
  on rest-state endpoint targets. Full Adam continuation and the reference contract are saved.
- **Controlled update variants:** network-only, offset-only and bounded full-parameter updates
  are separately named ablations. A global 0.002 parameter-step cap delays early coverage loss
  in the fixed-state fixture; it does not reject updates based on safety values.
- **Reproducible missions:** a shared seeded queue has eight waypoints with altitude changes;
  spheres move on analytic absolute-time periodic trajectories with correct per-node velocity.
  Wind changes and centered payload attachment are separate composable events. Task progress
  stops at collision, completion or timeout; terminal video padding creates no controls/updates.
- **Auditable execution:** deterministic update opportunities are independent of machine load;
  a separate paced scheduler uses measured remaining service time and publishes only completed
  snapshots at actual boundaries. Proposed/fallback/applied held residuals are named and saved,
  alongside actual plant, actuator, collision and progress diagnostics.
- **Recorded visualization:** the existing scene-first style retains the full colored skill
  repertoire, actual paths and wind traces. Active waypoint markers follow each shared queue.
  The centered 5 cm payload has a true-geometry outline and brief metric inset.

## Causal findings

Read [failure_diagnosis.md](artifacts/da_plcbf/navigation-revision-20260905/failure_diagnosis.md)
and its independently checked numeric audits before attributing failure to a single cause.

The two archived 4.0–5.2 s windows reproduce 62 applied commands exactly, including rejection
flags, using the original code and recorded update opportunities. Complete learner/Adam
snapshots are saved at every boundary. A 144-cell fixed-state/model/parameter/compensation/cadence
factorial shows both harmful early updates and optimistic estimation. At the same 4.16 s state
and recorded estimate, initial frozen fallback coverage is +0.05234 m²; four updates make it
−0.05835 m². Corrected cadence changes these to +0.06063 and −0.05141 m², respectively.

The independent crossing fixture is different: all 17 candidates are collision-clear, but the
upper-x arena residual becomes −0.047247 at +20 ms. One predictive refinement produces a residual
of −0.00000095, passing the unchanged tolerance. Thirty synchronized repaired-branch GPU calls
have median 15.152 ms and p95 15.503 ms. This is a controller-branch measurement, not a complete
online learning-budget result.

The initial reference-bank learner improves mean obstacle-free trajectory error across
development learner seeds 7, 9 and 11. For seed 7, the frozen changed-model RMSE is 0.11620 m;
the original current-state objective reaches 0.11330 m and the reference bank reaches 0.09834 m
after 80 updates at learning rate 0.001. Terminal braking is not uniformly improved. All seven
initial variants still lose the early oracle certificate after four fixed-event-state updates.

The further parameter-group ablation localizes this early damage mainly to the shared network,
not only global skill offsets. Network-only updates lose eligibility by update four; smaller
offset-only steps and the 0.002 full-step cap retain it through the sampled update 20. The cap
improves tracking more slowly and still loses eligibility by update 80. This is a tested
stabilization direction with a measured limit, not a stable-safety learner claim.

A fresh 0–8 s closed-loop comparison uses the corrected shared compensated prefix, with frozen,
unrestricted reference learning and capped reference learning. Under oracle wind, all three
retain positive shell clearance and have zero degraded controls. Under their own rate-2.4 wind
estimates, minimum shell margins are −0.04856, −0.07967 and −0.05128 m, respectively; degraded
controls are 24, 39 and 24. No branch physically collides in this bounded window. The cap reduces
unrestricted update harm, but stays near frozen and does not fix the shared estimation failure.

An estimator-only replay fixes the actual states and commands and varies response rate and
velocity noise. With exact observations, rates 1.2 / 2.4 / 6 settle below 0.1 m/s wind error after
3.14 / 1.58 / 0.64 s. With shared 0.01 m/s velocity noise, none settle by the end of the window;
late RMSE is 0.261 / 0.218 / 0.280 m/s and 26% of post-event inferred samples hit the ±5 component
clip. This is a lag/noise diagnosis, not evidence that a faster closed-loop estimator is safe.
The [ablation index](docs/da_plcbf_revision_ablation_index.md) keeps each change attributable.

## Frozen paired evaluation

`CAMPAIGN_PROTOCOL.json` fixes the source archive, checkpoint, hyperparameters and world seeds
before evaluation. The teacher seed is fixed at 7; the evaluated randomness is route jitter
and moving-obstacle phases. No outcome-specific tuning or per-world safety admission is used.
Each principal condition has ten paired world draws, seeds 100–109, with eight moving obstacles
and 40 s maximum duration. The combined condition changes wind at 8 s and 24 s and attaches a
centered +25% mass payload at 16 s.

| Condition | Frozen: complete / positive shell / zero degraded | Adaptive: same criteria | Mean adaptive-minus-frozen completion time |
|---|---:|---:|---:|
| 8 obstacles, unchanged dynamics | 10 / 10 / 10 | 10 / 10 / 10 | +0.004 s |
| 8 obstacles, wind only | 10 / 10 / 10 | 10 / 10 / 10 | −0.028 s |
| 8 obstacles, payload only | 10 / 10 / 10 | 10 / 10 / 10 | −0.016 s |
| 8 obstacles, wind + payload | 10 / 10 / 10 | 10 / 10 / 10 | +0.092 s |
| 16 obstacles, wind + payload | 9 / 9 / 7 | 9 / 9 / 9 | −0.240 s (8 pairs both complete) |

The isolated wind-only and payload-only conditions use the same ten seeded worlds and frozen
checkpoint. Both methods pass all ten missions in each, including actual operational/motor and
applied derivative checks. Exact pre-event physical states, commands and library histories match
the corresponding unchanged run in every method/world, verifying that the separate events compose
without a hidden prefix change. Wind-only completion difference has interval [−0.068, 0] s;
payload-only [−0.028, −0.004] s, representing one 40 ms control interval in four worlds.
These small differences do not establish a meaningful safety advantage. The
[isolated statistics](artifacts/da_plcbf/navigation-revision-20260905/ISOLATED_DISTURBANCE_STATISTICS.json)
and [composition audit](artifacts/da_plcbf/navigation-revision-20260905/ISOLATED_DISTURBANCE_COMPOSITION_AUDIT.json)
retain all 20 pairs. There are **50 paired world/condition trials overall**, not 50 independent
world geometries: the four eight-obstacle conditions share seeds 100–109.

The paired bootstrap 95% interval for combined completion-time difference is +0.012 to +0.188 s.
The small unchanged difference is one 40 ms control interval in one world. These observations
show no adaptive task/safety superiority. Ten successes give a Wilson 95% interval of roughly
72.2%–100%, so they are not a general safety claim. Raw points, intervals and full outcomes are
in [heldout_statistics.json](artifacts/da_plcbf/navigation-revision-20260905/heldout_statistics.json).
The deterministic trials ran concurrently; their service costs cannot establish deployability.

The denser 16-obstacle development seed 0 completes all eight waypoints in both methods.
Frozen shell clearance is −0.00492 m with 14 degraded controls; adaptive clearance is +0.00717 m
with one degraded control. This difference starts before the first wind event. Exact replay
of 41 updates and same-state probes show that both libraries fail at the frozen branch's later
failure states: earlier path divergence matters. A separate ten-world 16-obstacle confirmation
uses seeds 200–209 and the same frozen checkpoint. Both methods remain physically collision-free
in all ten worlds, complete nine missions, and keep positive shell margin in nine. The failing
worlds differ: frozen times out in seed205 and breaches the shell in seed206; adaptive breaches
the shell and times out in seed209. Frozen has 24 degraded controls across three worlds; adaptive
has 20 in one world. The adaptive-minus-frozen waypoint difference averages zero, with paired
bootstrap 95% interval [−0.3, +0.3] waypoints. Among the eight pairs where both complete, the mean
time difference is −0.240 s with interval [−0.695, +0.220] s. Shell-margin difference averages
+0.00128 m with interval [−0.00998, +0.01383] m. This family does not establish overall superiority.
All points are in [dense confirmation statistics](artifacts/da_plcbf/navigation-revision-20260905/dense_confirmation_statistics.json).

Seed206's export originally failed on nonfinite diagnostics from two rejected QP proposals.
The output-only patch preserves them with explicit masks/counts and keeps executed rows finite.
Its rerun reproduces the complete physical trajectories and all final learner/Adam arrays exactly;
only the repaired export is counted. That frozen run has five executed derivative-check violations
as well as its shell breach; finite diagnostics do not mean constraints pass.

The 8-obstacle worlds are relatively mild: they produce one to three distinct nominal-blocking
episodes. The 16-obstacle worlds produce three to eight, with median 5.5 per mission. Median
nominal blocked fraction is 18.2% frozen / 17.1% adaptive there, and the PL-CBF policy row has a
positive executed dual in a median 16.7% / 16.2% of controls. Thus the denser family contains
repeated measured threats, rather than relying on drawn object count. The input-bound
[encounter audit](artifacts/da_plcbf/navigation-revision-20260905/ENCOUNTER_SEVERITY_AUDIT.json)
retains every world and the geometric scope.

## Measured execution budget

The final exclusive-GPU run, `paced-validation3`, completes all eight waypoints with positive
shell clearance, no degraded controls and **zero measured deadline misses in either branch**.
The startup fixes are validated, but **zero of 335 allowed learning opportunities launch**.
No advanced snapshot is published or used, so the explicit runtime-feasibility result is false.
This is successful sampled mission control, not successful online-learning service.

Steady learner warmup is about 22 ms, reserved as 27.72 ms under the declared 1.25 safety factor.
Adaptive controller median is 14.02 ms; with the 3 ms reserve, the illustrative serialized sum is
44.75 ms before plant and recording work, exceeding the 40 ms period. The actual scheduler uses
remaining wall time on each boundary, rather than this illustrative sum. Its conservative budget
therefore leaves the reference-bank learner unexercised online.

| Exclusive paced attempt | Fixed / adaptive deadline misses | Finite online updates |
|---|---:|---:|
| `paced-validation` | 13 / 1 | 0 |
| `paced-validation2` | 4 / 1 | 0 |
| `paced-validation3` | 0 / 0 | 0 |

The final implementation preconstructs declared event models, converts host telemetry before
slicing and synchronizes learner inputs before its warmup stopwatch. These remove observed
startup costs without changing the command hold, physical constraints or scheduler reserve.
The [three-attempt audit](artifacts/da_plcbf/navigation-revision-20260905/paced_runtime_audit_three_attempts.json)
checks raw arrays, independently recomputes deadlines from wall timestamps, verifies publication
chronology and binds each input hash. All attempts remain archived. Deterministic campaign timing
is not used to fill the remaining online-learning budget gap.

## Evidence and source boundaries

New navigation/failure evidence is under
`artifacts/da_plcbf/navigation-revision-20260905/`; learning ablations are under
`artifacts/da_plcbf/learning-revision-20260905/`. The original `competent-revision-20260904`
artifacts are untouched. The source archive used by the paired campaign has 169 verified
file hashes and explicitly includes the predictive hold-length guard. Later optional update-cap,
contract-binding and runtime-accounting work is separately recorded; it does not change the
completed frozen campaign.

Exact replay claims are local to the recorded GPU/software/cache configuration in
`ENVIRONMENT.json`. A seed206 diagnostic launched without the campaign compilation cache/allocation settings first
missed full continuation parity by 5.77e−6; that attempt is retained and excluded. Repeating with
the declared launch settings together gives bitwise equality through 50 updates and reproduces all four
recorded first-failure commands exactly. Hardware/compiler-independent bitwise equality is not
claimed. At 19.36 s, both libraries still have an eligible collision certificate but no feasible
QP, and apply the same degraded action at the frozen state; at 19.48 s both lose eligibility.
This controlled probe supports earlier path divergence rather than rescue of the later frozen
state. The opposite seed209 failure is also localizable: at 13.84 s, frozen parameters provide a
nondegraded direct fallback at the adaptive first-loss state, while adaptive parameters do not.
The preceding control reverses that ranking. Read the
[symmetric dense failure note](artifacts/da_plcbf/navigation-revision-20260905/PAIRED_DENSE_FAILURE_DIAGNOSIS.md)
for all eight queries and their distinction from a full counterfactual recovery episode.

Some older fixture checkpoints have no reference-contract binding. New checkpoints bind both
the numeric reference and its JSON settings; old unsigned inputs remain explicitly labeled.
The artifact SHA manifests additionally bind the complete existing checkpoint/contract
pairs and all videos. Stored failed development outputs remain identified as failed attempts.
`current-source/SOURCE.tar.gz` and `SOURCE_SHA256.json` capture the final working implementation
under the navigation evidence root. Each of the two evidence roots has `ARTIFACT_SHA256.json`;
verify complete inventories with `benchmark/da_plcbf_revision_manifest.py --verify` using those
roots. The manifests exclude only themselves. They do not relabel the older source archives.

## Reproduction entry points

Use the repository's `.pixi/envs/gpu-tests/bin/python`, with `PYTHONPATH=.` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. Numerical execution and rendering are separate commands.

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .pixi/envs/gpu-tests/bin/python examples/da_plcbf/navigation_demo.py run \
  --checkpoint artifacts/da_plcbf/learning-revision-20260905/reference-ablation-seed7/candidate/checkpoint \
  --world-config artifacts/da_plcbf/navigation-revision-20260905/heldout-shard-a/combined-seed100/world.json \
  --config artifacts/da_plcbf/navigation-revision-20260905/heldout-shard-a/combined-seed100/config.json \
  --output-dir /tmp/da-plcbf-new-navigation-run --no-render

MUJOCO_GL=egl JAX_PLATFORMS=cpu PYTHONPATH=. \
  .pixi/envs/gpu-tests/bin/python examples/da_plcbf/navigation_demo.py render \
  --input-dir /tmp/da-plcbf-new-navigation-run \
  --output-dir /tmp/da-plcbf-new-navigation-video
```

For exact campaign-source reproduction, extract `CAMPAIGN_SOURCE.tar.gz` over a checkout of the
base commit recorded in `CAMPAIGN_PROTOCOL.json`, then use its saved configuration and schedule.
The current working implementation includes later explicitly named ablations and metadata fixes.
`benchmark/da_plcbf_navigation_campaign.py`, `da_plcbf_navigation_summary.py`,
`da_plcbf_failure_replay.py`, `da_plcbf_failure_factorial.py`, and the reference-learning benchmark
scripts provide the corresponding reproducible numerical and analysis entry points.

## Focused validation and review route

The final navigation/recording suite passes **49 tests** (two EGL rendering tests are selected
separately). It covers paired no-learning identity, full held execution, terminal censoring,
world/event semantics, scheduler chronology and rejected-proposal serialization. The final execution/runtime subset passes another **27 tests**, overlapping the runner checks.
Learner, reference-binding, predictive-QP and renderer checks are recorded separately in the evidence
index; repeated runs are not summed into an inflated test count.

Start with the [evidence index](artifacts/da_plcbf/navigation-revision-20260905/EVIDENCE_INDEX.md),
then the failure diagnosis and learner note. The two main videos are the frozen held-out
`combined-seed100` mission and the 16-obstacle development failure diagnostic. Both show recorded
1600 × 900, 20 fps scenes with complete skill repertoires and physically sized payload cues.

The [paired-world figure](artifacts/da_plcbf/navigation-revision-20260905/campaign-comparison-figures-v2/navigation_paired_worlds.png)
and [uncertainty figure](artifacts/da_plcbf/navigation-revision-20260905/campaign-comparison-figures-v2/navigation_paired_uncertainty.png)
show the complete initial 30-pair evaluation. PNG/PDF outputs bind their input statistics and
source; isolated-event controls are reported in their separate complete evaluation.

## Limits that remain substantive

- Template tracking is still an imperfect surrogate for encounter-state safe coverage. Every
  finite update is published, so an improving loss may reduce the available certificates.
- Accurate mass-scaled acceleration mapping and point-model force compensation already handle
  much of centered mass/uniform-wind change. Frozen parameters do not mean ignored physics.
- The estimator is a declared noiseless-telemetry low-pass point estimate; its lag and reference
  model error are measured separately from learning. No uncertainty guarantee is claimed.
- Collision checks use relative swept chords at integration nodes. Analytic obstacle time
  derivatives are consistent locally; nonlinear inter-node arcs and parameter/perception
  switches are not covered by a forward-invariance theorem.
- Positive sampled physical margins, repaired local QPs and successful finite missions remain
  distinct from continuous-time safety. A missed measured deadline is a failure and does not
  automatically extend the simulator's command hold.

The main unresolved work is to reduce reference-bank BPTT service cost while retaining its
obstacle-free behavior protection, and to prevent small finite updates from removing the few
useful encounter-state fallbacks. The current cap delays this loss but does not solve it; the
noise-sensitive wind estimate adds a separate model-error failure. These are measured limits of
this implementation, not a rejection of the broader adaptation hypothesis.

Technical details: [held execution](docs/da_plcbf_held_execution_revision.md),
[state-conditioned learning](docs/da_plcbf_state_conditioned_learning.md).
