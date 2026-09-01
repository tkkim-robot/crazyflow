# DA-PLCBF engineering-review handoff

## Current status — 2026-09-01

This is the current DA-PLCBF **engineering-review checkpoint** after the GPU-online-adaptation
correction requested during handoff. The implementation and new four-condition review videos are
ready to inspect. They are not claim-grade scientific evidence, a completed publication run, or a
hardware-safety result.

Repository state at this checkpoint:

- repository: `https://github.com/tkkim-robot/crazyflow.git`;
- branch: `plcbf`;
- takeover baseline: `22cec20b1f296de00a2d1dbd6d6ac7bb594de27d`;
- the current engineering checkpoint is committed and pushed at the tip of `origin/plcbf`;
- no reset or merge into `main` is authorized by this handoff;
- the explicitly requested compact engineering-review bundle is tracked with `REVIEW_ONLY.md`;
  all other ignored development artifacts under `artifacts/da_plcbf/` remain local evidence only.

The current source uses adaptation-evidence schema **7** and execution contract
`gpu-preferred-jit-fixed-budget-lineage-bound-no-replay-v1`. On a GPU machine, the real online
differentiable rollout, reverse-mode gradient, and ten-step optimizer burst execute as one compiled
GPU graph. The hard-evidence graph also executes on that accelerator, while the causal estimator
remains CPU-canonical. Active/candidate snapshots stay immutable and content-addressed, and only a
controller-boundary compare-and-swap can publish an admitted candidate.

Cross-process byte-identical BPTT replay is no longer a production or claim-eligibility
requirement. Validation still binds the scheduled initial policy, candidate/event metadata,
snapshot lineage, retained hard evidence, recomputed hard-admission report, and publication state
transition. This deliberately accepts harmless accelerator/driver floating-point differences.

Current review evidence:

- campaign-faithful RTX 4090 BPTT benchmark:
  `artifacts/da_plcbf/gpu-bptt-online-20260901-v4.json`;
- full shape `K=64, B=64, H=50`, eight obstacle slots, ten optimizer updates: median **141.47 ms**,
  p95 **141.60 ms**, worst **141.65 ms**, with all updates accepted and finite/nonzero gradients;
- fresh asynchronous four-condition GPU development/review run:
  `artifacts/da_plcbf/gpu-online-review-all-v2-20260901/`;
- **8/8** one-fold outcomes completed for `nominal_only` and `da_plcbf_full` across the four
  conditions; the run has four 1600x900 MP4s, four contact sheets, and 32 keyframes;
- all four contact sheets were manually inspected: the fixed-span ego scene, hazards, trajectory
  history, fallback library, selected rollout, and GPU BPTT active/completed banners are legible;
- this review directory is deliberately **unsealed** and has no digest-bound visual-review records,
  manifest, or `SHA256SUMS`; it is not claim eligible.

The review traces/videos were generated immediately before the final Ruff-only formatting pass;
that pass changed source bytes but not executable semantics. The v4 timing artifact and focused
post-format tests bind the final formatted Python source. The unsealed videos are for human review,
not source-hash-sealed evidence.

The measured ten-step BPTT update is not a 20 Hz operation by itself. It does fit the configured
500 ms adaptation cadence asynchronously. In the fresh four-condition run, warm BPTT jobs were
roughly 191--230 ms and complete candidate admission roughly 362--443 ms. The full controller had
26.20--26.51 ms median and 56.04--58.15 ms p95 latency; complete wall steps had 33.57--33.86 ms
median and 72.97--84.34 ms p95 latency. Thus the median fits a 20 Hz period on this machine but the
tail does not, every full-method row missed the stricter configured 20 ms deadline, and no
hard-real-time guarantee exists.

The fresh run is useful precisely because it does not hide failures:

| Full-method condition | Minimum hard margin | Failure steps | Collision steps | Admitted online update |
|---|---:|---:|---:|---|
| static | 0.116558408 | 0 | 0 | yes, step 59 |
| dynamics change | -2.39778634 | 17 | 0 | yes, step 109 |
| ballistic ball | -0.498397676 | 6 | 1 | no |
| interceptor drone | -0.465297841 | 22 | 19 | no |

Every full-method outcome is explicitly blocked from a safety claim because `realtime_probe`
depends on machine load and is hardware-feasibility evidence, not load-invariant safety evidence.
One fold cannot support comparative statistics. The method reduced failure/collision counts versus
the paired nominal trace in the ballistic and interceptor examples, but performed worse on the
dynamics-change minimum margin/failure count and remained unsafe in three of four conditions. This
is evidence that GPU learning, admission, publication, filtering, and evasive behavior execute; it
is not evidence that the present method is generally safe or superior.

The remaining boundary is important:

- this engineering-review checkpoint means the current source, focused gates, benchmark, and
  unsealed development campaign can be inspected; the exact-source broad repository/docs/package
  rerun is still tracked separately below;
- claim-grade means frozen clean source plus the complete predeclared experiments, including the
  2,800-trial core campaign and four reviewed final videos;
- claim-grade work has **not** been run;
- the dynamics-knowledge study no longer has a permanent independent-BPTT-replay blocker;
  confirmatory eligibility is now determined by its predeclared schedule and retained execution /
  hard-admission evidence. Blanket safety-superiority claims remain forbidden.

The earlier sealed v3 development pilot, producer smokes, and full-suite logs below are historical
evidence from the CPU-BPTT source state. They remain useful diagnostics but do not validate the new
source digest.
Do not call the project scientifically complete until the separate claim-grade definition is
satisfied.

## Task sources and claim boundary

Read these before changing implementation decisions:

1. The user's request and current Codex task history.
2. Shared chat, **Online Safety Filter Learning**:
   <https://chatgpt.com/s/t_6a94247c0bb081918b7acd5c6baad95c>
3. `DA_PLCBF_PLAN.md`.
4. Crazyflow paper HTML: <https://arxiv.org/html/2606.01478v1>
5. Official upstream repository: <https://github.com/learnsyslab/crazyflow>

The intended safety statement remains deliberately narrow:

> Under the logged model/scenario samples, constraints, numerical tolerances, and finite horizon,
> the hard rollout and filter checks observed the reported margins and violation rates.

It is not an infinite-horizon, distribution-free, real-world, or hardware guarantee. Preserve
physical failures, degraded/no-safe-fallback intervals, execution failures, deadline misses,
rejected candidates, and counterexamples. A completed confirmatory run remains valid if it shows
no improvement. Never retune or omit folds after inspecting confirmatory outcomes.

The upstream audit found no existing upstream DA-PLCBF/PL-CBF implementation to adopt. The local
branch reused only the compatible symplectic-vectorizer idea from the thrust-limits work. Do not
merge that branch wholesale: its rotor clipping changes stopped/idle semantics. The broad upstream
randomization redesign was also intentionally not imported.

## Environment and test tiers

Pixi 0.76.0 is installed at:

```text
/home/tk/.pixi/bin/pixi
```

The locked `tests`, `gpu-tests`, `docs`, and `dist`/`release` environments have been resolved on
this machine. Observed accelerator/runtime context:

- NVIDIA GeForce RTX 4090, 24,564 MiB;
- NVIDIA driver 555.42.06 (current takeover machine); historical source artifacts may record a
  different driver and must retain their own provenance;
- compute capability 8.9;
- JAX/jaxlib 0.11.1;
- `jax.default_backend()` returned `gpu` in `gpu-tests`.

Test deactivation is reversible and policy-based. No test was deleted, renamed, or permanently
skipped:

- `tests/core-tests.txt` lists 32 files;
- the separately executed Version-B runtime file makes the curated local tier **33 of 80** files;
- the other 47 files remain inactive only in the normal local loop;
- `pixi run -e tests tests` runs the curated tier and Version B in an isolated process;
- `pixi run -e tests tests-full` runs all 79 files and still isolates Version B;
- CI uses `tests-full`;
- render tests remain a separate explicit gate because repository Pytest defaults exclude the
  `render` marker.

Version-B isolation is a deterministic process-lifetime boundary after hundreds of JAX
compilations; it is not omitted coverage. See `tests/README.md` for the policy.

Always use a unique JAX cache for each substantial gate and run CPU and GPU gates serially. A
shared fixed `/tmp` cache caused confusing long-suite behavior during takeover.

```bash
CRAZYFLOW_CPU_CACHE=$(mktemp -d /tmp/crazyflow-jax-cpu.XXXXXX)
JAX_COMPILATION_CACHE_DIR="$CRAZYFLOW_CPU_CACHE" \
  /home/tk/.pixi/bin/pixi run -e tests tests-full

CRAZYFLOW_GPU_CACHE=$(mktemp -d /tmp/crazyflow-jax-gpu.XXXXXX)
JAX_COMPILATION_CACHE_DIR="$CRAZYFLOW_GPU_CACHE" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/tk/.pixi/bin/pixi run -e gpu-tests tests-full
```

Current exact-source DA-PLCBF test status:

- CPU non-render unit coverage passes under the documented Version-B process split: the
  complementary suite reported **708 passed, 3 skipped, 6 render tests deselected** in 400.92 s,
  and isolated Version B reported **10 passed, 1 skipped** in 63.47 s. Combined: **718 passed,
  4 skipped, 6 deselected** in 464.39 s summed;
- a monolithic run reached more than 89% with no assertion failure, then XLA exited 134 while
  compiling `test_no_current_certificate_is_explicitly_degraded`. That exact node passes alone
  (**1 passed** in 17.97 s), and the complete Version-B module passes above, so the monolithic
  result is recorded as accumulated compiler/resource exhaustion rather than a test failure;
- focused default-renderer tests reported **11 passed, 2 deselected**, and the render/encode/replay
  tier reported **2 passed, 11 deselected** before the mechanical formatting pass;
- after that pass, the four directly affected modules reported **113 passed, 2 skipped, 1 render
  deselected**, the renderer replay reported **1 passed, 6 deselected**, and Ruff lint/format,
  compileall, and `git diff --check` all passed;
- the complete repository-wide CPU/GPU/docs/package rerun on this exact post-correction source is
  still pending. The larger counts retained later in this handoff are historical and must not be
  substituted for that gate.

## Implemented architecture

### Simulator and repository foundations

- Integration/vectorization argument ordering is corrected across first-principles and SO-RPY
  variants.
- Quaternion/rotation handling, seeded reset/goal sequences, and simulation reset/step semantics
  have regressions.
- Visualization marker orientation is deterministic, and renderer reuse rejects incompatible
  camera/resolution changes until close.
- `imageio-ffmpeg==0.6.0` and Optax are direct dependencies; evidence provenance records the
  encoder and runtime.
- DA-PLCBF APIs and documentation generation are present.

### DA-PLCBF runtime and mathematics

The implementation under `crazyflow/safety/da_plcbf/` includes:

- fixed-shape scenario tapes with independent named RNG streams and semantic digests;
- static, dynamics-change, ballistic-ball, interceptor-drone, and combined falsification scenes;
- swept contact/constraint checks and true-plant replay from realized motor forces;
- a double-integrator reference system and Crazyflow quadrotor Version-A direct-wrench model;
- arena, altitude, speed, angular-rate, tilt, obstacle, and capsule barriers;
- a coupled motor-force polytope, active-set projection, KKT audit, and SciPy cross-checks;
- a shared structural plus latent-residual fallback actor, fixed structural slots, adaptive slots,
  trainable skill codes/durations, and a common hover/brake horizon tail;
- hard sampled policy values, conservative differentiable values, normalized descriptors, and
  coverage/redundancy/diversity/action/rate/terminal/trust losses;
- fixed-budget truncated BPTT;
- immutable content-addressed active/candidate snapshots, model versions, hard admission,
  staleness rejection, atomic publication, rollback, and explicit degraded operation;
- deterministic positive-hard-value selection with an admissible-set proxy and logged hysteresis;
- a low-dimensional online mass/drag/wind/rotor-efficiency estimator and Cartesian R=4/R=8 model
  samples;
- continuous Version-A PL-CBF and nonlinear/discrete Version-B through the full Crazyflow
  allocation/rotor/integrator path with exact post-checks;
- seven matched methods and four predeclared core conditions;
- separate core, candidate-quality, dynamics-knowledge, Version-A/B, performance, and
  falsification producers so unlike claims are not mixed.

The fallback initialization is a concurrent grid of structural policies, not a runtime maneuver
state machine. Attacker modes belong to the environment. If no action passes exact checks, the
runtime records degraded operation and uses an actuator-bounded substitute where necessary; that
substitute is never labeled certified.

### Device and validation contract

The production role split is intentional:

- controller, plant, online BPTT, and hard candidate-evidence graphs execute on GPU in GPU runs;
- estimator updates and point-model/sample construction remain CPU-canonical;
- BPTT compilation/warmup is completed before the warm control loop;
- realtime mode submits one immutable, single-flight background candidate job and publishes only
  at a controller boundary;
- logical-simulation mode remains available for deterministic development tests, but the final
  defaults and CLI development/final profiles use realtime mode.

Every BPTT runtime input leaf remains digest-bound. Candidate leaves, final loss/gradient/update
metrics, device identity, validation inputs, and snapshot lineage are persisted. Validation does
not rerun the BPTT graph. It verifies the candidate payload and lineage, recomputes the hard report
from retained evidence, and replays the publication state transition.

Schema 7 uses only registered shared-actor codecs for persisted PyTrees; artifacts do not provide
executable/deserializing code. Older schemas and contracts must be regenerated rather than
migrated or relabeled.

### Artifact integrity

- NPZ/JSON/JSONL schemas are strict, deterministic, content-addressed, atomic, and write-once.
- Manifests bind source/runtime, semantic digests, exact inventory, hashes, and resume state.
- Core validation reconstructs schedules, tapes, traces, plant transitions, barriers, contacts,
  failure labels, metrics, paired statistics, sidecars, and video bindings.
- Candidate, Version-A/B, falsification, BPTT, and performance producers implement source-bound,
  fail-closed verification/finalization. Their pre-GPU-correction smoke artifacts below are
  historical; fresh claim-grade producer runs are still pending.
- Scientific dashboard validation decodes real PNG/MP4 content, checks frame/keyframe pixels,
  reconstructs contact sheets, and requires human visual-review records for final evidence.

The dynamics-knowledge producer uses the same no-BPTT-replay boundary. Its confirmatory metric
family can be eligible only when its full predeclared protocol completes without execution or
adaptation failures; this never authorizes a blanket safety-superiority claim.

## Fixed experiment profile

`CampaignConfig.final_core` fixes:

- K=64 policies, H=50 certificate steps, B=64 training scenarios;
- 4 obstacle-prediction samples and 4 dynamics-uncertainty samples;
- 151 control nodes at 0.02 s;
- 10 BPTT updates per burst and adaptation every 25 nodes;
- seven ordered methods:
  `nominal_only`, `analytic_cbf_hocbf`, `fixed_fallback_pcbf`,
  `handcrafted_fixed_library_plcbf`, `offline_frozen_sdcbf_style`,
  `da_plcbf_no_online_model_adaptation`, and `da_plcbf_full`;
- four ordered conditions: `static`, `dynamics_change`, `ballistic_ball`, and
  `interceptor_drone`;
- 100 paired folds per condition, producing 2,800 scheduled closed-loop trials;
- realtime, asynchronous GPU adaptation by default, while retaining wall-time/deadline and
  contention measurements; logical-simulation mode is still selectable for deterministic tests.

The final CLI rejects shape/matrix overrides. Development uses final geometry with fewer trials;
smoke uses K=16/H=2 and is never claim eligible.

## Engineering-review evidence inventory

All paths below are under:

```text
/home/tk/Desktop/mycode/crazyflow/artifacts/da_plcbf/
  engineering-review-20260901-v2-JNVFOwTb/
```

### Historical pre-GPU-correction repository gates

- CPU non-render suite: **1,281 passed, 35 skipped, 18 deselected** in 574.98 s; isolated Version B:
  **10 passed, 1 skipped** in 57.52 s. Log: `full-cpu-final-source.log`.
- GPU non-render suite: **1,308 passed, 31 skipped, 18 deselected** in 1,093.15 s; isolated
  Version B: **11 passed** in 71.27 s. Log: `full-gpu-final-source.log`.
- Render, documentation, doctest, lint/format/compile, diff-hygiene, and wheel/sdist verification:
  **18 render tests passed** (1,350 non-render tests deselected), strict docs built, **94 doctests
  passed**, all 273 Python files were already Ruff-formatted, Ruff lint and compileall passed,
  `git diff --check` passed, and fresh wheel/sdist builds passed Twine validation. Logs:
  `render-final-source.log`, `docs-build-final-source.log`, `doctests-final-source.log`,
  `package-build-final-source.log`, and `package-twine-final-source.log`.

These exact counts predate the GPU-online-adaptation correction and must not be reported as passes
of the current source. The deselections are the repository's explicit `render` tier, which runs
separately. CPU skips are GPU-only checks. Version B runs in an isolated process as documented in
`tests/README.md`; it is not omitted coverage.

The final QP implementation ranks refinement candidates against the actual normalized
primal/dual gates and applies a fixed inward repair when exact-equality iterations do not produce
an accepted float32 point. The previously failing RTX seed 3 and whole-solver JIT regressions pass
in the focused and full gates.

### Historical pre-GPU-correction producer smokes

The immutable historical smokes live in `smokes-final-v3/`. Each producer was generated and then
verified in a fresh process under the prior CPU-BPTT source contract:

- public-protocol BPTT CPU artifact: valid for its recorded historical source;
- RTX performance: **7/7** requested components passed correctness and measurement gates;
- matched Version A/B: **3/3** scheduled cases retained and historical-source verified;
- candidate quality: **5/5** outcomes, zero failures, strict verification valid;
- dynamics knowledge: **4/4** outcomes, zero operational failures, strict structural verification
  valid; the stored obsolete independent-BPTT-replay blocker describes only that old artifact and
  is not a current eligibility requirement;
- falsification: **56/56** evaluations, 54 counterexamples, 8 unique ranked counterexamples, zero
  evaluator/replay operational failures, and strict historical-source verification valid.

The dynamics result remains descriptive because its stored status/blocker are intentionally
unchanged. During falsification, LLVM once exhausted Linux virtual-memory-map entries near the
host's `vm.max_map_count`; RAM was not exhausted. The same atomic cache resumed without a sysctl
change, retained every scheduled candidate, and strictly verified.

The performance smoke is descriptive, with one warm execution sample per component. Its observed
medians were: deterministic rollout 0.497 ms, uncertain rollout 0.567 ms, fused BPTT 1.234 ms,
Version A 3.318 ms, QP 2.350 ms, Version B 6.317 ms, and host admission 0.695 ms. This is evidence
that the compiled smoke shapes ran quickly on this RTX 4090; it is not a hard-real-time, 20 Hz,
deployment, or hardware guarantee.

### Historical sealed final-shape development pilot

The review artifact is `core-development-pilot-v3/`, source-bound to
`bbf1b146c103829dbe37b56e289c7239807328ea86c80129468fc9fe1555aeac`.
It contains 28/28 complete method/condition outcomes, zero execution failures, four immutable
tapes, 28 method traces, eight adaptation-evidence sidecars, 28 dashboard sidecars, four MP4s,
32 original-resolution keyframes, four visual-review records, and 206 manifest inventory files.

Strict numerical validation under the old replay contract passed in both normal GPU-capable and
forced-CPU processes. The sealed `SHA256SUMS` inventory passes. The four 1600x900, 15 fps,
151-frame videos and all 32 keyframes were inspected at original resolution; every required visual
check passed and was bound to the exact trace/video/keyframe digests. This establishes integrity of
the historical pilot, not correctness of the GPU-corrected source.

Full DA-PLCBF development results were:

| Condition | Minimum hard margin | Failure steps | Degraded steps |
|---|---:|---:|---:|
| static | 0.262222553 | 0 | 10 |
| dynamics change | 0.0186448703 | 0 | 60 |
| ballistic ball | 0.253276177 | 0 | 34 |
| interceptor drone | 0.241955792 | 0 | 10 |

Direct review files:

- `../gpu-online-review-all-v2-20260901/` — unsealed GPU development/review run generated before
  the final formatting-only pass;
- `../gpu-bptt-online-20260901-v4.json` — final formatted-source campaign-faithful GPU timing;
- `core-development-pilot-v3/manifest.json`;
- `core-development-pilot-v3/aggregate/report.md`;
- `core-development-pilot-v3/aggregate/scientific_report.md`;
- `core-development-pilot-v3/videos/`;
- `core-development-pilot-v3/visual_reviews/`.

The manifest deliberately reports `status=incomplete` and `scientific_evidence=false`. That means
the one-fold development schedule is not final-claim eligible, not that execution is incomplete.
All 28 runs completed, but 0/72 confirmatory comparisons can support a final superiority claim.
The generic report heading “Synthetic schema/replay smoke only” likewise denotes the conservative
evidence class; the traces are real finite-horizon controller simulations, not fabricated stubs.

The first sealing attempt, `core-development-pilot-v2`, correctly refused to create a manifest
because generated `adaptation_evidence.npz` files had no inventory role. `artifacts.py` now maps
those files to `adaptation-evidence`, and a direct regression guards the mapping. No v2 manifest
was written. Because that source fix changed the source digest, the complete pilot was honestly
regenerated as v3 before sealing rather than relabeled.

Earlier `continuation-final-shape-pilot-20260831-review-v2/v3` artifacts remain diagnostic. They
identified the constraints that led to the former CPU-authoritative BPTT and compiled replay
design; those constraints are not requirements of schema 7. The artifacts predate the present
schema/source and must not be promoted.

### Engineering-review checklist

- [x] intended source, tests, dependency, CI, and test-tier changes are inspectable in one working
  tree diff;
- [x] current-source focused CPU/GPU BPTT, evidence, runtime-provenance, and renderer gates pass;
- [x] campaign-faithful full-shape GPU BPTT is measured rather than extrapolated from toy smoke;
- [x] a real asynchronous GPU campaign records submissions, hard decisions, and admitted snapshot
  use;
- [x] all four GPU-corrected ego-centric review videos/contact sheets were generated and manually
  inspected; they predate only the final formatting-only pass and remain unsealed development
  evidence without formal visual-review records;
- [x] diagnostic failures, counterexamples, limitations, and non-claim status remain visible;
- [x] current-source DA-PLCBF CPU non-render unit coverage passes in the documented Version-B
  process split (**718 passed, 4 skipped, 6 render tests deselected**);
- [ ] complete current-source repository-wide CPU/GPU/docs/package rerun after the GPU architecture
  change;
- [x] the engineering checkpoint is committed and pushed to `origin/plcbf`; no merge was performed.

## Claim-grade continuation — separate and pending

Engineering review does not authorize or complete this phase. After review and explicit direction
to begin claim-grade work, freeze the experiment schedule and dependencies against the reviewed
branch tip. Then:

1. run clean-source BPTT GPU, DA performance/contention, candidate-quality, Version-A/B, dynamics,
   falsification schedules with unique IDs and strict verification;
2. run the fixed 2,800-trial core campaign: seven methods × four conditions × 100 paired folds;
3. retain every scheduled outcome, exception, rejection, degraded interval, and counterexample;
4. render exactly four final `da_plcbf_full` videos, one for each predeclared condition;
5. perform evidence-specific visual review of every final video and bind the reviews into final
   validation;
6. finalize only after exact inventory, source/runtime hashes, hard-evidence validation, statistics,
   video/keyframe/contact-sheet checks, and review bindings all pass;
7. update `DA_PLCBF_PLAN.md` and `artifacts/da_plcbf/INDEX.md` with exact commands and hashes;
8. commit/push compact source and metadata only when explicitly authorized.

The 2,800 trials, final four videos, frozen clean evidence, compact index, commit, and push are all
pending. Do not substitute a smaller development matrix while retaining a `final` label.

## Risks and non-negotiable boundaries

1. **Source binding is fail-closed.** Any numerical source edit invalidates earlier pilots and
   smokes. Use new directories; never overwrite or migrate them in place.
2. **GPU adaptation is the production default.** Keep BPTT and hard evidence on the recorded GPU
   in GPU runs, keep the estimator CPU-canonical, and retain immutable boundary-only publication.
3. **QP float32 behavior is backend-sensitive.** Preserve the seed-3 and whole-solver-JIT
   regressions and rerun them on both CPU and RTX before broad gates.
4. **Dynamics knowledge has no replay-only blocker.** Eligibility still requires the full
   predeclared protocol and never permits a blanket superiority claim.
5. **The final core campaign is expensive.** Resource cost is not permission to weaken `final`.
6. **Timing evidence is descriptive.** The canonical isolated full-shape BPTT benchmark measured
   141.47 ms median / 141.60 ms p95 / 141.65 ms worst after JIT. Warm BPTT jobs in the fresh
   concurrent campaign took roughly 191--230 ms, and complete candidate admission took roughly
   362--443 ms. Controller tail/queue contention still prevents a hard-real-time guarantee.
7. **Candidate-study R=4/R=8 scopes are explicit.** Held-out hard-scoring shapes are not
   uncertainty-aware differentiable training. SHAC remains unavailable without a faithful
   training-only implementation.
8. **The offline SDCBF-style baseline is not an exact external reproduction.** Preserve that
   label and the source/license qualification in the plan.
9. **No hardware-flight claim exists.** Current scope is finite-horizon simulation.

## High-value file map

- `DA_PLCBF_PLAN.md` — design, evidence protocol, and claim boundary.
- `tests/README.md` / `tests/core-tests.txt` — curated and full test-tier policy.
- `crazyflow/safety/da_plcbf/experiments.py` — core trials, role placement, BPTT jobs, and seven
  methods.
- `crazyflow/safety/da_plcbf/adaptation_evidence.py` — schema-7 candidate/admission/lineage and
  no-cross-process-BPTT validation contract.
- `crazyflow/safety/da_plcbf/campaign_artifacts.py` — persisted core reconstruction and gates.
- `crazyflow/safety/da_plcbf/artifacts.py` — trace/event/manifest/video/review validation.
- `crazyflow/safety/da_plcbf/polytope_qp.py` — active-set projection and RTX float32 refinement.
- `crazyflow/safety/da_plcbf/scientific_evaluation.py` — paired metrics and inference.
- `crazyflow/safety/da_plcbf/scientific_dashboard.py` — MP4/keyframes/contact sheets/reviews.
- `crazyflow/safety/da_plcbf/candidate_protocol.py` and `ablation_campaign.py` — proposal study.
- `crazyflow/safety/da_plcbf/dynamics_knowledge_campaign.py` — oracle/estimated/R4/R8 study and
  protocol-based eligibility.
- `crazyflow/safety/da_plcbf/version_b_evidence.py` — matched Version-A/Version-B evidence.
- `crazyflow/safety/da_plcbf/falsification_experiments.py` — fixed-budget adversarial evidence.
- `benchmark/da_plcbf_gpu_bptt.py` — campaign-faithful online GPU BPTT benchmark;
  `benchmark/bptt.py` / `benchmark/da_plcbf.py` — broader performance artifacts and verifiers.
- `examples/da_plcbf/` — campaign CLIs.
- `artifacts/da_plcbf/engineering-review-20260901-v2-JNVFOwTb/core-development-pilot-v3/` — sealed
  review pilot, manifest, reports, traces, videos, reviews, and checksums.
- `artifacts/da_plcbf/README.md` / `INDEX.md` — ignored bulk evidence policy and reviewed index.

## Definition of engineering-review ready

Engineering readiness is about a reviewable, tested implementation and honest development
evidence, not a superiority result. The current GPU correction is ready for focused review; the
complete broad post-change gate remains listed explicitly above rather than borrowing the old
source digest's pass counts.

## Definition of claim-grade complete

Do not describe DA-PLCBF as claim-grade complete until all of the following are true:

- a clean frozen commit passes complete CPU/GPU/Version-B/render/docs/package gates;
- adaptation candidate origin, seeded root, publication lineage, hard admission, rollback,
  staleness, and executed-control lineage independently validate from retained evidence; the BPTT
  graph itself is not rerun across processes;
- every producer included in claims has an appropriate independent numerical validation and no
  eligibility blocker, without imposing byte-identical accelerator arithmetic;
- all clean-source final studies strictly verify;
- all 2,800 predeclared core outcomes are retained and reconstructed;
- statistics report supported and unsupported results without cherry-picking;
- exactly four final full-method MP4s have evidence-specific passing visual reviews;
- every quantitative and visual binding revalidates from immutable artifacts;
- compact plan/index metadata records exact hashes and reproduction commands;
- authorized commits are pushed and the user receives reviewable report/video paths plus all
  limitations.
