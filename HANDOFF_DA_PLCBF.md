# DA-PLCBF continuation handoff

## Read this first

This file is the authoritative checkpoint for continuing the current Codex goal on another
computer/session. The work is **paused, not complete**. No test, benchmark, GPU job, campaign,
renderer, or background worker was left running when this checkpoint was created.

The checkpoint intentionally contains substantial implemented code that has passed formatting,
lint, compilation, and many focused tests, but the most recent evidence-integrity changes have not
yet passed the complete CPU/GPU suite or a fresh real-GPU end-to-end campaign. Do not describe the
branch as ready for review and do not promote any existing local artifact to final evidence.

The active objective is still:

> On branch `plcbf`, implement and rigorously validate the shared chat's full simulation-grade
> DA-PLCBF design: finite-horizon policy-library certificates, a shared latent-residual fallback
> library refined by truncated BPTT, immutable active/candidate snapshots and hard non-regression
> admission, continuous direct-wrench and discrete full-stack filters with exact post-checks,
> online parametric dynamics adaptation and uncertainty rollouts, four required conditions,
> seven matched methods, at least 100 paired final trials per condition, strict reproducible
> evidence, and four visually reviewed MP4 dashboards. Report failures and finite-horizon limits
> without overclaiming. Do not substitute a maneuver state machine or hidden safety heuristic.

The Codex goal remains active. It was not marked complete or blocked because the user requested a
safe pause and transfer, not termination.

## Repository and branch state

- Repository: `https://github.com/tkkim-robot/crazyflow.git`
- Upstream: `https://github.com/learnsyslab/crazyflow.git`
- Working branch: `plcbf`
- Clean baseline for this branch: `7bb7aa4b49b0ec8539b17d05c37bfb97d31a4539`
- `main` and `origin/main` were already pushed at `7bb7aa4` before `plcbf` work began.
- The continuation checkpoint is the commit on `origin/plcbf` that contains this file.
- Do **not** reset `plcbf` to `main`; that would discard the implementation.
- Do **not** merge `plcbf` into `main` until the full ready-for-review definition is satisfied and
  the user explicitly asks for that merge.

On the next computer:

```bash
git fetch origin --prune
git switch plcbf
git pull --ff-only origin plcbf
git status --short --branch
git log -1 --oneline --decorate
```

The checkout should be clean immediately after pulling. Bulk numerical artifacts and MP4 files are
ignored by git and therefore do not transfer with the branch. Their development results are
recorded below for context only; regenerate evidence on the new machine.

## Sources and interpretation boundary

Primary task sources:

1. User request and this task history.
2. Shared chat, **Online Safety Filter Learning**:
   <https://chatgpt.com/s/t_6a94247c0bb081918b7acd5c6baad95c>
3. Crazyflow paper HTML:
   <https://arxiv.org/html/2606.01478v1>
4. Official upstream repository:
   <https://github.com/learnsyslab/crazyflow>

The shared post was successfully read in full earlier in this task. `DA_PLCBF_PLAN.md` records the
derived design, equations/contracts, evidence gates, and ready-for-review definition. It should be
read completely before changing implementation decisions.

The intended safety statement is deliberately narrow:

> Under the logged model/scenario samples, constraints, numerical tolerances, and finite horizon,
> the hard rollout and filter checks observed the reported margins and violation rates.

This is not an infinite-horizon, distribution-free, real-world, or hardware guarantee. Physical
failures, no-safe-fallback intervals, execution failures, deadline misses, rejected candidates, and
counterexamples must remain in the evidence. A final campaign can be useful evidence even if it
does **not** show DA-PLCBF superiority; never retune or omit folds after viewing confirmatory data.

## Upstream branch audit already completed

The latest fetched upstream heads at the pause were:

- `upstream/main`: `dede875`
- `upstream/feat.thrust_limits`: `45f9bc0`
- `upstream/feat.randomizations`: `d651db3`
- `upstream/exp`: `7595dcf`
- `upstream/feat.dynamics_example`: `8a352bf`

No upstream branch contains BPTT for this safety method, CBF/PL-CBF/SDCBF, the active/candidate
architecture, or the requested experiments. The only compatible upstream idea reused was the
symplectic vectorizer fix that prevents the excluded `dt` argument from being placed on the mapped
axis. Do not cherry-pick the remainder of `feat.thrust_limits`: its later rotor clipping maps a
stopped rotor toward calibrated airborne minimum thrust and changes powered-off/idle semantics.
The local implementation keeps filter-level motor feasibility and exact post-checks explicit.
The randomization branch is a broad parameter-sharing API redesign and was not a safe targeted fix.

## What is implemented

### Simulator/repository foundations

- Corrected integration/vectorization argument ordering across first-principles and SO-RPY
  dynamics variants.
- Added robust quaternion/rotation handling and regression tests.
- Made seeded environment resets reproduce goal sequences.
- Hardened simulation reset/step semantics and integration tests.
- Made visualization marker orientation deterministic.
- Added persistent-renderer lifecycle checks that reject camera/resolution changes until close.
- Added explicit `imageio-ffmpeg==0.6.0` and direct Optax dependency; the encoder path/version/hash
  are recorded in provenance.
- Added DA-PLCBF API documentation and strict reference-page generation.

### DA-PLCBF mathematical/runtime implementation

Implemented under `crazyflow/safety/da_plcbf/`:

- fixed-shape scenario tapes with independent named RNG streams and semantic digests;
- static, dynamics-change, ballistic-ball, interceptor-drone, and combined falsification scenes;
- swept contact/constraint checks and true-plant replay from realized motor forces;
- double-integrator reference values/filter and quadrotor Version-A direct-wrench model;
- analytic arena/altitude/speed/angular-rate/tilt/obstacle/capsule barriers;
- coupled motor-force polytope, exact small active-set projection, KKT checks, and independent
  solver cross-checks;
- shared structural plus latent-residual fallback actor, fixed structural slots, adaptive slots,
  trainable skill codes/durations, common hover/brake horizon tail, and task-agnostic observation;
- hard sampled policy values, conservative differentiable values, normalized descriptors,
  coverage/redundancy/diversity/action/rate/terminal/trust losses;
- fixed-budget truncated BPTT for the policy library;
- immutable content-addressed active/candidate snapshots, model versions, hard admission,
  staleness rejection, atomic boundary publication, rollback, and explicit degraded operation;
- deterministic positive-hard-value selection with an admissible-set proxy and logged hysteresis;
- low-dimensional online mass/drag/wind/rotor-efficiency estimator and Cartesian R=4/R=8 model
  samples;
- continuous Version-A PL-CBF path and nonlinear/discrete Version-B path through Crazyflow's full
  allocation/rotor/integrator stack with exact post-checks;
- seven matched methods:
  `nominal_only`, `analytic_cbf_hocbf`, `fixed_fallback_pcbf`,
  `handcrafted_fixed_library_plcbf`, `offline_frozen_sdcbf_style`,
  `da_plcbf_no_online_model_adaptation`, and `da_plcbf_full`;
- confirmatory paired inference with retained failures, Bonferroni family accounting, exact paired
  sign analysis, bootstrap raw settings, and a global-conjunction superiority summary;
- separate candidate-quality, dynamics-knowledge, Version-A/B, performance, and falsification
  campaigns so unlike claims are not mixed into the core safety study.

The structural fallback initialization is a grid of policy parameters evaluated concurrently; it
is not a runtime maneuver state machine. Scenario attacker modes define the environment, not the
fallback controller. If no candidate/fallback passes exact checks, the code emits an explicit
degraded result and an actuator-bounded midpoint/zero substitute where necessary; that substitute
is never labeled certified.

### Artifact and scientific-integrity implementation

- Deterministic strict NPZ/JSON/JSONL schemas, atomic write-once files, semantic SHA-256 digests,
  manifest inventories, `SHA256SUMS`, source/runtime binding, orphan rejection, and resume checks.
- Main campaign validation reconstructs the schedule, tapes, traces, plant transitions, barriers,
  contacts, failure labels, metrics, paired statistics, reports, sidecars, and rendered-video
  bindings.
- Online adaptation evidence now retains proposal-active, decision-active, candidate, and
  publication-active snapshots, complete hard-validation arrays/thresholds/report, causal context,
  decision/publication lineage, and whether the admitted snapshot drove executed control.
- The newest patch also persists the actual BPTT execution backend/device and replays the exact
  deterministic BPTT burst on that original device, requiring byte-exact candidate leaves and
  matching final gradient/loss/update evidence. Cold-start is normally GPU; online work is normally
  isolated on CPU. This newest cross-device path compiles and lints but still needs the real-GPU
  mixed-device test listed below.
- Final-intended core campaigns now reject dirty source at creation/finalization, while smoke and
  development runs may remain explicitly dirty/non-claim-grade.
- Candidate-ablation finalization now checks source-before/source-after, strictly reconstructs
  before writing the completion marker, withholds the marker on failure, and auto-verifies in CLI.
- Dynamics-knowledge campaigns already had exact startup snapshot reuse, source guards,
  reconstruction, and marker gating.
- Version-A/B evidence now has exact source bracketing, clean-source mode, strict verify CLI, and
  no-write-on-drift behavior.
- Falsification now has deterministic candidate/tape regeneration, crash-stable success/failure
  caches, true-plant/barrier/contact/failure replay, fixed search/ranking reconstruction, strict
  whole-campaign manifests, completion markers, and current-source verification.
- BPTT and DA performance artifacts now use exact schemas, raw-derived summary reconstruction,
  source/runtime binding, XML/STL asset hashing, clean-source flags, write-once output, strict
  verification CLIs, and rehashed-tamper regressions.
- Scientific dashboard finalization fully decodes PNGs, compares keyframe pixels to decoded MP4
  frames, deterministically reconstructs contact sheets, and forces video replay for scientific
  evidence. Fake PNG headers cannot satisfy the review gate.

## Important fixed profiles

Core final profile (`CampaignConfig.final_core`):

- K=64 policies;
- H=50 certificate steps;
- B=64 training scenarios;
- obstacle prediction samples=4;
- dynamics uncertainty samples=4;
- 151 control nodes at dt=0.02 s;
- 10 BPTT updates per burst;
- adaptation interval 25 nodes;
- seven ordered methods;
- four ordered conditions: `static`, `dynamics_change`, `ballistic_ball`,
  `interceptor_drone`;
- 100 paired folds per condition;
- 2,800 scheduled closed-loop trials;
- logical-simulation adaptation for safety evidence, with wall-time/deadline measurements retained.

The final CLI forbids overriding this matrix or its shapes. Development profile defaults use the
same final shape but allow fewer trials/methods/conditions. Smoke profile is K=16/H=2 and is never
claim eligible.

## Test and audit status at this checkpoint

### Passing at the pause

- Whole-tree `git diff --check`: pass.
- Whole-tree `ruff format --check`: pass after formatting the final audit edits.
- Whole-tree `ruff check`: pass.
- Whole-tree Python `compileall` over `crazyflow`, `examples`, `benchmark`, and `tests`: pass.
- New clean-final gate plus adaptation-evidence unit file: 4 passed in 2.78 s.
- Benchmark focused suites (`test_bptt.py`, `test_performance_benchmark.py`): 18 passed.
- Candidate hardening tests: 13 passed; candidate scoped Ruff/compile pass.
- Version-A/B hardening tests: 7 fast tests passed; real CPU integration smoke passed in 8.95 s.
- Falsification focused suite: 20 passed; scoped Ruff/compile pass.
- Earlier candidate protocol/ablation focused suite before the final source-guard patch: 47 passed.
- Earlier dynamics-knowledge focused suite: 31 passed, 1 deselected; scoped Ruff clean.
- Earlier documentation build: `pixi run -e docs docs-build` passed.
- Earlier Markdown/docstring suite: 94 passed.
- Earlier simulator-foundation batch reached 218 passed, 24 skipped, 12 deselected; its one BPTT
  constructor failure was caused by an in-flight strict-schema change and is covered by the later
  passing 18-test benchmark suite.
- A prior full repository run before the latest integrity patches reported 1,213 passed,
  31 skipped, 18 deselected. It is historical only and must not replace a fresh whole-suite run.

### Incomplete/failed runs that must not be hidden

- A later full `tests/unit/safety/da_plcbf` CPU run reached approximately 97%, then showed two
  failures and aborted fatally inside the JAX compilation cache while executing a Version-B JIT
  test. Pytest died before printing the two failure summaries. Rerun the entire safety suite with a
  fresh compilation-cache directory; do not assume those were resolved.
- The newest device-aware adaptation/BPTT replay path has not run its full-development numerical
  replay test after the final patch.
- No fresh full CPU suite, full GPU suite, render suite, docs suite, or package build has run after
  all final audit changes.
- A real GPU BPTT smoke wrote a development-only artifact at
  `/tmp/crazyflow-benchmark-audit.UfpdEp/bptt-public-gpu-smoke.json`, but it was produced while the
  repository was dirty and its strict CLI verification was not run before the pause. `/tmp` will
  not transfer to another computer.
- A prior real-GPU falsification smoke completed 56/56 candidates with 0 evaluator operational
  failures, found 54 counterexamples, retained 8 unique ranked tapes, and replayed the worst tape
  across all seven methods (7/7 replay successes). It then correctly refused manifest/marker
  finalization because concurrent source edits changed the digest. Historical reconstruction
  passed only in explicit non-promotion mode. Local path:
  `/tmp/crazyflow-falsification-full-smoke.9bU4gY`.
- Prior candidate and dynamics GPU smokes similarly exercised real work and correctly rejected
  strict promotion after concurrent source drift. Regenerate them on the frozen tree.

## Hardware observed on the original machine

- NVIDIA GeForce RTX 4090, 24,564 MiB
- NVIDIA driver 560.35.03
- Compute capability 8.9
- JAX 0.11.1 / jaxlib 0.11.1
- `jax.default_backend()` returned `gpu`
- Python environments are managed by Pixi.
- Approximately 105 GiB disk space was free at the pause.

The latest development-only BPTT GPU smoke used the public smoke shape (2 environments, four
actions, two optimizer updates): compile 6.0839 s, warmup 0.02991 s, raw executions 0.01960 s and
0.02148 s, mean 0.02054 s, finite nonzero gradients and parameter change. It is functional evidence
only, not a paper comparison or claim-grade timing.

Earlier clean-main forward-simulator development measurements on the same 4090 reached about
1.05 billion world-steps/s at 262,144 one-drone worlds with a fused 50-step rollout. Those CSVs
were produced before this final branch was clean and are not final evidence. The paper reports
about 700 million steps/s at one million worlds; rerun a clean scaling sweep and state the world
count/hardware/protocol difference. No MPPI implementation exists in this repository, so the
paper's MPPI number has not been reproduced here.

## Local artifacts that existed before transfer

`artifacts/da_plcbf/` is intentionally ignored except for `README.md` and `INDEX.md`. The original
machine had about 12 MiB of development artifacts, including:

- `20260831-gpu-smoke-core-k16-v3/`
- `20260831-visual-development-v1/` with one dynamics-change MP4/contact sheet
- `20260831-candidate-ablation-smoke-v1/` and `v2/`
- several dynamics-knowledge smoke directories
- older CPU/GPU BPTT JSON files
- older Version-A/B final-shape JSON/Markdown

These predate the newest schemas/source digest and are not current evidence. The committed
`artifacts/da_plcbf/INDEX.md` deliberately states that no scientific run has passed every gate.
Do not edit that statement until a new final directory passes strict numerical, source, replay,
and visual validation.

## Immediate continuation sequence

### 1. Establish a clean, isolated test environment

```bash
git status --short --branch
nvidia-smi
mkdir -p /tmp/crazyflow-jax-cache-continuation
export JAX_COMPILATION_CACHE_DIR=/tmp/crazyflow-jax-cache-continuation
pixi run -e gpu-tests python -c 'import jax; print(jax.__version__, jax.default_backend(), jax.devices())'
```

Do not edit source while any evidence-producing command is running. Source-drift guards are
deliberately fail-closed and will invalidate/withhold final markers.

### 2. Validate the newest integrity changes first

```bash
pixi run ruff format --check .
pixi run ruff check .
git diff --check
pixi run -e tests python -m compileall -q crazyflow examples benchmark tests

pixi run -e tests pytest -q \
  tests/unit/safety/da_plcbf/test_adaptation_evidence.py \
  tests/unit/safety/da_plcbf/test_experiments.py::test_final_intended_campaign_requires_clean_committed_provenance \
  tests/unit/safety/da_plcbf/test_experiments.py::test_full_development_slice_logs_model_and_candidate_lifecycle

pixi run -e tests pytest -q \
  tests/unit/test_bptt.py \
  tests/unit/safety/da_plcbf/test_performance_benchmark.py \
  tests/unit/safety/da_plcbf/test_ablation_campaign.py \
  tests/unit/safety/da_plcbf/test_version_b_evidence.py \
  tests/unit/safety/da_plcbf/test_falsification_experiments.py \
  tests/unit/safety/da_plcbf/test_dynamics_knowledge_campaign.py
```

Then run a real GPU core smoke that contains both cold-start GPU BPTT and post-startup CPU BPTT,
and strictly reconstruct its `adaptation_evidence.npz`. The easiest complete route is a fresh core
smoke campaign with `da_plcbf_full` and enough control nodes to submit/resolve an online job. Do not
accept only a cold-start proof.

### 3. Run fresh strict smokes for every evidence producer

Use fresh directories under `/tmp` or ignored `artifacts/da_plcbf/`; never overwrite old evidence.

```bash
# BPTT smoke + strict current-source/runtime verification
pixi run -e gpu-tests python benchmark/bptt.py \
  --protocol public --device gpu --smoke --repeats 2 \
  --output /tmp/crazyflow-bptt-smoke.json
pixi run -e gpu-tests python benchmark/bptt.py \
  --verify-artifact /tmp/crazyflow-bptt-smoke.json

# DA performance smoke + strict verification
pixi run -e gpu-tests python benchmark/da_plcbf.py \
  --device gpu --preset smoke --components all --repeats 3 --warmups 1 \
  --contention none --output /tmp/crazyflow-da-performance-smoke.json
pixi run -e gpu-tests python benchmark/da_plcbf.py \
  --verify-artifact /tmp/crazyflow-da-performance-smoke.json

# Candidate campaign
pixi run -e gpu-tests python examples/da_plcbf/candidate_ablation.py run \
  --profile smoke --output /tmp/crazyflow-candidate-smoke --no-resume
pixi run -e gpu-tests python examples/da_plcbf/candidate_ablation.py verify \
  --output /tmp/crazyflow-candidate-smoke

# Dynamics-knowledge campaign
pixi run -e gpu-tests python examples/da_plcbf/dynamics_knowledge.py run \
  --profile smoke --output /tmp/crazyflow-dynamics-smoke --no-resume
pixi run -e gpu-tests python examples/da_plcbf/dynamics_knowledge.py verify \
  --output /tmp/crazyflow-dynamics-smoke

# Version A/B evidence
pixi run -e gpu-tests python examples/da_plcbf/version_b_evidence.py \
  --profile smoke --device gpu --output /tmp/crazyflow-version-ab-smoke.json
pixi run -e gpu-tests python examples/da_plcbf/version_b_evidence.py \
  --verify-artifact /tmp/crazyflow-version-ab-smoke.json

# Falsification
pixi run -e gpu-tests python examples/da_plcbf/falsify.py run \
  --profile smoke --run-dir /tmp/crazyflow-falsification-smoke
pixi run -e gpu-tests python examples/da_plcbf/falsify.py verify \
  --run-dir /tmp/crazyflow-falsification-smoke
```

The exact flags are documented by each command's `--help`. If a smoke fails, preserve its failure
cache/directory and turn the cause into a regression test; do not delete the failure and rerun as
if it never happened.

### 4. Run complete repository gates

Use a fresh JAX cache if the prior fatal compilation-cache issue recurs.

```bash
pixi run -e tests pytest -q
pixi run -e gpu-tests pytest -q
pixi run -e gpu-tests pytest -q -m render
pixi run -e docs docs-build
pixi run -e tests test-docs
pixi run -e dist build
```

The default pytest configuration excludes the `render` marker; the separate render command is
required. Record exact pass/skip/deselect counts and durations in this file or the final report.

### 5. Run a final-shape development pilot before freezing claims

This is the last place to find implementation/display problems and revise source without touching
confirmatory data:

```bash
pixi run -e gpu-tests python examples/da_plcbf/campaign.py run \
  --profile development \
  --run-dir artifacts/da_plcbf/continuation-final-shape-pilot \
  --trials 1 --root-seed 20260831

pixi run -e gpu-tests python examples/da_plcbf/campaign.py render \
  --run-dir artifacts/da_plcbf/continuation-final-shape-pilot \
  --methods da_plcbf_full \
  --conditions static,dynamics_change,ballistic_ball,interceptor_drone \
  --videos-per-condition 1 --fps 15 --width 1600 --height 900 --keyframes 8
```

Review all four contact sheets and representative full-resolution keyframes with the image viewer.
Check data numerically against trace/events/sidecar, not by appearance alone. If labels, camera,
occlusion, unsafe/degraded coloring, event annotations, scales, timing, or unavailable-evidence
labels are unclear, revise the renderer and repeat source tests plus the pilot. Do not inspect final
confirmatory outcomes and then change the method/configuration.

### 6. Freeze source in a clean commit

After every development revision and all gates pass:

```bash
git status --short
git add -A
git commit -m "Complete DA-PLCBF implementation and validation harness"
git push origin plcbf
git status --short
```

The status must be clean before every claim-grade command using `--require-clean-source`. Ignored
artifact outputs do not make Git dirty. Do not edit tracked files until all claim-grade runs,
rendering, reviews, manifests, and replay validation are complete.

### 7. Run claim-grade independent experiments

Use unique run IDs. Suggested commands:

```bash
# Paper-informed BPTT reconstruction on CPU (paper timing scope) and GPU (4090 scope)
pixi run -e gpu-tests python benchmark/bptt.py \
  --protocol paper --device cpu --repeats 5 --require-clean-source \
  --output artifacts/da_plcbf/final-bptt-paper-cpu.json
pixi run -e gpu-tests python benchmark/bptt.py \
  --verify-artifact artifacts/da_plcbf/final-bptt-paper-cpu.json --require-clean-source

pixi run -e gpu-tests python benchmark/bptt.py \
  --protocol paper --device gpu --repeats 5 --require-clean-source \
  --output artifacts/da_plcbf/final-bptt-paper-gpu.json
pixi run -e gpu-tests python benchmark/bptt.py \
  --verify-artifact artifacts/da_plcbf/final-bptt-paper-gpu.json --require-clean-source

# DA K/B/R/H sweep, tail latency, correctness, and contention
pixi run -e gpu-tests python benchmark/da_plcbf.py \
  --device gpu --preset final --components all --repeats 50 --warmups 5 \
  --contention cpu,gpu --require-clean-source \
  --output artifacts/da_plcbf/final-da-performance.json
pixi run -e gpu-tests python benchmark/da_plcbf.py \
  --verify-artifact artifacts/da_plcbf/final-da-performance.json --require-clean-source

# Candidate-quality confirmatory schedule (100 folds)
pixi run -e gpu-tests python examples/da_plcbf/candidate_ablation.py run \
  --profile confirmatory --output artifacts/da_plcbf/final-candidate-ablation --no-resume
pixi run -e gpu-tests python examples/da_plcbf/candidate_ablation.py verify \
  --output artifacts/da_plcbf/final-candidate-ablation

# Dynamics-knowledge matched final schedule (100 trials per variant)
pixi run -e gpu-tests python examples/da_plcbf/dynamics_knowledge.py run \
  --profile final --output artifacts/da_plcbf/final-dynamics-knowledge --no-resume
pixi run -e gpu-tests python examples/da_plcbf/dynamics_knowledge.py verify \
  --output artifacts/da_plcbf/final-dynamics-knowledge

# Matched Version-A/Version-B evidence
pixi run -e gpu-tests python examples/da_plcbf/version_b_evidence.py \
  --profile final --device gpu --require-clean-source \
  --output artifacts/da_plcbf/final-version-ab.json
pixi run -e gpu-tests python examples/da_plcbf/version_b_evidence.py \
  --verify-artifact artifacts/da_plcbf/final-version-ab.json --require-clean-source

# Fixed-budget empirical falsification with seven-method worst-tape replay
pixi run -e gpu-tests python examples/da_plcbf/falsify.py run \
  --profile final --run-dir artifacts/da_plcbf/final-falsification
pixi run -e gpu-tests python examples/da_plcbf/falsify.py verify \
  --run-dir artifacts/da_plcbf/final-falsification
```

Also rerun the original Crazyflow forward-throughput protocol on the frozen commit:

```bash
pixi run -e benchmark python benchmark/main.py \
  --device=gpu --worlds=262144,524288,1048576 \
  --n_steps=50 --rollout_steps=50 --include_gym=False
```

Resource exhaustion at a larger world count is a retained scaling boundary, not a reason to report
only the last successful point as if it were the paper's one-million-world protocol.

### 8. Run the predeclared 2,800-trial core campaign

This can take many physical hours. Resume is supported, but only with identical config, source,
commit/branch/dirty state, tapes, and valid recorded successes.

```bash
pixi run -e gpu-tests python examples/da_plcbf/campaign.py run \
  --profile final \
  --run-dir artifacts/da_plcbf/final-core-20260831 \
  --root-seed 20260831
```

If interrupted after valid outcomes are committed:

```bash
pixi run -e gpu-tests python examples/da_plcbf/campaign.py run \
  --profile final \
  --run-dir artifacts/da_plcbf/final-core-20260831 \
  --root-seed 20260831 --resume
```

Do not use final-profile overrides; the CLI rejects them. Inspect:

- all 2,800 scheduled outcomes retained;
- zero unrecorded/missing assignments;
- every execution exception preserved;
- online methods have post-startup candidate resolution and campaign-level proof that an admitted
  online snapshot drove executed control;
- full method produces accepted estimator updates in dynamics-change folds;
- physical failures and degraded periods agree with true-state/tape replay;
- confirmatory and exploratory conclusions match the exact retained data;
- no broad superiority statement unless every predeclared confirmatory member supports it.

### 9. Render exactly four final videos and perform real visual review

```bash
pixi run -e gpu-tests python examples/da_plcbf/campaign.py render \
  --run-dir artifacts/da_plcbf/final-core-20260831 \
  --methods da_plcbf_full \
  --conditions static,dynamics_change,ballistic_ball,interceptor_drone \
  --videos-per-condition 1 --fps 15 --width 1600 --height 900 --keyframes 8
```

For each video, inspect the canonical contact sheet and full-resolution keyframes. Create one
canonical `visual_reviews/<video-stem>.md` using `VisualReviewRecord` and
`write_visual_review_record`; rendering must never auto-assert that a display is legible. Every
review must address all eight checks:

1. `original_resolution_inspected`
2. `labels_legible_without_console`
3. `unsafe_and_degraded_visibly_distinct`
4. `overlays_agree_with_trace`
5. `event_annotations_agree_with_trace`
6. `camera_and_occlusion_acceptable`
7. `scales_units_and_timing_clear`
8. `unavailable_evidence_explicit`

If any check fails, record `revise`, change the renderer only before finalization, rerun the
necessary source/test/campaign work required by source binding, and inspect again. Passing review
records must contain evidence-specific notes rather than boilerplate.

Finalize and replay-validate only after all four reviews pass:

```bash
pixi run -e gpu-tests python examples/da_plcbf/campaign.py finalize \
  --run-dir artifacts/da_plcbf/final-core-20260831 --verify-replay
pixi run -e gpu-tests python examples/da_plcbf/campaign.py validate \
  --run-dir artifacts/da_plcbf/final-core-20260831 --verify-replay
```

Final validation checks exact file inventory, hashes, schemas, physical/numerical reconstruction,
statistics, MP4 codec/frame count/duration/dimensions/non-static content, exact decoded-frame
digest, keyframe pixels, deterministic contact sheet, and visual-review bindings.

### 10. Publish metadata only after evidence is immutable

After every strict validator passes:

- update `DA_PLCBF_PLAN.md` checkboxes truthfully;
- update `artifacts/da_plcbf/INDEX.md` with run ID, manifest SHA-256, `SHA256SUMS` SHA-256, storage
  location, exact reproduction command, and review status;
- record paper comparisons with protocol/hardware differences;
- state all counterexamples, unsupported endpoints, missed deadlines, and finite-horizon limits;
- commit only compact metadata, not bulk NPZ/MP4 output;
- push `plcbf`;
- mark the Codex goal complete only when no required gate remains.

## Known risks and things to verify carefully

1. **Newest mixed-device BPTT proof is not fully tested.** It now derives and stores backend/device
   from the actual trained result and replays on that device. Confirm cold-start GPU plus online CPU
   in one real trial and strict artifact reconstruction.
2. **Legacy adaptation artifacts are intentionally incompatible.** New validation requires
   `bptt_execution_backend` and `bptt_execution_device_id`. Regenerate old sidecars; do not silently
   migrate or treat them as current evidence.
3. **Full safety-suite fatal exit needs reproduction.** Use a fresh JAX compilation cache and retain
   the first actual failing trace/summary.
4. **Final core cost is large.** K64/H50 plus online CPU-isolated BPTT and strict replay can take many
   hours. Do not disable replay or reduce folds to make it finish faster while preserving a “final”
   label.
5. **Disk use can be substantial.** K64/H50 fallback rollout sidecars across 2,800 trials may use
   many GiB. Check free space before launch and archive a complete content-addressed directory.
6. **Performance is descriptive.** Version-A/Version-B earlier dirty-tree final-shape probes were
   roughly 166 ms and 468 ms per decision respectively and missed the 20 ms deadline. Regenerate
   clean evidence; do not claim real-time operation unless new retained timings support it.
7. **Candidate ablation boundaries are explicit.** R=4/R=8 there are held-out hard-scoring shapes;
   they are not falsely labeled as uncertainty-aware differentiable training. SHAC is explicitly
   unavailable until a faithful training-only implementation exists.
8. **Offline SDCBF-style baseline is labeled as style/matched learned library, not an exact external
   reproduction.** Its source/license constraints are documented in the plan.
9. **No hardware flight claim.** No physical platform/authority was provided; simulation completion
   is the current gate.
10. **Do not merge the upstream rotor-clipping branch wholesale.** It changes idle semantics.

## High-value file map

- `DA_PLCBF_PLAN.md` — full design/evidence plan and ready-for-review definition.
- `crazyflow/safety/da_plcbf/experiments.py` — core trial, BPTT job, seven-method campaign.
- `crazyflow/safety/da_plcbf/campaign_artifacts.py` — persisted core reconstruction and gates.
- `crazyflow/safety/da_plcbf/adaptation_evidence.py` — exact candidate/admission/BPTT proof.
- `crazyflow/safety/da_plcbf/artifacts.py` — trace/event/manifest/video/review validation.
- `crazyflow/safety/da_plcbf/scientific_evaluation.py` — paired metrics/inference.
- `crazyflow/safety/da_plcbf/scientific_dashboard.py` — MP4/keyframes/contact-sheet/review records.
- `crazyflow/safety/da_plcbf/candidate_protocol.py` and `ablation_campaign.py` — proposal study.
- `crazyflow/safety/da_plcbf/dynamics_knowledge_campaign.py` — oracle/estimated/R4/R8 study.
- `crazyflow/safety/da_plcbf/version_b_evidence.py` — matched Version A/B evidence.
- `crazyflow/safety/da_plcbf/falsification_experiments.py` — fixed-budget adversarial evidence.
- `benchmark/bptt.py` — public/paper-informed Crazyflow BPTT benchmark and verifier.
- `benchmark/da_plcbf.py` — K/B/R/H performance/correctness/contention benchmark and verifier.
- `examples/da_plcbf/` — campaign CLIs.
- `tests/unit/safety/da_plcbf/` — mathematical, runtime, scientific, artifact, and tamper tests.
- `artifacts/da_plcbf/README.md` — ignored-artifact policy.
- `artifacts/da_plcbf/INDEX.md` — committed reviewed-evidence index (currently empty by design).

## Definition of done

Do not call this ready until all of the following are true:

- complete format/lint/compile/docs/package/CPU/GPU/render gates pass on the final source;
- DA-PLCBF BPTT gradients, learning, and exact candidate origin replay pass directly;
- active/candidate isolation, hard admission, stale/bad/nonfinite/slow rejection, rollback, and
  executed-control lineage pass fault injection and end-to-end tests;
- dynamics adaptation, R4/R8 uncertainty, Version A/B, candidate, performance, and falsification
  final artifacts strictly verify against clean current source/runtime;
- the full 2,800-trial paired core schedule is retained and strictly reconstructed;
- statistics state supported and unsupported results without cherry-picking;
- exactly four final full-method MP4s exist and each has an evidence-specific passing visual review;
- replay regenerates/validates every visual and quantitative binding;
- plan and compact evidence index are updated with exact hashes/commands;
- `plcbf` is pushed and the user receives clickable video/report paths plus honest limitations.
