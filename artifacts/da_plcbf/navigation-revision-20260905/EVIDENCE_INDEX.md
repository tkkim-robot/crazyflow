# Navigation revision evidence index

This index describes the working revision after base commit
`00e89a742a1271b93655bf1bb4581a667dc13a14`. The source is not yet committed.
Read the repository-level `DA_PLCBF_NAVIGATION_REVIEW.md` for supported claims and limitations.
All numerical evidence below is preserved separately from the earlier competent revision.

## Source and protocol

- `current-source/SOURCE.tar.gz`, `current-source/SOURCE_SHA256.json`: final working source over
  the exact base commit, including code, tests, benchmark scripts and review notes.
- `ARTIFACT_SHA256.json` here and in the sibling learning root: complete path/size/SHA inventories,
  including failed attempts. Only the manifest itself is excluded from each inventory.

- `CAMPAIGN_PROTOCOL.json`, `CAMPAIGN_SOURCE.tar.gz`, `CAMPAIGN_SOURCE_AUDIT.json`:
  original frozen method, candidate checkpoint, teacher seed 7, world seeds 100–109;
  169 archived source hashes verified before evaluation.
- `DENSE_CONFIRMATION_PROTOCOL.json`: separate predeclared 16-obstacle seeds 200–209.
- `CAMPAIGN_OUTPUT_PATCH.json`, `.diff`, `.tar.gz`: only rejected/unused proposal reporting,
  export validation and finite-only diagnostic aggregation changed. Executed rows remain finite.
- `DENSE206_OUTPUT_PATCH_REPLAY_AUDIT.json`: all dense physical states and all 53 final
  checkpoint arrays match the preserved pre-patch seed206 partial artifact exactly.
- `ISOLATED_DISTURBANCE_PROTOCOL.json`, `ISOLATED_DISTURBANCE_EXECUTION_PROTOCOL.json`:
  wind-only and payload-only control conditions, same world seeds 100–109 and frozen method.
- `STATIC_COMPOSITION_DEVELOPMENT_PROTOCOL.json`: one static-obstacle seed0 development run.
- `REPLAY_JAX_CACHE.tar.gz`, `REPLAY_JAX_CACHE_MANIFEST.json`: archived eight-entry compiled
  cache used by the exact dense replays, with environment-specific restoration limits.
- `ENVIRONMENT.json`: exact software/GPU and launch/cache settings, and replay/timing scope.
- `PACED_SOURCE*`, `PACED2_SOURCE*`, `PACED3_SOURCE*`: separately captured measured-execution attempts.
  These include later current-source fixes and do not redefine the frozen campaign.

## Failure mechanism and controller repair

- `failure_diagnosis.md`, `failure_diagnosis_audit.json`, `branch_clearance_audit.json`:
  independently checked diagnosis and collision-censored branch accounting.
- `legacy-estimated-replay/`, `legacy-oracle-replay-2/`: 62 original commands/rejection vectors
  reconstructed exactly, with full learner/Adam states at every replay boundary.
- `estimated-factorial/`: 144 state/model/parameter/compensation/cadence cells and raw trajectories.
- `held-operational-fixture/`: the exact 6.2 s crossing residual and explicit predictive-QP repair,
  unchanged limits and tolerance, plus synchronized controller-only timings.
- `development-dense16-oracle-seed0/pre_failure/`: exact replay of 41 updates, same-state probes
  at pre-failure and later states, and bounded independent route/authority attempts. Failed
  routes do not establish physical impossibility.

- `dense-confirmation-repair/combined-seed209/pre_failure/`: symmetric four-state probe,
  50 updates and all recorded commands replay exactly. At the first adaptive loss, frozen's
  direct fallback is locally valid; the preceding state reverses that ranking. This is not a
  complete counterfactual recovery episode.
- `dense-confirmation-repair/combined-seed206/pre_failure_replay_v2/`:
  50 updates reproduce the full continuation state exactly with the declared cache settings;
  four first-failure same-state comparisons do not rescue the frozen physical state.

## Learner evidence

Sibling directory `../learning-revision-20260905/` contains:

- `reference-ablation-seed7/`, `reference-ablation-seed9/`, `reference-ablation-seed11/`:
  nominal warmup, compensation matrix, moving-state bank, seven objective/rate variants,
  full Adam continuation and immutable nominal reference contracts.
- `reference-coverage-seed7/`: fixed encounter-state comparison of the initial variants.
- `stabilization-seed7/`, `stabilization-coverage-seed7/`: network-only, offset-only and
  globally capped update variants, weighted gradient diagnostics, coverage through 80 updates.
- `stabilization-closed-loop/`: fresh corrected 0–8 s oracle and estimated-model branches.
- `estimator-replay-sensitivity/`: fixed-telemetry rate/noise experiment, separate from closed-loop
  control and learning.

Read `docs/da_plcbf_state_conditioned_learning.md` and
`docs/da_plcbf_revision_ablation_index.md` for configuration tables and commands.

## Navigation outcomes and videos

- `heldout-shard-a/`, `heldout-shard-b/`, `heldout_statistics.json`:
  ten unchanged and ten combined paired worlds, eight moving obstacles, eight waypoints, 40 s.
- `dense-confirmation/`, `dense-confirmation-repair/`, `dense_confirmation_statistics.json`:
  ten combined paired worlds with 16 moving obstacles. The failed original seed206 export is
  excluded; its numerically identical repaired export is counted exactly once.
- `isolated-wind/`, `isolated-payload/`, `ISOLATED_DISTURBANCE_STATISTICS.json`,
  `ISOLATED_DISTURBANCE_COMPOSITION_AUDIT.json`: 20 paired isolated-event controls, each method
  passes all ten per condition; geometry, event models and every pre-event prefix verified.
- `heldout-shard-a/combined-seed100/navigation_comparison_demo.mp4`:
  representative candidate-checkpoint held-out combined mission, with video/frame/input hashes.
- `development-dense16-oracle-seed0/navigation_comparison_demo.mp4`:
  development failure comparison. Its divergence begins before wind, so it is not a disturbance
  recovery demonstration. Side-by-side diagnostic probes explain the distinct physical paths.
- `campaign-comparison-figures-v2/`: final raw paired and uncertainty PNG/PDF figures, all 30
  initial paired draws, with exact source/input provenance. The original figure folder is a
  superseded layout only, with identical numerical observations.
- `ENCOUNTER_SEVERITY_AUDIT.json`: raw per-world nominal blocking episodes/fractions and executed
  policy-dual activity; the 16-obstacle family has three to eight distinct episodes.
- Each completed mission retains `world.json`, `config.json`, schedule, source hash, full dense
  plant states, raw candidate/actuator/operational records, symmetric probes, checkpoint snapshots,
  final learner/Adam checkpoint and nominal reference contracts.

## Preserved failed or limited outputs

- `legacy-oracle-replay/` is the incomplete first replay attempt. Its assertion incorrectly
  compared inferred wind to oracle truth. Use `legacy-oracle-replay-2/` for the completed replay.
- `dense-confirmation/combined-seed206/` finished the numerical run but failed strict export on
  a nonfinite **rejected** QP proposal. Keep its partial files; use `dense-confirmation-repair/`
  for the finalized result. No applied command or physical state was nonfinite.
- `paced-validation/` and `paced-validation2/` are failed online-learning runtime checks:
  both perform zero learning updates and have measured deadline misses. Safe mission completion
  in those outputs is not a successful adaptation-budget result.
- `dense-confirmation-repair/combined-seed206/pre_failure/` is the excluded first analysis
  replay, launched without the campaign compilation-cache settings. Full-state parity failed
  at 5.77e−6; no causal rows were emitted. The v2 replay restores exact parity.
- The provisional seed0 eight-obstacle videos use the explicitly labeled provisional learning
  rate. Principal paired conclusions use the separately frozen candidate checkpoint.
- `development-dense16-estimated-seed0/` has a legacy “same point model” display label inherited
  from the frozen renderer; the methods have separate estimates with the same allowed model
  information. Use numeric metadata and the updated current label; do not interpret the estimates
  as numerically identical.
- Development eight-obstacle outputs with `POST_SAVE_PROVENANCE_REPAIR.json` had only their source
  record repaired after numerical export. The numerical arrays were not altered.

- `paced-validation3/` is the final startup-fixed timing attempt: zero deadline misses in both
  branches, but no finite updates launched from 335 allowed opportunities. It still fails online
  learning runtime feasibility. `paced_runtime_audit_three_attempts.json` independently checks
  all three recorded attempts and separates task success from completed/published/used learning.
- `PAIRED_DENSE_FAILURE_DIAGNOSIS.md` explains both opposite shell failures and local probes.
- `static-composition-development/`, `STATIC_COMPOSITION_DEVELOPMENT_AUDIT.json` validate the
  static trajectory provider in one declared development mission: both methods complete eight
  waypoints at 30.96 s, with positive shell and zero degraded controls.

## Focused checks

- `final-navigation-tests.txt`: 49 passed, 2 rendering tests deselected by default configuration.
- `final-execution-and-runtime-tests.txt`: 27 held-policy/runner/runtime tests pass (overlap above).
- `final-egl-render-tests.txt`: the 2 EGL rendering tests passed separately.
- `FINAL_VIDEO_VERIFICATION.json`: both main videos and all ten associated trace/frame files
  match their original video audits; recorded properties are H264, 1600 × 900, 20 fps, 801 frames.
- `final-source-checks.txt`: Ruff lint and formatting pass for all 41 changed/new Python files;
  Git whitespace check passes.
- `final-runner-tests.txt`: earlier 14 runner/schedule tests passed; these overlap the final suite.
- Sibling learning `final-learner-tests.txt`: 19 learner tests passed.
- Sibling `reference-binding-tests.txt`, `reference-dual-binding-tests.txt`: reference contract
  NPZ and JSON binding regressions, 8 and 2 selected tests respectively (overlapping groups).
- Controller/cadence checks and exact replay evidence are recorded in the held-execution note;
  experimental outcomes are not encoded as a universal adaptive-must-win unit test.

The final source and artifact manifests bind the completed evidence and list exclusions rather
than rewriting earlier outputs. Concurrent deterministic timings are diagnostics only.

Recheck the full source/evidence inventory after moving or sharing the revision:

```bash
.pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_revision_manifest.py --verify \
  --artifact-roots artifacts/da_plcbf/navigation-revision-20260905 \
  artifacts/da_plcbf/learning-revision-20260905
```
