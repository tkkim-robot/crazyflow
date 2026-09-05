# DA-PLCBF competent-library revision

This is the review entry point for the changes following the external review of
`cbaa4ba31bb164f5eedc3b404198140bd5b1f7cc`. The containing commit on `plcbf` includes the
implementation, tests, review notes, and complete indexed evidence for this revision. Prior
experiments remain in [DA_PLCBF_REVISION_REVIEW.md](DA_PLCBF_REVISION_REVIEW.md).

## Outcome and interpretation

The experiment now starts with a measured, useful repertoire, clones a complete optimizer and
physical checkpoint, exposes all execution modes, and admits learner results only after their
measured completion. It preserves the obstacle-agnostic persistent learner and keeps obstacles
and task goals outside its observations, rollout objective, and numerical update guard.

The stronger wind result is **partial coverage recovery, not evidence that learning is necessary
for successful navigation**. With identical point-model compensation, adaptive skills reach the
goal with positive shell clearance and no missed 40 ms deadlines, but have nine degraded
intervals. The compensated frozen library reaches the goal with no degraded intervals. At the
fixed neutral reference, adaptive direction occupancy improves from 11/16 immediately after
the wind change to 15/16 at 19.6 s; the compensated frozen library remains at 11/16. Adaptive
mean terminal speed there is 0.314 m/s, above the declared 0.30 m/s criterion, so this is not
full recovery of every nominal competence criterion. No threshold was relaxed after the run.

The unchanged-dynamics control succeeds for all three methods without degraded execution.
The previously unassisted adaptive branch fails the stronger wind case while both frozen
baselines succeed. That failure is retained, including its emergency actions and negative shell
clearance. It must not be relabeled as a successful adaptation demonstration.

The correctly warmed independent-estimator repeat also meets the compute budget but fails the
safety task: both frozen methods violate the requested shell and the adaptive method collides.
The payload case succeeds with compensation, while the crossing case remains collision-clear
but uses emergency commands. These are distinct outcomes, not one blanket acceptance result.

## Implemented changes

- [Numerical and execution details](docs/da_plcbf_numerics_revision.md): absolute-time obstacle
  derivatives in the smooth certificate, vacuous empty-obstacle handling, explicit softmin
  conservatism, checks through the whole held action, wind-aware emergency braking, execution
  attribution, and the exact-hover automatic-differentiation repair.
- [Learning and experiment protocol](docs/da_plcbf_learning_protocol.md): independent spatial
  descriptors, separate braking/attitude/rate costs, measured competence, complete checkpoints,
  prefix provenance, compensation controls, independent point estimators, and publication timing.
- [Matched runner](crazyflow/safety/da_plcbf/competent_library_experiment.py): three branches from
  one physical/parameter/Adam checkpoint, symmetric probes at every method's recorded state and
  model, common-reference full trajectories, dense plant states, feasible-route witnesses, and
  bounded wind/payload/crossing cases.
- [Replay renderer](crazyflow/safety/da_plcbf/mujoco_comparison_video.py): demo and diagnostic
  modes, approximately 88.5% scene height, stable skill colors, actual complete fallback paths,
  selected-endpoint rings, fixed-metric XY/XZ reference views, world wind tracers, persistent
  collision and margin status, and the actual `cf21B_500` visual model. Payload boxes use the
  recorded geometry and prescribed event time. Rendering never reruns the controller.

## Shared checkpoint and fair comparisons

The saved library uses K=16, seed 7, 500 finite obstacle-free point-model BPTT warmup updates,
integration step 0.02 s, and 60 prediction steps (1.2 s). Warmup is model training at rest and
occasional 0.5 m/s forward samples; it is not 500 updates performed during physical flight.
The actor is then frozen for the shared 4 s calm physical prefix. Branches copy the exact
13-state boundary, parameters, previous parameters, Adam history, and version 500.

The nominal rest reference occupies and aligns all 16 intended directions. Mean endpoint
pairwise distance is 0.461518 m; trajectory pairwise RMS distance is 0.263895 m; mean/p95 terminal
speed is 0.138593/0.247227 m/s; maximum tilt and angular rate are 0.171506 rad and 0.665866 rad/s.
All declared nominal competence checks pass. At the moving boundary, forward velocity is about
2.06 m/s and absolute direction occupancy is 2/16, despite distinct braking alternatives.
Rest competence is not a claim of omnidirectional motion from every moving state.

Every method has the same nominal goal controller, plant, motor limits, initial actor parameters,
geometry, and predeclared emergency function. At the branch boundary, `compensated` and the
default `adaptive` enable the **same** point-model force compensation outside the behavioral
acceleration bound; only adaptive continues BPTT. `fixed` retains uncompensated frozen skills.
Compare compensated against adaptive to isolate learning. Before the boundary, all saved
common-reference trajectories use the same original actor, including the compensated panel.
The explicit `adaptive_model_compensation=false` option retains the earlier unassisted ablation.

Checkpoints contain numeric NPZ arrays and a JSON structure, shape/dtype checks, and SHA-256.
The saved actual pre-event scenario and controller arguments are compared with the effective
new run; the prefix bytes are hashed and their final state must exactly match the checkpoint.
Changing pre-event geometry, physical discretization, skill specification, or controller factory
requires a new prefix. A crossing scene therefore gets its own matched checkpoint. Older
no-provenance checkpoints remain readable as artifacts but cannot seed new experiments.

## Runtime and safety scope

Actions are held for two 0.02 s plant substeps: 25 Hz control with the original 1.2 s prediction
horizon. Each branch runs separately on one RTX 4090 (driver 555.42.06), avoiding renderer/test
contention. The controller runs first. One synchronized BPTT call may use measured remaining
slack, with a rolling p95 estimate, 1.25 multiplier, and 3 ms reserve. Only a complete result
publishes at a later actual control boundary. The terminal checkpoint publication also waits
for the real terminal boundary; its version can exceed the last controller-used version by one.

Service logs include scheduled/start/completion wall times, controller/learner durations,
snapshot versions, publication timestamps, snapshot age, and every missed deadline. Probes,
plots, and feasibility witnesses run outside timed branches. The schedule assumes serialized
GPU execution and makes no overlap claim. A miss remains a failed budget check.

The simulation applies its chosen command at a sampled boundary and does not model sensor or
actuator transport latency. Passing measured deadlines is not an operating-system hard real-time
guarantee or a hardware-safety result. Collision certificates remain finite-horizon predictions
under a point model, not proofs across arbitrary library/model changes or perception updates.
Held operational constraints are checked through the action interval; full future rollouts are
not certificates of every operational limit. Swept checks use linear interpolation between
integration nodes. Positive physical/shell clearance and accepted-QP execution remain separate.

## Retained development trials

All paths in this section are under `artifacts/da_plcbf/competent-revision-20260904/`.

| Directory | Status and interpretation |
|---|---|
| `shared-checkpoint-0` | Initial competent checkpoint; predates recorded effective prefix provenance. |
| `wind-oracle-0` | Failed timing pilot: a plant-originated input compiled on the second control call; substantial deadline misses and no adaptive updates. |
| `wind-oracle-1` | Wind [2, 0.8, 0] pilot: all methods safe, zero deadline misses, 398 completed updates. Predates terminal publication waiting, nonzero gradient telemetry, and corrected neutral-reference validity/protocol metadata; not final evidence for those features. |
| `wind-oracle-strong-2` | Unassisted wind [4, 1.6, 0] failure: 381 emergency intervals, 400 negative-H intervals, minimum shell clearance -0.107874 m, final goal error 4.11245 m, no physical collision. Frozen and compensated methods succeed. The initial supplied lower detour also collides; its full witness record remains. |
| `wind-estimated-5` | Failed timing pilot: estimator-generated JAX placements triggered new controller/learner compilations; its completion audit also predates explicit estimator synchronization. Replaced by `wind-estimated-8` for measured-estimator results, without deleting the pilot. |

The last trial motivated matching known-model compensation with the strong frozen baseline.
It does not justify attributing the resulting improvement entirely to learning. The later upper
detour is explicitly a revised reference route; the failed lower route is not overwritten.

## Final numerical evidence

These directories are tracked under `artifacts/da_plcbf/competent-revision-20260904/`.
All runs use 20 s of simulated motion, a branch at 4 s, and 500 control intervals in total.
All **final** branches have zero measured post-event deadline misses and zero midpoint commands.
`Q/F/E` counts below mean accepted QP / direct fallback / emergency; degraded intervals are
reported separately because an executable fallback can lack a full-horizon certificate.

| Case / directory | Method | Minimum shell clearance (m) | Goal error (m) | Q/F/E | Degraded |
|---|---|---:|---:|---|---:|
| Wind [4,1.6,0], oracle / `wind-oracle-compensated-3` | Fixed | 0.006741 | 0.000296 | 497/3/0 | 2 |
| | Compensated | 0.006850 | 0.000298 | 500/0/0 | 0 |
| | Adaptive | 0.005863 | 0.000295 | 479/21/0 | 9 |
| Unchanged / `unchanged-control-4` | Fixed | 0.006280 | 0.000116 | 494/6/0 | 0 |
| | Compensated | 0.006284 | 0.000152 | 495/5/0 | 0 |
| | Adaptive | 0.006019 | 0.000384 | 491/9/0 | 0 |
| Centered +25% mass / `payload-oracle-6` | Fixed | 0.009343 | 4.901452 | 500/0/0 | 0 |
| | Compensated | 0.009415 | 0.000026 | 500/0/0 | 0 |
| | Adaptive | 0.012694 | 0.000017 | 500/0/0 | 0 |
| Crossing sphere / `crossing-oracle-7` | Fixed | 0.360316 | 0.000017 | 491/2/7 | 7 |
| | Compensated | 0.355168 | 0.000016 | 492/3/5 | 5 |
| | Adaptive | 0.354754 | 0.000016 | 492/3/5 | 5 |
| Wind [4,1.6,0], independent estimates / `wind-estimated-8` | Fixed | -0.078856 | 0.000294 | 475/12/13 | 25 |
| | Compensated | -0.080066 | 0.000293 | 472/14/14 | 26 |
| | Adaptive | -0.159144 | collision | 104/13/383 | 396 |

Each final directory contains `competent_comparison.npz/json`, `dense_plant_states.npz`,
`symmetric_probe_trajectories.npz`, `final_adaptive_checkpoint.npz/json`,
`feasibility_reference.npz/json`, `comparison.png`, and a separate `execution_audit.json`.
The audit independently checks physical states with NumPy and records hashes of its inputs;
it does not rewrite the original summaries. The original summary field `safe_goal_success`
means positive collision-shell clearance plus goal error below 0.5 m. The audit calls this
`collision_clear_goal` and keeps physical limits, degraded execution, and timing separate.

All final saved physical nodes satisfy the configured speed, angular-rate, tilt, and arena
limits. This includes the crossing case, whose negative operational derivative residuals are
not physical node violations. Crossing adaptive/compensated each execute two emergencies before
the shared boundary and three afterward; fixed executes two before and five afterward. The first
emergency is at 3.4 s; first negative recorded analytic residuals occur at 6.2 s for adaptive and
compensated, and 6.0 s for fixed. The saved residual array describes the applied command at the
initial boundary, not every internal held substep; the core additionally checks held substeps.

The bounded `crossing-oracle-7/alternative_policy_probe.json` reproduces the compensated
emergency at 6.2 s and forces each of the 17 originally eligible certificates through unchanged
QP and execution checks. All yield the same initially accepted QP wrench, then fail the held
operational derivative residual (-0.047247); the held physical margin and collision margin remain
positive. Every direct candidate fallback also fails that held derivative check. Choosing another
eligible certificate therefore does not repair this sampled rejection. This is a diagnosis at
one recorded state, not proof that no physically safe action exists. The diagnostic script is
retained beside its JSON; no alternative-policy search was added to runtime based on this result.

In the final estimated case, adaptive first loses hard-H coverage at 4.16 s, first uses emergency
control at 4.68 s, first violates the shell in the interval beginning at 4.70 s, and first
collides in the interval beginning at 4.94 s. Minimum physical clearance is -0.009144 m.
Post-contact numerical continuation is retained for debugging; it is not physically meaningful
flight. Demo replay freezes the collided panel and preserves the failure status. State estimates
use each method's own transitions and response rate 2.4/s; telemetry itself is noiseless. This
result exposes sensitivity to estimator lag and online updates, not a robustness guarantee.

The supplied upper detour, using the same plant/motor bounds and nominal acceleration limit,
passes its physical and goal checks for oracle wind, unchanged dynamics, payload, and the
estimated case's **oracle reference**. It is never used for learning or filter decisions.
The supplied crossing detour fails its checks and remains saved. The collision-clear actual
crossing trajectories are separate evidence; a failed supplied route does not prove impossibility.

### Measured service and verified solver branches

Each row below describes the adaptive branch's 400 post-event controller calls. Learner counts
vary with actual available wall-clock slack; exact counts are not a portable deterministic
reproduction promise.

| Case | Controller median / p95 (ms) | Learner median / p95 (ms) | Complete finite updates |
|---|---|---|---:|
| Oracle wind | 15.871 / 18.021 | 11.388 / 11.531 | 393 |
| Unchanged | 15.824 / 17.187 | 11.385 / 11.492 | 395 |
| Payload | 15.920 / 19.376 | 11.343 / 11.475 | 386 |
| Crossing | 15.874 / 20.086 | 11.394 / 11.531 | 369 |
| Estimated wind | 15.911 / 19.426 | 11.452 / 11.572 | 371 |

`execution-branches-gpu.json` verifies actual branch predicates before reporting timings, with
K=16, H=60, two held substeps, 30 synchronized samples per branch, and complete controller outputs.
The natural full-QP fixture uses 3 m/s forward speed, 0.3 rad pitch, and -2 rad/s pitch rate;
the fast shortcut was not disabled to manufacture a full-solver label.

| Verified branch | Median / p95 / maximum (ms) |
|---|---|
| Accepted fast QP | 13.872 / 14.648 / 14.694 |
| Accepted full active-set QP | 15.450 / 15.593 / 15.618 |
| Certified direct fallback | 13.955 / 14.072 / 14.083 |
| Emergency | 14.000 / 14.088 / 14.107 |

These are isolated computational probes, not worst-case execution-time bounds. Their timing
excludes the plant, estimator, learner, and host telemetry; the paced episodes separately include
that service. Compilation and rendering are excluded from both reported steady-state comparisons.

## Checks and reproduction

Observed focused checks are listed separately to preserve their scope:

- Final protocol, scheduler, and legacy online tests: **21 passed, 1 skipped in 93.82 s**.
  The skip explicitly requires CUDA; new oracle and estimated CPU protocol runs both passed.
  Log: `artifacts/da_plcbf/competent-revision-20260904/final-protocol-tests.txt`.
- Final renderer checks: **13 passed in 3.52 s**, including actual MuJoCo frames, payload geometry
  and serialization, recorded fallback paths, persistent violations, collision-freeze time,
  and the open goal ring that keeps the arriving drone visible.
- Latest core isolation checks: **11 passed in 43.65 s**, covering temporal finite differences,
  empty/masked geometry, softmin duplication/resolution, invalid future motion, emergency/held
  execution, and exact-hover gradients.
- Learner/checkpoint/payload/reference checks: **17 passed**. The full optimizer roundtrip
  preserves the next update exactly; descriptor diversity cannot reward redundant velocity
  coordinates or nonzero terminal velocity.
- An earlier core regression run passed 45 checks; its precise sequencing is documented in
  [the numerical note](docs/da_plcbf_numerics_revision.md). It is not presented as a full suite
  rerun after every final edit. Ruff, formatting, and `git diff --check` pass for the final changes.

Run from the repository root, using a fresh output directory. Avoid competing numerical or
rendering jobs while collecting measured service. The persisted wall-clock schedule means update
counts and resulting learned trajectories can vary across machines or system load.

```bash
export PYTHONPATH=.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/tmp/crazyflow-competent-jax-cache
DA_PLCBF_REPRO=/tmp/da-plcbf-competent-reproduction

.pixi/envs/gpu-tests/bin/python examples/da_plcbf/competent_library_demo.py prepare \
  --output-dir "$DA_PLCBF_REPRO/checkpoint"

.pixi/envs/gpu-tests/bin/python examples/da_plcbf/competent_library_demo.py run \
  --config artifacts/da_plcbf/competent-revision-20260904/wind-oracle-compensated-3/config.json \
  --checkpoint "$DA_PLCBF_REPRO/checkpoint/competent_checkpoint" \
  --output-dir "$DA_PLCBF_REPRO/wind" --no-render

.pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_artifact_audit.py \
  "$DA_PLCBF_REPRO/wind"

.pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_execution_branches.py \
  --device gpu --checkpoint "$DA_PLCBF_REPRO/checkpoint/competent_checkpoint" \
  --control-interval-steps 2 --samples 30 --output "$DA_PLCBF_REPRO/branches.json"
```

Repeat `run` with each final directory's saved `config.json`. Static wind, unchanged, estimated
wind, and centered payload can share the regenerated checkpoint. For crossing, omit `--checkpoint`
to prepare the correct geometry-specific physical prefix. Never reuse the static scene prefix
for moving geometry.

Replay is independent of numerical execution:

```bash
MUJOCO_GL=egl JAX_PLATFORMS=cpu .pixi/envs/gpu-tests/bin/python \
  examples/da_plcbf/online_constant_wind_demo.py render \
  --trace "$DA_PLCBF_REPRO/wind/competent_comparison.npz" \
  --summary "$DA_PLCBF_REPRO/wind/competent_comparison.json" \
  --video "$DA_PLCBF_REPRO/wind/compensated_vs_adaptive_demo.mp4" \
  --left-method compensated --right-method adaptive --mode demo \
  --width 1600 --height 900 --fps 20 --probe-pause-time 12 --probe-pause-seconds 2
```

Use `--mode diagnostic` for the detailed telemetry layout. The demonstration files below compare
the compensated frozen library with compensated adaptive skills; none is titled as proof of
adaptive necessity. The wind probe pause holds one recorded common-state observation, without
inventing intermediate optimization or physical motion.

| Replay | File under the artifact root |
|---|---|
| Oracle wind: partial repertoire recovery, both reach goal | [Wind demo](artifacts/da_plcbf/competent-revision-20260904/wind-oracle-compensated-3/compensated_vs_adaptive_demo.mp4) |
| Centered payload: supplied mass/inertia switch | [Payload demo](artifacts/da_plcbf/competent-revision-20260904/payload-oracle-6/compensated_vs_adaptive_demo.mp4) |
| Moving crossing obstacle: emergency intervals remain visible | [Crossing demo](artifacts/da_plcbf/competent-revision-20260904/crossing-oracle-7/compensated_vs_adaptive_demo.mp4) |
| Independent estimates: retained collision failure | [Failure replay](artifacts/da_plcbf/competent-revision-20260904/wind-estimated-8/compensated_vs_adaptive_demo.mp4) |

`SHA256SUMS` under the artifact root indexes the numerical evidence, audit outputs, and rendered
media. The complete indexed directory is explicitly tracked despite the repository's default
artifact ignore rule, including retained failures and superseded goal-sphere renders.

`source_snapshot.tar.gz`, `SOURCE_SHA256SUMS`, and `CODE_PROVENANCE.json` preserve the immutable
pre-publication source capture. Verify that source manifest against the extracted snapshot;
its two review/handoff documents retain the earlier local-only publication wording. The current
commit updates that wording, while implementation, tests, numerical data, and videos match the
reviewed capture. Git identifies the published source revision; the archived provenance's base
commit and local-worktree description identify the earlier capture, not its publication status.

## Limits and next research step

Centered payload attachment changes mass and centered box inertia consistently in the plant and
all known point models. Rotor moment arms and center of mass stay fixed; the 5 cm box lies inside
the existing 5 cm spherical collision enclosure. This is a prescribed parameter switch, not
contact-resolved pickup or momentum exchange. The wind estimator is not used as a mass estimator.
The crossing obstacle follows one prescribed constant-velocity trajectory from time zero; 4 s
is the shared branch boundary, not the onset of obstacle motion. Its absolute-time derivative
enters the runtime certificate. Off-center and tethered loads remain deferred new dynamics work.

The unresolved scientific milestone is sustained useful behavior recovery that improves safety
over the competent compensated frozen baseline without degraded intervals under the measured
budget. Current results do not establish it. The next work should examine moving-state training
targets, terminal braking, and early update-induced certificate loss, using the retained common
state probes. Increasing disturbance strength or weakening the baseline is not evidence of that
milestone. Any future richer model needs independently validated whole-system dynamics and
collision geometry before making stronger safety claims.
