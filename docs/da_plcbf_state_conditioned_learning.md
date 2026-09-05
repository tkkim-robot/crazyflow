# State-conditioned reference learning and bounded development ablations

This note covers the learner revision following `DA_PLCBF_REVIEW_AND_NEXT_STEPS_00e89a7.md`.
The new objective improves obstacle-free trajectory tracking in the recorded development bank.
That improvement alone does **not** establish retained runtime certificates or a navigation
advantage: the first seven variants lost oracle-model eligibility in an early encounter probe.
The later parameter-group extension is recorded separately from the already frozen navigation
campaign.

## Objective and continuation contract

[`state_conditioned_learning.py`](../crazyflow/safety/da_plcbf/state_conditioned_learning.py)
adds an immutable nominal-model teacher contract to the existing persistent AdamW learner.
For each current proprioceptive state, targets are the teacher's trajectory from that **same
initial state** under the saved **nominal** dynamics model. This replaces the old absolute
displacement target, which could ask an already moving vehicle to produce an incompatible
endpoint within the fixed horizon. The target is not a frozen actor rollout under the current
changed model; such a target would be satisfied by doing nothing to the parameters.

The actor observes local displacement from the rollout start, attitude, velocity, angular rate,
phase, and a fixed latent identity. No goal, obstacle, collision value, selected policy, or QP
result reaches its training interface. Position is used only through local displacement.
Proprioceptive samples from a mission can depend on its controller; this is not a claim that the
visited-state distribution is independent of the environment.

The development contract uses a fixed bank with rest, horizontal velocities of both signs
including `vx=±2.06 m/s`, signed vertical velocity, pitch, roll and body rate. It also retains the
13-component historical event state as a proprioceptive sample, without its obstacle metadata.
Each update trains on the current state and two rotating anchors. Anchor indices depend on the
persisted library version, so resume reproduces the next batch without a hidden random state.

The loss contains current-state 3D endpoint tracking and trajectory position/velocity tracking
at 25%, 50%, 75% and 100% of the horizon. Anchor trajectory errors provide retention across
states. Covariance and pairwise diversity remain spatial-only; nine-dimensional descriptor
arrays remain diagnostics for archive compatibility. Braking penalizes positive excess of
squared terminal speed relative to the same nominal teacher. Attitude, rate, behavioral effort,
action changes, motor saturation and movement from the preceding parameters are separate soft
terms. Physics feedforward is excluded from the behavioral-effort term.

The teacher defines a nominal behavior reference, not a proof that every target is attainable
under a changed plant. In particular, force cancellation cannot instantaneously remove the
attitude transient needed after a wind change. Terminal speed and actuator reserves are recorded
alongside tracking errors rather than inferred from the objective.

Prediction and BPTT use the same zero-order command hold as execution. The principal settings
are `K=16`, `H=60`, `dt=0.02 s`, and `control_interval_steps=2`: a 1.2 s prediction horizon with
40 ms feedback boundaries. Model compensation is enabled throughout teacher construction,
warmup and deployment in the new protocol; it is not switched on only at the event.

The numeric NPZ/JSON reference contract stores the teacher parameters, nominal model, complete
bank, actor configuration, reference settings, skill specification and actuator model with a
SHA-256. A sibling learner checkpoint stores all current/previous parameters, Adam moments and
counters, library version and point model. Both load with `allow_pickle=False`. The public
`build_reference_skill_learner_from_checkpoint` helper restores the checkpoint and its sibling
`nominal_reference` contract and checks their specification and actuator agreement. The focused
roundtrip test verifies equality of the next rotating-bank update and returned metrics.
Newly written reference checkpoints also bind `reference_contract_npz_sha256` and
`reference_contract_manifest_sha256` in metadata;
the deployment loader rejects a different numeric teacher/model/bank even when its skill spec
and actuator match. Older checkpoints remain readable with the explicit in-memory status
`reference_contract_binding="legacy_unbound"`. Existing archived runs are not rewritten; their
external manifests bind their original files. The two digests cover numeric reference inputs
and JSON configuration. Older single-digest bindings are explicitly labeled partial.

## Parameter restrictions remain optimizer rules

The optional velocity-offset bound projects offsets into a fixed interval. Frozen duration
offsets preserve their current values. The `network` and `offsets` training modes mask both
gradients and actual Adam updates, preventing stale moments or weight decay from moving a frozen
parameter group. Adam history is retained and advanced rather than reset. An optional global
parameter-update norm cap scales the Adam step before applying it. It is expressed in parameter
coordinates, not physical acceleration or trajectory distance.

Every finite completed update publishes its parameters and optimizer state. These restrictions
do not inspect collision values, goals, competence or navigation outcomes. The finite guard
continues to check the loss, gradients, proposed parameters and optimizer state. All new options
have backward-compatible defaults: hold one integration step, all parameter groups trainable,
durations trainable, and no offset or parameter-step bound. Existing checkpoint defaults remain
loadable.

## Fixed seven-variant development experiment

[`da_plcbf_reference_ablation.py`](../benchmark/da_plcbf_reference_ablation.py) records the fixed
comparison under `artifacts/da_plcbf/learning-revision-20260905/reference-ablation-seed{7,9,11}`.
Each seed constructs a nominal teacher with 400 updates from rest or modest forward velocity,
then runs 60 broader-bank reference warmup updates. All variants for that seed resume the same
nominal parameters and complete Adam history. The reference teacher is immutable during the
additional warmup and adaptation.

The variants are old current-state targets at LR multipliers 0.25/0.5/1, reference current plus
two anchors at the same multipliers, and reference current-only at 0.5. The base LR is `0.001`.
Reference variants freeze durations and bound velocity offsets to ±0.35 m/s. Each executes 80
updates at the same recorded moving event state and oracle wind `[4.0,1.6,0.0] m/s`. This is a
fixed-state behavior test, not a simulated flight or a time-varying estimator replay. Full raw
bank rollouts, motor forces, all update records, and checkpoints after 1/4/8/20/80 updates are
retained. All seven variants completed 80 finite updates in each of the three seeds.

For seed 7, the frozen changed-model bank has trajectory position RMSE `0.11620 m` and mean
terminal speed `0.22186 m/s`:

| Update rule | LR | Final bank RMSE (m) | Mean terminal speed (m/s) |
| --- | ---: | ---: | ---: |
| Old current-only | .00025 | .11401 | .24330 |
| Old current-only | .0005 | .11457 | .25912 |
| Old current-only | .001 | .11330 | .28462 |
| Reference + two anchors | .00025 | .10535 | .23071 |
| Reference + two anchors | .0005 | .10148 | .23396 |
| Reference + two anchors | .001 | .09834 | .23771 |
| Reference current-only | .0005 | .11289 | .22075 |

The predeclared behavior rule permits at most 0.05 m/s worse terminal-speed p95 and chooses
the lowest bank RMSE among its reference candidates. It chooses LR `.001` for all three seeds;
the selected RMSEs are `.09834`, `.09851` and `.09904 m`. This rule tolerates a braking tradeoff
and was **not** a safety-stability rule. The saved deployment candidate contains the shared
nominal checkpoint, never the wind-trained parameters. The later held-out world campaign uses
teacher seed 7; its world seeds are not additional learner seeds.

Rest competence and fast-state behavior remain different claims. The new seed-7 nominal
checkpoint occupies 16/16 direction bins from rest, with mean terminal speed about `.177 m/s`.
At the recorded fast event state it occupies 2/16 absolute direction bins with mean terminal
speed about `.228 m/s`. The broad bank does not make the from-rest direction criterion hold at
every moving state.

The actual-first-update gradient diagnostic uses the persisted bank index (version 460).
For reference learning, weighted trainable gradient norms are about `.541` for endpoint
tracking, `.221` for sampled trajectory positions, `.149` for retention, and `.0084` for the
braking-excess term. These norms and their directional cosines are diagnostics, not additive
fractions of the final Adam step. Earlier iteration-zero gradient files remain archived and are
explicitly superseded for this comparison by `initial_gradient_components_actual_bank.json`.

## Same-state coverage is evaluated separately

[`da_plcbf_reference_coverage_probe.py`](../benchmark/da_plcbf_reference_coverage_probe.py)
evaluates saved snapshots at the actual historical 4.0 s and 4.16 s states, using either that
recorded boundary's wind estimate or the oracle wind. All variants at a state/model pair use
the same nominal action, geometry and selection history. Exact nominal-action equality is
checked. Hard collision values, conservative smooth values, input validity, eligibility, QP
acceptance/reasons, complete rollouts and applied actions are archived. Geometry is introduced
only in this offline evaluator after training.

For the first seed-7 ablation, at the 4.0 s state with the oracle model, the new frozen teacher
starts with maximum fallback hard value `+.04069`, one eligible fallback and an accepted QP.
After four updates all seven variants lose eligibility and QP acceptance. Reference-bank
LR `.00025/.0005/.001` produce maximum fallback hard values `−.00214/−.04153/−.10434`.
The zero-wind estimate at that state remains optimistic: all these snapshots accept its QP.
At 4.16 s, the new frozen teacher already has no eligible fallback even under the recorded
estimate, despite a positive hard value, and its oracle hard value is negative. These are
counterfactuals from the new teacher trained at a fixed state; they are not the exact legacy
replay and do not represent newly executed navigation paths.

## Parameter-group stabilization extension

The separate `stabilization-seed7` artifact resumes the same initial checkpoint and Adam. Its
declared seven variants train only the residual network or only per-skill velocity offsets at
LR `.00025/.0005/.001`, plus all trainable parameters at LR `.001` with a `.002` global step
norm cap. Durations stay frozen, the offset bound is unchanged, and every finite update
publishes. Each retains all 80 updates, parameter-group change norms, and raw bank rollouts and
checkpoints after 1/4/8/20/80 updates. It does not change the frozen navigation campaign.

| Update rule | LR | Final bank RMSE (m) | Mean terminal speed (m/s) |
| --- | ---: | ---: | ---: |
| Network only | .00025 | .10679 | .23096 |
| Network only | .0005 | .10358 | .23366 |
| Network only | .001 | .10113 | .23652 |
| Velocity offsets only | .00025 | .11252 | .22448 |
| Velocity offsets only | .0005 | .10989 | .22682 |
| Velocity offsets only | .001 | .10630 | .22999 |
| All, step norm cap .002 | .001 | .11334 | .22553 |

All 560 updates were finite. The unrestricted LR `.001` first network step has norm `.03988`
and the first offset step `.00908`; the capped variant's largest measured float32 step norm
is `.002000011`. Tracking improves while mean terminal speed increases for every extension
variant. The cap controls parameter change; it does not establish a physical safety bound.

The separate `stabilization-coverage-seed7` probe contains 168 same-state/model cells. At the
4.0 s oracle state, network-only LR `.00025/.0005/.001` loses eligibility by update 4, with hard
values `−.00045/−.03860/−.10017`. Offset-only updates retain eligibility through update 8 at all
three rates; the two smaller rates retain it at update 20, and all lose it by update 80. The
`.002` cap retains eligibility and an accepted QP at updates 1/4/8/20, with hard value `+.01571`
at update 20, then loses both by update 80 (`−.04379`). Its bank RMSE at update 20 is `.11537 m`
versus frozen `.11620 m`, so early retention accompanies a small tracking improvement.

These matched parameter masks contradict the hypothesis that global velocity offsets alone
drive the first loss: shared-network changes produce almost the complete early degradation.
The cap is a tested way to slow that change, not a lasting safety solution. At the 4.16 s state
the initial new teacher is already ineligible; the extension does not manufacture a recovery
claim from that deficient initial condition. Closed-loop transfer is a separate experiment.

## Bounded closed-loop transfer check

[`da_plcbf_stabilization_closed_loop.py`](../benchmark/da_plcbf_stabilization_closed_loop.py)
executes one fresh common 0–4 s calm prefix in the original two-obstacle, straight-goal scene.
It uses the nominal candidate's exact parameters and Adam history, the corrected 40 ms control
hold and compensation enabled throughout. The state at 4 s is about `x=3.640 m, vx=2.011 m/s`;
it is a newly executed state, not the historical fixed-state probe location. The prefix estimator
observes the common transitions. Each estimated branch clones that history and then estimates
wind from its own applied commands and transitions; oracle branches receive the true point model.

From the exact common state and selector history, each information mode compares frozen,
unrestricted reference LR `.001`, and the `.002` step-capped reference learner for four more
seconds. Opportunities occur every second control. BPTT uses the boundary's pre-action state
and model, and each finite result publishes at the following boundary. The script stops a branch
at its first physical collision, if any, and never counts subsequent task progress. No branch
collided in this run; all reached the fixed 8 s observation end.

| Information / update rule | Minimum shell clearance (m) | Degraded controls | End goal distance (m) |
| --- | ---: | ---: | ---: |
| Oracle / frozen | +.00665 | 0 | .618 |
| Oracle / unrestricted | +.00726 | 0 | .296 |
| Oracle / capped | +.00676 | 0 | .643 |
| Estimated / frozen | −.04856 | 24 | .402 |
| Estimated / unrestricted | −.07967 | 39 | 2.125 |
| Estimated / capped | −.05128 | 24 | .402 |

All four learning branches published 50 finite updates, with identical scheduled opportunities.
Initial post-event actions match exactly across methods within each information mode. The
physical minimum clearances remained positive, and the recorded held operational postchecks
passed throughout. Estimated branches nevertheless breached the requested obstacle shell and
lost fallback hard coverage. Unrestricted learning worsened that outcome; the capped learner
stayed close to the frozen baseline. The cap did not solve the shared estimation-related shell
failure or establish an advantage over frozen control. Under oracle information, unrestricted
learning reduced the 8 s goal distance while all three stayed within the shell. These six short
branches are not a success-rate estimate or proof of sustained safety.

`stabilization-closed-loop` retains the complete shared prefix, 20 ms physical states, every
control's actual parameters, point estimate, command, full candidate rollouts, raw hard/smooth
values, QP fields and held checks. It also saves every completed Adam snapshot and estimator
state needed for reconstruction. Snapshots use the directory's common `nominal_reference`
contract; pass that explicit `contract_stem` when loading a checkpoint in a branch subdirectory.

## Estimator-only rate and observation-noise sensitivity

[`da_plcbf_estimator_replay_sensitivity.py`](../benchmark/da_plcbf_estimator_replay_sensitivity.py)
replays the same `estimated_frozen` dense trajectory and commands, including its shared calm
prefix, on CPU. It does not rerun the controller, plant or learner. Rates `1.2/2.4/6.0 s⁻¹` are
compared with velocity-observation noise standard deviation `0/.01 m/s`. One seeded (`20260905`)
iid Gaussian vector is drawn per 20 ms state sample and shared across rates; the same noisy
sample participates in both adjacent finite differences. Position, attitude, body rate and
actions remain exact. This is a narrow observation-noise model, not sensor-system validation.

Settling below `.1 m/s` means three consecutive samples with vector wind-error norm below the
threshold. The reported delay starts at the first of those samples; the artifact also records
the third-sample confirmation time. Late vector RMSE is measured on `[7,8] s`.

| Rate (s⁻¹) | Velocity noise std (m/s) | Settling delay after wind event (s) | Late vector RMSE (m/s) |
| ---: | ---: | ---: | ---: |
| 1.2 | 0 | 3.14 | .07277 |
| 2.4 | 0 | 1.58 | .001483 |
| 6.0 | 0 | .64 | .000001006 |
| 1.2 | .01 | Not reached by 8 s | .26103 |
| 2.4 | .01 | Not reached by 8 s | .21763 |
| 6.0 | .01 | Not reached by 8 s | .28014 |

The noiseless rate-2.4 replay matches recorded boundary wind estimates to maximum absolute error
`7.16e-7 m/s`. Every measurement in all six replays is finite. In the noisy runs, 26% of the
post-event inferred wind samples reach the existing ±5 m/s component clip; all three rates use
the same clipped samples. Faster filtering reduces the noiseless lag but does not monotonically
improve this noisy replay. None of these estimator-only numbers establishes better closed-loop
safety or BPTT behavior at a different response rate. All observations, estimated winds,
instantaneous clipped inferences and errors are retained under `estimator-replay-sensitivity`.

## Compensation and remaining adaptation demand

The archived compensation matrix includes the exact older checkpoint, old/new wind models,
compensation off/on, and both rest and the moving event state. Even with zero wind, changing
the compensation setting changes the behavior because it also cancels velocity-dependent
drag. At rest the older actor's mean terminal speed changes from about `.139` to `.218 m/s`;
at the event state it changes from `.133` to `.233 m/s`. This is why the new principal protocol
keeps the mapping consistent throughout warmup and execution.

The new teacher's compensated frozen actor already handles much of the plant change. On the
two-state matrix, a centered 25% payload produces only about `.00705 m` trajectory RMSE from
the nominal reference, compared with `.12039 m` for the wind change. Both have zero recorded
motor saturation and positive hover authority. These are model-aware baseline measurements;
they do not demonstrate that residual learning is necessary for the payload condition.

## Focused checks and reproduction

The focused learner tests cover same-state targets, changed-model non-vacuity, a gradient finite
difference away from actuator kinks, persistent checkpoint/contract resume with rotating bank
indices, frozen parameters with nonzero Adam history, offset projection, finite publication and
the parameter-step cap. The observed 19-test learner run is saved as `final-learner-tests.txt`.
After adding reference binding, all eight reference tests passed in `reference-binding-tests.txt`;
the final two targeted resume/binding regressions, including a JSON-only setting mutation,
passed in `reference-dual-binding-tests.txt`. These are identified runs, not cumulative
independent test counts. They validate implementation behavior, not navigation success.

```bash
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false pixi run -e gpu-tests python \
  benchmark/da_plcbf_reference_ablation.py --device gpu --seed 7 \
  --output-dir /tmp/reference-ablation-seed7

PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false pixi run -e gpu-tests python \
  benchmark/da_plcbf_update_stabilization.py --device gpu \
  --source-dir /tmp/reference-ablation-seed7 --output-dir /tmp/stabilization-seed7

PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false pixi run -e gpu-tests python \
  benchmark/da_plcbf_reference_coverage_probe.py --device gpu \
  --ablation-dir /tmp/stabilization-seed7 --output-dir /tmp/stabilization-coverage-seed7

PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false pixi run -e gpu-tests python \
  benchmark/da_plcbf_stabilization_closed_loop.py --device gpu \
  --checkpoint /tmp/reference-ablation-seed7/candidate/checkpoint \
  --output-dir /tmp/stabilization-closed-loop

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' PYTHONPATH=. pixi run -e gpu-tests python \
  benchmark/da_plcbf_estimator_replay_sensitivity.py \
  --closed-loop-dir /tmp/stabilization-closed-loop \
  --output-dir /tmp/estimator-replay-sensitivity

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' pixi run -e gpu-tests pytest -q \
  tests/unit/safety/da_plcbf/test_persistent_skill_learner.py \
  tests/unit/safety/da_plcbf/test_state_conditioned_learning.py
```
