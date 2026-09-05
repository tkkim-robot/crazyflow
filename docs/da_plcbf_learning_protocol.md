# DA-PLCBF learning and matched-checkpoint protocol

This note describes the learning and experimental protocol introduced after the `cbaa4ba` review.
The current comparison starts from one measured nominal-dynamics repertoire and one shared
physical prefix. It separates changes in learned parameters from changes in model compensation.
Episode outcomes and timing measurements belong in the accompanying evidence report; this
protocol does not establish that adaptation improves every encounter.

The implementation is in
[`persistent_skill_learner.py`](../crazyflow/safety/da_plcbf/persistent_skill_learner.py),
[`learner_checkpoint.py`](../crazyflow/safety/da_plcbf/learner_checkpoint.py),
[`competent_library_experiment.py`](../crazyflow/safety/da_plcbf/competent_library_experiment.py), and
[`deadline_schedule.py`](../crazyflow/safety/da_plcbf/deadline_schedule.py). The separate
[numerical revision note](da_plcbf_numerics_revision.md) describes collision values, QP acceptance,
held-action checks, and emergency execution.

## Independent spatial objective and retained diagnostics

The learner still returns nine diagnostic coordinates per skill: final displacement, mean
velocity, and terminal velocity, each in three dimensions. This preserves saved-trace and
renderer compatibility. It does not mean the optimizer maximizes nine-dimensional diversity.

For the symplectic rollout, displacement equals the sum of the post-step velocities times `dt`.
Mean velocity uses those post-step samples, and its target is displacement divided by the full
prediction duration. These coordinates therefore duplicate displacement information. Terminal
velocity measures braking quality and should not be increased to manufacture apparent diversity.

The target, covariance, and pairwise losses now use **only the three normalized displacement
coordinates**. The target term is mean squared displacement error. The covariance term is
`-logdet(covariance + epsilon * I_3)`. The pairwise term penalizes proximity between distinct
displacement endpoints using a Gaussian kernel. The implementation accepts either three-column
spatial arrays or nine-column diagnostic arrays and slices the latter explicitly.

Separate losses penalize terminal velocity, body tilt through the smooth quantity `1 - R_zz`,
and body angular rate. Desired-acceleration magnitude, acceleration changes, motor saturation,
and parameter movement from the previous persistent iterate retain separate regularization
terms. Braking and attitude/rate penalties are soft objectives, not safety certificates or policy
publication criteria. Exact weights and normalization scales are saved in `PersistentSkillConfig`.

The fallback actor receives local displacement, quaternion, linear/angular velocity, phase, and
a fixed latent identity. Skill directions and duration/speed targets are constructed independently
of obstacles and the navigation goal. Optional physics feedforward uses the available point
model. The learner does not receive obstacle positions, the task goal, collision values, selected
policies, or QP outcomes. Online training states are proprioceptive samples from the actual
mission; this does not constitute a claim that the state distribution is independent of the
controller's behavior.

## What measured competence means

The competent-checkpoint runner defaults to 16 skills, a 60-step horizon at `dt=0.02 s`, a maximum
target speed of `0.9 m/s`, and maximum target duration of `0.7 s`. It uses a structured directional
initialization (`initial_skill_scale=1`) followed by 500 obstacle-free nominal-model updates.
Four out of five warmup samples start at rest; the fifth has forward velocity `0.5 m/s`.
This is a shared warmup experiment, not evidence of learning a repertoire from an unstructured
initialization during flight.

`skill_library_competency` measures actual rollouts. Its recorded diagnostics include occupied
direction bins, alignment with each skill's target direction, endpoint pairwise spread,
synchronized trajectory spread, terminal speed, tilt, and angular rate. Thresholds are explicit
and saved with the measurements. Competency reporting never accepts or rejects an optimizer step.

Competence is state-dependent. The retained nominal checkpoint measured 16/16 occupied and
aligned directions from rest, mean endpoint separation about `0.462 m`, and mean terminal speed
about `0.139 m/s`. At its moving event state, forward velocity was about `2.06 m/s`: absolute
endpoint directions occupied only 2/16 bins, while endpoint spread remained about `0.329 m` and
mean terminal speed about `0.133 m/s`. Both measurements are retained. The rest criterion passes;
the same all-direction criterion at the moving state does not. The latter supports a description
of diverse braking alternatives at that state, not full directional competence everywhere.

## Complete continuation state and physical-prefix provenance

The paired NPZ/JSON checkpoint stores current parameters, previous parameters used by the trust
term, the complete AdamW optimizer state including moments and counters, cumulative gradient
steps, library version, and the latest point dynamics model. It also stores the skill specification,
learner configuration, actuator parameters, and the 13-component physical state. Loading rebuilds
the optimizer structure and restores its history rather than initializing new moments.

NPZ arrays are numeric and loaded with `allow_pickle=False`. JSON records the structure,
shape/dtype manifest, NPZ SHA-256, and JAX/Flax/Optax versions. Loading checks the manifest and
rejects corruption or an inability to restore the exact dtype. Existing checkpoints are not
overwritten. The focused continuation regression checks exact equality of the next update and
its metrics after a save/load roundtrip in the tested environment.

The experiment checkpoint additionally records the actual pre-event scenario fields, prefix
configuration, controller arguments, and controller-factory source fingerprint. Its separate
`shared_prefix.npz` has a SHA-256. Loading verifies that hash, the physical/control sample counts,
the recorded control states, and exact agreement between the prefix's final physical state and
the learner checkpoint. This prevents a changed scenario default from silently reinterpreting an
older recorded prefix.

Changing the pre-event geometry, holding interval, skill specification settings, learner settings,
or prefix controller is rejected. Post-event wind, oracle versus estimated wind information,
episode duration, runtime schedule, and the adaptive compensation option may change without
changing that prefix. Older checkpoints without effective prefix provenance remain loadable as
historical data; a new competent experiment requires regenerating their shared prefix. A final
adaptive checkpoint is a complete learner continuation state, but is not automatically a new
competent-prefix checkpoint.

## Matched branches and compensation intervention

One calm physical prefix is executed using the warmed, frozen library. At the event, all branches
clone its physical state, parameters, optimizer history, skill specification, and selector history.
The prefix is recorded once and reused. The two compensated branches do not retroactively execute
a different pre-event controller, and pre-event counterfactual probes use the shared prefix actor.

The current post-event methods are:

| Method | Learned parameters after the event | Fallback model feedforward |
| --- | --- | --- |
| `fixed` | Frozen checkpoint | Disabled |
| `compensated` | Frozen checkpoint | Enabled |
| `adaptive` | Persistent BPTT updates when service time is available | Enabled by default |

The navigation nominal uses the same model-compensation rule and limits in all three branches.
The compensated fallback and default adaptive fallback also use the same feedforward formula
and physical motor bounds. Known force compensation is added outside the behavioral acceleration
saturation. This is a controller intervention, not a learned improvement. Setting
`adaptive_model_compensation=false` retains the unassisted adaptive ablation. Its value and the
post-event compensation assignment are written into the experiment summary.

With `model_mode="oracle"`, every branch receives the changed point model immediately. With
`model_mode="estimated"`, each branch owns a fresh wind estimator and updates it from its own
state transitions and applied wrench. Plant wind is not supplied to that estimator. Each
controller decision and learner call use one frozen point estimate; there is no ensemble or
robust uncertainty set. The centered-payload experiment supplies known mass/inertia and permits
only oracle mode because wind estimation is not mass identification.

An adaptive advantage over `fixed` can combine feedforward and learning. **Only the comparison
against the equally compensated frozen branch isolates the additional effect of parameter
learning under the matched compensation setup.** Success after enabling feedforward must not be
presented as evidence that BPTT caused that success.

## Persistent updates, measured availability, and probes

Every numerically finite completed update becomes the next persistent learner version, including
its optimizer history. The numerical guard checks loss metrics, gradients, proposed parameters,
and optimizer state; invalid model/rollout computations surface as invalid loss values. There is
no collision test, goal-progress test, competence threshold, or library-quality acceptance gate
on updates. Runtime policy selection and execution checks are separate from learning publication.

The budgeted runner paces control releases against a monotonic clock. With two held integration
steps, the action period is `0.04 s` while the prediction horizon remains `1.2 s`. It synchronizes
complete controller results, performs the plant/telemetry work, and starts at most one serialized
learner update when its measured service estimate plus reserve fits before the next deadline.
Whole completed snapshots become available only at a subsequent boundary. Gradient/update
telemetry belongs to that published snapshot, and the summary distinguishes the last completed
library version from the last version used by a controller.

Deadline misses remain recorded failures. This measures sampled-simulation service feasibility;
sensor/actuator transport delays and operating-system real-time guarantees are excluded. The
numerical plant advances fixed integration steps even if a wall-clock deadline is missed, so an
overrun episode does not establish safety under the corresponding real elapsed action hold.

Symmetric probes run after the timed episodes and cannot alter learning or execution. At each
recorded time, every already-published library is evaluated at each method's measured state with
that anchor's identical point model and obstacle prediction. Probe artifacts retain states,
rollouts, hard collision values, and validity-aware safe counts. Fixed neutral-state probes
separately show changes in repertoire geometry. They do not replace encounter-state safety
measurements, and a counterfactual library deficit does not imply that the actual frozen-method
trajectory collided.

## Retained failure and observed checks

The retained `wind-oracle-strong-2` pilot used wind `[4.0, 1.6, 0.0] m/s` with the unassisted
adaptive actor. It completed 393 finite updates with zero deadline misses, but violated the
requested inflated obstacle shell and ended about `4.11 m` from the goal. Its minimum physical
clearance remained positive (`0.0421 m`); this was a shell/progress failure, not a physical collision.
Both frozen branches retained positive shell clearance and reached the goal. This result remains
under `artifacts/da_plcbf/competent-revision-20260904/wind-oracle-strong-2`; it must not be replaced
by, or relabeled as, a compensated success. That historical configuration predates the new
adaptive feedforward default.

Two separately observed focused CPU test runs are recorded here:

- **17 learner/payload/reference tests passed** across `test_persistent_skill_learner.py`,
  `test_rigid_payload.py`, and `test_feasibility_reference.py`. These cover independent spatial
  losses, persistent optimizer continuation, checkpoint integrity, centered-load consistency,
  and an actual-actuator known-model route witness.
- **19 protocol tests passed in 92.12 s** across `test_competent_library_experiment.py` and
  `test_deadline_schedule.py`. They include tiny oracle and estimated end-to-end experiments,
  checkpoint compatibility/provenance, prefix corruption and boundary-state mismatches,
  independent estimator resets, published gradient/version telemetry, and saved symmetric-probe
  roundtrips. The tiny runs use `K=4`, horizon `10`, two warmup updates, event `0.04 s`, duration
  `0.16 s`, and unlimited scheduling; they validate the execution path, not mission performance
  or measured 25 Hz feasibility. Subsequent production timing evidence is reported separately.

These are identified test runs, not an aggregate count or a claim that every suite was rerun
after each later reporting change. CPU-only reproduction from the repository root is:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' pixi run -e gpu-tests pytest -q \
  tests/unit/safety/da_plcbf/test_persistent_skill_learner.py \
  tests/unit/safety/da_plcbf/test_rigid_payload.py \
  tests/unit/safety/da_plcbf/test_feasibility_reference.py

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' pixi run -e gpu-tests pytest -q \
  tests/unit/safety/da_plcbf/test_competent_library_experiment.py \
  tests/unit/safety/da_plcbf/test_deadline_schedule.py
```
