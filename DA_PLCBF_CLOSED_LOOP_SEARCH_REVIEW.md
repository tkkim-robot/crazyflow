# DA-PLCBF: a confirmed continuous collision-versus-survival case

This revision follows `0bcd4a17b03d0fc99f4bdcc024b866090072fa43`. It addresses the
closed-loop-search review and the user's proposed obstacle-buffer/HOCBF experiments.

The complete frozen controller now reaches a **modeled physical-sphere intersection** in a
continuous two-obstacle scene, while the continuously adapted controller completes the mission.
Both start at time zero with both obstacles already present and equally observable. The result
survives finer plant integration, wind-onset freezing, an early same-state parameter intervention,
nearby scene perturbations, and actual paced learning.

**Scope:** this establishes the mechanism for the matched **uncompensated fallback mapping**.
The stronger mapping with explicit model-based wind compensation remains collider-safe in this
scene and its tested neighborhood. No superiority over that comparator is claimed. Numerical
rigid-body geometry, measured MuJoCo contacts in a separate continuation, and hardware safety
remain different forms of evidence.

All new evidence is under `artifacts/da_plcbf/closed-loop-search-20260905`.

## What changed

- The continuous runner has an explicit simulation-only `termination_geometry="modeled_collider"`
  option. It records enclosure and requested-shell breaches, continues the same physical control
  path, and stops after the first definite rotated XML-sphere or floor intersection. The original
  enclosure mode remains the default for older callers; its censored outcomes are explicitly unknown.
- Discovery now executes the complete paired controllers and ranks actual collider outcomes.
  A negative frozen/positive adaptive initial hard-H gap is not a promotion prerequisite.
- World specifications support independent additional movers, static guards, initial velocity,
  and waypoint motion before wind onset. Predictions and collision checks share absolute time.
- Compiled functions and bound nominal-reference learners are reused. Discovery retains compact
  controls and dense plant states instead of rendering or serializing every learner checkpoint.
- Executed escape paths drive guard, second-mover, and local-parameter proposals. Each modified
  world is rerun from time zero. Old-path intersection is labeled only as a proposal diagnostic.
- Full-scene confirmation adds onset freezing, no wind, parameter reversion, complete QP audits,
  held-out local scenes, refined integration, and the existing measured publication scheduler.
- Branch video shell metadata now includes obstacle radius + robot enclosure + requested clearance.

No actor, reference-learning objective, update-acceptance rule, emergency rescue, actuator limit,
or controller safety threshold was weakened to obtain the result. Every finite learner update
continues to publish at the next permitted boundary.

## HOCBF and buffer findings

**Analytic obstacle HOCBF was already disabled in both compared navigation controllers.**
The enabled independent operational constraints cover arena/altitude, speed, angular rate,
tilt, and actuators. Their predictive refinements mask obstacles. The held-command obstacle
check is part of PLCBF execution acceptance; it is not a hidden additional obstacle HOCBF.

The physical XML sphere has radius **0.086 m**, with a body-fixed offset **0.020 m** above the
body origin. The controller uses a **0.106 m** body-origin enclosure, plus the requested extra
clearance. The original extra clearance is **0.150 m**, placing the shell approximately
0.15–0.19 m beyond the actual collider, depending on orientation and approach direction.

The new experiments test extra clearances **0.15, 0.05, 0.02, and 0.00 m**, identically in each
fixed/adaptive pair. The physical obstacles and robot geometry remain unchanged. Nominal
control and emergency braking remain model-compensated in both mappings; “uncompensated”
describes the fallback actor's explicit feedforward setting, not unequal model information.
Both methods receive the same current point dynamics model.

The initial U132 guard diagnostic demonstrates non-monotonic effects of buffer size: fixed
uses 22 degraded commands at 15 cm but none at the smaller tested buffers, while avoiding
physical collision in all four settings. Shrinking a shell can make more fallback maneuvers
admissible as well as reduce physical separation. The final selected case retains the desired
physical outcome at all four settings, including the original 15 cm.

See `CONSTRAINT_AUDIT.md` and the two buffer-sweep records for exact definitions and results.

## What was actually searched

| Full-episode discovery family | Uncompensated pairs | Compensated pairs | Physical outcomes |
|---|---:|---:|---|
| Existing single-sphere proposals, including rejected old H screens | 18 | 18 | 36 both separated |
| One mover plus two static guards | 12 | 11 | 23 both separated |
| Guard targeting U132's actual continuous lateral escape, four buffers | 4 | 4 | 8 both separated |
| Two independently prescribed staggered movers | 2 | 2 | 1 fixed-only collision; 3 both separated |
| **Total** | **36** | **35** | **71 distinct paired full episodes** |

There were no adaptive-only or both-method collision outcomes in these 71 discovery pairs.
All recorded operational nodes passed. One collider-separated adaptive run timed out; it is
retained as a task failure (`single-initial-v1-0026`, uncompensated) and was not promoted. The remaining 70 adaptive runs completed.
Frozen completed 70 and collided in one. “Both separated” is a geometry classification, not a
substitute for mission completion.

The initial family budgets were 128 pairs each. Search was paused and then stopped early after
the staggered scene passed numerical, causal, local, and paced confirmation. The retained
proposal files include unexecuted draws; these do not count as trials. The moving-state world
support and generator were tested, but **no full moving-state campaign was executed** before
the successful case was found. This is not an exhaustive negative search of the compensated case.

New cached hard-H geometry evaluations: **0**. The 12,288 historical cached rows were used as
proposal data only. The offline generator audit's 512 configurations and 1,024 mutations are
validation checks, not controller executions. A separate U132 smoke pair exactly reproduces
the old continuous states/actions/values/versions and is reported as a reproduction check.

Confirmation is counted separately: the first deterministic package has **36 paired comparisons**
using **70 newly executed method episodes** (the onset comparison reuses the original adaptive
trace); the targeted package adds **9 pairs**. Parameter interventions execute their actual
common prefixes from time zero; no obstacle-free state is substituted. Paced runs are separate.

## Selected complete scene

Selected trial: `staggered-initial-v1/staggered-initial-v1-0000`.

The robot begins at rest at `(0,0,1.4)` m. Wind switches on at **2.20 s** and persists with
velocity `(-1.657435, 1.125150, 0.498339)` m/s. Two bounded analytic sphere passages have
arrival times **4.510908 s** and **5.057568 s**, radii **0.679883 m** and **0.647222 m**, and
arrival speeds **1.384800 m/s** and **3.384875 m/s**. Full directions, offsets, amplitudes, and
all physical/controller settings are saved before execution. Navigation starts at **6.16 s**.

| Original deterministic comparison | Frozen | Continuously adapted |
|---|---:|---:|
| First modeled-sphere contact | **4.615011 s** | None |
| Minimum recorded XML-sphere clearance | **−0.017175 m** | **+0.172447 m** |
| Minimum requested-shell clearance | −0.187821 m | +0.005625 m |
| Degraded controls | 28 | 0 |
| Accepted executed QPs | 88 | 263 |
| Fallback / emergency controls | 22 / 6 | 0 / 0 |
| Waypoints completed | 0/2 | 2/2 |
| Stop/completion boundary | 4.64 s | 10.52 s |

Frozen therefore fails despite retaining its earlier response, short fallback execution, and
emergency brake. The collision is present at actual recorded plant nodes, not only in an
interpolated curve. Controls and mission credit stop at the first event's enclosing control
boundary; no post-contact mission success is counted.

## Causality and numerical confirmation

**Wind-onset freezing.** Both variants have identical observed calm histories through the
onset input, including commands, states, values, and version **715**. Freezing there produces
contact with −0.017110 m minimum recorded clearance; continued adaptation remains clear.
Thus the selected advantage is not explained by the preceding 55 calm updates alone.
With wind removed, both methods complete with approximately +0.1677/+0.1675 m clearance.

**Earlier deterministic parameter intervention.** At **3.00 s**, both branches share the same real full-scene
state, previous selection, goal, point model, and full learner state. Holding the available
learned version **735** (55 calm + 20 wind updates) completes with **+0.174312 m** clearance.
Reverting only the control parameters to the original version **660** collides, with
**−0.011145 m** minimum clearance. Both prefixes were actually executed from time zero.
Complete reference-bound Adam checkpoints and exact input hashes are retained.

At that common boundary, learned/original libraries have **12/9 eligible full QPs**, all accepted
under unchanged execution checks. Independent full-QP audit actions match the executed actions
exactly. This is a consequential case even though both libraries initially have executable
certificates; an exclusive initial certificate-gap screen would discard the comparison.

The first consequential applied-input difference from the onset-frozen comparator occurs at
**2.44 s**, with an active learned-policy QP constraint and six completed wind updates. At the
later **4.20 s** intervention, original-parameter reversion survives: earlier adaptation has
already changed the physical route. That negative necessity result remains published.

**Plant integration.** Prediction remains at 20 ms and commands remain held for 40 ms.
The plant alone is refined from 20 to 10 and 5 ms, giving 2/4/8 physical substeps per command.

| Plant step | Frozen recorded XML clearance | Adapted recorded XML clearance |
|---|---:|---:|
| 20 ms | −0.017175 m | +0.172447 m |
| 10 ms | −0.015374 m | +0.172368 m |
| 5 ms | −0.014414 m | +0.172737 m |

The 10→5 ms change is approximately 0.96 mm fixed and 0.37 mm adaptive. This is measured
integration sensitivity, not a rigorous continuous-plant error bound. Geometric curvature bounds
are separately calculated for the recorded-state interpolation. Definite penetration also
appears at 2/3/5 actual integration nodes on the three grids.

**Held-out neighborhood.** Twelve scenes were drawn before outcomes, with independent bounded
perturbations to obstacle positions/radii, arrival times/speeds, initial position/velocity, and
wind vector/onset. All are admissible; none were replaced or retuned. Every frozen run shows
definite modeled contact, while all adapted runs complete with no degraded commands and
**+0.171581 to +0.174178 m** minimum physical clearance.

The original 20 ms evaluation has **12/12 observed collisions but 11/12 satisfying the stricter
2 mm penetration target**. The narrow draw (`neighborhood_009`, −1.43 mm) was separately rerun
at 5 ms and still collides (−26.81 mm through its later first-contact hold); adaptive remains
clear (+171.61 mm). The original strict 11/12 count is unchanged. Penetration depths from
different terminating holds are not estimates of crash severity.

## Matched buffer sweep on the selected geometry

These are complete reruns with identical physical obstacles and matched settings in each pair.
Values are minimum recorded signed XML-sphere clearances in metres.

| Extra clearance | Uncompensated fixed | Uncompensated adapted | Compensated fixed | Compensated adapted |
|---|---:|---:|---:|---:|
| 0.15 m | −0.01717 | +0.17245 | +0.17069 | +0.17178 |
| 0.05 m | −0.03921 | +0.07338 | +0.07274 | +0.07248 |
| 0.02 m | −0.04314 | +0.04473 | +0.04231 | +0.04400 |
| 0.00 m | −0.05746 | +0.02683 | +0.02285 | +0.02472 |

The uncompensated physical outcome persists without an extra buffer. The compensated frozen
controller remains safe in all four settings and all twelve neighborhood scenes. It also has
zero degraded controls in the main deterministic case versus one for compensated adaptation.
The successful mechanism result must not be presented as a win over known-model compensation.

## Paced execution and startup diagnosis

The first paced replay reproduces uncompensated fixed contact and adaptive mission completion.
Adaptive completes **262 finite updates**, uses **110 before the first arrival**, retains
**+0.172671 m** physical clearance, and has **zero deadline misses**. Every published snapshot
follows measured learner completion. The fixed method has one **time-zero** deadline miss:
46.98 ms total interval service, of which 10.10 ms is controller computation. The compensated
paced pair both complete with zero deadline misses; its adaptive run completes 294 updates.

The new host recording/collider-audit path was not included in pre-epoch warmup. A targeted
repair now exercises and discards those host diagnostics before timing starts, alongside the
existing disposable controller/learner warmup. It also records separate pre-controller,
recording, plant, and collider-audit costs. Tests verify unchanged live initial state and learner
leaves, with no extra recorded controls or published updates. The first paced artifacts remain
intact; the startup cause is a hypothesis to check against the new measurements.

The final replay (`paced-staggered-0000-v2`) records **zero deadline misses in all four method
runs**. Uncompensated adaptation completes **250 finite updates**, uses **106 before first
arrival**, completes at **10.52 s**, and retains **+0.173427 m** physical clearance with no
degraded commands. Fixed still intersects at **4.615011 s**. The compensated pair both survive;
adaptation completes 292 updates and retains one degraded command.

The fixed first interval still spends **25.02 ms in host recording**, within a **37.28 ms total
service interval** (9.49 ms controller, 0.75 ms collider audit). Thus startup recording remains a
measured cost; the second run does not prove that warmup eliminated the source of the first
run's overrun. The first run is retained, and no hard-real-time guarantee is inferred. Different
actual update availability between paced runs is recorded rather than replaced with deterministic
versions or assumed nominal timing. In particular, the final uncompensated paced controller uses
version **730 at 3.00 s** (53 calm + 17 wind updates), while the deterministic causal intervention
uses **735** (55 + 20). The deterministic intervention is not presented as a paced checkpoint.
Independent audits verify every publication/control ordering and bitwise identical fixed states
and actions across deterministic, first paced, and final paced execution.

## Validation and review figures

The final consolidated CPU suite passes **135 tests in 68.88 s**, with three rendering-marked
tests excluded. It covers continuous termination, legacy censoring, collision precedence over
waypoint credit, unchanged warmup state, matched multi-obstacle worlds, causal updates and
parameter reversion, outcome classification, completed-trial identity, partial-attempt preservation,
contact replay splicing, and lossless publication/exclusion rules. The exact command and output
are retained under `validation`. GPU full-episode confirmations are reported above; no
full-repository test result is implied. Changed Python files pass Ruff and `git diff --check`.

Actual final clips are rendered and fully decoded separately. Rendering exposed an execution-mode
validation error for explicitly inactive contact-replay rows; the correction and regression are
included. Ordinary flight rows still require exact mode/flag agreement. Earlier overlapping test
runs remain recorded, with the consolidated result superseding their counts.

The compact figures show the collision versus survival mechanism, the earlier parameter
intervention, and the compensated control alongside every tested buffer:

- [Continuous and causal results](artifacts/da_plcbf/closed-loop-search-20260905/figures-v2/closed_loop_and_causal_results.png)
- [Matched buffer outcomes](artifacts/da_plcbf/closed-loop-search-20260905/figures-v2/matched_buffer_outcomes.png)

Their recorded input hashes and plotted data are retained beside PNG and PDF exports. Curves
show the encounter interval from complete episodes. Flight evidence ends at modeled contact;
the remainder of a terminating numerical hold is not a physical impact response.

## Local videos and measured contact continuation

The final local clips use the actual final paced recordings:

- `videos/paced-collision-v2/comparison.mp4`: fixed modeled contact and adapted completion.
- `videos/paced-compensated-v2/comparison.mp4`: the stronger compensated comparison, both complete.

Both use the same camera orientation, selected so the actual impact is visible. The previous
`*-v1` views hid the fixed drone behind the large sphere at impact and are superseded. The
controller recordings and physical geometry were unchanged for camera revisions.

The fixed flight's modeled contact at **4.615011 s** triggers a separate, explicitly labeled
MuJoCo motor-off continuation. MuJoCo measures obstacle contact at **4.616011 s** and ground
contact at **5.414011 s**. The original wind, prescribed absolute-time obstacle motion, and
recorded mass/inertia are retained. Flight telemetry before handoff and the entire adaptive trace
are preserved; no control, update, or waypoint credit is displayed for fixed after handoff.
This continuation demonstrates contact response, and does not extend the original controller's
mission or provide an additional safety trial.

The exact contact-physics source and historical wrapper are saved with their original hashes,
alongside contact poses, forces, model XML, and flight input hashes. Later changes to the wrapper
only affect presentation. Video verification and final stills are recorded under `videos/VIDEO_REVIEW.json`. Both final
1600×900 clips fully decode all **281 frames** at 20 fps (14.05 s), with no FFmpeg errors.
The collision, grounded body, adaptive completion, and compensated survival frames were visually
inspected. Videos display source simulation time; 20 fps playback selects the nearest recorded
control frame, with the source-clock quantization documented.
All MP4 files and extra inspection frames remain local. Eight representative stills are included
in the Git publication, alongside the complete video-review metadata.

## Evidence and reproduction

The exact completed deterministic and first paced sources are archived before subsequent
changes. Discovery retains source versions, immutable protocols, checkpoint IDs, complete
append-only trial ledgers, and compact controls/dense states. Interrupted unledgered attempts
and unexecuted proposals never count as completed episodes. Distinct scene identities exclude
seed-only relabeling. Videos and large rendering/prediction tensors remain local.

Use the existing `.pixi/envs/gpu-tests/bin/python` environment with `PYTHONPATH=.` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. All outputs must be fresh directories:

```bash
python benchmark/da_plcbf_closed_loop_search.py --family single --count 128 --seed 37103 --output NEW_SINGLE
python benchmark/da_plcbf_closed_loop_search.py --family guards --count 128 --seed 38201 --output NEW_GUARDS
python benchmark/da_plcbf_closed_loop_search.py --family staggered --count 128 --seed 39317 --output NEW_STAGGERED
python benchmark/da_plcbf_closed_loop_refinement.py --parents NEW_GUARDS/trials.jsonl --count 32 --seed 57301 --output NEW_REFINEMENT
python benchmark/da_plcbf_closed_loop_search.py --output NEW_REFINEMENT --resume
python benchmark/da_plcbf_closed_loop_confirmation.py --selected SELECTED_RESULT_JSON --output NEW_CONFIRMATION --mode all
```

The fixed scene is saved in the selected result; regeneration is not needed to replay it.
To repeat the nine targeted diagnostics, copy `targeted-confirmation-v1/run.py` and its
`protocol.json` into a fresh sibling directory, then execute that copy. Its relative inputs
refer to the retained original confirmation directory; do not execute it in the original output
folder. The refinement helper's `refinement-helper-smoke-v2` is a generator check with zero
executed episodes; its earlier `refinement-helper-smoke` output is superseded development data.
Numerical discovery, deterministic confirmation, and actual paced availability remain distinct
claims. No hardware or hard-real-time guarantee follows from this simulation case.
