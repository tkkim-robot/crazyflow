# Hover-first behavior explanation and MuJoCo contact revision

This is the current review entry after the five visual/functionality comments, published on
`main` after base `00e89a742a1271b93655bf1bb4581a667dc13a14` from `plcbf`. The Git commit identifies
the published revision; archived source snapshots retain their original prepublication metadata.
New evidence is under `artifacts/da_plcbf/hover-explanation-20260905`. Older artifact directories
and their manifests are retained unchanged. The preceding navigation campaign is documented in
[DA_PLCBF_NAVIGATION_REVIEW.md](DA_PLCBF_NAVIGATION_REVIEW.md), with a new collider-scope correction.

## Videos to review

- [Hover, wind learning, recovery, centered payload, then navigation](artifacts/da_plcbf/hover-explanation-20260905/hover-wind-payload-navigation/navigation_comparison_demo.mp4): 63.05 s, 1600×900, 20 fps; ordinary-speed simulation time.
- [Compensated fallback control](artifacts/da_plcbf/hover-explanation-20260905/compensated-control-v1/navigation_comparison_demo.mp4): 19.10 s; explains the earlier similar fans. The source includes one 40 ms navigation command at release before its 19.04 s timeout; this is not a navigation trial.
- [Recorded failed encounter, actual MuJoCo impact and ground contact](artifacts/da_plcbf/hover-explanation-20260905/contact/legacy-estimated-physical-contact-v2/render-v2/contact_replay_demo.mp4): 7.55 s at explicitly labeled 0.5× playback, with source time shown.
- [Safety-abort motor-cut drop](artifacts/da_plcbf/hover-explanation-20260905/contact/seed209-adaptive-safety-abort-drop/render-v1/contact_replay_demo.mp4): 9.05 s at 0.5×; measured ground contact with no obstacle impact.

The first contact camera preview hid the impact behind the obstacle. `render-v2` changes the
camera angle while retaining the same solid geometry and exact saved poses. Its independent
review verifies source/pose equality, hashes, visible contact and ground tail, and full video
decoding. The older camera preview is retained for provenance, not used as the final clip.

## What the new demonstration shows

The robot initially holds its starting position. Colored curves are 1.2 s predictions of the
fallback alternatives, not executed evasive motions. Both physical vehicles use the same
wind-aware nominal hover controller. Only the right fallback library continues learning.

| Simulation time | Event |
|---|---|
| 0–3 s | Calm hover; the shared initial library covers 16 direction bins. |
| 3–11 s | Wind `[1.6, 0.8, 0]` m/s; both vehicles tilt to hold their original position. |
| 11–19 s | Wind stops; the persistent learner returns toward the original maneuver set. |
| 19–23 s | A centered 5 cm box adds 25% mass; inertia changes, center of mass stays fixed. |
| 23 s onward | Release the shared eight-waypoint navigation mission with eight moving obstacles. |

No waypoints are credited during hover. Both methods have the same checkpoint, physics,
allocator, fallback mapping, nominal controller and update cadence. The fixed library never
updates. The adaptive learner retains the complete Adam state and publishes every finite
update, including during calm hover and navigation. There are no event-triggered optimizer
resets, safety acceptance gates or obstacle/goal inputs to the learner.

## Why the previous fans looked similar

The previous principal comparison already canceled the known wind in **both** fallback maps.
Similar fan shapes were therefore expected. That comparison is preserved; the separate
`compensated-control-v1` run repeats the hover/wind/recovery protocol using the principal
compensated checkpoint. At 10 s that control retains 16 direction bins in both panes; their
coordinate trajectory difference is only 0.01525 m RMS. It still has a short wind-off transient.

The main explanation is explicitly a **matched uncompensated fallback experiment**: neither
fallback map has built-in wind correction, and both predict through the same known dynamics.
The nominal task controller remains wind-aware in both panes. This creates an identifiable
behavior-restoration problem without giving the adaptive side additional physics information
or actuator authority. It must not be presented as a win over the compensated principal baseline.

The named restoration learner tracks the immutable nominal teacher's position and velocity
trajectories. Raw diversity, effort, attitude and related penalties are zero in this experiment:
those penalties can keep changing an already matched nominal library. After 200 nominal settling
updates, the current competent parameters become the frozen teacher and the initial parameters
for both panes. Full Adam history is retained; the reference is never rebased after events.
The initial saved version is 660. The original duration-gated actor is used in the main video.

A separate backward-compatible `gate_residual_with_skill_duration=False` ablation lets the
learned residual act during the braking tail. It improves some controlled wind metrics but
starts with 14 direction bins, so it is retained as a diagnostic rather than substituted into
the main radial demonstration. The default remains `True`.

## Measured outcome

Both physical hover histories are identical. The wind-on transient has a maximum position
error of 8.65 cm and a maximum tilt of 7.81 degrees; steady wind requires about 5.16 degrees
of tilt. The calm fans retain 16 direction bins. Their largest matched point difference during
the calm prefix is 0.73 mm, decreasing to 0.11 mm by 2.8 s while full-Adam learning continues.

The following comparisons use the **same recorded physical state and current model**, with
the nominal teacher evaluated at that same state. They therefore isolate library behavior.

| Diagnostic | Fixed fallback | Adaptive fallback |
|---|---:|---:|
| At 10 s, position tracking RMSE | 0.09065 m | 0.02033 m |
| At 10 s, endpoint centroid error | 0.29538 m | 0.02031 m |
| At 10 s, occupied direction bins | 7 / 16 | 15 / 16 |
| At 10 s, centered relative-shape RMSE | 0.00733 m | 0.01506 m |
| At 22 s, after centered payload, position RMSE | 0.00395 m | 0.00438 m |
| Navigation waypoints completed | 8 / 8 | 8 / 8 |
| Completion time, including hover prefix | 57.84 s | 58.08 s |
| Minimum inflated-shell clearance | +0.01538 m | +0.01544 m |
| Degraded controls | 0 | 0 |

Wind removal initially leaves the learned correction pointing the wrong way: adaptive RMSE
is 0.11302 m at 11 s, then falls to 0.00656 m at 18 s. This visible transient and recovery
are real; the library is not reset at the event. The main gain is cancellation of shared drift,
with partial recovery of directional coverage. Relative pairwise shape is not uniformly improved.
The mild centered payload does not establish an adaptation advantage. Both methods complete
this one development mission; it is not a new held-out campaign or a safety superiority result.

The [learning mechanism note](docs/da_plcbf_hover_learning_mechanism.md) gives the full
time-resolved metrics, all retained variants and independent phase/mapping checks.

Raw same-state probes, teacher trajectories, actual position/attitude histories and a summary
figure are saved in `learning/actual-hover-analysis`. The main run retains applied commands,
dense plant states, predictor trajectories, source hashes, immutable reference files, initial/
periodic/final learner snapshots and all publication records. It publishes 1,451 finite updates.
Execution is deterministic and synchronized; this video does not establish a real-time deadline.
The previous paced run's inability to fit online learning within its budget remains unresolved.

## Geometry, shadows and collision behavior

The square shadow edge came from a directional shadow volume sized using a fixed 2.5 m scene
extent. The renderer now sizes its shadow bounds from the recorded scene and projected ground
shadows. It also restores decorative geometry categories that the marker adapter dropped:
prediction curves, safety shells and wind arrows no longer cast physical shadows.

The old navigation safety sphere had radius 0.05 m. The actual `cf21B_500.xml` collider is a
0.086 m sphere offset by `[0, 0, 0.02]` m from the body origin. New navigation runs use a
conservative **0.106 m** enclosing radius; smaller legacy reproductions require explicit opt-in.
Old “physical collision-free” statistics only establish clearance of their smaller configured
sphere. They do not prove absence of contact with the rendered asset.

`contact_replay.py` provides a separate MuJoCo contact continuation. A measured collider crossing
or an explicitly requested unsafe/degraded abort hands the recorded pose and velocity to a free
rigid body. Motors are then cut as the declared demonstration response. MuJoCo computes impact,
rotation and the fall to the ground; replay masks suppress stale control and prediction claims.
An unsafe shell crossing is not labeled an obstacle contact unless MuJoCo actually reports one.
The source controller trace and safety accounting remain unchanged.

Contact dynamics use the recorded point-model mass/inertia with the XML collider geometry.
The point-model inertia differs from the visual XML's default inertia; this is recorded rather
than silently changing angular dynamics at handoff. Moving obstacles follow their prescribed
trajectory and do not become uncontrolled free objects after impact.

In the archived failed encounter, actual MuJoCo obstacle contact occurs at 4.886739 s and
ground contact at 5.574739 s. The separate seed-209 safety-abort example cuts motors at
13.84 s and reaches the ground at 14.596 s, with **no obstacle contact**. Contact parameters
are explicit numerical/material assumptions, not calibrated crash loads. The
[contact model note](artifacts/da_plcbf/hover-explanation-20260905/contact/CONTACT_MODEL.md)
records the external obstacle drive, soft-contact overlap, moving-surface momentum test,
raw contact times, source/model hashes and exact reproduction commands.

The centered payload changes mass from 43.38 g to 54.22 g and adds the box inertia about the
same origin. There is no off-center load, tether, moving center of mass or attachment impulse.
The video labels that distinction directly and uses true-sized geometry.

## Reproduction

Use `.pixi/envs/gpu-tests/bin/python`, `PYTHONPATH=.`, and
`XLA_PYTHON_CLIENT_PREALLOCATE=false` for numerical runs. Output directories must be new.

```bash
python benchmark/da_plcbf_hover_mechanism_probe.py \
  --checkpoint artifacts/da_plcbf/learning-revision-20260905/reference-ablation-seed7/candidate/checkpoint \
  --cases restoration_compensated restoration_uncompensated_gated restoration_uncompensated_full_residual \
  --output-dir /tmp/hover-probe
python benchmark/da_plcbf_hover_navigation.py \
  --checkpoint /tmp/hover-probe/restoration_uncompensated_gated/initial_checkpoint \
  --output-dir /tmp/hover-navigation
python benchmark/da_plcbf_hover_mechanism_probe.py \
  --navigation-run /tmp/hover-navigation --output-dir /tmp/hover-analysis --device cpu
MUJOCO_GL=egl JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
python examples/da_plcbf/navigation_demo.py render \
  --input-dir /tmp/hover-navigation --output-dir /tmp/hover-navigation \
  --comparison-note 'Shared wind-aware hover control · both fallback maps start without wind correction'
```

Focused regression evidence and the final video/artifact inventory are retained under the new
artifact root. Preliminary probe-0 restoration cases retained a braking penalty; probe-1 removes
it and supplies the final gated candidate. The failed `compensated-control.log` only reports a
missing checkpoint path and produced no numerical result; `compensated-control-v1` is the valid
control. Preliminary renderer/contact diagnostic outputs are not final videos.

Validation includes 21 navigation/phase/recording tests, 22 learner/checkpoint/restoration
tests, eight contact-physics tests, and 31 renderer/recording tests. Some recording tests
overlap between those groups; these counts should not be added as unique tests. The initial
hover-runner test failure was a float32 expectation mismatch and its corrected regression
passes. A later caption-only change mentions simultaneous wind and payload; all 2,053 saved
phase captions across the two completed runs are verified unchanged. The exact numerical
generating runner is retained in `numerical-generation-source`, with its original hash.
Three additional contact-render source-selection tests pass. The reusable contact video CLI
is `examples/da_plcbf/contact_replay_demo.py`; each rendered contact artifact records its exact
command, input hashes, source snapshot, sampled frame indices and saved poses. All four final
videos decode fully. `FINAL_ARTIFACT_INDEX.json` identifies them, and `ARTIFACT_SHA256.json`
inventories the complete new artifact root. `current-source/SOURCE.tar.gz` captures the final
working source; earlier numerical generation snapshots remain separately identified.
