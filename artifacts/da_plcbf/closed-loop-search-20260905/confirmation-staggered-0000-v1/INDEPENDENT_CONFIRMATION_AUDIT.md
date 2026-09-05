# Independent check of the completed deterministic confirmation

This audit inspected the retained records, recomputed collider geometry at actual integration
nodes, checked control/publication chronology, and restored saved checkpoints on CPU. It did
not rerun the GPU episodes. Scope: `confirmation-staggered-0000-v1`, completed before the
separate earlier-intervention, narrow-neighbor refinement, buffer, and paced follow-ups.

## Supported result

The selected two-mover scene retains the original **0.15 m requested clearance** and **0.106 m
body-origin enclosure**. Both methods see the prescribed obstacles from time zero. Wind starts
at 2.20 s. The uncompensated frozen controller physically intersects the modeled XML sphere;
the adaptive controller completes both waypoints with positive requested-shell clearance.

| Plant step | Frozen collider upper bound | Adaptive collider lower bound | Frozen stop |
|---|---:|---:|---:|
| 0.020 s | −0.01717480 m | +0.17244728 m | 4.64 s |
| 0.010 s | −0.01537383 m | +0.17236835 m | 4.64 s |
| 0.005 s | −0.01441370 m | +0.17273666 m | 4.64 s |

The predictor stays at 0.020 s and each command lasts 0.040 s: the retained traces contain
2/4/8 actual plant substeps respectively. Direct quaternion-offset sphere calculations also
find negative clearance at 2/3/5 retained frozen integration nodes, respectively. The collision
therefore does not depend solely on interpolated swept chords. It is a modeled geometric
intersection, not a measured MuJoCo contact event. No arrival is credited after intersection.

The finest-versus-middle change is 0.960 mm in the frozen margin and 0.368 mm in the adaptive
margin. The selected pair passes the prespecified −2 mm/+10 mm target even after subtracting
this observed integration sensitivity. This is an empirical refinement check, not a rigorous
continuous-time integration-error bound. The original discovery and repeated confirmation
states/actions match exactly for both methods.

## Causal and local checks

- Freeze-at-wind-onset shares the exact observed calm prefix through the onset input with
  continuing adaptation, including state, action, values, and used versions. Its version-715
  library contains 55 calm updates; it subsequently collides. The no-wind pair both finish
  safely, with approximately +0.168 m collider clearance.
- All 12 prespecified uncompensated local draws produce definite frozen-only collision;
  adaptive clearance is +0.171581 to +0.174178 m, with zero degraded controls and all operational
  checks passing. **Only 11/12 meet the stricter −2 mm penetration target.** Draw 009 penetrates
  1.431 mm, first intersects at 4.598113 s, and stops at 4.60 s. All draws were retained, without
  replacement or retuning; these are local selected-case results.
- The 4.20 s intervention has identical physical state, model, goal, previous selection,
  complete published learner state, and preceding execution. Learned version 765 yields one
  eligible accepted full QP, while reverted version 660 yields no eligible QP. Both full audit
  actions exactly reproduce the executed actions. **Both intervention branches survive**:
  learned held has zero degraded controls; original-parameter reversion has ten. This late
  intervention supports a certificate effect, not necessity of late adaptation for survival.

The stronger compensated baseline survives the central case, both integration refinements,
freeze-at-onset, no wind, reversion, and all 12 local draws. Its local collider clearance is
+0.168785 to +0.171437 m. The compensated adaptive method also survives, but has zero or one
degraded control depending on the draw. This scene does not demonstrate superiority over
the compensated baseline.

## Provenance and limits

Every checked next-boundary used version equals the previous completed version; finite
updates publish at the following 0.040 s boundary, and terminal controls launch no update.
The four saved uncompensated intervention checkpoints restore with verified reference NPZ
and manifest bindings and exact recorded parameter/Adam hashes. The reverted original is
correctly marked available at 0 s; the common learned snapshot is available at 4.20 s.

All four execution-protocol source hashes were verified before subsequent API edits and their
exact bytes preserved in `SOURCES_AT_EXECUTION.tar.gz`, with archive/file hashes recorded in
`SOURCES_AT_EXECUTION.json`. Deterministic availability is simulation chronology; paced
publication and deadline claims require the separately executed canonical scheduler run.
