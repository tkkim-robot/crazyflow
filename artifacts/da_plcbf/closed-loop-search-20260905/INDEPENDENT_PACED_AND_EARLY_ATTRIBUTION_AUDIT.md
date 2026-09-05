# Independent paced and early-intervention audit

This audit checked retained controls, actual service/publication timestamps, dense states,
complete-QP records, and CPU checkpoint restoration. No additional GPU episode was run.
Exact measurements and input-file hashes are in
`INDEPENDENT_PACED_AND_EARLY_ATTRIBUTION_AUDIT.json`.

## Paced v2 result

All four v2 methods have **zero recorded deadline misses** against the unchanged 40 ms period.

| Mapping/method | Collider lower bound | End time | Finite updates | Maximum measured boundary time |
|---|---:|---:|---:|---:|
| Uncompensated fixed | −0.01717482 m | collision stop 4.64 s | 0 | 37.277 ms |
| Uncompensated adaptive | +0.17342708 m | completed 10.52 s | 250 | 35.162 ms |
| Compensated fixed | +0.17069424 m | completed 11.28 s | 0 | 32.694 ms |
| Compensated adaptive | +0.17177606 m | completed 11.88 s | 292 | 33.950 ms |

Uncompensated adaptation has zero degraded controls; compensated adaptation has one. Before
the first obstacle arrival, the adaptive controllers actually use 106 and 112 completed
updates respectively. Both fixed methods' dense states and executed actions match exactly
between deterministic confirmation, paced v1, and paced v2. Thus the uncompensated collision
persists in the zero-miss v2 run with the same executed controls.

**Retain the v1 failure.** Uncompensated fixed missed its first deadline: 46.976 ms total versus
10.096 ms controller service. All other v1 methods had zero misses. The new discarded host
warmups cost 4.446/4.149 ms for uncompensated fixed/adaptive and 3.449/3.931 ms for compensated
fixed/adaptive. Nevertheless, v2 uncompensated fixed still spends 25.020 ms in first-boundary
host recording, within a 37.277 ms total; collider auditing takes only 0.752 ms there. The
observed warmup change and zero misses do **not** prove the precise cause of v1's startup miss
or eliminate occasional startup variability. These are measured sampled simulations, not an
OS or hardware hard-real-time guarantee.

## Causal update availability

Every retained publication links to a finite learner service with the same completed version,
training time, and measured completion time. Publication follows completion and a subsequent
simulation boundary. Every control uses the latest publication available at its actual start;
trace versions match those input records. All deadline flags agree with the recorded completion
timestamps. Restored final checkpoints have versions 910 and 952, exactly the initial 660 plus
250 and 292 finite updates, with verified reference bindings.

At the **3.00 s control**, uncompensated v2 uses version 730: 53 calm and 17 wind updates.
All 70 publications are already available before wall-clock 3.00 s. Compensated v2 uses
version 735: 55 calm and 20 wind updates; 74 publications precede wall-clock 3.00 s and the
75th appears at the actual 3.00 s boundary. Do not relabel the uncompensated paced state or
version as the deterministic version-735 intervention below.

## Separate deterministic 3.00 s parameter intervention

The follow-up boundary was selected from the previously recorded first 2 cm path-divergence
time, before observing new intervention outcomes. Both branches have exactly matching
pre-intervention states, commands, selected policies, values, versions, and publication history.
At 3.00 s, physical state, previous selection, goal, point-model leaves, and the published full
learner state match. Both stop learning thereafter. Only the actor parameters supplied to the
controller differ: available version 735 versus original version 660.

For the uncompensated pair, holding learned parameters completes with +0.17431177 m collider
clearance and zero degraded controls; reverting original parameters collides with upper bound
−0.01114508 m and 31 degraded controls. Both initial complete QPs are valid: learned has 12
eligible/accepted candidates and selects policy 9; reverted has 9 eligible/accepted candidates
and selects policy 7. Both executed policy duals are positive, and both complete-QP audit
actions reproduce the actual commands exactly. This is a causal separation in subsequent
controller outcomes, **not** an initial absence of all frozen certificates.

All eight intervention checkpoints restore with exact recorded parameter/Adam hashes and
verified reference NPZ/manifest bindings. Both reverted used snapshots exactly equal their
original full checkpoint states, available since 0 s; the learned snapshots are available at
3.00 s. The compensated intervention remains both-separated. The earlier completed 4.20 s
uncompensated intervention also remains both-separated and must remain visible as a negative
late-intervention result.
