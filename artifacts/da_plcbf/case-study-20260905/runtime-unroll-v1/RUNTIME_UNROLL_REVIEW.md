# Unrolling the physical substeps inside each held command

Unrolling the two-step inner integration loop reduced median synchronized learner service
from **20.655 ms to 10.791 ms** on the RTX4090. The original two-anchor retention batch,
16-state rotating bank, 16-skill library, 20 ms integration, 40 ms command holds and full
1.2 s horizon remain in place. All 50 measured updates completed finitely, advancing the
same initial optimizer continuation from version 760 to 810.

| Measured service | Before | Inner scan unrolled |
|---|---:|---:|
| Median | 20.655 ms | 10.791 ms |
| p95 | 21.094 ms | 11.394 ms |
| Maximum | 22.120 ms | 11.492 ms |

The change is `unroll=command_hold_steps` on the inner `lax.scan` in
`quad_rollouts.py`. Commands are still evaluated only at their original boundaries and
held through every retained physical integration state. The outer command-horizon scan
is unchanged. This also benefits controller predictions that use the shared helper.

## Numerical effects

This compiler change is **not bitwise identical**. Fusion changes floating-point results;
the original tight blanket check (`rtol=3e-5`, `atol=1e-7`) failed for some near-zero
trajectory and gradient coordinates. Its failed result is retained in
`NUMERICAL_EQUIVALENCE.json`, rather than silently replacing its tolerances.

Across the authenticated physical state and three additional proprioceptive states,
with both calm and changed-wind models, the largest measured wind-position difference
was **6.50 micrometres**. The parameter-gradient relative L2 difference was
**1.24e-5**, and the next-parameter maximum difference was **5.96e-8**. All shapes,
dtypes, integer counters and policy-validity flags matched. The largest bounded motor
force difference was **2.31e-7 N**. `NUMERICAL_EFFECTS.json` records the individual physical
quantities; both raw captures and exact before/after source copies remain local.

Eight CPU checks pass, including explicit step-by-step execution for 1-, 2- and 3-step
holds, partial final holds, motor bounds, held wrench identity, reverse-mode gradients,
and full checkpoint/Adam continuation. The changed source requires fresh closed-loop
confirmation; prior trajectories should not be described as byte-identical reruns.

## Timing scope

GPU work was coordinated to avoid concurrent project GPU jobs. Compilation and five
discarded warmup updates precede 50 synchronized measurements. Hashing, checkpoint
serialization, input synchronization and controller/plant execution are excluded from
the learner service interval. Every finite completed update has a full bound checkpoint.

This demonstrates a service improvement. It does not itself demonstrate a paced flight
with positive updates inside every 40 ms production interval; the selected encounter must
be run through the actual scheduler to establish that result.
