# Reference learner service and anchor microbatch experiment

Reducing the rotating anchor microbatch did not materially reduce learner service on the
RTX4090 in this coordinated isolated run. The two-anchor default remains justified.
One anchor retained the complete 16-state bank over its rotation; zero anchors is an
explicit retention-disabled diagnostic ablation.

| Anchors per update | Median ms | p95 ms | Maximum ms | Finite completed updates |
|---|---:|---:|---:|---:|
| 2 | 20.655 | 21.094 | 22.120 | 50/50 |
| 1 | 20.549 | 20.888 | 22.538 | 50/50 |
| 0 | 19.667 | 20.192 | 20.953 | 50/50 |

All variants resumed the same bound atlas checkpoint at library version760, retaining
exact parameters, previous parameters, Adam history, current point model, physical state,
immutable nominal teacher and complete anchor bank. Only `anchor_batch_size` changed.
Five warmup updates were discarded; each measured continuation began from version760 and
finished at810. Every finite completed update has a full checksummed checkpoint and a
consecutive before/after state digest in `completed_updates.json`.

The median full value-and-gradient probe remained about20–21ms with retained anchors,
while nominal-teacher and batched current-model rollouts each took about3.5–4ms.
These separately compiled probes are nonadditive. Batch width is not the dominant service
cost in this measurement; eliminating retention is not a demonstrated solution to pacing.

This is synchronized learner service, not a paced controller test. Compilation, input
transfer/synchronization, checkpoint serialization, hashing and controller/plant scheduling
are excluded. No positive-update40ms flight deadline claim follows from these measurements.
