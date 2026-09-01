# DA-PLCBF aggregate report

Synthetic schema/replay smoke only; this is not scientific or safety evidence.

> Under the logged model/scenario samples, constraints, numerical tolerances, and finite horizon, the hard rollout and filter checks observed the reported margins and violation rates. This does not prove infinite-horizon, distribution-free, real-world, or hardware safety.

This table is descriptive. It does not assert statistical superiority.

| Method | Condition | Seed | Minimum hard margin | Failures | Degraded |
|---|---|---:|---:|---:|---:|
| da_plcbf_full | ballistic_ball | 0 | -0.498397676 | 6 | 52 |
| nominal_only | ballistic_ball | 0 | -0.926510688 | 17 | 150 |
| da_plcbf_full | dynamics_change | 0 | -2.39778634 | 17 | 80 |
| nominal_only | dynamics_change | 0 | -1.41966202 | 16 | 150 |
| da_plcbf_full | interceptor_drone | 0 | -0.465297841 | 22 | 22 |
| nominal_only | interceptor_drone | 0 | -0.872193447 | 109 | 150 |
| da_plcbf_full | static | 0 | 0.116558408 | 0 | 10 |
| nominal_only | static | 0 | 0.275924517 | 0 | 150 |
