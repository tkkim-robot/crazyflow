# Isolated disturbance controls

These 20 paired runs complete the wind-only and payload-only conditions on the same seeded waypoint/moving-obstacle task. They reuse seeds 100–109 from the unchanged/combined campaign, the fixed nominal reference candidate (teacher seed 7, LR .001), the original frozen controller/learner/runner and deterministic publication schedule. Only the documented rejected-proposal reporting/export patch differs. They are compositional controls, not an independent new seed family or a deployment timing experiment.

| Condition | Method | Joint strict successes | Minimum shell clearance | Median completion |
|---|---|---:|---:|---:|
| Wind only | Frozen | 10/10 | 0.009572 m | 31.00 s |
| Wind only | Adaptive | 10/10 | 0.009416 m | 30.96 s |
| Payload only | Frozen | 10/10 | 0.008628 m | 31.14 s |
| Payload only | Adaptive | 10/10 | 0.008521 m | 31.10 s |

Joint strict success requires all eight waypoints, positive inflated-shell clearance, no physical collision, zero degraded controls, passing recorded actual operational and motor checks, and zero violating applied derivative checks under the existing recorded audit tolerances. All four method/condition cells pass each criterion. A descriptive 95% Wilson interval for each 10/10 count is [0.7225,1]; these small, reused world samples do not establish a general safety guarantee.

Paired adaptive-minus-frozen completion time averages −0.028 s for wind (bootstrap 95% interval [−0.068,0]) and −0.016 s for payload ([−0.028,−0.004]). The payload difference is one 40 ms control boundary in four worlds; it does not establish a meaningful safety benefit. Paired mean shell differences are +0.000149 m for wind ([−0.000324,+0.000629]) and −0.000199 m for payload ([−0.001244,+0.000727]). All waypoint and degradation-count differences are zero. Adaptive runs complete 3422 finite updates across wind worlds and 3418 across payload worlds, without a demonstrated safety advantage over the frozen candidate.

The composition audit verifies every world geometry/config and candidate/runner hash, both the isolated run and its unchanged baseline. All 40 method runs match their unchanged counterparts bit for bit before the first event in state, applied/nominal command, goal, selected policy, library version, cumulative updates, update norm, descriptors, estimated wind and valid-control mask. Dense plant states also match through the event boundary. Recorded active-row wind, payload mass and centered payload inertia match the declared event schedules. The exact 5 cm payload stays inside the existing 5 cm-radius ego enclosure. Event-window clearances and degraded counts use actual executed intervals; terminal padding contributes no samples. No first-event prefix failure or physical/operational failure was observed.

The follow-up does not change the separate negative runtime result from paced validation. Service times in these deterministic concurrent runs are excluded from deployment timing claims.

Files:

- `ISOLATED_DISTURBANCE_PROTOCOL.json`: immutable pre-run declaration.
- `ISOLATED_DISTURBANCE_EXECUTION_PROTOCOL.json`: exact source and candidate lock, including the three output-only changes.
- `ISOLATED_DISTURBANCE_STATISTICS.json`: all individual paired differences and Wilson/bootstrap statistics.
- `ISOLATED_DISTURBANCE_COMPOSITION_AUDIT.json`: per-method prefix, event-model, clearance-interval and joint-success evidence.
- `ISOLATED_DISTURBANCE_COMMANDS.md`: reproduction commands.
- `isolated-wind/` and `isolated-payload/`: all 20 complete numerical runs and raw diagnostics.

The analysis-only audit script is `benchmark/da_plcbf_navigation_composition_audit.py`; its SHA256 is recorded in its output. The separately declared static seed 0 mission also completed: both methods reached all eight waypoints at 30.96 s, with zero degraded controls and shell margins of 0.013208 m (frozen) and 0.013454 m (adaptive). Actual operational, motor and applied derivative checks passed. Recorded obstacle centers are constant and their analytic velocities are zero. `STATIC_COMPOSITION_DEVELOPMENT_AUDIT.json` records this single development mission; it makes no static-family claim.
