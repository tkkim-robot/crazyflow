# Hover, wind removal, and fallback learning

The new recorded hover sequence shows adaptation correcting wind drift in the fallback
maneuvers, followed by recovery after wind removal. The vehicle itself follows the same
model-aware hover controller in both branches. This separates the learned fallback behavior
from the nominal controller's ability to hold position. Both branches subsequently complete
the eight-waypoint route; this example does not establish a navigation advantage for learning.

The principal equally compensated comparison remains a separate control. Adding known-model
feedforward already keeps its maneuver fan close to its nominal behavior. Removing that
feedforward from **both** fallback libraries creates the separately named
`matched_uncompensated` mechanism experiment. The model information, actuator authority,
fallback mapping, initial parameters, and full Adam state match between its branches. Learning
receives the current point model; the frozen branch uses the same model to predict its paths.
The nominal task controller remains compensated in both branches.

## Actual physical sequence

The [recorded run](../artifacts/da_plcbf/hover-explanation-20260905/hover-wind-payload-navigation/navigation_comparison.json)
holds the initial position until 23 s, applies wind `(1.6, 0.8, 0)` m/s at 3 s, removes wind at
11 s, attaches a centered 25% additional mass at 19 s, and starts navigation at 23 s. The
collision enclosure has radius 0.106 m. Integration remains 20 ms, commands are held for
40 ms, and the deterministic schedule permits one update per control boundary. Learning
starts at switch-on, with 75 completed updates before wind, 200 during wind, 200 after wind
removal, and 100 during the payload-only hover. These are simulated update opportunities,
not a measured real-time deployment claim. Both branches have exact current model information.

The actual hover state histories are identical. Wind onset causes a maximum position error of
0.08648 m and a maximum tilt of 7.81 degrees; wind removal causes a 0.08585 m transient. Steady
wind requires about 5.16 degrees of tilt. Thus the drone stays near the same point, with a
visible physically recorded transient. Learning does not cause this position recovery: the
shared nominal controller does. The centered payload causes no substantial hover deficit.

The [matched-state analysis](../artifacts/da_plcbf/hover-explanation-20260905/learning/actual-hover-analysis/actual_hover_analysis.json)
compares both recorded libraries from the adaptive branch's **same position, velocity,
attitude, rate, and current model**. Its reference is the saved nominal teacher rolled out
from that same state under its fixed nominal model. No reference is rebased after an event.
All probe starting states are checked for exact equality between libraries. Both fixed-state
anchors and adaptive-state anchors are retained in the analysis.

| Recorded time | Condition | Fixed path RMSE | Learned path RMSE | Fixed / learned direction bins |
| --- | --- | ---: | ---: | ---: |
| 3 s | Wind begins, before wind-trained update | 0.09123 m | 0.09124 m | 6 / 6 |
| 4 s | One second of adaptation | 0.09588 m | 0.02042 m | 7 / 15 |
| 10 s | Wind remains on | 0.09065 m | 0.02033 m | 7 / 15 |
| 11 s | Wind removed | 0.000013 m | 0.11302 m | 15 / 5 |
| 16 s | Five seconds after removal | 0.000016 m | 0.00974 m | 16 / 16 |
| 18 s | Seven seconds after removal | 0.000014 m | 0.00656 m | 16 / 15 |
| 22 s | Centered payload, wind absent | 0.00395 m | 0.00438 m | 16 / 16 |

Path RMSE averages squared position-coordinate errors over all skills and horizon nodes.
Direction bins use the existing 16 target directions and a 0.10 m minimum displacement.
Occupancy counts actual displacements from the shared starting point; translating a fan can
change occupancy even when its relative shape remains spread out.

At 10 s, learning reduces the endpoint centroid error from 0.29538 m to 0.02031 m. However,
the centered shape RMSE increases from 0.00733 m to 0.01506 m. The visible recovery is chiefly
correction of common wind drift, with some change to relative maneuver shape. It is partial
restoration, not exact reproduction of every radial trajectory. After wind removal the
learned correction initially points the wrong way and must be unlearned; that transient is
retained. At 18 s its centroid error is 0.00868 m and centered shape RMSE is 0.00581 m.

The calm fan begins with all 16 bins. Its largest fixed/learned point difference among the
0–3 s probes is 0.000732 m at 0.2 s, declining to 0.000108 m at 2.8 s. This small drift is
measured while the complete previous Adam history is retained and learning continues.

The [six-panel figure](../artifacts/da_plcbf/hover-explanation-20260905/learning/actual-hover-analysis/actual_hover_analysis.png)
separates physical hover error, tilt, absolute path tracking, centroid drift, centered shape,
and direction occupancy. The [PDF](../artifacts/da_plcbf/hover-explanation-20260905/learning/actual-hover-analysis/actual_hover_analysis.pdf)
and raw reference trajectories accompany it. The analysis records SHA-256 hashes of all seven
input trace/reference files.

Both methods finish 8/8 waypoints with zero degraded controls. Minimum inflated clearances
are 0.01538 m fixed and 0.01544 m adaptive; completion times are 57.84 s and 58.08 s. These
similar task outcomes do not support a claim that learning was necessary for this route.
The payload interval likewise supplies no learning advantage: the fixed library already
tracks the reference closely.

## Why the earlier fans looked similar

The [compensated control](../artifacts/da_plcbf/hover-explanation-20260905/compensated-control-v1/navigation_comparison.json)
uses the original compensated checkpoint and objective in both panes. At 10 s both libraries
occupy all 16 bins. Their same-state coordinate path difference is 0.01525 m RMS, with a
maximum point difference of 0.07366 m. At 18 s those differences are 0.00299 m and 0.01266 m.
There is a short adaptation transient immediately after wind removal: at 11 s the learned
library has 13 bins while the frozen library has 16; both have 16 again at 12 s. The principal
control ends at 19.04 s, with only one 40 ms navigation command after its 19 s release. That
truncated control is not a navigation-success experiment; its timeout is retained explicitly.

In the earlier fixed-hover-state probe, the compensated frozen library's wind tracking error
was already only 0.02851 m and learning reduced it to 0.01835 m. The large visible change in
the new mechanism experiment is therefore not evidence of superiority over model feedforward.
It demonstrates learned correction when both fallback actors use the same uncompensated map.

## Restoration objective and retained development cases

The previous reference teacher was saved at version 400 while the deployed parameters were
at version 460. In addition, raw diversity, pairwise, action, attitude, and related penalties
can have a nonzero gradient even when actual trajectories match the nominal teacher. A
linear positive-part braking-excess penalty was also sensitive to small numerical differences
at its zero kink. These are legitimate optimization pressures, but they are unsuitable for
an explanatory example that expects the initial repertoire to remain approximately unchanged.

The separately named restoration objective uses only nominal-reference position/displacement
and velocity tracking, including terminal velocity, with the existing rotating state-bank
retention. Its raw regularizer weights and separate terminal-braking weight are zero. The
teacher is first rebased to the starting parameters, 200 nominal settling updates run with
the existing Adam history, and the teacher is then rebased to the resulting version-660
parameters before deployment. No optimizer reset is used. The nominal model and teacher
remain fixed throughout wind, wind removal, payload attachment, and navigation. No obstacle,
goal, clearance, filter acceptance, or current-model frozen rollout enters the learner target.
Every finite update publishes; there is no safety-based rejection.

The [pure restoration probe](../artifacts/da_plcbf/hover-explanation-20260905/learning/hover-probe-1/protocol.json)
retains all three controlled cases below. Each starts from its own declared, fully settled
common checkpoint, then performs 200 wind-on and 200 wind-off updates at a fixed upright
hover state. These fixed-state measurements motivated the physical run; they do not replace it.

| Fallback mapping | Initial bins | Wind error, fixed → learned | Learned error after 200 wind-off updates |
| --- | ---: | ---: | ---: |
| Compensated | 16 | 0.02853 → 0.01785 m | 0.00216 m |
| Uncompensated, original residual duration gate | 16 | 0.09126 → 0.03162 m | 0.00609 m |
| Uncompensated, residual available during braking | 14 | 0.08234 → 0.02557 m | 0.00401 m |

The backward-compatible `gate_residual_with_skill_duration=True` default exactly preserves the
old actor. Setting it false allows learned acceleration during the late braking interval;
otherwise that interval's wind rejection is limited to the fixed velocity-feedback term.
The optional case improves centroid recovery but starts with only 14 occupied bins, so the
main recorded sequence uses the original gate and the 16-bin initial repertoire. The optional
flag is stored in checkpoints and must agree with the reference actor contract.

All 400 wind-on/off updates in each case are finite. Behavior returns close to the reference
without the parameters returning to their original values: for example, the original-gate
network retains parameter-distance norm 3.84 after wind removal. Durations remain frozen.
The initial nominal loss gradient in the full-size pure tracking experiment is small but not
bitwise zero because reference and batched rollouts differ slightly numerically. A discarded
20-update calm copy drifts by only 0.000096 m path RMSE in the selected original-gate case.

The [initial probe](../artifacts/da_plcbf/hover-explanation-20260905/learning/hover-probe-0/protocol.json)
is retained with the original principal control and three restoration variants that still
used braking-excess weight 2. The corrected pure-tracking run records weight 0 and includes
copies of all three generation source files whose hashes were verified against its protocol.
The initial failed zero-gradient regression is also retained; the corrected pure objective
passes it. No failed numerical result was replaced.

## Reproduction and checks

The [driver](../benchmark/da_plcbf_hover_mechanism_probe.py) creates new directories and refuses
to overwrite previous results. The selected deployment checkpoint is
`artifacts/da_plcbf/hover-explanation-20260905/learning/hover-probe-1/restoration_uncompensated_gated/initial_checkpoint`;
its sibling `nominal_reference` is bound by both NPZ and JSON digests.

```sh
env PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false .pixi/envs/gpu-tests/bin/python \
  benchmark/da_plcbf_hover_mechanism_probe.py \
  --cases restoration_compensated restoration_uncompensated_gated restoration_uncompensated_full_residual \
  --output-dir artifacts/da_plcbf/hover-explanation-20260905/learning/hover-probe-reproduction \
  --device gpu

env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .pixi/envs/gpu-tests/bin/python \
  benchmark/da_plcbf_hover_mechanism_probe.py \
  --navigation-run artifacts/da_plcbf/hover-explanation-20260905/hover-wind-payload-navigation \
  --output-dir artifacts/da_plcbf/hover-explanation-20260905/learning/actual-hover-analysis-reproduction \
  --device cpu
```

The [focused learner regression log](../artifacts/da_plcbf/hover-explanation-20260905/learning/hover-learner-regression.txt)
records **22 passed** across persistent learning, reference/checkpoint behavior, and the new
restoration tests. The new tests check braking-tail expressiveness, the unchanged default,
nominal reference matching, retained Adam motion even at zero current gradient, frozen
durations, and rejection of an inconsistent reference gate. Ruff passes for the affected
learner modules, driver, and new test file.
