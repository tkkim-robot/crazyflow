# Native actuator parameter reduction and initial authority atlas

The study uses `cf21B_500`, with native coefficient provenance from
`crazyflow/drones/params.toml`. This is a local model reduction of repository
parameters, including entries marked TODO; it is not a new hardware identification.

For each one-sided native RPM response, `dr/dt=a*(r_cmd-r)+b*(r_cmd^2-r^2)`.
Linearizing at `r=r_cmd=r0` gives `delta_r_dot=(a+2*b*r0)*(delta_r_cmd-delta_r)`.
The monotone thrust-coordinate transform `s=c0+c1*r+c2*r^2` has nonzero local
slope, so its local pole is unchanged. The first-order effort surrogate chooses
the mean of the spin-up/down poles, not the mean of the two time constants:
`tau=2/(lambda_up+lambda_down)`.

| Per-motor nominal effort (N) | RPM | Up tau (s) | Down tau (s) | Symmetric tau (s) |
|---:|---:|---:|---:|---:|
| 0.021362631 | 7326.616 | 0.064014276 | 0.094205705 | 0.076229437 |
| 0.106389450 | 15896.296 | 0.057068439 | 0.062145309 | 0.059498771 |
| 0.200000000 | 21660.718 | 0.053186570 | 0.050569064 | 0.051844801 |

The middle row is nominal hover at `mass*9.81/4`. The chosen nominal time constant
is **0.059498771217 s**. The two endpoint rows show the operating-region
variation discarded by a single symmetric surrogate. P2 retains the nonlinear
curves and directional response, and is therefore intentionally mismatched to P0.
All three thrust-to-RPM-to-thrust residuals are below `1e-15 N` in float64.

The accompanying `da_plcbf_actuator_parameters.json` is the machine-readable
artifact. It retains the native coefficients, fit points, parameter source hash,
and 38 initial trim cases: identity and uniform/single/adjacent/opposite effectiveness
at 0.85 or 0.70, each with lag multipliers 1, 1.5, 2 and 3, plus two declared deep
0.40 controls. The 36 proposed support cases admit the specified coupled hover;
the least per-motor headroom is 0.048015066 N. Both deep controls are infeasible
for zero-rate level hover. The single-motor deep control still has total maximum
thrust above weight, illustrating why total thrust alone is insufficient.

All torque-headroom entries retain collective thrust and the other torque entries.
Lag scales affect transients but not steady trim. Every case keeps the separate
maneuver classification `witness_not_found`: this static artifact supplies no
maneuver witness and does not by itself admit a case to obstacle-performance tuning.
No comparative controller outcomes were used to choose these parameter cases.

The source SHA-256 is `c3f4d2991925ee0ab0354107a2cfcc6be890dfbd985795e9d5d5055b55b851de`.

## Validation and reproduction

The fit is reproduced by `fit_native_time_constant("cf21B_500")` in
`crazyflow.safety.da_plcbf.actuator_dynamics`. For a current model,
`hover_authority(model)` solves the full four-equation trim and pure-axis headroom.

```
JAX_PLATFORMS=cpu .pixi/envs/gpu-tests/bin/python -m pytest -q tests/test_da_plcbf_actuator_dynamics.py
.pixi/envs/default/bin/ruff check crazyflow/safety/da_plcbf/actuator_dynamics.py tests/test_da_plcbf_actuator_dynamics.py
```

Focused validation: 22 tests pass in the repository GPU-tests environment using
its CPU backend. Body integration refinement is checked against an analytic vertical
solution, and saturated allocation is checked against independent SciPy bounded
least squares. Forward/reverse AD and finite differences include motor-state
sensitivities. The float32 zero-hover acceleration check allows `3e-5 rad/s^2`
for tiny mixer cancellation amplified by inverse inertia; this is an arithmetic
tolerance, not an expanded command or safety limit. Eight body substeps over a
0.1 s vertical analytic case satisfy the declared `8e-7` state-component tolerance.
P1/P2 validation and maneuver competence are separate artifacts.
