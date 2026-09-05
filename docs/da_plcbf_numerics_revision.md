# DA-PLCBF numerical and execution revision

This note describes the controller changes following the `cbaa4ba` review. The implementation
uses one frozen policy-library snapshot and one point dynamics estimate per decision. Its
collision values describe finite-horizon predicted rollouts; they do not establish invariance
across policy selection, library publication, dynamics-estimate changes, perception updates,
or repeated replanning. They are not robustness, hardware-safety, or infinite-horizon guarantees.

The main implementation is in
[`continuous_version_a.py`](../crazyflow/safety/da_plcbf/continuous_version_a.py),
[`version_a_filter.py`](../crazyflow/safety/da_plcbf/version_a_filter.py),
[`version_a_barriers.py`](../crazyflow/safety/da_plcbf/version_a_barriers.py), and
[`quad_rollouts.py`](../crazyflow/safety/da_plcbf/quad_rollouts.py).

## Hard values, empty collision windows, and the smooth QP value

For obstacle radius `r`, ego radius `r_ego`, and requested clearance `d`, each collision value is

\[
  h(p,c)=\|p-c\|^2-(r+r_{\rm ego}+d)^2.
\]

The hard policy value `H_i` minimizes these values over active nodes and swept relative-position
segments. Segment minima are exact for the piecewise-linear interpolation of the recorded ego
and obstacle nodes, including nonzero segments shorter than a floating-point epsilon scale.
They are not exact continuous rigid-body trajectories between integrator nodes. Hard values and
their smooth substitutes have units of square metres.

A valid zero-slot or entirely masked collision horizon has `H_i=+inf`, active index `-1`, and
`input_valid=True`. Invalid active geometry instead has value `-inf`. Empty horizons remove the
collision constraint from the actual QP while retaining operational and actuator constraints.
No finite collision certificate is required in that branch. The solver uses finite internal
placeholders; its final zero-multiplier diagnostic slot is padding added after the collision-free
solve, not a fictitious active collision row. Consumers must distinguish this vacuous case from
a large finite safety margin. Appearance/disappearance transitions retain fixed array shapes.

For `n` finite node/segment values `h_j`, the QP uses the unnormalized lower bound

\[
 \widetilde H_i=-\tau_i\log\sum_{j=1}^{n}\exp(-h_j/\tau_i),
 \qquad
 \widetilde H_i\le H_i\le\widetilde H_i+\tau_i\log n.
\]

The configured temperature defaults to `0.005 m²`. With the default gap budget `B=0.03 m²`,
`tau_i=min(tau, B/log(n))` for `n>1`; the one-constraint gap is zero. Diagnostics expose both
`effective_smooth_temperature` and `smooth_gap_bound`. This caps resolution-dependent
conservatism without dividing the exponential sum by `n`, which would destroy the lower-bound
property. With an uncapped fixed temperature, duplicating every constraint `m` times lowers the
value by exactly `tau*log(m)`. The implementation does **not** claim duplicate invariance.

State gradients, time partials, and the QP bound all use this same smooth value. The hard value
remains available for finite-horizon and held-action checks. The smoothing resolves the outer
minimum's tied branches; policy switches, motor clipping, mask changes, and other piecewise
rollout operations can still be nonsmooth. This is not a cubic-spline certificate implementation.

## Temporal derivative and moving predictions

`RuntimeObstacleTrajectories.velocities` optionally supplies one velocity per obstacle or per
prediction node. At a fixed ego rollout, the defined local absolute-time update is

\[
 c_j(t+\delta)=c_j(t)+\delta v_j,
 \qquad
 \partial_t\widetilde H_i
 =\left.\frac{d}{d\delta}\widetilde H_i(x,c(t+\delta))\right|_{\delta=0}.
\]

Masks and radii stay fixed during this derivative. If velocities are absent, each node uses the
following segment's finite-difference slope and the terminal node repeats the last slope.
Inferred slopes are zero on intervals whose endpoints are not both active. This is an explicit
local affine prediction-shift convention, exact for constant-velocity absolute-time predictions;
it is not a derivative of arbitrary perception replacement or a discontinuous mask update.

For control-affine point dynamics `xdot=f(x)+G(x)u`, the selected collision row enforces

\[
  \partial_t\widetilde H_i+\nabla_x\widetilde H_i\,[f(x)+G(x)u]
  +\alpha\widetilde H_i\ge0.
\]

Equivalently, its QP row is `a=-grad(H_tilde) G` and its bound is
`b=partial_t(H_tilde)+grad(H_tilde) f+alpha H_tilde`. Candidate rollouts from forward
differentiation are reused; computing the geometry-only time partial does not roll out the
dynamics again. Every active prediction-node velocity must be finite before a temporal
certificate is selectable. This extra validation matters because differentiating an invalid
constant `-inf` branch can otherwise return a misleading numerical zero. Masked NaN padding is
ignored.

The separately selectable analytic obstacle-HOCBF baseline now uses relative velocity
`v_ego-v_obstacle`: `hdot=2 r·v_relative` and
`hddot=2||v_relative||²+2 r·a_ego`, under its **zero obstacle-acceleration** assumption.
The corrected PL-CBF mode continues to omit analytic obstacle rows while preserving arena,
altitude, speed, angular-rate, tilt, and actuator constraints.

## Control holding, execution modes, and attribution

`control_interval_steps` separates the action-holding period from the rollout/integration step.
The default is one. With `dt=0.02`, horizon `60`, and a hold of two substeps, the action period is
`0.04 s` (25 Hz) and the rollout horizon remains `1.2 s`. This configuration does not assert a
measured deadline guarantee; episode and benchmark timing must establish actual costs.

Each proposed constant wrench is integrated through all held symplectic substeps. The check
uses the matching obstacle-prediction prefix and its minimum hard swept margin. It also checks
operational physical values at all held nodes, and operational analytic domains/residuals at
every substep start. `next_estimated_state` is the final held state. These are discrete execution
checks under the point model, not a proof between all continuous times.

Actual execution has four explicit modes:

1. **`qp`**: the selected-row QP passes its original numerical, actuator, and barrier checks and
   the complete held-interval collision and operational checks.
2. **`fallback`**: the selected candidate's first wrench passes actuator, initial operational,
   and held-action checks. `fallback_valid` additionally requires the selected hard horizon
   value to pass its threshold. A finite-horizon-valid fallback can run without a nonnegative
   smooth QP certificate; a merely executable fallback with negative hard horizon value remains
   explicitly degraded. The analytic-only baseline does not execute library fallbacks.
3. **`emergency`**: an obstacle-agnostic, wind-aware velocity brake with attitude/rate
   stabilization and hard motor bounds. Its target is the current position with zero target
   velocity, default braking gain `2.0`, and acceleration cap `4.0 m/s²`. All methods use this
   same predeclared function and their available point model. It receives no goal, obstacle,
   library, or safety value. It is always degraded and is not a recovery or avoidance certificate.
4. **`midpoint`**: reserved for invalid resources that prevent an executable emergency command.
   Invalid actuator resources produce a NaN action rather than a claim of executable control.

`selected_policy_dual` describes the proposed QP's selected row. Only `executed_policy_dual`
attributes that dual to actual accepted QP execution; it is zero for fallback/emergency/midpoint.
The step exposes selected smooth value, time partial, selectable count, collision-row activity,
execution mode, applied held operational margin/residual/pass flag, and nine QP rejection flags.

The pre-existing exact shortcut remains enabled: project the nominal action onto the selected
policy halfspace in the positive-definite QP metric, then check **all** actuator/operational
constraints and KKT conditions. If those pass, it is a solution of the complete convex QP;
otherwise the full active-set solver runs. Collision-free/analytic-only cases first check the
nominal action. Policy selection still maximizes the selected face's admissible motor-box volume;
this revision does not add a search over alternative policies when the chosen QP fails.

## Exact-hover differentiation

At zero body rate, differentiating `norm(omega)` produced NaNs even though a later branch chose
a small-angle limit. The quaternion exponential now computes `s=dt²||omega||²` first and uses
the analytic polynomials `1-s/24+s²/1920` for the vector sinc factor and `1-s/8+s²/384` for the
scalar cosine factor near zero. The unused square-root branch receives a safe nonzero argument.

The existing subnormal-rate flush is preserved in the primal calculation with a stopped-gradient
correction, retaining the physical derivative at zero. The new regression checks forward/reverse
AD agreement and the direct-wrench torque sensitivity of the quaternion vector part at identity:
`dq_vector/dtorque = 0.5 dt² J_inverse` for one symplectic step.

## Observed checks and reproducible timing probes

Run commands from the repository root. CPU selection prevents accidental GPU contention:

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false pixi run -e gpu-tests pytest -q \
  tests/unit/safety/da_plcbf/test_continuous_isolation.py
```

Latest result: **11 passed in 43.65 s**, after the active-velocity validity fix. Coverage includes
smooth ties/finite differences; ego radius and masked padding; analytic obstacle isolation;
submillimetre swept geometry; duplicate/resolution conservatism; approaching and crossing
absolute-time finite differences; empty/masked/appearing/disappearing windows; invalid future
velocity; emergency behavior; two-substep operational checks; and exact-hover gradients.

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false pixi run -e gpu-tests pytest -q \
  tests/unit/safety/da_plcbf/test_continuous_version_a.py \
  tests/unit/safety/da_plcbf/test_version_a_filter.py \
  tests/unit/safety/da_plcbf/test_direct_wrench.py
```

Observed during this revision: **45 passed in 68.50 s**. This run preceded the final lazy
emergency held-check evaluation and the future-velocity validity guard; it is recorded as that
earlier regression result, not a claim that the combined suite was rerun after every final edit.
Ruff checks and formatting passed for the changed controller, isolation tests, and new benchmark.

The new [`da_plcbf_execution_branches.py`](../benchmark/da_plcbf_execution_branches.py) records
complete synchronized controller-call samples, preserving full diagnostics. It verifies accepted
fast QP, accepted full active-set QP, certified fallback, and emergency predicates before labelling
their timings. A bounded prescribed-state search leaves the exact shortcut enabled; absent
branches are explicitly reported as absent. Compilation, discovery, and extra warm calls are
excluded from samples. These are computational probes, not avoidance episodes or worst-case
execution-time bounds.

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false pixi run -e gpu-tests python \
  benchmark/da_plcbf_execution_branches.py --device cpu --policy-count 4 --horizon 20 \
  --control-interval-steps 2 --samples 2 --output /tmp/da-plcbf-branches-smoke.json

pixi run -e gpu-tests python benchmark/da_plcbf_execution_branches.py \
  --device gpu --checkpoint CHECKPOINT_STEM --control-interval-steps 2 \
  --samples 30 --output OUTPUT.json
```

The CPU smoke observed all four branches; the accepted full-solver fixture used speed `3.0 m/s`,
pitch `0.3 rad`, and pitch rate `-2.0 rad/s`. Its two timing samples only validate the measurement
path and are not a performance result for the main `K=16`, 60-step experiment. A checkpoint run
restores the saved actor/spec/config without learner updates and records its checksum. Existing
output files are not overwritten. Reserve the target device and avoid concurrent heavy jobs when
collecting reportable timing measurements.
