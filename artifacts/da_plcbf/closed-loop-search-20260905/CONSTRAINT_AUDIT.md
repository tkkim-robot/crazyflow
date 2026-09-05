# Constraint and outcome-search audit

Audited starting implementation: `0bcd4a17b03d0fc99f4bdcc024b866090072fa43`.
This is a source audit, not an additional closed-loop experiment. Line references below
identify that starting revision; the search revision may move them.

## There is no additional analytic obstacle HOCBF in this comparison

`build_navigation_controller()` creates `ContinuousVersionAConfig` without overriding
`analytic_obstacle_hocbf`; its default is **false** (`navigation_experiment.py:198`,
`continuous_version_a.py:181`). The runtime explicitly replaces the barrier setting with that
flag (`continuous_version_a.py:870`). Therefore the more generic
`VersionABarrierConfig.include_obstacle_hocbf=True` default is **not** the effective setting of
the fixed/adaptive navigation comparison. Removing an analytic obstacle HOCBF would leave this
comparison unchanged.

The analytic-only baseline elsewhere in the repository can enable obstacle HOCBF while disabling
the policy-library row. That separately named method is not secretly added to the PLCBF pair.

The enabled independent operational faces are six arena/altitude faces, speed, angular rate,
and tilt (`version_a_barriers.py:194`, `:500`). Position/altitude and tilt use higher-order
barrier equations; speed/rate use first-order equations. The current world limits are 3.5 m/s,
12 rad/s, 0.9 rad tilt, arena lower `[-5,-4,0.15]`, upper `[5,4,4]`, with 0.08 m arena
clearance. Motor-allocation limits remain enforced. These can influence an escape, but they
are not obstacle-avoidance faces.

The held-command operational check and predictive operational QP repair explicitly mask every
obstacle and force `include_obstacle_hocbf=False` (`continuous_version_a.py:643`, `:691`). The
repair uses the same nine operational residuals; it does not add a hidden geometric obstacle
constraint. Retain these physical limits in the primary matched experiment.

## The buffer is substantial and is a valid matched scenario variable

The actual XML collider is a sphere of radius **0.086 m**, centered **0.020 m above the body
origin**, with its offset rotated by attitude. The controller's body-origin enclosing sphere
has radius **0.106 m**. Its signed obstacle clearance is therefore between 0 and 0.040 m more
conservative than the actual offset sphere, depending on attitude and approach direction.

The current explicit requested clearance is another **0.150 m**. For an obstacle of physical
radius `r`, controller collision values and held-command checks use

```text
||body_origin - obstacle_center||² - (r + 0.106 + clearance)².
```

Thus the existing requested shell can be 0.15–0.19 m outside the actual collider boundary.
This can allow a sizeable shell violation without actual collision. It is a plausible contributor
to the observed physical survival, but that causal effect requires paired execution to establish.

Use labeled clearance settings **0.15, 0.05, 0.02, 0.00 m** with the **same setting in both
methods**. Retain the 0.15 m scene as the original reference. Change only
`world.config.obstacle_clearance` and derive all geometry/trace metadata from the world. Keep
`ego_radius=0.106`, XML radius/offset, physical obstacle radii, actuator limits and control period
unchanged. Zero extra clearance remains an enclosure-based geometric test, not a point robot.
Its result is a different requested-clearance configuration and must be identified as such.

The conservative smooth minimum is a second source of conservatism in value space. The current
temperature is 0.005 m² and the hard/soft gap bound is capped at 0.03 m²
(`continuous_version_a.py:183`, `:426`). This is not an additional fixed radial inflation:
the actual gap depends on the competing minima and enabled node/segment count. Keep it fixed
in the initial clearance sweep so that changed outcome can be attributed to the stated change.

## The actual rescues are part of the complete controller

Every decision recomputes the nominal plus sixteen fallback rollouts at the current state.
The nominal rollout and emergency velocity brake use known-model compensation even in the
matched uncompensated **fallback** mapping (`navigation_experiment.py:226`,
`continuous_version_a.py:712`). The fallback compensation setting must therefore be reported
as a mapping distinction, not as absence of model information throughout the frozen method.

The selected QP must pass instantaneous actuator/barrier checks and a geometric/operational
check over the held command. If the QP fails, a selected fallback prefix can still execute when
its immediate held action is executable, even if its complete rollout is uncertified; it is
marked degraded. If that is unavailable, the obstacle-independent compensated velocity brake
executes as best effort (`continuous_version_a.py:965–1019`). A negative initial hard H does
not imply that the resulting sequence of prefixes and braking will collide.

The held geometric check is part of PLCBF execution acceptance. Disabling it would change the
runtime safety contract and is not equivalent to disabling analytic obstacle HOCBF. No such
removal is needed for the user's requested matched reduced-clearance or dense-obstacle tests.

## Correctness conditions for the next search

- At the reviewed revision, the navigation runner stops at the body-origin enclosure
  (`navigation_experiment.py:666`). If that happens before XML intersection, later contact is
  **censored**, not an observed safe episode. The continuous search must retain the same control
  path through enclosure breaches and stop only at its declared actual-collider/floor event or
  task termination. An adaptive physical operational violation is a separate failure even if
  it avoids an obstacle.
- The existing collider audit bounds curvature of analytic obstacles and the rotating offset
  under **interpolated recorded states**, not unrecorded plant integration error
  (`case_study_world.py:321`). A sign whose interval straddles zero is unresolved. Confirm a
  promoted case with finer plant integration while preserving the control period and prediction
  horizon, in addition to its geometric interpolation check.
- Evaluate obstacles, guard spheres and extra movers from time zero on the same absolute clock
  for both methods. Rerun the complete scene after geometry changes; an obstacle-free late prefix
  is not the trajectory of the new full scene.
- Count complete paired episodes separately from cached geometry checks and controlled branches.
  Rank actual collider outcomes/near misses without an exclusive initial hard-H gate. Preserve
  both-safe, fixed-only collision, adaptive-only collision, both-collision, operational failure,
  timeout and unresolved/censored outcomes.
- Preserve the compensated mapping, no-wind ablation and freeze-at-wind-onset comparator for a
  promoted result. The original t=4 s snapshot includes 75 calm plus 25 wind updates, so its entire
  difference from version 660 cannot be attributed solely to those 25 wind updates.
- `screen_cases()` currently writes `inflated_radius=r+0.15` (`case_attribution.py:446`) despite
  using `drone_radius=0.106`. The intended body-origin shell is
  `r + world.config.ego_radius + world.config.obstacle_clearance`. Fix the metadata, deriving it
  from the same world as the numerical audit. This discrepancy does not explain old numerical
  collision outcomes.

Matched removal of a constraint that is already disabled needs no new experiment. Matched
clearance variation, prescribed guards, staggered movers and a moving waypoint leg are supported
development directions. The resulting scene remains an empirical case; none of these changes
guarantees that an adaptive-survival/frozen-collision outcome exists.
