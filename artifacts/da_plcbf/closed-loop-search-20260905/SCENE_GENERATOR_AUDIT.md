# Offline scene-generator audit

This audit examined `random_scene`, `existing_proposals`, and `mutate_scene` before
the full-episode campaign. It generated 128 configurations per family using seed
37103 independently for each family, plus two mutations of every configuration
using seed 37203. These are **512 geometry validations and 1,024 mutation
validations, not executed paired episodes**. The exact inspected source hash and
per-configuration measurements are in `SCENE_GENERATOR_AUDIT.json`.

All 512 initial scenes and all 1,024 sampled mutations passed world validation.
Each family contained 64 uncompensated and 64 compensated pair proposals and 32
proposals at each optional clearance: 15, 5, 2, and 0 cm. Every scene retained the
0.106 m body-origin enclosure, unchanged physical sphere radii, and common arena.

| Family | Minimum initial shell clearance | Additional findings |
|---|---:|---|
| Cached single mover | 4.617 m | Navigation starts 1.20–1.24 s after arrival. |
| One mover and guards | 0.108 m | 24/128 guard pairs physically overlap; 230/256 guards lie above hover. |
| Staggered movers | 4.807 m | Navigation starts 1.10–1.14 s after the second arrival. |
| Moving waypoint leg | 3.855 m | Initial speeds 0.314–1.096 m/s; navigation starts at 2 s. |

Overlapping guards are legitimate fixed unions of spheres and do not invalidate
the physics. They should not be described as two separated corridor walls. The
guard placement is meaningfully close to the initial vehicle and biases against
upper escapes: 39/128 unions also intersect the straight waypoint route's requested
shell. Whether either controller can negotiate these unions requires full runs.

## Moving-state timing diagnostic

A deliberately approximate point-mass timing calculation used the same nominal
position/velocity gains (2.0/2.8), the componentwise acceleration command
`1.2*tanh((2*(goal-position)-2.8*velocity)/1.2)`, a 0.02 s symplectic integration
step, and waypoint checks at 0.04 s boundaries. It omits attitude, allocation,
wind, and the safety filter, and therefore is **not a controller evaluation**.

In this surrogate the two waypoints finish at 7.52–8.04 s. None of the 128 routes
finishes before arrival, and three finish before arrival plus the driver's 0.8 s
encounter-completion margin. Thus early task completion is a modest possible
waste of budget for this sample, not evidence of an actual censored collision.
The driver's `encounter_completed` requirement is necessary and should remain.

The more substantial issue is encounter alignment: only 60/128 surrogate positions
are within 1 m of the prescribed crossing center at arrival. Their median distance
is 1.11 m and maximum is 3.04 m. Later arrivals often occur after the vehicle has
started its return leg. Outcome refinement should place a prescribed crossing near
a recorded moving-state passage and then rerun both full episodes. A nominal
route miss remains a useful outcome to count, rather than a collision test that
should be silently discarded.

## Disturbance and refinement distributions

Multiplying the sampled vertical wind component by 0.2 before normalizing the full
vector does **not** bound vertical wind to 20% of its final magnitude. The sampled
guard family includes vertical wind from −2.40 to +2.56 m/s. This is valid 3D wind,
but it must not be described as negligible vertical forcing. `mutate_scene`
currently clips vertical wind to ±0.8 m/s, so this development refinement has a
different explicitly defined disturbance distribution.

The generic mutation changes the first mover and wind, and slightly shifts guards.
It does not change the second mover's speed/direction or moving initial velocity.
The separate executed-escape helper adds guard or independently timed mover
proposals to challenge the **observed feedback trajectory**, rather than merely
rescreening the committed library. Its old-path audits are proposal diagnostics;
they do not establish the feedback response to the new scene.

## Exact continuous smoke comparison and a measured escape

`SMOKE_CONTINUOUS_TRACE_COMPARISON.json` binds and compares the new compact U132
smoke against the previous `continuous-deterministic-v1` run. Both methods match
exactly in all 579 dense states, 289 control states/applied actions, hard and smooth
values, augmented policy selection, modes, QP/fallback/emergency flags, executed
duals, and parameter versions. Maximum difference is zero. No compiler or wrapper
discrepancy is observed. The old display `selected_policy` is shifted by one;
comparison uses the old raw `selected_index` with its augmented nominal entry.

The actual full-episode frozen escape is primarily toward negative y. At 4.84 s
the two body positions are 0.238 m apart: fixed approximately
`[-0.055, -0.491, 1.358]`, adapted `[-0.011, -0.258, 1.377]`. The earlier controlled
branch's upward/braking explanation should not replace this continuous-scene
measurement.

One concrete development proposal adds a static sphere of radius 0.18 m at offset
`[-0.055, -0.60, -0.042]` from the initial hover point. It starts outside the initial
requested shell even with the 15 cm buffer. If the **old recorded paths** were
retained through 7 s, fixed would penetrate the modeled collider by 0.155 m and
adapted would retain 0.066 m collider clearance. Adapted would breach the requested
shell by 0.106 m. The explicit scene and this limited diagnostic are recorded in
`u132-lateral-guard-scene.json` and `U132_LATERAL_GUARD_PROPOSAL.json`.

This is not a discovered collision-versus-survival episode. The guard must be
visible from time zero in both reruns, and either controller may choose a different
earlier escape. Clearance variants likewise require complete reruns.

## Implemented executed-escape refinement

`benchmark/da_plcbf_closed_loop_refinement.py` now reads completed trial outcomes
and their dense physical paths. It balances parent selection across compensation
mappings while retaining buffer and approach-direction strata. Physical scene
identity excludes the random seed, so changing only a seed does not increase the
count of distinct configurations.

The helper cycles through static-guard placement, a second prescribed mover, and
a small local mutation aimed at a recorded closest approach. Blockers are fitted
near measured fixed-controller escape states, checked against the recorded adapted
path, and validated at time zero. If no valid blocker separates the old paths, a
valid local mutation is retained with an explicit fallback label. No controller,
emergency rule, physical radius, or information schedule is changed. Audits retain
the time support of each old path and cannot be read as new controller outcomes.

The CPU helper smoke in `refinement-helper-smoke-v2` generates three distinct
proposals from U132 with exact parent scene and trace bindings. It executes zero
new paired episodes. The emitted `proposals.json` can be consumed by the regular
full-episode driver using its existing `--resume` option; executed results and
counts remain the driver's responsibility.

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  .pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_closed_loop_refinement.py \
  --parents artifacts/da_plcbf/closed-loop-search-20260905/smoke-u132-v1/trials.jsonl \
  --output /tmp/crazyflow-escape-proposals-new --count 3 --seed 57301
```
