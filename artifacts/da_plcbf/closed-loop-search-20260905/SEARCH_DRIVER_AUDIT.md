# Independent discovery-driver audit

Scope: the initial `benchmark/da_plcbf_closed_loop_search.py` implementation during this revision.
These are CPU orchestration/contract checks, not additional physical closed-loop experiments.

The implementation runs the same complete controller factory in both methods. It reconstructs
the prescribed world at absolute time zero for each run, queries the same absolute-time obstacle
predictions and known dynamics, and allows each trajectory to evolve under its own applied
commands. Finite updates use that method's recorded pre-control state and publish for the next
control boundary. The frozen method retains its checkpoint. A freeze-at-onset run reproduces
the continuously adapting method's calm history, then holds the available onset version.

The cached-candidate proposal path intentionally samples rejected hard-H rows; these do not need
an initial certificate advantage to receive a full episode. Random generators add static guards,
an independently prescribed second mover, or navigation and initial velocity before encounter.
Reduced extra clearance preserves the actual collider, enclosing radius, obstacle trajectories
and physical operating limits.

One classifier defect was found and repaired by the driver owner before the initial campaign:
promotion previously checked numerical bounds without excluding censored records. Promotion
now requires observed fixed collision, separated adaptive geometry, uncensored histories, positive
adaptive operational margins and completed encounter/task. A known already-observed collision
takes precedence over a later censor label in physical classification.

`tests/unit/safety/da_plcbf/test_closed_loop_search.py` contains 19 checks covering the four physical
outcome classes; shell/ground distinctions; censor, uncertainty, timeout and operational failure;
rejected-H proposals; the four scene families; matched information; actual command propagation;
completed/rejected snapshot publication; and the onset-freeze comparator. The orchestration tests
use a simple deterministic controller/plant stub to isolate driver behavior; they do not establish
the physical safety or numerical equivalence of the real learned controller.

Validation command:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  .pixi/envs/gpu-tests/bin/python -m pytest \
  tests/unit/safety/da_plcbf/test_closed_loop_search.py -q
```

Result: **19 passed in 1.41 s**. Ruff passed. Subsequent formatting changed no behavior.

Two publication/restart concerns were reported to the driver owner, independent of controller
physics: preserve source bindings when resuming instead of silently replacing an earlier batch's
protocol; retain/classify partial attempts when an interrupted pair has written its directory
before its trial ledger row. Count distinct physical scene/mapping pairs without treating a
different unused explicit-scene seed alone as a different geometry.
