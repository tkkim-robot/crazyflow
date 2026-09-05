# Held execution and operational prediction

`dt` is the rigid-body integration step. `PersistentSkillConfig.control_interval_steps` is the number of those steps for which each learned command is held. The nominal rollout uses the same `command_hold_steps`, and `_make_controller` rejects a learner/execution cadence mismatch. The main 60-step, 20 ms integration configuration therefore remains a 1.2 s horizon with 30 policy evaluations at a 40 ms command/QP period. The 20 ms feedback ablation remains explicitly separate.

`zero_order_hold_rollout` evaluates the command at each boundary, then advances the existing direct-wrench symplectic integrator under that constant wrench. Learning, nominal prediction, fallback prediction and probes share this implementation. Command diagnostics are repeated over each hold. A partial final hold contributes only the requested integration nodes.

A committed-skill replay retains its original start-position anchor and advances phase across command boundaries. The regression compares every state and wrench over the complete 1.2 s horizon against an independent execution loop, including the first held interval. The receding-horizon runtime starts a fresh local skill anchor and phase at each QP boundary. That runtime behavior, especially a QP-modified command, is not the same trajectory as committing to the originally predicted fallback. No claim of their equality is made.

## Crossing fixture and predictive operational repair

The immutable crossing fixture at 6.2 s is reproduced by `benchmark/da_plcbf_held_operational_probe.py`. Its command-boundary QP satisfies the instantaneous operational faces but violates the **arena_x_upper** HOCBF at the next substep, **20 ms** into a 40 ms hold, with residual approximately **−0.047247**. Physical arena, speed, tilt and rate margins remain positive. Changing feedback prediction cadence alone does not fix this fixture.

The revised controller first solves the existing QP. Only an initially accepted proposal that fails the operational hold check triggers up to three predictive refinements. At a proposed held wrench `u0`, the predictor differentiates each future substep residual `r_j(u)` through the same held-wrench integrator. It adds the local affine face

```text
−Dr_j(u0) u ≤ r_j(u0) − Dr_j(u0) u0.
```

All original motor, instantaneous operational and selected collision faces remain in the QP. The nominal objective, policy selection, physical limits and numerical postcheck tolerances are unchanged. Differentiating the predicted state makes the effects of torque on future attitude available to this correction. An independently replayed nonlinear hold must still pass physical, derivative and collision checks before the revised QP action is accepted. A failed or nonfinite refinement continues to the existing fallback/emergency decision.

The refinement is a local sequential convex approximation, not a global ZOH bound or a continuous-time invariance proof. It can fail, become infeasible, or consume additional controller service time. Its number of iterations and the pre-refinement residual are recorded. A zero-iteration configuration provides the original formulation as a labeled ablation.

Held telemetry preserves separate matrices of operational derivative residuals at substep starts and physical dimensionless margins at all held nodes. Their nine columns use `safety_constraint_names(0)`: lower x/y/altitude, upper x/y/altitude, speed, angular rate, tilt. Proposed QP, direct fallback and applied-action residuals are stored separately so an emergency command does not hide the rejected proposal's cause.

## Recorded fixture validation

The saved GPU fixture in `artifacts/da_plcbf/navigation-revision-20260905/held-operational-fixture` retains all candidate states/wrenches and named held matrices for each variant. The original committed GPU QP wrench is reproduced to the existing diagnostic tolerance. All three cases use the identical saved state/model, 60 integration steps, nominal objective, obstacle prediction, physical limits and 40 ms executed hold.

| Policy feedback / operational formulation | Upper-x held residual | Accepted QP | Median / p95 controller service |
| --- | ---: | --- | --- |
| 20 ms / original | −0.04724696 | no | 16.000 / 17.259 ms |
| 40 ms / original | −0.04724696 | no | 12.769 / 13.089 ms |
| 40 ms / predictive | −0.00000095 | yes | 15.152 / 15.503 ms |

The predictive variant uses one refinement; its maximum of 30 synchronized service samples is 16.408 ms. Its minimum physical dimensionless margin is 0.0810624 and hard collision value stays 10.24975 m². The changed pitch torque is approximately −0.00030105 N·m. These measurements establish a repair of this bounded fixture and the cost of this observed branch, not a worst-case execution budget for other states or multiple refinements.

The bounded predictive refinement supports at most **two integration substeps per command hold** in this implementation. Longer holds must explicitly set `predictive_operational_iterations=0`; the original nonlinear held-action postchecks remain enabled. This prevents silent combinatorial growth of the enumerated QP as additional substep faces are appended. Longer-hold predictive formulations require separate implementation and timing validation.
