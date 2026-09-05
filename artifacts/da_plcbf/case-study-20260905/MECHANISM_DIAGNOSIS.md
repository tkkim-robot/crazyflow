The selected persistent-wind cases demonstrate improved certificate availability and, in the stronger family, preservation of the requested safety shell. They do not demonstrate frozen collision versus adaptive survival. The original frozen controller survives by combining short fallback prefixes, renewed predictions, later QP actions, and sometimes its normal emergency brake. A negative value for every initial committed rollout does not imply that every sequence of those control decisions will collide.

This report analyzes the completed `closed-loop-v2` and `closed-loop-v3` development branches and the `qp-v3` full-policy audits. Each pair starts from the authenticated same hover state and absolute time. The adaptive parameters were already available and are held fixed during these branches. These are controlled branch results; continuous-prefix, critical-skill replacement, neighborhood, and paced results are separate experiments.

The reproducible CPU analysis is `benchmark/da_plcbf_case_mechanism.py`. `mechanism-v2` analyzes `closed-loop-v2`; `mechanism-v3` analyzes `closed-loop-v3`. Their per-case JSON files preserve input hashes, original committed skill geometry, every frozen skill's XML clearance, common nominal/emergency/stationary trajectories, and the initial control chronology. The saved atlas trajectory advances skill phase through the horizon. The deployed controller recomputes its candidate rollouts at every 40 ms boundary and can select another action.

**Why both methods survived the first 12 cases.** All initially selected frozen committed skills in `closed-loop-v2` already avoid the actual oriented XML sphere: their minimum clearances range from +0.0610 to +0.1652 m. Their negative collision values arise from violations of the larger requested safety shell, with shell clearances from -0.1067 to -0.0112 m. Every case retains between one and eight physically clear frozen skills. Five of the six uncompensated cases also have an initially predicted emergency trajectory that avoids the XML sphere; all six compensated emergency trajectories intersect it.

The actual frozen controller first executes a selected fallback prefix in every case. No emergency command is used in this family. The first accepted QP occurs 0.04–1.08 s after the branch begins. The frozen method incurs 1–14 degraded controls, but it ultimately preserves the shell in all 12 cases. The adaptive held snapshot has zero degraded/fallback/emergency controls throughout all 12 windows and supplies learned positive-dual QP constraints at 13–31 executed controls. For the compensated family these contributions start immediately; for some uncompensated cases the first positive learned dual appears only 0.08–0.48 s later. An initially useful certificate need not immediately change the action.

At the second control boundary (+40 ms), the actual command differs from the advancing-phase original skill in every case. This establishes that executing the original whole rollout is an inappropriate counterfactual for the normal frozen controller. Small compilation-dependent first-hold numerical differences are separately recorded in the JSON rather than asserted to be bitwise zero.

**The stronger physical-threat family.** `geometry-v3` also requires intersection of the initially predicted actual-XML emergency trajectory and fits the radius while retaining a positive adaptive shell value. In two selected uncompensated cases, every original committed fallback trajectory intersects the XML sphere. The actual frozen controller still avoids XML intersection by composing controls over time.

The strongest analyzed uncompensated branch is `uncompensated-000-t0100-132`, starting at 4.00 s, one second after wind onset. Its learned snapshot had 25 completed wind updates before this control call. The learned and frozen initial best augmented policy identities are both 5, meaning fallback skill index 4.

| Quantity | Frozen | Adapted snapshot held |
|---|---:|---:|
| Initial runtime augmented hard value, m² | -0.234590 | +0.035299 |
| Initially eligible complete QPs | 0 | 1 |
| Initially accepted QPs after held checks | 0 | 1 |
| Degraded executed controls in 3 s | 36 | 0 |
| Fallback / emergency controls | 19 / 18 | 0 / 0 |
| Learned positive-dual QP controls | 0 | 18 |
| Minimum safety-shell clearance, m | -0.081758 | +0.007777 |
| Minimum actual-XML geometry clearance, m | +0.094023 | +0.183946 |

The original skill-4 committed rollout would intersect the XML sphere by 4.46 mm, first at approximately 5.15836 s, and the initial emergency rollout intersects by 59.26 mm. All 16 committed frozen fallback trajectories intersect. These are model-trajectory geometry diagnostics, not measured MuJoCo contacts.

The actual frozen controller does something different. It executes policy-5 fallback commands from 4.00 through 4.68 s, switches to the normal emergency brake at 4.72 s, briefly executes the nominal candidate as a fallback at 5.44 s, and returns to accepted nominal QP control at 5.48 s. When emergency begins, the robot is already at approximately (-0.05079, +0.01588, 1.54061) m, moving at (-0.25313, -0.00785, +0.29127) m/s. The earlier fallback maneuver has changed the state from which braking occurs. The shell breach begins at 4.72969 s, but the oriented physical collider remains clear by at least 94.02 mm. Calling this an actual collision would be incorrect.

The adaptive branch uses a learned positive-dual policy-5 QP immediately, then other learned constraints as needed, and keeps the shell. It obtains a stronger requested-margin outcome with no degraded execution. This accumulated snapshot is sufficient in the controlled branch; identifying one uniquely necessary skill still requires the explicit mixed-evaluator ablation.

The stronger compensated example is `compensated-002-t0100-487`. The initial frozen selected policy is 10, corresponding to fallback skill index 9. Its committed trajectory is physically clear by 142.82 mm but violates the requested shell by 21.85 mm. Four original frozen skills remain physically clear, so this example does not exhaust the physical frozen repertoire.

| Quantity | Frozen | Adapted snapshot held |
|---|---:|---:|
| Initial runtime augmented hard value, m² | -0.026717 | +0.035004 |
| Initially eligible / accepted complete QPs | 0 / 0 | 1 / 1 |
| Degraded executed controls | 24 | 0 |
| Fallback / emergency controls | 22 / 2 | 0 / 0 |
| Learned positive-dual QP controls | 0 | 21 |
| Minimum safety-shell clearance, m | -0.000903 | +0.007324 |
| Minimum actual-XML geometry clearance, m | +0.161256 | +0.174233 |

Here the frozen controller executes fallbacks until 4.88 s, applies emergency braking for two controls, and resumes QP at 4.96 s. Its shell intersection begins at 4.89646 s. The 0.9 mm shell violation is a narrow outcome requiring integration and neighborhood confirmation; it is not persuasive physical-collision evidence. The adaptive branch maintains the shell without degradation.

**Complete-QP diagnosis.** The current-source `qp-v3` audit evaluates all originally eligible policies for all 12 selected geometry-v3 cases. Every fixed method has zero eligible candidates. Every adaptive method has one or two eligible candidates, and every one passes the unchanged full QP, predictive refinement, and held checks. Thus these initial states do not exhibit an eligible-policy selector failure or a hidden operational infeasibility behind favorable motor-box scores. The useful difference is availability of executable learned certificates. This analysis provides no justification for disabling the frozen emergency behavior or changing the selector to manufacture a survival gap.

**Cached versus runtime values.** The original 2e-6 m² equality check failed for a geometry-v2 compensated example: cached H was -0.02915365856 and full-controller H was -0.02904355526, a +0.00011010330 m² difference. Independent CPU isolation established:

- Initial physical states are identical, and all source/teacher model fields are identical. Contract/checkpoint actuator fields match; the checkpoint loader also verifies spec and actuator equality.
- Saved atlas ego paths scored against the actual runtime obstacle prediction yield -0.02915376425 m², within 1.06e-7 m² of the cached value. Runtime obstacle coordinates differ from the original geometry formula by at most 1.16e-7 m.
- Re-evaluating the same fixed library on CPU gives at most 8.73e-5 m positional difference from the saved GPU atlas. The CPU differentiated primal differs from the CPU ordinary primal by at most 1.70e-6 in full state and has the same maximum H to displayed precision.

These checks isolate a small compiled/backend rollout discrepancy, rather than a different obstacle prediction, initial state, or declared model. They do not definitively identify the exact GPU compiler transformation; no same-device claim is made from the CPU comparison. `qp-v3` retains each measured cache/runtime difference and independently checks the original development thresholds using the runtime evaluator. The largest observed absolute delta there is 0.000329104 m², well below the retained 0.025 m² adaptive threshold. Controller tolerances were not relaxed.

To reproduce the committed-versus-executed geometry analysis without using the GPU:

```bash
env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  .pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_case_mechanism.py \
  --closed-loop artifacts/da_plcbf/case-study-20260905/closed-loop-v3 \
  --atlas artifacts/da_plcbf/case-study-20260905/atlas-v1 \
  --output-dir /tmp/da-plcbf-mechanism-review
```

The present evidence supports a selected-case improvement in certified execution and maintained clearance. A headline claim of physical collision prevention, broad superiority, or paced online adaptation requires its own demonstrated outcome.
