# Recovery consistency and wind demonstration

The committed-backup command governor repairs the previously harmful learned-adaptive development flight while preserving success in the favorable fault and no-change cases. A continuation from the same saved pre-loss physical and learner state also succeeds with the governor. This is a bounded controller repair supported by development evidence; it is not a general safety guarantee or evidence of a new generalization result.

The new wind videos show one uninterrupted 43-second episode: 19 seconds of preflight followed by navigation, with actual wind changes in both stages. Learned frozen and learned adaptive both complete all four waypoints safely. The handcrafted baseline collides. This demonstration does **not** establish an adaptive advantage over learned frozen.

## What caused the harmful run to lose recovery?

The targeted diagnosis uses the original corrected harmful scene, with full physical and optimizer snapshots at every control boundary around the loss. It separates the earlier adaptation-induced command divergence from the later disappearance of a usable fresh candidate.

At 2.48 s, adaptation changes the selected certificate and produces a different feasible QP intervention, as reported in the preceding numerical iteration. At shared saved states, the current library has four eligible candidates at 3.20 s, two at 3.64 s, one at 3.68 s, and none at 3.72 s. All eligible branches in the audited interval pass their QP and direct held-command physical checks. There is no earlier selected-candidate failure with a different eligible branch still executable. By 3.60 s the onset snapshot has no eligible fresh candidate at the adaptive run's state either.

The publication at 3.72 s is not the sole deletion of a viable option: the previous publication also has no eligible candidate at that same state. The newest publication slightly improves the best hard value, approximately −0.0098 versus −0.0152. The normal controller enters emergency at 3.72 s, its short held collision check first fails at 4.28 s, and physical collision is detected at 4.45 s. Thus the loss involves the evolving physical trajectory and replanned control, not just a single bad parameter publication.

Controlled continuations use the complete saved physical state, absolute obstacle time, actual goal, current model, previous command, and complete learner/Adam state. Holding or restoring parameters here is a diagnostic intervention, not an online update gate.

| Saved start | Continue learning | Hold current snapshot | Restore onset snapshot |
|---|---|---|---|
| 2.48 s | Collision | Safe full mission | Safe full mission |
| 3.20 s | Collision | Collision | Safe full mission |
| 3.64 s | Collision | Collision | Collision |
| 3.68 s | Collision | Collision | Collision |

The selected fallback, replayed with its original parameters, anchor and elapsed phase, remains clear over the predicted 1.2-second horizon from all four starts. Maximum position discrepancy against the original prediction is only 12–31 micrometres. Restarting the same fallback also clears that horizon, but changes its path by 0.132–0.224 m. At 3.68 s the faithful prediction error is about 15.85 micrometres; restarting changes the path by about 0.196 m. These finite-horizon tests do not establish full mission completion. They show that an available predicted recovery can remain safe when executed faithfully even though subsequent QP execution loses recovery.

The supported explanation is consequently narrower than “the learned policy became unsafe”: adaptation changes feasible interventions and the ensuing states, while ordinary replanning can abandon a checked recovery. Early snapshot interventions change the eventual outcome; late snapshot restoration no longer saves the trajectory. The experiment does not identify a single neural weight or loss component as a uniquely sufficient cause.

## Independent audit of the QP regressions

An 80-digit Decimal reference solves the original and inward-tightened box-plus-policy-row QPs at the first legacy/corrected divergence for handcrafted and learned frozen in all three cases. All six matched-state comparisons have feasible solutions for both formulations. Recorded commands differ from their corresponding high-precision optima by approximately 4.1e-9–1.03e-8 N. The first old/new command differences, approximately 2.38e-7–2.98e-7 N, are consistent with the inward tightening.

This finds an altered feasible intervention rather than initial tightening-induced infeasibility or an acceptance inconsistency at those six boundaries. It does not prove every later QP is correct or that tightening preserves closed-loop behavior. The inward QP formulation and original physical tolerances are retained; baseline regressions remain visible below.

## The repair

`CommittedBackupController` wraps the existing PL-CBF controller. A proposed command is accepted only when its held execution passes the existing physical checks and its successor has a checked recovery using either the retained snapshot or a fresh skill. If that test fails, the governor executes a currently checked backup, preserving its parameter snapshot, original anchor, and elapsed maneuver phase. Fresh learner publications continue to enter the ordinary candidate bank; the governor does not reject learning updates, change the loss, or freeze the optimizer.

Checks cover finite states, collision clearance, operational constraints, motor effort and command bounds. A separate recorded execution mode identifies backup commands; these are not mislabeled as successful original QP/CBF-row decisions. The runtime stores the actual checked backup path and its certified time interval for audit and visualization. Warmup cannot publish an episode backup. The implementation rejects mismatched multi-bank libraries instead of indexing them as a single bank.

The first governor version exposed an implementation mistake: at the next boundary it demanded a new 1.2-second prediction even when the previous successor check promised only the remaining 1.16 seconds. In two saved examples, that extra 40 ms was obstructed while the previously checked tail remained clear. The final version tracks the promised deadline and rechecks that remaining tail under the current observations; it neither silently extends the deadline nor treats an expired tail as valid. The source and all 12 first-version flights are retained. In that version the onset-freeze no-change arm collided, the handcrafted gain arm missed its last waypoint, and handcrafted no-change slightly violated an operational bound. The final version fixes the onset-freeze regression but makes the two handcrafted cases collide; this is not a universal improvement.

The check remains finite-horizon and conditional on the supplied current model and obstacle predictions. There is no terminal invariant-set argument, guaranteed future wind/fault knowledge, or proof of indefinite recursive feasibility. When no checked recovery remains, the runtime explicitly records the missing plan and degraded control. The observed handcrafted failures demonstrate this limitation.

## Final fixed-case outcomes

All arms share the final governor, corrected learner numerics, inward QP and physical limits. These are the same three previously observed development cases.

| Case | Handcrafted frozen | Learned frozen | Learned adaptive | Learning frozen at fault onset |
|---|---|---|---|---|
| Previously harmful fault | Safe, 2/2 waypoints | Safe, 2/2 | Safe, 2/2 | Safe, 2/2 |
| Previously favorable fault | Collision at 2.575 s | Safe, 4/4 | Safe, 4/4 | Safe, 4/4 |
| No-change regression | Collision at 2.650 s | Safe, 4/4 | Safe, 4/4 | Safe, 4/4 |

Learned adaptive completes 349 finite credited online updates in the 14-second harmful flight and 599 in each 24-second navigation flight. Every control in all learned-frozen, learned-adaptive and onset-freeze final runs has a recorded checked backup. In the harmful adaptive run the governor first executes backup at 3.68 s and does so for seven controls. The favorable case no longer uniquely requires continued adaptation: learned frozen and onset-freeze also succeed with the governor.

Whole-flight prefixes across fresh JAX builds are not asserted to be bitwise identical. To establish a more direct repair test, a separate governor continuation starts at the actual saved harmful state at **3.68 s**. Assertions verify the exact original state, parameters, goal, nominal command and first underlying controller proposal. The governor changes the executed command from that common starting point while adaptation continues. It reaches both waypoints at 9.28 and 11.60 s and completes 14 s without collision or operational violation; its minimum operational margin is approximately 0.1774. The corresponding unmodified continuation collides at 4.45 s.

## Wind video and its interpretation

The fixed demonstration uses navigation seed 62104 and the previously used geometry, extended by a 19-second preflight. Obstacles hold their original time-zero positions during preflight, then follow the original analytic trajectories on `max(t−19,0)`. Positions are continuous at navigation start. This is a new, explicitly declared development demonstration, not a replacement for the fixed harmful-case test above.

| Absolute time | Stage/event | Wind (x, y, z), m/s |
|---|---|---|
| 0 s | Preflight | (0, 0, 0) |
| 3 s | Preflight wind on | (1.6, 0.8, 0) |
| 11 s | Preflight calm | (0, 0, 0) |
| 19 s | Navigation begins | (0, 0, 0) |
| 21 s | Actuator effectiveness fault | (0, 0, 0) |
| 23 s | Navigation wind | (1.2, −0.6, 0) |
| 29 s | Wind changes direction | (−1.0, 0.8, 0) |
| 37 s | Calm | (0, 0, 0) |
| 43 s | Episode ends | (0, 0, 0) |

Wind enters the physical body's aerodynamic model and is composed with actuator faults and recovery events. The shared F2 adapter receives the current observed wind, with no future wind schedule passed to controller or learner. The model cache keys on current/past fault and wind values, including configured observation delay. This is an oracle-current-wind comparison, not an unmeasured-wind estimator or a learning-only compensation claim.

Learned frozen and learned adaptive safely complete all four waypoints and 43 s. Learned frozen reaches its final waypoint at 32.60 s; learned adaptive reaches it at 36.00 s. Adaptive performs 1,074 finite credited updates, with 474 already used before navigation starts. Its optimizer and parameter state persist across the transition. Neither learned arm needs a backup execution in this particular wind flight, although both satisfy the backup-admission checks throughout. Handcrafted frozen collides at 21.77 s after 30 backup controls; its first control without a checked plan is at 21.64 s. That contact remains visible in the comparison.

Both movies replay the entire physical episode at 20 fps and 1600×900. Full decoding passes for both files; each contains 861 frames (43.05 seconds encoded, including the physical 43.00-second endpoint). Every frame’s replay data and all saved preview pixel hashes pass the audit, including wind/stage boundaries and contact-pose freezing. Cyan tracers and the wind arrow use the recorded physical wind field; the stage badge shows preflight/navigation and the current wind vector. Colored paths are current candidate predictions, white is the flown trail, and gold marks the actual checked backup being executed. Motor bars show recorded command and actual thrust. These visual predictions are not presented as flown future trajectories. The frozen/adaptive movie does not artificially widen the adaptive fan or splice in another simulation.

## Verification, failed attempts and evidence scope

The final regression suite passes 73 tests covering the governor, model-input cache, wind/fault composition, physical wind acceleration, obstacle-clock continuity, recording/replay, experiment scheduling and the retained numerical repairs. Separate earlier governor and wind suites are retained, not added into a larger unique test count. Actual and estimated wind vectors at every control, and dense recorded wind vectors, match the declared schedule. The physical hold audit passes for all 27 completed governor/demo flights. All use the aligned deterministic clock; the earlier asynchronous sensing/application/hold gap is still unresolved, and no uncontended runtime/deadline or deployment-safety claim is made.

The evidence preserves unsuccessful preparation and diagnostic attempts. Reconstructing the old learner from logged inputs failed an exact parameter hash at control 2 because a fresh fused computation differed slightly; a new fully instrumented harmful run therefore supplies the actual snapshots. It reproduces eligibility at every control and the same collision, but not every old parameter byte. The common-state analyses authenticate against that instrumented execution. The first continuation harness failed before flight; the second ran 20 branches but incorrectly started navigation early in its 12 normal-controller branches. Those 12 are explicitly invalidated. Its eight direct-fallback replays are unaffected. A third harness failed a host/device goal dtype assertion before flight. The fourth authenticates the actual goal and completes all 12 corrected normal-controller continuations. No invalid branch is included in the causal outcome table. A read-only analysis label initially included the checkpoint's starting library version in the preflight-update count; its amendment and corrected analysis are retained.

The [flight table](../artifacts/da_plcbf/recovery-interaction-20260907/v1/analysis-v2/flight_source.csv) includes all 12 initial-governor flights, 12 final-governor flights, and three wind flights. The [compact review package](../artifacts/da_plcbf/recovery-interaction-20260907/v1/publication-v1/README.md) retains reports, logs, executed source snapshots, test sources, flight summaries/bindings, causal records and video audit metadata. Large local trajectory/checkpoint arrays and movies are hash-indexed; the package is not a complete replay archive. Previous published evidence is unchanged.
