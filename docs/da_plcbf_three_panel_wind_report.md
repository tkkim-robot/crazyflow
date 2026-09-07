# Three-panel wind replay and fallback-bias diagnosis

The video now places **handcrafted frozen, learned frozen, and learned adaptive** from left to right in one synchronized 2400×900, 20 fps movie. All panels cover the same 43-second physical clock, including preflight, navigation and the recorded wind changes. Every motor display uses the same 0–0.25 N scale.

The learned panels reuse the decoded viewports from the verified previous videos. Their flight states, policy predictions, wind and camera views are unchanged; headers and telemetry are recomposed for the three-panel layout. The handcrafted panel also reuses its original viewport until the physical contact handoff. Afterward it displays a separately simulated MuJoCo contact continuation with zero rotor thrust. That segment is explicitly labeled “Impact → motors off · MuJoCo contact replay · predictions stopped.” No fallback predictions or controller actions are invented after termination.

## Why the preflight fans look more similar

The main cause is a change in the fallback controller's wind compensation, which should have been more prominent when presenting the actuator-study video.

| Setting | Earlier wind preflight | Current actuator wind flight |
|---|---|---|
| Wind at 3 s | (1.6, 0.8, 0) m/s | (1.6, 0.8, 0) m/s |
| Wind removed | 11 s | 11 s |
| Explicit fallback wind compensation | Off in both methods (`matched_uncompensated`) | On in every method (`model_compensation=True`) |
| Nominal task controller compensation | On | On |
| Dynamics | Earlier body-state study | Actuator-state study with motor lag |

The old preflight metadata explicitly records `prefix=False`, `post_event=False`, and `checkpoint=False` for fallback compensation. The current shared acceleration-to-motor adapter adds the model-based external-force cancellation before allocating motor commands. It does this for frozen and adaptive fallbacks, as well as nominal and emergency control. All methods receive the same current observed wind. Consequently, frozen fallback paths already resist much of the wind displacement; learning has less uncompensated wind error to visibly correct. The wind strength was not reduced.

A controlled rollout probe fixes the full physical/motor state at the recorded 2.96-second frozen boundary, keeps the exact same learned checkpoint parameters, actuator model and 1.2-second horizon, and compares calm versus wind with the compensation setting on or off. It performs no new flight or retraining.

| Shared fallback compensation | RMS endpoint displacement caused by wind | RMS displacement across the entire path |
|---|---:|---:|
| On, as in the current video | 11.69 cm | 8.16 cm |
| Off, diagnostic ablation | 37.57 cm | 19.33 cm |

This isolates a **69% reduction in endpoint displacement** from compensation in the current model. It does not assert that removing compensation reproduces the entire older experiment, whose dynamics and learning setup also differ.

The actual current video paths are similar rather than copied or identical. Comparing corresponding fallback paths after subtracting each path's own starting position gives adaptive/frozen RMS differences of 0.30 cm at 3.04 s, 4.38 cm at 4 s, 6.39 cm at 7 s, and 6.10 cm just before wind removal. By 18.96 s, after calm returns, the difference is 0.53 cm. These are differences between recorded fans at each method's own state, not an isolation of neural-parameter effects. The adaptive arm has used 175 online versions by 7 s and 474 before navigation; the frozen arm uses no online updates.

The current video therefore studies adaptation **on top of shared wind compensation**. The earlier preflight exposed the larger wind error left for learning when fallback compensation was disabled in both methods. Disabling compensation in only the frozen comparator would create an unfair comparison. No controller or learner setting was changed to make this three-panel presentation more dramatic.

## Contact continuation

The independently audited geometric intersection is at 21.766753 s. The existing swept-contact transfer starts 1 ms earlier, at 21.765753 s, to avoid initializing the rigid-body contact model inside the obstacle. The recorded pose and linear/angular velocity are transferred without an invented impact impulse. MuJoCo then measures 18 obstacle-contact steps, reaches first ground contact at 22.253753 s, and records 19,953 ground-contact steps over the remaining episode. There are no MuJoCo warning counts. Obstacles and wind continue on the original absolute clock.

This is an explicit crash-presentation assumption: rotor thrust is set to zero at the handoff. It is not a claim that the original controller issued a motor-shutdown command or that the pre-contact actuator experiment simulated post-impact behavior. The original contact-stopped flight arrays and outcomes remain unchanged. The contact replay duration limit was extended from 20 to 60 seconds to support the required 21.234-second continuation; integration and contact equations were unchanged.

## Evidence

The source videos and full contact model/trajectory are hash-bound by the new video's binding and manifest. Frame records separate original-flight replay from the contact continuation and verify that the two learned viewport interiors match their original decoded source pixels before final encoding. The [wind probe](../artifacts/da_plcbf/three-panel-wind-20260907/v1/wind-bias-v2/report.json) includes exact metrics and old/current configuration evidence. The [contact record](../artifacts/da_plcbf/three-panel-wind-20260907/v1/contact-v1/contact_replay.json) includes the trigger, measured contacts, motor-off assumption and source hashes. A preliminary probe failed on a checkpoint-field name before producing rollouts; its source and log are retained separately.

The existing contact and actuator-video regression suites pass 27 tests. This work changes the presentation and adds a controlled diagnostic; it does not add a new adaptive-versus-frozen flight result. Both learned methods still complete the four-waypoint wind mission safely, and learned frozen still reaches its last waypoint sooner in this particular demonstration.

Final video validation passes: 861 frames at 2400×900 and 20 fps, full decoding, 425 contact-suffix frames matched to the separate simulation, unchanged learned viewport interiors before encoding, and every saved preview hash.
