# Wind learning comparison without direct wind feedforward

This reruns the three controllers in the previous 43-second wind scene, with direct wind cancellation disabled in **every** arm. The new synchronized video places handcrafted frozen, learned frozen, and learned adaptive from left to right. The earlier compensated video and its results remain available as a separate baseline.

The change is deliberately limited to the wind term in the shared acceleration-to-motor adapter. Ordinary state feedback, calm-air drag compensation, actuator lag handling, and motor allocation remain active. Physics, safety predictions and the learner still receive the actual current wind. This is therefore adaptation without direct wind feedforward, **not** adaptation with unknown wind. No future wind schedule is supplied to the controller.

`ActuatorSkillConfig.wind_feedforward=False` zeros the wind only in a local copy of the compensation model. It does not modify the physical or prediction model. The teacher's actor contract receives the same setting. Default `True` preserves existing behavior and existing reference fingerprints; opting out is included in the fingerprint. The derived checkpoints preserve all parameter, previous-parameter and optimizer leaves exactly. The learned frozen and adaptive arms start from the same complete learner state, and only the adaptive arm updates online. The learner objective, safety checks, governor, geometry, wind schedule and actuator fault are unchanged.

## Observed outcome

| Controller | Recorded outcome | Waypoints by 43 s | Online updates used |
|---|---|---:|---:|
| Handcrafted frozen | Collider intersection at 21.7664 s; flight ends at 21.77 s | 0/4 | 0 |
| Learned frozen | Full duration, no collision | 3/4 | 0 |
| Learned adaptive | Full duration, no collision | 3/4 | 1,074 |

The learned waypoint arrival times are 22.40, 29.96 and 38.76 s for frozen, and 22.40, 30.00 and 38.68 s for adaptive. Neither finishes the fourth waypoint within the fixed duration. The previous compensated learned flights completed all four waypoints. This run does **not** demonstrate an adaptive task-success advantage, and the time horizon was not extended to convert either failure into a success.

Every learned-arm command has a checked backup and every applied hold is covered by the declared timing check. Handcrafted has four controls without a checked backup before its collision. The committed backup executes at 30 handcrafted, 11 learned-frozen and 40 adaptive control decisions. All methods satisfy the recorded operational limits; that does not imply collision avoidance or mission completion. This is one fixed development demonstration, not fresh validation or a success-rate estimate.

## What changes visually

The colored fallback paths separate more during the preflight wind. The figures below compare corresponding recorded paths after subtracting each path's own starting position; they do not isolate the effect of neural parameters from the difference in flight states.

| Physical time | Adaptive/frozen path RMS difference | Adaptive online updates used |
|---|---:|---:|
| 2.96 s, before wind | 0.11 cm | 74 |
| 4.00 s, wind active | 17.29 cm | 100 |
| 7.00 s, wind active | 20.20 cm | 175 |
| 10.96 s, just before wind removal | 21.50 cm | 274 |
| 18.96 s, calm before navigation | 2.36 cm | 474 |

At 7 s the earlier compensated comparison differed by 6.39 cm RMS. Removing direct cancellation leaves more wind-related error for online learning to affect. A visibly different fallback fan is not by itself evidence of better mission performance.

## Video and contact presentation

All panels render new recorded flights at 20 fps, on the same 0–43 s clock, with a shared 0–0.25 N motor scale. Headers explicitly state that direct wind feedforward is off and state feedback remains on. The cyan wind cues follow the recorded schedule. Colored paths are the stored current predictions, white is the flown trail, and gold indicates execution of a checked backup. The display includes waypoint progress and online update counts.

The handcrafted contact suffix is a separate motor-off MuJoCo simulation, explicitly labeled on screen. Its handoff is 21.765402 s, 1 ms before swept geometric contact, with the recorded pose and velocities transferred directly. MuJoCo records 18 obstacle-contact steps, first ground contact at 22.254402 s, and 19,945 ground-contact steps; no warning counts occur. Thrust and commands are zero and fallback predictions stop throughout this suffix. This is a crash-presentation assumption, not a claim that the original controller commanded motor shutdown. Wind and obstacles continue on the original physical clock.

## Reproduction and evidence

Run `python -m benchmark.da_plcbf_wind_learning_comparison` in the GPU environment to create the three new flights. The output directory is exclusive so rerunning requires a new version directory. Render each method with `python -m benchmark.da_plcbf_wind_learning_video PD_F` (and `F2`, `A_BAL`) using CPU JAX and OSMesa, then run its `combine` stage. The analysis and audit drivers are `benchmark.da_plcbf_wind_learning_analysis` and `benchmark.da_plcbf_wind_learning_audit`.

The [analysis report](../artifacts/da_plcbf/wind-learning-comparison-20260907/v1/analysis-v1/report.json) binds each flight to its source, verifies the exact prior scene and actual/estimated wind, and records outcomes and fan metrics. The [video validation](../artifacts/da_plcbf/wind-learning-comparison-20260907/v1/video-v1/validation.json) verifies 861 frames at 2400×900 and 20 fps, full decoding, every original-flight sample against the saved state/command data, contact poses against the separate simulation, and panel preview hashes. The movie contains the inclusive 43-second endpoint, giving an encoded duration of 43.05 seconds.

The [review archive](../artifacts/da_plcbf/wind-learning-comparison-20260907/v1/publication-v1/manifest.json) includes source snapshots, derived and critical learner checkpoints, physical trajectories, control decisions, sampled prediction fans, contact dynamics, reports and audits. Full prediction arrays, verbose events and media remain locally available and hash-indexed. Previous artifacts are not overwritten.

Validation: 25 wind/learner tests and 52 actuator-video, contact-replay and diagnostic-runtime tests pass. The new wind test verifies bitwise identical calm commands, no direct command response to changing wind at fixed state when disabled, and a continuing physical acceleration response to actual wind. An initial regression command named a nonexistent test file and ran no tests; its log is retained alongside the corrected successful run.
