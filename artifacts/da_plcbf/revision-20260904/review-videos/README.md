# Revised DA-PLCBF review videos

These four files are MuJoCo-rendered replays of recorded differentiable Version-A trajectories.
They use 1600×900 H.264 video at 20 fps. Each wind video contains one explicitly labeled two-second
probe pause; the displayed simulation timestamp remains fixed during that pause.

| Watch order | Video | Duration | What it demonstrates |
|---|---|---:|---|
| 1 | [Wind-triggered adaptation](wind_triggered_adaptation.mp4) | 18.05 s | Methods agree before the wind step; only the adaptive library then receives online updates. The frozen drone collides while adaptation preserves the clearance shell and reaches the goal. |
| 2 | [Isolated static PL-CBF](static_plcbf_only.mp4) | 8.05 s | Nominal collision versus fixed PL-CBF avoidance, with analytic obstacle HOCBF disabled. |
| 3 | [Learning from startup](online_skill_construction.mp4) | 18.05 s | Online construction from a braking library, with adaptation active from startup. |
| 4 | [Learning versus wind compensation](learned_vs_wind_compensation.mp4) | 18.05 s | Explicit model compensation keeps the physical body clear in this run but violates the inflated shell; learned skills preserve the shell. |

The primary wind-triggered video pauses at simulation **5.60 s**. At the same recorded adaptive
state and point model, library H is frozen **-0.282512**, compensated **-0.080393**, and adaptive
**+0.072078**, with **0 / 0 / 4** collision-clear fallback policies. Those are counterfactual
library comparisons. Each main panel separately shows its actual executed trajectory, current
H, command status, and collision history. The startup comparisons pause at simulation **5.40 s**.

The renderer preserves the original source traces. It does not run the controller or add contact
physics. Collision values cover the finite prediction horizon under the estimated model;
instantaneous operational constraints and the explicit 0.05 m drone collision radius are reported
separately. A positive dual displayed alongside a rejected QP belongs to that proposal; the command
status identifies whether the QP or a fallback actually executed.

Each MP4 has six decoded full-resolution PNG timeline frames, a contact sheet, and a visual-review
JSON with inspection scope and observations. The compensation video also has a seventh frame at its closest physical approach. All frames of all four videos decoded without errors.
Visual inspection sampled the initial state, wind onset, recorded coverage probe, encounter, and
terminal goal recovery; it is not represented as inspection of every individual frame.

- [Static contact sheet](static_plcbf_only_contact_sheet.jpg)
- [Wind-triggered contact sheet](wind_triggered_adaptation_contact_sheet.jpg)
- [Startup-learning contact sheet](online_skill_construction_contact_sheet.jpg)
- [Wind-compensation contact sheet](learned_vs_wind_compensation_contact_sheet.jpg)
- [Media/source identities and render configurations](video_manifest.json)
- [MP4 checksums](SHA256SUMS)

Source numerical results remain in the [static gate](../review-static/),
[startup-learning comparison](../cold-start-shared-feedforward-3/), and
[wind-triggered comparison](../wind-triggered-controlled-ablation/) directories.
The [repository revision review](../../../../DA_PLCBF_REVISION_REVIEW.md) contains the research
interpretation and limits.

Markdown links in this README are relative to this directory. Path strings in `video_manifest.json`
and `*_visual_review.json` are relative to the repository root (for example,
`artifacts/da_plcbf/revision-20260904/review-videos/static_plcbf_only.mp4`); resolve them against the
root of the checkout, rather than the JSON file's directory. These paths do not depend on a local
username or checkout location.
