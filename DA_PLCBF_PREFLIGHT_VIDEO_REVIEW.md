# Preflight restored before the unchanged case study

This presentation-only revision follows `95a9225b6cf8e71897f93f57f34e2b9ad9b7357e`.
It restores the earlier wind-on/off hover demonstration before the current uncompensated
fixed/adaptive collision comparison. No numerical experiment is rerun or changed.

## Video sequence

The local video is `artifacts/da_plcbf/preflight-video-20260905/comparison-v1/preflight_then_case.mp4`.
It is 36.05 seconds, 1600×900, 20 fps, with 721 fully decoded frames.

| Video time | Content |
|---|---|
| 0–3 s | Calm hover; both panes begin with the same library. |
| 3–11 s | Wind `[1.6, 0.8, 0]` m/s; both hover, while the right fallback library learns. |
| 11–19 s | Wind removed; learning continues and the adaptive fan returns toward the original. |
| 19–22 s | Explicit transition into the original case from its original state/checkpoint. |
| 22–36.05 s | The existing paced collision-versus-survival video, unchanged. |

The preflight has no payload change or navigation. It uses frames 0–379 of the earlier
`hover-wind-payload-navigation` recording, ending before its 19 s payload event. New captions
explain adaptation and recovery; the old upcoming-payload footer is removed.

The preflight uses the same initial checkpoint and immutable nominal reference as the current
uncompensated case. Within the preflight, full Adam history persists through wind removal;
there is no event reset. Its learning is deterministic synchronized execution, distinct from
the subsequent case's measured paced learning. The transition makes clear that preflight
updates are not carried into the case, preserving the original comparison.

Recovery is partial rather than exact: adaptive same-state path RMSE rises to 11.30 cm just
after wind removal, then decreases to 0.594 cm at 18.8 s. Direction bins are then 16 fixed and
15 adaptive. At 10 s in wind, the counts are 7 fixed and 15 adaptive. These are existing
measured results, not new numerical trials.

## Verification

The case clip is concatenated by stream copy. All **281 original decoded case-frame hashes**
exactly match the final video's suffix from frame 440 (22 s). Source video SHA-256 hashes
remain unchanged. Both full source histories and the older case-study result directory remain
unchanged. Wind-on, wind-off recovery, transition, and collision frames were visually inspected.

`comparison-v1/VIDEO_REVIEW.json` records input/output hashes, the exact timeline, FFmpeg
commands, frame counts, and equivalence checks. `case_frame_md5.json` retains all case-frame
hashes. The independent preflight source audit records checkpoint/reference equality, original
analysis-input hashes, event timing, and measured recovery. The composition source is archived
under the new artifact root. Videos and extra inspection stills remain local; the publication
manifest identifies all inclusions and omissions.

The existing case study and its claim limits remain in
[DA_PLCBF_CLOSED_LOOP_SEARCH_REVIEW.md](DA_PLCBF_CLOSED_LOOP_SEARCH_REVIEW.md). In particular,
the stronger compensated fixed comparator still survives; this intro does not change that result.

## Reproduction

With the two original local MP4s available, run from the repository root with a fresh output path:

```bash
.pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_preflight_video.py --output NEW_VIDEO_DIRECTORY
```

The command performs no controller or learner execution. It checks the preflight event schedule,
excludes payload/navigation, generates explanatory captions and the transition, and verifies
all original case frames after composition. Source MP4 files are intentionally absent from Git.
