# Minor presentation polish — 2026-09-06

Accepted numerical case: 95a9225b6cf8e71897f93f57f34e2b9ad9b7357e.
Previous preflight presentation: 5c72f06eee735c55cc8abb4a73cef85cbfd4c54c.

The stopped title card is removed. The moving preflight gives a brief comparison-scope
label and an explicit original-state/library restart disclosure, then cuts directly into
the unchanged case at 19 s. The full video is 33.05 s / 661 frames; all 281 case frames match.
No controller, learner, dynamics, constraints, checkpoints, timing measurements, or numerical
results are changed or rerun. The stronger compensated fixed baseline still survives.

Modified existing files, exactly:
- benchmark/da_plcbf_preflight_video.py
- DA_PLCBF_CLOSED_LOOP_SEARCH_REVIEW.md
- DA_PLCBF_PREFLIGHT_VIDEO_REVIEW.md
- HANDOFF_DA_PLCBF.md

New files in this directory retain composition source/captions, input/output hashes,
frame-equality checks, representative stills, and an independent full-decode/artifact audit.
The publication manifest enumerates every included and local-only file. All generated MP4s
and extra inspection stills remain local. Earlier files and manifests are unchanged.

Paced-v2 adaptive clearance from the actual modeled XML collider is 0.1734270816 m;
remaining clearance beyond the requested body-origin safety shell is 0.0062918289 m.
Deterministic values remain bound to their separate original run. Physical separation must
not be relabeled as additional separation beyond the safety buffer.

Checks: Ruff/format/whitespace passed; full ffmpeg -xerror decode passed; all 281 case-frame
SHA256 hashes match; no inserted pause; 644 previous artifact files and 7 original preflight
telemetry/reference inputs retain their original hashes. This is presentation evidence,
not a new numerical safety trial or scientific campaign.
