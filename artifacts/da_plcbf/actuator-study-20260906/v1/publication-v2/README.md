# Actuator study reviewer guide

The selected video illustrates one recorded deterministic/P0 F2/A case. Broad comparisons come from the sealed main analysis; timed adaptation and transfer claims require their separately declared experiments. Candidate-case transfer results remain separate from validation sensitivity curves. The collector does not decide the scientific verdict; use the completed report's evidence and qualifications.

Sealed main: 384/384 method episodes. Numerical source revision: `377f59f942cbe9a0eaf313b4c8bce40ceeb8b003`. Final reporting revision: `cd5dc88baead4e1bb5286a3c2ad3635da0ce1245` (caller supplied; individual source bytes are hashed).

Links resolve in the extracted archive layout. MP4 files remain local-only.

- [Research report](<../docs/da_plcbf_actuator_report.md>)
- [Technical note](<../docs/da_plcbf_actuator_theory.md>)
- [Sealed protocol](<../artifacts/da_plcbf/actuator-study-20260906/v1/main384-protocol-sealed-v1.json>) · [Main analysis](<../artifacts/da_plcbf/actuator-study-20260906/v1/main384-sealed-analysis-v1/analysis.json>)
- [Main campaign and all method/seed summaries](<../artifacts/da_plcbf/actuator-study-20260906/v1/main384-sealed-results-v1/campaign_result.json>)
- Main safety: [PNG](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/01_main_safety.png>) · [PDF](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/01_main_safety.pdf>) · [Data](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/01_main_safety.csv>)
- Fixed bank recovery: [PNG](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/02_fixed_bank_recovery.png>) · [PDF](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/02_fixed_bank_recovery.pdf>) · [Data](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/02_fixed_bank_recovery.csv>)
- Safety and execution cost: [PNG](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/03_safety_and_execution_cost.png>) · [PDF](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/03_safety_and_execution_cost.pdf>) · [Data](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/03_safety_and_execution_cost.csv>)
- Library and retention ablations: [PNG](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/04_library_and_retention_ablations.png>) · [PDF](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/04_library_and_retention_ablations.pdf>) · [Data](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/04_library_and_retention_ablations.csv>)
- One factor mismatch: [PNG](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/05_one_factor_mismatch.png>) · [PDF](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/05_one_factor_mismatch.pdf>) · [Data](<../artifacts/da_plcbf/actuator-study-20260906/v1/publication-figures-v2/05_one_factor_mismatch.csv>)

Selected development-case timing/transfer evidence (descriptive):

- [candidate-1002-paced-v1](<../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-paced-v1/campaign_result.json>)
- [candidate-1002-delayed-v1](<../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-delayed-v1/campaign_result.json>)
- [candidate-1002-p1-v1](<../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-p1-v1/campaign_result.json>)
- [candidate-1002-p2-v1](<../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-p2-v1/campaign_result.json>)
- [candidate-1002-motor-noise-v1](<../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-motor-noise-v1/campaign_result.json>)

Video: `artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-video-v2/comparison.mp4` (5260701 bytes, omitted). SHA-256: `78c0156cc2158a4fc38e58b8cbbb8e73351a293456e6ce8c18553076a7359096`. [Selected poster](<../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-video-v2/frame-000150.png>) · [Frame audit](<../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-video-v2/frame_audit.jsonl>) · [Render binding](<../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-video-v2/render_binding.json>).

Separate companion (paced/P0): `artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-paced-video-v2/comparison.mp4` (1961296 bytes, omitted). SHA-256: `fd5c4e37759daf44ddc7a307858e6bcca03c9e46bb34d9bf0188be89a658d801`. [Poster](<../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-paced-video-v2/frame-000138.png>) · [Frame audit](<../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-paced-video-v2/frame_audit.jsonl>) · [Render binding](<../artifacts/da_plcbf/actuator-study-20260906/v1/candidate-1002-paced-video-v2/render_binding.json>).

Study reports, campaign summaries and tuning metadata are included without filtering by outcome, including negative results, interrupted attempts, tuning decisions and numerical-cache sensitivity. Most raw rollout arrays, event streams and optimized caches remain local-only. This compact bundle cannot support complete physical replay on its own; see [inclusion and verification scope](REVIEW_ONLY.md).
