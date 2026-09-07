# Generated artifacts kept outside main

Videos, generated evidence archives, and run artifacts of 5 MB or larger are kept locally rather than tracked on `main`. Source code, compact reports and figures remain tracked. Runtime assets such as the drone mesh remain part of the repository.

The table records files excluded when publishing the actuator and hover work to `main`. Local copies are preserved in the original workspace. Historical commits on existing branches are unchanged; this cleanup does not purge old Git objects. The code changes are squashed so the diagnostics branch’s large archive commits are not added to `main` history. Links in historical reports to these generated products refer to local evidence and may not resolve in a fresh checkout. Regenerate the artifacts with their recorded benchmark drivers or restore a local copy.

| Local artifact | Bytes | Git blob ID |
|---|---:|---|
| `artifacts/da_plcbf/actuator-diagnostics-20260906/v1/publication-v1/review-only.tar.gz` | 103053169 | `4bb886e37b1105074a7c957ee800dd5d8827e7cc` |
| `artifacts/da_plcbf/actuator-study-20260906/v1/publication-v2/manifest.json` | 16456441 | `ae2b8e4ead8884868281fd7e6507ef9941252664` |
| `artifacts/da_plcbf/actuator-study-20260906/v1/publication-v2/review-only.tar.xz` | 67342784 | `3e91ebff49b96d8dbeb7fed4a55bece7732a01bd` |
| `artifacts/da_plcbf/case-study-20260905/publication/source_delta.tar.gz` | 89553 | `f4a1ad85d3509befd920e542eb6edffdf32c0cf5` |
| `artifacts/da_plcbf/closed-loop-search-20260905/confirmation-staggered-0000-v1/SOURCES_AT_EXECUTION.tar.gz` | 33509 | `de0cc1e482188ec2402a7facba215aaff4a442cc` |
| `artifacts/da_plcbf/closed-loop-search-20260905/paced-staggered-0000-v1/SOURCES_AT_EXECUTION.tar.gz` | 33738 | `a89566862bde7005b29c4644dace1a7a7cf194d6` |
| `artifacts/da_plcbf/closed-loop-search-20260905/publication/source_delta.tar.gz` | 132401 | `ca453bcb248d0b239c143c780fc13b6e6d03ea21` |
| `artifacts/da_plcbf/competent-revision-20260904/crossing-oracle-7/competent_comparison.npz` | 18365181 | `83c00c104a7b5c2cf7651712237377e8bcb1d24f` |
| `artifacts/da_plcbf/competent-revision-20260904/crossing-oracle-7/symmetric_probe_trajectories.npz` | 22375211 | `9b2321d20cc362de846d2dd5fd9c84946c67ecaa` |
| `artifacts/da_plcbf/competent-revision-20260904/payload-oracle-6/competent_comparison.npz` | 18711010 | `36500b83eee3819afa9afb485e6d4c2ec206652a` |
| `artifacts/da_plcbf/competent-revision-20260904/payload-oracle-6/symmetric_probe_trajectories.npz` | 22417857 | `b08eada6ec71889dac9c3cd2a1a5dca44adc1042` |
| `artifacts/da_plcbf/competent-revision-20260904/source_snapshot.tar.gz` | 170561 | `648bd6ca97f8f8ce5a4265c6e79b2faf0cdbd68a` |
| `artifacts/da_plcbf/competent-revision-20260904/unchanged-control-4/competent_comparison.npz` | 19185102 | `b028281c009503914189a21407c26b08a854964d` |
| `artifacts/da_plcbf/competent-revision-20260904/unchanged-control-4/symmetric_probe_trajectories.npz` | 22468178 | `03fe81931e5fbd8069f974de7c878e35d2e0b04a` |
| `artifacts/da_plcbf/competent-revision-20260904/wind-estimated-5/competent_comparison.npz` | 17987609 | `911b9db8e6c7f88fd76179568e8c7aa0e714dbca` |
| `artifacts/da_plcbf/competent-revision-20260904/wind-estimated-5/symmetric_probe_trajectories.npz` | 22270474 | `ac8f7d3b530b7b538700ab34d7dea4fc359b5779` |
| `artifacts/da_plcbf/competent-revision-20260904/wind-estimated-8/competent_comparison.npz` | 18451880 | `b79d40c0e3fdfd24c7dd1b656c19fbef5eb30657` |
| `artifacts/da_plcbf/competent-revision-20260904/wind-estimated-8/symmetric_probe_trajectories.npz` | 22313061 | `7eb5d3139a3347ea94ba3e354dffaf85a2c5e2cf` |
| `artifacts/da_plcbf/competent-revision-20260904/wind-oracle-0/competent_comparison.npz` | 17362961 | `e8abce931675460bda9e8b32a82d33e3ec9a51ba` |
| `artifacts/da_plcbf/competent-revision-20260904/wind-oracle-0/symmetric_probe_trajectories.npz` | 22316371 | `0665ff3ea8dc36be39200f731cf27ce881228957` |
| `artifacts/da_plcbf/competent-revision-20260904/wind-oracle-1/competent_comparison.npz` | 18422215 | `645496d5a7e15d6fbda3676bfb8a3422cfd147ee` |
| `artifacts/da_plcbf/competent-revision-20260904/wind-oracle-1/symmetric_probe_trajectories.npz` | 22311294 | `06b70f2f22c72dc3d87f68e64cd32904ee9ff754` |
| `artifacts/da_plcbf/competent-revision-20260904/wind-oracle-compensated-3/competent_comparison.npz` | 18454758 | `545d1b2aaec2d0f8ca015ed6da2377456e71a95e` |
| `artifacts/da_plcbf/competent-revision-20260904/wind-oracle-compensated-3/symmetric_probe_trajectories.npz` | 22285865 | `3cddfa5fc921e4114082e46c6280dbc678ad4902` |
| `artifacts/da_plcbf/competent-revision-20260904/wind-oracle-strong-2/competent_comparison.npz` | 18188121 | `3ab12542802ffd3d158ff54549e5859dc28c339e` |
| `artifacts/da_plcbf/competent-revision-20260904/wind-oracle-strong-2/symmetric_probe_trajectories.npz` | 22306712 | `2f7be97172e8447a40a1a20731d87760ae317dd1` |
| `artifacts/da_plcbf/corrected-online-wind-review-20260901-v5/online_constant_wind.npz` | 7673589 | `93149c6f121a605891ef0c27a8e4f4286021d6e6` |
| `artifacts/da_plcbf/gpu-online-review-all-v2-20260901/methods/da_plcbf_full/ballistic_ball/0/dashboard_evidence.npz` | 6596375 | `08e3e6efecc010a8ca5cf6b42295f0e20558ff24` |
| `artifacts/da_plcbf/gpu-online-review-all-v2-20260901/methods/da_plcbf_full/dynamics_change/0/dashboard_evidence.npz` | 6473401 | `176ac1b9db0fb9cfcd4062b80fbe4fe48c15c884` |
| `artifacts/da_plcbf/gpu-online-review-all-v2-20260901/methods/da_plcbf_full/interceptor_drone/0/dashboard_evidence.npz` | 6429423 | `b9ba44e754b8cf1c3d863db97adc1aac4a454edf` |
| `artifacts/da_plcbf/gpu-online-review-all-v2-20260901/methods/da_plcbf_full/static/0/dashboard_evidence.npz` | 6422812 | `e9a72de4b2ea90377321a8da1afb0562fdde6e68` |
| `artifacts/da_plcbf/hover-explanation-20260905/current-source/SOURCE.tar.gz` | 953316 | `978380c5462f487e7bd068b0dfe7dca5676acdb3` |
| `artifacts/da_plcbf/hover-return-20260907/v1/publication-v1/review-evidence.tar.gz` | 80882258 | `fff2370f4e30420e9835408eaba040ad550e78f0` |
| `artifacts/da_plcbf/navigation-revision-20260905/CAMPAIGN_OUTPUT_PATCH.tar.gz` | 39610 | `9936a7e809702af8ce0623add77cfcae28289b03` |
| `artifacts/da_plcbf/navigation-revision-20260905/CAMPAIGN_SOURCE.tar.gz` | 793390 | `e4063df6b14218f20281a49ea8b4ad0d142a0d11` |
| `artifacts/da_plcbf/navigation-revision-20260905/PACED2_SOURCE.tar.gz` | 605652 | `f2f5a8315d8e51460f4dd64a059b7bb851db6f99` |
| `artifacts/da_plcbf/navigation-revision-20260905/PACED3_SOURCE.tar.gz` | 608476 | `c83ce04406764ae0587b3c7417e99c5beaef4bf4` |
| `artifacts/da_plcbf/navigation-revision-20260905/PACED_SOURCE.tar.gz` | 605332 | `4800ac04c2c316330c704e0ab6824f9b87b11911` |
| `artifacts/da_plcbf/navigation-revision-20260905/current-source/SOURCE.tar.gz` | 908857 | `206b2cea4ba443c9dac3198c6a45b84860376804` |
| `artifacts/da_plcbf/navigation-revision-20260905/development-inputs/dense16_source_snapshot.tar.gz` | 538173 | `1eec872991eba17c322dceafa4ba0c766cb82489` |
| `artifacts/da_plcbf/navigation-revision-20260905/development-inputs/unchanged_source_snapshot.tar.gz` | 538174 | `03cc95665feab21c1a4fd438f162da484f47fac4` |
| `artifacts/da_plcbf/numerical-stabilization-20260906/v1/publication-v1/review-only.tar.gz` | 17036722 | `98708e16cbac1c9f18aa2ad5e0187df9495a91a7` |
| `artifacts/da_plcbf/recovery-interaction-20260907/v1/publication-v1/review-only.tar.gz` | 40788614 | `9d9ba869ec5e675f4f92d121266cf446cd027e91` |
| `artifacts/da_plcbf/revision-20260904/cold-start-encounter-2/online_constant_wind.npz` | 18142069 | `05219b1e29c0ddb95c26e6d2a952437218f90a99` |
| `artifacts/da_plcbf/revision-20260904/cold-start-pilot-1/online_constant_wind.npz` | 18160658 | `91e6408798796a68642f6c624b89c0283f9f9179` |
| `artifacts/da_plcbf/revision-20260904/cold-start-shared-feedforward-3/online_constant_wind.npz` | 18936993 | `456e62081df3c79b3d17b01c98762a89c1611d21` |
| `artifacts/da_plcbf/revision-20260904/wind-compensated-nominal-1/online_constant_wind.npz` | 15539241 | `1be010d40b57166e86aeef2b91a90a42dd736f96` |
| `artifacts/da_plcbf/revision-20260904/wind-pilot-0/online_constant_wind.npz` | 15188899 | `7cd64efa0d244f891df7686d0be19656047a817b` |
| `artifacts/da_plcbf/revision-20260904/wind-triggered-controlled-ablation/online_constant_wind.npz` | 18909172 | `ad60e4b581f715983876b029bdc8cadd955fcd67` |
| `artifacts/da_plcbf/three-panel-wind-20260907/v1/publication-v1/review-evidence.tar.gz` | 6946642 | `4f87e2f7e9f7696f358148798a730477ed202add` |
| `artifacts/da_plcbf/wind-learning-comparison-20260907/v1/publication-v1/review-evidence.tar.gz` | 32154403 | `4285217acae0bc8e7f5060bf07f0b7db1f3632d7` |
