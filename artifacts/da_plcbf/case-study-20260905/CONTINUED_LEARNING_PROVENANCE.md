# Continued-learning provenance reproduction

Both selected confirmation continued-learning branches were rerun independently from the exact saved time-4.00 s atlas learner (version 760). Original confirmation protocols, control rows, and outcomes were preserved. Only the continued-learning branches were recomputed; causal ablations and neighbors were not rerun.

| Case | Numeric history | Discrete history | Completed finite updates | Initial / final / last executed version |
| --- | --- | --- | --- | --- |
| uncompensated | Exactly equal, maximum difference 0 | Exactly equal | 75 | 760 / 835 / 834 |
| compensated | Exactly equal, maximum difference 0 | Exactly equal | 75 | 760 / 835 / 834 |

The comparison includes every dense physical state and timestamp, recorded action, hard and smooth value, executed dual, executed flag, published version and availability clock, selected policy, execution mode, QP validity, degraded status, eligibility mask, and finite-update decision. There were 75 executed commands at 0.04 s intervals, from 4.00 through 6.96 s, plus an unexecuted terminal diagnostic at 7.00 s.

Each new control row records the complete published persistent-state hash and separate parameter, previous-parameter, and Adam-state hashes. Every completed update records its training-state/model hashes, before/after persistent identities, finite-update decision, deterministic next-boundary publication time, and synchronized wall service clocks. All 75 after-state hashes link exactly to the next published row, and every measured update completion precedes the next measured controller start. Final version 835 is the final available checkpoint; the last executed command used version 834.

Each `continued-provenance-v2/learner/` retains complete `initial_checkpoint` and `final_checkpoint` NPZ/manifests, the bound immutable `nominal_reference` NPZ/manifest, and `publication_ledger.json`. `CHECKPOINT_BINDING_AUDIT.json` confirms both checkpoints restore through the reference-aware learner loader with the exact published full persistent-state identities and authenticated reference NPZ/manifest hashes. Recorded source/input hashes still match.

This is deterministic continuation with measured service clocks, not a paced deployment or deadline-availability result. Original runs lacked per-update hashes, so equality is established for their saved observable histories, not for nonexistent original optimizer hashes. The separate paced-runtime experiment remains the source for wall-clock deployment claims.

Evidence for each family is in:

- `confirmation-uncompensated-v1/continued-provenance-v2/REPRODUCTION_COMPARISON.json`
- `confirmation-compensated-v1/continued-provenance-v2/REPRODUCTION_COMPARISON.json`
- Each directory also contains `protocol.json`, `summary.json`, `controls.json`, `dense_states.npz`, `CHECKPOINT_BINDING_AUDIT.json`, and the complete learner files above.

Regression validation: `CONTINUATION_PROVENANCE_TESTS.txt` records 12 passing focused CPU tests in 9.90 s. The parameterized continuation test exercises rejected then accepted updates with and without provenance, checks that rejection does not refresh availability or change hashes, checks accepted parameter/Adam/previous-parameter identity, and verifies the terminal diagnostic is unexecuted. Ruff passes the modified attribution, confirmation, and provenance-test sources.

Reproduction (the output directory must not exist):

```bash
env PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/tmp/crazyflow-navigation-jax-cache .pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_case_confirmation.py --continue-provenance-only artifacts/da_plcbf/case-study-20260905/confirmation-uncompensated-v1 --output-dir /tmp/continued-uncompensated-reproduction --device gpu
env PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/tmp/crazyflow-navigation-jax-cache .pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_case_confirmation.py --continue-provenance-only artifacts/da_plcbf/case-study-20260905/confirmation-compensated-v1 --output-dir /tmp/continued-compensated-reproduction --device gpu
```
