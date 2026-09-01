# GPU online review bundle — not claim-grade evidence

This directory is a compact engineering-review copy of the one-fold RTX 4090 development run
described in `HANDOFF_DA_PLCBF.md`. It exists so reviewers of branch `plcbf` can inspect the four
videos, contact sheets, keyframes, aggregate reports, scenario tapes, traces, events, timings, and
renderer-bound dashboard evidence directly from Git.

It is intentionally **unsealed**. It has no formal visual-review records, `manifest.json`, or
`SHA256SUMS`, is absent from `artifacts/da_plcbf/INDEX.md`, and must not support a scientific or
safety-superiority claim.

To keep the branch reviewable, this committed bundle omits:

- `methods/da_plcbf_full/*/0/adaptation_evidence.npz` (four large raw adaptation-evidence tensors);
- `methods/nominal_only/*/0/dashboard_evidence.npz` (four unused comparison renderer sidecars).

The complete local run directory contained those files when the numerical and visual checks in the
handoff were performed. Their omission means this Git copy is not independently replay-complete.
Regenerate a complete run with the command recorded in the handoff before attempting validation or
claim-grade sealing.
