# DA-PLCBF artifact policy

Scientific run directories are intentionally ignored by git because traces, checkpoints, plots,
and videos are bulk products. Every complete run must validate against its own `manifest.json` and
`SHA256SUMS` before it is used as evidence. The manifest binds the configuration, provenance,
seeds, semantic scenario-tape mappings and digests, traces, metrics, timing samples, plots, and
videos.

Scenario tapes are one immutable v3 NPZ per paired fold, not a fictitious batched tape. Version 3
binds the predeclared ballistic encounter stratum, time-to-impact bin, bounded construction
attempts, and intended/realized continuous closest-approach metadata into the tape digest.
It also binds acceleration and realized-motor-force observation-noise sequences from two
independent named RNG streams, so paired estimators consume byte-identical exogenous noise rather
than sampling it during execution.
`seeds.json` explicitly maps every condition/fold to its path and semantic digest. Use
`scenario_tapes/<condition>/<fold>.npz` for condition-specific generation. A shared
`scenario_tapes/<fold>.npz` may be mapped to several conditions only when the path and digest are
identical. Every method for a condition/fold must bind its trace to that exact digest.

`INDEX.md` is the compact committed evidence index. A row may be added only after the run validator,
replay check, numerical review, and visual review pass. Each row must include the immutable run ID,
manifest SHA-256, `SHA256SUMS` SHA-256, storage location, exact reproduction command, and review
status. An empty index means that no run is yet claimed as reviewed evidence.

The JSON and CSV files below ignored run directories are not expected to appear in git. This policy
is deliberate: evidence is transferred or archived as a complete content-addressed directory, not
as selected loose files.

An explicitly requested, branch-local **engineering review bundle** is the only exception. Such a
bundle must carry a `REVIEW_ONLY.md`, remain absent from `INDEX.md`, identify every omitted artifact,
and state that it is unsealed, incomplete for replay/claim purposes, and unsuitable for scientific
claims. This exception exists only to make compact reports, traces, renderer evidence, and videos
available to a human reviewer; it does not relax final-run validation or indexing requirements.

Run video rendering and replay validation in a fresh offline process, after the numeric control run
has exited. Besides keeping rendering out of reported control latency, this avoids forking the
FFmpeg encoder from a multithreaded JAX process. Render-marked tests are therefore run separately:

```bash
pytest -m render tests/unit/safety/da_plcbf/test_dashboard.py \
  tests/unit/safety/da_plcbf/test_artifact_smoke.py
```
