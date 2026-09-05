# Compact persistent-wind case-study evidence

Read the repository's `DA_PLCBF_SAFETY_CASE_REVIEW.md` first. This is an engineering case study,
not a sealed broad safety campaign. Selected controlled branches show clearance/certificate
benefits; all physical geometries remain separated, and the continuous scene does not show
an adaptive safety advantage. The paced scene demonstrates actual completed online updates.

`publication/PUBLICATION_MANIFEST.json` lists every original local file, size, SHA-256,
publication decision, and reason. The manifest itself is excluded from its own hash inventory.
Generated videos, superseded render trials, profiling intermediate checkpoints, and large
rollout tensors stay local. No older artifact is deleted or rewritten.

The published package retains the complete 12,288-candidate ledger in
`publication/all_candidates.jsonl.gz`, the per-policy geometry matrices, frozen protocols,
configs, prefix/snapshot/reference data, all full-QP screens, selected and neighborhood
controls/dense states, continuation audits, compute measurements, and final still figures.
Each compressed ledger row adds its family name; all source row contents are preserved.
`publication/DERIVATION.json` binds its three source ledgers and the exact compact control traces.

`compact_control_trace.npz` removes the large hypothetical rollout tensor while retaining
time, observed full state, actual wrench, policy values/selection, QP/fallback/emergency mode,
actuator/held diagnostics, versions, validity masks, and available service telemetry. The
original source NPZ hashes remain in the derivation record. Dense geometric audits use the
separately retained integration states and analytic world, not only rendered positions.

`publication/source_delta.tar.gz` contains changed/new source and review files over the base
commit identified in `publication/SOURCE.json`. Recorded pre-unroll source/numerical captures
remain separate. A newly generated atlas under the revised compiler graph is not claimed to
be byte-identical to the historical saved optimizer continuation.

Local videos, excluded from Git:

- `videos/controlled-encounter/comparison-v2.mp4`: controlled branch, adapted snapshot held fixed.
- `videos/paced-continuous/comparison-v2.mp4`: continuous scene with actual online publications.

The earlier `comparison.mp4` files are explicitly superseded. Final v2 videos have full decoder,
scene-panel, frame-inspection, and source-hash records in `videos/VIDEO_REVIEW.json`.

This compact checkout does not contain the original bulky renderer inputs or every profiling
intermediate checkpoint. Reproduce from the saved world/checkpoints using the documented code,
or use the retained full local files for exact original rendering. Do not interpret a missing
large source tensor as an empty or successful run.

Verify every included file from the repository root:

```python
import hashlib, json
from pathlib import Path
root = Path("artifacts/da_plcbf/case-study-20260905")
manifest = json.loads((root / "publication/PUBLICATION_MANIFEST.json").read_text())
for entry in manifest["files"]:
    if entry["included"]:
        data = (root / entry["path"]).read_bytes()
        assert len(data) == entry["bytes"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
```
