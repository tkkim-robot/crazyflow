# Compact engineering review publication

The user requested review on `main` and then asked to remove generated videos from Git.
Videos remain on the originating workstation. Bulky numerical traces, the compiled JAX cache,
and preliminary camera images also remain local. The complete local revision is preserved on
the local `plcbf` branch; it is not the remote publication target.

This checkout contains source, reports, configuration, logs, compact checkpoints/pose arrays,
source archives and review figures. `REVIEW_PUBLICATION.json` identifies every included and
omitted file in the three newest evidence directories, including byte sizes and original
SHA-256 hashes. Nothing was deleted from the original local artifact directories.

This is an engineering review bundle: it is incomplete for full numerical replay, unsealed
for scientific-claim purposes, and unsuitable as independent evidence of general safety or
real-time performance. It remains absent from the historical scientific `INDEX.md`.

The original `ARTIFACT_SHA256.json` files and source snapshots remain unchanged. They describe
the complete original folders, including files intentionally absent from Git. Run the whole-
directory validator only on a complete local copy. To check the published subset from the
repository root:

```python
import hashlib
import json
from pathlib import Path

record = json.loads(Path("artifacts/da_plcbf/REVIEW_PUBLICATION.json").read_text())
for name, expected in record["files"].items():
    if expected["included"]:
        data = Path(name).read_bytes()
        assert len(data) == expected["bytes"], name
        assert hashlib.sha256(data).hexdigest() == expected["sha256"], name
```

Generated DA-PLCBF videos are absent from the current `main` tree. Some were uploaded before
the user's correction and remain in earlier Git history; this cleanup does not force-rewrite
shared history. Historical Markdown/video-provenance references identify local outputs and do
not imply that an MP4 is downloadable from the current checkout.

Start with [the current review](../../DA_PLCBF_HOVER_REVIEW.md) and
[handoff](../../HANDOFF_DA_PLCBF.md). The underlying numerical findings and their limitations
are unchanged by this publication cleanup.
