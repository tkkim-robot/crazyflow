# Test tiers

The default Pixi test task intentionally runs a focused core tier:

```bash
pixi run -e tests tests
```

`core-tests.txt` is a representative development subset spanning simulator foundations and the
DA-PLCBF actor, rollout, filtering, estimation, campaign evidence, snapshot, and runtime paths. The
default tier covers 33 of the repository's 80 test files: 32 are listed in the response file and
Version B runs in an isolated process. The remaining 47 files stay available but are inactive in
the normal local loop. They include broader Crazyflow coverage, Phase-1 reference mathematics,
additional BPTT/loss/QP checks, studies and ablations, rendering, report generation, artifact
tamper detection, and longer evidence workflows. Only the full tier is authoritative for those
contracts.

No test is deleted, renamed, or permanently skipped. Run the complete non-render suite with:

```bash
pixi run -e tests tests-full
```

Both tasks execute `test_version_b_runtime.py` in a fresh second Python process. During the 2026-08
handoff, the file passed independently while a combined long-lived JAX process failed after
hundreds of unrelated compilations. Process isolation supplies a deterministic reclamation
boundary without weakening or skipping coverage; dated pass counts belong in the handoff rather
than in this policy document.

CI and the DA-PLCBF ready-for-review gate use `tests-full`. Render tests remain a separate explicit
gate because the repository-wide Pytest configuration excludes the `render` marker by default:

```bash
pixi run -e gpu-tests pytest -q -m render
```

Passing the core tier is a development signal only; it is not a substitute for the full CPU, GPU,
render, documentation, package, and scientific-evidence gates.
