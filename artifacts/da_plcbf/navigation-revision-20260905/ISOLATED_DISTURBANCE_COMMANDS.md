# Isolated-event follow-up commands

Run these only after the parent confirms paced validation has finished and supplies the output-only patched frozen source. Lock that tree and its output-only changes in `ISOLATED_DISTURBANCE_EXECUTION_PROTOCOL.json` before launch. `ISOLATED_DISTURBANCE_PROTOCOL.json` is the immutable declaration; both event conditions use the exact original frozen candidate/config, seeds100–109, eight obstacles,40s and deterministic publication cadence. The seed0 static run is development evidence only.

Set `NAV_PATCHED_SOURCE` to the supplied absolute source directory. Each command runs from `/home/tk/Desktop/mycode/crazyflow`; use its absolute Python binary. The two isolated-condition commands can run concurrently after clearance; the optional static command follows when a slot is free. Preserve logs and all partial outputs on failure.

```bash
export NAV_PATCHED_SOURCE=/absolute/path/provided/by/parent
PYTHONPATH="$NAV_PATCHED_SOURCE" XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/tmp/crazyflow-navigation-jax-cache .pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_navigation_campaign.py --source-tree "$NAV_PATCHED_SOURCE" --checkpoint /home/tk/Desktop/mycode/crazyflow/artifacts/da_plcbf/learning-revision-20260905/reference-ablation-seed7/candidate/checkpoint --output /home/tk/Desktop/mycode/crazyflow/artifacts/da_plcbf/navigation-revision-20260905/isolated-wind --seeds 100 101 102 103 104 105 106 107 108 109 --conditions wind --obstacles 8 --duration 40 --model oracle --execution deterministic --learner-kind reference
PYTHONPATH="$NAV_PATCHED_SOURCE" XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/tmp/crazyflow-navigation-jax-cache .pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_navigation_campaign.py --source-tree "$NAV_PATCHED_SOURCE" --checkpoint /home/tk/Desktop/mycode/crazyflow/artifacts/da_plcbf/learning-revision-20260905/reference-ablation-seed7/candidate/checkpoint --output /home/tk/Desktop/mycode/crazyflow/artifacts/da_plcbf/navigation-revision-20260905/isolated-payload --seeds 100 101 102 103 104 105 106 107 108 109 --conditions payload --obstacles 8 --duration 40 --model oracle --execution deterministic --learner-kind reference
PYTHONPATH="$NAV_PATCHED_SOURCE" XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_COMPILATION_CACHE_DIR=/tmp/crazyflow-navigation-jax-cache .pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_navigation_campaign.py --source-tree "$NAV_PATCHED_SOURCE" --checkpoint /home/tk/Desktop/mycode/crazyflow/artifacts/da_plcbf/learning-revision-20260905/reference-ablation-seed7/candidate/checkpoint --output /home/tk/Desktop/mycode/crazyflow/artifacts/da_plcbf/navigation-revision-20260905/static-composition-development --seeds 0 --conditions static --obstacles 8 --duration 40 --model oracle --execution deterministic --learner-kind reference
```

After all20 paired event worlds export successfully, use the effective source hashes locked before launch:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' .pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_navigation_summary.py artifacts/da_plcbf/navigation-revision-20260905 --protocol ISOLATED_DISTURBANCE_EXECUTION_PROTOCOL.json --pattern 'isolated-*/*/navigation_comparison.json' --output artifacts/da_plcbf/navigation-revision-20260905/ISOLATED_DISTURBANCE_STATISTICS.json
```

The summary rejects missing, unexpected, duplicated, source-mismatched or checkpoint-mismatched worlds. Separately verify event lists, constant-world geometry equality, frozen config equality and numerical pre-event prefixes against existing unchanged runs. All methods start the same candidate; no wind/payload-specific training or tuning is allowed. Report safety, progress, encounter severity and degraded controls separately. Timing remains excluded from deterministic concurrent-run claims.

Run the independent recorded composition audit after the complete statistics check:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' .pixi/envs/gpu-tests/bin/python benchmark/da_plcbf_navigation_composition_audit.py artifacts/da_plcbf/navigation-revision-20260905 --output artifacts/da_plcbf/navigation-revision-20260905/ISOLATED_DISTURBANCE_COMPOSITION_AUDIT.json
```

Both commands refuse to overwrite existing evidence. Use a new output filename for independent reproduction when the canonical outputs already exist.
