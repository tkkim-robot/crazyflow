# Opposite failures in dense confirmation worlds

The two opposite shell-clearance outcomes have different local explanations. In seed 206,
the adaptive library does not repair the frozen method's first degraded state when queried
at that same state. In seed 209, the frozen library offers a nondegraded direct fallback at
the adaptive method's first degraded state, although the ranking reverses one control earlier.
These are bounded same-state comparisons; neither is a full counterfactual recovery episode.

Both investigations replay 50 recorded adaptive updates, including the complete Adam history,
and match all 43 learner/optimizer/model leaves exactly in dtype, shape, and bytes at the next
saved checkpoint. All four queried commands from each recorded anchor method also reproduce
with maximum absolute wrench error zero. Each library is evaluated with the anchor method's
same physical state, model, goal, obstacles, and previous selection. Adaptive parameters come
only from their original recorded training path.

| World and anchor | Time (s) | Frozen fallback max H (m²) | Adaptive fallback max H (m²) | Frozen result | Adaptive result |
| --- | ---: | ---: | ---: | --- | --- |
| 206, frozen state | 19.00 | 0.049024224 | 0.050130799 | accepted QP | accepted QP |
| 206, frozen state | 19.32 | 0.031943589 | 0.036976904 | accepted QP | accepted QP |
| 206, frozen state | 19.36 | 0.030110925 | 0.034436226 | degraded | degraded |
| 206, frozen state | 19.48 | −0.006366089 | −0.006748945 | degraded | degraded |
| 209, adaptive state | 13.00 | 0.612841010 | 0.621303737 | accepted QP | accepted QP |
| 209, adaptive state | 13.80 | −0.001407996 | 0.005307123 | degraded direct fallback | direct fallback |
| 209, adaptive state | 13.84 | 0.000822484 | −0.002870873 | direct fallback | degraded direct fallback |
| 209, adaptive state | 14.00 | −0.006098405 | −0.015983388 | degraded direct fallback | degraded direct fallback |

H is the recorded hard collision rollout value. QP eligibility, QP feasibility, and held-command
checks remain separate requirements. A positive hard H alone does not establish QP acceptance.

At seed 206's first degraded control, 19.36 s, both libraries retain one eligible candidate,
but the solver reports `qp_infeasible`. Both rejected proposals have 18 nonfinite held residual
entries, preserved in the raw fixture, and both lead to the same degraded applied wrench.
The applied minimum operational derivative residual is −0.097155094. At 19.48 s, both libraries
have negative hard coverage and zero eligible candidates. The successful adaptive episode
therefore reflects an earlier difference in the path taken, rather than a library swap that
rescues these queried frozen states. The first loss occurs after the wind event at 8 s and the
payload event at 16 s; that event ordering does not establish disturbance-adaptation necessity.

At seed 209's first adaptive degraded control, 13.84 s, the frozen library retains a small positive
hard H and passes the direct-fallback checks; the adaptive library's hard H is negative. Neither
library has a smooth QP-eligible candidate there. At 13.80 s the adaptive library instead has the
positive hard value and nondegraded fallback. By 14.00 s both hard values are negative. This loss
occurs after the 8 s wind event and before the 16 s payload event. The result identifies a local
loss of fallback coverage without showing that a frozen-library intervention would finish the
episode successfully.

The full ten-world dense comparison remains mixed: each method completes nine routes and has
nine positive-shell outcomes, with failures on different worlds. The paired mean waypoint
difference is zero. Completion-time comparisons include only the eight worlds where both
methods finish. The plots in `campaign-comparison-figures-v2/` show every paired world and the
saved uncertainty intervals.

Validated evidence:

- `dense-confirmation-repair/combined-seed206/pre_failure_replay_v2/` reconstructs controls 400–500 and probes 475, 483, 484, 487.
- `dense-confirmation-repair/combined-seed209/pre_failure/` reconstructs controls 300–400 and probes 325, 345, 346, 350.
- Each contains the exact analysis source, source/input hashes, full reconstruction arrays,
  checkpoints, and `same_state_full_fixtures.npz` with every candidate state/wrench trajectory
  and the common query model. Use that query model when reproducing a counterfactual; the saved
  learner checkpoint preserves its own latest model estimate as part of exact continuation.

The first seed 206 attempt in `dense-confirmation-repair/combined-seed206/pre_failure/` is excluded: it failed exact continuation
parity with maximum leaf error 5.7667494e−6 and emitted no causal comparison rows. The valid retry
used the campaign's `JAX_COMPILATION_CACHE_DIR=/tmp/crazyflow-navigation-jax-cache` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, with imports explicitly checked against the output-patched
frozen package. Both settings changed together, so the failed attempt does not isolate which
setting caused the numerical difference. Exact reproducibility is tied to the recorded software,
device, and compiled cache; no tolerance was relaxed to admit that failed continuation.

`REPLAY_JAX_CACHE.tar.gz` preserves the compiled cache used by the valid replays;
`REPLAY_JAX_CACHE_MANIFEST.json` records every cache-file hash and restoration requirements.
Each valid probe's `REPLAY_BYTE_AUDIT.json` independently checks all 43 continuation leaves,
all four recorded anchor commands, the saved analysis source, and the nominal-reference inputs.
Analysis directories were relocated beside their original episodes without changing existing
file contents; execution provenance retains the historical output paths.
