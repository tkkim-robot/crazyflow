# Conditional finite-horizon coverage and evaluation protocol

This note proves a geometric implication conditional on uniform trajectory-error
bounds. It does not establish those bounds from the learner's loss, prove that
BPTT improves them, or prove recursive safety of the implemented controller.
The associated protocol utilities support a later frozen experiment; this note
does not seal an experiment or choose its episode budget.

## 1. Fixed trajectories and paired policy identities

Fix the initial full state, absolute start time, horizon \(T\), model snapshot,
policy snapshot, policy phase/anchor, and command-hold convention for each of two
declared predictions, denoted \(A\) and \(B\). Their dynamics or policies may
differ, but every compared policy identity \(i\) is paired explicitly. The full
actuator state affects the trajectories even though the geometric expressions
below depend only on position and attitude.

For body-attached collider sphere \(k\), define its center
\(a^r_{ik}(t)=p^r_i(t)+R^r_i(t)b^r_k\), \(r\in\{A,B\}\). Let obstacle \(j\)
have center \(c^r_j(t)\), and let
\(\rho^r_{jk}(t)=r^r_{\mathrm{obs},j}(t)+r^r_{\mathrm{ego},k}(t)+d^r(t)\)
be the sum of radii and requested separation. Define

\[
g^r_{ijk}(t)=\|a^r_{ik}(t)-c^r_j(t)\|-\rho^r_{jk}(t),\qquad
G^r_i=\inf_{t\in[0,T],j,k}g^r_{ijk}(t).
\]

Assume finite, nonempty obstacle/collider index sets, finite margins, and
nonnegative radii/separation. An empty obstacle set is a separate unconstrained
case; expressions such as \(\infty-\infty\) must not be used in this lemma.

## 2. Uniform error bound, including a rotating offset collider

Suppose the following bounds hold over the complete continuous horizon:

\[
\|p^A_i-p^B_i\|\le\epsilon_{p,i},\quad
\angle((R^B_i)^T R^A_i)\le\epsilon_{\theta,i}\le\pi,\quad
\|b^A_k-b^B_k\|\le\epsilon_{b,k},\quad \|b^B_k\|\le\ell_k.
\]

For rotations in three dimensions,
\(\|R^A_i-R^B_i\|_2=2\sin(\angle((R^B_i)^T R^A_i)/2)\).
Adding and subtracting \(R^A_i b^B_k\), then using the triangle inequality,
gives

\[
\|a^A_{ik}-a^B_{ik}\|
\le\epsilon_{p,i}+2\ell_k\sin(\epsilon_{\theta,i}/2)+\epsilon_{b,k}.
\]

Let obstacle-center error be at most \(\epsilon_{c,j}\), and let the error in
the total radius/separation be at most \(\eta_{jk}\). Add any **justified**
center-position numerical-error allowances for the two predictions, denoted
\(\epsilon^{A}_{\mathrm{num},ijk}+\epsilon^{B}_{\mathrm{num},ijk}\), to obtain

\[
\delta_{ijk}=\epsilon_{p,i}+2\ell_k\sin(\epsilon_{\theta,i}/2)
+\epsilon_{b,k}+\epsilon_{c,j}
+\epsilon^{A}_{\mathrm{num},ijk}+\epsilon^{B}_{\mathrm{num},ijk},\qquad
E_i=\max_{j,k}(\delta_{ijk}+\eta_{jk}).
\]

The allowances must have distinct meanings: do not add the same position or
attitude integration error twice. Omit the numerical terms when \(A,B\) already
denote exact continuous trajectories and their error bounds include integration.
Radius/geometry uncertainty and numerical center uncertainty both require an
independent justification; a chosen safety buffer is not evidence for either.
For a different geometric representation, a separately justified uniform error
in its signed separation value can be added to \(E_i\). Alternatively, valid
outer enclosures give a one-sided no-contact implication. An arbitrary mesh-to-
sphere fit does not supply a two-sided signed-distance error bound by itself.

**Lemma.** Under these assumptions, \(|G^A_i-G^B_i|\le E_i\).

**Proof.** Put \(z^r=a^r_{ik}-c^r_j\). The reverse triangle inequality gives
\(|\|z^A\|-\|z^B\||\le\|z^A-z^B\|\le\delta_{ijk}\).
Consequently \(|g^A_{ijk}-g^B_{ijk}|\le\delta_{ijk}+\eta_{jk}\le E_i\)
at every time and index. Thus \(g^A\ge g^B-E_i\) and \(g^B\ge g^A-E_i\).
Taking infima proves both inequalities. \(\square\)

In particular, \(G^B_i>E_i\) implies strictly positive predicted separation for
policy \(i\) throughout the horizon. With identical centered colliders and
obstacle paths this reduces to \(E_i=\epsilon_{p,i}\). Position error alone is
insufficient for an offset collider with changing attitude.

For the paired finite library,

\[
\max_i(G^B_i-E_i)\le\max_iG^A_i\le\max_i(G^B_i+E_i),\qquad
\left|\max_iG^A_i-\max_iG^B_i\right|\le\max_iE_i.
\]

The first inequality retains policy-specific allowances and can be tighter than
the second. These statements require the same paired identity set; they do not
compare unrelated library sizes without an additional matching argument.

## 3. Squared-distance values have square-metre error bounds

Suppose the implementation uses
\(h^r_{ijk}=\|z^r_{ijk}\|^2-(\rho^r_{jk})^2\), rather than distance margins.
Assume reference bounds \(\|z^B_{ijk}(t)\|\le D_{ijk}\) and
\(\rho^B_{jk}(t)\le P_{jk}\). Then

\[
\begin{aligned}
|h^A_{ijk}-h^B_{ijk}|
&\le |(z^A-z^B)^T(z^A+z^B)|
   +|\rho^A-\rho^B|(\rho^A+\rho^B)\\
&\le 2D_{ijk}\delta_{ijk}+\delta_{ijk}^2
    +2P_{jk}\eta_{jk}+\eta_{jk}^2.
\end{aligned}
\]

Taking the maximum of this right-hand side over \(j,k\), then the same infimum
and library-maximum argument, proves the corresponding squared-value bounds.
Every term has units \(\mathrm{m}^2\). A metre-valued tracking error cannot be
subtracted directly from a square-metre certificate. The reference distance and
radius bounds must apply over the whole horizon, not only at the minimizing
sample.

Positive \(h\) implies positive distance separation when the summed radius is
nonnegative. A conservatively enclosing sphere can certify absence of contact
for the enclosed geometry, but its overlap does not prove actual-collider
contact. A tighter physical collider and the safety enclosure must remain
separate outcome metrics.

## 4. Samples, numerical allowances, and the scope of the result

If the continuous distance-margin function is known to be \(L\)-Lipschitz in
time, a sample grid including both horizon endpoints with maximum spacing
\(\Delta\) gives

\[
\inf_{t,j,k}g_{ijk}(t)\ge\min_{n,j,k}g_{ijk}(t_n)-L\Delta/2.
\]

The nearest grid point is within \(\Delta/2\), which proves the inequality.
For fixed offsets, a valid bound can use relative center speed
\(\|v_i\|+\|\omega_i\|\ell_k+\|\dot c_j\|\), plus any radius/separation
rate. Those speed bounds must hold between samples. Squared values need their
own bound, for example \(2D\) times relative center speed plus \(2P\) times
radius rate, with continuous bounds \(D,P\). Verified sample-value errors are
additional allowances. Integrator convergence tests are evidence of numerical
consistency, not automatically rigorous global error certificates.

A sampled mean tracking loss is not a supremum error bound. A maximum on a
finite probe bank is conditional on that bank. This lemma proves neither BPTT
convergence nor monotonic improvement of finite updates, QP feasibility,
continued fallback feasibility, or infinite-time invariance. Restarting a
policy, changing its anchor, publishing parameters, changing the model, and
holding a command create additional obligations for any stronger claim.

Sampled-data CBF results explicitly address piecewise-constant input and
between-sample safety; satisfying a continuous CBF inequality only at sample
times is insufficient. Applying those theorems here would require checking
their hypotheses and obtaining the required reachable-set or derivative bounds
for the augmented motor-state system. The finite-horizon tube lemma above does
not perform that step. [Breeden, Garg, and Panagou](https://arxiv.org/html/2103.03677)

## 5. Prior-work boundary

The following comparison concerns the cited methods and this study's declared
architecture, not an empirical claim that one dominates another.

| Dimension | Learning on the Fly, revision 2 | Predictive safety filter | This actuator study |
|---|---|---|---|
| Learned object | Hover/tracking control policy; full or low-rank updates | Learning controller proposed externally; model can learn from data | Persistent, reusable fallback repertoire |
| Information access | State or visual features; task reward/reference | Current state, proposed input, constraints, model uncertainty | Actor/loss exclude obstacles and goals; runtime filter uses them |
| Safety mechanism | Reported task adaptation; no PL-CBF filter presented | Predictive constraint enforcement with a terminal safe-set construction | Declared finite-horizon rollout/QP and held-command checks |
| Model learning | Learned acceleration residual plus analytical model; analytical-only backward gradient | Data-driven model with state/input dependent uncertainty | Initially current-parameter oracle; mismatch/estimation are separately labeled |
| Runtime cost | Concurrent model/policy updates and deployment; reported policy adaptation about 1.5 seconds | Repeated model-predictive optimization | Measure controller, learner, completed publication, and delayed command costs |

Learning on the Fly already performs online BPTT policy adaptation through a
differentiable simulator. The proposed distinction is the obstacle-free
recovery repertoire and its runtime safety-filter role; novelty cannot be the
first use of online BPTT. Its reported update cost is contextual and is not a
matched-hardware benchmark for this implementation.
[Pan et al., Learning on the Fly, v2](https://arxiv.org/html/2508.21065v2)

An optimization comparator without a justified terminal safe set and the
framework's uncertainty conditions must be labeled an **MPC-based finite-horizon
safety filter**. It does not inherit the recursive guarantees of the cited
predictive safety filter merely by solving a finite-horizon problem.
[Wabersich and Zeilinger](https://arxiv.org/html/1812.05506v4)

## 6. Statistical interpretation

The protocol module requires explicit numeric ranges, disjoint physical worlds
and generation seeds for development/validation/test, resolved methods and
training support, outcome definitions, and prespecified analysis. It has no
test-driven tuning callback and refuses to overwrite a sealed file. The hash
detects changes; it does not establish that the declared model or experimental
choices are scientifically adequate.

Primary paired estimates average within each declared world-by-library-seed
cell. The crossed bootstrap independently resamples world IDs and library seed
IDs, retains their Cartesian product, and applies the same resampling to both
methods. Each dynamics/scenario cell is analyzed separately. Missing or
incomplete cells block the complete-matrix analysis; attempts and reasons remain
in the ledger rather than being silently dropped.

With three library seeds, the seed-distribution component is weakly measured.
Report the three seed-specific effects and a world-only conditional analysis
alongside the crossed interval; more worlds do not create more independently
trained libraries. Percentile bootstrap intervals are approximate, can be
conservative for crossed interaction effects, and can degenerate when no event
is observed. They are not proofs of zero event risk or reliable tail coverage.
Separate row/column resampling follows the crossed-data idea of the pigeonhole
bootstrap; its asymptotic results are not a small-three-seed coverage guarantee.
[Owen, The pigeonhole bootstrap](https://arxiv.org/abs/0712.1111)

The Wilson diagnostic first reduces each world to whether **any** of the fixed
library seeds exhibits the binary event. Its denominator is the number of
independent worlds. It estimates a conditional world-level event probability,
not a per-episode collision probability and not uncertainty over newly trained
libraries. Zero observed events still produce a positive upper limit. The
primary paired analysis retains the full crossed structure.
[Wilson's score construction](https://doi.org/10.1080/01621459.1927.10502953)

## 7. Manifest structure and use

All fields below must contain resolved values when `freeze_protocol` or
`seal_protocol` is called. Optional additional JSON fields are hashed too; no
Python callbacks, nonfinite numbers, null placeholders, or unresolved text can
enter a frozen manifest. The synthetic manifest in the unit tests is an API
fixture, not a proposed experimental budget.

| Section | Required content |
|---|---|
| `schema`, `protocol_id` | Schema `da_plcbf_actuator_protocol_v1` and a versioned study identifier |
| `source` | Full 40-character Git `commit`; nonempty `files_sha256` mapping identifying the actual source contents, including uncommitted study files if applicable |
| `models` | Each named model has `definition` and a nonempty numeric/resolved `parameters` mapping |
| `control` | Positive `integration_step_s`, `command_period_s`, `prediction_horizon_s`, ordered step ≤ period ≤ horizon; add resolved holding, publication, and latency details |
| `numeric_ranges` | Named `[lower, upper]` finite numeric ranges, with units in the names; add categorical support elsewhere rather than encoding it as numeric ranges |
| `methods` | Each method has `definition` and a nonempty resolved `configuration`, including information, adapter, controller, optimizer, and timing settings |
| `training_support` | One explicit object per method, including training seeds, transitions/updates, support, checkpoint hashes, and selection rule; explicitly describe methods requiring no training |
| `splits` | Exactly `development`, `validation`, `test`; each contains `world_seeds`, `library_seeds`, and `worlds` |
| Each world entry | `world_id`, `cell`, and `physical_spec`; the ID is `physical_world_id(physical_spec)`, not a generation seed or proposal counter |
| `outcomes` | Metric objects with `definition`, `kind` (`binary` or `continuous`), and `units`; include common exposure/termination semantics and all reported outcome distinctions |
| `analysis` | Numeric confidence/resampling count/seed; crossed cluster axes; percentile interval; principal method/metric/cell comparisons; missing-result and multiplicity policies |
| `budget_rationale` | The measured development/validation basis for the selected test budget, written before viewing test outcomes |

`analysis` uses `confidence_level`, `n_resamples` (at least 1,000),
`resampling_seed`, `cluster_axes: ["world", "library_seed"]`,
`interval: "percentile"`, `primary_multiplicity: "bonferroni"`,
`secondary_multiplicity: "descriptive_only"` or `"bonferroni"`, and
`missing_results: "refuse_incomplete_matrix"`. Its `primary_comparisons` mapping
assigns each comparison ID `method_a`, `method_b`, `metrics`, and `cells`.
`paired_protocol_bootstrap` applies the declared confidence correction across
all listed primary metric-by-cell comparisons. Secondary analyses require their
own prespecified accounting; the primary helper does not implement them.

Physical specifications must use one fixed serialization schema and include the
complete initial body/motor state, actual model and event parameters, obstacle
paths/geometry, waypoints, noise tapes or their authenticated definitions, and
exposure duration. Administrative seed/method/split/cell labels belong outside
the physical specification. The utility detects identical canonical content;
it cannot infer equivalence of different encodings or missing physical fields.

After profiling and validation, the runner constructs an in-memory
`FrozenProtocol`, checks its runtime sections with `require_section`, and seals
it into a new path. Existing paths are never overwritten. A changed method,
source, or protocol requires a new version; a correctness repair affecting
sealed results also requires explicit invalidation and symmetric reruns.
Sealing does not assert that the recorded source hashes match files on disk:
the runner must calculate and check those hashes when beginning execution.

`TrialLedger` retains interrupted, simulator-error, censored, and inadmissible
attempts with reasons. A later first completed attempt retains the same physical
trial ID and a distinct attempt ID; a completed result cannot be replaced.
Completed records must contain the declared primary metrics. World screening
cannot reject a scene after a completed comparative result exists. The ledger
validates identity and accounting, while the runner remains responsible for
honest collision detection, full exposure, publication timing, and method-
independent screening. It does not turn an incomplete physical episode into a
completed one by inspecting arbitrary scalar metrics.
