"""Faithful fixed-budget proposal-training ablations for DA-PLCBF.

The routines in this module only propose candidate policy parameters.  They do not admit a
candidate, choose a runtime fallback, or certify an action.  Candidate admission and runtime use
the immutable-snapshot validation and exact hard finite-horizon post-checks elsewhere in this
package.

Two deliberately simple training comparators are provided:

* sampling-only evaluates an incumbent plus a fixed number of deterministic Gaussian parameter
  proposals and returns the best finite objective value;
* hybrid uses the same sampling kernel to seed a fixed number of truncated-BPTT updates and then
  evaluates the final iterate once, retaining the sampling seed when the final iterate is worse.

Every objective execution is accounted for.  In particular, a hybrid budget of ``E`` objective
evaluations and ``U`` gradient updates performs ``E - U - 1`` sampling evaluations, ``U``
value-and-gradient evaluations, and one final objective evaluation.  This makes matched-budget
comparisons explicit rather than treating an optimizer update as free.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from crazyflow.safety.da_plcbf.baselines import ImplementationAvailability
from crazyflow.safety.da_plcbf.bptt import BPTTFunctions


@dataclass(frozen=True, slots=True)
class ProposalBudget:
    """Exact candidate-objective and gradient-update budget for one proposal burst."""

    objective_evaluations: int
    gradient_updates: int = 0

    def validate(self) -> None:
        """Reject booleans, negative counts, and an empty objective budget."""
        if (
            isinstance(self.objective_evaluations, bool)
            or not isinstance(self.objective_evaluations, Integral)
            or self.objective_evaluations <= 0
        ):
            raise ValueError("objective_evaluations must be a positive integer")
        if (
            isinstance(self.gradient_updates, bool)
            or not isinstance(self.gradient_updates, Integral)
            or self.gradient_updates < 0
        ):
            raise ValueError("gradient_updates must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class SamplingProposalConfig:
    """Deterministic fixed-budget random-search settings.

    The first evaluation is always the unmodified incumbent.  Every other proposal adds an
    independent Gaussian perturbation with leaf scale
    ``absolute_stddev + relative_stddev * rms(leaf)``.
    """

    budget: ProposalBudget
    seed: int = 0
    relative_stddev: float = 0.05
    absolute_stddev: float = 1e-3

    def validate(self) -> None:
        """Validate the fixed evaluation budget and perturbation distribution."""
        self.budget.validate()
        if self.budget.gradient_updates != 0:
            raise ValueError("sampling-only requires gradient_updates=0")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            not math.isfinite(self.relative_stddev)
            or self.relative_stddev < 0
            or not math.isfinite(self.absolute_stddev)
            or self.absolute_stddev < 0
        ):
            raise ValueError("proposal standard deviations must be finite and nonnegative")
        if self.relative_stddev == 0 and self.absolute_stddev == 0:
            raise ValueError("at least one proposal standard deviation must be positive")


@dataclass(frozen=True, slots=True)
class HybridProposalConfig:
    """Sampling-seed plus truncated-BPTT settings under one exact evaluation budget."""

    budget: ProposalBudget
    seed: int = 0
    relative_stddev: float = 0.05
    absolute_stddev: float = 1e-3
    objective_consistency_rtol: float = 1e-5
    objective_consistency_atol: float = 1e-6

    def validate(self) -> None:
        """Require room for sampling, every gradient evaluation, and a final evaluation."""
        self.budget.validate()
        if self.budget.gradient_updates <= 0:
            raise ValueError("hybrid proposal requires at least one gradient update")
        minimum = self.budget.gradient_updates + 2
        if self.budget.objective_evaluations < minimum:
            raise ValueError(
                "hybrid objective_evaluations must include at least one sampling evaluation, "
                "every gradient update, and one final evaluation"
            )
        sampling = SamplingProposalConfig(
            budget=ProposalBudget(
                self.budget.objective_evaluations - self.budget.gradient_updates - 1
            ),
            seed=self.seed,
            relative_stddev=self.relative_stddev,
            absolute_stddev=self.absolute_stddev,
        )
        sampling.validate()
        tolerances = (self.objective_consistency_rtol, self.objective_consistency_atol)
        if not all(math.isfinite(value) and value >= 0 for value in tolerances):
            raise ValueError("objective consistency tolerances must be finite and nonnegative")

    @property
    def sampling_evaluations(self) -> int:
        """Number of incumbent/proposal objective calls before BPTT."""
        return int(self.budget.objective_evaluations) - int(self.budget.gradient_updates) - 1


class SamplingDeviceResult(NamedTuple):
    """JAX-compatible output of the sampling proposal kernel."""

    params: Any
    candidate_losses: Array
    selected_loss: Array
    incumbent_loss: Array
    selected_index: Array
    improved: Array
    input_valid: Array


@dataclass(frozen=True, slots=True)
class ProposalAccounting:
    """Measured execution counts for one proposal call."""

    requested_objective_evaluations: int
    actual_objective_evaluations: int
    gradient_evaluations: int
    attempted_gradient_updates: int
    accepted_gradient_updates: int
    sampling_evaluations: int
    final_evaluations: int


@dataclass(frozen=True, slots=True)
class ProposalTiming:
    """Synchronized host wall-clock timings; compilation is never hidden as an evaluation."""

    compile_seconds: float
    sampling_seconds: float
    gradient_seconds: float
    final_evaluation_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class ProposalResult:
    """Host-auditable candidate proposal and its complete budget/timing evidence.

    ``selected_index`` addresses ``candidate_losses`` for a sampling result.  A value of ``-1``
    denotes the separately evaluated final BPTT iterate of a hybrid result.
    """

    params: Any
    selected_loss: float
    incumbent_loss: float
    selected_index: int
    candidate_losses: Array
    gradient_losses: Array
    gradient_update_accepted: Array
    improved: bool
    input_valid: bool
    seed: int
    accounting: ProposalAccounting
    timing: ProposalTiming


def implemented_proposal_ablation_availability(
    base: ImplementationAvailability | None = None,
) -> ImplementationAvailability:
    """Opt a runner into only the project comparators implemented by this module pair.

    Merely importing the registry does not assert dispatch support.  A campaign runner should call
    this helper only after wiring sampling-only, hybrid, and :mod:`independent_actor` execution.
    SHAC is intentionally left unchanged and therefore remains unavailable by default.
    """
    selected = base or ImplementationAvailability()
    if not isinstance(selected, ImplementationAvailability):
        raise TypeError("base must be an ImplementationAvailability")
    return replace(
        selected,
        faithful_sampling_only_training=True,
        faithful_hybrid_proposal_bptt_training=True,
        faithful_independent_actors=True,
    )


def _validate_parameter_tree(params: Any) -> None:
    leaves = jax.tree.leaves(params)
    if not leaves:
        raise ValueError("parameter tree must contain at least one array leaf")
    for leaf in leaves:
        array = jnp.asarray(leaf)
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            raise TypeError("every parameter leaf must have a floating or complex dtype")
        if array.size == 0:
            raise ValueError("parameter leaves must not be empty")
        if not bool(jnp.all(jnp.isfinite(array))):
            raise ValueError("parameter leaves must be finite")


def _sample_parameter_batch(
    params: Any, *, evaluation_count: int, seed: int, relative_stddev: float, absolute_stddev: float
) -> Any:
    """Return a PyTree with a leading proposal axis and the incumbent at index zero."""
    leaves, treedef = jax.tree.flatten(params)
    root_key = jax.random.key(seed)
    keys = tuple(jax.random.fold_in(root_key, index) for index in range(len(leaves)))
    sampled_leaves: list[Array] = []
    proposal_count = evaluation_count - 1
    for leaf, key in zip(leaves, keys, strict=True):
        leaf = jnp.asarray(leaf)
        rms = jnp.sqrt(jnp.mean(jnp.real(leaf * jnp.conj(leaf))))
        scale = (
            jnp.asarray(absolute_stddev, dtype=leaf.dtype)
            + jnp.asarray(relative_stddev, dtype=leaf.dtype) * rms
        )
        # Address every proposal by its index rather than drawing one shape-dependent tensor.
        # Therefore a smaller evaluation budget is an exact prefix of a larger one with the same
        # seed, which is required for the matched sampling/hybrid campaign.
        if proposal_count:
            noise = jnp.stack(
                tuple(
                    jax.random.normal(
                        jax.random.fold_in(key, proposal_index), leaf.shape, dtype=leaf.dtype
                    )
                    for proposal_index in range(proposal_count)
                )
            )
            proposals = leaf[None, ...] + scale * noise
            sampled_leaves.append(jnp.concatenate((leaf[None, ...], proposals), axis=0))
        else:
            sampled_leaves.append(leaf[None, ...])
    return jax.tree.unflatten(treedef, sampled_leaves)


def build_sampling_proposal_kernel(
    objective: Callable[[Any], Array],
    config: SamplingProposalConfig,
    *,
    project_params: Callable[[Any], Any] | None = None,
) -> Callable[[Any], SamplingDeviceResult]:
    """Build a JIT-compiled, deterministic sampling-only proposal kernel.

    ``objective`` must be a pure scalar JAX function of the parameter PyTree.  The kernel remains
    usable under an outer :func:`jax.jit`; wall-clock accounting is intentionally provided only by
    :func:`run_sampling_only_proposal`.
    """
    if not callable(objective):
        raise TypeError("objective must be callable")
    if project_params is not None and not callable(project_params):
        raise TypeError("project_params must be callable or None")
    config.validate()
    evaluation_count = int(config.budget.objective_evaluations)
    project = (lambda candidate: candidate) if project_params is None else project_params

    def propose(params: Any) -> SamplingDeviceResult:
        candidates = _sample_parameter_batch(
            params,
            evaluation_count=evaluation_count,
            seed=int(config.seed),
            relative_stddev=config.relative_stddev,
            absolute_stddev=config.absolute_stddev,
        )
        # Projection is part of the proposal protocol, not an after-the-fact repair.  It keeps
        # disabled/structural/fixed leaves byte-exact both while scoring and in the selected
        # candidate returned to the caller.
        candidates = jax.vmap(project)(candidates)
        losses = jax.vmap(objective)(candidates)
        if losses.shape != (evaluation_count,):
            raise ValueError("objective must return one scalar for each parameter candidate")
        finite = jnp.isfinite(losses)
        safe_losses = jnp.where(finite, losses, jnp.inf)
        best_index = jnp.argmin(safe_losses)
        incumbent_valid = finite[0]
        # An invalid incumbent denotes invalid proposal inputs.  It may not be replaced by a
        # fortunate random draw because doing so would conceal the failed input boundary.
        selected_index = jnp.where(incumbent_valid, best_index, 0)
        selected = jax.tree.map(lambda leaf: leaf[selected_index], candidates)
        selected_loss = losses[selected_index]
        improved = incumbent_valid & jnp.isfinite(selected_loss) & (selected_loss < losses[0])
        return SamplingDeviceResult(
            params=selected,
            candidate_losses=losses,
            selected_loss=selected_loss,
            incumbent_loss=losses[0],
            selected_index=selected_index,
            improved=improved,
            input_valid=incumbent_valid,
        )

    return jax.jit(propose)


def _synchronize(value: Any) -> None:
    leaves = jax.tree.leaves(value)
    if leaves:
        jax.block_until_ready(leaves)


def _compile_sampling_kernel(
    kernel: Callable[[Any], SamplingDeviceResult], params: Any
) -> tuple[Callable[[Any], SamplingDeviceResult], float]:
    start = time.perf_counter()
    if hasattr(kernel, "lower"):
        executable = kernel.lower(params).compile()
    else:  # pragma: no cover - public runner accepts compatible user wrappers.
        executable = kernel
    elapsed = time.perf_counter() - start
    return executable, elapsed


def run_sampling_only_proposal(
    params: Any,
    objective: Callable[[Any], Array],
    config: SamplingProposalConfig,
    *,
    project_params: Callable[[Any], Any] | None = None,
) -> ProposalResult:
    """Execute one synchronized sampling-only burst with exact evaluation accounting."""
    config.validate()
    _validate_parameter_tree(params)
    total_start = time.perf_counter()
    kernel = build_sampling_proposal_kernel(objective, config, project_params=project_params)
    executable, compile_seconds = _compile_sampling_kernel(kernel, params)
    sampling_start = time.perf_counter()
    device = executable(params)
    _synchronize(device)
    sampling_seconds = time.perf_counter() - sampling_start
    total_seconds = time.perf_counter() - total_start
    evaluations = int(config.budget.objective_evaluations)
    return ProposalResult(
        params=device.params,
        selected_loss=float(device.selected_loss),
        incumbent_loss=float(device.incumbent_loss),
        selected_index=int(device.selected_index),
        candidate_losses=device.candidate_losses,
        gradient_losses=jnp.empty((0,), dtype=device.candidate_losses.dtype),
        gradient_update_accepted=jnp.empty((0,), dtype=bool),
        improved=bool(device.improved),
        input_valid=bool(device.input_valid),
        seed=int(config.seed),
        accounting=ProposalAccounting(
            requested_objective_evaluations=evaluations,
            actual_objective_evaluations=evaluations,
            gradient_evaluations=0,
            attempted_gradient_updates=0,
            accepted_gradient_updates=0,
            sampling_evaluations=evaluations,
            final_evaluations=0,
        ),
        timing=ProposalTiming(
            compile_seconds=compile_seconds,
            sampling_seconds=sampling_seconds,
            gradient_seconds=0.0,
            final_evaluation_seconds=0.0,
            total_seconds=total_seconds,
        ),
    )


def _loss_total(step_metrics: Any) -> Array:
    loss = getattr(step_metrics, "loss", None)
    total = getattr(loss, "total", None)
    if total is None:
        raise TypeError("BPTT step metrics must expose a scalar loss.total")
    total = jnp.asarray(total)
    if total.shape:
        raise ValueError("BPTT step metrics loss.total must be scalar")
    return total


def run_hybrid_proposal_bptt(
    params: Any,
    objective: Callable[[Any], Array],
    bptt_functions: BPTTFunctions,
    bptt_arguments: Sequence[Any],
    config: HybridProposalConfig,
    *,
    project_params: Callable[[Any], Any] | None = None,
) -> ProposalResult:
    """Run deterministic sampling followed by a fixed number of truncated-BPTT updates.

    ``bptt_arguments`` are passed after the BPTT state to ``bptt_functions.step``.  The supplied
    scalar ``objective`` must be the same objective used by that BPTT step.  The first gradient
    evaluation is cross-checked against the selected sampling loss; a mismatch raises instead of
    producing a confounded hybrid comparator.
    """
    config.validate()
    _validate_parameter_tree(params)
    if not isinstance(bptt_functions, BPTTFunctions):
        raise TypeError("bptt_functions must be a BPTTFunctions")
    if not isinstance(bptt_arguments, Sequence):
        raise TypeError("bptt_arguments must be a sequence")
    if project_params is not None and not callable(project_params):
        raise TypeError("project_params must be callable or None")
    project = (lambda candidate: candidate) if project_params is None else project_params

    sampling_config = SamplingProposalConfig(
        budget=ProposalBudget(config.sampling_evaluations),
        seed=config.seed,
        relative_stddev=config.relative_stddev,
        absolute_stddev=config.absolute_stddev,
    )
    total_start = time.perf_counter()
    sampling_kernel = build_sampling_proposal_kernel(
        objective, sampling_config, project_params=project_params
    )
    sampling_executable, sampling_compile = _compile_sampling_kernel(sampling_kernel, params)
    sampling_start = time.perf_counter()
    sampled = sampling_executable(params)
    _synchronize(sampled)
    sampling_seconds = time.perf_counter() - sampling_start

    state = bptt_functions.initialize(project(sampled.params))
    gradient_losses: list[Array] = []
    accepted: list[Array] = []
    bptt_compile_start = time.perf_counter()
    step_function = bptt_functions.step
    if hasattr(step_function, "lower"):
        step_executable = step_function.lower(state, *bptt_arguments).compile()
    else:  # pragma: no cover - supports compatible externally supplied wrappers.
        step_executable = step_function
    bptt_compile = time.perf_counter() - bptt_compile_start
    gradient_start = time.perf_counter()
    for update_index in range(int(config.budget.gradient_updates)):
        state, metrics = step_executable(state, *bptt_arguments)
        _synchronize(metrics)
        loss = _loss_total(metrics)
        if update_index == 0 and bool(sampled.input_valid):
            if not np.isclose(
                float(loss),
                float(sampled.selected_loss),
                rtol=config.objective_consistency_rtol,
                atol=config.objective_consistency_atol,
            ):
                raise ValueError(
                    "hybrid sampling objective does not match the BPTT step objective at the seed"
                )
        gradient_losses.append(loss)
        accepted.append(jnp.asarray(metrics.update_accepted))
    gradient_seconds = time.perf_counter() - gradient_start

    final_compile_start = time.perf_counter()
    final_kernel = jax.jit(objective)
    projected_final_params = project(state.params)
    final_executable = final_kernel.lower(projected_final_params).compile()
    final_compile = time.perf_counter() - final_compile_start
    final_start = time.perf_counter()
    final_loss_array = final_executable(projected_final_params)
    _synchronize(final_loss_array)
    final_seconds = time.perf_counter() - final_start

    seed_loss = float(sampled.selected_loss)
    final_loss = float(final_loss_array)
    choose_final = math.isfinite(final_loss) and (
        not math.isfinite(seed_loss) or final_loss < seed_loss
    )
    selected_params = projected_final_params if choose_final else sampled.params
    selected_loss = final_loss if choose_final else seed_loss
    gradient_loss_array = jnp.stack(gradient_losses)
    accepted_array = jnp.stack(accepted).astype(bool)
    accepted_count = int(jnp.sum(accepted_array))
    total_seconds = time.perf_counter() - total_start
    requested = int(config.budget.objective_evaluations)
    return ProposalResult(
        params=selected_params,
        selected_loss=selected_loss,
        incumbent_loss=float(sampled.incumbent_loss),
        selected_index=(-1 if choose_final else int(sampled.selected_index)),
        candidate_losses=sampled.candidate_losses,
        gradient_losses=gradient_loss_array,
        gradient_update_accepted=accepted_array,
        improved=(
            bool(sampled.input_valid)
            and math.isfinite(selected_loss)
            and selected_loss < float(sampled.incumbent_loss)
        ),
        input_valid=bool(sampled.input_valid),
        seed=int(config.seed),
        accounting=ProposalAccounting(
            requested_objective_evaluations=requested,
            actual_objective_evaluations=(
                config.sampling_evaluations + int(config.budget.gradient_updates) + 1
            ),
            gradient_evaluations=int(config.budget.gradient_updates),
            attempted_gradient_updates=int(config.budget.gradient_updates),
            accepted_gradient_updates=accepted_count,
            sampling_evaluations=config.sampling_evaluations,
            final_evaluations=1,
        ),
        timing=ProposalTiming(
            compile_seconds=sampling_compile + bptt_compile + final_compile,
            sampling_seconds=sampling_seconds,
            gradient_seconds=gradient_seconds,
            final_evaluation_seconds=final_seconds,
            total_seconds=total_seconds,
        ),
    )


def run_bptt_only_proposal(
    params: Any,
    bptt_functions: BPTTFunctions,
    bptt_arguments: Sequence[Any],
    budget: ProposalBudget,
    *,
    project_params: Callable[[Any], Any] | None = None,
) -> ProposalResult:
    """Run a BPTT-only proposal with one charged objective evaluation per gradient update.

    The BPTT step computes a value and gradient together, so the protocol requires
    ``objective_evaluations == gradient_updates``.  No uncharged final objective call is made.
    Consequently ``selected_loss`` is the last charged loss (evaluated immediately before the
    final accepted/rejected update), while ``params`` is the projected post-update candidate.
    Campaigns must evaluate all returned candidates with a separate common hard scorer; they must
    not present ``selected_loss`` as an evaluation of the post-update parameters.
    """
    budget.validate()
    if budget.gradient_updates <= 0:
        raise ValueError("BPTT-only requires at least one gradient update")
    if budget.objective_evaluations != budget.gradient_updates:
        raise ValueError("BPTT-only requires objective_evaluations == gradient_updates")
    _validate_parameter_tree(params)
    if not isinstance(bptt_functions, BPTTFunctions):
        raise TypeError("bptt_functions must be a BPTTFunctions")
    if not isinstance(bptt_arguments, Sequence):
        raise TypeError("bptt_arguments must be a sequence")
    if project_params is not None and not callable(project_params):
        raise TypeError("project_params must be callable or None")
    project = (lambda candidate: candidate) if project_params is None else project_params

    total_start = time.perf_counter()
    state = bptt_functions.initialize(project(params))
    compile_start = time.perf_counter()
    step_function = bptt_functions.step
    if hasattr(step_function, "lower"):
        step_executable = step_function.lower(state, *bptt_arguments).compile()
    else:  # pragma: no cover - supports compatible externally supplied wrappers.
        step_executable = step_function
    compile_seconds = time.perf_counter() - compile_start

    losses: list[Array] = []
    accepted: list[Array] = []
    gradient_start = time.perf_counter()
    for _ in range(int(budget.gradient_updates)):
        state, metrics = step_executable(state, *bptt_arguments)
        _synchronize(metrics)
        losses.append(_loss_total(metrics))
        accepted.append(jnp.asarray(metrics.update_accepted))
    gradient_seconds = time.perf_counter() - gradient_start

    gradient_losses = jnp.stack(losses)
    accepted_array = jnp.stack(accepted).astype(bool)
    input_valid = bool(jnp.isfinite(gradient_losses[0]))
    selected_loss = float(gradient_losses[-1])
    incumbent_loss = float(gradient_losses[0])
    output_params = project(state.params)
    requested = int(budget.objective_evaluations)
    return ProposalResult(
        params=output_params,
        selected_loss=selected_loss,
        incumbent_loss=incumbent_loss,
        selected_index=-1,
        candidate_losses=jnp.empty((0,), dtype=gradient_losses.dtype),
        gradient_losses=gradient_losses,
        gradient_update_accepted=accepted_array,
        improved=(input_valid and math.isfinite(selected_loss) and selected_loss < incumbent_loss),
        input_valid=input_valid,
        seed=0,
        accounting=ProposalAccounting(
            requested_objective_evaluations=requested,
            actual_objective_evaluations=requested,
            gradient_evaluations=requested,
            attempted_gradient_updates=requested,
            accepted_gradient_updates=int(jnp.sum(accepted_array)),
            sampling_evaluations=0,
            final_evaluations=0,
        ),
        timing=ProposalTiming(
            compile_seconds=compile_seconds,
            sampling_seconds=0.0,
            gradient_seconds=gradient_seconds,
            final_evaluation_seconds=0.0,
            total_seconds=time.perf_counter() - total_start,
        ),
    )


def require_matched_objective_budget(*results: ProposalResult) -> None:
    """Reject a scientific comparison whose requested/actual objective budgets differ."""
    if len(results) < 2:
        raise ValueError("at least two proposal results are required for a matched comparison")
    requested = {item.accounting.requested_objective_evaluations for item in results}
    actual = {item.accounting.actual_objective_evaluations for item in results}
    if len(requested) != 1 or len(actual) != 1 or requested != actual:
        raise ValueError("proposal results do not have a matched objective-evaluation budget")


__all__ = [
    "HybridProposalConfig",
    "ProposalAccounting",
    "ProposalBudget",
    "ProposalResult",
    "ProposalTiming",
    "SamplingDeviceResult",
    "SamplingProposalConfig",
    "build_sampling_proposal_kernel",
    "implemented_proposal_ablation_availability",
    "require_matched_objective_budget",
    "run_hybrid_proposal_bptt",
    "run_bptt_only_proposal",
    "run_sampling_only_proposal",
]
