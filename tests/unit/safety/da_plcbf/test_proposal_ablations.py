from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflow.safety.da_plcbf.actor import (
    SharedActorConfig,
    SharedActorParams,
    SharedActorSpec,
    initialize_shared_actor,
)
from crazyflow.safety.da_plcbf.bptt import BPTTFunctions, BPTTState, BPTTStepMetrics
from crazyflow.safety.da_plcbf.proposal_ablations import (
    HybridProposalConfig,
    ProposalBudget,
    SamplingProposalConfig,
    build_sampling_proposal_kernel,
    require_matched_objective_budget,
    run_bptt_only_proposal,
    run_hybrid_proposal_bptt,
    run_sampling_only_proposal,
)


class _Loss(NamedTuple):
    total: jax.Array


def _quadratic_bptt(*, learning_rate: float = 0.2) -> BPTTFunctions:
    def initialize(params: object) -> BPTTState:
        return BPTTState(params=params, optimizer_state=(), steps=jnp.array(0, dtype=jnp.int32))

    def update(state: BPTTState, target: jax.Array) -> tuple[BPTTState, BPTTStepMetrics]:
        def objective(candidate: object) -> jax.Array:
            return jnp.sum((candidate["x"] - target) ** 2)

        loss, gradient = jax.value_and_grad(objective)(state.params)
        proposed = jax.tree.map(
            lambda value, grad: value - learning_rate * grad, state.params, gradient
        )
        delta = jnp.sqrt(
            sum(
                jnp.sum((new - old) ** 2)
                for new, old in zip(
                    jax.tree.leaves(proposed), jax.tree.leaves(state.params), strict=True
                )
            )
        )
        gradient_norm = jnp.sqrt(sum(jnp.sum(value**2) for value in jax.tree.leaves(gradient)))
        metrics = BPTTStepMetrics(
            loss=_Loss(loss),
            gradient_norm=gradient_norm,
            parameter_delta_norm=delta,
            update_accepted=jnp.isfinite(loss),
        )
        return (state.replace(params=proposed, steps=state.steps + 1), metrics)

    step = jax.jit(update)
    return BPTTFunctions(initialize=initialize, step=step, burst=step)


def test_sampling_only_is_reproducible_jittable_and_never_worse_than_incumbent() -> None:
    params = {"x": jnp.array([2.0, -1.0]), "bias": jnp.array([0.5])}

    def objective(candidate: object) -> jax.Array:
        return jnp.sum(candidate["x"] ** 2) + 0.1 * jnp.sum(candidate["bias"] ** 2)

    config = SamplingProposalConfig(
        budget=ProposalBudget(12), seed=81, relative_stddev=0.4, absolute_stddev=0.1
    )
    kernel = build_sampling_proposal_kernel(objective, config)
    compiled = jax.jit(kernel)(params)
    first = run_sampling_only_proposal(params, objective, config)
    second = run_sampling_only_proposal(params, objective, config)

    np.testing.assert_array_equal(first.candidate_losses, second.candidate_losses)
    np.testing.assert_array_equal(compiled.candidate_losses, first.candidate_losses)
    for first_leaf, second_leaf in zip(
        jax.tree.leaves(first.params), jax.tree.leaves(second.params), strict=True
    ):
        np.testing.assert_array_equal(first_leaf, second_leaf)
    assert first.selected_loss <= first.incumbent_loss
    assert first.accounting.actual_objective_evaluations == 12
    assert first.accounting.sampling_evaluations == 12
    assert first.accounting.gradient_evaluations == 0
    assert first.timing.total_seconds >= first.timing.sampling_seconds >= 0
    assert first.seed == 81


def test_sampling_operates_on_the_exact_shared_actor_parameter_pytree() -> None:
    spec = SharedActorSpec(
        base_codes=jnp.array([[1.0, 0.0], [-1.0, 0.0]]),
        base_desired_velocities=jnp.array([[0.2, 0.0], [-0.2, 0.0]]),
        base_durations=jnp.ones(2),
        adaptive_mask=jnp.array([False, True]),
    )
    params = initialize_shared_actor(
        jax.random.key(7),
        spec,
        dimension=2,
        n_obstacles=1,
        config=SharedActorConfig(hidden_width=4),
    )

    def objective(candidate: SharedActorParams) -> jax.Array:
        target = jnp.array([[0.0, 0.0], [0.6, -0.4]])
        return jnp.sum((candidate.velocity_offsets - target) ** 2)

    result = run_sampling_only_proposal(
        params,
        objective,
        SamplingProposalConfig(
            ProposalBudget(16), seed=12, relative_stddev=0.1, absolute_stddev=0.25
        ),
    )

    assert isinstance(result.params, SharedActorParams)
    assert result.params.input_kernel.shape == params.input_kernel.shape
    assert result.params.velocity_offsets.shape == spec.base_desired_velocities.shape
    assert result.selected_loss <= result.incumbent_loss


def test_sampling_changes_with_seed_but_incumbent_is_always_candidate_zero() -> None:
    params = {"x": jnp.array([1.5, -0.4])}

    def objective(candidate: object) -> jax.Array:
        return jnp.sum(candidate["x"] ** 2)

    first = run_sampling_only_proposal(
        params, objective, SamplingProposalConfig(ProposalBudget(8), seed=1, relative_stddev=0.5)
    )
    second = run_sampling_only_proposal(
        params, objective, SamplingProposalConfig(ProposalBudget(8), seed=2, relative_stddev=0.5)
    )

    assert float(first.candidate_losses[0]) == pytest.approx(float(second.candidate_losses[0]))
    assert not np.array_equal(
        np.asarray(first.candidate_losses[1:]), np.asarray(second.candidate_losses[1:])
    )


def test_hybrid_has_exact_matched_budget_and_bptt_improves_the_sampling_seed() -> None:
    params = {"x": jnp.array([3.0, -2.0])}
    target = jnp.array([0.25, -0.5])

    def objective(candidate: object) -> jax.Array:
        return jnp.sum((candidate["x"] - target) ** 2)

    sampling = run_sampling_only_proposal(
        params,
        objective,
        SamplingProposalConfig(
            ProposalBudget(10), seed=9, relative_stddev=0.15, absolute_stddev=0.01
        ),
    )
    hybrid = run_hybrid_proposal_bptt(
        params,
        objective,
        _quadratic_bptt(),
        (target,),
        HybridProposalConfig(
            ProposalBudget(10, gradient_updates=4),
            seed=9,
            relative_stddev=0.15,
            absolute_stddev=0.01,
        ),
    )

    require_matched_objective_budget(sampling, hybrid)
    assert hybrid.accounting.sampling_evaluations == 5
    assert hybrid.accounting.gradient_evaluations == 4
    assert hybrid.accounting.final_evaluations == 1
    assert hybrid.accounting.actual_objective_evaluations == 10
    assert hybrid.accounting.accepted_gradient_updates == 4
    assert hybrid.selected_index == -1
    assert hybrid.selected_loss < float(hybrid.candidate_losses[hybrid.candidate_losses.argmin()])
    assert np.all(np.asarray(hybrid.gradient_update_accepted))
    assert np.all(np.diff(np.asarray(hybrid.gradient_losses)) < 0)


def test_hybrid_rejects_an_objective_that_does_not_match_its_bptt_step() -> None:
    params = {"x": jnp.array([2.0])}
    target = jnp.array([0.0])
    with pytest.raises(ValueError, match="does not match"):
        run_hybrid_proposal_bptt(
            params,
            lambda candidate: 2.0 * jnp.sum(candidate["x"] ** 2),
            _quadratic_bptt(),
            (target,),
            HybridProposalConfig(ProposalBudget(5, gradient_updates=1)),
        )


def test_invalid_incumbent_fails_closed_without_hiding_the_fixed_budget() -> None:
    params = {"x": jnp.array([1.0])}
    result = run_sampling_only_proposal(
        params,
        lambda _candidate: jnp.asarray(jnp.inf),
        SamplingProposalConfig(ProposalBudget(4), seed=3),
    )

    assert not result.input_valid
    assert not result.improved
    assert result.selected_index == 0
    assert np.isinf(result.selected_loss)
    assert result.accounting.actual_objective_evaluations == 4
    np.testing.assert_array_equal(result.params["x"], params["x"])


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (SamplingProposalConfig(ProposalBudget(0)), "positive integer"),
        (SamplingProposalConfig(ProposalBudget(2, gradient_updates=1)), "gradient_updates=0"),
        (
            SamplingProposalConfig(ProposalBudget(2), relative_stddev=0, absolute_stddev=0),
            "at least one",
        ),
        (HybridProposalConfig(ProposalBudget(4, gradient_updates=0)), "at least one"),
        (HybridProposalConfig(ProposalBudget(3, gradient_updates=2)), "must include"),
    ],
)
def test_invalid_proposal_configs_are_rejected(config: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_parameter_tree_boundary_rejects_nonfinite_and_integer_leaves() -> None:
    def objective(candidate: object) -> jax.Array:
        return jnp.sum(candidate["x"] ** 2)

    config = SamplingProposalConfig(ProposalBudget(2))
    with pytest.raises(ValueError, match="finite"):
        run_sampling_only_proposal({"x": jnp.array([jnp.nan])}, objective, config)
    with pytest.raises(TypeError, match="floating"):
        run_sampling_only_proposal({"x": jnp.array([1], dtype=jnp.int32)}, objective, config)


def test_budget_matcher_rejects_different_requested_budgets_and_singletons() -> None:
    params = {"x": jnp.array([1.0])}

    def objective(candidate: object) -> jax.Array:
        return jnp.sum(candidate["x"] ** 2)

    first = run_sampling_only_proposal(
        params, objective, SamplingProposalConfig(ProposalBudget(2), seed=1)
    )
    second = run_sampling_only_proposal(
        params, objective, SamplingProposalConfig(ProposalBudget(3), seed=1)
    )
    with pytest.raises(ValueError, match="at least two"):
        require_matched_objective_budget(first)
    with pytest.raises(ValueError, match="matched"):
        require_matched_objective_budget(first, second)


def test_bptt_only_charges_exactly_one_objective_per_gradient_without_a_hidden_final_call() -> None:
    params = {"x": jnp.array([2.0, -1.0])}
    target = jnp.array([0.25, -0.5])
    result = run_bptt_only_proposal(
        params, _quadratic_bptt(), (target,), ProposalBudget(4, gradient_updates=4)
    )

    assert result.accounting.requested_objective_evaluations == 4
    assert result.accounting.actual_objective_evaluations == 4
    assert result.accounting.gradient_evaluations == 4
    assert result.accounting.final_evaluations == 0
    assert result.gradient_losses.shape == (4,)
    assert result.selected_loss == pytest.approx(float(result.gradient_losses[-1]))
    assert np.all(result.gradient_update_accepted)


def test_bptt_only_rejects_an_unmatched_charged_budget() -> None:
    with pytest.raises(ValueError, match="objective_evaluations == gradient_updates"):
        run_bptt_only_proposal(
            {"x": jnp.array([1.0])},
            _quadratic_bptt(),
            (jnp.array([0.0]),),
            ProposalBudget(4, gradient_updates=3),
        )


def test_sampling_projection_is_applied_before_scoring_and_selection() -> None:
    params = {"adaptive": jnp.array([1.0]), "fixed": jnp.array([3.0])}

    def project(candidate: object) -> object:
        return {"adaptive": candidate["adaptive"], "fixed": params["fixed"]}

    result = run_sampling_only_proposal(
        params,
        lambda candidate: jnp.sum(candidate["adaptive"] ** 2) + jnp.sum(candidate["fixed"] ** 2),
        SamplingProposalConfig(ProposalBudget(8), seed=5, relative_stddev=0.5, absolute_stddev=0.2),
        project_params=project,
    )

    np.testing.assert_array_equal(result.params["fixed"], params["fixed"])
    assert np.all(np.isfinite(result.candidate_losses))


def test_sampling_draws_are_nested_prefixes_across_evaluation_budgets() -> None:
    params = {"x": jnp.array([1.0, -0.5])}
    objective = lambda candidate: jnp.sum(candidate["x"] ** 2)  # noqa: E731
    small = run_sampling_only_proposal(
        params,
        objective,
        SamplingProposalConfig(
            ProposalBudget(4), seed=19, relative_stddev=0.2, absolute_stddev=0.1
        ),
    )
    large = run_sampling_only_proposal(
        params,
        objective,
        SamplingProposalConfig(
            ProposalBudget(10), seed=19, relative_stddev=0.2, absolute_stddev=0.1
        ),
    )

    np.testing.assert_array_equal(small.candidate_losses, large.candidate_losses[:4])
