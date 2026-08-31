"""Auditable minimum-intervention projections onto small affine polytopes.

The Version-A DA-PLCBF filter has four wrench inputs and only a handful of affine faces (eight
motor-force bounds and, normally, one PL-CBF face).  At that size, exhaustive active-set
enumeration is preferable to hiding a large general-purpose optimizer behind the safety filter:
every candidate and every optimality residual has a direct interpretation.
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax import Array


class PolytopeQPResult(NamedTuple):
    """Result and KKT audit record for an affine-polytope projection.

    ``input_valid`` checks finite data, a positive-definite weight, and nonnegative tolerances.
    ``feasible`` is true only when a numerically valid KKT candidate satisfies every face.  An
    invalid or infeasible problem returns a NaN action and infinite objective/residuals; this is a
    deliberate fail-closed sentinel, not a clipped control that could be mistaken for a solution.

    Multipliers correspond to constraints in ``matrix @ action <= upper_bound`` and are therefore
    nonnegative at a KKT point.  Residuals use the original, unnormalised constraints.
    """

    action: Array
    feasible: Array
    input_valid: Array
    objective: Array
    active_mask: Array
    active_count: Array
    multipliers: Array
    primal_residual: Array
    dual_residual: Array
    stationarity_residual: Array
    complementarity_residual: Array


def _check_shapes(nominal: Array, weight: Array, matrix: Array, upper_bound: Array) -> None:
    if nominal.ndim != 1:
        raise ValueError(f"nominal must be one-dimensional, got shape {nominal.shape}")
    dimension = nominal.shape[0]
    if dimension == 0:
        raise ValueError("nominal must contain at least one action dimension")
    if not jnp.issubdtype(nominal.dtype, jnp.floating):
        raise ValueError(f"nominal must have a floating-point dtype, got {nominal.dtype}")
    if weight.shape not in ((dimension,), (dimension, dimension)):
        raise ValueError(
            "weight must be a diagonal vector or square matrix matching nominal; "
            f"got nominal {nominal.shape} and weight {weight.shape}"
        )
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError(f"matrix must have shape (constraints, {dimension}), got {matrix.shape}")
    if upper_bound.shape != (matrix.shape[0],):
        raise ValueError(
            "upper_bound must have one entry per constraint; "
            f"got matrix {matrix.shape} and upper_bound {upper_bound.shape}"
        )


def _weight_matrix_and_validity(
    weight: Array, dimension: int, rank_tolerance: Array
) -> tuple[Array, Array]:
    """Return a sanitised SPD weight and a scalar validity flag."""
    identity = jnp.eye(dimension, dtype=weight.dtype)
    if weight.ndim == 1:
        finite = jnp.all(jnp.isfinite(weight))
        positive = jnp.all(weight > 0)
        valid = finite & positive
        safe_diagonal = jnp.where(valid, weight, jnp.ones_like(weight))
        return jnp.diag(safe_diagonal), valid

    finite = jnp.all(jnp.isfinite(weight))
    finite_weight = jnp.where(jnp.isfinite(weight), weight, identity)
    scale = jnp.maximum(
        jnp.max(jnp.abs(finite_weight)), jnp.asarray(jnp.finfo(weight.dtype).tiny, weight.dtype)
    )
    symmetry_residual = jnp.max(jnp.abs(finite_weight - finite_weight.T))
    symmetric = symmetry_residual <= rank_tolerance * scale
    symmetric_weight = 0.5 * (finite_weight + finite_weight.T)
    eigenvalues = jnp.linalg.eigvalsh(symmetric_weight)
    eigenvalue_scale = jnp.maximum(
        jnp.max(jnp.abs(eigenvalues)), jnp.asarray(jnp.finfo(weight.dtype).tiny, weight.dtype)
    )
    positive_definite = jnp.min(eigenvalues) > rank_tolerance * eigenvalue_scale
    valid = finite & symmetric & positive_definite
    return jnp.where(valid, symmetric_weight, identity), valid


def _constraint_maximum(values: Array) -> Array:
    """Maximum over constraints, with zero as the empty-set value."""
    if values.shape[-1] == 0:
        return jnp.zeros(values.shape[:-1], dtype=values.dtype)
    return jnp.max(values, axis=-1)


def project_affine_polytope(
    nominal: Array,
    weight: Array,
    matrix: Array,
    upper_bound: Array,
    *,
    tolerance: float = 2e-6,
    rank_tolerance: float = 1e-7,
) -> PolytopeQPResult:
    r"""Project ``nominal`` onto ``matrix @ action <= upper_bound``.

    The objective is

    .. math::
        \tfrac12 (u-u_\mathrm{nom})^T W (u-u_\mathrm{nom}).

    ``weight`` can be a positive diagonal vector or a symmetric positive-definite matrix.  The
    solver enumerates every linearly independent active set of size at most the action dimension.
    A strictly convex QP always has a KKT representation in that collection, even when more faces
    meet at the optimum.  Constraint rows are normalised internally for numerical conditioning;
    the returned multipliers and residuals are converted back to the caller's original scaling.

    The candidate collection is static for fixed input shapes, so the implementation is compatible
    with :func:`jax.jit`.  It is intended for small safety-filter problems, not large generic QPs.

    Args:
        nominal: Unfiltered action, shape ``(d,)``.
        weight: Positive diagonal, shape ``(d,)``, or full SPD matrix, shape ``(d, d)``.
        matrix: Affine constraint normals, shape ``(m, d)``.
        upper_bound: Constraint upper bounds, shape ``(m,)``.
        tolerance: Feasibility and nonnegative-multiplier tolerance after row normalisation.
        rank_tolerance: Relative threshold for SPD and active-row independence checks.

    Returns:
        The projected action and a complete primal/dual KKT audit record.

    Raises:
        ValueError: If static array shapes violate the API contract.
    """
    _check_shapes(nominal, weight, matrix, upper_bound)
    weight = jnp.asarray(weight, dtype=nominal.dtype)
    matrix = jnp.asarray(matrix, dtype=nominal.dtype)
    upper_bound = jnp.asarray(upper_bound, dtype=nominal.dtype)
    dimension = nominal.shape[0]
    constraint_count = matrix.shape[0]

    tolerance_array = jnp.asarray(tolerance, dtype=nominal.dtype)
    rank_tolerance_array = jnp.asarray(rank_tolerance, dtype=nominal.dtype)
    tolerances_valid = (
        jnp.isfinite(tolerance_array)
        & jnp.isfinite(rank_tolerance_array)
        & (tolerance_array >= 0)
        & (rank_tolerance_array > 0)
    )

    nominal_finite = jnp.all(jnp.isfinite(nominal))
    constraints_finite = jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.isfinite(upper_bound))
    safe_nominal = jnp.where(jnp.isfinite(nominal), nominal, jnp.zeros_like(nominal))
    safe_matrix = jnp.where(jnp.isfinite(matrix), matrix, jnp.zeros_like(matrix))
    safe_upper_bound = jnp.where(
        jnp.isfinite(upper_bound), upper_bound, jnp.zeros_like(upper_bound)
    )
    weight_matrix, weight_valid = _weight_matrix_and_validity(
        weight, dimension, rank_tolerance_array
    )
    input_valid = nominal_finite & constraints_finite & weight_valid & tolerances_valid
    # The validated weight is SPD.  Cholesky inversion is both cheaper and materially more stable
    # than a generic GPU LU here; the latter can lose enough float32 accuracy to reject a valid
    # four-face projection at the safety tolerance.
    weight_cholesky = jnp.linalg.cholesky(weight_matrix)
    weight_inverse = jsp_linalg.cho_solve(
        (weight_cholesky, True), jnp.eye(dimension, dtype=nominal.dtype)
    )

    row_norm = jnp.linalg.norm(safe_matrix, axis=-1)
    nonzero_row = row_norm > jnp.finfo(nominal.dtype).eps
    safe_row_norm = jnp.where(nonzero_row, row_norm, jnp.ones_like(row_norm))
    normalised_matrix = safe_matrix / safe_row_norm[:, None]
    normalised_bound = safe_upper_bound / safe_row_norm

    candidate_actions: list[Array] = [safe_nominal[None, :]]
    candidate_objectives: list[Array] = [jnp.zeros((1,), dtype=nominal.dtype)]
    candidate_masks: list[Array] = [jnp.zeros((1, constraint_count), dtype=bool)]
    candidate_multipliers: list[Array] = [jnp.zeros((1, constraint_count), dtype=nominal.dtype)]
    unconstrained_residual = (
        jnp.einsum("md,d->m", normalised_matrix, safe_nominal, precision=jax.lax.Precision.HIGHEST)
        - normalised_bound
    )
    candidate_feasible: list[Array] = [
        (_constraint_maximum(unconstrained_residual[None, :]) <= tolerance_array)
    ]

    for active_count in range(1, min(dimension, constraint_count) + 1):
        index_tuples = tuple(itertools.combinations(range(constraint_count), active_count))
        indices = jnp.asarray(index_tuples, dtype=jnp.int32)
        active_matrix = normalised_matrix[indices]
        active_bound = normalised_bound[indices]

        gram = jnp.einsum(
            "nki,ij,nlj->nkl",
            active_matrix,
            weight_inverse,
            active_matrix,
            precision=jax.lax.Precision.HIGHEST,
        )
        equality_delta = active_bound - jnp.einsum(
            "nkd,d->nk", active_matrix, safe_nominal, precision=jax.lax.Precision.HIGHEST
        )
        # Solve the equality-constrained projection through the positive-definite Gram system
        # rather than a batched indefinite KKT LU.  CUDA's batched LU may let one singular
        # enumerated KKT system contaminate otherwise independent systems in the same batch.  The
        # Gram form is algebraically identical for a full-rank set:
        #
        #   delta = W^-1 A^T (A W^-1 A^T)^-1 (b - A u_nom)
        #   lambda = -(A W^-1 A^T)^-1 (b - A u_nom).
        #
        # Float32 solves cannot reliably distinguish a row set whose relative Gram eigenvalue is
        # only a handful of machine epsilons from zero.  Apply a dtype-derived lower bound in
        # addition to the caller's stricter optional threshold.  Rejected systems become identity
        # systems before the batched solve, so no singular member reaches the backend solver.
        eigenvalues = jnp.linalg.eigvalsh(gram)
        eigenvalue_scale = jnp.maximum(
            jnp.max(jnp.abs(eigenvalues), axis=-1),
            jnp.asarray(jnp.finfo(nominal.dtype).tiny, nominal.dtype),
        )
        effective_rank_tolerance = jnp.maximum(
            rank_tolerance_array, jnp.asarray(64.0 * jnp.finfo(nominal.dtype).eps, nominal.dtype)
        )
        full_rank = jnp.min(eigenvalues, axis=-1) > effective_rank_tolerance * eigenvalue_scale
        safe_gram = jnp.where(
            full_rank[:, None, None], gram, jnp.eye(active_count, dtype=nominal.dtype)[None, ...]
        )
        safe_rhs = jnp.where(full_rank[:, None], equality_delta, jnp.zeros_like(equality_delta))
        gram_solution = jnp.linalg.solve(safe_gram, safe_rhs[..., None])[..., 0]

        def refine(_: int, solution: Array) -> Array:
            residual = safe_rhs - jnp.einsum(
                "nij,nj->ni", safe_gram, solution, precision=jax.lax.Precision.HIGHEST
            )
            correction = jnp.linalg.solve(safe_gram, residual[..., None])[..., 0]
            return solution + correction

        gram_solution = jax.lax.fori_loop(0, 2, refine, gram_solution)
        gram_solution = jnp.where(full_rank[:, None], gram_solution, jnp.zeros_like(gram_solution))

        # Refining only the Gram equation does not remove the final float32 cancellation in
        # ``u_nom + W^-1 A^T y``.  Correct residuals measured on reconstructed actions and retain
        # the best iterate instead of assuming the final float32 refinement is monotone.  Four
        # fixed iterations cover the observed RTX-4090 worst case while preserving a static JIT
        # graph and without relaxing a constraint or tolerance.
        def refine_action(_: int, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            solution, best_solution, best_error = carry
            candidate_delta = jnp.einsum(
                "ij,nkj,nk->ni",
                weight_inverse,
                active_matrix,
                solution,
                precision=jax.lax.Precision.HIGHEST,
            )
            candidate_action = safe_nominal + candidate_delta
            equality_residual = (
                jnp.einsum(
                    "nkd,nd->nk",
                    active_matrix,
                    candidate_action,
                    precision=jax.lax.Precision.HIGHEST,
                )
                - active_bound
            )
            error = jnp.max(jnp.abs(equality_residual), axis=-1)
            error = jnp.where(full_rank, error, jnp.inf)
            improved = error < best_error
            best_solution = jnp.where(improved[:, None], solution, best_solution)
            best_error = jnp.where(improved, error, best_error)
            correction = jnp.linalg.solve(
                safe_gram, jnp.where(full_rank[:, None], -equality_residual, 0.0)[..., None]
            )[..., 0]
            return solution + correction, best_solution, best_error

        initial_error = jnp.full((gram_solution.shape[0],), jnp.inf, dtype=nominal.dtype)
        _, gram_solution, _ = jax.lax.fori_loop(
            0, 4, refine_action, (gram_solution, gram_solution, initial_error)
        )
        deltas = jnp.einsum(
            "ij,nkj,nk->ni",
            weight_inverse,
            active_matrix,
            gram_solution,
            precision=jax.lax.Precision.HIGHEST,
        )
        actions = safe_nominal + deltas
        active_multipliers = -gram_solution

        all_residuals = (
            jnp.einsum("md,nd->nm", normalised_matrix, actions, precision=jax.lax.Precision.HIGHEST)
            - normalised_bound
        )
        primal_ok = _constraint_maximum(all_residuals) <= tolerance_array
        dual_ok = jnp.min(active_multipliers, axis=-1) >= -tolerance_array
        finite_candidate = jnp.all(jnp.isfinite(actions), axis=-1) & jnp.all(
            jnp.isfinite(active_multipliers), axis=-1
        )
        deltas = actions - safe_nominal
        objectives = 0.5 * jnp.einsum(
            "ni,ij,nj->n", deltas, weight_matrix, deltas, precision=jax.lax.Precision.HIGHEST
        )

        masks = jnp.sum(jax.nn.one_hot(indices, constraint_count, dtype=jnp.int32), axis=1).astype(
            bool
        )
        scaled_active_multipliers = active_multipliers / safe_row_norm[indices]
        multipliers = jnp.sum(
            jax.nn.one_hot(indices, constraint_count, dtype=nominal.dtype)
            * scaled_active_multipliers[..., None],
            axis=1,
        )

        candidate_actions.append(actions)
        candidate_objectives.append(objectives)
        candidate_masks.append(masks)
        candidate_multipliers.append(multipliers)
        candidate_feasible.append(full_rank & primal_ok & dual_ok & finite_candidate)

    actions = jnp.concatenate(candidate_actions, axis=0)
    objectives = jnp.concatenate(candidate_objectives, axis=0)
    masks = jnp.concatenate(candidate_masks, axis=0)
    multipliers = jnp.concatenate(candidate_multipliers, axis=0)
    feasible_candidates = jnp.concatenate(candidate_feasible, axis=0) & input_valid

    best_index = jnp.argmin(jnp.where(feasible_candidates, objectives, jnp.inf))
    feasible = jnp.any(feasible_candidates)
    selected_action = actions[best_index]
    selected_objective = objectives[best_index]
    selected_mask = masks[best_index]
    selected_multipliers = multipliers[best_index]

    raw_constraint_residuals = (
        jnp.einsum("md,d->m", safe_matrix, selected_action, precision=jax.lax.Precision.HIGHEST)
        - safe_upper_bound
    )
    stationarity = jnp.einsum(
        "ij,j->i",
        weight_matrix,
        selected_action - safe_nominal,
        precision=jax.lax.Precision.HIGHEST,
    ) + jnp.einsum(
        "mi,m->i", safe_matrix, selected_multipliers, precision=jax.lax.Precision.HIGHEST
    )
    primal_residual = jnp.maximum(_constraint_maximum(raw_constraint_residuals), 0)
    dual_residual = jnp.maximum(_constraint_maximum(-selected_multipliers), 0)
    stationarity_residual = jnp.max(jnp.abs(stationarity))
    complementarity_residual = _constraint_maximum(
        jnp.abs(selected_multipliers * raw_constraint_residuals)
    )

    nan_action = jnp.full_like(nominal, jnp.nan)
    return PolytopeQPResult(
        action=jnp.where(feasible, selected_action, nan_action),
        feasible=feasible,
        input_valid=input_valid,
        objective=jnp.where(feasible, selected_objective, jnp.inf),
        active_mask=jnp.where(feasible, selected_mask, jnp.zeros_like(selected_mask)),
        active_count=jnp.where(
            feasible, jnp.sum(selected_mask, dtype=jnp.int32), jnp.asarray(0, jnp.int32)
        ),
        multipliers=jnp.where(feasible, selected_multipliers, jnp.zeros_like(selected_multipliers)),
        primal_residual=jnp.where(feasible, primal_residual, jnp.inf),
        dual_residual=jnp.where(feasible, dual_residual, jnp.inf),
        stationarity_residual=jnp.where(feasible, stationarity_residual, jnp.inf),
        complementarity_residual=jnp.where(feasible, complementarity_residual, jnp.inf),
    )


__all__ = ["PolytopeQPResult", "project_affine_polytope"]
