"""Differentiable Adaptive Policy-Library Control Barrier Functions.

The package keeps smooth training surrogates separate from hard sampled values used for runtime
selection and validation.  Passing a hard check certifies only the configured finite horizon and
scenario batch; it is not an infinite-horizon or distribution-free guarantee.

Public objects are imported lazily so an offline artifact/video submodule does not additionally
initialize the DA-PLCBF training stack before spawning the pinned video encoder.  Crazyflow's
top-level package may still import JAX as part of its existing environment registration contract.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BoxHalfspaceResult",
    "BPTTFunctions",
    "BPTTState",
    "BPTTStepMetrics",
    "CircleScenarioBatch",
    "FalsificationConfig",
    "FalsificationResult",
    "FalsificationStage",
    "FalsificationStatus",
    "LibraryLossConfig",
    "LibraryLossMetrics",
    "ReferenceFilterResult",
    "RolloutBatch",
    "RolloutConfig",
    "conservative_softmin",
    "build_bptt_functions",
    "box_halfspace_fraction_2d",
    "boundary_candidate_set_sha256",
    "double_integrator_step",
    "hard_policy_margins",
    "library_loss",
    "load_falsification_result",
    "project_box_halfspace",
    "reference_plcbf_filter",
    "run_fixed_budget_falsification",
    "rollout_structured_library",
    "swept_trajectory_constraints",
    "save_falsification_result",
    "training_policy_margins",
    "trajectory_constraints",
    "tree_all_finite",
    "verify_falsification_replay",
]


_EXPORTS = {
    "BPTTFunctions": ("bptt", "BPTTFunctions"),
    "BPTTState": ("bptt", "BPTTState"),
    "BPTTStepMetrics": ("bptt", "BPTTStepMetrics"),
    "BoxHalfspaceResult": ("qp", "BoxHalfspaceResult"),
    "CircleScenarioBatch": ("types", "CircleScenarioBatch"),
    "FalsificationConfig": ("falsification", "FalsificationConfig"),
    "FalsificationResult": ("falsification", "FalsificationResult"),
    "FalsificationStage": ("falsification", "FalsificationStage"),
    "FalsificationStatus": ("falsification", "FalsificationStatus"),
    "LibraryLossConfig": ("config", "LibraryLossConfig"),
    "LibraryLossMetrics": ("losses", "LibraryLossMetrics"),
    "ReferenceFilterResult": ("reference_filter", "ReferenceFilterResult"),
    "RolloutBatch": ("rollouts", "RolloutBatch"),
    "RolloutConfig": ("config", "RolloutConfig"),
    "box_halfspace_fraction_2d": ("reference_filter", "box_halfspace_fraction_2d"),
    "boundary_candidate_set_sha256": ("falsification", "boundary_candidate_set_sha256"),
    "build_bptt_functions": ("bptt", "build_bptt_functions"),
    "conservative_softmin": ("values", "conservative_softmin"),
    "double_integrator_step": ("double_integrator", "double_integrator_step"),
    "hard_policy_margins": ("values", "hard_policy_margins"),
    "library_loss": ("losses", "library_loss"),
    "load_falsification_result": ("falsification", "load_falsification_result"),
    "project_box_halfspace": ("qp", "project_box_halfspace"),
    "reference_plcbf_filter": ("reference_filter", "reference_plcbf_filter"),
    "run_fixed_budget_falsification": ("falsification", "run_fixed_budget_falsification"),
    "rollout_structured_library": ("rollouts", "rollout_structured_library"),
    "swept_trajectory_constraints": ("values", "swept_trajectory_constraints"),
    "save_falsification_result": ("falsification", "save_falsification_result"),
    "training_policy_margins": ("values", "training_policy_margins"),
    "trajectory_constraints": ("values", "trajectory_constraints"),
    "tree_all_finite": ("bptt", "tree_all_finite"),
    "verify_falsification_replay": ("falsification", "verify_falsification_replay"),
}


def __getattr__(name: str) -> Any:
    """Load one public object on first access without eagerly importing JAX."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public objects in interactive discovery."""
    return sorted((*globals(), *__all__))
