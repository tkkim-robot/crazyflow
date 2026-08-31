"""Matched, descriptive evidence for DA-PLCBF Version A and Version B.

This module is intentionally separate from the seven-method scientific campaign.  Version A is
an airborne direct-wrench, control-affine model; Version B executes the Crazyflow force/torque
controller, allocation and clipping, rotor lag, first-principles dynamics, and integrator.  The
two implementations therefore cannot share a theorem or an interchangeable safety guarantee.

The bounded protocol below gives both implementations the same observable initial state, static
safety geometry, waypoint target, nominal wrench intent, decision duration, shared fallback
library, and certificate horizon.  It records all continuous-QP/KKT evidence from Version A and
all exact nonlinear/full-stack postchecks from Version B.  Comparisons are descriptive only.
Operational failures are retained as scheduled outcomes instead of being dropped.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from crazyflow.control import Control
from crazyflow.dynamics import Dynamics
from crazyflow.safety.da_plcbf.actor import (
    SharedActorConfig,
    SharedActorSpec,
    initialize_shared_actor,
)
from crazyflow.safety.da_plcbf.full_stack import build_unclipped_full_stack_step
from crazyflow.safety.da_plcbf.library import build_shared_quad_library_spec
from crazyflow.safety.da_plcbf.quad_actor_losses import rigid_body_safety_batch_from_circles
from crazyflow.safety.da_plcbf.quad_policy import QuadPolicyConfig, waypoint_nominal_wrench
from crazyflow.safety.da_plcbf.types import CircleScenarioBatch
from crazyflow.safety.da_plcbf.version_a_barriers import VersionABarrierConfig, VersionAModel
from crazyflow.safety.da_plcbf.version_a_filter import VersionAActuator, VersionAFilterConfig
from crazyflow.safety.da_plcbf.version_a_runtime import version_a_runtime_step
from crazyflow.safety.da_plcbf.version_b_runtime import (
    VersionBRuntimeConfig,
    replace_version_b_state,
    sim_data_to_version_b_state,
    version_b_runtime_step,
)
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from jax import Array


VERSION_COMPARISON_SCHEMA_VERSION = 2
VERSION_COMPARISON_KIND = "crazyflow.da_plcbf.matched-version-a-version-b.v1"
CASE_GENERATOR_VERSION = "matched-static-sphere-v1"
COMMAND_READY_TIMING_SCOPE = (
    "compiled device decision through synchronized applied-action acceptance leaves and a host "
    "finite-action guard; excludes compilation and artifact serialization"
)
FIXED_CASE_COUNT = 3
MINIMUM_FINAL_RANDOMIZED_CASES = 100
INTENDED_POLICY_COUNT = 64
INTENDED_CERTIFICATE_HORIZON = 50
CORE_CAMPAIGN_SUBSTEPS = 10
SHORT_INTERVAL_SUBSTEPS = 2


@dataclass(frozen=True, slots=True)
class _SourceState:
    """Exact executable-source and Git state bracketing one evidence run."""

    source_tree_sha256: str
    git_commit: str | None
    git_status: str | None

    @property
    def dirty(self) -> bool | None:
        """Return fail-closed cleanliness when Git state is available."""
        return None if self.git_status is None else bool(self.git_status)


_ROOT_KEYS = {
    "schema_version",
    "artifact_kind",
    "protocol",
    "provenance",
    "claim_boundary",
    "cases",
    "full_shape_preflight",
    "short_interval_shape_probe",
    "summary",
    "content_sha256",
}
_PROTOCOL_KEYS = {
    "profile",
    "case_generator_version",
    "root_seed",
    "fixed_case_count",
    "randomized_case_count",
    "total_case_count",
    "case_set_sha256",
    "sim_frequency_hz",
    "decision_dt_seconds",
    "n_substeps",
    "certificate_horizon",
    "policy_count",
    "policy_gain",
    "version_a_policy_alpha",
    "version_b_decay",
    "version_a_interval_tolerance",
    "version_b_tolerance",
    "version_b_qp_iterations",
    "timing_scope",
    "matching_contract",
}
_MATCHING_KEYS = {
    "observable_initial_state",
    "static_safety_geometry",
    "waypoint_target",
    "nominal_wrench_intent",
    "decision_duration",
    "certificate_horizon",
    "shared_policy_library",
    "unmatched_hidden_plant_state",
}
_PROVENANCE_KEYS = {
    "generated_at_utc",
    "repository",
    "git_commit",
    "git_dirty",
    "source_tree_sha256",
    "python_version",
    "host_platform",
    "jax_version",
    "jaxlib_version",
    "numpy_version",
    "crazyflow_version",
    "requested_device",
    "jax_platform",
    "device_kind",
    "device_count",
    "jax_enable_x64",
    "sim_configuration",
    "compile_seconds",
    "compile_failures",
}
_CLAIM_KEYS = {
    "finite_scope",
    "matched_quantities",
    "unmatched_quantities",
    "permitted_interpretation",
    "prohibited_interpretations",
}
_CASE_KEYS = {"matched_inputs", "version_a", "version_b", "comparison"}
_INPUT_KEYS = {
    "index",
    "case_id",
    "source",
    "condition",
    "seed",
    "state",
    "target_position",
    "target_velocity",
    "obstacle_center",
    "obstacle_radius",
    "obstacle_active",
    "arena_lower",
    "arena_upper",
    "speed_limit",
    "angular_rate_limit",
    "tilt_limit_radians",
    "nominal_wrench_intent",
    "nominal_input_valid",
}
_VERSION_A_KEYS = {
    "status",
    "failure",
    "latency_seconds",
    "action",
    "next_state",
    "nominal_wrench",
    "nominal_match_max_abs_error",
    "has_certificate",
    "selected_index",
    "selected_value",
    "qp_feasible",
    "qp_accepted",
    "qp_objective",
    "qp_primal_residual",
    "qp_dual_residual",
    "qp_stationarity_residual",
    "qp_complementarity_residual",
    "continuous_policy_residual",
    "minimum_analytic_barrier_residual",
    "minimum_motor_margin",
    "allocation_roundtrip_error",
    "allocation_identity_error",
    "applied_postcheck_passed",
    "action_executable",
    "proposal_interval_margin",
    "fallback_interval_margin",
    "applied_interval_margin",
    "proposal_discrete_residual",
    "fallback_discrete_residual",
    "applied_discrete_residual",
    "applied_discrete_applicable",
    "used_fallback",
    "used_midpoint",
    "degraded",
    "claim_eligible",
}
_VERSION_B_KEYS = {
    "status",
    "failure",
    "latency_seconds",
    "action",
    "next_state",
    "applied_motor_forces",
    "nominal_wrench",
    "nominal_match_max_abs_error",
    "has_certificate",
    "selected_index",
    "selected_value",
    "proposal_feasible",
    "proposal_accepted",
    "fallback_accepted",
    "fallback_substituted",
    "used_fallback",
    "qp_constraint_residual",
    "qp_multiplier",
    "qp_objective",
    "linearization_exact_residual",
    "proposal_exact_residual",
    "fallback_exact_residual",
    "applied_exact_residual",
    "proposal_interval_margin",
    "fallback_interval_margin",
    "applied_interval_margin",
    "proposal_actuator_residual",
    "fallback_actuator_residual",
    "applied_actuator_residual",
    "postcheck_replay_error",
    "held_replay_state_error",
    "rotor_lower_residual",
    "audit_residual",
    "command_bound_residual",
    "allocation_roundtrip_error",
    "physical_upper_residual",
    "command_committed",
    "applied_accepted",
    "degraded",
    "claim_eligible",
}
_COMPARISON_KEYS = {
    "both_executed",
    "both_claim_eligible",
    "version_a_only_claim_eligible",
    "version_b_only_claim_eligible",
    "neither_claim_eligible",
    "action_linf_difference",
    "next_observable_state_linf_difference",
    "applied_interval_margin_b_minus_a",
}
_SUMMARY_KEYS = {"scheduled_cases", "version_a", "version_b", "matched", "interpretation"}
_METHOD_SUMMARY_KEYS = {"successes", "operational_failures", "claim_eligible", "degraded"}
_MATCHED_SUMMARY_KEYS = {
    "both_successful",
    "both_claim_eligible",
    "version_a_only_claim_eligible",
    "version_b_only_claim_eligible",
    "neither_claim_eligible",
}
_PREFLIGHT_KEYS = {
    "scheduled",
    "protocol_role",
    "case_id",
    "matched_inputs",
    "policy_count",
    "certificate_horizon",
    "n_substeps",
    "actor_hidden_width",
    "structured_library",
    "compile_seconds",
    "compile_failures",
    "version_a",
    "version_b",
    "both_executed",
    "matched_acceptance_postcheck_passed",
    "version_b_integration_supported",
    "decision_deadline_seconds",
    "version_a_deadline_met",
    "version_a_deadline_ratio",
    "version_b_deadline_met",
    "version_b_deadline_ratio",
    "interpretation",
}


@dataclass(frozen=True, slots=True)
class VersionComparisonProfile:
    """Static shape and deterministic case budget for one bounded comparison."""

    name: Literal["smoke", "final"]
    randomized_case_count: int
    root_seed: int
    n_substeps: int
    certificate_horizon: int
    policy_count: int = 1
    policy_gain: float = 1.5
    version_a_policy_alpha: float = 2.0
    version_a_interval_tolerance: float = 2e-5
    version_b_tolerance: float = 2e-5
    version_b_qp_iterations: int = 32

    def validate(self) -> None:
        """Reject a profile that could be mislabeled or change compiled shapes silently."""
        if self.name not in ("smoke", "final"):
            raise ValueError("profile name must be 'smoke' or 'final'")
        expected_randomized = 0 if self.name == "smoke" else MINIMUM_FINAL_RANDOMIZED_CASES
        if self.randomized_case_count != expected_randomized:
            raise ValueError(f"{self.name} requires exactly {expected_randomized} randomized cases")
        integers = (
            self.root_seed,
            self.n_substeps,
            self.certificate_horizon,
            self.policy_count,
            self.version_b_qp_iterations,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise TypeError("profile integer settings must be integers, not booleans")
        if self.root_seed < 0 or min(integers[1:]) <= 0:
            raise ValueError("root_seed must be nonnegative and static counts must be positive")
        finite_positive = (
            self.policy_gain,
            self.version_a_policy_alpha,
            self.version_a_interval_tolerance,
            self.version_b_tolerance,
        )
        if not all(math.isfinite(value) and value > 0 for value in finite_positive):
            raise ValueError("profile floating-point settings must be finite and positive")


@dataclass(frozen=True, slots=True)
class MatchedVersionCase:
    """One deterministic matched observable state, task intent, and static geometry."""

    index: int
    case_id: str
    source: Literal["fixed", "randomized"]
    condition: str
    seed: int
    state: tuple[float, ...]
    target_position: tuple[float, float, float]
    target_velocity: tuple[float, float, float]
    obstacle_center: tuple[float, float, float]
    obstacle_radius: float
    obstacle_active: bool
    arena_lower: tuple[float, float, float] = (-4.0, -4.0, 0.1)
    arena_upper: tuple[float, float, float] = (4.0, 4.0, 4.1)
    speed_limit: float = 8.0
    angular_rate_limit: float = 20.0
    tilt_limit_radians: float = 1.4

    def validate(self) -> None:
        """Validate shapes, finite values, and portable labels before JAX sees the case."""
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("case index must be a nonnegative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("case seed must be a nonnegative integer")
        if self.source not in ("fixed", "randomized"):
            raise ValueError("case source must be fixed or randomized")
        if not self.case_id or not self.condition:
            raise ValueError("case_id and condition must be nonempty")
        expected = {
            "state": (self.state, 13),
            "target_position": (self.target_position, 3),
            "target_velocity": (self.target_velocity, 3),
            "obstacle_center": (self.obstacle_center, 3),
            "arena_lower": (self.arena_lower, 3),
            "arena_upper": (self.arena_upper, 3),
        }
        for name, (values, length) in expected.items():
            if len(values) != length or not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must contain {length} finite values")
        scalars = (
            self.obstacle_radius,
            self.speed_limit,
            self.angular_rate_limit,
            self.tilt_limit_radians,
        )
        if not all(math.isfinite(value) and value > 0 for value in scalars):
            raise ValueError("case limits and radius must be finite and positive")
        if any(lower >= upper for lower, upper in zip(self.arena_lower, self.arena_upper)):
            raise ValueError("arena lower bounds must be below upper bounds")
        quaternion = np.asarray(self.state[3:7], dtype=np.float64)
        if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-6:
            raise ValueError("state quaternion must be unit length in xyzw order")

    def generator_mapping(self) -> dict[str, Any]:
        """Return only deterministic generator fields used by the case-set digest."""
        value = asdict(self)
        for name in (
            "state",
            "target_position",
            "target_velocity",
            "obstacle_center",
            "arena_lower",
            "arena_upper",
        ):
            value[name] = list(value[name])
        return value


def comparison_profile(
    name: Literal["smoke", "final"], *, root_seed: int = 260601478
) -> VersionComparisonProfile:
    """Return the immutable smoke or final profile; final always schedules 100 random cases."""
    profile = VersionComparisonProfile(
        name=name,
        randomized_case_count=0 if name == "smoke" else MINIMUM_FINAL_RANDOMIZED_CASES,
        root_seed=root_seed,
        n_substeps=SHORT_INTERVAL_SUBSTEPS if name == "smoke" else CORE_CAMPAIGN_SUBSTEPS,
        certificate_horizon=1 if name == "smoke" else 2,
    )
    profile.validate()
    return profile


def _quaternion_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    quaternion = np.asarray(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dtype=np.float64,
    )
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def _fixed_cases() -> tuple[MatchedVersionCase, ...]:
    identity = (0.0, 0.0, 0.0, 1.0)
    return (
        MatchedVersionCase(
            index=0,
            case_id="fixed-safe-airborne",
            source="fixed",
            condition="safe_airborne",
            seed=0,
            state=(0.1, 0.2, 1.0, *identity, 0.05, -0.02, 0.1, 0.01, 0.02, -0.01),
            target_position=(0.1, 0.2, 1.0),
            target_velocity=(0.0, 0.0, 0.0),
            obstacle_center=(2.5, 2.5, 1.0),
            obstacle_radius=0.25,
            obstacle_active=True,
        ),
        MatchedVersionCase(
            index=1,
            case_id="fixed-near-obstacle-approach",
            source="fixed",
            condition="near_obstacle_approach",
            seed=1,
            state=(-0.45, 0.0, 1.0, *identity, 0.8, -0.02, 0.1, 0.01, 0.02, -0.01),
            target_position=(1.0, 0.0, 1.0),
            target_velocity=(0.0, 0.0, 0.0),
            obstacle_center=(0.0, 0.0, 1.0),
            obstacle_radius=0.25,
            obstacle_active=True,
        ),
        MatchedVersionCase(
            index=2,
            case_id="fixed-colliding-initial-state",
            source="fixed",
            condition="colliding_initial_state",
            seed=2,
            state=(0.0, 0.0, 1.0, *identity, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            target_position=(0.0, 0.0, 1.0),
            target_velocity=(0.0, 0.0, 0.0),
            obstacle_center=(0.0, 0.0, 1.0),
            obstacle_radius=0.25,
            obstacle_active=True,
        ),
    )


def _randomized_case(index: int, seed: int) -> MatchedVersionCase:
    rng = np.random.Generator(np.random.PCG64(seed))
    position = np.asarray(
        (rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.6), rng.uniform(0.55, 3.45)), dtype=np.float64
    )
    roll = float(rng.uniform(-0.35, 0.35))
    pitch = float(rng.uniform(-0.35, 0.35))
    yaw = float(rng.uniform(-math.pi, math.pi))
    quaternion = _quaternion_xyzw(roll, pitch, yaw)
    velocity = rng.uniform(-0.9, 0.9, size=3)
    rate = rng.uniform(-0.8, 0.8, size=3)
    obstacle_radius = float(rng.uniform(0.12, 0.3))
    direction = rng.normal(size=3)
    direction[2] *= 0.35
    direction /= np.linalg.norm(direction)
    mode = (index - FIXED_CASE_COUNT) % 5
    condition = (
        "random_clear",
        "random_distant_obstacle",
        "random_near_obstacle",
        "random_boundary_obstacle",
        "random_colliding_obstacle",
    )[mode]
    obstacle_active = mode != 0
    distance = (
        1.4,
        float(rng.uniform(0.8, 1.25)),
        obstacle_radius + float(rng.uniform(0.1, 0.24)),
        obstacle_radius + float(rng.uniform(0.015, 0.045)),
        obstacle_radius * float(rng.uniform(0.1, 0.8)),
    )[mode]
    obstacle = position + direction * distance
    obstacle = np.clip(obstacle, (-3.5, -3.5, 0.25), (3.5, 3.5, 3.95))
    if mode in (2, 3):
        toward = obstacle - position
        toward /= max(float(np.linalg.norm(toward)), 1e-12)
        velocity = 0.65 * toward + 0.15 * velocity
    target = np.clip(
        position + rng.uniform((-1.4, -1.4, -0.8), (1.4, 1.4, 0.8)),
        (-3.5, -3.5, 0.35),
        (3.5, 3.5, 3.85),
    )
    target_velocity = rng.uniform(-0.3, 0.3, size=3)
    state = tuple(float(value) for value in (*position, *quaternion, *velocity, *rate))
    return MatchedVersionCase(
        index=index,
        case_id=f"random-{index - FIXED_CASE_COUNT:03d}-{condition}",
        source="randomized",
        condition=condition,
        seed=seed,
        state=state,
        target_position=tuple(float(value) for value in target),
        target_velocity=tuple(float(value) for value in target_velocity),
        obstacle_center=tuple(float(value) for value in obstacle),
        obstacle_radius=obstacle_radius,
        obstacle_active=obstacle_active,
    )


def generate_matched_version_cases(
    profile: VersionComparisonProfile,
) -> tuple[MatchedVersionCase, ...]:
    """Generate the three fixed cases and the final profile's 100 deterministic random cases."""
    profile.validate()
    cases = list(_fixed_cases())
    seed_sequence = np.random.SeedSequence(profile.root_seed)
    children = seed_sequence.spawn(profile.randomized_case_count)
    for offset, child in enumerate(children):
        seed = int(child.generate_state(1, dtype=np.uint32)[0])
        cases.append(_randomized_case(FIXED_CASE_COUNT + offset, seed))
    for expected_index, case in enumerate(cases):
        case.validate()
        if case.index != expected_index:
            raise AssertionError("case generator produced a non-contiguous index")
    return tuple(cases)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _domain_sha256(domain: bytes, value: Any) -> str:
    return hashlib.sha256(domain + b"\0" + _canonical_json(value)).hexdigest()


def matched_case_set_sha256(cases: Sequence[MatchedVersionCase]) -> str:
    """Content-address the deterministic case definitions independently of runtime results."""
    return _domain_sha256(
        b"crazyflow.da_plcbf.matched-version-cases.v1", [case.generator_mapping() for case in cases]
    )


def _float_or_none(value: Any) -> float | None:
    scalar = float(np.asarray(jax.device_get(value)))
    return scalar if math.isfinite(scalar) else None


def _int_value(value: Any) -> int:
    return int(np.asarray(jax.device_get(value)))


def _bool_value(value: Any) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _finite_vector(value: Any, length: int) -> list[float]:
    array = np.asarray(jax.device_get(value), dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"expected a finite vector of shape ({length},), got {array.shape}")
    return [float(item) for item in array]


def _failure(stage: str, error: BaseException) -> dict[str, str]:
    return {"stage": stage, "type": type(error).__name__, "message": str(error)[:2000]}


def _empty_method_result(keys: set[str], stage: str, error: BaseException) -> dict[str, Any]:
    result = {key: None for key in keys}
    result.update(
        {
            "status": "operational_failure",
            "failure": _failure(stage, error),
            "latency_seconds": None,
            "claim_eligible": False,
        }
    )
    return result


def _version_a_mapping(
    result: Any, nominal: Array, latency: float, tolerance: float
) -> dict[str, Any]:
    filtered = result.continuous_filter
    applied_postcheck = result.applied_continuous_postcheck
    selected_index = _int_value(filtered.selected_index)
    selected_value = _float_or_none(result.certificates.certificates.values[selected_index])
    applied_interval = _float_or_none(result.applied_interval_margin)
    applied_discrete = _float_or_none(result.applied_discrete_residual)
    degraded = _bool_value(result.degraded)
    applicable = _bool_value(result.applied_discrete_applicable)
    executable = _bool_value(applied_postcheck.actuator_passed)
    postcheck = _bool_value(applied_postcheck.passed)
    claim_eligible = (
        not degraded
        and applicable
        and executable
        and postcheck
        and applied_interval is not None
        and applied_interval >= -tolerance
        and applied_discrete is not None
        and applied_discrete >= -tolerance
    )
    nominal_vector = _finite_vector(result.nominal.wrench, 4)
    nominal_intent = _finite_vector(nominal, 4)
    return {
        "status": "success",
        "failure": None,
        "latency_seconds": float(latency),
        "action": _finite_vector(result.action, 4),
        "next_state": _finite_vector(result.next_state, 13),
        "nominal_wrench": nominal_vector,
        "nominal_match_max_abs_error": float(
            np.max(np.abs(np.asarray(nominal_vector) - np.asarray(nominal_intent)))
        ),
        "has_certificate": _bool_value(filtered.has_certificate),
        "selected_index": selected_index,
        "selected_value": selected_value,
        "qp_feasible": _bool_value(filtered.qp_feasible),
        "qp_accepted": _bool_value(filtered.qp_accepted),
        "qp_objective": _float_or_none(filtered.qp.objective),
        "qp_primal_residual": _float_or_none(filtered.qp.primal_residual),
        "qp_dual_residual": _float_or_none(filtered.qp.dual_residual),
        "qp_stationarity_residual": _float_or_none(filtered.qp.stationarity_residual),
        "qp_complementarity_residual": _float_or_none(filtered.qp.complementarity_residual),
        "continuous_policy_residual": _float_or_none(applied_postcheck.policy_barrier_residual),
        "minimum_analytic_barrier_residual": _float_or_none(
            applied_postcheck.minimum_analytic_barrier_residual
        ),
        "minimum_motor_margin": _float_or_none(applied_postcheck.minimum_motor_margin),
        "allocation_roundtrip_error": _float_or_none(applied_postcheck.allocation_roundtrip_error),
        "allocation_identity_error": _float_or_none(
            filtered.motor_polytope.allocation_identity_error
        ),
        "applied_postcheck_passed": postcheck,
        "action_executable": executable,
        "proposal_interval_margin": _float_or_none(result.proposal_interval_margin),
        "fallback_interval_margin": _float_or_none(result.fallback_interval_margin),
        "applied_interval_margin": applied_interval,
        "proposal_discrete_residual": _float_or_none(result.proposal_discrete_residual),
        "fallback_discrete_residual": _float_or_none(result.fallback_discrete_residual),
        "applied_discrete_residual": applied_discrete,
        "applied_discrete_applicable": applicable,
        "used_fallback": _bool_value(result.used_interval_fallback),
        "used_midpoint": _bool_value(result.used_interval_midpoint),
        "degraded": degraded,
        "claim_eligible": claim_eligible,
    }


def _version_b_mapping(
    result: Any, nominal: Array, latency: float, tolerance: float
) -> dict[str, Any]:
    filtered = result.discrete_filter
    held = result.applied_evidence.held
    rollout = held.rollout
    exact = _float_or_none(result.applied_exact_residual)
    interval = _float_or_none(result.applied_evidence.evaluation.interval_margin)
    actuator_residual = _float_or_none(result.applied_evidence.evaluation.actuator_residual)
    postcheck_error = _float_or_none(result.postcheck_replay_error)
    applied_accepted = _bool_value(result.applied_accepted)
    degraded = _bool_value(result.degraded)
    claim_eligible = (
        applied_accepted
        and not degraded
        and exact is not None
        and exact >= -tolerance
        and interval is not None
        and interval >= -tolerance
        and actuator_residual is not None
        and actuator_residual <= tolerance
        and postcheck_error is not None
        and postcheck_error <= tolerance
    )
    nominal_vector = _finite_vector(nominal, 4)
    return {
        "status": "success",
        "failure": None,
        "latency_seconds": float(latency),
        "action": _finite_vector(result.action, 4),
        "next_state": _finite_vector(sim_data_to_version_b_state(result.next_data), 13),
        "applied_motor_forces": _finite_vector(result.applied_motor_forces, 4),
        "nominal_wrench": nominal_vector,
        "nominal_match_max_abs_error": 0.0,
        "has_certificate": _bool_value(result.has_certificate),
        "selected_index": _int_value(result.selected_index),
        "selected_value": _float_or_none(result.selected_value),
        "proposal_feasible": _bool_value(filtered.proposal_feasible),
        "proposal_accepted": _bool_value(filtered.proposal_accepted),
        "fallback_accepted": _bool_value(filtered.fallback_accepted),
        "fallback_substituted": _bool_value(filtered.fallback_substituted),
        "used_fallback": _bool_value(filtered.used_fallback),
        "qp_constraint_residual": _float_or_none(filtered.qp_constraint_residual),
        "qp_multiplier": _float_or_none(filtered.qp_multiplier),
        "qp_objective": _float_or_none(filtered.qp_objective),
        "linearization_exact_residual": _float_or_none(filtered.linearization_exact_residual),
        "proposal_exact_residual": _float_or_none(filtered.proposal_exact_residual),
        "fallback_exact_residual": _float_or_none(filtered.fallback_exact_residual),
        "applied_exact_residual": exact,
        "proposal_interval_margin": _float_or_none(filtered.proposal_interval_margin),
        "fallback_interval_margin": _float_or_none(filtered.fallback_interval_margin),
        "applied_interval_margin": interval,
        "proposal_actuator_residual": _float_or_none(filtered.proposal_actuator_residual),
        "fallback_actuator_residual": _float_or_none(filtered.fallback_actuator_residual),
        "applied_actuator_residual": actuator_residual,
        "postcheck_replay_error": postcheck_error,
        "held_replay_state_error": _float_or_none(held.replay_state_error),
        "rotor_lower_residual": _float_or_none(held.rotor_lower_residual),
        "audit_residual": _float_or_none(held.audit_residual),
        "command_bound_residual": _float_or_none(rollout.command_bound_residual),
        "allocation_roundtrip_error": _float_or_none(rollout.allocation_roundtrip_error),
        "physical_upper_residual": _float_or_none(rollout.physical_upper_residual),
        "command_committed": _bool_value(rollout.command_committed),
        "applied_accepted": applied_accepted,
        "degraded": degraded,
        "claim_eligible": claim_eligible,
    }


def _comparison_mapping(
    version_a: Mapping[str, Any], version_b: Mapping[str, Any]
) -> dict[str, Any]:
    both_executed = version_a["status"] == "success" and version_b["status"] == "success"
    a_claim = bool(version_a["claim_eligible"])
    b_claim = bool(version_b["claim_eligible"])
    action_difference = None
    state_difference = None
    margin_difference = None
    if both_executed:
        action_difference = float(
            np.max(np.abs(np.asarray(version_a["action"]) - np.asarray(version_b["action"])))
        )
        state_difference = float(
            np.max(
                np.abs(np.asarray(version_a["next_state"]) - np.asarray(version_b["next_state"]))
            )
        )
        margin_a = version_a["applied_interval_margin"]
        margin_b = version_b["applied_interval_margin"]
        if margin_a is not None and margin_b is not None:
            margin_difference = float(margin_b - margin_a)
    return {
        "both_executed": both_executed,
        "both_claim_eligible": a_claim and b_claim,
        "version_a_only_claim_eligible": a_claim and not b_claim,
        "version_b_only_claim_eligible": b_claim and not a_claim,
        "neither_claim_eligible": not a_claim and not b_claim,
        "action_linf_difference": action_difference,
        "next_observable_state_linf_difference": state_difference,
        "applied_interval_margin_b_minus_a": margin_difference,
    }


def _summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def method(name: str) -> dict[str, int]:
        rows = [case[name] for case in cases]
        return {
            "successes": sum(row["status"] == "success" for row in rows),
            "operational_failures": sum(row["status"] == "operational_failure" for row in rows),
            "claim_eligible": sum(bool(row["claim_eligible"]) for row in rows),
            "degraded": sum(row["status"] == "success" and bool(row["degraded"]) for row in rows),
        }

    comparisons = [case["comparison"] for case in cases]
    return {
        "scheduled_cases": len(cases),
        "version_a": method("version_a"),
        "version_b": method("version_b"),
        "matched": {
            "both_successful": sum(bool(row["both_executed"]) for row in comparisons),
            "both_claim_eligible": sum(bool(row["both_claim_eligible"]) for row in comparisons),
            "version_a_only_claim_eligible": sum(
                bool(row["version_a_only_claim_eligible"]) for row in comparisons
            ),
            "version_b_only_claim_eligible": sum(
                bool(row["version_b_only_claim_eligible"]) for row in comparisons
            ),
            "neither_claim_eligible": sum(
                bool(row["neither_claim_eligible"]) for row in comparisons
            ),
        },
        "interpretation": (
            "Descriptive matched-protocol counts only; Version A and Version B have different "
            "plants and neither row transfers a guarantee or establishes superiority."
        ),
    }


def _claim_boundary(profile: VersionComparisonProfile) -> dict[str, Any]:
    preflight_scope = (
        " The final profile additionally schedules one separately timed K=64/H=50/dt=20 ms "
        "core-campaign preflight and one labeled dt=4 ms diagnostic."
        if profile.name == "final"
        else " The smoke profile does not schedule either K=64/H=50 check."
    )
    return {
        "finite_scope": (
            f"Exactly {FIXED_CASE_COUNT + profile.randomized_case_count} deterministic cases, "
            f"H={profile.certificate_horizon}, {profile.n_substeps} 500 Hz plant substeps per "
            "decision."
            f"{preflight_scope}"
        ),
        "matched_quantities": [
            "observable 13-state initial condition (xyzw quaternion)",
            "static sphere and arena geometry",
            "waypoint target and derived nominal wrench intent",
            "decision duration, certificate horizon, policy library, and policy gain",
        ],
        "unmatched_quantities": [
            "Version B retains rotor speed, controller memory, allocation, clipping, and rotor lag",
            "Version A applies a direct wrench to a control-affine airborne plant",
        ],
        "permitted_interpretation": (
            "Per-case descriptive comparison of independently checked finite-horizon outcomes."
        ),
        "prohibited_interpretations": [
            "no infinite-horizon, distribution-free, or formal global safety proof",
            "no transfer of Version A's affine-QP guarantee to Version B",
            "no claim that aggregate acceptance counts prove one plant/filter is safer",
            "no replacement for the separate seven-method paired scientific campaign",
        ],
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _source_tree_sha256(repository: Path) -> str:
    """Hash implementation/runtime inputs while excluding docs, tests, and artifacts."""
    digest = hashlib.sha256(b"crazyflow.da_plcbf.matched-source-tree.v2\0")
    package = repository / "crazyflow"
    runtime_suffixes = frozenset({".py", ".toml", ".xml", ".stl"})
    paths = (
        [
            path
            for path in package.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() in runtime_suffixes
        ]
        if package.is_dir()
        else []
    )
    for root in (repository / "examples" / "da_plcbf", repository / "benchmark"):
        if root.is_dir():
            paths.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    paths.extend(
        path for name in ("pyproject.toml", "pixi.lock") if (path := repository / name).is_file()
    )
    for path in sorted(set(paths)):
        relative = path.relative_to(repository).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _require_current_source_digest(
    stored_digest: str, repository: str | os.PathLike[str] | None = None
) -> None:
    current_repository = _repository_root() if repository is None else Path(repository).resolve()
    if not (current_repository / "pyproject.toml").is_file():
        raise ValueError("current-source repository must contain pyproject.toml")
    if not hmac.compare_digest(stored_digest, _source_tree_sha256(current_repository)):
        raise ValueError("current source tree differs from artifact provenance")


def _git_value(repository: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _source_state(repository: Path) -> _SourceState:
    """Capture the exact source digest, revision, and complete Git worktree status."""
    return _SourceState(
        source_tree_sha256=_source_tree_sha256(repository),
        git_commit=_git_value(repository, "rev-parse", "HEAD"),
        git_status=_git_value(repository, "status", "--porcelain=v1", "--untracked-files=all"),
    )


def _require_clean_source(state: _SourceState) -> None:
    """Reject dirty or unverifiable Git state for claim-grade execution/verification."""
    if state.dirty is not False:
        detail = "unavailable" if state.dirty is None else "dirty"
        raise RuntimeError(
            f"claim-grade Version A/B evidence requires a clean source tree ({detail})"
        )


def _require_unchanged_source(before: _SourceState, after: _SourceState) -> None:
    """Reject any executable-source, commit, or Git-status change during execution."""
    if after != before:
        raise RuntimeError("source/git state changed while Version A/B evidence was executing")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _resolve_device(requested: str) -> jax.Device:
    if requested not in ("auto", "cpu", "gpu"):
        raise ValueError("device must be auto, cpu, or gpu")
    if requested == "auto":
        gpu = jax.devices("gpu") if any(d.platform == "gpu" for d in jax.devices()) else []
        return gpu[0] if gpu else jax.devices()[0]
    devices = jax.devices(requested)
    if not devices:
        raise RuntimeError(f"no JAX {requested} device is available")
    return devices[0]


def _problem(device: str, profile: VersionComparisonProfile) -> dict[str, Any]:
    resolved = _resolve_device(device)
    context = jax.default_device(resolved) if resolved is not None else nullcontext()
    with context:
        sim = Sim(
            dynamics=Dynamics.first_principles,
            control=Control.force_torque,
            integrator=Integrator.symplectic_euler,
            freq=500,
            force_torque_freq=500,
            device=resolved.platform,
            enable_mjx=False,
        )
        controller = sim.data.controls.force_torque.params
        physical = sim.data.params
        mass = physical.mass[0, 0, 0]
        gravity = physical.gravity_vec
        hover_force = mass * -gravity[2] / 4.0
        rpm2thrust = physical.rpm2thrust
        discriminant = rpm2thrust[1] ** 2 - 4.0 * rpm2thrust[2] * (rpm2thrust[0] - hover_force)
        hover_rpm = (-rpm2thrust[1] + jnp.sqrt(discriminant)) / (2.0 * rpm2thrust[2])
        data = sim.data.replace(
            states=sim.data.states.replace(
                pos=sim.data.states.pos.at[0, 0, 2].set(1.0),
                rotor_vel=jnp.full_like(sim.data.states.rotor_vel, hover_rpm),
            )
        )
        model = VersionAModel(
            mass=mass,
            gravity_vec=gravity,
            inertia=physical.J[0, 0],
            inertia_inv=physical.J_inv[0, 0],
            drag_matrix=physical.drag_matrix,
            wind_velocity=jnp.zeros(3),
            external_force=jnp.zeros(3),
            external_torque=jnp.zeros(3),
        )
        actuator = VersionAActuator(
            arm_length=controller["L"],
            thrust_to_torque=controller["thrust2torque"],
            mixing_matrix=controller["mixing_matrix"],
            thrust_min=controller["thrust_min"],
            thrust_max=controller["thrust_max"],
        )
        if profile.policy_count >= 8:
            spec = build_shared_quad_library_spec(policy_count=profile.policy_count)
            actor_config = SharedActorConfig(hidden_width=32)
        else:
            spec = SharedActorSpec(
                base_codes=jnp.zeros((profile.policy_count, 2)),
                base_desired_velocities=jnp.zeros((profile.policy_count, 3)),
                base_durations=jnp.full((profile.policy_count,), 0.2),
                adaptive_mask=jnp.zeros((profile.policy_count,), dtype=bool),
            )
            actor_config = SharedActorConfig(hidden_width=4, max_duration=0.5)
        actor_params = initialize_shared_actor(
            jax.random.key(profile.root_seed), spec, dimension=3, n_obstacles=1, config=actor_config
        )
        quad_config = QuadPolicyConfig()
        barrier_config = VersionABarrierConfig(minimum_tie_tolerance=1e-7)
        filter_config = VersionAFilterConfig(policy_alpha=profile.version_a_policy_alpha)
        decision_dt = profile.n_substeps / sim.freq
        version_b_config = VersionBRuntimeConfig(
            n_substeps=profile.n_substeps,
            certificate_horizon=profile.certificate_horizon,
            policy_gain=profile.policy_gain,
            decay=math.exp(-profile.version_a_policy_alpha * decision_dt),
            tolerance=profile.version_b_tolerance,
            qp_iterations=profile.version_b_qp_iterations,
        )
    return {
        "resolved_device": resolved,
        "sim": sim,
        "data": data,
        "model": model,
        "actuator": actuator,
        "spec": spec,
        "actor_config": actor_config,
        "actor_params": actor_params,
        "quad_config": quad_config,
        "barrier_config": barrier_config,
        "filter_config": filter_config,
        "version_b_config": version_b_config,
        "one_step": build_unclipped_full_stack_step(sim),
        "decision_dt": decision_dt,
    }


def _case_arrays(case: MatchedVersionCase, problem: Mapping[str, Any]) -> dict[str, Any]:
    dtype = problem["data"].states.pos.dtype
    state = jnp.asarray(case.state, dtype=dtype)
    target_position = jnp.asarray(case.target_position, dtype=dtype)
    target_velocity = jnp.asarray(case.target_velocity, dtype=dtype)
    scenarios = CircleScenarioBatch(
        obstacle_centers=jnp.asarray([[case.obstacle_center]], dtype=dtype),
        obstacle_radii=jnp.asarray([[case.obstacle_radius]], dtype=dtype),
        obstacle_mask=jnp.asarray([[case.obstacle_active]], dtype=bool),
        arena_lower=jnp.asarray([case.arena_lower], dtype=dtype),
        arena_upper=jnp.asarray([case.arena_upper], dtype=dtype),
        speed_limit=jnp.asarray([case.speed_limit], dtype=dtype),
    )
    safety = rigid_body_safety_batch_from_circles(
        scenarios,
        angular_rate_max=case.angular_rate_limit,
        tilt_max_radians=case.tilt_limit_radians,
    )
    data = replace_version_b_state(problem["data"], state)
    nominal = waypoint_nominal_wrench(
        state,
        target_position,
        target_velocity,
        problem["model"],
        problem["actuator"],
        problem["quad_config"],
    )
    return {
        "state": state,
        "target_position": target_position,
        "target_velocity": target_velocity,
        "scenarios": scenarios,
        "safety": safety,
        "data": data,
        "nominal": nominal.wrench,
        "nominal_input_valid": nominal.input_valid,
    }


def _matched_input_mapping(case: MatchedVersionCase, arrays: Mapping[str, Any]) -> dict[str, Any]:
    mapping = case.generator_mapping()
    mapping["nominal_wrench_intent"] = _finite_vector(arrays["nominal"], 4)
    mapping["nominal_input_valid"] = _bool_value(arrays["nominal_input_valid"])
    return mapping


def _compiled_functions(
    problem: Mapping[str, Any], profile: VersionComparisonProfile
) -> tuple[Any, Any]:
    def version_a(
        state: Array,
        target_position: Array,
        target_velocity: Array,
        scenarios: CircleScenarioBatch,
        safety: Any,
    ) -> Any:
        return version_a_runtime_step(
            state,
            target_position,
            target_velocity,
            problem["actor_params"],
            problem["spec"],
            scenarios,
            safety,
            problem["model"],
            problem["actuator"],
            problem["actor_config"],
            problem["quad_config"],
            problem["barrier_config"],
            problem["filter_config"],
            dt=problem["decision_dt"],
            certificate_horizon=profile.certificate_horizon,
            policy_gain=profile.policy_gain,
            interval_tolerance=profile.version_a_interval_tolerance,
        )

    def version_b(data: Any, nominal: Array, scenarios: CircleScenarioBatch, safety: Any) -> Any:
        return version_b_runtime_step(
            data,
            nominal,
            problem["actor_params"],
            problem["spec"],
            scenarios,
            safety,
            problem["model"],
            problem["actuator"],
            problem["actor_config"],
            problem["quad_config"],
            problem["barrier_config"],
            problem["one_step"],
            jnp.asarray((0.0, -1.0, -1.0, -1.0), dtype=nominal.dtype),
            jnp.asarray((10.0, 1.0, 1.0, 1.0), dtype=nominal.dtype),
            jnp.ones((4,), dtype=nominal.dtype),
            jnp.asarray((10.0, 1.0, 1.0, 1.0), dtype=nominal.dtype),
            problem["version_b_config"],
        )

    return jax.jit(version_a), jax.jit(version_b)


def _compile_one(
    function: Any, arguments: tuple[Any, ...]
) -> tuple[Any | None, float | None, dict[str, str] | None]:
    started = time.perf_counter()
    try:
        compiled = function.lower(*arguments).compile()
    except Exception as error:  # noqa: BLE001 - operational failures are evidence
        return None, None, _failure("compile", error)
    return compiled, float(time.perf_counter() - started), None


def _synchronize_command_ready(result: Any, *, version: Literal["a", "b"]) -> tuple[bool, bool]:
    """Wait for the applied-action decision and reject a nonfinite host command.

    Blocking only the returned action can omit acceptance leaves that are downstream of the final
    applied-action replay.  The synchronized tuple deliberately includes the authoritative
    acceptance/degraded result.  The host finite check is part of the timed command-ready path;
    report rendering and artifact serialization are not.
    """
    if version == "a":
        acceptance = (
            result.applied_continuous_postcheck.passed
            & result.applied_discrete_applicable
            & (~result.degraded)
        )
    elif version == "b":
        acceptance = result.applied_accepted
    else:  # pragma: no cover - Literal plus internal calls make this defensive only
        raise ValueError("version must be 'a' or 'b'")
    jax.block_until_ready((result.action, acceptance, result.degraded))
    command = np.asarray(jax.device_get(result.action), dtype=np.float64)
    if command.shape != (4,) or not np.all(np.isfinite(command)):
        raise ValueError("runtime did not produce a finite command-ready wrench of shape (4,)")
    return (
        bool(np.asarray(jax.device_get(acceptance))),
        bool(np.asarray(jax.device_get(result.degraded))),
    )


def _provenance(
    problem: Mapping[str, Any],
    requested_device: str,
    compile_seconds: Mapping[str, float | None],
    compile_failures: Mapping[str, Mapping[str, str] | None],
    source: _SourceState,
) -> dict[str, Any]:
    repository = _repository_root()
    device = problem["resolved_device"]
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "repository": str(repository),
        "git_commit": source.git_commit,
        "git_dirty": source.dirty,
        "source_tree_sha256": source.source_tree_sha256,
        "python_version": platform.python_version(),
        "host_platform": platform.platform(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "numpy_version": np.__version__,
        "crazyflow_version": _package_version("crazyflow"),
        "requested_device": requested_device,
        "jax_platform": device.platform,
        "device_kind": device.device_kind,
        "device_count": len(jax.devices(device.platform)),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "sim_configuration": {
            "dynamics": "first_principles",
            "control": "force_torque",
            "integrator": "symplectic_euler",
            "frequency_hz": 500,
            "force_torque_frequency_hz": 500,
            "floor_clamp": False,
            "mjx": False,
            "initial_rotor_state": "physical hover RPM",
        },
        "compile_seconds": dict(compile_seconds),
        "compile_failures": dict(compile_failures),
    }


def _preflight_interpretation(scheduled: bool, role: str) -> str:
    if not scheduled:
        return (
            f"The smoke profile does not schedule the {role} K=64/H=50 check; no corresponding "
            "Version-B integration observation exists, and realtime operation is not assessed."
        )
    if role == "core_campaign_intended_shape":
        return (
            "One matched K=64/H=50, dt=20 ms core-campaign decision is an exact intended-shape "
            "execution preflight only. Version-B integration support requires its exact nonlinear "
            "accepted postcheck. The deadline flag is one warm observation, not a latency "
            "distribution, and cannot establish realtime operation, safety superiority, or broad "
            "coverage."
        )
    return (
        "One matched K=64/H=50, dt=4 ms decision is retained only as a short-interval diagnostic. "
        "It is not the core campaign's dt=20 ms runtime contract and cannot support an exact "
        "intended-shape or realtime claim; its deadline flag is one warm observation only."
    )


def _unscheduled_shape_preflight(*, role: str, n_substeps: int) -> dict[str, Any]:
    return {
        "scheduled": False,
        "protocol_role": role,
        "case_id": None,
        "matched_inputs": None,
        "policy_count": INTENDED_POLICY_COUNT,
        "certificate_horizon": INTENDED_CERTIFICATE_HORIZON,
        "n_substeps": n_substeps,
        "actor_hidden_width": 32,
        "structured_library": True,
        "compile_seconds": {"version_a": None, "version_b": None},
        "compile_failures": {"version_a": None, "version_b": None},
        "version_a": None,
        "version_b": None,
        "both_executed": False,
        "matched_acceptance_postcheck_passed": False,
        "version_b_integration_supported": False,
        "decision_deadline_seconds": n_substeps / 500.0,
        "version_a_deadline_met": False,
        "version_a_deadline_ratio": None,
        "version_b_deadline_met": False,
        "version_b_deadline_ratio": None,
        "interpretation": _preflight_interpretation(False, role),
    }


def _intended_shape_profile(
    profile: VersionComparisonProfile, *, n_substeps: int
) -> VersionComparisonProfile:
    """Bind a final preflight to K=64/H=50 while preserving declared numeric tolerances."""
    intended = VersionComparisonProfile(
        name="final",
        randomized_case_count=MINIMUM_FINAL_RANDOMIZED_CASES,
        root_seed=profile.root_seed,
        n_substeps=n_substeps,
        certificate_horizon=INTENDED_CERTIFICATE_HORIZON,
        policy_count=INTENDED_POLICY_COUNT,
        policy_gain=profile.policy_gain,
        version_a_policy_alpha=profile.version_a_policy_alpha,
        version_a_interval_tolerance=profile.version_a_interval_tolerance,
        version_b_tolerance=profile.version_b_tolerance,
        version_b_qp_iterations=profile.version_b_qp_iterations,
    )
    intended.validate()
    return intended


def _run_shape_preflight(
    profile: VersionComparisonProfile, *, device: str, role: str, n_substeps: int
) -> dict[str, Any]:
    """Run one separately compiled K=64/H=50 matched shape check."""
    if profile.name != "final":
        return _unscheduled_shape_preflight(role=role, n_substeps=n_substeps)
    intended = _intended_shape_profile(profile, n_substeps=n_substeps)
    problem = _problem(device, intended)
    resolved = problem["resolved_device"]
    context = jax.default_device(resolved) if resolved is not None else nullcontext()
    case = _fixed_cases()[0]
    with context:
        arrays = _case_arrays(case, problem)
        version_a_jit, version_b_jit = _compiled_functions(problem, intended)
        version_a_compiled, compile_a, failure_a = _compile_one(
            version_a_jit,
            (
                arrays["state"],
                arrays["target_position"],
                arrays["target_velocity"],
                arrays["scenarios"],
                arrays["safety"],
            ),
        )
        version_b_compiled, compile_b, failure_b = _compile_one(
            version_b_jit,
            (arrays["data"], arrays["nominal"], arrays["scenarios"], arrays["safety"]),
        )
        if version_a_compiled is None:
            error = RuntimeError(failure_a["message"] if failure_a else "compile failed")
            version_a = _empty_method_result(_VERSION_A_KEYS, "compile", error)
        else:
            started = time.perf_counter()
            try:
                result_a = version_a_compiled(
                    arrays["state"],
                    arrays["target_position"],
                    arrays["target_velocity"],
                    arrays["scenarios"],
                    arrays["safety"],
                )
                _synchronize_command_ready(result_a, version="a")
                version_a = _version_a_mapping(
                    result_a,
                    arrays["nominal"],
                    time.perf_counter() - started,
                    intended.version_a_interval_tolerance,
                )
            except Exception as error:  # noqa: BLE001 - retained scheduled failure
                version_a = _empty_method_result(_VERSION_A_KEYS, "execute", error)
        if version_b_compiled is None:
            error = RuntimeError(failure_b["message"] if failure_b else "compile failed")
            version_b = _empty_method_result(_VERSION_B_KEYS, "compile", error)
        else:
            started = time.perf_counter()
            try:
                result_b = version_b_compiled(
                    arrays["data"], arrays["nominal"], arrays["scenarios"], arrays["safety"]
                )
                _synchronize_command_ready(result_b, version="b")
                version_b = _version_b_mapping(
                    result_b,
                    arrays["nominal"],
                    time.perf_counter() - started,
                    intended.version_b_tolerance,
                )
            except Exception as error:  # noqa: BLE001 - retained scheduled failure
                version_b = _empty_method_result(_VERSION_B_KEYS, "execute", error)
    both_executed = version_a["status"] == "success" and version_b["status"] == "success"
    version_b_supported = bool(version_b["claim_eligible"])
    version_a_deadline_met = bool(
        version_a["status"] == "success" and version_a["latency_seconds"] <= problem["decision_dt"]
    )
    version_b_deadline_met = bool(
        version_b["status"] == "success" and version_b["latency_seconds"] <= problem["decision_dt"]
    )
    return {
        "scheduled": True,
        "protocol_role": role,
        "case_id": case.case_id,
        "matched_inputs": _matched_input_mapping(case, arrays),
        "policy_count": intended.policy_count,
        "certificate_horizon": intended.certificate_horizon,
        "n_substeps": intended.n_substeps,
        "actor_hidden_width": 32,
        "structured_library": True,
        "compile_seconds": {"version_a": compile_a, "version_b": compile_b},
        "compile_failures": {"version_a": failure_a, "version_b": failure_b},
        "version_a": version_a,
        "version_b": version_b,
        "both_executed": both_executed,
        "matched_acceptance_postcheck_passed": bool(
            version_a["claim_eligible"] and version_b["claim_eligible"]
        ),
        "version_b_integration_supported": version_b_supported,
        "decision_deadline_seconds": problem["decision_dt"],
        "version_a_deadline_met": version_a_deadline_met,
        "version_a_deadline_ratio": (
            None
            if version_a["status"] != "success"
            else version_a["latency_seconds"] / problem["decision_dt"]
        ),
        "version_b_deadline_met": version_b_deadline_met,
        "version_b_deadline_ratio": (
            None
            if version_b["status"] != "success"
            else version_b["latency_seconds"] / problem["decision_dt"]
        ),
        "interpretation": _preflight_interpretation(True, role),
    }


def _run_full_shape_preflight(profile: VersionComparisonProfile, *, device: str) -> dict[str, Any]:
    """Run the exact K=64/H=50/dt=20 ms core-campaign shape check."""
    return _run_shape_preflight(
        profile,
        device=device,
        role="core_campaign_intended_shape",
        n_substeps=CORE_CAMPAIGN_SUBSTEPS,
    )


def _run_short_interval_shape_probe(
    profile: VersionComparisonProfile, *, device: str
) -> dict[str, Any]:
    """Retain a separate K=64/H=50/dt=4 ms diagnostic without relabeling it final."""
    return _run_shape_preflight(
        profile, device=device, role="short_interval_diagnostic", n_substeps=SHORT_INTERVAL_SUBSTEPS
    )


def run_matched_version_comparison(
    profile: VersionComparisonProfile, *, device: str = "auto", require_clean_source: bool = False
) -> dict[str, Any]:
    """Execute every scheduled case and return a sealed, strictly valid evidence artifact."""
    profile.validate()
    repository = _repository_root()
    source_before = _source_state(repository)
    if require_clean_source:
        _require_clean_source(source_before)
    cases = generate_matched_version_cases(profile)
    problem = _problem(device, profile)
    resolved = problem["resolved_device"]
    context = jax.default_device(resolved) if resolved is not None else nullcontext()
    with context:
        arrays = [_case_arrays(case, problem) for case in cases]
        version_a_jit, version_b_jit = _compiled_functions(problem, profile)
        first = arrays[0]
        version_a_compiled, compile_a, failure_a = _compile_one(
            version_a_jit,
            (
                first["state"],
                first["target_position"],
                first["target_velocity"],
                first["scenarios"],
                first["safety"],
            ),
        )
        version_b_compiled, compile_b, failure_b = _compile_one(
            version_b_jit, (first["data"], first["nominal"], first["scenarios"], first["safety"])
        )
        records: list[dict[str, Any]] = []
        for case, case_arrays in zip(cases, arrays):
            if version_a_compiled is None:
                error = RuntimeError(failure_a["message"] if failure_a else "compile failed")
                version_a = _empty_method_result(_VERSION_A_KEYS, "compile", error)
            else:
                started = time.perf_counter()
                try:
                    result_a = version_a_compiled(
                        case_arrays["state"],
                        case_arrays["target_position"],
                        case_arrays["target_velocity"],
                        case_arrays["scenarios"],
                        case_arrays["safety"],
                    )
                    _synchronize_command_ready(result_a, version="a")
                    latency = time.perf_counter() - started
                    version_a = _version_a_mapping(
                        result_a,
                        case_arrays["nominal"],
                        latency,
                        profile.version_a_interval_tolerance,
                    )
                except Exception as error:  # noqa: BLE001 - retained scheduled failure
                    version_a = _empty_method_result(_VERSION_A_KEYS, "execute", error)

            if version_b_compiled is None:
                error = RuntimeError(failure_b["message"] if failure_b else "compile failed")
                version_b = _empty_method_result(_VERSION_B_KEYS, "compile", error)
            else:
                started = time.perf_counter()
                try:
                    result_b = version_b_compiled(
                        case_arrays["data"],
                        case_arrays["nominal"],
                        case_arrays["scenarios"],
                        case_arrays["safety"],
                    )
                    _synchronize_command_ready(result_b, version="b")
                    latency = time.perf_counter() - started
                    version_b = _version_b_mapping(
                        result_b, case_arrays["nominal"], latency, profile.version_b_tolerance
                    )
                except Exception as error:  # noqa: BLE001 - retained scheduled failure
                    version_b = _empty_method_result(_VERSION_B_KEYS, "execute", error)

            records.append(
                {
                    "matched_inputs": _matched_input_mapping(case, case_arrays),
                    "version_a": version_a,
                    "version_b": version_b,
                    "comparison": _comparison_mapping(version_a, version_b),
                }
            )

    protocol = {
        "profile": profile.name,
        "case_generator_version": CASE_GENERATOR_VERSION,
        "root_seed": profile.root_seed,
        "fixed_case_count": FIXED_CASE_COUNT,
        "randomized_case_count": profile.randomized_case_count,
        "total_case_count": len(cases),
        "case_set_sha256": matched_case_set_sha256(cases),
        "sim_frequency_hz": 500,
        "decision_dt_seconds": problem["decision_dt"],
        "n_substeps": profile.n_substeps,
        "certificate_horizon": profile.certificate_horizon,
        "policy_count": profile.policy_count,
        "policy_gain": profile.policy_gain,
        "version_a_policy_alpha": profile.version_a_policy_alpha,
        "version_b_decay": problem["version_b_config"].decay,
        "version_a_interval_tolerance": profile.version_a_interval_tolerance,
        "version_b_tolerance": profile.version_b_tolerance,
        "version_b_qp_iterations": profile.version_b_qp_iterations,
        "timing_scope": COMMAND_READY_TIMING_SCOPE,
        "matching_contract": {
            "observable_initial_state": "identical 13-vector [pos, quat_xyzw, vel, body_rate]",
            "static_safety_geometry": "identical sphere, arena, and physical limits",
            "waypoint_target": "identical position and velocity target",
            "nominal_wrench_intent": "one Version-A waypoint wrench passed byte-identically to B",
            "decision_duration": "identical n_substeps / 500 Hz",
            "certificate_horizon": "identical decision-count horizon H",
            "shared_policy_library": "identical initialized actor parameters and policy gain",
            "unmatched_hidden_plant_state": (
                "B starts rotors at hover RPM and retains controller state; A has neither"
            ),
        },
    }
    full_shape_preflight = _run_full_shape_preflight(profile, device=device)
    short_interval_shape_probe = _run_short_interval_shape_probe(profile, device=device)
    source_after = _source_state(repository)
    _require_unchanged_source(source_before, source_after)
    if require_clean_source:
        _require_clean_source(source_after)

    document: dict[str, Any] = {
        "schema_version": VERSION_COMPARISON_SCHEMA_VERSION,
        "artifact_kind": VERSION_COMPARISON_KIND,
        "protocol": protocol,
        "provenance": _provenance(
            problem,
            device,
            {"version_a": compile_a, "version_b": compile_b},
            {"version_a": failure_a, "version_b": failure_b},
            source_after,
        ),
        "claim_boundary": _claim_boundary(profile),
        "cases": records,
        "full_shape_preflight": full_shape_preflight,
        "short_interval_shape_probe": short_interval_shape_probe,
        "summary": _summary(records),
    }
    document["content_sha256"] = _domain_sha256(
        b"crazyflow.da_plcbf.matched-version-artifact.v1", document
    )
    validate_version_comparison_artifact(document)
    return document


def _require_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} fields do not match schema: {actual}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a lowercase SHA-256 string")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    if value.lower() != value:
        raise ValueError(f"{label} must be lowercase")
    return value


def _require_finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _require_optional_finite(value: Any, label: str) -> float | None:
    return None if value is None else _require_finite(value, label)


def _require_vector(value: Any, length: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must be a list of length {length}")
    for index, item in enumerate(value):
        _require_finite(item, f"{label}[{index}]")


def _validate_failure(value: Any, label: str) -> None:
    mapping = _require_keys(value, {"stage", "type", "message"}, label)
    if not all(isinstance(mapping[name], str) and mapping[name] for name in mapping):
        raise ValueError(f"{label} strings must be nonempty")


def _validate_method(
    row: Any, keys: set[str], label: str, tolerance: float, *, version: Literal["a", "b"]
) -> None:
    mapping = _require_keys(row, keys, label)
    if mapping["status"] not in ("success", "operational_failure"):
        raise ValueError(f"{label}.status is invalid")
    if not isinstance(mapping["claim_eligible"], bool):
        raise ValueError(f"{label}.claim_eligible must be boolean")
    if mapping["status"] == "operational_failure":
        _validate_failure(mapping["failure"], f"{label}.failure")
        if mapping["claim_eligible"]:
            raise ValueError(f"{label} operational failure cannot be claim eligible")
        allowed = {"status", "failure", "claim_eligible"}
        if any(mapping[name] is not None for name in keys - allowed):
            raise ValueError(f"{label} operational failure contains fabricated metrics")
        return
    if mapping["failure"] is not None:
        raise ValueError(f"{label} successful row must not have a failure")
    _require_finite(mapping["latency_seconds"], f"{label}.latency_seconds", nonnegative=True)
    _require_vector(mapping["action"], 4, f"{label}.action")
    _require_vector(mapping["next_state"], 13, f"{label}.next_state")
    _require_vector(mapping["nominal_wrench"], 4, f"{label}.nominal_wrench")
    _require_finite(
        mapping["nominal_match_max_abs_error"],
        f"{label}.nominal_match_max_abs_error",
        nonnegative=True,
    )
    boolean_names = (
        {
            "has_certificate",
            "qp_feasible",
            "qp_accepted",
            "applied_postcheck_passed",
            "action_executable",
            "applied_discrete_applicable",
            "used_fallback",
            "used_midpoint",
            "degraded",
            "claim_eligible",
        }
        if version == "a"
        else {
            "has_certificate",
            "proposal_feasible",
            "proposal_accepted",
            "fallback_accepted",
            "fallback_substituted",
            "used_fallback",
            "command_committed",
            "applied_accepted",
            "degraded",
            "claim_eligible",
        }
    )
    if any(not isinstance(mapping[name], bool) for name in boolean_names):
        raise ValueError(f"{label} boolean diagnostics are invalid")
    if isinstance(mapping["selected_index"], bool) or not isinstance(
        mapping["selected_index"], int
    ):
        raise ValueError(f"{label}.selected_index must be an integer")
    vector_names = {"applied_motor_forces"} if version == "b" else set()
    for name in vector_names:
        _require_vector(mapping[name], 4, f"{label}.{name}")
    excluded = {
        "status",
        "failure",
        "latency_seconds",
        "action",
        "next_state",
        "nominal_wrench",
        "selected_index",
        *boolean_names,
        *vector_names,
    }
    for name in keys - excluded:
        _require_optional_finite(mapping[name], f"{label}.{name}")
    if mapping["nominal_match_max_abs_error"] > 1e-7:
        raise ValueError(f"{label} did not receive the matched nominal wrench")
    if version == "a":
        applied_interval = mapping["applied_interval_margin"]
        applied_discrete = mapping["applied_discrete_residual"]
        expected = (
            not mapping["degraded"]
            and mapping["applied_discrete_applicable"]
            and mapping["action_executable"]
            and mapping["applied_postcheck_passed"]
            and applied_interval is not None
            and applied_interval >= -tolerance
            and applied_discrete is not None
            and applied_discrete >= -tolerance
        )
    else:
        exact = mapping["applied_exact_residual"]
        interval = mapping["applied_interval_margin"]
        actuator = mapping["applied_actuator_residual"]
        postcheck = mapping["postcheck_replay_error"]
        expected = (
            mapping["applied_accepted"]
            and not mapping["degraded"]
            and exact is not None
            and exact >= -tolerance
            and interval is not None
            and interval >= -tolerance
            and actuator is not None
            and actuator <= tolerance
            and postcheck is not None
            and postcheck <= tolerance
        )
    if mapping["claim_eligible"] != expected:
        raise ValueError(f"{label}.claim_eligible disagrees with persisted postchecks")


def _validate_matched_nominal_binding(
    inputs: Mapping[str, Any],
    version_a: Mapping[str, Any],
    version_b: Mapping[str, Any],
    label: str,
) -> None:
    """Bind both successful method rows to the nominal wrench persisted as matched input."""
    intent = np.asarray(inputs["nominal_wrench_intent"], dtype=np.float64)
    for method, row in (("version_a", version_a), ("version_b", version_b)):
        if row["status"] != "success":
            continue
        actual = np.asarray(row["nominal_wrench"], dtype=np.float64)
        error = float(np.max(np.abs(actual - intent)))
        if error > 1e-7:
            raise ValueError(f"{label}.{method} nominal wrench differs from matched input")
        if not math.isclose(row["nominal_match_max_abs_error"], error, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"{label}.{method} nominal match error was not recomputed")


def _validate_shape_preflight(
    value: Any, profile: VersionComparisonProfile, *, label: str, role: str, n_substeps: int
) -> None:
    preflight = _require_keys(value, _PREFLIGHT_KEYS, label)
    if profile.name == "smoke":
        expected = _unscheduled_shape_preflight(role=role, n_substeps=n_substeps)
        if _canonical_json(preflight) != _canonical_json(expected):
            raise ValueError(f"smoke {label} must be explicitly unscheduled")
        return
    expected_static = {
        "scheduled": True,
        "protocol_role": role,
        "case_id": "fixed-safe-airborne",
        "policy_count": INTENDED_POLICY_COUNT,
        "certificate_horizon": INTENDED_CERTIFICATE_HORIZON,
        "n_substeps": n_substeps,
        "actor_hidden_width": 32,
        "structured_library": True,
        "interpretation": _preflight_interpretation(True, role),
    }
    for name, expected in expected_static.items():
        if preflight[name] != expected:
            raise ValueError(f"{label}.{name} is invalid")
    inputs = _require_keys(preflight["matched_inputs"], _INPUT_KEYS, f"{label}.matched_inputs")
    for name, expected in _fixed_cases()[0].generator_mapping().items():
        if inputs[name] != expected:
            raise ValueError(f"{label}.matched_inputs.{name} changed")
    _require_vector(inputs["nominal_wrench_intent"], 4, f"{label}.nominal_wrench_intent")
    if not isinstance(inputs["nominal_input_valid"], bool):
        raise ValueError(f"{label} nominal_input_valid must be boolean")
    compile_seconds = _require_keys(
        preflight["compile_seconds"], {"version_a", "version_b"}, f"{label}.compile_seconds"
    )
    compile_failures = _require_keys(
        preflight["compile_failures"], {"version_a", "version_b"}, f"{label}.compile_failures"
    )
    for name in ("version_a", "version_b"):
        if compile_seconds[name] is None:
            _validate_failure(compile_failures[name], f"{label}.compile_failures.{name}")
        else:
            _require_finite(
                compile_seconds[name], f"{label}.compile_seconds.{name}", nonnegative=True
            )
            if compile_failures[name] is not None:
                raise ValueError(f"{label} successful compilation also records a failure")
    _validate_method(
        preflight["version_a"],
        _VERSION_A_KEYS,
        f"{label}.version_a",
        profile.version_a_interval_tolerance,
        version="a",
    )
    _validate_method(
        preflight["version_b"],
        _VERSION_B_KEYS,
        f"{label}.version_b",
        profile.version_b_tolerance,
        version="b",
    )
    _validate_matched_nominal_binding(inputs, preflight["version_a"], preflight["version_b"], label)
    both_executed = (
        preflight["version_a"]["status"] == "success"
        and preflight["version_b"]["status"] == "success"
    )
    if preflight["both_executed"] != both_executed:
        raise ValueError(f"{label}.both_executed is inconsistent")
    both_accepted = bool(
        preflight["version_a"]["claim_eligible"] and preflight["version_b"]["claim_eligible"]
    )
    if not isinstance(preflight["matched_acceptance_postcheck_passed"], bool):
        raise ValueError(f"{label} matched acceptance/postcheck flag must be boolean")
    if preflight["matched_acceptance_postcheck_passed"] != both_accepted:
        raise ValueError(f"{label} matched acceptance/postcheck flag is inconsistent")
    if not isinstance(preflight["version_b_integration_supported"], bool):
        raise ValueError(f"{label} integration support flag must be boolean")
    if preflight["version_b_integration_supported"] != bool(
        preflight["version_b"]["claim_eligible"]
    ):
        raise ValueError(f"{label} Version-B support must equal its exact accepted postcheck")
    deadline = _require_finite(
        preflight["decision_deadline_seconds"], f"{label}.decision_deadline_seconds"
    )
    if not math.isclose(deadline, n_substeps / 500.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"{label} deadline is not n_substeps / 500 Hz")
    for method in ("version_a", "version_b"):
        deadline_name = f"{method}_deadline_met"
        ratio_name = f"{method}_deadline_ratio"
        if not isinstance(preflight[deadline_name], bool):
            raise ValueError(f"{label}.{deadline_name} must be boolean")
        row = preflight[method]
        if row["status"] == "success":
            expected_ratio = row["latency_seconds"] / deadline
            ratio = _require_finite(preflight[ratio_name], f"{label}.{ratio_name}")
            if not math.isclose(ratio, expected_ratio, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{label}.{ratio_name} is inconsistent")
            expected_met = row["latency_seconds"] <= deadline
        else:
            if preflight[ratio_name] is not None:
                raise ValueError(f"{label}.{ratio_name} must be null after failure")
            expected_met = False
        if preflight[deadline_name] != expected_met:
            raise ValueError(f"{label}.{deadline_name} is inconsistent")


def validate_version_comparison_artifact(
    document: Mapping[str, Any],
    *,
    repository: str | os.PathLike[str] | None = None,
    require_current_source: bool = False,
    require_clean_source: bool = False,
) -> None:
    """Strictly validate schema, digest, schedule, invariants, and source requirements."""
    root = _require_keys(document, _ROOT_KEYS, "artifact")
    if root["schema_version"] != VERSION_COMPARISON_SCHEMA_VERSION:
        raise ValueError("unsupported version comparison schema")
    if root["artifact_kind"] != VERSION_COMPARISON_KIND:
        raise ValueError("artifact kind is invalid")
    stored_digest = _require_sha256(root["content_sha256"], "content_sha256")
    body = dict(root)
    body.pop("content_sha256")
    computed_digest = _domain_sha256(b"crazyflow.da_plcbf.matched-version-artifact.v1", body)
    if not hmac.compare_digest(stored_digest, computed_digest):
        raise ValueError("artifact content_sha256 mismatch")

    protocol = _require_keys(root["protocol"], _PROTOCOL_KEYS, "protocol")
    _require_keys(protocol["matching_contract"], _MATCHING_KEYS, "matching_contract")
    profile = comparison_profile(protocol["profile"], root_seed=protocol["root_seed"])
    if protocol["case_generator_version"] != CASE_GENERATOR_VERSION:
        raise ValueError("unknown case generator version")
    expected_cases = generate_matched_version_cases(profile)
    expected_protocol = {
        "fixed_case_count": FIXED_CASE_COUNT,
        "randomized_case_count": profile.randomized_case_count,
        "total_case_count": len(expected_cases),
        "case_set_sha256": matched_case_set_sha256(expected_cases),
        "sim_frequency_hz": 500,
        "n_substeps": profile.n_substeps,
        "certificate_horizon": profile.certificate_horizon,
        "policy_count": profile.policy_count,
        "policy_gain": profile.policy_gain,
        "version_a_policy_alpha": profile.version_a_policy_alpha,
        "version_a_interval_tolerance": profile.version_a_interval_tolerance,
        "version_b_tolerance": profile.version_b_tolerance,
        "version_b_qp_iterations": profile.version_b_qp_iterations,
        "timing_scope": COMMAND_READY_TIMING_SCOPE,
    }
    for name, expected in expected_protocol.items():
        if protocol[name] != expected:
            raise ValueError(f"protocol.{name} disagrees with the named profile")
    dt = _require_finite(protocol["decision_dt_seconds"], "protocol.decision_dt_seconds")
    if not math.isclose(dt, profile.n_substeps / 500.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("protocol decision duration is not n_substeps / frequency")
    decay = _require_finite(protocol["version_b_decay"], "protocol.version_b_decay")
    expected_decay = math.exp(-profile.version_a_policy_alpha * dt)
    if not math.isclose(decay, expected_decay, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Version B decay does not match Version A policy_alpha and dt")

    provenance = _require_keys(root["provenance"], _PROVENANCE_KEYS, "provenance")
    provenance_source = _require_sha256(
        provenance["source_tree_sha256"], "provenance.source_tree_sha256"
    )
    if require_current_source or require_clean_source:
        _require_current_source_digest(provenance_source, repository)
    if provenance["git_dirty"] is not None and not isinstance(provenance["git_dirty"], bool):
        raise ValueError("provenance.git_dirty must be boolean or null")
    if require_clean_source:
        if provenance["git_dirty"] is not False:
            raise ValueError("artifact was not produced from a verified clean source tree")
        current_repository = (
            _repository_root() if repository is None else Path(repository).resolve()
        )
        _require_clean_source(_source_state(current_repository))
    _require_keys(
        provenance["sim_configuration"],
        {
            "dynamics",
            "control",
            "integrator",
            "frequency_hz",
            "force_torque_frequency_hz",
            "floor_clamp",
            "mjx",
            "initial_rotor_state",
        },
        "provenance.sim_configuration",
    )
    compile_seconds = _require_keys(
        provenance["compile_seconds"], {"version_a", "version_b"}, "compile_seconds"
    )
    compile_failures = _require_keys(
        provenance["compile_failures"], {"version_a", "version_b"}, "compile_failures"
    )
    for name in ("version_a", "version_b"):
        if compile_seconds[name] is None:
            _validate_failure(compile_failures[name], f"compile_failures.{name}")
        else:
            _require_finite(compile_seconds[name], f"compile_seconds.{name}", nonnegative=True)
            if compile_failures[name] is not None:
                raise ValueError("successful compilation cannot also record a failure")

    _require_keys(root["claim_boundary"], _CLAIM_KEYS, "claim_boundary")
    if root["claim_boundary"] != _claim_boundary(profile):
        raise ValueError("claim boundary was altered")
    rows = root["cases"]
    if not isinstance(rows, list) or len(rows) != len(expected_cases):
        raise ValueError("case outcome count does not match the deterministic schedule")
    for index, (row, expected_case) in enumerate(zip(rows, expected_cases)):
        case_row = _require_keys(row, _CASE_KEYS, f"cases[{index}]")
        inputs = _require_keys(case_row["matched_inputs"], _INPUT_KEYS, f"cases[{index}].inputs")
        generated = expected_case.generator_mapping()
        for name, expected in generated.items():
            if inputs[name] != expected:
                raise ValueError(f"cases[{index}].matched_inputs.{name} changed from schedule")
        _require_vector(inputs["nominal_wrench_intent"], 4, f"cases[{index}].nominal")
        if not isinstance(inputs["nominal_input_valid"], bool):
            raise ValueError("nominal_input_valid must be boolean")
        _validate_method(
            case_row["version_a"],
            _VERSION_A_KEYS,
            f"cases[{index}].version_a",
            profile.version_a_interval_tolerance,
            version="a",
        )
        _validate_method(
            case_row["version_b"],
            _VERSION_B_KEYS,
            f"cases[{index}].version_b",
            profile.version_b_tolerance,
            version="b",
        )
        _validate_matched_nominal_binding(
            inputs, case_row["version_a"], case_row["version_b"], f"cases[{index}]"
        )
        comparison = _require_keys(
            case_row["comparison"], _COMPARISON_KEYS, f"cases[{index}].comparison"
        )
        expected_comparison = _comparison_mapping(case_row["version_a"], case_row["version_b"])
        if _canonical_json(comparison) != _canonical_json(expected_comparison):
            raise ValueError(f"cases[{index}] descriptive comparison was not recomputed honestly")

    _validate_shape_preflight(
        root["full_shape_preflight"],
        profile,
        label="full_shape_preflight",
        role="core_campaign_intended_shape",
        n_substeps=CORE_CAMPAIGN_SUBSTEPS,
    )
    _validate_shape_preflight(
        root["short_interval_shape_probe"],
        profile,
        label="short_interval_shape_probe",
        role="short_interval_diagnostic",
        n_substeps=SHORT_INTERVAL_SUBSTEPS,
    )

    summary = _require_keys(root["summary"], _SUMMARY_KEYS, "summary")
    _require_keys(summary["version_a"], _METHOD_SUMMARY_KEYS, "summary.version_a")
    _require_keys(summary["version_b"], _METHOD_SUMMARY_KEYS, "summary.version_b")
    _require_keys(summary["matched"], _MATCHED_SUMMARY_KEYS, "summary.matched")
    expected_summary = _summary(rows)
    if _canonical_json(summary) != _canonical_json(expected_summary):
        raise ValueError("artifact summary disagrees with per-case outcomes")


def _read_json_strict(path: Path) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate JSON key in {path}")
        return dict(pairs)

    value = json.loads(path.read_bytes(), object_pairs_hook=hook)
    if not isinstance(value, dict):
        raise ValueError("version comparison JSON root must be an object")
    return value


def load_version_comparison_artifact(
    path: str | os.PathLike[str],
    *,
    repository: str | os.PathLike[str] | None = None,
    require_current_source: bool = True,
    require_clean_source: bool = False,
) -> dict[str, Any]:
    """Load one artifact with strict semantic and, by default, current-source validation."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    document = _read_json_strict(source)
    validate_version_comparison_artifact(
        document,
        repository=repository,
        require_current_source=require_current_source,
        require_clean_source=require_clean_source,
    )
    return document


def _atomic_write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def save_version_comparison_artifact(
    document: Mapping[str, Any],
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    repository: str | os.PathLike[str] | None = None,
    require_clean_source: bool = False,
) -> str:
    """Validate and atomically save a canonical JSON artifact; return its content digest."""
    validate_version_comparison_artifact(
        document,
        repository=repository,
        require_current_source=True,
        require_clean_source=require_clean_source,
    )
    destination = Path(path)
    _atomic_write(destination, _canonical_json(document) + b"\n", overwrite=overwrite)
    loaded = load_version_comparison_artifact(
        destination, repository=repository, require_clean_source=require_clean_source
    )
    return str(loaded["content_sha256"])


def render_version_comparison_report(document: Mapping[str, Any]) -> str:
    """Render a compact descriptive report bound to the validated JSON artifact."""
    validate_version_comparison_artifact(document)
    protocol = document["protocol"]
    summary = document["summary"]
    preflight = document["full_shape_preflight"]
    short_probe = document["short_interval_shape_probe"]

    def latency_text(method: str) -> str:
        values = sorted(
            float(row[method]["latency_seconds"])
            for row in document["cases"]
            if row[method]["status"] == "success"
        )
        if not values:
            return "unavailable"
        p95 = values[math.ceil(0.95 * len(values)) - 1]
        return (
            f"median {1000.0 * float(np.median(values)):.3f} ms, "
            f"p95 {1000.0 * p95:.3f} ms, max {1000.0 * max(values):.3f} ms"
        )

    def append_shape(lines: list[str], title: str, probe: Mapping[str, Any]) -> None:
        lines.extend((f"## {title}", ""))
        if not probe["scheduled"]:
            lines.extend(
                (
                    f"{probe['protocol_role']} K=64/H=50 was not scheduled by this smoke "
                    "profile. No integration observation exists, and realtime operation was not "
                    "assessed.",
                    "",
                )
            )
            return
        deadline_ms = 1000.0 * probe["decision_deadline_seconds"]
        integration = (
            "supported for this one case"
            if probe["version_b_integration_supported"]
            else "not supported"
        )

        def method_line(method: str, display: str, acceptance: str) -> str:
            row = probe[method]
            compile_time = probe["compile_seconds"][method]
            if row["status"] != "success":
                failure = row["failure"]
                return (
                    f"- {display}: operational failure at {failure['stage']} "
                    f"({failure['type']}: {failure['message']}); compile={compile_time}."
                )
            return (
                f"- {display}: {compile_time} s compile; "
                f"{1000.0 * row['latency_seconds']:.3f} ms warm command-ready execution "
                f"({probe[f'{method}_deadline_ratio']:.2f}× dt; deadline met: "
                f"{probe[f'{method}_deadline_met']}); {row['status']}; {acceptance}: "
                f"{row['claim_eligible']}."
            )

        lines.extend(
            (
                f"One `{probe['case_id']}` decision used K={probe['policy_count']}, "
                f"H={probe['certificate_horizon']}, {probe['n_substeps']}×500 Hz substeps "
                f"(dt={deadline_ms:.1f} ms), structured policies, and a "
                f"2x{probe['actor_hidden_width']} shared actor.",
                "",
                method_line("version_a", "Version A", "claim eligible"),
                method_line("version_b", "Version B", "exact accepted"),
                "- Matched acceptance/postchecks: "
                f"**{probe['matched_acceptance_postcheck_passed']}**.",
                f"- Correctness-only Version-B integration: **{integration}**.",
                "- Version-B observed deadline met in this one warm execution: "
                f"**{probe['version_b_deadline_met']}**.",
                "- Realtime operation: **not established**; one observation has no tail-latency "
                "distribution.",
                "",
                probe["interpretation"],
                "",
            )
        )

    lines = [
        "# Matched DA-PLCBF Version A / Version B evidence",
        "",
        f"Artifact SHA-256: `{document['content_sha256']}`",
        "",
        "This is a bounded, descriptive comparison. It does **not** transfer Version A's "
        "control-affine guarantee to Version B and does not prove either implementation safer.",
        "",
        "## Protocol",
        "",
        f"- Profile: `{protocol['profile']}`",
        f"- Cases: {protocol['total_case_count']} "
        f"({protocol['fixed_case_count']} fixed + {protocol['randomized_case_count']} randomized)",
        f"- Decision interval: {protocol['decision_dt_seconds']:.6f} s",
        f"- Shared certificate horizon: H={protocol['certificate_horizon']}",
        f"- Case-set SHA-256: `{protocol['case_set_sha256']}`",
        "",
        "## Outcomes",
        "",
        "| Outcome | Version A | Version B |",
        "|---|---:|---:|",
        f"| Successful executions | {summary['version_a']['successes']} | "
        f"{summary['version_b']['successes']} |",
        f"| Operational failures | {summary['version_a']['operational_failures']} | "
        f"{summary['version_b']['operational_failures']} |",
        f"| Independently claim-eligible cases | {summary['version_a']['claim_eligible']} | "
        f"{summary['version_b']['claim_eligible']} |",
        f"| Explicit degraded outcomes | {summary['version_a']['degraded']} | "
        f"{summary['version_b']['degraded']} |",
        "",
        f"Both independently claim-eligible: {summary['matched']['both_claim_eligible']}; "
        f"A-only: {summary['matched']['version_a_only_claim_eligible']}; "
        f"B-only: {summary['matched']['version_b_only_claim_eligible']}; "
        f"neither: {summary['matched']['neither_claim_eligible']}.",
        "",
        "Warm bounded-profile command-ready timing (synchronized wall clock; descriptive, "
        f"one `{document['provenance']['jax_platform']}` run):",
        "",
        f"Timing scope: {protocol['timing_scope']}.",
        "",
        f"- Version A: {latency_text('version_a')}",
        f"- Version B: {latency_text('version_b')}",
        "",
    ]
    append_shape(lines, "Exact core-campaign intended-shape preflight", preflight)
    append_shape(lines, "Short-interval shape diagnostic", short_probe)
    lines.extend(
        (
            "## Fixed discriminating cases",
            "",
            "| Case | Version A | Version B | A interval | B interval |",
            "|---|---|---|---:|---:|",
        )
    )
    for row in document["cases"][:FIXED_CASE_COUNT]:
        inputs = row["matched_inputs"]
        a = row["version_a"]
        b = row["version_b"]
        a_label = (
            a["status"]
            if a["status"] != "success"
            else ("eligible" if a["claim_eligible"] else "degraded/rejected")
        )
        b_label = (
            b["status"]
            if b["status"] != "success"
            else ("eligible" if b["claim_eligible"] else "degraded/rejected")
        )
        a_margin = (
            "n/a" if a["applied_interval_margin"] is None else f"{a['applied_interval_margin']:.6g}"
        )
        b_margin = (
            "n/a" if b["applied_interval_margin"] is None else f"{b['applied_interval_margin']:.6g}"
        )
        lines.append(f"| `{inputs['case_id']}` | {a_label} | {b_label} | {a_margin} | {b_margin} |")
    lines.extend(
        (
            "",
            "## Exact limitations",
            "",
            "- Version B includes hidden hover-rotor/controller state, allocation, clipping, rotor "
            "lag, and the nonlinear full stack; Version A has no corresponding hidden state.",
            "- Version A's continuous QP and Version B's nonlinear discrete postcheck are each "
            "judged only by their own persisted acceptance contract.",
            "- Aggregate counts are descriptive and are not a statistical superiority test.",
            "- The colliding fixed case checks fail-closed behavior; it is not expected to become "
            "safe in one control decision.",
            "- This artifact is separate from, and cannot substitute for, the seven-method paired "
            "campaign or long-horizon rendered videos.",
            "",
        )
    )
    return "\n".join(lines)


def save_version_comparison_report(
    document: Mapping[str, Any],
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    repository: str | os.PathLike[str] | None = None,
    require_clean_source: bool = False,
) -> None:
    """Atomically save a Markdown report whose digest points at the validated JSON."""
    validate_version_comparison_artifact(
        document,
        repository=repository,
        require_current_source=True,
        require_clean_source=require_clean_source,
    )
    _atomic_write(
        Path(path), render_version_comparison_report(document).encode("utf-8"), overwrite=overwrite
    )


__all__ = [
    "CASE_GENERATOR_VERSION",
    "CORE_CAMPAIGN_SUBSTEPS",
    "FIXED_CASE_COUNT",
    "INTENDED_CERTIFICATE_HORIZON",
    "INTENDED_POLICY_COUNT",
    "MINIMUM_FINAL_RANDOMIZED_CASES",
    "SHORT_INTERVAL_SUBSTEPS",
    "MatchedVersionCase",
    "VERSION_COMPARISON_KIND",
    "VERSION_COMPARISON_SCHEMA_VERSION",
    "VersionComparisonProfile",
    "comparison_profile",
    "generate_matched_version_cases",
    "load_version_comparison_artifact",
    "matched_case_set_sha256",
    "render_version_comparison_report",
    "run_matched_version_comparison",
    "save_version_comparison_artifact",
    "save_version_comparison_report",
    "validate_version_comparison_artifact",
]
