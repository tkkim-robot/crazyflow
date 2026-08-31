"""Artifact-backed empirical falsification of the real DA-PLCBF trial evaluator.

The generic fixed-budget search in :mod:`crazyflow.safety.da_plcbf.falsification` deliberately
knows nothing about Crazyflow.  This module supplies the missing experiment contract: a declared
14-dimensional intervention, immutable scenario tapes, the real ``da_plcbf_full``
:func:`~crazyflow.safety.da_plcbf.experiments.run_trial` dispatch, validated trace/event/metric
caching, and counterexample ranking.

This remains a finite empirical search.  It is useful for finding and replaying implementation
counterexamples, but it is neither a safety proof nor a paired superiority test.  Every candidate
is evaluated once in the dedicated combined falsification condition.  Its actual initial state is
perturbed while its declared constant-velocity reference remains fixed, and the evaluator returns
that trial's exact ``ScientificTrialMetrics.minimum_hard_margin`` value.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from crazyflow.safety.da_plcbf.artifacts import (
    load_events,
    load_metrics,
    load_trace,
    save_trace,
    write_events,
    write_metrics,
)
from crazyflow.safety.da_plcbf.baselines import MethodID, method_spec
from crazyflow.safety.da_plcbf.experiments import (
    ConditionID,
    ExperimentConfig,
    _CampaignExecutableCache,
    build_experiment_resources,
    run_trial,
    scenario_config_for_condition,
)
from crazyflow.safety.da_plcbf.falsification import (
    FALSIFICATION_CLAIM_BOUNDARY,
    FalsificationConfig,
    FalsificationResult,
    FalsificationStatus,
    boundary_candidate_set_sha256,
    load_falsification_result,
    run_fixed_budget_falsification,
    save_falsification_result,
)
from crazyflow.safety.da_plcbf.scenarios import (
    ScenarioTape,
    ScenarioTapeConfig,
    generate_scenario_tape,
    load_scenario_tape,
    save_scenario_tape,
)
from crazyflow.safety.da_plcbf.scientific_evaluation import (
    BoundaryCandidateSet,
    BoundaryVariable,
    FalsificationAxis,
    TrialAssignment,
    generate_boundary_candidates,
    make_paired_trial_schedule,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from crazyflow.safety.da_plcbf.experiments import ExperimentResources, TrialRun


FALSIFICATION_EXPERIMENT_SCHEMA_VERSION = 2
FALSIFICATION_EVALUATOR_PROTOCOL = "da-plcbf-combined-run-trial-minimum-hard-margin-v4"
FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY = (
    "This bounded search reports empirical finite-horizon outcomes of the identified full-method "
    "evaluator. A negative retained margin is a counterexample for that tested configuration. "
    "No observed sample, ranking, or seven-method replay establishes universal safety or method "
    "superiority; superiority requires a separately predeclared paired statistical comparison."
)
_CACHE_SCHEMA_VERSION = 2
_FAILURE_CACHE_SCHEMA_VERSION = 1
_REPLAY_SCHEMA_VERSION = 2
_MANIFEST_SCHEMA_VERSION = 1
_COMPLETE_MARKER_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_METHODS = tuple(item.value for item in MethodID)
_FALSIFICATION_CONDITIONS = (ConditionID.FALSIFICATION_COMBINED.value,)


class RetainedOperationalFailure(RuntimeError):
    """Stable exception used to preserve an evaluator failure across crash/resume."""


@dataclass(frozen=True, slots=True)
class DecodedIntervention:
    """The exact 14-D physical/scenario intervention consumed by the evaluator."""

    initial_x_offset_m: float
    initial_vx_offset_mps: float
    event_time_scale: float
    event_time_offset_fraction: float
    wind_speed_limit_mps: float
    wind_gust_amplitude_mps: float
    mass_scale_upper: float
    drag_scale_upper: float
    rotor_symmetric_efficiency_lower: float
    rotor_single_efficiency_lower: float
    acceleration_noise_std: float
    motor_force_noise_std: float
    collective_thrust_authority: float
    weakest_rotor_authority: float

    def values(self) -> tuple[float, ...]:
        """Return fields in the canonical flattened-variable order."""
        return tuple(float(getattr(self, field.name)) for field in fields(self))


def falsification_boundary_variables() -> tuple[BoundaryVariable, ...]:
    """Return the fixed seven-axis, fourteen-scalar intervention declaration."""
    return (
        BoundaryVariable(
            FalsificationAxis.INITIAL_STATE,
            "initial_state",
            ("x_offset_m", "vx_offset_mps"),
            (-0.50, -0.30),
            (0.50, 0.30),
            (0.0, 0.0),
        ),
        BoundaryVariable(
            FalsificationAxis.OBSTACLE_TIMING,
            "event_timing",
            ("scale", "offset_fraction"),
            (0.80, -0.05),
            (1.20, 0.05),
            (1.0, 0.0),
        ),
        BoundaryVariable(
            FalsificationAxis.WIND,
            "wind",
            ("speed_limit_mps", "gust_amplitude_mps"),
            (2.0, 0.20),
            (3.5, 0.80),
            (3.0, 0.75),
        ),
        BoundaryVariable(
            FalsificationAxis.MASS,
            "payload_drag",
            ("mass_scale_upper", "drag_scale_upper"),
            (1.15, 1.10),
            (1.65, 1.65),
            (1.45, 1.45),
        ),
        BoundaryVariable(
            FalsificationAxis.ROTOR_EFFICIENCY,
            "rotor_efficiency",
            ("symmetric_event_lower", "single_rotor_event_lower"),
            (0.60, 0.35),
            (0.90, 0.60),
            (0.75, 0.50),
        ),
        BoundaryVariable(
            FalsificationAxis.ESTIMATOR_ERROR,
            "estimator_noise",
            ("acceleration_std", "motor_force_std"),
            (0.0, 0.0),
            (0.08, 0.0015),
            (0.03, 0.0005),
        ),
        BoundaryVariable(
            FalsificationAxis.ACTUATOR_SATURATION,
            "actuator_authority",
            ("collective_fraction", "weakest_rotor_fraction"),
            (0.55, 0.65),
            (1.00, 1.00),
            (1.00, 1.00),
        ),
    )


def decode_falsification_intervention(values: np.ndarray | Sequence[float]) -> DecodedIntervention:
    """Decode and bound-check one canonical float64[14] intervention vector."""
    array = np.asarray(values, dtype=np.float64)
    variables = falsification_boundary_variables()
    lower = np.concatenate([np.asarray(item.lower) for item in variables])
    upper = np.concatenate([np.asarray(item.upper) for item in variables])
    if array.shape != (14,) or not np.all(np.isfinite(array)):
        raise ValueError("falsification intervention must be one finite float64[14] vector")
    if np.any(array < lower) or np.any(array > upper):
        raise ValueError("falsification intervention exceeds its declared bounds")
    decoded = DecodedIntervention(*map(float, array))
    if decoded.rotor_single_efficiency_lower > decoded.rotor_symmetric_efficiency_lower:
        raise ValueError("single-rotor efficiency lower bound cannot exceed the symmetric lower")
    if decoded.wind_gust_amplitude_mps > 0.4 * decoded.wind_speed_limit_mps:
        raise ValueError("wind gust must not exceed 0.4 times the wind speed limit")
    return decoded


@dataclass(frozen=True, slots=True)
class FalsificationProfile:
    """One immutable search/evaluator budget; CLI profiles cannot override these fields."""

    name: str
    experiment: ExperimentConfig
    conditions: tuple[str, ...]
    randomized_candidate_count: int
    search: FalsificationConfig
    maximum_ranked_counterexamples: int
    maximum_replayed_counterexamples: int

    def validate(self) -> None:
        """Validate the fixed profile and its full Phase-7 intervention coverage."""
        if self.name not in {"smoke", "development", "final"}:
            raise ValueError("unsupported falsification profile")
        self.experiment.validate()
        if self.conditions != _FALSIFICATION_CONDITIONS:
            raise ValueError("profiles must use the dedicated combined falsification condition")
        dimension = sum(len(item.component_names) for item in falsification_boundary_variables())
        if self.randomized_candidate_count < 2 * dimension:
            raise ValueError("profile candidate count cannot cover both sides of every bound")
        self.search.validate(candidate_count=self.randomized_candidate_count)
        if self.maximum_ranked_counterexamples <= 0 or self.maximum_replayed_counterexamples <= 0:
            raise ValueError("counterexample ranking and replay limits must be positive")
        if self.maximum_replayed_counterexamples > self.maximum_ranked_counterexamples:
            raise ValueError("replay limit cannot exceed the ranking limit")

    def metadata(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible profile provenance."""
        return {
            "name": self.name,
            "experiment": asdict(self.experiment),
            "conditions": list(self.conditions),
            "randomized_candidate_count": self.randomized_candidate_count,
            "search": asdict(self.search),
            "maximum_ranked_counterexamples": self.maximum_ranked_counterexamples,
            "maximum_replayed_counterexamples": self.maximum_replayed_counterexamples,
        }


def falsification_profile(name: str, *, root_seed: int = 20260831) -> FalsificationProfile:
    """Return one of the three fixed smoke/development/final profiles."""
    if name == "smoke":
        experiment = ExperimentConfig(
            # Three states leave one nonterminal publication boundary, so even the one-candidate
            # GPU probe genuinely exercises the full method's online BPTT/validation path.
            control_steps=3,
            certificate_horizon=1,
            policy_count=16,
            prediction_samples=4,
            training_scenario_count=2,
            validation_scenarios_per_fold=1,
            bptt_burst_steps=1,
            adaptation_interval_steps=2,
            estimator_interval_steps=1,
            estimator_window_steps=3,
            random_seed=root_seed,
            validation_runtime_budget_seconds=120.0,
            validation_minimum_diversity=1e-8,
            realtime_pacing=False,
        )
        profile = FalsificationProfile(
            name,
            experiment,
            _FALSIFICATION_CONDITIONS,
            28,
            FalsificationConfig(refinement_seed_count=1, refinement_rounds=1),
            8,
            1,
        )
    elif name == "development":
        experiment = replace(
            ExperimentConfig.final_defaults(random_seed=root_seed),
            control_steps=31,
            certificate_horizon=10,
            policy_count=16,
            training_scenario_count=8,
            bptt_burst_steps=3,
            adaptation_interval_steps=5,
            estimator_interval_steps=6,
            estimator_window_steps=6,
            validation_minimum_diversity=1e-8,
            realtime_pacing=False,
        )
        profile = FalsificationProfile(
            name,
            experiment,
            _FALSIFICATION_CONDITIONS,
            56,
            FalsificationConfig(refinement_seed_count=2, refinement_rounds=2),
            24,
            3,
        )
    elif name == "final":
        profile = FalsificationProfile(
            name,
            replace(ExperimentConfig.final_defaults(random_seed=root_seed), realtime_pacing=False),
            _FALSIFICATION_CONDITIONS,
            280,
            FalsificationConfig(refinement_seed_count=4, refinement_rounds=4),
            100,
            10,
        )
    else:
        raise ValueError(f"unknown falsification profile {name!r}")
    profile.validate()
    return profile


def generate_falsification_candidates(
    profile: FalsificationProfile, *, root_seed: int
) -> BoundaryCandidateSet:
    """Generate the complete deterministic candidate set for one fixed profile."""
    profile.validate()
    if root_seed != profile.experiment.random_seed:
        raise ValueError("candidate root_seed must match the profile experiment random_seed")
    return generate_boundary_candidates(
        falsification_boundary_variables(),
        count=profile.randomized_candidate_count,
        root_seed=root_seed,
        search_name=f"da-plcbf-{profile.name}",
        boundary_band_fraction=0.05,
        background_nominal_fraction=0.25,
    )


def _warped_fraction(value: float, intervention: DecodedIntervention) -> float:
    return float(
        np.clip(
            intervention.event_time_scale * value + intervention.event_time_offset_fraction,
            0.0,
            1.0,
        )
    )


def intervened_scenario_config(
    condition: str, experiment: ExperimentConfig, intervention: DecodedIntervention
) -> ScenarioTapeConfig:
    """Apply the exact declared intervention to one condition's generator configuration."""
    if condition != ConditionID.FALSIFICATION_COMBINED.value:
        raise ValueError("falsification intervention requires the combined condition")
    base = scenario_config_for_condition(condition, experiment)
    reference_position = (
        base.vehicle_initial_position
        if base.reference_initial_position is None
        else base.reference_initial_position
    )
    reference_velocity = (
        base.vehicle_initial_velocity
        if base.reference_initial_velocity is None
        else base.reference_initial_velocity
    )
    initial_position = list(reference_position)
    initial_velocity = list(reference_velocity)
    initial_position[0] += intervention.initial_x_offset_m
    initial_velocity[0] += intervention.initial_vx_offset_mps
    release = tuple(
        _warped_fraction(value, intervention) for value in base.ball_release_fraction_range
    )
    if release[1] >= 1.0:
        release = (min(release[0], 1.0 - 2e-6), 1.0 - 1e-6)
    if release[1] <= release[0]:
        release = (max(0.0, release[0] - 1e-6), release[0] + 1e-6)
    crossing_fraction = _warped_fraction(0.5, intervention)
    speed_scale = 1.0 / intervention.event_time_scale
    config = replace(
        base,
        vehicle_initial_position=tuple(initial_position),
        vehicle_initial_velocity=tuple(initial_velocity),
        reference_initial_position=reference_position,
        reference_initial_velocity=reference_velocity,
        ball_release_fraction_range=release,
        crossing_fraction_range=(crossing_fraction, crossing_fraction),
        attacker_speed_range=tuple(value * speed_scale for value in base.attacker_speed_range),
        attacker_initial_speed_fraction=float(
            np.clip(
                base.attacker_initial_speed_fraction - intervention.event_time_offset_fraction,
                0.0,
                1.0,
            )
        ),
        interceptor_prediction_horizon=(
            base.interceptor_prediction_horizon * intervention.event_time_scale
        ),
        wind_speed_limit=intervention.wind_speed_limit_mps,
        wind_gust_amplitude=intervention.wind_gust_amplitude_mps,
        mass_scale_bounds=(base.mass_scale_bounds[0], intervention.mass_scale_upper),
        drag_scale_bounds=(base.drag_scale_bounds[0], intervention.drag_scale_upper),
        rotor_efficiency_bounds=(intervention.rotor_symmetric_efficiency_lower, 1.0),
        rotor_single_efficiency_lower=intervention.rotor_single_efficiency_lower,
        wind_change_fraction=_warped_fraction(base.wind_change_fraction, intervention),
        mass_change_fraction=_warped_fraction(base.mass_change_fraction, intervention),
        drag_change_fraction=_warped_fraction(base.drag_change_fraction, intervention),
        rotor_symmetric_change_fraction=_warped_fraction(
            base.rotor_symmetric_change_fraction, intervention
        ),
        rotor_single_change_fraction=_warped_fraction(
            base.rotor_single_change_fraction, intervention
        ),
        estimator_acceleration_noise_std=intervention.acceleration_noise_std,
        estimator_motor_force_noise_std=intervention.motor_force_noise_std,
    )
    config.validate()
    return config


def _intervened_resources(
    resources: ExperimentResources, intervention: DecodedIntervention
) -> ExperimentResources:
    maximum = np.broadcast_to(np.asarray(resources.actuator.thrust_max), (4,)).astype(
        np.float32, copy=True
    )
    maximum *= intervention.collective_thrust_authority
    maximum[0] *= intervention.weakest_rotor_authority
    minimum = np.broadcast_to(np.asarray(resources.actuator.thrust_min), (4,))
    if np.any(maximum <= minimum):
        raise ValueError("actuator intervention leaves no valid per-rotor force interval")
    actuator = resources.actuator._replace(thrust_max=jnp.asarray(maximum))
    return replace(resources, actuator=actuator)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256_document(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_bytes(value: str, name: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{name} must be one lowercase SHA-256 hex digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be one lowercase SHA-256 hex digest") from error
    if len(decoded) != 32:
        raise ValueError(f"{name} must be one lowercase SHA-256 hex digest")
    return decoded


def _repository_root(value: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a repository containing the dependency declaration."""
    if value is not None:
        root = Path(value).resolve()
    else:
        root = Path(__file__).resolve()
        while root.parent != root and not (root / "pyproject.toml").is_file():
            root = root.parent
    if not (root / "pyproject.toml").is_file():
        raise ValueError("repository root must contain pyproject.toml")
    return root


def falsification_source_tree_sha256(repository: str | os.PathLike[str] | None = None) -> str:
    """Hash executable sources, runtime assets, and root dependency locks.

    Tests, documentation, and generated artifacts are deliberately excluded.  The digest is part
    of every evaluator/cache/replay identity, so a result cannot be silently promoted after a
    numerical implementation or packaged model asset changes.
    """
    root = _repository_root(repository)
    roots_and_suffixes = (
        (root / "crazyflow", frozenset({".py", ".toml", ".xml", ".stl"})),
        (root / "examples" / "da_plcbf", frozenset({".py"})),
        (root / "benchmark", frozenset({".py"})),
    )
    paths = {
        path
        for source_root, suffixes in roots_and_suffixes
        if source_root.is_dir()
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }
    for name in ("pyproject.toml", "pixi.lock", "uv.lock"):
        path = root / name
        if path.is_file():
            paths.add(path)
    digest = hashlib.sha256(b"crazyflow.da_plcbf.falsification-source-tree.v1\0")
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    """Load one JSON object while rejecting duplicate keys and non-finite constants."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate JSON key in {path}")
        return dict(items)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def falsification_evaluator_sha256(
    profile: FalsificationProfile,
    *,
    candidate_set_sha256: str,
    source_tree_sha256: str | None = None,
    repository: str | os.PathLike[str] | None = None,
    strict_physical_validation: bool = True,
) -> str:
    """Digest the complete semantic evaluator/profile/intervention protocol."""
    profile.validate()
    _sha256_bytes(candidate_set_sha256, "candidate_set_sha256")
    source_digest = (
        falsification_source_tree_sha256(repository)
        if source_tree_sha256 is None
        else source_tree_sha256
    )
    _sha256_bytes(source_digest, "source_tree_sha256")
    if not isinstance(strict_physical_validation, bool):
        raise TypeError("strict_physical_validation must be boolean")
    document = {
        "protocol": FALSIFICATION_EVALUATOR_PROTOCOL,
        "profile": profile.metadata(),
        "candidate_set_sha256": candidate_set_sha256,
        "source_tree_sha256": source_digest,
        "strict_physical_validation": strict_physical_validation,
        "method": method_spec(MethodID.DA_PLCBF_FULL).metadata(),
        "variables": [
            {
                "axis": item.axis.value,
                "name": item.name,
                "components": list(item.component_names),
                "lower": list(item.lower),
                "upper": list(item.upper),
                "nominal": list(item.nominal),
            }
            for item in falsification_boundary_variables()
        ],
        "objective": "exact-combined-trial-scientific-minimum-hard-margin",
        "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
    }
    return _sha256_document(document)


@dataclass(frozen=True, slots=True)
class ConditionTrialEvidence:
    """Validated immutable artifacts for one condition in an evaluator call."""

    condition: str
    tape_sha256: str
    trace_sha256: str
    minimum_hard_margin: float
    tape_file_sha256: str
    trace_file_sha256: str
    events_file_sha256: str
    metrics_file_sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationCallEvidence:
    """Side-channel provenance for one call retained by the generic search executor."""

    candidate_sha256: str
    values: tuple[float, ...]
    status: str
    cache_hit: bool
    margin: float | None
    trials: tuple[ConditionTrialEvidence, ...]
    error_type: str | None = None
    error_message: str | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assignment_map(profile: FalsificationProfile, root_seed: int) -> dict[str, TrialAssignment]:
    schedule = make_paired_trial_schedule(
        root_seed=root_seed, methods=_METHODS, conditions=profile.conditions, trials_per_condition=1
    )
    return {
        assignment.condition: assignment
        for assignment in schedule.assignments
        if assignment.method == MethodID.DA_PLCBF_FULL.value
    }


def _replay_assignment_map(
    profile: FalsificationProfile, root_seed: int
) -> dict[MethodID, TrialAssignment]:
    schedule = make_paired_trial_schedule(
        root_seed=root_seed, methods=_METHODS, conditions=profile.conditions, trials_per_condition=1
    )
    assignments = {MethodID(item.method): item for item in schedule.assignments}
    if tuple(assignments) != tuple(MethodID):
        raise RuntimeError("seven-method replay schedule is incomplete or out of order")
    return assignments


class DAFalsificationEvaluator:
    """Real-trial evaluator with replay-validated, write-once success and failure caches."""

    def __init__(
        self,
        profile: FalsificationProfile,
        *,
        root_seed: int,
        candidate_set_sha256: str,
        cache_directory: str | os.PathLike[str],
        trial_runner: Callable[..., TrialRun] = run_trial,
        resource_builder: Callable[..., ExperimentResources] = build_experiment_resources,
        repository: str | os.PathLike[str] | None = None,
        source_tree_sha256: str | None = None,
        strict_physical_validation: bool = True,
        validate_only: bool = False,
    ) -> None:
        profile.validate()
        if root_seed != profile.experiment.random_seed:
            raise ValueError("evaluator root_seed must match the profile experiment random_seed")
        if not isinstance(strict_physical_validation, bool):
            raise TypeError("strict_physical_validation must be boolean")
        if not isinstance(validate_only, bool):
            raise TypeError("validate_only must be boolean")
        self.profile = profile
        self.root_seed = int(root_seed)
        self.candidate_set_sha256 = candidate_set_sha256
        _sha256_bytes(candidate_set_sha256, "candidate_set_sha256")
        self.source_tree_sha256 = (
            falsification_source_tree_sha256(repository)
            if source_tree_sha256 is None
            else source_tree_sha256
        )
        _sha256_bytes(self.source_tree_sha256, "source_tree_sha256")
        self.strict_physical_validation = strict_physical_validation
        self.evaluator_sha256 = falsification_evaluator_sha256(
            profile,
            candidate_set_sha256=candidate_set_sha256,
            source_tree_sha256=self.source_tree_sha256,
            strict_physical_validation=strict_physical_validation,
        )
        self.cache_directory = Path(cache_directory)
        self._failure_directory = self.cache_directory / "operational_failures"
        self._staging_directory = self.cache_directory / ".staging"
        if validate_only:
            if not self.cache_directory.is_dir():
                raise FileNotFoundError("falsification cache directory is missing")
        else:
            self.cache_directory.mkdir(parents=True, exist_ok=True)
            self._failure_directory.mkdir(exist_ok=True)
            for orphan in tuple(self._failure_directory.iterdir()):
                if re.fullmatch(r"\.[0-9a-f]{64}\.json\..+", orphan.name) and orphan.is_file():
                    orphan.unlink()
            self._staging_directory.mkdir(exist_ok=True)
            for orphan in tuple(self._staging_directory.iterdir()):
                if orphan.is_dir() and not orphan.is_symlink():
                    shutil.rmtree(orphan)
                else:
                    orphan.unlink()
        self._trial_runner = trial_runner
        self._resource_builder = resource_builder
        self._assignments = _assignment_map(profile, root_seed)
        self._executable_cache = _CampaignExecutableCache()
        self.calls: list[EvaluationCallEvidence] = []

    def candidate_sha256(self, values: np.ndarray | Sequence[float]) -> str:
        """Return the value-specific digest bound to this exact evaluator and candidate set."""
        decoded = decode_falsification_intervention(values)
        array = np.asarray(decoded.values(), dtype="<f8")
        digest = hashlib.sha256(b"crazyflow.da_plcbf.falsification-candidate.v2\0")
        digest.update(_sha256_bytes(self.candidate_set_sha256, "candidate_set_sha256"))
        digest.update(_sha256_bytes(self.evaluator_sha256, "evaluator_sha256"))
        digest.update(array.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _error_evidence(
        candidate_sha256: str, values: tuple[float, ...], error: Exception, *, cache_hit: bool
    ) -> EvaluationCallEvidence:
        error_type = type(error).__qualname__
        message = " ".join(str(error).split())[:512] or error_type
        return EvaluationCallEvidence(
            candidate_sha256,
            values,
            "operational_failure",
            cache_hit,
            None,
            (),
            error_type,
            message,
        )

    def _retained_exception_message(self, evidence: EvaluationCallEvidence) -> str:
        assert evidence.error_type is not None and evidence.error_message is not None
        return f"{evidence.error_type}: {evidence.error_message}"[
            : self.profile.search.maximum_failure_message_characters
        ]

    def __call__(self, values: np.ndarray) -> float:
        """Return the exact combined-condition margin or a crash-stable retained failure."""
        decoded = decode_falsification_intervention(values)
        candidate_sha256 = self.candidate_sha256(values)
        try:
            cached = self._load_cache(candidate_sha256, decoded.values())
            retained_failure = self._load_failure(candidate_sha256, decoded.values())
        except Exception as error:
            evidence = self._error_evidence(
                candidate_sha256, decoded.values(), error, cache_hit=False
            )
            self.calls.append(evidence)
            raise
        if cached is not None and retained_failure is not None:
            raise ValueError("candidate has both success and operational-failure cache records")
        if cached is not None:
            evidence = replace(cached, cache_hit=True)
            self.calls.append(evidence)
            assert evidence.margin is not None
            return evidence.margin
        if retained_failure is not None:
            evidence = replace(retained_failure, cache_hit=True)
            self.calls.append(evidence)
            raise RetainedOperationalFailure(self._retained_exception_message(evidence))
        try:
            evidence = self._evaluate_and_commit(candidate_sha256, decoded)
        except Exception as error:
            evidence = self._error_evidence(
                candidate_sha256, decoded.values(), error, cache_hit=False
            )
            self._save_failure(evidence)
            self.calls.append(evidence)
            raise RetainedOperationalFailure(self._retained_exception_message(evidence)) from error
        self.calls.append(evidence)
        assert evidence.margin is not None
        return evidence.margin

    def _evaluate_and_commit(
        self, candidate_sha256: str, intervention: DecodedIntervention
    ) -> EvaluationCallEvidence:
        trials: list[tuple[TrialRun, ScenarioTape]] = []
        for condition in self.profile.conditions:
            assignment = self._assignments[condition]
            tape_config = intervened_scenario_config(
                condition, self.profile.experiment, intervention
            )
            tape = generate_scenario_tape(
                assignment.scenario_root_seed, tape_config, fold=assignment.scenario_fold
            )
            obstacle_count = tape.static_positions.shape[0] + tape.dynamic_positions.shape[1]
            resources = self._resource_builder(
                self.profile.experiment,
                obstacle_count=obstacle_count,
                initialization_seed=int(assignment.shared_stochastic_seed & 0xFFFFFFFF),
            )
            resources = _intervened_resources(resources, intervention)
            run = self._trial_runner(
                assignment,
                tape,
                self.profile.experiment,
                resources=resources,
                executable_cache=self._executable_cache,
            )
            if run.assignment.key != assignment.key:
                raise RuntimeError("falsification runner returned a mismatched trial assignment")
            run.trace.validate()
            run.scientific_metrics.validate()
            exact_margin = float(run.scientific_metrics.minimum_hard_margin)
            if exact_margin != float(np.min(run.trace.hard_barriers)):
                raise RuntimeError("scientific minimum margin does not match immutable trace")
            trials.append((run, tape))

        target = self.cache_directory / candidate_sha256
        if target.exists():
            loaded = self._load_cache(candidate_sha256, intervention.values())
            if loaded is None:
                raise RuntimeError("candidate cache disappeared during validation")
            return loaded
        with tempfile.TemporaryDirectory(
            dir=self._staging_directory, prefix=f"{candidate_sha256}."
        ) as temporary_name:
            temporary = Path(temporary_name)
            self._write_trial_artifacts(temporary, trials)
            evidence = self._validate_cache_directory(
                temporary, candidate_sha256, intervention.values(), expected_manifest=False
            )
            (temporary / "manifest.json").write_bytes(
                _canonical_json(self._cache_manifest(evidence)) + b"\n"
            )
            evidence = self._validate_cache_directory(
                temporary, candidate_sha256, intervention.values(), expected_manifest=True
            )
            os.replace(temporary, target)
        loaded = self._load_cache(candidate_sha256, intervention.values())
        if loaded is None or loaded != evidence:
            raise RuntimeError("committed candidate cache changed during atomic publication")
        return loaded

    def _write_trial_artifacts(
        self, root: Path, trials: Sequence[tuple[TrialRun, ScenarioTape]]
    ) -> tuple[ConditionTrialEvidence, ...]:
        evidence: list[ConditionTrialEvidence] = []
        for run, tape in trials:
            condition = run.assignment.condition
            directory = root / "conditions" / condition
            directory.mkdir(parents=True)
            tape_path = directory / "tape.npz"
            trace_path = directory / "trace.npz"
            events_path = directory / "events.jsonl"
            metrics_path = directory / "metrics.json"
            save_scenario_tape(tape, tape_path)
            save_trace(run.trace, trace_path)
            write_events(run.events, events_path, trace=run.trace)
            write_metrics(run.trace, metrics_path)
            evidence.append(
                ConditionTrialEvidence(
                    condition,
                    tape.sha256,
                    run.trace.content_sha256,
                    float(run.scientific_metrics.minimum_hard_margin),
                    _file_sha256(tape_path),
                    _file_sha256(trace_path),
                    _file_sha256(events_path),
                    _file_sha256(metrics_path),
                )
            )
        return tuple(evidence)

    def _cache_manifest(self, evidence: EvaluationCallEvidence) -> dict[str, Any]:
        return {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "artifact_type": "crazyflow-da-plcbf-falsification-evaluation-cache",
            "candidate_sha256": evidence.candidate_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "strict_physical_validation": self.strict_physical_validation,
            "method": MethodID.DA_PLCBF_FULL.value,
            "conditions": list(self.profile.conditions),
            "values": list(evidence.values),
            "minimum_hard_margin": evidence.margin,
            "trials": [asdict(item) for item in evidence.trials],
            "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
        }

    def _failure_path(self, candidate_sha256: str) -> Path:
        return self._failure_directory / f"{candidate_sha256}.json"

    def _save_failure(self, evidence: EvaluationCallEvidence) -> None:
        if evidence.status != "operational_failure" or evidence.margin is not None:
            raise ValueError("only operational failures may enter the failure cache")
        document = {
            "schema_version": _FAILURE_CACHE_SCHEMA_VERSION,
            "artifact_type": "crazyflow-da-plcbf-falsification-operational-failure",
            "candidate_sha256": evidence.candidate_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "values": list(evidence.values),
            "status": evidence.status,
            "error_type": evidence.error_type,
            "error_message": evidence.error_message,
            "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
        }
        self._failure_directory.mkdir(exist_ok=True)
        _write_new_or_identical(self._failure_path(evidence.candidate_sha256), document)

    def _load_failure(
        self, candidate_sha256: str, values: Sequence[float]
    ) -> EvaluationCallEvidence | None:
        path = self._failure_path(candidate_sha256)
        if not path.exists():
            return None
        document = _read_object(path)
        expected = {
            "schema_version",
            "artifact_type",
            "candidate_sha256",
            "candidate_set_sha256",
            "evaluator_sha256",
            "source_tree_sha256",
            "values",
            "status",
            "error_type",
            "error_message",
            "claim_boundary",
        }
        if set(document) != expected or (
            document["schema_version"] != _FAILURE_CACHE_SCHEMA_VERSION
            or document["artifact_type"] != "crazyflow-da-plcbf-falsification-operational-failure"
            or document["candidate_sha256"] != candidate_sha256
            or document["candidate_set_sha256"] != self.candidate_set_sha256
            or document["evaluator_sha256"] != self.evaluator_sha256
            or document["source_tree_sha256"] != self.source_tree_sha256
            or tuple(document["values"]) != tuple(values)
            or document["status"] != "operational_failure"
            or document["claim_boundary"] != FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY
        ):
            raise ValueError("operational-failure cache does not bind to this evaluator")
        if not isinstance(document["error_type"], str) or not isinstance(
            document["error_message"], str
        ):
            raise ValueError("operational-failure cache error fields are invalid")
        return EvaluationCallEvidence(
            candidate_sha256,
            tuple(map(float, values)),
            "operational_failure",
            False,
            None,
            (),
            document["error_type"],
            document["error_message"],
        )

    def _load_cache(
        self, candidate_sha256: str, values: Sequence[float]
    ) -> EvaluationCallEvidence | None:
        directory = self.cache_directory / candidate_sha256
        if not directory.exists():
            return None
        return self._validate_cache_directory(
            directory, candidate_sha256, values, expected_manifest=True
        )

    def _validate_cache_directory(
        self,
        directory: Path,
        candidate_sha256: str,
        values: Sequence[float],
        *,
        expected_manifest: bool,
    ) -> EvaluationCallEvidence:
        expected_root_entries = (
            {"conditions", "manifest.json"} if expected_manifest else {"conditions"}
        )
        if (
            not directory.is_dir()
            or {item.name for item in directory.iterdir()} != expected_root_entries
        ):
            raise ValueError("candidate cache root entries are invalid")
        conditions_root = directory / "conditions"
        if not conditions_root.is_dir() or {item.name for item in conditions_root.iterdir()} != set(
            self.profile.conditions
        ):
            raise ValueError("candidate cache condition entries are invalid")
        if expected_manifest:
            manifest_path = directory / "manifest.json"
            manifest = _read_object(manifest_path)
            expected_keys = {
                "schema_version",
                "artifact_type",
                "candidate_sha256",
                "candidate_set_sha256",
                "evaluator_sha256",
                "source_tree_sha256",
                "strict_physical_validation",
                "method",
                "conditions",
                "values",
                "minimum_hard_margin",
                "trials",
                "claim_boundary",
            }
            if not isinstance(manifest, dict) or set(manifest) != expected_keys:
                raise ValueError("candidate cache manifest schema is invalid")
            if (
                manifest["schema_version"] != _CACHE_SCHEMA_VERSION
                or manifest["artifact_type"] != "crazyflow-da-plcbf-falsification-evaluation-cache"
                or manifest["candidate_sha256"] != candidate_sha256
                or manifest["candidate_set_sha256"] != self.candidate_set_sha256
                or manifest["evaluator_sha256"] != self.evaluator_sha256
                or manifest["source_tree_sha256"] != self.source_tree_sha256
                or manifest["strict_physical_validation"] is not self.strict_physical_validation
                or manifest["method"] != MethodID.DA_PLCBF_FULL.value
                or tuple(manifest["conditions"]) != self.profile.conditions
                or tuple(manifest["values"]) != tuple(values)
                or manifest["claim_boundary"] != FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY
            ):
                raise ValueError("candidate cache manifest does not bind to this evaluator")
            serialized_trials = manifest["trials"]
            serialized_margin = float(manifest["minimum_hard_margin"])
        else:
            serialized_trials = None
            serialized_margin = math.inf

        trials: list[ConditionTrialEvidence] = []
        for condition in self.profile.conditions:
            condition_directory = directory / "conditions" / condition
            if not condition_directory.is_dir() or {
                item.name for item in condition_directory.iterdir()
            } != {"tape.npz", "trace.npz", "events.jsonl", "metrics.json"}:
                raise ValueError("candidate cache trial entries are invalid")
            tape_path = condition_directory / "tape.npz"
            trace_path = condition_directory / "trace.npz"
            events_path = condition_directory / "events.jsonl"
            metrics_path = condition_directory / "metrics.json"
            tape = load_scenario_tape(tape_path)
            assignment = self._assignments[condition]
            expected_tape = generate_scenario_tape(
                assignment.scenario_root_seed,
                intervened_scenario_config(
                    condition, self.profile.experiment, decode_falsification_intervention(values)
                ),
                fold=assignment.scenario_fold,
            )
            if (
                int(tape.root_seed) != assignment.scenario_root_seed
                or int(tape.generation_fold) != assignment.scenario_fold
                or tape.sha256 != expected_tape.sha256
            ):
                raise ValueError("cached tape does not regenerate from assignment/config/values")
            trace = load_trace(trace_path)
            events = load_events(events_path, trace=trace)
            metrics = load_metrics(metrics_path, trace=trace)
            if str(trace.scenario_tape_sha256) != tape.sha256:
                raise ValueError("cached trace is not bound to its immutable scenario tape")
            margin = float(metrics["minimum_hard_margin"])
            if margin != float(np.min(trace.hard_barriers)):
                raise ValueError("cached metric margin does not match cached trace")
            started = [event for event in events if event.name == "trial_started"]
            if len(started) != 1 or started[0].details != {
                "method": assignment.method,
                "condition": assignment.condition,
            }:
                raise ValueError("cached events do not bind the exact scheduled assignment")
            if self.strict_physical_validation:
                from crazyflow.safety.da_plcbf.campaign_artifacts import (
                    _validate_trace_physical_evidence,
                )

                _validate_trace_physical_evidence(
                    trace, tape, self.profile.experiment, condition=condition
                )
            trials.append(
                ConditionTrialEvidence(
                    condition,
                    tape.sha256,
                    trace.content_sha256,
                    margin,
                    _file_sha256(tape_path),
                    _file_sha256(trace_path),
                    _file_sha256(events_path),
                    _file_sha256(metrics_path),
                )
            )
        if len(trials) != 1:
            raise ValueError("combined-condition cache must contain exactly one trial")
        aggregate = trials[0].minimum_hard_margin
        if expected_manifest:
            if aggregate != serialized_margin or serialized_trials != [
                asdict(item) for item in trials
            ]:
                raise ValueError("candidate cache manifest disagrees with validated artifacts")
        return EvaluationCallEvidence(
            candidate_sha256, tuple(map(float, values)), "success", False, aggregate, tuple(trials)
        )


def _validate_search_calls(
    result: FalsificationResult, calls: Sequence[EvaluationCallEvidence]
) -> None:
    if len(calls) != len(result.evaluations):
        raise ValueError("evaluator call evidence is incomplete")
    for evaluation, call in zip(result.evaluations, calls, strict=True):
        if evaluation.values != call.values:
            raise ValueError("evaluator call values do not match falsification result")
        if evaluation.status is FalsificationStatus.SUCCESS:
            if call.status != "success" or call.margin != evaluation.margin or not call.trials:
                raise ValueError("successful evaluation lacks matching trial artifacts")
            if len(call.trials) != 1 or call.trials[0].minimum_hard_margin != call.margin:
                raise ValueError("successful evaluation margin disagrees with its exact trial")
        else:
            if (
                evaluation.status is not FalsificationStatus.EXCEPTION
                or evaluation.error_type != RetainedOperationalFailure.__qualname__
                or call.status != "operational_failure"
                or call.margin is not None
                or call.error_type is None
                or call.error_message is None
            ):
                raise ValueError("operational evaluator failure was not retained crash-stably")
            expected_message = f"{call.error_type}: {call.error_message}"[
                : result.config.maximum_failure_message_characters
            ]
            if evaluation.error_message != expected_message:
                raise ValueError("generic failure row disagrees with retained failure evidence")


def rank_unique_counterexamples(
    result: FalsificationResult, calls: Sequence[EvaluationCallEvidence], *, limit: int
) -> tuple[dict[str, Any], ...]:
    """Return unique worst successful counterexamples with replayable tape identities."""
    _validate_search_calls(result, calls)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("counterexample ranking limit must be positive")
    ranked: list[tuple[float, int, EvaluationCallEvidence]] = []
    seen: set[str] = set()
    for evaluation, call in zip(result.evaluations, calls, strict=True):
        if not evaluation.counterexample or call.candidate_sha256 in seen:
            continue
        seen.add(call.candidate_sha256)
        assert evaluation.margin is not None
        ranked.append((evaluation.margin, evaluation.evaluation_index, call))
    ranked.sort(key=lambda item: (item[0], item[1], item[2].candidate_sha256))
    return tuple(
        {
            "rank": rank,
            "evaluation_index": evaluation_index,
            "candidate_sha256": call.candidate_sha256,
            "minimum_hard_margin": margin,
            "values": list(call.values),
            "condition_trials": [asdict(item) for item in call.trials],
        }
        for rank, (margin, evaluation_index, call) in enumerate(ranked[:limit], start=1)
    )


def seven_method_replay_registry() -> tuple[dict[str, Any], ...]:
    """Describe the exact seven-method replay matrix without implying paired superiority."""
    entries = []
    for method in MethodID:
        spec = method_spec(method)
        entries.append(
            {
                "order": len(entries) + 1,
                "method": spec.metadata(),
                "discovery_evaluator": method is MethodID.DA_PLCBF_FULL,
                "replay_role": (
                    "falsification discovery method"
                    if method is MethodID.DA_PLCBF_FULL
                    else "descriptive matched-tape replay comparator"
                ),
                "claim_boundary": (
                    "A replay is descriptive. This registry contains no paired sample size, "
                    "confidence interval, or superiority decision."
                ),
            }
        )
    return tuple(entries)


def _successful_replay_record(
    directory: Path,
    tape: ScenarioTape,
    method: MethodID,
    profile: FalsificationProfile,
    assignment: TrialAssignment,
    *,
    order: int,
    execution_role: str,
    strict_physical_validation: bool,
) -> dict[str, Any]:
    expected_files = {"trace.npz", "events.jsonl", "metrics.json"}
    if not directory.is_dir() or {item.name for item in directory.iterdir()} != expected_files:
        raise ValueError("successful replay method directory is incomplete")
    trace_path = directory / "trace.npz"
    events_path = directory / "events.jsonl"
    metrics_path = directory / "metrics.json"
    trace = load_trace(trace_path)
    events = load_events(events_path, trace=trace)
    metrics = load_metrics(metrics_path, trace=trace)
    if str(trace.scenario_tape_sha256) != tape.sha256:
        raise ValueError("replay trace is not bound to the shared immutable tape")
    started = [event for event in events if (event.name == "trial_started")]
    if len(started) != 1 or started[0].details != {
        "method": assignment.method,
        "condition": assignment.condition,
    }:
        raise ValueError("replay events do not identify the scheduled method and condition")
    if assignment.method != method.value:
        raise ValueError("replay assignment does not identify the requested method")
    if strict_physical_validation:
        from crazyflow.safety.da_plcbf.campaign_artifacts import _validate_trace_physical_evidence

        _validate_trace_physical_evidence(
            trace, tape, profile.experiment, condition=assignment.condition
        )
    margin = float(metrics["minimum_hard_margin"])
    if margin != float(np.min(trace.hard_barriers)):
        raise ValueError("replay minimum margin disagrees with its immutable trace")
    return {
        "order": order,
        "method": method_spec(method).metadata(),
        "status": "success",
        "execution_role": execution_role,
        "minimum_hard_margin": margin,
        "trace_sha256": trace.content_sha256,
        "trace_file_sha256": _file_sha256(trace_path),
        "events_file_sha256": _file_sha256(events_path),
        "metrics_file_sha256": _file_sha256(metrics_path),
        "error_type": None,
        "error_message": None,
    }


def _failed_replay_record(
    method: MethodID, error: Exception, *, order: int, execution_role: str
) -> dict[str, Any]:
    return {
        "order": order,
        "method": method_spec(method).metadata(),
        "status": "operational_failure",
        "execution_role": execution_role,
        "minimum_hard_margin": None,
        "trace_sha256": None,
        "trace_file_sha256": None,
        "events_file_sha256": None,
        "metrics_file_sha256": None,
        "error_type": type(error).__qualname__,
        "error_message": " ".join(str(error).split())[:512] or type(error).__qualname__,
    }


def _validate_replay_directory(
    directory: Path,
    profile: FalsificationProfile,
    *,
    rank: int,
    candidate_sha256: str,
    values: Sequence[float],
    result_sha256: str,
    evaluator_sha256: str,
    source_tree_sha256: str,
    strict_physical_validation: bool,
) -> dict[str, Any]:
    if not directory.is_dir() or {item.name for item in directory.iterdir()} != {
        "tape.npz",
        "methods",
        "manifest.json",
    }:
        raise ValueError("counterexample replay directory is incomplete")
    manifest = _read_object(directory / "manifest.json")
    expected_manifest_keys = {
        "schema_version",
        "artifact_type",
        "profile",
        "rank",
        "candidate_sha256",
        "values",
        "condition",
        "falsification_result_sha256",
        "evaluator_sha256",
        "source_tree_sha256",
        "strict_physical_validation",
        "tape_sha256",
        "tape_file_sha256",
        "methods",
        "claim_boundary",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_keys:
        raise ValueError("counterexample replay manifest schema is invalid")
    if (
        manifest["schema_version"] != _REPLAY_SCHEMA_VERSION
        or manifest["artifact_type"] != "crazyflow-da-plcbf-seven-method-replay"
        or manifest["profile"] != profile.metadata()
        or manifest["rank"] != rank
        or manifest["candidate_sha256"] != candidate_sha256
        or tuple(manifest["values"]) != tuple(values)
        or manifest["condition"] != ConditionID.FALSIFICATION_COMBINED.value
        or manifest["falsification_result_sha256"] != result_sha256
        or manifest["evaluator_sha256"] != evaluator_sha256
        or manifest["source_tree_sha256"] != source_tree_sha256
        or manifest["strict_physical_validation"] is not strict_physical_validation
        or manifest["claim_boundary"] != FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY
    ):
        raise ValueError("counterexample replay manifest has mismatched provenance")
    tape_path = directory / "tape.npz"
    tape = load_scenario_tape(tape_path)
    if manifest["tape_sha256"] != tape.sha256 or manifest["tape_file_sha256"] != _file_sha256(
        tape_path
    ):
        raise ValueError("counterexample replay tape provenance is invalid")
    intervention = decode_falsification_intervention(values)
    assignments = _replay_assignment_map(profile, profile.experiment.random_seed)
    tape_assignment = assignments[MethodID.DA_PLCBF_FULL]
    regenerated = generate_scenario_tape(
        tape_assignment.scenario_root_seed,
        intervened_scenario_config(
            ConditionID.FALSIFICATION_COMBINED.value, profile.experiment, intervention
        ),
        fold=tape_assignment.scenario_fold,
    )
    if (
        int(tape.root_seed) != tape_assignment.scenario_root_seed
        or int(tape.generation_fold) != tape_assignment.scenario_fold
        or tape.sha256 != regenerated.sha256
    ):
        raise ValueError("counterexample replay tape does not deterministically regenerate")

    rows = manifest["methods"]
    if not isinstance(rows, list) or len(rows) != len(MethodID):
        raise ValueError("counterexample replay must retain all seven method outcomes")
    methods_root = directory / "methods"
    if not methods_root.is_dir():
        raise ValueError("counterexample replay methods directory is missing")
    successful_names: set[str] = set()
    expected_row_keys = {
        "order",
        "method",
        "status",
        "execution_role",
        "minimum_hard_margin",
        "trace_sha256",
        "trace_file_sha256",
        "events_file_sha256",
        "metrics_file_sha256",
        "error_type",
        "error_message",
    }
    for order, (method, row) in enumerate(zip(MethodID, rows, strict=True), start=1):
        role = "discovery_evaluation" if method is MethodID.DA_PLCBF_FULL else "matched_tape_replay"
        if (
            not isinstance(row, dict)
            or set(row) != expected_row_keys
            or row["order"] != order
            or row["method"] != method_spec(method).metadata()
            or row["execution_role"] != role
        ):
            raise ValueError("counterexample replay method row provenance is invalid")
        method_directory = methods_root / method.value
        if row["status"] == "success":
            expected = _successful_replay_record(
                method_directory,
                tape,
                method,
                profile,
                assignments[method],
                order=order,
                execution_role=role,
                strict_physical_validation=strict_physical_validation,
            )
            if row != expected:
                raise ValueError("counterexample replay method row disagrees with artifacts")
            successful_names.add(method.value)
        elif row["status"] == "operational_failure":
            if method_directory.exists() or any(
                row[name] is not None
                for name in (
                    "minimum_hard_margin",
                    "trace_sha256",
                    "trace_file_sha256",
                    "events_file_sha256",
                    "metrics_file_sha256",
                )
            ):
                raise ValueError("failed replay method must not expose partial success artifacts")
            if not isinstance(row["error_type"], str) or not isinstance(row["error_message"], str):
                raise ValueError("failed replay method must retain its operational error")
        else:
            raise ValueError("counterexample replay method status is invalid")
    if {item.name for item in methods_root.iterdir()} != successful_names:
        raise ValueError("counterexample replay contains unexpected method directories")
    return manifest


def _write_counterexample_replay(
    profile: FalsificationProfile,
    evaluator: DAFalsificationEvaluator,
    call: EvaluationCallEvidence,
    output_directory: Path,
    *,
    rank: int,
    result_sha256: str,
) -> dict[str, Any]:
    candidate_sha256 = call.candidate_sha256
    target = output_directory / f"rank-{rank:03d}-{candidate_sha256}"
    if target.exists():
        return _validate_replay_directory(
            target,
            profile,
            rank=rank,
            candidate_sha256=candidate_sha256,
            values=call.values,
            result_sha256=result_sha256,
            evaluator_sha256=evaluator.evaluator_sha256,
            source_tree_sha256=evaluator.source_tree_sha256,
            strict_physical_validation=evaluator.strict_physical_validation,
        )
    condition = ConditionID.FALSIFICATION_COMBINED.value
    source = evaluator.cache_directory / candidate_sha256 / "conditions" / condition
    cached = evaluator._load_cache(candidate_sha256, call.values)
    if cached is None or cached.margin != call.margin or len(cached.trials) != 1:
        raise ValueError("ranked counterexample discovery cache is unavailable or inconsistent")
    source_tape = source / "tape.npz"
    tape = load_scenario_tape(source_tape)
    if tape.sha256 != cached.trials[0].tape_sha256:
        raise ValueError("ranked counterexample tape disagrees with discovery evidence")
    intervention = decode_falsification_intervention(call.values)
    assignments = _replay_assignment_map(profile, evaluator.root_seed)
    output_directory.mkdir(parents=True, exist_ok=True)
    staging_root = output_directory / ".staging"
    staging_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=staging_root, prefix=f"{candidate_sha256}."
    ) as temporary_name:
        temporary = Path(temporary_name)
        shutil.copy2(source_tape, temporary / "tape.npz")
        methods_root = temporary / "methods"
        methods_root.mkdir()
        rows: list[dict[str, Any]] = []
        for order, method in enumerate(MethodID, start=1):
            role = (
                "discovery_evaluation"
                if method is MethodID.DA_PLCBF_FULL
                else "matched_tape_replay"
            )
            method_directory = methods_root / method.value
            if method is MethodID.DA_PLCBF_FULL:
                method_directory.mkdir()
                for name in ("trace.npz", "events.jsonl", "metrics.json"):
                    shutil.copy2(source / name, method_directory / name)
                row = _successful_replay_record(
                    method_directory,
                    tape,
                    method,
                    profile,
                    assignments[method],
                    order=order,
                    execution_role=role,
                    strict_physical_validation=evaluator.strict_physical_validation,
                )
                if (
                    row["minimum_hard_margin"] != call.margin
                    or row["trace_sha256"] != cached.trials[0].trace_sha256
                ):
                    raise ValueError("full-method replay artifacts disagree with discovery result")
                rows.append(row)
                continue
            try:
                assignment = assignments[method]
                obstacle_count = tape.static_positions.shape[0] + tape.dynamic_positions.shape[1]
                resources = evaluator._resource_builder(
                    profile.experiment,
                    obstacle_count=obstacle_count,
                    initialization_seed=int(assignment.shared_stochastic_seed & 0xFFFFFFFF),
                )
                resources = _intervened_resources(resources, intervention)
                run = evaluator._trial_runner(
                    assignment,
                    tape,
                    profile.experiment,
                    resources=resources,
                    executable_cache=evaluator._executable_cache,
                )
                if run.assignment.key != assignment.key:
                    raise RuntimeError("replay runner returned a mismatched trial assignment")
                run.trace.validate()
                run.scientific_metrics.validate()
                exact_margin = float(run.scientific_metrics.minimum_hard_margin)
                if exact_margin != float(np.min(run.trace.hard_barriers)):
                    raise RuntimeError("replay scientific margin does not match immutable trace")
                method_directory.mkdir()
                save_trace(run.trace, method_directory / "trace.npz")
                write_events(run.events, method_directory / "events.jsonl", trace=run.trace)
                write_metrics(run.trace, method_directory / "metrics.json")
                row = _successful_replay_record(
                    method_directory,
                    tape,
                    method,
                    profile,
                    assignment,
                    order=order,
                    execution_role=role,
                    strict_physical_validation=evaluator.strict_physical_validation,
                )
                if row["minimum_hard_margin"] != exact_margin:
                    raise RuntimeError("saved replay margin changed during strict validation")
                rows.append(row)
            except Exception as error:
                shutil.rmtree(method_directory, ignore_errors=True)
                rows.append(_failed_replay_record(method, error, order=order, execution_role=role))
        manifest = {
            "schema_version": _REPLAY_SCHEMA_VERSION,
            "artifact_type": "crazyflow-da-plcbf-seven-method-replay",
            "profile": profile.metadata(),
            "rank": rank,
            "candidate_sha256": candidate_sha256,
            "values": list(call.values),
            "condition": condition,
            "falsification_result_sha256": result_sha256,
            "evaluator_sha256": evaluator.evaluator_sha256,
            "source_tree_sha256": evaluator.source_tree_sha256,
            "strict_physical_validation": evaluator.strict_physical_validation,
            "tape_sha256": tape.sha256,
            "tape_file_sha256": _file_sha256(temporary / "tape.npz"),
            "methods": rows,
            "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
        }
        (temporary / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        _validate_replay_directory(
            temporary,
            profile,
            rank=rank,
            candidate_sha256=candidate_sha256,
            values=call.values,
            result_sha256=result_sha256,
            evaluator_sha256=evaluator.evaluator_sha256,
            source_tree_sha256=evaluator.source_tree_sha256,
            strict_physical_validation=evaluator.strict_physical_validation,
        )
        os.replace(temporary, target)
    return _validate_replay_directory(
        target,
        profile,
        rank=rank,
        candidate_sha256=candidate_sha256,
        values=call.values,
        result_sha256=result_sha256,
        evaluator_sha256=evaluator.evaluator_sha256,
        source_tree_sha256=evaluator.source_tree_sha256,
        strict_physical_validation=evaluator.strict_physical_validation,
    )


def run_descriptive_counterexample_replays(
    profile: FalsificationProfile,
    evaluator: DAFalsificationEvaluator,
    ranking: Sequence[Mapping[str, Any]],
    output_directory: str | os.PathLike[str],
    *,
    result_sha256: str,
) -> tuple[dict[str, Any], ...]:
    """Actually replay bounded worst retained tapes across all seven methods descriptively."""
    profile.validate()
    if evaluator.profile != profile:
        raise ValueError("replay evaluator and profile must be identical")
    _sha256_bytes(result_sha256, "result_sha256")
    calls_by_digest = {
        call.candidate_sha256: call for call in evaluator.calls if call.status == "success"
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _cleanup_known_atomic_temporaries(output, ("replay_summary.json",))
    staging = output / ".staging"
    staging.mkdir(exist_ok=True)
    for orphan in tuple(staging.iterdir()):
        if orphan.is_dir() and not orphan.is_symlink():
            shutil.rmtree(orphan)
        else:
            orphan.unlink()
    summaries: list[dict[str, Any]] = []
    selected = tuple(ranking[: profile.maximum_replayed_counterexamples])
    for expected_rank, item in enumerate(selected, start=1):
        if item.get("rank") != expected_rank:
            raise ValueError("ranked replay inputs must be contiguous and worst-first")
        candidate_sha256 = item.get("candidate_sha256")
        _sha256_bytes(candidate_sha256, "candidate_sha256")
        try:
            call = calls_by_digest[candidate_sha256]
        except KeyError as error:
            raise ValueError(
                "ranked replay input lacks matching successful call evidence"
            ) from error
        if tuple(item.get("values", ())) != call.values:
            raise ValueError("ranked replay values disagree with successful call evidence")
        manifest = _write_counterexample_replay(
            profile, evaluator, call, output, rank=expected_rank, result_sha256=result_sha256
        )
        success_count = sum(row["status"] == "success" for row in manifest["methods"])
        target_name = f"rank-{expected_rank:03d}-{candidate_sha256}"
        summaries.append(
            {
                "rank": expected_rank,
                "candidate_sha256": candidate_sha256,
                "directory": target_name,
                "manifest_file_sha256": _file_sha256(output / target_name / "manifest.json"),
                "successful_methods": success_count,
                "operational_failures": len(MethodID) - success_count,
            }
        )
    summary_document = {
        "schema_version": 2,
        "artifact_type": "crazyflow-da-plcbf-seven-method-replay-summary",
        "profile": profile.metadata(),
        "falsification_result_sha256": result_sha256,
        "evaluator_sha256": evaluator.evaluator_sha256,
        "source_tree_sha256": evaluator.source_tree_sha256,
        "strict_physical_validation": evaluator.strict_physical_validation,
        "replay_limit": profile.maximum_replayed_counterexamples,
        "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
        "replays": summaries,
    }
    _write_new_or_identical(output / "replay_summary.json", summary_document)
    return tuple(summaries)


def _write_new_or_identical(path: Path, document: Mapping[str, Any]) -> None:
    payload = _canonical_json(document) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"existing artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
        temporary = Path(out.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cleanup_known_atomic_temporaries(directory: Path, names: Sequence[str]) -> None:
    """Remove only tool-owned interrupted atomic-write temporaries in a scoped directory."""
    if not directory.is_dir():
        return
    prefixes = tuple(f".{name}." for name in names)
    for path in tuple(directory.iterdir()):
        if path.name.startswith(prefixes):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"unsafe interrupted artifact temporary: {path}")
            path.unlink()


def _call_document(call: EvaluationCallEvidence, index: int) -> dict[str, Any]:
    return {
        "evaluation_index": index,
        "candidate_sha256": call.candidate_sha256,
        "values": list(call.values),
        "status": call.status,
        "margin": call.margin,
        "trials": [asdict(item) for item in call.trials],
        "error_type": call.error_type,
        "error_message": call.error_message,
    }


def _configuration_document(
    profile: FalsificationProfile,
    *,
    root_seed: int,
    candidate_set_sha256: str,
    source_tree_sha256: str,
    evaluator_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": FALSIFICATION_EXPERIMENT_SCHEMA_VERSION,
        "artifact_type": "crazyflow-da-plcbf-falsification-configuration",
        "profile": profile.metadata(),
        "root_seed": root_seed,
        "candidate_set_sha256": candidate_set_sha256,
        "source_tree_sha256": source_tree_sha256,
        "evaluator_sha256": evaluator_sha256,
        "strict_physical_validation": True,
        "evaluation_budget": profile.search.evaluation_budget(
            generate_falsification_candidates(profile, root_seed=root_seed)
        ),
        "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
    }


def _expected_summary(
    profile: FalsificationProfile,
    result: FalsificationResult,
    ranking: Sequence[Mapping[str, Any]],
    replays: Sequence[Mapping[str, Any]],
    *,
    candidate_set_sha256: str,
    evaluator_sha256: str,
    source_tree_sha256: str,
) -> dict[str, Any]:
    return {
        "profile": profile.name,
        "evaluation_budget": result.evaluation_budget,
        "successful_evaluations": sum(
            item.status is FalsificationStatus.SUCCESS for item in result.evaluations
        ),
        "operational_failures": len(result.failures),
        "counterexamples": len(result.counterexamples),
        "unique_ranked_counterexamples": len(ranking),
        "descriptive_replays": len(replays),
        "replay_operational_failures": sum(int(item["operational_failures"]) for item in replays),
        "result_sha256": result.sha256,
        "candidate_set_sha256": candidate_set_sha256,
        "evaluator_sha256": evaluator_sha256,
        "source_tree_sha256": source_tree_sha256,
        "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
    }


def _artifact_file_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"falsification artifact contains a symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"falsification artifact contains a non-file: {relative}")
        name = relative.as_posix()
        if name in {"manifest.json", "complete.marker"}:
            continue
        if ".staging" in relative.parts:
            raise ValueError("falsification staging directory is not empty")
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)})
    return rows


def _campaign_content_sha256(files: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        b"crazyflow.da_plcbf.falsification-campaign-content.v1\0" + _canonical_json(list(files))
    ).hexdigest()


def _manifest_document(
    root: Path, configuration: Mapping[str, Any], *, result_sha256: str, summary: Mapping[str, Any]
) -> dict[str, Any]:
    files = _artifact_file_rows(root)
    return {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "artifact_type": "crazyflow-da-plcbf-falsification-campaign-manifest",
        "profile": configuration["profile"]["name"],
        "root_seed": configuration["root_seed"],
        "source_tree_sha256": configuration["source_tree_sha256"],
        "candidate_set_sha256": configuration["candidate_set_sha256"],
        "evaluator_sha256": configuration["evaluator_sha256"],
        "falsification_result_sha256": result_sha256,
        "campaign_content_sha256": _campaign_content_sha256(files),
        "summary": dict(summary),
        "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
        "files": files,
    }


def _complete_marker_document(manifest: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    summary = manifest["summary"]
    return {
        "schema_version": _COMPLETE_MARKER_SCHEMA_VERSION,
        "artifact_type": "crazyflow-da-plcbf-falsification-complete-marker",
        "execution_complete": True,
        "manifest_file_sha256": _file_sha256(manifest_path),
        "campaign_content_sha256": manifest["campaign_content_sha256"],
        "source_tree_sha256": manifest["source_tree_sha256"],
        "falsification_result_sha256": manifest["falsification_result_sha256"],
        "evaluation_budget": summary["evaluation_budget"],
        "successful_evaluations": summary["successful_evaluations"],
        "operational_failures": summary["operational_failures"],
        "counterexamples": summary["counterexamples"],
        "descriptive_replays": summary["descriptive_replays"],
        "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
    }


def _reconstruct_calls(
    evaluator: DAFalsificationEvaluator, result: FalsificationResult
) -> tuple[EvaluationCallEvidence, ...]:
    calls: list[EvaluationCallEvidence] = []
    successful_digests: set[str] = set()
    failure_digests: set[str] = set()
    for row in result.evaluations:
        digest = evaluator.candidate_sha256(row.values)
        if row.status is FalsificationStatus.SUCCESS:
            evidence = evaluator._load_cache(digest, row.values)
            if evidence is None:
                raise ValueError("successful falsification row lacks a validated candidate cache")
            calls.append(evidence)
            successful_digests.add(digest)
        else:
            evidence = evaluator._load_failure(digest, row.values)
            if evidence is None:
                raise ValueError("failed falsification row lacks retained failure evidence")
            calls.append(evidence)
            failure_digests.add(digest)
    if successful_digests & failure_digests:
        raise ValueError("one candidate cannot be both a retained success and failure")
    expected_cache_entries = successful_digests | {"operational_failures", ".staging"}
    if {path.name for path in evaluator.cache_directory.iterdir()} != expected_cache_entries:
        raise ValueError("candidate cache contains orphan or unreferenced entries")
    failure_root = evaluator.cache_directory / "operational_failures"
    expected_failures = {f"{digest}.json" for digest in failure_digests}
    if (
        not failure_root.is_dir()
        or {path.name for path in failure_root.iterdir()} != expected_failures
    ):
        raise ValueError("operational-failure cache contains orphan or missing records")
    staging = evaluator.cache_directory / ".staging"
    if not staging.is_dir() or any(staging.iterdir()):
        raise ValueError("candidate cache staging directory is not empty")
    reconstructed = tuple(calls)
    _validate_search_calls(result, reconstructed)
    return reconstructed


def verify_da_plcbf_falsification(
    output_directory: str | os.PathLike[str],
    *,
    repository: str | os.PathLike[str] | None = None,
    require_current_source: bool = True,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Reconstruct and strictly verify one persisted falsification campaign.

    Stored rankings, summaries, checksums, and booleans are assertions only.  This function
    regenerates the fixed candidate set/tapes, replays every successful true-plant transition,
    recomputes metrics/ranking, and validates every seven-method replay before accepting them.
    """
    root = Path(output_directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError("falsification campaign directory is missing")
    expected_root_entries = {
        "configuration.json",
        "cache",
        "falsification_result.json",
        "replays",
        "orchestration.json",
        "ranked_counterexamples.json",
        "seven_method_replay_registry.json",
    }
    if (root / "manifest.json").exists():
        expected_root_entries.add("manifest.json")
    if (root / "complete.marker").exists():
        expected_root_entries.add("complete.marker")
    if {path.name for path in root.iterdir()} != expected_root_entries:
        raise ValueError("falsification campaign root contains orphan or missing entries")
    configuration = _read_object(root / "configuration.json")
    expected_configuration_keys = {
        "schema_version",
        "artifact_type",
        "profile",
        "root_seed",
        "candidate_set_sha256",
        "source_tree_sha256",
        "evaluator_sha256",
        "strict_physical_validation",
        "evaluation_budget",
        "claim_boundary",
    }
    if set(configuration) != expected_configuration_keys:
        raise ValueError("falsification configuration schema is invalid")
    profile_name = configuration["profile"].get("name")
    root_seed = configuration["root_seed"]
    if (
        not isinstance(profile_name, str)
        or isinstance(root_seed, bool)
        or not isinstance(root_seed, int)
    ):
        raise ValueError("falsification configuration profile/root seed is invalid")
    profile = falsification_profile(profile_name, root_seed=root_seed)
    candidates = generate_falsification_candidates(profile, root_seed=root_seed)
    candidate_digest = boundary_candidate_set_sha256(candidates)
    source_digest = configuration["source_tree_sha256"]
    _sha256_bytes(source_digest, "source_tree_sha256")
    evaluator_digest = falsification_evaluator_sha256(
        profile,
        candidate_set_sha256=candidate_digest,
        source_tree_sha256=source_digest,
        strict_physical_validation=True,
    )
    expected_configuration = _configuration_document(
        profile,
        root_seed=root_seed,
        candidate_set_sha256=candidate_digest,
        source_tree_sha256=source_digest,
        evaluator_sha256=evaluator_digest,
    )
    if configuration != expected_configuration:
        raise ValueError("falsification configuration does not canonically reconstruct")
    if require_current_source:
        current = falsification_source_tree_sha256(repository)
        if current != source_digest:
            raise ValueError("current source tree differs from falsification campaign source")

    result = load_falsification_result(root / "falsification_result.json")
    if (
        result.search_name != f"da-plcbf-{profile.name}"
        or result.evaluator_name != "da-plcbf-full-run-trial-minimum-hard-margin"
        or result.evaluator_sha256 != evaluator_digest
        or result.config != profile.search
        or boundary_candidate_set_sha256(result.candidates) != candidate_digest
        or result.evaluation_budget != configuration["evaluation_budget"]
    ):
        raise ValueError("falsification result does not match the fixed campaign configuration")
    evaluator = DAFalsificationEvaluator(
        profile,
        root_seed=root_seed,
        candidate_set_sha256=candidate_digest,
        cache_directory=root / "cache",
        source_tree_sha256=source_digest,
        strict_physical_validation=True,
        validate_only=True,
    )
    calls = _reconstruct_calls(evaluator, result)
    ranking = rank_unique_counterexamples(
        result, calls, limit=profile.maximum_ranked_counterexamples
    )

    replay_root = root / "replays"
    selected = ranking[: profile.maximum_replayed_counterexamples]
    replay_summaries: list[dict[str, Any]] = []
    expected_replay_entries = {"replay_summary.json", ".staging"}
    for expected_rank, item in enumerate(selected, start=1):
        digest = str(item["candidate_sha256"])
        directory_name = f"rank-{expected_rank:03d}-{digest}"
        expected_replay_entries.add(directory_name)
        manifest = _validate_replay_directory(
            replay_root / directory_name,
            profile,
            rank=expected_rank,
            candidate_sha256=digest,
            values=item["values"],
            result_sha256=result.sha256,
            evaluator_sha256=evaluator_digest,
            source_tree_sha256=source_digest,
            strict_physical_validation=True,
        )
        successes = sum(row["status"] == "success" for row in manifest["methods"])
        replay_summaries.append(
            {
                "rank": expected_rank,
                "candidate_sha256": digest,
                "directory": directory_name,
                "manifest_file_sha256": _file_sha256(
                    replay_root / directory_name / "manifest.json"
                ),
                "successful_methods": successes,
                "operational_failures": len(MethodID) - successes,
            }
        )
    if (
        not replay_root.is_dir()
        or {path.name for path in replay_root.iterdir()} != expected_replay_entries
    ):
        raise ValueError("replay directory contains orphan or missing entries")
    replay_staging = replay_root / ".staging"
    if not replay_staging.is_dir() or any(replay_staging.iterdir()):
        raise ValueError("replay staging directory is not empty")
    replay_summary = {
        "schema_version": 2,
        "artifact_type": "crazyflow-da-plcbf-seven-method-replay-summary",
        "profile": profile.metadata(),
        "falsification_result_sha256": result.sha256,
        "evaluator_sha256": evaluator_digest,
        "source_tree_sha256": source_digest,
        "strict_physical_validation": True,
        "replay_limit": profile.maximum_replayed_counterexamples,
        "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
        "replays": replay_summaries,
    }
    if _read_object(replay_root / "replay_summary.json") != replay_summary:
        raise ValueError("replay summary does not recompute from replay artifacts")

    expected_orchestration = {
        "schema_version": FALSIFICATION_EXPERIMENT_SCHEMA_VERSION,
        "artifact_type": "crazyflow-da-plcbf-falsification-orchestration",
        "profile": profile.metadata(),
        "root_seed": root_seed,
        "candidate_set_sha256": candidate_digest,
        "source_tree_sha256": source_digest,
        "evaluator_sha256": evaluator_digest,
        "strict_physical_validation": True,
        "falsification_result_sha256": result.sha256,
        "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
        "generic_claim_boundary": FALSIFICATION_CLAIM_BOUNDARY,
        "evaluations": [_call_document(call, index) for index, call in enumerate(calls)],
        "descriptive_seven_method_replays": replay_summaries,
    }
    if _read_object(root / "orchestration.json") != expected_orchestration:
        raise ValueError("orchestration does not recompute from retained calls")
    expected_ranking = {
        "schema_version": 2,
        "result_sha256": result.sha256,
        "source_tree_sha256": source_digest,
        "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
        "counterexamples": list(ranking),
    }
    if _read_object(root / "ranked_counterexamples.json") != expected_ranking:
        raise ValueError("ranked counterexamples do not recompute from fixed search outcomes")
    expected_registry = {
        "schema_version": 2,
        "result_sha256": result.sha256,
        "source_tree_sha256": source_digest,
        "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
        "methods": list(seven_method_replay_registry()),
        "executed_replays": replay_summaries,
    }
    if _read_object(root / "seven_method_replay_registry.json") != expected_registry:
        raise ValueError("seven-method registry does not match the fixed descriptive protocol")

    summary = _expected_summary(
        profile,
        result,
        ranking,
        replay_summaries,
        candidate_set_sha256=candidate_digest,
        evaluator_sha256=evaluator_digest,
        source_tree_sha256=source_digest,
    )
    manifest_path = root / "manifest.json"
    if require_complete or manifest_path.exists():
        manifest = _read_object(manifest_path)
        expected_manifest = _manifest_document(
            root, configuration, result_sha256=result.sha256, summary=summary
        )
        if manifest != expected_manifest:
            raise ValueError("campaign manifest does not bind the canonical complete file set")
        marker_path = root / "complete.marker"
        if require_complete or marker_path.exists():
            marker = _read_object(marker_path)
            expected_marker = _complete_marker_document(manifest, manifest_path)
            if marker != expected_marker:
                raise ValueError("campaign completion marker does not bind the manifest")
    report = dict(summary)
    report["current_source_verified"] = require_current_source
    report["historical_inspection_only"] = not require_current_source
    return report


def run_da_plcbf_falsification(
    profile_name: str,
    output_directory: str | os.PathLike[str],
    *,
    root_seed: int = 20260831,
    repository: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Execute/resume one fixed profile and atomically finalize strict empirical evidence."""
    profile = falsification_profile(profile_name, root_seed=root_seed)
    candidates = generate_falsification_candidates(profile, root_seed=root_seed)
    candidate_digest = boundary_candidate_set_sha256(candidates)
    source_digest = falsification_source_tree_sha256(repository)
    evaluator_digest = falsification_evaluator_sha256(
        profile,
        candidate_set_sha256=candidate_digest,
        source_tree_sha256=source_digest,
        strict_physical_validation=True,
    )
    configuration = _configuration_document(
        profile,
        root_seed=root_seed,
        candidate_set_sha256=candidate_digest,
        source_tree_sha256=source_digest,
        evaluator_sha256=evaluator_digest,
    )
    output = Path(output_directory)
    if output.exists() and not output.is_dir():
        raise FileExistsError("falsification output exists and is not a directory")
    output.mkdir(parents=True, exist_ok=True)
    _cleanup_known_atomic_temporaries(
        output,
        (
            "configuration.json",
            "falsification_result.json",
            "orchestration.json",
            "ranked_counterexamples.json",
            "seven_method_replay_registry.json",
            "manifest.json",
            "complete.marker",
        ),
    )
    configuration_path = output / "configuration.json"
    if not configuration_path.exists() and any(output.iterdir()):
        raise FileExistsError("nonempty falsification output lacks a bound configuration")
    _write_new_or_identical(configuration_path, configuration)
    if (output / "complete.marker").exists():
        return verify_da_plcbf_falsification(
            output, repository=repository, require_current_source=True, require_complete=True
        )
    evaluator = DAFalsificationEvaluator(
        profile,
        root_seed=root_seed,
        candidate_set_sha256=candidate_digest,
        cache_directory=output / "cache",
        repository=repository,
        source_tree_sha256=source_digest,
        strict_physical_validation=True,
    )
    result = run_fixed_budget_falsification(
        candidates,
        evaluator,
        search_name=f"da-plcbf-{profile.name}",
        evaluator_name="da-plcbf-full-run-trial-minimum-hard-margin",
        evaluator_sha256=evaluator.evaluator_sha256,
        config=profile.search,
    )
    _validate_search_calls(result, evaluator.calls)
    result_path = output / "falsification_result.json"
    if result_path.exists():
        if load_falsification_result(result_path).sha256 != result.sha256:
            raise FileExistsError("existing falsification result differs")
    else:
        save_falsification_result(result, result_path)
    ranking = rank_unique_counterexamples(
        result, evaluator.calls, limit=profile.maximum_ranked_counterexamples
    )
    replays = run_descriptive_counterexample_replays(
        profile, evaluator, ranking, output / "replays", result_sha256=result.sha256
    )
    orchestration = {
        "schema_version": FALSIFICATION_EXPERIMENT_SCHEMA_VERSION,
        "artifact_type": "crazyflow-da-plcbf-falsification-orchestration",
        "profile": profile.metadata(),
        "root_seed": root_seed,
        "candidate_set_sha256": candidate_digest,
        "source_tree_sha256": source_digest,
        "evaluator_sha256": evaluator.evaluator_sha256,
        "strict_physical_validation": True,
        "falsification_result_sha256": result.sha256,
        "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
        "generic_claim_boundary": FALSIFICATION_CLAIM_BOUNDARY,
        "evaluations": [_call_document(call, index) for index, call in enumerate(evaluator.calls)],
        "descriptive_seven_method_replays": list(replays),
    }
    _write_new_or_identical(output / "orchestration.json", orchestration)
    _write_new_or_identical(
        output / "ranked_counterexamples.json",
        {
            "schema_version": 2,
            "result_sha256": result.sha256,
            "source_tree_sha256": source_digest,
            "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
            "counterexamples": list(ranking),
        },
    )
    _write_new_or_identical(
        output / "seven_method_replay_registry.json",
        {
            "schema_version": 2,
            "result_sha256": result.sha256,
            "source_tree_sha256": source_digest,
            "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
            "methods": list(seven_method_replay_registry()),
            "executed_replays": list(replays),
        },
    )
    if falsification_source_tree_sha256(repository) != source_digest:
        raise RuntimeError("source tree changed while falsification evidence was executing")
    summary = _expected_summary(
        profile,
        result,
        ranking,
        replays,
        candidate_set_sha256=candidate_digest,
        evaluator_sha256=evaluator.evaluator_sha256,
        source_tree_sha256=source_digest,
    )
    manifest = _manifest_document(
        output, configuration, result_sha256=result.sha256, summary=summary
    )
    manifest_path = output / "manifest.json"
    _write_new_or_identical(manifest_path, manifest)
    _write_new_or_identical(
        output / "complete.marker", _complete_marker_document(manifest, manifest_path)
    )
    return verify_da_plcbf_falsification(
        output, repository=repository, require_current_source=True, require_complete=True
    )


def probe_falsification_candidate(
    profile_name: str,
    cache_directory: str | os.PathLike[str],
    *,
    candidate_index: int = 0,
    root_seed: int = 20260831,
) -> dict[str, Any]:
    """Run one genuine full-method candidate for a bounded GPU smoke integration probe."""
    profile = falsification_profile(profile_name, root_seed=root_seed)
    candidates = generate_falsification_candidates(profile, root_seed=root_seed)
    if isinstance(candidate_index, bool) or not 0 <= candidate_index < candidates.count:
        raise ValueError("candidate_index lies outside the fixed randomized candidate set")
    candidate_set_digest = boundary_candidate_set_sha256(candidates)
    evaluator = DAFalsificationEvaluator(
        profile,
        root_seed=root_seed,
        candidate_set_sha256=candidate_set_digest,
        cache_directory=cache_directory,
    )
    values = candidates.values[candidate_index]
    margin = evaluator(values)
    evidence = evaluator.calls[-1]
    return {
        "profile": profile.name,
        "candidate_index": candidate_index,
        "candidate_sha256": evidence.candidate_sha256,
        "candidate_set_sha256": candidate_set_digest,
        "evaluator_sha256": evaluator.evaluator_sha256,
        "source_tree_sha256": evaluator.source_tree_sha256,
        "minimum_hard_margin": margin,
        "counterexample": margin <= profile.search.counterexample_margin,
        "condition_trials": [asdict(item) for item in evidence.trials],
        "claim_boundary": FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
    }


__all__ = [
    "DAFalsificationEvaluator",
    "DecodedIntervention",
    "EvaluationCallEvidence",
    "FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY",
    "FALSIFICATION_EXPERIMENT_SCHEMA_VERSION",
    "FalsificationProfile",
    "RetainedOperationalFailure",
    "decode_falsification_intervention",
    "falsification_boundary_variables",
    "falsification_evaluator_sha256",
    "falsification_profile",
    "falsification_source_tree_sha256",
    "generate_falsification_candidates",
    "intervened_scenario_config",
    "probe_falsification_candidate",
    "rank_unique_counterexamples",
    "run_da_plcbf_falsification",
    "run_descriptive_counterexample_replays",
    "seven_method_replay_registry",
    "verify_da_plcbf_falsification",
]
