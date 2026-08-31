from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from crazyflow.safety.da_plcbf.artifact_smoke import synthetic_trace
from crazyflow.safety.da_plcbf.artifacts import ArtifactEvent
from crazyflow.safety.da_plcbf.baselines import MethodID
from crazyflow.safety.da_plcbf.experiments import build_experiment_resources
from crazyflow.safety.da_plcbf.falsification import (
    boundary_candidate_set_sha256,
    load_falsification_result,
    run_fixed_budget_falsification,
)
from crazyflow.safety.da_plcbf.falsification_experiments import (
    FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY,
    ConditionTrialEvidence,
    DAFalsificationEvaluator,
    EvaluationCallEvidence,
    RetainedOperationalFailure,
    _canonical_json,
    _complete_marker_document,
    _intervened_resources,
    _manifest_document,
    _read_object,
    decode_falsification_intervention,
    falsification_boundary_variables,
    falsification_evaluator_sha256,
    falsification_profile,
    falsification_source_tree_sha256,
    generate_falsification_candidates,
    intervened_scenario_config,
    rank_unique_counterexamples,
    run_da_plcbf_falsification,
    run_descriptive_counterexample_replays,
    seven_method_replay_registry,
    verify_da_plcbf_falsification,
)
from crazyflow.safety.da_plcbf.scenarios import generate_scenario_tape
from crazyflow.safety.da_plcbf.scientific_evaluation import FalsificationAxis

if TYPE_CHECKING:
    from pathlib import Path


def _nominal_values() -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(variable.nominal, dtype=np.float64)
            for variable in falsification_boundary_variables()
        ]
    )


def test_fourteen_dimensional_decode_is_bounded_and_covers_every_required_axis() -> None:
    variables = falsification_boundary_variables()
    values = _nominal_values()
    decoded = decode_falsification_intervention(values)

    assert values.shape == (14,)
    assert {variable.axis for variable in variables} == {
        FalsificationAxis.INITIAL_STATE,
        FalsificationAxis.OBSTACLE_TIMING,
        FalsificationAxis.WIND,
        FalsificationAxis.MASS,
        FalsificationAxis.ROTOR_EFFICIENCY,
        FalsificationAxis.ESTIMATOR_ERROR,
        FalsificationAxis.ACTUATOR_SATURATION,
    }
    assert decoded.values() == tuple(values)
    with pytest.raises(ValueError, match=r"float64\[14\]"):
        decode_falsification_intervention(values[:-1])
    bad = values.copy()
    bad[0] = 10.0
    with pytest.raises(ValueError, match="declared bounds"):
        decode_falsification_intervention(bad)


def test_profiles_are_fixed_deterministic_and_scenario_interventions_validate() -> None:
    for name, count, budget, replay_limit in (
        ("smoke", 28, 56, 1),
        ("development", 56, 168, 3),
        ("final", 280, 728, 10),
    ):
        profile = falsification_profile(name, root_seed=17)
        candidates = generate_falsification_candidates(profile, root_seed=17)
        repeated = generate_falsification_candidates(profile, root_seed=17)
        assert candidates.count == count
        assert profile.search.evaluation_budget(candidates) == budget
        assert profile.experiment.control_steps >= 3
        assert profile.maximum_replayed_counterexamples == replay_limit
        np.testing.assert_array_equal(candidates.values, repeated.values)
        assert boundary_candidate_set_sha256(candidates) == boundary_candidate_set_sha256(repeated)

    profile = falsification_profile("smoke", root_seed=17)
    decoded = decode_falsification_intervention(_nominal_values())
    assert profile.conditions == ("falsification_combined",)
    for condition in profile.conditions:
        config = intervened_scenario_config(condition, profile.experiment, decoded)
        config.validate()
        assert config.steps == (
            profile.experiment.control_steps + profile.experiment.certificate_horizon + 1
        )
        assert config.reference_initial_position is not None
        assert config.reference_initial_velocity is not None
        assert config.vehicle_initial_position == config.reference_initial_position
        assert config.vehicle_initial_velocity == config.reference_initial_velocity

    perturbed_values = _nominal_values()
    perturbed_values[:4] = (0.25, -0.10, 1.20, 0.05)
    perturbed = intervened_scenario_config(
        profile.conditions[0],
        profile.experiment,
        decode_falsification_intervention(perturbed_values),
    )
    assert perturbed.reference_initial_position == (0.0, 0.0, 1.5)
    assert perturbed.reference_initial_velocity == (0.45, 0.10, 0.0)
    assert perturbed.vehicle_initial_position == (0.25, 0.0, 1.5)
    assert perturbed.vehicle_initial_velocity == (0.35, 0.10, 0.0)
    assert perturbed.crossing_fraction_range == pytest.approx((0.65, 0.65))


def test_evaluator_digest_binds_profile_and_candidate_set() -> None:
    smoke = falsification_profile("smoke", root_seed=4)
    candidates = generate_falsification_candidates(smoke, root_seed=4)
    digest = boundary_candidate_set_sha256(candidates)

    first = falsification_evaluator_sha256(smoke, candidate_set_sha256=digest)
    repeated = falsification_evaluator_sha256(smoke, candidate_set_sha256=digest)
    development = falsification_evaluator_sha256(
        falsification_profile("development", root_seed=4), candidate_set_sha256=digest
    )

    assert first == repeated
    assert len(first) == 64
    assert first != development
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        falsification_evaluator_sha256(smoke, candidate_set_sha256="x" * 64)
    with pytest.raises(ValueError, match="root_seed must match"):
        generate_falsification_candidates(smoke, root_seed=5)


def test_evaluator_digest_binds_runtime_source_assets_and_dependency_locks(tmp_path: Path) -> None:
    package = tmp_path / "crazyflow" / "drones"
    example = tmp_path / "examples" / "da_plcbf"
    benchmark = tmp_path / "benchmark"
    for directory in (package, example, benchmark):
        directory.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (tmp_path / "pixi.lock").write_text("version: 1\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (package / "model.py").write_text("MASS = 1.0\n")
    mesh = package / "body.stl"
    mesh.write_bytes(b"solid body\nendsolid body\n")
    (example / "falsify.py").write_text("PROFILE = 'final'\n")
    (benchmark / "bptt.py").write_text("K = 64\n")

    before = falsification_source_tree_sha256(tmp_path)
    mesh.write_bytes(b"solid changed\nendsolid changed\n")
    after_mesh = falsification_source_tree_sha256(tmp_path)
    assert before != after_mesh
    (tmp_path / "uv.lock").write_text("version = 2\n")
    after_lock = falsification_source_tree_sha256(tmp_path)
    assert after_lock != after_mesh
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("not executable\n")
    assert falsification_source_tree_sha256(tmp_path) == after_lock


def test_every_declared_intervention_component_changes_combined_trial_inputs() -> None:
    profile = falsification_profile("smoke", root_seed=19)
    nominal_values = _nominal_values()
    nominal = decode_falsification_intervention(nominal_values)
    condition = profile.conditions[0]
    nominal_config = intervened_scenario_config(condition, profile.experiment, nominal)
    nominal_tape = generate_scenario_tape(23, nominal_config, fold=0)
    alternate_values = (-0.5, -0.3, 0.8, 0.05, 3.5, 0.2, 1.15, 1.10, 0.60, 0.35, 0.0, 0.0)

    for index, alternate in enumerate(alternate_values):
        changed_values = nominal_values.copy()
        changed_values[index] = alternate
        changed = decode_falsification_intervention(changed_values)
        changed_config = intervened_scenario_config(condition, profile.experiment, changed)
        changed_tape = generate_scenario_tape(23, changed_config, fold=0)
        assert changed_tape.sha256 != nominal_tape.sha256, (
            f"intervention component {index} is inert"
        )

    obstacle_count = (
        nominal_tape.static_positions.shape[0] + nominal_tape.dynamic_positions.shape[1]
    )
    resources = build_experiment_resources(
        profile.experiment, obstacle_count=obstacle_count, initialization_seed=7
    )
    nominal_resources = _intervened_resources(resources, nominal)
    for index, alternate in ((12, 0.55), (13, 0.65)):
        changed_values = nominal_values.copy()
        changed_values[index] = alternate
        changed_resources = _intervened_resources(
            resources, decode_falsification_intervention(changed_values)
        )
        assert not np.array_equal(
            np.asarray(changed_resources.actuator.thrust_max),
            np.asarray(nominal_resources.actuator.thrust_max),
        ), f"intervention component {index} is inert"


@dataclass(frozen=True)
class _FakeMetrics:
    minimum_hard_margin: float

    def validate(self) -> None:
        assert np.isfinite(self.minimum_hard_margin)


def _fake_runner(counter: list[str]) -> Any:
    def run(assignment: Any, tape: Any, config: Any, **_kwargs: Any) -> Any:
        counter.append(assignment.condition)
        trace = synthetic_trace(tape.sha256, steps=config.control_steps, dt=config.dt)
        margin = float(np.min(trace.hard_barriers))
        event = ArtifactEvent(
            sequence=0,
            step=0,
            time_seconds=0.0,
            category="runtime",
            name="trial_started",
            severity="info",
            snapshot_version=0,
            model_version=0,
            details={"method": assignment.method, "condition": assignment.condition},
        )
        return SimpleNamespace(
            assignment=assignment,
            tape=tape,
            trace=trace,
            events=(event,),
            scientific_metrics=_FakeMetrics(margin),
        )

    return run


def test_success_cache_is_written_only_after_strict_artifact_validation_and_reused(
    tmp_path: Path,
) -> None:
    profile = falsification_profile("smoke", root_seed=8)
    candidates = generate_falsification_candidates(profile, root_seed=8)
    candidate_set_digest = boundary_candidate_set_sha256(candidates)
    calls: list[str] = []
    evaluator = DAFalsificationEvaluator(
        profile,
        root_seed=8,
        candidate_set_sha256=candidate_set_digest,
        cache_directory=tmp_path / "cache",
        trial_runner=_fake_runner(calls),
        strict_physical_validation=False,
    )
    values = _nominal_values()

    first = evaluator(values)
    second = evaluator(values)

    assert first == second
    assert calls == list(profile.conditions)
    assert not evaluator.calls[0].cache_hit
    assert evaluator.calls[1].cache_hit
    cache = tmp_path / "cache" / evaluator.candidate_sha256(values)
    assert (cache / "manifest.json").is_file()
    for condition in profile.conditions:
        directory = cache / "conditions" / condition
        assert {path.name for path in directory.iterdir()} == {
            "tape.npz",
            "trace.npz",
            "events.jsonl",
            "metrics.json",
        }

    ranking = (
        {
            "rank": 1,
            "candidate_sha256": evaluator.calls[-1].candidate_sha256,
            "values": list(evaluator.calls[-1].values),
        },
    )
    replays = run_descriptive_counterexample_replays(
        profile, evaluator, ranking, tmp_path / "replays", result_sha256="a" * 64
    )
    assert replays[0]["successful_methods"] == 7
    assert replays[0]["operational_failures"] == 0
    assert len(calls) == 7
    repeated_replays = run_descriptive_counterexample_replays(
        profile, evaluator, ranking, tmp_path / "replays", result_sha256="a" * 64
    )
    assert repeated_replays == replays
    assert len(calls) == 7
    replay_manifest = json.loads(
        next((tmp_path / "replays").glob("rank-*/manifest.json")).read_text()
    )
    assert [row["method"]["method_id"] for row in replay_manifest["methods"]] == [
        item.value for item in MethodID
    ]
    assert (
        sum(row["execution_role"] == "discovery_evaluation" for row in replay_manifest["methods"])
        == 1
    )

    metrics = cache / "conditions" / profile.conditions[0] / "metrics.json"
    payload = json.loads(metrics.read_text())
    payload["minimum_hard_margin"] += 1.0
    metrics.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        evaluator(values)
    assert len(calls) == 7
    assert evaluator.calls[-1].status == "operational_failure"


def test_operational_runner_failure_is_exposed_to_generic_executor_and_never_cached(
    tmp_path: Path,
) -> None:
    profile = falsification_profile("smoke", root_seed=9)
    candidates = generate_falsification_candidates(profile, root_seed=9)
    candidate_set_digest = boundary_candidate_set_sha256(candidates)

    attempts = 0

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("retained-real-trial-failure")

    evaluator = DAFalsificationEvaluator(
        profile,
        root_seed=9,
        candidate_set_sha256=candidate_set_digest,
        cache_directory=tmp_path / "cache",
        trial_runner=fail,
    )
    with pytest.raises(RetainedOperationalFailure, match="retained-real-trial-failure"):
        evaluator(_nominal_values())
    with pytest.raises(RetainedOperationalFailure, match="retained-real-trial-failure"):
        evaluator(_nominal_values())

    assert evaluator.calls[-1].status == "operational_failure"
    assert evaluator.calls[-1].cache_hit
    assert attempts == 1
    assert not list((tmp_path / "cache").glob("[0-9a-f]*"))
    retained = list((tmp_path / "cache" / "operational_failures").glob("*.json"))
    assert len(retained) == 1


def test_orphan_candidate_cache_is_rejected_without_overwrite(tmp_path: Path) -> None:
    profile = falsification_profile("smoke", root_seed=12)
    candidates = generate_falsification_candidates(profile, root_seed=12)
    calls: list[str] = []
    evaluator = DAFalsificationEvaluator(
        profile,
        root_seed=12,
        candidate_set_sha256=boundary_candidate_set_sha256(candidates),
        cache_directory=tmp_path / "cache",
        trial_runner=_fake_runner(calls),
        strict_physical_validation=False,
    )
    digest = evaluator.candidate_sha256(_nominal_values())
    orphan = tmp_path / "cache" / digest
    orphan.mkdir()
    marker = orphan / "do-not-overwrite.txt"
    marker.write_text("orphan evidence\n")

    with pytest.raises(ValueError, match="candidate cache root entries"):
        evaluator(_nominal_values())
    assert marker.read_text() == "orphan evidence\n"
    assert calls == []


def test_whole_campaign_resume_source_drift_and_semantic_tamper_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keep this whole-campaign integrity regression CPU-fast. Separate integration coverage runs
    # the real evaluator and physical replay; this fixture only substitutes trial generation,
    # while still exercising every fixed search row, cache, replay, manifest, and verifier edge.
    import crazyflow.safety.da_plcbf.campaign_artifacts as campaign_artifacts

    calls: list[str] = []
    original_init = DAFalsificationEvaluator.__init__

    def fake_init(self: DAFalsificationEvaluator, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._trial_runner = _fake_runner(calls)

    monkeypatch.setattr(DAFalsificationEvaluator, "__init__", fake_init)
    monkeypatch.setattr(
        campaign_artifacts, "_validate_trace_physical_evidence", lambda *_args, **_kwargs: None
    )
    source_repository = tmp_path / "source"
    (source_repository / "crazyflow").mkdir(parents=True)
    (source_repository / "crazyflow" / "implementation.py").write_text("VERSION = 1\n")
    (source_repository / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    root = tmp_path / "campaign"
    first = run_da_plcbf_falsification("smoke", root, root_seed=31, repository=source_repository)
    assert first == verify_da_plcbf_falsification(
        root, repository=source_repository, require_current_source=True, require_complete=True
    )
    assert (root / "manifest.json").is_file()
    assert (root / "complete.marker").is_file()

    # Simulate interruption after the manifest was durably published but before the marker. A
    # resume must reuse all write-once cache/replay evidence and restore the identical marker.
    attempts = len(calls)
    (root / "complete.marker").unlink()
    resumed = run_da_plcbf_falsification("smoke", root, root_seed=31, repository=source_repository)
    assert resumed == first
    assert len(calls) == attempts

    alternate_repository = tmp_path / "different-source"
    (alternate_repository / "crazyflow").mkdir(parents=True)
    (alternate_repository / "pyproject.toml").write_text("[project]\nname='different'\n")
    with pytest.raises(ValueError, match="current source tree differs"):
        verify_da_plcbf_falsification(
            root,
            repository=alternate_repository,
            require_current_source=True,
            require_complete=True,
        )

    # Alter a scientific row and then deliberately recompute both outer checksum documents. The
    # semantic verifier must still reject it by reconstructing calls from the immutable caches.
    orchestration_path = root / "orchestration.json"
    orchestration = _read_object(orchestration_path)
    orchestration["evaluations"][0]["margin"] += 0.25
    orchestration_path.write_bytes(_canonical_json(orchestration) + b"\n")
    configuration = _read_object(root / "configuration.json")
    result = load_falsification_result(root / "falsification_result.json")
    previous_manifest = _read_object(root / "manifest.json")
    manifest = _manifest_document(
        root, configuration, result_sha256=result.sha256, summary=previous_manifest["summary"]
    )
    (root / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
    marker = _complete_marker_document(manifest, root / "manifest.json")
    (root / "complete.marker").write_bytes(_canonical_json(marker) + b"\n")
    with pytest.raises(ValueError, match="orchestration does not recompute"):
        verify_da_plcbf_falsification(root, require_current_source=False, require_complete=True)


def test_descriptive_replay_retains_per_method_operational_failures(tmp_path: Path) -> None:
    profile = falsification_profile("smoke", root_seed=11)
    candidates = generate_falsification_candidates(profile, root_seed=11)
    calls: list[str] = []
    successful_runner = _fake_runner(calls)

    def selective_runner(assignment: Any, *args: Any, **kwargs: Any) -> Any:
        if assignment.method == MethodID.NOMINAL_ONLY.value:
            raise RuntimeError("retained-replay-method-failure")
        return successful_runner(assignment, *args, **kwargs)

    evaluator = DAFalsificationEvaluator(
        profile,
        root_seed=11,
        candidate_set_sha256=boundary_candidate_set_sha256(candidates),
        cache_directory=tmp_path / "cache",
        trial_runner=selective_runner,
        strict_physical_validation=False,
    )
    values = _nominal_values()
    evaluator(values)
    evidence = evaluator.calls[-1]
    summaries = run_descriptive_counterexample_replays(
        profile,
        evaluator,
        (
            {
                "rank": 1,
                "candidate_sha256": evidence.candidate_sha256,
                "values": list(evidence.values),
            },
        ),
        tmp_path / "replays",
        result_sha256="b" * 64,
    )

    assert summaries[0]["successful_methods"] == 6
    assert summaries[0]["operational_failures"] == 1
    manifest = json.loads(next((tmp_path / "replays").glob("rank-*/manifest.json")).read_text())
    nominal = manifest["methods"][0]
    assert nominal["status"] == "operational_failure"
    assert nominal["error_type"] == "RuntimeError"
    assert nominal["error_message"] == "retained-replay-method-failure"
    replay_root = next((tmp_path / "replays").glob("rank-*"))
    assert not (replay_root / "methods" / MethodID.NOMINAL_ONLY.value).exists()


def test_unique_counterexample_ranking_and_replay_registry_are_explicitly_descriptive() -> None:
    profile = falsification_profile("smoke", root_seed=10)
    candidates = generate_falsification_candidates(profile, root_seed=10)
    result = run_fixed_budget_falsification(
        candidates,
        lambda values: -float(np.sum(values**2) + 1.0),
        search_name="da-plcbf-smoke",
        evaluator_name="ranking-fixture",
        evaluator_sha256="f" * 64,
        config=profile.search,
    )
    calls = tuple(
        EvaluationCallEvidence(
            # Deliberately collapse equal numeric points to the same identity.
            hashlib.sha256(np.asarray(row.values).tobytes()).hexdigest(),
            row.values,
            "success",
            False,
            row.margin,
            (
                ConditionTrialEvidence(
                    "falsification_combined",
                    "a" * 64,
                    "b" * 64,
                    row.margin,
                    "c" * 64,
                    "d" * 64,
                    "e" * 64,
                    "f" * 64,
                ),
            ),
        )
        for row in result.evaluations
    )
    ranking = rank_unique_counterexamples(result, calls, limit=8)
    registry = seven_method_replay_registry()

    assert ranking
    assert len({item["candidate_sha256"] for item in ranking}) == len(ranking)
    assert [item["minimum_hard_margin"] for item in ranking] == sorted(
        item["minimum_hard_margin"] for item in ranking
    )
    assert len(registry) == 7
    assert sum(item["discovery_evaluator"] for item in registry) == 1
    assert all("no paired sample size" in item["claim_boundary"] for item in registry)
    assert "No observed sample" in FALSIFICATION_EXPERIMENT_CLAIM_BOUNDARY
