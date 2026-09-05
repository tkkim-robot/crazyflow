from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import pytest

from crazyflow.safety.da_plcbf import competent_library_experiment as experiment
from crazyflow.safety.da_plcbf.competent_library_experiment import (
    CompetentExperimentConfig,
    _load_prefix,
    _prefix_provenance,
    _scenario,
    prepare_competent_checkpoint,
    run_competent_experiment,
    validate_checkpoint_compatibility,
)
from crazyflow.safety.da_plcbf.learner_checkpoint import load_learner_checkpoint
from crazyflow.safety.da_plcbf.online_constant_wind import (
    comparison_trace_for_methods,
    load_online_constant_wind_result,
)

if TYPE_CHECKING:
    from pathlib import Path


def _metadata(config: CompetentExperimentConfig) -> dict[str, Any]:
    return {
        "experiment_config": asdict(config),
        "prefix_provenance": _prefix_provenance(config, _scenario(config)),
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"control_interval_steps": 1},
        {"policy_count": 8},
        {"horizon": 12},
        {"nominal_acceleration_limit": 0.8},
        {"maximum_skill_speed": 0.7},
        {"learning_rate": 0.002},
        {"disturbance": "crossing"},
    ],
)
def test_checkpoint_rejects_changed_prefix_controller_geometry_or_skill_spec(
    overrides: dict[str, Any],
) -> None:
    original = CompetentExperimentConfig()
    with jax.default_device(jax.devices("cpu")[0]):
        with pytest.raises(ValueError, match="checkpoint"):
            validate_checkpoint_compatibility(replace(original, **overrides), _metadata(original))


@pytest.mark.parametrize(
    "overrides",
    [
        {"wind_after": (0.5, -0.2, 0.0), "model_mode": "estimated"},
        {"disturbance": "unchanged", "duration_seconds": 24.0},
        {"disturbance": "payload", "payload_mass_fraction": 0.1},
        {"schedule": "unlimited", "probe_every_controls": 5},
        {"adaptive_model_compensation": False},
    ],
)
def test_checkpoint_allows_changes_that_only_affect_post_event_execution(
    overrides: dict[str, Any],
) -> None:
    original = CompetentExperimentConfig()
    with jax.default_device(jax.devices("cpu")[0]):
        validate_checkpoint_compatibility(replace(original, **overrides), _metadata(original))


def test_checkpoint_rejects_legacy_metadata_without_recorded_effective_prefix() -> None:
    config = CompetentExperimentConfig()
    with pytest.raises(ValueError, match="regenerate"):
        validate_checkpoint_compatibility(config, {"experiment_config": asdict(config)})


def test_checkpoint_detects_geometry_default_changed_since_it_was_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = CompetentExperimentConfig()
    with jax.default_device(jax.devices("cpu")[0]):
        recorded = _metadata(config)

        def changed_default(requested: CompetentExperimentConfig) -> Any:
            scenario = _scenario(requested)
            return replace(scenario, obstacle_clearance=scenario.obstacle_clearance + 0.01)

        monkeypatch.setattr(experiment, "_scenario", changed_default)
        with pytest.raises(ValueError, match="prefix scenario differs"):
            validate_checkpoint_compatibility(config, recorded)


@pytest.fixture(scope="module")
def tiny_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[CompetentExperimentConfig, Path, jax.Device]:
    config = CompetentExperimentConfig(
        policy_count=4,
        horizon=10,
        warmup_steps=2,
        control_interval_steps=2,
        event_time_seconds=0.04,
        duration_seconds=0.16,
        wind_after=(0.5, 0.2, 0.0),
        schedule="unlimited",
        probe_every_controls=1,
    )
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        stem = prepare_competent_checkpoint(
            config, tmp_path_factory.mktemp("competent") / "checkpoint", cpu
        )
    return config, stem, cpu


def test_prefix_checksum_and_final_state_must_match_checkpoint(
    tiny_checkpoint: tuple[CompetentExperimentConfig, Path, jax.Device], tmp_path: Path
) -> None:
    _, stem, cpu = tiny_checkpoint
    with jax.default_device(cpu):
        checkpoint = load_learner_checkpoint(stem, device=cpu)
    prefix_bytes = (stem.parent / "shared_prefix.npz").read_bytes()
    copied_stem = tmp_path / "checkpoint"
    copied_prefix = tmp_path / "shared_prefix.npz"
    copied_prefix.write_bytes(prefix_bytes[:-1] + bytes([prefix_bytes[-1] ^ 1]))
    with pytest.raises(ValueError, match="checksum mismatch"):
        _load_prefix(
            copied_stem,
            metadata=checkpoint.metadata,
            expected_final_state=checkpoint.physical_state,
        )
    copied_prefix.write_bytes(prefix_bytes)
    wrong_state = np.asarray(checkpoint.physical_state).copy()
    wrong_state[0] += 0.01
    with pytest.raises(ValueError, match="final physical state differs"):
        _load_prefix(copied_stem, metadata=checkpoint.metadata, expected_final_state=wrong_state)


@pytest.mark.parametrize("model_mode", ["oracle", "estimated"])
def test_tiny_matched_experiment_preserves_snapshots_estimators_and_saved_probes(
    tiny_checkpoint: tuple[CompetentExperimentConfig, Path, jax.Device],
    tmp_path: Path,
    model_mode: str,
) -> None:
    original, checkpoint_stem, cpu = tiny_checkpoint
    config = replace(original, model_mode=model_mode)
    directory = tmp_path / model_mode
    with jax.default_device(cpu):
        result = run_competent_experiment(
            config, directory, checkpoint_stem=checkpoint_stem, device=cpu
        )
        checkpoint = load_learner_checkpoint(checkpoint_stem, device=cpu)
        final = load_learner_checkpoint(directory / "final_adaptive_checkpoint", device=cpu)
    summary = result.summary
    assert summary["same_event_state_and_parameters"] is True
    assert summary["checkpoint_npz_sha256"] == checkpoint.sha256
    assert summary["all_checks_passed"] is False  # Short failures must remain saveable.
    names = ("fixed", "compensated", "adaptive")
    event_index = 1
    initial_version = int(checkpoint.state.library_version)
    assert initial_version == config.warmup_steps
    np.testing.assert_allclose(result.trace.time_seconds, [0.0, 0.04, 0.08, 0.12])
    for name in names:
        method = result.methods[name]
        np.testing.assert_array_equal(
            method.full_state[: event_index + 1],
            result.methods["fixed"].full_state[: event_index + 1],
        )
        np.testing.assert_array_equal(method.full_state[event_index], checkpoint.physical_state)
        np.testing.assert_array_equal(method.estimated_wind[:event_index], 0.0)
        if model_mode == "oracle":
            expected = np.broadcast_to(config.wind_after, (3, 3))
        else:
            # Every branch starts its own zero estimate, then uses both held-action transitions.
            elapsed = result.trace.time_seconds[event_index:] - config.event_time_seconds
            expected = (1.0 - np.exp(-2.4 * elapsed[:, None])) * np.asarray(config.wind_after)
        np.testing.assert_allclose(
            method.estimated_wind[event_index:], expected, rtol=0.0, atol=2e-5
        )
        if name != "adaptive":
            np.testing.assert_array_equal(method.library_version, initial_version)
            np.testing.assert_array_equal(method.parameter_update_norm, 0.0)

    adaptive = result.methods["adaptive"]
    np.testing.assert_array_equal(
        adaptive.library_version,
        [initial_version, initial_version, initial_version + 1, initial_version + 2],
    )
    np.testing.assert_array_equal(adaptive.gradient_norm[: event_index + 1], 0.0)
    assert np.all(adaptive.gradient_norm[event_index + 1 :] > 0.0)
    assert np.all(adaptive.parameter_update_norm[event_index + 1 :] > 0.0)
    timing = summary["methods"]["adaptive"]
    assert timing["attempted_updates"] == timing["finite_updates"] == 3
    assert (
        timing["final_library_version"] == int(final.state.library_version) == initial_version + 3
    )
    assert timing["last_controller_used_library_version"] == initial_version + 2
    for publication in timing["snapshot_publications"]:
        assert publication["completed_wall_time"] <= publication["published_wall_time"]
        assert publication["published_simulation_time"] == pytest.approx(
            publication["training_simulation_time"] + config.control_period
        )

    loaded = load_online_constant_wind_result(
        directory / "competent_comparison.npz", directory / "competent_comparison.json"
    )
    assert loaded.summary == json.loads(json.dumps(summary))
    for name in names:
        for field in ("full_state", "estimated_wind", "gradient_norm", "library_version"):
            np.testing.assert_array_equal(
                getattr(loaded.methods[name], field), getattr(result.methods[name], field)
            )
    pair = comparison_trace_for_methods(loaded, "compensated", "adaptive")
    assert pair.coverage_probes is not None
    assert pair.repertoire_probes is not None
    np.testing.assert_array_equal(
        pair.repertoire_probes["left_rollouts"],
        result.trace.repertoire_probes["compensated_rollouts"],
    )
    probes = loaded.summary["shared_probes"]
    assert len(probes) == len(result.trace.time_seconds)
    with np.load(directory / "symmetric_probe_trajectories.npz", allow_pickle=False) as archive:
        for index, probe in enumerate(probes):
            for anchor in names:
                reference = probe["anchors"][anchor]
                for candidate in names:
                    key = f"{anchor}__{candidate}"
                    trajectories = archive[f"{key}_states"][index]
                    # Nominal and every fallback start at the exact same measured anchor.
                    np.testing.assert_array_equal(
                        trajectories[:, 0],
                        np.broadcast_to(reference["full_state"], trajectories[:, 0].shape),
                    )
                    hard = archive[f"{key}_values"][index]
                    assert reference["libraries"][candidate]["maximum_library_value"] == hard.max()
        # Candidate zero is the shared nominal, independent of which fallback library is probed.
        np.testing.assert_array_equal(
            archive["adaptive__fixed_states"][:, 0], archive["adaptive__adaptive_states"][:, 0]
        )
