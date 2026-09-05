from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from examples.da_plcbf.contact_replay_demo import _nearest_indices, load_presentation

if TYPE_CHECKING:
    from pathlib import Path


def _contact_fixture(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    source = directory / "source"
    source.mkdir()
    scenario = {
        "dt": 0.02,
        "obstacle_mask": [True],
        "obstacle_initial_centers": [[1.0, 2.0, 3.0]],
        "obstacle_velocities": [[0.5, 0.0, -0.25]],
    }
    (source / "feasibility_reference.json").write_text(json.dumps({"scenario": scenario}))
    replay = {
        "source_directory": str(source),
        "pre_failure_closed_loop_branches": {"test": {"start_time": 0.0}},
    }
    (source / "replay.json").write_text(json.dumps(replay))
    states = np.zeros((5, 13))
    states[:, 0] = np.arange(5) * 0.2
    states[:, 2] = 1.5
    states[:, 3:7] = np.asarray([0.0, 0.0, 0.6, 0.8])
    np.savez(source / "replay.npz", branch_test_states=states)
    # The handoff is deliberately between source nodes, and contact attitude differs.
    times = 0.055 + np.arange(9) * 0.005
    contacts = np.zeros((len(times), 13))
    contacts[:, 0] = 10 + np.arange(len(times))
    contacts[:, 2] = 1.0 - np.arange(len(times)) * 0.1
    contacts[:, 3:7] = np.asarray([0.6, 0.0, 0.0, 0.8])
    np.savez(
        directory / "contact_replay.npz",
        time_seconds=times,
        full_state=contacts,
        obstacle_centers=np.tile([[1.0, 2.0, 3.0]], (len(times), 1, 1)),
    )
    (directory / "contact_model.xml").write_text(
        '<mujoco><worldbody><geom name="obstacle_geom_0" size="0.2"/></worldbody></mujoco>'
    )
    metadata = {
        "trigger": {"time_seconds": 0.055},
        "source": {
            "replay_directory": str(source),
            "branch": "test",
            "input_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source.iterdir()
            },
        },
        "npz_sha256": hashlib.sha256((directory / "contact_replay.npz").read_bytes()).hexdigest(),
        "xml_sha256": hashlib.sha256((directory / "contact_model.xml").read_bytes()).hexdigest(),
    }
    (directory / "contact_replay.json").write_text(json.dumps(metadata))
    return states, contacts


def test_contact_splice_preserves_saved_pose_and_quaternion_on_each_side(tmp_path: Path) -> None:
    states, contacts = _contact_fixture(tmp_path)
    result = load_presentation(tmp_path, prelude_seconds=0.04, fps=100, playback_rate=0.5)
    before = ~result.contact_phase
    assert np.all(result.times[before] < 0.055)
    assert np.all(result.times[~before] >= 0.055)
    np.testing.assert_array_equal(result.states[before], states[result.source_indices[before]])
    np.testing.assert_array_equal(result.states[~before], contacts[result.source_indices[~before]])
    first_contact = np.flatnonzero(result.contact_phase)[0]
    assert result.source_indices[first_contact] == 0
    assert result.times[first_contact] == 0.055
    # High frame rates hold the last recorded pre-handoff node without showing source's future.
    assert result.times[first_contact - 1] == 0.04
    np.testing.assert_array_equal(result.obstacle_radii, [0.2])
    np.testing.assert_allclose(
        result.obstacle_centers[before, 0],
        np.asarray([1.0, 2.0, 3.0]) + result.times[before, None] * np.asarray([0.5, 0.0, -0.25]),
    )


def test_contact_splice_rejects_changed_source_or_contact_inputs(tmp_path: Path) -> None:
    _contact_fixture(tmp_path)
    source = tmp_path / "source/replay.json"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="approach input checksum mismatch"):
        load_presentation(tmp_path)
    source.write_text(source.read_text().removesuffix("\n"))
    xml = tmp_path / "contact_model.xml"
    xml.write_text(xml.read_text().replace('size="0.2"', 'size="0.4"'))
    with pytest.raises(ValueError, match="contact input checksum mismatch"):
        load_presentation(tmp_path)


def test_saved_sample_selection_never_extrapolates() -> None:
    times = np.asarray([1.0, 1.02, 1.04])
    np.testing.assert_array_equal(
        _nearest_indices(times, np.asarray([1.0, 1.011, 1.04])), [0, 1, 2]
    )
    with pytest.raises(ValueError, match="outside"):
        _nearest_indices(times, np.asarray([0.99]))
    with pytest.raises(ValueError, match="outside"):
        _nearest_indices(times, np.asarray([1.05]))
