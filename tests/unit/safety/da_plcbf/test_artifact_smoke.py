import json
from pathlib import Path

import pytest

from crazyflow.safety.da_plcbf.artifact_smoke import create_synthetic_smoke_run
from crazyflow.safety.da_plcbf.artifacts import validate_run_artifacts
from crazyflow.safety.da_plcbf.scenarios import load_scenario_tape


@pytest.mark.render
def test_complete_synthetic_smoke_has_strict_layout_hashes_metrics_and_replay(
    tmp_path: Path,
) -> None:
    run, result = create_synthetic_smoke_run(tmp_path, run_id="artifact-smoke-test")
    assert result["status"] == "synthetic-smoke"
    assert result["scientific_evidence"] is False
    assert result["method_runs"] == 1
    assert result["videos"] == 1
    assert (run / "manifest.json").is_file()
    assert (run / "SHA256SUMS").is_file()
    assert (run / "videos" / "synthetic-dashboard.mp4").is_file()
    assert (run / "scenario_tapes" / "0.npz").is_file()
    manifest = json.loads((run / "manifest.json").read_text())
    seeds = json.loads((run / "seeds.json").read_text())
    assert manifest["scenario_tapes"][0]["path"] == "scenario_tapes/0.npz"
    assert (
        manifest["scenario_tapes"][0]["content_sha256"]
        == seeds["scenario_tapes"][0]["content_sha256"]
    )
    tape = load_scenario_tape(run / "scenario_tapes" / "0.npz")
    assert int(tape.schema_version) == 3
    assert tape.ballistic_generation_attempts[0] >= 1
    assert tape.ballistic_realized_closest_distance[0] > 0.0
    assert validate_run_artifacts(run, verify_replay=True) == result


def test_repository_artifact_policy_keeps_only_compact_index_files() -> None:
    repository = Path(__file__).resolve().parents[4]
    ignore = (repository / ".gitignore").read_text()
    policy = (repository / "artifacts" / "da_plcbf" / "README.md").read_text()
    index = (repository / "artifacts" / "da_plcbf" / "INDEX.md").read_text()
    assert "/artifacts/da_plcbf/**" in ignore
    assert "!/artifacts/da_plcbf/README.md" in ignore
    assert "!/artifacts/da_plcbf/INDEX.md" in ignore
    assert "content-addressed" in policy
    assert "No scientific run" in index
