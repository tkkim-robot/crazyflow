"""The robustness launcher must isolate builds and preserve failed executions."""

import json
from pathlib import Path
from typing import TextIO

import pytest

from benchmark import da_plcbf_actuator_diagnostic_robustness as launcher


def fake_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, returncode: int = 0
) -> list[tuple]:
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n")
    monkeypatch.setattr(launcher, "PROTOCOL", protocol)
    calls = []

    class Process:
        def __init__(
            self, command: list[str], *, cwd: Path, env: dict[str, str], stdout: TextIO, stderr: int
        ) -> None:
            cache = Path(env["JAX_COMPILATION_CACHE_DIR"])
            realization = command[command.index("--realization") + 1]
            calls.append((realization, cache, list(cache.iterdir()), command, cwd))
            (cache / "compiled-entry").write_text(realization)
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            (output / "campaign_result.json").write_text(json.dumps({"planned": 2, "completed": 2}))
            stdout.write("retained process output\n")
            self.pid = 1200 + len(calls)

        def wait(self) -> int:
            return returncode

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    return calls


def test_builds_are_fresh_and_perturbations_reuse_only_first_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = fake_process(monkeypatch, tmp_path)
    report = launcher.run(tmp_path / "results", tmp_path / "caches")
    assert report["completed_flight_count"] == 8
    assert report["distinct_process_count"] == 4
    assert [row[0] for row in calls] == [
        "fresh_build_1",
        "fresh_build_2",
        "perturb_plus",
        "perturb_minus",
    ]
    assert calls[0][1] != calls[1][1]
    assert calls[0][2] == calls[1][2] == []
    assert calls[2][1] == calls[3][1] == calls[0][1]
    assert calls[2][2] and calls[3][2]
    assert all(row[3].count("--realization") == 1 for row in calls)


@pytest.mark.parametrize("preexisting", ["results", "caches"])
def test_preexisting_evidence_or_cache_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preexisting: str
) -> None:
    calls = fake_process(monkeypatch, tmp_path)
    (tmp_path / preexisting).mkdir()
    with pytest.raises(FileExistsError):
        launcher.run(tmp_path / "results", tmp_path / "caches")
    assert calls == []


def test_failed_process_is_retained_and_stops_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = fake_process(monkeypatch, tmp_path, returncode=1)
    with pytest.raises(RuntimeError, match="fresh_build_1"):
        launcher.run(tmp_path / "results", tmp_path / "caches")
    assert len(calls) == 1
    report = json.loads((tmp_path / "results" / "launcher.json").read_text())
    assert report["status"] == "failed_retained"
    assert report["invocations"][0]["returncode"] == 1
    assert (tmp_path / "results" / "fresh_build_1.log").read_text() == "retained process output\n"
