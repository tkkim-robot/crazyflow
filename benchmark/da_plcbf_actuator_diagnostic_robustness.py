"""Execute the sealed four-realization robustness check in separate processes.

Fresh builds require two previously absent cache directories. The two signed
initial-state perturbations reuse build one's cache, as the sealed protocol says.
Every process, cache inventory, log, return code, and retained result is recorded.
An unsuccessful invocation stops the sequence and is never silently rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/da_plcbf/actuator-diagnostics-20260906/v1"
PROTOCOL = ROOT / "artifacts/da_plcbf/actuator-diagnostic-20260906/v1/protocol-sealed-v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(directory: Path) -> dict:
    return {
        str(path.relative_to(directory)): {"bytes": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def run(output: Path, cache_parent: Path) -> dict:
    """Run exactly the eight sealed flight identities with auditable process isolation."""
    if output.exists() or cache_parent.exists():
        raise FileExistsError("robustness output and fresh cache parent must both be absent")
    output.mkdir(parents=True)
    cache_parent.mkdir(parents=True)
    caches = {name: cache_parent / name for name in ("fresh_build_1", "fresh_build_2")}
    for cache in caches.values():
        cache.mkdir()
    rows = []
    report = {
        "status": "running",
        "protocol": str(PROTOCOL),
        "protocol_file_sha256": sha(PROTOCOL),
        "launcher_sha256": sha(Path(__file__)),
        "cache_parent": str(cache_parent.resolve()),
        "fresh_caches_absent_before_launch": True,
        "invocations": rows,
    }
    manifest = output / "launcher.json"
    for realization in ("fresh_build_1", "fresh_build_2", "perturb_plus", "perturb_minus"):
        cache = caches.get(realization, caches["fresh_build_1"])
        campaign = output / realization
        log = output / f"{realization}.log"
        environment = os.environ.copy()
        environment.update(
            PYTHONPATH=str(ROOT),
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            JAX_COMPILATION_CACHE_DIR=str(cache.resolve()),
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        command = [
            sys.executable,
            str(ROOT / "benchmark/da_plcbf_actuator_diagnostics.py"),
            "run",
            "--protocol",
            str(PROTOCOL),
            "--stage",
            "numerical_robustness",
            "--realization",
            realization,
            "--output",
            str(campaign),
        ]
        row = {
            "realization": realization,
            "command": command,
            "cache": str(cache.resolve()),
            "cache_before": inventory(cache),
            "started_unix_seconds": time.time(),
            "status": "running",
        }
        if realization.startswith("fresh_build") and row["cache_before"]:
            raise ValueError("fresh-build cache unexpectedly contains files")
        rows.append(row)
        with log.open("x") as stream:
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT
            )
            row["pid"] = process.pid
            manifest.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"starting": realization, "pid": process.pid}), flush=True)
            row["returncode"] = process.wait()
        row["finished_unix_seconds"] = time.time()
        row["cache_after"] = inventory(cache)
        row["log_sha256"] = sha(log)
        result = campaign / "campaign_result.json"
        if result.is_file():
            summary = json.loads(result.read_text())
            row["result_sha256"] = sha(result)
            row["planned"] = summary["planned"]
            row["completed"] = summary["completed"]
        passed = row["returncode"] == 0 and row.get("completed") == row.get("planned") == 2
        row["status"] = "completed" if passed else "failed_retained"
        report["status"] = "running" if passed else "failed_retained"
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"finished": realization, "status": row["status"]}), flush=True)
        if not passed:
            raise RuntimeError(f"retained unsuccessful realization: {realization}")
    report["status"] = "completed"
    report["completed_flight_count"] = sum(row["completed"] for row in rows)
    report["distinct_process_count"] = len({row["pid"] for row in rows})
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-parent", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve(), args.cache_parent.resolve()), indent=2))


if __name__ == "__main__":
    main()
