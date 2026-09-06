"""Collect the completed study without rerunning numerical experiments.

Run from the repository root with the full committed reporting-source revision as
the sole argument. The output directory must not already exist. Collection checks
and verifies the archive; MP4 files and most rollout arrays remain local-only.
"""

from pathlib import Path
import subprocess
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: collect-publication-v2.py REPORTING_REVISION_40_HEX")
    root = Path(__file__).resolve().parents[4]
    base = Path(__file__).resolve().parent
    args = [
        sys.executable,
        "-m",
        "benchmark.da_plcbf_actuator_publication",
        "collect",
        "--study-root",
        str(base),
        "--main-campaign",
        str(base / "main384-sealed-results-v1"),
        "--main-analysis",
        str(base / "main384-sealed-analysis-v1/analysis.json"),
        "--figure-dir",
        str(base / "publication-figures-v2"),
        "--video-dir",
        str(base / "candidate-1002-video-v2"),
        "--video-poster",
        "frame-000150.png",
        "--companion-video",
        str(base / "candidate-1002-paced-video-v2"),
        "frame-000138.png",
        "--report",
        str(root / "docs/da_plcbf_actuator_report.md"),
        "--technical-note",
        str(root / "docs/da_plcbf_actuator_theory.md"),
        "--reporting-revision",
        sys.argv[1],
        "--output",
        str(base / "publication-v2"),
    ]
    for name in ("paced", "delayed", "p1", "p2", "motor-noise"):
        args.extend(["--promoted-case-campaign", str(base / f"candidate-1002-{name}-v1")])
    subprocess.run(args, cwd=root, check=True)


if __name__ == "__main__":
    main()
