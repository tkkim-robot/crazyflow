"""80-digit reference projections for the first six frozen-library QP divergences.

These points have box constraints plus one policy row, with no SQP refinement.
Solve the original physical row and its recorded inward tightening independently
of JAX's active-set solver. Exact Decimal conversion preserves each input float.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts/da_plcbf/numerical-stabilization-20260906/v1/development-36-v3"


def number(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def reference(
    nominal: np.ndarray,
    row: np.ndarray,
    bound: float,
    lower: np.ndarray,
    upper: np.ndarray,
    margins: np.ndarray,
) -> dict:
    """Project onto a box intersected with one halfspace using monotone dual search."""
    with localcontext() as context:
        context.prec = 80
        lo, hi, u, a, m = [
            [number(v) for v in values] for values in (lower, upper, nominal, row, margins)
        ]
        span = [h - low for low, h in zip(lo, hi, strict=True)]
        z0 = [(v - low) / d for v, low, d in zip(u, lo, span, strict=True)]
        normalized_row = [v * d for v, d in zip(a, span, strict=True)]
        normalized_bound = (
            number(bound) - sum(v * low for v, low in zip(a, lo, strict=True)) - m[-1]
        )
        box_lo, box_hi = m[4:8], [Decimal(1) - v for v in m[:4]]

        def point(dual: Decimal) -> list[Decimal]:
            return [
                min(h, max(low, z - dual * r))
                for z, r, low, h in zip(z0, normalized_row, box_lo, box_hi, strict=True)
            ]

        def slack(z: list[Decimal]) -> Decimal:
            return normalized_bound - sum(r * v for r, v in zip(normalized_row, z, strict=True))

        minimum = sum(
            r * (low if r >= 0 else h)
            for r, low, h in zip(normalized_row, box_lo, box_hi, strict=True)
        )
        feasible = (
            all(low <= h for low, h in zip(box_lo, box_hi, strict=True))
            and minimum <= normalized_bound
        )
        if not feasible:
            return {
                "feasible": False,
                "minimum_row_lhs": str(minimum),
                "bound": str(normalized_bound),
            }
        dual = Decimal(0)
        if slack(point(dual)) < 0:
            left, right = Decimal(0), Decimal(1)
            while slack(point(right)) < 0:
                right *= 2
            for _ in range(300):
                mid = (left + right) / 2
                if slack(point(mid)) < 0:
                    left = mid
                else:
                    right = mid
            dual = right
        z = point(dual)
        command = [low + d * v for low, d, v in zip(lo, span, z, strict=True)]
        raw_slack = number(bound) - sum(r * v for r, v in zip(a, command, strict=True))
        stationarity = [
            v - original + dual * r for v, original, r in zip(z, z0, normalized_row, strict=True)
        ]
        kkt = max(
            abs(g) if low < v < h else max(Decimal(0), -g) if v == low else max(Decimal(0), g)
            for v, low, h, g in zip(z, box_lo, box_hi, stationarity, strict=True)
        )
        assert slack(z) >= Decimal("-1e-70") and kkt < Decimal("1e-65")
        return {
            "feasible": True,
            "command": [float(v) for v in command],
            "command_80digit": [str(v) for v in command],
            "dual": str(dual),
            "tightened_slack": str(slack(z)),
            "original_raw_slack": str(raw_slack),
            "stationarity_residual": str(kkt),
            "objective": str(sum((v - q) ** 2 for v, q in zip(z, z0, strict=True)) / 2),
        }


def main() -> None:
    from crazyflow.safety.da_plcbf.actuator_study import nominal_actuator_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    model = nominal_actuator_model()
    lower, upper = np.asarray(model.command_lower), np.asarray(model.command_upper)
    records, sources = [], {}
    for case in ("harm", "gain", "no_change_regression"):
        for method in ("PD_F", "F2"):
            data = []
            for mode in ("legacy", "inward"):
                path = OLD / f"{case}__{method}__matched__{mode}/attempt-00/controls.npz"
                sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(path) as arrays:
                    data.append(
                        {
                            k: arrays[k]
                            for k in arrays.files
                            if k not in {"candidate_states", "candidate_commands"}
                        }
                    )
            old, new = data
            n = min(len(old["time"]), len(new["time"]))
            index = int(
                np.flatnonzero(
                    np.any(old["planned_command"][:n] != new["planned_command"][:n], axis=1)
                )[0]
            )
            shared = {
                k: np.array_equal(old[k][index], new[k][index])
                for k in (
                    "controller_input_state",
                    "estimated_model_sha256",
                    "nominal_command",
                    "selected_index",
                    "selected_row",
                    "selected_bound",
                    "previous_command",
                    "goal",
                )
            }
            assert all(shared.values())
            assert old["sqp_iterations"][index] == new["sqp_iterations"][index] == 0
            comparisons = {}
            for mode, values in zip(("original", "tightened"), data, strict=True):
                margins = values["qp_numerics_inward_margins"][index]
                result = reference(
                    values["nominal_command"][index],
                    values["selected_row"][index],
                    values["selected_bound"][index],
                    lower,
                    upper,
                    margins,
                )
                result["saved_qp_command"] = values["qp_proposed_command"][index].tolist()
                result["saved_executed_command"] = values["planned_command"][index].tolist()
                result["saved_rejection_flags"] = values["qp_rejection_flags"][index].tolist()
                result["saved_row_scales"] = values["qp_numerics_row_scales"][index].tolist()
                result["saved_inward_margins"] = margins.tolist()
                result["max_saved_vs_reference_command_delta"] = (
                    float(np.max(np.abs(values["qp_proposed_command"][index] - result["command"])))
                    if result["feasible"]
                    else None
                )
                comparisons[mode] = result
            records.append(
                {
                    "case": case,
                    "method": method,
                    "time": float(old["time"][index]),
                    "shared_inputs": shared,
                    "references": comparisons,
                }
            )
    report = {
        "rows": records,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": sources,
        "qualification": (
            "80-digit solution of the physical QP from recorded float32 row/command inputs; "
            "tightened reference applies recorded normalized-coordinate margins. Float32 affine "
            "coordinate-conversion rounding is not declared exact physical arithmetic. "
            "All six selected points have no operational SQP rows."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "pairs": len(records),
                "all_reference_qps_feasible": all(
                    r["references"][m]["feasible"]
                    for r in records
                    for m in ("original", "tightened")
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
