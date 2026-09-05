"""Execute the frozen additional diagnostics after the first full confirmation."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax

from benchmark.da_plcbf_closed_loop_confirmation import parameter_reversion_case
from benchmark.da_plcbf_closed_loop_search import EpisodeEvaluator, archive_sources, write_json
from crazyflow.safety.da_plcbf.case_study_world import HoverEncounterConfig

output = Path(__file__).resolve().parent
root = output.parent
protocol = json.loads((output / "protocol.json").read_text())
source_id, hashes = archive_sources(output)
for name in ("benchmark/da_plcbf_closed_loop_confirmation.py",):
    data = Path(name).read_bytes()
    target = output / "source_versions" / source_id / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    hashes[name] = hashlib.sha256(data).hexdigest()
write_json(output / "EXECUTION_SOURCES.json", hashes)
scene = HoverEncounterConfig.from_dict(protocol["source_scene"])
engine = EpisodeEvaluator(jax.devices("gpu")[0])
results = {}
for mapping in protocol["mappings"]:
    source = json.loads((root / "confirmation-staggered-0000-v1" / mapping / "original/result.json").read_text())
    result = parameter_reversion_case(
        engine, scene, mapping, source["methods"]["fixed"], output / mapping / "early_parameter_reversion",
        boundary_seconds=protocol["early_parameter_reversion_time_seconds"],
        boundary_reason=protocol["boundary_reason"],
    )
    results[f"{mapping}/early_parameter_reversion"] = result
    print(json.dumps({"variant": f"{mapping}/early_parameter_reversion", "outcome": result["outcome_class"]}), flush=True)
narrow = json.loads((root / protocol["narrow_neighbor"]).read_text())
fine = EpisodeEvaluator(jax.devices("gpu")[0], plant_substeps=protocol["narrow_refined_plant_substeps"])
fine.bundles = engine.bundles
result = fine.evaluate(HoverEncounterConfig.from_dict(narrow["scene"]), "uncompensated", output / "narrow_neighbor_fine")
results["narrow_neighbor_fine"] = result
print(json.dumps({"variant": "narrow_neighbor_fine", "outcome": result["outcome_class"]}), flush=True)
for buffer in protocol["buffers_m"]:
    for mapping in protocol["mappings"]:
        name = f"{mapping}/clearance_{buffer:.2f}"
        result = engine.evaluate(replace(scene, obstacle_clearance=buffer), mapping, output / name)
        results[name] = result
        print(json.dumps({"variant": name, "outcome": result["outcome_class"]}), flush=True)
write_json(output / "SUMMARY.json", results)
