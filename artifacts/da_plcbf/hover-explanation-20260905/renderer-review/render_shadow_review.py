"""Shadow-only before/after review of unchanged recorded numerical navigation frames."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
from crazyflow.safety.da_plcbf.online_constant_wind import load_online_constant_wind_result
from crazyflow.safety.da_plcbf import mujoco_comparison_video as renderer

root = Path('/home/tk/Desktop/mycode/crazyflow/artifacts/da_plcbf/hover-explanation-20260905/renderer-review')
source = Path('/home/tk/Desktop/mycode/crazyflow/artifacts/da_plcbf/navigation-revision-20260905/heldout-shard-a/combined-seed100')
trace = load_online_constant_wind_result(source / 'navigation_comparison.npz', source / 'navigation_comparison.json').trace
indices = np.asarray([np.argmin(abs(trace.time_seconds - time)) for time in (6.6, 16.4, 27.0)])
frame_indices, configure_shadows = renderer._frame_indices, renderer._configure_scene_shadows
outputs = []
try:
    renderer._frame_indices = lambda time, fps: indices
    for label, configure in [('before_shadow_fix', lambda *a: None), ('after_shadow_fix', configure_shadows)]:
        renderer._configure_scene_shadows = configure
        for index, frame in zip(indices, renderer.comparison_video_frames(trace, renderer.ComparisonRenderConfig(mode='demo')), strict=True):
            path = root / f'{label}_{trace.time_seconds[index]:.2f}s.png'
            if path.exists(): raise FileExistsError(path)
            Image.fromarray(frame).save(path)
            outputs.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
finally:
    renderer._frame_indices, renderer._configure_scene_shadows = frame_indices, configure_shadows
(root / 'shadow_review_metadata.json').write_text(json.dumps({'source': str(source), 'source_npz_sha256': hashlib.sha256((source / 'navigation_comparison.npz').read_bytes()).hexdigest(), 'source_json_sha256': hashlib.sha256((source / 'navigation_comparison.json').read_bytes()).hexdigest(), 'time_seconds': trace.time_seconds[indices].tolist(), 'scope': 'Same recorded poses, obstacle centers, predictions, camera, current labels and illumination. Before uses original default shadow volume; after only enables the new scene-bound shadow volume. No numerical replay or invented obstacle/payload change.', 'outputs': outputs}, indent=2) + '\n')
print(json.dumps(outputs, indent=2))
