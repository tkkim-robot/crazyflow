"""Render one unchanged saved numerical sample to inspect the payload visibility revision."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from PIL import Image
from crazyflow.safety.da_plcbf.online_constant_wind import load_online_constant_wind_result
from crazyflow.safety.da_plcbf import mujoco_comparison_video as renderer
source = Path('artifacts/da_plcbf/competent-revision-20260904/payload-oracle-6')
output = Path(__file__).resolve().parent
result = load_online_constant_wind_result(source / 'competent_comparison.npz', source / 'competent_comparison.json')
trace = replace(result.trace, fixed=result.methods['compensated'], left_label='COMPENSATED FROZEN LIBRARY')
index = int(np.argmin(np.abs(trace.time_seconds - 4.4)))
original = renderer._frame_indices
try:
    renderer._frame_indices = lambda time, fps: np.asarray((index,))
    frames = renderer.comparison_video_frames(trace, renderer.ComparisonRenderConfig(mode='demo'))
    Image.fromarray(next(frames)).save(output / 'payload_detail_4p4s.png')
    frames.close()
finally:
    renderer._frame_indices = original
(output / 'metadata.json').write_text(json.dumps({'source': str(source), 'sample_index': index, 'recorded_time_seconds': float(trace.time_seconds[index]), 'changes': 'Outline and metric payload inset only; unchanged recorded positions, quaternions, full fallback paths, model and event metadata'}, indent=2) + '\n')
