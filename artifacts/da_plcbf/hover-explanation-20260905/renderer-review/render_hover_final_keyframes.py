"""Keyframe visual review from the saved main hover-first mechanism trace."""
from pathlib import Path
from dataclasses import asdict
import json
import hashlib
import numpy as np
from PIL import Image
from crazyflow.safety.da_plcbf.online_constant_wind import load_online_constant_wind_result
from crazyflow.safety.da_plcbf import mujoco_comparison_video as renderer
root=Path('/home/tk/Desktop/mycode/crazyflow/artifacts/da_plcbf/hover-explanation-20260905')
source=root/'hover-wind-payload-navigation'
output=root/'renderer-review/hover-main-keyframes-final';output.mkdir(exist_ok=False)
trace=load_online_constant_wind_result(source/'navigation_comparison.npz',source/'navigation_comparison.json').trace
config=renderer.ComparisonRenderConfig(mode='demo',hover_camera_distance=2.8,camera_distance=3.8,comparison_note='Shared wind-aware hover control · both fallback maps start without wind correction')
indices=np.asarray([np.argmin(abs(trace.time_seconds-t)) for t in (2.8,3.2,10.8,11.2,18.8,19.4,23.,25.,35.)])
original=renderer._frame_indices
try:
 renderer._frame_indices=lambda time,fps:indices
 for index,frame in zip(indices,renderer.comparison_video_frames(trace,config),strict=True):
  Image.fromarray(frame).save(output/f'review_{trace.time_seconds[index]:.2f}s.png')
finally:renderer._frame_indices=original
(output/'metadata.json').write_text(json.dumps({'source':str(source),'source_npz_sha256':hashlib.sha256((source/'navigation_comparison.npz').read_bytes()).hexdigest(),'source_json_sha256':hashlib.sha256((source/'navigation_comparison.json').read_bytes()).hexdigest(),'render_config':asdict(config),'sample_times_seconds':trace.time_seconds[indices].tolist(),'scope':'Actual recorded states and predictions; equal metric camera distances, with2s smooth widening at recorded navigation release. No controller/learner replay or invented trajectory separation.'},indent=2)+'\n')
print(output)
