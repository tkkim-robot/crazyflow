"""Render three saved samples per selected numerical comparison; no simulation is rerun."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from crazyflow.safety.da_plcbf.online_constant_wind import load_online_constant_wind_result
from crazyflow.safety.da_plcbf import mujoco_comparison_video as renderer
base=Path('artifacts/da_plcbf/navigation-revision-20260905')
for source,sample_times in ((base/'heldout-shard-a/combined-seed100',(6.6,16.4,27.0)),(base/'development-dense16-oracle-seed0',(6.6,7.12,16.4))):
    trace=load_online_constant_wind_result(source/'navigation_comparison.npz',source/'navigation_comparison.json').trace
    indices=np.asarray([np.argmin(abs(trace.time_seconds-time)) for time in sample_times])
    original=renderer._frame_indices
    try:
        renderer._frame_indices=lambda time,fps:indices
        frames=renderer.comparison_video_frames(trace,renderer.ComparisonRenderConfig(mode='demo'))
        for index,frame in zip(indices,frames,strict=True):
            Image.fromarray(frame).save(source/f'review_frame_{trace.time_seconds[index]:.2f}s.png')
    finally:
        renderer._frame_indices=original
    (source/'visual_review_metadata.json').write_text(json.dumps({'source_trace':'navigation_comparison.npz','sample_times_seconds':trace.time_seconds[indices].tolist(),'frame_rendering':'Same scene renderer/configuration and actual sampled state/world/model data used by video; no controller or learner replay.'},indent=2)+'\n')
