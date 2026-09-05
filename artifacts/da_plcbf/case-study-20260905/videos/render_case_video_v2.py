"""Render one case per fresh EGL process and reject black scene panels before encoding."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import crazyflow.safety.da_plcbf.mujoco_comparison_video as renderer
from crazyflow.safety.da_plcbf.online_constant_wind import load_online_constant_wind_result

parser=argparse.ArgumentParser();parser.add_argument('case',choices=('controlled-encounter','paced-continuous'));args=parser.parse_args()
root=Path('artifacts/da_plcbf/case-study-20260905');directory=root/'videos'/args.case
if args.case=='controlled-encounter':
 source=root/'closed-loop-v3/uncompensated-000-t0100-132';stem='comparison'
 title='Persistent wind encounter | controlled same-state branch'
 note='Controlled branch at t=4s | adapted snapshot held fixed | shell violation is not collision'
else:
 source=root/'continuous-paced-v1';stem='navigation_comparison'
 title='Persistent wind through the encounter | actual online updates'
 note='Persistent wind | live finite updates | both methods finish safely'
paths=(source/f'{stem}.npz',source/f'{stem}.json')
hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
trace=replace(load_online_constant_wind_result(*paths).trace,title=title)
config=renderer.ComparisonRenderConfig(mode='demo',fps=20,width=1600,height=900,hover_camera_distance=2.8,camera_azimuth=-45,comparison_note=note)
original=renderer._render_world
checked=[]
def check(*arguments,**kwargs):
 frame=original(*arguments,**kwargs)
 mean=float(frame.mean())
 index=int(arguments[3]);world=int(arguments[4]);sim=arguments[0]
 checked.append({'index':index,'world':world,'scene_mean_rgb':mean})
 if mean<1:
  details={'index':index,'world':world,'camera':{'lookat':sim.viewer.viewer.cam.lookat.tolist(),'distance':sim.viewer.viewer.cam.distance,'azimuth':sim.viewer.viewer.cam.azimuth,'elevation':sim.viewer.viewer.cam.elevation},'mean':mean}
  (directory/'BLACK_PANEL_DIAGNOSTIC_V2.json').write_text(json.dumps(details,indent=2)+'\n')
  raise RuntimeError(f'Black scene panel: {details}')
 return frame
renderer._render_world=check
video=renderer.render_comparison_video(trace,directory/'comparison-v2.mp4',config)
assert hashes=={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
metadata={'scope':'Render-only camera/caption revision from unchanged saved traces; one fresh EGL process per video.','source_sha256':hashes,'render_source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(renderer.__file__)]},'presentation_only_overrides':{'title':title,'comparison_note':note,'camera_azimuth':-45},'render_config':asdict(config),'video_path':str(video.path.resolve()),'video_sha256':hashlib.sha256(video.path.read_bytes()).hexdigest(),'frame_count':video.frame_count,'duration_seconds':video.frame_count/video.fps,'source_time_support_seconds':[float(trace.time_seconds[0]),float(trace.time_seconds[-1])],'rendered_scene_panels_checked':len(checked),'minimum_scene_mean_rgb':min(r['scene_mean_rgb'] for r in checked),'no_black_scene_panels':True,'git_policy':'Local ignored video; do not stage.'}
(directory/'RENDER_METADATA_V2.json').write_text(json.dumps(metadata,indent=2)+'\n')
(directory/'SCENE_PANEL_CHECKS_V2.json').write_text(json.dumps(checked,indent=2)+'\n')
print(json.dumps(metadata),flush=True)
