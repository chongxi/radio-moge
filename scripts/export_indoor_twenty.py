from pathlib import Path
from collections import defaultdict
import base64,gzip,hashlib,json,os,random,time
import numpy as np
import torch
from PIL import Image
from radio_moge.system import build_system
from radio_moge.data import read_manifest,evaluation_image
from radio_moge.decoder import postprocess
from radio_moge.pointcloud import verify_geometry
from radio_moge.metrics import depth_metrics
from radio_moge.compare import fingerprint
from radio_moge.checkpoint import atomic_json

torch.set_num_threads(2)
base=Path('/home/buildhub/shared/radio-moge');old=base/'chongxi/nyuv2-ablations-20261021/fusion-features/heldout-viewer-v1';run=base/'chongxi/indoor-fusion-20261021'
out=run/'heldout-twenty-v2';out.mkdir(exist_ok=False)
manifest=read_manifest(base/'datasets/nyuv2-labeled-s42/test.jsonl');om=json.loads((old/'manifest.json').read_text());og=json.loads((old/'geometry/manifest.json').read_text())
assert og['depth_manifest_sha256']==hashlib.sha256((old/'manifest.json').read_bytes()).hexdigest()
assert [r['id'] for r in manifest]==[r['id'] for r in om['samples']]==[r['id'] for r in og['samples']]
groups=defaultdict(list)
for i,r in enumerate(manifest):groups[r['scene_type']].append(i)
rng=random.Random(42)
for rows in groups.values():rng.shuffle(rows)
selected=[];seen=set()
while len(selected)<20:
 for kind in sorted(groups):
  while groups[kind] and manifest[groups[kind][-1]]['group'] in seen:groups[kind].pop()
  if groups[kind]:
   i=groups[kind].pop();selected.append(i);seen.add(manifest[i]['group'])
   if len(selected)==20:break
state=torch.load(run/'features/train/best.pt',map_location='cpu',weights_only=True);assert state['step']==29000
config=state['config'];testrows=[manifest[i] for i in selected];testpaths={r['image'] for r in testrows};testhashes={hashlib.sha256(Path(r['image']).read_bytes()).hexdigest() for r in testrows};testgroups={r['group'] for r in testrows}
splits=[]
for path in [Path(config['data']['train_manifest']),Path(config['data']['val_manifest']),Path(config['data']['val_manifest']).with_name('val-full.jsonl')]:
 n=0
 with path.open() as f:
  for line in f:
   r=json.loads(line);n+=1
   assert os.path.abspath(path.parent/r['image']) not in testpaths
   assert r.get('sha256') not in testhashes and r.get('group') not in testgroups
 splits.append({'manifest':str(path),'images':n,'overlap':0})
for r in testrows:assert hashlib.sha256(np.asarray(Image.open(r['image']).convert('RGB'),dtype=np.uint8).tobytes()).hexdigest()==r['sha256']
report={'selected_ids':[r['id'] for r in testrows],'selection':'Seed42; round-robin scene types, unique scenes, fixed before inference;20 of previous290 held-out NYUv2 frames.','splits':splits,'checkpoint_step':state['step'],'adapter_sha256':fingerprint(state['adapter']),'old_adapter_sha256':om['run']['checkpoint_sha256'],'train_images':256886,'old_train_images':1043,'samples':[],'status':'exporting'}
atomic_json(report,out/'selection.json');print('Selection and split checks passed',flush=True)
system,adapter=build_system(config,'cuda');adapter.load_state_dict(state['adapter']);adapter.eval()
with torch.no_grad():
 for j,i in enumerate(selected):
  r=manifest[i];s=om['samples'][i];g=og['samples'][i];image,gt=evaluation_image(r,config['evaluation']['max_side']);image=image.cuda();h,w=image.shape[-2:]
  saved=np.frombuffer(gzip.decompress((old/s['depth']).read_bytes()),dtype='<f4').reshape(4,h,w)
  with torch.autocast('cuda',dtype=torch.bfloat16):features,target=system.encode(image,adapter.indices,config['evaluation']['num_tokens']);encoded=adapter(features)
  pred=postprocess(system.decoder(encoded,(h,w)),image,config['models'].get('moge_source'))
  teacher=postprocess(system.decoder(target,(h,w)),image,config['models'].get('moge_source'))
  np.testing.assert_allclose(teacher['depth'][0].cpu().numpy(),saved[2],rtol=1e-5,atol=1e-5,equal_nan=True)
  np.testing.assert_allclose(gt.numpy(),saved[3],rtol=0,atol=0,equal_nan=True)
  depth=pred['depth'][0].cpu().numpy();k,mask,err=verify_geometry(depth,pred)
  oldmask=np.array(Image.open(old/g['after']['mask']))>0;teachermask=np.array(Image.open(old/g['teacher']['mask']))>0
  np.testing.assert_array_equal(teachermask,teacher['mask'][0].cpu().numpy())
  depths=[saved[1].copy(),depth,saved[2].copy()];masks=[oldmask,mask,teachermask]
  metrics={}
  for name,d in zip(['old','new','teacher'],depths):
   metrics[name]={'sensor':depth_metrics(torch.from_numpy(d),gt,gt>0),'teacher':depth_metrics(torch.from_numpy(d),torch.from_numpy(saved[2].copy()),torch.from_numpy(teachermask))}
  stride=4;blob=b''.join(np.asarray(d[::stride,::stride],dtype='<f4').tobytes() for d in depths)+b''.join(np.asarray(m[::stride,::stride],dtype=np.uint8).tobytes() for m in masks)
  sample={'id':r['id'],'scene':r['scene'],'type':r['scene_type'],'width':w,'height':h,'stride':stride,'range':s['range'],'cameras':[g['after']['intrinsics'],k,g['teacher']['intrinsics']],'payload':base64.b64encode(gzip.compress(blob,compresslevel=6,mtime=0)).decode(),'rgb':'data:image/jpeg;base64,'+base64.b64encode((old/s['rgb']).read_bytes()).decode(),'metrics':metrics,'projection_error_m':err}
  report['samples'].append(sample);print(f'Exported {j+1}/20',flush=True)
report['summary']={name:{ref:{metric:float(np.mean([s['metrics'][name][ref][metric] for s in report['samples']])) for metric in ['abs_rel','rmse','delta1']} for ref in ['sensor','teacher']} for name in ['old','new','teacher']}
report['status']='complete';report['finished_at']=time.time();atomic_json(report,out/'data.json');print(json.dumps(report['summary']),flush=True)
