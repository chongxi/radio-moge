from pathlib import Path
import subprocess,os,select,json,time,hashlib,signal
A=Path('/home/buildhub/shared/radio-moge/chongxi/new-architecture-rgb-20261104');R=Path('/home/buildhub/shared/radio-moge/chongxi/geometry-corpus-20261102');p=A/'procthor-x11-setup-02';o=Path(__file__).parent
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
e=os.environ.copy();e['LD_LIBRARY_PATH']=':'.join(str(x) for x in (p/'sysroot/usr/lib/x86_64-linux-gnu',A/'procthor-render-libs-01/sysroot/usr/lib/x86_64-linux-gnu',R/'scannet-cpu-render-env-01/sysroot/usr/lib/x86_64-linux-gnu'))
e.update(LIBGL_ALWAYS_SOFTWARE='1',GALLIUM_DRIVER='llvmpipe',LP_NUM_THREADS='1',LIBGL_DRIVERS_PATH=str(p/'sysroot/usr/lib/x86_64-linux-gnu/dri'),CUDA_VISIBLE_DEVICES='',MESA_SHADER_CACHE_DIR=str(A/'procthor-render-libs-01/shader-cache'),SDL_AUDIODRIVER='dummy')
e.pop('VK_ICD_FILENAMES',None);e.pop('PYOPENGL_PLATFORM',None)
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):e[k]='1'
r,w=os.pipe()
with (o/'xvfb.log').open('xb') as f:x=subprocess.Popen([str(p/'Xvfb-relocated'),'-displayfd',str(w),'-screen','0','800x600x24','-nolisten','tcp','-ac','-noreset','+extension','GLX','-xkbdir',str(p/'sysroot/usr/share/X11/xkb'),'-fp',str(p/'sysroot/usr/share/fonts/X11/misc')],env=e,pass_fds=(w,),stdout=f,stderr=subprocess.STDOUT)
os.close(w)
try:
 assert select.select([r],[],[],20)[0], 'X server timeout'
 b=os.read(r,64);assert b,'X server failed';e['DISPLAY']=':'+b.decode().strip()
 q=subprocess.run([str(p/'sysroot/usr/bin/glxinfo'),'-B'],env=e,capture_output=True,text=True,timeout=20);(o/'graphics.txt').write_text(q.stdout+q.stderr);assert q.returncode==0 and 'llvmpipe' in q.stdout
 (o/'display.json').write_text(json.dumps(dict(display=e['DISPLAY'],pid=x.pid,cpu_only=True))+'\n')
 plan_path=A/'structural-multiview-plan-03/plan.json'
 assert hashlib.sha256(plan_path.read_bytes()).hexdigest()=='213afbf753dc262076248ffca1e155025192acf0b40a246d5be9a1a9ef92e5a6'
 plan=json.loads(plan_path.read_text())
 script=A/'code-structural-multiview-v3/scripts/render_structural_room_batch.py'
 assert hashlib.sha256(script.read_bytes()).hexdigest()=='585fe5a957312306a9197cd1546332ca23845f68bda58785d66562374ead3e59'
 assert all(not (A/j['name']).exists() for j in plan['physical_jobs'])
 def status(value):
  temporary=o/'status.json.incoming';temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(o/'status.json')
 for job in plan['physical_jobs']:
  name=job['name'];command=[str(A/'procthor-render-env-01/bin/python'),'-u',str(script),'--root',str(A),'--output',str(A/name),'--split',job['split'],'--start-scene',str(job['start']),'--scenes',str(job['scenes']),'--views','20','--maximum-seconds',str(job['max_seconds']),'--maximum-quarantined-scenes',str(job['maximum_quarantined_scenes'])]
  child=subprocess.Popen(command,env=e,start_new_session=True)
  status(dict(stage='running',job=name,child_pid=child.pid,scope=job))
  try:result=child.wait(timeout=job['max_seconds']+90)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   status(dict(stage='timed_out_no_retry',job=name));raise
  if result!=0:
   status(dict(stage='failed_no_retry',job=name,returncode=result));raise RuntimeError('Generation child failed')
  report_path=A/name/'report.json';report=json.loads(report_path.read_text())
  if not report['all_requested_scenes_accounted']:
   status(dict(stage='partial_closed_no_retry',job=name,report=str(report_path)));raise RuntimeError('Preserve closed prefix; no automatic extension')
  assert report['frame_policy']==plan['frame_policy']
  assert report['scene_failure_policy']==plan['scene_failure_policy']
  assert report['scenes_closed']+report['scenes_quarantined']==job['scenes']
  assert report['scenes_quarantined']<=job['maximum_quarantined_scenes']
  assert sum(report[k] for k in ('completed','rejected_frames','held_candidate_frames','unattempted_held_slots'))==job['scenes']*20
  assert report['completed']>=report['scenes_closed']*18
  status(dict(stage='job_complete',job=name,report=str(report_path),report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),accepted=report['completed'],rejected=report['rejected_frames']))
 status(dict(stage='all_generation_jobs_complete_review_pending',optimizer_started=False,retained_count_requires_frozen_closure=True))
finally:
 os.close(r)
 if x.poll() is None:x.terminate();x.wait(timeout=10)
