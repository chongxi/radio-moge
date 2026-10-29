"""One short read-only utilization sample of the existing production trainer."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from monitor_joint_full_day import write


def sample(root, remote_root, output_name='performance-utilization-verification.json', after_step=0):
    if Path(output_name).name != output_name or not output_name.endswith('.json'):
        raise ValueError('Output must be a JSON basename')
    output = root / output_name
    if output.exists():
        print(json.dumps(dict(stage='already_sampled',proof=str(output))))
        return
    code='''import json,os,statistics,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]);after_step=int(sys.argv[2]);state=json.loads((r/'status.json').read_text())
if state['stage']!='training':
 print(json.dumps(dict(stage='pending',reason='Not yet in production training')));sys.exit(0)
perf=json.loads((r/'train/performance.json').read_text());pid=state['child_pid'];assert Path(f'/proc/{pid}').exists()
def last():
 with (r/'train/train.jsonl').open('rb') as f:
  f.seek(0,2);length=f.tell();f.seek(max(0,length-65536));data=f.read().decode()
 for line in reversed(data.splitlines()):
  try:return json.loads(line)
  except json.JSONDecodeError:continue
 raise ValueError('No completed training row')
def process():
 p=Path(f'/proc/{pid}');s=(p/'stat').read_text().split(') ',1)[1].split()
 io={k:int(v) for k,v in (line.split(':',1) for line in (p/'io').read_text().splitlines())}
 return dict(state=s[0],cpu_seconds=(int(s[11])+int(s[12]))/os.sysconf('SC_CLK_TCK'),io=io)
start=time.time();before=last();proc_before=process();assert before['step']>perf['anchor_step'] and before['effective_batch']==perf['effective_batch']
if after_step:
 cfg=json.loads((r/'config.json').read_text());interval=cfg['train']['eval_every'];position=before['schedule_position']
 until_eval=(interval-position%interval)/perf['schedule_ratio']
 age=time.time()-(r/'train/train.jsonl').stat().st_mtime
 if before['step']<after_step+20 or age>20 or until_eval<120:
  print(json.dumps(dict(stage='pending',reason='Wait for a fresh production window away from validation',last_step=before['step'])));sys.exit(0)
val_before=(r/'train/validation.jsonl').stat().st_size
eval_dirs_before=sorted(p.name for p in (r/'train').glob('eval-*'))
rows=[]
fields=['utilization.gpu','utilization.memory','memory.used','memory.total','power.draw','power.limit']
for i in range(30):
 begin=time.monotonic();values=[float(x.strip()) for x in subprocess.check_output(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],text=True).strip().split(',')]
 rows.append(dict(at=time.time(),**dict(zip(fields,values))))
 if i<29:time.sleep(max(0,1-(time.monotonic()-begin)))
after=last();proc_after=process();finished=time.time()
assert json.loads((r/'status.json').read_text())['child_pid']==pid
assert (r/'train/validation.jsonl').stat().st_size==val_before,'Sampling crossed evaluation; label separately'
assert sorted(p.name for p in (r/'train').glob('eval-*'))==eval_dirs_before,'Validation started during sampling'
assert after['step']>before['step'] and after['effective_batch']==perf['effective_batch']
images=(after['logical_data_cursor']-before['logical_data_cursor'])*4
elapsed=after['elapsed_seconds']-before['elapsed_seconds'];optimizer=after['optimizer_seconds']-before['optimizer_seconds']
assert images>0 and elapsed>0 and optimizer>0
print(json.dumps(dict(stage='passed',checked_at=finished,started_at=start,pid=pid,samples=rows,
 summary={k:dict(mean=statistics.mean(x[k] for x in rows),min=min(x[k] for x in rows),max=max(x[k] for x in rows)) for k in fields},
 profile=perf,before=before,after=after,production_images=images,logged_wall_span_seconds=elapsed,
 production_wall_images_per_second=images/elapsed,production_train_loop_images_per_second=images/optimizer,
 no_completed_validation_during_sample=True,process_before=proc_before,process_after=proc_after,
 cpu_core_equivalents=(proc_after['cpu_seconds']-proc_before['cpu_seconds'])/(finished-start))))
'''
    result=subprocess.run(['ssh','-F','/home/hub/.ssh/radio-moge-config','radio-moge-gpu',
                           shlex.join(['/opt/conda/bin/python3','-c',code,remote_root,str(after_step)])],
                          text=True,capture_output=True,check=True)
    proof=json.loads(result.stdout)
    if proof['stage']=='pending':
        print(json.dumps(proof))
        return
    proof['copied_at']=time.time()
    write(output,proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in ('samples','before','after','profile','process_before','process_after')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('remote_root')
    p.add_argument('--output-name',default='performance-utilization-verification.json')
    p.add_argument('--after-step',type=int,default=0)
    a=p.parse_args()
    sample(a.root,a.remote_root,a.output_name,a.after_step)
