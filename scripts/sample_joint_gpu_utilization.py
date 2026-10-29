"""One short read-only utilization sample of the existing production trainer."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from monitor_joint_full_day import write


def sample(root, remote_root):
    output = root / 'performance-utilization-verification.json'
    if output.exists():
        print(json.dumps(dict(stage='already_sampled',proof=str(output))))
        return
    code='''import json,os,statistics,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]);state=json.loads((r/'status.json').read_text());assert state['stage']=='training'
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
val_before=(r/'train/validation.jsonl').stat().st_size
rows=[]
fields=['utilization.gpu','utilization.memory','memory.used','memory.total','power.draw','power.limit']
for i in range(30):
 begin=time.monotonic();values=[float(x.strip()) for x in subprocess.check_output(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],text=True).strip().split(',')]
 rows.append(dict(at=time.time(),**dict(zip(fields,values))))
 if i<29:time.sleep(max(0,1-(time.monotonic()-begin)))
after=last();proc_after=process();finished=time.time()
assert json.loads((r/'status.json').read_text())['child_pid']==pid
assert (r/'train/validation.jsonl').stat().st_size==val_before,'Sampling crossed evaluation; label separately'
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
                           shlex.join(['/opt/conda/bin/python3','-c',code,remote_root])],
                          text=True,capture_output=True,check=True)
    proof=json.loads(result.stdout)
    proof['copied_at']=time.time()
    write(output,proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in ('samples','before','after','profile','process_before','process_after')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('remote_root')
    a=p.parse_args()
    sample(a.root,a.remote_root)
