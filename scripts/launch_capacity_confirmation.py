"""Launch the predeclared read-only capacity confirmation once after training exits."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from backup_architecture_screen import atomic


def start(root, remote_root):
    if (root / "confirmation-launch.json").exists():
        raise FileExistsError("Inspect existing confirmation; never duplicate")
    plan = json.loads((root / "plan.json").read_text())
    revision = plan["execution_revision"]
    source = Path(__file__).with_name("confirm_architecture_finalists.py")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    code = (
        """from pathlib import Path
import hashlib,json,os,subprocess,time
r=Path(ROOT);repo=r.parent/'code'/REVISION[:7]
read=lambda p:json.loads(p.read_text())
for name in ['confirmation-launch.json','confirmation-status.json','confirmation-error.json']:
 assert not (r/name).exists(),name
queue=read(r/'queue-status.json');status=read(r/'status.json')
assert queue['child_pid'] is None and queue['child_exit']==0
assert status['stage']=='complete' and len(status['completed'])==COUNT
assert all(x['step']==1000 and x['cursor']==16000 and x['reason']=='step_limit' for x in status['completed'])
for pid in [queue['pid'],status['pid']]:
 p=Path('/proc')/str(pid)/'stat'
 assert not p.exists() or p.read_text().split(') ')[1].split()[0]=='Z'
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not gpu, 'GPU is in use; wait for existing work'
script=repo/'scripts/confirm_architecture_finalists.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()==DIGEST
shortlist=read(r/'shortlist.json')
assert len(shortlist['selected'])==COUNT and shortlist['frozen_at']<status['started_at']
def atomic(p,d):
 t=p.with_suffix(p.suffix+'.incoming');t.write_text(json.dumps(d,indent=2));t.replace(p)
deployment=dict(stage='deployed',revision=REVISION,repo=str(repo),checked_at=time.time(),script_sha256=DIGEST,source='Same immutable tested deployment as training; no new remote code')
atomic(r/'confirmation-code-deployment.json',deployment)
command=['/home/buildhub/radio-moge/.venv/bin/python','-u',str(script),'--root',str(r),'--revision',REVISION]
env=os.environ.copy();env.update(PYTHONPATH=str(repo/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
with (r/'confirmation-console.log').open('x') as log:
 child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch=dict(pid=child.pid,started_at=time.time(),command=command,scope='AllCOUNT predeclared endpoints;224features/56geometry/32NYUGT;0updates,1800s budget;previously viewed comparison panel')
atomic(r/'confirmation-launch.json',launch)
print(json.dumps(dict(launch=launch,deployment=deployment)))
""".replace("ROOT", repr(str(remote_root)))
        .replace("REVISION", repr(revision))
        .replace("DIGEST", repr(digest))
        .replace("COUNT", str(len(plan["candidates"])))
    )
    result = subprocess.run(
        [
            "ssh",
            "-F",
            "/home/hub/.ssh/radio-moge-config",
            "radio-moge-gpu",
            shlex.join(["/opt/conda/bin/python3", "-c", code]),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    atomic(root / "confirmation-launch.json", payload["launch"])
    atomic(root / "confirmation-code-deployment.json", payload["deployment"])
    print(json.dumps(payload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("remote_root", type=Path)
    args = parser.parse_args()
    start(args.root, args.remote_root)
