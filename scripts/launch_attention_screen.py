"""Launch the explicitly approved ten-trial attention screen once, with a fresh tracker per arm."""

import argparse
import base64
import hashlib
import json
import netrc
from pathlib import Path
import shlex
import subprocess
import time

from backup_architecture_screen import atomic


def launch(root, remote_base):
    assert not (root / "launch.json").exists(), "Inspect the existing launch; never duplicate"
    plan = json.loads((root / "plan.json").read_text())
    deployment = json.loads((root / "code-deployment.json").read_text())
    assert plan["screen_profile"] == "attention_tricks" and len(plan["candidates"]) == 10
    assert deployment["stage"] == "passed" and deployment["remote_cpu_tests_passed"]
    revision = plan["execution_revision"]
    assert revision == deployment["execution_revision"]
    files = {}
    for p in root.iterdir():
        if p.is_file() and p.suffix in (".json", ".jsonl"):
            data = p.read_bytes()
            files[p.name] = dict(data=base64.b64encode(data).decode(),
                                 sha256=hashlib.sha256(data).hexdigest())
    code = '''from pathlib import Path
import json,sys,os,hashlib,base64,subprocess,fcntl,time
p=json.load(sys.stdin);key=p.pop('key');r=Path(p['root']);repo=Path(p['repo'])
assert not r.exists(), 'Inspect existing root before any launch'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU is in use'
with (r.parent/'attention-24h/.training.lock').open('a+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
r.mkdir()
for name,item in p['files'].items():
 data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256'];f=r/name;t=f.with_suffix(f.suffix+'.incoming');t.write_bytes(data);t.replace(f);assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256']
cmd=['/home/buildhub/radio-moge/.venv/bin/python','-u',str(repo/'scripts/run_architecture_queue.py'),'--root',str(r),'--repo',str(repo),'--revision',p['revision']]
env=os.environ.copy();env.update(PYTHONPATH=str(repo/'src'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_API_KEY=key)
with (r/'wrapper-console.log').open('x') as log:
 child=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
del key;env.pop('WANDB_API_KEY',None)
launch=dict(pid=child.pid,started_at=time.time(),command=cmd,execution_revision=p['revision'],scope='Authorized five attention variants x2initializationseeds,1000updates/600s per trial,3600s total; no automatic extension')
f=r/'launch.json';t=f.with_suffix('.incoming');t.write_text(json.dumps(launch,indent=2));t.replace(f)
print(json.dumps(dict(launch=launch,metadata_sha256={k:v['sha256'] for k,v in p['files'].items()})))
'''
    credential = netrc.netrc(str(Path.home() / ".netrc")).authenticators("api.wandb.ai")[2]
    payload = dict(key=credential, root=str(remote_base / root.name),
                   repo=str(remote_base / "code" / revision[:7]), revision=revision, files=files)
    response = subprocess.run(
        ["ssh", "-F", "/home/hub/.ssh/radio-moge-config", "radio-moge-gpu",
         shlex.join(["/opt/conda/bin/python3", "-c", code])],
        input=json.dumps(payload), capture_output=True, text=True,
    )
    del credential, payload
    if response.returncode:
        raise RuntimeError("Launch failed; inspect remote status before retry. Credential-bearing output suppressed.")
    result = json.loads(response.stdout)
    atomic(root / "launch.json", result["launch"])
    atomic(root / "initial-metadata-sync.json", dict(
        stage="passed", checked_at=time.time(), sha256=result["metadata_sha256"]))
    print(json.dumps(result["launch"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("remote_base", type=Path)
    args = parser.parse_args()
    launch(args.root, args.remote_base)
