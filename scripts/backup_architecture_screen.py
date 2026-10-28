"""One serial verified backup of scratch checkpoints and completed confirmation artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".incoming")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def start(base, remote_base, name, bundle_metadata=False):
    root, remote = base / name, remote_base / name
    review = root / "final-review"
    if (review / "launch.json").exists():
        raise FileExistsError("Inspect existing backup; never duplicate controller")
    review.mkdir(exist_ok=True)
    source = review / "source-verification.json"
    if not source.exists():
        code = """from pathlib import Path
import hashlib,json,time,torch,tarfile
r=Path(ROOT)
read=lambda p:json.loads(p.read_text())
assert read(r/'status.json')['stage']=='complete'
assert read(r/'confirmation-status.json')['stage']=='complete'
assert read(r/'confirmation-verification.json')['stage']=='passed'
assert read(r/'wandb-final-verification.json')['stage']=='passed'
assert not Path('/proc/'+str(read(r/'confirmation-status.json')['pid'])).exists()
plan=read(r/'plan.json');prior=read(r/'confirmation/checkpoint-sources.json')
def identity(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
names=['plan.json','launch.json','status.json','queue-status.json','train.jsonl','validation.jsonl','confirmation.jsonl','data-verification.json','configuration-verification.json','code-deployment.json','startup-verification.json','startup-wandb-verification.json','preflight-verification.json','curve-verification.json','wandb-final-verification.json','screen-report.json','shortlist.json','confirmation-policy.json','confirmation-code-deployment.json','confirmation-launch.json','confirmation-status.json','confirmation-verification.json','cache-index.json','cache-load-verification.json']
paths=[r/n for n in names]+list(r.glob('*-console.log'))+list((r/'confirmation').rglob('*.json'))+list((r/'confirmation').rglob('*.npz'))+list((r/'curve-audits').glob('*.json'))
paths += [r/n for n in ['capacity-report.json','result-report.json','previous-1024-reference.json','planned-parameters.json','reused-code-verification.json','seed43-verification.json'] if (r/n).exists()]
for spec in plan['candidates']:
 a=r/spec['name'];paths+=list(a.glob('*.json'))+list(a.glob('*.jsonl'))+list(a.glob('validation-*/*.json'))+list(a.glob('train_probe-*/*.json'))
paths=sorted(set(paths),key=lambda p:(p.suffix!='.npz',str(p)))
paths+=[r/'statistics.pt']+[r/spec['name']/'latest.pt' for spec in plan['candidates']]
files=[]
for p in paths:
 before=identity(p);relative=str(p.relative_to(r));saved=prior.get(p.parent.name) if p.name=='latest.pt' else None
 if saved:
  assert saved['identity']==before and saved['source']==str(p);digest=saved['sha256']
 else:
  with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 assert identity(p)==before
 item=dict(relative=relative,source=str(p),bytes=before[2],sha256=digest,identity=before)
 if p.suffix=='.pt':
  if saved:item['checkpoint']=dict(step=saved['step'],cursor=saved['cursor'],adam_steps=saved['adam_steps'],metadata_reused_from_read_only_confirmation=True)
  else:
   state=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
   if p.name=='statistics.pt':
    assert all(torch.isfinite(v).all() for v in state.values());item['statistics_keys']=list(state)
   else:
    stop=read(p.parent/'stop-report.json');counters=sorted({int(s['step']) for s in state['optimizer']['state'].values()})
    assert state['step']==stop['step']==1000 and state['cursor']==16000 and counters==[1000]
    assert state['optimizer']['param_groups'][0]['lr']==.0003 and state['execution_revision']==plan['execution_revision']
    item['checkpoint']=dict(step=state['step'],cursor=state['cursor'],adam_steps=counters,optimizer_lr=.0003,execution_revision=state['execution_revision'],initialization=state['initialization'])
   del state
 assert identity(p)==before
 files.append(item)
bundled=[]
if BUNDLE_METADATA:
 bundled=[f['relative'] for f in files if Path(f['relative']).suffix not in ('.pt','.npz')]
 archive=r/'backup-metadata.tar';temporary=archive.with_suffix('.incoming')
 assert not archive.exists() and not temporary.exists(), 'Inspect existing metadata bundle before retry'
 with tarfile.open(temporary,'w') as tar:
  for item in files:
   if item['relative'] in bundled:
    p=Path(item['source']);assert identity(p)==item['identity']
    tar.add(p,arcname=item['relative'],recursive=False)
    assert identity(p)==item['identity']
 temporary.replace(archive)
 before=identity(archive)
 with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 assert identity(archive)==before
 files.append(dict(relative=archive.name,source=str(archive),bytes=before[2],sha256=digest,identity=before,metadata_bundle=True))
print(json.dumps(dict(stage='passed',checked_at=time.time(),files=files,bundled_metadata=bundled,cache_excluded='45GB frozen feature cache remains persistent remote NFS; original raw data already backed up.')))
""".replace("ROOT", repr(str(remote))).replace("BUNDLE_METADATA", repr(bundle_metadata))
        result = subprocess.run(
            [
                "ssh",
                "-F",
                "/home/hub/.ssh/radio-moge-config",
                "radio-moge-gpu",
                "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "
                + shlex.join(["/home/buildhub/radio-moge/.venv/bin/python", "-c", code]),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        atomic(source, json.loads(result.stdout))
    verified = json.loads(source.read_text())
    staging = review / "transfer-staging"
    staging.mkdir(exist_ok=True)
    items = []
    for item in verified["files"]:
        if item["relative"] in verified.get("bundled_metadata", []):
            continue
        relative = Path(item["relative"])
        if relative.suffix == ".npz":
            destination = root / relative
        elif relative.suffix == ".pt":
            destination = review / (
                relative.parent / "last.pt" if relative.name == "latest.pt" else relative
            )
        elif item.get("metadata_bundle"):
            destination = review / relative
        else:
            destination = review / "artifacts" / relative
        items.append(
            dict(
                source=item["source"],
                destination=str(destination),
                bytes=item["bytes"],
                sha256=item["sha256"],
            )
        )
    plan = dict(
        purpose="Completed scratch architecture screen and read-only finalist confirmation backup",
        staging=str(staging),
        datasets=[],
        bwlimit_kib=20480,
        ssh_command="ssh -F /home/hub/.ssh/radio-moge-config",
        ssh_host="radio-moge-gpu",
        source_root=str(remote),
        checkpoints_first=items,
    )
    atomic(review / "plan.json", plan)
    script = base / "code/2bb0342/mirror_attention_datasets.py"
    assert (
        hashlib.sha256(script.read_bytes()).hexdigest()
        == "d1efdc6cffde534a53e715f659fbd47a418fd169aa910865983019701b9569d8"
    )
    command = [
        "nice",
        "-n",
        "15",
        "/usr/bin/python3",
        "-u",
        str(script),
        "--plan",
        str(review / "plan.json"),
    ]
    with (review / "console.log").open("w") as log:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launch = dict(
        pid=child.pid,
        command=command,
        started_at=time.time(),
        files=len(items),
        total_bytes=sum(i["bytes"] for i in items),
    )
    atomic(review / "launch.json", launch)
    print(json.dumps(launch))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--remote-base", type=Path, required=True)
    p.add_argument("--name", default="adapter-scratch-diverse-screen-20261028")
    p.add_argument(
        "--bundle-metadata",
        action="store_true",
        help="Transfer small artifacts in one verified archive, retaining per-file hashes",
    )
    a = p.parse_args()
    start(a.base, a.remote_base, a.name, a.bundle_metadata)
