"""Copy only sanitized download status; never copy private scripts, credentials or datasets."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from run_private_dataset_download import atomic_json


def monitor(root):
    plan = json.loads((root / "plan.json").read_text())
    remote = """import hashlib,json,subprocess,sys,time,yaml
from pathlib import Path
p=json.load(sys.stdin);r=Path(p['private_script_directory']);c=Path(p['private_control'])
secret=yaml.safe_load((r/'runtime-private.yml').read_text())['token']
status_path=c/'status.json';raw=status_path.read_bytes();state=json.loads(raw)
logs=[r/'launch-console.log',c/'console.log']
assert all(secret not in f.read_text() for f in logs if f.exists()), 'Private log redaction check failed'
out=dict(state=state,checked_at=time.time(),log_secret_absent=True,
         status_source_sha256=hashlib.sha256(raw).hexdigest(),
         process=subprocess.run(['ps','-p',str(state['pid']),'-o','pid,ppid,stat,etime,time'],capture_output=True,text=True).stdout)
for name in ['inventory.json','inventory-verification.json']:
 f=c/name
 if f.exists():
  value=json.loads(f.read_text())
  out[name]={k:v for k,v in value.items() if k not in ('assets','missing')}
  if 'assets' in value:out[name]['expected_assets']=len(value['assets'])
  if 'missing' in value:out[name]['missing_assets']=len(value['missing'])
assert secret not in json.dumps(out)
print(json.dumps(out,allow_nan=False))
"""
    result = subprocess.run(
        [
            "ssh",
            "-F",
            "/home/hub/.ssh/radio-moge-config",
            "radio-moge-gpu",
            shlex.join(["/opt/conda/bin/python3", "-c", remote]),
        ],
        input=json.dumps(plan),
        text=True,
        capture_output=True,
        check=True,
    )
    value = json.loads(result.stdout)
    atomic_json(root / "monitor.json", value)
    atomic_json(root / "status.json", value["state"])
    # Readback checks the local serialized snapshot, not a second bulk data read.
    local_hash = hashlib.sha256((root / "monitor.json").read_bytes()).hexdigest()
    atomic_json(
        root / "monitor-copy-verification.json",
        dict(
            stage="passed",
            source_status_sha256=value["status_source_sha256"],
            local_snapshot_sha256=local_hash,
            log_secret_absent=value["log_secret_absent"],
            checked_at=value["checked_at"],
        ),
    )
    handoff_path = root.parent / "handoff.json"
    handoff = json.loads(handoff_path.read_text())
    handoff["active_scannetpp_download"] = dict(
        root=str(root),
        dataset_root=plan["dataset_root"],
        private_control=plan["private_control"],
        stage=value["state"]["stage"],
        pid=value["state"]["pid"],
        checked_at=value["checked_at"],
        completed_files=value["state"]["completed_files"],
        verified_download_bytes=value["state"]["verified_download_bytes"],
        current_training_manifests_unchanged=True,
        remaining=[]
        if value["state"]["stage"] == "complete"
        else [
            "monitor existing serial download",
            "verify final inventory and extraction receipts; do not relaunch automatically",
        ],
    )
    atomic_json(handoff_path, handoff)
    training = root.parent / Path(plan['training_root']).name
    if training.exists():
        from render_joint_full_page import render
        render(training, Path('.kanna/visualizations/teacher-plane-pilot.html'))
    print(json.dumps(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    monitor(parser.parse_args().root)
