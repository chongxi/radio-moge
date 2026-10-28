"""Verify completed capacity runs against W&B, accepting only complete metric rows."""

import argparse
from pathlib import Path
import json
import netrc
import shlex
import subprocess
import time
import hashlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
r = parser.parse_args().root
plan = json.loads((r / "plan.json").read_text())
assert json.loads((r / "status.json").read_text())["stage"] == "complete"
keys = [
    "macro/spatial/cosine",
    "macro/summary/cosine",
    "spatial/norm_ratio",
    "summary/norm_ratio",
    "optimizer_seconds",
    "elapsed_seconds",
    "inference_ms_per_image",
]
arms = {}
for spec in plan["candidates"]:
    a = r / spec["name"]
    streams = {}
    for ns in ("validation", "train_probe"):
        streams[ns] = [json.loads(s) for s in (a / (ns + "-nfe1.jsonl")).read_text().splitlines()]
    arms[spec["name"]] = dict(
        id=json.loads((a / "wandb-run.json").read_text())["id"],
        config=json.loads((a / "config.json").read_text()),
        streams=streams,
    )
assert len({v["id"] for v in arms.values()}) == len(plan["candidates"])
source = hashlib.sha256(json.dumps(arms, sort_keys=True).encode()).hexdigest()
progress = r / "wandb-final-progress.json"
verified = {}
if progress.exists():
    prior = json.loads(progress.read_text())
    assert prior["source_sha256"] == source
    verified = prior["runs"]
remaining = {k: v for k, v in arms.items() if k not in verified}
code = """import json,sys,os,time
p=json.load(sys.stdin);os.environ['WANDB_API_KEY']=p.pop('key');os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
import wandb
api=wandb.Api(timeout=60);proof={};failure=None
for name,item in p['arms'].items():
 section='load_run'
 try:
  run=api.run('brainsim/radio-moge/'+item['id']);section='finished_state';assert run.state=='finished',(name,run.state)
  section='experiment_config';assert run.config['experiment']==item['config']
  summary=dict(run.summary);section='last_step_and_lr';assert summary['optimizer_step']==1000 and summary['train/lr_used']==.0003
  errors=[];steps={};ignored={}
  for ns,rows in item['streams'].items():
   section=ns+'_history_fetch'
   history=list(run.scan_history(keys=['optimizer_step']+[ns+'/'+k for k in p['keys']],page_size=100))
   complete=[v for v in history if v.get('optimizer_step') is not None and all(v.get(ns+'/'+k) is not None for k in p['keys'])];ignored[ns]=len(history)-len(complete)
   actual={v['optimizer_step']:v for v in complete};expected={v['step']:v for v in rows};section=ns+'_history_steps';assert set(actual)==set(expected), (sorted(actual),sorted(expected))
   section=ns+'_metric_values'
   for step,row in expected.items():errors.extend(abs(actual[step][ns+'/'+k]-row[k]) for k in p['keys'])
   steps[ns]=sorted(actual)
  section='metric_tolerance';assert max(errors)<1e-10,max(errors)
  proof[name]=dict(id=item['id'],state=run.state,step=summary['optimizer_step'],lr=summary['train/lr_used'],max_metric_difference=max(errors),verified_steps=steps,ignored_history_rows_missing_requested_metrics=ignored)
 except Exception as e:
  failure=dict(name=name,section=section,exception_type=type(e).__name__,detail=str(e) if isinstance(e,AssertionError) else 'SDK/network exception text omitted')
  break
print(json.dumps(dict(stage='pending' if failure else 'passed',checked_at=time.time(),runs=proof,failure=failure)))
"""
if remaining:
    credential = netrc.netrc(str(Path.home() / ".netrc")).authenticators("api.wandb.ai")[2]
    p = subprocess.run(
        [
            "ssh",
            "-F",
            "/home/hub/.ssh/radio-moge-config",
            "radio-moge-gpu",
            shlex.join(["/home/buildhub/radio-moge/.venv/bin/python", "-c", code]),
        ],
        input=json.dumps(dict(key=credential, arms=remaining, keys=keys)),
        text=True,
        capture_output=True,
    )
    del credential
    if p.returncode:
        raise RuntimeError(
            "Read-only W&B helper failed outside guarded run check; credential-bearing output omitted."
        )
    d = json.loads(p.stdout)
    verified.update(d["runs"])
else:
    d = dict(stage="passed", checked_at=time.time(), failure=None)
d["runs"] = verified
d["source_sha256"] = source
f = progress
t = f.with_suffix(".incoming")
t.write_text(json.dumps(d, indent=2))
t.replace(f)
if d["stage"] == "passed":
    assert len(verified) == len(arms)
    f = r / "wandb-final-verification.json"
    t = f.with_suffix(".incoming")
    t.write_text(json.dumps(d, indent=2))
    t.replace(f)
print(json.dumps(dict(stage=d["stage"], runs=len(verified), failure=d.get("failure"))))
