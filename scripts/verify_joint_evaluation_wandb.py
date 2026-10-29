"""Check one independently audited evaluation against its existing W&B history."""

import argparse
import hashlib
import json
import netrc
from pathlib import Path
import shlex
import subprocess

from monitor_joint_full_day import write
from wandb_verification import REMOTE_COMPARISON


REMOTE = REMOTE_COMPARISON+'''import json,os,sys,time
p=json.load(sys.stdin);os.environ['WANDB_API_KEY']=p.pop('key');os.environ['WANDB_BASE_URL']='https://api.forge.coreweave.com'
import wandb
run=wandb.Api(timeout=60).run(p['run_path'])
assert run.state in ('running','finished')
config_differences=compare_metadata(run.config['experiment'],p['config'])
keys=['optimizer_step']+['validation/'+k for k in p['expected']]
for row in run.scan_history(keys=keys,page_size=1000):
 if row.get('optimizer_step')!=p['step'] or any(row.get('validation/'+k) is None for k in p['expected']):continue
 errors={k:abs(row['validation/'+k]-v) for k,v in p['expected'].items()}
 assert max(errors.values())<1e-10
 print(json.dumps(dict(stage='passed',checked_at=time.time(),id=run.id,state=run.state,
                      step=p['step'],verified_metrics=len(errors),max_metric_difference=max(errors.values()),
                      config_verified=True,config_float_differences=config_differences,float_tolerance='one ULP')))
 break
else:raise AssertionError('Audited evaluation missing from W&B history')
'''


def verify(root, step):
    audit = json.loads((root / f'eval-{step:07d}-verification.json').read_text())
    metrics_path = root / f'train/eval-{step:07d}/metrics.json'
    metrics_sha = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    assert audit['stage'] == 'passed' and audit['sha256']['metrics.json'] == metrics_sha
    identity = json.loads((root / 'train/wandb-run.json').read_text())
    startup = root/'startup-verification.json'
    if (root/'plan.json').exists() and json.loads((root/'plan.json').read_text()).get('additional_updates') == 30000:
        proof = json.loads(startup.read_text())
        assert proof['stage'] == 'passed' and identity == proof['new_wandb']
        assert identity['id'] != 'eadda6e3'
    else:
        assert identity['id'] == 'eadda6e3'
    assert identity['mode'] == 'online'
    proof_path = root / f'eval-{step:07d}-wandb-verification.json'
    if proof_path.exists():
        prior = json.loads(proof_path.read_text())
        assert prior['stage'] == 'passed' and prior['metrics_sha256'] == metrics_sha
        print(json.dumps(dict(stage='already_verified', step=step)))
        return
    rows = [json.loads(line) for line in (root / 'train/validation.jsonl').read_text().splitlines()]
    row = next(row for row in rows if row['step'] == step)
    expected = {k: v for k, v in row.items() if k != 'step' and type(v) in (int, float)}
    metrics = json.loads(metrics_path.read_text())
    assert all(expected[k] == v for k, v in metrics.items())
    config = json.loads((root / 'config.json').read_text())
    secret = netrc.netrc(str(Path.home() / '.netrc')).authenticators('api.wandb.ai')[2]
    payload = dict(key=secret, run_path='/'.join(identity[k] for k in ('entity', 'project', 'id')),
                   config=config, step=step, expected=expected)
    result = subprocess.run(
        ['ssh', '-F', '/home/hub/.ssh/radio-moge-config', 'radio-moge-gpu',
         shlex.join(['/home/buildhub/radio-moge/.venv/bin/python', '-c', REMOTE])],
        input=json.dumps(payload), text=True, capture_output=True,
    )
    del secret, payload
    if result.returncode:
        raise RuntimeError('Read-only evaluation verification failed; credential-bearing output withheld')
    value = json.loads(result.stdout)
    value['metrics_sha256'] = metrics_sha
    value['independent_audit'] = audit['sha256']
    write(proof_path, value)
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('step', type=int)
    args = parser.parse_args()
    verify(args.root, args.step)
