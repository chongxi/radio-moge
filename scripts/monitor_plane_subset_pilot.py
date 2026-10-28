"""Read the existing pilot, atomically mirror only small logs, and refresh its web page."""

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.incoming')
    temporary.write_bytes(value)
    temporary.replace(path)


def monitor(base, remote_base, name='teacher-plane-subset-pilot-20261028'):
    root = base / name
    source = '''from pathlib import Path
import json,base64,subprocess,time,hashlib
r=Path(REMOTE_ROOT)
files={}
for pattern in ('status.json','plan.json','*-config.json','preflight-verification.json',
                'wrapper-console.log','*-console.log',
                '*/config.json','*/provenance.json','*/wandb-run.json','*/stop-report.json',
                '*/train.jsonl','*/validation.jsonl','*/eval-*/metrics.json',
                '*/eval-*/per_image.json','initial-adapter-verification.json',
                '*/train-probe.jsonl','*/train-probe-*/metrics.json','*/train-probe-*/per_image.json',
                'assessment/*/*/metrics.json','assessment/*/*/per_image.json'):
 for p in r.glob(pattern):
  assert p.stat().st_size < 4_000_000
  payload=p.read_bytes()
  files[str(p.relative_to(r))]=dict(data=base64.b64encode(payload).decode(),sha256=hashlib.sha256(payload).hexdigest())
state=json.loads((r/'status.json').read_text())
pids=[p for p in (state.get('pid'),state.get('child_pid')) if p]
processes=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid,stat,etime,cputime,rss,comm'],capture_output=True,text=True).stdout
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout
print(json.dumps(dict(observed_at=time.time(),files=files,processes=processes,gpu=gpu)))
'''.replace('REMOTE_ROOT', repr(str(remote_base / name)))
    result = subprocess.run(
        ['ssh', '-F', '/home/hub/.ssh/radio-moge-config', 'radio-moge-gpu',
         '/opt/conda/bin/python3 -c ' + shlex.quote(source)], capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    report = json.loads(result.stdout)
    receipts = {}
    for relative, content in report.pop('files').items():
        payload = base64.b64decode(content['data'])
        assert hashlib.sha256(payload).hexdigest() == content['sha256']
        atomic(root / relative, payload)
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == content['sha256']
        receipts[relative] = content['sha256']
    report['transfer_sha256'] = receipts
    report['status'] = json.loads((root / 'status.json').read_text())
    report['arms'] = {}
    def complete_rows(path):
        return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
                if line.endswith('\n')] if path.exists() else []

    plan_path = root / 'plan.json'
    arms = json.loads(plan_path.read_text()).get('arms', ['plane', 'control']) if plan_path.exists() else ['plane', 'control']
    for arm in arms:
        path = root / arm / 'train.jsonl'
        rows = complete_rows(path)
        val = root / arm / 'validation.jsonl'
        vals = complete_rows(val)
        probes = complete_rows(root / arm / 'train-probe.jsonl')
        report['arms'][arm] = dict(
            logged_rows=len(rows), last=rows[-1] if rows else None,
            exact_lr=all(r['lr'] == r['lr_used'] == 1e-5 for r in rows) if rows else None,
            finite=all(math.isfinite(v) for r in rows for v in r.values()
                       if isinstance(v, (float, int))) if rows else None,
            validation_steps=[r['step'] for r in vals],
            training_diagnostic_steps=[r['step'] for r in probes])
    atomic(root / 'monitor.json', json.dumps(report, indent=2).encode())
    handoff_path = base / 'handoff.json'
    handoff = json.loads(handoff_path.read_text())
    expanded = name == 'teacher-plane-coverage512-pilot-20261028'
    flatness = name == 'teacher-plane-ownflat512-pilot-20261028'
    joint = name == 'adapter-joint-cosine512-pilot-20261028'
    dual = name == 'adapter-dual-cosine512-pilot-20261028'
    key = ('active_adapter_dual_cosine_pilot' if dual else 'active_adapter_joint_pilot' if joint else 'active_plane_flatness_pilot' if flatness else
           'active_plane_coverage_pilot' if expanded else 'active_plane_pilot')
    handoff.setdefault(key, {}).update(
        stage=report['status']['stage'], training_launched=True,
        last_observed_at=report['observed_at'], status=report['status'], arms=report['arms'],
        monitor=str(root / 'monitor.json'), viewer_url='https://buildhub-nano-test--p7869.pub.astera.work/')
    handoff['updated_at'] = time.time()
    atomic(handoff_path, json.dumps(handoff, indent=2).encode())
    renderer = ('render_dual_cosine_page.py' if dual else 'render_joint_cosine_page.py' if joint else 'render_plane_flatness_page.py' if flatness else
                'render_plane_coverage_page.py' if expanded else 'render_plane_pilot_page.py')
    subprocess.run([sys.executable, str(Path(__file__).with_name(renderer)),
                    '--base', str(base), '--output',
                    str(Path(__file__).resolve().parents[1] / '.kanna/visualizations/teacher-plane-pilot.html')],
                   check=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--remote-base', type=Path, required=True)
    parser.add_argument('--name', default='teacher-plane-subset-pilot-20261028')
    args = parser.parse_args()
    monitor(args.base, args.remote_base, args.name)
