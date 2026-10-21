"""Run the fixed NYUv2 fusion/geometry comparison sequentially on one GPU.

Does not evaluate the outer test set or start/stop any compute machine.
Credentials are inherited from the environment and never written to the plan.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    root=args.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'plan.json').exists():
            raise FileExistsError('A plan already exists; inspect status before resuming any stage')
        baseline=torch.load(args.baseline,map_location='cpu',weights_only=True,mmap=True)
        stages=[]
        for name in ['fusion-features','mlp-geometry','fusion-geometry']:
            stage=root/name
            stage.mkdir()
            config=json.loads((ROOT/'configs/ablations'/f'{name}.json').read_text())
            for key in ['train_manifest','val_manifest']:
                assert config['data'][key]==baseline['config']['data'][key], 'Data split changed'
            assert config['models']==baseline['config']['models'], 'Frozen models changed'
            config['output_dir']=str(stage/'train')
            config['code_revision']=args.revision
            init=(str(args.baseline.resolve()) if name=='mlp-geometry' else
                  str(root/'fusion-features/train/best.pt') if name=='fusion-geometry' else None)
            config['initialization_checkpoint']=init
            atomic_json(config,stage/'config.json')
            stages.append(dict(name=name,config=str(stage/'config.json'),init=init,
                               output=config['output_dir'],steps=config['train']['steps']))
        plan=dict(revision=args.revision,stages=stages,baseline=str(args.baseline.resolve()),
                  baseline_step=baseline['step'],selection='validation/teacher_depth/abs_rel',
                  test_policy='No outer test evaluation during training or checkpoint selection.',
                  comparison_note='Geometry stages add 1000 steps from each feature-stage best. '
                  'They are practical refinements, not equal-total-step causal ablations.',
                  gpu_policy='Keep the GPU running after this queue finishes.')
        del baseline
        atomic_json(plan,root/'plan.json')
        completed=[]
        for stage in stages:
            status=dict(stage='training',active=stage['name'],completed=completed,
                        started_at=time.time(),revision=args.revision)
            if stage['init']:
                source=torch.load(stage['init'],map_location='cpu',weights_only=True,mmap=True)
                status['initialization_step']=source['step']
                del source
            atomic_json(status,root/'status.json')
            command=[sys.executable,'-u','-m','radio_moge.train','--config',stage['config']]
            if stage['init']:
                command+=['--init',stage['init']]
            print('Starting '+stage['name'],flush=True)
            with (root/stage['name']/'console.log').open('w') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                status['training_pid']=process.pid
                atomic_json(status,root/'status.json')
                result=process.wait()
            if result:
                atomic_json({**status,'stage':'failed','returncode':result,'updated_at':time.time()},
                            root/'status.json')
                raise SystemExit(result)
            completed.append(stage['name'])
            print('Completed '+stage['name'],flush=True)
        atomic_json(dict(stage='complete',completed=completed,updated_at=time.time(),
                         revision=args.revision),root/'status.json')


if __name__=='__main__':
    main()
