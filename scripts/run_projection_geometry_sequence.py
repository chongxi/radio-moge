"""One authorized filtered B arm: finish existing CPU work, gate, then run once.

No retries, budget extension, altered backend, or replacement of a failed proof.
All model/runtime subprocesses execute the separately frozen source snapshot.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(root, execution, revision, lock, cpu_pid):
    from radio_moge.corpus_admission import checked_json, file_ref
    from radio_moge.geometry_cache import publish_json
    from radio_moge.ordinary_training import CPU_REQUIREMENTS, implementation_digest
    control = root/'projection-filtered-geometry-sequence-01'
    control.mkdir(exist_ok=False)
    status = dict(stage='waiting_existing_cpu_verifier', started_at=time.time(),
                  existing_cpu_pid=cpu_pid, automatic_retry=False, user_authorized=True)

    def progress(stage, **fields):
        from radio_moge.checkpoint import atomic_json
        status.update(stage=stage, updated_at=time.time(), **fields)
        atomic_json(status, control/'status.json')
        print(json.dumps(status), flush=True)

    def command(stage, args):
        progress(stage)
        with (control/(stage+'.log')).open('xb') as log:
            subprocess.run([sys.executable, *args], cwd=execution, env=os.environ,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        progress('waiting_existing_cpu_verifier')
        cpu = root/'projection-filtered-geometry-cpu-01'
        while True:
            current = checked_json(file_ref(cpu/'status.json'))
            if current['stage'] == 'bounded_cpu_checks_complete_not_training_admitted':
                break
            if 'fail' in current['stage'] or not Path(f'/proc/{cpu_pid}').exists():
                raise RuntimeError('Existing CPU verifier did not complete; no replacement launched')
            time.sleep(10)
        prepared = root/'projection-filtered-geometry-gate-inputs-01'
        inputs = checked_json(file_ref(prepared/'preparation.json'))
        binding = checked_json(inputs['binding'])
        if binding['implementation_sha256'] != implementation_digest():
            raise ValueError('Immutable execution differs from the prepared binding')
        sidecar_ref, target_ref = file_ref(cpu/'sidecar-report.json'), file_ref(cpu/'target-report.json')
        sidecar, target = checked_json(sidecar_ref), checked_json(target_ref)
        if (sidecar['stage'] != 'ordinary_geometry_sidecar_cpu_verified_not_admitted'
                or target['stage'] != 'ordinary_target_camera_cpu_verified_not_admitted'
                or target['input_proof'] != sidecar_ref or target['presentations'] != 128
                or sidecar['manifest_rows'] != 1597911 or sidecar['optimizer_updates'] != 0
                or target['optimizer_updates'] != 0 or target['training_admission'] is not False
                or not sidecar['prepared_rgb_bitwise_equal'] or not target['camera_rays_verified']
                or not target['geometry_member_hashes_verified'] or not target['torch_rng_unchanged']):
            raise ValueError('New actual CPU scope/target proof is incomplete')
        base = dict(config=inputs['config'], performance=inputs['performance'], binding=binding,
            observer=file_ref(execution/'scripts/probe_ordinary_geometry_model.py'),
            target_proof=target_ref, training_admission=False, optimizer_updates=0)
        diagnostic = root/'projection-filtered-geometry-diagnostic-request-01'
        diagnostic.mkdir(exist_ok=False)
        publish_json(diagnostic/'request.json', base)
        reports = {}
        for mode in ('current', 'sdpa_math'):
            destination = root/('projection-filtered-repeatability-'+mode+'-01')
            command('diagnostic_'+mode, ['scripts/diagnose_ordinary_feature_reproducibility.py',
                '--gate-dir', str(diagnostic), '--output', str(destination), '--lock', str(lock), '--mode', mode])
            reports[mode] = file_ref(destination/'report.json')
        gate = root/'projection-filtered-geometry-model-gate-01'
        gate.mkdir(exist_ok=False)
        request = deepcopy(base)
        request['repeatability_evidence'] = reports
        publish_json(gate/'request.json', request)
        # Reuse completed production-loop fixtures; the new projection guards
        # and sidecar were tested on the exact immutable snapshot below. Actual
        # data-specific pixel/camera evidence above is fresh for this pair.
        prior_ref = file_ref(root/'ordinary-trainer-integration-01/report.json')
        prior = checked_json(prior_ref)
        if any(prior.get(k) is not True for k in CPU_REQUIREMENTS):
            raise ValueError('Earlier production-loop integration evidence is incomplete')
        tested = ('test_projection_fork.py', 'test_projection_sidecar.py',
                  'test_ordinary_geometry_data.py', 'test_ordinary_training.py',
                  'test_ordinary_geometry_runner.py', 'test_ordinary_model_probe.py')
        evidence = dict(execution_revision=revision, passed=75, tests_repeated=False,
            recorded_command_id='exec-5ff8bd96-9502-4bad-9959-aedc596fffe8',
            recorded_result_id='c588013f-2da9-4aa1-b0f8-5b1045e436d3',
            observation_handoff=file_ref(root/'handoff-20261104T081838Z.json'),
            test_sources={n:file_ref(execution/'tests'/n) for n in tested},
            note='Prior observed immutable-snapshot test run; not new candidate-specific optimizer execution.')
        publish_json(gate/'reused-test-evidence.json', evidence)
        publish_json(gate/'cpu-report.json', dict(stage='ordinary_production_loop_cpu_verified',
            binding=binding, **{k: True for k in CPU_REQUIREMENTS},
            prior_production_loop=prior_ref, projection_tests=file_ref(gate/'reused-test-evidence.json'),
            actual_sidecar=sidecar_ref, actual_targets=target_ref,
            production_optimizer_updates=0, training_admission=False,
            scope='Reused tiny production-loop fixtures and immutable projection tests plus new actual128 RGB/target verification. Actual CUDA is independently required.'))
        command('actual_cuda_model_gate', ['scripts/probe_ordinary_geometry_model.py',
                '--request', str(gate/'request.json'), '--lock', str(lock)])
        command('prepare_training', ['scripts/run_ordinary_geometry_training.py', 'prepare',
                '--gate-dir', str(gate), '--execution-revision', revision])
        config = checked_json(inputs['config'])
        training_root = Path(config['output_dir']).parent
        command('actual_training', ['scripts/run_ordinary_geometry_training.py', 'run',
                '--request', str(training_root/'run-request.json'), '--lock', str(lock)])
        progress('completed_fixed250_pending_saved_result_review',
                 completion=file_ref(training_root/'completed.json'))
    except BaseException as error:
        progress('stopped_no_automatic_retry', error=type(error).__name__, message=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'execution', 'lock'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--cpu-pid', type=int, required=True)
    args = parser.parse_args()
    main(args.root, args.execution, args.revision, args.lock, args.cpu_pid)
