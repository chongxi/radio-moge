"""Combine selected models' saved predictions without another GPU/test forward pass.

Keeps the original exports immutable. Geometry refinements whose selected weights
equal their initialization reuse that model's predictions, with an explicit audit.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image
import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.compare import fingerprint
from radio_moge.config import ROOT
from radio_moge.metrics import depth_metrics


def read_export(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    geometry = json.loads((path / 'geometry/manifest.json').read_text())
    assert geometry['depth_manifest_sha256'] == hashlib.sha256(
        (path / 'manifest.json').read_bytes()).hexdigest(), 'Geometry provenance mismatch'
    assert geometry['checkpoint_sha256'] == manifest['run']['checkpoint_sha256']
    assert [s['id'] for s in geometry['samples']] == [s['id'] for s in manifest['samples']]
    return manifest, geometry


def planes(path, sample):
    return np.frombuffer(gzip.decompress((path / sample['depth']).read_bytes()),
                         dtype='<f4').reshape(4, sample['height'], sample['width'])


def agreement(depths, mask):
    # Same target-valid mask and metric implementation as internal validation.
    return depth_metrics(torch.from_numpy(depths[1].copy()),
                         torch.from_numpy(depths[2].copy()), torch.from_numpy(mask))


def stage_result(path, label):
    history = [json.loads(line) for line in (path / 'validation.jsonl').read_text().splitlines()]
    best = torch.load(path / 'best.pt', map_location='cpu', weights_only=True, mmap=True)
    chosen = min(history, key=lambda row: row['teacher_depth/abs_rel'])
    assert best['step'] == chosen['step'], 'Selected checkpoint disagrees with validation criterion'
    initial, final = history[0], history[-1]
    return dict(label=label, updates=final['step'], selected_step=best['step'],
                selected_sha256=fingerprint(best['adapter']),
                initial_teacher=initial['teacher_depth/abs_rel'],
                final_teacher=final['teacher_depth/abs_rel'],
                initial_sensor=initial['gt_student/abs_rel'],
                final_sensor=final['gt_student/abs_rel'])


def copy_export(source, destination):
    def copy_file(src, dst):
        # Large immutable arrays/images can share storage; metadata and web files
        # are independent copies so later writes cannot alter an earlier viewer.
        if Path(src).suffix in {'.gz', '.jpg', '.png'}:
            try:
                os.link(src, dst)
                return dst
            except OSError:
                pass
        return shutil.copy2(src, dst)
    shutil.copytree(source, destination, copy_function=copy_file, dirs_exist_ok=True)
    for file in (ROOT / 'web').iterdir():
        if file.is_dir():
            shutil.copytree(file, destination / file.name, dirs_exist_ok=True)
        else:
            shutil.copy2(file, destination / file.name)


def build(baseline, fusion, root, output):
    torch.set_num_threads(1)
    if output.exists():
        raise FileExistsError(f'Use a new comparison directory: {output}')
    sources = [baseline, fusion]
    exports = [read_export(path) for path in sources]
    ids = [[s['id'] for s in manifest['samples']] for manifest, _ in exports]
    assert ids[0] == ids[1], 'Different test frames or order'
    stages = [stage_result(baseline.parent / 'train', 'MLP · features')]
    stages += [stage_result(root / name / 'train', label) for name, label in (
        ('fusion-features', 'Fusion · features'),
        ('mlp-geometry', 'MLP · geometry refinement'),
        ('fusion-geometry', 'Fusion · geometry refinement'))]
    for index, name in ((0, 'mlp'), (1, 'fusion')):
        audit = json.loads((root / f'{name}-selected-model-equivalence.json').read_text())
        assert audit['inference_equivalent'] and audit['selected_step'] == 0
        selected = torch.load(root / f'{name}-geometry/train/best.pt',
                              map_location='cpu', weights_only=True, mmap=True)
        assert fingerprint(selected['adapter']) == exports[index][0]['run']['checkpoint_sha256']
        assert stages[index]['selected_sha256'] == exports[index][0]['run']['checkpoint_sha256']
    scores = [{}, {}]
    max_teacher_difference = 0.0
    for i in range(len(ids[0])):
        loaded = [planes(source, manifest['samples'][i])
                  for source, (manifest, _) in zip(sources, exports)]
        np.testing.assert_allclose(loaded[0][2:], loaded[1][2:], rtol=1e-5, atol=1e-5,
                                   equal_nan=True, err_msg='Teacher or sensor target changed')
        finite = np.isfinite(loaded[0][2]) & np.isfinite(loaded[1][2])
        max_teacher_difference = max(max_teacher_difference,
                                    float(np.abs(loaded[0][2][finite] - loaded[1][2][finite]).max()))
        masks = [np.asarray(Image.open(source / geom['samples'][i]['teacher']['mask'])) > 0
                 for source, (_, geom) in zip(sources, exports)]
        np.testing.assert_array_equal(masks[0], masks[1])
        for j in range(2):
            scores[j][ids[0][i]] = agreement(loaded[j], masks[j])
        if (i + 1) % 50 == 0:
            print(f'Checked saved predictions: {i + 1}/{len(ids[0])}', flush=True)
    model_data = []
    for i, (name, label, run) in enumerate((('mlp', 'MLP · selected feature model', 'e994e59d'),
                                          ('fusion', 'Intermediate fusion · selected feature model', 'c09153e6'))):
        summary = {key: float(np.mean([r[key] for r in scores[i].values()]))
                   for key in ('abs_rel', 'rmse', 'delta1', 'invalid_fraction')}
        model_data.append(dict(id=name, label=label,
                               checkpoint_sha256=exports[i][0]['run']['checkpoint_sha256'],
                               wandb_url=f'https://forge.coreweave.com/wandb/brainsim/radio-moge/runs/{run}',
                               note='Selected feature checkpoint: step 2500. The geometry refinement '
                                    'retained its step-0 initialization under the validation MoGe-depth '
                                    'criterion; its selected predictions are identical to these.',
                               teacher_agreement=dict(samples=scores[i], summary=summary),
                               sensor_summary=exports[i][0]['summary']['after']))
    policy = ('Table: 116 internal validation images, start versus final step. Checkpoints were '
              'selected by the lowest validation MoGe-depth AbsRel before viewing test scores. '
              'Both geometry refinements selected step 0; their selected outputs reuse the feature '
              'models. Their final 1000-step weights are preserved but are not displayed as best. '
              'Geometry stages add 1000 updates to step-2500 models; this is not an equal-step '
              'causal ablation. The interactive viewer uses all 290 outer-test frames.')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    copy_export(fusion, staging)
    copy_export(baseline, staging / 'mlp')
    for current, dest, hrefs in (('fusion', staging, ['mlp/', './']),
                                 ('mlp', staging / 'mlp', ['./', '../'])):
        models = [{**model, 'href': href} for model, href in zip(model_data, hrefs)]
        atomic_json(dict(current_id=current, models=models, stages=stages, policy=policy,
                         teacher_max_difference_m=max_teacher_difference), dest / 'comparison.json')
    os.rename(staging, output)
    print(json.dumps(dict(output=str(output), images=len(ids[0]),
                          models=[{k: v for k, v in model.items() if k != 'teacher_agreement'}
                                  | {'teacher_agreement': model['teacher_agreement']['summary']}
                                  for model in model_data]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--fusion', type=Path, required=True)
    parser.add_argument('--ablation-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.baseline.resolve(), args.fusion.resolve(), args.ablation_root.resolve(),
          args.output.resolve())
