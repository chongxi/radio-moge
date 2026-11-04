"""Narrow, evidence-bound COCO hold for the approved paired continuation.

Only main validation loses COCO. Every retained record, order and runtime file
identity must match the archived validation; other evaluation views stay fixed.
"""
from copy import deepcopy

from .corpus_admission import checked_json, checked_path
from .data import read_manifest, manifest_digest


COMPARISON_SHA = '03994851303ea4af337a495b6534dbf204b347b353d85891d5484b65fb8ece78'


def projection_comparison(ref):
    if ref['sha256'] != COMPARISON_SHA:
        raise ValueError('Unreviewed projection comparison plan')
    return checked_json(ref)


def projection_plan(plan):
    ref = plan.get('projection_comparison')
    if ref is None:
        return None
    common = projection_comparison(ref)
    hold = checked_json(common['hold'])
    heldout = checked_json(common['heldout'])
    if (plan['manifest'] != common['manifest']
            or plan['planned_inputs'] != common['planned_inputs']
            or plan['source_checkpoint'] != common['source_checkpoint']
            or plan['reference_images'] != common['common']['population']
            or plan['manifest'] != hold['rgb']['output']
            or plan['reference_images'] != hold['rgb']['retained_rows']
            or common['validation'] != heldout['new_selection']
            or 'coco2017' in hold['rgb']['retained_counts']):
        raise ValueError('Projection scope differs from the reviewed paired plan')
    return common


def ordinary_projection_scope(binding):
    ref = binding.get('projection_comparison')
    if ref is None:
        return {}, 1637908
    common = projection_comparison(ref)
    source = {k: common['source_checkpoint'][k] for k in ('path', 'sha256')}
    sidecar = checked_json(binding['sidecar'])
    if (binding['source_checkpoint'] != source or binding['manifest'] != common['manifest']
            or sidecar.get('projection_comparison') != ref
            or sidecar.get('index') != common['candidate_geometry_index']
            or sidecar.get('training_admission') is not False):
        raise ValueError('Ordinary filtered scope differs from the paired comparison')
    plan = dict(projection_comparison=ref, manifest=common['manifest'],
        planned_inputs=common['planned_inputs'], source_checkpoint=common['source_checkpoint'],
        reference_images=common['common']['population'])
    projection_plan(plan)
    planned = checked_json(common['planned_inputs'])
    paired_keys = [[r['index'], r['augmentation_seed'], r['height'], r['width']] for r in planned]
    if checked_json(binding['planned_keys']) != paired_keys:
        raise ValueError('Ordinary geometry keys differ from the paired feature arm')
    return plan, plan['reference_images']


def validate_retained_validation(old_rows, new_rows, old_digest, new_digest):
    expected = [r for r in old_rows if r['dataset'] != 'coco2017']
    if not expected or len(expected) == len(old_rows) or new_rows != expected:
        raise ValueError('Validation must retain every non-COCO record in original order')
    if manifest_digest(old_rows) != old_digest or manifest_digest(new_rows) != new_digest:
        raise ValueError('Validation runtime identities differ from bound digests')


def normalize_projection_validation(config, state, digests, plan):
    """Return copies for the existing strict fork check; never mutate artifacts."""
    common = projection_plan(plan)
    normalized, normalized_digests = deepcopy(config), dict(digests)
    if common is None:
        return normalized, normalized_digests
    ref = common['validation']
    if config['data']['val_manifest'] != ref['path']:
        raise ValueError('Projection continuation uses another validation manifest')
    new_rows = read_manifest(checked_path(ref))
    old_path = state['config']['data']['val_manifest']
    old_rows = read_manifest(old_path)
    validate_retained_validation(old_rows, new_rows, state['data_digests']['val'], digests['val'])
    if len(old_rows) != 1268 or len(new_rows) != 1012:
        raise ValueError('Projection validation scope differs')
    normalized['data']['val_manifest'] = old_path
    normalized_digests['val'] = state['data_digests']['val']
    return normalized, normalized_digests
