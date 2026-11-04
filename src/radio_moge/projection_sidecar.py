"""A data-only derivative of historical geometry acceptance after a source hold.

This certificate never admits a trainer. New camera/gradient/runtime evidence is
still required by the ordinary geometry training gate.
"""
from .corpus_admission import checked_json
from .projection_fork import projection_comparison


STAGE = 'projection_filtered_geometry_sidecar_verified'


def validate_derivation(gate, parent, accounting, hold):
    geometry = hold['geometry']
    if (gate.get('schema_version') != 1 or gate.get('stage') != STAGE
            or gate.get('training_admission') is not False
            or parent.get('schema_version') != 1 or parent.get('stage') != 'accepted_for_training'
            or parent.get('training_admission') is not True
            or accounting.get('stage') != 'accounting_verified_not_accepted'
            or accounting.get('training_admission') is not False or accounting.get('unresolved') != 0
            or any(parent[k] != accounting[k] for k in ('index', 'manifest_sha256', 'selected_images'))
            or geometry['input'] != parent['index'] or geometry['output'] != gate['index']
            or gate['manifest_sha256'] != parent['manifest_sha256']
            or gate['selected_images'] != geometry['selected_images']
            or gate['total_images'] != accounting['total_images']
            or geometry['other_source_selection_unchanged'] is not True
            or geometry['prior_exclusions_and_candidate_evidence_preserved'] is not True
            or geometry['training_admission'] is not False
            or geometry['prior_selected_counts'].get('coco2017') != geometry['removed_selected_images']
            or {k: v for k, v in geometry['prior_selected_counts'].items() if k != 'coco2017'}
                != geometry['selected_counts']
            or sum(geometry['selected_counts'].values()) != gate['selected_images']
            or parent['selected_images'] - geometry['removed_selected_images'] != gate['selected_images']):
        raise ValueError('Projection sidecar is not an exact source-hold derivative')
    return dict(selected=gate['selected_images'], projection_hold=geometry['removed_selected_images'],
                excluded=accounting['total_images'] - parent['selected_images'])


def checked_derivation(gate):
    pair = projection_comparison(gate['projection_comparison'])
    if gate['materialization'] != pair['hold'] or gate['index'] != pair['candidate_geometry_index']:
        raise ValueError('Projection sidecar differs from the pinned paired comparison')
    parent = checked_json(gate['parent_acceptance'])
    accounting = checked_json(parent['accounting_report'])
    hold = checked_json(gate['materialization'])
    return validate_derivation(gate, parent, accounting, hold)
