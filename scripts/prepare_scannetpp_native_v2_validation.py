"""Freeze native v2 validation identities before viewing new-policy results.

Metadata only. No RGB/mesh/pack payload read, model inference or source admission.
Controls and confirmation use distinct training scenes outside known flags.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import publish_json
from radio_moge.mesh_clip import MeshClipContract


SEED = 'scannet-native-v2-validation-20261104-v1'


def rank(text):
    return hashlib.sha256((SEED+'\0'+text).encode()).hexdigest()


def select(db, quarantines):
    if len(quarantines) != 19 or len(set(quarantines)) != 19:
        raise ValueError('Exactly nineteen unique prior quarantine identities required')
    query = ('SELECT ordinal,scene_group,row_json,outcome,policy,evidence_json,entry_key,details_json '
             'FROM images WHERE ordinal=?')
    def record(ordinal, role):
        item = db.execute(query, (ordinal,)).fetchone()
        if item is None or item[3] != 'native_candidate':
            raise ValueError('Selected identity is not an audited native candidate')
        ordinal, group, encoded, _, policy, evidence, key, details = item
        row = json.loads(encoded)
        if (row['dataset'] != 'scannetpp_dslr' or row['split'] != 'train'
                or row['official_split'] != 'nvs_sem_train' or row['group'] != group
                or group != 'scannetpp:'+row['scene']):
            raise ValueError('Selected row is not an exact training-scene perspective DSLR')
        return dict(corpus_ordinal=ordinal, role=role, group=group, image=row,
                    old_policy_sha256=policy, old_entry_key=key,
                    old_receipt=json.loads(evidence), old_metrics=json.loads(details))
    result = [record(i, 'known_quarantine') for i in sorted(quarantines)]
    excluded = {r['group'] for r in result}
    groups = {g: n for g, n in db.execute('SELECT scene_group,COUNT(*) FROM images '
        "WHERE outcome='native_candidate' GROUP BY scene_group") if g not in excluded}
    selected_groups = sorted(groups, key=lambda g: (rank('scene:'+g), g))[:32]
    if len(selected_groups) != 32:
        raise ValueError('Need thirty-two disjoint nonquarantined training scenes')
    for number, group in enumerate(selected_groups):
        offset = int(rank('frame:'+group), 16) % groups[group]
        row = db.execute("SELECT ordinal FROM images WHERE outcome='native_candidate' "
                         'AND scene_group=? ORDER BY ordinal LIMIT 1 OFFSET ?',
                         (group, offset)).fetchone()
        result.append(record(row[0], 'ordinary_control' if number < 16 else 'confirmation'))
    if len({r['corpus_ordinal'] for r in result}) != 51:
        raise ValueError('Frozen validation identities must be unique')
    return result


def run(audit_path, review_path, output):
    audit_ref, review_ref = file_ref(audit_path), file_ref(review_path)
    audit, review = checked_json(audit_ref), checked_json(review_ref)
    if (audit['stage'] != 'native_outcomes_accounted_not_accepted'
            or audit['expected_scope'] != 841431 or audit['training_admission'] is not False
            or audit['producer_array_readback_complete'] is not True
            or review['quarantined_candidates'] != 19):
        raise ValueError('Completed full native accounting and current nineteen-case review required')
    index = checked_path(audit['index'])
    with sqlite3.connect(f'file:{index}?mode=ro', uri=True) as db:
        rows = select(db, review['quarantined_candidate_ordinals'])
    output.mkdir(exist_ok=False)
    publish_json(output/'selection.json', rows)
    report = dict(stage='v2_validation_identities_frozen_not_rendered', audit=audit_ref,
        review=review_ref, index=audit['index'], seed=SEED, selection=file_ref(output/'selection.json'),
        clip_contract=MeshClipContract().as_dict(), selection_count=51,
        counts={'known_quarantine': 19, 'ordinary_control': 16, 'confirmation': 16},
        scene_counts={role: len({r['group'] for r in rows if r['role'] == role})
                      for role in ('known_quarantine', 'ordinary_control', 'confirmation')},
        selection_basis='hash-ranked scenes then hash-selected ordinal offset; no new-render results',
        confirmation_execution_allowed=False, physical_uncertainty_budget_frozen=False,
        rgb_or_mesh_payload_read=False, gpu_used=False, optimizer_updates=0,
        training_admission=False, source_flags_cleared=False, implementation=file_ref(Path(__file__)))
    publish_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.audit, args.review, args.output)
    print(json.dumps({k: result[k] for k in ('stage', 'selection_count', 'scene_counts')}))
