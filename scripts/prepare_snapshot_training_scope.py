"""Materialize a frozen stage scope without admitting it for training.

The parent RGB manifest remains the identity anchor. Only the captured closed
packs enter this stage. Known exclusions and candidates without an unflagged
alternative are withheld; this does not declare their RGB globally unusable.
No source review, payload check, or model gate is inferred from this routing.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from indoor_download import write_json
from radio_moge.corpus_admission import checked_json, checked_path, file_ref
from radio_moge.geometry_cache import canonical_json, publish_json


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def make_inventory(snapshot_db, target):
    db = sqlite3.connect(target)
    db.execute('PRAGMA cache_size=-32768')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('ATTACH DATABASE ? AS frozen', (str(snapshot_db),))
    db.executescript('''
        CREATE TABLE images(ordinal INTEGER PRIMARY KEY,image TEXT UNIQUE NOT NULL,
            sha256 TEXT UNIQUE NOT NULL,dataset TEXT NOT NULL,scene_group TEXT NOT NULL,
            row_json TEXT, row_sha256 TEXT NOT NULL, unflagged_candidates INTEGER NOT NULL,
            reviewed_candidates INTEGER NOT NULL, explicitly_excluded INTEGER NOT NULL,
            disposition TEXT NOT NULL);
        INSERT INTO images
        SELECT i.ordinal,i.image,i.sha256,i.dataset,i.scene_group,NULL,i.row_sha256,
            SUM(c.native_flag=0 AND c.external_flag=0),
            SUM(c.native_flag=0 AND c.external_flag=0 AND c.reviewed_source=1),
            MAX(c.excluded),'pending'
        FROM frozen.images i JOIN frozen.candidates c ON c.image_sha256=i.sha256
        GROUP BY i.sha256;
        CREATE INDEX by_dataset ON images(dataset);
        CREATE INDEX by_group ON images(scene_group);
        UPDATE images SET disposition=CASE
            WHEN explicitly_excluded=1 THEN 'explicit_geometry_exclusion'
            WHEN unflagged_candidates=0 THEN 'stage_quality_deferral'
            WHEN reviewed_candidates=1 THEN 'source_reviewed_candidate'
            ELSE 'pending_source_review' END;
    ''')
    if db.execute('SELECT COUNT(*) FROM images WHERE unflagged_candidates>1').fetchone()[0]:
        raise ValueError('Ambiguous duplicate unflagged candidates need explicit routing')
    return db


def bind_row(db, entry, source):
    row = dict(entry['image'])
    claimed = row.pop('corpus_ordinal', None)
    found = db.execute('SELECT ordinal,image,dataset,scene_group,row_sha256,row_json '
                       'FROM images WHERE sha256=?', (row['sha256'],)).fetchone()
    meta = entry['geometry_metadata']
    if (found is None or found[1:5] != (row['image'], row['dataset'], row['group'], digest(row))
            or (claimed is not None and claimed != found[0]) or row['split'] != 'train'
            or row['dataset'] != source['dataset'] or meta['dataset'] != row['dataset']
            or meta['policy_sha256'] != source['policy_sha256']
            or meta['image_sha256'] != row['sha256'] or digest(meta) != entry['key']):
        raise ValueError('Stage RGB identity differs from the frozen snapshot')
    if found[5] is None:
        db.execute('UPDATE images SET row_json=? WHERE ordinal=?',
                   (canonical_json(row).decode(), found[0]))


def run(snapshot_path, output):
    snapshot_ref = file_ref(snapshot_path)
    snapshot = checked_json(snapshot_ref)
    if (snapshot['stage'] != 'candidate_snapshot_metadata_complete_not_accepted'
            or snapshot['training_admission'] is not False or snapshot['optimizer_updates'] != 0):
        raise ValueError('A complete frozen candidate snapshot is required')
    capture = checked_json(snapshot['capture'])
    parent_ref = snapshot['parent_inventory_report']
    parent = checked_json(parent_ref)
    manifest = parent['manifest_sha256']
    if parent['stage'] != 'inventory_complete':
        raise ValueError('Parent inventory is incomplete')
    prior_exclusions = {}
    for line in checked_path(snapshot['exclusions']).read_text().splitlines():
        value = json.loads(line)
        if value['decision'] != 'exclude' or value['sha256'] in prior_exclusions:
            raise ValueError('Malformed or duplicate existing exclusion')
        prior_exclusions[value['sha256']] = value
    sources = {s['attempt']: s for s in capture['sources']}
    if len(sources) != len(capture['sources']):
        raise ValueError('Duplicate source attempts')
    output.mkdir(exist_ok=False)
    (output / 'sources').mkdir()
    state = dict(stage='preparing_snapshot_scope', started_at=time.time(),
        processed_packs=0, processed_entries=0, training_admission=False, optimizer_updates=0)
    write_json(output / 'status.json', state)
    try:
        with tempfile.TemporaryDirectory(prefix='snapshot-scope-') as temporary:
            local = Path(temporary) / 'snapshot.sqlite'
            shutil.copyfile(checked_path(snapshot['index']), local)
            if file_ref(local)['sha256'] != snapshot['index']['sha256']:
                raise ValueError('Local snapshot copy differs')
            work = Path(temporary) / 'inventory.sqlite'
            db = make_inventory(local, work)
            plans = {}
            for name, source in sources.items():
                policy = checked_json(source['policy'])
                if (policy['dataset'] != source['dataset'] or policy['manifest_sha256'] != manifest
                        or digest(policy) != source['policy_sha256']):
                    raise ValueError('Captured source policy changed')
                if source.get('source_review'):
                    checked_path(source['source_review'])
                plans[name] = dict(attempt=name, dataset=source['dataset'],
                    policy=source['policy'], policy_sha256=source['policy_sha256'],
                    previous_source_review=source.get('source_review'), packs=[], candidate_images=0,
                    training_admission=False)
            for pack_id, name, encoded_ref, encoded_tar, count in db.execute(
                    'SELECT id,source,receipt_json,tar_json,entries FROM frozen.packs ORDER BY id'):
                ref, tar = json.loads(encoded_ref), json.loads(encoded_tar)
                receipt = checked_json(ref)
                source = sources[name]
                if (receipt['policy_sha256'] != source['policy_sha256']
                        or len(receipt['entries']) != count or receipt['sha256'] != tar['sha256']
                        or receipt['bytes'] != tar['bytes']
                        or Path(tar['path']).stat().st_size != tar['bytes']):
                    raise ValueError('Captured pack receipt changed')
                for entry in receipt['entries']:
                    bind_row(db, entry, source)
                plans[name]['packs'].append(dict(pack_id=pack_id, receipt=ref, tar=tar, count=count))
                plans[name]['candidate_images'] += count
                state['processed_packs'] += 1
                state['processed_entries'] += count
                if state['processed_packs'] % 64 == 0:
                    db.commit()
                    state['updated_at'] = time.time()
                    write_json(output / 'status.json', state)
            if (state['processed_packs'] != snapshot['processed_packs']
                    or state['processed_entries'] != snapshot['processed_entries']
                    or db.execute('SELECT COUNT(*) FROM images WHERE row_json IS NULL').fetchone()[0]
                    or db.execute('SELECT COUNT(*) FROM images').fetchone()[0] != snapshot['unique_rgb']):
                raise ValueError('Stage scope omits frozen images or packs')
            counts = dict(db.execute('SELECT dataset,COUNT(*) FROM images GROUP BY dataset'))
            if counts != snapshot['dataset_counts']:
                raise ValueError('Stage source counts changed')
            dispositions = [dict(dataset=d, disposition=k, images=n) for d, k, n in db.execute(
                'SELECT dataset,disposition,COUNT(*) FROM images GROUP BY 1,2 ORDER BY 1,2')]
            outcomes = Counter()
            excluded_path = output / 'stage-exclusions.jsonl'
            with excluded_path.with_suffix('.jsonl.incoming').open('x') as exclusions:
                for ordinal, image, checksum, dataset, group, kind in db.execute(
                        'SELECT ordinal,image,sha256,dataset,scene_group,disposition '
                        'FROM images ORDER BY ordinal'):
                    outcomes[kind] += 1
                    if kind not in ('explicit_geometry_exclusion', 'stage_quality_deferral'):
                        continue
                    prior = prior_exclusions.get(checksum)
                    if kind == 'explicit_geometry_exclusion' and (prior is None
                            or prior['corpus_ordinal'] != ordinal or prior['image'] != image
                            or prior['dataset'] != dataset or prior.get('group', group) != group):
                        raise ValueError('Explicit stage exclusion changed its original identity')
                    value = dict(corpus_ordinal=ordinal, image=image, sha256=checksum,
                        dataset=dataset, group=group, decision='exclude',
                        reason=('Existing evidenced geometry exclusion' if kind == 'explicit_geometry_exclusion'
                                else 'All captured candidates retain native or external quality flags'),
                        scope='this_frozen_training_stage_only', disposition=kind,
                        rgb_replay_excluded=bool((prior or {}).get('rgb_replay_excluded', False)),
                        evidence=[snapshot_ref, snapshot['index'], snapshot['exclusions']])
                    exclusions.write(canonical_json(value).decode() + '\n')
            # Existing RGB exclusions stay excluded; new quality deferrals
            # apply only to this stage's geometry supervision.
            db.commit()
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Stage inventory integrity failed')
            db.close()
            shutil.copyfile(work, output / 'inventory.sqlite.incoming')
            for ref in (snapshot_ref, snapshot['index'], snapshot['capture'], parent_ref,
                        snapshot['exclusions']):
                checked_path(ref)
            os.rename(output / 'inventory.sqlite.incoming', output / 'inventory.sqlite')
            os.rename(excluded_path.with_suffix('.jsonl.incoming'), excluded_path)
            source_refs = []
            for name, plan in plans.items():
                path = output / 'sources' / f'{name}.json'
                publish_json(path, dict(plan, snapshot=snapshot_ref,
                    payloads_verified=False, stage='snapshot_source_scope_prepared'))
                source_refs.append(file_ref(path))
            publish_json(output / 'scope.json', dict(kind='frozen_candidate_subset',
                snapshot=snapshot_ref, manifest_sha256=manifest, expected_counts=counts,
                parent_inventory_report=parent_ref, stage_only_exclusions=file_ref(excluded_path)))
            publish_json(output / 'inventory.json', dict(stage='inventory_complete',
                scope_kind='frozen_candidate_subset', manifest=parent['manifest'],
                manifest_sha256=manifest, inventory_sha256=file_ref(output / 'inventory.sqlite')['sha256'],
                dataset_counts=counts, images=sum(counts.values()), duplicate_encoded_image_rows=0,
                heldout_checks=parent['heldout_checks'], parent_inventory_report=parent_ref,
                snapshot=snapshot_ref, canonical_heldout_isolation_accepted=False))
            state.update(stage='snapshot_training_scope_prepared_not_accepted',
                completed_at=time.time(), snapshot=snapshot_ref, expected_counts=counts,
                images=sum(counts.values()), dispositions=dispositions, outcome_counts=dict(outcomes),
                proposed_geometry_images=sum(counts.values()) - outcomes['explicit_geometry_exclusion']
                    - outcomes['stage_quality_deferral'],
                sources=source_refs, inventory=file_ref(output / 'inventory.sqlite'),
                inventory_report=file_ref(output / 'inventory.json'), scope=file_ref(output / 'scope.json'),
                exclusions=file_ref(excluded_path), code=file_ref(Path(__file__)),
                payload_verification_complete=False, source_reviews_complete=False,
                canonical_heldout_isolation_accepted=False, model_preflight_complete=False)
            publish_json(output / 'report.json', state)
            write_json(output / 'status.json', state)
            return state
    except BaseException as error:
        state.update(stage='failed', error=type(error).__name__, message=str(error))
        write_json(output / 'status.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.snapshot, args.output)
    print(json.dumps({k: result[k] for k in ('stage', 'images', 'outcome_counts')}), flush=True)
