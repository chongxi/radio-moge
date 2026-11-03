"""Explicit recovery into a new attempt; never alter a failed cache or admit it."""
import json
import os
from pathlib import Path
import sqlite3
import time

from .geometry_cache import canonical_json, publish_json, read_geometry_member, sha_file


def recovered_rows(root, name, dataset, manifest):
    if name is None:
        return set(), None
    if Path(name).name != name:
        raise ValueError('Recovery must be a new attempt basename')
    path = Path(root) / name / 'report.json'
    report = json.loads(path.read_text())
    if (report.get('stage') != 'complete' or report.get('recovered_verified_subset') is not True
            or report.get('dataset') != dataset or report.get('manifest_sha256') != manifest
            or report.get('training_admission') is not False or report.get('optimizer_updates') != 0):
        raise ValueError('A completed, reread, explicitly recovered subset is required')
    scope = path.parent / 'scope.jsonl'
    if sha_file(scope) != report['scope_sha256']:
        raise ValueError('Recovered subset changed')
    rows = [json.loads(line) for line in scope.read_text().splitlines()]
    ids = {r['sha256'] for r in rows}
    if len(ids) != len(rows) or len(ids) != report['verified_cache_members']:
        raise ValueError('Recovered subset is duplicated or incomplete')
    return ids, dict(path=str(path.resolve()), sha256=sha_file(path))


def remaining_scope(rows, recovered):
    """Keep original order and reject recovery from outside this exact scope."""
    identities = [row['sha256'] for row in rows]
    if len(set(identities)) != len(rows) or not recovered <= set(identities):
        raise ValueError('Recovered identities must be unique members of the selected scope')
    return [row for row in rows if row['sha256'] not in recovered]


def validate_teacher_recovery_policy(previous, current):
    code_only = {'generator_sha256', 'recovery_helper_sha256'}
    if (previous.get('label_kind') != 'teacher_pseudo' or current.get('label_kind') != 'teacher_pseudo'
            or {k: v for k, v in previous.items() if k not in code_only}
            != {k: v for k, v in current.items() if k not in code_only}):
        raise ValueError('Recovered teacher inference, masks or source policy differs')


def recover_closed_packs(config, original_name, attempt):
    """Reread complete tar/receipt pairs; abandon uncommitted tails as evidence.

    The new report covers ONLY these verified entries. The rest of the frozen
    source, including previous rejections, must be processed in a separate run.
    Old and new conversion policies remain distinct in final corpus accounting.
    """
    root = Path(config['root'])
    if any(Path(n).name != n for n in (original_name, attempt)):
        raise ValueError('Use attempt basenames')
    original = root / original_name
    state = json.loads((original / 'status.json').read_text())
    if state['stage'] != 'failed' or state.get('optimizer_updates') != 0:
        raise ValueError('Only an inactive failed generation can be recovered')
    policy = json.loads((original / 'packs/policy.json').read_text())
    import hashlib
    policy_hash = hashlib.sha256(canonical_json(policy)).hexdigest()
    inventory = json.loads((root / 'inventory.json').read_text())
    if (state['policy_sha256'] != policy_hash
            or policy['manifest_sha256'] != config['manifest_sha256']
            or inventory['manifest_sha256'] != config['manifest_sha256']
            or sha_file(root / 'inventory.sqlite') != inventory['inventory_sha256']):
        raise ValueError('Original policy or frozen inventory differs')
    output = root / attempt
    output.mkdir(exist_ok=False)
    packs = output / 'packs'
    packs.mkdir()
    status = dict(stage='verifying_closed_packs', mode='generate', dataset=policy['dataset'],
        completed=0, rejected=0, verified_cache_members=0, started_at=time.time(),
        training_admission=False, training_started=False, optimizer_updates=0,
        policy_sha256=policy_hash, manifest_sha256=config['manifest_sha256'],
        recovered_verified_subset=True, original_attempt=original_name,
        original_status_sha256=sha_file(original / 'status.json'))
    # Status updates are atomic, using the same existing small-file helper.
    from indoor_download import write_json
    write_json(output / 'status.json', status)
    items, seen = [], set()
    try:
        with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db, \
                (output / 'scope.jsonl.incoming').open('x') as scope:
            for number, path in enumerate(sorted((original / 'packs').glob('pack-*.json'))):
                if path.name != f'pack-{number:06d}.json':
                    raise ValueError('Closed receipt sequence has a gap')
                receipt = json.loads(path.read_text())
                tar = path.parent / receipt['tar']
                if (tar.parent != path.parent or tar.name != f'pack-{number:06d}.tar'
                        or receipt['policy_sha256'] != policy_hash
                        or tar.stat().st_size != receipt['bytes'] or sha_file(tar) != receipt['sha256']):
                    raise ValueError('Closed pack identity differs')
                for entry in receipt['entries']:
                    _, _, _, meta = read_geometry_member(tar, entry)
                    row = dict(entry['image'])
                    identity = db.execute('SELECT ordinal,image,row_json FROM images WHERE sha256=?',
                                          (row['sha256'],)).fetchall()
                    if len(identity) != 1:
                        raise ValueError('Recovered entry is not a unique frozen RGB')
                    ordinal, image, encoded = identity[0]
                    expected = dict(json.loads(encoded), image=image)
                    if (row.pop('corpus_ordinal', ordinal) != ordinal or row != expected
                            or meta['policy_sha256'] != policy_hash or meta['dataset'] != policy['dataset']
                            or row['dataset'] != policy['dataset'] or row['split'] != 'train'
                            or meta['image_sha256'] != row['sha256'] or row['sha256'] in seen):
                        raise ValueError('Recovered entry differs from its image/policy or is duplicated')
                    seen.add(row['sha256'])
                    scope.write(json.dumps(dict(ordinal=ordinal, sha256=row['sha256'])) + '\n')
                # Keep original payload and provenance bytes. No new physical GT
                # or source-quality acceptance is implied by these links.
                (packs / path.name).symlink_to(path.resolve())
                (packs / tar.name).symlink_to(tar.resolve())
                items.append(dict(receipt=path.name, sha256=sha_file(path), count=len(receipt['entries'])))
                status.update(completed=len(seen), verified_cache_members=len(seen),
                              packs_verified=len(items), updated_at=time.time())
                write_json(output / 'status.json', status)
                if len(items) % 16 == 0:
                    print(json.dumps(status), flush=True)
        if not seen:
            raise ValueError('No closed packs to recover')
        os.replace(output / 'scope.jsonl.incoming', output / 'scope.jsonl')
        (packs / 'policy.json').symlink_to((original / 'packs/policy.json').resolve())
        publish_json(packs / 'packs.json', dict(stage='cache_written_not_accepted',
            training_admission=False, policy_sha256=policy_hash, packs=items, count=len(seen)))
        status.update(stage='complete', expected_scope=len(seen), completed_at=time.time(),
            scope_sha256=sha_file(output / 'scope.jsonl'),
            uncommitted_tails_excluded=[p.name for p in (original / 'packs').glob('*.incoming')])
        write_json(output / 'status.json', status)
        publish_json(output / 'report.json', status)
        return status
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__, message=str(error))
        write_json(output / 'status.json', status)
        raise
