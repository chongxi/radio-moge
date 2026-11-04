"""Materialize forward-only COCO projection holds; never publish a training seal.

Preserve historical manifests, labels and decisions. Pending individual camera
review, omit the entire unverified source from new RGB/replay manifests and from
the selected geometry sample table. Other source records remain byte-identical.
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


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def publish(path, value):
    pending = path.with_suffix(path.suffix + '.incoming')
    with pending.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    os.rename(pending, path)


def filter_rgb(source, expected, output):
    counts, held, digest = Counter(), Counter(), hashlib.sha256()
    pending = output.with_suffix('.jsonl.incoming')
    held_path = output.with_name(output.stem + '-held.jsonl')
    held_pending = held_path.with_suffix('.jsonl.incoming')
    with Path(source).open('rb') as src, pending.open('xb') as dst, held_pending.open('xb') as omit:
        for raw in src:
            digest.update(raw)
            row = json.loads(raw)
            if row['dataset'] == 'coco2017':
                omit.write(raw)
                held[row['dataset']] += 1
            else:
                dst.write(raw)
                counts[row['dataset']] += 1
    if digest.hexdigest() != expected or held['coco2017'] != 39997:
        raise ValueError('Frozen RGB identity or held-source scope differs')
    os.rename(pending, output)
    os.rename(held_pending, held_path)
    return dict(input=dict(path=str(source), sha256=expected), output=ref(output),
                held=ref(held_path), retained_counts=dict(counts), held_counts=dict(held),
                retained_rows=sum(counts.values()), held_rows=sum(held.values()),
                retained_row_bytes_unchanged=True)


def filter_geometry(source, expected, output, review):
    source = Path(source)
    if any(Path(str(source) + suffix).exists() for suffix in ('-wal', '-shm', '-journal')):
        raise ValueError('Only a closed immutable source index may be copied')
    with tempfile.TemporaryDirectory(prefix='projection-hold-') as tmp:
        local = Path(tmp) / 'geometry.sqlite'
        digest = hashlib.sha256()
        before = source.stat()
        with source.open('rb') as src, local.open('xb') as dst:
            while chunk := src.read(4 * 1024 * 1024):
                digest.update(chunk)
                dst.write(chunk)
        after = source.stat()
        if (digest.hexdigest() != expected
                or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
            raise ValueError('Frozen geometry index identity changed')
        with sqlite3.connect(local) as db:
            old = dict(db.execute("SELECT dataset,COUNT(*) FROM images WHERE disposition='selected' GROUP BY dataset"))
            if sum(old.values()) != 598298 or old.get('coco2017') != 39984:
                raise ValueError('Historical selected geometry scope differs')
            db.execute('CREATE TABLE projection_holds AS SELECT ordinal,image,sha256,disposition AS prior_disposition '
                       "FROM images WHERE dataset='coco2017'")
            if db.execute('SELECT COUNT(*) FROM projection_holds').fetchone()[0] != 39997:
                raise ValueError('Geometry and RGB held-source scopes differ')
            db.execute('CREATE TABLE projection_hold_review (review_json TEXT NOT NULL)')
            db.execute('INSERT INTO projection_hold_review VALUES (?)', (json.dumps(review, sort_keys=True),))
            db.execute("UPDATE images SET disposition='projection_hold' WHERE dataset='coco2017' AND disposition='selected'")
            db.execute('DELETE FROM samples')
            db.execute('INSERT INTO samples SELECT ROW_NUMBER() OVER (ORDER BY ordinal)-1,ordinal '
                       "FROM images WHERE disposition='selected' ORDER BY ordinal")
            new = dict(db.execute("SELECT dataset,COUNT(*) FROM images WHERE disposition='selected' GROUP BY dataset"))
            if new != {k: v for k, v in old.items() if k != 'coco2017'}:
                raise ValueError('Other source selection changed')
            if db.execute('SELECT COUNT(*) FROM samples').fetchone()[0] != sum(new.values()):
                raise ValueError('Sample table differs from retained geometry')
            if db.execute("SELECT COUNT(*) FROM samples s JOIN images i ON s.ordinal=i.ordinal WHERE i.dataset='coco2017'").fetchone()[0]:
                raise ValueError('Held source remains sampleable')
            if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                raise ValueError('Filtered index failed SQLite readback')
        pending = output.with_suffix('.sqlite.incoming')
        shutil.copyfile(local, pending)
        if sha(local) != sha(pending):
            raise ValueError('Filtered index copy differs')
        os.rename(pending, output)
    return dict(input=dict(path=str(source), sha256=expected), output=ref(output),
                prior_selected_counts=old, selected_counts=new, selected_images=sum(new.values()),
                removed_selected_images=old['coco2017'], held_source_rows=39997,
                prior_exclusions_and_candidate_evidence_preserved=True,
                other_source_selection_unchanged=True, training_admission=False)


def run(root, review_path, output):
    review = ref(review_path)
    decision = json.loads(Path(review_path).read_text())
    if decision.get('forward_hold', {}).get('source') != 'coco2017':
        raise ValueError('Explicit COCO forward projection hold required')
    output.mkdir(exist_ok=False)
    rgb = filter_rgb(root / 'rgb-replay-candidates-01/candidates-rgb.jsonl',
        '4de531643528b46f72fc8f1317a5c20af5aadf234d50946ad063aaba26f25f27', output / 'rgb.jsonl')
    replay = filter_rgb(root / 'reviewed-stage1-replay-02/rgb.jsonl',
        '1ff48734105a0ca14532f2a511916d7db9cc057ece37d2d74210bc860ea81aaa', output / 'legacy-replay.jsonl')
    geometry = filter_geometry(root / 'reviewed-stage4-data-01/accounting/accounting.sqlite',
        'c5ec2e5cf58f078519be7ccee54bc804baf1abce877e269f49bb63b146870b40', output / 'geometry-candidates.sqlite', review)
    if ref(review_path) != review:
        raise ValueError('Projection review changed while materializing hold')
    report = dict(stage='projection_hold_materialized_not_training_accepted', review=review,
        rgb=rgb, legacy_replay=replay, geometry=geometry, training_admission=False,
        optimizer_updates=0, historical_inputs_modified=False, implementation=ref(__file__),
        remaining_sources_pixel_projection_fully_certified=False,
        future_requirements=['new config and admission bound to filtered inputs',
            'projection-screen future validation without silently changing historical scores',
            'individually verify COCO projection before any re-admission'])
    publish(output / 'report.json', report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.review, args.output)
