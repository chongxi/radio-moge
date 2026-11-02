"""Bounded raw-NYU reproduction audit: six TRAIN pairs from three office scenes.

Fetch only exact MAT-referenced raw files, not full archives. Compare literal
toolbox output with published images/rawDepths. No training labels or corrections
are published, and no held-out frame values are opened.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import time

import h5py
import numpy as np
from PIL import Image

from indoor_download import RemoteZip, write_json
from probe_native_depth import Evidence, quantiles
from radio_moge.geometry_cache import sha_file
from radio_moge.nyu_projection import reproduce_projected_depth, reproduce_undistorted_rgb


def audit(config, attempt):
    root = Path(config['root'])
    evidence = Evidence(root / attempt)
    result = dict(stage='probing', started_at=time.time(), training_admission=False,
        optimizer_updates=0, manifest_sha256=config['manifest_sha256'], records=[],
        purpose='Reproduce published toolbox outputs, NOT approve physical calibration or metric training')
    write_json(evidence.root / 'report.json', result)
    try:
        with sqlite3.connect(f'file:{root / "inventory.sqlite"}?mode=ro', uri=True) as db:
            groups = [r[0] for r in db.execute("SELECT DISTINCT scene_group FROM images "
                "WHERE dataset='nyuv2' AND scene_group LIKE 'nyuv2:office_%' ORDER BY 1 LIMIT 3")]
            rows = [json.loads(r[0]) for group in groups for r in db.execute(
                'SELECT row_json FROM images WHERE scene_group=? ORDER BY ordinal LIMIT 2', (group,))]
        if len(groups) != 3 or len(rows) != 6 or any(r['split'] != 'train' for r in rows):
            raise ValueError('Expected six frozen training pairs across three scenes')
        source = root / 'native-source/nyuv2/nyu_depth_v2_labeled.mat'
        identity = json.loads((root / 'nyuv2-source-verification.json').read_text())
        if (identity['stage'] != 'source_identity_verified'
                or identity['manifest_sha256'] != config['manifest_sha256']
                or sha_file(source) != identity['source_sha256']):
            raise ValueError('Verified original MAT source required')
        result['toolbox_sha256'] = sha_file(root / 'native-probe-01/nyuv2/toolbox.zip')
        archives = []
        with h5py.File(source, 'r') as mat:
            for row in rows:
                index = int(row['id'].split('_')[-1]) - 1
                native = {}
                for kind, key in (('rgb', 'rawRgbFilenames'), ('depth', 'rawDepthFilenames')):
                    name = ''.join(chr(int(x)) for x in mat[mat[key][0, index]][:].ravel())
                    if row['group'] != 'nyuv2:' + name.split('/')[0]:
                        raise ValueError('Raw frame path does not belong to the frozen training scene')
                    found = None
                    for part in range(2):
                        if len(archives) <= part:
                            url = ('https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/'
                                   f'offices_part{part + 1}.zip')
                            archive = RemoteZip(url, evidence.session)
                            archives.append((url, archive))
                        url, archive = archives[part]
                        matches = [n for n in archive.names() if n == name or n.endswith('/' + name)]
                        if len(matches) > 1:
                            raise ValueError('Ambiguous publisher raw-frame member')
                        if matches:
                            member = matches[0]
                            contents = archive.read(member)
                            native[kind] = evidence.save(row['id'] + '/' + Path(name).name,
                                                          contents, url, member)
                            found = member
                            break
                    if not found:
                        raise ValueError('Exact MAT-referenced raw frame absent from publisher office archives')
                with Image.open(native['rgb']) as image:
                    raw_rgb = np.array(image)
                with Image.open(native['depth']) as image:
                    # Publisher explicitly calls swapbytes after imread(PGM).
                    raw_depth = np.asarray(image, dtype=np.uint16).byteswap()
                projected = reproduce_projected_depth(raw_depth)
                rgb = reproduce_undistorted_rgb(raw_rgb)
                expected_rgb = np.asarray(mat['images'][index]).transpose(2, 1, 0)
                expected = np.asarray(mat['rawDepths'][index]).T
                support = (projected > 0) & (expected > 0)
                difference = np.abs(projected - expected)
                record = dict(id=row['id'], scene=row['group'],
                    raw_sensor_codes=quantiles(raw_depth), rgb_exact_fraction=float((rgb == expected_rgb).mean()),
                    rgb_mae_u8=float(np.abs(rgb.astype(float) - expected_rgb).mean()),
                    depth_support_agreement=float(((projected > 0) == (expected > 0)).mean()),
                    depth_common_error_m=quantiles(difference[support]),
                    depth_all_error_m=quantiles(difference), expected_raw_support_fraction=float((expected > 0).mean()),
                    native_depth_output_is_sensor_z_not_rgb_z=True, calibration_acceptance='PENDING')
                result['records'].append(record)
                write_json(evidence.root / 'report.json', result)
                print(json.dumps(record), flush=True)
        result.update(stage='diagnostic_complete', completed_at=time.time())
        write_json(evidence.root / 'report.json', result)
    except BaseException as error:
        result.update(stage='failed', error=type(error).__name__, message=str(error)[:1000])
        write_json(evidence.root / 'report.json', result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt:
        parser.error('Use a new evidence attempt basename')
    audit(json.loads(args.config.read_text()), args.attempt)
