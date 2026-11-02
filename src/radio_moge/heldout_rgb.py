"""Read-only heldout RGB identity with explicit legacy-manifest contracts.

No evaluation depth is opened. Original splits and hash meanings are preserved;
the training-only RGB reader is deliberately not used by this audit.
"""
import hashlib
import io
from pathlib import Path

from PIL import Image

from .rgb_audit import ARCHIVE_HASH_SOURCES, decoded_hash


MIX_SPLITS = {
    'val-full.jsonl': 'val-full', 'val-selection.jsonl': 'val', 'val.jsonl': 'val',
    'val-irs.jsonl': 'val', 'val-structured3d.jsonl': 'val', 'val-tartanair.jsonl': 'val',
    'test-coco.jsonl': 'test-coco', 'test-nyu.jsonl': 'test-nyu',
    'test-irs.jsonl': 'test', 'test-structured3d.jsonl': 'test', 'test-tartanair.jsonl': 'test',
    'test-irs-official.jsonl': 'official-test',
}
MIX_DATASETS = {'arkitscenes', 'hypersim', 'nyuv2', 'coco2017', 'tartanair', 'structured3d', 'irs'}


def manifest_protocol(path):
    """Names select a reviewed schema; the caller must also bind the file SHA."""
    path = Path(path)
    if path.parent.name == 'attention-full-mix-v1-20261022' and path.name in MIX_SPLITS:
        return dict(schema='encoded_mix', split=MIX_SPLITS[path.name],
                    scene_disjoint_required=path.name != 'test-irs-official.jsonl')
    contracts = {
        ('attention-scannetpp-snapshot-v5-20261101', 'val-scannetpp.jsonl'): 'encoded_dslr',
        ('nyuv2-internal-val-s43', 'val.jsonl'): 'nyu_raw_rgb_hash',
        ('scannetpp-iphone-lidar-val-v1-20261031', 'val-scannetpp-lidar.jsonl'): 'encoded_iphone',
    }
    schema = contracts.get((path.parent.name, path.name))
    if (path.parent.name == 'val' and path.parent.parent.name == 'scannetpp-laser-gt-pilot-v1-20261101'
            and path.name == 'val.jsonl'):
        schema = 'encoded_panocam'
    if schema is None:
        raise ValueError('Unreviewed heldout manifest schema')
    return dict(schema=schema, split=None if schema == 'nyu_raw_rgb_hash' else 'val',
                scene_disjoint_required=True)


def identity(row, manifest, protocol):
    """Resolve only RGB metadata; retain original SHA convention and split."""
    schema = protocol['schema']
    if row.get('split') != protocol['split']:
        raise ValueError('Heldout split differs from the bound manifest protocol')
    if schema == 'nyu_raw_rgb_hash':
        if (row.get('dataset') is not None or row.get('sha256_kind') is not None
                or not row.get('scene') or row.get('group') != row['scene']
                or not row.get('id', '').startswith('nyuv2_') or row.get('image_rotation_cw', 0) != 0):
            raise ValueError('Legacy NYU heldout identity differs')
        dataset, group, convention = 'nyuv2', 'nyuv2:' + row['scene'], 'raw_rgb_bytes'
    else:
        dataset, group, convention = row.get('dataset'), row.get('group'), 'encoded_file_bytes'
        expected = {'encoded_dslr': 'scannetpp_dslr', 'encoded_iphone': 'scannetpp_iphone_lidar',
                    'encoded_panocam': 'scannetpp_panocam'}.get(schema)
        if schema == 'encoded_mix':
            if dataset not in MIX_DATASETS or row.get('sha256_kind') != convention:
                raise ValueError('Frozen mixed heldout identity differs')
            if protocol['split'] == 'official-test' and (
                    dataset != 'irs' or row.get('official_split') != 'test'):
                raise ValueError('IRS official frame-audit protocol differs')
        elif (dataset != expected or row.get('sha256_kind') not in
                ((convention,) if schema == 'encoded_dslr' else (None, convention))):
            raise ValueError('Reviewed ScanNet heldout identity differs')
        if dataset.startswith('scannetpp') and group != 'scannetpp:' + row.get('scene', ''):
            raise ValueError('ScanNet heldout scene identity differs')
    if (not isinstance(group, str) or not group or not isinstance(row.get('sha256'), str)
            or len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256'])):
        raise ValueError('Missing heldout group or checksum')
    return dict(dataset=dataset, group=group,
        image=str((Path(manifest).parent / row['image']).resolve()),
        original_sha256=row['sha256'], original_sha256_kind=convention,
        original_split=row.get('split'), scene_disjoint_required=protocol['scene_disjoint_required'])


def inspect_heldout_rgb(row, manifest, protocol):
    info = identity(row, manifest, protocol)
    data = Path(info['image']).read_bytes()
    encoded = hashlib.sha256(data).hexdigest()
    if info['original_sha256_kind'] == 'encoded_file_bytes' and encoded != row['sha256']:
        raise ValueError('Heldout encoded RGB hash differs')
    with Image.open(io.BytesIO(data)) as file:
        if file.getexif().get(274, 1) != 1:
            raise ValueError('Undocumented heldout EXIF orientation')
        image = file.convert('RGB')
    raw_hash = decoded_hash(image)
    raw_bytes_hash = hashlib.sha256(image.tobytes()).hexdigest()
    if info['original_sha256_kind'] == 'raw_rgb_bytes' and raw_bytes_hash != row['sha256']:
        raise ValueError('Legacy NYU decoded RGB hash differs')
    supplied = row.get('decoded_rgb_sha256')
    if supplied:
        if info['dataset'] == 'nyuv2':
            expected = raw_bytes_hash
        elif info['dataset'] in ARCHIVE_HASH_SOURCES:
            expected = raw_hash
        else:
            raise ValueError('Unreviewed heldout decoded-hash convention')
        if supplied != expected:
            raise ValueError('Heldout stored decoded RGB hash differs')
    rotation = row.get('image_rotation_cw', 0)
    if type(rotation) is not int or rotation not in (0, 90, 180, 270):
        raise ValueError('Undocumented heldout RGB rotation')
    if rotation:
        image = image.transpose({90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
                                 270: Image.Transpose.ROTATE_90}[rotation])
    if ('width' in row or 'height' in row) and image.size != (row.get('width'), row.get('height')):
        raise ValueError('Frozen heldout oriented dimensions differ')
    if protocol['schema'] == 'encoded_iphone':
        alignment = row.get('alignment', {})
        if image.size != (alignment.get('output_width'), alignment.get('output_height')):
            raise ValueError('Published iPhone output RGB dimensions differ')
    info.update(encoded_sha256=encoded, raw_canonical_rgb_sha256=raw_hash,
        canonical_rgb_sha256=decoded_hash(image), width=image.width, height=image.height,
        image_rotation_cw=rotation)
    image.close()
    return info
