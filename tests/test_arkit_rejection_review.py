import io
import zipfile

import pytest

from review_arkit_rejections import DirectoryTail, directory_absences, missing_request, selection


def directory_fixture():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('lowres_depth/123_1.000.png', b'native payload must not be retained' * 100)
        archive.writestr('lowres_depth/123_2.000.png', b'other payload')
    payload = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        start = archive.start_dir
    return payload[start:], start, len(payload)


def test_missing_member_proof_replays_zip_directory_without_member_payloads():
    data, start, size = directory_fixture()
    result = directory_absences(data, start, size, ['lowres_depth/123_1.500.png'])
    assert result['member_count'] == 2 and result['absent_members'] == ['lowres_depth/123_1.500.png']
    assert b'native payload' not in data
    with zipfile.ZipFile(DirectoryTail(data, start, size)) as archive:
        with pytest.raises(ValueError, match='outside retained directory'):
            archive.read('lowres_depth/123_1.000.png')


def test_existing_duplicate_and_truncated_absence_claims_are_rejected():
    data, start, size = directory_fixture()
    with pytest.raises(ValueError, match='exists'):
        directory_absences(data, start, size, ['lowres_depth/123_1.000.png'])
    with pytest.raises(ValueError, match='Distinct'):
        directory_absences(data, start, size, ['absent', 'absent'])
    with pytest.raises(ValueError, match='extent'):
        directory_absences(data[:-1], start, size, ['absent'])


def test_missing_timestamp_is_exact_publisher_train_identity():
    row = dict(dataset='arkitscenes',split='train',source_member='vga_wide/123_1.500.png',
               source_url='https://docs-assets.developer.apple.com/test/Training/123/vga_wide.zip')
    item = dict(image=row, reason='missing_exact_timestamp_native_member',
                stats=dict(reason='missing_exact_timestamp_native_member',detail='lowres_depth/123_1.500.png'))
    assert missing_request(item)[1] == 'lowres_depth/123_1.500.png'
    item['stats']['detail'] = 'lowres_depth/123_1.501.png'
    with pytest.raises(ValueError):
        missing_request(item)


def test_visual_selection_covers_every_rotation_and_failure_class_without_duplicates():
    rows = []
    for turn in (0,90,180,270):
        for passed in (False,True):
            for i in range(5):
                rows.append(dict(image=dict(image=f'{turn}-{passed}-{i}',image_rotation_cw=turn),
                    stats=dict(image_registration_pass=passed,valid_fraction=i/100,
                               paired_rgb_mae=.01*i,native_confident_stable_pixels=100*i)))
    selected = selection(rows)
    assert len(selected) == 16
    assert {(x['image']['image_rotation_cw'],x['stats']['image_registration_pass'])
            for x in rows if x['image']['image'] in selected} == {(r,p) for r in (0,90,180,270) for p in (False,True)}
