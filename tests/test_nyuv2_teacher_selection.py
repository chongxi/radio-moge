from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from cache_nyuv2_teacher_geometry import select_preflight
sys.path.pop(0)


def test_nyu_preflight_preserves_scene_types_and_raw_projection_regressions():
    rows = []
    for i in range(16):
        rows.append(dict(id=f'nyuv2_{i + 1:04d}', sha256=f'{i:064x}', corpus_ordinal=i,
                         scene_type='office' if i < 8 else 'kitchen', group=f'scene:{i // 2}', split='train'))
    result = select_preflight(rows)
    assert {r['scene_type'] for r in result} == {'office', 'kitchen'}
    assert {'nyuv2_0005', 'nyuv2_0006', 'nyuv2_0010', 'nyuv2_0011'} <= {r['id'] for r in result}
    assert len(result) == len({r['id'] for r in result})
    assert result == select_preflight(list(reversed(rows)))
    for kind in ('office', 'kitchen'):
        assert len({r['group'] for r in result if r['scene_type'] == kind}) >= 2
