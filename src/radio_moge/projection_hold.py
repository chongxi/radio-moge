"""Forward training hold for a source with unreviewed camera projection.

Historical artifacts/evaluation remain readable. This is a prohibition, not
projection certification of other sources or admission of a filtered index.
"""
from .corpus_data import CorpusRecords


def require_unheld_sources(records, role):
    if isinstance(records, CorpusRecords):
        # Inspect sample membership, not retained excluded/held provenance rows.
        found = records.db.execute(
            'SELECT 1 FROM samples s JOIN images i ON s.ordinal=i.ordinal '
            "WHERE i.dataset='coco2017' LIMIT 1").fetchone() is not None
    else:
        found = any(row.get('dataset') == 'coco2017' for row in records)
    if found:
        raise ValueError(
            f'COCO projection hold: {role} contains unreviewed coco2017 inputs. '
            'Use a separately verified filtered scope; historical seals do not clear this hold.')
