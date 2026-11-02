from copy import deepcopy

from scripts.run_loss_weight_followup import select_candidate


def fixture():
    return {'arms': {
        'custom-f01': dict(spatial_cos=.95, summary_cos=.92, nyu_absrel=.10),
        'custom-f1': dict(spatial_cos=.96, summary_cos=.93, nyu_absrel=.09),
        'official-f01': dict(spatial_cos=.96, summary_cos=.93, nyu_absrel=.09),
        'official-f1': dict(spatial_cos=.965, summary_cos=.935, nyu_absrel=.08),
    }}


def test_choose_one_improving_candidate_without_test_metrics():
    report = fixture()
    assert select_candidate(report) == 'official-f1'
    report['arms']['official-f1']['nyu_absrel'] = .10
    assert select_candidate(report) == 'custom-f1'


def test_no_followup_when_cosine_or_cross_domain_depth_does_not_improve():
    report = fixture()
    for family in ('custom', 'official'):
        report['arms'][family + '-f1'] = deepcopy(report['arms'][family + '-f01'])
    assert select_candidate(report) is None
