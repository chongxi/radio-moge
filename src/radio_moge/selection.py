"""Independent validation checkpoint selection; never changes optimization."""

import math
import re


def validate_selections(selections):
    if len(selections) > 8:
        raise ValueError('At most eight checkpoint metrics')
    for name, spec in selections.items():
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}', name):
            raise ValueError('Invalid checkpoint metric name')
        if spec['mode'] not in ('min', 'max') or not isinstance(spec['metric'], str):
            raise ValueError('Checkpoint metric requires min/max and a metric key')
        if 'baseline' in spec and not math.isfinite(spec['baseline']):
            raise ValueError('Checkpoint metric baseline must be finite')


def selection_winners(selections, best, metrics, step):
    winners = []
    if metrics.get('postprocess_failed', 1) != 0:
        return winners
    for name, spec in selections.items():
        if spec['metric'].startswith('resolution/'):
            prefix = '/'.join(spec['metric'].split('/')[:2])
            if metrics.get(prefix + '/postprocess_failed', 1) != 0:
                continue
        value = metrics.get(spec['metric'])
        if value is None or not math.isfinite(value):
            continue
        previous = best.get(name, {}).get('value')
        if previous is None or (value < previous if spec['mode'] == 'min' else value > previous):
            best[name] = dict(value=value, step=step, metric=spec['metric'], mode=spec['mode'])
            winners.append(name)
    return winners
