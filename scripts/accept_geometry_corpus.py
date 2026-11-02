"""Build fully accounted data, or seal already-completed acceptance evidence.

Neither command runs a model or performs an optimizer update. Incomplete jobs,
unresolved rows, unreviewed sources, and analytic-only model probes fail closed.
"""
import argparse
import json
from pathlib import Path

from radio_moge.corpus_admission import build_accounting, seal_acceptance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('account', 'seal'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.mode == 'account':
        result = build_accounting(plan, args.output)
    else:
        result = seal_acceptance(plan['accounting_report'], plan['checks'],
                                 plan['training_config'], args.output)
    print(json.dumps({key: result[key] for key in ('stage', 'training_admission')}), flush=True)
