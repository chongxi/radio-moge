"""Verify closed shards from a failed attempt into a separate, limited-scope cache."""
import argparse
import json
from pathlib import Path

from radio_moge.cache_recovery import recover_closed_packs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--original', required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    print(json.dumps(recover_closed_packs(json.loads(args.config.read_text()),
                                          args.original, args.attempt)), flush=True)
