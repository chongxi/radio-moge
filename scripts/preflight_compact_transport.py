"""Actual RGB/augmentation equality and shared-memory preflight, without models or GPU."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from threading import Event, Thread
import time

import torch

from radio_moge.checkpoint import atomic_json
from radio_moge.prefetch import PreparedWindows
from radio_moge.performance import grouped_window
from radio_moge.train import make_loader


def run(root):
    output = root / 'preflight-compact-transport.json'
    if output.exists():
        raise FileExistsError('Inspect existing preflight; do not repeat automatically')
    assert not torch.cuda.is_available(), 'Run this preflight with CUDA_VISIBLE_DEVICES empty'
    cfg = json.loads((root / 'config.json').read_text())
    path = Path(cfg['data']['train_manifest'])
    content = path.read_bytes()
    records = []
    # Metadata identities were already checked by training. This diagnostic reads
    # only its selected images; the eventual trainer still performs its full check.
    for line in content.splitlines():
        row = json.loads(line)
        row['image'] = str(path.parent / row['image'])
        records.append(row)
    assert len(records) == 796478
    cursor, updates, accumulation = 62260, 8, 8
    cfg['data'].update(workers=0, compact_transport=False)
    loader = make_loader(records, cfg, cursor, updates * accumulation)
    iterator = iter(loader)
    expected = [[v.clone() for v in grouped_window(iterator, accumulation, 16)]
                for _ in range(updates)]
    stop = Event()
    observed = []

    def sample_shm():
        while not stop.wait(.005):
            observed.append(shutil.disk_usage('/dev/shm').used)

    thread = Thread(target=sample_shm, daemon=True)
    thread.start()
    results = []
    try:
        for prepared in (False, True):
            candidate = deepcopy(cfg)
            candidate['data'].update(workers=4, compact_transport=True)
            loader = make_loader(records, candidate, cursor, updates * accumulation)
            iterator = None
            if prepared:
                stream = PreparedWindows(loader, accumulation, 16, updates)
            else:
                iterator = iter(loader)
                stream = (grouped_window(iterator, accumulation, 16) for _ in range(updates))
            started = time.time()
            try:
                for before, actual in zip(expected, stream, strict=True):
                    actual = list(actual)
                    assert len(before) == len(actual)
                    for a, b in zip(before, actual, strict=True):
                        assert torch.equal(a, b), 'Augmented pixels or batch order changed'
                    # Let the bounded queue fill; exercise real /dev/shm pressure.
                    time.sleep(.05)
            finally:
                stream.close()
            results.append(dict(prepared=prepared, images=updates * 32, exact_pixels=True,
                                elapsed_with_artificial_consumer_delay=time.time() - started))
            del stream, loader, iterator
    finally:
        stop.set()
        thread.join()
    proof = dict(stage='passed', checked_at=time.time(), images_per_pipeline=updates * 32,
                 models_loaded=False, optimizer_updates=0, cuda_used=False,
                 original_float_reference_workers=0, candidate_workers=4,
                 results=results, shared_memory_total_bytes=shutil.disk_usage('/dev/shm').total,
                 observed_shared_memory_max_bytes=max(observed, default=0),
                 shm_sampling_seconds=.005, manifest_sha256=hashlib.sha256(content).hexdigest(),
                 note='Correctness/transport stress check, not GPU throughput measurement')
    atomic_json(proof, output)
    print(json.dumps(proof))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
