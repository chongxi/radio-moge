"""Replace declared RGB slots while retaining every other original sampler key."""
import hashlib
import json
import random

from .data import DeterministicBatches
from .rgb_input_audit import ReferenceRecords


class ReplacementRecords(ReferenceRecords):
    def __init__(self, real, synthetic):
        super().__init__(real)
        self.extend(synthetic)
        self.real_size, self.synthetic_size = len(real), len(synthetic)
        self.real_digest, self.synthetic_digest = real.digest(), synthetic.digest()
        self.metadata_sha256 = hashlib.sha256(json.dumps(dict(
            real=self.real_digest, synthetic=self.synthetic_digest,
            sizes=[self.real_size, self.synthetic_size]), sort_keys=True).encode()).hexdigest()


def validate_replacement(config):
    spec = config['data'].get('rgb_replacement')
    if spec is None:
        return
    if (not isinstance(spec, dict) or set(spec) != {'manifest', 'every_images', 'real_start_cursor', 'seed'}
            or spec['every_images'] != 8 or type(spec['every_images']) is not int
            or type(spec['real_start_cursor']) is not int or spec['real_start_cursor'] < 0
            or type(spec['seed']) is not int
            or not isinstance(spec['manifest'], dict)
            or set(spec['manifest']) != {'path', 'sha256'}):
        raise ValueError('RGB replacement needs an explicit frozen manifest and one in eight slots')
    if (config['loss'] != dict(spatial=1., summary=1., cosine=1., summary_cosine=1.)
            or any(config.get(k) is not None for k in
                   ('geometry_supervision', 'ordinary_geometry', 'feature_replay', 'corpus_loss_control'))
            or any(config['data'].get(k) for k in ('mixture_weights', 'geometry_corpus', 'coverage_epochs'))):
        raise ValueError('RGB replacement is restricted to the unchanged pure-feature objective')


class ReplacementBatches:
    """Exactly four synthetic slots per 32 images; all other keys stay original.

    Real permutation/crop seeds/resolution draws continue at real_start_cursor.
    Synthetic images use a separate permutation, retaining the replaced slot's
    original crop seed and bucket. Resume depends only on the logical cursor.
    """
    def __init__(self, records, spec, batch_size, buckets, seed, start, count):
        if (not isinstance(records, ReplacementRecords) or records.synthetic_size < 1
                or records.real_size < 1 or batch_size != 4 or spec['every_images'] != 8):
            raise ValueError('RGB replacement requires verified real/synthetic records and logical batch four')
        self.records, self.spec = records, spec
        self.batch_size, self.buckets, self.seed = batch_size, buckets, seed
        self.start, self.count = start, count

    def __len__(self):
        return self.count

    def __iter__(self):
        n = self.records.synthetic_size
        previous_epoch, order = None, None
        real = DeterministicBatches(self.records.real_size, self.batch_size, self.buckets,
            self.seed, self.spec['real_start_cursor'] + self.start, self.count)
        for absolute, batch in enumerate(real, self.start):
            result = []
            for offset, key in enumerate(batch):
                slot = absolute * self.batch_size + offset
                if (slot + 1) % self.spec['every_images'] == 0:
                    epoch, frame = divmod(slot // self.spec['every_images'], n)
                    if epoch != previous_epoch:
                        order = list(range(n))
                        random.Random(self.spec['seed'] + epoch).shuffle(order)
                        previous_epoch = epoch
                    key = (self.records.real_size + order[frame], *key[1:])
                result.append(key)
            yield result
