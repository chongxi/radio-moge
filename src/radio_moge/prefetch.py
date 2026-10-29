"""Bounded CPU preparation ahead of training, with no sampler or CUDA compute changes."""

from queue import Empty, Full, Queue
from threading import Event, Thread

import torch

from .performance import grouped_window


class FloatLoader:
    """Transfer uint8 RGB through /dev/shm, normalize exactly in the parent process."""

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        return FloatIterator(iter(self.loader))

    def __len__(self):
        return len(self.loader)


class FloatIterator:
    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        return self

    def __next__(self):
        image = next(self.iterator)
        if image.dtype != torch.uint8:
            raise TypeError('Compact RGB transport requires uint8')
        output = torch.empty(image.shape, dtype=torch.float32, pin_memory=image.is_pinned())
        output.copy_(image)
        return output.div_(255)

    def _shutdown_workers(self):
        shutdown = getattr(self.iterator, '_shutdown_workers', None)
        if shutdown is not None:
            shutdown()


class PreparedWindows:
    """Prepare complete, same-shape update windows outside the training thread.

    DataLoader still transfers its original small logical batches through shared
    memory. Only the main process assembles and pins larger tensors. This matters
    on hosts with a small /dev/shm. Prefetch never changes the checkpoint cursor.
    """

    def __init__(self, loader, accumulation, compute_batch, count, ahead=2, pin=False):
        if ahead not in (1, 2) or count < 0:
            raise ValueError('Expected one or two bounded prepared windows')
        self.iterator = iter(loader)  # Start DataLoader workers in the calling thread.
        self.queue = Queue(maxsize=ahead)
        self.stop = Event()
        self.count = count
        self.consumed = 0

        def put(item):
            while not self.stop.is_set():
                try:
                    self.queue.put(item, timeout=.1)
                    return
                except Full:
                    pass

        def prepare():
            try:
                for _ in range(count):
                    if self.stop.is_set():
                        break
                    tensors = list(grouped_window(self.iterator, accumulation, compute_batch))
                    if pin:
                        tensors = [t if t.is_pinned() else t.pin_memory() for t in tensors]
                    put((tensors, None))
            except BaseException as error:
                put((None, error))

        self.thread = Thread(target=prepare, name='prepare-training-windows', daemon=True)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        if self.consumed >= self.count:
            raise StopIteration
        while True:
            try:
                tensors, error = self.queue.get(timeout=.1)
                break
            except Empty:
                if not self.thread.is_alive():
                    raise RuntimeError('Data preparation ended before all windows arrived') from None
        if error is not None:
            self.close()
            raise error
        self.consumed += 1
        return tensors

    def close(self):
        self.stop.set()
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('Data preparation is still waiting; do not start another loader')
        # The producer has exited, so worker shutdown cannot race with next().
        shutdown = getattr(self.iterator, '_shutdown_workers', None)
        if shutdown is not None:
            shutdown()


def windows(loader, accumulation, compute_batch, count, prepared=False, pin=False):
    if prepared:
        return PreparedWindows(loader, accumulation, compute_batch, count, pin=pin)
    iterator = iter(loader)
    return (grouped_window(iterator, accumulation, compute_batch) for _ in range(count))
