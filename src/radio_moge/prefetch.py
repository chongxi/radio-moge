"""Bounded CPU preparation and optional CUDA prefetch with unchanged input samples."""

from queue import Empty, Full, Queue
from threading import Event, Semaphore, Thread
import time

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
        return normalize_uint8(image, pin=image.is_pinned())

    def _shutdown_workers(self):
        shutdown = getattr(self.iterator, '_shutdown_workers', None)
        if shutdown is not None:
            shutdown()


def normalize_uint8(image, pin=False):
    if image.dtype != torch.uint8:
        raise TypeError('Compact RGB transport requires uint8')
    output = torch.empty(image.shape, dtype=torch.float32, pin_memory=pin)
    output.copy_(image)
    return output.div_(255)


class PreparedWindows:
    """Prepare complete, same-shape update windows outside the training thread.

    DataLoader still transfers its original small logical batches through shared
    memory. Only the main process assembles and pins larger tensors. This matters
    on hosts with a small /dev/shm. Prefetch never changes the checkpoint cursor.
    """

    def __init__(self, loader, accumulation, compute_batch, count, ahead=2, pin=False,
                 measure=False, compact_grouping=True):
        if ahead not in (1, 2) or count < 0:
            raise ValueError('Expected one or two bounded prepared windows')
        # Group compact logical batches before normalization. This avoids concatenating
        # float32 tensors and then copying them again into page-locked allocations.
        grouped_uint8 = compact_grouping and isinstance(loader, FloatLoader)
        self.iterator = iter(loader.loader if grouped_uint8 else loader)
        self.queue = Queue(maxsize=ahead)
        self.stop = Event()
        self.count = count
        self.consumed = 0
        # Opt-in for bounded diagnostics; long training runs retain no timing list.
        self.preparation_timings = [] if measure else None

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
                    if measure:
                        wall_start, cpu_start = time.perf_counter(), time.thread_time()
                    tensors = list(grouped_window(self.iterator, accumulation, compute_batch))
                    if grouped_uint8:
                        tensors = [normalize_uint8(t, pin=pin) for t in tensors]
                    elif pin:
                        tensors = [t if t.is_pinned() else t.pin_memory() for t in tensors]
                    if measure:
                        self.preparation_timings.append(dict(
                            prepare_wall_seconds=time.perf_counter() - wall_start,
                            prepare_thread_cpu_seconds=time.thread_time() - cpu_start,
                            images=sum(len(t) for t in tensors)))
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


class DeviceWindows:
    """Keep one complete future window on a separate CUDA copy stream.

    A semaphore reserves the queue slot *before* copying, bounding device staging
    to the current window and one future window. CPU decoding remains bounded by
    PreparedWindows. Consumers wait on a CUDA event, not a device-wide barrier.
    """

    def __init__(self, cpu_windows, device, measure=False):
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Device prefetch requires CUDA')
        self.cpu_windows = cpu_windows
        self.preparation_timings = cpu_windows.preparation_timings
        self.count, self.consumed = cpu_windows.count, 0
        self.queue, self.slot, self.stop = Queue(maxsize=1), Semaphore(1), Event()
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.ready_before_consume = [] if measure else None
        self.closed = False

        def put(item):
            while not self.stop.is_set():
                try:
                    self.queue.put(item, timeout=.1)
                    return
                except Full:
                    pass

        def copy():
            try:
                for _ in range(self.count):
                    while not self.stop.is_set():
                        if self.slot.acquire(timeout=.1):
                            break
                    if self.stop.is_set():
                        break
                    source = next(self.cpu_windows)
                    with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
                        tensors = [t.to(self.device, non_blocking=True) for t in source]
                        ready = torch.cuda.Event()
                        ready.record(self.copy_stream)
                    put((tensors, ready, None))
                    del source, tensors
            except BaseException as error:
                put((None, None, error))

        self.thread = Thread(target=copy, name='prefetch-device-windows', daemon=True)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        if self.consumed >= self.count:
            raise StopIteration
        while True:
            try:
                tensors, ready, error = self.queue.get(timeout=.1)
                break
            except Empty:
                if not self.thread.is_alive():
                    raise RuntimeError('Device prefetch stopped before all windows arrived') from None
        self.slot.release()
        if error is not None:
            self.close()
            raise error
        stream = torch.cuda.current_stream(self.device)
        if self.ready_before_consume is not None:
            self.ready_before_consume.append(ready.query())
        stream.wait_event(ready)
        for tensor in tensors:
            # The allocator must not reuse storage while the compute stream reads it.
            tensor.record_stream(stream)
        self.consumed += 1
        return tensors

    def close(self):
        if self.closed:
            return
        self.stop.set()
        try:
            self.cpu_windows.close()
        finally:
            while True:
                try:
                    self.queue.get_nowait()
                except Empty:
                    break
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError('Device prefetch is still active; do not start another loader')
            self.copy_stream.synchronize()
            self.closed = True


def windows(loader, accumulation, compute_batch, count, prepared=False, pin=False):
    if prepared:
        return PreparedWindows(loader, accumulation, compute_batch, count, pin=pin)
    iterator = iter(loader)
    return (grouped_window(iterator, accumulation, compute_batch) for _ in range(count))
