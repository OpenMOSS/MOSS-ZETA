"""Bounded, ordered request evidence writes outside the inference event loop."""
import json
from pathlib import Path
import queue
import threading
import time


class RequestJournal:
    def __init__(self, path, *, max_pending_bytes=64 * 1024 * 1024,
                 batch_bytes=1024 * 1024, opener=None):
        if max_pending_bytes < 1 or batch_bytes < 1:
            raise ValueError('Journal limits must be positive')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_pending_bytes = max_pending_bytes
        self.batch_bytes = batch_bytes
        self._queue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._error = None
        self._closed = False
        self._pending = self._accepted = self._written = self._peak = 0
        self._last_write_ms = 0.0
        # Opening once is startup work; no request opens or closes the file.
        self._output = (opener or open)(self.path, 'ab')
        self._thread = threading.Thread(target=self._run, name='request-journal', daemon=True)
        self._thread.start()

    def _check_locked(self):
        if self._error is not None:
            raise RuntimeError('Request journal write failed') from self._error
        if self._closed:
            raise RuntimeError('Request journal is closed')

    def check(self):
        with self._lock:
            self._check_locked()

    def append(self, record):
        # Freeze the exact record before handing it to another thread.
        data = (json.dumps(record, ensure_ascii=False) + '\n').encode('utf8')
        with self._lock:
            self._check_locked()
            if self._pending + len(data) > self.max_pending_bytes:
                raise RuntimeError('Request journal capacity exceeded; refusing unrecorded request')
            self._pending += len(data)
            self._accepted += 1
            self._peak = max(self._peak, self._pending)
            self._queue.put(data)

    def snapshot(self):
        with self._lock:
            return dict(mode='async-bounded-v1', pending_bytes=self._pending,
                        max_pending_bytes=self.max_pending_bytes, peak_pending_bytes=self._peak,
                        accepted_records=self._accepted, written_records=self._written,
                        last_write_ms=self._last_write_ms,
                        failed=self._error is not None, closed=self._closed)

    def _run(self):
        try:
            with self._output as output:
                while True:
                    data = self._queue.get()
                    if data is None:
                        break
                    batch, size, stopping = [data], len(data), False
                    while size < self.batch_bytes:
                        try:
                            item = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            stopping = True
                            break
                        batch.append(item)
                        size += len(item)
                    started = time.perf_counter()
                    if output.write(b''.join(batch)) != size:
                        raise OSError('Short request journal write')
                    output.flush()
                    elapsed = (time.perf_counter() - started) * 1000
                    with self._lock:
                        self._pending -= size
                        self._written += len(batch)
                        self._last_write_ms = elapsed
                    if stopping:
                        break
        except BaseException as error:
            with self._lock:
                self._error = error

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)
        self._thread.join()
        with self._lock:
            if self._error is not None:
                raise RuntimeError('Request journal failed to drain') from self._error
            if self._pending or self._written != self._accepted:
                raise RuntimeError('Request journal closed with missing evidence')
