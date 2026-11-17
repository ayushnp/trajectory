"""Bounded hand-off from the fast path to the writer. submit() never blocks and never raises."""
from __future__ import annotations

import logging
import queue
import time
from pathlib import Path

from .store import JsonlWriter

log = logging.getLogger("trajectory.tee")
DROP_LOG_INTERVAL_S = 10.0


class Tee:
    def __init__(self, maxsize: int, data_dir: str | Path, fsync_interval_s: float = 5.0,
                 start_writer: bool = True, chainer=None, pairer=None, attacher=None, green=None,
                 content=None, index=None) -> None:
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped_records = 0
        self.submitted = 0
        self._last_drop_log = 0.0
        self._dropped_since_log = 0
        self.writer = JsonlWriter(self.q, data_dir, fsync_interval_s, chainer, pairer, attacher, green,
                                 content, index)
        if start_writer:
            self.writer.start()

    @property
    def depth(self) -> int:
        return self.q.qsize()

    def submit(self, record) -> bool:
        try:
            self.q.put_nowait(record)
            self.submitted += 1
            return True
        except queue.Full:
            if hasattr(record, "release"):
                record.release()     # never reaches the writer: return its capture budget now
            self.dropped_records += 1
            self._dropped_since_log += 1
            now = time.monotonic()
            if now - self._last_drop_log >= DROP_LOG_INTERVAL_S:
                log.warning("record queue full: dropped %d record(s) (total %d)",
                            self._dropped_since_log, self.dropped_records)
                self._last_drop_log = now
                self._dropped_since_log = 0
            return False

    def close(self) -> None:
        self.writer.stop()
