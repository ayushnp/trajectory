"""JSONL writer: the source of truth. One thread, append-only, one file per UTC day."""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("tux_trace.store")
_STOP = object()
MAX_BATCH = 500


class JsonlWriter:
    def __init__(self, q: queue.Queue, data_dir: str | Path, fsync_interval_s: float = 5.0) -> None:
        self.q = q
        self.dir = Path(data_dir) / "calls"
        self.fsync_interval_s = fsync_interval_s
        self.written = 0
        self.write_errors = 0
        self.parse_errors = 0
        self.parse_unknown_events = 0
        self.parsed_calls = 0
        self.usage_unreported = 0
        self._thread = threading.Thread(target=self._run, name="jsonl-writer", daemon=True)
        self._last_fsync = time.monotonic()
        self._fh = None
        self._fh_day: str | None = None

    def start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Drain what is queued, then stop. Blocks the caller only at shutdown."""
        self.q.put(_STOP)  # blocking put is fine here: shutdown, not the request path
        self._thread.join(timeout)

    # -- internals ---------------------------------------------------------------------------
    def _run(self) -> None:
        stopping = False
        while not stopping:
            try:
                first = self.q.get(timeout=1.0)
            except queue.Empty:
                self._maybe_fsync()
                continue
            batch = [first]
            while len(batch) < MAX_BATCH:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            if _STOP in batch:
                stopping = True
                batch = [b for b in batch if b is not _STOP]
            try:
                self._write_batch(batch)
            except Exception:  # disk full, bad record, ...: lose the batch, keep the thread alive
                self.write_errors += len(batch)
                log.exception("failed to write %d records", len(batch))
            finally:
                for _ in range(len(batch) + (1 if stopping else 0)):
                    self.q.task_done()
            self._maybe_fsync()
        self._close()

    def _write_batch(self, batch: list[dict]) -> None:
        lines: dict[str, list[str]] = {}
        for item in batch:
            try:
                rec = item.to_record() if hasattr(item, "to_record") else item
            except Exception:
                self.write_errors += 1
                log.exception("failed to build record")
                continue
            self._count(item, rec)
            day = (rec.get("timing", {}).get("received_at") or "")[:10] \
                or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            lines.setdefault(day, []).append(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))
        for day, ls in lines.items():
            fh = self._open(day)
            fh.write("\n".join(ls) + "\n")
            fh.flush()
        self.written += sum(len(v) for v in lines.values())

    def _count(self, item, rec: dict) -> None:
        self.parse_unknown_events += getattr(item, "unknown_events", 0)
        self.parse_errors += getattr(item, "event_errors", 0) + (1 if getattr(item, "parse_failed", False) else 0)
        ok = (rec.get("response", {}).get("status") or 0) < 400
        if rec.get("request", {}).get("format", "unknown") != "unknown" and ok:
            self.parsed_calls += 1
            if not rec.get("usage", {}).get("input_total", {}).get("reported"):
                self.usage_unreported += 1

    def _open(self, day: str):
        if self._fh_day != day:
            self._close()
            self._fh = open(self.dir / f"{day}.jsonl", "a", encoding="utf-8", newline="\n")
            self._fh_day = day
        return self._fh

    def _maybe_fsync(self) -> None:
        if self._fh and time.monotonic() - self._last_fsync >= self.fsync_interval_s:
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                log.exception("fsync failed")
            self._last_fsync = time.monotonic()

    def _close(self) -> None:
        if self._fh:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except OSError:
                pass
            self._fh.close()
            self._fh = None
            self._fh_day = None
