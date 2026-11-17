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

log = logging.getLogger("trajectory.store")
_STOP = object()
MAX_BATCH = 500


class JsonlWriter:
    def __init__(self, q: queue.Queue, data_dir: str | Path, fsync_interval_s: float = 5.0,
                 chainer=None, pairer=None, attacher=None, green=None,
                 content=None, index=None) -> None:
        self.q = q
        self.pairer = pairer
        self.attacher = attacher
        self.green = green
        self.content = content           # ContentStore or None
        self.index = index               # SqliteIndex or None
        self.index_errors = 0
        self.chainer = chainer          # runs here, on the writer thread: no locks needed
        self.chain_errors = 0
        self.dir = Path(data_dir) / "calls"
        self.fsync_interval_s = fsync_interval_s
        self.written = 0
        self.write_errors = 0
        self.parse_errors = 0
        self.parse_unknown_events = 0
        self.parsed_calls = 0
        self.usage_unreported = 0
        self.green_total = 0
        self.green_not_ok = 0
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
        indexed: list[dict] = []
        for item in batch:
            try:
                rec = item.to_record() if hasattr(item, "to_record") else item
            except Exception:
                self.write_errors += 1
                log.exception("failed to build record")
                if hasattr(item, "release"):
                    item.release()          # the contract: captured bytes are always freed
                continue
            if self.chainer is not None:
                try:
                    self.chainer.assign(rec)
                except Exception:       # fail open: an unchained record is still a record
                    self.chain_errors += 1
                    log.exception("chainer failed")
            if self.pairer is not None:
                try:
                    self.pairer.attach(rec, getattr(item, "tool_results", None) or [])
                    self.pairer.register(rec)
                except Exception:       # fail open: unpaired tool calls are still recorded
                    self.chain_errors += 1
                    log.exception("tool pairing failed")
            if self.attacher is not None:
                try:
                    self.attacher.note_results(rec)
                    self.attacher.assess(rec, getattr(item, "first_text", None))
                    self.attacher.register(rec)
                except Exception:       # fail open: an unattached call is still a record
                    self.chain_errors += 1
                    log.exception("subagent attachment failed")
            if self.green is not None:
                rec["green"] = self.green.estimate(rec)       # never raises; errors become status=error
            self._store_content(item, rec)
            self._count(item, rec)
            day = (rec.get("timing", {}).get("received_at") or "")[:10] \
                or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            lines.setdefault(day, []).append(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))
            indexed.append(rec)
        for day, ls in lines.items():
            fh = self._open(day)
            fh.write("\n".join(ls) + "\n")
            fh.flush()
        self.written += sum(len(v) for v in lines.values())
        if self.index is not None and indexed:
            try:                 # the index is rebuildable from JSONL, so its failure never loses data
                self.index.upsert(indexed)
            except Exception:
                self.index_errors += 1
                log.exception("sqlite index update failed")

    def _store_content(self, item, rec: dict) -> None:
        """Write the content file (if enabled), then ALWAYS free the captured bytes."""
        try:
            if self.content is not None and self.content.enabled and hasattr(item, "content_parts"):
                from .content import decode_response
                req, chunks, enc = item.content_parts()
                path = self.content.save(rec["call_id"], req, decode_response(chunks, enc),
                                         meta={"format": rec["request"]["format"], "path": rec["request"]["path"]})
                if path:
                    rec["content"] = {"stored": True, "path": path}
        except Exception:
            log.exception("content step failed")
        finally:
            if hasattr(item, "release"):
                item.release()

    @property
    def unknown_green_share(self):
        return round(self.green_not_ok / self.green_total, 4) if self.green_total else None

    def _count(self, item, rec: dict) -> None:
        if rec.get("green", {}).get("status") not in (None, "disabled") and rec["request"].get("format") != "unknown":
            self.green_total += 1
            if rec["green"]["status"] != "ok":
                self.green_not_ok += 1
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
