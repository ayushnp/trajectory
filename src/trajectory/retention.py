"""Retention (spec section 16): delete data older than `retention_days`.

Deletion is deliberately narrow and boring:
  * only files inside the proxy's own data_dir, in calls/ and content/;
  * only names this program creates: calls/YYYY-MM-DD.jsonl and content/<uuid>.json;
  * anything else in those folders is left alone, and nothing is followed through a symlink.
The age of a JSONL file comes from its date-stamped name; a content file's age comes from its mtime.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("trajectory.retention")
DAY_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.jsonl$")
CONTENT_FILE = re.compile(r"^[0-9a-fA-F\-]{36}\.json$")


def _real_subdir(root: Path, name: str) -> Path | None:
    """root/name, only if it is a genuine directory that lives directly inside root.

    A symlink or (on Windows) a directory junction can make root/content point somewhere else
    entirely; deleting 'its' files would then delete files outside the data directory. Resolving
    the directory and requiring its parent to be root catches both.
    """
    d = root / name
    try:
        if not d.is_dir() or d.is_symlink():
            return None
        if d.resolve().parent != root.resolve():
            return None
        if getattr(d.stat(), "st_file_attributes", 0) & 0x400:       # FILE_ATTRIBUTE_REPARSE_POINT
            return None
        return d
    except OSError:
        return None


def cutoff_day(retention_days: int, now: datetime | None = None) -> str:
    """Days strictly before this UTC date are deleted. retention_days=7 keeps today and 6 prior days."""
    now = now or datetime.now(timezone.utc)
    return (now - timedelta(days=retention_days - 1)).strftime("%Y-%m-%d")


def purge(data_dir: str | Path, retention_days: int, index=None, now: datetime | None = None,
          dry_run: bool = False) -> dict:
    now = now or datetime.now(timezone.utc)
    root = Path(data_dir).resolve()
    cut = cutoff_day(retention_days, now)
    out = {"cutoff_day": cut, "jsonl_deleted": 0, "content_deleted": 0, "index_calls_deleted": 0, "errors": 0}

    calls = _real_subdir(root, "calls")
    if calls is not None:
        for f in calls.iterdir():
            m = DAY_FILE.match(f.name)
            if m and m.group(1) < cut and f.is_file() and not f.is_symlink():
                try:
                    if not dry_run:
                        f.unlink()
                    out["jsonl_deleted"] += 1
                except OSError:
                    out["errors"] += 1
                    log.exception("could not delete %s", f.name)

    content = _real_subdir(root, "content")
    limit = now.timestamp() - (retention_days * 86400)
    if content is not None:
        for f in content.iterdir():
            if CONTENT_FILE.match(f.name) and f.is_file() and not f.is_symlink():
                try:
                    if f.stat().st_mtime < limit:
                        if not dry_run:
                            f.unlink()
                        out["content_deleted"] += 1
                except OSError:
                    out["errors"] += 1
                    log.exception("could not delete %s", f.name)

    if index is not None and not dry_run:
        try:
            out["index_calls_deleted"] = index.delete_before(cut)
        except Exception:
            out["errors"] += 1
            log.exception("could not purge the sqlite index")
    if out["jsonl_deleted"] or out["content_deleted"]:
        log.info("retention: %s", out)
    return out


class RetentionTask:
    """Runs purge() at startup and then every `interval_s`. A daemon thread; failure is logged, never fatal."""

    def __init__(self, data_dir, retention_days: int, index=None, interval_s: float = 3600.0) -> None:
        self.data_dir, self.days, self.index, self.interval_s = data_dir, retention_days, index, interval_s
        self.last: dict | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="retention", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.last = purge(self.data_dir, self.days, self.index)
            except Exception:
                log.exception("retention run failed")
            self._stop.wait(self.interval_s)
