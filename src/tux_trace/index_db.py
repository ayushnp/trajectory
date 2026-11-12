"""SQLite index over the JSONL records (spec section 16).

JSONL is authoritative. This database is a derived, rebuildable view for querying: delete it and
`tux-trace reindex` recreates it from the files. Writes come from the single writer thread; WAL
mode lets readers (the API, a human with sqlite3) query concurrently.

Tables
  calls     one row per model call, with the fields worth filtering and sorting on
  sessions  one row per session, kept current as calls arrive
  greens    one row per estimate: ranges, basis, method version
  links     relationships between calls and sessions: subagent edges (with confidence), tool-result
            pairings (with gap_ms), and continued-from links
Unreported values are NULL, never 0.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

SCHEMA_VERSION = 1
DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS calls (
  call_id TEXT PRIMARY KEY, session_id TEXT, step_index INTEGER, turn_index INTEGER,
  received_at TEXT NOT NULL, day TEXT NOT NULL, format TEXT, path TEXT,
  model_requested TEXT, model_served TEXT, status INTEGER, stop_reason TEXT,
  ttft_any_ms REAL, ttft_visible_ms REAL, total_ms REAL, duration_ms REAL,
  input_total INTEGER, output INTEGER, cache_read INTEGER, cache_write INTEGER,
  tool_call_count INTEGER, incomplete INTEGER, error_type TEXT, harness TEXT, client_hash TEXT,
  compaction_suspected INTEGER, unattached INTEGER);
CREATE INDEX IF NOT EXISTS calls_session ON calls(session_id, step_index);
CREATE INDEX IF NOT EXISTS calls_time ON calls(received_at);
CREATE INDEX IF NOT EXISTS calls_day ON calls(day);
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, first_seen TEXT, last_seen TEXT, steps INTEGER, turns INTEGER,
  harness TEXT, continued_from TEXT, parent_call_id TEXT);
CREATE TABLE IF NOT EXISTS greens (
  call_id TEXT PRIMARY KEY, status TEXT, basis TEXT, method TEXT, method_version TEXT,
  energy_min REAL, energy_mean REAL, energy_max REAL, gwp_min REAL, gwp_mean REAL, gwp_max REAL,
  assumptions_sha256 TEXT);
CREATE TABLE IF NOT EXISTS links (
  kind TEXT NOT NULL, from_id TEXT NOT NULL, to_id TEXT NOT NULL, confidence TEXT, score REAL,
  signals TEXT, gap_ms REAL, day TEXT, PRIMARY KEY (kind, from_id, to_id));
"""


def _m(rec: dict, name: str):
    """A usage metric's value, or None if the provider did not report it."""
    u = (rec.get("usage") or {}).get(name) or {}
    return u.get("value") if u.get("reported") else None


class SqliteIndex:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(DDL)
        self._db.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # -- writing ----------------------------------------------------------------------------
    def upsert(self, records: list[dict]) -> None:
        with self._lock:
            self._db.execute("BEGIN")
            try:
                for r in records:
                    self._upsert_one(r)
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def _upsert_one(self, r: dict) -> None:
        req, res, t, o = r["request"], r["response"], r["timing"], r["outcome"]
        day = (t.get("received_at") or "")[:10]
        err = (o.get("error") or {}).get("type")
        self._db.execute(
            "INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r["call_id"], r.get("session_id"), r.get("step_index"), r.get("turn_index"),
             t.get("received_at"), day, req.get("format"), req.get("path"),
             req.get("model_requested"), res.get("model_served"), res.get("status"), res.get("stop_reason"),
             t.get("ttft_any_ms"), t.get("ttft_visible_ms"), t.get("total_ms"), t.get("duration_ms"),
             _m(r, "input_total"), _m(r, "output"), _m(r, "cache_read"), _m(r, "cache_write"),
             len(res.get("tool_calls") or []), int(bool(o.get("incomplete"))), err,
             (r.get("client") or {}).get("harness_hint"), (r.get("client") or {}).get("key_hash"),
             int(bool(r.get("compaction_suspected"))), int(bool(r.get("unattached")))))
        sid = r.get("session_id")
        if sid:
            self._db.execute(
                "INSERT INTO sessions(session_id, first_seen, last_seen, steps, turns, harness, continued_from, parent_call_id)"
                " VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET"
                " last_seen=MAX(last_seen, excluded.last_seen), first_seen=MIN(first_seen, excluded.first_seen),"
                " steps=(SELECT COUNT(DISTINCT step_index) FROM calls WHERE session_id=excluded.session_id),"
                " turns=MAX(turns, excluded.turns)",
                (sid, t.get("received_at"), t.get("received_at"), 1, (r.get("turn_index") or 0) + 1,
                 (r.get("client") or {}).get("harness_hint"), r.get("continued_from_session"), r.get("parent_call_id")))
            if r.get("continued_from_session"):
                self._link("continued_from", sid, r["continued_from_session"], day=day)
        g = r.get("green") or {}
        if g.get("status"):
            e, w = g.get("energy_kwh") or {}, g.get("gwp_kgco2eq") or {}
            self._db.execute("INSERT OR REPLACE INTO greens VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                             (r["call_id"], g.get("status"), g.get("basis"), g.get("method"), g.get("method_version"),
                              e.get("min"), e.get("mean"), e.get("max"), w.get("min"), w.get("mean"), w.get("max"),
                              g.get("assumptions_sha256")))
        edge = r.get("subagent_edge")
        if edge and r.get("parent_call_id"):
            self._link("subagent", r["call_id"], r["parent_call_id"], edge.get("confidence"), edge.get("score"),
                       ",".join(edge.get("signals") or []), None, day)
        for res_ in (req.get("tool_results") or []):
            if res_.get("parent_call_id") and not res_.get("repeat"):
                self._link("tool_result", r["call_id"], res_["parent_call_id"], None, None,
                           res_.get("tool_call_id"), res_.get("gap_ms"), day)

    def _link(self, kind, a, b, conf=None, score=None, signals=None, gap=None, day=None) -> None:
        self._db.execute("INSERT OR REPLACE INTO links VALUES (?,?,?,?,?,?,?,?)", (kind, a, b, conf, score, signals, gap, day))

    # -- maintenance ------------------------------------------------------------------------
    def delete_before(self, day: str) -> int:
        """Remove everything recorded on a UTC day earlier than `day` (YYYY-MM-DD). Returns calls removed."""
        with self._lock:
            ids = [x[0] for x in self._db.execute("SELECT call_id FROM calls WHERE day < ?", (day,))]
            self._db.execute("BEGIN")
            self._db.execute("DELETE FROM greens WHERE call_id IN (SELECT call_id FROM calls WHERE day < ?)", (day,))
            self._db.execute("DELETE FROM links WHERE day < ?", (day,))
            self._db.execute("DELETE FROM calls WHERE day < ?", (day,))
            self._db.execute("DELETE FROM sessions WHERE session_id NOT IN (SELECT DISTINCT session_id FROM calls WHERE session_id IS NOT NULL)")
            self._db.execute("COMMIT")
            return len(ids)

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    # -- rebuilding -------------------------------------------------------------------------
    @classmethod
    def rebuild(cls, data_dir: str | Path, db_path: str | Path | None = None) -> tuple["SqliteIndex", int]:
        """Drop and recreate the index from data/calls/*.jsonl. Returns (index, records_indexed)."""
        db_path = Path(db_path or Path(data_dir) / "index.db")
        for suffix in ("", "-wal", "-shm"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
        idx = cls(db_path)
        n, batch = 0, []
        for f in sorted((Path(data_dir) / "calls").glob("*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        batch.append(json.loads(line))
                    except ValueError:
                        continue                    # a torn final line is skipped, not fatal
                    if len(batch) >= 500:
                        idx.upsert(batch); n += len(batch); batch = []
        if batch:
            idx.upsert(batch); n += len(batch)
        return idx, n
