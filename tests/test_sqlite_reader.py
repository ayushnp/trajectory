"""The SQLite read path must give the same answers as the in-memory reader, and survive what the
in-memory one could not (more than MAX_IN_MEMORY records, an old-schema index)."""
import json
import sqlite3

from trajectory import sessions as sess
from trajectory.index_db import SCHEMA_VERSION, SqliteIndex
from trajectory.tools import join_results


def call(cid, sid, step, turn=0, parent=None, tools=(), results=(), at=None, green=True):
    return {
        "call_id": cid, "session_id": sid, "step_index": step, "turn_index": turn, "parent_call_id": parent,
        "client": {"harness_hint": "claude-code", "key_hash": "k"},
        "request": {"format": "anthropic", "path": "/v1/messages", "model_requested": "m", "tool_results": list(results)},
        "response": {"status": 200, "model_served": "m", "stop_reason": "end_turn", "tool_calls": list(tools)},
        "timing": {"received_at": at or f"2026-11-13T10:00:{step:02d}+00:00", "t_received": step * 10**9,
                   "t_end": step * 10**9 + 5 * 10**8, "ttft_any_ms": 100.0, "total_ms": 500.0},
        "usage": {m: {"reported": True, "value": 10} for m in ("input_total", "output", "cache_read", "cache_write")},
        "outcome": {"incomplete": False, "client_disconnected": False, "error": None},
        "green": {"status": "ok", "basis": "exact", "energy_kwh": {"min": 1, "mean": 2, "max": 3},
                  "gwp_kgco2eq": {"min": 1, "mean": 2, "max": 3}} if green else None,
    }


def fixture():
    return [
        call("a1", "S1", 0, tools=[{"id": "t1", "name": "Bash"}]),
        call("a2", "S1", 1, results=[{"tool_call_id": "t1", "parent_call_id": "a1", "result_bytes": 42,
                                      "result_is_error": False, "gap_ms": 12.5}]),
        call("b1", "S2", 0, parent="a1"),                         # a subagent hanging off a1
        call("c1", "S3", 0, at="2026-11-13T11:00:00+00:00"),
    ]


def test_trajectory_matches_in_memory_reader(tmp_path):
    recs = fixture()
    idx = SqliteIndex(tmp_path / "i.db")
    idx.upsert(recs)
    got = idx.trajectory_records("S1")
    join_results(got)
    mem = json.loads(json.dumps(recs))
    join_results(mem)
    assert sess.trajectory("S1", got) == sess.trajectory("S1", mem)
    t = sess.trajectory("S1", got)
    assert t["steps"][0]["tool_calls"][0]["result_present"] is True
    assert [c["session"]["session_id"] for c in t["children"]] == ["S2"]


def test_session_page_and_summaries_match(tmp_path):
    recs = fixture()
    idx = SqliteIndex(tmp_path / "i.db")
    idx.upsert(recs)
    ids, total = idx.session_page(10, 0)
    assert total == 3 and ids[0] == "S3"                          # most recent first
    got = idx.records_for_sessions(ids)
    join_results(got)
    mem = json.loads(json.dumps(recs))
    join_results(mem)
    want, _ = sess.list_sessions(mem, 10, 0)
    by_id = {r["call_id"]: r for r in got}
    groups = sess.group_sessions(got)
    have = [sess.session_summary(s, groups[s], by_id) for s in ids]
    assert have == want


def test_session_beyond_the_old_20000_cap_is_complete(tmp_path):
    idx = SqliteIndex(tmp_path / "i.db")
    idx.upsert([call(f"old{i}", "BIG", i, at=f"2026-11-12T00:00:{i % 60:02d}+00:00") for i in range(3)])
    idx.upsert([call(f"f{i}", f"F{i}", 0) for i in range(500)])   # newer filler
    t = sess.trajectory("BIG", idx.trajectory_records("BIG"))
    assert t["session"]["steps"] == 3


def test_get_joined_folds_in_results(tmp_path):
    idx = SqliteIndex(tmp_path / "i.db")
    idx.upsert(fixture())
    rec = idx.get_joined("a1")
    tc = rec["response"]["tool_calls"][0]
    assert tc["result_present"] and tc["result_bytes"] == 42 and tc["gap_ms"] == 12.5
    assert idx.get_joined("nope") is None


def test_old_schema_is_dropped_and_flagged_for_refill(tmp_path):
    p = tmp_path / "i.db"
    db = sqlite3.connect(p)
    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    db.execute("INSERT INTO meta VALUES ('schema_version','1')")
    db.execute("CREATE TABLE calls (call_id TEXT PRIMARY KEY)")
    db.commit()
    db.close()
    idx = SqliteIndex(p)
    assert idx.needs_backfill
    assert idx.query("SELECT value FROM meta WHERE key='schema_version'")[0][0] == str(SCHEMA_VERSION)
    d = tmp_path / "calls"
    d.mkdir()
    (d / "2026-11-13.jsonl").write_text("\n".join(json.dumps(r) for r in fixture()) + "\n{torn", encoding="utf-8")
    assert idx.backfill(tmp_path) == 4 and not idx.needs_backfill
    assert idx.get_doc("c1")["session_id"] == "S3"


def test_empty_and_unknown(tmp_path):
    idx = SqliteIndex(tmp_path / "i.db")
    assert idx.trajectory_records("nope") == [] and idx.records_for_sessions([]) == []
    assert idx.recent_docs(10, 0) == ([], 0)
