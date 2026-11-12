"""Content storage + redaction, retention, SQLite index/reindex, capture budget, /metrics, smoke, sampling."""
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from tux_trace.content import MASK, ContentStore, Redactor
from tux_trace.index_db import SqliteIndex
from tux_trace.record import BUDGET, CallTrace
from tux_trace.retention import cutoff_day, purge

SECRET_KEY = "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
SSE = [b'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-haiku-5-5","usage":{"input_tokens":9}}}\n\n',
       b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"my key is ' + SECRET_KEY.encode() + b'"}}\n\n',
       b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n',
       b'event: message_stop\ndata: {"type":"message_stop"}\n\n']


def wait_for(pred, timeout=6.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def records(build):
    out = []
    for f in sorted((build.data_dir / "calls").glob("*.jsonl")):
        out += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x]
    return out


# ================================================================== redaction
@pytest.mark.parametrize("secret", [
    "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", "sk-proj-abcdefghijklmnop1234567890",
    "Bearer abcdefghijklmnopqrstuvwxyz0123456789", "AKIAIOSFODNN7EXAMPLE",
    "ghp_abcdefghijklmnopqrstuvwxyz0123456789", "xoxb-1234567890-abcdefghij",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj34\n-----END RSA PRIVATE KEY-----",
    'api_key = "hunter2hunter2"', "password: correcthorsebattery"])
def test_default_patterns_redact_common_secrets(secret):
    out = Redactor()("before " + secret + " after")
    assert MASK in out and out.startswith("before ") and out.endswith(" after")
    core = secret.split()[-1].strip('"')
    assert core not in out


def test_redaction_leaves_ordinary_text_alone_and_counts_hits():
    r = Redactor()
    text = "Please refactor the sk-learn pipeline and explain tokens in the docs."
    assert r(text) == text and r.hits == 0
    r("key sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    assert r.hits >= 1


def test_user_patterns_are_added_to_the_defaults():
    r = Redactor([r"ACME-\d{6}"])
    assert "ACME-123456" not in r("id ACME-123456") and "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" not in r("sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    assert "ACME-123456" in Redactor(use_defaults=False)("ACME-123456")


# ================================================================== content store
def test_content_store_writes_redacted_file_and_disabled_writes_nothing(tmp_path):
    cs = ContentStore(tmp_path, True, Redactor())
    body = json.dumps({"messages": [{"role": "user", "content": "use " + SECRET_KEY}]}).encode()
    path = cs.save("abc", body, "reply with " + SECRET_KEY)
    assert path == "content/abc.json"
    text = (tmp_path / path).read_text(encoding="utf-8")
    assert SECRET_KEY not in text and MASK in text
    doc = json.loads(text)
    assert doc["redacted"] is True and doc["request"]["messages"][0]["role"] == "user"
    off = ContentStore(tmp_path / "off", False, Redactor())
    assert off.save("x", b"{}", "r") is None and not (tmp_path / "off").exists()


def test_content_store_never_raises(tmp_path):
    blocked = tmp_path / "data"
    blocked.write_text("a file where the directory should be")
    cs = ContentStore(blocked, True, Redactor())
    assert cs.save("id", b"{}", "r") is None and cs.errors == 1


def test_end_to_end_content_is_redacted_but_the_relayed_bytes_are_not(proxy_factory):
    proxy, _, _ = proxy_factory(SSE)
    body = json.dumps({"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "my token " + SECRET_KEY}]})
    r = httpx.post(proxy + "/v1/messages", content=body, headers={"x-api-key": "k"})
    assert SECRET_KEY.encode() in r.content                          # pass-through: untouched
    assert wait_for(lambda: len(records(proxy_factory)) == 1)
    rec = records(proxy_factory)[0]
    assert rec["content"]["stored"] is True and rec["content"]["path"].startswith("content/")
    assert SECRET_KEY not in json.dumps(rec)                          # never inline in the record
    stored = (proxy_factory.data_dir / rec["content"]["path"]).read_text(encoding="utf-8")
    assert SECRET_KEY not in stored and MASK in stored
    api = httpx.get(proxy + f"/_trace/api/calls/{rec['call_id']}/content").json()
    assert api["redacted"] is True and SECRET_KEY not in json.dumps(api)


def test_store_content_false_stores_nothing(proxy_factory):
    proxy, _, _ = proxy_factory(SSE, store_content=False)
    httpx.post(proxy + "/v1/messages", content=b'{"model":"m"}')
    assert wait_for(lambda: len(records(proxy_factory)) == 1)
    assert records(proxy_factory)[0]["content"] == {"stored": False, "path": None}
    assert not (proxy_factory.data_dir / "content").exists()
    r = httpx.get(proxy + f"/_trace/api/calls/{records(proxy_factory)[0]['call_id']}/content")
    assert r.status_code == 404 and r.json()["stored"] is False


def test_content_endpoint_will_not_serve_files_outside_the_data_dir(proxy_factory, tmp_path):
    proxy, _, _ = proxy_factory(SSE)
    httpx.post(proxy + "/v1/messages", content=b'{"model":"m"}')
    assert wait_for(lambda: len(records(proxy_factory)) == 1)
    cid = records(proxy_factory)[0]["call_id"]
    outside = tmp_path.parent / "outside_secret.json"
    outside.write_text('{"leak": true}')
    # forge the stored path in the on-disk record and reload: the endpoint must refuse to leave data_dir
    f = next((proxy_factory.data_dir / "calls").glob("*.jsonl"))
    rec = json.loads(f.read_text(encoding="utf-8").splitlines()[0])
    rec["content"] = {"stored": True, "path": "../outside_secret.json"}
    f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    r = httpx.get(proxy + f"/_trace/api/calls/{cid}/content")
    assert r.status_code == 404 and "leak" not in r.text
    outside.unlink()


# ================================================================== retention (destructive: be strict)
def make_data(tmp_path, days_old):
    now = datetime.now(timezone.utc)
    calls = tmp_path / "calls"
    content = tmp_path / "content"
    calls.mkdir(parents=True), content.mkdir(parents=True)
    names = {}
    for d in days_old:
        day = (now - timedelta(days=d)).strftime("%Y-%m-%d")
        (calls / f"{day}.jsonl").write_text("{}\n")
        cid = str(uuid.uuid4())
        p = content / f"{cid}.json"
        p.write_text("{}")
        ts = (now - timedelta(days=d)).timestamp()
        os.utime(p, (ts, ts))
        names[d] = (f"{day}.jsonl", p.name)
    return names


def test_cutoff_keeps_exactly_retention_days():
    now = datetime(2026, 1, 10, 12, tzinfo=timezone.utc)
    assert cutoff_day(7, now) == "2026-01-04"          # keeps 4..10 = 7 UTC days
    assert cutoff_day(1, now) == "2026-01-10"          # keeps today only


def test_purge_deletes_old_and_keeps_recent(tmp_path):
    names = make_data(tmp_path, [0, 3, 6, 8, 20])
    out = purge(tmp_path, 7)
    assert out["jsonl_deleted"] == 2 and out["content_deleted"] == 2
    for d in (0, 3, 6):
        assert (tmp_path / "calls" / names[d][0]).exists() and (tmp_path / "content" / names[d][1]).exists()
    for d in (8, 20):
        assert not (tmp_path / "calls" / names[d][0]).exists() and not (tmp_path / "content" / names[d][1]).exists()


def test_purge_only_touches_files_it_created(tmp_path):
    make_data(tmp_path, [30])
    bystanders = [tmp_path / "calls" / "notes.txt", tmp_path / "calls" / "2020-01-01.jsonl.bak",
                  tmp_path / "content" / "readme.md", tmp_path / "content" / "short.json",
                  tmp_path / "index.db", tmp_path / "important.jsonl"]
    for b in bystanders:
        b.write_text("keep me")
        os.utime(b, (1, 1))                           # ancient
    purge(tmp_path, 1)
    assert all(b.exists() for b in bystanders)


def test_purge_does_not_follow_symlinks_or_touch_other_directories(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    target = elsewhere / "2000-01-01.jsonl"
    target.write_text("precious")
    data = tmp_path / "data"
    (data / "calls").mkdir(parents=True)
    link = data / "calls" / "2000-01-01.jsonl"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted here")
    purge(data, 1)
    assert target.exists()


def test_dry_run_deletes_nothing(tmp_path):
    names = make_data(tmp_path, [40])
    out = purge(tmp_path, 7, dry_run=True)
    assert out["jsonl_deleted"] == 1 and (tmp_path / "calls" / names[40][0]).exists()


def test_purge_on_a_missing_directory_is_a_noop(tmp_path):
    assert purge(tmp_path / "nope", 7)["errors"] == 0


def test_purge_also_trims_the_sqlite_index(tmp_path):
    idx = SqliteIndex(tmp_path / "index.db")
    old = fake_record("old", "s1", "2020-01-01T00:00:00+00:00")
    new = fake_record("new", "s2", datetime.now(timezone.utc).isoformat())
    idx.upsert([old, new])
    (tmp_path / "calls").mkdir()
    out = purge(tmp_path, 7, idx)
    assert out["index_calls_deleted"] == 1
    assert [r[0] for r in idx.query("SELECT call_id FROM calls")] == ["new"]
    assert [r[0] for r in idx.query("SELECT session_id FROM sessions")] == ["s2"]


# ================================================================== sqlite index
def fake_record(cid, sid, when, step=0, out=7, cache=None, green="exact", parent=None, edge=None):
    return {"call_id": cid, "session_id": sid, "step_index": step, "turn_index": 0,
            "client": {"harness_hint": "claude-code", "key_hash": "h"},
            "request": {"format": "anthropic", "path": "/v1/messages", "model_requested": "m",
                        "tool_results": [{"tool_call_id": "t1", "parent_call_id": "prev", "gap_ms": 12.5}] if step else []},
            "response": {"status": 200, "model_served": "m", "stop_reason": "end_turn", "tool_calls": [{"id": "t"}]},
            "timing": {"received_at": when, "ttft_any_ms": 100.0, "total_ms": 900.0, "duration_ms": 950.0},
            "usage": {"input_total": {"value": 50, "reported": True}, "output": {"value": out, "reported": True},
                      "cache_read": {"value": cache, "reported": cache is not None}, "cache_write": {"value": None, "reported": False}},
            "outcome": {"error": None, "incomplete": False},
            "green": {"status": "ok", "basis": green, "method": "ecologits", "method_version": "0.11.3",
                      "energy_kwh": {"min": 1e-6, "mean": 2e-6, "max": 3e-6}, "gwp_kgco2eq": {"min": 1e-7, "mean": 2e-7, "max": 3e-7},
                      "assumptions_sha256": "abc"},
            "parent_call_id": parent, "subagent_edge": edge}


def test_index_stores_nulls_for_unreported_metrics_never_zero(tmp_path):
    idx = SqliteIndex(tmp_path / "i.db")
    idx.upsert([fake_record("a", "s", "2026-01-01T00:00:00+00:00", cache=None)])
    row = idx.query("SELECT cache_read, cache_write, output FROM calls")[0]
    assert row == (None, None, 7)
    idx.upsert([fake_record("b", "s", "2026-01-01T00:00:01+00:00", step=1, cache=0)])
    assert idx.query("SELECT cache_read FROM calls WHERE call_id='b'")[0][0] == 0      # reported zero stays 0


def test_index_tables_links_and_greens(tmp_path):
    idx = SqliteIndex(tmp_path / "i.db")
    edge = {"confidence": "high", "score": 0.85, "signals": ["echo", "overlap"]}
    idx.upsert([fake_record("p", "sp", "2026-01-01T00:00:00+00:00"),
                fake_record("c", "sc", "2026-01-01T00:00:02+00:00", parent="p", edge=edge),
                fake_record("p2", "sp", "2026-01-01T00:00:03+00:00", step=1)])
    assert {r[0] for r in idx.query("SELECT name FROM sqlite_master WHERE type='table'")} >= {"sessions", "calls", "greens", "links"}
    sub = idx.query("SELECT from_id, to_id, confidence, score, signals FROM links WHERE kind='subagent'")
    assert sub == [("c", "p", "high", 0.85, "echo,overlap")]
    gap = idx.query("SELECT gap_ms FROM links WHERE kind='tool_result'")
    assert gap == [(12.5,)]
    g = idx.query("SELECT basis, energy_min, energy_mean, energy_max FROM greens WHERE call_id='p'")[0]
    assert g == ("exact", 1e-6, 2e-6, 3e-6)
    assert idx.query("SELECT steps FROM sessions WHERE session_id='sp'") == [(2,)]


def test_upserting_the_same_call_twice_does_not_duplicate(tmp_path):
    idx = SqliteIndex(tmp_path / "i.db")
    r = fake_record("a", "s", "2026-01-01T00:00:00+00:00")
    idx.upsert([r]); idx.upsert([r])
    assert idx.query("SELECT COUNT(*) FROM calls") == [(1,)] and idx.query("SELECT COUNT(*) FROM greens") == [(1,)]


def test_reindex_rebuilds_the_database_from_jsonl(tmp_path):
    (tmp_path / "calls").mkdir()
    recs = [fake_record("a", "s", "2026-01-01T00:00:00+00:00"), fake_record("b", "s", "2026-01-01T00:00:05+00:00", step=1)]
    (tmp_path / "calls" / "2026-01-01.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n{torn line", encoding="utf-8")
    (tmp_path / "index.db").write_text("garbage that is not a database")
    idx, n = SqliteIndex.rebuild(tmp_path)
    assert n == 2 and idx.query("SELECT COUNT(*) FROM calls") == [(2,)]       # torn line skipped, bad db replaced
    idx.close()


def test_live_index_matches_a_rebuild(proxy_factory):
    proxy, _, _ = proxy_factory(SSE)
    for _ in range(3):
        httpx.post(proxy + "/v1/messages", content=json.dumps({"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "q"}]}))
    assert wait_for(lambda: len(records(proxy_factory)) == 3)
    time.sleep(0.3)
    live = sqlite3.connect(proxy_factory.data_dir / "index.db")
    live_rows = live.execute("SELECT call_id, session_id, output FROM calls ORDER BY call_id").fetchall()
    live.close()
    copy = proxy_factory.data_dir / "rebuild"
    copy.mkdir()
    (copy / "calls").mkdir()
    for f in (proxy_factory.data_dir / "calls").glob("*.jsonl"):
        (copy / "calls" / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    idx, n = SqliteIndex.rebuild(copy)
    assert n == 3 and idx.query("SELECT call_id, session_id, output FROM calls ORDER BY call_id") == live_rows
    idx.close()


def test_index_failure_never_loses_the_jsonl_record(proxy_factory, monkeypatch):
    monkeypatch.setattr(SqliteIndex, "upsert", lambda self, recs: (_ for _ in ()).throw(sqlite3.OperationalError("disk I/O")))
    proxy, _, _ = proxy_factory(SSE)
    r = httpx.post(proxy + "/v1/messages", content=b'{"model":"m"}')
    assert r.status_code == 200
    assert wait_for(lambda: len(records(proxy_factory)) == 1)
    assert wait_for(lambda: httpx.get(proxy + "/stats").json()["index_errors"] >= 1)


# ================================================================== capture budget
def test_budget_reserves_and_returns_bytes():
    from tux_trace.record import CaptureBudget
    b = CaptureBudget(1000)
    tr = CallTrace(method="POST", path="/x", t_received=0, budget=b)
    tr.add_req_chunk(b"x" * 400)
    assert b.used == 400 and tr.reserved == 400
    tr.release()
    assert b.used == 0 and tr.reserved == 0
    tr.release()                                                  # idempotent
    assert b.used == 0


def test_exhausted_budget_stops_capture_but_not_counting():
    from tux_trace.record import CaptureBudget
    b = CaptureBudget(100)
    tr = CallTrace(method="POST", path="/x", t_received=0, budget=b)
    tr.add_req_chunk(b"a" * 80)
    tr.add_req_chunk(b"b" * 80)                                   # does not fit
    tr.add_resp_chunk(1, b"c" * 500)                              # does not fit
    assert tr.body_bytes == 160 and tr.resp_bytes == 500          # facts still counted
    assert tr.req_truncated and tr.resp_truncated and b.denied == 2
    assert tr.to_record()["response"]["capture_truncated"] is True
    tr.release()
    assert b.used == 0


def test_two_calls_share_one_budget():
    from tux_trace.record import CaptureBudget
    b = CaptureBudget(1000)
    t1 = CallTrace(method="POST", path="/x", t_received=0, budget=b)
    t2 = CallTrace(method="POST", path="/x", t_received=0, budget=b)
    t1.add_req_chunk(b"a" * 700)
    t2.add_req_chunk(b"b" * 700)                                  # only 300 left: refused
    assert t2.req_truncated and not t1.req_truncated
    t1.release()
    t3 = CallTrace(method="POST", path="/x", t_received=0, budget=b)
    t3.add_req_chunk(b"c" * 700)                                  # released bytes are reusable
    assert not t3.req_truncated


def test_dropped_records_return_their_budget(tmp_path):
    from tux_trace.record import CaptureBudget
    from tux_trace.tee import Tee
    tee = Tee(maxsize=1, data_dir=tmp_path, start_writer=False)
    budget = CaptureBudget(10**9)
    traces = []
    for _ in range(5):
        tr = CallTrace(method="POST", path="/x", t_received=0, budget=budget)
        tr.add_req_chunk(b"z" * 1000)
        traces.append(tr)
        tee.submit(tr)
    assert tee.dropped_records == 4
    assert budget.used == 1000                                   # only the one still queued is charged
    traces[0].release()
    assert budget.used == 0


def test_budget_is_back_to_baseline_after_real_traffic(proxy_factory):
    proxy, _, _ = proxy_factory(SSE)
    base = BUDGET.used
    for _ in range(5):
        httpx.post(proxy + "/v1/messages", content=b'{"model":"m","messages":[]}')
    assert wait_for(lambda: len(records(proxy_factory)) == 5)
    assert wait_for(lambda: BUDGET.used == base), f"leaked {BUDGET.used - base} bytes"


# ================================================================== sampling, /metrics, upstream failure
def test_sample_rate_zero_records_nothing_but_still_relays(proxy_factory):
    proxy, _, _ = proxy_factory(SSE, sample_rate=0.0)
    base = BUDGET.used
    for _ in range(5):
        r = httpx.post(proxy + "/v1/messages", content=b'{"model":"m"}')
        assert r.status_code == 200 and r.content == b"".join(SSE)
    time.sleep(0.5)
    assert records(proxy_factory) == []
    st = httpx.get(proxy + "/stats").json()
    assert st["calls_sampled_out"] == 5 and BUDGET.used == base


def test_sample_rate_is_roughly_honoured(proxy_factory):
    proxy, _, _ = proxy_factory([b"data: x\n\n"], sample_rate=0.5)
    with httpx.Client() as c:
        for _ in range(120):
            c.post(proxy + "/v1/messages", content=b"{}")
    time.sleep(1.0)
    n = len(records(proxy_factory))
    assert 30 <= n <= 90, n


def test_metrics_endpoint_is_valid_prometheus_text(proxy_factory):
    proxy, _, _ = proxy_factory(SSE)
    httpx.post(proxy + "/v1/messages", content=b'{"model":"m"}')
    time.sleep(0.4)
    r = httpx.get(proxy + "/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    text = r.text
    assert "# TYPE tux_trace_calls_total counter" in text and "# TYPE tux_trace_queue_depth gauge" in text
    for line in text.strip().splitlines():
        if not line.startswith("#"):
            name, value = line.split(" ")
            assert name.startswith("tux_trace_") and float(value) == float(value)
    assert "None" not in text and "null" not in text           # null stats are omitted, not printed


def test_upstream_dying_mid_stream_truncates_for_the_client_and_is_recorded(tmp_path):
    """The stub sends part of a stream then drops the connection. The client must see exactly the
    bytes that were sent (no invented ending) and the record must say the call was cut short."""
    import socket
    import threading

    from conftest import Server, free_port
    from tux_trace.app import create_app
    from tux_trace.config import Config

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]
    part = SSE[0] + SSE[1]

    def serve():
        conn, _ = srv.accept()
        conn.recv(65536)
        head = (b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
        conn.sendall(head + b"%x\r\n" % len(part) + part + b"\r\n")
        time.sleep(0.2)
        conn.close()                                              # no terminating 0-chunk: an abrupt death
    threading.Thread(target=serve, daemon=True).start()
    cfg = Config(upstream={"base_url": f"http://127.0.0.1:{port}"}, data_dir=str(tmp_path))
    with Server(create_app(cfg), free_port()) as px:
        got = b""
        try:
            with httpx.stream("POST", f"http://127.0.0.1:{px.port}/v1/messages", content=b"{}", timeout=10) as r:
                for chunk in r.iter_raw():
                    got += chunk
        except httpx.HTTPError:
            pass                                                   # the client sees the truncation as a protocol error
        assert got in (part, b"") or part.startswith(got) or got.startswith(part[:20])
        assert wait_for(lambda: any((tmp_path / "calls").glob("*.jsonl")))
        rec = json.loads(next((tmp_path / "calls").glob("*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
    assert rec["outcome"]["incomplete"] is True
    assert rec["response"]["status"] == 200


# ================================================================== smoke command
def test_smoke_reports_failure_clearly_when_nothing_is_listening(capsys):
    from tux_trace.__main__ import main
    with pytest.raises(SystemExit) as e:
        main(["smoke", "--url", "http://127.0.0.1:1", "--config", "config/config.yaml"])
    assert e.value.code == 2 and "Is `tux-trace serve` running" in capsys.readouterr().out


def test_smoke_passes_against_a_real_proxy(proxy_factory, capsys, monkeypatch):
    from tux_trace.__main__ import main
    proxy, _, _ = proxy_factory(SSE)

    class C:                                              # point smoke at the temp data dir
        data_dir = str(proxy_factory.data_dir)
        class listen:
            host, port = "127.0.0.1", 0
    import tux_trace.__main__ as m
    monkeypatch.setattr(m, "_cfg", lambda args: C)
    with pytest.raises(SystemExit) as e:
        main(["smoke", "--url", proxy])
    out = capsys.readouterr().out
    assert e.value.code in (0, 1)                         # the stub only serves /v1/messages
    assert "record anthropic" in out and "sent   anthropic" in out


@pytest.mark.skipif(os.name != "nt", reason="directory junctions are a Windows feature")
def test_purge_does_not_delete_through_a_directory_junction(tmp_path):
    """content/ replaced by a junction to some other folder holding old files must be left alone."""
    import subprocess
    other = tmp_path / "other_folder"
    other.mkdir()
    victim = other / (str(uuid.uuid4()) + ".json")       # matches our naming pattern AND is ancient
    victim.write_text("precious")
    os.utime(victim, (1, 1))
    data = tmp_path / "data"
    (data / "calls").mkdir(parents=True)
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(data / "content"), str(other)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("could not create a junction here")
    purge(data, 1)
    assert victim.exists(), "purge followed a junction out of the data directory"
