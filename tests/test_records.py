"""M1: one record per call with transport facts, bounded queue, fail-open recording."""
import json
import queue
import socket
import time

import httpx

from trajectory.record import CallTrace
from trajectory.tee import Tee

SSE = [b"data: 1\n\n", b"data: 2\n\n", b"data: [DONE]\n\n"]


def wait_for(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def read_records(build):
    out = []
    for f in sorted((build.data_dir / "calls").glob("*.jsonl")):
        out += [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line]
    return out


def test_one_record_per_call_with_transport_facts(proxy_factory):
    proxy, _, _ = proxy_factory(SSE)
    for path in ("/v1/messages", "/v1/messages", "/v1/messages"):
        httpx.post(proxy + path, content=b'{"model":"m"}',
                   headers={"x-api-key": "sk-secret-value-123", "user-agent": "claude-cli/2.0"})
    assert wait_for(lambda: len(read_records(proxy_factory)) == 3)
    r = read_records(proxy_factory)[0]
    assert r["request"]["format"] == "anthropic" and r["request"]["path"] == "/v1/messages"
    assert r["request"]["body_bytes"] == len(b'{"model":"m"}')
    assert r["response"]["status"] == 200
    assert r["response"]["content_bytes"] == sum(map(len, SSE))
    assert r["timing"]["chunk_count"] >= 1
    assert r["timing"]["t_sent"] >= r["timing"]["t_received"]
    assert r["timing"]["duration_ms"] is not None and r["timing"]["duration_ms"] >= 0
    assert r["timing"]["total_ms"] is None        # no content deltas in this stub stream
    assert r["outcome"]["error"] is None
    assert r["client"]["harness_hint"] == "claude-code"
    # unknown-yet fields are null / not-reported, never fabricated zeros
    assert r["timing"]["ttft_any_ms"] is None
    assert r["usage"]["cache_write"] == {"value": None, "reported": False}
    assert len({x["call_id"] for x in read_records(proxy_factory)}) == 3
    assert httpx.get(proxy + "/stats").json()["calls_total"] == 3


def test_no_credentials_on_disk(proxy_factory):
    proxy, _, _ = proxy_factory(SSE)
    httpx.post(proxy + "/v1/messages", content=b"{}",
               headers={"x-api-key": "sk-secret-value-123", "authorization": "Bearer tok-abcdef-123456",
                        "cookie": "session=hunter2"})
    assert wait_for(lambda: len(read_records(proxy_factory)) == 1)
    raw = "".join(f.read_text(encoding="utf-8") for f in (proxy_factory.data_dir / "calls").glob("*.jsonl"))
    for secret in ("sk-secret-value-123", "tok-abcdef-123456", "hunter2"):
        assert secret not in raw
    assert len(read_records(proxy_factory)[0]["client"]["key_hash"]) == 64


def test_unknown_path_is_relayed_and_recorded_as_unknown(proxy_factory):
    proxy, _, _ = proxy_factory(SSE)
    r = httpx.get(proxy + "/some/other")
    assert r.status_code == 418
    assert wait_for(lambda: len(read_records(proxy_factory)) == 1)
    assert read_records(proxy_factory)[0]["request"]["format"] == "unknown"


def test_client_disconnect_is_recorded_incomplete(proxy_factory):
    chunks = [b"data: %d\n\n" % i for i in range(5)]
    proxy, _, _ = proxy_factory(chunks, delay=0.5)
    port = int(proxy.rsplit(":", 1)[1])
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}")
    s.recv(4096)          # first bytes arrive
    s.close()             # ...then the client vanishes mid-stream
    assert wait_for(lambda: len(read_records(proxy_factory)) == 1, 8)
    o = read_records(proxy_factory)[0]["outcome"]
    assert o["incomplete"] is True and o["client_disconnected"] is True
    assert httpx.get(proxy + "/stats").json()["calls_incomplete"] == 1


def test_upstream_down_is_recorded_as_error():
    from starlette.testclient import TestClient

    from trajectory.app import create_app
    from trajectory.config import Config
    import tempfile
    d = tempfile.mkdtemp()
    with TestClient(create_app(Config(upstream={"base_url": "http://127.0.0.1:1"}, data_dir=d))) as c:
        assert c.post("/v1/messages", content=b"{}").status_code == 502
    recs = [json.loads(line) for f in (__import__("pathlib").Path(d) / "calls").glob("*.jsonl")
            for line in f.read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 1 and recs[0]["outcome"]["error"]["status"] == 502
    assert recs[0]["response"]["status"] == 502


def test_queue_full_drops_and_counts_without_blocking(tmp_path):
    tee = Tee(maxsize=5, data_dir=tmp_path, start_writer=False)   # nothing drains it
    t0 = time.perf_counter()
    results = [tee.submit({"i": i}) for i in range(1000)]
    elapsed = time.perf_counter() - t0
    assert sum(results) == 5 and tee.dropped_records == 995
    assert elapsed < 0.5, "submit() must never block"


def test_burst_through_real_proxy_with_tiny_queue_still_relays_everything(proxy_factory):
    proxy, _, _ = proxy_factory(SSE, queue={"maxsize": 1})
    with httpx.Client() as c:
        bodies = [c.post(proxy + "/v1/messages", content=b"{}").content for _ in range(50)]
    assert all(b == b"".join(SSE) for b in bodies)            # relaying unaffected
    st = httpx.get(proxy + "/stats").json()
    assert st["calls_total"] == 50
    assert st["records_written"] + st["dropped_records"] == 50


def test_failopen_when_recording_is_broken(proxy_factory, monkeypatch):
    proxy, _, _ = proxy_factory(SSE)
    monkeypatch.setattr(CallTrace, "to_record", lambda self: 1 / 0)          # writer-side bug
    r1 = httpx.post(proxy + "/v1/messages", content=b"{}")
    import trajectory.proxy as px
    monkeypatch.setattr(px, "detect_format", lambda p: 1 / 0)                # request-side bug
    r2 = httpx.post(proxy + "/v1/messages", content=b"{}")
    for r in (r1, r2):
        assert r.status_code == 200 and r.content == b"".join(SSE)
    assert wait_for(lambda: httpx.get(proxy + "/stats").json()["record_errors"] >= 1)


def test_writer_survives_unwritable_disk(tmp_path):
    q = queue.Queue()
    from trajectory.store import JsonlWriter
    w = JsonlWriter(q, tmp_path)
    w.start()
    (tmp_path / "calls").rmdir()
    (tmp_path / "calls").write_text("now a file, not a dir")   # opening the day file will fail
    q.put({"timing": {"received_at": "2026-01-01T00:00:00"}})
    assert wait_for(lambda: w.write_errors == 1)
    assert w._thread.is_alive()
