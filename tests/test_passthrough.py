import hashlib
import time

import httpx

SSE = [b"event: message_start\ndata: {\"a\":1}\n\n", b": ping\n\n",
       b"event: content_block_delta\r\ndata: {\"b\":2}\r\n\r\n", b"data: [DONE]\n\n"]


def test_bytes_identical(proxy_factory):
    proxy, _, log = proxy_factory(SSE)
    with httpx.Client(trust_env=False) as c:
        r = c.post(proxy + "/v1/messages", content=b'{"k":  1 }',
                   headers={"authorization": "Bearer placeholder", "content-type": "application/json"})
    assert hashlib.sha256(r.content).digest() == hashlib.sha256(b"".join(SSE)).digest()
    assert r.headers["x-upstream-id"] == "abc123"
    assert log[0]["body"] == b'{"k":  1 }'                 # request body untouched
    assert log[0]["headers"]["authorization"] == "Bearer placeholder"
    assert "x-tux-trace-id" in log[0]["headers"]            # the one added header


def test_trace_header_can_be_disabled(proxy_factory):
    proxy, _, log = proxy_factory(SSE, add_trace_header=False)
    httpx.post(proxy + "/v1/messages", content=b"{}")
    assert "x-tux-trace-id" not in log[0]["headers"]


def test_status_and_headers_relayed_verbatim(proxy_factory):
    proxy, _, _ = proxy_factory()
    r = httpx.get(proxy + "/anything/else")
    assert r.status_code == 418 and r.content == b"hello" and r.headers["x-custom"] == "1"


def test_no_buffering(proxy_factory):
    proxy, _, _ = proxy_factory([b"data: 1\n\n", b"data: 2\n\n", b"data: 3\n\n"], delay=1.0)
    t0 = time.monotonic()
    stamps = []
    with httpx.stream("POST", proxy + "/v1/messages", content=b"{}") as r:
        for _ in r.iter_raw():
            stamps.append(time.monotonic() - t0)
    assert stamps[0] < 0.5, f"first chunk arrived after {stamps[0]:.2f}s (buffered?)"
    assert stamps[-1] > 1.5


def test_upstream_down_gives_502():
    from starlette.testclient import TestClient

    from tux_trace.app import create_app
    from tux_trace.config import Config
    app = create_app(Config(upstream={"base_url": "http://127.0.0.1:1"}))
    with TestClient(app) as c:
        assert c.post("/v1/messages", content=b"{}").status_code == 502
        h = c.get("/healthz").json()
        assert h["status"] == "degraded" and h["upstream_reachable"] is False


def test_healthz_ok(proxy_factory):
    proxy, _, _ = proxy_factory()
    h = httpx.get(proxy + "/healthz").json()
    assert h["status"] == "ok" and h["upstream_reachable"] is True


def test_request_framing_headers_reach_upstream(proxy_factory):
    """Upstream must see a well-formed request whose body matches what the client sent."""
    proxy, _, log = proxy_factory(SSE)
    big = b"x" * 300_000
    httpx.post(proxy + "/v1/messages", content=big, headers={"content-type": "application/json"})
    h = log[0]["headers"]
    assert log[0]["body"] == big
    assert h["content-type"] == "application/json"
    assert not ("content-length" in h and "transfer-encoding" in h), h  # both = smuggling risk
