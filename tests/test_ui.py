"""Read API + UI smoke tests."""
import time

import httpx

SSE = [b'event: message_start\ndata: {"type":"message_start","message":{"model":"m","usage":{"input_tokens":3}}}\n\n',
       b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n',
       b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n',
       b'event: message_stop\ndata: {"type":"message_stop"}\n\n']


def wait_calls(proxy, n, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        d = httpx.get(proxy + "/_trace/api/calls").json()
        if d["total"] >= n:
            return d
        time.sleep(0.1)
    raise AssertionError("records never appeared")


def test_ui_and_static_are_served(proxy_factory):
    proxy, _, _ = proxy_factory(SSE)
    page = httpx.get(proxy + "/_trace")
    assert page.status_code == 200 and "tux-trace" in page.text
    assert httpx.get(proxy + "/_trace/static/app.js").status_code == 200
    assert httpx.get(proxy + "/_trace/static/style.css").status_code == 200


def test_calls_api_lists_and_details_a_call(proxy_factory):
    proxy, _, _ = proxy_factory(SSE)
    httpx.post(proxy + "/v1/messages", content=b'{"model":"m","stream":true}')
    d = wait_calls(proxy, 1)
    row = d["calls"][0]
    assert row["format"] == "anthropic" and row["stop_reason"] == "end_turn"
    assert row["usage"]["output"] == {"value": 2, "reported": True}
    full = httpx.get(proxy + f"/_trace/api/calls/{row['call_id']}").json()
    assert full["call_id"] == row["call_id"]
    assert httpx.get(proxy + "/_trace/api/calls/nope").status_code == 404
    assert httpx.get(proxy + "/_trace/api/calls?limit=abc").status_code == 400


def test_trace_paths_do_not_break_relaying_other_paths(proxy_factory):
    proxy, _, _ = proxy_factory(SSE)
    assert httpx.get(proxy + "/some/other").status_code == 418      # still relayed to upstream
