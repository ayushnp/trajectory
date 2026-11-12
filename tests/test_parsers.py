"""M2: SSE reader, per-format parsers, usage mapping and timing. Uses real captured Tux streams."""
import json
from pathlib import Path

import pytest

from tux_trace import usage as U
from tux_trace.parsers import make_parser
from tux_trace.parsers.sse import SSEReader
from tux_trace.record import CallTrace

FIX = Path(__file__).parent / "fixtures"
MS = 1_000_000


def parse(fmt, raw: bytes, chunker=None, t0=0, step=10 * MS):
    """Feed raw SSE bytes through the real reader+parser; chunk i arrives at t0 + i*step."""
    p, rd = make_parser(fmt), SSEReader()
    chunks = chunker(raw) if chunker else [raw]
    for i, c in enumerate(chunks):
        for ev in rd.feed(c, t0 + i * step):
            p.feed(ev)
    for ev in rd.flush(t0 + len(chunks) * step):
        p.feed(ev)
    return p.finish()


def lines_as_chunks(raw):
    return [raw[i:i + 7] for i in range(0, len(raw), 7)]   # hostile: split mid-token


# ---------------------------------------------------------------- SSE reader
def test_sse_reader_handles_crlf_comments_and_split_lines():
    raw = b": ping\r\n\r\nevent: a\r\ndata: {\"x\":1}\r\n\r\nevent: b\ndata: l1\ndata: l2\n\n"
    for chunker in (lambda r: [r], lines_as_chunks):
        rd, out = SSEReader(), []
        for c in chunker(raw):
            out += rd.feed(c, 5)
        assert [(e.event, e.data) for e in out] == [("a", '{"x":1}'), ("b", "l1\nl2")]


def test_sse_reader_flushes_unterminated_final_event():
    rd = SSEReader()
    assert rd.feed(b"data: {\"a\":1}", 1) == []
    assert [e.data for e in rd.flush(2)] == ['{"a":1}']


# ---------------------------------------------------------------- anthropic
def test_anthropic_real_stream():
    r = parse("anthropic", (FIX / "anthropic/simple_text.sse").read_bytes())
    assert r.model_served == "claude-haiku-5-5" and r.stop_reason == "end_turn" and r.terminated
    assert r.usage["input"] == 16 and r.usage["output"] == 10      # last value, not 5+10
    assert r.event_errors == 0 and r.unknown_events == 0
    assert r.t_first_visible is not None


def test_anthropic_survives_arbitrary_chunking():
    raw = (FIX / "anthropic/simple_text.sse").read_bytes()
    a, b = parse("anthropic", raw), parse("anthropic", raw, lines_as_chunks)
    assert (a.usage, a.stop_reason, a.model_served) == (b.usage, b.stop_reason, b.model_served)


def test_anthropic_cache_counters_and_input_is_inclusive():
    miss = parse("anthropic", (FIX / "anthropic/cache_miss.sse").read_bytes())
    hit = parse("anthropic", (FIX / "anthropic/cache_hit.sse").read_bytes())
    assert miss.usage["cache_write"] == 11033 and miss.usage["cache_read"] == 0
    assert hit.usage["cache_read"] == 11033 and hit.usage["cache_write"] == 0
    c = U.canonical("anthropic", hit.usage)
    assert c["input_total"]["value"] == 13 + 11033       # inclusive, unlike the raw number
    assert c["raw"]["input"] == 13
    assert c["cache_read"] == {"value": 11033, "reported": True}


def test_anthropic_tool_use_whole_and_streamed_and_thinking():
    def ev(t, o):
        return f"event: {t}\ndata: {json.dumps(o)}\n\n".encode()
    whole = (ev("message_start", {"type": "message_start", "message": {"model": "m", "usage": {"input_tokens": 1}}})
             + ev("content_block_start", {"type": "content_block_start", "index": 0,
                  "content_block": {"type": "thinking", "thinking": ""}})
             + ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                  "delta": {"type": "thinking_delta", "thinking": "hmm"}})
             + ev("content_block_start", {"type": "content_block_start", "index": 1,
                  "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"p": "a"}}})
             + ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                  "usage": {"output_tokens": 9}})
             + ev("message_stop", {"type": "message_stop"}))
    r = parse("anthropic", whole, step=MS)
    assert r.tool_calls == [{"id": "toolu_1", "name": "Read", "args": '{"p":"a"}'}]
    assert r.has_thinking and r.thinking_visibility == "summary"
    assert r.t_first_visible is None and r.t_first_content is not None   # thinking is not "visible"

    streamed = (ev("content_block_start", {"type": "content_block_start", "index": 0,
                   "content_block": {"type": "tool_use", "id": "t9", "name": "Bash", "input": {}}})
                + ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                     "delta": {"type": "input_json_delta", "partial_json": '{"cmd":'}})
                + ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                     "delta": {"type": "input_json_delta", "partial_json": '"ls"}'}}))
    assert parse("anthropic", streamed).tool_calls == [{"id": "t9", "name": "Bash", "args": '{"cmd":"ls"}'}]


def test_anthropic_unknown_and_garbage_events_are_not_fatal():
    raw = (b"event: weird\ndata: {\"type\":\"weird\"}\n\n" b"data: {not json\n\n"
           b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n")
    r = parse("anthropic", raw)
    assert r.unknown_events == 1 and r.event_errors == 1 and r.stop_reason == "end_turn"


def test_anthropic_midstream_error_event():
    raw = b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"busy"}}\n\n'
    r = parse("anthropic", raw)
    assert r.error == {"type": "overloaded_error", "message": "busy"} and not r.terminated


# ---------------------------------------------------------------- openai chat
def test_openai_chat_real_stream():
    r = parse("openai_chat", (FIX / "openai_chat/simple_text.sse").read_bytes())
    assert r.model_served == "gpt-6-luna" and r.stop_reason == "stop" and r.terminated
    c = U.canonical("openai_chat", r.usage)
    assert c["input_total"] == {"value": 12, "reported": True}
    assert c["output"] == {"value": 34, "reported": True}
    assert c["cache_read"] == {"value": 0, "reported": True}      # reported zero is real
    assert c["cache_write"] == {"value": None, "reported": False}  # never reported by this API
    assert r.t_first_visible is not None


def test_openai_chat_without_include_usage_reports_unknown_not_zero():
    raw = (b'data: {"model":"m","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
           b'data: {"model":"m","choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    r = parse("openai_chat", raw)
    c = U.canonical("openai_chat", r.usage)
    assert all(c[k] == {"value": None, "reported": False}
               for k in ("input_total", "output", "cache_read", "cache_write"))


def test_openai_chat_tool_call_fragments_concatenate_by_index():
    def d(o):
        return f"data: {json.dumps(o)}\n\n".encode()
    raw = (d({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": ""}}]}}]})
           + d({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"a":'}}]}}]})
           + d({"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "c2", "function": {"name": "g", "arguments": "{}"}}]}}]})
           + d({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}}]})
           + b"data: [DONE]\n\n")
    r = parse("openai_chat", raw)
    assert r.tool_calls == [{"id": "c1", "name": "f", "args": '{"a":1}'}, {"id": "c2", "name": "g", "args": "{}"}]


# ---------------------------------------------------------------- openai responses
def test_openai_responses_real_stream():
    r = parse("openai_responses", (FIX / "openai_responses/simple_text.sse").read_bytes())
    assert r.model_served == "gpt-6-luna" and r.terminated and r.stop_reason == "completed"
    assert r.has_thinking and r.thinking_visibility == "encrypted"
    c = U.canonical("openai_responses", r.usage)
    assert c["input_total"] == {"value": 12, "reported": True}
    assert c["output"] == {"value": 31, "reported": True}
    assert c["cache_write"] == {"value": 0, "reported": True}       # Tux does report this one
    assert r.usage["reasoning"] == 20
    assert r.unknown_events == 0 and r.event_errors == 0


def test_openai_responses_function_call():
    def ev(t, o):
        return f"event: {t}\ndata: {json.dumps({'type': t, **o})}\n\n".encode()
    raw = (ev("response.output_item.added", {"output_index": 0, "item": {"type": "function_call", "id": "fc_1",
                                              "call_id": "call_1", "name": "Read", "arguments": ""}})
           + ev("response.function_call_arguments.delta", {"item_id": "fc_1", "delta": '{"p":'})
           + ev("response.function_call_arguments.delta", {"item_id": "fc_1", "delta": '"x"}'})
           + ev("response.completed", {"response": {"model": "m", "output": [{"type": "function_call"}],
                                                    "usage": {"input_tokens": 3, "output_tokens": 4}}}))
    r = parse("openai_responses", raw)
    assert r.tool_calls == [{"id": "call_1", "name": "Read", "args": '{"p":"x"}'}]
    assert r.stop_reason == "tool_calls"


# ---------------------------------------------------------------- timing
def test_ttft_from_known_chunk_arrival_times():
    """Three chunks arriving at +100/+300/+900 ms: first text at 300 ms, last content at 900 ms."""
    def ev(t, o):
        return f"event: {t}\ndata: {json.dumps(o)}\n\n".encode()
    c1 = ev("message_start", {"type": "message_start", "message": {"model": "m", "usage": {"input_tokens": 1}}})
    c2 = ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                    "delta": {"type": "text_delta", "text": "a"}})
    c3 = ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                    "delta": {"type": "text_delta", "text": "b"}}) + \
        ev("message_stop", {"type": "message_stop"})
    from tux_trace import timing
    p, rd = make_parser("anthropic"), SSEReader()
    times = [1_000 * MS + 100 * MS, 1_000 * MS + 300 * MS, 1_000 * MS + 900 * MS]
    for c, t in zip((c1, c2, c3), times):
        for e in rd.feed(c, t):
            p.feed(e)
    r = p.finish()
    d = timing.derive(1_000 * MS, r.t_first_content, r.t_first_visible, r.t_last_content, times)
    assert d["ttft_any_ms"] == 300.0 and d["ttft_visible_ms"] == 300.0 and d["total_ms"] == 900.0
    assert d["inter_chunk_p50_ms"] == 200.0 and d["inter_chunk_p95_ms"] == 600.0


def test_timing_marks_that_never_happened_are_null_not_zero():
    from tux_trace import timing
    d = timing.derive(0, None, None, None, [5 * MS])
    assert d["ttft_any_ms"] is None and d["ttft_visible_ms"] is None and d["total_ms"] is None
    assert d["inter_chunk_p50_ms"] is None


# ---------------------------------------------------------------- end to end through CallTrace
def make_trace(fmt, raw, ctype="text/event-stream", encoding=None, step=10 * MS, req=b""):
    tr = CallTrace(method="POST", path="/x", t_received=0, fmt=fmt)
    tr.content_type, tr.content_encoding = ctype, encoding
    for c in (req and [req]) or []:
        tr.add_req_chunk(c)
    for i in range(0, len(raw), 200):
        tr.add_resp_chunk((i // 200 + 1) * step, raw[i:i + 200])
    tr.status, tr.completed, tr.t_end = 200, True, 10_000 * MS
    return tr


def test_calltrace_builds_full_record_from_real_anthropic_stream():
    body = json.dumps({"model": "claude-haiku-5-5", "stream": True, "system": "s",
                       "messages": [{"role": "user", "content": "hi"}], "tools": [{}, {}]}).encode()
    tr = make_trace("anthropic", (FIX / "anthropic/simple_text.sse").read_bytes(), req=body)
    rec = tr.to_record()
    assert rec["request"]["model_requested"] == "claude-haiku-5-5" and rec["request"]["stream"] is True
    assert rec["request"]["message_count"] == 1 and rec["request"]["tool_count"] == 2
    assert rec["request"]["system_prompt_sha256"]
    assert rec["response"]["model_served"] == "claude-haiku-5-5"
    assert rec["response"]["stop_reason"] == "end_turn"
    assert rec["usage"]["input_total"] == {"value": 16, "reported": True}
    assert rec["usage"]["output"] == {"value": 10, "reported": True}
    assert rec["timing"]["ttft_any_ms"] > 0 and rec["timing"]["ttft_visible_ms"] > 0
    assert rec["outcome"]["incomplete"] is False
    assert tr.resp_chunks == [] and not tr.req_body    # captured bytes released


def test_truncated_stream_is_incomplete_even_if_transport_finished():
    raw = (FIX / "anthropic/simple_text.sse").read_bytes()
    raw = raw[:raw.index(b"event: message_stop")]                 # terminator missing
    rec = make_trace("anthropic", raw).to_record()
    assert rec["outcome"]["incomplete"] is True
    assert rec["usage"]["output"]["reported"] is True             # what we did see is kept


def test_http_error_response_is_recorded_as_error():
    raw = b'{"type":"error","error":{"type":"invalid_request_error","message":"bad model"}}'
    tr = make_trace("anthropic", raw, ctype="application/json")
    tr.status = 400
    rec = tr.to_record()
    assert rec["outcome"]["error"]["type"] == "invalid_request_error" and rec["outcome"]["error"]["status"] == 400
    assert rec["usage"]["input_total"]["reported"] is False


def test_gzip_encoded_stream_is_decoded_for_parsing_only():
    import gzip
    raw = (FIX / "anthropic/simple_text.sse").read_bytes()
    rec = make_trace("anthropic", gzip.compress(raw), encoding="gzip").to_record()
    assert rec["usage"]["output"] == {"value": 10, "reported": True}


def test_non_stream_json_response():
    raw = json.dumps({"model": "m", "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 6},
                      "content": []}).encode()
    rec = make_trace("anthropic", raw, ctype="application/json").to_record()
    assert rec["usage"]["output"] == {"value": 6, "reported": True} and rec["response"]["stop_reason"] == "end_turn"


def test_garbage_bodies_never_raise():
    for ctype in ("text/event-stream", "application/json", None):
        rec = make_trace("anthropic", b"\xff\xfe not valid at all {{{", ctype=ctype, req=b"\x00\x01").to_record()
        assert rec["call_id"]


@pytest.mark.parametrize("fmt", ["anthropic", "openai_chat", "openai_responses"])
def test_fixture_chunking_invariance(fmt):
    raw = (FIX / fmt / "simple_text.sse").read_bytes()
    a, b = parse(fmt, raw), parse(fmt, raw, lines_as_chunks)
    assert (a.usage, a.stop_reason, a.model_served, a.terminated) == (b.usage, b.stop_reason, b.model_served, b.terminated)
