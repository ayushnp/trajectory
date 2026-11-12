"""M4: tool call <-> result pairing, gap_ms, result sizes."""
import json

from tux_trace.chain import digest_request
from tux_trace.tools import ToolPairer, extract_results, join_results

S = 1_000_000_000


def call_rec(call_id, t_received, t_end, tool_calls, key="k"):
    return {"call_id": call_id, "client": {"key_hash": key},
            "timing": {"t_received": t_received * S, "t_end": t_end * S},
            "request": {}, "response": {"tool_calls": [dict(tc) for tc in tool_calls]}}


def pair(pairer, rec, fmt, body):
    pairer.attach(rec, extract_results(fmt, body))
    pairer.register(rec)


def anth_body(*results, extra_text=None):
    content = [{"type": "tool_result", **r} for r in results]
    if extra_text:
        content.append({"type": "text", "text": extra_text})
    return {"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "content": []},
                         {"role": "user", "content": content}]}


# ---------------------------------------------------------------- core behaviour
def test_result_in_next_request_pairs_with_gap_greater_than_zero():
    p = ToolPairer()
    r1 = call_rec("c1", 0, 2, [{"id": "toolu_1", "name": "Read", "args": "{}"}])
    pair(p, r1, "anthropic", {"messages": [{"role": "user", "content": "go"}]})
    r2 = call_rec("c2", 5, 7, [])
    pair(p, r2, "anthropic", anth_body({"tool_use_id": "toolu_1", "content": "file contents"}))
    res = r2["request"]["tool_results"][0]
    assert res["parent_call_id"] == "c1" and res["tool_call_id"] == "toolu_1"
    assert res["gap_ms"] == 3000.0 and res["gap_ms"] > 0          # 5 s request - 2 s parent end
    assert res["result_bytes"] == len("file contents")


def test_join_folds_result_onto_parent_and_marks_unanswered_calls():
    p = ToolPairer()
    r1 = call_rec("c1", 0, 2, [{"id": "a", "name": "Read", "args": "{}"}, {"id": "b", "name": "Bash", "args": "{}"}])
    pair(p, r1, "anthropic", {"messages": []})
    r2 = call_rec("c2", 4, 6, [])
    pair(p, r2, "anthropic", anth_body({"tool_use_id": "a", "content": "xx"}))      # b never answered
    join_results([r1, r2])
    a, b = r1["response"]["tool_calls"]
    assert a["result_present"] is True and a["result_bytes"] == 2 and a["gap_ms"] == 2000.0
    assert a["result_call_id"] == "c2"
    assert b["result_present"] is False and b["gap_ms"] is None and b["result_bytes"] is None


def test_parallel_tool_calls_all_pair_in_one_request():
    p = ToolPairer()
    r1 = call_rec("c1", 0, 1, [{"id": f"t{i}", "name": "Read", "args": "{}"} for i in range(3)])
    pair(p, r1, "anthropic", {"messages": []})
    r2 = call_rec("c2", 2, 3, [])
    pair(p, r2, "anthropic", anth_body(*({"tool_use_id": f"t{i}", "content": "x" * (i + 1)} for i in range(3))))
    join_results([r1, r2])
    assert [tc["result_bytes"] for tc in r1["response"]["tool_calls"]] == [1, 2, 3]
    assert all(tc["result_present"] for tc in r1["response"]["tool_calls"])


def test_error_flag_is_captured():
    p = ToolPairer()
    r1 = call_rec("c1", 0, 1, [{"id": "t", "name": "Bash", "args": "{}"}])
    pair(p, r1, "anthropic", {"messages": []})
    r2 = call_rec("c2", 2, 3, [])
    pair(p, r2, "anthropic", anth_body({"tool_use_id": "t", "content": "boom", "is_error": True}))
    join_results([r1, r2])
    assert r1["response"]["tool_calls"][0]["result_is_error"] is True


def test_resent_result_in_later_steps_does_not_create_a_new_gap():
    """Every later request repeats the whole history, including old tool results."""
    p = ToolPairer()
    r1 = call_rec("c1", 0, 1, [{"id": "t", "name": "Read", "args": "{}"}])
    pair(p, r1, "anthropic", {"messages": []})
    r2 = call_rec("c2", 3, 4, [])
    pair(p, r2, "anthropic", anth_body({"tool_use_id": "t", "content": "x"}))
    r3 = call_rec("c3", 100, 101, [])
    pair(p, r3, "anthropic", anth_body({"tool_use_id": "t", "content": "x"}))
    assert r2["request"]["tool_results"][0]["gap_ms"] == 2000.0
    assert r3["request"]["tool_results"][0]["repeat"] is True and r3["request"]["tool_results"][0]["gap_ms"] is None
    join_results([r1, r2, r3])
    assert r1["response"]["tool_calls"][0]["gap_ms"] == 2000.0          # first answer wins


def test_result_for_unknown_call_is_counted_not_fabricated():
    p = ToolPairer()
    r = call_rec("c9", 0, 1, [])
    pair(p, r, "anthropic", anth_body({"tool_use_id": "never-seen", "content": "x"}))
    res = r["request"]["tool_results"][0]
    assert res["parent_call_id"] is None and res["gap_ms"] is None and p.results_unmatched == 1


def test_clients_cannot_claim_each_others_tool_calls():
    p = ToolPairer()
    r1 = call_rec("c1", 0, 1, [{"id": "same-id", "name": "Read", "args": "{}"}], key="alice")
    pair(p, r1, "anthropic", {"messages": []})
    r2 = call_rec("c2", 2, 3, [], key="bob")
    pair(p, r2, "anthropic", anth_body({"tool_use_id": "same-id", "content": "x"}))
    assert r2["request"]["tool_results"][0]["parent_call_id"] is None


def test_missing_parent_t_end_gives_null_gap_not_zero():
    p = ToolPairer()
    r1 = call_rec("c1", 0, 1, [{"id": "t", "name": "Read", "args": "{}"}])
    r1["timing"]["t_end"] = None
    pair(p, r1, "anthropic", {"messages": []})
    r2 = call_rec("c2", 2, 3, [])
    pair(p, r2, "anthropic", anth_body({"tool_use_id": "t", "content": "x"}))
    assert r2["request"]["tool_results"][0]["gap_ms"] is None


def test_pending_memory_is_bounded_and_expires():
    p = ToolPairer(max_pending=10, ttl_s=60)
    for i in range(50):
        p.register(call_rec(f"c{i}", i, i, [{"id": f"t{i}", "name": "x", "args": ""}]))
    assert len(p._pending) <= 10
    p.expire(10_000 * S)
    assert len(p._pending) == 0


# ---------------------------------------------------------------- result extraction per format
def test_anthropic_result_sizes_for_string_list_and_missing_content():
    body = anth_body({"tool_use_id": "a", "content": "héllo"},
                     {"tool_use_id": "b", "content": [{"type": "text", "text": "abc"}, {"type": "text", "text": "de"}]},
                     {"tool_use_id": "c"})
    r = extract_results("anthropic", body)
    assert [x["bytes"] for x in r] == [len("héllo".encode()), 5, 0]


def test_openai_chat_and_responses_results():
    chat = {"messages": [{"role": "tool", "tool_call_id": "c1", "content": "12345"}]}
    resp = {"input": [{"type": "function_call_output", "call_id": "c2", "output": "xyz"}]}
    assert extract_results("openai_chat", chat) == [{"id": "c1", "bytes": 5, "is_error": None}]
    assert extract_results("openai_responses", resp) == [{"id": "c2", "bytes": 3, "is_error": None}]
    assert extract_results("unknown", chat) == []


def test_result_content_itself_is_never_kept():
    r = extract_results("anthropic", anth_body({"tool_use_id": "a", "content": "SECRET-FILE-CONTENTS"}))
    assert "SECRET" not in json.dumps(r)


def test_garbage_bodies_do_not_raise():
    for body in ({"messages": [None, 3, {"content": [None, {"type": "tool_result"}]}]}, {"messages": "x"}, {}):
        extract_results("anthropic", body)
        extract_results("openai_chat", body)
        extract_results("openai_responses", {"input": [None, 1]})


# ---------------------------------------------------------------- end to end through the real writer
def test_end_to_end_through_proxy(proxy_factory):
    """A tool_use response, then a request carrying its result, through the real proxy+writer."""
    import time

    import httpx

    def sse(tool_name, tid):
        ev = lambda t, o: f"event: {t}\ndata: {json.dumps(o)}\n\n".encode()  # noqa: E731
        return [ev("message_start", {"type": "message_start", "message": {"model": "m", "usage": {"input_tokens": 1}}}),
                ev("content_block_start", {"type": "content_block_start", "index": 0,
                   "content_block": {"type": "tool_use", "id": tid, "name": tool_name, "input": {"a": 1}}}),
                ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 2}}),
                ev("message_stop", {"type": "message_stop"})]

    proxy, _, _ = proxy_factory(sse("Read", "toolu_live"), delay=0.0)
    h = {"x-api-key": "k"}
    httpx.post(proxy + "/v1/messages", headers=h, content=json.dumps(
        {"model": "m", "messages": [{"role": "user", "content": "go"}]}))
    time.sleep(0.4)
    httpx.post(proxy + "/v1/messages", headers=h, content=json.dumps(
        {"model": "m", "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_live", "name": "Read", "input": {"a": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_live", "content": "file body"}]}]}))
    end = time.time() + 5
    recs = []
    while time.time() < end and len(recs) < 2:
        recs = httpx.get(proxy + "/_trace/api/calls?limit=10").json()["calls"]
        time.sleep(0.1)
    full = [httpx.get(proxy + f"/_trace/api/calls/{c['call_id']}").json() for c in recs]
    full.sort(key=lambda r: r["timing"]["received_at"])
    first, second = full
    assert first["response"]["tool_calls"][0]["result_present"] is True
    assert first["response"]["tool_calls"][0]["gap_ms"] > 0
    assert first["response"]["tool_calls"][0]["result_bytes"] == len("file body")
    assert first["session_id"] == second["session_id"] and second["step_index"] == 1
