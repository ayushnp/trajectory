import json

from trajectory.conversation import build


def _sse(*events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def test_anthropic_new_messages_and_joined_stream():
    req = {"model": "m", "system": "be nice", "tools": [{"name": "Bash"}], "messages": [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": [{"type": "text", "text": "old answer"}]},
        {"role": "user", "content": [{"type": "text", "text": "what is 2+2?"}]}]}
    resp = _sse(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "It is "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "4."}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "Bash"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"cmd":'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"ls"}'}})
    v = build({"call_id": "c", "format": "anthropic", "redacted": True, "request": req, "response": resp})
    assert v["request"]["earlier_messages"] == 2
    assert [m["role"] for m in v["request"]["new_messages"]] == ["user"]
    assert v["request"]["new_messages"][0]["blocks"][0]["text"] == "what is 2+2?"
    assert v["request"]["tools"]["names"] == ["Bash"]
    blocks = v["response"]["blocks"]
    assert blocks[0] == {"kind": "text", "text": "It is 4.", "truncated": False}
    assert blocks[1]["name"] == "Bash" and blocks[1]["args"] == {"cmd": "ls"}


def test_tool_result_is_shown_as_new():
    req = {"messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                                      "content": [{"type": "text", "text": "boom"}]}]}]}
    v = build({"format": "anthropic", "request": req, "response": None})
    b = v["request"]["new_messages"][0]["blocks"][0]
    assert b["kind"] == "tool_result" and b["is_error"] and b["text"] == "boom"
    assert v["response"]["blocks"] == [] and "no response" in v["response"]["note"]


def test_openai_chat_stream():
    resp = ('data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"llo"}}]}\n\ndata: [DONE]\n\n')
    v = build({"format": "openai_chat", "request": {"messages": [{"role": "user", "content": "hi"}]}, "response": resp})
    assert v["response"]["blocks"][0]["text"] == "hello"


def test_long_text_is_cut_and_says_so():
    v = build({"format": "anthropic", "request": {"messages": [{"role": "user", "content": "x" * 30000}]}, "response": None})
    b = v["request"]["new_messages"][0]["blocks"][0]
    assert b["truncated"] and b["total_chars"] == 30000 and len(b["text"]) == 20000


def test_non_json_request_does_not_crash():
    v = build({"format": "unknown", "request": "raw bytes", "response": None})
    assert v["request"]["text"] == "raw bytes"


def test_favicon_is_answered_locally():
    from starlette.applications import Starlette
    from starlette.testclient import TestClient
    from trajectory import api
    r = TestClient(Starlette(routes=api.routes())).get("/favicon.ico")
    assert r.status_code == 204
