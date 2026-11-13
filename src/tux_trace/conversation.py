"""Readable view of one stored call: what was newly asked, and what the model said back.

Read-only and best-effort. It works on the redacted content file (data/content/<id>.json), never on
the live stream. The request body carries the WHOLE conversation on every call, so only the part that
is new since the model last spoke is returned in full; everything older is summarised as a count.
System prompt and tool definitions are returned as sizes, with the text behind a separate flag.
"""
from __future__ import annotations

import json
from typing import Any

from .parsers.sse import SSEReader

MAX_TEXT = 20_000     # per block; longer text is cut and says so


def _clip(s: str) -> dict:
    if len(s) <= MAX_TEXT:
        return {"text": s, "truncated": False}
    return {"text": s[:MAX_TEXT], "truncated": True, "total_chars": len(s)}


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                t = b.get("text")
                parts.append(t if isinstance(t, str) else json.dumps(b.get("content", b), ensure_ascii=False))
            else:
                parts.append(str(b))
        return "\n".join(p for p in parts if p)
    return "" if content is None else json.dumps(content, ensure_ascii=False)


# ------------------------------------------------------------------ request
def _blocks_anthropic(msg: dict) -> list[dict]:
    c = msg.get("content")
    if isinstance(c, str):
        return [{"kind": "text", **_clip(c)}]
    out = []
    for b in c or []:
        t = b.get("type")
        if t == "text":
            out.append({"kind": "text", **_clip(b.get("text", ""))})
        elif t == "tool_result":
            out.append({"kind": "tool_result", "tool_use_id": b.get("tool_use_id"),
                        "is_error": bool(b.get("is_error")), **_clip(_text_of(b.get("content")))})
        elif t == "tool_use":
            out.append({"kind": "tool_call", "id": b.get("id"), "name": b.get("name"), "args": b.get("input")})
        elif t in ("thinking", "redacted_thinking"):
            out.append({"kind": "thinking", **_clip(b.get("thinking") or "(not visible)")})
        else:
            out.append({"kind": t or "other", **_clip(json.dumps(b, ensure_ascii=False))})
    return out


def _new_tail(messages: list[dict]) -> tuple[list[dict], int]:
    """Trailing messages after the last assistant message = what is new in this request."""
    i = len(messages)
    while i > 0 and messages[i - 1].get("role") != "assistant":
        i -= 1
    return messages[i:], i


def _request_view(req: Any, fmt: str) -> dict:
    if not isinstance(req, dict):
        return {"note": "request body was not JSON", **_clip(str(req or ""))}
    view: dict[str, Any] = {"model": req.get("model")}
    system = req.get("system") if fmt == "anthropic" else req.get("instructions")
    if system is not None:
        s = _text_of(system)
        view["system"] = {"chars": len(s), **_clip(s)}
    tools = req.get("tools") or []
    view["tools"] = {"count": len(tools), "names": [t.get("name") or (t.get("function") or {}).get("name") for t in tools][:60]}

    if fmt == "anthropic" or (fmt == "openai_chat" and "messages" in req):
        msgs = req.get("messages") or []
        tail, earlier = _new_tail(msgs)
        view["earlier_messages"] = earlier
        conv = []
        for m in tail:
            if fmt == "anthropic":
                blocks = _blocks_anthropic(m)
            else:
                blocks = ([{"kind": "tool_result", "tool_use_id": m.get("tool_call_id"), **_clip(_text_of(m.get("content")))}]
                          if m.get("role") == "tool" else [{"kind": "text", **_clip(_text_of(m.get("content")))}])
            conv.append({"role": m.get("role"), "blocks": blocks})
        view["new_messages"] = conv
    else:   # responses API: `input` is a string or a list of items
        inp = req.get("input")
        items = [{"role": "user", "content": inp}] if isinstance(inp, str) else (inp or [])
        tail_start = 0
        for idx, it in enumerate(items):
            if isinstance(it, dict) and (it.get("role") == "assistant" or it.get("type") in ("function_call", "reasoning")):
                tail_start = idx + 1
        view["earlier_messages"] = tail_start
        conv = []
        for it in items[tail_start:]:
            if not isinstance(it, dict):
                continue
            if it.get("type") == "function_call_output":
                conv.append({"role": "tool", "blocks": [{"kind": "tool_result", "tool_use_id": it.get("call_id"),
                                                         **_clip(_text_of(it.get("output")))}]})
            else:
                conv.append({"role": it.get("role", "user"), "blocks": [{"kind": "text", **_clip(_text_of(it.get("content")))}]})
        view["new_messages"] = conv
    return view


# ------------------------------------------------------------------ response
def _events(resp: str) -> list[tuple[str | None, dict]]:
    rd = SSEReader()
    evs = rd.feed(resp.encode("utf-8"), 0) + rd.flush(0)
    out = []
    for e in evs:
        if not e.data or e.data == "[DONE]":
            continue
        try:
            out.append((e.event, json.loads(e.data)))
        except ValueError:
            continue
    return out


def _response_anthropic(evs: list[tuple[str | None, dict]]) -> list[dict]:
    blocks: dict[int, dict] = {}
    for _, d in evs:
        t = d.get("type")
        i = d.get("index", 0)
        if t == "content_block_start":
            cb = d.get("content_block") or {}
            blocks[i] = {"type": cb.get("type"), "text": "", "id": cb.get("id"), "name": cb.get("name"), "json": ""}
        elif t == "content_block_delta":
            b = blocks.setdefault(i, {"type": "text", "text": "", "json": ""})
            dl = d.get("delta") or {}
            if dl.get("type") == "text_delta":
                b["text"] += dl.get("text", "")
            elif dl.get("type") == "thinking_delta":
                b["text"] += dl.get("thinking", "")
            elif dl.get("type") == "input_json_delta":
                b["json"] += dl.get("partial_json", "")
    out = []
    for i in sorted(blocks):
        b = blocks[i]
        if b["type"] == "tool_use":
            try:
                args = json.loads(b["json"]) if b["json"] else {}
            except ValueError:
                args = b["json"]
            out.append({"kind": "tool_call", "id": b["id"], "name": b["name"], "args": args})
        elif b["type"] in ("thinking", "redacted_thinking"):
            out.append({"kind": "thinking", **_clip(b["text"] or "(no visible reasoning text)")})
        else:
            out.append({"kind": "text", **_clip(b["text"])})
    return out


def _response_chat(evs: list[tuple[str | None, dict]]) -> list[dict]:
    text, calls = "", {}
    for _, d in evs:
        for ch in d.get("choices") or []:
            dl = ch.get("delta") or {}
            text += dl.get("content") or ""
            for tc in dl.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"id": None, "name": None, "json": ""})
                c["id"] = tc.get("id") or c["id"]
                fn = tc.get("function") or {}
                c["name"] = fn.get("name") or c["name"]
                c["json"] += fn.get("arguments") or ""
    out = [{"kind": "text", **_clip(text)}] if text else []
    for i in sorted(calls):
        c = calls[i]
        try:
            args = json.loads(c["json"]) if c["json"] else {}
        except ValueError:
            args = c["json"]
        out.append({"kind": "tool_call", "id": c["id"], "name": c["name"], "args": args})
    return out


def _response_responses(evs: list[tuple[str | None, dict]]) -> list[dict]:
    text, calls, reasoning = "", {}, ""
    for _, d in evs:
        t = d.get("type", "")
        if t == "response.output_text.delta":
            text += d.get("delta", "")
        elif t in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            reasoning += d.get("delta", "")
        elif t == "response.output_item.added" and (d.get("item") or {}).get("type") == "function_call":
            it = d["item"]
            calls[it.get("id") or it.get("call_id")] = {"id": it.get("call_id"), "name": it.get("name"), "json": ""}
        elif t == "response.function_call_arguments.delta":
            c = calls.get(d.get("item_id"))
            if c is not None:
                c["json"] += d.get("delta", "")
    out = []
    if reasoning:
        out.append({"kind": "thinking", **_clip(reasoning)})
    if text:
        out.append({"kind": "text", **_clip(text)})
    for c in calls.values():
        try:
            args = json.loads(c["json"]) if c["json"] else {}
        except ValueError:
            args = c["json"]
        out.append({"kind": "tool_call", "id": c["id"], "name": c["name"], "args": args})
    return out


def _response_json(d: dict, fmt: str) -> list[dict]:
    """A non-streamed reply."""
    if fmt == "anthropic":
        return _blocks_anthropic({"content": d.get("content")})
    if fmt == "openai_chat":
        out = []
        for ch in d.get("choices") or []:
            m = ch.get("message") or {}
            if m.get("content"):
                out.append({"kind": "text", **_clip(_text_of(m["content"]))})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                out.append({"kind": "tool_call", "id": tc.get("id"), "name": fn.get("name"), "args": fn.get("arguments")})
        return out
    out = []
    for it in d.get("output") or []:
        if it.get("type") == "message":
            out.append({"kind": "text", **_clip(_text_of(it.get("content")))})
        elif it.get("type") == "function_call":
            out.append({"kind": "tool_call", "id": it.get("call_id"), "name": it.get("name"), "args": it.get("arguments")})
    return out


def _response_view(resp: Any, fmt: str) -> dict:
    if not resp:
        return {"blocks": [], "note": "no response body was stored (request failed or was disconnected)"}
    if isinstance(resp, dict):
        return {"blocks": _response_json(resp, fmt)}
    s = str(resp)
    if s.lstrip().startswith("{"):
        try:
            return {"blocks": _response_json(json.loads(s), fmt)}
        except ValueError:
            pass
    evs = _events(s)
    if not evs:
        return {"blocks": [{"kind": "text", **_clip(s)}], "note": "not an event stream; shown raw"}
    fn = {"anthropic": _response_anthropic, "openai_chat": _response_chat}.get(fmt, _response_responses)
    return {"blocks": fn(evs)}


def build(doc: dict) -> dict:
    """doc is the parsed content file."""
    fmt = doc.get("format") or "anthropic"
    return {"call_id": doc.get("call_id"), "format": fmt, "redacted": bool(doc.get("redacted")),
            "request": _request_view(doc.get("request"), fmt),
            "response": _response_view(doc.get("response"), fmt)}
