"""Tool call <-> result pairing (spec section 13). Runs on the writer thread.

A tool call appears in response N; its result appears in request N+1, matched by id.
Only the id, size and error flag of a result are kept, never its content.

JSONL is append-only, so a parent record is never rewritten when its result shows up. The
pairing is stored on the CHILD record (`request.tool_results[]`, each naming the call and the
parent record it answers). `join_results()` lets readers fold them back onto the parent's
tool calls, and reports calls that never got a result as "no result observed".

`gap_ms` is an UPPER BOUND on tool duration, not the duration: it contains the tool itself,
the agent's own processing, the network and any human wait (permission prompts!). It is
named gap_ms everywhere and must be labelled "gap" in the UI.
"""
from __future__ import annotations

import json
from collections import OrderedDict


def _size(v) -> int:
    if v is None:
        return 0
    if isinstance(v, str):
        return len(v.encode("utf-8", "replace"))
    if isinstance(v, list):
        total = 0
        for b in v:
            if isinstance(b, dict) and b.get("type") == "text":
                total += _size(b.get("text"))
            elif isinstance(b, dict) and b.get("type") in ("image", "document"):
                total += len(json.dumps(b.get("source") or {}, default=str))
            else:
                total += _size(b if isinstance(b, str) else json.dumps(b, default=str))
        return total
    return len(json.dumps(v, default=str).encode("utf-8", "replace"))


def extract_results(fmt: str, body: dict) -> list[dict]:
    """-> [{id, bytes, is_error}] for every tool result carried by a request body."""
    out: list[dict] = []
    if fmt == "anthropic":
        for m in body.get("messages") or []:
            content = m.get("content") if isinstance(m, dict) else None
            if not isinstance(content, list):
                continue
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id"):
                    out.append({"id": b["tool_use_id"], "bytes": _size(b.get("content")),
                                "is_error": bool(b.get("is_error"))})
    elif fmt == "openai_chat":
        for m in body.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id"):
                # OpenAI chat has no error flag on tool messages
                out.append({"id": m["tool_call_id"], "bytes": _size(m.get("content")), "is_error": None})
    elif fmt == "openai_responses":
        inp = body.get("input")
        for it in inp if isinstance(inp, list) else []:
            if isinstance(it, dict) and it.get("type") in ("function_call_output", "custom_tool_call_output") \
                    and it.get("call_id"):
                out.append({"id": it["call_id"], "bytes": _size(it.get("output")), "is_error": None})
    return out


class ToolPairer:
    """Remembers tool calls awaiting a result; annotates a request's results as it arrives.

    Pending calls are keyed by (client, tool id): ids are unique per provider response, and
    keying by client keeps two users of one proxy from ever claiming each other's calls.
    """

    def __init__(self, ttl_s: float = 1800, max_pending: int = 5000) -> None:
        self.ttl_ns = int(ttl_s * 1e9)
        self.max_pending = max_pending
        self._pending: OrderedDict[tuple, dict] = OrderedDict()   # key -> {parent, t_end, paired}
        self.results_unmatched = 0

    def attach(self, rec: dict, results: list[dict]) -> None:
        """Write this request's tool results onto the record, joined to the calls they answer."""
        client = rec["client"].get("key_hash") or "-"
        t_req = rec["timing"]["t_received"]
        out = []
        for res in results:
            entry = {"tool_call_id": res["id"], "result_bytes": res["bytes"],
                     "result_is_error": res["is_error"], "parent_call_id": None, "gap_ms": None}
            hit = self._pending.get((client, res["id"]))
            if hit is None:
                self.results_unmatched += 1        # a call we never saw (proxy restarted, other client)
            elif hit["paired"]:
                entry["parent_call_id"] = hit["parent"]     # resent in a later step: not a new gap
                entry["repeat"] = True
            else:
                hit["paired"] = True
                entry["parent_call_id"] = hit["parent"]
                if hit["t_end"] is not None:
                    entry["gap_ms"] = round((t_req - hit["t_end"]) / 1e6, 3)
            out.append(entry)
        rec["request"]["tool_results"] = out

    def register(self, rec: dict) -> None:
        """Remember this record's tool calls so a later request can answer them."""
        client = rec["client"].get("key_hash") or "-"
        for tc in rec["response"].get("tool_calls") or []:
            if tc.get("id"):
                self._pending[(client, tc["id"])] = {
                    "parent": rec["call_id"], "t_end": rec["timing"].get("t_end"), "paired": False}
        while len(self._pending) > self.max_pending:
            self._pending.popitem(last=False)

    def expire(self, now_ns: int) -> None:
        stale = [k for k, v in self._pending.items() if v["t_end"] is not None and now_ns - v["t_end"] > self.ttl_ns]
        for k in stale:
            del self._pending[k]


def join_results(records: list[dict]) -> None:
    """Reader-side: fold child `request.tool_results` back onto parent tool calls (in place).

    Sets result_present / result_bytes / result_is_error / gap_ms / result_call_id on every tool
    call. A call with no matching result keeps result_present=False ("no result observed").
    """
    calls = {r["call_id"]: r for r in records}
    for r in records:
        for tc in r["response"].get("tool_calls") or []:
            tc.setdefault("result_present", False)
            tc.setdefault("result_bytes", None)
            tc.setdefault("result_is_error", None)
            tc.setdefault("gap_ms", None)
    for child in records:
        for res in child["request"].get("tool_results") or []:
            parent = calls.get(res.get("parent_call_id") or "")
            if parent is None or res.get("repeat"):
                continue
            for tc in parent["response"].get("tool_calls") or []:
                if tc.get("id") == res["tool_call_id"] and not tc["result_present"]:
                    tc.update(result_present=True, result_bytes=res["result_bytes"],
                              result_is_error=res["result_is_error"], gap_ms=res["gap_ms"],
                              result_call_id=child["call_id"])
