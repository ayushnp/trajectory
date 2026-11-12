"""Subagent attachment (spec section 14). Runs on the writer thread.

A subagent's calls come through the proxy as an ordinary chain. We try to attach the root of
such a chain to the parent tool call that spawned it, and say honestly how sure we are.

Signals
  overlap  the child's request arrives while the parent's tool call is still pending
           (after the parent response ended, before its result is delivered)
  echo     the child's first human message appears inside the parent's tool-call arguments
           (delegation tools embed the child's prompt)

Scoring (only high and medium are attached)
  echo + overlap -> high      overlap only -> medium      echo only -> low (not attached)
  neither -> no attachment

Calls that are chain roots, start while another session has a pending tool call, but carry no
echo, go to the `unattached` bucket. They are never guessed into a parent.

Privacy: the child's prompt text is used for matching and discarded. Only the verdict
(signals, confidence, parent) is stored; no prompt text ever reaches a record.
"""
from __future__ import annotations

import re
from collections import OrderedDict

ECHO_CHARS = 200
MIN_ECHO_CHARS = 12          # shorter prompts ("ok", "continue") match too much to mean anything
_WS = re.compile(r"\s+")
SCORES = {"high": 0.85, "medium": 0.5, "low": 0.25}


def normalize(text: str, n: int = ECHO_CHARS) -> str:
    """Whitespace-collapsed, lowercased, first n chars. JSON string escapes are undone first so
    text embedded in a tool call's JSON arguments (\\n, \\\\, \\") compares equal to the original."""
    t = text or ""
    if "\\" in t:
        t = t.replace("\\\\", "\x00").replace("\\n", " ").replace("\\t", " ").replace('\\"', '"').replace("\x00", "\\")
    return _WS.sub(" ", t).strip().lower()[:n]


def first_human_text(fmt: str, body: dict) -> str | None:
    """Text of the first human message in a request (transient; never stored)."""
    def text_of(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content
                            if isinstance(p, dict) and p.get("type") in ("text", "input_text") and p.get("text"))
        return ""
    if fmt in ("anthropic", "openai_chat"):
        for m in body.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                    continue
                t = text_of(content)
                if t.strip():
                    return t
    elif fmt == "openai_responses":
        inp = body.get("input")
        if isinstance(inp, str):
            return inp
        for it in inp if isinstance(inp, list) else []:
            if isinstance(it, dict) and it.get("role") == "user":
                t = text_of(it.get("content"))
                if t.strip():
                    return t
    return None


class SubagentAttacher:
    def __init__(self, ttl_s: float = 1800, max_pending: int = 2000) -> None:
        self.ttl_ns = int(ttl_s * 1e9)
        self.max_pending = max_pending
        # (client, tool id) -> pending call
        self._pending: OrderedDict[tuple, dict] = OrderedDict()
        self.child_candidates = 0
        self.unattached = 0
        self.attached = 0

    # ---- bookkeeping, called for every record in order ---------------------------------------
    def register(self, rec: dict) -> None:
        client = rec["client"].get("key_hash") or "-"
        for tc in rec["response"].get("tool_calls") or []:
            if tc.get("id"):
                self._pending[(client, tc["id"])] = {
                    "call_id": rec["call_id"], "session_id": rec.get("session_id"),
                    "t_end": rec["timing"].get("t_end"), "args": normalize_args(tc.get("args") or ""),
                    # pending from first appearance: a subagent can start before the parent stream ends
                    "t_open": tc.get("t_start") or rec["timing"].get("t_end"),
                    "name": tc.get("name"), "answered_at": None,
                }
        while len(self._pending) > self.max_pending:
            self._pending.popitem(last=False)

    def note_results(self, rec: dict) -> None:
        """A request that delivers a result closes that call's pending window."""
        client = rec["client"].get("key_hash") or "-"
        t = rec["timing"]["t_received"]
        for res in rec["request"].get("tool_results") or []:
            p = self._pending.get((client, res["tool_call_id"]))
            if p is not None and p["answered_at"] is None and not res.get("repeat"):
                p["answered_at"] = t

    # ---- the decision ------------------------------------------------------------------------
    def assess(self, rec: dict, first_text: str | None) -> None:
        """Decide whether this record roots a subagent chain; set parent_call_id / attachment."""
        if rec.get("step_index") != 0 or rec.get("session_id") is None:
            return                                   # only chain roots can be children
        if rec["outcome"].get("retry_index"):
            return
        client = rec["client"].get("key_hash") or "-"
        t = rec["timing"]["t_received"]
        needle = normalize(first_text or "")
        usable_needle = len(needle) >= MIN_ECHO_CHARS

        pending = []       # tool calls from other sessions of this client that are open right now
        for (c, _), p in self._pending.items():
            if c != client or p["session_id"] == rec["session_id"]:
                continue
            if p["t_open"] is None or t < p["t_open"]:
                continue                              # the call had not appeared yet
            if p["answered_at"] is not None and t >= p["answered_at"]:
                continue                              # result already delivered: no longer pending
            if t - p["t_open"] > self.ttl_ns:
                continue
            pending.append(p)

        if pending:
            self.child_candidates += 1
        echoes = [p for p in pending if usable_needle and needle in p["args"]]
        chosen, signals, conf = None, [], None
        if echoes:
            # echo + overlap is the strongest evidence; if several calls echo, the latest wins
            chosen, signals, conf = max(echoes, key=lambda p: p["t_open"]), ["echo", "overlap"], "high"
        elif len(pending) == 1:
            # overlap alone is only meaningful when exactly one call could have spawned the child.
            # With several pending calls (e.g. Agent + Read in parallel) picking one is a guess.
            chosen, signals, conf = pending[0], ["overlap"], "medium"
        elif len(pending) > 1:
            rec["overlap_candidates"] = len(pending)      # why it is unattached, for the UI

        if chosen is not None:
            rec["parent_call_id"] = chosen["call_id"]
            rec["subagent_edge"] = {"parent_call_id": chosen["call_id"], "confidence": conf,
                                    "signals": signals, "score": SCORES[conf]}
            self.attached += 1
            return

        # Echo without overlap is a weak (low) signal: reported, not attached.
        if usable_needle:
            for (c, _), p in self._pending.items():
                if c == client and p["session_id"] != rec["session_id"] and needle in p["args"]:
                    rec["subagent_edge"] = {"parent_call_id": p["call_id"], "confidence": "low",
                                            "signals": ["echo"], "score": SCORES["low"], "attached": False}
                    break
        if pending:            # started mid-session, nothing tied it to one specific parent
            rec["unattached"] = True
            # which sessions had a call open at that moment, so the UI can show it under them
            rec["overlapped_sessions"] = sorted({p["session_id"] for p in pending if p["session_id"]})
            self.unattached += 1

    @property
    def unattached_rate(self) -> float | None:
        return round(self.unattached / self.child_candidates, 4) if self.child_candidates else None

    def expire(self, now_ns: int) -> None:
        stale = [k for k, v in self._pending.items() if v["t_end"] is not None and now_ns - v["t_end"] > self.ttl_ns]
        for k in stale:
            del self._pending[k]


def normalize_args(args: str) -> str:
    """Tool-call arguments in the same normal form as `normalize`, but over the whole string."""
    return normalize(args, n=1_000_000)
