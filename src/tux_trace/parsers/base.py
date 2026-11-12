"""Shared parser state. One parser instance per call; never shared across threads."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .sse import SSEEvent


@dataclass
class ParseResult:
    model_served: str | None = None
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)       # flat provider-native, see usage.py
    tool_calls: list[dict] = field(default_factory=list)
    has_thinking: bool = False
    thinking_visibility: str = "none"               # summary | encrypted | none
    t_first_content: int | None = None
    t_first_visible: int | None = None
    t_last_content: int | None = None
    error: dict | None = None
    terminated: bool = False                        # saw the format's end-of-stream marker
    unknown_events: int = 0
    event_errors: int = 0


class StreamParser:
    KNOWN: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self.r = ParseResult()
        self._tools: dict = {}

    # -- driver ----------------------------------------------------------------------------
    def feed(self, ev: SSEEvent) -> None:
        """One bad event must never kill parsing of the rest."""
        try:
            data = ev.data.strip()
            if data == "[DONE]":
                self.r.terminated = True
                return
            if not data:
                return
            obj = json.loads(data)
            if isinstance(obj, dict):
                self.handle(ev, obj)
        except Exception:
            self.r.event_errors += 1

    def handle(self, ev: SSEEvent, obj: dict) -> None:
        raise NotImplementedError

    def handle_json(self, obj: dict) -> None:
        """Non-streaming response body."""
        raise NotImplementedError

    def finish(self) -> ParseResult:
        self.r.tool_calls = list(self._tools.values())
        if not self.r.has_thinking:
            self.r.thinking_visibility = "none"
        return self.r

    # -- helpers shared by the formats -----------------------------------------------------
    def mark(self, kind: str, t: int) -> None:
        """kind: text | thinking | tool. First/last content; first *visible* text."""
        r = self.r
        if r.t_first_content is None:
            r.t_first_content = t
        if kind == "text" and r.t_first_visible is None:
            r.t_first_visible = t
        r.t_last_content = t

    def note_thinking(self, readable: bool) -> None:
        self.r.has_thinking = True
        if readable:
            self.r.thinking_visibility = "summary"
        elif self.r.thinking_visibility != "summary":
            self.r.thinking_visibility = "encrypted"  # present but opaque to us

    def tool(self, key, id=None, name=None, args_add=None, args_set=None) -> None:
        cur = self._tools.get(key)
        if cur is None:
            cur = self._tools[key] = {"id": None, "name": None, "args": ""}
        if id:
            cur["id"] = id
        if name:
            cur["name"] = name
        if args_set is not None:
            cur["args"] = args_set
        if args_add:
            cur["args"] += args_add

    def add_usage(self, flat: dict) -> None:
        self.r.usage.update(flat)  # last value wins; cumulative counters are never summed

    @staticmethod
    def extract_error(obj: dict) -> dict | None:
        err = obj.get("error")
        if isinstance(err, dict):
            return {"type": err.get("type") or err.get("code") or "error", "message": err.get("message")}
        if isinstance(err, str) and err:
            return {"type": "error", "message": err}
        return None
