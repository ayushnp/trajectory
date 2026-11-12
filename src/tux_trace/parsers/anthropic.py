"""Anthropic /v1/messages SSE."""
from __future__ import annotations

from .. import usage
from .base import StreamParser
from .sse import SSEEvent


class AnthropicParser(StreamParser):
    KNOWN = frozenset({"message_start", "content_block_start", "content_block_delta",
                       "content_block_stop", "message_delta", "message_stop", "ping", "error"})

    def handle(self, ev: SSEEvent, obj: dict) -> None:
        r, t = self.r, ev.t
        typ = obj.get("type") or ev.event
        if typ not in self.KNOWN:
            r.unknown_events += 1
            return
        if typ == "message_start":
            m = obj.get("message") or {}
            r.model_served = m.get("model") or r.model_served
            self.add_usage(usage.from_anthropic(m.get("usage")))
        elif typ == "content_block_start":
            blk = obj.get("content_block") or {}
            idx, bt = obj.get("index"), blk.get("type")
            if bt == "tool_use":
                self.tool(idx, id=blk.get("id"), name=blk.get("name"))
                inp = blk.get("input")
                if inp:  # whole tool call arrives in one event when toolStreaming is off
                    import json
                    self.tool(idx, args_set=json.dumps(inp, separators=(",", ":")))
                self.mark("tool", t)
            elif bt in ("thinking", "redacted_thinking"):
                self.note_thinking(readable=(bt == "thinking"))
        elif typ == "content_block_delta":
            d = obj.get("delta") or {}
            dt = d.get("type")
            if dt == "text_delta":
                self.mark("text", t)
            elif dt == "thinking_delta":
                self.note_thinking(readable=True)
                self.mark("thinking", t)
            elif dt == "input_json_delta":
                self.tool(obj.get("index"), args_add=d.get("partial_json"))
                self.mark("tool", t)
            # signature_delta and anything else: not content
        elif typ == "message_delta":
            d = obj.get("delta") or {}
            r.stop_reason = d.get("stop_reason") or r.stop_reason
            self.add_usage(usage.from_anthropic(obj.get("usage")))
        elif typ == "message_stop":
            r.terminated = True
            self.add_usage(usage.from_anthropic(obj.get("usage")))
        elif typ == "error":
            r.error = self.extract_error(obj) or {"type": "error", "message": None}

    def handle_json(self, obj: dict) -> None:
        r = self.r
        r.model_served = obj.get("model")
        r.stop_reason = obj.get("stop_reason")
        self.add_usage(usage.from_anthropic(obj.get("usage")))
        for i, blk in enumerate(obj.get("content") or []):
            if blk.get("type") == "tool_use":
                import json
                self.tool(i, id=blk.get("id"), name=blk.get("name"),
                          args_set=json.dumps(blk.get("input"), separators=(",", ":")))
            elif blk.get("type") in ("thinking", "redacted_thinking"):
                self.note_thinking(readable=blk.get("type") == "thinking")
        r.terminated = True
        r.error = self.extract_error(obj) if obj.get("type") == "error" else None
