"""OpenAI /v1/chat/completions SSE."""
from __future__ import annotations

from .. import usage
from .base import StreamParser
from .sse import SSEEvent


class OpenAIChatParser(StreamParser):
    def handle(self, ev: SSEEvent, obj: dict) -> None:
        r, t = self.r, ev.t
        err = self.extract_error(obj)
        if err:
            r.error = err
            return
        r.model_served = obj.get("model") or r.model_served
        if obj.get("usage"):
            self.add_usage(usage.from_openai_chat(obj["usage"]))
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            d = choice.get("delta") or {}
            if d.get("content"):
                self.mark("text", t)
            for key in ("reasoning_content", "reasoning"):  # some gateways expose readable reasoning
                if d.get(key):
                    self.note_thinking(readable=True)
                    self.mark("thinking", t)
            for tc in d.get("tool_calls") or []:
                fn = tc.get("function") or {}
                # fragments are keyed by `index`; id/name arrive on the first fragment only
                self.tool(tc.get("index", 0), id=tc.get("id"), name=fn.get("name"),
                          args_add=fn.get("arguments"))
                self.mark("tool", t)
            if choice.get("finish_reason"):
                r.stop_reason = choice["finish_reason"]

    def handle_json(self, obj: dict) -> None:
        r = self.r
        r.model_served = obj.get("model")
        self.add_usage(usage.from_openai_chat(obj.get("usage")))
        for ch in obj.get("choices") or []:
            r.stop_reason = ch.get("finish_reason") or r.stop_reason
            for tc in (ch.get("message") or {}).get("tool_calls") or []:
                fn = tc.get("function") or {}
                self.tool(tc.get("index", tc.get("id")), id=tc.get("id"), name=fn.get("name"),
                          args_set=fn.get("arguments"))
        r.terminated = True
        r.error = self.extract_error(obj)
