"""OpenAI /v1/responses SSE."""
from __future__ import annotations

from .. import usage
from .base import StreamParser
from .sse import SSEEvent


class OpenAIResponsesParser(StreamParser):
    def handle(self, ev: SSEEvent, obj: dict) -> None:
        r, t = self.r, ev.t
        typ = obj.get("type") or ev.event or ""
        if typ in ("response.created", "response.in_progress"):
            r.model_served = (obj.get("response") or {}).get("model") or r.model_served
        elif typ == "response.output_item.added":
            item = obj.get("item") or {}
            it = item.get("type")
            if it == "reasoning":
                # encrypted_content is opaque: counted, never decoded. Readable summaries (if any)
                # arrive as response.reasoning_summary_text.delta.
                self.note_thinking(readable=False)
            elif it in ("function_call", "custom_tool_call"):
                self.tool(item.get("id") or obj.get("output_index"),
                          id=item.get("call_id") or item.get("id"), name=item.get("name"),
                          args_set=item.get("arguments") or item.get("input") or None)
                self.mark("tool", t)
        elif typ == "response.output_text.delta":
            self.mark("text", t)
        elif typ in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            self.note_thinking(readable=True)
            self.mark("thinking", t)
        elif typ in ("response.function_call_arguments.delta", "response.custom_tool_call_input.delta"):
            self.tool(obj.get("item_id") or obj.get("output_index"), args_add=obj.get("delta"))
            self.mark("tool", t)
        elif typ in ("response.function_call_arguments.done", "response.custom_tool_call_input.done"):
            self.tool(obj.get("item_id") or obj.get("output_index"),
                      args_set=obj.get("arguments") or obj.get("input"))
        elif typ == "response.output_item.done":
            item = obj.get("item") or {}
            if item.get("type") in ("function_call", "custom_tool_call"):
                self.tool(item.get("id") or obj.get("output_index"),
                          id=item.get("call_id") or item.get("id"), name=item.get("name"),
                          args_set=item.get("arguments") or item.get("input") or None)
        elif typ in ("response.completed", "response.incomplete", "response.failed"):
            resp = obj.get("response") or {}
            r.terminated = True
            r.model_served = resp.get("model") or r.model_served
            self.add_usage(usage.from_openai_responses(resp.get("usage")))
            if typ == "response.completed":
                outs = [o.get("type") for o in resp.get("output") or []]
                r.stop_reason = "tool_calls" if any(o in ("function_call", "custom_tool_call") for o in outs) \
                    else "completed"
            elif typ == "response.incomplete":
                r.stop_reason = (resp.get("incomplete_details") or {}).get("reason") or "incomplete"
            else:
                r.stop_reason = "failed"
                r.error = self.extract_error(resp) or {"type": "failed", "message": None}
            for o in resp.get("output") or []:
                if o.get("type") == "reasoning":
                    self.note_thinking(readable=bool(o.get("summary")))
        elif typ == "error":
            r.error = self.extract_error(obj) or {"type": obj.get("code") or "error",
                                                  "message": obj.get("message")}
        elif not typ.startswith("response."):
            r.unknown_events += 1
        # other response.* lifecycle events (content_part.*, output_text.done, ...) carry nothing new

    def handle_json(self, obj: dict) -> None:
        r = self.r
        r.model_served = obj.get("model")
        self.add_usage(usage.from_openai_responses(obj.get("usage")))
        r.terminated = True
        r.error = self.extract_error(obj)
        for o in obj.get("output") or []:
            if o.get("type") in ("function_call", "custom_tool_call"):
                self.tool(o.get("id"), id=o.get("call_id") or o.get("id"), name=o.get("name"),
                          args_set=o.get("arguments") or o.get("input"))
            elif o.get("type") == "reasoning":
                self.note_thinking(readable=bool(o.get("summary")))
