"""Call record schema (spec section 8) and the mutable per-call trace the relay fills in.

M1 fills transport facts only. Fields that later milestones own stay null / not-reported,
never 0: "we were not told" and "zero" are different things.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import BaseModel


class Metric(BaseModel):
    value: int | None = None
    reported: bool = False


class Client(BaseModel):
    key_hash: str | None = None
    user_agent: str | None = None
    harness_hint: str = "unknown"


class Request(BaseModel):
    method: str
    path: str
    format: str = "unknown"
    model_requested: str | None = None
    stream: bool | None = None
    message_count: int | None = None
    tool_count: int | None = None
    body_bytes: int = 0
    system_prompt_sha256: str | None = None
    prefix_digests: list[str] = []


class Timing(BaseModel):
    received_at: str
    t_received: int
    t_sent: int | None = None
    t_headers: int | None = None
    t_first_content: int | None = None
    t_first_visible: int | None = None
    t_last_content: int | None = None
    t_end: int | None = None
    ttft_any_ms: float | None = None
    ttft_visible_ms: float | None = None
    total_ms: float | None = None
    chunk_count: int = 0
    inter_chunk_p50_ms: float | None = None
    inter_chunk_p95_ms: float | None = None


class Response(BaseModel):
    status: int | None = None
    model_served: str | None = None
    stop_reason: str | None = None
    content_bytes: int = 0
    has_thinking: bool | None = None
    thinking_visibility: str | None = None
    tool_calls: list[dict] = []


class Usage(BaseModel):
    input_total: Metric = Metric()
    output: Metric = Metric()
    cache_read: Metric = Metric()
    cache_write: Metric = Metric()


class Outcome(BaseModel):
    error: dict | None = None
    retry_index: int = 0
    client_disconnected: bool = False
    incomplete: bool = False


class Green(BaseModel):
    status: str = "disabled"  # M6 turns this on


class Content(BaseModel):
    stored: bool = False
    path: str | None = None


class CallRecord(BaseModel):
    schema_version: int = 1
    call_id: str
    trace_id: str | None = None
    session_id: str | None = None
    step_index: int | None = None
    turn_index: int | None = None
    parent_call_id: str | None = None
    client: Client
    request: Request
    timing: Timing
    response: Response
    usage: Usage = Usage()
    outcome: Outcome
    green: Green = Green()
    content: Content = Content()


def harness_hint(user_agent: str | None) -> str:
    ua = (user_agent or "").lower()
    if "claude-cli" in ua or "claude-code" in ua or "claude code" in ua:
        return "claude-code"
    if "opencode" in ua:
        return "opencode"
    return "unknown"


def credential_hash(headers: dict[str, str]) -> str | None:
    """sha256 of the credential, only to tell clients apart. The credential itself is never kept."""
    cred = headers.get("x-api-key") or headers.get("authorization") or headers.get("api-key")
    return hashlib.sha256(cred.encode()).hexdigest() if cred else None


@dataclass
class CallTrace:
    """Accumulates facts while the relay runs. Plain attribute writes only: cheap, cannot raise."""
    method: str
    path: str
    t_received: int
    received_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str | None = None
    key_hash: str | None = None
    user_agent: str | None = None
    fmt: str = "unknown"
    body_bytes: int = 0
    t_sent: int | None = None
    t_headers: int | None = None
    t_end: int | None = None
    status: int | None = None
    chunk_count: int = 0
    resp_bytes: int = 0
    error: dict | None = None
    completed: bool = False
    finalized: bool = False

    def to_record(self) -> dict:
        disconnected = not self.completed and self.error is None
        rec = CallRecord(
            call_id=self.call_id,
            trace_id=self.trace_id,
            client=Client(key_hash=self.key_hash, user_agent=self.user_agent,
                          harness_hint=harness_hint(self.user_agent)),
            request=Request(method=self.method, path=self.path, format=self.fmt,
                            body_bytes=self.body_bytes),
            timing=Timing(received_at=self.received_at, t_received=self.t_received,
                          t_sent=self.t_sent, t_headers=self.t_headers, t_end=self.t_end,
                          chunk_count=self.chunk_count,
                          total_ms=None if self.t_end is None
                          else round((self.t_end - self.t_received) / 1e6, 3)),
            response=Response(status=self.status, content_bytes=self.resp_bytes),
            outcome=Outcome(error=self.error, client_disconnected=disconnected,
                            incomplete=not self.completed),
        )
        return rec.model_dump(mode="json")
