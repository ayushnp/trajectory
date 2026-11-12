"""Call record schema (spec section 8) and the mutable per-call trace the relay fills in.

The relay only appends raw bytes to a CallTrace (cheap, cannot raise). Everything expensive,
parsing SSE, JSON, usage mapping, timing derivation, runs in to_record() on the writer thread.
Fields we were not told are null / reported:false, never 0.
"""
from __future__ import annotations

import hashlib
import json
import uuid
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import BaseModel

from . import timing as timing_mod
from . import usage as usage_mod
from .chain import digest_request
from .parsers import make_parser
from .parsers.sse import SSEReader

CAPTURE_CAP = 16 * 1024 * 1024  # per direction, per call; beyond this we stop copying


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
    prefix_digests: list[str] = []      # per-message, 16 hex chars; the chainer's input
    human_indexes: list[int] = []       # which messages are human turns (not tool results)
    opaque_reasoning: bool = False      # history contained reasoning we hash around, never decode


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
    duration_ms: float | None = None   # t_end - t_received, whatever the content
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
    capture_truncated: bool = False


class Usage(BaseModel):
    input_total: Metric = Metric()
    output: Metric = Metric()
    cache_read: Metric = Metric()
    cache_write: Metric = Metric()
    raw: dict = {}   # provider-native numbers, before normalisation


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
    continued_from_session: str | None = None
    compaction_suspected: bool = False
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
    content_type: str | None = None
    content_encoding: str | None = None
    req_body: bytearray = field(default_factory=bytearray)
    req_truncated: bool = False
    resp_chunks: list = field(default_factory=list)      # [(monotonic_ns, bytes)], capped
    resp_kept: int = 0
    resp_truncated: bool = False
    # set by to_record(); read by the writer for self-metrics
    unknown_events: int = 0
    event_errors: int = 0
    parse_failed: bool = False

    # -- fast-path helpers: plain attribute writes, cannot raise -----------------------------
    def add_req_chunk(self, chunk: bytes) -> None:
        self.body_bytes += len(chunk)
        if not self.req_truncated:
            if len(self.req_body) + len(chunk) > CAPTURE_CAP:
                self.req_truncated = True
            else:
                self.req_body += chunk

    def add_resp_chunk(self, t: int, chunk: bytes) -> None:
        self.chunk_count += 1
        self.resp_bytes += len(chunk)
        if not self.resp_truncated:
            if self.resp_kept + len(chunk) > CAPTURE_CAP:
                self.resp_truncated = True
            else:
                self.resp_chunks.append((t, chunk))
                self.resp_kept += len(chunk)

    # -- slow path: runs on the writer thread ------------------------------------------------
    def _parse_request(self) -> dict:
        info: dict = {}
        if self.req_truncated or not self.req_body:
            return info
        body = json.loads(bytes(self.req_body))
        if not isinstance(body, dict):
            return info
        model = body.get("model")
        info["model_requested"] = model if isinstance(model, str) else None
        info["stream"] = body.get("stream") if isinstance(body.get("stream"), bool) else None
        msgs = body.get("input") if self.fmt == "openai_responses" else body.get("messages")
        if isinstance(msgs, list):
            info["message_count"] = len(msgs)
        elif isinstance(msgs, str):
            info["message_count"] = 1
        if isinstance(body.get("tools"), list):
            info["tool_count"] = len(body["tools"])
        elif msgs is not None:
            info["tool_count"] = 0
        system = body.get("system") if self.fmt == "anthropic" else body.get("instructions")
        if system:
            text = system if isinstance(system, str) else json.dumps(system, sort_keys=True)
            info["system_prompt_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        try:   # chaining input is best-effort: failing here must not lose the other request facts
            d = digest_request(self.fmt, body)
            if d is not None:
                info["prefix_digests"], info["human_indexes"], info["opaque_reasoning"] = d
        except Exception:
            self.parse_failed = True
        return info

    def _decoded_chunks(self) -> list:
        enc = (self.content_encoding or "").lower().strip()
        if not enc or enc == "identity":
            return list(self.resp_chunks)
        if enc in ("gzip", "x-gzip", "deflate"):
            d = zlib.decompressobj(47)  # auto-detects gzip / zlib framing
            out = []
            for t, c in self.resp_chunks:
                piece = d.decompress(c)
                if piece:
                    out.append((t, piece))
            return out
        raise ValueError(f"unsupported content-encoding {enc}")

    def _parse_response(self):
        parser = make_parser(self.fmt)
        if parser is None or not self.resp_chunks:
            return None
        chunks = self._decoded_chunks()
        ctype = (self.content_type or "").lower()
        if "event-stream" in ctype:
            reader = SSEReader()
            for t, c in chunks:
                for ev in reader.feed(c, t):
                    parser.feed(ev)
            for ev in reader.flush(chunks[-1][0] if chunks else self.t_received):
                parser.feed(ev)
        elif "json" in ctype:
            parser.handle_json(json.loads(b"".join(c for _, c in chunks)))
        else:
            return None
        return parser.finish()

    def to_record(self) -> dict:
        req_info: dict = {}
        res = None
        try:
            req_info = self._parse_request()
        except Exception:
            self.parse_failed = True
        try:
            res = self._parse_response()
        except Exception:
            self.parse_failed = True
        if res is not None:
            self.unknown_events, self.event_errors = res.unknown_events, res.event_errors

        disconnected = not self.completed and self.error is None
        error = self.error
        if error is None and res is not None and res.error:
            error = {**res.error, "status": self.status}
        if error is None and self.status is not None and self.status >= 400:
            error = {"type": "http_error", "message": None, "status": self.status}
        incomplete = (not self.completed) or (res is not None and not res.terminated)

        t: dict = {}
        if res is not None:
            t = timing_mod.derive(self.t_received, res.t_first_content, res.t_first_visible,
                                  res.t_last_content, [ct for ct, _ in self.resp_chunks])
        u = usage_mod.canonical(self.fmt, res.usage) if res is not None and res.usage else {}

        rec = CallRecord(
            call_id=self.call_id,
            trace_id=self.trace_id,
            client=Client(key_hash=self.key_hash, user_agent=self.user_agent,
                          harness_hint=harness_hint(self.user_agent)),
            request=Request(method=self.method, path=self.path, format=self.fmt,
                            body_bytes=self.body_bytes, **req_info),
            timing=Timing(received_at=self.received_at, t_received=self.t_received,
                          t_sent=self.t_sent, t_headers=self.t_headers, t_end=self.t_end,
                          chunk_count=self.chunk_count,
                          duration_ms=None if self.t_end is None
                          else round((self.t_end - self.t_received) / 1e6, 3),
                          **t),
            response=Response(
                status=self.status, content_bytes=self.resp_bytes,
                model_served=res.model_served if res else None,
                stop_reason=res.stop_reason if res else None,
                has_thinking=res.has_thinking if res else None,
                thinking_visibility=res.thinking_visibility if res else None,
                tool_calls=res.tool_calls if res else [],
                capture_truncated=self.resp_truncated or self.req_truncated),
            usage=Usage(**u) if u else Usage(),
            outcome=Outcome(error=error, client_disconnected=disconnected, incomplete=incomplete),
        )
        # free the captured bytes as soon as the record exists
        self.req_body, self.resp_chunks = bytearray(), []
        return rec.model_dump(mode="json")
