"""Fast path: forward the request upstream and stream the reply back, byte for byte.

Upstream client is aiohttp, not httpx: measured ~0.4 ms vs ~2.5 ms per request on loopback,
which was nearly all of the proxy's added latency.

Recording is fail-open: everything touching the CallTrace or the tee is wrapped so that a bug
there can only lose a record, never alter or interrupt the relayed bytes.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator

import aiohttp
from multidict import CIMultiDict
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from .parsers import detect_format
from .record import CallTrace, credential_hash

log = logging.getLogger("tux_trace.proxy")

# Hop-by-hop headers describe one TCP connection, not the message; they must not be relayed.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}
TRACE_HEADER = "x-tux-trace-id"
# Keep aiohttp from adding or altering anything on its own.
_NO_AUTO = dict(auto_decompress=False, skip_auto_headers=("Accept", "Accept-Encoding", "User-Agent", "Content-Type"))


def make_client(base_url: str, connect_timeout_s: float, read_timeout_s: float) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        base_url=base_url,
        timeout=aiohttp.ClientTimeout(total=None, connect=connect_timeout_s, sock_read=read_timeout_s),
        connector=aiohttp.TCPConnector(limit=200, keepalive_timeout=60),
        trust_env=False,          # never route loopback traffic through a system proxy
        cookie_jar=aiohttp.DummyCookieJar(),  # never store cookies
        **_NO_AUTO,
    )


def _forward_request_headers(request: Request, trace_id: str | None) -> CIMultiDict:
    h = CIMultiDict()
    for k, v in request.scope["headers"]:
        name = k.decode("latin-1")
        if name.lower() not in HOP_BY_HOP:
            h.add(name, v.decode("latin-1"))
    if trace_id:
        h.add(TRACE_HEADER, trace_id)
    return h


def _start_trace(request: Request, t_received: int, trace_id: str | None) -> CallTrace | None:
    try:
        h = request.headers
        return CallTrace(
            method=request.method, path=request.url.path, t_received=t_received,
            trace_id=trace_id, key_hash=credential_hash(h), user_agent=h.get("user-agent"),
            fmt=detect_format(request.url.path),
        )
    except Exception:
        log.exception("could not start trace")
        return None


def _sampled(trace: CallTrace, rate: float) -> bool:
    return (uuid.UUID(trace.call_id).int % 10_000) < rate * 10_000


def _finish(state, trace: CallTrace | None) -> None:
    """Hand the trace to the writer exactly once. Never raises."""
    if trace is None or trace.finalized:
        return
    try:
        trace.finalized = True
        trace.t_end = time.monotonic_ns()
        if trace.error is not None:
            state.stats.calls_error += 1
        if not trace.completed:
            state.stats.calls_incomplete += 1
        if state.cfg.sample_rate >= 1.0 or _sampled(trace, state.cfg.sample_rate):
            state.tee.submit(trace)
    except Exception:
        state.stats.record_errors += 1
        log.exception("could not finish trace")


async def relay(request: Request) -> Response:
    state = request.app.state
    t_received = time.monotonic_ns()
    trace_id = str(uuid.uuid4()) if state.cfg.add_trace_header else None
    trace = _start_trace(request, t_received, trace_id)
    headers = _forward_request_headers(request, trace_id)
    target = request.url.path + ("?" + request.url.query if request.url.query else "")

    has_body = (request.method not in ("GET", "HEAD", "OPTIONS", "DELETE")
                or "content-length" in request.headers or "transfer-encoding" in request.headers)

    async def counted_body() -> AsyncIterator[bytes]:
        async for chunk in request.stream():
            if trace is not None:
                trace.add_req_chunk(chunk)
            yield chunk

    # Our own handling time ends when we hand the request to the HTTP client (spec section 18).
    t_sent = time.monotonic_ns()
    state.stats.observe_overhead_ns(t_sent - t_received)  # also counts calls_total
    if trace is not None:
        trace.t_sent = t_sent
    try:
        upstream = await state.client.request(
            request.method, target, headers=headers,
            data=counted_body() if has_body else None,  # streamed, not buffered
            allow_redirects=False,  # a 302 from Tux belongs to the client, not to us
        )
    except Exception as e:
        status, msg = ((504, "upstream timeout") if isinstance(e, (aiohttp.ServerTimeoutError, TimeoutError))
                       else (502, "upstream unreachable") if isinstance(e, aiohttp.ClientConnectorError)
                       else (502, f"upstream error: {type(e).__name__}"))
        if trace is not None:
            trace.status = status
            trace.error = {"type": type(e).__name__, "message": msg, "status": status}
        _finish(state, trace)
        return Response(('{"error":"tux-trace: %s"}' % msg).encode(), status, media_type="application/json")

    if trace is not None:
        trace.t_headers = time.monotonic_ns()
        trace.status = upstream.status
        trace.content_type = upstream.headers.get("content-type")
        trace.content_encoding = upstream.headers.get("content-encoding")

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.content.iter_any():
                if trace is not None:  # plain appends: cheap, cannot block; parsing happens later
                    trace.add_resp_chunk(time.monotonic_ns(), chunk)
                yield chunk
            if trace is not None:
                trace.completed = True
        except Exception as e:  # upstream broke mid-stream: client sees truncation, we note why
            if trace is not None:
                trace.error = {"type": type(e).__name__, "message": str(e)[:200], "status": upstream.status}
            raise
        finally:
            upstream.release()  # also runs when the client disconnects mid-stream
            _finish(state, trace)

    resp = StreamingResponse(body(), status_code=upstream.status)
    resp.raw_headers = [
        (k.encode("latin-1"), v.encode("latin-1")) for k, v in upstream.headers.items()
        if k.lower() not in HOP_BY_HOP
    ]
    return resp
