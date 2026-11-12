"""Shared fixtures: a stub upstream server on a real socket, and the proxy in front of it."""
import asyncio
import socket
import threading
import time

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from tux_trace.app import create_app
from tux_trace.config import Config


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, app, port):
        self.port = port
        self.server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="error",
            access_log=False, server_header=False, date_header=False))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("server did not start")

    def __exit__(self, *a):
        self.server.should_exit = True
        self.thread.join(5)


def make_stub(chunks, delay=0.0, log=None):
    """Upstream stub: streams `chunks` as SSE with `delay` seconds between them."""
    async def messages(request: Request):
        body = await request.body()
        if log is not None:
            log.append({"headers": dict(request.headers), "body": body, "path": request.url.path})

        async def gen():
            for c in chunks:
                yield c
                if delay:
                    await asyncio.sleep(delay)
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"x-upstream-id": "abc123"})

    async def other(request: Request):
        return Response(b"hello", 418, headers={"x-custom": "1"})

    return Starlette(routes=[
        Route("/v1/messages", messages, methods=["POST"]),
        Route("/{p:path}", other, methods=["GET", "POST"]),
    ])


@pytest.fixture(autouse=True)
def _never_write_to_the_real_data_dir():
    """Fail loudly if a test creates records under ./data, which is where live traffic is traced."""
    from pathlib import Path
    real = Path("data") / "calls"
    before = {f: f.stat().st_size for f in real.glob("*.jsonl")} if real.exists() else {}
    yield
    after = {f: f.stat().st_size for f in real.glob("*.jsonl")} if real.exists() else {}
    assert after == before, "a test wrote to the real ./data directory; give its Config a tmp_path data_dir"


@pytest.fixture
def proxy_factory(tmp_path):
    stack = []

    def build(chunks=(b"data: x\n\n",), delay=0.0, **cfg_over):
        log = []
        up = Server(make_stub(list(chunks), delay, log), free_port()).__enter__()
        stack.append(up)
        cfg_over.setdefault("data_dir", str(tmp_path))
        cfg = Config(upstream={"base_url": f"http://127.0.0.1:{up.port}"}, **cfg_over)
        px = Server(create_app(cfg), free_port()).__enter__()
        stack.append(px)
        build.data_dir = tmp_path
        return f"http://127.0.0.1:{px.port}", f"http://127.0.0.1:{up.port}", log

    yield build
    for s in reversed(stack):
        s.__exit__()
