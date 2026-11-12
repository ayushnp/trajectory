"""Ingress. Own endpoints live under /healthz and /stats; everything else is relayed."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import aiohttp
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import api
from .chain import Chainer
from .config import Config
from .green import Estimator
from .proxy import make_client, relay
from .stats import Stats
from .subagents import SubagentAttacher
from .tools import ToolPairer
from .tee import Tee

# NOTE: deliberately no GZipMiddleware / compression of any kind (spec §25).


async def healthz(request: Request) -> JSONResponse:
    state = request.app.state
    try:
        # Any HTTP response, even 302/404, proves Tux is reachable.
        async with state.client.get("/", timeout=aiohttp.ClientTimeout(total=2), allow_redirects=False):
            pass
        reachable = True
    except (aiohttp.ClientError, TimeoutError):
        reachable = False
    return JSONResponse({
        "status": "ok" if reachable else "degraded",
        "uptime_s": round(time.monotonic() - state.stats.started, 1),
        "upstream_reachable": reachable,
        "queue_depth": state.tee.depth,
    })


async def stats(request: Request) -> JSONResponse:
    return JSONResponse(request.app.state.stats.snapshot(request.app.state.tee))


def create_app(cfg: Config) -> Starlette:
    @asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.client = make_client(
            cfg.upstream.base_url, cfg.upstream.connect_timeout_s, cfg.upstream.read_timeout_s
        )
        yield
        await app.state.client.close()
        app.state.tee.close()  # drain queued records to disk

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
    app = Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/stats", stats, methods=["GET"]),
            *api.routes(),
            Route("/{path:path}", relay, methods=methods),
        ],
        lifespan=lifespan,
    )
    app.state.cfg = cfg
    app.state.stats = Stats()
    app.state.index = api.RecordIndex(cfg.data_dir)
    app.state.tee = Tee(cfg.queue.maxsize, cfg.data_dir, cfg.queue.fsync_interval_s,
                        chainer=Chainer(cfg.chain.candidate_ttl_s, cfg.chain.max_steps),
                        pairer=ToolPairer(cfg.chain.candidate_ttl_s),
                        attacher=SubagentAttacher(cfg.chain.candidate_ttl_s),
                        green=Estimator(cfg.green.enabled, cfg.green.model_map, cfg.green.assumptions))
    return app
