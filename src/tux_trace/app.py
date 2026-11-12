"""Ingress. Own endpoints live under /healthz and /stats; everything else is relayed."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from . import api
from .chain import Chainer
from .config import Config
from .green import Estimator
from .proxy import make_client, relay
from .stats import Stats
from .subagents import SubagentAttacher
from .tools import ToolPairer
from .content import ContentStore, Redactor
from .index_db import SqliteIndex
from .record import BUDGET
from .retention import RetentionTask
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


# Monotonic counters; everything else in /stats is a gauge (a level, share or percentile).
COUNTERS = {
    "calls_total", "calls_incomplete", "calls_error", "calls_sampled_out", "dropped_records",
    "records_written", "record_errors", "chain_errors", "green_errors", "parse_errors",
    "parse_unknown_events", "subagents_attached", "subagents_unattached", "capture_budget_denied",
    "content_stored", "content_errors", "index_errors",
}


async def metrics(request: Request) -> PlainTextResponse:
    """The same numbers as /stats in Prometheus text format. Non-numeric / null values are omitted."""
    snap = request.app.state.stats.snapshot(request.app.state.tee)
    lines = []
    for k, v in snap.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        name = "tux_trace_" + k
        kind = "counter" if k in COUNTERS else "gauge"
        lines += [f"# TYPE {name} {kind}", f"{name} {v}"]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


def create_app(cfg: Config) -> Starlette:
    @asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.client = make_client(
            cfg.upstream.base_url, cfg.upstream.connect_timeout_s, cfg.upstream.read_timeout_s
        )
        app.state.retention.start()
        yield
        app.state.retention.stop()
        await app.state.client.close()
        app.state.tee.close()  # drain queued records to disk
        if app.state.sqlite is not None:
            app.state.sqlite.close()

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
    app = Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/stats", stats, methods=["GET"]),
            Route("/metrics", metrics, methods=["GET"]),
            *api.routes(),
            Route("/{path:path}", relay, methods=methods),
        ],
        lifespan=lifespan,
    )
    app.state.cfg = cfg
    app.state.stats = Stats()
    app.state.index = api.RecordIndex(cfg.data_dir)
    BUDGET.configure(cfg.queue.capture_budget_mb * 1024 * 1024)
    app.state.sqlite = SqliteIndex(Path(cfg.data_dir) / "index.db") if cfg.sqlite_index else None
    app.state.tee = Tee(cfg.queue.maxsize, cfg.data_dir, cfg.queue.fsync_interval_s,
                        chainer=Chainer(cfg.chain.candidate_ttl_s, cfg.chain.max_steps),
                        pairer=ToolPairer(cfg.chain.candidate_ttl_s),
                        attacher=SubagentAttacher(cfg.chain.candidate_ttl_s),
                        green=Estimator(cfg.green.enabled, cfg.green.model_map, cfg.green.assumptions),
                        content=ContentStore(cfg.data_dir, cfg.store_content,
                                             Redactor(cfg.redact_patterns, cfg.redact_defaults)),
                        index=app.state.sqlite)
    app.state.retention = RetentionTask(cfg.data_dir, cfg.retention_days, app.state.sqlite,
                                        cfg.retention_interval_s)
    return app
