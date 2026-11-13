"""Read-only API + static UI. Lives under /_trace so it can never shadow a path Tux serves.

The index reads the JSONL files incrementally (tracking a byte offset per file), so polling
is cheap. JSONL stays the source of truth; nothing here writes.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import sessions as sess
from .tools import join_results

STATIC = Path(__file__).parent / "static"
MAX_IN_MEMORY = 20_000


class RecordIndex:
    def __init__(self, data_dir: str | Path) -> None:
        self.dir = Path(data_dir) / "calls"
        self._offsets: dict[Path, int] = {}
        self._records: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _join(self, records: list[dict]) -> None:
        """Fold tool results back onto the calls they answer. Done on read: JSONL is append-only."""
        join_results(records)

    def refresh(self) -> None:
        with self._lock:
            if not self.dir.exists():
                return
            added = False
            for f in sorted(self.dir.glob("*.jsonl")):
                pos = self._offsets.get(f, 0)
                if f.stat().st_size <= pos:
                    continue
                with open(f, "rb") as fh:
                    fh.seek(pos)
                    data = fh.read()
                end = data.rfind(b"\n") + 1          # only consume complete lines
                for line in data[:end].splitlines():
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    self._records.append(rec)
                    self._by_id[rec["call_id"]] = rec
                self._offsets[f] = pos + end
                added = True
            if added:
                self._join(self._records)
            if len(self._records) > MAX_IN_MEMORY:
                for old in self._records[:-MAX_IN_MEMORY]:
                    self._by_id.pop(old["call_id"], None)
                self._records = self._records[-MAX_IN_MEMORY:]

    def recent(self, limit: int, offset: int) -> tuple[list[dict], int]:
        self.refresh()
        with self._lock:
            ordered = sorted(self._records, key=lambda r: r["timing"]["received_at"], reverse=True)
            return ordered[offset:offset + limit], len(ordered)

    def all(self) -> list[dict]:
        self.refresh()
        with self._lock:
            return list(self._records)

    def get(self, call_id: str) -> dict | None:
        self.refresh()
        with self._lock:
            return self._by_id.get(call_id)


def _row(r: dict) -> dict:
    """The compact per-call view the table needs."""
    return {
        "call_id": r["call_id"],
        "received_at": r["timing"]["received_at"],
        "format": r["request"]["format"],
        "path": r["request"]["path"],
        "model": r["response"].get("model_served") or r["request"].get("model_requested"),
        "status": r["response"]["status"],
        "stop_reason": r["response"].get("stop_reason"),
        "ttft_any_ms": r["timing"].get("ttft_any_ms"),
        "ttft_visible_ms": r["timing"].get("ttft_visible_ms"),
        "total_ms": r["timing"].get("total_ms"),
        "duration_ms": r["timing"].get("duration_ms"),
        "usage": r["usage"],
        "tool_calls": [t.get("name") for t in r["response"].get("tool_calls") or []],
        "thinking": r["response"].get("thinking_visibility"),
        "incomplete": r["outcome"]["incomplete"],
        "disconnected": r["outcome"]["client_disconnected"],
        "error": r["outcome"]["error"],
        "harness": r["client"]["harness_hint"],
        "session_id": r.get("session_id"),
        "green": r.get("green"),
    }


async def calls(request: Request) -> JSONResponse:
    try:
        limit = max(1, min(int(request.query_params.get("limit", 100)), 500))
        offset = max(0, int(request.query_params.get("offset", 0)))
    except ValueError:
        return JSONResponse({"error": "limit/offset must be integers"}, 400)
    sq = request.app.state.sqlite
    rows, total = sq.recent_docs(limit, offset) if sq else request.app.state.index.recent(limit, offset)
    return JSONResponse({"total": total, "calls": [_row(r) for r in rows]})


def _get_call(request: Request, call_id: str) -> dict | None:
    sq = request.app.state.sqlite
    return sq.get_joined(call_id) if sq else request.app.state.index.get(call_id)


async def call_detail(request: Request) -> JSONResponse:
    rec = _get_call(request, request.path_params["call_id"])
    return JSONResponse(rec) if rec else JSONResponse({"error": "not found"}, 404)


def _paging(request: Request, default: int = 50):
    limit = max(1, min(int(request.query_params.get("limit", default)), 500))
    offset = max(0, int(request.query_params.get("offset", 0)))
    return limit, offset


async def sessions_list(request: Request) -> JSONResponse:
    try:
        limit, offset = _paging(request)
    except ValueError:
        return JSONResponse({"error": "limit/offset must be integers"}, 400)
    sq = request.app.state.sqlite
    if sq is None:
        items, total = sess.list_sessions(request.app.state.index.all(), limit, offset)
        return JSONResponse({"total": total, "sessions": items})
    ids, total = sq.session_page(limit, offset)
    records = sq.records_for_sessions(ids)
    join_results(records)
    by_id = {r["call_id"]: r for r in records}
    groups = sess.group_sessions(records)
    return JSONResponse({"total": total, "sessions": [sess.session_summary(sid, groups[sid], by_id)
                                                      for sid in ids if sid in groups]})


async def session_trajectory(request: Request) -> JSONResponse:
    sid = request.path_params["session_id"]
    sq = request.app.state.sqlite
    if sq is None:
        records = request.app.state.index.all()
    else:
        records = sq.trajectory_records(sid)
        join_results(records)
    t = sess.trajectory(sid, records)
    return JSONResponse(t) if t else JSONResponse({"error": "not found"}, 404)


async def call_conversation(request: Request) -> JSONResponse:
    """Readable prompt/response for one call (new messages only, streamed text joined)."""
    resp = await call_content(request)
    if resp.status_code != 200:
        return resp
    from .conversation import build
    return JSONResponse(build(json.loads(resp.body)))


async def call_content(request: Request) -> JSONResponse:
    """Stored (redacted) prompt/response text for one call, when store_content was on."""
    rec = _get_call(request, request.path_params["call_id"])
    if rec is None:
        return JSONResponse({"error": "not found"}, 404)
    c = rec.get("content") or {}
    if not c.get("stored") or not c.get("path"):
        return JSONResponse({"stored": False, "reason": "no content was stored for this call"}, 404)
    root = Path(request.app.state.cfg.data_dir).resolve()
    path = (root / c["path"]).resolve()
    # the path comes from our own record, but never serve anything outside the data directory
    if root not in path.parents or not path.is_file():
        return JSONResponse({"stored": False, "reason": "content file is gone (retention?)"}, 404)
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


async def favicon(request: Request) -> Response:
    """Browsers ask for a tab icon on every page load; answer it here so it is never relayed or recorded."""
    return Response(status_code=204, headers={"cache-control": "public, max-age=86400"})


async def ui(request: Request) -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"cache-control": "no-store"})


def routes() -> list:
    return [
        Route("/favicon.ico", favicon, methods=["GET"]),
        Route("/_trace", ui, methods=["GET"]),
        Route("/_trace/", ui, methods=["GET"]),
        Route("/_trace/api/calls", calls, methods=["GET"]),
        Route("/_trace/api/calls/{call_id}", call_detail, methods=["GET"]),
        Route("/_trace/api/calls/{call_id}/content", call_content, methods=["GET"]),
        Route("/_trace/api/calls/{call_id}/conversation", call_conversation, methods=["GET"]),
        Route("/_trace/api/sessions", sessions_list, methods=["GET"]),
        Route("/_trace/api/sessions/{session_id}/trajectory", session_trajectory, methods=["GET"]),
        Mount("/_trace/static", StaticFiles(directory=STATIC), name="static"),
    ]
