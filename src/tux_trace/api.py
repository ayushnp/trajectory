"""Read-only API + static UI. Lives under /_trace so it can never shadow a path Tux serves.

The index reads the JSONL files incrementally (tracking a byte offset per file), so polling
is cheap. JSONL stays the source of truth; nothing here writes.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

STATIC = Path(__file__).parent / "static"
MAX_IN_MEMORY = 20_000


class RecordIndex:
    def __init__(self, data_dir: str | Path) -> None:
        self.dir = Path(data_dir) / "calls"
        self._offsets: dict[Path, int] = {}
        self._records: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._lock = threading.Lock()

    def refresh(self) -> None:
        with self._lock:
            if not self.dir.exists():
                return
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
            if len(self._records) > MAX_IN_MEMORY:
                for old in self._records[:-MAX_IN_MEMORY]:
                    self._by_id.pop(old["call_id"], None)
                self._records = self._records[-MAX_IN_MEMORY:]

    def recent(self, limit: int, offset: int) -> tuple[list[dict], int]:
        self.refresh()
        with self._lock:
            ordered = sorted(self._records, key=lambda r: r["timing"]["received_at"], reverse=True)
            return ordered[offset:offset + limit], len(ordered)

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
    }


async def calls(request: Request) -> JSONResponse:
    try:
        limit = max(1, min(int(request.query_params.get("limit", 100)), 500))
        offset = max(0, int(request.query_params.get("offset", 0)))
    except ValueError:
        return JSONResponse({"error": "limit/offset must be integers"}, 400)
    rows, total = request.app.state.index.recent(limit, offset)
    return JSONResponse({"total": total, "calls": [_row(r) for r in rows]})


async def call_detail(request: Request) -> JSONResponse:
    rec = request.app.state.index.get(request.path_params["call_id"])
    return JSONResponse(rec) if rec else JSONResponse({"error": "not found"}, 404)


async def ui(request: Request) -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"cache-control": "no-store"})


def routes() -> list:
    return [
        Route("/_trace", ui, methods=["GET"]),
        Route("/_trace/", ui, methods=["GET"]),
        Route("/_trace/api/calls", calls, methods=["GET"]),
        Route("/_trace/api/calls/{call_id}", call_detail, methods=["GET"]),
        Mount("/_trace/static", StaticFiles(directory=STATIC), name="static"),
    ]
