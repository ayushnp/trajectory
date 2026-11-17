"""Incremental SSE reader. Feed it the bytes already forwarded; it never touches the stream."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SSEEvent:
    event: str | None
    data: str
    t: int  # monotonic ns arrival of the chunk that completed this event


class SSEReader:
    """Splits on \\n, tolerates \\r\\n, ignores comments (': ping') and blank-line noise."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._event: str | None = None
        self._data: list[str] = []

    def feed(self, chunk: bytes, t: int) -> list[SSEEvent]:
        self._buf += chunk
        out: list[SSEEvent] = []
        buf = self._buf
        while True:
            i = buf.find(b"\n")
            if i < 0:
                break
            line = bytes(buf[:i])
            del buf[:i + 1]
            ev = self._line(line.removesuffix(b"\r"), t)
            if ev is not None:
                out.append(ev)
        return out

    def flush(self, t: int) -> list[SSEEvent]:
        """Stream ended: an unterminated final line / event is still worth keeping."""
        out: list[SSEEvent] = []
        if self._buf:
            line = bytes(self._buf).removesuffix(b"\r")
            self._buf.clear()
            ev = self._line(line, t)
            if ev is not None:
                out.append(ev)
        ev = self._dispatch(t)
        if ev is not None:
            out.append(ev)
        return out

    def _line(self, line: bytes, t: int) -> SSEEvent | None:
        if not line:
            return self._dispatch(t)
        if line.startswith(b":"):
            return None
        name, _, value = line.partition(b":")
        value = value.removeprefix(b" ")
        field = name.decode("utf-8", "replace")
        if field == "data":
            self._data.append(value.decode("utf-8", "replace"))
        elif field == "event":
            self._event = value.decode("utf-8", "replace")
        return None  # id / retry / unknown fields are ignored

    def _dispatch(self, t: int) -> SSEEvent | None:
        if not self._data and self._event is None:
            return None
        ev = SSEEvent(self._event, "\n".join(self._data), t)
        self._event, self._data = None, []
        return ev
