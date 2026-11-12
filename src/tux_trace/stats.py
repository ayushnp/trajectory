"""Self-metrics. Plain counters; only touched from the event loop thread."""
from __future__ import annotations

import time
from collections import deque


class Stats:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.calls_total = 0
        self.calls_incomplete = 0
        self.calls_error = 0
        self.record_errors = 0
        self._overhead_ms: deque[float] = deque(maxlen=5000)

    def observe_overhead_ns(self, ns: int) -> None:
        self.calls_total += 1
        self._overhead_ms.append(ns / 1e6)

    def _pct(self, q: float) -> float | None:
        if not self._overhead_ms:
            return None
        data = sorted(self._overhead_ms)
        return round(data[min(len(data) - 1, int(q * len(data)))], 3)

    def snapshot(self, tee=None) -> dict:
        return {
            "calls_total": self.calls_total,
            "calls_incomplete": self.calls_incomplete,
            "calls_error": self.calls_error,
            "dropped_records": tee.dropped_records if tee else 0,
            "records_written": tee.writer.written if tee else 0,
            "record_errors": self.record_errors + (tee.writer.write_errors if tee else 0),
            "queue_depth": tee.depth if tee else 0,
            "parse_errors": tee.writer.parse_errors if tee else 0,
            "parse_unknown_events": tee.writer.parse_unknown_events if tee else 0,
            "unknown_token_share": (round(tee.writer.usage_unreported / tee.writer.parsed_calls, 4)
                                    if tee and tee.writer.parsed_calls else None),
            "overhead_ms_p50": self._pct(0.50),
            "overhead_ms_p95": self._pct(0.95),
        }
