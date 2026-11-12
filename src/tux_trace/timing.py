"""Timing derivations (spec section 10). All inputs are time.monotonic_ns() values.

A mark is the arrival time, at the proxy, of the upstream chunk that completed the event. It
is therefore chunk-granular: two events in one chunk share a timestamp. A mark that never
occurred is None, never 0.
"""
from __future__ import annotations

import math


def _ms(later: int | None, earlier: int | None) -> float | None:
    if later is None or earlier is None:
        return None
    return round((later - earlier) / 1e6, 3)


def percentile(sorted_values: list[float], q: float) -> float | None:
    """Nearest-rank percentile of an already sorted list."""
    if not sorted_values:
        return None
    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


def derive(t_received: int, t_first_content: int | None, t_first_visible: int | None,
           t_last_content: int | None, chunk_times: list[int]) -> dict:
    gaps = sorted((b - a) / 1e6 for a, b in zip(chunk_times, chunk_times[1:]))
    p50, p95 = percentile(gaps, 0.50), percentile(gaps, 0.95)
    return {
        "t_first_content": t_first_content,
        "t_first_visible": t_first_visible,
        "t_last_content": t_last_content,
        "ttft_any_ms": _ms(t_first_content, t_received),
        "ttft_visible_ms": _ms(t_first_visible, t_received),
        "total_ms": _ms(t_last_content, t_received),
        "inter_chunk_p50_ms": None if p50 is None else round(p50, 3),
        "inter_chunk_p95_ms": None if p95 is None else round(p95, 3),
    }
