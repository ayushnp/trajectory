"""Aggregate many calls' green estimates into the numbers the Green view shows. Pure functions.

Honesty rules (the same ones sessions.py follows):
  * Everything is an estimate, and every figure stays a {min, mean, max} range. Ranges are summed
    component-wise, which is what the per-session totals already do. A sum of ranges is not a
    confidence interval.
  * Only MODEL calls are counted. Health probes, favicons and other relayed requests have no tokens
    and would otherwise look like "calls with no estimate".
  * A call without an estimate is counted and its reason named; it is never treated as zero energy.
  * Water is absent from older records. It is summed over the calls that have it and the count of
    those calls is reported, so a partial water total is never presented as the whole.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

MAX_BUCKETS = 400
HOURLY_UP_TO = timedelta(hours=72)
RANGES = ("energy_kwh", "gwp_kgco2eq", "water_l")


def slim(rec: dict) -> dict:
    """The few fields the aggregation needs, from a full call record."""
    return {
        "received_at": (rec.get("timing") or {}).get("received_at"),
        "session_id": rec.get("session_id"),
        "model": (rec.get("response") or {}).get("model_served")
                 or (rec.get("request") or {}).get("model_requested") or "?",
        "harness": (rec.get("client") or {}).get("harness_hint"),
        "green": rec.get("green"),
    }


def _zero() -> dict:
    return {"min": 0.0, "mean": 0.0, "max": 0.0}


def _add(acc: dict, r: dict | None) -> bool:
    if not isinstance(r, dict) or any(not isinstance(r.get(k), (int, float)) for k in ("min", "mean", "max")):
        return False
    for k in ("min", "mean", "max"):
        acc[k] += r[k]
    return True


def _sum_ranges(greens: list[dict]) -> dict:
    out = {k: _zero() for k in RANGES}
    n = {k: 0 for k in RANGES}
    for g in greens:
        for k in RANGES:
            if _add(out[k], g.get(k)):
                n[k] += 1
    return {"sums": {k: (out[k] if n[k] else None) for k in RANGES}, "counts": n}


def _parse(ts: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(ts) if ts else None
    except ValueError:
        return None


def _floor(dt: datetime, unit: str) -> datetime:
    dt = dt.astimezone(timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0) if unit == "hour" \
        else dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _timeline(ok: list[dict]) -> dict:
    stamped = [(d, r) for r in ok if (d := _parse(r.get("received_at"))) is not None]
    if not stamped:
        return {"bucket": "hour", "points": []}
    lo, hi = min(d for d, _ in stamped), max(d for d, _ in stamped)
    unit = "hour" if hi - lo <= HOURLY_UP_TO else "day"
    step = timedelta(hours=1) if unit == "hour" else timedelta(days=1)
    if (hi - lo) / step > MAX_BUCKETS:
        unit, step = "day", timedelta(days=1)
    buckets: dict[datetime, list[dict]] = {}
    for d, r in stamped:
        buckets.setdefault(_floor(d, unit), []).append(r["green"])
    points, t, end = [], _floor(lo, unit), _floor(hi, unit)
    while t <= end and len(points) <= MAX_BUCKETS:
        gs = buckets.get(t, [])
        agg = _sum_ranges(gs)
        points.append({
            "t": t.isoformat(), "calls": len(gs),
            "energy_kwh": agg["sums"]["energy_kwh"],
            "output_tokens": sum((g.get("inputs") or {}).get("output_tokens") or 0 for g in gs),
        })
        t += step
    return {"bucket": unit, "points": points}


def aggregate(rows: list[dict], hours: float | None = None, now: datetime | None = None) -> dict:
    """rows: slim() dicts of MODEL calls only."""
    now = now or datetime.now(timezone.utc)
    ok = [r for r in rows if (r.get("green") or {}).get("status") == "ok"]
    without: Counter = Counter()
    for r in rows:
        g = r.get("green") or {}
        if g.get("status") != "ok":
            without[g.get("reason") or g.get("error") or g.get("status") or "no estimate block"] += 1

    greens = [r["green"] for r in ok]
    total = _sum_ranges(greens)
    in_tok = sum((g.get("inputs") or {}).get("input_tokens") or 0 for g in greens)
    out_tok = sum((g.get("inputs") or {}).get("output_tokens") or 0 for g in greens)
    cache_tok = sum((g.get("inputs") or {}).get("cache_read_tokens") or 0 for g in greens)
    e_mean = (total["sums"]["energy_kwh"] or {}).get("mean")

    by_model: dict[str, list[dict]] = {}
    by_session: dict[str, list[dict]] = {}
    for r in ok:
        by_model.setdefault(r["model"], []).append(r)
        if r.get("session_id"):
            by_session.setdefault(r["session_id"], []).append(r)

    models = []
    for model, rs in by_model.items():
        gs = [r["green"] for r in rs]
        agg = _sum_ranges(gs)
        bases = Counter(g.get("basis") for g in gs)
        models.append({
            "model": model, "calls": len(rs), "basis": bases.most_common(1)[0][0], "bases": dict(bases),
            "output_tokens": sum((g.get("inputs") or {}).get("output_tokens") or 0 for g in gs),
            "ecologits_model": next((g.get("inputs", {}).get("ecologits_model") for g in gs), None),
            "energy_kwh": agg["sums"]["energy_kwh"], "gwp_kgco2eq": agg["sums"]["gwp_kgco2eq"],
            "water_l": agg["sums"]["water_l"],
        })
    models.sort(key=lambda m: -(m["energy_kwh"] or {}).get("mean", 0))

    sessions = []
    for sid, rs in by_session.items():
        agg = _sum_ranges([r["green"] for r in rs])
        stamps = sorted(r["received_at"] for r in rs if r.get("received_at"))
        sessions.append({
            "session_id": sid, "calls": len(rs), "harness": rs[0].get("harness"),
            "models": sorted({r["model"] for r in rs}),
            "first_seen": stamps[0] if stamps else None, "last_seen": stamps[-1] if stamps else None,
            "energy_kwh": agg["sums"]["energy_kwh"], "gwp_kgco2eq": agg["sums"]["gwp_kgco2eq"],
        })
    sessions.sort(key=lambda s: -(s["energy_kwh"] or {}).get("mean", 0))

    stamps = sorted(r["received_at"] for r in rows if r.get("received_at"))
    return {
        "generated_at": now.isoformat(), "hours": hours,
        "window": {"first": stamps[0] if stamps else None, "last": stamps[-1] if stamps else None},
        "calls": {"model_calls": len(rows), "estimated": len(ok), "without_estimate": len(rows) - len(ok),
                  "without_reasons": dict(without),
                  "unchained": sum(1 for r in ok if not r.get("session_id"))},
        "totals": {**total["sums"], "water_calls": total["counts"]["water_l"]},
        "tokens": {"input": in_tok, "output": out_tok, "cache_read": cache_tok},
        "per_call": {
            "energy_kwh_mean": (e_mean / len(ok)) if ok and e_mean is not None else None,
            "energy_kwh_per_1k_output_tokens": (e_mean / (out_tok / 1000)) if out_tok and e_mean is not None else None,
        },
        "basis": dict(Counter(g.get("basis") for g in greens)),
        "warnings": dict(Counter(w for g in greens for w in (g.get("warnings") or []))),
        "by_model": models,
        "by_session": sessions[:10], "sessions_total": len(sessions),
        "timeline": _timeline(ok),
        "assumptions": {
            "methods": sorted({f"{g.get('method')} {g.get('method_version')}" for g in greens}),
            "regions": sorted({(g.get("inputs") or {}).get("region") for g in greens
                               if (g.get("inputs") or {}).get("region")}),
            "assumption_hashes": dict(Counter(g.get("assumptions_sha256") for g in greens if g.get("assumptions_sha256"))),
            "notes": [n for n, _ in Counter(g.get("note") for g in greens if g.get("note")).most_common()],
        },
    }
