"""Session and trajectory views built from call records (spec sections 19-20). Pure functions.

Honesty rules baked in:
  * Energy / emissions are ranges (min..max, plus mean). A session total is the sum of its calls'
    ranges. Calls whose estimate rests on a non-exact basis are counted and named, never mixed in
    silently: `green.basis_counts` says how many calls used which basis.
  * A token total sums only the calls that reported that metric, and says how many did not.
    Nothing unreported is turned into 0.
  * gap_ms is an upper bound (tool + agent + network + human wait). It is labelled "gap".
"""
from __future__ import annotations

from collections import Counter

from .timing import percentile

METRICS = ("input_total", "output", "cache_read", "cache_write")


def _tokens(calls: list[dict]) -> dict:
    out = {}
    for m in METRICS:
        vals = [c["usage"][m]["value"] for c in calls if c["usage"].get(m, {}).get("reported")
                and c["usage"][m].get("value") is not None]
        out[m] = {"value": sum(vals) if vals else None, "reported": bool(vals),
                  "calls_reported": len(vals), "calls_missing": len(calls) - len(vals)}
    return out


def _green(calls: list[dict]) -> dict:
    ok = [c for c in calls if (c.get("green") or {}).get("status") == "ok"]
    basis = Counter(c["green"]["basis"] for c in ok)
    res: dict = {"calls_estimated": len(ok), "calls_without_estimate": len(calls) - len(ok),
                 "basis_counts": dict(basis), "all_exact": bool(ok) and set(basis) == {"exact"}}
    for key in ("energy_kwh", "gwp_kgco2eq"):
        if ok:
            res[key] = {"min": sum(c["green"][key]["min"] for c in ok),
                        "mean": sum(c["green"][key]["mean"] for c in ok),
                        "max": sum(c["green"][key]["max"] for c in ok)}
        else:
            res[key] = None
    res["note"] = "estimated, not measured" if ok else None
    return res


def _ttft_p50(calls: list[dict]) -> float | None:
    vals = sorted(c["timing"]["ttft_any_ms"] for c in calls if c["timing"].get("ttft_any_ms") is not None)
    v = percentile(vals, 0.5)
    return None if v is None else round(v, 1)


def real_calls(records: list[dict]) -> list[dict]:
    """Model calls only: records the chainer could place in a session."""
    return [r for r in records if r.get("session_id")]


def group_sessions(records: list[dict]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for r in real_calls(records):
        by.setdefault(r["session_id"], []).append(r)
    for calls in by.values():
        calls.sort(key=lambda r: (r["step_index"] or 0, r["timing"]["received_at"]))
    return by


def session_summary(sid: str, calls: list[dict], records_by_id: dict[str, dict]) -> dict:
    first, last = calls[0], calls[-1]
    root = first
    parent = records_by_id.get(root.get("parent_call_id") or "")
    return {
        "session_id": sid,
        "first_seen": min(c["timing"]["received_at"] for c in calls),
        "last_seen": max(c["timing"]["received_at"] for c in calls),
        "steps": len(calls),
        "turns": max((c["turn_index"] or 0) for c in calls) + 1,
        "models": sorted({c["response"].get("model_served") or c["request"].get("model_requested") or "?" for c in calls}),
        "harness": root["client"]["harness_hint"],
        "tokens": _tokens(calls),
        "ttft_p50_ms": _ttft_p50(calls),
        "green": _green(calls),
        "incomplete_calls": sum(1 for c in calls if c["outcome"]["incomplete"]),
        "error_calls": sum(1 for c in calls if c["outcome"]["error"]),
        "compaction_suspected": any(c.get("compaction_suspected") for c in calls),
        "continued_from_session": root.get("continued_from_session"),
        # subagent relationship, as inferred (never stronger than the evidence)
        "parent_session_id": parent["session_id"] if parent else None,
        "parent_call_id": root.get("parent_call_id"),
        "subagent_edge": root.get("subagent_edge") if root.get("parent_call_id") else None,
        "unattached": bool(root.get("unattached")),
        "overlap_candidates": root.get("overlap_candidates"),
    }


def list_sessions(records: list[dict], limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    by_id = {r["call_id"]: r for r in records}
    groups = group_sessions(records)
    summaries = [session_summary(sid, calls, by_id) for sid, calls in groups.items()]
    summaries.sort(key=lambda s: s["last_seen"], reverse=True)
    return summaries[offset:offset + limit], len(summaries)


def _step(c: dict, prev: dict | None) -> dict:
    gap = None
    if prev is not None and prev["timing"].get("t_end") is not None:
        gap = round((c["timing"]["t_received"] - prev["timing"]["t_end"]) / 1e6, 1)
    return {
        "call_id": c["call_id"], "step_index": c["step_index"], "turn_index": c["turn_index"],
        "received_at": c["timing"]["received_at"],
        "model": c["response"].get("model_served") or c["request"].get("model_requested"),
        "format": c["request"]["format"], "status": c["response"]["status"],
        "stop_reason": c["response"].get("stop_reason"),
        "ttft_any_ms": c["timing"].get("ttft_any_ms"), "ttft_visible_ms": c["timing"].get("ttft_visible_ms"),
        "total_ms": c["timing"].get("total_ms"), "duration_ms": c["timing"].get("duration_ms"),
        "usage": {m: c["usage"][m] for m in METRICS},
        "gap_since_previous_ms": gap,      # upper bound; labelled "gap" in the UI, never "tool time"
        "green": c.get("green"),
        "tool_calls": [{
            "id": t.get("id"), "name": t.get("name"), "args": t.get("args"),
            "result_present": t.get("result_present", False), "result_bytes": t.get("result_bytes"),
            "result_is_error": t.get("result_is_error"), "gap_ms": t.get("gap_ms"),
        } for t in c["response"].get("tool_calls") or []],
        "incomplete": c["outcome"]["incomplete"], "client_disconnected": c["outcome"]["client_disconnected"],
        "error": c["outcome"]["error"], "retry_index": c["outcome"].get("retry_index", 0),
        "compaction_suspected": bool(c.get("compaction_suspected")),
        "opaque_reasoning": c["request"].get("opaque_reasoning", False),
        "thinking": c["response"].get("thinking_visibility"),
    }


def trajectory(sid: str, records: list[dict]) -> dict | None:
    by_id = {r["call_id"]: r for r in records}
    groups = group_sessions(records)
    calls = groups.get(sid)
    if not calls:
        return None
    steps, prev = [], None
    for c in calls:
        steps.append(_step(c, prev))
        if not c["outcome"].get("retry_index"):
            prev = c
    step_ids = {c["call_id"] for c in calls}

    # Children: sessions whose root call is attached to a tool call in THIS session.
    children, unattached = [], []
    for csid, ccalls in groups.items():
        if csid == sid:
            continue
        root = ccalls[0]
        if root.get("parent_call_id") in step_ids:
            children.append({
                "session": session_summary(csid, ccalls, by_id),
                "edge": root.get("subagent_edge"),                     # confidence + signals + score
                "parent_call_id": root["parent_call_id"],
            })
        elif root.get("unattached") and sid in (root.get("overlapped_sessions") or []):
            unattached.append(session_summary(csid, ccalls, by_id))
    children.sort(key=lambda c: c["session"]["first_seen"])
    unattached.sort(key=lambda s: s["first_seen"])

    summary = session_summary(sid, calls, by_id)
    return {"session": summary, "steps": steps, "children": children, "unattached": unattached}
