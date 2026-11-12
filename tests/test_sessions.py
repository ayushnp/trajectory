"""M7 server side: session aggregation, trajectory, and the honesty rules for totals."""
from tux_trace import sessions as S

NS = 1_000_000_000


def metric(v):
    return {"value": v, "reported": v is not None}


def call(cid, sess, step, turn=0, t=0.0, dur=1.0, out=10, inp=100, cache=None, green="exact",
         energy=(1e-5, 2e-5, 3e-5), tools=(), parent=None, edge=None, unattached=False, overlapped=(),
         ttft=500.0, error=None, incomplete=False, model="claude-haiku-5-5"):
    g = None
    if green == "none":
        g = {"status": "unavailable"}
    elif green:
        g = {"status": "ok", "basis": green,
             "energy_kwh": dict(zip(("min", "mean", "max"), energy)),
             "gwp_kgco2eq": dict(zip(("min", "mean", "max"), (e / 2 for e in energy)))}
    r = {"call_id": cid, "session_id": sess, "step_index": step, "turn_index": turn,
         "client": {"harness_hint": "claude-code"},
         "request": {"format": "anthropic", "model_requested": model, "opaque_reasoning": False},
         "response": {"status": 200, "model_served": model, "stop_reason": "end_turn",
                      "tool_calls": [dict(x) for x in tools]},
         "timing": {"received_at": f"2026-01-01T00:00:{int(t):02d}+00:00", "t_received": int(t * NS),
                    "t_end": int((t + dur) * NS), "ttft_any_ms": ttft, "ttft_visible_ms": ttft,
                    "total_ms": dur * 1000, "duration_ms": dur * 1000},
         "usage": {"input_total": metric(inp), "output": metric(out), "cache_read": metric(cache),
                   "cache_write": metric(None)},
         "outcome": {"error": error, "incomplete": incomplete, "client_disconnected": False, "retry_index": 0},
         "green": g, "subagent_edge": edge, "unattached": unattached, "overlapped_sessions": list(overlapped)}
    if parent:
        r["parent_call_id"] = parent
    return r


def summary(records, sid):
    items, _ = S.list_sessions(records)
    return next(s for s in items if s["session_id"] == sid)


# ---------------------------------------------------------------- totals are honest
def test_token_totals_sum_only_reported_values_and_say_how_many_were_missing():
    recs = [call("a", "s", 0, cache=900), call("b", "s", 1, cache=None), call("c", "s", 2, cache=100)]
    t = summary(recs, "s")["tokens"]
    assert t["cache_read"] == {"value": 1000, "reported": True, "calls_reported": 2, "calls_missing": 1}
    assert t["output"]["value"] == 30 and t["output"]["calls_missing"] == 0


def test_a_metric_nobody_reported_is_none_not_zero():
    t = summary([call("a", "s", 0), call("b", "s", 1)], "s")["tokens"]
    assert t["cache_write"] == {"value": None, "reported": False, "calls_reported": 0, "calls_missing": 2}


def test_energy_total_is_the_sum_of_ranges_never_a_bare_number():
    recs = [call("a", "s", 0, energy=(1e-5, 2e-5, 3e-5)), call("b", "s", 1, energy=(2e-5, 4e-5, 6e-5))]
    g = summary(recs, "s")["green"]["energy_kwh"]
    assert set(g) == {"min", "mean", "max"}
    assert abs(g["min"] - 3e-5) < 1e-12 and abs(g["mean"] - 6e-5) < 1e-12 and abs(g["max"] - 9e-5) < 1e-12
    assert g["min"] <= g["mean"] <= g["max"]


def test_mixed_bases_are_counted_and_flagged_never_silently_merged():
    recs = [call("a", "s", 0, green="exact"), call("b", "s", 1, green="fallback"), call("c", "s", 2, green="proxy")]
    g = summary(recs, "s")["green"]
    assert g["basis_counts"] == {"exact": 1, "fallback": 1, "proxy": 1} and g["all_exact"] is False
    only = summary([call("a", "t", 0), call("b", "t", 1)], "t")["green"]
    assert only["all_exact"] is True


def test_calls_without_an_estimate_are_reported_not_counted_as_zero():
    recs = [call("a", "s", 0), call("b", "s", 1, green="none")]
    g = summary(recs, "s")["green"]
    assert g["calls_estimated"] == 1 and g["calls_without_estimate"] == 1


def test_session_with_no_estimates_has_null_energy_not_zero():
    g = summary([call("a", "s", 0, green="none")], "s")["green"]
    assert g["energy_kwh"] is None and g["note"] is None


def test_ttft_p50_ignores_missing_values():
    recs = [call("a", "s", 0, ttft=100), call("b", "s", 1, ttft=None), call("c", "s", 2, ttft=300)]
    assert summary(recs, "s")["ttft_p50_ms"] == 100.0


# ---------------------------------------------------------------- listing
def test_unchained_records_are_not_sessions():
    recs = [call("a", "s", 0), {**call("x", None, None), "session_id": None}]
    items, total = S.list_sessions(recs)
    assert total == 1 and items[0]["session_id"] == "s"


def test_sessions_are_newest_first_and_paged():
    recs = [call("a", "old", 0, t=1), call("b", "new", 0, t=30)]
    items, total = S.list_sessions(recs)
    assert [s["session_id"] for s in items] == ["new", "old"] and total == 2
    assert [s["session_id"] for s in S.list_sessions(recs, limit=1, offset=1)[0]] == ["old"]


def test_steps_turns_and_models_summarised():
    recs = [call("a", "s", 0, turn=0), call("b", "s", 1, turn=0), call("c", "s", 2, turn=1, model="claude-sonnet-5-5")]
    s = summary(recs, "s")
    assert s["steps"] == 3 and s["turns"] == 2 and s["models"] == ["claude-haiku-5-5", "claude-sonnet-5-5"]


def test_failures_and_incomplete_calls_are_counted():
    recs = [call("a", "s", 0), call("b", "s", 1, error={"type": "x"}), call("c", "s", 2, incomplete=True)]
    s = summary(recs, "s")
    assert s["error_calls"] == 1 and s["incomplete_calls"] == 1


# ---------------------------------------------------------------- trajectory
def test_trajectory_orders_steps_and_computes_gap_as_upper_bound():
    recs = [call("b", "s", 1, t=10.0), call("a", "s", 0, t=0.0, dur=2.0)]
    t = S.trajectory("s", recs)
    assert [x["step_index"] for x in t["steps"]] == [0, 1]
    assert t["steps"][0]["gap_since_previous_ms"] is None
    assert t["steps"][1]["gap_since_previous_ms"] == 8000.0          # 10 s - (0 s + 2 s)
    assert "tool_time" not in str(t)                                  # never named tool time


def test_trajectory_carries_tool_call_results_and_missing_results():
    tools = [{"id": "t1", "name": "Read", "args": "{}", "result_present": True, "result_bytes": 5,
              "result_is_error": False, "gap_ms": 40.0},
             {"id": "t2", "name": "Bash", "args": "{}", "result_present": False}]
    t = S.trajectory("s", [call("a", "s", 0, tools=tools)])
    tc = t["steps"][0]["tool_calls"]
    assert tc[0]["result_present"] and tc[0]["gap_ms"] == 40.0
    assert tc[1]["result_present"] is False and tc[1]["result_bytes"] is None      # "no result observed"


def test_unknown_session_gives_none():
    assert S.trajectory("nope", [call("a", "s", 0)]) is None


def test_retries_do_not_become_the_previous_step_for_gap_purposes():
    first = call("a", "s", 0, t=0.0, dur=1.0)
    retry = call("b", "s", 0, t=2.0, dur=1.0)
    retry["outcome"]["retry_index"] = 1
    nxt = call("c", "s", 1, t=10.0)
    t = S.trajectory("s", [first, retry, nxt])
    assert t["steps"][2]["gap_since_previous_ms"] == 9000.0           # measured from the original, t=1 s


# ---------------------------------------------------------------- subagents
def test_attached_child_appears_under_its_parent_with_confidence():
    parent = call("p", "P", 0, tools=[{"id": "t", "name": "Agent", "args": "{}"}])
    edge = {"parent_call_id": "p", "confidence": "high", "signals": ["echo", "overlap"], "score": 0.85}
    child = call("c", "C", 0, parent="p", edge=edge)
    t = S.trajectory("P", [parent, child])
    assert len(t["children"]) == 1 and t["children"][0]["edge"]["confidence"] == "high"
    assert t["children"][0]["session"]["session_id"] == "C"
    assert summary([parent, child], "C")["parent_session_id"] == "P"
    assert t["unattached"] == []


def test_unattached_child_goes_in_its_own_section_never_guessed_into_a_parent():
    parent = call("p", "P", 0, tools=[{"id": "t1", "name": "Agent"}, {"id": "t2", "name": "Read"}])
    orphan = call("o", "O", 0, unattached=True, overlapped=["P"])
    t = S.trajectory("P", [parent, orphan])
    assert t["children"] == [] and [u["session_id"] for u in t["unattached"]] == ["O"]
    assert summary([parent, orphan], "O")["parent_session_id"] is None


def test_unattached_child_is_not_shown_under_unrelated_sessions():
    a = call("a", "A", 0)
    b = call("b", "B", 0)
    orphan = call("o", "O", 0, unattached=True, overlapped=["A"])
    assert S.trajectory("B", [a, b, orphan])["unattached"] == []
    assert len(S.trajectory("A", [a, b, orphan])["unattached"]) == 1


def test_a_session_does_not_list_itself_as_a_child():
    only = call("a", "s", 0)
    t = S.trajectory("s", [only])
    assert t["children"] == [] and t["unattached"] == []


# ---------------------------------------------------------------- end to end through the API
def test_api_endpoints(proxy_factory):
    import json
    import time

    import httpx

    def ev(t, o):
        return f"event: {t}\ndata: {json.dumps(o)}\n\n".encode()
    sse = [ev("message_start", {"type": "message_start", "message": {"model": "claude-haiku-5-5", "usage": {"input_tokens": 9}}}),
           ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
           ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
           ev("message_stop", {"type": "message_stop"})]
    proxy, _, _ = proxy_factory(sse)
    body = {"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "q"}]}
    httpx.post(proxy + "/v1/messages", json=body, headers={"x-api-key": "k"})
    end, d = time.time() + 5, {"sessions": []}
    while time.time() < end and not d["sessions"]:
        d = httpx.get(proxy + "/_trace/api/sessions").json()
        time.sleep(0.1)
    s = d["sessions"][0]
    assert s["steps"] == 1 and s["tokens"]["output"]["value"] == 5 and s["green"]["calls_estimated"] == 1
    t = httpx.get(proxy + f"/_trace/api/sessions/{s['session_id']}/trajectory").json()
    assert len(t["steps"]) == 1 and t["steps"][0]["usage"]["output"]["value"] == 5
    assert httpx.get(proxy + "/_trace/api/sessions/nope/trajectory").status_code == 404
    assert httpx.get(proxy + "/_trace/api/sessions?limit=x").status_code == 400
    cid = t["steps"][0]["call_id"]
    r = httpx.get(proxy + f"/_trace/api/calls/{cid}/content")
    assert r.status_code == 404 and r.json()["stored"] is False
