"""M6: green metrics (EcoLogits, library mode)."""
import json

import pytest

from tux_trace.green import Estimator

pytest.importorskip("ecologits")


def rec(model="claude-sonnet-5-5", fmt="anthropic", out=1000, inp=5000, cache=4000, total_ms=20_000.0,
        out_reported=True, requested=None):
    return {"request": {"format": fmt, "model_requested": requested or model},
            "response": {"model_served": model},
            "timing": {"total_ms": total_ms, "duration_ms": total_ms},
            "usage": {"output": {"value": out if out_reported else None, "reported": out_reported},
                      "input_total": {"value": inp, "reported": True},
                      "cache_read": {"value": cache, "reported": True}}}


def est(**kw):
    return Estimator(True, "config/model_map.yaml", kw.pop("assumptions", None), **kw)


# ---------------------------------------------------------------- spec test 9 / step 6
def test_known_model_returns_a_valid_range():
    g = est().estimate(rec())
    assert g["status"] == "ok" and g["basis"] == "exact"
    for k in ("energy_kwh", "gwp_kgco2eq"):
        r = g[k]
        assert 0 < r["min"] <= r["mean"] <= r["max"], r
    assert g["method"] == "ecologits" and g["method_version"]
    assert g["assumptions_sha256"] and "estimated, not measured" in g["note"]


def test_unknown_model_falls_back_and_still_returns_a_range():
    g = est().estimate(rec(model="deepseek-v4-pro"))
    assert g["status"] == "ok" and g["basis"] == "fallback"
    assert 0 < g["energy_kwh"]["min"] <= g["energy_kwh"]["max"]
    assert g["inputs"]["model"] == "deepseek-v4-pro"          # we report what was really called
    assert g["inputs"]["ecologits_model"] == "gpt-5.5"        # ...and what stood in for it


def test_hand_mapped_model_is_labelled_proxy():
    g = est().estimate(rec(model="gpt-6.1-terra", fmt="openai_chat"))
    assert g["basis"] == "proxy" and g["inputs"]["ecologits_model"] == "gpt-5.6-terra"


def test_registry_models_are_exact_whichever_provider_format_they_arrive_on():
    assert est().estimate(rec(model="gpt-6-luna", fmt="openai_responses"))["basis"] == "exact"
    assert est().estimate(rec(model="claude-haiku-5-5", fmt="anthropic"))["basis"] == "exact"
    # a Claude model called through the OpenAI-shaped endpoint still resolves
    assert est().estimate(rec(model="claude-sonnet-5-5", fmt="openai_chat"))["basis"] == "exact"


def test_library_failure_leaves_status_error_without_losing_the_record(monkeypatch):
    e = est()
    monkeypatch.setattr(e, "_compute", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    g = e.estimate(rec())
    assert g["status"] == "error" and g["error"] == "RuntimeError" and g["note"]
    assert e.errors == 1


def test_estimator_never_raises_on_garbage_records():
    e = est()
    for bad in ({}, {"request": {}}, {"request": {"format": "anthropic"}, "response": {}, "timing": {}, "usage": None}):
        g = e.estimate(bad)
        assert g["status"] in ("error", "unavailable")


# ---------------------------------------------------------------- honesty requirements
def test_output_tokens_not_reported_means_unavailable_not_zero():
    g = est().estimate(rec(out_reported=False))
    assert g["status"] == "unavailable" and "energy_kwh" not in g


def test_a_range_is_always_present_never_a_bare_number():
    g = est().estimate(rec())
    for k in ("energy_kwh", "gwp_kgco2eq"):
        assert set(g[k]) == {"min", "mean", "max"}


def test_input_and_cache_counts_do_not_change_the_estimate():
    """EcoLogits' energy model has no input-token term, so the spec's all_input vs fresh_only
    variants would be identical. We say so rather than invent a difference."""
    a = est().estimate(rec(inp=100, cache=0))
    b = est().estimate(rec(inp=500_000, cache=499_000))
    assert a["energy_kwh"] == b["energy_kwh"] and a["gwp_kgco2eq"] == b["gwp_kgco2eq"]
    assert "input tokens are not part of" in a["note"]
    assert b["inputs"]["input_tokens"] == 500_000 and b["inputs"]["cache_read_tokens"] == 499_000


def test_more_output_means_more_energy():
    lo = est().estimate(rec(out=100, total_ms=5_000))["energy_kwh"]["mean"]
    hi = est().estimate(rec(out=4000, total_ms=80_000))["energy_kwh"]["mean"]
    assert hi > lo


def test_region_changes_emissions_but_not_energy():
    world = est().estimate(rec())
    fra = est(assumptions={"region": "FRA", "zone": "FRA"}).estimate(rec())
    assert fra["energy_kwh"] == world["energy_kwh"]
    assert fra["gwp_kgco2eq"]["mean"] < world["gwp_kgco2eq"]["mean"]
    assert "world-average" in world["note"] and "world-average" not in fra["note"]


def test_library_warnings_are_passed_through():
    g = est().estimate(rec(model="claude-sonnet-5-5"))
    assert "model-arch-not-released" in g["warnings"]


def test_assumptions_hash_changes_with_assumptions():
    assert est().assumptions_sha256 != est(assumptions={"region": "FRA", "zone": "FRA"}).assumptions_sha256
    assert est().assumptions_sha256 == est().assumptions_sha256


# ---------------------------------------------------------------- caching / switches
def test_identical_inputs_hit_the_cache_and_give_identical_numbers():
    e = est()
    a, b = e.estimate(rec()), e.estimate(rec())
    assert a == b and e.cache_hits == 1


def test_disabled_estimator_does_nothing():
    assert Estimator(False).estimate(rec()) == {"status": "disabled"}


# ---------------------------------------------------------------- end to end through the real writer
def test_end_to_end_through_proxy(proxy_factory):
    import time

    import httpx

    def ev(t, o):
        return f"event: {t}\ndata: {json.dumps(o)}\n\n".encode()
    sse = [ev("message_start", {"type": "message_start", "message": {"model": "claude-haiku-5-5", "usage": {"input_tokens": 9}}}),
           ev("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
           ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 40}}),
           ev("message_stop", {"type": "message_stop"})]
    proxy, _, _ = proxy_factory(sse)
    httpx.post(proxy + "/v1/messages", content=json.dumps({"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "x"}]}))
    end, rows = time.time() + 5, []
    while time.time() < end and not rows:
        rows = httpx.get(proxy + "/_trace/api/calls").json()["calls"]
        time.sleep(0.1)
    full = httpx.get(proxy + f"/_trace/api/calls/{rows[0]['call_id']}").json()
    g = full["green"]
    assert g["status"] == "ok" and g["basis"] == "exact" and g["inputs"]["output_tokens"] == 40
    assert g["energy_kwh"]["min"] <= g["energy_kwh"]["mean"] <= g["energy_kwh"]["max"]


def test_failopen_when_estimator_explodes(proxy_factory, monkeypatch):
    import httpx

    import tux_trace.green as green
    monkeypatch.setattr(green.Estimator, "_estimate", lambda self, rec: 1 / 0)
    proxy, _, _ = proxy_factory([b"data: x\n\n"])
    r = httpx.post(proxy + "/v1/messages", content=b"{}")
    assert r.status_code == 200 and r.content == b"data: x\n\n"


def test_silencing_the_library_logger_does_not_break_it():
    """Regression: touching the 'ecologits' logger before importing the library broke warning_once.
    Must run in a fresh interpreter, because pytest has already imported ecologits."""
    import subprocess
    import sys
    code = "\n".join([
        "from tux_trace.green import Estimator",
        "e = Estimator(True, 'config/model_map.yaml', None)",
        "r = {'request': {'format': 'anthropic', 'model_requested': 'claude-haiku-5-5'},",
        "     'response': {'model_served': 'claude-haiku-5-5'},",
        "     'timing': {'total_ms': 1000.0, 'duration_ms': 1000.0},",
        "     'usage': {'output': {'value': 10, 'reported': True}}}",
        "print(e.estimate(r)['status'])",
    ])
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "ok", out.stderr
    assert "not released" not in out.stderr          # the console stays quiet
