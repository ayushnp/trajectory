"""The Green view's aggregation: ranges stay ranges, gaps are counted not zeroed, only model calls count."""
from datetime import datetime, timezone

from trajectory.green_view import aggregate, slim
from trajectory.index_db import SqliteIndex

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)


def g(e, basis="exact", water=True, out=100, warn=()):
    d = {"status": "ok", "basis": basis, "method": "ecologits", "method_version": "0.11.3",
         "inputs": {"output_tokens": out, "input_tokens": 1000, "cache_read_tokens": 900, "region": "EU",
                    "ecologits_model": "m"},
         "energy_kwh": {"min": e * .5, "mean": e, "max": e * 1.5},
         "gwp_kgco2eq": {"min": e * .1, "mean": e * .2, "max": e * .3},
         "warnings": list(warn), "assumptions_sha256": "abc", "note": "estimated, not measured"}
    if water:
        d["water_l"] = {"min": e, "mean": e * 2, "max": e * 3}
    return d


def row(at, green, sid="S1", model="m1"):
    return {"received_at": at, "session_id": sid, "model": model, "harness": "claude-code", "green": green}


def test_ranges_sum_componentwise_and_stay_ranges():
    a = aggregate([row("2026-11-13T10:00:00+00:00", g(2.0)), row("2026-11-13T10:30:00+00:00", g(4.0))], now=NOW)
    assert a["totals"]["energy_kwh"] == {"min": 3.0, "mean": 6.0, "max": 9.0}
    assert a["calls"]["estimated"] == 2 and a["calls"]["without_estimate"] == 0


def test_calls_without_an_estimate_are_counted_with_a_reason_not_zeroed():
    a = aggregate([row("2026-11-13T10:00:00+00:00", g(2.0)),
                   row("2026-11-13T10:01:00+00:00", {"status": "unavailable", "reason": "output tokens not reported"}),
                   row("2026-11-13T10:02:00+00:00", None)], now=NOW)
    assert a["calls"]["without_estimate"] == 2
    assert a["calls"]["without_reasons"] == {"output tokens not reported": 1, "no estimate block": 1}
    assert a["totals"]["energy_kwh"]["mean"] == 2.0
    assert a["per_call"]["energy_kwh_mean"] == 2.0          # averaged over estimated calls only


def test_partial_water_is_reported_as_partial():
    a = aggregate([row("2026-11-13T10:00:00+00:00", g(1.0)), row("2026-11-13T10:01:00+00:00", g(1.0, water=False))], now=NOW)
    assert a["totals"]["water_calls"] == 1 and a["calls"]["estimated"] == 2
    assert a["totals"]["water_l"]["mean"] == 2.0


def test_no_data_gives_nulls_not_zeros():
    a = aggregate([], now=NOW)
    assert a["totals"]["energy_kwh"] is None and a["per_call"]["energy_kwh_mean"] is None
    assert a["timeline"]["points"] == [] and a["by_model"] == []


def test_models_are_ranked_and_non_exact_basis_is_visible():
    a = aggregate([row("2026-11-13T10:00:00+00:00", g(1.0), model="small"),
                   row("2026-11-13T10:01:00+00:00", g(9.0, basis="fallback"), model="big")], now=NOW)
    assert [m["model"] for m in a["by_model"]] == ["big", "small"]
    assert a["by_model"][0]["basis"] == "fallback" and a["basis"] == {"exact": 1, "fallback": 1}


def test_timeline_buckets_hourly_and_marks_empty_hours_as_no_calls_not_zero_energy():
    a = aggregate([row("2026-11-13T09:10:00+00:00", g(1.0)), row("2026-11-13T11:50:00+00:00", g(1.0))], now=NOW)
    t = a["timeline"]
    assert t["bucket"] == "hour" and [p["calls"] for p in t["points"]] == [1, 0, 1]
    assert t["points"][1]["energy_kwh"] is None


def test_long_windows_switch_to_daily_buckets():
    a = aggregate([row("2026-11-01T09:00:00+00:00", g(1.0)), row("2026-11-13T09:00:00+00:00", g(1.0))], now=NOW)
    assert a["timeline"]["bucket"] == "day" and len(a["timeline"]["points"]) == 13


def test_warnings_and_assumptions_are_collected():
    a = aggregate([row("2026-11-13T10:00:00+00:00", g(1.0, warn=("model-arch-not-released",))),
                   row("2026-11-13T10:01:00+00:00", g(1.0, warn=("model-arch-not-released", "x")))], now=NOW)
    assert a["warnings"] == {"model-arch-not-released": 2, "x": 1}
    assert a["assumptions"]["regions"] == ["EU"] and a["assumptions"]["methods"] == ["ecologits 0.11.3"]


def test_slim_extracts_what_is_needed():
    r = slim({"timing": {"received_at": "t"}, "session_id": "s", "response": {"model_served": "m"},
              "request": {"model_requested": "x"}, "client": {"harness_hint": "h"}, "green": {"status": "ok"}})
    assert r == {"received_at": "t", "session_id": "s", "model": "m", "harness": "h", "green": {"status": "ok"}}


def test_sqlite_green_rows_only_returns_model_calls_in_the_window(tmp_path):
    from tests.test_sqlite_reader import call
    idx = SqliteIndex(tmp_path / "i.db")
    probe = call("probe", None, 0, at="2026-11-13T10:00:00+00:00", green=False)
    probe["request"]["format"] = "unknown"
    idx.upsert([call("a", "S", 0, at="2026-11-13T09:00:00+00:00"),
                call("b", "S", 1, at="2026-11-13T11:00:00+00:00"), probe])
    assert len(idx.green_rows(None)) == 2
    assert [r["received_at"] for r in idx.green_rows("2026-11-13T10:00:00+00:00")] == ["2026-11-13T11:00:00+00:00"]
    assert idx.green_rows(None)[0]["green"]["status"] == "ok"
