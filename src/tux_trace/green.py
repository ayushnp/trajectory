"""Green metrics via EcoLogits, library mode (spec section 15). Runs on the writer thread.

Estimates only. Ranges always; never a bare mean; never called "measured".

What the library actually models (verified against ecologits 0.11.3, not assumed):
  * Energy is a function of OUTPUT tokens, the model's active parameters, and request latency.
    INPUT tokens are not a term in the formula, so the spec's all_input / fresh_only variants
    would be numerically identical. We therefore compute one estimate and say so in `note`;
    input and cache counts are kept in `inputs` for reference only.
  * Region changes GWP but not energy. Tux does not tell us where inference runs, so the
    default is the library's world-average mix and the note says so.
  * Every registry entry we use carries an "architecture not released / multimodal" warning;
    those are passed through, because they mean wider error bars than the range suggests.

Model resolution (basis):
  exact     the Tux id is in the EcoLogits registry (we look under the format's provider first)
  proxy     hand-mapped in model_map.yaml
  fallback  nothing known: a stand-in model is used and labelled as such
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path

import yaml

log = logging.getLogger("tux_trace.green")
NOTE = "estimated, not measured"
PROVIDER_FOR_FORMAT = {"anthropic": "anthropic", "openai_chat": "openai", "openai_responses": "openai"}


def _quiet_ecologits() -> None:
    """The library warns on the console for every call; we record those warnings in the block.

    Order matters: ecologits installs its own Logger subclass (with warning_once) at import time.
    Touching logging.getLogger("ecologits") BEFORE importing it would create a plain Logger under
    that name and break the library. So import first, then adjust the level.
    """
    import ecologits  # noqa: F401  (side effect: registers its logger class)
    logging.getLogger("ecologits").setLevel(logging.ERROR)


def _rng(v) -> dict | None:
    """RangeValue | float -> {min, mean, max}. A range's mean is the midpoint of its bounds."""
    if v is None:
        return None
    if hasattr(v, "min") and hasattr(v, "max"):
        lo, hi = float(v.min), float(v.max)
        return {"min": lo, "mean": (lo + hi) / 2, "max": hi}
    x = float(v)
    return {"min": x, "mean": x, "max": x}


class GreenError(Exception):
    pass


class Estimator:
    def __init__(self, enabled: bool = True, model_map_path: str | Path | None = None,
                 assumptions: dict | None = None, cache_size: int = 4096) -> None:
        self.enabled = enabled
        self.assumptions = dict(assumptions or {"region": "unknown"})
        self.model_map = self._load_map(model_map_path)
        self._cache: OrderedDict[tuple, dict] = OrderedDict()
        self._cache_size = cache_size
        self.version: str | None = None
        self.calls = self.errors = self.cache_hits = 0
        if enabled:
            _quiet_ecologits()
            try:
                import ecologits
                self.version = getattr(ecologits, "__version__", None)
            except Exception:
                self.version = None
        # Same assumptions + same library version => same numbers: hash both so a change is visible.
        blob = json.dumps({"assumptions": self.assumptions, "version": self.version,
                           "map": self.model_map}, sort_keys=True, default=str)
        self.assumptions_sha256 = hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def _load_map(path) -> dict:
        if not path or not Path(path).exists():
            return {}
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    # -- model resolution -----------------------------------------------------------------
    def resolve(self, model: str | None, fmt: str) -> tuple[str, str, str]:
        """-> (provider, ecologits_model, basis)."""
        from ecologits.model_repository import models
        provider = PROVIDER_FOR_FORMAT.get(fmt, "openai")
        if model:
            for p in (provider, "openai", "anthropic"):          # the registry is per-provider
                if models.find_model(provider=p, model_name=model) is not None:
                    return p, model, "exact"
            entry = self.model_map.get(model)
            if entry:
                return entry.get("provider", provider), entry["target"], entry.get("basis", "proxy")
        fb = self.model_map.get("*") or {"provider": "openai", "target": "gpt-5.5", "basis": "fallback"}
        return fb.get("provider", "openai"), fb["target"], "fallback"

    # -- computation ----------------------------------------------------------------------
    def estimate(self, rec: dict) -> dict:
        """Build the `green` block for a record. Never raises; failures become status=error."""
        if not self.enabled:
            return {"status": "disabled"}
        try:
            return self._estimate(rec)
        except Exception as e:                       # fail open: the call is still recorded
            self.errors += 1
            log.warning("green estimate failed: %s", type(e).__name__)
            return {"status": "error", "method": "ecologits", "method_version": self.version,
                    "error": type(e).__name__, "note": NOTE}

    def _estimate(self, rec: dict) -> dict:
        u = rec.get("usage") or {}
        out_tok = (u.get("output") or {})
        if not out_tok.get("reported") or out_tok.get("value") is None:
            return {"status": "unavailable", "method": "ecologits", "method_version": self.version,
                    "reason": "output tokens not reported", "note": NOTE}
        fmt = rec["request"]["format"]
        served = rec["response"].get("model_served") or rec["request"].get("model_requested")
        provider, target, basis = self.resolve(served, fmt)
        out = int(out_tok["value"])
        # Latency caps GPU time in the model. Prefer the measured generation time; fall back to the
        # whole call. A missing latency makes the library assume an unbounded one (higher estimate).
        t = rec["timing"]
        latency_ms = t.get("total_ms") if t.get("total_ms") else t.get("duration_ms")
        latency_s = round(latency_ms / 1000, 3) if latency_ms else None
        zone = self.assumptions.get("zone")                        # ISO-3166 alpha-3, or None = world

        key = (provider, target, out, latency_s, zone)
        res = self._cache.get(key)
        if res is not None:
            self._cache.move_to_end(key)
            self.cache_hits += 1
        else:
            res = self._compute(provider, target, out, latency_s, zone)
            self._cache[key] = res
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        self.calls += 1

        reg = self.assumptions.get("region", "unknown")
        return {
            "status": "ok", "method": "ecologits", "method_version": self.version,
            "basis": basis,
            "inputs": {
                "model": served, "ecologits_model": target if basis != "exact" else served,
                "output_tokens": out,
                "input_tokens": (u.get("input_total") or {}).get("value"),      # reference only
                "cache_read_tokens": (u.get("cache_read") or {}).get("value"),  # reference only
                "latency_s": latency_s, "region": reg,
            },
            "energy_kwh": res["energy_kwh"], "gwp_kgco2eq": res["gwp_kgco2eq"],
            "water_l": res.get("water_l"),
            "warnings": res["warnings"],
            "assumptions_sha256": self.assumptions_sha256,
            "note": self._note(zone),
        }

    def _note(self, zone: str | None) -> str:
        """Say exactly what was assumed. Only the electricity mix is region-specific; the datacenter
        PUE / water factors stay at the provider defaults EcoLogits ships (US-based), because we have
        no data on Tux's facilities."""
        parts = [NOTE, "input tokens are not part of the EcoLogits energy model"]
        if zone is None:
            parts.append("no region set: EcoLogits' default (US) electricity mix assumed")
        else:
            label = self.assumptions.get("region", zone)
            parts.append(f"electricity mix: {zone} ({label}); if {zone} is an average over several countries, "
                         "real emissions depend on which one")
        parts.append("datacenter PUE/WUE are EcoLogits provider defaults, not Tux-specific")
        return "; ".join(parts)

    def _compute(self, provider, target, out, latency_s, zone) -> dict:
        from ecologits.tracers.utils import llm_impacts
        r = llm_impacts(provider, target, out, latency_s if latency_s else float("inf"), zone)
        if r.has_errors:
            raise GreenError(",".join(e.code for e in r.errors))
        return {
            "energy_kwh": _rng(r.energy.value),
            "gwp_kgco2eq": _rng(r.gwp.value),
            "water_l": _rng(r.wcf.value) if r.wcf is not None else None,
            "warnings": [w.code for w in (r.warnings or [])],
        }
