"""Provider usage -> canonical token fields (spec section 11).

Each from_* function flattens a provider usage object to {input, output, cache_read,
cache_write, reasoning}, containing only the keys the provider actually sent. Callers merge
successive usage objects with dict.update(): last value wins, never summed.

Measured deviation from the spec's table: Anthropic's `input_tokens` EXCLUDES cached tokens
(13 fresh + 11033 cache_read in a real cached call), whereas OpenAI's `prompt_tokens` /
`input_tokens` INCLUDE them. `canonical()` therefore makes input_total inclusive for Anthropic.
The provider's own numbers are preserved in `usage.raw`.
"""
from __future__ import annotations


def _num(v):
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _put(out: dict, key: str, value) -> None:
    v = _num(value)
    if v is not None:
        out[key] = v


def _d(u, key):
    v = u.get(key) if isinstance(u, dict) else None
    return v if isinstance(v, dict) else {}


def from_anthropic(u) -> dict:
    out: dict = {}
    if not isinstance(u, dict):
        return out
    _put(out, "input", u.get("input_tokens"))
    _put(out, "output", u.get("output_tokens"))
    _put(out, "cache_read", u.get("cache_read_input_tokens"))
    _put(out, "cache_write", u.get("cache_creation_input_tokens"))
    _put(out, "reasoning", _d(u, "output_tokens_details").get("thinking_tokens"))
    return out


def from_openai_chat(u) -> dict:
    out: dict = {}
    if not isinstance(u, dict):
        return out
    _put(out, "input", u.get("prompt_tokens"))
    _put(out, "output", u.get("completion_tokens"))
    _put(out, "cache_read", _d(u, "prompt_tokens_details").get("cached_tokens"))
    _put(out, "reasoning", _d(u, "completion_tokens_details").get("reasoning_tokens"))
    return out  # cache_write: not reported by this API


def from_openai_responses(u) -> dict:
    out: dict = {}
    if not isinstance(u, dict):
        return out
    _put(out, "input", u.get("input_tokens"))
    _put(out, "output", u.get("output_tokens"))
    details = _d(u, "input_tokens_details")
    _put(out, "cache_read", details.get("cached_tokens"))
    _put(out, "cache_write", details.get("cache_write_tokens"))  # Tux reports it; may be null
    _put(out, "reasoning", _d(u, "output_tokens_details").get("reasoning_tokens"))
    return out


FROM = {"anthropic": from_anthropic, "openai_chat": from_openai_chat,
        "openai_responses": from_openai_responses}


def _metric(flat: dict, key: str) -> dict:
    v = flat.get(key)
    return {"value": v, "reported": True} if v is not None else {"value": None, "reported": False}


def canonical(fmt: str, flat: dict) -> dict:
    """The four canonical fields plus the provider-native numbers under `raw`."""
    input_total = _metric(flat, "input")
    if fmt == "anthropic" and input_total["reported"]:
        input_total["value"] = flat["input"] + flat.get("cache_read", 0) + flat.get("cache_write", 0)
    return {
        "input_total": input_total,
        "output": _metric(flat, "output"),
        "cache_read": _metric(flat, "cache_read"),
        "cache_write": _metric(flat, "cache_write"),
        "raw": dict(flat),
    }
