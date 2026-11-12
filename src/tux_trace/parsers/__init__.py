"""Wire-format detection. By request path, never by sniffing the body (spec section 9)."""
from __future__ import annotations

_FORMATS = {
    "/v1/messages": "anthropic",
    "/v1/chat/completions": "openai_chat",
    "/v1/responses": "openai_responses",
}


def detect_format(path: str) -> str:
    return _FORMATS.get(path.rstrip("/") or "/", "unknown")
