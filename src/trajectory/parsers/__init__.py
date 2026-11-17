"""Wire-format detection (by request path, never by sniffing the body) and parser factory."""
from __future__ import annotations

from .base import ParseResult, StreamParser

_FORMATS = {
    "/v1/messages": "anthropic",
    "/v1/chat/completions": "openai_chat",
    "/v1/responses": "openai_responses",
}


def detect_format(path: str) -> str:
    return _FORMATS.get(path.rstrip("/") or "/", "unknown")


def make_parser(fmt: str) -> StreamParser | None:
    if fmt == "anthropic":
        from .anthropic import AnthropicParser
        return AnthropicParser()
    if fmt == "openai_chat":
        from .openai_chat import OpenAIChatParser
        return OpenAIChatParser()
    if fmt == "openai_responses":
        from .openai_responses import OpenAIResponsesParser
        return OpenAIResponsesParser()
    return None


__all__ = ["ParseResult", "StreamParser", "detect_format", "make_parser"]
