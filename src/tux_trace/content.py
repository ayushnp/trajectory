"""Optional content storage with redaction (spec section 16). Runs on the writer thread.

Content (prompt and response text) is NEVER put in a call record. When `store_content` is on it
goes to its own file, data/content/<call_id>.json, and the record only points at it.

Redaction is applied to the STORED copy only. What the client received, and what Tux received, is
never touched: the proxy is pass-through.

Redaction is best-effort pattern matching. It cannot know a secret it has no pattern for, so the
stored content should still be treated as sensitive.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger("tux_trace.content")
MASK = "[REDACTED]"

# Documented defaults, applied in addition to the user's `redact_patterns`.
DEFAULT_PATTERNS = [
    r"sk-ant-[A-Za-z0-9_\-]{16,}",                       # Anthropic API keys
    r"sk-[A-Za-z0-9_\-]{16,}",                           # OpenAI-style keys
    r"Bearer\s+[A-Za-z0-9._~+/=\-]{16,}",                # bearer tokens
    r"AKIA[0-9A-Z]{16}",                                 # AWS access key ids
    r"gh[pousr]_[A-Za-z0-9]{30,}",                       # GitHub tokens
    r"xox[abprs]-[A-Za-z0-9\-]{10,}",                    # Slack tokens
    r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}",   # JWTs
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    r"(?i)(?:api[_-]?key|secret|password|passwd|token)\"?\s*[:=]\s*\"?[^\s\",'}]{8,}",   # key=value pairs
]


class Redactor:
    def __init__(self, extra_patterns: list[str] | None = None, use_defaults: bool = True) -> None:
        pats = (DEFAULT_PATTERNS if use_defaults else []) + list(extra_patterns or [])
        self._res = [re.compile(p) for p in pats]
        self.hits = 0

    def __call__(self, text: str) -> str:
        for rx in self._res:
            text, n = rx.subn(MASK, text)
            self.hits += n
        return text


class ContentStore:
    def __init__(self, data_dir: str | Path, enabled: bool, redactor: Redactor) -> None:
        self.dir = Path(data_dir) / "content"
        self.enabled = enabled
        self.redact = redactor
        self.stored = 0
        self.errors = 0

    def save(self, call_id: str, request_body: bytes | None, response_text: str | None,
             meta: dict | None = None) -> str | None:
        """Write one content file. Returns its relative path, or None. Never raises."""
        if not self.enabled:
            return None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            req = None
            if request_body:
                text = request_body.decode("utf-8", "replace")
                try:   # keep it as JSON when it parses, so the file is readable
                    req = json.loads(self.redact(text))
                except ValueError:
                    req = self.redact(text)
            doc = {"call_id": call_id, "redacted": True, "request": req,
                   "response": self.redact(response_text) if response_text else None, **(meta or {})}
            path = self.dir / f"{call_id}.json"
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            self.stored += 1
            return f"content/{call_id}.json"
        except Exception:
            self.errors += 1
            log.exception("could not store content for %s", call_id)
            return None


def decode_response(chunks: list[tuple[int, bytes]], encoding: str | None) -> str | None:
    """The response bytes as text, undoing content-encoding for the stored copy only."""
    import zlib
    if not chunks:
        return None
    raw = b"".join(c for _, c in chunks)
    enc = (encoding or "").lower().strip()
    try:
        if enc in ("gzip", "x-gzip", "deflate"):
            raw = zlib.decompressobj(47).decompress(raw)
    except Exception:
        return None
    return raw.decode("utf-8", "replace")
