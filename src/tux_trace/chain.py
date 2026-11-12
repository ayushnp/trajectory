"""Session / step / turn inference (spec section 12).

Runs on the single writer thread, so it needs no locking and never touches the request path.

Per-message digests identify a conversation prefix. A new call continues a session when a
recent call from the same client has digests that are a prefix of the new call's digests.

Thinking / reasoning blocks are left out of the digests: clients may drop them from history
between calls, and including them would make a normal continuation look like a divergence.
Their presence is recorded (`opaque_reasoning`) so nobody assumes equal digests mean equal
meaning for those messages.

Digests are truncated to 16 hex chars (64 bits): plenty to tell prefixes apart, and it keeps
a 300-message conversation from adding ~19 KB to every record.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass

DIGEST_LEN = 16
SEP = "\x1f"


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:DIGEST_LEN]


def _norm(v) -> str:
    """Stable string for tool arguments, whether they arrive as a JSON string or an object."""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return v
    return json.dumps(v, sort_keys=True, separators=(",", ":"), default=str)


# ---------------------------------------------------------------- per-format digests
def _anthropic(messages: list) -> tuple[list[str], list[int], bool]:
    digests, human, opaque = [], [], False
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            digests.append(_h(str(m)))
            continue
        role, content = str(m.get("role") or ""), m.get("content")
        parts: list[str] = []
        has_text = has_result = False
        if isinstance(content, str):
            parts.append(content)
            has_text = bool(content.strip())
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    parts.append(str(b))
                    continue
                t = b.get("type")
                if t == "text":
                    x = b.get("text") or ""
                    parts.append(x)
                    has_text = has_text or bool(x.strip())
                elif t == "tool_use":
                    parts.append(f"tool_use:{b.get('id')}:{b.get('name')}:{_norm(b.get('input'))}")
                elif t == "tool_result":
                    parts.append(f"tool_result:{b.get('tool_use_id')}")
                    has_result = True
                elif t in ("thinking", "redacted_thinking"):
                    opaque = True
                else:  # images, documents, ...: type plus content, minus cache markers
                    parts.append(f"{t}:{_norm({k: v for k, v in b.items() if k != 'cache_control'})}")
        digests.append(_h(role + "\x00" + SEP.join(parts)))
        # Anthropic tool results are role=user too: a human turn has text and no tool_result.
        if role == "user" and has_text and not has_result:
            human.append(i)
    return digests, human, opaque


def _chat(messages: list) -> tuple[list[str], list[int], bool]:
    digests, human = [], []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            digests.append(_h(str(m)))
            continue
        role, content = str(m.get("role") or ""), m.get("content")
        parts: list[str] = []
        has_text = False
        if isinstance(content, str):
            parts.append(content)
            has_text = bool(content.strip())
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    x = p.get("text") or ""
                    parts.append(x)
                    has_text = has_text or bool(x.strip())
                else:
                    parts.append(_norm(p))
        for tc in m.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            parts.append(f"tool_call:{tc.get('id')}:{fn.get('name')}:{_norm(fn.get('arguments'))}")
        if role == "tool":
            parts.append(f"tool_result:{m.get('tool_call_id')}")
        digests.append(_h(role + "\x00" + SEP.join(parts)))
        if role == "user" and has_text:
            human.append(i)
    return digests, human, False


def _responses(inp) -> tuple[list[str], list[int], bool]:
    if isinstance(inp, str):
        inp = [{"role": "user", "content": inp}]
    digests, human, opaque = [], [], False
    for item in inp if isinstance(inp, list) else []:
        if not isinstance(item, dict):
            digests.append(_h(str(item)))
            continue
        t = item.get("type")
        if t == "reasoning":
            opaque = True            # encrypted: never decoded, and not part of the digest
            continue
        role, parts, has_text = "", [], False
        if t in (None, "message"):
            role, content = str(item.get("role") or ""), item.get("content")
            if isinstance(content, str):
                parts.append(content)
                has_text = bool(content.strip())
            elif isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") in ("input_text", "output_text", "text"):
                        x = p.get("text") or ""
                        parts.append(x)
                        has_text = has_text or bool(x.strip())
                    else:
                        parts.append(_norm(p))
        elif t in ("function_call", "custom_tool_call"):
            role = "assistant"
            parts.append(f"tool_call:{item.get('call_id')}:{item.get('name')}:"
                         f"{_norm(item.get('arguments', item.get('input')))}")
        elif t in ("function_call_output", "custom_tool_call_output"):
            role = "tool"
            parts.append(f"tool_result:{item.get('call_id')}")
        else:
            parts.append(f"{t}:{_norm(item)}")
        digests.append(_h(role + "\x00" + SEP.join(parts)))
        if role == "user" and has_text:
            human.append(len(digests) - 1)
    return digests, human, opaque


def digest_request(fmt: str, body: dict) -> tuple[list[str], list[int], bool] | None:
    """-> (per-message digests, indexes of human messages, saw_opaque_reasoning)."""
    if fmt == "anthropic" and isinstance(body.get("messages"), list):
        return _anthropic(body["messages"])
    if fmt == "openai_chat" and isinstance(body.get("messages"), list):
        return _chat(body["messages"])
    if fmt == "openai_responses" and isinstance(body.get("input"), (list, str)):
        return _responses(body["input"])
    return None


# ---------------------------------------------------------------- linking
@dataclass(slots=True)
class _Head:
    session_id: str
    digests: list[str]
    step_index: int
    turn_index: int
    t_last: int
    retry_index: int = 0


def _common_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Chainer:
    """One head (the latest call) per live session, per client."""

    def __init__(self, ttl_s: float = 1800, max_steps: int = 500, max_heads_per_client: int = 256) -> None:
        self.ttl_ns = int(ttl_s * 1e9)
        self.max_steps = max_steps
        self.max_heads = max_heads_per_client
        self._heads: dict[str, list[_Head]] = {}
        self.sessions_created = 0

    def assign(self, rec: dict) -> None:
        """Fill session_id / step_index / turn_index (and compaction fields) on a record dict."""
        req = rec["request"]
        digests = req.get("prefix_digests") or []
        if not digests:
            return                       # nothing to chain on (unknown path, unparsable body)
        # A chain belongs to one client, one wire format and one model. Identical text sent to a
        # different model (or endpoint) is a separate call, not a retry or a continuation. A model
        # switch mid-conversation therefore starts a new session; merging unrelated calls would be worse.
        key = "|".join([rec["client"].get("key_hash") or "-", req.get("format") or "-",
                        req.get("model_requested") or "-"])
        now = rec["timing"]["t_received"]
        status = rec["response"].get("status")
        ok = status is not None and status < 400   # failed calls get a place but never become heads
        human = req.get("human_indexes") or []
        heads = self._live(key, now)

        full = [h for h in heads if len(h.digests) <= len(digests)
                and digests[:len(h.digests)] == h.digests]
        if full:
            cand = max(full, key=lambda h: (len(h.digests), h.t_last))
            n = len(cand.digests)
            if n == len(digests):                                   # identical request: a retry
                self._set(rec, cand.session_id, cand.step_index, cand.turn_index)
                rec["outcome"]["retry_index"] = cand.retry_index + 1
                if ok:
                    cand.retry_index += 1
                    cand.t_last = now
                return
            if cand.step_index + 1 >= self.max_steps:               # bound growth; keep the link
                self._new_session(rec, key, digests, now, ok, continued_from=cand.session_id)
                return
            step = cand.step_index + 1
            turn = cand.turn_index + (1 if any(i >= n for i in human) else 0)
            self._set(rec, cand.session_id, step, turn)
            if ok:
                cand.digests, cand.step_index, cand.turn_index = digests, step, turn
                cand.t_last, cand.retry_index = now, 0
            return

        # No head is a prefix. A head that shares a start but then shrinks or diverges means the
        # history was rewritten (compaction, rewind, edit): new session, linked to the old one.
        best, best_l = None, 0
        for h in heads:
            n = _common_prefix(h.digests, digests)
            if n > best_l or (n == best_l and best is not None and n > 0 and h.t_last > best.t_last):
                best, best_l = h, n
        if best is not None and best_l >= 1:
            self._new_session(rec, key, digests, now, ok, continued_from=best.session_id, compaction=True)
        else:
            self._new_session(rec, key, digests, now, ok)

    # -- helpers -----------------------------------------------------------------------------
    @staticmethod
    def _set(rec: dict, session_id: str, step: int, turn: int) -> None:
        rec["session_id"], rec["step_index"], rec["turn_index"] = session_id, step, turn

    def _new_session(self, rec, key, digests, now, ok, continued_from=None, compaction=False) -> None:
        sid = str(uuid.uuid4())
        self.sessions_created += 1
        self._set(rec, sid, 0, 0)
        if continued_from:
            rec["continued_from_session"] = continued_from
        if compaction:
            rec["compaction_suspected"] = True
        if ok:
            heads = self._heads.setdefault(key, [])
            heads.append(_Head(sid, digests, 0, 0, now))
            if len(heads) > self.max_heads:
                heads.remove(min(heads, key=lambda h: h.t_last))

    def _live(self, key: str, now: int) -> list[_Head]:
        heads = [h for h in self._heads.get(key, []) if abs(now - h.t_last) <= self.ttl_ns]
        if heads:
            self._heads[key] = heads
        else:
            self._heads.pop(key, None)
        return heads
