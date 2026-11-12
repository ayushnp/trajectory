"""Live check: a small agent loop against real Tux, THROUGH the proxy.

Not part of the pytest suite (needs Tux). Run: python tests/live_agent_sim.py [proxy_url]
Does two human turns, the first of which needs tool use, so the records should show one
session, steps 0..n, and turn_index 0 then 1.
"""
import json
import sys

import httpx

PROXY = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18081"
MODEL = "claude-haiku-5-5"
HEADERS = {"x-api-key": "does-not-matter", "anthropic-version": "2023-06-01",
           "content-type": "application/json", "user-agent": "claude-cli/live-sim"}
TOOLS = [{"name": "get_weather", "description": "Get the current weather for a city.",
          "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]


def call(messages):
    body = {"model": MODEL, "max_tokens": 300, "stream": True, "tools": TOOLS,
            "system": "You are terse. Use the tool when asked about weather.", "messages": messages}
    blocks, cur = [], None
    with httpx.stream("POST", f"{PROXY}/v1/messages", headers=HEADERS, json=body, timeout=60) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            if ev["type"] == "content_block_start":
                cur = dict(ev["content_block"])
                if cur["type"] == "tool_use":
                    cur["_json"] = ""
                blocks.append(cur)
            elif ev["type"] == "content_block_delta":
                d = ev["delta"]
                if d["type"] == "text_delta":
                    cur["text"] = cur.get("text", "") + d["text"]
                elif d["type"] == "input_json_delta":
                    cur["_json"] += d["partial_json"]
    out = []
    for b in blocks:
        if b["type"] == "tool_use":
            out.append({"type": "tool_use", "id": b["id"], "name": b["name"],
                        "input": json.loads(b.pop("_json") or "{}") or b.get("input", {})})
        elif b["type"] == "text":
            out.append({"type": "text", "text": b.get("text", "")})
    return out


def turn(messages, user_text):
    messages.append({"role": "user", "content": user_text})
    for _ in range(6):
        content = call(messages)
        messages.append({"role": "assistant", "content": content})
        uses = [b for b in content if b["type"] == "tool_use"]
        if not uses:
            print("  assistant:", " ".join(b["text"] for b in content if b["type"] == "text")[:100])
            return
        results = []
        for u in uses:
            print("  tool:", u["name"], u["input"])
            results.append({"type": "tool_result", "tool_use_id": u["id"],
                            "content": f"Sunny, 21C in {u['input'].get('city', '?')}"})
        messages.append({"role": "user", "content": results})


msgs = []
print("turn 1"); turn(msgs, "What's the weather in Paris? Use the tool.")
print("turn 2"); turn(msgs, "And in Tokyo? Use the tool.")
