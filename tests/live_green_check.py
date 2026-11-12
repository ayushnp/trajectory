"""Live check of green metrics: real streamed calls through the proxy, then print the records.

Not part of the pytest suite (needs Tux). Run: python tests/live_green_check.py [proxy_url]
"""
import glob
import json
import sys
import time

import httpx

PROXY = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18081"
PROMPT = "Write two sentences about rivers."
CALLS = [
    ("/v1/messages", {"x-api-key": "x", "anthropic-version": "2023-06-01"},
     {"model": "claude-haiku-5-5", "max_tokens": 120, "stream": True, "messages": [{"role": "user", "content": PROMPT}]}),
    ("/v1/messages", {"x-api-key": "x", "anthropic-version": "2023-06-01"},
     {"model": "claude-sonnet-5-5", "max_tokens": 200, "stream": True, "messages": [{"role": "user", "content": PROMPT}]}),
    ("/v1/chat/completions", {"authorization": "Bearer x"},
     {"model": "gpt-6-luna", "max_tokens": 120, "stream": True, "stream_options": {"include_usage": True},
      "messages": [{"role": "user", "content": PROMPT}]}),
    ("/v1/responses", {"authorization": "Bearer x"},
     {"model": "gpt-6.1-sol", "max_output_tokens": 300, "stream": True, "input": PROMPT}),
]

before = httpx.get(f"{PROXY}/stats").json()["calls_total"]
for path, headers, body in CALLS:
    with httpx.stream("POST", PROXY + path, headers=headers, json=body, timeout=90) as r:
        n = sum(len(c) for c in r.iter_raw())
    print(f"{path:22} {body['model']:18} http {r.status_code}  streamed {n} bytes")
time.sleep(2)

rows = []
for f in sorted(glob.glob("data/calls/*.jsonl")):
    rows += [json.loads(line) for line in open(f, encoding="utf-8")]
rows = [r for r in rows if r["request"]["format"] != "unknown"][-len(CALLS):]


def rng(v):
    return f'{v["min"]:.2e} .. {v["mean"]:.2e} .. {v["max"]:.2e}'


print()
for r in rows:
    g, u = r["green"], r["usage"]
    head = (f'{r["request"]["format"]:17} {str(r["response"]["model_served"]):18} stop={r["response"]["stop_reason"]!s:14} '
            f'chunks={r["timing"]["chunk_count"]:3} out={u["output"]["value"]}')
    if g["status"] != "ok":
        print(head, "GREEN:", g["status"], g.get("reason") or g.get("error"))
        continue
    print(head)
    print(f'{"":17} basis={g["basis"]:5} latency={g["inputs"]["latency_s"]}s  energy kWh {rng(g["energy_kwh"])}   gwp kg {rng(g["gwp_kgco2eq"])}')
print("\nnote    :", rows[0]["green"].get("note"))
print("warnings:", rows[0]["green"].get("warnings"), "| version", rows[0]["green"].get("method_version"))
s = httpx.get(f"{PROXY}/stats").json()
print({k: s[k] for k in ("calls_total", "green_errors", "unknown_green_share", "chain_errors", "overhead_ms_p50", "overhead_ms_p95")})
