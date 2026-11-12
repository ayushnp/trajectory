"""M0 latency check: added latency of the proxy vs. calling a stub upstream directly.

Run: python tests/bench_latency.py
"""
import statistics
import tempfile
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from conftest import Server, create_app, free_port, make_stub  # noqa: E402
from tux_trace.config import Config  # noqa: E402

N = 500
BODY = b'{"model":"x","messages":[]}'


def sample(client, url):
    out = []
    for _ in range(N):
        t = time.perf_counter()
        client.post(url, content=BODY).read()
        out.append((time.perf_counter() - t) * 1000)
    return out


def pct(v, q):
    return sorted(v)[int(q * len(v))]


with Server(make_stub([b"data: x\n\n"]), free_port()) as up, \
        Server(create_app(Config(upstream={"base_url": f"http://127.0.0.1:{up.port}"}, data_dir=tempfile.mkdtemp())), free_port()) as px:
    with httpx.Client(trust_env=False) as c:
        sample(c, f"http://127.0.0.1:{up.port}/v1/messages")  # warm-up
        sample(c, f"http://127.0.0.1:{px.port}/v1/messages")
        direct = sample(c, f"http://127.0.0.1:{up.port}/v1/messages")
        proxied = sample(c, f"http://127.0.0.1:{px.port}/v1/messages")
        stats = c.get(f"http://127.0.0.1:{px.port}/stats").json()

added = [p - d for p, d in zip(sorted(proxied), sorted(direct))]
print(f"direct   p50={pct(direct, .5):.2f}ms  p95={pct(direct, .95):.2f}ms")
print(f"proxied  p50={pct(proxied, .5):.2f}ms  p95={pct(proxied, .95):.2f}ms")
print(f"added    p50={pct(proxied, .5) - pct(direct, .5):.2f}ms  "
      f"p95={pct(proxied, .95) - pct(direct, .95):.2f}ms   (targets: <3 / <10)")
print(f"own handling (/stats): p50={stats['overhead_ms_p50']}ms p95={stats['overhead_ms_p95']}ms")
