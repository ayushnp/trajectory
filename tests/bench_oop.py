"""M0 latency check with stub and proxy in SEPARATE processes, measured with a raw-socket
client (so the client's own overhead is negligible).

Run: python tests/bench_oop.py [extra env as KEY=VAL ...]
"""
import os
import shutil
import socket
import tempfile
import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
ROOT = HERE.parent
UP, PX = 19080, 19081
BODY = b'{"model":"x"}'
REQ = (b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
       b"Content-Length: %d\r\n\r\n" % len(BODY)) + BODY

# The benchmark proxy must NEVER write into the real ./data directory (live traces live there).
SCRATCH = tempfile.mkdtemp(prefix="tux_bench_")
env = {**os.environ, "TRAJECTORY_UPSTREAM__BASE_URL": f"http://127.0.0.1:{UP}",
       "TRAJECTORY_LISTEN__PORT": str(PX), "TRAJECTORY_DATA_DIR": SCRATCH}
for kv in sys.argv[1:]:
    k, v = kv.split("=", 1)
    env[k] = v


def run(port, n=1000, warm=200):
    s = socket.create_connection(("127.0.0.1", port))
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    out = []
    for i in range(warm + n):
        t = time.perf_counter()
        s.sendall(REQ)
        buf = b""
        while not buf.endswith(b"0\r\n\r\n") and not (b"content-length: 0" in buf.lower() and buf.endswith(b"\r\n\r\n")):
            d = s.recv(65536)
            if not d:
                break
            buf += d
        if i >= warm:
            out.append((time.perf_counter() - t) * 1000)
    s.close()
    return sorted(out)


procs = [
    subprocess.Popen([sys.executable, str(HERE / "bench_stub.py"), str(UP)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
    subprocess.Popen([sys.executable, "-m", "trajectory", "serve"], cwd=ROOT, env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
]
try:
    time.sleep(4)
    d = run(UP)
    p = run(PX)
    q = lambda v, x: v[int(x * len(v))]  # noqa: E731
    print(f"direct  p50={q(d, .5):.2f}ms p95={q(d, .95):.2f}ms")
    print(f"proxied p50={q(p, .5):.2f}ms p95={q(p, .95):.2f}ms")
    print(f"added   p50={q(p, .5) - q(d, .5):.2f}ms p95={q(p, .95) - q(d, .95):.2f}ms  (targets <3 / <10)")
finally:
    for pr in procs:
        pr.kill()
    time.sleep(0.3)
    shutil.rmtree(SCRATCH, ignore_errors=True)
