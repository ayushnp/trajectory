"""Standalone stub upstream for out-of-process benchmarking: python tests/bench_stub.py PORT"""
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).parent))
from conftest import make_stub  # noqa: E402

uvicorn.run(make_stub([b"data: x\n\n"]), host="127.0.0.1", port=int(sys.argv[1]),
            log_level="error", access_log=False, server_header=False, date_header=False)
