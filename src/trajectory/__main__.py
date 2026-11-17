from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import uvicorn

from .app import create_app
from .config import effective_config_text, load_config


def _cfg(args):
    path = Path(args.config)
    return load_config(path if path.exists() else None)


def cmd_serve(args) -> int:
    cfg = _cfg(args)
    print("effective config:\n" + effective_config_text(cfg))
    uvicorn.run(
        create_app(cfg),
        host=cfg.listen.host,
        port=cfg.listen.port,
        access_log=False,
        server_header=False,   # uvicorn must not inject a Server header
        date_header=False,     # ...nor a Date header; relay upstream's exactly
    )
    return 0


def cmd_reindex(args) -> int:
    from .index_db import SqliteIndex
    cfg = _cfg(args)
    idx, n = SqliteIndex.rebuild(cfg.data_dir)
    print(f"reindexed {n} records into {Path(cfg.data_dir) / 'index.db'}")
    idx.close()
    return 0


def cmd_purge(args) -> int:
    from .index_db import SqliteIndex
    from .retention import purge
    cfg = _cfg(args)
    days = args.days or cfg.retention_days
    idx = SqliteIndex(Path(cfg.data_dir) / "index.db") if cfg.sqlite_index else None
    out = purge(cfg.data_dir, days, idx, dry_run=args.dry_run)
    print(("DRY RUN, nothing deleted: " if args.dry_run else "") + json.dumps(out))
    return 0


def cmd_smoke(args) -> int:
    """Send one request per wire format through a RUNNING proxy and check that it recorded them."""
    import glob

    import httpx
    cfg = _cfg(args)
    base = args.url or f"http://{cfg.listen.host}:{cfg.listen.port}"
    probes = [
        ("anthropic", "/v1/messages", {"x-api-key": "smoke", "anthropic-version": "2023-06-01"},
         {"model": args.anthropic_model, "max_tokens": 16, "messages": [{"role": "user", "content": "Say ok."}]}),
        ("openai_chat", "/v1/chat/completions", {"authorization": "Bearer smoke"},
         {"model": args.openai_model, "max_tokens": 16, "messages": [{"role": "user", "content": "Say ok."}]}),
    ]
    problems: list[str] = []
    try:
        h = httpx.get(base + "/healthz", timeout=5).json()
    except Exception as e:
        print(f"FAIL  cannot reach the proxy at {base}: {type(e).__name__}. Is `trajectory serve` running?")
        return 2
    print(f"proxy  {base}  status={h['status']}  upstream_reachable={h['upstream_reachable']}")
    if not h["upstream_reachable"]:
        problems.append("proxy cannot reach Tux")

    before = httpx.get(base + "/stats", timeout=5).json()["records_written"]
    sent = []
    for fmt, path, headers, body in probes:
        try:
            r = httpx.post(base + path, headers=headers, json=body, timeout=60)
            print(f"sent   {fmt:12} {path:22} -> HTTP {r.status_code} ({len(r.content)} bytes)")
            sent.append(fmt)
        except Exception as e:
            problems.append(f"{fmt}: request failed ({type(e).__name__})")
    # records are written off the request path, so give the writer a moment
    deadline = time.time() + 10
    while time.time() < deadline:
        if httpx.get(base + "/stats", timeout=5).json()["records_written"] >= before + len(sent):
            break
        time.sleep(0.2)

    calls = []
    for f in sorted(glob.glob(str(Path(cfg.data_dir) / "calls" / "*.jsonl"))):
        calls += [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()]
    mine = [c for c in calls if (c["client"].get("key_hash") and c["request"]["format"] in sent)][-len(sent):] if sent else []
    got = {c["request"]["format"] for c in mine}
    for fmt in sent:
        if fmt not in got:
            problems.append(f"no record on disk for format {fmt}")
    for c in mine:
        u, t = c["usage"], c["timing"]
        print(f"record {c['request']['format']:12} status={c['response']['status']} model={c['response'].get('model_served')} "
              f"in={u['input_total']['value']} out={u['output']['value']} ttft_ms={t['ttft_any_ms']} "
              f"green={c['green'].get('status')} session={(c.get('session_id') or '-')[:8]}")
    print("\nFAIL" if problems else "\nOK", *(f"\n  - {p}" for p in problems))
    return 1 if problems else 0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="trajectory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in (("serve", cmd_serve, "run the proxy"),
                            ("reindex", cmd_reindex, "rebuild the SQLite index from the JSONL files"),
                            ("purge", cmd_purge, "delete data older than retention_days now"),
                            ("smoke", cmd_smoke, "send a request per wire format through a running proxy and verify it was recorded")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", default="config/config.yaml")
        p.set_defaults(fn=fn)
        if name == "purge":
            p.add_argument("--days", type=int, help="override retention_days")
            p.add_argument("--dry-run", action="store_true", help="show what would be deleted")
        if name == "smoke":
            p.add_argument("--url", help="proxy base URL (default: from config)")
            p.add_argument("--anthropic-model", default="claude-haiku-5-5")
            p.add_argument("--openai-model", default="gpt-6-luna")
    args = ap.parse_args(argv)
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
