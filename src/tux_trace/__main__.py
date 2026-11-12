from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import create_app
from .config import effective_config_text, load_config


def main() -> None:
    ap = argparse.ArgumentParser(prog="tux-trace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="run the proxy")
    serve.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    if args.cmd == "serve":
        path = Path(args.config)
        cfg = load_config(path if path.exists() else None)
        print("effective config:\n" + effective_config_text(cfg))
        uvicorn.run(
            create_app(cfg),
            host=cfg.listen.host,
            port=cfg.listen.port,
            access_log=False,
            server_header=False,   # uvicorn must not inject a Server header
            date_header=False,     # ...nor a Date header; relay upstream's exactly
        )


if __name__ == "__main__":
    main()
