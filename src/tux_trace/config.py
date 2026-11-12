"""Load and validate config. YAML file, overridable by TUX_TRACE_<SECTION>__<KEY> env vars."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Listen(_Strict):
    host: str = "127.0.0.1"
    port: int = Field(18081, ge=1, le=65535)


class Upstream(_Strict):
    base_url: str = "http://127.0.0.1:18080"
    connect_timeout_s: float = Field(10, gt=0)
    read_timeout_s: float = Field(600, gt=0)


class Green(_Strict):
    enabled: bool = True
    mode: Literal["library"] = "library"      # the HTTP API mode was deliberately not built
    model_map: str = "config/model_map.yaml"
    assumptions: dict[str, Any] = {"region": "unknown"}


class Queue(_Strict):
    maxsize: int = Field(10_000, ge=1)
    fsync_interval_s: float = Field(5.0, gt=0)
    capture_budget_mb: int = Field(512, ge=1)   # total bytes held for parsing across ALL in-flight and queued calls


class Chain(_Strict):
    candidate_ttl_s: int = Field(1800, ge=1)
    max_steps: int = Field(500, ge=1)


class Config(_Strict):
    listen: Listen = Listen()
    upstream: Upstream = Upstream()
    data_dir: str = "data"
    add_trace_header: bool = True
    sample_rate: float = Field(1.0, ge=0, le=1)
    store_content: bool = True
    retention_days: int = Field(7, ge=1)
    redact_patterns: list[str] = []
    redact_defaults: bool = True            # also apply the built-in key/token patterns (content.py)
    retention_interval_s: float = Field(3600, gt=0)
    sqlite_index: bool = True
    green: Green = Green()
    queue: Queue = Queue()
    chain: Chain = Chain()

    def model_post_init(self, _ctx: Any) -> None:
        for p in self.redact_patterns:
            re.compile(p)  # refuse to start on a bad regex
        if not self.upstream.base_url.startswith(("http://", "https://")):
            raise ValueError("upstream.base_url must start with http:// or https://")


ENV_PREFIX = "TUX_TRACE_"


def _apply_env(data: dict[str, Any], env: dict[str, str]) -> None:
    for key, raw in env.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX):].lower().split("__")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        node = data
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                break
        else:
            node[path[-1]] = value


def load_config(path: str | Path | None = None, env: dict[str, str] | None = None) -> Config:
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    _apply_env(data, dict(os.environ if env is None else env))
    try:
        return Config(**data)
    except (ValidationError, ValueError) as e:
        raise SystemExit(f"invalid config: {e}")


def effective_config_text(cfg: Config) -> str:
    """Printable config. No secrets live in it today; redact defensively anyway."""
    text = json.dumps(cfg.model_dump(), indent=2)
    return re.sub(r'("[^"]*(?:key|token|secret|password)[^"]*":\s*)"[^"]*"', r'\1"***"', text, flags=re.I)
