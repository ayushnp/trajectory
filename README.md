# tux-trace

A small local proxy that sits between a coding agent (Claude Code, opencode, ...) and the Tux model
gateway. It relays every request unchanged, and records one structured trace per model call:
timings, tokens, cache counters, tool calls, sessions/steps/turns, subagents and an energy estimate.

```
agent  ──►  tux-trace :18081  ──►  Tux :18080
                │
                └─ records (JSONL, source of truth) ─► SQLite index ─► dashboard /_trace
```

**It is pass-through.** It never modifies a request or a response, it adds exactly one request header
(`x-tux-trace-id`, switchable), and if anything in the recording path fails the agent still gets its
complete response.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,green]"      # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m tux_trace serve                    # listens on 127.0.0.1:18081
.venv/Scripts/python -m tux_trace smoke                    # first thing to run on a new machine
```

Open the dashboard at **http://127.0.0.1:18081/_trace**.

`smoke` sends one request per wire format through the running proxy and checks that two records
appeared on disk with the right format. It exits non-zero and says why if not.

## Wiring an agent

### Claude Code

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:18081 claude
```

> **Check your settings first.** Values in `~/.claude/settings.json` `env` override the shell
> variable. If that file sets `ANTHROPIC_BASE_URL` to Tux (`http://127.0.0.1:18080`), the line above
> does nothing and your traffic bypasses the proxy silently. Either change that one setting, or run
> with a per-run override that touches nothing global:
>
> ```bash
> echo '{"env":{"ANTHROPIC_BASE_URL":"http://127.0.0.1:18081"}}' > proxy-settings.json
> claude --settings proxy-settings.json
> ```
>
> Confirm it worked: `curl http://127.0.0.1:18081/stats` and watch `calls_total` rise.

### opencode

In `opencode.json` point the provider's `baseURL` at the proxy instead of Tux:
`http://127.0.0.1:18080/v1` becomes `http://127.0.0.1:18081/v1`.

> **Keep the `/v1`.** opencode appends the endpoint name to whatever base URL it is given. Without the
> `/v1` it calls `/responses`, which Tux answers with `404 {"detail":"Not Found"}` (the proxy just
> relays it, and records it as a non-model request). With it, the call is `/v1/responses` and works.
> A provider that uses the Anthropic package may want a base URL without `/v1`, since that package adds
> `/v1/messages` itself; check each provider's own `baseURL` if one of them returns 404.

Confirm it worked the same way as for Claude Code: `calls_total` on `/stats` rises, and the dashboard
shows the session with `harness opencode`.

### Paths

The incoming path is forwarded to Tux verbatim (`/v1/messages` stays `/v1/messages`). The format is
detected from the path, never by sniffing the body:

| Path | Format |
| --- | --- |
| `/v1/messages` | `anthropic` |
| `/v1/chat/completions` | `openai_chat` |
| `/v1/responses` | `openai_responses` (confirmed real on this Tux) |
| anything else | relayed, recorded as `unknown` (transport facts only) |

## What is recorded

One JSON line per call in `data/calls/YYYY-MM-DD.jsonl`, written after the stream ends so a cut-off
call is visible as `incomplete: true` rather than missing.

* **Timing** (monotonic clock): `ttft_any_ms` (first text, thinking or tool call), `ttft_visible_ms`
  (first human-readable text), `total_ms`, chunk count and inter-chunk p50/p95. A mark that never
  happened is `null`, never `0`. Non-streamed responses have no first token, so no TTFT.
* **Tokens**: `input_total`, `output`, `cache_read`, `cache_write`, each `{value, reported}`. A
  provider that does not report a metric gives `reported: false` (shown as `—`), never `0`.
  See *Tokens* below for a deviation from the original spec.
* **Chaining**: `session_id`, `step_index`, `turn_index` from per-message digests. A history that
  shrinks or diverges starts a new session with `compaction_suspected` and `continued_from_session`.
* **Tools**: each call's id, name and arguments; the matching result's size, error flag and `gap_ms`.
* **Subagents**: `parent_call_id` plus an edge `{confidence, signals, score}`; ambiguous cases are
  `unattached`.
* **Green**: energy and emissions as `{min, mean, max}`, with `basis`, `method_version` and caveats.
* **Content** (optional): redacted prompt/response text in `data/content/<call_id>.json`, referenced
  from the record, never inline.

Credentials are never stored: only a SHA-256 of the credential, to tell clients apart.

### Tokens: where Tux differs from the original spec

* Anthropic's `input_tokens` **excludes** cached tokens (measured: 13 fresh + 11,033 cache-read).
  `input_total` is therefore `input + cache_read + cache_write` for Anthropic. The provider's own
  numbers are kept in `usage.raw`. OpenAI's `prompt_tokens` already includes cached tokens.
* Tux reports `cache_write_tokens` on `/v1/responses` (so it is reported there) but not on
  `/v1/chat/completions` (shown as `—`).
* Cumulative counters are never summed: the last value seen wins.

### `gap_ms` is not tool time

`gap_ms` = (next request received) − (previous response ended). It contains the tool, the agent's own
processing, the network and **any human wait** (a permission prompt can make it minutes). It is an
upper bound and is labelled "gap" everywhere. A tool call with no result in any later request shows
"no result observed".

## Dashboard

`/_trace` has two views, chosen with the buttons at the top (or `g t` / `g e`):

**Trajectory** (tracing and observation) has three panes: **Sessions** (one card per session, with a
filter box), **Trajectory** (steps grouped into collapsible **turns**; tool calls nested; subagents under
"attachment is inferred"; orphans under "could not be attached") and **Detail** (the prompt and reply
for a step, then collapsible timing, energy, tool calls and the full record). The sessions and detail
panes can be collapsed. **All calls** is a flat table; health probes and other non-model requests are
hidden unless you tick the box. `j` / `k` step through a session.

**Green metrics** shows energy, emissions and water for every model call in a window (1 h, 24 h, 7 d,
all): headline tiles with their range, coverage (how many calls have an estimate and why the rest do
not), energy by model, energy over time (with a table view), the biggest sessions, what drives the
estimate, and a collapsible "how these numbers are made" with the method, region and library warnings.
The data comes from `GET /_trace/api/green?hours=<n|all>`.

Open or closed state of every collapsible, the chosen view, window and theme are remembered in the
browser, and survive the 3-second refresh. Light and dark themes follow the OS until you use the toggle.
`#session=<id>`, `#session=<id>&step=<call id>`, `#mode=green` and `#theme=light` are deep links.

Rules the UI follows: `—` means not reported (never `0`); energy, emissions and water are always a
range, always marked an estimate, shown in a readable unit (Wh, g, mL) with the exact figures in the
tooltip; a non-exact estimate basis is flagged; a call with no estimate is counted and its reason
shown, never treated as zero; session token totals sum only the calls that reported the metric and
say how many did not.

The dashboard has **no authentication** and is bound to loopback only. It shows tool-call arguments
and (if enabled) stored content, so treat it as sensitive.

## Subagents

A child chain is attached to the **model call** that issued the tool call which started it. Every
helper that one call starts attaches to that same call, however many it starts. The tool call inside
it (`tool_call_id` on the edge) is recorded only when it can be known, and left blank otherwise.

| Evidence | Confidence | Attached? |
| --- | --- | --- |
| prompt echo found in one call's tool arguments, plus overlap | high | yes |
| overlap only, one call open | medium | yes |
| overlap only, one call open, its other tools already claimed by siblings (`elimination`) | medium | yes, tool call narrowed |
| overlap only, tool calls from **different** model calls open at once | none | no, goes to `unattached` |
| echo found in several different calls | none | no, goes to `unattached` |
| echo only (no overlap) | low | no, reported only |

"Overlap" means the child began while the parent's tool call was open. Measured on real Claude Code:
the subagent launches **~90 ms before the parent's response finishes streaming**, so a call counts as
pending from the moment it first appears in the stream. The child's prompt text is used for matching
and then discarded; only the verdict is stored. `unattached_rate` on `/stats` is the best single
indicator that chaining is broken.

**Parallel subagents.** Each tool call is claimed by at most one child where that can be told. Siblings
whose prompts share a long opening are told apart by comparing the child's whole first prompt, not just
its first 200 characters. Identical prompts, or a prompt not found in any argument, still attach to the
parent call; only the tool call stays blank. The last sibling can narrow to the one remaining tool by
elimination (medium, never high).

**Nested subagents.** A subagent's own tool calls are ordinary open calls of its session. When a
grandchild starts, the outer call is already claimed by the subagent, so the subagent's own call is
preferred over it.

**Harness boilerplate.** Claude Code puts `<system-reminder>` blocks before the real instruction in a
helper's first message. They are identical for every helper, so they are skipped when matching.

## Green metrics

Library mode only (EcoLogits, in-process; nothing leaves the machine). The HTTP-API mode in the
original spec was deliberately not built.

* **One estimate per call, not two.** The original spec asked for `all_input` and `fresh_only`
  variants. EcoLogits' energy model has no input-token term (it depends on output tokens, the
  model's active parameters and request latency), so both would be identical. Input and cache counts
  are stored for reference only, and prompt caching does **not** lower the estimate.
* **Model basis**: `exact` = in EcoLogits' registry (12 of Tux's 14 models), `proxy` = hand-mapped in
  `config/model_map.yaml` (`gpt-6.1-terra`), `fallback` = nothing known (`deepseek-v4-pro`, mapped to
  `gpt-5.5`). Non-exact bases are flagged in the UI.
* **Region**: all Tux models are EU-hosted, so the EU electricity mix `EEE` is used. It is an
  average across countries; Sweden is ~0.035 and Poland ~0.589 kgCO₂eq/kWh, a ~17x spread that
  dwarfs the model uncertainty. Set `green.assumptions.zone` to a country code (e.g. `SWE`) if you
  know it. Datacenter PUE/WUE remain EcoLogits' provider defaults (US-based), because there is no
  data on Tux's facilities.
* Every registry model carries "architecture not released, expect lower precision". The
  `{min, mean, max}` range is **not** a confidence interval.

These are estimates, never measurements, and should not be used in external reporting without human
review.

## Configuration

`config/config.yaml`, any key overridable by environment as `TUX_TRACE_<SECTION>__<KEY>` (e.g.
`TUX_TRACE_LISTEN__PORT=19000`, `TUX_TRACE_STORE_CONTENT=false`). Invalid config refuses to start.

| Key | Default | Meaning |
| --- | --- | --- |
| `listen.host` / `listen.port` | `127.0.0.1` / `18081` | where the proxy binds |
| `upstream.base_url` | `http://127.0.0.1:18080` | Tux |
| `upstream.read_timeout_s` | `600` | models can think for minutes |
| `add_trace_header` | `true` | add `x-tux-trace-id` (the only thing ever added) |
| `data_dir` | `data` | JSONL, content, SQLite |
| `sample_rate` | `1.0` | fraction of calls recorded (relaying is unaffected) |
| `store_content` | `true` | write redacted prompt/response text to `data/content/` |
| `redact_patterns` / `redact_defaults` | `[]` / `true` | extra regexes / built-in key and token patterns |
| `retention_days` | `7` | older JSONL, content files and index rows are deleted |
| `sqlite_index` | `true` | maintain `data/index.db` |
| `queue.maxsize` | `10000` | bounded record queue; when full, records are dropped and counted |
| `queue.capture_budget_mb` | `512` | cap on bytes held for parsing across all calls |
| `chain.candidate_ttl_s` / `chain.max_steps` | `1800` / `500` | chain memory bounds |
| `green.enabled` / `green.assumptions` | `true` / EU, zone `EEE` | energy estimates |
| `green.model_map` | `config/model_map.yaml` | hand-mapped and fallback models for estimates |
| `upstream.connect_timeout_s` | `10` | connect timeout to Tux |
| `queue.fsync_interval_s` | `5` | how often the JSONL writer fsyncs |
| `retention_interval_s` | `3600` | how often automatic retention runs |

## Commands

| Command | What it does |
| --- | --- |
| `serve` | run the proxy |
| `smoke` | send a request per wire format through the running proxy and verify the records |
| `reindex` | rebuild `data/index.db` from the JSONL files (the index is derived; JSONL is authoritative) |
| `purge [--days N] [--dry-run]` | delete data older than the retention period now |

Retention also runs automatically at startup and hourly. It deletes only `calls/YYYY-MM-DD.jsonl` and
`content/<uuid>.json` inside the data directory, refuses to follow symlinks or Windows junctions, and
never touches other files.

## Observability

* `GET /healthz`: `status`, `uptime_s`, `upstream_reachable`, `queue_depth`
* `GET /stats`: counters (see below) and `overhead_ms_p50/p95` (the proxy's own handling time)
* `GET /metrics`: the same numbers as Prometheus text
* `GET /_trace/api/{calls,sessions}` and `.../sessions/<id>/trajectory`, `.../calls/<id>[/content]`

Key `/stats` fields: `dropped_records`, `record_errors`, `parse_errors`, `parse_unknown_events`,
`chain_errors`, `green_errors`, `unattached_rate`, `unknown_token_share`, `unknown_green_share`,
`capture_bytes_in_use`, `capture_budget_denied`. Non-zero errors never affect relaying; they mean a
record is missing detail.

## Measured results

All measured on the development machine (Windows 11, Python 3.13, loopback), against a stub upstream
so model time is excluded, with a raw-socket client in a separate process.

**Added latency** (proxy minus direct), full recording pipeline on, 5 runs of 1,000 requests:

| | p50 | p95 | target |
| --- | --- | --- | --- |
| added latency | 1.4 to 1.7 ms | 1.8 to 2.2 ms (one run 3.8 ms) | <3 ms / <10 ms |
| proxy's own handling time (`/stats`) | 0.2 to 0.3 ms | 0.4 to 0.6 ms | n/a |

Turning recording off (`sample_rate: 0`) gives ~0.9 ms added, so recording costs ~0.5 ms. Content files
and SQLite each cost about 0.1 ms, within noise. The numbers move with machine load: the same older
build measured 0.4 ms in one session and 1.3 ms in another, so compare builds interleaved, not
across days.

**Record completeness** on a live set of 20 successful calls across all three formats, including a real
Claude Code session with a subagent: stop reason, model, input and output tokens, session placement,
energy estimate and stored content were present on 20/20; TTFT on 18/20 (the other two were
non-streamed requests, which have no first token by definition). 9 tool calls recorded.
`parse_errors`, `chain_errors`, `green_errors`, `record_errors` and `dropped_records` were all 0;
`unattached_rate` was 0.0 and the real delegation attached at **high** confidence.

**Parallel subagents on real Claude Code traffic**, three runs of three helpers started in one step.
Run 1: none attached (Claude Code's `<system-reminder>` blocks were being compared instead of the
instruction; fixed). Run 2: two attached, the third latched onto a sibling's `Bash` call (fixed).
Run 3, after both fixes: all three attached to the same parent call, `unattached_rate` 0.0. They came
out at **medium** confidence with the tool call blank, not high: the prompt echo did not match in that
run, and the cause is not yet found.

**opencode** (one session, 8 steps, `gpt-6-luna` over `/v1/responses`): all calls recorded as
`openai_responses` with tokens, cache read and write, tool calls (7 issued, 7 results paired) and an
`exact` green estimate; `record_errors`, `chain_errors`, `green_errors` and `dropped_records` all 0.

**Tests**: 232 pass, 1 skipped (creating a symlink needs a privilege this account lacks; the Windows
directory-junction version of the same safety test does run).

## Known gaps and uncertainties

* **opencode** was run end to end once, with one OpenAI model (`gpt-6-luna` over `/v1/responses`): one
  session of 8 steps, tokens, cache read and write, tool calls and results paired (7 of 7), and a green
  estimate on every call. Not run: opencode with a Claude model (the Anthropic package), with
  `/v1/chat/completions`, or with subagents.
* **Subagent attachment** is validated on real traffic for a single subagent from Claude Code only.
  Parallel and nested subagents are covered by synthetic tests, not real traffic. One thing is
  assumed, not measured: that every sibling's request arrives while all of the batch's calls are
  still open. A child that starts after its sibling's result was delivered would lose that call
  from the candidates. Other agent harnesses are untested.
* **A model switch mid-conversation starts a new session**, because chains are scoped to client, wire
  format and model. This avoids merging identical prompts sent to different models.
* **Redaction is pattern-based.** A secret with no matching pattern is stored. `store_content` is on
  by default, so treat `data/content/` as sensitive, or turn it off.
* **`compaction_suspected`** also fires for forks and edited history; a true compaction cannot be told
  apart from those by prefix alone.
* **Tool results from OpenAI formats** have no error flag (`result_is_error` is `null`).
* **Pending tool calls and chain heads live in memory** (bounded, 30-minute expiry). A proxy restart
  mid-conversation orphans in-flight pairing; those results are counted as unmatched, not invented.
* **The dashboard reads the SQLite index** (schema v2 keeps a full copy of each record, so no scan of
  the JSONL and no 20,000-record cap). JSONL is still the source of truth: an index from an older schema
  is dropped and refilled from it at startup, and `tux-trace reindex` rebuilds it on demand. With
  `sqlite_index: false` the dashboard falls back to reading the JSONL into memory (capped at 20,000
  records). It polls every 3 s; untested at tens of thousands of sessions.
* **Dashboard layout** was checked in headless Edge at 1400-1600 px wide, in both themes. Narrow
  screens (under 1100 px stack the panes) and other browsers were not looked at.
* **Retention** is covered by unit tests and a CLI run on back-dated files, not by a multi-day soak.
* **EcoLogits is called through `ecologits.tracers.utils.llm_impacts`**, which is not its headline API.
  The version is recorded on every call, but a library upgrade could break it (tests would catch it).
* **Water factors** use the world default (`electricity-mix-wue-world` warning); treat them as weak.
* **Tux itself**: bad model names return HTTP 200 with a "not available" message and zero tokens
  (Tux's behaviour, identical when called directly), so a typo looks like a successful empty call.
  Questions still worth asking the Tux owners: whether Tux returns a request-id header to join on,
  and whether it ever rewrites a request (which would make our token counts differ from theirs).

## Layout

```
src/tux_trace/
  app.py proxy.py           ingress and the byte-for-byte relay (fast path)
  tee.py store.py           bounded queue and the single writer thread (slow path)
  record.py parsers/        call record, per-format SSE parsers
  usage.py timing.py        token mapping, timing marks
  chain.py tools.py subagents.py   sessions, tool pairing, subagent attachment
  green.py green_view.py    EcoLogits integration (config/model_map.yaml); window aggregation
  content.py retention.py index_db.py   content + redaction, retention, SQLite
  api.py sessions.py conversation.py static/   read API, session views, readable prompt/reply, dashboard
tests/                      232 tests, real Tux streams in tests/fixtures/
```

`tests/bench_oop.py` re-measures overhead. It runs its proxy against a throwaway data directory; an
earlier version wrote into the real `data/` and polluted it, which is why this is called out.
