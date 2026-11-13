"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const API = "/_trace/api";
const state = { mode: "trajectory", view: "sessions", session: null, step: null, showUnknown: false,
                hours: "all", filter: "", steps: [], greenSeen: false };

// ---- persisted UI state. Collapsibles are rebuilt every 3 s by the refresh, so what is open lives
// here, not in the DOM, and is re-applied after each render. ----
const store = {
  get(k, d) { try { const v = localStorage.getItem("tt-" + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("tt-" + k, JSON.stringify(v)); } catch (e) { /* private mode: state just resets */ } },
};
const openSet = new Set(store.get("open", []));
const closedSet = new Set(store.get("closed", []));   // blocks that default to open but the user closed
function isOpen(id, dflt) { return openSet.has(id) || (dflt && !closedSet.has(id)); }
document.addEventListener("toggle", (e) => {
  const d = e.target;
  if (!(d instanceof HTMLDetailsElement) || !d.dataset.id) return;
  const id = d.dataset.id;
  if (d.open) { openSet.add(id); closedSet.delete(id); } else { openSet.delete(id); closedSet.add(id); }
  if (openSet.size > 400) openSet.delete(openSet.values().next().value);
  if (closedSet.size > 400) closedSet.delete(closedSet.values().next().value);
  store.set("open", [...openSet]); store.set("closed", [...closedSet]);
}, true);
// <details> with a stable id and a default; the id is what survives re-rendering
function blk(id, title, body, o = {}) {
  return `<details class="blk ${o.cls || ""}" data-id="${esc(id)}"${isOpen(id, o.open) ? " open" : ""}>
    <summary><span>${title}</span>${o.side ? `<span class="sp">${o.side}</span>` : ""}</summary><div class="body">${body}</div></details>`;
}

// ---- formatting. "not reported" is an em dash, never 0. Everything from the wire goes through esc(). ----
const NA = '<span class="na">&mdash;</span>';
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const ms = (v) => (v == null ? NA : v >= 1000 ? (v / 1000).toFixed(2) + " s" : Math.round(v) + " ms");
const num = (v) => (v == null ? NA : Number(v).toLocaleString());
const metric = (m) => (m && m.reported && m.value != null ? num(m.value) : NA);
const clock = (iso) => new Date(iso).toLocaleTimeString([], { hour12: false });
const short = (id) => (id ? String(id).slice(0, 8) : "");
const sci = (v) => {
  if (v == null) return "—";
  if (v === 0) return "0";
  const e = Math.floor(Math.log10(Math.abs(v)));
  return (v / Math.pow(10, e)).toFixed(1) + "e" + e;
};
// readable magnitude for the green view: 0.2009 kWh -> "200.9 Wh". Values only; the unit is returned apart.
function human(v, base) {
  if (v == null) return { n: "—", u: base };
  const steps = { kWh: [[1, "kWh"], [1e-3, "Wh"], [1e-6, "mWh"]], kg: [[1, "kg"], [1e-3, "g"], [1e-6, "mg"]], L: [[1, "L"], [1e-3, "mL"], [1e-6, "µL"]] }[base];
  const a = Math.abs(v);
  const [f, u] = steps.find(([f]) => a >= f) || steps[steps.length - 1];
  const x = v / f;
  return { n: x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2), u };
}
const SCALE = { kWh: 1, Wh: 1e3, mWh: 1e6, kg: 1, g: 1e3, mg: 1e6, L: 1, mL: 1e3, "µL": 1e6 };
// a min-max range shown in the unit that suits its mean, so both ends are comparable at a glance
const hRange = (r, base) => {
  if (!r) return NA;
  const m = SCALE[human(r.mean, base).u];
  const fmt = (x) => { const y = x * m; return y >= 100 ? y.toFixed(0) : y >= 10 ? y.toFixed(1) : y.toFixed(2); };
  return `${fmt(r.min)}&ndash;${fmt(r.max)} ${human(r.mean, base).u}`;
};
function total(m) {
  if (!m || !m.reported) return NA;
  const miss = m.calls_missing ? ` <span class="tag" title="${m.calls_missing} call(s) did not report this">+${m.calls_missing} unreported</span>` : "";
  return num(m.value) + miss;
}
const ago = (iso) => {
  const s = Math.max(0, (Date.now() - new Date(iso)) / 1000);
  return s < 90 ? Math.round(s) + " s ago" : s < 5400 ? Math.round(s / 60) + " min ago" : s < 129600 ? Math.round(s / 3600) + " h ago" : Math.round(s / 86400) + " d ago";
};

// ---- green: always a range, always an estimate, non-exact basis is flagged ----
// shown in a readable unit (Wh, g, mL); the exact kWh / kg / L figures stay in the tooltip
const UNIT = { "kWh": "kWh", "kgCO2eq": "kg", "kgCO2": "kg", "L": "L" };
function range(r, unit) {
  if (!r) return NA;
  const base = UNIT[unit];
  if (!base) return `~${sci(r.min)}&ndash;${sci(r.max)}${unit ? " " + unit : ""}`;
  const co2 = base === "kg" ? " CO₂e" : "";
  return `<span title="min ${sci(r.min)} / mean ${sci(r.mean)} / max ${sci(r.max)} ${base}">~${hRange(r, base)}${co2}</span>`;
}
function basisTag(b) {
  return !b || b === "exact" ? "" : ` <span class="tag warn" title="model is not in the EcoLogits registry (${esc(b)}); weaker estimate">${esc(b)}</span>`;
}
function greenCell(g) {
  if (!g || g.status === "disabled") return NA;
  if (g.status !== "ok") return `<span class="tag" title="${esc(g.reason || g.error || "")}">${esc(g.status)}</span>`;
  const r = g.energy_kwh;
  return `<span title="Estimated, not measured. Range, not a confidence interval.">${range(r, "kWh")}</span>${basisTag(g.basis)}`;
}
function greenTotal(g) {
  if (!g || !g.calls_estimated) return NA;
  const mixed = Object.entries(g.basis_counts || {}).filter(([k]) => k !== "exact").map(([k, n]) => `${n} ${esc(k)}`).join(", ");
  const warn = mixed ? ` <span class="tag warn" title="includes estimates that are not an exact model match">${mixed}</span>` : "";
  const miss = g.calls_without_estimate ? ` <span class="tag" title="no estimate for these calls">${g.calls_without_estimate} without</span>` : "";
  return `<span title="Estimated, not measured. Sum of the calls' ranges.">${range(g.energy_kwh, "kWh")}</span>${warn}${miss}`;
}
function flags(c) {
  const out = [];
  if (c.error) out.push(`<span class="tag bad" title="${esc(JSON.stringify(c.error))}">error</span>`);
  if (c.client_disconnected || c.disconnected) out.push('<span class="tag warn">client left</span>');
  else if (c.incomplete) out.push('<span class="tag warn">incomplete</span>');
  if (c.retry_index) out.push(`<span class="tag">retry ${c.retry_index}</span>`);
  if (c.compaction_suspected) out.push('<span class="tag warn" title="history shrank or diverged: compaction, rewind or fork">history rewritten?</span>');
  if (c.thinking && c.thinking !== "none") out.push(`<span class="tag" title="thinking visibility">think: ${esc(c.thinking)}</span>`);
  return out.join(" ");
}
const confTag = (e) => {
  if (!e) return "";
  const cls = { high: "hi", medium: "med", low: "low" }[e.confidence] || "low";
  return `<span class="tag ${cls}" title="inferred, not observed. signals: ${esc((e.signals || []).join(", "))}; score ${esc(e.score)}">${esc(e.confidence)} confidence</span>`;
};
async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(url + " " + r.status);
  return r.json();
}

// ================================================================== sessions list
function sessionCard(s) {
  const t = s.tokens;
  const rel = s.parent_session_id
    ? ` <span class="tag" title="subagent of session ${esc(short(s.parent_session_id))}">subagent</span> ${confTag(s.subagent_edge)}`
    : s.unattached ? ' <span class="tag warn" title="started during another session&#39;s tool call but could not be attached">unattached</span>' : "";
  const prob = [s.error_calls ? `<span class="tag bad">${s.error_calls} error</span>` : "",
    s.incomplete_calls ? `<span class="tag warn">${s.incomplete_calls} incomplete</span>` : "",
    s.compaction_suspected ? '<span class="tag warn" title="history rewritten">compaction?</span>' : ""].filter(Boolean).join(" ");
  return `<div class="card ${s.session_id === state.session ? "sel" : ""}" data-sid="${esc(s.session_id)}" tabindex="0">
    <div class="top"><span class="ids">${esc(short(s.session_id))}</span><span class="muted" title="${esc(s.last_seen)}">${clock(s.last_seen)} &middot; ${ago(s.last_seen)}</span></div>
    <div>${esc(s.models.join(", "))} &middot; ${esc(s.harness)}${rel}</div>
    <div class="meta">${s.steps} step${s.steps === 1 ? "" : "s"} &middot; ${s.turns} turn${s.turns === 1 ? "" : "s"}
      &middot; in ${num(t.input_total.value)} &middot; out ${num(t.output.value)} &middot; TTFT p50 ${ms(s.ttft_p50_ms)}</div>
    <div class="meta">energy ${greenTotal(s.green)}</div>
    ${prob ? `<div class="meta">${prob}</div>` : ""}
  </div>`;
}
let lastSessions = [];
function paintSessions() {
  const q = state.filter.trim().toLowerCase();
  const list = !q ? lastSessions : lastSessions.filter((s) =>
    [s.session_id, s.harness, ...(s.models || [])].join(" ").toLowerCase().includes(q));
  $("#sessions").innerHTML = list.map(sessionCard).join("");
  $("#sempty").hidden = lastSessions.length > 0;
  $("#sempty").textContent = "No sessions yet. Point an agent at this proxy and send something.";
  if (lastSessions.length && !list.length) { $("#sempty").hidden = false; $("#sempty").textContent = "No session matches the filter."; }
}
async function refreshSessions() {
  const d = await getJSON(`${API}/sessions?limit=100`);
  $("#scount").textContent = `(${d.total})`;
  lastSessions = d.sessions;
  paintSessions();
}

// ================================================================== trajectory
function toolLine(t) {
  const res = !t.result_present ? '<span class="tag warn" title="no tool result was seen in a later request">no result observed</span>'
    : `${t.result_is_error ? '<span class="tag bad">error</span> ' : ""}${num(t.result_bytes)} B${t.gap_ms != null ? ` &middot; <span title="upper bound on tool time">gap ${ms(t.gap_ms)}</span>` : ""}`;
  return `<li><b>${esc(t.name || "?")}</b> &rarr; ${res}</li>`;
}
function stepRow(st, opts = {}) {
  const sel = st.call_id === state.step ? "sel" : "";
  const cls = [sel, st.error ? "failed" : st.incomplete ? "incomplete" : "", opts.child ? "child" : ""].join(" ");
  const tools = st.tool_calls.length ? `<ul class="tools">${st.tool_calls.map(toolLine).join("")}</ul>` : "";
  const gap = st.gap_since_previous_ms == null ? NA
    : `<span title="upper bound: tool + agent + network + any human wait">gap ${ms(st.gap_since_previous_ms)}</span>`;
  return `<tr data-cid="${esc(st.call_id)}" class="${cls}">
    <td>${opts.child ? "&#8627; " : ""}${st.turn_index == null ? NA : st.turn_index}.${st.step_index == null ? NA : st.step_index}</td>
    <td>${esc(st.model || "?")}${tools}</td>
    <td class="num">${ms(st.ttft_any_ms)}</td>
    <td class="num">${metric(st.usage.input_total)}</td><td class="num">${metric(st.usage.output)}</td>
    <td class="num" title="cache read / cache write">${metric(st.usage.cache_read)} <span class="muted">/ ${metric(st.usage.cache_write)}</span></td>
    <td>${st.stop_reason ? esc(st.stop_reason) : NA}</td><td class="num">${gap}</td>
    <td class="num">${greenCell(st.green)}</td><td>${flags(st)}</td></tr>`;
}
const STEP_HEAD = `<table class="steps"><thead><tr>
  <th title="turn.step">#</th><th>Model / tools</th><th class="num">TTFT</th><th class="num">In</th><th class="num">Out</th>
  <th class="num" title="cache read / cache write">Cache&nbsp;r&nbsp;/&nbsp;w</th><th>Stop</th><th class="num">Gap</th>
  <th class="num">Energy&nbsp;est.</th><th class="wrap"></th></tr></thead><tbody>`;

function summaryBlock(s) {
  const t = s.tokens;
  const cell = (k, v) => `<div><div class="k">${k}</div><div class="v">${v}</div></div>`;
  return `<div class="sum">${cell("steps", s.steps)}${cell("turns", s.turns)}${cell("input", total(t.input_total))}
    ${cell("output", total(t.output))}${cell("cache read", total(t.cache_read))}${cell("cache write", total(t.cache_write))}
    ${cell("TTFT p50", ms(s.ttft_p50_ms))}${cell("energy (est.)", greenTotal(s.green))}
    ${cell("emissions (est.)", s.green.gwp_kgco2eq ? range(s.green.gwp_kgco2eq, "kgCO2eq") : NA)}</div>`;
}
function turnBlocks(sid, steps) {
  const turns = new Map();
  steps.forEach((s) => { const k = s.turn_index ?? "?"; if (!turns.has(k)) turns.set(k, []); turns.get(k).push(s); });
  const last = [...turns.keys()].pop();
  return [...turns.entries()].map(([k, ss]) => {
    const out = ss.reduce((a, s) => a + ((s.usage.output && s.usage.output.value) || 0), 0);
    const tools = ss.reduce((a, s) => a + s.tool_calls.length, 0);
    const bad = ss.filter((s) => s.error || s.incomplete).length;
    const side = `<span class="muted">${ss.length} step${ss.length === 1 ? "" : "s"} &middot; ${tools} tool${tools === 1 ? "" : "s"} &middot; out ${num(out)}</span>${bad ? `<span class="tag warn">${bad} issue${bad === 1 ? "" : "s"}</span>` : ""}`;
    return blk(`turn:${sid}:${k}`, `Turn ${esc(k)}`, `<div class="tscroll">${STEP_HEAD}${ss.map((s) => stepRow(s)).join("")}</tbody></table></div>`,
      { open: k === last || turns.size <= 3, side, cls: "turn" });
  }).join("");
}
async function showSession(sid) {
  const changed = state.session !== sid;
  state.session = sid;
  history.replaceState(null, "", "#session=" + encodeURIComponent(sid));
  const t = await getJSON(`${API}/sessions/${encodeURIComponent(sid)}/trajectory`);
  if (state.session !== sid) return;
  const s = t.session;
  state.steps = t.steps.map((x) => x.call_id);
  $("#traj-title").innerHTML = `Trajectory <span class="ids muted">${esc(short(sid))}</span>` +
    (s.continued_from_session ? ` <span class="tag warn" title="continues session ${esc(short(s.continued_from_session))}">continued from ${esc(short(s.continued_from_session))}</span>` : "");
  $("#traj-tools").hidden = false;
  let html = summaryBlock(s) + turnBlocks(sid, t.steps);

  if (t.children.length) {
    html += blk(`kids:${sid}`, `Subagents (${t.children.length}) <span class="muted">&mdash; attachment is inferred</span>`,
      t.children.map((c) => `<div class="card sub">
        <div class="top"><span class="ids">${esc(short(c.session.session_id))}</span> ${confTag(c.edge)}</div>
        <div class="meta">${esc(c.session.models.join(", "))} &middot; attached to step ${esc(stepNo(t.steps, c.parent_call_id))}
          ${c.session.green && c.session.green.calls_estimated ? `&middot; energy ${greenTotal(c.session.green)}` : ""}</div>
        <div class="meta"><a href="#" data-sid="${esc(c.session.session_id)}" class="open-session">open session</a></div></div>`).join(""),
      { open: true });
  }
  if (t.unattached.length) {
    html += blk(`unatt:${sid}`, `Could not be attached (${t.unattached.length})`,
      `<div class="muted">These began while a tool call here was open, but nothing identified which one spawned them. They are not guessed into a parent.</div>` +
      t.unattached.map((u) => `<div style="margin-top:6px">${esc(short(u.session_id))} &middot; ${esc(u.models.join(", "))}
        ${u.overlap_candidates ? `<span class="tag">${u.overlap_candidates} candidate calls</span>` : ""}
        &middot; <a href="#" data-sid="${esc(u.session_id)}" class="open-session">open session</a></div>`).join(""),
      { open: true, cls: "unatt" });
  }
  const keep = $("#traj").scrollTop;
  $("#traj").innerHTML = html;
  $("#traj").scrollTop = changed ? 0 : keep;
  if (state.step) {
    const r = $(`#traj tr[data-cid="${CSS.escape(state.step)}"]`);
    const d = r && r.closest("details.blk");
    if (d && !d.open) d.open = true;
  }
  $$("#sessions .card").forEach((c) => c.classList.toggle("sel", c.dataset.sid === sid));
}
function stepNo(steps, callId) {
  const s = steps.find((x) => x.call_id === callId);
  return s ? `${s.turn_index}.${s.step_index}` : "?";
}

// ================================================================== detail
function kv(rows) { return `<dl class="kv">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`; }
async function showStep(cid) {
  state.step = cid;
  const c = await getJSON(`${API}/calls/${encodeURIComponent(cid)}`);
  if (state.step !== cid) return;
  const u = c.usage, t = c.timing, g = c.green;
  const timing = [["TTFT any / text", `${ms(t.ttft_any_ms)} / ${ms(t.ttft_visible_ms)}`],
    ["total / duration", `${ms(t.total_ms)} / ${ms(t.duration_ms)}`],
    ["chunks", `${t.chunk_count} &middot; inter-chunk p50 ${ms(t.inter_chunk_p50_ms)} &middot; p95 ${ms(t.inter_chunk_p95_ms)}`]];
  const tokens = [["input (total)", metric(u.input_total)], ["output", metric(u.output)],
    ["cache read", metric(u.cache_read)], ["cache write", metric(u.cache_write)], ["provider usage", esc(JSON.stringify(u.raw || {}))]];
  const about = [["call", esc(c.call_id)], ["trace id", esc(c.trace_id || "—")], ["when", esc(t.received_at)],
    ["session", c.session_id ? `${esc(short(c.session_id))} &middot; step ${c.step_index}, turn ${c.turn_index}` : NA],
    ["client", `${esc(c.client.harness_hint)} &middot; ${esc(c.client.user_agent || "—")}`],
    ["request", `${esc(c.request.method)} ${esc(c.request.path)} &middot; ${esc(c.request.model_requested || "—")} &middot; ${num(c.request.message_count)} msgs &middot; ${num(c.request.tool_count)} tools &middot; ${num(c.request.body_bytes)} B`],
    ["served by", esc(c.response.model_served || "—")], ["stop", esc(c.response.stop_reason || "—")],
    ["thinking", esc(c.response.thinking_visibility || "—")], ["outcome", esc(JSON.stringify(c.outcome))]];
  if (c.subagent_edge) about.push(["subagent of", `call ${esc(short(c.parent_call_id))} ${confTag(c.subagent_edge)}`]);
  if (c.unattached) about.push(["attachment", '<span class="tag warn">could not be attached</span>']);
  let green = "";
  if (g && g.status === "ok") {
    green = kv([["energy (est.)", `${range(g.energy_kwh, "kWh")} &middot; basis ${esc(g.basis)}${basisTag(g.basis)}`],
      ["emissions (est.)", `${range(g.gwp_kgco2eq, "kgCO2eq")}`],
      ...(g.water_l ? [["water (est.)", `${range(g.water_l, "L")}`]] : []),
      ["caveats", `${esc(g.note)}${(g.warnings || []).length ? " &middot; " + esc(g.warnings.join(", ")) : ""}`],
      ["method", `${esc(g.method)} ${esc(g.method_version)} &middot; assumptions ${esc(short(g.assumptions_sha256))}`]]);
  } else if (g) green = kv([["energy (est.)", esc(g.status + (g.reason ? ": " + g.reason : ""))]]);

  const calls = c.response.tool_calls || [];
  const tools = calls.length ? blk(`d:tools`, `Tool calls (${calls.length})`,
    `<ul class="tools">${calls.map(toolLine).join("")}</ul><pre>${esc(JSON.stringify(calls.map((x) => ({ id: x.id, name: x.name, args: x.args })), null, 2))}</pre>`, { open: true }) : "";
  const delivered = (c.request.tool_results || []).length
    ? blk(`d:results`, `Results delivered in this request (${c.request.tool_results.length})`, `<pre>${esc(JSON.stringify(c.request.tool_results, null, 2))}</pre>`) : "";
  $("#detail").innerHTML = `<div id="convo"><div class="skel"></div></div>
    ${blk("d:timing", "Timing &amp; tokens", kv(timing) + kv(tokens), { open: true })}
    ${g ? blk("d:green", "Energy &amp; emissions (estimated)", green, { open: true }) : ""}
    ${tools}${delivered}${blk("d:about", "About this call", kv(about))}
    ${blk("d:raw", "Full record (JSON)", `<pre>${esc(JSON.stringify(c, null, 2))}</pre>`, { cls: "raw" })}`;
  loadConversation(cid);
  $$("tr[data-cid]").forEach((tr) => tr.classList.toggle("sel", tr.dataset.cid === cid));
  const row = $(`tr[data-cid="${CSS.escape(cid)}"]`);
  if (row && row.scrollIntoView) row.scrollIntoView({ block: "nearest" });
}
function stepBy(d) {
  if (!state.steps.length) return;
  const i = state.steps.indexOf(state.step);
  const n = state.steps[Math.min(state.steps.length - 1, Math.max(0, (i < 0 ? (d > 0 ? -1 : state.steps.length) : i) + d))];
  if (n) {
    // make sure the turn holding it is open
    const row = $(`tr[data-cid="${CSS.escape(n)}"]`);
    const det = row && row.closest("details.blk");
    if (det && !det.open) det.open = true;
    showStep(n);
  }
}

// ================================================================== conversation (prompt / response text)
function blockHtml(b) {
  const clip = b.truncated ? ` <span class="tag warn">cut at ${num(b.text.length)} of ${num(b.total_chars)} chars</span>` : "";
  if (b.kind === "text") return `<pre class="msg">${esc(b.text)}</pre>${clip}`;
  if (b.kind === "thinking") return `<details class="blk tight"><summary>reasoning</summary><div class="body"><pre class="msg">${esc(b.text)}</pre></div></details>`;
  if (b.kind === "tool_call") return `<div class="tc"><b>${esc(b.name || "?")}</b><pre class="msg">${esc(typeof b.args === "string" ? b.args : JSON.stringify(b.args, null, 2))}</pre></div>`;
  if (b.kind === "tool_result") return `<div class="tc">${b.is_error ? '<span class="tag bad">error</span> ' : ""}<span class="muted">tool result</span><pre class="msg">${esc(b.text)}</pre>${clip}</div>`;
  return `<pre class="msg">${esc(b.text || JSON.stringify(b))}</pre>`;
}
const bubble = (role, inner) => `<div class="bubble ${role === "user" ? "user" : "assistant"}"><div class="role">${esc(role)}</div>${inner}</div>`;
async function loadConversation(cid) {
  const el = $("#convo");
  try {
    const v = await getJSON(`${API}/calls/${encodeURIComponent(cid)}/conversation`);
    if (state.step !== cid) return;
    const rq = v.request, msgs = rq.new_messages || [];
    const asked = msgs.map((m) => bubble(m.role, m.blocks.map(blockHtml).join(""))).join("")
      || '<div class="muted">nothing new in this request</div>';
    const sys = rq.system ? blk("d:sys", `System prompt <span class="muted">${num(rq.system.chars)} chars</span>`, `<pre class="msg">${esc(rq.system.text)}</pre>`, { cls: "sys" }) : "";
    const tl = rq.tools && rq.tools.count ? blk("d:tooldefs", `${rq.tools.count} tool definitions`, `<div class="muted">${esc((rq.tools.names || []).join(", "))}</div>`, { cls: "sys" }) : "";
    el.innerHTML = `<div class="sectionlabel"><h3>Sent <span class="muted">&middot; new in this request; ${num(rq.earlier_messages)} earlier messages resent</span></h3></div>${asked}${sys}${tl}
      <div class="sectionlabel"><h3>Model replied</h3></div>${bubble("assistant", v.response.blocks.map(blockHtml).join("") || '<span class="muted">empty</span>')}
      ${v.response.note ? `<div class="muted">${esc(v.response.note)}</div>` : ""}
      <div class="muted" style="margin-top:6px">stored copy, pattern-redacted &middot; treat as sensitive</div>`;
  } catch (e) {
    el.innerHTML = '<div class="note">No prompt/response text stored for this call (store_content off, expired by retention, or not a model call).</div>';
  }
}

// ================================================================== all calls
function callRow(c) {
  const cls = c.error ? "failed" : c.incomplete ? "incomplete" : "";
  const status = c.status == null ? NA : `<span class="${c.status >= 400 ? "tag bad" : ""}">${c.status}</span>`;
  return `<tr data-cid="${esc(c.call_id)}" class="${cls}">
    <td title="${esc(c.received_at)}">${clock(c.received_at)}</td><td class="fmt-${esc(c.format)}">${esc(c.format)}</td>
    <td>${c.model ? esc(c.model) : NA}</td><td>${status}</td>
    <td class="num">${ms(c.ttft_any_ms)}</td><td class="num">${ms(c.ttft_visible_ms)}</td><td class="num">${ms(c.total_ms ?? c.duration_ms)}</td>
    <td class="num">${metric(c.usage.input_total)}</td><td class="num">${metric(c.usage.output)}</td>
    <td class="num">${metric(c.usage.cache_read)}</td><td class="num">${metric(c.usage.cache_write)}</td>
    <td class="num">${greenCell(c.green)}</td><td>${c.stop_reason ? esc(c.stop_reason) : NA}</td>
    <td>${c.tool_calls.length ? esc(c.tool_calls.join(", ")) : NA}</td><td>${flags(c)}</td></tr>`;
}
async function refreshCalls() {
  const d = await getJSON(`${API}/calls?limit=200`);
  const calls = d.calls.filter((c) => state.showUnknown || c.format !== "unknown");
  $("#count").textContent = `(${d.total} recorded, showing ${calls.length})`;
  $("#empty").hidden = calls.length > 0;
  $("#calls tbody").innerHTML = calls.map(callRow).join("");
}

// ================================================================== GREEN METRICS
const DEF = {
  energy: "Estimated electricity used by the model servers to produce the replies. EcoLogits models it from the model's size, the number of output tokens and the request latency. It is not measured.",
  gwp: "Greenhouse-gas emissions (kg CO₂-equivalent) for that electricity, using the EU-wide mix (EEE). The real figure depends on which country served the request.",
  water: "Estimated water use, including datacenter cooling and the water behind the electricity. Uses a world-average factor, so treat it as the weakest of the three.",
};
function rangeBar(r) {
  if (!r || !(r.max > 0)) return "";
  const lo = (r.min / r.max) * 100, mean = (r.mean / r.max) * 100;
  return `<div class="rangebar" title="min / mean / max. The marker is the mean."><i style="left:${lo.toFixed(1)}%;right:0"></i><b style="left:calc(${mean.toFixed(1)}% - 1px)"></b></div>`;
}
function kpi(label, r, base, def, ctx) {
  const h = human(r && r.mean, base);
  return `<div class="kpi"><div class="lab"><span>${label}</span><span title="${esc(def)}" style="cursor:help">&#9432;</span></div>
    <div class="hero">${r ? h.n : NA}<small>${r ? h.u : ""}</small></div>
    <div class="rng">${r ? `range ${hRange(r, base)} <span class="muted">(estimate)</span>` : "no estimate yet"}</div>
    ${rangeBar(r)}${ctx ? `<div class="ctx">${ctx}</div>` : ""}</div>`;
}
function hbars(items, key, base, max) {
  return `<div class="hbars">${items.map((m) => {
    const r = m[key]; const w = r && max ? (r.mean / max) * 100 : 0;
    const lo = r && max ? (r.min / max) * 100 : 0, hi = r && max ? (r.max / max) * 100 : 0;
    return `<div class="hbar"><div class="row"><span class="name" title="${esc(m.model)}">${esc(m.model)}${basisTag(m.basis)}</span>
      <span class="val">${r ? human(r.mean, base).n + " " + human(r.mean, base).u : NA}</span></div>
      <div class="track"><div class="fill" style="width:${w.toFixed(2)}%"></div><div class="whisk" style="left:${lo.toFixed(2)}%;width:${Math.max(0, hi - lo).toFixed(2)}%"></div></div>
      <div class="sub">${num(m.calls)} call${m.calls === 1 ? "" : "s"} &middot; ${num(m.output_tokens)} output tokens &middot; range ${hRange(r, base)}</div></div>`;
  }).join("")}</div>`;
}
function columns(points, bucket) {
  const max = Math.max(...points.map((p) => (p.energy_kwh ? p.energy_kwh.mean : 0)), 0);
  const lab = (iso) => { const d = new Date(iso); return bucket === "hour" ? d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false }) : d.toLocaleDateString([], { month: "short", day: "numeric" }); };
  const cols = points.map((p) => {
    const e = p.energy_kwh; const h = e && max ? Math.max(2, (e.mean / max) * 100) : 0;
    const tip = e ? `${lab(p.t)}\n${human(e.mean, "kWh").n} ${human(e.mean, "kWh").u} (range ${hRange(e, "kWh")})\n${p.calls} call${p.calls === 1 ? "" : "s"}` : `${lab(p.t)}\nno calls`;
    return `<div class="col ${e ? "" : "empty"}" data-tip="${esc(tip)}" tabindex="0"><i style="height:${h}%"></i></div>`;
  }).join("");
  const rows = points.filter((p) => p.energy_kwh).map((p) => `<tr><td>${esc(lab(p.t))}</td><td class="num">${num(p.calls)}</td><td class="num">${hRange(p.energy_kwh, "kWh")}</td><td class="num">${num(p.output_tokens)}</td></tr>`).join("");
  return `<div class="cols" role="img" aria-label="Estimated energy per ${bucket}">${cols}</div>
    <div class="axis"><span>${esc(lab(points[0].t))}</span><span>peak ${human(max, "kWh").n} ${human(max, "kWh").u} per ${bucket}</span><span>${esc(lab(points[points.length - 1].t))}</span></div>
    ${blk("g:tl-table", "Table view", `<table class="plain"><thead><tr><th>${bucket === "hour" ? "Hour" : "Day"}</th><th class="num">Calls</th><th class="num">Energy (est. range)</th><th class="num">Output tokens</th></tr></thead><tbody>${rows}</tbody></table>`, { cls: "tight" })}`;
}
const WARN_TEXT = {
  "model-arch-not-released": "The provider has not published this model's architecture, so EcoLogits guessed its size. Expect lower precision.",
  "model-arch-multimodal": "The model also handles images or audio; the estimate assumes text only.",
  "electricity-mix-wue-world": "Water-use factors fall back to a world average instead of a region-specific value.",
};
function renderGreen(a) {
  if (!a.calls.model_calls) {
    return `<div class="card2 empty"><b>No model calls in this window</b>Send a prompt through the proxy, or widen the time window.</div>`;
  }
  const t = a.totals, c = a.calls, tk = a.tokens, pc = a.per_call;
  const cov = c.model_calls ? (c.estimated / c.model_calls) * 100 : 0;
  const res = Object.entries(c.without_reasons || {}).map(([r, n]) => `${n} &times; ${esc(r)}`).join("; ");
  const kpis = `<div class="kpis">
    ${kpi("Energy", t.energy_kwh, "kWh", DEF.energy, pc.energy_kwh_mean != null ? `${human(pc.energy_kwh_mean, "kWh").n} ${human(pc.energy_kwh_mean, "kWh").u} per call on average` : "")}
    ${kpi("Emissions", t.gwp_kgco2eq, "kg", DEF.gwp, `CO₂-equivalent, EU electricity mix`)}
    ${kpi("Water", t.water_l, "L", DEF.water, t.water_l ? (t.water_calls < c.estimated ? `<span class="tag warn">partial</span> counted for ${t.water_calls} of ${c.estimated} estimated calls` : "weakest estimate: world-average factor") : "not available in these records")}
    <div class="kpi"><div class="lab"><span>Coverage</span><span title="Calls with an estimate out of all model calls" style="cursor:help">&#9432;</span></div>
      <div class="hero">${cov.toFixed(0)}<small>%</small></div>
      <div class="rng">${num(c.estimated)} of ${num(c.model_calls)} model calls estimated</div>
      <div class="meter" style="margin-top:12px"><i class="a" style="width:${cov}%"></i></div>
      <div class="ctx">${c.without_estimate ? `${num(c.without_estimate)} without an estimate: ${res}. They are left out of every total, not counted as zero.` : "every model call has an estimate"}</div></div></div>`;

  const maxE = Math.max(...a.by_model.map((m) => (m.energy_kwh && m.energy_kwh.max) || 0), 0);
  const models = `<div class="card2"><header><h2>Energy by model</h2><span class="muted">ranked; the thin line is the min&ndash;max range</span></header>
    ${hbars(a.by_model, "energy_kwh", "kWh", maxE)}
    ${a.by_model.some((m) => m.basis !== "exact") ? `<div class="note" style="margin-top:12px"><b>Not every model is an exact match.</b> Models tagged <span class="tag warn">proxy</span> or <span class="tag warn">fallback</span> were mapped to a similar one in the EcoLogits registry, so their estimates are weaker.</div>` : ""}</div>`;

  const sess = a.by_session.length ? `<div class="card2"><header><h2>Biggest sessions</h2><span class="muted">top ${a.by_session.length} of ${num(a.sessions_total)}</span></header>
    <table class="plain"><thead><tr><th>Session</th><th class="num">Calls</th><th class="num">Energy (est.)</th></tr></thead><tbody>
    ${a.by_session.map((s) => `<tr><td><a href="#" class="goto-session" data-sid="${esc(s.session_id)}"><span class="ids">${esc(short(s.session_id))}</span></a> <span class="muted">${esc(s.models.join(", "))}</span></td><td class="num">${num(s.calls)}</td><td class="num">${hRange(s.energy_kwh, "kWh")}</td></tr>`).join("")}</tbody></table></div>` : "";

  const tl = a.timeline.points.length ? `<div class="card2"><header><h2>Energy over time</h2><span class="muted">estimated mean per ${a.timeline.bucket}; hover a column</span></header>${columns(a.timeline.points, a.timeline.bucket)}</div>` : "";

  const tokens = `<div class="card2"><header><h2>What drives it</h2></header>
    <dl class="kv"><dt>Output tokens</dt><dd>${num(tk.output)} <span class="muted">&mdash; the main driver of the estimate</span></dd>
    <dt>Input tokens</dt><dd>${num(tk.input)} <span class="muted">&mdash; stored, but EcoLogits has no input term</span></dd>
    <dt>of which cache read</dt><dd>${num(tk.cache_read)} <span class="muted">&mdash; caching does not lower the estimate</span></dd>
    <dt>Energy per 1k output tokens</dt><dd>${pc.energy_kwh_per_1k_output_tokens != null ? human(pc.energy_kwh_per_1k_output_tokens, "kWh").n + " " + human(pc.energy_kwh_per_1k_output_tokens, "kWh").u : NA}</dd></dl></div>`;

  const warns = Object.entries(a.warnings).map(([w, n]) => `<li><b>${esc(w)}</b> <span class="muted">&times; ${num(n)}</span><div class="muted">${esc(WARN_TEXT[w] || "Reported by EcoLogits.")}</div></li>`).join("");
  const as = a.assumptions;
  const how = blk("g:how", "How these numbers are made, and what they assume",
    `<dl class="kv"><dt>Method</dt><dd>${as.methods.map((m) => `<span class="pill">${esc(m)}</span>`).join("") || NA}</dd>
      <dt>Region</dt><dd>${as.regions.map((m) => `<span class="pill">${esc(m)}</span>`).join("") || NA} <span class="muted">EU electricity mix (EEE): an average over several countries</span></dd>
      <dt>Estimates</dt><dd>${Object.entries(a.basis).map(([b, n]) => `<span class="pill">${num(n)} ${esc(b)}</span>`).join("")} <span class="muted">exact = model is in the EcoLogits registry</span></dd>
      <dt>Assumption sets</dt><dd>${Object.entries(as.assumption_hashes).map(([h, n]) => `<span class="pill" title="${esc(h)}">${esc(short(h))} &times; ${num(n)}</span>`).join("") || NA}</dd></dl>
     ${as.notes.map((n) => `<p class="muted">${esc(n)}</p>`).join("")}
     ${warns ? `<h3 style="margin-top:12px">Warnings from the library</h3><ul class="tools">${warns}</ul>` : ""}`, { open: false });
  const caveat = `<div class="note"><b>Estimates, not measurements.</b> Every figure is a range from a model of how the model runs, and the range is <b>not</b> a confidence interval.
    The real values could fall outside it. Do not use these in external reporting without human review.</div>`;

  return `${kpis}<div class="grid2">${models}${tokens}</div>${tl ? `<div style="margin-top:12px">${tl}</div>` : ""}
    ${sess ? `<div style="margin-top:12px">${sess}</div>` : ""}<div style="margin-top:12px">${how}</div><div style="margin-top:12px">${caveat}</div>`;
}
async function refreshGreen() {
  if (!state.greenSeen) { $("#green").innerHTML = '<div class="kpis"><div class="skel"></div><div class="skel"></div><div class="skel"></div><div class="skel"></div></div>'; }
  const a = await getJSON(`${API}/green?hours=${encodeURIComponent(state.hours)}`);
  state.greenSeen = true;
  const w = a.window;
  $("#green-window").textContent = w.first ? `${new Date(w.first).toLocaleString([], { hour12: false })} → ${new Date(w.last).toLocaleString([], { hour12: false })}` : "";
  const keep = window.scrollY;
  $("#green").innerHTML = renderGreen(a);
  window.scrollTo(0, keep);
}

// ================================================================== stats + wiring
async function refreshStats() {
  const s = await getJSON("/stats");
  const pct = (v) => (v == null ? "&mdash;" : (v * 100).toFixed(1) + "%");
  $("#stats").innerHTML = [
    ["calls", s.calls_total], ["incomplete", s.calls_incomplete], ["errors", s.calls_error],
    ["dropped records", s.dropped_records], ["queue depth", s.queue_depth],
    ["unattached rate", pct(s.unattached_rate)], ["tokens unreported", pct(s.unknown_token_share)],
    ["green unavailable", pct(s.unknown_green_share)],
    ["overhead p50 / p95", `${s.overhead_ms_p50 ?? "&mdash;"} / ${s.overhead_ms_p95 ?? "&mdash;"} ms`],
  ].map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("");
  const bad = (s.dropped_records || 0) > 0 || (s.record_errors || 0) > 0 || (s.chain_errors || 0) > 0;
  const warn = (s.calls_error || 0) > 0 || (s.unattached_rate || 0) > 0.2;
  $("#hdot").className = "hdot " + (bad ? "bad" : warn ? "warn" : "");
  $("#hsum").textContent = `${num(s.calls_total)} calls since start` + (bad ? " · recording problem" : warn ? " · check details" : "");
}
function setMode(m) {
  state.mode = m; store.set("mode", m);
  $$("button.mode").forEach((b) => { const on = b.dataset.mode === m; b.classList.toggle("on", on); b.setAttribute("aria-selected", on); });
  $("#mode-trajectory").hidden = m !== "trajectory";
  $("#mode-green").hidden = m !== "green";
  const q = new URLSearchParams(location.hash.slice(1));
  if (m === "green") { q.set("mode", "green"); } else q.delete("mode");
  history.replaceState(null, "", "#" + q.toString());
  tick();
}
function setView(v) {
  state.view = v;
  $$(".tab").forEach((t) => t.classList.toggle("on", t.dataset.view === v));
  $("#view-sessions").hidden = v !== "sessions";
  $("#view-calls").hidden = v !== "calls";
  tick();
}
function setPane(id, collapsed) {
  const pane = $("#" + id);
  pane.classList.toggle("collapsed", collapsed);
  const main = $("#view-sessions");
  main.classList.toggle(id === "sessions-pane" ? "no-sessions" : "no-detail", collapsed);
  const b = pane.querySelector(".chev");
  b.innerHTML = (id === "sessions-pane") === collapsed ? "&#8250;" : "&#8249;";
  b.title = collapsed ? "Expand this pane" : "Collapse this pane";
  store.set("pane:" + id, collapsed);
}

document.addEventListener("click", (e) => {
  const mode = e.target.closest("button.mode");
  if (mode) { setMode(mode.dataset.mode); return; }
  const tab = e.target.closest(".tab");
  if (tab) { setView(tab.dataset.view); return; }
  const w = e.target.closest("#window button");
  if (w) { state.hours = w.dataset.hours; $$("#window button").forEach((b) => b.classList.toggle("on", b === w)); store.set("hours", state.hours); refreshGreen(); return; }
  const col = e.target.closest("[data-collapse]");
  if (col) { const id = col.dataset.collapse; setPane(id, !$("#" + id).classList.contains("collapsed")); return; }
  const pane = e.target.closest(".pane.collapsed");
  if (pane) { setPane(pane.id, false); return; }
  if (e.target.closest("#expand-all")) { $$("#traj details.turn").forEach((d) => { d.open = true; }); return; }
  if (e.target.closest("#collapse-all")) { $$("#traj details.turn").forEach((d) => { d.open = false; }); return; }
  if (e.target.closest("#prev-step")) { stepBy(-1); return; }
  if (e.target.closest("#next-step")) { stepBy(1); return; }
  if (e.target.closest("#theme")) { toggleTheme(); return; }
  if (e.target.closest("#help")) { $("#helpbox").hidden = false; return; }
  if (e.target.closest("#help-close") || e.target.id === "helpbox") { $("#helpbox").hidden = true; return; }
  const goto = e.target.closest(".goto-session");
  if (goto) { e.preventDefault(); state.session = goto.dataset.sid; setMode("trajectory"); showSession(goto.dataset.sid); return; }
  const link = e.target.closest(".open-session");
  if (link) { e.preventDefault(); showSession(link.dataset.sid); return; }
  const card = e.target.closest("#sessions .card");
  if (card) { showSession(card.dataset.sid); return; }
  const row = e.target.closest("tr[data-cid]");
  if (row) showStep(row.dataset.cid);
});
function toggleTheme() {
  const cur = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  const next = cur === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("tt-theme", next); } catch (e) { /* ignore */ }
}
state.showUnknown = $("#show-unknown").checked;
$("#show-unknown").addEventListener("change", (e) => { state.showUnknown = e.target.checked; refreshCalls(); });
$("#sfilter").addEventListener("input", (e) => { state.filter = e.target.value; paintSessions(); });

let chord = 0;
document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement && document.activeElement.tagName);
  if (e.key === "Escape") { $("#helpbox").hidden = true; if (typing) document.activeElement.blur(); return; }
  if (typing) return;
  if (e.key === "/") { e.preventDefault(); $("#sfilter").focus(); return; }
  if (e.key === "?") { $("#helpbox").hidden = false; return; }
  if (e.key === "j") { stepBy(1); return; }
  if (e.key === "k") { stepBy(-1); return; }
  if (e.key === "g") { chord = Date.now(); return; }
  if (Date.now() - chord < 900) {
    if (e.key === "t") setMode("trajectory"); else if (e.key === "e") setMode("green");
    chord = 0;
  }
});
document.addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && e.target.matches && e.target.matches("#sessions .card")) { e.preventDefault(); showSession(e.target.dataset.sid); }
});

async function tick() {
  try {
    const jobs = [refreshStats()];
    if (state.mode === "green") jobs.push(refreshGreen());
    else if (state.view === "sessions") {
      jobs.push(refreshSessions());
      if (state.session) jobs.push(showSession(state.session));
    } else jobs.push(refreshCalls());
    await Promise.all(jobs);
    $("#live").className = "live";
    $("#live").innerHTML = "&#9679; live";
  } catch (err) {
    $("#live").className = "live down";
    $("#live").innerHTML = "&#9679; proxy unreachable";
  }
}
// Deep links: #session=<id>[&step=<call id>][&mode=green|&view=calls]. Saved preferences fill what the link does not say.
(function init() {
  const q = new URLSearchParams(location.hash.slice(1));
  if (q.get("theme") === "light" || q.get("theme") === "dark") document.documentElement.dataset.theme = q.get("theme");
  if (q.get("session")) state.session = q.get("session");
  if (q.get("step")) state.step = q.get("step");
  state.hours = store.get("hours", "all");
  $$("#window button").forEach((b) => b.classList.toggle("on", b.dataset.hours === state.hours));
  if (store.get("pane:sessions-pane", false)) setPane("sessions-pane", true);
  if (store.get("pane:detail-pane", false)) setPane("detail-pane", true);
  const mode = q.get("mode") || (q.get("session") ? "trajectory" : store.get("mode", "trajectory"));
  if (q.get("view") === "calls") state.view = "calls";
  $$(".tab").forEach((t) => t.classList.toggle("on", t.dataset.view === state.view));
  $("#view-sessions").hidden = state.view !== "sessions";
  $("#view-calls").hidden = state.view !== "calls";
  setMode(mode === "green" ? "green" : "trajectory");
})();
tick().then(() => { if (state.step && state.mode === "trajectory") showStep(state.step); });
setInterval(tick, 3000);
