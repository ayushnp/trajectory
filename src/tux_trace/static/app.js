"use strict";
const $ = (s) => document.querySelector(s);
const API = "/_trace/api";
const state = { view: "sessions", session: null, step: null, showUnknown: false };

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
// a session-level token total also says how many calls did not report it
function total(m) {
  if (!m || !m.reported) return NA;
  const miss = m.calls_missing ? ` <span class="tag" title="${m.calls_missing} call(s) did not report this">+${m.calls_missing} unreported</span>` : "";
  return num(m.value) + miss;
}

// ---- green: always a range, always an estimate, non-exact basis is flagged ----
function range(r, unit) {
  return r ? `~${sci(r.min)}&ndash;${sci(r.max)}${unit ? " " + unit : ""}` : NA;
}
function basisTag(b) {
  return !b || b === "exact" ? "" : ` <span class="tag warn" title="model is not in the EcoLogits registry (${esc(b)}); weaker estimate">${esc(b)}</span>`;
}
function greenCell(g) {
  if (!g || g.status === "disabled") return NA;
  if (g.status !== "ok") return `<span class="tag" title="${esc(g.reason || g.error || "")}">${esc(g.status)}</span>`;
  const r = g.energy_kwh;
  return `<span title="min ${sci(r.min)} / mean ${sci(r.mean)} / max ${sci(r.max)} kWh. Estimated, not measured.">${range(r)}</span>${basisTag(g.basis)}`;
}
function greenTotal(g) {
  if (!g || !g.calls_estimated) return NA;
  const mixed = Object.entries(g.basis_counts || {}).filter(([k]) => k !== "exact")
    .map(([k, n]) => `${n} ${esc(k)}`).join(", ");
  const warn = mixed ? ` <span class="tag warn" title="includes estimates that are not an exact model match">${mixed}</span>` : "";
  const miss = g.calls_without_estimate ? ` <span class="tag" title="no estimate for these calls">${g.calls_without_estimate} without</span>` : "";
  return `<span title="mean ${sci(g.energy_kwh.mean)} kWh. Estimated, not measured.">${range(g.energy_kwh, "kWh")}</span>${warn}${miss}`;
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
    s.compaction_suspected ? '<span class="tag warn" title="history rewritten">compaction?</span>' : ""].join(" ");
  return `<div class="card ${s.session_id === state.session ? "sel" : ""}" data-sid="${esc(s.session_id)}">
    <div class="top"><span class="ids">${esc(short(s.session_id))}</span><span class="muted">${clock(s.last_seen)}</span></div>
    <div>${esc(s.models.join(", "))} &middot; ${esc(s.harness)}${rel}</div>
    <div class="meta">${s.steps} step${s.steps === 1 ? "" : "s"} &middot; ${s.turns} turn${s.turns === 1 ? "" : "s"}
      &middot; in ${num(t.input_total.value)} &middot; out ${num(t.output.value)} &middot; TTFT p50 ${ms(s.ttft_p50_ms)}</div>
    <div class="meta">energy ${greenTotal(s.green)}</div>
    ${prob ? `<div class="meta">${prob}</div>` : ""}
  </div>`;
}

async function refreshSessions() {
  const d = await getJSON(`${API}/sessions?limit=100`);
  $("#scount").textContent = `(${d.total})`;
  $("#sempty").hidden = d.sessions.length > 0;
  $("#sessions").innerHTML = d.sessions.map(sessionCard).join("");
}

// ================================================================== trajectory
function stepRow(st, opts = {}) {
  const sel = st.call_id === state.step ? "sel" : "";
  const cls = [sel, st.error ? "failed" : st.incomplete ? "incomplete" : "", opts.child ? "child" : "",
    opts.turnStart ? "turn-start" : ""].join(" ");
  const tools = st.tool_calls.length
    ? `<ul class="tools">${st.tool_calls.map(toolLine).join("")}</ul>` : "";
  const gap = st.gap_since_previous_ms == null ? NA
    : `<span title="upper bound: tool + agent + network + any human wait">gap ${ms(st.gap_since_previous_ms)}</span>`;
  return `<tr data-cid="${esc(st.call_id)}" class="${cls}">
    <td>${opts.child ? "&#8627; " : ""}${st.turn_index == null ? NA : st.turn_index}.${st.step_index == null ? NA : st.step_index}</td>
    <td>${esc(st.model || "?")}${tools}</td>
    <td class="num">${ms(st.ttft_any_ms)}</td>
    <td class="num">${metric(st.usage.input_total)}</td>
    <td class="num">${metric(st.usage.output)}</td>
    <td class="num">${metric(st.usage.cache_read)}</td>
    <td class="num">${metric(st.usage.cache_write)}</td>
    <td>${st.stop_reason ? esc(st.stop_reason) : NA}</td>
    <td class="num">${gap}</td>
    <td class="num">${greenCell(st.green)}</td>
    <td>${flags(st)}</td>
  </tr>`;
}
function toolLine(t) {
  const res = !t.result_present ? '<span class="tag warn" title="no tool result was seen in a later request">no result observed</span>'
    : `${t.result_is_error ? '<span class="tag bad">error</span> ' : ""}${num(t.result_bytes)} B${t.gap_ms != null ? ` &middot; <span title="upper bound on tool time">gap ${ms(t.gap_ms)}</span>` : ""}`;
  return `<li><b>${esc(t.name || "?")}</b> &rarr; ${res}</li>`;
}

function trajTable(steps, opts = {}) {
  let lastTurn = null;
  const rows = steps.map((s) => {
    const row = stepRow(s, { ...opts, turnStart: !opts.child && lastTurn !== null && s.turn_index !== lastTurn });
    lastTurn = s.turn_index;
    return row;
  }).join("");
  return `<table class="steps"><thead><tr>
    <th title="turn.step">#</th><th>Model / tools</th><th class="num">TTFT</th><th class="num">In</th><th class="num">Out</th>
    <th class="num">Cache&nbsp;read</th><th class="num">Cache&nbsp;write</th><th>Stop</th><th class="num">Gap</th>
    <th class="num">Energy&nbsp;est.</th><th></th></tr></thead><tbody>${rows}</tbody></table>`;
}

function summaryBlock(s) {
  const t = s.tokens;
  return `<div class="sum">
    <div><div class="k">steps</div><div class="v">${s.steps}</div></div>
    <div><div class="k">turns</div><div class="v">${s.turns}</div></div>
    <div><div class="k">input</div><div class="v">${total(t.input_total)}</div></div>
    <div><div class="k">output</div><div class="v">${total(t.output)}</div></div>
    <div><div class="k">cache read</div><div class="v">${total(t.cache_read)}</div></div>
    <div><div class="k">cache write</div><div class="v">${total(t.cache_write)}</div></div>
    <div><div class="k">TTFT p50</div><div class="v">${ms(s.ttft_p50_ms)}</div></div>
    <div><div class="k">energy (est.)</div><div class="v">${greenTotal(s.green)}</div></div>
    <div><div class="k">emissions (est.)</div><div class="v">${s.green.gwp_kgco2eq ? range(s.green.gwp_kgco2eq, "kgCO2eq") : NA}</div></div>
  </div>`;
}

async function showSession(sid) {
  state.session = sid;
  history.replaceState(null, "", "#session=" + encodeURIComponent(sid));
  const t = await getJSON(`${API}/sessions/${encodeURIComponent(sid)}/trajectory`);
  const s = t.session;
  $("#traj-title").innerHTML = `Trajectory <span class="ids muted">${esc(short(sid))}</span>` +
    (s.continued_from_session ? ` <span class="tag warn" title="continues session ${esc(short(s.continued_from_session))}">continued from ${esc(short(s.continued_from_session))}</span>` : "");
  let html = summaryBlock(s) + trajTable(t.steps);

  if (t.children.length) {
    html += `<h3>Subagents (${t.children.length}) &mdash; attachment is inferred</h3>`;
    html += t.children.map((c) => `<div class="sub">
      <div>${esc(short(c.session.session_id))} &middot; ${esc(c.session.models.join(", "))} ${confTag(c.edge)}
        <span class="muted">attached to step ${esc(stepNo(t.steps, c.parent_call_id))}</span></div>
      ${c.session.green && c.session.green.calls_estimated ? `<div class="meta muted">energy ${greenTotal(c.session.green)}</div>` : ""}
      <div class="muted" style="margin:4px 0">
        <a href="#" data-sid="${esc(c.session.session_id)}" class="open-session">open session</a></div>
    </div>`).join("");
  }
  if (t.unattached.length) {
    html += `<h3>Could not be attached (${t.unattached.length})</h3>
      <div class="sub unatt"><div class="muted">These began while a tool call here was open, but nothing identified which one
      spawned them. They are not guessed into a parent.</div>` +
      t.unattached.map((u) => `<div style="margin-top:6px">${esc(short(u.session_id))} &middot; ${esc(u.models.join(", "))}
        ${u.overlap_candidates ? `<span class="tag">${u.overlap_candidates} candidate calls</span>` : ""}
        &middot; <a href="#" data-sid="${esc(u.session_id)}" class="open-session">open session</a></div>`).join("") + "</div>";
  }
  $("#traj").innerHTML = html;
  document.querySelectorAll("#sessions .card").forEach((c) => c.classList.toggle("sel", c.dataset.sid === sid));
}
function stepNo(steps, callId) {
  const s = steps.find((x) => x.call_id === callId);
  return s ? `${s.turn_index}.${s.step_index}` : "?";
}

// ================================================================== detail
async function showStep(cid) {
  state.step = cid;
  const c = await getJSON(`${API}/calls/${encodeURIComponent(cid)}`);
  const u = c.usage, t = c.timing, g = c.green;
  const rows = [
    ["call", esc(c.call_id)], ["trace id", esc(c.trace_id || "—")], ["when", esc(t.received_at)],
    ["session", c.session_id ? `${esc(short(c.session_id))} &middot; step ${c.step_index}, turn ${c.turn_index}` : NA],
    ["client", `${esc(c.client.harness_hint)} &middot; ${esc(c.client.user_agent || "—")}`],
    ["request", `${esc(c.request.method)} ${esc(c.request.path)} &middot; ${esc(c.request.model_requested || "—")} &middot; ${num(c.request.message_count)} msgs &middot; ${num(c.request.tool_count)} tools &middot; ${num(c.request.body_bytes)} B`],
    ["served by", esc(c.response.model_served || "—")],
    ["stop", esc(c.response.stop_reason || "—")],
    ["TTFT any / text", `${ms(t.ttft_any_ms)} / ${ms(t.ttft_visible_ms)}`],
    ["total / duration", `${ms(t.total_ms)} / ${ms(t.duration_ms)}`],
    ["chunks", `${t.chunk_count} &middot; inter-chunk p50 ${ms(t.inter_chunk_p50_ms)} &middot; p95 ${ms(t.inter_chunk_p95_ms)}`],
    ["tokens", `in ${metric(u.input_total)} &middot; out ${metric(u.output)} &middot; cache read ${metric(u.cache_read)} &middot; cache write ${metric(u.cache_write)}`],
    ["provider usage", esc(JSON.stringify(u.raw || {}))],
    ["thinking", esc(c.response.thinking_visibility || "—")],
    ["outcome", esc(JSON.stringify(c.outcome))],
  ];
  if (c.subagent_edge) rows.push(["subagent of", `call ${esc(short(c.parent_call_id))} ${confTag(c.subagent_edge)}`]);
  if (c.unattached) rows.push(["attachment", '<span class="tag warn">could not be attached</span>']);
  if (g && g.status === "ok") {
    rows.push(["energy (est.)", `${range(g.energy_kwh, "kWh")} (mean ${sci(g.energy_kwh.mean)}) &middot; basis ${esc(g.basis)}${basisTag(g.basis)}`]);
    rows.push(["emissions (est.)", `${range(g.gwp_kgco2eq, "kgCO2eq")} (mean ${sci(g.gwp_kgco2eq.mean)})`]);
    rows.push(["green caveats", `${esc(g.note)}${(g.warnings || []).length ? " &middot; " + esc(g.warnings.join(", ")) : ""}`]);
    rows.push(["green method", `${esc(g.method)} ${esc(g.method_version)} &middot; assumptions ${esc(short(g.assumptions_sha256))}`]);
  } else if (g) rows.push(["energy (est.)", esc(g.status + (g.reason ? ": " + g.reason : ""))]);
  const dl = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  const tools = (c.response.tool_calls || []).length
    ? `<h3>Tool calls</h3><ul class="tools">${c.response.tool_calls.map(toolLine).join("")}</ul>
       <pre>${esc(JSON.stringify(c.response.tool_calls.map((x) => ({ id: x.id, name: x.name, args: x.args })), null, 2))}</pre>` : "";
  const delivered = (c.request.tool_results || []).length
    ? `<h3>Results delivered in this request</h3><pre>${esc(JSON.stringify(c.request.tool_results, null, 2))}</pre>` : "";
  $("#detail").innerHTML = `<dl>${dl}</dl>${tools}${delivered}<h3>Full record</h3><pre>${esc(JSON.stringify(c, null, 2))}</pre>`;
  document.querySelectorAll("tr[data-cid]").forEach((tr) => tr.classList.toggle("sel", tr.dataset.cid === cid));
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
  // Health probes and other non-model requests are noise by default; they stay one checkbox away.
  const calls = d.calls.filter((c) => state.showUnknown || c.format !== "unknown");
  $("#count").textContent = `(${d.total} recorded, showing ${calls.length})`;
  $("#empty").hidden = calls.length > 0;
  $("#calls tbody").innerHTML = calls.map(callRow).join("");
}

// ================================================================== stats + wiring
async function refreshStats() {
  const s = await getJSON("/stats");
  const pct = (v) => (v == null ? "&mdash;" : (v * 100).toFixed(1) + "%");
  $("#stats").innerHTML = [
    ["calls", s.calls_total], ["incomplete", s.calls_incomplete], ["errors", s.calls_error],
    ["dropped", s.dropped_records], ["queue", s.queue_depth],
    ["unattached rate", pct(s.unattached_rate)], ["tokens unreported", pct(s.unknown_token_share)],
    ["green unavailable", pct(s.unknown_green_share)],
    ["overhead p50/p95", `${s.overhead_ms_p50 ?? "&mdash;"} / ${s.overhead_ms_p95 ?? "&mdash;"} ms`],
  ].map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join("");
}

document.addEventListener("click", (e) => {
  const tab = e.target.closest(".tab");
  if (tab) {
    state.view = tab.dataset.view;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t === tab));
    $("#view-sessions").hidden = state.view !== "sessions";
    $("#view-calls").hidden = state.view !== "calls";
    tick();
    return;
  }
  const link = e.target.closest(".open-session");
  if (link) { e.preventDefault(); showSession(link.dataset.sid); return; }
  const card = e.target.closest("#sessions .card");
  if (card) { showSession(card.dataset.sid); return; }
  const row = e.target.closest("tr[data-cid]");
  if (row) showStep(row.dataset.cid);
});
$("#show-unknown").addEventListener("change", (e) => { state.showUnknown = e.target.checked; refreshCalls(); });

async function tick() {
  try {
    const jobs = [refreshStats()];
    if (state.view === "sessions") {
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
// Deep links: #session=<id>[&step=<call id>] opens a session (and a step) directly.
(function fromHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  if (q.get("session")) state.session = q.get("session");
  if (q.get("step")) state.step = q.get("step");
  if (q.get("view") === "calls") document.querySelector('.tab[data-view="calls"]').click();
})();
tick().then(() => { if (state.step) showStep(state.step); });
setInterval(tick, 3000);
