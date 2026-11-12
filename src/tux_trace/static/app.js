"use strict";
const $ = (s) => document.querySelector(s);
const API = "/_trace/api";
let selected = null;

// "not reported" is shown as an em dash, never as 0.
const NA = '<span class="na">&mdash;</span>';
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const ms = (v) => (v == null ? NA : v >= 1000 ? (v / 1000).toFixed(2) + " s" : Math.round(v) + " ms");
const metric = (m) => (m && m.reported && m.value != null ? m.value.toLocaleString() : NA);
const clock = (iso) => new Date(iso).toLocaleTimeString([], { hour12: false });

// Energy is always shown as a range, never a bare number, and always marked as an estimate.
// A non-exact basis (hand-mapped or fallback model) is flagged so it is not mistaken for exact.
function sci(v) {
  if (v === 0) return "0";
  const e = Math.floor(Math.log10(Math.abs(v)));
  const m = v / Math.pow(10, e);
  return m.toFixed(1) + "e" + e;
}
function greenCell(g) {
  if (!g || g.status === "disabled") return NA;
  if (g.status !== "ok") return `<span class="tag" title="${esc(g.reason || g.error || "")}">${esc(g.status)}</span>`;
  const r = g.energy_kwh;
  const basis = g.basis === "exact" ? "" : ` <span class="tag warn" title="model not in the EcoLogits registry: ${g.basis}">${g.basis}</span>`;
  return `<span title="min ${sci(r.min)} / mean ${sci(r.mean)} / max ${sci(r.max)} kWh. Estimated, not measured.">~${sci(r.min)}&ndash;${sci(r.max)}</span>${basis}`;
}

function flags(c) {
  const out = [];
  if (c.error) out.push(`<span class="tag bad" title="${esc(JSON.stringify(c.error))}">error</span>`);
  if (c.disconnected) out.push('<span class="tag warn">client left</span>');
  else if (c.incomplete) out.push('<span class="tag warn">incomplete</span>');
  if (c.thinking && c.thinking !== "none") out.push(`<span class="tag" title="thinking visibility">think: ${c.thinking}</span>`);
  return out.join(" ");
}

function row(c) {
  const cls = c.error ? "failed" : c.incomplete ? "incomplete" : "";
  const status = c.status == null ? NA : `<span class="${c.status >= 400 ? "tag bad" : ""}">${c.status}</span>`;
  const tools = c.tool_calls.length ? esc(c.tool_calls.join(", ")) : NA;
  return `<tr data-id="${c.call_id}" class="${cls} ${c.call_id === selected ? "sel" : ""}">
    <td title="${esc(c.received_at)}">${clock(c.received_at)}</td>
    <td class="fmt-${c.format}">${esc(c.format)}</td>
    <td>${c.model ? esc(c.model) : NA}</td>
    <td>${status}</td>
    <td class="num">${ms(c.ttft_any_ms)}</td>
    <td class="num">${ms(c.ttft_visible_ms)}</td>
    <td class="num">${ms(c.total_ms ?? c.duration_ms)}</td>
    <td class="num">${metric(c.usage.input_total)}</td>
    <td class="num">${metric(c.usage.output)}</td>
    <td class="num">${metric(c.usage.cache_read)}</td>
    <td class="num">${metric(c.usage.cache_write)}</td>
    <td class="num">${greenCell(c.green)}</td>
    <td>${c.stop_reason ? esc(c.stop_reason) : NA}</td>
    <td>${tools}</td>
    <td>${flags(c)}</td>
  </tr>`;
}

async function refreshCalls() {
  const r = await fetch(`${API}/calls?limit=200`, { cache: "no-store" });
  const d = await r.json();
  $("#count").textContent = `(${d.total} recorded, showing ${d.calls.length})`;
  $("#empty").hidden = d.calls.length > 0;
  $("#calls tbody").innerHTML = d.calls.map(row).join("");
}

async function refreshStats() {
  const s = await (await fetch("/stats", { cache: "no-store" })).json();
  const pct = (v) => (v == null ? "&mdash;" : (v * 100).toFixed(1) + "%");
  $("#stats").innerHTML = [
    ["calls", s.calls_total], ["incomplete", s.calls_incomplete], ["errors", s.calls_error],
    ["dropped", s.dropped_records], ["queue", s.queue_depth],
    ["parse errors", s.parse_errors], ["tokens unreported", pct(s.unknown_token_share)],
    ["proxy overhead p50/p95", `${s.overhead_ms_p50 ?? "&mdash;"} / ${s.overhead_ms_p95 ?? "&mdash;"} ms`],
  ].map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join("");
}

async function showDetail(id) {
  selected = id;
  const r = await fetch(`${API}/calls/${id}`, { cache: "no-store" });
  if (!r.ok) return;
  const c = await r.json();
  const u = c.usage, t = c.timing;
  const dl = [
    ["call", c.call_id], ["trace id", c.trace_id || "—"], ["when", c.timing.received_at],
    ["client", `${c.client.harness_hint} · ${c.client.user_agent || "—"}`],
    ["request", `${c.request.method} ${c.request.path} · model ${c.request.model_requested || "—"} · ${c.request.message_count ?? "—"} msgs · ${c.request.tool_count ?? "—"} tools · ${c.request.body_bytes.toLocaleString()} B`],
    ["served by", c.response.model_served || "—"],
    ["stop", c.response.stop_reason || "—"],
    ["TTFT any / text", `${ms(t.ttft_any_ms)} / ${ms(t.ttft_visible_ms)}`],
    ["total / duration", `${ms(t.total_ms)} / ${ms(t.duration_ms)}`],
    ["chunks", `${t.chunk_count} · inter-chunk p50 ${ms(t.inter_chunk_p50_ms)} · p95 ${ms(t.inter_chunk_p95_ms)}`],
    ["tokens", `in ${metric(u.input_total)} · out ${metric(u.output)} · cache read ${metric(u.cache_read)} · cache write ${metric(u.cache_write)}`],
    ["provider-native usage", esc(JSON.stringify(u.raw || {}))],
    ["energy (est.)", c.green && c.green.status === "ok"
        ? `${sci(c.green.energy_kwh.min)} .. ${sci(c.green.energy_kwh.max)} kWh (mean ${sci(c.green.energy_kwh.mean)}) · basis ${c.green.basis}`
        : (c.green ? c.green.status : "—")],
    ["emissions (est.)", c.green && c.green.status === "ok"
        ? `${sci(c.green.gwp_kgco2eq.min)} .. ${sci(c.green.gwp_kgco2eq.max)} kgCO2eq`
        : "—"],
    ["green caveats", c.green && c.green.status === "ok" ? `${c.green.note} · ${(c.green.warnings || []).join(", ")}` : "—"],
    ["thinking", c.response.thinking_visibility || "—"],
    ["outcome", esc(JSON.stringify(c.outcome))],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${k === "tokens" || k.startsWith("TTFT") || k.startsWith("total") || k === "chunks" ? v : esc(v)}</dd>`).join("");
  const tools = (c.response.tool_calls || []).length
    ? `<h2>Tool calls</h2><pre>${esc(JSON.stringify(c.response.tool_calls, null, 2))}</pre>` : "";
  $("#detail").innerHTML = `<dl>${dl}</dl>${tools}<h2>Full record</h2><pre>${esc(JSON.stringify(c, null, 2))}</pre>`;
  $("#detail-pane").hidden = false;
  document.querySelector("main").classList.add("split");
  document.querySelectorAll("#calls tbody tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.id === id));
}

$("#calls tbody").addEventListener("click", (e) => {
  const tr = e.target.closest("tr");
  if (tr) showDetail(tr.dataset.id);
});
$("#close").addEventListener("click", () => {
  selected = null;
  $("#detail-pane").hidden = true;
  document.querySelector("main").classList.remove("split");
  document.querySelectorAll("#calls tbody tr.sel").forEach((tr) => tr.classList.remove("sel"));
});

async function tick() {
  try {
    await Promise.all([refreshCalls(), refreshStats()]);
    $("#live").className = "live";
    $("#live").innerHTML = "&#9679; live";
  } catch (e) {
    $("#live").className = "live down";
    $("#live").innerHTML = "&#9679; proxy unreachable";
  }
}
tick();
setInterval(tick, 3000);
