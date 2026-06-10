/* Trading Agent dashboard — single-page app. Read-only, refreshes every 20s. */
"use strict";

const REFRESH_MS = 20_000;
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const fmt$ = (v, dp = 2) =>
  v == null ? "—" : (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp });
const fmtNum = (v, dp = 1) => (v == null ? "—" : Number(v).toFixed(dp));
const fmtPct = (v, dp = 2) => (v == null ? "—" : `${v >= 0 ? "+" : ""}${Number(v).toFixed(dp)}%`);
const cls = (v) => (v > 0 ? "up" : v < 0 ? "down" : "flat");
const arrow = (v) => (v > 0 ? "▲" : v < 0 ? "▼" : "•");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const cap = (n) => (n == null ? "—" : n >= 1e12 ? (n / 1e12).toFixed(1) + "T" : n >= 1e9 ? (n / 1e9).toFixed(1) + "B" : n >= 1e6 ? (n / 1e6).toFixed(0) + "M" : String(n));

async function api(path) {
  const r = await fetch(`/api/${path}`);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

/* ---------- chart helpers ---------- */
Chart.defaults.font.family = "Inter, sans-serif";
Chart.defaults.color = "#8b93a3";
Chart.defaults.borderColor = "rgba(255,255,255,0.06)";
const charts = {};

function lineChart(canvasId, labels, datasets, opts = {}) {
  const el = document.getElementById(canvasId);
  if (!el) return;
  if (charts[canvasId]) charts[canvasId].destroy();
  charts[canvasId] = new Chart(el, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 350 },
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: !!opts.legend, labels: { boxWidth: 12, boxHeight: 2 } },
        tooltip: {
          backgroundColor: "#1d222b", borderColor: "#2a313d", borderWidth: 1,
          padding: 10, displayColors: false,
          callbacks: opts.money !== false
            ? { label: (c) => `${c.dataset.label || ""} ${fmt$(c.parsed.y)}` } : {},
        },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 7, maxRotation: 0 }, grid: { display: false } },
        y: { ticks: { maxTicksLimit: 5, callback: (v) => opts.money !== false ? "$" + Number(v).toLocaleString() : v }, grid: { color: "rgba(255,255,255,0.04)" } },
      },
      ...opts.extra,
    },
  });
}

function gradient(ctx, color) {
  const g = ctx.createLinearGradient(0, 0, 0, 260);
  g.addColorStop(0, color.replace(")", ", 0.18)").replace("rgb", "rgba"));
  g.addColorStop(1, color.replace(")", ", 0)").replace("rgb", "rgba"));
  return g;
}

/* ---------- shared widgets ---------- */
const BD_ORDER = ["technical", "momentum", "mtf", "statistical", "regime", "ml", "risk_reward", "relative_strength", "research"];
const BD_MAX = { technical: 20, momentum: 15, mtf: 15, statistical: 12, regime: 8, ml: 10, risk_reward: 10, relative_strength: 10, research: 25 };
const BD_LABEL = { mtf: "multi-timeframe", ml: "machine learning", risk_reward: "risk / reward", relative_strength: "rel. strength" };

function breakdownBars(bd) {
  if (!bd || !Object.keys(bd).length) return `<div class="empty" style="padding:14px">No breakdown captured</div>`;
  return `<div class="bars">` + BD_ORDER.filter((k) => k in bd).map((k) => {
    const v = bd[k], max = BD_MAX[k] || 20, w = Math.min(100, Math.abs(v) / max * 100);
    return `<div class="bar-row">
      <span class="name">${BD_LABEL[k] || k}</span>
      <div class="bar-track"><div class="bar-fill ${v < 0 ? "neg" : ""}" style="width:${w}%"></div></div>
      <span class="pts">${v > 0 ? "+" : ""}${fmtNum(v, 0)}</span>
    </div>`;
  }).join("") + `</div>`;
}

function scoreRing(score, gate = 70) {
  const s = Math.max(0, Math.min(100, score || 0));
  const color = s >= gate ? "var(--green)" : s >= gate - 10 ? "var(--amber)" : "var(--text-faint)";
  return `<div class="score-ring" style="background:conic-gradient(${color} ${s * 3.6}deg, var(--border) 0)">
    <div class="inner">${fmtNum(s, 0)}</div></div>`;
}

function signalLines(signals) {
  if (!signals?.length) return "";
  return `<div class="siglist">` + signals.map((s) =>
    `<div class="sigline ${s.ok ? "ok" : "no"}"><span class="mark">${s.ok ? "✅" : "✗"}</span><span>${esc(s.text)}</span></div>`
  ).join("") + `</div>`;
}

function researchLines(r) {
  if (!r || !Object.keys(r).length) return "";
  const rows = [];
  if (r.analyst_rating) rows.push({ ok: true, text: `Analysts: ${r.analyst_rating} (${r.analyst_n ?? "?"} analysts${r.upside_pct != null ? `, ${fmtPct(r.upside_pct, 1)} to target` : ""})` });
  if (r.insider_summary) rows.push({ ok: !/sold/i.test(r.insider_summary), text: `Insiders: ${r.insider_summary}` });
  if (r.news_headline) rows.push({ ok: r.news_emoji === "🟢", text: `News: ${r.news_headline}` });
  if (r.bull_pct != null) rows.push({ ok: r.bull_pct >= 60, text: `Social sentiment: ${fmtNum(r.bull_pct, 0)}% bullish` });
  if (r.earnings_label) rows.push({ ok: (r.earnings_days ?? 99) > 7, text: `Earnings ${r.earnings_label}` });
  if (r.points != null && r.points !== 0) rows.push({ ok: r.points > 0, text: `Research adjustment: ${r.points > 0 ? "+" : ""}${r.points} points` });
  return signalLines(rows);
}

/* =================== OVERVIEW =================== */
let eqRange = "1M";

async function renderOverview() {
  const page = $("#page-overview");
  let o;
  try { o = await api("overview"); } catch { return showError(page); }
  const icons = { trade: "●", option: "◆", skip: "○", scan: "↻", info: "ℹ", alert: "▲", manage: "✦" };
  page.innerHTML = `
    <div class="card hero">
      <div class="label">Portfolio value</div>
      <div class="equity">${fmt$(o.equity)}</div>
      <div class="today ${cls(o.today_pnl)}">${arrow(o.today_pnl)} ${fmt$(Math.abs(o.today_pnl))} (${fmtPct(o.today_pnl_pct)}) today</div>
      <div class="ranges">${["1D", "1W", "1M", "3M", "ALL"].map((r) =>
        `<button data-range="${r}" class="${r === eqRange ? "active" : ""}">${r}</button>`).join("")}</div>
      <div class="chartwrap"><canvas id="eq-chart"></canvas></div>
    </div>
    <div class="stats">
      <div class="card stat"><div class="k">Buying power</div><div class="v">${fmt$(o.buying_power, 0)}</div></div>
      <div class="card stat"><div class="k">Total return</div><div class="v ${cls(o.total_return_pct)}">${fmtPct(o.total_return_pct)}</div><div class="sub">since $100k start</div></div>
      <div class="card stat"><div class="k">Win rate</div><div class="v">${o.win_rate == null ? "—" : fmtNum(o.win_rate, 0) + "%"}</div><div class="sub">${o.trades_closed} closed trades</div></div>
      <div class="card stat"><div class="k">Open positions</div><div class="v">${o.open_positions}${o.open_options ? ` <span style="font-size:14px;color:var(--accent)">+ ${o.open_options} options</span>` : ""}</div><div class="sub">regime: ${esc(o.risk_state || "—")}</div></div>
    </div>
    <h2 class="section">Live activity</h2>
    <div class="card feed">
      ${o.activity.length ? o.activity.map((e) => `
        <div class="ev ${e.kind}"><span class="t">${esc(e.t.slice(5, 16))}</span>
        <span class="icon">${icons[e.kind] || "•"}</span><span>${esc(e.text)}</span></div>`).join("")
      : `<div class="empty">No recent activity</div>`}
    </div>`;
  $$(".ranges button", page).forEach((b) => b.onclick = () => { eqRange = b.dataset.range; drawEquity(); $$(".ranges button", page).forEach((x) => x.classList.toggle("active", x === b)); });
  drawEquity();
}

async function drawEquity() {
  let h;
  try { h = await api(`equity?range=${eqRange}`); } catch { return; }
  const pts = h.points || [];
  const labels = pts.map((p) => {
    const d = typeof p.t === "number" ? new Date(p.t * 1000) : new Date(p.t);
    return eqRange === "1D" ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      : d.toLocaleDateString([], { month: "short", day: "numeric" });
  });
  const vals = pts.map((p) => p.equity);
  const up = vals.length > 1 && vals[vals.length - 1] >= vals[0];
  const color = up ? "rgb(0, 200, 5)" : "rgb(255, 80, 0)";
  const el = document.getElementById("eq-chart");
  if (!el) return;
  lineChart("eq-chart", labels, [{
    data: vals, borderColor: color, borderWidth: 2, pointRadius: 0, tension: 0.25, fill: true,
    backgroundColor: gradient(el.getContext("2d"), color), label: "Equity",
  }]);
}

/* =================== POSITIONS =================== */
async function renderPositions() {
  const page = $("#page-positions");
  let d;
  try { d = await api("positions"); } catch { return showError(page); }
  const stocks = d.positions || [], options = d.options || [];

  let html = "";
  if (!stocks.length && !options.length) {
    html = `<div class="card empty"><div class="big">🌙</div>No open positions right now.<br>
      <span style="font-size:12.5px">The bot only trades setups scoring 70+ (65 for crypto) — empty slots in choppy markets are by design.</span></div>`;
  }
  if (stocks.length) {
    html += stocks.map((p) => {
      const w = p.progress != null ? Math.round(p.progress * 100) : 0;
      const rsn = p.reasoning || {};
      return `<div class="card poscard hoverable">
      <div class="poshead">
        <span class="sym">${esc(p.symbol)}</span>
        <span class="cname">${esc(p.company?.name || "")}${p.company?.sector ? " · " + esc(p.company.sector) : ""}</span>
        <span class="pnl-big ${cls(p.pnl)}">${arrow(p.pnl)} ${fmt$(Math.abs(p.pnl ?? 0))} (${fmtPct(p.pnl_pct)})</span>
      </div>
      <div class="blurb">${esc(p.company?.blurb || "")}</div>
      <div class="posmeta">
        <div class="m"><div class="k">Shares</div><div class="v">${fmtNum(p.qty, p.qty % 1 ? 4 : 0)}</div></div>
        <div class="m"><div class="k">Entry</div><div class="v">${fmt$(p.entry)}</div></div>
        <div class="m"><div class="k">Now</div><div class="v">${fmt$(p.current)}</div></div>
        <div class="m"><div class="k">Target</div><div class="v up">${fmt$(p.target)}</div></div>
        <div class="m"><div class="k">Stop</div><div class="v down">${fmt$(p.stop)}</div></div>
        <div class="m"><div class="k">Value</div><div class="v">${fmt$(p.value, 0)}</div></div>
        <div class="m"><div class="k">R now</div><div class="v">${p.r_multiple == null ? "—" : (p.r_multiple >= 0 ? "+" : "") + fmtNum(p.r_multiple, 2) + "R"}</div></div>
        <div class="m"><div class="k">Score</div><div class="v">${fmtNum(p.score, 0)}</div></div>
      </div>
      <div class="progress"><div class="fill" style="width:${w}%"></div></div>
      <div class="progress-labels"><span>entry ${fmt$(p.entry)}</span><span>${w}% of the way</span><span>target ${fmt$(p.target)}</span></div>
      ${p.breakeven || p.trailing || p.tranches?.length ? `<div class="wfoot" style="margin-top:10px">
        ${p.breakeven ? "<span>🔒 stop at breakeven</span>" : ""}
        ${p.trailing ? "<span>📈 trailing stop active</span>" : ""}
        ${p.tranches?.length ? `<span>✂️ scaled out at ${p.tranches.join(", ")}</span>` : ""}</div>` : ""}
      <details class="why"><summary>Why the bot opened this</summary><div class="whybody">
        ${rsn.summary ? `<div class="summary-line">${esc(rsn.summary)}</div>` : ""}
        <div class="score-row" style="margin-top:14px">${scoreRing(p.score)}${breakdownBars(rsn.breakdown)}</div>
        ${signalLines(rsn.signals)}
        ${researchLines(rsn.research)}
      </div></details>
    </div>`;
    }).join("");
  }

  html += `<h2 class="section">Options positions</h2>`;
  if (options.length) {
    html += `<div class="optgrid">` + options.map((o) => `
      <div class="card hoverable">
        <div class="poshead"><span class="sym">${esc(o.underlying)}</span>
          <span class="badge">${esc((o.type || "").toUpperCase())}</span>
          <span class="pnl-big ${cls(o.pnl)}">${fmt$(o.pnl)} (${fmtPct(o.pnl_pct, 1)})</span></div>
        <div class="posmeta" style="margin-top:12px">
          <div class="m"><div class="k">Strike</div><div class="v">${fmt$(o.strike, 0)}</div></div>
          <div class="m"><div class="k">Expiry</div><div class="v">${esc(o.expiration)}</div></div>
          <div class="m"><div class="k">Contracts</div><div class="v">${o.contracts}</div></div>
          <div class="m"><div class="k">Premium paid</div><div class="v">${fmt$(o.premium_paid)}</div></div>
          <div class="m"><div class="k">Value now</div><div class="v">${fmt$(o.value, 0)}</div></div>
          <div class="m"><div class="k">Target / stop</div><div class="v">${fmt$(o.target_premium)} / ${fmt$(o.stop_premium)}</div></div>
        </div>
        <div class="blurb" style="margin:8px 0 0">${esc(o.description || "")}</div>
      </div>`).join("") + `</div>`;
  } else {
    html += `<div class="card empty">No option positions — the bot buys an ATM call when a stock scores 80+.</div>`;
  }
  page.innerHTML = html;
}

/* =================== OPTIONS =================== */
async function renderOptionsPage() {
  const page = $("#page-options");
  let d;
  try { d = await api("options_overview"); } catch { return showError(page); }
  const pos = d.positions || [], setups = d.setups || [], cfg = d.config || {}, g = d.greeks;

  let html = "";
  // Portfolio Greeks summary (only when we actually have priced positions)
  if (g) {
    html += `<div class="stats" style="margin-top:0">
      <div class="card stat"><div class="k">Portfolio delta <span class="tip" data-tip="How many dollars the option book gains if every underlying rises $1. Positive = bullish exposure.">?</span></div>
        <div class="v ${cls(g.delta)}">${fmtNum(g.delta, 1)}</div></div>
      <div class="card stat"><div class="k">Theta / day <span class="tip" data-tip="Time decay: how many dollars the book loses each day just from the calendar, all else equal.">?</span></div>
        <div class="v ${cls(g.theta)}">${fmt$(g.theta)}</div></div>
      <div class="card stat"><div class="k">Vega <span class="tip" data-tip="Sensitivity to volatility: dollars gained per 1-point rise in implied volatility.">?</span></div>
        <div class="v">${fmtNum(g.vega, 1)}</div></div>
    </div>`;
  }

  if (pos.length) {
    html += `<h2 class="section">Open option positions</h2>`;
    html += pos.map((o) => `
      <div class="card poscard hoverable">
        <div class="poshead">
          <span class="sym">${esc(o.underlying)}</span>
          <span class="badge">${esc((o.type || "call").toUpperCase())} $${fmtNum(o.strike, 0)}</span>
          <span class="cname">${esc(o.company?.name || "")}</span>
          <span class="pnl-big ${cls(o.pnl)}">${arrow(o.pnl)} ${fmt$(Math.abs(o.pnl ?? 0))} (${fmtPct(o.pnl_pct, 1)})</span>
        </div>
        <div class="summary-line">🎯 ${esc(o.plain_english)}${o.dte != null ? ` — ${o.dte} days left` : ""}</div>
        <div class="blurb">${esc(o.company?.blurb || "")}</div>
        <div class="posmeta">
          <div class="m"><div class="k">Strike</div><div class="v">${fmt$(o.strike, 0)}</div></div>
          <div class="m"><div class="k">Expires</div><div class="v">${esc(o.expiration)}</div></div>
          <div class="m"><div class="k">Days left</div><div class="v">${o.dte ?? "—"}</div></div>
          <div class="m"><div class="k">Contracts</div><div class="v">${o.contracts}</div></div>
          <div class="m"><div class="k">Premium paid</div><div class="v">${fmt$(o.premium_paid)}</div></div>
          <div class="m"><div class="k">Value now</div><div class="v">${fmt$(o.value, 0)}</div></div>
          <div class="m"><div class="k">Target</div><div class="v up">${fmt$(o.target_premium)}</div></div>
          <div class="m"><div class="k">Stop</div><div class="v down">${fmt$(o.stop_premium)}</div></div>
        </div>
        ${o.description ? `<details class="why"><summary>Why the bot bought this</summary>
          <div class="whybody"><div class="summary-line">${esc(o.description)}</div>
          ${o.score ? `<div class="wfoot" style="margin-top:8px"><span>signal score ${fmtNum(o.score, 0)}/100 (needs ${fmtNum(cfg.min_score, 0)}+)</span></div>` : ""}
          </div></details>` : ""}
      </div>`).join("");
  } else {
    html += `<div class="card empty"><div class="big">🎯</div>No options positions yet — the bot will buy calls when a stock scores ${fmtNum(cfg.min_score, 0)}+.<br>
      <span style="font-size:12.5px">When it fires: ATM call, ${cfg.dte_min}–${cfg.dte_max} days out, max ${cfg.max_positions} at a time,
      sell at +${fmtNum(cfg.profit_target_pct, 0)}% or cut at −${fmtNum(cfg.stop_loss_pct, 0)}% of premium.</span></div>`;
  }

  html += `<h2 class="section">Best setups the bot is seeing</h2>`;
  if (setups.length) {
    html += `<div class="watchgrid">` + setups.map((s) => {
      const color = s.ready ? "var(--green)" : (s.score >= s.gate - 10 ? "var(--amber)" : "var(--text-faint)");
      return `<div class="card hoverable">
        <div class="whead"><span class="sym" style="font-size:16px">${esc(s.symbol)}</span>
          <span class="wname">${esc(s.name || "")}</span>
          <span class="pill ${s.ready ? "bullish" : "neutral"}">${s.ready ? "ready" : `${fmtNum(s.gap, 0)} pts away`}</span></div>
        <div class="scorebar">
          <div class="track"><div class="fill" style="width:${s.score}%;background:${color}"></div>
            <div class="gate-mark" style="left:${s.gate}%"></div></div>
          <div class="nums"><span>score ${fmtNum(s.score, 1)}</span><span>calls fire at ${fmtNum(s.gate, 0)}+</span></div>
        </div>
        <div class="wreasons"><span>${esc(s.plain_english)}</span></div>
        ${s.research_points ? `<div class="wfoot"><span>research ${s.research_points > 0 ? "+" : ""}${s.research_points} pts</span></div>` : ""}
      </div>`;
    }).join("") + `</div>`;
  } else {
    html += `<div class="card empty">No long candidates being scored right now.</div>`;
  }
  page.innerHTML = html;
}

/* =================== CRYPTO =================== */
async function renderCryptoPage() {
  const page = $("#page-crypto");
  let d;
  try { d = await api("crypto"); } catch { return showError(page); }
  const coins = d.coins || [], pos = d.positions || [];
  const n = { up: d.uptrend_count, neutral: coins.filter((c) => c.status === "neutral").length,
              short: coins.filter((c) => c.status === "short_biased").length };

  let html = `
    <div class="card hero" style="padding-bottom:18px">
      <div class="label">Crypto desk — 24/7</div>
      <div class="today" style="font-size:17px;margin-top:6px" >${esc(d.headline)}</div>
      <div class="wfoot" style="margin-top:10px">
        <span style="color:var(--green)">▲ ${n.up} uptrend</span>
        <span style="color:var(--amber)">• ${n.neutral} neutral</span>
        <span style="color:var(--red)">▼ ${n.short} short-biased</span>
        <span>${d.total - d.with_data} no Alpaca data</span>
        <span>longs need: score ${fmtNum(d.gate, 0)}+ · confirmed daily uptrend · outperform BTC</span>
      </div>
    </div>`;

  if (pos.length) {
    html += `<h2 class="section">Open crypto positions</h2>`;
    html += pos.map((p) => {
      const rsn = p.reasoning || {};
      return `<div class="card poscard hoverable">
        <div class="poshead">
          <span class="sym">${esc(p.symbol)}</span>
          <span class="cname">${esc(p.company?.name || "")}</span>
          <span class="pnl-big ${cls(p.pnl)}">${arrow(p.pnl)} ${fmt$(Math.abs(p.pnl ?? 0))} (${fmtPct(p.pnl_pct)})</span>
        </div>
        <div class="posmeta" style="margin-top:10px">
          <div class="m"><div class="k">Qty</div><div class="v">${fmtNum(p.qty, 4)}</div></div>
          <div class="m"><div class="k">Entry</div><div class="v">${fmt$(p.entry)}</div></div>
          <div class="m"><div class="k">Now</div><div class="v">${fmt$(p.current)}</div></div>
          <div class="m"><div class="k">Target</div><div class="v up">${fmt$(p.target)}</div></div>
          <div class="m"><div class="k">Stop</div><div class="v down">${fmt$(p.stop)}</div></div>
          <div class="m"><div class="k">Score</div><div class="v">${fmtNum(p.score, 0)}</div></div>
        </div>
        ${rsn.summary ? `<details class="why"><summary>Why the bot opened this</summary><div class="whybody">
          <div class="summary-line">${esc(rsn.summary)}</div>
          <div class="score-row" style="margin-top:12px">${scoreRing(p.score, 65)}${breakdownBars(rsn.breakdown)}</div>
          ${signalLines(rsn.signals)}
        </div></details>` : ""}
      </div>`;
    }).join("");
  }

  const statusPill = { uptrend: ["bullish", "uptrend ✓"], neutral: ["neutral", "neutral"],
                       short_biased: ["avoid", "short-biased"], no_data: ["avoid", "no data"] };
  html += `<h2 class="section">All ${d.total} watched pairs</h2><div class="watchgrid">`;
  html += coins.map((c) => {
    const [pillCls, pillTxt] = statusPill[c.status] || ["avoid", c.status];
    if (!c.has_data) {
      return `<div class="card" style="opacity:.55">
        <div class="whead"><span class="sym" style="font-size:16px">${esc(c.coin)}</span>
          <span class="wname">${esc(c.name)}</span><span class="pill avoid">no data</span></div>
        <div class="wreasons"><span>${esc(c.blurb)}</span><span>· Not available on Alpaca's crypto feed</span></div>
      </div>`;
    }
    const sc = c.score || 0;
    const color = sc >= c.gate ? "var(--green)" : sc >= c.gate - 10 ? "var(--amber)" : "var(--text-faint)";
    const px = c.price >= 100 ? fmt$(c.price, 0) : c.price >= 1 ? fmt$(c.price) : fmt$(c.price, 4);
    return `<div class="card wcard hoverable" data-sym="${esc(c.symbol)}">
      <div class="whead"><span class="sym" style="font-size:16px">${esc(c.coin)}</span>
        <span class="wname">${esc(c.name)}</span>
        <span class="pill ${pillCls}">${pillTxt}</span></div>
      <div style="display:flex;align-items:baseline;gap:10px;margin-top:8px">
        <span style="font-size:18px;font-weight:700;font-variant-numeric:tabular-nums">${px}</span>
        <span class="${cls(c.chg24_pct)}" style="font-size:13px;font-weight:600">${fmtPct(c.chg24_pct)} 24h</span>
      </div>
      <div class="scorebar">
        <div class="track"><div class="fill" style="width:${sc}%;background:${color}"></div>
          <div class="gate-mark" style="left:${c.gate}%"></div></div>
        <div class="nums"><span>score ${fmtNum(sc, 1)}</span><span>longs at ${fmtNum(c.gate, 0)}+</span></div>
      </div>
      <div class="wreasons">
        ${c.is_benchmark ? `<span>★ The benchmark — every other coin is measured against it</span>`
          : c.beats_btc == null ? ""
          : `<span class="${c.beats_btc ? "up" : ""}">${c.beats_btc ? "✓ Outperforming Bitcoin" : "✗ Lagging Bitcoin"}${c.rs20_pct != null ? ` (${fmtPct(c.rs20_pct, 1)} over 20d)` : ""}</span>`}
        <span>${c.uptrend ? "✓ Daily uptrend confirmed — eligible for a long" : "· Daily uptrend not confirmed — the bot stays out"}</span>
        <span>${esc(c.blurb)}</span>
      </div>
    </div>`;
  }).join("") + `</div>`;
  page.innerHTML = html;
  $$(".wcard", page).forEach((c) => c.onclick = () => openSymbolModal(c.dataset.sym));
}

/* =================== ORDERS =================== */
const fmtAgo = (iso) => {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
};
const fmtDur = (s) => {
  if (s == null) return "—";
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
};

async function renderOrders() {
  const page = $("#page-orders");
  let d;
  try { d = await api("orders"); } catch { return showError(page); }
  const s = d.summary || {}, pending = d.pending || [], prot = d.protective || [], fills = d.filled || [];

  let html = `
    <div class="card hero" style="padding-bottom:16px">
      <div class="label">Order desk</div>
      <div class="today" style="font-size:17px;margin-top:6px">
        ${s.pending ?? 0} pending · ${s.queued ?? 0} queued for the open · ${s.filled_today ?? 0} filled today</div>
      <div class="wfoot" style="margin-top:8px">
        <span>${s.protective ?? 0} protective exit${s.protective === 1 ? "" : "s"} working</span>
        ${s.waiting_to_arm ? `<span>${s.waiting_to_arm} more arm${s.waiting_to_arm === 1 ? "s" : ""} when pending entries fill</span>` : ""}
        <span>${d.market_open ? "🟢 market open — orders execute live" : "🌙 market closed — stock orders wait for the bell"}</span>
      </div>
    </div>`;

  html += `<h2 class="section">Waiting to open</h2>`;
  if (pending.length) {
    html += `<div class="watchgrid">` + pending.map((o) => `
      <div class="card hoverable">
        <div class="whead"><span class="sym" style="font-size:16px">${esc(o.symbol)}</span>
          <span class="wname">${esc(o.name || (o.option ? "option contract" : ""))}</span>
          ${o.queued ? `<span class="pill neutral">queued for open</span>` : `<span class="pill bullish">working</span>`}</div>
        <div class="posmeta" style="margin-top:10px;margin-bottom:6px">
          <div class="m"><div class="k">Side / type</div><div class="v">${esc(o.side)} ${esc(o.type)}</div></div>
          <div class="m"><div class="k">Qty</div><div class="v">${fmtNum(o.qty, o.qty % 1 ? 4 : 0)}</div></div>
          <div class="m"><div class="k">Limit</div><div class="v">${fmt$(o.limit_price)}</div></div>
          <div class="m"><div class="k">Last</div><div class="v">${fmt$(o.last)}</div></div>
        </div>
        <div class="wreasons"><span>${esc(o.plain)}</span></div>
        <div class="wfoot"><span>submitted ${fmtAgo(o.submitted_at)}</span><span>status: ${esc(o.status)}</span></div>
      </div>`).join("") + `</div>`;
  } else {
    html += `<div class="card empty">No entry orders working — the bot places one the moment a setup clears its gate.</div>`;
  }

  html += `<h2 class="section">Protecting open positions</h2>`;
  if (prot.length) {
    html += `<div class="card" style="overflow-x:auto"><table class="trades"><thead><tr>
      <th>Symbol</th><th>Protection</th><th>Qty</th><th>Triggers at</th><th>Last</th><th>Distance</th><th>Status</th></tr></thead><tbody>
      ${prot.map((o) => `<tr${o.armed ? "" : ` style="opacity:.55"`}>
        <td><b>${esc(o.symbol)}</b></td>
        <td><span class="pill ${o.kind === "take_profit" ? "bullish" : "bearish"}" style="margin-left:0">${o.kind === "take_profit" ? "take profit" : "stop loss"}</span></td>
        <td>${fmtNum(o.qty, o.qty % 1 ? 4 : 0)}</td>
        <td class="${o.kind === "take_profit" ? "up" : "down"}">${fmt$(o.price)}</td>
        <td>${fmt$(o.last)}</td>
        <td>${o.away_pct == null ? "—" : fmtPct(o.away_pct, 1) + " away"}</td>
        <td style="color:var(--text-dim)">${o.armed ? esc(o.status) : "arms after entry fills"}</td></tr>`).join("")}
      </tbody></table></div>`;
  } else {
    html += `<div class="card empty">No protective orders working right now.</div>`;
  }

  html += `<h2 class="section">Recently filled — last 7 days</h2>`;
  if (fills.length) {
    html += `<div class="card" style="overflow-x:auto"><table class="trades"><thead><tr>
      <th>Filled</th><th>Symbol</th><th>Side</th><th>Type</th><th>Qty</th><th>Price</th></tr></thead><tbody>
      ${fills.map((o) => `<tr>
        <td style="color:var(--text-dim)">${new Date(o.filled_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</td>
        <td><b>${esc(o.symbol)}</b>${o.option ? ` <span style="color:var(--accent);font-size:11px">option</span>` : ""}</td>
        <td class="${o.side === "buy" ? "up" : "down"}">${esc(o.side)}</td>
        <td>${esc(o.type)}</td>
        <td>${fmtNum(o.filled_qty ?? o.qty, (o.filled_qty ?? o.qty) % 1 ? 4 : 0)}</td>
        <td>${fmt$(o.filled_avg_price)}</td></tr>`).join("")}
      </tbody></table></div>`;
  } else {
    html += `<div class="card empty">Nothing filled in the last 7 days.</div>`;
  }
  page.innerHTML = html;
}

/* =================== BOT =================== */
async function renderBot() {
  const page = $("#page-bot");
  let d;
  try { d = await api("bot"); } catch { return showError(page); }
  const st = d.status || {}, k = d.kill || {};

  const meter = (usedFrac, pct, limitPct, pnl, label) => {
    const f = usedFrac == null ? 0 : usedFrac;
    const color = f < 0.5 ? "var(--green)" : f < 0.8 ? "var(--amber)" : "var(--red)";
    return `<div style="margin-top:10px">
      <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text-dim)">
        <span>${label}: <b class="${cls(pnl)}">${fmt$(pnl)} (${fmtPct(pct)})</b></span>
        <span>halts at ${fmtNum(limitPct, 0)}%</span></div>
      <div class="meter"><div class="fill" style="width:${Math.max(2, f * 100)}%;background:${color}"></div></div>
      <div style="font-size:11px;color:var(--text-faint)">${pct == null ? "no data yet" :
        f >= 1 ? "⛔ limit hit — trading halted" : `${Math.round(f * 100)}% of the buffer used`}</div>
    </div>`;
  };

  page.innerHTML = `
    <div class="card hero" style="padding-bottom:18px">
      <div class="label">Bot health</div>
      <div class="equity" style="font-size:26px;display:flex;align-items:center;gap:10px">
        <span class="dot ${st.online ? "on" : "off"}" style="width:11px;height:11px"></span>
        ${st.halted ? `<span class="down">HALTED — kill switch</span>`
          : st.online ? `<span class="up">RUNNING</span>` : `<span class="down">OFFLINE</span>`}
        <span style="font-size:13px;color:var(--text-dim);font-weight:500">${esc(st.mode || "")} mode · scans every ${st.scan_interval ?? "—"}s</span>
      </div>
      <div class="wfoot" style="margin-top:10px">
        <span>last scan ${fmtAgo(st.last_scan)}</span>
        <span>uptime ${fmtDur(st.uptime_s)}</span>
        <span>${st.scans_today ?? 0} scans today</span>
        <span>${st.market_open ? "🟢 market open" : "🌙 market closed"}</span>
      </div>
    </div>

    <h2 class="section">Risk &amp; kill switch</h2>
    <div class="card">
      ${meter(k.daily_used_frac, k.daily_pct, k.daily_limit_pct, k.daily_pnl, "Today")}
      ${meter(k.weekly_used_frac, k.weekly_pct, k.weekly_limit_pct, k.weekly_pnl, "This week")}
      <div class="wfoot" style="margin-top:12px">
        <span>also halts after ${k.max_consecutive_losses} consecutive losing trades</span>
        <span>measured from the bot's day-start equity — the Overview hero shows change since prior market close</span></div>
    </div>

    <h2 class="section">Connections</h2>
    <div class="card">
      ${(d.connections || []).map((c) => `
        <div class="connrow"><span class="dot ${c.ok ? "on" : "off"}"></span>
          <b>${esc(c.name)}</b><span style="color:var(--text-dim)">${esc(c.detail)}</span>
          <span style="margin-left:auto;color:${c.ok ? "var(--green)" : "var(--red)"};font-weight:600">${c.ok ? "connected" : "down"}</span></div>`).join("")}
      ${Object.keys(d.sources || {}).length ? `
        <div class="connrow" style="border-bottom:none;flex-wrap:wrap">
          <b>Research sources</b>
          ${Object.entries(d.sources).map(([name, v]) => `
            <span style="color:${v === "ok" ? "var(--green)" : "var(--amber)"};font-size:12px">● ${esc(name)}</span>`).join("")}
        </div>` : ""}
    </div>

    <h2 class="section">Active strategies</h2>
    <div class="card">
      ${(d.strategies || []).map((sgy) => `
        <div class="connrow">
          <span class="pill ${sgy.active ? "bullish" : "avoid"}" style="margin-left:0">${sgy.active ? "ON" : "OFF"}</span>
          <b>${esc(sgy.name)}</b>
          <span style="color:var(--text-dim);font-size:12.5px">${esc(sgy.desc)}</span></div>`).join("")}
    </div>

    <h2 class="section">Current settings</h2>
    <div class="cfggrid">
      ${(d.settings || []).map((cgf) => `
        <div class="cfg"><div class="k">${esc(cgf.k)}</div><div class="v">${esc(cgf.v)}</div></div>`).join("")}
    </div>`;
}

/* =================== REASONING =================== */
async function renderReasoning() {
  const page = $("#page-reasoning");
  let d;
  try { d = await api("reasoning"); } catch { return showError(page); }
  const trades = d.trades || [];
  if (!trades.length) {
    page.innerHTML = `<div class="card empty"><div class="big">🧠</div>No trades captured yet.<br>
      <span style="font-size:12.5px">Every future trade's full reasoning is snapshotted the moment it opens and kept forever.</span></div>`;
    return;
  }
  page.innerHTML = trades.map((t) => `
    <div class="card poscard">
      <div class="poshead">
        <span class="sym">${esc(t.symbol)}</span>
        <span class="trade-status ${t.status}">${t.status.toUpperCase()}</span>
        <span class="cname">${new Date(t.opened_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</span>
        ${t.status === "closed" && t.exit_pnl != null ? `<span class="pnl-big ${cls(t.exit_pnl)}">${fmt$(t.exit_pnl)}</span>` : ""}
      </div>
      ${t.summary ? `<div class="summary-line">${esc(t.summary)}</div>` : ""}
      <div class="score-row" style="margin-top:16px">
        ${scoreRing(t.score)}
        ${breakdownBars(t.breakdown)}
      </div>
      ${signalLines(t.signals)}
      ${researchLines(t.research)}
      <div class="levels">
        <span>Entry <b>${fmt$(t.entry)}</b></span>
        <span>Stop <b class="down">${fmt$(t.stop)}</b></span>
        <span>Target <b class="up">${fmt$(t.target)}</b></span>
        <span>Reward:risk <b>${t.rr ? fmtNum(t.rr, 1) + " : 1" : "—"}</b></span>
        ${t.qty ? `<span>Size <b>${fmtNum(t.qty, 0)}</b></span>` : ""}
        ${t.regime ? `<span>Regime <b>${esc(String(t.regime).split("/")[1] || t.regime).replaceAll("_", " ")}</b></span>` : ""}
      </div>
    </div>`).join("");
}

/* =================== WATCHING =================== */
let watchSort = "score", watchFilter = "all", watchQuery = "";

async function renderWatching() {
  const page = $("#page-watching");
  let d;
  try { d = await api("watching"); } catch { return showError(page); }
  let rows = d.symbols || [];
  if (watchFilter !== "all") rows = rows.filter((r) => r.status === watchFilter);
  if (watchQuery) rows = rows.filter((r) => (r.symbol + " " + (r.name || "")).toLowerCase().includes(watchQuery.toLowerCase()));
  rows.sort((a, b) => watchSort === "score" ? (b.score || 0) - (a.score || 0)
    : watchSort === "symbol" ? a.symbol.localeCompare(b.symbol)
    : (a.earnings_days ?? 999) - (b.earnings_days ?? 999));

  page.innerHTML = `
    <div class="controls">
      <input id="w-q" placeholder="Search symbol or company…" value="${esc(watchQuery)}" />
      <select id="w-filter">
        ${["all", "bullish", "neutral", "avoid"].map((f) => `<option value="${f}" ${f === watchFilter ? "selected" : ""}>${f === "all" ? "All statuses" : f}</option>`).join("")}
      </select>
      <select id="w-sort">
        <option value="score" ${watchSort === "score" ? "selected" : ""}>Sort: score</option>
        <option value="symbol" ${watchSort === "symbol" ? "selected" : ""}>Sort: symbol</option>
        <option value="earnings" ${watchSort === "earnings" ? "selected" : ""}>Sort: earnings soonest</option>
      </select>
      <span class="updated" style="margin-left:auto;align-self:center">top ${rows.length} deep-analyzed this scan</span>
    </div>
    <div class="watchgrid">
      ${rows.map((r) => {
        const sc = r.score || 0, gate = r.gate || 70;
        const color = sc >= gate ? "var(--green)" : sc >= gate - 10 ? "var(--amber)" : "var(--text-faint)";
        return `<div class="card wcard hoverable" data-sym="${esc(r.symbol)}">
        <div class="whead"><span class="sym" style="font-size:16px">${esc(r.symbol)}</span>
          <span class="wname">${esc(r.name || "")}</span>
          <span class="pill ${r.status}">${r.status}</span></div>
        <div class="scorebar">
          <div class="track"><div class="fill" style="width:${sc}%;background:${color}"></div>
            <div class="gate-mark" style="left:${gate}%"></div></div>
          <div class="nums"><span>score ${fmtNum(sc, 1)}</span><span>trades at ${gate}+</span></div>
        </div>
        <div class="wreasons">${(r.reasons || []).map((x) => `<span>· ${esc(x)}</span>`).join("")}</div>
        <div class="wfoot">
          ${r.sector ? `<span>${esc(r.sector)}</span>` : ""}
          ${r.market_cap ? `<span>${cap(r.market_cap)} cap</span>` : ""}
          ${r.earnings_label ? `<span>📅 earnings ${esc(r.earnings_label)}</span>` : ""}
          ${r.research_points ? `<span>research ${r.research_points > 0 ? "+" : ""}${r.research_points}</span>` : ""}
        </div>
      </div>`;
      }).join("")}
    </div>
    ${!rows.length ? `<div class="card empty">Nothing matches.</div>` : ""}`;

  $("#w-q").oninput = (e) => { watchQuery = e.target.value; renderWatching(); };
  $("#w-filter").onchange = (e) => { watchFilter = e.target.value; renderWatching(); };
  $("#w-sort").onchange = (e) => { watchSort = e.target.value; renderWatching(); };
  $$(".wcard", page).forEach((c) => c.onclick = () => openSymbolModal(c.dataset.sym));
}

async function openSymbolModal(symbol) {
  const bg = $("#modal-bg"), modal = $("#modal");
  modal.innerHTML = `<div class="skeleton" style="height:300px"></div>`;
  bg.classList.add("show");
  let d;
  try { d = await api(`symbol/${encodeURIComponent(symbol)}`); } catch { modal.innerHTML = `<div class="empty">Couldn't load ${esc(symbol)}</div>`; return; }
  const a = d.analysis || {}, c = d.company || {};
  modal.innerHTML = `
    <button class="close" id="modal-close">✕</button>
    <div class="poshead"><span class="sym" style="font-size:22px">${esc(symbol)}</span>
      <span class="cname">${esc(c.name || "")}${c.sector ? " · " + esc(c.sector) : ""}${c.market_cap ? " · " + cap(c.market_cap) + " cap" : ""}</span></div>
    <div class="blurb">${esc(c.blurb || "")}</div>
    <div class="chartwrap" style="height:220px;margin:14px 0"><canvas id="sym-chart"></canvas></div>
    <div class="score-row">${scoreRing(a.score, a.gate)}${breakdownBars(a.breakdown)}</div>
    ${signalLines(a.signals)}
    ${researchLines(d.research)}
    ${a.notes?.length ? `<div class="wfoot" style="margin-top:12px">${a.notes.map((n) => `<span>⚑ ${esc(n)}</span>`).join("")}</div>` : ""}`;
  $("#modal-close").onclick = () => bg.classList.remove("show");
  const ch = d.chart || [];
  if (ch.length) {
    lineChart("sym-chart", ch.map((p) => p.t.slice(5)), [
      { label: "Close", data: ch.map((p) => p.close), borderColor: "#e8eaf0", borderWidth: 1.8, pointRadius: 0, tension: 0.2 },
      { label: "21 EMA", data: ch.map((p) => p.ema21), borderColor: "#6e7bf2", borderWidth: 1.2, pointRadius: 0, tension: 0.2 },
      { label: "50 EMA", data: ch.map((p) => p.ema50), borderColor: "#f5a623", borderWidth: 1.2, pointRadius: 0, tension: 0.2 },
    ], { legend: true });
  }
}

/* =================== PERFORMANCE =================== */
const TIPS = {
  win: "Of all closed round-trip trades, the percentage that made money. High win rate with small wins can still lose overall — read it with avg win/loss.",
  sharpe: "Risk-adjusted return: average daily return divided by its volatility, annualized. Above 1 is good, above 2 is excellent. Needs weeks of data to mean much.",
  dd: "The worst peak-to-trough drop in account value. If you had $100k and it fell to $95k before recovering, that's a -5% max drawdown.",
  avgw: "Average profit on winning trades vs average loss on losing ones. The bot aims for small losses and larger wins (asymmetric risk/reward).",
};

async function renderPerformance() {
  const page = $("#page-performance");
  let p;
  try { p = await api("performance"); } catch { return showError(page); }
  const hist = p.equity || [];
  const labels = hist.map((x) => new Date((typeof x.t === "number" ? x.t * 1000 : x.t)).toLocaleDateString([], { month: "short", day: "numeric" }));
  // drawdown series
  let peak = -Infinity;
  const dd = hist.map((x) => { peak = Math.max(peak, x.equity); return peak > 0 ? ((x.equity - peak) / peak) * 100 : 0; });

  page.innerHTML = `
    <div class="stats">
      <div class="card stat"><div class="k">Win rate <span class="tip" data-tip="${TIPS.win}">?</span></div>
        <div class="v">${p.win_rate == null ? "—" : fmtNum(p.win_rate, 0) + "%"}</div><div class="sub">${p.wins}W / ${p.losses}L of ${p.trades}</div></div>
      <div class="card stat"><div class="k">Sharpe <span class="tip" data-tip="${TIPS.sharpe}">?</span></div>
        <div class="v">${p.sharpe ?? "—"}</div><div class="sub">annualized, daily returns</div></div>
      <div class="card stat"><div class="k">Max drawdown <span class="tip" data-tip="${TIPS.dd}">?</span></div>
        <div class="v down">${fmtNum(p.max_drawdown_pct, 2)}%</div></div>
      <div class="card stat"><div class="k">Avg win / loss <span class="tip" data-tip="${TIPS.avgw}">?</span></div>
        <div class="v"><span class="up">${fmt$(p.avg_win, 0)}</span> / <span class="down">${fmt$(p.avg_loss, 0)}</span></div></div>
    </div>
    <h2 class="section">Equity & drawdown</h2>
    <div class="card"><div class="chartwrap"><canvas id="perf-chart"></canvas></div></div>
    <h2 class="section">Monthly returns</h2>
    <div class="card"><div class="heatmap">${(p.monthly || []).map((m) => {
      const v = m.ret_pct, alpha = Math.min(0.55, Math.abs(v) / 6 + 0.08);
      const bgc = v >= 0 ? `rgba(0,200,5,${alpha})` : `rgba(255,80,0,${alpha})`;
      return `<div class="hm-cell" style="background:${bgc}"><div class="mo">${m.month}</div><div class="rv">${fmtPct(v, 1)}</div></div>`;
    }).join("") || `<div class="empty">Not enough history yet</div>`}</div></div>
    ${p.best ? `
    <h2 class="section">Best & worst trades</h2>
    <div class="stats">
      <div class="card stat"><div class="k">Best</div><div class="v up">${fmt$(p.best.pnl)}</div>
        <div class="sub">${esc(p.best.symbol)} · ${fmtPct(p.best.pnl_pct, 1)} · ${new Date(p.best.exit_at).toLocaleDateString()}</div></div>
      <div class="card stat"><div class="k">Worst</div><div class="v down">${fmt$(p.worst.pnl)}</div>
        <div class="sub">${esc(p.worst.symbol)} · ${fmtPct(p.worst.pnl_pct, 1)} · ${new Date(p.worst.exit_at).toLocaleDateString()}</div></div>
    </div>` : ""}
    <h2 class="section">Recent closed trades</h2>
    <div class="card" style="overflow-x:auto">
      ${p.recent?.length ? `<table class="trades"><thead><tr>
        <th>Symbol</th><th>Type</th><th>Qty</th><th>Entry</th><th>Exit</th><th>P/L</th><th>%</th><th>Closed</th></tr></thead><tbody>
        ${p.recent.map((t) => `<tr><td><b>${esc(t.symbol)}</b></td><td>${t.option ? "option" : "stock"}</td>
          <td>${fmtNum(t.qty, 0)}</td><td>${fmt$(t.entry_px)}</td><td>${fmt$(t.exit_px)}</td>
          <td class="${cls(t.pnl)}">${fmt$(t.pnl)}</td><td class="${cls(t.pnl)}">${fmtPct(t.pnl_pct, 1)}</td>
          <td style="color:var(--text-dim)">${new Date(t.exit_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</td></tr>`).join("")}
        </tbody></table>` : `<div class="empty">No closed trades yet</div>`}
    </div>`;

  if (hist.length) {
    lineChart("perf-chart", labels, [
      { label: "Equity", data: hist.map((x) => x.equity), borderColor: "#6e7bf2", borderWidth: 2, pointRadius: 0, tension: 0.25, yAxisID: "y" },
      { label: "Drawdown %", data: dd, borderColor: "rgba(255,80,0,0.7)", borderWidth: 1.2, pointRadius: 0, tension: 0.25, fill: true, backgroundColor: "rgba(255,80,0,0.07)", yAxisID: "y1" },
    ], { legend: true, extra: { scales: {
      x: { ticks: { maxTicksLimit: 8, maxRotation: 0 }, grid: { display: false } },
      y: { ticks: { maxTicksLimit: 5, callback: (v) => "$" + Number(v).toLocaleString() }, grid: { color: "rgba(255,255,255,0.04)" } },
      y1: { position: "right", ticks: { maxTicksLimit: 4, callback: (v) => v.toFixed(1) + "%" }, grid: { display: false } },
    } } });
  }
}

/* =================== shell =================== */
function showError(page) {
  page.innerHTML = `<div class="card empty"><div class="big">📡</div>Couldn't reach the dashboard server.<br>
    <span style="font-size:12.5px">It retries automatically every ${REFRESH_MS / 1000}s.</span></div>`;
}

const RENDER = { overview: renderOverview, positions: renderPositions, orders: renderOrders, reasoning: renderReasoning, watching: renderWatching, options: renderOptionsPage, crypto: renderCryptoPage, performance: renderPerformance, bot: renderBot };
let current = "overview";

function route() {
  const target = (location.hash.replace("#/", "") || "overview");
  current = RENDER[target] ? target : "overview";
  $$("nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === current));
  $$(".page").forEach((p) => p.classList.toggle("visible", p.id === `page-${current}`));
  RENDER[current]();
}

async function heartbeat() {
  try {
    const h = await api("health");
    $("#agent-status").innerHTML = h.agent_online
      ? `<span class="dot on"></span>agent live` : `<span class="dot off"></span>agent offline`;
    $("#market-status").textContent = h.market_open ? "market open" : "market closed";
    $("#offline-banner").classList.toggle("show", !h.agent_online);
    $("#mode-badge").textContent = h.paper ? "PAPER" : "LIVE";
    if (h.halted) { $("#market-status").textContent = "⛔ HALTED"; }
    $("#updated").textContent = "updated " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    $("#agent-status").innerHTML = `<span class="dot off"></span>server unreachable`;
  }
}

window.addEventListener("hashchange", route);
route();
heartbeat();
setInterval(() => { heartbeat(); RENDER[current](); }, REFRESH_MS);
