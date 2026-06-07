"""Beginner-friendly trading dashboard — "Cash App meets Robinhood".

Five plain-English pages (Your Money / Trades / Watching / Performance / Bot)
with big friendly numbers, green=up/red=down, emojis, and a glossary. Reads the
agent's live state + Alpaca account; falls back to clearly-labelled sample data
so every page looks full before the bot has traded.

Run:  streamlit run src/monitoring/dashboard_app.py  ->  http://localhost:8501
"""

from __future__ import annotations

import asyncio
import datetime as _dt

# A live asyncio loop per script run (Py3.14 + Alpaca/yfinance otherwise raise
# "Event loop is closed" across Streamlit reruns).
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from config import settings
from src.monitoring.sample_data import build_sample
from src.monitoring.state_store import StateStore

st.set_page_config(page_title="My Trading Bot", layout="wide", page_icon="💰",
                   initial_sidebar_state="collapsed")

GREEN, RED, GREY, INK, SUB = "#00B843", "#F23645", "#8A94A6", "#0E1117", "#6B7280"

st.markdown(f"""
<style>
  .block-container {{padding-top: 1rem; max-width: 980px;}}
  #MainMenu, footer {{visibility: hidden;}}
  .hero {{background: linear-gradient(135deg,#0E1117,#1c2230); color:#fff;
          border-radius:22px; padding:26px 28px; margin-bottom:14px;}}
  .hero .label {{color:#9aa4b2; font-size:15px; font-weight:600;}}
  .hero .big {{font-size:46px; font-weight:800; letter-spacing:-1px; margin:2px 0;}}
  .card {{background:#fff; border:1px solid #eef0f4; border-radius:18px;
          padding:18px 20px; margin-bottom:14px; box-shadow:0 1px 3px rgba(0,0,0,.04);}}
  .mini .v {{font-size:24px; font-weight:800; color:{INK};}}
  .mini .l {{font-size:13px; color:{SUB}; font-weight:600;}}
  .mini .s {{font-size:12px; color:{GREY};}}
  .pos-title {{font-size:20px; font-weight:800; color:{INK};}}
  .tag {{font-size:13px; font-weight:800; padding:4px 12px; border-radius:999px;}}
  .row {{display:flex; justify-content:space-between; align-items:center;}}
  .muted {{color:{SUB}; font-size:14px;}}
  .reason {{font-size:14px; color:{INK}; margin:2px 0;}}
  .bar {{position:relative; height:10px; background:#eef0f4; border-radius:999px; margin:10px 0;}}
  .bar > .fill {{position:absolute; left:0; top:0; bottom:0; background:{GREEN}; border-radius:999px;}}
  .bar > .dot {{position:absolute; top:-5px; width:20px; height:20px; background:#fff;
                border:3px solid {GREEN}; border-radius:50%; transform:translateX(-50%);}}
  .feed {{font-size:14px; color:{INK}; padding:8px 0; border-bottom:1px solid #f1f3f7;}}
  .pill {{display:inline-block; font-size:12px; font-weight:700; color:{SUB};
          background:#f4f6fa; border-radius:999px; padding:2px 10px; margin-right:6px;}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_store() -> StateStore:
    return StateStore(log_dir=settings.LOG_DIR)


def money(x: float, cents: bool = True) -> str:
    return f"${x:,.2f}" if cents else f"${x:,.0f}"


def signed(x: float) -> str:
    return f"+{money(x)}" if x >= 0 else f"-{money(abs(x))}"


def col_for(x: float) -> str:
    return GREEN if x >= 0 else RED


def dot(good: bool) -> str:
    return "🟢" if good else "🔴"


# --------------------------------------------------------------------------- #
# Data — live state mapped into the sample shape, or the sample itself
# --------------------------------------------------------------------------- #
_EMOJI = {"AAPL": "🍎", "NVDA": "🎮", "TSLA": "⚡", "MSFT": "🪟", "AMZN": "🛒",
          "GOOGL": "🔍", "META": "👥", "AMD": "💻", "NFLX": "📺", "JPM": "🏦",
          "BTC/USD": "🟠", "ETH/USD": "💎", "SOL/USD": "☀️", "HOOD": "🪶"}


def _emoji(sym: str) -> str:
    return _EMOJI.get(sym, "📈")


def build_live(state: dict) -> dict:
    started = 100_000.0
    equity = float(state.get("equity") or started)
    closed = state.get("closed_today", [])
    wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
    positions = []
    managed = {m["symbol"]: m for m in state.get("managed", [])}
    for p in state.get("open_positions", []):
        sym = p.get("symbol", "?")
        m = managed.get(sym, {})
        entry = float(m.get("entry") or 0)
        positions.append({
            "emoji": _emoji(sym), "name": sym, "symbol": sym,
            "shares": p.get("qty", 0), "paid": entry, "now": 0.0,
            "cost": 0.0, "value": 0.0, "pnl": float(p.get("pnl", 0)),
            "pnl_pct": float(p.get("pnl_pct", 0)), "target": float(m.get("target") or 0),
            "stop": float(m.get("current_stop") or 0),
            "progress": max(0.0, min(1.0, float(m.get("last_r") or 0) / 4.0)),
            "score": float(m.get("score") or 0),
            "reasons": ["Passed the bot's full checklist"],
        })
    # Options positions (shown separately from stocks).
    options = []
    for o in state.get("options", []):
        paid = float(o.get("cost_basis") or 0)
        value = float(o.get("value") or 0)
        options.append({
            "emoji": _emoji(o.get("underlying", "")), "underlying": o.get("underlying", "?"),
            "symbol": o.get("symbol", ""), "type": o.get("type", "call"),
            "strike": float(o.get("strike") or 0), "expiration": o.get("expiration", ""),
            "contracts": int(o.get("contracts") or 0),
            "premium_paid": float(o.get("premium_paid") or 0),
            "current_premium": float(o.get("current_premium") or 0),
            "paid": paid, "value": value, "pnl": float(o.get("pnl") or 0),
            "pnl_pct": float(o.get("pnl_pct") or 0), "score": float(o.get("score") or 0),
            "description": o.get("description", ""),
            "target_premium": float(o.get("target_premium") or 0),
            "stop_premium": float(o.get("stop_premium") or 0),
        })
    # Show the FULL watchlist: live score where the bot has deep-scanned it,
    # else "awaiting scan" (the pre-rank only deep-scans the top names each cycle).
    scores_by = {s["symbol"]: s for s in state.get("scores", [])}
    research_by = state.get("research", {})
    watching = []
    for sym in settings.load_watchlist():
        s = scores_by.get(sym)
        scored = s is not None
        sc = float(s.get("score", 0)) if scored else 0.0
        view = "BULLISH" if (scored and sc >= 70) else ("BEARISH" if (scored and sc < 50) else "NEUTRAL")
        watching.append({
            "emoji": _emoji(sym), "name": sym, "symbol": sym, "price": 0.0, "chg": 0.0,
            "view": view, "score": sc, "scored": scored,
            "status": "OWNED" if sym in managed else ("WATCHING" if (not scored or sc >= 50) else "AVOIDING"),
            "reasons": ["Bot's live confidence score"] if scored else ["Pre-screened — awaiting next deep scan"],
            "research": research_by.get(sym)})
    # Scored names first, by score.
    watching.sort(key=lambda w: (w["scored"], w["score"]), reverse=True)
    return {
        "is_sample": False, "started": started, "equity": equity, "options": options,
        "daily_pnl": float(state.get("daily_pnl", 0)),
        "daily_pct": float(state.get("daily_pnl", 0)) / started * 100,
        "all_time_pnl": equity - started, "all_time_pct": (equity - started) / started * 100,
        "win_rate": (wins / len(closed)) if closed else 0.0,
        "equity_history": state.get("equity_history", []),
        "positions": positions,
        "trades": [{"emoji": _emoji(t.get("symbol", "")), "name": t.get("symbol", "?"),
                    "symbol": t.get("symbol", "?"), "win": t.get("pnl", 0) >= 0,
                    "bought_on": "—", "bought_at": 0, "sold_on": "—", "sold_at": 0,
                    "days": 0, "pnl": float(t.get("pnl", 0)),
                    "pnl_pct": t.get("r_multiple", 0), "score": 0, "note": ""}
                   for t in closed],
        "watching": watching, "activity": [],
        "bot": {"running": not state.get("halted", False), "last_scan_min": 0,
                "next_scan_min": 5, "uptime": "—", "scans_today": 0,
                "setups_looked": 0, "trades_taken": len(positions),
                "setups_rejected": 0, "daily_loss_used": max(0.0, -float(state.get("daily_pnl", 0))),
                "daily_loss_limit": started * settings.DAILY_LOSS_LIMIT,
                "telegram": True, "broker": True, "logging": True,
                "min_score": settings.MIN_SCORE, "rr": settings.RR_RATIO,
                "risk_pct": settings.RISK_PER_TRADE * 100,
                "source_status": state.get("source_status", {})},
        "performance": {"sharpe": 0.0, "max_drawdown_pct": 0.0, "max_drawdown_dollar": 0,
                        "avg_winner": 0.0, "avg_loser": 0.0, "monthly": []},
    }


def load_data():
    state = get_store().read_state()
    has_real = bool(state.get("open_positions") or state.get("closed_today"))
    default = "Live" if has_real else "Demo"
    mode = st.session_state.get("data_mode", default)
    return (build_sample() if mode == "Demo" else build_live(state)), mode, default


# --------------------------------------------------------------------------- #
# Shared UI
# --------------------------------------------------------------------------- #
GLOSSARY = {
    "Safety net (stop loss)": "A safety net price. If the stock drops to this price, "
        "the bot automatically sells to prevent bigger losses.",
    "Target price": "The price the bot thinks the stock could reach. When it gets "
        "there, the bot sells for profit.",
    "Score": "How confident the bot is about a trade. 100 = extremely confident, "
        "0 = no confidence.",
    "4:1 ratio": "For every $1 the bot risks, it tries to make $4 back.",
}


def top_bar(d: dict, mode: str):
    up = d["daily_pnl"] >= 0
    st.markdown(
        f"<div class='row'>"
        f"<div><span class='pill'>🤖 Bot {dot(d['bot']['running'])}</span>"
        f"<span class='pill'>{'🧪 Sample data' if mode=='Demo' else '🔴 Live data'}</span></div>"
        f"<div style='text-align:right'><span style='font-weight:800;font-size:18px'>{money(d['equity'])}</span> "
        f"<span style='color:{col_for(d['daily_pnl'])};font-weight:700'>"
        f"{'▲' if up else '▼'} {signed(d['daily_pnl'])} today</span></div></div>",
        unsafe_allow_html=True)


def friendly_glossary():
    st.caption("Tap to learn a term:")
    cols = st.columns(len(GLOSSARY))
    for col, (term, defn) in zip(cols, GLOSSARY.items()):
        with col:
            with st.popover(f"❓ {term.split('(')[0].strip()}"):
                st.markdown(f"**{term}**\n\n{defn}")


# --------------------------------------------------------------------------- #
# PAGE 1 — Your Money
# --------------------------------------------------------------------------- #
def page_home(d):
    up = d["daily_pnl"] >= 0
    st.markdown(
        f"<div class='hero'><div class='label'>Your Portfolio</div>"
        f"<div class='big'>{money(d['equity'])}</div>"
        f"<div style='font-size:17px;font-weight:700;color:{'#00E676' if up else '#FF6E6E'}'>"
        f"{'▲' if up else '▼'} {signed(d['daily_pnl'])} today ({d['daily_pct']:+.2f}%) {dot(up)}</div></div>",
        unsafe_allow_html=True)

    # Equity chart with timeframe selector
    hist = d.get("equity_history", [])
    tf = st.radio("timeframe", ["1D", "1W", "1M", "3M", "ALL"], index=4,
                  horizontal=True, label_visibility="collapsed")
    if hist:
        n = {"1D": 2, "1W": 5, "1M": 21, "3M": 63, "ALL": len(hist)}[tf]
        seg = hist[-n:]
        ys = [p["equity"] for p in seg]
        line = GREEN if ys[-1] >= ys[0] else RED
        fig = go.Figure(go.Scatter(y=ys, mode="lines", line=dict(color=line, width=3),
                                   fill="tozeroy", fillcolor="rgba(0,184,67,.08)"))
        fig.update_layout(height=240, margin=dict(l=0, r=0, t=6, b=0),
                          xaxis=dict(visible=False),
                          yaxis=dict(visible=False), plot_bgcolor="#fff", paper_bgcolor="#fff")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # Four mini cards
    c1, c2, c3, c4 = st.columns(4)
    _mini(c1, "💵 Started", money(d["started"], cents=False), "your deposit")
    _mini(c2, "📈 Now Worth", money(d["equity"], cents=False), "right now")
    _mini(c3, "🎯 All Time", f"{signed(d['all_time_pnl'])}", f"{d['all_time_pct']:+.2f}% profit/loss",
          color=col_for(d["all_time_pnl"]))
    _mini(c4, "🏆 Win Rate", f"{d['win_rate']*100:.0f}%", "of trades are winners")

    st.markdown("### What You Own Right Now")
    if d["positions"]:
        for p in d["positions"]:
            _position_card(p)
    else:
        st.markdown(
            "<div class='card'><div class='pos-title'>🔍 No open trades right now</div>"
            "<p class='muted'>Your bot is scanning the markets every 5 minutes looking for the "
            "perfect setup. When it finds one that scores 70/100 or higher with 4-to-1 profit "
            "potential, it buys automatically.</p>"
            "<span class='pill'>Last scan: a few minutes ago</span>"
            "<span class='pill'>Next scan: soon</span></div>", unsafe_allow_html=True)

    if d.get("activity"):
        st.markdown("### Recent Activity")
        rows = "".join(f"<div class='feed'>{a['icon']} <b>{a['ago']}</b> — {a['text']}</div>"
                       for a in d["activity"])
        st.markdown(f"<div class='card'>{rows}</div>", unsafe_allow_html=True)


def _mini(col, label, value, sub, color=INK):
    col.markdown(f"<div class='card mini'><div class='l'>{label}</div>"
                 f"<div class='v' style='color:{color}'>{value}</div>"
                 f"<div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def _position_card(p):
    making = p["pnl"] >= 0
    tag = (f"<span class='tag' style='background:#e7f9ee;color:{GREEN}'>MAKING MONEY 🟢</span>"
           if making else f"<span class='tag' style='background:#fdeaec;color:{RED}'>DOWN A BIT 🔴</span>")
    prog = max(0, min(100, p["progress"] * 100))
    reasons = "".join(f"<div class='reason'>✅ {r}</div>" for r in p["reasons"])
    st.markdown(f"""
<div class='card'>
  <div class='row'><div class='pos-title'>{p['emoji']} {p['name']} ({p['symbol']})</div>{tag}</div>
  <p class='muted' style='margin:6px 0'>You own <b>{p['shares']}</b> shares ·
     paid <b>{money(p['paid'])}</b> · worth <b>{money(p['now']) if p['now'] else 'updating…'}</b></p>
  <div style='font-size:22px;font-weight:800;color:{col_for(p['pnl'])}'>
     {signed(p['pnl'])} ({p['pnl_pct']:+.1f}%) {dot(making)}</div>
  <p class='muted' style='margin-top:10px'>🎯 Target <b>{money(p['target'])}</b> (selling here for profit) &nbsp;·&nbsp;
     🛡️ Safety net <b>{money(p['stop'])}</b> (auto-sells if it drops here)</p>
  <div class='bar'><div class='fill' style='width:{prog}%'></div><div class='dot' style='left:{prog}%'></div></div>
  <div class='row muted'><span>bought {money(p['paid'])}</span>
     <span>{prog:.0f}% to target</span><span>target {money(p['target'])}</span></div>
  <p style='margin-top:12px;font-weight:700'>Bot bought this because:</p>
  {reasons}
  <span class='pill'>Confidence score: {p['score']:.0f}/100</span>
</div>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# PAGE 2 — All Your Trades
# --------------------------------------------------------------------------- #
def page_trades(d):
    trades = d["trades"]
    total = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    wr = len(wins) / len(trades) * 100 if trades else 0
    c1, c2, c3, c4 = st.columns(4)
    _mini(c1, "💰 Total Made/Lost", signed(total), "all trades", color=col_for(total))
    _mini(c2, "✅ Winning Trades", f"{len(wins)}", "made money")
    _mini(c3, "❌ Losing Trades", f"{len(losses)}", "lost money")
    _mini(c4, "📊 Win Rate", f"{wr:.0f}%", "of trades win")

    if trades:
        st.markdown(
            f"<div class='card'><p style='font-size:15px'>Your bot has made <b>{len(trades)}</b> "
            f"trades total. It won <b>{len(wins)}</b> and lost <b>{len(losses)}</b>. "
            f"Even with some losses, the winners were big enough that you're "
            f"<b style='color:{col_for(total)}'>{signed(total)}</b> overall. That's how the 4-to-1 "
            f"strategy works — small losses, big wins.</p></div>", unsafe_allow_html=True)
        for t in trades:
            _trade_card(t)
    else:
        st.markdown("<div class='card'><div class='pos-title'>📭 No completed trades yet</div>"
                    "<p class='muted'>Once the bot buys and sells something, it'll show up here "
                    "with how much you made or lost.</p></div>", unsafe_allow_html=True)


def _trade_card(t):
    win = t["win"]
    badge = (f"<span class='tag' style='background:#e7f9ee;color:{GREEN}'>✅ WIN</span>" if win
             else f"<span class='tag' style='background:#fdeaec;color:{RED}'>❌ LOSS</span>")
    res = "Profit" if win else "Loss"
    note = f"<p class='muted'>Why it lost: {t['note']}</p>" if (not win and t.get("note")) else ""
    bought = f"{t['bought_on']} at {money(t['bought_at']) if t['bought_at'] else '—'}"
    sold = f"{t['sold_on']} at {money(t['sold_at']) if t['sold_at'] else '—'}"
    st.markdown(f"""
<div class='card'>
  <div class='row'><div class='pos-title'>{t['emoji']} {t['name']} · Bought &amp; Sold</div>{badge}</div>
  <p class='muted' style='margin:6px 0'>Bought {bought} &nbsp;·&nbsp; Sold {sold}
     {('· held ' + str(t['days']) + ' days') if t['days'] else ''}</p>
  <div style='font-size:20px;font-weight:800;color:{col_for(t['pnl'])}'>
     {res}: {signed(t['pnl'])} ({t['pnl_pct']:+.1f}%) {dot(win)}</div>
  {note}
  {('<span class=pill>Bot confidence when bought: ' + str(int(t['score'])) + '/100</span>') if t.get('score') else ''}
</div>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# PAGE — Options (call/put bets, shown separately from stocks)
# --------------------------------------------------------------------------- #
def _opt_exp_pretty(iso: str) -> str:
    try:
        d = _dt.date.fromisoformat(iso)
        return f"{d.strftime('%b')} {d.day}, {d.year}"
    except (ValueError, TypeError):
        return iso or "—"


def page_options(d):
    opts = d.get("options", [])
    st.markdown("### 🎯 Your Options Bets")
    st.caption("Options are short-term bets on a stock's direction. A CALL profits "
               "if it goes UP, a PUT if it goes DOWN. The most you can lose is what "
               "you paid (the premium).")

    if not opts:
        st.markdown(
            "<div class='card'><div class='pos-title'>📭 No options bets right now</div>"
            "<p class='muted'>When the bot gets a high-confidence signal (70+), it can "
            "buy a call or put instead of the stock — aiming to double its money while "
            "risking only the premium. Active option bets show up here.</p></div>",
            unsafe_allow_html=True)
        return

    total_paid = sum(o["paid"] for o in opts)
    total_value = sum(o["value"] for o in opts)
    total_pnl = total_value - total_paid
    c1, c2, c3 = st.columns(3)
    _mini(c1, "💸 Premium Paid", money(total_paid, cents=False), f"{len(opts)} bet(s)")
    _mini(c2, "💵 Worth Now", money(total_value, cents=False), "current value")
    _mini(c3, "🎯 Profit/Loss", signed(total_pnl), "on option bets", color=col_for(total_pnl))

    for o in opts:
        _option_card(o)


def _option_card(o):
    making = o["pnl"] >= 0
    is_call = o["type"] == "call"
    kind_tag = (f"<span class='tag' style='background:#e7f9ee;color:{GREEN}'>📈 CALL (betting UP)</span>"
                if is_call else
                f"<span class='tag' style='background:#fdeaec;color:{RED}'>📉 PUT (betting DOWN)</span>")
    pl_tag = (f"<span class='tag' style='background:#e7f9ee;color:{GREEN}'>WINNING 🟢</span>"
              if making else f"<span class='tag' style='background:#fdeaec;color:{RED}'>DOWN 🔴</span>")
    # progress toward the +100% target (premium doubling), -50% stop at the left.
    paid_prem = o["premium_paid"] or 1e-9
    ratio = o["current_premium"] / paid_prem      # 0.5=stop, 1=flat, 2=target
    prog = max(0, min(100, (ratio - 0.5) / 1.5 * 100))
    st.markdown(f"""
<div class='card'>
  <div class='row'><div class='pos-title'>{o['emoji']} {o['description'] or o['underlying']}</div>{kind_tag}</div>
  <p class='muted' style='margin:6px 0'>{o['contracts']} contract(s) ·
     ${o['strike']:,.0f} strike · expires {_opt_exp_pretty(o['expiration'])}</p>
  <div style='font-size:22px;font-weight:800;color:{col_for(o['pnl'])}'>
     {signed(o['pnl'])} ({o['pnl_pct']:+.0f}%) {dot(making)} &nbsp;{pl_tag}</div>
  <p class='muted' style='margin:8px 0'>Paid <b>{money(o['paid'])}</b> ·
     worth now <b>{money(o['value'])}</b> &nbsp;·&nbsp; premium
     <b>${o['premium_paid']:.2f}</b> → <b>${o['current_premium']:.2f}</b>/share</p>
  <div class='bar'><div class='fill' style='width:{prog}%'></div><div class='dot' style='left:{prog}%'></div></div>
  <div class='row muted'><span>🛡️ stop ${o['stop_premium']:.2f} (-50%)</span>
     <span>now</span><span>🎯 double ${o['target_premium']:.2f} (+100%)</span></div>
  <span class='pill'>Confidence when bought: {o['score']:.0f}/100</span>
  <span class='pill'>Max loss: {money(o['paid'], cents=False)} (the premium)</span>
</div>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# PAGE 3 — Watching Now
# --------------------------------------------------------------------------- #
def page_watching(d):
    st.markdown("### 👁️ Markets your bot watches")
    st.caption("🟢 Green = bot likes it · 🔴 Red = bot avoiding it · ⚪ Grey = neutral/waiting")
    items = d["watching"]
    flt = st.radio("filter", ["All", "Stocks", "Crypto", "Bot Likes (70+)", "Bot Owns"],
                   horizontal=True, label_visibility="collapsed")
    if flt == "Stocks":
        items = [w for w in items if "/" not in w["symbol"]]
    elif flt == "Crypto":
        items = [w for w in items if "/" in w["symbol"]]
    elif flt == "Bot Likes (70+)":
        items = [w for w in items if w["score"] >= 70]
    elif flt == "Bot Owns":
        items = [w for w in items if w["status"] == "OWNED"]

    if not items:
        st.info("Nothing matches that filter right now.")
        return

    # Pagination — 25 per page.
    per_page = 25
    pages = max(1, (len(items) + per_page - 1) // per_page)
    pg = 1
    if pages > 1:
        pg = st.number_input(f"Page (1–{pages}) · showing {len(items)} markets",
                             min_value=1, max_value=pages, value=1, step=1)
    page_items = items[(pg - 1) * per_page: pg * per_page]
    cols = st.columns(3)
    for i, w in enumerate(page_items):
        with cols[i % 3]:
            _watch_card(w)


def _watch_card(w):
    scored = w.get("scored", True)
    view_emoji = {"BULLISH": "👍", "NEUTRAL": "😐", "BEARISH": "👎"}[w["view"]]
    view_word = {"BULLISH": "Bot likes it", "NEUTRAL": "Bot waiting", "BEARISH": "Bot avoiding"}[w["view"]]
    status_map = {"OWNED": "✅ OWNED", "WATCHING": "👀 WATCHING", "AVOIDING": "🚫 AVOIDING"}
    chg = w.get("chg", 0)
    price = f"{money(w['price'])} {'▲' if chg>=0 else '▼'} {chg:+.1f}%" if w.get("price") else "live"
    reasons = "".join(f"<div class='reason' style='font-size:13px'>• {r}</div>" for r in w["reasons"])
    if scored:
        bar_col = GREEN if w["score"] >= 70 else (RED if w["score"] < 50 else GREY)
        strength = "Strong" if w["score"] >= 70 else "Weak" if w["score"] < 50 else "Moderate"
        conf = (f"<div class='muted' style='font-size:13px'>Confidence: {w['score']:.0f}/100</div>"
                f"<div class='bar' style='height:8px'><div class='fill' "
                f"style='width:{w['score']}%;background:{bar_col}'></div></div>"
                f"<div class='muted' style='font-size:12px'>{strength}</div>")
    else:
        conf = ("<div class='muted' style='font-size:13px'>Confidence: — (awaiting scan)</div>"
                f"<div class='bar' style='height:8px'><div class='fill' "
                f"style='width:50%;background:{GREY}'></div></div>")
    st.markdown(f"""
<div class='card' style='min-height:230px'>
  <div class='pos-title' style='font-size:17px'>{w['emoji']} {w['name']}</div>
  <div class='muted' style='font-size:13px'>{w['symbol']} · {price}</div>
  <div style='margin:8px 0;font-weight:800'>{view_emoji} {view_word}</div>
  {conf}
  <div style='margin-top:8px'>{reasons}</div>
  {_research_block(w.get('research'))}
  <div style='margin-top:8px;font-weight:700;font-size:13px'>{status_map[w['status']]}</div>
</div>""", unsafe_allow_html=True)


_BADGE = {"green": GREEN, "blue": "#2563EB", "grey": GREY, "orange": "#F59E0B", "red": RED}


def _research_block(r) -> str:
    """Research summary rows inside a watch card (insider/analyst/news/social/earnings)."""
    if not r:
        return ""
    pts = int(r.get("points", 0))
    pts_col = GREEN if pts > 0 else (RED if pts < 0 else GREY)
    rating = r.get("analyst_rating", "N/A")
    badge = _BADGE.get(r.get("analyst_color", "grey"), GREY)
    target = float(r.get("target", 0) or 0)
    up = float(r.get("upside_pct", 0) or 0)
    tgt = (f" · 🎯 ${target:.0f} ({up:+.0f}%)" if target else "")
    head = (r.get("news_headline", "") or "")[:46]
    ed = r.get("earnings_days", -999)
    earn = (f"📅 Earnings {ed}d" if isinstance(ed, (int, float)) and ed >= 0 else "📅 Earnings —")
    bull = float(r.get("bull_pct", 0) or 0)
    social = (f"💬 {bull:.0f}% bull" if r.get("social_status") == "ok" else "💬 —")
    return (
        f"<div style='margin-top:8px;border-top:1px solid #f1f3f7;padding-top:6px;font-size:12px'>"
        f"<div><span style='font-weight:800;color:{pts_col}'>Research {pts:+d}</span></div>"
        f"<div>🏦 {r.get('insider_emoji','⚪')} {r.get('insider_summary','—')[:34]}</div>"
        f"<div>👔 <span style='font-weight:700;color:{badge}'>{rating}</span> "
        f"({r.get('analyst_n',0)}){tgt}</div>"
        f"<div>📰 {r.get('news_emoji','⚪')} {head}</div>"
        f"<div>{social} · {earn}</div>"
        f"</div>")


# --------------------------------------------------------------------------- #
# PAGE 4 — How Is It Doing
# --------------------------------------------------------------------------- #
def page_performance(d):
    st.markdown("### 📊 How is your bot doing?")
    st.caption("The goal is a line that goes up and to the right.")
    perf = d["performance"]
    hist = d.get("equity_history", [])
    if hist:
        ys = [p["equity"] for p in hist]
        base = ys[0]
        fig = go.Figure(go.Scatter(y=ys, mode="lines", line=dict(color=GREEN, width=3)))
        fig.add_hline(y=base, line_dash="dot", line_color=GREY,
                      annotation_text="you started here", annotation_position="top left")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="Your money", plot_bgcolor="#fff", paper_bgcolor="#fff")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    _explain("📈 How consistent is it?", f"Consistency score: {perf['sharpe']:.2f}",
             "Above 1.0 means the bot makes more than it risks. Top funds aim for 2+. "
             f"Yours is {perf['sharpe']:.2f} — " +
             ("pretty good!" if perf["sharpe"] >= 1 else "still building a track record."))
    _explain("📉 Worst rough patch", f"Biggest dip: {perf['max_drawdown_pct']:.2f}%",
             "The most the bot ever dropped from a high point before recovering. Lower is better. "
             f"That's about {money(perf['max_drawdown_dollar'], cents=False)} on a $100k account "
             "at its worst, before bouncing back.")
    _explain("🎯 Does the strategy make sense?",
             f"Average win {signed(perf['avg_winner'])} · average loss {signed(perf['avg_loser'])}",
             "When the bot wins it makes more than it loses when it's wrong — so even winning only "
             "part of the time, you come out ahead. That's the whole idea behind 4-to-1.")

    if perf["monthly"]:
        st.markdown("#### This month, day by day")
        cells = "".join(
            f"<span style='display:inline-block;width:64px;text-align:center;margin:3px;"
            f"padding:8px 0;border-radius:10px;font-size:12px;font-weight:700;"
            f"background:{'#e7f9ee' if v>0 else ('#fdeaec' if v<0 else '#f4f6fa')};"
            f"color:{GREEN if v>0 else (RED if v<0 else GREY)}'>"
            f"{day}<br>{signed(v) if v else '$0'}</span>"
            for day, v in perf["monthly"])
        st.markdown(f"<div class='card'>{cells}</div>", unsafe_allow_html=True)


def _explain(title, big, plain):
    st.markdown(f"<div class='card'><div style='font-weight:800;font-size:16px'>{title}</div>"
                f"<div style='font-size:18px;font-weight:700;margin:4px 0'>{big}</div>"
                f"<div class='muted'>{plain}</div></div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# PAGE 5 — Bot Status
# --------------------------------------------------------------------------- #
def page_bot(d):
    b = d["bot"]
    st.markdown(
        f"<div class='card'><div class='pos-title'>🤖 Your Trading Bot</div>"
        f"<div style='font-size:18px;font-weight:800;color:{GREEN if b['running'] else RED};margin:6px 0'>"
        f"Status: {'🟢 RUNNING AND HEALTHY' if b['running'] else '🔴 STOPPED'}</div>"
        f"<p class='muted'>Last checked markets: {b['last_scan_min']} min ago · "
        f"next check: {b['next_scan_min']} min · running for {b['uptime']}</p>"
        f"<p style='font-weight:700;margin-top:8px'>Today's activity</p>"
        f"<div class='muted'>• Scanned markets {b['scans_today']} times<br>"
        f"• Looked at {b['setups_looked']} potential setups<br>"
        f"• Found {b['trades_taken']} trade(s) worth taking<br>"
        f"• Rejected {b['setups_rejected']} (not good enough)</div></div>", unsafe_allow_html=True)

    used, limit = b["daily_loss_used"], b["daily_loss_limit"]
    pct = min(100, used / limit * 100) if limit else 0
    st.markdown(
        f"<div class='card'><div style='font-weight:800'>🛡️ Daily loss protection</div>"
        f"<p class='muted'>Used {money(used, cents=False)} of {money(limit, cents=False)} allowed today</p>"
        f"<div class='bar'><div class='fill' style='width:{max(2,pct):.0f}%;"
        f"background:{GREEN if pct<70 else RED}'></div></div>"
        f"<div class='muted'>{'✅ Safe zone' if pct<70 else '⚠️ Getting close'}</div>"
        f"<p style='margin-top:10px'>📱 Phone alerts: {dot(b['telegram'])} &nbsp; "
        f"🏦 Broker: {dot(b['broker'])} &nbsp; 💾 Saving trades: {dot(b['logging'])}</p></div>",
        unsafe_allow_html=True)

    # Research data-source health
    ss = b.get("source_status", {})
    if ss:
        labels = {"insider": "🏦 Insider (SEC)", "analyst": "👔 Analysts", "news": "📰 News",
                  "social": "💬 StockTwits", "earnings": "📅 Earnings"}
        rows = ""
        for key, lab in labels.items():
            stt = ss.get(key, "—")
            dt = "🟢" if stt == "ok" else ("⚪" if stt in ("insufficient", "n/a", "—") else "🔴")
            rows += f"<div class='row'><span>{lab}</span><span>{dt} {stt}</span></div>"
        st.markdown(f"<div class='card'><div style='font-weight:800'>🔬 Research data sources</div>"
                    f"<div style='margin-top:6px;font-size:14px'>{rows}</div>"
                    f"<p class='muted' style='font-size:12px;margin-top:6px'>A failed source "
                    f"contributes 0 points — the bot keeps trading on its other signals.</p></div>",
                    unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    if b["running"]:
        if c1.button("⛔ STOP ALL TRADING", type="primary", width="stretch"):
            get_store().request_halt("manual STOP from dashboard")
            st.success("Stop requested — the bot will close everything on its next check.")
    else:
        if c1.button("▶️ RESUME TRADING", width="stretch"):
            get_store().clear_halt()
            st.success("Resume requested.")

    st.markdown(
        f"<div class='card'><div style='font-weight:800'>⚙️ How picky is your bot?</div>"
        f"<p class='muted' style='margin-top:8px'>Minimum confidence to trade: "
        f"<b>{b['min_score']:.0f}/100</b> — only trades when it's at least that sure.</p>"
        f"<p class='muted'>Minimum profit potential: <b>{b['rr']:.0f}-to-1</b> — must see "
        f"${b['rr']:.0f} of potential for every $1 risked.</p>"
        f"<p class='muted'>Max risk per trade: <b>{b['risk_pct']:.0f}%</b> of the account — "
        f"never risks more than {money(d['started']*b['risk_pct']/100, cents=False)} on one trade.</p></div>",
        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
PAGES = {"💰 Home": page_home, "📋 Trades": page_trades, "🎯 Options": page_options,
         "👁️ Watching": page_watching, "📊 Performance": page_performance,
         "🤖 Bot": page_bot}


def _nav():
    try:
        from streamlit_option_menu import option_menu
        return option_menu(None, list(PAGES.keys()), orientation="horizontal",
                           icons=[""] * len(PAGES),
                           styles={"container": {"padding": "4px", "background-color": "#f4f6fa"},
                                   "nav-link-selected": {"background-color": "#0E1117"}})
    except Exception:
        return st.radio("nav", list(PAGES.keys()), horizontal=True, label_visibility="collapsed")


def main():
    try:
        d, mode, default = load_data()
    except Exception:
        st.error("😅 Couldn't load your data right now. Try the Refresh button in a moment.")
        return

    top_bar(d, mode)
    page = st.session_state.get("force_page") or _nav() or list(PAGES.keys())[0]

    # Controls row: refresh + data mode toggle + auto-refresh.
    cc1, cc2, cc3, cc4 = st.columns([1, 1, 1.4, 2.6])
    if cc1.button("🔄 Refresh"):
        st.rerun()
    new_mode = cc2.selectbox("Data", ["Demo", "Live"],
                             index=0 if mode == "Demo" else 1, label_visibility="collapsed")
    if new_mode != mode:
        st.session_state["data_mode"] = new_mode
        st.rerun()
    # Auto-refresh: re-reads the agent's state file on a timer (default on, 30s).
    auto = cc3.toggle("Auto-refresh", value=st.session_state.get("auto_refresh", True),
                      key="auto_refresh", help="Re-pull the bot's latest state on a timer.")
    if auto:
        st_autorefresh(interval=30_000, key="auto_refresh_tick")
        cc4.caption("🟢 Live updating every 30s")
    else:
        cc4.caption("⏸️ Paused — hit Refresh to update")

    try:
        PAGES[page](d)
    except Exception:
        st.error("😅 Something hiccuped showing this page. Hit Refresh — your money and bot "
                 "are safe; this is just the display.")

    st.divider()
    friendly_glossary()
    st.caption("This is a paper-trading bot for learning. Not financial advice.")


main()
