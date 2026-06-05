"""Layer 1 — Streamlit dashboard.

Five pages (Overview / Positions / Signal Scanner / Performance / Backtest) that
read the agent's shared state (``StateStore``) plus live Alpaca account data and
can run backtests on demand. Live pages auto-refresh every 30 s.

Run:  ``streamlit run src/monitoring/dashboard_app.py``  →  http://localhost:8501

The dashboard runs in its own process; it talks to the agent only through the
shared state file, so the HALT button writes a control flag the agent polls.
"""

from __future__ import annotations

import asyncio
import time

# Streamlit reruns this script in a worker thread that may not have an asyncio
# event loop (or has a closed one), which breaks libraries that use asyncio
# (Alpaca/yfinance) with "RuntimeError: Event loop is closed" on Python 3.14.
# Ensure every run has a live loop for the current thread.
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings
from src.monitoring.state_store import StateStore

st.set_page_config(page_title="Trading Agent", layout="wide", page_icon="📈")
REFRESH_SECONDS = 30


@st.cache_resource
def get_store() -> StateStore:
    return StateStore(log_dir=settings.LOG_DIR)


@st.cache_resource
def get_broker():
    from src.execution.broker import Broker
    return Broker(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY, paper=settings.PAPER)


def _equity_df(state: dict) -> pd.DataFrame:
    hist = state.get("equity_history", [])
    if not hist:
        return pd.DataFrame(columns=["t", "equity"])
    df = pd.DataFrame(hist)
    df["t"] = pd.to_datetime(df["t"])
    return df


def _metric_card(col, label, value, delta=None):
    col.metric(label, value, delta)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview(state):
    st.header("Live Overview")
    if not state:
        st.warning("No agent state yet — start the agent (`python main.py`) to populate.")
    equity = state.get("equity", 0.0)
    daily = state.get("daily_pnl", 0.0)
    weekly = state.get("weekly_pnl", 0.0)
    regime = state.get("risk_state", "unknown")
    halted = state.get("halted", False)

    c1, c2, c3, c4 = st.columns(4)
    _metric_card(c1, "Equity", f"${equity:,.2f}")
    _metric_card(c2, "Daily PnL", f"${daily:,.2f}", f"{daily:+,.2f}")
    _metric_card(c3, "Weekly PnL", f"${weekly:,.2f}", f"{weekly:+,.2f}")
    badge = "🟢 ACTIVE" if not halted else "🔴 HALTED"
    c4.markdown(f"### {badge}")
    c4.markdown(f"**Regime:** `{regime}`")

    # Equity chart
    df = _equity_df(state)
    if not df.empty:
        fig = go.Figure(go.Scatter(x=df["t"], y=df["equity"], mode="lines",
                                   line=dict(color="#2ca02c")))
        fig.update_layout(title="Account Equity", height=380, margin=dict(t=40, b=20))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Equity history accumulates as the agent runs.")

    st.divider()
    st.subheader("Kill Switch")
    if halted:
        st.error("Trading is HALTED.")
    else:
        if st.button("🛑  HALT ALL TRADING", type="primary", width="stretch"):
            get_store().request_halt("manual HALT from Streamlit dashboard")
            st.error("HALT requested — the agent will flatten the book on its next loop.")


def page_positions(state):
    st.header("Positions")
    st.subheader("Open positions")
    try:
        positions = get_broker().get_positions()
        rows = []
        for p in positions:
            rows.append({
                "Symbol": p.symbol, "Side": getattr(p, "side", ""),
                "Qty": p.qty, "Entry": float(p.avg_entry_price),
                "Current": float(getattr(p, "current_price", 0) or 0),
                "Unreal PnL": float(getattr(p, "unrealized_pl", 0) or 0),
                "Unreal %": float(getattr(p, "unrealized_plpc", 0) or 0) * 100,
            })
        st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame({"info": ["no open positions"]}),
                     width="stretch")
    except Exception as e:
        st.error(f"Could not load positions from Alpaca: {e}")

    st.subheader("Closed today")
    closed = state.get("closed_today", [])
    st.dataframe(pd.DataFrame(closed) if closed else pd.DataFrame({"info": ["none yet"]}),
                 width="stretch")


def _score_color(score):
    return "#d62728" if score < 50 else "#ff7f0e" if score < 70 else "#2ca02c"


def page_scanner(state):
    st.header("Signal Scanner")
    scores = state.get("scores", [])
    if not scores:
        st.info("No scores yet — the agent writes them each scan. You can also scan one "
                "symbol on demand below.")
    else:
        # Heatmap-style colored bars.
        fig = go.Figure()
        syms = [s["symbol"] for s in scores]
        vals = [s["score"] for s in scores]
        colors = [_score_color(v) for v in vals]
        fig.add_bar(x=vals, y=syms, orientation="h", marker_color=colors,
                    text=[f"{v:.0f}" for v in vals], textposition="outside")
        fig.update_layout(title="Current scores (0-100)", height=420, xaxis_range=[0, 100],
                          margin=dict(t=40))
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Symbol deep-dive")
    sym = st.selectbox("Symbol", settings.load_watchlist())
    if st.button("Analyze", width="stretch"):
        with st.spinner(f"Analyzing {sym}…"):
            _analyze_symbol(sym)


def _analyze_symbol(sym):
    from src.data.feed import MarketFeed
    from src.signals.regime import RegimeDetector
    from src.signals.technical import TechnicalAnalysis
    feed = MarketFeed(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY,
                      stock_feed=settings.STOCK_DATA_FEED)
    df = feed.get_bars(sym, "1Day", settings.LOOKBACK_BARS)
    if df.empty:
        st.error("No data."); return
    tech = TechnicalAnalysis().analyze(df)
    regime = RegimeDetector().detect(df)
    if tech is None:
        st.error("Analysis failed."); return
    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"${tech.values.get('price', 0):,.2f}")
    arrow = "↑" if tech.trend_bias == "long" else "↓" if tech.trend_bias == "short" else "→"
    c2.metric("Trend bias", f"{arrow} {tech.trend_bias}")
    c3.metric("Regime", regime.strategy)
    st.markdown(f"**Regime label:** `{regime.label}`")
    firing = [k for k, v in tech.signals.items() if v is True][:3]
    st.markdown("**Top signals firing:** " + (", ".join(firing) if firing else "none"))
    with st.expander("Full indicator values"):
        st.json({k: round(v, 4) if isinstance(v, float) else v
                 for k, v in tech.values.items()})


def page_performance(state):
    st.header("Performance")
    df = _equity_df(state)
    if df.empty:
        st.info("Performance metrics accumulate as the agent runs (equity history).")
        return
    df = df.set_index("t")
    fig = go.Figure(go.Scatter(x=df.index, y=df["equity"], mode="lines"))
    fig.update_layout(title="Equity curve (since agent start)", height=360)
    st.plotly_chart(fig, width="stretch")

    # Drawdown
    peak = df["equity"].cummax()
    dd = (df["equity"] - peak) / peak * 100
    ddfig = go.Figure(go.Scatter(x=df.index, y=dd, fill="tozeroy", line=dict(color="#d62728")))
    ddfig.update_layout(title="Drawdown (%)", height=260)
    st.plotly_chart(ddfig, width="stretch")

    closed = pd.DataFrame(state.get("closed_today", []))
    c1, c2, c3 = st.columns(3)
    if not closed.empty and "pnl" in closed:
        wins = (closed["pnl"] > 0).sum()
        c1.metric("Win rate (today)", f"{wins/len(closed)*100:.0f}%")
        c2.metric("Best trade", f"${closed['pnl'].max():,.2f}")
        c3.metric("Worst trade", f"${closed['pnl'].min():,.2f}")
    c1.metric("Max drawdown", f"{dd.min():.2f}%")
    st.caption("Note: deeper metrics (Sharpe/Sortino, monthly heatmap) populate from "
               "live history; for rigorous stats use the validation harness on backtests.")


def page_backtest():
    st.header("Backtest")
    c1, c2, c3 = st.columns(3)
    sym = c1.selectbox("Symbol", settings.load_watchlist())
    period = c2.selectbox("Period", ["1y", "2y", "5y"], index=1)
    min_score = c3.slider("Min score", 50, 90, 70, 5)
    if st.button("Run backtest", type="primary"):
        with st.spinner(f"Backtesting {sym} ({period})…"):
            from src.backtest.engine import Backtester
            r = Backtester().run_pipeline(sym, period=period, interval="1d", min_score=min_score)
        if r is None:
            st.error("Not enough data / no result.")
            return
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades", r.num_trades)
        m2.metric("Return", f"{r.total_return*100:.2f}%")
        m3.metric("Sharpe", f"{r.sharpe:.2f}")
        m4.metric("Max DD", f"{r.max_drawdown*100:.2f}%")
        if r.equity_curve is not None and len(r.equity_curve):
            fig = go.Figure(go.Scatter(x=r.equity_curve.index, y=r.equity_curve.values, mode="lines"))
            fig.update_layout(title=f"{sym} equity curve", height=360)
            st.plotly_chart(fig, width="stretch")
        if r.trades is not None and not r.trades.empty:
            st.dataframe(r.trades, width="stretch")


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def main():
    store = get_store()
    state = store.read_state()

    st.sidebar.title("📈 Trading Agent")
    fresh = store.is_fresh(max_age_seconds=180)
    st.sidebar.markdown(("🟢 Agent live" if fresh else "⚪ Agent idle/offline"))
    page = st.sidebar.radio("Page", ["Live Overview", "Positions", "Signal Scanner",
                                     "Performance", "Backtest"])
    st.sidebar.caption(f"Mode: {'PAPER' if settings.PAPER else 'LIVE'}")

    if page == "Live Overview":
        page_overview(state)
    elif page == "Positions":
        page_positions(state)
    elif page == "Signal Scanner":
        page_scanner(state)
    elif page == "Performance":
        page_performance(state)
    elif page == "Backtest":
        page_backtest()

    # Auto-refresh only the live pages (not while running a backtest/analysis).
    if page in ("Live Overview", "Positions", "Signal Scanner"):
        time.sleep(REFRESH_SECONDS)
        st.rerun()


main()
