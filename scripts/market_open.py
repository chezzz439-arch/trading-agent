"""Pre-market briefing — run ~9:25am ET on trading days.

Builds a morning briefing and sends it to Telegram (and saves to
``logs/morning_brief_YYYYMMDD.txt``):

* market regime (SPY) + VIX level/trend
* index-futures bias (ES=F, NQ=F)
* pre-market movers across the watchlist
* a basic overnight-news-sentiment proxy per symbol
* top 3 symbols to watch (with current scores) and symbols to avoid
* a volatility-based recommended position size for the day

Run:  python scripts/market_open.py    (or schedule via cron / the schedule skill)

Note: news sentiment is a crude keyword proxy over yfinance headlines, not a
real NLP model — treat it as a hint, not a signal.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

from config import settings

_POS_WORDS = {"beat", "beats", "surge", "soar", "jump", "rally", "upgrade", "record",
              "strong", "gain", "gains", "bullish", "rise", "rises", "top", "wins"}
_NEG_WORDS = {"miss", "misses", "plunge", "fall", "falls", "drop", "downgrade", "cut",
              "weak", "loss", "losses", "bearish", "slump", "lawsuit", "probe", "warn"}


def fetch_futures() -> dict:
    import yfinance as yf
    out = {}
    for name, ticker in (("ES (S&P)", "ES=F"), ("NQ (Nasdaq)", "NQ=F")):
        try:
            h = yf.download(ticker, period="2d", interval="1d", progress=False)["Close"].dropna()
            if len(h) >= 2:
                out[name] = float(h.iloc[-1] / h.iloc[-2] - 1) * 100
        except Exception:
            pass
    return out


def premarket_movers() -> list:
    import yfinance as yf
    moves = []
    for sym in settings.WATCHLIST:
        ticker = sym.replace("/", "-")
        try:
            h = yf.download(ticker, period="2d", interval="1d", progress=False,
                            prepost=True)["Close"].dropna()
            if len(h) >= 2:
                moves.append((sym, float(h.iloc[-1] / h.iloc[-2] - 1) * 100))
        except Exception:
            continue
    return sorted(moves, key=lambda t: t[1], reverse=True)


def news_sentiment(symbol: str) -> int:
    """Crude headline keyword sentiment: +1 bullish / -1 bearish / 0 neutral."""
    import yfinance as yf
    try:
        items = yf.Ticker(symbol.replace("/", "-")).news or []
        score = 0
        for it in items[:8]:
            title = (it.get("title") or it.get("content", {}).get("title", "")).lower()
            score += sum(w in title for w in _POS_WORDS)
            score -= sum(w in title for w in _NEG_WORDS)
        return (score > 0) - (score < 0)
    except Exception:
        return 0


def score_symbol(sym: str, feed, spy_df) -> tuple:
    """Lightweight score (technical+quant+regime+RR; ML/MTF neutral)."""
    from src.signals.technical import TechnicalAnalysis
    from src.signals.quant import QuantAnalysis
    from src.signals.regime import RegimeDetector
    from src.signals.rr_filter import RRFilter
    from src.signals.scorer import MasterScorer
    from src.signals.strategy import Signal
    try:
        df = feed.get_bars(sym, "1Day", settings.LOOKBACK_BARS)
        if df.empty or len(df) < 60:
            return sym, 0.0, "neutral"
        tech = TechnicalAnalysis().analyze(df)
        if tech is None or tech.trend_bias == "neutral":
            return sym, 0.0, "neutral"
        side = tech.trend_bias
        quant = QuantAnalysis().analyze(df, market_df=spy_df if not spy_df.empty else None)
        regime = RegimeDetector().detect(df, spy_df=spy_df if not spy_df.empty else None)
        sig = Signal(sym, side, float(df["close"].iloc[-1]), tech.values.get("rsi14") or 50,
                     df.index[-1], side)
        plan = RRFilter(swing_lookback=settings.SWING_LOOKBACK).evaluate(sig, df)
        sc = MasterScorer().score(sym, side, technical=tech, quant=quant, regime=regime,
                                  plan=plan)
        return sym, sc.total, side
    except Exception:
        return sym, 0.0, "neutral"


def recommended_size(vix, regime_vol) -> str:
    """Size by VIX when available, else by the regime's ATR-percentile volatility."""
    high = (vix is not None and vix > 28) or regime_vol == "extreme"
    elevated = (vix is not None and 22 <= vix <= 28) or regime_vol == "high"
    calm = (vix is not None and vix < 15) or (vix is None and regime_vol == "low")
    if high:
        return "0.5% per trade (high volatility — size down)"
    if elevated:
        return "0.75% per trade (elevated volatility)"
    if calm:
        return "up to 1.5% per trade (calm tape)"
    return "1% per trade (normal)"


def main() -> None:
    from src.data.feed import MarketFeed
    from src.signals.regime import RegimeDetector
    from src.signals.sentiment import fetch_vix
    from src.monitoring.telegram_bot import TelegramNotifier

    feed = MarketFeed(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY,
                      stock_feed=settings.STOCK_DATA_FEED)
    spy_df = feed.get_bars(settings.MARKET_PROXY, "1Day", settings.LOOKBACK_BARS)
    regime = RegimeDetector().detect(spy_df) if not spy_df.empty else None

    vix, vix_trend = fetch_vix()
    futures = fetch_futures()
    movers = premarket_movers()
    scored = sorted([score_symbol(s, feed, spy_df) for s in settings.WATCHLIST],
                    key=lambda t: t[1], reverse=True)

    top3 = [s for s in scored if s[1] > 0][:3]
    avoid = [s for s in scored if s[1] < 45][:3]
    reg_vol = regime.volatility if regime else "unknown"
    reg_label = regime.label if regime else "unknown"

    # --- Build briefing ---------------------------------------------------- #
    lines = [f"MORNING BRIEFING — {datetime.now(timezone.utc):%Y-%m-%d}", ""]
    lines.append(f"Market regime: {reg_label}")
    if vix is not None:
        lines.append(f"VIX: {vix:.1f} ({vix_trend})")
    else:
        lines.append(f"VIX: unavailable — using volatility regime '{reg_vol}' "
                     f"(ATR percentile)")
    if futures:
        lines.append("Futures: " + ", ".join(f"{k} {v:+.2f}%" for k, v in futures.items()))
    lines.append("")
    lines.append("Watch today:")
    for sym, sc, side in top3:
        sent = news_sentiment(sym)
        tag = "📈" if sent > 0 else "📉" if sent < 0 else "•"
        lines.append(f"  {tag} {sym} {side} — score {sc:.0f}/100")
    if not top3:
        lines.append("  (no symbol scored above 0 — stand aside)")
    lines.append("")
    lines.append("Avoid: " + (", ".join(f"{s} ({sc:.0f})" for s, sc, _ in avoid) or "none"))
    if movers:
        lines.append("")
        lines.append(f"Top mover: {movers[0][0]} {movers[0][1]:+.2f}%  |  "
                     f"Bottom: {movers[-1][0]} {movers[-1][1]:+.2f}%")
    lines.append("")
    lines.append(f"Recommended size: {recommended_size(vix, reg_vol)}")
    brief = "\n".join(lines)

    # --- Save + send ------------------------------------------------------- #
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    path = os.path.join(settings.LOG_DIR,
                        f"morning_brief_{datetime.now(timezone.utc):%Y%m%d}.txt")
    with open(path, "w") as f:
        f.write(brief + "\n")
    print(brief)
    print(f"\nSaved to {path}")

    notifier = TelegramNotifier()
    if notifier.enabled:
        notifier.send("*MORNING BRIEFING* ☀️\n" + brief)
        print("Sent to Telegram.")


if __name__ == "__main__":
    main()
