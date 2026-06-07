"""Research-layer demo / report.

  python scripts/research_report.py            # AAPL deep report + scorer + top-5 scan
  python scripts/research_report.py NVDA       # deep report for another symbol

Shows: (1) all four research layers for the symbol, (2) the master score with
vs without research, and (3) a scan ranking the watchlist's top names by total
score (technical + research).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from config import settings
from src.data.feed import MarketFeed, is_crypto
from src.signals.quant import QuantAnalysis
from src.signals.regime import RegimeDetector
from src.signals.research import ResearchEngine
from src.signals.rr_filter import RRFilter
from src.signals.scorer import MasterScorer
from src.signals.strategy import Signal
from src.signals.technical import TechnicalAnalysis

BAR = "=" * 70


def deep_report(sym, engine):
    print(BAR); print(f"RESEARCH REPORT — {sym}"); print(BAR)
    r = engine.analyze(sym)
    ins, an, nw, so = r.insider, r.analyst, r.news, r.social
    print("\n🏦 PRIORITY 1 — INSIDER ACTIVITY (SEC Form 4)")
    if ins:
        print(f"   {ins.emoji} {ins.summary}  ->  {ins.points:+d} pts   [{ins.status}]")
        print(f"   buys={ins.buys} sells={ins.sells} csuite_buys={ins.csuite_buys} "
              f"multiple_buyers={ins.multiple_buyers} large_buy={ins.large_buy}")
    print("\n👔 PRIORITY 2 — ANALYST RATINGS (yfinance)")
    if an:
        print(f"   {an.rating} | {an.n_analysts} analysts | target ${an.target:.2f} "
              f"({an.upside_pct:+.1f}%)  ->  {an.points:+d} pts   [{an.status}]")
        print(f"   never_short={an.never_short}")
    print("\n📰 PRIORITY 3 — NEWS SENTIMENT (finviz + Yahoo)")
    if nw:
        print(f"   {nw.emoji} {nw.label}  ->  {nw.points:+d} pts   [{nw.status}]  "
              f"({nw.headlines_analyzed} headlines, raw {nw.raw_score:+d})")
        print(f"   top: {nw.top_headline[:74]}")
        print(f"   very_negative_recent={nw.very_negative_recent}")
    print("\n💬 PRIORITY 4 — SOCIAL SENTIMENT (StockTwits)")
    if so:
        print(f"   {so.summary}  ->  {so.points:+d} pts   [{so.status}]  "
              f"high_volatility={so.high_volatility}")
    print("\n📅 EARNINGS")
    print(f"   {r.earnings.label}  (within7={r.earnings.within7} within3={r.earnings.within3})")
    print(f"\n{'-'*70}")
    print(f"   RESEARCH TOTAL: {r.total_points:+d} pts (clamped +/-25)")
    print(f"   vetoes: block_trade={r.block_trade} block_long={r.block_long} "
          f"block_short={r.block_short}  size_factor={r.size_factor}")
    return r


def scorer_view(sym, engine, feed, tech, quant, regime_d, scorer, rr):
    print("\n" + BAR); print(f"SCORER WITH RESEARCH — {sym}"); print(BAR)
    df = feed.get_bars(sym, "1Day", settings.LOOKBACK_BARS)
    spy = feed.get_bars(settings.MARKET_PROXY, "1Day", settings.LOOKBACK_BARS)
    t = tech.analyze(df)
    if t is None or t.trend_bias == "neutral":
        print("   no directional technical signal right now"); return
    side = t.trend_bias
    q = quant.analyze(df, market_df=spy if not spy.empty else None)
    reg = regime_d.detect(df, spy_df=spy if not spy.empty else None)
    sig = Signal(sym, side, float(df["close"].iloc[-1]), t.values.get("rsi14") or 50.0,
                 df.index[-1], t.trend_bias)
    plan = rr.evaluate(sig, df)
    res = engine.analyze(sym)
    base = scorer.score(sym, side, technical=t, quant=q, regime=reg, plan=plan)
    full = scorer.score(sym, side, technical=t, quant=q, regime=reg, plan=plan, research=res)
    print(f"   side: {side.upper()}")
    print(f"   technical-only total: {base.total:.1f}")
    print(f"   + research {res.total_points:+d}  ->  TOTAL {full.total:.1f}/100   "
          f"(gate {settings.MIN_SCORE:.0f} -> {'PASS' if full.passed else 'blocked'})")
    print(f"   breakdown: {full.breakdown}")


def scan_top5(engine, feed, tech, quant, regime_d, scorer, rr):
    print("\n" + BAR); print("SCAN — TOP SYMBOLS BY TOTAL SCORE (technical + research)"); print(BAR)
    universe = [s for s in settings.load_watchlist() if not is_crypto(s)]
    bars = feed.get_bars_batch(universe, "1Day", settings.LOOKBACK_BARS)
    spy = feed.get_bars(settings.MARKET_PROXY, "1Day", settings.LOOKBACK_BARS)
    # cheap prerank, then research only on the technical top 12 (cached, polite)
    pre = []
    for s in universe:
        df = bars.get(s)
        if df is None or df.empty or len(df) < 60:
            continue
        t = tech.analyze(df)
        if t is None or t.trend_bias == "neutral":
            continue
        pre.append((scorer.prerank_score(s, t.trend_bias, t), s, t, df))
    pre.sort(reverse=True, key=lambda x: x[0])
    rows = []
    for _, s, t, df in pre[:12]:
        side = t.trend_bias
        q = quant.analyze(df, market_df=spy if not spy.empty else None)
        reg = regime_d.detect(df, spy_df=spy if not spy.empty else None)
        sig = Signal(s, side, float(df["close"].iloc[-1]), t.values.get("rsi14") or 50.0,
                     df.index[-1], t.trend_bias)
        plan = rr.evaluate(sig, df)
        res = engine.analyze(s)
        sc = scorer.score(s, side, technical=t, quant=q, regime=reg, plan=plan, research=res)
        base = sc.total - res.total_points
        ok, why = res.allows(side)
        rows.append((sc.total, s, side, base, res, ok, why))
    rows.sort(reverse=True, key=lambda x: x[0])
    print(f"\n{'SYM':6} {'SIDE':5} {'TECH':>5} {'RSCH':>5} {'TOTAL':>6}  RESEARCH SUMMARY")
    for total, s, side, base, res, ok, why in rows[:5]:
        arrow = "↑" if side == "long" else "↓"
        veto = "" if ok else f"  ⛔ {why}"
        ins = res.insider.emoji if res.insider else "⚪"
        an = res.analyst.rating if res.analyst else "N/A"
        print(f"{s:6} {arrow}{side[:4]:4} {base:5.1f} {res.total_points:+5d} {total:6.1f}  "
              f"🏦{ins} 👔{an} 📰{res.news.emoji if res.news else '⚪'} "
              f"💬{res.social.bull_pct:.0f}%{veto}")


def main():
    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    k, s = settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY
    feed = MarketFeed(k, s, stock_feed=settings.STOCK_DATA_FEED, cache_ttl=240)
    tech = TechnicalAnalysis(); quant = QuantAnalysis(); regime_d = RegimeDetector()
    scorer = MasterScorer(min_score=settings.MIN_SCORE, rr_target=settings.RR_RATIO)
    rr = RRFilter(rr_ratio=settings.RR_RATIO, atr_period=settings.ATR_PERIOD,
                  atr_multiplier=settings.ATR_MULTIPLIER, swing_lookback=settings.SWING_LOOKBACK,
                  path_veto=settings.RR_PATH_VETO)
    engine = ResearchEngine(enabled=True)

    deep_report(sym, engine)
    scorer_view(sym, engine, feed, tech, quant, regime_d, scorer, rr)
    scan_top5(engine, feed, tech, quant, regime_d, scorer, rr)


if __name__ == "__main__":
    main()
