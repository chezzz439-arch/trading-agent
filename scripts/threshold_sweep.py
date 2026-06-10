"""Threshold sensitivity: how many tradeable signals fire at MIN_SCORE 60/65/70.

Replays the production pipeline scorer bar-by-bar (same path as
Backtester._pipeline_enter) with the score gate disabled, records the master
score for every bar that has a non-neutral bias AND a valid RR plan, then tallies
how many of those signals clear each threshold. This counts *signals*, not
trades — the live agent fires a signal whenever score >= MIN_SCORE regardless of
whether it's already in a position.
"""

from __future__ import annotations

import sys
from collections import defaultdict

from src.backtest.engine import Backtester
from src.data.feed import is_crypto
from src.signals.quant import QuantAnalysis
from src.signals.regime import RegimeDetector
from src.signals.scorer import MasterScorer
from src.signals.strategy import Signal
from src.signals.technical import TechnicalAnalysis

THRESHOLDS = (60.0, 65.0, 70.0)
WARMUP = 210


def sweep_symbol(symbol, period="2y", interval="1d", market_df=None):
    bt = Backtester()
    df = bt.load_data(symbol, period, interval)
    if df.empty or len(df) < WARMUP + 5:
        return None
    tech_a, quant_a, regime_d = TechnicalAnalysis(), QuantAnalysis(), RegimeDetector()
    # min_score=0 so score.passed never gates — we get the raw score every bar.
    scorer = MasterScorer(min_score=0.0, rr_target=bt.rr_filter.rr_ratio)
    scores = []
    for i in range(WARMUP, len(df)):
        window = df.iloc[: i + 1]
        tech = tech_a.analyze(window)
        if tech is None or tech.trend_bias == "neutral":
            continue
        side = tech.trend_bias
        mkt = market_df.loc[: window.index[-1]] if market_df is not None and not market_df.empty else None
        quant = quant_a.analyze(window, market_df=mkt)
        regime = regime_d.detect(window, vix=None, spy_df=mkt)
        sig = Signal(symbol, side, float(window["close"].iloc[-1]),
                     tech.values.get("rsi14") or 50.0, window.index[-1], side)
        plan = bt.rr_filter.evaluate(sig, window)
        if plan is None:
            continue  # no valid RR plan -> not a tradeable signal
        sc = scorer.score(symbol, side, technical=tech, quant=quant, regime=regime,
                          ml=None, mtf=None, plan=plan)
        scores.append(sc.total)
    return scores


def main(symbols):
    bt = Backtester()
    market_df = bt.load_data("SPY", "2y", "1d")
    per_symbol = {}
    totals = defaultdict(int)
    valid_plan_bars = 0
    for sym in symbols:
        scores = sweep_symbol(sym, market_df=market_df)
        if scores is None:
            print(f"  {sym}: no data / too short")
            continue
        counts = {t: sum(1 for s in scores if s >= t) for t in THRESHOLDS}
        per_symbol[sym] = (len(scores), counts)
        valid_plan_bars += len(scores)
        for t in THRESHOLDS:
            totals[t] += counts[t]
        print(f"  {sym:8s} valid-plan bars={len(scores):4d} | "
              + " | ".join(f">={int(t)}: {counts[t]:3d}" for t in THRESHOLDS))

    print("\n" + "=" * 60)
    print(f"TOTAL valid-plan signal bars analyzed: {valid_plan_bars}")
    for t in THRESHOLDS:
        pct = 100 * totals[t] / valid_plan_bars if valid_plan_bars else 0
        print(f"  MIN_SCORE >= {int(t)}:  {totals[t]:4d} signals  ({pct:.1f}% of valid-plan bars)")
    # Marginal effect
    if totals[70.0] is not None:
        print("\nMarginal additional signals vs the 70 gate:")
        print(f"  60 vs 70:  +{totals[60.0] - totals[70.0]} signals  "
              f"({totals[60.0] / totals[70.0]:.2f}x)" if totals[70.0] else "")
        print(f"  65 vs 70:  +{totals[65.0] - totals[70.0]} signals  "
              f"({totals[65.0] / totals[70.0]:.2f}x)" if totals[70.0] else "")


if __name__ == "__main__":
    syms = sys.argv[1:] or [
        "AAPL", "TSLA", "NVDA", "SPY", "QQQ",
        "MSFT", "AMZN", "META", "GOOGL", "AMD",
    ]
    print(f"Threshold sweep over {len(syms)} symbols (2y daily, warmup={WARMUP}):\n")
    main(syms)
