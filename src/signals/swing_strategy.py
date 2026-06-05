"""Swing trading strategy — daily/weekly setups for 3–15 day holds.

Scans a symbol's daily + weekly candles for three families of setups and returns
the best one as a scored :class:`SwingSetup`:

* **Trend continuation** — pullback to a rising 21-EMA, bull flag, intact
  higher-high/higher-low structure.
* **Reversal** — weekly-oversold with daily bullish RSI divergence, double
  bottom, morning star / failed breakdown.
* **Breakout** — 52-week-high break on 2x volume, tight-range breakout,
  post-earnings-gap continuation.

Filters: trend/breakout setups require an up weekly trend and relative strength
vs SPY; reversal setups (counter-trend by nature) are exempt from those two but
still require some trend (ADX > 20) and clear of earnings. Volume is scored, not
gated, so low-volume pullbacks aren't all excluded.

Pattern detectors are deliberately simple, transparent heuristics — not
validated chart-pattern recognition. Treat the score as a ranking aid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import ta

logger = logging.getLogger(__name__)

EARNINGS_BLACKOUT_DAYS = 5
MIN_QUALIFY_SCORE = 60.0


@dataclass
class SwingSetup:
    symbol: str
    setup_type: str
    category: str            # trend / reversal / breakout
    side: str                # long (this system is long-biased)
    entry: float
    stop: float
    target: float
    rr: float
    score: float
    weekly_trend: str
    days_to_earnings: Optional[int]
    reason: str

    def as_row(self) -> dict:
        return {
            "symbol": self.symbol, "setup": self.setup_type, "category": self.category,
            "side": self.side, "entry": round(self.entry, 2), "stop": round(self.stop, 2),
            "target": round(self.target, 2), "rr": round(self.rr, 2),
            "score": round(self.score, 1), "weekly_trend": self.weekly_trend,
            "days_to_earnings": self.days_to_earnings, "reason": self.reason,
        }


def _round(x: float) -> float:
    return round(float(x), 2)


class SwingStrategy:
    def __init__(self, atr_mult: float = 1.5, rr_cap: float = 4.0,
                 swing_lookback: int = 20):
        self.atr_mult = atr_mult
        self.rr_cap = rr_cap
        self.swing_lookback = swing_lookback

    # ------------------------------------------------------------------ #
    def scan(self, symbol: str, daily: pd.DataFrame, weekly: pd.DataFrame,
             spy: Optional[pd.DataFrame] = None,
             days_to_earnings: Optional[int] = None) -> Optional[SwingSetup]:
        try:
            if daily is None or len(daily) < 60 or weekly is None or len(weekly) < 30:
                return None

            # Earnings blackout (only enforced when we actually have a date).
            if days_to_earnings is not None and 0 <= days_to_earnings <= EARNINGS_BLACKOUT_DAYS:
                return None

            ind = self._indicators(daily, weekly)
            if ind["adx"] is None or ind["adx"] < 20:
                return None  # universal: need some trend present

            rs_ok = self._relative_strength_ok(daily, spy)
            weekly_up = ind["weekly_trend"] == "up"

            # Collect matching setups (type, category, quality 0-30).
            matches: list[tuple[str, str, float]] = []
            if weekly_up and rs_ok:
                matches += self._trend_setups(daily, ind)
                matches += self._breakout_setups(daily, ind)
            if ind["weekly_rsi"] is not None and ind["weekly_rsi"] < 40:
                matches += self._reversal_setups(daily, weekly, ind)
            if not matches:
                return None

            setup_type, category, quality = max(matches, key=lambda m: m[2])
            plan = self._levels(daily, ind)
            if plan is None:
                return None
            entry, stop, target, rr = plan

            score = self._score(ind, rs_ok, weekly_up, quality, rr)
            if score < MIN_QUALIFY_SCORE:
                return None

            return SwingSetup(
                symbol=symbol, setup_type=setup_type, category=category, side="long",
                entry=entry, stop=stop, target=target, rr=rr, score=score,
                weekly_trend=ind["weekly_trend"], days_to_earnings=days_to_earnings,
                reason=f"{setup_type}; ADX {ind['adx']:.0f}; "
                       f"{'RS+' if rs_ok else 'RS-'}; wRSI {ind['weekly_rsi']:.0f}",
            )
        except Exception:
            logger.exception("SwingStrategy.scan failed for %s", symbol)
            return None

    # ------------------------------------------------------------------ #
    # Indicators
    # ------------------------------------------------------------------ #
    def _indicators(self, daily: pd.DataFrame, weekly: pd.DataFrame) -> dict:
        c, h, l, v = daily["close"], daily["high"], daily["low"], daily["volume"]
        wema = weekly["close"].ewm(span=10, adjust=False).mean()
        weekly_trend = "up" if (weekly["close"].iloc[-1] > wema.iloc[-1]
                                and wema.iloc[-1] > wema.iloc[-3]) else "down"
        return {
            "price": float(c.iloc[-1]),
            "ema21": c.ewm(span=21, adjust=False).mean(),
            "ema50": float(c.ewm(span=50, adjust=False).mean().iloc[-1]),
            "rsi": ta.momentum.RSIIndicator(c, 14).rsi(),
            "adx": float(ta.trend.ADXIndicator(h, l, c, 14).adx().iloc[-1]),
            "atr": float(ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range().iloc[-1]),
            "vol": float(v.iloc[-1]),
            "vol20": float(v.tail(20).mean()),
            "weekly_trend": weekly_trend,
            "weekly_rsi": float(ta.momentum.RSIIndicator(weekly["close"], 14).rsi().iloc[-1]),
        }

    @staticmethod
    def _relative_strength_ok(daily: pd.DataFrame, spy: Optional[pd.DataFrame]) -> bool:
        if spy is None or len(spy) < 21 or len(daily) < 21:
            return True  # can't compute -> don't gate
        stock_ret = daily["close"].iloc[-1] / daily["close"].iloc[-21] - 1
        spy_ret = spy["close"].iloc[-1] / spy["close"].iloc[-21] - 1
        return stock_ret > spy_ret

    # ------------------------------------------------------------------ #
    # Setup detectors (each: list of (type, category, quality 0-30))
    # ------------------------------------------------------------------ #
    def _trend_setups(self, daily, ind) -> list[tuple[str, str, float]]:
        out = []
        c = daily["close"]
        ema21 = ind["ema21"]
        price = ind["price"]
        rising_21 = ema21.iloc[-1] > ema21.iloc[-5]
        # Pullback to rising 21-EMA: recent low touched near the EMA, now back above.
        if price > ind["ema50"] and rising_21:
            recent_low = daily["low"].tail(5).min()
            near = abs(recent_low - ema21.iloc[-1]) / price < 0.03
            bounced = price > c.iloc[-2] and price > ema21.iloc[-1]
            if near and bounced:
                out.append(("pullback_to_21ema", "trend", 28.0))
        # Bull flag: strong run then tight consolidation near highs.
        run = c.iloc[-1] / c.iloc[-20] - 1
        rng = (daily["high"].tail(6).max() - daily["low"].tail(6).min()) / price
        if run > 0.15 and rng < 0.06:
            out.append(("bull_flag", "trend", 24.0))
        # Higher-high / higher-low structure.
        if self._hh_hl(daily):
            out.append(("higher_high_higher_low", "trend", 18.0))
        return out

    def _breakout_setups(self, daily, ind) -> list[tuple[str, str, float]]:
        out = []
        c, h = daily["close"], daily["high"]
        price = ind["price"]
        vol_surge = ind["vol"] > 2 * ind["vol20"]
        # 52-week high breakout on 2x volume.
        hi_252 = h.tail(252).max()
        if price >= hi_252 * 0.995 and vol_surge:
            out.append(("52w_high_breakout", "breakout", 30.0))
        # Tight-range (10+ bars) breakout: prior range compressed, today expands up.
        prior = daily.iloc[-12:-1]
        prng = (prior["high"].max() - prior["low"].min()) / price
        if prng < 0.06 and price > prior["high"].max():
            out.append(("range_breakout", "breakout", 26.0))
        # Earnings-gap continuation: a big gap up in the last 3 bars, holding.
        gaps = daily["open"] / daily["close"].shift(1) - 1
        if gaps.tail(3).max() > 0.05 and price > c.iloc[-2]:
            out.append(("earnings_gap_continuation", "breakout", 22.0))
        return out

    def _reversal_setups(self, daily, weekly, ind) -> list[tuple[str, str, float]]:
        out = []
        c, l = daily["close"], daily["low"]
        # Weekly oversold + daily bullish RSI divergence.
        rsi = ind["rsi"]
        if ind["weekly_rsi"] < 30 and self._bullish_divergence(c, rsi):
            out.append(("weekly_oversold_divergence", "reversal", 28.0))
        # Double bottom: two similar lows with a peak between.
        if self._double_bottom(daily):
            out.append(("double_bottom", "reversal", 24.0))
        # Morning star (3-candle bullish reversal).
        if self._morning_star(daily):
            out.append(("morning_star", "reversal", 20.0))
        # Failed breakdown: broke below recent support then reclaimed it.
        support = l.iloc[-15:-2].min()
        if l.tail(4).min() < support and c.iloc[-1] > support:
            out.append(("failed_breakdown", "reversal", 22.0))
        return out

    # ------------------------------------------------------------------ #
    # Pattern primitives (heuristic)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hh_hl(daily, lookback: int = 40) -> bool:
        w = daily.tail(lookback)
        if len(w) < 20:
            return False
        mid = len(w) // 2
        return (w["high"].iloc[mid:].max() > w["high"].iloc[:mid].max()
                and w["low"].iloc[mid:].min() > w["low"].iloc[:mid].min())

    @staticmethod
    def _bullish_divergence(close, rsi, lookback: int = 12) -> bool:
        if len(close) < lookback:
            return False
        c0, cprev = close.iloc[-1], close.iloc[-lookback]
        r0, rprev = rsi.iloc[-1], rsi.iloc[-lookback]
        return c0 < cprev and r0 > rprev   # lower price, higher RSI

    @staticmethod
    def _double_bottom(daily, lookback: int = 40) -> bool:
        w = daily.tail(lookback)
        if len(w) < 20:
            return False
        lows = w["low"].values
        i1 = int(np.argmin(lows[: len(lows) // 2]))
        i2 = len(lows) // 2 + int(np.argmin(lows[len(lows) // 2:]))
        if lows[i1] <= 0:
            return False
        similar = abs(lows[i1] - lows[i2]) / lows[i1] < 0.04
        peak_between = w["high"].iloc[i1:i2].max() > lows[i1] * 1.04
        reclaiming = w["close"].iloc[-1] > lows[i2] * 1.02
        return similar and peak_between and reclaiming

    @staticmethod
    def _morning_star(daily) -> bool:
        if len(daily) < 3:
            return False
        o, c = daily["open"], daily["close"]
        o2, c2 = o.iloc[-3], c.iloc[-3]
        o1, c1 = o.iloc[-2], c.iloc[-2]
        o0, c0 = o.iloc[-1], c.iloc[-1]
        rng2 = abs(c2 - o2) or 1e-9
        return (c2 < o2                                   # down candle
                and abs(c1 - o1) < 0.4 * rng2             # small middle
                and c0 > o0 and c0 > (o2 + c2) / 2)       # strong up close

    # ------------------------------------------------------------------ #
    # Levels + score
    # ------------------------------------------------------------------ #
    def _levels(self, daily, ind):
        entry = ind["price"]
        atr = ind["atr"]
        if not np.isfinite(atr) or atr <= 0:
            return None
        swing_low = float(daily["low"].tail(self.swing_lookback).min())
        stop = min(swing_low, entry - self.atr_mult * atr)
        risk = entry - stop
        if risk <= 0:
            return None
        # Target: next resistance (recent swing high) or rr_cap×R, whichever closer.
        resistance = float(daily["high"].tail(self.swing_lookback * 2).max())
        capped = entry + self.rr_cap * risk
        target = min(resistance, capped) if resistance > entry else capped
        if target <= entry:
            return None
        return _round(entry), _round(stop), _round(target), round((target - entry) / risk, 2)

    def _score(self, ind, rs_ok, weekly_up, setup_quality, rr) -> float:
        score = setup_quality                                  # up to 30
        score += 20 if weekly_up else 8                        # weekly alignment
        score += min(15, max(0, (ind["adx"] - 20) / 2))        # trend strength
        score += 10 if ind["vol"] > ind["vol20"] else 4        # volume confirmation
        score += 15 if rs_ok else 5                            # relative strength
        score += min(10, rr / self.rr_cap * 10)                # reward:risk
        return round(min(100.0, score), 1)
