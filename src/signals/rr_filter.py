"""Reward-to-risk filter with ATR stops and swing-structure targets.

The stop is placed ``ATR(14) * multiplier`` away from entry. The *target* is
derived from market structure — the most recent swing high (for longs) or swing
low (for shorts) within a lookback window. The trade is **rejected** unless the
structural target offers at least the configured reward:risk (default 5:1).

This makes the filter a genuine gate: a setup whose nearest structural target
isn't far enough away does not trade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.signals.strategy import Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradePlan:
    symbol: str
    side: str             # "long" or "short"
    entry: float
    stop: float
    target: float
    rr: float
    risk_per_share: float
    atr: float
    reason: str


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def swing_high(df: pd.DataFrame, lookback: int) -> float:
    """Highest high over the last ``lookback`` bars."""
    return float(df["high"].tail(lookback).max())


def swing_low(df: pd.DataFrame, lookback: int) -> float:
    """Lowest low over the last ``lookback`` bars."""
    return float(df["low"].tail(lookback).min())


def _round(value: float) -> float:
    # Alpaca rejects equity prices with > 2 decimal places (>= $1).
    return round(value, 2)


class RRFilter:
    def __init__(
        self,
        rr_ratio: float = 5.0,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        swing_lookback: int = 20,
    ) -> None:
        if rr_ratio <= 0 or atr_multiplier <= 0:
            raise ValueError("rr_ratio and atr_multiplier must be positive")
        self.rr_ratio = rr_ratio
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.swing_lookback = swing_lookback

    def evaluate(self, signal: Signal, df: pd.DataFrame) -> Optional[TradePlan]:
        """Build a TradePlan if the structural target clears ``rr_ratio``, else None."""
        try:
            if df is None or len(df) < self.atr_period + 1:
                return None

            atr_val = float(atr(df, self.atr_period).iloc[-1])
            if not np.isfinite(atr_val) or atr_val <= 0:
                return None

            entry = signal.entry_price
            risk_distance = self.atr_multiplier * atr_val

            if signal.side == "long":
                stop = entry - risk_distance
                target = swing_high(df, self.swing_lookback)
                if target <= entry:        # no structure above price -> no target
                    return None
            elif signal.side == "short":
                stop = entry + risk_distance
                target = swing_low(df, self.swing_lookback)
                if target >= entry:        # no structure below price -> no target
                    return None
            else:
                return None

            entry, stop, target = _round(entry), _round(stop), _round(target)
            risk_per_share = abs(entry - stop)
            reward_per_share = abs(target - entry)
            if risk_per_share <= 0 or stop <= 0 or target <= 0:
                return None

            rr = reward_per_share / risk_per_share
            if rr + 1e-9 < self.rr_ratio:
                logger.info(
                    "%s: rejected, RR %.2f < %.2f (entry=%.2f stop=%.2f target=%.2f)",
                    signal.symbol, rr, self.rr_ratio, entry, stop, target,
                )
                return None

            return TradePlan(
                symbol=signal.symbol,
                side=signal.side,
                entry=entry,
                stop=stop,
                target=target,
                rr=round(rr, 2),
                risk_per_share=round(risk_per_share, 4),
                atr=round(atr_val, 4),
                reason=signal.reason,
            )
        except Exception:
            logger.exception("RRFilter.evaluate failed for %s", getattr(signal, "symbol", "?"))
            return None
